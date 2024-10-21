import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_g_inequality_l376_37664

noncomputable section

def f (p : ℝ) (x : ℝ) : ℝ := p * x - p / x - 2 * Real.log x

def g (x : ℝ) : ℝ := 2 * Real.exp 1 / x

theorem f_increasing_and_g_inequality (p : ℝ) :
  ((∀ x > 0, Monotone (f p)) ↔ p ≥ 1) ∧
  ((∃ x₀ ∈ Set.Icc 1 (Real.exp 1), f p x₀ > g x₀) ↔ p > (4 * Real.exp 1) / ((Real.exp 1)^2 - 1)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_g_inequality_l376_37664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l376_37608

-- Define the propositions
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2*x + a) / Real.log 0.5

def q (a : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (-(5-2*a))^x₁ > (-(5-2*a))^x₂

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (¬(p a) ∧ q a) ∧ ¬(p a ∧ q a) → 1 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l376_37608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_to_squares_ratio_l376_37621

-- Define the squares and points
noncomputable def square_ABCD : Set (ℝ × ℝ) := sorry
noncomputable def square_EFGH : Set (ℝ × ℝ) := sorry
noncomputable def point_A : ℝ × ℝ := sorry
noncomputable def point_B : ℝ × ℝ := sorry
noncomputable def point_C : ℝ × ℝ := sorry
noncomputable def point_D : ℝ × ℝ := sorry
noncomputable def point_E : ℝ × ℝ := sorry
noncomputable def point_F : ℝ × ℝ := sorry
noncomputable def point_G : ℝ × ℝ := sorry
noncomputable def point_H : ℝ × ℝ := sorry

-- Define the hexagon AFEDCB
noncomputable def hexagon_AFEDCB : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem hexagon_to_squares_ratio :
  -- Squares have equal areas
  MeasureTheory.volume square_ABCD = MeasureTheory.volume square_EFGH →
  -- B is midpoint of EF
  point_B = (point_E + point_F) / 2 →
  -- C is midpoint of FG
  point_C = (point_F + point_G) / 2 →
  -- The ratio of hexagon area to total squares area is 3/4
  MeasureTheory.volume hexagon_AFEDCB / (2 * MeasureTheory.volume square_ABCD) = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_to_squares_ratio_l376_37621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_oxen_count_l376_37666

/-- Represents the number of oxen and months for each person renting the pasture -/
structure GrazingData where
  oxen : ℕ
  months : ℕ

/-- Calculates the total oxen-months for a given GrazingData -/
def oxenMonths (data : GrazingData) : ℕ := data.oxen * data.months

/-- Proves that a put 10 oxen for grazing given the conditions of the problem -/
theorem a_oxen_count (a b c : GrazingData) 
  (h1 : b.oxen = 12 ∧ b.months = 5)
  (h2 : c.oxen = 15 ∧ c.months = 3)
  (h3 : a.months = 7)
  (h4 : oxenMonths a + oxenMonths b + oxenMonths c = 175)
  (h5 : (oxenMonths c : ℚ) / (oxenMonths a + oxenMonths b + oxenMonths c) = 36 / 140) :
  a.oxen = 10 := by
  sorry

#check a_oxen_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_oxen_count_l376_37666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_width_at_critical_depth_navigation_affected_depth_l376_37642

/-- Represents a parabolic bridge -/
structure ParabolicBridge where
  base_width : ℝ
  height : ℝ

/-- Calculates the width of water under the bridge given the water level rise -/
noncomputable def water_width (bridge : ParabolicBridge) (h : ℝ) : ℝ :=
  10 * Real.sqrt (4 - h)

/-- Theorem: When water depth reaches 2.76m, the width under the bridge is exactly 18m -/
theorem water_width_at_critical_depth 
  (bridge : ParabolicBridge) 
  (h_base_width : bridge.base_width = 20) 
  (h_height : bridge.height = 4) :
  water_width bridge 0.76 = 18 := by
  sorry

/-- Corollary: Navigation is affected when water depth exceeds 2.76m -/
theorem navigation_affected_depth 
  (bridge : ParabolicBridge) 
  (h_base_width : bridge.base_width = 20) 
  (h_height : bridge.height = 4) 
  (h_normal_depth : ℝ) 
  (h_normal_depth_value : h_normal_depth = 2) 
  (h : ℝ) :
  h_normal_depth + h > 2.76 → water_width bridge h < 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_width_at_critical_depth_navigation_affected_depth_l376_37642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semiminor_axis_length_l376_37625

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  focus : Point
  semimajor_endpoint : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the length of the semi-minor axis of an ellipse -/
noncomputable def semiminor_axis (e : Ellipse) : ℝ :=
  let c := distance e.center e.focus
  let a := distance e.center e.semimajor_endpoint
  Real.sqrt (a^2 - c^2)

/-- Theorem: The length of the semi-minor axis of the specified ellipse is 4 -/
theorem semiminor_axis_length :
  let e := Ellipse.mk
    (Point.mk 0 4)  -- center
    (Point.mk 0 1)  -- focus
    (Point.mk 0 9)  -- semimajor endpoint
  semiminor_axis e = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semiminor_axis_length_l376_37625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_root_of_quadratic_l376_37605

theorem second_root_of_quadratic 
  (a b c : ℝ) 
  (h : a * (b + c) * (-1)^2 - b * (c + a) * (-1) + c * (a + b) = 0) : 
  a * (b + c) * (-c * (a + b) / (a * (b + c)))^2 - 
  b * (c + a) * (-c * (a + b) / (a * (b + c))) + 
  c * (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_root_of_quadratic_l376_37605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_l376_37683

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (1/a + a)*x + (1 + a^2)

-- Theorem statement
theorem solution_sets (a : ℝ) (h : a > 0) :
  (a = 1/2 → Set.Icc (1/2 : ℝ) 2 = {x : ℝ | f a x ≤ 0}) ∧
  (a > 1 → Set.Icc (1/a : ℝ) a = {x : ℝ | f a x ≤ 0}) ∧
  (a = 1 → {1} = {x : ℝ | f a x ≤ 0}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_l376_37683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_300_deg_angle_relation_l376_37651

open Real

/-- Sine of 300 degrees is equal to negative square root of 3 divided by 2 -/
theorem sin_300_deg : sin (300 * π / 180) = -sqrt 3 / 2 := by
  sorry

/-- Sine of 60 degrees in a unit circle -/
axiom sin_60_deg : sin (60 * π / 180) = sqrt 3 / 2

/-- 300 degrees is in the fourth quadrant -/
axiom fourth_quadrant : 270 * π / 180 < 300 * π / 180 ∧ 300 * π / 180 < 360 * π / 180

/-- Sine is negative in the fourth quadrant -/
axiom sin_negative_fourth_quadrant (θ : ℝ) :
  270 * π / 180 < θ ∧ θ < 360 * π / 180 → sin θ < 0

/-- 300 degrees equals 360 degrees minus 60 degrees -/
theorem angle_relation : 300 * π / 180 = 360 * π / 180 - 60 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_300_deg_angle_relation_l376_37651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l376_37616

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Point on a hyperbola -/
structure PointOnHyperbola (E : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / E.a^2 - y^2 / E.b^2 = 1

/-- Foci of a hyperbola -/
noncomputable def foci (E : Hyperbola) : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt (E.a^2 + E.b^2)
  (c, 0, -c, 0)

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem about hyperbola properties -/
theorem hyperbola_properties (E : Hyperbola) (P : PointOnHyperbola E) :
  let (x₁, y₁, x₂, y₂) := foci E
  let d₁ := distance P.x P.y x₁ y₁
  let d₂ := distance P.x P.y x₂ y₂
  (P.x - x₁) * (P.x - x₂) + (P.y - y₁) * (P.y - y₂) = 0 ∧ -- PF₁ ⟂ PF₂
  d₁ = 2 * d₂ →                                        -- |PF₁| = 2|PF₂|
  (∃ e : ℝ, e^2 = 5 ∧ e = Real.sqrt (E.a^2 + E.b^2) / E.a) ∧
  (E.a^2 = 2 ∧ E.b^2 = 8) ∧
  (∃ m : ℝ, ∃ G : ℝ × ℝ, G = (2/m, 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l376_37616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sufficient_not_necessary_l376_37699

-- Define a sequence of real numbers
def Sequence := ℕ → ℝ

-- Define a geometric sequence
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Define an arithmetic sequence
def IsArithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define the logarithm sequence
noncomputable def LogSequence (a : Sequence) : Sequence :=
  fun n => Real.log (a n) + 1

-- Statement of the theorem
theorem geometric_sufficient_not_necessary :
  (∃ a : Sequence, IsGeometric a ∧ ¬IsArithmetic (LogSequence a)) ∧
  (∀ a : Sequence, IsGeometric a → IsArithmetic (LogSequence a)) ∧
  (∃ a : Sequence, IsArithmetic (LogSequence a) ∧ ¬IsGeometric a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sufficient_not_necessary_l376_37699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l376_37643

-- Define the line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ := 6 * Real.sin θ

-- Define point P
def point_P : ℝ × ℝ := (1, 2)

-- State the theorem
theorem min_distance_sum (α : ℝ) :
  let intersections := {A : ℝ × ℝ | ∃ t, line_l t α = A ∧ (A.1^2 + A.2^2 = 6 * A.2)}
  ∃ A B, A ∈ intersections ∧ B ∈ intersections ∧
    ∀ A' B', A' ∈ intersections → B' ∈ intersections → A' ≠ B' →
      Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
      Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) ≤
      Real.sqrt ((A'.1 - point_P.1)^2 + (A'.2 - point_P.2)^2) +
      Real.sqrt ((B'.1 - point_P.1)^2 + (B'.2 - point_P.2)^2) ∧
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l376_37643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_not_in_sequence_l376_37623

def sequenceStep : List Nat → List Nat
  | [] => [2, 3]
  | [x] => [2, 3, x]
  | (x::y::xs) => 
    let product := x * y
    if product < 10 then
      x :: y :: product :: xs
    else
      x :: y :: (product / 10) :: (product % 10) :: xs

def digit_never_appears (d : Nat) (s : List Nat) : Prop :=
  ∀ n, n ∈ s → n ≠ d

theorem digits_not_in_sequence :
  ∀ s : List Nat, s = sequenceStep s →
    digit_never_appears 5 s ∧
    digit_never_appears 7 s ∧
    digit_never_appears 9 s := by
  sorry

#check digits_not_in_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_not_in_sequence_l376_37623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l376_37641

/-- Given a triangle with side lengths a, b, c and area S, 
    prove that a² + b² + c² ≥ 4√3 S, 
    with equality iff the triangle is equilateral -/
theorem triangle_inequality (a b c S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0) 
  (h_triangle : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S ∧ 
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * S ↔ a = b ∧ b = c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l376_37641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_in_right_triangle_l376_37639

/-- A triangle representation. -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Angle of a triangle at a specific vertex. -/
def Triangle.angle (t : Triangle) (i : Fin 3) : ℝ := sorry

/-- A triangle is acute-angled if all of its angles are less than 90 degrees. -/
def isAcuteAngled (t : Triangle) : Prop :=
  ∀ i : Fin 3, t.angle i < 90

/-- The area of a triangle. -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

/-- A triangle t1 is contained in another triangle t2. -/
def isContainedIn (t1 t2 : Triangle) : Prop := sorry

/-- A triangle is right-angled if one of its angles is 90 degrees. -/
def isRightAngled (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angle i = 90

/-- Main theorem: Any acute-angled triangle with area 1 can be placed inside a right triangle with area not exceeding √3. -/
theorem acute_triangle_in_right_triangle :
  ∀ t : Triangle,
    isAcuteAngled t →
    triangleArea t = 1 →
    ∃ rt : Triangle,
      isRightAngled rt ∧
      triangleArea rt ≤ Real.sqrt 3 ∧
      isContainedIn t rt :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_in_right_triangle_l376_37639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_g_zero_in_interval_l376_37634

noncomputable section

variable (m : ℝ)

def f (m : ℝ) (x : ℝ) : ℝ := (2 * m / 3) * x^3 + x^2 - 3 * x - m * x + 2

def g (m : ℝ) (x : ℝ) : ℝ := deriv (f m) x

theorem f_monotonicity (m : ℝ) :
  m = 1 →
  (MonotoneOn (f m) (Set.Iic (-2))) ∧
  (AntitoneOn (f m) (Set.Icc (-2) 1)) ∧
  (MonotoneOn (f m) (Set.Ioi 1)) :=
sorry

theorem g_zero_in_interval (m : ℝ) :
  (∃ x ∈ Set.Icc (-1) 1, g m x = 0) ↔
  m ∈ Set.Iic ((-(3 : ℝ) - Real.sqrt 7) / 2) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_g_zero_in_interval_l376_37634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derived_point_x2_eq_3x_min_distance_derived_point_l376_37676

/-- Definition of a derived point for a quadratic equation -/
noncomputable def derived_point (a b c : ℝ) : ℝ × ℝ :=
  let x₁ := min ((-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)) ((-b - Real.sqrt (b^2 - 4*a*c)) / (2*a))
  let x₂ := max ((-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)) ((-b - Real.sqrt (b^2 - 4*a*c)) / (2*a))
  (x₁, x₂)

/-- The derived point of x^2 = 3x is (0, 3) -/
theorem derived_point_x2_eq_3x : derived_point 1 (-3) 0 = (0, 3) := by sorry

/-- For x^2-(2m-1)x+m^2-m=0, the value of m that minimizes 
    the distance of the derived point from the origin is 1/2 -/
theorem min_distance_derived_point :
  let f (m : ℝ) := (derived_point 1 (1-2*m) (m^2-m)).1^2 + (derived_point 1 (1-2*m) (m^2-m)).2^2
  ∃ (m₀ : ℝ), ∀ (m : ℝ), f m ≥ f m₀ ∧ m₀ = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derived_point_x2_eq_3x_min_distance_derived_point_l376_37676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safari_snake_lion_ratio_l376_37631

/-- Represents a national park with lions, snakes, and giraffes -/
structure NationalPark where
  lions : ℕ
  snakes : ℕ
  giraffes : ℕ

/-- Safari National Park satisfies the given conditions -/
def safari (S : ℕ) : NationalPark :=
  { lions := 100,
    snakes := S,
    giraffes := if S ≥ 10 then S - 10 else 0 }

/-- The ratio of snakes to lions in Safari National Park -/
def snake_lion_ratio (park : NationalPark) : ℚ :=
  park.snakes / park.lions

/-- Theorem: The ratio of snakes to lions in Safari National Park is S:100 -/
theorem safari_snake_lion_ratio (S : ℕ) :
  snake_lion_ratio (safari S) = S / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_safari_snake_lion_ratio_l376_37631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_point_l376_37632

-- Define the power function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x ^ a

-- State the theorem
theorem power_function_point (a m : ℝ) :
  f a 2 = Real.sqrt 2 / 2 →
  f a m = 2 →
  m = 1 / 4 := by
  -- Introduce the hypotheses
  intro h1 h2
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_point_l376_37632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_decreasing_sequences_remainder_modulo_1000_l376_37607

/-- The number of non-decreasing sequences of positive integers satisfying the given conditions -/
def num_sequences : ℕ := Nat.choose 2013 10

/-- The upper bound for the sequence elements -/
def upper_bound : ℕ := 2013

/-- The length of the sequences -/
def sequence_length : ℕ := 10

theorem count_non_decreasing_sequences :
  (∃ (m : ℕ), num_sequences = Nat.choose m sequence_length) ∧
  (∀ (a : ℕ → ℕ), (∀ i ∈ Finset.range sequence_length, a i ≤ a (i + 1)) →
                   (∀ i ∈ Finset.range sequence_length, a i ≤ upper_bound) →
                   (∀ i ∈ Finset.range sequence_length, Even (a i - (i + 1)) ∨ Odd (a i - (i + 1)))) →
  num_sequences = Nat.choose upper_bound sequence_length :=
by sorry

theorem remainder_modulo_1000 :
  upper_bound % 1000 = 13 :=
by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_decreasing_sequences_remainder_modulo_1000_l376_37607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_f_points_and_sum_l376_37694

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the given conditions
axiom f_points : f 2 = 6 ∧ f 3 = 4 ∧ f 4 = 2
axiom g_points : g 2 = 4 ∧ g 3 = 2 ∧ g 5 = 6

-- Define the composition g ∘ f
noncomputable def g_f : ℝ → ℝ := g ∘ f

-- State the theorem
theorem g_f_points_and_sum : 
  (g_f 2 = 6 ∧ g_f 3 = 4) ∧ 
  (2 * 6 + 3 * 4 = 24) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_f_points_and_sum_l376_37694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_seven_l376_37627

theorem subsets_containing_seven :
  (Finset.filter (fun s => 7 ∈ s) (Finset.powerset {1, 2, 3, 4, 5, 6, 7})).card = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_seven_l376_37627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_questions_sufficient_and_necessary_l376_37680

/-- Represents the two villages -/
inductive Village : Type
  | A : Village  -- Village where people always tell the truth
  | B : Village  -- Village where people always lie

/-- Represents a person's response to a question -/
inductive Response : Type
  | Yes : Response
  | No : Response

/-- Represents the state of information after asking questions -/
structure Information : Type :=
  (tourist_village : Village)
  (person_village : Village)

/-- Function to determine if a person tells the truth based on their village -/
def tells_truth (v : Village) : Bool :=
  match v with
  | Village.A => true
  | Village.B => false

/-- Function to model a person's response to a question -/
def answer_question (person_village : Village) (true_answer : Bool) : Response :=
  if tells_truth person_village = true_answer
  then Response.Yes
  else Response.No

/-- Theorem stating that two questions are necessary and sufficient -/
theorem two_questions_sufficient_and_necessary :
  ∀ (tourist_village person_village : Village),
  ∃ (q1 q2 : Village → Response),
    (∀ (v1 v2 : Village),
      q1 v1 = q1 v2 ∧ q2 v1 = q2 v2 →
      v1 = v2) ∧
    (∀ (v : Village),
      ∃! (i : Information),
        q1 v = answer_question person_village true ∧
        q2 v = answer_question person_village (tells_truth v)) ∧
    ¬∃ (q : Village → Response),
      ∀ (v : Village),
        ∃! (i : Information),
          q v = answer_question person_village (tells_truth v) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_questions_sufficient_and_necessary_l376_37680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_blend_theorem_l376_37675

noncomputable def coffee_blend_price (price1 : ℝ) (price2 : ℝ) (total_weight : ℝ) (weight1 : ℝ) : ℝ :=
  let weight2 := total_weight - weight1
  let total_cost := price1 * weight1 + price2 * weight2
  total_cost / total_weight

theorem coffee_blend_theorem :
  coffee_blend_price 9 8 20 8 = 8.4 := by
  -- Unfold the definition of coffee_blend_price
  unfold coffee_blend_price
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_blend_theorem_l376_37675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_I_nat_is_empty_l376_37644

def I : Set ℤ := {x | x ≥ -1}

theorem complement_I_nat_is_empty :
  (Set.univ : Set ℕ) \ (I.image Int.toNat) = ∅ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_I_nat_is_empty_l376_37644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_closed_l376_37612

def is_closed_under_mult (S : Set ℤ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → (a * b) ∈ S

theorem at_least_one_closed (T V : Set ℤ) 
  (hT : T.Nonempty) (hV : V.Nonempty) 
  (hDisjoint : Disjoint T V) (hUnion : T ∪ V = Set.univ)
  (hTClosed : ∀ a b c, a ∈ T → b ∈ T → c ∈ T → (a * b * c) ∈ T)
  (hVClosed : ∀ x y z, x ∈ V → y ∈ V → z ∈ V → (x * y * z) ∈ V) :
  is_closed_under_mult T ∨ is_closed_under_mult V :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_closed_l376_37612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_angle_has_symmetry_axis_l376_37646

/-- Represents an n-sided angle in 3D space -/
structure NSidedAngle (n : ℕ) where
  -- Add necessary fields here
  mk :: -- This allows the structure to be defined without specifying fields

/-- Predicate to check if an n-sided angle is regular -/
def is_regular {n : ℕ} (angle : NSidedAngle n) : Prop :=
  -- All plane angles and dihedral angles are congruent
  True -- Placeholder, replace with actual condition when implemented

/-- Represents an axis of rotation in 3D space -/
structure RotationAxis where
  -- Add necessary fields here
  mk :: -- This allows the structure to be defined without specifying fields

/-- Predicate to check if rotating around an axis by 2π/n brings the angle into self-coincidence -/
def rotates_to_self_coincidence {n : ℕ} (angle : NSidedAngle n) (axis : RotationAxis) : Prop :=
  -- Rotation by 2π/n around the axis brings the angle into self-coincidence
  True -- Placeholder, replace with actual condition when implemented

/-- Theorem: If an n-sided angle is regular, then there exists an axis of symmetry -/
theorem regular_angle_has_symmetry_axis {n : ℕ} (angle : NSidedAngle n) 
  (h_regular : is_regular angle) : 
  ∃ (axis : RotationAxis), rotates_to_self_coincidence angle axis := by
  sorry -- The proof is omitted for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_angle_has_symmetry_axis_l376_37646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_l376_37692

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem monotonic_condition (a : ℝ) :
  (∀ x y, x ∈ Set.Icc 2 3 → y ∈ Set.Icc 2 3 → x < y → (f a x < f a y ∨ f a x > f a y)) →
  (a ≤ 2 ∨ a ≥ 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_l376_37692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jam_sharing_l376_37654

theorem jam_sharing (initial_jam : ℚ) : initial_jam / 15 = 
  let first_share := initial_jam / 3
  let after_lunch := first_share * 3 / 4
  let second_share := after_lunch / 2
  let after_dinner := second_share * 2 / 3
  let anna_snack := after_dinner / 5
  after_dinner - anna_snack
:= by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jam_sharing_l376_37654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l376_37698

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the theorem
theorem triangle_properties (abc : Triangle) :
  -- Part 1
  (abc.a * (Real.cos abc.A) = abc.c * (Real.cos abc.C) ∧ abc.a = Real.sqrt 3 * abc.c) →
  abc.A = π / 3 ∧
  -- Part 2
  (abc.a * (Real.cos abc.C) + abc.c * (Real.cos abc.A) = 3 * abc.b * (Real.sin abc.B) ∧ Real.cos abc.A = 3 / 5) →
  Real.cos abc.C = (4 - 6 * Real.sqrt 2) / 15 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l376_37698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_calculation_l376_37677

-- Define the given constants
noncomputable def tree_shadow : ℝ := 10
noncomputable def jane_shadow : ℝ := 0.5
noncomputable def slope_angle : ℝ := 15 * Real.pi / 180  -- Convert to radians
noncomputable def jane_height : ℝ := 1.5

-- Define the function to calculate the tree height
noncomputable def tree_height : ℝ :=
  (tree_shadow * jane_height / jane_shadow) * Real.sin slope_angle

-- Theorem statement
theorem tree_height_calculation :
  ∃ ε > 0, |tree_height - 7.764| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_calculation_l376_37677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l376_37670

theorem problem_1 : (3 : ℝ)⁻¹ - 1 / (1/3) * 3 - (1/12 - 1/2) * 4 = -7 := by 
  -- Proof steps would go here
  sorry

theorem problem_2 (a : ℝ) (h : a ≠ 1) : 
  (1 - 1/(a+1)) / ((a^2 - 1) / (a^2 + 2*a + 1)) = a / (a-1) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l376_37670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l376_37601

/-- Triangle ABC with given side lengths and angle -/
structure Triangle where
  a : ℝ
  b : ℝ
  B : ℝ

/-- The properties we want to prove about the triangle -/
def TriangleProperties (t : Triangle) : Prop :=
  t.a = Real.sqrt 2 ∧
  t.b = Real.sqrt 3 ∧
  t.B = 60 * Real.pi / 180 ∧
  ∃ (A C : ℝ) (c : ℝ),
    A = 45 * Real.pi / 180 ∧
    C = 75 * Real.pi / 180 ∧
    c = (Real.sqrt 2 + Real.sqrt 6) / 2

/-- Theorem stating that a triangle with the given properties exists -/
theorem triangle_theorem (t : Triangle) :
  TriangleProperties t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l376_37601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_decreasing_function_property_l376_37600

-- Define the domain
def D : Set ℝ := Set.Icc (-2) 2

-- Define properties of the function
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, x ∈ D → f (-x) = -f x

def isDecreasing (f : ℝ → ℝ) : Prop := ∀ x y, x ∈ D → y ∈ D → x < y → f x > f y

def isMonotonic (f : ℝ → ℝ) : Prop := 
  (∀ x y, x ∈ D → y ∈ D → x < y → f x < f y) ∨ 
  (∀ x y, x ∈ D → y ∈ D → x < y → f x > f y)

-- Theorem statements
theorem odd_function_property (f : ℝ → ℝ) :
  (∀ x, x ∈ D → f (-x) + f x = 0) → isOdd f := by sorry

theorem decreasing_function_property (f : ℝ → ℝ) :
  isMonotonic f → f 0 > f 1 → isDecreasing f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_decreasing_function_property_l376_37600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l376_37678

theorem solve_exponential_equation (y : ℝ) : 5 * (2 : ℝ) ^ y = 320 → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l376_37678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trig_sum_is_three_l376_37667

theorem max_trig_sum_is_three :
  ∃ (M : ℝ), M = 3 ∧
  (∀ θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ,
    Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + Real.cos θ₃ * Real.sin θ₄ + 
    Real.cos θ₄ * Real.sin θ₅ + Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁ ≤ M) ∧
  (∃ θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ,
    Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + Real.cos θ₃ * Real.sin θ₄ + 
    Real.cos θ₄ * Real.sin θ₅ + Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁ = M) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trig_sum_is_three_l376_37667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l376_37691

noncomputable def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + m = 0

def point_A : ℝ × ℝ := (2, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def line_equation (k b x y : ℝ) : Prop :=
  y = k * x + b

theorem circle_line_intersection
  (m : ℝ)
  (h1 : circle_equation point_A.1 point_A.2 m)
  (P Q : ℝ × ℝ)
  (h2 : distance P Q = 4 * Real.sqrt 6)
  (k b : ℝ)
  (h3 : ∀ x y, line_equation k b x y → circle_equation x y m) :
  (k = -4/3 ∧ b = 7/3) ∨ (k = -4/3 ∧ b = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l376_37691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_can_problem_l376_37637

/-- Proves that given a regular price of $0.60 per can, a 20% discount when purchased in 24-can cases, and a total price of $34.56, the number of cans purchased is 72. -/
theorem soda_can_problem (regular_price : ℚ) (discount_percent : ℚ) (total_price : ℚ) :
  regular_price = 60/100 →
  discount_percent = 20/100 →
  total_price = 3456/100 →
  (total_price / (regular_price * (1 - discount_percent))).floor = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_can_problem_l376_37637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_c_values_l376_37689

-- Define the function g
noncomputable def g (c : ℝ) (x : ℝ) : ℝ := c / (3 * x - 4)

-- Define the inverse function of g
noncomputable def g_inv (c : ℝ) (y : ℝ) : ℝ := 
  (c / y + 4) / 3

-- State the theorem
theorem product_of_c_values (c : ℝ) :
  g c 3 = g_inv c (c + 2) → 
  ∃ c₁ c₂ : ℝ, c₁ * c₂ = -8/3 ∧ 
  (g c₁ 3 = g_inv c₁ (c₁ + 2)) ∧ 
  (g c₂ 3 = g_inv c₂ (c₂ + 2)) ∧
  (∀ x : ℝ, g x 3 = g_inv x (x + 2) → x = c₁ ∨ x = c₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_c_values_l376_37689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l376_37613

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the sequence a_n
def a_n (n : ℕ) : ℝ := 2^(n - 1)

-- Define the sum S_n
noncomputable def S_n (n : ℕ) : ℝ := f 2 n - 1

theorem sequence_formula :
  ∀ (a : ℝ), a > 0 → a ≠ 1 →
  f a 1 = 2 →
  (∀ n : ℕ, S_n n = f a n - 1) →
  ∀ n : ℕ, a_n n = 2^(n - 1) :=
by
  sorry

#check sequence_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l376_37613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l376_37661

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) (b x) - vector_magnitude (vector_sum (a x) (b x))

theorem f_extrema :
  ∀ x ∈ Set.Icc (-π/3) (π/4),
    f x ≤ -1 ∧
    f x ≥ -3/2 ∧
    (∃ x₁ ∈ Set.Icc (-π/3) (π/4), f x₁ = -1) ∧
    (∃ x₂ ∈ Set.Icc (-π/3) (π/4), f x₂ = -3/2) := by
  sorry

#check f_extrema

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l376_37661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_extrema_of_f_l376_37659

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 1

-- Define the interval
def I : Set ℝ := Set.Icc (-2) 3

-- Theorem statement
theorem monotonicity_and_extrema_of_f :
  -- Monotonicity
  (∀ x y, x ∈ I → y ∈ I → x < y → y ≤ 0 → f x < f y) ∧
  (∀ x y, x ∈ I → y ∈ I → 0 ≤ x → x < y → y ≤ 2 → f x > f y) ∧
  (∀ x y, x ∈ I → y ∈ I → 2 ≤ x → x < y → f x < f y) ∧
  -- Extrema
  (∀ x, x ∈ I → f x ≤ f 0) ∧
  (∀ x, x ∈ I → f x ≤ f 3) ∧
  (∀ x, x ∈ I → f x ≥ f 2) ∧
  f (-2) = 9 ∧
  f 0 = 1 ∧
  f 2 = -7 ∧
  f 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_extrema_of_f_l376_37659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l376_37619

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (Real.exp x)

-- State the theorem
theorem derivative_of_f :
  Differentiable ℝ f ∧ ∀ x > 0, deriv f x = (1 - x * Real.log x) / (x * Real.exp x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l376_37619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_4_1_l376_37649

/-- The displacement function of the object --/
noncomputable def s (t : ℝ) : ℝ := 3 + t^2

/-- The average speed of the object in the time interval [2, 2.1] --/
noncomputable def average_speed : ℝ := (s 2.1 - s 2) / (2.1 - 2)

/-- Theorem stating that the average speed is equal to 4.1 --/
theorem average_speed_is_4_1 : average_speed = 4.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_4_1_l376_37649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_pi_over_four_l376_37687

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_is_pi_over_four (a b : ℝ × ℝ) 
  (h1 : Real.sqrt (a.1^2 + a.2^2) = 1)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 2)
  (h3 : a.1 * b.1 + a.2 * b.2 = 1) :
  angle_between_vectors a b = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_pi_over_four_l376_37687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shift_min_symmetric_shift_min_symmetric_shift_value_l376_37648

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.cos (x + m) - Real.sqrt 3 * Real.sin (x + m)

theorem symmetric_shift (m : ℝ) : 
  (∀ x, f m x = f m (-x)) ↔ ∃ k : ℤ, m = 2 * Real.pi * k - Real.pi / 3 ∨ m = (2 * k + 1) * Real.pi - Real.pi / 3 :=
sorry

theorem min_symmetric_shift : 
  ∃ m₀ : ℝ, m₀ > 0 ∧ (∀ x, f m₀ x = f m₀ (-x)) ∧ 
  ∀ m, m > 0 → (∀ x, f m x = f m (-x)) → m₀ ≤ m :=
sorry

theorem min_symmetric_shift_value : 
  ∃ m₀ : ℝ, m₀ = 2 * Real.pi / 3 ∧ 
  m₀ > 0 ∧ (∀ x, f m₀ x = f m₀ (-x)) ∧ 
  ∀ m, m > 0 → (∀ x, f m x = f m (-x)) → m₀ ≤ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shift_min_symmetric_shift_min_symmetric_shift_value_l376_37648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l376_37633

-- Define the function f
noncomputable def f (x : ℝ) := Real.log x + 3 * x - 11

-- State the theorem
theorem root_exists_in_interval :
  ∃ x ∈ Set.Ioo 3 4, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l376_37633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_range_l376_37685

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / (x^2)

theorem f_monotonicity_and_range :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ > f x₂) ∧
  (∀ m : ℝ, m > 1 → ∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f x < 2 + m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_range_l376_37685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_1647_to_hundredth_l376_37602

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The statement that 1.647 rounded to the nearest hundredth equals 1.65 -/
theorem round_1647_to_hundredth :
  roundToHundredth 1.647 = 1.65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_1647_to_hundredth_l376_37602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l376_37674

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_p_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

/-- Theorem statement -/
theorem parabola_properties (C : Parabola) (D : ℝ × ℝ) (M N : ℝ × ℝ) :
  D.1 = C.p ∧ D.2 = 0 →
  (∃ (k : ℝ), M.2 - C.focus.2 = k * (M.1 - C.focus.1) ∧
               N.2 - C.focus.2 = k * (N.1 - C.focus.1)) →
  M.2 - D.2 = 0 →
  (M.1 - C.focus.1)^2 + (M.2 - C.focus.2)^2 = 9 →
  (∀ x y, C.eq x y ↔ y^2 = 4*x) ∧
  (∃ (A B : ℝ × ℝ), 
    C.eq A.1 A.2 ∧ C.eq B.1 B.2 ∧
    (A.2 - D.2) / (A.1 - D.1) = (M.2 - D.2) / (M.1 - D.1) ∧
    (B.2 - D.2) / (B.1 - D.1) = (N.2 - D.2) / (N.1 - D.1) ∧
    (∀ α β : ℝ,
      (α = Real.arctan ((M.2 - N.2) / (M.1 - N.1)) ∧
       β = Real.arctan ((A.2 - B.2) / (A.1 - B.1))) →
      |α - β| ≤ Real.arctan ((1 : ℝ) / 2)) ∧
    B.1 - Real.sqrt 2 * B.2 - 4 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l376_37674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_not_negative_one_l376_37656

theorem intersection_not_negative_one (A B : Set ℤ) : 
  A = {x : ℤ | x < 3} → B ⊆ (Set.range (fun n : ℕ => (n : ℤ))) → A ∩ B ≠ {-1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_not_negative_one_l376_37656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_lock_number_l376_37665

def is_valid_lock_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 5 ∧
  digits.toFinset.card = 5 ∧
  n % 111 = 0 ∧
  (digits[0]! * 10000 + digits[1]! * 1000 + digits[3]! * 100 + digits[4]! * 10 + digits[2]!) > n ∧
  (digits[0]! * 10000 + digits[1]! * 1000 + digits[3]! * 100 + digits[4]! * 10 + digits[2]!) % 111 = 0 ∧
  (digits[0]! * 10000 + digits[1]! * 1000 + digits[4]! * 100 + digits[2]! * 10 + digits[3]!) > 
    (digits[0]! * 10000 + digits[1]! * 1000 + digits[3]! * 100 + digits[4]! * 10 + digits[2]!) ∧
  (digits[0]! * 10000 + digits[1]! * 1000 + digits[4]! * 100 + digits[2]! * 10 + digits[3]!) % 111 = 0

theorem unique_lock_number :
  ∃! n : ℕ, is_valid_lock_number n ∧ n = 74259 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_lock_number_l376_37665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_when_a_is_2_f_monotone_increasing_iff_l376_37603

-- Define the function f(x) = e^x - ax - 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

-- Part 1: Properties when a = 2
theorem f_properties_when_a_is_2 :
  let a := 2
  ∀ x y : ℝ,
    (x > Real.log 2 → y > Real.log 2 → x < y → f a x < f a y) ∧
    (x < Real.log 2 → y < Real.log 2 → x < y → f a x > f a y) ∧
    (∀ z : ℝ, f a (Real.log 2) ≤ f a z) ∧
    (f a (Real.log 2) = 1 - 2 * Real.log 2) := by
  sorry

-- Part 2: Condition for f to be monotonically increasing
theorem f_monotone_increasing_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_when_a_is_2_f_monotone_increasing_iff_l376_37603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_have_two_common_tangents_l376_37615

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + (y + 3)^2 = 9

-- Function to calculate the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem stating that the circles have exactly 2 common tangents
theorem circles_have_two_common_tangents :
  ∃ (n : ℕ), n = 2 ∧
  (∀ (x y : ℝ), circle1 x y → ∃ (cx cy r : ℝ), (x - cx)^2 + (y - cy)^2 = r^2 ∧ r = 2 ∧ cx = 2 ∧ cy = 0) ∧
  (∀ (x y : ℝ), circle2 x y → ∃ (cx cy r : ℝ), (x - cx)^2 + (y - cy)^2 = r^2 ∧ r = 3 ∧ cx = 3 ∧ cy = -3) ∧
  (distance 2 0 3 (-3) > 2) ∧
  (distance 2 0 3 (-3) < 2 + 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_have_two_common_tangents_l376_37615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_bridge_time_l376_37653

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  total_distance / train_speed_ms

/-- Theorem stating that the time for the train to pass the bridge is approximately 42.7 seconds -/
theorem train_passing_bridge_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |time_to_pass_bridge 300 115 35 - 42.7| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_bridge_time_l376_37653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l376_37669

/-- Proves that the speed of a stream is 4 km/h given the conditions of a swimmer --/
theorem stream_speed (swim_speed : ℝ) (upstream_time downstream_time : ℝ) (stream_speed : ℝ)
  (h1 : swim_speed = 12)
  (h2 : upstream_time = 2 * downstream_time)
  (h3 : (swim_speed + stream_speed) * downstream_time = (swim_speed - stream_speed) * upstream_time) :
  stream_speed = 4 := by
  sorry

#check stream_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l376_37669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geography_class_grade_distribution_l376_37693

/-- The number of students earning a B in a geography class -/
noncomputable def students_with_B (total_students : ℕ) (prob_A prob_B prob_C : ℝ) : ℝ :=
  (total_students : ℝ) / (prob_A / prob_B + 1 + prob_C / prob_B)

/-- Theorem: In a geography class of 50 students, where the probability of earning an A is 0.8 times
    the probability of earning a B, and the probability of earning a C is 1.2 times the probability
    of earning a B, assuming all grades are A, B, or C, the number of students earning a B is 50/3. -/
theorem geography_class_grade_distribution :
  let total_students : ℕ := 50
  let prob_A : ℝ := 0.8
  let prob_B : ℝ := 1
  let prob_C : ℝ := 1.2
  students_with_B total_students prob_A prob_B prob_C = 50 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geography_class_grade_distribution_l376_37693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l376_37645

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / (x + 1)

theorem f_monotone_increasing :
  (∀ x y, x < y ∧ x > -1 ∧ y < Real.pi → f x < f y) ∧
  (∀ x y, x < y ∧ x < -1 ∧ y < -1 → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l376_37645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_perpendicular_line_equation_l376_37620

-- Define a line with slope -1/3
def line_L (x y : ℝ) : Prop := y = -1/3 * x + c
  where c : ℝ := 0  -- y-intercept, set to 0 for simplicity

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_slope :
  ∀ m : ℝ, perpendicular (-1/3) m → m = 3 :=
by
  intro m h
  -- Proof steps would go here
  sorry

-- Define point E (example coordinates)
def E : ℝ × ℝ := (2, 1)

-- Theorem for the equation of the perpendicular line through E
theorem perpendicular_line_equation :
  ∃ (a b c : ℝ), a * E.1 + b * E.2 + c = 0 ∧ a = 3 ∧ b = -1 ∧ c = -5 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_perpendicular_line_equation_l376_37620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_existence_l376_37628

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

-- Helper functions
def on_circle (c : Circle) (p : ℝ × ℝ) : Prop := 
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

def parallel (l1 l2 : Line) : Prop := 
  l1.direction.1 * l2.direction.2 = l1.direction.2 * l2.direction.1

def point_on_line (l : Line) (p : ℝ × ℝ) : Prop := 
  let (x, y) := p
  let (px, py) := l.point
  let (dx, dy) := l.direction
  (x - px) * dy = (y - py) * dx

-- Define the problem statement
theorem inscribed_triangle_existence 
  (circle : Circle) 
  (line1 line2 : Line) 
  (P : ℝ × ℝ) :
  ∃ (t : Triangle), 
    (on_circle circle t.a ∧ on_circle circle t.b ∧ on_circle circle t.c) ∧ 
    (parallel (Line.mk t.a t.b) line1) ∧
    (parallel (Line.mk t.a t.c) line2) ∧
    (point_on_line (Line.mk t.b t.c) P) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_existence_l376_37628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_area_sum_l376_37686

-- Define the radius and central angle
def radius : ℝ := 6
def centralAngle : ℝ := 90

-- Define the region
structure Region where
  arcs : Fin 3 → Set (ℝ × ℝ)
  isCircularArc : ∀ i, ∀ p ∈ arcs i, ∃ center : ℝ × ℝ, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2
  intersectAtTangency : ∀ i j, i ≠ j → ∃ p, p ∈ arcs i ∧ p ∈ arcs j

-- Define the area expression
structure AreaExpression where
  a : ℤ
  b : ℕ
  c : ℤ
  isSimplestForm : ∀ d : ℕ, d > 1 → ¬(d * d ∣ b)

-- Define a function to calculate the area of the region
noncomputable def AreaOfRegion (R : Region) : ℝ := sorry

-- Theorem statement
theorem region_area_sum (R : Region) (A : AreaExpression) :
  AreaOfRegion R = A.a * Real.sqrt A.b + A.c * Real.pi →
  A.a + A.b + A.c = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_area_sum_l376_37686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equals_32_l376_37647

theorem power_product_equals_32 (m n : ℝ) (h : 3 * m + 2 * n = 5) : (8 : ℝ)^m * (4 : ℝ)^n = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equals_32_l376_37647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_PTV_is_60_degrees_l376_37626

-- Define the necessary types
variable (Point Line : Type)

-- Define the angle measure function
variable (angle : Point → Point → Point → ℝ)

-- Define the parallel and perpendicular relations
variable (parallel perpendicular : Line → Line → Prop)

-- Define the line constructor
variable (mk_line : Point → Point → Line)

-- State the theorem
theorem angle_PTV_is_60_degrees 
  (T P V : Point)
  (m n : Line)
  (h1 : parallel m n) 
  (h2 : perpendicular (mk_line T V) n) 
  (h3 : angle T P V = 150) : 
  angle P T V = 60 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_PTV_is_60_degrees_l376_37626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_sum_l376_37657

theorem lcm_ratio_sum (a b : ℕ) : 
  Nat.lcm a b = 54 → 
  3 * a = 2 * b → 
  a + b = 45 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_sum_l376_37657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_two_or_five_l376_37609

/-- A function that checks if a number contains either digit 2 or 5 -/
def containsTwoOrFive (n : ℕ) : Bool :=
  sorry

/-- The set of numbers between 200 and 499 (inclusive) that contain either 2 or 5 -/
def numbersWithTwoOrFive : Finset ℕ :=
  Finset.filter (fun n => containsTwoOrFive (n + 200)) (Finset.range 300)

theorem count_numbers_with_two_or_five : numbersWithTwoOrFive.card = 230 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_two_or_five_l376_37609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_theorem_l376_37684

-- Define the vectors a and b
def a (m : ℝ) : ℝ × ℝ := (m, -2)
def b : ℝ × ℝ := (3, 4)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2)

-- Define vector scalar multiplication
def vec_scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Theorem statement
theorem vector_magnitude_theorem (m : ℝ) :
  parallel (a m) b →
  magnitude (vec_sub (a m) (vec_scalar_mult (3/2) b)) = 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_theorem_l376_37684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l376_37636

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 1) / 1 = (z + 2) / (-2)

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  3 * x - 7 * y - 2 * z + 7 = 0

/-- The theorem stating that (2, 3, -4) is the unique intersection point -/
theorem intersection_point :
  ∃! (p : ℝ × ℝ × ℝ), 
    let (x, y, z) := p
    line_equation x y z ∧ plane_equation x y z ∧ p = (2, 3, -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l376_37636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_descendants_imply_infinite_sequence_l376_37635

-- Define a type for people
variable {Person : Type}

-- Define the relation "is a child of"
variable (is_child_of : Person → Person → Prop)

-- Define the relation "is a descendant of"
def is_descendant_of (is_child_of : Person → Person → Prop) (x y : Person) : Prop :=
  TC is_child_of x y

-- Define what it means to have infinitely many descendants
def has_infinitely_many_descendants (is_child_of : Person → Person → Prop) (a : Person) : Prop :=
  ¬ (∃ n : ℕ, ∀ d : Person, is_descendant_of is_child_of d a → (∃ m : ℕ, m ≤ n))

-- Define an infinite sequence of descendants
def infinite_descendant_sequence (is_child_of : Person → Person → Prop) (a : Person) (seq : ℕ → Person) : Prop :=
  seq 0 = a ∧ ∀ n : ℕ, is_child_of (seq (n + 1)) (seq n)

-- State the theorem
theorem descendants_imply_infinite_sequence
  (h_finite_children : ∀ p : Person, ∃ n : ℕ, ∀ c : Person, is_child_of c p → (∃ m : ℕ, m ≤ n))
  (a : Person)
  (h_inf_descendants : has_infinitely_many_descendants is_child_of a) :
  ∃ seq : ℕ → Person, infinite_descendant_sequence is_child_of a seq :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_descendants_imply_infinite_sequence_l376_37635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_off_pyramid_volume_l376_37663

/-- A right square pyramid with given dimensions -/
structure RightSquarePyramid where
  base_edge : ℝ
  slant_edge : ℝ
  height : ℝ

/-- The volume of a right square pyramid -/
noncomputable def pyramid_volume (p : RightSquarePyramid) : ℝ :=
  (1/3) * p.base_edge^2 * p.height

/-- The theorem stating the volume of the cut-off pyramid -/
theorem cut_off_pyramid_volume 
  (original : RightSquarePyramid)
  (cut_height : ℝ) :
  original.base_edge = 12 →
  original.slant_edge = 15 →
  original.height = 9 →
  cut_height = 4 →
  ∃ (cut_off : RightSquarePyramid),
    cut_off.height = original.height - cut_height ∧
    cut_off.base_edge = original.base_edge * (cut_off.height / original.height) ∧
    pyramid_volume cut_off = 2000/27 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_off_pyramid_volume_l376_37663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_line_relations_l376_37652

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)

-- State the theorem
theorem plane_line_relations 
  (α β : Plane) (m n : Line)
  (h_distinct : α ≠ β ∧ m ≠ n)
  (h_non_intersecting : ¬ ∃ (l : Line), contains α l ∧ contains β l) :
  (parallel_planes α β ∧ contains α m → parallel_line_plane m β) ∧
  (perpendicular n α ∧ perpendicular n β ∧ perpendicular m α → perpendicular m β) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_line_relations_l376_37652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_block_bounds_l376_37650

/-- Function representing the smallest number of 3-stamp blocks that can be removed
    from an n × n stamp sheet to prevent further removals -/
def b : ℕ → ℕ := sorry

/-- Theorem stating the bounds for the function b(n) -/
theorem stamp_block_bounds :
  ∃ (c d : ℝ), ∀ (n : ℕ), n > 0 →
    ((1 / 7 : ℝ) * n^2 - c * n : ℝ) ≤ b n ∧ (b n : ℝ) ≤ (1 / 5 : ℝ) * n^2 + d * n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_block_bounds_l376_37650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_calculation_l376_37630

/-- Profit percent calculation given selling price and loss percentage at reduced price --/
theorem profit_percent_calculation (P : ℝ) (h : P > 0) : 
  let reduced_price := (2/3) * P
  let cost_price := reduced_price / 0.82
  let profit := P - cost_price
  let profit_percent := (profit / cost_price) * 100
  ∃ (ε : ℝ), ε > 0 ∧ |profit_percent - 23| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_calculation_l376_37630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operation_equality_l376_37662

theorem set_operation_equality : (({1, 2} : Set ℕ) ∩ {1, 2, 3}) ∪ {2, 3, 4} = {1, 2, 3, 4} := by
  -- Define the sets
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {1, 2, 3}
  let C : Set ℕ := {2, 3, 4}
  
  -- Prove the equality
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operation_equality_l376_37662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_zero_l376_37658

-- Define the curves in polar coordinates
noncomputable def curve_P (θ : ℝ) : ℝ × ℝ := (10 * Real.sin θ * Real.cos θ, 10 * Real.sin θ * Real.sin θ)
noncomputable def curve_Q (θ : ℝ) : ℝ × ℝ := (10 / Real.sin θ * Real.cos θ, 10)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem min_distance_zero :
  ∀ ε > 0, ∃ θ₁ θ₂ : ℝ, distance (curve_P θ₁) (curve_Q θ₂) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_zero_l376_37658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l376_37622

/-- The speed that minimizes the cost per kilometer for a ship -/
noncomputable def optimal_speed : ℝ := 20

/-- The maximum speed of the ship in km/h -/
noncomputable def max_speed : ℝ := 25

/-- The constant of proportionality for fuel cost -/
noncomputable def k : ℝ := 35 / 1000

/-- The fixed costs per hour -/
noncomputable def fixed_costs : ℝ := 560

/-- The cost per kilometer as a function of speed -/
noncomputable def cost_per_km (v : ℝ) : ℝ := (k * v^3 + fixed_costs) / v

theorem optimal_speed_minimizes_cost :
  0 < optimal_speed ∧ 
  optimal_speed ≤ max_speed ∧
  ∀ v : ℝ, 0 < v ∧ v ≤ max_speed → cost_per_km optimal_speed ≤ cost_per_km v :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l376_37622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l376_37679

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (4 - Real.sqrt (7 - Real.sqrt (2 * x)))

theorem g_domain : Set ℝ = { x : ℝ | 0 ≤ x ∧ x ≤ 24.5 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l376_37679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l376_37660

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2

theorem f_properties :
  ∃ (A ω φ B : ℝ),
    (∀ x, f x = A * Real.sin (ω * x + φ) + B) ∧
    (A = 1 ∧ ω = 2 ∧ φ = -π/6 ∧ B = -1/2) ∧
    (∀ x, f x = Real.sin (2 * x - π/6) - 1/2) ∧
    (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧
    (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π/6) (k * π + π/3),
      ∀ y ∈ Set.Icc (k * π - π/6) (k * π + π/3),
      x ≤ y → f x ≤ f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l376_37660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l376_37672

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def has_properties (t : Triangle) : Prop :=
  let AB := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let AC := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  let BC := Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)
  let angle_A := Real.arccos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC))
  let area := 0.5 * AB * AC * Real.sin angle_A
  AB = 8 ∧ angle_A = 30 * (Real.pi / 180) ∧ area = 16

-- State the theorem
theorem triangle_side_length (t : Triangle) :
  has_properties t →
  Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l376_37672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l376_37617

-- Define the circle on which P moves
def circle_P (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Define the rotation of a vector by 90 degrees counterclockwise
def rotate90 (v : ℝ × ℝ) : ℝ × ℝ := (-v.2, v.1)

-- Define the locus equation for Q
def locus_Q (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 2

-- Theorem statement
theorem locus_of_Q (p : ℝ × ℝ) (h : circle_P p.1 p.2) :
  ∃ q : ℝ × ℝ, q = (p.1 + (-p.1), p.2 + (-p.2)) + rotate90 (-p.1, -p.2) ∧ locus_Q q.1 q.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l376_37617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_line_l376_37688

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- A predicate stating that a line passes through a focus and a vertex of an ellipse -/
def passes_through_focus_and_vertex (l : Line) (e : Ellipse) : Prop :=
  ∃ (f_x f_y v_x v_y : ℝ),
    -- Focus condition
    f_x^2 / e.a^2 + f_y^2 / e.b^2 = 1 ∧
    f_x^2 + f_y^2 = e.a^2 - e.b^2 ∧
    -- Vertex condition
    v_x^2 / e.a^2 + v_y^2 / e.b^2 = 1 ∧
    (v_x = e.a ∨ v_x = -e.a ∨ v_y = e.b ∨ v_y = -e.b) ∧
    -- Line passes through both points
    l.a * f_x + l.b * f_y + l.c = 0 ∧
    l.a * v_x + l.b * v_y + l.c = 0

/-- The main theorem -/
theorem ellipse_eccentricity_from_line (e : Ellipse) (l : Line) :
  l.a = 1 ∧ l.b = 2 ∧ l.c = -2 →
  passes_through_focus_and_vertex l e →
  eccentricity e = 2 * Real.sqrt 5 / 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_line_l376_37688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_properties_l376_37624

-- Define the circles C₁ and C₂
def C₁ (p : ℝ × ℝ) : Prop := (p.1^2 + p.2^2 - 3*p.1 - 3*p.2 + 3) = 0
def C₂ (p : ℝ × ℝ) : Prop := (p.1^2 + p.2^2 - 2*p.1 - 2*p.2) = 0

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define the common chord line
def common_chord (p : ℝ × ℝ) : Prop := p.1 + p.2 - 3 = 0

-- State the theorem
theorem circles_intersection_properties :
  -- 1. The equation of the line containing the common chord AB is x + y - 3 = 0
  (∀ p : ℝ × ℝ, C₁ p ∧ C₂ p → common_chord p) ∧
  -- 2. C₁ has the smallest area among all circles passing through A and B
  (∀ C : (ℝ × ℝ → Prop), C A ∧ C B →
    ∃ r : ℝ, r > 0 ∧ ∀ p : ℝ × ℝ, C p ↔ (p.1 - A.1)^2 + (p.2 - A.2)^2 = r^2) →
  ∃ r₁ : ℝ, r₁ > 0 ∧
    (∀ p : ℝ × ℝ, C₁ p ↔ (p.1 - 3/2)^2 + (p.2 - 3/2)^2 = r₁^2) ∧
    (∀ C : (ℝ × ℝ → Prop), C A ∧ C B →
      ∃ r : ℝ, r > 0 ∧ (∀ p : ℝ × ℝ, C p ↔ (p.1 - A.1)^2 + (p.2 - A.2)^2 = r^2) ∧ r₁ ≤ r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_properties_l376_37624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l376_37629

/-- Given a hyperbola with standard form (x²/a² - y²/b² = 1), one focus at (1, 0), and eccentricity √5,
    prove that its equation is 5x² - 5/4y² = 1 -/
theorem hyperbola_equation (a b c : ℝ) (h1 : c = 1) (h2 : c^2 = a^2 + b^2) (h3 : c/a = Real.sqrt 5) :
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ 5*x^2 - 5/4*y^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l376_37629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l376_37606

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 16) :
  let r := (d / 2) * (Real.sqrt 2 - 1) / Real.sqrt 2
  (4 / 3) * Real.pi * r^3 = 1024 * (2 * Real.sqrt 2 - 3)^3 * Real.pi / 9 := by
  sorry

#check inscribed_sphere_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l376_37606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l376_37681

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x - 2) / Real.sqrt (x^2 - 5*x + 6)

-- Define the domain of g
def domain_g : Set ℝ := {x | x < 2 ∨ x > 3}

-- Theorem stating that domain_g is the correct domain of g
theorem g_domain : 
  ∀ x, x ∈ domain_g ↔ (∃ y, g x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l376_37681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_9_45_l376_37682

noncomputable def clock_angle (hour : ℝ) (minute : ℝ) : ℝ :=
  |60 * hour - 11 * minute| / 2

theorem angle_at_9_45 :
  clock_angle 9 45 = 22.5 := by
  -- Unfold the definition of clock_angle
  unfold clock_angle
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_9_45_l376_37682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nacl_formed_l376_37640

/-- Represents the number of moles of a substance -/
def Moles : Type := ℕ

/-- Represents a chemical reaction -/
structure Reaction where
  reactant1 : Moles
  reactant2 : Moles
  product : Moles

instance : OfNat Moles n where
  ofNat := n

instance : Min Moles where
  min := Nat.min

/-- The NaOH + HCl → NaCl + H2O reaction -/
def naclReaction : Reaction :=
  { reactant1 := 1, reactant2 := 1, product := 1 }

/-- The amount of NaOH available -/
def naohAvailable : Moles := 3

/-- The amount of HCl available -/
def hclAvailable : Moles := 3

/-- Calculates the amount of product formed based on available reactants and the reaction -/
def productFormed (availableReactant1 availableReactant2 : Moles) (reaction : Reaction) : Moles :=
  min availableReactant1 availableReactant2

theorem nacl_formed :
  productFormed naohAvailable hclAvailable naclReaction = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nacl_formed_l376_37640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l376_37604

noncomputable section

/-- The function representing the graph -/
def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

/-- The vertical asymptote -/
def vertical_asymptote : ℝ := 3

/-- The horizontal asymptote -/
def horizontal_asymptote : ℝ := 1

/-- The point of intersection of the asymptotes -/
def intersection_point : ℝ × ℝ := (3, 1)

theorem asymptotes_intersection :
  (∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x, 0 < |x - vertical_asymptote| ∧ |x - vertical_asymptote| < δ → |f x| > 1/ε) ∧
  (∀ ε > (0 : ℝ), ∃ M : ℝ, ∀ x, |x| > M → |f x - horizontal_asymptote| < ε) ∧
  intersection_point = (vertical_asymptote, horizontal_asymptote) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l376_37604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parachuteOpensAtSixSeconds_l376_37690

/-- The height of a rocket as a function of time -/
def rocketHeight (t : ℝ) : ℝ := -t^2 + 12*t + 1

/-- The time when the rocket reaches a specific height -/
noncomputable def timeAtHeight (h : ℝ) : ℝ := 
  let t := (12 + Real.sqrt (144 - 4*(-1)*(1 - h))) / 2
  if rocketHeight t = h then t else 0

theorem parachuteOpensAtSixSeconds :
  timeAtHeight 37 = 6 := by
  sorry

#eval rocketHeight 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parachuteOpensAtSixSeconds_l376_37690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l376_37673

open Real

-- Define the curves C₁ and C₂
noncomputable def C₁ (x y : ℝ) : Prop := x + y = 1
noncomputable def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the polar coordinates of points A and B
noncomputable def ρ_A (α : ℝ) : ℝ := 1 / (cos α + sin α)
noncomputable def ρ_B (α : ℝ) : ℝ := 4 * cos α

-- State the theorem
theorem intersection_angle (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : ρ_B α / ρ_A α = 4) : 
  α = 3*π/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l376_37673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l376_37655

/-- Given that cos(100°) / (1 - 4 * sin(25°) * cos(25°) * cos(50°)) = tan(x°), prove that x = 95. -/
theorem angle_equality (x : ℝ) :
  (Real.cos (100 * π / 180)) / (1 - 4 * Real.sin (25 * π / 180) * Real.cos (25 * π / 180) * Real.cos (50 * π / 180)) = Real.tan (x * π / 180) →
  x = 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l376_37655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_equivalence_l376_37697

theorem prime_divisibility_equivalence (p : ℕ) (hp : Nat.Prime p) :
  (∃ x₀ : ℤ, (p : ℤ) ∣ (x₀^2 - x₀ + 3)) ↔ (∃ y₀ : ℤ, (p : ℤ) ∣ (y₀^2 - y₀ + 25)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_equivalence_l376_37697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_license_plates_l376_37618

-- Define the number of letters in the alphabet
def num_letters : ℕ := 26

-- Define the number of digits
def num_digits : ℕ := 10

-- Define the number of letter positions in the license plate
def num_letter_positions : ℕ := 2

-- Define the number of digit positions in the license plate
def num_digit_positions : ℕ := 3

-- Define the function to calculate the number of license plates
def num_license_plates : ℕ := (Nat.descFactorial num_letters num_letter_positions) * (num_digits ^ num_digit_positions)

-- Theorem statement
theorem max_license_plates :
  num_license_plates = (Nat.descFactorial num_letters num_letter_positions) * (num_digits ^ num_digit_positions) :=
by
  -- Unfold the definition of num_license_plates
  unfold num_license_plates
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_license_plates_l376_37618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_f_odd_f_inequality_solution_l376_37671

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_property (x₁ x₂ : ℝ) (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) : 
  f (x₁ * x₂) = f x₁ + f x₂

axiom f_increasing : StrictMono f

-- Theorem statements
theorem f_zero : f 1 = 0 ∧ f (-1) = 0 := by sorry

theorem f_odd (x : ℝ) (hx : x ≠ 0) : f (-x) = f x := by sorry

theorem f_inequality_solution : 
  {x : ℝ | f x + f (x - 1/2) ≤ 0} = 
    Set.Icc ((1 - Real.sqrt 17) / 4) 0 ∪ 
    Set.Ioo 0 (1/2) ∪ 
    Set.Icc (1/2) ((1 + Real.sqrt 17) / 4) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_f_odd_f_inequality_solution_l376_37671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l376_37668

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x / (Real.sin x + 3)

-- State the theorem
theorem sum_of_max_and_min_f :
  ∃ (max min : ℝ), 
    (∀ x, f x ≤ max) ∧ 
    (∃ x, f x = max) ∧
    (∀ x, min ≤ f x) ∧ 
    (∃ x, f x = min) ∧
    max + min = -1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l376_37668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_orientation_theorem_l376_37610

/-- A simple planar graph -/
structure SimplePlanarGraph where
  vertices : Type
  edges : Set (vertices × vertices)
  simple : ∀ e ∈ edges, e.1 ≠ e.2
  planar : True  -- Planarity condition (simplified for now)

/-- An orientation of edges in a graph -/
def orientation (G : SimplePlanarGraph) := G.edges → Bool

/-- A cycle in a graph -/
def cycle (G : SimplePlanarGraph) := List G.vertices

/-- The fraction of edges in a cycle sharing the same orientation -/
noncomputable def orientedFraction (G : SimplePlanarGraph) (o : orientation G) (c : cycle G) : ℚ :=
  sorry

theorem edge_orientation_theorem (G : SimplePlanarGraph) :
  ∃ (o : orientation G), ∀ (c : cycle G), orientedFraction G o c ≤ 3/4 ∧
  ∀ (ε : ℚ), ε > 0 → ∃ (G' : SimplePlanarGraph) (o' : orientation G') (c' : cycle G'),
    orientedFraction G' o' c' > 3/4 - ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_orientation_theorem_l376_37610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l376_37614

noncomputable def f (x : ℝ) := Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

theorem f_properties :
  let f := f
  -- The smallest positive period is π
  ∃ T : ℝ, T > 0 ∧ T = Real.pi ∧ ∀ x, f (x + T) = f x ∧
    ∀ T' : ℝ, T' > 0 ∧ (∀ x, f (x + T') = f x) → T ≤ T' ∧
  -- The function is monotonically increasing on the given intervals
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi/12) (k * Real.pi + 5*Real.pi/12)) ∧
  -- The symmetry center
  ∀ k : ℤ, ∀ x : ℝ, f (k * Real.pi/2 + Real.pi/6 + x) = f (k * Real.pi/2 + Real.pi/6 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l376_37614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_max_value_f_l376_37638

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin (x - Real.pi/3), 1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, 1)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem parallel_vectors_tan (x : ℝ) :
  (∃ (k : ℝ), m x = k • (n x)) → Real.tan x = 2 + Real.sqrt 3 := by sorry

theorem max_value_f :
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/2) ∧
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi/2) → f y ≤ f x) →
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/2) ∧ f x = (6 - Real.sqrt 3) / 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_max_value_f_l376_37638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_depleted_l376_37695

/-- Represents the tablet's battery life and usage --/
structure TabletBattery where
  idle_life : ℚ  -- Battery life when not in use (in hours)
  active_life : ℚ  -- Battery life when actively used (in hours)
  total_use : ℚ  -- Total time the tablet was on (in hours)
  active_use : ℚ  -- Time the tablet was actively used (in hours)

/-- Calculates the remaining battery life after usage --/
def remaining_battery (tb : TabletBattery) : ℚ :=
  1 - (tb.active_use / tb.active_life + (tb.total_use - tb.active_use) / tb.idle_life)

/-- Theorem stating that the battery is depleted after Marie's usage --/
theorem battery_depleted (tb : TabletBattery) 
  (h1 : tb.idle_life = 10)
  (h2 : tb.active_life = 2)
  (h3 : tb.total_use = 12)
  (h4 : tb.active_use = 3) : 
  remaining_battery tb ≤ 0 := by
  sorry

/-- Example calculation --/
def example_tablet : TabletBattery := {
  idle_life := 10,
  active_life := 2,
  total_use := 12,
  active_use := 3
}

#eval remaining_battery example_tablet

end NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_depleted_l376_37695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_minus_sqrt3_product_l376_37696

theorem tan_sum_minus_sqrt3_product :
  Real.tan (42 * Real.pi / 180) + Real.tan (78 * Real.pi / 180) - 
  Real.sqrt 3 * Real.tan (42 * Real.pi / 180) * Real.tan (78 * Real.pi / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_minus_sqrt3_product_l376_37696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l376_37611

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -4 then -0.2 * x + 1.2
  else if x ≤ 0 then -0.5 * x
  else if x ≤ 2 then 2.5 * x
  else (x + 13) / 3

theorem f_satisfies_conditions :
  (f (-4) = 2 ∧ f 0 = 0 ∧ f 2 = 5) ∧
  (f (-9) = 3 ∧ f 5 = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l376_37611
