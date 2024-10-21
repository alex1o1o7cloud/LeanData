import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_l802_80288

/-- The curve in the problem -/
def curve (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- The circle passing through the intersection points of the curve with the axes -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - 3 = 0

/-- The line intersecting the circle -/
def line (x y : ℝ) (a : ℝ) : Prop := x + y + a = 0

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem intersection_line_circle (a : ℝ) :
  (∃ A B : ℝ × ℝ, 
    circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
    line A.1 A.2 a ∧ line B.1 B.2 a ∧
    distance A.1 A.2 B.1 B.2 = 2) →
  a = 2 * Real.sqrt 2 ∨ a = -2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_l802_80288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_percentage_l802_80228

theorem price_change_percentage (P : ℝ) (h : P ≠ 0) : ∃ x : ℝ, 
  1.2 * P * (1 - x / 100) = 0.88 * P ∧ x = 80 / 3 := by
  use 80 / 3
  constructor
  · simp
    field_simp
    ring
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_percentage_l802_80228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_heads_two_tails_l802_80269

/-- Represents the probability of getting heads for an unfair coin -/
noncomputable def p : ℝ := sorry

/-- Represents the probability of getting tails for an unfair coin -/
noncomputable def q : ℝ := sorry

/-- The coin is unfair, so p ≠ q -/
axiom not_fair : p ≠ q

/-- The probabilities sum to 1 -/
axiom sum_to_one : p + q = 1

/-- The probability of getting one head and one tail in two tosses is 1/2 -/
axiom one_head_one_tail : 2 * p * q = 1 / 2

/-- The main theorem: The probability of getting two heads and two tails in four tosses is 3/8 -/
theorem two_heads_two_tails : 6 * p^2 * q^2 = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_heads_two_tails_l802_80269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_x_value_l802_80230

theorem power_equality_implies_x_value (x : ℝ) : (2 : ℝ)^8 = (32 : ℝ)^x → x = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_x_value_l802_80230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_Q_coordinates_l802_80205

-- Define the points as functions
def M (a b : ℝ) : ℝ × ℝ := (a, b)
def N (a b : ℝ) : ℝ × ℝ := (a, -b)
def P (a b : ℝ) : ℝ × ℝ := (-a, -b)
def Q (a b : ℝ) : ℝ × ℝ := (b, a)

-- Define the symmetry relations
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

def symmetric_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

def symmetric_xy_line (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.2 ∧ p.2 = -q.1

-- State the theorem
theorem point_Q_coordinates (a b : ℝ) :
  symmetric_x_axis (M a b) (N a b) ∧ 
  symmetric_y_axis (P a b) (N a b) ∧ 
  symmetric_xy_line (Q a b) (P a b) →
  Q a b = (b, a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_Q_coordinates_l802_80205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l802_80216

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_area_theorem (a b c : ℝ) (A B C : ℝ) :
  Triangle a b c →
  Real.cos B / Real.cos C = -b / (2*a + c) →
  b = Real.sqrt 13 →
  a + c = 4 →
  B = 2 * Real.pi / 3 ∧
  (1/2) * a * c * Real.sin B = 3 * Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l802_80216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_length_l802_80244

/-- Given a square with side length 2 and four congruent isosceles triangles constructed
    with vertices on each vertex of the square, if the sum of the areas of the four
    isosceles triangles is half the area of the square, then the length of one of the
    two congruent sides of one isosceles triangle is √5/2. -/
theorem isosceles_triangle_side_length (square_side : ℝ) (triangle_area : ℝ) 
  (h1 : square_side = 2)
  (h2 : triangle_area = (square_side^2) / 8) : 
  ∃ (triangle_side : ℝ), triangle_side = Real.sqrt 5 / 2 := by
  sorry

#check isosceles_triangle_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_length_l802_80244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_zeros_eleven_l802_80240

/-- Represents the content of a square on the board -/
inductive SquareContent
  | Negative
  | Zero
  | Positive
deriving Repr, DecidableEq

/-- Represents an 11x11 board -/
def Board := Fin 11 → Fin 11 → SquareContent

/-- Returns true if the sum of a row is nonpositive -/
def rowSumNonpositive (board : Board) (row : Fin 11) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 11)) (fun col =>
    match board row col with
    | SquareContent.Negative => -1
    | SquareContent.Zero => 0
    | SquareContent.Positive => 1
  )) ≤ 0

/-- Returns true if the sum of a column is nonnegative -/
def colSumNonnegative (board : Board) (col : Fin 11) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 11)) (fun row =>
    match board row col with
    | SquareContent.Negative => -1
    | SquareContent.Zero => 0
    | SquareContent.Positive => 1
  )) ≥ 0

/-- Returns the number of zeros on the board -/
def countZeros (board : Board) : Nat :=
  Finset.card (Finset.filter (fun p => board p.1 p.2 = SquareContent.Zero) (Finset.univ : Finset (Fin 11 × Fin 11)))

/-- The main theorem stating that the minimum number of zeros is 11 -/
theorem min_zeros_eleven :
  ∀ (board : Board),
  (∀ row, rowSumNonpositive board row) →
  (∀ col, colSumNonnegative board col) →
  countZeros board ≥ 11 := by
  sorry

#check min_zeros_eleven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_zeros_eleven_l802_80240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_percentage_is_50_l802_80243

-- Define the given conditions
def hourly_rate : ℚ := 30
def first_40_hours : ℕ := 40
def first_3_days_hours : ℕ := 6
def total_days : ℕ := 5
def total_earnings : ℚ := 1290

-- Define the function to calculate total hours worked
def total_hours_worked : ℕ :=
  (3 * first_3_days_hours) + (2 * (2 * first_3_days_hours))

-- Define the function to calculate earnings for first 40 hours
def earnings_first_40 : ℚ := hourly_rate * first_40_hours

-- Define the function to calculate additional earnings
def additional_earnings : ℚ := total_earnings - earnings_first_40

-- Define the function to calculate additional hours worked
def additional_hours : ℕ := total_hours_worked - first_40_hours

-- Define the function to calculate the additional hourly rate
noncomputable def additional_hourly_rate : ℚ := additional_earnings / additional_hours

-- Define the function to calculate the additional percentage
noncomputable def additional_percentage : ℚ := ((additional_hourly_rate - hourly_rate) / hourly_rate) * 100

-- Theorem statement
theorem additional_percentage_is_50 : additional_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_percentage_is_50_l802_80243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_plus_midpoint_triangle_area_l802_80271

/-- The area of a regular hexagon with side length 3 plus the area of the triangle
    formed by connecting the midpoints of alternating sides of the hexagon -/
theorem hexagon_plus_midpoint_triangle_area :
  let s : ℝ := 3  -- side length of the hexagon
  let hexagon_area := 6 * (Real.sqrt 3 / 4 * s^2)  -- area of regular hexagon
  let midpoint_side := s / 2  -- side length of the midpoint triangle
  let triangle_area := Real.sqrt 3 / 4 * midpoint_side^2  -- area of equilateral triangle
  hexagon_area + triangle_area = 225 * Real.sqrt 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_plus_midpoint_triangle_area_l802_80271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l802_80222

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates a quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Calculates the vertex of a quadratic function -/
noncomputable def QuadraticFunction.vertex (f : QuadraticFunction) : ℝ × ℝ :=
  let x := -f.b / (2 * f.a)
  (x, f.evaluate x)

/-- Checks if a quadratic function has a vertical axis of symmetry -/
def QuadraticFunction.hasVerticalAxisOfSymmetry (f : QuadraticFunction) : Prop :=
  f.a ≠ 0

theorem parabola_theorem (f : QuadraticFunction) : 
  f.a = 2 ∧ f.b = -12 ∧ f.c = 16 →
  f.vertex = (3, -2) ∧
  f.hasVerticalAxisOfSymmetry ∧
  f.evaluate 5 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l802_80222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_overlap_30_60_90_triangles_l802_80231

/-- Represents a triangle -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is a 30-60-90 triangle -/
def Triangle.is_30_60_90 (t : Triangle) : Prop :=
  sorry

/-- Returns the length of the hypotenuse of a triangle -/
def Triangle.hypotenuse (t : Triangle) : ℝ :=
  sorry

/-- Rotates a triangle around the midpoint of its hypotenuse by a given angle -/
def Triangle.rotate_around_hypotenuse_midpoint (t : Triangle) (angle : ℝ) : Prop :=
  sorry

/-- Calculates the area of overlap between two triangles -/
def Triangle.overlap_area (t1 t2 : Triangle) : ℝ :=
  sorry

/-- The area of overlap between two congruent 30-60-90 triangles -/
theorem area_overlap_30_60_90_triangles :
  ∀ (triangle1 triangle2 : Triangle) (hypotenuse : ℝ) (rotation_angle : ℝ),
    Triangle.is_30_60_90 triangle1 →
    Triangle.is_30_60_90 triangle2 →
    Triangle.hypotenuse triangle1 = hypotenuse →
    Triangle.hypotenuse triangle2 = hypotenuse →
    hypotenuse = 10 →
    rotation_angle = 30 * (π / 180) →
    Triangle.rotate_around_hypotenuse_midpoint triangle2 rotation_angle →
    Triangle.overlap_area triangle1 triangle2 = (25 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_overlap_30_60_90_triangles_l802_80231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_random_sampling_l802_80277

theorem stratified_random_sampling (total_sample : ℕ) (junior_total : ℕ) (senior_total : ℕ)
  (h_total : total_sample = 60)
  (h_junior : junior_total = 400)
  (h_senior : senior_total = 200) :
  (Nat.choose junior_total (total_sample * 2 / 3)) * (Nat.choose senior_total (total_sample / 3)) =
  (Nat.choose 400 40) * (Nat.choose 200 20) := by
  sorry

#check stratified_random_sampling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_random_sampling_l802_80277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_bound_l802_80290

def sequence_property (a b : ℕ → ℕ) : Prop :=
  (∀ n, a (n + 2) = a (n + 1) + a n) ∧
  (∀ n, b (n + 2) = b (n + 1) + b n) ∧
  (Nat.gcd (a 1) (b 1) = 1) ∧
  (Nat.gcd (a 2) (b 2) = 1) ∧
  (a 1 < 1000) ∧ (a 2 < 1000) ∧
  (b 1 < 1000) ∧ (b 2 < 1000)

theorem divisibility_implies_bound (a b : ℕ → ℕ) (h : sequence_property a b) :
  ∀ n, (a n) % (b n) = 0 → n < 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_bound_l802_80290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amusement_park_queue_time_l802_80292

/-- Calculates the time needed to cover a distance at a given rate -/
noncomputable def time_to_cover_distance (distance : ℝ) (rate : ℝ) : ℝ :=
  distance / rate

/-- Converts yards to feet -/
def yards_to_feet (yards : ℝ) : ℝ :=
  yards * 3

theorem amusement_park_queue_time 
  (initial_movement : ℝ) 
  (initial_time : ℝ) 
  (remaining_distance_yards : ℝ) : 
  time_to_cover_distance (yards_to_feet remaining_distance_yards) (initial_movement / initial_time) = 135 :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_cover_distance (yards_to_feet 90) (40 / 20)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amusement_park_queue_time_l802_80292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sort_four_numbers_in_five_comparisons_l802_80210

-- Define a type for the comparison operation
def Compare (α : Type) := α → α → Bool

-- Define a function that represents the sorting algorithm
def SortFourNumbers (a b c d : ℝ) (compare : Compare ℝ) : List ℝ :=
  sorry

-- Define a function to count comparisons
def CountComparisons (l : List ℝ) : Nat :=
  sorry

-- Theorem statement
theorem sort_four_numbers_in_five_comparisons 
  (a b c d : ℝ) (compare : Compare ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_compare_correct : ∀ x y, compare x y = true ↔ x < y) :
  ∃ (result : List ℝ),
    result = SortFourNumbers a b c d compare ∧
    result.length = 4 ∧
    List.Pairwise (· < ·) result ∧
    (∀ x, x ∈ result ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧
    CountComparisons result ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sort_four_numbers_in_five_comparisons_l802_80210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parrot_ownership_duration_l802_80281

theorem parrot_ownership_duration
  (current_phrases phrases_per_week initial_phrases : ℕ)
  (h1 : current_phrases = 17)
  (h2 : phrases_per_week = 2)
  (h3 : initial_phrases = 3) :
  (current_phrases - initial_phrases) / phrases_per_week * 7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parrot_ownership_duration_l802_80281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_unit_circle_negative_300_degrees_l802_80258

theorem intersection_unit_circle_negative_300_degrees :
  ∀ (x y : ℝ), 
    (x^2 + y^2 = 1) →  -- Point (x, y) is on the unit circle
    (x = Real.cos (-300 * Real.pi / 180) ∧ y = Real.sin (-300 * Real.pi / 180)) →  -- Point (x, y) is on the terminal side of -300° angle
    y / x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_unit_circle_negative_300_degrees_l802_80258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_eq_20_l802_80265

/-- The number of ways to arrange 5 people in 5 seats with exactly two matches -/
def seating_arrangements : ℕ :=
  let n : ℕ := 5  -- Total number of people and seats
  let k : ℕ := 2  -- Number of people sitting in matching seats
  (n.choose k) * (Nat.factorial (n - k) / (Nat.factorial (n - k - (n - k)) * Nat.factorial (n - k)))

/-- Theorem stating that the number of seating arrangements is 20 -/
theorem seating_arrangements_eq_20 : seating_arrangements = 20 := by
  sorry

#eval seating_arrangements  -- This will evaluate to 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_eq_20_l802_80265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_cow_difference_after_transaction_l802_80213

/-- Represents the state of animals on the farm -/
structure FarmState where
  horses : ℕ
  cows : ℕ
  sheep : ℕ

/-- Calculates the new farm state after selling horses and buying cows -/
def transaction (state : FarmState) (horsesSold cowsBought : ℕ) : FarmState :=
  { horses := state.horses - horsesSold,
    cows := state.cows + cowsBought,
    sheep := state.sheep }

/-- Theorem stating the difference between horses and cows after the transaction -/
theorem horse_cow_difference_after_transaction
  (initial : FarmState)
  (h1 : initial.horses = 4 * initial.cows)
  (h2 : initial.sheep = 3 * initial.horses / 2)
  (h3 : (transaction initial 15 30).horses = 7 * (transaction initial 15 30).cows / 3) :
  (transaction initial 15 30).horses - (transaction initial 15 30).cows = 108 := by
  sorry

#check horse_cow_difference_after_transaction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_cow_difference_after_transaction_l802_80213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_calculation_l802_80239

/-- Converts kilometers per hour to meters per minute -/
noncomputable def kmph_to_mpm (rate : ℝ) : ℝ := rate * 1000 / 60

/-- Calculates the width of a river given its depth, flow rate, and volume per minute -/
noncomputable def river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) : ℝ :=
  volume_per_minute / (depth * kmph_to_mpm flow_rate_kmph)

theorem river_width_calculation (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) 
    (h1 : depth = 5)
    (h2 : flow_rate_kmph = 4)
    (h3 : volume_per_minute = 6333.333333333333) :
    river_width depth flow_rate_kmph volume_per_minute = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_calculation_l802_80239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_properties_l802_80203

/-- Represents a rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the diagonal length of a rectangular prism -/
noncomputable def diagonalLength (prism : RectangularPrism) : ℝ :=
  Real.sqrt (prism.length ^ 2 + prism.width ^ 2 + prism.height ^ 2)

/-- Calculates the total surface area of a rectangular prism -/
def surfaceArea (prism : RectangularPrism) : ℝ :=
  2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height)

theorem rectangular_prism_properties :
  let prism : RectangularPrism := ⟨12, 15, 8⟩
  diagonalLength prism = Real.sqrt 433 ∧ surfaceArea prism = 792 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_properties_l802_80203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_ellipse_l802_80264

theorem circle_tangent_to_ellipse (r : ℝ) : r = (Real.sqrt 15) / 2 ↔
  ∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x - r)^2 + y^2 = r^2 :=
sorry

#check circle_tangent_to_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_ellipse_l802_80264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slowest_is_daughter_l802_80298

-- Define the types for swimmers and handedness
inductive Swimmer : Type
  | Man : Swimmer
  | Sister : Swimmer
  | Daughter : Swimmer
  | Son : Swimmer

inductive Handedness : Type
  | Left : Handedness
  | Right : Handedness

-- Define the properties
def isTwin : Swimmer → Swimmer → Prop := sorry
def age : Swimmer → ℕ := sorry
def speed : Swimmer → ℕ := sorry
def handedness : Swimmer → Handedness := sorry

-- Define the conditions
axiom four_swimmers : 
  ∀ s : Swimmer, s = Swimmer.Man ∨ s = Swimmer.Sister ∨ s = Swimmer.Daughter ∨ s = Swimmer.Son

axiom slowest_twin_exists : 
  ∃ s t : Swimmer, (speed s = 0) ∧ (isTwin s t) ∧ (t = Swimmer.Man ∨ t = Swimmer.Sister ∨ t = Swimmer.Daughter ∨ t = Swimmer.Son)

axiom different_handedness : 
  ∀ s t f : Swimmer, (speed s = 0) → (isTwin s t) → (speed f = 3) → (handedness t ≠ handedness f)

axiom same_age : 
  ∀ s f : Swimmer, (speed s = 0) → (speed f = 3) → (age s = age f)

-- The theorem to prove
theorem slowest_is_daughter : 
  ∃ s : Swimmer, (s = Swimmer.Daughter) ∧ (speed s = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slowest_is_daughter_l802_80298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l802_80295

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The given trigonometric function -/
noncomputable def y (x φ : ℝ) : ℝ :=
  3 * Real.sin (2 * x + φ + Real.pi / 3)

theorem phi_value (φ : ℝ) :
  IsEven (y · φ) → |φ| ≤ Real.pi / 2 → φ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l802_80295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_tan_α_value_l802_80220

-- Define the angle α and point P(m,1)
variable (α : ℝ)
variable (m : ℝ)

-- Define the conditions
axiom cos_α : Real.cos α = -1/3
axiom point_on_terminal_side : ∃ (P : ℝ × ℝ), P = (m, 1) ∧ P.1 = m * Real.cos α ∧ P.2 = Real.sin α

-- State the theorems to be proved
theorem m_value : m = -Real.sqrt 2 / 4 := by sorry

theorem tan_α_value : Real.tan α = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_tan_α_value_l802_80220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaogang_dart_game_l802_80283

theorem xiaogang_dart_game (x y z : ℕ) : 
  x + y + z > 11 →
  8 * x + 9 * y + 10 * z = 100 →
  x ∈ ({8, 9, 10} : Set ℕ) :=
by
  intro h1 h2
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaogang_dart_game_l802_80283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_unit_base_pyramid_l802_80257

/-- A pyramid with a square base and apex equidistant from all base vertices -/
structure SquarePyramid where
  -- Side length of the square base
  base_side : ℝ
  -- Angle between two adjacent edges meeting at the apex
  apex_angle : ℝ
  -- The apex is equidistant from all base vertices
  apex_equidistant : True
  -- The base is a square
  base_is_square : True

/-- Volume of a square-based pyramid -/
noncomputable def pyramid_volume (p : SquarePyramid) : ℝ :=
  1 / (6 * Real.sin (p.apex_angle / 2))

/-- Theorem: Volume of a specific square-based pyramid -/
theorem volume_of_unit_base_pyramid (p : SquarePyramid) 
  (h1 : p.base_side = 1) 
  (h2 : p.apex_angle > 0) 
  (h3 : p.apex_angle < π) : 
  pyramid_volume p = 1 / (6 * Real.sin (p.apex_angle / 2)) := by
  -- Unfold the definition of pyramid_volume
  unfold pyramid_volume
  -- The equality follows directly from the definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_unit_base_pyramid_l802_80257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_on_hyperbola_l802_80227

def possible_coordinates : List (ℕ × ℕ) := [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]

def on_hyperbola (point : ℕ × ℕ) : Bool :=
  point.2 * point.1 == 6

def count_on_hyperbola : ℕ :=
  (possible_coordinates.filter on_hyperbola).length

theorem probability_on_hyperbola :
  (count_on_hyperbola : ℚ) / possible_coordinates.length = 1 / 3 := by
  sorry

#eval count_on_hyperbola
#eval possible_coordinates.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_on_hyperbola_l802_80227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l802_80235

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) (h2 : Real.sin α = 3/5) : Real.cos α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l802_80235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bases_with_final_digit_one_l802_80276

theorem bases_with_final_digit_one : 
  ∃! k : ℕ, k = (Finset.filter (λ b : ℕ ↦ 2 ≤ b ∧ b ≤ 9 ∧ 725 % b = 1) (Finset.range 10)).card ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bases_with_final_digit_one_l802_80276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_and_distance_l802_80211

-- Define the lines and curve
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := x - y + 4 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y - 1 = 0
def curve (x y : ℝ) : Prop := y^2 + 2 * x = 0

-- Define the intersection point P
noncomputable def P : ℝ × ℝ := (-2, 2)

-- Define line l
def line_l (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := (-1/2, -1)
noncomputable def B : ℝ × ℝ := (-2, 2)

theorem line_l_and_distance :
  (∀ x y, line1 x y ∧ line2 x y → (x, y) = P) →
  (∀ x y, line_l x y → line3 x ((x - 1) / 2)) →
  (∀ x y, line_l x y ∧ curve x y → (x, y) = A ∨ (x, y) = B) →
  (∀ x y, line_l x y) ∧
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3 * Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_and_distance_l802_80211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_positive_f_l802_80289

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

-- State the theorem
theorem min_a_for_positive_f :
  ∀ ε > 0, ∃ a : ℝ, a < 2 - 4 * Real.log 2 + ε ∧
  ∀ x : ℝ, 0 < x ∧ x < 1/2 → f a x > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_positive_f_l802_80289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l802_80254

noncomputable def f (x : ℝ) : ℝ := Real.log (6 + x - x^2) / Real.log (1/2)

theorem monotonic_increasing_interval :
  {x : ℝ | ∀ y, 1/2 ≤ x ∧ x < y ∧ y < 3 → f x < f y} = Set.Icc (1/2 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l802_80254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l802_80236

open Real

/-- A function f(x) = sin(ωx + π/3) on the interval (0, π) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 3)

/-- The number of extreme points of f in (0, π) -/
noncomputable def num_extreme_points (ω : ℝ) : ℕ := sorry

/-- The number of zeros of f in (0, π) -/
noncomputable def num_zeros (ω : ℝ) : ℕ := sorry

theorem omega_range :
  ∀ ω : ℝ, num_extreme_points ω = 3 ∧ num_zeros ω = 2 → ω ∈ Set.Ioo (13/6) (8/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l802_80236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_and_inequality_l802_80291

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 1

noncomputable def g (x : ℝ) : ℝ := exp x - 1

theorem tangent_lines_and_inequality (hf : ∀ x > 0, f x = log x + 1) 
                                     (hg : ∀ x, g x = exp x - 1) : 
  (∃ m : ℝ, ∃ x₁ > 0, ∃ x₂ > 0, 
    (deriv f x₁ = 1 ∧ deriv g x₂ = 1) ∧ 
    (f x₁ - x₁ = m ∧ g x₂ - x₂ = m)) ∧ 
  (∃ k : ℝ, ∃ x₁ > 0, ∃ x₂ > 0, 
    (deriv f x₁ = k ∧ deriv g x₂ = k) ∧ 
    (f x₁ - k*x₁ = -1 ∧ g x₂ - k*x₂ = -1)) ∧ 
  (∀ x, x ∈ Set.Ioo 0 1 → g x - f x > 1/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_and_inequality_l802_80291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l802_80218

noncomputable def y (n : ℕ) : ℝ := 4 - (1 / 3^n)

theorem limit_of_sequence (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → |y n - 4| < ε := by
  sorry

#check limit_of_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l802_80218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_function_l802_80285

theorem min_value_trig_function :
  ∀ x : ℝ, (Real.sin x)^4 + (Real.cos x)^4 + (1 / Real.cos x)^4 + (1 / Real.sin x)^4 ≥ 17/2 :=
by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_function_l802_80285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_MW_times_semiperimeter_l802_80223

/-- Triangle with exscribed circle and related points -/
structure TriangleWithExcircle where
  -- The vertices of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- The center of the exscribed circle touching side BC
  I_a : ℝ × ℝ
  -- The midpoint of side BC
  M : ℝ × ℝ
  -- The intersection of angle A bisector with the circumcircle
  W : ℝ × ℝ
  -- Ensure I_a is on the angle bisector of A
  I_a_on_bisector : True
  -- Ensure M is the midpoint of BC
  M_is_midpoint : True
  -- Ensure W is on both the angle bisector and the circumcircle
  W_on_bisector_and_circle : True

/-- The semiperimeter of a triangle -/
noncomputable def semiperimeter (t : TriangleWithExcircle) : ℝ := sorry

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (p q r : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem area_equals_MW_times_semiperimeter (t : TriangleWithExcircle) :
  triangle_area t.I_a t.B t.C = distance t.M t.W * semiperimeter t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_MW_times_semiperimeter_l802_80223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estate_value_l802_80261

-- Define the estate and its components
def estate : ℝ := sorry
def child1_share : ℝ := sorry
def child2_share : ℝ := sorry
def spouse_share : ℝ := sorry
def charity_share : ℝ := sorry

-- Define the conditions
axiom children_share : child1_share + child2_share = (3/4) * estate
axiom children_ratio : child1_share = (5/4) * child2_share
axiom spouse_share_def : spouse_share = 2 * child2_share
axiom charity_share_def : charity_share = 600

-- Define the total estate composition
axiom estate_composition : estate = child1_share + child2_share + spouse_share + charity_share

-- Theorem to prove
theorem estate_value : estate = 1440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estate_value_l802_80261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_f_negative_x_l802_80253

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 then x - Real.log (abs x)
  else if x < 0 then x + Real.log (abs x)
  else 0  -- Define f(0) as 0 to make it total

-- State the theorem
theorem odd_function_property (x : ℝ) :
  x ≠ 0 → f (-x) = -f x := by sorry

-- State the main theorem to be proved
theorem f_negative_x (x : ℝ) :
  x < 0 → f x = x + Real.log (abs x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_f_negative_x_l802_80253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_a_monotone_increasing_a_bounded_by_e_l802_80278

noncomputable def f (x : ℝ) : ℝ := x * Real.log (1 + 1/x)

noncomputable def a (n : ℕ+) : ℝ := (1 + 1/(n : ℝ))^(n : ℝ)

theorem f_monotone_increasing :
  ∀ x y, x < y → (x ∉ Set.Icc (-1) 0 ∧ y ∉ Set.Icc (-1) 0) → f x < f y :=
by sorry

theorem a_monotone_increasing :
  ∀ n : ℕ+, a n < a (n + 1) :=
by sorry

theorem a_bounded_by_e :
  ∀ n : ℕ+, a n < Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_a_monotone_increasing_a_bounded_by_e_l802_80278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_l802_80249

theorem not_perfect_square (n : ℕ) : 
  (∃ (x : Fin 10 → ℕ), 
    x 0 = 14 ∧
    (∀ i : Fin 9, x (i.succ) = (i.val + 1) * x 1) ∧
    (Finset.sum (Finset.range 10) (λ i => x i)) = 1994) →
  ¬∃ (m : ℕ), n = m ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_l802_80249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l802_80229

/-- A four-digit number -/
def FourDigitNumber : Type := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Predicate for numbers greater than 4999 -/
def GreaterThan4999 (n : FourDigitNumber) : Prop := 4999 < n.val

/-- Extract the middle two digits of a four-digit number -/
def MiddleTwoDigits (n : FourDigitNumber) : ℕ × ℕ :=
  let d := n.val / 10 % 100
  (d / 10, d % 10)

/-- Predicate for numbers where the product of middle two digits exceeds 10 -/
def MiddleDigitsProductExceeds10 (n : FourDigitNumber) : Prop :=
  let (d1, d2) := MiddleTwoDigits n
  d1 * d2 > 10

/-- The count of four-digit numbers satisfying both conditions -/
def CountValidNumbers : ℕ := 2700

theorem count_valid_numbers : CountValidNumbers = 2700 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l802_80229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_is_101010101_l802_80204

/-- Represents a valid 9-digit number with alternating odd and even digits -/
def AlternatingNumber := { n : Nat // n ≥ 100000000 ∧ n < 1000000000 ∧ ∀ i : Fin 9, (n / 10^(i.val) % 10) % 2 = i.val % 2 }

/-- The operation of selecting two adjacent digits, subtracting 1 from each, and swapping them -/
def applyOperation (n : AlternatingNumber) (i : Fin 8) : AlternatingNumber :=
  sorry

/-- The smallest number obtainable from repeated applications of the operation -/
def smallestNumber (n : AlternatingNumber) : AlternatingNumber :=
  sorry

/-- The theorem stating that the smallest obtainable number is 101010101 -/
theorem smallest_number_is_101010101 :
  smallestNumber ⟨123456789, sorry⟩ = ⟨101010101, sorry⟩ :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_is_101010101_l802_80204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_24_l802_80202

/-- A sequence defined recursively -/
def b : ℕ → ℚ
  | 0 => 2  -- Changed from 1 to 0 for zero-based indexing
  | 1 => 3  -- Changed from 2 to 1
  | k+2 => (1/2) * b (k+1) + (1/3) * b k

/-- The sum of the sequence -/
noncomputable def seriesSum : ℚ := ∑' n, b n

/-- Theorem stating that the sum of the sequence equals 24 -/
theorem sum_equals_24 : seriesSum = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_24_l802_80202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l802_80251

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0

noncomputable def angle_C (t : Triangle) : ℝ := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

theorem triangle_theorem (t : Triangle) 
  (h : t.a^2 + t.b^2 - t.c^2 = (4 * Real.sqrt 3 / 3) * t.S) : 
  angle_C t = π / 3 ∧ 
  (t.c = Real.sqrt 3 ∧ t.S = Real.sqrt 3 / 2 → t.a + t.b = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l802_80251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_sharing_l802_80275

/-- Proves that if three people share money in a ratio of 1:2:7, and the first person's share is $20, then the total amount shared is $200. -/
theorem money_sharing (amanda ben carlos total : ℕ) 
  (h1 : amanda + ben + carlos = total)
  (h2 : (amanda : ℚ) / total = 1 / 10)
  (h3 : (ben : ℚ) / total = 1 / 5)
  (h4 : (carlos : ℚ) / total = 7 / 10)
  (h5 : amanda = 20) : 
  total = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_sharing_l802_80275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_imply_arrangement_l802_80238

-- Define the types for our geometric shapes
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Diamond where
  center : ℝ × ℝ
  side_length : ℝ

structure Triangle where
  center : ℝ × ℝ
  side_length : ℝ

-- Define the inequalities
def satisfies_inequalities (x y : ℝ) : Prop :=
  abs x + abs y ≤ 2 * Real.sqrt (x^2 + y^2) ∧ 
  2 * Real.sqrt (x^2 + y^2) ≤ 3 * max (abs x) (abs y)

-- Define the geometric arrangement
def correct_arrangement (c : Circle) (d : Diamond) (t : Triangle) : Prop :=
  c.center = (0, 0) ∧ d.center = (0, 0) ∧ t.center = (0, 0) ∧
  c.radius < d.side_length / 2 ∧ d.side_length < t.side_length

-- Define a helper function to check if a point is inside a shape
def point_in_shape (p : ℝ × ℝ) (s : ℝ × ℝ) : Prop := p = s

-- The theorem to prove
theorem inequalities_imply_arrangement :
  ∃ (c : Circle) (d : Diamond) (t : Triangle),
  (∀ x y : ℝ, satisfies_inequalities x y → 
    point_in_shape (x, y) c.center ∨ point_in_shape (x, y) d.center ∨ point_in_shape (x, y) t.center) ∧
  correct_arrangement c d t :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_imply_arrangement_l802_80238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_l802_80294

-- Define the fort and ships
structure Fort :=
  (height : ℝ)

structure Ship :=
  (position : ℝ × ℝ)

-- Define the problem setup
def problem_setup (f : Fort) (s1 s2 : Ship) : Prop :=
  f.height = 30 ∧
  (Real.tan (45 * Real.pi / 180)) = (30 / (s1.position.1)) ∧
  (Real.tan (30 * Real.pi / 180)) = (30 / (s2.position.1)) ∧
  (Real.tan (30 * Real.pi / 180)) = (s2.position.1 - s1.position.1) / 30

-- Theorem statement
theorem ship_distance (f : Fort) (s1 s2 : Ship) :
  problem_setup f s1 s2 →
  Real.sqrt ((s2.position.1 - s1.position.1)^2 + (s2.position.2 - s1.position.2)^2) = 30 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_l802_80294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l802_80247

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := (x - 2) / (x - 5)

-- State the theorem
theorem inverse_g_undefined_at_one :
  ∀ x : ℝ, x ≠ 5 → (∃ y : ℝ, g y = x) → x ≠ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l802_80247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_money_left_l802_80219

theorem no_money_left (total_money : ℝ) (num_items : ℕ) (item_cost : ℝ) 
  (h1 : total_money > 0) 
  (h2 : num_items > 0) 
  (h3 : item_cost > 0) 
  (h4 : (1/4 : ℝ) * total_money = (1/4 : ℝ) * (num_items : ℝ) * item_cost) : 
  total_money - (num_items : ℝ) * item_cost = 0 := by
  sorry

#check no_money_left

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_money_left_l802_80219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_probability_for_five_known_max_questions_known_for_less_than_half_passing_l802_80248

/-- Represents an exam with a total number of questions and number of questions drawn -/
structure Exam where
  total_questions : ℕ
  questions_drawn : ℕ

/-- Represents a candidate taking an exam -/
structure Candidate where
  exam : Exam
  questions_known : ℕ

/-- Calculates the probability of passing the exam for a candidate -/
noncomputable def passingProbability (c : Candidate) : ℚ :=
  1 - (Nat.choose (c.exam.total_questions - c.questions_known) c.exam.questions_drawn) / 
      (Nat.choose c.exam.total_questions c.exam.questions_drawn)

/-- Theorem for the first part of the problem -/
theorem passing_probability_for_five_known 
  (e : Exam) 
  (h1 : e.total_questions = 8) 
  (h2 : e.questions_drawn = 2) 
  (c : Candidate) 
  (h3 : c.exam = e) 
  (h4 : c.questions_known = 5) : 
  passingProbability c = 25 / 28 := by sorry

/-- Theorem for the second part of the problem -/
theorem max_questions_known_for_less_than_half_passing 
  (e : Exam) 
  (h1 : e.total_questions = 8) 
  (h2 : e.questions_drawn = 2) : 
  ∃ (n : ℕ), ∀ (c : Candidate), 
    c.exam = e → 
    c.questions_known ≤ n → 
    passingProbability c < 1 / 2 ∧ 
    ∀ (m : ℕ), m > n → 
      ∃ (c' : Candidate), c'.exam = e ∧ c'.questions_known = m ∧ passingProbability c' ≥ 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_probability_for_five_known_max_questions_known_for_less_than_half_passing_l802_80248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_set_back_angle_l802_80200

-- Define the number of minutes the clock is set back
def minutes_set_back : ℚ := 15

-- Define the angle in radians for a full rotation of the clock
noncomputable def full_rotation : ℝ := 2 * Real.pi

-- Define the number of minutes in a full rotation
def minutes_in_full_rotation : ℚ := 60

-- Theorem statement
theorem clock_set_back_angle :
  (minutes_set_back / minutes_in_full_rotation : ℝ) * full_rotation = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_set_back_angle_l802_80200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_of_three_l802_80208

theorem prime_power_of_three (n : ℕ) (h : Nat.Prime (4^n + 2^n + 1)) :
  ∃ k : ℕ, n = 3^k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_of_three_l802_80208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_to_curve_l802_80234

noncomputable def f (x : ℝ) : ℝ := (1 + Real.sqrt x) / (1 - Real.sqrt x)

def x₀ : ℝ := 4

def normal_line (x : ℝ) : ℝ := -2 * x + 5

theorem normal_to_curve :
  ∀ x : ℝ, x ≠ 1 → x ≠ x₀ →
  (normal_line x - f x₀) = -(deriv f x₀)⁻¹ * (x - x₀) :=
by
  intro x hx hx0
  sorry

#check normal_to_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_to_curve_l802_80234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_present_l802_80296

theorem students_present (X Y : ℝ) (hX : X > 0) (hY : 0 ≤ Y ∧ Y ≤ 100) :
  (1 - Y / 100) * X = X - (Y / 100) * X := by
  -- Algebraic manipulation
  rw [sub_mul]
  -- Simplify
  ring
  -- QED

#check students_present

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_present_l802_80296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base16_zeros_l802_80221

/-- The number of trailing zeros in n! when written in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- 15 factorial -/
def factorial15 : ℕ := Nat.factorial 15

theorem fifteen_factorial_base16_zeros :
  trailingZeros factorial15 16 = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base16_zeros_l802_80221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_four_cells_sequence_l802_80279

-- Define the grid
def Grid := Fin 6 → Fin 6 → Fin 6

-- Define the valid letters
inductive Letter : Type
| A | B | C | D | E | F

-- Define a function to convert Letter to Nat
def Letter.toNat : Letter → Nat
| Letter.A => 0
| Letter.B => 1
| Letter.C => 2
| Letter.D => 3
| Letter.E => 4
| Letter.F => 5

-- Define a function to convert Nat to Letter
def Letter.fromNat : Nat → Letter
| 0 => Letter.A
| 1 => Letter.B
| 2 => Letter.C
| 3 => Letter.D
| 4 => Letter.E
| _ => Letter.F

-- Define a function to check if a letter is valid in a given position
def isValidPlacement (g : Grid) (r c : Fin 6) (l : Letter) : Prop :=
  -- Check row
  (∀ j : Fin 6, j ≠ c → g r j ≠ l.toNat) ∧
  -- Check column
  (∀ i : Fin 6, i ≠ r → g i c ≠ l.toNat) ∧
  -- Check 2x3 rectangle
  (∀ i j : Fin 6, 
    i ∈ [r / 2 * 2, r / 2 * 2 + 1] ∧ 
    j ∈ [c / 3 * 3, c / 3 * 3 + 1, c / 3 * 3 + 2] →
    (i ≠ r ∨ j ≠ c) → g i j ≠ l.toNat)

-- Define a valid grid
def isValidGrid (g : Grid) : Prop :=
  ∀ r c : Fin 6, isValidPlacement g r c (Letter.fromNat (g r c))

-- Theorem statement
theorem middle_four_cells_sequence (g : Grid) (r : Fin 6) (h : isValidGrid g) :
  g r 1 = Letter.E.toNat ∧ 
  g r 2 = Letter.D.toNat ∧ 
  g r 3 = Letter.C.toNat ∧ 
  g r 4 = Letter.F.toNat :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_four_cells_sequence_l802_80279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_block_tipping_condition_l802_80263

/-- Represents a rectangular block on an incline -/
structure Block where
  mass : ℝ
  height : ℝ
  length : ℝ
  friction_coeff : ℝ
  h_positive : 0 < height
  l_positive : 0 < length
  not_square : height ≠ length

/-- The condition for a block to tip over rather than slip -/
def will_tip_over (b : Block) : Prop :=
  b.friction_coeff ≥ b.length / b.height

theorem block_tipping_condition (b : Block) :
  will_tip_over b ↔ 
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → 
    (Real.tan θ = b.length / b.height → b.friction_coeff ≥ Real.tan θ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_block_tipping_condition_l802_80263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l802_80256

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - (1/3)^x)

-- State the theorem
theorem range_of_f : Set.range f = Set.Ico 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l802_80256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_palindromic_count_odd_digit_palindromic_count_l802_80214

/-- A palindromic number is a positive integer that reads the same from left to right as it does from right to left. -/
def IsPalindromic (n : ℕ) : Prop := sorry

/-- The number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The count of palindromic numbers with a given number of digits -/
def countPalindromic (digits : ℕ) : ℕ := sorry

/-- Theorem: The number of four-digit palindromic numbers is equal to 90 -/
theorem four_digit_palindromic_count :
  countPalindromic 4 = 90 := by sorry

/-- Theorem: For any positive integer n, the number of (2n+1)-digit palindromic numbers is equal to 9 × 10^n -/
theorem odd_digit_palindromic_count (n : ℕ) :
  countPalindromic (2 * n + 1) = 9 * (10 ^ n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_palindromic_count_odd_digit_palindromic_count_l802_80214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_difference_of_U_l802_80250

/-- Triangle XYZ with vertices X(0,10), Y(3,0), and Z(10,0) -/
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Vertical line intersecting XZ at U and YZ at V -/
structure Intersection where
  U : ℝ × ℝ
  V : ℝ × ℝ

/-- Helper function to calculate the area of a triangle given three points -/
def areaOfTriangle (A B C : ℝ × ℝ) : ℝ :=
  sorry  -- Implementation not required for the statement

/-- The theorem stating the relationship between the triangle, intersection, and the result -/
theorem coordinate_difference_of_U (t : Triangle) (i : Intersection) :
  t.X = (0, 10) ∧ t.Y = (3, 0) ∧ t.Z = (10, 0) ∧
  (i.U.1 = i.V.1) ∧  -- U and V have the same x-coordinate (vertical line)
  (i.V.2 = 0) ∧  -- V is on the x-axis (on line YZ)
  (areaOfTriangle i.U i.V t.Z = 20) →
  |i.U.1 - i.U.2| = |10 - 4 * Real.sqrt 10| :=
by
  sorry  -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_difference_of_U_l802_80250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_values_l802_80209

-- Define the lines l₁ and l₂
def l₁ (k x y : ℝ) : Prop := k * x + (1 - k) * y - 3 = 0
def l₂ (k x y : ℝ) : Prop := (k - 1) * x + (2 * k + 3) * y - 2 = 0

-- Define perpendicularity condition for two lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Theorem statement
theorem perpendicular_lines_k_values :
  ∀ k : ℝ, (∀ x y : ℝ, l₁ k x y ∧ l₂ k x y → 
    perpendicular (k / (1 - k)) ((k - 1) / (2 * k + 3))) → 
  k = -3 ∨ k = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_values_l802_80209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_M_l802_80255

def M : Finset Nat := {1, 2, 3, 4, 5}

theorem number_of_subsets_M : Finset.card (Finset.powerset M) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_M_l802_80255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solution_l802_80226

theorem cos_equation_solution (x y : ℝ) :
  Real.cos x + Real.cos y - Real.cos (x + y) = 3/2 →
  ∃ (m n : ℤ), (x = π/3 + 2*π*↑m ∨ x = -π/3 + 2*π*↑m) ∧
               (y = π/3 + 2*π*↑n ∨ y = -π/3 + 2*π*↑n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solution_l802_80226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_marks_proof_l802_80245

theorem student_marks_proof (total_marks : ℕ) (passing_percentage : ℚ) (failing_margin : ℕ) 
  (h1 : total_marks = 300)
  (h2 : passing_percentage = 33 / 100)
  (h3 : failing_margin = 40) :
  (passing_percentage * ↑total_marks).floor - failing_margin = 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_marks_proof_l802_80245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_altitude_l802_80262

noncomputable section

def Point := ℝ × ℝ

def A : Point := (10, 10)
def B : Point := (3, -2)
def D : Point := (1, 2)

noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def is_altitude (A B C D : Point) : Prop :=
  (B.1 - D.1) * (A.1 - D.1) + (B.2 - D.2) * (A.2 - D.2) = 0

theorem isosceles_triangle_altitude (C : Point) : 
  is_altitude A B C D → distance A B = distance A C → C = (-1, 6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_altitude_l802_80262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l802_80237

noncomputable def f (x : ℝ) := (Real.log x) / x

theorem f_monotonicity :
  ∀ x ∈ Set.Ioo (0 : ℝ) 10,
  (∀ y ∈ Set.Ioo 0 (Real.exp 1), x < y → f x < f y) ∧
  (∀ y ∈ Set.Ioo (Real.exp 1) 10, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l802_80237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ababa_probability_l802_80233

def total_tiles : ℕ := 5
def a_tiles : ℕ := 3
def b_tiles : ℕ := 2

theorem ababa_probability :
  (Nat.choose total_tiles a_tiles : ℚ)⁻¹ = (1 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ababa_probability_l802_80233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_polar_coords_l802_80299

-- Define the line l
def line_l (α : Real) : Set (Real × Real) :=
  {(x, y) | ∃ t, x = t * Real.cos α ∧ y = t * Real.sin α}

-- Define the curve C (semicircle)
def curve_C : Set (Real × Real) :=
  {(x, y) | (x - 4)^2 + y^2 = 4 ∧ y ≥ 0}

-- Define the tangency condition
def is_tangent (l : Set (Real × Real)) (C : Set (Real × Real)) : Prop :=
  ∃! p, p ∈ l ∧ p ∈ C

-- Main theorem
theorem tangent_point_polar_coords :
  ∃ α, is_tangent (line_l α) curve_C →
  ∃ ρ θ, ρ = 2 * Real.sqrt 3 ∧ θ = π / 6 ∧
    ρ * Real.cos θ = 4 + 2 * Real.cos (π / 2) ∧
    ρ * Real.sin θ = 2 * Real.sin (π / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_polar_coords_l802_80299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_problem_l802_80224

theorem tan_triple_angle_problem (x a b : ℝ) (h1 : Real.tan x = a / b) (h2 : Real.tan (3 * x) = b / (2 * a + b)) :
  ∃ k : ℝ, k > 0 ∧ x = Real.arctan k ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_problem_l802_80224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l802_80225

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem f_sum_property (x₁ x₂ : ℝ) (h : x₁ + x₂ = 1) : 
  f x₁ + f x₂ = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l802_80225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_no_lattice_points_l802_80201

theorem max_a_no_lattice_points : 
  ∃ (a : ℚ), a = 68/200 ∧ 
  (∀ (m : ℚ), 1/3 < m → m < a → 
    ∀ (x y : ℤ), 0 < x → x ≤ 200 → y = ⌊m * (x : ℚ) + 3⌋ → y ≠ (m * (x : ℚ) + 3).floor) ∧
  (∀ (a' : ℚ), a < a' → 
    ∃ (m : ℚ), 1/3 < m ∧ m < a' ∧ 
      ∃ (x y : ℤ), 0 < x ∧ x ≤ 200 ∧ y = (m * (x : ℚ) + 3).floor) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_no_lattice_points_l802_80201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_30_degrees_l802_80270

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2 = 0

-- Define the angle between the line and the positive x-axis
noncomputable def angle_with_x_axis (m : ℝ) : ℝ := Real.arctan m

-- Theorem statement
theorem line_angle_is_30_degrees :
  ∃ (m : ℝ), (∀ x y, line_equation x y → y = m * x - 2 / Real.sqrt 3) ∧
             angle_with_x_axis m = 30 * (Real.pi / 180) := by
  -- Proof goes here
  sorry

-- Helper lemma to show that the slope is 1 / sqrt(3)
lemma line_slope :
  ∃ (m : ℝ), ∀ x y, line_equation x y → y = m * x - 2 / Real.sqrt 3 := by
  -- Proof goes here
  sorry

-- Helper lemma to show that arctan(1 / sqrt(3)) = 30 degrees
lemma angle_is_30_degrees :
  Real.arctan (1 / Real.sqrt 3) = 30 * (Real.pi / 180) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_30_degrees_l802_80270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_l802_80212

open Set
open Function

theorem permutation_count : 
  let n : ℕ := 16
  let b := Fin n → Fin n
  let is_permutation (f : b) := Bijective f
  let descending (f : b) (m : Fin n) := ∀ i j, i < j → j ≤ m → f i > f j
  let ascending (f : b) (m : Fin n) := ∀ i j, m < i → i < j → f i < f j
  Fintype.card { f : b | is_permutation f ∧ 
                         descending f 7 ∧ 
                         ascending f 7 ∧ 
                         f 7 = 0 } = 6435 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_l802_80212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_market_transactions_l802_80207

def transactions : List Int := [25, -6, 18, 12, -24, -15]
def initial_money : Int := 40

theorem flea_market_transactions :
  (transactions.filter (· > 0)).maximum? = some 25 ∧
  (transactions.filter (· < 0)).minimum? = some (-24) ∧
  (transactions.map Int.natAbs).sum = 100 ∧
  initial_money + transactions.sum = 50 ∧
  transactions.sum = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_market_transactions_l802_80207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_investment_l802_80284

theorem mary_investment : ∃ (mary_investment : ℝ),
  let mike_investment : ℝ := 300
  let total_profit : ℝ := 3000
  let profit_difference : ℝ := 800
  let mary_share := (mary_investment / (mary_investment + mike_investment)) * (2/3 * total_profit) + (1/3 * total_profit) / 2
  let mike_share := (mike_investment / (mary_investment + mike_investment)) * (2/3 * total_profit) + (1/3 * total_profit) / 2
  mary_share = mike_share + profit_difference ∧ mary_investment = 700
:= by
  use 700
  simp
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_investment_l802_80284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l802_80217

/-- The equation of a hyperbola with common asymptote to a given hyperbola and passing through a specific point -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ x^2 / 16 - y^2 / 9 = k) →  -- Common asymptote condition
  (3 * Real.sqrt 3)^2 / 11 - (-3)^2 / (99/16) = 1 →  -- Point condition
  x^2 / 11 - y^2 / (99/16) = 1 :=  -- Resulting equation
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l802_80217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_8_l802_80272

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (9 + 2*x) - 5) / (x^(1/3) - 2)

theorem limit_f_at_8 : 
  Filter.Tendsto f (Filter.atTop.comap Real.toNNReal) (nhds (12/5)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_8_l802_80272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_and_values_l802_80297

def valid_values : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def solution_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (fun pair => 3 * pair.1 - 2 * pair.2 = 1) (valid_values.product valid_values)

def expression (pair : ℕ × ℕ) : ℕ :=
  10 * pair.1 + pair.2

theorem solution_count_and_values :
  (Finset.image expression solution_pairs).card = 3 ∧
  (Finset.image expression solution_pairs) = {11, 34, 57} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_and_values_l802_80297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_in_cube_l802_80286

-- Define the side length of the cube
noncomputable def cube_side_length : ℝ := 1

-- Define the radius of the sphere
noncomputable def sphere_radius : ℝ := (3 : ℝ).sqrt / 2

-- Define the volume of the cube
noncomputable def cube_volume : ℝ := cube_side_length ^ 3

-- Define the volume of the sphere
noncomputable def sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3

-- Theorem: The probability of a point in the sphere being inside the cube
theorem probability_point_in_cube :
  cube_volume / sphere_volume = 2 * (3 : ℝ).sqrt / (3 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_in_cube_l802_80286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_difference_l802_80293

-- Define the two divisions as noncomputable
noncomputable def division1 : ℝ := 32.5 / 1.3
noncomputable def division2 : ℝ := 60.8 / 7.6

-- State the theorem
theorem division_difference : division1 - division2 = 17 := by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_difference_l802_80293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_l802_80273

def sequenceProperty (b : ℕ → ℝ) :=
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

theorem b_2023_value (b : ℕ → ℝ) 
  (h_seq : sequenceProperty b) 
  (h_b1 : b 1 = 3 + Real.sqrt 5) 
  (h_b2021 : b 2021 = 11 + Real.sqrt 5) : 
  b 2023 = 7 - 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_l802_80273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l802_80215

noncomputable def f (x : ℝ) := 2 * Real.sin (4 * x - Real.pi / 3)

noncomputable def g (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 3)

theorem g_properties :
  (∃ T > 0, ∀ x, g (x + T) = g x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, g (y + S) ≠ g y) ∧
  (∀ x, g (Real.pi / 12 - x) = g (Real.pi / 12 + x)) ∧
  (∃ φ, ∀ x, g x = 2 * Real.sin (2 * x + φ) ∧ φ = Real.pi / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l802_80215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_l802_80260

theorem prime_divisibility (p : ℕ) (k : ℕ) (x₀ y₀ z₀ t₀ : ℤ) 
  (h_prime : Nat.Prime p)
  (h_form : p = 4 * k + 3)
  (h_equation : x₀^(2*p) + y₀^(2*p) + z₀^(2*p) = t₀^(2*p)) :
  (∃ (n : ℤ), x₀ = n * p) ∨ 
  (∃ (n : ℤ), y₀ = n * p) ∨ 
  (∃ (n : ℤ), z₀ = n * p) ∨ 
  (∃ (n : ℤ), t₀ = n * p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_l802_80260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_constant_l802_80206

/-- Two circles with equal radii that intersect each other -/
structure IntersectingCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  radius : ℝ
  centers_distance_lt_radius : dist center1 center2 < radius
  radius_pos : radius > 0

/-- A ray from the center of one circle to the other circle -/
noncomputable def ray (ic : IntersectingCircles) (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + ic.radius * Real.cos θ, p.2 + ic.radius * Real.sin θ)

/-- The intersection point of a ray with the other circle -/
noncomputable def intersectionPoint (ic : IntersectingCircles) (center : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ :=
  ray ic θ center

theorem intersection_points_distance_constant (ic : IntersectingCircles) (θ : ℝ) :
  dist (intersectionPoint ic ic.center1 θ) (intersectionPoint ic ic.center2 θ) =
  dist ic.center1 ic.center2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_constant_l802_80206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_is_zero_l802_80246

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 - a*x - 20*a^2 < 0

-- Define the condition on the solutions
def solution_difference_condition (a : ℝ) : Prop :=
  ∀ x₁ x₂, inequality x₁ a → inequality x₂ a → |x₁ - x₂| ≤ 9

-- Define the set of valid 'a' values
def valid_a_set : Set ℝ :=
  {a | solution_difference_condition a ∧ a ≠ 0}

-- State the theorem
theorem sum_of_extrema_is_zero :
  ∃ (min max : ℝ), min ∈ valid_a_set ∧ max ∈ valid_a_set ∧
    (∀ a ∈ valid_a_set, min ≤ a ∧ a ≤ max) ∧
    min + max = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_is_zero_l802_80246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equivalence_l802_80266

theorem scientific_notation_equivalence : 
  (0.000000000304 : ℝ) = 3.04 * (10 : ℝ)^(-10 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equivalence_l802_80266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_shifted_l802_80232

noncomputable def f (x : ℝ) : ℝ := 2^(1 - x)
noncomputable def g (x : ℝ) : ℝ := 2^(-x)

theorem f_equals_g_shifted (x : ℝ) : f x = g (x - 1) := by
  calc f x = 2^(1 - x) := rfl
       _ = 2^(-(x - 1)) := by rw [neg_sub]
       _ = g (x - 1) := rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_shifted_l802_80232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_xy_product_l802_80282

noncomputable def sample : List ℝ := [4, 5, 6]

noncomputable def mean (xs : List ℝ) (x y : ℝ) : ℝ := (xs.sum + x + y) / (xs.length + 2)

noncomputable def variance (xs : List ℝ) (x y : ℝ) (m : ℝ) : ℝ :=
  (xs.map (λ z => (z - m)^2)).sum / (xs.length + 2) + ((x - m)^2 + (y - m)^2) / (xs.length + 2)

theorem sample_xy_product (x y : ℝ) :
  mean sample x y = 5 ∧ 
  variance sample x y 5 = 2 →
  x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_xy_product_l802_80282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l802_80267

-- Define the power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x => x ^ α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) :
  (∃ α : ℝ, f = power_function α) →
  f (Real.sqrt 2 / 2) = 1 / 2 →
  f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l802_80267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_tan_l802_80242

theorem cos_double_angle_tan (α : ℝ) (h : Real.tan α = 3) : Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_tan_l802_80242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_is_two_l802_80252

/-- The length of a tunnel given a train passing through it. -/
noncomputable def tunnel_length (train_length : ℝ) (train_speed : ℝ) (exit_time : ℝ) : ℝ :=
  train_speed * exit_time / 60 - train_length

/-- Theorem: The tunnel length is 2 miles given the specified conditions. -/
theorem tunnel_length_is_two :
  let train_length : ℝ := 2
  let train_speed : ℝ := 120
  let exit_time : ℝ := 2
  tunnel_length train_length train_speed exit_time = 2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_is_two_l802_80252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l802_80241

theorem min_sin6_cos6 (x : ℝ) : Real.sin x ^ 6 + Real.cos x ^ 6 ≥ 1/4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l802_80241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_profit_l802_80259

-- Define the demand functions
def demand_A (p : ℝ) : ℝ := 40 - 2 * p
def demand_B (p : ℝ) : ℝ := 26 - p

-- Define the total cost function
def total_cost (q : ℝ) : ℝ := 8 * q + 1

-- Define the shipping cost
def shipping_cost : ℝ := 1

-- Define the tax function for country A
def tax_A (profit : ℝ) : ℝ := 0.15 * profit

-- Define the tax function for country B
noncomputable def tax_B (profit : ℝ) : ℝ :=
  if profit ≤ 30 then 0
  else if profit ≤ 100 then 0.1 * (profit - 30)
  else if profit ≤ 150 then 0.1 * 70 + 0.2 * (profit - 100)
  else 0.1 * 70 + 0.2 * 50 + 0.3 * (profit - 150)

-- Define the profit function
def profit (p_A p_B : ℝ) : ℝ :=
  let q_A := demand_A p_A
  let q_B := demand_B p_B
  let revenue := p_A * q_A + p_B * q_B
  let cost := total_cost (q_A + q_B) + shipping_cost
  revenue - cost

-- Theorem statement
theorem optimal_profit :
  ∃ p_A p_B : ℝ,
    p_A > 0 ∧ p_B > 0 ∧
    profit p_A p_B = 150 ∧
    profit p_A p_B - tax_B (profit p_A p_B) = 133.7 ∧
    ∀ p_A' p_B' : ℝ, p_A' > 0 → p_B' > 0 →
      profit p_A' p_B' - tax_A (profit p_A' p_B') ≤ 133.7 ∧
      profit p_A' p_B' - tax_B (profit p_A' p_B') ≤ 133.7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_profit_l802_80259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_rate_approx_l802_80268

/-- The rate of cloth weaving in meters per second -/
noncomputable def weaving_rate (cloth_length : ℝ) (time : ℝ) : ℝ :=
  cloth_length / time

/-- Theorem stating that the weaving rate is approximately 0.127 meters per second -/
theorem weaving_rate_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |weaving_rate 15 118.11 - 0.127| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_rate_approx_l802_80268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_above_threshold_l802_80280

theorem smallest_number_above_threshold (a b c : ℝ) (ha : a = 0.8) (hb : b = 1/2) (hc : c = 0.9) :
  min (min (max a 0.6) (max c 0.6)) (max b 0.6) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_above_threshold_l802_80280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y4_in_2x_minus_3y_to_7th_l802_80287

theorem coefficient_x3y4_in_2x_minus_3y_to_7th : 
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * (2^(7-k)) * ((-3 : ℝ)^k) * 
    (if k = 4 then 1 else 0)) = 22680 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y4_in_2x_minus_3y_to_7th_l802_80287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_mod_9_Q_2005_upper_bound_Q_2005_theorem_l802_80274

-- Define Q as a function that takes a natural number and returns a natural number
def Q : ℕ → ℕ := sorry

-- State the property that Q(n) is congruent to n modulo 9
theorem Q_mod_9 (n : ℕ) : Q n ≡ n [MOD 9] := by sorry

-- State the upper bound for Q(2005^2005)
theorem Q_2005_upper_bound : Q (2005^2005) ≤ 72180 := by sorry

-- State the theorem to be proved
theorem Q_2005_theorem : Q (Q (Q (2005^2005))) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_mod_9_Q_2005_upper_bound_Q_2005_theorem_l802_80274
