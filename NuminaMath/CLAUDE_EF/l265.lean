import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contractor_fired_two_workers_l265_26558

/-- Represents the work completion rate per day for a given number of workers -/
def work_rate (workers : ℕ) (days : ℕ) (work_fraction : ℚ) : ℚ :=
  work_fraction / days

/-- Calculates the number of workers needed to complete a given fraction of work in a given number of days -/
def workers_needed (work_fraction : ℚ) (days : ℕ) (rate : ℚ) : ℚ :=
  (work_fraction / days) / rate

/-- Calculates the number of workers fired by the contractor -/
def contractor_fired_workers (initial_workers : ℕ) (total_days : ℕ) (work_done_days : ℕ) 
    (work_done_fraction : ℚ) (remaining_days : ℕ) : ℕ :=
  let initial_rate := work_rate initial_workers work_done_days work_done_fraction
  let remaining_fraction := 1 - work_done_fraction
  let required_workers := workers_needed remaining_fraction remaining_days initial_rate
  (initial_workers : ℤ) - (required_workers.ceil : ℤ) |>.toNat

/-- Proves that the contractor fired 2 workers -/
theorem contractor_fired_two_workers : 
  contractor_fired_workers 10 100 20 (1/4) 75 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contractor_fired_two_workers_l265_26558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l265_26573

open Real

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the conditions
def conditions (A B C : ℝ) : Prop :=
  triangle_ABC A B C ∧ 
  Real.sin A = Real.sin B ∧ 
  Real.sin A = -Real.cos C

-- Define the median AM
def median_AM (AM : ℝ) : Prop :=
  AM = Real.sqrt 7

-- Define the area of the triangle
noncomputable def area_triangle (A B C : ℝ) : ℝ :=
  2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)

-- State the theorem
theorem triangle_properties 
  (A B C AM : ℝ) 
  (h1 : conditions A B C) 
  (h2 : median_AM AM) : 
  A = Real.pi/6 ∧ B = Real.pi/6 ∧ C = 2*Real.pi/3 ∧ 
  area_triangle A B C = Real.sqrt 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l265_26573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l265_26561

/-- Given a train with speed excluding stoppages and stopping time per hour,
    calculate the speed including stoppages -/
noncomputable def train_speed_with_stops (speed_without_stops : ℝ) (stop_time : ℝ) : ℝ :=
  let running_time := (60 - stop_time) / 60
  speed_without_stops * running_time

/-- Theorem stating that the train speed with stops is approximately 31 kmph -/
theorem train_speed_theorem (speed_without_stops : ℝ) (stop_time : ℝ) 
  (h1 : speed_without_stops = 45)
  (h2 : stop_time = 18.67) :
  ∃ ε > 0, |train_speed_with_stops speed_without_stops stop_time - 31| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l265_26561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_left_handed_fraction_l265_26531

theorem football_team_left_handed_fraction :
  ∀ (total_players throwers right_handed : ℕ),
    total_players = 70 →
    throwers = 34 →
    right_handed = 58 →
    let non_throwers := total_players - throwers
    let left_handed_non_throwers := non_throwers - (right_handed - throwers)
    (left_handed_non_throwers : ℚ) / non_throwers = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_left_handed_fraction_l265_26531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_caesar_cipher_decryption_l265_26577

/-- Represents the Russian alphabet --/
def RussianAlphabet := Fin 33

/-- Represents a message with text and numbers --/
structure Message where
  text : List Char
  numbers : List Int

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : Int) : Int :=
  sorry

/-- Applies Caesar cipher shift to a letter --/
def shiftLetter (letter : Char) (shift : Int) : Char :=
  sorry

/-- Decrypts a message using Caesar cipher --/
def decryptMessage (msg : Message) (shift : Int) : Message :=
  sorry

theorem caesar_cipher_decryption 
  (msg : Message) 
  (shift : Int) 
  (h1 : msg.numbers.length ≥ 2)
  (h2 : toBase10 (msg.numbers.head!) + toBase10 (msg.numbers.tail!.head!) = 22) :
  ∃ (effective_shift : Int), 
    effective_shift = 11 ∧ 
    decryptMessage msg effective_shift = decryptMessage msg shift :=
by sorry

#check caesar_cipher_decryption

end NUMINAMATH_CALUDE_ERRORFEEDBACK_caesar_cipher_decryption_l265_26577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_identity_l265_26529

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1/2, -Real.sqrt 3/2; Real.sqrt 3/2, 1/2]

theorem smallest_power_identity (n : ℕ) : 
  (n > 0 ∧ A^n = 1 ∧ ∀ m : ℕ, m > 0 → m < n → A^m ≠ 1) ↔ n = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_identity_l265_26529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_age_after_person_leaves_l265_26541

theorem new_average_age_after_person_leaves (initial_people : ℕ) (initial_avg_age : ℚ) 
  (leaving_person_age : ℕ) (remaining_people : ℕ) :
  initial_people = 8 →
  initial_avg_age = 28 →
  leaving_person_age = 24 →
  remaining_people = 7 →
  (initial_people * initial_avg_age - leaving_person_age) / remaining_people = 28 + 4/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_age_after_person_leaves_l265_26541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l265_26557

/-- Quadratic function type -/
def QuadraticFunction (a : ℝ) : ℝ → ℝ := λ x => (a * x - 1) * (x - a)

theorem quadratic_function_properties (a : ℝ) (h : a ≠ 0) :
  let f := QuadraticFunction a
  -- The point (1/2, 5) is not on the graph of the function
  (f (1/2) ≠ 5) ∧
  -- The intersection point of the graph with the y-axis is above the origin
  (f 0 > 0) ∧
  -- The function has a minimum value, and the minimum value is not greater than zero
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min ≤ 0) ∧
  -- The axis of symmetry of the graph is not always to the left of the y-axis
  (¬∀ (a : ℝ), a ≠ 0 → (a^2 + 1) / (2 * a) < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l265_26557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_PUQV_l265_26563

noncomputable section

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

-- Define points U and V
def PointOnAB (U : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ U = (t, 0)
def PointOnCD (V : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ V = (1, t)

-- Define intersection points P and Q
def IntersectionP (P U V : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ, P = (t * U.1, s * V.2) ∧ t + s = 1

def IntersectionQ (Q U V : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ, Q = (1 - t * (1 - U.1), 1 - s * (1 - V.2)) ∧ t + s = 1

-- Define the area of quadrilateral PUQV
def AreaPUQV (P U Q V : ℝ × ℝ) : ℝ :=
  abs ((P.1 - U.1) * (Q.2 - V.2) - (Q.1 - V.1) * (P.2 - U.2)) / 2

-- Theorem statement
theorem max_area_PUQV 
  (A B C D U V P Q : ℝ × ℝ) 
  (h_square : Square A B C D)
  (h_U : PointOnAB U)
  (h_V : PointOnCD V)
  (h_P : IntersectionP P U V)
  (h_Q : IntersectionQ Q U V) :
  ∀ U' V' P' Q', 
    PointOnAB U' → PointOnCD V' → 
    IntersectionP P' U' V' → IntersectionQ Q' U' V' →
    AreaPUQV P U Q V ≤ 1/4 ∧ 
    (∃ U₀ V₀ P₀ Q₀, AreaPUQV P₀ U₀ Q₀ V₀ = 1/4) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_PUQV_l265_26563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_around_square_l265_26502

/-- Represents a square in 2D space -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Represents the configuration of squares around the initial square -/
structure SquareConfiguration where
  initial_square : Square
  surrounding_squares : List Square

/-- The maximum number of non-overlapping squares that can be placed around a square -/
def max_surrounding_squares : ℕ := 8

/-- The initial square ABCD -/
def initial_square : Square := { center := (0, 0), side_length := 2 }

/-- The larger square A₁B₁C₁D₁ -/
def larger_square : Square := { center := (0, 0), side_length := 4 }

/-- Predicate to check if two squares are non-overlapping -/
def non_overlapping (s1 s2 : Square) : Prop := sorry

/-- Predicate to check if a square is placed around the initial square -/
def is_surrounding (s : Square) : Prop := sorry

/-- The main theorem stating that no more than 8 non-overlapping squares can be placed around the initial square -/
theorem max_squares_around_square (config : SquareConfiguration) :
  config.initial_square = initial_square →
  (∀ s, s ∈ config.surrounding_squares → is_surrounding s) →
  (∀ s1 s2, s1 ∈ config.surrounding_squares → s2 ∈ config.surrounding_squares → s1 ≠ s2 → non_overlapping s1 s2) →
  config.surrounding_squares.length ≤ max_surrounding_squares := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_around_square_l265_26502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l265_26544

theorem problem_statement (x y z : ℝ) : 
  (x + y + z = 1 → Real.sqrt (3*x + 1) + Real.sqrt (3*y + 2) + Real.sqrt (3*z + 3) ≤ 3 * Real.sqrt 3) ∧ 
  (x + 2*y + 3*z = 6 → 
    (∀ a b c : ℝ, a + 2*b + 3*c = 6 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧ 
    (∃ x₀ y₀ z₀ : ℝ, x₀ + 2*y₀ + 3*z₀ = 6 ∧ x₀^2 + y₀^2 + z₀^2 = 18/7)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l265_26544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourteenth_term_is_nine_l265_26571

/-- Definition of the sequence -/
noncomputable def a (n : ℕ) : ℝ := Real.sqrt (3 * (2 * n - 1))

/-- The theorem stating that the 14th term of the sequence equals 9 -/
theorem fourteenth_term_is_nine : a 14 = 9 := by
  -- Expand the definition of a
  unfold a
  -- Simplify the expression inside the square root
  simp [Real.sqrt_eq_iff_sq_eq]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourteenth_term_is_nine_l265_26571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l265_26554

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 : ℝ)^x + 1 else x^2 + a*x

-- State the theorem
theorem find_a_value (a : ℝ) : f a (f a 0) = 6 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l265_26554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_quadratic_l265_26524

/-- A function y is quadratic if it can be written in the form y = ax^2 + bx + c, where a ≠ 0 -/
def is_quadratic (y : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, y x = a * x^2 + b * x + c

/-- The function y defined in terms of m -/
noncomputable def y (m : ℝ) : ℝ → ℝ := λ x => (m + 2) * (x ^ (m^2 - 2))

/-- Theorem stating that m = 2 is the only value for which y is a quadratic function -/
theorem unique_m_for_quadratic : 
  (∃! m : ℝ, is_quadratic (y m)) ∧ (∀ m : ℝ, is_quadratic (y m) → m = 2) := by
  sorry

#check unique_m_for_quadratic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_quadratic_l265_26524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_property_l265_26540

-- Define a convex polygon
def ConvexPolygon (n : ℕ) := Fin n → ℝ × ℝ

-- Define a point inside the polygon
def InsidePolygon (p : ℝ × ℝ) (polygon : ConvexPolygon n) : Prop := sorry

-- Collinearity of three points
def CollinearPoints (p q r : ℝ × ℝ) : Prop := sorry

-- Define the property that for any line from O to Aᵢ, there's another vertex Aⱼ on this line
def HasPropertyO (O : ℝ × ℝ) (polygon : ConvexPolygon n) : Prop :=
  ∀ i : Fin n, ∃ j : Fin n, i ≠ j ∧ CollinearPoints O (polygon i) (polygon j)

theorem unique_point_property {n : ℕ} (polygon : ConvexPolygon n) :
  ∃! O : ℝ × ℝ, InsidePolygon O polygon ∧ HasPropertyO O polygon := by
  sorry

#check unique_point_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_property_l265_26540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_asymptote_distance_l265_26533

/-- The distance from the foci of the hyperbola x²/2 - y² = -1 to its asymptotes -/
theorem hyperbola_foci_asymptote_distance : 
  let h : ℝ → ℝ → ℝ := fun x y => x^2/2 - y^2 + 1
  let a : ℝ := Real.sqrt 2
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 3
  let foci : Set (ℝ × ℝ) := {(0, c), (0, -c)}
  let asymptotes : Set (ℝ → ℝ) := {fun x => x/Real.sqrt 2, fun x => -x/Real.sqrt 2}
  ∀ (f : ℝ × ℝ) (g : ℝ → ℝ), f ∈ foci → g ∈ asymptotes → 
    ∃ (d : ℝ), d = Real.sqrt 2 ∧ 
      d = abs (g f.1 - f.2) / Real.sqrt (1 + (g 1 - g 0)^2)
:= by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_asymptote_distance_l265_26533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_price_for_revenue_increase_l265_26545

/-- Represents the electricity pricing problem -/
structure ElectricityPricing where
  a : ℝ  -- Last year's electricity consumption
  k : ℝ  -- Proportionality constant
  x : ℝ  -- This year's electricity price

/-- Revenue function for the power department -/
noncomputable def revenue (ep : ElectricityPricing) : ℝ :=
  (ep.k / (ep.x - 0.4) + ep.a) * (ep.x - 0.3)

/-- Theorem stating the lowest price that ensures at least 20% revenue increase -/
theorem lowest_price_for_revenue_increase (ep : ElectricityPricing) 
  (h1 : ep.k = 0.2 * ep.a)
  (h2 : 0.55 ≤ ep.x ∧ ep.x ≤ 0.75) :
  (revenue ep ≥ 1.2 * (ep.a * (0.8 - 0.3))) → ep.x ≥ 0.6 := by
  sorry

#check lowest_price_for_revenue_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_price_for_revenue_increase_l265_26545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l265_26593

noncomputable section

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 -/
def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  (|a * x₀ + b * y₀ + c|) / Real.sqrt (a^2 + b^2)

/-- The line equation √3x + √3y = 4 -/
def line_equation (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + Real.sqrt 3 * y = 4

/-- The circle equation x² + y² = 4 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem line_intersects_circle :
  ∃ x y : ℝ, line_equation x y ∧ circle_equation x y := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l265_26593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_EFGH_area_perimeter_product_l265_26506

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Define the square EFGH -/
def E : Point := ⟨1, 5⟩
def F : Point := ⟨5, 6⟩
def G : Point := ⟨6, 2⟩
def H : Point := ⟨2, 1⟩

/-- The side length of the square -/
noncomputable def sideLength : ℝ := distance E F

/-- The area of the square -/
noncomputable def area : ℝ := sideLength ^ 2

/-- The perimeter of the square -/
noncomputable def perimeter : ℝ := 4 * sideLength

theorem square_EFGH_area_perimeter_product :
  area * perimeter = 68 * Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_EFGH_area_perimeter_product_l265_26506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_topic_unchosen_l265_26567

/-- The number of finalists and topics -/
def n : ℕ := 4

/-- The total number of possible scenarios -/
def total_scenarios : ℕ := n^n

/-- The number of ways to choose topics leaving exactly one unchosen -/
def scenarios_with_one_unchosen : ℕ := total_scenarios - 
  (n.choose 2 * (n.choose 3 * 2 * 2 + n.choose 2) + n + Nat.factorial n)

/-- Theorem stating that the number of scenarios where exactly one topic 
    is not chosen is equal to 144 -/
theorem exactly_one_topic_unchosen : scenarios_with_one_unchosen = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_topic_unchosen_l265_26567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electronics_store_prices_l265_26500

/-- Calculate the final selling price of an item given its cost price, mark-up rate, and discount rate -/
def finalSellingPrice (costPrice markupRate discountRate : ℚ) : ℚ :=
  costPrice * (1 + markupRate) * (1 - discountRate)

/-- Theorem stating the final selling prices of three items in an electronics store -/
theorem electronics_store_prices :
  let speakerPrice := finalSellingPrice 160 (20/100) (10/100)
  let keyboardPrice := finalSellingPrice 220 (30/100) (15/100)
  let mousePrice := finalSellingPrice 120 (40/100) (25/100)
  speakerPrice = 172.8 ∧ keyboardPrice = 243.1 ∧ mousePrice = 126 := by
  sorry

#eval finalSellingPrice 160 (20/100) (10/100)
#eval finalSellingPrice 220 (30/100) (15/100)
#eval finalSellingPrice 120 (40/100) (25/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_electronics_store_prices_l265_26500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l265_26535

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := f x / (a * (1 - x))

-- Statement for the tangent line
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x, m * x + b = 2 * x - 2 ∧
  (∀ ε > 0, ∃ δ > 0, ∀ h, |h| < δ → |f (1 + h) - (f 1 + m * h)| ≤ ε * |h|) :=
sorry

-- Statement for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x, x ∈ Set.Ioo 0 1 → g a x < -2) ↔ a ∈ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l265_26535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_by_18_24_meters_l265_26515

/-- The distance A beats B in a race, given their speeds over a fixed distance --/
noncomputable def distanceABeatsB (distance : ℝ) (timeA timeB : ℝ) : ℝ :=
  let speedA := distance / timeA
  let speedB := distance / timeB
  speedA * timeB - distance

/-- Theorem stating that A beats B by approximately 18.24 meters --/
theorem a_beats_b_by_18_24_meters :
  let distance := 128
  let timeA := 28
  let timeB := 32
  ∃ ε > 0, abs (distanceABeatsB distance timeA timeB - 18.24) < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distanceABeatsB 128 28 32

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_by_18_24_meters_l265_26515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_equals_two_implies_x_is_e_l265_26568

/-- Given f(x) = x ln x, prove that if f'(x₀) = 2, then x₀ = e -/
theorem derivative_equals_two_implies_x_is_e (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = x * Real.log x) →
  (deriv f) x₀ = 2 →
  x₀ = Real.exp 1 :=
by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_equals_two_implies_x_is_e_l265_26568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_96_l265_26547

/-- Represents a geometric sequence -/
structure GeometricSequence where
  first_term : ℚ
  second_term : ℚ

/-- The sixth term of a geometric sequence -/
def sixth_term (seq : GeometricSequence) : ℚ :=
  seq.first_term * (seq.second_term / seq.first_term)^5

/-- Theorem: The sixth term of the specific geometric sequence is 96 -/
theorem sixth_term_is_96 :
  let seq := GeometricSequence.mk 3 6
  sixth_term seq = 96 := by
  sorry

#eval sixth_term (GeometricSequence.mk 3 6)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_96_l265_26547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_sum_l265_26517

theorem max_value_trig_sum :
  ∀ θ₁ θ₂ θ₃ θ₄ : ℝ,
  Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₁ ≤ 2 ∧
  ∃ φ₁ φ₂ φ₃ φ₄ : ℝ, Real.cos φ₁ * Real.sin φ₂ + Real.cos φ₂ * Real.sin φ₃ + Real.cos φ₃ * Real.sin φ₄ + Real.cos φ₄ * Real.sin φ₁ = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_sum_l265_26517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_sqrt_845_l265_26519

/-- The area of a rhombus given its vertices in a rectangular coordinate system. -/
noncomputable def rhombusArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let d1 := (v3.1 - v1.1, v3.2 - v1.2)
  let d2 := (v4.1 - v2.1, v4.2 - v2.2)
  let length1 := Real.sqrt (d1.1^2 + d1.2^2)
  let length2 := Real.sqrt (d2.1^2 + d2.2^2)
  (1/2) * length1 * length2

/-- Theorem stating that the area of the given rhombus is √845. -/
theorem rhombus_area_is_sqrt_845 :
  rhombusArea (2, 4.5) (11, 7) (4, 1.5) (-5, 5) = Real.sqrt 845 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_sqrt_845_l265_26519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hit_at_least_once_both_miss_mutually_exclusive_l265_26527

-- Define the sample space
structure SampleSpace where
  hit1 : Bool
  hit2 : Bool

-- Define the event of hitting the target at least once
def HitAtLeastOnce (ω : SampleSpace) : Prop :=
  ω.hit1 ∨ ω.hit2

-- Define the event of both shots missing
def BothMiss (ω : SampleSpace) : Prop :=
  ¬ω.hit1 ∧ ¬ω.hit2

-- Theorem stating that HitAtLeastOnce and BothMiss are mutually exclusive
theorem hit_at_least_once_both_miss_mutually_exclusive :
  ∀ ω : SampleSpace, ¬(HitAtLeastOnce ω ∧ BothMiss ω) :=
by
  intro ω
  intro h
  cases h with
  | intro h1 h2 =>
    cases h1 with
    | inl h1_left => exact h2.left h1_left
    | inr h1_right => exact h2.right h1_right

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hit_at_least_once_both_miss_mutually_exclusive_l265_26527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_cos_theta_l265_26523

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * Real.cos x

theorem max_value_cos_theta : 
  ∃ θ : ℝ, (∀ x : ℝ, f x ≤ f θ) → Real.cos θ = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_cos_theta_l265_26523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l265_26526

-- Define the variables and constants
variable (d x : ℝ)
variable (a b c : ℝ)

-- State the theorem
theorem relationship_abc (h1 : 1 < x) (h2 : x < d) 
  (ha : a = (Real.log x / Real.log d)^2)
  (hb : b = Real.log (x^2) / Real.log d)
  (hc : c = Real.log (Real.log x / Real.log d) / Real.log d) :
  c < a ∧ a < b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l265_26526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_theorem_l265_26559

-- Define the volume of the wire in cubic centimeters
noncomputable def wire_volume : ℝ := 11

-- Define the diameter of the wire in millimeters
noncomputable def wire_diameter_mm : ℝ := 1

-- Define the conversion factor from millimeters to meters
noncomputable def mm_to_m : ℝ := 1 / 1000

-- Define pi as a constant
noncomputable def π : ℝ := Real.pi

-- Theorem statement
theorem wire_length_theorem :
  let diameter_m := wire_diameter_mm * mm_to_m
  let radius_m := diameter_m / 2
  let length_m := wire_volume * (1000000 * mm_to_m^3) / (π * radius_m^2)
  ∃ ε > 0, abs (length_m - 14.0064) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_theorem_l265_26559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juliet_supporter_in_capulet_probability_l265_26504

/-- Represents the total population of Venezia -/
noncomputable def total_population : ℝ := 1

/-- Fraction of population living in Montague province -/
noncomputable def montague_fraction : ℝ := 5/8

/-- Fraction of population living in Capulet province -/
noncomputable def capulet_fraction : ℝ := 3/8

/-- Fraction of Montague residents supporting Juliet -/
noncomputable def montague_juliet_support : ℝ := 1/5

/-- Fraction of Capulet residents supporting Juliet -/
noncomputable def capulet_juliet_support : ℝ := 7/10

/-- The probability that a randomly chosen Juliet supporter resides in Capulet -/
theorem juliet_supporter_in_capulet_probability :
  (capulet_fraction * capulet_juliet_support * total_population) /
  ((montague_fraction * montague_juliet_support * total_population) +
   (capulet_fraction * capulet_juliet_support * total_population)) = 21/31 := by
  sorry

#eval (21 : ℚ) / 31

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juliet_supporter_in_capulet_probability_l265_26504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l265_26508

theorem triangle_problem (a b c A : ℝ) :
  Real.cos A = 3/5 →
  b * c = 5 →
  (1/2 : ℝ) * b * c * Real.sin A = 2 ∧
  (b + c = 6 → a = 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l265_26508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potassium_acetate_hydrogen_atoms_l265_26595

-- Define the chemical formula of potassium acetate
def potassium_acetate : String := "CH₃COOK"

-- Define Avogadro's constant
def avogadro_constant : ℝ := 6.02e23

-- Count the number of hydrogen atoms in the chemical formula
def hydrogen_count (formula : String) : ℕ := 
  formula.count 'H'

-- Theorem statement
theorem potassium_acetate_hydrogen_atoms :
  (hydrogen_count potassium_acetate : ℝ) * avogadro_constant = 3 * avogadro_constant := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_potassium_acetate_hydrogen_atoms_l265_26595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_eq_one_div_390_l265_26549

/-- The sum of 1 / (3^a * 2^b * 7^c) for all positive integers a, b, c where 1 ≤ a < b < c -/
noncomputable def tripleSum : ℝ :=
  ∑' (a : ℕ), ∑' (b : ℕ), ∑' (c : ℕ),
    if 1 ≤ a ∧ a < b ∧ b < c then
      (1 : ℝ) / (3 ^ a * 2 ^ b * 7 ^ c)
    else
      0

/-- The sum of 1 / (3^a * 2^b * 7^c) for all positive integers a, b, c where 1 ≤ a < b < c equals 1/390 -/
theorem tripleSum_eq_one_div_390 : tripleSum = 1 / 390 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_eq_one_div_390_l265_26549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_cut_surface_area_l265_26530

theorem cube_cut_surface_area (cube_volume : Real) (height_A height_B height_C height_D : Real) :
  cube_volume = 1 ∧
  height_A = 1/3 ∧
  height_B = 1/4 ∧
  height_C = 1/6 ∧
  height_D = 1 - (height_A + height_B + height_C) →
  2 + 4 * (height_A + height_B + height_C + height_D) = 6 := by
  intro h
  have h1 : height_D = 1/4 := by
    rw [h.right.right.right.right]
    simp [h.right.left, h.right.right.left, h.right.right.right.left]
    norm_num
  have h2 : height_A + height_B + height_C + height_D = 1 := by
    rw [h1, h.right.left, h.right.right.left, h.right.right.right.left]
    norm_num
  rw [h2]
  norm_num
  
#check cube_cut_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_cut_surface_area_l265_26530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_in_open_interval_l265_26514

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5 - a) * x - 4 * a else a^x

-- State the theorem
theorem increasing_f_implies_a_in_open_interval :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → 1 < a ∧ a < 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_in_open_interval_l265_26514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_below_line_l265_26538

/-- The number of unit squares lying entirely below the line 3x + 75y = 675 in the first quadrant -/
def squares_below_line : ℕ := 896

/-- The equation of the line -/
def line_equation (x y : ℚ) : Prop := 3 * x + 75 * y = 675

/-- The x-intercept of the line -/
noncomputable def x_intercept : ℚ := 225

/-- The y-intercept of the line -/
noncomputable def y_intercept : ℚ := 9

/-- The slope of the line -/
noncomputable def line_slope : ℚ := -1/25

theorem count_squares_below_line :
  ∃ (count : ℕ), count = squares_below_line ∧
  (∀ x y : ℚ, 0 ≤ x ∧ 0 ≤ y ∧ x ≤ x_intercept ∧ y ≤ y_intercept →
    (⌊x⌋ + 1 ≤ x ∨ ⌊y⌋ + 1 ≤ y ∨ line_equation x y) →
    (x < ⌊x⌋ + 1 ∧ y < ⌊y⌋ + 1 ∧ y < line_slope * x)) →
  count = squares_below_line :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_below_line_l265_26538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l265_26565

noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x^(m^2 - 2*m - 3)

theorem power_function_properties (m : ℤ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- f is an even function
  (∀ x y : ℝ, 0 < x → x < y → f m y < f m x) →  -- f is monotonically decreasing on (0, +∞)
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l265_26565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_spent_is_700_l265_26551

/-- Represents the price reduction percentage as a real number between 0 and 1 -/
def price_reduction : ℚ := 3/10

/-- Represents the reduced price per kg in Rupees -/
def reduced_price : ℚ := 70

/-- Represents the additional amount of oil that can be purchased after the price reduction in kg -/
def additional_oil : ℚ := 3

/-- Calculates the amount spent on oil after the price reduction -/
def amount_spent_after_reduction (price_reduction : ℚ) (reduced_price : ℚ) (additional_oil : ℚ) : ℚ :=
  (reduced_price * additional_oil) / (1 / (1 - price_reduction) - 1)

/-- Theorem stating that the amount spent after the price reduction is 700 Rupees -/
theorem amount_spent_is_700 :
  amount_spent_after_reduction price_reduction reduced_price additional_oil = 700 := by
  sorry

#eval amount_spent_after_reduction price_reduction reduced_price additional_oil

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_spent_is_700_l265_26551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_y_axis_l265_26522

theorem symmetry_y_axis (θ : ℝ) : 
  (Real.cos θ = -Real.cos (θ + π/6) ∧ Real.sin θ = Real.sin (θ + π/6)) → θ = 5*π/12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_y_axis_l265_26522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_strategy_19_exists_strategy_23_l265_26576

/-- Represents a deck of cards -/
structure Deck where
  cards : Finset (Fin 36)
  suits : Fin 36 → Fin 4
  card_count : cards.card = 36
  suit_count : ∀ s : Fin 4, (cards.filter (λ c ↦ suits c = s)).card = 9

/-- Represents the strategy of the assistant and psychic -/
structure Strategy where
  encode : Deck → Fin 36 → Bool
  decode : Deck → Fin 36 → Fin 4

/-- The number of correct guesses made by the psychic using a given strategy -/
def correct_guesses (d : Deck) (s : Strategy) : ℕ :=
  (d.cards.filter (λ c ↦ s.decode d c = d.suits c)).card

/-- Theorem stating that there exists a strategy to guess at least 19 cards correctly -/
theorem exists_strategy_19 (d : Deck) : ∃ s : Strategy, correct_guesses d s ≥ 19 := by
  sorry

/-- Theorem stating that there exists a strategy to guess at least 23 cards correctly -/
theorem exists_strategy_23 (d : Deck) : ∃ s : Strategy, correct_guesses d s ≥ 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_strategy_19_exists_strategy_23_l265_26576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_equation_l265_26569

theorem sin_alpha_equation (α : Real) :
  Real.sin α * (1 + Real.sqrt 3 * Real.tan (π / 18)) = 1 →
  α = 13 * π / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_equation_l265_26569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_juggling_with_integer_avg_l265_26510

/-- A function that determines if a sequence of three integers is a juggling sequence -/
def isJugglingSequence (j₀ j₁ j₂ : ℕ) : Prop :=
  ∀ n : Fin 3, ∃! m : Fin 3, (n.val + [j₀, j₁, j₂].get n) % 3 = m.val

/-- The main theorem stating the existence of a non-juggling sequence with integer average -/
theorem exists_non_juggling_with_integer_avg : 
  ∃ j₀ j₁ j₂ : ℕ, 
    (j₀ + j₁ + j₂) % 3 = 0 ∧ 
    ¬ isJugglingSequence j₀ j₁ j₂ := by
  -- We'll use the sequence 2, 1, 0 as our example
  use 2, 1, 0
  apply And.intro
  · -- Prove that (2 + 1 + 0) % 3 = 0
    simp
  · -- Prove that 2, 1, 0 is not a juggling sequence
    intro h
    -- We'll show that f(0) = f(1) = f(2) = 2, violating injectivity
    have h0 : (0 + 2) % 3 = 2 := by norm_num
    have h1 : (1 + 1) % 3 = 2 := by norm_num
    have h2 : (2 + 0) % 3 = 2 := by norm_num
    -- This leads to a contradiction with the uniqueness part of ∃!
    sorry -- The full proof would expand on this contradiction

#check exists_non_juggling_with_integer_avg

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_juggling_with_integer_avg_l265_26510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l265_26574

/-- The parabola y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola y² = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- A line passing through the focus -/
structure FocalLine where
  m : ℝ  -- slope of the line

/-- Intersection points of the focal line with the parabola -/
def intersection (l : FocalLine) : ℝ × ℝ × ℝ × ℝ := sorry

/-- The x-coordinates of the intersection points -/
def x₁ (l : FocalLine) : ℝ := (intersection l).1
def x₂ (l : FocalLine) : ℝ := (intersection l).2.1

/-- The length of the chord AB -/
noncomputable def chordLength (l : FocalLine) : ℝ := 
  let (x₁, y₁, x₂, y₂) := intersection l
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem parabola_chord_length 
  (l : FocalLine)
  (h : x₁ l + x₂ l = 9) :
  chordLength l = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l265_26574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l265_26562

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℚ) : Prop := m1 = m2

/-- The slope of a line ax + by + c = 0 is -a/b -/
def line_slope (a b : ℚ) : ℚ := -a / b

theorem parallel_lines_a_value :
  ∀ (a : ℚ),
  (are_parallel (line_slope (a - 2) a) (line_slope 2 3)) →
  a = 6 := by
  intro a h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l265_26562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l265_26537

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the chord length
def chord_length : ℝ := 2

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, hyperbola a b x y ∧ circle_eq x y) →
  eccentricity a b = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l265_26537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_square_figure_l265_26528

/-- The perimeter of a figure composed of four squares with diagonals -/
noncomputable def perimeter_of_figure : ℝ := 14 + 4 * Real.sqrt 2

/-- The side length of each square in the figure -/
def square_side_length : ℝ := 2

/-- The number of squares in the figure -/
def number_of_squares : ℕ := 4

/-- The length of a diagonal in a square with side length 2 -/
noncomputable def diagonal_length : ℝ := Real.sqrt 8

/-- Theorem stating the perimeter of the figure -/
theorem perimeter_of_square_figure :
  perimeter_of_figure = 
    3 * square_side_length +  -- Horizontal extent
    2 * (2 * square_side_length) +  -- Vertical extent
    number_of_squares * (diagonal_length / 2)  -- Diagonal contribution
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_square_figure_l265_26528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_properties_l265_26594

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define a chord of the parabola
structure Chord where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_parabola : parabola A.1 A.2 ∧ parabola B.1 B.2

-- Define the property of being bisected by P
def bisected_by_P (c : Chord) : Prop :=
  P.1 = (c.A.1 + c.B.1) / 2 ∧ P.2 = (c.A.2 + c.B.2) / 2

-- Define the equation of a line in general form
def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a*x + b*y + c = 0

-- Define the length of a chord
noncomputable def chord_length (c : Chord) : ℝ :=
  Real.sqrt ((c.A.1 - c.B.1)^2 + (c.A.2 - c.B.2)^2)

-- State the theorem
theorem chord_properties (c : Chord) 
  (h1 : c.A ≠ c.B)
  (h2 : bisected_by_P c) :
  (line_equation 2 (-1) (-3) c.A.1 c.A.2 ∧ 
   line_equation 2 (-1) (-3) c.B.1 c.B.2) ∧
  chord_length c = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_properties_l265_26594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_faculty_count_l265_26598

/-- The initial number of faculty members -/
def X : ℝ := sorry

/-- The initial average salary -/
def S : ℝ := sorry

/-- The equation representing the changes in faculty numbers -/
axiom faculty_equation : (0.75 * X + 35) * 1.10 * 0.80 = 195

/-- The theorem stating that X is approximately 253 -/
theorem initial_faculty_count : ∃ ε > 0, |X - 253| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_faculty_count_l265_26598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_range_l265_26511

-- Define the point in the plane
noncomputable def point (a : ℝ) : ℝ × ℝ := (a - 1, (3 * a + 1) / (a - 1))

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

-- State the theorem
theorem point_in_third_quadrant_range :
  ∀ a : ℝ, in_third_quadrant (point a) ↔ -1/3 < a ∧ a < 1 :=
by
  sorry

#check point_in_third_quadrant_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_range_l265_26511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l265_26582

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem max_sum_arithmetic_sequence 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_15 : S a 15 > 0) 
  (h_16 : S a 16 < 0) :
  ∃ n : ℕ, (∀ m : ℕ, S a m ≤ S a n) ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l265_26582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spaceship_distance_l265_26512

/-- Represents the distance between two points in light-years -/
def Distance : Type := ℝ

/-- The distance from Earth to Planet X -/
noncomputable def earth_to_x : Distance := sorry

/-- The distance from Planet X to Planet Y -/
def x_to_y : Distance := (0.1 : ℝ)

/-- The distance from Planet Y back to Earth -/
def y_to_earth : Distance := (0.1 : ℝ)

/-- The total distance traveled by the spaceship -/
def total_distance : Distance := (0.7 : ℝ)

theorem spaceship_distance : earth_to_x = (0.5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spaceship_distance_l265_26512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l265_26564

-- Define the function f(x) = 2^x - 2^(-x)
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) - Real.exp (-x * Real.log 2)

-- Theorem statement
theorem f_increasing_and_odd : 
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l265_26564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l265_26550

-- Define the constant term of the expansion
noncomputable def constant_term (a : ℝ) : ℝ := (1 / a)^3 * (Nat.choose 6 3)

-- Define the area of the closed figure
noncomputable def area (a : ℝ) : ℝ := ∫ x in (0 : ℝ)..1, x^(1/a) - x^2

-- Theorem statement
theorem area_of_closed_figure :
  ∃ a : ℝ, constant_term a = 540 ∧ area a = 5/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l265_26550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_sine_difference_l265_26599

theorem max_a_for_sine_difference (a : ℝ) : 
  (∀ m x : ℝ, 0 ≤ m ∧ m ≤ a ∧ 0 ≤ x ∧ x ≤ π → |Real.sin x - Real.sin (x + m)| ≤ 1) ↔ 
  a ≤ π / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_sine_difference_l265_26599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l265_26507

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^(a^2 - 4*a - 9)

theorem f_properties (a : ℝ) (h : a = 1) :
  (∀ x, f a x = f a (-x)) ∧
  (∀ x y, 0 < x → x < y → f a y < f a x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l265_26507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_dot_product_l265_26592

/-- Circle with equation x²+y²+4x-5=0 -/
def my_circle (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 5 = 0

/-- Chord AB with midpoint (-1, 1) -/
def chord_midpoint (a b : ℝ × ℝ) : Prop :=
  (a.1 + b.1) / 2 = -1 ∧ (a.2 + b.2) / 2 = 1

/-- Line AB intersects x-axis at point P (0, 0) -/
def intersects_x_axis (a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, a.1 + t * (b.1 - a.1) = 0 ∧ a.2 + t * (b.2 - a.2) = 0

/-- Dot product of vectors PA and PB -/
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem chord_dot_product (a b : ℝ × ℝ) :
  my_circle a.1 a.2 ∧ my_circle b.1 b.2 ∧
  chord_midpoint a b ∧
  intersects_x_axis a b →
  dot_product a b = -5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_dot_product_l265_26592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spending_l265_26583

/-- Represents the amount Ben spent -/
def B : ℝ := sorry

/-- Represents the amount David spent -/
def D : ℝ := sorry

/-- For every dollar Ben spent, David spent 50 cents less -/
axiom david_spent_less : D = B / 2

/-- Ben paid $15.00 more than David -/
axiom ben_paid_more : B = D + 15

/-- Theorem: Ben and David's total spending is $45.00 -/
theorem total_spending : B + D = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spending_l265_26583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tracy_feeds_one_and_half_cups_l265_26591

/-- The number of cups in one pound of dog food -/
def cups_per_pound : ℚ := 2.25

/-- The number of pounds of food consumed by both dogs per day -/
def pounds_per_day : ℚ := 4

/-- The number of times Tracy feeds her dogs per day -/
def meals_per_day : ℕ := 3

/-- The number of dogs Tracy has -/
def number_of_dogs : ℕ := 2

/-- The number of cups of food Tracy feeds each dog per meal -/
noncomputable def cups_per_dog_per_meal : ℚ := 
  (pounds_per_day * cups_per_pound) / (meals_per_day * number_of_dogs)

theorem tracy_feeds_one_and_half_cups : cups_per_dog_per_meal = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tracy_feeds_one_and_half_cups_l265_26591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_is_9_56_l265_26543

/-- A rectangular yard with two congruent isosceles right triangular flower beds -/
structure Yard :=
  (length : ℝ)
  (width : ℝ)
  (trapezoid_side1 : ℝ)
  (trapezoid_side2 : ℝ)

/-- The area of an isosceles right triangle -/
noncomputable def isosceles_right_triangle_area (side : ℝ) : ℝ :=
  (1/2) * side^2

/-- The area of the flower beds -/
noncomputable def flower_beds_area (y : Yard) : ℝ :=
  2 * isosceles_right_triangle_area ((y.trapezoid_side2 - y.trapezoid_side1) / 2)

/-- The total area of the yard -/
noncomputable def yard_area (y : Yard) : ℝ :=
  y.length * y.width

/-- The fraction of the yard occupied by the flower beds -/
noncomputable def flower_beds_fraction (y : Yard) : ℝ :=
  flower_beds_area y / yard_area y

/-- Theorem stating that the fraction of the yard occupied by the flower beds is 9/56 -/
theorem flower_beds_fraction_is_9_56 (y : Yard) 
  (h1 : y.trapezoid_side1 = 20)
  (h2 : y.trapezoid_side2 = 35)
  (h3 : y.length = 35)
  (h4 : y.width = 10) : 
  flower_beds_fraction y = 9/56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_is_9_56_l265_26543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l265_26587

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos (2 * x)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  (∀ x : ℝ, f (π/3 + x) = f (π/3 - x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l265_26587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l265_26566

/-- A power function f(x) = (m^2 - m - 1)x^(-m) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(-m)

/-- The function f is decreasing on the interval (0, +∞) -/
def is_decreasing (m : ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f m y < f m x

/-- There exists a unique m > 0 such that f is decreasing and m = 2 -/
theorem power_function_decreasing :
  ∃! m : ℝ, m > 0 ∧ is_decreasing m ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l265_26566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_l265_26536

/-- Given four points in 3D space, proves that the lines formed by two pairs of points are parallel. -/
theorem lines_parallel (A B C D : ℝ × ℝ × ℝ) : 
  A = (1, 2, 3) → 
  B = (-2, -1, 6) → 
  C = (3, 2, 1) → 
  D = (4, 3, 0) → 
  ∃ (k : ℝ), k ≠ 0 ∧ 
    (B.fst - A.fst, B.snd - A.snd, (B.2.2 - A.2.2)) = 
    k • (D.fst - C.fst, D.snd - C.snd, (D.2.2 - C.2.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_l265_26536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l265_26548

theorem trig_inequality : 
  Real.cos (2 * Real.pi / 5) < Real.sin (3 * Real.pi / 5) ∧ 
  Real.sin (3 * Real.pi / 5) < Real.tan (2 * Real.pi / 5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l265_26548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_beta_l265_26589

theorem sin_alpha_minus_beta (α β : ℝ) : 
  α ∈ Set.Ioo (π/3) (5*π/6) → 
  β ∈ Set.Ioo (π/3) (5*π/6) → 
  Real.sin (α + π/6) = 4/5 → 
  Real.cos (β - 5*π/6) = 5/13 → 
  Real.sin (α - β) = 16/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_beta_l265_26589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_days_l265_26552

noncomputable section

/-- The number of days A needs to complete the task -/
def days_A : ℝ := 18

/-- The number of days B needs to complete the task -/
def days_B : ℝ := 24

/-- The number of days C needs to complete the task -/
def days_C : ℝ := 27

/-- B needs 6 more days than A to complete the task -/
axiom B_needs_more_days : days_B = days_A + 6

/-- C needs 3 more days than B to complete the task -/
axiom C_needs_more_days : days_C = days_B + 3

/-- The amount of work completed by A in one day -/
noncomputable def work_rate_A : ℝ := 1 / days_A

/-- The amount of work completed by B in one day -/
noncomputable def work_rate_B : ℝ := 1 / days_B

/-- The amount of work completed by C in one day -/
noncomputable def work_rate_C : ℝ := 1 / days_C

/-- A working for 3 days and B working for 4 days together complete 
    the same amount of work as C does in 9 days -/
axiom work_equality : 3 * work_rate_A + 4 * work_rate_B = 9 * work_rate_C

theorem task_completion_days : 
  days_A = 18 ∧ days_B = 24 ∧ days_C = 27 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_days_l265_26552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_l265_26505

theorem inscribed_hexagon_area (circle_area : ℝ) (h : circle_area = 400 * Real.pi) :
  let circle_radius : ℝ := Real.sqrt (circle_area / Real.pi)
  let triangle_area : ℝ := (circle_radius ^ 2 * Real.sqrt 3) / 4
  let hexagon_area : ℝ := 6 * triangle_area
  hexagon_area = 600 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_l265_26505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_8_four_times_in_five_rolls_l265_26586

/-- A fair 10-sided die -/
def fair_10_sided_die : Finset ℕ := Finset.range 10

/-- The probability of rolling at least an 8 on a single roll -/
def p_at_least_8 : ℚ := 3 / 10

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The minimum number of successful rolls required -/
def min_successes : ℕ := 4

/-- The probability of rolling at least an 8 at least four times in five rolls -/
theorem prob_at_least_8_four_times_in_five_rolls :
  (Finset.sum (Finset.range 2)
    (λ k ↦ (num_rolls.choose (num_rolls - k)) * p_at_least_8 ^ (num_rolls - k) * (1 - p_at_least_8) ^ k)) =
  2859.3 / 10000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_8_four_times_in_five_rolls_l265_26586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chromium_percentage_in_new_alloy_l265_26596

/-- Represents an alloy with its chromium percentage and weight -/
structure Alloy where
  chromium_percent : ℚ
  weight : ℚ

/-- Calculates the total chromium weight from a list of alloys -/
def total_chromium (alloys : List Alloy) : ℚ :=
  alloys.foldl (fun acc a => acc + a.chromium_percent / 100 * a.weight) 0

/-- Calculates the total weight from a list of alloys -/
def total_weight (alloys : List Alloy) : ℚ :=
  alloys.foldl (fun acc a => acc + a.weight) 0

/-- Theorem: The percentage of chromium in the new alloy is approximately 9.47% -/
theorem chromium_percentage_in_new_alloy (alloy_a alloy_b alloy_c : Alloy)
    (h_a : alloy_a = { chromium_percent := 12, weight := 12 })
    (h_b : alloy_b = { chromium_percent := 10, weight := 20 })
    (h_c : alloy_c = { chromium_percent := 8, weight := 28 }) :
    let alloys := [alloy_a, alloy_b, alloy_c]
    let new_chromium_percent := (total_chromium alloys / total_weight alloys) * 100
    ∃ ε > 0, |new_chromium_percent - 947/100| < ε := by
  sorry

#eval let alloys := [
    { chromium_percent := 12, weight := 12 },
    { chromium_percent := 10, weight := 20 },
    { chromium_percent := 8, weight := 28 }
  ]
  (total_chromium alloys / total_weight alloys) * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chromium_percentage_in_new_alloy_l265_26596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_value_l265_26518

/-- Represents a digit in base 9 --/
def Base9Digit := Fin 9

/-- Represents a number in base 9 with up to 4 digits --/
def Base9Number := List Base9Digit

/-- Addition function for base 9 numbers --/
def base9Add (a b : Base9Number) : Base9Number :=
  sorry

/-- Convert a base 10 number to a base 9 number --/
def toBase9 (n : ℕ) : Base9Number :=
  sorry

/-- Convert a natural number to a Base9Digit --/
def natToBase9Digit (n : ℕ) : Base9Digit :=
  ⟨n % 9, by
    apply Nat.mod_lt
    exact Nat.zero_lt_succ 8⟩

/-- The addition problem in the question --/
def additionProblem (triangle : Base9Digit) : Prop :=
  let num1 := [natToBase9Digit 4, natToBase9Digit 3, natToBase9Digit 2, triangle]
  let num2 := [natToBase9Digit 0, triangle, natToBase9Digit 7, natToBase9Digit 1]
  let num3 := [natToBase9Digit 0, natToBase9Digit 0, triangle, natToBase9Digit 3]
  let result := [natToBase9Digit 4, natToBase9Digit 8, triangle, natToBase9Digit 5]
  base9Add (base9Add num1 num2) num3 = result

theorem triangle_value : ∃! (triangle : Base9Digit), additionProblem triangle ∧ triangle.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_value_l265_26518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_water_amount_l265_26581

/-- Represents the amount of water added or lost in a given hour -/
def water_change : Fin 8 → Int
  | 0 => 8    -- 1st hour
  | 1 => 10   -- 2nd hour
  | 2 => 10   -- 3rd hour
  | 3 => 14   -- 4th hour
  | 4 => 14   -- 5th hour
  | 5 => 12   -- 6th hour
  | 6 => 4    -- 7th hour (12 added, 8 lost)
  | 7 => 7    -- 8th hour (12 added, 5 lost)

/-- The final amount of water in the pool after 8 hours -/
def final_water : Int := (Finset.univ.sum water_change)

theorem pool_water_amount :
  final_water = 79 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_water_amount_l265_26581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_240_l265_26501

theorem divisibility_by_240 (n : ℤ) (h : Nat.Coprime n.natAbs 30) : 
  (240 : ℤ) ∣ (-n^4 - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_240_l265_26501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l265_26572

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

theorem f_is_odd : ∀ x, f (-x) = -f x := by
  intro x
  simp [f]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l265_26572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_negatives_l265_26509

theorem compare_negatives : -(0.3 : ℝ) > -|(-(1/3) : ℝ)| := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_negatives_l265_26509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_alpha_l265_26590

theorem sin_squared_alpha (α β : ℝ) (h1 : Real.cos (α - β) = 4/5) (h2 : 0 < α) (h3 : α < π/2) :
  (Real.sin α) ^ 2 = 144/169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_alpha_l265_26590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l265_26521

theorem expansion_coefficient (n : ℕ) (h : n ≥ 3) :
  (n.choose 1 * (-Real.sqrt 2)) / (n.choose 3 * (-2 * Real.sqrt 2)) = 1 / 2 →
  (n.choose 2 : ℤ) * (-2) = -12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l265_26521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l265_26525

noncomputable def data : List ℝ := [2, 3, 4, 5, 6]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

theorem variance_of_data : variance data = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l265_26525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_C_equals_union_l265_26588

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B : Set ℝ := {x | (x + 1) / (x - 1) < 0}
def C : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- State the theorem
theorem set_C_equals_union : C = Set.Icc (-3) (-1) ∪ Set.Icc 1 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_C_equals_union_l265_26588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_approx_28_3_l265_26555

/-- The distance between two points on a 2D plane, where one point is at the origin (0, 0) 
    and the other is at coordinates (-15, 24). -/
noncomputable def distance_between_towns : ℝ :=
  Real.sqrt ((15 : ℝ) ^ 2 + (24 : ℝ) ^ 2)

/-- Theorem stating that the distance between the two towns is approximately 28.3 units. -/
theorem distance_approx_28_3 : 
  |distance_between_towns - 28.3| < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_approx_28_3_l265_26555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_battery_theorem_l265_26503

/-- Represents the duration of a phone battery usage scenario -/
noncomputable def phone_battery_duration (standby_time : ℝ) (depletion_rate : ℝ) : ℝ :=
  let talk_time := standby_time / (depletion_rate + 1)
  2 * talk_time

theorem phone_battery_theorem (standby_time : ℝ) (depletion_rate : ℝ) 
  (h1 : standby_time = 6)
  (h2 : depletion_rate = 35) :
  phone_battery_duration standby_time depletion_rate = 35/3 := by
  sorry

#eval (35/3 : ℚ)  -- This should output 11 + 2/3, which is equivalent to 11 hours 40 minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_battery_theorem_l265_26503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l265_26556

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 5

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := Real.sqrt 5

-- Define the area of the circle
noncomputable def circle_area : ℝ := 5 * Real.pi

-- Theorem statement
theorem circle_properties :
  (∀ x y, circle_equation x y ↔ ((x - 2)^2 + (y + 1)^2 = circle_radius^2)) ∧
  circle_area = Real.pi * circle_radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l265_26556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_distance_theorem_l265_26539

/-- Represents a plane in 3D space --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Checks if two planes are parallel --/
def are_parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

/-- Calculates the distance between two parallel planes --/
noncomputable def distance_between_planes (p1 p2 : Plane) : ℝ :=
  abs (p2.d / 2 - p1.d) / Real.sqrt (p1.a^2 + p1.b^2 + p1.c^2)

theorem plane_distance_theorem (p1 p2 : Plane) :
  p1 = Plane.mk 3 (-1) 4 (-3) →
  p2 = Plane.mk 6 (-2) 8 7 →
  are_parallel p1 p2 →
  distance_between_planes p1 p2 = 13 * Real.sqrt 26 / 52 := by
  sorry

#check plane_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_distance_theorem_l265_26539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pair_problem_l265_26584

theorem integer_pair_problem (a b q r : ℕ) : 
  a > b → 
  a^2 + b^2 = q * (a + b) + r →
  q^2 + 2 * r = 2020 →
  ((a = 53 ∧ b = 29) ∨ (a = 53 ∧ b = 15)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pair_problem_l265_26584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_ratio_l265_26546

theorem cookie_ratio (num_cookies : ℕ) (num_chips : ℕ) (avg_pieces : ℕ) : 
  num_cookies = 48 → 
  num_chips = 108 → 
  avg_pieces = 3 → 
  (num_cookies * avg_pieces - num_chips) / num_chips = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_ratio_l265_26546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_simplification_l265_26532

theorem exponent_simplification (k : ℤ) : 
  (3 : ℝ)^(-k) - (3 : ℝ)^(-(k+2)) + (3 : ℝ)^(-(k+1)) - (3 : ℝ)^(-(k-1)) = (-16 * (3 : ℝ)^(-k)) / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_simplification_l265_26532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_and_angle_bisector_l265_26597

noncomputable section

-- Define the triangle vertices
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (2, 4)
def C : ℝ × ℝ := (0, 3)

-- Define the midpoint of AB
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the slope and y-intercept of line l (median to AB)
noncomputable def m_l : ℝ := (C.2 - D.2) / (C.1 - D.1)
noncomputable def b_l : ℝ := C.2 - m_l * C.1

-- Define the slope and y-intercept of line CD
def m_CD : ℝ := 4/3
def b_CD : ℝ := 3

-- Hypothesis that line l is the angle bisector of ∠ACD
def line_l_is_angle_bisector : Prop := sorry

theorem triangle_median_and_angle_bisector :
  (m_l = 1 ∧ b_l = 3) ∧
  (line_l_is_angle_bisector → (m_CD = 4/3 ∧ b_CD = 3)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_and_angle_bisector_l265_26597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_mixed_numbers_exist_l265_26553

theorem distinct_mixed_numbers_exist : ∃ (a₁ a₂ a₃ a₄ a₅ a₆ b₁ b₂ b₃ b₄ b₅ b₆ : ℕ),
  (∀ i, i ∈ [1, 2, 3, 4, 5, 6] → 0 < b_i ∧ b_i < a_i) ∧ 
  (∀ i, i ∈ [1, 2, 3, 4, 5, 6] → Nat.gcd a_i b_i = 1) ∧
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = b₁ + b₂ + b₃ + b₄ + b₅ + b₆) ∧
  (∀ i j, i ∈ [1, 2, 3, 4, 5, 6] → j ∈ [1, 2, 3, 4, 5, 6] → i ≠ j → a_i / b_i ≠ a_j / b_j) ∧
  (∀ i j, i ∈ [1, 2, 3, 4, 5, 6] → j ∈ [1, 2, 3, 4, 5, 6] → i ≠ j → a_i % b_i ≠ a_j % b_j) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_mixed_numbers_exist_l265_26553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_profit_problem_check_result_l265_26579

noncomputable def profit_percentage (cost : ℝ) (selling_price : ℝ) : ℝ :=
  (selling_price - cost) / cost * 100

theorem bicycle_profit_problem (a_cost : ℝ) (a_profit_percent : ℝ) (c_price : ℝ)
  (h1 : a_cost = 150)
  (h2 : a_profit_percent = 20)
  (h3 : c_price = 225) :
  profit_percentage (a_cost * (1 + a_profit_percent / 100)) c_price = 25 :=
by
  sorry

-- Replace #eval with a theorem that states the expected result
theorem check_result :
  profit_percentage (150 * (1 + 20 / 100)) 225 = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_profit_problem_check_result_l265_26579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_value_l265_26513

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log (x + Real.sqrt (x^2 + 1))

-- State the theorem
theorem find_b_value :
  ∃ b : ℝ, b = -2 ∧
  ∀ a : ℝ, (f (1 + a) + 1 + Real.log (Real.sqrt 2 + 1) < 0) ↔ a < b :=
by
  -- We'll use -2 as the value of b
  use -2
  constructor
  · -- Prove b = -2
    rfl
  · -- Prove the equivalence
    intro a
    -- We'll leave the actual proof as sorry for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_value_l265_26513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expansion_sum_of_coefficients_l265_26534

theorem sum_of_coefficients_expansion (d : ℝ) : 
  let expanded := -(4 - d) * (d + 3 * (4 - d))
  expanded = -2*d^2 + 20*d - 48 := by sorry

theorem sum_of_coefficients (d : ℝ) :
  let expanded := -(4 - d) * (d + 3 * (4 - d))
  (-2 : ℝ) + 20 + (-48) = -30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expansion_sum_of_coefficients_l265_26534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l265_26560

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 3*x + 1

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (0, 1)

-- Define the slope of the tangent line
def tangent_slope : ℝ := 3

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, (3*x - y + 1 = 0) ↔ 
  (∃ t : ℝ, (x, y) = (t, f t) ∧ 
   (y - f tangent_point.1) / (x - tangent_point.1) = tangent_slope) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l265_26560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circle_ratio_max_ratio_at_45_degrees_l265_26585

open Real

/-- The ratio of the radius of the inscribed circle to the radius of the circumscribed circle
    in a right-angled triangle with acute angle α -/
noncomputable def circleRatio (α : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin (π/4 - α/2) * Real.sin (α/2)

theorem right_triangle_circle_ratio (α : ℝ) (h : 0 < α ∧ α < π/2) :
  ∃ (r R : ℝ), r > 0 ∧ R > 0 ∧ 
  (r / R = circleRatio α) ∧
  (∀ β, 0 < β ∧ β < π/2 → circleRatio α ≥ circleRatio β) ∧
  (circleRatio α = circleRatio (π/4) → α = π/4) :=
by sorry

/-- The maximum ratio occurs when α = π/4 (45 degrees) -/
theorem max_ratio_at_45_degrees :
  ∀ α, 0 < α ∧ α < π/2 → circleRatio α ≤ circleRatio (π/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circle_ratio_max_ratio_at_45_degrees_l265_26585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sample_mean_calculation_l265_26570

/-- Calculates the total sample mean given stratified random sampling data --/
theorem total_sample_mean_calculation 
  (male_samples : ℕ) 
  (female_samples : ℕ) 
  (male_mean : ℝ) 
  (female_mean : ℝ) :
  let total_samples := male_samples + female_samples
  let total_sum := (male_samples : ℝ) * male_mean + (female_samples : ℝ) * female_mean
  total_sum / total_samples = 10.8 :=
by
  -- The proof would go here
  sorry

/-- Verifies the total sample mean for the given survey data --/
example : 
  let male_samples := 20
  let female_samples := 30
  let male_mean := 12
  let female_mean := 10
  let total_samples := male_samples + female_samples
  let total_sum := (male_samples : ℝ) * male_mean + (female_samples : ℝ) * female_mean
  total_sum / total_samples = 10.8 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sample_mean_calculation_l265_26570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_ab_l265_26575

open Real

-- Define the function f(x) = 3^x
noncomputable def f (x : ℝ) : ℝ := (3 : ℝ) ^ x

-- State the theorem
theorem max_value_of_f_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : f (a + b) = 9) :
  ∀ x y, 0 < x ∧ 0 < y ∧ f (x + y) = 9 → f (a * b) ≥ f (x * y) ∧ f (a * b) ≤ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_ab_l265_26575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_arrangement_theorem_l265_26516

/-- Represents an apple with a weight -/
structure Apple where
  weight : ℝ

/-- Represents a bag containing two apples -/
structure Bag where
  apple1 : Apple
  apple2 : Apple

/-- The theorem to be proved -/
theorem apple_arrangement_theorem 
  (apples : Finset Apple) 
  (h1 : apples.card = 300)
  (h2 : ∀ a b, a ∈ apples → b ∈ apples → a.weight ≤ 2 * b.weight) :
  ∃ bags : Finset Bag, 
    (bags.card = 150) ∧ 
    (∀ a, a ∈ apples → ∃ b ∈ bags, a = b.apple1 ∨ a = b.apple2) ∧
    (∀ b1 b2, b1 ∈ bags → b2 ∈ bags → 
      (b1.apple1.weight + b1.apple2.weight) ≤ 
      1.5 * (b2.apple1.weight + b2.apple2.weight)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_arrangement_theorem_l265_26516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_500_20_2_l265_26520

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- The difference between compound and simple interest -/
noncomputable def interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  compound_interest principal rate time - simple_interest principal rate time

theorem interest_difference_500_20_2 :
  interest_difference 500 20 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_500_20_2_l265_26520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l265_26578

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x - a) / x

-- State the theorem
theorem monotonic_increasing_f_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 2 4, Monotone (fun x => f a x)) →
  a ≥ -Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l265_26578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l265_26580

/-- Given two acute angles α and β in the cartesian coordinate plane,
    where the terminal side of α intersects the unit circle at a point with abscissa √5/5,
    and the terminal side of β intersects the unit circle at a point with ordinate √2/10,
    then 2α + β = 3π/4. -/
theorem angle_sum_theorem (α β : ℝ) : 
  0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 →  -- acute angles
  Real.cos α = Real.sqrt 5 / 5 →       -- abscissa of A
  Real.sin β = Real.sqrt 2 / 10 →      -- ordinate of B
  2 * α + β = 3 * π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l265_26580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l265_26542

open Real

theorem function_inequality (f : ℝ → ℝ) (a : ℝ) (h_a : a > 0) :
  (∀ x ∈ Set.Ioo 0 a, DifferentiableAt ℝ f x ∧ x * (deriv f x) < 1) →
  ∀ x ∈ Set.Ioo 0 a, f x - f a > log (x / a) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l265_26542
