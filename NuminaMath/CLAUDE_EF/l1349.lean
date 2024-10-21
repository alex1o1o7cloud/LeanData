import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_time_l1349_134919

/-- Calculates the time a bus spends stopped per hour given its average speeds with and without stoppages. -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 50 → speed_with_stops = 40 → 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 12 := by
  sorry

-- Remove the #eval line as it's causing issues with universe levels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_time_l1349_134919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_lateral_face_angle_l1349_134999

/-- A regular hexagonal pyramid where the height is equal to the side length of the base -/
structure RegularHexagonalPyramid where
  -- Side length of the base (which is also the height of the pyramid)
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The angle between a lateral face and the base of a regular hexagonal pyramid -/
noncomputable def lateral_face_angle (p : RegularHexagonalPyramid) : ℝ :=
  Real.arctan (2 / Real.sqrt 3)

/-- Theorem: In a regular hexagonal pyramid where the height is equal to the side length of the base,
    the angle between a lateral face and the base is arctan(2/√3) -/
theorem regular_hexagonal_pyramid_lateral_face_angle (p : RegularHexagonalPyramid) :
  lateral_face_angle p = Real.arctan (2 / Real.sqrt 3) := by
  -- Unfold the definition of lateral_face_angle
  unfold lateral_face_angle
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_lateral_face_angle_l1349_134999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_to_doubled_length_rectangular_solid_ratio_l1349_134954

/-- The ratio of the surface area of a cube to the surface area of a rectangular solid
    with doubled length is 3/5. -/
theorem cube_to_doubled_length_rectangular_solid_ratio :
  ∀ s : ℝ, s > 0 →
  (6 * s^2) / (2 * (2*s) * s + 2 * (2*s) * s + 2 * s * s) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_to_doubled_length_rectangular_solid_ratio_l1349_134954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_size_in_acres_l1349_134943

/-- Represents the dimensions of a trapezoid on a map --/
structure MapTrapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ

/-- Calculates the area of a trapezoid in square centimeters --/
noncomputable def trapezoid_area (t : MapTrapezoid) : ℝ :=
  (t.shorter_base + t.longer_base) * t.height / 2

/-- Converts square centimeters to square miles using the given scale --/
noncomputable def cm2_to_miles2 (area_cm2 : ℝ) : ℝ :=
  area_cm2 * 9

/-- Converts square miles to acres --/
noncomputable def miles2_to_acres (area_miles2 : ℝ) : ℝ :=
  area_miles2 * 640

/-- The main theorem stating the actual size of the plot in acres --/
theorem plot_size_in_acres (t : MapTrapezoid) 
    (h1 : t.shorter_base = 12)
    (h2 : t.longer_base = 18)
    (h3 : t.height = 15) : 
  miles2_to_acres (cm2_to_miles2 (trapezoid_area t)) = 1296000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_size_in_acres_l1349_134943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_height_is_14_l1349_134981

-- Define the heights of the animals and the conversion factor
noncomputable def camel_height_feet : ℚ := 28
noncomputable def camel_to_hare_ratio : ℚ := 24
noncomputable def inches_per_foot : ℚ := 12

-- Define the height of the hare in inches
noncomputable def hare_height_inches : ℚ := camel_height_feet * inches_per_foot / camel_to_hare_ratio

-- Theorem to prove
theorem hare_height_is_14 : hare_height_inches = 14 := by
  -- Unfold the definition of hare_height_inches
  unfold hare_height_inches
  -- Unfold the definitions of the constants
  unfold camel_height_feet camel_to_hare_ratio inches_per_foot
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_height_is_14_l1349_134981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_sum_l1349_134970

theorem cos_squared_sum (A B C : ℝ) 
  (h1 : Real.sin A + Real.sin B + Real.sin C = 0)
  (h2 : Real.cos A + Real.cos B + Real.cos C = 0) :
  Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_sum_l1349_134970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_27_l1349_134992

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 15 then x^3 - 4 else x - 20

-- Theorem statement
theorem f_composition_27 : f (f (f 27)) = 319 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_27_l1349_134992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l1349_134991

/-- A power function passing through (√2/2, 1/2) -/
def f (x : ℝ) : ℝ := x^2

/-- The function g(x) = e^x * f(x) -/
noncomputable def g (x : ℝ) : ℝ := Real.exp x * f x

/-- The derivative of g(x) -/
noncomputable def g' (x : ℝ) : ℝ := (2*x + x^2) * Real.exp x

theorem g_decreasing_interval :
  ∀ x ∈ Set.Ioo (-2 : ℝ) 0, g' x < 0 ∧
  ∀ y ∉ Set.Icc (-2 : ℝ) 0, g' y > 0 := by
  sorry

#check g_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l1349_134991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_eight_l1349_134973

theorem cube_root_of_negative_eight :
  (-8 : ℝ) ^ (1/3 : ℝ) = -2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_eight_l1349_134973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperatureMeasurement_l1349_134968

/-- Represents a polynomial of degree ≤ 3 -/
def polynomial (a b c d : ℝ) (t : ℝ) : ℝ := a * t^3 + b * t^2 + c * t + d

/-- The average value of a function over an interval -/
noncomputable def averageValue (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (1 / (b - a)) * ∫ x in a..b, f x

theorem temperatureMeasurement (a b c d : ℝ) :
  ∃ (t₁ t₂ : ℝ),
    t₁ = 1 + 4/15 ∧ 
    t₂ = 4 + 11/15 ∧ 
    averageValue (polynomial a b c d) 0 6 = (polynomial a b c d t₁ + polynomial a b c d t₂) / 2 := by
  sorry

#check temperatureMeasurement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperatureMeasurement_l1349_134968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_min_area_PECF_dot_product_at_min_area_l1349_134945

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}

-- Define the line L: x + y + 3 = 0
def line_L : Set (ℝ × ℝ) :=
  {p | p.1 + p.2 + 3 = 0}

-- Define the function for the area of quadrilateral PECF
noncomputable def area_PECF (P : ℝ × ℝ) : ℝ :=
  2 * Real.sqrt ((P.1 - 1)^2 + (P.2 - 2)^2 - 4)

-- Define the dot product of PE and PF
noncomputable def dot_product_PE_PF (P : ℝ × ℝ) : ℝ :=
  14 * (1 - 2 * ((2 / Real.sqrt ((P.1 - 1)^2 + (P.2 - 2)^2))^2))

theorem circle_C_properties :
  (∃ (a : ℝ), a > 0 ∧ circle_C = {p | (p.1 - a)^2 + (p.2 - 2*a)^2 = 4*a^2}) ∧
  (∃ (x : ℝ), (x, 0) ∈ circle_C) ∧
  (∃ (p q : ℝ × ℝ), p ∈ circle_C ∧ q ∈ circle_C ∧ p ≠ q ∧
    p.2 = p.1 + 2 ∧ q.2 = q.1 + 2 ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 14) := by
  sorry

theorem min_area_PECF :
  ∃ (P : ℝ × ℝ), P ∈ line_L ∧
    (∀ (Q : ℝ × ℝ), Q ∈ line_L → area_PECF P ≤ area_PECF Q) ∧
    area_PECF P = 2 * Real.sqrt 14 := by
  sorry

theorem dot_product_at_min_area :
  ∀ (P : ℝ × ℝ), P ∈ line_L →
    (∀ (Q : ℝ × ℝ), Q ∈ line_L → area_PECF P ≤ area_PECF Q) →
    dot_product_PE_PF P = 5 * Real.sqrt 14 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_min_area_PECF_dot_product_at_min_area_l1349_134945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1349_134957

/-- The curve C in the xy-plane -/
def curve_C (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x + 4*y - 8 = 0

/-- The distance function from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x + 4*y - 8| / Real.sqrt 17

/-- Theorem: The point (9/5, 4/5) on curve C minimizes the distance to line l -/
theorem min_distance_point :
  curve_C (9/5) (4/5) ∧
  ∀ x y : ℝ, curve_C x y →
    distance_to_line (9/5) (4/5) ≤ distance_to_line x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1349_134957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_crane_height_l1349_134928

/-- Represents the heights of buildings and cranes in feet -/
structure ConstructionSite where
  building1_height : ℝ
  building2_height : ℝ
  building3_height : ℝ
  crane2_height : ℝ
  crane3_height : ℝ

/-- The average height of cranes is 13% taller than the average height of buildings -/
def crane_height_ratio : ℝ := 1.13

theorem first_crane_height (site : ConstructionSite) 
  (h1 : site.building1_height = 200)
  (h2 : site.building2_height = 100)
  (h3 : site.building3_height = 140)
  (h4 : site.crane2_height = 120)
  (h5 : site.crane3_height = 147) :
  ∃ (crane1_height : ℝ), 
    (crane1_height + site.crane2_height + site.crane3_height) / 3 = 
    crane_height_ratio * (site.building1_height + site.building2_height + site.building3_height) / 3 ∧
    abs (crane1_height - 230.2) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_crane_height_l1349_134928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_symmetry_l1349_134972

theorem sin_cos_symmetry (k : ℤ) (x : ℝ) :
  Real.sin (π/4 + k*π - x) = Real.cos (π/4 + k*π + x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_symmetry_l1349_134972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l1349_134915

/-- Represents a sequence of A's and B's -/
inductive ABSequence
  | A : ABSequence
  | B : ABSequence

/-- Checks if a run of A's has even length -/
def evenLengthA (s : List ABSequence) : Bool :=
  sorry

/-- Checks if a run of B's has odd length -/
def oddLengthB (s : List ABSequence) : Bool :=
  sorry

/-- Checks if a sequence satisfies the given conditions -/
def validSequence (s : List ABSequence) : Bool :=
  evenLengthA s && oddLengthB s

/-- Generates all possible sequences of a given length -/
def generateSequences (n : ℕ) : List (List ABSequence) :=
  sorry

/-- Counts the number of valid sequences of a given length -/
def countValidSequences (n : ℕ) : ℕ :=
  (List.filter validSequence (generateSequences n)).length

/-- The main theorem stating that there are 375 valid sequences of length 16 -/
theorem valid_sequences_count : countValidSequences 16 = 375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l1349_134915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_coloring_l1349_134971

/-- A type representing a finite projective plane of order p -/
structure ProjectivePlane (p : ℕ) where
  points : Type
  lines : Type
  point_on_line : points → lines → Prop

/-- Theorem stating the existence of a valid coloring for the board -/
theorem exists_valid_coloring (p : ℕ) [Fact p.Prime] :
  ∃ (plane : ProjectivePlane p),
    (∀ row : Fin (p^2 + p + 1), ∃! (k : Fin (p + 1)), ∃ (pt : plane.points) (ln : plane.lines), plane.point_on_line pt ln) ∧
    (∀ col : Fin (p^2 + p + 1), ∃! (k : Fin (p + 1)), ∃ (pt : plane.points) (ln : plane.lines), plane.point_on_line pt ln) ∧
    (∀ a b c d : plane.points,
      ∀ l1 l2 l3 l4 : plane.lines,
      plane.point_on_line a l1 → plane.point_on_line b l1 →
      plane.point_on_line a l2 → plane.point_on_line d l2 →
      plane.point_on_line c l3 → plane.point_on_line b l3 →
      plane.point_on_line c l4 → plane.point_on_line d l4 →
      (a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_coloring_l1349_134971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_duration_l1349_134912

/-- Represents a time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts a time to total seconds -/
def timeToSeconds (t : Time) : ℕ :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Checks if the clock hands are in opposite directions at a given time -/
def handsOpposite (t : Time) : Prop :=
  let totalMinutes := t.hours * 60 + t.minutes + t.seconds / 60
  let hourAngle := (totalMinutes % 720) / 2
  let minuteAngle := totalMinutes % 60 * 6
  (hourAngle + 180) % 360 = minuteAngle

/-- The main theorem -/
theorem trip_duration :
  ∀ (start finish : Time),
    8 ≤ start.hours ∧ start.hours < 12 ∧
    start.hours < finish.hours ∧ finish.hours < 12 ∧
    handsOpposite start ∧ handsOpposite finish →
    timeToSeconds finish - timeToSeconds start = 3 * 3600 + 16 * 60 + 22 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_duration_l1349_134912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1349_134948

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 - 3*x else -2*x + 1

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ (M = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1349_134948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_groceries_expense_calculation_l1349_134960

def monthly_salary (savings : ℕ) (savings_percentage : ℚ) : ℚ :=
  (savings : ℚ) / savings_percentage

def total_expenses_without_groceries (rent milk education petrol misc : ℕ) : ℕ :=
  rent + milk + education + petrol + misc

def groceries_expense (salary : ℚ) (expenses_without_groceries savings : ℕ) : ℚ :=
  salary - ((expenses_without_groceries : ℚ) + (savings : ℚ))

theorem groceries_expense_calculation (rent milk education petrol misc savings : ℕ) 
  (savings_percentage : ℚ) (h1 : savings_percentage = 1/10) 
  (h2 : savings = 1800) (h3 : rent = 5000) (h4 : milk = 1500) 
  (h5 : education = 2500) (h6 : petrol = 2000) (h7 : misc = 700) :
  groceries_expense 
    (monthly_salary savings savings_percentage)
    (total_expenses_without_groceries rent milk education petrol misc)
    savings
  = 4500 := by
  sorry

#eval groceries_expense 
  (monthly_salary 1800 (1/10))
  (total_expenses_without_groceries 5000 1500 2500 2000 700)
  1800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_groceries_expense_calculation_l1349_134960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l1349_134983

-- Define the complex plane
variable (z : ℂ)

-- Define the equations
def equation1 (z : ℂ) : Prop := Complex.abs (z - 4) = 3 * Complex.abs (z + 4)
def equation2 (z : ℂ) (k : ℝ) : Prop := Complex.abs z = k

-- Define the intersection condition
def intersect_at_two_points (k : ℝ) : Prop :=
  ∃ (z1 z2 : ℂ), z1 ≠ z2 ∧ 
    equation1 z1 ∧ equation1 z2 ∧ 
    equation2 z1 k ∧ equation2 z2 k ∧
    ∀ (z : ℂ), equation1 z ∧ equation2 z k → (z = z1 ∨ z = z2)

-- State the theorem
theorem intersection_condition (k : ℝ) :
  intersect_at_two_points k → (28 < k ∧ k < 44) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l1349_134983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_of_Q_l1349_134996

-- Define the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point P
def P (m n : ℝ) : Prop := on_circle m n

-- Define point Q
def Q (x y m n : ℝ) : Prop := x = m - n ∧ y = 2 * m * n

-- Theorem statement
theorem path_of_Q (m n x y : ℝ) :
  P m n → Q x y m n → x^2 + y = 1 ∧ abs x ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_of_Q_l1349_134996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_lending_time_l1349_134927

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem money_lending_time : 
  ∃ (time : ℝ), 
    simple_interest 4000 8 time = 640 ∧ 
    time = 2 := by
  use 2
  constructor
  · simp [simple_interest]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_lending_time_l1349_134927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_implies_a_zero_l1349_134988

/-- The function f(x) = x² + a/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a/x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*x - a/x^2

theorem tangent_parallel_implies_a_zero (a : ℝ) :
  (f_derivative a 1 = 2) → a = 0 := by
  intro h
  -- The proof steps would go here
  sorry

#check tangent_parallel_implies_a_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_implies_a_zero_l1349_134988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1349_134949

def A : Finset ℕ := {1, 3, 5, 7, 9}
def B : Finset ℕ := {0, 3, 6, 9, 12}

theorem set_operations :
  ((A : Set ℕ) ∩ (Set.univ \ (B : Set ℕ)) = {1, 5, 7}) ∧
  (Finset.card (Finset.powerset (A ∪ B)) - 1 = 255) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1349_134949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1349_134967

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  t.a = 6 ∧
  2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b ∧
  1/2 * t.b * t.c * Real.sin t.A = 7/3 * Real.sqrt 3

-- Theorem statement
theorem triangle_perimeter (t : Triangle) 
  (h : triangle_conditions t) : t.a + t.b + t.c = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1349_134967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_implies_a_range_l1349_134925

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x - 3 else Real.log x / Real.log a

-- State the theorem
theorem function_monotonicity_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → 3 < a ∧ a ≤ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_implies_a_range_l1349_134925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_plus_two_sin_double_l1349_134990

theorem cos_squared_plus_two_sin_double (α : ℝ) :
  Real.tan (α + π/4) = -3 → (Real.cos α)^2 + 2 * Real.sin (2 * α) = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_plus_two_sin_double_l1349_134990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_lower_bound_l1349_134953

theorem cos_lower_bound (x : ℝ) (h : x ≥ 0) : Real.cos x ≥ 1 - (1/2) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_lower_bound_l1349_134953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_satisfies_conditions_l1349_134941

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 4 = 0

-- Define the line
def line_eq (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the chord length
noncomputable def chord_length (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem line_satisfies_conditions :
  -- The line passes through (1,1)
  line_eq 1 1 ∧
  -- There exist two distinct points on both the line and the circle
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    -- The chord length is 2√2
    chord_length x₁ y₁ x₂ y₂ = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_satisfies_conditions_l1349_134941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_primes_with_unique_digits_l1349_134969

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that returns the set of digits in a natural number -/
def digits (n : ℕ) : Finset ℕ :=
  if n = 0 then {0} else Finset.filter (λ d ↦ d < 10) (Finset.range (n + 1))

/-- The theorem stating the smallest possible sum of primes using digits 0-9 once -/
theorem smallest_sum_of_primes_with_unique_digits :
  ∃ (S : Finset ℕ),
    (∀ n ∈ S, isPrime n) ∧
    (Finset.sum (Finset.biUnion S digits) id = Finset.sum (Finset.range 10) id) ∧
    (∀ T : Finset ℕ,
      (∀ m ∈ T, isPrime m) →
      (Finset.sum (Finset.biUnion T digits) id = Finset.sum (Finset.range 10) id) →
      Finset.sum S id ≤ Finset.sum T id) ∧
    Finset.sum S id = 207 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_primes_with_unique_digits_l1349_134969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_reduce_to_one_l1349_134929

/-- The operation described in the problem -/
def adjacent_operation (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number can be reduced to 1 using the operation -/
def can_reduce_to_one (n : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (seq : ℕ → ℕ), 
    seq 0 = n ∧ 
    seq k = 1 ∧
    ∀ i : ℕ, i < k → seq (i + 1) = adjacent_operation (seq i)

/-- Theorem stating that any positive integer can be reduced to 1 -/
theorem all_reduce_to_one : ∀ n : ℕ, n > 0 → can_reduce_to_one n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_reduce_to_one_l1349_134929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_b_55_l1349_134905

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

def b_sequence_index (n : ℕ) : ℕ := n + sum_of_naturals (n - 1)

theorem a_10_equals_b_55 : arithmetic_sequence 10 = arithmetic_sequence (b_sequence_index 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_b_55_l1349_134905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_power_eq_y_power_x_l1349_134993

theorem lambda_power_eq_y_power_x (n : ℚ) :
  let x : ℝ := ((n + 1) / n : ℚ) ^ (n : ℝ)
  let y : ℝ := ((n + 1) / n : ℚ) ^ ((n + 1) : ℝ)
  Real.exp y = y ^ x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_power_eq_y_power_x_l1349_134993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l1349_134986

/-- A pair of integers (x, y) that satisfies x^2 + y^2 = 65 -/
def LatticePoint : Type := { p : ℤ × ℤ // p.1^2 + p.2^2 = 65 }

/-- The set of all lattice points (x, y) satisfying x^2 + y^2 = 65 -/
noncomputable def allLatticePoints : Finset LatticePoint := sorry

/-- A pair of real numbers (a, b) such that ax + by = 2 has at least one integer solution (x, y) satisfying x^2 + y^2 = 65 -/
def ValidPair : Type := { p : ℝ × ℝ // ∃ (lp : LatticePoint), p.1 * lp.val.1 + p.2 * lp.val.2 = 2 }

/-- The set of all valid pairs (a, b) -/
noncomputable def allValidPairs : Finset ValidPair := sorry

/-- Proof that allValidPairs is finite -/
instance : Fintype ValidPair := sorry

/-- The main theorem: there are exactly 128 valid pairs -/
theorem count_valid_pairs : Fintype.card ValidPair = 128 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l1349_134986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1349_134982

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > 0 → y > 0 → f (x + f y + x * y) = x * f y + f (x + y)) : 
  ∀ x : ℝ, x > 0 → f x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1349_134982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l1349_134995

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism
    to the volume of the prism, given that the base of the prism is a rectangle with sides 2r and 3r. -/
theorem cone_prism_volume_ratio (r h : ℝ) (r_pos : r > 0) (h_pos : h > 0) :
  (1 / 3 : ℝ) * π * r^2 * h / (2 * r * 3 * r * h) = π / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l1349_134995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l1349_134934

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) := (2*x + 1/Real.sqrt x)^n

theorem binomial_expansion_constant_term 
  (n : ℕ) 
  (h : ∃ (k : ℝ) (x : ℝ), binomial_expansion x n = k + x * (binomial_expansion x n)) :
  n = 6 ∧ 
  ∃ r, (n.choose r * 2^(n-r) : ℝ) = 160 ∧ 
    ∀ s, s ≠ r → (n.choose s : ℝ) ≤ (n.choose r : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l1349_134934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_coin_problem_l1349_134904

/-- Rachel's coin problem -/
theorem rachel_coin_problem :
  ∀ (n : ℕ), -- n represents the number of nickels
  (n ≥ 3) →  -- at least 3 nickels
  (3030 - n ≥ 10 * n) →  -- at least 10 times as many pennies as nickels
  (n ≤ 275) →  -- derived from the previous condition
  (3030 + 4 * 275) - (3030 + 4 * 3) = 1088 :=
by
  intro n h1 h2 h3
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_coin_problem_l1349_134904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_chord_length_theorem_l1349_134961

-- Define the circle
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

-- Define the point P
def point_P : ℝ × ℝ := (4, -1)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := 3*x + 4*y - 8 = 0

-- Define the line with slope 135°
def line_135 (x y : ℝ) : Prop := x + y - 3 = 0

-- Theorem for tangent lines
theorem tangent_lines_theorem :
  ∃ (x y : ℝ), (circle_C x y ∧ (tangent_line_1 x ∨ tangent_line_2 x y)) :=
sorry

-- Theorem for chord length
theorem chord_length_theorem :
  ∃ (x y : ℝ), (circle_C x y ∧ line_135 x y ∧
    (x - point_P.1)^2 + (y - point_P.2)^2 = 8) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_chord_length_theorem_l1349_134961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_implies_value_l1349_134974

-- Define IsMonomial as a predicate on real numbers
def IsMonomial (x : ℝ) : Prop := sorry

theorem monomial_sum_implies_value (a b x y : ℝ) : 
  IsMonomial (-2 * a^2 * b^(x+y) + 1/3 * a^x * b^5) → 
  (1/2 * x^3 - 1/6 * x * y^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_implies_value_l1349_134974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_two_no_six_l1349_134958

/-- The set of digits excluding 6 -/
def digits_no_six : Finset Nat := {0, 1, 2, 3, 4, 5, 7, 8, 9}

/-- The set of digits excluding 2 and 6 -/
def digits_no_two_six : Finset Nat := {0, 1, 3, 4, 5, 7, 8, 9}

/-- The set of non-zero digits excluding 6 -/
def non_zero_digits_no_six : Finset Nat := {1, 2, 3, 4, 5, 7, 8, 9}

/-- The set of non-zero digits excluding 2 and 6 -/
def non_zero_digits_no_two_six : Finset Nat := {1, 3, 4, 5, 7, 8, 9}

/-- A function that counts the number of three-digit integers without 6 -/
def count_no_six : Nat :=
  (non_zero_digits_no_six.card) * (digits_no_six.card) * (digits_no_six.card)

/-- A function that counts the number of three-digit integers without 2 and 6 -/
def count_no_two_six : Nat :=
  (non_zero_digits_no_two_six.card) * (digits_no_two_six.card) * (digits_no_two_six.card)

/-- The main theorem -/
theorem count_integers_with_two_no_six :
  count_no_six - count_no_two_six = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_two_no_six_l1349_134958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1349_134942

theorem divisibility_condition (x p : ℕ) (h_prime : Nat.Prime p) (h_bound : x ≤ 2*p) :
  (x^(p-1) ∣ (p-1)^x + 1) ↔ 
  ((x = 1 ∧ p = 2) ∨ 
   (x = 2 ∧ p = 2) ∨ 
   (x = 1) ∨ 
   (x = 3 ∧ p = 3)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1349_134942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_selection_count_l1349_134998

-- Define the number of courses and students
def num_courses : ℕ := 4
def num_students : ℕ := 4

-- Define the number of courses with no students
def empty_courses : ℕ := 2

-- Define a course selection scheme
def CourseSelection := Fin num_students → Fin num_courses

-- Define a predicate for valid course selections
def is_valid_selection (selection : CourseSelection) : Prop :=
  (∃ (c1 c2 : Fin num_courses), ∀ s, selection s ≠ c1 ∧ selection s ≠ c2) ∧
  (∀ c, ∃ s, selection s = c)

-- Provide instances for Fintype and DecidablePred
instance : Fintype CourseSelection := by sorry

instance : DecidablePred is_valid_selection := by sorry

-- State the theorem
theorem course_selection_count :
  (Finset.filter is_valid_selection (Finset.univ : Finset CourseSelection)).card = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_selection_count_l1349_134998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_when_area_twice_perimeter_l1349_134901

-- Define Triangle as a structure
structure Triangle where
  area : ℝ
  perimeter : ℝ
  inradius : ℝ

theorem inscribed_circle_radius_when_area_twice_perimeter 
  (T : Triangle) 
  (h : T.area = 2 * T.perimeter) : 
  T.inradius = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_when_area_twice_perimeter_l1349_134901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_max_value_min_value_l1349_134926

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + Real.sqrt 3 * cos x * cos (π / 2 - x)

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 7 * π / 12 }

-- Theorem for the axis of symmetry
theorem axis_of_symmetry (k : ℤ) :
  ∀ x, f (π / 3 + k * π / 2 + x) = f (π / 3 + k * π / 2 - x) :=
by sorry

-- Theorem for the maximum value
theorem max_value :
  ∃ x ∈ domain, ∀ y ∈ domain, f y ≤ f x ∧ f x = 3 / 2 :=
by sorry

-- Theorem for the minimum value
theorem min_value :
  ∃ x ∈ domain, ∀ y ∈ domain, f x ≤ f y ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_max_value_min_value_l1349_134926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_third_quadrant_fraction_value_given_tan_l1349_134902

-- Part 1
theorem sin_value_third_quadrant (α : ℝ) (h1 : Real.cos α = -4/5) (h2 : π < α ∧ α < 3*π/2) :
  Real.sin α = -3/5 := by sorry

-- Part 2
theorem fraction_value_given_tan (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin θ + Real.cos θ) / (2 * Real.sin θ + Real.cos θ) = 4/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_third_quadrant_fraction_value_given_tan_l1349_134902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_equation_proof_l1349_134962

-- Define the exponential function
noncomputable def exp_curve (k a x : ℝ) : ℝ := Real.exp (k * x + a)

-- Define the logarithmic transformation
noncomputable def z (y : ℝ) : ℝ := Real.log y

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.25 * x - 2.58

-- Theorem statement
theorem regression_equation_proof :
  ∀ (x y : ℝ), y = exp_curve 0.25 (-2.58) x ↔ z y = regression_line x :=
by
  intro x y
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_equation_proof_l1349_134962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_range_l1349_134978

/-- Profit function for agricultural product sales --/
noncomputable def profit (x : ℝ) : ℝ :=
  if 100 ≤ x ∧ x ≤ 130 then 800 * x - 39000
  else if 130 < x ∧ x ≤ 150 then 65000
  else 0

/-- Theorem stating the range of market demand for profit ≥ 57000 --/
theorem profit_range (x : ℝ) :
  (100 ≤ x ∧ x ≤ 150) → (profit x ≥ 57000 ↔ 120 ≤ x ∧ x ≤ 150) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_range_l1349_134978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_angle_l1349_134946

/-- A convex polygon with n sides has a sum of interior angles equal to (n-2) * 180 degrees --/
axiom interior_angle_sum (n : ℕ) : n ≥ 3 → (n - 2) * 180 = List.sum (List.range n)

/-- The sum of the remaining angles after one is erased is 1703 degrees --/
axiom remaining_sum : ∃ (angles : List ℕ), List.sum angles = 1703 ∧ angles.length ≥ 2

/-- The erased angle is 97 degrees --/
theorem erased_angle : 
  ∃ (n : ℕ) (angles : List ℕ), 
    n ≥ 3 ∧ 
    angles.length = n - 1 ∧ 
    List.sum angles = 1703 ∧ 
    (n - 2) * 180 = List.sum angles + 97 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_angle_l1349_134946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passion_fruit_profit_l1349_134955

/-- Represents the daily data for passion fruit sales -/
structure DailyData where
  price_diff : Int
  quantity : Int

/-- Calculates the total profit from passion fruit sales -/
def calculate_profit (data : List DailyData) (standard_price cost_price : Int) : Int :=
  let price_diff_sum := data.foldr (fun d acc => acc + d.price_diff * d.quantity) 0
  let standard_profit := (standard_price - cost_price) * (data.foldr (fun d acc => acc + d.quantity) 0)
  price_diff_sum + standard_profit

/-- The main theorem stating the profit calculation -/
theorem passion_fruit_profit :
  let data : List DailyData := [
    ⟨1, 20⟩, ⟨-2, 35⟩, ⟨3, 10⟩, ⟨-1, 30⟩, ⟨2, 15⟩, ⟨5, 5⟩, ⟨-4, 50⟩
  ]
  let standard_price := 10
  let cost_price := 8
  calculate_profit data standard_price cost_price = 135 := by
  sorry

#eval calculate_profit [
  ⟨1, 20⟩, ⟨-2, 35⟩, ⟨3, 10⟩, ⟨-1, 30⟩, ⟨2, 15⟩, ⟨5, 5⟩, ⟨-4, 50⟩
] 10 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_passion_fruit_profit_l1349_134955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1349_134914

/-- A parabola with vertex at the origin and axis of symmetry on the x-axis -/
structure Parabola where
  /-- The equation of the parabola in the form y² = 2px -/
  p : ℝ

/-- The focus of a parabola -/
noncomputable def Parabola.focus (para : Parabola) : ℝ × ℝ := (para.p / 2, 0)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_equation (para : Parabola) 
  (point_on_parabola : (-5 : ℝ)^2 * (2 * para.p) = (2 * Real.sqrt 5)^2)
  (focus_distance : distance (-5, 2 * Real.sqrt 5) (para.focus) = 6) :
  para.p = -2 ∨ para.p = -18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1349_134914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_figure_area_l1349_134975

/-- Represents a trapezoid in oblique view -/
structure ObliqueTrapezoid where
  baseAngle : ℝ
  leg : ℝ
  topBase : ℝ

/-- Represents the original horizontally placed figure -/
structure OriginalFigure where
  topBase : ℝ
  bottomBase : ℝ
  height : ℝ

/-- Calculates the area of the original figure -/
noncomputable def area (fig : OriginalFigure) : ℝ :=
  (fig.topBase + fig.bottomBase) * fig.height / 2

/-- The main theorem -/
theorem original_figure_area
  (oblique : ObliqueTrapezoid)
  (original : OriginalFigure)
  (h1 : oblique.baseAngle = π/4)  -- 45 degrees in radians
  (h2 : oblique.leg = 1)
  (h3 : oblique.topBase = 1)
  (h4 : original.topBase = 1)
  (h5 : original.height = 2)
  (h6 : original.bottomBase = 1 + Real.sqrt 2) :
  area original = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_figure_area_l1349_134975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_equilateral_triangle_area_ratio_l1349_134917

/-- Given an equilateral triangle DEF with side length s, if each side is extended by 4s
    to form triangle D'E'F', then the ratio of the area of D'E'F' to the area of DEF is 25. -/
theorem extended_equilateral_triangle_area_ratio (s : ℝ) (h : s > 0) :
  let triangle_DEF : Set (ℝ × ℝ) := {p | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    (p = (1 - t) • (0, 0) + t • (s, 0) ∨
     p = (1 - t) • (s, 0) + t • (s/2, s*Real.sqrt 3/2) ∨
     p = (1 - t) • (s/2, s*Real.sqrt 3/2) + t • (0, 0))}
  let triangle_DEF' : Set (ℝ × ℝ) := {p | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    (p = (1 - t) • (0, 0) + t • (5*s, 0) ∨
     p = (1 - t) • (5*s, 0) + t • (5*s/2, 5*s*Real.sqrt 3/2) ∨
     p = (1 - t) • (5*s/2, 5*s*Real.sqrt 3/2) + t • (0, 0))}
  let area (triangle : Set (ℝ × ℝ)) : ℝ := sorry
  area triangle_DEF' / area triangle_DEF = 25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_equilateral_triangle_area_ratio_l1349_134917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1349_134997

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (2*x + 1)/(x - 2) + (x - 3)/(3*x)

-- Define the solution set
def solution_set : Set ℝ := Set.union (Set.Ioo (-2/5 : ℝ) 0) (Set.Ico 2 3)

-- Theorem statement
theorem inequality_solution : 
  {x : ℝ | f x ≥ 4} = solution_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1349_134997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1349_134956

noncomputable section

-- Define the triangle ABC
def Triangle (A B C : ℝ) : Prop := 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define vectors m and n
def m (A : ℝ) : ℝ × ℝ := (Real.cos A + 1, Real.sqrt 3)
def n (A : ℝ) : ℝ × ℝ := (Real.sin A, 1)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

-- Main theorem
theorem triangle_problem (A B C : ℝ) :
  Triangle A B C →
  parallel (m A) (n A) →
  (1 + Real.sin (2 * B)) / (Real.cos B ^ 2 - Real.sin B ^ 2) = -3 →
  A = Real.pi / 3 ∧ Real.tan C = (8 + 5 * Real.sqrt 3) / 11 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1349_134956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_square_roots_l1349_134987

theorem simplify_square_roots : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_square_roots_l1349_134987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l1349_134959

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def IsPerp (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of a line in the form y = mx + c is m -/
def slope_of_line (m c : ℝ) : ℝ := m

/-- The slope of a line in the form ay + bx = c is -b/a -/
noncomputable def slope_of_general_line (a b c : ℝ) : ℝ := -b / a

theorem perpendicular_lines_b_value :
  ∀ b : ℝ,
  IsPerp (slope_of_line (-3) 7) (slope_of_general_line 9 b 18) →
  b = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l1349_134959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_school_time_l1349_134951

noncomputable def total_time (walking_time : ℝ) (walking_distance : ℝ) (total_distance : ℝ) (speed_ratio : ℝ) : ℝ :=
  walking_time + (total_distance - walking_distance) / (speed_ratio * (walking_distance / walking_time))

theorem joe_school_time :
  let walking_time : ℝ := 9
  let walking_distance : ℝ := 1/3
  let total_distance : ℝ := 1
  let speed_ratio : ℝ := 4
  total_time walking_time walking_distance total_distance speed_ratio = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_school_time_l1349_134951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_positive_period_l1349_134920

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos (2 * x) + 3

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  is_periodic f T ∧ T > 0 ∧ ∀ T', is_periodic f T' ∧ T' > 0 → T ≤ T'

theorem f_minimum_positive_period :
  minimum_positive_period f π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_positive_period_l1349_134920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stellas_profit_is_negative_17_25_l1349_134994

/-- Represents the inventory and pricing of items in Stella's antique shop -/
structure ShopInventory where
  dolls : Nat
  clocks : Nat
  glasses : Nat
  vases : Nat
  postcards : Nat
  doll_price : Nat
  clock_price : Nat
  glass_price : Nat
  vase_price : Nat
  postcard_price : Nat

/-- Calculates the revenue after applying discounts -/
def revenue_after_discounts (inventory : ShopInventory) : Rat :=
  let doll_revenue := inventory.dolls * inventory.doll_price
  let clock_revenue := inventory.clocks * inventory.clock_price * 95 / 100  -- 5% discount on all clocks
  let glass_revenue := (inventory.glasses - inventory.glasses / 3) * inventory.glass_price  -- buy 2 get 1 free
  let vase_revenue := inventory.vases * inventory.vase_price
  let postcard_revenue := inventory.postcards * inventory.postcard_price
  (doll_revenue + clock_revenue + glass_revenue + vase_revenue + postcard_revenue : Rat)

/-- Calculates the profit after discounts and sales tax -/
def profit_after_tax (inventory : ShopInventory) (cost : Nat) (tax_rate : Rat) : Rat :=
  let revenue := revenue_after_discounts inventory
  let tax := revenue * tax_rate
  revenue - cost - tax

/-- Theorem stating that Stella's profit is -$17.25 -/
theorem stellas_profit_is_negative_17_25 
  (inventory : ShopInventory)
  (h_dolls : inventory.dolls = 6)
  (h_clocks : inventory.clocks = 4)
  (h_glasses : inventory.glasses = 8)
  (h_vases : inventory.vases = 3)
  (h_postcards : inventory.postcards = 10)
  (h_doll_price : inventory.doll_price = 8)
  (h_clock_price : inventory.clock_price = 25)
  (h_glass_price : inventory.glass_price = 6)
  (h_vase_price : inventory.vase_price = 12)
  (h_postcard_price : inventory.postcard_price = 3)
  (h_cost : Nat := 250)
  (h_tax_rate : Rat := 5 / 100) :
  profit_after_tax inventory h_cost h_tax_rate = -17.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stellas_profit_is_negative_17_25_l1349_134994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_I_maximum_mark_l1349_134932

theorem paper_I_maximum_mark :
  let pass_percentage : ℚ := 65 / 100
  let secured_marks : ℕ := 112
  let failed_by : ℕ := 58
  let passing_marks : ℕ := secured_marks + failed_by
  let maximum_mark : ℕ := (passing_marks : ℚ) / pass_percentage |>.ceil.toNat
  maximum_mark = 262 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_I_maximum_mark_l1349_134932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_four_minus_sin_four_l1349_134910

theorem cos_four_minus_sin_four (α : ℝ) (h : Real.sin α = Real.sqrt 5 / 5) :
  Real.cos α ^ 4 - Real.sin α ^ 4 = 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_four_minus_sin_four_l1349_134910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_ceiling_evaluation_l1349_134900

theorem complex_fraction_ceiling_evaluation :
  (⌈(19 : ℝ) / 8 - ⌈(35 : ℝ) / 19⌉⌉) / (⌈(35 : ℝ) / 8 + ⌈(8 : ℝ) * 19 / 35⌉⌉) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_ceiling_evaluation_l1349_134900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l1349_134913

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + Real.sqrt (a * x^2 + 1))

-- State the theorem
theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l1349_134913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_floor_equation_ten_is_solution_ten_is_smallest_solution_l1349_134916

theorem smallest_solution_floor_equation :
  ∀ x : ℝ, (⌊x⌋ : ℝ) = 10 + 150 * (x - ⌊x⌋) → x ≥ 10 :=
by sorry

theorem ten_is_solution :
  (⌊(10 : ℝ)⌋ : ℝ) = 10 + 150 * ((10 : ℝ) - ⌊(10 : ℝ)⌋) :=
by sorry

theorem ten_is_smallest_solution :
  ∀ x : ℝ, (⌊x⌋ : ℝ) = 10 + 150 * (x - ⌊x⌋) → x = 10 ∨ x > 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_floor_equation_ten_is_solution_ten_is_smallest_solution_l1349_134916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1349_134940

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/4) * a * x^4 - (1/2) * x^2

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ t : ℝ, |f_prime a (t + 2) - f_prime a t| ≤ 1/4) →
  (∃ a_max : ℝ, a ≤ a_max ∧ a_max = 9/8) ∧
  (∃ x : ℝ, f (9/8) x = -2/9) ∧
  (∀ x : ℝ, f (9/8) x ≥ -2/9) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1349_134940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_pi_thirds_l1349_134903

theorem cos_five_pi_thirds : Real.cos (5 * π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_pi_thirds_l1349_134903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_problem_l1349_134977

/-- The number of marbles that Ajay needs to receive from Vijay to have an equal number of marbles -/
def x : ℤ := sorry

/-- The number of marbles Ajay has -/
def A : ℤ := sorry

/-- The number of marbles Vijay has -/
def V : ℤ := sorry

/-- The number of marbles Rajesh has -/
def R : ℤ := sorry

/-- If Ajay gets 'x' marbles from Vijay, both will have an equal number of marbles -/
axiom condition1 : A + x = V - x

/-- If Ajay gives Vijay twice as many marbles, Vijay will have 30 more marbles than Ajay -/
axiom condition2 : V + 2*x = A - 2*x + 30

/-- If Rajesh gives half of his marbles to both Ajay and Vijay, the total number of marbles shared among them would be 120 -/
axiom condition3 : A + V + R = 120

/-- The value of 'x' is 15 -/
theorem marble_problem : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_problem_l1349_134977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_flight_prob_cube_3_l1349_134965

/-- Represents a cube with a given edge length -/
structure Cube where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- Represents the volume of a cube -/
def cube_volume (c : Cube) : ℝ := c.edge_length ^ 3

/-- Represents the safe region inside a cube -/
def safe_region (c : Cube) (safety_distance : ℝ) : Cube where
  edge_length := c.edge_length - 2 * safety_distance
  edge_length_pos := by
    sorry -- Proof that the new edge length is positive

/-- The probability of safe flight in a cube -/
noncomputable def safe_flight_probability (c : Cube) (safety_distance : ℝ) : ℝ :=
  cube_volume (safe_region c safety_distance) / cube_volume c

/-- Theorem stating the probability of safe flight in a cube with edge length 3 and safety distance 1 -/
theorem safe_flight_prob_cube_3 :
  let c : Cube := ⟨3, by norm_num⟩
  safe_flight_probability c 1 = 1 / 27 := by
    sorry -- Proof of the theorem


end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_flight_prob_cube_3_l1349_134965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_triple_with_prime_differences_l1349_134979

theorem prime_triple_with_prime_differences :
  ∀ p q r : ℤ,
  Nat.Prime p.natAbs ∧ Nat.Prime q.natAbs ∧ Nat.Prime r.natAbs →
  Nat.Prime (|p - q|).natAbs ∧ Nat.Prime (|q - r|).natAbs ∧ Nat.Prime (|r - p|).natAbs →
  (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 5 ∧ q = 2 ∧ r = 7) ∨ 
  (p = 7 ∧ q = 2 ∧ r = 5) ∨ (p = 2 ∧ q = 7 ∧ r = 5) ∨ 
  (p = 5 ∧ q = 7 ∧ r = 2) ∨ (p = 7 ∧ q = 5 ∧ r = 2) :=
by sorry

#check prime_triple_with_prime_differences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_triple_with_prime_differences_l1349_134979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_side_ratio_l1349_134923

/-- Definition of a triangle -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Definition of similarity between triangles -/
def Similar (T1 T2 : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ T2.a = k * T1.a ∧ T2.b = k * T1.b ∧ T2.c = k * T1.c

/-- Definition of perimeter of a triangle -/
def Perimeter (T : Triangle) : ℝ := T.a + T.b + T.c

/-- Definition of sides of a triangle -/
def Sides (T : Triangle) : Set ℝ := {T.a, T.b, T.c}

/-- Definition of corresponding sides in similar triangles -/
def Corresponding (T1 T2 : Triangle) (s1 s2 : ℝ) : Prop :=
  (s1 = T1.a ∧ s2 = T2.a) ∨ (s1 = T1.b ∧ s2 = T2.b) ∨ (s1 = T1.c ∧ s2 = T2.c)

/-- Two similar triangles with perimeter ratio 1:4 have side ratio 1:4 -/
theorem similar_triangles_perimeter_side_ratio :
  ∀ (T1 T2 : Triangle) (P1 P2 : ℝ),
  Similar T1 T2 →
  P1 = Perimeter T1 →
  P2 = Perimeter T2 →
  P1 / P2 = 1 / 4 →
  ∀ (s1 s2 : ℝ),
  s1 ∈ Sides T1 →
  s2 ∈ Sides T2 →
  Corresponding T1 T2 s1 s2 →
  s1 / s2 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_side_ratio_l1349_134923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_art_collection_volume_l1349_134980

/-- The volume of the art collection at Greenville State University -/
theorem art_collection_volume 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (box_cost : ℝ) 
  (min_spent : ℝ) 
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 15)
  (h4 : box_cost = 0.5)
  (h5 : min_spent = 255) :
  box_length * box_width * box_height * (min_spent / box_cost) = 3060000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_art_collection_volume_l1349_134980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1349_134947

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ (∀ (x : ℝ), f x ≤ M) ∧ (∃ (y : ℝ), f y = M)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1349_134947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_number_l1349_134944

def game_sequence (n : ℕ) : ℕ := 6 * 2^n - 4

theorem smallest_winning_number :
  ∀ N : ℕ,
    Even N ∧
    (∀ k < N, Even k → (game_sequence k).succ < 2015) →
    game_sequence 1007 = N ∧
    (game_sequence 1007).succ ≥ 2015 := by
  sorry

#check smallest_winning_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_number_l1349_134944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l1349_134966

noncomputable section

open Real

-- Define the function f
def f (α : ℝ) : ℝ :=
  (sin (α - π/2) * cos (3*π/2 + α) * tan (π - α)) /
  (tan (-α - π) * sin (-α - π))

-- Theorem statement
theorem f_simplification_and_value (α : ℝ) 
  (h1 : π < α ∧ α < 3*π/2) -- α is in the third quadrant
  (h2 : cos (α - 3*π/2) = 1/5) :
  (f α = -cos α) ∧ 
  (f α = 2 * sqrt 6 / 5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l1349_134966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l1349_134936

-- Define the function f
noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- State the theorem
theorem phi_value
  (A ω φ : ℝ)
  (x₀ : ℝ)
  (h1 : A > 0)
  (h2 : ω > 0)
  (h3 : |φ| < Real.pi / 2)
  (h4 : f A ω φ x₀ = 0)  -- x₀ is a zero point
  (h5 : ∀ x, f A ω φ x = f A ω φ (-x - Real.pi / 4))  -- -π/8 is a symmetry axis
  (h6 : ∀ x, |x + Real.pi / 8| ≥ Real.pi / 4)  -- Minimum value of |x₀ + π/8| is π/4
  (h7 : ∃ x, |x + Real.pi / 8| = Real.pi / 4)  -- The minimum is achieved
  : φ = -Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l1349_134936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l1349_134963

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x^2 + x - Real.log x

-- Define the line
def line (x y : ℝ) : Prop := 2*x - y - 2 = 0

-- State the theorem
theorem shortest_distance_curve_to_line :
  ∃ (d : ℝ), d = 2 * Real.sqrt 5 / 5 ∧
  ∀ (P : ℝ × ℝ), curve P.fst = P.snd →
    ∀ (Q : ℝ × ℝ), line Q.fst Q.snd →
      d ≤ Real.sqrt ((P.fst - Q.fst)^2 + (P.snd - Q.snd)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l1349_134963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_purchase_problem_l1349_134911

theorem corn_purchase_problem (corn_price bean_price total_weight total_cost : ℚ) 
  (h1 : corn_price = 99/100)
  (h2 : bean_price = 51/100)
  (h3 : total_weight = 22)
  (h4 : total_cost = 2013/100) :
  ∃ (corn_weight : ℚ), 
    corn_weight + (total_weight - corn_weight) = total_weight ∧
    corn_price * corn_weight + bean_price * (total_weight - corn_weight) = total_cost ∧
    (corn_weight * 10).floor / 10 = 186/10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_purchase_problem_l1349_134911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smart_integer_div_by_5_fraction_l1349_134909

def sum_of_digits (n : ℕ) : ℕ :=
  sorry  -- Implementation of sum of digits

def is_smart_integer (n : ℕ) : Prop :=
  n % 2 = 0 ∧ 30 < n ∧ n < 150 ∧ (sum_of_digits n = 10)

def count_smart_integers : ℕ :=
  sorry  -- Count of smart integers

def count_smart_integers_div_by_5 : ℕ :=
  sorry  -- Count of smart integers divisible by 5

theorem smart_integer_div_by_5_fraction :
  (count_smart_integers_div_by_5 : ℚ) / count_smart_integers = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smart_integer_div_by_5_fraction_l1349_134909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1349_134918

noncomputable section

-- Define the revenue function
def R (x : ℝ) : ℝ :=
  if x ≤ 40 then 40 * x - 0.5 * x^2
  else 1500 - 25000 / x

-- Define the total cost function
def TotalCost (x : ℝ) : ℝ := 2 + 0.1 * x

-- Define the profit function
def f (x : ℝ) : ℝ := R x - TotalCost x

-- State the theorem
theorem max_profit :
  ∃ (x : ℝ), x > 0 ∧ f x = 300 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1349_134918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequalities_l1349_134924

open Real

-- Define the function f(x) = sin(x) / x
noncomputable def f (x : ℝ) : ℝ := sin x / x

theorem sin_inequalities :
  -- f(x) is monotonically decreasing on (0, π/2)
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/2 → f x₁ > f x₂) ∧
  -- For 0 < x < π/4, sin(x) > (2√2/π)x
  (∀ x, 0 < x ∧ x < π/4 → sin x > (2 * Real.sqrt 2 / π) * x) ∧
  -- For 0 < x < π/4, sin(x) < √(2x/π)
  (∀ x, 0 < x ∧ x < π/4 → sin x < Real.sqrt ((2 * x) / π)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequalities_l1349_134924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarterly_to_annual_rate_l1349_134937

/-- The effective annual rate for quarterly compounding -/
noncomputable def effective_annual_rate (nominal_rate : ℝ) : ℝ :=
  (1 + nominal_rate / 4) ^ 4 - 1

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem quarterly_to_annual_rate : 
  round_to_hundredth (100 * effective_annual_rate 0.08) = 8.24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarterly_to_annual_rate_l1349_134937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l1349_134938

-- Define the domain for the functions (non-zero real numbers)
def NonZeroReal : Type := {x : ℝ // x ≠ 0}

-- Define the two functions
noncomputable def f (x : NonZeroReal) : ℝ := 1 / |x.val|
noncomputable def g (x : NonZeroReal) : ℝ := 1 / Real.sqrt (x.val^2)

-- State the theorem
theorem f_equals_g : ∀ (x : NonZeroReal), f x = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l1349_134938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_honda_production_ratio_l1349_134930

/-- Represents the production ratio problem at Honda -/
theorem honda_production_ratio 
  (total_production : ℕ) 
  (second_shift_production : ℕ) 
  (day_shift_multiple : ℚ)
  (h1 : total_production = 5500)
  (h2 : second_shift_production = 1100)
  (h3 : total_production = second_shift_production + (day_shift_multiple * second_shift_production).floor) :
  day_shift_multiple = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_honda_production_ratio_l1349_134930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_pi_thirds_minus_alpha_l1349_134939

theorem cos_two_pi_thirds_minus_alpha (α : ℝ) : 
  Real.sin (π / 6 - α) = 1 / 3 → Real.cos (2 * π / 3 - α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_pi_thirds_minus_alpha_l1349_134939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_or_yellow_is_half_l1349_134976

-- Define the spinner
structure Spinner :=
  (sections : Fin 4)
  (green_angle : ℚ)
  (blue_angle : ℚ)

-- Define the properties of the spinner
def spinner_properties (s : Spinner) : Prop :=
  s.green_angle = 90 ∧ s.blue_angle = 90

-- Define the probability of landing on Red or Yellow
noncomputable def prob_red_or_yellow (s : Spinner) : ℚ :=
  1 - (s.green_angle + s.blue_angle) / 360

-- Theorem statement
theorem prob_red_or_yellow_is_half (s : Spinner) 
  (h : spinner_properties s) : 
  prob_red_or_yellow s = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_or_yellow_is_half_l1349_134976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_weight_l1349_134907

theorem new_student_weight (initial_students : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (new_weight : ℝ) :
  initial_students = 19 →
  initial_avg = 15 →
  new_avg = 14.4 →
  (initial_students : ℝ) * initial_avg + new_weight = (initial_students + 1 : ℝ) * new_avg →
  new_weight = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_weight_l1349_134907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1349_134922

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 5 * Real.sqrt (x - 1) + Real.sqrt (10 - 2 * x)

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 1 5 ∧ 
  (∀ (y : ℝ), y ∈ Set.Icc 1 5 → f y ≤ f x) ∧
  f x = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1349_134922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_implies_a_bound_l1349_134908

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.sqrt ((1 + x) * (2 - x))
noncomputable def g (a x : ℝ) := Real.log (x - a)

-- Define the domains A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | a < x}

-- State the theorem
theorem domain_intersection_implies_a_bound (a : ℝ) : 
  (∃ x, x ∈ A ∩ B a) → a < 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_implies_a_bound_l1349_134908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_quarter_revenue_equation_l1349_134933

/-- Represents the monthly growth rate of revenue -/
def x : ℝ := sorry

/-- Represents the January revenue in hundreds of thousands of yuan -/
def january_revenue : ℝ := 400

/-- Represents the total first quarter revenue in hundreds of thousands of yuan -/
def first_quarter_revenue : ℝ := 2000

/-- Theorem stating that the equation representing the total revenue of the first quarter
    is 400 + 400(1+x) + 400(1+x)^2 = 2000, given the January revenue and monthly growth rate -/
theorem first_quarter_revenue_equation :
  january_revenue + january_revenue * (1 + x) + january_revenue * (1 + x)^2 = first_quarter_revenue := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_quarter_revenue_equation_l1349_134933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_finite_values_l1349_134935

/-- Product of digits of a natural number in decimal system -/
def P (n : ℕ) : ℕ := sorry

/-- The sequence defined by n_{k+1} = n_k + P(n_k) -/
def sequenceP (n₁ : ℕ) : ℕ → ℕ
  | 0 => n₁
  | k + 1 => sequenceP n₁ k + P (sequenceP n₁ k)

/-- The set of distinct values in the sequence -/
def distinctValues (n₁ : ℕ) : Set ℕ :=
  {n | ∃ k, sequenceP n₁ k = n}

/-- The main theorem: the sequence takes only finitely many distinct values -/
theorem sequence_finite_values (n₁ : ℕ) :
  (distinctValues n₁).Finite := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_finite_values_l1349_134935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_nested_evaluation_l1349_134950

-- Define the function p
noncomputable def p (x y : ℝ) : ℝ :=
  if x ≥ 0 ∧ y ≥ 0 then 2*x + 3*y
  else if x < 0 ∧ y < 0 then x^2 - y
  else 4*x + 2*y

-- State the theorem
theorem p_nested_evaluation :
  p (p 2 (-3)) (p (-4) (-3)) = 61 := by
  -- Evaluate p(2, -3)
  have h1 : p 2 (-3) = 2 := by
    rw [p]
    simp [if_neg]
    norm_num
  
  -- Evaluate p(-4, -3)
  have h2 : p (-4) (-3) = 19 := by
    rw [p]
    simp [if_pos]
    norm_num

  -- Evaluate the final p(2, 19)
  rw [h1, h2]
  rw [p]
  simp [if_pos]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_nested_evaluation_l1349_134950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_50_59_approx_17_14_l1349_134931

/-- Represents the frequency distribution of test scores -/
structure ScoreDistribution where
  scores_90_100 : ℕ
  scores_80_89 : ℕ
  scores_70_79 : ℕ
  scores_60_69 : ℕ
  scores_50_59 : ℕ
  scores_below_50 : ℕ

/-- Calculates the total number of students -/
def totalStudents (dist : ScoreDistribution) : ℕ :=
  dist.scores_90_100 + dist.scores_80_89 + dist.scores_70_79 + 
  dist.scores_60_69 + dist.scores_50_59 + dist.scores_below_50

/-- Calculates the percentage of students in a specific score range -/
noncomputable def percentageInRange (dist : ScoreDistribution) (studentsInRange : ℕ) : ℝ :=
  (studentsInRange : ℝ) / (totalStudents dist : ℝ) * 100

theorem percentage_50_59_approx_17_14 (dist : ScoreDistribution) 
  (h : dist = { scores_90_100 := 3, scores_80_89 := 7, scores_70_79 := 10,
                scores_60_69 := 5, scores_50_59 := 6, scores_below_50 := 4 }) : 
  ∃ ε > 0, |percentageInRange dist dist.scores_50_59 - 17.14| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_50_59_approx_17_14_l1349_134931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_2_lt_X_le_4_eq_three_sixteenths_l1349_134906

/-- A random variable X with probability distribution P(X = k) = 1 / (2^k) for k = 1, 2, ... -/
noncomputable def X (k : ℕ) : ℝ := 1 / (2 ^ k)

/-- The probability that 2 < X ≤ 4 -/
noncomputable def prob_2_lt_X_le_4 : ℝ := X 3 + X 4

theorem prob_2_lt_X_le_4_eq_three_sixteenths :
  prob_2_lt_X_le_4 = 3/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_2_lt_X_le_4_eq_three_sixteenths_l1349_134906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_4_l1349_134964

theorem tan_alpha_minus_pi_4 (α : Real) (h1 : α ∈ Set.Ioo (π/2) π) (h2 : Real.sin α = 3/5) :
  Real.tan (α - π/4) = -7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_4_l1349_134964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_equals_two_l1349_134921

-- Define a function f with an inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the properties of f and its inverse
axiom f_has_inverse : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Define the given conditions
axiom f_3 : f 3 = 4
axiom f_5 : f 5 = 1
axiom f_2 : f 2 = 5

-- State the theorem to be proved
theorem inverse_composition_equals_two :
  f_inv (f_inv 5 + f_inv 4) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_equals_two_l1349_134921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_odd_implies_a_equals_one_l1349_134985

/-- Given a real number a and a function f(x) = e^x - ae^(-x) whose second derivative
    is an odd function, prove that a = 1. -/
theorem second_derivative_odd_implies_a_equals_one (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = Real.exp x - a * Real.exp (-x)) ∧
   (∃ f' : ℝ → ℝ, ∀ x, HasDerivAt f (f' x) x) ∧
   (∃ f'' : ℝ → ℝ, (∀ x, HasDerivAt f' (f'' x) x) ∧
   (∀ x, f'' (-x) = -f'' x))) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_odd_implies_a_equals_one_l1349_134985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_sum_of_digits_l1349_134984

def is_single_digit_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ p < 10

def is_sum_of_squares_prime (d e : ℕ) : Prop :=
  Nat.Prime (d^2 + e^2)

def largest_product (d e : ℕ) : ℕ :=
  d * e * (d^2 + e^2)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_product_sum_of_digits :
  ∃ (d e : ℕ),
    is_single_digit_prime d ∧
    is_single_digit_prime e ∧
    d < e ∧
    is_sum_of_squares_prime d e ∧
    (∀ (d' e' : ℕ),
      is_single_digit_prime d' ∧
      is_single_digit_prime e' ∧
      d' < e' ∧
      is_sum_of_squares_prime d' e' →
      largest_product d' e' ≤ largest_product d e) ∧
    sum_of_digits (largest_product d e) = 13 :=
  sorry

#eval sum_of_digits 742

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_sum_of_digits_l1349_134984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l1349_134989

def sequence_a : ℕ → ℚ
  | 0 => 1  -- We define a_0 as 1 to handle the base case
  | 1 => 3
  | n+2 => sequence_a (n+1) + 1 / sequence_a n

theorem a_5_value : sequence_a 5 = 55 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l1349_134989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iggy_running_time_l1349_134952

/-- Represents the distances Iggy runs each day of the week in miles -/
def weekly_distances : List ℚ := [3, 4, 6, 8, 3]

/-- Iggy's pace in minutes per mile -/
def pace : ℚ := 10

/-- Calculates the total time Iggy spends running in a week in hours -/
def total_running_time (distances : List ℚ) (pace : ℚ) : ℚ :=
  (distances.sum * pace) / 60

/-- Proves that Iggy's total running time for the week is 4 hours -/
theorem iggy_running_time :
  total_running_time weekly_distances pace = 4 := by
  -- Unfold the definition of total_running_time
  unfold total_running_time
  -- Simplify the sum of weekly_distances
  simp [weekly_distances, pace]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_iggy_running_time_l1349_134952
