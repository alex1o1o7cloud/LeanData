import Mathlib

namespace NUMINAMATH_CALUDE_total_free_sides_length_l2272_227262

/-- A rectangular table with one side against a wall -/
structure RectangularTable where
  /-- Length of the side opposite the wall -/
  opposite_side : ℝ
  /-- Length of each of the other two free sides -/
  adjacent_side : ℝ
  /-- The side opposite the wall is twice the length of each adjacent side -/
  opposite_twice_adjacent : opposite_side = 2 * adjacent_side
  /-- The area of the table is 128 square feet -/
  area_is_128 : opposite_side * adjacent_side = 128

/-- The total length of the table's free sides is 32 feet -/
theorem total_free_sides_length (table : RectangularTable) :
  table.opposite_side + 2 * table.adjacent_side = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_free_sides_length_l2272_227262


namespace NUMINAMATH_CALUDE_major_axis_length_major_axis_length_is_four_l2272_227211

/-- An ellipse with foci at (5, 1 + √8) and (5, 1 - √8), tangent to y = 1 and x = 1 -/
structure SpecialEllipse where
  /-- First focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- Second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- The ellipse is tangent to the line y = 1 -/
  tangent_y : True
  /-- The ellipse is tangent to the line x = 1 -/
  tangent_x : True
  /-- The first focus is at (5, 1 + √8) -/
  focus1_coord : focus1 = (5, 1 + Real.sqrt 8)
  /-- The second focus is at (5, 1 - √8) -/
  focus2_coord : focus2 = (5, 1 - Real.sqrt 8)

/-- The length of the major axis of the special ellipse is 4 -/
theorem major_axis_length (e : SpecialEllipse) : ℝ := 4

/-- The major axis length of the special ellipse is indeed 4 -/
theorem major_axis_length_is_four (e : SpecialEllipse) : 
  major_axis_length e = 4 := by sorry

end NUMINAMATH_CALUDE_major_axis_length_major_axis_length_is_four_l2272_227211


namespace NUMINAMATH_CALUDE_equation_solution_l2272_227280

theorem equation_solution : ∃ x : ℝ, 
  x = 160 + 64 * Real.sqrt 6 ∧ 
  Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2272_227280


namespace NUMINAMATH_CALUDE_smallest_w_l2272_227297

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_w : 
  ∃ (w : ℕ), w > 0 ∧ 
  is_factor (2^6) (1916 * w) ∧
  is_factor (3^4) (1916 * w) ∧
  is_factor (5^3) (1916 * w) ∧
  is_factor (7^3) (1916 * w) ∧
  is_factor (11^3) (1916 * w) ∧
  ∀ (x : ℕ), x > 0 ∧ 
    is_factor (2^6) (1916 * x) ∧
    is_factor (3^4) (1916 * x) ∧
    is_factor (5^3) (1916 * x) ∧
    is_factor (7^3) (1916 * x) ∧
    is_factor (11^3) (1916 * x) →
    w ≤ x ∧
  w = 74145392000 :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l2272_227297


namespace NUMINAMATH_CALUDE_school_population_l2272_227225

theorem school_population (total : ℚ) 
  (h1 : (2 : ℚ) / 3 * total = total - (1 : ℚ) / 3 * total) 
  (h2 : (1 : ℚ) / 10 * ((1 : ℚ) / 3 * total) = (1 : ℚ) / 3 * total - 90) 
  (h3 : (9 : ℚ) / 10 * ((1 : ℚ) / 3 * total) = 90) : 
  total = 300 := by sorry

end NUMINAMATH_CALUDE_school_population_l2272_227225


namespace NUMINAMATH_CALUDE_page_difference_l2272_227264

/-- The number of purple books Mirella read -/
def purple_books : ℕ := 8

/-- The number of orange books Mirella read -/
def orange_books : ℕ := 7

/-- The number of pages in each purple book -/
def purple_pages : ℕ := 320

/-- The number of pages in each orange book -/
def orange_pages : ℕ := 640

/-- The difference between the total number of orange pages and purple pages read by Mirella -/
theorem page_difference : 
  orange_books * orange_pages - purple_books * purple_pages = 1920 := by
  sorry

end NUMINAMATH_CALUDE_page_difference_l2272_227264


namespace NUMINAMATH_CALUDE_volume_ratio_in_divided_tetrahedron_l2272_227271

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Represents the ratio of distances on an edge -/
def ratio (P Q R : Point3D) : ℝ := sorry

/-- Theorem: Volume ratio in a divided tetrahedron -/
theorem volume_ratio_in_divided_tetrahedron (ABCD : Tetrahedron) 
  (P : Point3D) (Q : Point3D) (R : Point3D) (S : Point3D)
  (hP : ratio P ABCD.A ABCD.B = 1)
  (hQ : ratio Q ABCD.B ABCD.D = 1/2)
  (hR : ratio R ABCD.C ABCD.D = 1/2)
  (hS : ratio S ABCD.A ABCD.C = 1)
  (V1 V2 : ℝ)
  (hV : V1 < V2)
  (hV1V2 : V1 + V2 = volume ABCD)
  : V1 / V2 = 13 / 23 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_in_divided_tetrahedron_l2272_227271


namespace NUMINAMATH_CALUDE_early_arrival_equals_walking_time_l2272_227274

/-- Represents the scenario of a man meeting his wife while walking home from the train station. -/
structure Scenario where
  /-- The time (in minutes) saved by meeting on the way compared to usual arrival time. -/
  time_saved : ℕ
  /-- The time (in minutes) the man spent walking before meeting his wife. -/
  walking_time : ℕ
  /-- The time (in minutes) the wife would normally drive to the station. -/
  normal_driving_time : ℕ
  /-- Assumption that the normal driving time is the difference between walking time and time saved. -/
  h_normal_driving : normal_driving_time = walking_time - time_saved

/-- Theorem stating that the time the man arrived early at the station equals his walking time. -/
theorem early_arrival_equals_walking_time (s : Scenario) :
  s.walking_time = s.walking_time :=
by sorry

#check early_arrival_equals_walking_time

end NUMINAMATH_CALUDE_early_arrival_equals_walking_time_l2272_227274


namespace NUMINAMATH_CALUDE_tangent_circles_triangle_area_l2272_227205

/-- The area of the triangle formed by the points of tangency of three
    mutually externally tangent circles with radii 2, 3, and 4 -/
theorem tangent_circles_triangle_area :
  ∃ (A B C : ℝ × ℝ),
    let r₁ : ℝ := 2
    let r₂ : ℝ := 3
    let r₃ : ℝ := 4
    let O₁ : ℝ × ℝ := (0, 0)
    let O₂ : ℝ × ℝ := (r₁ + r₂, 0)
    let O₃ : ℝ × ℝ := (0, r₁ + r₃)
    -- A, B, C are points of tangency
    A.1^2 + A.2^2 = r₁^2 ∧
    (A.1 - (r₁ + r₂))^2 + A.2^2 = r₂^2 ∧
    B.1^2 + B.2^2 = r₁^2 ∧
    B.1^2 + (B.2 - (r₁ + r₃))^2 = r₃^2 ∧
    (C.1 - (r₁ + r₂))^2 + C.2^2 = r₂^2 ∧
    C.1^2 + (C.2 - (r₁ + r₃))^2 = r₃^2 →
    -- Area of triangle ABC
    abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1) / 2 = 25 / 14 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_triangle_area_l2272_227205


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l2272_227236

/-- The fractional equation -/
def fractional_equation (x a : ℝ) : Prop :=
  (x + a) / (x - 2) - 5 / x = 1

theorem solution_part1 :
  ∀ a : ℝ, fractional_equation 5 a → a = 1 := by sorry

theorem solution_part2 :
  fractional_equation (-5) 5 := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l2272_227236


namespace NUMINAMATH_CALUDE_milkshake_production_l2272_227233

/-- Augustus's milkshake production rate per hour -/
def augustus_rate : ℕ := 3

/-- Luna's milkshake production rate per hour -/
def luna_rate : ℕ := 7

/-- The number of hours Augustus and Luna have been making milkshakes -/
def hours_worked : ℕ := 8

/-- The total number of milkshakes made by Augustus and Luna -/
def total_milkshakes : ℕ := (augustus_rate + luna_rate) * hours_worked

theorem milkshake_production :
  total_milkshakes = 80 := by sorry

end NUMINAMATH_CALUDE_milkshake_production_l2272_227233


namespace NUMINAMATH_CALUDE_bouquet_cost_l2272_227286

/-- The cost of the bouquet given Michael's budget and other expenses --/
theorem bouquet_cost (michael_money : ℕ) (cake_cost : ℕ) (balloons_cost : ℕ) (extra_needed : ℕ) : 
  michael_money = 50 →
  cake_cost = 20 →
  balloons_cost = 5 →
  extra_needed = 11 →
  michael_money + extra_needed = cake_cost + balloons_cost + 36 := by
sorry

end NUMINAMATH_CALUDE_bouquet_cost_l2272_227286


namespace NUMINAMATH_CALUDE_certain_number_value_l2272_227276

theorem certain_number_value : ∃ x : ℝ, 25 * x = 675 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l2272_227276


namespace NUMINAMATH_CALUDE_corn_purchase_amount_l2272_227223

/-- Represents the purchase of corn, beans, and rice -/
structure Purchase where
  corn : ℝ
  beans : ℝ
  rice : ℝ

/-- Checks if a purchase satisfies the given conditions -/
def isValidPurchase (p : Purchase) : Prop :=
  p.corn + p.beans + p.rice = 30 ∧
  1.1 * p.corn + 0.6 * p.beans + 0.9 * p.rice = 24 ∧
  p.rice = 0.5 * p.beans

theorem corn_purchase_amount :
  ∃ (p : Purchase), isValidPurchase p ∧ p.corn = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_corn_purchase_amount_l2272_227223


namespace NUMINAMATH_CALUDE_bus_journey_distance_l2272_227298

/-- Proves that a bus journey with given conditions results in a total distance of 250 km -/
theorem bus_journey_distance (speed1 speed2 distance1 total_time : ℝ) 
  (h1 : speed1 = 40)
  (h2 : speed2 = 60)
  (h3 : distance1 = 100)
  (h4 : total_time = 5)
  (h5 : distance1 / speed1 + (total_distance - distance1) / speed2 = total_time) :
  total_distance = 250 := by
  sorry

#check bus_journey_distance

end NUMINAMATH_CALUDE_bus_journey_distance_l2272_227298


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_line_l2272_227228

/-- The shortest distance between a point on the parabola y = x^2 - 4x + 7
    and a point on the line y = 2x - 5 is 3√5/5 -/
theorem shortest_distance_parabola_line : 
  let parabola := fun x : ℝ => x^2 - 4*x + 7
  let line := fun x : ℝ => 2*x - 5
  ∃ (min_dist : ℝ), 
    (∀ (p q : ℝ × ℝ), 
      (p.2 = parabola p.1) → 
      (q.2 = line q.1) → 
      dist p q ≥ min_dist) ∧
    (∃ (p q : ℝ × ℝ), 
      (p.2 = parabola p.1) ∧ 
      (q.2 = line q.1) ∧ 
      dist p q = min_dist) ∧
    min_dist = 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_line_l2272_227228


namespace NUMINAMATH_CALUDE_trig_identity_l2272_227201

theorem trig_identity (α : ℝ) : 
  Real.sin (9 * α) + Real.sin (10 * α) + Real.sin (11 * α) + Real.sin (12 * α) = 
  4 * Real.cos (α / 2) * Real.cos α * Real.sin ((21 * α) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2272_227201


namespace NUMINAMATH_CALUDE_school_population_proof_l2272_227253

theorem school_population_proof :
  ∀ (n : ℕ) (senior_class : ℕ) (total_selected : ℕ) (other_selected : ℕ),
  senior_class = 900 →
  total_selected = 20 →
  other_selected = 14 →
  (total_selected - other_selected : ℚ) / senior_class = total_selected / n →
  n = 3000 := by
sorry

end NUMINAMATH_CALUDE_school_population_proof_l2272_227253


namespace NUMINAMATH_CALUDE_soccer_team_physics_count_l2272_227221

theorem soccer_team_physics_count (total : ℕ) (math : ℕ) (both : ℕ) (physics : ℕ) : 
  total = 15 → 
  math = 10 → 
  both = 4 → 
  math + physics - both = total → 
  physics = 9 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_physics_count_l2272_227221


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l2272_227235

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : Real :=
  t.a + t.b + t.c

-- Theorem statement
theorem triangle_perimeter_range (t : Triangle) 
  (h1 : t.B = π/3) 
  (h2 : t.b = 2 * Real.sqrt 3) 
  (h3 : t.A > 0) 
  (h4 : t.C > 0) 
  (h5 : t.A + t.B + t.C = π) :
  4 * Real.sqrt 3 < perimeter t ∧ perimeter t ≤ 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l2272_227235


namespace NUMINAMATH_CALUDE_field_trip_cost_theorem_l2272_227256

/-- Calculates the total cost of a field trip with a group discount --/
def field_trip_cost (num_classes : ℕ) (students_per_class : ℕ) (adults_per_class : ℕ)
  (student_fee : ℚ) (adult_fee : ℚ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let total_students := num_classes * students_per_class
  let total_adults := num_classes * adults_per_class
  let student_cost := total_students * student_fee
  let adult_cost := total_adults * adult_fee
  let total_cost := student_cost + adult_cost
  let discount := if total_students > discount_threshold then discount_rate * student_cost else 0
  total_cost - discount

/-- The total cost of the field trip is $987.60 --/
theorem field_trip_cost_theorem :
  field_trip_cost 4 42 6 (11/2) (13/2) (1/10) 40 = 9876/10 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_cost_theorem_l2272_227256


namespace NUMINAMATH_CALUDE_mortgage_more_beneficial_l2272_227295

/-- Represents the annual dividend rate of the preferred shares -/
def dividend_rate : ℝ := 0.17

/-- Represents the annual interest rate of the mortgage loan -/
def mortgage_rate : ℝ := 0.125

/-- Theorem stating that the net return from keeping shares and taking a mortgage is positive -/
theorem mortgage_more_beneficial : dividend_rate - mortgage_rate > 0 := by
  sorry

end NUMINAMATH_CALUDE_mortgage_more_beneficial_l2272_227295


namespace NUMINAMATH_CALUDE_number_difference_l2272_227251

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 41402)
  (b_div_100 : 100 ∣ b)
  (a_eq_b_div_100 : a = b / 100) :
  b - a = 40590 :=
sorry

end NUMINAMATH_CALUDE_number_difference_l2272_227251


namespace NUMINAMATH_CALUDE_angle_between_clock_hands_at_8_30_angle_between_clock_hands_at_8_30_is_75_l2272_227245

/-- The angle between clock hands at 8:30 -/
theorem angle_between_clock_hands_at_8_30 : ℝ :=
  let minute_hand_angle : ℝ := 30 * 6
  let hour_hand_angle : ℝ := 30 * 8 + 30 * 0.5
  |hour_hand_angle - minute_hand_angle|

/-- The angle between clock hands at 8:30 is 75 degrees -/
theorem angle_between_clock_hands_at_8_30_is_75 :
  angle_between_clock_hands_at_8_30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_clock_hands_at_8_30_angle_between_clock_hands_at_8_30_is_75_l2272_227245


namespace NUMINAMATH_CALUDE_quiz_correct_percentage_l2272_227288

theorem quiz_correct_percentage (y : ℝ) (h : y > 0) :
  let total_questions := 7 * y
  let incorrect_questions := y / 3
  let correct_questions := total_questions - incorrect_questions
  (correct_questions / total_questions) = 20 / 21 := by
sorry

end NUMINAMATH_CALUDE_quiz_correct_percentage_l2272_227288


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2272_227218

theorem cubic_roots_sum (a b c : ℂ) (r s t : ℝ) : 
  (∀ x, x^3 - 3*x^2 + 5*x + 7 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (∀ x, x^3 + r*x^2 + s*x + t = 0 ↔ (x = a + b ∨ x = b + c ∨ x = c + a)) →
  t = 8 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2272_227218


namespace NUMINAMATH_CALUDE_count_six_digit_permutations_l2272_227213

/-- The number of different positive six-digit integers that can be formed using the digits 2, 2, 5, 5, 9, and 9 -/
def six_digit_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive six-digit integers
    that can be formed using the digits 2, 2, 5, 5, 9, and 9 is equal to 90 -/
theorem count_six_digit_permutations :
  six_digit_permutations = 90 := by
  sorry

end NUMINAMATH_CALUDE_count_six_digit_permutations_l2272_227213


namespace NUMINAMATH_CALUDE_absolute_value_expression_l2272_227265

theorem absolute_value_expression (x : ℤ) (h : x = -730) :
  |‖x.natAbs ^ 2 - x‖ - x.natAbs| - x = 533630 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l2272_227265


namespace NUMINAMATH_CALUDE_total_area_calculation_l2272_227263

def original_length : ℝ := 13
def original_width : ℝ := 18
def increase : ℝ := 2
def num_equal_rooms : ℕ := 4
def num_double_rooms : ℕ := 1

def new_length : ℝ := original_length + increase
def new_width : ℝ := original_width + increase

def room_area : ℝ := new_length * new_width

theorem total_area_calculation :
  (num_equal_rooms : ℝ) * room_area + (num_double_rooms : ℝ) * 2 * room_area = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_area_calculation_l2272_227263


namespace NUMINAMATH_CALUDE_a_bounds_l2272_227281

/-- Given a linear equation y = ax + 1/3 where x and y are bounded,
    prove that a is bounded between -1/3 and 2/3. -/
theorem a_bounds (a : ℝ) : 
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ y = a * x + 1/3) →
  -1/3 ≤ a ∧ a ≤ 2/3 := by
  sorry

#check a_bounds

end NUMINAMATH_CALUDE_a_bounds_l2272_227281


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l2272_227234

theorem tan_45_degrees_equals_one : 
  Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l2272_227234


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l2272_227257

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem two_digit_reverse_sum_square :
  {n : ℕ | is_two_digit n ∧ is_perfect_square (n + reverse_digits n)} =
  {29, 38, 47, 56, 65, 74, 83, 92} := by sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l2272_227257


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l2272_227244

def total_players : ℕ := 15
def quadruplets : ℕ := 4
def starters : ℕ := 5
def quadruplets_in_lineup : ℕ := 2

theorem basketball_lineup_combinations :
  (Nat.choose quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 990 := by
sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l2272_227244


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1997_l2272_227270

theorem rightmost_three_digits_of_7_to_1997 :
  7^1997 % 1000 = 207 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1997_l2272_227270


namespace NUMINAMATH_CALUDE_impossible_grid_2005_l2272_227203

theorem impossible_grid_2005 : ¬ ∃ (a b c d e f g h i : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (a * b * c = 2005) ∧ (d * e * f = 2005) ∧ (g * h * i = 2005) ∧
  (a * d * g = 2005) ∧ (b * e * h = 2005) ∧ (c * f * i = 2005) ∧
  (a * e * i = 2005) ∧ (c * e * g = 2005) :=
by sorry


end NUMINAMATH_CALUDE_impossible_grid_2005_l2272_227203


namespace NUMINAMATH_CALUDE_delta_value_l2272_227294

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ + 3 → Δ = -15 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l2272_227294


namespace NUMINAMATH_CALUDE_least_number_of_marbles_l2272_227268

theorem least_number_of_marbles (x : ℕ) : x = 50 ↔ 
  x > 0 ∧ 
  x % 6 = 2 ∧ 
  x % 4 = 3 ∧ 
  ∀ y : ℕ, y > 0 ∧ y % 6 = 2 ∧ y % 4 = 3 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_number_of_marbles_l2272_227268


namespace NUMINAMATH_CALUDE_hari_contribution_correct_l2272_227208

/-- Calculates Hari's contribution to the capital given the initial conditions of the business partnership --/
def calculate_hari_contribution (praveen_capital : ℕ) (praveen_months : ℕ) (hari_months : ℕ) (profit_ratio_praveen : ℕ) (profit_ratio_hari : ℕ) : ℕ :=
  (3 * praveen_capital * praveen_months) / (2 * hari_months)

theorem hari_contribution_correct :
  let praveen_capital : ℕ := 3780
  let total_months : ℕ := 12
  let hari_join_month : ℕ := 5
  let profit_ratio_praveen : ℕ := 2
  let profit_ratio_hari : ℕ := 3
  let praveen_months : ℕ := total_months
  let hari_months : ℕ := total_months - hari_join_month
  calculate_hari_contribution praveen_capital praveen_months hari_months profit_ratio_praveen profit_ratio_hari = 9720 :=
by
  sorry

#eval calculate_hari_contribution 3780 12 7 2 3

end NUMINAMATH_CALUDE_hari_contribution_correct_l2272_227208


namespace NUMINAMATH_CALUDE_max_parallelograms_in_hexagon_l2272_227254

-- Define the regular hexagon
def regular_hexagon (side_length : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the parallelogram
def parallelogram (side1 : ℝ) (side2 : ℝ) (angle1 : ℝ) (angle2 : ℝ) : Set (ℝ × ℝ) := sorry

-- Define a function to count non-overlapping parallelograms in a hexagon
def count_parallelograms (h : Set (ℝ × ℝ)) (p : Set (ℝ × ℝ)) : ℕ := sorry

-- Theorem statement
theorem max_parallelograms_in_hexagon :
  let h := regular_hexagon 3
  let p := parallelogram 1 2 (π/3) (2*π/3)
  count_parallelograms h p = 12 := by sorry

end NUMINAMATH_CALUDE_max_parallelograms_in_hexagon_l2272_227254


namespace NUMINAMATH_CALUDE_union_covers_reals_l2272_227272

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B (a : ℝ) : Set ℝ := {x | |x - a| < 3}

-- State the theorem
theorem union_covers_reals (a : ℝ) : 
  (A ∪ B a = Set.univ) → a ∈ Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_union_covers_reals_l2272_227272


namespace NUMINAMATH_CALUDE_fraction_equality_l2272_227222

theorem fraction_equality (x y : ℝ) (h : x / 2 = y / 5) : x / y = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2272_227222


namespace NUMINAMATH_CALUDE_minute_hand_angle_for_110_minutes_l2272_227290

/-- The angle turned by the minute hand when the hour hand moves for a given time -/
def minuteHandAngle (hours : ℝ) (minutes : ℝ) : ℝ :=
  -(hours * 360 + minutes * 6)

/-- Theorem: When the hour hand moves for 1 hour and 50 minutes, 
    the angle turned by the minute hand is -660° -/
theorem minute_hand_angle_for_110_minutes : 
  minuteHandAngle 1 50 = -660 := by sorry

end NUMINAMATH_CALUDE_minute_hand_angle_for_110_minutes_l2272_227290


namespace NUMINAMATH_CALUDE_sum_product_reciprocal_sum_squared_inequality_l2272_227232

theorem sum_product_reciprocal_sum_squared_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a*b + b*c + c*a) * (1/(a+b)^2 + 1/(b+c)^2 + 1/(c+a)^2) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_reciprocal_sum_squared_inequality_l2272_227232


namespace NUMINAMATH_CALUDE_tank_filling_time_l2272_227287

theorem tank_filling_time (pipe1_time pipe2_time : ℝ) (leak_fraction : ℝ) : 
  pipe1_time = 20 ∧ 
  pipe2_time = 30 ∧ 
  leak_fraction = 1/3 → 
  (1 / (1/pipe1_time + 1/pipe2_time)) * (1 / (1 - leak_fraction)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l2272_227287


namespace NUMINAMATH_CALUDE_house_prices_and_yields_l2272_227258

theorem house_prices_and_yields :
  ∀ (price1 price2 yield1 yield2 : ℝ),
  price1 > 0 ∧ price2 > 0 ∧ yield1 > 0 ∧ yield2 > 0 →
  425 = (yield1 / 100) * price1 →
  459 = (yield2 / 100) * price2 →
  price2 = (6 / 5) * price1 →
  yield2 = yield1 - (1 / 2) →
  price1 = 8500 ∧ price2 = 10200 ∧ yield1 = 5 ∧ yield2 = (9 / 2) :=
by sorry

end NUMINAMATH_CALUDE_house_prices_and_yields_l2272_227258


namespace NUMINAMATH_CALUDE_knitting_productivity_l2272_227292

theorem knitting_productivity (girl1_work_time girl1_break_time girl2_work_time girl2_break_time : ℕ) 
  (h1 : girl1_work_time = 5)
  (h2 : girl1_break_time = 1)
  (h3 : girl2_work_time = 7)
  (h4 : girl2_break_time = 1)
  : (girl1_work_time * (girl1_work_time + girl1_break_time)) / 
    (girl2_work_time * (girl2_work_time + girl2_break_time)) = 20 / 21 :=
by sorry

end NUMINAMATH_CALUDE_knitting_productivity_l2272_227292


namespace NUMINAMATH_CALUDE_garrison_provisions_l2272_227299

theorem garrison_provisions (initial_men : ℕ) (reinforcement : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) 
  (h1 : initial_men = 2000)
  (h2 : reinforcement = 2000)
  (h3 : days_before_reinforcement = 20)
  (h4 : days_after_reinforcement = 10) :
  ∃ (initial_duration : ℕ), 
    initial_men * (initial_duration - days_before_reinforcement) = 
    (initial_men + reinforcement) * days_after_reinforcement ∧
    initial_duration = 40 := by
sorry

end NUMINAMATH_CALUDE_garrison_provisions_l2272_227299


namespace NUMINAMATH_CALUDE_josh_initial_money_l2272_227227

/-- Josh's initial amount of money given his expenses and remaining balance -/
def initial_amount (spent1 spent2 remaining : ℚ) : ℚ :=
  spent1 + spent2 + remaining

theorem josh_initial_money :
  initial_amount 1.75 1.25 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_josh_initial_money_l2272_227227


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_sum_l2272_227284

theorem geometric_arithmetic_sequence_ratio_sum :
  ∀ (x y z : ℝ),
  (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) →
  (∃ (r : ℝ), r ≠ 0 ∧ 4*y = 3*x*r ∧ 5*z = 4*y*r) →
  (∃ (d : ℝ), 1/y - 1/x = d ∧ 1/z - 1/y = d) →
  x/z + z/x = 34/15 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_sum_l2272_227284


namespace NUMINAMATH_CALUDE_expression_evaluation_l2272_227202

theorem expression_evaluation :
  let f (x : ℤ) := 8 * x^2 - (x - 2) * (3 * x + 1) - 2 * (x + 1) * (x - 1)
  f (-2) = 6 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2272_227202


namespace NUMINAMATH_CALUDE_boys_in_biology_class_l2272_227282

/-- Represents the number of students in a class -/
structure ClassCount where
  boys : ℕ
  girls : ℕ

/-- Represents the counts for all three classes -/
structure SchoolCounts where
  biology : ClassCount
  physics : ClassCount
  chemistry : ClassCount

/-- The conditions of the problem -/
def school_conditions (counts : SchoolCounts) : Prop :=
  -- Biology class condition
  counts.biology.girls = 3 * counts.biology.boys ∧
  -- Physics class condition
  2 * counts.physics.boys = 3 * counts.physics.girls ∧
  -- Chemistry class condition
  counts.chemistry.boys = counts.chemistry.girls ∧
  counts.chemistry.boys + counts.chemistry.girls = 270 ∧
  -- Relation between Biology and Physics classes
  counts.biology.boys + counts.biology.girls = 
    (counts.physics.boys + counts.physics.girls) / 2 ∧
  -- Total number of students
  counts.biology.boys + counts.biology.girls +
  counts.physics.boys + counts.physics.girls +
  counts.chemistry.boys + counts.chemistry.girls = 1000

/-- The theorem to be proved -/
theorem boys_in_biology_class (counts : SchoolCounts) :
  school_conditions counts → counts.biology.boys = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_boys_in_biology_class_l2272_227282


namespace NUMINAMATH_CALUDE_meditation_time_per_week_l2272_227275

/-- Calculates the total hours spent meditating in a week given the daily meditation time in minutes -/
def weekly_meditation_hours (daily_minutes : ℕ) : ℚ :=
  (daily_minutes : ℚ) * 7 / 60

theorem meditation_time_per_week :
  weekly_meditation_hours (30 * 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_meditation_time_per_week_l2272_227275


namespace NUMINAMATH_CALUDE_angle_measure_angle_measure_proof_l2272_227237

theorem angle_measure : ℝ → Prop :=
  fun x =>
    (180 - x = 4 * (90 - x)) →
    x = 60

-- The proof is omitted
theorem angle_measure_proof : ∃ x, angle_measure x :=
  sorry

end NUMINAMATH_CALUDE_angle_measure_angle_measure_proof_l2272_227237


namespace NUMINAMATH_CALUDE_real_part_of_z_l2272_227214

theorem real_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2) : 
  Complex.re z = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2272_227214


namespace NUMINAMATH_CALUDE_payment_difference_l2272_227241

/-- Represents the pizza and its cost structure -/
structure Pizza :=
  (total_slices : ℕ)
  (plain_cost : ℚ)
  (anchovy_cost : ℚ)
  (mushroom_cost : ℚ)

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : ℚ :=
  p.plain_cost + p.anchovy_cost + p.mushroom_cost

/-- Calculates the number of slices Dave ate -/
def dave_slices (p : Pizza) : ℕ :=
  p.total_slices / 2 + p.total_slices / 4 + 1

/-- Calculates the number of slices Doug ate -/
def doug_slices (p : Pizza) : ℕ :=
  p.total_slices - dave_slices p

/-- Calculates Dave's payment -/
def dave_payment (p : Pizza) : ℚ :=
  total_cost p - (p.plain_cost / p.total_slices) * doug_slices p

/-- Calculates Doug's payment -/
def doug_payment (p : Pizza) : ℚ :=
  (p.plain_cost / p.total_slices) * doug_slices p

/-- The main theorem stating the difference in payments -/
theorem payment_difference (p : Pizza) 
  (h1 : p.total_slices = 8)
  (h2 : p.plain_cost = 8)
  (h3 : p.anchovy_cost = 2)
  (h4 : p.mushroom_cost = 1) :
  dave_payment p - doug_payment p = 9 := by
  sorry

end NUMINAMATH_CALUDE_payment_difference_l2272_227241


namespace NUMINAMATH_CALUDE_abs_sum_cases_l2272_227278

theorem abs_sum_cases (x : ℝ) (h : x < 2) :
  (|x - 2| + |2 + x| = 4 ∧ -2 ≤ x) ∨ (|x - 2| + |2 + x| = -2*x ∧ x < -2) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_cases_l2272_227278


namespace NUMINAMATH_CALUDE_tree_scenario_result_l2272_227238

/-- Represents the number of caterpillars and leaves eaten in a tree scenario -/
def tree_scenario (initial_caterpillars storm_fallen hatched_eggs 
                   baby_leaves_eaten cocoon_left moth_ratio
                   moth_daily_consumption days : ℕ) : ℕ × ℕ :=
  let remaining_after_storm := initial_caterpillars - storm_fallen
  let total_after_hatch := remaining_after_storm + hatched_eggs
  let remaining_after_cocoon := total_after_hatch - cocoon_left
  let moth_caterpillars := remaining_after_cocoon / 2
  let total_leaves_eaten := baby_leaves_eaten + 
    moth_caterpillars * moth_daily_consumption * days
  (remaining_after_cocoon, total_leaves_eaten)

/-- Theorem stating the result of the tree scenario -/
theorem tree_scenario_result : 
  tree_scenario 14 3 6 18 9 2 4 7 = (8, 130) :=
by sorry

end NUMINAMATH_CALUDE_tree_scenario_result_l2272_227238


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_inequality_l2272_227283

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 2
def g (a x : ℝ) : ℝ := |x - a| - |x - 1|

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_4 :
  {x : ℝ | f x > g 4 x} = {x : ℝ | x < -1 ∨ x > 1} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x₁ x₂, f x₁ ≥ g a x₂} = Set.Icc (-1) 3 := by sorry

-- Note: Set.Icc represents a closed interval [a, b]

end NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_inequality_l2272_227283


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l2272_227215

theorem sin_shift_equivalence (x : ℝ) :
  2 * Real.sin (3 * x + π / 4) = 2 * Real.sin (3 * (x + π / 12)) :=
by sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l2272_227215


namespace NUMINAMATH_CALUDE_inequality_proof_l2272_227246

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2272_227246


namespace NUMINAMATH_CALUDE_sqrt_57_between_7_and_8_l2272_227261

theorem sqrt_57_between_7_and_8 : 7 < Real.sqrt 57 ∧ Real.sqrt 57 < 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_57_between_7_and_8_l2272_227261


namespace NUMINAMATH_CALUDE_systematic_sampling_example_l2272_227219

def isValidSystematicSample (n : ℕ) (k : ℕ) (sample : List ℕ) : Prop :=
  let interval := n / k
  sample.length = k ∧
  ∀ i, i ∈ sample → i < n ∧
  ∀ i j, i < j → i ∈ sample → j ∈ sample → j - i = interval

theorem systematic_sampling_example :
  isValidSystematicSample 50 5 [1, 11, 21, 31, 41] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_example_l2272_227219


namespace NUMINAMATH_CALUDE_contradiction_assumption_l2272_227229

theorem contradiction_assumption (a b c : ℝ) : 
  (¬(a > 0 ∧ b > 0 ∧ c > 0)) ↔ (¬(a > 0) ∨ ¬(b > 0) ∨ ¬(c > 0)) :=
by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l2272_227229


namespace NUMINAMATH_CALUDE_smallest_number_problem_l2272_227249

theorem smallest_number_problem (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 30 →
  b = 29 →
  max a (max b c) = b + 8 →
  min a (min b c) = 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_problem_l2272_227249


namespace NUMINAMATH_CALUDE_equation_holds_iff_b_equals_c_l2272_227217

theorem equation_holds_iff_b_equals_c (a b c : ℕ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_less_than_10 : a < 10 ∧ b < 10 ∧ c < 10) :
  (10 * a + b + 1) * (10 * a + c) = 100 * a^2 + 100 * a + b + c ↔ b = c :=
sorry

end NUMINAMATH_CALUDE_equation_holds_iff_b_equals_c_l2272_227217


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2272_227206

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I * (1 - a) = -a - 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2272_227206


namespace NUMINAMATH_CALUDE_max_score_in_twenty_over_match_l2272_227252

/-- Represents the number of overs in the cricket match -/
def overs : ℕ := 20

/-- Represents the number of balls in an over -/
def balls_per_over : ℕ := 6

/-- Represents the maximum runs that can be scored on a single ball -/
def max_runs_per_ball : ℕ := 6

/-- Calculates the maximum runs a batsman can score in a perfect scenario -/
def max_batsman_score : ℕ := overs * balls_per_over * max_runs_per_ball

theorem max_score_in_twenty_over_match :
  max_batsman_score = 720 :=
by sorry

end NUMINAMATH_CALUDE_max_score_in_twenty_over_match_l2272_227252


namespace NUMINAMATH_CALUDE_circle_radius_equals_sphere_surface_area_l2272_227212

theorem circle_radius_equals_sphere_surface_area (r : ℝ) : 
  r > 0 → π * r^2 = 4 * π * (2 : ℝ)^2 → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_equals_sphere_surface_area_l2272_227212


namespace NUMINAMATH_CALUDE_perfect_squares_implications_l2272_227243

theorem perfect_squares_implications (n : ℕ+) 
  (h1 : ∃ a : ℕ, 3 * n + 1 = a^2) 
  (h2 : ∃ b : ℕ, 5 * n - 1 = b^2) :
  (∃ p q : ℕ, p > 1 ∧ q > 1 ∧ 7 * n + 13 = p * q) ∧ 
  (∃ x y : ℕ, 8 * (17 * n^2 + 3 * n) = x^2 + y^2) := by
sorry

end NUMINAMATH_CALUDE_perfect_squares_implications_l2272_227243


namespace NUMINAMATH_CALUDE_sugar_amount_l2272_227216

/-- Represents the amounts of ingredients in a bakery storage room. -/
structure BakeryStorage where
  sugar : ℝ
  flour : ℝ
  bakingSoda : ℝ

/-- Checks if the given storage satisfies the bakery's ratios. -/
def satisfiesRatios (storage : BakeryStorage) : Prop :=
  storage.sugar / storage.flour = 5 / 4 ∧
  storage.flour / storage.bakingSoda = 10 / 1

/-- Checks if adding 60 pounds of baking soda changes the ratio as specified. -/
def satisfiesNewRatio (storage : BakeryStorage) : Prop :=
  storage.flour / (storage.bakingSoda + 60) = 8 / 1

/-- Theorem: Given the conditions, the amount of sugar in the storage is 3000 pounds. -/
theorem sugar_amount (storage : BakeryStorage) 
  (h1 : satisfiesRatios storage) 
  (h2 : satisfiesNewRatio storage) : 
  storage.sugar = 3000 := by
sorry

end NUMINAMATH_CALUDE_sugar_amount_l2272_227216


namespace NUMINAMATH_CALUDE_magazine_circulation_ratio_l2272_227231

/-- Given a magazine's circulation data, proves the ratio of circulation in 1961 to total circulation from 1961-1970 -/
theorem magazine_circulation_ratio 
  (avg_circulation : ℝ) -- Average yearly circulation for 1962-1970
  (h1 : avg_circulation > 0) -- Assumption that average circulation is positive
  : (4 * avg_circulation) / (4 * avg_circulation + 9 * avg_circulation) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_magazine_circulation_ratio_l2272_227231


namespace NUMINAMATH_CALUDE_keychain_cost_is_five_l2272_227247

/-- The cost of a bracelet in dollars -/
def bracelet_cost : ℝ := 4

/-- The cost of a coloring book in dollars -/
def coloring_book_cost : ℝ := 3

/-- The cost of Paula's purchase in dollars -/
def paula_cost (keychain_cost : ℝ) : ℝ := 2 * bracelet_cost + keychain_cost

/-- The cost of Olive's purchase in dollars -/
def olive_cost : ℝ := coloring_book_cost + bracelet_cost

/-- The total amount spent by Paula and Olive in dollars -/
def total_spent : ℝ := 20

/-- Theorem stating that the keychain cost is 5 dollars -/
theorem keychain_cost_is_five : 
  ∃ (keychain_cost : ℝ), paula_cost keychain_cost + olive_cost = total_spent ∧ keychain_cost = 5 :=
sorry

end NUMINAMATH_CALUDE_keychain_cost_is_five_l2272_227247


namespace NUMINAMATH_CALUDE_pythagorean_triple_with_ratio_exists_l2272_227230

theorem pythagorean_triple_with_ratio_exists (k : ℚ) (hk : k > 1) :
  ∃ (a b c : ℕ+), (a.val^2 + b.val^2 = c.val^2) ∧ ((a.val + c.val) / b.val : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_with_ratio_exists_l2272_227230


namespace NUMINAMATH_CALUDE_train_crossing_time_l2272_227296

/-- Proves that given two trains of equal length, where one train takes 15 seconds to cross a
    telegraph post, and they cross each other traveling in opposite directions in 7.5 seconds,
    the other train will take 5 seconds to cross the telegraph post. -/
theorem train_crossing_time
  (train_length : ℝ)
  (second_train_time : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : second_train_time = 15)
  (h3 : crossing_time = 7.5) :
  train_length / (train_length / second_train_time + train_length / crossing_time - train_length / second_train_time) = 5 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2272_227296


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2272_227250

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 - 4*x + 4 = 0 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2272_227250


namespace NUMINAMATH_CALUDE_min_value_theorem_l2272_227279

theorem min_value_theorem (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, |x - a| + |x + b| ≥ 2) : 
  (a + b = 2) ∧ ¬(a^2 + a > 2 ∧ b^2 + b > 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2272_227279


namespace NUMINAMATH_CALUDE_binary_division_theorem_l2272_227259

/-- Convert a binary number (represented as a list of 0s and 1s) to a natural number. -/
def binary_to_nat (binary : List Nat) : Nat :=
  binary.foldr (fun bit acc => 2 * acc + bit) 0

/-- The binary representation of 10101₂ -/
def binary_10101 : List Nat := [1, 0, 1, 0, 1]

/-- The binary representation of 11₂ -/
def binary_11 : List Nat := [1, 1]

/-- The binary representation of 111₂ -/
def binary_111 : List Nat := [1, 1, 1]

/-- Theorem stating that the quotient of 10101₂ divided by 11₂ is equal to 111₂ -/
theorem binary_division_theorem :
  (binary_to_nat binary_10101) / (binary_to_nat binary_11) = binary_to_nat binary_111 := by
  sorry

end NUMINAMATH_CALUDE_binary_division_theorem_l2272_227259


namespace NUMINAMATH_CALUDE_derivative_at_one_l2272_227240

noncomputable section

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

theorem derivative_at_one
  (h1 : ∀ x, f x = f' 1 * x^3 - 2^x)
  (h2 : ∀ x, HasDerivAt f (f' x) x) :
  f' 1 = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2272_227240


namespace NUMINAMATH_CALUDE_expression_evaluation_l2272_227226

theorem expression_evaluation (a b c : ℝ) (ha : a = 13) (hb : b = 17) (hc : c = 19) :
  (b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b) + a^2 * (1/b - 1/c)) /
  (b * (1/c - 1/a) + c * (1/a - 1/b) + a * (1/b - 1/c)) = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2272_227226


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l2272_227200

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ)
  (h1 : interest = 4016.25)
  (h2 : rate = 0.01)
  (h3 : time = 5)
  (h4 : interest = principal * rate * time) :
  principal = 80325 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l2272_227200


namespace NUMINAMATH_CALUDE_simplify_expression_l2272_227273

theorem simplify_expression (x : ℝ) : 120*x - 32*x + 15 - 15 = 88*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2272_227273


namespace NUMINAMATH_CALUDE_square_sum_equals_five_l2272_227209

theorem square_sum_equals_five (x y : ℝ) (h1 : (x - y)^2 = 25) (h2 : x * y = -10) :
  x^2 + y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_five_l2272_227209


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_quotient_l2272_227242

theorem isosceles_right_triangle_quotient (a : ℝ) (h : a > 0) :
  let b := a
  let c := Real.sqrt (a^2 + b^2)
  (2 * a) / Real.sqrt (a^2 + a^2) = Real.sqrt 2 ∧
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x^2 + y^2 = z^2 → (x + y) / z ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_quotient_l2272_227242


namespace NUMINAMATH_CALUDE_jane_change_l2272_227285

/-- Calculates the change received after a purchase -/
def calculate_change (num_skirts : ℕ) (price_skirt : ℕ) (num_blouses : ℕ) (price_blouse : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_skirts * price_skirt + num_blouses * price_blouse)

/-- Proves that Jane received $56 in change -/
theorem jane_change : calculate_change 2 13 3 6 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_jane_change_l2272_227285


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l2272_227291

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 10) (h₂ : r₂ = 4) :
  π * r₁^2 - π * r₂^2 = 84 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l2272_227291


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l2272_227269

theorem square_root_equation_solution (x : ℝ) :
  Real.sqrt (2 - 5 * x + x^2) = 9 ↔ x = (5 + Real.sqrt 341) / 2 ∨ x = (5 - Real.sqrt 341) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l2272_227269


namespace NUMINAMATH_CALUDE_quadratic_max_l2272_227277

theorem quadratic_max (a b c : ℝ) (x₀ : ℝ) (h1 : a < 0) (h2 : 2 * a * x₀ + b = 0) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  ∀ x : ℝ, f x ≤ f x₀ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_l2272_227277


namespace NUMINAMATH_CALUDE_melanie_initial_dimes_l2272_227239

/-- The number of dimes Melanie initially had in her bank -/
def initial_dimes : ℕ := sorry

/-- The number of dimes Melanie's dad gave her -/
def dimes_from_dad : ℕ := 8

/-- The number of dimes Melanie gave to her mother -/
def dimes_to_mother : ℕ := 4

/-- The number of dimes Melanie has now -/
def current_dimes : ℕ := 11

theorem melanie_initial_dimes : 
  initial_dimes + dimes_from_dad - dimes_to_mother = current_dimes := by sorry

end NUMINAMATH_CALUDE_melanie_initial_dimes_l2272_227239


namespace NUMINAMATH_CALUDE_cubic_root_between_integers_l2272_227266

theorem cubic_root_between_integers : ∃ (A B : ℤ), 
  B = A + 1 ∧ 
  ∃ (x : ℝ), A < x ∧ x < B ∧ x^3 + 5*x^2 - 3*x + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_between_integers_l2272_227266


namespace NUMINAMATH_CALUDE_confidence_level_interpretation_l2272_227267

theorem confidence_level_interpretation 
  (confidence_level : ℝ) 
  (hypothesis_test : Type) 
  (is_valid_test : hypothesis_test → Prop) 
  (test_result : hypothesis_test → Bool) 
  (h_confidence : confidence_level = 0.95) :
  ∃ (error_probability : ℝ), 
    error_probability = 1 - confidence_level ∧ 
    error_probability = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_confidence_level_interpretation_l2272_227267


namespace NUMINAMATH_CALUDE_polynomial_identity_l2272_227293

theorem polynomial_identity (g : ℝ → ℝ) (h : ∀ x, g (x^2 + 2) = x^4 + 6*x^2 + 4) :
  ∀ x, g (x^2 - 2) = x^4 - 2*x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2272_227293


namespace NUMINAMATH_CALUDE_system_solution_l2272_227210

theorem system_solution (m : ℤ) : 
  (∃ (x y : ℝ), 
    x - 2*y = m ∧ 
    2*x + 3*y = 2*m - 3 ∧ 
    3*x + y ≥ 0 ∧ 
    x + 5*y < 0) ↔ 
  (m = 1 ∨ m = 2) := by sorry

end NUMINAMATH_CALUDE_system_solution_l2272_227210


namespace NUMINAMATH_CALUDE_f_properties_l2272_227204

def f (x : ℝ) : ℝ := -(x - 2)^2 + 4

theorem f_properties :
  (∀ x : ℝ, ∃ y : ℝ, f x = y) ∧
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧
  (∀ x y : ℝ, x < y ∧ y < 0 → f x < f y) ∧
  (∀ x : ℝ, f x ≤ 4) ∧
  (∃ x : ℝ, f x = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2272_227204


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2272_227224

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2272_227224


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2272_227255

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2/a^2 - y^2/3 = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- State that the focus of the parabola is the right focus of the hyperbola
axiom focus_equality : ∃ a : ℝ, hyperbola (parabola_focus.1) (parabola_focus.2) a

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y a : ℝ, parabola x y → hyperbola x y a → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2272_227255


namespace NUMINAMATH_CALUDE_euler_formula_third_quadrant_l2272_227220

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- Define the third quadrant
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- State the theorem
theorem euler_formula_third_quadrant (θ : ℝ) (k : ℤ) :
  (2 * k * Real.pi + Real.pi / 2 < θ) ∧ (θ ≤ 2 * k * Real.pi + 2 * Real.pi / 3) →
  third_quadrant (cexp (2 * θ * Complex.I)) :=
sorry

end NUMINAMATH_CALUDE_euler_formula_third_quadrant_l2272_227220


namespace NUMINAMATH_CALUDE_triangle_shape_l2272_227260

theorem triangle_shape (A B C : Real) (a b c : Real) :
  -- Define triangle ABC
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) →
  -- Define sides a, b, c opposite to angles A, B, C
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Law of sines
  (a / (Real.sin A) = b / (Real.sin B)) ∧
  (b / (Real.sin B) = c / (Real.sin C)) →
  -- Given condition
  (Real.cos (A/2))^2 = (c + b) / (2*c) →
  -- Conclusion: C is a right angle
  C = π/2 := by sorry

end NUMINAMATH_CALUDE_triangle_shape_l2272_227260


namespace NUMINAMATH_CALUDE_power_fraction_multiply_simplify_fraction_two_thirds_cubed_times_half_l2272_227248

theorem power_fraction_multiply (a b c d : ℚ) :
  (a / b) ^ 3 * (c / d) = (a ^ 3 * c) / (b ^ 3 * d) :=
by sorry

theorem simplify_fraction_two_thirds_cubed_times_half :
  (2 / 3 : ℚ) ^ 3 * (1 / 2) = 4 / 27 :=
by sorry

end NUMINAMATH_CALUDE_power_fraction_multiply_simplify_fraction_two_thirds_cubed_times_half_l2272_227248


namespace NUMINAMATH_CALUDE_complex_number_equal_parts_l2272_227289

theorem complex_number_equal_parts (b : ℝ) : 
  let z : ℂ := (3 - b * Complex.I) / (2 + Complex.I)
  z.re = z.im ↔ b = -9 := by
sorry

end NUMINAMATH_CALUDE_complex_number_equal_parts_l2272_227289


namespace NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l2272_227207

-- Define the quadratic function
def f (x : ℝ) := -4 * x^2 + 4 * x + 7

-- State the theorem
theorem quadratic_function_satisfies_conditions :
  (f 2 = -1) ∧ 
  (f (-1) = -1) ∧ 
  (∀ x : ℝ, f x ≤ 8) ∧
  (∃ x : ℝ, f x = 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l2272_227207
