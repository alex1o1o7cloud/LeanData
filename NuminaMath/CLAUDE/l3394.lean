import Mathlib

namespace NUMINAMATH_CALUDE_circles_tangent_internally_l3394_339486

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 16 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 64

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (4, -3)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (0, 0)
def radius2 : ℝ := 8

-- Theorem stating that the circles are tangent internally
theorem circles_tangent_internally :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = radius2 - radius1 := by sorry

end NUMINAMATH_CALUDE_circles_tangent_internally_l3394_339486


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l3394_339459

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse defined by its center and semi-axes lengths -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis length
  b : ℝ  -- semi-minor axis length

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- The given points -/
def points : List Point := [
  { x := 1, y := 1 },
  { x := 0, y := 0 },
  { x := 0, y := 3 },
  { x := 4, y := 0 },
  { x := 4, y := 3 }
]

theorem ellipse_minor_axis_length :
  ∃ (e : Ellipse),
    (∀ p ∈ points, pointOnEllipse p e) ∧
    (e.center.x = 2 ∧ e.center.y = 1.5) ∧
    (e.a = 2) ∧
    (e.b * 2 = 2 * Real.sqrt 3) :=
by sorry

#check ellipse_minor_axis_length

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l3394_339459


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l3394_339421

theorem trigonometric_expression_evaluation (α : Real) (h : Real.tan α = 2) :
  (Real.cos (-π/2 - α) * Real.tan (π + α) - Real.sin (π/2 - α)) /
  (Real.cos (3*π/2 + α) + Real.cos (π - α)) = -5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l3394_339421


namespace NUMINAMATH_CALUDE_amp_example_l3394_339470

-- Define the & operation
def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

-- State the theorem
theorem amp_example : 50 - amp 8 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_amp_example_l3394_339470


namespace NUMINAMATH_CALUDE_x_range_proof_l3394_339403

def S (n : ℕ) : ℝ := sorry

def a : ℕ → ℝ := sorry

theorem x_range_proof :
  (∀ n : ℕ, n ≥ 2 → S (n - 1) + S n = 2 * n^2 + 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) > a n) →
  a 1 = x →
  2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_x_range_proof_l3394_339403


namespace NUMINAMATH_CALUDE_age_problem_l3394_339405

theorem age_problem (p q : ℕ) : 
  (p - 8 = (q - 8) / 2) →  -- 8 years ago, p was half of q's age
  (p * 4 = q * 3) →        -- The ratio of their present ages is 3:4
  (p + q = 28) :=          -- The total of their present ages is 28
by sorry

end NUMINAMATH_CALUDE_age_problem_l3394_339405


namespace NUMINAMATH_CALUDE_one_sofa_in_room_l3394_339445

/-- Represents the number of sofas in the room -/
def num_sofas : ℕ := 1

/-- Represents the total number of legs in the room -/
def total_legs : ℕ := 40

/-- Represents the number of legs on a sofa -/
def legs_per_sofa : ℕ := 4

/-- Represents the number of legs from furniture other than sofas -/
def other_furniture_legs : ℕ := 
  4 * 4 +  -- 4 tables with 4 legs each
  2 * 4 +  -- 2 chairs with 4 legs each
  3 * 3 +  -- 3 tables with 3 legs each
  1 * 1 +  -- 1 table with 1 leg
  1 * 2    -- 1 rocking chair with 2 legs

/-- Theorem stating that there is exactly one sofa in the room -/
theorem one_sofa_in_room : 
  num_sofas * legs_per_sofa + other_furniture_legs = total_legs :=
by sorry

end NUMINAMATH_CALUDE_one_sofa_in_room_l3394_339445


namespace NUMINAMATH_CALUDE_scout_troop_girls_l3394_339475

theorem scout_troop_girls (initial_total : ℕ) : 
  let initial_girls : ℕ := (6 * initial_total) / 10
  let final_total : ℕ := initial_total
  let final_girls : ℕ := initial_girls - 4
  (initial_girls : ℚ) / initial_total = 6 / 10 →
  (final_girls : ℚ) / final_total = 1 / 2 →
  initial_girls = 24 := by
sorry

end NUMINAMATH_CALUDE_scout_troop_girls_l3394_339475


namespace NUMINAMATH_CALUDE_number_of_mappings_l3394_339435

def M : Finset ℕ := {0, 1, 2}
def N : Finset ℕ := {1, 2, 3, 4}

theorem number_of_mappings : Fintype.card (M → N) = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_of_mappings_l3394_339435


namespace NUMINAMATH_CALUDE_distance_after_translation_l3394_339416

/-- The distance between two points after translation --/
theorem distance_after_translation (x1 y1 x2 y2 tx ty : ℝ) :
  let p1 := (x1 + tx, y1 + ty)
  let p2 := (x2 + tx, y2 + ty)
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 73 :=
by
  sorry

#check distance_after_translation 5 3 (-3) 0 3 (-3)

end NUMINAMATH_CALUDE_distance_after_translation_l3394_339416


namespace NUMINAMATH_CALUDE_savings_difference_l3394_339413

def regular_price : ℕ := 120
def free_window_threshold : ℕ := 5
def bulk_discount_threshold : ℕ := 10
def bulk_discount_amount : ℕ := 10
def alice_windows : ℕ := 9
def bob_windows : ℕ := 12

def calculate_cost (windows : ℕ) : ℕ :=
  let free_windows := windows / free_window_threshold
  let paid_windows := windows - free_windows
  let price := if windows > bulk_discount_threshold
               then regular_price - bulk_discount_amount
               else regular_price
  paid_windows * price

def individual_savings : ℕ :=
  (alice_windows * regular_price - calculate_cost alice_windows) +
  (bob_windows * regular_price - calculate_cost bob_windows)

def combined_savings : ℕ :=
  (alice_windows + bob_windows) * regular_price -
  calculate_cost (alice_windows + bob_windows)

theorem savings_difference :
  combined_savings - individual_savings = 300 := by sorry

end NUMINAMATH_CALUDE_savings_difference_l3394_339413


namespace NUMINAMATH_CALUDE_fraction_division_equality_l3394_339494

theorem fraction_division_equality : (2 / 5) / (3 / 7) = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l3394_339494


namespace NUMINAMATH_CALUDE_queenie_work_days_l3394_339426

/-- Calculates the number of days worked given the daily rate, overtime rate, overtime hours, and total payment -/
def days_worked (daily_rate : ℕ) (overtime_rate : ℕ) (overtime_hours : ℕ) (total_payment : ℕ) : ℕ :=
  (total_payment / (daily_rate + overtime_rate * overtime_hours))

/-- Proves that given the specified conditions, the number of days worked is 4 -/
theorem queenie_work_days : 
  let daily_rate : ℕ := 150
  let overtime_rate : ℕ := 5
  let overtime_hours : ℕ := 4
  let total_payment : ℕ := 770
  days_worked daily_rate overtime_rate overtime_hours total_payment = 4 := by
sorry

#eval days_worked 150 5 4 770

end NUMINAMATH_CALUDE_queenie_work_days_l3394_339426


namespace NUMINAMATH_CALUDE_largest_divisible_by_9_l3394_339434

def original_number : ℕ := 547654765476

def remove_digits (n : ℕ) (positions : List ℕ) : ℕ :=
  sorry

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem largest_divisible_by_9 :
  ∀ (positions : List ℕ),
    let result := remove_digits original_number positions
    is_divisible_by_9 result →
    result ≤ 5476547646 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_by_9_l3394_339434


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l3394_339429

/-- The probability of no rain on a given day -/
def prob_no_rain : ℝ := 0.3

/-- The probability of 5 inches of rain on a given day -/
def prob_5_inches : ℝ := 0.4

/-- The probability of 12 inches of rain on a given day -/
def prob_12_inches : ℝ := 0.3

/-- The amount of rainfall in inches when it rains 5 inches -/
def rain_5_inches : ℝ := 5

/-- The amount of rainfall in inches when it rains 12 inches -/
def rain_12_inches : ℝ := 12

/-- The number of days in the week -/
def days_in_week : ℕ := 7

/-- The expected rainfall for one day -/
def expected_daily_rainfall : ℝ :=
  prob_no_rain * 0 + prob_5_inches * rain_5_inches + prob_12_inches * rain_12_inches

/-- Theorem: The expected total rainfall for the week is 39.2 inches -/
theorem expected_weekly_rainfall :
  (days_in_week : ℝ) * expected_daily_rainfall = 39.2 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l3394_339429


namespace NUMINAMATH_CALUDE_min_distance_line_curve_l3394_339419

/-- The line represented by the parametric equations x = t, y = 6 - 2t -/
def line (t : ℝ) : ℝ × ℝ := (t, 6 - 2*t)

/-- The curve represented by the equation (x - 1)² + (y + 2)² = 5 -/
def curve (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

/-- The minimum distance between a point on the line and a point on the curve -/
theorem min_distance_line_curve :
  ∃ (d : ℝ), d = Real.sqrt 5 / 5 ∧
  ∀ (t θ : ℝ),
    let (x₁, y₁) := line t
    let (x₂, y₂) := (1 + Real.sqrt 5 * Real.cos θ, -2 + Real.sqrt 5 * Real.sin θ)
    curve x₂ y₂ →
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_line_curve_l3394_339419


namespace NUMINAMATH_CALUDE_range_of_m_l3394_339425

def prop_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*m*x + 7*m - 10 ≠ 0

def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x^2 - m*x + 4 ≥ 0

theorem range_of_m :
  ∀ m : ℝ, (prop_p m ∨ prop_q m) ∧ (prop_p m ∧ prop_q m) →
  m ∈ Set.Ioo 2 4 ∪ {4} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3394_339425


namespace NUMINAMATH_CALUDE_garden_furniture_cost_l3394_339478

def bench_cost : ℝ := 150

def table_cost (bench_cost : ℝ) : ℝ := 2 * bench_cost

def combined_cost (bench_cost table_cost : ℝ) : ℝ := bench_cost + table_cost

theorem garden_furniture_cost : combined_cost bench_cost (table_cost bench_cost) = 450 := by
  sorry

end NUMINAMATH_CALUDE_garden_furniture_cost_l3394_339478


namespace NUMINAMATH_CALUDE_largest_valid_number_l3394_339477

def is_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n % 100 = (n / 100 % 10 + n / 1000) % 10) ∧
  (n % 10 = (n / 10 % 10 + n / 100 % 10) % 10)

theorem largest_valid_number : 
  (∀ m : ℕ, is_valid m → m ≤ 9099) ∧ is_valid 9099 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3394_339477


namespace NUMINAMATH_CALUDE_fiftieth_number_is_fourteen_l3394_339422

/-- Defines the cumulative sum of elements up to the nth row -/
def cumulativeSum (n : ℕ) : ℕ := 
  (List.range n).foldl (fun acc i => acc + 2 * (i + 1)) 0

/-- Defines the value of each element in the nth row -/
def rowValue (n : ℕ) : ℕ := 2 * n

theorem fiftieth_number_is_fourteen : 
  ∃ (n : ℕ), cumulativeSum n < 50 ∧ 50 ≤ cumulativeSum (n + 1) ∧ rowValue (n + 1) = 14 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_number_is_fourteen_l3394_339422


namespace NUMINAMATH_CALUDE_modulus_of_z_equals_sqrt2_over_2_l3394_339496

/-- The modulus of the complex number z = i / (1 - i) is equal to √2/2 -/
theorem modulus_of_z_equals_sqrt2_over_2 : 
  let z : ℂ := Complex.I / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_equals_sqrt2_over_2_l3394_339496


namespace NUMINAMATH_CALUDE_xiaomin_final_score_l3394_339453

/-- Calculates the final score for the "Book-loving Youth" selection -/
def final_score (honor_score : ℝ) (speech_score : ℝ) : ℝ :=
  0.4 * honor_score + 0.6 * speech_score

/-- Theorem: Xiaomin's final score in the "Book-loving Youth" selection is 86 points -/
theorem xiaomin_final_score :
  let honor_score := 80
  let speech_score := 90
  final_score honor_score speech_score = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_xiaomin_final_score_l3394_339453


namespace NUMINAMATH_CALUDE_ab_equals_zero_l3394_339412

theorem ab_equals_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_zero_l3394_339412


namespace NUMINAMATH_CALUDE_sum_is_zero_or_negative_two_l3394_339414

-- Define the conditions
def is_neither_positive_nor_negative (a : ℝ) : Prop := a = 0

def largest_negative_integer (b : ℤ) : Prop := b = -1

def reciprocal_is_self (c : ℝ) : Prop := c = 1 ∨ c = -1

-- Theorem statement
theorem sum_is_zero_or_negative_two 
  (a : ℝ) (b : ℤ) (c : ℝ) 
  (ha : is_neither_positive_nor_negative a)
  (hb : largest_negative_integer b)
  (hc : reciprocal_is_self c) :
  a + b + c = 0 ∨ a + b + c = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_zero_or_negative_two_l3394_339414


namespace NUMINAMATH_CALUDE_opera_ticket_price_increase_l3394_339441

theorem opera_ticket_price_increase (last_year_price this_year_price : ℝ) 
  (h1 : last_year_price = 85)
  (h2 : this_year_price = 102) :
  (this_year_price - last_year_price) / last_year_price * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_opera_ticket_price_increase_l3394_339441


namespace NUMINAMATH_CALUDE_elena_operation_l3394_339460

theorem elena_operation (x : ℝ) : (((3 * x + 5) - 3) * 2) / 2 = 17 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_elena_operation_l3394_339460


namespace NUMINAMATH_CALUDE_julia_tag_monday_l3394_339449

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 14

/-- The additional number of kids Julia played tag with on Monday compared to Tuesday -/
def additional_monday_kids : ℕ := 8

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := tuesday_kids + additional_monday_kids

theorem julia_tag_monday : monday_kids = 22 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_monday_l3394_339449


namespace NUMINAMATH_CALUDE_sister_dolls_count_hannah_dolls_relation_total_dolls_sum_l3394_339469

/-- The number of dolls Hannah's sister has -/
def sister_dolls : ℕ := 8

/-- The number of dolls Hannah has -/
def hannah_dolls : ℕ := 5 * sister_dolls

/-- The total number of dolls Hannah and her sister have -/
def total_dolls : ℕ := 48

theorem sister_dolls_count : sister_dolls = 8 :=
  by sorry

theorem hannah_dolls_relation : hannah_dolls = 5 * sister_dolls :=
  by sorry

theorem total_dolls_sum : sister_dolls + hannah_dolls = total_dolls :=
  by sorry

end NUMINAMATH_CALUDE_sister_dolls_count_hannah_dolls_relation_total_dolls_sum_l3394_339469


namespace NUMINAMATH_CALUDE_four_point_equal_inradii_congruent_triangles_l3394_339457

-- Define a type for points in a plane
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a type for triangles
structure Triangle :=
  (a b c : Point)

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop := sorry

-- Define a function to calculate the inradius of a triangle
def inradius (t : Triangle) : ℝ := sorry

-- Define a function to check if two triangles are congruent
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Main theorem
theorem four_point_equal_inradii_congruent_triangles 
  (A B C D : Point) : 
  (¬collinear A B C ∧ ¬collinear A B D ∧ ¬collinear A C D ∧ ¬collinear B C D) →
  (inradius ⟨A, B, C⟩ = inradius ⟨A, B, D⟩) →
  (inradius ⟨A, B, C⟩ = inradius ⟨A, C, D⟩) →
  (inradius ⟨A, B, C⟩ = inradius ⟨B, C, D⟩) →
  (congruent ⟨A, B, C⟩ ⟨A, B, D⟩) ∧ 
  (congruent ⟨A, B, C⟩ ⟨A, C, D⟩) ∧ 
  (congruent ⟨A, B, C⟩ ⟨B, C, D⟩) :=
by sorry

end NUMINAMATH_CALUDE_four_point_equal_inradii_congruent_triangles_l3394_339457


namespace NUMINAMATH_CALUDE_nine_points_chords_l3394_339415

/-- The number of different chords that can be drawn by connecting two points
    out of n points marked on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of different chords that can be drawn by connecting two points
    out of nine points marked on the circumference of a circle is equal to 36 -/
theorem nine_points_chords : num_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_points_chords_l3394_339415


namespace NUMINAMATH_CALUDE_second_number_11th_row_l3394_339481

/-- Given a lattice with 11 rows, where each row contains 6 numbers,
    and the last number in each row is n × 6 (where n is the row number),
    prove that the second number in the 11th row is 62. -/
theorem second_number_11th_row (rows : Nat) (numbers_per_row : Nat)
    (last_number : Nat → Nat) :
  rows = 11 →
  numbers_per_row = 6 →
  (∀ n, last_number n = n * numbers_per_row) →
  (last_number 10 + 2 = 62) := by
  sorry

end NUMINAMATH_CALUDE_second_number_11th_row_l3394_339481


namespace NUMINAMATH_CALUDE_kirill_height_l3394_339490

/-- Represents the heights of Kirill, his brother, sister, and cousin --/
structure FamilyHeights where
  kirill : ℕ
  brother : ℕ
  sister : ℕ
  cousin : ℕ

/-- The conditions of the problem --/
def HeightConditions (h : FamilyHeights) : Prop :=
  h.brother = h.kirill + 14 ∧
  h.sister = 2 * h.kirill ∧
  h.cousin = h.sister + 3 ∧
  h.kirill + h.brother + h.sister + h.cousin = 432

/-- The theorem stating Kirill's height --/
theorem kirill_height :
  ∀ h : FamilyHeights, HeightConditions h → h.kirill = 69 :=
by sorry

end NUMINAMATH_CALUDE_kirill_height_l3394_339490


namespace NUMINAMATH_CALUDE_ceiling_times_self_156_l3394_339439

theorem ceiling_times_self_156 :
  ∃! (y : ℝ), ⌈y⌉ * y = 156 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ceiling_times_self_156_l3394_339439


namespace NUMINAMATH_CALUDE_at_least_two_satisfying_functions_l3394_339433

/-- A function satisfying the given equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + (f y)^2) = x + y^2

/-- Theorem stating that there are at least two distinct functions satisfying the equation -/
theorem at_least_two_satisfying_functions :
  ∃ f g : ℝ → ℝ, f ≠ g ∧ SatisfyingFunction f ∧ SatisfyingFunction g :=
sorry

end NUMINAMATH_CALUDE_at_least_two_satisfying_functions_l3394_339433


namespace NUMINAMATH_CALUDE_julie_work_hours_l3394_339427

/-- Given Julie's work schedule and earnings, calculate her required weekly hours during the school year --/
theorem julie_work_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ)
                         (school_year_weeks : ℕ) (school_year_earnings : ℕ) :
  summer_weeks = 12 →
  summer_hours_per_week = 48 →
  summer_earnings = 5000 →
  school_year_weeks = 48 →
  school_year_earnings = 5000 →
  (summer_hours_per_week * summer_weeks * school_year_earnings) / (summer_earnings * school_year_weeks) = 12 := by
  sorry

end NUMINAMATH_CALUDE_julie_work_hours_l3394_339427


namespace NUMINAMATH_CALUDE_x_days_to_complete_work_l3394_339450

/-- The number of days required for x and y to complete the work together -/
def days_xy : ℚ := 12

/-- The number of days required for y to complete the work alone -/
def days_y : ℚ := 24

/-- The fraction of work completed by a worker in one day -/
def work_per_day (days : ℚ) : ℚ := 1 / days

theorem x_days_to_complete_work : 
  1 / (work_per_day days_xy - work_per_day days_y) = 24 := by sorry

end NUMINAMATH_CALUDE_x_days_to_complete_work_l3394_339450


namespace NUMINAMATH_CALUDE_set_equality_l3394_339498

theorem set_equality : {x : ℕ | x - 1 ≤ 2} = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_set_equality_l3394_339498


namespace NUMINAMATH_CALUDE_min_distance_curve_line_l3394_339465

/-- The minimum distance between a curve and a line -/
theorem min_distance_curve_line (a m n : ℝ) (h1 : a > 0) :
  let b := -1/2 * a^2 + 3 * Real.log a
  let line := {p : ℝ × ℝ | p.2 = 2 * p.1 + 1/2}
  let Q := (m, n)
  Q ∈ line →
  ∃ (min_dist : ℝ), min_dist = 9/5 ∧
    ∀ (p : ℝ × ℝ), p ∈ line → (a - p.1)^2 + (b - p.2)^2 ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_line_l3394_339465


namespace NUMINAMATH_CALUDE_distinct_scores_is_nineteen_l3394_339467

/-- Represents the number of distinct possible scores for a basketball player -/
def distinctScores : ℕ :=
  let shotTypes := 3  -- free throw, 2-point basket, 3-point basket
  let totalShots := 8
  let pointValues := [1, 2, 3]
  19  -- The actual count of distinct scores

/-- Theorem stating that the number of distinct possible scores is 19 -/
theorem distinct_scores_is_nineteen :
  distinctScores = 19 := by sorry

end NUMINAMATH_CALUDE_distinct_scores_is_nineteen_l3394_339467


namespace NUMINAMATH_CALUDE_spatial_geometry_theorem_l3394_339461

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- State the theorem
theorem spatial_geometry_theorem 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (perpendicular m β ∧ perpendicular n β → parallel_lines m n) ∧
  (perpendicular m α ∧ perpendicular m β → parallel_planes α β) :=
sorry

end NUMINAMATH_CALUDE_spatial_geometry_theorem_l3394_339461


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l3394_339489

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (y : ℕ), y = Nat.lcm x (Nat.lcm 15 21) ∧ y = 105) →
  x ≤ 105 ∧ ∃ (z : ℕ), z = 105 ∧ (∃ (w : ℕ), w = Nat.lcm z (Nat.lcm 15 21) ∧ w = 105) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l3394_339489


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3394_339479

theorem trigonometric_problem (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : 0 < β) (h4 : β < π/2) 
  (h5 : Real.sin (π/3 - α) = 3/5) 
  (h6 : Real.cos (β/2 - π/3) = 2*Real.sqrt 5/5) : 
  (Real.sin α = (4*Real.sqrt 3 - 3)/10) ∧ 
  (Real.cos (β/2 - α) = 11*Real.sqrt 5/25) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3394_339479


namespace NUMINAMATH_CALUDE_second_number_is_thirteen_l3394_339431

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstNumber : ℕ) (n : ℕ) : ℕ :=
  firstNumber + (totalStudents / sampleSize) * (n - 1)

/-- Theorem: In the given systematic sampling, the second number is 13 -/
theorem second_number_is_thirteen :
  let totalStudents := 500
  let sampleSize := 50
  let firstNumber := 3
  systematicSample totalStudents sampleSize firstNumber 2 = 13 := by
  sorry


end NUMINAMATH_CALUDE_second_number_is_thirteen_l3394_339431


namespace NUMINAMATH_CALUDE_min_tan_half_angle_l3394_339443

theorem min_tan_half_angle (A B C : Real) (h1 : A + B + C = π) 
  (h2 : Real.tan (A/2) + Real.tan (B/2) = 1) :
  ∃ (m : Real), m = 3/4 ∧ ∀ x, x = Real.tan (C/2) → x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_tan_half_angle_l3394_339443


namespace NUMINAMATH_CALUDE_camera_price_difference_l3394_339423

-- Define the list price
def list_price : ℚ := 49.95

-- Define the price at Budget Buys
def budget_buys_price : ℚ := list_price - 10

-- Define the price at Value Mart
def value_mart_price : ℚ := list_price * (1 - 0.2)

-- Theorem to prove
theorem camera_price_difference :
  (max budget_buys_price value_mart_price - min budget_buys_price value_mart_price) * 100 = 1 := by
  sorry


end NUMINAMATH_CALUDE_camera_price_difference_l3394_339423


namespace NUMINAMATH_CALUDE_phone_cost_calculation_phone_cost_proof_l3394_339401

theorem phone_cost_calculation (current_percentage : Real) (additional_amount : Real) : Real :=
  let total_cost := additional_amount / (1 - current_percentage)
  total_cost

theorem phone_cost_proof :
  phone_cost_calculation 0.4 780 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_phone_cost_calculation_phone_cost_proof_l3394_339401


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3394_339472

theorem exponential_equation_solution :
  ∃ x : ℝ, (2 : ℝ) ^ (x + 6) = 64 ^ (x - 1) ∧ x = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3394_339472


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3394_339446

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℤ, x^3 < 1) ↔ (∃ x : ℤ, x^3 ≥ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3394_339446


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3394_339420

theorem system_of_equations_solution (x y z : ℤ) 
  (eq1 : x + y + z = 600)
  (eq2 : x - y = 200)
  (eq3 : x + z = 500) :
  x = 300 ∧ y = 100 ∧ z = 200 :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3394_339420


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3394_339417

def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1, 3}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3394_339417


namespace NUMINAMATH_CALUDE_candy_bar_calories_l3394_339499

theorem candy_bar_calories
  (distance : ℕ) -- Total distance walked
  (calories_per_mile : ℕ) -- Calories burned per mile
  (net_deficit : ℕ) -- Net calorie deficit
  (h1 : distance = 3) -- Cary walks 3 miles round-trip
  (h2 : calories_per_mile = 150) -- Cary burns 150 calories per mile
  (h3 : net_deficit = 250) -- Cary's net calorie deficit is 250 calories
  : distance * calories_per_mile - net_deficit = 200 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_calories_l3394_339499


namespace NUMINAMATH_CALUDE_mixed_committee_probability_l3394_339406

def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 6

def probability_mixed_committee : ℚ :=
  1 - (2 * Nat.choose boys committee_size) / Nat.choose total_members committee_size

theorem mixed_committee_probability :
  probability_mixed_committee = 33187 / 33649 :=
sorry

end NUMINAMATH_CALUDE_mixed_committee_probability_l3394_339406


namespace NUMINAMATH_CALUDE_select_twelve_students_l3394_339483

/-- Represents the number of students in each course -/
structure CourseDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the sampling information -/
structure SamplingInfo where
  total_students : ℕ
  selected_students : ℕ

/-- Checks if the course distribution forms an arithmetic sequence with the given common difference -/
def is_arithmetic_sequence (dist : CourseDistribution) (diff : ℤ) : Prop :=
  dist.second = dist.first - diff ∧ dist.third = dist.second - diff

/-- Calculates the number of students to be selected from the first course -/
def students_to_select (dist : CourseDistribution) (info : SamplingInfo) : ℕ :=
  (dist.first * info.selected_students) / info.total_students

/-- Main theorem: Given the conditions, prove that 12 students should be selected from the first course -/
theorem select_twelve_students 
  (dist : CourseDistribution)
  (info : SamplingInfo)
  (h1 : dist.first + dist.second + dist.third = info.total_students)
  (h2 : info.total_students = 600)
  (h3 : info.selected_students = 30)
  (h4 : is_arithmetic_sequence dist (-40)) :
  students_to_select dist info = 12 := by
  sorry

end NUMINAMATH_CALUDE_select_twelve_students_l3394_339483


namespace NUMINAMATH_CALUDE_product_less_than_factor_l3394_339447

theorem product_less_than_factor : ∃ (a b : ℝ), 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ a * b < min a b := by
  sorry

end NUMINAMATH_CALUDE_product_less_than_factor_l3394_339447


namespace NUMINAMATH_CALUDE_min_colors_for_grid_l3394_339482

-- Define the grid as a type alias for pairs of integers
def Grid := ℤ × ℤ

-- Define the distance function between two cells
def distance (a b : Grid) : ℕ :=
  max (Int.natAbs (a.1 - b.1)) (Int.natAbs (a.2 - b.2))

-- Define the color function
def color (cell : Grid) : Fin 4 :=
  Fin.ofNat ((cell.1 + cell.2).natAbs % 4)

-- State the theorem
theorem min_colors_for_grid : 
  (∀ a b : Grid, distance a b = 6 → color a ≠ color b) ∧
  (∀ n : ℕ, n < 4 → ∃ a b : Grid, distance a b = 6 ∧ 
    Fin.ofNat (n % 4) = color a ∧ Fin.ofNat (n % 4) = color b) :=
sorry

end NUMINAMATH_CALUDE_min_colors_for_grid_l3394_339482


namespace NUMINAMATH_CALUDE_count_solutions_x_plus_y_plus_z_eq_10_l3394_339436

def positive_integer_solutions (n : ℕ) : ℕ :=
  (n - 1) * (n - 2) / 2

theorem count_solutions_x_plus_y_plus_z_eq_10 :
  positive_integer_solutions 10 = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_x_plus_y_plus_z_eq_10_l3394_339436


namespace NUMINAMATH_CALUDE_expression_equals_two_l3394_339464

theorem expression_equals_two (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) :
  (2 * x^2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l3394_339464


namespace NUMINAMATH_CALUDE_jeff_score_problem_l3394_339473

theorem jeff_score_problem (scores : List ℝ) (desired_average : ℝ) : 
  scores = [89, 92, 88, 95, 91] → 
  desired_average = 93 → 
  (scores.sum + 103) / 6 = desired_average :=
by sorry

end NUMINAMATH_CALUDE_jeff_score_problem_l3394_339473


namespace NUMINAMATH_CALUDE_quadrilateral_area_between_squares_l3394_339497

/-- Represents a square with a given side length -/
structure Square (α : Type*) [LinearOrderedField α] where
  side : α

/-- Represents a trapezoid with given bases and height -/
structure Trapezoid (α : Type*) [LinearOrderedField α] where
  base1 : α
  base2 : α
  height : α

/-- Calculates the area of a trapezoid -/
def trapezoid_area {α : Type*} [LinearOrderedField α] (t : Trapezoid α) : α :=
  (t.base1 + t.base2) * t.height / 2

theorem quadrilateral_area_between_squares
  (s1 s2 s3 s4 : Square ℝ)
  (h1 : s1.side = 2)
  (h2 : s2.side = 4)
  (h3 : s3.side = 6)
  (h4 : s4.side = 8)
  (h_coplanar : True)  -- Assumption of coplanarity
  (h_arrangement : True)  -- Assumption of side-by-side arrangement on line AB
  : ∃ t : Trapezoid ℝ, 
    t.base1 = 3 ∧ 
    t.base2 = 10 ∧ 
    t.height = 2 ∧ 
    trapezoid_area t = 13 :=
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_between_squares_l3394_339497


namespace NUMINAMATH_CALUDE_root_product_expression_l3394_339448

theorem root_product_expression (p q : ℝ) 
  (hα : ∃ α : ℝ, α^2 + p*α + 2 = 0)
  (hβ : ∃ β : ℝ, β^2 + p*β + 2 = 0)
  (hγ : ∃ γ : ℝ, γ^2 + q*γ - 3 = 0)
  (hδ : ∃ δ : ℝ, δ^2 + q*δ - 3 = 0)
  (hαβ_distinct : ∀ α β : ℝ, α^2 + p*α + 2 = 0 → β^2 + p*β + 2 = 0 → α ≠ β)
  (hγδ_distinct : ∀ γ δ : ℝ, γ^2 + q*γ - 3 = 0 → δ^2 + q*δ - 3 = 0 → γ ≠ δ) :
  ∃ (α β γ δ : ℝ), (α - γ)*(β - γ)*(α + δ)*(β + δ) = 3*(q^2 - p^2) + 15 :=
by sorry

end NUMINAMATH_CALUDE_root_product_expression_l3394_339448


namespace NUMINAMATH_CALUDE_honey_water_percentage_l3394_339400

/-- Given that 1.7 kg of flower-nectar containing 50% water yields 1 kg of honey,
    prove that the percentage of water in the resulting honey is 15%. -/
theorem honey_water_percentage
  (nectar_weight : ℝ)
  (honey_weight : ℝ)
  (nectar_water_percentage : ℝ)
  (h1 : nectar_weight = 1.7)
  (h2 : honey_weight = 1)
  (h3 : nectar_water_percentage = 50)
  : (honey_weight - (nectar_weight * (1 - nectar_water_percentage / 100))) / honey_weight * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_honey_water_percentage_l3394_339400


namespace NUMINAMATH_CALUDE_yoga_time_calculation_l3394_339418

/-- Calculates the yoga time given exercise ratios and bicycle riding time -/
def yoga_time (gym_bike_ratio : Rat) (yoga_exercise_ratio : Rat) (bike_time : ℕ) : ℕ :=
  let gym_time := (gym_bike_ratio.num * bike_time) / gym_bike_ratio.den
  let total_exercise_time := gym_time + bike_time
  ((yoga_exercise_ratio.num * total_exercise_time) / yoga_exercise_ratio.den).toNat

/-- Proves that given the specified ratios and bicycle riding time, the yoga time is 20 minutes -/
theorem yoga_time_calculation :
  yoga_time (2 / 3) (2 / 3) 18 = 20 := by
  sorry

#eval yoga_time (2 / 3) (2 / 3) 18

end NUMINAMATH_CALUDE_yoga_time_calculation_l3394_339418


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3394_339402

theorem binomial_coefficient_ratio (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (3*x - 2)^6 = a₀ + a₁*(2*x - 1) + a₂*(2*x - 1)^2 + a₃*(2*x - 1)^3 + 
                 a₄*(2*x - 1)^4 + a₅*(2*x - 1)^5 + a₆*(2*x - 1)^6 →
  (a₁ + a₃ + a₅) / (a₀ + a₂ + a₄ + a₆) = -63/65 := by
sorry


end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3394_339402


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_40_integers_from_7_l3394_339471

theorem arithmetic_mean_of_40_integers_from_7 :
  let start : ℕ := 7
  let count : ℕ := 40
  let sequence := (fun i => start + i - 1)
  let sum := (sequence 1 + sequence count) * count / 2
  (sum : ℚ) / count = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_40_integers_from_7_l3394_339471


namespace NUMINAMATH_CALUDE_min_value_a_l3394_339407

theorem min_value_a (m n : ℝ) (h1 : 0 < n) (h2 : n < m) (h3 : m < 1/a) 
  (h4 : (n^(1/m)) / (m^(1/n)) > (n^a) / (m^a)) : 
  ∀ ε > 0, ∃ a : ℝ, a ≥ 1 ∧ a < 1 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l3394_339407


namespace NUMINAMATH_CALUDE_survey_participants_l3394_339404

theorem survey_participants (sample : ℕ) (percentage : ℚ) (total : ℕ) 
  (h1 : sample = 40)
  (h2 : percentage = 20 / 100)
  (h3 : sample = percentage * total) :
  total = 200 := by
sorry

end NUMINAMATH_CALUDE_survey_participants_l3394_339404


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_quadratic_form_minimum_attainable_l3394_339487

theorem quadratic_form_minimum (x y : ℝ) :
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5 ≥ 8 :=
by sorry

theorem quadratic_form_minimum_attainable :
  ∃ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5 = 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_quadratic_form_minimum_attainable_l3394_339487


namespace NUMINAMATH_CALUDE_cost_difference_is_two_point_five_l3394_339440

/-- Represents the pizza sharing scenario between Bob and Samantha -/
structure PizzaSharing where
  totalSlices : ℕ
  plainPizzaCost : ℚ
  oliveCost : ℚ
  bobOliveSlices : ℕ
  bobPlainSlices : ℕ

/-- Calculates the cost difference between Bob and Samantha's payments -/
def costDifference (ps : PizzaSharing) : ℚ :=
  let totalCost := ps.plainPizzaCost + ps.oliveCost
  let costPerSlice := totalCost / ps.totalSlices
  let bobCost := costPerSlice * (ps.bobOliveSlices + ps.bobPlainSlices)
  let samanthaCost := costPerSlice * (ps.totalSlices - ps.bobOliveSlices - ps.bobPlainSlices)
  bobCost - samanthaCost

/-- Theorem stating that the cost difference is $2.5 -/
theorem cost_difference_is_two_point_five :
  let ps : PizzaSharing := {
    totalSlices := 12,
    plainPizzaCost := 12,
    oliveCost := 3,
    bobOliveSlices := 4,
    bobPlainSlices := 3
  }
  costDifference ps = 5/2 := by sorry

end NUMINAMATH_CALUDE_cost_difference_is_two_point_five_l3394_339440


namespace NUMINAMATH_CALUDE_tangent_line_intersection_product_l3394_339438

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = x / 2
def asymptote2 (x y : ℝ) : Prop := y = -x / 2

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2

-- Define a line tangent to the hyperbola at point P
def tangent_line (l : ℝ × ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  point_on_hyperbola P ∧ l P

-- Define intersection points of the tangent line with asymptotes
def intersection_points (l : ℝ × ℝ → Prop) (M N : ℝ × ℝ) : Prop :=
  l M ∧ l N ∧ asymptote1 M.1 M.2 ∧ asymptote2 N.1 N.2

-- Theorem statement
theorem tangent_line_intersection_product (P M N : ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  point_on_hyperbola P →
  tangent_line l P →
  intersection_points l M N →
  M.1 * N.1 + M.2 * N.2 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_product_l3394_339438


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_lub_l3394_339456

/-- A trapezoid inscribed in a unit circle -/
structure InscribedTrapezoid where
  /-- The length of side AB -/
  s₁ : ℝ
  /-- The length of side CD -/
  s₂ : ℝ
  /-- The distance from the center to the intersection of diagonals -/
  d : ℝ
  /-- s₁ and s₂ are between 0 and 2 (diameter of unit circle) -/
  h_s₁_bounds : 0 < s₁ ∧ s₁ ≤ 2
  h_s₂_bounds : 0 < s₂ ∧ s₂ ≤ 2
  /-- d is positive (intersection is not at the center) -/
  h_d_pos : d > 0

/-- The least upper bound of (s₁ - s₂) / d for inscribed trapezoids is 2 -/
theorem inscribed_trapezoid_lub :
  ∀ T : InscribedTrapezoid, (T.s₁ - T.s₂) / T.d ≤ 2 ∧
  ∀ ε > 0, ∃ T : InscribedTrapezoid, (T.s₁ - T.s₂) / T.d > 2 - ε := by
  sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_lub_l3394_339456


namespace NUMINAMATH_CALUDE_inequalities_proof_l3394_339485

theorem inequalities_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3394_339485


namespace NUMINAMATH_CALUDE_lcm_24_36_45_l3394_339424

theorem lcm_24_36_45 : Nat.lcm 24 (Nat.lcm 36 45) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_45_l3394_339424


namespace NUMINAMATH_CALUDE_chicken_salad_cost_l3394_339491

/-- Given the following conditions:
  - Lee and his friend had a total of $18
  - Chicken wings cost $6
  - Two sodas cost $2 in total
  - The tax was $3
  - They received $3 in change
  Prove that the chicken salad cost $4 -/
theorem chicken_salad_cost (total_money : ℕ) (wings_cost : ℕ) (sodas_cost : ℕ) (tax : ℕ) (change : ℕ) :
  total_money = 18 →
  wings_cost = 6 →
  sodas_cost = 2 →
  tax = 3 →
  change = 3 →
  total_money - change - (wings_cost + sodas_cost + tax) = 4 :=
by sorry

end NUMINAMATH_CALUDE_chicken_salad_cost_l3394_339491


namespace NUMINAMATH_CALUDE_find_M_l3394_339468

theorem find_M : ∃ (M : ℕ+), (36 : ℕ)^2 * 81^2 = 18^2 * M^2 ∧ M = 162 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l3394_339468


namespace NUMINAMATH_CALUDE_birdhouse_to_lawn_chair_ratio_l3394_339409

def car_distance : ℝ := 200
def lawn_chair_distance : ℝ := 2 * car_distance
def birdhouse_distance : ℝ := 1200

theorem birdhouse_to_lawn_chair_ratio :
  birdhouse_distance / lawn_chair_distance = 3 := by sorry

end NUMINAMATH_CALUDE_birdhouse_to_lawn_chair_ratio_l3394_339409


namespace NUMINAMATH_CALUDE_interval_length_implies_difference_l3394_339444

theorem interval_length_implies_difference (r s : ℝ) : 
  (∀ x, r ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ s) → 
  ((s - 4) / 3 - (r - 4) / 3 = 12) → 
  s - r = 36 := by
sorry

end NUMINAMATH_CALUDE_interval_length_implies_difference_l3394_339444


namespace NUMINAMATH_CALUDE_product_of_integers_l3394_339463

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 26)
  (diff_sq_eq : x^2 - y^2 = 52) : 
  x * y = 168 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l3394_339463


namespace NUMINAMATH_CALUDE_sum_positive_implies_one_positive_l3394_339462

theorem sum_positive_implies_one_positive (a b : ℝ) : 
  a + b > 0 → a > 0 ∨ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_one_positive_l3394_339462


namespace NUMINAMATH_CALUDE_mary_chopped_six_tables_l3394_339437

/-- Represents the number of sticks of wood produced by different furniture items -/
structure FurnitureWood where
  chair : Nat
  table : Nat
  stool : Nat

/-- Represents the chopping and burning scenario -/
structure WoodScenario where
  furniture : FurnitureWood
  chopped_chairs : Nat
  chopped_stools : Nat
  burn_rate : Nat
  warm_hours : Nat

/-- Calculates the number of tables chopped given a wood scenario -/
def tables_chopped (scenario : WoodScenario) : Nat :=
  let total_wood := scenario.warm_hours * scenario.burn_rate
  let wood_from_chairs := scenario.chopped_chairs * scenario.furniture.chair
  let wood_from_stools := scenario.chopped_stools * scenario.furniture.stool
  let wood_from_tables := total_wood - wood_from_chairs - wood_from_stools
  wood_from_tables / scenario.furniture.table

theorem mary_chopped_six_tables :
  let mary_scenario : WoodScenario := {
    furniture := { chair := 6, table := 9, stool := 2 },
    chopped_chairs := 18,
    chopped_stools := 4,
    burn_rate := 5,
    warm_hours := 34
  }
  tables_chopped mary_scenario = 6 := by
  sorry

end NUMINAMATH_CALUDE_mary_chopped_six_tables_l3394_339437


namespace NUMINAMATH_CALUDE_math_competition_score_xiao_hua_correct_answers_l3394_339452

theorem math_competition_score (total_questions : Nat) (correct_points : Int) (wrong_points : Int) (total_score : Int) : Int :=
  let attempted_questions := total_questions
  let hypothetical_score := total_questions * correct_points
  let score_difference := hypothetical_score - total_score
  let points_per_wrong_answer := correct_points + wrong_points
  let wrong_answers := score_difference / points_per_wrong_answer
  total_questions - wrong_answers

theorem xiao_hua_correct_answers : 
  math_competition_score 15 8 (-4) 72 = 11 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_score_xiao_hua_correct_answers_l3394_339452


namespace NUMINAMATH_CALUDE_fraction_is_one_twelfth_l3394_339411

def problem (A E : ℝ) (f : ℝ) : Prop :=
  -- Al has more money than Eliot
  A > E ∧
  -- Eliot has $200 in his account
  E = 200 ∧
  -- The difference between their accounts is a fraction of the sum of their accounts
  A - E = f * (A + E) ∧
  -- If Al's account increases by 10% and Eliot's by 20%, Al would have $20 more than Eliot
  A * 1.1 = E * 1.2 + 20

theorem fraction_is_one_twelfth (A E : ℝ) (f : ℝ) :
  problem A E f → f = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_one_twelfth_l3394_339411


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3394_339410

/-- Given two points P and Q symmetric with respect to the y-axis, prove that the sum of their x-coordinates is -8 --/
theorem symmetric_points_sum (a b : ℝ) : 
  (∃ (P Q : ℝ × ℝ), P = (a, -3) ∧ Q = (4, b) ∧ 
   (P.1 = -Q.1) ∧ (P.2 = Q.2)) → 
  a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3394_339410


namespace NUMINAMATH_CALUDE_john_purchase_proof_l3394_339451

def john_purchase (q : ℝ) : Prop :=
  let initial_money : ℝ := 50
  let drink_cost : ℝ := q
  let small_pizza_cost : ℝ := 1.5 * q
  let medium_pizza_cost : ℝ := 2.5 * q
  let total_cost : ℝ := 2 * drink_cost + small_pizza_cost + medium_pizza_cost
  let money_left : ℝ := initial_money - total_cost
  money_left = 50 - 6 * q

theorem john_purchase_proof (q : ℝ) : john_purchase q := by
  sorry

end NUMINAMATH_CALUDE_john_purchase_proof_l3394_339451


namespace NUMINAMATH_CALUDE_quilt_cost_calculation_l3394_339430

/-- The cost of a rectangular quilt -/
def quilt_cost (length width price_per_sq_ft : ℝ) : ℝ :=
  length * width * price_per_sq_ft

/-- Theorem: The cost of a 12 ft by 15 ft quilt at $70 per square foot is $12,600 -/
theorem quilt_cost_calculation :
  quilt_cost 12 15 70 = 12600 := by
  sorry

end NUMINAMATH_CALUDE_quilt_cost_calculation_l3394_339430


namespace NUMINAMATH_CALUDE_average_temperature_l3394_339455

def temperatures : List ℝ := [53, 59, 61, 55, 50]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 55.6 := by sorry

end NUMINAMATH_CALUDE_average_temperature_l3394_339455


namespace NUMINAMATH_CALUDE_rabbit_farm_number_l3394_339428

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem rabbit_farm_number : 
  ∃! n : ℕ, is_six_digit n ∧ 
            is_perfect_square n ∧ 
            is_perfect_cube n ∧ 
            is_prime (n - 6) ∧ 
            n = 117649 :=
by sorry

end NUMINAMATH_CALUDE_rabbit_farm_number_l3394_339428


namespace NUMINAMATH_CALUDE_lathe_problem_l3394_339454

/-- Represents the work efficiency of a lathe -/
structure Efficiency : Type :=
  (value : ℝ)

/-- Represents a lathe with its efficiency and start time -/
structure Lathe : Type :=
  (efficiency : Efficiency)
  (startTime : ℝ)

/-- The number of parts processed by a lathe after a given time -/
def partsProcessed (l : Lathe) (time : ℝ) : ℝ :=
  l.efficiency.value * (time - l.startTime)

theorem lathe_problem (a b c : Lathe) :
  a.startTime = c.startTime - 10 →
  c.startTime = b.startTime - 5 →
  partsProcessed b (b.startTime + 10) = partsProcessed c (b.startTime + 10) →
  partsProcessed a (c.startTime + 30) = partsProcessed c (c.startTime + 30) →
  ∃ t : ℝ, t = 15 ∧ partsProcessed a (b.startTime + t) = partsProcessed b (b.startTime + t) :=
by sorry

end NUMINAMATH_CALUDE_lathe_problem_l3394_339454


namespace NUMINAMATH_CALUDE_similar_triangle_shortest_side_l3394_339442

/-- Given two similar right triangles, where the first triangle has a side of 15 units and a hypotenuse of 34 units, and the second triangle has a hypotenuse of 68 units, the shortest side of the second triangle is 2√931 units. -/
theorem similar_triangle_shortest_side :
  ∀ (a b c d e : ℝ),
  a^2 + 15^2 = 34^2 →  -- Pythagorean theorem for the first triangle
  a ≤ 15 →  -- a is the shortest side of the first triangle
  c^2 + d^2 = 68^2 →  -- Pythagorean theorem for the second triangle
  c / a = d / 15 →  -- triangles are similar
  c / a = 68 / 34 →  -- ratio of hypotenuses
  c = 2 * Real.sqrt 931 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_shortest_side_l3394_339442


namespace NUMINAMATH_CALUDE_word_count_theorems_l3394_339476

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of words we're considering -/
def word_length : ℕ := 5

/-- The number of 5-letter words -/
def num_words : ℕ := alphabet_size ^ word_length

/-- The number of 5-letter words with all different letters -/
def num_words_diff : ℕ := 
  (List.range word_length).foldl (fun acc i => acc * (alphabet_size - i)) alphabet_size

/-- The number of 5-letter words without any letter repeating consecutively -/
def num_words_no_consec : ℕ := alphabet_size * (alphabet_size - 1)^(word_length - 1)

theorem word_count_theorems : 
  (num_words = 26^5) ∧ 
  (num_words_diff = 26 * 25 * 24 * 23 * 22) ∧ 
  (num_words_no_consec = 26 * 25^4) := by
  sorry

end NUMINAMATH_CALUDE_word_count_theorems_l3394_339476


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3394_339474

theorem inequality_system_solution (x : ℝ) :
  (4 * x + 5 > x - 1) ∧ ((3 * x - 1) / 2 < x) → -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3394_339474


namespace NUMINAMATH_CALUDE_rational_with_smallest_abs_value_l3394_339480

theorem rational_with_smallest_abs_value : ∃ q : ℚ, |q| < |1| := by
  -- The proof would go here, but we're only writing the statement
  sorry

end NUMINAMATH_CALUDE_rational_with_smallest_abs_value_l3394_339480


namespace NUMINAMATH_CALUDE_largest_c_for_g_range_two_l3394_339458

/-- The function g(x) defined as x^2 - 5x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - 5*x + c

/-- Theorem stating that the largest value of c such that 2 is in the range of g(x) is 33/4 -/
theorem largest_c_for_g_range_two :
  ∃ (c_max : ℝ), c_max = 33/4 ∧
  (∀ c : ℝ, (∃ x : ℝ, g c x = 2) → c ≤ c_max) ∧
  (∃ x : ℝ, g c_max x = 2) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_g_range_two_l3394_339458


namespace NUMINAMATH_CALUDE_factor_expression_l3394_339466

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3394_339466


namespace NUMINAMATH_CALUDE_constant_expression_in_linear_system_l3394_339493

theorem constant_expression_in_linear_system (a k : ℝ) (x y : ℝ → ℝ) :
  (∀ a, x a + 2 * y a = -a + 1) →
  (∀ a, x a - 3 * y a = 4 * a + 6) →
  (∃ c, ∀ a, k * x a - y a = c) →
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_constant_expression_in_linear_system_l3394_339493


namespace NUMINAMATH_CALUDE_consecutive_color_draw_probability_l3394_339495

def num_green_chips : ℕ := 4
def num_blue_chips : ℕ := 3
def num_red_chips : ℕ := 5
def total_chips : ℕ := num_green_chips + num_blue_chips + num_red_chips

theorem consecutive_color_draw_probability :
  (Nat.factorial 3 * Nat.factorial num_green_chips * Nat.factorial num_blue_chips * Nat.factorial num_red_chips) / 
  Nat.factorial total_chips = 1 / 4620 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_color_draw_probability_l3394_339495


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3394_339488

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (1 + Complex.I) ^ 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3394_339488


namespace NUMINAMATH_CALUDE_product_of_numbers_l3394_339408

theorem product_of_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3394_339408


namespace NUMINAMATH_CALUDE_total_cost_pants_and_belt_l3394_339492

theorem total_cost_pants_and_belt (pants_price belt_price total_cost : ℝ) :
  pants_price = 34 →
  pants_price = belt_price - 2.93 →
  total_cost = pants_price + belt_price →
  total_cost = 70.93 := by
sorry

end NUMINAMATH_CALUDE_total_cost_pants_and_belt_l3394_339492


namespace NUMINAMATH_CALUDE_constant_function_theorem_l3394_339484

def IsNonZeroInteger (x : ℚ) : Prop := ∃ (n : ℤ), n ≠ 0 ∧ x = n

theorem constant_function_theorem (f : ℚ → ℚ) 
  (h : ∀ x y, IsNonZeroInteger x → IsNonZeroInteger y → 
    f ((x + y) / 3) = (f x + f y) / 2) :
  ∃ c, ∀ x, IsNonZeroInteger x → f x = c := by
sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l3394_339484


namespace NUMINAMATH_CALUDE_triangle_problem_l3394_339432

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  -- Condition 1: Vectors are parallel
  (2 * Real.sin (t.A / 2) * (2 * Real.cos (t.A / 4)^2 - 1) = Real.sqrt 3 * Real.cos t.A) →
  -- Condition 2: a = √7
  (t.a = Real.sqrt 7) →
  -- Condition 3: Area of triangle ABC is 3√3/2
  (1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 3 / 2) →
  -- Conclusion 1: A = π/3
  (t.A = Real.pi / 3) ∧
  -- Conclusion 2: b + c = 5
  (t.b + t.c = 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3394_339432
