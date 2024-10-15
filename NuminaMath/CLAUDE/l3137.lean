import Mathlib

namespace NUMINAMATH_CALUDE_a_range_l3137_313712

theorem a_range (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x : ℝ, |Real.sin x| > a)
  (h3 : ∀ x : ℝ, x ∈ [π/4, 3*π/4] → (Real.sin x)^2 + a * Real.sin x - 1 ≥ 0) :
  a ∈ Set.Ici (Real.sqrt 2 / 2) ∩ Set.Iio 1 :=
by sorry

end NUMINAMATH_CALUDE_a_range_l3137_313712


namespace NUMINAMATH_CALUDE_max_students_social_practice_l3137_313708

theorem max_students_social_practice (max_fund car_rental per_student_cost : ℕ) 
  (h1 : max_fund = 800)
  (h2 : car_rental = 300)
  (h3 : per_student_cost = 15) :
  ∃ (max_students : ℕ), 
    max_students = 33 ∧ 
    max_students * per_student_cost + car_rental ≤ max_fund ∧
    ∀ (n : ℕ), n * per_student_cost + car_rental ≤ max_fund → n ≤ max_students :=
sorry

end NUMINAMATH_CALUDE_max_students_social_practice_l3137_313708


namespace NUMINAMATH_CALUDE_always_quadratic_radical_l3137_313732

theorem always_quadratic_radical (a : ℝ) : ∃ (x : ℝ), x ^ 2 = a ^ 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_always_quadratic_radical_l3137_313732


namespace NUMINAMATH_CALUDE_monic_quartic_specific_values_l3137_313767

-- Define a monic quartic polynomial
def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_specific_values (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h1 : f (-2) = -4)
  (h2 : f 1 = -1)
  (h3 : f (-3) = -9)
  (h4 : f 5 = -25) :
  f 2 = -64 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_specific_values_l3137_313767


namespace NUMINAMATH_CALUDE_range_of_a_circles_intersect_l3137_313778

noncomputable section

-- Define the circles and line
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def circle_D (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x = 0
def line (x y a : ℝ) : Prop := x + y - a = 0

-- Define the inequality condition
def inequality_condition (x y m : ℝ) : Prop :=
  x^2 + y^2 - (m + Real.sqrt 2 / 2) * x - (m + Real.sqrt 2 / 2) * y ≤ 0

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x y, circle_C x y ∧ line x y a) →
  2 - Real.sqrt 2 ≤ a ∧ a ≤ 2 + Real.sqrt 2 :=
sorry

-- Theorem for the intersection of circles
theorem circles_intersect (m : ℝ) :
  (∀ x y, circle_C x y → inequality_condition x y m) →
  ∃ x y, circle_C x y ∧ circle_D x y m :=
sorry

end NUMINAMATH_CALUDE_range_of_a_circles_intersect_l3137_313778


namespace NUMINAMATH_CALUDE_difference_of_squares_l3137_313745

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3137_313745


namespace NUMINAMATH_CALUDE_sets_properties_l3137_313719

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Theorem statement
theorem sets_properties :
  (A ∩ B = {x | 2 < x ∧ x ≤ 3}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ 3 ∨ x ≥ 4}) ∧
  (∀ a : ℝ, B ∩ C a = C a → 2 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_sets_properties_l3137_313719


namespace NUMINAMATH_CALUDE_hula_hoop_difference_is_three_l3137_313794

/-- The difference in hula hooping times between Nancy and Casey -/
def hula_hoop_time_difference (nancy_time : ℕ) (morgan_time : ℕ) : ℕ :=
  let casey_time := morgan_time / 3
  nancy_time - casey_time

/-- Theorem stating that the difference in hula hooping times between Nancy and Casey is 3 minutes -/
theorem hula_hoop_difference_is_three :
  hula_hoop_time_difference 10 21 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hula_hoop_difference_is_three_l3137_313794


namespace NUMINAMATH_CALUDE_shooting_events_l3137_313779

-- Define the sample space
variable (Ω : Type)
variable [MeasurableSpace Ω]

-- Define the events
variable (both_hit : Set Ω)
variable (exactly_one_hit : Set Ω)
variable (both_miss : Set Ω)
variable (at_least_one_hit : Set Ω)

-- Define the probability measure
variable (P : Measure Ω)

-- Theorem statement
theorem shooting_events :
  (Disjoint exactly_one_hit both_hit) ∧
  (both_miss = at_least_one_hit.compl) := by
sorry

end NUMINAMATH_CALUDE_shooting_events_l3137_313779


namespace NUMINAMATH_CALUDE_chromium_percentage_calculation_l3137_313742

/-- Percentage of chromium in the first alloy -/
def chromium_percentage_1 : ℝ := 12

/-- Mass of the first alloy in kg -/
def mass_1 : ℝ := 15

/-- Mass of the second alloy in kg -/
def mass_2 : ℝ := 35

/-- Percentage of chromium in the new alloy -/
def chromium_percentage_new : ℝ := 10.6

/-- Percentage of chromium in the second alloy -/
def chromium_percentage_2 : ℝ := 10

theorem chromium_percentage_calculation :
  (chromium_percentage_1 / 100 * mass_1 + chromium_percentage_2 / 100 * mass_2) / (mass_1 + mass_2) * 100 = chromium_percentage_new :=
sorry

end NUMINAMATH_CALUDE_chromium_percentage_calculation_l3137_313742


namespace NUMINAMATH_CALUDE_max_triangle_area_l3137_313746

/-- The ellipse E defined by x²/13 + y²/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 13 + p.2^2 / 4 = 1}

/-- The left focus F₁ of the ellipse -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus F₂ of the ellipse -/
def F₂ : ℝ × ℝ := sorry

/-- A point P on the ellipse, not coinciding with left and right vertices -/
def P : ℝ × ℝ := sorry

/-- The area of triangle F₂PF₁ -/
def triangleArea (p : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating that the maximum area of triangle F₂PF₁ is 6 -/
theorem max_triangle_area :
  ∀ p ∈ Ellipse, p ≠ F₁ ∧ p ≠ F₂ → triangleArea p ≤ 6 ∧ ∃ q ∈ Ellipse, triangleArea q = 6 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l3137_313746


namespace NUMINAMATH_CALUDE_incorrect_expression_l3137_313734

theorem incorrect_expression (x y : ℚ) (h : x / y = 4 / 5) : 
  (x + 2 * y) / y = 14 / 5 ∧ 
  y / (2 * x - y) = 5 / 3 ∧ 
  (4 * x - y) / y = 11 / 5 ∧ 
  x / (3 * y) = 4 / 15 ∧ 
  (2 * x - y) / x ≠ 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l3137_313734


namespace NUMINAMATH_CALUDE_max_product_constraint_l3137_313744

theorem max_product_constraint (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 40) :
  x * y ≤ 400 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3137_313744


namespace NUMINAMATH_CALUDE_wrong_mark_value_l3137_313702

/-- Proves that the wrongly entered mark is 73 given the conditions of the problem -/
theorem wrong_mark_value (correct_mark : ℕ) (class_size : ℕ) (average_increase : ℚ) 
  (h1 : correct_mark = 63)
  (h2 : class_size = 20)
  (h3 : average_increase = 1/2) :
  ∃ x : ℕ, x = 73 ∧ (x : ℚ) - correct_mark = class_size * average_increase := by
  sorry


end NUMINAMATH_CALUDE_wrong_mark_value_l3137_313702


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l3137_313747

theorem cubic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x + y)) + (y^3 / (y + z)) + (z^3 / (z + x)) ≥ (x*y + y*z + z*x) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l3137_313747


namespace NUMINAMATH_CALUDE_cans_difference_l3137_313750

/-- The number of cans collected by Sarah yesterday -/
def sarah_yesterday : ℕ := 50

/-- The number of cans collected by Lara yesterday -/
def lara_yesterday : ℕ := sarah_yesterday + 30

/-- The number of cans collected by Alex yesterday -/
def alex_yesterday : ℕ := 90

/-- The number of cans collected by Sarah today -/
def sarah_today : ℕ := 40

/-- The number of cans collected by Lara today -/
def lara_today : ℕ := 70

/-- The number of cans collected by Alex today -/
def alex_today : ℕ := 55

/-- The total number of cans collected yesterday -/
def total_yesterday : ℕ := sarah_yesterday + lara_yesterday + alex_yesterday

/-- The total number of cans collected today -/
def total_today : ℕ := sarah_today + lara_today + alex_today

theorem cans_difference : total_yesterday - total_today = 55 := by
  sorry

end NUMINAMATH_CALUDE_cans_difference_l3137_313750


namespace NUMINAMATH_CALUDE_fraction_order_l3137_313784

theorem fraction_order : 
  let f1 : ℚ := -16/12
  let f2 : ℚ := -18/14
  let f3 : ℚ := -20/15
  f3 = f1 ∧ f1 < f2 := by sorry

end NUMINAMATH_CALUDE_fraction_order_l3137_313784


namespace NUMINAMATH_CALUDE_extreme_points_imply_a_l3137_313777

/-- Given a function f(x) = a * ln(x) + b * x^2 + x, if x = 1 and x = 2 are extreme points,
    then a = -2/3 --/
theorem extreme_points_imply_a (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * Real.log x + b * x^2 + x
  (∃ (c : ℝ), c ≠ 1 ∧ c ≠ 2 ∧ 
    (deriv f 1 = 0 ∧ deriv f 2 = 0) ∧
    (∀ x ∈ Set.Ioo 1 2, deriv f x ≠ 0)) →
  a = -2/3 := by
sorry

end NUMINAMATH_CALUDE_extreme_points_imply_a_l3137_313777


namespace NUMINAMATH_CALUDE_point_P_coordinates_l3137_313710

-- Define the coordinate system and points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, -1)

-- Define vectors
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- State the theorem
theorem point_P_coordinates :
  ∃ P : ℝ × ℝ, 
    vec A P = (2 : ℝ) • vec P B ∧ 
    P = (7/3, 1/3) := by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l3137_313710


namespace NUMINAMATH_CALUDE_min_value_expression_l3137_313773

theorem min_value_expression (a b c : ℝ) 
  (ha : -0.5 < a ∧ a < 0.5) 
  (hb : -0.5 < b ∧ b < 0.5) 
  (hc : -0.5 < c ∧ c < 0.5) : 
  1 / ((1 - a) * (1 - b) * (1 - c)) + 1 / ((1 + a) * (1 + b) * (1 + c)) ≥ 4.74 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3137_313773


namespace NUMINAMATH_CALUDE_proposition_counterexample_l3137_313772

theorem proposition_counterexample : 
  ∃ (α β : Real), 
    α > β ∧ 
    0 < α ∧ α < Real.pi / 2 ∧
    0 < β ∧ β < Real.pi / 2 ∧
    Real.tan α ≤ Real.tan β :=
by sorry

end NUMINAMATH_CALUDE_proposition_counterexample_l3137_313772


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l3137_313765

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧ 
  N = 23 ∧ 
  (1743 % N = 2019 % N) ∧ 
  (2019 % N = 3008 % N) ∧ 
  ∀ (M : ℕ), M > N → (1743 % M ≠ 2019 % M ∨ 2019 % M ≠ 3008 % M) := by
  sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l3137_313765


namespace NUMINAMATH_CALUDE_perpendicular_line_exists_l3137_313701

-- Define a line in a 2D plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in a 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for a point to not be on a line
def Point.notOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c ≠ 0

-- Define what it means for a line to be perpendicular to another line
def Line.perpendicularTo (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- The theorem statement
theorem perpendicular_line_exists (P : Point) (L : Line) (h : P.notOnLine L) :
  ∃ (M : Line), M.perpendicularTo L ∧ P.notOnLine M := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_exists_l3137_313701


namespace NUMINAMATH_CALUDE_valid_sequence_only_for_3_and_4_l3137_313754

/-- A sequence of positive integers satisfying the given recurrence relation -/
def ValidSequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ k, 2 ≤ k → k < n → a (k+1) = (a k ^ 2 + 1) / (a (k-1) + 1) - 1

/-- The theorem stating that only n = 3 and n = 4 satisfy the condition -/
theorem valid_sequence_only_for_3_and_4 :
  ∀ n : ℕ, n > 0 → (∃ a : ℕ → ℕ, ValidSequence a n) ↔ (n = 3 ∨ n = 4) :=
sorry

end NUMINAMATH_CALUDE_valid_sequence_only_for_3_and_4_l3137_313754


namespace NUMINAMATH_CALUDE_prob_two_even_balls_l3137_313721

/-- The probability of drawing two even-numbered balls without replacement from 16 balls numbered 1 to 16 is 7/30. -/
theorem prob_two_even_balls (n : ℕ) (h : n = 16) :
  (Nat.card {i : Fin n | i.val % 2 = 0} : ℚ) / n *
  ((Nat.card {i : Fin n | i.val % 2 = 0} - 1) : ℚ) / (n - 1) = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_even_balls_l3137_313721


namespace NUMINAMATH_CALUDE_nuts_problem_l3137_313724

/-- The number of nuts after one day's operation -/
def nuts_after_day (n : ℕ) : ℕ := 
  if 2 * n > 8 then 2 * n - 8 else 0

/-- The number of nuts after d days, starting with n nuts -/
def nuts_after_days (n : ℕ) (d : ℕ) : ℕ :=
  match d with
  | 0 => n
  | d + 1 => nuts_after_day (nuts_after_days n d)

theorem nuts_problem :
  nuts_after_days 7 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nuts_problem_l3137_313724


namespace NUMINAMATH_CALUDE_ratio_chain_l3137_313739

theorem ratio_chain (a b c d : ℚ) 
  (h1 : a / b = 1 / 4)
  (h2 : b / c = 13 / 9)
  (h3 : c / d = 5 / 13) :
  a / d = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ratio_chain_l3137_313739


namespace NUMINAMATH_CALUDE_road_trip_gas_usage_l3137_313735

/-- Calculates the total gallons of gas used on a road trip --/
theorem road_trip_gas_usage
  (highway_miles : ℝ)
  (highway_efficiency : ℝ)
  (city_miles : ℝ)
  (city_efficiency : ℝ)
  (h1 : highway_miles = 210)
  (h2 : highway_efficiency = 35)
  (h3 : city_miles = 54)
  (h4 : city_efficiency = 18) :
  highway_miles / highway_efficiency + city_miles / city_efficiency = 9 :=
by
  sorry

#check road_trip_gas_usage

end NUMINAMATH_CALUDE_road_trip_gas_usage_l3137_313735


namespace NUMINAMATH_CALUDE_train_length_l3137_313796

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 9 → speed_kmh * (1000 / 3600) * time_s = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3137_313796


namespace NUMINAMATH_CALUDE_rectangle_area_l3137_313705

/-- Rectangle ABCD with point E on AB and point F on AC -/
structure RectangleConfig where
  /-- Length of side AD -/
  a : ℝ
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- Point E -/
  E : ℝ × ℝ
  /-- Point F -/
  F : ℝ × ℝ
  /-- ABCD is a rectangle -/
  is_rectangle : A.1 = D.1 ∧ A.2 = B.2 ∧ B.1 = C.1 ∧ C.2 = D.2
  /-- AB = 2 × AD -/
  ab_twice_ad : B.1 - A.1 = 2 * a
  /-- E is the midpoint of AB -/
  e_midpoint : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  /-- F is on AC -/
  f_on_ac : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
  /-- F is on DE -/
  f_on_de : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (D.1 + s * (E.1 - D.1), D.2 + s * (E.2 - D.2))
  /-- Area of quadrilateral BFED is 50 -/
  area_bfed : abs ((B.1 - F.1) * (E.2 - D.2) - (B.2 - F.2) * (E.1 - D.1)) / 2 = 50

/-- The area of rectangle ABCD is 300 -/
theorem rectangle_area (config : RectangleConfig) : (config.B.1 - config.A.1) * (config.B.2 - config.D.2) = 300 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3137_313705


namespace NUMINAMATH_CALUDE_geometric_sequence_theorem_l3137_313756

def geometric_sequence (a₁ : ℚ) (q : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n + 1 => a₁ * q^n

def sum_sequence (a : ℕ → ℚ) : ℕ → ℚ
  | 0 => 0
  | n + 1 => sum_sequence a n + a (n + 1)

def b (S : ℕ → ℚ) (n : ℕ) : ℚ := S n + 1 / (S n)

def is_arithmetic_sequence (a b c : ℚ) : Prop := b - a = c - b

theorem geometric_sequence_theorem (a : ℕ → ℚ) (S : ℕ → ℚ) 
  (h1 : a 1 = 3/2)
  (h2 : ∀ n, S n = sum_sequence a n)
  (h3 : is_arithmetic_sequence (-2 * S 2) (S 3) (4 * S 4)) :
  (∃ q : ℚ, ∀ n, a n = geometric_sequence (3/2) q n) ∧
  (∃ l m : ℚ, ∀ n, l ≤ b S n ∧ b S n ≤ m ∧ m - l = 1/6) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_theorem_l3137_313756


namespace NUMINAMATH_CALUDE_sector_area_l3137_313715

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 10) (h2 : θ = 42) :
  (θ / 360) * π * r^2 = 35 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3137_313715


namespace NUMINAMATH_CALUDE_probability_joined_1890_to_1969_l3137_313759

def total_provinces : ℕ := 13
def joined_1890_to_1969 : ℕ := 4

theorem probability_joined_1890_to_1969 :
  (joined_1890_to_1969 : ℚ) / total_provinces = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_joined_1890_to_1969_l3137_313759


namespace NUMINAMATH_CALUDE_square_area_with_side_30_l3137_313755

theorem square_area_with_side_30 :
  let side : ℝ := 30
  let square_area := side * side
  square_area = 900 := by sorry

end NUMINAMATH_CALUDE_square_area_with_side_30_l3137_313755


namespace NUMINAMATH_CALUDE_complex_division_equivalence_l3137_313717

theorem complex_division_equivalence : Complex.I * (4 - 3 * Complex.I) = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equivalence_l3137_313717


namespace NUMINAMATH_CALUDE_largest_two_digit_multiple_of_17_l3137_313792

theorem largest_two_digit_multiple_of_17 : ∃ n : ℕ, n = 85 ∧ 
  (∀ m : ℕ, m ≤ 99 → m ≥ 10 → m % 17 = 0 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_multiple_of_17_l3137_313792


namespace NUMINAMATH_CALUDE_tank_b_height_is_six_l3137_313741

/-- The height of a cylindrical tank B, given another tank A and their properties --/
def heightOfTankB (heightA circumferenceA circumferenceB : ℝ) : ℝ :=
  let radiusA := circumferenceA / (2 * Real.pi)
  let radiusB := circumferenceB / (2 * Real.pi)
  let capacityA := Real.pi * radiusA^2 * heightA
  6

/-- Theorem stating that the height of tank B is 6 meters under given conditions --/
theorem tank_b_height_is_six (heightA circumferenceA circumferenceB : ℝ)
  (h_heightA : heightA = 10)
  (h_circumferenceA : circumferenceA = 6)
  (h_circumferenceB : circumferenceB = 10)
  (h_capacity_ratio : Real.pi * (circumferenceA / (2 * Real.pi))^2 * heightA = 
                      0.6 * (Real.pi * (circumferenceB / (2 * Real.pi))^2 * heightOfTankB heightA circumferenceA circumferenceB)) :
  heightOfTankB heightA circumferenceA circumferenceB = 6 := by
  sorry

#check tank_b_height_is_six

end NUMINAMATH_CALUDE_tank_b_height_is_six_l3137_313741


namespace NUMINAMATH_CALUDE_apple_tripling_theorem_l3137_313757

theorem apple_tripling_theorem (a b c : ℕ) :
  (3 * a + b + c = (17/10) * (a + b + c)) →
  (a + 3 * b + c = (3/2) * (a + b + c)) →
  (a + b + 3 * c = (9/5) * (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_apple_tripling_theorem_l3137_313757


namespace NUMINAMATH_CALUDE_rectangle_area_l3137_313793

/-- Given a rectangle with perimeter 176 inches and length 8 inches more than its width,
    prove that its area is 1920 square inches. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 4 * w + 16 = 176) : w * (w + 8) = 1920 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3137_313793


namespace NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_of_16385_l3137_313752

def n : ℕ := 16385

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := 
  sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := 
  sorry

-- Theorem statement
theorem sum_digits_greatest_prime_divisor_of_16385 : 
  sum_of_digits (greatest_prime_divisor n) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_of_16385_l3137_313752


namespace NUMINAMATH_CALUDE_candy_distribution_l3137_313711

theorem candy_distribution (total_candy : ℕ) (candy_per_student : ℕ) (num_students : ℕ) : 
  total_candy = 18 → candy_per_student = 2 → total_candy = candy_per_student * num_students → num_students = 9 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3137_313711


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l3137_313776

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x + 3

-- State the theorem
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (f a)) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l3137_313776


namespace NUMINAMATH_CALUDE_count_inequalities_l3137_313727

def is_inequality (e : String) : Bool :=
  match e with
  | "x - y" => false
  | "x ≤ y" => true
  | "x + y" => false
  | "x^2 - 3y" => false
  | "x ≥ 0" => true
  | "1/2x ≠ 3" => true
  | _ => false

def expressions : List String := [
  "x - y",
  "x ≤ y",
  "x + y",
  "x^2 - 3y",
  "x ≥ 0",
  "1/2x ≠ 3"
]

theorem count_inequalities :
  (expressions.filter is_inequality).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_inequalities_l3137_313727


namespace NUMINAMATH_CALUDE_divisibility_by_24_l3137_313704

theorem divisibility_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  24 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l3137_313704


namespace NUMINAMATH_CALUDE_jacket_cost_price_l3137_313783

theorem jacket_cost_price (original_price discount profit : ℝ) 
  (h1 : original_price = 500)
  (h2 : discount = 0.3)
  (h3 : profit = 50) :
  original_price * (1 - discount) = 300 + profit := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_price_l3137_313783


namespace NUMINAMATH_CALUDE_britney_lemon_tea_l3137_313770

/-- The number of people sharing the lemon tea -/
def number_of_people : ℕ := 5

/-- The number of cups each person gets -/
def cups_per_person : ℕ := 2

/-- The total number of cups of lemon tea Britney brewed -/
def total_cups : ℕ := number_of_people * cups_per_person

theorem britney_lemon_tea : total_cups = 10 := by
  sorry

end NUMINAMATH_CALUDE_britney_lemon_tea_l3137_313770


namespace NUMINAMATH_CALUDE_sugar_servings_calculation_l3137_313786

/-- Calculates the number of servings in a container given the total amount and serving size -/
def number_of_servings (total_amount : ℚ) (serving_size : ℚ) : ℚ :=
  total_amount / serving_size

/-- Proves that a container with 35 2/3 cups of sugar contains 23 7/9 servings when each serving is 1 1/2 cups -/
theorem sugar_servings_calculation :
  let total_sugar : ℚ := 35 + 2/3
  let serving_size : ℚ := 1 + 1/2
  number_of_servings total_sugar serving_size = 23 + 7/9 := by
  sorry

#eval number_of_servings (35 + 2/3) (1 + 1/2)

end NUMINAMATH_CALUDE_sugar_servings_calculation_l3137_313786


namespace NUMINAMATH_CALUDE_smallest_3digit_prime_factor_of_binom_300_150_l3137_313726

theorem smallest_3digit_prime_factor_of_binom_300_150 :
  let n := Nat.choose 300 150
  ∃ (p : Nat), Prime p ∧ 100 ≤ p ∧ p < 1000 ∧ p ∣ n ∧
    ∀ (q : Nat), Prime q → 100 ≤ q → q < p → ¬(q ∣ n) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_3digit_prime_factor_of_binom_300_150_l3137_313726


namespace NUMINAMATH_CALUDE_square_difference_l3137_313720

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) :
  (x - y)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l3137_313720


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l3137_313716

theorem fourth_root_equation_solution :
  ∀ x : ℝ, (x > 0 ∧ x^(1/4) = 16 / (8 - x^(1/4))) ↔ x = 256 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l3137_313716


namespace NUMINAMATH_CALUDE_cubic_polynomial_existence_l3137_313749

theorem cubic_polynomial_existence (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ f : ℤ → ℤ, 
    (∃ p q r : ℤ, ∀ x, f x = p * x^3 + q * x^2 + r * x + (a * b * c)) ∧ 
    (p > 0) ∧
    (f a = a^3) ∧ (f b = b^3) ∧ (f c = c^3) :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_existence_l3137_313749


namespace NUMINAMATH_CALUDE_bing_dwen_dwen_prices_l3137_313737

theorem bing_dwen_dwen_prices 
  (total_budget : ℝ) 
  (budget_A : ℝ) 
  (price_difference : ℝ) 
  (quantity_ratio : ℝ) :
  total_budget = 1700 →
  budget_A = 800 →
  price_difference = 25 →
  quantity_ratio = 3 →
  ∃ (price_B : ℝ) (price_A : ℝ),
    price_B = 15 ∧
    price_A = 40 ∧
    price_A = price_B + price_difference ∧
    (total_budget - budget_A) / price_B = quantity_ratio * (budget_A / price_A) := by
  sorry

#check bing_dwen_dwen_prices

end NUMINAMATH_CALUDE_bing_dwen_dwen_prices_l3137_313737


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3137_313797

def A : Set Int := {-2, -1, 0, 1, 2}
def B : Set Int := {x | 2 * x - 1 > 0}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3137_313797


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_singleton_two_l3137_313771

def A : Set ℝ := {-1, 2, 3}
def B : Set ℝ := {x : ℝ | x * (x - 3) < 0}

theorem A_intersect_B_eq_singleton_two : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_singleton_two_l3137_313771


namespace NUMINAMATH_CALUDE_keith_total_expenses_l3137_313751

def speakers_cost : ℚ := 136.01
def cd_player_cost : ℚ := 139.38
def tires_cost : ℚ := 112.46

theorem keith_total_expenses : 
  speakers_cost + cd_player_cost + tires_cost = 387.85 := by
  sorry

end NUMINAMATH_CALUDE_keith_total_expenses_l3137_313751


namespace NUMINAMATH_CALUDE_square_in_base_b_l3137_313709

/-- Represents a number in base b with digits d₂d₁d₀ --/
def base_b_number (b : ℕ) (d₂ d₁ d₀ : ℕ) : ℕ := d₂ * b^2 + d₁ * b + d₀

/-- The number 144 in base b --/
def number_144_b (b : ℕ) : ℕ := base_b_number b 1 4 4

theorem square_in_base_b (b : ℕ) (h : b > 4) :
  ∃ (n : ℕ), number_144_b b = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_in_base_b_l3137_313709


namespace NUMINAMATH_CALUDE_integer_between_sqrt2_and_sqrt8_l3137_313753

theorem integer_between_sqrt2_and_sqrt8 (a : ℤ) : Real.sqrt 2 < a ∧ a < Real.sqrt 8 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt2_and_sqrt8_l3137_313753


namespace NUMINAMATH_CALUDE_prime_triple_divisibility_l3137_313728

theorem prime_triple_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) →
  ((p = 2 ∧ q = 5 ∧ r = 3) ∨ 
   (p = 5 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 5)) :=
by sorry

end NUMINAMATH_CALUDE_prime_triple_divisibility_l3137_313728


namespace NUMINAMATH_CALUDE_sum_2011_is_29_l3137_313729

/-- Given a sequence of 2011 consecutive five-digit numbers, this function
    returns the sum of digits for the nth number in the sequence. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence starts with a five-digit number. -/
axiom start_five_digit : ∃ k : ℕ, 10000 ≤ k ∧ k < 100000 ∧ ∀ i, 1 ≤ i ∧ i ≤ 2011 → k + i - 1 < 100000

/-- The sum of digits of the 21st number is 37. -/
axiom sum_21 : sumOfDigits 21 = 37

/-- The sum of digits of the 54th number is 7. -/
axiom sum_54 : sumOfDigits 54 = 7

/-- The main theorem: the sum of digits of the 2011th number is 29. -/
theorem sum_2011_is_29 : sumOfDigits 2011 = 29 := sorry

end NUMINAMATH_CALUDE_sum_2011_is_29_l3137_313729


namespace NUMINAMATH_CALUDE_b_investment_is_7200_l3137_313736

/-- Represents the investment and profit distribution in a partnership business. -/
structure PartnershipBusiness where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- The investment of partner B in the business. -/
def b_investment (pb : PartnershipBusiness) : ℕ :=
  7200

/-- Theorem stating that B's investment is 7200, given the conditions of the problem. -/
theorem b_investment_is_7200 (pb : PartnershipBusiness) 
    (h1 : pb.a_investment = 2400)
    (h2 : pb.c_investment = 9600)
    (h3 : pb.total_profit = 9000)
    (h4 : pb.a_profit_share = 1125) :
  b_investment pb = 7200 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_is_7200_l3137_313736


namespace NUMINAMATH_CALUDE_cos_thirteen_pi_thirds_l3137_313703

theorem cos_thirteen_pi_thirds : Real.cos (13 * Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirteen_pi_thirds_l3137_313703


namespace NUMINAMATH_CALUDE_solve_system_for_x_l3137_313763

theorem solve_system_for_x :
  ∀ (x y : ℚ),
  (3 * x - 2 * y = 8) →
  (x + 3 * y = 7) →
  x = 38 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_x_l3137_313763


namespace NUMINAMATH_CALUDE_least_positive_integer_t_l3137_313700

theorem least_positive_integer_t : ∃ (t : ℕ+), 
  (∀ (x y : ℕ+), (x^2 + y^2)^2 + 2*t*x*(x^2 + y^2) = t^2*y^2 → t ≥ 25) ∧ 
  (∃ (x y : ℕ+), (x^2 + y^2)^2 + 2*25*x*(x^2 + y^2) = 25^2*y^2) := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_t_l3137_313700


namespace NUMINAMATH_CALUDE_keyboard_cost_l3137_313733

/-- Given the total cost of keyboards and printers, and the cost of a single printer,
    calculate the cost of a single keyboard. -/
theorem keyboard_cost (total_cost printer_cost : ℕ) : 
  total_cost = 2050 →
  printer_cost = 70 →
  ∃ (keyboard_cost : ℕ), 
    keyboard_cost * 15 + printer_cost * 25 = total_cost ∧ 
    keyboard_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_keyboard_cost_l3137_313733


namespace NUMINAMATH_CALUDE_inverse_power_inequality_l3137_313799

theorem inverse_power_inequality (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  (1 / a) ^ (1 / b) ≤ 1 := by sorry

end NUMINAMATH_CALUDE_inverse_power_inequality_l3137_313799


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3137_313766

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3137_313766


namespace NUMINAMATH_CALUDE_tourist_distribution_theorem_l3137_313780

/-- The number of ways to distribute tourists among guides --/
def distribute_tourists (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  num_guides ^ num_tourists

/-- The number of ways all tourists can choose the same guide --/
def all_same_guide (num_guides : ℕ) : ℕ := num_guides

/-- The number of valid distributions of tourists among guides --/
def valid_distributions (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  distribute_tourists num_tourists num_guides - all_same_guide num_guides

theorem tourist_distribution_theorem :
  valid_distributions 8 3 = 6558 := by
  sorry

end NUMINAMATH_CALUDE_tourist_distribution_theorem_l3137_313780


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l3137_313725

def total_cost : ℝ := 120

def first_bracket_percentage : ℝ := 0.4
def second_bracket_percentage : ℝ := 0.3
def tax_free_percentage : ℝ := 1 - first_bracket_percentage - second_bracket_percentage

def first_bracket_tax_rate : ℝ := 0.06
def second_bracket_tax_rate : ℝ := 0.08
def second_bracket_discount : ℝ := 0.05

def first_bracket_cost : ℝ := total_cost * first_bracket_percentage
def second_bracket_cost : ℝ := total_cost * second_bracket_percentage
def tax_free_cost : ℝ := total_cost * tax_free_percentage

theorem tax_free_items_cost :
  tax_free_cost = 36 := by sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l3137_313725


namespace NUMINAMATH_CALUDE_sasha_salt_adjustment_l3137_313761

theorem sasha_salt_adjustment (x y : ℝ) 
  (h1 : y > 0)  -- Yesterday's extra salt was positive
  (h2 : x > 0)  -- Initial salt amount is positive
  (h3 : x + y = 2*x + y/2)  -- Total salt needed is the same for both days
  : (3*x) / (2*x) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sasha_salt_adjustment_l3137_313761


namespace NUMINAMATH_CALUDE_set_forms_triangle_l3137_313762

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three given lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set (6, 8, 13) can form a triangle -/
theorem set_forms_triangle : can_form_triangle 6 8 13 := by
  sorry


end NUMINAMATH_CALUDE_set_forms_triangle_l3137_313762


namespace NUMINAMATH_CALUDE_A_intersect_B_l3137_313714

def A : Set ℝ := {-1, 0, 1}

def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem A_intersect_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l3137_313714


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_bound_l3137_313785

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  -- We don't need to specify the exact structure, just that it exists
  dummy : Unit

/-- The radius of the inscribed sphere of a regular quadrilateral pyramid -/
def inscribed_sphere_radius (p : RegularQuadrilateralPyramid) : ℝ :=
  sorry

/-- The radius of the circumscribed sphere of a regular quadrilateral pyramid -/
def circumscribed_sphere_radius (p : RegularQuadrilateralPyramid) : ℝ :=
  sorry

/-- The theorem stating that the ratio of the circumscribed sphere radius to the inscribed sphere radius
    is greater than or equal to 1 + √2 for any regular quadrilateral pyramid -/
theorem sphere_radius_ratio_bound (p : RegularQuadrilateralPyramid) :
  circumscribed_sphere_radius p / inscribed_sphere_radius p ≥ 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_bound_l3137_313785


namespace NUMINAMATH_CALUDE_tangent_x_intercept_difference_l3137_313760

/-- 
Given a point (x₀, y₀) on the curve y = e^x, if the tangent line at this point 
intersects the x-axis at (x₁, 0), then x₁ - x₀ = -1.
-/
theorem tangent_x_intercept_difference (x₀ : ℝ) : 
  let y₀ : ℝ := Real.exp x₀
  let f : ℝ → ℝ := λ x => Real.exp x
  let f' : ℝ → ℝ := λ x => Real.exp x
  let tangent_line : ℝ → ℝ := λ x => f' x₀ * (x - x₀) + y₀
  let x₁ : ℝ := x₀ - 1 / f' x₀
  (tangent_line x₁ = 0) → (x₁ - x₀ = -1) := by
  sorry

#check tangent_x_intercept_difference

end NUMINAMATH_CALUDE_tangent_x_intercept_difference_l3137_313760


namespace NUMINAMATH_CALUDE_ellipse_properties_l3137_313782

/-- Ellipse structure -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Focal distance of an ellipse -/
def focal_distance (e : Ellipse) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_properties (e : Ellipse) (l : Line) :
  l.point.1 = focal_distance e / 2 →  -- line passes through right focus
  l.slope = Real.tan (60 * π / 180) →  -- slope angle is 60°
  distance_point_to_line (-focal_distance e / 2, 0) l = 2 →  -- distance from left focus to line is 2
  focal_distance e = 4 ∧  -- focal distance is 4
  (e.b = 2 → e.a = 3 ∧ focal_distance e = 2 * Real.sqrt 5) :=  -- when b = 2, a = 3 and c = 2√5
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3137_313782


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3137_313789

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  (q > 0) →                     -- positive sequence
  (a 3 = 2) →                   -- given condition
  (a 4 = 8 * a 7) →             -- given condition
  (a 9 = 1 / 32) :=              -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3137_313789


namespace NUMINAMATH_CALUDE_composite_4p_plus_1_l3137_313781

theorem composite_4p_plus_1 (p : ℕ) (h1 : p ≥ 5) (h2 : Nat.Prime p) (h3 : Nat.Prime (2 * p + 1)) :
  ¬(Nat.Prime (4 * p + 1)) :=
sorry

end NUMINAMATH_CALUDE_composite_4p_plus_1_l3137_313781


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3137_313758

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

theorem intersection_complement_theorem : A ∩ (U \ B) = {x | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3137_313758


namespace NUMINAMATH_CALUDE_min_mushrooms_collected_l3137_313713

/-- Represents the number of mushrooms collected by Vasya and Masha over two days -/
structure MushroomCollection where
  vasya_day1 : ℕ
  vasya_day2 : ℕ

/-- Calculates the total number of mushrooms collected by both Vasya and Masha -/
def total_mushrooms (c : MushroomCollection) : ℚ :=
  (c.vasya_day1 + c.vasya_day2 : ℚ) + 
  ((3/4 : ℚ) * c.vasya_day1 + (6/5 : ℚ) * c.vasya_day2)

/-- Checks if the collection satisfies the given conditions -/
def is_valid_collection (c : MushroomCollection) : Prop :=
  (3/4 : ℚ) * c.vasya_day1 + (6/5 : ℚ) * c.vasya_day2 = 
  (11/10 : ℚ) * (c.vasya_day1 + c.vasya_day2)

/-- The main theorem stating the minimum number of mushrooms collected -/
theorem min_mushrooms_collected :
  ∃ (c : MushroomCollection), 
    is_valid_collection c ∧ 
    (∀ (c' : MushroomCollection), is_valid_collection c' → 
      total_mushrooms c ≤ total_mushrooms c') ∧
    ⌈total_mushrooms c⌉ = 19 := by
  sorry


end NUMINAMATH_CALUDE_min_mushrooms_collected_l3137_313713


namespace NUMINAMATH_CALUDE_unique_triplet_l3137_313731

theorem unique_triplet : ∃! (a b c : ℕ), 
  a > 1 ∧ b > 1 ∧ c > 1 ∧
  (b * c + 1) % a = 0 ∧
  (a * c + 1) % b = 0 ∧
  (a * b + 1) % c = 0 ∧
  a = 2 ∧ b = 3 ∧ c = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_triplet_l3137_313731


namespace NUMINAMATH_CALUDE_base_k_fraction_l3137_313769

/-- If k is a positive integer and 7/51 equals 0.23̅ₖ in base k, then k equals 16. -/
theorem base_k_fraction (k : ℕ) (h1 : k > 0) 
  (h2 : (7 : ℚ) / 51 = (2 * k + 3 : ℚ) / (k^2 - 1)) : k = 16 := by
  sorry

end NUMINAMATH_CALUDE_base_k_fraction_l3137_313769


namespace NUMINAMATH_CALUDE_binomial_probability_ge_two_l3137_313706

/-- A random variable following a Binomial distribution B(10, 1/2) -/
def ξ : ℕ → ℝ := sorry

/-- The probability mass function of ξ -/
def pmf (k : ℕ) : ℝ := sorry

/-- The cumulative distribution function of ξ -/
def cdf (k : ℕ) : ℝ := sorry

theorem binomial_probability_ge_two :
  (1 - cdf 1) = 1013 / 1024 :=
sorry

end NUMINAMATH_CALUDE_binomial_probability_ge_two_l3137_313706


namespace NUMINAMATH_CALUDE_certain_number_proof_l3137_313748

theorem certain_number_proof (N : ℝ) : 
  (2 / 5 : ℝ) * N - (3 / 5 : ℝ) * 125 = 45 → N = 300 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3137_313748


namespace NUMINAMATH_CALUDE_johns_paintball_expenditure_l3137_313795

/-- John's monthly paintball expenditure --/
theorem johns_paintball_expenditure :
  ∀ (plays_per_month boxes_per_play box_cost : ℕ),
  plays_per_month = 3 →
  boxes_per_play = 3 →
  box_cost = 25 →
  plays_per_month * boxes_per_play * box_cost = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_paintball_expenditure_l3137_313795


namespace NUMINAMATH_CALUDE_soda_cost_l3137_313707

/-- Represents the cost of items in cents -/
structure Cost where
  burger : ℚ
  soda : ℚ
  fries : ℚ

/-- The total cost of Keegan's purchase in cents -/
def keegan_total (c : Cost) : ℚ := 3 * c.burger + 2 * c.soda + c.fries

/-- The total cost of Alex's purchase in cents -/
def alex_total (c : Cost) : ℚ := 2 * c.burger + 3 * c.soda + c.fries

theorem soda_cost :
  ∃ (c : Cost),
    keegan_total c = 975 ∧
    alex_total c = 900 ∧
    c.soda = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l3137_313707


namespace NUMINAMATH_CALUDE_angle_AOB_measure_l3137_313723

/-- A configuration of rectangles with specific properties -/
structure RectangleConfiguration where
  /-- The number of equal rectangles -/
  num_rectangles : ℕ
  /-- Assertion that one side of each rectangle is twice the other -/
  side_ratio : Prop
  /-- Assertion that points C, O, and B are collinear -/
  collinear_COB : Prop
  /-- Assertion that triangle ACO is right-angled and isosceles -/
  triangle_ACO_properties : Prop

/-- Theorem stating that given the specific configuration, angle AOB measures 135° -/
theorem angle_AOB_measure (config : RectangleConfiguration) 
  (h1 : config.num_rectangles = 5)
  (h2 : config.side_ratio)
  (h3 : config.collinear_COB)
  (h4 : config.triangle_ACO_properties) :
  ∃ (angle_AOB : ℝ), angle_AOB = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_AOB_measure_l3137_313723


namespace NUMINAMATH_CALUDE_segment_length_segment_length_is_ten_l3137_313743

theorem segment_length : ℝ → Prop :=
  fun length => ∃ x₁ x₂ : ℝ,
    (|x₁ - Real.sqrt 25| = 5) ∧
    (|x₂ - Real.sqrt 25| = 5) ∧
    (x₁ ≠ x₂) ∧
    (length = |x₁ - x₂|) ∧
    (length = 10)

-- The proof goes here
theorem segment_length_is_ten : segment_length 10 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_segment_length_is_ten_l3137_313743


namespace NUMINAMATH_CALUDE_total_limbs_is_108_l3137_313787

/-- The total number of legs, arms, and tentacles of Daniel's animals -/
def total_limbs : ℕ :=
  let horses := 2
  let dogs := 5
  let cats := 7
  let turtles := 3
  let goats := 1
  let snakes := 4
  let spiders := 2
  let birds := 3
  let starfish_arms := 5
  let octopus_tentacles := 6
  let three_legged_dog := 1

  horses * 4 + 
  dogs * 4 + 
  cats * 4 + 
  turtles * 4 + 
  goats * 4 + 
  snakes * 0 + 
  spiders * 8 + 
  birds * 2 + 
  starfish_arms + 
  octopus_tentacles + 
  three_legged_dog * 3

theorem total_limbs_is_108 : total_limbs = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_limbs_is_108_l3137_313787


namespace NUMINAMATH_CALUDE_no_rational_solutions_for_quadratic_l3137_313738

theorem no_rational_solutions_for_quadratic (k : ℕ+) : 
  ¬∃ (x : ℚ), k * x^2 + 18 * x + 3 * k = 0 := by
sorry

end NUMINAMATH_CALUDE_no_rational_solutions_for_quadratic_l3137_313738


namespace NUMINAMATH_CALUDE_two_digit_addition_problem_l3137_313788

theorem two_digit_addition_problem (A B : ℕ) : 
  A ≠ B →
  A * 10 + 7 + 30 + B = 73 →
  A = 3 := by
sorry

end NUMINAMATH_CALUDE_two_digit_addition_problem_l3137_313788


namespace NUMINAMATH_CALUDE_total_spent_is_correct_l3137_313722

-- Define the value of a penny in dollars
def penny_value : ℚ := 1 / 100

-- Define the value of a dime in dollars
def dime_value : ℚ := 1 / 10

-- Define the number of pennies spent on ice cream
def ice_cream_pennies : ℕ := 2

-- Define the number of dimes spent on baseball cards
def baseball_cards_dimes : ℕ := 12

-- Theorem statement
theorem total_spent_is_correct :
  (ice_cream_pennies : ℚ) * penny_value + (baseball_cards_dimes : ℚ) * dime_value = 122 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_correct_l3137_313722


namespace NUMINAMATH_CALUDE_monday_profit_ratio_l3137_313740

def total_profit : ℚ := 1200
def wednesday_profit : ℚ := 500
def tuesday_profit : ℚ := (1 / 4) * total_profit

def monday_profit : ℚ := total_profit - tuesday_profit - wednesday_profit

theorem monday_profit_ratio : 
  monday_profit / total_profit = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_monday_profit_ratio_l3137_313740


namespace NUMINAMATH_CALUDE_planes_perpendicular_l3137_313774

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line_line : Line → Line → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : perp_line_plane m α) 
  (h4 : perp_line_plane n β) 
  (h5 : perp_line_line m n) : 
  perp_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l3137_313774


namespace NUMINAMATH_CALUDE_energy_drink_consumption_l3137_313768

/-- Represents the relationship between relaxation hours and energy drink consumption --/
def inverse_proportional (h g : ℝ) (k : ℝ) : Prop := h * g = k

theorem energy_drink_consumption 
  (h₁ h₂ g₁ g₂ : ℝ) 
  (h₁_pos : h₁ > 0) 
  (h₂_pos : h₂ > 0) 
  (g₁_pos : g₁ > 0) 
  (h₁_val : h₁ = 4) 
  (h₂_val : h₂ = 2) 
  (g₁_val : g₁ = 5) 
  (prop_const : ℝ) 
  (inv_prop₁ : inverse_proportional h₁ g₁ prop_const) 
  (inv_prop₂ : inverse_proportional h₂ g₂ prop_const) : 
  g₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_energy_drink_consumption_l3137_313768


namespace NUMINAMATH_CALUDE_new_member_age_l3137_313790

theorem new_member_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) : 
  n = 10 → initial_avg = 15 → new_avg = 17 → 
  ∃ (new_member_age : ℝ), 
    (n * initial_avg + new_member_age) / (n + 1) = new_avg ∧ 
    new_member_age = 37 := by
  sorry

end NUMINAMATH_CALUDE_new_member_age_l3137_313790


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3137_313791

theorem quadratic_inequality (x : ℝ) : x^2 - 36*x + 323 ≤ 3 ↔ 16 ≤ x ∧ x ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3137_313791


namespace NUMINAMATH_CALUDE_base_number_problem_l3137_313764

theorem base_number_problem (a : ℝ) : 
  (a > 0) → (a^14 - a^12 = 3 * a^12) → a = 2 := by sorry

end NUMINAMATH_CALUDE_base_number_problem_l3137_313764


namespace NUMINAMATH_CALUDE_fraction_division_l3137_313775

theorem fraction_division (a b : ℚ) (ha : a = 3) (hb : b = 4) :
  (1 / b) / (1 / a) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l3137_313775


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l3137_313730

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l3137_313730


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3137_313798

-- Define a function to determine if an angle is in the first quadrant
def is_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2

-- Define a function to determine if an angle is in the first or third quadrant
def is_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi < α ∧ α < k * Real.pi + Real.pi / 2

-- Theorem statement
theorem half_angle_quadrant (α : ℝ) :
  is_first_quadrant α → is_first_or_third_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3137_313798


namespace NUMINAMATH_CALUDE_shark_observation_l3137_313718

theorem shark_observation (p_truth : ℝ) (p_shark : ℝ) (n : ℕ) :
  p_truth = 1/6 →
  p_shark = 0.027777777777777773 →
  p_shark = p_truth * (1 / n) →
  n = 6 := by sorry

end NUMINAMATH_CALUDE_shark_observation_l3137_313718
