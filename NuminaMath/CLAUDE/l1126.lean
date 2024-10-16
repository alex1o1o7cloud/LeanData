import Mathlib

namespace NUMINAMATH_CALUDE_camper_difference_l1126_112675

/-- Represents the number of campers for each week -/
structure CamperCounts where
  threeWeeksAgo : ℕ
  twoWeeksAgo : ℕ
  lastWeek : ℕ

/-- The camping site scenario -/
def campingSite : CamperCounts → Prop
  | ⟨threeWeeksAgo, twoWeeksAgo, lastWeek⟩ =>
    threeWeeksAgo + twoWeeksAgo + lastWeek = 150 ∧
    twoWeeksAgo = 40 ∧
    lastWeek = 80 ∧
    threeWeeksAgo < twoWeeksAgo

theorem camper_difference (c : CamperCounts) (h : campingSite c) :
  c.twoWeeksAgo - c.threeWeeksAgo = 10 := by
  sorry

end NUMINAMATH_CALUDE_camper_difference_l1126_112675


namespace NUMINAMATH_CALUDE_marys_marbles_count_l1126_112678

def dans_marbles : ℕ := 5
def marys_marbles_multiplier : ℕ := 2

theorem marys_marbles_count : dans_marbles * marys_marbles_multiplier = 10 := by
  sorry

end NUMINAMATH_CALUDE_marys_marbles_count_l1126_112678


namespace NUMINAMATH_CALUDE_martins_trip_distance_l1126_112657

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that Martin's trip distance is 72.0 miles -/
theorem martins_trip_distance :
  let speed : ℝ := 12.0
  let time : ℝ := 6.0
  distance speed time = 72.0 := by
sorry

end NUMINAMATH_CALUDE_martins_trip_distance_l1126_112657


namespace NUMINAMATH_CALUDE_largest_prefix_for_two_digit_quotient_l1126_112660

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem largest_prefix_for_two_digit_quotient :
  ∀ n : ℕ, n ≤ 9 →
    (is_two_digit ((n * 100 + 72) / 6) ↔ n ≤ 5) ∧
    (∀ m : ℕ, m ≤ 9 ∧ m > 5 → ¬(is_two_digit ((m * 100 + 72) / 6))) :=
by sorry

end NUMINAMATH_CALUDE_largest_prefix_for_two_digit_quotient_l1126_112660


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1126_112616

theorem product_of_three_numbers (a b c : ℕ) : 
  (a * b * c = 224) ∧ 
  (a < b) ∧ (b < c) ∧ 
  (a * 2 = c) ∧
  (∀ x y z : ℕ, x * y * z = 224 ∧ x < y ∧ y < z ∧ x * 2 = z → x = a ∧ y = b ∧ z = c) :=
by sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1126_112616


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1126_112665

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1126_112665


namespace NUMINAMATH_CALUDE_four_digit_reverse_pairs_l1126_112630

def reverse_digits (n : ℕ) : ℕ :=
  let digits := String.toList (toString n)
  String.toNat! (String.mk (List.reverse digits))

def ends_with_three_zeros (n : ℕ) : Prop :=
  n % 1000 = 0

theorem four_digit_reverse_pairs : 
  ∀ (a b : ℕ), 
    1000 ≤ a ∧ a < 10000 ∧
    1000 ≤ b ∧ b < 10000 ∧
    b = reverse_digits a ∧
    ends_with_three_zeros (a * b) →
    ((a = 5216 ∧ b = 6125) ∨
     (a = 5736 ∧ b = 6375) ∨
     (a = 5264 ∧ b = 4625) ∨
     (a = 5784 ∧ b = 4875))
  := by sorry

end NUMINAMATH_CALUDE_four_digit_reverse_pairs_l1126_112630


namespace NUMINAMATH_CALUDE_total_production_l1126_112620

/-- The daily production of fertilizer in tons -/
def daily_production : ℕ := 105

/-- The number of days of production -/
def days : ℕ := 24

/-- Theorem stating the total production over the given number of days -/
theorem total_production : daily_production * days = 2520 := by
  sorry

end NUMINAMATH_CALUDE_total_production_l1126_112620


namespace NUMINAMATH_CALUDE_optimal_partition_l1126_112632

def minimizeAbsoluteErrors (diameters : List ℝ) : 
  ℝ × ℝ × ℝ := sorry

theorem optimal_partition (diameters : List ℝ) 
  (h1 : diameters.length = 120) 
  (h2 : List.Sorted (· ≤ ·) diameters) :
  let (d, a, b) := minimizeAbsoluteErrors diameters
  d = (diameters.get! 59 + diameters.get! 60) / 2 ∧
  a = (diameters.take 60).sum / 60 ∧
  b = (diameters.drop 60).sum / 60 :=
sorry

end NUMINAMATH_CALUDE_optimal_partition_l1126_112632


namespace NUMINAMATH_CALUDE_jerry_age_l1126_112666

/-- Given that Mickey's age is 5 years less than 200% of Jerry's age and Mickey is 19 years old,
    prove that Jerry is 12 years old. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 2 * jerry_age - 5)
  (h2 : mickey_age = 19) : 
  jerry_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_jerry_age_l1126_112666


namespace NUMINAMATH_CALUDE_curve_family_condition_l1126_112655

/-- A family of curves parameterized by p -/
def curve_family (p x y : ℝ) : Prop :=
  y = p^2 + (2*p - 1)*x + 2*x^2

/-- The condition for a point (x, y) to have at least one curve passing through it -/
def has_curve_passing_through (x y : ℝ) : Prop :=
  ∃ p : ℝ, curve_family p x y

/-- The theorem stating the equivalence between the existence of a curve passing through (x, y) 
    and the inequality y ≥ x² - x -/
theorem curve_family_condition (x y : ℝ) : 
  has_curve_passing_through x y ↔ y ≥ x^2 - x :=
sorry

end NUMINAMATH_CALUDE_curve_family_condition_l1126_112655


namespace NUMINAMATH_CALUDE_apple_distribution_l1126_112681

theorem apple_distribution (total_apples : ℕ) (given_to_father : ℕ) (num_friends : ℕ) :
  total_apples = 55 →
  given_to_father = 10 →
  num_friends = 4 →
  (total_apples - given_to_father) % (num_friends + 1) = 0 →
  (total_apples - given_to_father) / (num_friends + 1) = 9 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l1126_112681


namespace NUMINAMATH_CALUDE_circle_condition_l1126_112669

/-- A circle in the xy-plane is represented by the equation x^2 + y^2 + Dx + Ey + F = 0,
    where D^2 + E^2 - 4F > 0 -/
def is_circle (D E F : ℝ) : Prop := D^2 + E^2 - 4*F > 0

/-- The equation x^2 + y^2 - 2x + 2k + 3 = 0 represents a circle -/
def our_equation_is_circle (k : ℝ) : Prop := is_circle (-2) 0 (2*k + 3)

theorem circle_condition (k : ℝ) : our_equation_is_circle k ↔ k < -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l1126_112669


namespace NUMINAMATH_CALUDE_statement_a_is_false_statement_b_is_true_statement_c_is_true_statement_d_is_true_statement_e_is_true_l1126_112608

-- Statement A
theorem statement_a_is_false : ∃ (a b c : ℝ), a > b ∧ c < 0 ∧ a * c ≤ b * c :=
  sorry

-- Statement B
theorem statement_b_is_true : ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y → 
  (2 * x * y) / (x + y) < Real.sqrt (x * y) :=
  sorry

-- Statement C
theorem statement_c_is_true : ∀ (s : ℝ), s > 0 → 
  ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = s → 
  x * y ≤ (s / 2) * (s / 2) :=
  sorry

-- Statement D
theorem statement_d_is_true : ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y → 
  (x^2 + y^2) / 2 > ((x + y) / 2)^2 :=
  sorry

-- Statement E
theorem statement_e_is_true : ∀ (p : ℝ), p > 0 → 
  ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = p → 
  x + y ≥ 2 * Real.sqrt p :=
  sorry

end NUMINAMATH_CALUDE_statement_a_is_false_statement_b_is_true_statement_c_is_true_statement_d_is_true_statement_e_is_true_l1126_112608


namespace NUMINAMATH_CALUDE_signup_ways_eq_81_l1126_112618

/-- The number of interest groups available --/
def num_groups : ℕ := 3

/-- The number of students signing up --/
def num_students : ℕ := 4

/-- The number of ways students can sign up for interest groups --/
def signup_ways : ℕ := num_groups ^ num_students

/-- Theorem: The number of ways four students can sign up for one of three interest groups is 81 --/
theorem signup_ways_eq_81 : signup_ways = 81 := by
  sorry

end NUMINAMATH_CALUDE_signup_ways_eq_81_l1126_112618


namespace NUMINAMATH_CALUDE_folded_paper_area_ratio_l1126_112663

/-- Represents a rectangular piece of paper with specific folding properties -/
structure FoldedPaper where
  width : ℝ
  length : ℝ
  area : ℝ
  foldedArea : ℝ
  lengthIsDoubleWidth : length = 2 * width
  areaIsLengthTimesWidth : area = length * width
  foldedAreaCalculation : foldedArea = area - 2 * (width * width / 4)

/-- Theorem stating that the ratio of folded area to original area is 1/2 -/
theorem folded_paper_area_ratio 
    (paper : FoldedPaper) : paper.foldedArea / paper.area = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_area_ratio_l1126_112663


namespace NUMINAMATH_CALUDE_jenny_reading_time_l1126_112611

/-- Calculates the average daily reading time including breaks -/
def averageDailyReadingTime (numBooks : ℕ) (totalDays : ℕ) (readingSpeed : ℕ) 
  (breakDuration : ℕ) (breakInterval : ℕ) (bookWords : List ℕ) : ℕ :=
  let totalWords := bookWords.sum
  let readingMinutes := totalWords / readingSpeed
  let readingHours := readingMinutes / 60
  let numBreaks := readingHours
  let breakMinutes := numBreaks * breakDuration
  let totalMinutes := readingMinutes + breakMinutes
  totalMinutes / totalDays

/-- Theorem: Jenny's average daily reading time is 124 minutes -/
theorem jenny_reading_time :
  let numBooks := 5
  let totalDays := 15
  let readingSpeed := 60  -- words per minute
  let breakDuration := 15  -- minutes
  let breakInterval := 60  -- minutes
  let bookWords := [12000, 18000, 24000, 15000, 21000]
  averageDailyReadingTime numBooks totalDays readingSpeed breakDuration breakInterval bookWords = 124 := by
  sorry

end NUMINAMATH_CALUDE_jenny_reading_time_l1126_112611


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1126_112696

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1126_112696


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1126_112672

theorem arithmetic_mean_problem (a b c d : ℝ) :
  (a + b + c + d + 105) / 5 = 90 →
  (a + b + c + d) / 4 = 86.25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1126_112672


namespace NUMINAMATH_CALUDE_divisors_of_420_l1126_112653

/-- The sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of divisors function -/
def num_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the sum and number of divisors for 420 -/
theorem divisors_of_420 : 
  sum_of_divisors 420 = 1344 ∧ num_of_divisors 420 = 24 := by sorry

end NUMINAMATH_CALUDE_divisors_of_420_l1126_112653


namespace NUMINAMATH_CALUDE_tom_trade_amount_l1126_112625

/-- The amount Tom initially gave to trade his Super Nintendo for an NES -/
def trade_amount (super_nintendo_value : ℝ) (credit_percentage : ℝ) 
  (nes_price : ℝ) (game_value : ℝ) (change : ℝ) : ℝ :=
  let credit := super_nintendo_value * credit_percentage
  let remaining := nes_price - credit
  let total_needed := remaining + game_value
  total_needed + change

/-- Theorem stating the amount Tom initially gave -/
theorem tom_trade_amount : 
  trade_amount 150 0.8 160 30 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_tom_trade_amount_l1126_112625


namespace NUMINAMATH_CALUDE_even_increasing_neg_implies_l1126_112676

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- A function f: ℝ → ℝ is increasing on (-∞, 0) if
    for all x, y ∈ (-∞, 0), x < y implies f(x) < f(y) -/
def IncreasingOnNegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → x < 0 → y ≤ 0 → f x < f y

theorem even_increasing_neg_implies (f : ℝ → ℝ)
    (h_even : IsEven f) (h_inc : IncreasingOnNegatives f) :
    f (-1) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_neg_implies_l1126_112676


namespace NUMINAMATH_CALUDE_remainder_sum_l1126_112661

theorem remainder_sum (x : ℤ) (h : x % 6 = 3) : 
  (x^2 % 30) + (x^3 % 11) = 14 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l1126_112661


namespace NUMINAMATH_CALUDE_least_number_of_radios_l1126_112619

/-- Represents the problem of finding the least number of radios bought by a dealer. -/
theorem least_number_of_radios (n d : ℕ) (h_d_pos : d > 0) : 
  (∃ (d : ℕ), d > 0 ∧ 
    10 * n - 30 - (3 * d) / (2 * n) = 80 ∧ 
    ∀ m : ℕ, m < n → ¬(∃ (d' : ℕ), d' > 0 ∧ 10 * m - 30 - (3 * d') / (2 * m) = 80)) →
  n = 11 :=
sorry

end NUMINAMATH_CALUDE_least_number_of_radios_l1126_112619


namespace NUMINAMATH_CALUDE_parabola_vertex_l1126_112699

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 3)^2 + 4

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, 4)

/-- Theorem: The vertex of the parabola y = -2(x-3)^2 + 4 is (3, 4) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1126_112699


namespace NUMINAMATH_CALUDE_fraction_value_l1126_112674

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1126_112674


namespace NUMINAMATH_CALUDE_parabola_equation_from_dot_product_parabola_equation_is_x_squared_eq_8y_l1126_112606

/-- A parabola with vertex at the origin and focus on the positive y-axis -/
structure UpwardParabola where
  focus : ℝ
  focus_positive : 0 < focus

/-- The equation of an upward parabola given its focus -/
def parabola_equation (p : UpwardParabola) (x y : ℝ) : Prop :=
  x^2 = 2 * p.focus * y

/-- A line passing through two points on a parabola -/
structure IntersectingLine (p : UpwardParabola) where
  slope : ℝ
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_equation p x₁ y₁ ∧
    parabola_equation p x₂ y₂ ∧
    y₁ = slope * x₁ + p.focus ∧
    y₂ = slope * x₂ + p.focus

/-- The main theorem -/
theorem parabola_equation_from_dot_product
  (p : UpwardParabola)
  (l : IntersectingLine p)
  (h : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_equation p x₁ y₁ ∧
    parabola_equation p x₂ y₂ ∧
    y₁ = l.slope * x₁ + p.focus ∧
    y₂ = l.slope * x₂ + p.focus ∧
    x₁ * x₂ + y₁ * y₂ = -12) :
  p.focus = 4 :=
sorry

/-- The equation of the parabola is x² = 8y -/
theorem parabola_equation_is_x_squared_eq_8y
  (p : UpwardParabola)
  (h : p.focus = 4) :
  ∀ x y, parabola_equation p x y ↔ x^2 = 8*y :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_from_dot_product_parabola_equation_is_x_squared_eq_8y_l1126_112606


namespace NUMINAMATH_CALUDE_omega_squared_plus_omega_plus_one_eq_zero_l1126_112633

theorem omega_squared_plus_omega_plus_one_eq_zero :
  let ω : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I
  ω^2 + ω + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_omega_squared_plus_omega_plus_one_eq_zero_l1126_112633


namespace NUMINAMATH_CALUDE_pentagon_triangle_angle_sum_l1126_112683

/-- The measure of an interior angle of a regular pentagon in degrees -/
def regular_pentagon_angle : ℝ := 108

/-- The measure of an interior angle of a regular triangle in degrees -/
def regular_triangle_angle : ℝ := 60

/-- Theorem: The sum of angles formed by two adjacent sides of a regular pentagon 
    and one side of a regular triangle that share a vertex is 168 degrees -/
theorem pentagon_triangle_angle_sum : 
  regular_pentagon_angle + regular_triangle_angle = 168 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_triangle_angle_sum_l1126_112683


namespace NUMINAMATH_CALUDE_factorization_equality_l1126_112644

theorem factorization_equality (x : ℝ) : x * (x - 3) + (3 - x) = (x - 3) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1126_112644


namespace NUMINAMATH_CALUDE_sue_falls_count_l1126_112600

structure Friend where
  name : String
  age : Nat
  falls : Nat

def steven : Friend := { name := "Steven", age := 20, falls := 3 }
def stephanie : Friend := { name := "Stephanie", age := 24, falls := steven.falls + 13 }
def sam : Friend := { name := "Sam", age := 24, falls := 1 }
def sue : Friend := { name := "Sue", age := 26, falls := 0 }  -- falls will be calculated

def sonya_falls : Nat := stephanie.falls / 2 - 2
def sophie_falls : Nat := sam.falls + 4

def youngest_age : Nat := min steven.age (min stephanie.age (min sam.age sue.age))

theorem sue_falls_count :
  sue.falls = sue.age - youngest_age :=
by sorry

end NUMINAMATH_CALUDE_sue_falls_count_l1126_112600


namespace NUMINAMATH_CALUDE_building_heights_sum_l1126_112682

/-- Proves that the total height of three buildings is 340 feet -/
theorem building_heights_sum (middle_height : ℝ) (left_height : ℝ) (right_height : ℝ)
  (h1 : middle_height = 100)
  (h2 : left_height = 0.8 * middle_height)
  (h3 : right_height = left_height + middle_height - 20) :
  left_height + middle_height + right_height = 340 := by
  sorry

end NUMINAMATH_CALUDE_building_heights_sum_l1126_112682


namespace NUMINAMATH_CALUDE_container_capacity_l1126_112623

theorem container_capacity : ∀ (C : ℝ), 
  (0.3 * C + 18 = 0.75 * C) → C = 40 :=
fun C h => by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l1126_112623


namespace NUMINAMATH_CALUDE_hyperbola_k_squared_l1126_112685

/-- A hyperbola centered at the origin, opening vertically -/
structure VerticalHyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point (x, y) lies on the hyperbola -/
def VerticalHyperbola.contains (h : VerticalHyperbola) (x y : ℝ) : Prop :=
  y^2 / h.a^2 - x^2 / h.b^2 = 1

/-- The main theorem -/
theorem hyperbola_k_squared (h : VerticalHyperbola) 
  (h1 : h.contains 4 3)
  (h2 : h.contains 0 2)
  (h3 : h.contains 2 k) : k^2 = 17/4 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_k_squared_l1126_112685


namespace NUMINAMATH_CALUDE_function_nonpositive_implies_a_geq_3_l1126_112648

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 3 - a

-- State the theorem
theorem function_nonpositive_implies_a_geq_3 :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ 0) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_nonpositive_implies_a_geq_3_l1126_112648


namespace NUMINAMATH_CALUDE_f_value_at_2_l1126_112626

def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l1126_112626


namespace NUMINAMATH_CALUDE_inverse_f_at_negative_eight_l1126_112643

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - Real.log x / Real.log 3

theorem inverse_f_at_negative_eight (a : ℝ) :
  f a 1 = 1 → f a (3^9) = -8 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_negative_eight_l1126_112643


namespace NUMINAMATH_CALUDE_tan_two_alpha_plus_pi_fourth_l1126_112698

theorem tan_two_alpha_plus_pi_fourth (α : Real) 
  (h : (2 * (Real.cos α) ^ 2 + Real.cos (π / 2 + 2 * α) - 1) / 
       (Real.sqrt 2 * Real.sin (2 * α + π / 4)) = 4) : 
  Real.tan (2 * α + π / 4) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_tan_two_alpha_plus_pi_fourth_l1126_112698


namespace NUMINAMATH_CALUDE_part_one_part_two_l1126_112693

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Part 1
theorem part_one : 
  A 2 ∩ (Set.univ \ B 2) = {x | 2 < x ∧ x ≤ 4 ∨ 5 ≤ x ∧ x < 7} :=
by sorry

-- Part 2
theorem part_two :
  {a : ℝ | a ≠ 1 ∧ A a ∪ B a = A a} = {a | 1 < a ∧ a ≤ 3 ∨ a = -1} :=
by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1126_112693


namespace NUMINAMATH_CALUDE_x_value_l1126_112622

theorem x_value : ∃ x : ℚ, (3 * x) / 7 = 21 ∧ x = 49 := by sorry

end NUMINAMATH_CALUDE_x_value_l1126_112622


namespace NUMINAMATH_CALUDE_triangle_area_l1126_112607

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  b = 7 →
  c = 5 →
  B = 2 * π / 3 →
  (1/2) * a * c * Real.sin B = (15 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1126_112607


namespace NUMINAMATH_CALUDE_circle_radius_problem_l1126_112603

-- Define the circle and points
def Circle : Type := ℝ → Prop
def Point : Type := ℝ × ℝ

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define the radius of the circle
def radius (c : Circle) : ℝ := sorry

-- Define the center of the circle
def center (c : Circle) : Point := sorry

-- Define a point on the circle
def on_circle (p : Point) (c : Circle) : Prop := sorry

-- Define a secant
def is_secant (p q r : Point) (c : Circle) : Prop := 
  ¬(on_circle p c) ∧ on_circle q c ∧ on_circle r c

-- Theorem statement
theorem circle_radius_problem (c : Circle) (p q r : Point) :
  distance p (center c) = 17 →
  is_secant p q r c →
  distance p q = 11 →
  distance q r = 8 →
  radius c = 4 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l1126_112603


namespace NUMINAMATH_CALUDE_lisa_process_ends_at_39_l1126_112615

/-- The function that represents one step of Lisa's process -/
def f (x : ℕ) : ℕ :=
  (x / 10) + 4 * (x % 10)

/-- The sequence of numbers generated by Lisa's process -/
def lisa_sequence (x : ℕ) : ℕ → ℕ
  | 0 => x
  | n + 1 => f (lisa_sequence x n)

/-- The theorem stating that Lisa's process always ends at 39 when starting with 53^2022 - 1 -/
theorem lisa_process_ends_at_39 :
  ∃ n : ℕ, ∀ m : ℕ, m ≥ n → lisa_sequence (53^2022 - 1) m = 39 :=
sorry

end NUMINAMATH_CALUDE_lisa_process_ends_at_39_l1126_112615


namespace NUMINAMATH_CALUDE_b_grazing_months_l1126_112605

/-- Represents the number of months B put his oxen for grazing -/
def b_months : ℕ := sorry

/-- Total rent of the pasture in Rs. -/
def total_rent : ℕ := 280

/-- C's share of the rent in Rs. -/
def c_share : ℕ := 72

/-- Calculates the total oxen-months for all farmers -/
def total_oxen_months : ℕ := 10 * 7 + 12 * b_months + 15 * 3

/-- Theorem stating that B put his oxen for grazing for 5 months -/
theorem b_grazing_months : b_months = 5 := by
  sorry

end NUMINAMATH_CALUDE_b_grazing_months_l1126_112605


namespace NUMINAMATH_CALUDE_banknote_problem_l1126_112673

theorem banknote_problem :
  ∀ (x y z : ℕ),
    x + y + z = 24 →
    10 * x + 20 * y + 50 * z = 1000 →
    x ≥ 1 →
    y ≥ 1 →
    z ≥ 1 →
    y = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_banknote_problem_l1126_112673


namespace NUMINAMATH_CALUDE_tank_depth_is_six_l1126_112688

/-- Represents a rectangular tank with given length, width, and depth. -/
structure Tank where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the total surface area of the tank to be plastered. -/
def Tank.surfaceArea (t : Tank) : ℝ :=
  2 * (t.length * t.depth + t.width * t.depth) + t.length * t.width

/-- The cost of plastering per square meter in rupees. -/
def plasteringCostPerSqm : ℝ := 0.75

/-- The total cost of plastering the tank in rupees. -/
def totalPlasteringCost (t : Tank) : ℝ :=
  plasteringCostPerSqm * t.surfaceArea

/-- Theorem stating that a tank with given dimensions and plastering cost has a depth of 6 meters. -/
theorem tank_depth_is_six (t : Tank) (h1 : t.length = 25) (h2 : t.width = 12) 
    (h3 : totalPlasteringCost t = 558) : t.depth = 6 := by
  sorry

#check tank_depth_is_six

end NUMINAMATH_CALUDE_tank_depth_is_six_l1126_112688


namespace NUMINAMATH_CALUDE_sons_shoveling_time_l1126_112654

/-- Proves that Wayne's son takes 21 hours to shovel the entire driveway alone,
    given that Wayne and his son together take 3 hours,
    and Wayne shovels 6 times as fast as his son. -/
theorem sons_shoveling_time (total_work : ℝ) (joint_time : ℝ) (wayne_speed_ratio : ℝ) :
  total_work > 0 →
  joint_time = 3 →
  wayne_speed_ratio = 6 →
  (total_work / joint_time) * (wayne_speed_ratio + 1) * 21 = total_work :=
by sorry

end NUMINAMATH_CALUDE_sons_shoveling_time_l1126_112654


namespace NUMINAMATH_CALUDE_jose_to_john_ratio_l1126_112642

-- Define the total amount and the ratios
def total_amount : ℕ := 4800
def ratio_john : ℕ := 2
def ratio_jose : ℕ := 4
def ratio_binoy : ℕ := 6

-- Define John's share
def john_share : ℕ := 1600

-- Theorem to prove
theorem jose_to_john_ratio :
  let total_ratio := ratio_john + ratio_jose + ratio_binoy
  let share_value := total_amount / total_ratio
  let jose_share := share_value * ratio_jose
  jose_share / john_share = 2 := by
  sorry

end NUMINAMATH_CALUDE_jose_to_john_ratio_l1126_112642


namespace NUMINAMATH_CALUDE_polygon_diagonals_l1126_112659

theorem polygon_diagonals (n : ℕ) (h1 : n * 10 = 360) : n * (n - 3) / 2 = 594 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l1126_112659


namespace NUMINAMATH_CALUDE_negative_510_in_third_quadrant_l1126_112671

-- Define a function to normalize an angle to the range [-360°, 0°)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360 - 360

-- Define a function to determine the quadrant of an angle
def quadrant (angle : Int) : Nat :=
  let normalizedAngle := normalizeAngle angle
  if -90 < normalizedAngle && normalizedAngle ≤ 0 then 4
  else if -180 < normalizedAngle && normalizedAngle ≤ -90 then 3
  else if -270 < normalizedAngle && normalizedAngle ≤ -180 then 2
  else 1

-- Theorem: -510° is in the third quadrant
theorem negative_510_in_third_quadrant : quadrant (-510) = 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_510_in_third_quadrant_l1126_112671


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l1126_112677

/-- Represents the color of a frog -/
inductive FrogColor
  | Green
  | Red
  | Blue

/-- Represents a row of frogs -/
def FrogRow := List FrogColor

/-- Checks if a frog arrangement is valid -/
def is_valid_arrangement (row : FrogRow) : Bool :=
  sorry

/-- Counts the number of frogs of each color in a row -/
def count_frogs (row : FrogRow) : Nat × Nat × Nat :=
  sorry

/-- Generates all possible arrangements of frogs -/
def all_arrangements : List FrogRow :=
  sorry

/-- Counts the number of valid arrangements -/
def count_valid_arrangements : Nat :=
  sorry

theorem frog_arrangement_count :
  count_valid_arrangements = 96 :=
sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l1126_112677


namespace NUMINAMATH_CALUDE_collision_count_l1126_112691

/-- The number of collisions between two groups of balls moving in opposite directions in a trough with a wall -/
def totalCollisions (n m : ℕ) : ℕ :=
  n * m + n * (n - 1) / 2

/-- Theorem stating that the total number of collisions for 20 balls moving towards a wall
    and 16 balls moving in the opposite direction is 510 -/
theorem collision_count : totalCollisions 20 16 = 510 := by
  sorry

end NUMINAMATH_CALUDE_collision_count_l1126_112691


namespace NUMINAMATH_CALUDE_differential_savings_example_l1126_112689

/-- Calculates the differential savings when tax rate is reduced -/
def differential_savings (income : ℝ) (old_rate : ℝ) (new_rate : ℝ) : ℝ :=
  income * old_rate - income * new_rate

/-- Proves that the differential savings for a specific case is correct -/
theorem differential_savings_example : 
  differential_savings 34500 0.42 0.28 = 4830 := by
  sorry

end NUMINAMATH_CALUDE_differential_savings_example_l1126_112689


namespace NUMINAMATH_CALUDE_cube_root_of_neg_125_l1126_112638

theorem cube_root_of_neg_125 : ∃ x : ℝ, x^3 = -125 ∧ x = -5 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_neg_125_l1126_112638


namespace NUMINAMATH_CALUDE_equation_solution_l1126_112695

theorem equation_solution (a : ℝ) : 
  (∀ x, 2 * x + a = 3 ↔ x = -1) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1126_112695


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1126_112692

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 4)
  parallel a b → x = 2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1126_112692


namespace NUMINAMATH_CALUDE_truck_filling_time_truck_filling_time_proof_l1126_112609

/-- The time taken to fill a truck with stone blocks given specific worker rates and capacity -/
theorem truck_filling_time : ℕ :=
  let truck_capacity : ℕ := 6000
  let stella_initial_rate : ℕ := 250
  let twinkle_initial_rate : ℕ := 200
  let stella_changed_rate : ℕ := 220
  let twinkle_changed_rate : ℕ := 230
  let additional_workers_count : ℕ := 6
  let additional_workers_initial_rate1 : ℕ := 300
  let additional_workers_initial_rate2 : ℕ := 180
  let additional_workers_changed_rate1 : ℕ := 280
  let additional_workers_changed_rate2 : ℕ := 190
  let initial_period : ℕ := 2
  let second_period : ℕ := 4
  let additional_workers_initial_period : ℕ := 1

  8

theorem truck_filling_time_proof : truck_filling_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_truck_filling_time_truck_filling_time_proof_l1126_112609


namespace NUMINAMATH_CALUDE_weight_of_new_person_l1126_112652

theorem weight_of_new_person 
  (n : ℕ) 
  (initial_average : ℝ) 
  (weight_increase : ℝ) 
  (replaced_weight : ℝ) : 
  n = 8 → 
  weight_increase = 5 → 
  replaced_weight = 35 → 
  (n * initial_average + (n * weight_increase + replaced_weight)) / n = 
    initial_average + weight_increase → 
  n * weight_increase + replaced_weight = 75 := by
sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l1126_112652


namespace NUMINAMATH_CALUDE_max_flight_time_l1126_112684

/-- The maximum flight time for a projectile launched under specific conditions -/
theorem max_flight_time (V₀ g : ℝ) (h₁ : V₀ > 0) (h₂ : g > 0) : 
  ∃ (τ : ℝ), τ = (2 * V₀ / g) * (2 / Real.sqrt 12.5) ∧ 
  ∀ (α : ℝ), 0 ≤ α ∧ α ≤ π / 2 ∧ Real.sin (2 * α) ≥ 0.96 → 
  (2 * V₀ * Real.sin α) / g ≤ τ :=
sorry

end NUMINAMATH_CALUDE_max_flight_time_l1126_112684


namespace NUMINAMATH_CALUDE_beaver_leaves_count_l1126_112631

theorem beaver_leaves_count :
  ∀ (beaver_dens raccoon_dens : ℕ),
    beaver_dens = raccoon_dens + 3 →
    5 * beaver_dens = 6 * raccoon_dens →
    5 * beaver_dens = 90 :=
by
  sorry

#check beaver_leaves_count

end NUMINAMATH_CALUDE_beaver_leaves_count_l1126_112631


namespace NUMINAMATH_CALUDE_angle_on_ray_l1126_112604

/-- Given an angle α where its initial side coincides with the non-negative half-axis of the x-axis
    and its terminal side lies on the ray 4x - 3y = 0 (x ≤ 0), cos α - sin α = 1/5 -/
theorem angle_on_ray (α : Real) : 
  (∃ (x y : Real), x ≤ 0 ∧ 4 * x - 3 * y = 0 ∧ 
   x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
   y = Real.sin α * Real.sqrt (x^2 + y^2)) → 
  Real.cos α - Real.sin α = 1/5 := by
sorry

end NUMINAMATH_CALUDE_angle_on_ray_l1126_112604


namespace NUMINAMATH_CALUDE_polygon_diagonals_l1126_112621

theorem polygon_diagonals (n : ℕ) : 
  (n ≥ 3) → (n - 3 ≤ 5) → (n = 8) :=
by sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l1126_112621


namespace NUMINAMATH_CALUDE_tangent_inclination_range_implies_x_coordinate_range_l1126_112646

/-- The curve C defined by y = x^2 + 2x + 3 -/
def C (x : ℝ) : ℝ := x^2 + 2*x + 3

/-- The derivative of C -/
def C' (x : ℝ) : ℝ := 2*x + 2

theorem tangent_inclination_range_implies_x_coordinate_range :
  ∀ x : ℝ,
  (∃ y : ℝ, y = C x) →
  (π/4 ≤ Real.arctan (C' x) ∧ Real.arctan (C' x) ≤ π/2) →
  x ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_inclination_range_implies_x_coordinate_range_l1126_112646


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1126_112667

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1126_112667


namespace NUMINAMATH_CALUDE_min_value_problem_l1126_112601

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2/x + 8/y = 1) :
  x + y ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1126_112601


namespace NUMINAMATH_CALUDE_circle_properties_l1126_112694

-- Define the circle C
def circle_C (D E F : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0}

-- Define the line l
def line_l (D E F : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | D * p.1 + E * p.2 + F = 0}

-- Define the circle M
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

theorem circle_properties (D E F : ℝ) 
    (h1 : D^2 + E^2 = F^2) 
    (h2 : F > 0) : 
  F > 4 ∧ 
  (let d := |F - 2| / 2
   let r := Real.sqrt (F^2 - 4*F) / 2
   d^2 - r^2 = 1) ∧
  (∃ M : Set (ℝ × ℝ), M = circle_M ∧ 
    (∀ p ∈ M, p ∈ line_l D E F → False) ∧
    (∀ p ∈ M, p ∈ circle_C D E F → False)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1126_112694


namespace NUMINAMATH_CALUDE_least_multiple_36_with_digit_sum_multiple_9_l1126_112617

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem least_multiple_36_with_digit_sum_multiple_9 :
  ∀ k : ℕ, k > 0 → 36 * k ≠ 36 →
    digit_sum (36 * k) % 9 = 0 → digit_sum 36 % 9 = 0 ∧ 36 < 36 * k :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_36_with_digit_sum_multiple_9_l1126_112617


namespace NUMINAMATH_CALUDE_num_tetrahedrons_in_cube_l1126_112650

/-- A cube is represented by its 8 vertices -/
structure Cube :=
  (vertices : Fin 8 → Point)

/-- A tetrahedron is represented by its 4 vertices -/
structure Tetrahedron :=
  (vertices : Fin 4 → Point)

/-- Function to check if a set of 4 vertices forms a valid tetrahedron -/
def is_valid_tetrahedron (c : Cube) (t : Tetrahedron) : Prop :=
  sorry

/-- The number of valid tetrahedrons that can be formed from the vertices of a cube -/
def num_tetrahedrons (c : Cube) : ℕ :=
  sorry

/-- Theorem stating that the number of tetrahedrons formed from a cube's vertices is 58 -/
theorem num_tetrahedrons_in_cube (c : Cube) : num_tetrahedrons c = 58 :=
  sorry

end NUMINAMATH_CALUDE_num_tetrahedrons_in_cube_l1126_112650


namespace NUMINAMATH_CALUDE_rhombus_area_l1126_112697

/-- The area of a rhombus with side length √145 and diagonal difference 12 is 236 + 6√254 -/
theorem rhombus_area (side_length : ℝ) (diagonal_difference : ℝ) (area : ℝ) : 
  side_length = Real.sqrt 145 →
  diagonal_difference = 12 →
  area = 236 + 6 * Real.sqrt 254 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_l1126_112697


namespace NUMINAMATH_CALUDE_hazel_walk_l1126_112641

/-- Hazel's walk problem -/
theorem hazel_walk (first_hour_distance : ℝ) (h1 : first_hour_distance = 2) :
  let second_hour_distance := 2 * first_hour_distance
  first_hour_distance + second_hour_distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_hazel_walk_l1126_112641


namespace NUMINAMATH_CALUDE_chocolate_box_pieces_l1126_112634

theorem chocolate_box_pieces (initial_boxes : ℕ) (given_away : ℕ) (remaining_pieces : ℕ) :
  initial_boxes = 7 →
  given_away = 3 →
  remaining_pieces = 16 →
  ∃ (pieces_per_box : ℕ), 
    pieces_per_box * (initial_boxes - given_away) = remaining_pieces ∧
    pieces_per_box = 4 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_box_pieces_l1126_112634


namespace NUMINAMATH_CALUDE_dogs_return_simultaneously_l1126_112664

theorem dogs_return_simultaneously
  (L : ℝ)  -- Distance between the two people
  (v V u : ℝ)  -- Speeds of slow person, fast person, and dogs respectively
  (h1 : 0 < v)
  (h2 : v < V)
  (h3 : 0 < u)
  (h4 : V < u) :
  (2 * L * u) / ((u + V) * (u + v)) = (2 * L * u) / ((u + v) * (u + V)) :=
by sorry

end NUMINAMATH_CALUDE_dogs_return_simultaneously_l1126_112664


namespace NUMINAMATH_CALUDE_fraction_sum_lower_bound_sum_lower_bound_l1126_112658

-- Part 1
theorem fraction_sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  1 / a + 1 / (b + 1) ≥ 4 / 5 := by sorry

-- Part 2
theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + a * b = 8) :
  a + b ≥ 4 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_lower_bound_sum_lower_bound_l1126_112658


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l1126_112649

def number_of_lineups (total_players : ℕ) (captain_count : ℕ) (regular_players : ℕ) : ℕ :=
  total_players * (Nat.choose (total_players - 1) regular_players)

theorem basketball_lineup_count :
  number_of_lineups 12 1 5 = 5544 := by
sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l1126_112649


namespace NUMINAMATH_CALUDE_joes_dad_marshmallow_fraction_l1126_112668

theorem joes_dad_marshmallow_fraction :
  ∀ (dad_marshmallows joe_marshmallows dad_roasted joe_roasted total_roasted : ℕ),
    dad_marshmallows = 21 →
    joe_marshmallows = 4 * dad_marshmallows →
    joe_roasted = joe_marshmallows / 2 →
    total_roasted = 49 →
    total_roasted = joe_roasted + dad_roasted →
    (dad_roasted : ℚ) / dad_marshmallows = 1 / 3 := by
  sorry

#check joes_dad_marshmallow_fraction

end NUMINAMATH_CALUDE_joes_dad_marshmallow_fraction_l1126_112668


namespace NUMINAMATH_CALUDE_unique_a_for_perpendicular_chords_l1126_112679

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define a point on the x-axis
def point_on_x_axis (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the property of perpendicular lines intersecting the hyperbola
def perpendicular_lines_property (a : ℝ) : Prop :=
  ∀ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, l₁ x y ↔ l₂ y (-x)) →  -- l₁ and l₂ are perpendicular
    (l₁ a 0 ∧ l₂ a 0) →  -- both lines pass through (a, 0)
    ∀ (px py qx qy rx ry sx sy : ℝ),
      (hyperbola px py ∧ hyperbola qx qy ∧ l₁ px py ∧ l₁ qx qy) →  -- P and Q on l₁ and hyperbola
      (hyperbola rx ry ∧ hyperbola sx sy ∧ l₂ rx ry ∧ l₂ sx sy) →  -- R and S on l₂ and hyperbola
      (px - qx)^2 + (py - qy)^2 = (rx - sx)^2 + (ry - sy)^2  -- |PQ| = |RS|

-- The main theorem
theorem unique_a_for_perpendicular_chords :
  ∃! (a : ℝ), a > 1 ∧ perpendicular_lines_property a ∧ a = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_unique_a_for_perpendicular_chords_l1126_112679


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1126_112629

theorem geometric_sequence_fourth_term
  (a : ℝ)  -- first term
  (a₆ : ℝ) -- sixth term
  (h₁ : a = 81)
  (h₂ : a₆ = 32)
  (h₃ : ∃ r : ℝ, r > 0 ∧ a₆ = a * r^5) :
  ∃ a₄ : ℝ, a₄ = 24 ∧ ∃ r : ℝ, r > 0 ∧ a₄ = a * r^3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1126_112629


namespace NUMINAMATH_CALUDE_base_eight_digit_product_l1126_112670

/-- Represents a number in base 8 as a list of digits --/
def BaseEight := List Nat

/-- Converts a natural number to its base 8 representation --/
def toBaseEight (n : Nat) : BaseEight :=
  sorry

/-- Decrements each digit in a BaseEight number by 1, removing 0s --/
def decrementDigits (b : BaseEight) : BaseEight :=
  sorry

/-- Computes the product of a list of natural numbers --/
def product (l : List Nat) : Nat :=
  sorry

theorem base_eight_digit_product (n : Nat) :
  n = 7654 →
  product (decrementDigits (toBaseEight n)) = 10 :=
sorry

end NUMINAMATH_CALUDE_base_eight_digit_product_l1126_112670


namespace NUMINAMATH_CALUDE_octal_to_decimal_l1126_112647

theorem octal_to_decimal : (3 * 8^2 + 6 * 8^1 + 7 * 8^0) = 247 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_l1126_112647


namespace NUMINAMATH_CALUDE_cos_420_degrees_l1126_112686

theorem cos_420_degrees : Real.cos (420 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_420_degrees_l1126_112686


namespace NUMINAMATH_CALUDE_value_of_two_over_x_l1126_112651

theorem value_of_two_over_x (x : ℂ) (h : 1 - 5 / x + 9 / x^2 = 0) :
  2 / x = Complex.ofReal (5 / 9) - Complex.I * Complex.ofReal (Real.sqrt 11 / 9) ∨
  2 / x = Complex.ofReal (5 / 9) + Complex.I * Complex.ofReal (Real.sqrt 11 / 9) := by
sorry

end NUMINAMATH_CALUDE_value_of_two_over_x_l1126_112651


namespace NUMINAMATH_CALUDE_john_tax_difference_l1126_112624

/-- Calculates the difference in taxes paid given old and new tax rates and incomes -/
def tax_difference (old_rate new_rate old_income new_income : ℚ) : ℚ :=
  new_rate * new_income - old_rate * old_income

/-- Proves that the difference in taxes paid by John is $250,000 -/
theorem john_tax_difference :
  tax_difference (20 / 100) (30 / 100) 1000000 1500000 = 250000 := by
  sorry

end NUMINAMATH_CALUDE_john_tax_difference_l1126_112624


namespace NUMINAMATH_CALUDE_solve_congruence_l1126_112610

theorem solve_congruence :
  ∃ n : ℕ, 0 ≤ n ∧ n < 43 ∧ (11 * n) % 43 = 7 % 43 ∧ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_solve_congruence_l1126_112610


namespace NUMINAMATH_CALUDE_product_xy_is_60_l1126_112640

/-- A line passing through the origin with slope 1/2 -/
def line_k (x y : ℝ) : Prop := y = (1/2) * x

theorem product_xy_is_60 (x y : ℝ) :
  line_k x 6 ∧ line_k 10 y → x * y = 60 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_is_60_l1126_112640


namespace NUMINAMATH_CALUDE_negative_integer_solutions_l1126_112628

def satisfies_inequalities (x : ℤ) : Prop :=
  3 * x - 2 ≥ 2 * x - 5 ∧ x / 2 - (x - 2) / 3 < 1 / 2

theorem negative_integer_solutions :
  {x : ℤ | x < 0 ∧ satisfies_inequalities x} = {-3, -2} :=
by sorry

end NUMINAMATH_CALUDE_negative_integer_solutions_l1126_112628


namespace NUMINAMATH_CALUDE_dog_walking_distance_l1126_112627

theorem dog_walking_distance (total_weekly_miles : ℝ) (dog1_daily_miles : ℝ) (dog2_daily_miles : ℝ) :
  total_weekly_miles = 70 ∧ dog1_daily_miles = 2 →
  dog2_daily_miles = 8 := by
sorry

end NUMINAMATH_CALUDE_dog_walking_distance_l1126_112627


namespace NUMINAMATH_CALUDE_lcm_gcf_relation_l1126_112656

theorem lcm_gcf_relation (n : ℕ) :
  (Nat.lcm n 12 = 42) ∧ (Nat.gcd n 12 = 6) → n = 21 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relation_l1126_112656


namespace NUMINAMATH_CALUDE_biker_bob_distance_l1126_112612

/-- The total distance covered by Biker Bob between town A and town B -/
def total_distance : ℝ := 155

/-- The distance of the first segment (west) -/
def distance_west : ℝ := 45

/-- The distance of the second segment (northwest) -/
def distance_northwest : ℝ := 25

/-- The distance of the third segment (south) -/
def distance_south : ℝ := 35

/-- The distance of the fourth segment (east) -/
def distance_east : ℝ := 50

/-- Theorem stating that the total distance is the sum of all segment distances -/
theorem biker_bob_distance : 
  total_distance = distance_west + distance_northwest + distance_south + distance_east := by
  sorry

#check biker_bob_distance

end NUMINAMATH_CALUDE_biker_bob_distance_l1126_112612


namespace NUMINAMATH_CALUDE_f_odd_when_a_zero_f_increasing_iff_three_roots_iff_l1126_112613

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * abs (2 * a - x) + 2 * x

-- Statement 1: f is odd when a = 0
theorem f_odd_when_a_zero : 
  ∀ x : ℝ, f 0 (-x) = -(f 0 x) :=
sorry

-- Statement 2: f is increasing on ℝ iff -1 ≤ a ≤ 1
theorem f_increasing_iff :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

-- Statement 3: f(x) - tf(2a) = 0 has three distinct roots iff 1 < t < 9/8
theorem three_roots_iff :
  ∀ a t : ℝ, a ∈ Set.Icc (-2) 2 →
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f a x - t * f a (2 * a) = 0 ∧
    f a y - t * f a (2 * a) = 0 ∧
    f a z - t * f a (2 * a) = 0) ↔
  1 < t ∧ t < 9/8 :=
sorry

end NUMINAMATH_CALUDE_f_odd_when_a_zero_f_increasing_iff_three_roots_iff_l1126_112613


namespace NUMINAMATH_CALUDE_distance_traveled_l1126_112636

theorem distance_traveled (initial_reading lunch_reading : ℝ) 
  (h1 : initial_reading = 212.3)
  (h2 : lunch_reading = 372.0) :
  lunch_reading - initial_reading = 159.7 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l1126_112636


namespace NUMINAMATH_CALUDE_four_propositions_true_l1126_112662

theorem four_propositions_true : 
  (∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0) ∧
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) ∧
  (¬ ∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0) ∧
  (¬ ∀ x y : ℝ, x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_four_propositions_true_l1126_112662


namespace NUMINAMATH_CALUDE_pop_spending_proof_l1126_112645

/-- The amount of money Pop spent on cereal -/
def pop_spending : ℝ := 15

/-- The amount of money Crackle spent on cereal -/
def crackle_spending : ℝ := 3 * pop_spending

/-- The amount of money Snap spent on cereal -/
def snap_spending : ℝ := 2 * crackle_spending

/-- The total amount spent on cereal -/
def total_spending : ℝ := 150

theorem pop_spending_proof :
  pop_spending + crackle_spending + snap_spending = total_spending ∧
  pop_spending = 15 := by
  sorry

end NUMINAMATH_CALUDE_pop_spending_proof_l1126_112645


namespace NUMINAMATH_CALUDE_teacher_age_l1126_112614

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) :
  num_students = 100 →
  student_avg_age = 17 →
  total_avg_age = 18 →
  (num_students : ℝ) * student_avg_age + (num_students + 1 : ℝ) * total_avg_age - (num_students : ℝ) * student_avg_age = 118 :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l1126_112614


namespace NUMINAMATH_CALUDE_decimal_259_to_base5_l1126_112635

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: decimal_to_base5 (n / 5)

theorem decimal_259_to_base5 :
  decimal_to_base5 259 = [4, 1, 0, 2] := by sorry

end NUMINAMATH_CALUDE_decimal_259_to_base5_l1126_112635


namespace NUMINAMATH_CALUDE_line_intercept_ratio_l1126_112680

theorem line_intercept_ratio (b : ℝ) (u v : ℝ) (h : b ≠ 0) : 
  (5 * u + b = 0) → (3 * v + b = 0) → u / v = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_ratio_l1126_112680


namespace NUMINAMATH_CALUDE_abs_4x_minus_2_not_positive_l1126_112687

theorem abs_4x_minus_2_not_positive (x : ℚ) : 
  ¬(|4*x - 2| > 0) ↔ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_4x_minus_2_not_positive_l1126_112687


namespace NUMINAMATH_CALUDE_total_amount_l1126_112690

-- Define the amounts received by A, B, and C
variable (A B C : ℝ)

-- Define the conditions
axiom condition1 : A = (1/3) * (B + C)
axiom condition2 : B = (2/7) * (A + C)
axiom condition3 : A = B + 20

-- Define the total amount
def total : ℝ := A + B + C

-- Theorem statement
theorem total_amount : total A B C = 720 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_l1126_112690


namespace NUMINAMATH_CALUDE_sum_remainder_nine_l1126_112637

theorem sum_remainder_nine (n : ℤ) : ((9 - n) + (n + 4)) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_nine_l1126_112637


namespace NUMINAMATH_CALUDE_maddie_weekend_watching_time_maddie_weekend_watching_time_proof_l1126_112602

/-- Given Maddie's TV watching schedule, prove she watched 105 minutes over the weekend -/
theorem maddie_weekend_watching_time : ℕ → Prop :=
  λ weekend_minutes : ℕ =>
    let total_episodes : ℕ := 8
    let minutes_per_episode : ℕ := 44
    let monday_minutes : ℕ := 138
    let thursday_minutes : ℕ := 21
    let friday_episodes : ℕ := 2

    let total_minutes : ℕ := total_episodes * minutes_per_episode
    let weekday_minutes : ℕ := monday_minutes + thursday_minutes + (friday_episodes * minutes_per_episode)

    weekend_minutes = total_minutes - weekday_minutes ∧ weekend_minutes = 105

/-- Proof of the theorem -/
theorem maddie_weekend_watching_time_proof : maddie_weekend_watching_time 105 := by
  sorry

end NUMINAMATH_CALUDE_maddie_weekend_watching_time_maddie_weekend_watching_time_proof_l1126_112602


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_prism_l1126_112639

theorem sphere_surface_area_rectangular_prism (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  4 * π * radius^2 = 50 * π := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_prism_l1126_112639
