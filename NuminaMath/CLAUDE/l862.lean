import Mathlib

namespace NUMINAMATH_CALUDE_cat_speed_l862_86232

/-- Proves that a cat's speed is 90 km/h given specific conditions -/
theorem cat_speed (rat_speed : ℝ) (head_start : ℝ) (catch_time : ℝ) :
  rat_speed = 36 →
  head_start = 6 →
  catch_time = 4 →
  rat_speed * (head_start + catch_time) = 90 * catch_time :=
by
  sorry

#check cat_speed

end NUMINAMATH_CALUDE_cat_speed_l862_86232


namespace NUMINAMATH_CALUDE_exists_permutation_1984_divisible_by_7_l862_86219

/-- A permutation of the digits of 1984 -/
def Permutation1984 : Type :=
  { p : Nat // p ∈ ({1498, 1849, 1948, 1984, 1894, 1489, 9148} : Set Nat) }

/-- Theorem: For any positive integer N, there exists a permutation of 1984's digits
    that when added to N, is divisible by 7 -/
theorem exists_permutation_1984_divisible_by_7 (N : Nat) :
  ∃ (p : Permutation1984), 7 ∣ (N + p.val) := by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_1984_divisible_by_7_l862_86219


namespace NUMINAMATH_CALUDE_large_hexagon_area_l862_86225

/-- Represents a regular hexagon -/
structure RegularHexagon where
  area : ℝ

/-- The large regular hexagon containing smaller hexagons -/
def large_hexagon : RegularHexagon := sorry

/-- One of the smaller regular hexagons -/
def small_hexagon : RegularHexagon := sorry

/-- The number of small hexagons in the large hexagon -/
def num_small_hexagons : ℕ := 7

/-- The number of small hexagons in the shaded area -/
def num_shaded_hexagons : ℕ := 6

/-- The area of the shaded part (6 small hexagons) -/
def shaded_area : ℝ := 180

theorem large_hexagon_area : large_hexagon.area = 270 := by sorry

end NUMINAMATH_CALUDE_large_hexagon_area_l862_86225


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_neg_three_range_of_a_when_p_equiv_q_l862_86284

-- Define the conditions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 6*x + 8 ≤ 0

-- Theorem for part (1)
theorem range_of_x_when_a_is_neg_three :
  ∀ x : ℝ, (p x (-3) ∧ q x) ↔ -4 ≤ x ∧ x < -3 :=
sorry

-- Theorem for part (2)
theorem range_of_a_when_p_equiv_q :
  ∀ a : ℝ, (∀ x : ℝ, p x a ↔ q x) ↔ -2 < a ∧ a < -4/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_neg_three_range_of_a_when_p_equiv_q_l862_86284


namespace NUMINAMATH_CALUDE_inequality_solution_set_l862_86278

theorem inequality_solution_set (x : ℝ) :
  (3 * x + 2 ≥ 5) ↔ (x ≥ 1) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l862_86278


namespace NUMINAMATH_CALUDE_skating_minutes_tenth_day_l862_86262

def minutes_per_day_first_5 : ℕ := 75
def days_first_period : ℕ := 5
def minutes_per_day_next_3 : ℕ := 120
def days_second_period : ℕ := 3
def total_days : ℕ := 10
def target_average : ℕ := 95

theorem skating_minutes_tenth_day : 
  ∃ (x : ℕ), 
    (minutes_per_day_first_5 * days_first_period + 
     minutes_per_day_next_3 * days_second_period + x) / total_days = target_average ∧
    x = 215 := by
  sorry

end NUMINAMATH_CALUDE_skating_minutes_tenth_day_l862_86262


namespace NUMINAMATH_CALUDE_inequality_reversal_l862_86272

theorem inequality_reversal (x y : ℝ) (h : x < y) : ¬(-2 * x < -2 * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l862_86272


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l862_86235

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a triangle formed by three points -/
structure Triangle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- Given three mutually externally tangent circles with radii 1, 2, and 3,
    returns the triangle formed by their points of tangency -/
def tangentTriangle (c1 c2 c3 : Circle) : Triangle := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Three mutually externally tangent circles with radii 1, 2, and 3 -/
def circle1 : Circle := { center := (0, 0), radius := 1 }
def circle2 : Circle := { center := (3, 0), radius := 2 }
def circle3 : Circle := { center := (0, 4), radius := 3 }

theorem tangent_triangle_area :
  triangleArea (tangentTriangle circle1 circle2 circle3) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_triangle_area_l862_86235


namespace NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l862_86241

theorem abs_m_minus_n_equals_five (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : 
  |m - n| = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l862_86241


namespace NUMINAMATH_CALUDE_allowance_spending_l862_86220

theorem allowance_spending (weekly_allowance : ℚ) 
  (arcade_fraction : ℚ) (candy_amount : ℚ) : 
  weekly_allowance = 3.75 →
  arcade_fraction = 3/5 →
  candy_amount = 1 →
  let remaining_after_arcade := weekly_allowance - arcade_fraction * weekly_allowance
  let toy_store_amount := remaining_after_arcade - candy_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_allowance_spending_l862_86220


namespace NUMINAMATH_CALUDE_cookies_per_box_type1_is_12_l862_86293

/-- Represents the number of cookies in a box of the first type -/
def cookies_per_box_type1 : ℕ := 12

/-- Represents the number of cookies in a box of the second type -/
def cookies_per_box_type2 : ℕ := 20

/-- Represents the number of cookies in a box of the third type -/
def cookies_per_box_type3 : ℕ := 16

/-- Represents the number of boxes sold of the first type -/
def boxes_sold_type1 : ℕ := 50

/-- Represents the number of boxes sold of the second type -/
def boxes_sold_type2 : ℕ := 80

/-- Represents the number of boxes sold of the third type -/
def boxes_sold_type3 : ℕ := 70

/-- Represents the total number of cookies sold -/
def total_cookies_sold : ℕ := 3320

/-- Theorem stating that the number of cookies in each box of the first type is 12 -/
theorem cookies_per_box_type1_is_12 :
  cookies_per_box_type1 * boxes_sold_type1 +
  cookies_per_box_type2 * boxes_sold_type2 +
  cookies_per_box_type3 * boxes_sold_type3 = total_cookies_sold :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_box_type1_is_12_l862_86293


namespace NUMINAMATH_CALUDE_leah_saves_fifty_cents_l862_86271

/-- Represents the daily savings of Leah in dollars -/
def leah_daily_savings : ℝ := sorry

/-- Represents Josiah's total savings in dollars -/
def josiah_total_savings : ℝ := 0.25 * 24

/-- Represents Leah's total savings in dollars -/
def leah_total_savings : ℝ := leah_daily_savings * 20

/-- Represents Megan's total savings in dollars -/
def megan_total_savings : ℝ := 2 * leah_daily_savings * 12

/-- The total amount saved by all three children -/
def total_savings : ℝ := 28

/-- Theorem stating that Leah's daily savings amount to $0.50 -/
theorem leah_saves_fifty_cents :
  josiah_total_savings + leah_total_savings + megan_total_savings = total_savings →
  leah_daily_savings = 0.50 := by sorry

end NUMINAMATH_CALUDE_leah_saves_fifty_cents_l862_86271


namespace NUMINAMATH_CALUDE_circle_and_m_range_l862_86273

-- Define the circle S
def circle_S (x y : ℝ) := (x - 4)^2 + (y - 4)^2 = 25

-- Define the line that contains the center of S
def center_line (x y : ℝ) := 2*x - y - 4 = 0

-- Define the intersecting line
def intersecting_line (x y m : ℝ) := x + y - m = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (7, 8)
def point_B : ℝ × ℝ := (8, 7)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_and_m_range :
  ∀ (m : ℝ),
  (∃ (C D : ℝ × ℝ), 
    circle_S C.1 C.2 ∧ 
    circle_S D.1 D.2 ∧
    intersecting_line C.1 C.2 m ∧
    intersecting_line D.1 D.2 m ∧
    -- Angle COD is obtuse
    (C.1 * D.1 + C.2 * D.2 < 0)) →
  circle_S point_A.1 point_A.2 ∧
  circle_S point_B.1 point_B.2 ∧
  (∃ (center : ℝ × ℝ), center_line center.1 center.2 ∧ circle_S center.1 center.2) →
  1 < m ∧ m < 7 :=
sorry

end NUMINAMATH_CALUDE_circle_and_m_range_l862_86273


namespace NUMINAMATH_CALUDE_employee_discount_percentage_l862_86255

theorem employee_discount_percentage
  (wholesale_cost : ℝ)
  (markup_percentage : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : markup_percentage = 20)
  (h3 : employee_paid_price = 168) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discount_amount := retail_price - employee_paid_price
  let discount_percentage := (discount_amount / retail_price) * 100
  discount_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_employee_discount_percentage_l862_86255


namespace NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_z_l862_86229

def i : ℂ := Complex.I

theorem imaginary_part_of_pure_imaginary_z (a : ℝ) :
  let z : ℂ := a + 15 / (3 - 4 * i)
  (z.re = 0) → z.im = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_z_l862_86229


namespace NUMINAMATH_CALUDE_max_product_at_three_l862_86295

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n-1)

def product_of_terms (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  (a₁ * r^((n-1)/2))^n

theorem max_product_at_three (a₁ r : ℝ) (h₁ : a₁ = 3) (h₂ : r = 2/5) :
  ∀ k : ℕ, k ≠ 0 → product_of_terms a₁ r 3 ≥ product_of_terms a₁ r k :=
by sorry

end NUMINAMATH_CALUDE_max_product_at_three_l862_86295


namespace NUMINAMATH_CALUDE_negative_twenty_is_spend_l862_86231

/-- Represents a monetary transaction -/
inductive Transaction
| receive (amount : ℕ)
| spend (amount : ℕ)

/-- Converts a transaction to its signed representation -/
def signedAmount (t : Transaction) : ℤ :=
  match t with
  | Transaction.receive n => n
  | Transaction.spend n => -n

/-- The convention of representing transactions -/
structure TransactionConvention where
  positiveIsReceive : ∀ (n : ℕ), signedAmount (Transaction.receive n) > 0
  negativeIsSpend : ∀ (n : ℕ), signedAmount (Transaction.spend n) < 0

/-- The main theorem -/
theorem negative_twenty_is_spend (conv : TransactionConvention) :
  signedAmount (Transaction.spend 20) = -20 :=
by sorry

end NUMINAMATH_CALUDE_negative_twenty_is_spend_l862_86231


namespace NUMINAMATH_CALUDE_division_problem_l862_86277

theorem division_problem (x y : ℕ+) 
  (h1 : x = 10 * y + 3)
  (h2 : 2 * x = 21 * y + 1) :
  11 * y - x = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l862_86277


namespace NUMINAMATH_CALUDE_bricks_per_course_l862_86256

/-- Proves that the number of bricks in each course is 400 --/
theorem bricks_per_course (initial_courses : ℕ) (added_courses : ℕ) (total_bricks : ℕ) :
  initial_courses = 3 →
  added_courses = 2 →
  total_bricks = 1800 →
  ∃ (bricks_per_course : ℕ),
    bricks_per_course * (initial_courses + added_courses) - bricks_per_course / 2 = total_bricks ∧
    bricks_per_course = 400 := by
  sorry

end NUMINAMATH_CALUDE_bricks_per_course_l862_86256


namespace NUMINAMATH_CALUDE_grid_division_theorem_l862_86236

/-- Represents a grid division into squares and corners -/
structure GridDivision where
  squares : ℕ  -- number of 2x2 squares
  corners : ℕ  -- number of 3-cell corners

/-- Checks if a grid division is valid for a 7x14 grid -/
def is_valid_division (d : GridDivision) : Prop :=
  4 * d.squares + 3 * d.corners = 7 * 14

theorem grid_division_theorem :
  -- Part a: There exists a valid division where squares = corners
  (∃ d : GridDivision, is_valid_division d ∧ d.squares = d.corners) ∧
  -- Part b: There does not exist a valid division where squares > corners
  (¬ ∃ d : GridDivision, is_valid_division d ∧ d.squares > d.corners) := by
  sorry

end NUMINAMATH_CALUDE_grid_division_theorem_l862_86236


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l862_86275

theorem max_value_theorem (A M C : ℕ) (h : A + M + C = 15) :
  2 * (A * M * C) + A * M + M * C + C * A ≤ 325 :=
by sorry

theorem max_value_achieved (A M C : ℕ) (h : A + M + C = 15) :
  ∃ A M C, A + M + C = 15 ∧ 2 * (A * M * C) + A * M + M * C + C * A = 325 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l862_86275


namespace NUMINAMATH_CALUDE_min_fourth_integer_l862_86281

theorem min_fourth_integer (A B C D : ℕ+) : 
  (A + B + C + D : ℚ) / 4 = 16 →
  A = 3 * B →
  B = C - 2 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  D ≥ 52 :=
by sorry

end NUMINAMATH_CALUDE_min_fourth_integer_l862_86281


namespace NUMINAMATH_CALUDE_tree_height_l862_86294

/-- The height of a tree given specific conditions involving a rope and a person. -/
theorem tree_height (rope_ground_distance : ℝ) (person_distance : ℝ) (person_height : ℝ) 
  (h1 : rope_ground_distance = 4)
  (h2 : person_distance = 3)
  (h3 : person_height = 1.6)
  (h4 : person_distance < rope_ground_distance) : 
  ∃ (tree_height : ℝ), tree_height = 6.4 := by
  sorry


end NUMINAMATH_CALUDE_tree_height_l862_86294


namespace NUMINAMATH_CALUDE_students_on_pullout_couch_l862_86299

theorem students_on_pullout_couch (total_students : ℕ) (num_rooms : ℕ) (students_per_bed : ℕ) (beds_per_room : ℕ) :
  total_students = 30 →
  num_rooms = 6 →
  students_per_bed = 2 →
  beds_per_room = 2 →
  (total_students / num_rooms - students_per_bed * beds_per_room : ℕ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_students_on_pullout_couch_l862_86299


namespace NUMINAMATH_CALUDE_sarah_money_l862_86209

/-- Given that Bridge and Sarah have 300 cents in total, and Bridge has 50 cents more than Sarah,
    prove that Sarah has 125 cents. -/
theorem sarah_money : 
  ∀ (sarah_cents bridge_cents : ℕ), 
    sarah_cents + bridge_cents = 300 →
    bridge_cents = sarah_cents + 50 →
    sarah_cents = 125 := by
  sorry

end NUMINAMATH_CALUDE_sarah_money_l862_86209


namespace NUMINAMATH_CALUDE_hash_two_three_four_l862_86283

/-- The # operation for real numbers -/
def hash (r s t : ℝ) : ℝ := r + s + t + r*s + r*t + s*t + r*s*t

/-- Theorem stating that 2 # 3 # 4 = 59 -/
theorem hash_two_three_four : hash 2 3 4 = 59 := by
  sorry

end NUMINAMATH_CALUDE_hash_two_three_four_l862_86283


namespace NUMINAMATH_CALUDE_smallest_three_way_sum_of_squares_l862_86221

/-- A function that returns true if a number can be expressed as the sum of two squares -/
def isSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^2 + b^2 = n

/-- A function that counts the number of ways a number can be expressed as the sum of two squares -/
def countSumOfTwoSquares (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card

/-- The theorem stating that 325 is the smallest number that can be expressed as the sum of two squares in three distinct ways -/
theorem smallest_three_way_sum_of_squares :
  (∀ m : ℕ, m < 325 → countSumOfTwoSquares m < 3) ∧
  countSumOfTwoSquares 325 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_way_sum_of_squares_l862_86221


namespace NUMINAMATH_CALUDE_find_b_l862_86276

theorem find_b (a b c : ℤ) (eq1 : a + 5 = b) (eq2 : 5 + b = c) (eq3 : b + c = a) : b = -10 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l862_86276


namespace NUMINAMATH_CALUDE_curve_description_l862_86204

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def right_half (x y : ℝ) : Prop := x = Real.sqrt (1 - y^2)

def lower_half (x y : ℝ) : Prop := y = -Real.sqrt (1 - x^2)

def curve_equation (x y : ℝ) : Prop :=
  (x - Real.sqrt (1 - y^2)) * (y + Real.sqrt (1 - x^2)) = 0

theorem curve_description (x y : ℝ) :
  unit_circle x y ∧ (right_half x y ∨ lower_half x y) ↔ curve_equation x y :=
sorry

end NUMINAMATH_CALUDE_curve_description_l862_86204


namespace NUMINAMATH_CALUDE_number_of_men_in_first_group_l862_86215

-- Define the number of men in the first group
def M : ℕ := sorry

-- Define the given conditions
def hours_per_day_group1 : ℕ := 10
def earnings_per_week_group1 : ℕ := 1000
def men_group2 : ℕ := 9
def hours_per_day_group2 : ℕ := 6
def earnings_per_week_group2 : ℕ := 1350
def days_per_week : ℕ := 7

-- Theorem to prove
theorem number_of_men_in_first_group :
  (M * hours_per_day_group1 * days_per_week) / earnings_per_week_group1 =
  (men_group2 * hours_per_day_group2 * days_per_week) / earnings_per_week_group2 →
  M = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_in_first_group_l862_86215


namespace NUMINAMATH_CALUDE_age_problem_l862_86214

theorem age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 28 →
  (a + c) / 2 = 29 →
  b = 26 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l862_86214


namespace NUMINAMATH_CALUDE_integral_x_sin_ax_over_x2_plus_k2_l862_86264

/-- The integral of x*sin(ax)/(x^2 + k^2) from 0 to infinity equals (π/2)*e^(-ak) for positive a and k -/
theorem integral_x_sin_ax_over_x2_plus_k2 (a k : ℝ) (ha : a > 0) (hk : k > 0) :
  ∫ (x : ℝ) in Set.Ici 0, (x * Real.sin (a * x)) / (x^2 + k^2) = (Real.pi / 2) * Real.exp (-a * k) := by
  sorry

end NUMINAMATH_CALUDE_integral_x_sin_ax_over_x2_plus_k2_l862_86264


namespace NUMINAMATH_CALUDE_dot_product_AP_BP_l862_86228

/-- The dot product of vectors AP and BP, where P is a point on a specific ellipse satisfying certain conditions. -/
theorem dot_product_AP_BP : ∃ (x y : ℝ), 
  (x^2 / 12 + y^2 / 16 = 1) ∧ 
  (((x - 0)^2 + (y - (-2))^2).sqrt - ((x - 0)^2 + (y - 2)^2).sqrt = 2) →
  (x * x + (y + 2) * (y - 2) = 9) := by
sorry

end NUMINAMATH_CALUDE_dot_product_AP_BP_l862_86228


namespace NUMINAMATH_CALUDE_frequency_limit_theorem_l862_86292

/-- A fair coin toss experiment -/
structure CoinToss where
  /-- The number of tosses -/
  n : ℕ
  /-- The number of heads -/
  heads : ℕ
  /-- The number of heads is less than or equal to the number of tosses -/
  heads_le_n : heads ≤ n

/-- The frequency of heads in a coin toss experiment -/
def frequency (ct : CoinToss) : ℚ :=
  ct.heads / ct.n

/-- The limit of the frequency of heads as the number of tosses approaches infinity -/
theorem frequency_limit_theorem :
  ∀ ε > 0, ∃ N : ℕ, ∀ ct : CoinToss, ct.n ≥ N → |frequency ct - 1/2| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_limit_theorem_l862_86292


namespace NUMINAMATH_CALUDE_cubic_equation_roots_difference_l862_86286

theorem cubic_equation_roots_difference (x : ℝ) : 
  (64 * x^3 - 144 * x^2 + 92 * x - 15 = 0) →
  (∃ a d : ℝ, {a - d, a, a + d} ⊆ {x | 64 * x^3 - 144 * x^2 + 92 * x - 15 = 0}) →
  (∃ r₁ r₂ r₃ : ℝ, 
    r₁ < r₂ ∧ r₂ < r₃ ∧
    {r₁, r₂, r₃} = {x | 64 * x^3 - 144 * x^2 + 92 * x - 15 = 0} ∧
    r₃ - r₁ = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_difference_l862_86286


namespace NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_length_l862_86202

/-- The length of the repeating block in the decimal expansion of 7/13 -/
def repeating_block_length : ℕ := 6

/-- 7 is prime -/
axiom seven_prime : Nat.Prime 7

/-- 13 is prime -/
axiom thirteen_prime : Nat.Prime 13

/-- The theorem stating that the length of the repeating block in the decimal expansion of 7/13 is 6 -/
theorem seven_thirteenths_repeating_block_length :
  repeating_block_length = 6 := by sorry

end NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_length_l862_86202


namespace NUMINAMATH_CALUDE_is_projection_matrix_l862_86268

def projection_matrix (M : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  M * M = M

theorem is_projection_matrix : 
  let M : Matrix (Fin 2) (Fin 2) ℚ := !![9/34, 25/34; 3/5, 15/34]
  projection_matrix M := by
  sorry

end NUMINAMATH_CALUDE_is_projection_matrix_l862_86268


namespace NUMINAMATH_CALUDE_quadratic_inequality_l862_86298

theorem quadratic_inequality (x : ℝ) : x^2 + 3*x < 10 ↔ -5 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l862_86298


namespace NUMINAMATH_CALUDE_solve_system_1_solve_system_2_l862_86282

-- First system of equations
theorem solve_system_1 (x y : ℝ) : 
  x - y - 1 = 0 ∧ 4 * (x - y) - y = 0 → x = 5 ∧ y = 4 := by
  sorry

-- Second system of equations
theorem solve_system_2 (x y : ℝ) :
  3 * x - y - 2 = 0 ∧ (6 * x - 2 * y + 1) / 5 + 3 * y = 10 → x = 5 / 3 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_1_solve_system_2_l862_86282


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l862_86261

theorem quadratic_one_solution (a : ℝ) : 
  (∃! x : ℝ, x^2 + a*x + 1 = 0) ↔ (a = 2 ∨ a = -2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l862_86261


namespace NUMINAMATH_CALUDE_range_of_a_l862_86291

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (Real.exp x - a)^2 + x^2 - 2*a*x + a^2 ≤ 1/2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l862_86291


namespace NUMINAMATH_CALUDE_sqrt_x_plus_2_equals_2_implies_x_plus_2_squared_equals_16_l862_86266

theorem sqrt_x_plus_2_equals_2_implies_x_plus_2_squared_equals_16 (x : ℝ) :
  Real.sqrt (x + 2) = 2 → (x + 2)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_2_equals_2_implies_x_plus_2_squared_equals_16_l862_86266


namespace NUMINAMATH_CALUDE_stephanies_speed_l862_86244

/-- Given a distance of 15 miles and a time of 3 hours, prove that the speed is 5 miles per hour. -/
theorem stephanies_speed (distance : ℝ) (time : ℝ) (h1 : distance = 15) (h2 : time = 3) :
  distance / time = 5 := by
  sorry

end NUMINAMATH_CALUDE_stephanies_speed_l862_86244


namespace NUMINAMATH_CALUDE_problem_solution_l862_86260

-- Define the propositions
def proposition_A (a : ℝ) : Prop :=
  ∀ x, x^2 + (2*a - 1)*x + a^2 > 0

def proposition_B (a : ℝ) : Prop :=
  ∀ x y, x < y → (a^2 - 1)^x > (a^2 - 1)^y

-- Define the theorem
theorem problem_solution :
  (∀ a : ℝ, (proposition_A a ∨ proposition_B a) ↔ (a < -1 ∧ a > -Real.sqrt 2) ∨ a > 1/4) ∧
  (∀ a : ℝ, a < -1 ∧ a > -Real.sqrt 2 → a^3 + 1 < a^2 + a) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l862_86260


namespace NUMINAMATH_CALUDE_stock_price_decrease_l862_86248

theorem stock_price_decrease (x : ℝ) (h : x > 0) :
  let increase_factor := 1.3
  let decrease_factor := 1 - 1 / increase_factor
  x = (1 - decrease_factor) * (increase_factor * x) :=
by sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l862_86248


namespace NUMINAMATH_CALUDE_binomial_18_10_l862_86227

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 8008) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 43758 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l862_86227


namespace NUMINAMATH_CALUDE_existence_of_epsilon_and_u_l862_86213

theorem existence_of_epsilon_and_u (n : ℕ+) :
  ∃ (ε : ℝ), 0 < ε ∧ ε < (1 : ℝ) / 2014 ∧
  ∀ (a : Fin n → ℝ), (∀ i, 0 < a i) →
  ∃ (u : ℝ), u > 0 ∧ ∀ i, ε < u * (a i) - ⌊u * (a i)⌋ ∧ u * (a i) - ⌊u * (a i)⌋ < (1 : ℝ) / 2014 :=
sorry

end NUMINAMATH_CALUDE_existence_of_epsilon_and_u_l862_86213


namespace NUMINAMATH_CALUDE_simplify_calculations_l862_86237

theorem simplify_calculations :
  (3.5 * 10.1 = 35.35) ∧
  (0.58 * 98 = 56.84) ∧
  (3.6 * 6.91 + 6.4 * 6.91 = 69.1) ∧
  ((19.1 - (1.64 + 2.36)) / 2.5 = 6.04) := by
  sorry

end NUMINAMATH_CALUDE_simplify_calculations_l862_86237


namespace NUMINAMATH_CALUDE_hyperbola_equation_l862_86201

/-- Given a hyperbola and a circle with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 5 = 0 →
    ∃ t : ℝ, (b*x + a*y = 0 ∨ b*x - a*y = 0) →
      (x - 3)^2 + y^2 = 4) →
  3^2 = a^2 - b^2 →
  a^2 = 5 ∧ b^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l862_86201


namespace NUMINAMATH_CALUDE_stock_yield_calculation_l862_86249

theorem stock_yield_calculation (a_price b_price b_yield : ℝ) 
  (h1 : a_price = 96)
  (h2 : b_price = 115.2)
  (h3 : b_yield = 0.12)
  (h4 : a_price * b_yield = b_price * (a_yield : ℝ)) :
  a_yield = 0.10 :=
by
  sorry

end NUMINAMATH_CALUDE_stock_yield_calculation_l862_86249


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l862_86242

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sin (2025 * x))^4 + (Real.cos (2016 * x))^2019 * (Real.cos (2025 * x))^2018 = 1 ↔ 
  (∃ n : ℤ, x = π / 4050 + π * n / 2025) ∨ (∃ k : ℤ, x = π * k / 9) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l862_86242


namespace NUMINAMATH_CALUDE_calculate_expression_l862_86253

theorem calculate_expression : 3^2 * 7 + 5 * 4^2 - 45 / 3 = 128 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l862_86253


namespace NUMINAMATH_CALUDE_comparison_square_and_power_l862_86287

theorem comparison_square_and_power (n : ℕ) (h : n ≥ 3) : (n + 1)^2 < 3^n := by
  sorry

end NUMINAMATH_CALUDE_comparison_square_and_power_l862_86287


namespace NUMINAMATH_CALUDE_smallest_a1_l862_86217

theorem smallest_a1 (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_rec : ∀ n > 1, a n = 7 * a (n - 1) - 2 * n) :
  (∀ a₁ : ℝ, (∀ n, a n > 0) → (∀ n > 1, a n = 7 * a (n - 1) - 2 * n) → a₁ ≥ a 1) →
  a 1 = 13 / 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_a1_l862_86217


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l862_86224

theorem quadratic_inequality_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 1 ≤ 0) → (m ≥ 2 ∨ m ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l862_86224


namespace NUMINAMATH_CALUDE_tims_bill_denomination_l862_86246

theorem tims_bill_denomination :
  let unknown_bills : ℕ := 13
  let five_dollar_bills : ℕ := 11
  let one_dollar_bills : ℕ := 17
  let total_amount : ℕ := 128
  let min_bills_used : ℕ := 16
  
  ∃ (x : ℕ),
    x * unknown_bills + 5 * five_dollar_bills + one_dollar_bills = total_amount ∧
    unknown_bills + five_dollar_bills + one_dollar_bills ≥ min_bills_used ∧
    x = 4 :=
by sorry

end NUMINAMATH_CALUDE_tims_bill_denomination_l862_86246


namespace NUMINAMATH_CALUDE_cone_height_calculation_l862_86200

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents a cone with a given base radius and height -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Theorem: Given three spheres and a cone touching externally on a flat surface,
    the height of the cone is 28 -/
theorem cone_height_calculation (s₁ s₂ s₃ : Sphere) (c : Cone) :
  s₁.radius = 20 →
  s₂.radius = 40 →
  s₃.radius = 40 →
  c.baseRadius = 21 →
  (∃ (arrangement : ℝ → ℝ → ℝ), 
    arrangement s₁.radius s₂.radius = arrangement s₁.radius s₃.radius ∧
    arrangement s₂.radius s₃.radius = s₂.radius + s₃.radius ∧
    arrangement s₁.radius s₂.radius = Real.sqrt ((s₁.radius + s₂.radius)^2 - (s₂.radius - s₁.radius)^2)) →
  c.height = 28 := by
  sorry


end NUMINAMATH_CALUDE_cone_height_calculation_l862_86200


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l862_86212

theorem fixed_point_of_exponential_function (a : ℝ) :
  let f : ℝ → ℝ := λ x => a^(x - 3) + 3
  f 3 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l862_86212


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l862_86216

theorem polygon_interior_angles_sum (n : ℕ) : n ≥ 3 →
  (2 * n - 2) * 180 = 2160 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l862_86216


namespace NUMINAMATH_CALUDE_sqrt_sum_equation_l862_86222

theorem sqrt_sum_equation (a b : ℚ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3) →
  ((a = 1/2 ∧ b = 3/2) ∨ (a = 3/2 ∧ b = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equation_l862_86222


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l862_86250

/-- Given a geometric sequence {a_n} where a_1+1, a_3+4, a_5+7 form an arithmetic sequence,
    the common difference of this arithmetic sequence is 3. -/
theorem geometric_arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d) :
  ∃ d : ℝ, d = 3 ∧ (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l862_86250


namespace NUMINAMATH_CALUDE_find_unknown_number_l862_86270

theorem find_unknown_number : ∃ x : ℝ, (20 + 40 + 60) / 3 = ((10 + 70 + x) / 3) + 4 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_number_l862_86270


namespace NUMINAMATH_CALUDE_circumscribed_sphere_volume_l862_86259

theorem circumscribed_sphere_volume (cube_surface_area : ℝ) (h : cube_surface_area = 24) :
  let cube_edge := Real.sqrt (cube_surface_area / 6)
  let sphere_radius := cube_edge * Real.sqrt 3 / 2
  (4 / 3) * Real.pi * sphere_radius ^ 3 = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_volume_l862_86259


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l862_86254

theorem cubic_roots_relation (a b c r s t : ℝ) : 
  (∀ x, x^3 + 3*x^2 + 4*x - 11 = (x - a) * (x - b) * (x - c)) →
  (∀ x, x^3 + r*x^2 + s*x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a))) →
  t = 23 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l862_86254


namespace NUMINAMATH_CALUDE_plane_division_l862_86207

/-- Represents a line on a plane -/
structure Line

/-- Represents a point on a plane -/
structure Point

/-- λ(P) represents the number of lines passing through a point P -/
def lambda (P : Point) (lines : Finset Line) : ℕ := sorry

/-- The set of all intersection points of the given lines -/
def intersectionPoints (lines : Finset Line) : Finset Point := sorry

/-- Theorem: For n lines on a plane, the total number of regions formed is 1+n+∑(λ(P)-1),
    and the number of unbounded regions is 2n -/
theorem plane_division (n : ℕ) (lines : Finset Line) 
  (h : lines.card = n) :
  (∃ (regions unboundedRegions : ℕ),
    regions = 1 + n + (intersectionPoints lines).sum (λ P => lambda P lines - 1) ∧
    unboundedRegions = 2 * n) :=
  sorry

end NUMINAMATH_CALUDE_plane_division_l862_86207


namespace NUMINAMATH_CALUDE_mowing_area_calculation_l862_86206

/-- Given that 3 mowers can mow 3 hectares in 3 days, 
    this theorem proves that 5 mowers can mow 25/3 hectares in 5 days. -/
theorem mowing_area_calculation 
  (mowers_initial : ℕ) 
  (days_initial : ℕ) 
  (area_initial : ℚ) 
  (mowers_final : ℕ) 
  (days_final : ℕ) 
  (h1 : mowers_initial = 3) 
  (h2 : days_initial = 3) 
  (h3 : area_initial = 3) 
  (h4 : mowers_final = 5) 
  (h5 : days_final = 5) :
  (area_initial * mowers_final * days_final) / (mowers_initial * days_initial) = 25 / 3 := by
  sorry

#check mowing_area_calculation

end NUMINAMATH_CALUDE_mowing_area_calculation_l862_86206


namespace NUMINAMATH_CALUDE_kgonal_number_formula_l862_86251

/-- The nth k-gonal number -/
def N (n k : ℕ) : ℚ :=
  match k with
  | 3 => (1/2 : ℚ) * n^2 + (1/2 : ℚ) * n
  | 4 => (n^2 : ℚ)
  | 5 => (3/2 : ℚ) * n^2 - (1/2 : ℚ) * n
  | 6 => (2 : ℚ) * n^2 - (n : ℚ)
  | _ => ((k - 2 : ℚ) / 2) * n^2 + ((4 - k : ℚ) / 2) * n

theorem kgonal_number_formula (n k : ℕ) (h : k ≥ 3) :
  N n k = ((k - 2 : ℚ) / 2) * n^2 + ((4 - k : ℚ) / 2) * n := by
  sorry

end NUMINAMATH_CALUDE_kgonal_number_formula_l862_86251


namespace NUMINAMATH_CALUDE_one_is_optimal_l862_86210

/-- Represents the number of teams that chose a particular number -/
def TeamChoices := ℕ → ℕ

/-- Calculates the score based on the game rules -/
def score (N : ℕ) (choices : TeamChoices) : ℕ :=
  if choices N > N then N else 0

/-- Theorem stating that 1 is the optimal choice -/
theorem one_is_optimal :
  ∀ (N : ℕ) (choices : TeamChoices),
    0 ≤ N ∧ N ≤ 20 →
    score 1 choices ≥ score N choices :=
sorry

end NUMINAMATH_CALUDE_one_is_optimal_l862_86210


namespace NUMINAMATH_CALUDE_same_last_digit_count_l862_86288

def has_same_last_digit (x : ℕ) : Bool :=
  x % 10 = (64 - x) % 10

def count_same_last_digit : ℕ :=
  (List.range 63).filter (λ x => has_same_last_digit (x + 1)) |>.length

theorem same_last_digit_count : count_same_last_digit = 13 := by
  sorry

end NUMINAMATH_CALUDE_same_last_digit_count_l862_86288


namespace NUMINAMATH_CALUDE_students_between_positions_l862_86208

theorem students_between_positions (n : ℕ) (h : n = 9) : 
  (n - 2) - (3 + 1) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_students_between_positions_l862_86208


namespace NUMINAMATH_CALUDE_race_head_start_l862_86239

theorem race_head_start (L : ℝ) (vₐ vᵦ : ℝ) (h : vₐ = (17 / 14) * vᵦ) :
  let x := (3 / 17) * L
  L / vₐ = (L - x) / vᵦ :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l862_86239


namespace NUMINAMATH_CALUDE_bill_total_is_95_l862_86289

/-- Represents a person's order at the restaurant -/
structure Order where
  appetizer_share : ℚ
  drinks_cost : ℚ
  dessert_cost : ℚ

/-- Calculates the total cost of an order -/
def total_cost (order : Order) : ℚ :=
  order.appetizer_share + order.drinks_cost + order.dessert_cost

/-- Represents the restaurant bill -/
def restaurant_bill (mary nancy fred steve : Order) : Prop :=
  let appetizer_total : ℚ := 28
  let appetizer_share : ℚ := appetizer_total / 4
  mary.appetizer_share = appetizer_share ∧
  nancy.appetizer_share = appetizer_share ∧
  fred.appetizer_share = appetizer_share ∧
  steve.appetizer_share = appetizer_share ∧
  mary.drinks_cost = 14 ∧
  nancy.drinks_cost = 11 ∧
  fred.drinks_cost = 12 ∧
  steve.drinks_cost = 6 ∧
  mary.dessert_cost = 8 ∧
  nancy.dessert_cost = 0 ∧
  fred.dessert_cost = 10 ∧
  steve.dessert_cost = 6

theorem bill_total_is_95 (mary nancy fred steve : Order) 
  (h : restaurant_bill mary nancy fred steve) : 
  total_cost mary + total_cost nancy + total_cost fred + total_cost steve = 95 := by
  sorry

end NUMINAMATH_CALUDE_bill_total_is_95_l862_86289


namespace NUMINAMATH_CALUDE_area_of_rectangle_l862_86280

/-- A square with two points on its sides forming a rectangle --/
structure SquareWithRectangle where
  -- Side length of the square
  side : ℝ
  -- Ratio of PT to PQ
  pt_ratio : ℝ
  -- Ratio of SU to SR
  su_ratio : ℝ
  -- Assumptions
  side_pos : 0 < side
  pt_ratio_pos : 0 < pt_ratio
  pt_ratio_lt_one : pt_ratio < 1
  su_ratio_pos : 0 < su_ratio
  su_ratio_lt_one : su_ratio < 1

/-- The perimeter of the rectangle PTUS --/
def rectangle_perimeter (s : SquareWithRectangle) : ℝ :=
  2 * (s.side * s.pt_ratio + s.side * s.su_ratio)

/-- The area of the rectangle PTUS --/
def rectangle_area (s : SquareWithRectangle) : ℝ :=
  (s.side * s.pt_ratio) * (s.side * s.su_ratio)

/-- Theorem: If PQRS is a square, T on PQ with PT:TQ = 1:2, U on SR with SU:UR = 1:2,
    and the perimeter of PTUS is 40 cm, then the area of PTUS is 75 cm² --/
theorem area_of_rectangle (s : SquareWithRectangle)
    (h_pt : s.pt_ratio = 1/3)
    (h_su : s.su_ratio = 1/3)
    (h_perimeter : rectangle_perimeter s = 40) :
    rectangle_area s = 75 := by
  sorry

end NUMINAMATH_CALUDE_area_of_rectangle_l862_86280


namespace NUMINAMATH_CALUDE_maisy_earnings_difference_l862_86258

/-- Represents Maisy's job details -/
structure Job where
  hours : ℕ
  wage : ℕ
  bonus : ℕ

/-- Calculates the weekly earnings for a job -/
def weekly_earnings (job : Job) : ℕ :=
  job.hours * job.wage + job.bonus

/-- Theorem: Maisy earns $15 more per week at her new job -/
theorem maisy_earnings_difference :
  let current_job : Job := ⟨8, 10, 0⟩
  let new_job : Job := ⟨4, 15, 35⟩
  weekly_earnings new_job - weekly_earnings current_job = 15 :=
by sorry

end NUMINAMATH_CALUDE_maisy_earnings_difference_l862_86258


namespace NUMINAMATH_CALUDE_maple_trees_planted_proof_l862_86263

/-- The number of maple trees planted in a park -/
def maple_trees_planted (initial : ℕ) (final : ℕ) : ℕ := final - initial

/-- Theorem stating that 11 maple trees were planted -/
theorem maple_trees_planted_proof :
  let initial_trees : ℕ := 53
  let final_trees : ℕ := 64
  maple_trees_planted initial_trees final_trees = 11 := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_planted_proof_l862_86263


namespace NUMINAMATH_CALUDE_product_zero_implications_l862_86233

theorem product_zero_implications (a b c : ℝ) : 
  (((a * b * c = 0) → (a = 0 ∨ b = 0 ∨ c = 0)) ∧
   ((a = 0 ∨ b = 0 ∨ c = 0) → (a * b * c = 0)) ∧
   ((a * b * c ≠ 0) → (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)) ∧
   ((a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → (a * b * c ≠ 0))) :=
by sorry

end NUMINAMATH_CALUDE_product_zero_implications_l862_86233


namespace NUMINAMATH_CALUDE_anna_ate_14_apples_l862_86269

def apples_tuesday : ℕ := 4

def apples_wednesday (tuesday : ℕ) : ℕ := 2 * tuesday

def apples_thursday (tuesday : ℕ) : ℕ := tuesday / 2

def total_apples (tuesday wednesday thursday : ℕ) : ℕ := 
  tuesday + wednesday + thursday

theorem anna_ate_14_apples : 
  total_apples apples_tuesday 
               (apples_wednesday apples_tuesday) 
               (apples_thursday apples_tuesday) = 14 := by
  sorry

end NUMINAMATH_CALUDE_anna_ate_14_apples_l862_86269


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l862_86279

/-- Given two lines l₁ and l₂ in the form x + ay = 1 and ax + y = 1 respectively,
    if they are parallel, then the distance between them is √2. -/
theorem parallel_lines_distance (a : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | x + a * y = 1}
  let l₂ := {(x, y) : ℝ × ℝ | a * x + y = 1}
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ → (x₂, y₂) ∈ l₂ → 
    (y₂ - y₁) / (x₂ - x₁) = (y₁ - y₂) / (x₁ - x₂)) →
  (∃ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ l₁ ∧ p₂ ∈ l₂ ∧ 
    Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l862_86279


namespace NUMINAMATH_CALUDE_exactly_two_successes_probability_l862_86203

/-- The probability of success in a single trial -/
def p : ℚ := 3/5

/-- The number of trials -/
def n : ℕ := 5

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The binomial probability formula -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1-p)^(n-k)

/-- The main theorem: probability of exactly 2 successes in 5 trials with p = 3/5 is 144/625 -/
theorem exactly_two_successes_probability :
  binomial_probability n k p = 144/625 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_successes_probability_l862_86203


namespace NUMINAMATH_CALUDE_pen_price_calculation_l862_86297

/-- Given the total number of pens purchased -/
def num_pens : ℕ := 30

/-- Given the total number of pencils purchased -/
def num_pencils : ℕ := 75

/-- Given the total cost of pens and pencils -/
def total_cost : ℝ := 750

/-- Given the average price of a pencil -/
def avg_price_pencil : ℝ := 2

/-- The average price of a pen -/
def avg_price_pen : ℝ := 20

theorem pen_price_calculation :
  (num_pens : ℝ) * avg_price_pen + (num_pencils : ℝ) * avg_price_pencil = total_cost :=
by sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l862_86297


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_minus_2_squared_l862_86252

theorem imaginary_part_of_i_minus_2_squared (i : ℂ) : 
  (i * i = -1) → Complex.im ((i - 2) ^ 2) = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_minus_2_squared_l862_86252


namespace NUMINAMATH_CALUDE_curve_properties_l862_86267

-- Define the curve
def curve (x y : ℝ) : Prop := x^3 + x*y + y^3 = 3

-- Define symmetry with respect to y = -x
def symmetric_about_neg_x (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ f (-y) (-x)

-- Define a point being on the curve
def point_on_curve (x y : ℝ) : Prop := curve x y

-- Define the concept of a curve approaching a line
def approaches_line (f : ℝ → ℝ → Prop) (m b : ℝ) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x y, f x y → (|x| > M ∨ |y| > M) → |y - (m*x + b)| < ε

theorem curve_properties :
  symmetric_about_neg_x curve ∧
  point_on_curve (Real.rpow 3 (1/3 : ℝ)) 0 ∧
  point_on_curve 1 1 ∧
  point_on_curve 0 (Real.rpow 3 (1/3 : ℝ)) ∧
  approaches_line curve (-1) 0 :=
sorry

end NUMINAMATH_CALUDE_curve_properties_l862_86267


namespace NUMINAMATH_CALUDE_zoo_pictures_l862_86247

/-- Represents the number of pictures Debby took at the zoo -/
def Z : ℕ := sorry

/-- The total number of pictures Debby initially took -/
def total_initial : ℕ := Z + 12

/-- The number of pictures Debby deleted -/
def deleted : ℕ := 14

/-- The number of pictures Debby has remaining -/
def remaining : ℕ := 22

theorem zoo_pictures : Z = 24 :=
  sorry

end NUMINAMATH_CALUDE_zoo_pictures_l862_86247


namespace NUMINAMATH_CALUDE_invisible_square_exists_l862_86240

/-- A point with integer coordinates is invisible if the gcd of its coordinates is greater than 1 -/
def invisible (p q : ℤ) : Prop := Nat.gcd p.natAbs q.natAbs > 1

/-- There exists a square with side length n*k where all integer coordinate points are invisible -/
theorem invisible_square_exists (n : ℕ) : ∃ k : ℕ, k ≥ 2 ∧ 
  ∀ p q : ℤ, 0 ≤ p ∧ p ≤ n * k ∧ 0 ≤ q ∧ q ≤ n * k → invisible p q :=
sorry

end NUMINAMATH_CALUDE_invisible_square_exists_l862_86240


namespace NUMINAMATH_CALUDE_f_402_equals_zero_l862_86238

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom period_condition : ∀ x, f (x + 4) - f x = 2 * f 2
axiom symmetry_condition : ∀ x, f (2 - x) = f x

-- Theorem to prove
theorem f_402_equals_zero : f 402 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_402_equals_zero_l862_86238


namespace NUMINAMATH_CALUDE_enclosed_area_is_five_twelfths_l862_86274

noncomputable def f (x : ℝ) : ℝ := x^(1/2)
noncomputable def g (x : ℝ) : ℝ := x^3

theorem enclosed_area_is_five_twelfths :
  ∫ x in (0)..(1), (f x - g x) = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_enclosed_area_is_five_twelfths_l862_86274


namespace NUMINAMATH_CALUDE_max_similar_triangles_five_points_l862_86226

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle formed by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- The set of all triangles formed by choosing 3 points from a set of 5 points -/
def allTriangles (points : Finset Point) : Finset Triangle := sorry

/-- The set of all similar triangles from a set of triangles -/
def similarTriangles (triangles : Finset Triangle) : Finset (Finset Triangle) := sorry

/-- The theorem stating that the maximum number of similar triangles from 5 points is 4 -/
theorem max_similar_triangles_five_points (points : Finset Point) :
  points.card = 5 →
  (similarTriangles (allTriangles points)).sup (λ s => s.card) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_similar_triangles_five_points_l862_86226


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l862_86245

/-- The average speed of a car given its distances traveled in two consecutive hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : 
  d1 = 90 → d2 = 40 → (d1 + d2) / 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l862_86245


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l862_86265

theorem quadratic_root_relation (p q : ℤ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x = 4*y) ∧ 
  (abs p < 100) ∧ (abs q < 100) ↔ 
  ((p = 5 ∨ p = -5) ∧ q = 4) ∨
  ((p = 10 ∨ p = -10) ∧ q = 16) ∨
  ((p = 15 ∨ p = -15) ∧ q = 36) ∨
  ((p = 20 ∨ p = -20) ∧ q = 64) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l862_86265


namespace NUMINAMATH_CALUDE_largest_common_divisor_problem_l862_86296

theorem largest_common_divisor_problem : Nat.gcd (69 - 5) (86 - 6) = 16 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_problem_l862_86296


namespace NUMINAMATH_CALUDE_unique_solution_l862_86234

/-- For every positive integer n, there exists a positive integer c_n
    such that a^n + b^n = c_n^(n+1) -/
def satisfies_condition (a b : ℕ+) : Prop :=
  ∀ n : ℕ+, ∃ c_n : ℕ+, (a : ℕ)^(n : ℕ) + (b : ℕ)^(n : ℕ) = (c_n : ℕ)^((n : ℕ) + 1)

/-- The only pair of positive integers (a,b) satisfying the condition is (2,2) -/
theorem unique_solution :
  ∀ a b : ℕ+, satisfies_condition a b ↔ a = 2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l862_86234


namespace NUMINAMATH_CALUDE_not_prime_a_l862_86205

theorem not_prime_a (a b : ℕ+) (h : ∃ k : ℤ, (5 * a^4 + a^2 : ℤ) = k * (b^4 + 3 * b^2 + 4)) : 
  ¬ Nat.Prime a.val := by
  sorry

end NUMINAMATH_CALUDE_not_prime_a_l862_86205


namespace NUMINAMATH_CALUDE_rectangle_length_l862_86243

/-- Given a rectangle with perimeter 42 and width 4, its length is 17. -/
theorem rectangle_length (P w l : ℝ) (h1 : P = 42) (h2 : w = 4) (h3 : P = 2 * (l + w)) : l = 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l862_86243


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_5_or_7_l862_86290

def four_digit_numbers : ℕ := 9000

def digits_without_5_or_7 : ℕ := 8

def first_digit_options : ℕ := 7

def numbers_without_5_or_7 : ℕ := first_digit_options * (digits_without_5_or_7 ^ 3)

theorem four_digit_numbers_with_5_or_7 :
  four_digit_numbers - numbers_without_5_or_7 = 5416 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_5_or_7_l862_86290


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l862_86230

/-- An arithmetic sequence with the given properties has a common difference of 2 -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) > a n) →  -- increasing sequence
  (a 1 + a 2 + a 3 = 12) →  -- sum of first three terms
  ((a 3)^2 = a 2 * (a 4 + 1)) →  -- geometric sequence condition
  (∃ d : ℝ, ∀ n, a (n + 1) - a n = d) →  -- arithmetic sequence
  (∃ d : ℝ, (∀ n, a (n + 1) - a n = d) ∧ d = 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l862_86230


namespace NUMINAMATH_CALUDE_ellen_croissants_l862_86211

/-- The price of a can of cola in pence -/
def cola_price : ℕ := sorry

/-- The price of a croissant in pence -/
def croissant_price : ℕ := sorry

/-- The total amount of money Ellen has in pence -/
def total_money : ℕ := sorry

/-- Assumption that Ellen can spend all her money on 6 cans of cola and 7 croissants -/
axiom combination1 : 6 * cola_price + 7 * croissant_price = total_money

/-- Assumption that Ellen can spend all her money on 8 cans of cola and 4 croissants -/
axiom combination2 : 8 * cola_price + 4 * croissant_price = total_money

/-- Theorem stating that Ellen can buy 16 croissants if she decides to buy only croissants -/
theorem ellen_croissants : total_money / croissant_price = 16 := by sorry

end NUMINAMATH_CALUDE_ellen_croissants_l862_86211


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_1_simplify_and_evaluate_2_l862_86285

-- Problem 1
theorem simplify_and_evaluate_1 : 2 * Real.sqrt 3 * 31.5 * 612 = 6 := by sorry

-- Problem 2
theorem simplify_and_evaluate_2 : 
  (Real.log 3 / Real.log 4 - Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 1/4 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_1_simplify_and_evaluate_2_l862_86285


namespace NUMINAMATH_CALUDE_triangle_height_sum_bound_l862_86257

/-- For a triangle with side lengths a ≤ b ≤ c, heights h_a, h_b, h_c,
    semiperimeter p, and circumradius R, the sum of heights is bounded. -/
theorem triangle_height_sum_bound (a b c h_a h_b h_c p R : ℝ) :
  a ≤ b → b ≤ c → a > 0 → b > 0 → c > 0 →
  p = (a + b + c) / 2 →
  R > 0 →
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a*c + c^2)) / (4 * p * R) := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_sum_bound_l862_86257


namespace NUMINAMATH_CALUDE_rice_bag_weight_qualification_l862_86223

def is_qualified (weight : ℝ) : Prop :=
  9.9 ≤ weight ∧ weight ≤ 10.1

theorem rice_bag_weight_qualification :
  is_qualified 10 ∧
  ¬ is_qualified 9.2 ∧
  ¬ is_qualified 10.2 ∧
  ¬ is_qualified 9.8 :=
by sorry

end NUMINAMATH_CALUDE_rice_bag_weight_qualification_l862_86223


namespace NUMINAMATH_CALUDE_special_property_implies_interval_l862_86218

/-- A positive integer n < 1000 has the property that 1/n is a repeating decimal
    of period 3 and 1/(n+6) is a repeating decimal of period 2 -/
def has_special_property (n : ℕ) : Prop :=
  n > 0 ∧ n < 1000 ∧
  ∃ (a b c : ℕ), (1 : ℚ) / n = (a * 100 + b * 10 + c : ℚ) / 999 ∧
  ∃ (x y : ℕ), (1 : ℚ) / (n + 6) = (x * 10 + y : ℚ) / 99

theorem special_property_implies_interval :
  ∀ n : ℕ, has_special_property n → n ∈ Set.Icc 1 250 :=
by
  sorry

end NUMINAMATH_CALUDE_special_property_implies_interval_l862_86218
