import Mathlib

namespace NUMINAMATH_CALUDE_fraction_subtraction_l3970_397032

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3970_397032


namespace NUMINAMATH_CALUDE_toy_cost_calculation_l3970_397006

/-- Represents the initial weekly cost price of a toy in Rupees -/
def initial_cost : ℝ := 1300

/-- Number of toys sold -/
def num_toys : ℕ := 18

/-- Discount rate applied to the toys -/
def discount_rate : ℝ := 0.1

/-- Total revenue from the sale in Rupees -/
def total_revenue : ℝ := 27300

theorem toy_cost_calculation :
  initial_cost * num_toys * (1 - discount_rate) = total_revenue - 3 * initial_cost := by
  sorry

#check toy_cost_calculation

end NUMINAMATH_CALUDE_toy_cost_calculation_l3970_397006


namespace NUMINAMATH_CALUDE_smallest_multiple_of_84_with_6_and_7_l3970_397016

def is_multiple_of_84 (n : ℕ) : Prop := n % 84 = 0

def contains_only_6_and_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 6 ∨ d = 7

theorem smallest_multiple_of_84_with_6_and_7 :
  (is_multiple_of_84 76776) ∧
  (contains_only_6_and_7 76776) ∧
  (∀ n : ℕ, n < 76776 → ¬(is_multiple_of_84 n ∧ contains_only_6_and_7 n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_84_with_6_and_7_l3970_397016


namespace NUMINAMATH_CALUDE_outermost_ring_count_9x9_l3970_397095

/-- Represents a square grid with alternating circles and rhombuses -/
structure AlternatingGrid (n : ℕ) where
  size : ℕ
  size_pos : size > 0
  is_square : ∃ k : ℕ, size = k * k

/-- The number of elements in the outermost ring of an AlternatingGrid -/
def outermost_ring_count (grid : AlternatingGrid n) : ℕ :=
  4 * (grid.size - 1)

/-- Theorem: The number of elements in the outermost ring of a 9x9 AlternatingGrid is 81 -/
theorem outermost_ring_count_9x9 :
  ∀ (grid : AlternatingGrid 9), grid.size = 9 → outermost_ring_count grid = 81 :=
by
  sorry


end NUMINAMATH_CALUDE_outermost_ring_count_9x9_l3970_397095


namespace NUMINAMATH_CALUDE_first_row_dots_l3970_397025

def green_dots_sequence (n : ℕ) : ℕ := 3 * n + 3

theorem first_row_dots : green_dots_sequence 0 = 3 := by sorry

end NUMINAMATH_CALUDE_first_row_dots_l3970_397025


namespace NUMINAMATH_CALUDE_incorrect_fraction_transformation_l3970_397099

theorem incorrect_fraction_transformation (a b : ℝ) (hb : b ≠ 0) :
  ¬(∀ (a b : ℝ), b ≠ 0 → |(-a)| / b = a / (-b)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_fraction_transformation_l3970_397099


namespace NUMINAMATH_CALUDE_square_plus_linear_plus_one_l3970_397085

theorem square_plus_linear_plus_one (a : ℝ) : 
  a^2 + a - 5 = 0 → a^2 + a + 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_linear_plus_one_l3970_397085


namespace NUMINAMATH_CALUDE_mix_g_weekly_amount_l3970_397045

/-- Calculates the weekly amount of Mix G birdseed needed for pigeons -/
def weekly_mix_g_amount (num_pigeons : ℕ) (daily_consumption : ℕ) (days : ℕ) : ℕ :=
  num_pigeons * daily_consumption * days

/-- Theorem stating that the weekly amount of Mix G birdseed needed is 168 grams -/
theorem mix_g_weekly_amount :
  weekly_mix_g_amount 6 4 7 = 168 :=
by sorry

end NUMINAMATH_CALUDE_mix_g_weekly_amount_l3970_397045


namespace NUMINAMATH_CALUDE_john_reading_days_l3970_397088

/-- Given that John reads 4 books a day and 48 books in 6 weeks, prove that he reads on 2 days per week. -/
theorem john_reading_days 
  (books_per_day : ℕ) 
  (total_books : ℕ) 
  (total_weeks : ℕ) 
  (h1 : books_per_day = 4) 
  (h2 : total_books = 48) 
  (h3 : total_weeks = 6) : 
  (total_books / books_per_day) / total_weeks = 2 :=
by sorry

end NUMINAMATH_CALUDE_john_reading_days_l3970_397088


namespace NUMINAMATH_CALUDE_amelia_dinner_l3970_397077

def dinner_problem (initial_amount : ℝ) (first_course : ℝ) (second_course_extra : ℝ) (dessert_percentage : ℝ) : Prop :=
  let second_course := first_course + second_course_extra
  let dessert := dessert_percentage * second_course
  let total_cost := first_course + second_course + dessert
  let money_left := initial_amount - total_cost
  money_left = 20

theorem amelia_dinner :
  dinner_problem 60 15 5 0.25 := by
  sorry

end NUMINAMATH_CALUDE_amelia_dinner_l3970_397077


namespace NUMINAMATH_CALUDE_a_profit_share_l3970_397050

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_profit_share (initial_a initial_b : ℕ) (withdraw_a advance_b : ℕ) (months : ℕ) (total_profit : ℕ) : ℕ :=
  let investment_months_a := initial_a * months + (initial_a - withdraw_a) * (12 - months)
  let investment_months_b := initial_b * months + (initial_b + advance_b) * (12 - months)
  let total_investment_months := investment_months_a + investment_months_b
  (investment_months_a * total_profit) / total_investment_months

/-- Theorem stating that A's share of the profit is 357 given the problem conditions --/
theorem a_profit_share :
  calculate_profit_share 6000 4000 1000 1000 8 630 = 357 :=
by
  sorry

#eval calculate_profit_share 6000 4000 1000 1000 8 630

end NUMINAMATH_CALUDE_a_profit_share_l3970_397050


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3970_397026

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {y | ∃ x, y = 2 * Real.sin x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {y | 0 < y ∧ y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3970_397026


namespace NUMINAMATH_CALUDE_units_digit_of_large_power_l3970_397019

theorem units_digit_of_large_power (n : ℕ) : n > 0 → (7^(8^5) : ℕ) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_large_power_l3970_397019


namespace NUMINAMATH_CALUDE_fewer_onions_l3970_397062

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

theorem fewer_onions : (tomatoes + corn) - onions = 5200 := by
  sorry

end NUMINAMATH_CALUDE_fewer_onions_l3970_397062


namespace NUMINAMATH_CALUDE_largest_integer_with_conditions_l3970_397058

theorem largest_integer_with_conditions : 
  ∃ (n : ℕ), n = 243 ∧ 
  (∀ m : ℕ, (200 < m ∧ m < 250 ∧ ∃ k : ℕ, 12 * m = k^2) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_conditions_l3970_397058


namespace NUMINAMATH_CALUDE_fraction_equality_l3970_397060

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 3 / 5) 
  (h2 : r / t = 8 / 9) : 
  (3 * m^2 * r - n * t^2) / (5 * n * t^2 - 9 * m^2 * r) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3970_397060


namespace NUMINAMATH_CALUDE_sum_60_is_negative_120_l3970_397030

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_20 : (20 : ℚ) / 2 * (2 * a + 19 * d) = 200
  sum_50 : (50 : ℚ) / 2 * (2 * a + 49 * d) = 50

/-- The sum of the first 60 terms of the arithmetic progression is -120 -/
theorem sum_60_is_negative_120 (ap : ArithmeticProgression) :
  (60 : ℚ) / 2 * (2 * ap.a + 59 * ap.d) = -120 := by
  sorry

end NUMINAMATH_CALUDE_sum_60_is_negative_120_l3970_397030


namespace NUMINAMATH_CALUDE_max_value_bound_max_value_achievable_l3970_397089

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def max_value (u v w : E) : ℝ :=
  ‖u - 3 • v‖^2 + ‖v - 3 • w‖^2 + ‖w - 3 • u‖^2

theorem max_value_bound (u v w : E) 
  (hu : ‖u‖ = 2) (hv : ‖v‖ = 3) (hw : ‖w‖ = 4) :
  max_value u v w ≤ 377 :=
by sorry

theorem max_value_achievable :
  ∃ (u v w : E), ‖u‖ = 2 ∧ ‖v‖ = 3 ∧ ‖w‖ = 4 ∧ max_value u v w = 377 :=
by sorry

end NUMINAMATH_CALUDE_max_value_bound_max_value_achievable_l3970_397089


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3970_397092

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h1 : ∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1 < x ∧ x < 2) :
  ∀ x : ℝ, 2*x^2 + b*x + a < 0 ↔ -1 < x ∧ x < 1/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3970_397092


namespace NUMINAMATH_CALUDE_angle_C_measure_l3970_397022

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem statement
theorem angle_C_measure (abc : Triangle) (h1 : abc.A = 50) (h2 : abc.B = 60) : abc.C = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l3970_397022


namespace NUMINAMATH_CALUDE_hexagon_tileable_with_squares_l3970_397071

-- Define a hexagon type
structure Hexagon :=
  (A B C D E F : ℝ × ℝ)

-- Define the property of being convex
def is_convex (h : Hexagon) : Prop := sorry

-- Define the property of being inscribed
def is_inscribed (h : Hexagon) : Prop := sorry

-- Define perpendicularity of segments
def perpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

-- Define equality of segments
def segments_equal (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

-- Define the property of being tileable with squares
def tileable_with_squares (h : Hexagon) : Prop := sorry

theorem hexagon_tileable_with_squares (h : Hexagon) 
  (convex : is_convex h)
  (inscribed : is_inscribed h)
  (perp_AD_CE : perpendicular h.A h.D h.C h.E)
  (eq_AD_CE : segments_equal h.A h.D h.C h.E)
  (perp_BE_AC : perpendicular h.B h.E h.A h.C)
  (eq_BE_AC : segments_equal h.B h.E h.A h.C)
  (perp_CF_EA : perpendicular h.C h.F h.E h.A)
  (eq_CF_EA : segments_equal h.C h.F h.E h.A) :
  tileable_with_squares h := by
  sorry

end NUMINAMATH_CALUDE_hexagon_tileable_with_squares_l3970_397071


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3970_397069

theorem quadratic_equation_properties (a b c : ℝ) (ha : a ≠ 0) :
  -- Statement 1
  (a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
  -- Statement 2
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + c = 0 ∧ a*y^2 + c = 0 →
    ∃ u v : ℝ, u ≠ v ∧ a*u^2 + b*u + c = 0 ∧ a*v^2 + b*v + c = 0) ∧
  -- Statement 4
  ∃ m n : ℝ, m ≠ n ∧ a*m^2 + b*m + c = a*n^2 + b*n + c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3970_397069


namespace NUMINAMATH_CALUDE_certain_number_proof_l3970_397011

theorem certain_number_proof : 
  ∃ x : ℚ, (x + 720) / 125 = 7392 / 462 ∧ x = 1280 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3970_397011


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l3970_397035

theorem ratio_w_to_y (w x y z : ℝ) 
  (hw_x : w / x = 5 / 2)
  (hy_z : y / z = 2 / 3)
  (hx_z : x / z = 10) :
  w / y = 37.5 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l3970_397035


namespace NUMINAMATH_CALUDE_nine_b_value_l3970_397086

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) : 9 * b = 216 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nine_b_value_l3970_397086


namespace NUMINAMATH_CALUDE_inequality_proof_l3970_397056

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (x / (y^2 + z)) + (y / (z^2 + x)) + (z / (x^2 + y)) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3970_397056


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l3970_397051

/-- The number of ways to put n indistinguishable balls into k distinguishable boxes -/
def ball_distribution (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 28 ways to put 6 indistinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes : ball_distribution 6 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l3970_397051


namespace NUMINAMATH_CALUDE_school_fee_calculation_l3970_397036

/-- Represents the amount of money given by Luke's mother -/
def mother_contribution : ℕ :=
  50 + 2 * 20 + 3 * 10

/-- Represents the amount of money given by Luke's father -/
def father_contribution : ℕ :=
  4 * 50 + 20 + 10

/-- Represents the total school fee -/
def school_fee : ℕ :=
  mother_contribution + father_contribution

theorem school_fee_calculation :
  school_fee = 350 :=
by sorry

end NUMINAMATH_CALUDE_school_fee_calculation_l3970_397036


namespace NUMINAMATH_CALUDE_alpha_value_l3970_397083

theorem alpha_value (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan (α + β) = 3 →
  Real.tan β = 1/2 →
  α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l3970_397083


namespace NUMINAMATH_CALUDE_existence_and_uniqueness_l3970_397076

open Real

/-- The differential equation y' = y - x^2 + 2x - 2 -/
def diff_eq (x y : ℝ) : ℝ := y - x^2 + 2*x - 2

/-- A solution to the differential equation -/
def is_solution (f : ℝ → ℝ) : Prop :=
  ∀ x, (deriv f) x = diff_eq x (f x)

theorem existence_and_uniqueness :
  ∀ (x₀ y₀ : ℝ), ∃! f : ℝ → ℝ,
    is_solution f ∧ f x₀ = y₀ :=
sorry

end NUMINAMATH_CALUDE_existence_and_uniqueness_l3970_397076


namespace NUMINAMATH_CALUDE_cookout_buns_needed_l3970_397044

/-- Calculates the number of packs of buns needed for a cookout --/
def buns_needed (total_guests : ℕ) (burgers_per_guest : ℕ) (no_meat_guests : ℕ) (no_bread_guests : ℕ) (buns_per_pack : ℕ) : ℕ :=
  let guests_eating_burgers := total_guests - no_meat_guests
  let total_burgers := guests_eating_burgers * burgers_per_guest
  let buns_needed := total_burgers - (no_bread_guests * burgers_per_guest)
  (buns_needed + buns_per_pack - 1) / buns_per_pack

theorem cookout_buns_needed :
  buns_needed 10 3 1 1 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookout_buns_needed_l3970_397044


namespace NUMINAMATH_CALUDE_removed_number_for_mean_l3970_397053

theorem removed_number_for_mean (n : ℕ) (h : n ≥ 9) :
  ∃ x : ℕ, x ≤ n ∧ 
    (((n * (n + 1)) / 2 - x) / (n - 1) : ℚ) = 19/4 →
    x = 7 :=
  sorry

end NUMINAMATH_CALUDE_removed_number_for_mean_l3970_397053


namespace NUMINAMATH_CALUDE_intersection_M_N_l3970_397021

def M : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3970_397021


namespace NUMINAMATH_CALUDE_mary_lamb_count_l3970_397093

/-- The number of lambs Mary has after a series of events --/
def final_lamb_count (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) 
                     (lambs_traded : ℕ) (extra_lambs_found : ℕ) : ℕ :=
  initial_lambs + lambs_with_babies * babies_per_lamb - lambs_traded + extra_lambs_found

/-- Theorem stating that Mary ends up with 14 lambs --/
theorem mary_lamb_count : 
  final_lamb_count 6 2 2 3 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_mary_lamb_count_l3970_397093


namespace NUMINAMATH_CALUDE_min_magnitude_a_plus_tb_collinear_a_minus_tb_c_l3970_397079

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (-3, 2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (3, -1)

/-- The squared magnitude of a vector -/
def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

/-- Theorem: Minimum value of |a+tb| and its corresponding t -/
theorem min_magnitude_a_plus_tb :
  (∃ t : ℝ, magnitude_squared (a.1 + t * b.1, a.2 + t * b.2) = (7 * Real.sqrt 5 / 5)^2) ∧
  (∀ t : ℝ, magnitude_squared (a.1 + t * b.1, a.2 + t * b.2) ≥ (7 * Real.sqrt 5 / 5)^2) ∧
  (magnitude_squared (a.1 + 4/5 * b.1, a.2 + 4/5 * b.2) = (7 * Real.sqrt 5 / 5)^2) :=
sorry

/-- Theorem: Value of t when a-tb is collinear with c -/
theorem collinear_a_minus_tb_c :
  ∃ t : ℝ, t = 3/5 ∧ (a.1 - t * b.1) * c.2 = (a.2 - t * b.2) * c.1 :=
sorry

end NUMINAMATH_CALUDE_min_magnitude_a_plus_tb_collinear_a_minus_tb_c_l3970_397079


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3970_397028

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (m - 1) * x^2 - 2*x - 1 = 0 ∧ (m - 1) * y^2 - 2*y - 1 = 0) ↔ 
  (m ≥ 0 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3970_397028


namespace NUMINAMATH_CALUDE_least_perimeter_l3970_397057

/-- Represents a triangle with two known sides and an integral third side -/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  is_triangle : side1 + side2 > side3 ∧ side1 + side3 > side2 ∧ side2 + side3 > side1

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℕ := t.side1 + t.side2 + t.side3

/-- The specific triangle from the problem -/
def problem_triangle : Triangle → Prop
  | t => t.side1 = 24 ∧ t.side2 = 51

theorem least_perimeter :
  ∀ t : Triangle, problem_triangle t →
  ∀ u : Triangle, problem_triangle u →
  perimeter t ≥ 103 ∧ (∃ v : Triangle, problem_triangle v ∧ perimeter v = 103) :=
by sorry

end NUMINAMATH_CALUDE_least_perimeter_l3970_397057


namespace NUMINAMATH_CALUDE_min_value_theorem_l3970_397007

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 4) :
  ((x + 1) * (2*y + 1)) / (x * y) ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3970_397007


namespace NUMINAMATH_CALUDE_right_tangential_trapezoid_shorter_leg_l3970_397068

/-- In a right tangential trapezoid, the shorter leg equals 2ac/(a+c) where a and c are the lengths of the bases. -/
theorem right_tangential_trapezoid_shorter_leg
  (a c d : ℝ)
  (h_positive : a > 0 ∧ c > 0 ∧ d > 0)
  (h_right_tangential : d^2 + (a - c)^2 = (a + c - d)^2)
  (h_shorter_leg : d ≤ a + c - d) :
  d = 2 * a * c / (a + c) := by
sorry

end NUMINAMATH_CALUDE_right_tangential_trapezoid_shorter_leg_l3970_397068


namespace NUMINAMATH_CALUDE_distance_to_school_is_correct_l3970_397020

/-- The distance from Xiaohong's home to school in meters -/
def distance_to_school : ℝ := 2720

/-- The distance dad drove Xiaohong from school in meters -/
def distance_driven : ℝ := 1000

/-- The total travel time (drive + walk) in minutes -/
def total_travel_time : ℝ := 22.5

/-- The time it takes to bike from home to school in minutes -/
def biking_time : ℝ := 40

/-- Xiaohong's walking speed in meters per minute -/
def walking_speed : ℝ := 80

/-- The difference between dad's driving speed and biking speed in meters per minute -/
def speed_difference : ℝ := 800

theorem distance_to_school_is_correct :
  ∃ (driving_speed : ℝ),
    driving_speed > 0 ∧
    distance_to_school / (driving_speed - speed_difference) = biking_time ∧
    (distance_driven / driving_speed) + ((distance_to_school - distance_driven) / walking_speed) = total_travel_time :=
by sorry

end NUMINAMATH_CALUDE_distance_to_school_is_correct_l3970_397020


namespace NUMINAMATH_CALUDE_article_price_reduction_l3970_397034

/-- Proves that given an article with an original cost of 50, sold at a 25% profit,
    if the selling price is reduced by 10.50 and the profit becomes 30%,
    then the reduction in the buying price is 20%. -/
theorem article_price_reduction (original_cost : ℝ) (original_profit_percent : ℝ)
  (price_reduction : ℝ) (new_profit_percent : ℝ) :
  original_cost = 50 →
  original_profit_percent = 25 →
  price_reduction = 10.50 →
  new_profit_percent = 30 →
  ∃ (buying_price_reduction : ℝ),
    buying_price_reduction = 20 ∧
    (original_cost * (1 + original_profit_percent / 100) - price_reduction) =
    (original_cost * (1 - buying_price_reduction / 100)) * (1 + new_profit_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_article_price_reduction_l3970_397034


namespace NUMINAMATH_CALUDE_westville_summer_retreat_soccer_percentage_l3970_397084

theorem westville_summer_retreat_soccer_percentage 
  (total : ℝ) 
  (soccer_percentage : ℝ) 
  (swim_percentage : ℝ) 
  (soccer_and_swim_percentage : ℝ) 
  (basketball_percentage : ℝ) 
  (basketball_soccer_no_swim_percentage : ℝ) 
  (h1 : soccer_percentage = 0.7) 
  (h2 : swim_percentage = 0.5) 
  (h3 : soccer_and_swim_percentage = 0.3 * soccer_percentage) 
  (h4 : basketball_percentage = 0.2) 
  (h5 : basketball_soccer_no_swim_percentage = 0.25 * basketball_percentage) : 
  (soccer_percentage * total - soccer_and_swim_percentage * total - basketball_soccer_no_swim_percentage * total) / 
  ((1 - swim_percentage) * total) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_westville_summer_retreat_soccer_percentage_l3970_397084


namespace NUMINAMATH_CALUDE_special_factorization_of_630_l3970_397087

theorem special_factorization_of_630 : ∃ (a b x y z : ℕ), 
  (a + 1 = b) ∧ 
  (x + 1 = y) ∧ 
  (y + 1 = z) ∧ 
  (a * b = 630) ∧ 
  (x * y * z = 630) ∧ 
  (a + b + x + y + z = 75) := by
  sorry

end NUMINAMATH_CALUDE_special_factorization_of_630_l3970_397087


namespace NUMINAMATH_CALUDE_problem_statement_l3970_397001

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : a * b = 1) : 
  (a + b ≥ 2) ∧ (a^3 + b^3 ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3970_397001


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3970_397043

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3970_397043


namespace NUMINAMATH_CALUDE_train_schedule_l3970_397070

theorem train_schedule (x y z : ℕ) : 
  x < 24 → y < 24 → z < 24 →
  (60 * y + z) - (60 * x + y) = 60 * z + x →
  x = 0 ∨ x = 12 := by
sorry

end NUMINAMATH_CALUDE_train_schedule_l3970_397070


namespace NUMINAMATH_CALUDE_crayons_per_friend_l3970_397037

def total_crayons : ℕ := 210
def num_friends : ℕ := 30

theorem crayons_per_friend :
  total_crayons / num_friends = 7 :=
by sorry

end NUMINAMATH_CALUDE_crayons_per_friend_l3970_397037


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l3970_397096

/-- A quadratic function with specific properties -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: For a quadratic function p(x) with axis of symmetry at x = 8.5 and p(-1) = -4, p(18) = -4 -/
theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c (17 - x) = p a b c x) →  -- axis of symmetry at x = 8.5
  p a b c (-1) = -4 →
  p a b c 18 = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l3970_397096


namespace NUMINAMATH_CALUDE_total_amount_is_105_l3970_397073

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : ShareDistribution) : Prop :=
  s.y = 0.45 * s.x ∧ s.z = 0.30 * s.x ∧ s.y = 27

/-- The theorem to prove -/
theorem total_amount_is_105 (s : ShareDistribution) :
  problem_conditions s → s.x + s.y + s.z = 105 := by sorry

end NUMINAMATH_CALUDE_total_amount_is_105_l3970_397073


namespace NUMINAMATH_CALUDE_inspection_ratio_l3970_397074

theorem inspection_ratio (j n : ℝ) (hj : j > 0) (hn : n > 0) : 
  0.005 * j + 0.007 * n = 0.0075 * (j + n) → n / j = 5 := by sorry

end NUMINAMATH_CALUDE_inspection_ratio_l3970_397074


namespace NUMINAMATH_CALUDE_stating_auntie_em_parking_probability_l3970_397075

/-- The number of parking spaces in the lot -/
def total_spaces : ℕ := 20

/-- The number of cars that arrive before Auntie Em -/
def cars_before : ℕ := 15

/-- The number of spaces Auntie Em's SUV requires -/
def suv_spaces : ℕ := 2

/-- The probability that Auntie Em can park her SUV -/
def prob_auntie_em_can_park : ℚ := 232 / 323

/-- 
Theorem stating that the probability of Auntie Em being able to park her SUV
is equal to 232/323, given the conditions of the parking lot problem.
-/
theorem auntie_em_parking_probability :
  let remaining_spaces := total_spaces - cars_before
  let total_arrangements := Nat.choose total_spaces cars_before
  let unfavorable_arrangements := Nat.choose (remaining_spaces + cars_before - 1) (remaining_spaces - 1)
  (1 : ℚ) - (unfavorable_arrangements : ℚ) / (total_arrangements : ℚ) = prob_auntie_em_can_park :=
by sorry

end NUMINAMATH_CALUDE_stating_auntie_em_parking_probability_l3970_397075


namespace NUMINAMATH_CALUDE_solution_of_equation_l3970_397009

theorem solution_of_equation : ∃! x : ℝ, (3 / (x - 2) - 1 = 0) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l3970_397009


namespace NUMINAMATH_CALUDE_certain_amount_problem_l3970_397049

theorem certain_amount_problem (x y : ℝ) : 
  x = 7 → x + y = 15 → 5 * y - 3 * x = 19 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_problem_l3970_397049


namespace NUMINAMATH_CALUDE_rectangle_area_l3970_397046

/-- Given a rectangle with perimeter 24 and one side length x (x > 0),
    prove that its area y is equal to (12-x)x -/
theorem rectangle_area (x : ℝ) (hx : x > 0) : 
  let perimeter : ℝ := 24
  let y : ℝ := x * (perimeter / 2 - x)
  y = (12 - x) * x :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3970_397046


namespace NUMINAMATH_CALUDE_initial_fee_correct_l3970_397033

/-- The initial fee for Jim's taxi service -/
def initial_fee : ℝ := 2.25

/-- The charge per 2/5 mile segment -/
def charge_per_segment : ℝ := 0.35

/-- The length of a trip in miles -/
def trip_length : ℝ := 3.6

/-- The total charge for the trip -/
def total_charge : ℝ := 5.4

/-- Theorem stating that the initial fee is correct given the conditions -/
theorem initial_fee_correct : 
  initial_fee + (trip_length / (2/5) * charge_per_segment) = total_charge :=
by sorry

end NUMINAMATH_CALUDE_initial_fee_correct_l3970_397033


namespace NUMINAMATH_CALUDE_problem_solution_l3970_397002

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- State the theorem
theorem problem_solution :
  ∀ m : ℝ,
  (∀ x : ℝ, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-1) 1) →
  (m = 1 ∧
   {x : ℝ | |x + 1| + |x - 2| > 4 * m} = {x : ℝ | x < -3/2 ∨ x > 5/2}) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3970_397002


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l3970_397023

/-- The number of intern teachers --/
def num_teachers : ℕ := 5

/-- The number of classes --/
def num_classes : ℕ := 3

/-- The minimum number of teachers per class --/
def min_teachers_per_class : ℕ := 1

/-- The maximum number of teachers per class --/
def max_teachers_per_class : ℕ := 2

/-- A function that calculates the number of ways to allocate teachers to classes --/
def allocation_schemes (n_teachers : ℕ) (n_classes : ℕ) (min_per_class : ℕ) (max_per_class : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of allocation schemes is 90 --/
theorem allocation_schemes_count :
  allocation_schemes num_teachers num_classes min_teachers_per_class max_teachers_per_class = 90 :=
sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l3970_397023


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l3970_397014

def triangle_vertex (y : ℝ) : ℝ × ℝ := (0, y)

theorem triangle_area_theorem (y : ℝ) (h1 : y < 0) :
  let v1 : ℝ × ℝ := (8, 6)
  let v2 : ℝ × ℝ := (0, 0)
  let v3 : ℝ × ℝ := triangle_vertex y
  let area : ℝ := (1/2) * abs (v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2 - v1.2 * v2.1 - v2.2 * v3.1 - v3.2 * v1.1)
  area = 24 → y = -4.8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l3970_397014


namespace NUMINAMATH_CALUDE_f_composition_at_one_l3970_397059

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/2) * x - 1 else 2^x

theorem f_composition_at_one :
  f (f 1) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_at_one_l3970_397059


namespace NUMINAMATH_CALUDE_exponent_of_nine_in_nine_to_seven_l3970_397015

theorem exponent_of_nine_in_nine_to_seven (h : ∀ y : ℕ, y > 14 → ¬(3^y ∣ 9^7)) :
  ∃ n : ℕ, 9^7 = 9^n ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_exponent_of_nine_in_nine_to_seven_l3970_397015


namespace NUMINAMATH_CALUDE_password_factorization_l3970_397080

theorem password_factorization (a b c d : ℝ) :
  (a^2 - b^2) * c^2 - (a^2 - b^2) * d^2 = (a + b) * (a - b) * (c + d) * (c - d) := by
  sorry

end NUMINAMATH_CALUDE_password_factorization_l3970_397080


namespace NUMINAMATH_CALUDE_jogger_distance_l3970_397005

/-- Proves that given a jogger who jogs at 12 km/hr, if jogging at 20 km/hr would result in 15 km 
    more distance covered, then the actual distance jogged is 22.5 km. -/
theorem jogger_distance (actual_speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  actual_speed = 12 →
  faster_speed = 20 →
  faster_speed * (extra_distance / (faster_speed - actual_speed)) = 
    actual_speed * (extra_distance / (faster_speed - actual_speed)) + extra_distance →
  extra_distance = 15 →
  actual_speed * (extra_distance / (faster_speed - actual_speed)) = 22.5 :=
by sorry


end NUMINAMATH_CALUDE_jogger_distance_l3970_397005


namespace NUMINAMATH_CALUDE_matrix_product_l3970_397065

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 4; 3, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, -7; 2, 3]

theorem matrix_product :
  A * B = !![8, 5; -4, -27] := by sorry

end NUMINAMATH_CALUDE_matrix_product_l3970_397065


namespace NUMINAMATH_CALUDE_max_value_of_f_l3970_397067

/-- The quadratic function f(x) = -2x^2 + 9 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 9

/-- Theorem: The maximum value of f(x) = -2x^2 + 9 is 9 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3970_397067


namespace NUMINAMATH_CALUDE_underground_ticket_cost_l3970_397094

/-- The cost of one ticket to the underground. -/
def ticket_cost (tickets_per_minute : ℕ) (total_minutes : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (tickets_per_minute * total_minutes)

/-- Theorem stating that the cost of one ticket is $3. -/
theorem underground_ticket_cost :
  ticket_cost 5 6 90 = 3 := by
  sorry

end NUMINAMATH_CALUDE_underground_ticket_cost_l3970_397094


namespace NUMINAMATH_CALUDE_survey_sample_size_l3970_397041

/-- Represents a survey with a given population size and number of selected participants. -/
structure Survey where
  population_size : ℕ
  selected_participants : ℕ

/-- Calculates the sample size of a given survey. -/
def sample_size (s : Survey) : ℕ := s.selected_participants

/-- Theorem stating that for a survey with 4000 students and 500 randomly selected,
    the sample size is 500. -/
theorem survey_sample_size :
  let s : Survey := { population_size := 4000, selected_participants := 500 }
  sample_size s = 500 := by sorry

end NUMINAMATH_CALUDE_survey_sample_size_l3970_397041


namespace NUMINAMATH_CALUDE_f_properties_l3970_397066

noncomputable def f (x : ℝ) := Real.exp x * Real.cos x - x

theorem f_properties :
  let a := 0
  let b := Real.pi / 2
  ∃ (tangent_line : ℝ → ℝ),
    (∀ x, tangent_line x = 1) ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f a) ∧
    (∀ x ∈ Set.Icc a b, f b ≤ f x) ∧
    f a = 1 ∧
    f b = -Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3970_397066


namespace NUMINAMATH_CALUDE_blue_balls_unchanged_jungkook_blue_balls_l3970_397082

/-- Represents the number of balls of each color Jungkook has -/
structure BallCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Jungkook's initial ball count -/
def initial_count : BallCount :=
  { red := 5, blue := 4, yellow := 3 }

/-- Yoon-gi gives Jungkook a yellow ball -/
def give_yellow_ball (count : BallCount) : BallCount :=
  { count with yellow := count.yellow + 1 }

/-- The number of blue balls remains unchanged after receiving a yellow ball -/
theorem blue_balls_unchanged (count : BallCount) :
  (give_yellow_ball count).blue = count.blue :=
by sorry

/-- Jungkook has 4 blue balls after receiving a yellow ball from Yoon-gi -/
theorem jungkook_blue_balls :
  (give_yellow_ball initial_count).blue = 4 :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_unchanged_jungkook_blue_balls_l3970_397082


namespace NUMINAMATH_CALUDE_min_value_inequality_l3970_397098

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + b / a) * (1 + 4 * a / b) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3970_397098


namespace NUMINAMATH_CALUDE_total_amount_is_117_l3970_397004

/-- Represents the distribution of money among three parties -/
structure Distribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the total amount distributed -/
def total_amount (d : Distribution) : ℝ := d.x + d.y + d.z

/-- Theorem: Given the conditions, the total amount is 117 rupees -/
theorem total_amount_is_117 (d : Distribution) 
  (h1 : d.y = 27)  -- y's share is 27 rupees
  (h2 : d.y = 0.45 * d.x)  -- y gets 45 paisa for each rupee x gets
  (h3 : d.z = 0.50 * d.x)  -- z gets 50 paisa for each rupee x gets
  : total_amount d = 117 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_117_l3970_397004


namespace NUMINAMATH_CALUDE_danjiangkou_tourists_scientific_notation_l3970_397055

/-- Converts a positive integer to scientific notation -/
def to_scientific_notation (n : ℕ) : ℚ × ℤ :=
  sorry

theorem danjiangkou_tourists_scientific_notation :
  to_scientific_notation 456000 = (4.56, 5) :=
sorry

end NUMINAMATH_CALUDE_danjiangkou_tourists_scientific_notation_l3970_397055


namespace NUMINAMATH_CALUDE_sum_of_digits_divisible_by_nine_l3970_397090

/-- The sum of digits in a number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The concatenation of numbers from 1 to n -/
def concatenateNumbers (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all digits in the concatenation of numbers from 1 to 2015 is divisible by 9 -/
theorem sum_of_digits_divisible_by_nine :
  ∃ k : ℕ, sumOfDigits (concatenateNumbers 2015) = 9 * k := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_divisible_by_nine_l3970_397090


namespace NUMINAMATH_CALUDE_third_circle_radius_l3970_397048

/-- Given two externally tangent circles and a third circle tangent to both and their common external tangent, prove the radius of the third circle --/
theorem third_circle_radius (r1 r2 r3 : ℝ) : 
  r1 = 1 →                            -- radius of circle A
  r2 = 4 →                            -- radius of circle B
  (r1 + r2)^2 = r1^2 + r2^2 + 6*r1*r2 → -- circles A and B are externally tangent
  (r1 + r3)^2 = (r1 - r3)^2 + 4*r3 →    -- circle with radius r3 is tangent to circle A
  (r2 + r3)^2 = (r2 - r3)^2 + 16*r3 →   -- circle with radius r3 is tangent to circle B
  r3 = 4/9 :=                           -- radius of the third circle
by sorry

end NUMINAMATH_CALUDE_third_circle_radius_l3970_397048


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l3970_397027

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 11 = 16 →
  a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l3970_397027


namespace NUMINAMATH_CALUDE_weight_per_rep_l3970_397029

-- Define the given conditions
def reps_per_set : ℕ := 10
def num_sets : ℕ := 3
def total_weight : ℕ := 450

-- Define the theorem to prove
theorem weight_per_rep :
  total_weight / (reps_per_set * num_sets) = 15 := by
  sorry

end NUMINAMATH_CALUDE_weight_per_rep_l3970_397029


namespace NUMINAMATH_CALUDE_four_digit_number_properties_l3970_397010

/-- P function for a four-digit number -/
def P (x : ℕ) : ℤ :=
  let y := (x % 10) * 1000 + x / 10
  (x - y) / 9

/-- Check if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k * k

/-- Definition of s -/
def s (a b : ℕ) : ℕ := 1100 + 20 * a + b

/-- Definition of t -/
def t (a b : ℕ) : ℕ := b * 1000 + a * 100 + 23

/-- Main theorem -/
theorem four_digit_number_properties :
  (P 5324 = 88) ∧
  (∀ a b : ℕ, 1 ≤ a → a ≤ 4 → 1 ≤ b → b ≤ 9 →
    (∃ min_pt : ℤ, min_pt = -161 ∧
      is_perfect_square (P (t a b) - P (s a b) - a - b) ∧
      (∀ a' b' : ℕ, 1 ≤ a' → a' ≤ 4 → 1 ≤ b' → b' ≤ 9 →
        is_perfect_square (P (t a' b') - P (s a' b') - a' - b') →
        P (t a' b') ≥ min_pt))) :=
sorry

end NUMINAMATH_CALUDE_four_digit_number_properties_l3970_397010


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3970_397063

/-- Given a quadratic equation x^2 + px + q = 0 with roots p and q, 
    the product pq is either 0 or -2 -/
theorem quadratic_roots_product (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) → 
  pq = 0 ∨ pq = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3970_397063


namespace NUMINAMATH_CALUDE_unique_ranking_l3970_397031

-- Define the set of students
inductive Student : Type
  | A | B | C | D | E

-- Define the ranking type
def Ranking := Student → Fin 5

-- Define the guesses made by each student
def guesses (r : Ranking) : Prop :=
  (r Student.B = 2 ∨ r Student.C = 4) ∧
  (r Student.E = 3 ∨ r Student.D = 4) ∧
  (r Student.A = 0 ∨ r Student.E = 3) ∧
  (r Student.C = 0 ∨ r Student.D = 1) ∧
  (r Student.A = 2 ∨ r Student.D = 3)

-- Define the condition that each ranking was guessed correctly by someone
def eachRankingGuessedCorrectly (r : Ranking) : Prop :=
  ∀ s : Student, ∃ g : Student, 
    (g = Student.A ∧ (r s = 2 ∨ r s = 4)) ∨
    (g = Student.B ∧ (r s = 3 ∨ r s = 4)) ∨
    (g = Student.C ∧ (r s = 0 ∨ r s = 3)) ∨
    (g = Student.D ∧ (r s = 0 ∨ r s = 1)) ∨
    (g = Student.E ∧ (r s = 2 ∨ r s = 3))

-- The theorem to prove
theorem unique_ranking : 
  ∃! r : Ranking, guesses r ∧ eachRankingGuessedCorrectly r ∧
    r Student.A = 0 ∧ r Student.D = 1 ∧ r Student.B = 2 ∧ 
    r Student.E = 3 ∧ r Student.C = 4 :=
  sorry

end NUMINAMATH_CALUDE_unique_ranking_l3970_397031


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l3970_397012

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_products_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l3970_397012


namespace NUMINAMATH_CALUDE_brush_width_ratio_l3970_397000

theorem brush_width_ratio (w l b : ℝ) (h1 : w = 4) (h2 : l = 9) : 
  b * Real.sqrt (w^2 + l^2) = (w * l) / 3 → l / b = 3 * Real.sqrt 97 / 4 := by
  sorry

end NUMINAMATH_CALUDE_brush_width_ratio_l3970_397000


namespace NUMINAMATH_CALUDE_half_of_recipe_l3970_397003

theorem half_of_recipe (original_recipe : ℚ) (half_recipe : ℚ) : 
  original_recipe = 4.5 → half_recipe = original_recipe / 2 → half_recipe = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_half_of_recipe_l3970_397003


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3970_397038

-- Equation 1
theorem solve_equation_one (x : ℝ) : 2 * x - 7 = 5 * x - 1 → x = -2 := by
  sorry

-- Equation 2
theorem solve_equation_two (x : ℝ) : (x - 2) / 2 - (x - 1) / 6 = 1 → x = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3970_397038


namespace NUMINAMATH_CALUDE_range_of_s_squared_minus_c_squared_l3970_397039

theorem range_of_s_squared_minus_c_squared (k : ℝ) (x y : ℝ) :
  k > 0 →
  x = k * y →
  let r := Real.sqrt (x^2 + y^2)
  let s := y / r
  let c := x / r
  (∀ z, s^2 - c^2 = z → -1 ≤ z ∧ z ≤ 1) ∧
  (∃ z, s^2 - c^2 = z ∧ z = -1) ∧
  (∃ z, s^2 - c^2 = z ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_s_squared_minus_c_squared_l3970_397039


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3970_397061

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : GeometricSequence a)
    (h_product : a 1 * a 7 * a 13 = 8) :
  a 3 * a 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3970_397061


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3970_397040

theorem line_intersects_circle (a : ℝ) (h : a ≥ 0) :
  ∃ (x y : ℝ), (a * x - y + Real.sqrt 2 * a = 0) ∧ (x^2 + y^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3970_397040


namespace NUMINAMATH_CALUDE_compare_x_powers_l3970_397008

theorem compare_x_powers (x : ℝ) (h : 0 < x ∧ x < 1) : x^2 < Real.sqrt x ∧ Real.sqrt x < x ∧ x < 1/x := by
  sorry

end NUMINAMATH_CALUDE_compare_x_powers_l3970_397008


namespace NUMINAMATH_CALUDE_cone_surface_area_l3970_397047

/-- The surface area of a cone with given height and base area -/
theorem cone_surface_area (h : ℝ) (base_area : ℝ) (h_pos : h > 0) (base_pos : base_area > 0) :
  let r := Real.sqrt (base_area / Real.pi)
  let l := Real.sqrt (r^2 + h^2)
  h = 4 ∧ base_area = 9 * Real.pi → Real.pi * r * l + base_area = 24 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3970_397047


namespace NUMINAMATH_CALUDE_min_solution_value_l3970_397013

def system (x y : ℝ) : Prop :=
  3^(-x) * y^4 - 2*y^2 + 3^x ≤ 0 ∧ 27^x + y^4 - 3^x - 1 = 0

def solution_value (x y : ℝ) : ℝ := x^3 + y^3

theorem min_solution_value :
  ∃ (min : ℝ), min = -1 ∧
  (∀ x y : ℝ, system x y → solution_value x y ≥ min) ∧
  (∃ x y : ℝ, system x y ∧ solution_value x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_solution_value_l3970_397013


namespace NUMINAMATH_CALUDE_power_multiplication_l3970_397042

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3970_397042


namespace NUMINAMATH_CALUDE_odd_sum_probability_in_4x4_grid_l3970_397052

theorem odd_sum_probability_in_4x4_grid : 
  let n : ℕ := 16
  let grid_size : ℕ := 4
  let total_arrangements : ℕ := n.factorial
  let valid_arrangements : ℕ := (Nat.choose grid_size 2)^2 * (n/2).factorial * (n/2).factorial
  (valid_arrangements : ℚ) / total_arrangements = 1 / 360 := by
sorry

end NUMINAMATH_CALUDE_odd_sum_probability_in_4x4_grid_l3970_397052


namespace NUMINAMATH_CALUDE_race_time_calculation_l3970_397081

/-- A theorem about a race between two runners --/
theorem race_time_calculation (race_distance : ℝ) (b_time : ℝ) (a_lead : ℝ) (a_time : ℝ) : 
  race_distance = 120 →
  b_time = 45 →
  a_lead = 24 →
  a_time = 56.25 →
  (race_distance / a_time = (race_distance - a_lead) / b_time) := by
sorry

end NUMINAMATH_CALUDE_race_time_calculation_l3970_397081


namespace NUMINAMATH_CALUDE_odd_count_after_ten_operations_l3970_397097

/-- Represents the state of the board after n operations -/
structure BoardState (n : ℕ) where
  odd_count : ℕ  -- Number of odd numbers on the board
  total_count : ℕ  -- Total number of numbers on the board

/-- Performs one operation on the board -/
def next_state (state : BoardState n) : BoardState (n + 1) :=
  sorry

/-- Initial state of the board with 0 and 1 -/
def initial_state : BoardState 0 :=
  { odd_count := 1, total_count := 2 }

/-- The state of the board after n operations -/
def board_state (n : ℕ) : BoardState n :=
  match n with
  | 0 => initial_state
  | n + 1 => next_state (board_state n)

theorem odd_count_after_ten_operations :
  (board_state 10).odd_count = 683 :=
sorry

end NUMINAMATH_CALUDE_odd_count_after_ten_operations_l3970_397097


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3970_397078

/-- Calculates the area of a rectangular field given specific fencing conditions -/
theorem rectangular_field_area (uncovered_side : ℝ) (total_fencing : ℝ) : uncovered_side = 20 → total_fencing = 76 → uncovered_side * ((total_fencing - uncovered_side) / 2) = 560 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3970_397078


namespace NUMINAMATH_CALUDE_homework_completion_l3970_397091

theorem homework_completion (total : ℝ) (h : total > 0) : 
  let monday := (3 / 5 : ℝ) * total
  let tuesday := (1 / 3 : ℝ) * (total - monday)
  let wednesday := total - monday - tuesday
  wednesday = (4 / 15 : ℝ) * total := by
  sorry

end NUMINAMATH_CALUDE_homework_completion_l3970_397091


namespace NUMINAMATH_CALUDE_campaign_fund_distribution_l3970_397018

theorem campaign_fund_distribution (total : ℝ) (family_percent : ℝ) (own_savings : ℝ) :
  total = 10000 →
  family_percent = 0.3 →
  own_savings = 4200 →
  ∃ (friends_contribution : ℝ),
    friends_contribution = total * 0.4 ∧
    total = friends_contribution + (family_percent * (total - friends_contribution)) + own_savings :=
by sorry

end NUMINAMATH_CALUDE_campaign_fund_distribution_l3970_397018


namespace NUMINAMATH_CALUDE_max_value_implies_m_eq_one_min_value_of_y_l3970_397017

noncomputable section

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x

-- Define the derivative of f(x)
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 1 / x - m

-- Part 1: Prove that if the maximum value of f(x) is -1, then m = 1
theorem max_value_implies_m_eq_one (m : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > 0, f m x ≤ f m x₀) ∧ f m (1 / m) = -1 → m = 1 :=
sorry

-- Part 2: Prove that the minimum value of y is 2 / (1 + e)
theorem min_value_of_y :
  ∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 →
  f 1 x₁ = 0 ∧ f 1 x₂ = 0 →
  Real.exp x₁ ≤ x₂ →
  (x₁ - x₂) * f_derivative 1 (x₁ + x₂) ≥ 2 / (1 + Real.exp 1) :=
sorry

end

end NUMINAMATH_CALUDE_max_value_implies_m_eq_one_min_value_of_y_l3970_397017


namespace NUMINAMATH_CALUDE_bus_problem_l3970_397024

/-- The number of people who got off at the second stop of a bus route -/
def people_off_second_stop (initial : ℕ) (first_off : ℕ) (second_on : ℕ) (third_off : ℕ) (third_on : ℕ) (final : ℕ) : ℕ :=
  initial - first_off - final + second_on - third_off + third_on

theorem bus_problem : people_off_second_stop 50 15 2 4 3 28 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l3970_397024


namespace NUMINAMATH_CALUDE_congruent_face_tetrahedron_volume_l3970_397064

/-- A tetrahedron with congruent triangular faces -/
structure CongruentFaceTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  triangle_inequality_ab : a < b + c
  triangle_inequality_bc : b < a + c
  triangle_inequality_ca : c < a + b

/-- The volume of a tetrahedron with congruent triangular faces -/
noncomputable def volume (t : CongruentFaceTetrahedron) : ℝ :=
  (1 / (6 * Real.sqrt 2)) * Real.sqrt ((-t.a^2 + t.b^2 + t.c^2) * (t.a^2 - t.b^2 + t.c^2) * (t.a^2 + t.b^2 - t.c^2))

/-- Theorem: The volume of a tetrahedron with congruent triangular faces is given by the formula -/
theorem congruent_face_tetrahedron_volume (t : CongruentFaceTetrahedron) :
  ∃ V, V = volume t ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_congruent_face_tetrahedron_volume_l3970_397064


namespace NUMINAMATH_CALUDE_fast_food_order_l3970_397054

/-- The cost of a burger in dollars -/
def burger_cost : ℕ := 5

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a smoothie in dollars -/
def smoothie_cost : ℕ := 4

/-- The total cost of the order in dollars -/
def total_cost : ℕ := 17

/-- The number of smoothies ordered -/
def num_smoothies : ℕ := 2

theorem fast_food_order :
  burger_cost + sandwich_cost + smoothie_cost * num_smoothies = total_cost := by
  sorry

end NUMINAMATH_CALUDE_fast_food_order_l3970_397054


namespace NUMINAMATH_CALUDE_max_value_theorem_l3970_397072

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  2 * Real.sqrt (a * b * c / 8) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3970_397072
