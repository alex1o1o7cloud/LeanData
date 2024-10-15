import Mathlib

namespace NUMINAMATH_CALUDE_one_integer_solution_l1137_113760

def circle_center : ℝ × ℝ := (4, 6)
def circle_radius : ℝ := 8

def point (x : ℤ) : ℝ × ℝ := (2 * x, -x)

def inside_or_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 ≤ circle_radius^2

theorem one_integer_solution : 
  ∃! x : ℤ, inside_or_on_circle (point x) :=
sorry

end NUMINAMATH_CALUDE_one_integer_solution_l1137_113760


namespace NUMINAMATH_CALUDE_intersection_M_N_l1137_113755

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N :
  M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1137_113755


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1137_113753

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | x^2 + 2*x + a > 0}
  (a > 1 → solution_set = Set.univ) ∧
  (a = 1 → solution_set = {x : ℝ | x ≠ -1}) ∧
  (a < 1 → solution_set = {x : ℝ | x > -1 + Real.sqrt (1 - a) ∨ x < -1 - Real.sqrt (1 - a)}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1137_113753


namespace NUMINAMATH_CALUDE_concert_ticket_discount_l1137_113761

theorem concert_ticket_discount (ticket_price : ℕ) (total_tickets : ℕ) (total_paid : ℕ) 
  (h1 : ticket_price = 40)
  (h2 : total_tickets = 12)
  (h3 : total_paid = 476)
  (h4 : total_tickets > 10) : 
  (ticket_price * total_tickets - total_paid) / (ticket_price * (total_tickets - 10)) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_discount_l1137_113761


namespace NUMINAMATH_CALUDE_field_area_minus_ponds_l1137_113728

/-- The area of a square field with sides of 10 meters, minus the area of three non-overlapping 
    circular ponds each with a radius of 3 meters, is equal to 100 - 27π square meters. -/
theorem field_area_minus_ponds (π : ℝ) : ℝ := by
  -- Define the side length of the square field
  let square_side : ℝ := 10
  -- Define the radius of each circular pond
  let pond_radius : ℝ := 3
  -- Define the number of ponds
  let num_ponds : ℕ := 3
  -- Calculate the area of the square field
  let square_area : ℝ := square_side ^ 2
  -- Calculate the area of one circular pond
  let pond_area : ℝ := π * pond_radius ^ 2
  -- Calculate the total area of all ponds
  let total_pond_area : ℝ := num_ponds * pond_area
  -- Calculate the remaining area (field area minus pond area)
  let remaining_area : ℝ := square_area - total_pond_area
  -- Prove that the remaining area is equal to 100 - 27π
  sorry

#check field_area_minus_ponds

end NUMINAMATH_CALUDE_field_area_minus_ponds_l1137_113728


namespace NUMINAMATH_CALUDE_optimal_income_maximizes_take_home_pay_l1137_113707

/-- Represents the take-home pay as a function of the tax rate x -/
def takeHomePay (x : ℝ) : ℝ := 1000 * (x + 10) - 10 * x * (x + 10)

/-- The optimal tax rate that maximizes take-home pay -/
def optimalRate : ℝ := 45

/-- The income corresponding to the optimal tax rate -/
def optimalIncome : ℝ := (optimalRate + 10) * 1000

theorem optimal_income_maximizes_take_home_pay :
  optimalIncome = 55000 ∧
  ∀ x : ℝ, takeHomePay x ≤ takeHomePay optimalRate :=
sorry

end NUMINAMATH_CALUDE_optimal_income_maximizes_take_home_pay_l1137_113707


namespace NUMINAMATH_CALUDE_min_floor_equation_l1137_113732

theorem min_floor_equation (n : ℕ) : 
  (∃ k : ℕ, k^2 + Int.floor (n / k^2 : ℚ) = 1991 ∧ 
   ∀ m : ℕ, m^2 + Int.floor (n / m^2 : ℚ) ≥ 1991) ↔ 
  990208 ≤ n ∧ n ≤ 991231 := by
sorry

end NUMINAMATH_CALUDE_min_floor_equation_l1137_113732


namespace NUMINAMATH_CALUDE_odd_digits_base4_437_l1137_113739

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 437 is 4 -/
theorem odd_digits_base4_437 : countOddDigits (toBase4 437) = 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_437_l1137_113739


namespace NUMINAMATH_CALUDE_betty_stones_count_l1137_113774

/-- The number of stones in each bracelet -/
def stones_per_bracelet : ℕ := 14

/-- The number of bracelets Betty can make -/
def number_of_bracelets : ℕ := 10

/-- The total number of stones Betty bought -/
def total_stones : ℕ := stones_per_bracelet * number_of_bracelets

theorem betty_stones_count : total_stones = 140 := by
  sorry

end NUMINAMATH_CALUDE_betty_stones_count_l1137_113774


namespace NUMINAMATH_CALUDE_smallest_possible_median_l1137_113796

def number_set (y : ℤ) : Finset ℤ := {y, 3*y, 4, 1, 7}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧
  2 * (s.filter (· ≥ m)).card ≥ s.card

theorem smallest_possible_median :
  ∃ y : ℤ, is_median 1 (number_set y) ∧
  ∀ m : ℤ, ∀ z : ℤ, is_median m (number_set z) → m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_median_l1137_113796


namespace NUMINAMATH_CALUDE_sequence_integer_count_l1137_113778

def sequence_term (n : ℕ) : ℚ :=
  24300 / (5 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬ is_integer (sequence_term k)) →
  (∃! (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬ is_integer (sequence_term k)) ∧
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬ is_integer (sequence_term k) ∧
    k = 3) :=
by sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l1137_113778


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l1137_113782

def num_white_balls : ℕ := 4
def num_red_balls : ℕ := 2

theorem probability_of_red_ball :
  let total_balls := num_white_balls + num_red_balls
  (num_red_balls : ℚ) / total_balls = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l1137_113782


namespace NUMINAMATH_CALUDE_at_least_one_less_than_one_negation_all_not_less_than_one_l1137_113781

theorem at_least_one_less_than_one (a b c : ℝ) (ha : a < 3) (hb : b < 3) (hc : c < 3) :
  a < 1 ∨ b < 1 ∨ c < 1 := by
  sorry

theorem negation_all_not_less_than_one (a b c : ℝ) :
  (¬(a < 1 ∨ b < 1 ∨ c < 1)) ↔ (a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_one_negation_all_not_less_than_one_l1137_113781


namespace NUMINAMATH_CALUDE_line_intersection_l1137_113725

theorem line_intersection :
  ∃! p : ℚ × ℚ, 
    (3 * p.2 = -2 * p.1 + 6) ∧ 
    (-2 * p.2 = 6 * p.1 + 4) ∧ 
    p = (-12/7, 22/7) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l1137_113725


namespace NUMINAMATH_CALUDE_veranda_area_is_136_l1137_113789

/-- Calculates the area of a veranda surrounding a rectangular room. -/
def verandaArea (roomLength roomWidth verandaWidth : ℝ) : ℝ :=
  (roomLength + 2 * verandaWidth) * (roomWidth + 2 * verandaWidth) - roomLength * roomWidth

/-- Theorem stating that the area of the veranda is 136 m² given the specified dimensions. -/
theorem veranda_area_is_136 :
  let roomLength : ℝ := 18
  let roomWidth : ℝ := 12
  let verandaWidth : ℝ := 2
  verandaArea roomLength roomWidth verandaWidth = 136 := by
  sorry

#eval verandaArea 18 12 2

end NUMINAMATH_CALUDE_veranda_area_is_136_l1137_113789


namespace NUMINAMATH_CALUDE_fan_sales_and_profit_l1137_113767

/-- Represents an electric fan model with its purchase price and selling price -/
structure FanModel where
  purchasePrice : ℝ
  sellingPrice : ℝ

/-- Represents sales data for a week -/
structure WeeklySales where
  modelAQty : ℕ
  modelBQty : ℕ
  revenue : ℝ

/-- The main theorem about electric fan sales and profits -/
theorem fan_sales_and_profit 
  (modelA modelB : FanModel)
  (week1 week2 : WeeklySales)
  (totalUnits : ℕ)
  (totalBudget profitGoal : ℝ) :
  modelA.purchasePrice = 200 →
  modelB.purchasePrice = 170 →
  week1.modelAQty = 3 →
  week1.modelBQty = 5 →
  week1.revenue = 1800 →
  week2.modelAQty = 4 →
  week2.modelBQty = 10 →
  week2.revenue = 3100 →
  totalUnits = 30 →
  totalBudget = 5400 →
  profitGoal = 1400 →
  modelA.sellingPrice = 250 ∧
  modelB.sellingPrice = 210 ∧
  (∀ a : ℕ, a ≤ totalUnits → 
    modelA.purchasePrice * a + modelB.purchasePrice * (totalUnits - a) ≤ totalBudget →
    a ≤ 10) ∧
  ¬(∃ a : ℕ, a ≤ 10 ∧ 
    (modelA.sellingPrice - modelA.purchasePrice) * a + 
    (modelB.sellingPrice - modelB.purchasePrice) * (totalUnits - a) ≥ profitGoal) :=
by sorry

end NUMINAMATH_CALUDE_fan_sales_and_profit_l1137_113767


namespace NUMINAMATH_CALUDE_fraction_product_l1137_113737

theorem fraction_product : (3 : ℚ) / 7 * 5 / 8 * 9 / 13 * 11 / 17 = 1485 / 12376 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1137_113737


namespace NUMINAMATH_CALUDE_chip_cost_calculation_l1137_113756

def days_per_week : ℕ := 5
def weeks : ℕ := 4
def total_spent : ℚ := 10

def total_days : ℕ := days_per_week * weeks

theorem chip_cost_calculation :
  total_spent / total_days = 1/2 := by sorry

end NUMINAMATH_CALUDE_chip_cost_calculation_l1137_113756


namespace NUMINAMATH_CALUDE_geq_three_necessary_not_sufficient_for_gt_three_l1137_113752

theorem geq_three_necessary_not_sufficient_for_gt_three :
  (∀ x : ℝ, x > 3 → x ≥ 3) ∧
  (∃ x : ℝ, x ≥ 3 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_geq_three_necessary_not_sufficient_for_gt_three_l1137_113752


namespace NUMINAMATH_CALUDE_ways_to_pick_one_ball_ways_to_pick_two_different_colored_balls_l1137_113706

-- Define the number of red and white balls
def num_red_balls : ℕ := 8
def num_white_balls : ℕ := 7

-- Theorem for the first question
theorem ways_to_pick_one_ball : 
  num_red_balls + num_white_balls = 15 := by sorry

-- Theorem for the second question
theorem ways_to_pick_two_different_colored_balls : 
  num_red_balls * num_white_balls = 56 := by sorry

end NUMINAMATH_CALUDE_ways_to_pick_one_ball_ways_to_pick_two_different_colored_balls_l1137_113706


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_6_proof_l1137_113717

/-- The greatest four-digit number divisible by 6 -/
def greatest_four_digit_divisible_by_6 : ℕ := 9996

/-- A number is a four-digit number if it's between 1000 and 9999 inclusive -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_6_proof :
  (is_four_digit greatest_four_digit_divisible_by_6) ∧ 
  (greatest_four_digit_divisible_by_6 % 6 = 0) ∧
  (∀ n : ℕ, is_four_digit n → n % 6 = 0 → n ≤ greatest_four_digit_divisible_by_6) :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_6_proof_l1137_113717


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_times_two_l1137_113754

/-- The sum of two repeating decimals multiplied by 2 -/
theorem repeating_decimal_sum_times_two :
  2 * ((5 : ℚ) / 9 + (7 : ℚ) / 9) = (8 : ℚ) / 3 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_times_two_l1137_113754


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1137_113746

theorem negation_of_proposition (p : Prop) :
  (¬p ↔ ∃ x > 0, Real.exp x < x + 1) ↔ (p ↔ ∀ x > 0, Real.exp x ≥ x + 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1137_113746


namespace NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_skew_lines_parallel_to_planes_implies_parallel_planes_l1137_113730

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (skew : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the planes α and β
variable (α β : Plane)

-- Theorem 1: Condition ①
theorem perpendicular_line_implies_parallel_planes 
  (a : Line) 
  (h1 : perpendicular a α) 
  (h2 : perpendicular a β) : 
  parallel α β := by sorry

-- Theorem 2: Condition ④
theorem skew_lines_parallel_to_planes_implies_parallel_planes 
  (a b : Line) 
  (h1 : contains α a) 
  (h2 : contains β b) 
  (h3 : line_parallel_plane a β) 
  (h4 : line_parallel_plane b α) 
  (h5 : skew a b) : 
  parallel α β := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_skew_lines_parallel_to_planes_implies_parallel_planes_l1137_113730


namespace NUMINAMATH_CALUDE_melissa_total_points_l1137_113799

/-- The number of points Melissa scores in each game -/
def points_per_game : ℕ := 120

/-- The number of games played -/
def num_games : ℕ := 10

/-- The total points scored by Melissa in all games -/
def total_points : ℕ := points_per_game * num_games

theorem melissa_total_points : total_points = 1200 := by
  sorry

end NUMINAMATH_CALUDE_melissa_total_points_l1137_113799


namespace NUMINAMATH_CALUDE_symmetric_point_on_parabola_l1137_113750

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y = (x-h)^2 + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- Checks if a point lies on a given parabola -/
def isOnParabola (p : Point) (parab : Parabola) : Prop :=
  p.y = (p.x - parab.h)^2 + parab.k

/-- Finds the symmetric point with respect to the axis of symmetry of a parabola -/
def symmetricPoint (p : Point) (parab : Parabola) : Point :=
  ⟨2 * parab.h - p.x, p.y⟩

theorem symmetric_point_on_parabola (parab : Parabola) (p : Point) :
  isOnParabola p parab → p.x = -1 → 
  symmetricPoint p parab = Point.mk 3 6 := by
  sorry

#check symmetric_point_on_parabola

end NUMINAMATH_CALUDE_symmetric_point_on_parabola_l1137_113750


namespace NUMINAMATH_CALUDE_reflect_point_3_2_l1137_113764

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis in a Cartesian coordinate system -/
def reflect_x_axis (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

/-- Theorem: Reflecting the point (3,2) across the x-axis results in (3,-2) -/
theorem reflect_point_3_2 :
  reflect_x_axis ⟨3, 2⟩ = ⟨3, -2⟩ := by
  sorry

end NUMINAMATH_CALUDE_reflect_point_3_2_l1137_113764


namespace NUMINAMATH_CALUDE_temperature_difference_l1137_113771

theorem temperature_difference (M L N : ℝ) : 
  M = L + N →
  (∃ (M_4 L_4 : ℝ), 
    M_4 = M - 5 ∧
    L_4 = L + 3 ∧
    abs (M_4 - L_4) = 2) →
  (N = 6 ∨ N = 10) ∧ 6 * 10 = 60 := by
sorry

end NUMINAMATH_CALUDE_temperature_difference_l1137_113771


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1137_113769

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 - 5*m + 6 : ℂ) + (m^2 - 3*m : ℂ) * Complex.I = Complex.I * ((m^2 - 3*m : ℂ)) → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1137_113769


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1137_113772

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 48 → 
    b = 64 → 
    c^2 = a^2 + b^2 → 
    c = 80 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1137_113772


namespace NUMINAMATH_CALUDE_complex_determinant_equation_l1137_113738

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_determinant_equation : 
  ∃ z : ℂ, determinant z (-Complex.I) (1 - Complex.I) (1 + Complex.I) = 0 ∧ z = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_determinant_equation_l1137_113738


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l1137_113721

theorem p_or_q_is_true (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l1137_113721


namespace NUMINAMATH_CALUDE_c_alone_time_l1137_113759

-- Define the work rates of A, B, and C
variable (A B C : ℚ)

-- Define the conditions from the problem
variable (h1 : A + B = 1 / 15)
variable (h2 : A + B + C = 1 / 12)

-- The theorem to prove
theorem c_alone_time : C = 1 / 60 :=
  sorry

end NUMINAMATH_CALUDE_c_alone_time_l1137_113759


namespace NUMINAMATH_CALUDE_polynomial_equality_l1137_113709

/-- Given polynomials h and k such that h(x) + k(x) = 3x^2 + 2x - 5 and h(x) = x^4 - 3x^2 + 1,
    prove that k(x) = -x^4 + 6x^2 + 2x - 6 -/
theorem polynomial_equality (x : ℝ) (h k : ℝ → ℝ) 
    (h_def : h = fun x => x^4 - 3*x^2 + 1)
    (sum_eq : ∀ x, h x + k x = 3*x^2 + 2*x - 5) :
  k x = -x^4 + 6*x^2 + 2*x - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1137_113709


namespace NUMINAMATH_CALUDE_decreasing_function_property_l1137_113705

-- Define a real-valued function f on the positive real numbers
variable (f : ℝ → ℝ)

-- Define the property of f being decreasing on (0, +∞)
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f y < f x

-- State the theorem
theorem decreasing_function_property
  (h : IsDecreasingOn f) : f 3 < f 2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_property_l1137_113705


namespace NUMINAMATH_CALUDE_sequence_sum_l1137_113736

theorem sequence_sum : 
  let S := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
           1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - 3)
  S = 13 / 4 + (3 / 4) * Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1137_113736


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1137_113798

theorem smallest_part_of_proportional_division (x y z : ℚ) :
  x + y + z = 64 ∧ 
  y = 2 * x ∧ 
  z = 3 * x →
  x = 32 / 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1137_113798


namespace NUMINAMATH_CALUDE_product_xyz_l1137_113794

theorem product_xyz (x y z k : ℝ) 
  (h1 : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (h2 : x^3 + y^3 + k*(x^2 + y^2) = 2008)
  (h3 : y^3 + z^3 + k*(y^2 + z^2) = 2008)
  (h4 : z^3 + x^3 + k*(z^2 + x^2) = 2008) :
  x * y * z = -1004 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l1137_113794


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1137_113749

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  slope : ℝ
  intercept : ℝ

-- Function to check if a line passes through a point
def passesThrough (l : Line2D) (p : Point2D) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.intercept = -l.slope * l.intercept ∨ l.slope = -1

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line2D),
    (passesThrough l1 ⟨-2, 3⟩ ∧ hasEqualIntercepts l1) ∧
    (passesThrough l2 ⟨-2, 3⟩ ∧ hasEqualIntercepts l2) ∧
    ((l1.slope = -3/2 ∧ l1.intercept = 0) ∨ (l2.slope = -1 ∧ l2.intercept = 1)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1137_113749


namespace NUMINAMATH_CALUDE_expected_winnings_unique_coin_l1137_113715

/-- A unique weighted coin with four possible outcomes -/
structure WeightedCoin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_edge : ℚ
  prob_other : ℚ
  winnings_heads : ℚ
  winnings_tails : ℚ
  winnings_edge : ℚ
  winnings_other : ℚ

/-- The expected winnings from flipping the coin -/
def expected_winnings (c : WeightedCoin) : ℚ :=
  c.prob_heads * c.winnings_heads +
  c.prob_tails * c.winnings_tails +
  c.prob_edge * c.winnings_edge +
  c.prob_other * c.winnings_other

/-- The specific coin described in the problem -/
def unique_coin : WeightedCoin :=
  { prob_heads := 3/7
  , prob_tails := 1/4
  , prob_edge := 1/7
  , prob_other := 2/7
  , winnings_heads := 2
  , winnings_tails := 4
  , winnings_edge := -6
  , winnings_other := -2 }

theorem expected_winnings_unique_coin :
  expected_winnings unique_coin = 3/7 := by sorry

end NUMINAMATH_CALUDE_expected_winnings_unique_coin_l1137_113715


namespace NUMINAMATH_CALUDE_correct_fruit_baskets_l1137_113773

/-- The number of ways to choose from n identical items -/
def chooseFrom (n : ℕ) : ℕ := n + 1

/-- The number of possible fruit baskets given the number of apples and oranges -/
def fruitBaskets (apples oranges : ℕ) : ℕ :=
  chooseFrom apples * chooseFrom oranges - 1

theorem correct_fruit_baskets :
  fruitBaskets 6 8 = 62 := by
  sorry

end NUMINAMATH_CALUDE_correct_fruit_baskets_l1137_113773


namespace NUMINAMATH_CALUDE_historians_contemporaries_probability_l1137_113735

/-- Represents the number of years in the time period --/
def totalYears : ℕ := 300

/-- Represents the lifespan of each historian --/
def lifespan : ℕ := 80

/-- Represents the probability space of possible birth year combinations --/
def totalPossibilities : ℕ := totalYears * totalYears

/-- Represents the number of favorable outcomes (contemporaneous birth year combinations) --/
def favorableOutcomes : ℕ := totalPossibilities - 2 * ((totalYears - lifespan) * (totalYears - lifespan) / 2)

/-- The probability of two historians being contemporaries --/
def probabilityOfContemporaries : ℚ := favorableOutcomes / totalPossibilities

theorem historians_contemporaries_probability :
  probabilityOfContemporaries = 104 / 225 := by
  sorry

end NUMINAMATH_CALUDE_historians_contemporaries_probability_l1137_113735


namespace NUMINAMATH_CALUDE_sum_123_consecutive_even_from_2_l1137_113785

/-- Sum of consecutive even numbers -/
def sum_consecutive_even (start : ℕ) (count : ℕ) : ℕ :=
  count * (2 * start + (count - 1) * 2) / 2

/-- Theorem: The sum of 123 consecutive even numbers starting from 2 is 15252 -/
theorem sum_123_consecutive_even_from_2 :
  sum_consecutive_even 2 123 = 15252 := by
  sorry

end NUMINAMATH_CALUDE_sum_123_consecutive_even_from_2_l1137_113785


namespace NUMINAMATH_CALUDE_three_zeroes_implies_a_range_l1137_113776

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then (x - 1) / Real.exp x else -x - 1

-- Define g as a function of f and b
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a x - b

-- State the theorem
theorem three_zeroes_implies_a_range :
  ∀ a : ℝ, (∃ b : ℝ, (∃! z1 z2 z3 : ℝ, z1 ≠ z2 ∧ z1 ≠ z3 ∧ z2 ≠ z3 ∧
    g a b z1 = 0 ∧ g a b z2 = 0 ∧ g a b z3 = 0 ∧
    (∀ z : ℝ, g a b z = 0 → z = z1 ∨ z = z2 ∨ z = z3))) →
  a > -1 / Real.exp 2 - 1 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_zeroes_implies_a_range_l1137_113776


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1137_113745

/-- 
Given a quadratic equation (k-1)x^2 - 2x + 1 = 0, 
this theorem states the conditions on k for the equation to have two distinct real roots.
-/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 1) * x^2 - 2 * x + 1 = 0 ∧ 
   (k - 1) * y^2 - 2 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1137_113745


namespace NUMINAMATH_CALUDE_smallest_triangle_longer_leg_l1137_113740

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hyp_short : shorterLeg = hypotenuse / 2
  hyp_long : longerLeg = shorterLeg * Real.sqrt 3

/-- Represents a sequence of four 30-60-90 triangles -/
structure TriangleSequence where
  t1 : Triangle30_60_90
  t2 : Triangle30_60_90
  t3 : Triangle30_60_90
  t4 : Triangle30_60_90
  hyp_relation1 : t1.longerLeg = t2.hypotenuse
  hyp_relation2 : t2.longerLeg = t3.hypotenuse
  hyp_relation3 : t3.longerLeg = t4.hypotenuse
  largest_hyp : t1.hypotenuse = 16

theorem smallest_triangle_longer_leg (seq : TriangleSequence) : seq.t4.longerLeg = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_longer_leg_l1137_113740


namespace NUMINAMATH_CALUDE_electricity_constant_is_correct_l1137_113797

/-- Represents the relationship between electricity bill and consumption -/
def electricity_equation (x y : ℝ) : Prop := y = 0.54 * x

/-- The constant in the electricity equation -/
def electricity_constant : ℝ := 0.54

/-- Theorem stating that the constant in the electricity equation is 0.54 -/
theorem electricity_constant_is_correct :
  ∀ x y : ℝ, electricity_equation x y → 
  ∃ c : ℝ, (∀ x' y' : ℝ, electricity_equation x' y' → y' = c * x') ∧ c = electricity_constant :=
sorry

end NUMINAMATH_CALUDE_electricity_constant_is_correct_l1137_113797


namespace NUMINAMATH_CALUDE_max_distance_point_circle_l1137_113784

/-- The maximum distance between a point and a circle -/
theorem max_distance_point_circle :
  let center : ℝ × ℝ := (1, 2)
  let radius : ℝ := 2
  let P : ℝ × ℝ := (3, 3)
  (∀ M : ℝ × ℝ, (M.1 - center.1)^2 + (M.2 - center.2)^2 = radius^2 →
    Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) ≤ Real.sqrt 5 + 2) ∧
  (∃ M : ℝ × ℝ, (M.1 - center.1)^2 + (M.2 - center.2)^2 = radius^2 ∧
    Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) = Real.sqrt 5 + 2) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_point_circle_l1137_113784


namespace NUMINAMATH_CALUDE_expression_simplification_l1137_113701

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 2) * (8 * p - 12) = 36 * p - 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1137_113701


namespace NUMINAMATH_CALUDE_convex_ngon_diagonals_and_triangles_l1137_113733

/-- A convex n-gon where no three diagonals intersect at the same point -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry
  no_triple_intersection : sorry

/-- The number of diagonals in a convex n-gon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of internal triangles formed by sides and diagonals of a convex n-gon -/
def num_internal_triangles (n : ℕ) : ℕ :=
  Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6

/-- Theorem stating the number of diagonals and internal triangles in a convex n-gon -/
theorem convex_ngon_diagonals_and_triangles (n : ℕ) (A : ConvexNGon n) :
  (num_diagonals n = n * (n - 3) / 2) ∧
  (num_internal_triangles n = Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6) :=
by sorry

end NUMINAMATH_CALUDE_convex_ngon_diagonals_and_triangles_l1137_113733


namespace NUMINAMATH_CALUDE_quadratic_properties_l1137_113700

/-- A quadratic function with a < 0, f(-1) = 0, and axis of symmetry x = 1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_neg : a < 0
  root_neg_one : a * (-1)^2 + b * (-1) + c = 0
  axis_sym : -b / (2 * a) = 1

/-- Properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (f.a - f.b + f.c = 0) ∧
  (∀ m : ℝ, f.a * m^2 + f.b * m + f.c ≤ -4 * f.a) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → 
    f.a * x₁^2 + f.b * x₁ + f.c + 1 = 0 → 
    f.a * x₂^2 + f.b * x₂ + f.c + 1 = 0 → 
    x₁ < -1 ∧ x₂ > 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1137_113700


namespace NUMINAMATH_CALUDE_distance_between_closest_points_of_circles_l1137_113710

/-- Given two circles with centers at (1,1) and (20,5), both tangent to the x-axis,
    the distance between their closest points is √377 - 6. -/
theorem distance_between_closest_points_of_circles :
  let center1 : ℝ × ℝ := (1, 1)
  let center2 : ℝ × ℝ := (20, 5)
  let radius1 : ℝ := center1.2  -- y-coordinate of center1
  let radius2 : ℝ := center2.2  -- y-coordinate of center2
  let distance_between_centers : ℝ := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  distance_between_centers - (radius1 + radius2) = Real.sqrt 377 - 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_closest_points_of_circles_l1137_113710


namespace NUMINAMATH_CALUDE_line_equation_l1137_113743

/-- Given two parallel lines l₁ and l₂, prove that the line passing through H(-1, 1) with its
    midpoint M lying on x - y - 1 = 0 has the equation x + y = 0. -/
theorem line_equation (A B C₁ C₂ : ℝ) (h₁ : C₁ ≠ C₂) (h₂ : A - B + C₁ + C₂ = 0) :
  ∃ (l : ℝ → ℝ → Prop),
    (∀ x y, l x y ↔ x + y = 0) ∧
    l (-1) 1 ∧
    ∃ (M : ℝ × ℝ),
      (M.1 - M.2 - 1 = 0) ∧
      (∃ (t : ℝ), 
        A * (t * (-1) + (1 - t) * M.1) + B * (t * 1 + (1 - t) * M.2) + C₁ = 0 ∧
        A * (t * (-1) + (1 - t) * M.1) + B * (t * 1 + (1 - t) * M.2) + C₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l1137_113743


namespace NUMINAMATH_CALUDE_line_perp_condition_l1137_113714

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)
variable (para : Line → Plane → Prop)

-- State the theorem
theorem line_perp_condition 
  (a b : Line) (α : Plane) :
  perpPlane a α → para b α → perp a b :=
sorry

end NUMINAMATH_CALUDE_line_perp_condition_l1137_113714


namespace NUMINAMATH_CALUDE_interception_time_correct_l1137_113744

/-- Represents the time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat

/-- Represents the naval pursuit scenario -/
structure NavalPursuit where
  initialDistance : Real
  initialTime : TimeOfDay
  destroyerInitialSpeed : Real
  cargoShipSpeed : Real
  speedChangeTime : Real
  destroyerReducedSpeed : Real

/-- Calculates the time of interception given the naval pursuit scenario -/
def timeOfInterception (scenario : NavalPursuit) : Real :=
  sorry

/-- Theorem stating that the time of interception is 3 hours and 40 minutes after the initial time -/
theorem interception_time_correct (scenario : NavalPursuit) 
  (h1 : scenario.initialDistance = 20)
  (h2 : scenario.initialTime = ⟨9, 0⟩)
  (h3 : scenario.destroyerInitialSpeed = 16)
  (h4 : scenario.cargoShipSpeed = 10)
  (h5 : scenario.speedChangeTime = 3)
  (h6 : scenario.destroyerReducedSpeed = 13) :
  timeOfInterception scenario = 3 + 40 / 60 := by
  sorry

end NUMINAMATH_CALUDE_interception_time_correct_l1137_113744


namespace NUMINAMATH_CALUDE_max_area_isosceles_trapezoid_in_circle_l1137_113747

/-- An isosceles trapezoid inscribed in a circle -/
structure IsoscelesTrapezoidInCircle where
  r : ℝ  -- radius of the circle
  x : ℝ  -- length of the legs of the trapezoid
  a : ℝ  -- length of one parallel side
  d : ℝ  -- length of the other parallel side
  h : x ≥ 2 * r  -- condition that legs are at least as long as the diameter
  tangent : a + d = 2 * x  -- condition for tangent quadrilateral

/-- The area of an isosceles trapezoid inscribed in a circle -/
def area (t : IsoscelesTrapezoidInCircle) : ℝ := 2 * t.x * t.r

/-- Theorem: The maximum area of an isosceles trapezoid inscribed in a circle with radius r is 4r^2 -/
theorem max_area_isosceles_trapezoid_in_circle (t : IsoscelesTrapezoidInCircle) :
  area t ≤ 4 * t.r^2 :=
sorry

end NUMINAMATH_CALUDE_max_area_isosceles_trapezoid_in_circle_l1137_113747


namespace NUMINAMATH_CALUDE_toris_height_l1137_113779

theorem toris_height (initial_height growth : ℝ) 
  (h1 : initial_height = 4.4)
  (h2 : growth = 2.86) :
  initial_height + growth = 7.26 := by
sorry

end NUMINAMATH_CALUDE_toris_height_l1137_113779


namespace NUMINAMATH_CALUDE_log_difference_equality_l1137_113788

theorem log_difference_equality : 
  Real.sqrt (Real.log 12 / Real.log 4 - Real.log 12 / Real.log 5) = 
  Real.sqrt ((Real.log 12 * Real.log 1.25) / (Real.log 4 * Real.log 5)) := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equality_l1137_113788


namespace NUMINAMATH_CALUDE_fair_tickets_sold_l1137_113729

theorem fair_tickets_sold (baseball_tickets : ℕ) (fair_tickets : ℕ) : 
  baseball_tickets = 56 → 
  fair_tickets = 2 * baseball_tickets + 6 →
  fair_tickets = 118 := by
sorry

end NUMINAMATH_CALUDE_fair_tickets_sold_l1137_113729


namespace NUMINAMATH_CALUDE_max_intersection_points_l1137_113791

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  vertices : Fin 4 → ℝ × ℝ

/-- Represents an equilateral triangle in 2D space -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents the configuration of a circle, rectangle, and equilateral triangle -/
structure Configuration where
  circle : Circle
  rectangle : Rectangle
  triangle : EquilateralTriangle

/-- Predicate to check if two polygons are distinct -/
def are_distinct (rect : Rectangle) (tri : EquilateralTriangle) : Prop :=
  ∀ (i : Fin 4) (j : Fin 3), rect.vertices i ≠ tri.vertices j

/-- Predicate to check if two polygons do not overlap -/
def do_not_overlap (rect : Rectangle) (tri : EquilateralTriangle) : Prop :=
  sorry  -- Definition of non-overlapping polygons

/-- Function to count the number of intersection points -/
def count_intersections (config : Configuration) : ℕ :=
  sorry  -- Definition to count intersection points

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersection_points (config : Configuration) 
  (h1 : are_distinct config.rectangle config.triangle)
  (h2 : do_not_overlap config.rectangle config.triangle) :
  count_intersections config ≤ 14 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_l1137_113791


namespace NUMINAMATH_CALUDE_smallest_negative_quadratic_l1137_113726

theorem smallest_negative_quadratic (n : ℤ) : 
  (∀ m : ℤ, m < n → 4 * m^2 - 28 * m + 48 ≥ 0) ∧ 
  (4 * n^2 - 28 * n + 48 < 0) → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_negative_quadratic_l1137_113726


namespace NUMINAMATH_CALUDE_find_added_number_l1137_113768

theorem find_added_number (x y : ℤ) : 
  x % 82 = 5 → (x + y) % 41 = 12 → y = 7 := by sorry

end NUMINAMATH_CALUDE_find_added_number_l1137_113768


namespace NUMINAMATH_CALUDE_lifting_ratio_after_training_l1137_113787

/-- Calculates the ratio of lifting total to bodyweight after training -/
theorem lifting_ratio_after_training 
  (initial_total : ℝ)
  (initial_weight : ℝ)
  (total_increase_percent : ℝ)
  (weight_increase : ℝ)
  (h1 : initial_total = 2200)
  (h2 : initial_weight = 245)
  (h3 : total_increase_percent = 0.15)
  (h4 : weight_increase = 8) :
  (initial_total * (1 + total_increase_percent)) / (initial_weight + weight_increase) = 10 :=
by
  sorry

#check lifting_ratio_after_training

end NUMINAMATH_CALUDE_lifting_ratio_after_training_l1137_113787


namespace NUMINAMATH_CALUDE_sum_not_prime_l1137_113716

theorem sum_not_prime (a b c d : ℕ+) (h : a * b = c * d) : 
  ¬ Nat.Prime (a.val + b.val + c.val + d.val) := by
sorry

end NUMINAMATH_CALUDE_sum_not_prime_l1137_113716


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l1137_113793

theorem cubic_polynomial_problem (a b c : ℝ) (Q : ℝ → ℝ) :
  (∀ x, x^3 - 2*x^2 + 4*x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (∃ p q r s : ℝ, ∀ x, Q x = p*x^3 + q*x^2 + r*x + s) →
  Q a = b + c - 3 →
  Q b = a + c - 3 →
  Q c = a + b - 3 →
  Q (a + b + c) = -17 →
  (∀ x, Q x = -20/7*x^3 + 34/7*x^2 - 12/7*x + 13/7) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_problem_l1137_113793


namespace NUMINAMATH_CALUDE_negation_proposition_true_l1137_113790

theorem negation_proposition_true : ∃ (a b : ℝ), (2 * a + b > 5) ∧ (a ≠ 2 ∨ b ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_proposition_true_l1137_113790


namespace NUMINAMATH_CALUDE_wire_cutting_l1137_113762

theorem wire_cutting (total_length : ℝ) (difference : ℝ) 
  (h1 : total_length = 30)
  (h2 : difference = 2) :
  ∃ (shorter longer : ℝ),
    shorter + longer = total_length ∧
    longer = shorter + difference ∧
    shorter = 14 ∧
    longer = 16 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l1137_113762


namespace NUMINAMATH_CALUDE_specific_tree_height_l1137_113704

/-- Represents the height of a tree after a given number of years -/
def tree_height (initial_height : ℝ) (annual_growth : ℝ) (years : ℝ) : ℝ :=
  initial_height + annual_growth * years

/-- Theorem stating the height of a specific tree after x years -/
theorem specific_tree_height (x : ℝ) :
  tree_height 2.5 0.22 x = 2.5 + 0.22 * x := by
  sorry

end NUMINAMATH_CALUDE_specific_tree_height_l1137_113704


namespace NUMINAMATH_CALUDE_removed_number_is_1011_l1137_113765

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that removing 1011 from the set {1, 2, ..., 2021}
    makes the sum of the remaining numbers divisible by 2022 -/
theorem removed_number_is_1011 :
  ∀ x : ℕ, x ≤ 2021 →
    (sum_first_n 2021 - x) % 2022 = 0 → x = 1011 := by
  sorry

#check removed_number_is_1011

end NUMINAMATH_CALUDE_removed_number_is_1011_l1137_113765


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_bounded_zeros_product_lt_one_l1137_113718

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem f_nonnegative_iff_a_bounded (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
sorry

theorem zeros_product_lt_one (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (hz₁ : f a x₁ = 0) (hz₂ : f a x₂ = 0) :
  x₁ * x₂ < 1 :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_bounded_zeros_product_lt_one_l1137_113718


namespace NUMINAMATH_CALUDE_sin_cos_difference_36_degrees_l1137_113720

theorem sin_cos_difference_36_degrees : 
  Real.sin (36 * π / 180) * Real.cos (36 * π / 180) - 
  Real.cos (36 * π / 180) * Real.sin (36 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_36_degrees_l1137_113720


namespace NUMINAMATH_CALUDE_next_perfect_square_formula_l1137_113780

/-- A perfect square is an integer that is the square of another integer. -/
def isPerfectSquare (x : ℤ) : Prop := ∃ n : ℤ, x = n^2

/-- The next perfect square after a given perfect square. -/
def nextPerfectSquare (x : ℤ) : ℤ := x + 2 * Int.sqrt x + 1

/-- Theorem: For any perfect square x, the next perfect square is x + 2√x + 1. -/
theorem next_perfect_square_formula (x : ℤ) (h : isPerfectSquare x) :
  isPerfectSquare (nextPerfectSquare x) ∧ 
  ∀ y, isPerfectSquare y ∧ y > x → y ≥ nextPerfectSquare x :=
by sorry

end NUMINAMATH_CALUDE_next_perfect_square_formula_l1137_113780


namespace NUMINAMATH_CALUDE_kids_in_camp_l1137_113795

/-- The number of kids who go to camp in Lawrence county during summer break -/
theorem kids_in_camp (total_kids : ℕ) (kids_at_home : ℕ) 
  (h1 : total_kids = 313473) 
  (h2 : kids_at_home = 274865) : 
  total_kids - kids_at_home = 38608 := by
  sorry

end NUMINAMATH_CALUDE_kids_in_camp_l1137_113795


namespace NUMINAMATH_CALUDE_a4_plus_b4_equals_17_l1137_113792

theorem a4_plus_b4_equals_17 (a b : ℝ) (h1 : a^2 - b^2 = 5) (h2 : a * b = 2) : a^4 + b^4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_a4_plus_b4_equals_17_l1137_113792


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angles_equal_l1137_113770

/-- An isosceles triangle is a triangle with at least two equal sides -/
structure IsoscelesTriangle where
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ
  is_isosceles : side_a = side_b ∨ side_b = side_c ∨ side_c = side_a

/-- The two base angles of an isosceles triangle are equal -/
theorem isosceles_triangle_base_angles_equal (t : IsoscelesTriangle) :
  ∃ (angle1 angle2 : ℝ), angle1 = angle2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angles_equal_l1137_113770


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1137_113708

def n : ℕ := 12
def k : ℕ := 4
def p : ℚ := 1/2

theorem coin_flip_probability :
  (n.choose k) * p^k * (1 - p)^(n - k) = 495/4096 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1137_113708


namespace NUMINAMATH_CALUDE_sequence_sum_equals_square_l1137_113727

-- Define the sequence sum function
def sequenceSum (n : ℕ) : ℕ :=
  2 * (List.range n).sum + n

-- State the theorem
theorem sequence_sum_equals_square (n : ℕ) :
  n > 0 → sequenceSum n = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_square_l1137_113727


namespace NUMINAMATH_CALUDE_equal_intercept_line_equations_l1137_113786

/-- A line passing through a point with equal intercepts on both axes --/
structure EqualInterceptLine where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_point : a * 2 + b * 3 + c = 0
  equal_intercepts : a ≠ 0 ∧ b ≠ 0 → -c/a = -c/b

/-- The equations of the line passing through (2,3) with equal intercepts --/
theorem equal_intercept_line_equations :
  ∀ (l : EqualInterceptLine), 
  (l.a = 3 ∧ l.b = -2 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -5) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equations_l1137_113786


namespace NUMINAMATH_CALUDE_b_range_characterization_l1137_113703

theorem b_range_characterization :
  ∀ b : ℝ, (0 < b ∧ b ≤ 1/4) ↔ (b > 0 ∧ ∀ x : ℝ, |x - 5/4| < b → |x - 1| < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_b_range_characterization_l1137_113703


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l1137_113777

theorem cyclist_speed_ratio :
  ∀ (T₁ T₂ o₁ o₂ : ℝ),
  T₁ > 0 ∧ T₂ > 0 ∧ o₁ > 0 ∧ o₂ > 0 →
  o₁ + T₁ = o₂ + T₂ →
  T₁ = 2 * o₂ →
  T₂ = 4 * o₁ →
  T₁ / T₂ = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l1137_113777


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1137_113757

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + x)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 63 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1137_113757


namespace NUMINAMATH_CALUDE_pool_capacity_l1137_113724

/-- The capacity of a pool given three valves with specific flow rates. -/
theorem pool_capacity (v1 : ℝ) (r : ℝ) : 
  (v1 * 120 = r) →  -- First valve fills the pool in 2 hours
  ((v1 + (v1 + 50) + (v1 - 25)) * 36 = r) →  -- All valves fill the pool in 36 minutes
  (r = 9000) :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_l1137_113724


namespace NUMINAMATH_CALUDE_probability_of_y_selection_l1137_113702

theorem probability_of_y_selection (p_x p_both : ℝ) (h1 : p_x = 1/3) (h2 : p_both = 0.13333333333333333) : 
  ∃ p_y : ℝ, p_y = 0.4 ∧ p_both = p_x * p_y :=
sorry

end NUMINAMATH_CALUDE_probability_of_y_selection_l1137_113702


namespace NUMINAMATH_CALUDE_mode_is_two_hours_l1137_113734

/-- Represents the time spent on volunteer activities -/
inductive VolunteerTime
  | OneHour
  | OneAndHalfHours
  | TwoHours
  | TwoAndHalfHours
  | ThreeHours

/-- The number of students for each volunteer time category -/
def studentCount : VolunteerTime → Nat
  | VolunteerTime.OneHour => 20
  | VolunteerTime.OneAndHalfHours => 32
  | VolunteerTime.TwoHours => 38
  | VolunteerTime.TwoAndHalfHours => 8
  | VolunteerTime.ThreeHours => 2

/-- The total number of students -/
def totalStudents : Nat := 100

/-- The mode of the data set -/
def dataMode : VolunteerTime := VolunteerTime.TwoHours

theorem mode_is_two_hours :
  (∀ t : VolunteerTime, studentCount dataMode ≥ studentCount t) ∧
  dataMode = VolunteerTime.TwoHours :=
by sorry

end NUMINAMATH_CALUDE_mode_is_two_hours_l1137_113734


namespace NUMINAMATH_CALUDE_pencils_bought_with_promotion_l1137_113763

/-- Represents the number of pencils Petya's mom gave him money for -/
def pencils_mom_paid_for : ℕ := 49

/-- Represents the number of additional pencils Petya could buy with the promotion -/
def additional_pencils : ℕ := 12

/-- Represents the total number of pencils Petya could buy with the promotion -/
def total_pencils_bought : ℕ := pencils_mom_paid_for + additional_pencils

theorem pencils_bought_with_promotion :
  pencils_mom_paid_for = 49 ∧ 
  additional_pencils = 12 ∧
  total_pencils_bought = pencils_mom_paid_for + additional_pencils :=
by sorry

end NUMINAMATH_CALUDE_pencils_bought_with_promotion_l1137_113763


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1137_113723

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    a chord of the ellipse, and the midpoint of the chord,
    prove that the eccentricity of the ellipse is √5/5 -/
theorem ellipse_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hchord : ∀ x y : ℝ, x - y + 5 = 0 → ∃ t : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (hmidpoint : ∃ x1 y1 x2 y2 : ℝ, 
    (x1^2 / a^2) + (y1^2 / b^2) = 1 ∧ 
    (x2^2 / a^2) + (y2^2 / b^2) = 1 ∧ 
    x1 - y1 + 5 = 0 ∧ 
    x2 - y2 + 5 = 0 ∧ 
    (x1 + x2) / 2 = -4 ∧ 
    (y1 + y2) / 2 = 1) : 
  (Real.sqrt (1 - b^2 / a^2)) = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1137_113723


namespace NUMINAMATH_CALUDE_vector_at_negative_three_l1137_113766

def line_parameterization (t : ℝ) : ℝ × ℝ := sorry

theorem vector_at_negative_three :
  (∀ t : ℝ, ∃ x y : ℝ, line_parameterization t = (x, y)) →
  line_parameterization 1 = (2, 5) →
  line_parameterization 4 = (8, -7) →
  line_parameterization (-3) = (-6, 21) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_three_l1137_113766


namespace NUMINAMATH_CALUDE_dan_has_16_balloons_l1137_113748

/-- The number of red balloons that Fred has -/
def fred_balloons : ℕ := 10

/-- The number of red balloons that Sam has -/
def sam_balloons : ℕ := 46

/-- The total number of red balloons -/
def total_balloons : ℕ := 72

/-- The number of red balloons that Dan has -/
def dan_balloons : ℕ := total_balloons - (fred_balloons + sam_balloons)

theorem dan_has_16_balloons : dan_balloons = 16 := by
  sorry

end NUMINAMATH_CALUDE_dan_has_16_balloons_l1137_113748


namespace NUMINAMATH_CALUDE_solve_for_m_l1137_113783

def f (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + m

def g (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + 5*m

theorem solve_for_m :
  ∃ m : ℚ, 3 * (f m 5) = 2 * (g m 5) ∧ m = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l1137_113783


namespace NUMINAMATH_CALUDE_adam_final_amount_l1137_113722

def initial_amount : ℝ := 1025.25
def console_percentage : ℝ := 0.45
def euro_found : ℝ := 50
def exchange_rate : ℝ := 1.18
def allowance_percentage : ℝ := 0.10

theorem adam_final_amount :
  let amount_spent := initial_amount * console_percentage
  let money_left := initial_amount - amount_spent
  let euro_exchanged := euro_found * exchange_rate
  let money_after_exchange := money_left + euro_exchanged
  let allowance := initial_amount * allowance_percentage
  let final_amount := money_after_exchange + allowance
  final_amount = 725.4125 := by
  sorry

end NUMINAMATH_CALUDE_adam_final_amount_l1137_113722


namespace NUMINAMATH_CALUDE_minimum_occupied_seats_l1137_113742

/-- Represents a seating arrangement in a cinema row. -/
structure CinemaRow where
  total_seats : ℕ
  occupied_seats : ℕ

/-- Checks if a seating arrangement ensures the next person sits adjacent to someone. -/
def is_valid_arrangement (row : CinemaRow) : Prop :=
  ∀ i : ℕ, i < row.total_seats → 
    ∃ j : ℕ, j < row.total_seats ∧ 
      (j = i - 1 ∨ j = i + 1) ∧ 
      (∃ k : ℕ, k < row.occupied_seats ∧ j = k * 3)

/-- The theorem to be proved. -/
theorem minimum_occupied_seats :
  ∃ (row : CinemaRow), 
    row.total_seats = 150 ∧ 
    row.occupied_seats = 50 ∧ 
    is_valid_arrangement row ∧
    (∀ (other : CinemaRow), 
      other.total_seats = 150 → 
      is_valid_arrangement other → 
      other.occupied_seats ≥ 50) := by
  sorry

end NUMINAMATH_CALUDE_minimum_occupied_seats_l1137_113742


namespace NUMINAMATH_CALUDE_infection_spread_theorem_l1137_113713

/-- Represents the infection spread in a cube grid -/
structure InfectionSpread where
  edge : Nat
  t : Nat
  h_edge : edge = 2015
  h_t_range : 1 ≤ t ∧ t ≤ edge

/-- The minimum number of initially infected cells for possible complete infection -/
def min_cells_possible (is : InfectionSpread) : Nat :=
  is.t ^ 3

/-- The minimum number of initially infected cells for certain complete infection -/
def min_cells_certain (is : InfectionSpread) : Nat :=
  is.edge ^ 3 - (is.edge - (is.t - 1)) ^ 3 + 1

/-- Theorem stating the minimum number of cells for possible and certain infection -/
theorem infection_spread_theorem (is : InfectionSpread) :
  (min_cells_possible is = is.t ^ 3) ∧
  (min_cells_certain is = is.edge ^ 3 - (is.edge - (is.t - 1)) ^ 3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_infection_spread_theorem_l1137_113713


namespace NUMINAMATH_CALUDE_right_triangle_configurations_l1137_113758

def points_on_line : ℕ := 58

theorem right_triangle_configurations :
  let total_points := 2 * points_on_line
  let ways_hypotenuse_on_line := points_on_line.choose 2 * points_on_line
  let ways_leg_on_line := points_on_line * points_on_line
  ways_hypotenuse_on_line * 2 + ways_leg_on_line * 2 = 6724 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_configurations_l1137_113758


namespace NUMINAMATH_CALUDE_process_600_parts_l1137_113741

/-- Linear regression equation relating parts processed to time spent -/
def linear_regression (x : ℝ) : ℝ := 0.01 * x + 0.5

/-- Theorem stating that processing 600 parts takes 6.5 hours -/
theorem process_600_parts : linear_regression 600 = 6.5 := by sorry

end NUMINAMATH_CALUDE_process_600_parts_l1137_113741


namespace NUMINAMATH_CALUDE_area_of_absolute_value_equation_l1137_113712

def enclosed_area (f : ℝ × ℝ → ℝ) : ℝ := sorry

theorem area_of_absolute_value_equation :
  enclosed_area (fun (x, y) => |x| + |3 * y| + |x - y| - 20) = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_area_of_absolute_value_equation_l1137_113712


namespace NUMINAMATH_CALUDE_x_twelfth_power_l1137_113751

theorem x_twelfth_power (x : ℂ) : x + 1/x = 2 * Real.sqrt 2 → x^12 = -4096 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_l1137_113751


namespace NUMINAMATH_CALUDE_sugar_for_frosting_l1137_113719

theorem sugar_for_frosting (total_sugar : Real) (cake_sugar : Real) (h1 : total_sugar = 0.8) (h2 : cake_sugar = 0.2) :
  total_sugar - cake_sugar = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sugar_for_frosting_l1137_113719


namespace NUMINAMATH_CALUDE_distribute_4_3_l1137_113775

/-- The number of ways to distribute n distinct objects among k distinct groups,
    where each group must receive at least one object -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 4 distinct objects among 3 distinct groups,
    where each group must receive at least one object, results in 60 different ways -/
theorem distribute_4_3 : distribute 4 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_distribute_4_3_l1137_113775


namespace NUMINAMATH_CALUDE_gain_percent_problem_l1137_113731

/-- Calculate the gain percent given the gain in paise and the cost price in rupees. -/
def gain_percent (gain_paise : ℕ) (cost_price_rupees : ℕ) : ℚ :=
  (gain_paise : ℚ) / (cost_price_rupees * 100 : ℚ) * 100

/-- Theorem stating that the gain percent is 1% when the gain is 70 paise on a cost price of Rs. 70. -/
theorem gain_percent_problem : gain_percent 70 70 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_problem_l1137_113731


namespace NUMINAMATH_CALUDE_largest_common_term_under_1000_l1137_113711

/-- The first arithmetic progression {1, 4, 7, 10, ...} -/
def progression1 (n : ℕ) : ℕ := 1 + 3 * n

/-- The second arithmetic progression {5, 14, 23, 32, ...} -/
def progression2 (n : ℕ) : ℕ := 5 + 9 * n

/-- A term is common if it appears in both progressions -/
def is_common_term (a : ℕ) : Prop :=
  ∃ n m : ℕ, progression1 n = a ∧ progression2 m = a

theorem largest_common_term_under_1000 :
  (∀ a : ℕ, a < 1000 → is_common_term a → a ≤ 976) ∧
  is_common_term 976 :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_under_1000_l1137_113711
