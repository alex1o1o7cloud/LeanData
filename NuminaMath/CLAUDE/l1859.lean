import Mathlib

namespace NUMINAMATH_CALUDE_power_function_increasing_m_l1859_185914

/-- A function f is a power function if it can be written as f(x) = ax^n for some constants a and n, where a ≠ 0 -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- A function f is increasing on (0, +∞) if for any x1 < x2 in (0, +∞), f(x1) < f(x2) -/
def isIncreasingOnPositiveReals (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2

/-- The main theorem -/
theorem power_function_increasing_m (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (m^2 - m - 5) * x^m
  isPowerFunction f ∧ isIncreasingOnPositiveReals f → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_increasing_m_l1859_185914


namespace NUMINAMATH_CALUDE_circle_properties_l1859_185904

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line L
def line_L (x y : ℝ) : Prop := x - y = 2

theorem circle_properties :
  ∃ (center_x center_y radius : ℝ),
    -- The circle C can be represented in standard form
    (∀ x y, circle_C x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    -- The radius of C is 1
    radius = 1 ∧
    -- The distance from the center of C to the line L is √2
    (|center_x - center_y - 2| / Real.sqrt 2 = Real.sqrt 2) ∧
    -- The minimum distance from a point on C to the line L is √2 - 1
    (∃ min_dist : ℝ, min_dist = Real.sqrt 2 - 1 ∧
      ∀ x y, circle_C x y → |x - y - 2| / Real.sqrt 2 ≥ min_dist) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1859_185904


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_value_l1859_185912

/-- A geometric sequence with first term 2 and satisfying a₃a₅ = 4a₆² has a₃ = 1 -/
theorem geometric_sequence_a3_value (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n, a n = 2 * r^(n-1))  -- {aₙ} is a geometric sequence
  → a 1 = 2                          -- a₁ = 2
  → a 3 * a 5 = 4 * (a 6)^2          -- a₃a₅ = 4a₆²
  → a 3 = 1                          -- a₃ = 1
:= by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_value_l1859_185912


namespace NUMINAMATH_CALUDE_y_intercept_of_line_with_slope_3_and_x_intercept_4_l1859_185995

/-- A line is defined by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate of the point where the line crosses the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.slope * (-l.point.1) + l.point.2

theorem y_intercept_of_line_with_slope_3_and_x_intercept_4 :
  let l : Line := { slope := 3, point := (4, 0) }
  y_intercept l = -12 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_with_slope_3_and_x_intercept_4_l1859_185995


namespace NUMINAMATH_CALUDE_existence_of_complementary_sequences_l1859_185935

def s (x y : ℝ) : Set ℕ := {s | ∃ n : ℕ, s = ⌊n * x + y⌋}

theorem existence_of_complementary_sequences (r : ℚ) (hr : r > 1) :
  ∃ u v : ℝ, (s r 0 ∩ s u v = ∅) ∧ (s r 0 ∪ s u v = Set.univ) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_complementary_sequences_l1859_185935


namespace NUMINAMATH_CALUDE_cubic_function_property_l1859_185926

/-- Given a cubic function f(x) = ax³ + bx² + cx + d where f(1) = 4,
    prove that 12a - 6b + 3c - 2d = 40 -/
theorem cubic_function_property (a b c d : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^3 + b * x^2 + c * x + d)
  (h_f1 : f 1 = 4) :
  12 * a - 6 * b + 3 * c - 2 * d = 40 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1859_185926


namespace NUMINAMATH_CALUDE_thousandths_place_of_seven_thirty_seconds_l1859_185965

theorem thousandths_place_of_seven_thirty_seconds (n : ℕ) : 
  (7 : ℚ) / 32 = n / 1000 + (8 : ℚ) / 1000 + m / 10000 → n < 9 ∧ 0 ≤ m ∧ m < 10 :=
by sorry

end NUMINAMATH_CALUDE_thousandths_place_of_seven_thirty_seconds_l1859_185965


namespace NUMINAMATH_CALUDE_rectangle_segment_length_l1859_185959

/-- Given a rectangle with dimensions 10 units by 5 units, prove that the total length
    of segments in a new figure formed by removing three sides is 15 units. The remaining
    segments include two full heights and two parts of the width (3 units and 2 units). -/
theorem rectangle_segment_length :
  let original_width : ℕ := 10
  let original_height : ℕ := 5
  let remaining_width_part1 : ℕ := 3
  let remaining_width_part2 : ℕ := 2
  let total_length : ℕ := 2 * original_height + remaining_width_part1 + remaining_width_part2
  total_length = 15 := by sorry

end NUMINAMATH_CALUDE_rectangle_segment_length_l1859_185959


namespace NUMINAMATH_CALUDE_bags_difference_l1859_185900

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 8

/-- The number of bags Tiffany found the next day -/
def next_day_bags : ℕ := 7

/-- Theorem stating the difference in bags between Monday and the next day -/
theorem bags_difference : monday_bags - next_day_bags = 1 := by
  sorry

end NUMINAMATH_CALUDE_bags_difference_l1859_185900


namespace NUMINAMATH_CALUDE_train_passing_time_l1859_185986

/-- The time taken for a train to pass a telegraph post -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 90 →
  train_speed_kmh = 36 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1859_185986


namespace NUMINAMATH_CALUDE_average_of_ABC_l1859_185980

theorem average_of_ABC (A B C : ℚ) 
  (eq1 : 2023 * C - 4046 * A = 8092)
  (eq2 : 2023 * B - 6069 * A = 10115) :
  (A + B + C) / 3 = 2 * A + 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_ABC_l1859_185980


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_seven_l1859_185977

theorem no_solution_iff_k_eq_seven (k : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 7 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_seven_l1859_185977


namespace NUMINAMATH_CALUDE_sqrt_a_sqrt_a_eq_a_pow_three_fourths_l1859_185969

theorem sqrt_a_sqrt_a_eq_a_pow_three_fourths (a : ℝ) (h : a > 0) :
  Real.sqrt (a * Real.sqrt a) = a ^ (3/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_sqrt_a_eq_a_pow_three_fourths_l1859_185969


namespace NUMINAMATH_CALUDE_sum_A_B_linear_combo_A_B_diff_A_B_specific_l1859_185911

-- Define A and B as functions of a and b
def A (a b : ℚ) : ℚ := 4 * a^2 * b - 3 * a * b + b^2
def B (a b : ℚ) : ℚ := a^2 - 3 * a^2 * b + 3 * a * b - b^2

-- Theorem 1: A + B = a² + a²b
theorem sum_A_B (a b : ℚ) : A a b + B a b = a^2 + a^2 * b := by sorry

-- Theorem 2: 3A + 4B = 4a² + 3ab - b²
theorem linear_combo_A_B (a b : ℚ) : 3 * A a b + 4 * B a b = 4 * a^2 + 3 * a * b - b^2 := by sorry

-- Theorem 3: A - B = -63/8 when a = 2 and b = -1/4
theorem diff_A_B_specific : A 2 (-1/4) - B 2 (-1/4) = -63/8 := by sorry

end NUMINAMATH_CALUDE_sum_A_B_linear_combo_A_B_diff_A_B_specific_l1859_185911


namespace NUMINAMATH_CALUDE_smallest_n_with_19_odd_digit_squares_l1859_185931

/-- A function that returns true if a number has an odd number of digits, false otherwise -/
def has_odd_digits (n : ℕ) : Bool :=
  sorry

/-- A function that counts how many numbers from 1 to n have squares with an odd number of digits -/
def count_odd_digit_squares (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 44 is the smallest natural number N such that
    among the squares of integers from 1 to N, exactly 19 of them have an odd number of digits -/
theorem smallest_n_with_19_odd_digit_squares :
  ∀ n : ℕ, n < 44 → count_odd_digit_squares n < 19 ∧ count_odd_digit_squares 44 = 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_19_odd_digit_squares_l1859_185931


namespace NUMINAMATH_CALUDE_sequence_general_term_l1859_185925

/-- Given a sequence {aₙ} with sum of first n terms Sₙ = (2/3)n² - (1/3)n,
    prove that the general term is aₙ = (4/3)n - 1 -/
theorem sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ) 
    (h : ∀ n, S n = 2/3 * n^2 - 1/3 * n) :
  ∀ n, a n = 4/3 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1859_185925


namespace NUMINAMATH_CALUDE_max_missed_questions_to_pass_l1859_185924

theorem max_missed_questions_to_pass (total_questions : ℕ) (passing_percentage : ℚ) 
  (h1 : total_questions = 40)
  (h2 : passing_percentage = 75/100) : 
  ∃ (max_missed : ℕ), max_missed = 10 ∧ 
    (total_questions - max_missed : ℚ) / total_questions ≥ passing_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_missed_questions_to_pass_l1859_185924


namespace NUMINAMATH_CALUDE_permutation_count_l1859_185951

/-- The number of X's in the original string -/
def num_X : ℕ := 4

/-- The number of Y's in the original string -/
def num_Y : ℕ := 5

/-- The number of Z's in the original string -/
def num_Z : ℕ := 9

/-- The total length of the string -/
def total_length : ℕ := num_X + num_Y + num_Z

/-- The length of the first section where X is not allowed -/
def first_section : ℕ := 5

/-- The length of the middle section where Y is not allowed -/
def middle_section : ℕ := 6

/-- The length of the last section where Z is not allowed -/
def last_section : ℕ := 7

/-- The number of permutations satisfying the given conditions -/
def M : ℕ := sorry

theorem permutation_count : M % 1000 = 30 := by sorry

end NUMINAMATH_CALUDE_permutation_count_l1859_185951


namespace NUMINAMATH_CALUDE_line_point_sum_l1859_185976

/-- The line equation y = -5/3x + 15 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is where the line crosses the x-axis -/
def point_P : ℝ × ℝ := (9, 0)

/-- Point Q is where the line crosses the y-axis -/
def point_Q : ℝ × ℝ := (0, 15)

/-- Point T is on the line segment PQ -/
def point_T_on_PQ (r s : ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
  r = t * point_P.1 + (1 - t) * point_Q.1 ∧
  s = t * point_P.2 + (1 - t) * point_Q.2

/-- The area of triangle POQ is twice the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((point_P.1 - 0) * (point_Q.2 - 0) - (point_Q.1 - 0) * (point_P.2 - 0)) / 2 =
  2 * abs ((point_P.1 - 0) * (s - 0) - (r - 0) * (point_P.2 - 0)) / 2

theorem line_point_sum (r s : ℝ) :
  line_equation r s →
  point_T_on_PQ r s →
  area_condition r s →
  r + s = 12 := by sorry

end NUMINAMATH_CALUDE_line_point_sum_l1859_185976


namespace NUMINAMATH_CALUDE_total_lemonade_poured_l1859_185966

def first_intermission : ℝ := 0.25 + 0.125
def second_intermission : ℝ := 0.16666666666666666 + 0.08333333333333333 + 0.16666666666666666
def third_intermission : ℝ := 0.25 + 0.125
def fourth_intermission : ℝ := 0.3333333333333333 + 0.08333333333333333 + 0.16666666666666666

theorem total_lemonade_poured :
  first_intermission + second_intermission + third_intermission + fourth_intermission = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_total_lemonade_poured_l1859_185966


namespace NUMINAMATH_CALUDE_infinite_set_equal_digit_sum_l1859_185997

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a natural number contains zero in its decimal notation -/
def contains_zero (n : ℕ) : Prop := sorry

theorem infinite_set_equal_digit_sum (k : ℕ) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ t ∈ S, ¬contains_zero t ∧ sum_of_digits t = sum_of_digits (k * t) := by
  sorry

end NUMINAMATH_CALUDE_infinite_set_equal_digit_sum_l1859_185997


namespace NUMINAMATH_CALUDE_smallest_other_integer_l1859_185988

theorem smallest_other_integer (x : ℕ) (a b : ℕ) : 
  a = 45 →
  a > 0 →
  b > 0 →
  x > 0 →
  Nat.gcd a b = x + 5 →
  Nat.lcm a b = x * (x + 5) →
  a + b < 100 →
  ∃ (b_min : ℕ), b_min = 12 ∧ ∀ (b' : ℕ), b' ≠ a ∧ 
    Nat.gcd a b' = x + 5 ∧
    Nat.lcm a b' = x * (x + 5) ∧
    a + b' < 100 →
    b' ≥ b_min :=
sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l1859_185988


namespace NUMINAMATH_CALUDE_disinfectant_purchase_problem_l1859_185939

/-- The price difference between outdoor and indoor disinfectant -/
def price_difference : ℕ := 30

/-- The cost of 2 indoor and 3 outdoor disinfectant barrels -/
def sample_cost : ℕ := 340

/-- The total number of barrels to be purchased -/
def total_barrels : ℕ := 200

/-- The maximum total cost allowed -/
def max_cost : ℕ := 14000

/-- The price of indoor disinfectant -/
def indoor_price : ℕ := 50

/-- The price of outdoor disinfectant -/
def outdoor_price : ℕ := 80

/-- The minimum number of indoor disinfectant barrels to be purchased -/
def min_indoor_barrels : ℕ := 67

theorem disinfectant_purchase_problem :
  (outdoor_price = indoor_price + price_difference) ∧
  (2 * indoor_price + 3 * outdoor_price = sample_cost) ∧
  (∀ m : ℕ, m ≤ total_barrels →
    indoor_price * m + outdoor_price * (total_barrels - m) ≤ max_cost →
    m ≥ min_indoor_barrels) :=
by sorry

end NUMINAMATH_CALUDE_disinfectant_purchase_problem_l1859_185939


namespace NUMINAMATH_CALUDE_apex_high_debate_points_l1859_185968

theorem apex_high_debate_points :
  ∀ (total_points : ℚ),
  total_points > 0 →
  ∃ (remaining_points : ℕ),
  (1/5 : ℚ) * total_points + (1/3 : ℚ) * total_points + 12 + remaining_points = total_points ∧
  remaining_points ≤ 18 ∧
  remaining_points = 18 :=
by sorry

end NUMINAMATH_CALUDE_apex_high_debate_points_l1859_185968


namespace NUMINAMATH_CALUDE_power_sum_modulo_l1859_185948

theorem power_sum_modulo (n : ℕ) :
  (Nat.pow 7 2008 + Nat.pow 9 2008) % 64 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_modulo_l1859_185948


namespace NUMINAMATH_CALUDE_am_gm_inequality_for_two_l1859_185908

theorem am_gm_inequality_for_two (x : ℝ) (hx : x > 0) :
  x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_for_two_l1859_185908


namespace NUMINAMATH_CALUDE_factorization_sum_l1859_185945

theorem factorization_sum (a b c d e f g h j k : ℤ) :
  (∃ (x y : ℝ), 27 * x^6 - 512 * y^6 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2)) →
  a + b + c + d + e + f + g + h + j + k = 152 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l1859_185945


namespace NUMINAMATH_CALUDE_quotient_base4_l1859_185973

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a decimal number to its base 4 representation -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec helper (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else helper (m / 4) ((m % 4) :: acc)
  helper n []

/-- Theorem: The quotient of 1213₄ divided by 13₄ is equal to 32₄ -/
theorem quotient_base4 :
  let a := base4ToDecimal [3, 1, 2, 1]  -- 1213₄
  let b := base4ToDecimal [3, 1]        -- 13₄
  decimalToBase4 (a / b) = [2, 3]       -- 32₄
  := by sorry

end NUMINAMATH_CALUDE_quotient_base4_l1859_185973


namespace NUMINAMATH_CALUDE_bag_of_balls_l1859_185958

theorem bag_of_balls (white green yellow red purple : ℕ) 
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 8)
  (h4 : red = 5)
  (h5 : purple = 7)
  (h6 : (white + green + yellow : ℝ) / (white + green + yellow + red + purple) = 0.8) :
  white + green + yellow + red + purple = 60 := by
  sorry

end NUMINAMATH_CALUDE_bag_of_balls_l1859_185958


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1859_185906

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 2 → x > 3)) ↔ (∃ x : ℝ, x > 2 ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1859_185906


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1859_185974

/-- The x-intercept of the line 4x + 7y = 28 is the point (7,0) -/
theorem x_intercept_of_line (x y : ℚ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1859_185974


namespace NUMINAMATH_CALUDE_class_composition_solution_l1859_185994

/-- Represents the class composition problem --/
structure ClassComposition where
  total_students : ℕ
  girls : ℕ
  boys : ℕ

/-- Checks if the given class composition satisfies the problem conditions --/
def satisfies_conditions (c : ClassComposition) : Prop :=
  c.total_students = c.girls + c.boys ∧
  c.girls * 2 = c.boys * 3 ∧
  (c.total_students * 2 - 150 = c.girls * 5)

/-- The theorem stating the solution to the class composition problem --/
theorem class_composition_solution :
  ∃ c : ClassComposition, c.total_students = 300 ∧ c.girls = 180 ∧ c.boys = 120 ∧
  satisfies_conditions c := by
  sorry

#check class_composition_solution

end NUMINAMATH_CALUDE_class_composition_solution_l1859_185994


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1859_185971

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 3 x ≥ x + 9} = {x : ℝ | x < -11/3 ∨ x > 7} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x ∈ (Set.Icc 0 1), f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1859_185971


namespace NUMINAMATH_CALUDE_optimal_price_and_quantity_l1859_185970

/-- Represents the sales and pricing model for a product -/
structure SalesModel where
  initialPurchasePrice : ℝ
  initialSellingPrice : ℝ
  initialSalesVolume : ℝ
  priceElasticity : ℝ
  targetProfit : ℝ
  maxCost : ℝ

/-- Calculates the sales volume for a given selling price -/
def salesVolume (model : SalesModel) (sellingPrice : ℝ) : ℝ :=
  model.initialSalesVolume - model.priceElasticity * (sellingPrice - model.initialSellingPrice)

/-- Calculates the profit for a given selling price -/
def profit (model : SalesModel) (sellingPrice : ℝ) : ℝ :=
  (sellingPrice - model.initialPurchasePrice) * (salesVolume model sellingPrice)

/-- Calculates the cost for a given selling price -/
def cost (model : SalesModel) (sellingPrice : ℝ) : ℝ :=
  model.initialPurchasePrice * (salesVolume model sellingPrice)

/-- Theorem stating that the optimal selling price and purchase quantity satisfy the constraints -/
theorem optimal_price_and_quantity (model : SalesModel) 
  (h_model : model = { 
    initialPurchasePrice := 40,
    initialSellingPrice := 50,
    initialSalesVolume := 500,
    priceElasticity := 10,
    targetProfit := 8000,
    maxCost := 10000
  }) :
  ∃ (optimalPrice optimalQuantity : ℝ),
    optimalPrice = 80 ∧
    optimalQuantity = 200 ∧
    profit model optimalPrice = model.targetProfit ∧
    cost model optimalPrice ≤ model.maxCost :=
  sorry

end NUMINAMATH_CALUDE_optimal_price_and_quantity_l1859_185970


namespace NUMINAMATH_CALUDE_solve_system_l1859_185934

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 7) (eq2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1859_185934


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1859_185905

theorem absolute_value_equation_solution :
  ∀ x : ℚ, |x - 5| = 3*x + 6 ↔ x = -1/4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1859_185905


namespace NUMINAMATH_CALUDE_f_decreasing_range_l1859_185942

/-- A piecewise function f(x) defined on the real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

/-- The theorem stating the range of 'a' for which f is decreasing on ℝ. -/
theorem f_decreasing_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) ↔ 1/7 ≤ a ∧ a < 1/3 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_range_l1859_185942


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l1859_185903

/-- Given a geometric progression with the first three terms, find the fourth term -/
theorem fourth_term_of_geometric_progression (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3^(1/4 : ℝ)) 
    (h₂ : a₂ = 3^(1/6 : ℝ)) (h₃ : a₃ = 3^(1/12 : ℝ)) : 
  ∃ (a₄ : ℝ), a₄ = (a₃ * a₂) / a₁ ∧ a₄ = 1 := by
sorry


end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l1859_185903


namespace NUMINAMATH_CALUDE_square_root_of_49_l1859_185922

theorem square_root_of_49 : 
  {x : ℝ | x^2 = 49} = {7, -7} := by sorry

end NUMINAMATH_CALUDE_square_root_of_49_l1859_185922


namespace NUMINAMATH_CALUDE_range_of_a_squared_minus_2b_l1859_185909

/-- A quadratic function with two real roots in [0, 1] -/
structure QuadraticWithRootsInUnitInterval where
  a : ℝ
  b : ℝ
  has_two_roots_in_unit_interval : ∃ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ 1 ∧ 
    x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0

/-- The range of a^2 - 2b for quadratic functions with roots in [0, 1] -/
theorem range_of_a_squared_minus_2b (f : QuadraticWithRootsInUnitInterval) :
  ∃ (z : ℝ), z = f.a^2 - 2*f.b ∧ 0 ≤ z ∧ z ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_squared_minus_2b_l1859_185909


namespace NUMINAMATH_CALUDE_total_height_increase_four_centuries_l1859_185937

/-- Represents the increase in height per decade for a specific plant species -/
def height_increase_per_decade : ℝ := 75

/-- Represents the number of decades in 4 centuries -/
def decades_in_four_centuries : ℕ := 40

/-- Theorem: The total increase in height over 4 centuries is 3000 meters -/
theorem total_height_increase_four_centuries : 
  height_increase_per_decade * (decades_in_four_centuries : ℝ) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_total_height_increase_four_centuries_l1859_185937


namespace NUMINAMATH_CALUDE_sequence_fifth_term_l1859_185963

theorem sequence_fifth_term (a : ℕ → ℤ) :
  (∀ n : ℕ, a n = 4 * n - 3) →
  a 5 = 17 := by
sorry

end NUMINAMATH_CALUDE_sequence_fifth_term_l1859_185963


namespace NUMINAMATH_CALUDE_problem_statement_l1859_185920

theorem problem_statement (a b : ℝ) (h1 : a * b = -3) (h2 : a + b = 2) :
  a^2 * b + a * b^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1859_185920


namespace NUMINAMATH_CALUDE_security_deposit_is_1110_l1859_185940

/-- Calculates the security deposit for a cabin rental -/
def calculate_security_deposit (daily_rate : ℚ) (duration : ℕ) (pet_fee : ℚ) 
  (service_fee_rate : ℚ) (deposit_rate : ℚ) : ℚ :=
  let subtotal := daily_rate * duration + pet_fee
  let service_fee := service_fee_rate * subtotal
  let total := subtotal + service_fee
  deposit_rate * total

/-- Theorem stating that the security deposit for the given conditions is $1110.00 -/
theorem security_deposit_is_1110 :
  calculate_security_deposit 125 14 100 (1/5) (1/2) = 1110 := by
  sorry

#eval calculate_security_deposit 125 14 100 (1/5) (1/2)

end NUMINAMATH_CALUDE_security_deposit_is_1110_l1859_185940


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l1859_185916

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 + a 3 = 10 →
  a 4 + a 6 = 5/4 →
  ∃ (q : ℝ), ∀ n : ℕ, a n = 2^(4-n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l1859_185916


namespace NUMINAMATH_CALUDE_function_domain_implies_m_range_l1859_185927

/-- The function f(x) = √(mx² - (1-m)x + m) has domain R if and only if m ≥ 1/3 -/
theorem function_domain_implies_m_range (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (m * x^2 - (1 - m) * x + m)) ↔ m ≥ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_implies_m_range_l1859_185927


namespace NUMINAMATH_CALUDE_coeff_x2y2_is_168_l1859_185975

/-- The coefficient of x^2y^2 in the expansion of ((1+x)^8(1+y)^4) -/
def coeff_x2y2 : ℕ :=
  (Nat.choose 8 2) * (Nat.choose 4 2)

/-- Theorem stating that the coefficient of x^2y^2 in ((1+x)^8(1+y)^4) is 168 -/
theorem coeff_x2y2_is_168 : coeff_x2y2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x2y2_is_168_l1859_185975


namespace NUMINAMATH_CALUDE_cos_2x_at_min_y_l1859_185936

theorem cos_2x_at_min_y (x : ℝ) : 
  let y := 2 * (Real.sin x)^6 + (Real.cos x)^6
  (∀ z : ℝ, y ≤ 2 * (Real.sin z)^6 + (Real.cos z)^6) →
  Real.cos (2 * x) = 3 - 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cos_2x_at_min_y_l1859_185936


namespace NUMINAMATH_CALUDE_total_area_of_three_triangles_l1859_185960

theorem total_area_of_three_triangles (base height : ℝ) (h1 : base = 40) (h2 : height = 20) :
  3 * (1/2 * base * height) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_area_of_three_triangles_l1859_185960


namespace NUMINAMATH_CALUDE_roots_and_coefficients_l1859_185961

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the roots
def is_root (a b c x : ℝ) : Prop := quadratic_equation a b c x

-- Theorem statement
theorem roots_and_coefficients (a b c X₁ X₂ : ℝ) 
  (ha : a ≠ 0) 
  (hX₁ : is_root a b c X₁) 
  (hX₂ : is_root a b c X₂) 
  (hX₁₂ : X₁ ≠ X₂) : 
  (X₁ + X₂ = -b / a) ∧ (X₁ * X₂ = c / a) := by
  sorry

end NUMINAMATH_CALUDE_roots_and_coefficients_l1859_185961


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1859_185967

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

theorem intersection_A_complement_B : A ∩ (U \ B) = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1859_185967


namespace NUMINAMATH_CALUDE_candy_distribution_l1859_185902

/-- The number of distinct pieces of candy --/
def n : ℕ := 8

/-- The number of bags --/
def k : ℕ := 3

/-- The number of ways to distribute n distinct objects into k groups,
    where each group must have at least one object --/
def distribute_distinct (n k : ℕ) : ℕ :=
  (n - k + 1).choose (k - 1) * n.factorial

theorem candy_distribution :
  distribute_distinct n k = 846720 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1859_185902


namespace NUMINAMATH_CALUDE_add_like_terms_l1859_185918

theorem add_like_terms (a : ℝ) : 3 * a + 2 * a = 5 * a := by sorry

end NUMINAMATH_CALUDE_add_like_terms_l1859_185918


namespace NUMINAMATH_CALUDE_bakery_purchase_maximization_l1859_185923

/-- Represents the problem of maximizing purchases at a bakery --/
theorem bakery_purchase_maximization 
  (total_money : ℚ)
  (pastry_cost : ℚ)
  (coffee_cost : ℚ)
  (discount : ℚ)
  (discount_threshold : ℕ)
  (h1 : total_money = 50)
  (h2 : pastry_cost = 6)
  (h3 : coffee_cost = (3/2))
  (h4 : discount = (1/2))
  (h5 : discount_threshold = 5) :
  ∃ (pastries coffee : ℕ),
    (pastries > discount_threshold → 
      pastries * (pastry_cost - discount) + coffee * coffee_cost ≤ total_money) ∧
    (pastries ≤ discount_threshold → 
      pastries * pastry_cost + coffee * coffee_cost ≤ total_money) ∧
    pastries + coffee = 9 ∧
    ∀ (p c : ℕ), 
      ((p > discount_threshold → 
        p * (pastry_cost - discount) + c * coffee_cost ≤ total_money) ∧
      (p ≤ discount_threshold → 
        p * pastry_cost + c * coffee_cost ≤ total_money)) →
      p + c ≤ 9 := by
sorry


end NUMINAMATH_CALUDE_bakery_purchase_maximization_l1859_185923


namespace NUMINAMATH_CALUDE_books_lost_during_move_phil_books_lost_l1859_185941

theorem books_lost_during_move (initial_books : ℕ) (pages_per_book : ℕ) (pages_left : ℕ) : ℕ :=
  let total_pages := initial_books * pages_per_book
  let pages_lost := total_pages - pages_left
  pages_lost / pages_per_book

theorem phil_books_lost :
  books_lost_during_move 10 100 800 = 2 := by
  sorry

end NUMINAMATH_CALUDE_books_lost_during_move_phil_books_lost_l1859_185941


namespace NUMINAMATH_CALUDE_unique_triple_l1859_185910

theorem unique_triple : 
  ∃! (x y z : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x^2 + y - z = 100 ∧ 
    x + y^2 - z = 124 ∧
    x = 12 ∧ y = 13 ∧ z = 57 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l1859_185910


namespace NUMINAMATH_CALUDE_problem_solution_l1859_185929

theorem problem_solution (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 7 * a + 2 * b = 54) :
  a + b = -103 / 31 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1859_185929


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l1859_185933

/-- Represents a 6x6x6 cube composed of unit cubes -/
structure Cube :=
  (size : Nat)
  (total_units : Nat)
  (painted_per_face : Nat)
  (unpainted_columns : Nat)
  (unpainted_rows : Nat)

/-- The number of unpainted unit cubes in the cube -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - (c.painted_per_face * 6 - 24)

/-- Theorem stating the number of unpainted cubes in the specific cube configuration -/
theorem unpainted_cubes_count (c : Cube) 
  (h1 : c.size = 6)
  (h2 : c.total_units = 216)
  (h3 : c.painted_per_face = 10)
  (h4 : c.unpainted_columns = 2)
  (h5 : c.unpainted_rows = 2) :
  unpainted_cubes c = 168 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_count_l1859_185933


namespace NUMINAMATH_CALUDE_unique_a_divisibility_l1859_185946

theorem unique_a_divisibility (a : ℤ) (h1 : 0 < a) (h2 : a < 13) 
  (h3 : (13 : ℤ) ∣ (53^2017 + a)) : a = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_divisibility_l1859_185946


namespace NUMINAMATH_CALUDE_inequality_solution_l1859_185952

theorem inequality_solution (x : ℝ) : 
  (2 ≤ |3*x - 6| ∧ |3*x - 6| ≤ 15) ↔ 
  (x ∈ Set.Icc (-3) (4/3) ∪ Set.Icc (8/3) 7) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1859_185952


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l1859_185984

/-- Given a grocery store inventory, calculate the number of diet soda bottles -/
theorem diet_soda_bottles (total : ℕ) (regular : ℕ) (h1 : total = 17) (h2 : regular = 9) :
  total - regular = 8 := by
  sorry

end NUMINAMATH_CALUDE_diet_soda_bottles_l1859_185984


namespace NUMINAMATH_CALUDE_product_modulo_25_l1859_185928

theorem product_modulo_25 (m : ℕ) (h1 : 65 * 76 * 87 ≡ m [ZMOD 25]) (h2 : m < 25) : m = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_25_l1859_185928


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1859_185979

def U : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {y | ∃ x ∈ U, y = |x|}

theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1859_185979


namespace NUMINAMATH_CALUDE_product_103_97_l1859_185992

theorem product_103_97 : 103 * 97 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_103_97_l1859_185992


namespace NUMINAMATH_CALUDE_smallest_valid_survey_size_l1859_185953

def is_valid_survey_size (N : ℕ) : Prop :=
  (N * 1 / 10 : ℚ).num % (N * 1 / 10 : ℚ).den = 0 ∧
  (N * 3 / 10 : ℚ).num % (N * 3 / 10 : ℚ).den = 0 ∧
  (N * 2 / 5 : ℚ).num % (N * 2 / 5 : ℚ).den = 0

theorem smallest_valid_survey_size :
  ∃ (N : ℕ), N > 0 ∧ is_valid_survey_size N ∧ ∀ (M : ℕ), M > 0 ∧ is_valid_survey_size M → N ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_survey_size_l1859_185953


namespace NUMINAMATH_CALUDE_perpendicular_vectors_dot_product_l1859_185999

/-- Given two vectors m and n in ℝ², where m = (2, 5) and n = (-5, t),
    if m is perpendicular to n, then (m + n) · (m - 2n) = -29 -/
theorem perpendicular_vectors_dot_product (t : ℝ) :
  let m : Fin 2 → ℝ := ![2, 5]
  let n : Fin 2 → ℝ := ![-5, t]
  (m • n = 0) →  -- m is perpendicular to n
  (m + n) • (m - 2 • n) = -29 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_dot_product_l1859_185999


namespace NUMINAMATH_CALUDE_not_universal_quantifier_negation_equivalence_not_necessary_not_sufficient_sufficient_not_necessary_l1859_185978

-- Define the proposition
def P : Prop := ∃ x : ℝ, x^2 + x + 1 = 0

-- Statement 1
theorem not_universal_quantifier : ¬(∀ x : ℝ, x^2 + x + 1 = 0) := by sorry

-- Statement 2
theorem negation_equivalence : 
  (¬∃ x : ℝ, x + 1 ≤ 2) ↔ (∀ x : ℝ, x + 1 > 2) := by sorry

-- Statement 3
theorem not_necessary_not_sufficient (A B : Set ℝ) :
  ¬(∀ x : ℝ, x ∈ A → x ∈ A ∩ B) ∧ ∀ x : ℝ, x ∈ A ∩ B → x ∈ A := by sorry

-- Statement 4
theorem sufficient_not_necessary :
  (∀ x : ℝ, x > 3 → x^2 > 9) ∧ ¬(∀ x : ℝ, x^2 > 9 → x > 3) := by sorry

end NUMINAMATH_CALUDE_not_universal_quantifier_negation_equivalence_not_necessary_not_sufficient_sufficient_not_necessary_l1859_185978


namespace NUMINAMATH_CALUDE_swimmer_speed_proof_l1859_185993

def swimmer_problem (distance : ℝ) (current_speed : ℝ) (time : ℝ) : Prop :=
  let still_water_speed := (distance / time) + current_speed
  still_water_speed = 3

theorem swimmer_speed_proof :
  swimmer_problem 8 1.4 5 :=
sorry

end NUMINAMATH_CALUDE_swimmer_speed_proof_l1859_185993


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l1859_185991

/-- Given three lines that intersect at the same point, prove the value of k. -/
theorem intersection_of_three_lines (x y : ℝ) (k : ℝ) : 
  y = -4 * x + 2 ∧ 
  y = 3 * x - 18 ∧ 
  y = 7 * x + k 
  → k = -206 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l1859_185991


namespace NUMINAMATH_CALUDE_container_volume_ratio_l1859_185932

/-- Theorem: Container Volume Ratio
Given two containers where the first is 3/7 full and transfers all its water to the second,
making it 2/3 full, the ratio of the volume of the first container to the volume of the second
container is 14/9.
-/
theorem container_volume_ratio (container1 container2 : ℝ) :
  container1 > 0 ∧ container2 > 0 →  -- Ensure containers have positive volume
  (3 / 7 : ℝ) * container1 = (2 / 3 : ℝ) * container2 → -- Water transfer equation
  container1 / container2 = 14 / 9 := by
sorry


end NUMINAMATH_CALUDE_container_volume_ratio_l1859_185932


namespace NUMINAMATH_CALUDE_math_books_count_l1859_185921

/-- Proves that the number of math books bought is 60 given the specified conditions -/
theorem math_books_count (total_books : ℕ) (math_book_price history_book_price : ℕ) (total_price : ℕ) :
  total_books = 90 →
  math_book_price = 4 →
  history_book_price = 5 →
  total_price = 390 →
  ∃ (math_books : ℕ), math_books = 60 ∧ 
    math_books + (total_books - math_books) = total_books ∧
    math_book_price * math_books + history_book_price * (total_books - math_books) = total_price :=
by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l1859_185921


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l1859_185907

def original_width : ℕ := 5
def original_height : ℕ := 6
def original_black_tiles : ℕ := 12
def original_white_tiles : ℕ := 18
def border_width : ℕ := 1

def extended_width : ℕ := original_width + 2 * border_width
def extended_height : ℕ := original_height + 2 * border_width

def total_extended_tiles : ℕ := extended_width * extended_height
def new_white_tiles : ℕ := total_extended_tiles - (original_width * original_height)
def total_white_tiles : ℕ := original_white_tiles + new_white_tiles

theorem extended_pattern_ratio :
  (original_black_tiles : ℚ) / total_white_tiles = 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l1859_185907


namespace NUMINAMATH_CALUDE_total_speech_time_l1859_185938

def speech_time (outline_time writing_time rewrite_time practice_time break1_time break2_time : ℝ) : ℝ :=
  outline_time + writing_time + rewrite_time + practice_time + break1_time + break2_time

theorem total_speech_time :
  let outline_time : ℝ := 30
  let break1_time : ℝ := 10
  let writing_time : ℝ := outline_time + 28
  let rewrite_time : ℝ := 15
  let break2_time : ℝ := 5
  let practice_time : ℝ := (writing_time + rewrite_time) / 2
  speech_time outline_time writing_time rewrite_time practice_time break1_time break2_time = 154.5 := by
  sorry

end NUMINAMATH_CALUDE_total_speech_time_l1859_185938


namespace NUMINAMATH_CALUDE_mary_initial_nickels_l1859_185964

/-- The number of nickels Mary initially had -/
def initial_nickels : ℕ := sorry

/-- The number of nickels Mary's dad gave her -/
def nickels_from_dad : ℕ := 5

/-- The total number of nickels Mary has now -/
def total_nickels : ℕ := 12

/-- Theorem stating that Mary initially had 7 nickels -/
theorem mary_initial_nickels : 
  initial_nickels = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_initial_nickels_l1859_185964


namespace NUMINAMATH_CALUDE_integer_solutions_equation_l1859_185962

theorem integer_solutions_equation : 
  {(x, y) : ℤ × ℤ | 3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3} = 
  {(0, 0), (6, 6), (-6, -6)} := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_equation_l1859_185962


namespace NUMINAMATH_CALUDE_triangle_shape_l1859_185913

theorem triangle_shape (A : ℝ) (hA : 0 < A ∧ A < π) 
  (h : Real.sin A + Real.cos A = 12/25) : A > π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1859_185913


namespace NUMINAMATH_CALUDE_max_prism_volume_in_hexagonal_pyramid_l1859_185947

/-- Represents a regular hexagonal pyramid -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_leg_length : ℝ

/-- Represents a right square prism -/
structure SquarePrism where
  side_length : ℝ

/-- Calculates the volume of a right square prism -/
def prism_volume (p : SquarePrism) : ℝ := p.side_length ^ 3

/-- Theorem stating the maximum volume of the square prism within the hexagonal pyramid -/
theorem max_prism_volume_in_hexagonal_pyramid 
  (pyramid : HexagonalPyramid) 
  (prism : SquarePrism) 
  (h1 : pyramid.base_side_length = 2) 
  (h2 : prism.side_length ≤ pyramid.base_side_length) 
  (h3 : prism.side_length > 0) :
  prism_volume prism ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_prism_volume_in_hexagonal_pyramid_l1859_185947


namespace NUMINAMATH_CALUDE_A_minus_2B_A_minus_2B_specific_y_value_when_independent_l1859_185972

/-- Given algebraic expressions A and B -/
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y

def B (x y : ℝ) : ℝ := x^2 - x * y + x

/-- Theorem 1: A - 2B = 5xy - 2x + 2y -/
theorem A_minus_2B (x y : ℝ) : A x y - 2 * B x y = 5 * x * y - 2 * x + 2 * y := by sorry

/-- Theorem 2: A - 2B = -7 when x = -1 and y = 3 -/
theorem A_minus_2B_specific : A (-1) 3 - 2 * B (-1) 3 = -7 := by sorry

/-- Theorem 3: y = 2/5 when A - 2B is independent of x -/
theorem y_value_when_independent (y : ℝ) :
  (∀ x, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2/5 := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_A_minus_2B_specific_y_value_when_independent_l1859_185972


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_same_plane_l1859_185949

-- Define a type for planes
structure Plane where
  -- We don't need to specify the exact properties of a plane for this problem

-- Define a perpendicular relation between planes
def perpendicular (p q : Plane) : Prop := sorry

-- Define a parallel relation between planes
def parallel (p q : Plane) : Prop := sorry

-- Define an intersecting relation between planes
def intersecting (p q : Plane) : Prop := sorry

-- State the theorem
theorem planes_perpendicular_to_same_plane 
  (α β γ : Plane) 
  (h1 : α ≠ β) (h2 : α ≠ γ) (h3 : β ≠ γ) 
  (h4 : perpendicular α γ) (h5 : perpendicular β γ) : 
  parallel α β ∨ intersecting α β := by
  sorry


end NUMINAMATH_CALUDE_planes_perpendicular_to_same_plane_l1859_185949


namespace NUMINAMATH_CALUDE_range_of_g_l1859_185987

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 29 ≤ g x ∧ g x ≤ 93 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l1859_185987


namespace NUMINAMATH_CALUDE_diagonal_angle_is_45_degrees_l1859_185957

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define the angle formed by a diagonal and a side of a square
def diagonal_angle (s : Square) : ℝ := sorry

-- Theorem statement
theorem diagonal_angle_is_45_degrees (s : Square) : 
  diagonal_angle s = 45 := by sorry

end NUMINAMATH_CALUDE_diagonal_angle_is_45_degrees_l1859_185957


namespace NUMINAMATH_CALUDE_motorboat_stream_speed_l1859_185955

/-- Proves that the speed of the stream is 3 kmph given the conditions of the motorboat problem -/
theorem motorboat_stream_speed 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (total_time : ℝ) 
  (h1 : boat_speed = 21) 
  (h2 : distance = 72) 
  (h3 : total_time = 7) :
  ∃ (stream_speed : ℝ), 
    stream_speed = 3 ∧ 
    distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed) = total_time :=
by
  sorry

end NUMINAMATH_CALUDE_motorboat_stream_speed_l1859_185955


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l1859_185990

/-- The number of distinct arrangements of beads on a bracelet -/
def distinct_bracelet_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem: The number of distinct arrangements of 8 beads on a bracelet is 2520 -/
theorem eight_bead_bracelet_arrangements :
  distinct_bracelet_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l1859_185990


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l1859_185930

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define shapes
inductive Shape
  | Square
  | Circle

-- Define colors
inductive Color
  | Black
  | White

-- Define a coloring function
def ColoringFunction := Point → Color

-- Define similarity between sets of points
def SimilarSets (s1 s2 : Set Point) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ ∀ (p1 p2 : Point), p1 ∈ s1 → p2 ∈ s2 →
    ∃ (q1 q2 : Point), q1 ∈ s2 ∧ q2 ∈ s2 ∧
      (q1.x - q2.x)^2 + (q1.y - q2.y)^2 = k * ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem exists_valid_coloring :
  ∃ (f : Shape → ColoringFunction),
    (∀ (s : Shape) (p : Point), f s p = Color.Black ∨ f s p = Color.White) ∧
    SimilarSets {p | f Shape.Square p = Color.White} {p | f Shape.Circle p = Color.White} ∧
    SimilarSets {p | f Shape.Square p = Color.Black} {p | f Shape.Circle p = Color.Black} :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l1859_185930


namespace NUMINAMATH_CALUDE_product_of_decimals_l1859_185950

theorem product_of_decimals : (0.05 : ℝ) * 0.3 * 2 = 0.03 := by sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1859_185950


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l1859_185919

theorem partial_fraction_decomposition_sum (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ (x : ℝ), x ≠ p ∧ x ≠ q ∧ x ≠ r → 
    1 / (x^3 - 15*x^2 + 50*x - 56) = A / (x - p) + B / (x - q) + C / (x - r)) →
  (x^3 - 15*x^2 + 50*x - 56 = (x - p) * (x - q) * (x - r)) →
  1 / A + 1 / B + 1 / C = 225 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l1859_185919


namespace NUMINAMATH_CALUDE_decreasing_linear_function_l1859_185983

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f y < f x

theorem decreasing_linear_function (k : ℝ) :
  is_decreasing (λ x : ℝ => (k + 1) * x) → k < -1 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_l1859_185983


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1859_185985

/-- 
For a quadratic equation (k-1)x^2 + 4x + 2 = 0 to have real roots,
k must satisfy the condition k ≤ 3 and k ≠ 1.
-/
theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 4 * x + 2 = 0) ↔ (k ≤ 3 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1859_185985


namespace NUMINAMATH_CALUDE_decimal_digit_17_99_l1859_185915

/-- The fraction we're examining -/
def f : ℚ := 17 / 99

/-- The position of the digit we're looking for -/
def n : ℕ := 150

/-- Function to get the nth digit after the decimal point in the decimal representation of a rational number -/
noncomputable def nth_decimal_digit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 150th digit after the decimal point in 17/99 is 7 -/
theorem decimal_digit_17_99 : nth_decimal_digit f n = 7 := by sorry

end NUMINAMATH_CALUDE_decimal_digit_17_99_l1859_185915


namespace NUMINAMATH_CALUDE_quadratic_inequality_relations_l1859_185917

/-- 
Given a quadratic inequality ax^2 - bx + c > 0 with solution set (-1, 2),
prove the relationships between a, b, and c.
-/
theorem quadratic_inequality_relations (a b c : ℝ) : 
  (∀ x : ℝ, ax^2 - b*x + c > 0 ↔ -1 < x ∧ x < 2) →
  (a < 0 ∧ b = a ∧ c = -2*a) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_relations_l1859_185917


namespace NUMINAMATH_CALUDE_no_intersection_points_l1859_185998

theorem no_intersection_points (x y : ℝ) : 
  ¬∃ x y, (y = 3 * x^2 - 4 * x + 5) ∧ (y = -x^2 + 6 * x - 8) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_points_l1859_185998


namespace NUMINAMATH_CALUDE_minimum_percentage_bad_work_l1859_185989

theorem minimum_percentage_bad_work (total_works : ℝ) (h_total_positive : total_works > 0) :
  let bad_works := 0.2 * total_works
  let good_works := 0.8 * total_works
  let misclassified_good := 0.1 * good_works
  let misclassified_bad := 0.1 * bad_works
  let rechecked_works := bad_works - misclassified_bad + misclassified_good
  let actual_bad_rechecked := bad_works - misclassified_bad
  ⌊(actual_bad_rechecked / rechecked_works * 100)⌋ = 69 :=
by sorry

end NUMINAMATH_CALUDE_minimum_percentage_bad_work_l1859_185989


namespace NUMINAMATH_CALUDE_max_value_log_expression_l1859_185943

open Real

theorem max_value_log_expression (x : ℝ) (h : x > -1) :
  ∃ M, M = -2 ∧ 
  (log (x + 1 / (x + 1) + 3) / log (1/2) ≤ M) ∧
  ∃ x₀, x₀ > -1 ∧ log (x₀ + 1 / (x₀ + 1) + 3) / log (1/2) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_log_expression_l1859_185943


namespace NUMINAMATH_CALUDE_max_attendance_l1859_185901

/-- Represents the number of students that can attend an event --/
structure EventAttendance where
  boys : ℕ
  girls : ℕ

/-- Represents the capacities of the three auditoriums --/
structure AuditoriumCapacities where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Checks if the attendance satisfies the given conditions --/
def satisfiesConditions (attendance : EventAttendance) (capacities : AuditoriumCapacities) : Prop :=
  -- The ratio of boys to girls is 7:11
  11 * attendance.boys = 7 * attendance.girls
  -- There are 72 more girls than boys
  ∧ attendance.girls = attendance.boys + 72
  -- The total attendance doesn't exceed any individual auditorium's capacity
  ∧ attendance.boys + attendance.girls ≤ capacities.A
  ∧ attendance.boys + attendance.girls ≤ capacities.B
  ∧ attendance.boys + attendance.girls ≤ capacities.C

/-- The main theorem stating the maximum number of students that can attend --/
theorem max_attendance (capacities : AuditoriumCapacities)
    (hA : capacities.A = 180)
    (hB : capacities.B = 220)
    (hC : capacities.C = 150) :
    ∃ (attendance : EventAttendance),
      satisfiesConditions attendance capacities
      ∧ ∀ (other : EventAttendance),
          satisfiesConditions other capacities →
          attendance.boys + attendance.girls ≥ other.boys + other.girls
      ∧ attendance.boys + attendance.girls = 324 :=
  sorry


end NUMINAMATH_CALUDE_max_attendance_l1859_185901


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l1859_185982

theorem sum_of_even_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x - 1) * (x + 1)^9 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
                                   a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₂ + a₄ + a₆ + a₈ + a₁₀ = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l1859_185982


namespace NUMINAMATH_CALUDE_math_physics_majors_consecutive_probability_l1859_185981

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem math_physics_majors_consecutive_probability :
  let total_people : ℕ := 12
  let math_majors : ℕ := 5
  let physics_majors : ℕ := 4
  let biology_majors : ℕ := 3
  let favorable_outcomes : ℕ := choose total_people math_majors * factorial (math_majors - 1) * 
                                 choose (total_people - math_majors) physics_majors * 
                                 factorial (physics_majors - 1) * factorial biology_majors
  let total_outcomes : ℕ := factorial (total_people - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_math_physics_majors_consecutive_probability_l1859_185981


namespace NUMINAMATH_CALUDE_largest_root_is_four_l1859_185954

/-- The polynomial P(x) -/
def P (x r s : ℝ) : ℝ := x^6 - 12*x^5 + 40*x^4 - r*x^3 + s*x^2

/-- The line L(x) -/
def L (x d e : ℝ) : ℝ := d*x - e

/-- Theorem stating that the largest root of P(x) = L(x) is 4 -/
theorem largest_root_is_four 
  (r s d e : ℝ) 
  (h : ∃ (x₁ x₂ x₃ : ℝ), 
    (x₁ < x₂ ∧ x₂ < x₃) ∧ 
    (∀ x : ℝ, P x r s = L x d e ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    (∀ x : ℝ, (x - x₁)^2 * (x - x₂)^2 * (x - x₃) = P x r s - L x d e)) : 
  (∃ (x : ℝ), P x r s = L x d e ∧ ∀ y : ℝ, P y r s = L y d e → y ≤ x) ∧ 
  (∀ x : ℝ, P x r s = L x d e → x ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_largest_root_is_four_l1859_185954


namespace NUMINAMATH_CALUDE_total_amount_is_175_l1859_185996

/-- Represents the share of each person in rupees -/
structure Shares :=
  (first : ℝ)
  (second : ℝ)
  (third : ℝ)

/-- Calculates the total amount given the shares -/
def total_amount (s : Shares) : ℝ :=
  s.first + s.second + s.third

/-- Theorem: The total amount is 175 rupees -/
theorem total_amount_is_175 :
  ∃ (s : Shares),
    s.second = 45 ∧
    s.second = 0.45 * s.first ∧
    s.third = 0.30 * s.first ∧
    total_amount s = 175 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_175_l1859_185996


namespace NUMINAMATH_CALUDE_pencil_dozens_l1859_185956

theorem pencil_dozens (total_pencils : ℕ) (pencils_per_dozen : ℕ) (h1 : total_pencils = 144) (h2 : pencils_per_dozen = 12) :
  total_pencils / pencils_per_dozen = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencil_dozens_l1859_185956


namespace NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l1859_185944

/-- The maximum number of parts that three planes can divide 3D space into -/
def max_parts_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of parts that three planes can divide 3D space into is 8 -/
theorem max_parts_three_planes_is_eight :
  max_parts_three_planes = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l1859_185944
