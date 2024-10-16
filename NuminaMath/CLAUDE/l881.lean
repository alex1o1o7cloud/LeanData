import Mathlib

namespace NUMINAMATH_CALUDE_certain_number_value_l881_88187

theorem certain_number_value : ∃ x : ℝ, (0.60 * 50 = 0.42 * x + 17.4) ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l881_88187


namespace NUMINAMATH_CALUDE_function_value_at_point_l881_88165

theorem function_value_at_point (h : ℝ → ℝ) (b : ℝ) :
  (∀ x, h x = 4 * x - 5) →
  h b = 1 ↔ b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_point_l881_88165


namespace NUMINAMATH_CALUDE_billie_whipped_cream_cans_l881_88139

/-- The number of cans of whipped cream needed to cover the remaining pies after Tiffany eats some -/
def whipped_cream_cans_needed (pies_per_day : ℕ) (days : ℕ) (cream_cans_per_pie : ℕ) (pies_eaten : ℕ) : ℕ :=
  (pies_per_day * days - pies_eaten) * cream_cans_per_pie

/-- Theorem stating the number of whipped cream cans Billie needs to buy -/
theorem billie_whipped_cream_cans : 
  whipped_cream_cans_needed 3 11 2 4 = 58 := by
  sorry

end NUMINAMATH_CALUDE_billie_whipped_cream_cans_l881_88139


namespace NUMINAMATH_CALUDE_spider_group_ratio_l881_88103

/-- Represents a group of spiders -/
structure SpiderGroup where
  /-- Number of spiders in the group -/
  count : ℕ
  /-- Number of legs per spider -/
  legsPerSpider : ℕ
  /-- The group has more spiders than half the legs of a single spider -/
  more_than_half : count > legsPerSpider / 2
  /-- Total number of legs in the group -/
  totalLegs : ℕ
  /-- The total legs is the product of count and legs per spider -/
  total_legs_eq : totalLegs = count * legsPerSpider

/-- The theorem to be proved -/
theorem spider_group_ratio (g : SpiderGroup)
  (h1 : g.legsPerSpider = 8)
  (h2 : g.totalLegs = 112) :
  (g.count : ℚ) / (g.legsPerSpider / 2 : ℚ) = 7 / 2 :=
sorry

end NUMINAMATH_CALUDE_spider_group_ratio_l881_88103


namespace NUMINAMATH_CALUDE_smaller_root_of_quadratic_l881_88102

theorem smaller_root_of_quadratic (x : ℝ) : 
  (x - 2/3)^2 + (x - 2/3)*(x - 1/3) = 0 → 
  (x = 1/2 ∨ x = 2/3) ∧ 1/2 < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_root_of_quadratic_l881_88102


namespace NUMINAMATH_CALUDE_max_value_problem_1_l881_88132

theorem max_value_problem_1 (x : ℝ) (h : x < 5/4) :
  ∃ (y : ℝ), y = 4*x - 2 + 1/(4*x - 5) ∧ y ≤ 1 ∧ (∀ (z : ℝ), z = 4*x - 2 + 1/(4*x - 5) → z ≤ y) :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_1_l881_88132


namespace NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l881_88176

theorem negation_of_or_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l881_88176


namespace NUMINAMATH_CALUDE_tim_cabinet_price_l881_88144

/-- The amount Tim paid for a cabinet with a discount -/
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

/-- Proof that Tim paid $1020 for the cabinet -/
theorem tim_cabinet_price :
  let original_price : ℝ := 1200
  let discount_rate : ℝ := 0.15
  discounted_price original_price discount_rate = 1020 := by
sorry

end NUMINAMATH_CALUDE_tim_cabinet_price_l881_88144


namespace NUMINAMATH_CALUDE_ball_probabilities_l881_88129

def total_balls : ℕ := 15

def prob_three_red : ℚ := 2 / 91

theorem ball_probabilities (r : ℕ) (h1 : r + (r + 6) = total_balls)
  (h2 : Nat.choose r 3 / Nat.choose total_balls 3 = prob_three_red) :
  r = 5 ∧ Nat.choose (r + 6) 3 / Nat.choose total_balls 3 = 33 / 91 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l881_88129


namespace NUMINAMATH_CALUDE_number_order_l881_88182

/-- Represents a number in a given base -/
structure BaseNumber where
  digits : List Nat
  base : Nat

/-- Convert a BaseNumber to its decimal representation -/
def toDecimal (n : BaseNumber) : Nat :=
  n.digits.enum.foldl (fun acc (i, d) => acc + d * n.base ^ i) 0

/-- Define the given numbers -/
def a : BaseNumber := ⟨[14, 3], 16⟩
def b : BaseNumber := ⟨[0, 1, 2], 6⟩
def c : BaseNumber := ⟨[0, 0, 0, 1], 4⟩
def d : BaseNumber := ⟨[1, 1, 0, 1, 1, 1], 2⟩

/-- Theorem stating the order of the given numbers -/
theorem number_order :
  toDecimal b > toDecimal c ∧ toDecimal c > toDecimal a ∧ toDecimal a > toDecimal d := by
  sorry

end NUMINAMATH_CALUDE_number_order_l881_88182


namespace NUMINAMATH_CALUDE_digits_after_decimal_is_six_l881_88119

/-- The number of digits to the right of the decimal point when 5^7 / (10^5 * 8^2) is expressed as a decimal -/
def digits_after_decimal : ℕ :=
  let fraction := (5^7 : ℚ) / ((10^5 * 8^2) : ℚ)
  6

theorem digits_after_decimal_is_six :
  digits_after_decimal = 6 := by sorry

end NUMINAMATH_CALUDE_digits_after_decimal_is_six_l881_88119


namespace NUMINAMATH_CALUDE_second_error_greater_l881_88131

/-- Given two measured lines with their lengths and errors, prove that the absolute error of the second measurement is greater than the first. -/
theorem second_error_greater (length1 length2 error1 error2 : ℝ) 
  (h1 : length1 = 50)
  (h2 : length2 = 200)
  (h3 : error1 = 0.05)
  (h4 : error2 = 0.4) : 
  error2 > error1 := by
  sorry

end NUMINAMATH_CALUDE_second_error_greater_l881_88131


namespace NUMINAMATH_CALUDE_distance_ratio_theorem_l881_88195

/-- Given two points (4,3) and (2,-3) on a coordinate plane, this theorem proves:
    1. The direct distance between them is 2√10
    2. The horizontal distance between them is 2
    3. The ratio of the horizontal distance to the direct distance is not an integer -/
theorem distance_ratio_theorem :
  let p1 : ℝ × ℝ := (4, 3)
  let p2 : ℝ × ℝ := (2, -3)
  let direct_distance := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let horizontal_distance := |p1.1 - p2.1|
  let ratio := horizontal_distance / direct_distance
  (direct_distance = 2 * Real.sqrt 10) ∧
  (horizontal_distance = 2) ∧
  ¬(∃ n : ℤ, ratio = n) :=
by sorry

end NUMINAMATH_CALUDE_distance_ratio_theorem_l881_88195


namespace NUMINAMATH_CALUDE_dinner_fraction_l881_88186

theorem dinner_fraction (total_money : ℚ) (ice_cream_cost : ℚ) (money_left : ℚ) :
  total_money = 80 ∧ ice_cream_cost = 18 ∧ money_left = 2 →
  (total_money - ice_cream_cost - money_left) / total_money = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_dinner_fraction_l881_88186


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l881_88145

/-- Given a class of students with an incorrect average and one student's mark wrongly noted,
    calculate the wrongly noted mark. -/
theorem wrong_mark_calculation 
  (n : ℕ) -- number of students
  (initial_avg : ℚ) -- initial (incorrect) average
  (correct_mark : ℚ) -- correct mark for the student
  (correct_avg : ℚ) -- correct average after fixing the mark
  (h1 : n = 25) -- there are 25 students
  (h2 : initial_avg = 100) -- initial average is 100
  (h3 : correct_mark = 10) -- the correct mark is 10
  (h4 : correct_avg = 98) -- the correct average is 98
  : 
  -- The wrongly noted mark
  (n : ℚ) * initial_avg - ((n : ℚ) * correct_avg - correct_mark) = 60 :=
by sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l881_88145


namespace NUMINAMATH_CALUDE_fast_food_order_cost_correct_l881_88188

/-- Calculates the total cost of a fast food order with discount and tax --/
def fastFoodOrderCost (burgerPrice sandwichPrice smoothiePrice : ℚ)
                      (smoothieQuantity : ℕ)
                      (discountRate taxRate : ℚ)
                      (discountThreshold : ℚ)
                      (orderTime : ℕ) : ℚ :=
  let totalBeforeDiscount := burgerPrice + sandwichPrice + smoothiePrice * smoothieQuantity
  let discountedPrice := if totalBeforeDiscount > discountThreshold ∧ orderTime ≥ 1400 ∧ orderTime ≤ 1600
                         then totalBeforeDiscount * (1 - discountRate)
                         else totalBeforeDiscount
  let finalPrice := discountedPrice * (1 + taxRate)
  finalPrice

theorem fast_food_order_cost_correct :
  fastFoodOrderCost 5.75 4.50 4.25 2 0.20 0.12 15 1545 = 16.80 := by
  sorry

end NUMINAMATH_CALUDE_fast_food_order_cost_correct_l881_88188


namespace NUMINAMATH_CALUDE_intersection_equal_B_l881_88193

def A : Set ℝ := {x | x^2 - 4*x - 21 = 0}

def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem intersection_equal_B (m : ℝ) : 
  (A ∩ B m) = B m ↔ m = 0 ∨ m = -1/7 ∨ m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equal_B_l881_88193


namespace NUMINAMATH_CALUDE_max_sum_with_reciprocals_l881_88113

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  x + y ≤ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b + 1/a + 1/b = 5 ∧ a + b = 4 := by
  sorry

#check max_sum_with_reciprocals

end NUMINAMATH_CALUDE_max_sum_with_reciprocals_l881_88113


namespace NUMINAMATH_CALUDE_board_coverage_five_by_five_uncoverable_three_by_seven_uncoverable_four_by_six_coverable_five_by_six_coverable_three_by_eight_coverable_l881_88160

/-- Represents a rectangular board -/
structure Board where
  width : Nat
  height : Nat

/-- Checks if a board can be completely covered by dominoes -/
def canBeCovered (b : Board) : Prop :=
  (b.width * b.height) % 2 = 0

/-- Theorem: A board can be covered iff its area is even -/
theorem board_coverage (b : Board) :
  canBeCovered b ↔ (b.width * b.height) % 2 = 0 := by sorry

/-- The 5x5 board cannot be covered -/
theorem five_by_five_uncoverable : 
  ¬ canBeCovered ⟨5, 5⟩ := by sorry

/-- The 3x7 board cannot be covered -/
theorem three_by_seven_uncoverable : 
  ¬ canBeCovered ⟨3, 7⟩ := by sorry

/-- The 4x6 board can be covered -/
theorem four_by_six_coverable : 
  canBeCovered ⟨4, 6⟩ := by sorry

/-- The 5x6 board can be covered -/
theorem five_by_six_coverable : 
  canBeCovered ⟨5, 6⟩ := by sorry

/-- The 3x8 board can be covered -/
theorem three_by_eight_coverable : 
  canBeCovered ⟨3, 8⟩ := by sorry

end NUMINAMATH_CALUDE_board_coverage_five_by_five_uncoverable_three_by_seven_uncoverable_four_by_six_coverable_five_by_six_coverable_three_by_eight_coverable_l881_88160


namespace NUMINAMATH_CALUDE_fraction_relation_l881_88112

theorem fraction_relation (x y z w : ℚ) 
  (h1 : x / y = 12)
  (h2 : z / y = 4)
  (h3 : z / w = 3 / 4) :
  w / x = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relation_l881_88112


namespace NUMINAMATH_CALUDE_binomial_coefficient_probability_l881_88147

/-- The number of terms in the binomial expansion of (x-1)^10 -/
def n : ℕ := 11

/-- The number of positive coefficients in the expansion -/
def positive_coeff : ℕ := 6

/-- The number of negative coefficients in the expansion -/
def negative_coeff : ℕ := n - positive_coeff

/-- The probability of selecting two coefficients with a negative product -/
def prob_negative_product : ℚ := (positive_coeff * negative_coeff : ℚ) / (n.choose 2 : ℚ)

theorem binomial_coefficient_probability :
  prob_negative_product = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_probability_l881_88147


namespace NUMINAMATH_CALUDE_work_time_ratio_l881_88196

/-- The time it takes for Dev and Tina to complete the task together -/
def T : ℝ := 10

/-- The time it takes for Dev to complete the task alone -/
def dev_time : ℝ := T + 20

/-- The time it takes for Tina to complete the task alone -/
def tina_time : ℝ := T + 5

/-- The time it takes for Alex to complete the task alone -/
def alex_time : ℝ := T + 10

/-- The ratio of time taken by Dev, Tina, and Alex working alone -/
def time_ratio : Prop :=
  ∃ (k : ℝ), k > 0 ∧ dev_time = 6 * k ∧ tina_time = 3 * k ∧ alex_time = 4 * k

theorem work_time_ratio : time_ratio := by
  sorry

end NUMINAMATH_CALUDE_work_time_ratio_l881_88196


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l881_88151

/-- An arithmetic sequence is a sequence where the difference between 
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_15th_term 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_5 : a 5 = 5) 
  (h_10 : a 10 = 15) : 
  a 15 = 25 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l881_88151


namespace NUMINAMATH_CALUDE_max_value_of_f_l881_88171

def f (x : ℝ) := x - 5

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-5) 13 ∧
  f x = 8 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-5) 13 → f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l881_88171


namespace NUMINAMATH_CALUDE_side_BC_equation_l881_88124

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def altitude_from_AC : Line := { a := 2, b := -3, c := 1 }
def altitude_from_AB : Line := { a := 1, b := 1, c := -1 }

def vertex_A : ℝ × ℝ := (1, 2)

theorem side_BC_equation (t : Triangle) 
  (h1 : t.A = vertex_A)
  (h2 : altitude_from_AC.a * t.B.1 + altitude_from_AC.b * t.B.2 + altitude_from_AC.c = 0)
  (h3 : altitude_from_AC.a * t.C.1 + altitude_from_AC.b * t.C.2 + altitude_from_AC.c = 0)
  (h4 : altitude_from_AB.a * t.B.1 + altitude_from_AB.b * t.B.2 + altitude_from_AB.c = 0)
  (h5 : altitude_from_AB.a * t.C.1 + altitude_from_AB.b * t.C.2 + altitude_from_AB.c = 0) :
  ∃ (l : Line), l.a * t.B.1 + l.b * t.B.2 + l.c = 0 ∧
                l.a * t.C.1 + l.b * t.C.2 + l.c = 0 ∧
                l = { a := 2, b := 3, c := 7 } := by
  sorry

end NUMINAMATH_CALUDE_side_BC_equation_l881_88124


namespace NUMINAMATH_CALUDE_complex_division_equality_l881_88104

theorem complex_division_equality : (3 - I) / (1 + I) = 1 - 2*I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l881_88104


namespace NUMINAMATH_CALUDE_line_passes_through_point_l881_88115

/-- Given a line equation 3x + ay - 5 = 0 passing through point A(1, 2), prove that a = 1 -/
theorem line_passes_through_point (a : ℝ) : 
  (3 * 1 + a * 2 - 5 = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l881_88115


namespace NUMINAMATH_CALUDE_ship_total_distance_l881_88105

/-- Represents the daily travel of a ship --/
structure DailyTravel where
  distance : ℝ
  direction : String

/-- Calculates the total distance traveled by a ship over 4 days --/
def totalDistance (day1 day2 day3 day4 : DailyTravel) : ℝ :=
  day1.distance + day2.distance + day3.distance + day4.distance

/-- Theorem: The ship's total travel distance over 4 days is 960 miles --/
theorem ship_total_distance :
  let day1 := DailyTravel.mk 100 "north"
  let day2 := DailyTravel.mk (3 * 100) "east"
  let day3 := DailyTravel.mk (3 * 100 + 110) "east"
  let day4 := DailyTravel.mk 150 "30-degree angle with north"
  totalDistance day1 day2 day3 day4 = 960 := by
  sorry

#check ship_total_distance

end NUMINAMATH_CALUDE_ship_total_distance_l881_88105


namespace NUMINAMATH_CALUDE_cuboid_volume_is_48_l881_88192

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cuboid -/
def surfaceArea (d : CuboidDimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Calculates the volume of a cuboid -/
def volume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem stating the volume of a specific cuboid -/
theorem cuboid_volume_is_48 :
  ∃ (d : CuboidDimensions),
    d.length = 2 * d.width ∧
    d.height = 3 * d.width ∧
    surfaceArea d = 88 ∧
    volume d = 48 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_is_48_l881_88192


namespace NUMINAMATH_CALUDE_sum_of_abs_first_six_terms_l881_88164

def sequence_a (n : ℕ) : ℤ :=
  -5 + 2 * (n - 1)

theorem sum_of_abs_first_six_terms :
  (∀ n, sequence_a (n + 1) - sequence_a n = 2) →
  sequence_a 1 = -5 →
  (Finset.range 6).sum (fun i => |sequence_a (i + 1)|) = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_abs_first_six_terms_l881_88164


namespace NUMINAMATH_CALUDE_complex_division_equality_l881_88118

theorem complex_division_equality : (3 - I) / (2 + I) = 1 - I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l881_88118


namespace NUMINAMATH_CALUDE_gas_cost_calculation_l881_88149

/-- Calculates the total cost of filling up a car's gas tank multiple times with different gas prices -/
theorem gas_cost_calculation (tank_capacity : ℝ) (prices : List ℝ) :
  tank_capacity = 12 ∧ 
  prices = [3, 3.5, 4, 4.5] →
  (prices.map (· * tank_capacity)).sum = 180 := by
  sorry

#check gas_cost_calculation

end NUMINAMATH_CALUDE_gas_cost_calculation_l881_88149


namespace NUMINAMATH_CALUDE_two_color_theorem_l881_88133

/-- A type representing a plane divided by lines -/
structure DividedPlane where
  n : ℕ  -- number of lines
  regions : Set (Set ℝ × ℝ)  -- regions as sets of points
  adjacent : regions → regions → Prop  -- adjacency relation

/-- A coloring of the plane -/
def Coloring (p : DividedPlane) := p.regions → Bool

/-- A valid two-coloring of the plane -/
def ValidColoring (p : DividedPlane) (c : Coloring p) : Prop :=
  ∀ r1 r2 : p.regions, p.adjacent r1 r2 → c r1 ≠ c r2

/-- The main theorem: any divided plane has a valid two-coloring -/
theorem two_color_theorem (p : DividedPlane) : ∃ c : Coloring p, ValidColoring p c := by
  sorry

end NUMINAMATH_CALUDE_two_color_theorem_l881_88133


namespace NUMINAMATH_CALUDE_transformation_result_l881_88126

-- Define the original curve
def original_curve (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the transformation φ
def φ (x y : ℝ) : ℝ × ℝ := (3*x, 4*y)

-- Define the new curve
def new_curve (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 16) = 1

-- Theorem statement
theorem transformation_result :
  ∀ (x y : ℝ), original_curve x y → new_curve (φ x y).1 (φ x y).2 :=
by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l881_88126


namespace NUMINAMATH_CALUDE_bills_ratio_l881_88153

/-- Proves that the ratio of bills Geric had to bills Kyla had at the beginning is 2:1 --/
theorem bills_ratio (jessa_bills_after geric_bills kyla_bills : ℕ) : 
  jessa_bills_after = 7 →
  geric_bills = 16 →
  kyla_bills = (jessa_bills_after + 3) - 2 →
  (geric_bills : ℚ) / kyla_bills = 2 := by
  sorry

end NUMINAMATH_CALUDE_bills_ratio_l881_88153


namespace NUMINAMATH_CALUDE_wand_price_theorem_l881_88148

theorem wand_price_theorem (reduced_price : ℚ) (reduction_factor : ℚ) (original_price : ℚ) : 
  reduced_price = 8 → 
  reduction_factor = 1/8 → 
  reduced_price = reduction_factor * original_price → 
  original_price = 64 := by
sorry

end NUMINAMATH_CALUDE_wand_price_theorem_l881_88148


namespace NUMINAMATH_CALUDE_cubic_root_implies_h_value_l881_88197

theorem cubic_root_implies_h_value :
  ∀ h : ℝ, ((-3 : ℝ)^3 + h * (-3) - 18 = 0) → h = -15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_implies_h_value_l881_88197


namespace NUMINAMATH_CALUDE_knights_adjacent_probability_l881_88138

def numKnights : ℕ := 20
def chosenKnights : ℕ := 4

def probability_no_adjacent (n k : ℕ) : ℚ :=
  (n - 3) * (n - 5) * (n - 7) * (n - 9) / (n.choose k)

theorem knights_adjacent_probability :
  ∃ (Q : ℚ), Q = 1 - probability_no_adjacent numKnights chosenKnights :=
sorry

end NUMINAMATH_CALUDE_knights_adjacent_probability_l881_88138


namespace NUMINAMATH_CALUDE_acme_cheaper_than_beta_l881_88179

/-- Acme's pricing function -/
def acme_price (n : ℕ) : ℕ := 45 + 10 * n

/-- Beta's pricing function -/
def beta_price (n : ℕ) : ℕ := 15 * n

/-- Beta's minimum order quantity -/
def beta_min_order : ℕ := 5

/-- The minimum number of shirts above Beta's minimum order for which Acme is cheaper -/
def min_shirts_above_min : ℕ := 5

theorem acme_cheaper_than_beta :
  ∀ n : ℕ, n ≥ beta_min_order + min_shirts_above_min →
    acme_price (beta_min_order + min_shirts_above_min) < beta_price (beta_min_order + min_shirts_above_min) ∧
    ∀ m : ℕ, m < beta_min_order + min_shirts_above_min → acme_price m ≥ beta_price m :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_than_beta_l881_88179


namespace NUMINAMATH_CALUDE_first_fish_length_is_0_3_l881_88116

/-- The length of the second fish in feet -/
def second_fish_length : ℝ := 0.2

/-- The difference in length between the first and second fish in feet -/
def length_difference : ℝ := 0.1

/-- The length of the first fish in feet -/
def first_fish_length : ℝ := second_fish_length + length_difference

theorem first_fish_length_is_0_3 : first_fish_length = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_first_fish_length_is_0_3_l881_88116


namespace NUMINAMATH_CALUDE_probability_of_winning_pair_l881_88136

def total_cards : ℕ := 12
def cards_per_color : ℕ := 4
def num_colors : ℕ := 3
def num_numbers : ℕ := 4

def winning_pairs : ℕ := 
  (num_colors * (cards_per_color.choose 2)) + (num_numbers * (num_colors.choose 2))

def total_pairs : ℕ := total_cards.choose 2

theorem probability_of_winning_pair :
  (winning_pairs : ℚ) / total_pairs = 5 / 11 := by sorry

end NUMINAMATH_CALUDE_probability_of_winning_pair_l881_88136


namespace NUMINAMATH_CALUDE_even_digits_base9_567_l881_88190

/-- Converts a natural number from base 10 to base 9 --/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers --/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of even digits in the base-9 representation of 567₁₀ is 2 --/
theorem even_digits_base9_567 : countEvenDigits (toBase9 567) = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_base9_567_l881_88190


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l881_88175

/-- The original polynomial before multiplication by 3 -/
def original_poly (x : ℝ) : ℝ := x^4 + 2*x^3 + 5*x^2 + x + 2

/-- The expanded polynomial after multiplication by 3 -/
def expanded_poly (x : ℝ) : ℝ := 3 * (original_poly x)

/-- The coefficients of the expanded polynomial -/
def coefficients : List ℝ := [3, 6, 15, 3, 6]

/-- Theorem: The sum of the squares of the coefficients of the expanded form of 3(x^4 + 2x^3 + 5x^2 + x + 2) is 315 -/
theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 315 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l881_88175


namespace NUMINAMATH_CALUDE_circumference_difference_concentric_circles_l881_88172

/-- Given two concentric circles where the outer circle's radius is 12 feet greater than the inner circle's radius, the difference in their circumferences is 24π feet. -/
theorem circumference_difference_concentric_circles (r : ℝ) : 
  2 * π * (r + 12) - 2 * π * r = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_circumference_difference_concentric_circles_l881_88172


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l881_88170

/-- Number of ways to distribute n identical balls into k different boxes without empty boxes -/
def distribute_no_empty (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Number of ways to distribute n identical balls into k different boxes, allowing empty boxes -/
def distribute_with_empty (n k : ℕ) : ℕ :=
  (n + k - 1) * (n + k) * (n + k + 1) / (k * (k - 1) * (k - 2))

theorem ball_distribution_theorem :
  (distribute_no_empty 7 4 = 20) ∧
  (distribute_with_empty 7 4 = 120) := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l881_88170


namespace NUMINAMATH_CALUDE_speed_ratio_l881_88174

/-- The race scenario where A and B run at different speeds and finish at the same time -/
structure RaceScenario where
  speed_A : ℝ
  speed_B : ℝ
  distance_A : ℝ
  distance_B : ℝ
  finish_time : ℝ

/-- The conditions of the race -/
def race_conditions (r : RaceScenario) : Prop :=
  r.distance_A = 84 ∧ 
  r.distance_B = 42 ∧ 
  r.finish_time = r.distance_A / r.speed_A ∧
  r.finish_time = r.distance_B / r.speed_B

/-- The theorem stating the ratio of A's speed to B's speed -/
theorem speed_ratio (r : RaceScenario) (h : race_conditions r) : 
  r.speed_A / r.speed_B = 2 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_l881_88174


namespace NUMINAMATH_CALUDE_vector_AB_equals_zero_three_l881_88154

-- Define points A and B
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (1, 2)

-- Define vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem vector_AB_equals_zero_three : vectorAB = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_AB_equals_zero_three_l881_88154


namespace NUMINAMATH_CALUDE_pool_filling_time_l881_88199

/-- Represents the volume of water in a pool as a function of time -/
def water_volume (t : ℕ) : ℝ := sorry

/-- The full capacity of the pool -/
def full_capacity : ℝ := sorry

theorem pool_filling_time :
  (∀ t, water_volume (t + 1) = 2 * water_volume t) →  -- Volume doubles every hour
  (water_volume 8 = full_capacity) →                  -- Full capacity reached in 8 hours
  (water_volume 6 = full_capacity / 2) :=             -- Half capacity reached in 6 hours
by sorry

end NUMINAMATH_CALUDE_pool_filling_time_l881_88199


namespace NUMINAMATH_CALUDE_adult_admission_price_l881_88122

/-- Proves that the admission price for adults is 8 dollars given the specified conditions -/
theorem adult_admission_price
  (total_amount : ℕ)
  (total_tickets : ℕ)
  (children_tickets : ℕ)
  (children_price : ℕ)
  (h1 : total_amount = 201)
  (h2 : total_tickets = 33)
  (h3 : children_tickets = 21)
  (h4 : children_price = 5) :
  (total_amount - children_tickets * children_price) / (total_tickets - children_tickets) = 8 := by
  sorry


end NUMINAMATH_CALUDE_adult_admission_price_l881_88122


namespace NUMINAMATH_CALUDE_y_derivative_l881_88109

-- Define the function
noncomputable def y (x : ℝ) : ℝ := 
  -(Real.sinh x) / (2 * (Real.cosh x)^2) + (3/2) * Real.arcsin (Real.tanh x)

-- State the theorem
theorem y_derivative (x : ℝ) : 
  deriv y x = Real.cosh (2*x) / (Real.cosh x)^3 := by sorry

end NUMINAMATH_CALUDE_y_derivative_l881_88109


namespace NUMINAMATH_CALUDE_wonderland_roads_l881_88167

/-- The number of vertices in the complete graph -/
def n : ℕ := 5

/-- The number of edges shown on Alice's map -/
def shown_edges : ℕ := 7

/-- The total number of edges in a complete graph with n vertices -/
def total_edges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of missing edges -/
def missing_edges : ℕ := total_edges n - shown_edges

theorem wonderland_roads :
  missing_edges = 3 := by sorry

end NUMINAMATH_CALUDE_wonderland_roads_l881_88167


namespace NUMINAMATH_CALUDE_square_root_calculation_l881_88134

theorem square_root_calculation : (Real.sqrt 2 + 1)^2 - Real.sqrt (9/2) = 3 + Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_calculation_l881_88134


namespace NUMINAMATH_CALUDE_a_value_m_range_l881_88128

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

-- Theorem 1: Prove that a = 1
theorem a_value (a : ℝ) : 
  (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by sorry

-- Theorem 2: Prove that the minimum value of m is 4
theorem m_range : 
  ∃ m : ℝ, (∃ n : ℝ, f 1 n ≤ m - f 1 (-n)) ∧
  (∀ m' : ℝ, (∃ n : ℝ, f 1 n ≤ m' - f 1 (-n)) → m' ≥ m) ∧
  m = 4 := by sorry

end NUMINAMATH_CALUDE_a_value_m_range_l881_88128


namespace NUMINAMATH_CALUDE_square_sum_equals_eight_l881_88108

theorem square_sum_equals_eight : (-2)^2 + 2^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_eight_l881_88108


namespace NUMINAMATH_CALUDE_green_marbles_in_basket_b_l881_88106

/-- Represents a basket with two types of marbles -/
structure Basket :=
  (color1 : Nat)
  (color2 : Nat)

/-- Calculates the absolute difference between two numbers -/
def absDiff (a b : Nat) : Nat :=
  max a b - min a b

/-- Finds the maximum difference among a list of baskets -/
def maxDiff (baskets : List Basket) : Nat :=
  baskets.map (λ b => absDiff b.color1 b.color2) |>.maximum?
    |>.getD 0

theorem green_marbles_in_basket_b :
  let basketA : Basket := ⟨4, 2⟩
  let basketC : Basket := ⟨3, 9⟩
  let basketB : Basket := ⟨x, 1⟩
  let allBaskets : List Basket := [basketA, basketB, basketC]
  maxDiff allBaskets = 6 →
  x = 7 :=
by sorry

end NUMINAMATH_CALUDE_green_marbles_in_basket_b_l881_88106


namespace NUMINAMATH_CALUDE_f_of_2_equals_3_l881_88156

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - 1

-- State the theorem
theorem f_of_2_equals_3 : f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_f_of_2_equals_3_l881_88156


namespace NUMINAMATH_CALUDE_binomial_18_10_l881_88137

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 45760 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l881_88137


namespace NUMINAMATH_CALUDE_cubic_three_roots_m_range_l881_88140

/-- Given a cubic function f(x) = x³ - 6x² + 9x + m, if there exist three distinct
    real roots, then the parameter m must be in the open interval (-4, 0). -/
theorem cubic_three_roots_m_range (m : ℝ) :
  (∃ a b c : ℝ, a < b ∧ b < c ∧
    (a^3 - 6*a^2 + 9*a + m = 0) ∧
    (b^3 - 6*b^2 + 9*b + m = 0) ∧
    (c^3 - 6*c^2 + 9*c + m = 0)) →
  -4 < m ∧ m < 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_three_roots_m_range_l881_88140


namespace NUMINAMATH_CALUDE_david_pushups_count_l881_88159

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 7

/-- The difference between David's and Zachary's push-ups -/
def difference : ℕ := 30

/-- The number of push-ups David did -/
def david_pushups : ℕ := zachary_pushups + difference

theorem david_pushups_count : david_pushups = 37 := by
  sorry

end NUMINAMATH_CALUDE_david_pushups_count_l881_88159


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l881_88166

open Set

def R : Set ℝ := univ

def A : Set ℝ := {x | x > 0}

def B : Set ℝ := {x | x^2 - x - 2 > 0}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (R \ B) = Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l881_88166


namespace NUMINAMATH_CALUDE_number_problem_l881_88101

theorem number_problem (x : ℝ) : (0.2 * x = 0.2 * 650 + 190) → x = 1600 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l881_88101


namespace NUMINAMATH_CALUDE_division_remainder_problem_l881_88189

theorem division_remainder_problem (L S : ℕ) : 
  L - S = 1365 → 
  L = 1637 → 
  ∃ (q : ℕ), q = 6 ∧ L = q * S + (L % S) → 
  L % S = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l881_88189


namespace NUMINAMATH_CALUDE_hyperbola_center_is_3_4_l881_88162

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x + 16 * y^2 - 128 * y - 896 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 4)

/-- Theorem: The center of the hyperbola is (3, 4) -/
theorem hyperbola_center_is_3_4 :
  ∀ (x y : ℝ), hyperbola_equation x y →
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (dx dy : ℝ),
  dx^2 + dy^2 < ε^2 →
  hyperbola_equation (hyperbola_center.1 + dx) (hyperbola_center.2 + dy) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_3_4_l881_88162


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l881_88183

-- Define a 7-arithmetic fractional-linear function
def is_7_arithmetic_fractional_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x : ℝ, f x = (a * x + b) / (c * x + d)

-- State the theorem
theorem unique_function_satisfying_conditions :
  ∃! f : ℝ → ℝ, 
    is_7_arithmetic_fractional_linear f ∧ 
    f 0 = 0 ∧ 
    f 1 = 4 ∧ 
    f 4 = 2 ∧
    ∀ x : ℝ, f x = x / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l881_88183


namespace NUMINAMATH_CALUDE_total_ways_eq_19_l881_88178

/-- The number of direct bus services from place A to place B -/
def direct_services : ℕ := 4

/-- The number of bus services from place A to place C -/
def services_A_to_C : ℕ := 5

/-- The number of bus services from place C to place B -/
def services_C_to_B : ℕ := 3

/-- The total number of ways to travel from place A to place B -/
def total_ways : ℕ := direct_services + services_A_to_C * services_C_to_B

theorem total_ways_eq_19 : total_ways = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_eq_19_l881_88178


namespace NUMINAMATH_CALUDE_outfit_combinations_l881_88158

def num_shirts : ℕ := 5
def num_pants : ℕ := 3
def num_hats : ℕ := 2

theorem outfit_combinations : num_shirts * num_pants * num_hats = 30 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l881_88158


namespace NUMINAMATH_CALUDE_curve_properties_l881_88125

-- Define the curve
def curve (x y : ℝ) : Prop := abs x + y^2 - 3*y = 0

-- Theorem for the axis of symmetry and range of y
theorem curve_properties :
  (∀ x y : ℝ, curve x y ↔ curve (-x) y) ∧
  (∀ y : ℝ, (∃ x : ℝ, curve x y) → 0 ≤ y ∧ y ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_curve_properties_l881_88125


namespace NUMINAMATH_CALUDE_rex_driving_lessons_l881_88161

/-- The number of hour-long lessons Rex wants to take before his test -/
def total_lessons : ℕ := 40

/-- The number of hours of lessons Rex takes per week -/
def hours_per_week : ℕ := 4

/-- The number of weeks Rex has already completed -/
def completed_weeks : ℕ := 6

/-- The number of additional weeks Rex needs to reach his goal -/
def additional_weeks : ℕ := 4

/-- Theorem stating that the total number of hour-long lessons Rex wants to take is 40 -/
theorem rex_driving_lessons :
  total_lessons = hours_per_week * (completed_weeks + additional_weeks) :=
by sorry

end NUMINAMATH_CALUDE_rex_driving_lessons_l881_88161


namespace NUMINAMATH_CALUDE_sector_area_l881_88135

theorem sector_area (circumference : ℝ) (central_angle : ℝ) : 
  circumference = 16 * Real.pi → 
  central_angle = Real.pi / 4 → 
  (central_angle / (2 * Real.pi)) * ((circumference^2) / (4 * Real.pi)) = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l881_88135


namespace NUMINAMATH_CALUDE_nth_equation_proof_l881_88194

theorem nth_equation_proof (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_proof_l881_88194


namespace NUMINAMATH_CALUDE_t_shirt_cost_is_20_l881_88110

/-- The cost of a single t-shirt -/
def t_shirt_cost : ℝ := sorry

/-- The number of t-shirts bought -/
def num_t_shirts : ℕ := 3

/-- The cost of pants -/
def pants_cost : ℝ := 50

/-- The total amount spent -/
def total_spent : ℝ := 110

theorem t_shirt_cost_is_20 : t_shirt_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_t_shirt_cost_is_20_l881_88110


namespace NUMINAMATH_CALUDE_max_value_theorem_l881_88114

theorem max_value_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a^2 * (b + c - a) = b^2 * (a + c - b))
  (h2 : b^2 * (a + c - b) = c^2 * (b + a - c)) :
  ∀ x : ℝ, (2*b + 3*c) / a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l881_88114


namespace NUMINAMATH_CALUDE_largest_quantity_l881_88100

theorem largest_quantity (A B C : ℚ) : 
  A = 2020/2019 + 2020/2021 →
  B = 2021/2022 + 2023/2022 →
  C = 2022/2021 + 2022/2023 →
  A > B ∧ A > C :=
by sorry

end NUMINAMATH_CALUDE_largest_quantity_l881_88100


namespace NUMINAMATH_CALUDE_three_digit_integers_with_7_no_4_l881_88198

/-- The set of digits excluding 0, 4, and 7 -/
def digits_no_047 : Finset Nat := Finset.filter (fun d => d ≠ 0 ∧ d ≠ 4 ∧ d ≠ 7) (Finset.range 10)

/-- The set of digits excluding 0 and 4 -/
def digits_no_04 : Finset Nat := Finset.filter (fun d => d ≠ 0 ∧ d ≠ 4) (Finset.range 10)

/-- The set of digits excluding 4 -/
def digits_no_4 : Finset Nat := Finset.filter (fun d => d ≠ 4) (Finset.range 10)

/-- The number of three-digit integers without 7 and 4 -/
def count_no_47 : Nat := digits_no_047.card * digits_no_4.card * digits_no_4.card

/-- The number of three-digit integers without 4 -/
def count_no_4 : Nat := digits_no_04.card * digits_no_4.card * digits_no_4.card

theorem three_digit_integers_with_7_no_4 :
  count_no_4 - count_no_47 = 200 := by sorry

end NUMINAMATH_CALUDE_three_digit_integers_with_7_no_4_l881_88198


namespace NUMINAMATH_CALUDE_rate_percent_is_twelve_l881_88123

/-- Calculates the rate percent on simple interest given principal, amount, and time. -/
def calculate_rate_percent (principal amount : ℚ) (time : ℕ) : ℚ :=
  let simple_interest := amount - principal
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that the rate percent on simple interest is 12% for the given conditions. -/
theorem rate_percent_is_twelve :
  let principal : ℚ := 750
  let amount : ℚ := 1200
  let time : ℕ := 5
  calculate_rate_percent principal amount time = 12 := by
  sorry

#eval calculate_rate_percent 750 1200 5

end NUMINAMATH_CALUDE_rate_percent_is_twelve_l881_88123


namespace NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l881_88121

/-- The circle equation: x^2 + y^2 + 4x + 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The line equation: ax + 2by + 4 = 0 -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x + 2*b*y + 4 = 0

/-- The chord length is 4 -/
def chord_length_is_4 (a b : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ ∧
    circle_equation x₂ y₂ ∧
    line_equation a b x₁ y₁ ∧
    line_equation a b x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4^2

theorem min_value_of_a2_plus_b2 :
  ∀ a b : ℝ, chord_length_is_4 a b →
  ∃ min : ℝ, min = 2 ∧ a^2 + b^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l881_88121


namespace NUMINAMATH_CALUDE_even_decreasing_implies_increasing_l881_88185

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f y < f x

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem even_decreasing_implies_increasing
  (f : ℝ → ℝ) (h_even : is_even f) (h_decr : decreasing_on f (Set.Ici 0)) :
  increasing_on f (Set.Iic 0) :=
sorry

end NUMINAMATH_CALUDE_even_decreasing_implies_increasing_l881_88185


namespace NUMINAMATH_CALUDE_mo_negative_bo_positive_l881_88143

-- Define the two types of people
inductive PersonType
| Positive
| Negative

-- Define a person with a type
structure Person where
  name : String
  type : PersonType

-- Define the property of asking a question
def asksQuestion (p : Person) (q : Prop) : Prop :=
  match p.type with
  | PersonType.Positive => q
  | PersonType.Negative => ¬q

-- Define Mo and Bo
def Mo : Person := { name := "Mo", type := PersonType.Negative }
def Bo : Person := { name := "Bo", type := PersonType.Positive }

-- Define the question Mo asked
def moQuestion : Prop := Mo.type = PersonType.Negative ∧ Bo.type = PersonType.Negative

-- Theorem stating that Mo is negative and Bo is positive
theorem mo_negative_bo_positive :
  asksQuestion Mo moQuestion ∧ (Mo.type = PersonType.Negative ∧ Bo.type = PersonType.Positive) :=
by sorry


end NUMINAMATH_CALUDE_mo_negative_bo_positive_l881_88143


namespace NUMINAMATH_CALUDE_merchant_profit_comparison_l881_88117

/-- Represents the profit calculation for two merchants selling goods --/
theorem merchant_profit_comparison
  (x : ℝ) -- cost price of goods for each merchant
  (h_pos : x > 0) -- assumption that cost price is positive
  : x < 1.08 * x := by
  sorry

#check merchant_profit_comparison

end NUMINAMATH_CALUDE_merchant_profit_comparison_l881_88117


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l881_88146

/-- A line passing through point (1,1) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point (1,1) -/
  passes_through_point : slope + y_intercept = 2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : y_intercept = slope * y_intercept ∨ y_intercept = 0

/-- The equation of an EqualInterceptLine is x + y = 2 or y = x -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = 1 ∧ l.y_intercept = 1) ∨ (l.slope = 1 ∧ l.y_intercept = 0) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l881_88146


namespace NUMINAMATH_CALUDE_system_solution_l881_88111

theorem system_solution (u v : ℝ) : 
  u + v = 10 ∧ 3 * u - 2 * v = 5 → u = 5 ∧ v = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l881_88111


namespace NUMINAMATH_CALUDE_cost_of_pens_calculation_l881_88152

/-- The cost of the box of pens Linda bought -/
def cost_of_pens : ℝ := 1.70

/-- The number of notebooks Linda bought -/
def num_notebooks : ℕ := 3

/-- The cost of each notebook -/
def cost_per_notebook : ℝ := 1.20

/-- The cost of the box of pencils -/
def cost_of_pencils : ℝ := 1.50

/-- The total amount Linda spent -/
def total_spent : ℝ := 6.80

theorem cost_of_pens_calculation :
  cost_of_pens = total_spent - (↑num_notebooks * cost_per_notebook + cost_of_pencils) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_pens_calculation_l881_88152


namespace NUMINAMATH_CALUDE_happy_cattle_ranch_population_l881_88142

/-- The number of cows after n years, given an initial population and growth rate -/
def cowPopulation (initialPopulation : ℕ) (growthRate : ℚ) (years : ℕ) : ℚ :=
  initialPopulation * (1 + growthRate) ^ years

/-- Theorem: The cow population on Happy Cattle Ranch after 2 years -/
theorem happy_cattle_ranch_population :
  cowPopulation 200 (1/2) 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_happy_cattle_ranch_population_l881_88142


namespace NUMINAMATH_CALUDE_student_arrangement_l881_88184

/-- The number of ways to arrange students with specific conditions -/
def arrangement_count : ℕ := 120

/-- The number of male students -/
def male_students : ℕ := 3

/-- The number of female students -/
def female_students : ℕ := 4

/-- The number of students that must stand at the ends -/
def end_students : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := male_students + female_students

theorem student_arrangement :
  arrangement_count = 
    (end_students * (total_students - end_students).factorial) :=
sorry

end NUMINAMATH_CALUDE_student_arrangement_l881_88184


namespace NUMINAMATH_CALUDE_probability_at_least_one_defective_l881_88191

/-- The probability of selecting at least one defective bulb when choosing 2 bulbs at random from a box containing 20 bulbs, of which 4 are defective, is 7/19. -/
theorem probability_at_least_one_defective (total_bulbs : Nat) (defective_bulbs : Nat) 
    (h1 : total_bulbs = 20) 
    (h2 : defective_bulbs = 4) : 
  let p := 1 - (total_bulbs - defective_bulbs : ℚ) * (total_bulbs - defective_bulbs - 1) / 
           (total_bulbs * (total_bulbs - 1))
  p = 7 / 19 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_defective_l881_88191


namespace NUMINAMATH_CALUDE_geometric_series_relation_l881_88155

/-- Given two infinite geometric series with specific properties, prove that n = 6 -/
theorem geometric_series_relation (n : ℝ) : 
  let a₁ : ℝ := 15  -- First term of both series
  let b₁ : ℝ := 6   -- Second term of first series
  let b₂ : ℝ := 6 + n  -- Second term of second series
  let r₁ : ℝ := b₁ / a₁  -- Common ratio of first series
  let r₂ : ℝ := b₂ / a₁  -- Common ratio of second series
  let S₁ : ℝ := a₁ / (1 - r₁)  -- Sum of first series
  let S₂ : ℝ := a₁ / (1 - r₂)  -- Sum of second series
  S₂ = 3 * S₁ →  -- Condition: sum of second series is three times the sum of first series
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l881_88155


namespace NUMINAMATH_CALUDE_solve_equation_l881_88107

theorem solve_equation (x : ℝ) : 3 * x + 1 = -(5 - 2 * x) → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l881_88107


namespace NUMINAMATH_CALUDE_cube_surface_area_from_prism_volume_cube_surface_area_from_prism_volume_proof_l881_88169

/-- The surface area of a cube with volume equal to a 10x10x8 inch rectangular prism is 1200 square inches. -/
theorem cube_surface_area_from_prism_volume : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun prism_length prism_width prism_height cube_surface_area =>
    prism_length = 10 ∧
    prism_width = 10 ∧
    prism_height = 8 ∧
    cube_surface_area = 6 * (prism_length * prism_width * prism_height) ^ (2/3) ∧
    cube_surface_area = 1200

/-- Proof of the theorem -/
theorem cube_surface_area_from_prism_volume_proof :
  cube_surface_area_from_prism_volume 10 10 8 1200 := by
  sorry

#check cube_surface_area_from_prism_volume
#check cube_surface_area_from_prism_volume_proof

end NUMINAMATH_CALUDE_cube_surface_area_from_prism_volume_cube_surface_area_from_prism_volume_proof_l881_88169


namespace NUMINAMATH_CALUDE_new_ratio_after_transaction_l881_88150

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  horses : ℕ
  cows : ℕ

/-- Represents the transaction of selling horses and buying cows -/
def performTransaction (farm : FarmAnimals) : FarmAnimals :=
  { horses := farm.horses - 15, cows := farm.cows + 15 }

/-- Theorem stating the new ratio of horses to cows after the transaction -/
theorem new_ratio_after_transaction (initial : FarmAnimals)
    (h1 : initial.horses = 4 * initial.cows)
    (h2 : (performTransaction initial).horses = (performTransaction initial).cows + 60) :
    (performTransaction initial).horses / (performTransaction initial).cows = 7 / 3 := by
  sorry


end NUMINAMATH_CALUDE_new_ratio_after_transaction_l881_88150


namespace NUMINAMATH_CALUDE_cannot_finish_third_l881_88120

/-- Represents the relative positions of runners in a race -/
def BeatsIn (runners : Type) := runners → runners → Prop

/-- A race with 5 runners and their relative positions -/
structure Race (runners : Type) :=
  (beats : BeatsIn runners)
  (P Q R S T : runners)
  (p_beats_q : beats P Q)
  (p_beats_r : beats P R)
  (p_beats_s : beats P S)
  (q_beats_s : beats Q S)
  (s_beats_r : beats S R)
  (t_after_p : beats P T)
  (t_before_q : beats T Q)

/-- Represents the finishing position of a runner -/
def FinishPosition (runners : Type) := runners → ℕ

theorem cannot_finish_third (runners : Type) (race : Race runners) 
  (finish : FinishPosition runners) :
  (finish race.P ≠ 3) ∧ (finish race.R ≠ 3) := by sorry

end NUMINAMATH_CALUDE_cannot_finish_third_l881_88120


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l881_88130

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l881_88130


namespace NUMINAMATH_CALUDE_largest_n_with_1992_divisors_and_phi_divides_l881_88127

/-- The number of positive divisors of n -/
def divisor_count (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

theorem largest_n_with_1992_divisors_and_phi_divides (n : ℕ) :
  (phi n ∣ n) →
  (divisor_count n = 1992) →
  n ≤ 2^1991 :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_1992_divisors_and_phi_divides_l881_88127


namespace NUMINAMATH_CALUDE_min_value_expression_l881_88180

theorem min_value_expression (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x^2 + y^2 + z^2 = 1) :
  (z + 1)^2 / (2 * x * y * z) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l881_88180


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l881_88163

/-- A regular dodecahedron with 20 vertices -/
structure Dodecahedron :=
  (vertices : Finset Nat)
  (h_card : vertices.card = 20)

/-- The probability of two randomly chosen vertices being endpoints of an edge -/
def edge_probability (d : Dodecahedron) : ℚ :=
  3 / 19

theorem dodecahedron_edge_probability (d : Dodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l881_88163


namespace NUMINAMATH_CALUDE_complex_equation_sum_l881_88173

theorem complex_equation_sum (x y : ℝ) : 
  (x + y * Complex.I) / (1 + Complex.I) = (2 : ℂ) + Complex.I → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l881_88173


namespace NUMINAMATH_CALUDE_triangle_angle_c_l881_88181

theorem triangle_angle_c (A B C : Real) :
  -- ABC is a triangle
  A + B + C = π →
  -- Given condition
  |Real.cos A - Real.sqrt 3 / 2| + (1 - Real.tan B)^2 = 0 →
  -- Conclusion
  C = π * 7 / 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l881_88181


namespace NUMINAMATH_CALUDE_age_difference_in_decades_l881_88177

/-- Given that the sum of x's and y's ages is 18 years greater than the sum of y's and z's ages,
    prove that z is 1.8 decades younger than x. -/
theorem age_difference_in_decades (x y z : ℕ) (h : x + y = y + z + 18) :
  (x - z : ℚ) / 10 = 1.8 := by sorry

end NUMINAMATH_CALUDE_age_difference_in_decades_l881_88177


namespace NUMINAMATH_CALUDE_yellow_pairs_count_l881_88168

theorem yellow_pairs_count (total_students : ℕ) (blue_students : ℕ) (yellow_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  total_students = 156 →
  blue_students = 68 →
  yellow_students = 88 →
  total_pairs = 78 →
  blue_blue_pairs = 31 →
  total_students = blue_students + yellow_students →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 41 ∧ 
    yellow_yellow_pairs + blue_blue_pairs + (blue_students - 2 * blue_blue_pairs) = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_yellow_pairs_count_l881_88168


namespace NUMINAMATH_CALUDE_polynomial_equality_l881_88157

theorem polynomial_equality (t s : ℚ) : 
  (∀ x : ℚ, (3*x^2 - 4*x + 9) * (5*x^2 + t*x + 12) = 15*x^4 + s*x^3 + 33*x^2 + 12*x + 108) 
  ↔ 
  (t = 37/5 ∧ s = 11/5) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l881_88157


namespace NUMINAMATH_CALUDE_reorganize_books_leftover_l881_88141

/-- The number of books left over when reorganizing boxes -/
def books_left_over (initial_boxes : ℕ) (books_per_initial_box : ℕ) (books_per_new_box : ℕ) : ℕ :=
  (initial_boxes * books_per_initial_box) % books_per_new_box

/-- Theorem stating the number of books left over in the specific scenario -/
theorem reorganize_books_leftover :
  books_left_over 2020 42 45 = 30 := by
  sorry

end NUMINAMATH_CALUDE_reorganize_books_leftover_l881_88141
