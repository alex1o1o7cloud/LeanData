import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_area_error_l3768_376851

/-- The combined percentage error in the calculated area of a rectangle, given length and width measurement errors -/
theorem rectangle_area_error (length_error width_error : ℝ) :
  length_error = 0.02 →
  width_error = 0.03 →
  (1 + length_error) * (1 + width_error) - 1 = 0.0206 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_error_l3768_376851


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3768_376853

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (1 : ℂ) / (1 - Complex.I) = ↑a + ↑b * Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3768_376853


namespace NUMINAMATH_CALUDE_free_trade_superior_l3768_376887

/-- Represents a country with its production capacity for zucchinis and cauliflower -/
structure Country where
  zucchini_capacity : ℝ
  cauliflower_capacity : ℝ

/-- Represents the market conditions and consumption preferences -/
structure MarketConditions where
  price_ratio : ℝ  -- Price of zucchini / Price of cauliflower
  consumption_ratio : ℝ  -- Ratio of zucchini to cauliflower consumption

/-- Calculates the total vegetable consumption under free trade conditions -/
def free_trade_consumption (a b : Country) (market : MarketConditions) : ℝ :=
  sorry

/-- Calculates the total vegetable consumption for the unified country without trade -/
def unified_consumption (a b : Country) : ℝ :=
  sorry

/-- Theorem stating that free trade leads to higher total consumption -/
theorem free_trade_superior (a b : Country) (market : MarketConditions) :
  free_trade_consumption a b market > unified_consumption a b :=
sorry

end NUMINAMATH_CALUDE_free_trade_superior_l3768_376887


namespace NUMINAMATH_CALUDE_pony_price_is_18_l3768_376822

/-- The regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The discount rate for Pony jeans as a decimal -/
def pony_discount : ℝ := 0.14

/-- The sum of discount rates for Fox and Pony jeans as a decimal -/
def total_discount : ℝ := 0.22

/-- The total savings from purchasing 5 pairs of jeans -/
def total_savings : ℝ := 8.64

/-- The number of Fox jeans purchased -/
def fox_count : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_count : ℕ := 2

/-- The regular price of Pony jeans -/
def pony_price : ℝ := 18

theorem pony_price_is_18 :
  fox_count * fox_price * (total_discount - pony_discount) +
  pony_count * pony_price * pony_discount = total_savings :=
by sorry

end NUMINAMATH_CALUDE_pony_price_is_18_l3768_376822


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3768_376860

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_factorial_eight_ten : 
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3768_376860


namespace NUMINAMATH_CALUDE_platform_length_specific_platform_length_l3768_376855

/-- The length of a platform given train speed, crossing time, and train length -/
theorem platform_length 
  (train_speed_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (train_length_m : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time_s
  total_distance - train_length_m

/-- Proof of the specific platform length problem -/
theorem specific_platform_length : 
  platform_length 72 26 260.0416 = 259.9584 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_specific_platform_length_l3768_376855


namespace NUMINAMATH_CALUDE_total_people_waiting_l3768_376811

/-- The number of people waiting at each entrance -/
def people_per_entrance : ℕ := 283

/-- The number of entrances -/
def num_entrances : ℕ := 5

/-- The total number of people waiting to get in -/
def total_people : ℕ := people_per_entrance * num_entrances

theorem total_people_waiting :
  total_people = 1415 := by sorry

end NUMINAMATH_CALUDE_total_people_waiting_l3768_376811


namespace NUMINAMATH_CALUDE_problem_solution_l3768_376827

theorem problem_solution (a b m n k : ℝ) 
  (h1 : a + b = 0) 
  (h2 : m * n = 1) 
  (h3 : k^2 = 2) : 
  2011*a + 2012*b + m*n*a + k^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3768_376827


namespace NUMINAMATH_CALUDE_probability_three_by_three_square_l3768_376876

/-- A square with 16 equally spaced points around its perimeter -/
structure SquareWithPoints :=
  (side_length : ℕ)
  (num_points : ℕ)

/-- The probability of selecting two points that are one unit apart -/
def probability_one_unit_apart (s : SquareWithPoints) : ℚ :=
  sorry

/-- Theorem stating the probability for a 3x3 square with 16 points -/
theorem probability_three_by_three_square :
  ∃ s : SquareWithPoints, s.side_length = 3 ∧ s.num_points = 16 ∧ 
  probability_one_unit_apart s = 1 / 10 :=
sorry

end NUMINAMATH_CALUDE_probability_three_by_three_square_l3768_376876


namespace NUMINAMATH_CALUDE_correct_answer_is_105_l3768_376848

theorem correct_answer_is_105 (x : ℕ) : 
  (x - 5 = 95) → (x + 5 = 105) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_answer_is_105_l3768_376848


namespace NUMINAMATH_CALUDE_glass_volume_l3768_376861

theorem glass_volume (V : ℝ) 
  (h1 : 0.4 * V = V - 0.6 * V) -- pessimist's glass is 60% empty
  (h2 : 0.6 * V - 0.4 * V = 46) -- difference in water volume is 46 ml
  : V = 230 := by
  sorry

end NUMINAMATH_CALUDE_glass_volume_l3768_376861


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_plus_c_l3768_376836

theorem value_of_a_minus_b_plus_c 
  (a b c : ℝ) 
  (ha : |a| = 5)
  (hb : |b| = 3)
  (hc : |c| = 6)
  (hab : |a+b| = -(a+b))
  (hac : |a+c| = a+c) :
  a - b + c = 4 ∨ a - b + c = -2 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_plus_c_l3768_376836


namespace NUMINAMATH_CALUDE_tan_value_from_trig_equation_l3768_376818

theorem tan_value_from_trig_equation (α : ℝ) 
  (h : (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = 5/16) :
  Real.tan α = -1/3 := by sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_equation_l3768_376818


namespace NUMINAMATH_CALUDE_sequence_a_4_equals_zero_l3768_376829

theorem sequence_a_4_equals_zero :
  let a : ℕ+ → ℤ := fun n => n.val^2 - 3*n.val - 4
  a 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_4_equals_zero_l3768_376829


namespace NUMINAMATH_CALUDE_total_distance_traveled_l3768_376852

/-- The rate at which the train travels in miles per minute -/
def train_rate : ℚ := 1 / 2

/-- The distance already traveled before time measurement began in miles -/
def initial_distance : ℚ := 5

/-- The total time of travel in minutes -/
def total_time : ℚ := 90

/-- Theorem stating the total distance traveled by the train -/
theorem total_distance_traveled :
  train_rate * total_time + initial_distance = 50 := by sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l3768_376852


namespace NUMINAMATH_CALUDE_unique_relative_minimum_l3768_376899

/-- The function f(x) = x^4 - x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - x^3 - x^2 + a*x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 4*x^3 - 3*x^2 - 2*x + a

theorem unique_relative_minimum (a : ℝ) :
  (∃ (x : ℝ), f a x = x ∧ 
    ∀ (y : ℝ), y ≠ x → f a y > f a x) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_relative_minimum_l3768_376899


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3768_376816

def f (x : ℝ) := x^3 - x - 1

theorem equation_solution_exists :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 1.5 ∧ |f x| ≤ 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3768_376816


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3768_376854

/-- Given a square with side length 2y that is divided into a center square
    with side length y and four congruent rectangles, prove that the perimeter
    of one of these rectangles is 3y. -/
theorem rectangle_perimeter (y : ℝ) (y_pos : 0 < y) :
  let large_square_side := 2 * y
  let center_square_side := y
  let rectangle_width := (large_square_side - center_square_side) / 2
  let rectangle_length := center_square_side
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  rectangle_perimeter = 3 * y :=
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3768_376854


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3768_376849

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) : 
  x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3768_376849


namespace NUMINAMATH_CALUDE_non_technicians_percentage_l3768_376813

/-- Represents the composition of workers in a factory -/
structure Factory where
  total : ℕ
  technicians : ℕ
  permanent_technicians : ℕ
  permanent_non_technicians : ℕ
  temporary : ℕ

/-- The conditions of the factory as described in the problem -/
def factory_conditions (f : Factory) : Prop :=
  f.technicians = f.total / 2 ∧
  f.permanent_technicians = f.technicians / 2 ∧
  f.permanent_non_technicians = (f.total - f.technicians) / 2 ∧
  f.temporary = f.total / 2

/-- The theorem stating that under the given conditions, 
    non-technicians make up 50% of the workforce -/
theorem non_technicians_percentage (f : Factory) 
  (h : factory_conditions f) : 
  (f.total - f.technicians) * 100 / f.total = 50 := by
  sorry


end NUMINAMATH_CALUDE_non_technicians_percentage_l3768_376813


namespace NUMINAMATH_CALUDE_chord_length_l3768_376806

-- Define the circle and points
variable (O A B C D : Point)
variable (r : ℝ)

-- Define the circle properties
def is_circle (O : Point) (r : ℝ) : Prop := sorry

-- Define diameter
def is_diameter (O A D : Point) : Prop := sorry

-- Define chord
def is_chord (O A B C : Point) : Prop := sorry

-- Define arc measure
def arc_measure (O C D : Point) : ℝ := sorry

-- Define angle measure
def angle_measure (A B O : Point) : ℝ := sorry

-- Define distance between points
def distance (P Q : Point) : ℝ := sorry

-- Theorem statement
theorem chord_length 
  (h_circle : is_circle O r)
  (h_diameter : is_diameter O A D)
  (h_chord : is_chord O A B C)
  (h_BO : distance B O = 7)
  (h_angle : angle_measure A B O = 45)
  (h_arc : arc_measure O C D = 90) :
  distance B C = 7 := by sorry

end NUMINAMATH_CALUDE_chord_length_l3768_376806


namespace NUMINAMATH_CALUDE_g_difference_l3768_376826

/-- A linear function with a constant difference of 4 between consecutive integers -/
def g_property (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x + y) = g x + g y) ∧ 
  (∀ h : ℝ, g (h + 1) - g h = 4)

/-- The difference between g(3) and g(7) is -16 -/
theorem g_difference (g : ℝ → ℝ) (hg : g_property g) : g 3 - g 7 = -16 := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l3768_376826


namespace NUMINAMATH_CALUDE_ellipse_property_l3768_376892

/-- Given an ellipse with foci at (2, 0) and (8, 0) passing through (5, 3),
    prove that the sum of its semi-major axis length and the y-coordinate of its center is 3√2. -/
theorem ellipse_property (a b h k : ℝ) : 
  a > 0 → b > 0 →
  (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ 
    (x - 2)^2 + y^2 + (x - 8)^2 + y^2 = ((x - 2)^2 + y^2 + (x - 8)^2 + y^2)) →
  (5 - h)^2 / a^2 + (3 - k)^2 / b^2 = 1 →
  a + k = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_property_l3768_376892


namespace NUMINAMATH_CALUDE_two_arithmetic_sequences_sum_l3768_376857

def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem two_arithmetic_sequences_sum : 
  let seq1_sum := arithmetic_sum 2 10 5
  let seq2_sum := arithmetic_sum 10 10 5
  seq1_sum + seq2_sum = 260 := by
  sorry

end NUMINAMATH_CALUDE_two_arithmetic_sequences_sum_l3768_376857


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3768_376812

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (ha : ‖a‖ = 3)
  (hb : ‖b‖ = 4)
  (hab : ‖a + b‖ = 2) :
  ‖a - b‖ = Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3768_376812


namespace NUMINAMATH_CALUDE_vector_expression_in_quadrilateral_l3768_376845

/-- Given a quadrilateral OABC in space, prove that MN = -1/2 * a + 1/2 * b + 1/2 * c -/
theorem vector_expression_in_quadrilateral
  (O A B C M N : EuclideanSpace ℝ (Fin 3))
  (a b c : EuclideanSpace ℝ (Fin 3))
  (h1 : A - O = a)
  (h2 : B - O = b)
  (h3 : C - O = c)
  (h4 : M - O = (1/2) • (A - O))
  (h5 : N - B = (1/2) • (C - B)) :
  N - M = (-1/2) • a + (1/2) • b + (1/2) • c := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_in_quadrilateral_l3768_376845


namespace NUMINAMATH_CALUDE_point_on_curve_with_slope_l3768_376886

def curve (x : ℝ) : ℝ := x^2 + x - 2

def tangent_slope (x : ℝ) : ℝ := 2*x + 1

theorem point_on_curve_with_slope : 
  ∃ (x y : ℝ), curve x = y ∧ tangent_slope x = 3 ∧ x = 1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_with_slope_l3768_376886


namespace NUMINAMATH_CALUDE_meal_cost_proof_l3768_376864

/-- Given the cost of two different meal combinations, 
    prove the cost of a single sandwich, coffee, and pie. -/
theorem meal_cost_proof (sandwich_cost coffee_cost pie_cost : ℚ) : 
  5 * sandwich_cost + 8 * coffee_cost + 2 * pie_cost = (5.40 : ℚ) →
  3 * sandwich_cost + 11 * coffee_cost + 2 * pie_cost = (4.95 : ℚ) →
  sandwich_cost + coffee_cost + pie_cost = (1.55 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_proof_l3768_376864


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l3768_376841

structure Grid :=
  (size : Nat)
  (shaded : List (Nat × Nat))

def isExternal (g : Grid) (pos : Nat × Nat) : Bool :=
  let (x, y) := pos
  x = 1 ∨ x = g.size ∨ y = 1 ∨ y = g.size

def countExternalEdges (g : Grid) : Nat :=
  g.shaded.foldl (fun acc pos =>
    acc + (if isExternal g pos then
             (if pos.1 = 1 then 1 else 0) +
             (if pos.1 = g.size then 1 else 0) +
             (if pos.2 = 1 then 1 else 0) +
             (if pos.2 = g.size then 1 else 0)
           else 0)
  ) 0

theorem shaded_region_perimeter (g : Grid) :
  g.size = 3 ∧
  g.shaded = [(1,2), (2,1), (2,3), (3,2)] →
  countExternalEdges g = 10 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l3768_376841


namespace NUMINAMATH_CALUDE_card_shop_problem_l3768_376874

/-- The total cost of cards bought from two boxes -/
def total_cost (cost1 cost2 : ℚ) (cards1 cards2 : ℕ) : ℚ :=
  cost1 * cards1 + cost2 * cards2

/-- Theorem: The total cost of 6 cards from each box is $18.00 -/
theorem card_shop_problem :
  total_cost (25/20) (35/20) 6 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_card_shop_problem_l3768_376874


namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l3768_376884

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * (a + b)

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  (∀ x y z : ℝ, star x (y + z) = star x y + star x z) ∧
  (∀ x y z : ℝ, x + star y z = star (x + y) (x + z)) ∧
  (∀ x y z : ℝ, star x (star y z) = star (star x y) (star x z)) →
  False :=
by sorry

end NUMINAMATH_CALUDE_no_distributive_laws_hold_l3768_376884


namespace NUMINAMATH_CALUDE_volume_maximized_at_10cm_l3768_376846

/-- The volume of a lidless container made from a rectangular sheet -/
def containerVolume (sheetLength sheetWidth height : ℝ) : ℝ :=
  (sheetLength - 2 * height) * (sheetWidth - 2 * height) * height

/-- The statement that the volume is maximized at a specific height -/
theorem volume_maximized_at_10cm (sheetLength sheetWidth : ℝ) 
  (hLength : sheetLength = 90) 
  (hWidth : sheetWidth = 48) :
  ∃ (maxHeight : ℝ), maxHeight = 10 ∧ 
  ∀ (h : ℝ), 0 < h → h < 24 → 
  containerVolume sheetLength sheetWidth h ≤ containerVolume sheetLength sheetWidth maxHeight :=
sorry

end NUMINAMATH_CALUDE_volume_maximized_at_10cm_l3768_376846


namespace NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l3768_376821

theorem square_difference_formula_inapplicable :
  ¬∃ (a b : ℝ → ℝ), ∀ x, (x + 1) * (1 + x) = a x ^ 2 - b x ^ 2 :=
sorry

end NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l3768_376821


namespace NUMINAMATH_CALUDE_class_average_score_l3768_376805

theorem class_average_score (total_questions : ℕ) 
  (score_3_percent : ℝ) (score_2_percent : ℝ) (score_1_percent : ℝ) (score_0_percent : ℝ) :
  total_questions = 3 →
  score_3_percent = 0.3 →
  score_2_percent = 0.5 →
  score_1_percent = 0.1 →
  score_0_percent = 0.1 →
  score_3_percent + score_2_percent + score_1_percent + score_0_percent = 1 →
  3 * score_3_percent + 2 * score_2_percent + 1 * score_1_percent + 0 * score_0_percent = 2 := by
sorry

end NUMINAMATH_CALUDE_class_average_score_l3768_376805


namespace NUMINAMATH_CALUDE_license_plate_count_l3768_376878

/-- The number of letters in the alphabet -/
def number_of_letters : ℕ := 26

/-- The number of digits (0-9) -/
def number_of_digits : ℕ := 10

/-- The number of even (or odd) digits -/
def number_of_even_digits : ℕ := 5

/-- The total number of license plates with 2 letters followed by 2 digits,
    where one digit is odd and the other is even -/
def total_license_plates : ℕ := number_of_letters^2 * number_of_digits * number_of_even_digits

theorem license_plate_count : total_license_plates = 33800 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3768_376878


namespace NUMINAMATH_CALUDE_exactly_one_hit_probability_l3768_376858

def hit_probability : ℝ := 0.5

theorem exactly_one_hit_probability :
  let p := hit_probability
  let q := 1 - p
  p * q + q * p = 0.5 := by sorry

end NUMINAMATH_CALUDE_exactly_one_hit_probability_l3768_376858


namespace NUMINAMATH_CALUDE_max_value_of_sum_l3768_376834

theorem max_value_of_sum (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) :
  x + y ≤ 11 / 4 ∧ ∃ (x₀ y₀ : ℝ), 5 * x₀ + 3 * y₀ ≤ 10 ∧ 3 * x₀ + 6 * y₀ ≤ 12 ∧ x₀ + y₀ = 11 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l3768_376834


namespace NUMINAMATH_CALUDE_sum_of_exponents_l3768_376823

theorem sum_of_exponents (a b c : ℕ+) : 
  4^(a : ℕ) * 5^(b : ℕ) * 6^(c : ℕ) = 8^8 * 9^9 * 10^10 → 
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 36 := by
sorry

end NUMINAMATH_CALUDE_sum_of_exponents_l3768_376823


namespace NUMINAMATH_CALUDE_flowerbed_fraction_is_five_thirty_sixths_l3768_376872

/-- Represents the dimensions and properties of a rectangular yard with flower beds. -/
structure YardWithFlowerBeds where
  length : ℝ
  width : ℝ
  trapezoid_side1 : ℝ
  trapezoid_side2 : ℝ
  
/-- Calculates the fraction of the yard occupied by flower beds. -/
def flowerbed_fraction (yard : YardWithFlowerBeds) : ℚ :=
  sorry

/-- Theorem stating that the fraction of the yard occupied by flower beds is 5/36. -/
theorem flowerbed_fraction_is_five_thirty_sixths 
  (yard : YardWithFlowerBeds) 
  (h1 : yard.length = 30)
  (h2 : yard.width = 6)
  (h3 : yard.trapezoid_side1 = 20)
  (h4 : yard.trapezoid_side2 = 30) :
  flowerbed_fraction yard = 5 / 36 :=
sorry

end NUMINAMATH_CALUDE_flowerbed_fraction_is_five_thirty_sixths_l3768_376872


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3768_376800

theorem quadratic_roots_product (c d : ℝ) : 
  (3 * c^2 + 9 * c - 21 = 0) → 
  (3 * d^2 + 9 * d - 21 = 0) → 
  (3 * c - 4) * (6 * d - 8) = 122 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3768_376800


namespace NUMINAMATH_CALUDE_obrienHatsAfterLoss_l3768_376894

/-- The number of hats Policeman O'Brien has after losing one -/
def obrienHats (simpsonHats : ℕ) : ℕ :=
  2 * simpsonHats + 5 - 1

theorem obrienHatsAfterLoss (simpsonHats : ℕ) (h : simpsonHats = 15) : 
  obrienHats simpsonHats = 34 := by
  sorry

end NUMINAMATH_CALUDE_obrienHatsAfterLoss_l3768_376894


namespace NUMINAMATH_CALUDE_mole_cannot_survive_winter_l3768_376859

/-- Represents the amount of grain in bags -/
structure GrainReserves where
  largeBags : ℕ
  smallBags : ℕ

/-- Represents the exchange rate between large and small bags -/
structure ExchangeRate where
  largeBags : ℕ
  smallBags : ℕ

/-- Represents the grain consumption per month -/
structure MonthlyConsumption where
  largeBags : ℕ

def canSurviveWinter (reserves : GrainReserves) (consumption : MonthlyConsumption) 
                     (exchangeRate : ExchangeRate) (months : ℕ) : Prop :=
  reserves.largeBags ≥ consumption.largeBags * months

theorem mole_cannot_survive_winter : 
  let reserves := GrainReserves.mk 20 32
  let consumption := MonthlyConsumption.mk 7
  let exchangeRate := ExchangeRate.mk 2 3
  let winterMonths := 3
  ¬(canSurviveWinter reserves consumption exchangeRate winterMonths) := by
  sorry

#check mole_cannot_survive_winter

end NUMINAMATH_CALUDE_mole_cannot_survive_winter_l3768_376859


namespace NUMINAMATH_CALUDE_selection_ways_l3768_376868

theorem selection_ways (male_count female_count : ℕ) 
  (h1 : male_count = 5)
  (h2 : female_count = 4) :
  male_count + female_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_l3768_376868


namespace NUMINAMATH_CALUDE_bruce_pizza_production_l3768_376804

/-- The number of batches of pizza dough Bruce can make in a week -/
def pizzas_per_week (batches_per_sack : ℕ) (sacks_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  batches_per_sack * sacks_per_day * days_in_week

/-- Theorem stating that Bruce can make 525 batches of pizza dough in a week -/
theorem bruce_pizza_production :
  pizzas_per_week 15 5 7 = 525 := by
  sorry

end NUMINAMATH_CALUDE_bruce_pizza_production_l3768_376804


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3768_376807

/-- A line passing through a point and perpendicular to another line -/
structure PerpendicularLine where
  point : ℝ × ℝ
  other_line : ℝ → ℝ → ℝ → ℝ

/-- The equation of the perpendicular line -/
def perpendicular_line_equation (l : PerpendicularLine) : ℝ → ℝ → ℝ → ℝ :=
  fun x y c => 3 * x + 2 * y + c

theorem perpendicular_line_through_point (l : PerpendicularLine)
  (h1 : l.point = (-1, 2))
  (h2 : l.other_line = fun x y c => 2 * x - 3 * y + c) :
  perpendicular_line_equation l (-1) 2 (-1) = 0 ∧
  perpendicular_line_equation l = fun x y c => 3 * x + 2 * y - 1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3768_376807


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3768_376810

theorem inverse_variation_problem (x w : ℝ) (h : ∃ (c : ℝ), ∀ (x w : ℝ), x^4 * w^(1/4) = c) :
  (x = 3 ∧ w = 16) → (x = 6 → w = 1/4096) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3768_376810


namespace NUMINAMATH_CALUDE_cab_delay_l3768_376881

theorem cab_delay (S : ℝ) (h : S > 0) : 
  let reduced_speed := (5 / 6) * S
  let usual_time := 30
  let new_time := usual_time * (S / reduced_speed)
  new_time - usual_time = 6 := by
sorry

end NUMINAMATH_CALUDE_cab_delay_l3768_376881


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l3768_376831

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (rate : ℚ) (interest : ℚ) (time : ℕ) :
  rate = 4 / 100 →
  interest = 128 →
  time = 4 →
  ∃ (principal : ℚ), principal * rate * time = interest ∧ principal = 800 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l3768_376831


namespace NUMINAMATH_CALUDE_vacation_pictures_l3768_376825

theorem vacation_pictures (zoo museum beach deleted : ℕ) :
  zoo = 120 →
  museum = 75 →
  beach = 45 →
  deleted = 93 →
  zoo + museum + beach - deleted = 147 :=
by sorry

end NUMINAMATH_CALUDE_vacation_pictures_l3768_376825


namespace NUMINAMATH_CALUDE_car_travel_time_ratio_l3768_376870

/-- Proves that the ratio of time taken at 70 km/h to the original time is 3:2 -/
theorem car_travel_time_ratio :
  let distance : ℝ := 630
  let original_time : ℝ := 6
  let new_speed : ℝ := 70
  let new_time : ℝ := distance / new_speed
  new_time / original_time = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_car_travel_time_ratio_l3768_376870


namespace NUMINAMATH_CALUDE_negation_of_existence_square_leq_power_of_two_negation_l3768_376847

theorem negation_of_existence (p : Nat → Prop) :
  (¬ ∃ n : Nat, p n) ↔ (∀ n : Nat, ¬ p n) := by sorry

theorem square_leq_power_of_two_negation :
  (¬ ∃ n : Nat, n^2 > 2^n) ↔ (∀ n : Nat, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_leq_power_of_two_negation_l3768_376847


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3768_376880

/-- Given two vectors a and b in ℝ², where a is perpendicular to a + b, prove that the second component of b is -6. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 = 4 ∧ a.2 = 2 ∧ b.1 = -2) 
  (perp : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) : 
  b.2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3768_376880


namespace NUMINAMATH_CALUDE_semicircle_is_arc_l3768_376842

-- Define a circle
def Circle := Set (ℝ × ℝ)

-- Define an arc
def Arc (c : Circle) := Set (ℝ × ℝ)

-- Define a semicircle
def Semicircle (c : Circle) := Arc c

-- Theorem: A semicircle is an arc
theorem semicircle_is_arc (c : Circle) : Semicircle c → Arc c := by
  sorry

end NUMINAMATH_CALUDE_semicircle_is_arc_l3768_376842


namespace NUMINAMATH_CALUDE_exists_polyhedron_with_specific_floating_state_l3768_376862

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  volume : ℝ
  surfaceArea : ℝ
  (volume_positive : volume > 0)
  (surfaceArea_positive : surfaceArea > 0)

/-- Represents the state of a floating polyhedron -/
structure FloatingState (p : ConvexPolyhedron) where
  submergedVolume : ℝ
  submergedSurfaceArea : ℝ
  (submerged_volume_valid : 0 ≤ submergedVolume ∧ submergedVolume ≤ p.volume)
  (submerged_area_valid : 0 ≤ submergedSurfaceArea ∧ submergedSurfaceArea ≤ p.surfaceArea)

/-- Theorem stating the existence of a convex polyhedron satisfying the given conditions -/
theorem exists_polyhedron_with_specific_floating_state :
  ∃ (p : ConvexPolyhedron) (f : FloatingState p),
    f.submergedVolume = 0.9 * p.volume ∧
    f.submergedSurfaceArea < 0.5 * p.surfaceArea :=
  sorry

end NUMINAMATH_CALUDE_exists_polyhedron_with_specific_floating_state_l3768_376862


namespace NUMINAMATH_CALUDE_uki_biscuits_per_day_l3768_376889

/-- Represents the daily production and pricing of bakery items -/
structure BakeryData where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  cupcakes_per_day : ℕ
  cookie_packets_per_day : ℕ
  total_earnings_five_days : ℝ

/-- Calculates the number of biscuit packets that can be baked daily -/
def biscuits_per_day (data : BakeryData) : ℕ :=
  20

/-- Theorem stating that given the bakery data, Uki can bake 20 packets of biscuits per day -/
theorem uki_biscuits_per_day (data : BakeryData)
    (h1 : data.cupcake_price = 1.5)
    (h2 : data.cookie_price = 2)
    (h3 : data.biscuit_price = 1)
    (h4 : data.cupcakes_per_day = 20)
    (h5 : data.cookie_packets_per_day = 10)
    (h6 : data.total_earnings_five_days = 350) :
    biscuits_per_day data = 20 := by
  sorry

end NUMINAMATH_CALUDE_uki_biscuits_per_day_l3768_376889


namespace NUMINAMATH_CALUDE_optimal_purchase_is_maximum_l3768_376844

/-- The cost of a red pencil in kopecks -/
def red_cost : ℕ := 27

/-- The cost of a blue pencil in kopecks -/
def blue_cost : ℕ := 23

/-- The total spending limit in kopecks -/
def spending_limit : ℕ := 940

/-- The maximum allowed difference between the number of blue and red pencils -/
def max_difference : ℕ := 10

/-- Represents the number of red and blue pencils purchased -/
structure PencilPurchase where
  red : ℕ
  blue : ℕ

/-- Checks if a purchase satisfies all conditions -/
def is_valid_purchase (p : PencilPurchase) : Prop :=
  p.red * red_cost + p.blue * blue_cost ≤ spending_limit ∧
  p.blue - p.red ≤ max_difference

/-- The optimal purchase of pencils -/
def optimal_purchase : PencilPurchase := ⟨14, 24⟩

theorem optimal_purchase_is_maximum :
  is_valid_purchase optimal_purchase ∧
  ∀ p : PencilPurchase, is_valid_purchase p → 
    p.red + p.blue ≤ optimal_purchase.red + optimal_purchase.blue :=
sorry

end NUMINAMATH_CALUDE_optimal_purchase_is_maximum_l3768_376844


namespace NUMINAMATH_CALUDE_jeds_stamp_cards_l3768_376895

/-- Jed's stamp card collection problem -/
theorem jeds_stamp_cards (X : ℕ) : 
  (X + 6 * 4 - 2 * 2 = 40) → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_jeds_stamp_cards_l3768_376895


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3768_376871

/-- An arithmetic progression with first three terms 2x - 3, 3x - 1, and 5x + 1 has x = 0 --/
theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ : ℝ := 2*x - 3
  let a₂ : ℝ := 3*x - 1
  let a₃ : ℝ := 5*x + 1
  (a₂ - a₁ = a₃ - a₂) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3768_376871


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3768_376896

def f (x : ℝ) := -x + 1

theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (1/2 : ℝ) 2 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (1/2 : ℝ) 2 → f x ≤ f c ∧
  f c = 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3768_376896


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3768_376856

theorem half_angle_quadrant (α : Real) (k : Int) : 
  (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3/2 * Real.pi) →
  ((∃ n : Int, 2 * n * Real.pi + Real.pi/2 < α/2 ∧ α/2 < 2 * n * Real.pi + 3/4 * Real.pi) ∨
   (∃ n : Int, (2 * n + 1) * Real.pi + Real.pi/2 < α/2 ∧ α/2 < (2 * n + 1) * Real.pi + 3/4 * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3768_376856


namespace NUMINAMATH_CALUDE_no_solutions_odd_fermat_prime_sum_l3768_376863

theorem no_solutions_odd_fermat_prime_sum (n : ℕ) (p : ℕ) 
  (h_n_odd : Odd n) (h_n_gt_1 : n > 1) (h_p_prime : Nat.Prime p) :
  ¬ ∃ (x y z : ℤ), x^n + y^n = z^n ∧ x + y = p :=
sorry

end NUMINAMATH_CALUDE_no_solutions_odd_fermat_prime_sum_l3768_376863


namespace NUMINAMATH_CALUDE_lauren_mail_total_l3768_376888

/-- The number of pieces of mail Lauren sent on Monday -/
def monday : ℕ := 65

/-- The number of pieces of mail Lauren sent on Tuesday -/
def tuesday : ℕ := monday + 10

/-- The number of pieces of mail Lauren sent on Wednesday -/
def wednesday : ℕ := tuesday - 5

/-- The number of pieces of mail Lauren sent on Thursday -/
def thursday : ℕ := wednesday + 15

/-- The total number of pieces of mail Lauren sent over four days -/
def total : ℕ := monday + tuesday + wednesday + thursday

theorem lauren_mail_total : total = 295 := by sorry

end NUMINAMATH_CALUDE_lauren_mail_total_l3768_376888


namespace NUMINAMATH_CALUDE_smallest_possible_a_l3768_376898

theorem smallest_possible_a (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a + b + c) / 3 = 20 →
  a ≤ b →
  b ≤ c →
  c ≥ 25 →
  ∃ (a' b' c' : ℕ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (a' + b' + c') / 3 = 20 ∧
    a' ≤ b' ∧
    b' ≤ c' ∧
    c' ≥ 25 ∧
    a' = 1 ∧
    ∀ (a'' : ℕ), a'' > 0 → 
      (∃ (b'' c'' : ℕ), b'' > 0 ∧ c'' > 0 ∧
        (a'' + b'' + c'') / 3 = 20 ∧
        a'' ≤ b'' ∧
        b'' ≤ c'' ∧
        c'' ≥ 25) →
      a'' ≥ a' := by
  sorry


end NUMINAMATH_CALUDE_smallest_possible_a_l3768_376898


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3768_376830

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 8

-- Define the points and foci
variable (P Q F₁ F₂ : ℝ × ℝ)

-- Define the chord passing through left focus
def chord_through_left_focus : Prop := 
  (∃ t : ℝ, P = F₁ + t • (Q - F₁)) ∨ (∃ t : ℝ, Q = F₁ + t • (P - F₁))

-- Define the length of PQ
def PQ_length : Prop := 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 7

-- Define F₂ as the right focus
def right_focus (F₂ : ℝ × ℝ) : Prop :=
  F₂.1 > 0 ∧ F₂.1^2 - F₂.2^2 = 8

-- Theorem statement
theorem hyperbola_triangle_perimeter 
  (h_hyperbola_P : hyperbola P.1 P.2)
  (h_hyperbola_Q : hyperbola Q.1 Q.2)
  (h_chord : chord_through_left_focus P Q F₁)
  (h_PQ_length : PQ_length P Q)
  (h_right_focus : right_focus F₂) :
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) +
  Real.sqrt ((Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2) +
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) =
  14 + 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3768_376830


namespace NUMINAMATH_CALUDE_circle_m_range_l3768_376815

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + x + y - m = 0

-- Define what it means for the equation to represent a circle
def represents_circle (m : ℝ) : Prop :=
  ∃ (center_x center_y radius : ℝ), 
    ∀ (x y : ℝ), circle_equation x y m ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2

-- Theorem statement
theorem circle_m_range (m : ℝ) :
  represents_circle m → m > -1/2 := by sorry

end NUMINAMATH_CALUDE_circle_m_range_l3768_376815


namespace NUMINAMATH_CALUDE_hemisphere_to_spheres_l3768_376883

/-- The radius of a sphere when a hemisphere is divided into equal parts -/
theorem hemisphere_to_spheres (r : Real) (n : Nat) (r_small : Real) : 
  r = 2 → n = 18 → (2/3 * π * r^3) = (n * (4/3 * π * r_small^3)) → r_small = (2/3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_to_spheres_l3768_376883


namespace NUMINAMATH_CALUDE_equation_solution_l3768_376875

theorem equation_solution :
  ∃ x : ℚ, (4 * x^2 + 6 * x + 2) / (x + 2) = 4 * x + 7 ∧ x = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3768_376875


namespace NUMINAMATH_CALUDE_ten_women_circular_reseating_l3768_376865

/-- The number of ways n women can be reseated in a circular arrangement,
    where each woman sits in her original seat or a seat adjacent to it. -/
def C : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => C (n + 1) + C n

theorem ten_women_circular_reseating : C 10 = 89 := by
  sorry

end NUMINAMATH_CALUDE_ten_women_circular_reseating_l3768_376865


namespace NUMINAMATH_CALUDE_repetend_of_three_thirteenths_l3768_376808

/-- The decimal representation of 3/13 has a 6-digit repetend of 230769 -/
theorem repetend_of_three_thirteenths : ∃ (n : ℕ), 
  (3 : ℚ) / 13 = (230769 : ℚ) / 999999 + n / (999999 * 13) := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_three_thirteenths_l3768_376808


namespace NUMINAMATH_CALUDE_bus_profit_analysis_l3768_376801

/-- Represents the monthly profit of a bus service -/
def monthly_profit (passengers : ℕ) : ℤ :=
  2 * passengers - 4000

theorem bus_profit_analysis :
  let break_even := 2000
  let profit_4230 := monthly_profit 4230
  -- 1. Independent variable is passengers, dependent is profit (implicit in the function definition)
  -- 2. Break-even point
  monthly_profit break_even = 0 ∧
  -- 3. Profit for 4230 passengers
  profit_4230 = 4460 := by
  sorry

end NUMINAMATH_CALUDE_bus_profit_analysis_l3768_376801


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3768_376843

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 2*x*(x+1) - 3*(x+1)
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3/2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3768_376843


namespace NUMINAMATH_CALUDE_cost_of_birdhouses_l3768_376890

-- Define the constants
def planks_per_birdhouse : ℕ := 7
def nails_per_birdhouse : ℕ := 20
def cost_per_nail : ℚ := 0.05
def cost_per_plank : ℕ := 3
def num_birdhouses : ℕ := 4

-- Define the theorem
theorem cost_of_birdhouses :
  (num_birdhouses * (planks_per_birdhouse * cost_per_plank +
   nails_per_birdhouse * cost_per_nail) : ℚ) = 88 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_birdhouses_l3768_376890


namespace NUMINAMATH_CALUDE_product_ABCD_eq_one_l3768_376820

/-- Given A, B, C, and D as defined, prove that their product is 1 -/
theorem product_ABCD_eq_one (A B C D : ℝ) 
  (hA : A = Real.sqrt 2008 + Real.sqrt 2009)
  (hB : B = -(Real.sqrt 2008) - Real.sqrt 2009)
  (hC : C = Real.sqrt 2008 - Real.sqrt 2009)
  (hD : D = Real.sqrt 2009 - Real.sqrt 2008) : 
  A * B * C * D = 1 := by
  sorry


end NUMINAMATH_CALUDE_product_ABCD_eq_one_l3768_376820


namespace NUMINAMATH_CALUDE_triple_composition_even_l3768_376873

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem triple_composition_even (g : ℝ → ℝ) (h : IsEven g) : IsEven (fun x ↦ g (g (g x))) := by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l3768_376873


namespace NUMINAMATH_CALUDE_kingsleys_friends_l3768_376809

theorem kingsleys_friends (chairs_per_trip : ℕ) (total_trips : ℕ) (total_chairs : ℕ) :
  chairs_per_trip = 5 →
  total_trips = 10 →
  total_chairs = 250 →
  (total_chairs / (chairs_per_trip * total_trips)) - 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_kingsleys_friends_l3768_376809


namespace NUMINAMATH_CALUDE_subset_sum_property_l3768_376803

theorem subset_sum_property (n : ℕ) (hn : n > 1) :
  ∀ (S : Finset ℕ), S ⊆ Finset.range (2 * n) → S.card = n + 2 →
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = c :=
by sorry

end NUMINAMATH_CALUDE_subset_sum_property_l3768_376803


namespace NUMINAMATH_CALUDE_parabola_tangent_and_intersection_l3768_376814

/-- Parabola in the first quadrant -/
structure Parabola where
  n : ℝ
  pos_n : n > 0

/-- Point on the parabola -/
structure ParabolaPoint (c : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = c.n * x
  first_quadrant : x > 0 ∧ y > 0

/-- Line with slope and y-intercept -/
structure Line where
  m : ℝ
  b : ℝ

/-- Theorem about parabola tangent and intersection properties -/
theorem parabola_tangent_and_intersection
  (c : Parabola)
  (p : ParabolaPoint c)
  (h1 : p.x = 2)
  (h2 : (p.x + c.n / 4)^2 + p.y^2 = (5/2)^2) -- Distance from P to focus is 5/2
  (l₂ : Line)
  (h3 : l₂.m ≠ 0) :
  -- 1. The tangent at P intersects x-axis at (-2, 0)
  ∃ (q : ℝ × ℝ), q = (-2, 0) ∧
    (∃ (k : ℝ), k * (q.1 - p.x) + p.y = q.2 ∧ 
      ∀ (x y : ℝ), y^2 = c.n * x → (y - p.y) = k * (x - p.x) → x = q.1 ∧ y = q.2) ∧
  -- 2. If slopes of PA, PE, PB form arithmetic sequence, l₂ passes through (2, 0)
  (∀ (a b e : ℝ × ℝ),
    (a.2)^2 = c.n * a.1 ∧ (b.2)^2 = c.n * b.1 ∧ -- A and B on parabola
    a.1 = l₂.m * a.2 + l₂.b ∧ b.1 = l₂.m * b.2 + l₂.b ∧ -- A and B on l₂
    e.1 = -2 ∧ e.2 = -(l₂.b + 2) / l₂.m → -- E on l₁
    (((a.2 - p.y) / (a.1 - p.x) + (b.2 - p.y) / (b.1 - p.x)) / 2 = (e.2 - p.y) / (e.1 - p.x)) →
    l₂.b = 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_and_intersection_l3768_376814


namespace NUMINAMATH_CALUDE_tegwens_family_size_l3768_376867

theorem tegwens_family_size :
  ∀ (g b : ℕ),
  g > 0 →  -- At least one girl (Tegwen)
  b = g - 1 →  -- Tegwen has same number of brothers as sisters
  g = (3 * (b - 1)) / 2 →  -- Each brother has 50% more sisters than brothers
  g + b = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_tegwens_family_size_l3768_376867


namespace NUMINAMATH_CALUDE_divisible_by_sixteen_l3768_376840

theorem divisible_by_sixteen (m n : ℤ) : ∃ k : ℤ, (5*m + 3*n + 1)^5 * (3*m + n + 4)^4 = 16*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_sixteen_l3768_376840


namespace NUMINAMATH_CALUDE_single_elimination_256_players_l3768_376879

/-- A single-elimination tournament structure -/
structure Tournament :=
  (num_players : ℕ)
  (is_single_elimination : Bool)

/-- The number of games needed to determine a champion in a single-elimination tournament -/
def games_to_champion (t : Tournament) : ℕ :=
  t.num_players - 1

/-- Theorem: In a single-elimination tournament with 256 players, 255 games are needed to determine the champion -/
theorem single_elimination_256_players :
  ∀ t : Tournament, t.num_players = 256 → t.is_single_elimination = true →
  games_to_champion t = 255 :=
by
  sorry

end NUMINAMATH_CALUDE_single_elimination_256_players_l3768_376879


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l3768_376824

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 
    1043 = 23 * q + r ∧ 
    r > 0 ∧ 
    ∀ (q' r' : ℕ), 1043 = 23 * q' + r' ∧ r' > 0 → q' - r' ≤ q - r ∧ 
    q - r = 37 := by
  sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l3768_376824


namespace NUMINAMATH_CALUDE_list_number_property_l3768_376837

theorem list_number_property (L : List ℝ) (n : ℝ) :
  L.length = 21 →
  L.Nodup →
  n ∈ L →
  n = 0.2 * L.sum →
  n = 5 * ((L.sum - n) / 20) :=
by sorry

end NUMINAMATH_CALUDE_list_number_property_l3768_376837


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l3768_376891

-- Define the conditions
def p (x : ℝ) : Prop := |x - 4| ≤ 6
def q (x m : ℝ) : Prop := x ≤ 1 + m

-- State the theorem
theorem sufficient_condition_range (m : ℝ) :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) ↔ m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l3768_376891


namespace NUMINAMATH_CALUDE_margie_change_theorem_l3768_376882

-- Define the problem parameters
def num_apples : ℕ := 5
def cost_per_apple : ℚ := 30 / 100  -- 30 cents in dollars
def discount_rate : ℚ := 10 / 100   -- 10% discount
def paid_amount : ℚ := 10           -- 10-dollar bill

-- Define the theorem
theorem margie_change_theorem :
  let total_cost := num_apples * cost_per_apple
  let discounted_cost := total_cost * (1 - discount_rate)
  let change := paid_amount - discounted_cost
  change = 865 / 100 := by
sorry


end NUMINAMATH_CALUDE_margie_change_theorem_l3768_376882


namespace NUMINAMATH_CALUDE_line_equal_intercepts_l3768_376850

/-- A line passing through (1,2) with equal x and y intercepts has equation x+y-3=0 or 2x-y=0 -/
theorem line_equal_intercepts :
  ∀ (L : Set (ℝ × ℝ)), 
    ((1, 2) ∈ L) →
    (∃ a : ℝ, a ≠ 0 ∧ (a, 0) ∈ L ∧ (0, a) ∈ L) →
    (∀ x y : ℝ, (x, y) ∈ L ↔ (x + y = 3 ∨ 2*x - y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equal_intercepts_l3768_376850


namespace NUMINAMATH_CALUDE_gino_bears_count_l3768_376893

/-- The number of brown bears Gino has -/
def brown_bears : ℕ := 15

/-- The number of white bears Gino has -/
def white_bears : ℕ := 24

/-- The number of black bears Gino has -/
def black_bears : ℕ := 27

/-- The total number of bears Gino has -/
def total_bears : ℕ := brown_bears + white_bears + black_bears

theorem gino_bears_count : total_bears = 66 := by
  sorry

end NUMINAMATH_CALUDE_gino_bears_count_l3768_376893


namespace NUMINAMATH_CALUDE_square_difference_l3768_376897

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3768_376897


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3768_376885

theorem polynomial_factorization (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3768_376885


namespace NUMINAMATH_CALUDE_thursday_spending_l3768_376832

def monday_savings : ℕ := 15
def tuesday_savings : ℕ := 28
def wednesday_savings : ℕ := 13

def total_savings : ℕ := monday_savings + tuesday_savings + wednesday_savings

theorem thursday_spending :
  (total_savings : ℚ) / 2 = 28 := by sorry

end NUMINAMATH_CALUDE_thursday_spending_l3768_376832


namespace NUMINAMATH_CALUDE_only_blue_possible_all_blue_possible_l3768_376819

/-- Represents the number of sheep of each color -/
structure SheepCounts where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- Represents a valid transformation of sheep colors -/
inductive SheepTransform : SheepCounts → SheepCounts → Prop where
  | blue_red_to_green : ∀ b r g, SheepTransform ⟨b+1, r+1, g-2⟩ ⟨b, r, g⟩
  | blue_green_to_red : ∀ b r g, SheepTransform ⟨b+1, r-2, g+1⟩ ⟨b, r, g⟩
  | red_green_to_blue : ∀ b r g, SheepTransform ⟨b-2, r+1, g+1⟩ ⟨b, r, g⟩

/-- Represents a sequence of transformations -/
def TransformSequence : SheepCounts → SheepCounts → Prop :=
  Relation.ReflTransGen SheepTransform

/-- The theorem stating that only blue is possible as the final uniform color -/
theorem only_blue_possible (initial : SheepCounts) (final : SheepCounts) :
  initial = ⟨22, 18, 15⟩ →
  TransformSequence initial final →
  (final.blue = 0 ∧ final.red = 55) ∨ (final.blue = 0 ∧ final.green = 55) →
  False :=
sorry

/-- The theorem stating that all blue is possible -/
theorem all_blue_possible (initial : SheepCounts) (final : SheepCounts) :
  initial = ⟨22, 18, 15⟩ →
  ∃ final, TransformSequence initial final ∧ final = ⟨55, 0, 0⟩ :=
sorry

end NUMINAMATH_CALUDE_only_blue_possible_all_blue_possible_l3768_376819


namespace NUMINAMATH_CALUDE_prime_cube_difference_equation_l3768_376817

theorem prime_cube_difference_equation :
  ∀ p q r : ℕ,
    Prime p → Prime q → Prime r →
    5 * p = q^3 - r^3 →
    p = 67 ∧ q = 7 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_cube_difference_equation_l3768_376817


namespace NUMINAMATH_CALUDE_chameleons_changed_count_l3768_376839

/-- Represents the number of chameleons that changed color --/
def chameleons_changed (total : ℕ) (blue_factor : ℕ) (red_factor : ℕ) : ℕ :=
  let initial_blue := blue_factor * (total / (blue_factor + 1))
  total - initial_blue - (total - initial_blue) / red_factor

/-- Theorem stating that 80 chameleons changed color under the given conditions --/
theorem chameleons_changed_count :
  chameleons_changed 140 5 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_chameleons_changed_count_l3768_376839


namespace NUMINAMATH_CALUDE_intersection_x_axis_intersection_y_axis_l3768_376866

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 2 * x - 1

-- Theorem for the intersection with x-axis
theorem intersection_x_axis :
  ∃ (x : ℝ), line_equation x 0 ∧ x = 0.5 := by sorry

-- Theorem for the intersection with y-axis
theorem intersection_y_axis :
  line_equation 0 (-1) := by sorry

end NUMINAMATH_CALUDE_intersection_x_axis_intersection_y_axis_l3768_376866


namespace NUMINAMATH_CALUDE_stock_investment_l3768_376877

theorem stock_investment (dividend_rate : ℚ) (dividend_earned : ℚ) (stock_price : ℚ) :
  dividend_rate = 9 / 100 →
  dividend_earned = 120 →
  stock_price = 135 →
  ∃ (investment : ℚ), investment = 1800 ∧ 
    dividend_earned = dividend_rate * (investment * 100 / stock_price) :=
by sorry

end NUMINAMATH_CALUDE_stock_investment_l3768_376877


namespace NUMINAMATH_CALUDE_box_2_neg2_3_l3768_376828

/-- Define the box operation for integers a, b, and c -/
def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

/-- Theorem stating that box(2, -2, 3) = 69/4 -/
theorem box_2_neg2_3 : box 2 (-2) 3 = 69/4 := by sorry

end NUMINAMATH_CALUDE_box_2_neg2_3_l3768_376828


namespace NUMINAMATH_CALUDE_harris_dog_carrot_cost_l3768_376869

/-- The annual cost of carrots for Harris's dog -/
def annual_carrot_cost (carrots_per_day : ℕ) (carrots_per_bag : ℕ) (cost_per_bag : ℚ) (days_per_year : ℕ) : ℚ :=
  (days_per_year * carrots_per_day / carrots_per_bag) * cost_per_bag

/-- Theorem stating the annual cost of carrots for Harris's dog -/
theorem harris_dog_carrot_cost :
  annual_carrot_cost 1 5 2 365 = 146 := by
  sorry

end NUMINAMATH_CALUDE_harris_dog_carrot_cost_l3768_376869


namespace NUMINAMATH_CALUDE_ratio_composition_l3768_376835

theorem ratio_composition (a b c : ℚ) 
  (h1 : a / b = 11 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 11 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_composition_l3768_376835


namespace NUMINAMATH_CALUDE_similar_polygons_area_sum_l3768_376833

/-- Given two similar polygons with corresponding sides a and b, 
    we can construct a third similar polygon with side c -/
theorem similar_polygons_area_sum 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (similar : ∃ (k : ℝ), k > 0 ∧ b = k * a) :
  ∃ (c : ℝ), 
    c > 0 ∧ 
    c^2 = a^2 + b^2 ∧ 
    ∃ (k : ℝ), k > 0 ∧ c = k * a ∧
    c^2 / a^2 = (a^2 + b^2) / a^2 :=
by sorry

end NUMINAMATH_CALUDE_similar_polygons_area_sum_l3768_376833


namespace NUMINAMATH_CALUDE_company_employee_increase_l3768_376802

theorem company_employee_increase (january_employees : ℝ) (increase_percentage : ℝ) :
  january_employees = 434.7826086956522 →
  increase_percentage = 15 →
  january_employees * (1 + increase_percentage / 100) = 500 := by
sorry

end NUMINAMATH_CALUDE_company_employee_increase_l3768_376802


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l3768_376838

theorem smallest_integer_with_remainder_two (n : ℕ) : 
  (n > 1) →
  (n % 3 = 2) →
  (n % 5 = 2) →
  (n % 7 = 2) →
  (∀ m : ℕ, m > 1 ∧ m % 3 = 2 ∧ m % 5 = 2 ∧ m % 7 = 2 → m ≥ n) →
  n = 107 :=
by
  sorry

#check smallest_integer_with_remainder_two

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l3768_376838
