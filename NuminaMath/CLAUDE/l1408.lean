import Mathlib

namespace NUMINAMATH_CALUDE_data_ratio_l1408_140878

theorem data_ratio (a b c : ℝ) 
  (h1 : a = b - c) 
  (h2 : a = 12) 
  (h3 : a + b + c = 96) : 
  b / a = 4 := by
sorry

end NUMINAMATH_CALUDE_data_ratio_l1408_140878


namespace NUMINAMATH_CALUDE_harmonic_table_sum_remainder_l1408_140865

theorem harmonic_table_sum_remainder : ∃ k : ℕ, (2^2007 - 1) / 2007 ≡ 1 [MOD 2008] := by
  sorry

end NUMINAMATH_CALUDE_harmonic_table_sum_remainder_l1408_140865


namespace NUMINAMATH_CALUDE_even_digits_base7_528_l1408_140813

/-- Converts a natural number from base 10 to base 7 --/
def toBase7 (n : ℕ) : List ℕ := sorry

/-- Counts the number of even digits in a list of natural numbers --/
def countEvenDigits (digits : List ℕ) : ℕ := sorry

/-- The number of even digits in the base-7 representation of 528 is 0 --/
theorem even_digits_base7_528 :
  countEvenDigits (toBase7 528) = 0 := by sorry

end NUMINAMATH_CALUDE_even_digits_base7_528_l1408_140813


namespace NUMINAMATH_CALUDE_happy_snakes_not_purple_l1408_140812

structure Snake where
  purple : Bool
  happy : Bool
  can_add : Bool
  can_subtract : Bool

def Tom's_collection : Set Snake := sorry

theorem happy_snakes_not_purple :
  ∀ (s : Snake),
  s ∈ Tom's_collection →
  (s.happy → s.can_add) ∧
  (s.purple → ¬s.can_subtract) ∧
  (¬s.can_subtract → ¬s.can_add) →
  (s.happy → ¬s.purple) := by
  sorry

#check happy_snakes_not_purple

end NUMINAMATH_CALUDE_happy_snakes_not_purple_l1408_140812


namespace NUMINAMATH_CALUDE_number_multiplication_l1408_140892

theorem number_multiplication (x : ℤ) : x - 27 = 46 → x * 46 = 3358 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_l1408_140892


namespace NUMINAMATH_CALUDE_ellen_smoothie_total_cups_l1408_140883

/-- Represents the ingredients used in Ellen's smoothie recipe -/
structure SmoothieIngredients where
  strawberries : Float
  yogurt : Float
  orange_juice : Float
  honey : Float
  chia_seeds : Float
  spinach : Float

/-- Conversion factors for measurements -/
def ounce_to_cup : Float := 0.125
def tablespoon_to_cup : Float := 0.0625

/-- Ellen's smoothie recipe -/
def ellen_smoothie : SmoothieIngredients := {
  strawberries := 0.2,
  yogurt := 0.1,
  orange_juice := 0.2,
  honey := 1 * ounce_to_cup,
  chia_seeds := 2 * tablespoon_to_cup,
  spinach := 0.5
}

/-- Theorem stating the total cups of ingredients in Ellen's smoothie -/
theorem ellen_smoothie_total_cups : 
  ellen_smoothie.strawberries + 
  ellen_smoothie.yogurt + 
  ellen_smoothie.orange_juice + 
  ellen_smoothie.honey + 
  ellen_smoothie.chia_seeds + 
  ellen_smoothie.spinach = 1.25 := by sorry

end NUMINAMATH_CALUDE_ellen_smoothie_total_cups_l1408_140883


namespace NUMINAMATH_CALUDE_not_p_and_not_not_p_l1408_140807

theorem not_p_and_not_not_p (p : Prop) : ¬(p ∧ ¬p) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_not_p_l1408_140807


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1408_140816

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1408_140816


namespace NUMINAMATH_CALUDE_part1_part2_l1408_140828

-- Define the conditions p and q as functions of x and m
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0

def q (x m : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part 1
theorem part1 (m : ℝ) (h : m > 0) :
  (∀ x, p x → q x m) → m ≥ 4 := by sorry

-- Part 2
theorem part2 (x : ℝ) :
  (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) →
  (x ∈ Set.Icc (-3) (-2) ∪ Set.Ioc 6 7) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1408_140828


namespace NUMINAMATH_CALUDE_playground_fundraiser_correct_l1408_140874

def playground_fundraiser (johnson_amount sutton_amount rollin_amount total_amount : ℝ) : Prop :=
  johnson_amount = 2300 ∧
  johnson_amount = 2 * sutton_amount ∧
  rollin_amount = 8 * sutton_amount ∧
  rollin_amount = total_amount / 3 ∧
  total_amount * 0.98 = 27048

theorem playground_fundraiser_correct :
  ∃ johnson_amount sutton_amount rollin_amount total_amount : ℝ,
    playground_fundraiser johnson_amount sutton_amount rollin_amount total_amount :=
by
  sorry

end NUMINAMATH_CALUDE_playground_fundraiser_correct_l1408_140874


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l1408_140829

/-- Calculates the average speed of a car trip given the following conditions:
    - The trip lasts for 8 hours
    - The car averages 50 mph for the first 4 hours
    - The car averages 80 mph for the remaining 4 hours
-/
theorem car_trip_average_speed :
  let total_time : ℝ := 8
  let first_segment_time : ℝ := 4
  let second_segment_time : ℝ := total_time - first_segment_time
  let first_segment_speed : ℝ := 50
  let second_segment_speed : ℝ := 80
  let total_distance : ℝ := first_segment_speed * first_segment_time + second_segment_speed * second_segment_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 65 := by sorry

end NUMINAMATH_CALUDE_car_trip_average_speed_l1408_140829


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1408_140809

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  let side := d / Real.sqrt 2
  side * side = 72 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1408_140809


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1408_140826

theorem sqrt_product_simplification (x : ℝ) :
  Real.sqrt (50 * x^2) * Real.sqrt (18 * x^3) * Real.sqrt (98 * x) = 210 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1408_140826


namespace NUMINAMATH_CALUDE_extreme_values_imply_b_zero_l1408_140870

/-- A cubic function with extreme values at 1 and -1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x * (a * x^2 + b * x + c)

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem extreme_values_imply_b_zero (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : f' a b c 1 = 0) (h3 : f' a b c (-1) = 0) : b = 0 := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_imply_b_zero_l1408_140870


namespace NUMINAMATH_CALUDE_twenty_four_point_game_l1408_140842

theorem twenty_four_point_game : 
  let a := 3
  let b := 3
  let c := 8
  let d := 8
  8 / (a - c / b) = 24 := by
  sorry

end NUMINAMATH_CALUDE_twenty_four_point_game_l1408_140842


namespace NUMINAMATH_CALUDE_triangle_trigonometric_expression_l1408_140843

theorem triangle_trigonometric_expression (X Y Z : ℝ) : 
  (13 : ℝ) ^ 2 = X ^ 2 + Y ^ 2 - 2 * X * Y * Real.cos Z →
  (14 : ℝ) ^ 2 = X ^ 2 + Z ^ 2 - 2 * X * Z * Real.cos Y →
  (15 : ℝ) ^ 2 = Y ^ 2 + Z ^ 2 - 2 * Y * Z * Real.cos X →
  (Real.cos ((X - Y) / 2) / Real.sin (Z / 2)) - (Real.sin ((X - Y) / 2) / Real.cos (Z / 2)) = 28 / 13 := by
  sorry


end NUMINAMATH_CALUDE_triangle_trigonometric_expression_l1408_140843


namespace NUMINAMATH_CALUDE_rectangle_perimeters_l1408_140879

def is_valid_perimeter (p : ℕ) : Prop :=
  ∃ (x y : ℕ), 
    (x > 0 ∧ y > 0) ∧
    (3 * (2 * (x + y)) = 10) ∧
    (p = 2 * (x + y) ∨ p = 2 * (3 * x) ∨ p = 2 * (3 * y))

theorem rectangle_perimeters : 
  {p : ℕ | is_valid_perimeter p} = {14, 16, 18, 22, 26} :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeters_l1408_140879


namespace NUMINAMATH_CALUDE_initial_money_calculation_l1408_140825

theorem initial_money_calculation (remaining_money : ℝ) (spent_percentage : ℝ) (initial_money : ℝ) : 
  remaining_money = 2800 →
  spent_percentage = 0.3 →
  initial_money * (1 - spent_percentage) = remaining_money →
  initial_money = 4000 := by
sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l1408_140825


namespace NUMINAMATH_CALUDE_smallest_five_digit_square_cube_l1408_140833

theorem smallest_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n ≤ 99999) ∧ 
  (∃ a : ℕ, n = a^2) ∧ 
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m ≤ 99999 ∧ (∃ c : ℕ, m = c^2) ∧ (∃ d : ℕ, m = d^3) → m ≥ n) ∧
  n = 15625 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_square_cube_l1408_140833


namespace NUMINAMATH_CALUDE_bake_sale_group_l1408_140848

theorem bake_sale_group (p : ℕ) : 
  p > 0 → 
  (p : ℚ) / 2 - 2 = (2 * p : ℚ) / 5 → 
  (p : ℚ) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_group_l1408_140848


namespace NUMINAMATH_CALUDE_digit_symmetrical_equation_l1408_140821

theorem digit_symmetrical_equation (a b : ℤ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10*a + b) * (100*b + 10*(a + b) + a) = (100*a + 10*(a + b) + b) * (10*b + a) := by
  sorry

end NUMINAMATH_CALUDE_digit_symmetrical_equation_l1408_140821


namespace NUMINAMATH_CALUDE_root_sum_relation_l1408_140868

theorem root_sum_relation (a b c d e : ℝ) (ha : a ≠ 0) : 
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 4 ∨ x = -3 ∨ x = 0) →
  (b + c) / a = -13 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_relation_l1408_140868


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l1408_140814

theorem rectangle_side_ratio (a b c d : ℝ) (h : a / c = b / d ∧ a / c = 4 / 5) : b / d = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l1408_140814


namespace NUMINAMATH_CALUDE_geometry_biology_overlap_difference_l1408_140857

theorem geometry_biology_overlap_difference (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h1 : total = 232)
  (h2 : geometry = 144)
  (h3 : biology = 119) :
  let max_overlap := min geometry biology
  let min_overlap := geometry + biology - total
  max_overlap - min_overlap = 88 := by
sorry

end NUMINAMATH_CALUDE_geometry_biology_overlap_difference_l1408_140857


namespace NUMINAMATH_CALUDE_sams_watermelons_l1408_140851

theorem sams_watermelons (grown : ℕ) (eaten : ℕ) (h1 : grown = 4) (h2 : eaten = 3) :
  grown - eaten = 1 := by
  sorry

end NUMINAMATH_CALUDE_sams_watermelons_l1408_140851


namespace NUMINAMATH_CALUDE_a_5_value_l1408_140845

def geometric_sequence_with_ratio_difference (a : ℕ → ℝ) (k : ℝ) :=
  ∀ n, a (n + 2) / a (n + 1) - a (n + 1) / a n = k

theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence_with_ratio_difference a 2 →
  a 1 = 1 →
  a 2 = 2 →
  a 5 = 384 := by
sorry

end NUMINAMATH_CALUDE_a_5_value_l1408_140845


namespace NUMINAMATH_CALUDE_sum_three_digit_even_integers_eq_247050_l1408_140802

/-- The sum of all three-digit positive even integers -/
def sum_three_digit_even_integers : ℕ :=
  let first : ℕ := 100  -- First three-digit even integer
  let last : ℕ := 998   -- Last three-digit even integer
  let count : ℕ := (last - first) / 2 + 1  -- Number of terms
  count * (first + last) / 2

/-- Theorem stating that the sum of all three-digit positive even integers is 247050 -/
theorem sum_three_digit_even_integers_eq_247050 :
  sum_three_digit_even_integers = 247050 := by
  sorry

#eval sum_three_digit_even_integers

end NUMINAMATH_CALUDE_sum_three_digit_even_integers_eq_247050_l1408_140802


namespace NUMINAMATH_CALUDE_concert_attendance_l1408_140877

theorem concert_attendance (adults : ℕ) (children : ℕ) : 
  children = 3 * adults →
  7 * adults + 3 * children = 6000 →
  adults + children = 1500 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l1408_140877


namespace NUMINAMATH_CALUDE_composition_result_l1408_140890

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 3 * x + 4

-- State the theorem
theorem composition_result : f (g (f 3)) = 79 := by
  sorry

end NUMINAMATH_CALUDE_composition_result_l1408_140890


namespace NUMINAMATH_CALUDE_max_sum_at_five_l1408_140803

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The theorem stating when S_n reaches its maximum value -/
theorem max_sum_at_five (seq : ArithmeticSequence)
    (h1 : seq.a 1 + seq.a 5 + seq.a 9 = 6)
    (h2 : seq.S 11 = -11) :
    ∃ n_max : ℕ, n_max = 5 ∧ ∀ n : ℕ, seq.S n ≤ seq.S n_max := by
  sorry

end NUMINAMATH_CALUDE_max_sum_at_five_l1408_140803


namespace NUMINAMATH_CALUDE_max_value_of_f_l1408_140894

def f (x : ℕ) : ℤ := 2 * x - 3

def S : Set ℕ := {x : ℕ | 1 ≤ x ∧ x ≤ 10/3}

theorem max_value_of_f :
  ∃ (m : ℤ), m = 3 ∧ ∀ (x : ℕ), x ∈ S → f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1408_140894


namespace NUMINAMATH_CALUDE_hypotenuse_length_l1408_140864

/-- A right triangle with specific properties -/
structure RightTriangle where
  /-- First leg of the triangle -/
  a : ℝ
  /-- Second leg of the triangle -/
  b : ℝ
  /-- Hypotenuse of the triangle -/
  c : ℝ
  /-- The triangle is right-angled -/
  right_angle : a^2 + b^2 = c^2
  /-- The perimeter of the triangle is 40 -/
  perimeter : a + b + c = 40
  /-- The area of the triangle is 30 -/
  area : a * b / 2 = 30
  /-- The ratio of the legs is 3:4 -/
  leg_ratio : 3 * a = 4 * b

theorem hypotenuse_length (t : RightTriangle) : t.c = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l1408_140864


namespace NUMINAMATH_CALUDE_grace_weeding_hours_l1408_140838

/-- Represents Grace's landscaping business earnings in September --/
def graces_earnings (mowing_rate : ℕ) (weeding_rate : ℕ) (mulching_rate : ℕ)
                    (mowing_hours : ℕ) (weeding_hours : ℕ) (mulching_hours : ℕ) : ℕ :=
  mowing_rate * mowing_hours + weeding_rate * weeding_hours + mulching_rate * mulching_hours

/-- Theorem stating that Grace spent 9 hours pulling weeds in September --/
theorem grace_weeding_hours :
  ∀ (mowing_rate weeding_rate mulching_rate mowing_hours mulching_hours total_earnings : ℕ),
    mowing_rate = 6 →
    weeding_rate = 11 →
    mulching_rate = 9 →
    mowing_hours = 63 →
    mulching_hours = 10 →
    total_earnings = 567 →
    ∃ (weeding_hours : ℕ),
      graces_earnings mowing_rate weeding_rate mulching_rate mowing_hours weeding_hours mulching_hours = total_earnings ∧
      weeding_hours = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_grace_weeding_hours_l1408_140838


namespace NUMINAMATH_CALUDE_paving_stone_width_l1408_140800

/-- The width of a paving stone given the dimensions of a rectangular courtyard and the number of stones required to pave it. -/
theorem paving_stone_width 
  (courtyard_length : ℝ) 
  (courtyard_width : ℝ) 
  (num_stones : ℕ) 
  (stone_length : ℝ) 
  (h1 : courtyard_length = 50) 
  (h2 : courtyard_width = 16.5) 
  (h3 : num_stones = 165) 
  (h4 : stone_length = 2.5) : 
  ∃ (stone_width : ℝ), stone_width = 2 ∧ 
    courtyard_length * courtyard_width = ↑num_stones * stone_length * stone_width := by
  sorry

end NUMINAMATH_CALUDE_paving_stone_width_l1408_140800


namespace NUMINAMATH_CALUDE_difference_of_squares_153_147_l1408_140820

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_153_147_l1408_140820


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1408_140891

/-- A cylinder with a square axial section of area 4 has a surface area of 6π -/
theorem cylinder_surface_area (r h : Real) : 
  r * h = 2 →  -- axial section is a square
  r * r = 1 →  -- area of square is 4
  2 * Real.pi * r * r + 2 * Real.pi * r * h = 6 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_cylinder_surface_area_l1408_140891


namespace NUMINAMATH_CALUDE_unique_m_value_l1408_140895

theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_value_l1408_140895


namespace NUMINAMATH_CALUDE_shop_profit_days_l1408_140882

theorem shop_profit_days (mean_profit : ℝ) (first_15_mean : ℝ) (last_15_mean : ℝ)
  (h1 : mean_profit = 350)
  (h2 : first_15_mean = 245)
  (h3 : last_15_mean = 455) :
  (mean_profit * (15 + 15) = first_15_mean * 15 + last_15_mean * 15) → (15 + 15 = 30) :=
by sorry

end NUMINAMATH_CALUDE_shop_profit_days_l1408_140882


namespace NUMINAMATH_CALUDE_place_two_before_eq_l1408_140832

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_lt_10 : hundreds < 10
  t_lt_10 : tens < 10
  u_lt_10 : units < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def to_nat (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Represents the operation of placing 2 before a three-digit number -/
def place_two_before (n : ThreeDigitNumber) : ℕ :=
  2000 + 100 * n.hundreds + 10 * n.tens + n.units

/-- Theorem stating that placing 2 before a three-digit number results in 2000 + 100h + 10t + u -/
theorem place_two_before_eq (n : ThreeDigitNumber) :
  place_two_before n = 2000 + 100 * n.hundreds + 10 * n.tens + n.units := by
  sorry

end NUMINAMATH_CALUDE_place_two_before_eq_l1408_140832


namespace NUMINAMATH_CALUDE_circle_tangency_radius_l1408_140839

theorem circle_tangency_radius 
  (d1 d2 r1 r2 r y : ℝ) 
  (h1 : d1 < d2) 
  (h2 : r1 = d1 / 2) 
  (h3 : r2 = d2 / 2) 
  (h4 : (r + r1)^2 = (r - 2*r2 - r1)^2 + y^2) 
  (h5 : (r + r2)^2 = (r - r2)^2 + y^2) : 
  r = ((d1 + d2) * d2) / (2 * d1) := by
sorry

end NUMINAMATH_CALUDE_circle_tangency_radius_l1408_140839


namespace NUMINAMATH_CALUDE_isabel_songs_total_l1408_140886

/-- The number of country albums Isabel bought -/
def country_albums : ℕ := 6

/-- The number of pop albums Isabel bought -/
def pop_albums : ℕ := 2

/-- The number of songs in each album -/
def songs_per_album : ℕ := 9

/-- The total number of songs Isabel bought -/
def total_songs : ℕ := (country_albums + pop_albums) * songs_per_album

theorem isabel_songs_total : total_songs = 72 := by
  sorry

end NUMINAMATH_CALUDE_isabel_songs_total_l1408_140886


namespace NUMINAMATH_CALUDE_cookies_remaining_l1408_140808

/-- Given the initial number of cookies and the number of cookies taken by each person,
    prove that the remaining number of cookies is 6. -/
theorem cookies_remaining (initial : ℕ) (eaten : ℕ) (brother : ℕ) (friend1 : ℕ) (friend2 : ℕ) (friend3 : ℕ)
    (h1 : initial = 22)
    (h2 : eaten = 2)
    (h3 : brother = 1)
    (h4 : friend1 = 3)
    (h5 : friend2 = 5)
    (h6 : friend3 = 5) :
    initial - eaten - brother - friend1 - friend2 - friend3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookies_remaining_l1408_140808


namespace NUMINAMATH_CALUDE_max_xy_value_l1408_140852

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + 3 * y = 12) :
  x * y ≤ 3 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 4 * x + 3 * y = 12 ∧ x * y = 3 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l1408_140852


namespace NUMINAMATH_CALUDE_triangle_side_range_l1408_140847

theorem triangle_side_range (A B C : ℝ) (AB BC : ℝ) :
  AB = Real.sqrt 3 →
  C = π / 3 →
  (∃ (A₁ A₂ : ℝ), A₁ ≠ A₂ ∧ 
    Real.sin A₁ = BC / 2 ∧ 
    Real.sin A₂ = BC / 2 ∧ 
    A₁ ∈ Set.Ioo (π / 3) (2 * π / 3) ∧ 
    A₂ ∈ Set.Ioo (π / 3) (2 * π / 3)) →
  BC > Real.sqrt 3 ∧ BC < 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_range_l1408_140847


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1408_140834

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    a^2 + b^2 = c^2 →  -- right-angled triangle condition
    a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
    c = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1408_140834


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l1408_140811

/-- The quadratic equation mx^2 + x - m^2 + 1 = 0 has -1 as a root if and only if m = 1 -/
theorem quadratic_root_implies_m_value (m : ℝ) : 
  (m * (-1)^2 + (-1) - m^2 + 1 = 0) ↔ (m = 1) := by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l1408_140811


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1408_140824

theorem sum_of_fractions : (3 : ℚ) / 8 + (7 : ℚ) / 9 = (83 : ℚ) / 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1408_140824


namespace NUMINAMATH_CALUDE_M_intersect_N_l1408_140889

/-- The set M defined by the condition √x < 4 -/
def M : Set ℝ := {x | Real.sqrt x < 4}

/-- The set N defined by the condition 3x ≥ 1 -/
def N : Set ℝ := {x | 3 * x ≥ 1}

/-- The intersection of sets M and N -/
theorem M_intersect_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1408_140889


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_46_l1408_140850

/-- The coefficient of x^2 in the expansion of (2x^3 + 4x^2 - 3x + 5)(3x^2 - 9x + 1) -/
def coefficient_x_squared : ℤ :=
  let p1 := [2, 4, -3, 5]  -- Coefficients of 2x^3 + 4x^2 - 3x + 5
  let p2 := [3, -9, 1]     -- Coefficients of 3x^2 - 9x + 1
  46

/-- Proof that the coefficient of x^2 in the expansion of (2x^3 + 4x^2 - 3x + 5)(3x^2 - 9x + 1) is 46 -/
theorem coefficient_x_squared_is_46 : coefficient_x_squared = 46 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_46_l1408_140850


namespace NUMINAMATH_CALUDE_alternate_arrangement_count_l1408_140872

def number_of_men : ℕ := 2
def number_of_women : ℕ := 2

theorem alternate_arrangement_count :
  (number_of_men = 2 ∧ number_of_women = 2) →
  (∃ (count : ℕ), count = 8 ∧
    count = (number_of_men * number_of_women * 1 * 1) +
            (number_of_women * number_of_men * 1 * 1)) :=
by sorry

end NUMINAMATH_CALUDE_alternate_arrangement_count_l1408_140872


namespace NUMINAMATH_CALUDE_kishore_savings_theorem_l1408_140862

/-- Represents Mr. Kishore's financial situation --/
structure KishoreFinances where
  rent : ℕ
  milk : ℕ
  groceries : ℕ
  education : ℕ
  petrol : ℕ
  miscellaneous : ℕ
  savings_percentage : ℚ

/-- Calculates the total expenses --/
def total_expenses (k : KishoreFinances) : ℕ :=
  k.rent + k.milk + k.groceries + k.education + k.petrol + k.miscellaneous

/-- Calculates the monthly salary --/
def monthly_salary (k : KishoreFinances) : ℚ :=
  (total_expenses k : ℚ) / (1 - k.savings_percentage)

/-- Calculates the savings amount --/
def savings_amount (k : KishoreFinances) : ℚ :=
  k.savings_percentage * monthly_salary k

/-- Theorem: Mr. Kishore's savings are approximately 2683.33 Rs. --/
theorem kishore_savings_theorem (k : KishoreFinances) 
  (h1 : k.rent = 5000)
  (h2 : k.milk = 1500)
  (h3 : k.groceries = 4500)
  (h4 : k.education = 2500)
  (h5 : k.petrol = 2000)
  (h6 : k.miscellaneous = 5650)
  (h7 : k.savings_percentage = 1/10) :
  ∃ (ε : ℚ), abs (savings_amount k - 2683.33) < ε ∧ ε < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_kishore_savings_theorem_l1408_140862


namespace NUMINAMATH_CALUDE_expression_evaluation_l1408_140860

theorem expression_evaluation : (24 * 2 - 6) / ((6 - 2) * 2) = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1408_140860


namespace NUMINAMATH_CALUDE_inscribed_square_distances_l1408_140805

/-- A circle with radius 5 containing an inscribed square -/
structure InscribedSquareCircle where
  radius : ℝ
  radius_eq : radius = 5

/-- A point on the circumference of the circle -/
structure CircumferencePoint (c : InscribedSquareCircle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.radius)^2 + point.2^2 = c.radius^2

/-- Vertices of the inscribed square -/
def square_vertices (c : InscribedSquareCircle) : Fin 4 → ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the distances from a point on the circumference to the square vertices -/
theorem inscribed_square_distances
  (c : InscribedSquareCircle)
  (m : CircumferencePoint c)
  (h : distance m.point (square_vertices c 0) = 6) :
  ∃ (perm : Fin 3 → Fin 3),
    distance m.point (square_vertices c 1) = 8 ∧
    distance m.point (square_vertices c 2) = Real.sqrt 2 ∧
    distance m.point (square_vertices c 3) = 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_distances_l1408_140805


namespace NUMINAMATH_CALUDE_triangle_properties_l1408_140887

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a + b = 10 →
  c = 2 * Real.sqrt 7 →
  c * Real.sin B = Real.sqrt 3 * b * Real.cos C →
  C = π / 3 ∧ 
  (1/2) * a * b * Real.sin C = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1408_140887


namespace NUMINAMATH_CALUDE_weekly_commute_cost_l1408_140885

-- Define the parameters
def workDays : ℕ := 5
def carToll : ℚ := 12.5
def motorcycleToll : ℚ := 7
def milesPerGallon : ℚ := 35
def commuteDistance : ℚ := 14
def gasPrice : ℚ := 3.75
def carTrips : ℕ := 3
def motorcycleTrips : ℕ := 2

-- Define the theorem
theorem weekly_commute_cost :
  let carTollCost := carToll * carTrips
  let motorcycleTollCost := motorcycleToll * motorcycleTrips
  let totalDistance := commuteDistance * 2 * workDays
  let totalGasUsed := totalDistance / milesPerGallon
  let gasCost := totalGasUsed * gasPrice
  let totalCost := carTollCost + motorcycleTollCost + gasCost
  totalCost = 59 := by sorry

end NUMINAMATH_CALUDE_weekly_commute_cost_l1408_140885


namespace NUMINAMATH_CALUDE_discriminant_less_than_negative_one_l1408_140818

/-- A quadratic function that doesn't intersect with y = x and y = -x -/
structure NonIntersectingQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : ∀ x : ℝ, a * x^2 + b * x + c ≠ x
  h2 : ∀ x : ℝ, a * x^2 + b * x + c ≠ -x

/-- The discriminant of a quadratic function is less than -1 -/
theorem discriminant_less_than_negative_one (f : NonIntersectingQuadratic) :
  |f.b^2 - 4 * f.a * f.c| > 1 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_less_than_negative_one_l1408_140818


namespace NUMINAMATH_CALUDE_watermelons_with_seeds_l1408_140873

theorem watermelons_with_seeds (ripe : ℕ) (unripe : ℕ) (seedless : ℕ) : 
  ripe = 11 → unripe = 13 → seedless = 15 → 
  ripe + unripe - seedless = 9 := by
  sorry

end NUMINAMATH_CALUDE_watermelons_with_seeds_l1408_140873


namespace NUMINAMATH_CALUDE_cube_roots_sum_power_nine_l1408_140849

/-- Given complex numbers x and y as defined, prove that x⁹ + y⁹ ≠ -1 --/
theorem cube_roots_sum_power_nine (x y : ℂ) : 
  x = (-1 + Complex.I * Real.sqrt 3) / 2 →
  y = (-1 - Complex.I * Real.sqrt 3) / 2 →
  x^9 + y^9 ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_cube_roots_sum_power_nine_l1408_140849


namespace NUMINAMATH_CALUDE_parabola_shift_through_origin_l1408_140876

-- Define the parabola function
def parabola (x : ℝ) : ℝ := (x + 3)^2 - 1

-- Define the shifted parabola function
def shifted_parabola (h : ℝ) (x : ℝ) : ℝ := parabola (x - h)

-- Theorem statement
theorem parabola_shift_through_origin :
  ∀ h : ℝ, shifted_parabola h 0 = 0 ↔ h = 2 ∨ h = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_through_origin_l1408_140876


namespace NUMINAMATH_CALUDE_inverse_cube_root_relation_l1408_140880

/-- Given that y varies inversely as the cube root of x, prove that when x = 8 and y = 2,
    then x = 1/8 when y = 8 -/
theorem inverse_cube_root_relation (x y : ℝ) (k : ℝ) : 
  (∀ x y, y * (x ^ (1/3 : ℝ)) = k) →  -- y varies inversely as the cube root of x
  (2 * (8 ^ (1/3 : ℝ)) = k) →         -- when x = 8, y = 2
  (8 * (x ^ (1/3 : ℝ)) = k) →         -- when y = 8
  x = 1/8 := by
sorry

end NUMINAMATH_CALUDE_inverse_cube_root_relation_l1408_140880


namespace NUMINAMATH_CALUDE_greatest_abdba_divisible_by_13_l1408_140888

def is_valid_abdba (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (a b d : ℕ),
    a < 10 ∧ b < 10 ∧ d < 10 ∧
    a ≠ b ∧ b ≠ d ∧ a ≠ d ∧
    n = a * 10000 + b * 1000 + d * 100 + b * 10 + a

theorem greatest_abdba_divisible_by_13 :
  ∀ n : ℕ, is_valid_abdba n → n % 13 = 0 → n ≤ 96769 :=
by sorry

end NUMINAMATH_CALUDE_greatest_abdba_divisible_by_13_l1408_140888


namespace NUMINAMATH_CALUDE_sample_xy_value_l1408_140871

theorem sample_xy_value (x y : ℝ) : 
  (x + 1 + y + 5) / 4 = 2 →
  ((x - 2)^2 + (1 - 2)^2 + (y - 2)^2 + (5 - 2)^2) / 4 = 5 →
  x * y = -4 := by
sorry

end NUMINAMATH_CALUDE_sample_xy_value_l1408_140871


namespace NUMINAMATH_CALUDE_prob_all_same_color_is_correct_l1408_140863

def yellow_marbles : ℕ := 3
def green_marbles : ℕ := 7
def purple_marbles : ℕ := 5
def total_marbles : ℕ := yellow_marbles + green_marbles + purple_marbles
def drawn_marbles : ℕ := 4

def prob_all_same_color : ℚ :=
  (green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)) +
  (purple_marbles * (purple_marbles - 1) * (purple_marbles - 2) * (purple_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem prob_all_same_color_is_correct : prob_all_same_color = 532 / 4095 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_same_color_is_correct_l1408_140863


namespace NUMINAMATH_CALUDE_bookseller_sales_l1408_140815

/-- Bookseller's monthly sales problem -/
theorem bookseller_sales 
  (b1 b2 b3 b4 : ℕ) 
  (h1 : b1 + b2 + b3 = 45)
  (h2 : b4 = (3 * (b1 + b2)) / 4)
  (h3 : (b1 + b2 + b3 + b4) / 4 = 18) :
  b3 = 9 ∧ b1 + b2 = 36 ∧ b4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_bookseller_sales_l1408_140815


namespace NUMINAMATH_CALUDE_intersection_A_B_l1408_140819

def A : Set ℝ := {x | ∃ y, y = Real.log x}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1408_140819


namespace NUMINAMATH_CALUDE_potato_sales_total_weight_l1408_140899

theorem potato_sales_total_weight :
  let morning_sales : ℕ := 29
  let afternoon_sales : ℕ := 17
  let bag_weight : ℕ := 7
  let total_bags : ℕ := morning_sales + afternoon_sales
  let total_weight : ℕ := total_bags * bag_weight
  total_weight = 322 := by sorry

end NUMINAMATH_CALUDE_potato_sales_total_weight_l1408_140899


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l1408_140840

/-- Given two quadratic equations with coefficients p, q and p', q', 
    this theorem proves that the product of differences of their roots 
    can be expressed in terms of these coefficients. -/
theorem quadratic_roots_product (p q p' q' : ℝ) 
  (hα : ∃ α : ℝ, α^2 + p*α + q = 0)
  (hβ : ∃ β : ℝ, β^2 + p*β + q = 0)
  (hα' : ∃ α' : ℝ, α'^2 + p'*α' + q' = 0)
  (hβ' : ∃ β' : ℝ, β'^2 + p'*β' + q' = 0)
  (h_distinct : ∀ (α β : ℝ), α^2 + p*α + q = 0 → β^2 + p*β + q = 0 → α ≠ β) 
  (h_distinct' : ∀ (α' β' : ℝ), α'^2 + p'*α' + q' = 0 → β'^2 + p'*β' + q' = 0 → α' ≠ β') :
  ∃ (α β α' β' : ℝ), 
    (α^2 + p*α + q = 0) ∧ 
    (β^2 + p*β + q = 0) ∧ 
    (α'^2 + p'*α' + q' = 0) ∧ 
    (β'^2 + p'*β' + q' = 0) ∧
    ((α - α') * (α - β') * (β - α') * (β - β') = (q - q')^2 + (p - p') * (q'*p - p'*q)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_product_l1408_140840


namespace NUMINAMATH_CALUDE_cannon_hit_probability_l1408_140836

theorem cannon_hit_probability (P1 P2 P3 : ℝ) 
  (h1 : P2 = 0.2)
  (h2 : P3 = 0.3)
  (h3 : (1 - P1) * (1 - P2) * (1 - P3) = 0.28) :
  P1 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_cannon_hit_probability_l1408_140836


namespace NUMINAMATH_CALUDE_valid_spy_placement_exists_l1408_140810

/-- Represents a position on the board -/
structure Position where
  x : Fin 6
  y : Fin 6

/-- Represents the vision of a spy -/
inductive Vision where
  | ahead : Position → Vision
  | right : Position → Vision
  | left : Position → Vision

/-- Checks if a spy at the given position can see the target position -/
def canSee (spyPos : Position) (targetPos : Position) : Prop :=
  ∃ (v : Vision),
    match v with
    | Vision.ahead p => p.x = spyPos.x ∧ p.y = spyPos.y + 1 ∨ p.y = spyPos.y + 2
    | Vision.right p => p.x = spyPos.x + 1 ∧ p.y = spyPos.y
    | Vision.left p => p.x = spyPos.x - 1 ∧ p.y = spyPos.y

/-- A valid spy placement is a list of 18 positions where no spy can see another -/
def ValidSpyPlacement (placement : List Position) : Prop :=
  placement.length = 18 ∧
  ∀ (spy1 spy2 : Position),
    spy1 ∈ placement → spy2 ∈ placement → spy1 ≠ spy2 →
    ¬(canSee spy1 spy2 ∨ canSee spy2 spy1)

/-- Theorem stating that a valid spy placement exists -/
theorem valid_spy_placement_exists : ∃ (placement : List Position), ValidSpyPlacement placement :=
  sorry

end NUMINAMATH_CALUDE_valid_spy_placement_exists_l1408_140810


namespace NUMINAMATH_CALUDE_john_bench_press_sets_l1408_140897

/-- The number of sets John does in his workout -/
def number_of_sets (weight_per_rep : ℕ) (reps_per_set : ℕ) (total_weight : ℕ) : ℕ :=
  total_weight / (weight_per_rep * reps_per_set)

/-- Theorem: John does 3 sets of bench presses -/
theorem john_bench_press_sets :
  number_of_sets 15 10 450 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_bench_press_sets_l1408_140897


namespace NUMINAMATH_CALUDE_inequality_proofs_l1408_140846

def M : Set ℝ := {x | x ≥ 2}

theorem inequality_proofs :
  (∀ (a b c d : ℝ), a ∈ M → b ∈ M → c ∈ M → d ∈ M →
    Real.sqrt (a * b) + Real.sqrt (c * d) ≥ Real.sqrt (a + b) + Real.sqrt (c + d)) ∧
  (∀ (a b c d : ℝ), a ∈ M → b ∈ M → c ∈ M → d ∈ M →
    Real.sqrt (a * b) + Real.sqrt (c * d) ≥ Real.sqrt (a + c) + Real.sqrt (b + d)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l1408_140846


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l1408_140831

theorem smallest_absolute_value : ∀ x : ℝ, |0| ≤ |x| := by
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l1408_140831


namespace NUMINAMATH_CALUDE_treadmill_theorem_l1408_140858

def treadmill_problem (days : Nat) (distance_per_day : Real) 
  (speeds : List Real) (constant_speed : Real) : Prop :=
  days = 4 ∧
  distance_per_day = 3 ∧
  speeds = [6, 4, 3, 5] ∧
  constant_speed = 5 ∧
  let actual_time := (List.map (fun s => distance_per_day / s) speeds).sum
  let constant_time := (days * distance_per_day) / constant_speed
  (actual_time - constant_time) * 60 = 27

theorem treadmill_theorem : 
  ∃ (days : Nat) (distance_per_day : Real) (speeds : List Real) (constant_speed : Real),
  treadmill_problem days distance_per_day speeds constant_speed :=
sorry

end NUMINAMATH_CALUDE_treadmill_theorem_l1408_140858


namespace NUMINAMATH_CALUDE_triangle_side_value_l1408_140861

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions of the problem
def satisfiesConditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 13 ∧
  t.a + t.c = 4 ∧
  (Real.cos t.B) / (Real.cos t.C) = -t.b / (2 * t.a + t.c)

-- State the theorem
theorem triangle_side_value (t : Triangle) (h : satisfiesConditions t) :
  t.a = 1 ∨ t.a = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_value_l1408_140861


namespace NUMINAMATH_CALUDE_fraction_less_than_mode_l1408_140875

def data_list : List ℕ := [3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 10, 11, 15, 21, 23, 26, 27]

def mode (l : List ℕ) : ℕ := 
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def count_less_than_mode (l : List ℕ) : ℕ :=
  l.filter (· < mode l) |>.length

theorem fraction_less_than_mode :
  (count_less_than_mode data_list : ℚ) / data_list.length = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_mode_l1408_140875


namespace NUMINAMATH_CALUDE_hexagon_angle_d_measure_l1408_140817

/-- Theorem: Measure of angle D in a hexagon with specific angle conditions -/
theorem hexagon_angle_d_measure (A B C D E F : ℝ) : 
  A = 90 → 
  B = 120 → 
  C = D → 
  E = 2 * C + 20 → 
  F = 60 → 
  A + B + C + D + E + F = 720 → 
  D = 107.5 := by sorry

end NUMINAMATH_CALUDE_hexagon_angle_d_measure_l1408_140817


namespace NUMINAMATH_CALUDE_profit_share_difference_theorem_l1408_140884

/-- Represents an investor's contribution to the business --/
structure Investor where
  investment : ℕ
  duration : ℕ

/-- Calculates the difference in profit shares between two investors --/
def profit_share_difference (suresh rohan sudhir : Investor) (total_profit : ℕ) : ℕ :=
  let total_investment_months := suresh.investment * suresh.duration + 
                                 rohan.investment * rohan.duration + 
                                 sudhir.investment * sudhir.duration
  let rohan_share := (rohan.investment * rohan.duration * total_profit) / total_investment_months
  let sudhir_share := (sudhir.investment * sudhir.duration * total_profit) / total_investment_months
  rohan_share - sudhir_share

/-- Theorem stating the difference in profit shares --/
theorem profit_share_difference_theorem (suresh rohan sudhir : Investor) (total_profit : ℕ) :
  suresh.investment = 18000 ∧ suresh.duration = 12 ∧
  rohan.investment = 12000 ∧ rohan.duration = 9 ∧
  sudhir.investment = 9000 ∧ sudhir.duration = 8 ∧
  total_profit = 3795 →
  profit_share_difference suresh rohan sudhir total_profit = 345 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_theorem_l1408_140884


namespace NUMINAMATH_CALUDE_john_house_planks_l1408_140855

theorem john_house_planks :
  ∀ (total_nails nails_per_plank additional_nails : ℕ),
    total_nails = 11 →
    nails_per_plank = 3 →
    additional_nails = 8 →
    ∃ (num_planks : ℕ),
      num_planks * nails_per_plank + additional_nails = total_nails ∧
      num_planks = 1 := by
sorry

end NUMINAMATH_CALUDE_john_house_planks_l1408_140855


namespace NUMINAMATH_CALUDE_probability_non_defective_pencils_l1408_140856

/-- The probability of selecting 3 non-defective pencils from a box of 10 pencils with 2 defective pencils -/
theorem probability_non_defective_pencils :
  let total_pencils : ℕ := 10
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations := Nat.choose total_pencils selected_pencils
  let non_defective_combinations := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_defective_pencils_l1408_140856


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1408_140835

theorem trigonometric_identities :
  (((Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) ^ 2) = 1/2) ∧
  ((Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3)) = -1) ∧
  ((Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - Real.cos (66 * π / 180) * Real.cos (54 * π / 180)) = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1408_140835


namespace NUMINAMATH_CALUDE_fraction_simplification_l1408_140866

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) :
  (1/2 * a) / (1/2 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1408_140866


namespace NUMINAMATH_CALUDE_smallest_special_number_l1408_140801

def unit_digit (n : ℕ) : ℕ := n % 10

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem smallest_special_number : 
  ∀ n : ℕ, 
    (unit_digit n = 5 ∧ 
     is_perfect_square n ∧ 
     (∃ k : ℕ, k * k = n ∧ digit_sum k = 9)) → 
    n ≥ 2025 :=
sorry

end NUMINAMATH_CALUDE_smallest_special_number_l1408_140801


namespace NUMINAMATH_CALUDE_cut_rectangle_properties_l1408_140822

/-- Represents a rectangle cut into four pieces by two equal diagonals intersecting at right angles -/
structure CutRectangle where
  width : ℝ
  height : ℝ
  diag_intersect_center : Bool
  diag_right_angle : Bool
  diag_equal_length : Bool

/-- Theorem about properties of a specific cut rectangle -/
theorem cut_rectangle_properties (rect : CutRectangle) 
  (h_width : rect.width = 20)
  (h_height : rect.height = 30)
  (h_center : rect.diag_intersect_center = true)
  (h_right : rect.diag_right_angle = true)
  (h_equal : rect.diag_equal_length = true) :
  ∃ (square_side triangle_area pentagon_area hole_area : ℝ),
    square_side = 20 ∧
    triangle_area = 100 ∧
    pentagon_area = 200 ∧
    hole_area = 200 := by
  sorry


end NUMINAMATH_CALUDE_cut_rectangle_properties_l1408_140822


namespace NUMINAMATH_CALUDE_incorrect_transformation_l1408_140806

theorem incorrect_transformation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / 2 = b / 3) :
  ¬(2 * a = 3 * b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_transformation_l1408_140806


namespace NUMINAMATH_CALUDE_min_value_quadratic_max_value_quadratic_l1408_140804

-- Question 1
theorem min_value_quadratic (m : ℝ) : m^2 - 6*m + 10 ≥ 1 := by sorry

-- Question 2
theorem max_value_quadratic (x : ℝ) : -2*x^2 - 4*x + 3 ≤ 5 := by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_max_value_quadratic_l1408_140804


namespace NUMINAMATH_CALUDE_infinitely_many_n_with_bounded_prime_divisors_l1408_140827

theorem infinitely_many_n_with_bounded_prime_divisors :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ 
  (∀ n ∈ S, ∀ p : ℕ, Prime p → p ∣ (n^2 + n + 1) → p ≤ Real.sqrt n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_n_with_bounded_prime_divisors_l1408_140827


namespace NUMINAMATH_CALUDE_expedition_investigation_days_l1408_140837

theorem expedition_investigation_days 
  (upstream_speed : ℕ) 
  (downstream_speed : ℕ) 
  (total_days : ℕ) 
  (final_day_distance : ℕ) 
  (h1 : upstream_speed = 17)
  (h2 : downstream_speed = 25)
  (h3 : total_days = 60)
  (h4 : final_day_distance = 24) :
  ∃ (upstream_days downstream_days investigation_days : ℕ),
    upstream_days + downstream_days + investigation_days = total_days ∧
    upstream_speed * upstream_days - downstream_speed * downstream_days = final_day_distance - downstream_speed ∧
    investigation_days = 23 := by
  sorry

#check expedition_investigation_days

end NUMINAMATH_CALUDE_expedition_investigation_days_l1408_140837


namespace NUMINAMATH_CALUDE_estimate_excellent_scores_result_l1408_140854

/-- Estimates the number of excellent scores in a population based on a sample. -/
def estimate_excellent_scores (total_population : ℕ) (sample_size : ℕ) (excellent_in_sample : ℕ) : ℕ :=
  (total_population * excellent_in_sample) / sample_size

/-- Theorem stating that the estimated number of excellent scores is 152 given the problem conditions. -/
theorem estimate_excellent_scores_result :
  estimate_excellent_scores 380 50 20 = 152 := by
  sorry

end NUMINAMATH_CALUDE_estimate_excellent_scores_result_l1408_140854


namespace NUMINAMATH_CALUDE_chinese_riddle_championship_arrangement_l1408_140823

theorem chinese_riddle_championship_arrangement (n : ℕ) (students : ℕ) (teacher : ℕ) (parents : ℕ) :
  n = 6 →
  students = 3 →
  teacher = 1 →
  parents = 2 →
  (students.factorial * 2 * (n - students - 1).factorial) = 72 :=
by sorry

end NUMINAMATH_CALUDE_chinese_riddle_championship_arrangement_l1408_140823


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1408_140844

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x > 0, f x > 0}

/-- The property f(x f(y)) = y f(x) for all positive real x, y -/
def HasFunctionalProperty (f : PositiveRealFunction) :=
  ∀ x y, x > 0 → y > 0 → f.val (x * f.val y) = y * f.val x

/-- The property that f(x) → 0 as x → +∞ -/
def TendsToZeroAtInfinity (f : PositiveRealFunction) :=
  ∀ ε > 0, ∃ M, ∀ x > M, f.val x < ε

/-- The main theorem -/
theorem unique_function_satisfying_conditions
  (f : PositiveRealFunction)
  (h1 : HasFunctionalProperty f)
  (h2 : TendsToZeroAtInfinity f) :
  ∀ x > 0, f.val x = 1 / x :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1408_140844


namespace NUMINAMATH_CALUDE_fifteen_members_without_A_l1408_140893

/-- Represents the number of club members who did not receive an A in either activity. -/
def members_without_A (total_members art_A science_A both_A : ℕ) : ℕ :=
  total_members - (art_A + science_A - both_A)

/-- Theorem stating that 15 club members did not receive an A in either activity. -/
theorem fifteen_members_without_A :
  members_without_A 50 20 30 15 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_members_without_A_l1408_140893


namespace NUMINAMATH_CALUDE_circle_circumference_l1408_140898

/-- Given a circle with area 1800 cm² and ratio of area to circumference 15, 
    prove that its circumference is 120 cm. -/
theorem circle_circumference (A : ℝ) (r : ℝ) :
  A = 1800 →
  A / (2 * Real.pi * r) = 15 →
  2 * Real.pi * r = 120 :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_l1408_140898


namespace NUMINAMATH_CALUDE_triangle_max_area_l1408_140896

theorem triangle_max_area (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : Real.tan A * Real.tan B = 3/4) : 
  let a : ℝ := 4
  let b : ℝ := a * Real.sin B / Real.sin A
  let c : ℝ := a * Real.sin C / Real.sin A
  ∀ (S : ℝ), S = 1/2 * a * b * Real.sin C → S ≤ 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1408_140896


namespace NUMINAMATH_CALUDE_remaining_investment_rate_l1408_140869

def total_investment : ℝ := 12000
def investment_at_7_percent : ℝ := 5500
def total_interest : ℝ := 970

def remaining_investment : ℝ := total_investment - investment_at_7_percent
def interest_from_7_percent : ℝ := investment_at_7_percent * 0.07
def interest_from_remaining : ℝ := total_interest - interest_from_7_percent

theorem remaining_investment_rate : 
  (interest_from_remaining / remaining_investment) * 100 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remaining_investment_rate_l1408_140869


namespace NUMINAMATH_CALUDE_other_side_heads_probability_l1408_140830

/-- Represents the three types of coins -/
inductive Coin
  | Normal
  | DoubleHeads
  | DoubleTails

/-- Represents the possible sides of a coin -/
inductive Side
  | Heads
  | Tails

/-- The probability of selecting each coin -/
def coinSelectionProbability : ℚ := 1/3

/-- The probability of getting heads for each coin type -/
def probabilityOfHeads (c : Coin) : ℚ :=
  match c with
  | Coin.Normal => 1/2
  | Coin.DoubleHeads => 1
  | Coin.DoubleTails => 0

/-- The probability that the other side is heads given that heads was observed -/
def probabilityOtherSideHeads : ℚ := 2/3

theorem other_side_heads_probability :
  probabilityOtherSideHeads = 2/3 := by sorry


end NUMINAMATH_CALUDE_other_side_heads_probability_l1408_140830


namespace NUMINAMATH_CALUDE_quiz_result_proof_l1408_140881

theorem quiz_result_proof (total : ℕ) (correct_A : ℕ) (correct_B : ℕ) (correct_C : ℕ) 
  (all_wrong : ℕ) (all_correct : ℕ) 
  (h_total : total = 40)
  (h_A : correct_A = 10)
  (h_B : correct_B = 13)
  (h_C : correct_C = 15)
  (h_wrong : all_wrong = 15)
  (h_correct : all_correct = 1) :
  ∃ (two_correct : ℕ), two_correct = 13 ∧ 
  two_correct = total - all_wrong - all_correct - 
    (correct_A + correct_B + correct_C - 2 * all_correct - two_correct) := by
  sorry

end NUMINAMATH_CALUDE_quiz_result_proof_l1408_140881


namespace NUMINAMATH_CALUDE_subset_collection_m_eq_seven_l1408_140867

/-- A structure representing a collection of 3-element subsets of {1, ..., n} -/
structure SubsetCollection (n : ℕ) where
  m : ℕ
  subsets : Fin m → Finset (Fin n)
  m_gt_one : m > 1
  three_elements : ∀ i, (subsets i).card = 3
  unique_pairs : ∀ {x y : Fin n}, x ≠ y → ∃! i, {x, y} ⊆ subsets i
  one_common : ∀ {i j : Fin m}, i ≠ j → ∃! x, x ∈ subsets i ∩ subsets j

/-- The main theorem stating that for any valid SubsetCollection, m = 7 -/
theorem subset_collection_m_eq_seven {n : ℕ} (sc : SubsetCollection n) : sc.m = 7 :=
sorry

end NUMINAMATH_CALUDE_subset_collection_m_eq_seven_l1408_140867


namespace NUMINAMATH_CALUDE_chips_per_console_is_five_l1408_140853

/-- The number of computer chips created per day -/
def chips_per_day : ℕ := 467

/-- The number of video game consoles created per day -/
def consoles_per_day : ℕ := 93

/-- The number of computer chips needed per console -/
def chips_per_console : ℕ := chips_per_day / consoles_per_day

theorem chips_per_console_is_five : chips_per_console = 5 := by
  sorry

end NUMINAMATH_CALUDE_chips_per_console_is_five_l1408_140853


namespace NUMINAMATH_CALUDE_jeans_tshirt_ratio_l1408_140841

/-- Represents the prices of clothing items -/
structure ClothingPrices where
  socks : ℕ
  tshirt : ℕ
  jeans : ℕ

/-- Conditions for the clothing prices -/
def validPrices (p : ClothingPrices) : Prop :=
  p.socks = 5 ∧
  p.tshirt = p.socks + 10 ∧
  p.jeans = 30 ∧
  ∃ k : ℕ, p.jeans = k * p.tshirt

/-- The theorem stating the ratio of jeans to t-shirt prices -/
theorem jeans_tshirt_ratio (p : ClothingPrices) (h : validPrices p) :
  p.jeans / p.tshirt = 2 := by
  sorry

#check jeans_tshirt_ratio

end NUMINAMATH_CALUDE_jeans_tshirt_ratio_l1408_140841


namespace NUMINAMATH_CALUDE_book_club_and_env_painting_participants_l1408_140859

/-- Represents the number of participants in various activities and their intersections --/
structure ActivityParticipants where
  total : ℕ
  bookClub : ℕ
  funSports : ℕ
  envPainting : ℕ
  bookClubAndFunSports : ℕ
  funSportsAndEnvPainting : ℕ

/-- Theorem stating the number of participants in both Book Club and Environmental Theme Painting --/
theorem book_club_and_env_painting_participants (ap : ActivityParticipants) 
  (h1 : ap.total = 120)
  (h2 : ap.bookClub = 80)
  (h3 : ap.funSports = 50)
  (h4 : ap.envPainting = 40)
  (h5 : ap.bookClubAndFunSports = 20)
  (h6 : ap.funSportsAndEnvPainting = 10)
  (h7 : ap.bookClub + ap.funSports + ap.envPainting - ap.total - ap.bookClubAndFunSports - ap.funSportsAndEnvPainting = 20) :
  ap.bookClub + ap.funSports + ap.envPainting - ap.total - ap.bookClubAndFunSports - ap.funSportsAndEnvPainting = 20 := by
  sorry

#check book_club_and_env_painting_participants

end NUMINAMATH_CALUDE_book_club_and_env_painting_participants_l1408_140859
