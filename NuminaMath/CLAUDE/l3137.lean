import Mathlib

namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l3137_313787

theorem complex_arithmetic_equation : 
  -1^4 + (4 - (3/8 + 1/6 - 3/4) * 24) / 5 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l3137_313787


namespace NUMINAMATH_CALUDE_max_value_inequality_l3137_313716

theorem max_value_inequality (x : ℝ) : 
  (∀ y, y > x → (6 + 5*y + y^2) * Real.sqrt (2*y^2 - y^3 - y) > 0) → 
  ((6 + 5*x + x^2) * Real.sqrt (2*x^2 - x^3 - x) ≤ 0) → 
  x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3137_313716


namespace NUMINAMATH_CALUDE_decrypt_message_l3137_313724

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The encrypted message in base-7 --/
def encryptedMessage : List Nat := [4, 3, 5, 2]

/-- The decrypted message in base-10 --/
def decryptedMessage : Nat := 956

theorem decrypt_message :
  base7ToBase10 encryptedMessage = decryptedMessage := by
  sorry

end NUMINAMATH_CALUDE_decrypt_message_l3137_313724


namespace NUMINAMATH_CALUDE_special_power_function_unique_m_l3137_313702

/-- A power function with exponent (m^2 - 2m - 3) that has no intersection with axes and is symmetric about the origin -/
def special_power_function (m : ℕ+) : ℝ → ℝ := fun x ↦ x ^ (m.val ^ 2 - 2 * m.val - 3)

/-- The function has no intersection with x-axis and y-axis -/
def no_axis_intersection (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≠ 0) ∧ (f 0 ≠ 0)

/-- The function is symmetric about the origin -/
def origin_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Main theorem: If the special power function satisfies the conditions, then m = 2 -/
theorem special_power_function_unique_m (m : ℕ+) :
  no_axis_intersection (special_power_function m) ∧
  origin_symmetry (special_power_function m) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_power_function_unique_m_l3137_313702


namespace NUMINAMATH_CALUDE_max_x_minus_y_l3137_313748

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), w = a - b ∧ a^2 + b^2 - 4*a - 2*b - 4 = 0) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l3137_313748


namespace NUMINAMATH_CALUDE_oven_temperature_l3137_313746

theorem oven_temperature (required_temp increase_needed : ℕ) 
  (h1 : required_temp = 546)
  (h2 : increase_needed = 396) :
  required_temp - increase_needed = 150 := by
sorry

end NUMINAMATH_CALUDE_oven_temperature_l3137_313746


namespace NUMINAMATH_CALUDE_nested_sqrt_24_l3137_313790

/-- The solution to the equation x = √(24 + x), where x is non-negative -/
theorem nested_sqrt_24 : 
  ∃ x : ℝ, x ≥ 0 ∧ x = Real.sqrt (24 + x) → x = 6 := by sorry

end NUMINAMATH_CALUDE_nested_sqrt_24_l3137_313790


namespace NUMINAMATH_CALUDE_edward_money_theorem_l3137_313788

def edward_money_problem (initial_money spent1 spent2 : ℕ) : Prop :=
  let total_spent := spent1 + spent2
  let remaining_money := initial_money - total_spent
  remaining_money = 17

theorem edward_money_theorem :
  edward_money_problem 34 9 8 := by
  sorry

end NUMINAMATH_CALUDE_edward_money_theorem_l3137_313788


namespace NUMINAMATH_CALUDE_average_weight_calculation_l3137_313793

theorem average_weight_calculation (num_men num_women : ℕ) (avg_weight_men avg_weight_women : ℚ) :
  num_men = 8 →
  num_women = 6 →
  avg_weight_men = 170 →
  avg_weight_women = 130 →
  let total_weight := num_men * avg_weight_men + num_women * avg_weight_women
  let total_people := num_men + num_women
  abs ((total_weight / total_people) - 153) < 1 := by
sorry

end NUMINAMATH_CALUDE_average_weight_calculation_l3137_313793


namespace NUMINAMATH_CALUDE_greatest_integer_third_side_l3137_313737

theorem greatest_integer_third_side (a b c : ℕ) : 
  a = 7 ∧ b = 10 ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧  -- Triangle inequality
  c ≤ a + b - 1 →                           -- Strict inequality
  (∀ d : ℕ, d > c → ¬(a + b > d ∧ a + d > b ∧ b + d > a)) →
  c = 16 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_third_side_l3137_313737


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3137_313719

theorem sufficient_not_necessary_condition :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → x / y + y / x ≥ 2) ∧
  ¬(∀ x y : ℝ, x / y + y / x ≥ 2 → x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3137_313719


namespace NUMINAMATH_CALUDE_dress_price_difference_l3137_313753

theorem dress_price_difference (original_price : ℝ) : 
  (original_price * 0.85 = 71.4) →
  (original_price - (71.4 * 1.25)) = 5.25 := by
sorry

end NUMINAMATH_CALUDE_dress_price_difference_l3137_313753


namespace NUMINAMATH_CALUDE_proportion_problem_l3137_313754

theorem proportion_problem (y : ℝ) : 
  (0.25 : ℝ) / 0.75 = y / 6 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l3137_313754


namespace NUMINAMATH_CALUDE_x_y_equation_l3137_313701

theorem x_y_equation (x y : ℚ) (hx : x = 2/3) (hy : y = 9/2) : (1/3) * x^4 * y^5 = 121.5 := by
  sorry

end NUMINAMATH_CALUDE_x_y_equation_l3137_313701


namespace NUMINAMATH_CALUDE_maurice_cookout_packages_l3137_313736

/-- The number of packages of ground beef Maurice needs to purchase for his cookout --/
def packages_needed (guests : ℕ) (burger_weight : ℕ) (package_weight : ℕ) : ℕ :=
  let total_people := guests + 1  -- Adding Maurice himself
  let total_weight := total_people * burger_weight
  (total_weight + package_weight - 1) / package_weight  -- Ceiling division

/-- Theorem stating that Maurice needs to purchase 4 packages of ground beef --/
theorem maurice_cookout_packages : packages_needed 9 2 5 = 4 := by
  sorry

#eval packages_needed 9 2 5

end NUMINAMATH_CALUDE_maurice_cookout_packages_l3137_313736


namespace NUMINAMATH_CALUDE_max_y_over_x_l3137_313799

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  x + y ≥ 3 ∧ x - y ≥ -1 ∧ 2*x - y ≤ 3

-- State the theorem
theorem max_y_over_x :
  ∃ (max : ℝ), max = 2 ∧
  ∀ (x y : ℝ), FeasibleRegion x y → y / x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_y_over_x_l3137_313799


namespace NUMINAMATH_CALUDE_equation_solutions_l3137_313708

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 21) / 2 ∧ x₂ = (5 - Real.sqrt 21) / 2 ∧
    x₁^2 - 5*x₁ + 1 = 0 ∧ x₂^2 - 5*x₂ + 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 5 ∧ y₂ = 10/3 ∧
    2*(y₁-5)^2 + y₁*(y₁-5) = 0 ∧ 2*(y₂-5)^2 + y₂*(y₂-5) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l3137_313708


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l3137_313765

theorem sum_of_specific_numbers : 
  217 + 2.017 + 0.217 + 2.0017 = 221.2357 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l3137_313765


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3137_313767

theorem quadratic_solution_sum (c d : ℝ) : 
  (∀ x, x^2 - 6*x + 15 = 27 ↔ x = c ∨ x = d) →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3137_313767


namespace NUMINAMATH_CALUDE_kellys_apples_l3137_313770

/-- The total number of apples Kelly has after picking more apples -/
def total_apples (initial : Float) (picked : Float) : Float :=
  initial + picked

/-- Theorem stating that Kelly's total apples is 161.0 -/
theorem kellys_apples :
  let initial := 56.0
  let picked := 105.0
  total_apples initial picked = 161.0 := by
  sorry

end NUMINAMATH_CALUDE_kellys_apples_l3137_313770


namespace NUMINAMATH_CALUDE_positive_numbers_not_all_equal_l3137_313785

/-- Given positive numbers a, b, and c that are not all equal -/
theorem positive_numbers_not_all_equal 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_not_all_equal : ¬(a = b ∧ b = c)) : 
  /- 1. (a-b)² + (b-c)² + (c-a)² ≠ 0 -/
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  /- 2. At least one of a > b, a < b, or a = b is true -/
  (a > b ∨ a < b ∨ a = b) ∧ 
  /- 3. It is possible for a ≠ c, b ≠ c, and a ≠ b to all be true simultaneously -/
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_not_all_equal_l3137_313785


namespace NUMINAMATH_CALUDE_amandas_flowers_l3137_313712

theorem amandas_flowers (amanda_flowers : ℕ) (peter_flowers : ℕ) : 
  peter_flowers = 3 * amanda_flowers →
  peter_flowers - 15 = 45 →
  amanda_flowers = 20 := by
sorry

end NUMINAMATH_CALUDE_amandas_flowers_l3137_313712


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l3137_313711

theorem two_numbers_with_given_means (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * b = 5) →
  (2 * a * b / (a + b) = 5/3) →
  ((a = 1 ∧ b = 5) ∨ (a = 5 ∧ b = 1)) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l3137_313711


namespace NUMINAMATH_CALUDE_total_games_won_l3137_313783

/-- The number of games won by the Chicago Bulls -/
def bulls_wins : ℕ := 70

/-- The number of games won by the Miami Heat -/
def heat_wins : ℕ := bulls_wins + 5

/-- The number of games won by the New York Knicks -/
def knicks_wins : ℕ := 2 * heat_wins

/-- The number of games won by the Los Angeles Lakers -/
def lakers_wins : ℕ := (3 * (bulls_wins + knicks_wins)) / 2

/-- The total number of games won by all four teams -/
def total_wins : ℕ := bulls_wins + heat_wins + knicks_wins + lakers_wins

theorem total_games_won : total_wins = 625 := by
  sorry

end NUMINAMATH_CALUDE_total_games_won_l3137_313783


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3137_313771

theorem polynomial_division_remainder : 
  ∃ (q : Polynomial ℝ), 
    x^4 + 2*x^3 - 3*x^2 + 4*x - 5 = (x^2 - 3*x + 2) * q + (24*x - 25) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3137_313771


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3137_313779

theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  s > 0 →
  6 * s^2 = 864 →
  s^3 = 1728 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3137_313779


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l3137_313707

/-- Proves that the cost of fencing per meter for a rectangular plot is 26.5 Rs. -/
theorem fencing_cost_per_meter
  (length : ℝ)
  (breadth : ℝ)
  (total_cost : ℝ)
  (h1 : length = 61)
  (h2 : breadth = 39)
  (h3 : total_cost = 5300)
  : (total_cost / (2 * (length + breadth))) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l3137_313707


namespace NUMINAMATH_CALUDE_shopping_trip_tax_percentage_l3137_313715

/-- Calculate the total tax percentage given spending percentages and tax rates -/
theorem shopping_trip_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.45)
  (h2 : food_percent = 0.45)
  (h3 : other_percent = 0.1)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.05)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.1) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.0325 := by
  sorry

#check shopping_trip_tax_percentage

end NUMINAMATH_CALUDE_shopping_trip_tax_percentage_l3137_313715


namespace NUMINAMATH_CALUDE_sum_is_three_or_seven_l3137_313705

theorem sum_is_three_or_seven (x y z : ℝ) 
  (eq1 : x + y / z = 2)
  (eq2 : y + z / x = 2)
  (eq3 : z + x / y = 2) :
  let S := x + y + z
  S = 3 ∨ S = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_is_three_or_seven_l3137_313705


namespace NUMINAMATH_CALUDE_power_comparisons_l3137_313756

theorem power_comparisons :
  (3^40 > 4^30 ∧ 4^30 > 5^20) ∧
  (16^31 > 8^41 ∧ 8^41 > 4^61) ∧
  (∀ a b : ℝ, a > 1 → b > 1 → a^5 = 2 → b^7 = 3 → a < b) := by
  sorry


end NUMINAMATH_CALUDE_power_comparisons_l3137_313756


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l3137_313773

theorem part_to_whole_ratio (N P : ℝ) 
  (h1 : (1/4) * (1/3) * P = 10)
  (h2 : 0.40 * N = 120) : 
  P/N = 1/2.5 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l3137_313773


namespace NUMINAMATH_CALUDE_max_altitude_product_right_triangle_l3137_313777

/-- Given a fixed side length and area, the product of altitudes is maximum for a right triangle --/
theorem max_altitude_product_right_triangle 
  (l : ℝ) (S : ℝ) (h_pos_l : l > 0) (h_pos_S : S > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    (1/2) * l * b = S ∧
    ∀ (x y : ℝ), x > 0 → y > 0 → (1/2) * l * y = S →
      (2*S/l) * (2*S/(l*x)) * (2*S/(l*y)) ≤ (2*S/l) * (2*S/(l*a)) * (2*S/(l*b)) :=
sorry

end NUMINAMATH_CALUDE_max_altitude_product_right_triangle_l3137_313777


namespace NUMINAMATH_CALUDE_widgets_in_shipping_box_is_300_l3137_313709

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.width * d.length * d.height

/-- Represents the problem setup -/
structure WidgetProblem where
  cartonDimensions : BoxDimensions
  shippingBoxDimensions : BoxDimensions
  widgetsPerCarton : ℕ

/-- Calculates the number of widgets in a shipping box -/
def widgetsInShippingBox (p : WidgetProblem) : ℕ :=
  let cartonsInBox := (boxVolume p.shippingBoxDimensions) / (boxVolume p.cartonDimensions)
  cartonsInBox * p.widgetsPerCarton

/-- The main theorem to prove -/
theorem widgets_in_shipping_box_is_300 (p : WidgetProblem) : 
  p.cartonDimensions = ⟨4, 4, 5⟩ ∧ 
  p.shippingBoxDimensions = ⟨20, 20, 20⟩ ∧ 
  p.widgetsPerCarton = 3 → 
  widgetsInShippingBox p = 300 := by
  sorry


end NUMINAMATH_CALUDE_widgets_in_shipping_box_is_300_l3137_313709


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3137_313752

theorem quadratic_one_solution (q : ℝ) :
  (∃! x : ℝ, q * x^2 - 10 * x + 2 = 0) ↔ q = 12.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3137_313752


namespace NUMINAMATH_CALUDE_intersection_property_l3137_313768

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in polar form √2ρcos(θ + π/4) = 1 -/
def Line : Type := Unit

/-- Represents a curve in polar form ρ = 2acosθ -/
def Curve (a : ℝ) : Type := Unit

/-- Returns true if the point is on the given line -/
def Point.onLine (p : Point) (l : Line) : Prop := sorry

/-- Returns true if the point is on the given curve -/
def Point.onCurve (p : Point) (c : Curve a) : Prop := sorry

/-- Calculates the squared distance between two points -/
def Point.distanceSquared (p q : Point) : ℝ := sorry

/-- Theorem: Given the conditions, prove that a = 3 -/
theorem intersection_property (a : ℝ) (l : Line) (c : Curve a) (M P Q : Point)
    (h₁ : a > 0)
    (h₂ : M.x = 0 ∧ M.y = -1)
    (h₃ : P.onLine l ∧ P.onCurve c)
    (h₄ : Q.onLine l ∧ Q.onCurve c)
    (h₅ : P.distanceSquared Q = 4 * P.distanceSquared M * Q.distanceSquared M) :
    a = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_property_l3137_313768


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3137_313713

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 1 → 
  a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3137_313713


namespace NUMINAMATH_CALUDE_probability_point_in_circle_l3137_313763

/-- The probability of a randomly selected point from a square with side length 6
    being inside or on a circle with radius 2 centered at the center of the square -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) : 
  square_side = 6 →
  circle_radius = 2 →
  (circle_radius^2 * Real.pi) / square_side^2 = Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l3137_313763


namespace NUMINAMATH_CALUDE_roselyn_initial_books_l3137_313795

def books_problem (books_to_rebecca : ℕ) (books_remaining : ℕ) : Prop :=
  let books_to_mara := 3 * books_to_rebecca
  let total_given := books_to_rebecca + books_to_mara
  let initial_books := total_given + books_remaining
  initial_books = 220

theorem roselyn_initial_books :
  books_problem 40 60 := by
  sorry

end NUMINAMATH_CALUDE_roselyn_initial_books_l3137_313795


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3137_313744

-- Equation 1
theorem solve_equation_one (x : ℚ) : 2 * (x - 3) = 1 - 3 * (x + 1) ↔ x = 4 / 5 := by sorry

-- Equation 2
theorem solve_equation_two (x : ℚ) : 3 * x + (x - 1) / 2 = 3 - (x - 1) / 3 ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3137_313744


namespace NUMINAMATH_CALUDE_a_plus_5_simplest_radical_form_l3137_313740

/-- A function that checks if an expression is in the simplest radical form -/
def is_simplest_radical_form (x : ℝ) : Prop :=
  ∀ y : ℝ, |x| = |y| → (∃ (n : ℕ) (z : ℝ), y = z^n ∧ n > 1) → x = y

/-- The expression |a+5| is in the simplest radical form -/
theorem a_plus_5_simplest_radical_form (a : ℝ) : 
  is_simplest_radical_form (a + 5) :=
sorry

end NUMINAMATH_CALUDE_a_plus_5_simplest_radical_form_l3137_313740


namespace NUMINAMATH_CALUDE_xiao_wang_speed_l3137_313726

/-- Represents the cycling speed of Xiao Wang in km/h -/
def cycling_speed : ℝ := 10

/-- The total distance between City A and City B in km -/
def total_distance : ℝ := 55

/-- The distance Xiao Wang cycled in km -/
def cycling_distance : ℝ := 25

/-- The time difference between cycling and bus ride in hours -/
def time_difference : ℝ := 1

theorem xiao_wang_speed :
  cycling_speed = 10 ∧
  cycling_speed > 0 ∧
  total_distance = 55 ∧
  cycling_distance = 25 ∧
  time_difference = 1 ∧
  (cycling_distance / cycling_speed) = 
    ((total_distance - cycling_distance) / (2 * cycling_speed)) + time_difference :=
by sorry

end NUMINAMATH_CALUDE_xiao_wang_speed_l3137_313726


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l3137_313721

/-- Given a rectangular metallic sheet with width 36 m, from which squares of 8 m are cut from each corner
    to form a box with volume 5120 m³, prove that the length of the original sheet is 48 m. -/
theorem metallic_sheet_length (L : ℝ) : 
  let W : ℝ := 36
  let cut_length : ℝ := 8
  let box_volume : ℝ := 5120
  (L - 2 * cut_length) * (W - 2 * cut_length) * cut_length = box_volume →
  L = 48 := by
sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_l3137_313721


namespace NUMINAMATH_CALUDE_approximate_probability_of_high_quality_l3137_313764

def sample_sizes : List ℕ := [20, 50, 100, 200, 500, 1000, 1500, 2000]

def high_quality_counts : List ℕ := [19, 47, 91, 184, 462, 921, 1379, 1846]

def frequencies : List ℚ := [
  950/1000, 940/1000, 910/1000, 920/1000, 924/1000, 921/1000, 919/1000, 923/1000
]

theorem approximate_probability_of_high_quality (ε : ℚ) (hε : ε = 1/100) :
  ∃ (p : ℚ), abs (p - (List.sum frequencies / frequencies.length)) ≤ ε ∧ p = 92/100 := by
  sorry

end NUMINAMATH_CALUDE_approximate_probability_of_high_quality_l3137_313764


namespace NUMINAMATH_CALUDE_darks_wash_time_l3137_313789

/-- Represents the time for washing and drying clothes -/
structure LaundryTime where
  whites_wash : ℕ
  whites_dry : ℕ
  darks_dry : ℕ
  colors_wash : ℕ
  colors_dry : ℕ
  total_time : ℕ

/-- Theorem stating the time for washing darks -/
theorem darks_wash_time (lt : LaundryTime) 
  (h1 : lt.whites_wash = 72)
  (h2 : lt.whites_dry = 50)
  (h3 : lt.darks_dry = 65)
  (h4 : lt.colors_wash = 45)
  (h5 : lt.colors_dry = 54)
  (h6 : lt.total_time = 344) :
  ∃ (darks_wash : ℕ), 
    lt.whites_wash + darks_wash + lt.colors_wash + 
    lt.whites_dry + lt.darks_dry + lt.colors_dry = lt.total_time ∧ 
    darks_wash = 58 := by
  sorry

end NUMINAMATH_CALUDE_darks_wash_time_l3137_313789


namespace NUMINAMATH_CALUDE_last_digit_of_product_l3137_313734

theorem last_digit_of_product : (3^65 * 6^59 * 7^71) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_product_l3137_313734


namespace NUMINAMATH_CALUDE_travel_problem_solvable_l3137_313774

/-- A strategy for three friends to travel between two cities --/
structure TravelStrategy where
  /-- The time taken for all friends to reach their destinations --/
  total_time : ℝ
  /-- Assertion that the strategy is valid --/
  is_valid : Prop

/-- The travel problem setup --/
structure TravelProblem where
  /-- Distance between the two cities in km --/
  distance : ℝ
  /-- Maximum walking speed in km/h --/
  walk_speed : ℝ
  /-- Maximum cycling speed in km/h --/
  cycle_speed : ℝ

/-- The existence of a valid strategy for the given travel problem --/
def exists_valid_strategy (problem : TravelProblem) : Prop :=
  ∃ (strategy : TravelStrategy), 
    strategy.is_valid ∧ 
    strategy.total_time ≤ 160/60 ∧  -- 2 hours and 40 minutes in hours
    problem.distance = 24 ∧
    problem.walk_speed ≤ 6 ∧
    problem.cycle_speed ≤ 18

/-- Theorem stating that there exists a valid strategy for the given problem --/
theorem travel_problem_solvable : 
  ∃ (problem : TravelProblem), exists_valid_strategy problem :=
sorry

end NUMINAMATH_CALUDE_travel_problem_solvable_l3137_313774


namespace NUMINAMATH_CALUDE_jacob_age_2005_l3137_313732

/-- Given that Jacob was one-third as old as his grandfather at the end of 2000,
    and the sum of the years in which they were born is 3858,
    prove that Jacob will be 40.5 years old at the end of 2005. -/
theorem jacob_age_2005 (jacob_age_2000 : ℝ) (grandfather_age_2000 : ℝ) :
  jacob_age_2000 = (1 / 3) * grandfather_age_2000 →
  (2000 - jacob_age_2000) + (2000 - grandfather_age_2000) = 3858 →
  jacob_age_2000 + 5 = 40.5 := by
sorry

end NUMINAMATH_CALUDE_jacob_age_2005_l3137_313732


namespace NUMINAMATH_CALUDE_regression_analysis_l3137_313704

/-- Unit prices -/
def unit_prices : List ℝ := [4, 5, 6, 7, 8, 9]

/-- Sales volumes -/
def sales_volumes : List ℝ := [90, 84, 83, 80, 75, 68]

/-- Empirical regression equation -/
def regression_equation (x : ℝ) (a : ℝ) : ℝ := -4 * x + a

theorem regression_analysis :
  let avg_sales := (sales_volumes.sum) / (sales_volumes.length : ℝ)
  let slope := -4
  let a := avg_sales + 4 * ((unit_prices.sum) / (unit_prices.length : ℝ))
  (avg_sales = 80) ∧ 
  (slope = -4) ∧
  (regression_equation 10 a = 66) := by sorry

end NUMINAMATH_CALUDE_regression_analysis_l3137_313704


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l3137_313723

theorem modular_arithmetic_problem :
  ∃ (a b c : ℤ),
    (7 * a) % 60 = 1 ∧
    (13 * b) % 60 = 1 ∧
    (17 * c) % 60 = 1 ∧
    (4 * a + 12 * b - 6 * c) % 60 = 58 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l3137_313723


namespace NUMINAMATH_CALUDE_sum_of_complex_numbers_l3137_313714

theorem sum_of_complex_numbers :
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := 4 - 7*I
  let z₃ : ℂ := -2 + 3*I
  z₁ + z₂ + z₃ = 5 + I := by sorry

end NUMINAMATH_CALUDE_sum_of_complex_numbers_l3137_313714


namespace NUMINAMATH_CALUDE_even_sum_not_both_odd_l3137_313762

theorem even_sum_not_both_odd (n m : ℤ) (h : Even (n^2 + m + n * m)) :
  ¬(Odd n ∧ Odd m) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_not_both_odd_l3137_313762


namespace NUMINAMATH_CALUDE_optimal_distance_optimal_distance_with_discount_l3137_313749

/-- Represents the optimal store distance problem --/
structure OptimalStoreDistance where
  s₀ : ℝ  -- Distance from home to city center
  v : ℝ   -- Base utility value

/-- Calculates the price at a given distance --/
def price (s_m : ℝ) : ℝ :=
  1000 * (1 - 0.02 * s_m)

/-- Calculates the transportation cost --/
def transportCost (s₀ s_m : ℝ) : ℝ :=
  0.5 * (s_m - s₀)^2

/-- Calculates the utility without discount --/
def utility (osd : OptimalStoreDistance) (s_m : ℝ) : ℝ :=
  osd.v - price s_m - transportCost osd.s₀ s_m

/-- Calculates the utility with discount --/
def utilityWithDiscount (osd : OptimalStoreDistance) (s_m : ℝ) : ℝ :=
  osd.v - 0.9 * price s_m - transportCost osd.s₀ s_m

/-- Theorem: Optimal store distance without discount --/
theorem optimal_distance (osd : OptimalStoreDistance) :
  ∃ s_m : ℝ, s_m = min 60 (osd.s₀ + 20) ∧
  ∀ s : ℝ, 0 ≤ s ∧ s ≤ 60 → utility osd s_m ≥ utility osd s :=
sorry

/-- Theorem: Optimal store distance with discount --/
theorem optimal_distance_with_discount (osd : OptimalStoreDistance) :
  ∃ s_m : ℝ, s_m = min 60 (osd.s₀ + 9) ∧
  ∀ s : ℝ, 0 ≤ s ∧ s ≤ 60 → utilityWithDiscount osd s_m ≥ utilityWithDiscount osd s :=
sorry

end NUMINAMATH_CALUDE_optimal_distance_optimal_distance_with_discount_l3137_313749


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3137_313745

theorem sin_2alpha_value (α : Real) 
  (h : (1 - Real.tan α) / (1 + Real.tan α) = 3 - 2 * Real.sqrt 2) : 
  Real.sin (2 * α) = (2 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3137_313745


namespace NUMINAMATH_CALUDE_figure_to_square_possible_l3137_313755

/-- A figure composed of unit squares -/
structure UnitSquareFigure where
  area : ℕ

/-- A part of a figure after cutting -/
structure FigurePart where
  area : ℕ

/-- Represents a square -/
structure Square where
  side_length : ℕ

/-- Theorem stating that a UnitSquareFigure can be cut into three parts to form a square -/
theorem figure_to_square_possible (fig : UnitSquareFigure) 
  (h : ∃ n : ℕ, n * n = fig.area) : 
  ∃ (part1 part2 part3 : FigurePart) (sq : Square),
    part1.area + part2.area + part3.area = fig.area ∧
    sq.side_length * sq.side_length = fig.area :=
sorry

end NUMINAMATH_CALUDE_figure_to_square_possible_l3137_313755


namespace NUMINAMATH_CALUDE_car_production_total_l3137_313761

theorem car_production_total (north_america europe asia south_america : ℕ) 
  (h1 : north_america = 3884)
  (h2 : europe = 2871)
  (h3 : asia = 5273)
  (h4 : south_america = 1945) :
  north_america + europe + asia + south_america = 13973 :=
by sorry

end NUMINAMATH_CALUDE_car_production_total_l3137_313761


namespace NUMINAMATH_CALUDE_integer_pair_inequality_l3137_313731

theorem integer_pair_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (1 ≤ m^n - n^m ∧ m^n - n^m ≤ m*n) ↔ 
  ((n = 1 ∧ m ≥ 2) ∨ (m = 2 ∧ n = 5) ∨ (m = 3 ∧ n = 2)) := by
sorry

end NUMINAMATH_CALUDE_integer_pair_inequality_l3137_313731


namespace NUMINAMATH_CALUDE_farm_chicken_count_l3137_313730

/-- Represents the number of chickens on a farm -/
structure FarmChickens where
  roosters : ℕ
  hens : ℕ

/-- Given a farm with the specified conditions, proves that the total number of chickens is 75 -/
theorem farm_chicken_count (farm : FarmChickens) 
  (hen_count : farm.hens = 67)
  (rooster_hen_relation : farm.hens = 9 * farm.roosters - 5) :
  farm.roosters + farm.hens = 75 := by
  sorry


end NUMINAMATH_CALUDE_farm_chicken_count_l3137_313730


namespace NUMINAMATH_CALUDE_max_cubes_fit_l3137_313703

theorem max_cubes_fit (large_side : ℕ) (small_edge : ℕ) : large_side = 10 ∧ small_edge = 2 →
  (large_side ^ 3) / (small_edge ^ 3) = 125 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_fit_l3137_313703


namespace NUMINAMATH_CALUDE_equation_solution_l3137_313720

theorem equation_solution : ∃ x : ℚ, (1 / 7 + 7 / x = 16 / x + 1 / 16) ∧ x = 112 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3137_313720


namespace NUMINAMATH_CALUDE_pirate_costume_cost_l3137_313766

theorem pirate_costume_cost (num_friends : ℕ) (cost_per_costume : ℕ) : 
  num_friends = 8 → cost_per_costume = 5 → num_friends * cost_per_costume = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_pirate_costume_cost_l3137_313766


namespace NUMINAMATH_CALUDE_cubic_extrema_l3137_313775

/-- A cubic function f(x) = ax³ + bx² where a > 0 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2

theorem cubic_extrema (a b : ℝ) (h₁ : a > 0) :
  (∀ x, f a b x ≤ f a b 0) ∧  -- maximum at x = 0
  (∀ x, f a b x ≥ f a b (1/3)) -- minimum at x = 1/3
  → a + 2*b = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_extrema_l3137_313775


namespace NUMINAMATH_CALUDE_difference_of_41st_terms_l3137_313728

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem difference_of_41st_terms : 
  let C := arithmetic_sequence 50 15
  let D := arithmetic_sequence 50 (-15)
  |C 41 - D 41| = 1200 := by sorry

end NUMINAMATH_CALUDE_difference_of_41st_terms_l3137_313728


namespace NUMINAMATH_CALUDE_gcd_of_90_and_405_l3137_313733

theorem gcd_of_90_and_405 : Nat.gcd 90 405 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_90_and_405_l3137_313733


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l3137_313759

theorem least_number_divisible_by_five_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → 
    ¬(∃ (q₁ q₂ q₃ q₄ q₅ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0)) ∧
  n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l3137_313759


namespace NUMINAMATH_CALUDE_complement_of_B_l3137_313717

-- Define the set B
def B : Set ℝ := {x | x^2 - 3*x + 2 < 0}

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- State the theorem
theorem complement_of_B : 
  Set.compl B = {x : ℝ | x ≤ 1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_l3137_313717


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant3_l3137_313739

/-- Represents a linear function y = mx + b -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in Quadrant 3 -/
def isInQuadrant3 (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Checks if a point is on the line defined by a linear function -/
def isOnLine (f : LinearFunction) (p : Point) : Prop :=
  p.y = f.slope * p.x + f.yIntercept

theorem linear_function_not_in_quadrant3 (f : LinearFunction) 
  (h1 : f.slope = -3)
  (h2 : f.yIntercept = 5) :
  ¬∃ p : Point, isInQuadrant3 p ∧ isOnLine f p :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant3_l3137_313739


namespace NUMINAMATH_CALUDE_crystal_sales_revenue_l3137_313758

def original_cupcake_price : ℚ := 3
def original_cookie_price : ℚ := 2
def discount_factor : ℚ := 1/2
def cupcakes_sold : ℕ := 16
def cookies_sold : ℕ := 8

theorem crystal_sales_revenue : 
  (original_cupcake_price * discount_factor * cupcakes_sold) + 
  (original_cookie_price * discount_factor * cookies_sold) = 32 := by
sorry

end NUMINAMATH_CALUDE_crystal_sales_revenue_l3137_313758


namespace NUMINAMATH_CALUDE_remainder_1999_11_mod_8_l3137_313784

theorem remainder_1999_11_mod_8 : 1999^11 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1999_11_mod_8_l3137_313784


namespace NUMINAMATH_CALUDE_new_assessed_value_calculation_l3137_313750

/-- Represents the property tax calculation in Township K -/
structure PropertyTax where
  initialValue : ℝ
  newValue : ℝ
  taxRate : ℝ
  taxIncrease : ℝ

/-- Theorem stating the relationship between tax increase and new assessed value -/
theorem new_assessed_value_calculation (p : PropertyTax)
  (h1 : p.initialValue = 20000)
  (h2 : p.taxRate = 0.1)
  (h3 : p.taxIncrease = 800)
  (h4 : p.taxRate * p.newValue - p.taxRate * p.initialValue = p.taxIncrease) :
  p.newValue = 28000 := by
  sorry

end NUMINAMATH_CALUDE_new_assessed_value_calculation_l3137_313750


namespace NUMINAMATH_CALUDE_min_value_expression_l3137_313769

theorem min_value_expression (p x : ℝ) (h1 : 0 < p) (h2 : p < 15) (h3 : p ≤ x) (h4 : x ≤ 15) :
  (∀ y, p ≤ y ∧ y ≤ 15 → |x - p| + |x - 15| + |x - p - 15| ≤ |y - p| + |y - 15| + |y - p - 15|) →
  |x - p| + |x - 15| + |x - p - 15| = 15 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3137_313769


namespace NUMINAMATH_CALUDE_candies_needed_to_fill_bags_l3137_313791

theorem candies_needed_to_fill_bags (total_candies : ℕ) (bag_capacity : ℕ) (h1 : total_candies = 254) (h2 : bag_capacity = 30) : 
  (bag_capacity - (total_candies % bag_capacity)) = 16 := by
sorry

end NUMINAMATH_CALUDE_candies_needed_to_fill_bags_l3137_313791


namespace NUMINAMATH_CALUDE_valid_B_values_l3137_313727

def is_valid_B (B : ℕ) : Prop :=
  B < 10 ∧ (∃ k : ℤ, 40000 + 1110 * B + 2 = 9 * k)

theorem valid_B_values :
  ∀ B : ℕ, is_valid_B B ↔ (B = 1 ∨ B = 4 ∨ B = 7) :=
by sorry

end NUMINAMATH_CALUDE_valid_B_values_l3137_313727


namespace NUMINAMATH_CALUDE_number_difference_l3137_313747

theorem number_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : |x - y| = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3137_313747


namespace NUMINAMATH_CALUDE_max_different_digits_is_eight_l3137_313706

/-- A natural number satisfying the divisibility condition -/
def DivisibleNumber (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ Finset.range 10 → d ≠ 0 → (n.digits 10).contains d → n % d = 0

/-- The maximum number of different digits in a DivisibleNumber -/
def MaxDifferentDigits : ℕ := 8

/-- Theorem stating the maximum number of different digits in a DivisibleNumber -/
theorem max_different_digits_is_eight :
  ∃ n : ℕ, DivisibleNumber n ∧ (n.digits 10).card = MaxDifferentDigits ∧
  ∀ m : ℕ, DivisibleNumber m → (m.digits 10).card ≤ MaxDifferentDigits :=
sorry

end NUMINAMATH_CALUDE_max_different_digits_is_eight_l3137_313706


namespace NUMINAMATH_CALUDE_donny_gas_change_l3137_313780

/-- Calculates the change Donny will receive after filling up his gas tank. -/
theorem donny_gas_change (tank_capacity : ℝ) (current_fuel : ℝ) (cost_per_liter : ℝ) (amount_paid : ℝ)
  (h1 : tank_capacity = 150)
  (h2 : current_fuel = 38)
  (h3 : cost_per_liter = 3)
  (h4 : amount_paid = 350) :
  amount_paid - (tank_capacity - current_fuel) * cost_per_liter = 14 := by
  sorry

end NUMINAMATH_CALUDE_donny_gas_change_l3137_313780


namespace NUMINAMATH_CALUDE_single_elimination_games_tournament_with_23_teams_l3137_313725

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  num_teams : ℕ
  num_games : ℕ

/-- The number of games in a single-elimination tournament is one less than the number of teams. -/
theorem single_elimination_games (t : SingleEliminationTournament) 
  (h : t.num_teams > 0) : t.num_games = t.num_teams - 1 := by
  sorry

/-- In a single-elimination tournament with 23 teams, 22 games are required to declare a winner. -/
theorem tournament_with_23_teams : 
  ∃ t : SingleEliminationTournament, t.num_teams = 23 ∧ t.num_games = 22 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_games_tournament_with_23_teams_l3137_313725


namespace NUMINAMATH_CALUDE_collinear_probability_is_7_6325_l3137_313772

/-- A 5x5 grid of dots -/
structure Grid :=
  (size : Nat)
  (h_size : size = 5)

/-- The number of collinear sets of four dots in a 5x5 grid -/
def collinear_sets (g : Grid) : Nat := 14

/-- The total number of ways to choose 4 dots from 25 -/
def total_sets (g : Grid) : Nat := 12650

/-- The probability of selecting four collinear dots from a 5x5 grid -/
def collinear_probability (g : Grid) : ℚ :=
  (collinear_sets g : ℚ) / (total_sets g : ℚ)

/-- Theorem stating that the probability of selecting four collinear dots from a 5x5 grid is 7/6325 -/
theorem collinear_probability_is_7_6325 (g : Grid) :
  collinear_probability g = 7 / 6325 := by
  sorry

end NUMINAMATH_CALUDE_collinear_probability_is_7_6325_l3137_313772


namespace NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l3137_313735

/-- Given a point P(-3, 5), its symmetrical point P' with respect to the x-axis has coordinates (-3, -5). -/
theorem symmetry_wrt_x_axis :
  let P : ℝ × ℝ := (-3, 5)
  let P' : ℝ × ℝ := (-3, -5)
  (P'.1 = P.1) ∧ (P'.2 = -P.2) := by sorry

end NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l3137_313735


namespace NUMINAMATH_CALUDE_value_of_expression_l3137_313776

-- Define the function g
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

-- State the theorem
theorem value_of_expression (p q r s : ℝ) : g p q r s 3 = 6 → 6*p - 3*q + 2*r - s = 0 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3137_313776


namespace NUMINAMATH_CALUDE_dog_cat_food_weight_difference_l3137_313760

theorem dog_cat_food_weight_difference :
  -- Define the constants from the problem
  let cat_food_bags : ℕ := 2
  let cat_food_weight_per_bag : ℕ := 3 -- in pounds
  let dog_food_bags : ℕ := 2
  let ounces_per_pound : ℕ := 16
  let total_pet_food_ounces : ℕ := 256

  -- Calculate total cat food weight in ounces
  let total_cat_food_ounces : ℕ := cat_food_bags * cat_food_weight_per_bag * ounces_per_pound
  
  -- Calculate total dog food weight in ounces
  let total_dog_food_ounces : ℕ := total_pet_food_ounces - total_cat_food_ounces
  
  -- Calculate weight per bag of dog food in ounces
  let dog_food_weight_per_bag_ounces : ℕ := total_dog_food_ounces / dog_food_bags
  
  -- Calculate weight per bag of cat food in ounces
  let cat_food_weight_per_bag_ounces : ℕ := cat_food_weight_per_bag * ounces_per_pound
  
  -- Calculate the difference in weight between dog and cat food bags in ounces
  let weight_difference_ounces : ℕ := dog_food_weight_per_bag_ounces - cat_food_weight_per_bag_ounces
  
  -- Convert the weight difference to pounds
  weight_difference_ounces / ounces_per_pound = 2 := by
  sorry

end NUMINAMATH_CALUDE_dog_cat_food_weight_difference_l3137_313760


namespace NUMINAMATH_CALUDE_pizza_toppings_l3137_313797

theorem pizza_toppings (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h1 : total_slices = 16)
  (h2 : pepperoni_slices = 8)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ 
     slice ∈ Finset.range mushroom_slices)) :
  ∃ both : ℕ, both = pepperoni_slices + mushroom_slices - total_slices :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l3137_313797


namespace NUMINAMATH_CALUDE_expression_evaluation_l3137_313794

theorem expression_evaluation : (5 + 2 + 6) * 2 / 3 - 4 / 3 = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3137_313794


namespace NUMINAMATH_CALUDE_digital_root_prime_probability_l3137_313757

/-- The digital root of a positive integer -/
def digitalRoot (n : ℕ+) : ℕ :=
  if n.val % 9 = 0 then 9 else n.val % 9

/-- Whether a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The count of numbers with prime digital roots in the first n positive integers -/
def countPrimeDigitalRoots (n : ℕ) : ℕ := sorry

theorem digital_root_prime_probability :
  (countPrimeDigitalRoots 1000 : ℚ) / 1000 = 444 / 1000 := by sorry

end NUMINAMATH_CALUDE_digital_root_prime_probability_l3137_313757


namespace NUMINAMATH_CALUDE_PA_square_property_l3137_313781

theorem PA_square_property : 
  {PA : ℕ | 
    10 ≤ PA ∧ PA < 100 ∧ 
    1000 ≤ PA^2 ∧ PA^2 < 10000 ∧
    (PA^2 / 1000 = PA / 10) ∧ 
    (PA^2 % 10 = PA % 10)} = {95, 96} := by
  sorry

end NUMINAMATH_CALUDE_PA_square_property_l3137_313781


namespace NUMINAMATH_CALUDE_chips_left_uneaten_l3137_313798

/-- Calculates the number of chips left uneaten when half of a batch of cookies is consumed. -/
theorem chips_left_uneaten (chips_per_cookie : ℕ) (dozens : ℕ) : 
  chips_per_cookie = 7 → dozens = 4 → (dozens * 12 / 2) * chips_per_cookie = 168 := by
  sorry

#check chips_left_uneaten

end NUMINAMATH_CALUDE_chips_left_uneaten_l3137_313798


namespace NUMINAMATH_CALUDE_exercise_distance_l3137_313751

/-- 
Proves that given a person who walks x miles at 3 miles per hour, 
runs 10 miles at 5 miles per hour, repeats this exercise 7 times a week, 
and spends a total of 21 hours exercising per week, 
the value of x must be 3 miles.
-/
theorem exercise_distance (x : ℝ) 
  (h_walk_speed : ℝ := 3)
  (h_run_speed : ℝ := 5)
  (h_run_distance : ℝ := 10)
  (h_days_per_week : ℕ := 7)
  (h_total_hours : ℝ := 21)
  (h_exercise_time : ℝ := h_total_hours / h_days_per_week)
  (h_time_equation : x / h_walk_speed + h_run_distance / h_run_speed = h_exercise_time) :
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_exercise_distance_l3137_313751


namespace NUMINAMATH_CALUDE_coefficient_x21_eq_932_l3137_313792

open Nat BigOperators Finset

/-- The coefficient of x^21 in the expansion of (1 + x + x^2 + ... + x^20)(1 + x + x^2 + ... + x^10)^3 -/
def coefficient_x21 : ℕ :=
  (Finset.range 22).sum (λ i => i.choose 3) -
  3 * ((Finset.range 15).sum (λ i => i.choose 3)) +
  1

/-- The geometric series (1 + x + x^2 + ... + x^n) -/
def geometric_sum (n : ℕ) (x : ℝ) : ℝ :=
  (Finset.range (n + 1)).sum (λ i => x ^ i)

theorem coefficient_x21_eq_932 :
  coefficient_x21 = 932 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x21_eq_932_l3137_313792


namespace NUMINAMATH_CALUDE_beetles_eaten_per_day_l3137_313718

/-- The number of beetles eaten by one bird per day -/
def beetles_per_bird : ℕ := 12

/-- The number of birds eaten by one snake per day -/
def birds_per_snake : ℕ := 3

/-- The number of snakes eaten by one jaguar per day -/
def snakes_per_jaguar : ℕ := 5

/-- The number of jaguars in the forest -/
def jaguars_in_forest : ℕ := 6

/-- The total number of beetles eaten per day in the forest -/
def total_beetles_eaten : ℕ := 
  jaguars_in_forest * snakes_per_jaguar * birds_per_snake * beetles_per_bird

theorem beetles_eaten_per_day :
  total_beetles_eaten = 1080 := by
  sorry

end NUMINAMATH_CALUDE_beetles_eaten_per_day_l3137_313718


namespace NUMINAMATH_CALUDE_sum_inequality_l3137_313710

theorem sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  let S := a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c)
  0 < S ∧ S < 1 := by sorry

end NUMINAMATH_CALUDE_sum_inequality_l3137_313710


namespace NUMINAMATH_CALUDE_mice_meet_in_six_days_l3137_313741

/-- The thickness of the wall in feet -/
def wall_thickness : ℚ := 64 + 31/32

/-- The distance burrowed by both mice after n days -/
def total_distance (n : ℕ) : ℚ := 2^n - 1/(2^(n-1)) + 1

/-- The number of days it takes for the mice to meet -/
def days_to_meet : ℕ := 6

/-- Theorem stating that the mice meet after 6 days -/
theorem mice_meet_in_six_days :
  total_distance days_to_meet = wall_thickness :=
sorry

end NUMINAMATH_CALUDE_mice_meet_in_six_days_l3137_313741


namespace NUMINAMATH_CALUDE_fish_total_weight_l3137_313738

/-- The weight of a fish given specific conditions about its parts -/
def fish_weight (tail_weight head_weight body_weight : ℝ) : Prop :=
  tail_weight = 4 ∧
  head_weight = tail_weight + (body_weight / 2) ∧
  body_weight = head_weight + tail_weight

theorem fish_total_weight :
  ∀ (tail_weight head_weight body_weight : ℝ),
    fish_weight tail_weight head_weight body_weight →
    tail_weight + head_weight + body_weight = 32 :=
by sorry

end NUMINAMATH_CALUDE_fish_total_weight_l3137_313738


namespace NUMINAMATH_CALUDE_intersection_points_count_l3137_313782

/-- Represents a line with a specific number of points -/
structure Line where
  numPoints : ℕ

/-- Represents a configuration of two parallel lines -/
structure ParallelLines where
  line1 : Line
  line2 : Line

/-- Calculates the number of intersection points for a given configuration of parallel lines -/
def intersectionPoints (pl : ParallelLines) : ℕ :=
  (pl.line1.numPoints.choose 2) * (pl.line2.numPoints.choose 2)

/-- The specific configuration of parallel lines in our problem -/
def problemConfig : ParallelLines :=
  { line1 := { numPoints := 10 }
    line2 := { numPoints := 11 } }

theorem intersection_points_count :
  intersectionPoints problemConfig = 2475 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l3137_313782


namespace NUMINAMATH_CALUDE_monotonicity_and_range_of_a_l3137_313729

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (x + a) / (a * Real.exp x)

theorem monotonicity_and_range_of_a :
  (a ≠ 0) →
  ((a > 0 → 
    (∀ x₁ x₂, x₁ < 1 - a ∧ x₂ < 1 - a → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂, x₁ > 1 - a ∧ x₂ > 1 - a → f a x₁ > f a x₂)) ∧
   (a < 0 → 
    (∀ x₁ x₂, x₁ < 1 - a ∧ x₂ < 1 - a → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, x₁ > 1 - a ∧ x₂ > 1 - a → f a x₁ < f a x₂))) ∧
  ((∀ x > 0, (3 + 2 * Real.log x) / Real.exp x ≤ f a x + 2 * x) →
   (a ∈ Set.Iic (-1/2) ∪ Set.Ioi 0)) := by
  sorry

end NUMINAMATH_CALUDE_monotonicity_and_range_of_a_l3137_313729


namespace NUMINAMATH_CALUDE_quadratic_property_l3137_313743

/-- Quadratic function -/
def f (c : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + c

theorem quadratic_property (c : ℝ) (x₁ : ℝ) (hc : c < 0) (hx₁ : f c x₁ > 0) :
  f c (x₁ - 2) < 0 ∧ f c (x₁ + 2) < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_property_l3137_313743


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l3137_313778

theorem no_positive_integer_solution :
  ¬ ∃ (x y z t : ℕ+), (x^2 + 5*y^2 = z^2) ∧ (5*x^2 + y^2 = t^2) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l3137_313778


namespace NUMINAMATH_CALUDE_power_of_64_five_sixths_l3137_313786

theorem power_of_64_five_sixths : (64 : ℝ) ^ (5/6) = 32 := by sorry

end NUMINAMATH_CALUDE_power_of_64_five_sixths_l3137_313786


namespace NUMINAMATH_CALUDE_school_teachers_l3137_313700

/-- Calculates the number of teachers in a school given specific conditions -/
theorem school_teachers (students : ℕ) (classes_per_student : ℕ) (classes_per_teacher : ℕ) (students_per_class : ℕ) :
  students = 2400 →
  classes_per_student = 5 →
  classes_per_teacher = 4 →
  students_per_class = 30 →
  (students * classes_per_student) / (classes_per_teacher * students_per_class) = 100 := by
  sorry

end NUMINAMATH_CALUDE_school_teachers_l3137_313700


namespace NUMINAMATH_CALUDE_dereks_initial_lunch_spending_l3137_313796

/-- Represents the problem of determining Derek's initial lunch spending --/
theorem dereks_initial_lunch_spending 
  (derek_initial : ℕ) 
  (derek_dad_lunch : ℕ) 
  (derek_extra_lunch : ℕ) 
  (dave_initial : ℕ) 
  (dave_mom_lunch : ℕ) 
  (dave_extra : ℕ) 
  (h1 : derek_initial = 40)
  (h2 : derek_dad_lunch = 11)
  (h3 : derek_extra_lunch = 5)
  (h4 : dave_initial = 50)
  (h5 : dave_mom_lunch = 7)
  (h6 : dave_extra = 33)
  : ∃ (derek_self_lunch : ℕ), 
    derek_self_lunch = 14 ∧ 
    dave_initial - dave_mom_lunch = 
    (derek_initial - (derek_self_lunch + derek_dad_lunch + derek_extra_lunch)) + dave_extra :=
by sorry

end NUMINAMATH_CALUDE_dereks_initial_lunch_spending_l3137_313796


namespace NUMINAMATH_CALUDE_lcm_18_24_30_l3137_313742

theorem lcm_18_24_30 : Nat.lcm (Nat.lcm 18 24) 30 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_30_l3137_313742


namespace NUMINAMATH_CALUDE_log_9_729_l3137_313722

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_9_729 : log 9 729 = 3 := by sorry

end NUMINAMATH_CALUDE_log_9_729_l3137_313722
