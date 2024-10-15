import Mathlib

namespace NUMINAMATH_CALUDE_min_pages_for_baseball_cards_l2960_296084

/-- Represents the number of cards that can be held by each type of page -/
structure PageCapacity where
  x : Nat
  y : Nat

/-- Calculates the minimum number of pages needed to hold all cards -/
def minPages (totalCards : Nat) (capacity : PageCapacity) : Nat :=
  let fullXPages := totalCards / capacity.x
  let remainingCards := totalCards % capacity.x
  if remainingCards = 0 then
    fullXPages
  else if remainingCards ≤ capacity.y then
    fullXPages + 1
  else
    fullXPages + 2

/-- Theorem stating the minimum number of pages needed for the given problem -/
theorem min_pages_for_baseball_cards :
  let totalCards := 1040
  let capacity : PageCapacity := { x := 12, y := 10 }
  minPages totalCards capacity = 87 := by
  sorry

#eval minPages 1040 { x := 12, y := 10 }

end NUMINAMATH_CALUDE_min_pages_for_baseball_cards_l2960_296084


namespace NUMINAMATH_CALUDE_house_sale_loss_percentage_l2960_296042

def initial_value : ℝ := 100000
def profit_percentage : ℝ := 0.10
def final_selling_price : ℝ := 99000

theorem house_sale_loss_percentage :
  let first_sale_price := initial_value * (1 + profit_percentage)
  let loss_amount := first_sale_price - final_selling_price
  let loss_percentage := loss_amount / first_sale_price * 100
  loss_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_house_sale_loss_percentage_l2960_296042


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2960_296023

theorem consecutive_integers_square_sum (x : ℤ) : 
  x^2 + (x+1)^2 + x^2 * (x+1)^2 = (x^2 + x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2960_296023


namespace NUMINAMATH_CALUDE_symmetric_point_6_1_l2960_296074

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point2D := ⟨0, 0⟩

/-- Function to find the symmetric point with respect to the origin -/
def symmetricPoint (p : Point2D) : Point2D :=
  ⟨-p.x, -p.y⟩

/-- Theorem: The point symmetric to (6, 1) with respect to the origin is (-6, -1) -/
theorem symmetric_point_6_1 :
  symmetricPoint ⟨6, 1⟩ = ⟨-6, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_6_1_l2960_296074


namespace NUMINAMATH_CALUDE_sum_of_ratios_l2960_296083

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) = f a * f b

theorem sum_of_ratios (f : ℝ → ℝ) 
  (h1 : functional_equation f) 
  (h2 : f 1 = 2) : 
  f 2 / f 1 + f 4 / f 3 + f 6 / f 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_l2960_296083


namespace NUMINAMATH_CALUDE_train_length_proof_l2960_296019

theorem train_length_proof (passing_time : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  passing_time = 8 →
  platform_length = 279 →
  crossing_time = 20 →
  ∃ (train_length : ℝ),
    train_length = passing_time * (train_length + platform_length) / crossing_time ∧
    train_length = 186 :=
by sorry

end NUMINAMATH_CALUDE_train_length_proof_l2960_296019


namespace NUMINAMATH_CALUDE_kendras_earnings_theorem_l2960_296077

/-- Kendra's total earnings in 2014 and 2015 -/
def kendras_total_earnings (laurel_2014 : ℝ) : ℝ :=
  let kendra_2014 := laurel_2014 - 8000
  let kendra_2015 := laurel_2014 * 1.2
  kendra_2014 + kendra_2015

/-- Theorem stating Kendra's total earnings given Laurel's 2014 earnings -/
theorem kendras_earnings_theorem (laurel_2014 : ℝ) 
  (h : laurel_2014 = 30000) : 
  kendras_total_earnings laurel_2014 = 58000 := by
  sorry

end NUMINAMATH_CALUDE_kendras_earnings_theorem_l2960_296077


namespace NUMINAMATH_CALUDE_study_time_calculation_l2960_296081

theorem study_time_calculation (total_hours : ℝ) (tv_fraction : ℝ) (study_fraction : ℝ) : 
  total_hours = 24 →
  tv_fraction = 1/5 →
  study_fraction = 1/4 →
  (total_hours * (1 - tv_fraction) * study_fraction) * 60 = 288 := by
sorry

end NUMINAMATH_CALUDE_study_time_calculation_l2960_296081


namespace NUMINAMATH_CALUDE_coral_reef_age_conversion_l2960_296097

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : Nat) : Nat :=
  let d0 := octal % 10
  let d1 := (octal / 10) % 10
  let d2 := (octal / 100) % 10
  let d3 := (octal / 1000) % 10
  d0 * 8^0 + d1 * 8^1 + d2 * 8^2 + d3 * 8^3

theorem coral_reef_age_conversion :
  octal_to_decimal 3456 = 1838 := by
  sorry

end NUMINAMATH_CALUDE_coral_reef_age_conversion_l2960_296097


namespace NUMINAMATH_CALUDE_power_of_two_expression_l2960_296037

theorem power_of_two_expression : 2^3 * 2^3 + 2^3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_expression_l2960_296037


namespace NUMINAMATH_CALUDE_only_third_set_forms_triangle_l2960_296041

/-- Checks if three lengths can form a triangle according to the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of line segments given in the problem -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(1, 2, 3), (2, 2, 4), (3, 4, 5), (3, 5, 9)]

theorem only_third_set_forms_triangle :
  ∃! set : ℝ × ℝ × ℝ, set ∈ segment_sets ∧ can_form_triangle set.1 set.2.1 set.2.2 :=
by
  sorry

end NUMINAMATH_CALUDE_only_third_set_forms_triangle_l2960_296041


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2960_296093

theorem quadratic_inequality (x : ℝ) : 
  -10 * x^2 + 4 * x + 2 > 0 ↔ (1 - Real.sqrt 6) / 5 < x ∧ x < (1 + Real.sqrt 6) / 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2960_296093


namespace NUMINAMATH_CALUDE_stuffed_animals_theorem_l2960_296056

/-- Given the number of stuffed animals for McKenna (M), Kenley (K), and Tenly (T),
    prove various properties about their stuffed animal collection. -/
theorem stuffed_animals_theorem (M K T : ℕ) (S : ℕ) (A F : ℚ) 
    (hM : M = 34)
    (hK : K = 2 * M)
    (hT : T = K + 5)
    (hS : S = M + K + T)
    (hA : A = S / 3)
    (hF : F = M / S) : 
  K = 68 ∧ 
  T = 73 ∧ 
  S = 175 ∧ 
  A = 175 / 3 ∧ 
  F = 34 / 175 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_theorem_l2960_296056


namespace NUMINAMATH_CALUDE_max_value_theorem_l2960_296015

theorem max_value_theorem (x y : ℝ) (h : x^2 - 3*x + 4*y = 7) :
  ∃ (M : ℝ), M = 16 ∧ ∀ (x' y' : ℝ), x'^2 - 3*x' + 4*y' = 7 → 3*x' + 4*y' ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2960_296015


namespace NUMINAMATH_CALUDE_charitable_distribution_result_l2960_296009

def charitable_distribution (initial : ℕ) : ℕ :=
  let to_farmer := initial / 2 + 1
  let after_farmer := initial - to_farmer
  let to_beggar := after_farmer / 2 + 2
  let after_beggar := after_farmer - to_beggar
  let to_boy := after_beggar / 2 + 3
  after_beggar - to_boy

theorem charitable_distribution_result :
  charitable_distribution 42 = 1 := by
  sorry

end NUMINAMATH_CALUDE_charitable_distribution_result_l2960_296009


namespace NUMINAMATH_CALUDE_angle_range_theorem_l2960_296088

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is monotonically increasing on an interval (a, b) if
    for all x, y in (a, b), x < y implies f(x) < f(y) -/
def MonoIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem angle_range_theorem (f : ℝ → ℝ) (A : ℝ) :
  IsOdd f →
  MonoIncreasing f 0 Real.pi →
  f (1/2) = 0 →
  f (Real.cos A) < 0 →
  (π/3 < A ∧ A < π/2) ∨ (2*π/3 < A ∧ A < π) :=
sorry

end NUMINAMATH_CALUDE_angle_range_theorem_l2960_296088


namespace NUMINAMATH_CALUDE_lottery_probability_l2960_296092

/-- A lottery with probabilities for certain number ranges -/
structure Lottery where
  prob_1_to_45 : ℚ
  prob_1_or_larger : ℚ
  prob_1_to_45_is_valid : prob_1_to_45 = 7/15
  prob_1_or_larger_is_valid : prob_1_or_larger = 14/15

/-- The probability of drawing a number less than or equal to 45 in the lottery -/
def prob_le_45 (l : Lottery) : ℚ := l.prob_1_to_45

theorem lottery_probability (l : Lottery) :
  prob_le_45 l = l.prob_1_to_45 := by sorry

end NUMINAMATH_CALUDE_lottery_probability_l2960_296092


namespace NUMINAMATH_CALUDE_sum_interior_angles_octagon_l2960_296024

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of interior angles of an octagon is 1080° -/
theorem sum_interior_angles_octagon :
  sum_interior_angles octagon_sides = 1080 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_octagon_l2960_296024


namespace NUMINAMATH_CALUDE_original_class_size_l2960_296073

theorem original_class_size (x : ℕ) : 
  (40 * x + 15 * 32 = (x + 15) * 36) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_class_size_l2960_296073


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_l2960_296098

def mean (list : List ℚ) : ℚ := (list.sum) / list.length

theorem mean_equality_implies_z (z : ℚ) : 
  mean [7, 10, 15, 21] = mean [18, z] → z = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_l2960_296098


namespace NUMINAMATH_CALUDE_sine_cosine_equality_l2960_296063

theorem sine_cosine_equality (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (612 * π / 180) → n = -18 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_equality_l2960_296063


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2960_296008

theorem necessary_not_sufficient_condition 
  (A B C : Set α) 
  (hAnonempty : A.Nonempty) 
  (hBnonempty : B.Nonempty) 
  (hCnonempty : C.Nonempty) 
  (hUnion : A ∪ B = C) 
  (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ 
  (∃ y, y ∈ C ∧ y ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2960_296008


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l2960_296001

/-- Given vectors in ℝ², prove that if a + 3b is parallel to c, then m = -6 -/
theorem vector_parallel_condition (a b c : ℝ × ℝ) (m : ℝ) 
  (ha : a = (-2, 3))
  (hb : b = (3, 1))
  (hc : c = (-7, m))
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a + 3 • b = k • c) :
  m = -6 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l2960_296001


namespace NUMINAMATH_CALUDE_relationship_xyz_l2960_296065

theorem relationship_xyz (x y z : ℝ) 
  (hx : x = Real.rpow 0.5 0.5)
  (hy : y = Real.rpow 0.5 1.3)
  (hz : z = Real.rpow 1.3 0.5) :
  y < x ∧ x < z := by
  sorry

end NUMINAMATH_CALUDE_relationship_xyz_l2960_296065


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_x_minus_one_to_eighth_l2960_296026

theorem coefficient_x_squared_in_x_minus_one_to_eighth (x : ℝ) : 
  (∃ a b c d e f g : ℝ, (x - 1)^8 = x^8 + 8*x^7 + 28*x^6 + 56*x^5 + 70*x^4 + a*x^3 + b*x^2 + c*x + d) ∧ 
  (∃ p q r s t u v : ℝ, (x - 1)^8 = p*x^7 + q*x^6 + r*x^5 + s*x^4 + t*x^3 + 28*x^2 + u*x + v) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_x_minus_one_to_eighth_l2960_296026


namespace NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_35_power_87_plus_93_power_49_l2960_296046

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to get the units digit of a power of 5
def unitsDigitPowerOf5 (n : ℕ) : ℕ := 5

-- Define a function to get the units digit of a power of 3
def unitsDigitPowerOf3 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0 -- This case should never occur

theorem units_digit_of_sum (a b : ℕ) :
  unitsDigit (a + b) = unitsDigit (unitsDigit a + unitsDigit b) :=
sorry

theorem units_digit_of_35_power_87_plus_93_power_49 :
  unitsDigit ((35 ^ 87) + (93 ^ 49)) = 8 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_35_power_87_plus_93_power_49_l2960_296046


namespace NUMINAMATH_CALUDE_green_peaches_count_l2960_296010

/-- Given a number of baskets, red peaches per basket, and total peaches,
    calculates the number of green peaches per basket. -/
def green_peaches_per_basket (num_baskets : ℕ) (red_per_basket : ℕ) (total_peaches : ℕ) : ℕ :=
  (total_peaches - num_baskets * red_per_basket) / num_baskets

/-- Proves that there are 4 green peaches in each basket given the problem conditions. -/
theorem green_peaches_count :
  green_peaches_per_basket 15 19 345 = 4 := by
  sorry

#eval green_peaches_per_basket 15 19 345

end NUMINAMATH_CALUDE_green_peaches_count_l2960_296010


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2960_296076

-- Define the quadrilateral ABCD
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the point of intersection of diagonals
def P (q : Quadrilateral) : Point := sorry

-- Define the distances from A, B, and P to line CD
def a (q : Quadrilateral) : ℝ := sorry
def b (q : Quadrilateral) : ℝ := sorry
def p (q : Quadrilateral) : ℝ := sorry

-- Define the length of side CD
def CD (q : Quadrilateral) : ℝ := sorry

-- Define the area of the quadrilateral
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area (q : Quadrilateral) : 
  area q = (a q * b q * CD q) / (2 * p q) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2960_296076


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2960_296049

theorem complex_fraction_evaluation :
  (⌈(19 : ℚ) / 7 - ⌈(35 : ℚ) / 19⌉⌉) / (⌈(35 : ℚ) / 7 + ⌈(7 * 19 : ℚ) / 35⌉⌉) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2960_296049


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2960_296095

theorem absolute_value_inequality (x : ℝ) : 
  |x - 2| + |x + 1| < 4 ↔ x ∈ Set.Ioo (-7/2) (-1) ∪ Set.Ico (-1) (5/2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2960_296095


namespace NUMINAMATH_CALUDE_odd_function_negative_values_l2960_296053

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_values
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = x + 1) :
  ∀ x < 0, f x = x - 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_negative_values_l2960_296053


namespace NUMINAMATH_CALUDE_inequality_proof_l2960_296030

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2*y^2 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2960_296030


namespace NUMINAMATH_CALUDE_polynomial_roots_product_l2960_296033

theorem polynomial_roots_product (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 2*x - 1 = 0 → x^6 - b*x - c = 0) → 
  b * c = 2030 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_product_l2960_296033


namespace NUMINAMATH_CALUDE_inequality_proof_l2960_296071

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_condition : a^2 + b^2 + c^2 + (a+b+c)^2 ≤ 4) :
  (a*b + 1) / (a+b)^2 + (b*c + 1) / (b+c)^2 + (c*a + 1) / (c+a)^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2960_296071


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l2960_296078

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- Proves that 12 kilometers per second is equal to 43,200 kilometers per hour -/
theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 12 = 43200 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l2960_296078


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_l2960_296062

theorem largest_power_dividing_factorial_squared (p : ℕ) (hp : Nat.Prime p) :
  (∀ n : ℕ, (Nat.factorial p)^n ∣ Nat.factorial (p^2)) ↔ n ≤ p + 1 :=
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_l2960_296062


namespace NUMINAMATH_CALUDE_polynomial_division_l2960_296035

theorem polynomial_division (x : ℝ) :
  x^5 + 3*x^4 - 28*x^3 + 15*x^2 - 21*x + 8 =
  (x - 3) * (x^4 + 6*x^3 - 10*x^2 - 15*x - 66) + (-100) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l2960_296035


namespace NUMINAMATH_CALUDE_gcd_18_30_42_l2960_296086

theorem gcd_18_30_42 : Nat.gcd 18 (Nat.gcd 30 42) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_42_l2960_296086


namespace NUMINAMATH_CALUDE_absolute_value_calculation_l2960_296091

theorem absolute_value_calculation : |-3| - (Real.sqrt 7 + 1)^0 - 2^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_calculation_l2960_296091


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2960_296082

theorem complex_expression_evaluation :
  ∀ (c d : ℂ), c = 7 - 3*I → d = 2 + 5*I → 3*c - 4*d = 13 - 29*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2960_296082


namespace NUMINAMATH_CALUDE_expression_value_l2960_296066

theorem expression_value (x y z : ℝ) (hx : x = 3) (hy : y = 2) (hz : z = 1) :
  3 * x^2 - 4 * y + 2 * z = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2960_296066


namespace NUMINAMATH_CALUDE_sheet_width_is_36_l2960_296012

/-- Represents the dimensions and properties of a metallic sheet and the box formed from it. -/
structure SheetAndBox where
  sheet_length : ℝ
  sheet_width : ℝ
  cut_square_side : ℝ
  box_volume : ℝ

/-- Calculates the volume of the box formed from the sheet. -/
def box_volume (s : SheetAndBox) : ℝ :=
  (s.sheet_length - 2 * s.cut_square_side) * (s.sheet_width - 2 * s.cut_square_side) * s.cut_square_side

/-- Theorem stating that given the specified conditions, the width of the sheet is 36 meters. -/
theorem sheet_width_is_36 (s : SheetAndBox) 
    (h1 : s.sheet_length = 48)
    (h2 : s.cut_square_side = 7)
    (h3 : s.box_volume = 5236)
    (h4 : box_volume s = s.box_volume) : 
  s.sheet_width = 36 := by
  sorry

#check sheet_width_is_36

end NUMINAMATH_CALUDE_sheet_width_is_36_l2960_296012


namespace NUMINAMATH_CALUDE_equilateral_triangle_revolution_surface_area_l2960_296013

/-- The surface area of a solid of revolution formed by rotating an equilateral triangle -/
theorem equilateral_triangle_revolution_surface_area 
  (side_length : ℝ) 
  (h_side : side_length = 2) : 
  let solid_surface_area := 2 * Real.pi * (side_length * Real.sqrt 3 / 2) * side_length
  solid_surface_area = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_revolution_surface_area_l2960_296013


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l2960_296044

theorem sufficient_condition_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l2960_296044


namespace NUMINAMATH_CALUDE_right_angle_point_location_l2960_296039

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the plane
def Point := ℝ × ℝ

-- Define the property of being on the circle
def OnCircle (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the angle between three points
def Angle (p1 p2 p3 : Point) : ℝ := sorry

-- Define a right angle
def IsRightAngle (angle : ℝ) : Prop :=
  angle = Real.pi / 2

-- Define the property of being diametrically opposite
def DiametricallyOpposite (p1 p2 : Point) (c : Circle) : Prop :=
  (p1.1 + p2.1) / 2 = c.center.1 ∧ (p1.2 + p2.2) / 2 = c.center.2

-- The main theorem
theorem right_angle_point_location
  (c : Circle) (C : Point) (ho : OnCircle C c) :
  ∃! X, OnCircle X c ∧ IsRightAngle (Angle C X c.center) ∧ DiametricallyOpposite C X c :=
sorry

end NUMINAMATH_CALUDE_right_angle_point_location_l2960_296039


namespace NUMINAMATH_CALUDE_joes_juices_l2960_296047

/-- The number of juices Joe bought at the market -/
def num_juices : ℕ := 7

/-- The cost of a single orange in dollars -/
def orange_cost : ℚ := 4.5

/-- The cost of a single juice in dollars -/
def juice_cost : ℚ := 0.5

/-- The cost of a single jar of honey in dollars -/
def honey_cost : ℚ := 5

/-- The cost of two plants in dollars -/
def two_plants_cost : ℚ := 18

/-- The total amount Joe spent at the market in dollars -/
def total_spent : ℚ := 68

/-- The number of oranges Joe bought -/
def num_oranges : ℕ := 3

/-- The number of jars of honey Joe bought -/
def num_honey : ℕ := 3

/-- The number of plants Joe bought -/
def num_plants : ℕ := 4

theorem joes_juices :
  num_juices * juice_cost = 
    total_spent - (num_oranges * orange_cost + num_honey * honey_cost + (num_plants / 2) * two_plants_cost) :=
by sorry

end NUMINAMATH_CALUDE_joes_juices_l2960_296047


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l2960_296090

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l2960_296090


namespace NUMINAMATH_CALUDE_average_temperature_l2960_296006

/-- The average temperature of three days with recorded temperatures of -14°F, -8°F, and +1°F is -7°F. -/
theorem average_temperature (temp1 temp2 temp3 : ℚ) : 
  temp1 = -14 → temp2 = -8 → temp3 = 1 → (temp1 + temp2 + temp3) / 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l2960_296006


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l2960_296011

/-- Given two circles C₁ and C₂, where C₁ has equation (x+1)²+(y-1)²=1 and C₂ is symmetric to C₁
    with respect to the line x-y-1=0, prove that the equation of C₂ is (x-2)²+(y+2)²=1 -/
theorem symmetric_circle_equation (x y : ℝ) :
  let C₁ : ℝ → ℝ → Prop := λ x y => (x + 1)^2 + (y - 1)^2 = 1
  let symmetry_line : ℝ → ℝ → Prop := λ x y => x - y - 1 = 0
  let C₂ : ℝ → ℝ → Prop := λ x y => (x - 2)^2 + (y + 2)^2 = 1
  (∀ x y, C₁ x y ↔ (x + 1)^2 + (y - 1)^2 = 1) →
  (∀ x₁ y₁ x₂ y₂, C₁ x₁ y₁ → C₂ x₂ y₂ → 
    ∃ x_sym y_sym, symmetry_line x_sym y_sym ∧
    (x₂ - x_sym = x_sym - x₁) ∧ (y₂ - y_sym = y_sym - y₁)) →
  (∀ x y, C₂ x y ↔ (x - 2)^2 + (y + 2)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l2960_296011


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_sqrt_three_l2960_296055

theorem reciprocal_of_negative_sqrt_three :
  ((-Real.sqrt 3)⁻¹ : ℝ) = -(Real.sqrt 3 / 3) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_sqrt_three_l2960_296055


namespace NUMINAMATH_CALUDE_trig_identity_l2960_296052

theorem trig_identity (α : ℝ) : 
  4.62 * (Real.cos (2 * α))^4 - 6 * (Real.cos (2 * α))^2 * (Real.sin (2 * α))^2 + (Real.sin (2 * α))^4 = Real.cos (8 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2960_296052


namespace NUMINAMATH_CALUDE_missing_digit_is_five_l2960_296016

def largest_number (x : ℕ) : ℕ :=
  if x ≥ 2 then 9000 + 100 * x + 21 else 9000 + 200 + x

def smallest_number (x : ℕ) : ℕ := 1000 + 200 + 90 + x

theorem missing_digit_is_five :
  ∀ x : ℕ, x < 10 →
    largest_number x - smallest_number x = 8262 →
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_missing_digit_is_five_l2960_296016


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2960_296007

theorem perpendicular_slope (x₁ y₁ x₂ y₂ : ℚ) (hx : x₁ ≠ x₂) :
  let m₁ := (y₂ - y₁) / (x₂ - x₁)
  let m₂ := -1 / m₁
  x₁ = 3 ∧ y₁ = -3 ∧ x₂ = -4 ∧ y₂ = 2 → m₂ = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2960_296007


namespace NUMINAMATH_CALUDE_fraction_addition_l2960_296058

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 9 = (11 : ℚ) / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2960_296058


namespace NUMINAMATH_CALUDE_ratio_problem_l2960_296004

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 5) :
  x / y = 13 / 9 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l2960_296004


namespace NUMINAMATH_CALUDE_magazine_cover_theorem_l2960_296014

theorem magazine_cover_theorem (n : ℕ) (S : ℝ) (h1 : n = 15) (h2 : S > 0) :
  ∃ (remaining_area : ℝ), remaining_area ≥ (8 / 15) * S ∧
  ∃ (remaining_magazines : ℕ), remaining_magazines = n - 7 :=
by
  sorry

end NUMINAMATH_CALUDE_magazine_cover_theorem_l2960_296014


namespace NUMINAMATH_CALUDE_evaluate_expression_l2960_296020

theorem evaluate_expression : (3^2)^2 - (2^3)^3 = -431 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2960_296020


namespace NUMINAMATH_CALUDE_nth_equation_l2960_296036

theorem nth_equation (n : ℕ) (h : n > 0) :
  (n + 2 : ℚ) / n - 2 / (n + 2) = ((n + 2)^2 + n^2 : ℚ) / (n * (n + 2)) - 1 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l2960_296036


namespace NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l2960_296005

/-- Represents a rectangle divided into nine squares with integer side lengths -/
structure NineSquareRectangle where
  a : ℕ  -- Side length of the smallest square
  b : ℕ  -- Side length of the second smallest square
  length : ℕ  -- Length of the rectangle
  width : ℕ   -- Width of the rectangle

/-- The perimeter of a rectangle -/
def perimeter (rect : NineSquareRectangle) : ℕ :=
  2 * (rect.length + rect.width)

/-- Conditions for a valid NineSquareRectangle configuration -/
def is_valid_configuration (rect : NineSquareRectangle) : Prop :=
  rect.b = 3 * rect.a ∧
  rect.length = 2 * rect.a + rect.b + 3 * rect.a + rect.b ∧
  rect.width = 12 * rect.a - 2 * rect.b + 8 * rect.a - rect.b

/-- Theorem stating the smallest possible perimeter of a NineSquareRectangle -/
theorem min_perimeter_nine_square_rectangle :
  ∀ rect : NineSquareRectangle, is_valid_configuration rect →
  perimeter rect ≥ 52 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l2960_296005


namespace NUMINAMATH_CALUDE_bulbs_chosen_l2960_296075

theorem bulbs_chosen (total : ℕ) (defective : ℕ) (prob : ℝ) :
  total = 20 →
  defective = 4 →
  prob = 0.368421052631579 →
  (∃ n : ℕ, n = 2 ∧ 1 - ((total - defective : ℝ) / total) ^ n = prob) :=
by sorry

end NUMINAMATH_CALUDE_bulbs_chosen_l2960_296075


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2960_296069

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 6) / (x - 4) ≥ 3 ↔ x ∈ Set.Iio 4 ∪ Set.Ioi 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2960_296069


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l2960_296061

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := a * 10 + b / 99

/-- The main theorem stating that the ratio of two specific repeating decimals equals 9/4 -/
theorem repeating_decimal_ratio : 
  (RepeatingDecimal 8 1) / (RepeatingDecimal 3 6) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l2960_296061


namespace NUMINAMATH_CALUDE_division_problem_l2960_296089

theorem division_problem (divisor : ℕ) : 
  (109 / divisor = 9) ∧ (109 % divisor = 1) → divisor = 12 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l2960_296089


namespace NUMINAMATH_CALUDE_paperboy_delivery_patterns_l2960_296043

def deliverySequences (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 15
  | m + 5 => deliverySequences (m + 4) + deliverySequences (m + 3) + 
             deliverySequences (m + 2) + deliverySequences (m + 1)

theorem paperboy_delivery_patterns : deliverySequences 12 = 2873 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_patterns_l2960_296043


namespace NUMINAMATH_CALUDE_binomial_expectation_six_half_l2960_296018

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- Theorem: The expected value of X ~ B(6, 1/2) is 3 -/
theorem binomial_expectation_six_half :
  let X : BinomialDistribution := ⟨6, 1/2, by norm_num⟩
  expectation X = 3 := by sorry

end NUMINAMATH_CALUDE_binomial_expectation_six_half_l2960_296018


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2960_296072

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 + a*y + 4 = 0 → y = x) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2960_296072


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2960_296050

theorem min_value_quadratic :
  ∃ (m : ℝ), (∀ x : ℝ, 4 * x^2 + 8 * x + 12 ≥ m) ∧
  (∃ x : ℝ, 4 * x^2 + 8 * x + 12 = m) ∧
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2960_296050


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2960_296031

/-- An arithmetic sequence with specific terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 9
  ninth_term : a 9 = 3

/-- The general term of the arithmetic sequence -/
def generalTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  15 - 3/2 * (n - 1)

/-- The term from which the sequence becomes negative -/
def negativeStartTerm (seq : ArithmeticSequence) : ℕ := 13

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = generalTerm seq n) ∧
  (∀ n, n ≥ negativeStartTerm seq → seq.a n < 0) ∧
  (∀ n, n < negativeStartTerm seq → seq.a n ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2960_296031


namespace NUMINAMATH_CALUDE_same_angle_from_P_l2960_296054

-- Define the basic geometric objects
structure Point : Type :=
  (x : ℝ) (y : ℝ)

structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define the given conditions
def P : Point := sorry
def Q : Point := sorry
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry

def circle1 : Circle := sorry
def circle2 : Circle := sorry

-- Define the properties of the configuration
def circles_intersect (c1 c2 : Circle) (P Q : Point) : Prop := sorry
def line_perpendicular_to_PQ (A Q B : Point) : Prop := sorry
def Q_between_A_and_B (A Q B : Point) : Prop := sorry
def tangent_intersection (A B C : Point) (c1 c2 : Circle) : Prop := sorry

-- Define the angle between three points
def angle (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem same_angle_from_P :
  circles_intersect circle1 circle2 P Q →
  line_perpendicular_to_PQ A Q B →
  Q_between_A_and_B A Q B →
  tangent_intersection A B C circle1 circle2 →
  angle B P Q = angle Q P A := by sorry

end NUMINAMATH_CALUDE_same_angle_from_P_l2960_296054


namespace NUMINAMATH_CALUDE_min_n_is_correct_l2960_296021

/-- The minimum positive integer n for which the expansion of (x^2 + 1/(3x^3))^n contains a constant term -/
def min_n : ℕ := 5

/-- Predicate to check if the expansion contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ (r : ℕ), 2 * n = 5 * r

theorem min_n_is_correct :
  (∀ m : ℕ, m > 0 ∧ m < min_n → ¬(has_constant_term m)) ∧
  has_constant_term min_n :=
sorry

end NUMINAMATH_CALUDE_min_n_is_correct_l2960_296021


namespace NUMINAMATH_CALUDE_root_sum_equals_three_l2960_296032

theorem root_sum_equals_three (x₁ x₂ : ℝ) 
  (h₁ : x₁ + Real.log x₁ = 3) 
  (h₂ : x₂ + (10 : ℝ) ^ x₂ = 3) : 
  x₁ + x₂ = 3 := by sorry

end NUMINAMATH_CALUDE_root_sum_equals_three_l2960_296032


namespace NUMINAMATH_CALUDE_consecutive_odd_product_sum_l2960_296038

theorem consecutive_odd_product_sum (a b c : ℤ) : 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧  -- a, b, c are odd
  (b = a + 2) ∧ (c = b + 2) ∧                -- a, b, c are consecutive
  (a * b * c = 9177) →                       -- their product is 9177
  (a + b + c = 63) :=                        -- their sum is 63
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_sum_l2960_296038


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2960_296064

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 + 2 * Real.sqrt 2 ∧ x₂ = 3 - 2 * Real.sqrt 2 ∧
    x₁^2 - 6*x₁ + 1 = 0 ∧ x₂^2 - 6*x₂ + 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1 ∧ y₂ = 1/2 ∧
    2*(y₁+1)^2 = 3*(y₁+1) ∧ 2*(y₂+1)^2 = 3*(y₂+1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2960_296064


namespace NUMINAMATH_CALUDE_missing_number_equation_l2960_296057

theorem missing_number_equation (x : ℤ) : 10010 - 12 * 3 * x = 9938 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l2960_296057


namespace NUMINAMATH_CALUDE_modulus_z_is_sqrt_2_l2960_296060

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number z -/
noncomputable def z : ℂ := (2 * i) / (1 + i)

/-- Theorem: The modulus of z is √2 -/
theorem modulus_z_is_sqrt_2 : Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_z_is_sqrt_2_l2960_296060


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l2960_296067

/-- The number of large seats on the Ferris wheel -/
def num_large_seats : ℕ := 7

/-- The total number of people that can be accommodated on large seats -/
def total_people_large_seats : ℕ := 84

/-- The number of people each large seat can hold -/
def people_per_large_seat : ℕ := total_people_large_seats / num_large_seats

theorem ferris_wheel_capacity : people_per_large_seat = 12 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l2960_296067


namespace NUMINAMATH_CALUDE_amount_to_hand_in_l2960_296027

/-- Represents the denominations of bills in US currency --/
inductive Denomination
  | Hundred
  | Fifty
  | Twenty
  | Ten
  | Five
  | One

/-- Represents the quantity of each denomination in the till --/
def till_contents : Denomination → ℕ
  | Denomination.Hundred => 2
  | Denomination.Fifty => 1
  | Denomination.Twenty => 5
  | Denomination.Ten => 3
  | Denomination.Five => 7
  | Denomination.One => 27

/-- The value of each denomination in dollars --/
def denomination_value : Denomination → ℕ
  | Denomination.Hundred => 100
  | Denomination.Fifty => 50
  | Denomination.Twenty => 20
  | Denomination.Ten => 10
  | Denomination.Five => 5
  | Denomination.One => 1

/-- The amount to be left in the till --/
def amount_to_leave : ℕ := 300

/-- Calculates the total value of bills in the till --/
def total_value : ℕ := sorry

/-- Theorem: The amount Jack will hand in is $142 --/
theorem amount_to_hand_in :
  total_value - amount_to_leave = 142 := by sorry

end NUMINAMATH_CALUDE_amount_to_hand_in_l2960_296027


namespace NUMINAMATH_CALUDE_two_digit_special_property_l2960_296087

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

theorem two_digit_special_property : 
  {n : ℕ | is_two_digit n ∧ n = 6 * sum_of_digits (n + 7)} = {24, 78} := by
  sorry

end NUMINAMATH_CALUDE_two_digit_special_property_l2960_296087


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_60_degree_angle_l2960_296028

theorem isosceles_triangle_with_60_degree_angle (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧ -- Angles are positive
  α + β + β = 180 ∧ -- Sum of angles in a triangle
  α = 60 ∧ -- One angle is 60°
  β = β -- Triangle is isosceles with two equal angles
  → α = 60 ∧ β = 60 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_with_60_degree_angle_l2960_296028


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l2960_296022

theorem roots_of_quadratic (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l2960_296022


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2960_296029

theorem complex_expression_evaluation : 
  (39/7) / ((8.4 * (6/7) * (6 - ((2.3 + 5/6.25) * 7) / (8 * 0.0125 + 6.9))) - 20.384/1.3) = 15/14 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2960_296029


namespace NUMINAMATH_CALUDE_perfect_square_analysis_l2960_296051

theorem perfect_square_analysis :
  (∃ (x : ℕ), 8^2050 = x^2) ∧
  (∃ (x : ℕ), 9^2048 = x^2) ∧
  (∀ (x : ℕ), 10^2051 ≠ x^2) ∧
  (∃ (x : ℕ), 11^2052 = x^2) ∧
  (∃ (x : ℕ), 12^2050 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_analysis_l2960_296051


namespace NUMINAMATH_CALUDE_painting_distance_l2960_296096

theorem painting_distance (wall_width painting_width : ℝ) 
  (hw : wall_width = 26) 
  (hp : painting_width = 4) : 
  (wall_width - painting_width) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_painting_distance_l2960_296096


namespace NUMINAMATH_CALUDE_remainder_512_power_512_mod_13_l2960_296045

theorem remainder_512_power_512_mod_13 : 512^512 ≡ 1 [ZMOD 13] := by sorry

end NUMINAMATH_CALUDE_remainder_512_power_512_mod_13_l2960_296045


namespace NUMINAMATH_CALUDE_largest_share_proof_l2960_296040

def profit_split (partners : ℕ) (ratios : List ℕ) (total_profit : ℕ) : ℕ :=
  let total_parts := ratios.sum
  let part_value := total_profit / total_parts
  (ratios.maximum? |>.getD 0) * part_value

theorem largest_share_proof (partners : ℕ) (ratios : List ℕ) (total_profit : ℕ) 
  (h_partners : partners = 5)
  (h_ratios : ratios = [1, 2, 3, 3, 6])
  (h_profit : total_profit = 36000) :
  profit_split partners ratios total_profit = 14400 :=
by
  sorry

#eval profit_split 5 [1, 2, 3, 3, 6] 36000

end NUMINAMATH_CALUDE_largest_share_proof_l2960_296040


namespace NUMINAMATH_CALUDE_number_with_inserted_zero_l2960_296094

def insert_zero (n : ℕ) : ℕ :=
  10000 * (n / 1000) + 1000 * ((n / 100) % 10) + (n % 100)

theorem number_with_inserted_zero (N : ℕ) :
  (insert_zero N = 9 * N) → (N = 225 ∨ N = 450 ∨ N = 675) := by
sorry

end NUMINAMATH_CALUDE_number_with_inserted_zero_l2960_296094


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l2960_296000

theorem regular_polygon_with_150_degree_angles (n : ℕ) : 
  n > 2 → (∀ θ : ℝ, θ = 150 → n * θ = (n - 2) * 180) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l2960_296000


namespace NUMINAMATH_CALUDE_balls_in_boxes_l2960_296068

-- Define the number of balls and boxes
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Define the function to calculate the number of ways to distribute balls
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

-- State the theorem
theorem balls_in_boxes : 
  distribute_balls num_balls num_boxes = 21 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l2960_296068


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2960_296034

/-- A quadratic form ax^2 + bxy + cy^2 is a perfect square if and only if its discriminant b^2 - 4ac is zero. -/
def is_perfect_square (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- If 4x^2 + mxy + 25y^2 is a perfect square, then m = ±20. -/
theorem perfect_square_condition (m : ℝ) :
  is_perfect_square 4 m 25 → m = 20 ∨ m = -20 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2960_296034


namespace NUMINAMATH_CALUDE_clayton_first_game_score_l2960_296059

def clayton_basketball_score (game1 : ℝ) : Prop :=
  let game2 : ℝ := 14
  let game3 : ℝ := 6
  let game4 : ℝ := (game1 + game2 + game3) / 3
  let total : ℝ := 40
  (game1 + game2 + game3 + game4 = total) ∧ (game1 = 10)

theorem clayton_first_game_score :
  ∃ (game1 : ℝ), clayton_basketball_score game1 :=
sorry

end NUMINAMATH_CALUDE_clayton_first_game_score_l2960_296059


namespace NUMINAMATH_CALUDE_alcohol_mixture_ratio_l2960_296070

/-- Proves that mixing equal volumes of two alcohol solutions results in a specific alcohol-to-water ratio -/
theorem alcohol_mixture_ratio (volume : ℝ) (p_concentration q_concentration : ℝ)
  (h_volume_pos : volume > 0)
  (h_p_conc : p_concentration = 0.625)
  (h_q_conc : q_concentration = 0.875) :
  let total_volume := 2 * volume
  let total_alcohol := volume * (p_concentration + q_concentration)
  let total_water := total_volume - total_alcohol
  (total_alcohol / total_water) = 3 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_ratio_l2960_296070


namespace NUMINAMATH_CALUDE_promotions_equivalent_l2960_296003

/-- Calculates the discount percentage for a given promotion --/
def discount_percentage (items_taken : ℕ) (items_paid : ℕ) : ℚ :=
  (items_taken - items_paid : ℚ) / items_taken * 100

/-- The original promotion "Buy one and get another for half price" --/
def original_promotion : ℚ := discount_percentage 2 (3/2)

/-- The alternative promotion "Take four and pay for three" --/
def alternative_promotion : ℚ := discount_percentage 4 3

/-- Theorem stating that both promotions offer the same discount --/
theorem promotions_equivalent : original_promotion = alternative_promotion := by
  sorry

end NUMINAMATH_CALUDE_promotions_equivalent_l2960_296003


namespace NUMINAMATH_CALUDE_stamp_cost_l2960_296017

/-- The cost of stamps problem -/
theorem stamp_cost (cost_per_stamp : ℕ) (num_stamps : ℕ) : 
  cost_per_stamp = 34 → num_stamps = 4 → cost_per_stamp * num_stamps = 136 := by
  sorry

end NUMINAMATH_CALUDE_stamp_cost_l2960_296017


namespace NUMINAMATH_CALUDE_g_minus_two_equals_eleven_l2960_296048

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 3*x + 1

-- State the theorem
theorem g_minus_two_equals_eleven : g (-2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_g_minus_two_equals_eleven_l2960_296048


namespace NUMINAMATH_CALUDE_line_intercept_sum_l2960_296002

/-- Given a line 5x + 8y + c = 0, if the sum of its x-intercept and y-intercept is 26, then c = -80 -/
theorem line_intercept_sum (c : ℝ) : 
  (∃ x y : ℝ, 5*x + 8*y + c = 0 ∧ 5*x + c = 0 ∧ 8*y + c = 0 ∧ x + y = 26) → 
  c = -80 :=
by sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l2960_296002


namespace NUMINAMATH_CALUDE_sum_of_three_squares_divisible_by_three_to_not_divisible_by_three_l2960_296079

theorem sum_of_three_squares_divisible_by_three_to_not_divisible_by_three
  (N : ℕ) (a b c : ℤ) (h1 : ∃ (a b c : ℤ), N = a^2 + b^2 + c^2)
  (h2 : 3 ∣ a) (h3 : 3 ∣ b) (h4 : 3 ∣ c) :
  ∃ (x y z : ℤ), N = x^2 + y^2 + z^2 ∧ ¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_divisible_by_three_to_not_divisible_by_three_l2960_296079


namespace NUMINAMATH_CALUDE_max_distance_after_braking_l2960_296099

/-- The distance function for a car after braking -/
def s (b : ℝ) (t : ℝ) : ℝ := -6 * t^2 + b * t

/-- Theorem: Maximum distance traveled by the car after braking -/
theorem max_distance_after_braking (b : ℝ) :
  s b (1/2) = 6 → ∃ (t_max : ℝ), ∀ (t : ℝ), s b t ≤ s b t_max ∧ s b t_max = 75/8 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_after_braking_l2960_296099


namespace NUMINAMATH_CALUDE_total_portfolios_l2960_296025

theorem total_portfolios (num_students : ℕ) (portfolios_per_student : ℕ) 
  (h1 : num_students = 15)
  (h2 : portfolios_per_student = 8) :
  num_students * portfolios_per_student = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_portfolios_l2960_296025


namespace NUMINAMATH_CALUDE_friend_payment_percentage_l2960_296080

def adoption_fee : ℝ := 200
def james_payment : ℝ := 150

theorem friend_payment_percentage : 
  (adoption_fee - james_payment) / adoption_fee * 100 = 25 := by sorry

end NUMINAMATH_CALUDE_friend_payment_percentage_l2960_296080


namespace NUMINAMATH_CALUDE_number_of_incorrect_statements_l2960_296085

-- Define the triangles
def triangle1 (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 9^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 12^2 ∧
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2)

def triangle2 (a b c : ℝ) : Prop :=
  a = 7 ∧ b = 24 ∧ c = 25

def triangle3 (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 6^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 8^2 ∧
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2)

def triangle4 (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 3 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 3)

-- Define the statements
def statement1 (A B C : ℝ × ℝ) : Prop :=
  triangle1 A B C → abs ((B.2 - A.2) * C.1 + (A.1 - B.1) * C.2 + (B.1 * A.2 - A.1 * B.2)) / 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 9

def statement2 (a b c : ℝ) : Prop :=
  triangle2 a b c → a^2 + b^2 = c^2

def statement3 (A B C : ℝ × ℝ) : Prop :=
  triangle3 A B C → ∃ (M : ℝ × ℝ), 
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    Real.sqrt ((M.1 - C.1)^2 + (M.2 - C.2)^2) = 5

def statement4 (a b c : ℝ) : Prop :=
  triangle4 a b c → a + b + c = 13

-- Theorem to prove
theorem number_of_incorrect_statements :
  ∃ (A1 B1 C1 A3 B3 C3 : ℝ × ℝ) (a2 b2 c2 a4 b4 c4 : ℝ),
    (¬ statement1 A1 B1 C1) ∧
    statement2 a2 b2 c2 ∧
    (¬ statement3 A3 B3 C3) ∧
    (¬ statement4 a4 b4 c4) := by
  sorry

end NUMINAMATH_CALUDE_number_of_incorrect_statements_l2960_296085
