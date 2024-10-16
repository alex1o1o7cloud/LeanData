import Mathlib

namespace NUMINAMATH_CALUDE_book_distribution_ways_l696_69634

theorem book_distribution_ways (n m : ℕ) (h1 : n = 3) (h2 : m = 2) : 
  n * (n - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_ways_l696_69634


namespace NUMINAMATH_CALUDE_m_range_l696_69685

theorem m_range (m : ℝ) (h1 : m < 0) (h2 : ∀ x : ℝ, x^2 + m*x + 1 > 0) : m ∈ Set.Ioo (-2 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l696_69685


namespace NUMINAMATH_CALUDE_shifted_parabola_equation_l696_69687

/-- Represents a vertical shift of a parabola -/
def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := 
  λ x => f x + shift

/-- The original parabola function -/
def original_parabola : ℝ → ℝ := 
  λ x => x^2 - 1

/-- The shifted parabola G -/
def G : ℝ → ℝ := vertical_shift original_parabola 3

/-- Theorem stating that G is equivalent to x^2 + 2 -/
theorem shifted_parabola_equation : G = λ x => x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_shifted_parabola_equation_l696_69687


namespace NUMINAMATH_CALUDE_average_problem_l696_69641

theorem average_problem (t b c : ℝ) (h : (t + b + c + 29) / 4 = 15) :
  (t + b + c + 14 + 15) / 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l696_69641


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l696_69678

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 3 = 0) 
  (h2 : n ≥ 30) : ∃ (blue red green yellow : ℕ),
  blue = n / 3 ∧ 
  red = n / 3 ∧ 
  green = 10 ∧ 
  yellow = n - (blue + red + green) ∧ 
  yellow ≥ 0 ∧ 
  ∀ m : ℕ, m < n → ¬(∃ (b r g y : ℕ), 
    b = m / 3 ∧ 
    r = m / 3 ∧ 
    g = 10 ∧ 
    y = m - (b + r + g) ∧ 
    y ≥ 0 ∧ 
    m % 3 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l696_69678


namespace NUMINAMATH_CALUDE_book_pages_calculation_l696_69606

/-- Represents the number of pages read in a book over a week -/
def BookPages : ℕ → ℕ → ℕ → ℕ → ℕ := λ d1 d2 d3 d4 =>
  d1 * 30 + d2 * 50 + d4

theorem book_pages_calculation :
  BookPages 2 4 1 70 = 330 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l696_69606


namespace NUMINAMATH_CALUDE_wolf_hunting_problem_l696_69690

theorem wolf_hunting_problem (hunting_wolves : ℕ) (pack_wolves : ℕ) (meat_per_wolf : ℕ) 
  (hunting_days : ℕ) (meat_per_deer : ℕ) : 
  hunting_wolves = 4 → 
  pack_wolves = 16 → 
  meat_per_wolf = 8 → 
  hunting_days = 5 → 
  meat_per_deer = 200 → 
  (hunting_wolves + pack_wolves) * meat_per_wolf * hunting_days / meat_per_deer / hunting_wolves = 1 := by
  sorry

#check wolf_hunting_problem

end NUMINAMATH_CALUDE_wolf_hunting_problem_l696_69690


namespace NUMINAMATH_CALUDE_alan_age_is_29_l696_69686

/-- Represents the ages of Alan and Chris --/
structure Ages where
  alan : ℕ
  chris : ℕ

/-- The condition that the sum of their ages is 52 --/
def sum_of_ages (ages : Ages) : Prop :=
  ages.alan + ages.chris = 52

/-- The complex age relationship between Alan and Chris --/
def age_relationship (ages : Ages) : Prop :=
  ages.chris = ages.alan - (ages.alan - (ages.alan - (ages.alan / 3)))

/-- The theorem stating Alan's age is 29 given the conditions --/
theorem alan_age_is_29 (ages : Ages) 
  (h1 : sum_of_ages ages) 
  (h2 : age_relationship ages) : 
  ages.alan = 29 := by
  sorry


end NUMINAMATH_CALUDE_alan_age_is_29_l696_69686


namespace NUMINAMATH_CALUDE_directed_line_a_value_l696_69656

/-- A line passing through two points with a specific direction vector form --/
structure DirectedLine where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  direction_vector : ℝ × ℝ

/-- The theorem stating the value of 'a' for the given line --/
theorem directed_line_a_value (L : DirectedLine) 
  (h1 : L.point1 = (-3, 7))
  (h2 : L.point2 = (2, 1))
  (h3 : ∃ a : ℝ, L.direction_vector = (a, -1)) :
  ∃ a : ℝ, L.direction_vector = (5/6, -1) := by
  sorry


end NUMINAMATH_CALUDE_directed_line_a_value_l696_69656


namespace NUMINAMATH_CALUDE_newspaper_reading_time_l696_69666

/-- Represents Hank's daily reading habits and total weekly reading time -/
structure ReadingHabits where
  newspaper_time : ℕ  -- Time spent reading newspaper each weekday morning
  novel_time : ℕ      -- Time spent reading novel each weekday evening (60 minutes)
  weekday_count : ℕ   -- Number of weekdays (5)
  weekend_count : ℕ   -- Number of weekend days (2)
  total_time : ℕ      -- Total reading time in a week (810 minutes)

/-- Theorem stating that given Hank's reading habits, he spends 30 minutes reading the newspaper each morning -/
theorem newspaper_reading_time (h : ReadingHabits) 
  (h_novel : h.novel_time = 60)
  (h_weekday : h.weekday_count = 5)
  (h_weekend : h.weekend_count = 2)
  (h_total : h.total_time = 810) :
  h.newspaper_time = 30 := by
  sorry


end NUMINAMATH_CALUDE_newspaper_reading_time_l696_69666


namespace NUMINAMATH_CALUDE_apple_plum_ratio_l696_69642

theorem apple_plum_ratio :
  ∀ (apples plums : ℕ),
    apples = 180 →
    apples + plums = 240 →
    (2 : ℚ) / 5 * (apples + plums) = 96 →
    (apples : ℚ) / plums = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_apple_plum_ratio_l696_69642


namespace NUMINAMATH_CALUDE_display_rows_count_l696_69640

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- The total number of cans in the first n rows of the display -/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

theorem display_rows_count :
  ∃ n : ℕ, total_cans n = 169 ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_display_rows_count_l696_69640


namespace NUMINAMATH_CALUDE_clean_city_people_l696_69653

/-- The number of people in group A -/
def group_A : ℕ := 54

/-- The number of people in group B -/
def group_B : ℕ := group_A - 17

/-- The number of people in group C -/
def group_C : ℕ := 2 * group_B

/-- The number of people in group D -/
def group_D : ℕ := group_A / 3

/-- The total number of people working together to clean the city -/
def total_people : ℕ := group_A + group_B + group_C + group_D

theorem clean_city_people : total_people = 183 := by
  sorry

end NUMINAMATH_CALUDE_clean_city_people_l696_69653


namespace NUMINAMATH_CALUDE_waiter_customers_l696_69629

theorem waiter_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) :
  num_tables = 6 →
  women_per_table = 3 →
  men_per_table = 5 →
  num_tables * (women_per_table + men_per_table) = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l696_69629


namespace NUMINAMATH_CALUDE_quadratic_function_characterization_l696_69665

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ y → (deriv f) ((x + y) / 2) = (f y - f x) / (y - x)

/-- The theorem stating that any differentiable function satisfying the functional equation
    is a quadratic function -/
theorem quadratic_function_characterization (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : SatisfiesFunctionalEquation f) :
    ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_characterization_l696_69665


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l696_69628

theorem pyramid_height_equals_cube_volume (cube_edge : Real) (pyramid_base : Real) (pyramid_height : Real) : 
  cube_edge = 6 →
  pyramid_base = 12 →
  (1 / 3) * pyramid_base^2 * pyramid_height = cube_edge^3 →
  pyramid_height = 4.5 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l696_69628


namespace NUMINAMATH_CALUDE_product_of_fraction_parts_l696_69615

/-- The decimal representation of the number we're considering -/
def repeating_decimal : ℚ := 0.018018018018018018018018018018018018018018018018018

/-- Express the repeating decimal as a fraction in lowest terms -/
def decimal_to_fraction (d : ℚ) : ℚ := d

/-- Calculate the product of numerator and denominator of a fraction -/
def numerator_denominator_product (q : ℚ) : ℕ :=
  (q.num.natAbs) * (q.den)

/-- Theorem stating that the product of numerator and denominator of 0.018̅ in lowest terms is 222 -/
theorem product_of_fraction_parts : 
  numerator_denominator_product (decimal_to_fraction repeating_decimal) = 222 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fraction_parts_l696_69615


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l696_69602

/-- Given a hyperbola with equation y²/a² - x²/b² = 1 where a > 0 and b > 0,
    if the eccentricity is √3, then the equation of its asymptotes is x ± √2y = 0 -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 3) →
  (∃ (x y : ℝ), x^2 - 2 * y^2 = 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l696_69602


namespace NUMINAMATH_CALUDE_roots_difference_is_one_l696_69613

-- Define the polynomial
def f (x : ℝ) : ℝ := 64 * x^3 - 144 * x^2 + 92 * x - 15

-- Define the roots
def roots : Set ℝ := {x : ℝ | f x = 0}

-- Define the arithmetic progression property
def is_arithmetic_progression (s : Set ℝ) : Prop :=
  ∃ (a d : ℝ), s = {a - d, a, a + d}

-- Theorem statement
theorem roots_difference_is_one :
  is_arithmetic_progression roots →
  ∃ (r₁ r₂ r₃ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₃ ∈ roots ∧
  r₁ < r₂ ∧ r₂ < r₃ ∧ r₃ - r₁ = 1 :=
sorry

end NUMINAMATH_CALUDE_roots_difference_is_one_l696_69613


namespace NUMINAMATH_CALUDE_circle_on_parabola_passes_through_focus_l696_69669

/-- A circle with center on the parabola y^2 = 8x and tangent to x + 2 = 0 passes through (2, 0) -/
theorem circle_on_parabola_passes_through_focus (c : ℝ × ℝ) (r : ℝ) :
  c.2^2 = 8 * c.1 →  -- center is on the parabola y^2 = 8x
  r = c.1 + 2 →      -- circle is tangent to x + 2 = 0
  (c.1 - 2)^2 + c.2^2 = r^2  -- point (2, 0) is on the circle
  := by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_passes_through_focus_l696_69669


namespace NUMINAMATH_CALUDE_quadratic_roots_integrality_l696_69661

theorem quadratic_roots_integrality (q : ℤ) :
  (q > 0 → ∃ (p : ℤ), ∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) ∧
  (q < 0 → ¬∃ (p : ℤ), ∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_integrality_l696_69661


namespace NUMINAMATH_CALUDE_equation_solutions_l696_69639

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - 8 = x + 4 ∧ x = 6) ∧
  (∃ x : ℚ, 1 - 3 * (x + 1) = 2 * (1 - 0.5 * x) ∧ x = -2) ∧
  (∃ x : ℚ, (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 ∧ x = -20) ∧
  (∃ y : ℚ, (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l696_69639


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_sum_zero_implies_extremes_sum_zero_l696_69627

theorem sum_and_reciprocal_sum_zero_implies_extremes_sum_zero
  (a b c d : ℝ)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (h_sum : a + b + c + d = 0)
  (h_reciprocal_sum : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_sum_zero_implies_extremes_sum_zero_l696_69627


namespace NUMINAMATH_CALUDE_jills_gifts_and_charity_l696_69670

/-- Calculates the amount Jill uses for gifts and charitable causes --/
def gifts_and_charity (net_salary : ℚ) : ℚ :=
  let discretionary_income := (1 / 5) * net_salary
  let vacation_fund := (30 / 100) * discretionary_income
  let savings := (20 / 100) * discretionary_income
  let eating_out := (35 / 100) * discretionary_income
  discretionary_income - (vacation_fund + savings + eating_out)

/-- Theorem stating that Jill uses $99 for gifts and charitable causes --/
theorem jills_gifts_and_charity :
  gifts_and_charity 3300 = 99 := by
  sorry

end NUMINAMATH_CALUDE_jills_gifts_and_charity_l696_69670


namespace NUMINAMATH_CALUDE_difference_of_squares_equals_cube_l696_69626

theorem difference_of_squares_equals_cube (r : ℕ+) :
  ∃ m n : ℤ, m^2 - n^2 = (r : ℤ)^3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_equals_cube_l696_69626


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l696_69623

theorem right_triangle_shorter_leg :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →
  c = 65 →
  a ≤ b →
  a = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l696_69623


namespace NUMINAMATH_CALUDE_line_passes_through_point_l696_69624

/-- The line equation kx - y - 3k + 3 = 0 passes through the point (3,3) for all values of k. -/
theorem line_passes_through_point :
  ∀ (k : ℝ), (3 : ℝ) * k - 3 - 3 * k + 3 = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l696_69624


namespace NUMINAMATH_CALUDE_sum_of_valid_m_is_three_l696_69681

-- Define the linear function
def linear_function (m : ℤ) (x : ℝ) : ℝ := (4 - m) * x - 3

-- Define the fractional equation
def fractional_equation (m : ℤ) (z : ℤ) : Prop :=
  m / (z - 1 : ℝ) - 2 = 3 / (1 - z : ℝ)

-- Main theorem
theorem sum_of_valid_m_is_three :
  ∃ (S : Finset ℤ),
    (∀ m ∈ S,
      (∀ x y : ℝ, x < y → linear_function m x < linear_function m y) ∧
      (∃ z : ℤ, z > 1 ∧ fractional_equation m z)) ∧
    (∀ m : ℤ,
      (∀ x y : ℝ, x < y → linear_function m x < linear_function m y) ∧
      (∃ z : ℤ, z > 1 ∧ fractional_equation m z) →
      m ∈ S) ∧
    (S.sum id = 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_valid_m_is_three_l696_69681


namespace NUMINAMATH_CALUDE_train_theorem_l696_69638

def train_problem (initial : ℕ) 
  (stop1_off stop1_on : ℕ)
  (stop2_off stop2_on stop2_first_off : ℕ)
  (stop3_off stop3_on stop3_first_off : ℕ)
  (stop4_off stop4_on stop4_second_off : ℕ)
  (stop5_off stop5_on : ℕ) : Prop :=
  let after_stop1 := initial - stop1_off + stop1_on
  let after_stop2 := after_stop1 - stop2_off + stop2_on - stop2_first_off
  let after_stop3 := after_stop2 - stop3_off + stop3_on - stop3_first_off
  let after_stop4 := after_stop3 - stop4_off + stop4_on - stop4_second_off
  let final := after_stop4 - stop5_off + stop5_on
  final = 26

theorem train_theorem : train_problem 48 13 5 9 10 2 7 4 3 16 7 5 8 15 := by
  sorry

end NUMINAMATH_CALUDE_train_theorem_l696_69638


namespace NUMINAMATH_CALUDE_unused_sector_angle_l696_69622

/-- Given a circular piece of cardboard with radius 18 cm, from which a sector is removed
    to form a cone with radius 15 cm and volume 1350π cubic centimeters,
    the measure of the angle of the unused sector is 60°. -/
theorem unused_sector_angle (r_cardboard : ℝ) (r_cone : ℝ) (v_cone : ℝ) :
  r_cardboard = 18 →
  r_cone = 15 →
  v_cone = 1350 * Real.pi →
  ∃ (angle : ℝ),
    angle = 60 ∧
    angle = 360 - (2 * r_cone * Real.pi) / (2 * r_cardboard * Real.pi) * 360 :=
by sorry

end NUMINAMATH_CALUDE_unused_sector_angle_l696_69622


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l696_69616

/-- A rhombus with side length 65 and shorter diagonal 72 has a longer diagonal of 108 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) : 
  side = 65 → shorter_diag = 72 → longer_diag = 108 → 
  side^2 = (shorter_diag / 2)^2 + (longer_diag / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l696_69616


namespace NUMINAMATH_CALUDE_discount_profit_calculation_l696_69652

/-- Given a discount percentage and an original profit percentage,
    calculate the new profit percentage after applying the discount. -/
def profit_after_discount (discount : ℝ) (original_profit : ℝ) : ℝ :=
  let original_price := 1 + original_profit
  let discounted_price := original_price * (1 - discount)
  (discounted_price - 1) * 100

/-- Theorem stating that a 5% discount on an item with 50% original profit
    results in a 42.5% profit. -/
theorem discount_profit_calculation :
  profit_after_discount 0.05 0.5 = 42.5 := by sorry

end NUMINAMATH_CALUDE_discount_profit_calculation_l696_69652


namespace NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_change_l696_69612

theorem revenue_change_after_price_and_quantity_change 
  (P Q : ℝ) (P_new Q_new R R_new : ℝ) 
  (h1 : P_new = 0.8 * P) 
  (h2 : Q_new = 1.6 * Q) 
  (h3 : R = P * Q) 
  (h4 : R_new = P_new * Q_new) : 
  R_new = 1.28 * R := by sorry

end NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_change_l696_69612


namespace NUMINAMATH_CALUDE_min_disks_to_cover_l696_69630

/-- Represents a disk in 2D space -/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a covering of a disk by smaller disks -/
def DiskCovering (large : Disk) (small : List Disk) : Prop :=
  ∀ p : ℝ × ℝ, (p.1 - large.center.1)^2 + (p.2 - large.center.2)^2 ≤ large.radius^2 →
    ∃ d ∈ small, (p.1 - d.center.1)^2 + (p.2 - d.center.2)^2 ≤ d.radius^2

/-- The theorem stating that 7 is the minimum number of smaller disks needed -/
theorem min_disks_to_cover (large : Disk) (small : List Disk) :
  large.radius = 1 →
  (∀ d ∈ small, d.radius = 1/2) →
  DiskCovering large small →
  small.length ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_min_disks_to_cover_l696_69630


namespace NUMINAMATH_CALUDE_vector_addition_l696_69676

theorem vector_addition : 
  let v1 : Fin 2 → ℝ := ![- 5, 3]
  let v2 : Fin 2 → ℝ := ![7, -6]
  v1 + v2 = ![2, -3] :=
by sorry

end NUMINAMATH_CALUDE_vector_addition_l696_69676


namespace NUMINAMATH_CALUDE_percentage_error_calculation_l696_69673

theorem percentage_error_calculation : 
  let correct_operation (x : ℝ) := 3 * x
  let incorrect_operation (x : ℝ) := x / 5
  let error (x : ℝ) := correct_operation x - incorrect_operation x
  let percentage_error (x : ℝ) := (error x / correct_operation x) * 100
  ∀ x : ℝ, x ≠ 0 → percentage_error x = (14 / 15) * 100 := by
sorry

end NUMINAMATH_CALUDE_percentage_error_calculation_l696_69673


namespace NUMINAMATH_CALUDE_last_two_digits_product_l696_69649

/-- Given an integer n, returns the tens digit of n. -/
def tens_digit (n : ℤ) : ℤ := (n / 10) % 10

/-- Given an integer n, returns the units digit of n. -/
def units_digit (n : ℤ) : ℤ := n % 10

/-- Theorem: For any integer n divisible by 6 with the sum of its last two digits being 15,
    the product of its last two digits is either 56 or 54. -/
theorem last_two_digits_product (n : ℤ) 
  (div_by_6 : n % 6 = 0)
  (sum_15 : tens_digit n + units_digit n = 15) :
  tens_digit n * units_digit n = 56 ∨ tens_digit n * units_digit n = 54 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l696_69649


namespace NUMINAMATH_CALUDE_textbook_distribution_is_four_l696_69648

/-- The number of ways to distribute 8 identical textbooks between the classroom and students,
    given that at least 2 books must be in the classroom and at least 3 books must be with students. -/
def textbook_distribution : ℕ :=
  let total_books : ℕ := 8
  let min_classroom : ℕ := 2
  let min_students : ℕ := 3
  let valid_distributions := List.range (total_books + 1)
    |>.filter (λ classroom_books => 
      classroom_books ≥ min_classroom ∧ 
      (total_books - classroom_books) ≥ min_students)
  valid_distributions.length

/-- Proof that the number of valid distributions is 4 -/
theorem textbook_distribution_is_four : textbook_distribution = 4 := by
  sorry

end NUMINAMATH_CALUDE_textbook_distribution_is_four_l696_69648


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l696_69610

/-- A four-digit number -/
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- Each digit is less than 7 -/
def DigitsLessThan7 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d < 7

/-- The number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

/-- The main theorem -/
theorem unique_four_digit_square (N : ℕ) : 
  FourDigitNumber N ∧ 
  DigitsLessThan7 N ∧ 
  IsPerfectSquare N ∧ 
  IsPerfectSquare (N + 3333) → 
  N = 1156 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l696_69610


namespace NUMINAMATH_CALUDE_divisors_of_power_difference_l696_69611

theorem divisors_of_power_difference (n : ℕ) :
  n = 11^60 - 17^24 →
  ∃ (d : ℕ), d ≥ 120 ∧ (∀ (x : ℕ), x ∣ n → x > 0 → x ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_power_difference_l696_69611


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l696_69655

theorem gasoline_price_increase (original_price original_quantity : ℝ) 
  (h1 : original_price > 0) (h2 : original_quantity > 0) : 
  let new_quantity := original_quantity * (1 - 0.1)
  let new_total_cost := original_price * original_quantity * (1 + 0.08)
  let price_increase_factor := new_total_cost / (new_quantity * original_price)
  price_increase_factor = 1.2 := by sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l696_69655


namespace NUMINAMATH_CALUDE_vat_volume_l696_69632

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The volume of juice in each glass (in pints) -/
def volume_per_glass : ℕ := 30

/-- Theorem: The total volume of orange juice in the vat is 150 pints -/
theorem vat_volume : num_glasses * volume_per_glass = 150 := by
  sorry

end NUMINAMATH_CALUDE_vat_volume_l696_69632


namespace NUMINAMATH_CALUDE_etienne_money_greater_by_10_percent_l696_69679

-- Define the exchange rate
def exchange_rate : ℝ := 1.2

-- Define Diana's amount in dollars
def diana_dollars : ℝ := 600

-- Define Etienne's amount in euros
def etienne_euros : ℝ := 450

-- Theorem to prove
theorem etienne_money_greater_by_10_percent :
  (etienne_euros * exchange_rate - diana_dollars) / diana_dollars = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_etienne_money_greater_by_10_percent_l696_69679


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l696_69625

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Stirling number of the second kind: number of ways to partition n objects into k non-empty subsets -/
def stirling2 (n k : ℕ) : ℕ := sorry

theorem distribute_five_balls_three_boxes : 
  distribute_balls 5 3 = 41 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l696_69625


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l696_69668

theorem defective_shipped_percentage
  (total_units : ℕ)
  (defective_rate : ℚ)
  (shipped_rate : ℚ)
  (h1 : defective_rate = 5 / 100)
  (h2 : shipped_rate = 4 / 100) :
  (defective_rate * shipped_rate) * 100 = 0.2 := by
sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l696_69668


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l696_69651

theorem infinite_geometric_series_ratio (a S : ℝ) (h1 : a = 400) (h2 : S = 2500) :
  let r := 1 - a / S
  r = 21 / 25 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l696_69651


namespace NUMINAMATH_CALUDE_pet_insurance_coverage_calculation_l696_69646

/-- Calculates the amount covered by pet insurance for a cat's visit -/
def pet_insurance_coverage (
  doctor_visit_cost : ℝ
  ) (health_insurance_rate : ℝ
  ) (cat_visit_cost : ℝ
  ) (total_out_of_pocket : ℝ
  ) : ℝ :=
  cat_visit_cost - (total_out_of_pocket - (doctor_visit_cost * (1 - health_insurance_rate)))

theorem pet_insurance_coverage_calculation :
  pet_insurance_coverage 300 0.75 120 135 = 60 := by
  sorry

end NUMINAMATH_CALUDE_pet_insurance_coverage_calculation_l696_69646


namespace NUMINAMATH_CALUDE_bus_time_calculation_l696_69695

def wake_up_time : ℕ := 6 * 60 + 45
def bus_departure_time : ℕ := 7 * 60 + 15
def class_duration : ℕ := 45
def num_classes : ℕ := 7
def lunch_duration : ℕ := 20
def science_lab_duration : ℕ := 60
def additional_time : ℕ := 90
def arrival_time : ℕ := 15 * 60 + 50

def total_school_time : ℕ := 
  num_classes * class_duration + lunch_duration + science_lab_duration + additional_time

def total_away_time : ℕ := arrival_time - bus_departure_time

theorem bus_time_calculation : 
  total_away_time - total_school_time = 30 := by sorry

end NUMINAMATH_CALUDE_bus_time_calculation_l696_69695


namespace NUMINAMATH_CALUDE_inequality_proof_l696_69677

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_prod : a * b * c * d = 1) 
  (h_ineq : a + b + c + d > a/b + b/c + c/d + d/a) : 
  a + b + c + d < b/a + c/b + d/c + a/d := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l696_69677


namespace NUMINAMATH_CALUDE_nathan_blanket_warmth_l696_69659

theorem nathan_blanket_warmth (total_blankets : ℕ) (warmth_per_blanket : ℕ) (fraction_used : ℚ) : 
  total_blankets = 14 → 
  warmth_per_blanket = 3 → 
  fraction_used = 1/2 →
  (↑total_blankets * fraction_used : ℚ).floor * warmth_per_blanket = 21 := by
sorry

end NUMINAMATH_CALUDE_nathan_blanket_warmth_l696_69659


namespace NUMINAMATH_CALUDE_rachel_total_steps_l696_69619

/-- Represents a landmark with its stair information -/
structure Landmark where
  name : String
  flightsUp : Nat
  flightsDown : Nat
  stepsPerFlight : Nat

/-- Calculates the total steps for a single landmark -/
def stepsForLandmark (l : Landmark) : Nat :=
  (l.flightsUp + l.flightsDown) * l.stepsPerFlight

/-- The list of landmarks Rachel visited -/
def landmarks : List Landmark := [
  { name := "Eiffel Tower", flightsUp := 347, flightsDown := 216, stepsPerFlight := 10 },
  { name := "Notre-Dame Cathedral", flightsUp := 178, flightsDown := 165, stepsPerFlight := 12 },
  { name := "Leaning Tower of Pisa", flightsUp := 294, flightsDown := 172, stepsPerFlight := 8 },
  { name := "Colosseum", flightsUp := 122, flightsDown := 93, stepsPerFlight := 15 },
  { name := "Sagrada Familia", flightsUp := 267, flightsDown := 251, stepsPerFlight := 11 },
  { name := "Park Güell", flightsUp := 134, flightsDown := 104, stepsPerFlight := 9 }
]

/-- Calculates the total steps for all landmarks -/
def totalSteps : Nat :=
  landmarks.map stepsForLandmark |>.sum

theorem rachel_total_steps :
  totalSteps = 24539 := by
  sorry

end NUMINAMATH_CALUDE_rachel_total_steps_l696_69619


namespace NUMINAMATH_CALUDE_min_roots_count_l696_69609

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem min_roots_count
  (f : ℝ → ℝ)
  (h1 : is_symmetric_about f 2)
  (h2 : is_symmetric_about f 7)
  (h3 : f 0 = 0) :
  ∃ N : ℕ, N ≥ 401 ∧
  (∀ m : ℕ, (∃ S : Finset ℝ, S.card = m ∧
    (∀ x ∈ S, -1000 ≤ x ∧ x ≤ 1000 ∧ f x = 0)) →
    m ≤ N) :=
  sorry

end NUMINAMATH_CALUDE_min_roots_count_l696_69609


namespace NUMINAMATH_CALUDE_extreme_value_condition_l696_69618

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- State the theorem
theorem extreme_value_condition (a b : ℝ) :
  (f a b 1 = 4) ∧ (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1) →
  a * b = -27 ∨ a * b = -2 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l696_69618


namespace NUMINAMATH_CALUDE_gcd_1515_600_l696_69682

theorem gcd_1515_600 : Nat.gcd 1515 600 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1515_600_l696_69682


namespace NUMINAMATH_CALUDE_sum_of_fractions_l696_69675

theorem sum_of_fractions : (1 : ℚ) / 3 + 2 / 9 + 1 / 6 = 13 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l696_69675


namespace NUMINAMATH_CALUDE_nine_left_second_hour_l696_69657

/-- Represents the flow of people in a store over two hours -/
structure StoreTraffic where
  first_hour_in : ℕ
  first_hour_out : ℕ
  second_hour_in : ℕ
  final_count : ℕ

/-- Calculates the number of people who left during the second hour -/
def second_hour_out (st : StoreTraffic) : ℕ :=
  st.first_hour_in - st.first_hour_out + st.second_hour_in - st.final_count

/-- Theorem stating that 9 people left during the second hour given the problem conditions -/
theorem nine_left_second_hour (st : StoreTraffic) 
    (h1 : st.first_hour_in = 94)
    (h2 : st.first_hour_out = 27)
    (h3 : st.second_hour_in = 18)
    (h4 : st.final_count = 76) : 
  second_hour_out st = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_left_second_hour_l696_69657


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l696_69637

theorem sum_of_coefficients (a b x y : ℝ) : 
  (x = 3 ∧ y = -2) → 
  (a * x + b * y = 2 ∧ b * x + a * y = -3) → 
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l696_69637


namespace NUMINAMATH_CALUDE_hcf_problem_l696_69650

theorem hcf_problem (a b : ℕ+) (h1 : Nat.lcm a b % 11 = 0) 
  (h2 : Nat.lcm a b % 12 = 0) (h3 : max a b = 480) : Nat.gcd a b = 40 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l696_69650


namespace NUMINAMATH_CALUDE_inequality_solutions_l696_69603

theorem inequality_solutions :
  (∀ x : ℝ, 5 * x + 3 < 11 + x ↔ x < 2) ∧
  (∀ x : ℝ, 2 * x + 1 < 3 * x + 3 ∧ (x + 1) / 2 ≤ (1 - x) / 6 + 1 ↔ -2 < x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l696_69603


namespace NUMINAMATH_CALUDE_prime_square_minus_one_remainder_l696_69635

theorem prime_square_minus_one_remainder (p : ℕ) (hp : Nat.Prime p) :
  ∃ r ∈ ({0, 3, 8} : Set ℕ), (p^2 - 1) % 12 = r :=
sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_remainder_l696_69635


namespace NUMINAMATH_CALUDE_customer_coin_count_l696_69699

/-- Represents the quantity of each type of coin turned in by the customer --/
structure CoinQuantities where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat
  halfDollars : Nat
  oneDollarCoins : Nat
  twoDollarCoins : Nat
  australianFiftyCentCoins : Nat
  mexicanOnePesoCoins : Nat

/-- Calculates the total number of coins turned in --/
def totalCoins (coins : CoinQuantities) : Nat :=
  coins.pennies +
  coins.nickels +
  coins.dimes +
  coins.quarters +
  coins.halfDollars +
  coins.oneDollarCoins +
  coins.twoDollarCoins +
  coins.australianFiftyCentCoins +
  coins.mexicanOnePesoCoins

/-- Theorem: The total number of coins turned in by the customer is 159 --/
theorem customer_coin_count :
  ∃ (coins : CoinQuantities),
    coins.pennies = 38 ∧
    coins.nickels = 27 ∧
    coins.dimes = 19 ∧
    coins.quarters = 24 ∧
    coins.halfDollars = 13 ∧
    coins.oneDollarCoins = 17 ∧
    coins.twoDollarCoins = 5 ∧
    coins.australianFiftyCentCoins = 4 ∧
    coins.mexicanOnePesoCoins = 12 ∧
    totalCoins coins = 159 := by
  sorry

end NUMINAMATH_CALUDE_customer_coin_count_l696_69699


namespace NUMINAMATH_CALUDE_line_direction_vector_l696_69672

def point := ℝ × ℝ

-- Define the two points on the line
def p1 : point := (-3, 4)
def p2 : point := (2, -1)

-- Define the direction vector type
def direction_vector := ℝ × ℝ

-- Function to calculate the direction vector between two points
def calc_direction_vector (p q : point) : direction_vector :=
  (q.1 - p.1, q.2 - p.2)

-- Function to scale a vector
def scale_vector (v : direction_vector) (s : ℝ) : direction_vector :=
  (s * v.1, s * v.2)

-- Theorem statement
theorem line_direction_vector : 
  ∃ (a : ℝ), calc_direction_vector p1 p2 = scale_vector (a, 2) (-5/2) ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l696_69672


namespace NUMINAMATH_CALUDE_steve_bench_wood_length_l696_69631

/-- Calculates the total length of wood needed for Steve's bench. -/
theorem steve_bench_wood_length : 
  let long_pieces : ℕ := 6
  let long_length : ℕ := 4
  let short_pieces : ℕ := 2
  let short_length : ℕ := 2
  long_pieces * long_length + short_pieces * short_length = 28 :=
by sorry

end NUMINAMATH_CALUDE_steve_bench_wood_length_l696_69631


namespace NUMINAMATH_CALUDE_triangle_problem_l696_69658

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b^2 = a^2 + c^2 - Real.sqrt 3 * a * c →
  B = π/6 ∧
  Real.sqrt 3 / 2 < Real.cos A + Real.sin C ∧ 
  Real.cos A + Real.sin C < 3/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l696_69658


namespace NUMINAMATH_CALUDE_roller_coaster_friends_l696_69660

theorem roller_coaster_friends (tickets_per_ride : ℕ) (total_tickets : ℕ) (num_friends : ℕ) : 
  tickets_per_ride = 6 → total_tickets = 48 → num_friends * tickets_per_ride = total_tickets → num_friends = 8 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_friends_l696_69660


namespace NUMINAMATH_CALUDE_bernoulli_misplacement_problem_l696_69697

def D : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | (n + 1) => n * (D n + D (n - 1))

theorem bernoulli_misplacement_problem :
  (D 4 : ℚ) / 24 = 3 / 8 ∧
  (6 * D 5 : ℚ) / 720 = 11 / 30 := by
  sorry


end NUMINAMATH_CALUDE_bernoulli_misplacement_problem_l696_69697


namespace NUMINAMATH_CALUDE_triangle_inequalities_l696_69607

-- Define the points and lengths
variables (P Q R S : ℝ × ℝ) (a b c : ℝ)

-- Define the conditions
def collinear (P Q R S : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ, 0 < t₁ ∧ t₁ < t₂ ∧ t₂ < t₃ ∧
  Q = P + t₁ • (S - P) ∧
  R = P + t₂ • (S - P) ∧
  S = P + t₃ • (S - P)

def segment_lengths (P Q R S : ℝ × ℝ) (a b c : ℝ) : Prop :=
  dist P Q = a ∧ dist P R = b ∧ dist P S = c

def can_form_triangle (a b c : ℝ) : Prop :=
  a + (b - a) > c - b ∧
  (b - a) + (c - b) > a ∧
  a + (c - b) > b - a

-- State the theorem
theorem triangle_inequalities
  (h_collinear : collinear P Q R S)
  (h_lengths : segment_lengths P Q R S a b c)
  (h_triangle : can_form_triangle a b c) :
  a < c / 3 ∧ b < 2 * a + c :=
sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l696_69607


namespace NUMINAMATH_CALUDE_derivative_log2_l696_69617

open Real

theorem derivative_log2 (x : ℝ) (h : x > 0) :
  deriv (fun x => log x / log 2) x = 1 / (x * log 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_log2_l696_69617


namespace NUMINAMATH_CALUDE_lesser_number_l696_69600

theorem lesser_number (a b : ℝ) (h1 : a + b = 55) (h2 : a - b = 7) : min a b = 24 := by
  sorry

end NUMINAMATH_CALUDE_lesser_number_l696_69600


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l696_69664

def original_price_A : ℝ := 500
def original_price_B : ℝ := 600
def original_price_C : ℝ := 700

def first_discount_rate : ℝ := 0.15
def second_discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.07
def flat_discount_B : ℝ := 200

def total_original_price : ℝ := original_price_A + original_price_B + original_price_C

noncomputable def final_price_A : ℝ := 
  (original_price_A * (1 - first_discount_rate) * (1 - second_discount_rate)) * (1 + sales_tax_rate)

noncomputable def final_price_B : ℝ := 
  (original_price_B * (1 - first_discount_rate) * (1 - second_discount_rate)) - flat_discount_B

noncomputable def final_price_C : ℝ := 
  (original_price_C * (1 - second_discount_rate)) * (1 + sales_tax_rate)

noncomputable def total_final_price : ℝ := final_price_A + final_price_B + final_price_C

noncomputable def percentage_reduction : ℝ := 
  (total_original_price - total_final_price) / total_original_price * 100

theorem price_reduction_theorem : 
  25.42 ≤ percentage_reduction ∧ percentage_reduction < 25.43 :=
sorry

end NUMINAMATH_CALUDE_price_reduction_theorem_l696_69664


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l696_69693

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y : ℝ, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l696_69693


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l696_69694

/-- Given a parabola y = ax^2 - a (a ≠ 0) intersecting a line y = kx at points 
    with x-coordinates summing to less than 0, prove that the line y = ax + k 
    passes through the first and fourth quadrants. -/
theorem parabola_line_intersection (a k : ℝ) (h_a : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, a * x₁^2 - a = k * x₁ ∧ 
               a * x₂^2 - a = k * x₂ ∧ 
               x₁ + x₂ < 0) →
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = a * x + k) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ y = a * x + k) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l696_69694


namespace NUMINAMATH_CALUDE_fraction_inequality_l696_69688

theorem fraction_inequality (x : ℝ) (h : x ≠ -5) :
  (x - 2) / (x + 5) ≥ 0 ↔ x < -5 ∨ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l696_69688


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_five_l696_69636

/-- Given two nonconstant geometric sequences with different common ratios,
    if a specific condition is met, prove that the sum of their common ratios is 5. -/
theorem sum_of_common_ratios_is_five
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r)
  (h : k * p^2 - k * r^2 = 5 * (k * p - k * r)) (hk : k ≠ 0) :
  p + r = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_five_l696_69636


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l696_69698

theorem geometric_arithmetic_sequence_problem (x y z : ℝ) 
  (h1 : (12 * y)^2 = 9 * x * 15 * z)  -- 9x, 12y, 15z form a geometric sequence
  (h2 : 2 / y = 1 / x + 1 / z)        -- 1/x, 1/y, 1/z form an arithmetic sequence
  : x / z + z / x = 34 / 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l696_69698


namespace NUMINAMATH_CALUDE_large_monkey_doll_cost_l696_69645

/-- The cost of a large monkey doll in dollars -/
def large_monkey_cost : ℝ := 6

/-- The total amount spent in dollars -/
def total_spent : ℝ := 300

/-- The number of additional dolls that can be bought if choosing small monkey dolls instead of large monkey dolls -/
def additional_small_dolls : ℕ := 25

/-- The number of fewer dolls that can be bought if choosing elephant dolls instead of large monkey dolls -/
def fewer_elephant_dolls : ℕ := 15

theorem large_monkey_doll_cost :
  (total_spent / (large_monkey_cost - 2) = total_spent / large_monkey_cost + additional_small_dolls) ∧
  (total_spent / (large_monkey_cost + 1) = total_spent / large_monkey_cost - fewer_elephant_dolls) := by
  sorry

end NUMINAMATH_CALUDE_large_monkey_doll_cost_l696_69645


namespace NUMINAMATH_CALUDE_angle_triple_complement_l696_69614

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l696_69614


namespace NUMINAMATH_CALUDE_good_number_theorem_l696_69692

/-- A good number is a number of the form a + b√2 where a and b are integers -/
def GoodNumber (x : ℝ) : Prop :=
  ∃ (a b : ℤ), x = a + b * Real.sqrt 2

/-- Polynomial with good number coefficients -/
def GoodPolynomial (p : Polynomial ℝ) : Prop :=
  ∀ (i : ℕ), GoodNumber (p.coeff i)

theorem good_number_theorem (A B Q : Polynomial ℝ) 
  (hA : GoodPolynomial A)
  (hB : GoodPolynomial B)
  (hB0 : B.coeff 0 = 1)
  (hABQ : A = B * Q) :
  GoodPolynomial Q :=
sorry

end NUMINAMATH_CALUDE_good_number_theorem_l696_69692


namespace NUMINAMATH_CALUDE_quadratic_max_l696_69691

theorem quadratic_max (x : ℝ) : 
  ∃ (m : ℝ), (∀ y : ℝ, -y^2 - 8*y + 16 ≤ m) ∧ (-x^2 - 8*x + 16 = m) → x = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_l696_69691


namespace NUMINAMATH_CALUDE_factorial_10_mod_13_l696_69680

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_10_mod_13_l696_69680


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l696_69689

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 3
  (f 3 = 0 ∧ f (-1) = 0) ∧
  ∀ x : ℝ, f x = 0 → x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l696_69689


namespace NUMINAMATH_CALUDE_cannot_construct_configuration_l696_69671

/-- Represents a rhombus figure with two colors -/
structure ColoredRhombus where
  white_part : Set (ℝ × ℝ)
  gray_part : Set (ℝ × ℝ)
  is_rhombus : white_part ∪ gray_part = unit_rhombus
  no_overlap : white_part ∩ gray_part = ∅

/-- Represents a configuration of multiple rhombuses -/
def Configuration := Set (ColoredRhombus × (ℝ × ℝ))

/-- Rotates a point around the origin -/
def rotate (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Translates a point -/
def translate (v : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Applies rotation and translation to a ColoredRhombus -/
def transform (r : ColoredRhombus) (θ : ℝ) (v : ℝ × ℝ) : ColoredRhombus := sorry

/-- Checks if a configuration can be constructed from a given rhombus -/
def is_constructible (r : ColoredRhombus) (c : Configuration) : Prop := sorry

/-- The specific configuration that we claim is impossible to construct -/
def impossible_configuration : Configuration := sorry

/-- The main theorem stating that the impossible configuration cannot be constructed -/
theorem cannot_construct_configuration (r : ColoredRhombus) : 
  ¬(is_constructible r impossible_configuration) := by sorry

end NUMINAMATH_CALUDE_cannot_construct_configuration_l696_69671


namespace NUMINAMATH_CALUDE_fish_sample_properties_l696_69621

/-- Represents the mass categories of fish -/
inductive MassCategory
  | Mass1 : MassCategory
  | Mass2 : MassCategory
  | Mass3 : MassCategory
  | Mass4 : MassCategory

/-- Maps mass categories to their actual mass values -/
def massValue : MassCategory → Float
  | MassCategory.Mass1 => 1.0
  | MassCategory.Mass2 => 1.2
  | MassCategory.Mass3 => 1.5
  | MassCategory.Mass4 => 1.8

/-- Represents the frequency of each mass category -/
def frequency : MassCategory → Nat
  | MassCategory.Mass1 => 4
  | MassCategory.Mass2 => 5
  | MassCategory.Mass3 => 8
  | MassCategory.Mass4 => 3

/-- The total number of fish in the sample -/
def sampleSize : Nat := 20

/-- The number of marked fish recaptured -/
def markedRecaptured : Nat := 2

/-- The total number of fish recaptured -/
def totalRecaptured : Nat := 100

/-- Theorem stating the properties of the fish sample -/
theorem fish_sample_properties :
  (∃ median : Float, median = 1.5) ∧
  (∃ mean : Float, mean = 1.37) ∧
  (∃ totalMass : Float, totalMass = 1370) := by
  sorry

end NUMINAMATH_CALUDE_fish_sample_properties_l696_69621


namespace NUMINAMATH_CALUDE_tangent_length_is_six_l696_69620

/-- A circle passing through three points -/
structure Circle3Points where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- The length of the tangent from a point to a circle -/
def tangentLength (origin : ℝ × ℝ) (circle : Circle3Points) : ℝ :=
  sorry

/-- Theorem: The length of the tangent from the origin to the specific circle is 6 -/
theorem tangent_length_is_six : 
  let origin : ℝ × ℝ := (0, 0)
  let circle : Circle3Points := { 
    p1 := (2, 3),
    p2 := (4, 6),
    p3 := (6, 15)
  }
  tangentLength origin circle = 6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_is_six_l696_69620


namespace NUMINAMATH_CALUDE_find_m_l696_69696

/-- The value of log base 10 of 2, approximated to 4 decimal places -/
def log10_2 : ℝ := 0.3010

/-- Theorem stating that the positive integer m satisfying the given inequality is 155 -/
theorem find_m (m : ℕ) (hm_pos : m > 0) 
  (h_ineq : (10 : ℝ)^(m-1) < (2 : ℝ)^512 ∧ (2 : ℝ)^512 < (10 : ℝ)^m) : 
  m = 155 := by
  sorry

#check find_m

end NUMINAMATH_CALUDE_find_m_l696_69696


namespace NUMINAMATH_CALUDE_mercury_column_height_for_constant_center_of_gravity_l696_69644

/-- Proves that the height of the mercury column for which the center of gravity
    remains at a constant distance from the top of the tube at any temperature
    is approximately 0.106 meters. -/
theorem mercury_column_height_for_constant_center_of_gravity
  (tube_length : ℝ)
  (cross_section_area : ℝ)
  (glass_expansion_coeff : ℝ)
  (mercury_expansion_coeff : ℝ)
  (h : tube_length = 1)
  (h₁ : cross_section_area = 1e-4)
  (h₂ : glass_expansion_coeff = 1 / 38700)
  (h₃ : mercury_expansion_coeff = 1 / 5550) :
  ∃ (height : ℝ), abs (height - 0.106) < 0.001 ∧
  ∀ (t : ℝ),
    (tube_length * (1 + glass_expansion_coeff / 3 * t) -
     height / 2 * (1 + (mercury_expansion_coeff - 2 * glass_expansion_coeff / 3) * t)) =
    (tube_length - height / 2) :=
sorry

end NUMINAMATH_CALUDE_mercury_column_height_for_constant_center_of_gravity_l696_69644


namespace NUMINAMATH_CALUDE_no_natural_solution_l696_69663

theorem no_natural_solution : ¬∃ (x y : ℕ), 2 * x + 3 * y = 6 := by sorry

end NUMINAMATH_CALUDE_no_natural_solution_l696_69663


namespace NUMINAMATH_CALUDE_proposition_relationship_l696_69633

-- Define propositions as variables of type Prop
variable (A B C : Prop)

-- Define the relationships between propositions
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

def necessary_and_sufficient (P Q : Prop) : Prop :=
  (P ↔ Q)

def necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

-- State the theorem
theorem proposition_relationship :
  sufficient_not_necessary A B →
  necessary_and_sufficient B C →
  necessary_not_sufficient C A :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l696_69633


namespace NUMINAMATH_CALUDE_matrix_computation_l696_69647

theorem matrix_computation (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec ![1, 3] = ![2, 5])
  (h2 : N.mulVec ![-2, 4] = ![3, 1]) :
  N.mulVec ![3, 11] = ![7.4, 17.2] := by
sorry

end NUMINAMATH_CALUDE_matrix_computation_l696_69647


namespace NUMINAMATH_CALUDE_squirrel_travel_time_l696_69601

-- Define the speed of the squirrel in miles per hour
def speed : ℝ := 4

-- Define the distance to be traveled in miles
def distance : ℝ := 1

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℝ := 60

-- Theorem statement
theorem squirrel_travel_time :
  (distance / speed) * minutes_per_hour = 15 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_travel_time_l696_69601


namespace NUMINAMATH_CALUDE_sqrt_288_simplification_l696_69605

theorem sqrt_288_simplification : Real.sqrt 288 = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_288_simplification_l696_69605


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l696_69674

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  (a ≥ 3/2 ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l696_69674


namespace NUMINAMATH_CALUDE_eulerian_path_figures_l696_69654

-- Define a structure for our figures
structure Figure where
  has_eulerian_path : Bool
  all_vertices_even_degree : Bool
  num_odd_degree_vertices : Nat

-- Define our theorem
theorem eulerian_path_figures :
  let figureA : Figure := { has_eulerian_path := true, all_vertices_even_degree := true, num_odd_degree_vertices := 0 }
  let figureB : Figure := { has_eulerian_path := true, all_vertices_even_degree := false, num_odd_degree_vertices := 0 }
  let figureC : Figure := { has_eulerian_path := false, all_vertices_even_degree := false, num_odd_degree_vertices := 3 }
  let figureD : Figure := { has_eulerian_path := true, all_vertices_even_degree := true, num_odd_degree_vertices := 0 }
  ∀ (f : Figure),
    (f.all_vertices_even_degree ∨ f.num_odd_degree_vertices = 2) ↔ f.has_eulerian_path :=
by
  sorry


end NUMINAMATH_CALUDE_eulerian_path_figures_l696_69654


namespace NUMINAMATH_CALUDE_special_circle_equation_l696_69662

/-- A circle passing through two points with a specific sum of intercepts -/
structure SpecialCircle where
  -- The circle passes through (4,2) and (-2,-6)
  passes_through_1 : x^2 + y^2 + D*x + E*y + F = 0 → 4^2 + 2^2 + 4*D + 2*E + F = 0
  passes_through_2 : x^2 + y^2 + D*x + E*y + F = 0 → (-2)^2 + (-6)^2 + (-2)*D + (-6)*E + F = 0
  -- Sum of intercepts is -2
  sum_of_intercepts : D + E = 2

/-- The standard equation of the special circle -/
def standard_equation (c : SpecialCircle) : Prop :=
  ∃ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = 25

/-- Theorem stating that the given circle has the specified standard equation -/
theorem special_circle_equation (c : SpecialCircle) : standard_equation c :=
  sorry

end NUMINAMATH_CALUDE_special_circle_equation_l696_69662


namespace NUMINAMATH_CALUDE_constant_difference_of_equal_derivatives_l696_69643

theorem constant_difference_of_equal_derivatives 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h : ∀ x, deriv f x = deriv g x) : 
  ∃ C, ∀ x, f x - g x = C :=
sorry

end NUMINAMATH_CALUDE_constant_difference_of_equal_derivatives_l696_69643


namespace NUMINAMATH_CALUDE_vector_parallel_implies_m_equals_two_l696_69683

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem vector_parallel_implies_m_equals_two (m : ℝ) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (2, 1)
  parallel (a.1 - 2 * b.1, a.2 - 2 * b.2) b →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_m_equals_two_l696_69683


namespace NUMINAMATH_CALUDE_ab_ab2_a_inequality_l696_69667

theorem ab_ab2_a_inequality (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_ab_ab2_a_inequality_l696_69667


namespace NUMINAMATH_CALUDE_find_cd_l696_69684

def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (10 * c + d : ℚ) / 99

theorem find_cd (c d : ℕ) (h_c : c < 10) (h_d : d < 10) : 
  42 * (repeating_decimal c d - (1 + (10 * c + d : ℚ) / 100)) = 4/5 → 
  c = 1 ∧ d = 9 := by
sorry

end NUMINAMATH_CALUDE_find_cd_l696_69684


namespace NUMINAMATH_CALUDE_correct_dice_configuration_l696_69604

/-- Represents a die face with a number of dots -/
structure DieFace where
  dots : Nat
  h : dots ≥ 1 ∧ dots ≤ 6

/-- Represents the configuration of four dice -/
structure DiceConfiguration where
  faceA : DieFace
  faceB : DieFace
  faceC : DieFace
  faceD : DieFace

/-- Theorem stating the correct number of dots on each face -/
theorem correct_dice_configuration :
  ∃ (config : DiceConfiguration),
    config.faceA.dots = 3 ∧
    config.faceB.dots = 5 ∧
    config.faceC.dots = 6 ∧
    config.faceD.dots = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_dice_configuration_l696_69604


namespace NUMINAMATH_CALUDE_parallelepiped_properties_l696_69608

/-- Represents a parallelepiped with given properties -/
structure Parallelepiped where
  height : ℝ
  lateral_edge_projection : ℝ
  rhombus_area : ℝ
  rhombus_diagonal : ℝ

/-- Calculate the lateral surface area of the parallelepiped -/
def lateral_surface_area (p : Parallelepiped) : ℝ := sorry

/-- Calculate the volume of the parallelepiped -/
def volume (p : Parallelepiped) : ℝ := sorry

/-- Theorem stating the properties of the specific parallelepiped -/
theorem parallelepiped_properties (p : Parallelepiped) 
  (h1 : p.height = 12)
  (h2 : p.lateral_edge_projection = 5)
  (h3 : p.rhombus_area = 24)
  (h4 : p.rhombus_diagonal = 8) : 
  lateral_surface_area p = 260 ∧ volume p = 312 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_properties_l696_69608
