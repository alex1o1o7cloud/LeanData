import Mathlib

namespace NUMINAMATH_CALUDE_M_has_three_elements_l3190_319051

def M : Set ℝ :=
  {m | ∃ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    m = x / |x| + y / |y| + z / |z| + (x * y * z) / |x * y * z|}

theorem M_has_three_elements :
  ∃ a b c : ℝ, M = {a, b, c} :=
sorry

end NUMINAMATH_CALUDE_M_has_three_elements_l3190_319051


namespace NUMINAMATH_CALUDE_book_shelf_average_width_l3190_319024

theorem book_shelf_average_width :
  let book_widths : List ℝ := [5, 3/4, 1.5, 3.25, 4, 3, 7/2, 12]
  (book_widths.sum / book_widths.length : ℝ) = 4.125 := by
  sorry

end NUMINAMATH_CALUDE_book_shelf_average_width_l3190_319024


namespace NUMINAMATH_CALUDE_songs_leftover_l3190_319001

theorem songs_leftover (total_songs : ℕ) (num_playlists : ℕ) (h1 : total_songs = 372) (h2 : num_playlists = 9) :
  total_songs % num_playlists = 3 := by
  sorry

end NUMINAMATH_CALUDE_songs_leftover_l3190_319001


namespace NUMINAMATH_CALUDE_prime_iff_k_t_greater_n_div_4_l3190_319095

theorem prime_iff_k_t_greater_n_div_4 (n : ℕ) (k t : ℕ) : 
  Odd n → n > 3 →
  (∀ k' < k, ¬ ∃ m : ℕ, k' * n + 1 = m * m) →
  (∀ t' < t, ¬ ∃ m : ℕ, t' * n = m * m) →
  (∃ m : ℕ, k * n + 1 = m * m) →
  (∃ m : ℕ, t * n = m * m) →
  (Nat.Prime n ↔ (k > n / 4 ∧ t > n / 4)) :=
by sorry

end NUMINAMATH_CALUDE_prime_iff_k_t_greater_n_div_4_l3190_319095


namespace NUMINAMATH_CALUDE_inequality_proof_l3190_319071

theorem inequality_proof (h : Real.log (1/2) < 0) : (1/2)^3 < (1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3190_319071


namespace NUMINAMATH_CALUDE_three_digit_square_ending_l3190_319074

theorem three_digit_square_ending (n : ℕ) : 
  100 ≤ n ∧ n ≤ 999 ∧ n^2 % 1000 = n → n = 376 ∨ n = 625 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_square_ending_l3190_319074


namespace NUMINAMATH_CALUDE_selling_price_with_loss_l3190_319085

def cost_price : ℝ := 1800
def loss_percentage : ℝ := 10

theorem selling_price_with_loss (cp : ℝ) (lp : ℝ) : 
  cp * (1 - lp / 100) = 1620 :=
by sorry

end NUMINAMATH_CALUDE_selling_price_with_loss_l3190_319085


namespace NUMINAMATH_CALUDE_exists_society_with_subgroup_l3190_319028

/-- Definition of a society with n girls and m boys -/
structure Society :=
  (n : ℕ) -- number of girls
  (m : ℕ) -- number of boys

/-- Definition of a relationship between boys and girls in a society -/
def Knows (s : Society) := 
  Fin s.m → Fin s.n → Prop

/-- Definition of a subgroup with the required property -/
def HasSubgroup (s : Society) (knows : Knows s) : Prop :=
  ∃ (girls : Fin 5 → Fin s.n) (boys : Fin 5 → Fin s.m),
    (∀ i j, knows (boys i) (girls j)) ∨ 
    (∀ i j, ¬knows (boys i) (girls j))

/-- Main theorem: Existence of n₀ and m₀ satisfying the property -/
theorem exists_society_with_subgroup :
  ∃ (n₀ m₀ : ℕ), ∀ (s : Society),
    s.n = n₀ → s.m = m₀ → 
    ∀ (knows : Knows s), HasSubgroup s knows :=
sorry

end NUMINAMATH_CALUDE_exists_society_with_subgroup_l3190_319028


namespace NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l3190_319010

/-- The radius of the incircle of a triangle with sides 5, 12, and 13 units is 2 units. -/
theorem incircle_radius_of_special_triangle : 
  ∀ (a b c r : ℝ), 
  a = 5 → b = 12 → c = 13 →
  r = (a * b) / (a + b + c) →
  r = 2 := by sorry

end NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l3190_319010


namespace NUMINAMATH_CALUDE_binomial_prob_example_l3190_319093

/-- The probability mass function of a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- Theorem: For X ~ B(4, 1/3), P(X = 1) = 32/81 -/
theorem binomial_prob_example :
  let n : ℕ := 4
  let p : ℝ := 1/3
  let k : ℕ := 1
  binomial_pmf n p k = 32/81 := by
sorry

end NUMINAMATH_CALUDE_binomial_prob_example_l3190_319093


namespace NUMINAMATH_CALUDE_max_value_fraction_l3190_319066

theorem max_value_fraction (x y : ℝ) : 
  (2*x + 3*y + 4) / Real.sqrt (2*x^2 + 3*y^2 + 5) ≤ Real.sqrt 28 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3190_319066


namespace NUMINAMATH_CALUDE_polynomial_value_l3190_319067

theorem polynomial_value (m n : ℤ) (h : m - 2*n = 7) : 
  2023 - 2*m + 4*n = 2009 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l3190_319067


namespace NUMINAMATH_CALUDE_min_squares_6x7_l3190_319073

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- A tiling of a rectangle with squares -/
def Tiling (r : Rectangle) := List Square

/-- The area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ :=
  r.width * r.height

/-- The area of a square -/
def squareArea (s : Square) : ℕ :=
  s.side * s.side

/-- Check if a tiling is valid for a given rectangle -/
def isValidTiling (r : Rectangle) (t : Tiling r) : Prop :=
  (t.map squareArea).sum = rectangleArea r

/-- The main theorem -/
theorem min_squares_6x7 :
  ∃ (t : Tiling ⟨6, 7⟩), 
    isValidTiling ⟨6, 7⟩ t ∧ 
    t.length = 7 ∧ 
    (∀ (t' : Tiling ⟨6, 7⟩), isValidTiling ⟨6, 7⟩ t' → t'.length ≥ 7) :=
  sorry

end NUMINAMATH_CALUDE_min_squares_6x7_l3190_319073


namespace NUMINAMATH_CALUDE_teena_loe_distance_l3190_319059

theorem teena_loe_distance (teena_speed loe_speed : ℝ) (time : ℝ) (ahead_distance : ℝ) :
  teena_speed = 55 →
  loe_speed = 40 →
  time = 1.5 →
  ahead_distance = 15 →
  ∃ initial_distance : ℝ,
    initial_distance = (teena_speed * time - loe_speed * time - ahead_distance) ∧
    initial_distance = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_teena_loe_distance_l3190_319059


namespace NUMINAMATH_CALUDE_consecutive_right_triangle_iff_345_l3190_319097

/-- A right-angled triangle with consecutive integer side lengths -/
structure ConsecutiveRightTriangle where
  n : ℕ
  n_pos : 0 < n
  is_right : (n + 1)^2 + n^2 = (n + 2)^2

/-- The property of having sides 3, 4, and 5 -/
def has_sides_345 (t : ConsecutiveRightTriangle) : Prop :=
  t.n = 3

theorem consecutive_right_triangle_iff_345 :
  ∀ t : ConsecutiveRightTriangle, has_sides_345 t ↔ True :=
sorry

end NUMINAMATH_CALUDE_consecutive_right_triangle_iff_345_l3190_319097


namespace NUMINAMATH_CALUDE_airplane_cost_is_428_l3190_319099

/-- The cost of an airplane, given the initial amount and change received. -/
def airplane_cost (initial_amount change : ℚ) : ℚ :=
  initial_amount - change

/-- Theorem stating that the cost of the airplane is $4.28 -/
theorem airplane_cost_is_428 :
  airplane_cost 5 0.72 = 4.28 := by
  sorry

end NUMINAMATH_CALUDE_airplane_cost_is_428_l3190_319099


namespace NUMINAMATH_CALUDE_two_sessions_scientific_notation_l3190_319046

theorem two_sessions_scientific_notation :
  78200000000 = 7.82 * (10 : ℝ)^10 := by sorry

end NUMINAMATH_CALUDE_two_sessions_scientific_notation_l3190_319046


namespace NUMINAMATH_CALUDE_courtyard_length_l3190_319030

/-- Prove that the length of a rectangular courtyard is 70 meters -/
theorem courtyard_length (width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) :
  width = 16.5 ∧ 
  num_stones = 231 ∧ 
  stone_length = 2.5 ∧ 
  stone_width = 2 →
  (num_stones * stone_length * stone_width) / width = 70 := by
sorry

end NUMINAMATH_CALUDE_courtyard_length_l3190_319030


namespace NUMINAMATH_CALUDE_zero_is_rational_l3190_319002

/-- A number is rational if it can be expressed as the quotient of two integers with a non-zero denominator -/
def IsRational (x : ℚ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

/-- Theorem: Zero is a rational number -/
theorem zero_is_rational : IsRational 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_is_rational_l3190_319002


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l3190_319035

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 5

def prob_treasure : ℚ := 1/4
def prob_traps : ℚ := 1/12
def prob_neither : ℚ := 2/3

theorem pirate_treasure_probability :
  (num_islands.choose num_treasure_islands) * 
  (prob_treasure ^ num_treasure_islands) * 
  (prob_neither ^ (num_islands - num_treasure_islands)) = 7/432 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l3190_319035


namespace NUMINAMATH_CALUDE_break_even_circus_production_l3190_319064

/-- Calculates the number of sold-out performances needed to break even for a circus production -/
def break_even_performances (overhead : ℕ) (production_cost : ℕ) (revenue : ℕ) : ℕ :=
  let total_cost (x : ℕ) := overhead + production_cost * x
  let total_revenue (x : ℕ) := revenue * x
  (overhead / (revenue - production_cost) : ℕ)

/-- Proves that 9 sold-out performances are needed to break even given the specific costs and revenue -/
theorem break_even_circus_production :
  break_even_performances 81000 7000 16000 = 9 := by
  sorry

end NUMINAMATH_CALUDE_break_even_circus_production_l3190_319064


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3190_319048

/-- Represents a 2D vector --/
structure Vector2D where
  x : ℚ
  y : ℚ

/-- Represents a parametric line in 2D --/
structure ParametricLine where
  origin : Vector2D
  direction : Vector2D

/-- The first line --/
def line1 : ParametricLine :=
  { origin := { x := 2, y := 3 },
    direction := { x := -1, y := 4 } }

/-- The second line --/
def line2 : ParametricLine :=
  { origin := { x := -1, y := 6 },
    direction := { x := 3, y := 5 } }

/-- The intersection point of two parametric lines --/
def intersection (l1 l2 : ParametricLine) : Vector2D :=
  { x := 28 / 17, y := 75 / 17 }

/-- Theorem stating that the intersection of line1 and line2 is (28/17, 75/17) --/
theorem intersection_of_lines :
  intersection line1 line2 = { x := 28 / 17, y := 75 / 17 } := by
  sorry

#check intersection_of_lines

end NUMINAMATH_CALUDE_intersection_of_lines_l3190_319048


namespace NUMINAMATH_CALUDE_remainder_problem_l3190_319029

theorem remainder_problem : (7 * 10^20 + 2^20 + 5) % 9 = 7 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l3190_319029


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l3190_319078

theorem rectangular_plot_breadth (length breadth : ℝ) 
  (h1 : length * breadth = 15 * breadth) 
  (h2 : length - breadth = 10) : 
  breadth = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l3190_319078


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l3190_319005

theorem junk_mail_distribution (total_mail : ℕ) (num_blocks : ℕ) 
  (h1 : total_mail = 192) 
  (h2 : num_blocks = 4) :
  total_mail / num_blocks = 48 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l3190_319005


namespace NUMINAMATH_CALUDE_gcd_288_123_l3190_319043

theorem gcd_288_123 : Nat.gcd 288 123 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_288_123_l3190_319043


namespace NUMINAMATH_CALUDE_fractional_equation_root_l3190_319003

theorem fractional_equation_root (x m : ℝ) : 
  ((x - 5) / (x + 2) = m / (x + 2) ∧ x + 2 ≠ 0) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l3190_319003


namespace NUMINAMATH_CALUDE_square_difference_given_product_and_sum_l3190_319011

theorem square_difference_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 16) (h2 : p + q = 10) : (p - q)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_product_and_sum_l3190_319011


namespace NUMINAMATH_CALUDE_inequality_of_cube_roots_l3190_319098

theorem inequality_of_cube_roots (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.rpow ((a / (b + c))^2) (1/3) + Real.rpow ((b / (c + a))^2) (1/3) + Real.rpow ((c / (a + b))^2) (1/3) ≥ 3 / Real.rpow 4 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_cube_roots_l3190_319098


namespace NUMINAMATH_CALUDE_quotient_calculation_l3190_319090

theorem quotient_calculation (divisor dividend remainder quotient : ℕ) : 
  divisor = 17 → dividend = 76 → remainder = 8 → quotient = 4 →
  dividend = divisor * quotient + remainder :=
by
  sorry

end NUMINAMATH_CALUDE_quotient_calculation_l3190_319090


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l3190_319089

theorem mystery_book_shelves :
  let books_per_shelf : ℕ := 4
  let picture_book_shelves : ℕ := 3
  let total_books : ℕ := 32
  let mystery_book_shelves : ℕ := (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf
  mystery_book_shelves = 5 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l3190_319089


namespace NUMINAMATH_CALUDE_missing_number_in_set_l3190_319008

theorem missing_number_in_set (x : ℝ) (a : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (a + 255 + 511 + 1023 + x) / 5 = 398.2 →
  a = 128 := by
sorry

end NUMINAMATH_CALUDE_missing_number_in_set_l3190_319008


namespace NUMINAMATH_CALUDE_maximize_product_l3190_319020

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 50) :
  x^4 * y^3 ≤ (200/7)^4 * (150/7)^3 ∧
  (x = 200/7 ∧ y = 150/7) → x^4 * y^3 = (200/7)^4 * (150/7)^3 :=
by sorry

end NUMINAMATH_CALUDE_maximize_product_l3190_319020


namespace NUMINAMATH_CALUDE_min_product_sum_l3190_319012

/-- Triangle ABC with side lengths a, b, c and height h from A to BC -/
structure Triangle :=
  (a b c h : ℝ)
  (positive_a : 0 < a)
  (positive_b : 0 < b)
  (positive_c : 0 < c)
  (positive_h : 0 < h)

/-- The problem statement -/
theorem min_product_sum (t : Triangle) (h1 : t.c = 10) (h2 : t.h = 3) :
  let min_product := Real.sqrt ((t.c^2 * t.h^2) / 4)
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = min_product ∧ a + b = 4 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_min_product_sum_l3190_319012


namespace NUMINAMATH_CALUDE_tan_20_plus_4sin_20_equals_sqrt_3_l3190_319014

theorem tan_20_plus_4sin_20_equals_sqrt_3 :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_20_plus_4sin_20_equals_sqrt_3_l3190_319014


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l3190_319019

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => (17/3) * x^3 - 38 * x^2 - (101/3) * x + 185/3
  (q 1 = -5) ∧ (q 2 = 1) ∧ (q 3 = -1) ∧ (q 4 = 23) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l3190_319019


namespace NUMINAMATH_CALUDE_two_number_difference_l3190_319094

theorem two_number_difference (a b : ℕ) : 
  a + b = 20460 → 
  b % 12 = 0 → 
  a = b / 10 → 
  b - a = 17314 := by sorry

end NUMINAMATH_CALUDE_two_number_difference_l3190_319094


namespace NUMINAMATH_CALUDE_book_sale_revenue_l3190_319018

theorem book_sale_revenue (total_books : ℕ) (price_per_book : ℚ) : 
  (3 * total_books = 108) →  -- Condition: 1/3 of total books is 36
  (price_per_book = 7/2) →   -- Price per book is $3.50
  (2 * total_books / 3 * price_per_book = 252) := by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l3190_319018


namespace NUMINAMATH_CALUDE_train_length_calculation_l3190_319052

/-- Prove that given a train traveling at a certain speed that crosses a bridge in a given time, 
    and the total length of the bridge and train is known, we can calculate the length of the train. -/
theorem train_length_calculation 
  (train_speed : ℝ) 
  (crossing_time : ℝ) 
  (total_length : ℝ) 
  (h1 : train_speed = 45) -- km/hr
  (h2 : crossing_time = 30 / 3600) -- 30 seconds converted to hours
  (h3 : total_length = 195) -- meters
  : ∃ (train_length : ℝ), train_length = 180 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l3190_319052


namespace NUMINAMATH_CALUDE_cube_difference_l3190_319017

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) :
  a^3 - b^3 = 108 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l3190_319017


namespace NUMINAMATH_CALUDE_extreme_points_when_a_is_one_extreme_points_condition_l3190_319034

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x + 1 else a*x

-- Theorem 1: When a = 1, f(x) has exactly two extreme points
theorem extreme_points_when_a_is_one :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧
  (∀ (x : ℝ), (∃ (ε : ℝ), ε > 0 ∧
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f 1 y ≤ f 1 x) ∨
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f 1 y ≥ f 1 x)) ↔ (x = x1 ∨ x = x2)) :=
sorry

-- Theorem 2: f(x) has exactly two extreme points iff 0 < a < 2
theorem extreme_points_condition :
  ∀ (a : ℝ), (∃! (x1 x2 : ℝ), x1 ≠ x2 ∧
  (∀ (x : ℝ), (∃ (ε : ℝ), ε > 0 ∧
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f a y ≤ f a x) ∨
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f a y ≥ f a x)) ↔ (x = x1 ∨ x = x2)))
  ↔ (0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_extreme_points_when_a_is_one_extreme_points_condition_l3190_319034


namespace NUMINAMATH_CALUDE_total_legs_calculation_l3190_319049

/-- The number of legs a spider has -/
def spider_legs : ℕ := 8

/-- The number of legs a centipede has -/
def centipede_legs : ℕ := 100

/-- The number of spiders in the room -/
def num_spiders : ℕ := 4

/-- The number of centipedes in the room -/
def num_centipedes : ℕ := 3

/-- The total number of legs for all spiders and centipedes -/
def total_legs : ℕ := num_spiders * spider_legs + num_centipedes * centipede_legs

theorem total_legs_calculation :
  total_legs = 332 := by sorry

end NUMINAMATH_CALUDE_total_legs_calculation_l3190_319049


namespace NUMINAMATH_CALUDE_increasing_function_condition_l3190_319087

/-- A function f is increasing on ℝ if for all x y, x < y implies f x < f y -/
def IncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_condition (f : ℝ → ℝ) (h : IncreasingOn f) :
  ∀ a b : ℝ, a + b < 0 ↔ f a + f b < f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l3190_319087


namespace NUMINAMATH_CALUDE_max_digit_sum_l3190_319096

theorem max_digit_sum (a b c x y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (1000 * (1 : ℚ) / (100 * a + 10 * b + c) = y) →  -- 0.abc = 1/y
  (0 < y ∧ y ≤ 50) →  -- y is an integer and 0 < y ≤ 50
  (1000 * (1 : ℚ) / (100 * y + 10 * y + y) = x) →  -- 0.yyy = 1/x
  (0 < x ∧ x ≤ 9) →  -- x is an integer and 0 < x ≤ 9
  (∀ a' b' c' : ℕ, 
    (a' < 10 ∧ b' < 10 ∧ c' < 10) →
    (∃ x' y' : ℕ, 
      (1000 * (1 : ℚ) / (100 * a' + 10 * b' + c') = y') ∧
      (0 < y' ∧ y' ≤ 50) ∧
      (1000 * (1 : ℚ) / (100 * y' + 10 * y' + y') = x') ∧
      (0 < x' ∧ x' ≤ 9)) →
    (a + b + c ≥ a' + b' + c')) →
  a + b + c = 8 := by
sorry

end NUMINAMATH_CALUDE_max_digit_sum_l3190_319096


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l3190_319055

theorem cubic_equation_real_root (k : ℝ) (hk : k ≠ 0) :
  ∃ x : ℝ, x^3 + k*x + k^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l3190_319055


namespace NUMINAMATH_CALUDE_min_values_ab_l3190_319054

theorem min_values_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ 1/x + 2/y < 9) = False ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x^2 + y^2 < 1/5) = False :=
by sorry

end NUMINAMATH_CALUDE_min_values_ab_l3190_319054


namespace NUMINAMATH_CALUDE_ratio_equality_l3190_319025

theorem ratio_equality (x : ℝ) : (1 : ℝ) / 3 = (5 : ℝ) / (3 * x) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3190_319025


namespace NUMINAMATH_CALUDE_evaluate_expression_l3190_319065

theorem evaluate_expression (a b : ℚ) (ha : a = 3) (hb : b = 2) :
  (a^4 + b^4) / (a^2 - a*b + b^2) = 97 / 7 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3190_319065


namespace NUMINAMATH_CALUDE_shaded_area_between_squares_l3190_319084

/-- The area of the shaded region in a figure with two concentric squares -/
theorem shaded_area_between_squares (large_side small_side : ℝ) 
  (h1 : large_side = 10) 
  (h2 : small_side = 4) 
  (h3 : large_side > small_side) : 
  (large_side^2 - small_side^2) / 4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_squares_l3190_319084


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3190_319036

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a = 1 → a + b = 1) ↔ (∃ (a b : ℝ), a = 1 ∧ a + b ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3190_319036


namespace NUMINAMATH_CALUDE_initial_birds_on_fence_l3190_319088

theorem initial_birds_on_fence (initial_birds : ℕ) (initial_storks : ℕ) : 
  (initial_birds + 5 = initial_storks + 4 + 3) → initial_birds = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_on_fence_l3190_319088


namespace NUMINAMATH_CALUDE_sara_oranges_l3190_319013

theorem sara_oranges (joan_oranges : ℕ) (total_oranges : ℕ) (h1 : joan_oranges = 37) (h2 : total_oranges = 47) :
  total_oranges - joan_oranges = 10 := by
  sorry

end NUMINAMATH_CALUDE_sara_oranges_l3190_319013


namespace NUMINAMATH_CALUDE_mean_exercise_days_jenkins_class_l3190_319063

/-- Represents the exercise data for a group of students -/
structure ExerciseData where
  students : List (Nat × Float)

/-- Calculates the mean number of days exercised -/
def calculateMean (data : ExerciseData) : Float :=
  let totalDays := data.students.foldl (fun acc (n, d) => acc + n.toFloat * d) 0
  let totalStudents := data.students.foldl (fun acc (n, _) => acc + n) 0
  totalDays / totalStudents.toFloat

/-- Rounds a float to the nearest hundredth -/
def roundToHundredth (x : Float) : Float :=
  (x * 100).round / 100

theorem mean_exercise_days_jenkins_class :
  let jenkinsData : ExerciseData := {
    students := [
      (2, 0.5),
      (4, 1),
      (5, 3),
      (3, 4),
      (7, 6),
      (2, 7)
    ]
  }
  roundToHundredth (calculateMean jenkinsData) = 3.83 := by
  sorry

end NUMINAMATH_CALUDE_mean_exercise_days_jenkins_class_l3190_319063


namespace NUMINAMATH_CALUDE_october_price_l3190_319072

/-- The price of a mobile phone after n months, given an initial price and monthly decrease rate. -/
def price (initial_price : ℝ) (decrease_rate : ℝ) (months : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ months

/-- Theorem: The price of a mobile phone in October is a · (0.97)^9 yuan, given that its initial price
    in January was a yuan and it decreases by 3% every month. -/
theorem october_price (a : ℝ) : price a 0.03 9 = a * 0.97^9 := by
  sorry

end NUMINAMATH_CALUDE_october_price_l3190_319072


namespace NUMINAMATH_CALUDE_absolute_value_non_negative_l3190_319086

theorem absolute_value_non_negative (a : ℝ) : ¬(|a| < 0) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_non_negative_l3190_319086


namespace NUMINAMATH_CALUDE_base_seven_43210_equals_10738_l3190_319023

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_43210_equals_10738 :
  base_seven_to_ten [0, 1, 2, 3, 4] = 10738 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_43210_equals_10738_l3190_319023


namespace NUMINAMATH_CALUDE_geometric_proportion_conclusion_l3190_319045

/-- A set of four real numbers forms a geometric proportion in any order -/
def GeometricProportionAnyOrder (a b c d : ℝ) : Prop :=
  (a / b = c / d ∧ a / b = d / c) ∧
  (a / c = b / d ∧ a / c = d / b) ∧
  (a / d = b / c ∧ a / d = c / b)

/-- The conclusion about four numbers forming a geometric proportion in any order -/
theorem geometric_proportion_conclusion (a b c d : ℝ) 
  (h : GeometricProportionAnyOrder a b c d) :
  (a = b ∧ b = c ∧ c = d) ∨ 
  (|a| = |b| ∧ |b| = |c| ∧ |c| = |d| ∧ 
   ((a > 0 ∧ b > 0 ∧ c < 0 ∧ d < 0) ∨
    (a > 0 ∧ c > 0 ∧ b < 0 ∧ d < 0) ∨
    (a > 0 ∧ d > 0 ∧ b < 0 ∧ c < 0) ∨
    (b > 0 ∧ c > 0 ∧ a < 0 ∧ d < 0) ∨
    (b > 0 ∧ d > 0 ∧ a < 0 ∧ c < 0) ∨
    (c > 0 ∧ d > 0 ∧ a < 0 ∧ b < 0))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_proportion_conclusion_l3190_319045


namespace NUMINAMATH_CALUDE_machine_value_depletion_rate_l3190_319007

/-- Proves that the annual value depletion rate is 0.1 for a machine with given initial and final values over 2 years -/
theorem machine_value_depletion_rate
  (initial_value : ℝ)
  (final_value : ℝ)
  (time_period : ℝ)
  (h1 : initial_value = 900)
  (h2 : final_value = 729)
  (h3 : time_period = 2)
  : ∃ (rate : ℝ), rate = 0.1 ∧ final_value = initial_value * (1 - rate) ^ time_period :=
sorry

end NUMINAMATH_CALUDE_machine_value_depletion_rate_l3190_319007


namespace NUMINAMATH_CALUDE_count_pairs_satisfying_inequality_l3190_319091

def S : Finset ℤ := {-3, -2, -1, 0, 1, 2, 3}

theorem count_pairs_satisfying_inequality :
  (Finset.filter (fun p : ℤ × ℤ => 
    p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2 ∧ p.2^2 < (5/4) * p.1^2)
    (S.product S)).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_satisfying_inequality_l3190_319091


namespace NUMINAMATH_CALUDE_alex_remaining_money_l3190_319070

theorem alex_remaining_money (weekly_income : ℝ) (tax_rate : ℝ) (water_bill : ℝ) 
  (tithe_rate : ℝ) (groceries : ℝ) (transportation : ℝ) :
  weekly_income = 900 →
  tax_rate = 0.15 →
  water_bill = 75 →
  tithe_rate = 0.20 →
  groceries = 150 →
  transportation = 50 →
  weekly_income - (tax_rate * weekly_income) - water_bill - (tithe_rate * weekly_income) - 
    groceries - transportation = 310 := by
  sorry

end NUMINAMATH_CALUDE_alex_remaining_money_l3190_319070


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l3190_319082

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k ∈ Finset.range 10 → k.succ ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ∈ Finset.range 10 → k.succ ∣ m) → n ≤ m) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l3190_319082


namespace NUMINAMATH_CALUDE_construction_labor_problem_l3190_319056

theorem construction_labor_problem (total_hired : ℕ) (operator_pay laborer_pay : ℚ) (total_payroll : ℚ) :
  total_hired = 35 →
  operator_pay = 140 →
  laborer_pay = 90 →
  total_payroll = 3950 →
  ∃ (operators laborers : ℕ),
    operators + laborers = total_hired ∧
    operators * operator_pay + laborers * laborer_pay = total_payroll ∧
    laborers = 19 := by
  sorry

end NUMINAMATH_CALUDE_construction_labor_problem_l3190_319056


namespace NUMINAMATH_CALUDE_grandma_inheritance_l3190_319083

theorem grandma_inheritance (total : ℕ) (shelby_share : ℕ) (remaining_grandchildren : ℕ) :
  total = 124600 →
  shelby_share = total / 2 →
  remaining_grandchildren = 10 →
  (total - shelby_share) / remaining_grandchildren = 6230 :=
by sorry

end NUMINAMATH_CALUDE_grandma_inheritance_l3190_319083


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l3190_319053

theorem rectangle_formation_count (h : ℕ) (v : ℕ) : h = 5 → v = 6 → Nat.choose h 2 * Nat.choose v 2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l3190_319053


namespace NUMINAMATH_CALUDE_algebra_test_average_l3190_319060

theorem algebra_test_average (total_average : ℝ) (male_count : ℕ) (female_average : ℝ) (female_count : ℕ) :
  total_average = 90 →
  male_count = 8 →
  female_average = 92 →
  female_count = 28 →
  (total_average * (male_count + female_count) - female_average * female_count) / male_count = 83 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_average_l3190_319060


namespace NUMINAMATH_CALUDE_chloe_dimes_needed_l3190_319006

/-- Represents the minimum number of dimes needed to purchase a hoodie -/
def min_dimes_needed (hoodie_cost : ℚ) (ten_dollar_bills : ℕ) (quarters : ℕ) (one_dollar_coins : ℕ) : ℕ :=
  let current_money : ℚ := 10 * ten_dollar_bills + 0.25 * quarters + one_dollar_coins
  ⌈(hoodie_cost - current_money) / 0.1⌉₊

/-- Theorem stating that Chloe needs 0 additional dimes to buy the hoodie -/
theorem chloe_dimes_needed : 
  min_dimes_needed 45.50 4 10 3 = 0 := by
  sorry

#eval min_dimes_needed 45.50 4 10 3

end NUMINAMATH_CALUDE_chloe_dimes_needed_l3190_319006


namespace NUMINAMATH_CALUDE_denise_spending_l3190_319079

/-- Represents the types of dishes available --/
inductive Dish
| Simple
| Meat
| Fish

/-- Represents the types of vitamins available --/
inductive Vitamin
| Milk
| Fruit
| Special

/-- Returns the price of a dish --/
def dishPrice (d : Dish) : ℕ :=
  match d with
  | Dish.Simple => 7
  | Dish.Meat => 11
  | Dish.Fish => 14

/-- Returns the price of a vitamin --/
def vitaminPrice (v : Vitamin) : ℕ :=
  match v with
  | Vitamin.Milk => 6
  | Vitamin.Fruit => 7
  | Vitamin.Special => 9

/-- Calculates the total price of a meal (dish + vitamin) --/
def mealPrice (d : Dish) (v : Vitamin) : ℕ :=
  dishPrice d + vitaminPrice v

/-- Represents a person's meal choice --/
structure MealChoice where
  dish : Dish
  vitamin : Vitamin

/-- The main theorem to prove --/
theorem denise_spending (julio_choice denise_choice : MealChoice)
  (h : mealPrice julio_choice.dish julio_choice.vitamin = 
       mealPrice denise_choice.dish denise_choice.vitamin + 6) :
  mealPrice denise_choice.dish denise_choice.vitamin = 14 ∨
  mealPrice denise_choice.dish denise_choice.vitamin = 17 := by
  sorry


end NUMINAMATH_CALUDE_denise_spending_l3190_319079


namespace NUMINAMATH_CALUDE_roots_have_different_signs_l3190_319038

/-- Given two quadratic polynomials with specific properties, prove that the roots of the first polynomial have different signs -/
theorem roots_have_different_signs (a b c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2*b*x₁ + c = 0 ∧ a * x₂^2 + 2*b*x₂ + c = 0) →  -- First polynomial has two distinct roots
  (∀ x : ℝ, a^2 * x^2 + 2*b^2*x + c^2 ≠ 0) →                                        -- Second polynomial has no roots
  ∃ x₁ x₂ : ℝ, x₁ * x₂ < 0 ∧ a * x₁^2 + 2*b*x₁ + c = 0 ∧ a * x₂^2 + 2*b*x₂ + c = 0  -- Roots of first polynomial have different signs
:= by sorry

end NUMINAMATH_CALUDE_roots_have_different_signs_l3190_319038


namespace NUMINAMATH_CALUDE_translation_theorem_l3190_319077

/-- Represents a point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- Translates a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

theorem translation_theorem :
  let M : Point := { x := -4, y := 3 }
  let M1 := translateHorizontal M (-3)
  let M2 := translateVertical M1 2
  M2 = { x := -7, y := 5 } := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l3190_319077


namespace NUMINAMATH_CALUDE_x_value_l3190_319016

theorem x_value (x : ℝ) (h : (x / 3) / 3 = 9 / (x / 3)) : x = 9 * Real.sqrt 3 ∨ x = -9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3190_319016


namespace NUMINAMATH_CALUDE_possible_m_values_l3190_319044

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem possible_m_values : 
  {m : ℝ | B m ⊆ A} = {-1/2, 0, 1/3} := by sorry

end NUMINAMATH_CALUDE_possible_m_values_l3190_319044


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3190_319033

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 6) * x^2 - 8 * x + 9 = 0) ↔ (a ≤ 70 / 9 ∧ a ≠ 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3190_319033


namespace NUMINAMATH_CALUDE_polygon_area_is_144_l3190_319004

/-- A polygon with perpendicular adjacent sides -/
structure PerpendicularPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  area : ℝ

/-- Our specific polygon -/
def our_polygon : PerpendicularPolygon where
  sides := 36
  side_length := 2
  perimeter := 72
  area := 144

theorem polygon_area_is_144 (p : PerpendicularPolygon) 
  (h1 : p.sides = 36) 
  (h2 : p.perimeter = 72) 
  (h3 : p.side_length = p.perimeter / p.sides) : 
  p.area = 144 := by
  sorry

#check polygon_area_is_144

end NUMINAMATH_CALUDE_polygon_area_is_144_l3190_319004


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_1234_l3190_319075

def last_two_digits (n : ℕ) : ℕ := n % 100

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_8_pow_1234 : tens_digit (8^1234) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_1234_l3190_319075


namespace NUMINAMATH_CALUDE_distance_inequality_l3190_319047

theorem distance_inequality (a : ℝ) :
  (abs (a - 1) < 3) → (-2 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_distance_inequality_l3190_319047


namespace NUMINAMATH_CALUDE_abc_remainder_mod_7_l3190_319050

theorem abc_remainder_mod_7 (a b c : ℕ) 
  (h_a : a < 7) (h_b : b < 7) (h_c : c < 7)
  (h1 : (a + 2*b + 3*c) % 7 = 0)
  (h2 : (2*a + 3*b + c) % 7 = 2)
  (h3 : (3*a + b + 2*c) % 7 = 4) :
  (a * b * c) % 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_remainder_mod_7_l3190_319050


namespace NUMINAMATH_CALUDE_daria_credit_card_debt_l3190_319042

/-- Calculates the discounted price of an item --/
def discountedPrice (price : ℚ) (discountPercent : ℚ) : ℚ :=
  price * (1 - discountPercent / 100)

/-- Represents Daria's furniture purchases --/
structure Purchases where
  couch : ℚ
  couchDiscount : ℚ
  table : ℚ
  tableDiscount : ℚ
  lamp : ℚ
  rug : ℚ
  rugDiscount : ℚ
  bookshelf : ℚ
  bookshelfDiscount : ℚ

/-- Calculates the total cost of purchases after discounts --/
def totalCost (p : Purchases) : ℚ :=
  discountedPrice p.couch p.couchDiscount +
  discountedPrice p.table p.tableDiscount +
  p.lamp +
  discountedPrice p.rug p.rugDiscount +
  discountedPrice p.bookshelf p.bookshelfDiscount

/-- Theorem: Daria owes $610 on her credit card before interest --/
theorem daria_credit_card_debt (p : Purchases) (savings : ℚ) :
  p.couch = 750 →
  p.couchDiscount = 10 →
  p.table = 100 →
  p.tableDiscount = 5 →
  p.lamp = 50 →
  p.rug = 200 →
  p.rugDiscount = 15 →
  p.bookshelf = 150 →
  p.bookshelfDiscount = 20 →
  savings = 500 →
  totalCost p - savings = 610 := by
  sorry


end NUMINAMATH_CALUDE_daria_credit_card_debt_l3190_319042


namespace NUMINAMATH_CALUDE_four_in_B_iff_m_in_range_B_subset_A_iff_m_in_range_l3190_319021

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Part 1: 4 ∈ B iff m ∈ [5/2, 3]
theorem four_in_B_iff_m_in_range (m : ℝ) : 
  (4 ∈ B m) ↔ (5/2 ≤ m ∧ m ≤ 3) :=
sorry

-- Part 2: B ⊂ A iff m ∈ (-∞, 3]
theorem B_subset_A_iff_m_in_range (m : ℝ) :
  (B m ⊂ A) ↔ (m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_four_in_B_iff_m_in_range_B_subset_A_iff_m_in_range_l3190_319021


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_k_value_l3190_319076

theorem quadratic_roots_imply_k_value (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 8 * x + k = 0 ↔ x = -2 + Real.sqrt 6 ∨ x = -2 - Real.sqrt 6) →
  k = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_k_value_l3190_319076


namespace NUMINAMATH_CALUDE_ratio_extended_points_l3190_319022

-- Define the triangle ABC
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b ∧ b < c ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the points B₁, A₁, C₂, B₂
def ExtendedPoints (a b c : ℝ) : Prop :=
  ∃ (A B C A₁ B₁ C₂ B₂ : ℝ × ℝ),
    Triangle a b c ∧
    dist B C = a ∧
    dist C A = b ∧
    dist A B = c ∧
    dist B B₁ = c ∧
    dist A A₁ = c ∧
    dist C C₂ = a ∧
    dist B B₂ = a

-- State the theorem
theorem ratio_extended_points (a b c : ℝ) :
  Triangle a b c → ExtendedPoints a b c →
  ∃ (A₁ B₁ C₂ B₂ : ℝ × ℝ), dist A₁ B₁ / dist C₂ B₂ = c / a :=
sorry

end NUMINAMATH_CALUDE_ratio_extended_points_l3190_319022


namespace NUMINAMATH_CALUDE_bettys_herb_garden_l3190_319027

theorem bettys_herb_garden (basil oregano : ℕ) : 
  oregano = 2 * basil + 2 →
  basil + oregano = 17 →
  basil = 5 := by sorry

end NUMINAMATH_CALUDE_bettys_herb_garden_l3190_319027


namespace NUMINAMATH_CALUDE_greg_sharon_harvest_difference_l3190_319037

theorem greg_sharon_harvest_difference :
  let greg_harvest : Real := 0.4
  let sharon_harvest : Real := 0.1
  greg_harvest - sharon_harvest = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_greg_sharon_harvest_difference_l3190_319037


namespace NUMINAMATH_CALUDE_max_fraction_sum_l3190_319000

theorem max_fraction_sum (a b c d : ℕ+) (h1 : a + c = 20) (h2 : (a : ℚ) / b + (c : ℚ) / d < 1) :
  (∀ a' b' c' d' : ℕ+, a' + c' = 20 → (a' : ℚ) / b' + (c' : ℚ) / d' < 1 → (a : ℚ) / b + (c : ℚ) / d ≤ (a' : ℚ) / b' + (c' : ℚ) / d') →
  (a : ℚ) / b + (c : ℚ) / d = 20 / 21 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l3190_319000


namespace NUMINAMATH_CALUDE_tangent_power_equality_l3190_319009

open Complex

theorem tangent_power_equality (α : ℝ) (n : ℕ) :
  ((1 + I * Real.tan α) / (1 - I * Real.tan α)) ^ n = 
  (1 + I * Real.tan (n * α)) / (1 - I * Real.tan (n * α)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_power_equality_l3190_319009


namespace NUMINAMATH_CALUDE_triangle_side_length_l3190_319081

/-- Represents a triangle with sides a, b, c and heights ha, hb, hc -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ

/-- Theorem: In a triangle with sides AC = 6 cm and BC = 3 cm, 
    if the half-sum of heights to AC and BC equals the height to AB, 
    then AB = 4 cm -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.b = 6)
  (h2 : t.c = 3)
  (h3 : (t.ha + t.hb) / 2 = t.hc) : 
  t.a = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3190_319081


namespace NUMINAMATH_CALUDE_total_letters_in_names_l3190_319069

theorem total_letters_in_names (jonathan_first : ℕ) (jonathan_surname : ℕ) 
  (sister_first : ℕ) (sister_surname : ℕ) 
  (h1 : jonathan_first = 8) (h2 : jonathan_surname = 10) 
  (h3 : sister_first = 5) (h4 : sister_surname = 10) : 
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_letters_in_names_l3190_319069


namespace NUMINAMATH_CALUDE_triple_transmission_more_reliable_l3190_319040

/-- Represents a transmission channel with error probabilities α and β -/
structure TransmissionChannel where
  α : Real
  β : Real
  α_pos : 0 < α
  α_lt_one : α < 1
  β_pos : 0 < β
  β_lt_one : β < 1

/-- Probability of decoding as 0 using single transmission when sending 0 -/
def singleTransmissionProb (channel : TransmissionChannel) : Real :=
  1 - channel.α

/-- Probability of decoding as 0 using triple transmission when sending 0 -/
def tripleTransmissionProb (channel : TransmissionChannel) : Real :=
  3 * channel.α * (1 - channel.α)^2 + (1 - channel.α)^3

/-- Theorem stating that triple transmission is more reliable than single transmission for decoding 0 when α < 0.5 -/
theorem triple_transmission_more_reliable (channel : TransmissionChannel) 
    (h : channel.α < 0.5) : 
    singleTransmissionProb channel < tripleTransmissionProb channel := by
  sorry

end NUMINAMATH_CALUDE_triple_transmission_more_reliable_l3190_319040


namespace NUMINAMATH_CALUDE_inverse_trig_inequality_l3190_319041

theorem inverse_trig_inequality : 
  Real.arctan (-5/4) < Real.arcsin (-2/5) ∧ Real.arcsin (-2/5) < Real.arccos (-3/4) := by
  sorry

end NUMINAMATH_CALUDE_inverse_trig_inequality_l3190_319041


namespace NUMINAMATH_CALUDE_digit_subtraction_reaches_zero_l3190_319062

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The sequence obtained by repeatedly subtracting the sum of digits -/
def digitSubtractionSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => digitSubtractionSequence n k - sumOfDigits (digitSubtractionSequence n k)

/-- The theorem stating that the digit subtraction sequence always reaches 0 -/
theorem digit_subtraction_reaches_zero (n : ℕ) :
  ∃ k : ℕ, digitSubtractionSequence n k = 0 :=
sorry

end NUMINAMATH_CALUDE_digit_subtraction_reaches_zero_l3190_319062


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3190_319032

theorem min_value_quadratic (x y : ℝ) : 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3190_319032


namespace NUMINAMATH_CALUDE_point_N_coordinates_l3190_319080

-- Define the points and lines
def M : ℝ × ℝ := (0, -1)
def N : ℝ × ℝ := (2, 3)

def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the perpendicular property
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem point_N_coordinates :
  line1 N.1 N.2 ∧
  perpendicular 
    ((N.2 - M.2) / (N.1 - M.1)) 
    (-(1 / 2)) →
  N = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l3190_319080


namespace NUMINAMATH_CALUDE_spinner_area_l3190_319031

theorem spinner_area (r : ℝ) (p_win : ℝ) (p_bonus : ℝ) 
  (h_r : r = 15)
  (h_p_win : p_win = 1/3)
  (h_p_bonus : p_bonus = 1/6) :
  p_win * π * r^2 + p_bonus * π * r^2 = 112.5 * π := by
  sorry

end NUMINAMATH_CALUDE_spinner_area_l3190_319031


namespace NUMINAMATH_CALUDE_preimage_of_point_l3190_319026

def f (x y : ℝ) : ℝ × ℝ := (2*x + y, x*y)

theorem preimage_of_point (x₁ y₁ x₂ y₂ : ℝ) :
  f x₁ y₁ = (1/6, -1/6) ∧ f x₂ y₂ = (1/6, -1/6) ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  ((x₁ = 1/4 ∧ y₁ = -1/3) ∨ (x₁ = -1/3 ∧ y₁ = 7/6)) ∧
  ((x₂ = 1/4 ∧ y₂ = -1/3) ∨ (x₂ = -1/3 ∧ y₂ = 7/6)) :=
sorry

end NUMINAMATH_CALUDE_preimage_of_point_l3190_319026


namespace NUMINAMATH_CALUDE_unique_solution_l3190_319068

theorem unique_solution : ∃! (x y : ℝ), 
  (2 * x + 3 * y = (7 - 2 * x) + (7 - 3 * y)) ∧ 
  (x - 2 * y = (x - 2) + (2 * y - 2)) ∧
  x = 2 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3190_319068


namespace NUMINAMATH_CALUDE_polynomial_equality_l3190_319057

theorem polynomial_equality (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 4 * x^4 + 5 * x^3 + 10 * x) = 
    (12 * x^5 + 6 * x^4 + 28 * x^3 + 30 * x^2 + 3 * x + 2)) →
  (∀ x, q x = -2 * x^6 + 12 * x^5 + 2 * x^4 + 23 * x^3 + 30 * x^2 - 7 * x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3190_319057


namespace NUMINAMATH_CALUDE_second_group_size_l3190_319015

/-- The number of persons in the first group -/
def first_group : ℕ := 78

/-- The number of days the first group works -/
def first_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_hours : ℕ := 5

/-- The number of days the second group works -/
def second_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_hours : ℕ := 6

/-- The total man-hours required to complete the job -/
def total_man_hours : ℕ := first_group * first_days * first_hours

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (second_days * second_hours)

theorem second_group_size :
  second_group = 130 := by sorry

end NUMINAMATH_CALUDE_second_group_size_l3190_319015


namespace NUMINAMATH_CALUDE_smallest_prime_ten_less_square_l3190_319061

theorem smallest_prime_ten_less_square : ∃ (n : ℕ), 
  (∀ m : ℕ, m < n → ¬(Nat.Prime m ∧ ∃ k : ℕ, m = k^2 - 10)) ∧ 
  (Nat.Prime n ∧ ∃ k : ℕ, n = k^2 - 10) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_ten_less_square_l3190_319061


namespace NUMINAMATH_CALUDE_min_value_theorem_l3190_319039

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  ∃ (m : ℝ), m = 24 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a * b * c = 64 → x + 4*y + 8*z ≤ a + 4*b + 8*c ∧
  (x + 4*y + 8*z = m ∨ a + 4*b + 8*c > m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3190_319039


namespace NUMINAMATH_CALUDE_appointment_ways_l3190_319092

def dedicated_fitters : ℕ := 5
def dedicated_turners : ℕ := 4
def versatile_workers : ℕ := 2
def total_workers : ℕ := dedicated_fitters + dedicated_turners + versatile_workers
def required_fitters : ℕ := 4
def required_turners : ℕ := 4

theorem appointment_ways : 
  (Nat.choose dedicated_fitters required_fitters * Nat.choose dedicated_turners (required_turners - 1) * Nat.choose versatile_workers 1) +
  (Nat.choose dedicated_fitters (required_fitters - 1) * Nat.choose dedicated_turners required_turners * Nat.choose versatile_workers 1) +
  (Nat.choose dedicated_fitters required_fitters * Nat.choose dedicated_turners (required_turners - 2) * Nat.choose versatile_workers 2) +
  (Nat.choose dedicated_fitters (required_fitters - 1) * Nat.choose dedicated_turners (required_turners - 1) * Nat.choose versatile_workers 2) +
  (Nat.choose dedicated_fitters (required_fitters - 1) * Nat.choose dedicated_turners (required_turners - 2) * Nat.choose versatile_workers 2) = 190 := by
  sorry

end NUMINAMATH_CALUDE_appointment_ways_l3190_319092


namespace NUMINAMATH_CALUDE_eastbound_plane_speed_l3190_319058

/-- Given two planes traveling in opposite directions, this theorem proves
    the speed of the eastbound plane given the conditions of the problem. -/
theorem eastbound_plane_speed
  (time : ℝ)
  (westbound_speed : ℝ)
  (total_distance : ℝ)
  (h_time : time = 3.5)
  (h_westbound : westbound_speed = 275)
  (h_distance : total_distance = 2100) :
  ∃ (eastbound_speed : ℝ),
    eastbound_speed = 325 ∧
    (eastbound_speed + westbound_speed) * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_eastbound_plane_speed_l3190_319058
