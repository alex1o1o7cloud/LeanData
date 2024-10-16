import Mathlib

namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l799_79976

/-- Given a circle with center (5, -2) and one endpoint of a diameter at (2, 3),
    prove that the other endpoint of the diameter is at (8, -7). -/
theorem circle_diameter_endpoint (center : ℝ × ℝ) (endpoint1 : ℝ × ℝ) (endpoint2 : ℝ × ℝ) : 
  center = (5, -2) → endpoint1 = (2, 3) → endpoint2 = (8, -7) → 
  (center.1 - endpoint1.1 = endpoint2.1 - center.1 ∧ 
   center.2 - endpoint1.2 = endpoint2.2 - center.2) := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l799_79976


namespace NUMINAMATH_CALUDE_factorization_of_2x_cubed_minus_8x_l799_79986

theorem factorization_of_2x_cubed_minus_8x (x : ℝ) : 2*x^3 - 8*x = 2*x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_cubed_minus_8x_l799_79986


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l799_79909

/-- Parabola defined by y = x^2 + 5 -/
def P : ℝ → ℝ := λ x ↦ x^2 + 5

/-- Point Q -/
def Q : ℝ × ℝ := (10, 10)

/-- Line through Q with slope m -/
def line (m : ℝ) : ℝ → ℝ := λ x ↦ m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line m x

theorem parabola_line_intersection :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 40 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l799_79909


namespace NUMINAMATH_CALUDE_tiffany_max_points_l799_79921

/-- Represents the ring toss game --/
structure RingTossGame where
  total_money : ℕ
  cost_per_play : ℕ
  rings_per_play : ℕ
  red_points : ℕ
  green_points : ℕ
  games_played : ℕ
  red_buckets_hit : ℕ
  green_buckets_hit : ℕ

/-- Calculates the maximum points possible for the given game state --/
def max_points (game : RingTossGame) : ℕ :=
  let points_so_far := game.red_buckets_hit * game.red_points + game.green_buckets_hit * game.green_points
  let remaining_games := game.total_money / game.cost_per_play - game.games_played
  let max_additional_points := remaining_games * game.rings_per_play * game.green_points
  points_so_far + max_additional_points

/-- Theorem stating that the maximum points Tiffany can get is 38 --/
theorem tiffany_max_points :
  let game := RingTossGame.mk 3 1 5 2 3 2 4 5
  max_points game = 38 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_max_points_l799_79921


namespace NUMINAMATH_CALUDE_quadratic_inequality_l799_79980

theorem quadratic_inequality (a x : ℝ) : 
  a * x^2 - (a + 2) * x + 2 < 0 ↔ 
    ((a < 0 ∧ (x < 2/a ∨ x > 1)) ∨
     (a = 0 ∧ x > 1) ∨
     (0 < a ∧ a < 2 ∧ 1 < x ∧ x < 2/a) ∨
     (a > 2 ∧ 2/a < x ∧ x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l799_79980


namespace NUMINAMATH_CALUDE_distinct_sums_count_l799_79907

/-- The set of ball numbers -/
def BallNumbers : Finset ℕ := {1, 2, 3, 4, 5}

/-- The sum of two numbers drawn from BallNumbers with replacement -/
def SumOfDraws : Finset ℕ := Finset.image (λ (x : ℕ × ℕ) => x.1 + x.2) (BallNumbers.product BallNumbers)

/-- The number of distinct possible sums -/
theorem distinct_sums_count : Finset.card SumOfDraws = 9 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_count_l799_79907


namespace NUMINAMATH_CALUDE_number_with_special_divisor_property_l799_79922

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def proper_divisor (d n : ℕ) : Prop := d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def divisor_difference_property (n : ℕ) : Prop :=
  ∀ d₁ d₂ : ℕ, proper_divisor d₁ n → proper_divisor d₂ n → d₁ ≠ d₂ → (d₁ - d₂) ∣ n

theorem number_with_special_divisor_property (n : ℕ) :
  n ≥ 2 →
  (∃ (d₁ d₂ d₃ d₄ : ℕ), d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧
    ∀ d : ℕ, d ∣ n → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄) →
  divisor_difference_property n →
  n = 4 ∨ is_prime n :=
sorry

end NUMINAMATH_CALUDE_number_with_special_divisor_property_l799_79922


namespace NUMINAMATH_CALUDE_black_block_is_t_shaped_l799_79948

/-- Represents the shape of a block --/
inductive BlockShape
  | L
  | T
  | S
  | I

/-- Represents a block in the rectangular prism --/
structure Block where
  shape : BlockShape
  visible : Bool
  inLowestLayer : Bool

/-- Represents the rectangular prism --/
structure RectangularPrism where
  blocks : Fin 4 → Block
  threeFullyVisible : ∃ (a b c : Fin 4), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (blocks a).visible ∧ (blocks b).visible ∧ (blocks c).visible
  onePartiallyVisible : ∃ (d : Fin 4), ¬(blocks d).visible
  blackBlockInLowestLayer : ∃ (d : Fin 4), ¬(blocks d).visible ∧ (blocks d).inLowestLayer

/-- The main theorem --/
theorem black_block_is_t_shaped (prism : RectangularPrism) : 
  ∃ (d : Fin 4), ¬(prism.blocks d).visible ∧ (prism.blocks d).shape = BlockShape.T := by
  sorry

end NUMINAMATH_CALUDE_black_block_is_t_shaped_l799_79948


namespace NUMINAMATH_CALUDE_two_p_plus_q_l799_79996

theorem two_p_plus_q (p q r : ℚ) (h1 : p / q = 5 / 4) (h2 : p = r^2) : 2 * p + q = 7 * q / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_p_plus_q_l799_79996


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l799_79981

def selling_price : ℝ := 1110
def cost_price : ℝ := 925

theorem shopkeeper_profit_percentage :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l799_79981


namespace NUMINAMATH_CALUDE_least_number_of_cookies_l799_79930

theorem least_number_of_cookies (a : ℕ) : 
  a > 0 ∧ 
  a % 4 = 3 ∧ 
  a % 5 = 2 ∧ 
  a % 7 = 4 ∧ 
  (∀ b : ℕ, b > 0 ∧ b % 4 = 3 ∧ b % 5 = 2 ∧ b % 7 = 4 → a ≤ b) → 
  a = 67 := by
sorry

end NUMINAMATH_CALUDE_least_number_of_cookies_l799_79930


namespace NUMINAMATH_CALUDE_no_partition_for_large_a_l799_79932

theorem no_partition_for_large_a (a : ℝ) (ha : a ≥ 2) :
  ¬ ∃ (n : ℕ) (A : ℕ → Set ℤ),
    (∀ i j, i ≠ j → A i ∩ A j = ∅) ∧
    (∀ i, Set.Infinite (A i)) ∧
    (⋃ i, A i) = Set.univ ∧
    (∀ i x y, x ∈ A i → y ∈ A i → x > y → x - y ≥ a ^ i) :=
by sorry

end NUMINAMATH_CALUDE_no_partition_for_large_a_l799_79932


namespace NUMINAMATH_CALUDE_parabola_b_value_l799_79911

/-- A parabola with equation y = x^2 + bx + 3 passing through the points (1, 5), (3, 5), and (0, 3) has b = 1 -/
theorem parabola_b_value : ∃ b : ℝ,
  (∀ x y : ℝ, y = x^2 + b*x + 3 →
    ((x = 1 ∧ y = 5) ∨ (x = 3 ∧ y = 5) ∨ (x = 0 ∧ y = 3))) →
  b = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l799_79911


namespace NUMINAMATH_CALUDE_fifth_term_is_negative_9216_l799_79940

def alternating_sequence (n : ℕ) : ℤ := 
  if n % 2 = 0 then (101 - n)^2 else -((101 - n)^2)

def sequence_sum (n : ℕ) : ℤ := 
  (List.range n).map alternating_sequence |>.sum

theorem fifth_term_is_negative_9216 (h : sequence_sum 100 = 5050) : 
  alternating_sequence 5 = -9216 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_negative_9216_l799_79940


namespace NUMINAMATH_CALUDE_largest_three_digit_geometric_l799_79942

def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ (r : ℚ), r ≠ 0 ∧ b = a * r ∧ c = b * r

def digits_are_distinct (n : ℕ) : Prop :=
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

theorem largest_three_digit_geometric : ∀ n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  digits_are_distinct n ∧
  is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10) ∧
  n / 100 ≤ 8 →
  n ≤ 842 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_geometric_l799_79942


namespace NUMINAMATH_CALUDE_interview_problem_l799_79965

theorem interview_problem (n : ℕ) 
  (h1 : n ≥ 2) 
  (h2 : (Nat.choose 2 2 * Nat.choose (n - 2) 1) / Nat.choose n 3 = 1 / 70) : 
  n = 21 := by
sorry

end NUMINAMATH_CALUDE_interview_problem_l799_79965


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l799_79973

/-- The function f(x) = x^4 - 2x^3 -/
def f (x : ℝ) : ℝ := x^4 - 2*x^3

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 4*x^3 - 6*x^2

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_deriv x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -2*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l799_79973


namespace NUMINAMATH_CALUDE_equation_solution_l799_79993

theorem equation_solution :
  ∀ y : ℝ, (2012 + y)^2 = 2*y^2 ↔ y = 2012*(Real.sqrt 2 + 1) ∨ y = -2012*(Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l799_79993


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_sum_half_l799_79963

theorem sin_cos_sum_equals_sqrt_sum_half :
  Real.sin (14 * π / 3) + Real.cos (-25 * π / 4) = (Real.sqrt 3 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_sum_half_l799_79963


namespace NUMINAMATH_CALUDE_article_pages_count_l799_79966

-- Define the constants
def total_word_limit : ℕ := 48000
def large_font_words_per_page : ℕ := 1800
def small_font_words_per_page : ℕ := 2400
def large_font_pages : ℕ := 4

-- Define the theorem
theorem article_pages_count :
  let words_in_large_font := large_font_pages * large_font_words_per_page
  let remaining_words := total_word_limit - words_in_large_font
  let small_font_pages := remaining_words / small_font_words_per_page
  large_font_pages + small_font_pages = 21 := by
sorry

end NUMINAMATH_CALUDE_article_pages_count_l799_79966


namespace NUMINAMATH_CALUDE_problem_solution_l799_79931

theorem problem_solution : ∃ x : ℝ, (0.25 * x = 0.15 * 1500 - 15) ∧ (x = 840) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l799_79931


namespace NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_seven_thirds_l799_79925

theorem abs_ratio_eq_sqrt_seven_thirds 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h : a^2 + b^2 = 5*a*b) : 
  |((a+b)/(a-b))| = Real.sqrt (7/3) := by
sorry

end NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_seven_thirds_l799_79925


namespace NUMINAMATH_CALUDE_monthly_salary_calculation_l799_79938

/-- Proves that a person's monthly salary is 6000 given the specified savings conditions -/
theorem monthly_salary_calculation (salary : ℝ) : 
  (salary * 0.2 = salary - (salary * 0.8 * 1.2 + 240)) → salary = 6000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_salary_calculation_l799_79938


namespace NUMINAMATH_CALUDE_john_share_l799_79941

/-- Given a total amount and a ratio, calculates the share for a specific part -/
def calculateShare (totalAmount : ℚ) (ratio : List ℚ) (part : ℚ) : ℚ :=
  let totalParts := ratio.sum
  let valuePerPart := totalAmount / totalParts
  valuePerPart * part

/-- Proves that given a total amount of 4200 and a ratio of 2:4:6:8, 
    the amount received by the person with 2 parts is 420 -/
theorem john_share : 
  let totalAmount : ℚ := 4200
  let ratio : List ℚ := [2, 4, 6, 8]
  calculateShare totalAmount ratio 2 = 420 := by sorry

end NUMINAMATH_CALUDE_john_share_l799_79941


namespace NUMINAMATH_CALUDE_exists_fourth_power_with_specific_divisors_l799_79946

/-- The number of divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem exists_fourth_power_with_specific_divisors :
  ∃ n : ℕ+, num_divisors (n ^ 4) = 301 ∨ num_divisors (n ^ 4) = 305 := by
  sorry

end NUMINAMATH_CALUDE_exists_fourth_power_with_specific_divisors_l799_79946


namespace NUMINAMATH_CALUDE_average_weight_abc_l799_79955

/-- Given three weights a, b, and c, prove that their average is 43 kg -/
theorem average_weight_abc (a b c : ℝ) 
  (hab : (a + b) / 2 = 40)  -- average of a and b is 40 kg
  (hbc : (b + c) / 2 = 43)  -- average of b and c is 43 kg
  (hb : b = 37)             -- weight of b is 37 kg
  : (a + b + c) / 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_abc_l799_79955


namespace NUMINAMATH_CALUDE_initial_boys_count_l799_79929

/-- The number of boys who went down the slide initially -/
def initial_boys : ℕ := sorry

/-- The number of additional boys who went down the slide -/
def additional_boys : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_boys : ℕ := 35

/-- Theorem stating that the initial number of boys is 22 -/
theorem initial_boys_count : initial_boys = 22 := by
  have h : initial_boys + additional_boys = total_boys := sorry
  sorry

end NUMINAMATH_CALUDE_initial_boys_count_l799_79929


namespace NUMINAMATH_CALUDE_function_characterization_l799_79999

-- Define the property that f must satisfy
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f y + f (x + y) ≥ (y + 1) * f x + f y

-- Theorem statement
theorem function_characterization (f : ℝ → ℝ) 
  (h : SatisfiesInequality f) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x := by
  sorry

end NUMINAMATH_CALUDE_function_characterization_l799_79999


namespace NUMINAMATH_CALUDE_second_group_size_l799_79905

/-- The number of men in the first group -/
def men_group1 : ℕ := 4

/-- The number of hours worked per day by the first group -/
def hours_per_day_group1 : ℕ := 10

/-- The earnings per week of the first group in rupees -/
def earnings_group1 : ℕ := 1000

/-- The number of hours worked per day by the second group -/
def hours_per_day_group2 : ℕ := 6

/-- The earnings per week of the second group in rupees -/
def earnings_group2 : ℕ := 1350

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of men in the second group -/
def men_group2 : ℕ := 9

theorem second_group_size :
  men_group2 * hours_per_day_group2 * days_per_week * earnings_group1 =
  men_group1 * hours_per_day_group1 * days_per_week * earnings_group2 :=
by sorry

end NUMINAMATH_CALUDE_second_group_size_l799_79905


namespace NUMINAMATH_CALUDE_inequality_proof_l799_79933

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  Real.sqrt x + Real.sqrt y + Real.sqrt z ≥ x * y + y * z + z * x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l799_79933


namespace NUMINAMATH_CALUDE_plus_shape_perimeter_l799_79956

/-- A shape formed by eight congruent squares arranged in a "plus" sign -/
structure PlusShape where
  /-- The side length of each square in the shape -/
  side_length : ℝ
  /-- The total area of the shape -/
  total_area : ℝ
  /-- The shape is formed by eight congruent squares -/
  area_eq : total_area = 8 * side_length ^ 2

/-- The perimeter of a PlusShape -/
def perimeter (shape : PlusShape) : ℝ := 12 * shape.side_length

theorem plus_shape_perimeter (shape : PlusShape) (h : shape.total_area = 648) :
  perimeter shape = 108 := by
  sorry

#check plus_shape_perimeter

end NUMINAMATH_CALUDE_plus_shape_perimeter_l799_79956


namespace NUMINAMATH_CALUDE_geometric_series_sum_l799_79982

/-- The sum of the geometric series 15 + 15r + 15r^2 + 15r^3 + ... for -1 < r < 1 -/
noncomputable def T (r : ℝ) : ℝ :=
  15 / (1 - r)

/-- For -1 < b < 1, if T(b)T(-b) = 3240, then T(b) + T(-b) = 432 -/
theorem geometric_series_sum (b : ℝ) (hb1 : -1 < b) (hb2 : b < 1) 
    (h : T b * T (-b) = 3240) : T b + T (-b) = 432 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l799_79982


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l799_79961

theorem decimal_to_fraction :
  (0.34 : ℚ) = 17 / 50 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l799_79961


namespace NUMINAMATH_CALUDE_lighthouse_coverage_l799_79992

/-- Represents a lighthouse with its illumination angle -/
structure Lighthouse where
  angle : ℝ

/-- Represents the Persian Gulf as a circle -/
def PersianGulf : ℝ := 360

/-- The number of lighthouses -/
def num_lighthouses : ℕ := 18

/-- The illumination angle of each lighthouse -/
def lighthouse_angle : ℝ := 20

/-- Proves that the lighthouses can cover the entire Persian Gulf -/
theorem lighthouse_coverage (lighthouses : Fin num_lighthouses → Lighthouse)
  (h1 : ∀ i, (lighthouses i).angle = lighthouse_angle)
  (h2 : lighthouse_angle * num_lighthouses = PersianGulf) :
  ∃ (arrangement : Fin num_lighthouses → ℝ),
    (∀ i, 0 ≤ arrangement i ∧ arrangement i < PersianGulf) ∧
    (∀ x, 0 ≤ x ∧ x < PersianGulf →
      ∃ i, x ∈ Set.Icc (arrangement i) ((arrangement i + (lighthouses i).angle) % PersianGulf)) :=
by sorry

end NUMINAMATH_CALUDE_lighthouse_coverage_l799_79992


namespace NUMINAMATH_CALUDE_triangle_area_l799_79959

/-- Given a right isosceles triangle that shares sides with squares of areas 100, 64, and 100,
    prove that the area of the triangle is 50. -/
theorem triangle_area (a b c : ℝ) (ha : a^2 = 100) (hb : b^2 = 64) (hc : c^2 = 100)
  (right_isosceles : a = c ∧ a^2 + c^2 = b^2) : (1/2) * a * c = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l799_79959


namespace NUMINAMATH_CALUDE_subtracted_value_l799_79902

def original_number : ℝ := 54

theorem subtracted_value (x : ℝ) :
  ((original_number - x) / 7 = 7) ∧
  ((original_number - 34) / 10 = 2) →
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l799_79902


namespace NUMINAMATH_CALUDE_inequality_theorem_l799_79926

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hxy₁ : x₁ * y₁ - z₁^2 > 0) (hxy₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ∧
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔
   x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l799_79926


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_l799_79903

theorem arithmetic_mean_reciprocals : 
  let numbers := [2, 3, 7, 11]
  let reciprocals := numbers.map (λ x => 1 / x)
  let sum := reciprocals.sum
  let mean := sum / 4
  mean = 493 / 1848 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_l799_79903


namespace NUMINAMATH_CALUDE_king_of_diamonds_in_top_two_l799_79985

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (ranks : ℕ)
  (jokers : ℕ)

/-- The probability of an event occurring -/
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

/-- Theorem stating the probability of the King of Diamonds being one of the top two cards -/
theorem king_of_diamonds_in_top_two (d : Deck) 
  (h1 : d.total_cards = 54)
  (h2 : d.suits = 4)
  (h3 : d.ranks = 13)
  (h4 : d.jokers = 2) :
  probability 2 d.total_cards = 1 / 27 := by
  sorry

#check king_of_diamonds_in_top_two

end NUMINAMATH_CALUDE_king_of_diamonds_in_top_two_l799_79985


namespace NUMINAMATH_CALUDE_household_gas_fee_l799_79971

def gas_fee (usage : ℕ) : ℚ :=
  if usage ≤ 60 then
    0.8 * usage
  else
    0.8 * 60 + 1.2 * (usage - 60)

theorem household_gas_fee :
  ∃ (usage : ℕ),
    usage > 60 ∧
    gas_fee usage / usage = 0.88 ∧
    gas_fee usage = 66 := by
  sorry

end NUMINAMATH_CALUDE_household_gas_fee_l799_79971


namespace NUMINAMATH_CALUDE_floor_tiling_l799_79987

theorem floor_tiling (n : ℕ) (h1 : 10 < n) (h2 : n < 20) :
  (∃ x : ℕ, n^2 = 9*x) ↔ n = 12 ∨ n = 15 ∨ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_floor_tiling_l799_79987


namespace NUMINAMATH_CALUDE_butter_fraction_for_chocolate_chip_cookies_l799_79917

theorem butter_fraction_for_chocolate_chip_cookies 
  (total_butter : ℝ)
  (peanut_butter_fraction : ℝ)
  (sugar_cookie_fraction : ℝ)
  (remaining_butter : ℝ)
  (h1 : total_butter = 10)
  (h2 : peanut_butter_fraction = 1/5)
  (h3 : sugar_cookie_fraction = 1/3)
  (h4 : remaining_butter = 2)
  : (total_butter - (peanut_butter_fraction * total_butter) - 
     sugar_cookie_fraction * (total_butter - peanut_butter_fraction * total_butter) - 
     remaining_butter) / total_butter = 1/3 := by
  sorry

#check butter_fraction_for_chocolate_chip_cookies

end NUMINAMATH_CALUDE_butter_fraction_for_chocolate_chip_cookies_l799_79917


namespace NUMINAMATH_CALUDE_museum_visitors_scientific_notation_l799_79923

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (n : ℝ) : ScientificNotation :=
  sorry

theorem museum_visitors_scientific_notation :
  toScientificNotation 3300000 = ScientificNotation.mk 3.3 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_museum_visitors_scientific_notation_l799_79923


namespace NUMINAMATH_CALUDE_sum_of_digits_in_multiple_of_72_l799_79970

theorem sum_of_digits_in_multiple_of_72 (A B : ℕ) : 
  A < 10 → B < 10 → (A * 100000 + 44610 + B) % 72 = 0 → A + B = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_multiple_of_72_l799_79970


namespace NUMINAMATH_CALUDE_count_pairs_eq_nine_l799_79900

/-- The number of distinct ordered pairs of positive integers (x, y) such that 1/x + 1/y = 1/6 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 6)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card

theorem count_pairs_eq_nine : count_pairs = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_nine_l799_79900


namespace NUMINAMATH_CALUDE_ellipse_vertex_distance_l799_79906

/-- The distance between the vertices of the ellipse x^2/49 + y^2/64 = 1 is 16 -/
theorem ellipse_vertex_distance :
  let a := Real.sqrt (max 49 64)
  let ellipse := {(x, y) : ℝ × ℝ | x^2/49 + y^2/64 = 1}
  2 * a = 16 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_vertex_distance_l799_79906


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l799_79919

-- Define the total number of students
def total_students : ℕ := 800

-- Define the number of students preferring spaghetti
def spaghetti_preference : ℕ := 320

-- Define the number of students preferring fettuccine
def fettuccine_preference : ℕ := 160

-- Theorem to prove the ratio
theorem pasta_preference_ratio : 
  (spaghetti_preference : ℚ) / (fettuccine_preference : ℚ) = 2 := by
  sorry


end NUMINAMATH_CALUDE_pasta_preference_ratio_l799_79919


namespace NUMINAMATH_CALUDE_statements_classification_correct_l799_79984

-- Define the type of statement
inductive StatementType
  | Universal
  | Existential

-- Define a structure to represent a statement
structure Statement where
  text : String
  type : StatementType
  isTrue : Bool

-- Define the four statements
def statement1 : Statement :=
  { text := "The diagonals of a square are perpendicular bisectors of each other"
  , type := StatementType.Universal
  , isTrue := true }

def statement2 : Statement :=
  { text := "All Chinese people speak Chinese"
  , type := StatementType.Universal
  , isTrue := false }

def statement3 : Statement :=
  { text := "Some numbers are greater than their squares"
  , type := StatementType.Existential
  , isTrue := true }

def statement4 : Statement :=
  { text := "Some real numbers have irrational square roots"
  , type := StatementType.Existential
  , isTrue := true }

-- Theorem to prove the correctness of the statements' classifications
theorem statements_classification_correct :
  statement1.type = StatementType.Universal ∧ statement1.isTrue = true ∧
  statement2.type = StatementType.Universal ∧ statement2.isTrue = false ∧
  statement3.type = StatementType.Existential ∧ statement3.isTrue = true ∧
  statement4.type = StatementType.Existential ∧ statement4.isTrue = true :=
by sorry

end NUMINAMATH_CALUDE_statements_classification_correct_l799_79984


namespace NUMINAMATH_CALUDE_henry_initial_amount_l799_79967

/-- Henry's initial amount of money -/
def henry_initial : ℕ := sorry

/-- Amount Henry earned from chores -/
def chores_earnings : ℕ := 2

/-- Amount of money Henry's friend had -/
def friend_money : ℕ := 13

/-- Total amount when they put their money together -/
def total_money : ℕ := 20

theorem henry_initial_amount :
  henry_initial + chores_earnings + friend_money = total_money ∧
  henry_initial = 5 := by sorry

end NUMINAMATH_CALUDE_henry_initial_amount_l799_79967


namespace NUMINAMATH_CALUDE_squares_below_line_l799_79998

/-- Represents a line in the form ax + by = c --/
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Counts the number of integer points strictly below a line in the first quadrant --/
def countPointsBelowLine (l : Line) : ℕ :=
  sorry

/-- The line 8x + 245y = 1960 --/
def problemLine : Line := { a := 8, b := 245, c := 1960 }

theorem squares_below_line :
  countPointsBelowLine problemLine = 853 :=
sorry

end NUMINAMATH_CALUDE_squares_below_line_l799_79998


namespace NUMINAMATH_CALUDE_donut_selection_count_l799_79954

theorem donut_selection_count :
  let n : ℕ := 6  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 84 := by
sorry

end NUMINAMATH_CALUDE_donut_selection_count_l799_79954


namespace NUMINAMATH_CALUDE_fruit_cost_theorem_l799_79994

def fruit_prices (o g w f : ℝ) : Prop :=
  o + g + w + f = 24 ∧ f = 3 * o ∧ w = o - 2 * g

theorem fruit_cost_theorem :
  ∀ o g w f : ℝ, fruit_prices o g w f → g + w = 4.8 :=
by sorry

end NUMINAMATH_CALUDE_fruit_cost_theorem_l799_79994


namespace NUMINAMATH_CALUDE_oranges_packed_l799_79949

/-- Calculates the total number of oranges packed given the number of oranges per box and the number of boxes used. -/
def totalOranges (orangesPerBox : ℕ) (boxesUsed : ℕ) : ℕ :=
  orangesPerBox * boxesUsed

/-- Proves that packing 10 oranges per box in 265 boxes results in 2650 oranges packed. -/
theorem oranges_packed :
  let orangesPerBox : ℕ := 10
  let boxesUsed : ℕ := 265
  totalOranges orangesPerBox boxesUsed = 2650 := by
  sorry

end NUMINAMATH_CALUDE_oranges_packed_l799_79949


namespace NUMINAMATH_CALUDE_egyptian_fraction_solutions_l799_79950

def EgyptianFractionSolutions : Set (ℕ × ℕ × ℕ × ℕ) := {
  (2, 3, 7, 42), (2, 3, 8, 24), (2, 3, 9, 18), (2, 3, 10, 15), (2, 3, 12, 12),
  (2, 4, 5, 20), (2, 4, 6, 12), (2, 4, 8, 8), (2, 5, 5, 10), (2, 6, 6, 6),
  (3, 3, 4, 12), (3, 3, 6, 6), (3, 4, 4, 6), (4, 4, 4, 4)
}

theorem egyptian_fraction_solutions :
  {(x, y, z, t) : ℕ × ℕ × ℕ × ℕ | 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧
    (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z + (1 : ℚ) / t = 1 ∧
    x ≤ y ∧ y ≤ z ∧ z ≤ t} = EgyptianFractionSolutions := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_solutions_l799_79950


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l799_79988

universe u

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {1, 2, 5}

theorem complement_intersection_theorem : (U \ A) ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l799_79988


namespace NUMINAMATH_CALUDE_extreme_value_condition_decreasing_function_condition_l799_79983

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 - b*x

-- Theorem for part (1)
theorem extreme_value_condition (a b : ℝ) :
  (∃ x : ℝ, f a b x = 2 ∧ ∀ y : ℝ, f a b y ≤ f a b x) ∧ f a b 1 = 2 →
  a = 1 ∧ b = 3 :=
sorry

-- Theorem for part (2)
theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 2 → f a (9*a) x > f a (9*a) y) →
  a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_decreasing_function_condition_l799_79983


namespace NUMINAMATH_CALUDE_johns_arcade_spending_l799_79957

theorem johns_arcade_spending (total_allowance : ℚ) 
  (remaining_after_toy_store : ℚ) (toy_store_fraction : ℚ) 
  (h1 : total_allowance = 9/4)
  (h2 : remaining_after_toy_store = 3/5)
  (h3 : toy_store_fraction = 1/3) : 
  ∃ (arcade_fraction : ℚ), 
    arcade_fraction = 3/5 ∧ 
    remaining_after_toy_store = (1 - arcade_fraction) * total_allowance * (1 - toy_store_fraction) :=
by sorry

end NUMINAMATH_CALUDE_johns_arcade_spending_l799_79957


namespace NUMINAMATH_CALUDE_johns_final_push_pace_l799_79972

/-- Proves that John's pace during his final push was 4.2 m/s given the race conditions --/
theorem johns_final_push_pace (initial_distance : ℝ) (steve_speed : ℝ) (final_distance : ℝ) (push_duration : ℝ) :
  initial_distance = 12 →
  steve_speed = 3.7 →
  final_distance = 2 →
  push_duration = 28 →
  (push_duration * steve_speed + initial_distance + final_distance) / push_duration = 4.2 :=
by sorry

end NUMINAMATH_CALUDE_johns_final_push_pace_l799_79972


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l799_79901

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x ≤ 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

-- Theorem statement
theorem union_of_A_and_B :
  A ∪ B = { x | -1 < x ∧ x < 2 } := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l799_79901


namespace NUMINAMATH_CALUDE_division_simplification_l799_79953

theorem division_simplification : 
  (250 : ℚ) / (15 + 13 * 3^2) = 125 / 66 := by sorry

end NUMINAMATH_CALUDE_division_simplification_l799_79953


namespace NUMINAMATH_CALUDE_prime_triplets_equation_l799_79943

theorem prime_triplets_equation (p q r : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  (p : ℚ) / q = 8 / (r - 1 : ℚ) + 1 ↔ 
  ((p = 3 ∧ q = 2 ∧ r = 17) ∨ 
   (p = 7 ∧ q = 3 ∧ r = 7) ∨ 
   (p = 5 ∧ q = 3 ∧ r = 13)) :=
by sorry

end NUMINAMATH_CALUDE_prime_triplets_equation_l799_79943


namespace NUMINAMATH_CALUDE_pager_fraction_l799_79960

theorem pager_fraction (total : ℝ) (total_pos : 0 < total) : 
  let cell_phone := (2/3 : ℝ) * total
  let neither := (1/3 : ℝ) * total
  let both := (0.4 : ℝ) * total
  let pager := (0.8 : ℝ) * total
  (cell_phone + (pager - both) = total - neither) →
  (pager / total = 0.8) :=
by
  sorry

end NUMINAMATH_CALUDE_pager_fraction_l799_79960


namespace NUMINAMATH_CALUDE_class_composition_l799_79989

theorem class_composition (num_boys : ℕ) (avg_boys avg_girls avg_class : ℚ) :
  num_boys = 12 →
  avg_boys = 84 →
  avg_girls = 92 →
  avg_class = 86 →
  ∃ (num_girls : ℕ), 
    (num_boys : ℚ) * avg_boys + (num_girls : ℚ) * avg_girls = 
    ((num_boys : ℚ) + (num_girls : ℚ)) * avg_class ∧
    num_girls = 4 :=
by sorry

end NUMINAMATH_CALUDE_class_composition_l799_79989


namespace NUMINAMATH_CALUDE_oldest_bride_age_l799_79952

theorem oldest_bride_age (bride_age groom_age : ℕ) : 
  bride_age = groom_age + 19 →
  bride_age + groom_age = 185 →
  bride_age = 102 := by
sorry

end NUMINAMATH_CALUDE_oldest_bride_age_l799_79952


namespace NUMINAMATH_CALUDE_square_area_comparison_l799_79937

theorem square_area_comparison (a b : ℝ) (h : b = 4 * a) :
  b ^ 2 = 16 * a ^ 2 := by sorry

end NUMINAMATH_CALUDE_square_area_comparison_l799_79937


namespace NUMINAMATH_CALUDE_total_amount_is_265_l799_79944

/-- Represents the distribution of money among six individuals -/
structure MoneyDistribution where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  t : ℝ
  u : ℝ

/-- The theorem stating the total amount given the conditions -/
theorem total_amount_is_265 (dist : MoneyDistribution) : 
  (dist.p = 3 * (dist.s / 1.95)) →
  (dist.q = 2.70 * (dist.s / 1.95)) →
  (dist.r = 2.30 * (dist.s / 1.95)) →
  (dist.s = 39) →
  (dist.t = 1.80 * (dist.s / 1.95)) →
  (dist.u = 1.50 * (dist.s / 1.95)) →
  (dist.p + dist.q + dist.r + dist.s + dist.t + dist.u = 265) := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_265_l799_79944


namespace NUMINAMATH_CALUDE_distance_between_axis_endpoints_l799_79947

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 16 * (x - 3)^2 + 4 * (y + 2)^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (3, -2)

-- Define the length of semi-major axis
def semi_major_axis : ℝ := 4

-- Define the length of semi-minor axis
def semi_minor_axis : ℝ := 2

-- Define a point on the ellipse
def point_on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

-- Define an endpoint of the major axis
def major_axis_endpoint (p : ℝ × ℝ) : Prop :=
  point_on_ellipse p ∧ 
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = semi_major_axis^2

-- Define an endpoint of the minor axis
def minor_axis_endpoint (p : ℝ × ℝ) : Prop :=
  point_on_ellipse p ∧ 
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = semi_minor_axis^2

-- Theorem statement
theorem distance_between_axis_endpoints :
  ∀ (C D : ℝ × ℝ), 
  major_axis_endpoint C → minor_axis_endpoint D →
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = 20 :=
sorry

end NUMINAMATH_CALUDE_distance_between_axis_endpoints_l799_79947


namespace NUMINAMATH_CALUDE_abs_negative_2023_l799_79951

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l799_79951


namespace NUMINAMATH_CALUDE_ten_thousands_representation_l799_79934

def ten_thousands : ℕ := 10000

def three_thousand_nine_hundred_seventy_six : ℕ := 3976

theorem ten_thousands_representation :
  three_thousand_nine_hundred_seventy_six * ten_thousands = 39760000 ∧
  three_thousand_nine_hundred_seventy_six = 3976 :=
by sorry

end NUMINAMATH_CALUDE_ten_thousands_representation_l799_79934


namespace NUMINAMATH_CALUDE_unique_operator_assignment_l799_79927

-- Define the arithmetic operators
inductive Operator
| Plus
| Minus
| Multiply
| Divide
| Equals

-- Define a function to apply an operator
def apply_operator (op : Operator) (a b : ℕ) : Prop :=
  match op with
  | Operator.Plus => a + b = b
  | Operator.Minus => a - b = b
  | Operator.Multiply => a * b = b
  | Operator.Divide => a / b = b
  | Operator.Equals => a = b

-- Define the theorem
theorem unique_operator_assignment :
  ∃! (A B C D E : Operator),
    apply_operator A 4 2 ∧
    apply_operator B 2 2 ∧
    apply_operator B 8 (4 * 2) ∧
    apply_operator C 4 2 ∧
    apply_operator D 2 3 ∧
    apply_operator B 5 5 ∧
    apply_operator B 4 (5 - 1) ∧
    apply_operator E 5 1 ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E :=
sorry

end NUMINAMATH_CALUDE_unique_operator_assignment_l799_79927


namespace NUMINAMATH_CALUDE_robins_hair_length_l799_79924

/-- Robin's hair length problem -/
theorem robins_hair_length (initial_length cut_length : ℕ) (h1 : initial_length = 14) (h2 : cut_length = 13) :
  initial_length - cut_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_robins_hair_length_l799_79924


namespace NUMINAMATH_CALUDE_candle_burn_time_l799_79990

/-- Proves that given a candle that lasts 8 nights when burned for 1 hour per night, 
    if 6 candles are used over 24 nights, then the average burn time per night is 2 hours. -/
theorem candle_burn_time 
  (candle_duration : ℕ) 
  (burn_time_per_night : ℕ) 
  (num_candles : ℕ) 
  (total_nights : ℕ) 
  (h1 : candle_duration = 8)
  (h2 : burn_time_per_night = 1)
  (h3 : num_candles = 6)
  (h4 : total_nights = 24) :
  (candle_duration * burn_time_per_night * num_candles) / total_nights = 2 := by
  sorry

end NUMINAMATH_CALUDE_candle_burn_time_l799_79990


namespace NUMINAMATH_CALUDE_mary_earnings_per_home_l799_79977

/-- Mary's earnings per home, given total earnings and number of homes cleaned -/
def earnings_per_home (total_earnings : ℕ) (homes_cleaned : ℕ) : ℕ :=
  total_earnings / homes_cleaned

/-- Proof that Mary earns $46 per home -/
theorem mary_earnings_per_home :
  earnings_per_home 276 6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_per_home_l799_79977


namespace NUMINAMATH_CALUDE_managers_salary_l799_79918

/-- Proves that the manager's salary is 3400 given the conditions of the problem -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (salary_increase : ℚ) : 
  num_employees = 20 →
  avg_salary = 1300 →
  salary_increase = 100 →
  (num_employees * avg_salary + 3400) / (num_employees + 1) = avg_salary + salary_increase :=
by
  sorry

#check managers_salary

end NUMINAMATH_CALUDE_managers_salary_l799_79918


namespace NUMINAMATH_CALUDE_zoo_problem_solution_l799_79920

/-- Represents the number of animals in each exhibit -/
structure ZooExhibits where
  rainForest : ℕ
  reptileHouse : ℕ
  aquarium : ℕ
  aviary : ℕ
  mammalHouse : ℕ

/-- Checks if the given numbers of animals satisfy the conditions of the zoo problem -/
def satisfiesZooConditions (exhibits : ZooExhibits) : Prop :=
  exhibits.reptileHouse = 3 * exhibits.rainForest - 5 ∧
  exhibits.reptileHouse = 16 ∧
  exhibits.aquarium = 2 * exhibits.reptileHouse ∧
  exhibits.aviary = (exhibits.aquarium - exhibits.rainForest) + 3 ∧
  exhibits.mammalHouse = ((exhibits.rainForest + exhibits.aquarium + exhibits.aviary) / 3 + 2)

/-- The theorem stating that there exists a unique solution to the zoo problem -/
theorem zoo_problem_solution : 
  ∃! exhibits : ZooExhibits, satisfiesZooConditions exhibits ∧ 
    exhibits.rainForest = 7 ∧ 
    exhibits.aquarium = 32 ∧ 
    exhibits.aviary = 28 ∧ 
    exhibits.mammalHouse = 24 :=
  sorry

end NUMINAMATH_CALUDE_zoo_problem_solution_l799_79920


namespace NUMINAMATH_CALUDE_sin_30_degrees_l799_79913

open Real

theorem sin_30_degrees :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l799_79913


namespace NUMINAMATH_CALUDE_window_width_is_ten_l799_79908

-- Define the window parameters
def window_length : ℝ := 6
def window_area : ℝ := 60

-- Theorem statement
theorem window_width_is_ten :
  ∃ w : ℝ, w * window_length = window_area ∧ w = 10 :=
by sorry

end NUMINAMATH_CALUDE_window_width_is_ten_l799_79908


namespace NUMINAMATH_CALUDE_symmetry_implies_exponential_l799_79935

-- Define the logarithm function base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the symmetry condition
def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_exponential (f : ℝ → ℝ) :
  (∀ x > 0, f (log3 x) = x) →
  symmetric_wrt_y_eq_x f log3 →
  ∀ x, f x = 3^x :=
sorry

end NUMINAMATH_CALUDE_symmetry_implies_exponential_l799_79935


namespace NUMINAMATH_CALUDE_jacksons_email_deletion_l799_79912

theorem jacksons_email_deletion (initial_deletion : ℕ) (first_received : ℕ) 
  (second_received : ℕ) (final_received : ℕ) (final_inbox : ℕ) :
  initial_deletion = 50 →
  first_received = 15 →
  second_received = 5 →
  final_received = 10 →
  final_inbox = 30 →
  ∃ (second_deletion : ℕ), 
    second_deletion = 50 ∧
    final_inbox = first_received + second_received + final_received - initial_deletion - second_deletion :=
by sorry

end NUMINAMATH_CALUDE_jacksons_email_deletion_l799_79912


namespace NUMINAMATH_CALUDE_intersection_when_m_2_B_subset_A_iff_l799_79928

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 6}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1}

-- Part 1: Intersection when m = 2
theorem intersection_when_m_2 : A ∩ B 2 = {x | 3 ≤ x ∧ x ≤ 5} := by
  sorry

-- Part 2: Condition for B to be a subset of A
theorem B_subset_A_iff (m : ℝ) : B m ⊆ A ↔ m ≤ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_when_m_2_B_subset_A_iff_l799_79928


namespace NUMINAMATH_CALUDE_unique_prime_solution_l799_79969

theorem unique_prime_solution :
  ∀ p q r : ℕ,
  Prime p → Prime q → Prime r →
  p + q^2 = r^4 →
  p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l799_79969


namespace NUMINAMATH_CALUDE_triangle_acute_angled_l799_79968

theorem triangle_acute_angled (a b c : ℝ) 
  (triangle_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (sides_relation : a^4 + b^4 = c^4) : 
  c^2 < a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_acute_angled_l799_79968


namespace NUMINAMATH_CALUDE_base_h_equation_l799_79974

/-- Converts a base-h number to decimal --/
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

/-- Checks if a list of digits is valid in base h --/
def valid_digits (digits : List Nat) (h : Nat) : Prop :=
  ∀ d ∈ digits, d < h

theorem base_h_equation (h : Nat) : 
  h > 8 → 
  valid_digits [8, 6, 7, 4] h → 
  valid_digits [4, 3, 2, 9] h → 
  valid_digits [1, 3, 0, 0, 3] h → 
  to_decimal [8, 6, 7, 4] h + to_decimal [4, 3, 2, 9] h = to_decimal [1, 3, 0, 0, 3] h → 
  h = 10 :=
sorry

end NUMINAMATH_CALUDE_base_h_equation_l799_79974


namespace NUMINAMATH_CALUDE_total_fruits_picked_l799_79945

theorem total_fruits_picked (mike_pears jason_pears fred_apples sarah_apples : ℕ)
  (h1 : mike_pears = 8)
  (h2 : jason_pears = 7)
  (h3 : fred_apples = 6)
  (h4 : sarah_apples = 12) :
  mike_pears + jason_pears + fred_apples + sarah_apples = 33 :=
by sorry

end NUMINAMATH_CALUDE_total_fruits_picked_l799_79945


namespace NUMINAMATH_CALUDE_only_yes_allows_deduction_l799_79978

/-- Represents the three types of natives on the island --/
inductive NativeType
  | Normal
  | Zombie
  | HalfZombie

/-- Represents possible answers in the native language --/
inductive Answer
  | Yes
  | No
  | Bal

/-- Function to determine if a native tells the truth based on their type and the question number --/
def tellsTruth (t : NativeType) (questionNumber : Nat) : Bool :=
  match t with
  | NativeType.Normal => true
  | NativeType.Zombie => false
  | NativeType.HalfZombie => questionNumber % 2 = 0

/-- The complex question asked by Inspector Craig --/
def inspectorQuestion (a : Answer) : Prop :=
  ∃ (t : NativeType), tellsTruth t 1 = (a = Answer.Yes)

/-- Theorem stating that "Yes" is the only answer that allows deduction of native type --/
theorem only_yes_allows_deduction :
  ∃! (a : Answer), ∀ (t : NativeType), inspectorQuestion a ↔ t = NativeType.HalfZombie :=
sorry


end NUMINAMATH_CALUDE_only_yes_allows_deduction_l799_79978


namespace NUMINAMATH_CALUDE_poster_width_l799_79962

theorem poster_width (height : ℝ) (area : ℝ) (width : ℝ) 
  (h1 : height = 7)
  (h2 : area = 28)
  (h3 : area = width * height) : 
  width = 4 := by sorry

end NUMINAMATH_CALUDE_poster_width_l799_79962


namespace NUMINAMATH_CALUDE_triangle_problem_l799_79997

def triangle_ABC (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_problem (a b c : ℝ) (h_triangle : triangle_ABC a b c) 
  (h_angle : Real.cos (π/3) = (b^2 + c^2 - a^2) / (2*b*c))
  (h_sides : a^2 - c^2 = (2/3) * b^2) :
  Real.tan (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = Real.sqrt 3 / 5 ∧
  (1/2 * b * c * Real.sin (π/3) = 3 * Real.sqrt 3 / 4 → a = Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l799_79997


namespace NUMINAMATH_CALUDE_remainder_theorem_l799_79915

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 60 * k - 1) :
  (n^2 + 2*n + 3) % 60 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l799_79915


namespace NUMINAMATH_CALUDE_bridgette_birds_l799_79991

/-- The number of birds Bridgette has -/
def num_birds : ℕ := sorry

/-- The number of dogs Bridgette has -/
def num_dogs : ℕ := 2

/-- The number of cats Bridgette has -/
def num_cats : ℕ := 3

/-- The number of baths dogs get per month -/
def dog_baths_per_month : ℕ := 2

/-- The number of baths cats get per month -/
def cat_baths_per_month : ℕ := 1

/-- The number of months between bird baths -/
def months_between_bird_baths : ℕ := 4

/-- The total number of baths given in a year -/
def total_baths_per_year : ℕ := 96

/-- The number of months in a year -/
def months_per_year : ℕ := 12

theorem bridgette_birds :
  num_birds = 4 :=
by sorry

end NUMINAMATH_CALUDE_bridgette_birds_l799_79991


namespace NUMINAMATH_CALUDE_julia_short_amount_l799_79979

def rock_price : ℚ := 7
def pop_price : ℚ := 12
def dance_price : ℚ := 5
def country_price : ℚ := 9

def discount_rate : ℚ := 0.15
def discount_threshold : ℕ := 3

def rock_desired : ℕ := 5
def pop_desired : ℕ := 3
def dance_desired : ℕ := 6
def country_desired : ℕ := 4

def rock_available : ℕ := 4
def dance_available : ℕ := 5

def julia_budget : ℚ := 80

def calculate_genre_cost (price : ℚ) (desired : ℕ) (available : ℕ) : ℚ :=
  price * (min desired available : ℚ)

def apply_discount (cost : ℚ) (quantity : ℕ) : ℚ :=
  if quantity ≥ discount_threshold then cost * (1 - discount_rate) else cost

theorem julia_short_amount : 
  let rock_cost := calculate_genre_cost rock_price rock_desired rock_available
  let pop_cost := calculate_genre_cost pop_price pop_desired pop_desired
  let dance_cost := calculate_genre_cost dance_price dance_desired dance_available
  let country_cost := calculate_genre_cost country_price country_desired country_desired
  let total_cost := rock_cost + pop_cost + dance_cost + country_cost
  let discounted_rock := apply_discount rock_cost rock_available
  let discounted_pop := apply_discount pop_cost pop_desired
  let discounted_dance := apply_discount dance_cost dance_available
  let discounted_country := apply_discount country_cost country_desired
  let total_discounted := discounted_rock + discounted_pop + discounted_dance + discounted_country
  total_discounted - julia_budget = 26.25 := by
  sorry

end NUMINAMATH_CALUDE_julia_short_amount_l799_79979


namespace NUMINAMATH_CALUDE_not_square_or_cube_l799_79904

theorem not_square_or_cube (n : ℕ+) :
  ¬ ∃ (k m : ℤ), (n * (n + 1) * (n + 2) * (n + 3) : ℤ) = k^2 ∨
                 (n * (n + 1) * (n + 2) * (n + 3) : ℤ) = m^3 :=
by sorry

end NUMINAMATH_CALUDE_not_square_or_cube_l799_79904


namespace NUMINAMATH_CALUDE_phone_price_calculation_l799_79958

/-- Proves that given specific conditions on phone accessories and contract,
    the phone price that results in a total yearly cost of $3700 is $1000. -/
theorem phone_price_calculation (phone_price : ℝ) : 
  (∀ (monthly_contract case_cost headphones_cost : ℝ),
    monthly_contract = 200 ∧
    case_cost = 0.2 * phone_price ∧
    headphones_cost = 0.5 * case_cost ∧
    phone_price + 12 * monthly_contract + case_cost + headphones_cost = 3700) →
  phone_price = 1000 := by
  sorry

end NUMINAMATH_CALUDE_phone_price_calculation_l799_79958


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l799_79936

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l799_79936


namespace NUMINAMATH_CALUDE_areas_sum_equal_largest_l799_79914

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle where
  -- The sides of the triangle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  -- The areas of the non-triangular regions
  area_D : ℝ
  area_E : ℝ
  area_F : ℝ
  -- Conditions
  isosceles : side1 = side2
  sides : side1 = 12 ∧ side2 = 12 ∧ side3 = 20
  largest_F : area_F ≥ area_D ∧ area_F ≥ area_E

/-- Theorem stating that D + E = F for the given inscribed triangle -/
theorem areas_sum_equal_largest (t : InscribedTriangle) : t.area_D + t.area_E = t.area_F := by
  sorry

end NUMINAMATH_CALUDE_areas_sum_equal_largest_l799_79914


namespace NUMINAMATH_CALUDE_sum_inequality_l799_79916

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_sum_inequality_l799_79916


namespace NUMINAMATH_CALUDE_order_of_expressions_l799_79975

theorem order_of_expressions : 7^(0.3 : ℝ) > (0.3 : ℝ)^7 ∧ (0.3 : ℝ)^7 > Real.log 0.3 := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l799_79975


namespace NUMINAMATH_CALUDE_geometric_sum_n1_l799_79964

theorem geometric_sum_n1 (x : ℝ) (h : x ≠ 1) :
  1 + x + x^2 = (1 - x^3) / (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_n1_l799_79964


namespace NUMINAMATH_CALUDE_intersection_A_B_l799_79910

def A : Set ℝ := {-1, 0, 1}

def B : Set ℝ := {x : ℝ | x * (x - 1) ≤ 0}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l799_79910


namespace NUMINAMATH_CALUDE_triangle_side_length_l799_79995

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  2 * b = a + c →  -- arithmetic sequence condition
  B = π / 6 →  -- 30° in radians
  (1 / 2) * a * c * Real.sin B = 3 / 2 →  -- area condition
  b = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l799_79995


namespace NUMINAMATH_CALUDE_tea_consumption_l799_79939

/-- Represents the relationship between hours spent reading and liters of tea consumed -/
structure ReadingTeaData where
  hours : ℝ
  liters : ℝ

/-- The constant of proportionality for the inverse relationship -/
def proportionality_constant (data : ReadingTeaData) : ℝ :=
  data.hours * data.liters

theorem tea_consumption (wednesday thursday friday : ReadingTeaData)
  (h_wednesday : wednesday.hours = 8 ∧ wednesday.liters = 3)
  (h_thursday : thursday.hours = 5)
  (h_friday : friday.hours = 10)
  (h_inverse_prop : proportionality_constant wednesday = proportionality_constant thursday
                  ∧ proportionality_constant wednesday = proportionality_constant friday) :
  thursday.liters = 4.8 ∧ friday.liters = 2.4 := by
  sorry

#check tea_consumption

end NUMINAMATH_CALUDE_tea_consumption_l799_79939
