import Mathlib

namespace NUMINAMATH_CALUDE_diamond_equation_solution_l1821_182142

-- Define the diamond operation
def diamond (A B : ℝ) : ℝ := 4 * A + 3 * B * 2

-- Theorem statement
theorem diamond_equation_solution :
  ∃! A : ℝ, diamond A 5 = 64 ∧ A = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l1821_182142


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1821_182143

/-- Properties of a hyperbola -/
theorem hyperbola_properties (x y : ℝ) :
  (y^2 / 9 - x^2 / 16 = 1) →
  (∃ (imaginary_axis_length : ℝ) (asymptote_slope : ℝ) (focus_y : ℝ) (eccentricity : ℝ),
    imaginary_axis_length = 8 ∧
    asymptote_slope = 3/4 ∧
    focus_y = 5 ∧
    eccentricity = 5/3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1821_182143


namespace NUMINAMATH_CALUDE_tank_fill_time_l1821_182130

def fill_time_A : ℝ := 30
def fill_rate_B_multiplier : ℝ := 5

theorem tank_fill_time :
  let fill_time_B := fill_time_A / fill_rate_B_multiplier
  let fill_rate_A := 1 / fill_time_A
  let fill_rate_B := 1 / fill_time_B
  let combined_fill_rate := fill_rate_A + fill_rate_B
  1 / combined_fill_rate = 5 := by sorry

end NUMINAMATH_CALUDE_tank_fill_time_l1821_182130


namespace NUMINAMATH_CALUDE_bean_game_uniqueness_l1821_182148

/-- Represents the state of beans on an infinite row of squares -/
def BeanState := ℤ → ℕ

/-- Represents a single move in the bean game -/
def Move := ℤ

/-- Applies a move to a given state -/
def applyMove (state : BeanState) (move : Move) : BeanState :=
  sorry

/-- Checks if a state is terminal (no square has more than one bean) -/
def isTerminal (state : BeanState) : Prop :=
  ∀ i : ℤ, state i ≤ 1

/-- Represents a sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to an initial state -/
def applyMoveSequence (initial : BeanState) (moves : MoveSequence) : BeanState :=
  sorry

/-- The final state after applying a sequence of moves -/
def finalState (initial : BeanState) (moves : MoveSequence) : BeanState :=
  applyMoveSequence initial moves

/-- The number of steps (moves) in a sequence -/
def numSteps (moves : MoveSequence) : ℕ :=
  moves.length

/-- Theorem: All valid move sequences result in the same final state and number of steps -/
theorem bean_game_uniqueness (initial : BeanState) 
    (moves1 moves2 : MoveSequence) 
    (h1 : isTerminal (finalState initial moves1))
    (h2 : isTerminal (finalState initial moves2)) :
    finalState initial moves1 = finalState initial moves2 ∧ 
    numSteps moves1 = numSteps moves2 :=
  sorry

end NUMINAMATH_CALUDE_bean_game_uniqueness_l1821_182148


namespace NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_five_l1821_182133

def numbers : List Nat := [3546, 3550, 3565, 3570, 3585]

def is_divisible_by_five (n : Nat) : Bool :=
  n % 5 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_not_divisible_by_five :
  ∃ n ∈ numbers, ¬is_divisible_by_five n ∧
  units_digit n * tens_digit n = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_five_l1821_182133


namespace NUMINAMATH_CALUDE_min_distance_complex_l1821_182188

/-- Given a complex number z satisfying |z + 3i| = 1, 
    the minimum value of |z - 1 + 2i| is √2 - 1. -/
theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 3 * Complex.I) = 1) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 2 - 1 ∧
  ∀ (w : ℂ), Complex.abs (w + 3 * Complex.I) = 1 →
  Complex.abs (w - 1 + 2 * Complex.I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l1821_182188


namespace NUMINAMATH_CALUDE_construction_time_for_330_meters_l1821_182176

/-- Represents the daily progress of road construction in meters -/
def daily_progress : ℕ := 30

/-- Calculates the cumulative progress given the number of days -/
def cumulative_progress (days : ℕ) : ℕ :=
  daily_progress * days

/-- Theorem stating that 330 meters of cumulative progress corresponds to 11 days of construction -/
theorem construction_time_for_330_meters :
  cumulative_progress 11 = 330 ∧ cumulative_progress 10 ≠ 330 := by
  sorry

end NUMINAMATH_CALUDE_construction_time_for_330_meters_l1821_182176


namespace NUMINAMATH_CALUDE_slope_intercept_form_through_points_l1821_182194

/-- Slope-intercept form of a line passing through two points -/
theorem slope_intercept_form_through_points
  (x₁ y₁ x₂ y₂ : ℚ)
  (h₁ : x₁ = -3)
  (h₂ : y₁ = 7)
  (h₃ : x₂ = 4)
  (h₄ : y₂ = -2)
  : ∃ (m b : ℚ), m = -9/7 ∧ b = 22/7 ∧ ∀ x y, y = m * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) :=
by sorry

end NUMINAMATH_CALUDE_slope_intercept_form_through_points_l1821_182194


namespace NUMINAMATH_CALUDE_parabola_square_intersection_l1821_182108

/-- A parabola y = px^2 has a common point with the square defined by vertices A(1,1), B(2,1), C(2,2), and D(1,2) if and only if 1/4 ≤ p ≤ 2 -/
theorem parabola_square_intersection (p : ℝ) : 
  (∃ x y : ℝ, y = p * x^2 ∧ 
    ((x = 1 ∧ y = 1) ∨ 
     (x = 2 ∧ y = 1) ∨ 
     (x = 2 ∧ y = 2) ∨ 
     (x = 1 ∧ y = 2) ∨
     (1 ≤ x ∧ x ≤ 2 ∧ y = 1) ∨
     (x = 2 ∧ 1 ≤ y ∧ y ≤ 2) ∨
     (1 ≤ x ∧ x ≤ 2 ∧ y = 2) ∨
     (x = 1 ∧ 1 ≤ y ∧ y ≤ 2))) ↔ 
  (1/4 : ℝ) ≤ p ∧ p ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_square_intersection_l1821_182108


namespace NUMINAMATH_CALUDE_sum_equals_four_sqrt_860_l1821_182154

theorem sum_equals_four_sqrt_860 (a b c d : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_squares : a^2 + b^2 = 2016 ∧ c^2 + d^2 = 2016)
  (products : a*c = 1008 ∧ b*d = 1008) :
  a + b + c + d = 4 * Real.sqrt 860 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_four_sqrt_860_l1821_182154


namespace NUMINAMATH_CALUDE_f_2005_of_2_pow_2006_l1821_182167

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- f₁(k) is the square of the sum of digits of k -/
def f₁ (k : ℕ) : ℕ := (sum_of_digits k) ^ 2

/-- fₙ₊₁(k) = f₁(fₙ(k)) for n ≥ 1 -/
def f (n : ℕ) (k : ℕ) : ℕ :=
  match n with
  | 0 => k
  | n + 1 => f₁ (f n k)

/-- The main theorem to prove -/
theorem f_2005_of_2_pow_2006 : f 2005 (2^2006) = 169 := by sorry

end NUMINAMATH_CALUDE_f_2005_of_2_pow_2006_l1821_182167


namespace NUMINAMATH_CALUDE_smallest_product_l1821_182161

def S : Finset Int := {-10, -4, 0, 2, 6}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int) (hx : x ∈ S) (hy : y ∈ S), x * y ≤ a * b ∧ x * y = -60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_l1821_182161


namespace NUMINAMATH_CALUDE_elliot_book_pages_left_l1821_182192

def pages_left_after_week (total_pages : ℕ) (pages_read : ℕ) (pages_per_day : ℕ) (days : ℕ) : ℕ :=
  total_pages - pages_read - (pages_per_day * days)

theorem elliot_book_pages_left : pages_left_after_week 381 149 20 7 = 92 := by
  sorry

end NUMINAMATH_CALUDE_elliot_book_pages_left_l1821_182192


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l1821_182140

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : ∀ x y, x < y → f x > f y) 
  (h_inequality : f (1 - a) < f (2 * a - 1)) : 
  a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l1821_182140


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_div_by_5_l1821_182104

/-- Converts a base 8 number to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Checks if a number is a 3-digit base 8 number -/
def is_3digit_base8 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_div_by_5 :
  ∀ n : ℕ, is_3digit_base8 n → (base8_to_base10 n) % 5 = 0 → n ≤ 776 :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base8_div_by_5_l1821_182104


namespace NUMINAMATH_CALUDE_monkey_climbing_theorem_l1821_182111

/-- Represents the climbing problem of a monkey on a tree -/
structure ClimbingProblem where
  treeHeight : ℕ
  climbRate : ℕ
  slipRate : ℕ
  restPeriod : ℕ
  restDuration : ℕ

/-- Calculates the time taken for the monkey to reach the top of the tree -/
def timeTakenToClimb (problem : ClimbingProblem) : ℕ :=
  sorry

/-- The theorem stating the solution to the specific climbing problem -/
theorem monkey_climbing_theorem :
  let problem : ClimbingProblem := {
    treeHeight := 253,
    climbRate := 7,
    slipRate := 4,
    restPeriod := 4,
    restDuration := 1
  }
  timeTakenToClimb problem = 109 := by
  sorry

end NUMINAMATH_CALUDE_monkey_climbing_theorem_l1821_182111


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1821_182136

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1821_182136


namespace NUMINAMATH_CALUDE_factor_36_minus_9x_squared_l1821_182184

theorem factor_36_minus_9x_squared (x : ℝ) : 36 - 9 * x^2 = 9 * (2 - x) * (2 + x) := by
  sorry

end NUMINAMATH_CALUDE_factor_36_minus_9x_squared_l1821_182184


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1821_182155

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N :
  M ∩ N = {(3, -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1821_182155


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1821_182165

theorem fraction_equation_solution :
  ∃! x : ℚ, (x - 3) / (x + 2) + (3*x - 9) / (x - 3) = 2 ∧ x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1821_182165


namespace NUMINAMATH_CALUDE_mayo_bottles_count_l1821_182116

/-- Given a ratio of ketchup : mustard : mayo bottles as 3 : 3 : 2, 
    and 6 ketchup bottles, prove that there are 4 mayo bottles. -/
theorem mayo_bottles_count 
  (ratio_ketchup : ℕ) 
  (ratio_mustard : ℕ) 
  (ratio_mayo : ℕ) 
  (ketchup_bottles : ℕ) 
  (h_ratio : ratio_ketchup = 3 ∧ ratio_mustard = 3 ∧ ratio_mayo = 2)
  (h_ketchup : ketchup_bottles = 6) : 
  ketchup_bottles * ratio_mayo / ratio_ketchup = 4 := by
sorry


end NUMINAMATH_CALUDE_mayo_bottles_count_l1821_182116


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1821_182126

theorem quadratic_inequality (x : ℝ) : x^2 < x + 6 ↔ -2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1821_182126


namespace NUMINAMATH_CALUDE_quadratic_intersection_l1821_182171

theorem quadratic_intersection
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0) :
  ∃! p : ℝ × ℝ,
    (λ x y => y = a * x^2 + b * x + c) p.1 p.2 ∧
    (λ x y => y = a * x^2 - b * x + c + d) p.1 p.2 ∧
    p.1 ≠ 0 ∧ p.2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l1821_182171


namespace NUMINAMATH_CALUDE_min_absolute_value_complex_l1821_182122

theorem min_absolute_value_complex (z : ℂ) (h : Complex.abs (z - 10) + Complex.abs (z + 3*I) = 15) :
  ∃ (w : ℂ), Complex.abs w = 2 ∧ ∀ (v : ℂ), (Complex.abs (v - 10) + Complex.abs (v + 3*I) = 15) → Complex.abs v ≥ Complex.abs w :=
by sorry

end NUMINAMATH_CALUDE_min_absolute_value_complex_l1821_182122


namespace NUMINAMATH_CALUDE_power_sum_problem_l1821_182135

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 25)
  (h4 : a * x^4 + b * y^4 = 59) :
  a * x^5 + b * y^5 = 145 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_problem_l1821_182135


namespace NUMINAMATH_CALUDE_luncheon_seating_capacity_l1821_182151

theorem luncheon_seating_capacity 
  (invited : ℕ) 
  (no_shows : ℕ) 
  (tables : ℕ) 
  (h1 : invited = 47) 
  (h2 : no_shows = 7) 
  (h3 : tables = 8) :
  (invited - no_shows) / tables = 5 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_seating_capacity_l1821_182151


namespace NUMINAMATH_CALUDE_la_retail_women_ratio_l1821_182139

/-- The ratio of women working in retail to the total number of women in Los Angeles -/
def retail_women_ratio (total_population : ℕ) (women_population : ℕ) (retail_women : ℕ) : ℚ :=
  retail_women / women_population

theorem la_retail_women_ratio :
  let total_population : ℕ := 6000000
  let women_population : ℕ := total_population / 2
  let retail_women : ℕ := 1000000
  retail_women_ratio total_population women_population retail_women = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_la_retail_women_ratio_l1821_182139


namespace NUMINAMATH_CALUDE_medicine_container_problem_l1821_182150

theorem medicine_container_problem (initial_volume : ℝ) (remaining_volume : ℝ) : 
  initial_volume = 63 ∧ remaining_volume = 28 →
  ∃ (x : ℝ), x = 18 ∧ 
    initial_volume * (1 - x / initial_volume) * (1 - x / initial_volume) = remaining_volume :=
by sorry

end NUMINAMATH_CALUDE_medicine_container_problem_l1821_182150


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1821_182100

/-- A linear function that passes through the first, third, and fourth quadrants -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k + 1) * x + k - 2

/-- Condition for the function to have a positive slope -/
def positive_slope (k : ℝ) : Prop := k + 1 > 0

/-- Condition for the y-intercept to be negative -/
def negative_y_intercept (k : ℝ) : Prop := k - 2 < 0

/-- Theorem stating the range of k for the linear function to pass through the first, third, and fourth quadrants -/
theorem linear_function_quadrants (k : ℝ) : 
  (∀ x, ∃ y, y = linear_function k x) ∧ 
  positive_slope k ∧ 
  negative_y_intercept k ↔ 
  -1 < k ∧ k < 2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l1821_182100


namespace NUMINAMATH_CALUDE_car_dealership_hourly_wage_l1821_182174

/-- Calculates the hourly wage for employees in a car dealership --/
theorem car_dealership_hourly_wage :
  let fiona_weekly_hours : ℕ := 40
  let john_weekly_hours : ℕ := 30
  let jeremy_weekly_hours : ℕ := 25
  let weeks_per_month : ℕ := 4
  let total_monthly_pay : ℕ := 7600

  let total_monthly_hours : ℕ := 
    (fiona_weekly_hours + john_weekly_hours + jeremy_weekly_hours) * weeks_per_month

  (total_monthly_pay : ℚ) / total_monthly_hours = 20 := by
  sorry


end NUMINAMATH_CALUDE_car_dealership_hourly_wage_l1821_182174


namespace NUMINAMATH_CALUDE_midpoint_to_directrix_distance_l1821_182170

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define the point P
def point_P : ℝ × ℝ := (2, 0)

-- Define the directrix of parabola C
def directrix_C : ℝ := -3

-- Theorem statement
theorem midpoint_to_directrix_distance :
  ∃ (A B : ℝ × ℝ),
    parabola_C A.1 A.2 ∧
    parabola_C B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    (A.1 + B.1) / 2 - directrix_C = 11 :=
sorry

end NUMINAMATH_CALUDE_midpoint_to_directrix_distance_l1821_182170


namespace NUMINAMATH_CALUDE_ceiling_minus_value_l1821_182132

theorem ceiling_minus_value (x ε : ℝ) 
  (h1 : ⌈x + ε⌉ - ⌊x + ε⌋ = 1) 
  (h2 : 0 < ε) 
  (h3 : ε < 1) : 
  ⌈x + ε⌉ - (x + ε) = 1 - ε := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_value_l1821_182132


namespace NUMINAMATH_CALUDE_length_of_cd_l1821_182103

/-- Given points R and S on line segment CD, where R divides CD in the ratio 3:5,
    S divides CD in the ratio 4:7, and RS = 3, the length of CD is 264. -/
theorem length_of_cd (C D R S : Real) : 
  (∃ m n : Real, C + m = R ∧ R + n = D ∧ m / n = 3 / 5) →  -- R divides CD in ratio 3:5
  (∃ p q : Real, C + p = S ∧ S + q = D ∧ p / q = 4 / 7) →  -- S divides CD in ratio 4:7
  (S - R = 3) →                                            -- RS = 3
  (D - C = 264) :=                                         -- Length of CD is 264
by sorry

end NUMINAMATH_CALUDE_length_of_cd_l1821_182103


namespace NUMINAMATH_CALUDE_point_on_circle_l1821_182149

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Check if a point lies on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  squaredDistance p c.center = c.radius^2

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- The given point P(-3,4) -/
def pointP : Point := ⟨-3, 4⟩

/-- The circle with center at origin and radius 5 -/
def circleO : Circle := ⟨origin, 5⟩

theorem point_on_circle : isOnCircle pointP circleO := by
  sorry

end NUMINAMATH_CALUDE_point_on_circle_l1821_182149


namespace NUMINAMATH_CALUDE_max_value_of_s_l1821_182178

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 22) :
  s ≤ (5 + Real.sqrt 93) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_s_l1821_182178


namespace NUMINAMATH_CALUDE_medication_forgotten_days_l1821_182114

theorem medication_forgotten_days (total_days : ℕ) (taken_days : ℕ) : 
  total_days = 31 → taken_days = 29 → total_days - taken_days = 2 := by
  sorry

end NUMINAMATH_CALUDE_medication_forgotten_days_l1821_182114


namespace NUMINAMATH_CALUDE_sqrt_10_greater_than_3_l1821_182112

theorem sqrt_10_greater_than_3 : Real.sqrt 10 > 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_10_greater_than_3_l1821_182112


namespace NUMINAMATH_CALUDE_frustum_small_cone_height_l1821_182158

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  height : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the height of the small cone removed from a frustum -/
def small_cone_height (f : Frustum) : ℝ :=
  f.height

/-- Theorem: The height of the small cone removed from a frustum with given dimensions is 30 cm -/
theorem frustum_small_cone_height :
  ∀ (f : Frustum),
    f.height = 30 ∧
    f.lower_base_area = 400 * Real.pi ∧
    f.upper_base_area = 100 * Real.pi →
    small_cone_height f = 30 := by
  sorry

end NUMINAMATH_CALUDE_frustum_small_cone_height_l1821_182158


namespace NUMINAMATH_CALUDE_pascal_triangle_52nd_number_l1821_182164

/-- The number of elements in the row of Pascal's triangle we're considering --/
def row_length : ℕ := 55

/-- The index of the number we're looking for in the row (0-indexed) --/
def target_index : ℕ := 51

/-- The row number in Pascal's triangle (0-indexed) --/
def row_number : ℕ := row_length - 1

/-- The binomial coefficient we need to calculate --/
def pascal_number : ℕ := Nat.choose row_number target_index

theorem pascal_triangle_52nd_number : pascal_number = 24804 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_52nd_number_l1821_182164


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1821_182120

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z = 9 * x * y * z) :
  x / Real.sqrt (x^2 + 2*y*z + 2) + y / Real.sqrt (y^2 + 2*z*x + 2) + z / Real.sqrt (z^2 + 2*x*y + 2) ≥ 1 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z = 9 * x * y * z) :
  x / Real.sqrt (x^2 + 2*y*z + 2) + y / Real.sqrt (y^2 + 2*z*x + 2) + z / Real.sqrt (z^2 + 2*x*y + 2) = 1 ↔
  x = y ∧ y = z ∧ x = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1821_182120


namespace NUMINAMATH_CALUDE_investment_problem_l1821_182185

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem statement for the investment problem -/
theorem investment_problem (P : ℝ) :
  (∃ r : ℝ, simple_interest P r 2 = 520 ∧ simple_interest P r 7 = 820) →
  P = 400 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l1821_182185


namespace NUMINAMATH_CALUDE_harkamal_purchase_l1821_182134

/-- The total cost of a purchase given the quantity and price per unit -/
def totalCost (quantity : ℕ) (pricePerUnit : ℕ) : ℕ :=
  quantity * pricePerUnit

theorem harkamal_purchase : 
  let grapeQuantity : ℕ := 8
  let grapePrice : ℕ := 70
  let mangoQuantity : ℕ := 9
  let mangoPrice : ℕ := 60
  totalCost grapeQuantity grapePrice + totalCost mangoQuantity mangoPrice = 1100 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_purchase_l1821_182134


namespace NUMINAMATH_CALUDE_optimal_rent_and_income_l1821_182181

def daily_net_income (rent : ℕ) : ℤ :=
  if rent ≤ 6 then
    50 * rent - 115
  else
    (50 - 3 * (rent - 6)) * rent - 115

def is_valid_rent (rent : ℕ) : Prop :=
  3 ≤ rent ∧ rent ≤ 20 ∧ daily_net_income rent > 0

theorem optimal_rent_and_income :
  ∃ (optimal_rent : ℕ) (max_income : ℤ),
    is_valid_rent optimal_rent ∧
    max_income = daily_net_income optimal_rent ∧
    optimal_rent = 11 ∧
    max_income = 270 ∧
    ∀ (rent : ℕ), is_valid_rent rent → daily_net_income rent ≤ max_income :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_rent_and_income_l1821_182181


namespace NUMINAMATH_CALUDE_f_is_decreasing_and_odd_l1821_182187

-- Define the function f(x) = -x
def f (x : ℝ) : ℝ := -x

-- State the theorem
theorem f_is_decreasing_and_odd :
  (∀ x y : ℝ, x < y → f x > f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_f_is_decreasing_and_odd_l1821_182187


namespace NUMINAMATH_CALUDE_alvin_marbles_l1821_182137

def marble_game (initial : ℕ) (game1 : ℤ) (game2 : ℤ) (game3 : ℤ) (game4 : ℤ) (give : ℕ) (receive : ℕ) : ℕ :=
  (initial : ℤ) + game1 + game2 + game3 + game4 - give + receive |>.toNat

theorem alvin_marbles : 
  marble_game 57 (-18) 25 (-12) 15 10 8 = 65 := by
  sorry

end NUMINAMATH_CALUDE_alvin_marbles_l1821_182137


namespace NUMINAMATH_CALUDE_conic_sections_identification_l1821_182115

/-- The equation y^4 - 9x^4 = 3y^2 - 3 represents the union of a hyperbola and an ellipse -/
theorem conic_sections_identification (x y : ℝ) : 
  (y^4 - 9*x^4 = 3*y^2 - 3) ↔ 
  ((y^2 - 3*x^2 = 3/2) ∨ (y^2 + 3*x^2 = 3/2)) :=
sorry

end NUMINAMATH_CALUDE_conic_sections_identification_l1821_182115


namespace NUMINAMATH_CALUDE_periodic_function_smallest_period_l1821_182117

/-- A function satisfying the given periodic property -/
def PeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) + f (x - 4) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x

/-- The smallest positive period of a function -/
def SmallestPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  IsPeriod f T ∧ T > 0 ∧ ∀ S : ℝ, IsPeriod f S ∧ S > 0 → T ≤ S

/-- The main theorem stating that functions satisfying the given condition have a smallest period of 24 -/
theorem periodic_function_smallest_period (f : ℝ → ℝ) (h : PeriodicFunction f) :
    SmallestPeriod f 24 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_smallest_period_l1821_182117


namespace NUMINAMATH_CALUDE_increasing_order_x_z_y_l1821_182169

theorem increasing_order_x_z_y (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  x < x^(x^x) ∧ x^(x^x) < x^x := by
  sorry

end NUMINAMATH_CALUDE_increasing_order_x_z_y_l1821_182169


namespace NUMINAMATH_CALUDE_right_triangle_properties_l1821_182110

/-- Right triangle ABC with given properties -/
structure RightTriangleABC where
  -- AB is the hypotenuse
  AB : ℝ
  BC : ℝ
  angleC : ℝ
  hypotenuse_length : AB = 5
  leg_length : BC = 3
  right_angle : angleC = 90

/-- The length of the altitude to the hypotenuse in the right triangle ABC -/
def altitude_to_hypotenuse (t : RightTriangleABC) : ℝ := 2.4

/-- The length of the median to the hypotenuse in the right triangle ABC -/
def median_to_hypotenuse (t : RightTriangleABC) : ℝ := 2.5

/-- Theorem stating the properties of the right triangle ABC -/
theorem right_triangle_properties (t : RightTriangleABC) :
  altitude_to_hypotenuse t = 2.4 ∧ median_to_hypotenuse t = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l1821_182110


namespace NUMINAMATH_CALUDE_integral_equals_eighteen_implies_a_equals_three_l1821_182102

theorem integral_equals_eighteen_implies_a_equals_three (a : ℝ) :
  (∫ (x : ℝ) in -a..a, x^2 + Real.sin x) = 18 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_eighteen_implies_a_equals_three_l1821_182102


namespace NUMINAMATH_CALUDE_max_divisor_of_f_l1821_182113

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_of_f :
  ∃ (m : ℕ), m = 36 ∧ 
  (∀ (n : ℕ), n > 0 → m ∣ f n) ∧
  (∀ (k : ℕ), k > 36 → ∃ (n : ℕ), n > 0 ∧ ¬(k ∣ f n)) :=
sorry

end NUMINAMATH_CALUDE_max_divisor_of_f_l1821_182113


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l1821_182119

/-- Calculates the total amount spent at a restaurant given the prices of items,
    discount rates, service fee rates, and tipping percentage. -/
def restaurant_bill (seafood_price rib_eye_price wine_price dessert_price : ℚ)
                    (wine_quantity : ℕ)
                    (food_discount service_fee_low service_fee_high tip_rate : ℚ) : ℚ :=
  let food_cost := seafood_price + rib_eye_price + dessert_price
  let wine_cost := wine_price * wine_quantity
  let total_before_discount := food_cost + wine_cost
  let discounted_food_cost := food_cost * (1 - food_discount)
  let after_discount := discounted_food_cost + wine_cost
  let service_fee_rate := if after_discount > 80 then service_fee_high else service_fee_low
  let service_fee := after_discount * service_fee_rate
  let total_after_service := after_discount + service_fee
  let tip := total_after_service * tip_rate
  total_after_service + tip

/-- The theorem states that given the specific prices and rates from the problem,
    the total amount spent at the restaurant is $167.67. -/
theorem restaurant_bill_calculation :
  restaurant_bill 45 38 18 12 2 0.1 0.12 0.15 0.2 = 167.67 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l1821_182119


namespace NUMINAMATH_CALUDE_inequality_proof_l1821_182179

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) :
  Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) ≤ Real.sqrt (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1821_182179


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1821_182146

theorem condition_necessary_not_sufficient (a b : ℝ) :
  (∃ a b : ℝ, (a > 0 ∨ b > 0) ∧ ¬(a + b > 0 ∧ a * b > 0)) ∧
  (∀ a b : ℝ, (a + b > 0 ∧ a * b > 0) → (a > 0 ∨ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1821_182146


namespace NUMINAMATH_CALUDE_fish_tank_water_l1821_182177

theorem fish_tank_water (initial_water : ℝ) (added_water : ℝ) :
  initial_water = 7.75 →
  added_water = 7 →
  initial_water + added_water = 14.75 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_tank_water_l1821_182177


namespace NUMINAMATH_CALUDE_election_theorem_l1821_182180

/-- Represents the number of boys in the club -/
def num_boys : ℕ := 15

/-- Represents the number of girls in the club -/
def num_girls : ℕ := 15

/-- Represents the total number of members in the club -/
def total_members : ℕ := num_boys + num_girls

/-- Calculates the number of ways to elect officials under the given constraints -/
def election_ways : ℕ := 2 * num_boys * num_girls * (num_boys - 1)

/-- Theorem stating the number of ways to elect officials -/
theorem election_theorem : election_ways = 6300 := by sorry

end NUMINAMATH_CALUDE_election_theorem_l1821_182180


namespace NUMINAMATH_CALUDE_binomial_factorial_product_l1821_182197

theorem binomial_factorial_product : Nat.choose 20 6 * Nat.factorial 6 = 27907200 := by
  sorry

end NUMINAMATH_CALUDE_binomial_factorial_product_l1821_182197


namespace NUMINAMATH_CALUDE_units_digit_3_pow_34_l1821_182147

def units_digit (n : ℕ) : ℕ := n % 10

def power_3_cycle (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur due to the modulo operation

theorem units_digit_3_pow_34 :
  units_digit (3^34) = 9 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_3_pow_34_l1821_182147


namespace NUMINAMATH_CALUDE_original_triangle_area_l1821_182196

theorem original_triangle_area (original_area new_area : ℝ) : 
  (new_area = 32) → 
  (new_area = 4 * original_area) → 
  (original_area = 8) := by
sorry

end NUMINAMATH_CALUDE_original_triangle_area_l1821_182196


namespace NUMINAMATH_CALUDE_evaluate_expression_l1821_182124

theorem evaluate_expression : 3 + (-3)^2 = 12 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1821_182124


namespace NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l1821_182109

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℕ → ℝ :=
  fun k => a₁ + (k - 1 : ℝ) * d

theorem third_term_of_arithmetic_sequence :
  ∀ (a₁ aₙ : ℝ) (n : ℕ),
  n = 10 →
  a₁ = 5 →
  aₙ = 32 →
  let d := (aₙ - a₁) / (n - 1 : ℝ)
  let seq := arithmetic_sequence a₁ d n
  seq 3 = 11 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l1821_182109


namespace NUMINAMATH_CALUDE_cassie_water_refills_l1821_182193

/-- Represents the number of cups of water Cassie aims to drink daily -/
def daily_cups : ℕ := 12

/-- Represents the capacity of Cassie's water bottle in ounces -/
def bottle_capacity : ℕ := 16

/-- Represents the number of ounces in a cup -/
def ounces_per_cup : ℕ := 8

/-- Calculates the number of times Cassie needs to refill her water bottle -/
def refills_needed : ℕ := (daily_cups * ounces_per_cup) / bottle_capacity

theorem cassie_water_refills :
  refills_needed = 6 :=
sorry

end NUMINAMATH_CALUDE_cassie_water_refills_l1821_182193


namespace NUMINAMATH_CALUDE_polar_line_properties_l1821_182144

/-- A line in polar coordinates passing through (4,0) and perpendicular to the polar axis -/
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 4

theorem polar_line_properties (ρ θ : ℝ) :
  polar_line ρ θ →
  (ρ * Real.cos θ = 4 ∧ ρ * Real.sin θ = 0) ∧
  (∀ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ → x = 4) :=
sorry

end NUMINAMATH_CALUDE_polar_line_properties_l1821_182144


namespace NUMINAMATH_CALUDE_inequality_implication_l1821_182159

theorem inequality_implication (a b c d : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) 
  (h : a + b * Real.sqrt 5 < c + d * Real.sqrt 5) : 
  a < c ∧ b < d := by sorry

end NUMINAMATH_CALUDE_inequality_implication_l1821_182159


namespace NUMINAMATH_CALUDE_trapezium_side_length_l1821_182160

/-- Given a trapezium with the specified properties, prove that the length of the unknown parallel side is 20 cm. -/
theorem trapezium_side_length 
  (known_side : ℝ) 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : known_side = 18) 
  (h2 : height = 12) 
  (h3 : area = 228) : 
  ∃ unknown_side : ℝ, 
    area = (1/2) * (known_side + unknown_side) * height ∧ 
    unknown_side = 20 :=
sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l1821_182160


namespace NUMINAMATH_CALUDE_quadratic_order_l1821_182101

theorem quadratic_order (m y₁ y₂ y₃ : ℝ) (hm : m < -2) 
  (h₁ : y₁ = (m - 1)^2 + 2*(m - 1))
  (h₂ : y₂ = m^2 + 2*m)
  (h₃ : y₃ = (m + 1)^2 + 2*(m + 1)) :
  y₃ < y₂ ∧ y₂ < y₁ := by
sorry

end NUMINAMATH_CALUDE_quadratic_order_l1821_182101


namespace NUMINAMATH_CALUDE_largest_common_value_less_than_500_l1821_182127

theorem largest_common_value_less_than_500 
  (ap1 : ℕ → ℕ) 
  (ap2 : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, ap1 n = 5 + 4 * n) 
  (h2 : ∀ n : ℕ, ap2 n = 7 + 8 * n) : 
  (∃ k : ℕ, k < 500 ∧ (∃ n m : ℕ, ap1 n = k ∧ ap2 m = k)) ∧
  (∀ l : ℕ, l < 500 → (∃ n m : ℕ, ap1 n = l ∧ ap2 m = l) → l ≤ 497) ∧
  (∃ n m : ℕ, ap1 n = 497 ∧ ap2 m = 497) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_value_less_than_500_l1821_182127


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1821_182123

/-- The number of ways to arrange books on a shelf --/
def arrange_books (n_science : ℕ) (n_literature : ℕ) : ℕ :=
  n_literature * (n_literature - 1) * Nat.factorial (n_science + n_literature - 2)

/-- Theorem stating the number of arrangements for the given problem --/
theorem book_arrangement_count :
  arrange_books 4 5 = 10080 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1821_182123


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1821_182145

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1821_182145


namespace NUMINAMATH_CALUDE_event_C_subset_event_B_l1821_182121

-- Define the sample space for tossing 3 coins
def SampleSpace := List Bool

-- Define the events A, B, and C
def event_A (outcome : SampleSpace) : Prop := outcome.contains true
def event_B (outcome : SampleSpace) : Prop := outcome.count true ≤ 2
def event_C (outcome : SampleSpace) : Prop := outcome.count true = 0

-- Theorem statement
theorem event_C_subset_event_B : 
  ∀ (outcome : SampleSpace), event_C outcome → event_B outcome :=
by
  sorry


end NUMINAMATH_CALUDE_event_C_subset_event_B_l1821_182121


namespace NUMINAMATH_CALUDE_sum_pairwise_ratios_lower_bound_l1821_182157

theorem sum_pairwise_ratios_lower_bound {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_pairwise_ratios_lower_bound_l1821_182157


namespace NUMINAMATH_CALUDE_sin_m_equals_cos_714_l1821_182162

theorem sin_m_equals_cos_714 (m : ℤ) :
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.cos (714 * π / 180) →
  m = 96 ∨ m = 84 := by
  sorry

end NUMINAMATH_CALUDE_sin_m_equals_cos_714_l1821_182162


namespace NUMINAMATH_CALUDE_condition_one_condition_two_l1821_182186

-- Define the lines l₁ and l₂
def l₁ (a b : ℝ) (x y : ℝ) : Prop := a * x - b * y + 4 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y + b = 0

-- Define perpendicularity condition
def perpendicular (a b : ℝ) : Prop := a * (a - 1) - b = 0

-- Define parallel condition
def parallel (a b : ℝ) : Prop := a * (a - 1) + b = 0

-- Define the condition that l₁ passes through (-3, -1)
def passes_through (a b : ℝ) : Prop := l₁ a b (-3) (-1)

-- Define the condition that intercepts are equal
def equal_intercepts (a b : ℝ) : Prop := b = -a

theorem condition_one (a b : ℝ) :
  perpendicular a b ∧ passes_through a b → a = 2 ∧ b = 2 :=
by sorry

theorem condition_two (a b : ℝ) :
  parallel a b ∧ equal_intercepts a b → a = 2 ∧ b = -2 :=
by sorry

end NUMINAMATH_CALUDE_condition_one_condition_two_l1821_182186


namespace NUMINAMATH_CALUDE_garage_sale_dvd_average_price_l1821_182182

/-- Calculate the average price of DVDs bought at a garage sale --/
theorem garage_sale_dvd_average_price : 
  let box1_count : ℕ := 10
  let box1_price : ℚ := 2
  let box2_count : ℕ := 5
  let box2_price : ℚ := 5
  let box3_count : ℕ := 3
  let box3_price : ℚ := 7
  let box4_count : ℕ := 4
  let box4_price : ℚ := 7/2
  let discount_rate : ℚ := 15/100
  let tax_rate : ℚ := 10/100
  let total_count : ℕ := box1_count + box2_count + box3_count + box4_count
  let total_cost : ℚ := 
    box1_count * box1_price + 
    box2_count * box2_price + 
    box3_count * box3_price + 
    box4_count * box4_price
  let discounted_cost : ℚ := total_cost * (1 - discount_rate)
  let final_cost : ℚ := discounted_cost * (1 + tax_rate)
  let average_price : ℚ := final_cost / total_count
  average_price = 17/5 := by sorry

end NUMINAMATH_CALUDE_garage_sale_dvd_average_price_l1821_182182


namespace NUMINAMATH_CALUDE_store_comparison_l1821_182173

/-- Represents the cost function for store A -/
def cost_A (x : ℕ) : ℝ :=
  if x = 0 then 0 else 140 * x + 60

/-- Represents the cost function for store B -/
def cost_B (x : ℕ) : ℝ := 150 * x

theorem store_comparison (x : ℕ) (h : x ≥ 1) :
  (cost_A x = 140 * x + 60) ∧
  (cost_B x = 150 * x) ∧
  (∀ y : ℕ, 1 ≤ y ∧ y < 6 → cost_A y < cost_B y) ∧
  (∀ z : ℕ, z > 6 → cost_A z > cost_B z) :=
by sorry

end NUMINAMATH_CALUDE_store_comparison_l1821_182173


namespace NUMINAMATH_CALUDE_sixth_train_departure_l1821_182190

def train_departure_time (start_time : Nat) (interval : Nat) (n : Nat) : Nat :=
  start_time + (n - 1) * interval

theorem sixth_train_departure :
  let start_time := 10 * 60  -- 10:00 AM in minutes
  let interval := 30         -- 30 minutes
  let sixth_train := 6
  train_departure_time start_time interval sixth_train = 12 * 60 + 30  -- 12:30 PM in minutes
  := by sorry

end NUMINAMATH_CALUDE_sixth_train_departure_l1821_182190


namespace NUMINAMATH_CALUDE_product_correction_l1821_182152

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem product_correction (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (reverse_digits a * b = 221) →  -- reversed a times b is 221
  (a * b = 527 ∨ a * b = 923) :=  -- correct product is 527 or 923
by sorry

end NUMINAMATH_CALUDE_product_correction_l1821_182152


namespace NUMINAMATH_CALUDE_line_L_equation_l1821_182131

-- Define the point A
def A : ℝ × ℝ := (2, 4)

-- Define the parallel lines
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the line on which the midpoint lies
def midpoint_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the equation of line L
def line_L (x y : ℝ) : Prop := 3*x - y - 2 = 0

-- Theorem statement
theorem line_L_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- L passes through A
    line_L 2 4 ∧
    -- L intersects the parallel lines
    line1 x₁ y₁ ∧ line2 x₂ y₂ ∧ line_L x₁ y₁ ∧ line_L x₂ y₂ ∧
    -- Midpoint of the segment lies on the given line
    midpoint_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) :=
by sorry

end NUMINAMATH_CALUDE_line_L_equation_l1821_182131


namespace NUMINAMATH_CALUDE_compound_has_two_hydrogen_l1821_182189

/-- Represents a chemical compound with hydrogen, carbon, and oxygen atoms. -/
structure Compound where
  hydrogen : ℕ
  carbon : ℕ
  oxygen : ℕ
  molecular_weight : ℕ

/-- Atomic weights of elements in g/mol -/
def atomic_weight (element : String) : ℕ :=
  match element with
  | "H" => 1
  | "C" => 12
  | "O" => 16
  | _ => 0

/-- Calculates the molecular weight of a compound based on its composition -/
def calculate_weight (c : Compound) : ℕ :=
  c.hydrogen * atomic_weight "H" +
  c.carbon * atomic_weight "C" +
  c.oxygen * atomic_weight "O"

/-- Theorem stating that a compound with 1 Carbon, 3 Oxygen, and 62 g/mol molecular weight has 2 Hydrogen atoms -/
theorem compound_has_two_hydrogen :
  ∀ (c : Compound),
    c.carbon = 1 →
    c.oxygen = 3 →
    c.molecular_weight = 62 →
    calculate_weight c = c.molecular_weight →
    c.hydrogen = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_compound_has_two_hydrogen_l1821_182189


namespace NUMINAMATH_CALUDE_triangle_sine_cosine_sum_l1821_182105

theorem triangle_sine_cosine_sum (A B C x y z : ℝ) :
  A + B + C = π →
  x * Real.sin A + y * Real.sin B + z * Real.sin C = 0 →
  (y + z * Real.cos A) * (z + x * Real.cos B) * (x + y * Real.cos C) + 
  (y * Real.cos A + z) * (z * Real.cos B + x) * (x * Real.cos C + y) = 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_cosine_sum_l1821_182105


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l1821_182175

theorem quadratic_equation_1 : 
  ∃ x₁ x₂ : ℝ, x₁^2 - 2*x₁ - 99 = 0 ∧ x₂^2 - 2*x₂ - 99 = 0 ∧ x₁ ≠ x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l1821_182175


namespace NUMINAMATH_CALUDE_wendys_cupcakes_l1821_182163

theorem wendys_cupcakes :
  ∀ (cupcakes cookies_baked pastries_left pastries_sold : ℕ),
    cookies_baked = 29 →
    pastries_left = 24 →
    pastries_sold = 9 →
    cupcakes + cookies_baked = pastries_left + pastries_sold →
    cupcakes = 4 := by
  sorry

end NUMINAMATH_CALUDE_wendys_cupcakes_l1821_182163


namespace NUMINAMATH_CALUDE_travel_time_calculation_l1821_182106

theorem travel_time_calculation (total_distance : ℝ) (average_speed : ℝ) (return_time : ℝ) :
  total_distance = 2000 ∧ 
  average_speed = 142.85714285714286 ∧ 
  return_time = 4 →
  total_distance / average_speed - return_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l1821_182106


namespace NUMINAMATH_CALUDE_triangle_area_bounds_l1821_182172

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 4

/-- The area of the triangle formed by the parabola and the line y = r -/
def triangleArea (r : ℝ) : ℝ := (r + 4)^(3/2)

/-- Theorem stating the relationship between r and the triangle area -/
theorem triangle_area_bounds (r : ℝ) :
  (16 ≤ triangleArea r ∧ triangleArea r ≤ 128) ↔ (8/3 ≤ r ∧ r ≤ 52/3) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_bounds_l1821_182172


namespace NUMINAMATH_CALUDE_total_earnings_l1821_182141

/-- Given that 5 men are equal to W women, W women are equal to B boys,
    and men's wages are 10, prove that the total amount earned by all groups is 150. -/
theorem total_earnings (W B : ℕ) (men_wage : ℕ) : 
  (5 = W) → (W = B) → (men_wage = 10) → 
  (5 * men_wage + W * men_wage + B * men_wage = 150) :=
by sorry

end NUMINAMATH_CALUDE_total_earnings_l1821_182141


namespace NUMINAMATH_CALUDE_x_power_27_minus_reciprocal_l1821_182183

theorem x_power_27_minus_reciprocal (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^27 - 1/(x^27) = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_power_27_minus_reciprocal_l1821_182183


namespace NUMINAMATH_CALUDE_min_of_four_expressions_bound_l1821_182129

theorem min_of_four_expressions_bound (r s u v : ℝ) :
  min (r - s^2) (min (s - u^2) (min (u - v^2) (v - r^2))) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_of_four_expressions_bound_l1821_182129


namespace NUMINAMATH_CALUDE_circle_radius_with_max_inscribed_rectangle_l1821_182125

theorem circle_radius_with_max_inscribed_rectangle (r : ℝ) : 
  r > 0 → 
  (∃ (rect_area : ℝ), rect_area = 50 ∧ rect_area = 2 * r^2) → 
  r = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_with_max_inscribed_rectangle_l1821_182125


namespace NUMINAMATH_CALUDE_mom_bought_71_packages_l1821_182198

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The total number of t-shirts Mom has -/
def total_shirts : ℕ := 426

/-- The number of packages Mom bought -/
def num_packages : ℕ := total_shirts / shirts_per_package

theorem mom_bought_71_packages : num_packages = 71 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_71_packages_l1821_182198


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l1821_182138

theorem quadratic_equation_solution_sum : ∃ (a b : ℝ), 
  (a^2 - 6*a + 15 = 24) ∧ 
  (b^2 - 6*b + 15 = 24) ∧ 
  (a ≥ b) ∧ 
  (3*a + 2*b = 15 + 3*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l1821_182138


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1821_182118

theorem absolute_value_inequality (b : ℝ) :
  (∃ x : ℝ, |x - 5| + |x - 7| < b) ↔ b > 2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1821_182118


namespace NUMINAMATH_CALUDE_solution_to_money_division_l1821_182195

/-- Represents the division of money among three parties -/
structure MoneyDivision where
  x : ℝ  -- Amount x gets
  y : ℝ  -- Amount y gets
  z : ℝ  -- Amount z gets
  a : ℝ  -- Amount y gets for each rupee x gets

/-- The conditions of the problem -/
def problem_conditions (d : MoneyDivision) : Prop :=
  d.y = d.a * d.x ∧
  d.z = 0.5 * d.x ∧
  d.x + d.y + d.z = 78 ∧
  d.y = 18

/-- The theorem stating the solution to the problem -/
theorem solution_to_money_division :
  ∀ d : MoneyDivision, problem_conditions d → d.a = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_money_division_l1821_182195


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l1821_182128

theorem no_solution_to_equation : ¬∃ x : ℝ, (x - 2) / (x + 2) - 16 / (x^2 - 4) = (x + 2) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l1821_182128


namespace NUMINAMATH_CALUDE_logarithm_inequality_l1821_182153

theorem logarithm_inequality (x y z : ℝ) 
  (hx : x = Real.log π)
  (hy : y = Real.log π / Real.log (1/2))
  (hz : z = Real.exp (-1/2)) : 
  y < z ∧ z < x := by sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l1821_182153


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1821_182156

theorem intersection_of_sets : 
  let M : Set ℕ := {0, 1, 2, 3}
  let N : Set ℕ := {1, 3, 4}
  M ∩ N = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1821_182156


namespace NUMINAMATH_CALUDE_negation_existence_absolute_value_l1821_182191

theorem negation_existence_absolute_value (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, |x| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_absolute_value_l1821_182191


namespace NUMINAMATH_CALUDE_union_covers_reals_l1821_182107

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem union_covers_reals (a : ℝ) :
  A ∪ B a = Set.univ → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l1821_182107


namespace NUMINAMATH_CALUDE_math_competition_probabilities_l1821_182199

-- Define the total number of questions and the number of questions each student answers
def total_questions : ℕ := 6
def questions_answered : ℕ := 3

-- Define the number of questions student A can correctly answer
def student_a_correct : ℕ := 4

-- Define the probability of student B correctly answering a question
def student_b_prob : ℚ := 2/3

-- Define the point values for correct answers
def points_a : ℕ := 15
def points_b : ℕ := 10

-- Define the probability that students A and B together correctly answer 3 questions
def prob_three_correct : ℚ := 31/135

-- Define the expected value of the total score
def expected_total_score : ℕ := 50

-- Theorem statement
theorem math_competition_probabilities :
  (prob_three_correct = 31/135) ∧
  (expected_total_score = 50) := by
  sorry

end NUMINAMATH_CALUDE_math_competition_probabilities_l1821_182199


namespace NUMINAMATH_CALUDE_bennys_cards_l1821_182166

theorem bennys_cards (x : ℕ) : 
  (x + 4) / 2 = 34 → x = 68 := by sorry

end NUMINAMATH_CALUDE_bennys_cards_l1821_182166


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1821_182168

/-- The parabola y = 2x^2 -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2

/-- The point A(-1, 2) -/
def point_A : ℝ × ℝ := (-1, 2)

/-- The line l: 4x + y + 2 = 0 -/
def line_l (x y : ℝ) : Prop := 4 * x + y + 2 = 0

/-- Theorem stating that the line l passes through point A and is tangent to the parabola -/
theorem line_tangent_to_parabola :
  line_l point_A.1 point_A.2 ∧
  parabola point_A.1 point_A.2 ∧
  ∃ (t : ℝ), t ≠ point_A.1 ∧
    (∀ (x y : ℝ), x ≠ point_A.1 → line_l x y → parabola x y → x = t) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1821_182168
