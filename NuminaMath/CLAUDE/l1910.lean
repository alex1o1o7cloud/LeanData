import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1910_191060

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 20) →
  (a 3 + a 4 = 40) →
  (a 5 + a 6 = 80) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1910_191060


namespace NUMINAMATH_CALUDE_f_derivative_at_two_l1910_191097

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_derivative_at_two
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : deriv (f a b) 1 = 0) :
  deriv (f a b) 2 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_f_derivative_at_two_l1910_191097


namespace NUMINAMATH_CALUDE_root_product_equals_27_l1910_191025

theorem root_product_equals_27 : (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_27_l1910_191025


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1910_191050

theorem system_solution_ratio (x y a b : ℝ) (h1 : 2 * x - y = a) (h2 : 3 * y - 6 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1910_191050


namespace NUMINAMATH_CALUDE_boat_distance_proof_l1910_191074

-- Define the given constants
def boat_speed : ℝ := 10
def stream_speed : ℝ := 2
def time_difference : ℝ := 1.5  -- 90 minutes in hours

-- Define the theorem
theorem boat_distance_proof :
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let upstream_time := (downstream_speed * time_difference) / (downstream_speed - upstream_speed)
  let distance := upstream_speed * upstream_time
  distance = 36 := by sorry

end NUMINAMATH_CALUDE_boat_distance_proof_l1910_191074


namespace NUMINAMATH_CALUDE_factorial_ratio_45_43_l1910_191043

-- Define factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_ratio_45_43 : factorial 45 / factorial 43 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_45_43_l1910_191043


namespace NUMINAMATH_CALUDE_composite_function_solution_l1910_191066

def δ (x : ℝ) : ℝ := 5 * x + 6
def φ (x : ℝ) : ℝ := 9 * x + 4

theorem composite_function_solution :
  ∀ x : ℝ, δ (φ x) = 14 → x = -4/15 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_solution_l1910_191066


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l1910_191003

theorem factorization_difference_of_squares (a : ℝ) : 
  a^2 - 9 = (a + 3) * (a - 3) := by sorry

#check factorization_difference_of_squares

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l1910_191003


namespace NUMINAMATH_CALUDE_winter_clothing_count_l1910_191021

/-- The number of boxes of clothing -/
def num_boxes : ℕ := 6

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 5

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 5

/-- The total number of pieces of winter clothing -/
def total_clothing : ℕ := num_boxes * (scarves_per_box + mittens_per_box)

theorem winter_clothing_count : total_clothing = 60 := by
  sorry

end NUMINAMATH_CALUDE_winter_clothing_count_l1910_191021


namespace NUMINAMATH_CALUDE_n_div_16_equals_4_pow_8086_l1910_191009

theorem n_div_16_equals_4_pow_8086 (n : ℕ) : n = 16^4044 → n / 16 = 4^8086 := by
  sorry

end NUMINAMATH_CALUDE_n_div_16_equals_4_pow_8086_l1910_191009


namespace NUMINAMATH_CALUDE_kayla_total_items_l1910_191058

/-- Represents the number of items bought by a person -/
structure Items :=
  (chocolate_bars : ℕ)
  (soda_cans : ℕ)

/-- The total number of items -/
def Items.total (i : Items) : ℕ := i.chocolate_bars + i.soda_cans

/-- Theresa bought twice the number of items as Kayla -/
def twice (kayla : Items) (theresa : Items) : Prop :=
  theresa.chocolate_bars = 2 * kayla.chocolate_bars ∧
  theresa.soda_cans = 2 * kayla.soda_cans

theorem kayla_total_items 
  (kayla theresa : Items)
  (h1 : twice kayla theresa)
  (h2 : theresa.chocolate_bars = 12)
  (h3 : theresa.soda_cans = 18) :
  kayla.total = 15 :=
by sorry

end NUMINAMATH_CALUDE_kayla_total_items_l1910_191058


namespace NUMINAMATH_CALUDE_greatest_number_of_bouquets_l1910_191081

/-- Represents the number of tulips of each color --/
structure TulipCount where
  white : Nat
  red : Nat
  blue : Nat
  yellow : Nat

/-- Represents the ratio of tulips in each bouquet --/
structure BouquetRatio where
  white : Nat
  red : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of bouquets that can be made with given tulips and ratio --/
def calculateBouquets (tulips : TulipCount) (ratio : BouquetRatio) : Nat :=
  min (tulips.white / ratio.white)
      (min (tulips.red / ratio.red)
           (min (tulips.blue / ratio.blue)
                (tulips.yellow / ratio.yellow)))

/-- Calculates the total number of flowers in a bouquet --/
def flowersPerBouquet (ratio : BouquetRatio) : Nat :=
  ratio.white + ratio.red + ratio.blue + ratio.yellow

theorem greatest_number_of_bouquets
  (tulips : TulipCount)
  (ratio : BouquetRatio)
  (h1 : tulips = ⟨21, 91, 37, 67⟩)
  (h2 : ratio = ⟨3, 7, 5, 9⟩)
  (h3 : flowersPerBouquet ratio ≥ 24)
  (h4 : flowersPerBouquet ratio ≤ 50)
  : calculateBouquets tulips ratio = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_of_bouquets_l1910_191081


namespace NUMINAMATH_CALUDE_factorization_proof_l1910_191063

theorem factorization_proof (a : ℝ) : 2 * a^2 - 2 * a + (1/2 : ℝ) = 2 * (a - 1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1910_191063


namespace NUMINAMATH_CALUDE_max_δ_is_seven_l1910_191028

/-- The sequence a_n = 1 + n^3 -/
def a (n : ℕ) : ℕ := 1 + n^3

/-- The greatest common divisor of consecutive terms in the sequence -/
def δ (n : ℕ) : ℕ := Nat.gcd (a (n + 1)) (a n)

/-- The maximum value of δ_n is 7 -/
theorem max_δ_is_seven : ∃ (n : ℕ), δ n = 7 ∧ ∀ (m : ℕ), δ m ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_δ_is_seven_l1910_191028


namespace NUMINAMATH_CALUDE_rectangular_solid_depth_l1910_191000

theorem rectangular_solid_depth (l w sa : ℝ) (h : ℝ) : 
  l = 6 → w = 5 → sa = 104 → sa = 2 * l * w + 2 * l * h + 2 * w * h → h = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_depth_l1910_191000


namespace NUMINAMATH_CALUDE_g_of_2_eq_8_l1910_191007

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def g (x : ℝ) : ℝ := 1 / (f.invFun x) + 7

theorem g_of_2_eq_8 : g 2 = 8 := by sorry

end NUMINAMATH_CALUDE_g_of_2_eq_8_l1910_191007


namespace NUMINAMATH_CALUDE_complex_number_in_quadrant_III_l1910_191088

def complex_number : ℂ := (-2 + Complex.I) * Complex.I^5

theorem complex_number_in_quadrant_III : 
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_complex_number_in_quadrant_III_l1910_191088


namespace NUMINAMATH_CALUDE_min_abs_z_on_circle_l1910_191062

theorem min_abs_z_on_circle (z : ℂ) (h : Complex.abs (z - (1 + Complex.I)) = 1) :
  ∃ (w : ℂ), Complex.abs (w - (1 + Complex.I)) = 1 ∧
             Complex.abs w = Real.sqrt 2 - 1 ∧
             ∀ (v : ℂ), Complex.abs (v - (1 + Complex.I)) = 1 → Complex.abs w ≤ Complex.abs v :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_on_circle_l1910_191062


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1910_191035

theorem complex_fraction_evaluation :
  (3/2 : ℚ) * (8/3 * (15/8 - 5/6)) / ((7/8 + 11/6) / (13/4)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1910_191035


namespace NUMINAMATH_CALUDE_thirtieth_roots_with_real_fifth_power_l1910_191022

theorem thirtieth_roots_with_real_fifth_power (ω : ℂ) (h : ω^3 = 1 ∧ ω ≠ 1) :
  ∃! (s : Finset ℂ), 
    (∀ z ∈ s, z^30 = 1) ∧ 
    (∀ z ∈ s, ∃ r : ℝ, z^5 = r) ∧
    s.card = 10 :=
sorry

end NUMINAMATH_CALUDE_thirtieth_roots_with_real_fifth_power_l1910_191022


namespace NUMINAMATH_CALUDE_discriminant_of_quadratic_equation_l1910_191023

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic equation 6x² + (6 + 1/6)x + 1/6 -/
def quadratic_equation (x : ℚ) : ℚ := 6*x^2 + (6 + 1/6)*x + 1/6

theorem discriminant_of_quadratic_equation : 
  discriminant 6 (6 + 1/6) (1/6) = 1225/36 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_quadratic_equation_l1910_191023


namespace NUMINAMATH_CALUDE_min_value_of_f_for_shangmei_numbers_l1910_191016

/-- Definition of a Shangmei number -/
def isShangmeiNumber (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a + c = 11 ∧ b + d = 11

/-- Definition of function f -/
def f (n : ℕ) : ℚ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (b - d : ℚ) / (a - c)

/-- Definition of function G -/
def G (n : ℕ) : ℤ :=
  let ab := n / 100
  let cd := n % 100
  (ab : ℤ) - cd

/-- Main theorem -/
theorem min_value_of_f_for_shangmei_numbers :
  ∀ M : ℕ,
    isShangmeiNumber M →
    (M / 1000 < (M / 100) % 10) →
    (G M) % 7 = 0 →
    f M ≥ -3 ∧ ∃ M₀, isShangmeiNumber M₀ ∧ (M₀ / 1000 < (M₀ / 100) % 10) ∧ (G M₀) % 7 = 0 ∧ f M₀ = -3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_for_shangmei_numbers_l1910_191016


namespace NUMINAMATH_CALUDE_regular_star_polygon_n_value_l1910_191008

/-- An n-pointed regular star polygon -/
structure RegularStarPolygon where
  n : ℕ
  edgeCount : ℕ
  edgeCount_eq : edgeCount = 2 * n
  angleA : ℝ
  angleB : ℝ
  angle_difference : angleB - angleA = 15

/-- The theorem stating that for a regular star polygon with the given properties, n must be 24 -/
theorem regular_star_polygon_n_value (star : RegularStarPolygon) : star.n = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_star_polygon_n_value_l1910_191008


namespace NUMINAMATH_CALUDE_count_off_ones_l1910_191056

theorem count_off_ones (n : ℕ) (h : n = 1994) : 
  (n / (Nat.lcm 3 4) : ℕ) = 166 := by
  sorry

end NUMINAMATH_CALUDE_count_off_ones_l1910_191056


namespace NUMINAMATH_CALUDE_least_multiplier_for_perfect_square_l1910_191057

def original_number : ℕ := 2^5 * 3^6 * 4^3 * 5^3 * 6^7

theorem least_multiplier_for_perfect_square :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, (original_number * n) = m^2) →
  15 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_multiplier_for_perfect_square_l1910_191057


namespace NUMINAMATH_CALUDE_increasing_cubic_function_parameter_negative_l1910_191054

/-- Given a function y = a(x^3 - 3x) that is increasing on the interval (-1, 1), prove that a < 0 --/
theorem increasing_cubic_function_parameter_negative
  (a : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, y x = a * (x^3 - 3*x))
  (h2 : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, StrictMono y):
  a < 0 :=
sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_parameter_negative_l1910_191054


namespace NUMINAMATH_CALUDE_ghee_composition_l1910_191013

theorem ghee_composition (original_quantity : ℝ) (vanaspati_percentage : ℝ) 
  (added_pure_ghee : ℝ) (new_vanaspati_percentage : ℝ) :
  original_quantity = 10 →
  vanaspati_percentage = 40 →
  added_pure_ghee = 10 →
  new_vanaspati_percentage = 20 →
  (vanaspati_percentage / 100) * original_quantity = 
    (new_vanaspati_percentage / 100) * (original_quantity + added_pure_ghee) →
  (100 - vanaspati_percentage) = 60 := by
sorry

end NUMINAMATH_CALUDE_ghee_composition_l1910_191013


namespace NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l1910_191029

theorem polygon_sides_and_diagonals :
  ∀ n : ℕ,
  (180 * (n - 2) = 3 * 360 + 180) →
  (n = 9 ∧ (n * (n - 3)) / 2 = 27) :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l1910_191029


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1910_191059

def front_seats : Nat := 4
def back_seats : Nat := 5
def people_to_seat : Nat := 2

def is_adjacent (row1 row2 seat1 seat2 : Nat) : Bool :=
  (row1 = row2 ∧ seat2 = seat1 + 1) ∨
  (row1 = 1 ∧ row2 = 2 ∧ (seat1 = seat2 ∨ seat1 + 1 = seat2))

def count_seating_arrangements : Nat :=
  let total_seats := front_seats + back_seats
  (total_seats.choose people_to_seat) -
  (front_seats - 1 + back_seats - 1 + front_seats)

theorem seating_arrangements_count :
  count_seating_arrangements = 58 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l1910_191059


namespace NUMINAMATH_CALUDE_largest_four_digit_perfect_cube_largest_four_digit_perfect_cube_is_9261_l1910_191049

theorem largest_four_digit_perfect_cube : ℕ → Prop :=
  fun n => (1000 ≤ n ∧ n ≤ 9999) ∧  -- n is a four-digit number
            (∃ m : ℕ, n = m^3) ∧    -- n is a perfect cube
            (∀ k : ℕ, (1000 ≤ k ∧ k ≤ 9999 ∧ ∃ m : ℕ, k = m^3) → k ≤ n)  -- n is the largest such number

theorem largest_four_digit_perfect_cube_is_9261 :
  largest_four_digit_perfect_cube 9261 := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_perfect_cube_largest_four_digit_perfect_cube_is_9261_l1910_191049


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l1910_191034

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 25 = 4

/-- The slopes of the asymptotes -/
def asymptote_slopes : Set ℝ := {0.8, -0.8}

/-- Theorem stating that the slopes of the asymptotes of the given hyperbola are ±0.8 -/
theorem hyperbola_asymptote_slopes :
  ∀ (x y : ℝ), hyperbola_eq x y → (∃ (m : ℝ), m ∈ asymptote_slopes ∧ 
    ∃ (b : ℝ), y = m * x + b) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l1910_191034


namespace NUMINAMATH_CALUDE_function_is_exponential_base_3_l1910_191046

-- Define the properties of the function f
def satisfies_functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem function_is_exponential_base_3 (f : ℝ → ℝ) 
  (h1 : satisfies_functional_equation f)
  (h2 : monotonically_increasing f) :
  ∀ x, f x = 3^x :=
sorry

end NUMINAMATH_CALUDE_function_is_exponential_base_3_l1910_191046


namespace NUMINAMATH_CALUDE_triangle_inequality_for_positive_reals_l1910_191064

theorem triangle_inequality_for_positive_reals (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  Real.sqrt (a^2 + b^2) ≤ a + b := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_for_positive_reals_l1910_191064


namespace NUMINAMATH_CALUDE_stationery_difference_l1910_191084

def georgia_stationery : ℚ := 25

def lorene_stationery : ℚ := 3 * georgia_stationery

def bria_stationery : ℚ := georgia_stationery + 10

def darren_stationery : ℚ := bria_stationery / 2

theorem stationery_difference :
  lorene_stationery + bria_stationery + darren_stationery - georgia_stationery = 102.5 := by
  sorry

end NUMINAMATH_CALUDE_stationery_difference_l1910_191084


namespace NUMINAMATH_CALUDE_probability_two_defective_out_of_ten_l1910_191033

/-- Given a set of products with some defective ones, this function calculates
    the probability of randomly selecting a defective product. -/
def probability_defective (total : ℕ) (defective : ℕ) : ℚ :=
  defective / total

/-- Theorem stating that for 10 products with 2 defective ones,
    the probability of randomly selecting a defective product is 1/5. -/
theorem probability_two_defective_out_of_ten :
  probability_defective 10 2 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_defective_out_of_ten_l1910_191033


namespace NUMINAMATH_CALUDE_three_numbers_problem_l1910_191005

theorem three_numbers_problem (a b c : ℝ) :
  (a + 1) * (b + 1) * (c + 1) = a * b * c + 1 ∧
  (a + 2) * (b + 2) * (c + 2) = a * b * c + 2 →
  a = -1 ∧ b = -1 ∧ c = -1 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l1910_191005


namespace NUMINAMATH_CALUDE_jerrys_age_l1910_191044

/-- Given that Mickey's age is 5 years more than 200% of Jerry's age,
    and Mickey is 21 years old, Jerry's age is 8 years. -/
theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 2 * jerry_age + 5 →
  mickey_age = 21 →
  jerry_age = 8 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l1910_191044


namespace NUMINAMATH_CALUDE_percent_problem_l1910_191061

theorem percent_problem (x : ℝ) (h : 0.4 * x = 160) : 0.5 * x = 200 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l1910_191061


namespace NUMINAMATH_CALUDE_min_pool_cost_is_5400_l1910_191027

/-- Represents the specifications of a rectangular pool -/
structure PoolSpecs where
  volume : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the minimum cost of constructing a rectangular pool given its specifications -/
def minPoolCost (specs : PoolSpecs) : ℝ :=
  sorry

/-- Theorem stating that the minimum cost of constructing the specified pool is 5400 yuan -/
theorem min_pool_cost_is_5400 :
  let specs : PoolSpecs := {
    volume := 18,
    depth := 2,
    bottomCost := 200,
    wallCost := 150
  }
  minPoolCost specs = 5400 :=
by sorry

end NUMINAMATH_CALUDE_min_pool_cost_is_5400_l1910_191027


namespace NUMINAMATH_CALUDE_parabola_equation_c_value_l1910_191076

/-- A parabola with vertex at (5, 1) passing through (2, 3) has equation x = ay^2 + by + c where c = 17/4 -/
theorem parabola_equation_c_value (a b c : ℝ) : 
  (∀ y : ℝ, 5 = a * 1^2 + b * 1 + c) →  -- vertex at (5, 1)
  (2 = a * 3^2 + b * 3 + c) →           -- passes through (2, 3)
  (∀ x y : ℝ, x = a * y^2 + b * y + c) →  -- equation of the form x = ay^2 + by + c
  c = 17/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_c_value_l1910_191076


namespace NUMINAMATH_CALUDE_rectangle_semicircle_ratio_l1910_191002

theorem rectangle_semicircle_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a * b = π * b^2 → a / b = π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_semicircle_ratio_l1910_191002


namespace NUMINAMATH_CALUDE_carolyn_practice_days_l1910_191006

/-- Represents the practice schedule of a musician --/
structure PracticeSchedule where
  piano_time : ℕ  -- Daily piano practice time in minutes
  violin_ratio : ℕ  -- Ratio of violin practice time to piano practice time
  total_monthly_time : ℕ  -- Total practice time in a month (in minutes)
  weeks_in_month : ℕ  -- Number of weeks in a month

/-- Calculates the number of practice days per week --/
def practice_days_per_week (schedule : PracticeSchedule) : ℚ :=
  let daily_total := schedule.piano_time * (1 + schedule.violin_ratio)
  let monthly_days := schedule.total_monthly_time / daily_total
  monthly_days / schedule.weeks_in_month

/-- Theorem stating that Carolyn practices 6 days a week --/
theorem carolyn_practice_days (schedule : PracticeSchedule) 
  (h1 : schedule.piano_time = 20)
  (h2 : schedule.violin_ratio = 3)
  (h3 : schedule.total_monthly_time = 1920)
  (h4 : schedule.weeks_in_month = 4) :
  practice_days_per_week schedule = 6 := by
  sorry

#eval practice_days_per_week ⟨20, 3, 1920, 4⟩

end NUMINAMATH_CALUDE_carolyn_practice_days_l1910_191006


namespace NUMINAMATH_CALUDE_triangle_properties_l1910_191031

theorem triangle_properties (A B C : Real) (a : Real × Real) :
  -- A, B, C are angles of a triangle
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π →
  -- Definition of vector a
  a = (Real.sqrt 2 * Real.cos ((A + B) / 2), Real.sin ((A - B) / 2)) →
  -- Magnitude of a
  Real.sqrt (a.1^2 + a.2^2) = Real.sqrt 6 / 2 →
  -- Conclusions
  (Real.tan A * Real.tan B = 1 / 3) ∧
  (∀ C', C' = π - A - B → Real.tan C' ≤ -Real.sqrt 3) ∧
  (∃ C', C' = π - A - B ∧ Real.tan C' = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1910_191031


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1910_191085

theorem sufficient_not_necessary (x : ℝ) : 
  (∃ (S T : Set ℝ), 
    S = {x | x > 2} ∧ 
    T = {x | x^2 - 3*x + 2 > 0} ∧ 
    S ⊂ T ∧ 
    ∃ y, y ∈ T ∧ y ∉ S) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1910_191085


namespace NUMINAMATH_CALUDE_constant_term_proof_l1910_191089

/-- The constant term in the expansion of (x^2 + 3)(x - 2/x)^6 -/
def constantTerm : ℤ := -240

/-- The expression (x^2 + 3)(x - 2/x)^6 -/
def expression (x : ℚ) : ℚ := (x^2 + 3) * (x - 2/x)^6

theorem constant_term_proof :
  ∃ (f : ℚ → ℚ), (∀ x ≠ 0, f x = expression x) ∧
  (∃ c : ℚ, ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε) ∧
  (c : ℤ) = constantTerm :=
sorry

end NUMINAMATH_CALUDE_constant_term_proof_l1910_191089


namespace NUMINAMATH_CALUDE_goose_eggs_count_l1910_191014

theorem goose_eggs_count (total_eggs : ℕ) : 
  (2 : ℚ) / 3 * (3 : ℚ) / 4 * (2 : ℚ) / 5 * total_eggs = 180 →
  total_eggs = 2700 := by
sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l1910_191014


namespace NUMINAMATH_CALUDE_fifth_monday_in_leap_year_l1910_191094

/-- Represents a date in February of a leap year -/
structure FebruaryDate :=
  (day : ℕ)
  (is_leap_year : Bool)

/-- Represents a day of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the weekday of a given February date -/
def weekday_of_date (d : FebruaryDate) : Weekday :=
  sorry

/-- Returns the number of Mondays up to and including a given date in February -/
def mondays_up_to (d : FebruaryDate) : ℕ :=
  sorry

/-- Theorem: In a leap year where February 7 is a Tuesday, 
    the fifth Monday in February falls on February 27 -/
theorem fifth_monday_in_leap_year :
  let feb7 : FebruaryDate := ⟨7, true⟩
  let feb27 : FebruaryDate := ⟨27, true⟩
  weekday_of_date feb7 = Weekday.Tuesday →
  mondays_up_to feb27 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_monday_in_leap_year_l1910_191094


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1910_191083

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let c := 3 * b
  let e := c / a
  (c + b/2) / (c - b/2) = 7/5 → e = 3 * Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1910_191083


namespace NUMINAMATH_CALUDE_candidate_a_votes_l1910_191093

def total_votes : ℕ := 560000
def invalid_percentage : ℚ := 15 / 100
def candidate_a_percentage : ℚ := 80 / 100

theorem candidate_a_votes : 
  (1 - invalid_percentage) * candidate_a_percentage * total_votes = 380800 := by
  sorry

end NUMINAMATH_CALUDE_candidate_a_votes_l1910_191093


namespace NUMINAMATH_CALUDE_bruno_pens_l1910_191020

/-- The number of pens in a dozen -/
def pens_per_dozen : ℕ := 12

/-- The total number of pens Bruno will have -/
def total_pens : ℕ := 30

/-- The number of dozens of pens Bruno wants to buy -/
def dozens_to_buy : ℚ := total_pens / pens_per_dozen

/-- Theorem stating that Bruno wants to buy 2.5 dozens of pens -/
theorem bruno_pens : dozens_to_buy = 5/2 := by sorry

end NUMINAMATH_CALUDE_bruno_pens_l1910_191020


namespace NUMINAMATH_CALUDE_bobs_raise_l1910_191042

/-- Calculates the raise per hour given the following conditions:
  * Bob works 40 hours per week
  * His housing benefit is reduced by $60 per month
  * He earns $5 more per week after the changes
-/
theorem bobs_raise (hours_per_week : ℕ) (benefit_reduction_per_month : ℚ) (extra_earnings_per_week : ℚ) :
  hours_per_week = 40 →
  benefit_reduction_per_month = 60 →
  extra_earnings_per_week = 5 →
  ∃ (raise_per_hour : ℚ), 
    raise_per_hour * hours_per_week - (benefit_reduction_per_month / 4) + extra_earnings_per_week = 0 ∧
    raise_per_hour = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_bobs_raise_l1910_191042


namespace NUMINAMATH_CALUDE_probability_of_green_ball_l1910_191010

theorem probability_of_green_ball (total_balls green_balls : ℕ) 
  (h1 : total_balls = 10)
  (h2 : green_balls = 4) : 
  (green_balls : ℚ) / total_balls = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_green_ball_l1910_191010


namespace NUMINAMATH_CALUDE_q_min_at_two_l1910_191037

/-- The function q in terms of x -/
def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 18

/-- The theorem stating that q is minimized to 0 when x = 2 -/
theorem q_min_at_two : 
  (∀ x : ℝ, q x ≥ q 2) ∧ q 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_q_min_at_two_l1910_191037


namespace NUMINAMATH_CALUDE_hoseok_marbles_l1910_191082

theorem hoseok_marbles : ∃ x : ℕ+, x * 80 + 260 = x * 100 ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_marbles_l1910_191082


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1910_191079

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^3 + 3*x - 5) - 7*(2*x^2 + x - 8) = 8*x^4 - 8*x^2 - 17*x + 56 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1910_191079


namespace NUMINAMATH_CALUDE_trapezoid_mn_length_l1910_191047

/-- Represents a trapezoid ABCD with points M and N on its sides -/
structure Trapezoid (α : Type*) [LinearOrderedField α] :=
  (a b : α)  -- lengths of BC and AD respectively
  (area_ratio : α)  -- ratio of areas of MBCN to MADN

/-- 
  Given a trapezoid ABCD with BC = a and AD = b, 
  if MN is parallel to AD and the areas of trapezoids MBCN and MADN are in the ratio 1:5, 
  then MN = sqrt((5a^2 + b^2) / 6)
-/
theorem trapezoid_mn_length 
  {α : Type*} [LinearOrderedField α] (t : Trapezoid α) 
  (h_ratio : t.area_ratio = 1/5) :
  ∃ mn : α, mn^2 = (5*t.a^2 + t.b^2) / 6 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_mn_length_l1910_191047


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l1910_191095

def television_price : ℝ := 650
def number_of_televisions : ℕ := 2
def discount_percentage : ℝ := 0.25

theorem discounted_price_calculation :
  let total_price := television_price * number_of_televisions
  let discount_amount := total_price * discount_percentage
  let final_price := total_price - discount_amount
  final_price = 975 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l1910_191095


namespace NUMINAMATH_CALUDE_cos_50_minus_tan_40_equals_sqrt_3_l1910_191001

theorem cos_50_minus_tan_40_equals_sqrt_3 :
  4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_50_minus_tan_40_equals_sqrt_3_l1910_191001


namespace NUMINAMATH_CALUDE_kevin_koala_leaves_kevin_koala_leaves_min_l1910_191052

theorem kevin_koala_leaves (n : ℕ) : n > 1 ∧ ∃ k : ℕ, n^2 = k^6 → n ≥ 8 :=
by sorry

theorem kevin_koala_leaves_min : ∃ k : ℕ, 8^2 = k^6 :=
by sorry

end NUMINAMATH_CALUDE_kevin_koala_leaves_kevin_koala_leaves_min_l1910_191052


namespace NUMINAMATH_CALUDE_stratified_sample_teachers_under_40_l1910_191073

/-- Calculates the number of teachers under 40 in a stratified sample -/
def stratified_sample_size (total_population : ℕ) (under_40_population : ℕ) (sample_size : ℕ) : ℕ :=
  (under_40_population * sample_size) / total_population

/-- Theorem: The stratified sample size for teachers under 40 is 50 -/
theorem stratified_sample_teachers_under_40 :
  stratified_sample_size 490 350 70 = 50 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_teachers_under_40_l1910_191073


namespace NUMINAMATH_CALUDE_raisin_distribution_l1910_191098

theorem raisin_distribution (total_raisins total_boxes box1_raisins box2_raisins : ℕ) 
  (h1 : total_raisins = 437)
  (h2 : total_boxes = 5)
  (h3 : box1_raisins = 72)
  (h4 : box2_raisins = 74)
  (h5 : ∃ (equal_box_raisins : ℕ), 
    total_raisins = box1_raisins + box2_raisins + 3 * equal_box_raisins) :
  ∃ (equal_box_raisins : ℕ), equal_box_raisins = 97 ∧ 
    total_raisins = box1_raisins + box2_raisins + 3 * equal_box_raisins :=
by sorry

end NUMINAMATH_CALUDE_raisin_distribution_l1910_191098


namespace NUMINAMATH_CALUDE_stating_transport_equation_transport_scenario_proof_l1910_191036

/-- Represents the scenario of two transports moving towards each other. -/
structure TransportScenario where
  x : ℝ  -- Speed of transport A in mph
  T : ℝ  -- Time in hours after transport A's departure when they are 348 miles apart

/-- 
  Theorem stating the relationship between the speeds and time
  for the given transport scenario.
-/
theorem transport_equation (scenario : TransportScenario) :
  let x := scenario.x
  let T := scenario.T
  2 * x * T + 18 * T - x - 18 = 258 := by
  sorry

/-- 
  Proves that the equation holds for the given transport scenario
  where two transports start 90 miles apart, with one traveling at speed x mph
  and the other at (x + 18) mph, starting 1 hour later, and end up 348 miles apart.
-/
theorem transport_scenario_proof (x : ℝ) (T : ℝ) :
  let scenario : TransportScenario := { x := x, T := T }
  (2 * x * T + 18 * T - x - 18 = 258) ↔
  (x * T + (x + 18) * (T - 1) = 348 - 90) := by
  sorry

end NUMINAMATH_CALUDE_stating_transport_equation_transport_scenario_proof_l1910_191036


namespace NUMINAMATH_CALUDE_correct_answers_for_given_score_l1910_191068

/-- Represents a test result -/
structure TestResult where
  totalQuestions : ℕ
  correctAnswers : ℕ
  score : ℤ

/-- Calculates the score based on correct and incorrect answers -/
def calculateScore (correct incorrect : ℕ) : ℤ :=
  (correct : ℤ) - 2 * (incorrect : ℤ)

theorem correct_answers_for_given_score 
  (result : TestResult) 
  (h1 : result.totalQuestions = 100)
  (h2 : result.score = calculateScore result.correctAnswers (result.totalQuestions - result.correctAnswers))
  (h3 : result.score = 76) :
  result.correctAnswers = 92 := by
  sorry


end NUMINAMATH_CALUDE_correct_answers_for_given_score_l1910_191068


namespace NUMINAMATH_CALUDE_hula_hoop_problem_l1910_191091

/-- Hula hoop problem -/
theorem hula_hoop_problem (nancy casey morgan alex : ℕ) : 
  nancy = 10 →
  casey = nancy - 3 →
  morgan = 3 * casey →
  alex = (nancy + casey + morgan) / 2 →
  alex = 19 := by sorry

end NUMINAMATH_CALUDE_hula_hoop_problem_l1910_191091


namespace NUMINAMATH_CALUDE_initial_marbles_correct_l1910_191080

/-- The number of marbles Josh initially had in his collection -/
def initial_marbles : ℕ := 9

/-- The number of marbles Josh lost -/
def lost_marbles : ℕ := 5

/-- The number of marbles Josh has left -/
def remaining_marbles : ℕ := 4

/-- Theorem stating that the initial number of marbles is correct -/
theorem initial_marbles_correct : initial_marbles = lost_marbles + remaining_marbles := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_correct_l1910_191080


namespace NUMINAMATH_CALUDE_function_range_l1910_191018

theorem function_range (a : ℝ) (f : ℝ → ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x, f x = a^(-x)) (h4 : f (-2) > f (-3)) : 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1910_191018


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1910_191011

theorem inequality_solution_set : 
  {x : ℝ | 2 * (x^2 - x) < 4} = {x : ℝ | -1 < x ∧ x < 2} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1910_191011


namespace NUMINAMATH_CALUDE_three_reflection_theorem_l1910_191041

/-- A circular billiard table -/
structure BilliardTable where
  R : ℝ
  R_pos : R > 0

/-- A point on the billiard table -/
structure Point (bt : BilliardTable) where
  x : ℝ
  y : ℝ
  on_table : x^2 + y^2 ≤ bt.R^2

/-- Predicate for a valid starting point A -/
def valid_start_point (bt : BilliardTable) (A : Point bt) : Prop :=
  A.x^2 + A.y^2 > (bt.R/3)^2 ∧ A.x^2 + A.y^2 < bt.R^2

/-- Predicate for a valid reflection path -/
def valid_reflection_path (bt : BilliardTable) (A : Point bt) : Prop :=
  ∃ (B C : Point bt),
    B ≠ A ∧ C ≠ A ∧ B ≠ C ∧
    (A.x^2 + A.y^2 = B.x^2 + B.y^2) ∧
    (B.x^2 + B.y^2 = C.x^2 + C.y^2) ∧
    (C.x^2 + C.y^2 = A.x^2 + A.y^2)

theorem three_reflection_theorem (bt : BilliardTable) (A : Point bt) :
  valid_start_point bt A ↔ valid_reflection_path bt A :=
sorry

end NUMINAMATH_CALUDE_three_reflection_theorem_l1910_191041


namespace NUMINAMATH_CALUDE_ellipse_intersection_max_y_intercept_l1910_191090

/-- An ellipse with major axis 2√2 times the minor axis, passing through (2, √2/2) --/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : a = 2 * Real.sqrt 2 * b
  h4 : (2 / a)^2 + ((Real.sqrt 2 / 2) / b)^2 = 1

/-- A line intersecting the ellipse at two points --/
structure IntersectingLine (e : Ellipse) where
  k : ℝ
  m : ℝ
  h1 : k ≠ 0  -- Line is not parallel to coordinate axes

/-- The distance between intersection points is 2√2 --/
def intersection_distance (e : Ellipse) (l : IntersectingLine e) : Prop :=
  ∃ (x1 x2 : ℝ), 
    (x1^2 / e.a^2) + ((l.k * x1 + l.m)^2 / e.b^2) = 1 ∧
    (x2^2 / e.a^2) + ((l.k * x2 + l.m)^2 / e.b^2) = 1 ∧
    (x2 - x1)^2 + (l.k * (x2 - x1))^2 = 8

/-- The theorem to be proved --/
theorem ellipse_intersection_max_y_intercept (e : Ellipse) :
  ∃ (max_m : ℝ), max_m = Real.sqrt 14 - Real.sqrt 7 ∧
  ∀ (l : IntersectingLine e), intersection_distance e l →
    l.m ≤ max_m ∧
    ∃ (l' : IntersectingLine e), intersection_distance e l' ∧ l'.m = max_m :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_max_y_intercept_l1910_191090


namespace NUMINAMATH_CALUDE_a_squared_plus_b_minus_c_in_M_l1910_191051

def P : Set ℤ := {x | ∃ k, x = 3*k + 1}
def Q : Set ℤ := {x | ∃ k, x = 3*k - 1}
def M : Set ℤ := {x | ∃ k, x = 3*k}

theorem a_squared_plus_b_minus_c_in_M (a b c : ℤ) 
  (ha : a ∈ P) (hb : b ∈ Q) (hc : c ∈ M) : 
  a^2 + b - c ∈ M :=
by sorry

end NUMINAMATH_CALUDE_a_squared_plus_b_minus_c_in_M_l1910_191051


namespace NUMINAMATH_CALUDE_problem_solution_l1910_191092

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 3 = y^2) 
  (h3 : x / 5 = 5*y) : 
  x = 625 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1910_191092


namespace NUMINAMATH_CALUDE_print_time_rounded_l1910_191087

/-- The number of pages to be printed -/
def total_pages : ℕ := 350

/-- The number of pages printed per minute -/
def pages_per_minute : ℕ := 25

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- The time required to print the pages, in minutes -/
def print_time : ℚ := total_pages / pages_per_minute

theorem print_time_rounded : round_to_nearest print_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_print_time_rounded_l1910_191087


namespace NUMINAMATH_CALUDE_square_of_negative_product_l1910_191077

theorem square_of_negative_product (a b : ℝ) : (-2 * a * b)^2 = 4 * a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l1910_191077


namespace NUMINAMATH_CALUDE_park_short_bushes_after_planting_l1910_191055

/-- The number of short bushes in a park after planting new ones. -/
def total_short_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem stating that the total number of short bushes after planting is 57. -/
theorem park_short_bushes_after_planting :
  total_short_bushes 37 20 = 57 := by
  sorry

end NUMINAMATH_CALUDE_park_short_bushes_after_planting_l1910_191055


namespace NUMINAMATH_CALUDE_dvd_sales_multiple_l1910_191024

/-- Proves that the multiple of production cost for DVD sales is 2.5 given the specified conditions --/
theorem dvd_sales_multiple (initial_cost : ℕ) (dvd_cost : ℕ) (daily_sales : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) (total_profit : ℕ) :
  initial_cost = 2000 →
  dvd_cost = 6 →
  daily_sales = 500 →
  days_per_week = 5 →
  num_weeks = 20 →
  total_profit = 448000 →
  ∃ (x : ℚ), x = 2.5 ∧ 
    (daily_sales * days_per_week * num_weeks * (dvd_cost * x - dvd_cost) : ℚ) - initial_cost = total_profit :=
by
  sorry

#check dvd_sales_multiple

end NUMINAMATH_CALUDE_dvd_sales_multiple_l1910_191024


namespace NUMINAMATH_CALUDE_foundation_dig_time_l1910_191086

/-- Represents the time taken to dig a foundation given the number of men -/
def digTime (men : ℕ) : ℝ := sorry

theorem foundation_dig_time :
  (digTime 20 = 6) →  -- It takes 20 men 6 days
  (∀ m₁ m₂ : ℕ, m₁ * digTime m₁ = m₂ * digTime m₂) →  -- Inverse proportion
  digTime 30 = 4 := by sorry

end NUMINAMATH_CALUDE_foundation_dig_time_l1910_191086


namespace NUMINAMATH_CALUDE_students_taking_courses_l1910_191039

theorem students_taking_courses (total : ℕ) (history : ℕ) (statistics : ℕ) (history_only : ℕ)
  (h_total : total = 89)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_only : history_only = 27) :
  ∃ (both : ℕ) (statistics_only : ℕ),
    history_only + statistics_only + both = 59 ∧
    both = history - history_only ∧
    statistics_only = statistics - both :=
by sorry

end NUMINAMATH_CALUDE_students_taking_courses_l1910_191039


namespace NUMINAMATH_CALUDE_seating_theorem_l1910_191067

/-- The number of ways to seat 2 students in a row of 5 desks with at least one empty desk between them -/
def seating_arrangements : ℕ := 12

/-- The number of desks in the row -/
def num_desks : ℕ := 5

/-- The number of students to be seated -/
def num_students : ℕ := 2

/-- Minimum number of empty desks between students -/
def min_empty_desks : ℕ := 1

theorem seating_theorem :
  seating_arrangements = 12 ∧
  num_desks = 5 ∧
  num_students = 2 ∧
  min_empty_desks = 1 →
  seating_arrangements = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l1910_191067


namespace NUMINAMATH_CALUDE_jane_exercise_goal_l1910_191015

/-- Jane's exercise routine -/
structure ExerciseRoutine where
  daily_hours : ℕ
  days_per_week : ℕ
  total_hours : ℕ

/-- Calculate the number of weeks Jane hit her goal -/
def weeks_goal_met (routine : ExerciseRoutine) : ℕ :=
  routine.total_hours / (routine.daily_hours * routine.days_per_week)

/-- Theorem: Jane hit her goal for 8 weeks -/
theorem jane_exercise_goal (routine : ExerciseRoutine) 
  (h1 : routine.daily_hours = 1)
  (h2 : routine.days_per_week = 5)
  (h3 : routine.total_hours = 40) : 
  weeks_goal_met routine = 8 := by
  sorry

end NUMINAMATH_CALUDE_jane_exercise_goal_l1910_191015


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1910_191026

theorem quadratic_equal_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*a*y + a + 2 = 0 → y = x) → 
  a = -1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1910_191026


namespace NUMINAMATH_CALUDE_prob_jack_queen_king_ace_value_l1910_191040

-- Define the total number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of cards for each face value
def cards_per_face : ℕ := 4

-- Define the probability of drawing the specific sequence
def prob_jack_queen_king_ace : ℚ :=
  (cards_per_face : ℚ) / total_cards *
  (cards_per_face : ℚ) / (total_cards - 1) *
  (cards_per_face : ℚ) / (total_cards - 2) *
  (cards_per_face : ℚ) / (total_cards - 3)

-- Theorem statement
theorem prob_jack_queen_king_ace_value :
  prob_jack_queen_king_ace = 16 / 4048375 := by
  sorry

end NUMINAMATH_CALUDE_prob_jack_queen_king_ace_value_l1910_191040


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1910_191048

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we're checking -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l1910_191048


namespace NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l1910_191017

/-- Given a bucket with known weights at different fill levels, 
    calculate the weight when it's full. -/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let three_fourths_weight := p
  let one_third_weight := q
  let full_weight := (8 * p - 11 * q) / 5
  full_weight

/-- Prove that the calculated full weight is correct -/
theorem bucket_weight_proof (p q : ℝ) : 
  bucket_weight p q = (8 * p - 11 * q) / 5 := by
  sorry

end NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l1910_191017


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l1910_191065

theorem cosine_sine_identity : 
  Real.cos (32 * π / 180) * Real.sin (62 * π / 180) - 
  Real.sin (32 * π / 180) * Real.sin (28 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l1910_191065


namespace NUMINAMATH_CALUDE_scout_weights_l1910_191012

/-- The weight measurement error of the scale -/
def error : ℝ := 2

/-- Míša's measured weight -/
def misa_measured : ℝ := 30

/-- Emil's measured weight -/
def emil_measured : ℝ := 28

/-- Combined measured weight of Míša and Emil -/
def combined_measured : ℝ := 56

/-- Míša's actual weight -/
def misa_actual : ℝ := misa_measured - error

/-- Emil's actual weight -/
def emil_actual : ℝ := emil_measured - error

theorem scout_weights :
  misa_actual = 28 ∧ emil_actual = 26 ∧
  misa_actual + emil_actual = combined_measured - error := by
  sorry

end NUMINAMATH_CALUDE_scout_weights_l1910_191012


namespace NUMINAMATH_CALUDE_valid_selections_eq_sixteen_l1910_191004

/-- The number of people to choose from -/
def n : ℕ := 5

/-- The number of positions to fill (team leader and deputy team leader) -/
def k : ℕ := 2

/-- The number of ways to select k people from n people, where order matters -/
def permutations (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

/-- The number of ways to select a team leader and deputy team leader
    when one specific person cannot be the deputy team leader -/
def valid_selections (n k : ℕ) : ℕ :=
  permutations n k - permutations (n - 1) (k - 1)

/-- The main theorem: prove that the number of valid selections is 16 -/
theorem valid_selections_eq_sixteen : valid_selections n k = 16 := by
  sorry

end NUMINAMATH_CALUDE_valid_selections_eq_sixteen_l1910_191004


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1910_191072

theorem polynomial_factorization (x : ℝ) : 
  x^9 + x^6 + x^3 + 1 = (x^3 + 1) * (x^6 - x^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1910_191072


namespace NUMINAMATH_CALUDE_franks_money_duration_l1910_191096

/-- The duration (in weeks) that Frank's money will last given his earnings and weekly spending. -/
def money_duration (lawn_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Frank's money will last for 9 weeks given his earnings and spending. -/
theorem franks_money_duration :
  money_duration 5 58 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_franks_money_duration_l1910_191096


namespace NUMINAMATH_CALUDE_triangle_area_l1910_191019

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
    prove that its area is 15√3 under certain conditions. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b < c →
  2 * a * c * Real.cos C + 2 * c^2 * Real.cos A = a + c →
  2 * c * Real.sin A - Real.sqrt 3 * a = 0 →
  let S := (1 / 2) * a * b * Real.sin C
  S = 15 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1910_191019


namespace NUMINAMATH_CALUDE_equation_solution_l1910_191099

theorem equation_solution :
  ∃! x : ℝ, x - 5 ≥ 0 ∧
  (7 / (Real.sqrt (x - 5) - 10) + 2 / (Real.sqrt (x - 5) - 3) +
   8 / (Real.sqrt (x - 5) + 3) + 13 / (Real.sqrt (x - 5) + 10) = 0) ∧
  x = 1486 / 225 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1910_191099


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1910_191030

theorem right_triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b < c) (right_triangle : a^2 + b^2 = c^2) :
  (1/a + 1/b + 1/c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) ∧
  ∀ M > 5 + 3 * Real.sqrt 2, ∃ a' b' c' : ℝ,
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' ≤ b' ∧ b' < c' ∧ a'^2 + b'^2 = c'^2 ∧
    (1/a' + 1/b' + 1/c') < M / (a' + b' + c') := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1910_191030


namespace NUMINAMATH_CALUDE_min_value_theorem_l1910_191038

/-- Given x > 0 and y > 0 satisfying ln(xy)^y = e^x, 
    the minimum value of x^2y - ln x - x is 1 -/
theorem min_value_theorem (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h : Real.log (x * y) ^ y = Real.exp x) : 
  ∃ (z : ℝ), z = 1 ∧ ∀ (w : ℝ), x^2 * y - Real.log x - x ≥ w → z ≤ w :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1910_191038


namespace NUMINAMATH_CALUDE_circle_area_relation_l1910_191032

/-- Two circles are tangent if they touch at exactly one point. -/
def CirclesTangent (A B : Set ℝ × ℝ) : Prop := sorry

/-- A circle passes through a point if the point lies on the circle's circumference. -/
def CirclePassesThrough (C : Set ℝ × ℝ) (p : ℝ × ℝ) : Prop := sorry

/-- The center of a circle. -/
def CircleCenter (C : Set ℝ × ℝ) : ℝ × ℝ := sorry

/-- The area of a circle. -/
def CircleArea (C : Set ℝ × ℝ) : ℝ := sorry

/-- Theorem: Given two circles A and B, where A is tangent to B and passes through B's center,
    if the area of A is 16π, then the area of B is 64π. -/
theorem circle_area_relation (A B : Set ℝ × ℝ) :
  CirclesTangent A B →
  CirclePassesThrough A (CircleCenter B) →
  CircleArea A = 16 * Real.pi →
  CircleArea B = 64 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_relation_l1910_191032


namespace NUMINAMATH_CALUDE_sum_of_abc_l1910_191070

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_abc_l1910_191070


namespace NUMINAMATH_CALUDE_mean_score_is_74_9_l1910_191071

structure ScoreDistribution where
  score : ℕ
  num_students : ℕ

def total_students : ℕ := 100

def score_data : List ScoreDistribution := [
  ⟨100, 10⟩,
  ⟨90, 15⟩,
  ⟨80, 20⟩,
  ⟨70, 30⟩,
  ⟨60, 20⟩,
  ⟨50, 4⟩,
  ⟨40, 1⟩
]

def sum_scores : ℕ := (score_data.map (λ x => x.score * x.num_students)).sum

theorem mean_score_is_74_9 : 
  (sum_scores : ℚ) / total_students = 749 / 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_score_is_74_9_l1910_191071


namespace NUMINAMATH_CALUDE_sandbox_ratio_l1910_191045

/-- A rectangular sandbox with specific dimensions. -/
structure Sandbox where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_multiple : ℝ
  width_eq : width = 5
  perimeter_eq : perimeter = 30
  length_eq : length = length_multiple * width

/-- The ratio of length to width for a sandbox with given properties is 2:1. -/
theorem sandbox_ratio (s : Sandbox) : s.length / s.width = 2 := by
  sorry


end NUMINAMATH_CALUDE_sandbox_ratio_l1910_191045


namespace NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l1910_191069

def third_smallest_prime : ℕ := sorry

theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime ^ 2) ^ 3 = 15625 := by sorry

end NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l1910_191069


namespace NUMINAMATH_CALUDE_square_area_larger_than_circle_l1910_191078

theorem square_area_larger_than_circle (R : ℝ) (h : R > 0) : 
  let AB := 2 * R * Real.sin (3 * Real.pi / 8)
  (AB ^ 2) > Real.pi * R ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_larger_than_circle_l1910_191078


namespace NUMINAMATH_CALUDE_jacob_future_age_l1910_191075

-- Define Jacob's current age
def jacob_age : ℕ := sorry

-- Define Michael's current age
def michael_age : ℕ := sorry

-- Define the number of years from now
variable (X : ℕ)

-- Jacob is 14 years younger than Michael
axiom age_difference : jacob_age = michael_age - 14

-- In 9 years, Michael will be twice as old as Jacob
axiom future_age_relation : michael_age + 9 = 2 * (jacob_age + 9)

-- Theorem: Jacob's age in X years from now is 5 + X
theorem jacob_future_age : jacob_age + X = 5 + X := by sorry

end NUMINAMATH_CALUDE_jacob_future_age_l1910_191075


namespace NUMINAMATH_CALUDE_not_equivalent_polar_point_l1910_191053

def is_equivalent_polar (r : ℝ) (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem not_equivalent_polar_point :
  ¬ is_equivalent_polar 2 (π/6) (11*π/6) := by
  sorry

end NUMINAMATH_CALUDE_not_equivalent_polar_point_l1910_191053
