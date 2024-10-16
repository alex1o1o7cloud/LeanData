import Mathlib

namespace NUMINAMATH_CALUDE_set_equality_l4123_412348

theorem set_equality (A B C : Set α) 
  (h1 : A ∪ B ⊆ C) 
  (h2 : A ∪ C ⊆ B) 
  (h3 : B ∪ C ⊆ A) : 
  A = B ∧ B = C := by
sorry

end NUMINAMATH_CALUDE_set_equality_l4123_412348


namespace NUMINAMATH_CALUDE_badminton_players_l4123_412324

/-- A sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  tennis_players : ℕ
  both_players : ℕ
  neither_players : ℕ

/-- Theorem stating the number of badminton players in the sports club -/
theorem badminton_players (club : SportsClub)
  (h1 : club.total_members = 30)
  (h2 : club.tennis_players = 19)
  (h3 : club.both_players = 9)
  (h4 : club.neither_players = 2) :
  club.total_members - club.tennis_players + club.both_players - club.neither_players = 18 :=
by sorry

end NUMINAMATH_CALUDE_badminton_players_l4123_412324


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4123_412339

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 10 → y = 68 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4123_412339


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l4123_412358

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_divisibility (m : ℕ) :
  ∃ k : ℕ, m ∣ (fibonacci k)^4 - (fibonacci k) - 2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l4123_412358


namespace NUMINAMATH_CALUDE_arithmetic_progression_properties_l4123_412325

/-- An arithmetic progression with the property that the sum of its first n terms is 5n² for any n -/
structure ArithmeticProgression where
  /-- The first term of the progression -/
  a₁ : ℝ
  /-- The common difference of the progression -/
  d : ℝ
  /-- Property: The sum of the first n terms is 5n² for any n -/
  sum_property : ∀ n : ℕ, n * (2 * a₁ + (n - 1) * d) / 2 = 5 * n^2

/-- Theorem stating the properties of the arithmetic progression -/
theorem arithmetic_progression_properties (ap : ArithmeticProgression) :
  ap.d = 10 ∧ ap.a₁ = 5 ∧ ap.a₁ + ap.d = 15 ∧ ap.a₁ + 2 * ap.d = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_properties_l4123_412325


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l4123_412316

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ n : ℕ, n > 0 ∧ 
    is_multiple n 45 ∧ 
    is_multiple n 75 ∧ 
    ¬is_multiple n 20 ∧
    ∀ m : ℕ, m > 0 → 
      is_multiple m 45 → 
      is_multiple m 75 → 
      ¬is_multiple m 20 → 
      n ≤ m ∧
  n = 225 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l4123_412316


namespace NUMINAMATH_CALUDE_fish_market_customers_l4123_412361

theorem fish_market_customers (num_tuna : ℕ) (tuna_weight : ℕ) (customer_want : ℕ) (unserved : ℕ) : 
  num_tuna = 10 → 
  tuna_weight = 200 → 
  customer_want = 25 → 
  unserved = 20 → 
  (num_tuna * tuna_weight) / customer_want + unserved = 100 := by
sorry

end NUMINAMATH_CALUDE_fish_market_customers_l4123_412361


namespace NUMINAMATH_CALUDE_pebble_difference_l4123_412373

/-- Given Shawn's pebble collection and painting process, prove the difference between blue and yellow pebbles. -/
theorem pebble_difference (total : ℕ) (red : ℕ) (blue : ℕ) (groups : ℕ) : 
  total = 40 →
  red = 9 →
  blue = 13 →
  groups = 3 →
  let remaining := total - red - blue
  let yellow := remaining / groups
  blue - yellow = 7 := by
  sorry

end NUMINAMATH_CALUDE_pebble_difference_l4123_412373


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l4123_412354

/-- An arithmetic sequence with 1990 terms -/
def ArithmeticSequence := Fin 1990 → ℝ

/-- The common difference of an arithmetic sequence -/
def commonDifference (a : ArithmeticSequence) : ℝ :=
  a 1 - a 0

/-- The condition that all terms in the sequence are positive -/
def allPositive (a : ArithmeticSequence) : Prop :=
  ∀ i j : Fin 1990, a i * a j > 0

/-- The b_k sequence defined in the problem -/
def b (a : ArithmeticSequence) (k : Fin 1990) : ℝ :=
  a k * a (1989 - k)

theorem arithmetic_sequence_max_product 
  (a : ArithmeticSequence) 
  (hd : commonDifference a ≠ 0) 
  (hp : allPositive a) : 
  (∀ k : Fin 1990, b a k ≤ b a 994 ∨ b a k ≤ b a 995) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l4123_412354


namespace NUMINAMATH_CALUDE_max_cylinder_volume_l4123_412302

/-- The maximum volume of a cylinder formed by rotating a rectangle with perimeter 20cm around one of its edges -/
theorem max_cylinder_volume : 
  ∃ (V : ℝ), V = (4000 / 27) * Real.pi ∧ 
  (∀ (x : ℝ), 0 < x → x < 10 → 
    π * x^2 * (10 - x) ≤ V) :=
by sorry

end NUMINAMATH_CALUDE_max_cylinder_volume_l4123_412302


namespace NUMINAMATH_CALUDE_min_lines_8x8_grid_is_14_l4123_412307

/-- The minimum number of straight lines required to separate all points in an 8x8 grid -/
def min_lines_8x8_grid : ℕ := 14

/-- The number of rows in the grid -/
def num_rows : ℕ := 8

/-- The number of columns in the grid -/
def num_columns : ℕ := 8

/-- The total number of points in the grid -/
def total_points : ℕ := num_rows * num_columns

/-- Theorem stating that the minimum number of lines to separate all points in an 8x8 grid is 14 -/
theorem min_lines_8x8_grid_is_14 : 
  min_lines_8x8_grid = (num_rows - 1) + (num_columns - 1) :=
sorry

end NUMINAMATH_CALUDE_min_lines_8x8_grid_is_14_l4123_412307


namespace NUMINAMATH_CALUDE_pages_difference_l4123_412355

theorem pages_difference (beatrix_pages cristobal_pages : ℕ) : 
  beatrix_pages = 704 →
  cristobal_pages = 15 + 3 * beatrix_pages →
  cristobal_pages - beatrix_pages = 1423 := by
sorry

end NUMINAMATH_CALUDE_pages_difference_l4123_412355


namespace NUMINAMATH_CALUDE_car_price_calculation_l4123_412341

/-- Calculates the price of a car given loan terms and payments -/
theorem car_price_calculation 
  (loan_years : ℕ) 
  (down_payment : ℚ) 
  (monthly_payment : ℚ) 
  (h_loan_years : loan_years = 5)
  (h_down_payment : down_payment = 5000)
  (h_monthly_payment : monthly_payment = 250) :
  down_payment + loan_years * 12 * monthly_payment = 20000 := by
  sorry

#check car_price_calculation

end NUMINAMATH_CALUDE_car_price_calculation_l4123_412341


namespace NUMINAMATH_CALUDE_large_triangle_perimeter_l4123_412318

/-- An isosceles triangle with two sides of length 20 and one side of length 10 -/
structure SmallTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : side2 = side3
  length_side1 : side1 = 10
  length_side2 : side2 = 20

/-- A triangle similar to SmallTriangle with shortest side of length 50 -/
structure LargeTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  shortest_side : side1 = 50
  similar_to_small : ∃ (k : ℝ), side1 = k * 10 ∧ side2 = k * 20 ∧ side3 = k * 20

/-- The perimeter of a triangle -/
def perimeter (t : LargeTriangle) : ℝ := t.side1 + t.side2 + t.side3

/-- Theorem stating that the perimeter of the larger triangle is 250 -/
theorem large_triangle_perimeter :
  ∀ (small : SmallTriangle) (large : LargeTriangle),
  perimeter large = 250 := by sorry

end NUMINAMATH_CALUDE_large_triangle_perimeter_l4123_412318


namespace NUMINAMATH_CALUDE_red_balls_estimate_l4123_412396

/-- Estimates the number of red balls in a bag given the total number of balls,
    the number of draws, and the number of red balls drawn. -/
def estimate_red_balls (total_balls : ℕ) (total_draws : ℕ) (red_draws : ℕ) : ℕ :=
  (total_balls * red_draws) / total_draws

/-- Theorem stating that under the given conditions, the estimated number of red balls is 6. -/
theorem red_balls_estimate :
  let total_balls : ℕ := 20
  let total_draws : ℕ := 100
  let red_draws : ℕ := 30
  estimate_red_balls total_balls total_draws red_draws = 6 := by
  sorry

#eval estimate_red_balls 20 100 30

end NUMINAMATH_CALUDE_red_balls_estimate_l4123_412396


namespace NUMINAMATH_CALUDE_temp_difference_l4123_412319

/-- The highest temperature in Xiangyang City on March 7, 2023 -/
def highest_temp : ℝ := 26

/-- The lowest temperature in Xiangyang City on March 7, 2023 -/
def lowest_temp : ℝ := 14

/-- The theorem states that the difference between the highest and lowest temperatures is 12°C -/
theorem temp_difference : highest_temp - lowest_temp = 12 := by
  sorry

end NUMINAMATH_CALUDE_temp_difference_l4123_412319


namespace NUMINAMATH_CALUDE_line_slope_product_l4123_412397

theorem line_slope_product (m n : ℝ) (h1 : m ≠ 0) (h2 : m = 4 * n) 
  (h3 : Real.arctan m = 2 * Real.arctan n) : m * n = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_product_l4123_412397


namespace NUMINAMATH_CALUDE_smallest_solution_l4123_412346

-- Define the equation
def equation (x : ℝ) : Prop :=
  3 * x / (x - 3) + (3 * x^2 - 27) / x = 15

-- State the theorem
theorem smallest_solution :
  ∃ (s : ℝ), s = 1 - Real.sqrt 10 ∧
  equation s ∧
  (∀ (x : ℝ), equation x → x ≥ s) :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_l4123_412346


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4123_412331

theorem polynomial_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  (((x^2 + a)^2) / ((a - b)*(a - c)) + ((x^2 + b)^2) / ((b - a)*(b - c)) + ((x^2 + c)^2) / ((c - a)*(c - b))) =
  x^4 + x^2*(a + b + c) + (a^2 + b^2 + c^2) := by
  sorry

#check polynomial_simplification

end NUMINAMATH_CALUDE_polynomial_simplification_l4123_412331


namespace NUMINAMATH_CALUDE_cinema_sampling_method_l4123_412372

/-- Represents a seating arrangement in a cinema --/
structure CinemaSeating where
  rows : ℕ
  seats_per_row : ℕ
  all_seats_filled : Bool

/-- Represents a sampling method --/
inductive SamplingMethod
  | LotteryMethod
  | RandomNumberTable
  | SystematicSampling
  | SamplingWithReplacement

/-- Defines the characteristics of systematic sampling --/
def is_systematic_sampling (seating : CinemaSeating) (selected_seat : ℕ) : Prop :=
  seating.all_seats_filled ∧
  selected_seat > 0 ∧
  selected_seat ≤ seating.seats_per_row ∧
  seating.rows > 1

/-- The main theorem to prove --/
theorem cinema_sampling_method (seating : CinemaSeating) (selected_seat : ℕ) :
  seating.rows = 50 →
  seating.seats_per_row = 60 →
  seating.all_seats_filled = true →
  selected_seat = 18 →
  is_systematic_sampling seating selected_seat →
  SamplingMethod.SystematicSampling = SamplingMethod.SystematicSampling :=
by
  sorry

end NUMINAMATH_CALUDE_cinema_sampling_method_l4123_412372


namespace NUMINAMATH_CALUDE_cone_volume_with_special_surface_area_l4123_412337

/-- 
Given a cone with base radius R, if its lateral surface area is equal to the sum of 
the areas of its base and axial section, then its volume is (2π²R³) / (3(π² - 1)).
-/
theorem cone_volume_with_special_surface_area (R : ℝ) (h : R > 0) : 
  let lateral_surface_area := π * R * (R^2 + (2 * π * R / (π^2 - 1))^2).sqrt
  let base_area := π * R^2
  let axial_section_area := R * (2 * π * R / (π^2 - 1))
  lateral_surface_area = base_area + axial_section_area →
  (1/3) * π * R^2 * (2 * π * R / (π^2 - 1)) = 2 * π^2 * R^3 / (3 * (π^2 - 1)) := by
sorry

end NUMINAMATH_CALUDE_cone_volume_with_special_surface_area_l4123_412337


namespace NUMINAMATH_CALUDE_restaurant_glasses_problem_l4123_412391

theorem restaurant_glasses_problem (x : ℕ) :
  let small_box_count : ℕ := 1
  let large_box_count : ℕ := 16 + small_box_count
  let total_boxes : ℕ := small_box_count + large_box_count
  let glasses_per_large_box : ℕ := 16
  let average_glasses_per_box : ℕ := 15
  let total_glasses : ℕ := 480
  (total_boxes * average_glasses_per_box + x + large_box_count * glasses_per_large_box = total_glasses) →
  x = 224 := by
sorry

end NUMINAMATH_CALUDE_restaurant_glasses_problem_l4123_412391


namespace NUMINAMATH_CALUDE_andy_math_problem_l4123_412320

theorem andy_math_problem (last_problem : ℕ) (total_solved : ℕ) (start_problem : ℕ) :
  last_problem = 125 →
  total_solved = 56 →
  start_problem = last_problem - total_solved + 1 →
  start_problem = 70 := by
sorry

end NUMINAMATH_CALUDE_andy_math_problem_l4123_412320


namespace NUMINAMATH_CALUDE_physics_class_size_l4123_412313

theorem physics_class_size 
  (total_students : ℕ) 
  (physics_students : ℕ) 
  (math_students : ℕ) 
  (both_subjects : ℕ) :
  total_students = 100 →
  physics_students = math_students + both_subjects →
  physics_students = 2 * math_students →
  both_subjects = 10 →
  physics_students = 62 := by
sorry

end NUMINAMATH_CALUDE_physics_class_size_l4123_412313


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l4123_412395

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 8 * a 9 * a 10 = -a 13 ^ 2) →
  (a 8 * a 9 * a 10 = -1000) →
  a 10 * a 12 = 100 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l4123_412395


namespace NUMINAMATH_CALUDE_megan_lead_actress_percentage_l4123_412369

def total_plays : ℕ := 100
def not_lead_plays : ℕ := 20

theorem megan_lead_actress_percentage :
  (total_plays - not_lead_plays) * 100 / total_plays = 80 := by
  sorry

end NUMINAMATH_CALUDE_megan_lead_actress_percentage_l4123_412369


namespace NUMINAMATH_CALUDE_prism_sphere_surface_area_l4123_412380

/-- Right triangular prism with specified properties -/
structure RightTriangularPrism where
  -- Base triangle
  AB : ℝ
  AC : ℝ
  angleBAC : ℝ
  -- Prism properties
  volume : ℝ
  -- Ensure all vertices lie on the same spherical surface
  onSphere : Bool

/-- Theorem stating the surface area of the sphere containing the prism -/
theorem prism_sphere_surface_area (p : RightTriangularPrism) 
  (h1 : p.AB = 2)
  (h2 : p.AC = 1)
  (h3 : p.angleBAC = π / 3)  -- 60° in radians
  (h4 : p.volume = Real.sqrt 3)
  (h5 : p.onSphere = true) :
  ∃ (r : ℝ), 4 * π * r^2 = 8 * π := by
    sorry


end NUMINAMATH_CALUDE_prism_sphere_surface_area_l4123_412380


namespace NUMINAMATH_CALUDE_lake_pleasant_excursion_l4123_412385

theorem lake_pleasant_excursion (total_kids : ℕ) 
  (h1 : total_kids = 40)
  (h2 : ∃ tubing_kids : ℕ, 4 * tubing_kids = total_kids)
  (h3 : ∃ rafting_kids : ℕ, 2 * rafting_kids = tubing_kids) :
  rafting_kids = 5 := by
  sorry

end NUMINAMATH_CALUDE_lake_pleasant_excursion_l4123_412385


namespace NUMINAMATH_CALUDE_limits_of_f_l4123_412338

noncomputable def f (x : ℝ) : ℝ := 2^(1/x)

theorem limits_of_f :
  (∀ ε > 0, ∃ δ > 0, ∀ x < 0, |x| < δ → |f x| < ε) ∧
  (∀ M > 0, ∃ δ > 0, ∀ x > 0, x < δ → f x > M) ∧
  ¬ (∃ L : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |f x - L| < ε) :=
by sorry

end NUMINAMATH_CALUDE_limits_of_f_l4123_412338


namespace NUMINAMATH_CALUDE_odd_square_plus_n_times_odd_plus_one_parity_l4123_412323

theorem odd_square_plus_n_times_odd_plus_one_parity (o n : ℤ) 
  (ho : ∃ k : ℤ, o = 2 * k + 1) :
  Odd (o^2 + n*o + 1) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_odd_square_plus_n_times_odd_plus_one_parity_l4123_412323


namespace NUMINAMATH_CALUDE_ribbon_remaining_l4123_412356

/-- Proves that given a ribbon of 51 meters, after cutting 100 pieces of 15 centimeters each, 
    the remaining ribbon length is 36 meters. -/
theorem ribbon_remaining (total_length : ℝ) (num_pieces : ℕ) (piece_length : ℝ) :
  total_length = 51 →
  num_pieces = 100 →
  piece_length = 0.15 →
  total_length - (num_pieces : ℝ) * piece_length = 36 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_remaining_l4123_412356


namespace NUMINAMATH_CALUDE_quadratic_quotient_cubic_at_zero_l4123_412381

-- Define the set of integers from 1 to 5
def S : Set ℕ := {1, 2, 3, 4, 5}

-- Define the property that f(n) = n^3 for n in S
def cubic_on_S (f : ℚ → ℚ) : Prop :=
  ∀ n ∈ S, f n = n^3

-- Define the property that f is a quotient of two quadratic polynomials
def is_quadratic_quotient (f : ℚ → ℚ) : Prop :=
  ∃ (p q : ℚ → ℚ),
    (∀ x, ∃ a b c, p x = a*x^2 + b*x + c) ∧
    (∀ x, ∃ d e g, q x = d*x^2 + e*x + g) ∧
    (∀ x, q x ≠ 0) ∧
    (∀ x, f x = p x / q x)

-- The main theorem
theorem quadratic_quotient_cubic_at_zero
  (f : ℚ → ℚ)
  (h1 : is_quadratic_quotient f)
  (h2 : cubic_on_S f) :
  f 0 = 24/17 := by
sorry

end NUMINAMATH_CALUDE_quadratic_quotient_cubic_at_zero_l4123_412381


namespace NUMINAMATH_CALUDE_cube_root_sum_of_cubes_l4123_412388

theorem cube_root_sum_of_cubes : 
  (20^3 + 70^3 + 110^3 : ℝ)^(1/3) = 120 := by sorry

end NUMINAMATH_CALUDE_cube_root_sum_of_cubes_l4123_412388


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_even_numbers_l4123_412336

theorem largest_of_three_consecutive_even_numbers (x : ℤ) : 
  (∃ (a b c : ℤ), 
    (a + b + c = 312) ∧ 
    (b = a + 2) ∧ 
    (c = b + 2) ∧ 
    (Even a) ∧ (Even b) ∧ (Even c)) →
  (max a (max b c) = 106) :=
sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_even_numbers_l4123_412336


namespace NUMINAMATH_CALUDE_function_sum_derivative_difference_l4123_412300

/-- Given a function f(x) = a*sin(3x) + b*x^3 + 4 where a and b are real numbers,
    prove that f(2014) + f(-2014) + f'(2015) - f'(-2015) = 8 -/
theorem function_sum_derivative_difference (a b : ℝ) : 
  let f (x : ℝ) := a * Real.sin (3 * x) + b * x^3 + 4
  let f' (x : ℝ) := 3 * a * Real.cos (3 * x) + 3 * b * x^2
  f 2014 + f (-2014) + f' 2015 - f' (-2015) = 8 := by
sorry

end NUMINAMATH_CALUDE_function_sum_derivative_difference_l4123_412300


namespace NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l4123_412362

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : Nat)
  (green : Nat)
  (blue : Nat)
  (black : Nat)

/-- The minimum number of socks needed to ensure at least n pairs -/
def minSocksForPairs (drawer : SockDrawer) (n : Nat) : Nat :=
  sorry

theorem min_socks_for_fifteen_pairs :
  let drawer := SockDrawer.mk 120 100 70 50
  minSocksForPairs drawer 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l4123_412362


namespace NUMINAMATH_CALUDE_fraction_simplification_l4123_412343

theorem fraction_simplification :
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4123_412343


namespace NUMINAMATH_CALUDE_proposition_s_range_p_or_q_and_not_q_range_l4123_412332

-- Define propositions p, q, and s
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / (4 - m) + y^2 / m = 1 → x^2 / a^2 + y^2 / b^2 = 1

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0

def s (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

-- Theorem 1
theorem proposition_s_range (m : ℝ) : s m → m < 0 ∨ m ≥ 1 := by sorry

-- Theorem 2
theorem p_or_q_and_not_q_range (m : ℝ) : (p m ∨ q m) ∧ ¬(q m) → 1 ≤ m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_proposition_s_range_p_or_q_and_not_q_range_l4123_412332


namespace NUMINAMATH_CALUDE_green_garden_potato_yield_l4123_412382

/-- Represents Mr. Green's garden and potato yield calculation --/
theorem green_garden_potato_yield :
  let garden_length_steps : ℕ := 25
  let garden_width_steps : ℕ := 30
  let step_length_feet : ℕ := 3
  let non_productive_percentage : ℚ := 1/10
  let yield_per_square_foot : ℚ := 3/4

  let garden_length_feet : ℕ := garden_length_steps * step_length_feet
  let garden_width_feet : ℕ := garden_width_steps * step_length_feet
  let garden_area : ℕ := garden_length_feet * garden_width_feet
  let productive_area : ℚ := garden_area * (1 - non_productive_percentage)
  let total_yield : ℚ := productive_area * yield_per_square_foot

  total_yield = 4556.25 := by sorry

end NUMINAMATH_CALUDE_green_garden_potato_yield_l4123_412382


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_properties_l4123_412340

theorem consecutive_integers_sum_properties :
  (∀ k : ℤ, ¬∃ n : ℤ, 12 * k + 78 = n ^ 2) ∧
  (∃ k : ℤ, ∃ n : ℤ, 11 * k + 66 = n ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_properties_l4123_412340


namespace NUMINAMATH_CALUDE_equation_solution_l4123_412374

theorem equation_solution (y : ℝ) (x : ℝ) :
  (x + 1) / (x - 2) = (y^2 + 4*y + 1) / (y^2 + 4*y - 3) →
  x = -(3*y^2 + 12*y - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l4123_412374


namespace NUMINAMATH_CALUDE_sum_of_remainders_l4123_412364

theorem sum_of_remainders : Int.mod (Int.mod (5^(5^(5^5))) 500 + Int.mod (2^(2^(2^2))) 500) 500 = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remainders_l4123_412364


namespace NUMINAMATH_CALUDE_shifted_line_through_origin_l4123_412303

/-- A line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shift a line horizontally -/
def shift_line (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + l.slope * d }

/-- Check if a line passes through a point -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem shifted_line_through_origin (b : ℝ) :
  let original_line := Line.mk 2 b
  let shifted_line := shift_line original_line 2
  passes_through shifted_line 0 0 → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_shifted_line_through_origin_l4123_412303


namespace NUMINAMATH_CALUDE_sandys_savings_difference_l4123_412326

/-- Calculates the difference in savings between two years given the initial salary,
    savings percentages, and salary increase. -/
def savings_difference (initial_salary : ℝ) (savings_percent_year1 : ℝ) 
                       (savings_percent_year2 : ℝ) (salary_increase_percent : ℝ) : ℝ :=
  let salary_year2 := initial_salary * (1 + salary_increase_percent)
  let savings_year1 := initial_salary * savings_percent_year1
  let savings_year2 := salary_year2 * savings_percent_year2
  savings_year1 - savings_year2

/-- The difference in Sandy's savings between two years is $925.20 -/
theorem sandys_savings_difference :
  savings_difference 45000 0.083 0.056 0.115 = 925.20 := by
  sorry

end NUMINAMATH_CALUDE_sandys_savings_difference_l4123_412326


namespace NUMINAMATH_CALUDE_sum_of_squares_l4123_412379

theorem sum_of_squares (a b c : ℝ) : 
  (a + b + c) / 3 = 10 →
  (a * b * c) ^ (1/3 : ℝ) = 6 →
  3 / (1/a + 1/b + 1/c) = 4 →
  a^2 + b^2 + c^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l4123_412379


namespace NUMINAMATH_CALUDE_complex_sum_equality_l4123_412384

theorem complex_sum_equality (B Q R T : ℂ) : 
  B = 3 - 2*I ∧ Q = -5 + I ∧ R = 1 - 2*I ∧ T = 4 + 3*I → B - Q + R + T = 13 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l4123_412384


namespace NUMINAMATH_CALUDE_new_clock_conversion_l4123_412399

/-- Represents a time on the new clock -/
structure NewClockTime where
  hours : ℕ
  minutes : ℕ

/-- Represents a time in Beijing -/
structure BeijingTime where
  hours : ℕ
  minutes : ℕ

/-- Converts NewClockTime to total minutes -/
def newClockToMinutes (t : NewClockTime) : ℕ :=
  t.hours * 100 + t.minutes

/-- Converts BeijingTime to total minutes -/
def beijingToMinutes (t : BeijingTime) : ℕ :=
  t.hours * 60 + t.minutes

/-- The theorem to be proved -/
theorem new_clock_conversion (newClock : NewClockTime) (beijing : BeijingTime) :
  (newClockToMinutes ⟨5, 0⟩ = beijingToMinutes ⟨12, 0⟩) →
  (newClockToMinutes ⟨6, 75⟩ = beijingToMinutes ⟨16, 12⟩) := by
  sorry


end NUMINAMATH_CALUDE_new_clock_conversion_l4123_412399


namespace NUMINAMATH_CALUDE_actress_not_lead_plays_l4123_412317

theorem actress_not_lead_plays (total_plays : ℕ) (lead_percentage : ℚ) 
  (h1 : total_plays = 100)
  (h2 : lead_percentage = 80 / 100) :
  total_plays - (total_plays * lead_percentage).floor = 20 := by
sorry

end NUMINAMATH_CALUDE_actress_not_lead_plays_l4123_412317


namespace NUMINAMATH_CALUDE_arithmetic_progression_difference_l4123_412363

/-- An arithmetic progression with first term a₁, last term aₙ, common difference d, and sum Sₙ. -/
structure ArithmeticProgression (α : Type*) [Field α] where
  a₁ : α
  aₙ : α
  d : α
  n : ℕ
  Sₙ : α
  h₁ : aₙ = a₁ + (n - 1) * d
  h₂ : Sₙ = n / 2 * (a₁ + aₙ)

/-- The common difference of an arithmetic progression can be expressed in terms of its first term, 
last term, and sum. -/
theorem arithmetic_progression_difference (α : Type*) [Field α] (ap : ArithmeticProgression α) :
  ap.d = (ap.aₙ^2 - ap.a₁^2) / (2 * ap.Sₙ - (ap.a₁ + ap.aₙ)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_difference_l4123_412363


namespace NUMINAMATH_CALUDE_trig_inequality_l4123_412330

theorem trig_inequality : 2 * Real.sin (160 * π / 180) < Real.tan (50 * π / 180) ∧
                          Real.tan (50 * π / 180) < 1 + Real.cos (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l4123_412330


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_twice_perimeter_l4123_412308

/-- A triangle with an inscribed circle -/
structure Triangle :=
  (area : ℝ)
  (perimeter : ℝ)
  (inradius : ℝ)

/-- The theorem stating that for a triangle where the area is twice the perimeter, 
    the radius of the inscribed circle is 4 -/
theorem inscribed_circle_radius_when_area_twice_perimeter (t : Triangle) 
  (h : t.area = 2 * t.perimeter) : t.inradius = 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_twice_perimeter_l4123_412308


namespace NUMINAMATH_CALUDE_living_room_set_cost_l4123_412370

theorem living_room_set_cost (couch_cost sectional_cost other_cost : ℕ)
  (discount_rate : ℚ) (h1 : couch_cost = 2500) (h2 : sectional_cost = 3500)
  (h3 : other_cost = 2000) (h4 : discount_rate = 1/10) :
  (couch_cost + sectional_cost + other_cost) * (1 - discount_rate) = 7200 :=
by sorry

end NUMINAMATH_CALUDE_living_room_set_cost_l4123_412370


namespace NUMINAMATH_CALUDE_perpendicular_vectors_and_angle_l4123_412392

theorem perpendicular_vectors_and_angle (θ φ : ℝ) : 
  (0 < θ) → (θ < π) →
  (π / 2 < φ) → (φ < π) →
  (2 * Real.cos θ + Real.sin θ = 0) →
  (Real.sin (θ - φ) = Real.sqrt 10 / 10) →
  (Real.tan θ = -2 ∧ Real.cos φ = -(Real.sqrt 2 / 10)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_and_angle_l4123_412392


namespace NUMINAMATH_CALUDE_angle_sum_is_345_l4123_412335

def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

theorem angle_sum_is_345 (α β γ : ℝ) 
  (h1 : is_acute_angle α ∨ is_acute_angle β ∨ is_acute_angle γ)
  (h2 : is_acute_angle α ∨ is_acute_angle β ∨ is_acute_angle γ)
  (h3 : is_obtuse_angle α ∨ is_obtuse_angle β ∨ is_obtuse_angle γ)
  (h4 : (α + β + γ) / 15 = 23 ∨ (α + β + γ) / 15 = 24 ∨ (α + β + γ) / 15 = 25) :
  α + β + γ = 345 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_345_l4123_412335


namespace NUMINAMATH_CALUDE_find_a_l4123_412328

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {3, 7, a^2 - 2*a - 3}

-- Define set A
def A (a : ℝ) : Set ℝ := {7, |a - 7|}

-- Define the complement of A in U
def complement_A (a : ℝ) : Set ℝ := U a \ A a

-- Theorem statement
theorem find_a : ∃ (a : ℝ), 
  (U a = {3, 7, a^2 - 2*a - 3}) ∧ 
  (A a = {7, |a - 7|}) ∧ 
  (complement_A a = {5}) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_find_a_l4123_412328


namespace NUMINAMATH_CALUDE_exists_unique_subset_l4123_412389

theorem exists_unique_subset : ∃ (S : Set ℤ), 
  ∀ (n : ℤ), ∃! (pair : ℤ × ℤ), 
    pair.1 ∈ S ∧ pair.2 ∈ S ∧ n = 2 * pair.1 + pair.2 := by
  sorry

end NUMINAMATH_CALUDE_exists_unique_subset_l4123_412389


namespace NUMINAMATH_CALUDE_no_rational_roots_l4123_412312

/-- The polynomial we're investigating -/
def f (x : ℚ) : ℚ := 3 * x^4 - 2 * x^3 - 10 * x^2 + 4 * x + 1

/-- Theorem stating that the polynomial has no rational roots -/
theorem no_rational_roots : ∀ q : ℚ, f q ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l4123_412312


namespace NUMINAMATH_CALUDE_absolute_value_at_zero_l4123_412347

-- Define a fourth-degree polynomial with real coefficients
def fourthDegreePolynomial (a b c d e : ℝ) : ℝ → ℝ := λ x => a*x^4 + b*x^3 + c*x^2 + d*x + e

-- State the theorem
theorem absolute_value_at_zero (a b c d e : ℝ) :
  let g := fourthDegreePolynomial a b c d e
  (|g 1| = 16 ∧ |g 3| = 16 ∧ |g 4| = 16 ∧ |g 5| = 16 ∧ |g 6| = 16 ∧ |g 7| = 16) →
  |g 0| = 54 := by
  sorry


end NUMINAMATH_CALUDE_absolute_value_at_zero_l4123_412347


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l4123_412386

theorem tripled_base_and_exponent (a b x : ℝ) (h1 : b ≠ 0) :
  let r := (3*a)^(3*b)
  r = a^b * x^b → x = 27 * a^2 := by
sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l4123_412386


namespace NUMINAMATH_CALUDE_value_of_a_l4123_412345

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + 2 = 0}

-- Theorem statement
theorem value_of_a (a : ℝ) : B a ⊆ A ∧ B a ≠ ∅ → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l4123_412345


namespace NUMINAMATH_CALUDE_running_speed_calculation_l4123_412306

theorem running_speed_calculation (walking_speed running_speed total_distance total_time : ℝ) :
  walking_speed = 4 →
  total_distance = 4 →
  total_time = 0.75 →
  (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time →
  running_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_running_speed_calculation_l4123_412306


namespace NUMINAMATH_CALUDE_junior_toys_l4123_412368

theorem junior_toys (monday : ℕ) : 
  let wednesday := 2 * monday
  let friday := 4 * monday
  let saturday := wednesday / 2
  let total_toys := monday + wednesday + friday + saturday
  let num_rabbits := 16
  let toys_per_rabbit := 3
  total_toys = num_rabbits * toys_per_rabbit →
  monday = 6 := by
sorry

end NUMINAMATH_CALUDE_junior_toys_l4123_412368


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l4123_412342

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (180 * (n - 2) : ℝ) / n = 156 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l4123_412342


namespace NUMINAMATH_CALUDE_sum_is_composite_l4123_412360

theorem sum_is_composite (a b c d : ℕ) (h : a^2 + b^2 = c^2 + d^2) :
  ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ a + b + c + d = k * m :=
by sorry

end NUMINAMATH_CALUDE_sum_is_composite_l4123_412360


namespace NUMINAMATH_CALUDE_inverse_implies_negation_l4123_412309

theorem inverse_implies_negation (P : Prop) :
  (¬P → False) → (¬P) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_implies_negation_l4123_412309


namespace NUMINAMATH_CALUDE_pencil_sales_l4123_412350

/-- The number of pencils initially sold for a rupee -/
def N : ℕ := 20

/-- The cost price of one pencil -/
def C : ℚ := 1 / 13

/-- Theorem stating that N pencils sold for a rupee results in a 35% loss
    and 10 pencils sold for a rupee results in a 30% gain -/
theorem pencil_sales (N : ℕ) (C : ℚ) :
  (N : ℚ) * (0.65 * C) = 1 ∧ 10 * (1.3 * C) = 1 → N = 20 :=
by sorry

end NUMINAMATH_CALUDE_pencil_sales_l4123_412350


namespace NUMINAMATH_CALUDE_square_difference_equality_l4123_412351

theorem square_difference_equality (a b M : ℝ) : 
  (a + 2*b)^2 = (a - 2*b)^2 + M → M = 8*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l4123_412351


namespace NUMINAMATH_CALUDE_geometric_progression_properties_l4123_412314

/-- Represents a geometric progression with given properties -/
structure GeometricProgression where
  ratio : ℚ
  fourthTerm : ℚ
  sum : ℚ

/-- The number of terms in the geometric progression -/
def numTerms (gp : GeometricProgression) : ℕ := sorry

/-- Theorem stating the properties of the specific geometric progression -/
theorem geometric_progression_properties :
  ∃ (gp : GeometricProgression),
    gp.ratio = 1/3 ∧
    gp.fourthTerm = 1/54 ∧
    gp.sum = 121/162 ∧
    numTerms gp = 5 := by sorry

end NUMINAMATH_CALUDE_geometric_progression_properties_l4123_412314


namespace NUMINAMATH_CALUDE_ryan_reads_more_pages_l4123_412393

/-- Given that Ryan read 2100 pages in 7 days and his brother read 200 pages per day for 7 days,
    prove that Ryan read 100 more pages per day on average compared to his brother. -/
theorem ryan_reads_more_pages (ryan_total_pages : ℕ) (days : ℕ) (brother_daily_pages : ℕ)
    (h1 : ryan_total_pages = 2100)
    (h2 : days = 7)
    (h3 : brother_daily_pages = 200) :
    ryan_total_pages / days - brother_daily_pages = 100 := by
  sorry

end NUMINAMATH_CALUDE_ryan_reads_more_pages_l4123_412393


namespace NUMINAMATH_CALUDE_problem_solution_l4123_412333

theorem problem_solution (m n : ℕ+) 
  (h1 : ∃ k : ℕ, m = 111 * k)
  (h2 : ∃ l : ℕ, n = 31 * l)
  (h3 : m + n = 2017) :
  n - m = 463 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4123_412333


namespace NUMINAMATH_CALUDE_not_always_equal_l4123_412375

theorem not_always_equal : ∃ (a b : ℝ), 3 * (a + b) ≠ 3 * a + b := by
  sorry

end NUMINAMATH_CALUDE_not_always_equal_l4123_412375


namespace NUMINAMATH_CALUDE_no_valid_pairs_l4123_412359

theorem no_valid_pairs : 
  ¬∃ (a b x y : ℤ), 
    (a * x + b * y = 3) ∧ 
    (x^2 + y^2 = 85) ∧ 
    (3 * a - 5 * b = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_pairs_l4123_412359


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l4123_412365

/-- An isosceles triangle with an angle bisector dividing the perimeter -/
structure IsoscelesTriangleWithBisector where
  /-- The length of one of the equal sides -/
  side : ℝ
  /-- The length of the base -/
  base : ℝ
  /-- The length of the angle bisector -/
  bisector : ℝ
  /-- The angle bisector divides the perimeter into parts of 63 and 35 -/
  perimeter_division : side + bisector = 63 ∧ side + base / 2 = 35
  /-- The triangle is isosceles -/
  isosceles : side > 0

/-- The length of the equal sides in the given isosceles triangle is not 26.4, 33, or 38.5 -/
theorem isosceles_triangle_side_length
  (t : IsoscelesTriangleWithBisector) :
  t.side ≠ 26.4 ∧ t.side ≠ 33 ∧ t.side ≠ 38.5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l4123_412365


namespace NUMINAMATH_CALUDE_cone_generatrix_property_cylinder_generatrix_parallel_l4123_412327

-- Define the necessary geometric objects
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Cone where
  vertex : Point3D
  base_center : Point3D
  base_radius : ℝ

structure Cylinder where
  base_center : Point3D
  height : ℝ
  radius : ℝ

-- Define what a generatrix is for a cone and a cylinder
def is_generatrix_of_cone (l : Set Point3D) (c : Cone) : Prop :=
  ∃ p : Point3D, p ∈ l ∧ 
    (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 = c.base_radius^2 ∧
    p.z = c.base_center.z ∧
    c.vertex ∈ l

def are_parallel (l1 l2 : Set Point3D) : Prop :=
  ∃ v : Point3D, ∀ p q : Point3D, p ∈ l1 ∧ q ∈ l2 → 
    ∃ t : ℝ, q.x - p.x = t * v.x ∧ q.y - p.y = t * v.y ∧ q.z - p.z = t * v.z

def is_generatrix_of_cylinder (l : Set Point3D) (c : Cylinder) : Prop :=
  ∃ p q : Point3D, p ∈ l ∧ q ∈ l ∧
    (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 = c.radius^2 ∧
    p.z = c.base_center.z ∧
    (q.x - c.base_center.x)^2 + (q.y - c.base_center.y)^2 = c.radius^2 ∧
    q.z = c.base_center.z + c.height

-- State the theorems to be proved
theorem cone_generatrix_property (c : Cone) (p : Point3D) :
  (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 = c.base_radius^2 ∧ 
  p.z = c.base_center.z →
  is_generatrix_of_cone {q : Point3D | ∃ t : ℝ, q = Point3D.mk 
    (c.vertex.x + t * (p.x - c.vertex.x))
    (c.vertex.y + t * (p.y - c.vertex.y))
    (c.vertex.z + t * (p.z - c.vertex.z))} c :=
sorry

theorem cylinder_generatrix_parallel (c : Cylinder) (l1 l2 : Set Point3D) :
  is_generatrix_of_cylinder l1 c ∧ is_generatrix_of_cylinder l2 c →
  are_parallel l1 l2 :=
sorry

end NUMINAMATH_CALUDE_cone_generatrix_property_cylinder_generatrix_parallel_l4123_412327


namespace NUMINAMATH_CALUDE_max_value_implies_a_l4123_412310

/-- The function f(x) = 2x^3 - 3x^2 + a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 6 * x^2 - 6 * x

theorem max_value_implies_a (a : ℝ) :
  (∃ (max : ℝ), max = 6 ∧ ∀ (x : ℝ), f a x ≤ max) →
  a = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l4123_412310


namespace NUMINAMATH_CALUDE_first_number_problem_l4123_412353

theorem first_number_problem (x y : ℤ) (h1 : y = 11) (h2 : x + (y + 3) = 19) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_number_problem_l4123_412353


namespace NUMINAMATH_CALUDE_total_cleanings_is_777_l4123_412387

/-- Calculates the total number of times Michael, Angela, and Lucy clean themselves in 52 weeks --/
def total_cleanings : ℕ :=
  let weeks_in_year : ℕ := 52
  let days_in_week : ℕ := 7
  let month_in_weeks : ℕ := 4

  -- Michael's cleaning schedule
  let michael_baths_per_week : ℕ := 2
  let michael_showers_per_week : ℕ := 1
  let michael_vacation_weeks : ℕ := 3

  -- Angela's cleaning schedule
  let angela_showers_per_day : ℕ := 1
  let angela_vacation_weeks : ℕ := 2

  -- Lucy's regular cleaning schedule
  let lucy_baths_per_week : ℕ := 3
  let lucy_showers_per_week : ℕ := 2

  -- Lucy's modified schedule for one month
  let lucy_modified_baths_per_week : ℕ := 1
  let lucy_modified_showers_per_day : ℕ := 1

  -- Calculate total cleanings
  let michael_total := (michael_baths_per_week + michael_showers_per_week) * weeks_in_year - 
                       (michael_baths_per_week + michael_showers_per_week) * michael_vacation_weeks

  let angela_total := angela_showers_per_day * days_in_week * weeks_in_year - 
                      angela_showers_per_day * days_in_week * angela_vacation_weeks

  let lucy_regular_weeks := weeks_in_year - month_in_weeks
  let lucy_total := (lucy_baths_per_week + lucy_showers_per_week) * lucy_regular_weeks +
                    (lucy_modified_baths_per_week + lucy_modified_showers_per_day * days_in_week) * month_in_weeks

  michael_total + angela_total + lucy_total

theorem total_cleanings_is_777 : total_cleanings = 777 := by
  sorry

end NUMINAMATH_CALUDE_total_cleanings_is_777_l4123_412387


namespace NUMINAMATH_CALUDE_science_class_students_l4123_412371

theorem science_class_students :
  ∃! n : ℕ, 0 < n ∧ n < 60 ∧ n % 6 = 4 ∧ n % 8 = 5 ∧ n = 46 := by
sorry

end NUMINAMATH_CALUDE_science_class_students_l4123_412371


namespace NUMINAMATH_CALUDE_subtraction_of_negative_problem_solution_l4123_412352

theorem subtraction_of_negative (a b : ℤ) : a - (-b) = a + b := by sorry

theorem problem_solution : 2 - (-12) = 14 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_problem_solution_l4123_412352


namespace NUMINAMATH_CALUDE_seven_valid_positions_l4123_412378

/-- Represents a position where an additional square can be attached --/
inductive Position
| CentralExtension
| OuterEdge
| MiddleEdge

/-- Represents the cross-shaped polygon --/
structure CrossPolygon where
  squares : Fin 6 → Unit  -- Represents the 6 squares in the cross
  additional_positions : Fin 11 → Position  -- Represents the 11 possible positions

/-- Represents a configuration with an additional square attached --/
structure ExtendedPolygon where
  base : CrossPolygon
  additional_square_position : Fin 11

/-- Predicate to check if a configuration can be folded into a cube with one face missing --/
def can_fold_to_cube (ep : ExtendedPolygon) : Prop :=
  sorry  -- Definition of this predicate would depend on the geometry of the problem

/-- The main theorem to be proved --/
theorem seven_valid_positions (cp : CrossPolygon) :
  (∃ (valid_positions : Finset (Fin 11)), 
    valid_positions.card = 7 ∧ 
    (∀ p : Fin 11, p ∈ valid_positions ↔ can_fold_to_cube ⟨cp, p⟩)) :=
  sorry


end NUMINAMATH_CALUDE_seven_valid_positions_l4123_412378


namespace NUMINAMATH_CALUDE_podium_cube_count_theorem_l4123_412376

/-- Represents a three-step podium made of wooden cubes -/
structure Podium where
  total_cubes : ℕ
  no_white_faces : ℕ
  one_white_face : ℕ
  two_white_faces : ℕ
  three_white_faces : ℕ

/-- The podium is valid if it satisfies the conditions of the problem -/
def is_valid_podium (p : Podium) : Prop :=
  p.total_cubes = 144 ∧
  p.no_white_faces = 40 ∧
  p.one_white_face = 64 ∧
  p.two_white_faces = 32 ∧
  p.three_white_faces = 8

/-- Theorem stating that the sum of cubes with 0, 1, 2, and 3 white faces
    equals the total number of cubes, implying no cubes with 4, 5, or 6 white faces -/
theorem podium_cube_count_theorem (p : Podium) (h : is_valid_podium p) :
  p.no_white_faces + p.one_white_face + p.two_white_faces + p.three_white_faces = p.total_cubes :=
by sorry


end NUMINAMATH_CALUDE_podium_cube_count_theorem_l4123_412376


namespace NUMINAMATH_CALUDE_box_dimensions_l4123_412390

theorem box_dimensions (a b c : ℕ) 
  (h1 : a + c = 17) 
  (h2 : a + b = 13) 
  (h3 : b + c = 20) : 
  (a = 5 ∧ b = 8 ∧ c = 12) ∨ 
  (a = 5 ∧ b = 12 ∧ c = 8) ∨ 
  (a = 8 ∧ b = 5 ∧ c = 12) ∨ 
  (a = 8 ∧ b = 12 ∧ c = 5) ∨ 
  (a = 12 ∧ b = 5 ∧ c = 8) ∨ 
  (a = 12 ∧ b = 8 ∧ c = 5) :=
sorry

end NUMINAMATH_CALUDE_box_dimensions_l4123_412390


namespace NUMINAMATH_CALUDE_interior_angles_increase_l4123_412315

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: If a convex polygon with n sides has a sum of interior angles of 3240 degrees,
    then a convex polygon with n + 3 sides has a sum of interior angles of 3780 degrees. -/
theorem interior_angles_increase (n : ℕ) :
  sum_interior_angles n = 3240 → sum_interior_angles (n + 3) = 3780 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_increase_l4123_412315


namespace NUMINAMATH_CALUDE_keychainSavings_l4123_412383

/-- Represents the cost and quantity of a pack of key chains -/
structure KeyChainPack where
  quantity : ℕ
  cost : ℚ

/-- Calculates the cost per key chain for a given pack -/
def costPerKeyChain (pack : KeyChainPack) : ℚ :=
  pack.cost / pack.quantity

/-- Calculates the total cost for a given number of key chains using a specific pack -/
def totalCost (pack : KeyChainPack) (totalKeyChains : ℕ) : ℚ :=
  (totalKeyChains / pack.quantity) * pack.cost

theorem keychainSavings :
  let pack1 : KeyChainPack := { quantity := 10, cost := 20 }
  let pack2 : KeyChainPack := { quantity := 4, cost := 12 }
  let totalKeyChains : ℕ := 20
  let savings := totalCost pack2 totalKeyChains - totalCost pack1 totalKeyChains
  savings = 20 := by sorry

end NUMINAMATH_CALUDE_keychainSavings_l4123_412383


namespace NUMINAMATH_CALUDE_x_geq_1_necessary_not_sufficient_for_lg_x_geq_1_l4123_412329

-- Define the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem x_geq_1_necessary_not_sufficient_for_lg_x_geq_1 :
  (∀ x : ℝ, lg x ≥ 1 → x ≥ 1) ∧
  ¬(∀ x : ℝ, x ≥ 1 → lg x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_geq_1_necessary_not_sufficient_for_lg_x_geq_1_l4123_412329


namespace NUMINAMATH_CALUDE_three_dogs_walking_time_l4123_412305

def base_charge : ℕ := 20
def per_minute_charge : ℕ := 1
def total_earnings : ℕ := 171
def one_dog_time : ℕ := 10
def two_dogs_time : ℕ := 7

def calculate_charge (dogs : ℕ) (minutes : ℕ) : ℕ :=
  dogs * (base_charge + per_minute_charge * minutes)

theorem three_dogs_walking_time :
  ∃ (x : ℕ), 
    calculate_charge 1 one_dog_time + 
    calculate_charge 2 two_dogs_time + 
    calculate_charge 3 x = total_earnings ∧ 
    x = 9 := by sorry

end NUMINAMATH_CALUDE_three_dogs_walking_time_l4123_412305


namespace NUMINAMATH_CALUDE_larry_cards_larry_cards_proof_l4123_412394

/-- If Larry initially has 67 cards and Dennis takes 9 cards away, 
    then Larry will have 58 cards remaining. -/
theorem larry_cards : ℕ → ℕ → ℕ → Prop :=
  fun initial_cards cards_taken remaining_cards =>
    initial_cards = 67 ∧ 
    cards_taken = 9 ∧ 
    remaining_cards = initial_cards - cards_taken →
    remaining_cards = 58

-- The proof would go here
theorem larry_cards_proof : larry_cards 67 9 58 := by
  sorry

end NUMINAMATH_CALUDE_larry_cards_larry_cards_proof_l4123_412394


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l4123_412366

/-- Given a hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0, 
    if one of its asymptotes is y = 3x, then b = 3 -/
theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) : 
  (∃ x y : ℝ, x^2 - y^2/b^2 = 1 ∧ y = 3*x) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l4123_412366


namespace NUMINAMATH_CALUDE_expected_rain_theorem_l4123_412367

/-- The number of days in the week --/
def days : ℕ := 7

/-- The probability of sun (0 inches of rain) --/
def prob_sun : ℝ := 0.3

/-- The probability of 3 inches of rain --/
def prob_rain_3 : ℝ := 0.4

/-- The probability of 8 inches of rain --/
def prob_rain_8 : ℝ := 0.3

/-- The amount of rain on a sunny day --/
def rain_sun : ℝ := 0

/-- The amount of rain on a moderately rainy day --/
def rain_moderate : ℝ := 3

/-- The amount of rain on a heavily rainy day --/
def rain_heavy : ℝ := 8

/-- The expected value of rain for a single day --/
def expected_daily_rain : ℝ := 
  prob_sun * rain_sun + prob_rain_3 * rain_moderate + prob_rain_8 * rain_heavy

/-- The expected value of total rain for the week --/
def expected_weekly_rain : ℝ := days * expected_daily_rain

theorem expected_rain_theorem : expected_weekly_rain = 25.2 := by
  sorry

end NUMINAMATH_CALUDE_expected_rain_theorem_l4123_412367


namespace NUMINAMATH_CALUDE_one_fourths_in_two_thirds_l4123_412349

theorem one_fourths_in_two_thirds : (2 : ℚ) / 3 / ((1 : ℚ) / 4) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_one_fourths_in_two_thirds_l4123_412349


namespace NUMINAMATH_CALUDE_inequality_always_true_l4123_412357

theorem inequality_always_true : ∀ x : ℝ, 4 * x^2 - 4 * x + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l4123_412357


namespace NUMINAMATH_CALUDE_sum_of_digits_888_base8_l4123_412322

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the sum of digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits of the base 8 representation of 888₁₀ is 13 -/
theorem sum_of_digits_888_base8 : sumDigits (toBase8 888) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_888_base8_l4123_412322


namespace NUMINAMATH_CALUDE_chocolate_cookies_sold_l4123_412311

/-- Proves that the number of chocolate cookies sold is 220 --/
theorem chocolate_cookies_sold (price_chocolate : ℕ) (price_vanilla : ℕ) (vanilla_sold : ℕ) (total_revenue : ℕ) :
  price_chocolate = 1 →
  price_vanilla = 2 →
  vanilla_sold = 70 →
  total_revenue = 360 →
  total_revenue = price_chocolate * (total_revenue - price_vanilla * vanilla_sold) + price_vanilla * vanilla_sold →
  total_revenue - price_vanilla * vanilla_sold = 220 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_cookies_sold_l4123_412311


namespace NUMINAMATH_CALUDE_diamond_three_two_l4123_412321

def diamond (a b : ℝ) : ℝ := a * b^3 - b^2 + 1

theorem diamond_three_two : diamond 3 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_two_l4123_412321


namespace NUMINAMATH_CALUDE_inverse_function_b_value_l4123_412304

/-- Given a function f and its inverse, prove the value of b -/
theorem inverse_function_b_value 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 1 / (2 * x + b)) 
  (h2 : ∀ x, f⁻¹ x = (2 - 3 * x) / (5 * x)) 
  : b = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_b_value_l4123_412304


namespace NUMINAMATH_CALUDE_allocation_methods_3_6_3_l4123_412344

/-- The number of ways to allocate doctors and nurses to schools -/
def allocation_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  (num_doctors.choose 1 * num_nurses.choose 2) *
  ((num_doctors - 1).choose 1 * (num_nurses - 2).choose 2) *
  ((num_doctors - 2).choose 1 * (num_nurses - 4).choose 2)

/-- Theorem stating that the number of allocation methods for 3 doctors and 6 nurses to 3 schools is 540 -/
theorem allocation_methods_3_6_3 :
  allocation_methods 3 6 3 = 540 := by
  sorry

end NUMINAMATH_CALUDE_allocation_methods_3_6_3_l4123_412344


namespace NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l4123_412398

/-- Given that α = 3, prove that the terminal side of α lies in the second quadrant. -/
theorem terminal_side_in_second_quadrant (α : ℝ) (h : α = 3) :
  (π / 2 : ℝ) < α ∧ α < π :=
sorry

end NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l4123_412398


namespace NUMINAMATH_CALUDE_flowers_lilly_can_buy_l4123_412334

def days_until_birthday : ℕ := 22
def savings_per_day : ℚ := 2
def cost_per_flower : ℚ := 4

theorem flowers_lilly_can_buy :
  (days_until_birthday : ℚ) * savings_per_day / cost_per_flower = 11 := by
  sorry

end NUMINAMATH_CALUDE_flowers_lilly_can_buy_l4123_412334


namespace NUMINAMATH_CALUDE_value_range_of_f_l4123_412377

-- Define the function
def f (x : ℝ) : ℝ := |x - 3| + 1

-- Define the domain
def domain : Set ℝ := Set.Icc 0 9

-- Theorem statement
theorem value_range_of_f :
  Set.Icc 1 7 = (Set.image f domain) := by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l4123_412377


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l4123_412301

/-- Proves that the ratio of downstream to upstream speed is 2:1 for a boat in a river --/
theorem boat_speed_ratio (v : ℝ) : 
  v > 3 →  -- Boat speed must be greater than river flow
  (4 / (v + 3) + 4 / (v - 3) = 1) →  -- Total travel time is 1 hour
  ((v + 3) / (v - 3) = 2) :=  -- Ratio of downstream to upstream speed
by
  sorry

#check boat_speed_ratio

end NUMINAMATH_CALUDE_boat_speed_ratio_l4123_412301
