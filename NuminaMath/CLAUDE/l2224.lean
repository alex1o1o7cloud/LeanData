import Mathlib

namespace NUMINAMATH_CALUDE_stack_probability_exact_l2224_222472

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates n! (n factorial) -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of crates in the stack -/
def numCrates : ℕ := 15

/-- The dimensions of each crate -/
def crateDim : CrateDimensions := ⟨2, 5, 7⟩

/-- The target height of the stack -/
def targetHeight : ℕ := 60

/-- The total number of possible orientations for the stack -/
def totalOrientations : ℕ := 3^numCrates

/-- The number of valid orientations that result in the target height -/
def validOrientations : ℕ := 
  choose numCrates 5 * choose 10 10 +
  choose numCrates 7 * choose 8 5 * choose 3 3 +
  choose numCrates 9 * choose 6 6

/-- The probability of the stack being exactly 60ft tall -/
def stackProbability : ℚ := validOrientations / totalOrientations

theorem stack_probability_exact : 
  stackProbability = 158158 / 14348907 := by sorry

end NUMINAMATH_CALUDE_stack_probability_exact_l2224_222472


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_l2224_222487

theorem smallest_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  5 ∣ n ∧ 6 ∣ n ∧ 2 ∣ n ∧
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) → 5 ∣ m → 6 ∣ m → 2 ∣ m → m ≥ n) ∧
  n = 1020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_l2224_222487


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l2224_222429

theorem multiplication_addition_equality : 42 * 25 + 58 * 42 = 3486 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l2224_222429


namespace NUMINAMATH_CALUDE_theater_parking_increase_l2224_222475

/-- Calculates the net increase in vehicles during a theater play --/
def net_increase_vehicles (play_duration : ℝ) 
  (car_arrival_rate car_departure_rate : ℝ)
  (motorcycle_arrival_rate motorcycle_departure_rate : ℝ)
  (van_arrival_rate van_departure_rate : ℝ) :
  (ℝ × ℝ × ℝ) :=
  let net_car_increase := (car_arrival_rate - car_departure_rate) * play_duration
  let net_motorcycle_increase := (motorcycle_arrival_rate - motorcycle_departure_rate) * play_duration
  let net_van_increase := (van_arrival_rate - van_departure_rate) * play_duration
  (net_car_increase, net_motorcycle_increase, net_van_increase)

/-- Theorem stating the net increase in vehicles during the theater play --/
theorem theater_parking_increase :
  let play_duration : ℝ := 2.5
  let car_arrival_rate : ℝ := 70
  let car_departure_rate : ℝ := 40
  let motorcycle_arrival_rate : ℝ := 120
  let motorcycle_departure_rate : ℝ := 60
  let van_arrival_rate : ℝ := 30
  let van_departure_rate : ℝ := 20
  net_increase_vehicles play_duration 
    car_arrival_rate car_departure_rate
    motorcycle_arrival_rate motorcycle_departure_rate
    van_arrival_rate van_departure_rate = (75, 150, 25) := by
  sorry

end NUMINAMATH_CALUDE_theater_parking_increase_l2224_222475


namespace NUMINAMATH_CALUDE_problem_statement_l2224_222479

theorem problem_statement (a : ℝ) (h : a = 5 - 2 * Real.sqrt 6) : a^2 - 10*a + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2224_222479


namespace NUMINAMATH_CALUDE_james_writing_speed_l2224_222426

/-- Represents the writing schedule and book information --/
structure WritingInfo where
  hours_per_day : ℕ
  weeks : ℕ
  total_pages : ℕ

/-- Calculates the number of pages written per hour --/
def pages_per_hour (info : WritingInfo) : ℚ :=
  info.total_pages / (info.hours_per_day * 7 * info.weeks)

/-- Theorem stating that given the specific writing schedule and book length,
    the number of pages written per hour is 5 --/
theorem james_writing_speed :
  let info : WritingInfo := ⟨3, 7, 735⟩
  pages_per_hour info = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_writing_speed_l2224_222426


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l2224_222466

theorem degree_to_radian_conversion (angle_deg : ℝ) (angle_rad : ℝ) : 
  angle_deg = 15 → angle_rad = π / 12 := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l2224_222466


namespace NUMINAMATH_CALUDE_binary_1011001100_equals_octal_5460_l2224_222453

def binary_to_octal (b : ℕ) : ℕ :=
  sorry

theorem binary_1011001100_equals_octal_5460 :
  binary_to_octal 1011001100 = 5460 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011001100_equals_octal_5460_l2224_222453


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l2224_222413

theorem quadratic_roots_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → x₂^2 - 3*x₂ + 2 = 0 → x₁ + x₂ - x₁ * x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l2224_222413


namespace NUMINAMATH_CALUDE_pqr_product_l2224_222431

theorem pqr_product (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p)
  (h1 : p + 2 / q = q + 2 / r) (h2 : q + 2 / r = r + 2 / p) :
  |p * q * r| = 2 := by
  sorry

end NUMINAMATH_CALUDE_pqr_product_l2224_222431


namespace NUMINAMATH_CALUDE_log_101600_value_l2224_222410

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_101600_value (h : log 102 = 0.3010) : log 101600 = 3.3010 := by
  sorry

end NUMINAMATH_CALUDE_log_101600_value_l2224_222410


namespace NUMINAMATH_CALUDE_proposition_a_is_true_l2224_222470

theorem proposition_a_is_true : ∀ x : ℝ, x = 2 → (x - 2) * (x - 1) = 0 := by
  sorry

#check proposition_a_is_true

end NUMINAMATH_CALUDE_proposition_a_is_true_l2224_222470


namespace NUMINAMATH_CALUDE_shaded_area_of_square_with_rectangles_shaded_area_is_22_l2224_222454

/-- The area of the shaded L-shaped region in a square with three rectangles removed -/
theorem shaded_area_of_square_with_rectangles (side_length : ℝ) 
  (rect1_length rect1_width : ℝ) 
  (rect2_length rect2_width : ℝ) 
  (rect3_length rect3_width : ℝ) : ℝ :=
  side_length * side_length - (rect1_length * rect1_width + rect2_length * rect2_width + rect3_length * rect3_width)

/-- The area of the shaded L-shaped region is 22 square units -/
theorem shaded_area_is_22 :
  shaded_area_of_square_with_rectangles 6 3 1 4 2 1 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_with_rectangles_shaded_area_is_22_l2224_222454


namespace NUMINAMATH_CALUDE_atomic_number_relationship_l2224_222476

/-- Given three elements with atomic numbers R, M, and Z, if their ions R^(X-), M^(n+), and Z^(m+) 
    have the same electronic structure, and n > m, then M > Z > R. -/
theorem atomic_number_relationship (R M Z n m x : ℤ) 
  (h1 : R + x = M - n) 
  (h2 : R + x = Z - m) 
  (h3 : n > m) : 
  M > Z ∧ Z > R := by sorry

end NUMINAMATH_CALUDE_atomic_number_relationship_l2224_222476


namespace NUMINAMATH_CALUDE_smallest_class_size_l2224_222423

theorem smallest_class_size (n : ℕ) : 
  n > 0 ∧ 
  (6 * 120 + (n - 6) * 70 : ℝ) ≤ (n * 85 : ℝ) ∧ 
  (∀ m : ℕ, m > 0 → m < n → (6 * 120 + (m - 6) * 70 : ℝ) > (m * 85 : ℝ)) → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2224_222423


namespace NUMINAMATH_CALUDE_distance_between_specific_lines_l2224_222403

/-- Line represented by a parametric equation -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Line represented by a slope-intercept equation -/
structure SlopeInterceptLine where
  slope : ℝ
  intercept : ℝ

/-- The distance between two lines -/
def distance_between_lines (l₁ : ParametricLine) (l₂ : SlopeInterceptLine) : ℝ :=
  sorry

/-- The given problem statement -/
theorem distance_between_specific_lines :
  let l₁ : ParametricLine := {
    x := λ t => 1 + t,
    y := λ t => 1 + 3*t
  }
  let l₂ : SlopeInterceptLine := {
    slope := 3,
    intercept := 4
  }
  distance_between_lines l₁ l₂ = 3 * Real.sqrt 10 / 5 :=
sorry

end NUMINAMATH_CALUDE_distance_between_specific_lines_l2224_222403


namespace NUMINAMATH_CALUDE_total_price_is_530_l2224_222417

/-- The total price of hats given the number of hats, their prices, and the number of green hats. -/
def total_price (total_hats : ℕ) (blue_price green_price : ℕ) (green_hats : ℕ) : ℕ :=
  let blue_hats := total_hats - green_hats
  blue_price * blue_hats + green_price * green_hats

/-- Theorem stating that the total price of hats is $530 given the specific conditions. -/
theorem total_price_is_530 :
  total_price 85 6 7 20 = 530 :=
by sorry

end NUMINAMATH_CALUDE_total_price_is_530_l2224_222417


namespace NUMINAMATH_CALUDE_distance_to_circle_center_l2224_222497

/-- The distance from a point on y = 2x to the center of (x-8)^2 + (y-1)^2 = 2,
    given symmetric tangents -/
theorem distance_to_circle_center (P : ℝ × ℝ) : 
  (∃ t : ℝ, P.1 = t ∧ P.2 = 2*t) →  -- P is on the line y = 2x
  (∃ l₁ l₂ : ℝ × ℝ → Prop,  -- l₁ and l₂ are tangent lines
    (∀ Q : ℝ × ℝ, l₁ Q → (Q.1 - 8)^2 + (Q.2 - 1)^2 = 2) ∧
    (∀ Q : ℝ × ℝ, l₂ Q → (Q.1 - 8)^2 + (Q.2 - 1)^2 = 2) ∧
    l₁ P ∧ l₂ P ∧
    (∀ Q : ℝ × ℝ, l₁ Q ↔ l₂ (2*P.1 - Q.1, 2*P.2 - Q.2))) →  -- l₁ and l₂ are symmetric about y = 2x
  Real.sqrt ((P.1 - 8)^2 + (P.2 - 1)^2) = 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_circle_center_l2224_222497


namespace NUMINAMATH_CALUDE_fraction_simplification_l2224_222416

theorem fraction_simplification :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 3) = 177 / 182 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2224_222416


namespace NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l2224_222451

theorem equation_holds_iff_specific_values :
  ∀ (a b p q : ℝ),
  (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
  (a = 2 ∧ b = -1 ∧ p = -1 ∧ q = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l2224_222451


namespace NUMINAMATH_CALUDE_intercept_ratio_l2224_222463

/-- Given two lines intersecting the y-axis at different points:
    - Line 1 has y-intercept 2, slope 5, and x-intercept (u, 0)
    - Line 2 has y-intercept 3, slope -7, and x-intercept (v, 0)
    The ratio of u to v is -14/15 -/
theorem intercept_ratio (u v : ℝ) : 
  (2 : ℝ) + 5 * u = 0 →  -- Line 1 equation at x-intercept
  (3 : ℝ) - 7 * v = 0 →  -- Line 2 equation at x-intercept
  u / v = -14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_intercept_ratio_l2224_222463


namespace NUMINAMATH_CALUDE_min_max_z_values_l2224_222484

theorem min_max_z_values (x y z : ℝ) 
  (h1 : x^2 ≤ y + z) 
  (h2 : y^2 ≤ z + x) 
  (h3 : z^2 ≤ x + y) : 
  (-1/4 : ℝ) ≤ z ∧ z ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_max_z_values_l2224_222484


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l2224_222457

/-- The fraction of the total area shaded in each iteration -/
def shaded_fraction : ℚ := 4 / 6

/-- The fraction of the remaining area subdivided in each iteration -/
def subdivision_fraction : ℚ := 1 / 6

/-- The sum of the shaded areas in an infinitely divided rectangle -/
def shaded_area_sum : ℚ := shaded_fraction / (1 - subdivision_fraction)

/-- 
Theorem: The sum of the shaded area in an infinitely divided rectangle, 
where 4/6 of each central subdivision is shaded in each iteration, 
is equal to 4/5 of the total area.
-/
theorem shaded_area_theorem : shaded_area_sum = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l2224_222457


namespace NUMINAMATH_CALUDE_chris_current_age_l2224_222492

-- Define Praveen's current age
def praveen_age : ℝ := sorry

-- Define Chris's current age
def chris_age : ℝ := sorry

-- Condition 1: Praveen's age after 10 years is 3 times his age 3 years back
axiom praveen_age_condition : praveen_age + 10 = 3 * (praveen_age - 3)

-- Condition 2: Chris is 2 years younger than Praveen was 4 years ago
axiom chris_age_condition : chris_age = (praveen_age - 4) - 2

-- Theorem to prove
theorem chris_current_age : chris_age = 3.5 := by sorry

end NUMINAMATH_CALUDE_chris_current_age_l2224_222492


namespace NUMINAMATH_CALUDE_sports_books_count_l2224_222469

theorem sports_books_count (total_books school_books : ℕ) 
  (h1 : total_books = 58) 
  (h2 : school_books = 19) : 
  total_books - school_books = 39 := by
sorry

end NUMINAMATH_CALUDE_sports_books_count_l2224_222469


namespace NUMINAMATH_CALUDE_variance_linear_transformation_l2224_222452

def variance (data : List ℝ) : ℝ := sorry

theorem variance_linear_transformation 
  (data : List ℝ) 
  (h : variance data = 1/3) : 
  variance (data.map (λ x => 3*x - 1)) = 3 := by sorry

end NUMINAMATH_CALUDE_variance_linear_transformation_l2224_222452


namespace NUMINAMATH_CALUDE_string_average_length_l2224_222465

theorem string_average_length (s₁ s₂ s₃ : ℝ) (h₁ : s₁ = 2) (h₂ : s₂ = 5) (h₃ : s₃ = 7) :
  (s₁ + s₂ + s₃) / 3 = 14 / 3 := by
  sorry

#check string_average_length

end NUMINAMATH_CALUDE_string_average_length_l2224_222465


namespace NUMINAMATH_CALUDE_negative_five_is_square_root_of_twenty_five_l2224_222480

theorem negative_five_is_square_root_of_twenty_five : ∃ x : ℝ, x^2 = 25 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_is_square_root_of_twenty_five_l2224_222480


namespace NUMINAMATH_CALUDE_range_of_y_l2224_222411

-- Define the function f
def f (x : ℝ) : ℝ := |x^2 - x| - 4*x

-- State the theorem
theorem range_of_y (y : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0) ↔ -4 < y ∧ y < 12 :=
sorry

end NUMINAMATH_CALUDE_range_of_y_l2224_222411


namespace NUMINAMATH_CALUDE_absolute_value_comparison_l2224_222474

theorem absolute_value_comparison (m n : ℝ) : m < n → n < 0 → abs m > abs n := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_comparison_l2224_222474


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2224_222460

theorem coin_flip_probability :
  let p : ℝ := 1/3  -- Probability of getting heads in a single flip
  let q : ℝ := 1 - p  -- Probability of getting tails in a single flip
  let num_players : ℕ := 4  -- Number of players
  let prob_same_flips : ℝ := (p^num_players) * (∑' n, q^(num_players * n)) -- Probability all players flip same number of times
  prob_same_flips = 1/65
  := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2224_222460


namespace NUMINAMATH_CALUDE_birdhouse_planks_l2224_222481

/-- The number of planks required to build one birdhouse -/
def planks_per_birdhouse : ℕ := sorry

/-- The number of nails required to build one birdhouse -/
def nails_per_birdhouse : ℕ := 20

/-- The cost of one nail in cents -/
def nail_cost : ℕ := 5

/-- The cost of one plank in cents -/
def plank_cost : ℕ := 300

/-- The total cost to build 4 birdhouses in cents -/
def total_cost_4_birdhouses : ℕ := 8800

theorem birdhouse_planks :
  planks_per_birdhouse = 7 ∧
  4 * (nails_per_birdhouse * nail_cost + planks_per_birdhouse * plank_cost) = total_cost_4_birdhouses :=
sorry

end NUMINAMATH_CALUDE_birdhouse_planks_l2224_222481


namespace NUMINAMATH_CALUDE_square_difference_formula_l2224_222498

theorem square_difference_formula (x y A : ℝ) : 
  (3*x + 2*y)^2 = (3*x - 2*y)^2 + A → A = 24*x*y := by sorry

end NUMINAMATH_CALUDE_square_difference_formula_l2224_222498


namespace NUMINAMATH_CALUDE_homework_difference_l2224_222448

/-- The number of pages of reading homework -/
def reading_pages : ℕ := 6

/-- The number of pages of math homework -/
def math_pages : ℕ := 10

/-- The number of pages of science homework -/
def science_pages : ℕ := 3

/-- The number of pages of history homework -/
def history_pages : ℕ := 5

/-- The theorem states that the difference between math homework pages and the sum of reading, science, and history homework pages is -4 -/
theorem homework_difference : 
  (math_pages : ℤ) - (reading_pages + science_pages + history_pages : ℤ) = -4 := by
  sorry

end NUMINAMATH_CALUDE_homework_difference_l2224_222448


namespace NUMINAMATH_CALUDE_randolph_age_l2224_222495

/-- Proves that Randolph's age is 55 given the conditions of the problem -/
theorem randolph_age :
  (∀ (sherry sydney randolph : ℕ),
    randolph = sydney + 5 →
    sydney = 2 * sherry →
    sherry = 25 →
    randolph = 55) :=
by sorry

end NUMINAMATH_CALUDE_randolph_age_l2224_222495


namespace NUMINAMATH_CALUDE_not_sum_of_six_odd_squares_l2224_222449

theorem not_sum_of_six_odd_squares (n : ℕ) : n = 1986 → ¬ ∃ (a b c d e f : ℕ), 
  (∃ (k₁ k₂ k₃ k₄ k₅ k₆ : ℕ), 
    a = 2 * k₁ + 1 ∧ 
    b = 2 * k₂ + 1 ∧ 
    c = 2 * k₃ + 1 ∧ 
    d = 2 * k₄ + 1 ∧ 
    e = 2 * k₅ + 1 ∧ 
    f = 2 * k₆ + 1) ∧ 
  n = a^2 + b^2 + c^2 + d^2 + e^2 + f^2 :=
by sorry

end NUMINAMATH_CALUDE_not_sum_of_six_odd_squares_l2224_222449


namespace NUMINAMATH_CALUDE_integer_solutions_for_k_l2224_222459

theorem integer_solutions_for_k (k : ℤ) : 
  (∃ x : ℤ, 9 * x - 3 = k * x + 14) ↔ k ∈ ({8, 10, -8, 26} : Set ℤ) := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_for_k_l2224_222459


namespace NUMINAMATH_CALUDE_cakes_sold_daily_l2224_222428

def cash_register_cost : ℕ := 1040
def bread_price : ℕ := 2
def bread_quantity : ℕ := 40
def cake_price : ℕ := 12
def daily_rent : ℕ := 20
def daily_electricity : ℕ := 2
def days_to_pay : ℕ := 8

def daily_bread_income : ℕ := bread_price * bread_quantity
def daily_expenses : ℕ := daily_rent + daily_electricity
def daily_profit_from_bread : ℕ := daily_bread_income - daily_expenses

theorem cakes_sold_daily (cakes_sold : ℕ) : 
  cakes_sold = 6 ↔ 
  days_to_pay * (daily_profit_from_bread + cake_price * cakes_sold) = cash_register_cost :=
by sorry

end NUMINAMATH_CALUDE_cakes_sold_daily_l2224_222428


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_implies_a_geq_7_l2224_222414

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the first question
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Theorem for the second question
theorem A_subset_C_implies_a_geq_7 (a : ℝ) :
  A ⊆ C a → a ≥ 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_implies_a_geq_7_l2224_222414


namespace NUMINAMATH_CALUDE_sin_B_value_l2224_222405

-- Define a right triangle ABC
structure RightTriangle :=
  (A B C : Real)
  (right_angle : C = 90)
  (bc_half_ac : B = 1/2 * A)

-- Theorem statement
theorem sin_B_value (t : RightTriangle) : Real.sin (t.B) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_B_value_l2224_222405


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2224_222400

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 150 → volume = 125 → ∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧ volume = side_length^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2224_222400


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l2224_222446

theorem smallest_factorization_coefficient : 
  ∃ (b : ℕ), b = 95 ∧ 
  (∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b*x + 2016 = (x + p) * (x + q)) ∧
  (∀ (b' : ℕ), b' < b → 
    ¬(∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b'*x + 2016 = (x + p) * (x + q))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l2224_222446


namespace NUMINAMATH_CALUDE_max_ratio_inscribed_circumscribed_sphere_radii_l2224_222442

/-- Given a right square pyramid with circumscribed and inscribed spheres, 
    this theorem states the maximum ratio of their radii. -/
theorem max_ratio_inscribed_circumscribed_sphere_radii 
  (R r d : ℝ) 
  (h_positive : R > 0 ∧ r > 0)
  (h_relation : d^2 + (R + r)^2 = 2 * R^2) :
  ∃ (max_ratio : ℝ), max_ratio = Real.sqrt 2 - 1 ∧ 
    r / R ≤ max_ratio ∧ 
    ∃ (r' d' : ℝ), r' / R = max_ratio ∧ 
      d'^2 + (R + r')^2 = 2 * R^2 := by
sorry

end NUMINAMATH_CALUDE_max_ratio_inscribed_circumscribed_sphere_radii_l2224_222442


namespace NUMINAMATH_CALUDE_total_valid_words_count_l2224_222421

def alphabet_size : ℕ := 25
def max_word_length : ℕ := 5

def valid_words (n : ℕ) : ℕ :=
  if n < 2 then 0
  else if n = 2 then 2
  else Nat.choose n 2 * alphabet_size ^ (n - 2)

def total_valid_words : ℕ :=
  (List.range (max_word_length - 1)).map (fun i => valid_words (i + 2))
    |> List.sum

theorem total_valid_words_count :
  total_valid_words = 160075 := by sorry

end NUMINAMATH_CALUDE_total_valid_words_count_l2224_222421


namespace NUMINAMATH_CALUDE_discount_difference_is_187_point_5_l2224_222443

def initial_amount : ℝ := 15000

def single_discount_rate : ℝ := 0.3
def first_successive_discount_rate : ℝ := 0.25
def second_successive_discount_rate : ℝ := 0.05

def single_discount_amount : ℝ := initial_amount * (1 - single_discount_rate)

def successive_discount_amount : ℝ :=
  initial_amount * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)

theorem discount_difference_is_187_point_5 :
  successive_discount_amount - single_discount_amount = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_is_187_point_5_l2224_222443


namespace NUMINAMATH_CALUDE_sum_of_squared_even_differences_l2224_222473

theorem sum_of_squared_even_differences : 
  (20^2 - 18^2) + (16^2 - 14^2) + (12^2 - 10^2) + (8^2 - 6^2) + (4^2 - 2^2) = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_even_differences_l2224_222473


namespace NUMINAMATH_CALUDE_subtracted_number_l2224_222486

theorem subtracted_number (x : ℕ) : 10000 - x = 9001 → x = 999 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l2224_222486


namespace NUMINAMATH_CALUDE_problem_statement_l2224_222478

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- State the theorem
theorem problem_statement (a b : ℝ) 
  (h1 : 0 < a ∧ a < 1/2) 
  (h2 : 0 < b ∧ b < 1/2) 
  (h3 : f (1/a) + f (2/b) = 10) : 
  a + b/2 ≥ 2/7 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2224_222478


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2224_222415

/-- Given a sphere with volume 72π cubic inches, its surface area is 36π * 2^(2/3) square inches. -/
theorem sphere_surface_area (V : ℝ) (r : ℝ) (S : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  S = 4 * Real.pi * r^2 → 
  S = 36 * Real.pi * 2^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2224_222415


namespace NUMINAMATH_CALUDE_flower_path_distance_l2224_222438

/-- Given eight equally spaced flowers along a straight path, 
    where the distance between the first and fifth flower is 80 meters, 
    prove that the distance between the first and last flower is 140 meters. -/
theorem flower_path_distance :
  ∀ (flower_positions : ℕ → ℝ),
    (∀ i j : ℕ, i < j → flower_positions j - flower_positions i = (j - i : ℝ) * (flower_positions 1 - flower_positions 0)) →
    (flower_positions 4 - flower_positions 0 = 80) →
    (flower_positions 7 - flower_positions 0 = 140) :=
by sorry

end NUMINAMATH_CALUDE_flower_path_distance_l2224_222438


namespace NUMINAMATH_CALUDE_line_inclination_gt_45_deg_l2224_222496

/-- The angle of inclination of a line ax + (a + 1)y + 2 = 0 is greater than 45° if and only if a < -1/2 or a > 0 -/
theorem line_inclination_gt_45_deg (a : ℝ) :
  let line := {(x, y) : ℝ × ℝ | a * x + (a + 1) * y + 2 = 0}
  let angle_of_inclination := Real.arctan (abs (a / (a + 1)))
  angle_of_inclination > Real.pi / 4 ↔ a < -1/2 ∨ a > 0 :=
sorry

end NUMINAMATH_CALUDE_line_inclination_gt_45_deg_l2224_222496


namespace NUMINAMATH_CALUDE_perpendicular_bisector_trajectory_l2224_222482

theorem perpendicular_bisector_trajectory (Z₁ Z₂ : ℂ) (h : Z₁ ≠ Z₂) :
  {Z : ℂ | Complex.abs (Z - Z₁) = Complex.abs (Z - Z₂)} =
  {Z : ℂ | (Z - (Z₁ + Z₂) / 2) • (Z₂ - Z₁) = 0} :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_trajectory_l2224_222482


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_B_complement_union_A_B_l2224_222441

-- Define the universal set U
def U : Set ℝ := {x | x^2 - 3*x + 2 ≥ 0}

-- Define set A
def A : Set ℝ := {x | |x - 2| > 1}

-- Define set B
def B : Set ℝ := {x | (x - 1) / (x - 2) > 0}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x | x < 1 ∨ x > 3} := by sorry

theorem intersection_A_complement_B : A ∩ (U \ B) = ∅ := by sorry

theorem complement_union_A_B : U \ (A ∪ B) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_B_complement_union_A_B_l2224_222441


namespace NUMINAMATH_CALUDE_quarter_circle_radius_l2224_222468

theorem quarter_circle_radius (x y z : ℝ) (h_right_angle : x^2 + y^2 = z^2)
  (h_xy_area : π * x^2 / 4 = 2 * π) (h_xz_arc : π * y / 2 = 6 * π) :
  z / 2 = Real.sqrt 152 := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_radius_l2224_222468


namespace NUMINAMATH_CALUDE_pattern_calculation_main_calculation_l2224_222456

theorem pattern_calculation : ℕ → Prop :=
  fun n => n * (n + 1) + (n + 1) * (n + 2) = 2 * (n + 1) * (n + 1)

theorem main_calculation : 
  75 * 222 + 76 * 225 - 25 * 14 * 15 - 25 * 15 * 16 = 302 := by
  sorry

end NUMINAMATH_CALUDE_pattern_calculation_main_calculation_l2224_222456


namespace NUMINAMATH_CALUDE_handshake_count_gathering_handshakes_l2224_222489

theorem handshake_count (twin_sets : ℕ) (triplet_sets : ℕ) : ℕ :=
  let twins := 2 * twin_sets
  let triplets := 3 * triplet_sets
  let twin_handshakes := twins * (twins - 2) / 2
  let cross_handshakes := twins * triplets
  twin_handshakes + cross_handshakes

theorem gathering_handshakes :
  handshake_count 8 5 = 352 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_gathering_handshakes_l2224_222489


namespace NUMINAMATH_CALUDE_sin_a_less_cos_b_in_obtuse_triangle_l2224_222409

/-- In a triangle ABC where angle C is obtuse, sin A < cos B -/
theorem sin_a_less_cos_b_in_obtuse_triangle (A B C : ℝ) (h_triangle : A + B + C = π) (h_obtuse : C > π/2) : 
  Real.sin A < Real.cos B := by
sorry

end NUMINAMATH_CALUDE_sin_a_less_cos_b_in_obtuse_triangle_l2224_222409


namespace NUMINAMATH_CALUDE_three_numbers_sequence_l2224_222420

theorem three_numbers_sequence (x y z : ℝ) : 
  (x + y + z = 35 ∧ 
   2 * y = x + z + 1 ∧ 
   y^2 = (x + 3) * z) → 
  ((x = 15 ∧ y = 12 ∧ z = 8) ∨ 
   (x = 5 ∧ y = 12 ∧ z = 18)) := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sequence_l2224_222420


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2224_222467

theorem cube_volume_ratio (q p : ℝ) (h : p = 3 * q) : q^3 / p^3 = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2224_222467


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2224_222444

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y = 1/x + 4/y + 8) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 1/a + 4/b + 8 → x + y ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2224_222444


namespace NUMINAMATH_CALUDE_annie_children_fruits_l2224_222404

/-- The number of fruits Annie's children received -/
def total_fruits (mike_oranges matt_apples mark_bananas : ℕ) : ℕ :=
  mike_oranges + matt_apples + mark_bananas

theorem annie_children_fruits :
  ∃ (mike_oranges matt_apples mark_bananas : ℕ),
    mike_oranges = 3 ∧
    matt_apples = 2 * mike_oranges ∧
    mark_bananas = mike_oranges + matt_apples ∧
    total_fruits mike_oranges matt_apples mark_bananas = 18 := by
  sorry

end NUMINAMATH_CALUDE_annie_children_fruits_l2224_222404


namespace NUMINAMATH_CALUDE_unique_solution_l2224_222499

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  x > 0 ∧ (x - 2) > 0 ∧ (x + 2) > 0 ∧
  log10 x + log10 (x - 2) = log10 3 + log10 (x + 2)

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2224_222499


namespace NUMINAMATH_CALUDE_percentage_of_female_employees_l2224_222455

theorem percentage_of_female_employees (total_employees : ℕ) 
  (computer_literate_percentage : ℚ) (female_computer_literate : ℕ) 
  (male_computer_literate_percentage : ℚ) :
  total_employees = 1100 →
  computer_literate_percentage = 62 / 100 →
  female_computer_literate = 462 →
  male_computer_literate_percentage = 1 / 2 →
  (↑female_computer_literate + (male_computer_literate_percentage * ↑(total_employees - female_computer_literate / computer_literate_percentage))) / ↑total_employees = 3 / 5 := by
  sorry

#check percentage_of_female_employees

end NUMINAMATH_CALUDE_percentage_of_female_employees_l2224_222455


namespace NUMINAMATH_CALUDE_red_to_colored_lipstick_ratio_l2224_222425

/-- Represents the number of students who attended school -/
def total_students : ℕ := 200

/-- Represents the number of students who wore blue lipstick -/
def blue_lipstick_students : ℕ := 5

/-- Represents the number of students who wore colored lipstick -/
def colored_lipstick_students : ℕ := total_students / 2

/-- Represents the number of students who wore red lipstick -/
def red_lipstick_students : ℕ := blue_lipstick_students * 5

/-- Theorem stating the ratio of students who wore red lipstick to those who wore colored lipstick -/
theorem red_to_colored_lipstick_ratio :
  (red_lipstick_students : ℚ) / colored_lipstick_students = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_red_to_colored_lipstick_ratio_l2224_222425


namespace NUMINAMATH_CALUDE_problem_1_l2224_222430

theorem problem_1 : 2023 * 2023 - 2024 * 2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2224_222430


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2224_222434

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  second_quadrant (-3) 2 := by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2224_222434


namespace NUMINAMATH_CALUDE_simplify_expression_l2224_222408

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a^2 + b^2) :
  a^2 / b + b^2 / a - 1 / (a^2 * b^2) = (a^4 + 2*a*b + b^4 - 1) / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2224_222408


namespace NUMINAMATH_CALUDE_train_crossing_time_l2224_222447

/-- The time taken for a train to cross a man walking in the same direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 600 →
  train_speed = 56 * 1000 / 3600 →
  man_speed = 2 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 40 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2224_222447


namespace NUMINAMATH_CALUDE_james_tshirt_cost_l2224_222488

def calculate_total_cost (num_shirts : ℕ) (discount_rate : ℚ) (original_price : ℚ) : ℚ :=
  num_shirts * (original_price * (1 - discount_rate))

theorem james_tshirt_cost :
  calculate_total_cost 6 (1/2) 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_tshirt_cost_l2224_222488


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2224_222461

theorem sum_of_roots_quadratic (x : ℝ) : 
  (2 * x^2 - 5 * x + 3 = 9) → 
  (∃ y : ℝ, 2 * y^2 - 5 * y + 3 = 9 ∧ x + y = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2224_222461


namespace NUMINAMATH_CALUDE_andy_location_after_10_turns_l2224_222401

/-- Represents a direction on the coordinate plane -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Andy's position and facing direction -/
structure State where
  x : Int
  y : Int
  dir : Direction
  moveCount : Nat

/-- Turns the current direction 90 degrees right -/
def turnRight (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

/-- Moves Andy according to his current state -/
def move (s : State) : State :=
  let newMoveCount := s.moveCount + 1
  match s.dir with
  | Direction.North => { s with y := s.y + newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }
  | Direction.East => { s with x := s.x + newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }
  | Direction.South => { s with y := s.y - newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }
  | Direction.West => { s with x := s.x - newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }

/-- Applies the move function n times to the initial state -/
def applyMoves (n : Nat) : State :=
  match n with
  | 0 => { x := 0, y := 0, dir := Direction.North, moveCount := 0 }
  | n + 1 => move (applyMoves n)

theorem andy_location_after_10_turns :
  let finalState := applyMoves 10
  finalState.x = 6 ∧ finalState.y = 5 :=
sorry

end NUMINAMATH_CALUDE_andy_location_after_10_turns_l2224_222401


namespace NUMINAMATH_CALUDE_wide_right_field_goals_l2224_222435

theorem wide_right_field_goals 
  (total_attempts : ℕ) 
  (missed_fraction : ℚ) 
  (wide_right_percentage : ℚ) : ℕ :=
by
  have h1 : total_attempts = 60 := by sorry
  have h2 : missed_fraction = 1 / 4 := by sorry
  have h3 : wide_right_percentage = 1 / 5 := by sorry
  
  let missed_goals := total_attempts * missed_fraction
  let wide_right_goals := missed_goals * wide_right_percentage
  
  exact 3
  
#check wide_right_field_goals

end NUMINAMATH_CALUDE_wide_right_field_goals_l2224_222435


namespace NUMINAMATH_CALUDE_complex_polygon_area_theorem_l2224_222406

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side_length : ℝ)

/-- Represents the configuration of three overlapping sheets -/
structure SheetConfiguration :=
  (bottom : Sheet)
  (middle : Sheet)
  (top : Sheet)
  (middle_rotation : ℝ)
  (top_rotation : ℝ)
  (top_shift : ℝ)

/-- Calculates the area of the complex polygon formed by overlapping sheets -/
noncomputable def complex_polygon_area (config : SheetConfiguration) : ℝ :=
  sorry

/-- The main theorem stating the area of the complex polygon -/
theorem complex_polygon_area_theorem (config : SheetConfiguration) :
  config.bottom.side_length = 8 ∧
  config.middle.side_length = 8 ∧
  config.top.side_length = 8 ∧
  config.middle_rotation = 45 ∧
  config.top_rotation = 90 ∧
  config.top_shift = 4 →
  complex_polygon_area config = 144 :=
by sorry

end NUMINAMATH_CALUDE_complex_polygon_area_theorem_l2224_222406


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l2224_222491

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l2224_222491


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2224_222436

theorem line_segment_endpoint (x : ℝ) : 
  (∃ (y : ℝ), (x = 3 - Real.sqrt 69 ∨ x = 3 + Real.sqrt 69) ∧ 
   ((3 - x)^2 + (8 - (-2))^2 = 13^2)) ↔ 
  (∃ (y : ℝ), ((3 - x)^2 + (y - (-2))^2 = 13^2) ∧ y = 8) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2224_222436


namespace NUMINAMATH_CALUDE_multiplicative_inverse_5_mod_31_l2224_222422

theorem multiplicative_inverse_5_mod_31 : ∃ x : ℤ, (5 * x) % 31 = 1 ∧ x % 31 = 25 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_5_mod_31_l2224_222422


namespace NUMINAMATH_CALUDE_exist_three_permuted_numbers_l2224_222432

/-- A function that checks if a number is a five-digit number in the decimal system -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A function that checks if two numbers are permutations of each other -/
def isPermutation (a b : ℕ) : Prop :=
  ∃ (digits_a digits_b : List ℕ),
    digits_a.length = 5 ∧
    digits_b.length = 5 ∧
    digits_a.toFinset = digits_b.toFinset ∧
    a = digits_a.foldl (fun acc d => acc * 10 + d) 0 ∧
    b = digits_b.foldl (fun acc d => acc * 10 + d) 0

/-- Theorem stating that there exist three five-digit numbers that are permutations of each other,
    where the sum of two equals twice the third -/
theorem exist_three_permuted_numbers :
  ∃ (a b c : ℕ),
    isFiveDigit a ∧ isFiveDigit b ∧ isFiveDigit c ∧
    isPermutation a b ∧ isPermutation b c ∧ isPermutation a c ∧
    a + b = 2 * c := by
  sorry

end NUMINAMATH_CALUDE_exist_three_permuted_numbers_l2224_222432


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2224_222440

/-- The perimeter of an equilateral triangle with an inscribed circle of radius 2 cm -/
theorem equilateral_triangle_perimeter (r : ℝ) (h : r = 2) :
  let a := 2 * r * Real.sqrt 3
  3 * a = 12 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2224_222440


namespace NUMINAMATH_CALUDE_gcd_of_2535_5929_11629_l2224_222412

theorem gcd_of_2535_5929_11629 : Nat.gcd 2535 (Nat.gcd 5929 11629) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_2535_5929_11629_l2224_222412


namespace NUMINAMATH_CALUDE_remainder_equality_l2224_222490

theorem remainder_equality (a b k : ℤ) (h : k ∣ (a - b)) : a % k = b % k := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l2224_222490


namespace NUMINAMATH_CALUDE_geometric_progression_middle_term_l2224_222418

theorem geometric_progression_middle_term :
  ∀ m : ℝ,
  (∃ r : ℝ, (m / (5 + 2 * Real.sqrt 6) = r) ∧ ((5 - 2 * Real.sqrt 6) / m = r)) →
  (m = 1 ∨ m = -1) :=
λ m h => by sorry

end NUMINAMATH_CALUDE_geometric_progression_middle_term_l2224_222418


namespace NUMINAMATH_CALUDE_work_completion_time_l2224_222477

/-- The number of days y needs to finish the work -/
def y_days : ℝ := 15

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 10

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 6.000000000000001

/-- The number of days x needs to finish the entire work alone -/
def x_days : ℝ := 18

theorem work_completion_time :
  y_days = 15 ∧ y_worked = 10 ∧ x_remaining = 6.000000000000001 →
  x_days = 18 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2224_222477


namespace NUMINAMATH_CALUDE_min_mozart_bach_not_beethoven_l2224_222445

theorem min_mozart_bach_not_beethoven 
  (total : ℕ) 
  (mozart : ℕ) 
  (bach : ℕ) 
  (beethoven : ℕ) 
  (h1 : total = 200)
  (h2 : mozart = 160)
  (h3 : bach = 120)
  (h4 : beethoven = 90)
  : ∃ (x : ℕ), x ≥ 10 ∧ 
    x ≤ mozart - beethoven ∧ 
    x ≤ bach - beethoven ∧ 
    x ≤ total - beethoven ∧
    x = min (mozart - beethoven) (min (bach - beethoven) (total - beethoven)) :=
by sorry

end NUMINAMATH_CALUDE_min_mozart_bach_not_beethoven_l2224_222445


namespace NUMINAMATH_CALUDE_min_value_problem_l2224_222483

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1/a + 1/b) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1/x + 1/y → 1/x + 2/y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l2224_222483


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l2224_222427

theorem arithmetic_geometric_mean_ratio 
  (x y : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h_eq : ((x + y) / 2) + Real.sqrt (x * y) = y - x) : 
  x / y = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l2224_222427


namespace NUMINAMATH_CALUDE_units_digit_of_15_to_15_l2224_222464

theorem units_digit_of_15_to_15 : ∃ n : ℕ, 15^15 ≡ 5 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_15_to_15_l2224_222464


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l2224_222493

/-- Given two concentric circles where a chord of length 80 units is tangent to the smaller circle,
    the area between the two circles is equal to 1600π square units. -/
theorem area_between_concentric_circles (O : ℝ × ℝ) (r₁ r₂ : ℝ) (A B : ℝ × ℝ) :
  let circle₁ := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r₁^2}
  let circle₂ := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r₂^2}
  r₁ > r₂ →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 80^2 →
  ∃ P ∈ circle₂, (P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2) = 0 →
  π * (r₁^2 - r₂^2) = 1600 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l2224_222493


namespace NUMINAMATH_CALUDE_cd_cost_fraction_l2224_222402

theorem cd_cost_fraction (m : ℝ) (n : ℕ) (h : n > 0) : 
  let total_cd_cost : ℝ := 2 * (1/3 * m)
  let cd_cost : ℝ := total_cd_cost / n
  let savings : ℝ := m - total_cd_cost
  (1/3 * m = (1/2 * n) * (cd_cost)) ∧ 
  (savings ≥ 1/4 * m) →
  cd_cost = 1/3 * m := by
sorry

end NUMINAMATH_CALUDE_cd_cost_fraction_l2224_222402


namespace NUMINAMATH_CALUDE_notched_circle_distance_l2224_222419

/-- Given a circle with radius √75 and a point B such that there exist points A and C on the circle
    where AB = 8, BC = 2, and angle ABC is a right angle, prove that the square of the distance
    from B to the center of the circle is 122. -/
theorem notched_circle_distance (O A B C : ℝ × ℝ) : 
  (∀ P : ℝ × ℝ, (P.1 - O.1)^2 + (P.2 - O.2)^2 = 75 → P = A ∨ P = C) →  -- A and C are on the circle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64 →  -- AB = 8
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 →   -- BC = 2
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →  -- Angle ABC is right angle
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = 122 := by
sorry


end NUMINAMATH_CALUDE_notched_circle_distance_l2224_222419


namespace NUMINAMATH_CALUDE_parallel_vectors_implies_x_squared_two_l2224_222437

/-- Two vectors in R^2 are parallel if and only if their cross product is zero -/
axiom parallel_iff_cross_product_zero {a b : ℝ × ℝ} :
  (∃ k : ℝ, a = k • b) ↔ a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b in R^2, if they are parallel, then x^2 = 2 -/
theorem parallel_vectors_implies_x_squared_two (x : ℝ) :
  let a : ℝ × ℝ := (x + 2, 1 + x)
  let b : ℝ × ℝ := (x - 2, 1 - x)
  (∃ k : ℝ, a = k • b) → x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_implies_x_squared_two_l2224_222437


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_6_l2224_222485

def vector_a (m : ℝ) : ℝ × ℝ := (2, m)
def vector_b : ℝ × ℝ := (1, -1)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

def vector_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

theorem perpendicular_vectors_m_equals_6 :
  ∀ m : ℝ, 
    let a := vector_a m
    let b := vector_b
    let sum := vector_add a (vector_scale 2 b)
    dot_product b sum = 0 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_6_l2224_222485


namespace NUMINAMATH_CALUDE_reader_group_size_l2224_222462

theorem reader_group_size (S L B : ℕ) (h1 : S = 250) (h2 : L = 230) (h3 : B = 80) :
  S + L - B = 400 := by
  sorry

end NUMINAMATH_CALUDE_reader_group_size_l2224_222462


namespace NUMINAMATH_CALUDE_min_value_zero_l2224_222450

theorem min_value_zero (x y : ℕ) (hx : x ≤ 2) (hy : y ≤ 3) :
  (x^2 * y^2 : ℝ) / (x^2 + y^2)^2 ≥ 0 ∧
  ∃ (a b : ℕ), a ≤ 2 ∧ b ≤ 3 ∧ (a^2 * b^2 : ℝ) / (a^2 + b^2)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_zero_l2224_222450


namespace NUMINAMATH_CALUDE_translation_theorem_l2224_222494

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point to the left by a given distance -/
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

/-- The theorem stating that translating point M(3, -4) 5 units to the left results in M'(-2, -4) -/
theorem translation_theorem :
  let M : Point := { x := 3, y := -4 }
  let M' : Point := translateLeft M 5
  M'.x = -2 ∧ M'.y = -4 := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l2224_222494


namespace NUMINAMATH_CALUDE_first_day_is_sunday_l2224_222439

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after n days -/
def afterDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (afterDays d n)

/-- Theorem: If the 18th day of a month is a Wednesday, then the 1st day of that month is a Sunday -/
theorem first_day_is_sunday (d : DayOfWeek) (h : afterDays d 17 = DayOfWeek.Wednesday) :
  d = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_first_day_is_sunday_l2224_222439


namespace NUMINAMATH_CALUDE_orchestra_members_count_l2224_222407

theorem orchestra_members_count : ∃! n : ℕ, 
  130 < n ∧ n < 260 ∧ 
  n % 6 = 1 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 3 ∧
  n = 241 := by
sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l2224_222407


namespace NUMINAMATH_CALUDE_mathopolis_intersections_l2224_222424

/-- A city with a grid-like street layout. -/
structure City where
  ns_streets : ℕ  -- Number of north-south streets
  ew_streets : ℕ  -- Number of east-west streets

/-- The number of intersections in a city with a grid-like street layout. -/
def num_intersections (c : City) : ℕ := c.ns_streets * c.ew_streets

/-- Mathopolis with its specific street layout. -/
def mathopolis : City := { ns_streets := 10, ew_streets := 10 }

/-- Theorem stating that Mathopolis has 100 intersections. -/
theorem mathopolis_intersections : num_intersections mathopolis = 100 := by
  sorry

#eval num_intersections mathopolis

end NUMINAMATH_CALUDE_mathopolis_intersections_l2224_222424


namespace NUMINAMATH_CALUDE_jason_commute_distance_l2224_222471

/-- Represents Jason's commute with convenience stores and a detour --/
structure JasonCommute where
  distance_house_to_first : ℝ
  distance_first_to_second : ℝ
  distance_second_to_third : ℝ
  distance_third_to_work : ℝ
  detour_distance : ℝ

/-- Calculates the total commute distance with detour --/
def total_commute_with_detour (j : JasonCommute) : ℝ :=
  j.distance_house_to_first + j.distance_first_to_second + 
  (j.distance_second_to_third + j.detour_distance) + j.distance_third_to_work

/-- Theorem stating Jason's commute distance with detour --/
theorem jason_commute_distance :
  ∀ j : JasonCommute,
  j.distance_house_to_first = 4 →
  j.distance_first_to_second = 6 →
  j.distance_second_to_third = j.distance_first_to_second + (2/3 * j.distance_first_to_second) →
  j.distance_third_to_work = j.distance_house_to_first →
  j.detour_distance = 3 →
  total_commute_with_detour j = 27 := by
  sorry

end NUMINAMATH_CALUDE_jason_commute_distance_l2224_222471


namespace NUMINAMATH_CALUDE_system_solution_product_l2224_222433

theorem system_solution_product : 
  ∃ (a b c d : ℚ),
    (4*a + 2*b + 6*c + 8*d = 48) ∧
    (2*(d+c) = b) ∧
    (4*b + 2*c = a) ∧
    (c + 2 = d) ∧
    (a * b * c * d = -88807680/4879681) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_product_l2224_222433


namespace NUMINAMATH_CALUDE_circle_properties_l2224_222458

-- Define the line
def line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the first circle (the one we're proving)
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

-- Define what it means for a point to be on a circle
def on_circle (x y : ℝ) (circle : ℝ → ℝ → Prop) : Prop := circle x y

-- Define what it means for a line to be tangent to a circle
def is_tangent (circle : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), on_circle x y circle ∧ line x y ∧
  ∀ (x' y' : ℝ), line x' y' → (x' - x)^2 + (y' - y)^2 ≥ 0

-- Define what it means for two circles to intersect
def circles_intersect (circle1 circle2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), on_circle x y circle1 ∧ on_circle x y circle2

-- State the theorem
theorem circle_properties :
  is_tangent circle1 line ∧ circles_intersect circle1 circle2 := by sorry

end NUMINAMATH_CALUDE_circle_properties_l2224_222458
