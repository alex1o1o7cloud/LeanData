import Mathlib

namespace NUMINAMATH_CALUDE_solution_in_interval_l3069_306904

def f (x : ℝ) := 4 * x^3 + x - 8

theorem solution_in_interval :
  (f 2 > 0) →
  (f 1.5 > 0) →
  (f 1 < 0) →
  ∃ x, x > 1 ∧ x < 1.5 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_in_interval_l3069_306904


namespace NUMINAMATH_CALUDE_mixed_grains_approximation_l3069_306993

/-- Given a total amount of grain and a sample with mixed grains, calculate the approximate amount of mixed grains in the entire batch. -/
def approximateMixedGrains (totalStones : ℕ) (sampleSize : ℕ) (mixedInSample : ℕ) : ℕ :=
  (totalStones * mixedInSample) / sampleSize

/-- Theorem stating that the approximate amount of mixed grains in the given scenario is 169 stones. -/
theorem mixed_grains_approximation :
  approximateMixedGrains 1534 254 28 = 169 := by
  sorry

#eval approximateMixedGrains 1534 254 28

end NUMINAMATH_CALUDE_mixed_grains_approximation_l3069_306993


namespace NUMINAMATH_CALUDE_both_selected_probability_l3069_306972

theorem both_selected_probability (p_ram p_ravi : ℚ) 
  (h_ram : p_ram = 5 / 7)
  (h_ravi : p_ravi = 1 / 5) :
  p_ram * p_ravi = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_both_selected_probability_l3069_306972


namespace NUMINAMATH_CALUDE_monotonic_decreasing_quadratic_l3069_306922

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem monotonic_decreasing_quadratic (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, x < y → f a x > f a y) →
  a ∈ Set.Ici 2 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_quadratic_l3069_306922


namespace NUMINAMATH_CALUDE_power_of_three_mod_eleven_l3069_306937

theorem power_of_three_mod_eleven : 3^2023 % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eleven_l3069_306937


namespace NUMINAMATH_CALUDE_solar_project_analysis_l3069_306994

/-- Represents the net profit of a solar power generation project over n years. -/
def net_profit (n : ℕ) : ℚ :=
  -4 * n^2 + 80 * n - 144

/-- Represents the average annual profit of the project over n years. -/
def avg_annual_profit (n : ℕ) : ℚ :=
  net_profit n / n

theorem solar_project_analysis :
  ∀ n : ℕ,
  (n > 0) →
  (net_profit n = -4 * n^2 + 80 * n - 144) ∧
  (net_profit 3 > 0) ∧
  (net_profit 2 ≤ 0) ∧
  (∀ k : ℕ, k > 0 → avg_annual_profit 6 ≥ avg_annual_profit k) :=
by sorry

end NUMINAMATH_CALUDE_solar_project_analysis_l3069_306994


namespace NUMINAMATH_CALUDE_ellipse_sum_theorem_l3069_306946

/-- Represents an ellipse with center (h, k) and semi-axes a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1

theorem ellipse_sum_theorem (e : Ellipse) 
    (h_center : e.h = -4 ∧ e.k = 2)
    (h_semi_major : e.a = 5)
    (h_semi_minor : e.b = 3) :
  e.h + e.k + e.a + e.b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_theorem_l3069_306946


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3069_306925

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 4) * (x^2 + 11*x + 30) + (x^2 + 8*x - 10) =
  (x^2 + 8*x + 7) * (x^2 + 8*x + 19) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3069_306925


namespace NUMINAMATH_CALUDE_triangle_toothpicks_count_l3069_306971

/-- The number of small triangles in the base of the large triangle -/
def base_triangles : ℕ := 101

/-- The total number of small triangles in the large triangle -/
def total_triangles (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of shared toothpicks in the structure -/
def shared_toothpicks (n : ℕ) : ℕ := 3 * total_triangles n / 2

/-- The number of boundary toothpicks -/
def boundary_toothpicks (n : ℕ) : ℕ := 3 * n

/-- The number of support toothpicks on the boundary -/
def support_toothpicks : ℕ := 3

/-- The total number of toothpicks required for the structure -/
def total_toothpicks (n : ℕ) : ℕ :=
  shared_toothpicks n + boundary_toothpicks n + support_toothpicks

theorem triangle_toothpicks_count :
  total_toothpicks base_triangles = 8032 :=
sorry

end NUMINAMATH_CALUDE_triangle_toothpicks_count_l3069_306971


namespace NUMINAMATH_CALUDE_no_points_in_circle_l3069_306913

theorem no_points_in_circle (r : ℝ) (A B : ℝ × ℝ) : r = 1 → 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 →
  ¬∃ P : ℝ × ℝ, (P.1 - A.1)^2 + (P.2 - A.2)^2 < r^2 ∧ 
                (P.1 - B.1)^2 + (P.2 - B.2)^2 < r^2 ∧
                (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.1 - B.1)^2 + (P.2 - B.2)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_no_points_in_circle_l3069_306913


namespace NUMINAMATH_CALUDE_marco_coins_l3069_306928

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  five_cent : ℕ
  ten_cent : ℕ
  twenty_cent : ℕ

/-- Calculates the number of different values obtainable from a given set of coins -/
def different_values (coins : CoinCounts) : ℕ :=
  59 - 3 * coins.five_cent - 2 * coins.ten_cent

theorem marco_coins :
  ∀ (coins : CoinCounts),
    coins.five_cent + coins.ten_cent + coins.twenty_cent = 15 →
    different_values coins = 28 →
    coins.twenty_cent = 4 := by
  sorry

end NUMINAMATH_CALUDE_marco_coins_l3069_306928


namespace NUMINAMATH_CALUDE_son_age_l3069_306976

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_son_age_l3069_306976


namespace NUMINAMATH_CALUDE_apple_juice_quantity_l3069_306936

/-- Given the total apple production and export percentage, calculate the quantity of apples used for juice -/
theorem apple_juice_quantity (total_production : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) : 
  total_production = 6 →
  export_percentage = 0.25 →
  juice_percentage = 0.60 →
  juice_percentage * (total_production * (1 - export_percentage)) = 2.7 := by
sorry

end NUMINAMATH_CALUDE_apple_juice_quantity_l3069_306936


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3069_306942

theorem arithmetic_calculation : (120 / 6 * 2 / 3 : ℚ) = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3069_306942


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l3069_306939

theorem isosceles_triangle_sides (p a : ℝ) (h1 : p = 14) (h2 : a = 4) :
  (∃ b c : ℝ, (a + b + c = p ∧ (b = c ∨ a = b ∨ a = c)) →
    ((b = 5 ∧ c = 5) ∨ (b = 4 ∧ c = 6) ∨ (b = 6 ∧ c = 4))) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l3069_306939


namespace NUMINAMATH_CALUDE_log_equation_solution_l3069_306911

/-- Given that log₃ₓ(343) = x and x is real, prove that x is a non-square, non-cube, non-integral rational number -/
theorem log_equation_solution (x : ℝ) (h : Real.log 343 / Real.log (3 * x) = x) :
  ∃ (a b : ℤ), x = (a : ℝ) / (b : ℝ) ∧ 
  b ≠ 0 ∧ 
  ¬ ∃ (n : ℤ), x = n ∧
  ¬ ∃ (n : ℝ), x = n ^ 2 ∧
  ¬ ∃ (n : ℝ), x = n ^ 3 :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3069_306911


namespace NUMINAMATH_CALUDE_saturday_practice_hours_l3069_306905

/-- Given a person's practice schedule, calculate the hours practiced on Saturdays -/
theorem saturday_practice_hours 
  (weekday_hours : ℕ) 
  (total_weeks : ℕ) 
  (total_practice_hours : ℕ) 
  (h1 : weekday_hours = 3)
  (h2 : total_weeks = 3)
  (h3 : total_practice_hours = 60) :
  (total_practice_hours - weekday_hours * 5 * total_weeks) / total_weeks = 5 := by
  sorry

#check saturday_practice_hours

end NUMINAMATH_CALUDE_saturday_practice_hours_l3069_306905


namespace NUMINAMATH_CALUDE_exists_perpendicular_intersection_line_l3069_306961

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus of C₂
def F : ℝ × ℝ := (1, 0)

-- Define a line passing through F with slope k
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the condition for two points being perpendicular from the origin
def perpendicular_from_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem exists_perpendicular_intersection_line :
  ∃ k : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    perpendicular_from_origin x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_intersection_line_l3069_306961


namespace NUMINAMATH_CALUDE_trig_calculation_l3069_306940

theorem trig_calculation : 
  (6 * (Real.tan (45 * π / 180))) - (2 * (Real.cos (60 * π / 180))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_calculation_l3069_306940


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l3069_306907

theorem triangle_perimeter_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  a = 1 →
  2 * Real.cos C + c = 2 * b →
  a = 2 * Real.sin (B / 2) * Real.sin (C / 2) / Real.sin ((B + C) / 2) →
  b = 2 * Real.sin (A / 2) * Real.sin (C / 2) / Real.sin ((A + C) / 2) →
  c = 2 * Real.sin (A / 2) * Real.sin (B / 2) / Real.sin ((A + B) / 2) →
  let p := a + b + c
  Real.sqrt 3 + 1 < p ∧ p < 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l3069_306907


namespace NUMINAMATH_CALUDE_all_nines_square_l3069_306916

/-- A function that generates a number with n 9's -/
def all_nines (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Theorem: For any positive integer n, (all_nines n)² = (all_nines n + 1)(all_nines n - 1) + 1 -/
theorem all_nines_square (n : ℕ+) :
  (all_nines n)^2 = (all_nines n + 1) * (all_nines n - 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_all_nines_square_l3069_306916


namespace NUMINAMATH_CALUDE_first_week_customers_l3069_306985

def commission_rate : ℚ := 1
def salary : ℚ := 500
def bonus : ℚ := 50
def total_earnings : ℚ := 760

def customers_first_week (C : ℚ) : Prop :=
  let commission := commission_rate * (C + 2*C + 3*C)
  total_earnings = salary + bonus + commission

theorem first_week_customers :
  ∃ C : ℚ, customers_first_week C ∧ C = 35 :=
sorry

end NUMINAMATH_CALUDE_first_week_customers_l3069_306985


namespace NUMINAMATH_CALUDE_octal_to_decimal_fraction_l3069_306953

theorem octal_to_decimal_fraction (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- c and d are base-10 digits
  (435 : Nat) = 4 * 8^2 + 3 * 8 + 5 →  -- 435 in octal
  285 = 200 + 10 * c + d →  -- 2cd in decimal
  (c + d) / 12 = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_fraction_l3069_306953


namespace NUMINAMATH_CALUDE_sticks_to_triangles_l3069_306962

/-- Represents a stick that can be cut into smaller pieces -/
structure Stick :=
  (length : ℕ)

/-- Represents a triangle with sides of specific lengths -/
structure Triangle :=
  (side1 : ℕ)
  (side2 : ℕ)
  (side3 : ℕ)

/-- The number of original sticks -/
def num_sticks : ℕ := 12

/-- The length of each original stick -/
def stick_length : ℕ := 13

/-- The number of triangles to be formed -/
def num_triangles : ℕ := 13

/-- The desired triangle side lengths -/
def target_triangle : Triangle :=
  { side1 := 3, side2 := 4, side3 := 5 }

/-- Theorem stating that the sticks can be cut to form the desired triangles -/
theorem sticks_to_triangles :
  ∃ (cut_pieces : List ℕ),
    (cut_pieces.sum = num_sticks * stick_length) ∧
    (∀ t : Triangle, t ∈ List.replicate num_triangles target_triangle →
      t.side1 ∈ cut_pieces ∧ t.side2 ∈ cut_pieces ∧ t.side3 ∈ cut_pieces) :=
sorry

end NUMINAMATH_CALUDE_sticks_to_triangles_l3069_306962


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l3069_306949

/-- Given Elaine's rent spending patterns over two years, prove that this year's rent
    is 187.5% of last year's rent. -/
theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.25 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l3069_306949


namespace NUMINAMATH_CALUDE_inequality_solution_l3069_306908

theorem inequality_solution (k : ℝ) : 
  (∀ x : ℝ, (k + 2) * x > k + 2 ↔ x < 1) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3069_306908


namespace NUMINAMATH_CALUDE_bookstore_sales_after_returns_l3069_306944

/-- Calculates the total sales after returns for a bookstore --/
theorem bookstore_sales_after_returns 
  (total_customers : ℕ) 
  (return_rate : ℚ) 
  (price_per_book : ℕ) : 
  total_customers = 1000 → 
  return_rate = 37 / 100 → 
  price_per_book = 15 → 
  (total_customers : ℚ) * (1 - return_rate) * (price_per_book : ℚ) = 9450 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_after_returns_l3069_306944


namespace NUMINAMATH_CALUDE_triangle_side_comparison_l3069_306987

theorem triangle_side_comparison (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (Real.sin A > Real.sin B) →
  (a > b) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_comparison_l3069_306987


namespace NUMINAMATH_CALUDE_mountain_trip_distance_l3069_306965

/-- Proves that the distance coming down the mountain is 8 km given the problem conditions --/
theorem mountain_trip_distance (d_up d_down : ℝ) : 
  (d_up / 3 + d_down / 4 = 4) →  -- Total time equation
  (d_down = d_up + 2) →          -- Difference in distances
  (d_down = 8) :=                -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_mountain_trip_distance_l3069_306965


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3069_306941

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a - 2 * i) * i = b - i → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3069_306941


namespace NUMINAMATH_CALUDE_unique_k_solution_l3069_306947

theorem unique_k_solution : ∃! k : ℝ, ∀ x : ℝ, 
  (3*x^2 - 4*x + 5) * (5*x^2 + k*x + 15) = 15*x^4 - 47*x^3 + 100*x^2 - 60*x + 75 :=
by sorry

end NUMINAMATH_CALUDE_unique_k_solution_l3069_306947


namespace NUMINAMATH_CALUDE_complement_probability_l3069_306956

theorem complement_probability (event_prob : ℚ) (h : event_prob = 1/4) :
  1 - event_prob = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_complement_probability_l3069_306956


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4872_l3069_306997

theorem largest_prime_factor_of_4872 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4872 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4872 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4872_l3069_306997


namespace NUMINAMATH_CALUDE_sin_cos_difference_zero_l3069_306926

theorem sin_cos_difference_zero : Real.sin (36 * π / 180) * Real.cos (36 * π / 180) - Real.cos (36 * π / 180) * Real.sin (36 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_zero_l3069_306926


namespace NUMINAMATH_CALUDE_ellipse_C_equation_l3069_306998

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 5 - y^2 / 4 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci and vertices of ellipse C
def ellipse_C_foci (x y : ℝ) : Prop := (x = Real.sqrt 5 ∧ y = 0) ∨ (x = -Real.sqrt 5 ∧ y = 0)
def ellipse_C_vertices (x y : ℝ) : Prop := (x = 3 ∧ y = 0) ∨ (x = -3 ∧ y = 0)

-- Theorem statement
theorem ellipse_C_equation :
  (∀ x y : ℝ, hyperbola x y → 
    ((x = Real.sqrt 5 ∧ y = 0) ∨ (x = -Real.sqrt 5 ∧ y = 0) → ellipse_C_vertices x y) ∧
    ((x = 3 ∧ y = 0) ∨ (x = -3 ∧ y = 0) → ellipse_C_foci x y)) →
  (∀ x y : ℝ, ellipse_C_foci x y ∨ ellipse_C_vertices x y → ellipse_C x y) :=
sorry

end NUMINAMATH_CALUDE_ellipse_C_equation_l3069_306998


namespace NUMINAMATH_CALUDE_polygon_diagonals_twice_sides_l3069_306990

theorem polygon_diagonals_twice_sides (n : ℕ) : 
  n ≥ 3 → (n * (n - 3) / 2 = 2 * n ↔ n = 7) := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_twice_sides_l3069_306990


namespace NUMINAMATH_CALUDE_salesman_visits_l3069_306919

def S : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | n + 3 => S (n + 2) + S (n + 1) + S n

theorem salesman_visits (n : ℕ) : S 12 = 927 := by
  sorry

end NUMINAMATH_CALUDE_salesman_visits_l3069_306919


namespace NUMINAMATH_CALUDE_prize_orders_count_l3069_306948

/-- Represents the number of bowlers in the tournament -/
def num_bowlers : ℕ := 6

/-- Represents the number of matches in the tournament -/
def num_matches : ℕ := num_bowlers - 1

/-- Calculates the number of possible outcomes for the tournament -/
def tournament_outcomes : ℕ := 2^num_matches

/-- Theorem stating that the number of different possible prize orders is 32 -/
theorem prize_orders_count : tournament_outcomes = 32 := by
  sorry

end NUMINAMATH_CALUDE_prize_orders_count_l3069_306948


namespace NUMINAMATH_CALUDE_translation_problem_l3069_306996

/-- A translation of the complex plane -/
def ComplexTranslation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

theorem translation_problem (t : ℂ → ℂ) (h : t (1 + 3*I) = 4 + 2*I) :
  ∃ w : ℂ, t = ComplexTranslation w ∧ t (3 - 2*I) = 6 - 3*I := by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l3069_306996


namespace NUMINAMATH_CALUDE_percentage_problem_l3069_306933

theorem percentage_problem (P : ℝ) : 
  (0.5 * 640 = P / 100 * 650 + 190) → P = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3069_306933


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3069_306909

theorem hyperbola_asymptote_slope (x y : ℝ) :
  (x^2 / 49 - y^2 / 36 = 4) →
  ∃ (m : ℝ), m > 0 ∧ (y = m * x ∨ y = -m * x) ∧ m = 6/7 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3069_306909


namespace NUMINAMATH_CALUDE_equal_numbers_exist_l3069_306924

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Condition for a quadratic polynomial -/
def IsQuadratic {α : Type*} [Field α] (f : QuadraticPolynomial α) :=
  ∃ a b c, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem equal_numbers_exist (f : QuadraticPolynomial ℝ) (l t v : ℝ)
    (hf : IsQuadratic f)
    (hl : f l = t + v)
    (ht : f t = l + v)
    (hv : f v = l + t) :
    l = t ∨ l = v ∨ t = v := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_exist_l3069_306924


namespace NUMINAMATH_CALUDE_ellipsoid_sum_center_axes_l3069_306973

/-- The equation of a tilted three-dimensional ellipsoid -/
def ellipsoid_equation (x y z x₀ y₀ z₀ A B C : ℝ) : Prop :=
  (x - x₀)^2 / A^2 + (y - y₀)^2 / B^2 + (z - z₀)^2 / C^2 = 1

/-- Theorem: Sum of center coordinates and semi-major axes lengths -/
theorem ellipsoid_sum_center_axes :
  ∀ (x₀ y₀ z₀ A B C : ℝ),
  ellipsoid_equation x y z x₀ y₀ z₀ A B C →
  x₀ = -2 →
  y₀ = 3 →
  z₀ = 1 →
  A = 6 →
  B = 4 →
  C = 2 →
  x₀ + y₀ + z₀ + A + B + C = 14 :=
by sorry

end NUMINAMATH_CALUDE_ellipsoid_sum_center_axes_l3069_306973


namespace NUMINAMATH_CALUDE_number_problem_l3069_306934

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 8 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3069_306934


namespace NUMINAMATH_CALUDE_eden_has_fourteen_bears_l3069_306910

/-- The number of stuffed bears Eden has after receiving her share from Daragh --/
def edens_final_bear_count (initial_bears : ℕ) (favorite_bears : ℕ) (sisters : ℕ) (edens_initial_bears : ℕ) : ℕ :=
  let remaining_bears := initial_bears - favorite_bears
  let bears_per_sister := remaining_bears / sisters
  edens_initial_bears + bears_per_sister

/-- Theorem stating that Eden will have 14 stuffed bears after receiving her share --/
theorem eden_has_fourteen_bears :
  edens_final_bear_count 20 8 3 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_eden_has_fourteen_bears_l3069_306910


namespace NUMINAMATH_CALUDE_solve_for_t_l3069_306918

theorem solve_for_t (s t : ℚ) 
  (eq1 : 12 * s + 7 * t = 165)
  (eq2 : s = t + 3) : 
  t = 129 / 19 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l3069_306918


namespace NUMINAMATH_CALUDE_max_value_at_2000_l3069_306920

/-- The function f(k) = k^2 / 1.001^k reaches its maximum value when k = 2000 --/
theorem max_value_at_2000 (k : ℕ) : 
  (k^2 : ℝ) / (1.001^k) ≤ (2000^2 : ℝ) / (1.001^2000) :=
sorry

end NUMINAMATH_CALUDE_max_value_at_2000_l3069_306920


namespace NUMINAMATH_CALUDE_larger_part_of_66_l3069_306914

theorem larger_part_of_66 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_sum : x + y = 66) (h_relation : 0.40 * x = 0.625 * y + 10) : 
  max x y = 50 := by
sorry

end NUMINAMATH_CALUDE_larger_part_of_66_l3069_306914


namespace NUMINAMATH_CALUDE_existence_of_m_l3069_306966

/-- Sum of digits function -/
def d (n : ℕ+) : ℕ :=
  sorry

/-- Main theorem -/
theorem existence_of_m (k : ℕ+) :
  ∃ m : ℕ+, ∃! (s : Finset ℕ+), s.card = k ∧ ∀ x ∈ s, x + d x = m :=
sorry

end NUMINAMATH_CALUDE_existence_of_m_l3069_306966


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3069_306984

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I)^2 = Complex.abs (1 + Complex.I)^2 → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3069_306984


namespace NUMINAMATH_CALUDE_mildred_blocks_l3069_306980

theorem mildred_blocks (initial_blocks found_blocks : ℕ) : 
  initial_blocks = 2 → found_blocks = 84 → initial_blocks + found_blocks = 86 := by
  sorry

end NUMINAMATH_CALUDE_mildred_blocks_l3069_306980


namespace NUMINAMATH_CALUDE_max_snacks_l3069_306932

theorem max_snacks (S : ℕ) : 
  (∀ n : ℕ, n ≤ S → n > 6 * 18 ∧ n < 7 * 18) → 
  S = 125 := by
  sorry

end NUMINAMATH_CALUDE_max_snacks_l3069_306932


namespace NUMINAMATH_CALUDE_inscribed_sphere_cone_relation_l3069_306927

/-- A right cone with a sphere inscribed in it -/
structure InscribedSphereCone where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  b : ℝ
  d : ℝ
  sphere_radius_eq : sphere_radius = b * (Real.sqrt d - 1)

/-- The theorem stating the relationship between b and d for the given cone and sphere -/
theorem inscribed_sphere_cone_relation (cone : InscribedSphereCone) 
  (h1 : cone.base_radius = 15)
  (h2 : cone.height = 30) :
  cone.b + cone.d = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_cone_relation_l3069_306927


namespace NUMINAMATH_CALUDE_tan_315_degrees_l3069_306992

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l3069_306992


namespace NUMINAMATH_CALUDE_marble_division_l3069_306951

theorem marble_division (x : ℝ) : 
  (5*x + 2) + (2*x - 1) + (x + 4) = 35 → 
  ∃ (a b c : ℕ), a + b + c = 35 ∧ 
    (a : ℝ) = 5*x + 2 ∧ 
    (b : ℝ) = 2*x - 1 ∧ 
    (c : ℝ) = x + 4 := by
  sorry

end NUMINAMATH_CALUDE_marble_division_l3069_306951


namespace NUMINAMATH_CALUDE_expression_simplification_l3069_306955

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) : 
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3069_306955


namespace NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l3069_306983

theorem square_of_negative_sqrt_two : (-Real.sqrt 2)^2 = 2 := by sorry

end NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l3069_306983


namespace NUMINAMATH_CALUDE_lisa_equal_earnings_l3069_306986

/-- Given Greta's work hours, Greta's hourly rate, and Lisa's hourly rate,
    calculates the number of hours Lisa needs to work to equal Greta's earnings. -/
def lisa_work_hours (greta_hours : ℕ) (greta_rate : ℚ) (lisa_rate : ℚ) : ℚ :=
  (greta_hours : ℚ) * greta_rate / lisa_rate

/-- Proves that Lisa needs to work 32 hours to equal Greta's earnings,
    given the specified conditions. -/
theorem lisa_equal_earnings : lisa_work_hours 40 12 15 = 32 := by
  sorry

end NUMINAMATH_CALUDE_lisa_equal_earnings_l3069_306986


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_2010_l3069_306981

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_2010 :
  sum_factorials 2010 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_2010_l3069_306981


namespace NUMINAMATH_CALUDE_total_age_proof_l3069_306938

/-- Given three people a, b, and c, where a is two years older than b, b is twice as old as c, 
    and b is 10 years old, prove that the total of their ages is 27 years. -/
theorem total_age_proof (a b c : ℕ) : 
  b = 10 → a = b + 2 → b = 2 * c → a + b + c = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_age_proof_l3069_306938


namespace NUMINAMATH_CALUDE_complex_sum_to_polar_l3069_306967

theorem complex_sum_to_polar : 
  ∃ (r θ : ℝ), 5 * (Complex.exp (3 * Real.pi * Complex.I / 4) + Complex.exp (-3 * Real.pi * Complex.I / 4)) = r * Complex.exp (θ * Complex.I) ∧ 
  r = -5 * Real.sqrt 2 ∧ 
  θ = Real.pi := by
sorry

end NUMINAMATH_CALUDE_complex_sum_to_polar_l3069_306967


namespace NUMINAMATH_CALUDE_vikki_tax_percentage_l3069_306902

/-- Calculates the tax percentage given the working conditions and take-home pay --/
def calculate_tax_percentage (hours_worked : ℕ) (hourly_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) (take_home_pay : ℚ) : ℚ :=
  let gross_earnings := hours_worked * hourly_rate
  let insurance_deduction := insurance_rate * gross_earnings
  let total_deductions := gross_earnings - take_home_pay
  let tax_deduction := total_deductions - insurance_deduction - union_dues
  (tax_deduction / gross_earnings) * 100

/-- Theorem stating that the tax percentage is 20% given Vikki's working conditions --/
theorem vikki_tax_percentage :
  calculate_tax_percentage 42 10 (5/100) 5 310 = 20 := by
  sorry

end NUMINAMATH_CALUDE_vikki_tax_percentage_l3069_306902


namespace NUMINAMATH_CALUDE_divisor_problem_l3069_306957

theorem divisor_problem (original : ℕ) (added : ℕ) (divisor : ℕ) : 
  original = 821562 →
  added = 6 →
  (original + added) % divisor = 0 →
  ∀ d : ℕ, d < added → (original + d) % divisor ≠ 0 →
  divisor = 6 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l3069_306957


namespace NUMINAMATH_CALUDE_congruence_implication_l3069_306943

theorem congruence_implication (a b c d n : ℤ) 
  (h1 : a * c ≡ 0 [ZMOD n])
  (h2 : b * c + a * d ≡ 0 [ZMOD n]) :
  b * c ≡ 0 [ZMOD n] ∧ a * d ≡ 0 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_congruence_implication_l3069_306943


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l3069_306901

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements in the arrangement -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when randomly arranged with three ones in a row -/
def prob_zeros_not_adjacent : ℚ := 3/5

theorem zeros_not_adjacent_probability :
  prob_zeros_not_adjacent = 3/5 :=
sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l3069_306901


namespace NUMINAMATH_CALUDE_job_completion_time_l3069_306988

theorem job_completion_time (total_work : ℝ) (time_together time_person2 : ℝ) 
  (h1 : time_together > 0)
  (h2 : time_person2 > 0)
  (h3 : total_work > 0)
  (h4 : total_work / time_together = total_work / time_person2 + total_work / (24 : ℝ)) :
  total_work / (total_work / time_together - total_work / time_person2) = 24 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l3069_306988


namespace NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l3069_306974

theorem quadratic_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - (a + 3) * x + 2 = 0 ∧ 
               a * y^2 - (a + 3) * y + 2 = 0 ∧ 
               x * y < 0) ↔ 
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l3069_306974


namespace NUMINAMATH_CALUDE_max_value_theorem_l3069_306963

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x + y^2 + z^2 = 1) : 
  x^2 + y^3 + z^4 ≤ 1 ∧ ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 
  a + b^2 + c^2 = 1 ∧ a^2 + b^3 + c^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3069_306963


namespace NUMINAMATH_CALUDE_sum_of_fractions_value_of_m_l3069_306915

noncomputable section

variable (θ : Real)
variable (m : Real)

-- Define the equation and its roots
def equation (x : Real) := 2 * x^2 - (Real.sqrt 3 + 1) * x + m

-- Conditions
axiom theta_range : 0 < θ ∧ θ < 2 * Real.pi
axiom roots : equation (Real.sin θ) = 0 ∧ equation (Real.cos θ) = 0

-- Theorems to prove
theorem sum_of_fractions :
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + (Real.cos θ)^2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2 :=
sorry

theorem value_of_m : m = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_value_of_m_l3069_306915


namespace NUMINAMATH_CALUDE_pencil_count_l3069_306931

theorem pencil_count (num_students : ℕ) (pencils_per_student : ℕ) 
  (h1 : num_students = 2) 
  (h2 : pencils_per_student = 9) : 
  num_students * pencils_per_student = 18 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3069_306931


namespace NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l3069_306977

theorem smallest_k_for_no_real_roots :
  ∃ (k : ℤ), k = 3 ∧
  (∀ (x : ℝ), 3 * x * (k * x - 5) - 2 * x^2 + 8 ≠ 0) ∧
  (∀ (k' : ℤ), k' < k →
    ∃ (x : ℝ), 3 * x * (k' * x - 5) - 2 * x^2 + 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l3069_306977


namespace NUMINAMATH_CALUDE_problem_1_l3069_306935

theorem problem_1 : 3.14 * 5.5^2 - 3.14 * 4.5^2 = 31.4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3069_306935


namespace NUMINAMATH_CALUDE_at_least_n_minus_two_have_real_root_l3069_306999

/-- A linear function of the form ax + b where a ≠ 0 -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0

/-- The product of all LinearFunctions except the i-th one -/
def productExcept (funcs : List LinearFunction) (i : Nat) : LinearFunction → LinearFunction :=
  sorry

/-- The polynomial formed by the sum of the product of n-1 functions and the remaining function -/
def formPolynomial (funcs : List LinearFunction) (i : Nat) : LinearFunction :=
  sorry

/-- A function has a real root if there exists a real number x such that f(x) = 0 -/
def hasRealRoot (f : LinearFunction) : Prop :=
  ∃ x : ℝ, f.a * x + f.b = 0

/-- The main theorem -/
theorem at_least_n_minus_two_have_real_root (funcs : List LinearFunction) :
  funcs.length ≥ 3 →
  ∃ (roots : List LinearFunction),
    roots.length ≥ funcs.length - 2 ∧
    ∀ f ∈ roots, ∃ i, f = formPolynomial funcs i ∧ hasRealRoot f :=
  sorry

end NUMINAMATH_CALUDE_at_least_n_minus_two_have_real_root_l3069_306999


namespace NUMINAMATH_CALUDE_equation_solution_l3069_306989

theorem equation_solution :
  ∀ x : ℚ, (40 : ℚ) / 60 = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3069_306989


namespace NUMINAMATH_CALUDE_quadratic_solution_l3069_306995

theorem quadratic_solution (x : ℝ) (h1 : 2 * x^2 - 6 * x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3069_306995


namespace NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_y2_l3069_306950

theorem min_value_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.5) : 
  x * y / (x^2 + y^2) ≥ 0.4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_y2_l3069_306950


namespace NUMINAMATH_CALUDE_polynomial_form_l3069_306964

/-- A polynomial satisfying the given functional equation -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (x + 1) * P (x - 1) + (x - 1) * P (x + 1) = 2 * x * P x

/-- The theorem stating the form of polynomials satisfying the functional equation -/
theorem polynomial_form (P : ℝ → ℝ) (hP : SatisfyingPolynomial P) :
  ∃ a d : ℝ, ∀ x : ℝ, P x = a * x^3 - a * x + d :=
sorry

end NUMINAMATH_CALUDE_polynomial_form_l3069_306964


namespace NUMINAMATH_CALUDE_rectangle_ratio_l3069_306979

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0) : 
  (s + 2*y = 3*s) → (x + s = 3*s) → ((3*s)^2 = 9*s^2) → (x/y = 2) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l3069_306979


namespace NUMINAMATH_CALUDE_circle_area_tripled_radius_l3069_306917

theorem circle_area_tripled_radius (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := π * (3*r)^2
  A' = 9 * A :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_radius_l3069_306917


namespace NUMINAMATH_CALUDE_linear_inequality_solution_set_l3069_306921

theorem linear_inequality_solution_set 
  (m n : ℝ) 
  (h_m : m = -1) 
  (h_n : n = -1) : 
  {x : ℝ | m * x - n ≤ 2} = {x : ℝ | x ≥ -1} := by
sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_set_l3069_306921


namespace NUMINAMATH_CALUDE_dinos_money_theorem_l3069_306991

/-- Calculates Dino's remaining money after expenses given his work hours and rates. -/
def dinos_remaining_money (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (expenses : ℕ) : ℕ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - expenses

/-- Theorem stating that Dino's remaining money at the end of the month is $500. -/
theorem dinos_money_theorem : 
  dinos_remaining_money 20 30 5 10 20 40 500 = 500 := by
  sorry

#eval dinos_remaining_money 20 30 5 10 20 40 500

end NUMINAMATH_CALUDE_dinos_money_theorem_l3069_306991


namespace NUMINAMATH_CALUDE_earthquake_energy_ratio_l3069_306912

-- Define the Richter scale energy relation
def richter_energy_ratio (x : ℝ) : ℝ := 10

-- Define the frequency function type
def frequency := ℝ → ℝ

-- Theorem statement
theorem earthquake_energy_ratio 
  (f : frequency) 
  (x y : ℝ) 
  (h1 : y - x = 2) 
  (h2 : f y = 2 * f x) :
  (richter_energy_ratio ^ y) / (richter_energy_ratio ^ x) = 200 := by
  sorry

end NUMINAMATH_CALUDE_earthquake_energy_ratio_l3069_306912


namespace NUMINAMATH_CALUDE_sqrt_81_equals_3_squared_l3069_306969

theorem sqrt_81_equals_3_squared : Real.sqrt 81 = 3^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_equals_3_squared_l3069_306969


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_and_perimeter_l3069_306975

/-- Given a square with diagonal 2x and perimeter 16x, prove its area is 16x² -/
theorem square_area_from_diagonal_and_perimeter (x : ℝ) :
  let diagonal := 2 * x
  let perimeter := 16 * x
  let side := perimeter / 4
  let area := side ^ 2
  diagonal ^ 2 = 2 * side ^ 2 ∧ perimeter = 4 * side → area = 16 * x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_and_perimeter_l3069_306975


namespace NUMINAMATH_CALUDE_comparison_sqrt_l3069_306954

theorem comparison_sqrt : 3 * Real.sqrt 2 > Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_comparison_sqrt_l3069_306954


namespace NUMINAMATH_CALUDE_bucket_capacity_l3069_306900

/-- Represents the number of buckets needed to fill the bathtub to the top -/
def full_bathtub : ℕ := 14

/-- Represents the number of buckets removed to reach the bath level -/
def removed_buckets : ℕ := 3

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Represents the total amount of water used in a week (in ounces) -/
def weekly_water_usage : ℕ := 9240

/-- Calculates the number of buckets used for each bath -/
def buckets_per_bath : ℕ := full_bathtub - removed_buckets

/-- Calculates the number of buckets used in a week -/
def weekly_buckets : ℕ := buckets_per_bath * days_per_week

/-- Theorem: The bucket holds 120 ounces of water -/
theorem bucket_capacity : weekly_water_usage / weekly_buckets = 120 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_l3069_306900


namespace NUMINAMATH_CALUDE_solve_equations_l3069_306906

theorem solve_equations :
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, 3 * x * (x - 1) = 2 * (x - 1) ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 1 ∧ x₂ = 2/3) ∧
  (∃ y₁ y₂ : ℝ, (∀ x : ℝ, x^2 - 6*x + 6 = 0 ↔ x = y₁ ∨ x = y₂) ∧ y₁ = 3 + Real.sqrt 3 ∧ y₂ = 3 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l3069_306906


namespace NUMINAMATH_CALUDE_circle_radius_decrease_l3069_306978

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := 0.25 * A
  let r' := Real.sqrt (A' / π)
  (r - r') / r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_decrease_l3069_306978


namespace NUMINAMATH_CALUDE_arcsin_b_range_l3069_306960

open Real Set

theorem arcsin_b_range (a b : ℝ) :
  (arcsin a = arccos b) →
  (∀ (x y : ℝ), a * x^2 + b * y^2 = 1 → x^2 + y^2 ≥ 2 / Real.sqrt 3) →
  Icc (π / 6) (π / 3) \ {π / 4} ⊆ {θ | ∃ b, arcsin b = θ} :=
by sorry

end NUMINAMATH_CALUDE_arcsin_b_range_l3069_306960


namespace NUMINAMATH_CALUDE_a_range_l3069_306970

/-- The piecewise function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x - a

/-- Theorem stating that if f(0) is the minimum value of f(x), then a is in [0,1] --/
theorem a_range (a : ℝ) : 
  (∀ x : ℝ, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3069_306970


namespace NUMINAMATH_CALUDE_concave_number_probability_l3069_306930

/-- A five-digit natural number formed by digits 0, 1, 2, 3, and 4 -/
def FiveDigitNumber := Fin 5 → Fin 5

/-- Predicate for a "concave number" -/
def IsConcave (n : FiveDigitNumber) : Prop :=
  n 0 > n 1 ∧ n 1 > n 2 ∧ n 2 < n 3 ∧ n 3 < n 4

/-- The set of all possible five-digit numbers -/
def AllNumbers : Finset FiveDigitNumber := sorry

/-- The set of all concave numbers -/
def ConcaveNumbers : Finset FiveDigitNumber := sorry

theorem concave_number_probability :
  (Finset.card ConcaveNumbers : ℚ) / (Finset.card AllNumbers : ℚ) = 23 / 1250 := by sorry

end NUMINAMATH_CALUDE_concave_number_probability_l3069_306930


namespace NUMINAMATH_CALUDE_point_condition_y_intercept_condition_l3069_306958

/-- The equation of the line -/
def line_equation (x y t : ℝ) : Prop :=
  2 * x + (t - 2) * y + 3 - 2 * t = 0

/-- Theorem: If the line passes through (1, 1), then t = 5 -/
theorem point_condition (t : ℝ) : line_equation 1 1 t → t = 5 := by
  sorry

/-- Theorem: If the y-intercept of the line is -3, then t = 9/5 -/
theorem y_intercept_condition (t : ℝ) : line_equation 0 (-3) t → t = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_point_condition_y_intercept_condition_l3069_306958


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_b_l3069_306968

theorem min_value_of_3a_plus_b (a b : ℝ) (h : 16 * a^2 + 2 * a + 8 * a * b + b^2 - 1 = 0) :
  ∃ (m : ℝ), m = 3 * a + b ∧ m ≥ -1 ∧ ∀ (x : ℝ), (∃ (a' b' : ℝ), x = 3 * a' + b' ∧ 16 * a'^2 + 2 * a' + 8 * a' * b' + b'^2 - 1 = 0) → x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_b_l3069_306968


namespace NUMINAMATH_CALUDE_product_of_fractions_l3069_306945

theorem product_of_fractions :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3069_306945


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3069_306952

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x + 1/2 > 0) ↔ (0 ≤ m ∧ m < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3069_306952


namespace NUMINAMATH_CALUDE_AMC10_paths_count_l3069_306982

/-- Represents the number of paths to spell "AMC10" given specific adjacency conditions -/
def number_of_AMC10_paths (
  adjacent_Ms : Nat
  ) (adjacent_Cs : Nat)
  (adjacent_1s : Nat)
  (adjacent_0s : Nat) : Nat :=
  adjacent_Ms * adjacent_Cs * adjacent_1s * adjacent_0s

/-- Theorem stating that the number of paths to spell "AMC10" is 48 -/
theorem AMC10_paths_count :
  number_of_AMC10_paths 4 3 2 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_AMC10_paths_count_l3069_306982


namespace NUMINAMATH_CALUDE_inequality_proof_l3069_306929

theorem inequality_proof (a b c : ℝ) 
  (h1 : 4 * a * c - b^2 ≥ 0) 
  (h2 : a > 0) : 
  a + c - Real.sqrt ((a - c)^2 + b^2) ≤ (4 * a * c - b^2) / (2 * a) ∧ 
  (a + c - Real.sqrt ((a - c)^2 + b^2) = (4 * a * c - b^2) / (2 * a) ↔ 
    (b = 0 ∧ a ≥ c) ∨ 4 * a * c = b^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3069_306929


namespace NUMINAMATH_CALUDE_fruit_basket_solution_l3069_306959

def fruit_basket_problem (initial_apples initial_oranges x : ℕ) : Prop :=
  -- Initial condition: oranges are twice the apples
  initial_oranges = 2 * initial_apples ∧
  -- After x removals, 1 apple and 16 oranges remain
  initial_apples - 3 * x = 1 ∧
  initial_oranges - 4 * x = 16

theorem fruit_basket_solution : 
  ∃ initial_apples initial_oranges : ℕ, fruit_basket_problem initial_apples initial_oranges 7 :=
sorry

end NUMINAMATH_CALUDE_fruit_basket_solution_l3069_306959


namespace NUMINAMATH_CALUDE_jack_driving_years_l3069_306903

/-- Represents the number of miles Jack drives in four months -/
def miles_per_four_months : ℕ := 37000

/-- Represents the total number of miles Jack has driven -/
def total_miles_driven : ℕ := 999000

/-- Calculates the number of years Jack has been driving -/
def years_driving : ℚ :=
  total_miles_driven / (miles_per_four_months * 3)

/-- Theorem stating that Jack has been driving for 9 years -/
theorem jack_driving_years :
  years_driving = 9 := by sorry

end NUMINAMATH_CALUDE_jack_driving_years_l3069_306903


namespace NUMINAMATH_CALUDE_leahs_coins_value_l3069_306923

/-- Represents the value of a coin in cents -/
inductive Coin
| Penny : Coin
| Nickel : Coin

/-- The value of a coin in cents -/
def coin_value : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5

/-- A collection of coins -/
structure CoinCollection :=
  (pennies : Nat)
  (nickels : Nat)

/-- The total number of coins in a collection -/
def total_coins (c : CoinCollection) : Nat :=
  c.pennies + c.nickels

/-- The total value of coins in a collection in cents -/
def total_value (c : CoinCollection) : Nat :=
  c.pennies * coin_value Coin.Penny + c.nickels * coin_value Coin.Nickel

/-- The main theorem -/
theorem leahs_coins_value (c : CoinCollection) :
  total_coins c = 15 ∧
  c.pennies = c.nickels + 2 →
  total_value c = 44 := by
  sorry


end NUMINAMATH_CALUDE_leahs_coins_value_l3069_306923
