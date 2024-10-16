import Mathlib

namespace NUMINAMATH_CALUDE_raft_travel_time_l1680_168051

/-- The number of days it takes for a ship to travel from Chongqing to Shanghai -/
def ship_cq_to_sh : ℝ := 5

/-- The number of days it takes for a ship to travel from Shanghai to Chongqing -/
def ship_sh_to_cq : ℝ := 7

/-- The number of days it takes for a raft to drift from Chongqing to Shanghai -/
def raft_cq_to_sh : ℝ := 35

/-- Theorem stating that the raft travel time satisfies the given conditions -/
theorem raft_travel_time :
  1 / ship_cq_to_sh - 1 / raft_cq_to_sh = 1 / ship_sh_to_cq + 1 / raft_cq_to_sh :=
by sorry

end NUMINAMATH_CALUDE_raft_travel_time_l1680_168051


namespace NUMINAMATH_CALUDE_no_rational_roots_l1680_168079

def polynomial (x : ℚ) : ℚ := 3 * x^4 - 4 * x^3 - 9 * x^2 + 10 * x + 5

theorem no_rational_roots :
  ∀ q : ℚ, polynomial q ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_rational_roots_l1680_168079


namespace NUMINAMATH_CALUDE_satisfaction_ratings_properties_l1680_168053

def satisfaction_ratings : List ℝ := [5, 7, 8, 9, 7, 5, 10, 8, 4, 7]

def mode (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem satisfaction_ratings_properties :
  mode satisfaction_ratings = 7 ∧
  range satisfaction_ratings = 6 ∧
  variance satisfaction_ratings = 3.2 :=
by sorry

end NUMINAMATH_CALUDE_satisfaction_ratings_properties_l1680_168053


namespace NUMINAMATH_CALUDE_list_property_l1680_168088

theorem list_property (S : ℝ) (n : ℝ) :
  let list_size : ℕ := 21
  let other_numbers_sum : ℝ := S - n
  let other_numbers_count : ℕ := list_size - 1
  let other_numbers_avg : ℝ := other_numbers_sum / other_numbers_count
  n = 4 * other_numbers_avg →
  n = S / 6 →
  other_numbers_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_list_property_l1680_168088


namespace NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l1680_168058

theorem sum_of_four_primes_divisible_by_60 (p q r s : ℕ) 
  (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (h_order : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) :
  ∃ k : ℕ, p + q + r + s = 60 * (2 * k + 1) :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l1680_168058


namespace NUMINAMATH_CALUDE_range_of_k_l1680_168087

/-- An odd function that is strictly decreasing on [0, +∞) -/
def OddDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f y < f x)

theorem range_of_k (f : ℝ → ℝ) (h_odd_dec : OddDecreasingFunction f) :
  (∀ k x : ℝ, f (k * x^2 + 2) + f (k * x + k) ≤ 0) ↔ 
  (∀ k : ℝ, 0 ≤ k) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l1680_168087


namespace NUMINAMATH_CALUDE_gcd_factorial_8_9_l1680_168036

theorem gcd_factorial_8_9 : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_8_9_l1680_168036


namespace NUMINAMATH_CALUDE_max_principals_is_four_l1680_168038

/-- Represents the duration of the period in years -/
def period_duration : ℕ := 15

/-- Represents the duration of each principal's term in years -/
def term_duration : ℕ := 4

/-- Calculates the maximum number of principals that can serve during the given period -/
def max_principals : ℕ := (period_duration - 1) / term_duration + 1

/-- Theorem stating that the maximum number of principals is 4 -/
theorem max_principals_is_four : max_principals = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_principals_is_four_l1680_168038


namespace NUMINAMATH_CALUDE_vasily_salary_higher_l1680_168073

/-- Represents the salary distribution for graduates --/
structure GraduateSalary where
  high : ℝ  -- Salary for 1/5 of graduates
  very_high : ℝ  -- Salary for 1/10 of graduates
  low : ℝ  -- Salary for 1/20 of graduates
  medium : ℝ  -- Salary for remaining graduates

/-- Calculates the expected salary for a student --/
def expected_salary (
  total_students : ℕ
  ) (graduating_students : ℕ
  ) (non_graduate_salary : ℝ
  ) (graduate_salary : GraduateSalary
  ) : ℝ :=
  sorry

/-- Calculates the salary after a number of years with annual increase --/
def salary_after_years (
  initial_salary : ℝ
  ) (annual_increase : ℝ
  ) (years : ℕ
  ) : ℝ :=
  sorry

theorem vasily_salary_higher (
  total_students : ℕ
  ) (graduating_students : ℕ
  ) (non_graduate_salary : ℝ
  ) (graduate_salary : GraduateSalary
  ) (fyodor_initial_salary : ℝ
  ) (fyodor_annual_increase : ℝ
  ) (years : ℕ
  ) : 
  total_students = 300 →
  graduating_students = 270 →
  non_graduate_salary = 25000 →
  graduate_salary.high = 60000 →
  graduate_salary.very_high = 80000 →
  graduate_salary.low = 25000 →
  graduate_salary.medium = 40000 →
  fyodor_initial_salary = 25000 →
  fyodor_annual_increase = 3000 →
  years = 4 →
  expected_salary total_students graduating_students non_graduate_salary graduate_salary = 39625 ∧
  expected_salary total_students graduating_students non_graduate_salary graduate_salary - 
    salary_after_years fyodor_initial_salary fyodor_annual_increase years = 2625 :=
by sorry

end NUMINAMATH_CALUDE_vasily_salary_higher_l1680_168073


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1680_168089

theorem simplify_sqrt_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1680_168089


namespace NUMINAMATH_CALUDE_ellipse_parameters_and_eccentricity_l1680_168014

/-- Given an ellipse and a line passing through its vertex and focus, prove the ellipse's parameters and eccentricity. -/
theorem ellipse_parameters_and_eccentricity 
  (a b : ℝ) 
  (h_pos : a > b ∧ b > 0) 
  (h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x - 2*y + 2 = 0 → (x = 0 ∧ y = 1) ∨ (x = -2 ∧ y = 0))) :
  a^2 = 5 ∧ b^2 = 1 ∧ (a^2 - b^2) / a^2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parameters_and_eccentricity_l1680_168014


namespace NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_abs_rational_l1680_168041

theorem smallest_positive_largest_negative_smallest_abs_rational :
  ∃ (x y : ℤ) (z : ℚ),
    (∀ n : ℤ, n > 0 → x ≤ n) ∧
    (∀ n : ℤ, n < 0 → y ≥ n) ∧
    (∀ q : ℚ, |z| ≤ |q|) ∧
    2 * x + 3 * y + 4 * z = -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_abs_rational_l1680_168041


namespace NUMINAMATH_CALUDE_initial_workers_count_l1680_168027

/-- Represents the construction project scenario -/
structure ConstructionProject where
  initial_duration : ℕ
  actual_duration : ℕ
  initial_workers : ℕ
  double_rate_workers : ℕ
  triple_rate_workers : ℕ
  double_rate_join_day : ℕ
  triple_rate_join_day : ℕ

/-- Theorem stating that the initial number of workers is 55 -/
theorem initial_workers_count (project : ConstructionProject) 
  (h1 : project.initial_duration = 24)
  (h2 : project.actual_duration = 19)
  (h3 : project.double_rate_workers = 8)
  (h4 : project.triple_rate_workers = 5)
  (h5 : project.double_rate_join_day = 11)
  (h6 : project.triple_rate_join_day = 17) :
  project.initial_workers = 55 := by
  sorry

end NUMINAMATH_CALUDE_initial_workers_count_l1680_168027


namespace NUMINAMATH_CALUDE_ages_of_linda_and_jane_l1680_168011

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem ages_of_linda_and_jane (jane_age linda_age kevin_age : ℕ) :
  linda_age = 2 * jane_age + 3 →
  is_prime (linda_age - jane_age) →
  linda_age + jane_age + 10 = kevin_age + 5 →
  kevin_age = 4 * jane_age →
  jane_age = 8 ∧ linda_age = 19 :=
by sorry

end NUMINAMATH_CALUDE_ages_of_linda_and_jane_l1680_168011


namespace NUMINAMATH_CALUDE_third_line_product_l1680_168029

/-- Given two positive real numbers a and b, prove that 
    x = -a/2 + √(a²/4 + b²) satisfies x(x + a) = b² -/
theorem third_line_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let x := -a/2 + Real.sqrt (a^2/4 + b^2)
  x * (x + a) = b^2 := by
  sorry

end NUMINAMATH_CALUDE_third_line_product_l1680_168029


namespace NUMINAMATH_CALUDE_exists_n_satisfying_divisor_inequality_l1680_168078

/-- The number of positive divisors of a natural number -/
def d (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a natural number n satisfying the given condition -/
theorem exists_n_satisfying_divisor_inequality :
  ∃ n : ℕ, ∀ i : ℕ, i ≤ 1402 →
    (d n : ℚ) / d (n + i) > 1401 ∧ (d n : ℚ) / d (n - i) > 1401 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_satisfying_divisor_inequality_l1680_168078


namespace NUMINAMATH_CALUDE_dales_peppers_theorem_l1680_168005

/-- The amount of green peppers bought by Dale's Vegetarian Restaurant in pounds -/
def green_peppers : ℝ := 2.8333333333333335

/-- The amount of red peppers bought by Dale's Vegetarian Restaurant in pounds -/
def red_peppers : ℝ := 2.8333333333333335

/-- The total amount of peppers bought by Dale's Vegetarian Restaurant in pounds -/
def total_peppers : ℝ := green_peppers + red_peppers

theorem dales_peppers_theorem : total_peppers = 5.666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_dales_peppers_theorem_l1680_168005


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l1680_168015

theorem at_least_one_real_root (c : ℝ) : 
  ∃ x : ℝ, (x^2 + c*x + 2 = 0) ∨ (x^2 + 2*x + c = 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l1680_168015


namespace NUMINAMATH_CALUDE_bamboo_pole_is_ten_feet_l1680_168024

/-- The length of a bamboo pole satisfying specific conditions relative to a door --/
def bamboo_pole_length : ℝ → Prop := fun x =>
  ∃ (door_width door_height : ℝ),
    door_width > 0 ∧ 
    door_height > 0 ∧ 
    x = door_width + 4 ∧ 
    x = door_height + 2 ∧ 
    x^2 = door_width^2 + door_height^2

/-- Theorem stating that the bamboo pole length is 10 feet --/
theorem bamboo_pole_is_ten_feet : 
  bamboo_pole_length 10 := by
  sorry

#check bamboo_pole_is_ten_feet

end NUMINAMATH_CALUDE_bamboo_pole_is_ten_feet_l1680_168024


namespace NUMINAMATH_CALUDE_min_tiles_for_region_l1680_168057

/-- The number of tiles needed to cover a rectangular region -/
def tiles_needed (tile_length : ℕ) (tile_width : ℕ) (region_length : ℕ) (region_width : ℕ) : ℕ :=
  let region_area := region_length * region_width
  let tile_area := tile_length * tile_width
  (region_area + tile_area - 1) / tile_area

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

/-- Theorem stating the minimum number of tiles needed to cover the given region -/
theorem min_tiles_for_region : 
  tiles_needed 5 6 (3 * feet_to_inches) (4 * feet_to_inches) = 58 := by
  sorry

#eval tiles_needed 5 6 (3 * feet_to_inches) (4 * feet_to_inches)

end NUMINAMATH_CALUDE_min_tiles_for_region_l1680_168057


namespace NUMINAMATH_CALUDE_resulting_polygon_has_16_sides_l1680_168023

/-- Represents a regular polygon --/
structure RegularPolygon where
  sides : ℕ
  sides_positive : sides > 0

/-- The resulting polygon formed by connecting the given regular polygons --/
def resulting_polygon (triangle square pentagon heptagon hexagon octagon : RegularPolygon) : ℕ :=
  2 + 2 + (4 * 3)

/-- Theorem stating that the resulting polygon has 16 sides --/
theorem resulting_polygon_has_16_sides 
  (triangle : RegularPolygon) 
  (square : RegularPolygon)
  (pentagon : RegularPolygon)
  (heptagon : RegularPolygon)
  (hexagon : RegularPolygon)
  (octagon : RegularPolygon)
  (h1 : triangle.sides = 3)
  (h2 : square.sides = 4)
  (h3 : pentagon.sides = 5)
  (h4 : heptagon.sides = 7)
  (h5 : hexagon.sides = 6)
  (h6 : octagon.sides = 8) :
  resulting_polygon triangle square pentagon heptagon hexagon octagon = 16 := by
  sorry

end NUMINAMATH_CALUDE_resulting_polygon_has_16_sides_l1680_168023


namespace NUMINAMATH_CALUDE_total_amount_earned_l1680_168084

/-- The total amount earned from selling rackets given the average price per pair and the number of pairs sold. -/
theorem total_amount_earned (avg_price : ℝ) (num_pairs : ℕ) : avg_price = 9.8 → num_pairs = 50 → avg_price * (num_pairs : ℝ) = 490 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_earned_l1680_168084


namespace NUMINAMATH_CALUDE_midpoint_exists_but_no_centroid_l1680_168082

/-- A triangle in 2D space -/
structure Triangle :=
  (v1 v2 v3 : ℝ × ℝ)

/-- Check if a point is inside a triangle -/
def isInsideTriangle (t : Triangle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Check if a point is on the perimeter of a triangle -/
def isOnPerimeter (t : Triangle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (a b m : ℝ × ℝ) : Prop :=
  sorry

/-- Check if a point is the centroid of a triangle -/
def isCentroid (t : Triangle) (c : ℝ × ℝ) : Prop :=
  sorry

theorem midpoint_exists_but_no_centroid (t : Triangle) (p : ℝ × ℝ) 
  (h : isInsideTriangle t p) :
  (∃ a b : ℝ × ℝ, isOnPerimeter t a ∧ isOnPerimeter t b ∧ isMidpoint a b p) ∧
  (¬ ∃ a b c : ℝ × ℝ, isOnPerimeter t a ∧ isOnPerimeter t b ∧ isOnPerimeter t c ∧
                      isCentroid (Triangle.mk a b c) p) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_exists_but_no_centroid_l1680_168082


namespace NUMINAMATH_CALUDE_complex_power_sum_l1680_168085

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1500 + 1/(z^1500) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1680_168085


namespace NUMINAMATH_CALUDE_ant_count_in_field_l1680_168070

/-- Calculates the number of ants in a rectangular field given its dimensions in feet and ant density per square inch -/
def number_of_ants (width_feet : ℝ) (length_feet : ℝ) (ants_per_sq_inch : ℝ) : ℝ :=
  width_feet * length_feet * 144 * ants_per_sq_inch

/-- Theorem stating that a 500 by 600 feet field with 4 ants per square inch contains 172,800,000 ants -/
theorem ant_count_in_field : number_of_ants 500 600 4 = 172800000 := by
  sorry

end NUMINAMATH_CALUDE_ant_count_in_field_l1680_168070


namespace NUMINAMATH_CALUDE_vishal_investment_percentage_l1680_168055

/-- Proves that Vishal invested 10% more than Trishul -/
theorem vishal_investment_percentage (raghu_investment trishul_investment vishal_investment : ℝ) :
  raghu_investment = 2100 →
  trishul_investment = 0.9 * raghu_investment →
  vishal_investment + trishul_investment + raghu_investment = 6069 →
  (vishal_investment - trishul_investment) / trishul_investment = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_vishal_investment_percentage_l1680_168055


namespace NUMINAMATH_CALUDE_locus_of_Q_l1680_168013

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -1)
def C : ℝ × ℝ := (1, 3)

-- Define a point P on line BC
def P : ℝ → ℝ × ℝ := λ t => ((1 - t) * B.1 + t * C.1, (1 - t) * B.2 + t * C.2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Define the locus equation
def locus_eq (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- State the theorem
theorem locus_of_Q (t : ℝ) :
  let p := P t
  let q := vec_add p (vec_add (vec_sub A p) (vec_add (vec_sub B p) (vec_sub C p)))
  locus_eq q.1 q.2 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_Q_l1680_168013


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1680_168071

-- Define an arithmetic sequence
def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_proof :
  (arithmeticSequence 8 (-3) 20 = -49) ∧
  (arithmeticSequence (-5) (-4) 100 = -401) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1680_168071


namespace NUMINAMATH_CALUDE_twelve_not_feasible_fourteen_feasible_l1680_168093

/-- Represents the conditions for forming a convex equiangular hexagon from equilateral triangular tiles. -/
def IsValidHexagonConfiguration (n ℓ a b c : ℕ) : Prop :=
  n = ℓ^2 - a^2 - b^2 - c^2 ∧ 
  ℓ > a + b ∧ 
  ℓ > a + c ∧ 
  ℓ > b + c

/-- States that 12 is not a feasible number of tiles for forming a convex equiangular hexagon. -/
theorem twelve_not_feasible : ¬ ∃ (ℓ a b c : ℕ), IsValidHexagonConfiguration 12 ℓ a b c :=
sorry

/-- States that 14 is a feasible number of tiles for forming a convex equiangular hexagon. -/
theorem fourteen_feasible : ∃ (ℓ a b c : ℕ), IsValidHexagonConfiguration 14 ℓ a b c :=
sorry

end NUMINAMATH_CALUDE_twelve_not_feasible_fourteen_feasible_l1680_168093


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1680_168035

/-- A point (x, y) is in the third quadrant if both x and y are negative. -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The linear function y = kx - k -/
def f (k x : ℝ) : ℝ := k * x - k

theorem linear_function_not_in_third_quadrant (k : ℝ) (h : k < 0) :
  ∀ x y : ℝ, f k x = y → ¬ in_third_quadrant x y :=
by sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1680_168035


namespace NUMINAMATH_CALUDE_inequality_solution_set_inequality_positive_reals_l1680_168001

-- Part 1: Inequality solution set
theorem inequality_solution_set (x : ℝ) :
  (x - 1) / (2 * x + 1) ≤ 0 ↔ -1/2 < x ∧ x ≤ 1 :=
sorry

-- Part 2: Inequality with positive real numbers
theorem inequality_positive_reals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/(b + c)) ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_inequality_positive_reals_l1680_168001


namespace NUMINAMATH_CALUDE_power_product_l1680_168096

theorem power_product (a b : ℝ) (n : ℕ) : (a * b) ^ n = a ^ n * b ^ n := by sorry

end NUMINAMATH_CALUDE_power_product_l1680_168096


namespace NUMINAMATH_CALUDE_count_divisible_integers_l1680_168091

theorem count_divisible_integers : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (1764 : ℤ) ∣ (m^2 - 3)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (1764 : ℤ) ∣ (m^2 - 3) → m ∈ S) ∧
    S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_integers_l1680_168091


namespace NUMINAMATH_CALUDE_banana_jar_candy_count_l1680_168004

/-- Given three jars of candy with specific relationships, prove the number of candy pieces in the banana jar. -/
theorem banana_jar_candy_count (peanut_butter grape banana : ℕ) 
  (h1 : peanut_butter = 4 * grape)
  (h2 : grape = banana + 5)
  (h3 : peanut_butter = 192) : 
  banana = 43 := by
  sorry

end NUMINAMATH_CALUDE_banana_jar_candy_count_l1680_168004


namespace NUMINAMATH_CALUDE_trapezoid_median_length_l1680_168031

/-- Given a triangle and a trapezoid with equal areas and the same altitude,
    if the base of the triangle is 24 inches and the base of the trapezoid is half the length of the triangle's base,
    then the median of the trapezoid is 12 inches. -/
theorem trapezoid_median_length 
  (triangle_area trapezoid_area : ℝ)
  (altitude : ℝ)
  (triangle_base trapezoid_base : ℝ)
  (trapezoid_median : ℝ) :
  triangle_area = trapezoid_area →
  triangle_base = 24 →
  trapezoid_base = triangle_base / 2 →
  trapezoid_median = (trapezoid_base + trapezoid_base) / 2 →
  trapezoid_median = 12 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_median_length_l1680_168031


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1680_168092

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a + i) / i = 1 + b * i → a + b = 0 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1680_168092


namespace NUMINAMATH_CALUDE_probability_of_double_l1680_168021

/-- Represents a domino with two ends --/
structure Domino :=
  (end1 : Nat)
  (end2 : Nat)

/-- A standard set of dominoes with numbers from 0 to 6 --/
def StandardDominoSet : Set Domino :=
  {d : Domino | d.end1 ≤ 6 ∧ d.end2 ≤ 6}

/-- Predicate for a double domino --/
def IsDouble (d : Domino) : Prop :=
  d.end1 = d.end2

/-- The total number of dominoes in a standard set --/
def TotalDominoes : Nat := 28

/-- The number of doubles in a standard set --/
def NumberOfDoubles : Nat := 7

theorem probability_of_double :
  (NumberOfDoubles : ℚ) / (TotalDominoes : ℚ) = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_double_l1680_168021


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1680_168074

/-- Given that quantities a and b vary inversely, prove that b = 0.375 when a = 1600 -/
theorem inverse_variation_problem (a b : ℝ) (h1 : a * b = 800 * 0.5) 
  (h2 : (2 * 800) * (b / 2) = a * b + 200) : 
  (a = 1600) → (b = 0.375) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1680_168074


namespace NUMINAMATH_CALUDE_convex_polygon_triangulation_l1680_168010

/-- A convex polygon -/
structure ConvexPolygon where
  vertices : ℕ

/-- A triangulation of a polygon -/
structure Triangulation (V : ConvexPolygon) where
  triangles : List (Fin V.vertices × Fin V.vertices × Fin V.vertices)

/-- The number of triangles a vertex is part of in a triangulation -/
def vertexTriangleCount (V : ConvexPolygon) (t : Triangulation V) (v : Fin V.vertices) : ℕ :=
  sorry

/-- Theorem stating the triangulation properties for convex polygons -/
theorem convex_polygon_triangulation (V : ConvexPolygon) :
  (∃ (t : Triangulation V), V.vertices % 3 = 0 →
    ∀ (v : Fin V.vertices), Odd (vertexTriangleCount V t v)) ∧
  (∃ (t : Triangulation V), V.vertices % 3 ≠ 0 →
    ∃ (v1 v2 : Fin V.vertices),
      Even (vertexTriangleCount V t v1) ∧
      Even (vertexTriangleCount V t v2) ∧
      ∀ (v : Fin V.vertices), v ≠ v1 → v ≠ v2 → Odd (vertexTriangleCount V t v)) :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_triangulation_l1680_168010


namespace NUMINAMATH_CALUDE_point_C_coordinates_main_theorem_l1680_168037

-- Define points A, B, and C in ℝ²
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (13, 9)
def C : ℝ × ℝ := (19, 12)

-- Define the vector from A to B
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the vector from B to C
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Theorem stating that C is the correct point
theorem point_C_coordinates : 
  BC = (1/2 : ℝ) • AB := by sorry

-- Main theorem to prove
theorem main_theorem : C = (19, 12) := by sorry

end NUMINAMATH_CALUDE_point_C_coordinates_main_theorem_l1680_168037


namespace NUMINAMATH_CALUDE_max_perpendicular_pairs_l1680_168042

/-- A line in a plane -/
structure Line

/-- A perpendicular pair of lines -/
structure PerpendicularPair (Line : Type) where
  line1 : Line
  line2 : Line

/-- A configuration of lines in a plane -/
structure PlaneConfiguration where
  lines : Finset Line
  perpendicular_pairs : Finset (PerpendicularPair Line)
  line_count : lines.card = 20

/-- The theorem stating the maximum number of perpendicular pairs -/
theorem max_perpendicular_pairs (config : PlaneConfiguration) :
  ∃ (max_config : PlaneConfiguration), 
    ∀ (c : PlaneConfiguration), c.perpendicular_pairs.card ≤ max_config.perpendicular_pairs.card ∧
    max_config.perpendicular_pairs.card = 100 :=
  sorry

end NUMINAMATH_CALUDE_max_perpendicular_pairs_l1680_168042


namespace NUMINAMATH_CALUDE_no_positive_solution_l1680_168026

theorem no_positive_solution :
  ¬ ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a * d^2 + b * d - c = 0) ∧
    (Real.sqrt a * d + Real.sqrt b * Real.sqrt d - Real.sqrt c = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_positive_solution_l1680_168026


namespace NUMINAMATH_CALUDE_cubic_extrema_l1680_168069

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x^2 + 4 * x - 7

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 4 * x + 4

/-- The discriminant of f' -/
def Δ (a : ℝ) : ℝ := (-4)^2 - 4 * 3 * a * 4

theorem cubic_extrema (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ f a max ∧ f a min ≤ f a x) ↔ 
  (a < 1/3 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_extrema_l1680_168069


namespace NUMINAMATH_CALUDE_average_score_is_1_9_l1680_168065

/-- Represents the score distribution for a test -/
structure ScoreDistribution where
  threePoints : Rat
  twoPoints : Rat
  onePoint : Rat
  zeroPoints : Rat

/-- Calculates the average score given a score distribution and number of students -/
def averageScore (dist : ScoreDistribution) (numStudents : ℕ) : ℚ :=
  (3 * dist.threePoints + 2 * dist.twoPoints + dist.onePoint) * numStudents / 100

/-- Theorem: The average score for the given test is 1.9 -/
theorem average_score_is_1_9 :
  let dist : ScoreDistribution := {
    threePoints := 30,
    twoPoints := 40,
    onePoint := 20,
    zeroPoints := 10
  }
  averageScore dist 30 = 19/10 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_1_9_l1680_168065


namespace NUMINAMATH_CALUDE_books_sold_l1680_168052

theorem books_sold (initial_books : Real) (bought_books : Real) (current_books : Real)
  (h1 : initial_books = 4.5)
  (h2 : bought_books = 175.3)
  (h3 : current_books = 62.8) :
  initial_books + bought_books - current_books = 117 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l1680_168052


namespace NUMINAMATH_CALUDE_point_above_line_l1680_168028

/-- A point (x, y) is above a line Ax + By + C = 0 if Ax + By + C < 0 -/
def IsAboveLine (x y A B C : ℝ) : Prop := A * x + B * y + C < 0

/-- The theorem states that for the point (-3, -1) to be above the line 3x - 2y - a = 0,
    a must be greater than -7 -/
theorem point_above_line (a : ℝ) :
  IsAboveLine (-3) (-1) 3 (-2) (-a) ↔ a > -7 := by
  sorry

end NUMINAMATH_CALUDE_point_above_line_l1680_168028


namespace NUMINAMATH_CALUDE_matrix_product_l1680_168090

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 2; 3, 4]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![4, 3; 2, 1]

theorem matrix_product :
  A * B = !![8, 5; 20, 13] := by sorry

end NUMINAMATH_CALUDE_matrix_product_l1680_168090


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1680_168020

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality
  (a : ℕ → ℝ) (d : ℝ) (n : ℕ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_positive : d > 0)
  (h_n : n > 1) :
  a 1 * a (n + 1) < a 2 * a n :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1680_168020


namespace NUMINAMATH_CALUDE_no_integer_solution_l1680_168044

theorem no_integer_solution : ∀ x y : ℤ, x^5 + y^5 + 1 ≠ (x+2)^5 + (y-3)^5 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1680_168044


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l1680_168040

theorem min_value_sum_of_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (m : ℝ), m = 16 - 2 * Real.sqrt 2 ∧ ∀ x y, x > 0 → y > 0 → x + y = 4 → x^2 + y^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l1680_168040


namespace NUMINAMATH_CALUDE_solution_value_l1680_168039

theorem solution_value (a b : ℝ) (h : a * 3^2 - b * 3 = 6) : 2023 - 6 * a + 2 * b = 2019 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1680_168039


namespace NUMINAMATH_CALUDE_complex_root_pair_l1680_168095

theorem complex_root_pair (z : ℂ) :
  (3 + 8*I : ℂ)^2 = -55 + 48*I →
  z^2 = -55 + 48*I →
  z = 3 + 8*I ∨ z = -3 - 8*I :=
by sorry

end NUMINAMATH_CALUDE_complex_root_pair_l1680_168095


namespace NUMINAMATH_CALUDE_square_sum_and_reciprocal_l1680_168008

theorem square_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_reciprocal_l1680_168008


namespace NUMINAMATH_CALUDE_iphone_price_decrease_l1680_168062

def initial_price : ℝ := 1000
def first_month_decrease : ℝ := 0.1
def final_price : ℝ := 720

theorem iphone_price_decrease : 
  let price_after_first_month := initial_price * (1 - first_month_decrease)
  let second_month_decrease := (price_after_first_month - final_price) / price_after_first_month
  second_month_decrease = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_iphone_price_decrease_l1680_168062


namespace NUMINAMATH_CALUDE_millipede_segment_ratio_l1680_168000

/-- Proves that the ratio of segments of two unknown-length millipedes to a 60-segment millipede is 4:1 --/
theorem millipede_segment_ratio : 
  ∀ (x : ℕ), -- x represents the number of segments in each of the two unknown-length millipedes
  (2 * x + 60 + 500 = 800) → -- Total segments equation
  ((2 * x) : ℚ) / 60 = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_millipede_segment_ratio_l1680_168000


namespace NUMINAMATH_CALUDE_profit_per_meter_l1680_168059

/-- Given a trader selling cloth, calculate the profit per meter. -/
theorem profit_per_meter (total_profit : ℝ) (total_meters : ℝ) (h1 : total_profit = 1400) (h2 : total_meters = 40) :
  total_profit / total_meters = 35 := by
sorry

end NUMINAMATH_CALUDE_profit_per_meter_l1680_168059


namespace NUMINAMATH_CALUDE_billy_lemon_heads_l1680_168094

/-- The number of friends Billy gave Lemon Heads to -/
def num_friends : ℕ := 6

/-- The number of Lemon Heads each friend ate -/
def lemon_heads_per_friend : ℕ := 12

/-- The initial number of Lemon Heads Billy had -/
def initial_lemon_heads : ℕ := num_friends * lemon_heads_per_friend

theorem billy_lemon_heads :
  initial_lemon_heads = 72 :=
by sorry

end NUMINAMATH_CALUDE_billy_lemon_heads_l1680_168094


namespace NUMINAMATH_CALUDE_triangle_area_l1680_168034

/-- The area of a triangular region bounded by two coordinate axes and the line 3x + 2y = 12 is 12 square units. -/
theorem triangle_area : Real := by
  -- Define the line equation
  let line_equation (x y : Real) := 3 * x + 2 * y = 12

  -- Define the x-intercept
  let x_intercept : Real := 4

  -- Define the y-intercept
  let y_intercept : Real := 6

  -- Define the area of the triangle
  let triangle_area : Real := (1 / 2) * x_intercept * y_intercept

  -- Prove that the area is 12 square units
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l1680_168034


namespace NUMINAMATH_CALUDE_area_triangle_ABG_l1680_168054

/-- Given a rectangle ABCD and a square AEFG, where AB = 6, AD = 4, and the area of triangle ADE is 2,
    prove that the area of triangle ABG is 3. -/
theorem area_triangle_ABG (A B C D E F G : ℝ × ℝ) : 
  (∀ X Y, X ≠ Y → (X = A ∧ Y = B) ∨ (X = B ∧ Y = C) ∨ (X = C ∧ Y = D) ∨ (X = D ∧ Y = A) → 
    (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (Y.1 - X.1)^2 + (Y.2 - X.2)^2) →  -- ABCD is a rectangle
  (∀ X Y, X ≠ Y → (X = A ∧ Y = E) ∨ (X = E ∧ Y = F) ∨ (X = F ∧ Y = G) ∨ (X = G ∧ Y = A) → 
    (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (E.1 - A.1)^2 + (E.2 - A.2)^2) →  -- AEFG is a square
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 →  -- AB = 6
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = 16 →  -- AD = 4
  abs ((E.1 - A.1) * (D.2 - A.2) - (E.2 - A.2) * (D.1 - A.1)) / 2 = 2 →  -- Area of triangle ADE = 2
  abs ((G.1 - A.1) * (B.2 - A.2) - (G.2 - A.2) * (B.1 - A.1)) / 2 = 3  -- Area of triangle ABG = 3
  := by sorry

end NUMINAMATH_CALUDE_area_triangle_ABG_l1680_168054


namespace NUMINAMATH_CALUDE_odd_triangle_perimeter_l1680_168064

/-- A triangle with two sides of lengths 2 and 3, and the third side being an odd number -/
structure OddTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℕ
  h1 : side1 = 2
  h2 : side2 = 3
  h3 : Odd side3
  h4 : side3 > 0  -- Ensuring positive length
  h5 : side1 + side2 > side3  -- Triangle inequality
  h6 : side1 + side3 > side2
  h7 : side2 + side3 > side1

/-- The perimeter of an OddTriangle is 8 -/
theorem odd_triangle_perimeter (t : OddTriangle) : t.side1 + t.side2 + t.side3 = 8 :=
by sorry

end NUMINAMATH_CALUDE_odd_triangle_perimeter_l1680_168064


namespace NUMINAMATH_CALUDE_largest_two_digit_remainder_2_mod_13_l1680_168007

theorem largest_two_digit_remainder_2_mod_13 :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 13 = 2 → n ≤ 93 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_remainder_2_mod_13_l1680_168007


namespace NUMINAMATH_CALUDE_some_multiplier_value_l1680_168018

theorem some_multiplier_value : ∃ m : ℕ, (422 + 404)^2 - (m * 422 * 404) = 324 ∧ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_multiplier_value_l1680_168018


namespace NUMINAMATH_CALUDE_units_digit_of_cube_minus_square_l1680_168067

def n : ℕ := 9867

theorem units_digit_of_cube_minus_square :
  (n^3 - n^2) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_cube_minus_square_l1680_168067


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1680_168048

def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

theorem right_triangle_sets :
  ¬(is_right_triangle 6 7 8) ∧
  ¬(is_right_triangle 1 (Real.sqrt 2) 5) ∧
  is_right_triangle 6 8 10 ∧
  ¬(is_right_triangle (Real.sqrt 5) (2 * Real.sqrt 3) (Real.sqrt 15)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1680_168048


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1680_168050

theorem right_triangle_inequality (a b c h : ℝ) (n : ℕ) (h1 : 0 < n) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 0 < h) 
  (h6 : a^2 + b^2 = c^2) (h7 : a * b = c * h) (h8 : a + b < c + h) :
  a^n + b^n < c^n + h^n := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1680_168050


namespace NUMINAMATH_CALUDE_current_at_6_seconds_l1680_168068

/-- The charge function Q(t) representing the amount of electricity flowing through a conductor. -/
def Q (t : ℝ) : ℝ := 3 * t^2 - 3 * t + 4

/-- The current function I(t) derived from Q(t). -/
def I (t : ℝ) : ℝ := 6 * t - 3

/-- Theorem stating that the current at t = 6 seconds is 33 amperes. -/
theorem current_at_6_seconds :
  I 6 = 33 := by sorry

end NUMINAMATH_CALUDE_current_at_6_seconds_l1680_168068


namespace NUMINAMATH_CALUDE_max_value_expression_l1680_168033

theorem max_value_expression (x y : ℝ) : 
  (Real.sqrt (3 - Real.sqrt 2) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) *
  (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1680_168033


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l1680_168046

/-- Given that the line x + y = b is the perpendicular bisector of the line segment 
    from (2,4) to (6,10), prove that b = 11. -/
theorem perpendicular_bisector_b_value : 
  let point1 : ℝ × ℝ := (2, 4)
  let point2 : ℝ × ℝ := (6, 10)
  let midpoint : ℝ × ℝ := ((point1.1 + point2.1) / 2, (point1.2 + point2.2) / 2)
  ∃ b : ℝ, (∀ (x y : ℝ), x + y = b ↔ ((x - midpoint.1) ^ 2 + (y - midpoint.2) ^ 2 = 
    (point1.1 - midpoint.1) ^ 2 + (point1.2 - midpoint.2) ^ 2)) → b = 11 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l1680_168046


namespace NUMINAMATH_CALUDE_fib_mod_10_periodic_fib_mod_10_smallest_period_l1680_168049

/-- Fibonacci sequence modulo 10 -/
def fib_mod_10 : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (fib_mod_10 n + fib_mod_10 (n + 1)) % 10

/-- The period of the Fibonacci sequence modulo 10 -/
def fib_mod_10_period : ℕ := 60

/-- Theorem: The Fibonacci sequence modulo 10 has a period of 60 -/
theorem fib_mod_10_periodic :
  ∀ n : ℕ, fib_mod_10 (n + fib_mod_10_period) = fib_mod_10 n :=
by
  sorry

/-- Theorem: 60 is the smallest positive period of the Fibonacci sequence modulo 10 -/
theorem fib_mod_10_smallest_period :
  ∀ k : ℕ, k > 0 → k < fib_mod_10_period →
    ∃ n : ℕ, fib_mod_10 (n + k) ≠ fib_mod_10 n :=
by
  sorry

end NUMINAMATH_CALUDE_fib_mod_10_periodic_fib_mod_10_smallest_period_l1680_168049


namespace NUMINAMATH_CALUDE_total_bills_is_126_l1680_168012

/-- Represents the number of bills and their total value -/
structure CashierMoney where
  five_dollar_bills : ℕ
  ten_dollar_bills : ℕ
  total_value : ℕ

/-- Theorem stating that given the conditions, the total number of bills is 126 -/
theorem total_bills_is_126 (money : CashierMoney) 
  (h1 : money.five_dollar_bills = 84)
  (h2 : money.total_value = 840)
  (h3 : money.total_value = 5 * money.five_dollar_bills + 10 * money.ten_dollar_bills) :
  money.five_dollar_bills + money.ten_dollar_bills = 126 := by
  sorry


end NUMINAMATH_CALUDE_total_bills_is_126_l1680_168012


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l1680_168081

theorem seven_eighths_of_48 : (7 : ℚ) / 8 * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l1680_168081


namespace NUMINAMATH_CALUDE_intersection_points_range_l1680_168016

/-- The range of m for which curves C₁ and C₂ have 4 distinct intersection points -/
theorem intersection_points_range (m : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ), 
    (∀ i j, (i, j) ∈ [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] → 
      (i - 1)^2 + j^2 = 1 ∧ j * (j - m*i - m) = 0) ∧
    (∀ i j k l, (i, j) ≠ (k, l) → (i, j) ∈ [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] → 
      (k, l) ∈ [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] → (i, j) ≠ (k, l))) ↔ 
  (m > -Real.sqrt 3 / 3 ∧ m < 0) ∨ (m > 0 ∧ m < Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_range_l1680_168016


namespace NUMINAMATH_CALUDE_reciprocal_of_2023_l1680_168060

theorem reciprocal_of_2023 : (2023⁻¹ : ℚ) = 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2023_l1680_168060


namespace NUMINAMATH_CALUDE_apple_arrangements_l1680_168032

def word : String := "APPLE"

def letter_count : Nat := word.length

def letter_frequencies : List (Char × Nat) := [('A', 1), ('P', 2), ('L', 1), ('E', 1)]

/-- The number of distinct arrangements of the letters in the word "APPLE" -/
def distinct_arrangements : Nat := 60

/-- Theorem stating that the number of distinct arrangements of the letters in "APPLE" is 60 -/
theorem apple_arrangements :
  distinct_arrangements = 60 :=
by sorry

end NUMINAMATH_CALUDE_apple_arrangements_l1680_168032


namespace NUMINAMATH_CALUDE_largest_geometric_digit_sequence_l1680_168047

/-- Checks if the given three digits form a geometric sequence -/
def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, b = r * a ∧ c = r * b

/-- Checks if the given number is a valid solution -/
def is_valid_solution (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit integer
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- Distinct digits
  is_geometric_sequence a b c ∧  -- Geometric sequence
  b % 2 = 0  -- Tens digit is even

theorem largest_geometric_digit_sequence :
  ∀ n : ℕ, is_valid_solution n → n ≤ 964 :=
sorry

end NUMINAMATH_CALUDE_largest_geometric_digit_sequence_l1680_168047


namespace NUMINAMATH_CALUDE_school_band_seats_l1680_168056

/-- Represents the number of seats needed for the school band --/
def total_seats (flute trumpet trombone drummer clarinet french_horn : ℕ) : ℕ :=
  flute + trumpet + trombone + drummer + clarinet + french_horn

/-- Theorem stating the total number of seats needed for the school band --/
theorem school_band_seats :
  ∃ (flute trumpet trombone drummer clarinet french_horn : ℕ),
    flute = 5 ∧
    trumpet = 3 * flute ∧
    trombone = trumpet - 8 ∧
    drummer = trombone + 11 ∧
    clarinet = 2 * flute ∧
    french_horn = trombone + 3 ∧
    total_seats flute trumpet trombone drummer clarinet french_horn = 65 := by
  sorry

end NUMINAMATH_CALUDE_school_band_seats_l1680_168056


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l1680_168075

theorem line_slope_intercept_product (m b : ℚ) : 
  m = -3/4 → b = 3/2 → m * b < -1 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l1680_168075


namespace NUMINAMATH_CALUDE_circle_equation_and_slope_range_l1680_168097

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the line y = 2x
def line_center (x y : ℝ) : Prop := y = 2 * x

-- Define the line x + y - 3 = 0
def line_intersect (x y : ℝ) : Prop := x + y - 3 = 0

-- Define points A and B
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define point M
def point_M : ℝ × ℝ := (0, 5)

-- Define the dot product of OA and OB
def OA_dot_OB_zero : Prop :=
  (point_A.1 - origin.1) * (point_B.1 - origin.1) + 
  (point_A.2 - origin.2) * (point_B.2 - origin.2) = 0

-- Define the slope range for line MP
def slope_range (k : ℝ) : Prop := k ≤ -1/2 ∨ k ≥ 2

theorem circle_equation_and_slope_range :
  (∀ x y, circle_C x y → ((x, y) = origin ∨ line_center x y)) ∧
  (∀ x y, line_intersect x y → circle_C x y → ((x, y) = point_A ∨ (x, y) = point_B)) ∧
  OA_dot_OB_zero →
  (∀ x y, circle_C x y ↔ (x - 1)^2 + (y - 2)^2 = 5) ∧
  (∀ k, (∃ x y, circle_C x y ∧ y - 5 = k * x) ↔ slope_range k) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_and_slope_range_l1680_168097


namespace NUMINAMATH_CALUDE_village_population_l1680_168061

theorem village_population (population : ℝ) : 
  (0.9 * population = 45000) → population = 50000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1680_168061


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_circles_lines_theorem_l1680_168076

-- Ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 / 16 = 1

-- Circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*y - 1 = 0

-- Lines
def line1 (a x y : ℝ) : Prop := a^2 * x - y + 6 = 0
def line2 (a x y : ℝ) : Prop := 4 * x - (a - 3) * y + 9 = 0

theorem ellipse_hyperbola_circles_lines_theorem :
  (∃ (F₁ F₂ P : ℝ × ℝ), ellipse P.1 P.2 ∧ |P.1 - F₁.1| + |P.2 - F₁.2| = 3 ∧ |P.1 - F₂.1| + |P.2 - F₂.2| ≠ 1) ∧
  (∀ (x y : ℝ), hyperbola x y → (|y| - |3/4 * x| = 12/5)) ∧
  (∃ (t₁ t₂ : ℝ × ℝ), t₁ ≠ t₂ ∧ (∀ (x y : ℝ), circle1 x y → (x - t₁.1)^2 + (y - t₁.2)^2 = 0) ∧
                                 (∀ (x y : ℝ), circle2 x y → (x - t₁.1)^2 + (y - t₁.2)^2 = 0) ∧
                                 (∀ (x y : ℝ), circle1 x y → (x - t₂.1)^2 + (y - t₂.2)^2 = 0) ∧
                                 (∀ (x y : ℝ), circle2 x y → (x - t₂.1)^2 + (y - t₂.2)^2 = 0)) ∧
  (∃ (a : ℝ), a ≠ -1 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ), line1 a x₁ y₁ ∧ line2 a x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) ≠ -1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_circles_lines_theorem_l1680_168076


namespace NUMINAMATH_CALUDE_calculate_total_profit_total_profit_is_150000_l1680_168022

/-- Calculates the total profit given investment ratios and B's profit -/
theorem calculate_total_profit (a_c_ratio : Rat) (a_b_ratio : Rat) (b_profit : ℕ) : ℕ :=
  let a_c_ratio := 2/1
  let a_b_ratio := 2/3
  let b_profit := 75000
  2 * b_profit

theorem total_profit_is_150000 : 
  calculate_total_profit (2/1) (2/3) 75000 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_profit_total_profit_is_150000_l1680_168022


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l1680_168080

structure Line3D where
  -- Assume we have a way to represent 3D lines
  mk :: 

structure Plane where
  -- Assume we have a way to represent planes
  mk ::

def contained_in (l : Line3D) (p : Plane) : Prop :=
  -- Definition for a line being contained in a plane
  sorry

def parallel_to_plane (l : Line3D) (p : Plane) : Prop :=
  -- Definition for a line being parallel to a plane
  sorry

def coplanar (l1 l2 : Line3D) : Prop :=
  -- Definition for two lines being coplanar
  sorry

def parallel_lines (l1 l2 : Line3D) : Prop :=
  -- Definition for two lines being parallel
  sorry

theorem line_plane_parallelism 
  (m n : Line3D) (α : Plane) 
  (h1 : contained_in m α) 
  (h2 : parallel_to_plane n α) 
  (h3 : coplanar m n) : 
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l1680_168080


namespace NUMINAMATH_CALUDE_froglet_is_sane_l1680_168030

-- Define the servants
inductive Servant
| LackeyLecc
| Froglet

-- Define the sanity state
inductive SanityState
| Sane
| Insane

-- Define a function to represent the claim of Lackey-Lecc
def lackey_lecc_claim (lackey_state froglet_state : SanityState) : Prop :=
  (lackey_state = SanityState.Sane ∧ froglet_state = SanityState.Sane) ∨
  (lackey_state = SanityState.Insane ∧ froglet_state = SanityState.Insane)

-- Theorem stating that Froglet is sane
theorem froglet_is_sane :
  ∀ (lackey_state : SanityState),
    (lackey_lecc_claim lackey_state SanityState.Sane) →
    SanityState.Sane = SanityState.Sane :=
by
  sorry


end NUMINAMATH_CALUDE_froglet_is_sane_l1680_168030


namespace NUMINAMATH_CALUDE_female_students_count_l1680_168017

theorem female_students_count (total_average : ℝ) (male_count : ℕ) (male_average : ℝ) (female_average : ℝ)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 85)
  (h4 : female_average = 92) :
  ∃ (female_count : ℕ),
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l1680_168017


namespace NUMINAMATH_CALUDE_surface_area_five_cube_removal_l1680_168009

/-- The surface area of a cube after removing central columns -/
def surface_area_after_removal (n : ℕ) : ℕ :=
  let original_surface_area := 6 * n^2
  let removed_surface_area := 6 * (n^2 - 1)
  let added_internal_surface := 2 * 3 * 4 * (n - 1)
  removed_surface_area + added_internal_surface

/-- Theorem stating that the surface area of a 5×5×5 cube after removing central columns is 192 -/
theorem surface_area_five_cube_removal :
  surface_area_after_removal 5 = 192 := by
  sorry

#eval surface_area_after_removal 5

end NUMINAMATH_CALUDE_surface_area_five_cube_removal_l1680_168009


namespace NUMINAMATH_CALUDE_pool_depth_calculation_l1680_168077

/-- Calculates the depth of a rectangular pool given its dimensions and draining specifications. -/
theorem pool_depth_calculation (width : ℝ) (length : ℝ) (drain_rate : ℝ) (drain_time : ℝ) (capacity_percentage : ℝ) :
  width = 50 →
  length = 150 →
  drain_rate = 60 →
  drain_time = 1000 →
  capacity_percentage = 0.8 →
  (width * length * (drain_rate * drain_time / capacity_percentage)) / (width * length) = 10 :=
by
  sorry

#check pool_depth_calculation

end NUMINAMATH_CALUDE_pool_depth_calculation_l1680_168077


namespace NUMINAMATH_CALUDE_tv_show_watch_time_l1680_168098

theorem tv_show_watch_time : 
  let regular_seasons : ℕ := 9
  let episodes_per_regular_season : ℕ := 22
  let episodes_in_last_season : ℕ := 26
  let episode_duration : ℚ := 1/2
  
  let total_episodes : ℕ := 
    regular_seasons * episodes_per_regular_season + episodes_in_last_season
  
  let total_watch_time : ℚ := total_episodes * episode_duration
  
  total_watch_time = 112 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_watch_time_l1680_168098


namespace NUMINAMATH_CALUDE_fifteenth_prime_l1680_168072

/-- Given that 5 is the third prime number, prove that the fifteenth prime number is 59. -/
theorem fifteenth_prime : 
  (∃ (f : ℕ → ℕ), f 3 = 5 ∧ (∀ n, n ≥ 1 → Prime (f n)) ∧ (∀ n m, n < m → f n < f m)) → 
  (∃ (g : ℕ → ℕ), g 15 = 59 ∧ (∀ n, n ≥ 1 → Prime (g n)) ∧ (∀ n m, n < m → g n < g m)) :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_prime_l1680_168072


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1680_168063

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 4) : a^3 + 1/a^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1680_168063


namespace NUMINAMATH_CALUDE_max_value_xyz_l1680_168043

theorem max_value_xyz (x y z : ℝ) (h1 : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) 
  (h2 : x + y + z = 1) (h3 : x^2 + y^2 + z^2 = 1) : 
  x + y^3 + z^4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xyz_l1680_168043


namespace NUMINAMATH_CALUDE_sum_of_f_zero_and_f_neg_two_l1680_168083

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f y = f (x + y)

theorem sum_of_f_zero_and_f_neg_two (f : ℝ → ℝ) 
  (h1 : functional_equation f) 
  (h2 : f 2 = 4) : 
  f 0 + f (-2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_zero_and_f_neg_two_l1680_168083


namespace NUMINAMATH_CALUDE_misread_number_correction_l1680_168066

theorem misread_number_correction (n : ℕ) (incorrect_avg correct_avg misread_value : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 18)
  (h3 : correct_avg = 22)
  (h4 : misread_value = 26) :
  ∃ (actual_value : ℚ), 
    n * correct_avg = n * incorrect_avg - misread_value + actual_value ∧ 
    actual_value = 66 := by
  sorry

end NUMINAMATH_CALUDE_misread_number_correction_l1680_168066


namespace NUMINAMATH_CALUDE_cos_240_degrees_l1680_168003

theorem cos_240_degrees : Real.cos (240 * π / 180) = -(1/2) := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l1680_168003


namespace NUMINAMATH_CALUDE_quarter_sector_area_l1680_168019

/-- The area of a quarter sector of a circle with diameter 10 meters -/
theorem quarter_sector_area (d : ℝ) (h : d = 10) : 
  (π * (d / 2)^2) / 4 = 6.25 * π := by
  sorry

end NUMINAMATH_CALUDE_quarter_sector_area_l1680_168019


namespace NUMINAMATH_CALUDE_soda_water_ratio_l1680_168002

theorem soda_water_ratio (water soda k : ℕ) : 
  water + soda = 54 →
  soda = k * water - 6 →
  k > 0 →
  soda * 5 = water * 4 := by
sorry

end NUMINAMATH_CALUDE_soda_water_ratio_l1680_168002


namespace NUMINAMATH_CALUDE_farmer_cows_problem_l1680_168025

theorem farmer_cows_problem (initial_cows : ℕ) : 
  (3 / 4 : ℚ) * (initial_cows + 5 : ℚ) = 42 → initial_cows = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_cows_problem_l1680_168025


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l1680_168006

theorem polynomial_equation_solution (a a1 a2 a3 a4 : ℝ) : 
  (∀ x, (x + a)^4 = x^4 + a1*x^3 + a2*x^2 + a3*x + a4) →
  (a1 + a2 + a3 = 64) →
  (a = 2) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l1680_168006


namespace NUMINAMATH_CALUDE_pumpkin_count_l1680_168086

/-- The total number of pumpkins grown by Sandy, Mike, Maria, and Sam -/
def total_pumpkins (sandy mike maria sam : ℕ) : ℕ := sandy + mike + maria + sam

/-- Theorem stating that the total number of pumpkins is 157 -/
theorem pumpkin_count : total_pumpkins 51 23 37 46 = 157 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_count_l1680_168086


namespace NUMINAMATH_CALUDE_max_handshakers_l1680_168099

/-- Given a room with N people, where N > 4, and at least two people have not shaken
    hands with everyone else, the maximum number of people who could have shaken
    hands with everyone else is N-2. -/
theorem max_handshakers (N : ℕ) (h1 : N > 4) (h2 : ∃ (a b : ℕ), a ≠ b ∧ a < N ∧ b < N ∧ 
  (∃ (c : ℕ), c < N ∧ c ≠ a ∧ c ≠ b)) : 
  ∃ (M : ℕ), M = N - 2 ∧ 
  (∀ (k : ℕ), k ≤ N → (∃ (S : Finset ℕ), S.card = k ∧ 
    (∀ (i j : ℕ), i ∈ S → j ∈ S → i ≠ j → (∃ (H : Prop), H)) → k ≤ M)) :=
sorry

end NUMINAMATH_CALUDE_max_handshakers_l1680_168099


namespace NUMINAMATH_CALUDE_fourth_person_height_l1680_168045

/-- Theorem: Height of the fourth person in a specific arrangement --/
theorem fourth_person_height 
  (h₁ h₂ h₃ h₄ : ℝ) 
  (height_order : h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄)
  (diff_first_three : h₂ - h₁ = 2 ∧ h₃ - h₂ = 2)
  (diff_last_two : h₄ - h₃ = 6)
  (average_height : (h₁ + h₂ + h₃ + h₄) / 4 = 76) :
  h₄ = 82 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l1680_168045
