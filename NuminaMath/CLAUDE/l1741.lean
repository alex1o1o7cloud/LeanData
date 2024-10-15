import Mathlib

namespace NUMINAMATH_CALUDE_range_of_f_l1741_174134

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 12

-- Statement to prove
theorem range_of_f :
  Set.range f = Set.Ici 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1741_174134


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_25_mod_26_l1741_174179

theorem largest_four_digit_congruent_to_25_mod_26 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n ≡ 25 [MOD 26] → n ≤ 9983 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_25_mod_26_l1741_174179


namespace NUMINAMATH_CALUDE_bankers_gain_is_nine_l1741_174159

/-- Calculates the banker's gain given the true discount, time period, and interest rate. -/
def bankers_gain (true_discount : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  let face_value := (true_discount * (100 + (rate * time))) / (rate * time)
  let bankers_discount := (face_value * rate * time) / 100
  bankers_discount - true_discount

/-- Theorem stating that the banker's gain is 9 given the specified conditions. -/
theorem bankers_gain_is_nine :
  bankers_gain 75 1 12 = 9 := by
  sorry

#eval bankers_gain 75 1 12

end NUMINAMATH_CALUDE_bankers_gain_is_nine_l1741_174159


namespace NUMINAMATH_CALUDE_mirror_pieces_l1741_174127

theorem mirror_pieces (total : ℕ) (swept : ℕ) (stolen : ℕ) (picked : ℕ) : 
  total = 60 →
  swept = total / 2 →
  stolen = 3 →
  picked = (total - swept - stolen) / 3 →
  picked = 9 := by
sorry

end NUMINAMATH_CALUDE_mirror_pieces_l1741_174127


namespace NUMINAMATH_CALUDE_car_distribution_l1741_174104

/-- The number of cars produced annually by American carmakers -/
def total_cars : ℕ := 5650000

/-- The number of car suppliers -/
def num_suppliers : ℕ := 5

/-- The number of cars received by the first supplier -/
def first_supplier : ℕ := 1000000

/-- The number of cars received by the second supplier -/
def second_supplier : ℕ := first_supplier + 500000

/-- The number of cars received by the third supplier -/
def third_supplier : ℕ := first_supplier + second_supplier

/-- The number of cars received by each of the fourth and fifth suppliers -/
def fourth_fifth_supplier : ℕ := (total_cars - (first_supplier + second_supplier + third_supplier)) / 2

theorem car_distribution :
  fourth_fifth_supplier = 325000 :=
sorry

end NUMINAMATH_CALUDE_car_distribution_l1741_174104


namespace NUMINAMATH_CALUDE_allan_balloons_l1741_174142

/-- The number of balloons Allan initially brought to the park -/
def initial_balloons : ℕ := 5

/-- The number of balloons Allan bought at the park -/
def bought_balloons : ℕ := 3

/-- The total number of balloons Allan brought to the park -/
def total_balloons : ℕ := initial_balloons + bought_balloons

theorem allan_balloons : total_balloons = 8 := by
  sorry

end NUMINAMATH_CALUDE_allan_balloons_l1741_174142


namespace NUMINAMATH_CALUDE_perimeter_of_rectangle_l1741_174166

def rhombus_in_rectangle (WE XF EG FH : ℝ) : Prop :=
  WE = 10 ∧ XF = 25 ∧ EG = 20 ∧ FH = 50

theorem perimeter_of_rectangle (WE XF EG FH : ℝ) 
  (h : rhombus_in_rectangle WE XF EG FH) : 
  ∃ (perimeter : ℝ), perimeter = 53 * Real.sqrt 29 - 73 := by
  sorry

#check perimeter_of_rectangle

end NUMINAMATH_CALUDE_perimeter_of_rectangle_l1741_174166


namespace NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l1741_174107

/-- The sum of the series Σ(k=0 to ∞) (3^(2^k) / (6^(2^k) - 2)) is equal to 3/4 -/
theorem series_sum_equals_three_fourths : 
  ∑' k : ℕ, (3 : ℝ)^(2^k) / ((6 : ℝ)^(2^k) - 2) = 3/4 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l1741_174107


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1741_174130

theorem circle_tangent_to_line (r : ℝ) (h : r = Real.sqrt 5) :
  ∃ (c1 c2 : ℝ × ℝ),
    c1.2 = 0 ∧ c2.2 = 0 ∧
    (∀ (x y : ℝ), (x - c1.1)^2 + (y - c1.2)^2 = r^2 ↔ (x + 2*y)^2 = 5) ∧
    (∀ (x y : ℝ), (x - c2.1)^2 + (y - c2.2)^2 = r^2 ↔ (x + 2*y)^2 = 5) ∧
    c1 = (5, 0) ∧ c2 = (-5, 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1741_174130


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l1741_174144

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : -1 < x ∧ x < 5)
  (h3 : 0 < x ∧ x < 3)
  (h4 : x + 2 < 4) :
  ∀ y : ℤ, (0 < x ∧ x < y) → y ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l1741_174144


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_one_l1741_174113

theorem sqrt_meaningful_iff_geq_one (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_one_l1741_174113


namespace NUMINAMATH_CALUDE_square_sum_fifteen_l1741_174192

theorem square_sum_fifteen (x y : ℝ) 
  (h1 : y + 4 = (x - 2)^2) 
  (h2 : x + 4 = (y - 2)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_square_sum_fifteen_l1741_174192


namespace NUMINAMATH_CALUDE_basketball_free_throws_l1741_174186

theorem basketball_free_throws :
  ∀ (two_points three_points free_throws : ℕ),
    2 * (2 * two_points) = 3 * three_points →
    free_throws = 2 * two_points →
    2 * two_points + 3 * three_points + free_throws = 74 →
    free_throws = 30 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l1741_174186


namespace NUMINAMATH_CALUDE_valid_paths_count_l1741_174160

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a move direction -/
inductive Move
  | Right
  | Up
  | Diagonal

/-- Defines a valid path on the Cartesian plane -/
def ValidPath (start finish : Point) (path : List Move) : Prop :=
  -- Path starts at the start point and ends at the finish point
  -- Each move is valid according to the problem conditions
  -- No right-angle turns in the path
  sorry

/-- Counts the number of valid paths between two points -/
def CountValidPaths (start finish : Point) : ℕ :=
  sorry

theorem valid_paths_count :
  CountValidPaths (Point.mk 0 0) (Point.mk 6 6) = 128 := by
  sorry

end NUMINAMATH_CALUDE_valid_paths_count_l1741_174160


namespace NUMINAMATH_CALUDE_max_x5_value_l1741_174114

theorem max_x5_value (x₁ x₂ x₃ x₄ x₅ : ℕ) 
  (h : x₁ + x₂ + x₃ + x₄ + x₅ = x₁ * x₂ * x₃ * x₄ * x₅) :
  x₅ ≤ 5 ∧ ∃ y₁ y₂ y₃ y₄ : ℕ, y₁ + y₂ + y₃ + y₄ + 5 = y₁ * y₂ * y₃ * y₄ * 5 :=
by sorry

end NUMINAMATH_CALUDE_max_x5_value_l1741_174114


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1741_174194

theorem solution_set_inequality (x : ℝ) : (x - 2) / (x + 1) < 0 ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1741_174194


namespace NUMINAMATH_CALUDE_trapezoid_area_l1741_174137

/-- The area of a trapezoid bounded by y = ax, y = bx, x = c, and x = d in the first quadrant -/
theorem trapezoid_area (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hcd : c < d) :
  let area := 0.5 * ((a * c + a * d + b * c + b * d) * (d - c))
  ∃ (trapezoid_area : ℝ), trapezoid_area = area := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1741_174137


namespace NUMINAMATH_CALUDE_function_value_problem_l1741_174153

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f (1/2 * x - 1) = 2 * x + 3)
  (h2 : f (m - 1) = 6) : 
  m = 3/4 := by
sorry

end NUMINAMATH_CALUDE_function_value_problem_l1741_174153


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l1741_174169

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l1741_174169


namespace NUMINAMATH_CALUDE_square_ratio_sum_l1741_174141

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 300 / 75 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt area_ratio →
  a + b + c = 4 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_sum_l1741_174141


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l1741_174151

theorem largest_prime_factor_of_12321 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 12321 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 12321 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l1741_174151


namespace NUMINAMATH_CALUDE_min_value_expression_l1741_174132

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → 42 + b^2 + 1/(a*b) ≤ 42 + y^2 + 1/(x*y) ∧
  ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 42 + b₀^2 + 1/(a₀*b₀) = 17/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1741_174132


namespace NUMINAMATH_CALUDE_remainder_4873_div_29_l1741_174177

theorem remainder_4873_div_29 : 4873 % 29 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4873_div_29_l1741_174177


namespace NUMINAMATH_CALUDE_ladybugs_with_spots_count_l1741_174158

/-- Given the total number of ladybugs and the number of ladybugs without spots,
    calculate the number of ladybugs with spots. -/
def ladybugsWithSpots (total : ℕ) (withoutSpots : ℕ) : ℕ :=
  total - withoutSpots

/-- Theorem stating that given 67,082 total ladybugs and 54,912 ladybugs without spots,
    there are 12,170 ladybugs with spots. -/
theorem ladybugs_with_spots_count :
  ladybugsWithSpots 67082 54912 = 12170 := by
  sorry

end NUMINAMATH_CALUDE_ladybugs_with_spots_count_l1741_174158


namespace NUMINAMATH_CALUDE_expression_evaluation_l1741_174133

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/3
  ((2*x + 3*y)^2 - (2*x + 3*y)*(2*x - 3*y)) / (3*y) = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1741_174133


namespace NUMINAMATH_CALUDE_max_value_of_e_l1741_174115

def b (n : ℕ) : ℤ := (5^n - 1) / 4

def e (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1)))

theorem max_value_of_e (n : ℕ) : e n = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_e_l1741_174115


namespace NUMINAMATH_CALUDE_symmetry_implies_periodic_l1741_174147

/-- A function that is symmetric with respect to two distinct points is periodic. -/
theorem symmetry_implies_periodic (f : ℝ → ℝ) (a b : ℝ) 
  (ha : ∀ x, f (a - x) = f (a + x))
  (hb : ∀ x, f (b - x) = f (b + x))
  (hab : a ≠ b) :
  ∀ x, f (x + (2*b - 2*a)) = f x :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_periodic_l1741_174147


namespace NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l1741_174140

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial : trailingZeroes 500 = 124 := by sorry

end NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l1741_174140


namespace NUMINAMATH_CALUDE_quadratic_root_l1741_174150

theorem quadratic_root (p q r : ℝ) (h : p ≠ 0 ∧ q ≠ r) :
  let f : ℝ → ℝ := λ x ↦ p * (q - r) * x^2 + q * (r - p) * x + r * (p - q)
  (f 1 = 0) →
  ∃ x : ℝ, x ≠ 1 ∧ f x = 0 ∧ x = r * (p - q) / (p * (q - r)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_l1741_174150


namespace NUMINAMATH_CALUDE_solve_system_l1741_174152

theorem solve_system (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 2 / x) :
  y = (5 + Real.sqrt 41) / 4 ∨ y = (5 - Real.sqrt 41) / 4 :=
by sorry

end NUMINAMATH_CALUDE_solve_system_l1741_174152


namespace NUMINAMATH_CALUDE_fixed_point_theorem_a_value_theorem_minimum_point_theorem_l1741_174101

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x - 2

theorem fixed_point_theorem (a : ℝ) :
  f a 0 = 1 := by sorry

theorem a_value_theorem (a : ℝ) :
  (∀ x, f_deriv a x ≥ -a * x - 1) → a = 1 := by sorry

theorem minimum_point_theorem :
  ∃ x₀, (∀ x, f 1 x ≥ f 1 x₀) ∧ -2 < f 1 x₀ ∧ f 1 x₀ < -1/4 := by sorry

end

end NUMINAMATH_CALUDE_fixed_point_theorem_a_value_theorem_minimum_point_theorem_l1741_174101


namespace NUMINAMATH_CALUDE_square_fence_perimeter_l1741_174189

theorem square_fence_perimeter 
  (total_posts : ℕ) 
  (post_width : ℚ) 
  (gap_width : ℕ) : 
  total_posts = 36 → 
  post_width = 1/3 → 
  gap_width = 6 → 
  (4 * ((total_posts / 4 + 1) * post_width + (total_posts / 4) * gap_width)) = 204 := by
  sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_l1741_174189


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l1741_174111

theorem binomial_square_coefficient (x : ℝ) : ∃ b : ℝ, ∃ t u : ℝ, 
  b * x^2 + 20 * x + 1 = (t * x + u)^2 ∧ b = 100 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l1741_174111


namespace NUMINAMATH_CALUDE_quadratic_root_implies_positive_triangle_l1741_174122

theorem quadratic_root_implies_positive_triangle (a b c : ℝ) 
  (h_root : ∃ (α β : ℝ), α > 0 ∧ β ≠ 0 ∧ Complex.I * Complex.I = -1 ∧ 
    (α + Complex.I * β) ^ 2 - (a + b + c) * (α + Complex.I * β) + (a * b + b * c + c * a) = 0) :
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (Real.sqrt a + Real.sqrt b > Real.sqrt c ∧ 
   Real.sqrt b + Real.sqrt c > Real.sqrt a ∧ 
   Real.sqrt c + Real.sqrt a > Real.sqrt b) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_positive_triangle_l1741_174122


namespace NUMINAMATH_CALUDE_number_of_girls_l1741_174112

/-- The number of girls in the group -/
def n : ℕ := sorry

/-- The average weight of the group before the new girl arrives -/
def A : ℝ := sorry

/-- The weight of the new girl -/
def W : ℝ := 80

theorem number_of_girls :
  (n * A = n * A - 55 + W) ∧ (n * (A + 1) = n * A - 55 + W) → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_l1741_174112


namespace NUMINAMATH_CALUDE_inequality_and_minimum_l1741_174143

theorem inequality_and_minimum (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum_prod : x + y + z ≥ x * y * z) : 
  (x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z) ∧ 
  (∃ (u : ℝ), u = x / (y * z) + y / (z * x) + z / (x * y) ∧ 
              u ≥ Real.sqrt 3 ∧ 
              ∀ (v : ℝ), v = x / (y * z) + y / (z * x) + z / (x * y) → v ≥ u) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_l1741_174143


namespace NUMINAMATH_CALUDE_germs_left_is_thirty_percent_l1741_174100

/-- The percentage of germs killed by spray A -/
def spray_a_kill_rate : ℝ := 50

/-- The percentage of germs killed by spray B -/
def spray_b_kill_rate : ℝ := 25

/-- The percentage of germs killed by both sprays -/
def overlap_kill_rate : ℝ := 5

/-- The percentage of germs left after using both sprays -/
def germs_left : ℝ := 100 - (spray_a_kill_rate + spray_b_kill_rate - overlap_kill_rate)

theorem germs_left_is_thirty_percent :
  germs_left = 30 := by sorry

end NUMINAMATH_CALUDE_germs_left_is_thirty_percent_l1741_174100


namespace NUMINAMATH_CALUDE_ant_ratio_is_two_to_one_l1741_174129

/-- The number of ants Abe finds -/
def abe_ants : ℕ := 4

/-- The number of ants Beth sees -/
def beth_ants : ℕ := (3 * abe_ants) / 2

/-- The number of ants Duke discovers -/
def duke_ants : ℕ := abe_ants / 2

/-- The total number of ants found by all four children -/
def total_ants : ℕ := 20

/-- The number of ants CeCe watches -/
def cece_ants : ℕ := total_ants - (abe_ants + beth_ants + duke_ants)

/-- The ratio of ants CeCe watches to ants Abe finds -/
def ant_ratio : ℚ := cece_ants / abe_ants

theorem ant_ratio_is_two_to_one : ant_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_ant_ratio_is_two_to_one_l1741_174129


namespace NUMINAMATH_CALUDE_sara_picked_six_pears_l1741_174124

/-- The number of pears picked by Tim -/
def tim_pears : ℕ := 5

/-- The total number of pears picked by Sara and Tim -/
def total_pears : ℕ := 11

/-- The number of pears picked by Sara -/
def sara_pears : ℕ := total_pears - tim_pears

theorem sara_picked_six_pears : sara_pears = 6 := by
  sorry

end NUMINAMATH_CALUDE_sara_picked_six_pears_l1741_174124


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1741_174139

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_mean : (a + b) / 2 = 1 / 2) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + y) / 2 = 1 / 2 → 1 / x + 1 / y ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1741_174139


namespace NUMINAMATH_CALUDE_floor_a_n_l1741_174170

/-- Sequence defined by the recurrence relation -/
def a : ℕ → ℚ
  | 0 => 1994
  | n + 1 => (a n)^2 / (a n + 1)

/-- Theorem stating that the floor of a_n is 1994 - n for 0 ≤ n ≤ 998 -/
theorem floor_a_n (n : ℕ) (h : n ≤ 998) : 
  ⌊a n⌋ = 1994 - n := by sorry

end NUMINAMATH_CALUDE_floor_a_n_l1741_174170


namespace NUMINAMATH_CALUDE_greatest_npmm_l1741_174126

/-- Represents a three-digit number with equal digits -/
def ThreeEqualDigits (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ n = d * 100 + d * 10 + d

/-- Represents a one-digit number -/
def OneDigit (n : ℕ) : Prop := n < 10 ∧ n > 0

/-- Represents a four-digit number -/
def FourDigits (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

/-- The main theorem -/
theorem greatest_npmm :
  ∀ MMM M NPMM : ℕ,
    ThreeEqualDigits MMM →
    OneDigit M →
    FourDigits NPMM →
    MMM * M = NPMM →
    NPMM ≤ 3996 :=
by
  sorry

#check greatest_npmm

end NUMINAMATH_CALUDE_greatest_npmm_l1741_174126


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1741_174121

/-- A geometric sequence with negative terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n < 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = -5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1741_174121


namespace NUMINAMATH_CALUDE_two_numbers_lcm_90_gcd_6_l1741_174174

theorem two_numbers_lcm_90_gcd_6 : ∃ (a b : ℕ+), 
  (¬(a ∣ b) ∧ ¬(b ∣ a)) ∧ 
  Nat.lcm a b = 90 ∧ 
  Nat.gcd a b = 6 ∧ 
  ({a, b} : Set ℕ+) = {18, 30} := by
sorry

end NUMINAMATH_CALUDE_two_numbers_lcm_90_gcd_6_l1741_174174


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l1741_174184

theorem rectangle_area_problem : ∃ (x y : ℝ), 
  (x + 3) * (y - 1) = x * y ∧ 
  (x - 3) * (y + 1.5) = x * y ∧ 
  x * y = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l1741_174184


namespace NUMINAMATH_CALUDE_not_prime_base_n_2022_l1741_174181

-- Define the base-n representation of 2022
def base_n_2022 (n : ℕ) : ℕ := 2 * n^3 + 2 * n + 2

-- Theorem statement
theorem not_prime_base_n_2022 (n : ℕ) (h : n ≥ 2) : ¬ Nat.Prime (base_n_2022 n) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_base_n_2022_l1741_174181


namespace NUMINAMATH_CALUDE_child_weight_l1741_174118

theorem child_weight (total_weight : ℝ) (weight_difference : ℝ) (dog_weight_ratio : ℝ)
  (hw_total : total_weight = 180)
  (hw_diff : weight_difference = 162)
  (hw_ratio : dog_weight_ratio = 0.3) :
  ∃ (father_weight child_weight dog_weight : ℝ),
    father_weight + child_weight + dog_weight = total_weight ∧
    father_weight + child_weight = weight_difference + dog_weight ∧
    dog_weight = dog_weight_ratio * child_weight ∧
    child_weight = 30 := by
  sorry

end NUMINAMATH_CALUDE_child_weight_l1741_174118


namespace NUMINAMATH_CALUDE_square_of_cube_of_third_smallest_prime_l1741_174165

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- State the theorem
theorem square_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 2 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_square_of_cube_of_third_smallest_prime_l1741_174165


namespace NUMINAMATH_CALUDE_soda_volume_difference_is_14_l1741_174146

/-- Calculates the difference in soda volume between Julio and Mateo -/
def soda_volume_difference : ℕ :=
  let julio_orange := 4
  let julio_grape := 7
  let mateo_orange := 1
  let mateo_grape := 3
  let liters_per_bottle := 2
  let julio_total := (julio_orange + julio_grape) * liters_per_bottle
  let mateo_total := (mateo_orange + mateo_grape) * liters_per_bottle
  julio_total - mateo_total

theorem soda_volume_difference_is_14 : soda_volume_difference = 14 := by
  sorry

end NUMINAMATH_CALUDE_soda_volume_difference_is_14_l1741_174146


namespace NUMINAMATH_CALUDE_shrimp_cost_per_pound_l1741_174162

/-- Calculates the cost per pound of shrimp for Wayne's shrimp cocktail appetizer. -/
theorem shrimp_cost_per_pound 
  (shrimp_per_guest : ℕ) 
  (num_guests : ℕ) 
  (shrimp_per_pound : ℕ) 
  (total_cost : ℚ) : 
  shrimp_per_guest = 5 → 
  num_guests = 40 → 
  shrimp_per_pound = 20 → 
  total_cost = 170 → 
  (total_cost / (shrimp_per_guest * num_guests / shrimp_per_pound : ℚ)) = 17 :=
by sorry

end NUMINAMATH_CALUDE_shrimp_cost_per_pound_l1741_174162


namespace NUMINAMATH_CALUDE_growth_rate_is_25_percent_l1741_174120

/-- The average monthly growth rate of new 5G physical base stations -/
def average_growth_rate : ℝ := 0.25

/-- The number of new 5G physical base stations opened in January -/
def january_stations : ℕ := 1600

/-- The number of new 5G physical base stations opened in March -/
def march_stations : ℕ := 2500

/-- Theorem stating that the average monthly growth rate is 25% -/
theorem growth_rate_is_25_percent :
  january_stations * (1 + average_growth_rate)^2 = march_stations := by
  sorry

#check growth_rate_is_25_percent

end NUMINAMATH_CALUDE_growth_rate_is_25_percent_l1741_174120


namespace NUMINAMATH_CALUDE_jakes_score_l1741_174167

theorem jakes_score (total_students : Nat) (avg_18 : ℚ) (avg_19 : ℚ) (avg_20 : ℚ) 
  (h1 : total_students = 20)
  (h2 : avg_18 = 75)
  (h3 : avg_19 = 76)
  (h4 : avg_20 = 77) :
  (total_students * avg_20 - (total_students - 1) * avg_19 : ℚ) = 96 := by
  sorry

end NUMINAMATH_CALUDE_jakes_score_l1741_174167


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_sqrt_three_only_irrational_l1741_174108

theorem sqrt_three_irrational :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ) :=
by
  sorry

-- Definitions for rational numbers in the problem
def zero_rational : ℚ := 0
def one_point_five_rational : ℚ := 3/2
def negative_two_rational : ℚ := -2

-- Theorem stating that √3 is the only irrational number among the given options
theorem sqrt_three_only_irrational :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ zero_rational = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ one_point_five_rational = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ negative_two_rational = (p : ℚ) / (q : ℚ)) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_sqrt_three_only_irrational_l1741_174108


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1741_174157

-- Define the polynomial
def p (x : ℝ) : ℝ := 10 * x^3 + 502 * x + 3010

-- Define the roots
theorem cubic_roots_sum (a b c : ℝ) (ha : p a = 0) (hb : p b = 0) (hc : p c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 903 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1741_174157


namespace NUMINAMATH_CALUDE_arithmetic_progression_cubic_coeff_conditions_l1741_174131

/-- A cubic polynomial with coefficients a, b, c whose roots form an arithmetic progression -/
structure ArithmeticProgressionCubic where
  a : ℝ
  b : ℝ
  c : ℝ
  roots_in_ap : ∃ (r₁ r₂ r₃ : ℝ), r₁ < r₂ ∧ r₂ < r₃ ∧
    r₂ - r₁ = r₃ - r₂ ∧
    r₁ + r₂ + r₃ = -a ∧
    r₁ * r₂ + r₁ * r₃ + r₂ * r₃ = b ∧
    r₁ * r₂ * r₃ = -c

/-- The coefficients of a cubic polynomial with roots in arithmetic progression satisfy specific conditions -/
theorem arithmetic_progression_cubic_coeff_conditions (p : ArithmeticProgressionCubic) :
  27 * p.c = 3 * p.a * p.b - 2 * p.a^3 ∧ 3 * p.b ≤ p.a^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cubic_coeff_conditions_l1741_174131


namespace NUMINAMATH_CALUDE_base_8_to_base_10_l1741_174172

theorem base_8_to_base_10 : 
  (3 * 8^3 + 5 * 8^2 + 2 * 8^1 + 6 * 8^0 : ℕ) = 1878 := by
  sorry

end NUMINAMATH_CALUDE_base_8_to_base_10_l1741_174172


namespace NUMINAMATH_CALUDE_parallel_transitive_l1741_174103

-- Define the type for lines in space
structure Line3D where
  -- We don't need to specify the internal structure of a line
  -- as we're only concerned with their relationships

-- Define the parallelism relation
def parallel (l1 l2 : Line3D) : Prop :=
  sorry  -- The actual definition is not important for this statement

-- State the theorem
theorem parallel_transitive (a b c : Line3D) 
  (hab : parallel a b) (hbc : parallel b c) : 
  parallel a c :=
sorry

end NUMINAMATH_CALUDE_parallel_transitive_l1741_174103


namespace NUMINAMATH_CALUDE_cube_painting_theorem_l1741_174117

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Represents the number of cubelets with exactly one painted face for each color -/
def single_color_cubelets (c : Cube n) : ℕ :=
  4 * 4 * 2

/-- The total number of cubelets with exactly one painted face -/
def total_single_color_cubelets (c : Cube n) : ℕ :=
  3 * single_color_cubelets c

theorem cube_painting_theorem (c : Cube 6) :
  total_single_color_cubelets c = 96 := by
  sorry


end NUMINAMATH_CALUDE_cube_painting_theorem_l1741_174117


namespace NUMINAMATH_CALUDE_quadratic_function_value_l1741_174154

/-- Given a quadratic function f(x) = ax^2 + bx + 2 with f(1) = 4 and f(2) = 10, prove that f(3) = 20 -/
theorem quadratic_function_value (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + 2)
  (h2 : f 1 = 4)
  (h3 : f 2 = 10) :
  f 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l1741_174154


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_absolute_value_location_l1741_174163

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

theorem absolute_value_location (a : ℝ) : |a| = 0 ∨ |a| > 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_absolute_value_location_l1741_174163


namespace NUMINAMATH_CALUDE_inequality_holds_l1741_174191

theorem inequality_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1741_174191


namespace NUMINAMATH_CALUDE_no_geometric_sequence_cosines_l1741_174182

theorem no_geometric_sequence_cosines :
  ¬ ∃ a : ℝ, 0 < a ∧ a < 2 * π ∧
    ∃ r : ℝ, (Real.cos (2 * a) = r * Real.cos a) ∧
             (Real.cos (3 * a) = r * Real.cos (2 * a)) :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_sequence_cosines_l1741_174182


namespace NUMINAMATH_CALUDE_stamps_per_page_l1741_174106

theorem stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 924) 
  (h2 : book2 = 1386) 
  (h3 : book3 = 1848) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 462 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_page_l1741_174106


namespace NUMINAMATH_CALUDE_green_apples_count_l1741_174199

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := sorry

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 6

/-- The number of students who took fruit -/
def students_taking_fruit : ℕ := 5

/-- The number of extra apples left over -/
def extra_apples : ℕ := 16

/-- Theorem stating that the number of green apples ordered is 15 -/
theorem green_apples_count : green_apples = 15 := by
  sorry

end NUMINAMATH_CALUDE_green_apples_count_l1741_174199


namespace NUMINAMATH_CALUDE_solution_product_l1741_174105

theorem solution_product (p q : ℝ) : 
  (p - 4) * (3 * p + 11) = p^2 - 19 * p + 72 →
  (q - 4) * (3 * q + 11) = q^2 - 19 * q + 72 →
  p ≠ q →
  (p + 4) * (q + 4) = -78 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_product_l1741_174105


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1741_174148

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9 * x^2 + 1 / x^6 ≥ 6 * Real.sqrt 3 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, x > 0 ∧ 9 * x^2 + 1 / x^6 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1741_174148


namespace NUMINAMATH_CALUDE_chess_pieces_remaining_l1741_174198

theorem chess_pieces_remaining (initial_pieces : ℕ) (scarlett_lost : ℕ) (hannah_lost : ℕ)
  (h1 : initial_pieces = 32)
  (h2 : scarlett_lost = 6)
  (h3 : hannah_lost = 8) :
  initial_pieces - (scarlett_lost + hannah_lost) = 18 :=
by sorry

end NUMINAMATH_CALUDE_chess_pieces_remaining_l1741_174198


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1741_174156

theorem imaginary_part_of_z (z : ℂ) : z = (Complex.I - 3) / (Complex.I + 1) → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1741_174156


namespace NUMINAMATH_CALUDE_line_equation_and_distance_l1741_174135

-- Define the point P
def P : ℝ × ℝ := (-1, 4)

-- Define line l₂
def l₂ (x y : ℝ) : Prop := 2 * x - y + 5 = 0

-- Define line l₁
def l₁ (x y : ℝ) : Prop := 2 * x - y + 6 = 0

-- Define line l₃
def l₃ (x y m : ℝ) : Prop := 4 * x - 2 * y + m = 0

-- State the theorem
theorem line_equation_and_distance (m : ℝ) : 
  (∀ x y, l₁ x y ↔ l₂ (x + 1/2) (y + 1/2)) ∧ -- l₁ is parallel to l₂
  l₁ P.1 P.2 ∧ -- P lies on l₁
  (∃ d, d = 2 * Real.sqrt 5 ∧ 
   d = |m - 12| / Real.sqrt (4^2 + (-2)^2)) → -- Distance between l₁ and l₃
  (m = -8 ∨ m = 32) := by
sorry

end NUMINAMATH_CALUDE_line_equation_and_distance_l1741_174135


namespace NUMINAMATH_CALUDE_solve_percentage_equation_l1741_174195

theorem solve_percentage_equation (x : ℝ) : 0.60 * x = (1 / 3) * x + 110 → x = 412.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_percentage_equation_l1741_174195


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1741_174185

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x^2 + 2*x + 1) / (x + 1) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1741_174185


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l1741_174164

/-- A line in 2D space represented by ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if two lines are perpendicular --/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The given line l --/
def l : Line := { a := 2, b := 1, c := 10 }

/-- The point through which l' passes --/
def p : Point := { x := -10, y := 0 }

/-- The theorem to prove --/
theorem intersection_point_coordinates :
  ∃ (l' : Line),
    (p.onLine l') ∧
    (l.perpendicular l') ∧
    (∃ (q : Point), q.onLine l ∧ q.onLine l' ∧ q.x = 2 ∧ q.y = 6) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l1741_174164


namespace NUMINAMATH_CALUDE_fiftieth_term_of_specific_sequence_l1741_174161

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem fiftieth_term_of_specific_sequence :
  arithmetic_sequence 3 2 50 = 101 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_specific_sequence_l1741_174161


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l1741_174123

theorem complex_multiplication_result : 
  (2 + 2 * Complex.I) * (1 - 2 * Complex.I) = 6 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l1741_174123


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l1741_174145

theorem salt_solution_mixture (x : ℝ) : 
  (1 : ℝ) + x > 0 →  -- Ensure total volume is positive
  0.60 * x = 0.10 * ((1 : ℝ) + x) → 
  x = 0.2 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l1741_174145


namespace NUMINAMATH_CALUDE_only_students_far_from_school_not_set_l1741_174102

-- Define the groups of objects
def right_angled_triangles : Set (Set ℝ) := sorry
def points_on_unit_circle : Set (ℝ × ℝ) := sorry
def students_far_from_school : Set String := sorry
def homeroom_teachers : Set String := sorry

-- Define a predicate for well-defined sets
def is_well_defined_set (S : Set α) : Prop := sorry

-- Theorem statement
theorem only_students_far_from_school_not_set :
  is_well_defined_set right_angled_triangles ∧
  is_well_defined_set points_on_unit_circle ∧
  ¬ is_well_defined_set students_far_from_school ∧
  is_well_defined_set homeroom_teachers :=
sorry

end NUMINAMATH_CALUDE_only_students_far_from_school_not_set_l1741_174102


namespace NUMINAMATH_CALUDE_money_difference_is_13_96_l1741_174175

def derek_initial : ℚ := 40
def derek_expenses : List ℚ := [14, 11, 5, 8]
def derek_discount_rate : ℚ := 0.1

def dave_initial : ℚ := 50
def dave_expenses : List ℚ := [7, 12, 9]
def dave_tax_rate : ℚ := 0.08

def calculate_remaining_money (initial : ℚ) (expenses : List ℚ) (rate : ℚ) (is_discount : Bool) : ℚ :=
  let total_expenses := expenses.sum
  let adjustment := total_expenses * rate
  if is_discount then
    initial - (total_expenses - adjustment)
  else
    initial - (total_expenses + adjustment)

theorem money_difference_is_13_96 :
  let derek_remaining := calculate_remaining_money derek_initial derek_expenses derek_discount_rate true
  let dave_remaining := calculate_remaining_money dave_initial dave_expenses dave_tax_rate false
  dave_remaining - derek_remaining = 13.96 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_is_13_96_l1741_174175


namespace NUMINAMATH_CALUDE_friends_in_all_activities_l1741_174190

theorem friends_in_all_activities (movie : ℕ) (picnic : ℕ) (games : ℕ) 
  (movie_and_picnic : ℕ) (movie_and_games : ℕ) (picnic_and_games : ℕ) 
  (total : ℕ) : 
  movie = 10 → 
  picnic = 20 → 
  games = 5 → 
  movie_and_picnic = 4 → 
  movie_and_games = 2 → 
  picnic_and_games = 0 → 
  total = 31 → 
  ∃ (all_three : ℕ), 
    all_three = 2 ∧ 
    total = movie + picnic + games - movie_and_picnic - movie_and_games - picnic_and_games + all_three :=
by sorry

end NUMINAMATH_CALUDE_friends_in_all_activities_l1741_174190


namespace NUMINAMATH_CALUDE_power_of_product_l1741_174125

theorem power_of_product (a : ℝ) : (3 * a)^2 = 9 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1741_174125


namespace NUMINAMATH_CALUDE_milk_discount_l1741_174116

/-- Calculates the discount on milk given grocery prices and remaining money --/
theorem milk_discount (initial_money : ℝ) (milk_price bread_price detergent_price banana_price_per_pound : ℝ)
  (banana_pounds : ℝ) (detergent_coupon : ℝ) (money_left : ℝ) :
  initial_money = 20 ∧
  milk_price = 4 ∧
  bread_price = 3.5 ∧
  detergent_price = 10.25 ∧
  banana_price_per_pound = 0.75 ∧
  banana_pounds = 2 ∧
  detergent_coupon = 1.25 ∧
  money_left = 4 →
  initial_money - (bread_price + (detergent_price - detergent_coupon) + 
    (banana_price_per_pound * banana_pounds) + money_left) = 2 :=
by sorry

end NUMINAMATH_CALUDE_milk_discount_l1741_174116


namespace NUMINAMATH_CALUDE_paco_cookies_theorem_l1741_174183

/-- Calculates the number of sweet cookies Paco ate given the initial quantities and eating conditions -/
def sweet_cookies_eaten (initial_sweet : ℕ) (initial_salty : ℕ) (sweet_salty_difference : ℕ) : ℕ :=
  initial_salty + sweet_salty_difference

theorem paco_cookies_theorem (initial_sweet initial_salty sweet_salty_difference : ℕ) 
  (h1 : initial_sweet = 39)
  (h2 : initial_salty = 6)
  (h3 : sweet_salty_difference = 9) :
  sweet_cookies_eaten initial_sweet initial_salty sweet_salty_difference = 15 := by
  sorry

#eval sweet_cookies_eaten 39 6 9

end NUMINAMATH_CALUDE_paco_cookies_theorem_l1741_174183


namespace NUMINAMATH_CALUDE_f_eq_g_shifted_l1741_174188

/-- Given two functions f and g defined as follows:
    f(x) = sin(x + π/2)
    g(x) = cos(x - π/2)
    Prove that f(x) = g(x + π/2) for all real x. -/
theorem f_eq_g_shifted (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin (x + π/2)
  let g : ℝ → ℝ := λ x => Real.cos (x - π/2)
  f x = g (x + π/2) := by
  sorry

end NUMINAMATH_CALUDE_f_eq_g_shifted_l1741_174188


namespace NUMINAMATH_CALUDE_min_voters_for_tall_to_win_l1741_174197

/-- Represents the voting structure and rules of the giraffe beauty contest -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat
  (total_voters_eq : total_voters = num_districts * precincts_per_district * voters_per_precinct)

/-- Calculates the minimum number of voters required to win the contest -/
def min_voters_to_win (contest : GiraffeContest) : Nat :=
  let min_districts_to_win := contest.num_districts / 2 + 1
  let min_precincts_to_win := contest.precincts_per_district / 2 + 1
  let min_votes_per_precinct := contest.voters_per_precinct / 2 + 1
  min_districts_to_win * min_precincts_to_win * min_votes_per_precinct

/-- The theorem stating the minimum number of voters required for Tall to win -/
theorem min_voters_for_tall_to_win (contest : GiraffeContest)
  (h1 : contest.total_voters = 135)
  (h2 : contest.num_districts = 5)
  (h3 : contest.precincts_per_district = 9)
  (h4 : contest.voters_per_precinct = 3) :
  min_voters_to_win contest = 30 := by
  sorry

#eval min_voters_to_win { total_voters := 135, num_districts := 5, precincts_per_district := 9, voters_per_precinct := 3, total_voters_eq := rfl }

end NUMINAMATH_CALUDE_min_voters_for_tall_to_win_l1741_174197


namespace NUMINAMATH_CALUDE_days_worked_l1741_174193

/-- Given a person works 8 hours each day and a total of 32 hours, prove that the number of days worked is 4. -/
theorem days_worked (hours_per_day : ℕ) (total_hours : ℕ) (h1 : hours_per_day = 8) (h2 : total_hours = 32) :
  total_hours / hours_per_day = 4 := by
  sorry

end NUMINAMATH_CALUDE_days_worked_l1741_174193


namespace NUMINAMATH_CALUDE_sum_of_digits_10_95_minus_195_l1741_174180

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem: sum of digits of 10^95 - 195 is 841 -/
theorem sum_of_digits_10_95_minus_195 : sum_of_digits (10^95 - 195) = 841 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_10_95_minus_195_l1741_174180


namespace NUMINAMATH_CALUDE_rectangle_area_preservation_l1741_174171

theorem rectangle_area_preservation (original_length original_width : ℝ) 
  (h_length : original_length = 280)
  (h_width : original_width = 80)
  (length_increase_percent : ℝ) 
  (h_increase : length_increase_percent = 60) : 
  let new_length := original_length * (1 + length_increase_percent / 100)
  let new_width := (original_length * original_width) / new_length
  let width_decrease_percent := (original_width - new_width) / original_width * 100
  width_decrease_percent = 37.5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_preservation_l1741_174171


namespace NUMINAMATH_CALUDE_tina_july_savings_l1741_174138

/-- Represents Tina's savings and spending --/
structure TinaSavings where
  june : ℕ
  july : ℕ
  august : ℕ
  books_spent : ℕ
  shoes_spent : ℕ
  remaining : ℕ

/-- Theorem stating that Tina saved $14 in July --/
theorem tina_july_savings (s : TinaSavings) 
  (h1 : s.june = 27)
  (h2 : s.august = 21)
  (h3 : s.books_spent = 5)
  (h4 : s.shoes_spent = 17)
  (h5 : s.remaining = 40)
  (h6 : s.june + s.july + s.august = s.books_spent + s.shoes_spent + s.remaining) :
  s.july = 14 := by
  sorry


end NUMINAMATH_CALUDE_tina_july_savings_l1741_174138


namespace NUMINAMATH_CALUDE_bean_jar_count_bean_jar_count_proof_l1741_174149

theorem bean_jar_count : ℕ → Prop :=
  fun total_beans =>
    let red_beans := total_beans / 4
    let remaining_after_red := total_beans - red_beans
    let white_beans := remaining_after_red / 3
    let remaining_after_white := remaining_after_red - white_beans
    let green_beans := remaining_after_white / 2
    (green_beans = 143) → (total_beans = 572)

theorem bean_jar_count_proof : bean_jar_count 572 := by
  sorry

end NUMINAMATH_CALUDE_bean_jar_count_bean_jar_count_proof_l1741_174149


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l1741_174173

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- State the theorem
theorem f_monotone_decreasing :
  ∀ f : ℝ → ℝ, (∀ x, deriv f x = f' x) →
  ∀ x ∈ Set.Ioo 0 2, deriv f (x + 1) < 0 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l1741_174173


namespace NUMINAMATH_CALUDE_scatter_plot_correlation_l1741_174196

/-- Represents a scatter plot of two variables -/
structure ScatterPlot where
  bottomLeft : Bool
  topRight : Bool

/-- Defines positive correlation between two variables -/
def positivelyCorrelated (x y : ℝ → ℝ) : Prop :=
  ∀ a b, a < b → x a < x b ∧ y a < y b

/-- Theorem: If a scatter plot goes from bottom left to top right, 
    the variables are positively correlated -/
theorem scatter_plot_correlation (plot : ScatterPlot) (x y : ℝ → ℝ) :
  plot.bottomLeft ∧ plot.topRight → positivelyCorrelated x y := by
  sorry


end NUMINAMATH_CALUDE_scatter_plot_correlation_l1741_174196


namespace NUMINAMATH_CALUDE_least_integer_with_given_remainders_l1741_174178

theorem least_integer_with_given_remainders : ∃ x : ℕ, 
  x > 0 ∧
  x % 3 = 2 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 7 = 6 ∧
  (∀ y : ℕ, y > 0 ∧ y % 3 = 2 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 7 = 6 → x ≤ y) ∧
  x = 419 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_with_given_remainders_l1741_174178


namespace NUMINAMATH_CALUDE_quadratic_coefficient_relation_quadratic_coefficient_relation_alt_l1741_174187

/-- Given two quadratic equations, prove the relationship between their coefficients -/
theorem quadratic_coefficient_relation (a b c d r s : ℝ) : 
  (r + s = -a ∧ r * s = b) →  -- roots of first equation
  (r^2 + s^2 = -c ∧ r^2 * s^2 = d) →  -- roots of second equation
  r * s = 2 * b →  -- additional condition
  c = -a^2 + 2*b ∧ d = b^2 := by
sorry

/-- Alternative formulation using polynomial roots -/
theorem quadratic_coefficient_relation_alt (a b c d : ℝ) :
  (∃ r s : ℝ, (r + s = -a ∧ r * s = b) ∧ 
              (r^2 + s^2 = -c ∧ r^2 * s^2 = d) ∧
              r * s = 2 * b) →
  c = -a^2 + 2*b ∧ d = b^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_relation_quadratic_coefficient_relation_alt_l1741_174187


namespace NUMINAMATH_CALUDE_brocard_angle_inequalities_l1741_174155

theorem brocard_angle_inequalities (α β γ φ : Real) 
  (triangle_angles : α + β + γ = π) 
  (brocard_angle : 0 < φ ∧ φ ≤ π/6)
  (brocard_identity : Real.sin (α - φ) * Real.sin (β - φ) * Real.sin (γ - φ) = Real.sin φ ^ 3) :
  φ ^ 3 ≤ (α - φ) * (β - φ) * (γ - φ) ∧ 8 * φ ^ 3 ≤ α * β * γ := by
  sorry

end NUMINAMATH_CALUDE_brocard_angle_inequalities_l1741_174155


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l1741_174110

theorem sine_cosine_inequality (x y : Real) (h1 : 0 ≤ x) (h2 : x ≤ y) (h3 : y ≤ Real.pi / 2) :
  (Real.sin (x / 2))^2 * Real.cos y ≤ 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l1741_174110


namespace NUMINAMATH_CALUDE_counterexample_square_inequality_l1741_174176

theorem counterexample_square_inequality : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_square_inequality_l1741_174176


namespace NUMINAMATH_CALUDE_quadrilateral_area_main_theorem_l1741_174136

/-- A line with slope -1 intersecting positive x and y axes -/
structure Line1 where
  slope : ℝ
  xIntercept : ℝ
  yIntercept : ℝ

/-- A line passing through (10,0) and intersecting y-axis -/
structure Line2 where
  xIntercept : ℝ
  yIntercept : ℝ

/-- The intersection point of the two lines -/
def intersectionPoint : ℝ × ℝ := (5, 5)

/-- The theorem stating the area of the quadrilateral -/
theorem quadrilateral_area 
  (l1 : Line1) 
  (l2 : Line2) : ℝ :=
  let o := (0, 0)
  let b := (0, l1.yIntercept)
  let e := intersectionPoint
  let c := (l2.xIntercept, 0)
  87.5

/-- Main theorem to prove -/
theorem main_theorem 
  (l1 : Line1) 
  (l2 : Line2) 
  (h1 : l1.slope = -1)
  (h2 : l1.xIntercept > 0)
  (h3 : l1.yIntercept > 0)
  (h4 : l2.xIntercept = 10)
  (h5 : l2.yIntercept > 0) :
  quadrilateral_area l1 l2 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_main_theorem_l1741_174136


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1741_174119

/-- Given a geometric sequence {a_n} where a₄ + a₆ = 3, prove that a₅(a₃ + 2a₅ + a₇) = 9 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- a_n is a geometric sequence with common ratio q
  a 4 + a 6 = 3 →               -- given condition
  a 5 * (a 3 + 2 * a 5 + a 7) = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1741_174119


namespace NUMINAMATH_CALUDE_correct_number_of_students_l1741_174168

/-- The number of students in a class preparing for a field trip --/
def number_of_students : ℕ := 30

/-- The amount each student contributes per Friday in dollars --/
def contribution_per_friday : ℕ := 2

/-- The number of Fridays in the collection period --/
def number_of_fridays : ℕ := 8

/-- The total amount collected for the trip in dollars --/
def total_amount : ℕ := 480

/-- Theorem stating that the number of students is correct given the conditions --/
theorem correct_number_of_students :
  number_of_students * contribution_per_friday * number_of_fridays = total_amount :=
sorry

end NUMINAMATH_CALUDE_correct_number_of_students_l1741_174168


namespace NUMINAMATH_CALUDE_greatest_difference_l1741_174109

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem greatest_difference (x y : ℕ) 
  (hx1 : 1 < x) (hx2 : x < 20) 
  (hy1 : 20 < y) (hy2 : y < 50) 
  (hxp : is_prime x) 
  (hym : ∃ k : ℕ, y = 7 * k) : 
  (∀ a b : ℕ, 1 < a → a < 20 → 20 < b → b < 50 → is_prime a → (∃ m : ℕ, b = 7 * m) → b - a ≤ y - x) ∧ y - x = 30 := by
  sorry

end NUMINAMATH_CALUDE_greatest_difference_l1741_174109


namespace NUMINAMATH_CALUDE_total_cupcakes_eq_768_l1741_174128

/-- The number of cupcakes ordered for each event -/
def cupcakes_per_event : ℝ := 96.0

/-- The number of different children's events -/
def number_of_events : ℝ := 8.0

/-- The total number of cupcakes needed -/
def total_cupcakes : ℝ := cupcakes_per_event * number_of_events

/-- Theorem stating that the total number of cupcakes is 768.0 -/
theorem total_cupcakes_eq_768 : total_cupcakes = 768.0 := by
  sorry

end NUMINAMATH_CALUDE_total_cupcakes_eq_768_l1741_174128
