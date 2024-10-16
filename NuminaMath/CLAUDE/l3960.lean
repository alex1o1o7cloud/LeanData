import Mathlib

namespace NUMINAMATH_CALUDE_at_most_one_greater_than_one_l3960_396067

theorem at_most_one_greater_than_one (x y : ℝ) (h : x + y < 2) :
  ¬(x > 1 ∧ y > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_greater_than_one_l3960_396067


namespace NUMINAMATH_CALUDE_f_has_one_zero_in_interval_min_value_of_f_plus_g_exists_x₂_for_all_x₁_l3960_396016

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x - Real.cos x
noncomputable def g (x : ℝ) : ℝ := x * Real.cos x - Real.sqrt 2 * Real.exp x

def interval : Set ℝ := Set.Icc 0 (Real.pi / 2)

theorem f_has_one_zero_in_interval :
  ∃! x, x ∈ Set.Ioo 0 (Real.pi / 2) ∧ f x = 0 :=
sorry

theorem min_value_of_f_plus_g :
  ∀ x₁ x₂, x₁ ∈ interval → x₂ ∈ interval →
    f x₁ + g x₂ ≥ -Real.sqrt 2 - 1 :=
sorry

theorem exists_x₂_for_all_x₁ (m : ℝ) :
  (∀ x₁ ∈ interval, ∃ x₂ ∈ interval, f x₁ + g x₂ ≥ m) ↔ m ≤ -Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_f_has_one_zero_in_interval_min_value_of_f_plus_g_exists_x₂_for_all_x₁_l3960_396016


namespace NUMINAMATH_CALUDE_tangent_sum_product_l3960_396020

theorem tangent_sum_product (a b c : Real) (h1 : a = 117 * π / 180)
                                           (h2 : b = 118 * π / 180)
                                           (h3 : c = 125 * π / 180)
                                           (h4 : a + b + c = 2 * π) :
  Real.tan a * Real.tan b * Real.tan c = Real.tan a + Real.tan b + Real.tan c := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_product_l3960_396020


namespace NUMINAMATH_CALUDE_soda_difference_l3960_396066

def regular_soda : ℕ := 67
def diet_soda : ℕ := 9

theorem soda_difference : regular_soda - diet_soda = 58 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_l3960_396066


namespace NUMINAMATH_CALUDE_stating_rhombus_solutions_count_l3960_396003

/-- Represents the number of solutions for inscribing a rhombus in a square and circumscribing it around a circle -/
inductive NumSolutions
  | two
  | one
  | zero

/-- 
  Given a square and a circle with the same center, determines the number of possible rhombuses 
  that can be inscribed in the square and circumscribed around the circle.
-/
def numRhombusSolutions (squareSide : ℝ) (circleRadius : ℝ) : NumSolutions :=
  sorry

/-- 
  Theorem stating that the number of rhombus solutions is either 2, 1, or 0
-/
theorem rhombus_solutions_count (squareSide : ℝ) (circleRadius : ℝ) :
  ∃ (n : NumSolutions), numRhombusSolutions squareSide circleRadius = n :=
  sorry

end NUMINAMATH_CALUDE_stating_rhombus_solutions_count_l3960_396003


namespace NUMINAMATH_CALUDE_ap_sum_70_l3960_396085

def arithmetic_progression (a d : ℚ) (n : ℕ) : ℚ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem ap_sum_70 (a d : ℚ) :
  arithmetic_progression a d 20 = 150 →
  arithmetic_progression a d 50 = 20 →
  arithmetic_progression a d 70 = -910/3 := by
  sorry

end NUMINAMATH_CALUDE_ap_sum_70_l3960_396085


namespace NUMINAMATH_CALUDE_birthday_gift_savings_is_86_l3960_396027

/-- The amount of money Liam and Claire save for their mother's birthday gift -/
def birthday_gift_savings (liam_oranges : ℕ) (liam_price : ℚ) (claire_oranges : ℕ) (claire_price : ℚ) : ℚ :=
  (liam_oranges / 2 : ℚ) * liam_price + claire_oranges * claire_price

/-- Theorem stating that Liam and Claire save $86 for their mother's birthday gift -/
theorem birthday_gift_savings_is_86 :
  birthday_gift_savings 40 (5/2) 30 (6/5) = 86 := by
  sorry

end NUMINAMATH_CALUDE_birthday_gift_savings_is_86_l3960_396027


namespace NUMINAMATH_CALUDE_license_plate_count_l3960_396004

/-- The number of consonants excluding 'Y' -/
def num_consonants_no_y : ℕ := 19

/-- The number of vowels including 'Y' -/
def num_vowels : ℕ := 6

/-- The number of consonants including 'Y' -/
def num_consonants_with_y : ℕ := 21

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The total number of valid license plates -/
def total_license_plates : ℕ := num_consonants_no_y * num_vowels * num_consonants_with_y * num_even_digits

theorem license_plate_count : total_license_plates = 11970 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3960_396004


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3960_396019

/-- Given a cube with surface area 6x^2, where x is the length of one side,
    prove that the volume of the cube is x^3. -/
theorem cube_volume_from_surface_area (x : ℝ) (h : x > 0) :
  let surface_area := 6 * x^2
  let side_length := x
  let volume := side_length^3
  surface_area = 6 * side_length^2 → volume = x^3 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3960_396019


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l3960_396068

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_perpendicular_lines 
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n) 
  (h_distinct_planes : α ≠ β) 
  (h_m_perp_n : perpendicular_lines m n) 
  (h_m_perp_α : perpendicular_line_plane m α) 
  (h_n_perp_β : perpendicular_line_plane n β) : 
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l3960_396068


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3960_396082

def N : ℕ := 64 * 45 * 91 * 49

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 126 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3960_396082


namespace NUMINAMATH_CALUDE_prime_square_plus_two_prime_l3960_396092

theorem prime_square_plus_two_prime (P : ℕ) : 
  Nat.Prime P → Nat.Prime (P^2 + 2) → P^4 + 1921 = 2002 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_plus_two_prime_l3960_396092


namespace NUMINAMATH_CALUDE_no_valid_pair_l3960_396057

def s : Finset ℤ := {2, 3, 4, 5, 9, 12, 18}
def b : Finset ℤ := {4, 5, 6, 7, 8, 11, 14, 19}

theorem no_valid_pair : ¬∃ (x y : ℤ), 
  x ∈ s ∧ y ∈ b ∧ 
  x % 3 = 2 ∧ y % 4 = 1 ∧ 
  (x % 2 = 0 ∧ y % 2 = 1 ∨ x % 2 = 1 ∧ y % 2 = 0) ∧ 
  x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_no_valid_pair_l3960_396057


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l3960_396064

-- Define the repeating decimals
def repeating_decimal_1 : ℚ := 8/9
def repeating_decimal_2 : ℚ := 15/11

-- State the theorem
theorem repeating_decimal_fraction : 
  repeating_decimal_1 / repeating_decimal_2 = 88 / 135 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l3960_396064


namespace NUMINAMATH_CALUDE_factorial_sum_of_powers_of_two_l3960_396063

theorem factorial_sum_of_powers_of_two (n : ℕ) :
  (∃ a b : ℕ, n.factorial = 2^a + 2^b) ↔ n = 3 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_of_powers_of_two_l3960_396063


namespace NUMINAMATH_CALUDE_unique_solution_system_l3960_396041

theorem unique_solution_system :
  ∃! (x y z : ℝ),
    x^2 - 22*y - 69*z + 703 = 0 ∧
    y^2 + 23*x + 23*z - 1473 = 0 ∧
    z^2 - 63*x + 66*y + 2183 = 0 ∧
    x = 20 ∧ y = -22 ∧ z = 23 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3960_396041


namespace NUMINAMATH_CALUDE_norm_scalar_multiple_l3960_396048

variable (n : ℕ)
variable (v : Fin n → ℝ)

theorem norm_scalar_multiple
  (h : ‖v‖ = 6) :
  ‖(5 : ℝ) • v‖ = 30 := by
sorry

end NUMINAMATH_CALUDE_norm_scalar_multiple_l3960_396048


namespace NUMINAMATH_CALUDE_basketball_team_starters_count_l3960_396001

def total_players : ℕ := 18
def num_triplets : ℕ := 3
def num_twins : ℕ := 2
def num_starters : ℕ := 7
def triplets_in_lineup : ℕ := 2
def twins_in_lineup : ℕ := 1

def remaining_players : ℕ := total_players - num_triplets - num_twins

theorem basketball_team_starters_count :
  (Nat.choose num_triplets triplets_in_lineup) *
  (Nat.choose num_twins twins_in_lineup) *
  (Nat.choose remaining_players (num_starters - triplets_in_lineup - twins_in_lineup)) = 4290 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_starters_count_l3960_396001


namespace NUMINAMATH_CALUDE_consecutive_integer_roots_l3960_396034

theorem consecutive_integer_roots (p q : ℤ) : 
  (∃ x y : ℤ, x^2 - p*x + q = 0 ∧ y^2 - p*y + q = 0 ∧ y = x + 1) →
  Prime q →
  (p = 3 ∨ p = -3) ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integer_roots_l3960_396034


namespace NUMINAMATH_CALUDE_not_divisible_by_two_2013_l3960_396054

-- Define a property for odd numbers
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Define what it means for a number to be not divisible by 2
def NotDivisibleByTwo (n : ℤ) : Prop := ¬ (∃ k : ℤ, n = 2 * k)

-- State the theorem
theorem not_divisible_by_two_2013 :
  (∀ n : ℤ, IsOdd n → NotDivisibleByTwo n) →
  IsOdd 2013 →
  NotDivisibleByTwo 2013 := by sorry

end NUMINAMATH_CALUDE_not_divisible_by_two_2013_l3960_396054


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_of_squares_l3960_396017

def is_consecutive_odd (a b c : ℤ) : Prop :=
  b = a + 2 ∧ c = b + 2 ∧ ∃ k : ℤ, a = 2 * k + 1

theorem consecutive_odd_sum_of_squares (a b c : ℤ) :
  is_consecutive_odd a b c → a^2 + b^2 + c^2 = 251 →
  ((a = 7 ∧ b = 9 ∧ c = 11) ∨ (a = -11 ∧ b = -9 ∧ c = -7)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_of_squares_l3960_396017


namespace NUMINAMATH_CALUDE_cylindrical_to_rectangular_l3960_396008

theorem cylindrical_to_rectangular :
  let r : ℝ := 6
  let θ : ℝ := 5 * π / 3
  let z : ℝ := 7
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y, z) = (3, 3 * Real.sqrt 3, 7) := by sorry

end NUMINAMATH_CALUDE_cylindrical_to_rectangular_l3960_396008


namespace NUMINAMATH_CALUDE_unique_zip_code_l3960_396049

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_palindrome (a b c : ℕ) : Prop := a = c

def is_consecutive (a b : ℕ) : Prop := b = a + 1

theorem unique_zip_code (a b c d e : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a + b + c + d + e = 20 ∧
  is_consecutive a b ∧
  c ≠ 0 ∧ c ≠ a ∧ c ≠ b ∧
  is_palindrome a b c ∧
  d = 2 * a ∧
  d + e = 13 ∧
  is_prime (a * 10000 + b * 1000 + c * 100 + d * 10 + e) →
  a * 10000 + b * 1000 + c * 100 + d * 10 + e = 34367 :=
by sorry

end NUMINAMATH_CALUDE_unique_zip_code_l3960_396049


namespace NUMINAMATH_CALUDE_extreme_points_sum_lower_bound_l3960_396060

theorem extreme_points_sum_lower_bound 
  (a : ℝ) 
  (ha : 0 < a ∧ a < 1/8) 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = x - a * x^2 - Real.log x) 
  (x₁ x₂ : ℝ) 
  (hx : x₁ + x₂ = 1 / (2*a) ∧ x₁ * x₂ = 1 / (2*a)) :
  f x₁ + f x₂ > 3 - 2 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_sum_lower_bound_l3960_396060


namespace NUMINAMATH_CALUDE_sum_of_digits_of_special_number_l3960_396089

/-- A number consisting of n digits all equal to 1 -/
def allOnes (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sumOfDigits (n / 10)

/-- The main theorem -/
theorem sum_of_digits_of_special_number :
  let L := allOnes 2022
  sumOfDigits (9 * L^2 + 2 * L) = 4044 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_special_number_l3960_396089


namespace NUMINAMATH_CALUDE_equal_numbers_l3960_396053

theorem equal_numbers (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ)
  (h1 : ∀ i : Fin 2011, x i + x (i + 1) = 2 * x' i)
  (h2 : ∃ σ : Equiv.Perm (Fin 2011), ∀ i, x' i = x (σ i)) :
  ∀ i j : Fin 2011, x i = x j :=
sorry

end NUMINAMATH_CALUDE_equal_numbers_l3960_396053


namespace NUMINAMATH_CALUDE_OL_length_OL_angle_tangent_intersection_product_l3960_396040

/-- Ellipse Γ: x²/4 + y² = 1 -/
def Γ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Point L in the third quadrant -/
def L : ℝ × ℝ := (-3, -3)

/-- OL = 3√2 -/
theorem OL_length : Real.sqrt (L.1^2 + L.2^2) = 3 * Real.sqrt 2 := by sorry

/-- Angle between negative x-axis and OL is π/4 -/
theorem OL_angle : Real.arctan (-L.2 / (-L.1)) = π / 4 := by sorry

/-- Function to represent a line passing through L with slope k -/
def line_through_L (k : ℝ) (x : ℝ) : ℝ := k * (x - L.1) + L.2

/-- Tangent line touches the ellipse at exactly one point -/
def is_tangent (k : ℝ) : Prop := 
  ∃! x, Γ x (line_through_L k x)

/-- The y-coordinates of the intersection points of the tangent lines with the y-axis -/
def y_intersections (k₁ k₂ : ℝ) : ℝ × ℝ := (line_through_L k₁ 0, line_through_L k₂ 0)

/-- Main theorem: The product of y-coordinates of intersection points is 9 -/
theorem tangent_intersection_product :
  ∃ k₁ k₂, is_tangent k₁ ∧ is_tangent k₂ ∧ k₁ ≠ k₂ ∧ 
    (y_intersections k₁ k₂).1 * (y_intersections k₁ k₂).2 = 9 := by sorry

end NUMINAMATH_CALUDE_OL_length_OL_angle_tangent_intersection_product_l3960_396040


namespace NUMINAMATH_CALUDE_factor_implies_k_value_l3960_396055

theorem factor_implies_k_value (k : ℚ) :
  (∀ x : ℚ, (x + 5) ∣ (k * x^3 + 27 * x^2 - k * x + 55)) →
  k = 73 / 12 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_k_value_l3960_396055


namespace NUMINAMATH_CALUDE_point_on_line_l3960_396086

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line (p1 p2 p3 p4 : Point) :
  p1 = Point.mk 2 5 →
  p2 = Point.mk 4 11 →
  p3 = Point.mk 6 17 →
  p4 = Point.mk 15 44 →
  collinear p1 p2 p3 →
  collinear p1 p2 p4 :=
by
  sorry


end NUMINAMATH_CALUDE_point_on_line_l3960_396086


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l3960_396021

theorem cubic_roots_problem (p q r : ℂ) (u v w : ℂ) : 
  (p^3 + 5*p^2 + 6*p - 8 = 0) →
  (q^3 + 5*q^2 + 6*q - 8 = 0) →
  (r^3 + 5*r^2 + 6*r - 8 = 0) →
  ((p+q)^3 + u*(p+q)^2 + v*(p+q) + w = 0) →
  ((q+r)^3 + u*(q+r)^2 + v*(q+r) + w = 0) →
  ((r+p)^3 + u*(r+p)^2 + v*(r+p) + w = 0) →
  w = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l3960_396021


namespace NUMINAMATH_CALUDE_tank_filling_time_l3960_396099

/-- The time (in hours) it takes to fill the tank without a leak -/
def fill_time : ℝ := 5

/-- The time (in hours) it takes for the leak to empty a full tank -/
def empty_time : ℝ := 30

/-- The extra time (in hours) it takes to fill the tank due to the leak -/
def extra_time : ℝ := 1

theorem tank_filling_time :
  extra_time = (1 / ((1 / fill_time) - (1 / empty_time))) - fill_time :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l3960_396099


namespace NUMINAMATH_CALUDE_yellow_candy_bounds_l3960_396025

/-- Represents the state of the candy distribution game -/
structure CandyGame where
  total : ℕ
  colors : ℕ
  yellow : ℕ
  other_colors : List ℕ

/-- Defines the rules of the candy distribution game -/
def valid_game (g : CandyGame) : Prop :=
  g.total = 22 ∧
  g.colors = 4 ∧
  g.yellow ≥ g.other_colors.head! ∧
  g.yellow ≥ g.other_colors.tail.head! ∧
  g.yellow ≥ g.other_colors.tail.tail.head! ∧
  g.yellow + g.other_colors.sum = g.total

/-- Defines the outcome of the game where both players have equal candies -/
def equal_outcome (g : CandyGame) : Prop :=
  ∃ (yi_candies ji_candies : ℕ),
    yi_candies = ji_candies ∧
    yi_candies + ji_candies = g.total

/-- Theorem stating the maximum and minimum number of yellow candies -/
theorem yellow_candy_bounds (g : CandyGame) :
  valid_game g → equal_outcome g →
  (g.yellow ≤ 16 ∧ g.yellow ≥ 8) :=
sorry

end NUMINAMATH_CALUDE_yellow_candy_bounds_l3960_396025


namespace NUMINAMATH_CALUDE_ostap_chess_scenario_exists_l3960_396032

theorem ostap_chess_scenario_exists : ∃ (N : ℕ), N + 5 * N + 10 * N = 64 := by
  sorry

end NUMINAMATH_CALUDE_ostap_chess_scenario_exists_l3960_396032


namespace NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l3960_396026

def marcus_points (three_point_goals two_point_goals free_throws four_point_goals : ℕ) : ℕ :=
  3 * three_point_goals + 2 * two_point_goals + free_throws + 4 * four_point_goals

def percentage (part whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

theorem marcus_percentage_of_team_points : 
  let marcus_total := marcus_points 5 10 8 2
  let team_total := 110
  abs (percentage marcus_total team_total - 46.36) < 0.01 := by sorry

end NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l3960_396026


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l3960_396081

theorem complex_magnitude_equation (x : ℝ) :
  x > 0 → Complex.abs (3 + x * Complex.I) = 7 → x = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l3960_396081


namespace NUMINAMATH_CALUDE_sum_of_smallest_and_largest_prime_l3960_396080

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def primes_between_10_and_50 : Set ℕ := {n : ℕ | 10 < n ∧ n < 50 ∧ is_prime n}

theorem sum_of_smallest_and_largest_prime :
  ∃ (min max : ℕ), min ∈ primes_between_10_and_50 ∧ 
                   max ∈ primes_between_10_and_50 ∧
                   (∀ p ∈ primes_between_10_and_50, min ≤ p) ∧
                   (∀ p ∈ primes_between_10_and_50, p ≤ max) ∧
                   min + max = 58 :=
sorry

end NUMINAMATH_CALUDE_sum_of_smallest_and_largest_prime_l3960_396080


namespace NUMINAMATH_CALUDE_large_square_area_l3960_396093

-- Define the squares
structure Square where
  side : ℕ

-- Define the problem setup
structure SquareProblem where
  small : Square
  medium : Square
  large : Square
  small_perimeter_lt_medium_side : 4 * small.side < medium.side
  exposed_area : (large.side ^ 2 - (small.side ^ 2 + medium.side ^ 2)) = 10

-- Theorem statement
theorem large_square_area (problem : SquareProblem) : problem.large.side ^ 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_large_square_area_l3960_396093


namespace NUMINAMATH_CALUDE_equation_result_l3960_396058

theorem equation_result (x : ℝ) : 
  14 * x + 5 - 21 * x^2 = -2 → 6 * x^2 - 4 * x + 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_result_l3960_396058


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3960_396087

theorem complex_absolute_value (ω : ℂ) : ω = 7 + 3*I → Complex.abs (ω^2 + 8*ω + 98) = Real.sqrt 41605 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3960_396087


namespace NUMINAMATH_CALUDE_ninth_root_unity_product_l3960_396038

theorem ninth_root_unity_product : 
  let x : ℂ := Complex.exp (2 * π * I / 9)
  (3 * x + x^2) * (3 * x^3 + x^6) * (3 * x^4 + x^8) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ninth_root_unity_product_l3960_396038


namespace NUMINAMATH_CALUDE_existence_of_integers_satisfying_inequality_l3960_396096

theorem existence_of_integers_satisfying_inequality :
  ∃ (A B : ℤ), (999/1000 : ℝ) < A + B * Real.sqrt 2 ∧ A + B * Real.sqrt 2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_satisfying_inequality_l3960_396096


namespace NUMINAMATH_CALUDE_log_823_bounds_sum_l3960_396005

theorem log_823_bounds_sum : ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 823 / Real.log 10 ∧ Real.log 823 / Real.log 10 < (d : ℝ) ∧ c + d = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_823_bounds_sum_l3960_396005


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3960_396013

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 2 + 2 * Complex.I) : 
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3960_396013


namespace NUMINAMATH_CALUDE_root_equation_coefficient_l3960_396073

theorem root_equation_coefficient (a : ℝ) : (2 : ℝ)^2 + a * 2 - 2 = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_coefficient_l3960_396073


namespace NUMINAMATH_CALUDE_sweet_potatoes_sold_l3960_396088

theorem sweet_potatoes_sold (total harvested : ℕ) (sold_to_lenon : ℕ) (unsold : ℕ) 
  (h1 : total = 80)
  (h2 : sold_to_lenon = 15)
  (h3 : unsold = 45) :
  total - sold_to_lenon - unsold = 20 :=
by sorry

end NUMINAMATH_CALUDE_sweet_potatoes_sold_l3960_396088


namespace NUMINAMATH_CALUDE_subset_condition_l3960_396056

theorem subset_condition (A B : Set ℕ) (m : ℕ) : 
  A = {0, 1, 2} → 
  B = {1, m} → 
  B ⊆ A → 
  m = 0 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_subset_condition_l3960_396056


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l3960_396069

-- Define the function f(x) = -x^3
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l3960_396069


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l3960_396052

theorem trigonometric_expression_value (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l3960_396052


namespace NUMINAMATH_CALUDE_electronic_devices_bought_l3960_396078

theorem electronic_devices_bought (original_price discount_price total_discount : ℕ) 
  (h1 : original_price = 800000)
  (h2 : discount_price = 450000)
  (h3 : total_discount = 16450000) :
  (total_discount / (original_price - discount_price) : ℕ) = 47 := by
  sorry

end NUMINAMATH_CALUDE_electronic_devices_bought_l3960_396078


namespace NUMINAMATH_CALUDE_square_root_meaningful_implies_x_geq_5_l3960_396075

theorem square_root_meaningful_implies_x_geq_5 (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_meaningful_implies_x_geq_5_l3960_396075


namespace NUMINAMATH_CALUDE_vector_subtraction_proof_l3960_396031

def a : ℝ × ℝ × ℝ := (5, -3, 2)
def b : ℝ × ℝ × ℝ := (-1, 4, -2)

theorem vector_subtraction_proof :
  a - 4 • b = (9, -19, 10) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_proof_l3960_396031


namespace NUMINAMATH_CALUDE_corn_harvest_problem_l3960_396083

/-- Represents the corn harvest problem -/
theorem corn_harvest_problem 
  (initial_harvest : ℝ) 
  (planned_harvest : ℝ) 
  (area_increase : ℝ) 
  (yield_improvement : ℝ) 
  (h1 : initial_harvest = 4340)
  (h2 : planned_harvest = 5520)
  (h3 : area_increase = 14)
  (h4 : yield_improvement = 5)
  (h5 : initial_harvest / 124 < 40) :
  ∃ (initial_area yield : ℝ),
    initial_area = 124 ∧ 
    yield = 35 ∧
    initial_harvest = initial_area * yield ∧
    planned_harvest = (initial_area + area_increase) * (yield + yield_improvement) := by
  sorry

end NUMINAMATH_CALUDE_corn_harvest_problem_l3960_396083


namespace NUMINAMATH_CALUDE_square_of_101_l3960_396061

theorem square_of_101 : (101 : ℕ)^2 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_square_of_101_l3960_396061


namespace NUMINAMATH_CALUDE_absolute_value_not_three_implies_not_three_l3960_396097

theorem absolute_value_not_three_implies_not_three (x : ℝ) : |x| ≠ 3 → x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_three_implies_not_three_l3960_396097


namespace NUMINAMATH_CALUDE_carla_cooks_three_steaks_l3960_396098

/-- Represents the cooking scenario for Carla --/
structure CookingScenario where
  waffle_time : ℕ    -- Time to cook a batch of waffles in minutes
  steak_time : ℕ     -- Time to cook one steak in minutes
  total_time : ℕ     -- Total cooking time in minutes

/-- Calculates the number of steaks Carla needs to cook --/
def steaks_to_cook (scenario : CookingScenario) : ℕ :=
  (scenario.total_time - scenario.waffle_time) / scenario.steak_time

/-- Theorem stating that Carla needs to cook 3 steaks --/
theorem carla_cooks_three_steaks (scenario : CookingScenario) 
  (h1 : scenario.waffle_time = 10)
  (h2 : scenario.steak_time = 6)
  (h3 : scenario.total_time = 28) :
  steaks_to_cook scenario = 3 := by
  sorry

#eval steaks_to_cook { waffle_time := 10, steak_time := 6, total_time := 28 }

end NUMINAMATH_CALUDE_carla_cooks_three_steaks_l3960_396098


namespace NUMINAMATH_CALUDE_inscribed_rectangle_epsilon_l3960_396002

-- Define the triangle
structure Triangle :=
  (MN NP PM : ℝ)

-- Define the rectangle
structure Rectangle :=
  (W X Y Z : ℝ × ℝ)

-- Define the area function
def rectangleArea (γ ε δ : ℝ) : ℝ := γ * δ - δ * ε^2

theorem inscribed_rectangle_epsilon (t : Triangle) (r : Rectangle) (γ ε : ℝ) :
  t.MN = 10 ∧ t.NP = 24 ∧ t.PM = 26 →
  (∃ δ, rectangleArea γ ε δ = 0) →
  (∃ δ, rectangleArea γ ε δ = 60) →
  ε = 5/12 := by
  sorry

#check inscribed_rectangle_epsilon

end NUMINAMATH_CALUDE_inscribed_rectangle_epsilon_l3960_396002


namespace NUMINAMATH_CALUDE_tuesday_rejects_l3960_396036

/-- The percentage of meters rejected as defective -/
def reject_rate : ℝ := 0.0007

/-- The number of meters rejected on Monday -/
def monday_rejects : ℕ := 7

/-- The increase in meters examined on Tuesday compared to Monday -/
def tuesday_increase : ℝ := 0.25

theorem tuesday_rejects : ℕ := by
  sorry

end NUMINAMATH_CALUDE_tuesday_rejects_l3960_396036


namespace NUMINAMATH_CALUDE_tensor_identity_l3960_396047

/-- Define a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Define the ⊗ operation -/
def tensor (m n : Vector2D) : Vector2D :=
  ⟨m.x * n.x + m.y * n.y, m.x * n.y + m.y * n.x⟩

theorem tensor_identity (p : Vector2D) : 
  (∀ m : Vector2D, tensor m p = m) → p = ⟨1, 0⟩ := by
  sorry

end NUMINAMATH_CALUDE_tensor_identity_l3960_396047


namespace NUMINAMATH_CALUDE_rotation_equivalence_l3960_396000

/-- 
Given a point that is rotated 450 degrees clockwise and x degrees counterclockwise 
about the same center to reach the same final position, prove that x = 270 degrees,
assuming x < 360.
-/
theorem rotation_equivalence (x : ℝ) : 
  (450 % 360 : ℝ) = (360 - x) % 360 → x < 360 → x = 270 := by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l3960_396000


namespace NUMINAMATH_CALUDE_cubic_function_coefficient_l3960_396007

/-- Given a cubic function f(x) = ax³ + 3x² + 2, 
    prove that if its second derivative at x = -1 is 4, 
    then the coefficient a must be 10/3. -/
theorem cubic_function_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = a * x^3 + 3 * x^2 + 2) →
  (deriv (deriv f)) (-1) = 4 →
  a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_coefficient_l3960_396007


namespace NUMINAMATH_CALUDE_solution_set_for_a_3_range_of_a_l3960_396072

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Part 1: Solution set for a = 3
theorem solution_set_for_a_3 :
  {x : ℝ | f 3 x ≥ 2} = {x : ℝ | x ≤ 2/3 ∨ x ≥ 2} := by sorry

-- Part 2: Range of a
theorem range_of_a :
  (∀ x : ℝ, f a x ≥ 5 - x) → a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_3_range_of_a_l3960_396072


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3960_396035

theorem equal_roots_quadratic (p : ℝ) : 
  (∃! p, ∀ x : ℝ, x^2 - p*x + p^2 = 0 → (∃! x : ℝ, x^2 - p*x + p^2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3960_396035


namespace NUMINAMATH_CALUDE_fraction_value_l3960_396090

theorem fraction_value (a b c : Int) (h1 : a = 5) (h2 : b = -3) (h3 : c = 2) :
  3 / (a + b + c : ℚ) = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l3960_396090


namespace NUMINAMATH_CALUDE_orthogonal_trajectories_and_intersection_angle_l3960_396030

-- Define the family of conics
def conic (a : ℝ) (x y : ℝ) : Prop :=
  (x + 2*y)^2 = a*(x + y)

-- Define the orthogonal trajectory
def orthogonal_trajectory (c : ℝ) (x y : ℝ) : Prop :=
  y = c*x^2 - 3*x

-- Theorem statement
theorem orthogonal_trajectories_and_intersection_angle :
  ∀ (a c : ℝ),
  (∃ (x y : ℝ), conic a x y ∧ orthogonal_trajectory c x y) ∧
  (∃ (x y : ℝ), conic a x y ∧ x = 0 ∧ y = 0 ∧
    ∃ (x' y' : ℝ), orthogonal_trajectory c x' y' ∧ x' = 0 ∧ y' = 0 ∧
    Real.arctan ((y' - y) / (x' - x)) = π / 4) :=
by sorry


end NUMINAMATH_CALUDE_orthogonal_trajectories_and_intersection_angle_l3960_396030


namespace NUMINAMATH_CALUDE_largest_three_digit_sum_22_l3960_396039

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.Nodup digits

theorem largest_three_digit_sum_22 :
  ∃ (n : ℕ), is_three_digit n ∧ 
             has_distinct_digits n ∧ 
             sum_of_digits n = 22 ∧
             ∀ (m : ℕ), is_three_digit m → 
                        has_distinct_digits m → 
                        sum_of_digits m = 22 → 
                        m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_sum_22_l3960_396039


namespace NUMINAMATH_CALUDE_arithmetic_sequence_collinearity_geometric_sequence_characterization_l3960_396009

def isArithmeticSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

def isGeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ+, a (n + 1) = q * a n

def S (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

def areCollinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

theorem arithmetic_sequence_collinearity (a : ℕ+ → ℝ) :
  isArithmeticSequence a →
  areCollinear (10, S a 10 / 10) (100, S a 100 / 100) (110, S a 110 / 110) :=
sorry

theorem geometric_sequence_characterization (a : ℕ+ → ℝ) (a₁ q : ℝ) :
  (∀ n : ℕ+, S a (n + 1) = a₁ + q * S a n) ∧ q ≠ 0 →
  isGeometricSequence a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_collinearity_geometric_sequence_characterization_l3960_396009


namespace NUMINAMATH_CALUDE_cyclic_equality_l3960_396070

theorem cyclic_equality (a b c x y z : ℝ) 
  (h1 : a = b * z + c * y)
  (h2 : b = c * x + a * z)
  (h3 : c = a * y + b * x)
  (hx : x ≠ 1 ∧ x ≠ -1)
  (hy : y ≠ 1 ∧ y ≠ -1)
  (hz : z ≠ 1 ∧ z ≠ -1) :
  a^2 / (1 - x^2) = b^2 / (1 - y^2) ∧ b^2 / (1 - y^2) = c^2 / (1 - z^2) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_equality_l3960_396070


namespace NUMINAMATH_CALUDE_polynomial_no_x_x2_terms_l3960_396014

theorem polynomial_no_x_x2_terms (m n : ℚ) : 
  (∀ x, 3 * (x^3 + 1/3 * x^2 + n * x) - (m * x^2 - 6 * x - 1) = 
        3 * x^3 + 1) → 
  m + n = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_no_x_x2_terms_l3960_396014


namespace NUMINAMATH_CALUDE_root_sum_fraction_l3960_396095

theorem root_sum_fraction (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p - 6 = 0 →
  q^3 - 6*q^2 + 11*q - 6 = 0 →
  r^3 - 6*r^2 + 11*r - 6 = 0 →
  (p / (p*q + 2)) + (q / (p*r + 2)) + (r / (q*p + 2)) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l3960_396095


namespace NUMINAMATH_CALUDE_total_dragons_is_eight_l3960_396012

/-- Represents the number of heads on a dragon -/
inductive DragonHeads
  | two
  | seven

/-- Counts the total number of dragons given the conditions of the problem -/
def count_dragons : Nat :=
  let two_headed := 6
  let seven_headed := 2
  two_headed + seven_headed

/-- The main theorem stating that the total number of dragons is 8 -/
theorem total_dragons_is_eight :
  count_dragons = 8 ∧
  ∃ (x y : Nat),
    x * 2 + y * 7 = 25 + 7 ∧  -- Total heads including the counting head
    x + y = count_dragons ∧
    x ≥ 0 ∧ y > 0 :=
by sorry

end NUMINAMATH_CALUDE_total_dragons_is_eight_l3960_396012


namespace NUMINAMATH_CALUDE_stock_value_indeterminate_l3960_396062

theorem stock_value_indeterminate (yield : ℝ) (market_value : ℝ) 
  (h_yield : yield = 0.08) (h_market_value : market_value = 150) :
  ∀ original_value : ℝ, 
  (original_value > 0 ∧ yield * original_value = market_value) ∨
  (original_value > 0 ∧ yield * original_value ≠ market_value) :=
by sorry

end NUMINAMATH_CALUDE_stock_value_indeterminate_l3960_396062


namespace NUMINAMATH_CALUDE_square_sum_value_l3960_396094

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 12) : a^2 + b^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l3960_396094


namespace NUMINAMATH_CALUDE_fraction_power_seven_l3960_396022

theorem fraction_power_seven : (5 / 3 : ℚ) ^ 7 = 78125 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_seven_l3960_396022


namespace NUMINAMATH_CALUDE_largest_interior_angle_of_triangle_l3960_396050

theorem largest_interior_angle_of_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 360 →
  a / 5 = b / 4 →
  a / 5 = c / 3 →
  max (180 - a) (max (180 - b) (180 - c)) = 90 :=
by sorry

end NUMINAMATH_CALUDE_largest_interior_angle_of_triangle_l3960_396050


namespace NUMINAMATH_CALUDE_jills_yard_area_l3960_396084

/-- Represents a rectangular yard with fence posts -/
structure FencedYard where
  shorterSidePosts : ℕ
  longerSidePosts : ℕ
  postSpacing : ℕ

/-- The total number of fence posts -/
def FencedYard.totalPosts (yard : FencedYard) : ℕ :=
  2 * (yard.shorterSidePosts + yard.longerSidePosts) - 4

/-- The length of the shorter side of the yard -/
def FencedYard.shorterSide (yard : FencedYard) : ℕ :=
  yard.postSpacing * (yard.shorterSidePosts - 1)

/-- The length of the longer side of the yard -/
def FencedYard.longerSide (yard : FencedYard) : ℕ :=
  yard.postSpacing * (yard.longerSidePosts - 1)

/-- The area of the yard -/
def FencedYard.area (yard : FencedYard) : ℕ :=
  yard.shorterSide * yard.longerSide

/-- Theorem: The area of Jill's yard is 144 square yards -/
theorem jills_yard_area :
  ∃ (yard : FencedYard),
    yard.totalPosts = 24 ∧
    yard.postSpacing = 3 ∧
    yard.longerSidePosts = 3 * yard.shorterSidePosts ∧
    yard.area = 144 :=
by
  sorry


end NUMINAMATH_CALUDE_jills_yard_area_l3960_396084


namespace NUMINAMATH_CALUDE_three_numbers_ratio_l3960_396010

theorem three_numbers_ratio (a b c : ℕ+) : 
  (Nat.lcm a (Nat.lcm b c) = 2400) → 
  (Nat.gcd a (Nat.gcd b c) = 40) → 
  (∃ (k : ℕ+), a = 3 * k ∧ b = 4 * k ∧ c = 5 * k) := by
sorry

end NUMINAMATH_CALUDE_three_numbers_ratio_l3960_396010


namespace NUMINAMATH_CALUDE_triangle_properties_l3960_396006

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if b * sin(A) = (√3/2) * a, a = 2c, and b = 2√6,
    then the measure of angle B is π/3 and the area of the triangle is 4√3. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- acute triangle condition
  b * Real.sin A = (Real.sqrt 3 / 2) * a →  -- given condition
  a = 2 * c →  -- given condition
  b = 2 * Real.sqrt 6 →  -- given condition
  B = π / 3 ∧ (1 / 2) * a * c * Real.sin B = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3960_396006


namespace NUMINAMATH_CALUDE_min_even_integers_l3960_396011

theorem min_even_integers (a b c d e f g : ℤ) : 
  a + b = 29 → 
  a + b + c + d = 47 → 
  a + b + c + d + e + f + g = 66 → 
  (∃ (count : ℕ), count ≥ 1 ∧ 
    count = (if Even a then 1 else 0) + 
            (if Even b then 1 else 0) + 
            (if Even c then 1 else 0) + 
            (if Even d then 1 else 0) + 
            (if Even e then 1 else 0) + 
            (if Even f then 1 else 0) + 
            (if Even g then 1 else 0) ∧
    ∀ (other_count : ℕ), 
      other_count = (if Even a then 1 else 0) + 
                    (if Even b then 1 else 0) + 
                    (if Even c then 1 else 0) + 
                    (if Even d then 1 else 0) + 
                    (if Even e then 1 else 0) + 
                    (if Even f then 1 else 0) + 
                    (if Even g then 1 else 0) →
      count ≤ other_count) :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l3960_396011


namespace NUMINAMATH_CALUDE_scaled_variance_l3960_396029

def variance (data : List ℝ) : ℝ := sorry

theorem scaled_variance (data : List ℝ) (h : variance data = 3) :
  variance (List.map (· * 2) data) = 12 := by sorry

end NUMINAMATH_CALUDE_scaled_variance_l3960_396029


namespace NUMINAMATH_CALUDE_number_calculation_l3960_396023

theorem number_calculation (x : ℝ) (h : 0.45 * x = 162) : x = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l3960_396023


namespace NUMINAMATH_CALUDE_diego_fruit_weight_l3960_396024

/-- The weight of watermelon Diego buys -/
def watermelon_weight : ℕ := 1

/-- The weight of grapes Diego buys -/
def grapes_weight : ℕ := 1

/-- The weight of oranges Diego buys -/
def oranges_weight : ℕ := 1

/-- The weight of apples Diego buys -/
def apples_weight : ℕ := 17

/-- The total weight of fruit Diego can carry in his bookbag -/
def total_weight : ℕ := watermelon_weight + grapes_weight + oranges_weight + apples_weight

theorem diego_fruit_weight : total_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_diego_fruit_weight_l3960_396024


namespace NUMINAMATH_CALUDE_bread_slices_per_loaf_l3960_396033

theorem bread_slices_per_loaf :
  ∀ (num_loaves : ℕ) (payment : ℕ) (change : ℕ) (slice_cost : ℚ),
    num_loaves = 3 →
    payment = 40 →
    change = 16 →
    slice_cost = 2/5 →
    (((payment - change : ℚ) / slice_cost) / num_loaves : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_per_loaf_l3960_396033


namespace NUMINAMATH_CALUDE_curve_transformation_l3960_396065

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y = (1/3) * Real.cos (2 * x)

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop := x' = 2 * x ∧ y' = 3 * y

-- State the theorem
theorem curve_transformation (x y x' y' : ℝ) :
  original_curve x y → transformation x y x' y' → y' = Real.cos x' := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l3960_396065


namespace NUMINAMATH_CALUDE_cube_equals_nine_times_implies_fifth_power_l3960_396015

theorem cube_equals_nine_times_implies_fifth_power (w : ℕ+) 
  (h : w.val ^ 3 = 9 * w.val) : w.val ^ 5 = 243 := by
  sorry

end NUMINAMATH_CALUDE_cube_equals_nine_times_implies_fifth_power_l3960_396015


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3960_396028

/-- The line equation passes through the point (2,2) for all values of k -/
theorem line_passes_through_point :
  ∀ (k : ℝ), (1 + 4*k) * 2 - (2 - 3*k) * 2 + 2 - 14*k = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3960_396028


namespace NUMINAMATH_CALUDE_yoo_seung_marbles_yoo_seung_marbles_proof_l3960_396059

/-- Proves that Yoo Seung has 108 marbles given the conditions in the problem -/
theorem yoo_seung_marbles : ℕ → ℕ → ℕ → Prop :=
  fun young_soo han_sol yoo_seung =>
    han_sol = young_soo + 15 ∧
    yoo_seung = 3 * han_sol ∧
    young_soo + han_sol + yoo_seung = 165 →
    yoo_seung = 108

/-- Proof of the theorem -/
theorem yoo_seung_marbles_proof : ∃ (young_soo han_sol yoo_seung : ℕ),
  yoo_seung_marbles young_soo han_sol yoo_seung :=
by
  sorry

end NUMINAMATH_CALUDE_yoo_seung_marbles_yoo_seung_marbles_proof_l3960_396059


namespace NUMINAMATH_CALUDE_consecutive_points_length_l3960_396079

/-- Given 5 consecutive points on a straight line, prove the length of ae -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (c - b = 2 * (d - c)) →  -- bc = 2 cd
  (e - d = 8) →            -- de = 8
  (b - a = 5) →            -- ab = 5
  (c - a = 11) →           -- ac = 11
  (e - a = 22) :=          -- ae = 22
by sorry

end NUMINAMATH_CALUDE_consecutive_points_length_l3960_396079


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3960_396037

theorem geometric_series_sum : 
  let a : ℤ := -3  -- first term
  let r : ℤ := -2  -- common ratio
  let n : ℕ := 9   -- number of terms
  let last_term : ℤ := -768  -- last term of the series
  let sum : ℚ := (a * (r^n - 1)) / (r - 1)  -- sum formula for geometric series
  (a * r^(n-1) = last_term) →  -- condition: last term matches the formula
  sum = 514 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3960_396037


namespace NUMINAMATH_CALUDE_asymptotes_of_hyperbola_l3960_396042

theorem asymptotes_of_hyperbola (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e1 := Real.sqrt (a^2 - b^2) / a
  let e2 := Real.sqrt (a^2 + b^2) / a
  let C1 := fun (x y : ℝ) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let C2 := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  e1 * e2 = Real.sqrt 15 / 4 →
  (∀ x y, C2 x y → (x + 2*y = 0 ∨ x - 2*y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_asymptotes_of_hyperbola_l3960_396042


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l3960_396043

theorem multiplication_puzzle (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → (30 + a) * (10 * b + 4) = 142 → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l3960_396043


namespace NUMINAMATH_CALUDE_remaining_work_for_x_l3960_396046

/-- The number of days x needs to finish the remaining work after y worked for 5 days --/
def remaining_days_for_x (x_days y_days : ℚ) : ℚ :=
  (1 - 5 / y_days) * x_days

theorem remaining_work_for_x :
  remaining_days_for_x 21 15 = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_work_for_x_l3960_396046


namespace NUMINAMATH_CALUDE_smallest_c_for_inequality_l3960_396018

theorem smallest_c_for_inequality (m n : ℕ) : 
  (∀ c : ℕ, (27 ^ c) * (2 ^ (24 - n)) > (3 ^ (24 + m)) * (5 ^ n) → c ≥ 9) ∧ 
  ((27 ^ 9) * (2 ^ (24 - n)) > (3 ^ (24 + m)) * (5 ^ n)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_inequality_l3960_396018


namespace NUMINAMATH_CALUDE_positive_sum_one_inequality_l3960_396077

theorem positive_sum_one_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x^2 - 1) * (1 / y^2 - 1) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_one_inequality_l3960_396077


namespace NUMINAMATH_CALUDE_inverse_proportion_quadrants_l3960_396091

theorem inverse_proportion_quadrants (k : ℝ) (h1 : k ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ k / x
  (f 1 = 1) →
  (∀ x : ℝ, x ≠ 0 → (x > 0 ∧ f x > 0) ∨ (x < 0 ∧ f x < 0)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_quadrants_l3960_396091


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l3960_396071

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + b*x + 9 = 0 → x.im ≠ 0) ↔ -6 < b ∧ b < 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l3960_396071


namespace NUMINAMATH_CALUDE_set_operations_l3960_396076

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem set_operations :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 5}) ∧
  ((U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)}) ∧
  (A ∩ (U \ B) = {x | 2 ≤ x ∧ x < 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3960_396076


namespace NUMINAMATH_CALUDE_intersection_distance_l3960_396074

theorem intersection_distance (m b k : ℝ) (h1 : b ≠ 0) (h2 : 1 = 2 * m + b) :
  let f := fun x => x^2 + 6 * x - 4
  let g := fun x => m * x + b
  let d := |f k - g k|
  (m = 4 ∧ b = -7) → d = 9 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l3960_396074


namespace NUMINAMATH_CALUDE_octagon_area_ratio_octagon_area_ratio_proof_l3960_396051

/-- The ratio of the area of a regular octagon circumscribed about a circle
    to the area of a regular octagon inscribed in the same circle is 2. -/
theorem octagon_area_ratio : ℝ → ℝ → Prop :=
  fun (area_circumscribed area_inscribed : ℝ) =>
    area_circumscribed / area_inscribed = 2

/-- Given a circle with radius r, the area of its circumscribed regular octagon
    is twice the area of its inscribed regular octagon. -/
theorem octagon_area_ratio_proof (r : ℝ) (r_pos : r > 0) :
  ∃ (area_circumscribed area_inscribed : ℝ),
    area_circumscribed > 0 ∧
    area_inscribed > 0 ∧
    octagon_area_ratio area_circumscribed area_inscribed :=
by
  sorry

end NUMINAMATH_CALUDE_octagon_area_ratio_octagon_area_ratio_proof_l3960_396051


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3960_396044

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 15 → 
    b = 36 → 
    c^2 = a^2 + b^2 → 
    c = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3960_396044


namespace NUMINAMATH_CALUDE_min_value_fraction_l3960_396045

theorem min_value_fraction (x y : ℝ) (hx : 1/2 ≤ x ∧ x ≤ 2) (hy : 4/3 ≤ y ∧ y ≤ 3/2) :
  (x^3 * y^3) / (x^6 + 3*x^4*y^2 + 3*x^3*y^3 + 3*x^2*y^4 + y^6) ≥ 27/1081 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3960_396045
