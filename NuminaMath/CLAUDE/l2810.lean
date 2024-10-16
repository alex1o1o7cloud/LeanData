import Mathlib

namespace NUMINAMATH_CALUDE_square_ratio_theorem_l2810_281042

theorem square_ratio_theorem : ∃ (a b c : ℕ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (180 : ℝ) / 45 = (a * (b.sqrt : ℝ) / c : ℝ)^2 ∧ 
  a + b + c = 4 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_theorem_l2810_281042


namespace NUMINAMATH_CALUDE_no_such_function_exists_l2810_281038

theorem no_such_function_exists :
  ¬∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f (f n) = n + 1987 := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l2810_281038


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2810_281053

theorem polynomial_divisibility : ∀ x : ℂ,
  (x^100 + x^75 + x^50 + x^25 + 1) % (x^9 + x^6 + x^3 + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2810_281053


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l2810_281085

theorem least_positive_integer_congruence (b : ℕ) : 
  (b % 3 = 2) ∧ 
  (b % 4 = 3) ∧ 
  (b % 5 = 4) ∧ 
  (b % 9 = 8) ∧ 
  (∀ x : ℕ, x < b → ¬((x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) ∧ (x % 9 = 8))) →
  b = 179 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l2810_281085


namespace NUMINAMATH_CALUDE_point_A_coordinates_l2810_281076

/-- A point in the second quadrant of the Cartesian coordinate system with coordinates dependent on an integer m -/
def point_A (m : ℤ) : ℝ × ℝ := (7 - 2*m, 5 - m)

/-- Predicate to check if a point is in the second quadrant -/
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- Theorem stating that if A(7-2m, 5-m) is in the second quadrant and m is an integer, then A(-1, 1) is the only solution -/
theorem point_A_coordinates : 
  ∃! m : ℤ, in_second_quadrant (point_A m) ∧ point_A m = (-1, 1) :=
sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l2810_281076


namespace NUMINAMATH_CALUDE_triangle_area_l2810_281041

theorem triangle_area (a b c : ℝ) (ha : a^2 = 225) (hb : b^2 = 225) (hc : c^2 = 64) :
  (1/2) * a * c = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2810_281041


namespace NUMINAMATH_CALUDE_chocolate_sales_theorem_l2810_281087

/-- Represents the chocolate sales problem -/
structure ChocolateSales where
  total_customers : ℕ
  price_A : ℕ
  price_B : ℕ
  max_B_ratio : ℚ
  price_increase_step : ℕ
  A_decrease_rate : ℕ
  B_decrease_rate : ℕ

/-- The main theorem for the chocolate sales problem -/
theorem chocolate_sales_theorem (cs : ChocolateSales)
  (h_total : cs.total_customers = 480)
  (h_price_A : cs.price_A = 90)
  (h_price_B : cs.price_B = 50)
  (h_max_B_ratio : cs.max_B_ratio = 3/5)
  (h_price_increase_step : cs.price_increase_step = 3)
  (h_A_decrease_rate : cs.A_decrease_rate = 5)
  (h_B_decrease_rate : cs.B_decrease_rate = 3) :
  ∃ (min_A : ℕ) (women_day_price_A : ℕ),
    min_A = 300 ∧
    women_day_price_A = 150 ∧
    min_A + (cs.total_customers - min_A) ≤ cs.total_customers ∧
    (cs.total_customers - min_A) ≤ cs.max_B_ratio * min_A ∧
    (min_A - (women_day_price_A - cs.price_A) / cs.price_increase_step * cs.A_decrease_rate) *
      women_day_price_A +
    ((cs.total_customers - min_A) - (women_day_price_A - cs.price_A) / cs.price_increase_step * cs.B_decrease_rate) *
      cs.price_B =
    min_A * cs.price_A + (cs.total_customers - min_A) * cs.price_B :=
by sorry


end NUMINAMATH_CALUDE_chocolate_sales_theorem_l2810_281087


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2810_281030

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 → a + b > 0) ∧
  (∃ a b : ℝ, a + b > 0 ∧ ¬(a > 0 ∧ b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2810_281030


namespace NUMINAMATH_CALUDE_even_power_special_number_l2810_281056

theorem even_power_special_number : 
  ∃ (n : ℕ) (k : ℕ), 
    k % 2 = 0 ∧ 
    1000 ≤ 55^k ∧ 
    55^k < 10000 ∧ 
    (55^k / 1000 = 3) ∧ 
    (55^k % 10 = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_even_power_special_number_l2810_281056


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_19_l2810_281097

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if all digits in a natural number are different -/
def has_different_digits (n : ℕ) : Prop := sorry

/-- Theorem stating that 982 is the largest number with all different digits whose digits add up to 19 -/
theorem largest_number_with_digit_sum_19 :
  ∀ n : ℕ, n ≤ 982 ∨ digit_sum n ≠ 19 ∨ ¬(has_different_digits n) := by sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_19_l2810_281097


namespace NUMINAMATH_CALUDE_trigonometric_calculation_quadratic_equation_solution_l2810_281051

-- Problem 1
theorem trigonometric_calculation :
  3 * Real.tan (45 * π / 180) - (1 / 3)⁻¹ + (Real.sin (30 * π / 180) - 2022)^0 + |Real.cos (30 * π / 180) - Real.sqrt 3 / 2| = 1 := by
  sorry

-- Problem 2
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x * (x + 3) - 5 * (x + 3)
  (f 5 = 0 ∧ f (-3) = 0) ∧ ∀ x, f x = 0 → x = 5 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_calculation_quadratic_equation_solution_l2810_281051


namespace NUMINAMATH_CALUDE_feeding_theorem_l2810_281006

/-- Represents the number of animal pairs in the sanctuary -/
def num_pairs : ℕ := 6

/-- Represents the feeding order constraint for tigers -/
def tiger_constraint : Prop := true

/-- Represents the constraint that no two same-gender animals can be fed consecutively -/
def alternating_gender_constraint : Prop := true

/-- Represents that the first animal fed is the male lion -/
def starts_with_male_lion : Prop := true

/-- Calculates the number of ways to feed the animals given the constraints -/
def feeding_ways : ℕ := 14400

/-- Theorem stating the number of ways to feed the animals -/
theorem feeding_theorem :
  num_pairs = 6 ∧
  tiger_constraint ∧
  alternating_gender_constraint ∧
  starts_with_male_lion →
  feeding_ways = 14400 := by
  sorry

end NUMINAMATH_CALUDE_feeding_theorem_l2810_281006


namespace NUMINAMATH_CALUDE_cryptarithm_unique_solution_l2810_281040

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Represents the cryptarithm KIC + KCI = ICK -/
def cryptarithm (K I C : Digit) : Prop :=
  100 * K.val + 10 * I.val + C.val +
  100 * K.val + 10 * C.val + I.val =
  100 * I.val + 10 * C.val + K.val

/-- The cryptarithm has a unique solution -/
theorem cryptarithm_unique_solution :
  ∃! (K I C : Digit), cryptarithm K I C ∧ K ≠ I ∧ K ≠ C ∧ I ≠ C ∧
  K.val = 4 ∧ I.val = 9 ∧ C.val = 5 := by sorry

end NUMINAMATH_CALUDE_cryptarithm_unique_solution_l2810_281040


namespace NUMINAMATH_CALUDE_first_alloy_copper_percentage_l2810_281011

/-- The percentage of copper in the final alloy -/
def final_alloy_percentage : ℝ := 15

/-- The total amount of the final alloy in ounces -/
def total_alloy : ℝ := 121

/-- The amount of the first alloy used in ounces -/
def first_alloy_amount : ℝ := 66

/-- The percentage of copper in the second alloy -/
def second_alloy_percentage : ℝ := 21

/-- The percentage of copper in the first alloy -/
def first_alloy_percentage : ℝ := 10

theorem first_alloy_copper_percentage :
  first_alloy_amount * (first_alloy_percentage / 100) +
  (total_alloy - first_alloy_amount) * (second_alloy_percentage / 100) =
  total_alloy * (final_alloy_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_first_alloy_copper_percentage_l2810_281011


namespace NUMINAMATH_CALUDE_supermarket_spending_l2810_281007

theorem supermarket_spending (F : ℚ) : 
  F + (1 : ℚ)/3 + (1 : ℚ)/10 + 8/120 = 1 → F = (1 : ℚ)/2 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2810_281007


namespace NUMINAMATH_CALUDE_ruble_payment_l2810_281005

theorem ruble_payment (n : ℕ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end NUMINAMATH_CALUDE_ruble_payment_l2810_281005


namespace NUMINAMATH_CALUDE_tangent_line_to_cubic_curve_l2810_281033

theorem tangent_line_to_cubic_curve (a : ℝ) :
  (∃ x y : ℝ, y = 3 * x + 1 ∧ y = x^3 - a ∧ 3 * x^2 = 3) →
  (a = -3 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_cubic_curve_l2810_281033


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2810_281080

/-- An increasing arithmetic sequence of integers -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℤ) 
  (h_seq : arithmetic_sequence a) 
  (h_prod : a 4 * a 5 = 12) : 
  a 2 * a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2810_281080


namespace NUMINAMATH_CALUDE_house_size_multiple_l2810_281058

/-- The size of Sara's house in square feet -/
def sara_house_size : ℝ := 1000

/-- The size of Nada's house in square feet -/
def nada_house_size : ℝ := 450

/-- The extra size in square feet that Sara's house has -/
def extra_size : ℝ := 100

/-- The theorem stating the multiple of Nada's house size compared to Sara's house size -/
theorem house_size_multiple : 
  ∃ (m : ℝ), m * nada_house_size + extra_size = sara_house_size ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_house_size_multiple_l2810_281058


namespace NUMINAMATH_CALUDE_abs_cubic_inequality_l2810_281066

theorem abs_cubic_inequality (x : ℝ) : 
  |x| ≤ 2 → |3*x - x^3| ≤ 2 := by sorry

end NUMINAMATH_CALUDE_abs_cubic_inequality_l2810_281066


namespace NUMINAMATH_CALUDE_flower_percentages_l2810_281067

def total_flowers : ℕ := 30
def red_flowers : ℕ := 7
def white_flowers : ℕ := 6
def blue_flowers : ℕ := 5
def yellow_flowers : ℕ := 4

def purple_flowers : ℕ := total_flowers - (red_flowers + white_flowers + blue_flowers + yellow_flowers)

def percentage (part : ℕ) (whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

theorem flower_percentages :
  (percentage (red_flowers + white_flowers + blue_flowers) total_flowers = 60) ∧
  (percentage purple_flowers total_flowers = 26.67) ∧
  (percentage yellow_flowers total_flowers = 13.33) :=
by sorry

end NUMINAMATH_CALUDE_flower_percentages_l2810_281067


namespace NUMINAMATH_CALUDE_largest_power_of_six_divisor_l2810_281064

theorem largest_power_of_six_divisor : 
  (∃ k : ℕ, 6^k ∣ (8 * 48 * 81) ∧ 
   ∀ m : ℕ, m > k → ¬(6^m ∣ (8 * 48 * 81))) → 
  (∃ k : ℕ, k = 5 ∧ 6^k ∣ (8 * 48 * 81) ∧ 
   ∀ m : ℕ, m > k → ¬(6^m ∣ (8 * 48 * 81))) := by
sorry

end NUMINAMATH_CALUDE_largest_power_of_six_divisor_l2810_281064


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2810_281090

def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  (a - 1) * x^2 + 2 * (a - 1) * x - 4 ≥ 0

def empty_solution_set (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬(quadratic_inequality a x)

theorem quadratic_inequality_range :
  ∀ a : ℝ, empty_solution_set a ↔ -3 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2810_281090


namespace NUMINAMATH_CALUDE_permutation_solution_l2810_281049

def is_valid_permutation (a : Fin 9 → ℕ) : Prop :=
  (∀ i j : Fin 9, i ≠ j → a i ≠ a j) ∧
  (∀ i : Fin 9, a i ∈ (Set.range (fun i : Fin 9 => i.val + 1)))

def satisfies_conditions (a : Fin 9 → ℕ) : Prop :=
  (a 0 + a 1 + a 2 + a 3 = a 3 + a 4 + a 5 + a 6) ∧
  (a 3 + a 4 + a 5 + a 6 = a 6 + a 7 + a 8 + a 0) ∧
  (a 0^2 + a 1^2 + a 2^2 + a 3^2 = a 3^2 + a 4^2 + a 5^2 + a 6^2) ∧
  (a 3^2 + a 4^2 + a 5^2 + a 6^2 = a 6^2 + a 7^2 + a 8^2 + a 0^2)

def solution : Fin 9 → ℕ := fun i =>
  match i with
  | ⟨0, _⟩ => 2
  | ⟨1, _⟩ => 4
  | ⟨2, _⟩ => 9
  | ⟨3, _⟩ => 5
  | ⟨4, _⟩ => 1
  | ⟨5, _⟩ => 6
  | ⟨6, _⟩ => 8
  | ⟨7, _⟩ => 3
  | ⟨8, _⟩ => 7

theorem permutation_solution :
  is_valid_permutation solution ∧ satisfies_conditions solution :=
by sorry

end NUMINAMATH_CALUDE_permutation_solution_l2810_281049


namespace NUMINAMATH_CALUDE_existence_of_m_and_k_l2810_281048

def f (p : ℕ × ℕ) : ℕ × ℕ :=
  let (a, b) := p
  if a < b then (2*a, b-a) else (a-b, 2*b)

def iter_f (k : ℕ) : (ℕ × ℕ) → (ℕ × ℕ) :=
  match k with
  | 0 => id
  | k+1 => f ∘ (iter_f k)

theorem existence_of_m_and_k (n : ℕ) (h : n > 1) :
  ∃ (m k : ℕ), m < n ∧ iter_f k (n, m) = (m, n) := by
  sorry

#check existence_of_m_and_k

end NUMINAMATH_CALUDE_existence_of_m_and_k_l2810_281048


namespace NUMINAMATH_CALUDE_inverse_function_property_l2810_281073

-- Define the function f
def f : ℕ → ℕ
| 1 => 3
| 2 => 13
| 3 => 8
| 5 => 1
| 8 => 0
| 13 => 5
| _ => 0  -- Default case for other inputs

-- Define the inverse function f_inv
def f_inv : ℕ → ℕ
| 0 => 8
| 1 => 5
| 3 => 1
| 5 => 13
| 8 => 3
| 13 => 2
| _ => 0  -- Default case for other inputs

-- Theorem statement
theorem inverse_function_property :
  f_inv ((f_inv 5 + f_inv 13) / f_inv 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_property_l2810_281073


namespace NUMINAMATH_CALUDE_max_teams_with_10_points_l2810_281043

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  total_teams : Nat
  points_per_win : Nat
  points_per_draw : Nat
  points_per_loss : Nat
  target_points : Nat

/-- The maximum number of teams that can achieve the target points -/
def max_teams_with_target_points (tournament : FootballTournament) : Nat :=
  sorry

/-- Theorem stating the maximum number of teams that can score exactly 10 points -/
theorem max_teams_with_10_points :
  let tournament := FootballTournament.mk 17 3 1 0 10
  max_teams_with_target_points tournament = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_teams_with_10_points_l2810_281043


namespace NUMINAMATH_CALUDE_division_result_l2810_281060

theorem division_result : 
  (7125 : ℝ) / 1.25 = 5700 → (712.5 : ℝ) / 12.5 = 57 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2810_281060


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l2810_281068

open Real

theorem arctan_tan_difference (θ₁ θ₂ : ℝ) (h₁ : θ₁ = 70 * π / 180) (h₂ : θ₂ = 20 * π / 180) :
  arctan (tan θ₁ - 3 * tan θ₂) = 50 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l2810_281068


namespace NUMINAMATH_CALUDE_grid_sum_theorem_l2810_281008

/-- Represents a 3x3 grid of numbers -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if all numbers in the grid are unique and between 1 and 9 -/
def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 9) ∧
  (∀ i j k l, g i j = g k l → (i = k ∧ j = l))

/-- Sum of the right column -/
def right_column_sum (g : Grid) : Nat :=
  g 0 2 + g 1 2 + g 2 2

/-- Sum of the bottom row -/
def bottom_row_sum (g : Grid) : Nat :=
  g 2 0 + g 2 1 + g 2 2

theorem grid_sum_theorem (g : Grid) 
  (h_valid : valid_grid g) 
  (h_right_sum : right_column_sum g = 32) 
  (h_corner : g 2 2 = 7) : 
  bottom_row_sum g = 18 :=
sorry

end NUMINAMATH_CALUDE_grid_sum_theorem_l2810_281008


namespace NUMINAMATH_CALUDE_expression_value_l2810_281003

theorem expression_value (a b c d m : ℝ) : 
  (a = -b) → (c * d = 1) → (abs m = 2) → 
  (3 * (a + b - 1) + (-c * d) ^ 2023 - 2 * m = -8 ∨ 
   3 * (a + b - 1) + (-c * d) ^ 2023 - 2 * m = 0) := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2810_281003


namespace NUMINAMATH_CALUDE_fred_paid_twenty_l2810_281017

/-- The amount Fred paid with at the movie theater -/
def fred_payment (ticket_price : ℚ) (num_tickets : ℕ) (borrowed_movie_price : ℚ) (change : ℚ) : ℚ :=
  ticket_price * num_tickets + borrowed_movie_price + change

/-- Theorem: Fred paid $20.00 at the movie theater -/
theorem fred_paid_twenty : fred_payment 5.92 2 6.79 1.37 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fred_paid_twenty_l2810_281017


namespace NUMINAMATH_CALUDE_expression_simplification_l2810_281074

variable (a b x : ℝ)

theorem expression_simplification (h : x ≥ a) :
  (Real.sqrt (b^2 + a^2 + x^2) - (x^3 - a^3) / Real.sqrt (b^2 + a^2 + x^2)) / (b^2 + a^2 + x^2) = 
  (b^2 + a^2 + a^3) / (b^2 + a^2 + x^2)^(3/2) := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2810_281074


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2810_281070

theorem exponent_multiplication (x : ℝ) (a b : ℕ) : x^a * x^b = x^(a + b) := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2810_281070


namespace NUMINAMATH_CALUDE_ten_sparklers_to_crackers_five_ornaments_one_cracker_more_valuable_l2810_281081

-- Define the exchange rates
def ornament_to_cracker : ℚ := 2
def sparkler_to_garland : ℚ := 2/5
def ornament_to_garland : ℚ := 1/4

-- Define the conversion function
def convert (item : String) (quantity : ℚ) : ℚ :=
  match item with
  | "sparkler" => quantity * sparkler_to_garland * (1 / ornament_to_garland) * ornament_to_cracker
  | "ornament" => quantity * ornament_to_cracker
  | _ => 0

-- Theorem for part (a)
theorem ten_sparklers_to_crackers :
  convert "sparkler" 10 = 32 := by sorry

-- Theorem for part (b)
theorem five_ornaments_one_cracker_more_valuable :
  convert "ornament" 5 + 1 > convert "sparkler" 2 := by sorry

end NUMINAMATH_CALUDE_ten_sparklers_to_crackers_five_ornaments_one_cracker_more_valuable_l2810_281081


namespace NUMINAMATH_CALUDE_pyramid_height_l2810_281098

/-- Given a square pyramid whose lateral faces unfold into a square with side length 18,
    prove that the height of the pyramid is 6. -/
theorem pyramid_height (s : ℝ) (h : s > 0) : 
  s * s = 18 * 18 / 2 → (6 : ℝ) * s = 18 * 18 / 2 := by
  sorry

#check pyramid_height

end NUMINAMATH_CALUDE_pyramid_height_l2810_281098


namespace NUMINAMATH_CALUDE_a_equals_five_l2810_281032

/-- Given the equation 632 - A9B = 41, where A and B are single digits, prove that A must equal 5. -/
theorem a_equals_five (A B : ℕ) (h1 : A ≤ 9) (h2 : B ≤ 9) (h3 : 632 - (100 * A + 10 * B) = 41) : A = 5 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_five_l2810_281032


namespace NUMINAMATH_CALUDE_repair_shop_earnings_121_l2810_281065

/-- Represents the earnings for a repair shop for a week. -/
def repair_shop_earnings (phone_cost laptop_cost computer_cost : ℕ) 
  (phone_repairs laptop_repairs computer_repairs : ℕ) : ℕ :=
  phone_cost * phone_repairs + laptop_cost * laptop_repairs + computer_cost * computer_repairs

/-- Theorem stating that the repair shop's earnings for the week is $121. -/
theorem repair_shop_earnings_121 :
  repair_shop_earnings 11 15 18 5 2 2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_repair_shop_earnings_121_l2810_281065


namespace NUMINAMATH_CALUDE_prob_irrational_not_adjacent_l2810_281094

/-- The number of rational terms in the expansion of (x + 2/√x)^6 -/
def num_rational_terms : ℕ := 4

/-- The number of irrational terms in the expansion of (x + 2/√x)^6 -/
def num_irrational_terms : ℕ := 3

/-- The total number of terms in the expansion -/
def total_terms : ℕ := num_rational_terms + num_irrational_terms

/-- The probability that irrational terms are not adjacent in the expansion of (x + 2/√x)^6 -/
theorem prob_irrational_not_adjacent : 
  (Nat.factorial num_rational_terms * (Nat.factorial (num_rational_terms + 1)) / 
   Nat.factorial num_irrational_terms) / 
  (Nat.factorial total_terms) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_irrational_not_adjacent_l2810_281094


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2810_281059

/-- Calculates the length of a bridge given train length, speed, and crossing time. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length = 148 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 227 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2810_281059


namespace NUMINAMATH_CALUDE_school_children_count_l2810_281091

/-- The actual number of children in the school -/
def actual_children : ℕ := 840

/-- The number of absent children -/
def absent_children : ℕ := 420

/-- The number of bananas each child gets initially -/
def initial_bananas : ℕ := 2

/-- The number of extra bananas each child gets due to absences -/
def extra_bananas : ℕ := 2

theorem school_children_count :
  ∀ (total_bananas : ℕ),
  total_bananas = initial_bananas * actual_children ∧
  total_bananas = (initial_bananas + extra_bananas) * (actual_children - absent_children) →
  actual_children = 840 := by
sorry

end NUMINAMATH_CALUDE_school_children_count_l2810_281091


namespace NUMINAMATH_CALUDE_ordering_of_trig_and_log_expressions_l2810_281025

theorem ordering_of_trig_and_log_expressions :
  let a := Real.sin (Real.cos 2)
  let b := Real.cos (Real.cos 2)
  let c := Real.log (Real.cos 1)
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ordering_of_trig_and_log_expressions_l2810_281025


namespace NUMINAMATH_CALUDE_specific_group_probability_l2810_281020

-- Define the number of students in the class
def n : ℕ := 32

-- Define the number of students chosen each day
def k : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of selecting a specific group
def probability : ℚ := 1 / (combination n k)

-- Theorem statement
theorem specific_group_probability :
  probability = 1 / 4960 := by
  sorry

end NUMINAMATH_CALUDE_specific_group_probability_l2810_281020


namespace NUMINAMATH_CALUDE_max_value_expression_l2810_281019

/-- The maximum value of (x + y) / z given the conditions -/
theorem max_value_expression (x y z : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ 
  y ≥ 10 ∧ y ≤ 99 ∧ 
  z ≥ 10 ∧ z ≤ 99 ∧ 
  (x + y + z) / 3 = 60 → 
  (x + y : ℚ) / z ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2810_281019


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l2810_281009

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  945 = 21 * q + r ∧ 
  q > 0 ∧ 
  r > 0 ∧ 
  ∀ (q' r' : ℕ), 945 = 21 * q' + r' ∧ q' > 0 ∧ r' > 0 → q - r ≥ q' - r' :=
by sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l2810_281009


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2810_281004

theorem sqrt_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 ∧ x = 1225 / 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2810_281004


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2810_281027

/-- Given that x² varies inversely with √w, prove that w = 1 when x = 6,
    given that x = 3 when w = 16. -/
theorem inverse_variation_problem (x w : ℝ) (k : ℝ) (h1 : x^2 * Real.sqrt w = k)
    (h2 : 3^2 * Real.sqrt 16 = k) (h3 : x = 6) : w = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2810_281027


namespace NUMINAMATH_CALUDE_inscribed_rectangle_delta_l2810_281001

/-- Triangle with side lengths a, b, c -/
structure Triangle (a b c : ℝ) where
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0

/-- Rectangle inscribed in a triangle -/
structure InscribedRectangle (T : Triangle a b c) where
  area : ℝ → ℝ  -- Area as a function of the rectangle's width

/-- The coefficient δ in the quadratic area formula of an inscribed rectangle -/
def delta (T : Triangle 15 39 36) (R : InscribedRectangle T) : ℚ :=
  60 / 169

theorem inscribed_rectangle_delta :
  ∀ (T : Triangle 15 39 36) (R : InscribedRectangle T),
  ∃ (γ : ℝ), ∀ (ω : ℝ), R.area ω = γ * ω - (delta T R : ℝ) * ω^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_delta_l2810_281001


namespace NUMINAMATH_CALUDE_unique_solution_for_absolute_value_equation_l2810_281045

theorem unique_solution_for_absolute_value_equation :
  ∃! x : ℤ, |x - 8 * (3 - 12)| - |5 - 11| = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_absolute_value_equation_l2810_281045


namespace NUMINAMATH_CALUDE_same_color_ratio_property_l2810_281046

/-- A coloring of natural numbers using 2017 colors -/
def Coloring := ℕ → Fin 2017

/-- The theorem stating that for any coloring of natural numbers using 2017 colors,
    there exist two natural numbers of the same color with a specific ratio property -/
theorem same_color_ratio_property (c : Coloring) :
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ c a = c b ∧ 
  ∃ (k : ℕ), k ≠ 0 ∧ b = k * a ∧ 2016 ∣ k := by
  sorry

end NUMINAMATH_CALUDE_same_color_ratio_property_l2810_281046


namespace NUMINAMATH_CALUDE_stellas_clocks_l2810_281050

/-- Stella's antique shop inventory problem -/
theorem stellas_clocks :
  ∀ (num_clocks : ℕ),
    (3 * 5 + num_clocks * 15 + 5 * 4 = 40 + 25) →
    num_clocks = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_stellas_clocks_l2810_281050


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l2810_281092

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the right branch of the hyperbola -/
def isOnRightBranch (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1 ∧ p.x > 0

/-- Checks if a point is on the given line -/
def isOnLine (p : Point) : Prop :=
  p.y = Real.sqrt 3 / 3 * p.x - 2

/-- The main theorem to be proved -/
theorem hyperbola_intersection_theorem (h : Hyperbola) 
    (hA : isOnRightBranch h A ∧ isOnLine A)
    (hB : isOnRightBranch h B ∧ isOnLine B)
    (hC : isOnRightBranch h C) :
    h.a = 2 * Real.sqrt 3 →
    h.b = Real.sqrt 3 →
    C.x = 4 * Real.sqrt 3 →
    C.y = 3 →
    ∃ m : ℝ, m = 4 ∧ A.x + B.x = m * C.x ∧ A.y + B.y = m * C.y := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l2810_281092


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2810_281057

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 3 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2810_281057


namespace NUMINAMATH_CALUDE_volume_ratio_specific_cone_l2810_281083

/-- Represents a right circular cone -/
structure Cone where
  base_diameter : ℝ
  height : ℝ

/-- Represents a plane intersecting the cone -/
structure IntersectingPlane where
  distance_from_apex : ℝ

/-- Calculates the volume ratio of the two parts resulting from intersecting a cone with a plane -/
def volume_ratio (cone : Cone) (plane : IntersectingPlane) : ℝ × ℝ :=
  sorry

/-- Theorem stating the volume ratio for the given cone and intersecting plane -/
theorem volume_ratio_specific_cone :
  let cone : Cone := { base_diameter := 26, height := 39 }
  let plane : IntersectingPlane := { distance_from_apex := 30 }
  volume_ratio cone plane = (0.4941, 0.5059) :=
sorry

end NUMINAMATH_CALUDE_volume_ratio_specific_cone_l2810_281083


namespace NUMINAMATH_CALUDE_cookout_kids_l2810_281016

/-- The number of kids at the cookout in 2004 -/
def kids_2004 : ℕ := 60

/-- The number of kids at the cookout in 2005 -/
def kids_2005 : ℕ := kids_2004 / 2

/-- The number of kids at the cookout in 2006 -/
def kids_2006 : ℕ := 20

/-- Theorem stating the relationship between the number of kids in different years -/
theorem cookout_kids : 
  (kids_2005 = kids_2004 / 2) ∧ 
  (kids_2006 = (2 * kids_2005) / 3) ∧ 
  (kids_2006 = 20) ∧ 
  (kids_2004 = 60) := by
  sorry

end NUMINAMATH_CALUDE_cookout_kids_l2810_281016


namespace NUMINAMATH_CALUDE_sum_of_common_elements_l2810_281024

-- Define the arithmetic progression
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

-- Define the geometric progression
def geometric_progression (k : ℕ) : ℕ := 10 * 2^k

-- Define the sequence of common elements
def common_elements (n : ℕ) : ℕ := 10 * 4^n

-- Theorem statement
theorem sum_of_common_elements : 
  (Finset.range 10).sum common_elements = 3495250 := by sorry

end NUMINAMATH_CALUDE_sum_of_common_elements_l2810_281024


namespace NUMINAMATH_CALUDE_uf_championship_ratio_l2810_281021

/-- The ratio of UF's points in the championship game to their average points per game -/
theorem uf_championship_ratio : 
  ∀ (total_points : ℕ) (num_games : ℕ) (opponent_points : ℕ) (win_margin : ℕ),
    total_points = 720 →
    num_games = 24 →
    opponent_points = 11 →
    win_margin = 2 →
    (opponent_points + win_margin : ℚ) / (total_points / num_games : ℚ) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_uf_championship_ratio_l2810_281021


namespace NUMINAMATH_CALUDE_solution_set_is_x_gt_one_l2810_281014

/-- A linear function y = kx + b with a table of x and y values -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0
  x_values : List ℝ := [-2, -1, 0, 1, 2, 3]
  y_values : List ℝ := [3, 2, 1, 0, -1, -2]
  table_valid : x_values.length = y_values.length

/-- The solution set of kx + b < 0 for the given linear function -/
def solutionSet (f : LinearFunction) : Set ℝ :=
  {x | f.k * x + f.b < 0}

/-- Theorem stating that the solution set is x > 1 -/
theorem solution_set_is_x_gt_one (f : LinearFunction) : 
  solutionSet f = {x | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_x_gt_one_l2810_281014


namespace NUMINAMATH_CALUDE_base7_product_digit_sum_l2810_281054

/-- Converts a base 7 number to base 10 -/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem base7_product_digit_sum :
  let a := 35  -- 35 in base 7
  let b := 42  -- 42 in base 7
  let product := (toBase10 a) * (toBase10 b)
  sumOfDigits (toBase7 product) = 18 := by sorry

end NUMINAMATH_CALUDE_base7_product_digit_sum_l2810_281054


namespace NUMINAMATH_CALUDE_new_person_weight_is_109_5_l2810_281063

/-- Calculates the weight of a new person in a group when the average weight changes --/
def newPersonWeight (numPersons : ℕ) (avgWeightIncrease : ℝ) (oldPersonWeight : ℝ) : ℝ :=
  oldPersonWeight + numPersons * avgWeightIncrease

/-- Theorem: The weight of the new person is 109.5 kg --/
theorem new_person_weight_is_109_5 :
  newPersonWeight 15 2.3 75 = 109.5 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_109_5_l2810_281063


namespace NUMINAMATH_CALUDE_tip_percentage_is_ten_percent_l2810_281023

/-- Calculates the tip percentage given the total bill, number of people, and amount paid per person. -/
def calculate_tip_percentage (total_bill : ℚ) (num_people : ℕ) (amount_per_person : ℚ) : ℚ :=
  let total_paid := num_people * amount_per_person
  let tip_amount := total_paid - total_bill
  (tip_amount / total_bill) * 100

/-- Proves that for a bill of $139.00 split among 8 people, if each pays $19.1125, the tip is 10%. -/
theorem tip_percentage_is_ten_percent :
  calculate_tip_percentage 139 8 (19 + 9/80) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_is_ten_percent_l2810_281023


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2810_281089

/-- Given an arithmetic sequence {a_n} where a_2 = 10 and a_4 = 18, 
    the common difference d equals 4. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- a is a sequence of real numbers indexed by natural numbers
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- a is arithmetic
  (h_a2 : a 2 = 10) -- a_2 = 10
  (h_a4 : a 4 = 18) -- a_4 = 18
  : a 3 - a 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2810_281089


namespace NUMINAMATH_CALUDE_complex_modulus_example_l2810_281079

theorem complex_modulus_example : Complex.abs (7/4 + 3*I) = Real.sqrt 193 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l2810_281079


namespace NUMINAMATH_CALUDE_quadratic_function_sum_of_coefficients_l2810_281052

theorem quadratic_function_sum_of_coefficients 
  (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : (1 : ℝ) = a * (1 : ℝ)^2 + b * (1 : ℝ) - 1) : 
  a + b = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_sum_of_coefficients_l2810_281052


namespace NUMINAMATH_CALUDE_volume_of_solid_is_62pi_over_3_l2810_281029

/-- The region S in the coordinate plane -/
def region_S : Set (ℝ × ℝ) :=
  {p | p.2 ≤ p.1 + 2 ∧ p.2 ≤ -p.1 + 6 ∧ p.2 ≤ 4}

/-- The volume of the solid formed by revolving region S around the y-axis -/
noncomputable def volume_of_solid : ℝ := sorry

/-- Theorem stating that the volume of the solid is 62π/3 -/
theorem volume_of_solid_is_62pi_over_3 :
  volume_of_solid = 62 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_volume_of_solid_is_62pi_over_3_l2810_281029


namespace NUMINAMATH_CALUDE_parallel_line_equation_l2810_281047

/-- A line in the Cartesian coordinate system -/
structure CartesianLine where
  slope : ℝ
  y_intercept : ℝ

/-- The equation of a line given its slope and y-intercept -/
def line_equation (l : CartesianLine) (x : ℝ) : ℝ :=
  l.slope * x + l.y_intercept

theorem parallel_line_equation 
  (l : CartesianLine) 
  (h1 : l.slope = -2) 
  (h2 : l.y_intercept = -3) : 
  ∀ x, line_equation l x = -2 * x - 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l2810_281047


namespace NUMINAMATH_CALUDE_plane_train_speed_ratio_l2810_281077

/-- The ratio of plane speed to train speed given specific travel conditions -/
theorem plane_train_speed_ratio :
  -- Train travel time
  ∀ (train_time : ℝ) (plane_time : ℝ) (wait_time : ℝ) (meet_time : ℝ),
  train_time = 20 →
  -- Plane travel time (including waiting)
  plane_time = 10 →
  -- Waiting time is more than 5 hours after train departure
  wait_time > 5 →
  -- Plane is above train 8/9 hours after departure
  meet_time = 8/9 →
  -- At that point, plane and train have traveled the same distance
  ∃ (train_speed : ℝ) (plane_speed : ℝ),
    train_speed * (wait_time + meet_time) = plane_speed * meet_time →
    train_speed * train_time = plane_speed * (plane_time - wait_time) →
    -- The ratio of plane speed to train speed is 5.75
    plane_speed / train_speed = 5.75 := by
  sorry

end NUMINAMATH_CALUDE_plane_train_speed_ratio_l2810_281077


namespace NUMINAMATH_CALUDE_equationA_is_linear_l2810_281000

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and at least one of a or b is non-zero. --/
def IsLinearInTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation x/2 + 3y = 2 --/
def EquationA (x y : ℝ) : Prop := x/2 + 3*y = 2

theorem equationA_is_linear : IsLinearInTwoVariables EquationA := by
  sorry


end NUMINAMATH_CALUDE_equationA_is_linear_l2810_281000


namespace NUMINAMATH_CALUDE_exactly_one_approve_probability_l2810_281071

def p_approve : ℝ := 0.7

def p_exactly_one_approve : ℝ :=
  3 * p_approve * (1 - p_approve) * (1 - p_approve)

theorem exactly_one_approve_probability :
  p_exactly_one_approve = 0.189 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_approve_probability_l2810_281071


namespace NUMINAMATH_CALUDE_evaluate_expression_l2810_281099

theorem evaluate_expression : 5^2 + 2*(5 - 2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2810_281099


namespace NUMINAMATH_CALUDE_simultaneous_cycle_is_twenty_l2810_281015

/-- The length of the letter sequence -/
def letter_cycle_length : ℕ := 5

/-- The length of the digit sequence -/
def digit_cycle_length : ℕ := 4

/-- The number of cycles needed for both sequences to return to their original state simultaneously -/
def simultaneous_cycle : ℕ := Nat.lcm letter_cycle_length digit_cycle_length

theorem simultaneous_cycle_is_twenty : simultaneous_cycle = 20 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_cycle_is_twenty_l2810_281015


namespace NUMINAMATH_CALUDE_sum_reciprocal_complements_l2810_281010

theorem sum_reciprocal_complements (a b c d : ℝ) 
  (h1 : a + b + c + d = 2) 
  (h2 : 1/a + 1/b + 1/c + 1/d = 2) : 
  1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_complements_l2810_281010


namespace NUMINAMATH_CALUDE_total_oranges_in_boxes_l2810_281039

def box1_capacity : ℕ := 80
def box2_capacity : ℕ := 50
def box1_fill_ratio : ℚ := 3/4
def box2_fill_ratio : ℚ := 3/5

theorem total_oranges_in_boxes :
  (↑box1_capacity * box1_fill_ratio).floor + (↑box2_capacity * box2_fill_ratio).floor = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_in_boxes_l2810_281039


namespace NUMINAMATH_CALUDE_solution_to_equation_l2810_281037

theorem solution_to_equation : ∃ x y : ℤ, x - 3 * y = 1 ∧ x = -2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2810_281037


namespace NUMINAMATH_CALUDE_area_of_region_t_l2810_281036

/-- A rhombus with side length 3 and one right angle -/
structure RightRhombus where
  side_length : ℝ
  angle_q : ℝ
  side_length_eq : side_length = 3
  angle_q_eq : angle_q = 90

/-- The region T inside the rhombus -/
def region_t (r : RightRhombus) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the area of region T is 2.25 -/
theorem area_of_region_t (r : RightRhombus) : area (region_t r) = 2.25 := by sorry

end NUMINAMATH_CALUDE_area_of_region_t_l2810_281036


namespace NUMINAMATH_CALUDE_circle_M_equations_l2810_281072

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center of circle M lies in part (I)
def centerLine (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the equation of circle M for part (I)
def circleM1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 25

-- Define the equation of circle M for part (II)
def circleM2 (x y : ℝ) : Prop := (x + 7/2)^2 + (y - 1/2)^2 = 25/2

theorem circle_M_equations :
  (∀ x y : ℝ, (∃ x0 y0 : ℝ, circle1 x0 y0 ∧ circle2 x0 y0 ∧ circleM1 x0 y0) →
    (∃ xc yc : ℝ, centerLine xc yc ∧ circleM1 x y)) ∧
  (∀ x y : ℝ, (∃ x0 y0 : ℝ, circle1 x0 y0 ∧ circle2 x0 y0 ∧ circleM2 x0 y0) →
    circleM2 x y) :=
by sorry

end NUMINAMATH_CALUDE_circle_M_equations_l2810_281072


namespace NUMINAMATH_CALUDE_f_monotonicity_and_roots_l2810_281031

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := 2 * x + 1 - Real.exp (a * x)

theorem f_monotonicity_and_roots :
  (∀ x y : ℝ, x < y → a ≤ 0 → f a x < f a y) ∧
  (a > 0 →
    (∀ x y : ℝ, x < y → x < (1/a) * Real.log (2/a) → f a x < f a y) ∧
    (∀ x y : ℝ, x < y → x > (1/a) * Real.log (2/a) → f a x > f a y)) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f a x₁ = 1 → f a x₂ = 1 → x₁ + x₂ > 2/a) :=
by sorry

end

end NUMINAMATH_CALUDE_f_monotonicity_and_roots_l2810_281031


namespace NUMINAMATH_CALUDE_fair_die_probability_at_least_one_six_l2810_281044

theorem fair_die_probability_at_least_one_six (n : ℕ) (p : ℚ) : 
  n = 3 → p = 1/6 → (1 : ℚ) - (1 - p)^n = 91/216 := by
  sorry

end NUMINAMATH_CALUDE_fair_die_probability_at_least_one_six_l2810_281044


namespace NUMINAMATH_CALUDE_ball_transfer_probability_l2810_281062

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  red : ℕ

/-- The probability of drawing a red ball from a bag -/
def redProbability (bag : Bag) : ℚ :=
  bag.red / (bag.white + bag.red)

/-- The probability of drawing a white ball from a bag -/
def whiteProbability (bag : Bag) : ℚ :=
  bag.white / (bag.white + bag.red)

/-- The probability of drawing a red ball from the second bag
    after transferring a ball from the first bag -/
def transferAndDrawRed (bagA bagB : Bag) : ℚ :=
  (redProbability bagA) * (redProbability (Bag.mk bagB.white (bagB.red + 1))) +
  (whiteProbability bagA) * (redProbability (Bag.mk (bagB.white + 1) bagB.red))

theorem ball_transfer_probability :
  let bagA : Bag := ⟨2, 3⟩
  let bagB : Bag := ⟨1, 2⟩
  transferAndDrawRed bagA bagB = 13 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ball_transfer_probability_l2810_281062


namespace NUMINAMATH_CALUDE_find_a_minus_b_l2810_281082

-- Define the functions
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 7
def h (a b x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem find_a_minus_b (a b : ℝ) :
  (∀ x, h a b x = x - 9) → a - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_a_minus_b_l2810_281082


namespace NUMINAMATH_CALUDE_solution_for_k_3_solution_for_k_neg_2_solution_for_k_lt_neg_2_solution_for_k_between_neg_2_and_0_l2810_281069

-- Define the inequality
def inequality (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 + (k - 2) * x - 2 < 0

-- Theorem for k = 3
theorem solution_for_k_3 :
  ∀ x : ℝ, inequality 3 x ↔ -1 < x ∧ x < 2/3 :=
sorry

-- Theorems for k < 0
theorem solution_for_k_neg_2 :
  ∀ x : ℝ, inequality (-2) x ↔ x ≠ -1 :=
sorry

theorem solution_for_k_lt_neg_2 :
  ∀ k x : ℝ, k < -2 → (inequality k x ↔ x < -1 ∨ x > 2/k) :=
sorry

theorem solution_for_k_between_neg_2_and_0 :
  ∀ k x : ℝ, -2 < k ∧ k < 0 → (inequality k x ↔ x > -1 ∨ x < 2/k) :=
sorry

end NUMINAMATH_CALUDE_solution_for_k_3_solution_for_k_neg_2_solution_for_k_lt_neg_2_solution_for_k_between_neg_2_and_0_l2810_281069


namespace NUMINAMATH_CALUDE_two_digit_numbers_exist_l2810_281055

/-- Two-digit positive integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := (n / 10) + (n % 10)

/-- Check if two numbers share a digit -/
def shareDigit (a b : ℕ) : Prop :=
  (a / 10 = b / 10) ∨ (a / 10 = b % 10) ∨ (a % 10 = b / 10) ∨ (a % 10 = b % 10)

theorem two_digit_numbers_exist : ∃ (a b : ℕ), 
  TwoDigitInt a ∧ TwoDigitInt b ∧
  a = b + 14 ∧
  shareDigit a b ∧
  sumOfDigits a = 2 * sumOfDigits b ∧
  ((a = 37 ∧ b = 23) ∨ (a = 31 ∧ b = 17)) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_exist_l2810_281055


namespace NUMINAMATH_CALUDE_third_box_weight_l2810_281093

/-- Given two boxes with weights and their weight difference, prove the weight of the third box -/
theorem third_box_weight (weight_first : ℝ) (weight_diff : ℝ) : 
  weight_first = 2 → weight_diff = 11 → weight_first + weight_diff = 13 := by
  sorry

end NUMINAMATH_CALUDE_third_box_weight_l2810_281093


namespace NUMINAMATH_CALUDE_probability_of_white_and_black_l2810_281028

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The number of black balls in the bag -/
def num_black : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_white + num_black

/-- The number of balls drawn -/
def drawn : ℕ := 2

/-- The probability of drawing one white ball and one black ball -/
def prob_white_and_black : ℚ := 2 / 3

theorem probability_of_white_and_black :
  (num_white * num_black : ℚ) / (total_balls.choose drawn) = prob_white_and_black := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_and_black_l2810_281028


namespace NUMINAMATH_CALUDE_sum_of_periodic_functions_periodicity_l2810_281026

/-- A periodic function with period T -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- A function with smallest positive period T -/
def HasSmallestPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  IsPeriodic f T ∧ ∀ S, 0 < S → S < T → ¬IsPeriodic f S

/-- The main theorem about the sum of two periodic functions -/
theorem sum_of_periodic_functions_periodicity
  (f₁ f₂ : ℝ → ℝ) (T : ℝ) (hT : T > 0)
  (h₁ : HasSmallestPeriod f₁ T) (h₂ : HasSmallestPeriod f₂ T) :
  ∃ y : ℝ → ℝ, (y = f₁ + f₂) ∧ 
  IsPeriodic y T ∧ 
  ¬(∃ S : ℝ, HasSmallestPeriod y S) :=
sorry

end NUMINAMATH_CALUDE_sum_of_periodic_functions_periodicity_l2810_281026


namespace NUMINAMATH_CALUDE_shoe_selection_probability_l2810_281075

/-- The number of pairs of shoes in the cabinet -/
def num_pairs : ℕ := 3

/-- The total number of shoes in the cabinet -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes selected -/
def selected_shoes : ℕ := 2

/-- The probability of selecting two shoes that do not form a pair -/
def prob_not_pair : ℚ := 4/5

theorem shoe_selection_probability :
  (Nat.choose total_shoes selected_shoes - num_pairs) / Nat.choose total_shoes selected_shoes = prob_not_pair :=
sorry

end NUMINAMATH_CALUDE_shoe_selection_probability_l2810_281075


namespace NUMINAMATH_CALUDE_spade_calculation_l2810_281013

-- Define the spade operation
def spade (x y : ℝ) : ℝ := (x + y + 1) * (x - y)

-- Theorem statement
theorem spade_calculation : spade 2 (spade 3 6) = -864 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l2810_281013


namespace NUMINAMATH_CALUDE_expression_evaluation_l2810_281096

theorem expression_evaluation : -2^3 + |2 - 3| - 2 * (-1)^2023 = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2810_281096


namespace NUMINAMATH_CALUDE_concentric_circles_intersection_l2810_281002

theorem concentric_circles_intersection (r_outer r_inner : ℝ) (h_outer : r_outer * 2 * Real.pi = 24 * Real.pi) (h_inner : r_inner * 2 * Real.pi = 14 * Real.pi) : r_outer - r_inner = 5 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_intersection_l2810_281002


namespace NUMINAMATH_CALUDE_three_color_theorem_min_three_colors_min_colors_is_three_l2810_281034

/-- Represents a 3D coordinate in the 3x3x3 grid --/
structure Coord where
  x : Fin 3
  y : Fin 3
  z : Fin 3

/-- Represents a coloring of the 3x3x3 grid --/
def Coloring := Coord → Fin 3

/-- Two coordinates are adjacent if they differ by 1 in exactly one dimension --/
def adjacent (c1 c2 : Coord) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ c1.z.val + 1 = c2.z.val) ∨
  (c1.x = c2.x ∧ c1.y = c2.y ∧ c1.z.val = c2.z.val + 1) ∨
  (c1.x = c2.x ∧ c1.y.val + 1 = c2.y.val ∧ c1.z = c2.z) ∨
  (c1.x = c2.x ∧ c1.y.val = c2.y.val + 1 ∧ c1.z = c2.z) ∨
  (c1.x.val + 1 = c2.x.val ∧ c1.y = c2.y ∧ c1.z = c2.z) ∨
  (c1.x.val = c2.x.val + 1 ∧ c1.y = c2.y ∧ c1.z = c2.z)

/-- A coloring is valid if no adjacent cubes have the same color --/
def validColoring (c : Coloring) : Prop :=
  ∀ c1 c2 : Coord, adjacent c1 c2 → c c1 ≠ c c2

/-- There exists a valid coloring using only 3 colors --/
theorem three_color_theorem : ∃ c : Coloring, validColoring c :=
  sorry

/-- Any valid coloring must use at least 3 colors --/
theorem min_three_colors (c : Coloring) (h : validColoring c) :
  ∃ c1 c2 c3 : Coord, c c1 ≠ c c2 ∧ c c2 ≠ c c3 ∧ c c1 ≠ c c3 :=
  sorry

/-- The minimum number of colors needed is exactly 3 --/
theorem min_colors_is_three :
  (∃ c : Coloring, validColoring c) ∧
  (∀ c : Coloring, validColoring c →
    ∃ c1 c2 c3 : Coord, c c1 ≠ c c2 ∧ c c2 ≠ c c3 ∧ c c1 ≠ c c3) :=
  sorry

end NUMINAMATH_CALUDE_three_color_theorem_min_three_colors_min_colors_is_three_l2810_281034


namespace NUMINAMATH_CALUDE_max_visible_cube_l2810_281061

/-- The size of the cube's edge -/
def n : ℕ := 13

/-- The number of unit cubes visible on one face -/
def face_visible : ℕ := n^2

/-- The number of unit cubes visible along one edge (excluding the corner) -/
def edge_visible : ℕ := n - 1

/-- The maximum number of unit cubes visible from a single point -/
def max_visible : ℕ := 3 * face_visible - 3 * edge_visible + 1

theorem max_visible_cube :
  max_visible = 472 :=
sorry

end NUMINAMATH_CALUDE_max_visible_cube_l2810_281061


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2810_281088

theorem expression_simplification_and_evaluation :
  ∀ a : ℤ, -1 < a ∧ a < Real.sqrt 5 ∧ a ≠ 0 ∧ a ≠ 1 →
  let expr := ((a + 1) / (2 * a - 2) - 5 / (2 * a^2 - 2) - (a + 3) / (2 * a + 2)) / (a^2 / (a^2 - 1))
  expr = -1 / (2 * a^2) ∧
  (a = 2 → expr = -1/8) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2810_281088


namespace NUMINAMATH_CALUDE_compound_interest_theorem_specific_case_calculation_l2810_281084

/-- Compound interest calculation function -/
def compound_interest (a : ℝ) (r : ℝ) (x : ℕ) : ℝ :=
  a * (1 + r) ^ x

/-- Theorem for compound interest calculation -/
theorem compound_interest_theorem (a r : ℝ) (x : ℕ) :
  compound_interest a r x = a * (1 + r) ^ x :=
by sorry

/-- Specific case calculation -/
theorem specific_case_calculation :
  let a : ℝ := 1000
  let r : ℝ := 0.0225
  let x : ℕ := 4
  abs (compound_interest a r x - 1093.08) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_theorem_specific_case_calculation_l2810_281084


namespace NUMINAMATH_CALUDE_equation_solution_l2810_281086

theorem equation_solution :
  ∃! x : ℝ, x ≠ 1 ∧ (x / (x - 1) - 1 = 1) :=
by
  use 2
  constructor
  · constructor
    · norm_num
    · field_simp
      ring
  · intro y hy
    have h1 : y ≠ 1 := hy.1
    have h2 : y / (y - 1) - 1 = 1 := hy.2
    -- Proof steps would go here
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2810_281086


namespace NUMINAMATH_CALUDE_final_result_calculation_l2810_281035

theorem final_result_calculation (chosen_number : ℤ) : 
  chosen_number = 120 → (chosen_number / 6 - 15 : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_final_result_calculation_l2810_281035


namespace NUMINAMATH_CALUDE_quadratic_equation_property_l2810_281095

/-- 
A quadratic equation with coefficients a, b, and c, where a ≠ 0,
satisfying a + b + c = 0 and having two equal real roots.
-/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  sum_zero : a + b + c = 0
  equal_roots : ∃ x : ℝ, ∀ y : ℝ, a * y^2 + b * y + c = 0 ↔ y = x

theorem quadratic_equation_property (eq : QuadraticEquation) : eq.a = eq.c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_property_l2810_281095


namespace NUMINAMATH_CALUDE_cubic_inequality_l2810_281078

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 > -36*x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2810_281078


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2810_281018

theorem fraction_equation_solution :
  ∃! x : ℚ, (x + 2) / (x - 3) = (x - 4) / (x + 5) :=
by
  use 1/7
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2810_281018


namespace NUMINAMATH_CALUDE_triangle_problem_l2810_281022

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (4 * a * Real.cos B = c^2 - 4 * b * Real.cos A) →
  (C = π / 3) →
  (a + b = 4 * Real.sqrt 2) →
  -- Conclusions
  (c = 4) ∧
  (1/2 * a * b * Real.sin C = (4 * Real.sqrt 3) / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2810_281022


namespace NUMINAMATH_CALUDE_ellipse_and_triangle_area_l2810_281012

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0

-- Define the inscribed circle
def inscribed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

-- Define the parabola E
def parabola_E (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ p > 0

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y = x + m ∧ 0 ≤ m ∧ m ≤ 1

-- State the theorem
theorem ellipse_and_triangle_area :
  ∀ (a b c p m : ℝ) (x y : ℝ),
  ellipse_C a b x y →
  inscribed_circle x y →
  parabola_E p x y →
  line_l m x y →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧ parabola_E p x₁ y₁ ∧ parabola_E p x₂ y₂) →
  (∃ (F : ℝ × ℝ), F.1 = c ∧ F.2 = 0 ∧ c^2 = a^2 - b^2) →
  (b = c) →
  (a^2 = 8 ∧ b^2 = 4) ∧
  (∃ (S : ℝ), S = (32 * Real.sqrt 6) / 9 ∧
    ∀ (S' : ℝ), S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_triangle_area_l2810_281012
