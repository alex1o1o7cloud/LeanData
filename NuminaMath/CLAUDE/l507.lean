import Mathlib

namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_B_subset_A_iff_a_in_range_l507_50745

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - a| ≤ 4}
def B : Set ℝ := {x : ℝ | (x - 2) * (x - 3) ≤ 0}

-- Theorem 1: A ∩ B = ∅ if and only if a ∈ (-∞, -2) ∪ (7, +∞)
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a < -2 ∨ a > 7 := by sorry

-- Theorem 2: B ⊆ A if and only if a ∈ [1, 6]
theorem B_subset_A_iff_a_in_range (a : ℝ) :
  B ⊆ A a ↔ 1 ≤ a ∧ a ≤ 6 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_B_subset_A_iff_a_in_range_l507_50745


namespace NUMINAMATH_CALUDE_diamonds_count_l507_50719

/-- Represents the number of gems in a treasure chest. -/
def total_gems : ℕ := 5155

/-- Represents the number of rubies in the treasure chest. -/
def rubies : ℕ := 5110

/-- Theorem stating that the number of diamonds in the treasure chest is 45. -/
theorem diamonds_count : total_gems - rubies = 45 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_count_l507_50719


namespace NUMINAMATH_CALUDE_correct_calculation_l507_50798

theorem correct_calculation (a b : ℝ) (h : a ≠ 0) : (a^2 + a*b) / a = a + b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l507_50798


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l507_50780

theorem quadratic_solution_sum (a b : ℕ+) : 
  (∃ x : ℝ, x^2 + 16*x = 96 ∧ x = Real.sqrt a - b) → a + b = 168 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l507_50780


namespace NUMINAMATH_CALUDE_inequality_proof_l507_50775

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) * (b + c) * (c + d) * (d + a) * (1 + (a * b * c * d) ^ (1/4)) ^ 4 ≥ 
  16 * a * b * c * d * (1 + a) * (1 + b) * (1 + c) * (1 + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l507_50775


namespace NUMINAMATH_CALUDE_trigonometric_identity_l507_50763

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : 0 < a) (k : 0 < b) 
  (hyp : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) : 
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l507_50763


namespace NUMINAMATH_CALUDE_double_inequality_proof_l507_50754

theorem double_inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (0 < 1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1))) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) ≤ 1 / 8) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) = 1 / 8 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_double_inequality_proof_l507_50754


namespace NUMINAMATH_CALUDE_set_A_equals_neg_one_zero_l507_50783

def A : Set ℤ := {x | x^2 + x ≤ 0}

theorem set_A_equals_neg_one_zero : A = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_set_A_equals_neg_one_zero_l507_50783


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l507_50729

-- Define the triangle
def triangle_side_1 : ℝ := 9
def triangle_side_2 : ℝ := 12
def triangle_hypotenuse : ℝ := 15

-- Define the rectangle
def rectangle_length : ℝ := 6

-- Theorem statement
theorem rectangle_perimeter : 
  -- Right triangle condition
  triangle_side_1^2 + triangle_side_2^2 = triangle_hypotenuse^2 →
  -- Rectangle area equals triangle area
  (1/2 * triangle_side_1 * triangle_side_2) = (rectangle_length * (1/2 * triangle_side_1 * triangle_side_2 / rectangle_length)) →
  -- Perimeter of the rectangle is 30
  2 * (rectangle_length + (1/2 * triangle_side_1 * triangle_side_2 / rectangle_length)) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l507_50729


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l507_50715

theorem max_value_complex_expression (x y : ℂ) :
  ∃ (M : ℝ), M = (5 * Real.sqrt 2) / 2 ∧
  Complex.abs (3 * x + 4 * y) / Real.sqrt (Complex.abs x ^ 2 + Complex.abs y ^ 2 + Complex.abs (x ^ 2 + y ^ 2)) ≤ M ∧
  ∃ (x₀ y₀ : ℂ), Complex.abs (3 * x₀ + 4 * y₀) / Real.sqrt (Complex.abs x₀ ^ 2 + Complex.abs y₀ ^ 2 + Complex.abs (x₀ ^ 2 + y₀ ^ 2)) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l507_50715


namespace NUMINAMATH_CALUDE_profit_percentage_is_20_percent_l507_50724

def selling_price : ℚ := 1170
def cost_price : ℚ := 975

theorem profit_percentage_is_20_percent :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_20_percent_l507_50724


namespace NUMINAMATH_CALUDE_jogger_difference_l507_50781

/-- The number of joggers bought by each person -/
structure Joggers where
  tyson : ℕ
  alexander : ℕ
  christopher : ℕ

/-- The conditions of the problem -/
def jogger_problem (j : Joggers) : Prop :=
  j.christopher = 20 * j.tyson ∧
  j.christopher = 80 ∧
  j.alexander = j.tyson + 22

/-- The theorem to prove -/
theorem jogger_difference (j : Joggers) (h : jogger_problem j) :
  j.christopher - j.alexander = 54 := by
  sorry

end NUMINAMATH_CALUDE_jogger_difference_l507_50781


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_4_10_l507_50795

theorem gcf_lcm_sum_4_10 : Nat.gcd 4 10 + Nat.lcm 4 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_4_10_l507_50795


namespace NUMINAMATH_CALUDE_roller_coaster_probability_l507_50721

/-- The number of cars in the roller coaster -/
def num_cars : ℕ := 4

/-- The number of times the passenger rides the roller coaster -/
def num_rides : ℕ := 3

/-- The probability of choosing a different car on the second ride -/
def prob_second_ride : ℚ := 3 / 4

/-- The probability of choosing a different car on the third ride -/
def prob_third_ride : ℚ := 1 / 2

/-- The probability of riding in 3 different cars over 3 rides -/
def prob_three_different_cars : ℚ := 3 / 8

theorem roller_coaster_probability :
  prob_three_different_cars = 1 * prob_second_ride * prob_third_ride :=
sorry

end NUMINAMATH_CALUDE_roller_coaster_probability_l507_50721


namespace NUMINAMATH_CALUDE_probability_is_correct_l507_50794

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def needed_stickers : ℕ := 6
def collected_stickers : ℕ := total_stickers - needed_stickers

def probability_complete_collection : ℚ :=
  (Nat.choose needed_stickers needed_stickers * Nat.choose collected_stickers (selected_stickers - needed_stickers)) /
  Nat.choose total_stickers selected_stickers

theorem probability_is_correct : probability_complete_collection = 5 / 442 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_correct_l507_50794


namespace NUMINAMATH_CALUDE_apple_cost_price_l507_50742

/-- 
Given:
- The selling price of an apple is 18
- The seller loses 1/6th of the cost price
Prove that the cost price is 21.6
-/
theorem apple_cost_price (selling_price : ℝ) (loss_fraction : ℝ) (cost_price : ℝ) : 
  selling_price = 18 → 
  loss_fraction = 1/6 → 
  selling_price = cost_price * (1 - loss_fraction) →
  cost_price = 21.6 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_price_l507_50742


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l507_50746

theorem square_of_real_not_always_positive : 
  ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l507_50746


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l507_50731

/-- Given a geometric sequence {a_n} where 2a₁, (3/2)a₂, a₃ form an arithmetic sequence,
    prove that the common ratio of the geometric sequence is either 1 or 2. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- {a_n} is a geometric sequence
  (2 * a 1 - (3/2 * a 2) = (3/2 * a 2) - a 3) →  -- 2a₁, (3/2)a₂, a₃ form an arithmetic sequence
  (a 2 / a 1 = 1 ∨ a 2 / a 1 = 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l507_50731


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l507_50779

theorem easter_egg_distribution (red_total orange_total : ℕ) 
  (h_red : red_total = 20) (h_orange : orange_total = 30) : ∃ (eggs_per_basket : ℕ), 
  eggs_per_basket ≥ 5 ∧ 
  red_total % eggs_per_basket = 0 ∧ 
  orange_total % eggs_per_basket = 0 ∧
  ∀ (n : ℕ), n ≥ 5 ∧ red_total % n = 0 ∧ orange_total % n = 0 → n ≥ eggs_per_basket :=
by
  sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l507_50779


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l507_50785

def is_divisible_by_11 (n : ℕ) : Prop :=
  ∃ k : ℤ, (945 * 10000 + n * 1000 + 631) = 11 * k

theorem seven_digit_divisible_by_11 (n : ℕ) (h : n < 10) :
  is_divisible_by_11 n → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l507_50785


namespace NUMINAMATH_CALUDE_time_to_top_floor_l507_50732

/-- The number of floors in the building -/
def num_floors : ℕ := 10

/-- The time in seconds to go up to an even-numbered floor -/
def even_floor_time : ℕ := 15

/-- The time in seconds to go up to an odd-numbered floor -/
def odd_floor_time : ℕ := 9

/-- The number of even-numbered floors -/
def num_even_floors : ℕ := num_floors / 2

/-- The number of odd-numbered floors -/
def num_odd_floors : ℕ := (num_floors + 1) / 2

/-- The total time in seconds to reach the top floor -/
def total_time_seconds : ℕ := num_even_floors * even_floor_time + num_odd_floors * odd_floor_time

/-- Conversion factor from seconds to minutes -/
def seconds_per_minute : ℕ := 60

theorem time_to_top_floor :
  total_time_seconds / seconds_per_minute = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_to_top_floor_l507_50732


namespace NUMINAMATH_CALUDE_complex_equality_sum_l507_50788

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem complex_equality_sum (a b : ℝ) (h : (1 + i) * i = a + b * i) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_sum_l507_50788


namespace NUMINAMATH_CALUDE_smallest_a1_l507_50720

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 13 * a (n - 1) - 2 * n)

/-- The theorem stating the smallest possible value of a₁ -/
theorem smallest_a1 (a : ℕ → ℝ) (h : RecurrenceSequence a) :
  ∀ a₁ : ℝ, a 1 = a₁ → a₁ ≥ 13 / 36 :=
sorry

end NUMINAMATH_CALUDE_smallest_a1_l507_50720


namespace NUMINAMATH_CALUDE_solve_for_d_l507_50723

theorem solve_for_d (c a m d : ℝ) (h : m = (c * a * d) / (a - d)) : 
  d = (m * a) / (m + c * a) := by
sorry

end NUMINAMATH_CALUDE_solve_for_d_l507_50723


namespace NUMINAMATH_CALUDE_least_six_digit_congruent_to_seven_mod_seventeen_l507_50703

theorem least_six_digit_congruent_to_seven_mod_seventeen :
  ∃ (n : ℕ), 
    n = 100008 ∧ 
    n ≥ 100000 ∧ 
    n < 1000000 ∧
    n % 17 = 7 ∧
    ∀ (m : ℕ), m ≥ 100000 ∧ m < 1000000 ∧ m % 17 = 7 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_six_digit_congruent_to_seven_mod_seventeen_l507_50703


namespace NUMINAMATH_CALUDE_fruit_remaining_l507_50736

-- Define the quantities of fruits picked and eaten
def mike_apples : ℝ := 7.0
def nancy_apples : ℝ := 3.0
def john_apples : ℝ := 5.0
def keith_apples : ℝ := 6.0
def lisa_apples : ℝ := 2.0
def oranges_picked_and_eaten : ℝ := 8.0
def cherries_picked_and_eaten : ℝ := 4.0

-- Define the total apples picked and eaten
def total_apples_picked : ℝ := mike_apples + nancy_apples + john_apples
def total_apples_eaten : ℝ := keith_apples + lisa_apples

-- Theorem statement
theorem fruit_remaining :
  (total_apples_picked - total_apples_eaten = 7.0) ∧
  (oranges_picked_and_eaten - oranges_picked_and_eaten = 0) ∧
  (cherries_picked_and_eaten - cherries_picked_and_eaten = 0) :=
by sorry

end NUMINAMATH_CALUDE_fruit_remaining_l507_50736


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l507_50708

theorem fraction_equation_solution (n : ℚ) :
  (1 : ℚ) / (n + 1) + (2 : ℚ) / (n + 1) + n / (n + 1) = 4 → n = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l507_50708


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_ribbons_l507_50796

/-- The lengths of Amanda's ribbons in inches -/
def ribbon_lengths : List ℕ := [8, 16, 20, 28]

/-- A function to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- The theorem stating that 2 is the largest prime that divides all ribbon lengths -/
theorem largest_prime_divisor_of_ribbons :
  ∃ (p : ℕ), is_prime p ∧ 
    (∀ (length : ℕ), length ∈ ribbon_lengths → p ∣ length) ∧
    (∀ (q : ℕ), is_prime q → 
      (∀ (length : ℕ), length ∈ ribbon_lengths → q ∣ length) → q ≤ p) ∧
    p = 2 :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_ribbons_l507_50796


namespace NUMINAMATH_CALUDE_max_xy_value_l507_50733

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) :
  x * y ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    4 * x₀^2 + 9 * y₀^2 + 3 * x₀ * y₀ = 30 ∧ x₀ * y₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l507_50733


namespace NUMINAMATH_CALUDE_distance_between_harper_and_jack_l507_50774

/-- Represents the distance between two runners at the end of a race. -/
def distance_between (race_length : ℕ) (jack_distance : ℕ) : ℕ :=
  race_length - jack_distance

/-- Proves that the distance between Harper and Jack at the end of the race is 848 meters. -/
theorem distance_between_harper_and_jack :
  let race_length_km : ℕ := 1
  let race_length_m : ℕ := race_length_km * 1000
  let jack_distance : ℕ := 152
  distance_between race_length_m jack_distance = 848 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_harper_and_jack_l507_50774


namespace NUMINAMATH_CALUDE_angle_in_first_quadrant_l507_50711

theorem angle_in_first_quadrant (α : Real) (h : Real.sin α + Real.cos α > 1) : 
  0 < α ∧ α < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_first_quadrant_l507_50711


namespace NUMINAMATH_CALUDE_sqrt_72_equals_6_sqrt_2_l507_50712

theorem sqrt_72_equals_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_72_equals_6_sqrt_2_l507_50712


namespace NUMINAMATH_CALUDE_function_range_implies_a_value_l507_50740

theorem function_range_implies_a_value (a : ℝ) (h1 : a > 0) : 
  (∀ x ∈ Set.Icc a (2 * a), (8 / x) ∈ Set.Icc (a / 4) 2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_range_implies_a_value_l507_50740


namespace NUMINAMATH_CALUDE_at_hash_product_l507_50716

-- Define the @ operation
def at_op (a b c : ℤ) : ℤ := a * b - b^2 + c

-- Define the # operation
def hash_op (a b c : ℤ) : ℤ := a + b - a * b^2 + c

-- Theorem statement
theorem at_hash_product : 
  let c : ℤ := 3
  (at_op 4 3 c) * (hash_op 4 3 c) = -156 := by
  sorry

end NUMINAMATH_CALUDE_at_hash_product_l507_50716


namespace NUMINAMATH_CALUDE_lee_overall_percentage_l507_50751

theorem lee_overall_percentage (t : ℝ) (h1 : t > 0) : 
  let james_solo := 0.70 * (t / 2)
  let james_total := 0.85 * t
  let together := james_total - james_solo
  let lee_solo := 0.75 * (t / 2)
  lee_solo + together = 0.875 * t := by
sorry

end NUMINAMATH_CALUDE_lee_overall_percentage_l507_50751


namespace NUMINAMATH_CALUDE_stationery_cost_l507_50741

/-- The total cost of a pen, pencil, and eraser with given price relationships -/
theorem stationery_cost (pencil_cost : ℚ) : 
  pencil_cost = 8 →
  (pencil_cost + (1/2 * pencil_cost) + (2 * (1/2 * pencil_cost))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_l507_50741


namespace NUMINAMATH_CALUDE_floor_tiles_l507_50744

theorem floor_tiles (black_tiles : ℕ) (total_tiles : ℕ) : 
  black_tiles = 441 → 
  ∃ (side_length : ℕ), 
    side_length * side_length = total_tiles ∧
    side_length = (black_tiles.sqrt : ℕ) + 2 * 3 →
    total_tiles = 729 :=
by sorry

end NUMINAMATH_CALUDE_floor_tiles_l507_50744


namespace NUMINAMATH_CALUDE_pascal_third_element_51st_row_l507_50760

/-- The number of elements in the nth row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

/-- The kth element in the nth row of Pascal's triangle -/
def pascal_element (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_third_element_51st_row : 
  pascal_element 51 2 = 1275 :=
sorry

end NUMINAMATH_CALUDE_pascal_third_element_51st_row_l507_50760


namespace NUMINAMATH_CALUDE_base_seven_digits_of_956_l507_50737

theorem base_seven_digits_of_956 : ∃ n : ℕ, (7^(n-1) ≤ 956 ∧ 956 < 7^n) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_956_l507_50737


namespace NUMINAMATH_CALUDE_modulus_of_z_l507_50702

def i : ℂ := Complex.I

def z : ℂ := (1 + i) * (1 + 2*i)

theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l507_50702


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l507_50725

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 12 * X^3 + 24 * X^2 - 10 * X + 5
  let divisor : Polynomial ℚ := 3 * X + 4
  let quotient : Polynomial ℚ := 4 * X^2 - 22/3
  dividend = divisor * quotient + (Polynomial.C (-197/9) : Polynomial ℚ) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l507_50725


namespace NUMINAMATH_CALUDE_angle_between_lines_l507_50747

def line1_direction : ℝ × ℝ := (2, 1)
def line2_direction : ℝ × ℝ := (4, 2)

theorem angle_between_lines (θ : ℝ) : 
  θ = Real.arccos (
    (line1_direction.1 * line2_direction.1 + line1_direction.2 * line2_direction.2) /
    (Real.sqrt (line1_direction.1^2 + line1_direction.2^2) * 
     Real.sqrt (line2_direction.1^2 + line2_direction.2^2))
  ) →
  Real.cos θ = 1 := by
sorry

end NUMINAMATH_CALUDE_angle_between_lines_l507_50747


namespace NUMINAMATH_CALUDE_height_range_selection_probability_overall_avg_height_l507_50786

-- Define the track and field team
def num_male : ℕ := 12
def num_female : ℕ := 8
def total_athletes : ℕ := num_male + num_female
def max_height : ℕ := 190
def min_height : ℕ := 160
def avg_height_male : ℝ := 175
def avg_height_female : ℝ := 165

-- Theorem 1: The range of heights is 30cm
theorem height_range : max_height - min_height = 30 := by sorry

-- Theorem 2: The probability of an athlete being selected in a random sample of 10 is 1/2
theorem selection_probability : (10 : ℝ) / total_athletes = (1 : ℝ) / 2 := by sorry

-- Theorem 3: The overall average height of the team is 171cm
theorem overall_avg_height :
  (num_male : ℝ) / total_athletes * avg_height_male +
  (num_female : ℝ) / total_athletes * avg_height_female = 171 := by sorry

end NUMINAMATH_CALUDE_height_range_selection_probability_overall_avg_height_l507_50786


namespace NUMINAMATH_CALUDE_prob_all_odd_is_one_42_l507_50739

/-- The number of slips in the hat -/
def total_slips : ℕ := 10

/-- The number of odd-numbered slips in the hat -/
def odd_slips : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability of drawing all odd-numbered slips -/
def prob_all_odd : ℚ := (odd_slips : ℚ) / total_slips *
                        (odd_slips - 1) / (total_slips - 1) *
                        (odd_slips - 2) / (total_slips - 2) *
                        (odd_slips - 3) / (total_slips - 3)

theorem prob_all_odd_is_one_42 : prob_all_odd = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_odd_is_one_42_l507_50739


namespace NUMINAMATH_CALUDE_exists_infinite_ap_not_in_polynomial_image_l507_50770

/-- A polynomial of degree 10 with integer coefficients -/
def IntPolynomial (P : ℤ → ℤ) : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ),
    ∀ x, P x = a₁₀ * x^10 + a₉ * x^9 + a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + 
             a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

/-- An infinite arithmetic progression -/
def InfiniteArithmeticProgression (a d : ℤ) : Set ℤ :=
  {n : ℤ | ∃ k : ℤ, n = a + k * d}

/-- The main theorem -/
theorem exists_infinite_ap_not_in_polynomial_image (P : ℤ → ℤ) 
    (h : IntPolynomial P) :
  ∃ (a d : ℤ), d ≠ 0 ∧ 
    ∀ n ∈ InfiniteArithmeticProgression a d, 
      ∀ k : ℤ, P k ≠ n :=
by sorry

end NUMINAMATH_CALUDE_exists_infinite_ap_not_in_polynomial_image_l507_50770


namespace NUMINAMATH_CALUDE_range_of_a_l507_50759

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l507_50759


namespace NUMINAMATH_CALUDE_decimal_division_l507_50743

theorem decimal_division : (0.1 : ℝ) / 0.004 = 25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l507_50743


namespace NUMINAMATH_CALUDE_potassium_dichromate_oxidizes_Br_and_I_l507_50738

/-- Standard reduction potential for I₂ + 2e⁻ → 2I⁻ -/
def E_I₂ : ℝ := 0.54

/-- Standard reduction potential for Cr₂O₇²⁻ + 14H⁺ + 6e⁻ → 2Cr³⁺ + 7H₂O -/
def E_Cr₂O₇ : ℝ := 1.33

/-- Standard oxidation potential for 2Br⁻ - 2e⁻ → Br₂ -/
def E_Br : ℝ := 1.07

/-- Standard oxidation potential for 2I⁻ - 2e⁻ → I₂ -/
def E_I : ℝ := 0.54

/-- A reaction is spontaneous if its cell potential is positive -/
def is_spontaneous (cell_potential : ℝ) : Prop := cell_potential > 0

/-- Theorem: Potassium dichromate can oxidize both Br⁻ and I⁻ -/
theorem potassium_dichromate_oxidizes_Br_and_I :
  is_spontaneous (E_Cr₂O₇ - E_Br) ∧ is_spontaneous (E_Cr₂O₇ - E_I) := by
  sorry


end NUMINAMATH_CALUDE_potassium_dichromate_oxidizes_Br_and_I_l507_50738


namespace NUMINAMATH_CALUDE_determinant_max_value_l507_50771

theorem determinant_max_value :
  let f : ℝ → ℝ := λ θ => 2 * Real.sqrt 2 * Real.cos θ + Real.cos (2 * θ)
  ∃ (θ : ℝ), f θ = 2 * Real.sqrt 2 + 1 ∧ ∀ (φ : ℝ), f φ ≤ 2 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_determinant_max_value_l507_50771


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l507_50762

theorem subtraction_multiplication_equality : (3.65 - 1.25) * 2 = 4.80 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l507_50762


namespace NUMINAMATH_CALUDE_min_distance_on_feb_9th_l507_50749

/-- Represents the squared distance between a space probe and Mars as a function of time -/
def D (a b c : ℝ) (t : ℝ) : ℝ := a * t^2 + b * t + c

/-- Theorem stating that the minimum distance occurs on February 9th -/
theorem min_distance_on_feb_9th (a b c : ℝ) :
  D a b c (-9) = 25 →
  D a b c 0 = 4 →
  D a b c 3 = 9 →
  ∃ (t_min : ℝ), t_min = -1 ∧ ∀ (t : ℝ), D a b c t_min ≤ D a b c t :=
sorry

end NUMINAMATH_CALUDE_min_distance_on_feb_9th_l507_50749


namespace NUMINAMATH_CALUDE_birds_flew_up_l507_50765

/-- The number of birds that flew up to a tree -/
theorem birds_flew_up (initial : ℕ) (total : ℕ) (h1 : initial = 14) (h2 : total = 35) :
  total - initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_birds_flew_up_l507_50765


namespace NUMINAMATH_CALUDE_prob_heart_or_king_two_draws_l507_50713

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards that are either hearts or kings -/
def heart_or_king : ℕ := 16

/-- The probability of drawing a card that is neither a heart nor a king -/
def prob_not_heart_or_king : ℚ := (deck_size - heart_or_king) / deck_size

/-- The probability of drawing at least one heart or king in two draws with replacement -/
def prob_at_least_one_heart_or_king : ℚ := 1 - prob_not_heart_or_king ^ 2

theorem prob_heart_or_king_two_draws :
  prob_at_least_one_heart_or_king = 88 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_heart_or_king_two_draws_l507_50713


namespace NUMINAMATH_CALUDE_xyz_absolute_value_l507_50769

theorem xyz_absolute_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (heq1 : x + 1 / y = y + 1 / z)
  (heq2 : y + 1 / z = z + 1 / x + 1) :
  |x * y * z| = 1 := by sorry

end NUMINAMATH_CALUDE_xyz_absolute_value_l507_50769


namespace NUMINAMATH_CALUDE_smallest_block_size_l507_50753

theorem smallest_block_size (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 378 → 
  l * m * n ≥ 560 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_size_l507_50753


namespace NUMINAMATH_CALUDE_arthurs_walk_l507_50727

/-- Arthur's walk problem -/
theorem arthurs_walk (blocks_west blocks_south : ℕ) (block_length : ℚ) :
  blocks_west = 8 →
  blocks_south = 10 →
  block_length = 1/4 →
  (blocks_west + blocks_south : ℚ) * block_length = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_walk_l507_50727


namespace NUMINAMATH_CALUDE_bicycle_sale_profit_l507_50701

theorem bicycle_sale_profit (final_price : ℝ) (initial_cost : ℝ) (intermediate_profit_rate : ℝ) :
  final_price = 225 →
  initial_cost = 150 →
  intermediate_profit_rate = 0.25 →
  ((final_price / (1 + intermediate_profit_rate) - initial_cost) / initial_cost) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_bicycle_sale_profit_l507_50701


namespace NUMINAMATH_CALUDE_hotel_rooms_l507_50734

/-- Proves that a hotel with the given conditions has 10 rooms -/
theorem hotel_rooms (R : ℕ) 
  (people_per_room : ℕ) 
  (towels_per_person : ℕ) 
  (total_towels : ℕ) 
  (h1 : people_per_room = 3) 
  (h2 : towels_per_person = 2) 
  (h3 : total_towels = 60) 
  (h4 : R * people_per_room * towels_per_person = total_towels) : 
  R = 10 := by
  sorry

#check hotel_rooms

end NUMINAMATH_CALUDE_hotel_rooms_l507_50734


namespace NUMINAMATH_CALUDE_lottery_win_probability_l507_50778

/-- A lottery event with two prize categories -/
structure LotteryEvent where
  firstPrizeProb : ℝ
  secondPrizeProb : ℝ

/-- The probability of winning a prize in the lottery event -/
def winPrizeProb (event : LotteryEvent) : ℝ :=
  event.firstPrizeProb + event.secondPrizeProb

/-- Theorem stating the probability of winning a prize in the given lottery event -/
theorem lottery_win_probability :
  ∃ (event : LotteryEvent), 
    event.firstPrizeProb = 0.1 ∧ 
    event.secondPrizeProb = 0.1 ∧ 
    winPrizeProb event = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_lottery_win_probability_l507_50778


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l507_50726

theorem pencil_pen_cost (p q : ℝ) 
  (h1 : 3 * p + 2 * q = 4.20)
  (h2 : 4 * p + 3 * q = 5.60) : 
  p + q = 1.40 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l507_50726


namespace NUMINAMATH_CALUDE_remainders_of_65_powers_l507_50799

theorem remainders_of_65_powers (n : ℕ) : 
  (65^(6*n) % 9 = 1) ∧ 
  (65^(6*n + 1) % 9 = 2) ∧ 
  (65^(6*n + 2) % 9 = 4) ∧ 
  (65^(6*n + 3) % 9 = 8) := by
sorry

end NUMINAMATH_CALUDE_remainders_of_65_powers_l507_50799


namespace NUMINAMATH_CALUDE_makeup_fraction_of_savings_l507_50728

/-- Given Leila's original savings and the cost of a sweater, prove the fraction spent on make-up -/
theorem makeup_fraction_of_savings (original_savings : ℚ) (sweater_cost : ℚ) 
  (h1 : original_savings = 80)
  (h2 : sweater_cost = 20) :
  (original_savings - sweater_cost) / original_savings = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_makeup_fraction_of_savings_l507_50728


namespace NUMINAMATH_CALUDE_reading_time_per_page_l507_50773

theorem reading_time_per_page 
  (planned_hours : ℝ) 
  (actual_fraction : ℝ) 
  (pages_read : ℕ) : 
  planned_hours = 3 → 
  actual_fraction = 3/4 → 
  pages_read = 9 → 
  (planned_hours * actual_fraction * 60) / pages_read = 15 := by
sorry

end NUMINAMATH_CALUDE_reading_time_per_page_l507_50773


namespace NUMINAMATH_CALUDE_hypotenuse_length_l507_50700

-- Define a right triangle
structure RightTriangle where
  a : ℝ  -- First leg
  b : ℝ  -- Second leg
  c : ℝ  -- Hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Theorem statement
theorem hypotenuse_length (t : RightTriangle) 
  (perimeter : t.a + t.b + t.c = 40)  -- Perimeter condition
  (area : (1/2) * t.a * t.b = 24)     -- Area condition
  : t.c = 18.8 := by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_hypotenuse_length_l507_50700


namespace NUMINAMATH_CALUDE_polynomial_value_l507_50761

/-- A quadratic polynomial of the form a(x^3 - x^2 + 3x) + b(2x^2 + x) + x^3 - 5 -/
def p (a b x : ℝ) : ℝ := a*(x^3 - x^2 + 3*x) + b*(2*x^2 + x) + x^3 - 5

/-- If p(2) = -17, then p(-2) = -1 -/
theorem polynomial_value (a b : ℝ) (h : p a b 2 = -17) : p a b (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l507_50761


namespace NUMINAMATH_CALUDE_smallest_t_for_complete_circle_l507_50768

/-- The smallest value of t such that when r = sin θ is plotted for 0 ≤ θ ≤ t,
    the resulting graph represents the entire circle is π. -/
theorem smallest_t_for_complete_circle : 
  ∃ t : ℝ, t > 0 ∧ 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = Real.sin θ) ∧
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = Real.sin θ) ∧
  (∀ t' : ℝ, t' < t → 
    ∃ x y : ℝ, x^2 + y^2 ≤ 1 ∧ 
    ∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t' → x ≠ (Real.sin θ) * (Real.cos θ) ∨ y ≠ Real.sin θ)) ∧
  t = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_smallest_t_for_complete_circle_l507_50768


namespace NUMINAMATH_CALUDE_friday_first_day_over_200_l507_50756

/-- Represents the days of the week -/
inductive Day
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Returns the number of days after Monday -/
def daysAfterMonday (d : Day) : Nat :=
  match d with
  | Day.monday => 0
  | Day.tuesday => 1
  | Day.wednesday => 2
  | Day.thursday => 3
  | Day.friday => 4
  | Day.saturday => 5
  | Day.sunday => 6

/-- Calculates the number of paperclips on a given day -/
def paperclipsOn (d : Day) : Nat :=
  4 * (3 ^ (daysAfterMonday d))

/-- Theorem: Friday is the first day with more than 200 paperclips -/
theorem friday_first_day_over_200 :
  (∀ d : Day, d ≠ Day.friday → paperclipsOn d ≤ 200) ∧
  paperclipsOn Day.friday > 200 :=
sorry

end NUMINAMATH_CALUDE_friday_first_day_over_200_l507_50756


namespace NUMINAMATH_CALUDE_largest_degree_with_asymptote_l507_50777

-- Define the denominator of our rational function
def q (x : ℝ) : ℝ := 3 * x^6 + 2 * x^3 - x + 4

-- Define a proposition that checks if a polynomial has a horizontal asymptote when divided by q(x)
def has_horizontal_asymptote (p : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x > M, |p x / q x - L| < ε

-- Define a function to get the degree of a polynomial
noncomputable def poly_degree (p : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem largest_degree_with_asymptote :
  ∃ (p : ℝ → ℝ), poly_degree p = 6 ∧ has_horizontal_asymptote p ∧
  ∀ (p' : ℝ → ℝ), poly_degree p' > 6 → ¬(has_horizontal_asymptote p') :=
sorry

end NUMINAMATH_CALUDE_largest_degree_with_asymptote_l507_50777


namespace NUMINAMATH_CALUDE_remaining_coin_value_l507_50722

def initial_quarters : Nat := 11
def initial_dimes : Nat := 15
def initial_nickels : Nat := 7

def purchased_quarters : Nat := 1
def purchased_dimes : Nat := 8
def purchased_nickels : Nat := 3

def quarter_value : Nat := 25
def dime_value : Nat := 10
def nickel_value : Nat := 5

theorem remaining_coin_value :
  (initial_quarters - purchased_quarters) * quarter_value +
  (initial_dimes - purchased_dimes) * dime_value +
  (initial_nickels - purchased_nickels) * nickel_value = 340 := by
  sorry

end NUMINAMATH_CALUDE_remaining_coin_value_l507_50722


namespace NUMINAMATH_CALUDE_quaternary_30012_to_decimal_l507_50791

/-- Converts a list of digits in base 4 to its decimal representation -/
def quaternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The quaternary number 30012 -/
def quaternary_30012 : List Nat := [2, 1, 0, 0, 3]

theorem quaternary_30012_to_decimal :
  quaternary_to_decimal quaternary_30012 = 774 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_30012_to_decimal_l507_50791


namespace NUMINAMATH_CALUDE_range_of_f_l507_50782

def f (x : ℝ) := x^2 - 2*x + 4

theorem range_of_f :
  ∀ y ∈ Set.Icc 3 7, ∃ x ∈ Set.Icc 0 3, f x = y ∧
  ∀ x ∈ Set.Icc 0 3, ∃ y ∈ Set.Icc 3 7, f x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l507_50782


namespace NUMINAMATH_CALUDE_triangle_height_equals_30_l507_50767

/-- Given a rectangle with perimeter 60 cm and a right triangle with base 15 cm,
    if their areas are equal, then the height of the triangle is 30 cm. -/
theorem triangle_height_equals_30 (rectangle_perimeter : ℝ) (triangle_base : ℝ) (h : ℝ) :
  rectangle_perimeter = 60 →
  triangle_base = 15 →
  (rectangle_perimeter / 4) * (rectangle_perimeter / 4) = (1 / 2) * triangle_base * h →
  h = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_equals_30_l507_50767


namespace NUMINAMATH_CALUDE_line_vector_at_negative_two_l507_50776

def line_vector (s : ℝ) : ℝ × ℝ := sorry

theorem line_vector_at_negative_two :
  line_vector 1 = (2, 5) →
  line_vector 4 = (8, -7) →
  line_vector (-2) = (-4, 17) := by sorry

end NUMINAMATH_CALUDE_line_vector_at_negative_two_l507_50776


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l507_50717

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set of the original inequality
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | (1/2 : ℝ) < x ∧ x < 2}

-- State the theorem
theorem quadratic_inequality_solution :
  ∃ (a : ℝ), 
    (∀ x, x ∈ solution_set a ↔ f a x > 0) ∧
    (a = -2) ∧
    (∀ x, a * x^2 - 5 * x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l507_50717


namespace NUMINAMATH_CALUDE_swanson_class_avg_l507_50784

/-- The average number of zits per kid in Ms. Swanson's class -/
def swanson_avg : ℝ := 5

/-- The number of kids in Ms. Swanson's class -/
def swanson_kids : ℕ := 25

/-- The number of kids in Mr. Jones' class -/
def jones_kids : ℕ := 32

/-- The average number of zits per kid in Mr. Jones' class -/
def jones_avg : ℝ := 6

/-- The difference in total zits between Mr. Jones' and Ms. Swanson's classes -/
def zit_difference : ℕ := 67

theorem swanson_class_avg : 
  swanson_avg * swanson_kids + zit_difference = jones_avg * jones_kids := by
  sorry

#check swanson_class_avg

end NUMINAMATH_CALUDE_swanson_class_avg_l507_50784


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_b_16_minus_1_l507_50705

-- Define b as a natural number
def b : ℕ := 2

-- Define the function for the number of distinct prime factors
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

-- Define the function for the largest prime factor
def largest_prime_factor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_b_16_minus_1 :
  num_distinct_prime_factors (b^16 - 1) = 4 →
  largest_prime_factor (b^16 - 1) = 257 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_b_16_minus_1_l507_50705


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l507_50735

theorem absolute_value_equation_solution :
  ∃! y : ℚ, |y - 3| = |y + 2| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l507_50735


namespace NUMINAMATH_CALUDE_ball_trajectory_l507_50709

/-- A rectangle with side lengths 2a and 2b -/
structure Rectangle (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  a : ℝ
  b : ℝ
  h : (5 : ℝ) * a = (3 : ℝ) * b

/-- The angle at which the ball is hit from corner A -/
def hitAngle (α : ℝ) := α

/-- The ball hits three different sides before reaching the center -/
def hitsThreeSides (rect : Rectangle ℝ) (α : ℝ) : Prop :=
  ∃ (p q r : ℝ × ℝ), 
    p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
    (p.1 = 0 ∨ p.1 = 2*rect.a ∨ p.2 = 0 ∨ p.2 = 2*rect.b) ∧
    (q.1 = 0 ∨ q.1 = 2*rect.a ∨ q.2 = 0 ∨ q.2 = 2*rect.b) ∧
    (r.1 = 0 ∨ r.1 = 2*rect.a ∨ r.2 = 0 ∨ r.2 = 2*rect.b)

theorem ball_trajectory (rect : Rectangle ℝ) (α : ℝ) :
  hitsThreeSides rect α ↔ Real.tan α = 9/25 := by sorry

end NUMINAMATH_CALUDE_ball_trajectory_l507_50709


namespace NUMINAMATH_CALUDE_square_root_of_four_l507_50797

theorem square_root_of_four :
  {y : ℝ | y^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l507_50797


namespace NUMINAMATH_CALUDE_smallest_twice_square_three_cube_l507_50790

def is_twice_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k^2

def is_three_times_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 3 * m^3

theorem smallest_twice_square_three_cube :
  (∀ n : ℕ, n > 0 ∧ n < 648 → ¬(is_twice_perfect_square n ∧ is_three_times_perfect_cube n)) ∧
  (is_twice_perfect_square 648 ∧ is_three_times_perfect_cube 648) :=
sorry

end NUMINAMATH_CALUDE_smallest_twice_square_three_cube_l507_50790


namespace NUMINAMATH_CALUDE_min_horse_pony_difference_l507_50707

/-- Represents a ranch with horses and ponies -/
structure Ranch where
  horses : ℕ
  ponies : ℕ
  horseshoed_ponies : ℕ
  icelandic_horseshoed_ponies : ℕ

/-- Conditions for the ranch -/
def valid_ranch (r : Ranch) : Prop :=
  r.horses > r.ponies ∧
  r.horses + r.ponies = 164 ∧
  r.horseshoed_ponies = (3 * r.ponies) / 10 ∧
  r.icelandic_horseshoed_ponies = (5 * r.horseshoed_ponies) / 8

theorem min_horse_pony_difference (r : Ranch) (h : valid_ranch r) :
  r.horses - r.ponies = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_horse_pony_difference_l507_50707


namespace NUMINAMATH_CALUDE_sons_age_is_eighteen_l507_50714

/-- Proves that the son's age is 18 years given the conditions in the problem -/
theorem sons_age_is_eighteen (son_age man_age : ℕ) : 
  man_age = son_age + 20 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_is_eighteen_l507_50714


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l507_50792

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  ArithmeticSequence a →
  (a 1 + a 4 + a 7 = 39) →
  (a 2 + a 5 + a 8 = 33) →
  (a 3 + a 6 + a 9 = 27) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l507_50792


namespace NUMINAMATH_CALUDE_correct_rounding_l507_50764

def round_to_thousandth (x : ℚ) : ℚ :=
  (⌊x * 1000 + 0.5⌋ : ℚ) / 1000

theorem correct_rounding :
  round_to_thousandth 2.098176 = 2.098 := by sorry

end NUMINAMATH_CALUDE_correct_rounding_l507_50764


namespace NUMINAMATH_CALUDE_f_neg_one_lt_f_one_l507_50704

/-- A function f: ℝ → ℝ that satisfies the given conditions -/
def f : ℝ → ℝ := sorry

/-- f is differentiable on ℝ -/
axiom f_differentiable : Differentiable ℝ f

/-- The functional equation for f -/
axiom f_eq (x : ℝ) : f x = x^2 + 2 * x * (deriv f 2)

/-- Theorem: f(-1) < f(1) -/
theorem f_neg_one_lt_f_one : f (-1) < f 1 := by sorry

end NUMINAMATH_CALUDE_f_neg_one_lt_f_one_l507_50704


namespace NUMINAMATH_CALUDE_max_y_over_x_on_circle_l507_50766

theorem max_y_over_x_on_circle :
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - Real.sqrt 3)^2 = 3}
  ∃ (max : ℝ), max = Real.sqrt 3 ∧ ∀ (p : ℝ × ℝ), p ∈ circle → p.2 / p.1 ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_y_over_x_on_circle_l507_50766


namespace NUMINAMATH_CALUDE_cards_per_page_l507_50730

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 3) 
  (h2 : old_cards = 13) 
  (h3 : pages = 2) : 
  (new_cards + old_cards) / pages = 8 :=
by sorry

end NUMINAMATH_CALUDE_cards_per_page_l507_50730


namespace NUMINAMATH_CALUDE_square_plus_product_equals_zero_l507_50710

theorem square_plus_product_equals_zero : (-2)^2 + (-2) * 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_product_equals_zero_l507_50710


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l507_50787

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 6) 
  (prod_sum_eq : a * b + a * c + b * c = 5) 
  (prod_eq : a * b * c = -12) : 
  a^3 + b^3 + c^3 = 90 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l507_50787


namespace NUMINAMATH_CALUDE_horner_method_equals_polynomial_f_at_5_equals_4881_l507_50755

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

def horner_method (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc c => acc * x + c) 0

theorem horner_method_equals_polynomial (x : ℝ) :
  horner_method [6, 5, 4, 3, 2, 1] x = f x :=
sorry

theorem f_at_5_equals_4881 :
  f 5 = 4881 :=
sorry

end NUMINAMATH_CALUDE_horner_method_equals_polynomial_f_at_5_equals_4881_l507_50755


namespace NUMINAMATH_CALUDE_gwens_spent_money_l507_50706

/-- Gwen's birthday money problem -/
theorem gwens_spent_money (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 5)
  (h2 : remaining_amount = 2) :
  initial_amount - remaining_amount = 3 := by
sorry

end NUMINAMATH_CALUDE_gwens_spent_money_l507_50706


namespace NUMINAMATH_CALUDE_angle_conversion_and_coterminal_l507_50758

-- Define α in degrees
def α : ℝ := 1680

-- Theorem statement
theorem angle_conversion_and_coterminal (α : ℝ) :
  ∃ (k : ℤ) (β : ℝ), 
    (α * π / 180 = 2 * k * π + β) ∧ 
    (0 ≤ β) ∧ (β < 2 * π) ∧
    (∃ (θ : ℝ), 
      (θ = -8 * π / 3) ∧ 
      (-4 * π < θ) ∧ (θ < -2 * π) ∧
      (∃ (m : ℤ), θ = 2 * m * π + β)) := by
  sorry

end NUMINAMATH_CALUDE_angle_conversion_and_coterminal_l507_50758


namespace NUMINAMATH_CALUDE_five_chairs_cost_l507_50750

/-- The cost of a single plastic chair -/
def chair_cost : ℝ := sorry

/-- The cost of a portable table -/
def table_cost : ℝ := sorry

/-- Three chairs cost the same as one table -/
axiom chair_table_relation : 3 * chair_cost = table_cost

/-- One table and two chairs cost $55 -/
axiom total_cost : table_cost + 2 * chair_cost = 55

/-- The cost of five plastic chairs is $55 -/
theorem five_chairs_cost : 5 * chair_cost = 55 := by sorry

end NUMINAMATH_CALUDE_five_chairs_cost_l507_50750


namespace NUMINAMATH_CALUDE_decreasing_function_range_l507_50748

theorem decreasing_function_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, x < y → x ∈ Set.Ioo (-1) 1 → y ∈ Set.Ioo (-1) 1 → f x > f y) →
  (1 - a ∈ Set.Ioo (-1) 1) →
  (a^2 - 1 ∈ Set.Ioo (-1) 1) →
  f (1 - a) < f (a^2 - 1) →
  0 < a ∧ a < Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_decreasing_function_range_l507_50748


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l507_50757

/-- For a parabola with equation y² = -x, the distance from its focus to its directrix is 1/2. -/
theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), y^2 = -x → 
  ∃ (focus_x focus_y directrix_x : ℝ),
    (focus_x = -1/4 ∧ focus_y = 0) ∧
    directrix_x = 1/4 ∧
    |focus_x - directrix_x| = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l507_50757


namespace NUMINAMATH_CALUDE_triangular_array_sum_of_digits_l507_50789

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_sum_of_digits :
  ∀ N : ℕ, N > 0 → N * (N + 1) / 2 = 3003 → sum_of_digits N = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_array_sum_of_digits_l507_50789


namespace NUMINAMATH_CALUDE_smallest_prime_square_mod_six_is_five_l507_50752

theorem smallest_prime_square_mod_six_is_five :
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p^2 % 6 = 1 ∧ 
    (∀ (q : ℕ), Nat.Prime q → q^2 % 6 = 1 → p ≤ q) ∧
    p = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_square_mod_six_is_five_l507_50752


namespace NUMINAMATH_CALUDE_olivers_card_collection_l507_50772

/-- Oliver's card collection problem -/
theorem olivers_card_collection :
  ∀ (alien_baseball monster_club battle_gremlins : ℕ),
  monster_club = 2 * alien_baseball →
  battle_gremlins = 48 →
  battle_gremlins = 3 * alien_baseball →
  monster_club = 32 := by
sorry

end NUMINAMATH_CALUDE_olivers_card_collection_l507_50772


namespace NUMINAMATH_CALUDE_fourth_root_fifth_root_approx_l507_50718

theorem fourth_root_fifth_root_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |((32 : ℝ) / 100000)^((1/5 : ℝ) * (1/4 : ℝ)) - 0.6687| < ε := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_fifth_root_approx_l507_50718


namespace NUMINAMATH_CALUDE_binary_arrangements_count_l507_50793

/-- The number of ways to arrange 3 ones and 3 zeros in a binary string -/
def binaryArrangements : ℕ := 20

/-- The length of the binary string -/
def stringLength : ℕ := 6

/-- The number of ones in the binary string -/
def numberOfOnes : ℕ := 3

theorem binary_arrangements_count :
  binaryArrangements = Nat.choose stringLength numberOfOnes := by
  sorry

end NUMINAMATH_CALUDE_binary_arrangements_count_l507_50793
