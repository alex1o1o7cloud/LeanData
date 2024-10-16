import Mathlib

namespace NUMINAMATH_CALUDE_equal_utility_days_l661_66118

/-- Utility function --/
def utility (math reading painting : ℝ) : ℝ := math^2 + reading * painting

/-- The problem statement --/
theorem equal_utility_days (t : ℝ) : 
  utility 4 t (12 - t) = utility 3 (t + 1) (11 - t) → t = 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_utility_days_l661_66118


namespace NUMINAMATH_CALUDE_divisibility_by_five_l661_66149

theorem divisibility_by_five : ∃ k : ℤ, 3^444 + 4^333 = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l661_66149


namespace NUMINAMATH_CALUDE_class_size_l661_66103

theorem class_size (football : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ)
  (h1 : football = 26)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 7) :
  football + tennis - both + neither = 36 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l661_66103


namespace NUMINAMATH_CALUDE_circle_tangent_line_m_values_l661_66164

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the translation vector
def translation_vector : ℝ × ℝ := (2, 1)

-- Define the translated circle
def translated_circle (x y : ℝ) : Prop :=
  original_circle (x - translation_vector.1) (y - translation_vector.2)

-- Define the tangent line
def tangent_line (x y m : ℝ) : Prop := x + y + m = 0

-- Theorem statement
theorem circle_tangent_line_m_values :
  ∃ m : ℝ, (m = -1 ∨ m = -5) ∧
  ∀ x y : ℝ, translated_circle x y →
  (∃ p : ℝ × ℝ, p.1 + p.2 + m = 0 ∧
  ∀ q : ℝ × ℝ, q.1 + q.2 + m = 0 →
  (p.1 - x)^2 + (p.2 - y)^2 ≤ (q.1 - x)^2 + (q.2 - y)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_line_m_values_l661_66164


namespace NUMINAMATH_CALUDE_cable_package_savings_l661_66198

/-- Calculates the savings from choosing a bundle package over individual subscriptions --/
theorem cable_package_savings
  (basic_cost movie_cost bundle_cost : ℕ)
  (sports_cost_diff : ℕ)
  (h1 : basic_cost = 15)
  (h2 : movie_cost = 12)
  (h3 : sports_cost_diff = 3)
  (h4 : bundle_cost = 25) :
  basic_cost + movie_cost + (movie_cost - sports_cost_diff) - bundle_cost = 11 := by
  sorry


end NUMINAMATH_CALUDE_cable_package_savings_l661_66198


namespace NUMINAMATH_CALUDE_largest_square_area_l661_66111

theorem largest_square_area (A B C : ℝ × ℝ) (h_right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (h_sum_squares : (B.1 - A.1)^2 + (B.2 - A.2)^2 + (C.1 - B.1)^2 + (C.2 - B.2)^2 + 2 * ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 500) :
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 125 := by
  sorry

#check largest_square_area

end NUMINAMATH_CALUDE_largest_square_area_l661_66111


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l661_66169

theorem imaginary_part_of_complex_product : 
  let z : ℂ := (1 - Complex.I) * (2 + Complex.I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l661_66169


namespace NUMINAMATH_CALUDE_smallest_square_for_five_disks_l661_66163

/-- A disk with radius 1 -/
structure UnitDisk where
  center : ℝ × ℝ

/-- A square with side length a -/
structure Square (a : ℝ) where
  center : ℝ × ℝ

/-- Predicate to check if two disks overlap -/
def disks_overlap (d1 d2 : UnitDisk) : Prop :=
  (d1.center.1 - d2.center.1)^2 + (d1.center.2 - d2.center.2)^2 < 4

/-- Predicate to check if a disk is contained in a square -/
def disk_in_square (d : UnitDisk) (s : Square a) : Prop :=
  abs (d.center.1 - s.center.1) ≤ a/2 - 1 ∧ abs (d.center.2 - s.center.2) ≤ a/2 - 1

/-- The main theorem -/
theorem smallest_square_for_five_disks :
  ∀ a : ℝ,
  (∃ (s : Square a) (d1 d2 d3 d4 d5 : UnitDisk),
    disk_in_square d1 s ∧ disk_in_square d2 s ∧ disk_in_square d3 s ∧ disk_in_square d4 s ∧ disk_in_square d5 s ∧
    ¬disks_overlap d1 d2 ∧ ¬disks_overlap d1 d3 ∧ ¬disks_overlap d1 d4 ∧ ¬disks_overlap d1 d5 ∧
    ¬disks_overlap d2 d3 ∧ ¬disks_overlap d2 d4 ∧ ¬disks_overlap d2 d5 ∧
    ¬disks_overlap d3 d4 ∧ ¬disks_overlap d3 d5 ∧
    ¬disks_overlap d4 d5) →
  a ≥ 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_for_five_disks_l661_66163


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l661_66122

theorem consecutive_odd_numbers_sum (a b c d e : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1 ∧ 
            b = 2*k + 3 ∧ 
            c = 2*k + 5 ∧ 
            d = 2*k + 7 ∧ 
            e = 2*k + 9) →
  a + b + c + d + e = 130 →
  c = 26 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l661_66122


namespace NUMINAMATH_CALUDE_number_wall_solution_l661_66130

/-- Represents a number wall with the given base numbers -/
structure NumberWall (m : ℤ) :=
  (base : Fin 4 → ℤ)
  (base_values : base 0 = m ∧ base 1 = 6 ∧ base 2 = -3 ∧ base 3 = 4)

/-- Calculates the value at the top of the number wall -/
def top_value (w : NumberWall m) : ℤ :=
  let level1_0 := w.base 0 + w.base 1
  let level1_1 := w.base 1 + w.base 2
  let level1_2 := w.base 2 + w.base 3
  let level2_0 := level1_0 + level1_1
  let level2_1 := level1_1 + level1_2
  level2_0 + level2_1

/-- The theorem to be proved -/
theorem number_wall_solution (m : ℤ) (w : NumberWall m) :
  top_value w = 20 → m = 7 := by sorry

end NUMINAMATH_CALUDE_number_wall_solution_l661_66130


namespace NUMINAMATH_CALUDE_problem_1_l661_66174

theorem problem_1 : -2.8 + (-3.6) + 3 - (-3.6) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l661_66174


namespace NUMINAMATH_CALUDE_alice_walked_distance_l661_66179

/-- The distance Alice walked in miles -/
def alice_distance (blocks_south : ℕ) (blocks_west : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_south + blocks_west : ℚ) * miles_per_block

/-- Theorem stating that Alice walked 3.25 miles -/
theorem alice_walked_distance :
  alice_distance 5 8 (1/4) = 3.25 := by sorry

end NUMINAMATH_CALUDE_alice_walked_distance_l661_66179


namespace NUMINAMATH_CALUDE_geometric_sum_base_case_l661_66109

theorem geometric_sum_base_case (a : ℝ) (h : a ≠ 1) :
  1 + a = (1 - a^2) / (1 - a) := by sorry

end NUMINAMATH_CALUDE_geometric_sum_base_case_l661_66109


namespace NUMINAMATH_CALUDE_speedster_convertibles_l661_66180

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) : 
  (3 * speedsters = total) →  -- 1/3 of total inventory is Speedsters
  (5 * convertibles = 4 * speedsters) →  -- 4/5 of Speedsters are convertibles
  (total - speedsters = 30) →  -- 30 vehicles are not Speedsters
  convertibles = 12 := by
sorry

end NUMINAMATH_CALUDE_speedster_convertibles_l661_66180


namespace NUMINAMATH_CALUDE_sixth_root_unity_product_l661_66145

theorem sixth_root_unity_product (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_unity_product_l661_66145


namespace NUMINAMATH_CALUDE_boys_without_calculators_l661_66147

theorem boys_without_calculators (total_boys : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ)
  (h1 : total_boys = 16)
  (h2 : students_with_calculators = 22)
  (h3 : girls_with_calculators = 13) :
  total_boys - (students_with_calculators - girls_with_calculators) = 7 := by
  sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l661_66147


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l661_66146

theorem sum_of_solutions_is_zero :
  let f (x : ℝ) := (-12 * x) / (x^2 - 1) - (3 * x) / (x + 1) + 9 / (x - 1)
  ∃ (a b : ℝ), (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b) ∧ a + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l661_66146


namespace NUMINAMATH_CALUDE_prime_square_mod_six_l661_66120

theorem prime_square_mod_six (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  p^2 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_six_l661_66120


namespace NUMINAMATH_CALUDE_jeremy_songs_theorem_l661_66173

theorem jeremy_songs_theorem (x y : ℕ) : 
  x % 2 = 0 ∧ 
  9 = 2 * Int.sqrt x - 5 ∧ 
  y = (9 + x) / 2 → 
  9 + x + y = 110 := by
sorry

end NUMINAMATH_CALUDE_jeremy_songs_theorem_l661_66173


namespace NUMINAMATH_CALUDE_divisibility_condition_l661_66131

theorem divisibility_condition (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l661_66131


namespace NUMINAMATH_CALUDE_find_divisor_l661_66171

theorem find_divisor : ∃ (d : ℕ), d > 1 ∧ 
  (3198 + 2) % d = 0 ∧ 
  3198 % d ≠ 0 ∧ 
  ∀ (k : ℕ), k > 1 → (3198 + 2) % k = 0 → 3198 % k ≠ 0 → d ≤ k :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l661_66171


namespace NUMINAMATH_CALUDE_equation_solution_l661_66159

theorem equation_solution (a : ℚ) : (3/2 * 2 - 2*a = 0) → (2*a - 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l661_66159


namespace NUMINAMATH_CALUDE_pen_pricing_gain_percentage_l661_66184

theorem pen_pricing_gain_percentage :
  ∀ (C S : ℝ),
  C > 0 →
  20 * C = 12 * S →
  (S - C) / C * 100 = 200 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_pricing_gain_percentage_l661_66184


namespace NUMINAMATH_CALUDE_tan_double_angle_l661_66194

theorem tan_double_angle (α : Real) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2 →
  Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l661_66194


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l661_66175

theorem point_on_terminal_side (m : ℝ) (α : ℝ) : 
  (∃ P : ℝ × ℝ, P = (m, m + 1) ∧ P.1 / Real.sqrt (P.1^2 + P.2^2) = 3/5) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l661_66175


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l661_66107

theorem nested_fraction_evaluation :
  1 + 2 / (3 + 4 / (5 + 6/7)) = 233/151 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l661_66107


namespace NUMINAMATH_CALUDE_base_n_representation_l661_66141

theorem base_n_representation (n : ℕ) (a b : ℤ) : 
  n > 8 → 
  n^2 - a*n + b = 0 → 
  a = n + 8 → 
  b = 8*n := by
sorry

end NUMINAMATH_CALUDE_base_n_representation_l661_66141


namespace NUMINAMATH_CALUDE_polynomial_transformation_l661_66101

/-- Given y = x + 1/x, prove that x^6 + x^5 - 5x^4 + x^3 + 3x^2 + x + 1 = 0 is equivalent to x^4*y^2 - 4*x^2*y^2 + 3*x^2 = 0 -/
theorem polynomial_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^6 + x^5 - 5*x^4 + x^3 + 3*x^2 + x + 1 = 0 ↔ x^4*y^2 - 4*x^2*y^2 + 3*x^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l661_66101


namespace NUMINAMATH_CALUDE_second_number_value_l661_66108

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 220 ∧ 
  a = 2 * b ∧ 
  c = (1 / 3) * a → 
  b = 60 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l661_66108


namespace NUMINAMATH_CALUDE_sum_cube_plus_twice_sum_squares_l661_66112

theorem sum_cube_plus_twice_sum_squares : (3 + 7)^3 + 2*(3^2 + 7^2) = 1116 := by
  sorry

end NUMINAMATH_CALUDE_sum_cube_plus_twice_sum_squares_l661_66112


namespace NUMINAMATH_CALUDE_loan_repayment_theorem_l661_66166

/-- Calculates the lump sum payment for a loan with given parameters -/
def lump_sum_payment (
  principal : ℝ)  -- Initial loan amount
  (rate : ℝ)      -- Annual interest rate as a decimal
  (num_payments : ℕ) -- Total number of annuity payments
  (delay : ℕ)     -- Years before first payment
  (payments_made : ℕ) -- Number of payments made before death
  (years_after_death : ℕ) -- Years after death until lump sum payment
  : ℝ :=
  sorry

theorem loan_repayment_theorem :
  let principal := 20000
  let rate := 0.04
  let num_payments := 10
  let delay := 3
  let payments_made := 5
  let years_after_death := 2
  abs (lump_sum_payment principal rate num_payments delay payments_made years_after_death - 119804.6) < 1 :=
sorry

end NUMINAMATH_CALUDE_loan_repayment_theorem_l661_66166


namespace NUMINAMATH_CALUDE_bills_divisible_by_101_l661_66143

theorem bills_divisible_by_101 
  (a b : ℕ) 
  (h_not_cong : a % 101 ≠ b % 101) 
  (h_total : ℕ) 
  (h_total_eq : h_total = 100) :
  ∃ (subset : Finset ℕ), subset.card ≤ h_total ∧ 
    (∃ (k₁ k₂ : ℕ), k₁ + k₂ = subset.card ∧ (k₁ * a + k₂ * b) % 101 = 0) :=
sorry

end NUMINAMATH_CALUDE_bills_divisible_by_101_l661_66143


namespace NUMINAMATH_CALUDE_complex_maximum_value_l661_66186

theorem complex_maximum_value (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₂ = 4)
  (h2 : 4 * z₁^2 - 2 * z₁ * z₂ + z₂^2 = 0) :
  ∃ (M : ℝ), M = 6 * Real.sqrt 6 ∧ 
    ∀ (w : ℂ), w = z₁ → Complex.abs ((w + 1)^2 * (w - 2)) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_complex_maximum_value_l661_66186


namespace NUMINAMATH_CALUDE_grain_spilled_calculation_l661_66193

/-- Calculates the amount of grain spilled into the water -/
def grain_spilled (original : ℕ) (remaining : ℕ) : ℕ :=
  original - remaining

/-- Theorem: The amount of grain spilled is the difference between original and remaining -/
theorem grain_spilled_calculation (original remaining : ℕ) 
  (h1 : original = 50870)
  (h2 : remaining = 918) :
  grain_spilled original remaining = 49952 := by
  sorry

#eval grain_spilled 50870 918

end NUMINAMATH_CALUDE_grain_spilled_calculation_l661_66193


namespace NUMINAMATH_CALUDE_square_difference_cubed_l661_66170

theorem square_difference_cubed : (5^2 - 4^2)^3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_cubed_l661_66170


namespace NUMINAMATH_CALUDE_marble_jar_problem_l661_66119

theorem marble_jar_problem (g y : ℕ) : 
  (g - 1 : ℚ) / (g + y - 1 : ℚ) = 1 / 8 →
  (g : ℚ) / (g + y - 3 : ℚ) = 1 / 6 →
  g + y = 9 := by
sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l661_66119


namespace NUMINAMATH_CALUDE_inequality_proof_l661_66181

/-- For all real x greater than -1, 1 - e^(-x) is greater than or equal to x/(x+1) -/
theorem inequality_proof (x : ℝ) (h : x > -1) : 1 - Real.exp (-x) ≥ x / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l661_66181


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l661_66113

def repeating_decimal : ℚ := 36 / 99

theorem repeating_decimal_fraction :
  repeating_decimal = 4 / 11 ∧
  4 + 11 = 15 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l661_66113


namespace NUMINAMATH_CALUDE_ellipse_sum_l661_66110

/-- The sum of h, k, a, and b for a specific ellipse -/
theorem ellipse_sum (h k a b : ℝ) : 
  ((3 : ℝ) = h) → ((-5 : ℝ) = k) → ((7 : ℝ) = a) → ((2 : ℝ) = b) → 
  h + k + a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l661_66110


namespace NUMINAMATH_CALUDE_decreasing_quadratic_condition_l661_66144

/-- A function f is decreasing on an interval [a, +∞) if for all x₁, x₂ in [a, +∞) with x₁ < x₂, f(x₁) > f(x₂) -/
def DecreasingOnInterval (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ → x₁ < x₂ → f x₁ > f x₂

theorem decreasing_quadratic_condition (a : ℝ) :
  DecreasingOnInterval (fun x => a * x^2 + 4 * (a + 1) * x - 3) 2 ↔ a ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_condition_l661_66144


namespace NUMINAMATH_CALUDE_smallest_prime_congruence_l661_66189

theorem smallest_prime_congruence : 
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p = 71 ∧ 
    ∀ (q : ℕ), Nat.Prime q → q < p → 
      ¬(∃ (q_inv : ℕ), (q * q_inv) % 143 = 1 ∧ (q + q_inv) % 143 = 25) ∧
    ∃ (p_inv : ℕ), (p * p_inv) % 143 = 1 ∧ (p + p_inv) % 143 = 25 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_congruence_l661_66189


namespace NUMINAMATH_CALUDE_no_perfect_square_2007_plus_4n_l661_66115

theorem no_perfect_square_2007_plus_4n :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), 2007 + 4^n = k^2 := by
sorry

end NUMINAMATH_CALUDE_no_perfect_square_2007_plus_4n_l661_66115


namespace NUMINAMATH_CALUDE_correct_division_l661_66151

theorem correct_division (n : ℚ) : n / 22 = 2 → n / 20 = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_l661_66151


namespace NUMINAMATH_CALUDE_sam_apples_per_sandwich_l661_66106

/-- The number of apples Sam eats per sandwich -/
def apples_per_sandwich (sandwiches_per_day : ℕ) (days_in_week : ℕ) (total_apples : ℕ) : ℚ :=
  total_apples / (sandwiches_per_day * days_in_week)

/-- Theorem stating that Sam eats 4 apples per sandwich -/
theorem sam_apples_per_sandwich :
  apples_per_sandwich 10 7 280 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sam_apples_per_sandwich_l661_66106


namespace NUMINAMATH_CALUDE_integer_solution_exists_l661_66136

theorem integer_solution_exists (a b : ℤ) : ∃ (x y z t : ℤ), 
  (x + y + 2*z + 2*t = a) ∧ (2*x - 2*y + z - t = b) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_exists_l661_66136


namespace NUMINAMATH_CALUDE_large_doll_price_correct_l661_66168

def total_spending : ℝ := 350
def price_difference : ℝ := 2
def extra_dolls : ℕ := 20

def large_doll_price : ℝ := 7
def small_doll_price : ℝ := large_doll_price - price_difference

theorem large_doll_price_correct :
  (total_spending / small_doll_price = total_spending / large_doll_price + extra_dolls) ∧
  (large_doll_price > 0) ∧
  (small_doll_price > 0) := by
  sorry

end NUMINAMATH_CALUDE_large_doll_price_correct_l661_66168


namespace NUMINAMATH_CALUDE_max_value_of_f_l661_66197

def f (x : ℝ) := x^3 - 3*x

theorem max_value_of_f :
  ∃ (m : ℝ), m = 18 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l661_66197


namespace NUMINAMATH_CALUDE_percentage_relationship_l661_66121

theorem percentage_relationship (a b c : ℝ) (h1 : 0.06 * a = 10) (h2 : c = b / a) :
  ∃ p : ℝ, p * b = 6 ∧ p * 100 = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relationship_l661_66121


namespace NUMINAMATH_CALUDE_playground_length_is_687_5_l661_66139

/-- A rectangular playground with given perimeter, breadth, and diagonal -/
structure Playground where
  perimeter : ℝ
  breadth : ℝ
  diagonal : ℝ

/-- The length of a rectangular playground -/
def length (p : Playground) : ℝ :=
  ((p.diagonal ^ 2) - (p.breadth ^ 2)) ^ (1/2)

/-- Theorem stating the length of the specific playground -/
theorem playground_length_is_687_5 (p : Playground) 
  (h1 : p.perimeter = 1200)
  (h2 : p.breadth = 500)
  (h3 : p.diagonal = 850) : 
  length p = 687.5 := by
  sorry

end NUMINAMATH_CALUDE_playground_length_is_687_5_l661_66139


namespace NUMINAMATH_CALUDE_difference_of_squares_l661_66140

theorem difference_of_squares : 635^2 - 615^2 = 25000 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l661_66140


namespace NUMINAMATH_CALUDE_mod_inverse_five_mod_thirtythree_l661_66190

theorem mod_inverse_five_mod_thirtythree :
  ∃ x : ℕ, x < 33 ∧ (5 * x) % 33 = 1 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_mod_inverse_five_mod_thirtythree_l661_66190


namespace NUMINAMATH_CALUDE_log_equality_l661_66196

theorem log_equality (c d : ℝ) (hc : c = Real.log 625 / Real.log 4) (hd : d = Real.log 25 / Real.log 5) :
  c = d :=
by sorry

end NUMINAMATH_CALUDE_log_equality_l661_66196


namespace NUMINAMATH_CALUDE_symmetric_difference_A_B_l661_66128

/-- Set difference -/
def set_difference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

/-- Symmetric difference -/
def symmetric_difference (M N : Set ℝ) : Set ℝ :=
  set_difference M N ∪ set_difference N M

/-- Set A -/
def A : Set ℝ := {t | ∃ x, t = x^2 - 3*x}

/-- Set B -/
def B : Set ℝ := {x | ∃ y, y = Real.log (-x)}

theorem symmetric_difference_A_B :
  symmetric_difference A B = {x | x < -9/4 ∨ x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_A_B_l661_66128


namespace NUMINAMATH_CALUDE_tangency_condition_single_intersection_condition_l661_66138

-- Define the line l: y = kx - 3k + 2
def line (k x : ℝ) : ℝ := k * x - 3 * k + 2

-- Define the curve C: (x-1)^2 + (y+1)^2 = 4
def curve (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 4

-- Define the domain of x for the curve
def x_domain (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1

-- Theorem for tangency condition
theorem tangency_condition (k : ℝ) : 
  (∃ x, x_domain x ∧ curve x (line k x) ∧ 
   (∀ x', x' ≠ x → ¬curve x' (line k x'))) ↔ 
  k = 5/12 :=
sorry

-- Theorem for single intersection condition
theorem single_intersection_condition (k : ℝ) :
  (∃! x, x_domain x ∧ curve x (line k x)) ↔ 
  (1/2 < k ∧ k ≤ 5/2) ∨ k = 5/12 :=
sorry

end NUMINAMATH_CALUDE_tangency_condition_single_intersection_condition_l661_66138


namespace NUMINAMATH_CALUDE_no_real_roots_implies_a_greater_than_one_l661_66153

theorem no_real_roots_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a ≠ 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_implies_a_greater_than_one_l661_66153


namespace NUMINAMATH_CALUDE_existence_of_periodic_even_function_l661_66116

theorem existence_of_periodic_even_function :
  ∃ f : ℝ → ℝ,
    (f 0 ≠ 0) ∧
    (∀ x : ℝ, f x = f (-x)) ∧
    (∀ x : ℝ, f (x + π) = f x) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_periodic_even_function_l661_66116


namespace NUMINAMATH_CALUDE_max_scores_is_45_l661_66155

/-- Represents a test with multiple-choice questions. -/
structure Test where
  num_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  unanswered_points : ℤ

/-- Calculates the maximum number of different possible total scores for a given test. -/
def max_different_scores (t : Test) : ℕ :=
  sorry

/-- The specific test described in the problem. -/
def problem_test : Test :=
  { num_questions := 10
  , correct_points := 4
  , incorrect_points := -1
  , unanswered_points := 0 }

/-- Theorem stating that the maximum number of different possible total scores for the problem_test is 45. -/
theorem max_scores_is_45 : max_different_scores problem_test = 45 := by
  sorry

end NUMINAMATH_CALUDE_max_scores_is_45_l661_66155


namespace NUMINAMATH_CALUDE_flower_beds_count_l661_66178

/-- Given that there are 25 seeds in each flower bed and 750 seeds planted altogether,
    prove that the number of flower beds is 30. -/
theorem flower_beds_count (seeds_per_bed : ℕ) (total_seeds : ℕ) (num_beds : ℕ) 
    (h1 : seeds_per_bed = 25)
    (h2 : total_seeds = 750)
    (h3 : num_beds * seeds_per_bed = total_seeds) :
  num_beds = 30 := by
  sorry

end NUMINAMATH_CALUDE_flower_beds_count_l661_66178


namespace NUMINAMATH_CALUDE_parallel_resistors_l661_66172

theorem parallel_resistors (x y R : ℝ) (hx : x = 4) (hy : y = 5) 
  (hR : 1 / R = 1 / x + 1 / y) : R = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistors_l661_66172


namespace NUMINAMATH_CALUDE_prob_valid_sequence_equals_377_4096_sum_numerator_denominator_l661_66160

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def valid_sequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

/-- The probability of a valid sequence of length 12 -/
def prob_valid_sequence : ℚ := (valid_sequences 12 : ℚ) / (total_sequences 12 : ℚ)

theorem prob_valid_sequence_equals_377_4096 :
  prob_valid_sequence = 377 / 4096 :=
sorry

theorem sum_numerator_denominator :
  377 + 4096 = 4473 :=
sorry

end NUMINAMATH_CALUDE_prob_valid_sequence_equals_377_4096_sum_numerator_denominator_l661_66160


namespace NUMINAMATH_CALUDE_kristen_turtles_l661_66183

theorem kristen_turtles (trey kris kristen : ℕ) : 
  trey = 7 * kris →
  kris = kristen / 4 →
  trey = kristen + 9 →
  kristen = 12 := by
sorry

end NUMINAMATH_CALUDE_kristen_turtles_l661_66183


namespace NUMINAMATH_CALUDE_train_speed_l661_66137

/-- The speed of a train given its length, time to cross a walking man, and the man's speed. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 500 →
  crossing_time = 29.997600191984642 →
  man_speed_kmh = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l661_66137


namespace NUMINAMATH_CALUDE_probability_even_sum_l661_66182

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def is_even_sum (pair : ℕ × ℕ) : Bool :=
  (pair.1 + pair.2) % 2 = 0

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (card_set.product card_set).filter (λ pair => pair.1 < pair.2 ∧ is_even_sum pair)

def total_outcomes : Finset (ℕ × ℕ) :=
  (card_set.product card_set).filter (λ pair => pair.1 < pair.2)

theorem probability_even_sum :
  (favorable_outcomes.card : ℚ) / total_outcomes.card = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_even_sum_l661_66182


namespace NUMINAMATH_CALUDE_sewer_capacity_l661_66188

/-- The amount of run-off produced per hour of rain in gallons -/
def runoff_per_hour : ℕ := 1000

/-- The number of days the sewers can handle before overflow -/
def days_before_overflow : ℕ := 10

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The total gallons of run-off the sewers can handle -/
def total_runoff_capacity : ℕ := runoff_per_hour * days_before_overflow * hours_per_day

theorem sewer_capacity :
  total_runoff_capacity = 240000 := by
  sorry

end NUMINAMATH_CALUDE_sewer_capacity_l661_66188


namespace NUMINAMATH_CALUDE_fair_coin_same_side_probability_l661_66124

theorem fair_coin_same_side_probability :
  let n : ℕ := 10
  let p : ℝ := 1 / 2
  (p ^ n : ℝ) = 1 / 1024 := by sorry

end NUMINAMATH_CALUDE_fair_coin_same_side_probability_l661_66124


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l661_66187

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_planes
  (m : Line) (α β : Plane)
  (h1 : parallel α β)
  (h2 : perpendicular m α) :
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l661_66187


namespace NUMINAMATH_CALUDE_haydens_earnings_l661_66104

/-- Represents Hayden's work day --/
structure WorkDay where
  totalHours : ℕ
  peakHours : ℕ
  totalRides : ℕ
  longDistanceRides : ℕ
  shortDistanceGallons : ℕ
  longDistanceGallons : ℕ
  maintenanceCost : ℕ
  tollCount : ℕ
  parkingExpense : ℕ
  positiveReviews : ℕ
  excellentReviews : ℕ

/-- Calculate Hayden's earnings for a given work day --/
def calculateEarnings (day : WorkDay) : ℚ :=
  sorry

/-- Theorem stating that Hayden's earnings for the given day equal $411.75 --/
theorem haydens_earnings : 
  let day : WorkDay := {
    totalHours := 12,
    peakHours := 3,
    totalRides := 6,
    longDistanceRides := 3,
    shortDistanceGallons := 10,
    longDistanceGallons := 20,
    maintenanceCost := 30,
    tollCount := 2,
    parkingExpense := 10,
    positiveReviews := 2,
    excellentReviews := 1
  }
  calculateEarnings day = 411.75 := by sorry

end NUMINAMATH_CALUDE_haydens_earnings_l661_66104


namespace NUMINAMATH_CALUDE_inequality_proof_l661_66185

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d = 3) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 3/4) ∧
  (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) ≥ 2/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l661_66185


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l661_66167

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 - x = 0} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l661_66167


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_inclusion_l661_66150

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

-- Theorem 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | |x + 2| + |x - 1| ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2
theorem range_of_a_for_inclusion :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x + |x - 1| ≤ 2) → -3/2 ≤ a ∧ a ≤ -1/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_inclusion_l661_66150


namespace NUMINAMATH_CALUDE_workday_meeting_percentage_l661_66129

/-- Represents the duration of a workday in hours -/
def workday_hours : ℝ := 10

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℝ := 60

/-- Calculates the total workday time in minutes -/
def workday_minutes : ℝ := workday_hours * 60

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℝ := 2 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes

/-- Theorem: The percentage of the workday spent in meetings is 30% -/
theorem workday_meeting_percentage :
  (total_meeting_minutes / workday_minutes) * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_workday_meeting_percentage_l661_66129


namespace NUMINAMATH_CALUDE_initial_sum_equation_l661_66148

/-- 
Given:
- Two interest rates: 15% and 10% per annum
- Compound interest applied annually for 3 years
- The difference in total interest between the two rates is Rs. 1500

Prove that the initial sum P satisfies the equation:
P * ((1 + 0.15)^3 - (1 + 0.10)^3) = 1500
-/
theorem initial_sum_equation (P : ℝ) : 
  P * ((1 + 0.15)^3 - (1 + 0.10)^3) = 1500 :=
by sorry

end NUMINAMATH_CALUDE_initial_sum_equation_l661_66148


namespace NUMINAMATH_CALUDE_complex_magnitude_l661_66176

theorem complex_magnitude (z : ℂ) (h : (1 - 2*Complex.I) * z = 3 + 4*Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l661_66176


namespace NUMINAMATH_CALUDE_table_height_is_36_l661_66177

/-- Represents a cuboidal block of wood -/
structure Block where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Represents the arrangement in Figure 1 -/
def figure1 (b : Block) (table_height : ℝ) : ℝ :=
  b.length + table_height - b.depth

/-- Represents the arrangement in Figure 2 -/
def figure2 (b : Block) (table_height : ℝ) : ℝ :=
  2 * b.length + table_height

/-- Theorem stating the height of the table given the conditions -/
theorem table_height_is_36 (b : Block) (h : ℝ) :
  figure1 b h = 36 → figure2 b h = 46 → h = 36 := by
  sorry

#check table_height_is_36

end NUMINAMATH_CALUDE_table_height_is_36_l661_66177


namespace NUMINAMATH_CALUDE_opposite_of_negative_half_l661_66195

theorem opposite_of_negative_half : -(-(1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_half_l661_66195


namespace NUMINAMATH_CALUDE_math_competition_team_selection_l661_66125

theorem math_competition_team_selection (n : ℕ) (k : ℕ) (total : ℕ) (exclude : ℕ) :
  n = 10 →
  k = 3 →
  total = Nat.choose (n - 1) k →
  exclude = Nat.choose (n - 3) k →
  total - exclude = 49 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_team_selection_l661_66125


namespace NUMINAMATH_CALUDE_nine_crosses_fit_chessboard_l661_66199

/-- Represents a cross pentomino -/
structure CrossPentomino where
  area : ℕ
  size : ℕ × ℕ

/-- Represents a chessboard -/
structure Chessboard where
  size : ℕ × ℕ
  area : ℕ

/-- Theorem: Nine cross pentominoes can fit within an 8x8 chessboard -/
theorem nine_crosses_fit_chessboard (cross : CrossPentomino) (board : Chessboard) : 
  cross.area = 5 ∧ 
  cross.size = (1, 1) ∧ 
  board.size = (8, 8) ∧ 
  board.area = 64 →
  9 * cross.area ≤ board.area :=
by sorry

end NUMINAMATH_CALUDE_nine_crosses_fit_chessboard_l661_66199


namespace NUMINAMATH_CALUDE_line_equation_through_points_l661_66162

/-- A line passing through two points. -/
structure Line2D where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0. -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The equation of the line passing through (-1, 2) and (2, 5) is x - y + 3 = 0. -/
theorem line_equation_through_points :
  let l : Line2D := { point1 := (-1, 2), point2 := (2, 5) }
  let eq : LineEquation := { a := 1, b := -1, c := 3 }
  (∀ x y : ℝ, (x = l.point1.1 ∧ y = l.point1.2) ∨ (x = l.point2.1 ∧ y = l.point2.2) →
    eq.a * x + eq.b * y + eq.c = 0) ∧
  (∀ x y : ℝ, eq.a * x + eq.b * y + eq.c = 0 →
    ∃ t : ℝ, x = l.point1.1 + t * (l.point2.1 - l.point1.1) ∧
              y = l.point1.2 + t * (l.point2.2 - l.point1.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_line_equation_through_points_l661_66162


namespace NUMINAMATH_CALUDE_balance_sheet_equation_l661_66114

/-- Given the equation 4m - t = 8000 where m = 4 and t = 4 + 100i, prove that t = -7988 - 100i. -/
theorem balance_sheet_equation (m t : ℂ) (h1 : 4 * m - t = 8000) (h2 : m = 4) (h3 : t = 4 + 100 * Complex.I) : 
  t = -7988 - 100 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_balance_sheet_equation_l661_66114


namespace NUMINAMATH_CALUDE_disneyland_arrangements_l661_66165

theorem disneyland_arrangements (n : ℕ) (k : ℕ) : n = 7 → k = 2 → n.factorial * k^n = 645120 := by
  sorry

end NUMINAMATH_CALUDE_disneyland_arrangements_l661_66165


namespace NUMINAMATH_CALUDE_gcd_10010_20020_l661_66117

theorem gcd_10010_20020 : Nat.gcd 10010 20020 = 10010 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10010_20020_l661_66117


namespace NUMINAMATH_CALUDE_coin_draw_probability_l661_66191

def penny_count : ℕ := 3
def nickel_count : ℕ := 3
def quarter_count : ℕ := 6
def dime_count : ℕ := 3
def total_coins : ℕ := penny_count + nickel_count + quarter_count + dime_count
def drawn_coins : ℕ := 8
def min_value : ℚ := 175/100

def successful_outcomes : ℕ := 9
def total_outcomes : ℕ := Nat.choose total_coins drawn_coins

theorem coin_draw_probability :
  (successful_outcomes : ℚ) / total_outcomes = 9 / 6435 :=
sorry

end NUMINAMATH_CALUDE_coin_draw_probability_l661_66191


namespace NUMINAMATH_CALUDE_no_triple_primes_l661_66192

theorem no_triple_primes : ¬ ∃ p : ℕ, Prime p ∧ Prime (p + 7) ∧ Prime (p + 14) := by
  sorry

end NUMINAMATH_CALUDE_no_triple_primes_l661_66192


namespace NUMINAMATH_CALUDE_final_rope_length_l661_66134

/-- Calculates the final length of a rope made by tying multiple pieces together -/
theorem final_rope_length
  (rope_lengths : List ℝ)
  (knot_loss : ℝ)
  (h_lengths : rope_lengths = [8, 20, 2, 2, 2, 7])
  (h_knot_loss : knot_loss = 1.2)
  : (rope_lengths.sum - knot_loss * (rope_lengths.length - 1 : ℝ)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_final_rope_length_l661_66134


namespace NUMINAMATH_CALUDE_janous_inequality_l661_66100

theorem janous_inequality (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ (a₀ + 2) * (b₀ + 2) = c₀ * d₀ := by
  sorry

#check janous_inequality

end NUMINAMATH_CALUDE_janous_inequality_l661_66100


namespace NUMINAMATH_CALUDE_inequality_proof_l661_66161

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_sum : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + c * d * a / (1 - b)^2 + 
  d * a * b / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l661_66161


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_contrapositive_true_negation_equivalence_inequality_condition_l661_66126

-- Statement 1
theorem contrapositive_equivalence (x : ℝ) :
  (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) := by sorry

-- Statement 2
theorem contrapositive_true (m : ℝ) :
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔ (¬∃ x : ℝ, x^2 + x - m = 0 → m ≤ 0) := by sorry

-- Statement 3
theorem negation_equivalence :
  (¬∃ x > 1, x^2 - 2*x - 3 = 0) ↔ (∀ x > 1, x^2 - 2*x - 3 ≠ 0) := by sorry

-- Statement 4
theorem inequality_condition (a : ℝ) :
  (∀ x, -2 < x ∧ x < -1 → (x + a)*(x + 1) < 0) → a > 2 := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_contrapositive_true_negation_equivalence_inequality_condition_l661_66126


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l661_66135

theorem polynomial_equation_solution (x : ℝ) : 
  let p : ℝ → ℝ := λ x => (1 + Real.sqrt 109) / 2
  p (x^2) - p (x^2 - 3) = (p x)^2 + 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l661_66135


namespace NUMINAMATH_CALUDE_squares_remaining_l661_66156

theorem squares_remaining (total : ℕ) (removed_fraction : ℚ) (result : ℕ) : 
  total = 12 →
  removed_fraction = 1/2 * 2/3 →
  result = total - (removed_fraction * total).num →
  result = 8 := by
  sorry

end NUMINAMATH_CALUDE_squares_remaining_l661_66156


namespace NUMINAMATH_CALUDE_percentage_of_science_students_l661_66127

theorem percentage_of_science_students (total_boys : ℕ) (school_A_percentage : ℚ) (non_science_boys : ℕ) : 
  total_boys = 250 →
  school_A_percentage = 1/5 →
  non_science_boys = 35 →
  (((school_A_percentage * total_boys) - non_science_boys) / (school_A_percentage * total_boys)) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_science_students_l661_66127


namespace NUMINAMATH_CALUDE_ammonia_formed_l661_66158

/-- Represents a chemical compound in the reaction -/
inductive Compound
| NH4NO3
| NaOH
| NH3
| H2O
| NaNO3

/-- Represents the stoichiometric coefficients in the balanced equation -/
def reaction_coefficients : Compound → ℕ
| Compound.NH4NO3 => 1
| Compound.NaOH => 1
| Compound.NH3 => 1
| Compound.H2O => 1
| Compound.NaNO3 => 1

/-- The number of moles of each reactant available -/
def available_moles : Compound → ℕ
| Compound.NH4NO3 => 2
| Compound.NaOH => 2
| _ => 0

/-- Theorem stating that 2 moles of NH3 are formed in the reaction -/
theorem ammonia_formed :
  let limiting_reactant := min (available_moles Compound.NH4NO3) (available_moles Compound.NaOH)
  limiting_reactant * (reaction_coefficients Compound.NH3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ammonia_formed_l661_66158


namespace NUMINAMATH_CALUDE_rectangle_with_equal_sums_l661_66132

/-- A regular polygon with 2004 sides -/
structure RegularPolygon2004 where
  vertices : Fin 2004 → ℕ
  vertex_range : ∀ i, 1 ≤ vertices i ∧ vertices i ≤ 501

/-- Four vertices form a rectangle in a regular 2004-sided polygon -/
def isRectangle (p : RegularPolygon2004) (a b c d : Fin 2004) : Prop :=
  (b - a) % 2004 = (d - c) % 2004 ∧ (c - b) % 2004 = (a - d) % 2004

/-- The sums of numbers assigned to opposite vertices are equal -/
def equalOppositeSums (p : RegularPolygon2004) (a b c d : Fin 2004) : Prop :=
  p.vertices a + p.vertices c = p.vertices b + p.vertices d

/-- Main theorem: There exist four vertices forming a rectangle with equal opposite sums -/
theorem rectangle_with_equal_sums (p : RegularPolygon2004) :
  ∃ a b c d : Fin 2004, isRectangle p a b c d ∧ equalOppositeSums p a b c d := by
  sorry


end NUMINAMATH_CALUDE_rectangle_with_equal_sums_l661_66132


namespace NUMINAMATH_CALUDE_range_of_a_l661_66123

theorem range_of_a (x y a : ℝ) : 
  x - y = 2 → 
  x + y = a → 
  x > -1 → 
  y < 0 → 
  -4 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l661_66123


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l661_66105

/-- Convert a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

/-- Two natural numbers are consecutive odd integers -/
def consecutiveOdd (m n : ℕ) : Prop := isOdd m ∧ isOdd n ∧ n = m + 2

theorem base_conversion_theorem (C D : ℕ) :
  consecutiveOdd C D →
  toBase10 243 C + toBase10 65 D = toBase10 107 (C + D) →
  C + D = 16 := by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l661_66105


namespace NUMINAMATH_CALUDE_inscribed_semicircle_radius_l661_66154

/-- An isosceles triangle with a semicircle inscribed -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The diameter of the semicircle lies along the base of the triangle -/
  diameter_on_base : radius * 2 ≤ base

/-- The radius of the inscribed semicircle in the given isosceles triangle -/
theorem inscribed_semicircle_radius (t : IsoscelesTriangleWithSemicircle)
    (h_base : t.base = 20)
    (h_height : t.height = 18) :
    t.radius = 180 / (Real.sqrt 424 + 10) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_semicircle_radius_l661_66154


namespace NUMINAMATH_CALUDE_factorization_validity_l661_66142

theorem factorization_validity (x : ℝ) : x^2 - x - 6 = (x - 3) * (x + 2) := by
  sorry

#check factorization_validity

end NUMINAMATH_CALUDE_factorization_validity_l661_66142


namespace NUMINAMATH_CALUDE_pi_between_three_and_four_l661_66102

theorem pi_between_three_and_four : 
  Irrational Real.pi ∧ 3 < Real.pi ∧ Real.pi < 4 := by sorry

end NUMINAMATH_CALUDE_pi_between_three_and_four_l661_66102


namespace NUMINAMATH_CALUDE_rational_expression_equals_240_l661_66152

theorem rational_expression_equals_240 (x : ℝ) (h : x = 4) :
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_equals_240_l661_66152


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l661_66157

/-- Given a geometric sequence with first term a₁ = 3 and second term a₂ = -1/2,
    prove that the 7th term a₇ = 1/15552 -/
theorem geometric_sequence_seventh_term :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -1/2
  let r : ℚ := a₂ / a₁
  let a₇ : ℚ := a₁ * r^6
  a₇ = 1/15552 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l661_66157


namespace NUMINAMATH_CALUDE_equation_solution_l661_66133

theorem equation_solution : 
  ∃! x : ℝ, (2 + x ≠ 0 ∧ 3 * x - 1 ≠ 0) ∧ (1 / (2 + x) = 2 / (3 * x - 1)) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l661_66133
