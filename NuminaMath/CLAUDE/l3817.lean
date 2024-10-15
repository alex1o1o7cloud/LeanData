import Mathlib

namespace NUMINAMATH_CALUDE_gilda_marbles_l3817_381768

theorem gilda_marbles (M : ℝ) (h : M > 0) : 
  let remaining_after_pedro : ℝ := M * (1 - 0.3)
  let remaining_after_ebony : ℝ := remaining_after_pedro * (1 - 0.1)
  let remaining_after_lisa : ℝ := remaining_after_ebony * (1 - 0.4)
  remaining_after_lisa / M = 0.378 := by
  sorry

end NUMINAMATH_CALUDE_gilda_marbles_l3817_381768


namespace NUMINAMATH_CALUDE_books_read_l3817_381756

theorem books_read (total : ℕ) (remaining : ℕ) (read : ℕ) : 
  total = 14 → remaining = 6 → read = total - remaining → read = 8 := by
sorry

end NUMINAMATH_CALUDE_books_read_l3817_381756


namespace NUMINAMATH_CALUDE_incorrect_multiplication_l3817_381793

theorem incorrect_multiplication (correct_multiplier : ℕ) (number_to_multiply : ℕ) (difference : ℕ) 
  (h1 : correct_multiplier = 43)
  (h2 : number_to_multiply = 134)
  (h3 : difference = 1206) :
  ∃ (incorrect_multiplier : ℕ), 
    number_to_multiply * correct_multiplier - number_to_multiply * incorrect_multiplier = difference ∧
    incorrect_multiplier = 34 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_multiplication_l3817_381793


namespace NUMINAMATH_CALUDE_unique_odd_number_with_eight_multiples_l3817_381799

theorem unique_odd_number_with_eight_multiples : 
  ∃! x : ℕ, 
    x % 2 = 1 ∧ 
    x > 0 ∧
    (∃ S : Finset ℕ, 
      S.card = 8 ∧
      (∀ y ∈ S, 
        y < 80 ∧ 
        y % 2 = 1 ∧
        ∃ k m : ℕ, k > 0 ∧ m % 2 = 1 ∧ y = k * x * m) ∧
      (∀ y : ℕ, 
        y < 80 → 
        y % 2 = 1 → 
        (∃ k m : ℕ, k > 0 ∧ m % 2 = 1 ∧ y = k * x * m) → 
        y ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_unique_odd_number_with_eight_multiples_l3817_381799


namespace NUMINAMATH_CALUDE_dot_product_specific_vectors_l3817_381743

theorem dot_product_specific_vectors :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (1, -1)
  (a.1 * b.1 + a.2 * b.2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_specific_vectors_l3817_381743


namespace NUMINAMATH_CALUDE_sticker_difference_l3817_381722

def total_stickers : ℕ := 58
def first_box_stickers : ℕ := 23

theorem sticker_difference : 
  total_stickers - first_box_stickers - first_box_stickers = 12 := by
  sorry

end NUMINAMATH_CALUDE_sticker_difference_l3817_381722


namespace NUMINAMATH_CALUDE_percent_increase_decrease_l3817_381761

theorem percent_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 50) (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) < M ↔ p < 100 * q / (100 - q) := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_decrease_l3817_381761


namespace NUMINAMATH_CALUDE_range_of_3a_plus_2b_l3817_381715

theorem range_of_3a_plus_2b (a b : ℝ) (h : a^2 + b^2 = 4) :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 * Real.sqrt 13) (2 * Real.sqrt 13) ∧ x = 3*a + 2*b :=
by sorry

end NUMINAMATH_CALUDE_range_of_3a_plus_2b_l3817_381715


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3817_381797

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3817_381797


namespace NUMINAMATH_CALUDE_smallest_product_l3817_381769

def digits : List Nat := [1, 2, 3, 4]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    is_valid_arrangement a b c d →
    product a b c d ≥ 312 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l3817_381769


namespace NUMINAMATH_CALUDE_min_value_expression_l3817_381778

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → a^2 + 4 * b^2 + 1 / (a * b) ≤ x^2 + 4 * y^2 + 1 / (x * y)) ∧
  a^2 + 4 * b^2 + 1 / (a * b) = 17 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3817_381778


namespace NUMINAMATH_CALUDE_equilateral_triangle_reflection_theorem_l3817_381701

/-- Represents a ray path in an equilateral triangle -/
structure RayPath where
  n : ℕ  -- number of reflections
  returns_to_start : Bool  -- whether the ray returns to the starting point
  passes_through_vertices : Bool  -- whether the ray passes through other vertices

/-- Checks if a number is a valid reflection count -/
def is_valid_reflection_count (n : ℕ) : Prop :=
  (n % 6 = 1 ∨ n % 6 = 5) ∧ n ≠ 5 ∧ n ≠ 17

/-- Main theorem: Characterizes valid reflection counts in an equilateral triangle -/
theorem equilateral_triangle_reflection_theorem :
  ∀ (path : RayPath),
    path.returns_to_start ∧ ¬path.passes_through_vertices ↔
    is_valid_reflection_count path.n :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_reflection_theorem_l3817_381701


namespace NUMINAMATH_CALUDE_penny_throwing_ratio_l3817_381726

/-- Given the conditions of the penny-throwing problem, prove that the ratio of Rocky's pennies to Gretchen's is 1:3 -/
theorem penny_throwing_ratio (rachelle gretchen rocky : ℕ) : 
  rachelle = 180 →
  gretchen = rachelle / 2 →
  rachelle + gretchen + rocky = 300 →
  rocky / gretchen = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_penny_throwing_ratio_l3817_381726


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l3817_381795

theorem quadratic_roots_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    |x₁ - x₂| = 2) →
  p = 2 * Real.sqrt (q + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l3817_381795


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3817_381791

theorem complex_modulus_problem (Z : ℂ) (a : ℝ) :
  Z = 3 + a * I ∧ Complex.abs Z = 5 → a = 4 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3817_381791


namespace NUMINAMATH_CALUDE_implication_equivalences_l3817_381719

variable (p q : Prop)

theorem implication_equivalences (h : p → q) :
  (∃ (f : p → q), True) ∧
  (p → q) ∧
  (¬q → ¬p) ∧
  ((p → q) ∧ (¬p ∨ q)) :=
by sorry

end NUMINAMATH_CALUDE_implication_equivalences_l3817_381719


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l3817_381705

theorem cookie_jar_problem (C : ℕ) : (C - 1 = (C + 5) / 2) → C = 7 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l3817_381705


namespace NUMINAMATH_CALUDE_parentheses_value_l3817_381746

theorem parentheses_value (x : ℚ) : x * (-2/3) = 2 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_value_l3817_381746


namespace NUMINAMATH_CALUDE_axis_of_symmetry_sin_l3817_381748

open Real

theorem axis_of_symmetry_sin (φ : ℝ) :
  (∀ x, |sin (2*x + φ)| ≤ |sin (π/3 + φ)|) →
  ∃ k : ℤ, 2*(2*π/3) + φ = k*π + π/2 :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_sin_l3817_381748


namespace NUMINAMATH_CALUDE_line_touches_x_axis_twice_l3817_381775

/-- Represents the equation d = x^2 - x^3 -/
def d (x : ℝ) : ℝ := x^2 - x^3

/-- The line touches the x-axis when d(x) = 0 -/
def touches_x_axis (x : ℝ) : Prop := d x = 0

theorem line_touches_x_axis_twice :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ touches_x_axis x₁ ∧ touches_x_axis x₂ ∧
  ∀ (x : ℝ), touches_x_axis x → (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_line_touches_x_axis_twice_l3817_381775


namespace NUMINAMATH_CALUDE_total_bugs_eaten_l3817_381703

def gecko_bugs : ℕ := 12

def lizard_bugs : ℕ := gecko_bugs / 2

def frog_bugs : ℕ := 3 * lizard_bugs

def toad_bugs : ℕ := frog_bugs + frog_bugs / 2

theorem total_bugs_eaten :
  gecko_bugs + lizard_bugs + frog_bugs + toad_bugs = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_bugs_eaten_l3817_381703


namespace NUMINAMATH_CALUDE_pool_fill_time_l3817_381779

/-- The time required to fill a pool, given its capacity and the water supply rate. -/
def fillTime (poolCapacity : ℚ) (numHoses : ℕ) (flowRatePerHose : ℚ) : ℚ :=
  poolCapacity / (numHoses * flowRatePerHose * 60)

/-- Theorem stating that the time to fill the pool is 100/3 hours. -/
theorem pool_fill_time :
  fillTime 36000 6 3 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pool_fill_time_l3817_381779


namespace NUMINAMATH_CALUDE_union_empty_iff_both_empty_union_eq_diff_iff_empty_diff_eq_inter_iff_empty_l3817_381759

-- Define the universe set
variable {U : Type}

-- Define sets A, B, C as subsets of U
variable (A B C : Set U)

-- Theorem 1
theorem union_empty_iff_both_empty :
  A ∪ B = ∅ ↔ A = ∅ ∧ B = ∅ := by sorry

-- Theorem 2
theorem union_eq_diff_iff_empty :
  A ∪ B = A \ B ↔ B = ∅ := by sorry

-- Theorem 3
theorem diff_eq_inter_iff_empty :
  A \ B = A ∩ B ↔ A = ∅ := by sorry

-- Additional theorems can be added similarly for the remaining equivalences

end NUMINAMATH_CALUDE_union_empty_iff_both_empty_union_eq_diff_iff_empty_diff_eq_inter_iff_empty_l3817_381759


namespace NUMINAMATH_CALUDE_glycol_concentration_mixture_l3817_381720

/-- Proves that mixing 16 gallons of 40% glycol concentration with 8 gallons of 10% glycol concentration 
    results in a 30% glycol concentration in the final 24-gallon mixture. -/
theorem glycol_concentration_mixture 
  (total_volume : ℝ) 
  (volume_40_percent : ℝ)
  (volume_10_percent : ℝ)
  (concentration_40_percent : ℝ)
  (concentration_10_percent : ℝ)
  (h1 : total_volume = 24)
  (h2 : volume_40_percent = 16)
  (h3 : volume_10_percent = 8)
  (h4 : concentration_40_percent = 0.4)
  (h5 : concentration_10_percent = 0.1)
  (h6 : volume_40_percent + volume_10_percent = total_volume) :
  (volume_40_percent * concentration_40_percent + volume_10_percent * concentration_10_percent) / total_volume = 0.3 := by
  sorry


end NUMINAMATH_CALUDE_glycol_concentration_mixture_l3817_381720


namespace NUMINAMATH_CALUDE_ages_when_john_is_50_l3817_381787

/- Define the initial ages and relationships -/
def john_initial_age : ℕ := 10
def alice_initial_age : ℕ := 2 * john_initial_age
def mike_initial_age : ℕ := alice_initial_age - 4

/- Define John's future age -/
def john_future_age : ℕ := 50

/- Define the theorem to prove -/
theorem ages_when_john_is_50 :
  (john_future_age + (alice_initial_age - john_initial_age) = 60) ∧
  (john_future_age + (mike_initial_age - john_initial_age) = 56) := by
  sorry

end NUMINAMATH_CALUDE_ages_when_john_is_50_l3817_381787


namespace NUMINAMATH_CALUDE_quadratic_prime_square_solution_l3817_381729

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

/-- The theorem states that the only integer solution to the equation 2x^2 - x - 36 = p^2,
    where p is a prime number, is x = 13. -/
theorem quadratic_prime_square_solution :
  ∀ x : ℤ, (∃ p : ℕ, is_prime p ∧ (2 * x^2 - x - 36 : ℤ) = p^2) ↔ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_prime_square_solution_l3817_381729


namespace NUMINAMATH_CALUDE_cleanup_drive_total_l3817_381723

/-- The total amount of garbage collected by three groups in a cleanup drive -/
theorem cleanup_drive_total (group1_pounds group2_pounds group3_ounces : ℕ) 
  (h1 : group1_pounds = 387)
  (h2 : group2_pounds = group1_pounds - 39)
  (h3 : group3_ounces = 560)
  (h4 : ∀ (x : ℕ), x * 16 = x * 1 * 16) :
  group1_pounds + group2_pounds + (group3_ounces / 16) = 770 := by
  sorry

end NUMINAMATH_CALUDE_cleanup_drive_total_l3817_381723


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3817_381762

theorem quadratic_no_real_roots (a b c : ℝ) (h : b^2 = a*c) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3817_381762


namespace NUMINAMATH_CALUDE_exam_score_problem_l3817_381772

theorem exam_score_problem (total_questions : ℕ) (correct_marks : ℕ) (wrong_marks : ℕ) (total_score : ℤ) :
  total_questions = 100 →
  correct_marks = 5 →
  wrong_marks = 2 →
  total_score = 210 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    (correct_marks * correct_answers : ℤ) - (wrong_marks * (total_questions - correct_answers) : ℤ) = total_score ∧
    correct_answers = 58 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3817_381772


namespace NUMINAMATH_CALUDE_necklace_divisibility_l3817_381747

/-- The number of ways to make an even number of necklaces -/
def D₀ (n : ℕ) : ℕ := sorry

/-- The number of ways to make an odd number of necklaces -/
def D₁ (n : ℕ) : ℕ := sorry

/-- Theorem: n - 1 divides D₁(n) - D₀(n) for all n ≥ 2 -/
theorem necklace_divisibility (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℤ, D₁ n - D₀ n = k * (n - 1) := by sorry

end NUMINAMATH_CALUDE_necklace_divisibility_l3817_381747


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l3817_381789

theorem tan_alpha_plus_pi_sixth (α : ℝ) 
  (h : Real.cos (3 * Real.pi / 2 - α) = 2 * Real.sin (α + Real.pi / 3)) : 
  Real.tan (α + Real.pi / 6) = -Real.sqrt 3 / 9 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l3817_381789


namespace NUMINAMATH_CALUDE_equivalent_discount_l3817_381749

theorem equivalent_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (equivalent_discount : ℝ) : 
  original_price = 50 →
  discount1 = 0.15 →
  discount2 = 0.10 →
  equivalent_discount = 0.235 →
  original_price * (1 - equivalent_discount) = 
  original_price * (1 - discount1) * (1 - discount2) := by
sorry

end NUMINAMATH_CALUDE_equivalent_discount_l3817_381749


namespace NUMINAMATH_CALUDE_log_xy_value_l3817_381710

open Real

theorem log_xy_value (x y : ℝ) (h1 : log (x * y^4) = 1) (h2 : log (x^3 * y) = 1) : 
  log (x * y) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l3817_381710


namespace NUMINAMATH_CALUDE_amy_muffin_problem_l3817_381718

/-- Represents the number of muffins Amy brings to school each day -/
def muffins_sequence (first_day : ℕ) : ℕ → ℕ
| 0 => first_day
| n + 1 => muffins_sequence first_day n + 1

/-- Calculates the total number of muffins brought to school over 5 days -/
def total_muffins_brought (first_day : ℕ) : ℕ :=
  (List.range 5).map (muffins_sequence first_day) |>.sum

/-- Theorem stating the solution to Amy's muffin problem -/
theorem amy_muffin_problem :
  ∃ (first_day : ℕ),
    total_muffins_brought first_day = 22 - 7 ∧
    first_day = 1 := by
  sorry

end NUMINAMATH_CALUDE_amy_muffin_problem_l3817_381718


namespace NUMINAMATH_CALUDE_mod_31_equivalence_l3817_381711

theorem mod_31_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ 78256 ≡ n [ZMOD 31] ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_mod_31_equivalence_l3817_381711


namespace NUMINAMATH_CALUDE_function_with_two_integer_solutions_l3817_381788

theorem function_with_two_integer_solutions (a : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ 
   (∀ z : ℤ, (Real.log (↑z) - a * (↑z)^2 - (a - 2) * ↑z > 0) ↔ (z = x ∨ z = y))) →
  (1 < a ∧ a ≤ (4 + Real.log 2) / 6) :=
sorry

end NUMINAMATH_CALUDE_function_with_two_integer_solutions_l3817_381788


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3817_381757

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (3 * x + y = 8) ∧ (2 * x - y = 7) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3817_381757


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l3817_381709

theorem complex_magnitude_equation (x : ℝ) :
  x > 0 → Complex.abs (3 + x * Complex.I) = 7 → x = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l3817_381709


namespace NUMINAMATH_CALUDE_trays_from_second_table_l3817_381742

theorem trays_from_second_table
  (trays_per_trip : ℕ)
  (num_trips : ℕ)
  (trays_from_first_table : ℕ)
  (h1 : trays_per_trip = 4)
  (h2 : num_trips = 3)
  (h3 : trays_from_first_table = 10) :
  trays_per_trip * num_trips - trays_from_first_table = 2 :=
by sorry

end NUMINAMATH_CALUDE_trays_from_second_table_l3817_381742


namespace NUMINAMATH_CALUDE_building_height_ratio_l3817_381792

/-- Proves the ratio of building heights given specific conditions -/
theorem building_height_ratio :
  let h₁ : ℝ := 600  -- Height of first building
  let h₂ : ℝ := 2 * h₁  -- Height of second building
  let h_total : ℝ := 7200  -- Total height of all three buildings
  let h₃ : ℝ := h_total - (h₁ + h₂)  -- Height of third building
  (h₃ / (h₁ + h₂) = 3) :=
by sorry

end NUMINAMATH_CALUDE_building_height_ratio_l3817_381792


namespace NUMINAMATH_CALUDE_binomial_expectation_and_variance_l3817_381717

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (b : BinomialDistribution) : ℝ := b.n * b.p

/-- The variance of a binomial distribution -/
def variance (b : BinomialDistribution) : ℝ := b.n * b.p * (1 - b.p)

/-- Theorem: For a binomial distribution with n=200 and p=0.01, 
    the expected value is 2 and the variance is 1.98 -/
theorem binomial_expectation_and_variance :
  ∃ b : BinomialDistribution, 
    b.n = 200 ∧ 
    b.p = 0.01 ∧ 
    expectedValue b = 2 ∧ 
    variance b = 1.98 := by
  sorry


end NUMINAMATH_CALUDE_binomial_expectation_and_variance_l3817_381717


namespace NUMINAMATH_CALUDE_infinite_fraction_value_l3817_381730

theorem infinite_fraction_value : 
  ∃ x : ℝ, x = 3 + 3 / (1 + 5 / x) ∧ x = (1 + Real.sqrt 61) / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_fraction_value_l3817_381730


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l3817_381732

/-- Given a geometric sequence {a_n} with common ratio q = 3,
    if S_3 + S_4 = 53/3, then a_3 = 3 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 3 * a n) →  -- common ratio q = 3
  (∀ n, S n = (a 1) * (3^n - 1) / 2) →  -- sum formula for geometric sequence
  S 3 + S 4 = 53 / 3 →  -- given condition
  a 3 = 3 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_a3_l3817_381732


namespace NUMINAMATH_CALUDE_age_difference_l3817_381739

/-- Given three people a, b, and c, where the total age of a and b is 20 years more than
    the total age of b and c, prove that c is 20 years younger than a. -/
theorem age_difference (a b c : ℕ) (h : a + b = b + c + 20) : a = c + 20 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3817_381739


namespace NUMINAMATH_CALUDE_existence_of_close_pairs_l3817_381770

theorem existence_of_close_pairs :
  ∀ (a b : Fin 7 → ℝ),
  (∀ i, 0 ≤ a i) →
  (∀ i, 0 ≤ b i) →
  (∀ i, a i + b i ≤ 2) →
  ∃ k m, k ≠ m ∧ |a k - a m| + |b k - b m| ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_close_pairs_l3817_381770


namespace NUMINAMATH_CALUDE_degrees_90_to_radians_l3817_381702

/-- Conversion of 90 degrees to radians -/
theorem degrees_90_to_radians : 
  (90 : ℝ) * (Real.pi / 180) = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_degrees_90_to_radians_l3817_381702


namespace NUMINAMATH_CALUDE_probability_eliminate_six_eq_seven_twentysix_l3817_381783

/-- Represents a team in the tournament -/
structure Team :=
  (players : ℕ)

/-- Represents the tournament structure -/
structure Tournament :=
  (teamA : Team)
  (teamB : Team)

/-- Calculates the binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Calculates the probability of one team eliminating exactly 6 players before winning -/
def probability_eliminate_six (t : Tournament) : ℚ :=
  if t.teamA.players = 7 ∧ t.teamB.players = 7 then
    (binomial 12 6 : ℚ) / (2 * (binomial 13 7 : ℚ))
  else
    0

/-- Theorem stating the probability of eliminating 6 players before winning -/
theorem probability_eliminate_six_eq_seven_twentysix (t : Tournament) :
  t.teamA.players = 7 ∧ t.teamB.players = 7 →
  probability_eliminate_six t = 7 / 26 :=
sorry

end NUMINAMATH_CALUDE_probability_eliminate_six_eq_seven_twentysix_l3817_381783


namespace NUMINAMATH_CALUDE_max_value_problem_l3817_381776

theorem max_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 2 * Real.sqrt (x * y) - 4 * x^2 - y^2 ≤ 2 * Real.sqrt (a * b) - 4 * a^2 - b^2) →
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 = (Real.sqrt 2 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l3817_381776


namespace NUMINAMATH_CALUDE_rachels_homework_l3817_381782

theorem rachels_homework (total_pages math_pages : ℕ) 
  (h1 : total_pages = 7) 
  (h2 : math_pages = 5) : 
  total_pages - math_pages = 2 := by
sorry

end NUMINAMATH_CALUDE_rachels_homework_l3817_381782


namespace NUMINAMATH_CALUDE_sum_edges_pyramid_prism_l3817_381707

/-- A triangular pyramid (tetrahedron) -/
structure TriangularPyramid where
  edges : ℕ
  edge_count : edges = 6

/-- A triangular prism -/
structure TriangularPrism where
  edges : ℕ
  edge_count : edges = 9

/-- The sum of edges of a triangular pyramid and a triangular prism is 15 -/
theorem sum_edges_pyramid_prism (p : TriangularPyramid) (q : TriangularPrism) :
  p.edges + q.edges = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_edges_pyramid_prism_l3817_381707


namespace NUMINAMATH_CALUDE_partner_contribution_correct_l3817_381713

/-- Calculates the partner's contribution given the investment details and profit ratio -/
def calculate_partner_contribution (a_investment : ℚ) (a_months : ℚ) (b_months : ℚ) (a_ratio : ℚ) (b_ratio : ℚ) : ℚ :=
  (a_investment * a_months * b_ratio) / (a_ratio * b_months)

theorem partner_contribution_correct :
  let a_investment : ℚ := 3500
  let a_months : ℚ := 12
  let b_months : ℚ := 3
  let a_ratio : ℚ := 2
  let b_ratio : ℚ := 3
  calculate_partner_contribution a_investment a_months b_months a_ratio b_ratio = 21000 := by
  sorry

end NUMINAMATH_CALUDE_partner_contribution_correct_l3817_381713


namespace NUMINAMATH_CALUDE_point_coordinates_on_terminal_side_l3817_381728

/-- Given a point P on the terminal side of angle 4π/3 with |OP| = 4,
    prove that the coordinates of P are (-2, -2√3) -/
theorem point_coordinates_on_terminal_side (P : ℝ × ℝ) :
  (P.1 = 4 * Real.cos (4 * Real.pi / 3) ∧ P.2 = 4 * Real.sin (4 * Real.pi / 3)) →
  P = (-2, -2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_on_terminal_side_l3817_381728


namespace NUMINAMATH_CALUDE_cylinder_base_area_ratio_l3817_381758

/-- Represents a cylinder with base area S and volume V -/
structure Cylinder where
  S : ℝ
  V : ℝ

/-- 
Given two cylinders with equal lateral areas and a volume ratio of 3/2,
prove that the ratio of their base areas is 9/4
-/
theorem cylinder_base_area_ratio 
  (A B : Cylinder) 
  (h1 : A.V / B.V = 3 / 2) 
  (h2 : ∃ (r₁ r₂ h₁ h₂ : ℝ), 
    A.S = π * r₁^2 ∧ 
    B.S = π * r₂^2 ∧ 
    A.V = π * r₁^2 * h₁ ∧ 
    B.V = π * r₂^2 * h₂ ∧ 
    2 * π * r₁ * h₁ = 2 * π * r₂ * h₂) : 
  A.S / B.S = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_cylinder_base_area_ratio_l3817_381758


namespace NUMINAMATH_CALUDE_rationalize_and_product_l3817_381755

theorem rationalize_and_product : ∃ (A B C : ℚ),
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C ∧
  A = 11/4 ∧ B = 5/4 ∧ C = 5 ∧ A * B * C = 275/16 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_product_l3817_381755


namespace NUMINAMATH_CALUDE_larger_square_side_length_l3817_381774

theorem larger_square_side_length 
  (small_square_side : ℝ) 
  (larger_square_perimeter : ℝ) 
  (h1 : small_square_side = 20) 
  (h2 : larger_square_perimeter = 4 * small_square_side + 20) : 
  larger_square_perimeter / 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_square_side_length_l3817_381774


namespace NUMINAMATH_CALUDE_race_tie_l3817_381798

/-- A race between two runners A and B -/
structure Race where
  length : ℝ
  speed_ratio : ℝ
  head_start : ℝ

/-- The race conditions -/
def race_conditions : Race where
  length := 100
  speed_ratio := 4
  head_start := 75

/-- Theorem stating that the given head start results in a tie -/
theorem race_tie (race : Race) (h1 : race.length = 100) (h2 : race.speed_ratio = 4) 
    (h3 : race.head_start = 75) : 
  race.length / race.speed_ratio = (race.length - race.head_start) / 1 := by
  sorry

#check race_tie race_conditions rfl rfl rfl

end NUMINAMATH_CALUDE_race_tie_l3817_381798


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l3817_381773

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Theorem statement
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l3817_381773


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3817_381766

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_properties (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (is_geometric_sequence (fun n => (a n)^2)) ∧
  (is_geometric_sequence (fun n => a (2*n))) ∧
  (is_geometric_sequence (fun n => 1 / (a n))) ∧
  (is_geometric_sequence (fun n => |a n|)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3817_381766


namespace NUMINAMATH_CALUDE_science_fair_participants_l3817_381737

theorem science_fair_participants (total_girls : ℕ) (total_boys : ℕ)
  (girls_participation_rate : ℚ) (boys_participation_rate : ℚ)
  (h1 : total_girls = 150)
  (h2 : total_boys = 100)
  (h3 : girls_participation_rate = 4 / 5)
  (h4 : boys_participation_rate = 3 / 4) :
  let participating_girls : ℚ := girls_participation_rate * total_girls
  let participating_boys : ℚ := boys_participation_rate * total_boys
  let total_participants : ℚ := participating_girls + participating_boys
  participating_girls / total_participants = 8 / 13 := by
sorry

end NUMINAMATH_CALUDE_science_fair_participants_l3817_381737


namespace NUMINAMATH_CALUDE_perimeter_calculation_l3817_381736

theorem perimeter_calculation : 
  let segments : List ℕ := [2, 3, 2, 6, 2, 4, 3]
  segments.sum = 22 := by sorry

end NUMINAMATH_CALUDE_perimeter_calculation_l3817_381736


namespace NUMINAMATH_CALUDE_sqrt_x_minus_8_meaningful_l3817_381724

theorem sqrt_x_minus_8_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 8) ↔ x ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_8_meaningful_l3817_381724


namespace NUMINAMATH_CALUDE_no_four_polynomials_exist_l3817_381780

-- Define a type for polynomials with real coefficients
def RealPolynomial := ℝ → ℝ

-- Define a predicate to check if a polynomial has a real root
def has_real_root (p : RealPolynomial) : Prop :=
  ∃ x : ℝ, p x = 0

-- Define a predicate to check if a polynomial has no real root
def has_no_real_root (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, p x ≠ 0

theorem no_four_polynomials_exist :
  ¬ ∃ (P₁ P₂ P₃ P₄ : RealPolynomial),
    (has_real_root (λ x => P₁ x + P₂ x + P₃ x)) ∧
    (has_real_root (λ x => P₁ x + P₂ x + P₄ x)) ∧
    (has_real_root (λ x => P₁ x + P₃ x + P₄ x)) ∧
    (has_real_root (λ x => P₂ x + P₃ x + P₄ x)) ∧
    (has_no_real_root (λ x => P₁ x + P₂ x)) ∧
    (has_no_real_root (λ x => P₁ x + P₃ x)) ∧
    (has_no_real_root (λ x => P₁ x + P₄ x)) ∧
    (has_no_real_root (λ x => P₂ x + P₃ x)) ∧
    (has_no_real_root (λ x => P₂ x + P₄ x)) ∧
    (has_no_real_root (λ x => P₃ x + P₄ x)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_four_polynomials_exist_l3817_381780


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3817_381734

/-- Sum of a geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric series -/
def a : ℚ := 2

/-- The common ratio of the geometric series -/
def r : ℚ := -2

/-- The number of terms in the geometric series -/
def n : ℕ := 10

theorem geometric_series_sum :
  geometric_sum a r n = 2050 / 3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3817_381734


namespace NUMINAMATH_CALUDE_units_digit_of_subtraction_is_seven_l3817_381777

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Converts a ThreeDigitNumber to its integer value -/
def to_int (n : ThreeDigitNumber) : Int :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Reverses a ThreeDigitNumber -/
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.units
  tens := n.tens
  units := n.hundreds
  is_valid := by sorry

/-- The main theorem -/
theorem units_digit_of_subtraction_is_seven (n : ThreeDigitNumber) 
  (h : n.hundreds = n.units + 3) : 
  (to_int n - to_int (reverse n)) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_subtraction_is_seven_l3817_381777


namespace NUMINAMATH_CALUDE_sum_of_roots_of_quadratic_l3817_381744

theorem sum_of_roots_of_quadratic : ∃ (x₁ x₂ : ℝ), 
  x₁^2 - 6*x₁ + 8 = 0 ∧ 
  x₂^2 - 6*x₂ + 8 = 0 ∧ 
  x₁ ≠ x₂ ∧
  x₁ + x₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_quadratic_l3817_381744


namespace NUMINAMATH_CALUDE_coefficient_sum_l3817_381706

theorem coefficient_sum (x : ℝ) (a₀ a₁ a₂ a₃ : ℝ) 
  (h : (1 - 2/x)^3 = a₀ + a₁*(1/x) + a₂*(1/x)^2 + a₃*(1/x)^3) :
  a₁ + a₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_l3817_381706


namespace NUMINAMATH_CALUDE_triangle_shape_l3817_381725

/-- A triangle with side lengths a, b, and c is either isosceles or right-angled if a^4 - b^4 + (b^2c^2 - a^2c^2) = 0 -/
theorem triangle_shape (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq : a^4 - b^4 + (b^2 * c^2 - a^2 * c^2) = 0) : 
  (a = b) ∨ (a^2 + b^2 = c^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l3817_381725


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3817_381740

theorem algebraic_expression_value (x y : ℝ) 
  (eq1 : x + y = 0.2) 
  (eq2 : x + 3*y = 1) : 
  x^2 + 4*x*y + 4*y^2 = 0.36 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3817_381740


namespace NUMINAMATH_CALUDE_louise_needs_30_boxes_l3817_381751

/-- Represents the number of pencils each box can hold for different colors --/
structure BoxCapacity where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Represents the number of pencils Louise has for each color --/
structure PencilCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the number of boxes needed for a given color --/
def boxesNeeded (capacity : ℕ) (count : ℕ) : ℕ :=
  (count + capacity - 1) / capacity

/-- Calculates the total number of boxes Louise needs --/
def totalBoxesNeeded (capacity : BoxCapacity) (count : PencilCount) : ℕ :=
  boxesNeeded capacity.red count.red +
  boxesNeeded capacity.blue count.blue +
  boxesNeeded capacity.yellow count.yellow +
  boxesNeeded capacity.green count.green

/-- The main theorem stating that Louise needs 30 boxes --/
theorem louise_needs_30_boxes :
  let capacity := BoxCapacity.mk 15 25 10 30
  let redCount := 45
  let blueCount := 3 * redCount + 6
  let yellowCount := 80
  let greenCount := 2 * (redCount + blueCount)
  let count := PencilCount.mk redCount blueCount yellowCount greenCount
  totalBoxesNeeded capacity count = 30 := by
  sorry


end NUMINAMATH_CALUDE_louise_needs_30_boxes_l3817_381751


namespace NUMINAMATH_CALUDE_polynomial_equality_l3817_381790

theorem polynomial_equality (a k n : ℚ) :
  (∀ x : ℚ, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) →
  a - n + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3817_381790


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l3817_381738

-- Define the functions
def y₁ (x : ℝ) : ℝ := x^2 + 2*x + 1
def y₂ (x b : ℝ) : ℝ := x^2 + b*x + 2
def y₃ (x c : ℝ) : ℝ := x^2 + c*x + 3

-- Define the number of roots for each function
def M₁ : ℕ := 1
def M₂ : ℕ := 1
def M₃ : ℕ := 2

-- Theorem statement
theorem intersection_points_theorem 
  (b c : ℝ) 
  (hb : b > 0) 
  (hc : c > 0) 
  (h_bc : b^2 = 2*c) 
  (h_M₁ : ∃! x, y₁ x = 0) 
  (h_M₂ : ∃! x, y₂ x b = 0) : 
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ y₃ x₁ c = 0 ∧ y₃ x₂ c = 0 ∧ ∀ x, y₃ x c = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l3817_381738


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3817_381712

theorem quadratic_equation_roots : ∃ (p q : ℤ),
  (∃ (x₁ x₂ : ℤ), 
    x₁ > 0 ∧ x₂ > 0 ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    p + q = 28) →
  (∃ (x₁ x₂ : ℤ), 
    (x₁ = 30 ∧ x₂ = 2) ∨ (x₁ = 2 ∧ x₂ = 30)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3817_381712


namespace NUMINAMATH_CALUDE_jane_ice_cream_purchase_l3817_381763

/-- The number of ice cream cones Jane purchased -/
def num_ice_cream_cones : ℕ := 15

/-- The number of pudding cups Jane purchased -/
def num_pudding_cups : ℕ := 5

/-- The cost of one ice cream cone in dollars -/
def ice_cream_cost : ℕ := 5

/-- The cost of one pudding cup in dollars -/
def pudding_cost : ℕ := 2

/-- The difference in dollars between ice cream and pudding expenses -/
def expense_difference : ℕ := 65

theorem jane_ice_cream_purchase :
  num_ice_cream_cones * ice_cream_cost = num_pudding_cups * pudding_cost + expense_difference :=
by sorry

end NUMINAMATH_CALUDE_jane_ice_cream_purchase_l3817_381763


namespace NUMINAMATH_CALUDE_chocolate_problem_l3817_381708

theorem chocolate_problem :
  ∃ n : ℕ, n ≥ 150 ∧ n % 17 = 15 ∧ ∀ m : ℕ, m ≥ 150 ∧ m % 17 = 15 → m ≥ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_chocolate_problem_l3817_381708


namespace NUMINAMATH_CALUDE_train_length_proof_l3817_381781

/-- Given a train crossing two platforms with constant speed, prove its length is 70 meters. -/
theorem train_length_proof (
  platform1_length : ℝ)
  (platform2_length : ℝ)
  (time1 : ℝ)
  (time2 : ℝ)
  (h1 : platform1_length = 170)
  (h2 : platform2_length = 250)
  (h3 : time1 = 15)
  (h4 : time2 = 20)
  (h5 : (platform1_length + train_length) / time1 = (platform2_length + train_length) / time2)
  : train_length = 70 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l3817_381781


namespace NUMINAMATH_CALUDE_AC_length_l3817_381745

/-- Right triangle ABC with altitude AH and circle through A and H -/
structure RightTriangleWithCircle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  -- ABC is a right triangle with right angle at A
  right_angle_at_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- AH is an altitude
  AH_perpendicular_BC : (H.1 - A.1) * (C.1 - B.1) + (H.2 - A.2) * (C.2 - B.2) = 0
  -- Circle passes through A, H, X, and Y
  circle_through_AHXY : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (H.1 - center.1)^2 + (H.2 - center.2)^2 = radius^2 ∧
    (X.1 - center.1)^2 + (X.2 - center.2)^2 = radius^2 ∧
    (Y.1 - center.1)^2 + (Y.2 - center.2)^2 = radius^2
  -- X is on AB
  X_on_AB : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ X = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  -- Y is on AC
  Y_on_AC : ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ Y = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- Given lengths
  AX_length : ((X.1 - A.1)^2 + (X.2 - A.2)^2)^(1/2 : ℝ) = 5
  AY_length : ((Y.1 - A.1)^2 + (Y.2 - A.2)^2)^(1/2 : ℝ) = 6
  AB_length : ((B.1 - A.1)^2 + (B.2 - A.2)^2)^(1/2 : ℝ) = 9

/-- Theorem: AC length is 13.5 -/
theorem AC_length (t : RightTriangleWithCircle) : 
  ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)^(1/2 : ℝ) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_AC_length_l3817_381745


namespace NUMINAMATH_CALUDE_smallest_hot_dog_packages_l3817_381784

theorem smallest_hot_dog_packages : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → 5 * m % 7 = 0 → m ≥ n) ∧ 5 * n % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_hot_dog_packages_l3817_381784


namespace NUMINAMATH_CALUDE_power_of_two_100_l3817_381727

theorem power_of_two_100 :
  (10^30 : ℕ) ≤ 2^100 ∧ 2^100 < (10^31 : ℕ) ∧ 2^100 % 1000 = 376 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_100_l3817_381727


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l3817_381735

def Q (n : ℕ) : ℚ := 2 / (n * (n + 1) * (n + 2))

theorem smallest_n_for_Q_less_than_threshold : 
  ∀ k : ℕ, k > 0 → k < 19 → Q (5 * k) ≥ 1 / 2500 ∧ Q (5 * 19) < 1 / 2500 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l3817_381735


namespace NUMINAMATH_CALUDE_pyramid_frustum_problem_l3817_381771

noncomputable def pyramid_frustum_theorem (AB BC height : ℝ) : Prop :=
  AB > 0 ∧ BC > 0 ∧ height > 0 →
  let ABCD := AB * BC
  let P_volume := (1/3) * ABCD * height
  let P'_volume := (1/8) * P_volume
  let F_height := height / 2
  let A'B' := AB / 2
  let B'C' := BC / 2
  let AC := Real.sqrt (AB^2 + BC^2)
  let A'C' := AC / 2
  let h := (73/8 : ℝ)
  let XT := h + F_height
  XT = 169/8

theorem pyramid_frustum_problem :
  pyramid_frustum_theorem 12 16 24 := by sorry

end NUMINAMATH_CALUDE_pyramid_frustum_problem_l3817_381771


namespace NUMINAMATH_CALUDE_xiao_liang_score_l3817_381767

/-- Calculates the comprehensive score for a speech contest given the weights and scores for each aspect. -/
def comprehensive_score (content_weight delivery_weight effectiveness_weight : ℚ)
                        (content_score delivery_score effectiveness_score : ℚ) : ℚ :=
  content_weight * content_score + delivery_weight * delivery_score + effectiveness_weight * effectiveness_score

/-- Theorem stating that Xiao Liang's comprehensive score is 91 points. -/
theorem xiao_liang_score :
  let content_weight : ℚ := 1/2
  let delivery_weight : ℚ := 2/5
  let effectiveness_weight : ℚ := 1/10
  let content_score : ℚ := 88
  let delivery_score : ℚ := 95
  let effectiveness_score : ℚ := 90
  comprehensive_score content_weight delivery_weight effectiveness_weight
                      content_score delivery_score effectiveness_score = 91 := by
  sorry


end NUMINAMATH_CALUDE_xiao_liang_score_l3817_381767


namespace NUMINAMATH_CALUDE_lemonade_pitcher_capacity_l3817_381700

theorem lemonade_pitcher_capacity (total_glasses : ℕ) (total_pitchers : ℕ) 
  (h1 : total_glasses = 30) (h2 : total_pitchers = 6) :
  total_glasses / total_pitchers = 5 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_pitcher_capacity_l3817_381700


namespace NUMINAMATH_CALUDE_woodworker_chairs_l3817_381731

/-- Calculates the number of chairs built given the total number of furniture legs,
    number of tables, legs per table, and legs per chair. -/
def chairs_built (total_legs : ℕ) (num_tables : ℕ) (legs_per_table : ℕ) (legs_per_chair : ℕ) : ℕ :=
  (total_legs - num_tables * legs_per_table) / legs_per_chair

/-- Proves that given 40 total furniture legs, 4 tables, 4 legs per table,
    and 4 legs per chair, the number of chairs built is 6. -/
theorem woodworker_chairs : chairs_built 40 4 4 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_chairs_l3817_381731


namespace NUMINAMATH_CALUDE_expression_evaluation_l3817_381764

theorem expression_evaluation (x : ℝ) (h : x < 0) :
  Real.sqrt (x^2 / (1 + (x + 1) / x)) = Real.sqrt (x^3 / (2*x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3817_381764


namespace NUMINAMATH_CALUDE_ad_fraction_of_page_l3817_381752

theorem ad_fraction_of_page 
  (page_width : ℝ) 
  (page_height : ℝ) 
  (cost_per_sq_inch : ℝ) 
  (total_cost : ℝ) : 
  page_width = 9 → 
  page_height = 12 → 
  cost_per_sq_inch = 8 → 
  total_cost = 432 → 
  (total_cost / cost_per_sq_inch) / (page_width * page_height) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ad_fraction_of_page_l3817_381752


namespace NUMINAMATH_CALUDE_solution_set_l3817_381760

def is_valid (n : ℕ) : Prop :=
  n ≥ 6 ∧ n.choose 4 * 24 ≤ n.factorial / ((n - 4).factorial)

theorem solution_set :
  {n : ℕ | is_valid n} = {6, 7, 8, 9} := by sorry

end NUMINAMATH_CALUDE_solution_set_l3817_381760


namespace NUMINAMATH_CALUDE_fraction_power_and_multiply_l3817_381753

theorem fraction_power_and_multiply :
  (2 / 3 : ℚ) ^ 3 * (1 / 4 : ℚ) = 2 / 27 := by sorry

end NUMINAMATH_CALUDE_fraction_power_and_multiply_l3817_381753


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3817_381750

theorem inequality_equivalence (x : ℝ) : (x - 3) / 2 ≥ 1 ↔ x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3817_381750


namespace NUMINAMATH_CALUDE_system_solutions_l3817_381785

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  lg (x^2 + y^2) = 2 - lg 5 ∧
  lg (x + y) + lg (x - y) = lg 1.2 + 1 ∧
  x + y > 0 ∧
  x - y > 0

-- Theorem statement
theorem system_solutions :
  ∀ x y : ℝ, system x y ↔ ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3817_381785


namespace NUMINAMATH_CALUDE_candles_remaining_l3817_381786

def total_candles : ℕ := 40
def alyssa_fraction : ℚ := 1/2
def chelsea_fraction : ℚ := 7/10

theorem candles_remaining (total : ℕ) (alyssa_frac chelsea_frac : ℚ) :
  total = total_candles →
  alyssa_frac = alyssa_fraction →
  chelsea_frac = chelsea_fraction →
  ↑total * (1 - alyssa_frac) * (1 - chelsea_frac) = 6 :=
by sorry

end NUMINAMATH_CALUDE_candles_remaining_l3817_381786


namespace NUMINAMATH_CALUDE_solution_set_trig_equation_l3817_381733

theorem solution_set_trig_equation :
  {x : ℝ | 3 * Real.sin x = 1 + Real.cos (2 * x)} =
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + (-1)^k * Real.pi / 6} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_trig_equation_l3817_381733


namespace NUMINAMATH_CALUDE_multiply_by_97_preserves_form_l3817_381704

theorem multiply_by_97_preserves_form (a b : ℕ) :
  ∃ (a' b' : ℕ), 97 * (3 * a^2 + 32 * b^2) = 3 * a'^2 + 32 * b'^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_97_preserves_form_l3817_381704


namespace NUMINAMATH_CALUDE_rectangle_area_l3817_381741

theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let perimeter := 2 * (l + b)
  perimeter = 64 → l * b = 192 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3817_381741


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_min_value_sum_of_reciprocals_achieved_l3817_381765

theorem min_value_sum_of_reciprocals (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : x + y = 4) :
  (1 / (x - 1) + 1 / (y - 1)) ≥ 2 :=
sorry

theorem min_value_sum_of_reciprocals_achieved (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : x + y = 4) :
  (1 / (x - 1) + 1 / (y - 1) = 2) ↔ (x = 2 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_min_value_sum_of_reciprocals_achieved_l3817_381765


namespace NUMINAMATH_CALUDE_prize_probability_l3817_381794

/-- The probability of at least one person winning a prize when 5 people each buy 1 ticket
    from a pool of 10 tickets, where 3 tickets have prizes. -/
theorem prize_probability (total_tickets : ℕ) (prize_tickets : ℕ) (buyers : ℕ) :
  total_tickets = 10 →
  prize_tickets = 3 →
  buyers = 5 →
  (1 : ℚ) - (Nat.choose (total_tickets - prize_tickets) buyers : ℚ) / (Nat.choose total_tickets buyers : ℚ) = 11/12 :=
by sorry

end NUMINAMATH_CALUDE_prize_probability_l3817_381794


namespace NUMINAMATH_CALUDE_mul_exp_analogy_l3817_381721

-- Define multiplication recursively
def mul_rec (k : ℕ) : ℕ → ℕ
| 0     => 0                   -- Base case
| n + 1 => k + mul_rec k n     -- Recursive step

-- Define exponentiation recursively
def exp_rec (k : ℕ) : ℕ → ℕ
| 0     => 1                   -- Base case
| n + 1 => k * exp_rec k n     -- Recursive step

-- Theorem stating the analogy between multiplication and exponentiation
theorem mul_exp_analogy :
  (∀ k n : ℕ, mul_rec k (n + 1) = k + mul_rec k n) ↔
  (∀ k n : ℕ, exp_rec k (n + 1) = k * exp_rec k n) :=
sorry

end NUMINAMATH_CALUDE_mul_exp_analogy_l3817_381721


namespace NUMINAMATH_CALUDE_factor_polynomial_l3817_381796

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3817_381796


namespace NUMINAMATH_CALUDE_expression_evaluation_l3817_381754

theorem expression_evaluation : 
  20 * ((150 / 3) + (40 / 5) + (16 / 25) + 2) = 1212.8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3817_381754


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3817_381714

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 4 → ¬(5 ∣ (2496 + m))) ∧ (5 ∣ (2496 + 4)) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3817_381714


namespace NUMINAMATH_CALUDE_inequality_proof_l3817_381716

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c)^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3817_381716
