import Mathlib

namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1235_123574

theorem square_sum_given_difference_and_product (a b : ℝ) 
  (h1 : a - b = 2) 
  (h2 : a * b = 10.5) : 
  a^2 + b^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1235_123574


namespace NUMINAMATH_CALUDE_divisor_sum_equality_implies_prime_power_l1235_123586

/-- σ(N) is the sum of the positive integer divisors of N -/
def sigma (N : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem divisor_sum_equality_implies_prime_power (m n : ℕ) :
  m ≥ n → n ≥ 2 →
  (sigma m - 1) / (m - 1) = (sigma n - 1) / (n - 1) →
  (sigma m - 1) / (m - 1) = (sigma (m * n) - 1) / (m * n - 1) →
  ∃ (p : ℕ) (e f : ℕ), Prime p ∧ e ≥ f ∧ f ≥ 1 ∧ m = p^e ∧ n = p^f :=
sorry

end NUMINAMATH_CALUDE_divisor_sum_equality_implies_prime_power_l1235_123586


namespace NUMINAMATH_CALUDE_female_employees_count_l1235_123523

/-- The total number of female employees in a company, given specific conditions. -/
theorem female_employees_count (total_employees : ℕ) (male_employees : ℕ) (female_managers : ℕ) :
  female_managers = 280 →
  (2 : ℚ) / 5 * total_employees = female_managers + (2 : ℚ) / 5 * male_employees →
  total_employees = male_employees + 700 →
  700 = total_employees - male_employees :=
by sorry

end NUMINAMATH_CALUDE_female_employees_count_l1235_123523


namespace NUMINAMATH_CALUDE_sand_truck_loads_l1235_123537

/-- Proves that the truck-loads of sand required is equal to 0.1666666666666666,
    given the total truck-loads of material needed and the truck-loads of dirt and cement. -/
theorem sand_truck_loads (total material_needed dirt cement sand : ℚ)
    (h1 : total = 0.6666666666666666)
    (h2 : dirt = 0.3333333333333333)
    (h3 : cement = 0.16666666666666666)
    (h4 : sand = total - (dirt + cement)) :
    sand = 0.1666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_sand_truck_loads_l1235_123537


namespace NUMINAMATH_CALUDE_fraction_ordering_l1235_123503

theorem fraction_ordering : (25 : ℚ) / 19 < 21 / 16 ∧ 21 / 16 < 23 / 17 := by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l1235_123503


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1235_123518

theorem quadratic_equation_solution : 
  ∀ x : ℝ, (2 * x^2 + 10 * x + 12 = -(x + 4) * (x + 6)) ↔ (x = -4 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1235_123518


namespace NUMINAMATH_CALUDE_average_of_pqrs_l1235_123568

theorem average_of_pqrs (p q r s : ℝ) (h : (5 / 4) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_pqrs_l1235_123568


namespace NUMINAMATH_CALUDE_telescope_visual_range_l1235_123571

theorem telescope_visual_range 
  (original_range : ℝ) 
  (percentage_increase : ℝ) 
  (new_range : ℝ) : 
  original_range = 50 → 
  percentage_increase = 200 → 
  new_range = original_range + (percentage_increase / 100) * original_range → 
  new_range = 150 := by
sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l1235_123571


namespace NUMINAMATH_CALUDE_two_row_arrangement_count_l1235_123533

/-- The number of permutations of k items chosen from n items -/
def permutations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.descFactorial n k

theorem two_row_arrangement_count
  (n k k₁ k₂ : ℕ)
  (h₁ : k₁ + k₂ = k)
  (h₂ : 1 ≤ k)
  (h₃ : k ≤ n) :
  (permutations n k₁) * (permutations (n - k₁) k₂) = permutations n k :=
sorry

end NUMINAMATH_CALUDE_two_row_arrangement_count_l1235_123533


namespace NUMINAMATH_CALUDE_min_distance_sum_l1235_123535

open Complex

theorem min_distance_sum (z₁ z₂ : ℂ) (h₁ : z₁ = -Real.sqrt 3 - I) (h₂ : z₂ = 3 + Real.sqrt 3 * I) :
  (∃ (θ : ℝ), ∀ (z : ℂ), z = (2 + Real.cos θ) + I * Real.sin θ →
    ∀ (w : ℂ), abs (w - z₁) + abs (w - z₂) ≥ abs (z - z₁) + abs (z - z₂)) ∧
  (∃ (z : ℂ), abs (z - z₁) + abs (z - z₂) = 2 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1235_123535


namespace NUMINAMATH_CALUDE_sin_585_degrees_l1235_123526

theorem sin_585_degrees : Real.sin (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l1235_123526


namespace NUMINAMATH_CALUDE_sequence_property_l1235_123529

/-- Given a sequence {a_n} with sum of first n terms S_n = 2a_n - a_1,
    and a_1, a_2+1, a_3 form an arithmetic sequence, prove a_n = 2^n -/
theorem sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = 2 * a n - a 1) → 
  (2 * (a 2 + 1) = a 3 + a 1) →
  ∀ n, a n = 2^n := by sorry

end NUMINAMATH_CALUDE_sequence_property_l1235_123529


namespace NUMINAMATH_CALUDE_number_equation_solution_l1235_123522

theorem number_equation_solution : 
  ∃! x : ℝ, 45 - (28 - (37 - (x - 18))) = 57 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1235_123522


namespace NUMINAMATH_CALUDE_solve_matrix_inverse_l1235_123508

def matrix_inverse_problem (c d x y : ℚ) : Prop :=
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, c; x, 13]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![13, y; 3, d]
  (A * B = 1) → (x = -3 ∧ y = 17/4 ∧ c + d = -16)

theorem solve_matrix_inverse :
  ∃ c d x y : ℚ, matrix_inverse_problem c d x y :=
sorry

end NUMINAMATH_CALUDE_solve_matrix_inverse_l1235_123508


namespace NUMINAMATH_CALUDE_closed_map_from_compact_preimage_l1235_123573

open Set
open TopologicalSpace
open MetricSpace
open ContinuousMap

theorem closed_map_from_compact_preimage
  {X Y : Type*} [MetricSpace X] [MetricSpace Y]
  (f : C(X, Y))
  (h : ∀ (K : Set Y), IsCompact K → IsCompact (f ⁻¹' K)) :
  ∀ (C : Set X), IsClosed C → IsClosed (f '' C) :=
by sorry

end NUMINAMATH_CALUDE_closed_map_from_compact_preimage_l1235_123573


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l1235_123515

-- Define the quadratic equation
def quadratic (m : ℤ) (x : ℤ) : Prop :=
  x^2 - 2*(2*m-3)*x + 4*m^2 - 14*m + 8 = 0

-- Define the theorem
theorem quadratic_roots_theorem :
  ∀ m : ℤ, 4 < m → m < 40 →
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ quadratic m x₁ ∧ quadratic m x₂) →
  ((m = 12 ∧ ∃ x₁ x₂ : ℤ, x₁ = 26 ∧ x₂ = 16 ∧ quadratic m x₁ ∧ quadratic m x₂) ∨
   (m = 24 ∧ ∃ x₁ x₂ : ℤ, x₁ = 52 ∧ x₂ = 38 ∧ quadratic m x₁ ∧ quadratic m x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l1235_123515


namespace NUMINAMATH_CALUDE_evaluate_expression_l1235_123502

theorem evaluate_expression : -(21 / 3^2 * 7 - 84 + 4 * 7) = -49 / 3 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1235_123502


namespace NUMINAMATH_CALUDE_jelly_bean_distribution_l1235_123506

theorem jelly_bean_distribution (total : ℕ) (x y : ℕ) : 
  total = 1200 →
  x + y = total →
  x = 3 * y - 400 →
  x = 800 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_distribution_l1235_123506


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1235_123553

def A : Set ℕ := {2, 4, 6, 8}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1235_123553


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l1235_123563

theorem sum_of_reciprocals_negative (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_eight : a * b * c = 8) : 
  1 / a + 1 / b + 1 / c < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l1235_123563


namespace NUMINAMATH_CALUDE_simplify_quadratic_radical_l1235_123597

theorem simplify_quadratic_radical (x y : ℝ) (h : x * y < 0) :
  x * Real.sqrt (-y / x^2) = Real.sqrt (-y) :=
sorry

end NUMINAMATH_CALUDE_simplify_quadratic_radical_l1235_123597


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_first_five_primes_l1235_123552

theorem smallest_four_digit_divisible_by_first_five_primes :
  ∃ (n : ℕ), (n ≥ 1000 ∧ n < 10000) ∧
             (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m → n ≤ m) ∧
             2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧
             n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_first_five_primes_l1235_123552


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1235_123575

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let perimeter : ℝ := 3 * side_length
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  area / perimeter = (5 * Real.sqrt 3) / 6 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1235_123575


namespace NUMINAMATH_CALUDE_power_four_inequality_l1235_123513

theorem power_four_inequality (x y : ℝ) : x^4 + y^4 ≥ x*y*(x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_power_four_inequality_l1235_123513


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l1235_123591

theorem hot_dogs_remainder : 25197621 % 4 = 1 := by sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l1235_123591


namespace NUMINAMATH_CALUDE_nonagon_perimeter_l1235_123546

theorem nonagon_perimeter : 
  let side_lengths : List ℕ := [2, 2, 3, 3, 1, 3, 2, 2, 2]
  List.sum side_lengths = 20 := by sorry

end NUMINAMATH_CALUDE_nonagon_perimeter_l1235_123546


namespace NUMINAMATH_CALUDE_f_minus_g_zero_iff_k_eq_9_4_l1235_123520

/-- The function f(x) = 5x^2 - 3x + 2 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 2

/-- The function g(x) = x^3 - 2x^2 + kx - 10 -/
def g (k x : ℝ) : ℝ := x^3 - 2 * x^2 + k * x - 10

/-- Theorem stating that f(5) - g(5) = 0 if and only if k = 9.4 -/
theorem f_minus_g_zero_iff_k_eq_9_4 : 
  ∀ k : ℝ, f 5 - g k 5 = 0 ↔ k = 9.4 := by sorry

end NUMINAMATH_CALUDE_f_minus_g_zero_iff_k_eq_9_4_l1235_123520


namespace NUMINAMATH_CALUDE_pauls_garage_sale_l1235_123538

/-- The number of books Paul sold in the garage sale -/
def books_sold (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : ℕ :=
  initial - given_away - remaining

/-- Proof that Paul sold 27 books in the garage sale -/
theorem pauls_garage_sale : books_sold 134 39 68 = 27 := by
  sorry

end NUMINAMATH_CALUDE_pauls_garage_sale_l1235_123538


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1235_123588

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 6 * x > 23) → x ≤ -3 ∧ 7 - 6 * (-3) > 23 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1235_123588


namespace NUMINAMATH_CALUDE_percentage_first_division_l1235_123545

theorem percentage_first_division (total_students : ℕ) 
  (second_division_percent : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  second_division_percent = 54 / 100 →
  just_passed = 48 →
  ∃ (first_division_percent : ℚ),
    first_division_percent + second_division_percent + (just_passed : ℚ) / total_students = 1 ∧
    first_division_percent = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_first_division_l1235_123545


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l1235_123559

theorem hot_dogs_remainder : 35876119 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l1235_123559


namespace NUMINAMATH_CALUDE_operation_results_in_zero_in_quotient_l1235_123569

-- Define the arithmetic operation
def operation : ℕ → ℕ → ℕ := (·+·)

-- Define the property of having a zero in the middle of the quotient
def has_zero_in_middle_of_quotient (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * 10 + 0 * 1 + b ∧ 0 < b ∧ b < 10

-- Theorem statement
theorem operation_results_in_zero_in_quotient :
  has_zero_in_middle_of_quotient (operation 6 4 / 3) :=
sorry

end NUMINAMATH_CALUDE_operation_results_in_zero_in_quotient_l1235_123569


namespace NUMINAMATH_CALUDE_partition_theorem_l1235_123540

def is_valid_partition (n : ℕ) : Prop :=
  ∃ (partition : List (Fin n × Fin n × Fin n)),
    (∀ (i j : Fin n), i ≠ j → (∃ (t : Fin n × Fin n × Fin n), t ∈ partition ∧ (i = t.1 ∨ i = t.2.1 ∨ i = t.2.2)) →
                              (∃ (t : Fin n × Fin n × Fin n), t ∈ partition ∧ (j = t.1 ∨ j = t.2.1 ∨ j = t.2.2)) →
                              (∀ (t : Fin n × Fin n × Fin n), t ∈ partition → (i = t.1 ∨ i = t.2.1 ∨ i = t.2.2) →
                                                                              (j ≠ t.1 ∧ j ≠ t.2.1 ∧ j ≠ t.2.2))) ∧
    (∀ (t : Fin n × Fin n × Fin n), t ∈ partition → t.1.val + t.2.1.val = t.2.2.val ∨
                                                    t.1.val + t.2.2.val = t.2.1.val ∨
                                                    t.2.1.val + t.2.2.val = t.1.val)

theorem partition_theorem (n : ℕ) (h : n ∈ Finset.range 10 ∪ {3900}) : 
  is_valid_partition n ↔ n = 3900 ∨ n = 3903 :=
sorry

end NUMINAMATH_CALUDE_partition_theorem_l1235_123540


namespace NUMINAMATH_CALUDE_students_using_red_l1235_123531

/-- Given a group of students painting a picture, calculate the number using red color. -/
theorem students_using_red (total green both : ℕ) (h1 : total = 70) (h2 : green = 52) (h3 : both = 38) :
  total = green + (green + both - total) - both → green + both - total = 56 := by
  sorry

end NUMINAMATH_CALUDE_students_using_red_l1235_123531


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l1235_123548

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ k < 3, k > 0 → (Nat.gcd k 63 = 1 ∨ Nat.gcd k 66 = 1)) ∧ 
  Nat.gcd 3 63 > 1 ∧ 
  Nat.gcd 3 66 > 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l1235_123548


namespace NUMINAMATH_CALUDE_road_trip_distance_ratio_l1235_123504

theorem road_trip_distance_ratio : 
  ∀ (tracy michelle katie : ℕ),
  tracy + michelle + katie = 1000 →
  tracy = 2 * michelle + 20 →
  michelle = 294 →
  (michelle : ℚ) / (katie : ℚ) = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_road_trip_distance_ratio_l1235_123504


namespace NUMINAMATH_CALUDE_wednesdays_in_jan_feb_2012_l1235_123525

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in the year 2012 -/
structure Date2012 where
  month : Nat
  day : Nat

/-- Returns the day of the week for a given date in 2012 -/
def dayOfWeek (d : Date2012) : DayOfWeek :=
  sorry

/-- Returns the number of days in a given month of 2012 -/
def daysInMonth (m : Nat) : Nat :=
  sorry

/-- Counts the number of Wednesdays in a given month of 2012 -/
def countWednesdays (month : Nat) : Nat :=
  sorry

theorem wednesdays_in_jan_feb_2012 :
  (dayOfWeek ⟨1, 1⟩ = DayOfWeek.Sunday) →
  (countWednesdays 1 = 4 ∧ countWednesdays 2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_wednesdays_in_jan_feb_2012_l1235_123525


namespace NUMINAMATH_CALUDE_negation_equivalence_l1235_123554

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ ≤ 0 ∧ x₀^2 ≥ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1235_123554


namespace NUMINAMATH_CALUDE_range_of_m_l1235_123585

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, (m + 1) * (x^2 + 1) ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (¬(p m ∧ q m)) → (m ≤ -2 ∨ m > -1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1235_123585


namespace NUMINAMATH_CALUDE_company_kw_price_percentage_l1235_123592

theorem company_kw_price_percentage (price kw : ℝ) (assets_a assets_b : ℝ) 
  (h1 : price = 2 * assets_b)
  (h2 : price = 0.75 * (assets_a + assets_b))
  (h3 : ∃ x : ℝ, price = assets_a * (1 + x / 100)) :
  ∃ x : ℝ, x = 20 ∧ price = assets_a * (1 + x / 100) :=
sorry

end NUMINAMATH_CALUDE_company_kw_price_percentage_l1235_123592


namespace NUMINAMATH_CALUDE_original_paint_intensity_l1235_123567

/-- Proves that the intensity of the original red paint was 45% given the specified conditions. -/
theorem original_paint_intensity
  (replace_fraction : Real)
  (replacement_solution_intensity : Real)
  (final_intensity : Real)
  (h1 : replace_fraction = 0.25)
  (h2 : replacement_solution_intensity = 0.25)
  (h3 : final_intensity = 0.40) :
  ∃ (original_intensity : Real),
    original_intensity = 0.45 ∧
    (1 - replace_fraction) * original_intensity +
    replace_fraction * replacement_solution_intensity = final_intensity :=
by
  sorry

end NUMINAMATH_CALUDE_original_paint_intensity_l1235_123567


namespace NUMINAMATH_CALUDE_product_mod_thousand_l1235_123510

theorem product_mod_thousand : (1234 * 5678) % 1000 = 652 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_thousand_l1235_123510


namespace NUMINAMATH_CALUDE_singing_percentage_is_32_l1235_123587

/-- Represents the rehearsal schedule and calculates the percentage of time spent singing -/
def rehearsal_schedule (total_time warm_up_time notes_time words_time : ℕ) : ℚ :=
  let singing_time := total_time - warm_up_time - notes_time - words_time
  (singing_time : ℚ) / total_time * 100

/-- Theorem stating that the percentage of time spent singing is 32% -/
theorem singing_percentage_is_32 :
  ∃ (words_time : ℕ), rehearsal_schedule 75 6 30 words_time = 32 := by
  sorry


end NUMINAMATH_CALUDE_singing_percentage_is_32_l1235_123587


namespace NUMINAMATH_CALUDE_tan_difference_identity_l1235_123595

theorem tan_difference_identity (n : ℝ) : 
  Real.tan ((n + 1) * π / 180) - Real.tan (n * π / 180) = 
  Real.sin (π / 180) / (Real.cos (n * π / 180) * Real.cos ((n + 1) * π / 180)) := by
sorry

end NUMINAMATH_CALUDE_tan_difference_identity_l1235_123595


namespace NUMINAMATH_CALUDE_sum_m_n_equals_51_l1235_123509

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The smallest positive integer with only two positive divisors -/
def m : ℕ := sorry

/-- The largest integer less than 50 with exactly three positive divisors -/
def n : ℕ := sorry

theorem sum_m_n_equals_51 : m + n = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_51_l1235_123509


namespace NUMINAMATH_CALUDE_jana_walking_distance_l1235_123547

/-- Given a walking speed of 1 mile per 24 minutes, prove that the distance walked in 36 minutes is 1.5 miles. -/
theorem jana_walking_distance (speed : ℚ) (time : ℕ) (distance : ℚ) : 
  speed = 1 / 24 → time = 36 → distance = speed * time → distance = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_jana_walking_distance_l1235_123547


namespace NUMINAMATH_CALUDE_partner_b_investment_l1235_123589

/-- Calculates the investment of partner B in a partnership business. -/
theorem partner_b_investment
  (a_investment : ℕ)
  (c_investment : ℕ)
  (total_profit : ℕ)
  (a_profit_share : ℕ)
  (h1 : a_investment = 6300)
  (h2 : c_investment = 10500)
  (h3 : total_profit = 12600)
  (h4 : a_profit_share = 3780) :
  ∃ b_investment : ℕ,
    b_investment = 13700 ∧
    (a_investment : ℚ) / (a_investment + b_investment + c_investment : ℚ) =
    (a_profit_share : ℚ) / (total_profit : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_partner_b_investment_l1235_123589


namespace NUMINAMATH_CALUDE_distance_minus_nine_to_nine_l1235_123536

-- Define the distance function for points on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem distance_minus_nine_to_nine : distance (-9) 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_minus_nine_to_nine_l1235_123536


namespace NUMINAMATH_CALUDE_range_of_a_l1235_123583

-- Define the complex number z
def z (x a : ℝ) : ℂ := x + (x - a) * Complex.I

-- Define the condition that |z| > |z+i| for all x in (1,2)
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 1 < x ∧ x < 2 → Complex.abs (z x a) > Complex.abs (z x a + Complex.I)

-- Theorem statement
theorem range_of_a (a : ℝ) : condition a → a ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1235_123583


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1235_123541

theorem quadratic_equation_properties (k : ℝ) :
  (∃ x y : ℝ, x^2 - k*x + k - 1 = 0 ∧ y^2 - k*y + k - 1 = 0 ∧ (x = y ∨ x ≠ y)) ∧
  (∃ x : ℝ, x^2 - k*x + k - 1 = 0 ∧ x < 0) → k < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1235_123541


namespace NUMINAMATH_CALUDE_ten_percent_of_400_minus_25_l1235_123561

theorem ten_percent_of_400_minus_25 : 
  (400 * (10 : ℝ) / 100) - 25 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ten_percent_of_400_minus_25_l1235_123561


namespace NUMINAMATH_CALUDE_max_value_theorem_l1235_123581

-- Define the optimization problem
def optimization_problem (a b : ℝ) : Prop :=
  4 * a + 3 * b ≤ 10 ∧ 3 * a + 5 * b ≤ 11

-- State the theorem
theorem max_value_theorem :
  ∃ (max : ℝ), max = 48 / 11 ∧
  ∀ (a b : ℝ), optimization_problem a b →
  2 * a + b ≤ max ∧
  ∃ (a₀ b₀ : ℝ), optimization_problem a₀ b₀ ∧ 2 * a₀ + b₀ = max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1235_123581


namespace NUMINAMATH_CALUDE_equation_two_roots_l1235_123517

-- Define the equation
def equation (x k : ℂ) : Prop :=
  x / (x + 3) + x / (x + 4) = k * x

-- Define the condition for having exactly two distinct roots
def has_two_distinct_roots (k : ℂ) : Prop :=
  ∃ x y : ℂ, x ≠ y ∧ equation x k ∧ equation y k ∧
  ∀ z : ℂ, equation z k → z = x ∨ z = y

-- State the theorem
theorem equation_two_roots :
  ∀ k : ℂ, has_two_distinct_roots k ↔ k = 7/12 ∨ k = 2*I ∨ k = -2*I :=
sorry

end NUMINAMATH_CALUDE_equation_two_roots_l1235_123517


namespace NUMINAMATH_CALUDE_shop_width_l1235_123560

/-- Proves that the width of a rectangular shop is 8 feet given the specified conditions. -/
theorem shop_width (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_sqft : ℕ) 
  (h1 : monthly_rent = 2400)
  (h2 : length = 10)
  (h3 : annual_rent_per_sqft = 360) :
  monthly_rent * 12 / (annual_rent_per_sqft * length) = 8 :=
by sorry

end NUMINAMATH_CALUDE_shop_width_l1235_123560


namespace NUMINAMATH_CALUDE_janice_stairs_l1235_123534

/-- The number of flights of stairs to reach Janice's office -/
def flights_per_staircase : ℕ := 3

/-- The number of times Janice goes up the stairs in a day -/
def times_up : ℕ := 5

/-- The number of times Janice goes down the stairs in a day -/
def times_down : ℕ := 3

/-- The total number of flights Janice walks up in a day -/
def flights_up : ℕ := flights_per_staircase * times_up

/-- The total number of flights Janice walks down in a day -/
def flights_down : ℕ := flights_per_staircase * times_down

/-- The total number of flights Janice walks in a day -/
def total_flights : ℕ := flights_up + flights_down

theorem janice_stairs : total_flights = 24 := by
  sorry

end NUMINAMATH_CALUDE_janice_stairs_l1235_123534


namespace NUMINAMATH_CALUDE_annika_hiking_time_l1235_123580

/-- Annika's hiking problem -/
theorem annika_hiking_time (rate : ℝ) (initial_distance : ℝ) (total_distance_east : ℝ) : 
  rate = 10 →
  initial_distance = 2.75 →
  total_distance_east = 3.625 →
  (2 * total_distance_east) * rate = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_annika_hiking_time_l1235_123580


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1235_123543

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (a5_eq_6 : a 5 = 6)
  (a3_eq_2 : a 3 = 2) :
  ∀ n, a (n + 1) - a n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1235_123543


namespace NUMINAMATH_CALUDE_greatest_n_value_l1235_123590

theorem greatest_n_value (n : ℤ) (h : 93 * n^3 ≤ 145800) : n ≤ 11 ∧ ∃ (m : ℤ), m = 11 ∧ 93 * m^3 ≤ 145800 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_l1235_123590


namespace NUMINAMATH_CALUDE_marys_cake_flour_l1235_123570

/-- Given a cake recipe that requires a certain amount of flour and an amount already added,
    calculate the remaining amount to be added. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- Prove that for Mary's cake, which requires 8 cups of flour and has 2 cups already added,
    the remaining amount to be added is 6 cups. -/
theorem marys_cake_flour : remaining_flour 8 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_marys_cake_flour_l1235_123570


namespace NUMINAMATH_CALUDE_f_five_eq_zero_l1235_123576

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f x

-- State the theorem
theorem f_five_eq_zero : f 5 = 0 := by sorry

end NUMINAMATH_CALUDE_f_five_eq_zero_l1235_123576


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1235_123532

/-- A triangle with two sides of lengths 3 and 4, and the third side length being a root of x^2 - 12x + 35 = 0 has a perimeter of 12. -/
theorem triangle_perimeter : ∃ (a b c : ℝ), 
  a = 3 ∧ b = 4 ∧ c^2 - 12*c + 35 = 0 ∧ 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  a + b + c = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1235_123532


namespace NUMINAMATH_CALUDE_sum_of_slopes_constant_l1235_123505

/-- An ellipse with eccentricity 1/2 passing through (2,0) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_ecc : a^2 - b^2 = (a/2)^2
  h_thru_point : 4/a^2 + 0/b^2 = 1

/-- A line passing through (1,0) intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  h_intersect : ∃ x y, x^2/E.a^2 + y^2/E.b^2 = 1 ∧ y = k*(x-1)

/-- The point P -/
def P : ℝ × ℝ := (4, 3)

/-- Slopes of PA and PB -/
def slopes (E : Ellipse) (L : IntersectingLine E) : ℝ × ℝ :=
  sorry

/-- The theorem to be proved -/
theorem sum_of_slopes_constant (E : Ellipse) (L : IntersectingLine E) :
  let (k₁, k₂) := slopes E L
  k₁ + k₂ = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_slopes_constant_l1235_123505


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1235_123519

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1235_123519


namespace NUMINAMATH_CALUDE_greatest_common_multiple_15_10_under_150_l1235_123524

def is_common_multiple (m n k : ℕ) : Prop := k % m = 0 ∧ k % n = 0

theorem greatest_common_multiple_15_10_under_150 :
  ∃ (k : ℕ), k < 150 ∧ 
             is_common_multiple 15 10 k ∧ 
             ∀ (j : ℕ), j < 150 → is_common_multiple 15 10 j → j ≤ k :=
by
  use 120
  sorry

#eval 120  -- Expected output: 120

end NUMINAMATH_CALUDE_greatest_common_multiple_15_10_under_150_l1235_123524


namespace NUMINAMATH_CALUDE_boat_purchase_payment_l1235_123564

theorem boat_purchase_payment (w x y z : ℝ) : 
  w + x + y + z = 60 ∧
  w = (1/2) * (x + y + z) ∧
  x = (1/3) * (w + y + z) ∧
  y = (1/4) * (w + x + z) →
  z = 13 := by sorry

end NUMINAMATH_CALUDE_boat_purchase_payment_l1235_123564


namespace NUMINAMATH_CALUDE_fishing_earnings_l1235_123539

/-- Calculates the total earnings from fishing over a period including a specific day --/
theorem fishing_earnings (rate : ℝ) (past_catch : ℝ) (today_multiplier : ℝ) :
  let past_earnings := rate * past_catch
  let today_catch := past_catch * today_multiplier
  let today_earnings := rate * today_catch
  let total_earnings := past_earnings + today_earnings
  (rate = 20 ∧ past_catch = 80 ∧ today_multiplier = 2) →
  total_earnings = 4800 :=
by
  sorry

end NUMINAMATH_CALUDE_fishing_earnings_l1235_123539


namespace NUMINAMATH_CALUDE_equation_solutions_l1235_123596

theorem equation_solutions :
  (∀ x : ℝ, 2 * (x + 1)^2 = 8 ↔ x = 1 ∨ x = -3) ∧
  (∀ x : ℝ, 2 * x^2 - x - 6 = 0 ↔ x = -3/2 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1235_123596


namespace NUMINAMATH_CALUDE_caitlin_bracelets_l1235_123501

/-- The number of bracelets Caitlin can make given the conditions -/
def num_bracelets : ℕ :=
  let total_beads : ℕ := 528
  let large_beads_per_bracelet : ℕ := 12
  let small_beads_per_bracelet : ℕ := 2 * large_beads_per_bracelet
  let large_beads : ℕ := total_beads / 2
  let small_beads : ℕ := total_beads / 2
  min (large_beads / large_beads_per_bracelet) (small_beads / small_beads_per_bracelet)

/-- Theorem stating that Caitlin can make 22 bracelets -/
theorem caitlin_bracelets : num_bracelets = 22 := by
  sorry

end NUMINAMATH_CALUDE_caitlin_bracelets_l1235_123501


namespace NUMINAMATH_CALUDE_max_x5_value_l1235_123598

theorem max_x5_value (x₁ x₂ x₃ x₄ x₅ : ℕ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h_eq : x₁ + x₂ + x₃ + x₄ + x₅ = x₁ * x₂ * x₃ * x₄ * x₅) :
  x₅ ≤ 5 := by
  sorry

#check max_x5_value

end NUMINAMATH_CALUDE_max_x5_value_l1235_123598


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1235_123550

theorem imaginary_part_of_complex_expression :
  Complex.im ((2 * Complex.I) / (1 - Complex.I) + 2) = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1235_123550


namespace NUMINAMATH_CALUDE_ounces_per_cup_ounces_per_cup_is_eight_l1235_123562

/-- The number of ounces in a cup, given Cassie's water consumption habits -/
theorem ounces_per_cup : ℕ :=
  let cups_per_day : ℕ := 12
  let bottle_capacity : ℕ := 16
  let refills_per_day : ℕ := 6
  (refills_per_day * bottle_capacity) / cups_per_day

/-- Proof that the number of ounces in a cup is 8 -/
theorem ounces_per_cup_is_eight : ounces_per_cup = 8 := by
  sorry

end NUMINAMATH_CALUDE_ounces_per_cup_ounces_per_cup_is_eight_l1235_123562


namespace NUMINAMATH_CALUDE_range_of_a_l1235_123528

def S (a : ℝ) : Set ℝ := {x | 2 * a * x^2 - x ≤ 0}

def T (a : ℝ) : Set ℝ := {x | 4 * a * x^2 - 4 * a * (1 - 2 * a) * x + 1 ≥ 0}

theorem range_of_a (a : ℝ) (h : S a ∪ T a = Set.univ) : 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1235_123528


namespace NUMINAMATH_CALUDE_watch_cost_price_l1235_123599

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (cp : ℚ), 
  (cp * (1 - 1/10) = cp * 0.9) ∧ 
  (cp * (1 + 1/10) = cp * 1.1) ∧ 
  (cp * 1.1 - cp * 0.9 = 500) ∧ 
  cp = 2500 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1235_123599


namespace NUMINAMATH_CALUDE_perfect_square_units_mod_16_l1235_123549

theorem perfect_square_units_mod_16 : 
  ∃ (S : Finset ℕ), (∀ n : ℕ, ∃ m : ℕ, n ^ 2 % 16 ∈ S) ∧ S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_units_mod_16_l1235_123549


namespace NUMINAMATH_CALUDE_alex_age_difference_l1235_123555

/-- Proves the number of years ago when Alex was one-third as old as his father -/
theorem alex_age_difference (alex_current_age : ℝ) (alex_father_age : ℝ) (years_ago : ℝ) : 
  alex_current_age = 16.9996700066 →
  alex_father_age = 2 * alex_current_age + 5 →
  alex_current_age - years_ago = (1 / 3) * (alex_father_age - years_ago) →
  years_ago = 6.4998350033 := by
sorry

end NUMINAMATH_CALUDE_alex_age_difference_l1235_123555


namespace NUMINAMATH_CALUDE_oil_leak_during_work_l1235_123593

/-- The amount of oil leaked while engineers were working, given the total amount leaked and the amount leaked before they started. -/
theorem oil_leak_during_work (total_leak : ℕ) (pre_work_leak : ℕ) 
  (h1 : total_leak = 11687)
  (h2 : pre_work_leak = 6522) :
  total_leak - pre_work_leak = 5165 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_during_work_l1235_123593


namespace NUMINAMATH_CALUDE_a_minus_c_values_l1235_123512

theorem a_minus_c_values (a b c : ℕ) 
  (h1 : a > b) 
  (h2 : a^2 - a*b - a*c + b*c = 7) : 
  a - c = 1 ∨ a - c = 7 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_values_l1235_123512


namespace NUMINAMATH_CALUDE_infinite_primes_l1235_123584

theorem infinite_primes : ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p) → 
  ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_l1235_123584


namespace NUMINAMATH_CALUDE_aladdin_travel_l1235_123557

/-- A continuous function that takes all values in [0,1) -/
def equator_travel (φ : ℝ → ℝ) : Prop :=
  Continuous φ ∧ ∀ y : ℝ, 0 ≤ y ∧ y < 1 → ∃ t : ℝ, φ t = y

/-- The maximum difference between any two values of φ is at least 1 -/
theorem aladdin_travel (φ : ℝ → ℝ) (h : equator_travel φ) :
  ∃ t₁ t₂ : ℝ, |φ t₁ - φ t₂| ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_aladdin_travel_l1235_123557


namespace NUMINAMATH_CALUDE_roger_lawn_mowing_earnings_l1235_123558

/-- Roger's lawn mowing earnings problem -/
theorem roger_lawn_mowing_earnings : 
  ∀ (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ),
    rate = 9 →
    total_lawns = 14 →
    forgotten_lawns = 8 →
    rate * (total_lawns - forgotten_lawns) = 54 := by
  sorry

end NUMINAMATH_CALUDE_roger_lawn_mowing_earnings_l1235_123558


namespace NUMINAMATH_CALUDE_restaurant_bill_division_l1235_123566

theorem restaurant_bill_division (total_bill : ℕ) (individual_payment : ℕ) (num_friends : ℕ) :
  total_bill = 135 →
  individual_payment = 45 →
  total_bill = individual_payment * num_friends →
  num_friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_division_l1235_123566


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l1235_123521

theorem abs_inequality_solution_set (x : ℝ) :
  |x + 1| - |x - 2| > 1 ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l1235_123521


namespace NUMINAMATH_CALUDE_penguin_colony_fish_consumption_l1235_123577

theorem penguin_colony_fish_consumption (initial_size : ℕ) : 
  (2 * (2 * initial_size) + 129 = 1077) → 
  (initial_size = 158) := by
  sorry

end NUMINAMATH_CALUDE_penguin_colony_fish_consumption_l1235_123577


namespace NUMINAMATH_CALUDE_inequality_proof_l1235_123511

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ (a + b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1235_123511


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l1235_123544

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p ^ k * (1 - p) ^ (n - k)

/-- The probability of exactly 5 successes in 7 independent trials with 3/4 probability of success in each trial is 5103/16384. -/
theorem chocolate_milk_probability :
  binomial_probability 7 5 (3/4) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l1235_123544


namespace NUMINAMATH_CALUDE_interior_nodes_theorem_l1235_123565

/-- A point with integer coordinates -/
structure Node where
  x : ℤ
  y : ℤ

/-- A triangle with vertices at nodes -/
structure Triangle where
  a : Node
  b : Node
  c : Node

/-- Checks if a node is inside a triangle -/
def Node.isInside (n : Node) (t : Triangle) : Prop := sorry

/-- Checks if a line through two nodes contains a vertex of the triangle -/
def Line.containsVertex (p q : Node) (t : Triangle) : Prop := sorry

/-- Checks if a line through two nodes is parallel to a side of the triangle -/
def Line.isParallelToSide (p q : Node) (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem interior_nodes_theorem (t : Triangle) 
  (h : ∃ (p q : Node), p.isInside t ∧ q.isInside t ∧ p ≠ q) :
  ∃ (x y : Node), 
    x.isInside t ∧ 
    y.isInside t ∧ 
    x ≠ y ∧
    (Line.containsVertex x y t ∨ Line.isParallelToSide x y t) := by
  sorry

end NUMINAMATH_CALUDE_interior_nodes_theorem_l1235_123565


namespace NUMINAMATH_CALUDE_problem_solution_l1235_123582

theorem problem_solution (x : ℝ) :
  x + Real.sqrt (x^2 + 2) + 1 / (x - Real.sqrt (x^2 + 2)) = 15 →
  x^2 + Real.sqrt (x^4 + 2) + 1 / (x^2 + Real.sqrt (x^4 + 2)) = 47089 / 1800 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1235_123582


namespace NUMINAMATH_CALUDE_income_savings_percentage_l1235_123594

theorem income_savings_percentage (I S : ℝ) 
  (h1 : S > 0) 
  (h2 : I > S) 
  (h3 : (I - S) + (1.35 * I - 2 * S) = 2 * (I - S)) : 
  S / I = 0.35 := by
sorry

end NUMINAMATH_CALUDE_income_savings_percentage_l1235_123594


namespace NUMINAMATH_CALUDE_rectangle_area_2_by_3_l1235_123507

/-- A rectangle with width and length in centimeters -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of a rectangle in square centimeters -/
def area (r : Rectangle) : ℝ := r.width * r.length

/-- Theorem: The area of a rectangle with width 2 cm and length 3 cm is 6 cm² -/
theorem rectangle_area_2_by_3 : 
  let r : Rectangle := { width := 2, length := 3 }
  area r = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_2_by_3_l1235_123507


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1235_123514

theorem complex_fraction_simplification (z : ℂ) (h : z = -1 + I) :
  (z + 2) / (z^2 + z) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1235_123514


namespace NUMINAMATH_CALUDE_intersection_points_10_5_l1235_123500

/-- The number of intersection points formed by line segments connecting points on x and y axes -/
def intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating that 10 points on x-axis and 5 points on y-axis result in 450 intersection points -/
theorem intersection_points_10_5 :
  intersection_points 10 5 = 450 := by sorry

end NUMINAMATH_CALUDE_intersection_points_10_5_l1235_123500


namespace NUMINAMATH_CALUDE_video_game_spending_is_correct_l1235_123556

def total_allowance : ℚ := 50

def movie_fraction : ℚ := 1/4
def burger_fraction : ℚ := 1/5
def ice_cream_fraction : ℚ := 1/10
def music_fraction : ℚ := 2/5

def video_game_spending : ℚ := total_allowance - (movie_fraction * total_allowance + burger_fraction * total_allowance + ice_cream_fraction * total_allowance + music_fraction * total_allowance)

theorem video_game_spending_is_correct : video_game_spending = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_video_game_spending_is_correct_l1235_123556


namespace NUMINAMATH_CALUDE_polynomial_constant_term_l1235_123551

def g (p q r s : ℤ) (x : ℝ) : ℝ := x^4 + p*x^3 + q*x^2 + r*x + s

theorem polynomial_constant_term 
  (p q r s : ℤ) 
  (h1 : p + q + r + s = 168)
  (h2 : ∀ x : ℝ, g p q r s x = 0 → (∃ n : ℤ, x = -n ∧ n > 0))
  (h3 : ∀ x : ℝ, (g p q r s x = 0) → (g p q r s (-x) = 0)) :
  s = 144 := by sorry

end NUMINAMATH_CALUDE_polynomial_constant_term_l1235_123551


namespace NUMINAMATH_CALUDE_zoo_animal_count_l1235_123579

/-- Represents the zoo layout and animal counts -/
structure Zoo where
  tigerEnclosures : Nat
  zebraEnclosuresPerTiger : Nat
  giraffeEnclosureMultiplier : Nat
  tigersPerEnclosure : Nat
  zebrasPerEnclosure : Nat
  giraffesPerEnclosure : Nat

/-- Calculates the total number of animals in the zoo -/
def totalAnimals (zoo : Zoo) : Nat :=
  let zebraEnclosures := zoo.tigerEnclosures * zoo.zebraEnclosuresPerTiger
  let giraffeEnclosures := zebraEnclosures * zoo.giraffeEnclosureMultiplier
  let tigers := zoo.tigerEnclosures * zoo.tigersPerEnclosure
  let zebras := zebraEnclosures * zoo.zebrasPerEnclosure
  let giraffes := giraffeEnclosures * zoo.giraffesPerEnclosure
  tigers + zebras + giraffes

/-- Theorem stating that the total number of animals in the zoo is 144 -/
theorem zoo_animal_count :
  ∀ (zoo : Zoo),
    zoo.tigerEnclosures = 4 →
    zoo.zebraEnclosuresPerTiger = 2 →
    zoo.giraffeEnclosureMultiplier = 3 →
    zoo.tigersPerEnclosure = 4 →
    zoo.zebrasPerEnclosure = 10 →
    zoo.giraffesPerEnclosure = 2 →
    totalAnimals zoo = 144 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l1235_123579


namespace NUMINAMATH_CALUDE_toms_initial_books_l1235_123542

/-- Given that Tom sold 4 books, bought 38 new books, and now has 39 books,
    prove that he initially had 5 books. -/
theorem toms_initial_books :
  ∀ (initial_books : ℕ),
    initial_books - 4 + 38 = 39 →
    initial_books = 5 := by
  sorry

end NUMINAMATH_CALUDE_toms_initial_books_l1235_123542


namespace NUMINAMATH_CALUDE_equidistant_point_existence_l1235_123578

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by y = mx + b -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Distance between a point and a circle -/
def distanceToCircle (p : Point) (c : Circle) : ℝ :=
  sorry

/-- Distance between a point and a line -/
def distanceToLine (p : Point) (l : Line) : ℝ :=
  sorry

/-- The main theorem -/
theorem equidistant_point_existence (c : Circle) (upper_tangent lower_tangent : Line) :
  c.radius = 5 →
  distanceToLine (0, c.center.2 + c.radius) upper_tangent = 3 →
  distanceToLine (0, c.center.2 - c.radius) lower_tangent = 7 →
  ∃! p : Point, 
    distanceToCircle p c = distanceToLine p upper_tangent ∧ 
    distanceToCircle p c = distanceToLine p lower_tangent :=
  sorry

end NUMINAMATH_CALUDE_equidistant_point_existence_l1235_123578


namespace NUMINAMATH_CALUDE_helens_oranges_l1235_123527

/-- Helen's orange counting problem -/
theorem helens_oranges (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 9 → received = 29 → total = initial + received → total = 38 := by sorry

end NUMINAMATH_CALUDE_helens_oranges_l1235_123527


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l1235_123530

/-- A hyperbola with equation x²/(a-3) + y²/(2-a) = 1, foci on the y-axis, and focal distance 4 -/
structure Hyperbola where
  a : ℝ
  equation : ∀ x y : ℝ, x^2 / (a - 3) + y^2 / (2 - a) = 1
  foci_on_y_axis : True  -- This is a placeholder for the foci condition
  focal_distance : ℝ
  focal_distance_value : focal_distance = 4

/-- The value of 'a' for the given hyperbola is 1/2 -/
theorem hyperbola_a_value (h : Hyperbola) : h.a = 1/2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l1235_123530


namespace NUMINAMATH_CALUDE_flea_treatment_effectiveness_l1235_123572

theorem flea_treatment_effectiveness (F : ℕ) : 
  (F : ℝ) * 0.4 * 0.55 * 0.7 * 0.8 = 20 → F - 20 = 142 := by
  sorry

end NUMINAMATH_CALUDE_flea_treatment_effectiveness_l1235_123572


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l1235_123516

-- Define an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_seq_sum (a : ℕ → ℝ) :
  is_arithmetic_seq a → a 2 = 5 → a 6 = 33 → a 3 + a 5 = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l1235_123516
