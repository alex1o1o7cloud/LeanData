import Mathlib

namespace NUMINAMATH_CALUDE_age_difference_l3439_343961

theorem age_difference (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (10 * a + b + 5) = 3 * (10 * b + a + 5)) :
  (10 * a + b) - (10 * b + a) = 45 :=
sorry

end NUMINAMATH_CALUDE_age_difference_l3439_343961


namespace NUMINAMATH_CALUDE_fermat_number_units_digit_F5_l3439_343901

theorem fermat_number_units_digit_F5 :
  (2^(2^5) + 1) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_units_digit_F5_l3439_343901


namespace NUMINAMATH_CALUDE_news_spread_time_correct_total_time_l3439_343924

/-- The number of people in the city -/
def city_population : ℕ := 3000000

/-- The time interval in minutes for each round of information spreading -/
def time_interval : ℕ := 10

/-- The number of new people informed by each person in one interval -/
def spread_rate : ℕ := 2

/-- The total number of people who know the news after k intervals -/
def people_informed (k : ℕ) : ℕ := 2^(k+1) - 1

/-- The minimum number of intervals needed to inform the entire city -/
def min_intervals : ℕ := 21

theorem news_spread_time :
  (people_informed min_intervals ≥ city_population) ∧
  (∀ k < min_intervals, people_informed k < city_population) :=
sorry

/-- The total time needed to inform the entire city in minutes -/
def total_time : ℕ := min_intervals * time_interval

theorem correct_total_time : total_time = 210 :=
sorry

end NUMINAMATH_CALUDE_news_spread_time_correct_total_time_l3439_343924


namespace NUMINAMATH_CALUDE_blueberry_pies_l3439_343953

theorem blueberry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) :
  total_pies = 30 →
  apple_ratio = 2 →
  blueberry_ratio = 3 →
  cherry_ratio = 5 →
  blueberry_ratio * (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_blueberry_pies_l3439_343953


namespace NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_seq_l3439_343925

theorem binomial_coeff_not_arithmetic_seq (n r : ℕ) (h1 : n ≥ r + 3) (h2 : r > 0) :
  ¬ (∃ d : ℚ, Nat.choose n r + d = Nat.choose n (r + 1) ∧
               Nat.choose n (r + 1) + d = Nat.choose n (r + 2) ∧
               Nat.choose n (r + 2) + d = Nat.choose n (r + 3)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_seq_l3439_343925


namespace NUMINAMATH_CALUDE_max_n_for_consecutive_product_l3439_343903

theorem max_n_for_consecutive_product (n : ℕ) : 
  (∃ k : ℕ, 9*n^2 + 5*n + 26 = k * (k + 1)) → n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_n_for_consecutive_product_l3439_343903


namespace NUMINAMATH_CALUDE_abs_value_inequality_iff_l3439_343956

theorem abs_value_inequality_iff (a b : ℝ) : a * |a| > b * |b| ↔ a > b := by sorry

end NUMINAMATH_CALUDE_abs_value_inequality_iff_l3439_343956


namespace NUMINAMATH_CALUDE_student_A_selection_probability_l3439_343954

/-- The number of students -/
def n : ℕ := 5

/-- The number of students to be selected -/
def k : ℕ := 2

/-- The probability of selecting student A -/
def prob_A : ℚ := 2/5

/-- The combination function -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem student_A_selection_probability :
  (combination (n - 1) (k - 1) : ℚ) / (combination n k : ℚ) = prob_A :=
sorry

end NUMINAMATH_CALUDE_student_A_selection_probability_l3439_343954


namespace NUMINAMATH_CALUDE_quadrilateral_area_product_not_1988_l3439_343910

/-- Represents a convex quadrilateral divided by its diagonals into four triangles -/
structure QuadrilateralWithDiagonals where
  S₁ : ℕ  -- Area of triangle AOB
  S₂ : ℕ  -- Area of triangle BOC
  S₃ : ℕ  -- Area of triangle COD
  S₄ : ℕ  -- Area of triangle DOA

/-- The product of the areas of the four triangles in a quadrilateral divided by its diagonals
    cannot end in 1988 -/
theorem quadrilateral_area_product_not_1988 (q : QuadrilateralWithDiagonals) :
  ∀ (n : ℕ), q.S₁ * q.S₂ * q.S₃ * q.S₄ ≠ 1988 + 10000 * n := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_product_not_1988_l3439_343910


namespace NUMINAMATH_CALUDE_pedro_plums_problem_l3439_343957

theorem pedro_plums_problem (total_fruits : ℕ) (total_cost : ℕ) 
  (plum_cost peach_cost : ℕ) (h1 : total_fruits = 32) 
  (h2 : total_cost = 52) (h3 : plum_cost = 2) (h4 : peach_cost = 1) :
  ∃ (plums peaches : ℕ), 
    plums + peaches = total_fruits ∧
    plum_cost * plums + peach_cost * peaches = total_cost ∧
    plums = 20 := by
  sorry

end NUMINAMATH_CALUDE_pedro_plums_problem_l3439_343957


namespace NUMINAMATH_CALUDE_f_increasing_implies_f_one_geq_25_l3439_343974

/-- A function f that is quadratic with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- Theorem stating that if f is increasing on [-2, +∞), then f(1) ≥ 25 -/
theorem f_increasing_implies_f_one_geq_25 (m : ℝ) 
  (h : ∀ x y, -2 ≤ x ∧ x < y → f m x < f m y) : 
  f m 1 ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_implies_f_one_geq_25_l3439_343974


namespace NUMINAMATH_CALUDE_files_remaining_l3439_343981

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 26)
  (h2 : video_files = 36)
  (h3 : deleted_files = 48) :
  music_files + video_files - deleted_files = 14 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l3439_343981


namespace NUMINAMATH_CALUDE_baking_dish_recipe_book_ratio_l3439_343963

/-- The cost of Liz's purchases -/
def total_cost : ℚ := 40

/-- The cost of the recipe book -/
def recipe_book_cost : ℚ := 6

/-- The cost of each ingredient -/
def ingredient_cost : ℚ := 3

/-- The number of ingredients purchased -/
def num_ingredients : ℕ := 5

/-- The additional cost of the apron compared to the recipe book -/
def apron_extra_cost : ℚ := 1

/-- The ratio of the baking dish cost to the recipe book cost -/
def baking_dish_to_recipe_book_ratio : ℚ := 2

theorem baking_dish_recipe_book_ratio :
  (total_cost - (recipe_book_cost + (recipe_book_cost + apron_extra_cost) + 
   (ingredient_cost * num_ingredients))) / recipe_book_cost = baking_dish_to_recipe_book_ratio := by
  sorry

end NUMINAMATH_CALUDE_baking_dish_recipe_book_ratio_l3439_343963


namespace NUMINAMATH_CALUDE_probability_of_third_six_l3439_343999

theorem probability_of_third_six (p_fair : ℝ) (p_biased : ℝ) (p_other : ℝ) : 
  p_fair = 1/6 →
  p_biased = 2/3 →
  p_other = 1/15 →
  (1/6^2 / (1/6^2 + (2/3)^2)) * (1/6) + ((2/3)^2 / (1/6^2 + (2/3)^2)) * (2/3) = 65/102 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_third_six_l3439_343999


namespace NUMINAMATH_CALUDE_fred_dime_count_l3439_343935

def final_dime_count (initial : ℕ) (borrowed : ℕ) (returned : ℕ) (given : ℕ) : ℕ :=
  initial - borrowed + returned + given

theorem fred_dime_count : final_dime_count 12 4 2 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fred_dime_count_l3439_343935


namespace NUMINAMATH_CALUDE_cake_division_l3439_343966

theorem cake_division (num_cakes : ℕ) (num_children : ℕ) (max_cuts : ℕ) :
  num_cakes = 9 →
  num_children = 4 →
  max_cuts = 2 →
  ∃ (whole_cakes : ℕ) (fractional_cake : ℚ),
    whole_cakes + fractional_cake = num_cakes / num_children ∧
    whole_cakes = 2 ∧
    fractional_cake = 1/4 ∧
    (∀ cake, cake ≤ max_cuts) :=
by sorry

end NUMINAMATH_CALUDE_cake_division_l3439_343966


namespace NUMINAMATH_CALUDE_exactly_two_true_l3439_343992

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the four propositions
def Prop1 (f : ℝ → ℝ) : Prop := IsOdd f → f 0 = 0
def Prop2 (f : ℝ → ℝ) : Prop := f 0 = 0 → IsOdd f
def Prop3 (f : ℝ → ℝ) : Prop := ¬(IsOdd f) → f 0 ≠ 0
def Prop4 (f : ℝ → ℝ) : Prop := f 0 ≠ 0 → ¬(IsOdd f)

-- The main theorem
theorem exactly_two_true (f : ℝ → ℝ) : 
  IsOdd f → (Prop1 f ∧ Prop4 f ∧ ¬Prop2 f ∧ ¬Prop3 f) := by sorry

end NUMINAMATH_CALUDE_exactly_two_true_l3439_343992


namespace NUMINAMATH_CALUDE_original_student_count_l3439_343975

/-- Prove that given the initial average weight, new student's weight, and new average weight,
    the number of original students is 29. -/
theorem original_student_count
  (initial_avg : ℝ)
  (new_student_weight : ℝ)
  (new_avg : ℝ)
  (h1 : initial_avg = 28)
  (h2 : new_student_weight = 22)
  (h3 : new_avg = 27.8)
  : ∃ n : ℕ, n = 29 ∧ 
    (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * new_avg :=
by
  sorry

end NUMINAMATH_CALUDE_original_student_count_l3439_343975


namespace NUMINAMATH_CALUDE_smallest_p_satisfying_gcd_conditions_l3439_343934

theorem smallest_p_satisfying_gcd_conditions : 
  ∃ (p : ℕ), 
    p > 1500 ∧ 
    Nat.gcd 90 (p + 150) = 30 ∧ 
    Nat.gcd (p + 90) 150 = 75 ∧ 
    (∀ (q : ℕ), q > 1500 → Nat.gcd 90 (q + 150) = 30 → Nat.gcd (q + 90) 150 = 75 → p ≤ q) ∧
    p = 1560 :=
by sorry

end NUMINAMATH_CALUDE_smallest_p_satisfying_gcd_conditions_l3439_343934


namespace NUMINAMATH_CALUDE_largest_d_for_negative_three_in_range_l3439_343985

/-- The function f(x) = x^2 + 4x + d -/
def f (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- Proposition: The largest value of d such that -3 is in the range of f(x) = x^2 + 4x + d is 1 -/
theorem largest_d_for_negative_three_in_range :
  (∃ (d : ℝ), ∀ (e : ℝ), (∃ (x : ℝ), f d x = -3) → e ≤ d) ∧
  (∃ (x : ℝ), f 1 x = -3) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_negative_three_in_range_l3439_343985


namespace NUMINAMATH_CALUDE_factor_implies_k_equals_8_l3439_343930

theorem factor_implies_k_equals_8 (m k : ℝ) : 
  (∃ q : ℝ, m^3 - k*m^2 - 24*m + 16 = (m^2 - 8*m) * q) → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_k_equals_8_l3439_343930


namespace NUMINAMATH_CALUDE_chip_defect_rate_line_A_l3439_343938

theorem chip_defect_rate_line_A :
  let total_chips : ℕ := 20
  let chips_line_A : ℕ := 12
  let chips_line_B : ℕ := 8
  let defect_rate_B : ℚ := 1 / 20
  let overall_defect_rate : ℚ := 8 / 100
  let defect_rate_A : ℚ := 1 / 10
  (chips_line_A : ℚ) * defect_rate_A + (chips_line_B : ℚ) * defect_rate_B = (total_chips : ℚ) * overall_defect_rate :=
by sorry

end NUMINAMATH_CALUDE_chip_defect_rate_line_A_l3439_343938


namespace NUMINAMATH_CALUDE_tens_digit_of_2035_pow_2037_minus_2039_l3439_343993

theorem tens_digit_of_2035_pow_2037_minus_2039 : ∃ n : ℕ, n < 10 ∧ n * 10 + 3 = (2035^2037 - 2039) % 100 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2035_pow_2037_minus_2039_l3439_343993


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l3439_343971

theorem set_inclusion_implies_a_range (a : ℝ) : 
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ a}
  let B := {y : ℝ | ∃ x ∈ A, y = 2*x + 3}
  let C := {z : ℝ | ∃ x ∈ A, z = x^2}
  C ⊆ B → (1/2 : ℝ) ≤ a ∧ a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l3439_343971


namespace NUMINAMATH_CALUDE_amount_after_two_years_l3439_343908

/-- The amount after n years with a given initial value and annual increase rate -/
def amountAfterYears (initialValue : ℝ) (increaseRate : ℝ) (years : ℕ) : ℝ :=
  initialValue * (1 + increaseRate) ^ years

/-- Theorem: Given an initial value of 3200 and an annual increase rate of 1/8,
    the value after two years will be 4050 -/
theorem amount_after_two_years :
  amountAfterYears 3200 (1/8) 2 = 4050 := by
  sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l3439_343908


namespace NUMINAMATH_CALUDE_no_ratio_p_squared_l3439_343923

theorem no_ratio_p_squared (p : ℕ) (hp : Prime p) :
  ∀ (x y l : ℕ), l ≥ 1 →
    (x * (x + 1)) / (y * (y + 1)) ≠ p^(2 * l) :=
by sorry

end NUMINAMATH_CALUDE_no_ratio_p_squared_l3439_343923


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3439_343977

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^4 - 2*x^3 + 3*x + 1) % (x - 2) = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3439_343977


namespace NUMINAMATH_CALUDE_gcd_459_357_l3439_343955

theorem gcd_459_357 : Int.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l3439_343955


namespace NUMINAMATH_CALUDE_sum_of_fractions_in_different_bases_l3439_343951

/-- Converts a number from a given base to base 10 --/
def toBase10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem sum_of_fractions_in_different_bases : 
  let a := toBase10 [2, 5, 4] 8
  let b := toBase10 [1, 2] 4
  let c := toBase10 [1, 3, 2] 5
  let d := toBase10 [2, 3] 3
  roundToNearest ((a / b : ℚ) + (c / d : ℚ)) = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_in_different_bases_l3439_343951


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l3439_343943

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability of a normal random variable being less than a given value -/
noncomputable def prob_less_than (X : NormalRV) (x : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry 
  (X : NormalRV) 
  (h : X.μ = 2) 
  (h2 : prob_less_than X 4 = 0.8) : 
  prob_less_than X 0 = 0.2 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l3439_343943


namespace NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l3439_343995

theorem min_value_arithmetic_sequence (a : ℝ) (m : ℕ+) :
  (∃ (S : ℕ+ → ℝ), S m = 36 ∧ 
    (∀ n : ℕ+, S n = n * a - 4 * (n * (n - 1)) / 2)) →
  ∀ a' : ℝ, (∃ m' : ℕ+, ∃ S' : ℕ+ → ℝ, 
    S' m' = 36 ∧ 
    (∀ n : ℕ+, S' n = n * a' - 4 * (n * (n - 1)) / 2)) →
  a' ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l3439_343995


namespace NUMINAMATH_CALUDE_f_properties_l3439_343939

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x

theorem f_properties (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1)))
  (h2 : ∀ x, f (x + 4) = f (-x)) :
  is_even f ∧ f 3 = 0 ∧ f 2023 = 0 := by sorry

end NUMINAMATH_CALUDE_f_properties_l3439_343939


namespace NUMINAMATH_CALUDE_complex_fraction_fourth_quadrant_l3439_343998

/-- Given that (1+i)/(2-i) = a + (b+1)i where a and b are real numbers and i is the imaginary unit,
    prove that the point corresponding to z = a + bi lies in the fourth quadrant of the complex plane. -/
theorem complex_fraction_fourth_quadrant (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (1 + i) / (2 - i) = a + (b + 1) * i →
  0 < a ∧ b < 0 :=
sorry

end NUMINAMATH_CALUDE_complex_fraction_fourth_quadrant_l3439_343998


namespace NUMINAMATH_CALUDE_chemistry_class_average_l3439_343978

theorem chemistry_class_average (n₁ n₂ n₃ n₄ : ℕ) (m₁ m₂ m₃ m₄ : ℚ) :
  let total_students := n₁ + n₂ + n₃ + n₄
  let total_marks := n₁ * m₁ + n₂ * m₂ + n₃ * m₃ + n₄ * m₄
  total_marks / total_students = (n₁ * m₁ + n₂ * m₂ + n₃ * m₃ + n₄ * m₄) / (n₁ + n₂ + n₃ + n₄) :=
by
  sorry

#eval (60 * 50 + 35 * 60 + 45 * 55 + 42 * 45) / (60 + 35 + 45 + 42)

end NUMINAMATH_CALUDE_chemistry_class_average_l3439_343978


namespace NUMINAMATH_CALUDE_fine_payment_l3439_343964

theorem fine_payment (F : ℚ) 
  (hF : F > 0)
  (hJoe : F / 4 + 3 + F / 3 - 3 + F / 2 - 4 = F) : 
  F / 2 - 4 = 5 * F / 12 := by
  sorry

end NUMINAMATH_CALUDE_fine_payment_l3439_343964


namespace NUMINAMATH_CALUDE_is_root_of_polynomial_l3439_343931

theorem is_root_of_polynomial (x : ℝ) : 
  x = 4 → x^3 - 5*x^2 + 7*x - 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_is_root_of_polynomial_l3439_343931


namespace NUMINAMATH_CALUDE_nicki_total_miles_run_l3439_343914

/-- Calculates the total miles run in a year given weekly mileage for each half -/
def total_miles_run (weeks_in_year : ℕ) (miles_first_half : ℕ) (miles_second_half : ℕ) : ℕ :=
  let half_year := weeks_in_year / 2
  (miles_first_half * half_year) + (miles_second_half * half_year)

theorem nicki_total_miles_run : total_miles_run 52 20 30 = 1300 := by
  sorry

#eval total_miles_run 52 20 30

end NUMINAMATH_CALUDE_nicki_total_miles_run_l3439_343914


namespace NUMINAMATH_CALUDE_reciprocal_sum_quarters_fifths_l3439_343900

theorem reciprocal_sum_quarters_fifths : (1 / (1 / 4 + 1 / 5) : ℚ) = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_quarters_fifths_l3439_343900


namespace NUMINAMATH_CALUDE_red_shirt_percentage_l3439_343994

theorem red_shirt_percentage (total_students : ℕ) (blue_percent : ℚ) (green_percent : ℚ) (other_colors : ℕ) 
  (h1 : total_students = 900)
  (h2 : blue_percent = 44 / 100)
  (h3 : green_percent = 10 / 100)
  (h4 : other_colors = 162) :
  (total_students - (blue_percent * total_students + green_percent * total_students + other_colors)) / total_students = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_red_shirt_percentage_l3439_343994


namespace NUMINAMATH_CALUDE_common_term_implies_fermat_number_l3439_343962

/-- Definition of the second-order arithmetic sequence -/
def a (n : ℕ) (k : ℕ) : ℕ :=
  (k - 2) * n * (n - 1) / 2 + n

/-- Definition of Fermat numbers -/
def fermat (m : ℕ) : ℕ :=
  2^(2^m) + 1

/-- Theorem stating that if k satisfies the condition, it must be a Fermat number -/
theorem common_term_implies_fermat_number (k : ℕ) (h1 : k > 2) :
  (∃ n m : ℕ, a n k = fermat m) → (∃ m : ℕ, k = fermat m) :=
sorry

end NUMINAMATH_CALUDE_common_term_implies_fermat_number_l3439_343962


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3439_343940

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 8) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3439_343940


namespace NUMINAMATH_CALUDE_lunch_break_duration_l3439_343980

/-- Represents the painting rate of an individual or group in terms of house percentage per hour -/
structure PaintingRate where
  rate : ℝ
  (nonneg : rate ≥ 0)

/-- Represents the duration of work in hours -/
def workDuration (startTime endTime : ℝ) : ℝ := endTime - startTime

/-- Represents the percentage of house painted given a painting rate and work duration -/
def percentPainted (r : PaintingRate) (duration : ℝ) : ℝ := r.rate * duration

theorem lunch_break_duration (paula : PaintingRate) (helpers : PaintingRate) 
  (lunchBreak : ℝ) : 
  -- Monday's condition
  percentPainted (PaintingRate.mk (paula.rate + helpers.rate) (by sorry)) (workDuration 8 16 - lunchBreak) = 0.5 →
  -- Tuesday's condition
  percentPainted helpers (workDuration 8 14.2 - lunchBreak) = 0.24 →
  -- Wednesday's condition
  percentPainted paula (workDuration 8 19.2 - lunchBreak) = 0.26 →
  -- Conclusion
  lunchBreak * 60 = 48 := by sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l3439_343980


namespace NUMINAMATH_CALUDE_system_solution_l3439_343941

theorem system_solution (x y m : ℚ) : 
  (2 * x + 3 * y = 4) → 
  (3 * x + 2 * y = 2 * m - 3) → 
  (x + y = -3/5) → 
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3439_343941


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3439_343912

/-- A regular polygon with perimeter 180 cm and side length 15 cm has 12 sides -/
theorem regular_polygon_sides (P : ℝ) (s : ℝ) (n : ℕ) 
  (h_perimeter : P = 180) 
  (h_side : s = 15) 
  (h_regular : P = n * s) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3439_343912


namespace NUMINAMATH_CALUDE_tire_price_proof_l3439_343989

/-- The regular price of a single tire -/
def regular_price : ℚ := 295 / 3

/-- The price of the fourth tire under the offer -/
def fourth_tire_price : ℚ := 5

/-- The total discount applied to the purchase -/
def total_discount : ℚ := 10

/-- The total amount Jane paid for four tires -/
def total_paid : ℚ := 290

/-- Theorem stating that the regular price of a tire is 295/3 given the sale conditions -/
theorem tire_price_proof :
  3 * regular_price + fourth_tire_price - total_discount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_tire_price_proof_l3439_343989


namespace NUMINAMATH_CALUDE_unique_number_with_special_properties_l3439_343949

/-- Returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Returns the product of digits of a natural number -/
def prod_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop := sorry

theorem unique_number_with_special_properties : 
  ∃! x : ℕ, 
    prod_of_digits x = 44 * x - 86868 ∧ 
    is_perfect_cube (sum_of_digits x) ∧
    x = 1989 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_special_properties_l3439_343949


namespace NUMINAMATH_CALUDE_tina_total_time_l3439_343960

/-- The time it takes to clean one key, in minutes -/
def time_per_key : ℕ := 3

/-- The number of keys left to clean -/
def keys_to_clean : ℕ := 14

/-- The time it takes to complete the assignment, in minutes -/
def assignment_time : ℕ := 10

/-- The total time it takes for Tina to clean the remaining keys and finish her assignment -/
def total_time : ℕ := time_per_key * keys_to_clean + assignment_time

theorem tina_total_time : total_time = 52 := by
  sorry

end NUMINAMATH_CALUDE_tina_total_time_l3439_343960


namespace NUMINAMATH_CALUDE_square_sum_minus_triple_product_l3439_343906

theorem square_sum_minus_triple_product (x y : ℝ) 
  (h1 : x * y = 3) 
  (h2 : x + y = 4) : 
  x^2 + y^2 - 3*x*y = 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_minus_triple_product_l3439_343906


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l3439_343904

theorem solve_equation_and_evaluate (x : ℝ) : 
  (5 * x - 7 = 15 * x + 21) → 3 * (x + 10) = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l3439_343904


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_l3439_343969

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) :
  perpPlane m α → perpPlane n β → perp m n → perpPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_l3439_343969


namespace NUMINAMATH_CALUDE_cubic_function_property_l3439_343915

theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x - 4
  f (-2) = 2 → f 2 = -10 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3439_343915


namespace NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l3439_343965

theorem largest_value_when_x_is_quarter (x : ℝ) (h : x = 1/4) :
  (1/x > x) ∧ (1/x > x^2) ∧ (1/x > (1/2)*x) ∧ (1/x > Real.sqrt x) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l3439_343965


namespace NUMINAMATH_CALUDE_mean_temperature_l3439_343984

def temperatures : List ℝ := [-3.5, -2.25, 0, 3.75, 4.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l3439_343984


namespace NUMINAMATH_CALUDE_chord_length_line_circle_intersection_specific_chord_length_l3439_343920

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection 
  (line : ℝ → ℝ → Prop) 
  (circle : ℝ → ℝ → Prop) : ℝ :=
by
  sorry

/-- Main theorem: The length of the chord formed by the intersection of 
    x + √3 y - 2 = 0 and x² + y² = 4 is 2√3 -/
theorem specific_chord_length : 
  chord_length_line_circle_intersection 
    (fun x y => x + Real.sqrt 3 * y - 2 = 0) 
    (fun x y => x^2 + y^2 = 4) = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_chord_length_line_circle_intersection_specific_chord_length_l3439_343920


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3439_343946

/-- Prove that the given expression equals 1/15 -/
theorem problem_1 : 
  (2 * (Nat.factorial 8 / Nat.factorial 3) + 7 * (Nat.factorial 8 / Nat.factorial 4)) / 
  (Nat.factorial 8 - Nat.factorial 9 / Nat.factorial 4) = 1 / 15 := by
  sorry

/-- Prove that the sum of combinations equals C(202, 4) -/
theorem problem_2 : 
  Nat.choose 200 198 + Nat.choose 200 196 + 2 * Nat.choose 200 197 = Nat.choose 202 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3439_343946


namespace NUMINAMATH_CALUDE_factorial_problem_l3439_343991

-- Define the factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_problem : (factorial 13 - factorial 12) / factorial 11 = 144 := by
  sorry

end NUMINAMATH_CALUDE_factorial_problem_l3439_343991


namespace NUMINAMATH_CALUDE_homework_completion_l3439_343913

/-- The fraction of homework Sanjay completed on Monday -/
def sanjay_monday : ℚ := 3/5

/-- The fraction of homework Deepak completed on Monday -/
def deepak_monday : ℚ := 2/7

/-- The fraction of remaining homework Sanjay completed on Tuesday -/
def sanjay_tuesday : ℚ := 1/3

/-- The fraction of remaining homework Deepak completed on Tuesday -/
def deepak_tuesday : ℚ := 3/10

/-- The combined fraction of original homework left for Sanjay and Deepak on Wednesday -/
def homework_left : ℚ := 23/30

theorem homework_completion :
  let sanjay_left := (1 - sanjay_monday) * (1 - sanjay_tuesday)
  let deepak_left := (1 - deepak_monday) * (1 - deepak_tuesday)
  sanjay_left + deepak_left = homework_left := by sorry

end NUMINAMATH_CALUDE_homework_completion_l3439_343913


namespace NUMINAMATH_CALUDE_trig_identity_l3439_343917

theorem trig_identity : 
  Real.cos (70 * π / 180) * Real.cos (335 * π / 180) + 
  Real.sin (110 * π / 180) * Real.sin (25 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3439_343917


namespace NUMINAMATH_CALUDE_downstream_speed_theorem_l3439_343958

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed given the rowing speeds in still water and upstream -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

/-- Theorem stating that given the specific conditions, the downstream speed is 31 kmph -/
theorem downstream_speed_theorem (s : RowingSpeed) 
  (h1 : s.stillWater = 28) 
  (h2 : s.upstream = 25) : 
  downstreamSpeed s = 31 := by
  sorry

#check downstream_speed_theorem

end NUMINAMATH_CALUDE_downstream_speed_theorem_l3439_343958


namespace NUMINAMATH_CALUDE_miss_at_least_once_probability_l3439_343996

/-- The probability of missing a target at least once in three shots -/
def miss_at_least_once (P : ℝ) : ℝ :=
  1 - P^3

theorem miss_at_least_once_probability (P : ℝ) 
  (h1 : 0 ≤ P) (h2 : P ≤ 1) : 
  miss_at_least_once P = 1 - P^3 := by
sorry

end NUMINAMATH_CALUDE_miss_at_least_once_probability_l3439_343996


namespace NUMINAMATH_CALUDE_fish_tank_problem_l3439_343927

theorem fish_tank_problem (x : ℕ) : x + (x - 4) = 20 → x - 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l3439_343927


namespace NUMINAMATH_CALUDE_equation_solutions_l3439_343911

theorem equation_solutions : 
  {x : ℝ | x^4 + (3-x)^4 + x^3 = 82} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3439_343911


namespace NUMINAMATH_CALUDE_large_block_volume_l3439_343968

/-- Volume of a rectangular block -/
def volume (width depth length : ℝ) : ℝ := width * depth * length

theorem large_block_volume :
  ∀ (w d l : ℝ),
  volume w d l = 4 →
  volume (2 * w) (2 * d) (2 * l) = 32 := by
  sorry

end NUMINAMATH_CALUDE_large_block_volume_l3439_343968


namespace NUMINAMATH_CALUDE_isosceles_triangles_in_right_triangle_l3439_343907

theorem isosceles_triangles_in_right_triangle :
  ∀ (a b c : ℝ) (S₁ S₂ S₃ : ℝ) (x : ℝ),
    a = 1 →
    b = Real.sqrt 3 →
    c^2 = a^2 + b^2 →
    S₁ + S₂ + S₃ = (1/2) * a * b →
    S₁ = (1/2) * (a/3) * x →
    S₂ = (1/2) * (b/3) * x →
    S₃ = (1/2) * (c/3) * x →
    x = Real.sqrt 109 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangles_in_right_triangle_l3439_343907


namespace NUMINAMATH_CALUDE_subset_implies_a_leq_4_l3439_343952

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + 4 ≥ 0}

-- State the theorem
theorem subset_implies_a_leq_4 : ∀ a : ℝ, A ⊆ B a → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_leq_4_l3439_343952


namespace NUMINAMATH_CALUDE_gaming_system_value_proof_l3439_343982

/-- The value of Tom's gaming system -/
def gaming_system_value : ℝ := 150

/-- The percentage of the gaming system's value given as store credit -/
def store_credit_percentage : ℝ := 0.80

/-- The amount Tom pays in cash -/
def cash_paid : ℝ := 80

/-- The change Tom receives -/
def change_received : ℝ := 10

/-- The value of the game Tom receives -/
def game_value : ℝ := 30

/-- The cost of the NES -/
def nes_cost : ℝ := 160

theorem gaming_system_value_proof :
  store_credit_percentage * gaming_system_value + cash_paid - change_received = nes_cost + game_value :=
by sorry

end NUMINAMATH_CALUDE_gaming_system_value_proof_l3439_343982


namespace NUMINAMATH_CALUDE_three_billion_three_hundred_million_scientific_notation_l3439_343928

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem three_billion_three_hundred_million_scientific_notation :
  to_scientific_notation 3300000000 = ScientificNotation.mk 3.3 9 sorry := by
  sorry

end NUMINAMATH_CALUDE_three_billion_three_hundred_million_scientific_notation_l3439_343928


namespace NUMINAMATH_CALUDE_megan_folders_l3439_343944

def number_of_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : ℕ :=
  (initial_files - deleted_files) / files_per_folder

theorem megan_folders :
  number_of_folders 93 21 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_megan_folders_l3439_343944


namespace NUMINAMATH_CALUDE_math_team_selection_l3439_343983

theorem math_team_selection (boys girls : ℕ) (h1 : boys = 7) (h2 : girls = 10) :
  (boys.choose 4) * (girls.choose 2) = 1575 :=
by sorry

end NUMINAMATH_CALUDE_math_team_selection_l3439_343983


namespace NUMINAMATH_CALUDE_expression_simplification_l3439_343905

theorem expression_simplification (x : ℝ) : 
  3 * x - 7 * x^2 + 5 - (6 - 5 * x + 7 * x^2) = -14 * x^2 + 8 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3439_343905


namespace NUMINAMATH_CALUDE_tv_final_price_l3439_343959

/-- Calculates the final price after applying successive discounts -/
def final_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (fun price discount => price * (1 - discount)) original_price

/-- Proves that the final price of a $450 TV after 10%, 20%, and 5% discounts is $307.80 -/
theorem tv_final_price : 
  let original_price : ℝ := 450
  let discounts : List ℝ := [0.1, 0.2, 0.05]
  final_price original_price discounts = 307.80 := by
sorry

#eval final_price 450 [0.1, 0.2, 0.05]

end NUMINAMATH_CALUDE_tv_final_price_l3439_343959


namespace NUMINAMATH_CALUDE_TI_is_euler_line_l3439_343986

-- Define the basic structures
variable (A B C I T X Y Z : ℝ × ℝ)

-- Define the properties
variable (h1 : is_incenter I A B C)
variable (h2 : is_antigonal_point T I A B C)
variable (h3 : is_antipedal_triangle X Y Z T A B C)

-- Define the Euler line
def euler_line (X Y Z : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the line TI
def line_TI (T I : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem TI_is_euler_line :
  line_TI T I = euler_line X Y Z :=
sorry

end NUMINAMATH_CALUDE_TI_is_euler_line_l3439_343986


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l3439_343970

/-- Given 4 siblings where the ages of the older siblings are 3, 6, and 7 years more than 
    the youngest, and the average age of all siblings is 30, 
    the age of the youngest sibling is 26. -/
theorem youngest_sibling_age (y : ℕ) : 
  (y + (y + 3) + (y + 6) + (y + 7)) / 4 = 30 → y = 26 := by
  sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l3439_343970


namespace NUMINAMATH_CALUDE_softball_team_ratio_l3439_343945

theorem softball_team_ratio (total_players : ℕ) (more_women : ℕ) : 
  total_players = 15 → more_women = 5 → 
  ∃ (men women : ℕ), 
    men + women = total_players ∧ 
    women = men + more_women ∧ 
    men * 2 = women := by
  sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l3439_343945


namespace NUMINAMATH_CALUDE_inequality_solution_l3439_343950

open Set

def solution_set : Set ℝ :=
  Ioo (-3 : ℝ) (-8/3) ∪ Ioo ((1 - Real.sqrt 89) / 4) ((1 + Real.sqrt 89) / 4)

theorem inequality_solution :
  {x : ℝ | (x - 2) / (x + 3) > (4 * x + 5) / (3 * x + 8) ∧ x ≠ -3 ∧ x ≠ -8/3} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3439_343950


namespace NUMINAMATH_CALUDE_pentagon_rectangle_apothem_ratio_l3439_343902

theorem pentagon_rectangle_apothem_ratio :
  let pentagon_side := (40 : ℝ) / (1 + Real.sqrt 5)
  let pentagon_apothem := pentagon_side * ((1 + Real.sqrt 5) / 4)
  let rectangle_width := (3 : ℝ) / 2
  let rectangle_apothem := rectangle_width / 2
  pentagon_apothem / rectangle_apothem = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_apothem_ratio_l3439_343902


namespace NUMINAMATH_CALUDE_min_gennadies_for_festival_l3439_343929

/-- Represents the number of people with a specific name -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat

/-- Calculates the minimum number of Gennadies required -/
def minGennadies (counts : NameCount) : Nat :=
  max 0 (counts.borises - 1 - counts.alexanders - counts.vasilies)

/-- Theorem stating the minimum number of Gennadies required for the given scenario -/
theorem min_gennadies_for_festival (counts : NameCount) 
  (h1 : counts.alexanders = 45)
  (h2 : counts.borises = 122)
  (h3 : counts.vasilies = 27) :
  minGennadies counts = 49 := by
  sorry

#eval minGennadies { alexanders := 45, borises := 122, vasilies := 27 }

end NUMINAMATH_CALUDE_min_gennadies_for_festival_l3439_343929


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_8_l3439_343947

/-- A geometric sequence with its sum of terms -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_geometric : ∀ n, a (n + 1) = a n * (a 1)⁻¹ * a 2
  sum_formula : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)

/-- The main theorem -/
theorem geometric_sequence_sum_8 (seq : GeometricSequence) 
    (h2 : seq.S 2 = 3)
    (h4 : seq.S 4 = 15) :
  seq.S 8 = 255 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_8_l3439_343947


namespace NUMINAMATH_CALUDE_percentage_of_hindu_boys_l3439_343942

theorem percentage_of_hindu_boys (total_boys : ℕ) (muslim_percent : ℚ) (sikh_percent : ℚ) (other_boys : ℕ) : 
  total_boys = 650 →
  muslim_percent = 44 / 100 →
  sikh_percent = 10 / 100 →
  other_boys = 117 →
  (total_boys - (muslim_percent * total_boys + sikh_percent * total_boys + other_boys)) / total_boys = 28 / 100 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_hindu_boys_l3439_343942


namespace NUMINAMATH_CALUDE_no_two_digit_factors_of_1806_l3439_343972

theorem no_two_digit_factors_of_1806 : 
  ¬∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 1806 :=
by sorry

end NUMINAMATH_CALUDE_no_two_digit_factors_of_1806_l3439_343972


namespace NUMINAMATH_CALUDE_right_triangle_equality_l3439_343916

theorem right_triangle_equality (a b c p : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a^2 + b^2 = c^2) (h5 : 2*p = a + b + c) : 
  let S := (1/2) * a * b
  p * (p - c) = (p - a) * (p - b) ∧ p * (p - c) = S := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_equality_l3439_343916


namespace NUMINAMATH_CALUDE_max_congruent_spherical_triangles_l3439_343987

/-- A spherical triangle on the surface of a sphere --/
structure SphericalTriangle where
  -- Add necessary fields for a spherical triangle
  is_on_sphere : Bool
  sides_are_great_circle_arcs : Bool
  sides_less_than_quarter : Bool

/-- A division of a sphere into congruent spherical triangles --/
structure SphereDivision where
  triangles : List SphericalTriangle
  are_congruent : Bool

/-- The maximum number of congruent spherical triangles that satisfy the conditions --/
def max_congruent_triangles : ℕ := 60

/-- Theorem stating that 60 is the maximum number of congruent spherical triangles --/
theorem max_congruent_spherical_triangles :
  ∀ (d : SphereDivision),
    (∀ t ∈ d.triangles, t.is_on_sphere ∧ t.sides_are_great_circle_arcs ∧ t.sides_less_than_quarter) →
    d.are_congruent →
    d.triangles.length ≤ max_congruent_triangles :=
by
  sorry

#check max_congruent_spherical_triangles

end NUMINAMATH_CALUDE_max_congruent_spherical_triangles_l3439_343987


namespace NUMINAMATH_CALUDE_money_distribution_l3439_343997

theorem money_distribution (total : ℕ) (faruk vasim ranjith : ℕ) : 
  faruk + vasim + ranjith = total →
  3 * vasim = 5 * faruk →
  8 * faruk = 3 * ranjith →
  ranjith - faruk = 1500 →
  vasim = 1500 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3439_343997


namespace NUMINAMATH_CALUDE_probability_is_half_l3439_343967

/-- A game board represented as a regular hexagon -/
structure HexagonalBoard :=
  (total_segments : ℕ)
  (shaded_segments : ℕ)
  (is_regular : total_segments = 6)
  (shaded_constraint : shaded_segments = 3)

/-- The probability of a spinner landing on a shaded region of a hexagonal board -/
def probability_shaded (board : HexagonalBoard) : ℚ :=
  board.shaded_segments / board.total_segments

/-- Theorem stating that the probability of landing on a shaded region is 1/2 -/
theorem probability_is_half (board : HexagonalBoard) :
  probability_shaded board = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l3439_343967


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3439_343979

theorem root_sum_reciprocal (a b c : ℝ) : 
  (a^3 - a - 2 = 0) → 
  (b^3 - b - 2 = 0) → 
  (c^3 - c - 2 = 0) → 
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = -3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3439_343979


namespace NUMINAMATH_CALUDE_tree_planting_ratio_l3439_343990

/-- Represents the number of trees planted by each grade --/
structure TreePlanting where
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ

/-- The conditions of the tree planting activity --/
def treePlantingConditions (t : TreePlanting) : Prop :=
  t.fourth = 30 ∧
  t.sixth = 3 * t.fifth - 30 ∧
  t.fourth + t.fifth + t.sixth = 240

/-- The theorem stating the ratio of trees planted by 5th graders to 4th graders --/
theorem tree_planting_ratio (t : TreePlanting) :
  treePlantingConditions t → (t.fifth : ℚ) / t.fourth = 2 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_ratio_l3439_343990


namespace NUMINAMATH_CALUDE_distance_between_vertices_l3439_343948

-- Define the equation
def equation (x y : ℝ) : Prop := Real.sqrt (x^2 + y^2) + |y - 2| = 4

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = 3 - (1/12) * x^2
def parabola2 (x y : ℝ) : Prop := y = (1/4) * x^2 - 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices : 
  ∃ (x1 y1 x2 y2 : ℝ), 
    parabola1 x1 y1 ∧ 
    parabola2 x2 y2 ∧ 
    (x1, y1) = vertex1 ∧ 
    (x2, y2) = vertex2 ∧ 
    Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l3439_343948


namespace NUMINAMATH_CALUDE_number_value_l3439_343922

theorem number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 75 := by
  sorry

end NUMINAMATH_CALUDE_number_value_l3439_343922


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3439_343932

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  b * (Real.sin B - Real.sin C) + (c - a) * (Real.sin A + Real.sin C) = 0 →
  a = Real.sqrt 3 →
  Real.sin C = (1 + Real.sqrt 3) / 2 * Real.sin B →
  -- Conclusions
  A = π / 3 ∧
  (1 / 2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 4 := by
sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l3439_343932


namespace NUMINAMATH_CALUDE_quadratic_function_point_comparison_l3439_343921

/-- Given a quadratic function y = x² - 4x + k passing through (-1, y₁) and (3, y₂), prove y₁ > y₂ -/
theorem quadratic_function_point_comparison (k : ℝ) (y₁ y₂ : ℝ)
  (h₁ : y₁ = (-1)^2 - 4*(-1) + k)
  (h₂ : y₂ = 3^2 - 4*3 + k) :
  y₁ > y₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_point_comparison_l3439_343921


namespace NUMINAMATH_CALUDE_rent_percentage_last_year_l3439_343933

theorem rent_percentage_last_year (E : ℝ) (P : ℝ) : 
  E > 0 → 
  (0.30 * (1.25 * E) = 1.875 * (P / 100) * E) → 
  P = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_rent_percentage_last_year_l3439_343933


namespace NUMINAMATH_CALUDE_X_4_equivalence_l3439_343988

-- Define the type for a die
def Die : Type := Fin 6

-- Define the type for a pair of dice
def DicePair : Type := Die × Die

-- Define the sum of points on a pair of dice
def sum_points (pair : DicePair) : Nat :=
  pair.1.val + 1 + pair.2.val + 1

-- Define the event X = 4
def X_equals_4 (pair : DicePair) : Prop :=
  sum_points pair = 4

-- Define the event where one die shows 3 and the other shows 1
def one_3_one_1 (pair : DicePair) : Prop :=
  (pair.1.val = 2 ∧ pair.2.val = 0) ∨ (pair.1.val = 0 ∧ pair.2.val = 2)

-- Define the event where both dice show 2
def both_2 (pair : DicePair) : Prop :=
  pair.1.val = 1 ∧ pair.2.val = 1

-- Theorem: X = 4 is equivalent to (one 3 and one 1) or (both 2)
theorem X_4_equivalence (pair : DicePair) :
  X_equals_4 pair ↔ one_3_one_1 pair ∨ both_2 pair :=
sorry

end NUMINAMATH_CALUDE_X_4_equivalence_l3439_343988


namespace NUMINAMATH_CALUDE_tangent_circle_radii_product_l3439_343937

/-- A circle passing through (3,4) and tangent to both axes -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_point : (center.1 - 3)^2 + (center.2 - 4)^2 = radius^2
  tangent_to_x_axis : center.2 = radius
  tangent_to_y_axis : center.1 = radius

/-- The two possible radii of tangent circles -/
def radii : ℝ × ℝ :=
  let a := TangentCircle.radius
  let equation := a^2 - 14*a + 25 = 0
  sorry

theorem tangent_circle_radii_product :
  let (r₁, r₂) := radii
  r₁ * r₂ = 25 := by sorry

end NUMINAMATH_CALUDE_tangent_circle_radii_product_l3439_343937


namespace NUMINAMATH_CALUDE_hex_to_binary_bits_l3439_343976

/-- The number of bits required to represent a positive integer in binary. -/
def bitsRequired (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

/-- The decimal representation of the hexadecimal number 1A1A1. -/
def hexNumber : ℕ := 106913

theorem hex_to_binary_bits :
  bitsRequired hexNumber = 17 := by
  sorry

end NUMINAMATH_CALUDE_hex_to_binary_bits_l3439_343976


namespace NUMINAMATH_CALUDE_na_minimum_at_3_l3439_343918

-- Define the sequence S_n
def S (n : ℕ) : ℤ := n^2 - 10*n

-- Define a_n as the difference between consecutive S_n terms
def a (n : ℕ) : ℤ := S n - S (n-1)

-- Define na_n
def na (n : ℕ) : ℤ := n * (a n)

-- Theorem statement
theorem na_minimum_at_3 :
  ∀ k : ℕ, k ≥ 1 → na 3 ≤ na k :=
sorry

end NUMINAMATH_CALUDE_na_minimum_at_3_l3439_343918


namespace NUMINAMATH_CALUDE_inequality_solution_l3439_343919

theorem inequality_solution (x : ℝ) : 
  (x^3 - 4*x) / (x^2 - 4*x + 4) > 0 ↔ (x > -2 ∧ x < 0) ∨ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3439_343919


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l3439_343909

theorem solve_fraction_equation :
  ∀ x : ℚ, (2 / 3 - 1 / 4 : ℚ) = 1 / x → x = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l3439_343909


namespace NUMINAMATH_CALUDE_binomial_variance_example_l3439_343973

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: The variance of a binomial random variable with n=10 and p=1/4 is 15/8 -/
theorem binomial_variance_example : 
  ∀ ξ : BinomialRV, ξ.n = 10 ∧ ξ.p = 1/4 → variance ξ = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l3439_343973


namespace NUMINAMATH_CALUDE_cost_per_metre_l3439_343936

/-- Given that John bought 9.25 m of cloth for $416.25, prove that the cost price per metre is $45. -/
theorem cost_per_metre (total_length : ℝ) (total_cost : ℝ) 
  (h1 : total_length = 9.25)
  (h2 : total_cost = 416.25) :
  total_cost / total_length = 45 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_metre_l3439_343936


namespace NUMINAMATH_CALUDE_total_water_volume_l3439_343926

def water_volume (num_containers : ℕ) (container_volume : ℝ) : ℝ :=
  (num_containers : ℝ) * container_volume

theorem total_water_volume : 
  water_volume 2812 4 = 11248 := by sorry

end NUMINAMATH_CALUDE_total_water_volume_l3439_343926
