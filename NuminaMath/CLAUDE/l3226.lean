import Mathlib

namespace NUMINAMATH_CALUDE_limit_f_at_origin_l3226_322674

/-- The function f(x, y) = (x^2 + y^2)^2 x^2 y^2 -/
def f (x y : ℝ) : ℝ := (x^2 + y^2)^2 * x^2 * y^2

/-- The limit of f(x, y) as x and y approach 0 is 1 -/
theorem limit_f_at_origin :
  ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ, x^2 + y^2 < δ^2 → |f x y - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_f_at_origin_l3226_322674


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3226_322687

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x(x+3) = 0 -/
def f (x : ℝ) : ℝ := x * (x + 3)

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3226_322687


namespace NUMINAMATH_CALUDE_equation_system_implies_third_equation_l3226_322683

theorem equation_system_implies_third_equation (a b : ℝ) :
  a^2 - 3*a*b + 2*b^2 + a - b = 0 →
  a^2 - 2*a*b + b^2 - 5*a + 7*b = 0 →
  a*b - 12*a + 15*b = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_system_implies_third_equation_l3226_322683


namespace NUMINAMATH_CALUDE_expand_polynomial_simplify_expression_l3226_322659

-- Problem 1
theorem expand_polynomial (x : ℝ) : x * (x + 3) * (x + 5) = x^3 + 8*x^2 + 15*x := by
  sorry

-- Problem 2
theorem simplify_expression (x y : ℝ) : (5*x + 2*y) * (5*x - 2*y) - 5*x * (5*x - 3*y) = -4*y^2 + 15*x*y := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_simplify_expression_l3226_322659


namespace NUMINAMATH_CALUDE_movie_count_theorem_l3226_322686

/-- The number of movies Timothy and Theresa watched in 2009 and 2010 -/
def total_movies (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ) : ℕ :=
  timothy_2009 + timothy_2010 + theresa_2009 + theresa_2010

theorem movie_count_theorem :
  ∀ (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ),
    timothy_2010 = timothy_2009 + 7 →
    timothy_2009 = 24 →
    theresa_2010 = 2 * timothy_2010 →
    theresa_2009 = timothy_2009 / 2 →
    total_movies timothy_2009 timothy_2010 theresa_2009 theresa_2010 = 129 :=
by
  sorry

end NUMINAMATH_CALUDE_movie_count_theorem_l3226_322686


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_factors_of_30_l3226_322684

def factors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

theorem sum_of_reciprocals_of_factors_of_30 :
  (factors_of_30.map (λ x => (1 : ℚ) / x)).sum = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_factors_of_30_l3226_322684


namespace NUMINAMATH_CALUDE_average_value_sequence_l3226_322680

theorem average_value_sequence (y : ℝ) : 
  (16*y + 8*y + 4*y + 2*y + 0) / 5 = 6*y := by
  sorry

end NUMINAMATH_CALUDE_average_value_sequence_l3226_322680


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l3226_322662

theorem minimum_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^(2*b))) :
  (∀ x y, x > 0 → y > 0 → 2/x + 1/y ≥ 2/a + 1/b) → 2/a + 1/b = 8 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l3226_322662


namespace NUMINAMATH_CALUDE_compute_expression_l3226_322696

theorem compute_expression : 18 * (250 / 3 + 36 / 9 + 16 / 32 + 2) = 1617 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3226_322696


namespace NUMINAMATH_CALUDE_water_remaining_l3226_322622

theorem water_remaining (poured_out : ℚ) (h : poured_out = 45 / 100) :
  1 - poured_out = 55 / 100 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l3226_322622


namespace NUMINAMATH_CALUDE_symmetric_angles_relation_l3226_322652

/-- If the terminal sides of angles α and β are symmetric about the x-axis,
    then α can be expressed as 2kπ - β for some integer k. -/
theorem symmetric_angles_relation (α β : Real) :
  (∃ k : ℤ, α + β = 2 * k * Real.pi) →
  (∃ k : ℤ, α = 2 * k * Real.pi - β) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_angles_relation_l3226_322652


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_eight_satisfies_inequality_eight_is_smallest_integer_l3226_322630

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, y < 3*y - 15 → y ≥ 8 :=
by
  sorry

theorem eight_satisfies_inequality : 
  (8 : ℤ) < 3*8 - 15 :=
by
  sorry

theorem eight_is_smallest_integer :
  ∃ y : ℤ, y < 3*y - 15 ∧ ∀ z : ℤ, z < 3*z - 15 → z ≥ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_eight_satisfies_inequality_eight_is_smallest_integer_l3226_322630


namespace NUMINAMATH_CALUDE_cube_root_square_l3226_322647

theorem cube_root_square (x : ℝ) : (x - 1)^(1/3) = 3 → (x - 1)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_square_l3226_322647


namespace NUMINAMATH_CALUDE_farm_sale_earnings_l3226_322658

/-- Calculates the total money earned from selling farm animals -/
def total_money_earned (num_cows : ℕ) (pig_cow_ratio : ℕ) (price_per_pig : ℕ) (price_per_cow : ℕ) : ℕ :=
  let num_pigs := num_cows * pig_cow_ratio
  let money_from_pigs := num_pigs * price_per_pig
  let money_from_cows := num_cows * price_per_cow
  money_from_pigs + money_from_cows

/-- Theorem stating that given the specific conditions, the total money earned is $48,000 -/
theorem farm_sale_earnings : total_money_earned 20 4 400 800 = 48000 := by
  sorry

end NUMINAMATH_CALUDE_farm_sale_earnings_l3226_322658


namespace NUMINAMATH_CALUDE_exists_special_polynomial_l3226_322633

/-- A fifth-degree polynomial with specific properties on [-1,1] -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ (x₁ x₂ : ℝ), -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧
    p x₁ = 1 ∧ p (-x₂) = 1 ∧ p (-x₁) = -1 ∧ p x₂ = -1) ∧
  p (-1) = 0 ∧ p 1 = 0 ∧
  ∀ x, x ∈ Set.Icc (-1) 1 → -1 ≤ p x ∧ p x ≤ 1

/-- There exists a fifth-degree polynomial with the special properties -/
theorem exists_special_polynomial :
  ∃ (p : ℝ → ℝ), ∃ (a b c d e f : ℝ),
    (∀ x, p x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) ∧
    special_polynomial p :=
sorry

end NUMINAMATH_CALUDE_exists_special_polynomial_l3226_322633


namespace NUMINAMATH_CALUDE_intersection_value_l3226_322676

theorem intersection_value (m n : ℝ) (h1 : n = 2 / m) (h2 : n = m + 3) :
  1 / m - 1 / n = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l3226_322676


namespace NUMINAMATH_CALUDE_systematic_sampling_l3226_322695

theorem systematic_sampling 
  (population_size : ℕ) 
  (num_groups : ℕ) 
  (sample_size : ℕ) 
  (first_draw : ℕ) :
  population_size = 60 →
  num_groups = 6 →
  sample_size = 6 →
  first_draw = 3 →
  let interval := population_size / num_groups
  let fifth_group_draw := first_draw + interval * 4
  fifth_group_draw = 43 := by
sorry


end NUMINAMATH_CALUDE_systematic_sampling_l3226_322695


namespace NUMINAMATH_CALUDE_probability_log_base_3_is_integer_l3226_322691

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_power_of_three (n : ℕ) : Prop := ∃ k : ℕ, n = 3^k

def count_three_digit_powers_of_three : ℕ := 2

def total_three_digit_numbers : ℕ := 900

theorem probability_log_base_3_is_integer :
  (count_three_digit_powers_of_three : ℚ) / (total_three_digit_numbers : ℚ) = 1 / 450 := by
  sorry

#check probability_log_base_3_is_integer

end NUMINAMATH_CALUDE_probability_log_base_3_is_integer_l3226_322691


namespace NUMINAMATH_CALUDE_quadratic_inequalities_intersection_l3226_322663

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

theorem quadratic_inequalities_intersection (a b : ℝ) :
  ({x : ℝ | x^2 + a*x + b < 0} = A ∩ B) →
  a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_intersection_l3226_322663


namespace NUMINAMATH_CALUDE_ab_plus_cd_equals_45_l3226_322670

theorem ab_plus_cd_equals_45 (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 5)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = 14) :
  a * b + c * d = 45 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_equals_45_l3226_322670


namespace NUMINAMATH_CALUDE_parallel_line_a_value_l3226_322612

/-- Given two points A and B on a line parallel to 2x - y + 1 = 0, prove a = 2 -/
theorem parallel_line_a_value (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (-1, a) ∧ 
    B = (a, 8) ∧ 
    (∃ (m : ℝ), (B.2 - A.2) = m * (B.1 - A.1) ∧ m = 2)) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_a_value_l3226_322612


namespace NUMINAMATH_CALUDE_otimes_inequality_iff_interval_l3226_322614

/-- Custom binary operation ⊗ on real numbers -/
def otimes (a b : ℝ) : ℝ := a * b + 2 * a + b

/-- Theorem stating the equivalence between the inequality and the interval -/
theorem otimes_inequality_iff_interval (x : ℝ) :
  otimes x (x - 2) < 0 ↔ -2 < x ∧ x < 1 :=
sorry

end NUMINAMATH_CALUDE_otimes_inequality_iff_interval_l3226_322614


namespace NUMINAMATH_CALUDE_fraction_simplification_l3226_322609

theorem fraction_simplification : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3226_322609


namespace NUMINAMATH_CALUDE_triangle_inequality_l3226_322632

theorem triangle_inequality (a b c α β γ : ℝ) (n : ℕ) : 
  a > 0 → b > 0 → c > 0 → 
  α > 0 → β > 0 → γ > 0 → 
  α + β + γ = π → 
  (π/3)^n ≤ (a*α^n + b*β^n + c*γ^n) / (a + b + c) ∧ 
  (a*α^n + b*β^n + c*γ^n) / (a + b + c) < π^n/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3226_322632


namespace NUMINAMATH_CALUDE_pond_radius_l3226_322664

/-- The radius of a circular pond with a diameter of 14 meters is 7 meters. -/
theorem pond_radius (diameter : ℝ) (h : diameter = 14) : diameter / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_pond_radius_l3226_322664


namespace NUMINAMATH_CALUDE_validSquaresCount_l3226_322634

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  topLeft : Nat × Nat

/-- Checks if a square contains at least 7 black squares -/
def hasAtLeast7BlackSquares (s : Square) : Bool :=
  sorry

/-- Counts the number of valid squares on the checkerboard -/
def countValidSquares : Nat :=
  sorry

/-- Theorem stating the correct number of valid squares -/
theorem validSquaresCount :
  countValidSquares = 140 := by sorry

end NUMINAMATH_CALUDE_validSquaresCount_l3226_322634


namespace NUMINAMATH_CALUDE_gift_contributors_l3226_322602

theorem gift_contributors (total : ℝ) (min_contribution : ℝ) (max_contribution : ℝ) :
  total = 20 →
  min_contribution = 1 →
  max_contribution = 9 →
  (∃ (n : ℕ), n ≥ 1 ∧ n * min_contribution ≤ total ∧ total ≤ n * max_contribution) →
  (∀ (m : ℕ), m ≥ 1 → m * min_contribution ≤ total → total ≤ m * max_contribution → m ≥ 12) :=
by sorry

end NUMINAMATH_CALUDE_gift_contributors_l3226_322602


namespace NUMINAMATH_CALUDE_stratified_sampling_second_group_l3226_322666

theorem stratified_sampling_second_group (total_sample : ℕ) 
  (ratio_first ratio_second ratio_third : ℕ) :
  ratio_first > 0 ∧ ratio_second > 0 ∧ ratio_third > 0 →
  total_sample = 240 →
  ratio_first = 5 ∧ ratio_second = 4 ∧ ratio_third = 3 →
  (ratio_second : ℚ) / (ratio_first + ratio_second + ratio_third : ℚ) * total_sample = 80 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_group_l3226_322666


namespace NUMINAMATH_CALUDE_james_paper_usage_l3226_322631

/-- The number of books James prints -/
def num_books : ℕ := 2

/-- The number of pages in each book -/
def pages_per_book : ℕ := 600

/-- The number of pages printed on each side of a sheet -/
def pages_per_side : ℕ := 4

/-- Whether the printing is double-sided (true) or single-sided (false) -/
def is_double_sided : Bool := true

/-- Calculates the total number of sheets of paper James uses -/
def sheets_used : ℕ :=
  let total_pages := num_books * pages_per_book
  let pages_per_sheet := pages_per_side * (if is_double_sided then 2 else 1)
  total_pages / pages_per_sheet

theorem james_paper_usage :
  sheets_used = 150 := by sorry

end NUMINAMATH_CALUDE_james_paper_usage_l3226_322631


namespace NUMINAMATH_CALUDE_sequence_convergence_l3226_322690

theorem sequence_convergence (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, (a (n + 2))^2 + a (n + 1) * a n ≤ a (n + 2) * (a (n + 1) + a n)) :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 2) = a n :=
sorry

end NUMINAMATH_CALUDE_sequence_convergence_l3226_322690


namespace NUMINAMATH_CALUDE_circle_P_properties_l3226_322616

/-- Given a circle P with center (a, b) and radius R -/
theorem circle_P_properties (a b R : ℝ) :
  R^2 - b^2 = 2 →
  R^2 - a^2 = 3 →
  (∃ x y : ℝ, y^2 - x^2 = 1) ∧
  (|b - a| = 1 →
    ((∃ x y : ℝ, x^2 + (y - 1)^2 = 3) ∨
     (∃ x y : ℝ, x^2 + (y + 1)^2 = 3))) :=
by sorry

end NUMINAMATH_CALUDE_circle_P_properties_l3226_322616


namespace NUMINAMATH_CALUDE_square_area_ratio_l3226_322654

/-- Given a large square and a small square with coinciding centers,
    if the area of the cross formed by the small square is 17 times
    the area of the small square, then the area of the large square
    is 81 times the area of the small square. -/
theorem square_area_ratio (small_side large_side : ℝ) : 
  small_side > 0 →
  large_side > 0 →
  2 * large_side * small_side - small_side^2 = 17 * small_side^2 →
  large_side^2 = 81 * small_side^2 := by
  sorry

#check square_area_ratio

end NUMINAMATH_CALUDE_square_area_ratio_l3226_322654


namespace NUMINAMATH_CALUDE_digit_extraction_l3226_322657

theorem digit_extraction (a b c : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) :
  let S := 100 * a + 10 * b + c
  (S / 100 = a) ∧ ((S / 10) % 10 = b) ∧ (S % 10 = c) := by
  sorry

end NUMINAMATH_CALUDE_digit_extraction_l3226_322657


namespace NUMINAMATH_CALUDE_admission_criteria_correct_l3226_322668

/-- Represents the admission score criteria for art students in a high school. -/
structure AdmissionCriteria where
  x : ℝ  -- Professional score
  y : ℝ  -- Total score of liberal arts
  z : ℝ  -- Physical education score

/-- Defines the correct admission criteria based on the given conditions. -/
def correct_criteria (c : AdmissionCriteria) : Prop :=
  c.x ≥ 95 ∧ c.y > 380 ∧ c.z > 45

/-- Theorem stating that the given inequalities correctly represent the admission criteria. -/
theorem admission_criteria_correct (c : AdmissionCriteria) :
  (c.x ≥ 95 ∧ c.y > 380 ∧ c.z > 45) ↔ correct_criteria c :=
by sorry

end NUMINAMATH_CALUDE_admission_criteria_correct_l3226_322668


namespace NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l3226_322697

/-- The area of a triangle with base 12 cm and height 15 cm is 90 cm². -/
theorem triangle_area_with_given_base_and_height :
  let base : ℝ := 12
  let height : ℝ := 15
  let area : ℝ := (1 / 2) * base * height
  area = 90 := by sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l3226_322697


namespace NUMINAMATH_CALUDE_car_sales_profit_loss_percentage_l3226_322640

/-- Calculates the overall profit or loss percentage for two car sales --/
theorem car_sales_profit_loss_percentage 
  (selling_price : ℝ) 
  (gain_percentage : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : selling_price > 0) 
  (h2 : gain_percentage > 0) 
  (h3 : loss_percentage > 0) 
  (h4 : gain_percentage = loss_percentage) : 
  ∃ (loss_percent : ℝ), 
    loss_percent > 0 ∧ 
    loss_percent < gain_percentage ∧
    loss_percent = (2 * selling_price - (selling_price / (1 + gain_percentage / 100) + selling_price / (1 - loss_percentage / 100))) / 
                   (selling_price / (1 + gain_percentage / 100) + selling_price / (1 - loss_percentage / 100)) * 100 := by
  sorry

end NUMINAMATH_CALUDE_car_sales_profit_loss_percentage_l3226_322640


namespace NUMINAMATH_CALUDE_f_monotonic_decreasing_iff_a_in_range_l3226_322643

-- Define the function f(x) = ax|x-a|
def f (a : ℝ) (x : ℝ) : ℝ := a * x * abs (x - a)

-- Define the property of being monotonically decreasing on an interval
def monotonically_decreasing (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → g y < g x

-- State the theorem
theorem f_monotonic_decreasing_iff_a_in_range :
  ∀ a : ℝ, (monotonically_decreasing (f a) 1 (3/2)) ↔ 
    (a < 0 ∨ (3/2 ≤ a ∧ a ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_decreasing_iff_a_in_range_l3226_322643


namespace NUMINAMATH_CALUDE_product_divisibility_l3226_322672

theorem product_divisibility (a b c : ℤ) 
  (h1 : (a + b + c)^2 = -(a*b + a*c + b*c))
  (h2 : a + b ≠ 0)
  (h3 : b + c ≠ 0)
  (h4 : a + c ≠ 0) :
  (∃ k : ℤ, (a + b) * (a + c) = k * (b + c)) ∧
  (∃ k : ℤ, (b + c) * (b + a) = k * (a + c)) ∧
  (∃ k : ℤ, (c + a) * (c + b) = k * (a + b)) :=
sorry

end NUMINAMATH_CALUDE_product_divisibility_l3226_322672


namespace NUMINAMATH_CALUDE_cat_max_distance_l3226_322644

/-- The maximum distance a cat can be from the origin, given it's tied to a post -/
theorem cat_max_distance (post_x post_y rope_length : ℝ) : 
  post_x = 6 → post_y = 8 → rope_length = 15 → 
  ∃ (max_distance : ℝ), max_distance = 25 ∧ 
  ∀ (cat_x cat_y : ℝ), 
    (cat_x - post_x)^2 + (cat_y - post_y)^2 ≤ rope_length^2 → 
    cat_x^2 + cat_y^2 ≤ max_distance^2 :=
by sorry

end NUMINAMATH_CALUDE_cat_max_distance_l3226_322644


namespace NUMINAMATH_CALUDE_problem_solution_l3226_322679

theorem problem_solution (x k : ℕ) (h1 : (2^x) - (2^(x-2)) = k * (2^10)) (h2 : x = 12) : k = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3226_322679


namespace NUMINAMATH_CALUDE_certain_negative_integer_l3226_322615

theorem certain_negative_integer (a b : ℤ) (x : ℤ) : 
  (-11 * a < 0) →
  (x < 0) →
  (x * b < 0) →
  ((-11 * a * x) * (x * b) + a * b = 89) →
  x = -1 :=
by sorry

end NUMINAMATH_CALUDE_certain_negative_integer_l3226_322615


namespace NUMINAMATH_CALUDE_bamboo_volume_sum_l3226_322675

/-- Given a sequence of 9 terms forming an arithmetic progression,
    where the sum of the first 4 terms is 3 and the sum of the last 3 terms is 4,
    prove that the sum of the 2nd, 3rd, and 8th terms is 17/6. -/
theorem bamboo_volume_sum (a : Fin 9 → ℚ) 
  (arithmetic_seq : ∀ i j k : Fin 9, a (i + 1) - a i = a (j + 1) - a j)
  (sum_first_four : a 0 + a 1 + a 2 + a 3 = 3)
  (sum_last_three : a 6 + a 7 + a 8 = 4) :
  a 1 + a 2 + a 7 = 17/6 := by
  sorry

end NUMINAMATH_CALUDE_bamboo_volume_sum_l3226_322675


namespace NUMINAMATH_CALUDE_triangle_sine_cosine_identity_l3226_322661

/-- For angles A, B, and C of a triangle, sin A + sin B + sin C = 4 cos(A/2) cos(B/2) cos(C/2). -/
theorem triangle_sine_cosine_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_cosine_identity_l3226_322661


namespace NUMINAMATH_CALUDE_trajectory_of_symmetric_point_l3226_322669

/-- The trajectory of point P, symmetric to a point Q on the curve y = x^2 - 2 with respect to point A(1, 0) -/
theorem trajectory_of_symmetric_point :
  let A : ℝ × ℝ := (1, 0)
  let C : ℝ → ℝ := fun x => x^2 - 2
  ∀ Q : ℝ × ℝ, (Q.2 = C Q.1) →
  ∀ P : ℝ × ℝ, (Q.1 = 2 - P.1 ∧ Q.2 = -P.2) →
  P.2 = -P.1^2 + 4*P.1 - 2 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_symmetric_point_l3226_322669


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3226_322692

theorem complex_modulus_problem (z : ℂ) :
  (1 + Complex.I) * z = 1 - 2 * Complex.I^3 →
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3226_322692


namespace NUMINAMATH_CALUDE_garden_flowers_l3226_322660

/-- Represents a rectangular garden with a rose planted at a specific position -/
structure Garden where
  rows_front : Nat  -- Number of rows in front of the rose
  rows_back : Nat   -- Number of rows behind the rose
  cols_left : Nat   -- Number of columns to the left of the rose
  cols_right : Nat  -- Number of columns to the right of the rose

/-- Calculates the total number of flowers in the garden -/
def total_flowers (g : Garden) : Nat :=
  (g.rows_front + 1 + g.rows_back) * (g.cols_left + 1 + g.cols_right)

/-- Theorem stating the total number of flowers in the specified garden -/
theorem garden_flowers :
  let g : Garden := {
    rows_front := 6,
    rows_back := 15,
    cols_left := 8,
    cols_right := 12
  }
  total_flowers g = 462 := by
  sorry

#eval total_flowers { rows_front := 6, rows_back := 15, cols_left := 8, cols_right := 12 }

end NUMINAMATH_CALUDE_garden_flowers_l3226_322660


namespace NUMINAMATH_CALUDE_x_intercept_is_six_l3226_322626

-- Define the line equation
def line_equation (x y : ℚ) : Prop := 4 * x - 3 * y = 24

-- Define x-intercept
def is_x_intercept (x : ℚ) : Prop := line_equation x 0

-- Theorem statement
theorem x_intercept_is_six : is_x_intercept 6 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_is_six_l3226_322626


namespace NUMINAMATH_CALUDE_rancher_feed_corn_cost_l3226_322650

/-- Represents the rancher's farm and animals -/
structure Farm where
  sheep : ℕ
  cattle : ℕ
  pasture_acres : ℕ
  cow_grass_consumption : ℕ
  sheep_grass_consumption : ℕ
  feed_corn_cost : ℕ
  feed_corn_cow_duration : ℕ
  feed_corn_sheep_duration : ℕ

/-- Calculates the yearly cost of feed corn for the farm -/
def yearly_feed_corn_cost (f : Farm) : ℕ :=
  let total_monthly_grass_consumption := f.cattle * f.cow_grass_consumption + f.sheep * f.sheep_grass_consumption
  let grazing_months := f.pasture_acres / total_monthly_grass_consumption
  let feed_corn_months := 12 - grazing_months
  let monthly_feed_corn_bags := f.cattle + f.sheep / f.feed_corn_sheep_duration
  let total_feed_corn_bags := monthly_feed_corn_bags * feed_corn_months
  total_feed_corn_bags * f.feed_corn_cost

/-- The main theorem stating the yearly cost of feed corn for the given farm -/
theorem rancher_feed_corn_cost :
  let farm := Farm.mk 8 5 144 2 1 10 1 2
  yearly_feed_corn_cost farm = 360 := by
  sorry

end NUMINAMATH_CALUDE_rancher_feed_corn_cost_l3226_322650


namespace NUMINAMATH_CALUDE_rectangle_area_l3226_322624

/-- The area of a rectangle with perimeter 40 feet and length twice its width is 800/9 square feet. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) 
  (h1 : perimeter = 40)
  (h2 : length = 2 * width)
  (h3 : perimeter = 2 * length + 2 * width)
  (h4 : area = length * width) :
  area = 800 / 9 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_l3226_322624


namespace NUMINAMATH_CALUDE_seating_arrangements_l3226_322607

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem seating_arrangements (n : ℕ) (k : ℕ) (h : n = 10 ∧ k = 3) :
  factorial n - factorial (n - k + 1) * factorial k = 3598560 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3226_322607


namespace NUMINAMATH_CALUDE_cleaning_fluid_purchase_l3226_322673

theorem cleaning_fluid_purchase :
  ∃ (x y : ℕ), 
    30 * x + 20 * y = 160 ∧ 
    x + y = 7 ∧
    ∀ (a b : ℕ), 30 * a + 20 * b = 160 → x + y ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_cleaning_fluid_purchase_l3226_322673


namespace NUMINAMATH_CALUDE_max_value_of_function_l3226_322642

theorem max_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  ∃ (max_val : ℝ), max_val = 1/8 ∧ ∀ y, 0 < y ∧ y < 1/2 → x * (1 - 2*x) ≤ max_val := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3226_322642


namespace NUMINAMATH_CALUDE_hidden_block_surface_area_l3226_322665

/-- Represents a block with a surface area -/
structure Block where
  surfaceArea : ℝ

/-- Represents a set of blocks created by cutting a larger block -/
structure CutBlocks where
  blocks : List Block
  numCuts : ℕ

/-- The proposition that the surface area of the hidden block is correct -/
def hiddenBlockSurfaceAreaIsCorrect (cb : CutBlocks) (hiddenSurfaceArea : ℝ) : Prop :=
  cb.numCuts = 3 ∧ 
  cb.blocks.length = 7 ∧ 
  (cb.blocks.map Block.surfaceArea).sum = 566 ∧
  hiddenSurfaceArea = 22

/-- Theorem stating that given the conditions, the hidden block's surface area is 22 -/
theorem hidden_block_surface_area 
  (cb : CutBlocks) (hiddenSurfaceArea : ℝ) : 
  hiddenBlockSurfaceAreaIsCorrect cb hiddenSurfaceArea := by
  sorry

#check hidden_block_surface_area

end NUMINAMATH_CALUDE_hidden_block_surface_area_l3226_322665


namespace NUMINAMATH_CALUDE_vector_linear_combination_l3226_322629

/-- Given two planar vectors a and b, prove that their linear combination results in the specified vector. -/
theorem vector_linear_combination (a b : ℝ × ℝ) :
  a = (1, 1) → b = (1, -1) → (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l3226_322629


namespace NUMINAMATH_CALUDE_weaving_sum_l3226_322600

/-- The sum of an arithmetic sequence with first term 5, common difference 16/29, and 30 terms -/
theorem weaving_sum : 
  let a₁ : ℚ := 5
  let d : ℚ := 16 / 29
  let n : ℕ := 30
  (n : ℚ) * a₁ + (n * (n - 1) : ℚ) / 2 * d = 390 := by
  sorry

end NUMINAMATH_CALUDE_weaving_sum_l3226_322600


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l3226_322655

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersecting_lines_k_value (x y k : ℚ) : 
  (y = 6 * x + 4) ∧ 
  (y = -3 * x - 30) ∧ 
  (y = 4 * x + k) → 
  k = -32/9 := by sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l3226_322655


namespace NUMINAMATH_CALUDE_round_robin_tournament_sessions_l3226_322646

/-- The number of matches in a round-robin tournament with n players -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The minimum number of sessions required for a tournament -/
def min_sessions (total_matches : ℕ) (max_per_session : ℕ) : ℕ :=
  (total_matches + max_per_session - 1) / max_per_session

theorem round_robin_tournament_sessions :
  let n : ℕ := 10  -- number of players
  let max_per_session : ℕ := 15  -- maximum matches per session
  min_sessions (num_matches n) max_per_session = 3 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_sessions_l3226_322646


namespace NUMINAMATH_CALUDE_right_square_prism_volume_l3226_322651

/-- Represents the dimensions of a rectangle --/
structure RectangleDimensions where
  length : ℝ
  width : ℝ

/-- Represents the volume of a right square prism --/
def prism_volume (base_side : ℝ) (height : ℝ) : ℝ :=
  base_side ^ 2 * height

/-- Theorem stating the possible volumes of the right square prism --/
theorem right_square_prism_volume 
  (lateral_surface : RectangleDimensions)
  (h_length : lateral_surface.length = 12)
  (h_width : lateral_surface.width = 8) :
  ∃ (v : ℝ), (v = prism_volume 3 8 ∨ v = prism_volume 2 12) ∧ 
             (v = 72 ∨ v = 48) := by
  sorry

end NUMINAMATH_CALUDE_right_square_prism_volume_l3226_322651


namespace NUMINAMATH_CALUDE_value_of_y_l3226_322688

theorem value_of_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l3226_322688


namespace NUMINAMATH_CALUDE_gasoline_reduction_l3226_322693

theorem gasoline_reduction (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 1.2 * P
  let new_total_cost := 1.14 * (P * Q)
  let new_quantity := new_total_cost / new_price
  (Q - new_quantity) / Q = 0.05 := by
sorry

end NUMINAMATH_CALUDE_gasoline_reduction_l3226_322693


namespace NUMINAMATH_CALUDE_circle_symmetry_l3226_322619

/-- Given a circle with center (1,2) and radius 1, symmetric about the line y = x + b,
    prove that b = 1 -/
theorem circle_symmetry (b : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 1 ↔ (y - x = b ∧ (x + y - 3)^2 + (y - x - b)^2 / 4 = 1)) →
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3226_322619


namespace NUMINAMATH_CALUDE_prob_two_out_of_three_germinate_l3226_322637

/-- The probability of exactly k successes in n independent Bernoulli trials 
    with probability p of success for each trial -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of exactly 2 successes out of 3 trials 
    with probability 4/5 of success for each trial -/
theorem prob_two_out_of_three_germinate : 
  binomial_probability 3 2 (4/5) = 48/125 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_out_of_three_germinate_l3226_322637


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3226_322694

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  (∀ a : ℝ, a = 1 → |a| = 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ |a| = 1) := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3226_322694


namespace NUMINAMATH_CALUDE_students_studying_all_subjects_l3226_322671

theorem students_studying_all_subjects (total : ℕ) (math : ℕ) (latin : ℕ) (chem : ℕ) 
  (multi : ℕ) (none : ℕ) (h1 : total = 425) (h2 : math = 351) (h3 : latin = 71) 
  (h4 : chem = 203) (h5 : multi = 199) (h6 : none = 8) : 
  ∃ x : ℕ, x = 9 ∧ 
  total - none = math + latin + chem - multi + x := by
  sorry

end NUMINAMATH_CALUDE_students_studying_all_subjects_l3226_322671


namespace NUMINAMATH_CALUDE_longest_piece_length_l3226_322625

theorem longest_piece_length (a b c : ℕ) (ha : a = 45) (hb : b = 75) (hc : c = 90) :
  Nat.gcd a (Nat.gcd b c) = 15 := by
  sorry

end NUMINAMATH_CALUDE_longest_piece_length_l3226_322625


namespace NUMINAMATH_CALUDE_unique_quaternary_polynomial_l3226_322636

/-- A polynomial with coefficients in {0, 1, 2, 3} -/
def QuaternaryPolynomial := List (Fin 4)

/-- Evaluate a quaternary polynomial at x = 2 -/
def evalAt2 (p : QuaternaryPolynomial) : ℕ :=
  p.enum.foldl (fun acc (i, coef) => acc + coef.val * 2^i) 0

theorem unique_quaternary_polynomial (n : ℕ) (hn : n > 0) :
  ∃! p : QuaternaryPolynomial, evalAt2 p = n := by sorry

end NUMINAMATH_CALUDE_unique_quaternary_polynomial_l3226_322636


namespace NUMINAMATH_CALUDE_max_d_value_l3226_322677

def a (n : ℕ+) : ℕ := 103 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (N : ℕ+), d N = 13 ∧ ∀ (n : ℕ+), d n ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3226_322677


namespace NUMINAMATH_CALUDE_schedule_theorem_l3226_322635

-- Define the number of periods in a day
def periods : ℕ := 7

-- Define the number of courses to be scheduled
def courses : ℕ := 4

-- Define a function to calculate the number of ways to schedule courses
def schedule_ways (p : ℕ) (c : ℕ) : ℕ := sorry

-- Theorem statement
theorem schedule_theorem : 
  schedule_ways periods courses = 120 := by sorry

end NUMINAMATH_CALUDE_schedule_theorem_l3226_322635


namespace NUMINAMATH_CALUDE_memory_card_cost_l3226_322604

/-- If three identical memory cards cost $45 in total, then eight of these memory cards will cost $120. -/
theorem memory_card_cost (cost_of_three : ℝ) : cost_of_three = 45 → 8 * (cost_of_three / 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_memory_card_cost_l3226_322604


namespace NUMINAMATH_CALUDE_algorithm_uniqueness_false_l3226_322618

-- Define the concept of an algorithm
structure Algorithm where
  finite : Bool
  determinate : Bool
  outputProperty : Bool

-- Define the property of uniqueness for algorithms
def isUnique (problemClass : Type) (alg : Algorithm) : Prop :=
  ∀ (otherAlg : Algorithm), alg = otherAlg

-- Theorem statement
theorem algorithm_uniqueness_false :
  ∃ (problemClass : Type) (alg1 alg2 : Algorithm),
    alg1.finite ∧ alg1.determinate ∧ alg1.outputProperty ∧
    alg2.finite ∧ alg2.determinate ∧ alg2.outputProperty ∧
    alg1 ≠ alg2 :=
sorry

end NUMINAMATH_CALUDE_algorithm_uniqueness_false_l3226_322618


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l3226_322613

theorem quadratic_equation_result (a : ℝ) (h : a^2 + 3*a - 4 = 0) : 2*a^2 + 6*a - 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l3226_322613


namespace NUMINAMATH_CALUDE_cake_cost_is_correct_l3226_322685

/-- The cost of a piece of cake in dollars -/
def cake_cost : ℚ := 7

/-- The cost of a cup of coffee in dollars -/
def coffee_cost : ℚ := 4

/-- The cost of a bowl of ice cream in dollars -/
def ice_cream_cost : ℚ := 3

/-- The total cost for Mell and her two friends in dollars -/
def total_cost : ℚ := 51

/-- Theorem stating that the cake cost is correct given the conditions -/
theorem cake_cost_is_correct :
  cake_cost = 7 ∧
  coffee_cost = 4 ∧
  ice_cream_cost = 3 ∧
  total_cost = 51 ∧
  (2 * coffee_cost + cake_cost) + 2 * (2 * coffee_cost + cake_cost + ice_cream_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_cake_cost_is_correct_l3226_322685


namespace NUMINAMATH_CALUDE_survey_III_participants_l3226_322611

/-- Represents the systematic sampling method for a school survey. -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_selected : ℕ
  survey_III_start : ℕ
  survey_III_end : ℕ

/-- The number of students participating in Survey III. -/
def students_in_survey_III (s : SystematicSampling) : ℕ :=
  let n_start := ((s.survey_III_start + s.first_selected - 1) + s.group_size - 1) / s.group_size
  let n_end := (s.survey_III_end + s.first_selected - 1) / s.group_size
  n_end - n_start + 1

/-- Theorem stating the number of students in Survey III for the given conditions. -/
theorem survey_III_participants (s : SystematicSampling) 
  (h1 : s.total_students = 1080)
  (h2 : s.sample_size = 90)
  (h3 : s.group_size = s.total_students / s.sample_size)
  (h4 : s.first_selected = 5)
  (h5 : s.survey_III_start = 847)
  (h6 : s.survey_III_end = 1080) :
  students_in_survey_III s = 19 := by
  sorry

#eval students_in_survey_III {
  total_students := 1080,
  sample_size := 90,
  group_size := 12,
  first_selected := 5,
  survey_III_start := 847,
  survey_III_end := 1080
}

end NUMINAMATH_CALUDE_survey_III_participants_l3226_322611


namespace NUMINAMATH_CALUDE_cricketer_average_score_l3226_322627

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_matches : ℕ) 
  (last_matches : ℕ) 
  (first_average : ℚ) 
  (last_average : ℚ) 
  (h1 : total_matches = first_matches + last_matches) 
  (h2 : total_matches = 10) 
  (h3 : first_matches = 6) 
  (h4 : last_matches = 4) 
  (h5 : first_average = 42) 
  (h6 : last_average = 34.25) : 
  (first_average * first_matches + last_average * last_matches) / total_matches = 38.9 := by
sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l3226_322627


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3226_322682

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) :
  area = (1 / 2) * leg1 * leg2 →
  leg1 = 30 →
  area = 150 →
  leg2 * leg2 + leg1 * leg1 = hypotenuse * hypotenuse →
  leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3226_322682


namespace NUMINAMATH_CALUDE_cubic_factorization_l3226_322698

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3226_322698


namespace NUMINAMATH_CALUDE_transport_percentage_l3226_322639

/-- Calculate the percentage of income spent on public transport -/
theorem transport_percentage (income : ℝ) (remaining : ℝ) 
  (h1 : income = 2000)
  (h2 : remaining = 1900) :
  (income - remaining) / income * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_transport_percentage_l3226_322639


namespace NUMINAMATH_CALUDE_arithmetic_mean_change_l3226_322667

theorem arithmetic_mean_change (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 10 →
  (b + c + d) / 3 = 11 →
  (a + c + d) / 3 = 12 →
  (a + b + d) / 3 = 13 →
  (a + b + c) / 3 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_change_l3226_322667


namespace NUMINAMATH_CALUDE_HE_in_possible_values_l3226_322681

/-- A quadrilateral with side lengths satisfying certain conditions -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℤ)
  (ef_eq : EF = 7)
  (fg_eq : FG = 21)
  (gh_eq : GH = 7)

/-- The possible values for HE in the quadrilateral -/
def possible_HE (q : Quadrilateral) : Set ℤ :=
  {n : ℤ | 15 ≤ n ∧ n ≤ 27}

/-- The theorem stating that HE must be in the set of possible values -/
theorem HE_in_possible_values (q : Quadrilateral) : q.HE ∈ possible_HE q := by
  sorry

end NUMINAMATH_CALUDE_HE_in_possible_values_l3226_322681


namespace NUMINAMATH_CALUDE_symmetric_center_phi_l3226_322645

theorem symmetric_center_phi (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (-2 * x + φ)) →
  0 < φ →
  φ < π →
  (∃ k : ℤ, -2 * (π / 3) + φ = k * π) →
  φ = 2 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_symmetric_center_phi_l3226_322645


namespace NUMINAMATH_CALUDE_three_numbers_problem_l3226_322617

theorem three_numbers_problem :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x = 1.4 * y ∧
  x / z = 14 / 11 ∧
  z - y = 0.125 * (x + y) - 40 ∧
  x = 280 ∧ y = 200 ∧ z = 220 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l3226_322617


namespace NUMINAMATH_CALUDE_gcd_12012_21021_l3226_322610

theorem gcd_12012_21021 : Nat.gcd 12012 21021 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12012_21021_l3226_322610


namespace NUMINAMATH_CALUDE_amount_ratio_l3226_322603

theorem amount_ratio (total : ℕ) (r_amount : ℕ) : 
  total = 7000 →
  r_amount = 2800 →
  (r_amount : ℚ) / ((total - r_amount) : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_amount_ratio_l3226_322603


namespace NUMINAMATH_CALUDE_intersection_distance_l3226_322623

/-- A cube with vertices at (0,0,0), (6,0,0), (6,6,0), (0,6,0), (0,0,6), (6,0,6), (6,6,6), and (0,6,6) -/
def cube : Set (Fin 3 → ℝ) :=
  {v | ∀ i, v i ∈ ({0, 6} : Set ℝ)}

/-- The plane cutting the cube -/
def plane (x y z : ℝ) : Prop :=
  -3 * x + 10 * y + 4 * z = 30

/-- The plane cuts the edges of the cube at these points -/
axiom plane_cuts : plane 0 3 0 ∧ plane 6 0 3 ∧ plane 2 6 6

/-- The intersection point on the edge from (0,0,0) to (0,0,6) -/
def U : Fin 3 → ℝ := ![0, 0, 3]

/-- The intersection point on the edge from (6,6,0) to (6,6,6) -/
def V : Fin 3 → ℝ := ![6, 6, 3]

/-- The theorem to be proved -/
theorem intersection_distance : 
  U ∈ cube ∧ V ∈ cube ∧ plane (U 0) (U 1) (U 2) ∧ plane (V 0) (V 1) (V 2) →
  Real.sqrt (((U 0 - V 0)^2 + (U 1 - V 1)^2 + (U 2 - V 2)^2) : ℝ) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l3226_322623


namespace NUMINAMATH_CALUDE_triangle_property_l3226_322638

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2a*sin(B) = √3*b, a = 6, and b = 2√3, then angle A = π/3 and the area is 6√3 --/
theorem triangle_property (a b c A B C : Real) : 
  0 < A ∧ A < π/2 →  -- Acute triangle condition
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  2 * a * Real.sin B = Real.sqrt 3 * b →  -- Given condition
  a = 6 →  -- Given condition
  b = 2 * Real.sqrt 3 →  -- Given condition
  A = π/3 ∧ (1/2 * b * c * Real.sin A = 6 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3226_322638


namespace NUMINAMATH_CALUDE_turban_price_is_correct_l3226_322641

def initial_yearly_salary : ℚ := 90
def months_before_raise : ℕ := 6
def raise_percentage : ℚ := 1/10
def total_months_worked : ℕ := 9
def final_cash_payment : ℚ := 65

def monthly_salary : ℚ := initial_yearly_salary / 12
def raised_monthly_salary : ℚ := monthly_salary * (1 + raise_percentage)

def total_cash_earned : ℚ := 
  monthly_salary * months_before_raise + 
  raised_monthly_salary * (total_months_worked - months_before_raise)

def turban_price : ℚ := total_cash_earned - final_cash_payment

theorem turban_price_is_correct : turban_price = 4.75 := by sorry

end NUMINAMATH_CALUDE_turban_price_is_correct_l3226_322641


namespace NUMINAMATH_CALUDE_specific_frustum_small_cone_altitude_l3226_322605

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the altitude of the small cone cut off from a frustum -/
def small_cone_altitude (f : Frustum) : ℝ :=
  f.altitude

/-- Theorem stating that for a specific frustum, the altitude of the small cone is 18 cm -/
theorem specific_frustum_small_cone_altitude :
  let f : Frustum := {
    altitude := 18,
    lower_base_area := 144 * Real.pi,
    upper_base_area := 36 * Real.pi
  }
  small_cone_altitude f = 18 := by sorry

end NUMINAMATH_CALUDE_specific_frustum_small_cone_altitude_l3226_322605


namespace NUMINAMATH_CALUDE_parallel_transitive_perpendicular_from_line_l3226_322648

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Axioms for parallel and perpendicular relations
axiom parallel_symm {a b : Plane} : parallel a b → parallel b a
axiom perpendicular_symm {a b : Plane} : perpendicular a b → perpendicular b a

-- Theorem 1
theorem parallel_transitive {α β γ : Plane} :
  parallel α β → parallel α γ → parallel β γ := by sorry

-- Theorem 2
theorem perpendicular_from_line {m : Line} {α β : Plane} :
  line_perpendicular m α → line_parallel m β → perpendicular α β := by sorry

end NUMINAMATH_CALUDE_parallel_transitive_perpendicular_from_line_l3226_322648


namespace NUMINAMATH_CALUDE_smaller_number_expression_l3226_322653

theorem smaller_number_expression (m n t s : ℝ) 
  (positive_m : 0 < m) 
  (positive_n : 0 < n) 
  (ratio : m / n = t) 
  (t_greater_one : t > 1) 
  (sum : m + n = s) : 
  n = s / (1 + t) := by
sorry

end NUMINAMATH_CALUDE_smaller_number_expression_l3226_322653


namespace NUMINAMATH_CALUDE_toad_ratio_is_25_to_1_l3226_322628

/-- Represents the number of toads per acre -/
structure ToadPopulation where
  green : ℕ
  brown : ℕ
  spotted_brown : ℕ

/-- The ratio of brown toads to green toads -/
def brown_to_green_ratio (pop : ToadPopulation) : ℚ :=
  pop.brown / pop.green

theorem toad_ratio_is_25_to_1 (pop : ToadPopulation) 
  (h1 : pop.green = 8)
  (h2 : pop.spotted_brown = 50)
  (h3 : pop.spotted_brown * 4 = pop.brown) : 
  brown_to_green_ratio pop = 25 := by
  sorry

end NUMINAMATH_CALUDE_toad_ratio_is_25_to_1_l3226_322628


namespace NUMINAMATH_CALUDE_even_periodic_function_value_l3226_322649

/-- A function that is even and has a period of 2 -/
def EvenPeriodicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ (∀ x, f (x + 2) = f x)

theorem even_periodic_function_value 
  (f : ℝ → ℝ) 
  (h_even_periodic : EvenPeriodicFunction f)
  (h_def : ∀ x ∈ Set.Ioo 0 1, f x = x + 1) :
  ∀ x ∈ Set.Ioo 1 2, f x = 3 - x := by
sorry

end NUMINAMATH_CALUDE_even_periodic_function_value_l3226_322649


namespace NUMINAMATH_CALUDE_number_of_students_l3226_322699

theorem number_of_students (n : ℕ) : 
  (n : ℝ) * 15 = 7 * 14 + 7 * 16 + 15 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l3226_322699


namespace NUMINAMATH_CALUDE_special_blend_probability_l3226_322621

theorem special_blend_probability : 
  let n : ℕ := 6  -- Total number of visits
  let k : ℕ := 5  -- Number of times the special blend is served
  let p : ℚ := 3/4  -- Probability of serving the special blend each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 1458/4096 := by
sorry

end NUMINAMATH_CALUDE_special_blend_probability_l3226_322621


namespace NUMINAMATH_CALUDE_min_colors_theorem_l3226_322608

/-- A coloring of positive integers -/
def Coloring (k : ℕ) := ℕ+ → Fin k

/-- A function from positive integers to positive integers -/
def IntFunction := ℕ+ → ℕ+

/-- The property that f(m+n) = f(m) + f(n) for integers of the same color -/
def SameColorAdditive (f : IntFunction) (c : Coloring k) : Prop :=
  ∀ m n : ℕ+, c m = c n → f (m + n) = f m + f n

/-- The property that there exist m and n such that f(m+n) ≠ f(m) + f(n) -/
def ExistsDifferentSum (f : IntFunction) : Prop :=
  ∃ m n : ℕ+, f (m + n) ≠ f m + f n

/-- The main theorem -/
theorem min_colors_theorem :
  (∃ k : ℕ+, ∃ c : Coloring k, ∃ f : IntFunction,
    SameColorAdditive f c ∧ ExistsDifferentSum f) ∧
  (∀ k : ℕ+, k < 3 → ¬∃ c : Coloring k, ∃ f : IntFunction,
    SameColorAdditive f c ∧ ExistsDifferentSum f) :=
sorry

end NUMINAMATH_CALUDE_min_colors_theorem_l3226_322608


namespace NUMINAMATH_CALUDE_number_square_relationship_l3226_322620

theorem number_square_relationship (n : ℝ) (h1 : n ≠ 0) (h2 : (n + n^2) / 2 = 5 * n) (h3 : n = 9) :
  (n + n^2) / 2 = 5 * n :=
by sorry

end NUMINAMATH_CALUDE_number_square_relationship_l3226_322620


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l3226_322689

theorem magnitude_of_complex_power (z : ℂ) :
  z = (4:ℝ)/7 + (3:ℝ)/7 * Complex.I →
  Complex.abs (z^8) = 390625/5764801 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l3226_322689


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3226_322656

def M : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}
def N : Set ℝ := {x | x ≤ -3 ∨ x ≥ 6}

theorem intersection_of_M_and_N :
  M ∩ N = {x | -5 ≤ x ∧ x ≤ -3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3226_322656


namespace NUMINAMATH_CALUDE_students_above_120_l3226_322606

/-- Normal distribution parameters -/
structure NormalDist where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability function for normal distribution -/
noncomputable def prob (nd : NormalDist) (a b : ℝ) : ℝ := sorry

/-- Theorem: Number of students scoring above 120 -/
theorem students_above_120 (nd : NormalDist) (total_students : ℕ) :
  nd.μ = 90 →
  prob nd 60 120 = 0.8 →
  total_students = 780 →
  ⌊(1 - prob nd 60 120) / 2 * total_students⌋ = 78 := by sorry

end NUMINAMATH_CALUDE_students_above_120_l3226_322606


namespace NUMINAMATH_CALUDE_log_power_base_equality_l3226_322601

theorem log_power_base_equality (a N m n : ℝ) 
  (ha : a > 0) (hN : N > 0) (hm : m ≠ 0) :
  Real.log N^n / Real.log a^m = n / m * Real.log N / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_log_power_base_equality_l3226_322601


namespace NUMINAMATH_CALUDE_f_prime_at_two_l3226_322678

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_prime_at_two
  (h1 : (1 - 0) / (2 - 0) = 1 / 2)  -- Slope of line through (0,0) and (2,1) is 1/2
  (h2 : f 0 = 0)                    -- f(0) = 0
  (h3 : f 2 = 2)                    -- f(2) = 2
  (h4 : (2 * (deriv f 2) - (f 2)) / (2^2) = 1 / 2)  -- Derivative of f(x)/x at x=2 equals slope
  : deriv f 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_f_prime_at_two_l3226_322678
