import Mathlib

namespace NUMINAMATH_CALUDE_lesser_number_problem_l223_22362

theorem lesser_number_problem (x y : ℝ) 
  (sum_eq : x + y = 50) 
  (diff_eq : x - y = 7) : 
  y = 21.5 := by
sorry

end NUMINAMATH_CALUDE_lesser_number_problem_l223_22362


namespace NUMINAMATH_CALUDE_gcf_of_75_and_135_l223_22345

theorem gcf_of_75_and_135 : Nat.gcd 75 135 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_135_l223_22345


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l223_22354

theorem percentage_increase_proof (original_earnings new_earnings : ℝ) 
  (h1 : original_earnings = 60)
  (h2 : new_earnings = 84) :
  ((new_earnings - original_earnings) / original_earnings) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l223_22354


namespace NUMINAMATH_CALUDE_high_quality_seed_probability_l223_22300

/-- Represents the composition of seeds in a batch -/
structure SeedBatch where
  second_grade : ℝ
  third_grade : ℝ
  fourth_grade : ℝ

/-- Represents the probabilities of producing high-quality products for each seed grade -/
structure QualityProbabilities where
  first_grade : ℝ
  second_grade : ℝ
  third_grade : ℝ
  fourth_grade : ℝ

/-- Calculates the probability of selecting a high-quality seed from a given batch -/
def high_quality_probability (batch : SeedBatch) (probs : QualityProbabilities) : ℝ :=
  let first_grade_proportion := 1 - (batch.second_grade + batch.third_grade + batch.fourth_grade)
  first_grade_proportion * probs.first_grade +
  batch.second_grade * probs.second_grade +
  batch.third_grade * probs.third_grade +
  batch.fourth_grade * probs.fourth_grade

/-- Theorem stating the probability of selecting a high-quality seed from the given batch -/
theorem high_quality_seed_probability :
  let batch := SeedBatch.mk 0.02 0.015 0.01
  let probs := QualityProbabilities.mk 0.5 0.15 0.1 0.05
  high_quality_probability batch probs = 0.4825 := by
  sorry


end NUMINAMATH_CALUDE_high_quality_seed_probability_l223_22300


namespace NUMINAMATH_CALUDE_distribute_seven_into_four_l223_22385

/-- Number of ways to distribute indistinguishable objects into distinct containers -/
def distribute_objects (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 indistinguishable objects into 4 distinct containers -/
theorem distribute_seven_into_four :
  distribute_objects 7 4 = 132 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_into_four_l223_22385


namespace NUMINAMATH_CALUDE_lost_shoes_count_l223_22374

/-- Given an initial number of shoe pairs and a remaining number of matching pairs,
    calculate the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * remaining_pairs

/-- Theorem stating that with 20 initial pairs and 15 remaining pairs,
    10 individual shoes are lost. -/
theorem lost_shoes_count : shoes_lost 20 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_lost_shoes_count_l223_22374


namespace NUMINAMATH_CALUDE_fraction_ordering_l223_22376

theorem fraction_ordering : (4 : ℚ) / 13 < 12 / 37 ∧ 12 / 37 < 15 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l223_22376


namespace NUMINAMATH_CALUDE_find_x_l223_22340

theorem find_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4 * a) ^ (2 * b) = (a ^ b * x ^ b) ^ 2 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_find_x_l223_22340


namespace NUMINAMATH_CALUDE_max_product_permutation_l223_22332

theorem max_product_permutation (a : Fin 1987 → ℕ) 
  (h_perm : Function.Bijective a) 
  (h_range : Set.range a = Finset.range 1988) : 
  (Finset.range 1988).sup (λ k => k * a k) ≥ 994^2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_permutation_l223_22332


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l223_22302

-- Define the function f
def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x - 3)

-- State the theorem
theorem tangent_slope_at_zero :
  (deriv f) 0 = -6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l223_22302


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_120_l223_22337

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

theorem largest_five_digit_with_product_120 :
  ∀ n : ℕ, is_five_digit n → digit_product n = 120 → n ≤ 85311 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_120_l223_22337


namespace NUMINAMATH_CALUDE_combined_age_l223_22367

theorem combined_age (tony_age belinda_age : ℕ) : 
  tony_age = 16 →
  belinda_age = 40 →
  belinda_age = 2 * tony_age + 8 →
  tony_age + belinda_age = 56 :=
by sorry

end NUMINAMATH_CALUDE_combined_age_l223_22367


namespace NUMINAMATH_CALUDE_base_10_to_base_8_l223_22379

theorem base_10_to_base_8 : 
  (3 * 8^3 + 1 * 8^2 + 4 * 8^1 + 0 * 8^0 : ℕ) = 1632 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_8_l223_22379


namespace NUMINAMATH_CALUDE_regular_10gon_triangle_probability_l223_22306

/-- Regular 10-gon -/
def regular_10gon : Set (ℝ × ℝ) := sorry

/-- Set of all segments in the 10-gon -/
def segments (polygon : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

/-- Predicate to check if three segments form a triangle with positive area -/
def forms_triangle (s1 s2 s3 : Set (ℝ × ℝ)) : Prop := sorry

/-- The probability of forming a triangle with positive area from three randomly chosen segments -/
def triangle_probability (polygon : Set (ℝ × ℝ)) : ℚ := sorry

/-- Main theorem: The probability of forming a triangle with positive area 
    from three distinct segments chosen randomly from a regular 10-gon is 343/715 -/
theorem regular_10gon_triangle_probability : 
  triangle_probability regular_10gon = 343 / 715 := by sorry

end NUMINAMATH_CALUDE_regular_10gon_triangle_probability_l223_22306


namespace NUMINAMATH_CALUDE_complex_fraction_equals_negative_i_l223_22317

theorem complex_fraction_equals_negative_i :
  ∀ (i : ℂ), i * i = -1 →
  (1 - i) / (1 + i) = -i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_negative_i_l223_22317


namespace NUMINAMATH_CALUDE_watermelon_weight_l223_22391

theorem watermelon_weight (total_weight : ℝ) (half_removed_weight : ℝ) 
  (h1 : total_weight = 63)
  (h2 : half_removed_weight = 34) :
  let watermelon_weight := total_weight - half_removed_weight * 2
  watermelon_weight = 58 := by
sorry

end NUMINAMATH_CALUDE_watermelon_weight_l223_22391


namespace NUMINAMATH_CALUDE_first_quartile_of_data_set_l223_22360

def data_set : List ℕ := [296, 301, 305, 293, 293, 305, 302, 303, 306, 294]

def first_quartile (l : List ℕ) : ℕ := sorry

theorem first_quartile_of_data_set :
  first_quartile data_set = 294 := by sorry

end NUMINAMATH_CALUDE_first_quartile_of_data_set_l223_22360


namespace NUMINAMATH_CALUDE_quiz_goal_achievement_l223_22357

theorem quiz_goal_achievement (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (current_as : ℕ) : 
  total_quizzes = 40 →
  goal_percentage = 85 / 100 →
  completed_quizzes = 25 →
  current_as = 20 →
  ∃ (max_non_as : ℕ), 
    max_non_as = 1 ∧ 
    (current_as + (total_quizzes - completed_quizzes - max_non_as)) / total_quizzes ≥ goal_percentage ∧
    ∀ (x : ℕ), x > max_non_as → 
      (current_as + (total_quizzes - completed_quizzes - x)) / total_quizzes < goal_percentage :=
by sorry

end NUMINAMATH_CALUDE_quiz_goal_achievement_l223_22357


namespace NUMINAMATH_CALUDE_sum_removal_proof_l223_22347

theorem sum_removal_proof : 
  let original_sum := (1 : ℚ) / 3 + 1 / 6 + 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18
  let removed_sum := (1 : ℚ) / 12 + 1 / 15
  original_sum - removed_sum = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_removal_proof_l223_22347


namespace NUMINAMATH_CALUDE_reggie_brother_long_shots_l223_22399

-- Define the point values for each shot type
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define Reggie's shots
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := 1

-- Define the point difference
def point_difference : ℕ := 2

-- Theorem to prove
theorem reggie_brother_long_shots :
  let reggie_points := reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points
  let brother_points := reggie_points + point_difference
  brother_points / long_shot_points = 4 :=
by sorry

end NUMINAMATH_CALUDE_reggie_brother_long_shots_l223_22399


namespace NUMINAMATH_CALUDE_find_b_find_a_range_l223_22348

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - b * x

-- Define the derivative of f
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := a / x + (1 - a) * x - b

-- Theorem 1: Find the value of b
theorem find_b (a : ℝ) (h : a ≠ 1) :
  (∃ b : ℝ, f_deriv a b 1 = 0) → (∃ b : ℝ, b = 1) :=
sorry

-- Theorem 2: Find the range of values for a
theorem find_a_range (a : ℝ) (h : a ≠ 1) :
  (∃ x : ℝ, x ≥ 1 ∧ f a 1 x < a / (a - 1)) →
  (a ∈ Set.Ioo (- Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∪ Set.Ioi 1) :=
sorry

end

end NUMINAMATH_CALUDE_find_b_find_a_range_l223_22348


namespace NUMINAMATH_CALUDE_price_per_book_is_two_l223_22366

/-- Represents the sale of books with given conditions -/
def BookSale (total_books : ℕ) (price_per_book : ℚ) : Prop :=
  (2 : ℚ) / 3 * total_books + 36 = total_books ∧
  (2 : ℚ) / 3 * total_books * price_per_book = 144

/-- Theorem stating that the price per book is $2 given the conditions -/
theorem price_per_book_is_two :
  ∃ (total_books : ℕ), BookSale total_books 2 := by
  sorry

end NUMINAMATH_CALUDE_price_per_book_is_two_l223_22366


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_of_13_l223_22319

theorem least_four_digit_multiple_of_13 : ∃ n : ℕ, 
  n % 13 = 0 ∧ 
  n ≥ 1000 ∧ 
  n < 10000 ∧ 
  (∀ m : ℕ, m % 13 = 0 ∧ m ≥ 1000 ∧ m < 10000 → n ≤ m) ∧
  n = 1001 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_of_13_l223_22319


namespace NUMINAMATH_CALUDE_pencil_count_l223_22349

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 2

/-- The number of pencils Tim added to the drawer -/
def added_pencils : ℕ := 3

/-- The total number of pencils in the drawer after Tim's action -/
def total_pencils : ℕ := initial_pencils + added_pencils

theorem pencil_count : total_pencils = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l223_22349


namespace NUMINAMATH_CALUDE_solve_equation_l223_22388

theorem solve_equation : ∃ x : ℝ, 15 * x = 5.7 ∧ x = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l223_22388


namespace NUMINAMATH_CALUDE_line_slope_is_two_l223_22361

/-- Given a line with y-intercept 2 and passing through the point (498, 998), its slope is 2 -/
theorem line_slope_is_two (f : ℝ → ℝ) (h1 : f 0 = 2) (h2 : f 498 = 998) :
  (f 498 - f 0) / (498 - 0) = 2 := by
sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l223_22361


namespace NUMINAMATH_CALUDE_value_of_a_l223_22303

def f (x : ℝ) : ℝ := 3 * x - 1

def A (a : ℝ) : Set ℝ := {1, a}
def B (a : ℝ) : Set ℝ := {a, 5}

theorem value_of_a : ∃ a : ℝ, (∀ x ∈ A a, f x ∈ B a) ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l223_22303


namespace NUMINAMATH_CALUDE_total_new_emails_formula_l223_22334

/-- Represents the number of new emails received in one deletion cycle -/
def new_emails_per_cycle : ℕ := 15 + 5

/-- Represents the final batch of emails received -/
def final_batch : ℕ := 10

/-- Calculates the total number of new emails after n cycles and a final batch -/
def total_new_emails (n : ℕ) : ℕ := n * new_emails_per_cycle + final_batch

/-- Theorem stating the total number of new emails after n cycles and a final batch -/
theorem total_new_emails_formula (n : ℕ) : 
  total_new_emails n = 20 * n + 10 := by
  sorry

#eval total_new_emails 5  -- Example evaluation

end NUMINAMATH_CALUDE_total_new_emails_formula_l223_22334


namespace NUMINAMATH_CALUDE_lcm_36_105_l223_22392

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_105_l223_22392


namespace NUMINAMATH_CALUDE_max_square_plots_l223_22397

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℕ
  width : ℕ

/-- Represents the available fencing and field dimensions -/
structure FencingProblem where
  field : FieldDimensions
  available_fencing : ℕ

/-- Calculates the number of square plots given the side length -/
def num_plots (f : FieldDimensions) (side : ℕ) : ℕ :=
  (f.length / side) * (f.width / side)

/-- Calculates the required internal fencing given the side length -/
def required_fencing (f : FieldDimensions) (side : ℕ) : ℕ :=
  f.length * ((f.width / side) - 1) + f.width * ((f.length / side) - 1)

/-- Theorem: The maximum number of square plots is 18 -/
theorem max_square_plots (p : FencingProblem) 
  (h1 : p.field.length = 30)
  (h2 : p.field.width = 60)
  (h3 : p.available_fencing = 2500) :
  ∃ (side : ℕ), 
    side > 0 ∧ 
    side ∣ p.field.length ∧ 
    side ∣ p.field.width ∧
    required_fencing p.field side ≤ p.available_fencing ∧
    num_plots p.field side = 18 ∧
    ∀ (other_side : ℕ), other_side > side → 
      ¬(other_side ∣ p.field.length ∧ 
        other_side ∣ p.field.width ∧
        required_fencing p.field other_side ≤ p.available_fencing) :=
  sorry

end NUMINAMATH_CALUDE_max_square_plots_l223_22397


namespace NUMINAMATH_CALUDE_parallel_lines_l223_22373

def is_parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ * c₂ ≠ a₂ * c₁

theorem parallel_lines :
  is_parallel 1 (-2) 1 2 (-4) 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_l223_22373


namespace NUMINAMATH_CALUDE_unique_number_with_digit_sum_l223_22346

/-- Given a natural number n, returns the sum of its digits. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number n is a three-digit number. -/
def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem unique_number_with_digit_sum : 
  ∃! n : ℕ, isThreeDigitNumber n ∧ n + sumOfDigits n = 328 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_digit_sum_l223_22346


namespace NUMINAMATH_CALUDE_smallest_x_abs_equation_l223_22311

theorem smallest_x_abs_equation : 
  (∀ x : ℝ, |2*x + 5| = 21 → x ≥ -13) ∧ 
  (|2*(-13) + 5| = 21) := by
sorry

end NUMINAMATH_CALUDE_smallest_x_abs_equation_l223_22311


namespace NUMINAMATH_CALUDE_soccer_ball_donation_l223_22312

theorem soccer_ball_donation (total_balls : ℕ) (balls_per_class : ℕ) 
  (elementary_classes_per_school : ℕ) (num_schools : ℕ) 
  (h1 : total_balls = 90) 
  (h2 : balls_per_class = 5)
  (h3 : elementary_classes_per_school = 4)
  (h4 : num_schools = 2) : 
  (total_balls / (balls_per_class * num_schools)) - elementary_classes_per_school = 5 := by
  sorry

#check soccer_ball_donation

end NUMINAMATH_CALUDE_soccer_ball_donation_l223_22312


namespace NUMINAMATH_CALUDE_sqrt_of_nine_l223_22342

theorem sqrt_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_l223_22342


namespace NUMINAMATH_CALUDE_y_sixth_power_root_l223_22389

theorem y_sixth_power_root (y : ℝ) (hy : y > 0) (h : Real.sin (Real.arctan y) = y^3) :
  ∃ (z : ℝ), z > 0 ∧ z^3 + z^2 - 1 = 0 ∧ y^6 = z := by
  sorry

end NUMINAMATH_CALUDE_y_sixth_power_root_l223_22389


namespace NUMINAMATH_CALUDE_two_cos_sixty_degrees_l223_22323

theorem two_cos_sixty_degrees : 2 * Real.cos (π / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_two_cos_sixty_degrees_l223_22323


namespace NUMINAMATH_CALUDE_no_arithmetic_progression_l223_22341

theorem no_arithmetic_progression : 
  ¬∃ (y : ℝ), (∃ (d : ℝ), (3*y + 1) - (y - 3) = d ∧ (5*y - 7) - (3*y + 1) = d) := by
  sorry

end NUMINAMATH_CALUDE_no_arithmetic_progression_l223_22341


namespace NUMINAMATH_CALUDE_min_value_expression_l223_22301

theorem min_value_expression (x y : ℝ) : (x*y)^2 + (x + 7)^2 + (2*y + 7)^2 ≥ 45 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l223_22301


namespace NUMINAMATH_CALUDE_mikes_books_l223_22398

theorem mikes_books (initial_books new_books : ℕ) : 
  initial_books = 35 → new_books = 56 → initial_books + new_books = 91 := by
  sorry

end NUMINAMATH_CALUDE_mikes_books_l223_22398


namespace NUMINAMATH_CALUDE_original_number_proof_l223_22313

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l223_22313


namespace NUMINAMATH_CALUDE_no_max_cos_squared_sum_l223_22368

theorem no_max_cos_squared_sum (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  ∃ d > 0, B - A = d ∧ C - B = d →  -- Arithmetic sequence with positive difference
  ¬ ∃ M : ℝ, ∀ A' B' C' : ℝ,
    (0 < A' ∧ 0 < B' ∧ 0 < C' ∧
     A' + B' + C' = π ∧
     ∃ d > 0, B' - A' = d ∧ C' - B' = d) →
    Real.cos A' ^ 2 + Real.cos C' ^ 2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_no_max_cos_squared_sum_l223_22368


namespace NUMINAMATH_CALUDE_function_constant_l223_22377

/-- A function satisfying the given functional equation is constant -/
theorem function_constant (f : ℝ → ℝ) 
    (h : ∀ (x y : ℝ), x > 0 → y > 0 → f (Real.sqrt (x * y)) = f ((x + y) / 2)) :
  ∀ (a b : ℝ), a > 0 → b > 0 → f a = f b := by sorry

end NUMINAMATH_CALUDE_function_constant_l223_22377


namespace NUMINAMATH_CALUDE_salary_increase_after_reduction_l223_22395

theorem salary_increase_after_reduction (original_salary : ℝ) (h : original_salary > 0) :
  let reduced_salary := original_salary * (1 - 0.35)
  let increase_factor := (1 / 0.65) - 1
  reduced_salary * (1 + increase_factor) = original_salary := by
  sorry

#eval (1 / 0.65 - 1) * 100 -- To show the approximate percentage increase

end NUMINAMATH_CALUDE_salary_increase_after_reduction_l223_22395


namespace NUMINAMATH_CALUDE_smallest_possible_b_l223_22378

theorem smallest_possible_b : 
  ∃ (b : ℝ), ∀ (a : ℝ), 
    (2 < a ∧ a < b) → 
    (2 + a ≤ b) → 
    (1 / a + 1 / b ≤ 1 / 2) → 
    (b = 3 + Real.sqrt 5) ∧
    (∀ (b' : ℝ), 
      (∃ (a' : ℝ), 
        (2 < a' ∧ a' < b') ∧ 
        (2 + a' ≤ b') ∧ 
        (1 / a' + 1 / b' ≤ 1 / 2)) → 
      b ≤ b') :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l223_22378


namespace NUMINAMATH_CALUDE_family_average_age_l223_22396

theorem family_average_age 
  (n : ℕ) 
  (youngest_age : ℕ) 
  (past_average : ℚ) : 
  n = 7 → 
  youngest_age = 5 → 
  past_average = 28 → 
  (((n - 1) * past_average + (n - 1) * youngest_age + youngest_age) / n : ℚ) = 209/7 := by
  sorry

end NUMINAMATH_CALUDE_family_average_age_l223_22396


namespace NUMINAMATH_CALUDE_intersection_locus_is_circle_l223_22352

/-- The locus of intersection points of two parameterized lines forms a circle -/
theorem intersection_locus_is_circle :
  ∀ (u x y : ℝ), 
    (3 * u - 4 * y + 2 = 0) →
    (2 * x - 3 * u * y - 4 = 0) →
    ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_is_circle_l223_22352


namespace NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l223_22375

theorem coefficient_x5_in_expansion :
  let n : ℕ := 36
  let k : ℕ := 5
  let coeff : ℤ := (n.choose k) * (-2 : ℤ) ^ (n - k)
  coeff = -8105545721856 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l223_22375


namespace NUMINAMATH_CALUDE_matinee_ticket_cost_l223_22310

/-- The cost of a matinee ticket in dollars -/
def matinee_cost : ℚ := 5

/-- The cost of an evening ticket in dollars -/
def evening_cost : ℚ := 7

/-- The cost of an opening night ticket in dollars -/
def opening_night_cost : ℚ := 10

/-- The cost of a bucket of popcorn in dollars -/
def popcorn_cost : ℚ := 10

/-- The number of matinee customers -/
def matinee_customers : ℕ := 32

/-- The number of evening customers -/
def evening_customers : ℕ := 40

/-- The number of opening night customers -/
def opening_night_customers : ℕ := 58

/-- The total revenue in dollars -/
def total_revenue : ℚ := 1670

theorem matinee_ticket_cost :
  matinee_cost * matinee_customers +
  evening_cost * evening_customers +
  opening_night_cost * opening_night_customers +
  popcorn_cost * ((matinee_customers + evening_customers + opening_night_customers) / 2) =
  total_revenue :=
sorry

end NUMINAMATH_CALUDE_matinee_ticket_cost_l223_22310


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l223_22365

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Theorem statement
theorem hyperbola_to_ellipse :
  ∀ x y : ℝ, hyperbola x y → 
  ∃ a b : ℝ, ellipse a b ∧ 
  (∀ c d : ℝ, hyperbola c 0 → (a = c ∨ a = -c) ∧ (b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l223_22365


namespace NUMINAMATH_CALUDE_quadratic_intersection_l223_22344

theorem quadratic_intersection
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hcd : c ≠ d) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := a * x^2 - b * x + d
  let x_intersect := (d - c) / (2 * b)
  let y_intersect := (a * (d - c)^2) / (4 * b^2) + (d + c) / 2
  ∃ (x y : ℝ), f x = g x ∧ f x = y ∧ x = x_intersect ∧ y = y_intersect :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_intersection_l223_22344


namespace NUMINAMATH_CALUDE_compound_interest_rate_l223_22324

theorem compound_interest_rate : ∃ (r : ℝ), 
  (1 + r)^2 = 7/6 ∧ 
  0.0800 < r ∧ 
  r < 0.0802 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l223_22324


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l223_22305

theorem complex_modulus_problem (z : ℂ) (h : z⁻¹ = 1 + I) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l223_22305


namespace NUMINAMATH_CALUDE_math_paths_count_l223_22355

/-- Represents the number of adjacent positions a letter can move to -/
def adjacent_positions : ℕ := 8

/-- Represents the length of the word "MATH" -/
def word_length : ℕ := 4

/-- Calculates the number of paths to spell "MATH" -/
def num_paths : ℕ := adjacent_positions ^ (word_length - 1)

/-- Theorem stating that the number of paths to spell "MATH" is 512 -/
theorem math_paths_count : num_paths = 512 := by sorry

end NUMINAMATH_CALUDE_math_paths_count_l223_22355


namespace NUMINAMATH_CALUDE_sock_cost_calculation_l223_22308

/-- The cost of each pair of socks that Niko bought --/
def sock_cost : ℝ := 2

/-- The number of pairs of socks Niko bought --/
def total_pairs : ℕ := 9

/-- The number of pairs Niko wants to sell with 25% profit --/
def pairs_with_percent_profit : ℕ := 4

/-- The number of pairs Niko wants to sell with $0.2 profit each --/
def pairs_with_fixed_profit : ℕ := 5

/-- The total profit Niko wants to make --/
def total_profit : ℝ := 3

/-- The profit percentage for the first group of socks --/
def profit_percentage : ℝ := 0.25

/-- The fixed profit amount for the second group of socks --/
def fixed_profit : ℝ := 0.2

theorem sock_cost_calculation :
  sock_cost * pairs_with_percent_profit * profit_percentage +
  pairs_with_fixed_profit * fixed_profit = total_profit ∧
  total_pairs = pairs_with_percent_profit + pairs_with_fixed_profit :=
by sorry

end NUMINAMATH_CALUDE_sock_cost_calculation_l223_22308


namespace NUMINAMATH_CALUDE_h_of_neg_one_eq_three_l223_22314

-- Define the functions
def f (x : ℝ) : ℝ := 3 * x + 6
def g (x : ℝ) : ℝ := x^3
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_of_neg_one_eq_three : h (-1) = 3 := by sorry

end NUMINAMATH_CALUDE_h_of_neg_one_eq_three_l223_22314


namespace NUMINAMATH_CALUDE_frogs_in_pond_a_l223_22329

theorem frogs_in_pond_a (frogs_b : ℕ) : 
  frogs_b + 2 * frogs_b = 48 → 2 * frogs_b = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_frogs_in_pond_a_l223_22329


namespace NUMINAMATH_CALUDE_floor_painting_rate_l223_22364

/-- Proves that the painting rate for a rectangular floor is 5 Rs/sq m given specific conditions -/
theorem floor_painting_rate (length : ℝ) (total_cost : ℝ) : 
  length = 13.416407864998739 →
  total_cost = 300 →
  ∃ (breadth : ℝ), 
    length = 3 * breadth ∧ 
    (5 : ℝ) = total_cost / (length * breadth) := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_rate_l223_22364


namespace NUMINAMATH_CALUDE_min_value_expression_l223_22307

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_prod : x * y * z = 2/3) :
  x^2 + 6*x*y + 18*y^2 + 12*y*z + 4*z^2 ≥ 18 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ * y₀ * z₀ = 2/3 ∧
    x₀^2 + 6*x₀*y₀ + 18*y₀^2 + 12*y₀*z₀ + 4*z₀^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l223_22307


namespace NUMINAMATH_CALUDE_sphere_surface_inequality_l223_22380

theorem sphere_surface_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  (x - y) * (y - z) * (x - z) ≤ 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_inequality_l223_22380


namespace NUMINAMATH_CALUDE_total_remaining_pictures_l223_22338

structure ColoringBook where
  purchaseDay : Nat
  totalPictures : Nat
  coloredPerDay : Nat

def daysOfColoring (book : ColoringBook) : Nat :=
  6 - book.purchaseDay

def picturesColored (book : ColoringBook) : Nat :=
  book.coloredPerDay * daysOfColoring book

def picturesRemaining (book : ColoringBook) : Nat :=
  book.totalPictures - picturesColored book

def books : List ColoringBook := [
  ⟨1, 24, 4⟩,
  ⟨2, 37, 5⟩,
  ⟨3, 50, 6⟩,
  ⟨4, 33, 3⟩,
  ⟨5, 44, 7⟩
]

theorem total_remaining_pictures :
  (books.map picturesRemaining).sum = 117 := by
  sorry

end NUMINAMATH_CALUDE_total_remaining_pictures_l223_22338


namespace NUMINAMATH_CALUDE_binary_representation_sqrt_theorem_l223_22304

/-- Given a positive integer d that is not a perfect square, s(n) denotes the number of digits 1 
    among the first n digits in the binary representation of √d -/
def s (d : ℕ+) (n : ℕ) : ℕ := sorry

/-- The theorem states that for a positive integer d that is not a perfect square, 
    there exists an integer A such that for all integers n ≥ A, 
    s(n) > √(2n) - 2 -/
theorem binary_representation_sqrt_theorem (d : ℕ+) 
    (h : ∀ (m : ℕ), m * m ≠ d) : 
    ∃ A : ℕ, ∀ n : ℕ, n ≥ A → s d n > Real.sqrt (2 * n) - 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_sqrt_theorem_l223_22304


namespace NUMINAMATH_CALUDE_expansion_theorem_l223_22326

theorem expansion_theorem (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expansion_theorem_l223_22326


namespace NUMINAMATH_CALUDE_sequence_properties_l223_22343

/-- Definition of an arithmetic sequence -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Definition of a geometric sequence -/
def geometric_seq (g : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, g (n + 1) = g n * q

/-- Main theorem -/
theorem sequence_properties
  (a g : ℕ → ℝ)
  (ha : arithmetic_seq a)
  (hg : geometric_seq g)
  (h1 : a 1 = 1)
  (h2 : g 1 = 1)
  (h3 : a 2 = g 2)
  (h4 : a 2 ≠ 1)
  (h5 : ∃ m : ℕ, m > 3 ∧ a m = g 3) :
  (∃ m : ℕ, m > 3 ∧
    (∃ d q : ℝ, d = m - 3 ∧ q = m - 2 ∧
      (∀ n : ℕ, a (n + 1) = a n + d) ∧
      (∀ n : ℕ, g (n + 1) = g n * q))) ∧
  (∃ k : ℕ, a k = g 4) ∧
  (∀ j : ℕ, ∃ k : ℕ, g (j + 1) = a k) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l223_22343


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l223_22327

theorem rectangular_prism_volume 
  (x y z : ℕ) 
  (h1 : x > 0 ∧ y > 0 ∧ z > 0)
  (h2 : 4 * (x + y + z - 3) = 40)
  (h3 : 2 * (x * y + x * z + y * z - 2 * (x + y + z - 3)) = 66) :
  x * y * z = 150 :=
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l223_22327


namespace NUMINAMATH_CALUDE_log_base_a1_13_l223_22381

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem log_base_a1_13 (a : ℕ → ℝ) :
  geometric_sequence a → a 9 = 13 → a 13 = 1 → Real.log 13 / Real.log (a 1) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_log_base_a1_13_l223_22381


namespace NUMINAMATH_CALUDE_replacement_paint_intensity_l223_22363

/-- Proves that the intensity of replacement paint is 25% given the original paint intensity,
    new paint intensity after mixing, and the fraction of original paint replaced. -/
theorem replacement_paint_intensity
  (original_intensity : ℝ)
  (new_intensity : ℝ)
  (replaced_fraction : ℝ)
  (h_original : original_intensity = 50)
  (h_new : new_intensity = 40)
  (h_replaced : replaced_fraction = 0.4)
  : (1 - replaced_fraction) * original_intensity + replaced_fraction * 25 = new_intensity :=
by sorry


end NUMINAMATH_CALUDE_replacement_paint_intensity_l223_22363


namespace NUMINAMATH_CALUDE_theater_capacity_filled_l223_22331

theorem theater_capacity_filled (seats : ℕ) (ticket_price : ℕ) (performances : ℕ) (total_revenue : ℕ) :
  seats = 400 →
  ticket_price = 30 →
  performances = 3 →
  total_revenue = 28800 →
  (total_revenue / ticket_price) / (seats * performances) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_theater_capacity_filled_l223_22331


namespace NUMINAMATH_CALUDE_acid_mixing_problem_l223_22393

/-- The largest integer concentration percentage achievable in the acid mixing problem -/
def largest_concentration : ℕ := 76

theorem acid_mixing_problem :
  ∀ r : ℕ,
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ (2.8 + 0.9 * x) / (4 + x) = r / 100) →
    r ≤ largest_concentration :=
by sorry

end NUMINAMATH_CALUDE_acid_mixing_problem_l223_22393


namespace NUMINAMATH_CALUDE_inequality_system_solution_l223_22390

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 2 < x ∧ (1/3) * x < -2) ↔ x < -6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l223_22390


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l223_22330

theorem reciprocal_of_sum : (1 / (1/3 + 1/5) : ℚ) = 15/8 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l223_22330


namespace NUMINAMATH_CALUDE_marble_problem_l223_22369

theorem marble_problem (x : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ)
  (h1 : angela = x)
  (h2 : brian = 3 * x)
  (h3 : caden = 2 * brian)
  (h4 : daryl = 4 * caden)
  (h5 : angela + brian + caden + daryl = 144) :
  x = 72 / 17 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l223_22369


namespace NUMINAMATH_CALUDE_min_unsuccessful_placements_l223_22328

/-- Represents a cell in the grid -/
inductive Cell
| Plus : Cell
| Minus : Cell

/-- Represents an 8x8 grid -/
def Grid := Fin 8 → Fin 8 → Cell

/-- Represents a T-shaped figure -/
structure TShape where
  row : Fin 8
  col : Fin 8
  orientation : Bool  -- True for horizontal, False for vertical

/-- Calculates the sum of a T-shape on the grid -/
def tShapeSum (g : Grid) (t : TShape) : Int :=
  sorry

/-- Counts the number of unsuccessful T-shape placements -/
def countUnsuccessful (g : Grid) : Nat :=
  sorry

/-- Theorem: The minimum number of unsuccessful T-shape placements is 132 -/
theorem min_unsuccessful_placements :
  ∀ g : Grid, countUnsuccessful g ≥ 132 :=
sorry

end NUMINAMATH_CALUDE_min_unsuccessful_placements_l223_22328


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l223_22316

theorem trigonometric_equation_solution :
  ∀ t : ℝ, 
    (2 * (Real.cos (2 * t))^6 - (Real.cos (2 * t))^4 + 1.5 * (Real.sin (4 * t))^2 - 3 * (Real.sin (2 * t))^2 = 0) ↔ 
    (∃ k : ℤ, t = (Real.pi / 8) * (2 * ↑k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l223_22316


namespace NUMINAMATH_CALUDE_find_number_l223_22383

theorem find_number : ∃ x : ℤ, x - 29 + 64 = 76 ∧ x = 41 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l223_22383


namespace NUMINAMATH_CALUDE_root_product_theorem_l223_22372

theorem root_product_theorem (x₁ x₂ x₃ : ℝ) : 
  (∃ (a : ℝ), a > 0 ∧ a^2 = 2023 ∧
    (a * x₁^3 - (2*a^2 + 1)*x₁^2 + 3 = 0) ∧
    (a * x₂^3 - (2*a^2 + 1)*x₂^2 + 3 = 0) ∧
    (a * x₃^3 - (2*a^2 + 1)*x₃^2 + 3 = 0) ∧
    x₁ < x₂ ∧ x₂ < x₃) →
  x₂ * (x₁ + x₃) = 3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l223_22372


namespace NUMINAMATH_CALUDE_line_relationships_l223_22318

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a simplified representation
  dummy : Unit

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop := sorry

theorem line_relationships (a b c : Line3D) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (¬ ∀ (a b c : Line3D), perpendicular a b → perpendicular a c → parallel b c) ∧ 
  (¬ ∀ (a b c : Line3D), perpendicular a b → perpendicular a c → perpendicular b c) ∧
  (∀ (a b c : Line3D), parallel a b → perpendicular b c → perpendicular a c) := by
  sorry

end NUMINAMATH_CALUDE_line_relationships_l223_22318


namespace NUMINAMATH_CALUDE_flower_arrangement_count_l223_22387

/-- The number of roses available for selection. -/
def num_roses : ℕ := 4

/-- The number of tulips available for selection. -/
def num_tulips : ℕ := 3

/-- The number of flower arrangements where exactly one of the roses or tulips is the same. -/
def arrangements_with_one_same : ℕ := 
  (num_roses * (num_tulips * (num_tulips - 1))) + 
  (num_tulips * (num_roses * (num_roses - 1)))

/-- Theorem stating that the number of flower arrangements where exactly one of the roses or tulips is the same is 60. -/
theorem flower_arrangement_count : arrangements_with_one_same = 60 := by
  sorry

end NUMINAMATH_CALUDE_flower_arrangement_count_l223_22387


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l223_22333

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  yIntercept : ℝ

/-- Checks if a point is in the first quadrant -/
def isFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

/-- The main theorem -/
theorem tangent_line_y_intercept :
  ∀ (l : TangentLine),
    l.circle1 = { center := (3, 0), radius := 3 } →
    l.circle2 = { center := (7, 0), radius := 2 } →
    (∃ (p1 p2 : ℝ × ℝ), isFirstQuadrant p1 ∧ isFirstQuadrant p2) →
    l.yIntercept = 24 * Real.sqrt 55 / 55 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_y_intercept_l223_22333


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l223_22339

theorem quadratic_roots_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∀ x : ℝ, x^2 + a*x + b = 0 → (2*x)^2 + b*(2*x) + c = 0) →
  a / c = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l223_22339


namespace NUMINAMATH_CALUDE_power_calculation_l223_22358

theorem power_calculation : (16^6 * 8^3) / 4^11 = 2048 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l223_22358


namespace NUMINAMATH_CALUDE_sequence_sum_log_l223_22320

theorem sequence_sum_log (a : ℕ → ℝ) :
  (∀ n, Real.log (a (n + 1)) = 1 + Real.log (a n)) →
  a 1 + a 2 + a 3 = 10 →
  Real.log (a 4 + a 5 + a 6) = 4 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_log_l223_22320


namespace NUMINAMATH_CALUDE_delaney_departure_time_l223_22371

structure TimeInMinutes where
  minutes : ℕ

def bus_departure : TimeInMinutes := ⟨480⟩  -- 8:00 a.m. in minutes since midnight
def travel_time : ℕ := 30
def late_by : ℕ := 20

theorem delaney_departure_time :
  ∃ (departure_time : TimeInMinutes),
    departure_time.minutes + travel_time = bus_departure.minutes + late_by ∧
    departure_time.minutes = 470  -- 7:50 a.m. in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_delaney_departure_time_l223_22371


namespace NUMINAMATH_CALUDE_function_properties_l223_22382

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x + m

theorem function_properties (m : ℝ) :
  (∀ x > 0, f m x ≤ 0) →
  m = 1 ∧ ∀ a b, 0 < a → a < b → (f m b - f m a) / (b - a) < 1 / (a * (a + 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l223_22382


namespace NUMINAMATH_CALUDE_total_distance_driven_l223_22384

/-- The total distance driven by Renaldo and Ernesto -/
def total_distance (renaldo_distance : ℝ) (ernesto_distance : ℝ) : ℝ :=
  renaldo_distance + ernesto_distance

/-- Theorem stating the total distance driven by Renaldo and Ernesto -/
theorem total_distance_driven :
  let renaldo_distance : ℝ := 15
  let ernesto_distance : ℝ := (1/3 * renaldo_distance) + 7
  total_distance renaldo_distance ernesto_distance = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_driven_l223_22384


namespace NUMINAMATH_CALUDE_log_expression_simplification_l223_22351

theorem log_expression_simplification :
  (2 * (Real.log 3 / Real.log 4) + Real.log 3 / Real.log 8) *
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l223_22351


namespace NUMINAMATH_CALUDE_f_of_g_5_l223_22335

def g (x : ℝ) : ℝ := 4 * x + 9

def f (x : ℝ) : ℝ := 6 * x - 11

theorem f_of_g_5 : f (g 5) = 163 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_5_l223_22335


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l223_22336

def S : Set ℝ := {x | x^2 + 2*x = 0}
def T : Set ℝ := {x | x^2 - 2*x = 0}

theorem intersection_of_S_and_T : S ∩ T = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l223_22336


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l223_22394

/-- Given a quadratic function f(x) = ax^2 + bx + c with vertex (3, 7) and one x-intercept at (-2, 0),
    the x-coordinate of the other x-intercept is 8. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 7 + a * (x - 3)^2) →  -- Vertex form of quadratic with vertex (3, 7)
  (a * (-2)^2 + b * (-2) + c = 0) →                 -- (-2, 0) is an x-intercept
  ∃ x, x ≠ -2 ∧ a * x^2 + b * x + c = 0 ∧ x = 8 :=  -- Other x-intercept exists and equals 8
by sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l223_22394


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l223_22315

/-- Two real numbers are inversely proportional -/
def InverseProportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : InverseProportion x₁ y₁)
  (h2 : InverseProportion x₂ y₂)
  (h3 : x₁ = 40)
  (h4 : y₁ = 5)
  (h5 : y₂ = 10) :
  x₂ = 20 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l223_22315


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l223_22325

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l223_22325


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_16_l223_22353

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_16 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → digit_sum n = 16 → n ≤ 952 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_16_l223_22353


namespace NUMINAMATH_CALUDE_sequence_properties_l223_22350

-- Define the sequence a_n and its sum S_n
def a (n : ℕ) : ℚ := sorry

def S (n : ℕ) : ℚ := sorry

-- Define the conditions
axiom S_def (n : ℕ) : S n = n * (6 + a n) / 2

axiom a_4 : a 4 = 12

-- Define b_n and its sum T_n
def b (n : ℕ) : ℚ := 1 / (n * a n)

def T (n : ℕ) : ℚ := sorry

-- Theorem to prove
theorem sequence_properties :
  (∀ n : ℕ, a n = 2 * n + 4) ∧
  (∀ n : ℕ, T n = 3/8 - (2*n+3)/(4*(n+1)*(n+2))) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l223_22350


namespace NUMINAMATH_CALUDE_tangent_parallel_points_solution_points_l223_22321

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + x - 1

-- Define the slope of the line parallel to 4x - y = 0
def parallel_slope : ℝ := 4

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  ∀ x : ℝ, curve_derivative x = parallel_slope ↔ x = 1 ∨ x = -1 :=
sorry

theorem solution_points :
  ∀ x y : ℝ, y = curve x ∧ curve_derivative x = parallel_slope ↔ 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_solution_points_l223_22321


namespace NUMINAMATH_CALUDE_determinant_one_l223_22309

-- Define the property that for all m and n, there exist h and k satisfying the equations
def satisfies_equations (a b c d : ℤ) : Prop :=
  ∀ m n : ℤ, ∃ h k : ℤ, a * h + b * k = m ∧ c * h + d * k = n

-- State the theorem
theorem determinant_one (a b c d : ℤ) (h : satisfies_equations a b c d) : |a * d - b * c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_determinant_one_l223_22309


namespace NUMINAMATH_CALUDE_train_speed_proof_l223_22359

/-- Proves that the speed of each train is 54 km/hr given the problem conditions -/
theorem train_speed_proof (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 120) (h2 : crossing_time = 8) : 
  let relative_speed := (2 * train_length) / crossing_time
  let train_speed_ms := relative_speed / 2
  let train_speed_kmh := train_speed_ms * 3.6
  train_speed_kmh = 54 := by
sorry


end NUMINAMATH_CALUDE_train_speed_proof_l223_22359


namespace NUMINAMATH_CALUDE_fred_balloon_count_l223_22356

theorem fred_balloon_count (total sam dan : ℕ) (h1 : total = 72) (h2 : sam = 46) (h3 : dan = 16) :
  total - (sam + dan) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fred_balloon_count_l223_22356


namespace NUMINAMATH_CALUDE_absent_days_calculation_l223_22322

/-- Calculates the number of days absent given the total days, daily wage, daily fine, and total earnings -/
def days_absent (total_days : ℕ) (daily_wage : ℕ) (daily_fine : ℕ) (total_earnings : ℕ) : ℕ :=
  total_days - (total_earnings + total_days * daily_fine) / (daily_wage + daily_fine)

theorem absent_days_calculation :
  days_absent 30 10 2 216 = 7 := by
  sorry

end NUMINAMATH_CALUDE_absent_days_calculation_l223_22322


namespace NUMINAMATH_CALUDE_geometric_mean_problem_l223_22370

theorem geometric_mean_problem (k : ℝ) :
  (k + 9) * (6 - k) = (2 * k)^2 → k = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_mean_problem_l223_22370


namespace NUMINAMATH_CALUDE_perpendicular_lines_l223_22386

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_lines 
  (a b c d : Line) (α β : Plane)
  (h1 : perp a b)
  (h2 : perp_line_plane a α)
  (h3 : perp_line_plane c α) :
  perp c b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l223_22386
