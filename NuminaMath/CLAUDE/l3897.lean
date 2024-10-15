import Mathlib

namespace NUMINAMATH_CALUDE_poly_arrangement_l3897_389784

/-- The original polynomial -/
def original_poly (x y : ℝ) : ℝ := -2 * x^3 * y + 4 * x * y^3 + 1 - 3 * x^2 * y^2

/-- The polynomial arranged in descending order of y -/
def arranged_poly (x y : ℝ) : ℝ := 4 * x * y^3 - 3 * x^2 * y^2 - 2 * x^3 * y + 1

/-- Theorem stating that the original polynomial is equal to the arranged polynomial -/
theorem poly_arrangement (x y : ℝ) : original_poly x y = arranged_poly x y := by
  sorry

end NUMINAMATH_CALUDE_poly_arrangement_l3897_389784


namespace NUMINAMATH_CALUDE_intersecting_lines_theorem_l3897_389790

/-- Given two lines that intersect at (-7, 9), prove that the line passing through their coefficients as points has the equation -7x + 9y = 1 -/
theorem intersecting_lines_theorem (A₁ B₁ A₂ B₂ : ℝ) : 
  (A₁ * (-7) + B₁ * 9 = 1) →  -- First line passes through (-7, 9)
  (A₂ * (-7) + B₂ * 9 = 1) →  -- Second line passes through (-7, 9)
  ∃ (k : ℝ), k * (-7) * (A₂ - A₁) = 9 * (B₂ - B₁) ∧   -- Points (A₁, B₁) and (A₂, B₂) satisfy -7x + 9y = k
             k = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_lines_theorem_l3897_389790


namespace NUMINAMATH_CALUDE_probability_consecutive_numbers_l3897_389738

/-- The total number of lottery numbers --/
def total_numbers : ℕ := 90

/-- The number of drawn lottery numbers --/
def drawn_numbers : ℕ := 5

/-- The set of all possible combinations of drawn numbers --/
def all_combinations : ℕ := Nat.choose total_numbers drawn_numbers

/-- The set of combinations with at least one pair of consecutive numbers --/
def consecutive_combinations : ℕ := 9122966

/-- The probability of drawing at least one pair of consecutive numbers --/
theorem probability_consecutive_numbers :
  (consecutive_combinations : ℚ) / all_combinations = 9122966 / 43949268 := by
  sorry

end NUMINAMATH_CALUDE_probability_consecutive_numbers_l3897_389738


namespace NUMINAMATH_CALUDE_sports_field_dimensions_l3897_389759

/-- The dimensions of a rectangular sports field with a surrounding path -/
theorem sports_field_dimensions (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ x : ℝ,
    x > 0 ∧
    x * (x + b) = (x + 2*a) * (x + b + 2*a) - x * (x + b) ∧
    x = (Real.sqrt (b^2 + 32*a^2) - b + 4*a) / 2 ∧
    x + b = (Real.sqrt (b^2 + 32*a^2) + b + 4*a) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sports_field_dimensions_l3897_389759


namespace NUMINAMATH_CALUDE_solve_for_d_l3897_389710

theorem solve_for_d (a c d n : ℝ) (h : n = (c * d * a) / (a - d)) :
  d = (n * a) / (c * d + n) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_d_l3897_389710


namespace NUMINAMATH_CALUDE_special_function_form_l3897_389785

/-- A positive continuous function satisfying the given inequality. -/
structure SpecialFunction where
  f : ℝ → ℝ
  continuous : Continuous f
  positive : ∀ x, f x > 0
  inequality : ∀ x y, f x - f y ≥ (x - y) * f ((x + y) / 2) * a
  a : ℝ

/-- The theorem stating that any function satisfying the SpecialFunction properties
    must be of the form c * exp(a * x) for some positive c. -/
theorem special_function_form (sf : SpecialFunction) :
  ∃ c : ℝ, c > 0 ∧ ∀ x, sf.f x = c * Real.exp (sf.a * x) := by
  sorry

end NUMINAMATH_CALUDE_special_function_form_l3897_389785


namespace NUMINAMATH_CALUDE_cos_160_sin_10_minus_sin_20_cos_10_l3897_389727

theorem cos_160_sin_10_minus_sin_20_cos_10 :
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) -
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_160_sin_10_minus_sin_20_cos_10_l3897_389727


namespace NUMINAMATH_CALUDE_test_retake_count_l3897_389794

theorem test_retake_count (total : ℕ) (passed : ℕ) (retake : ℕ) : 
  total = 2500 → passed = 375 → retake = total - passed → retake = 2125 := by
  sorry

end NUMINAMATH_CALUDE_test_retake_count_l3897_389794


namespace NUMINAMATH_CALUDE_even_function_product_nonnegative_l3897_389730

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem even_function_product_nonnegative
  (f : ℝ → ℝ) (h : is_even_function f) :
  ∀ x : ℝ, f x * f (-x) ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_even_function_product_nonnegative_l3897_389730


namespace NUMINAMATH_CALUDE_quilt_cost_theorem_l3897_389777

def quilt_width : ℕ := 16
def quilt_length : ℕ := 20
def patch_area : ℕ := 4
def initial_patch_cost : ℕ := 10
def initial_patch_count : ℕ := 10

def total_quilt_area : ℕ := quilt_width * quilt_length
def total_patches : ℕ := total_quilt_area / patch_area
def discounted_patch_cost : ℕ := initial_patch_cost / 2
def discounted_patches : ℕ := total_patches - initial_patch_count

def total_cost : ℕ := initial_patch_count * initial_patch_cost + discounted_patches * discounted_patch_cost

theorem quilt_cost_theorem : total_cost = 450 := by
  sorry

end NUMINAMATH_CALUDE_quilt_cost_theorem_l3897_389777


namespace NUMINAMATH_CALUDE_snow_probability_l3897_389735

/-- The probability of no snow on each of the first five days -/
def no_snow_prob (n : ℕ) : ℚ :=
  if n ≤ 5 then (n + 1) / (n + 2) else 7/8

/-- The probability of snow on at least one day out of seven -/
def snow_prob : ℚ :=
  1 - (no_snow_prob 1 * no_snow_prob 2 * no_snow_prob 3 * no_snow_prob 4 * no_snow_prob 5 * no_snow_prob 6 * no_snow_prob 7)

theorem snow_probability : snow_prob = 139/384 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l3897_389735


namespace NUMINAMATH_CALUDE_f_value_theorem_l3897_389787

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_value_theorem (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_odd : is_odd f)
  (h_def : ∀ x, 0 < x → x < 1 → f x = 1 / x) :
  f (-5/2) + f 0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_theorem_l3897_389787


namespace NUMINAMATH_CALUDE_only_real_number_line_bijection_is_correct_l3897_389747

-- Define the property of having a square root
def has_square_root (x : ℝ) : Prop := ∃ y : ℝ, y * y = x

-- Define the property of being irrational
def is_irrational (x : ℝ) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

-- Define the property of cube root being equal to itself
def cube_root_equals_self (x : ℝ) : Prop := x * x * x = x

-- Define the property of having no square root
def has_no_square_root (x : ℝ) : Prop := ¬ (∃ y : ℝ, y * y = x)

-- Define the one-to-one correspondence between real numbers and points on a line
def real_number_line_bijection : Prop := 
  ∃ f : ℝ → ℝ, Function.Bijective f ∧ (∀ x : ℝ, f x = x)

-- Define the property that the difference of two irrationals is irrational
def irrational_diff_is_irrational : Prop := 
  ∀ x y : ℝ, is_irrational x → is_irrational y → is_irrational (x - y)

theorem only_real_number_line_bijection_is_correct : 
  (¬ (∀ x : ℝ, has_square_root x → is_irrational x)) ∧
  (¬ (∀ x : ℝ, cube_root_equals_self x → (x = 0 ∨ x = 1))) ∧
  (¬ (∀ a : ℝ, has_no_square_root (-a))) ∧
  real_number_line_bijection ∧
  (¬ irrational_diff_is_irrational) :=
by sorry

end NUMINAMATH_CALUDE_only_real_number_line_bijection_is_correct_l3897_389747


namespace NUMINAMATH_CALUDE_w_value_l3897_389791

def cubic_poly (x : ℝ) := x^3 - 4*x^2 + 2*x + 1

def second_poly (x u v w : ℝ) := x^3 + u*x^2 + v*x + w

theorem w_value (p q r u v w : ℝ) :
  cubic_poly p = 0 ∧ cubic_poly q = 0 ∧ cubic_poly r = 0 →
  second_poly (p + q) u v w = 0 ∧ second_poly (q + r) u v w = 0 ∧ second_poly (r + p) u v w = 0 →
  w = 15 := by sorry

end NUMINAMATH_CALUDE_w_value_l3897_389791


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3897_389715

theorem sqrt_equation_solution :
  ∃ x : ℝ, x = 196 ∧ Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3897_389715


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l3897_389756

/-- The cost of a pen in rupees -/
def pen_cost : ℝ := sorry

/-- The cost of a pencil in rupees -/
def pencil_cost : ℝ := sorry

/-- The cost ratio of a pen to a pencil -/
def cost_ratio : ℝ := 5

/-- The total cost of 3 pens and 5 pencils in rupees -/
def total_cost : ℝ := 240

/-- The number of pens in a dozen -/
def dozen : ℕ := 12

theorem cost_of_dozen_pens :
  pen_cost = pencil_cost * cost_ratio ∧
  3 * pen_cost + 5 * pencil_cost = total_cost →
  dozen * pen_cost = 720 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l3897_389756


namespace NUMINAMATH_CALUDE_alice_overall_score_approx_80_percent_l3897_389782

/-- Represents a test with a number of questions and a score percentage -/
structure Test where
  questions : ℕ
  score : ℚ
  scoreInRange : 0 ≤ score ∧ score ≤ 1

/-- Alice's test results -/
def aliceTests : List Test := [
  ⟨20, 3/4, by norm_num⟩,
  ⟨50, 17/20, by norm_num⟩,
  ⟨30, 3/5, by norm_num⟩,
  ⟨40, 9/10, by norm_num⟩
]

/-- The total number of questions Alice answered correctly -/
def totalCorrect : ℚ :=
  aliceTests.foldl (fun acc test => acc + test.questions * test.score) 0

/-- The total number of questions across all tests -/
def totalQuestions : ℕ :=
  aliceTests.foldl (fun acc test => acc + test.questions) 0

/-- Alice's overall score as a percentage -/
def overallScore : ℚ := totalCorrect / totalQuestions

theorem alice_overall_score_approx_80_percent :
  abs (overallScore - 4/5) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_alice_overall_score_approx_80_percent_l3897_389782


namespace NUMINAMATH_CALUDE_club_selection_count_l3897_389764

theorem club_selection_count (n : ℕ) (h : n = 18) : 
  n * (Nat.choose (n - 1) 2) = 2448 := by
  sorry

end NUMINAMATH_CALUDE_club_selection_count_l3897_389764


namespace NUMINAMATH_CALUDE_wrench_force_calculation_l3897_389714

/-- Represents the force required to loosen a bolt with a wrench of a given length -/
structure WrenchForce where
  length : ℝ
  force : ℝ

/-- The inverse relationship between force and wrench length -/
def inverseProportion (w1 w2 : WrenchForce) : Prop :=
  w1.force * w1.length = w2.force * w2.length

theorem wrench_force_calculation 
  (w1 w2 : WrenchForce)
  (h1 : w1.length = 12)
  (h2 : w1.force = 300)
  (h3 : w2.length = 18)
  (h4 : inverseProportion w1 w2) :
  w2.force = 200 := by
  sorry

end NUMINAMATH_CALUDE_wrench_force_calculation_l3897_389714


namespace NUMINAMATH_CALUDE_marcus_walking_speed_l3897_389720

/-- Calculates Marcus's walking speed given the conditions of his dog care routine -/
theorem marcus_walking_speed (bath_time : ℝ) (total_time : ℝ) (walk_distance : ℝ) : 
  bath_time = 20 →
  total_time = 60 →
  walk_distance = 3 →
  (walk_distance / (total_time - bath_time - bath_time / 2)) * 60 = 6 := by
sorry

end NUMINAMATH_CALUDE_marcus_walking_speed_l3897_389720


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l3897_389718

theorem largest_prime_factor_of_3913 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3913 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3913 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l3897_389718


namespace NUMINAMATH_CALUDE_largest_solution_and_ratio_l3897_389751

theorem largest_solution_and_ratio : ∃ (a b c d : ℤ),
  let x : ℝ := (a + b * Real.sqrt c) / d
  (7 * x) / 9 + 2 = 4 / x ∧
  (∀ (a' b' c' d' : ℤ), 
    let x' : ℝ := (a' + b' * Real.sqrt c') / d'
    (7 * x') / 9 + 2 = 4 / x' → x' ≤ x) ∧
  x = (-9 + 3 * Real.sqrt 111) / 7 ∧
  a * c * d / b = -2313 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_and_ratio_l3897_389751


namespace NUMINAMATH_CALUDE_interval_intersection_l3897_389796

theorem interval_intersection (x : ℝ) : 
  (2 < 3*x ∧ 3*x < 3) ∧ (2 < 4*x ∧ 4*x < 3) ↔ (2/3 < x ∧ x < 3/4) := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l3897_389796


namespace NUMINAMATH_CALUDE_max_candy_pieces_l3897_389716

theorem max_candy_pieces (n : ℕ) (mean : ℚ) (min_pieces : ℕ) 
  (h1 : n = 40)
  (h2 : mean = 4)
  (h3 : min_pieces = 2) :
  ∃ (max_pieces : ℕ), max_pieces = 82 ∧ 
  (∀ (student_pieces : List ℕ), 
    student_pieces.length = n ∧ 
    (∀ x ∈ student_pieces, x ≥ min_pieces) ∧
    (student_pieces.sum / n : ℚ) = mean →
    ∀ x ∈ student_pieces, x ≤ max_pieces) :=
by sorry

end NUMINAMATH_CALUDE_max_candy_pieces_l3897_389716


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3897_389717

theorem quadratic_one_root (n : ℝ) : 
  (∃! x : ℝ, x^2 - 6*n*x - 9*n = 0) ∧ n ≥ 0 → n = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3897_389717


namespace NUMINAMATH_CALUDE_tara_megan_money_difference_l3897_389743

/-- The problem of determining how much more money Tara has than Megan. -/
theorem tara_megan_money_difference
  (scooter_cost : ℕ)
  (tara_money : ℕ)
  (megan_money : ℕ)
  (h1 : scooter_cost = 26)
  (h2 : tara_money > megan_money)
  (h3 : tara_money + megan_money = scooter_cost)
  (h4 : tara_money = 15) :
  tara_money - megan_money = 4 := by
  sorry

end NUMINAMATH_CALUDE_tara_megan_money_difference_l3897_389743


namespace NUMINAMATH_CALUDE_y_axis_symmetry_sum_l3897_389719

/-- Given a point M(a, b+3, 2c+1) with y-axis symmetric point M'(-4, -2, 15), prove a+b+c = -9 -/
theorem y_axis_symmetry_sum (a b c : ℝ) : 
  (a = 4) ∧ (b + 3 = -2) ∧ (2 * c + 1 = 15) → a + b + c = -9 := by
  sorry

end NUMINAMATH_CALUDE_y_axis_symmetry_sum_l3897_389719


namespace NUMINAMATH_CALUDE_no_sum_equal_powers_l3897_389795

theorem no_sum_equal_powers : ¬∃ (n m : ℕ), n * (n + 1) / 2 = 2^m + 3^m := by
  sorry

end NUMINAMATH_CALUDE_no_sum_equal_powers_l3897_389795


namespace NUMINAMATH_CALUDE_distribute_10_balls_3_boxes_l3897_389755

/-- The number of ways to distribute n identical balls into k boxes, where each box i must contain at least i balls. -/
def distributeWithMinimum (n : ℕ) (k : ℕ) : ℕ :=
  let remainingBalls := n - (k * (k + 1) / 2)
  Nat.choose (remainingBalls + k - 1) (k - 1)

/-- Theorem stating that there are 15 ways to distribute 10 identical balls into 3 boxes with the given conditions. -/
theorem distribute_10_balls_3_boxes : distributeWithMinimum 10 3 = 15 := by
  sorry

#eval distributeWithMinimum 10 3

end NUMINAMATH_CALUDE_distribute_10_balls_3_boxes_l3897_389755


namespace NUMINAMATH_CALUDE_min_cost_is_84_l3897_389702

/-- Represents a salon with prices for haircut, facial cleaning, and nails --/
structure Salon where
  haircut : ℕ
  facial : ℕ
  nails : ℕ

/-- Calculates the total cost for a salon --/
def totalCost (s : Salon) : ℕ := s.haircut + s.facial + s.nails

/-- The three salons with their respective prices --/
def gustranSalon : Salon := ⟨45, 22, 30⟩
def barbarasShop : Salon := ⟨30, 28, 40⟩
def fancySalon : Salon := ⟨34, 30, 20⟩

/-- Theorem stating that the minimum total cost among the three salons is 84 --/
theorem min_cost_is_84 : 
  min (totalCost gustranSalon) (min (totalCost barbarasShop) (totalCost fancySalon)) = 84 := by
  sorry


end NUMINAMATH_CALUDE_min_cost_is_84_l3897_389702


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_864_l3897_389766

theorem sum_of_roots_equals_864 
  (p q r s : ℝ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h_eq1 : ∀ x, x^2 - 8*p*x - 12*q = 0 ↔ x = r ∨ x = s)
  (h_eq2 : ∀ x, x^2 - 8*r*x - 12*s = 0 ↔ x = p ∨ x = q) :
  p + q + r + s = 864 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_864_l3897_389766


namespace NUMINAMATH_CALUDE_all_days_equal_availability_l3897_389781

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

-- Define the team members
inductive Member
| Alice
| Bob
| Charlie
| Diana

-- Define a function to represent availability
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Monday => false
  | Member.Alice, Day.Thursday => false
  | Member.Bob, Day.Tuesday => false
  | Member.Bob, Day.Wednesday => false
  | Member.Bob, Day.Friday => false
  | Member.Charlie, Day.Wednesday => false
  | Member.Charlie, Day.Thursday => false
  | Member.Charlie, Day.Friday => false
  | Member.Diana, Day.Monday => false
  | Member.Diana, Day.Tuesday => false
  | _, _ => true

-- Count available members for a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => isAvailable m d) [Member.Alice, Member.Bob, Member.Charlie, Member.Diana]).length

-- Theorem: All days have equal availability
theorem all_days_equal_availability :
  ∀ d1 d2 : Day, availableCount d1 = availableCount d2 :=
sorry

end NUMINAMATH_CALUDE_all_days_equal_availability_l3897_389781


namespace NUMINAMATH_CALUDE_existence_of_z_l3897_389793

theorem existence_of_z (a p x y : ℕ) (hp : Prime p) (hx : x > 0) (hy : y > 0) (ha : a > 0)
  (hx41 : ∃ n : ℕ, x^41 = a + n*p) (hy49 : ∃ n : ℕ, y^49 = a + n*p) :
  ∃ (z : ℕ), z > 0 ∧ ∃ (n : ℕ), z^2009 = a + n*p :=
by sorry

end NUMINAMATH_CALUDE_existence_of_z_l3897_389793


namespace NUMINAMATH_CALUDE_boys_combined_average_l3897_389797

/-- Represents a high school with average scores for boys, girls, and combined --/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Theorem stating that given the conditions, the average score for boys across two schools is 70.8 --/
theorem boys_combined_average (chs dhs : School)
  (h_chs_boys : chs.boys_avg = 68)
  (h_chs_girls : chs.girls_avg = 73)
  (h_chs_combined : chs.combined_avg = 70)
  (h_dhs_boys : dhs.boys_avg = 75)
  (h_dhs_girls : dhs.girls_avg = 85)
  (h_dhs_combined : dhs.combined_avg = 80) :
  ∃ (c d : ℝ), c > 0 ∧ d > 0 ∧
  (c * chs.boys_avg + d * dhs.boys_avg) / (c + d) = 70.8 := by
  sorry


end NUMINAMATH_CALUDE_boys_combined_average_l3897_389797


namespace NUMINAMATH_CALUDE_composition_ratio_theorem_l3897_389749

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composition_ratio_theorem :
  (f (g (f 2))) / (g (f (g 2))) = 41 / 7 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_theorem_l3897_389749


namespace NUMINAMATH_CALUDE_spade_equation_solution_l3897_389763

/-- The spade operation -/
def spade_op (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 7

/-- Theorem stating that if X spade 5 = 23, then X = 7.75 -/
theorem spade_equation_solution :
  ∀ X : ℝ, spade_op X 5 = 23 → X = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_spade_equation_solution_l3897_389763


namespace NUMINAMATH_CALUDE_girls_in_college_l3897_389705

theorem girls_in_college (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : boys + girls = 520) : girls = 200 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_college_l3897_389705


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3897_389771

/-- The sum of the infinite series ∑(k=1 to ∞) k^3 / 2^k is equal to 26. -/
theorem infinite_series_sum : ∑' k : ℕ, (k^3 : ℝ) / 2^k = 26 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3897_389771


namespace NUMINAMATH_CALUDE_total_arrangements_with_at_least_one_girl_l3897_389712

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def choose (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def num_people : ℕ := num_boys + num_girls
def num_selected : ℕ := 3
def num_tasks : ℕ := 3

theorem total_arrangements_with_at_least_one_girl : 
  (choose num_people num_selected - choose num_boys num_selected) * factorial num_tasks = 186 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_with_at_least_one_girl_l3897_389712


namespace NUMINAMATH_CALUDE_p_plus_q_values_l3897_389746

theorem p_plus_q_values (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 81*p - 162 = 0)
  (hq : 4*q^3 - 24*q^2 + 45*q - 27 = 0) :
  (p + q = 8) ∨ (p + q = 8 + 6*Real.sqrt 3) ∨ (p + q = 8 - 6*Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_p_plus_q_values_l3897_389746


namespace NUMINAMATH_CALUDE_apple_selling_price_l3897_389742

/-- The selling price of an apple given its cost price and loss ratio -/
def selling_price (cost_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  cost_price * (1 - loss_ratio)

/-- Theorem stating the selling price of an apple with given conditions -/
theorem apple_selling_price :
  let cost_price : ℚ := 20
  let loss_ratio : ℚ := 1/6
  selling_price cost_price loss_ratio = 50/3 := by
sorry

end NUMINAMATH_CALUDE_apple_selling_price_l3897_389742


namespace NUMINAMATH_CALUDE_tetrahedron_non_coplanar_selections_l3897_389768

/-- The number of ways to select 4 non-coplanar points from a tetrahedron -/
def nonCoplanarSelections : ℕ := 141

/-- Total number of points on the tetrahedron -/
def totalPoints : ℕ := 10

/-- Number of vertices of the tetrahedron -/
def vertices : ℕ := 4

/-- Number of midpoints of edges -/
def midpoints : ℕ := 6

/-- Number of points to be selected -/
def selectPoints : ℕ := 4

/-- Theorem stating that the number of ways to select 4 non-coplanar points
    from 10 points on a tetrahedron (4 vertices and 6 midpoints of edges) is 141 -/
theorem tetrahedron_non_coplanar_selections :
  totalPoints = vertices + midpoints ∧
  nonCoplanarSelections = 141 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_non_coplanar_selections_l3897_389768


namespace NUMINAMATH_CALUDE_total_matches_proof_l3897_389726

/-- Represents the number of matches for a team -/
structure MatchRecord where
  wins : ℕ
  draws : ℕ
  losses : ℕ

/-- Calculate the total number of matches played by a team -/
def totalMatches (record : MatchRecord) : ℕ :=
  record.wins + record.draws + record.losses

theorem total_matches_proof
  (home : MatchRecord)
  (rival : MatchRecord)
  (h1 : rival.wins = 2 * home.wins)
  (h2 : home.wins = 3)
  (h3 : home.draws = 4)
  (h4 : rival.draws = 4)
  (h5 : home.losses = 0)
  (h6 : rival.losses = 0) :
  totalMatches home + totalMatches rival = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_matches_proof_l3897_389726


namespace NUMINAMATH_CALUDE_new_average_weight_l3897_389724

theorem new_average_weight (n : ℕ) (w_avg : ℝ) (w_new : ℝ) :
  n = 29 →
  w_avg = 28 →
  w_new = 4 →
  (n * w_avg + w_new) / (n + 1) = 27.2 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l3897_389724


namespace NUMINAMATH_CALUDE_cubic_function_constraint_l3897_389780

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- f has both a maximum and a minimum value -/
def has_max_and_min (a : ℝ) : Prop := ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0

theorem cubic_function_constraint (a : ℝ) : 
  has_max_and_min a → a < -1 ∨ a > 2 := by sorry

end NUMINAMATH_CALUDE_cubic_function_constraint_l3897_389780


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_interval_l3897_389774

theorem quadratic_inequality_solution_interval (k : ℝ) : 
  (k > 0 ∧ ∃ x : ℝ, x^2 - 8*x + k < 0) ↔ (0 < k ∧ k < 16) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_interval_l3897_389774


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l3897_389709

theorem rectangle_shorter_side (area perimeter : ℝ) (h_area : area = 104) (h_perimeter : perimeter = 42) :
  ∃ (length width : ℝ), 
    length * width = area ∧ 
    2 * (length + width) = perimeter ∧ 
    min length width = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l3897_389709


namespace NUMINAMATH_CALUDE_distance_to_point_one_zero_l3897_389752

theorem distance_to_point_one_zero (z : ℂ) (h : z * (1 + Complex.I) = 4) :
  Complex.abs (z - 1) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_one_zero_l3897_389752


namespace NUMINAMATH_CALUDE_m_range_l3897_389748

theorem m_range (m : ℝ) : 
  (∀ θ : ℝ, m^2 + (Real.cos θ)^2 * m - 5*m + 4*(Real.sin θ)^2 ≥ 0) → 
  (m ≥ 4 ∨ m ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_m_range_l3897_389748


namespace NUMINAMATH_CALUDE_range_of_2x_minus_y_l3897_389739

theorem range_of_2x_minus_y (x y : ℝ) 
  (hx : 0 < x ∧ x < 4) 
  (hy : 0 < y ∧ y < 6) : 
  -6 < 2*x - y ∧ 2*x - y < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2x_minus_y_l3897_389739


namespace NUMINAMATH_CALUDE_child_b_share_l3897_389786

theorem child_b_share (total_amount : ℕ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_amount = 1800 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  (ratio_b * total_amount) / (ratio_a + ratio_b + ratio_c) = 600 := by
  sorry

end NUMINAMATH_CALUDE_child_b_share_l3897_389786


namespace NUMINAMATH_CALUDE_amanda_stroll_time_l3897_389737

/-- Amanda's stroll to Kimberly's house -/
theorem amanda_stroll_time (speed : ℝ) (distance : ℝ) (h1 : speed = 2) (h2 : distance = 6) :
  distance / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_amanda_stroll_time_l3897_389737


namespace NUMINAMATH_CALUDE_restaurant_order_combinations_l3897_389733

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of people ordering -/
def num_people : ℕ := 3

/-- The number of specialty dishes -/
def specialty_dishes : ℕ := 3

/-- The number of different meal combinations -/
def meal_combinations : ℕ := 1611

theorem restaurant_order_combinations :
  (menu_items ^ num_people) - (num_people * (specialty_dishes * (menu_items - specialty_dishes) ^ (num_people - 1))) = meal_combinations := by
  sorry

end NUMINAMATH_CALUDE_restaurant_order_combinations_l3897_389733


namespace NUMINAMATH_CALUDE_unique_fraction_sum_l3897_389769

theorem unique_fraction_sum : ∃! (a₂ a₃ a₄ a₅ a₆ a₇ : ℕ),
  (5 : ℚ) / 7 = a₂ / 2 + a₃ / 6 + a₄ / 24 + a₅ / 120 + a₆ / 720 + a₇ / 5040 ∧
  a₂ < 2 ∧ a₃ < 3 ∧ a₄ < 4 ∧ a₅ < 5 ∧ a₆ < 6 ∧ a₇ < 7 →
  a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_fraction_sum_l3897_389769


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3897_389741

/-- 
Given a hyperbola with equation x²/a² - y²/b² = 1,
if one of its asymptotes is y = (√7/3)x and 
the distance from one of its vertices to the nearer focus is 1,
then a = 3 and b = √7.
-/
theorem hyperbola_equation (a b : ℝ) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  (b/a = Real.sqrt 7 / 3) →               -- Asymptote condition
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c - a = 1) →  -- Vertex-focus distance condition
  (a = 3 ∧ b = Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3897_389741


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3897_389778

theorem min_sum_of_squares (x y z : ℝ) (h : x*y + y*z + x*z = 4) :
  x^2 + y^2 + z^2 ≥ 4 ∧ ∃ a b c : ℝ, a*b + b*c + a*c = 4 ∧ a^2 + b^2 + c^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3897_389778


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3897_389700

-- Define the hyperbola and its properties
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : a > 0 ∧ b > 0

-- Define the points and conditions
structure HyperbolaIntersection (h : Hyperbola) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_perp : (P.1 - Q.1) * (P.1 - F₁.1) + (P.2 - Q.2) * (P.2 - F₁.2) = 0  -- PQ ⊥ PF₁
  h_equal : (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2  -- |PF₁| = |PQ|
  h_on_hyperbola : P.1^2 / h.a^2 - P.2^2 / h.b^2 = 1 ∧ Q.1^2 / h.a^2 - Q.2^2 / h.b^2 = 1
  h_F₂_on_line : (Q.2 - P.2) * (F₂.1 - P.1) = (Q.1 - P.1) * (F₂.2 - P.2)  -- F₂ is on line PQ

-- Theorem statement
theorem hyperbola_eccentricity (h : Hyperbola) (i : HyperbolaIntersection h) :
  h.c / h.a = Real.sqrt (5 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3897_389700


namespace NUMINAMATH_CALUDE_inequality_proof_l3897_389734

theorem inequality_proof (a b c d : ℝ) : 
  (a + c)^2 * (b + d)^2 - 2 * (a * b^2 * c + b * c^2 * d + c * d^2 * a + d * a^2 * b + 4 * a * b * c * d) ≥ 0 ∧ 
  (a + c)^2 * (b + d)^2 - 4 * b * c * (c * d + d * a + a * b) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3897_389734


namespace NUMINAMATH_CALUDE_second_friend_shells_l3897_389775

theorem second_friend_shells (jovana_initial : ℕ) (first_friend : ℕ) (total : ℕ) : 
  jovana_initial = 5 → first_friend = 15 → total = 37 → 
  total - (jovana_initial + first_friend) = 17 := by
sorry

end NUMINAMATH_CALUDE_second_friend_shells_l3897_389775


namespace NUMINAMATH_CALUDE_gcd_52800_35275_l3897_389729

theorem gcd_52800_35275 : Nat.gcd 52800 35275 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_52800_35275_l3897_389729


namespace NUMINAMATH_CALUDE_eggs_sold_equals_540_l3897_389704

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 36

/-- The initial number of trays collected -/
def initial_trays : ℕ := 10

/-- The number of trays dropped accidentally -/
def dropped_trays : ℕ := 2

/-- The number of additional trays added -/
def additional_trays : ℕ := 7

/-- The total number of eggs sold -/
def total_eggs_sold : ℕ := eggs_per_tray * (initial_trays - dropped_trays + additional_trays)

theorem eggs_sold_equals_540 : total_eggs_sold = 540 := by
  sorry

end NUMINAMATH_CALUDE_eggs_sold_equals_540_l3897_389704


namespace NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_nine_l3897_389761

theorem unique_three_digit_number_divisible_by_nine :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 5 ∧ 
  (n / 100) % 10 = 3 ∧ 
  n % 9 = 0 ∧
  n = 315 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_nine_l3897_389761


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l3897_389728

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State that f is differentiable
variable (hf : Differentiable ℝ f)

-- Define the limit condition
variable (h_limit : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
  |((f 1 - f (1 + 2 * Δx)) / Δx) - 2| < ε)

-- Theorem statement
theorem tangent_slope_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_limit : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f 1 - f (1 + 2 * Δx)) / Δx) - 2| < ε) : 
  deriv f 1 = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l3897_389728


namespace NUMINAMATH_CALUDE_geometric_sum_seven_halves_l3897_389725

def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_seven_halves :
  geometric_sum (1/2) (1/2) 7 = 127/128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_seven_halves_l3897_389725


namespace NUMINAMATH_CALUDE_josh_film_cost_l3897_389713

/-- The cost of each film Josh bought -/
def film_cost : ℚ := 5

/-- The number of films Josh bought -/
def num_films : ℕ := 9

/-- The number of books Josh bought -/
def num_books : ℕ := 4

/-- The cost of each book -/
def book_cost : ℚ := 4

/-- The number of CDs Josh bought -/
def num_cds : ℕ := 6

/-- The cost of each CD -/
def cd_cost : ℚ := 3

/-- The total amount Josh spent -/
def total_spent : ℚ := 79

theorem josh_film_cost :
  film_cost * num_films + book_cost * num_books + cd_cost * num_cds = total_spent :=
by sorry

end NUMINAMATH_CALUDE_josh_film_cost_l3897_389713


namespace NUMINAMATH_CALUDE_western_rattlesnake_segments_l3897_389760

/-- The number of segments in Eastern rattlesnakes' tails -/
def eastern_segments : ℕ := 6

/-- The percentage difference in tail size as a fraction -/
def percentage_difference : ℚ := 1/4

/-- The number of segments in Western rattlesnakes' tails -/
def western_segments : ℕ := 8

/-- Theorem stating that the number of segments in Western rattlesnakes' tails is 8,
    given the conditions from the problem -/
theorem western_rattlesnake_segments :
  (western_segments : ℚ) - eastern_segments = percentage_difference * western_segments :=
sorry

end NUMINAMATH_CALUDE_western_rattlesnake_segments_l3897_389760


namespace NUMINAMATH_CALUDE_foundation_cost_theorem_l3897_389750

/-- Represents the dimensions of a concrete slab -/
structure SlabDimensions where
  length : Float
  width : Float
  height : Float

/-- Calculates the volume of a concrete slab -/
def slabVolume (d : SlabDimensions) : Float :=
  d.length * d.width * d.height

/-- Calculates the weight of concrete given its volume and density -/
def concreteWeight (volume density : Float) : Float :=
  volume * density

/-- Calculates the cost of concrete given its weight and price per pound -/
def concreteCost (weight pricePerPound : Float) : Float :=
  weight * pricePerPound

theorem foundation_cost_theorem 
  (slabDim : SlabDimensions)
  (concreteDensity : Float)
  (concretePricePerPound : Float)
  (numHomes : Nat) :
  slabDim.length = 100 →
  slabDim.width = 100 →
  slabDim.height = 0.5 →
  concreteDensity = 150 →
  concretePricePerPound = 0.02 →
  numHomes = 3 →
  concreteCost 
    (concreteWeight 
      (slabVolume slabDim * numHomes.toFloat) 
      concreteDensity) 
    concretePricePerPound = 45000 := by
  sorry

end NUMINAMATH_CALUDE_foundation_cost_theorem_l3897_389750


namespace NUMINAMATH_CALUDE_smallest_with_twelve_factors_l3897_389721

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The set of positive factors of a positive integer -/
def factors (n : ℕ+) : Set ℕ+ := sorry

theorem smallest_with_twelve_factors :
  ∃ (n : ℕ+), (num_factors n = 12) ∧
    (∀ m : ℕ+, m < n → num_factors m ≠ 12) ∧
    (n = 60) := by sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_factors_l3897_389721


namespace NUMINAMATH_CALUDE_exists_a_for_f_with_real_domain_and_range_l3897_389722

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1

-- State the theorem
theorem exists_a_for_f_with_real_domain_and_range :
  ∃ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a y = x) ∧ (∀ y : ℝ, ∃ x : ℝ, f a x = y) := by
  sorry

end NUMINAMATH_CALUDE_exists_a_for_f_with_real_domain_and_range_l3897_389722


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l3897_389765

theorem polynomial_identity_sum_of_squares : 
  ∀ (p q r s t u : ℤ), 
  (∀ x : ℝ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l3897_389765


namespace NUMINAMATH_CALUDE_vertical_angles_are_equal_l3897_389740

/-- Two angles are vertical if they are formed by two intersecting lines and are not adjacent. -/
def VerticalAngles (α β : Real) : Prop := sorry

theorem vertical_angles_are_equal (α β : Real) :
  VerticalAngles α β → α = β := by sorry

end NUMINAMATH_CALUDE_vertical_angles_are_equal_l3897_389740


namespace NUMINAMATH_CALUDE_isabella_book_purchase_l3897_389711

/-- The number of hardcover volumes bought by Isabella --/
def num_hardcovers : ℕ := 6

/-- The number of paperback volumes bought by Isabella --/
def num_paperbacks : ℕ := 12 - num_hardcovers

/-- The cost of a paperback volume in dollars --/
def paperback_cost : ℕ := 20

/-- The cost of a hardcover volume in dollars --/
def hardcover_cost : ℕ := 30

/-- The total number of volumes --/
def total_volumes : ℕ := 12

/-- The total cost of all volumes in dollars --/
def total_cost : ℕ := 300

theorem isabella_book_purchase :
  num_hardcovers = 6 ∧
  num_hardcovers + num_paperbacks = total_volumes ∧
  num_hardcovers * hardcover_cost + num_paperbacks * paperback_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_isabella_book_purchase_l3897_389711


namespace NUMINAMATH_CALUDE_system_solution_l3897_389799

/-- Given a system of equations, prove the solutions. -/
theorem system_solution (a b c : ℝ) :
  let eq1 := (y : ℝ) ^ 2 - z * x = a * (x + y + z) ^ 2
  let eq2 := x ^ 2 - y * z = b * (x + y + z) ^ 2
  let eq3 := z ^ 2 - x * y = c * (x + y + z) ^ 2
  (∃ s : ℝ,
    x = (2 * c - a - b + 1) * s ∧
    y = (2 * a - b - c + 1) * s ∧
    z = (2 * b - c - a + 1) * s ∧
    a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a = a + b + c) ∨
  (x = 0 ∧ y = 0 ∧ z = 0 ∧
    a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a ≠ a + b + c) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3897_389799


namespace NUMINAMATH_CALUDE_blue_pill_cost_is_correct_l3897_389788

/-- The cost of the blue pill in dollars -/
def blue_pill_cost : ℝ := 17

/-- The cost of the red pill in dollars -/
def red_pill_cost : ℝ := blue_pill_cost - 2

/-- The number of days for the treatment -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 672

theorem blue_pill_cost_is_correct :
  blue_pill_cost = 17 ∧
  red_pill_cost = blue_pill_cost - 2 ∧
  treatment_days * (blue_pill_cost + red_pill_cost) = total_cost := by
  sorry

#eval blue_pill_cost

end NUMINAMATH_CALUDE_blue_pill_cost_is_correct_l3897_389788


namespace NUMINAMATH_CALUDE_problem_statement_l3897_389723

theorem problem_statement (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) :
  10 * (6 * x + 14 * Real.pi) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3897_389723


namespace NUMINAMATH_CALUDE_base_3_10201_equals_100_l3897_389708

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_3_10201_equals_100 :
  base_3_to_10 [1, 0, 2, 0, 1] = 100 := by
  sorry

end NUMINAMATH_CALUDE_base_3_10201_equals_100_l3897_389708


namespace NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_l3897_389770

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_l3897_389770


namespace NUMINAMATH_CALUDE_all_odd_in_M_product_in_M_l3897_389731

-- Define the set M
def M : Set ℤ := {n : ℤ | ∃ (x y : ℤ), n = x^2 - y^2}

-- Statement 1: All odd numbers belong to M
theorem all_odd_in_M : ∀ (k : ℤ), (2 * k + 1) ∈ M := by sorry

-- Statement 3: If a ∈ M and b ∈ M, then ab ∈ M
theorem product_in_M : ∀ (a b : ℤ), a ∈ M → b ∈ M → (a * b) ∈ M := by sorry

end NUMINAMATH_CALUDE_all_odd_in_M_product_in_M_l3897_389731


namespace NUMINAMATH_CALUDE_quadratic_prime_values_l3897_389732

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial (a b c : ℤ) : ℤ → ℤ := fun x ↦ a * x^2 + b * x + c

/-- Predicate to check if a number is prime -/
def IsPrime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, 1 < m → m < n → ¬(n % m = 0)

theorem quadratic_prime_values 
  (a b c : ℤ) (n : ℤ) :
  (IsPrime (QuadraticPolynomial a b c (n - 1))) →
  (IsPrime (QuadraticPolynomial a b c n)) →
  (IsPrime (QuadraticPolynomial a b c (n + 1))) →
  ∃ m : ℤ, m ≠ n - 1 ∧ m ≠ n ∧ m ≠ n + 1 ∧ IsPrime (QuadraticPolynomial a b c m) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_prime_values_l3897_389732


namespace NUMINAMATH_CALUDE_partner_q_investment_time_l3897_389798

/-- The investment time of partner q given the investment and profit ratios -/
theorem partner_q_investment_time
  (investment_ratio : ℚ) -- Ratio of p's investment to q's investment
  (profit_ratio : ℚ) -- Ratio of p's profit to q's profit
  (p_time : ℕ) -- Time p invested in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 10)
  (h3 : p_time = 8) :
  ∃ q_time : ℕ, q_time = 16 ∧ 
  profit_ratio * investment_ratio * q_time = p_time :=
by sorry

end NUMINAMATH_CALUDE_partner_q_investment_time_l3897_389798


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3897_389783

theorem trig_expression_equality : 
  2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3897_389783


namespace NUMINAMATH_CALUDE_carla_daily_collection_l3897_389757

/-- The number of items Carla needs to collect each day -/
def daily_items (total_leaves total_bugs total_days : ℕ) : ℕ :=
  (total_leaves + total_bugs) / total_days

/-- Proof that Carla needs to collect 5 items per day -/
theorem carla_daily_collection :
  daily_items 30 20 10 = 5 :=
by sorry

end NUMINAMATH_CALUDE_carla_daily_collection_l3897_389757


namespace NUMINAMATH_CALUDE_chocolates_for_charlie_l3897_389703

/-- Represents the number of Saturdays in a month -/
def saturdays_in_month : ℕ := 4

/-- Represents the number of chocolates Kantana buys for herself each Saturday -/
def chocolates_for_self : ℕ := 2

/-- Represents the number of chocolates Kantana buys for her sister each Saturday -/
def chocolates_for_sister : ℕ := 1

/-- Represents the total number of chocolates Kantana bought for the month -/
def total_chocolates : ℕ := 22

/-- Theorem stating that Kantana bought 10 chocolates for Charlie's birthday gift -/
theorem chocolates_for_charlie : 
  total_chocolates - (saturdays_in_month * (chocolates_for_self + chocolates_for_sister)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_for_charlie_l3897_389703


namespace NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l3897_389767

theorem sum_of_integers_ending_in_3 :
  let first_term : ℕ := 103
  let last_term : ℕ := 493
  let common_difference : ℕ := 10
  let n : ℕ := (last_term - first_term) / common_difference + 1
  (n : ℤ) * (first_term + last_term) / 2 = 11920 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l3897_389767


namespace NUMINAMATH_CALUDE_leila_cake_count_l3897_389706

/-- The number of cakes Leila ate on Monday -/
def monday_cakes : ℕ := 6

/-- The number of cakes Leila ate on Friday -/
def friday_cakes : ℕ := 9

/-- The number of cakes Leila ate on Saturday -/
def saturday_cakes : ℕ := 3 * monday_cakes

/-- The total number of cakes Leila ate -/
def total_cakes : ℕ := monday_cakes + friday_cakes + saturday_cakes

theorem leila_cake_count : total_cakes = 33 := by
  sorry

end NUMINAMATH_CALUDE_leila_cake_count_l3897_389706


namespace NUMINAMATH_CALUDE_children_catered_count_l3897_389707

/-- Represents the number of children that can be catered with remaining food --/
def children_catered (total_adults : ℕ) (total_children : ℕ) (adults_meal_capacity : ℕ) (children_meal_capacity : ℕ) (adults_eaten : ℕ) (adult_child_consumption_ratio : ℚ) (adult_diet_restriction_percent : ℚ) (child_diet_restriction_percent : ℚ) : ℕ :=
  sorry

/-- Theorem stating the number of children that can be catered under given conditions --/
theorem children_catered_count : 
  children_catered 55 70 70 90 21 (3/2) (1/5) (3/20) = 63 :=
sorry

end NUMINAMATH_CALUDE_children_catered_count_l3897_389707


namespace NUMINAMATH_CALUDE_binomial_coefficient_1502_1_l3897_389754

theorem binomial_coefficient_1502_1 : Nat.choose 1502 1 = 1502 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1502_1_l3897_389754


namespace NUMINAMATH_CALUDE_impossible_transformation_l3897_389762

/-- Represents a natural number and its digits -/
structure DigitNumber where
  value : ℕ
  digits : List ℕ
  digits_valid : digits.all (· < 10)
  value_eq_digits : value = digits.foldl (fun acc d => acc * 10 + d) 0

/-- Defines the allowed operations on a DigitNumber -/
inductive Operation
  | multiply_by_two : Operation
  | rearrange_digits : Operation

/-- Applies an operation to a DigitNumber -/
def apply_operation (n : DigitNumber) (op : Operation) : DigitNumber :=
  match op with
  | Operation.multiply_by_two => sorry
  | Operation.rearrange_digits => sorry

/-- Checks if a DigitNumber is valid (non-zero first digit) -/
def is_valid (n : DigitNumber) : Prop :=
  n.digits.head? ≠ some 0

/-- Defines a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a DigitNumber -/
def apply_sequence (n : DigitNumber) (seq : OperationSequence) : DigitNumber :=
  seq.foldl apply_operation n

theorem impossible_transformation :
  ¬∃ (seq : OperationSequence),
    let start : DigitNumber := ⟨1, [1], sorry, sorry⟩
    let result := apply_sequence start seq
    result.value = 811 ∧ is_valid result :=
  sorry

end NUMINAMATH_CALUDE_impossible_transformation_l3897_389762


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l3897_389779

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = Real.pi) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l3897_389779


namespace NUMINAMATH_CALUDE_bamboo_with_nine_nodes_l3897_389745

/-- Given a geometric sequence of 9 terms, prove that if the product of the first 3 terms is 3
    and the product of the last 3 terms is 9, then the 5th term is √3. -/
theorem bamboo_with_nine_nodes (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence condition
  a 1 * a 2 * a 3 = 3 →             -- Product of first 3 terms
  a 7 * a 8 * a 9 = 9 →             -- Product of last 3 terms
  a 5 = Real.sqrt 3 :=              -- 5th term is √3
by sorry

end NUMINAMATH_CALUDE_bamboo_with_nine_nodes_l3897_389745


namespace NUMINAMATH_CALUDE_pyramid_volume_is_1280_l3897_389701

/-- Pyramid with square base ABCD and vertex E -/
structure Pyramid where
  baseArea : ℝ
  abeArea : ℝ
  cdeArea : ℝ
  distanceToMidpoint : ℝ

/-- Volume of the pyramid -/
def pyramidVolume (p : Pyramid) : ℝ := sorry

/-- Theorem stating the volume of the pyramid is 1280 -/
theorem pyramid_volume_is_1280 (p : Pyramid) 
  (h1 : p.baseArea = 256)
  (h2 : p.abeArea = 120)
  (h3 : p.cdeArea = 136)
  (h4 : p.distanceToMidpoint = 17) :
  pyramidVolume p = 1280 := by sorry

end NUMINAMATH_CALUDE_pyramid_volume_is_1280_l3897_389701


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_180_l3897_389744

theorem distinct_prime_factors_of_180 : Nat.card (Nat.factors 180).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_180_l3897_389744


namespace NUMINAMATH_CALUDE_min_shots_theorem_l3897_389736

/-- Represents a strategy for shooting at windows -/
def ShootingStrategy (n : ℕ) := ℕ → Fin n

/-- Determines if a shooting strategy is successful for all possible target positions -/
def is_successful_strategy (n : ℕ) (strategy : ShootingStrategy n) : Prop :=
  ∀ (start_pos : Fin n), ∃ (k : ℕ), strategy k = min (start_pos + k) (Fin.last n)

/-- The minimum number of shots needed to guarantee hitting the target -/
def min_shots_needed (n : ℕ) : ℕ := n / 2 + 1

/-- Theorem stating the minimum number of shots needed to guarantee hitting the target -/
theorem min_shots_theorem (n : ℕ) : 
  ∃ (strategy : ShootingStrategy n), is_successful_strategy n strategy ∧ 
  (∀ (other_strategy : ShootingStrategy n), 
    is_successful_strategy n other_strategy → 
    (∃ (k : ℕ), ∀ (i : ℕ), i < k → strategy i = other_strategy i) → 
    k ≥ min_shots_needed n) :=
sorry

end NUMINAMATH_CALUDE_min_shots_theorem_l3897_389736


namespace NUMINAMATH_CALUDE_sixth_power_sum_l3897_389758

theorem sixth_power_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 511 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l3897_389758


namespace NUMINAMATH_CALUDE_square_figure_perimeter_l3897_389772

/-- A figure composed of four identical squares with a specific arrangement -/
structure SquareFigure where
  /-- The side length of each square in the figure -/
  square_side : ℝ
  /-- The total area of the figure -/
  total_area : ℝ
  /-- The number of squares in the figure -/
  num_squares : ℕ
  /-- The number of exposed sides in the figure's perimeter -/
  exposed_sides : ℕ
  /-- Assertion that the figure is composed of four squares -/
  h_four_squares : num_squares = 4
  /-- Assertion that the total area is 144 cm² -/
  h_total_area : total_area = 144
  /-- Assertion that the exposed sides count is 9 based on the specific arrangement -/
  h_exposed_sides : exposed_sides = 9
  /-- Assertion that the total area is the sum of the areas of individual squares -/
  h_area_sum : total_area = num_squares * square_side ^ 2

/-- The perimeter of the SquareFigure -/
def perimeter (f : SquareFigure) : ℝ :=
  f.exposed_sides * f.square_side

/-- Theorem stating that the perimeter of the SquareFigure is 54 cm -/
theorem square_figure_perimeter (f : SquareFigure) : perimeter f = 54 := by
  sorry

end NUMINAMATH_CALUDE_square_figure_perimeter_l3897_389772


namespace NUMINAMATH_CALUDE_ages_proof_l3897_389773

/-- Represents the current age of Grant -/
def grant_age : ℕ := 25

/-- Represents the current age of the hospital -/
def hospital_age : ℕ := 40

/-- Represents the current age of the university -/
def university_age : ℕ := 30

/-- Represents the current age of the town library -/
def town_library_age : ℕ := 50

theorem ages_proof :
  (grant_age + 5 = (2 * (hospital_age + 5)) / 3) ∧
  (university_age = hospital_age - 10) ∧
  (town_library_age = university_age + 20) ∧
  (hospital_age < town_library_age) :=
by sorry

end NUMINAMATH_CALUDE_ages_proof_l3897_389773


namespace NUMINAMATH_CALUDE_max_spheres_in_cube_l3897_389753

/-- Represents a three-dimensional cube -/
structure Cube where
  edgeLength : ℝ

/-- Represents a sphere -/
structure Sphere where
  diameter : ℝ

/-- Calculates the maximum number of spheres that can fit in a cube -/
def maxSpheres (c : Cube) (s : Sphere) : ℕ :=
  sorry

/-- Theorem stating the maximum number of spheres in the given cube -/
theorem max_spheres_in_cube :
  ∃ (c : Cube) (s : Sphere),
    c.edgeLength = 4 ∧ s.diameter = 1 ∧ maxSpheres c s = 66 :=
by
  sorry

end NUMINAMATH_CALUDE_max_spheres_in_cube_l3897_389753


namespace NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l3897_389789

theorem gcd_of_powers_minus_one : 
  Nat.gcd (2^1100 - 1) (2^1122 - 1) = 2^22 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l3897_389789


namespace NUMINAMATH_CALUDE_arithmetic_sequence_decreasing_l3897_389792

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_decreasing
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a2 : (a 2 - 1)^3 + 2012 * (a 2 - 1) = 1)
  (h_a2011 : (a 2011 - 1)^3 + 2012 * (a 2011 - 1) = -1) :
  a 2011 < a 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_decreasing_l3897_389792


namespace NUMINAMATH_CALUDE_sqrt5_diamond_sqrt5_equals_20_l3897_389776

-- Define the custom operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt5_diamond_sqrt5_equals_20 : diamond (Real.sqrt 5) (Real.sqrt 5) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sqrt5_diamond_sqrt5_equals_20_l3897_389776
