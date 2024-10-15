import Mathlib

namespace NUMINAMATH_CALUDE_shirt_cost_l1838_183881

theorem shirt_cost (total_cost pants_cost tie_cost : ℕ) 
  (h1 : total_cost = 198)
  (h2 : pants_cost = 140)
  (h3 : tie_cost = 15) :
  total_cost - pants_cost - tie_cost = 43 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l1838_183881


namespace NUMINAMATH_CALUDE_age_ratio_nine_years_ago_l1838_183830

def henry_present_age : ℕ := 29
def jill_present_age : ℕ := 19

theorem age_ratio_nine_years_ago :
  (henry_present_age - 9) / (jill_present_age - 9) = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_nine_years_ago_l1838_183830


namespace NUMINAMATH_CALUDE_stratified_sample_size_is_72_l1838_183846

/-- Represents the number of teachers in each category -/
structure TeacherCounts where
  fullProf : Nat
  assocProf : Nat
  lecturers : Nat
  teachingAssistants : Nat

/-- Calculates the total number of teachers -/
def totalTeachers (counts : TeacherCounts) : Nat :=
  counts.fullProf + counts.assocProf + counts.lecturers + counts.teachingAssistants

/-- Calculates the sample size for stratified sampling -/
def stratifiedSampleSize (counts : TeacherCounts) (lecturersDrawn : Nat) : Nat :=
  let samplingRate := lecturersDrawn / counts.lecturers
  (totalTeachers counts) * samplingRate

/-- Theorem: Given the specific teacher counts and 16 lecturers drawn, 
    the stratified sample size is 72 -/
theorem stratified_sample_size_is_72 
  (counts : TeacherCounts) 
  (h1 : counts.fullProf = 120) 
  (h2 : counts.assocProf = 100) 
  (h3 : counts.lecturers = 80) 
  (h4 : counts.teachingAssistants = 60) 
  (h5 : stratifiedSampleSize counts 16 = 72) : 
  stratifiedSampleSize counts 16 = 72 := by
  sorry

#eval stratifiedSampleSize 
  { fullProf := 120, assocProf := 100, lecturers := 80, teachingAssistants := 60 } 16

end NUMINAMATH_CALUDE_stratified_sample_size_is_72_l1838_183846


namespace NUMINAMATH_CALUDE_furniture_store_problem_l1838_183829

/-- Furniture store problem -/
theorem furniture_store_problem 
  (a : ℝ) 
  (table_price : ℝ → ℝ) 
  (chair_price : ℝ → ℝ) 
  (table_retail : ℝ) 
  (chair_retail : ℝ) 
  (set_price : ℝ) 
  (h1 : table_price a = a) 
  (h2 : chair_price a = a - 140) 
  (h3 : table_retail = 380) 
  (h4 : chair_retail = 160) 
  (h5 : set_price = 940) 
  (h6 : 600 / (a - 140) = 1300 / a) 
  (x : ℝ) 
  (h7 : x + 5 * x + 20 ≤ 200) 
  (profit : ℝ → ℝ) 
  (h8 : profit x = (set_price - table_price a - 4 * chair_price a) * (1/2 * x) + 
                   (table_retail - table_price a) * (1/2 * x) + 
                   (chair_retail - chair_price a) * (5 * x + 20 - 4 * (1/2 * x))) :
  a = 260 ∧ 
  (∃ (max_x : ℝ), max_x = 30 ∧ 
    (∀ y, y + 5 * y + 20 ≤ 200 → profit y ≤ profit max_x) ∧ 
    profit max_x = 9200) := by
  sorry

end NUMINAMATH_CALUDE_furniture_store_problem_l1838_183829


namespace NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l1838_183837

theorem hua_luogeng_birthday_factorization (h : 19101112 = 1163 * 16424) :
  Nat.Prime 1163 ∧ ¬ Nat.Prime 16424 := by
  sorry

end NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l1838_183837


namespace NUMINAMATH_CALUDE_smallest_five_digit_square_cube_l1838_183879

theorem smallest_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ 
            (∃ x : ℕ, m = x^2) ∧ 
            (∃ y : ℕ, m = y^3) → 
    n ≤ m) ∧                  -- smallest such number
  n = 15625                   -- the answer
  := by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_square_cube_l1838_183879


namespace NUMINAMATH_CALUDE_hen_count_l1838_183835

theorem hen_count (total_heads : ℕ) (total_feet : ℕ) (hen_heads : ℕ) (hen_feet : ℕ) (cow_heads : ℕ) (cow_feet : ℕ) 
  (h1 : total_heads = 44)
  (h2 : total_feet = 128)
  (h3 : hen_heads = 1)
  (h4 : hen_feet = 2)
  (h5 : cow_heads = 1)
  (h6 : cow_feet = 4) :
  ∃ (num_hens : ℕ), num_hens = 24 ∧ 
    num_hens * hen_heads + (total_heads - num_hens) * cow_heads = total_heads ∧
    num_hens * hen_feet + (total_heads - num_hens) * cow_feet = total_feet :=
by sorry

end NUMINAMATH_CALUDE_hen_count_l1838_183835


namespace NUMINAMATH_CALUDE_sam_distance_sam_drove_220_miles_l1838_183857

/-- Calculates the total distance driven by Sam given Marguerite's speed and Sam's driving conditions. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_initial_time : ℝ) (sam_increased_time : ℝ) (speed_increase : ℝ) : ℝ :=
  let marguerite_speed := marguerite_distance / marguerite_time
  let sam_initial_distance := marguerite_speed * sam_initial_time
  let sam_increased_speed := marguerite_speed * (1 + speed_increase)
  let sam_increased_distance := sam_increased_speed * sam_increased_time
  sam_initial_distance + sam_increased_distance

/-- Proves that Sam drove 220 miles given the problem conditions. -/
theorem sam_drove_220_miles : sam_distance 150 3 2 2 0.2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_sam_drove_220_miles_l1838_183857


namespace NUMINAMATH_CALUDE_zero_product_property_l1838_183894

theorem zero_product_property {α : Type*} [Semiring α] {a b : α} :
  a * b = 0 → (a = 0 ∨ b = 0) := by sorry

end NUMINAMATH_CALUDE_zero_product_property_l1838_183894


namespace NUMINAMATH_CALUDE_no_integer_solution_l1838_183890

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 19 * x^2 - 76 * y^2 = 1976 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1838_183890


namespace NUMINAMATH_CALUDE_parallel_vectors_l1838_183852

theorem parallel_vectors (m n : ℝ × ℝ) : 
  m = (2, 8) → n = (-4, t) → m.1 * n.2 = m.2 * n.1 → t = -16 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l1838_183852


namespace NUMINAMATH_CALUDE_platform_length_l1838_183834

/-- Given a train of length 300 meters that crosses a platform in 38 seconds
    and a signal pole in 18 seconds, prove that the length of the platform
    is approximately 333.46 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_time = 38)
  (h3 : pole_time = 18) :
  ∃ (platform_length : ℝ), abs (platform_length - 333.46) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1838_183834


namespace NUMINAMATH_CALUDE_exist_expression_24_set1_exist_expression_24_set2_l1838_183831

-- Define a type for arithmetic operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a type for arithmetic expressions
inductive Expr
  | Num (n : ℕ)
  | BinOp (op : Operation) (e1 e2 : Expr)

-- Define a function to evaluate expressions
def eval : Expr → ℚ
  | Expr.Num n => n
  | Expr.BinOp Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.BinOp Operation.Sub e1 e2 => eval e1 - eval e2
  | Expr.BinOp Operation.Mul e1 e2 => eval e1 * eval e2
  | Expr.BinOp Operation.Div e1 e2 => eval e1 / eval e2

-- Define a function to check if an expression uses all given numbers exactly once
def usesAllNumbers (e : Expr) (nums : List ℕ) : Prop := sorry

-- Theorem for the first set of numbers
theorem exist_expression_24_set1 :
  ∃ (e : Expr), usesAllNumbers e [7, 12, 9, 12] ∧ eval e = 24 := by sorry

-- Theorem for the second set of numbers
theorem exist_expression_24_set2 :
  ∃ (e : Expr), usesAllNumbers e [3, 9, 5, 9] ∧ eval e = 24 := by sorry

end NUMINAMATH_CALUDE_exist_expression_24_set1_exist_expression_24_set2_l1838_183831


namespace NUMINAMATH_CALUDE_not_power_of_prime_l1838_183874

theorem not_power_of_prime (n : ℕ+) (q : ℕ) (h_prime : Nat.Prime q) :
  ¬∃ k : ℕ, (n : ℝ)^q + ((n - 1 : ℝ) / 2)^2 = (q : ℝ)^k := by
  sorry

end NUMINAMATH_CALUDE_not_power_of_prime_l1838_183874


namespace NUMINAMATH_CALUDE_sum_coordinates_of_B_l1838_183889

/-- Given that M(5,3) is the midpoint of AB and A has coordinates (10,2), 
    prove that the sum of coordinates of point B is 4. -/
theorem sum_coordinates_of_B (A B M : ℝ × ℝ) : 
  M = (5, 3) → 
  A = (10, 2) → 
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 + B.2 = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_B_l1838_183889


namespace NUMINAMATH_CALUDE_range_of_p_l1838_183845

/-- The set A of real numbers x satisfying the quadratic equation x^2 + (p+2)x + 1 = 0 -/
def A (p : ℝ) : Set ℝ := {x | x^2 + (p+2)*x + 1 = 0}

/-- The theorem stating the range of p given the conditions -/
theorem range_of_p (p : ℝ) (h : A p ∩ Set.Ici (0 : ℝ) = ∅) : p > -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_p_l1838_183845


namespace NUMINAMATH_CALUDE_letter_distribution_l1838_183824

theorem letter_distribution (n : ℕ) (k : ℕ) : 
  n = 4 ∧ k = 3 → k^n = 81 := by
  sorry

end NUMINAMATH_CALUDE_letter_distribution_l1838_183824


namespace NUMINAMATH_CALUDE_flowers_in_pot_l1838_183807

theorem flowers_in_pot (chrysanthemums : ℕ) (roses : ℕ) : 
  chrysanthemums = 5 → roses = 2 → chrysanthemums + roses = 7 := by
  sorry

end NUMINAMATH_CALUDE_flowers_in_pot_l1838_183807


namespace NUMINAMATH_CALUDE_computer_game_cost_l1838_183853

/-- The cost of a computer game, given the total cost of movie tickets and the total spent on entertainment. -/
theorem computer_game_cost (movie_tickets_cost total_spent : ℕ) : 
  movie_tickets_cost = 36 → total_spent = 102 → total_spent - movie_tickets_cost = 66 := by
  sorry

end NUMINAMATH_CALUDE_computer_game_cost_l1838_183853


namespace NUMINAMATH_CALUDE_square_gt_necessary_not_sufficient_l1838_183876

theorem square_gt_necessary_not_sufficient (a : ℝ) :
  (∀ a, a > 1 → a^2 > a) ∧ 
  (∃ a, a^2 > a ∧ ¬(a > 1)) :=
sorry

end NUMINAMATH_CALUDE_square_gt_necessary_not_sufficient_l1838_183876


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1838_183819

theorem quadratic_roots_condition (b c : ℝ) :
  (c < 0 → ∃ x : ℂ, x^2 + b*x + c = 0) ∧
  ¬(∃ x : ℂ, x^2 + b*x + c = 0 → c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1838_183819


namespace NUMINAMATH_CALUDE_triangle_area_l1838_183850

/-- A triangle with sides 8, 15, and 17 has an area of 60 -/
theorem triangle_area : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c area =>
    a = 8 ∧ b = 15 ∧ c = 17 →
    area = 60

/-- The proof of the theorem -/
lemma prove_triangle_area : triangle_area 8 15 17 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1838_183850


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l1838_183821

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2 - x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 4 * x - 1

-- Theorem statement
theorem tangent_line_at_origin : 
  ∀ x y : ℝ, (x + y = 0) ↔ (∃ t : ℝ, y = f t ∧ y - f 0 = f' 0 * (x - 0)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l1838_183821


namespace NUMINAMATH_CALUDE_mary_seashells_count_l1838_183847

/-- The number of seashells Sam found -/
def sam_seashells : ℕ := 18

/-- The total number of seashells Sam and Mary found together -/
def total_seashells : ℕ := 65

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := total_seashells - sam_seashells

theorem mary_seashells_count : mary_seashells = 47 := by sorry

end NUMINAMATH_CALUDE_mary_seashells_count_l1838_183847


namespace NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l1838_183848

def fibonacci_factorial_series : List Nat :=
  [1, 1, 2, 3, 5, 8, 13, 21, 34]

def last_two_digits (n : Nat) : Nat :=
  n % 100

def sum_last_two_digits (series : List Nat) : Nat :=
  (series.map (λ x => last_two_digits (Nat.factorial x))).sum

theorem fibonacci_factorial_last_two_digits_sum :
  sum_last_two_digits fibonacci_factorial_series = 50 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l1838_183848


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l1838_183872

/-- Parabola with vertex at origin and focus on positive x-axis -/
structure Parabola where
  focus : ℝ × ℝ
  focus_on_x_axis : focus.2 = 0 ∧ focus.1 > 0

/-- The curve E: x²+y²-6x+4y-3=0 -/
def curve_E (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 4*y - 3 = 0

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : point.2^2 = 2 * p.focus.1 * point.1

theorem parabola_point_coordinates (p : Parabola) 
  (h1 : ∃! x y, curve_E x y ∧ x = -p.focus.1) 
  (A : PointOnParabola p) 
  (h2 : A.point.1 * (A.point.1 - p.focus.1) + A.point.2 * A.point.2 = -4) :
  A.point = (1, 2) ∨ A.point = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l1838_183872


namespace NUMINAMATH_CALUDE_range_f_and_a_condition_l1838_183842

/-- The function f(x) = 3|x-1| + |3x+1| -/
def f (x : ℝ) : ℝ := 3 * abs (x - 1) + abs (3 * x + 1)

/-- The function g(x) = |x+2| + |x-a| -/
def g (a : ℝ) (x : ℝ) : ℝ := abs (x + 2) + abs (x - a)

/-- The set A, which is the range of f -/
def A : Set ℝ := Set.range f

/-- The set B, which is the range of g for a given a -/
def B (a : ℝ) : Set ℝ := Set.range (g a)

theorem range_f_and_a_condition (a : ℝ) :
  (A = Set.Ici 4) ∧ (A ∪ B a = B a) → a ∈ Set.Icc (-6) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_f_and_a_condition_l1838_183842


namespace NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l1838_183843

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

-- State the theorem
theorem f_composition_equals_pi_plus_one :
  f (f (f (-1))) = Real.pi + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l1838_183843


namespace NUMINAMATH_CALUDE_dictionary_cost_l1838_183862

theorem dictionary_cost (total_cost dinosaur_cost cookbook_cost : ℕ) 
  (h1 : total_cost = 37)
  (h2 : dinosaur_cost = 19)
  (h3 : cookbook_cost = 7) :
  total_cost - dinosaur_cost - cookbook_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_dictionary_cost_l1838_183862


namespace NUMINAMATH_CALUDE_base7_312_equals_base4_2310_l1838_183838

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base7_312_equals_base4_2310 :
  base10ToBase4 (base7ToBase10 [2, 1, 3]) = [2, 3, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_base7_312_equals_base4_2310_l1838_183838


namespace NUMINAMATH_CALUDE_relationship_abc_l1838_183875

theorem relationship_abc : ∀ (a b c : ℕ),
  a = 5^140 ∧ b = 3^210 ∧ c = 2^280 →
  c < a ∧ a < b := by
sorry

end NUMINAMATH_CALUDE_relationship_abc_l1838_183875


namespace NUMINAMATH_CALUDE_social_media_time_ratio_l1838_183898

/-- Proves that the ratio of daily time spent on social media to total daily time spent on phone is 1:2 -/
theorem social_media_time_ratio 
  (daily_phone_time : ℝ) 
  (weekly_social_media_time : ℝ) 
  (h1 : daily_phone_time = 6)
  (h2 : weekly_social_media_time = 21) :
  (weekly_social_media_time / 7) / daily_phone_time = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_social_media_time_ratio_l1838_183898


namespace NUMINAMATH_CALUDE_library_fiction_percentage_l1838_183865

/-- Proves that given the conditions of the library problem, the percentage of fiction novels in the original collection is approximately 30.66%. -/
theorem library_fiction_percentage 
  (total_volumes : ℕ)
  (transferred_fraction : ℚ)
  (transferred_fiction_fraction : ℚ)
  (remaining_fiction_percentage : ℚ)
  (h_total : total_volumes = 18360)
  (h_transferred : transferred_fraction = 1/3)
  (h_transferred_fiction : transferred_fiction_fraction = 1/5)
  (h_remaining_fiction : remaining_fiction_percentage = 35.99999999999999/100) :
  ∃ (original_fiction_percentage : ℚ), 
    (original_fiction_percentage ≥ 30.65/100) ∧ 
    (original_fiction_percentage ≤ 30.67/100) := by
  sorry

end NUMINAMATH_CALUDE_library_fiction_percentage_l1838_183865


namespace NUMINAMATH_CALUDE_pascal_triangle_32nd_row_31st_element_l1838_183815

theorem pascal_triangle_32nd_row_31st_element : Nat.choose 32 30 = 496 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_32nd_row_31st_element_l1838_183815


namespace NUMINAMATH_CALUDE_divisibility_problem_l1838_183841

theorem divisibility_problem (n : ℕ) (h1 : n > 0) (h2 : 100 % n = 3) (h3 : 197 % n = 3) : n = 97 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1838_183841


namespace NUMINAMATH_CALUDE_problem_C_most_suitable_for_systematic_sampling_l1838_183886

/-- Represents a sampling problem with population size and sample size -/
structure SamplingProblem where
  population_size : ℕ
  sample_size : ℕ

/-- Defines the suitability of a sampling method for a given problem -/
def systematic_sampling_suitability (problem : SamplingProblem) : ℕ :=
  if problem.population_size ≥ 1000 ∧ problem.sample_size ≥ 100 then 3
  else if problem.population_size < 100 ∨ problem.sample_size < 20 then 1
  else 2

/-- The sampling problems given in the question -/
def problem_A : SamplingProblem := ⟨48, 8⟩
def problem_B : SamplingProblem := ⟨210, 21⟩
def problem_C : SamplingProblem := ⟨1200, 100⟩
def problem_D : SamplingProblem := ⟨1200, 10⟩

/-- Theorem stating that problem C is most suitable for systematic sampling -/
theorem problem_C_most_suitable_for_systematic_sampling :
  systematic_sampling_suitability problem_C > systematic_sampling_suitability problem_A ∧
  systematic_sampling_suitability problem_C > systematic_sampling_suitability problem_B ∧
  systematic_sampling_suitability problem_C > systematic_sampling_suitability problem_D :=
sorry

end NUMINAMATH_CALUDE_problem_C_most_suitable_for_systematic_sampling_l1838_183886


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1838_183849

theorem quadratic_factorization (b : ℤ) : 
  (∃ (c d e f : ℤ), (35 : ℤ) * x ^ 2 + b * x + 35 = (c * x + d) * (e * x + f)) →
  (∃ (k : ℤ), b = 2 * k) ∧ 
  ¬(∀ (k : ℤ), ∃ (c d e f : ℤ), (35 : ℤ) * x ^ 2 + (2 * k) * x + 35 = (c * x + d) * (e * x + f)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1838_183849


namespace NUMINAMATH_CALUDE_all_terms_irrational_l1838_183816

theorem all_terms_irrational (a : ℕ → ℝ) 
  (h_pos : ∀ k, a k > 0)
  (h_rel : ∀ k, (a (k + 1) + k) * a k = 1) :
  ∀ k, Irrational (a k) := by
sorry

end NUMINAMATH_CALUDE_all_terms_irrational_l1838_183816


namespace NUMINAMATH_CALUDE_prob_win_match_value_l1838_183863

/-- Probability of player A winning a single game -/
def p : ℝ := 0.6

/-- Probability of player A winning the match in a best of 3 games -/
def prob_win_match : ℝ := p * p * (1 - p) + p * (1 - p) * p + p * p

/-- Theorem stating that the probability of player A winning the match is 0.648 -/
theorem prob_win_match_value : prob_win_match = 0.648 := by sorry

end NUMINAMATH_CALUDE_prob_win_match_value_l1838_183863


namespace NUMINAMATH_CALUDE_product_expansion_sum_l1838_183871

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (2 * x^2 - 4 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + 2 * c + d = -24 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l1838_183871


namespace NUMINAMATH_CALUDE_seating_arrangements_l1838_183827

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem seating_arrangements (total_people : ℕ) (restricted_people : ℕ) 
  (h1 : total_people = 10) 
  (h2 : restricted_people = 3) : 
  factorial total_people - 
  (factorial (total_people - restricted_people + 1) * factorial restricted_people + 
   restricted_people * choose (total_people - restricted_people + 1) 1 * 
   factorial (total_people - restricted_people) - 
   restricted_people * (factorial (total_people - restricted_people + 1) * 
   factorial restricted_people)) = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1838_183827


namespace NUMINAMATH_CALUDE_students_not_enrolled_l1838_183802

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ)
  (h1 : total = 94)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 40 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l1838_183802


namespace NUMINAMATH_CALUDE_f_condition_l1838_183896

/-- The function f(x) = x^2 + 2ax + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + a

/-- The theorem stating the condition for f(f(x)) > x for all x ∈ ℝ -/
theorem f_condition (a : ℝ) : 
  (∀ x : ℝ, f a (f a x) > x) ↔ (1 - Real.sqrt 3 / 2 < a ∧ a < 1 + Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_condition_l1838_183896


namespace NUMINAMATH_CALUDE_sum_of_coefficients_P_l1838_183867

/-- The polynomial in question -/
def P (x : ℝ) : ℝ := (2 * x^2021 - x^2020 + x^2019)^11 - 29

/-- Theorem stating that the sum of coefficients of P is 2019 -/
theorem sum_of_coefficients_P : (P 1) = 2019 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_P_l1838_183867


namespace NUMINAMATH_CALUDE_apple_cost_l1838_183882

/-- Given that apples cost m yuan per kilogram, prove that the cost of purchasing 3 kilograms of apples is 3m yuan. -/
theorem apple_cost (m : ℝ) : m * 3 = 3 * m := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_l1838_183882


namespace NUMINAMATH_CALUDE_total_steel_parts_l1838_183836

/-- Represents the number of machines of type A -/
def a : ℕ := sorry

/-- Represents the number of machines of type B -/
def b : ℕ := sorry

/-- The total number of machines -/
def total_machines : ℕ := 21

/-- The total number of chrome parts -/
def total_chrome_parts : ℕ := 66

/-- Steel parts in a type A machine -/
def steel_parts_A : ℕ := 3

/-- Chrome parts in a type A machine -/
def chrome_parts_A : ℕ := 2

/-- Steel parts in a type B machine -/
def steel_parts_B : ℕ := 2

/-- Chrome parts in a type B machine -/
def chrome_parts_B : ℕ := 4

theorem total_steel_parts :
  a + b = total_machines ∧
  chrome_parts_A * a + chrome_parts_B * b = total_chrome_parts →
  steel_parts_A * a + steel_parts_B * b = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_steel_parts_l1838_183836


namespace NUMINAMATH_CALUDE_converse_square_right_angles_false_l1838_183811

-- Define a quadrilateral
structure Quadrilateral :=
  (is_right_angled : Bool)
  (is_square : Bool)

-- Define the property that all angles are right angles
def all_angles_right (q : Quadrilateral) : Prop :=
  q.is_right_angled = true

-- Define the property of being a square
def is_square (q : Quadrilateral) : Prop :=
  q.is_square = true

-- Theorem: The converse of "All four angles of a square are right angles" is false
theorem converse_square_right_angles_false :
  ¬ (∀ q : Quadrilateral, all_angles_right q → is_square q) :=
by sorry

end NUMINAMATH_CALUDE_converse_square_right_angles_false_l1838_183811


namespace NUMINAMATH_CALUDE_correct_set_for_60_deg_terminal_side_l1838_183860

/-- The set of angles with the same terminal side as a 60° angle -/
def SameTerminalSideAs60Deg : Set ℝ :=
  {α | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3}

/-- Theorem stating that SameTerminalSideAs60Deg is the correct set -/
theorem correct_set_for_60_deg_terminal_side :
  SameTerminalSideAs60Deg = {α | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3} := by
  sorry

end NUMINAMATH_CALUDE_correct_set_for_60_deg_terminal_side_l1838_183860


namespace NUMINAMATH_CALUDE_equal_cost_sharing_l1838_183840

theorem equal_cost_sharing (X Y Z : ℝ) (h : Y > X) :
  let total_cost := X + Y + Z
  let equal_share := total_cost / 2
  let nina_payment := equal_share - Y
  nina_payment = (X + Z - Y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_cost_sharing_l1838_183840


namespace NUMINAMATH_CALUDE_min_a_value_l1838_183888

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + x

def holds_inequality (a : ℤ) : Prop :=
  ∀ x > 0, f x ≤ ((↑a / 2) - 1) * x^2 + ↑a * x - 1

theorem min_a_value :
  ∃ a : ℤ, holds_inequality a ∧ ∀ b : ℤ, b < a → ¬(holds_inequality b) :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l1838_183888


namespace NUMINAMATH_CALUDE_function_extrema_implies_a_range_l1838_183859

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- State the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ (x : ℝ), f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  (a > 6 ∨ a < -3) :=
sorry

end NUMINAMATH_CALUDE_function_extrema_implies_a_range_l1838_183859


namespace NUMINAMATH_CALUDE_dollar_equality_l1838_183854

-- Define the $ operation
def dollar (a b : ℝ) : ℝ := (a - b)^2

-- Theorem statement
theorem dollar_equality (x y : ℝ) : 
  dollar ((2*x + y)^2) ((x - 2*y)^2) = (3*x^2 + 8*x*y - 3*y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_dollar_equality_l1838_183854


namespace NUMINAMATH_CALUDE_soda_cans_purchased_l1838_183825

/-- The number of cans of soda that can be purchased for a given amount of money -/
theorem soda_cans_purchased (S Q D : ℚ) (h1 : S > 0) (h2 : Q > 0) (h3 : D ≥ 0) :
  let cans_per_quarter := S / Q
  let quarters_per_dollar := 4
  let cans_per_dollar := cans_per_quarter * quarters_per_dollar
  cans_per_dollar * D = 4 * D * S / Q :=
by sorry

end NUMINAMATH_CALUDE_soda_cans_purchased_l1838_183825


namespace NUMINAMATH_CALUDE_mabel_tomatoes_l1838_183878

/-- The number of tomato plants Mabel planted -/
def num_plants : ℕ := 4

/-- The number of tomatoes on the first plant -/
def first_plant_tomatoes : ℕ := 8

/-- The number of additional tomatoes on the second plant compared to the first -/
def second_plant_additional : ℕ := 4

/-- The factor by which the remaining plants' tomatoes exceed the sum of the first two plants -/
def remaining_plants_factor : ℕ := 3

/-- The total number of tomatoes Mabel has -/
def total_tomatoes : ℕ := 140

theorem mabel_tomatoes :
  let second_plant_tomatoes := first_plant_tomatoes + second_plant_additional
  let first_two_plants := first_plant_tomatoes + second_plant_tomatoes
  let remaining_plants_tomatoes := 2 * (remaining_plants_factor * first_two_plants)
  first_plant_tomatoes + second_plant_tomatoes + remaining_plants_tomatoes = total_tomatoes :=
by sorry

end NUMINAMATH_CALUDE_mabel_tomatoes_l1838_183878


namespace NUMINAMATH_CALUDE_num_faces_after_transformation_l1838_183866

/-- Represents the number of steps in the transformation process -/
def num_steps : ℕ := 5

/-- The initial number of vertices in a cube -/
def initial_vertices : ℕ := 8

/-- The initial number of edges in a cube -/
def initial_edges : ℕ := 12

/-- The factor by which vertices and edges increase in each step -/
def increase_factor : ℕ := 3

/-- Calculates the number of vertices after the transformation -/
def final_vertices : ℕ := initial_vertices * increase_factor ^ num_steps

/-- Calculates the number of edges after the transformation -/
def final_edges : ℕ := initial_edges * increase_factor ^ num_steps

/-- Theorem stating the number of faces after the transformation -/
theorem num_faces_after_transformation : 
  final_vertices - final_edges + 974 = 2 :=
sorry

end NUMINAMATH_CALUDE_num_faces_after_transformation_l1838_183866


namespace NUMINAMATH_CALUDE_pets_after_one_month_l1838_183820

/-- Calculates the number of pets in an animal shelter after one month --/
theorem pets_after_one_month
  (initial_dogs : ℕ)
  (initial_cats : ℕ)
  (initial_lizards : ℕ)
  (dog_adoption_rate : ℚ)
  (cat_adoption_rate : ℚ)
  (lizard_adoption_rate : ℚ)
  (new_pets : ℕ)
  (h_dogs : initial_dogs = 30)
  (h_cats : initial_cats = 28)
  (h_lizards : initial_lizards = 20)
  (h_dog_rate : dog_adoption_rate = 1/2)
  (h_cat_rate : cat_adoption_rate = 1/4)
  (h_lizard_rate : lizard_adoption_rate = 1/5)
  (h_new_pets : new_pets = 13) :
  ↑initial_dogs + ↑initial_cats + ↑initial_lizards -
  (↑initial_dogs * dog_adoption_rate + ↑initial_cats * cat_adoption_rate + ↑initial_lizards * lizard_adoption_rate) +
  ↑new_pets = 65 := by
  sorry


end NUMINAMATH_CALUDE_pets_after_one_month_l1838_183820


namespace NUMINAMATH_CALUDE_product_remainder_one_l1838_183880

theorem product_remainder_one (a b : ℕ) : 
  a % 3 = 1 → b % 3 = 1 → (a * b) % 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_one_l1838_183880


namespace NUMINAMATH_CALUDE_sum_of_valid_divisors_l1838_183895

/-- The sum of valid divisors of 360 that satisfy specific conditions --/
theorem sum_of_valid_divisors : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ 360 % x = 0 ∧ x ≥ 18 ∧ 360 / x ≥ 12) 
    (Finset.range 361)).sum id = 92 := by sorry

end NUMINAMATH_CALUDE_sum_of_valid_divisors_l1838_183895


namespace NUMINAMATH_CALUDE_expansion_coefficients_l1838_183839

-- Define the coefficient of x in the expansion
def S (n : ℕ) : ℚ := (n + 1 : ℚ) / (2 * (Nat.factorial (n - 1)))

-- Define the ratio of coefficients T_n / S_n
def T_S_ratio (n : ℕ) : ℚ := (1 / 4 : ℚ) * n^2 - (1 / 12 : ℚ) * n - (1 / 6 : ℚ)

-- Theorem statement
theorem expansion_coefficients (n : ℕ) (h : n ≥ 2) : 
  S n = (n + 1 : ℚ) / (2 * (Nat.factorial (n - 1))) ∧ 
  T_S_ratio n = (1 / 4 : ℚ) * n^2 - (1 / 12 : ℚ) * n - (1 / 6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l1838_183839


namespace NUMINAMATH_CALUDE_exist_distinct_prime_divisors_l1838_183873

theorem exist_distinct_prime_divisors (k n : ℕ+) (h : k > n!) :
  ∃ (p : Fin n → ℕ), (∀ i : Fin n, Nat.Prime (p i)) ∧
                     (∀ i j : Fin n, i ≠ j → p i ≠ p j) ∧
                     (∀ i : Fin n, (p i) ∣ (k + i.val + 1)) := by
  sorry

end NUMINAMATH_CALUDE_exist_distinct_prime_divisors_l1838_183873


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1838_183861

theorem cubic_roots_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 11*p - 3 = 0 →
  q^3 - 8*q^2 + 11*q - 3 = 0 →
  r^3 - 8*r^2 + 11*r - 3 = 0 →
  p / (q*r - 1) + q / (p*r - 1) + r / (p*q - 1) = 17/29 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1838_183861


namespace NUMINAMATH_CALUDE_discount_difference_l1838_183887

theorem discount_difference (bill : ℝ) (single_discount : ℝ) (first_discount : ℝ) (second_discount : ℝ) : 
  bill = 8000 ∧ 
  single_discount = 0.3 ∧ 
  first_discount = 0.26 ∧ 
  second_discount = 0.05 → 
  (bill * (1 - first_discount) * (1 - second_discount)) - (bill * (1 - single_discount)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l1838_183887


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1838_183899

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 4 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 ∧ x < n → 
    (¬∃ (y : ℕ), 5 * x = y^2) ∨ 
    (¬∃ (z : ℕ), 4 * x = z^3)) ∧
  n = 1600 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1838_183899


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1838_183892

theorem quadratic_inequality (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * a * x^2 + b * x + c = 0) →
  min c (a + c + 1) ≤ max (abs (b - a + 1)) (abs (b + a - 1)) ∧
  (min c (a + c + 1) = max (abs (b - a + 1)) (abs (b + a - 1)) ↔
    ((a = 1 ∧ b = 0 ∧ c = 0) ∨ (a ≤ -1 ∧ 2 * a - abs b + c = 0))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1838_183892


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l1838_183809

def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  is_increasing_geometric_sequence a →
  a 1 + a 4 = 9 →
  a 2 * a 3 = 8 →
  a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l1838_183809


namespace NUMINAMATH_CALUDE_conference_handshakes_l1838_183813

theorem conference_handshakes (n : ℕ) (h : n = 12) : n.choose 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1838_183813


namespace NUMINAMATH_CALUDE_largest_fraction_l1838_183805

theorem largest_fraction : 
  let fractions := [5/12, 7/15, 23/45, 89/178, 199/400]
  ∀ x ∈ fractions, (23/45 : ℚ) ≥ x := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l1838_183805


namespace NUMINAMATH_CALUDE_arithmetic_sequence_value_l1838_183806

/-- 
Given that -7, a, and 1 form an arithmetic sequence, prove that a = -3.
-/
theorem arithmetic_sequence_value (a : ℝ) : 
  (∃ d : ℝ, a - (-7) = d ∧ 1 - a = d) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_value_l1838_183806


namespace NUMINAMATH_CALUDE_wolf_nobel_count_l1838_183851

/-- Represents the number of scientists with different prize combinations -/
structure ScientistCounts where
  total : ℕ
  wolf : ℕ
  nobel : ℕ
  wolfNobel : ℕ

/-- The conditions of the workshop -/
def workshopConditions (s : ScientistCounts) : Prop :=
  s.total = 50 ∧
  s.wolf = 31 ∧
  s.nobel = 29 ∧
  s.total - s.wolf = (s.nobel - s.wolfNobel) + (s.total - s.wolf - (s.nobel - s.wolfNobel)) + 3

/-- The theorem stating that 18 Wolf Prize laureates were also Nobel Prize laureates -/
theorem wolf_nobel_count (s : ScientistCounts) :
  workshopConditions s → s.wolfNobel = 18 := by
  sorry

end NUMINAMATH_CALUDE_wolf_nobel_count_l1838_183851


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1838_183804

/-- The line 4x + 3y + k = 0 is tangent to the parabola y² = 16x if and only if k = 9 -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, 4*x + 3*y + k = 0 → y^2 = 16*x) ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1838_183804


namespace NUMINAMATH_CALUDE_largest_multiple_with_negation_constraint_l1838_183818

theorem largest_multiple_with_negation_constraint : 
  ∀ n : ℤ, n % 12 = 0 ∧ -n > -150 → n ≤ 144 := by sorry

end NUMINAMATH_CALUDE_largest_multiple_with_negation_constraint_l1838_183818


namespace NUMINAMATH_CALUDE_square_ratio_proof_l1838_183803

theorem square_ratio_proof : ∃ (a b c : ℕ), 
  (300 : ℚ) / 75 = (a * Real.sqrt b / c)^2 ∧ a + b + c = 4 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_proof_l1838_183803


namespace NUMINAMATH_CALUDE_stability_comparison_l1838_183817

/-- Represents an athlete's shooting performance -/
structure Athlete where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines stability of performance based on variance -/
def more_stable (a b : Athlete) : Prop :=
  a.variance < b.variance

theorem stability_comparison 
  (a b : Athlete) 
  (h_same_avg : a.average_score = b.average_score) 
  (h_var_a : a.variance = 0.4) 
  (h_var_b : b.variance = 2) : 
  more_stable a b :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l1838_183817


namespace NUMINAMATH_CALUDE_regular_star_points_l1838_183883

/-- An n-pointed regular star -/
structure RegularStar (n : ℕ) :=
  (edge_length : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (edge_congruent : edge_length > 0)
  (angle_A_congruent : angle_A > 0)
  (angle_B_congruent : angle_B > 0)
  (angle_difference : angle_B = angle_A + 10)
  (exterior_angle_sum : n * (angle_A + angle_B) = 360)

/-- The number of points in a regular star satisfying the given conditions is 36 -/
theorem regular_star_points : ∃ (n : ℕ), n > 0 ∧ ∃ (star : RegularStar n), n = 36 :=
sorry

end NUMINAMATH_CALUDE_regular_star_points_l1838_183883


namespace NUMINAMATH_CALUDE_problem_statement_l1838_183826

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2015 + b^2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1838_183826


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1838_183891

theorem radical_conjugate_sum_product (x y : ℝ) : 
  (x + Real.sqrt y) + (x - Real.sqrt y) = 8 ∧ 
  (x + Real.sqrt y) * (x - Real.sqrt y) = 15 →
  x + y = 5 := by sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1838_183891


namespace NUMINAMATH_CALUDE_grandfather_grandson_age_ratio_not_six_l1838_183810

theorem grandfather_grandson_age_ratio_not_six : 
  let grandson_age_now : ℕ := 12
  let grandfather_age_now : ℕ := 72
  let grandson_age_three_years_ago : ℕ := grandson_age_now - 3
  let grandfather_age_three_years_ago : ℕ := grandfather_age_now - 3
  ¬ (grandfather_age_three_years_ago = 6 * grandson_age_three_years_ago) :=
by sorry

end NUMINAMATH_CALUDE_grandfather_grandson_age_ratio_not_six_l1838_183810


namespace NUMINAMATH_CALUDE_shaded_area_circular_pattern_l1838_183832

/-- The area of the shaded region in a circular arc pattern -/
theorem shaded_area_circular_pattern (r : ℝ) (l : ℝ) : 
  r = 3 → l = 24 → (2 * l / (2 * r)) * (π * r^2 / 2) = 18 * π :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_area_circular_pattern_l1838_183832


namespace NUMINAMATH_CALUDE_system_solution_l1838_183855

theorem system_solution (x y : ℝ) 
  (h1 : x * y = -8)
  (h2 : x^2 * y + x * y^2 + 3*x + 3*y = 100) :
  x^2 + y^2 = 416 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1838_183855


namespace NUMINAMATH_CALUDE_sugar_salt_difference_l1838_183858

/-- 
Given a recipe that calls for specific amounts of ingredients and Mary's actions,
prove that the difference between the required cups of sugar and salt is 2.
-/
theorem sugar_salt_difference (sugar_required flour_required salt_required flour_added : ℕ) 
  (h1 : sugar_required = 11)
  (h2 : flour_required = 6)
  (h3 : salt_required = 9)
  (h4 : flour_added = 12) :
  sugar_required - salt_required = 2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_salt_difference_l1838_183858


namespace NUMINAMATH_CALUDE_power_zero_eq_one_neg_half_power_zero_l1838_183884

theorem power_zero_eq_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem neg_half_power_zero : (-1/2 : ℚ)^0 = 1 := by sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_neg_half_power_zero_l1838_183884


namespace NUMINAMATH_CALUDE_smallest_sum_of_two_distinct_primes_greater_than_500_l1838_183856

def is_prime (n : ℕ) : Prop := sorry

def sum_of_two_distinct_primes_greater_than_500 (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p > 500 ∧ q > 500 ∧ p ≠ q ∧ n = p + q

theorem smallest_sum_of_two_distinct_primes_greater_than_500 :
  (∀ m : ℕ, sum_of_two_distinct_primes_greater_than_500 m → m ≥ 1012) ∧
  sum_of_two_distinct_primes_greater_than_500 1012 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_two_distinct_primes_greater_than_500_l1838_183856


namespace NUMINAMATH_CALUDE_integer_list_mean_l1838_183800

theorem integer_list_mean (m : ℤ) : 
  let ones := m + 1
  let twos := m + 2
  let threes := m + 3
  let fours := m + 4
  let fives := m + 5
  let total_count := ones + twos + threes + fours + fives
  let sum := ones * 1 + twos * 2 + threes * 3 + fours * 4 + fives * 5
  (sum : ℚ) / total_count = 19 / 6 → m = 9 := by
sorry

end NUMINAMATH_CALUDE_integer_list_mean_l1838_183800


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l1838_183897

theorem perfect_square_quadratic (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 18 * x + 9 = (r * x + s)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l1838_183897


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1838_183833

/-- The inequality x^2 + ax - 2 < 0 has solutions within [2, 4] if and only if a ∈ (-∞, -1) -/
theorem inequality_solution_range (a : ℝ) :
  (∃ x ∈ Set.Icc 2 4, x^2 + a*x - 2 < 0) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1838_183833


namespace NUMINAMATH_CALUDE_edwards_purchases_cost_edwards_total_cost_l1838_183869

/-- Calculates the total cost of Edward's purchases after applying a discount -/
theorem edwards_purchases_cost (board_game_cost : ℝ) (action_figure_cost : ℝ) 
  (action_figure_count : ℕ) (puzzle_cost : ℝ) (card_deck_cost : ℝ) 
  (discount_percentage : ℝ) : ℝ :=
  let total_action_figures_cost := action_figure_cost * action_figure_count
  let discount_amount := total_action_figures_cost * (discount_percentage / 100)
  let discounted_action_figures_cost := total_action_figures_cost - discount_amount
  board_game_cost + discounted_action_figures_cost + puzzle_cost + card_deck_cost

/-- Proves that Edward's total purchase cost is $36.70 -/
theorem edwards_total_cost : 
  edwards_purchases_cost 2 7 4 6 3.5 10 = 36.7 := by
  sorry

end NUMINAMATH_CALUDE_edwards_purchases_cost_edwards_total_cost_l1838_183869


namespace NUMINAMATH_CALUDE_hyperbola_mn_value_l1838_183823

/-- Given a hyperbola with equation x²/m - y²/n = 1, eccentricity 2, and one focus at (1,0), prove that mn = 3/16 -/
theorem hyperbola_mn_value (m n : ℝ) (h1 : m * n ≠ 0) :
  (∀ x y : ℝ, x^2 / m - y^2 / n = 1) →  -- Hyperbola equation
  (∃ a b : ℝ, (x - a)^2 / m - (y - b)^2 / n = 1 ∧ ((a + 1)^2 + b^2)^(1/2) = 2) →  -- Eccentricity is 2
  (∃ x y : ℝ, x^2 / m - y^2 / n = 1 ∧ x = 1 ∧ y = 0) →  -- One focus at (1,0)
  m * n = 3/16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_mn_value_l1838_183823


namespace NUMINAMATH_CALUDE_convex_curve_triangle_inequalities_l1838_183814

/-- A convex curve in a metric space -/
class ConvexCurve (α : Type*) [MetricSpace α]

/-- The distance between two convex curves -/
def curve_distance {α : Type*} [MetricSpace α] (A B : ConvexCurve α) : ℝ := sorry

/-- Triangle inequalities for distances between convex curves -/
theorem convex_curve_triangle_inequalities
  {α : Type*} [MetricSpace α]
  (A B C : ConvexCurve α) :
  let AB := curve_distance A B
  let BC := curve_distance B C
  let AC := curve_distance A C
  (AB + BC ≥ AC) ∧ (AC + BC ≥ AB) ∧ (AB + AC ≥ BC) :=
by sorry

end NUMINAMATH_CALUDE_convex_curve_triangle_inequalities_l1838_183814


namespace NUMINAMATH_CALUDE_national_park_trees_l1838_183844

theorem national_park_trees (num_pines : ℕ) (num_redwoods : ℕ) : 
  num_pines = 600 →
  num_redwoods = num_pines + (num_pines * 20 / 100) →
  num_pines + num_redwoods = 1320 := by
sorry

end NUMINAMATH_CALUDE_national_park_trees_l1838_183844


namespace NUMINAMATH_CALUDE_smallest_a_for_integer_sqrt_8a_l1838_183812

theorem smallest_a_for_integer_sqrt_8a : 
  (∃ (a : ℕ), a > 0 ∧ ∃ (n : ℕ), n^2 = 8*a) → 
  (∀ (a : ℕ), a > 0 → (∃ (n : ℕ), n^2 = 8*a) → a ≥ 2) ∧
  (∃ (n : ℕ), n^2 = 8*2) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_integer_sqrt_8a_l1838_183812


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1838_183893

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, 2)

-- Define the line y = x
def line_y_eq_x (x : ℝ) : ℝ := x

-- Define the line l: y = kx + 1
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the dot product of vectors OP and OQ
def dot_product_OP_OQ (P Q : ℝ × ℝ) : ℝ :=
  P.1 * Q.1 + P.2 * Q.2

theorem circle_intersection_theorem (k : ℝ) :
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  (∃ c : ℝ × ℝ, c ∈ circle_C ∧ c.2 = line_y_eq_x c.1) ∧
  (∃ P Q : ℝ × ℝ, P ∈ circle_C ∧ Q ∈ circle_C ∧
    P.2 = line_l k P.1 ∧ Q.2 = line_l k Q.1) ∧
  (∃ P Q : ℝ × ℝ, P ∈ circle_C ∧ Q ∈ circle_C ∧
    P.2 = line_l k P.1 ∧ Q.2 = line_l k Q.1 ∧
    dot_product_OP_OQ P Q = -2) →
  k = 0 := by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1838_183893


namespace NUMINAMATH_CALUDE_inequalities_theorem_l1838_183877

theorem inequalities_theorem :
  (∀ a b c d : ℝ, a > b → c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a > b ∧ c > d ∧ a * c ≤ b * d) ∧
  (∃ a b c : ℝ, a < b ∧ a * c^2 ≥ b * c^2) ∧
  (∀ a b c : ℝ, a > b → b > 0 → c < 0 → c / a > c / b) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l1838_183877


namespace NUMINAMATH_CALUDE_stones_partition_exists_l1838_183870

/-- A partition of n into k parts is a list of k positive integers that sum to n. -/
def IsPartition (n k : ℕ) (partition : List ℕ) : Prop :=
  partition.length = k ∧ 
  partition.all (· > 0) ∧
  partition.sum = n

/-- A partition is similar if the maximum value is less than twice the minimum value. -/
def IsSimilarPartition (partition : List ℕ) : Prop :=
  partition.maximum? ≠ none ∧ 
  partition.minimum? ≠ none ∧ 
  (partition.maximum?.get! < 2 * partition.minimum?.get!)

theorem stones_partition_exists : 
  ∃ (partition : List ℕ), IsPartition 660 30 partition ∧ IsSimilarPartition partition := by
  sorry

end NUMINAMATH_CALUDE_stones_partition_exists_l1838_183870


namespace NUMINAMATH_CALUDE_jimin_has_greater_sum_l1838_183808

theorem jimin_has_greater_sum : 
  let jungkook_num1 : ℕ := 4
  let jungkook_num2 : ℕ := 4
  let jimin_num1 : ℕ := 3
  let jimin_num2 : ℕ := 6
  jimin_num1 + jimin_num2 > jungkook_num1 + jungkook_num2 :=
by
  sorry

end NUMINAMATH_CALUDE_jimin_has_greater_sum_l1838_183808


namespace NUMINAMATH_CALUDE_expression_equals_one_l1838_183868

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a - b + c = 0) :
  (a^2 * b^2) / ((a^2 + b*c) * (b^2 + a*c)) +
  (a^2 * c^2) / ((a^2 + b*c) * (c^2 + a*b)) +
  (b^2 * c^2) / ((b^2 + a*c) * (c^2 + a*b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1838_183868


namespace NUMINAMATH_CALUDE_price_increase_achieves_target_profit_l1838_183828

/-- Represents the supermarket's pomelo sales scenario -/
structure PomeloSales where
  initial_profit_per_kg : ℝ
  initial_daily_sales : ℝ
  price_increase : ℝ
  sales_decrease_per_yuan : ℝ
  target_daily_profit : ℝ

/-- Calculates the daily profit based on the price increase -/
def daily_profit (s : PomeloSales) : ℝ :=
  (s.initial_profit_per_kg + s.price_increase) *
  (s.initial_daily_sales - s.sales_decrease_per_yuan * s.price_increase)

/-- Theorem stating that a 5 yuan price increase achieves the target profit -/
theorem price_increase_achieves_target_profit (s : PomeloSales)
  (h1 : s.initial_profit_per_kg = 10)
  (h2 : s.initial_daily_sales = 500)
  (h3 : s.sales_decrease_per_yuan = 20)
  (h4 : s.target_daily_profit = 6000)
  (h5 : s.price_increase = 5) :
  daily_profit s = s.target_daily_profit :=
by sorry


end NUMINAMATH_CALUDE_price_increase_achieves_target_profit_l1838_183828


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_l1838_183864

/-- Represents a component of milk with its percentage -/
structure MilkComponent where
  name : String
  percentage : Float

/-- Represents a type of graph -/
inductive GraphType
  | PieChart
  | BarGraph
  | LineGraph
  | ScatterPlot

/-- Determines if a list of percentages sums to 100% (allowing for small floating-point errors) -/
def sumsToWhole (components : List MilkComponent) : Bool :=
  let sum := components.map (·.percentage) |>.sum
  sum > 99.99 && sum < 100.01

/-- Determines if a graph type is suitable for representing percentages of a whole -/
def isSuitableForPercentages (graphType : GraphType) : Bool :=
  match graphType with
  | GraphType.PieChart => true
  | _ => false

/-- Theorem: A pie chart is the most suitable graph type for representing milk components -/
theorem pie_chart_most_suitable (components : List MilkComponent) 
  (h_components : components = [
    ⟨"Water", 82⟩, 
    ⟨"Protein", 4.3⟩, 
    ⟨"Fat", 6⟩, 
    ⟨"Lactose", 7⟩, 
    ⟨"Other", 0.7⟩
  ])
  (h_sum : sumsToWhole components) :
  ∀ (graphType : GraphType), 
    isSuitableForPercentages graphType → graphType = GraphType.PieChart :=
by sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_l1838_183864


namespace NUMINAMATH_CALUDE_sum_of_odd_decreasing_function_is_negative_l1838_183885

-- Define a structure for our function properties
structure OddDecreasingFunction (f : ℝ → ℝ) : Prop where
  odd : ∀ x, f (-x) = -f x
  decreasing : ∀ x y, x < y → f x > f y

-- Main theorem
theorem sum_of_odd_decreasing_function_is_negative
  (f : ℝ → ℝ)
  (h_f : OddDecreasingFunction f)
  (α β γ : ℝ)
  (h_αβ : α + β > 0)
  (h_βγ : β + γ > 0)
  (h_γα : γ + α > 0) :
  f α + f β + f γ < 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_odd_decreasing_function_is_negative_l1838_183885


namespace NUMINAMATH_CALUDE_kalebs_clothing_l1838_183822

def total_clothing (first_load : ℕ) (num_equal_loads : ℕ) (pieces_per_equal_load : ℕ) : ℕ :=
  first_load + num_equal_loads * pieces_per_equal_load

theorem kalebs_clothing :
  total_clothing 19 5 4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_kalebs_clothing_l1838_183822


namespace NUMINAMATH_CALUDE_linda_car_rental_cost_l1838_183801

/-- Calculates the total cost of renting a car given the daily rate, mileage rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that the total cost for Linda's car rental is $165. -/
theorem linda_car_rental_cost :
  total_rental_cost 30 0.25 3 300 = 165 := by
  sorry

end NUMINAMATH_CALUDE_linda_car_rental_cost_l1838_183801
