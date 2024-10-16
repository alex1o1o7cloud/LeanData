import Mathlib

namespace NUMINAMATH_CALUDE_shirt_cost_l484_48414

theorem shirt_cost (total_cost shirt_cost coat_cost : ℝ) : 
  total_cost = 600 →
  shirt_cost = (1/3) * coat_cost →
  shirt_cost + coat_cost = total_cost →
  shirt_cost = 150 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l484_48414


namespace NUMINAMATH_CALUDE_profit_growth_equation_l484_48485

/-- Represents the profit growth of a supermarket over a 2-month period -/
theorem profit_growth_equation (initial_profit : ℝ) (final_profit : ℝ) (growth_rate : ℝ) :
  initial_profit = 5000 →
  final_profit = 7200 →
  initial_profit * (1 + growth_rate)^2 = final_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_growth_equation_l484_48485


namespace NUMINAMATH_CALUDE_cyclic_inequality_l484_48496

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y + y * z + z * x = x + y + z) :
  (1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) ≤ 1 ∧
  ((1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l484_48496


namespace NUMINAMATH_CALUDE_f_is_fraction_l484_48405

-- Define what a fraction is
def is_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (a b : ℚ), ∀ x, b ≠ 0 → f x = a / b

-- Define the specific function we're proving is a fraction
def f (x : ℚ) : ℚ := x / (x + 2)

-- Theorem statement
theorem f_is_fraction : is_fraction f := by sorry

end NUMINAMATH_CALUDE_f_is_fraction_l484_48405


namespace NUMINAMATH_CALUDE_mom_gets_eighteen_strawberries_l484_48437

def strawberries_for_mom (dozen_picked : ℕ) (eaten : ℕ) : ℕ :=
  dozen_picked * 12 - eaten

theorem mom_gets_eighteen_strawberries :
  strawberries_for_mom 2 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_mom_gets_eighteen_strawberries_l484_48437


namespace NUMINAMATH_CALUDE_stating_count_testing_methods_proof_l484_48475

/-- The number of different products -/
def total_products : ℕ := 7

/-- The number of defective products -/
def defective_products : ℕ := 4

/-- The number of non-defective products -/
def non_defective_products : ℕ := 3

/-- The test number on which the third defective product is identified -/
def third_defective_test : ℕ := 4

/-- 
  The number of testing methods where the third defective product 
  is exactly identified on the 4th test, given 7 total products 
  with 4 defective and 3 non-defective ones.
-/
def count_testing_methods : ℕ := 1080

/-- 
  Theorem stating that the number of testing methods where the third defective product 
  is exactly identified on the 4th test is equal to 1080, given the problem conditions.
-/
theorem count_testing_methods_proof : 
  count_testing_methods = 1080 ∧
  total_products = 7 ∧
  defective_products = 4 ∧
  non_defective_products = 3 ∧
  third_defective_test = 4 :=
by sorry

end NUMINAMATH_CALUDE_stating_count_testing_methods_proof_l484_48475


namespace NUMINAMATH_CALUDE_parabola_distance_sum_lower_bound_l484_48436

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point N
def N : ℝ × ℝ := (2, 2)

-- Statement of the theorem
theorem parabola_distance_sum_lower_bound :
  ∀ (M : ℝ × ℝ), parabola M.1 M.2 →
  dist M focus + dist M N ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_distance_sum_lower_bound_l484_48436


namespace NUMINAMATH_CALUDE_original_number_l484_48481

theorem original_number (x : ℚ) (h : 1 - 1/x = 5/2) : x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l484_48481


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l484_48466

theorem quadratic_inequality_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 1 ≤ 0) → (m ≥ 2 ∨ m ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l484_48466


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l484_48490

theorem quadratic_roots_nature (a b c m n : ℝ) : 
  a ≠ 0 → c ≠ 0 →
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = m ∨ x = n) →
  m * n < 0 →
  m < abs m →
  ∀ x, c * x^2 + (m - n) * a * x - a = 0 → x < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l484_48490


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l484_48495

theorem common_internal_tangent_length 
  (center_distance : ℝ) 
  (radius_small : ℝ) 
  (radius_large : ℝ) 
  (h1 : center_distance = 25) 
  (h2 : radius_small = 7) 
  (h3 : radius_large = 10) : 
  Real.sqrt ((center_distance ^ 2) - (radius_small + radius_large) ^ 2) = Real.sqrt 336 :=
sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l484_48495


namespace NUMINAMATH_CALUDE_complement_P_wrt_U_l484_48443

-- Define the sets U and P
def U : Set ℝ := Set.univ
def P : Set ℝ := Set.Ioo 0 (1/2)

-- State the theorem
theorem complement_P_wrt_U :
  (U \ P) = Set.Iic 0 ∪ Set.Ici (1/2) := by
  sorry

end NUMINAMATH_CALUDE_complement_P_wrt_U_l484_48443


namespace NUMINAMATH_CALUDE_certain_number_proof_l484_48438

theorem certain_number_proof (w : ℕ) (n : ℕ) : 
  w > 0 ∧ 
  168 ≤ w ∧
  ∃ (k : ℕ), k > 0 ∧ n * w = k * 2^5 ∧
  ∃ (l : ℕ), l > 0 ∧ n * w = l * 3^3 ∧
  ∃ (m : ℕ), m > 0 ∧ n * w = m * 14^2 →
  n = 1008 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l484_48438


namespace NUMINAMATH_CALUDE_cows_per_herd_l484_48425

theorem cows_per_herd (total_cows : ℕ) (num_herds : ℕ) (h1 : total_cows = 320) (h2 : num_herds = 8) :
  total_cows / num_herds = 40 := by
  sorry

end NUMINAMATH_CALUDE_cows_per_herd_l484_48425


namespace NUMINAMATH_CALUDE_remainder_2356912_div_8_l484_48492

theorem remainder_2356912_div_8 : 2356912 % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2356912_div_8_l484_48492


namespace NUMINAMATH_CALUDE_problem_solution_l484_48429

-- Define the set B
def B : Set ℝ := {m | ∀ x ≥ 2, x^2 - x - m > 0}

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem problem_solution :
  (B = {m | m < 2}) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B) → a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l484_48429


namespace NUMINAMATH_CALUDE_sequence_determination_l484_48448

theorem sequence_determination (p : ℕ) (hp : p.Prime ∧ p > 5) :
  ∀ (a : Fin ((p - 1) / 2) → ℕ),
  (∀ i, a i ∈ Finset.range ((p - 1) / 2 + 1) \ {0}) →
  (∀ i j, i ≠ j → ∃ r, (a i * a j) % p = r) →
  Function.Injective a :=
sorry

end NUMINAMATH_CALUDE_sequence_determination_l484_48448


namespace NUMINAMATH_CALUDE_smallest_n_for_integral_solutions_l484_48417

theorem smallest_n_for_integral_solutions :
  (∃ (n : ℕ), n > 0 ∧
    (∃ (x : ℤ), 15 * x^2 - n * x + 315 = 0) ∧
    (∀ (m : ℕ), m > 0 → m < n →
      ¬(∃ (y : ℤ), 15 * y^2 - m * y + 315 = 0))) →
  (∃ (n : ℕ), n = 150 ∧
    (∃ (x : ℤ), 15 * x^2 - n * x + 315 = 0) ∧
    (∀ (m : ℕ), m > 0 → m < n →
      ¬(∃ (y : ℤ), 15 * y^2 - m * y + 315 = 0))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integral_solutions_l484_48417


namespace NUMINAMATH_CALUDE_no_three_distinct_solutions_l484_48464

theorem no_three_distinct_solutions : 
  ¬∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    a * (a - 4) = 12 ∧ b * (b - 4) = 12 ∧ c * (c - 4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_no_three_distinct_solutions_l484_48464


namespace NUMINAMATH_CALUDE_sum_of_k_values_l484_48415

theorem sum_of_k_values : ∃ (S : Finset ℕ), 
  (∀ k ∈ S, ∃ j : ℕ, j > 0 ∧ k > 0 ∧ 1 / j + 1 / k = 1 / 4) ∧ 
  (∀ k : ℕ, k > 0 → (∃ j : ℕ, j > 0 ∧ 1 / j + 1 / k = 1 / 4) → k ∈ S) ∧
  (S.sum id = 51) := by
sorry

end NUMINAMATH_CALUDE_sum_of_k_values_l484_48415


namespace NUMINAMATH_CALUDE_factors_of_12650_l484_48478

theorem factors_of_12650 : Nat.card (Nat.divisors 12650) = 24 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_12650_l484_48478


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l484_48400

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  reflect_about_y_neg_x original_center = (-3, -8) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l484_48400


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l484_48467

theorem quadratic_inequality_minimum (b c : ℝ) : 
  (∀ x, (x^2 - (b+2)*x + c < 0) ↔ (2 < x ∧ x < 3)) →
  (∃ min : ℝ, min = 3 ∧ 
    ∀ x > 1, (x^2 - b*x + c) / (x - 1) ≥ min ∧ 
    ∃ x₀ > 1, (x₀^2 - b*x₀ + c) / (x₀ - 1) = min) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l484_48467


namespace NUMINAMATH_CALUDE_cos_equality_angle_l484_48434

theorem cos_equality_angle (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (280 * π / 180) → n = 80 := by
sorry

end NUMINAMATH_CALUDE_cos_equality_angle_l484_48434


namespace NUMINAMATH_CALUDE_milk_production_days_l484_48458

variable (x : ℝ)

def initial_cows := x + 2
def initial_milk := x + 4
def initial_days := x + 3
def new_cows := x + 5
def new_milk := x + 9

theorem milk_production_days : 
  (initial_cows x * initial_days x) / initial_milk x * new_milk x / new_cows x = 
  (new_milk x * initial_cows x * initial_days x) / (new_cows x * initial_milk x) := by
sorry

end NUMINAMATH_CALUDE_milk_production_days_l484_48458


namespace NUMINAMATH_CALUDE_cubic_function_unique_determination_l484_48486

def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem cubic_function_unique_determination 
  (f : ℝ → ℝ) 
  (h_cubic : ∃ a b c d : ℝ, ∀ x, f x = cubic_function a b c d x) 
  (h_max : f 1 = 4 ∧ (deriv f) 1 = 0)
  (h_min : f 3 = 0 ∧ (deriv f) 3 = 0)
  (h_origin : f 0 = 0) :
  ∀ x, f x = x^3 - 6*x^2 + 9*x :=
sorry

end NUMINAMATH_CALUDE_cubic_function_unique_determination_l484_48486


namespace NUMINAMATH_CALUDE_sin_75_degrees_l484_48457

theorem sin_75_degrees : Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_degrees_l484_48457


namespace NUMINAMATH_CALUDE_number_of_people_l484_48476

theorem number_of_people (average_age : ℝ) (youngest_age : ℝ) (average_age_at_birth : ℝ) :
  average_age = 30 →
  youngest_age = 3 →
  average_age_at_birth = 27 →
  ∃ n : ℕ, n = 7 ∧ 
    average_age * n = youngest_age + average_age_at_birth * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_number_of_people_l484_48476


namespace NUMINAMATH_CALUDE_no_rational_solution_l484_48488

theorem no_rational_solution :
  ¬ ∃ (x y z t : ℚ) (n : ℕ), (x + y * Real.sqrt 2) ^ (2 * n) + (z + t * Real.sqrt 2) ^ (2 * n) = 5 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l484_48488


namespace NUMINAMATH_CALUDE_housing_price_growth_equation_l484_48420

/-- 
Given:
- initial_price: The initial housing price in January 2016
- final_price: The final housing price in March 2016
- x: The average monthly growth rate over the two-month period

Prove that the equation initial_price * (1 + x)² = final_price holds.
-/
theorem housing_price_growth_equation 
  (initial_price final_price : ℝ) 
  (x : ℝ) 
  (h_initial : initial_price = 8300)
  (h_final : final_price = 8700) :
  initial_price * (1 + x)^2 = final_price := by
sorry

end NUMINAMATH_CALUDE_housing_price_growth_equation_l484_48420


namespace NUMINAMATH_CALUDE_add_fractions_l484_48452

theorem add_fractions : (3 : ℚ) / 4 + (5 : ℚ) / 6 = (19 : ℚ) / 12 := by sorry

end NUMINAMATH_CALUDE_add_fractions_l484_48452


namespace NUMINAMATH_CALUDE_ad_length_l484_48491

-- Define the points
variable (A B C D M : Point)

-- Define the length function
variable (length : Point → Point → ℝ)

-- State the conditions
variable (trisect : length A B = length B C ∧ length B C = length C D)
variable (midpoint : length A M = length M D)
variable (mc_length : length M C = 10)

-- Theorem statement
theorem ad_length : length A D = 60 := by sorry

end NUMINAMATH_CALUDE_ad_length_l484_48491


namespace NUMINAMATH_CALUDE_average_age_of_first_seven_students_l484_48418

theorem average_age_of_first_seven_students 
  (total_students : Nat) 
  (average_age_all : ℚ) 
  (second_group_size : Nat) 
  (average_age_second_group : ℚ) 
  (age_last_student : ℚ) 
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : second_group_size = 7)
  (h4 : average_age_second_group = 16)
  (h5 : age_last_student = 15) :
  let first_group_size := total_students - second_group_size - 1
  let total_age := average_age_all * total_students
  let second_group_total_age := average_age_second_group * second_group_size
  let first_group_total_age := total_age - second_group_total_age - age_last_student
  first_group_total_age / first_group_size = 14 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_first_seven_students_l484_48418


namespace NUMINAMATH_CALUDE_stationery_box_cost_l484_48427

/-- The cost of a single stationery box in yuan -/
def unit_price : ℕ := 23

/-- The number of stationery boxes to be purchased -/
def quantity : ℕ := 3

/-- The total cost of purchasing the stationery boxes -/
def total_cost : ℕ := unit_price * quantity

theorem stationery_box_cost : total_cost = 69 := by
  sorry

end NUMINAMATH_CALUDE_stationery_box_cost_l484_48427


namespace NUMINAMATH_CALUDE_max_pieces_3x3_cake_l484_48446

/-- Represents a rectangular cake -/
structure Cake where
  rows : ℕ
  cols : ℕ

/-- Represents a straight cut on the cake -/
structure Cut where
  max_intersections : ℕ

/-- Calculates the maximum number of pieces after one cut -/
def max_pieces_after_cut (cake : Cake) (cut : Cut) : ℕ :=
  2 * cut.max_intersections + 4

/-- Theorem: For a 3x3 cake, the maximum number of pieces after one cut is 14 -/
theorem max_pieces_3x3_cake (cake : Cake) (cut : Cut) :
  cake.rows = 3 ∧ cake.cols = 3 ∧ cut.max_intersections = 5 →
  max_pieces_after_cut cake cut = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_3x3_cake_l484_48446


namespace NUMINAMATH_CALUDE_probability_between_R_and_S_l484_48462

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PR and PQ = 8RS,
    the probability that a randomly selected point on PQ is between R and S is 5/8 -/
theorem probability_between_R_and_S (P Q R S : ℝ) (h1 : P < R) (h2 : R < S) (h3 : S < Q)
    (h4 : Q - P = 4 * (R - P)) (h5 : Q - P = 8 * (S - R)) :
    (S - R) / (Q - P) = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_between_R_and_S_l484_48462


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l484_48421

-- Define the quadratic function
def f (x : ℝ) := 2 * x^2 + 4 * x - 6

-- Define the solution set
def solution_set := {x : ℝ | f x < 0}

-- State the theorem
theorem quadratic_inequality_solution :
  solution_set = Set.Ioo (-3 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l484_48421


namespace NUMINAMATH_CALUDE_peach_pies_count_l484_48482

/-- Given a total of 30 pies distributed among apple, blueberry, and peach flavors
    in the ratio 3:2:5, prove that the number of peach pies is 15. -/
theorem peach_pies_count (total_pies : ℕ) (apple_ratio blueberry_ratio peach_ratio : ℕ) :
  total_pies = 30 →
  apple_ratio = 3 →
  blueberry_ratio = 2 →
  peach_ratio = 5 →
  peach_ratio * (total_pies / (apple_ratio + blueberry_ratio + peach_ratio)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_peach_pies_count_l484_48482


namespace NUMINAMATH_CALUDE_permutation_problem_l484_48409

theorem permutation_problem (n : ℕ) : (n * (n - 1) = 132) ↔ (n = 12) := by sorry

end NUMINAMATH_CALUDE_permutation_problem_l484_48409


namespace NUMINAMATH_CALUDE_circle_intersection_and_tangent_line_l484_48445

-- Define the lines and circles
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def O (x y : ℝ) : Prop := x^2 + y^2 = 4
def l₂ (x y : ℝ) : Prop := y - 2 = 4/3 * (x + 1)
def M_center_line (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the properties of circle M
def M (x y : ℝ) : Prop := (x - 8/3)^2 + (y - 4/3)^2 = 100/9

-- Theorem statement
theorem circle_intersection_and_tangent_line 
  (h₁ : ∀ x y, l₁ x y → l₂ x y → (x = -1 ∧ y = 2)) 
  (h₂ : ∃ x y, M x y ∧ M_center_line x y) 
  (h₃ : ∃ x y, M x y ∧ l₂ x y) 
  (h₄ : ∃ k, k > 0 ∧ ∀ x y, M x y → l₁ x y → 
    (∃ a₁ a₂, a₁ > 0 ∧ a₂ > 0 ∧ a₁ / a₂ = 2 ∧ a₁ + a₂ = 2 * π * k)) :
  (∃ x y, O x y ∧ l₁ x y ∧ 
    (∃ x' y', O x' y' ∧ l₁ x' y' ∧ (x - x')^2 + (y - y')^2 = 12)) ∧
  (∀ x y, M x y ↔ (x - 8/3)^2 + (y - 4/3)^2 = 100/9) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_tangent_line_l484_48445


namespace NUMINAMATH_CALUDE_training_completion_time_l484_48447

/-- Calculates the number of days required to complete a training regimen. -/
def trainingDays (totalHours : ℕ) (multiplicationMinutes : ℕ) (divisionMinutes : ℕ) : ℕ :=
  let totalMinutes := totalHours * 60
  let dailyMinutes := multiplicationMinutes + divisionMinutes
  totalMinutes / dailyMinutes

/-- Proves that given the specified training schedule, it takes 10 days to complete the training. -/
theorem training_completion_time :
  trainingDays 5 10 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_training_completion_time_l484_48447


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l484_48440

theorem polynomial_divisibility (p' q' : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x - 2) = 0 → x^5 - x^4 + x^3 - p' * x^2 + q' * x - 6 = 0) →
  p' = 0 ∧ q' = -9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l484_48440


namespace NUMINAMATH_CALUDE_parallel_lines_l484_48423

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are coincident -/
def coincident (l1 l2 : Line) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b ∧ l1.c = k * l2.c

/-- The main theorem -/
theorem parallel_lines (a : ℝ) : 
  let l1 : Line := ⟨a, 3, a^2 - 5⟩
  let l2 : Line := ⟨1, a - 2, 4⟩
  (parallel l1 l2 ∧ ¬coincident l1 l2) ↔ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_l484_48423


namespace NUMINAMATH_CALUDE_well_digging_rate_l484_48406

/-- The hourly rate paid to workers for digging a well --/
def hourly_rate (total_payment : ℚ) (num_workers : ℕ) (hours_day1 hours_day2 hours_day3 : ℕ) : ℚ :=
  total_payment / (num_workers * (hours_day1 + hours_day2 + hours_day3))

/-- Theorem stating that under the given conditions, the hourly rate is $10 --/
theorem well_digging_rate : 
  hourly_rate 660 2 10 8 15 = 10 := by
  sorry


end NUMINAMATH_CALUDE_well_digging_rate_l484_48406


namespace NUMINAMATH_CALUDE_price_reduction_achieves_profit_l484_48419

/-- Represents the daily sales and profit scenario of a store --/
structure StoreSales where
  initial_sales : ℕ := 20
  initial_profit_per_item : ℝ := 40
  sales_increase_rate : ℝ := 2
  min_profit_per_item : ℝ := 25

/-- Calculates the daily sales after a price reduction --/
def daily_sales (s : StoreSales) (price_reduction : ℝ) : ℝ :=
  s.initial_sales + s.sales_increase_rate * price_reduction

/-- Calculates the profit per item after a price reduction --/
def profit_per_item (s : StoreSales) (price_reduction : ℝ) : ℝ :=
  s.initial_profit_per_item - price_reduction

/-- Calculates the total daily profit after a price reduction --/
def daily_profit (s : StoreSales) (price_reduction : ℝ) : ℝ :=
  (daily_sales s price_reduction) * (profit_per_item s price_reduction)

/-- Theorem stating that a price reduction of 10 achieves the desired profit --/
theorem price_reduction_achieves_profit (s : StoreSales) :
  daily_profit s 10 = 1200 ∧ profit_per_item s 10 ≥ s.min_profit_per_item := by
  sorry


end NUMINAMATH_CALUDE_price_reduction_achieves_profit_l484_48419


namespace NUMINAMATH_CALUDE_max_value_of_f_l484_48468

def f (x : ℝ) (a : ℝ) := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f x a ≥ f y a) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l484_48468


namespace NUMINAMATH_CALUDE_power_multiplication_l484_48493

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l484_48493


namespace NUMINAMATH_CALUDE_tan_theta_value_l484_48465

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_theta_value (θ : Real) (h : ∃ (t : Real), t > 0 ∧ t * (-Real.sqrt 3 / 2) = Real.cos θ ∧ t * (1 / 2) = Real.sin θ) : 
  Real.tan θ = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l484_48465


namespace NUMINAMATH_CALUDE_hammond_marble_weight_l484_48499

/-- The weight of Hammond's marble statues and discarded marble -/
structure MarbleStatues where
  first_statue : ℕ
  second_statue : ℕ
  remaining_statues : ℕ
  discarded_marble : ℕ

/-- The initial weight of the marble block -/
def initial_weight (m : MarbleStatues) : ℕ :=
  m.first_statue + m.second_statue + 2 * m.remaining_statues + m.discarded_marble

/-- Theorem stating the initial weight of Hammond's marble block -/
theorem hammond_marble_weight :
  ∃ (m : MarbleStatues),
    m.first_statue = 10 ∧
    m.second_statue = 18 ∧
    m.remaining_statues = 15 ∧
    m.discarded_marble = 22 ∧
    initial_weight m = 80 := by
  sorry

end NUMINAMATH_CALUDE_hammond_marble_weight_l484_48499


namespace NUMINAMATH_CALUDE_expression_equivalence_l484_48463

theorem expression_equivalence : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
  (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l484_48463


namespace NUMINAMATH_CALUDE_m_range_l484_48498

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (2*m - 2)*x + 3

-- Define the proposition p
def p (m : ℝ) : Prop := ∀ x < 0, ∀ y < x, f m x > f m y

-- Define the proposition q
def q (m : ℝ) : Prop := ∀ x, x^2 - 4*x + 1 - m > 0

-- State the theorem
theorem m_range :
  (∀ m : ℝ, p m → m ≤ 1) →
  (∀ m : ℝ, q m → m < -3) →
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  ∃ a b : ℝ, a = -3 ∧ b = 1 ∧ ∀ m : ℝ, a ≤ m ∧ m ≤ b :=
sorry

end NUMINAMATH_CALUDE_m_range_l484_48498


namespace NUMINAMATH_CALUDE_whole_number_between_fractions_l484_48422

theorem whole_number_between_fractions (N : ℤ) :
  (3.5 < (N : ℚ) / 5 ∧ (N : ℚ) / 5 < 4.5) ↔ (N = 18 ∨ N = 19 ∨ N = 20 ∨ N = 21 ∨ N = 22) :=
by sorry

end NUMINAMATH_CALUDE_whole_number_between_fractions_l484_48422


namespace NUMINAMATH_CALUDE_tangent_circles_theorem_l484_48428

/-- A configuration of three pairwise tangent circles where the centers form a right triangle -/
structure TangentCircles where
  r1 : ℝ  -- radius of the smallest circle
  r2 : ℝ  -- radius of the medium circle
  r3 : ℝ  -- radius of the largest circle
  h1 : 0 < r1 ∧ 0 < r2 ∧ 0 < r3  -- all radii are positive
  h2 : r1 < r2 ∧ r2 < r3  -- circles are ordered by size
  h3 : (r1 + r2)^2 + (r1 + r3)^2 = (r2 + r3)^2  -- centers form a right triangle

/-- The theorem stating that given two circles with radii 4 and 6, the third circle has radius 2 -/
theorem tangent_circles_theorem (tc : TangentCircles) (h4 : tc.r2 = 4) (h5 : tc.r3 = 6) : tc.r1 = 2 := by
  sorry

#check tangent_circles_theorem

end NUMINAMATH_CALUDE_tangent_circles_theorem_l484_48428


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l484_48497

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = (2 + Real.sqrt 2) / 2 ∧ 2 * x₁^2 = 4 * x₁ - 1) ∧
  (x₂ = (2 - Real.sqrt 2) / 2 ∧ 2 * x₂^2 = 4 * x₂ - 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l484_48497


namespace NUMINAMATH_CALUDE_area_of_right_triangle_l484_48416

-- Define the right triangle ABC
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  angleABC : ℝ

-- Define the conditions of the triangle
def triangle : RightTriangle := {
  AB := 8,
  AC := 10,
  angleABC := 90
}

-- Theorem statement
theorem area_of_right_triangle (t : RightTriangle)
  (h1 : t.AB = 8)
  (h2 : t.AC = 10)
  (h3 : t.angleABC = 90) :
  (1 / 2) * t.AB * Real.sqrt (t.AC^2 - t.AB^2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_area_of_right_triangle_l484_48416


namespace NUMINAMATH_CALUDE_unique_solution_iff_k_zero_l484_48403

/-- 
Theorem: The pair of equations y = x^2 and y = 2x^2 + k have exactly one solution 
if and only if k = 0.
-/
theorem unique_solution_iff_k_zero (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 ∧ p.2 = 2*p.1^2 + k) ↔ k = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_k_zero_l484_48403


namespace NUMINAMATH_CALUDE_small_bottles_initial_count_small_bottles_initial_count_proof_l484_48410

theorem small_bottles_initial_count : ℝ → Prop :=
  fun S =>
    let big_bottles : ℝ := 12000
    let small_bottles_remaining_ratio : ℝ := 0.85
    let big_bottles_remaining_ratio : ℝ := 0.82
    let total_remaining : ℝ := 14090
    S * small_bottles_remaining_ratio + big_bottles * big_bottles_remaining_ratio = total_remaining →
    S = 5000

-- The proof goes here
theorem small_bottles_initial_count_proof : small_bottles_initial_count 5000 := by
  sorry

end NUMINAMATH_CALUDE_small_bottles_initial_count_small_bottles_initial_count_proof_l484_48410


namespace NUMINAMATH_CALUDE_smallest_multiple_l484_48471

theorem smallest_multiple (n : ℕ) : n = 481 ↔ 
  n > 0 ∧ 
  (∃ k : ℤ, n = 37 * k) ∧ 
  (∃ m : ℤ, n - 7 = 97 * m) ∧ 
  (∀ x : ℕ, x > 0 ∧ (∃ k : ℤ, x = 37 * k) ∧ (∃ m : ℤ, x - 7 = 97 * m) → x ≥ n) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l484_48471


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l484_48426

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l484_48426


namespace NUMINAMATH_CALUDE_line_inclination_angle_l484_48489

def angle_of_inclination (x y : ℝ → ℝ) : ℝ := by sorry

theorem line_inclination_angle (t : ℝ) :
  let x := λ t : ℝ => 1 + Real.sqrt 3 * t
  let y := λ t : ℝ => 3 - 3 * t
  angle_of_inclination x y = 120 * π / 180 := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l484_48489


namespace NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l484_48473

/-- Custom binary operation ⊗ -/
def otimes (a b : ℝ) : ℝ := a^2 - abs b

/-- Theorem stating that (-2) ⊗ (-1) = 3 -/
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l484_48473


namespace NUMINAMATH_CALUDE_block_tower_combinations_l484_48456

theorem block_tower_combinations :
  let initial_blocks : ℕ := 35
  let final_blocks : ℕ := 65
  let additional_blocks : ℕ := final_blocks - initial_blocks
  ∃! n : ℕ, n = (additional_blocks + 1) ∧
    n = (Finset.filter (fun p : ℕ × ℕ => p.1 + p.2 = additional_blocks)
      (Finset.product (Finset.range (additional_blocks + 1)) (Finset.range (additional_blocks + 1)))).card :=
by sorry

end NUMINAMATH_CALUDE_block_tower_combinations_l484_48456


namespace NUMINAMATH_CALUDE_correct_calculation_l484_48449

theorem correct_calculation : (-2)^3 + 6 / ((1/2) - (1/3)) = 28 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l484_48449


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l484_48413

theorem stratified_sampling_size (undergrads : ℕ) (masters : ℕ) (doctorates : ℕ) 
  (doctoral_sample : ℕ) (n : ℕ) : 
  undergrads = 12000 →
  masters = 1000 →
  doctorates = 200 →
  doctoral_sample = 20 →
  n = (undergrads + masters + doctorates) * doctoral_sample / doctorates →
  n = 1320 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l484_48413


namespace NUMINAMATH_CALUDE_incorrect_inequality_l484_48439

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬ (-2 * a > -2 * b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l484_48439


namespace NUMINAMATH_CALUDE_circle_through_intersections_and_tangent_to_line_l484_48461

/-- Given two circles and a line, prove that a specific circle passes through 
    the intersection points of the given circles and is tangent to the given line. -/
theorem circle_through_intersections_and_tangent_to_line :
  let C₁ : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 = 4
  let C₂ : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 - 2*x - 4*y + 4 = 0
  let l : ℝ × ℝ → Prop := λ (x, y) ↦ x + 2*y = 0
  let result_circle : ℝ × ℝ → Prop := λ (x, y) ↦ (x - 1/2)^2 + (y - 1)^2 = 5/4
  
  (∀ p, C₁ p ∧ C₂ p → result_circle p) ∧ 
  (∃ unique_p, l unique_p ∧ result_circle unique_p ∧ 
    ∀ q, l q ∧ result_circle q → q = unique_p) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_through_intersections_and_tangent_to_line_l484_48461


namespace NUMINAMATH_CALUDE_train_speed_l484_48451

/-- The average speed of a train without stoppages, given its speed with stoppages and stop time -/
theorem train_speed (speed_with_stops : ℝ) (stop_time : ℝ) : 
  speed_with_stops = 200 → stop_time = 20 → 
  (speed_with_stops * 60) / (60 - stop_time) = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l484_48451


namespace NUMINAMATH_CALUDE_sequence_term_equality_l484_48483

theorem sequence_term_equality (n : ℕ) : 
  2 * Real.log 5 + Real.log 3 = Real.log (4 * 19 - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_equality_l484_48483


namespace NUMINAMATH_CALUDE_browns_house_number_l484_48453

theorem browns_house_number :
  ∃! (n t : ℕ),
    20 < t ∧ t < 500 ∧
    1 ≤ n ∧ n ≤ t ∧
    n * (n + 1) = t * (t + 1) / 2 ∧
    n = 84 := by
  sorry

end NUMINAMATH_CALUDE_browns_house_number_l484_48453


namespace NUMINAMATH_CALUDE_triangle_1234_l484_48441

/-- Define the operation △ -/
def triangle (n m : ℕ) : ℕ := sorry

/-- Axiom for the first condition -/
axiom triangle_1 {a b c d : ℕ} (h1 : 0 < a ∧ a < 10) (h2 : 0 < b ∧ b < 10) 
  (h3 : 0 < c ∧ c < 10) (h4 : 0 < d ∧ d < 10) : 
  triangle (a * 1000 + b * 100 + c * 10 + d) 1 = b * 1000 + c * 100 + a * 10 + d

/-- Axiom for the second condition -/
axiom triangle_2 {a b c d : ℕ} (h1 : 0 < a ∧ a < 10) (h2 : 0 < b ∧ b < 10) 
  (h3 : 0 < c ∧ c < 10) (h4 : 0 < d ∧ d < 10) : 
  triangle (a * 1000 + b * 100 + c * 10 + d) 2 = c * 1000 + d * 100 + a * 10 + b

/-- The main theorem to prove -/
theorem triangle_1234 : triangle (triangle 1234 1) 2 = 3412 := by sorry

end NUMINAMATH_CALUDE_triangle_1234_l484_48441


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l484_48477

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  h_d_nonzero : d ≠ 0

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1 : ℝ) * seq.d

theorem arithmetic_geometric_ratio 
  (seq : ArithmeticSequence)
  (h_geom : seq.nthTerm 7 ^ 2 = seq.nthTerm 4 * seq.nthTerm 16) :
  ∃ q : ℝ, q ^ 2 = 3 ∧ 
    (seq.nthTerm 7 / seq.nthTerm 4 = q ∨ seq.nthTerm 7 / seq.nthTerm 4 = -q) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l484_48477


namespace NUMINAMATH_CALUDE_radian_measure_of_negative_120_degrees_l484_48450

theorem radian_measure_of_negative_120_degrees :
  let degree_to_radian (d : ℝ) := d * (π / 180)
  degree_to_radian (-120) = -(2 * π / 3) := by sorry

end NUMINAMATH_CALUDE_radian_measure_of_negative_120_degrees_l484_48450


namespace NUMINAMATH_CALUDE_exists_area_preserving_projection_l484_48469

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the concept of a triangle
structure Triangle (P : Plane) :=
  (area : ℝ)

-- Define the concept of parallel projection
def parallel_projection (P Q : Plane) (T : Triangle P) : Triangle Q := sorry

-- Theorem statement
theorem exists_area_preserving_projection
  (P Q : Plane)
  (intersect : P ≠ Q)
  (T : Triangle P) :
  ∃ (proj : Triangle Q), proj = parallel_projection P Q T ∧ proj.area = T.area :=
sorry

end NUMINAMATH_CALUDE_exists_area_preserving_projection_l484_48469


namespace NUMINAMATH_CALUDE_museum_ticket_price_l484_48412

theorem museum_ticket_price 
  (group_size : ℕ) 
  (total_paid : ℚ) 
  (tax_rate : ℚ) 
  (h1 : group_size = 25) 
  (h2 : total_paid = 945) 
  (h3 : tax_rate = 5 / 100) : 
  ∃ (face_value : ℚ), 
    face_value = 36 ∧ 
    total_paid = group_size * face_value * (1 + tax_rate) := by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_price_l484_48412


namespace NUMINAMATH_CALUDE_problem_solution_l484_48480

theorem problem_solution : (-1 : ℚ)^51 + 2^(4^2 + 5^2 - 7^2) = -127/128 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l484_48480


namespace NUMINAMATH_CALUDE_sixth_number_in_sequence_l484_48433

theorem sixth_number_in_sequence (numbers : List ℝ) 
  (h_count : numbers.length = 11)
  (h_sum_all : numbers.sum = 660)
  (h_sum_first_six : (numbers.take 6).sum = 588)
  (h_sum_last_six : (numbers.drop 5).sum = 390) :
  numbers[5] = 159 :=
by sorry

end NUMINAMATH_CALUDE_sixth_number_in_sequence_l484_48433


namespace NUMINAMATH_CALUDE_bella_steps_l484_48432

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℕ := 15840

/-- Bella's speed relative to Ella's -/
def speed_ratio : ℚ := 1 / 3

/-- The number of feet Bella covers in one step -/
def feet_per_step : ℕ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 1320

theorem bella_steps :
  distance * speed_ratio / (1 + speed_ratio) / feet_per_step = steps_taken := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_l484_48432


namespace NUMINAMATH_CALUDE_smallest_angle_SQR_l484_48430

-- Define the angles
def angle_PQR : ℝ := 40
def angle_PQS : ℝ := 28

-- Theorem statement
theorem smallest_angle_SQR : 
  let angle_SQR := angle_PQR - angle_PQS
  angle_SQR = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_SQR_l484_48430


namespace NUMINAMATH_CALUDE_sum_of_fractions_l484_48460

theorem sum_of_fractions (a b c : ℝ) (h : a * b * c = 1) :
  a / (a * b + a + 1) + b / (b * c + b + 1) + c / (c * a + c + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l484_48460


namespace NUMINAMATH_CALUDE_chess_game_draw_fraction_l484_48459

theorem chess_game_draw_fraction 
  (ellen_wins : ℚ) 
  (john_wins : ℚ) 
  (h1 : ellen_wins = 4/9) 
  (h2 : john_wins = 2/9) : 
  1 - (ellen_wins + john_wins) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_chess_game_draw_fraction_l484_48459


namespace NUMINAMATH_CALUDE_disjunction_true_l484_48442

theorem disjunction_true (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_true_l484_48442


namespace NUMINAMATH_CALUDE_find_A_l484_48402

theorem find_A : ∀ A : ℝ, 10 + A = 15 → A = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l484_48402


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l484_48444

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = 1 is √2 -/
theorem hyperbola_eccentricity : 
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - y^2 = 1}
  ∃ e : ℝ, e = Real.sqrt 2 ∧ 
    ∀ (a b c : ℝ), 
      (a = 1 ∧ b = 1 ∧ c^2 = a^2 + b^2) → 
      e = c / a :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l484_48444


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l484_48494

theorem circle_diameter_ratio (C D : Real) :
  -- Circle C is inside circle D
  C < D →
  -- Diameter of circle D is 20 cm
  D = 10 →
  -- Ratio of shaded area to area of circle C is 7:1
  (π * D^2 - π * C^2) / (π * C^2) = 7 →
  -- The diameter of circle C is 5√5 cm
  2 * C = 5 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l484_48494


namespace NUMINAMATH_CALUDE_bennetts_brothers_l484_48470

/-- Given that Aaron has four brothers and Bennett's number of brothers is two less than twice
    the number of Aaron's brothers, prove that Bennett has 6 brothers. -/
theorem bennetts_brothers (aaron_brothers : ℕ) (bennett_brothers : ℕ) 
    (h1 : aaron_brothers = 4)
    (h2 : bennett_brothers = 2 * aaron_brothers - 2) : 
  bennett_brothers = 6 := by
  sorry

end NUMINAMATH_CALUDE_bennetts_brothers_l484_48470


namespace NUMINAMATH_CALUDE_circle_radius_is_sqrt_2_l484_48487

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 1 = 0

-- Theorem statement
theorem circle_radius_is_sqrt_2 :
  ∃ (h k : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_sqrt_2_l484_48487


namespace NUMINAMATH_CALUDE_nates_run_ratio_l484_48404

theorem nates_run_ratio (total_distance field_length rest_distance : ℕ) 
  (h1 : total_distance = 1172)
  (h2 : field_length = 168)
  (h3 : rest_distance = 500)
  (h4 : ∃ k : ℕ, total_distance - rest_distance = k * field_length) :
  (total_distance - rest_distance) / field_length = 4 := by
sorry

end NUMINAMATH_CALUDE_nates_run_ratio_l484_48404


namespace NUMINAMATH_CALUDE_line_intersects_circle_l484_48479

/-- The line y - 1 = k(x - 1) intersects the circle x^2 + y^2 - 2y = 0 for any real number k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  (y - 1 = k * (x - 1)) ∧ (x^2 + y^2 - 2*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l484_48479


namespace NUMINAMATH_CALUDE_museum_paintings_l484_48431

theorem museum_paintings (initial : ℕ) (left : ℕ) (removed : ℕ) :
  initial = 1795 →
  left = 1322 →
  removed = initial - left →
  removed = 473 :=
by sorry

end NUMINAMATH_CALUDE_museum_paintings_l484_48431


namespace NUMINAMATH_CALUDE_sum_of_first_60_digits_l484_48401

/-- The decimal representation of 1/2222 -/
def decimal_rep : ℚ := 1 / 2222

/-- The repeating sequence in the decimal representation -/
def repeating_sequence : List ℕ := [0, 0, 0, 4, 5]

/-- The length of the repeating sequence -/
def sequence_length : ℕ := repeating_sequence.length

/-- The sum of digits in one repetition of the sequence -/
def sequence_sum : ℕ := repeating_sequence.sum

/-- The number of complete repetitions in the first 60 digits -/
def num_repetitions : ℕ := 60 / sequence_length

theorem sum_of_first_60_digits : 
  (num_repetitions * sequence_sum = 108) := by sorry

end NUMINAMATH_CALUDE_sum_of_first_60_digits_l484_48401


namespace NUMINAMATH_CALUDE_tower_configurations_mod_1000_l484_48411

/-- Recursively calculates the number of valid tower configurations for m cubes -/
def tower_configurations (m : ℕ) : ℕ :=
  match m with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | n + 1 => (n + 2) * tower_configurations n

/-- Represents the conditions for building towers with 9 cubes -/
def valid_tower_conditions (n : ℕ) : Prop :=
  n ≤ 9 ∧ 
  ∀ k : ℕ, k ≤ n → ∃ cube : ℕ, cube = k ∧
  ∀ i j : ℕ, i < j → j - i ≤ 3

/-- The main theorem stating that the number of different towers is congruent to 200 modulo 1000 -/
theorem tower_configurations_mod_1000 :
  valid_tower_conditions 9 →
  tower_configurations 9 % 1000 = 200 := by
  sorry


end NUMINAMATH_CALUDE_tower_configurations_mod_1000_l484_48411


namespace NUMINAMATH_CALUDE_second_coaster_speed_l484_48484

/-- The speed of the second rollercoaster given the speeds of the other coasters and the average speed -/
theorem second_coaster_speed 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : x₁ = 50)
  (h3 : x₃ = 73)
  (h4 : x₄ = 70)
  (h5 : x₅ = 40)
  (avg : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 59) :
  x₂ = 62 := by
  sorry

end NUMINAMATH_CALUDE_second_coaster_speed_l484_48484


namespace NUMINAMATH_CALUDE_angle_B_measure_l484_48455

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b*cos(C) + (2a+c)*cos(B) = 0, then the measure of angle B is 2π/3 -/
theorem angle_B_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b * Real.cos C + (2 * a + c) * Real.cos B = 0 →
  B = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l484_48455


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l484_48407

theorem opposite_of_negative_six : ∃ x : ℤ, ((-6 : ℤ) + x = 0) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l484_48407


namespace NUMINAMATH_CALUDE_total_amount_after_two_years_l484_48435

/-- Calculates the total amount returned after compound interest --/
def totalAmountAfterCompoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) (compoundInterest : ℝ) : ℝ :=
  principal + compoundInterest

/-- Theorem stating the total amount returned after two years of compound interest --/
theorem total_amount_after_two_years 
  (principal : ℝ) 
  (rate : ℝ) 
  (compoundInterest : ℝ) 
  (h1 : rate = 0.05) 
  (h2 : compoundInterest = 246) 
  (h3 : principal * ((1 + rate)^2 - 1) = compoundInterest) : 
  totalAmountAfterCompoundInterest principal rate 2 compoundInterest = 2646 := by
  sorry

#check total_amount_after_two_years

end NUMINAMATH_CALUDE_total_amount_after_two_years_l484_48435


namespace NUMINAMATH_CALUDE_green_candies_count_l484_48474

/-- Proves the number of green candies in a bag given the number of blue and red candies and the probability of picking a blue candy. -/
theorem green_candies_count (blue : ℕ) (red : ℕ) (prob_blue : ℚ) (green : ℕ) : 
  blue = 3 → red = 4 → prob_blue = 1/4 → 
  (blue : ℚ) / ((green : ℚ) + (blue : ℚ) + (red : ℚ)) = prob_blue →
  green = 5 := by sorry

end NUMINAMATH_CALUDE_green_candies_count_l484_48474


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_l484_48472

def n : ℕ := 1209600000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a > b ∧ b > c ∧ c > d ∧ d > e ∧
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    ∀ (x : ℕ), x ∣ n → (x ≤ e ∨ x = d ∨ x = c ∨ x = b ∨ x = a ∨ x = n))

theorem fifth_largest_divisor :
  is_fifth_largest_divisor 75600000 :=
sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_l484_48472


namespace NUMINAMATH_CALUDE_fraction_comparison_geometric_sum_comparison_l484_48424

theorem fraction_comparison (α β : ℝ) (hα : α = 1.00000000004) (hβ : β = 1.00000000002) :
  (1 + β) / (1 + β + β^2) > (1 + α) / (1 + α + α^2) := by sorry

theorem geometric_sum_comparison {a b : ℝ} {n : ℕ} (hab : a > b) (hb : b > 0) (hn : n > 0) :
  (b^n - 1) / (b^(n+1) - 1) > (a^n - 1) / (a^(n+1) - 1) := by sorry

end NUMINAMATH_CALUDE_fraction_comparison_geometric_sum_comparison_l484_48424


namespace NUMINAMATH_CALUDE_division_of_fraction_by_integer_l484_48454

theorem division_of_fraction_by_integer :
  (3 : ℚ) / 7 / 4 = 3 / 28 := by sorry

end NUMINAMATH_CALUDE_division_of_fraction_by_integer_l484_48454


namespace NUMINAMATH_CALUDE_double_root_condition_l484_48408

theorem double_root_condition (m : ℝ) :
  (∃! x : ℝ, (x - 3) / (x - 1) = m / (x - 1) ∧ x ≠ 1) ↔ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_double_root_condition_l484_48408
