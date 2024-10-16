import Mathlib

namespace NUMINAMATH_CALUDE_median_mode_difference_l2108_210856

def data : List ℕ := [42, 44, 44, 45, 45, 45, 51, 51, 51, 53, 53, 53, 62, 64, 66, 66, 67, 68, 70, 74, 74, 75, 75, 76, 81, 82, 85, 88, 89, 89]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem median_mode_difference : 
  |median data - (mode data : ℚ)| = 23 := by sorry

end NUMINAMATH_CALUDE_median_mode_difference_l2108_210856


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l2108_210851

theorem quadratic_root_implies_a_value (a : ℝ) : 
  (4^2 - 3*4 = a^2) → (a = 2 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l2108_210851


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l2108_210823

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l2108_210823


namespace NUMINAMATH_CALUDE_triangle_side_length_l2108_210835

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  b = 3 → c = 3 → B = π / 6 → a = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2108_210835


namespace NUMINAMATH_CALUDE_locus_of_P_l2108_210845

noncomputable def ellipse (x y : ℝ) : Prop := x^2/20 + y^2/16 = 1

def on_ellipse (M : ℝ × ℝ) : Prop := ellipse M.1 M.2

def intersect_x_axis (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 0 ∧ ellipse B.1 0 ∧ A.2 = 0 ∧ B.2 = 0

def tangent_line (l : ℝ → ℝ) (M : ℝ × ℝ) : Prop :=
  on_ellipse M ∧ ∀ x, l x = (M.2 / M.1) * (x - M.1) + M.2

def perpendicular_intersect (A B C D : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  C.1 = A.1 ∧ D.1 = B.1 ∧ l C.1 = C.2 ∧ l D.1 = D.2

def line_intersection (C B A D : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ, Q = (t * C.1 + (1 - t) * B.1, t * C.2 + (1 - t) * B.2) ∧
             Q = (s * A.1 + (1 - s) * D.1, s * A.2 + (1 - s) * D.2)

def symmetric_point (P Q M : ℝ × ℝ) : Prop :=
  P.1 + Q.1 = 2 * M.1 ∧ P.2 + Q.2 = 2 * M.2

theorem locus_of_P (A B M C D Q P : ℝ × ℝ) (l : ℝ → ℝ) :
  intersect_x_axis A B →
  on_ellipse M →
  M ≠ A →
  M ≠ B →
  tangent_line l M →
  perpendicular_intersect A B C D l →
  line_intersection C B A D Q →
  symmetric_point P Q M →
  (P.1^2 / 20 + P.2^2 / 36 = 1 ∧ P.2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_P_l2108_210845


namespace NUMINAMATH_CALUDE_max_gemstone_value_is_72_l2108_210807

/-- Represents a type of gemstone with its weight and value --/
structure Gemstone where
  weight : ℕ
  value : ℕ

/-- The problem setup --/
def treasureHuntProblem :=
  let sapphire : Gemstone := ⟨6, 15⟩
  let ruby : Gemstone := ⟨3, 9⟩
  let diamond : Gemstone := ⟨2, 5⟩
  let maxWeight : ℕ := 24
  let minEachType : ℕ := 10
  (sapphire, ruby, diamond, maxWeight, minEachType)

/-- The maximum value of gemstones that can be carried --/
def maxGemstoneValue (problem : Gemstone × Gemstone × Gemstone × ℕ × ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum value is 72 --/
theorem max_gemstone_value_is_72 :
  maxGemstoneValue treasureHuntProblem = 72 := by
  sorry

end NUMINAMATH_CALUDE_max_gemstone_value_is_72_l2108_210807


namespace NUMINAMATH_CALUDE_wood_carving_shelves_l2108_210880

theorem wood_carving_shelves (total_carvings : ℕ) (carvings_per_shelf : ℕ) (shelves_filled : ℕ) : 
  total_carvings = 56 → 
  carvings_per_shelf = 8 → 
  shelves_filled = total_carvings / carvings_per_shelf → 
  shelves_filled = 7 := by
sorry

end NUMINAMATH_CALUDE_wood_carving_shelves_l2108_210880


namespace NUMINAMATH_CALUDE_scientific_notation_of_600_billion_l2108_210840

theorem scientific_notation_of_600_billion :
  let billion : ℕ := 10^9
  600 * billion = 6 * 10^11 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_600_billion_l2108_210840


namespace NUMINAMATH_CALUDE_max_q_minus_r_for_1027_l2108_210869

theorem max_q_minus_r_for_1027 :
  ∀ q r : ℕ+, 
  1027 = 23 * q + r → 
  ∀ q' r' : ℕ+, 
  1027 = 23 * q' + r' → 
  q - r ≤ 29 ∧ ∃ q r : ℕ+, 1027 = 23 * q + r ∧ q - r = 29 :=
by sorry

end NUMINAMATH_CALUDE_max_q_minus_r_for_1027_l2108_210869


namespace NUMINAMATH_CALUDE_power_sum_difference_equals_ten_l2108_210867

theorem power_sum_difference_equals_ten : 2^5 + 5^2 / 5^1 - 3^3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_equals_ten_l2108_210867


namespace NUMINAMATH_CALUDE_gcd_242_154_l2108_210872

theorem gcd_242_154 : Nat.gcd 242 154 = 22 := by
  sorry

end NUMINAMATH_CALUDE_gcd_242_154_l2108_210872


namespace NUMINAMATH_CALUDE_initial_machines_count_l2108_210895

/-- The number of machines in the initial group -/
def initial_machines : ℕ := 15

/-- The number of bags produced per minute by the initial group -/
def initial_production_rate : ℕ := 45

/-- The number of machines in the larger group -/
def larger_group_machines : ℕ := 150

/-- The number of bags produced by the larger group -/
def larger_group_production : ℕ := 3600

/-- The time taken by the larger group to produce the bags (in minutes) -/
def production_time : ℕ := 8

theorem initial_machines_count :
  initial_machines = 15 ∧
  initial_production_rate = 45 ∧
  larger_group_machines = 150 ∧
  larger_group_production = 3600 ∧
  production_time = 8 →
  initial_machines * larger_group_production = initial_production_rate * larger_group_machines * production_time :=
by sorry

end NUMINAMATH_CALUDE_initial_machines_count_l2108_210895


namespace NUMINAMATH_CALUDE_pool_water_after_20_days_l2108_210873

/-- Calculates the remaining water in a swimming pool after a given number of days -/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (leak_rate : ℝ) (days : ℝ) : ℝ :=
  initial_amount - (evaporation_rate + leak_rate) * days

/-- Theorem stating the remaining water in the pool after 20 days -/
theorem pool_water_after_20_days :
  remaining_water 500 1.5 0.8 20 = 454 := by
  sorry

#eval remaining_water 500 1.5 0.8 20

end NUMINAMATH_CALUDE_pool_water_after_20_days_l2108_210873


namespace NUMINAMATH_CALUDE_sock_selection_l2108_210830

theorem sock_selection (n k : ℕ) (h1 : n = 7) (h2 : k = 4) : 
  Nat.choose n k = 35 := by
sorry

end NUMINAMATH_CALUDE_sock_selection_l2108_210830


namespace NUMINAMATH_CALUDE_extended_triangle_similarity_l2108_210893

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the similarity of triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the extension of a line segment
def extend (p1 p2 : ℝ × ℝ) (length : ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem extended_triangle_similarity (ABC : Triangle) (P : ℝ × ℝ) :
  distance ABC.A ABC.B = 8 →
  distance ABC.B ABC.C = 7 →
  distance ABC.C ABC.A = 6 →
  P = extend ABC.B ABC.C (distance ABC.B P - 7) →
  similar (Triangle.mk P ABC.A ABC.B) (Triangle.mk P ABC.C ABC.A) →
  distance P ABC.C = 9 := by
  sorry

end NUMINAMATH_CALUDE_extended_triangle_similarity_l2108_210893


namespace NUMINAMATH_CALUDE_initial_value_problem_l2108_210818

theorem initial_value_problem : ∃! x : ℤ, (x + 82) % 456 = 0 ∧ x = 374 := by sorry

end NUMINAMATH_CALUDE_initial_value_problem_l2108_210818


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l2108_210836

theorem isosceles_right_triangle_ratio (a c : ℝ) : 
  a > 0 → -- Ensure a is positive
  c^2 = 2 * a^2 → -- Pythagorean theorem for isosceles right triangle
  (2 * a) / c = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l2108_210836


namespace NUMINAMATH_CALUDE_cubic_three_distinct_roots_l2108_210877

/-- The cubic equation x^3 - 3x^2 - a = 0 has three distinct real roots if and only if a is in the open interval (-4, 0) -/
theorem cubic_three_distinct_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 3*x^2 - a = 0 ∧
    y^3 - 3*y^2 - a = 0 ∧
    z^3 - 3*z^2 - a = 0) ↔
  -4 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_distinct_roots_l2108_210877


namespace NUMINAMATH_CALUDE_f_solutions_when_a_neg_one_f_monotonic_increasing_iff_f_max_min_when_a_one_l2108_210886

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (x - 1) * |x - a|

-- Part 1
theorem f_solutions_when_a_neg_one :
  ∀ x : ℝ, f (-1) x = 1 ↔ (x ≤ -1 ∨ x = 1) :=
sorry

-- Part 2
theorem f_monotonic_increasing_iff :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≥ 1/3 :=
sorry

-- Part 3
theorem f_max_min_when_a_one :
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f 1 x ≤ 1) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f 1 x = 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f 1 x ≥ -1) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f 1 x = -1) :=
sorry

end NUMINAMATH_CALUDE_f_solutions_when_a_neg_one_f_monotonic_increasing_iff_f_max_min_when_a_one_l2108_210886


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2108_210846

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*x₁ - 15 = 0 ∧ x₂^2 - 2*x₂ - 15 = 0 ∧ x₁ = 5 ∧ x₂ = -3) ∧
  (∃ y₁ y₂ : ℝ, 2*y₁^2 + 3*y₁ = 1 ∧ 2*y₂^2 + 3*y₂ = 1 ∧ 
   y₁ = (-3 + Real.sqrt 17) / 4 ∧ y₂ = (-3 - Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2108_210846


namespace NUMINAMATH_CALUDE_f_is_odd_and_increasing_l2108_210850

def f (x : ℝ) : ℝ := x * |x|

theorem f_is_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_odd_and_increasing_l2108_210850


namespace NUMINAMATH_CALUDE_true_false_questions_count_l2108_210843

/-- Proves that the number of true/false questions is 6 given the conditions of the problem -/
theorem true_false_questions_count :
  ∀ (T F M : ℕ),
  T + F + M = 45 →
  M = 2 * F →
  F = T + 7 →
  T = 6 := by
sorry

end NUMINAMATH_CALUDE_true_false_questions_count_l2108_210843


namespace NUMINAMATH_CALUDE_pencil_division_l2108_210819

theorem pencil_division (num_students num_pencils : ℕ) 
  (h1 : num_students = 2) 
  (h2 : num_pencils = 18) : 
  num_pencils / num_students = 9 := by
sorry

end NUMINAMATH_CALUDE_pencil_division_l2108_210819


namespace NUMINAMATH_CALUDE_a_21_value_l2108_210890

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property of being a geometric sequence -/
def IsGeometric (b : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem a_21_value
  (a b : Sequence)
  (h1 : a 1 = 1)
  (h2 : IsGeometric b)
  (h3 : ∀ n : ℕ, b n = a (n + 1) / a n)
  (h4 : b 10 * b 11 = 52) :
  a 21 = 4 := by
sorry

end NUMINAMATH_CALUDE_a_21_value_l2108_210890


namespace NUMINAMATH_CALUDE_fraction_value_at_two_l2108_210815

theorem fraction_value_at_two :
  let f (x : ℝ) := (x^10 + 20*x^5 + 100) / (x^5 + 10)
  f 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_two_l2108_210815


namespace NUMINAMATH_CALUDE_not_54_after_60_ops_l2108_210878

/-- Represents the possible operations on the board number -/
inductive Operation
  | MultiplyBy2
  | DivideBy2
  | MultiplyBy3
  | DivideBy3

/-- Applies an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.MultiplyBy2 => n * 2
  | Operation.DivideBy2 => n / 2
  | Operation.MultiplyBy3 => n * 3
  | Operation.DivideBy3 => n / 3

/-- Applies a sequence of operations to a number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- Theorem: After 60 operations on 12, the result cannot be 54 -/
theorem not_54_after_60_ops :
  ∀ (ops : List Operation), ops.length = 60 → applyOperations 12 ops ≠ 54 := by
  sorry


end NUMINAMATH_CALUDE_not_54_after_60_ops_l2108_210878


namespace NUMINAMATH_CALUDE_average_age_combined_l2108_210837

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℝ) (avg_age_parents : ℝ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 10 →
  avg_age_parents = 40 →
  let total_individuals := num_students + num_parents
  let total_age := num_students * avg_age_students + num_parents * avg_age_parents
  (total_age / total_individuals : ℝ) = 28 := by
sorry

end NUMINAMATH_CALUDE_average_age_combined_l2108_210837


namespace NUMINAMATH_CALUDE_function_difference_inequality_l2108_210811

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivative condition
variable (h : ∀ x, HasDerivAt f (f' x) x ∧ HasDerivAt g (g' x) x ∧ f' x > g' x)

-- State the theorem
theorem function_difference_inequality (x₁ x₂ : ℝ) (h_lt : x₁ < x₂) :
  f x₁ - f x₂ < g x₁ - g x₂ := by
  sorry

end NUMINAMATH_CALUDE_function_difference_inequality_l2108_210811


namespace NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l2108_210806

theorem log_sqrt10_1000sqrt10 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l2108_210806


namespace NUMINAMATH_CALUDE_vector_c_value_l2108_210810

def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (-2, 4)

theorem vector_c_value :
  ∀ c : ℝ × ℝ, (4 • a) + (3 • b - 2 • a) + c = (0, 0) → c = (4, -6) := by
  sorry

end NUMINAMATH_CALUDE_vector_c_value_l2108_210810


namespace NUMINAMATH_CALUDE_canned_food_bins_l2108_210897

theorem canned_food_bins (soup vegetables pasta : Real) 
  (h1 : soup = 0.125)
  (h2 : vegetables = 0.125)
  (h3 : pasta = 0.5) :
  soup + vegetables + pasta = 0.75 := by
sorry

end NUMINAMATH_CALUDE_canned_food_bins_l2108_210897


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l2108_210866

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 27 → books_sold = 6 → books_per_shelf = 7 → 
  (initial_stock - books_sold) / books_per_shelf = 3 := by
sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l2108_210866


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2108_210826

theorem fraction_equation_solution : 
  {x : ℝ | (1 / (x^2 + 17*x + 20) + 1 / (x^2 + 12*x + 20) + 1 / (x^2 - 15*x + 20) = 0) ∧ 
           x ≠ -20 ∧ x ≠ -5 ∧ x ≠ -4 ∧ x ≠ -1} = 
  {x : ℝ | x = -20 ∨ x = -5 ∨ x = -4 ∨ x = -1} := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2108_210826


namespace NUMINAMATH_CALUDE_number_puzzle_l2108_210882

theorem number_puzzle : ∃ x : ℤ, x + 3*12 + 3*13 + 3*16 = 134 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2108_210882


namespace NUMINAMATH_CALUDE_count_arith_seq_39_eq_12_l2108_210859

/-- An arithmetic sequence of positive integers containing 3 and 39 -/
structure ArithSeq39 where
  d : ℕ+  -- Common difference
  a : ℕ+  -- First term
  h1 : ∃ k : ℕ, a + k * d = 3
  h2 : ∃ m : ℕ, a + m * d = 39

/-- The count of arithmetic sequences containing 3 and 39 -/
def count_arith_seq_39 : ℕ := sorry

/-- Theorem: There are exactly 12 infinite arithmetic sequences of positive integers
    that contain both 3 and 39 -/
theorem count_arith_seq_39_eq_12 : count_arith_seq_39 = 12 := by sorry

end NUMINAMATH_CALUDE_count_arith_seq_39_eq_12_l2108_210859


namespace NUMINAMATH_CALUDE_fraction_equality_l2108_210847

theorem fraction_equality (a b : ℝ) (h : a ≠ b) (h1 : a / 4 = b / 3) : b / (a - b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2108_210847


namespace NUMINAMATH_CALUDE_teachers_survey_l2108_210857

theorem teachers_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ) :
  total = 150 →
  high_bp = 90 →
  heart_trouble = 50 →
  both = 30 →
  (((total - (high_bp + heart_trouble - both)) : ℚ) / total) * 100 = 80 / 3 :=
by sorry

end NUMINAMATH_CALUDE_teachers_survey_l2108_210857


namespace NUMINAMATH_CALUDE_sum_greater_than_four_necessary_not_sufficient_l2108_210883

theorem sum_greater_than_four_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, (a > 1 ∧ b > 3) → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 1 ∧ b > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_necessary_not_sufficient_l2108_210883


namespace NUMINAMATH_CALUDE_probability_first_two_trials_l2108_210899

-- Define the probability of event A
def P_A : ℝ := 0.7

-- Define the number of trials
def num_trials : ℕ := 4

-- Define the probability of event A occurring exactly in the first two trials
def P_first_two : ℝ := P_A * P_A * (1 - P_A) * (1 - P_A)

-- Theorem statement
theorem probability_first_two_trials : P_first_two = 0.0441 := by
  sorry

end NUMINAMATH_CALUDE_probability_first_two_trials_l2108_210899


namespace NUMINAMATH_CALUDE_average_female_height_l2108_210808

/-- Given the overall average height of students is 180 cm, the average height of male students
    is 182 cm, and the ratio of men to women is 5:1, prove that the average female height is 170 cm. -/
theorem average_female_height
  (overall_avg : ℝ)
  (male_avg : ℝ)
  (ratio : ℕ)
  (h1 : overall_avg = 180)
  (h2 : male_avg = 182)
  (h3 : ratio = 5)
  : ∃ (female_avg : ℝ), female_avg = 170 ∧
    (ratio * male_avg + female_avg) / (ratio + 1) = overall_avg :=
by
  sorry

end NUMINAMATH_CALUDE_average_female_height_l2108_210808


namespace NUMINAMATH_CALUDE_triangle_area_13_14_15_l2108_210813

/-- The area of a triangle with sides 13, 14, and 15 is 84 -/
theorem triangle_area_13_14_15 : ∃ (area : ℝ), area = 84 ∧ 
  (∀ (s : ℝ), s = (13 + 14 + 15) / 2 → 
    area = Real.sqrt (s * (s - 13) * (s - 14) * (s - 15))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_13_14_15_l2108_210813


namespace NUMINAMATH_CALUDE_probability_factor_less_than_10_of_90_l2108_210820

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem probability_factor_less_than_10_of_90 :
  let all_factors := factors 90
  let factors_less_than_10 := all_factors.filter (λ x => x < 10)
  (factors_less_than_10.card : ℚ) / all_factors.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_10_of_90_l2108_210820


namespace NUMINAMATH_CALUDE_certain_event_occurs_always_rolling_greater_than_six_impossible_l2108_210879

/-- A type representing the outcome of an event -/
inductive EventOutcome
  | Occurred
  | NotOccurred

/-- A function representing a trial of an event -/
def Trial := Unit → EventOutcome

/-- Definition of a certain event -/
def CertainEvent (e : Trial) : Prop :=
  ∀ _ : Unit, e () = EventOutcome.Occurred

/-- Theorem stating that a certain event occurs in every trial -/
theorem certain_event_occurs_always (e : Trial) (h : CertainEvent e) :
  ∀ _ : Unit, e () = EventOutcome.Occurred :=
by
  sorry

/-- Definition of an impossible event -/
def ImpossibleEvent (e : Trial) : Prop :=
  ∀ _ : Unit, e () = EventOutcome.NotOccurred

/-- Definition of a standard die -/
def StandardDie := Fin 6

/-- Theorem stating that rolling a number greater than 6 on a standard die is impossible -/
theorem rolling_greater_than_six_impossible :
  ¬ ∃ (x : StandardDie), x.val > 6 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_event_occurs_always_rolling_greater_than_six_impossible_l2108_210879


namespace NUMINAMATH_CALUDE_octagon_intersection_only_hexagonal_prism_l2108_210809

/-- Represents the possible geometric solids --/
inductive GeometricSolid
  | TriangularPrism
  | RectangularPrism
  | PentagonalPrism
  | HexagonalPrism

/-- Represents the possible shapes resulting from a plane intersection --/
inductive IntersectionShape
  | Triangle
  | Quadrilateral
  | Pentagon
  | Hexagon
  | Heptagon
  | Octagon
  | Rectangle

/-- Returns the possible intersection shapes for a given geometric solid --/
def possibleIntersections (solid : GeometricSolid) : List IntersectionShape :=
  match solid with
  | GeometricSolid.TriangularPrism => [IntersectionShape.Quadrilateral, IntersectionShape.Triangle]
  | GeometricSolid.RectangularPrism => [IntersectionShape.Pentagon, IntersectionShape.Quadrilateral, IntersectionShape.Triangle, IntersectionShape.Rectangle]
  | GeometricSolid.PentagonalPrism => [IntersectionShape.Hexagon, IntersectionShape.Pentagon, IntersectionShape.Rectangle, IntersectionShape.Triangle]
  | GeometricSolid.HexagonalPrism => [IntersectionShape.Octagon, IntersectionShape.Heptagon, IntersectionShape.Rectangle]

/-- Theorem: Only the hexagonal prism can produce an octagonal intersection --/
theorem octagon_intersection_only_hexagonal_prism :
  ∀ (solid : GeometricSolid),
    (IntersectionShape.Octagon ∈ possibleIntersections solid) ↔ (solid = GeometricSolid.HexagonalPrism) :=
by sorry


end NUMINAMATH_CALUDE_octagon_intersection_only_hexagonal_prism_l2108_210809


namespace NUMINAMATH_CALUDE_sum_equals_power_of_two_l2108_210858

theorem sum_equals_power_of_two : 29 + 12 + 23 = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_power_of_two_l2108_210858


namespace NUMINAMATH_CALUDE_prime_before_non_prime_probability_l2108_210874

def prime_numbers : List ℕ := [2, 3, 5, 7, 11]
def non_prime_numbers : List ℕ := [1, 4, 6, 8, 9, 10, 12]

def total_numbers : ℕ := prime_numbers.length + non_prime_numbers.length

theorem prime_before_non_prime_probability :
  let favorable_permutations := (prime_numbers.length.factorial * non_prime_numbers.length.factorial : ℚ)
  let total_permutations := total_numbers.factorial
  (favorable_permutations / total_permutations : ℚ) = 1 / 792 := by
  sorry

end NUMINAMATH_CALUDE_prime_before_non_prime_probability_l2108_210874


namespace NUMINAMATH_CALUDE_basketball_games_l2108_210804

theorem basketball_games (x : ℕ) : 
  (3 * x / 4 : ℚ) = x * 3 / 4 ∧ 
  (2 * (x + 10) / 3 : ℚ) = x * 3 / 4 + 5 → 
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_games_l2108_210804


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_l2108_210803

theorem binomial_coefficient_identity (n k : ℕ+) (h : k ≤ n) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) ∧
  k * Nat.choose n k = (n - k + 1) * Nat.choose n (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_l2108_210803


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l2108_210896

/-- An isosceles triangle with given properties has an area of 54 square centimeters -/
theorem isosceles_triangle_area (a b : ℝ) (h_isosceles : a = b) (h_perimeter : 2 * a + b = 36)
  (h_base_angles : 2 * Real.arccos ((a^2 - b^2/4) / a^2) = 130 * π / 180)
  (h_inradius : (a * b) / (a + b + (a^2 - b^2/4).sqrt) = 3) : 
  a * b * Real.sin (Real.arccos ((a^2 - b^2/4) / a^2)) / 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l2108_210896


namespace NUMINAMATH_CALUDE_tv_price_increase_l2108_210863

theorem tv_price_increase (P : ℝ) (x : ℝ) (h1 : P > 0) :
  (0.80 * P + x / 100 * (0.80 * P) = 1.20 * P) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_increase_l2108_210863


namespace NUMINAMATH_CALUDE_bottle_eq_five_cups_l2108_210871

/-- Represents the volume of a container -/
structure Volume : Type :=
  (amount : ℕ)

/-- Define JUG in terms of BOTTLE and GLASS -/
axiom jug_def : Volume → Volume → Volume
axiom jug_eq_bottle_plus_glass : ∀ (b g : Volume), jug_def b g = Volume.mk (b.amount + g.amount)

/-- Define the relationship between JUGs and GLASSes -/
axiom two_jugs_eq_seven_glasses : ∀ (j g : Volume), 2 * (jug_def j g).amount = 7 * g.amount

/-- Define BOTTLE in terms of CUP and GLASS -/
axiom bottle_def : Volume → Volume → Volume
axiom bottle_eq_cup_plus_two_glasses : ∀ (c g : Volume), bottle_def c g = Volume.mk (c.amount + 2 * g.amount)

/-- The main theorem to prove -/
theorem bottle_eq_five_cups : 
  ∀ (b c g : Volume), 
  jug_def b g = Volume.mk (b.amount + g.amount) →
  2 * (jug_def b g).amount = 7 * g.amount →
  bottle_def c g = Volume.mk (c.amount + 2 * g.amount) →
  b = Volume.mk (5 * c.amount) :=
sorry

end NUMINAMATH_CALUDE_bottle_eq_five_cups_l2108_210871


namespace NUMINAMATH_CALUDE_gcd_228_1995_base3_11102_to_decimal_l2108_210855

-- Problem 1: GCD of 228 and 1995
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by sorry

-- Problem 2: Base 3 to decimal conversion
def base3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_11102_to_decimal :
  base3_to_decimal [1, 1, 1, 0, 2] = 119 := by sorry

end NUMINAMATH_CALUDE_gcd_228_1995_base3_11102_to_decimal_l2108_210855


namespace NUMINAMATH_CALUDE_benny_state_tax_l2108_210876

/-- Calculates the total state tax in cents per hour given an hourly wage in dollars, a tax rate percentage, and a fixed tax in cents. -/
def total_state_tax (hourly_wage : ℚ) (tax_rate_percent : ℚ) (fixed_tax_cents : ℕ) : ℕ :=
  sorry

/-- Proves that given Benny's hourly wage of $25, a 2% state tax rate, and a fixed tax of 50 cents per hour, the total amount of state taxes paid per hour is 100 cents. -/
theorem benny_state_tax :
  total_state_tax 25 2 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_benny_state_tax_l2108_210876


namespace NUMINAMATH_CALUDE_curve_c_properties_l2108_210849

/-- The curve C in a rectangular coordinate system -/
structure CurveC where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Point on the curve C -/
structure PointOnC (c : CurveC) where
  φ : ℝ
  x : ℝ
  y : ℝ
  h_x : x = c.a * Real.cos φ
  h_y : y = c.b * Real.sin φ

/-- Theorem about the curve C -/
theorem curve_c_properties (c : CurveC) 
  (m : PointOnC c) 
  (h_m_x : m.x = 2) 
  (h_m_y : m.y = Real.sqrt 3) 
  (h_m_φ : m.φ = π / 3) :
  (∀ x y, x^2 / 16 + y^2 / 4 = 1 ↔ ∃ φ, x = c.a * Real.cos φ ∧ y = c.b * Real.sin φ) ∧
  (∀ ρ₁ ρ₂ θ, 
    (∃ φ₁, ρ₁ * Real.cos θ = c.a * Real.cos φ₁ ∧ ρ₁ * Real.sin θ = c.b * Real.sin φ₁) →
    (∃ φ₂, ρ₂ * Real.cos (θ + π/2) = c.a * Real.cos φ₂ ∧ ρ₂ * Real.sin (θ + π/2) = c.b * Real.sin φ₂) →
    1 / ρ₁^2 + 1 / ρ₂^2 = 5 / 16) :=
by sorry

end NUMINAMATH_CALUDE_curve_c_properties_l2108_210849


namespace NUMINAMATH_CALUDE_prob_ratio_l2108_210842

/-- Represents the total number of cards in the box -/
def total_cards : ℕ := 50

/-- Represents the number of distinct card numbers -/
def distinct_numbers : ℕ := 10

/-- Represents the number of cards for each number -/
def cards_per_number : ℕ := 5

/-- Represents the number of cards drawn -/
def cards_drawn : ℕ := 5

/-- Calculates the probability of drawing five cards with the same number -/
def prob_same_number : ℚ :=
  (distinct_numbers : ℚ) / (total_cards.choose cards_drawn : ℚ)

/-- Calculates the probability of drawing four cards of one number and one card of a different number -/
def prob_four_and_one : ℚ :=
  ((distinct_numbers : ℚ) * (distinct_numbers - 1 : ℚ) * (cards_per_number : ℚ) * (cards_per_number : ℚ)) /
  (total_cards.choose cards_drawn : ℚ)

/-- Theorem stating the ratio of probabilities -/
theorem prob_ratio :
  prob_four_and_one / prob_same_number = 225 := by sorry

end NUMINAMATH_CALUDE_prob_ratio_l2108_210842


namespace NUMINAMATH_CALUDE_small_slice_price_l2108_210832

/-- The price of a small slice of pizza given the following conditions:
  1. Large slices are sold for Rs. 250 each
  2. 5000 slices were sold in total
  3. Total revenue was Rs. 1,050,000
  4. 2000 small slices were sold
-/
theorem small_slice_price (large_slice_price : ℕ) (total_slices : ℕ) (total_revenue : ℕ) (small_slices : ℕ) :
  large_slice_price = 250 →
  total_slices = 5000 →
  total_revenue = 1050000 →
  small_slices = 2000 →
  ∃ (small_slice_price : ℕ),
    small_slice_price * small_slices + large_slice_price * (total_slices - small_slices) = total_revenue ∧
    small_slice_price = 150 :=
by sorry

end NUMINAMATH_CALUDE_small_slice_price_l2108_210832


namespace NUMINAMATH_CALUDE_line_intersects_circle_midpoint_trajectory_line_equations_with_ratio_l2108_210814

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line L
def line_L (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the fixed point P
def point_P : ℝ × ℝ := (1, 1)

-- Theorem 1: Line L always intersects circle C at two distinct points
theorem line_intersects_circle (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ :=
sorry

-- Theorem 2: Trajectory of midpoint M
theorem midpoint_trajectory (x y : ℝ) :
  (∃ (m : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ ∧
    x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2) ↔
  x^2 + y^2 - x - 2*y + 1 = 0 :=
sorry

-- Theorem 3: Equations of line L when P divides AB in 1:2 ratio
theorem line_equations_with_ratio :
  ∃ (m : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ ∧
    2 * (point_P.1 - x₁) = x₂ - point_P.1 ∧
    2 * (point_P.2 - y₁) = y₂ - point_P.2 ↔
  (∀ x y, x - y = 0 ∨ x + y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_midpoint_trajectory_line_equations_with_ratio_l2108_210814


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_powers_l2108_210894

theorem quadratic_roots_sum_powers (t q : ℝ) (a₁ a₂ : ℝ) : 
  (∀ x : ℝ, x^2 - t*x + q = 0 ↔ x = a₁ ∨ x = a₂) →
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1003 → a₁^n + a₂^n = a₁ + a₂) →
  a₁^1004 + a₂^1004 = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_powers_l2108_210894


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2108_210860

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2108_210860


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_min_value_achievable_l2108_210892

theorem min_value_quadratic_form (x y : ℝ) : 
  3 * x^2 + 2 * x * y + 3 * y^2 + 5 ≥ 5 :=
by sorry

theorem min_value_achievable : 
  ∃ (x y : ℝ), 3 * x^2 + 2 * x * y + 3 * y^2 + 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_min_value_achievable_l2108_210892


namespace NUMINAMATH_CALUDE_same_monotonicity_implies_phi_value_l2108_210861

open Real

theorem same_monotonicity_implies_phi_value (φ : Real) :
  (∀ x ∈ Set.Icc 0 (π / 2), 
    (∀ y ∈ Set.Icc 0 (π / 2), x < y → cos (2 * x) > cos (2 * y)) ↔ 
    (∀ y ∈ Set.Icc 0 (π / 2), x < y → sin (x + φ) > sin (y + φ))) →
  φ = π / 2 := by
sorry

end NUMINAMATH_CALUDE_same_monotonicity_implies_phi_value_l2108_210861


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l2108_210839

theorem farmer_land_ownership (total_land : ℝ) : 
  (0.9 * total_land * 0.1 = 360) → total_land = 4000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l2108_210839


namespace NUMINAMATH_CALUDE_outfit_count_l2108_210888

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of pairs of pants available. -/
def num_pants : ℕ := 5

/-- The number of ties available. -/
def num_ties : ℕ := 4

/-- The number of belts available. -/
def num_belts : ℕ := 2

/-- An outfit consists of one shirt, one pair of pants, and optionally one tie and/or one belt. -/
def outfit := ℕ × ℕ × Option ℕ × Option ℕ

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_belts + 1)

/-- Theorem stating that the total number of possible outfits is 600. -/
theorem outfit_count : total_outfits = 600 := by sorry

end NUMINAMATH_CALUDE_outfit_count_l2108_210888


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l2108_210833

/-- The general term of the sequence -/
def sequenceTerm (n : ℕ) : ℚ := (2 * n) / (2 * n + 1)

/-- The 10th term of the sequence is 20/21 -/
theorem tenth_term_of_sequence : sequenceTerm 10 = 20 / 21 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l2108_210833


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2108_210812

theorem negation_of_proposition :
  (¬(∀ x y : ℝ, x > 0 ∧ y > 0 → x + y > 0)) ↔
  (∀ x y : ℝ, ¬(x > 0 ∧ y > 0) → x + y ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2108_210812


namespace NUMINAMATH_CALUDE_complex_magnitude_l2108_210854

theorem complex_magnitude (z : ℂ) (h : (2 - I) * z = 6 + 2 * I) : Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2108_210854


namespace NUMINAMATH_CALUDE_addition_subtraction_problem_l2108_210817

theorem addition_subtraction_problem : (5.75 + 3.09) - 1.86 = 6.98 := by
  sorry

end NUMINAMATH_CALUDE_addition_subtraction_problem_l2108_210817


namespace NUMINAMATH_CALUDE_rented_movie_cost_l2108_210852

theorem rented_movie_cost (ticket_price : ℚ) (num_tickets : ℕ) (bought_movie_price : ℚ) (total_spent : ℚ) :
  ticket_price = 10.62 →
  num_tickets = 2 →
  bought_movie_price = 13.95 →
  total_spent = 36.78 →
  total_spent - (ticket_price * num_tickets + bought_movie_price) = 1.59 :=
by sorry

end NUMINAMATH_CALUDE_rented_movie_cost_l2108_210852


namespace NUMINAMATH_CALUDE_max_value_of_f_l2108_210884

open Real

noncomputable def f (x : ℝ) := Real.log (3 * x) - 3 * x

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Ioo 0 (Real.exp 1) ∧
  (∀ x, x ∈ Set.Ioo 0 (Real.exp 1) → f x ≤ f c) ∧
  f c = -Real.log 3 - 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2108_210884


namespace NUMINAMATH_CALUDE_journey_time_proof_l2108_210841

/-- The total distance of the journey in miles -/
def total_distance : ℝ := 120

/-- The speed of the car in miles per hour -/
def car_speed : ℝ := 30

/-- The walking speed in miles per hour -/
def walking_speed : ℝ := 5

/-- The distance Tom and Harry initially travel by car -/
def initial_car_distance : ℝ := 40

/-- Theorem stating that under the given conditions, the total journey time is 52/3 hours -/
theorem journey_time_proof :
  ∃ (T d : ℝ),
    -- Tom and Harry's initial car journey
    car_speed * (4/3) = initial_car_distance ∧
    -- Harry's walk back
    walking_speed * (T - 4/3) = d ∧
    -- Dick's walk
    walking_speed * (T - 4/3) = total_distance - d ∧
    -- Tom's return journey
    car_speed * T = 2 * initial_car_distance + d ∧
    -- Total journey time
    T = 52/3 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_proof_l2108_210841


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l2108_210870

theorem quadratic_two_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - x + k + 1 = 0 ∧ y^2 - y + k + 1 = 0) → k ≤ -3/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l2108_210870


namespace NUMINAMATH_CALUDE_uphill_speed_calculation_l2108_210875

-- Define the problem parameters
def uphill_distance : ℝ := 100
def downhill_distance : ℝ := 50
def downhill_speed : ℝ := 60
def average_speed : ℝ := 36

-- Define the theorem
theorem uphill_speed_calculation (V_up : ℝ) :
  V_up > 0 →
  average_speed = (uphill_distance + downhill_distance) / 
    (uphill_distance / V_up + downhill_distance / downhill_speed) →
  V_up = 30 := by
sorry

end NUMINAMATH_CALUDE_uphill_speed_calculation_l2108_210875


namespace NUMINAMATH_CALUDE_tims_income_percentage_l2108_210844

theorem tims_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = 1.6 * tim) 
  (h2 : mary = 1.12 * juan) : 
  tim = 0.7 * juan := by
sorry

end NUMINAMATH_CALUDE_tims_income_percentage_l2108_210844


namespace NUMINAMATH_CALUDE_exists_sum_digits_div_11_l2108_210864

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among any 39 consecutive natural numbers, there is always one whose sum of digits is divisible by 11 -/
theorem exists_sum_digits_div_11 (start : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (sum_of_digits (start + k)) % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_exists_sum_digits_div_11_l2108_210864


namespace NUMINAMATH_CALUDE_container_initial_percentage_l2108_210848

theorem container_initial_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) :
  capacity = 80 →
  added_water = 20 →
  final_fraction = 3/4 →
  (capacity * final_fraction - added_water) / capacity = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_container_initial_percentage_l2108_210848


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2108_210800

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_edge := sphere_diameter / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2108_210800


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_25_l2108_210862

theorem no_primes_divisible_by_25 : ∀ p : ℕ, Nat.Prime p → ¬(25 ∣ p) := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_25_l2108_210862


namespace NUMINAMATH_CALUDE_consecutive_color_draw_probability_l2108_210891

def numTan : ℕ := 4
def numPink : ℕ := 3
def numViolet : ℕ := 5
def totalChips : ℕ := numTan + numPink + numViolet

theorem consecutive_color_draw_probability :
  (numTan.factorial * numPink.factorial * numViolet.factorial) / totalChips.factorial = 1 / 27720 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_color_draw_probability_l2108_210891


namespace NUMINAMATH_CALUDE_condition_relationship_l2108_210865

theorem condition_relationship :
  (∀ x : ℝ, (2 ≤ x ∧ x ≤ 3) → (x < -3 ∨ x ≥ 1)) ∧
  (∃ x : ℝ, (x < -3 ∨ x ≥ 1) ∧ ¬(2 ≤ x ∧ x ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2108_210865


namespace NUMINAMATH_CALUDE_triangle_similarity_l2108_210898

theorem triangle_similarity (DC CB AD : ℝ) (h1 : DC = 9) (h2 : CB = 6) 
  (h3 : AD > 0) (h4 : ∃ (AB : ℝ), AB = (1/3) * AD) (h5 : ∃ (ED : ℝ), ED = (2/3) * AD) : 
  ∃ (FC : ℝ), FC = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_l2108_210898


namespace NUMINAMATH_CALUDE_max_guaranteed_points_is_34_l2108_210834

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  num_teams : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- The specific tournament described in the problem -/
def tournament : FootballTournament :=
  { num_teams := 15
  , points_for_win := 3
  , points_for_draw := 1
  , points_for_loss := 0 }

/-- The maximum number of points that can be guaranteed for each of 6 teams -/
def max_guaranteed_points (t : FootballTournament) : Nat :=
  34

/-- Theorem stating that 34 is the maximum number of points that can be guaranteed for each of 6 teams -/
theorem max_guaranteed_points_is_34 :
  ∀ n : Nat, n > max_guaranteed_points tournament →
  ¬(∃ points : Fin tournament.num_teams → Nat,
    (∀ i j : Fin tournament.num_teams, i ≠ j →
      points i + points j ≤ tournament.points_for_win) ∧
    (∃ top_6 : Finset (Fin tournament.num_teams),
      top_6.card = 6 ∧ ∀ i ∈ top_6, points i ≥ n)) :=
by sorry

end NUMINAMATH_CALUDE_max_guaranteed_points_is_34_l2108_210834


namespace NUMINAMATH_CALUDE_probability_sum_less_2_or_greater_3_l2108_210838

/-- Represents a bag of balls with marks -/
structure Bag :=
  (total : ℕ)
  (zeros : ℕ)
  (ones : ℕ)
  (h_total : total = zeros + ones)

/-- Represents the number of balls drawn from the bag -/
def drawn : ℕ := 5

/-- The specific bag described in the problem -/
def problem_bag : Bag :=
  { total := 10
  , zeros := 5
  , ones := 5
  , h_total := rfl }

/-- The probability of drawing 5 balls with sum of marks less than 2 or greater than 3 -/
def probability (b : Bag) : ℚ :=
  38 / 63

theorem probability_sum_less_2_or_greater_3 :
  probability problem_bag = 38 / 63 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_less_2_or_greater_3_l2108_210838


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l2108_210824

/-- The capacity of a water tank in gallons. -/
def tank_capacity : ℝ := 270

/-- The difference in gallons between 40% full and 40% empty. -/
def gallon_difference : ℝ := 54

theorem tank_capacity_proof : 
  tank_capacity = 270 ∧ 
  (0.4 * tank_capacity) - (0.6 * tank_capacity) = gallon_difference :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l2108_210824


namespace NUMINAMATH_CALUDE_two_card_selections_65_l2108_210868

/-- The number of ways to select two different cards from a deck of 65 cards, where the order of selection matters. -/
def two_card_selections (total_cards : ℕ) : ℕ :=
  total_cards * (total_cards - 1)

/-- Theorem stating that selecting two different cards from a deck of 65 cards, where the order matters, can be done in 4160 ways. -/
theorem two_card_selections_65 :
  two_card_selections 65 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_two_card_selections_65_l2108_210868


namespace NUMINAMATH_CALUDE_correct_calculation_l2108_210825

theorem correct_calculation (a b : ℝ) : 4 * a^2 * b - 3 * b * a^2 = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2108_210825


namespace NUMINAMATH_CALUDE_num_bedrooms_is_three_l2108_210828

/-- The number of bedrooms in the house -/
def num_bedrooms : ℕ := 3

/-- Time to renovate one bedroom (in hours) -/
def bedroom_time : ℕ := 4

/-- Time to renovate the kitchen (in hours) -/
def kitchen_time : ℕ := 6

/-- Total renovation time (in hours) -/
def total_time : ℕ := 54

/-- Theorem: The number of bedrooms is 3 given the renovation times -/
theorem num_bedrooms_is_three :
  num_bedrooms = 3 ∧
  bedroom_time = 4 ∧
  kitchen_time = 6 ∧
  total_time = 54 ∧
  total_time = num_bedrooms * bedroom_time + kitchen_time + 2 * (num_bedrooms * bedroom_time + kitchen_time) :=
by sorry

end NUMINAMATH_CALUDE_num_bedrooms_is_three_l2108_210828


namespace NUMINAMATH_CALUDE_root_transformation_equation_l2108_210889

theorem root_transformation_equation : 
  ∀ (p q r s : ℂ),
  (p^4 + 4*p^3 - 5 = 0) → 
  (q^4 + 4*q^3 - 5 = 0) → 
  (r^4 + 4*r^3 - 5 = 0) → 
  (s^4 + 4*s^3 - 5 = 0) → 
  ∃ (x : ℂ),
  (x = (p+q+r)/s^3 ∨ x = (p+q+s)/r^3 ∨ x = (p+r+s)/q^3 ∨ x = (q+r+s)/p^3) →
  (5*x^6 - x^2 + 4*x = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_equation_l2108_210889


namespace NUMINAMATH_CALUDE_consecutive_multiples_of_twelve_l2108_210802

theorem consecutive_multiples_of_twelve (A B : ℕ) (h1 : Nat.gcd A B = 12) (h2 : A > B) (h3 : A - B = 12) :
  ∃ (m : ℕ), A = 12 * (m + 1) ∧ B = 12 * m :=
sorry

end NUMINAMATH_CALUDE_consecutive_multiples_of_twelve_l2108_210802


namespace NUMINAMATH_CALUDE_voting_ratio_l2108_210822

/-- Given a voting scenario where:
    - 2/9 of the votes have been counted
    - 3/4 of the counted votes are in favor
    - 0.7857142857142856 of the remaining votes are against
    Prove that the ratio of total votes against to total votes in favor is 4:1 -/
theorem voting_ratio (V : ℝ) (hV : V > 0) : 
  let counted := (2/9) * V
  let in_favor := (3/4) * counted
  let remaining := V - counted
  let against_remaining := 0.7857142857142856 * remaining
  let total_against := ((1/4) * counted) + against_remaining
  let total_in_favor := in_favor
  (total_against / total_in_favor) = 4 := by
  sorry


end NUMINAMATH_CALUDE_voting_ratio_l2108_210822


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_three_sqrt_three_l2108_210801

theorem sqrt_difference_equals_three_sqrt_three : 
  Real.sqrt 75 - Real.sqrt 12 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_three_sqrt_three_l2108_210801


namespace NUMINAMATH_CALUDE_floor_sum_equals_155_l2108_210881

theorem floor_sum_equals_155 (p q r s : ℝ) : 
  p > 0 → q > 0 → r > 0 → s > 0 →
  p^2 + q^2 = 3024 →
  r^2 + s^2 = 3024 →
  p * r = 1500 →
  q * s = 1500 →
  ⌊p + q + r + s⌋ = 155 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_equals_155_l2108_210881


namespace NUMINAMATH_CALUDE_absolute_value_expression_l2108_210805

theorem absolute_value_expression : 
  let x : ℤ := -2023
  ‖‖|x| - (x + 3)‖ - (|x| - 3)‖ - (x - 3) = 4049 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l2108_210805


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l2108_210816

theorem unique_modular_congruence : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -215 ≡ n [ZMOD 23] ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l2108_210816


namespace NUMINAMATH_CALUDE_original_number_proof_l2108_210885

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) :
  x = Real.sqrt 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2108_210885


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2108_210821

def set_A : Set Int := {x | |x| < 3}
def set_B : Set Int := {x | |x| > 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2108_210821


namespace NUMINAMATH_CALUDE_digit_120th_of_7_26th_l2108_210853

theorem digit_120th_of_7_26th : ∃ (seq : ℕ → ℕ), 
  (∀ n, seq n < 10) ∧ 
  (∀ n, seq (n + 9) = seq n) ∧
  (∀ n, (7 * 10^n) % 26 = (seq n * 10^8 + seq (n+1) * 10^7 + seq (n+2) * 10^6 + 
                           seq (n+3) * 10^5 + seq (n+4) * 10^4 + seq (n+5) * 10^3 + 
                           seq (n+6) * 10^2 + seq (n+7) * 10 + seq (n+8)) % 26) ∧
  seq 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_120th_of_7_26th_l2108_210853


namespace NUMINAMATH_CALUDE_eight_six_four_combinations_l2108_210827

/-- The number of unique outfit combinations given the number of shirts, ties, and belts. -/
def outfitCombinations (shirts : ℕ) (ties : ℕ) (belts : ℕ) : ℕ :=
  shirts * ties * belts

/-- Theorem stating that 8 shirts, 6 ties, and 4 belts result in 192 unique combinations. -/
theorem eight_six_four_combinations :
  outfitCombinations 8 6 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_eight_six_four_combinations_l2108_210827


namespace NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l2108_210829

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  total_cubes : ℕ
  painted_per_face : ℕ

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpainted_cubes (cube : PaintedCube) : ℕ :=
  cube.total_cubes - (6 * cube.painted_per_face - 12)

/-- Theorem stating that a 4x4x4 cube with 6 painted squares per face has 40 unpainted unit cubes -/
theorem unpainted_cubes_4x4x4 :
  let cube : PaintedCube := ⟨4, 64, 6⟩
  unpainted_cubes cube = 40 := by sorry

end NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l2108_210829


namespace NUMINAMATH_CALUDE_expression_equality_l2108_210831

theorem expression_equality : 
  |1 - Real.sqrt 3| + 3 * Real.tan (30 * π / 180) - (1/2)⁻¹ + (3 - π)^0 = 3.732 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2108_210831


namespace NUMINAMATH_CALUDE_gardener_path_tiles_l2108_210887

def park_width : ℕ := 13
def park_length : ℕ := 19

theorem gardener_path_tiles :
  ∀ (avoid : ℕ), avoid = 1 →
  (park_width + park_length - Nat.gcd park_width park_length) - avoid = 30 := by
sorry

end NUMINAMATH_CALUDE_gardener_path_tiles_l2108_210887
