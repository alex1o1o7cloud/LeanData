import Mathlib

namespace divisibility_by_1897_l463_46352

theorem divisibility_by_1897 (n : ℕ) : 
  1897 ∣ (2903^n - 803^n - 464^n + 261^n) := by
  sorry

end divisibility_by_1897_l463_46352


namespace bucket_capacity_proof_l463_46369

/-- The capacity of a bucket that satisfies the given conditions -/
def bucket_capacity : ℝ :=
  let tank_capacity : ℝ := 48
  let small_bucket_capacity : ℝ := 3
  3

theorem bucket_capacity_proof :
  let tank_capacity : ℝ := 48
  let small_bucket_capacity : ℝ := 3
  (tank_capacity / bucket_capacity : ℝ) = (tank_capacity / small_bucket_capacity) - 4 := by
  sorry

#check bucket_capacity_proof

end bucket_capacity_proof_l463_46369


namespace solve_equation_l463_46312

theorem solve_equation (x : ℝ) : 3 * x + 12 = (1/3) * (7 * x + 42) → x = 3 := by
  sorry

end solve_equation_l463_46312


namespace binomial_12_10_equals_66_l463_46303

theorem binomial_12_10_equals_66 : Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_10_equals_66_l463_46303


namespace fly_path_length_l463_46323

theorem fly_path_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c = 5) 
  (h4 : a^2 + b^2 = c^2) : ∃ (path_length : ℝ), path_length > 10 ∧ 
  path_length = 5 * c := by sorry

end fly_path_length_l463_46323


namespace situp_competition_result_l463_46341

/-- Adam's sit-up performance -/
def adam_situps (round : ℕ) : ℕ :=
  40 - 8 * (round - 1)

/-- Barney's sit-up performance -/
def barney_situps : ℕ := 45

/-- Carrie's sit-up performance -/
def carrie_situps : ℕ := 2 * barney_situps

/-- Jerrie's sit-up performance -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- Total sit-ups for Adam -/
def adam_total : ℕ :=
  (adam_situps 1) + (adam_situps 2) + (adam_situps 3)

/-- Total sit-ups for Barney -/
def barney_total : ℕ := barney_situps * 5

/-- Total sit-ups for Carrie -/
def carrie_total : ℕ := carrie_situps * 4

/-- Total sit-ups for Jerrie -/
def jerrie_total : ℕ := jerrie_situps * 6

/-- The combined total of sit-ups -/
def combined_total : ℕ :=
  adam_total + barney_total + carrie_total + jerrie_total

theorem situp_competition_result :
  combined_total = 1251 := by
  sorry

end situp_competition_result_l463_46341


namespace geometric_sequence_property_l463_46391

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_product : a 3 * a 7 = 8) : 
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_property_l463_46391


namespace initial_puppies_count_l463_46306

/-- The number of puppies Alyssa gave to her friends -/
def puppies_given : ℕ := 7

/-- The number of puppies Alyssa has now -/
def puppies_remaining : ℕ := 5

/-- The initial number of puppies Alyssa had -/
def initial_puppies : ℕ := puppies_given + puppies_remaining

theorem initial_puppies_count : initial_puppies = 12 := by
  sorry

end initial_puppies_count_l463_46306


namespace greatest_base_six_digit_sum_l463_46316

/-- Represents a positive integer in base 6 as a list of digits (least significant first) -/
def BaseNRepr (n : ℕ) : List ℕ :=
  sorry

/-- Computes the sum of digits in a base 6 representation -/
def sumDigits (repr : List ℕ) : ℕ :=
  sorry

theorem greatest_base_six_digit_sum :
  (∀ n : ℕ, n > 0 → n < 2401 → sumDigits (BaseNRepr n) ≤ 12) ∧
  (∃ n : ℕ, n > 0 ∧ n < 2401 ∧ sumDigits (BaseNRepr n) = 12) :=
sorry

end greatest_base_six_digit_sum_l463_46316


namespace geometric_series_ratio_l463_46394

theorem geometric_series_ratio (r : ℝ) (h : r ≠ 1) :
  (∃ (a : ℝ), a / (1 - r) = 64 * (a * r^4) / (1 - r)) →
  r = 1/2 := by
sorry

end geometric_series_ratio_l463_46394


namespace sphere_with_n_plus_one_points_l463_46346

open Set

variable {α : Type*} [MetricSpace α]

theorem sphere_with_n_plus_one_points
  (m n : ℕ)
  (points : Finset α)
  (h_card : points.card = m * n + 1)
  (h_distance : ∀ (subset : Finset α), subset ⊆ points → subset.card = m + 1 →
    ∃ (x y : α), x ∈ subset ∧ y ∈ subset ∧ x ≠ y ∧ dist x y ≤ 1) :
  ∃ (center : α), ∃ (subset : Finset α),
    subset ⊆ points ∧
    subset.card = n + 1 ∧
    ∀ x ∈ subset, dist center x ≤ 1 :=
sorry

end sphere_with_n_plus_one_points_l463_46346


namespace digit_sum_equals_sixteen_l463_46335

/-- Given distinct digits a, b, and c satisfying aba + aba = cbc,
    prove that a + b + c = 16 -/
theorem digit_sum_equals_sixteen
  (a b c : ℕ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_digits : a < 10 ∧ b < 10 ∧ c < 10)
  (h_equation : 100 * a + 10 * b + a + 100 * a + 10 * b + a = 100 * c + 10 * b + c) :
  a + b + c = 16 := by
  sorry

end digit_sum_equals_sixteen_l463_46335


namespace garden_area_l463_46340

/-- Calculates the area of a garden given property dimensions and garden proportions -/
theorem garden_area 
  (property_width : ℝ) 
  (property_length : ℝ) 
  (garden_width_ratio : ℝ) 
  (garden_length_ratio : ℝ) 
  (h1 : property_width = 1000) 
  (h2 : property_length = 2250) 
  (h3 : garden_width_ratio = 1 / 8) 
  (h4 : garden_length_ratio = 1 / 10) : 
  garden_width_ratio * property_width * garden_length_ratio * property_length = 28125 := by
  sorry

#check garden_area

end garden_area_l463_46340


namespace intersection_equals_singleton_two_l463_46397

def M : Set ℤ := {1, 2, 3, 4}
def N : Set ℤ := {-2, 2}

theorem intersection_equals_singleton_two : M ∩ N = {2} := by sorry

end intersection_equals_singleton_two_l463_46397


namespace final_comic_book_count_l463_46317

def initial_books : ℕ := 22
def books_bought : ℕ := 6

theorem final_comic_book_count :
  (initial_books / 2 + books_bought : ℕ) = 17 :=
by
  sorry

end final_comic_book_count_l463_46317


namespace sum_with_rearrangement_not_1999_nines_sum_with_rearrangement_1010_divisible_by_10_l463_46366

-- Define a function to represent digit rearrangement
def digitRearrangement (n : ℕ) : ℕ := sorry

-- Define a function to check if a number consists of 1999 nines
def is1999Nines (n : ℕ) : Prop := sorry

-- Part (a)
theorem sum_with_rearrangement_not_1999_nines (n : ℕ) : 
  ¬(is1999Nines (n + digitRearrangement n)) := sorry

-- Part (b)
theorem sum_with_rearrangement_1010_divisible_by_10 (n : ℕ) : 
  n + digitRearrangement n = 1010 → n % 10 = 0 := sorry

end sum_with_rearrangement_not_1999_nines_sum_with_rearrangement_1010_divisible_by_10_l463_46366


namespace b_power_a_equals_sixteen_l463_46372

theorem b_power_a_equals_sixteen (a b : ℝ) : 
  b = Real.sqrt (2 - a) + Real.sqrt (a - 2) - 4 → b^a = 16 := by
sorry

end b_power_a_equals_sixteen_l463_46372


namespace smallest_n_divisible_by_ten_thousand_l463_46349

def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_n_divisible_by_ten_thousand :
  ∀ n : ℕ, n > 0 → n < 9375 → ¬(10000 ∣ sum_of_naturals n) ∧ (10000 ∣ sum_of_naturals 9375) :=
by sorry

end smallest_n_divisible_by_ten_thousand_l463_46349


namespace value_of_x_l463_46328

theorem value_of_x (x y z : ℚ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 80) :
  x = 20 / 3 := by
sorry

end value_of_x_l463_46328


namespace correct_answer_l463_46386

theorem correct_answer : ∃ x : ℤ, (x + 3 = 45) ∧ (x - 3 = 39) := by
  sorry

end correct_answer_l463_46386


namespace simplify_expression_l463_46339

theorem simplify_expression (b : ℝ) : 3*b*(3*b^3 + 2*b) - 2*b^2 = 9*b^4 + 4*b^2 := by
  sorry

end simplify_expression_l463_46339


namespace OPRQ_shapes_l463_46389

-- Define the points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrilateral OPRQ
structure Quadrilateral where
  O : Point2D
  P : Point2D
  R : Point2D
  Q : Point2D

-- Define the conditions for parallelogram, rectangle, and rhombus
def is_parallelogram (quad : Quadrilateral) : Prop :=
  ∃ k l : ℝ, k ≠ 0 ∧ l ≠ 0 ∧
  quad.R.x = k * quad.P.x + l * quad.Q.x ∧
  quad.R.y = k * quad.P.y + l * quad.Q.y

def is_rectangle (quad : Quadrilateral) : Prop :=
  is_parallelogram quad ∧
  quad.P.x * quad.Q.x + quad.P.y * quad.Q.y = 0

def is_rhombus (quad : Quadrilateral) : Prop :=
  is_parallelogram quad ∧
  quad.P.x^2 + quad.P.y^2 = quad.Q.x^2 + quad.Q.y^2

-- Main theorem
theorem OPRQ_shapes (P Q : Point2D) (h : P ≠ Q) :
  ∃ (R : Point2D) (quad : Quadrilateral),
    quad.O = ⟨0, 0⟩ ∧ quad.P = P ∧ quad.Q = Q ∧ quad.R = R ∧
    (is_parallelogram quad ∨ is_rectangle quad ∨ is_rhombus quad) :=
  sorry

end OPRQ_shapes_l463_46389


namespace cricket_innings_calculation_l463_46311

/-- The number of innings played by a cricket player -/
def innings : ℕ := sorry

/-- The current average runs per innings -/
def current_average : ℚ := 22

/-- The increase in average after scoring 92 runs in the next innings -/
def average_increase : ℚ := 5

/-- The runs scored in the next innings -/
def next_innings_runs : ℕ := 92

theorem cricket_innings_calculation :
  (innings * current_average + next_innings_runs) / (innings + 1) = current_average + average_increase →
  innings = 13 := by sorry

end cricket_innings_calculation_l463_46311


namespace normal_distribution_std_dev_l463_46329

/-- For a normal distribution with mean 10.5, if a value 2 standard deviations
    below the mean is 8.5, then the standard deviation is 1. -/
theorem normal_distribution_std_dev (μ σ : ℝ) (x : ℝ) : 
  μ = 10.5 → x = μ - 2 * σ → x = 8.5 → σ = 1 := by sorry

end normal_distribution_std_dev_l463_46329


namespace quadratic_root_implies_m_value_l463_46345

theorem quadratic_root_implies_m_value :
  ∀ m : ℝ, ((-1)^2 - 2*(-1) + m = 0) → m = -3 := by
  sorry

end quadratic_root_implies_m_value_l463_46345


namespace wood_cutting_l463_46325

/-- Given a piece of wood that can be sawed into 9 sections of 4 meters each,
    prove that 11 cuts are needed to saw it into 3-meter sections. -/
theorem wood_cutting (wood_length : ℕ) (num_long_sections : ℕ) (long_section_length : ℕ) 
  (short_section_length : ℕ) (h1 : wood_length = num_long_sections * long_section_length)
  (h2 : num_long_sections = 9) (h3 : long_section_length = 4) (h4 : short_section_length = 3) : 
  (wood_length / short_section_length) - 1 = 11 := by
  sorry

end wood_cutting_l463_46325


namespace floor_ceiling_sum_seven_l463_46377

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℝ) + (⌈x⌉ : ℝ) = 7 ↔ 3 < x ∧ x < 4 :=
by sorry

end floor_ceiling_sum_seven_l463_46377


namespace not_in_second_quadrant_l463_46302

/-- A linear function y = x - 2 -/
def f (x : ℝ) : ℝ := x - 2

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem: The graph of y = x - 2 does not pass through the second quadrant -/
theorem not_in_second_quadrant :
  ¬ ∃ (x : ℝ), second_quadrant x (f x) := by
  sorry

end not_in_second_quadrant_l463_46302


namespace positive_integer_solution_for_exponential_equation_l463_46309

theorem positive_integer_solution_for_exponential_equation :
  ∀ (n a b c : ℕ), 
    n > 1 → a > 0 → b > 0 → c > 0 →
    n^a + n^b = n^c →
    (n = 2 ∧ b = a ∧ c = a + 1) :=
by
  sorry

end positive_integer_solution_for_exponential_equation_l463_46309


namespace third_circle_radius_l463_46318

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 34) (h₂ : r₂ = 14) : 
  π * r₃^2 = π * (r₁^2 - r₂^2) → r₃ = 8 * Real.sqrt 15 :=
by sorry

end third_circle_radius_l463_46318


namespace chocolate_distribution_l463_46363

theorem chocolate_distribution (total_chocolates : ℕ) (total_children : ℕ) 
  (boys : ℕ) (girls : ℕ) (chocolates_per_boy : ℕ) :
  total_chocolates = 3000 →
  total_children = 120 →
  boys = 60 →
  girls = 60 →
  chocolates_per_boy = 2 →
  (total_chocolates - boys * chocolates_per_boy) / girls = 48 :=
by
  sorry

end chocolate_distribution_l463_46363


namespace people_sitting_on_benches_l463_46333

theorem people_sitting_on_benches (num_benches : ℕ) (bench_capacity : ℕ) (available_spaces : ℕ) : 
  num_benches = 50 → bench_capacity = 4 → available_spaces = 120 → 
  num_benches * bench_capacity - available_spaces = 80 :=
by
  sorry

end people_sitting_on_benches_l463_46333


namespace cube_edge_increase_l463_46310

theorem cube_edge_increase (e : ℝ) (h : e > 0) :
  let A := 6 * e^2
  let A' := 2.25 * A
  let e' := Real.sqrt (A' / 6)
  (e' - e) / e = 0.5 := by
  sorry

end cube_edge_increase_l463_46310


namespace infinite_solutions_ratio_l463_46378

theorem infinite_solutions_ratio (a b c : ℚ) : 
  (∀ x, a * x^2 + b * x + c = (x - 1) * (2 * x + 1)) → 
  a = 2 ∧ b = -1 :=
by sorry

end infinite_solutions_ratio_l463_46378


namespace geometric_sequence_first_term_l463_46320

theorem geometric_sequence_first_term
  (a : ℕ → ℕ)
  (h1 : a 2 = 3)
  (h2 : a 3 = 9)
  (h3 : a 4 = 27)
  (h4 : a 5 = 81)
  (h5 : a 6 = 243)
  (h_geometric : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n) :
  a 1 = 1 := by
sorry

end geometric_sequence_first_term_l463_46320


namespace theresas_work_hours_l463_46355

theorem theresas_work_hours (total_weeks : ℕ) (target_average : ℕ) 
  (week1 week2 week3 week4 : ℕ) (additional_task : ℕ) :
  total_weeks = 5 →
  target_average = 12 →
  week1 = 10 →
  week2 = 14 →
  week3 = 11 →
  week4 = 9 →
  additional_task = 1 →
  ∃ (week5 : ℕ), 
    (week1 + week2 + week3 + week4 + week5 + additional_task) / total_weeks = target_average ∧
    week5 = 15 :=
by sorry

end theresas_work_hours_l463_46355


namespace shoes_in_box_l463_46358

/-- The number of pairs of shoes in the box -/
def num_pairs : ℕ := 5

/-- The probability of selecting two matching shoes at random -/
def prob_matching : ℚ := 1 / 9

/-- The total number of shoes in the box -/
def total_shoes : ℕ := 2 * num_pairs

/-- Theorem stating that given the conditions, the total number of shoes is 10 -/
theorem shoes_in_box :
  (num_pairs = 5) →
  (prob_matching = 1 / 9) →
  (total_shoes = 10) := by
  sorry


end shoes_in_box_l463_46358


namespace max_value_condition_l463_46319

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + (2*a - 1) * x

theorem max_value_condition (a : ℝ) :
  (∀ x > 0, f a x ≤ f a 1) → a > 1/2 :=
by sorry

end max_value_condition_l463_46319


namespace vector_parallelism_transitive_l463_46304

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (v w : V) : Prop := ∃ (k : ℝ), v = k • w

theorem vector_parallelism_transitive (a b c : V) 
  (hab : parallel a b) (hbc : parallel b c) (hb : b ≠ 0) : 
  parallel a c := by
  sorry

end vector_parallelism_transitive_l463_46304


namespace linear_function_k_value_l463_46338

theorem linear_function_k_value : ∀ k : ℝ, 
  (∀ x y : ℝ, y = k * x + 3) →  -- Linear function condition
  (2 = k * 1 + 3) →             -- Passes through (1, 2)
  k = -1 := by
sorry

end linear_function_k_value_l463_46338


namespace arithmetic_sequence_property_l463_46356

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n + q

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 4 + a 8 = π →
  a 6 * (a 2 + 2 * a 6 + a 10) = π^2 := by
  sorry

end arithmetic_sequence_property_l463_46356


namespace stating_max_books_borrowed_is_eight_l463_46314

/-- Represents the maximum number of books borrowed by a single student -/
def max_books_borrowed (total_students : ℕ) 
                       (zero_book_students : ℕ) 
                       (one_book_students : ℕ) 
                       (two_book_students : ℕ) 
                       (avg_books_per_student : ℕ) : ℕ :=
  let total_books := total_students * avg_books_per_student
  let remaining_students := total_students - (zero_book_students + one_book_students + two_book_students)
  let accounted_books := one_book_students + 2 * two_book_students
  let remaining_books := total_books - accounted_books
  remaining_books - (3 * (remaining_students - 1))

/-- 
Theorem stating that given the conditions in the problem, 
the maximum number of books borrowed by a single student is 8.
-/
theorem max_books_borrowed_is_eight :
  max_books_borrowed 35 2 12 10 2 = 8 := by
  sorry

end stating_max_books_borrowed_is_eight_l463_46314


namespace T_bounds_l463_46313

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 1 ∧ y = (3*x + 5) / (x + 3)}

theorem T_bounds :
  ∃ (q P : ℝ),
    (∀ y ∈ T, q ≤ y) ∧
    (∀ y ∈ T, y ≤ P) ∧
    q ∈ T ∧
    P ∉ T :=
sorry

end T_bounds_l463_46313


namespace geometric_series_first_term_l463_46330

theorem geometric_series_first_term 
  (sum : ℝ) 
  (sum_squares : ℝ) 
  (h1 : sum = 20) 
  (h2 : sum_squares = 80) : 
  ∃ (a r : ℝ), 
    a / (1 - r) = sum ∧ 
    a^2 / (1 - r^2) = sum_squares ∧ 
    a = 20 / 3 := by 
sorry

end geometric_series_first_term_l463_46330


namespace school_children_count_prove_school_children_count_l463_46344

theorem school_children_count : ℕ → Prop :=
  fun total_children =>
    let total_bananas := 2 * total_children
    total_bananas = 4 * (total_children - 350) →
    total_children = 700

-- Proof
theorem prove_school_children_count :
  ∃ (n : ℕ), school_children_count n :=
by
  sorry

end school_children_count_prove_school_children_count_l463_46344


namespace not_perfect_square_l463_46388

theorem not_perfect_square (n : ℕ+) : ¬∃ (m : ℕ), 4 * n^2 + 4 * n + 4 = m^2 := by
  sorry

end not_perfect_square_l463_46388


namespace quadratic_real_roots_condition_l463_46392

theorem quadratic_real_roots_condition (k : ℝ) : 
  (k ≠ 0) → 
  (∃ x : ℝ, k * x^2 - x + 1 = 0) ↔ 
  (k ≤ 1/4 ∧ k ≠ 0) :=
by sorry

end quadratic_real_roots_condition_l463_46392


namespace scientific_notation_of_2600000_l463_46362

theorem scientific_notation_of_2600000 :
  2600000 = 2.6 * (10 ^ 6) :=
by sorry

end scientific_notation_of_2600000_l463_46362


namespace total_sales_proof_l463_46395

def window_screen_sales (march_sales : ℕ) : ℕ :=
  let february_sales := march_sales / 4
  let january_sales := february_sales / 2
  january_sales + february_sales + march_sales

theorem total_sales_proof (march_sales : ℕ) (h : march_sales = 8800) :
  window_screen_sales march_sales = 12100 := by
  sorry

end total_sales_proof_l463_46395


namespace fixed_point_theorem_proof_l463_46343

def fixed_point_theorem (f : ℝ → ℝ) (h_inverse : Function.Bijective f) : Prop :=
  let f_inv := Function.invFun f
  (f_inv (-(-1) + 2) = 2) → (f ((-3) - 1) = -3)

theorem fixed_point_theorem_proof (f : ℝ → ℝ) (h_inverse : Function.Bijective f) :
  fixed_point_theorem f h_inverse := by
  sorry

end fixed_point_theorem_proof_l463_46343


namespace least_three_digit_7_heavy_l463_46353

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_heavy : 
  (∀ m : ℕ, is_three_digit m → is_7_heavy m → 103 ≤ m) ∧ 
  is_three_digit 103 ∧ 
  is_7_heavy 103 :=
sorry

end least_three_digit_7_heavy_l463_46353


namespace subset_implies_range_l463_46365

-- Define set A
def A : Set ℝ := {x | x^2 ≤ 5*x - 4}

-- Define set M parameterized by a
def M (a : ℝ) : Set ℝ := {x | x^2 - (a+2)*x + 2*a ≤ 0}

-- Theorem statement
theorem subset_implies_range (a : ℝ) : M a ⊆ A ↔ a ∈ Set.Icc 1 4 := by
  sorry

end subset_implies_range_l463_46365


namespace fraction_evaluation_l463_46368

theorem fraction_evaluation : (450 : ℚ) / (6 * 5 - 10 / 2) = 18 := by
  sorry

end fraction_evaluation_l463_46368


namespace remove_two_gives_eight_point_five_l463_46398

def original_list : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

def remove_number (list : List ℕ) (n : ℕ) : List ℕ :=
  list.filter (· ≠ n)

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem remove_two_gives_eight_point_five :
  average (remove_number original_list 2) = 8.5 := by
  sorry

end remove_two_gives_eight_point_five_l463_46398


namespace find_d_l463_46300

theorem find_d : ∃ d : ℚ, 
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * n^2 + 19 * n - 28 = 0) ∧
  (4 * (d - ↑⌊d⌋)^2 - 11 * (d - ↑⌊d⌋) + 3 = 0) ∧
  (0 ≤ d - ↑⌊d⌋ ∧ d - ↑⌊d⌋ < 1) ∧
  d = -29/4 := by
  sorry

end find_d_l463_46300


namespace profit_maximized_at_optimal_price_l463_46347

/-- The profit function for a product with a cost of 30 yuan per item,
    where x is the selling price and (200 - x) is the quantity sold. -/
def profit_function (x : ℝ) : ℝ := -x^2 + 230*x - 6000

/-- The selling price that maximizes the profit. -/
def optimal_price : ℝ := 115

theorem profit_maximized_at_optimal_price :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_price :=
sorry

end profit_maximized_at_optimal_price_l463_46347


namespace unique_paintable_number_l463_46374

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def paints_every_nth (start : ℕ) (step : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = start + k * step

def is_paintable (h t u : ℕ) : Prop :=
  h = 4 ∧
  t % 2 ≠ 0 ∧
  is_prime u ∧
  ∀ n : ℕ, n > 0 →
    (paints_every_nth 1 4 n ∨ paints_every_nth 3 t n ∨ paints_every_nth 5 u n) ∧
    ¬(paints_every_nth 1 4 n ∧ paints_every_nth 3 t n) ∧
    ¬(paints_every_nth 1 4 n ∧ paints_every_nth 5 u n) ∧
    ¬(paints_every_nth 3 t n ∧ paints_every_nth 5 u n)

theorem unique_paintable_number :
  ∀ h t u : ℕ, is_paintable h t u → 100 * t + 10 * u + h = 354 :=
sorry

end unique_paintable_number_l463_46374


namespace more_girls_than_boys_l463_46396

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 650 → boys = 272 → girls = total_students - boys → 
  girls - boys = 106 := by sorry

end more_girls_than_boys_l463_46396


namespace square_triangle_perimeter_relation_l463_46382

/-- Given a square with perimeter 40 and a larger equilateral triangle with 
    perimeter a + b√p (where p is prime), prove that if a = 30, b = 10, 
    and p = 3, then 7a + 5b + 3p = 269. -/
theorem square_triangle_perimeter_relation 
  (square_perimeter : ℝ) 
  (a b : ℝ) 
  (p : ℕ) 
  (h1 : square_perimeter = 40)
  (h2 : Nat.Prime p)
  (h3 : a = 30)
  (h4 : b = 10)
  (h5 : p = 3)
  : 7 * a + 5 * b + 3 * ↑p = 269 := by
  sorry

end square_triangle_perimeter_relation_l463_46382


namespace bella_steps_l463_46371

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℝ := 10560

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℝ := 3

/-- The length of Bella's step in feet -/
def step_length : ℝ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps : ℕ := 880

theorem bella_steps :
  (distance / (1 + speed_ratio)) / step_length = steps := by
  sorry

end bella_steps_l463_46371


namespace salary_distribution_l463_46324

def salary : ℚ := 140000

def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5
def remaining : ℚ := 14000

def food_fraction : ℚ := 1/5

theorem salary_distribution :
  food_fraction * salary + rent_fraction * salary + clothes_fraction * salary + remaining = salary :=
by sorry

end salary_distribution_l463_46324


namespace snacks_ryan_can_buy_l463_46376

def one_way_travel_time : ℝ := 2
def snack_cost (round_trip_time : ℝ) : ℝ := 10 * round_trip_time
def ryan_budget : ℝ := 2000

theorem snacks_ryan_can_buy :
  (ryan_budget / snack_cost (2 * one_way_travel_time)) = 50 := by
  sorry

end snacks_ryan_can_buy_l463_46376


namespace interest_rate_calculation_l463_46342

/-- Computes the annual interest rate given the principal, time, compounding frequency, and final amount -/
def calculate_interest_rate (principal : ℝ) (time : ℝ) (compounding_frequency : ℕ) (final_amount : ℝ) : ℝ :=
  sorry

theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (compounding_frequency : ℕ) (final_amount : ℝ) 
  (h1 : principal = 6000)
  (h2 : time = 1.5)
  (h3 : compounding_frequency = 2)
  (h4 : final_amount = 6000 + 945.75) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |calculate_interest_rate principal time compounding_frequency final_amount - 0.099| < ε :=
sorry

end interest_rate_calculation_l463_46342


namespace original_class_size_l463_46321

theorem original_class_size (A B C : ℕ) (N : ℕ) (D : ℕ) :
  A = 40 →
  B = 32 →
  C = 36 →
  D = N * A →
  D + 8 * B = (N + 8) * C →
  N = 8 := by
sorry

end original_class_size_l463_46321


namespace equivalence_condition_l463_46399

theorem equivalence_condition (a b : ℝ) (h : a * b > 0) :
  (a > b) ↔ (1 / a < 1 / b) :=
by sorry

end equivalence_condition_l463_46399


namespace chair_color_probability_l463_46379

theorem chair_color_probability (black_chairs brown_chairs : ℕ) 
  (h1 : black_chairs = 15) (h2 : brown_chairs = 18) : 
  let total_chairs := black_chairs + brown_chairs
  (black_chairs : ℚ) / total_chairs * ((black_chairs - 1) / (total_chairs - 1)) + 
  (brown_chairs : ℚ) / total_chairs * ((brown_chairs - 1) / (total_chairs - 1)) = 
  (15 : ℚ) / 33 * 14 / 32 + 18 / 33 * 17 / 32 := by
sorry

end chair_color_probability_l463_46379


namespace f_values_l463_46337

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 3 * x + 4 else 4 * x - 1

theorem f_values : f (-3) = -5 ∧ f 2 = 7 := by
  sorry

end f_values_l463_46337


namespace probability_both_types_selected_l463_46308

def num_type_a : ℕ := 3
def num_type_b : ℕ := 2
def total_tvs : ℕ := num_type_a + num_type_b
def num_selected : ℕ := 3

theorem probability_both_types_selected :
  (Nat.choose num_type_a 2 * Nat.choose num_type_b 1 +
   Nat.choose num_type_a 1 * Nat.choose num_type_b 2) /
  Nat.choose total_tvs num_selected = 9 / 10 := by sorry

end probability_both_types_selected_l463_46308


namespace research_project_hours_difference_l463_46334

/-- The research project problem -/
theorem research_project_hours_difference
  (total_payment : ℝ)
  (wage_difference : ℝ)
  (wage_ratio : ℝ)
  (h1 : total_payment = 480)
  (h2 : wage_difference = 8)
  (h3 : wage_ratio = 1.5) :
  ∃ (hours_p hours_q : ℝ),
    hours_q - hours_p = 10 ∧
    hours_p * (wage_ratio * (total_payment / hours_q)) = total_payment ∧
    hours_q * (total_payment / hours_q) = total_payment ∧
    wage_ratio * (total_payment / hours_q) = (total_payment / hours_q) + wage_difference :=
by sorry


end research_project_hours_difference_l463_46334


namespace sum_and_count_theorem_l463_46326

def sum_of_range (a b : ℕ) : ℕ := ((b - a + 1) * (a + b)) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem : 
  sum_of_range 40 60 + count_even_in_range 40 60 = 1061 := by
  sorry

end sum_and_count_theorem_l463_46326


namespace no_square_possible_l463_46385

/-- Represents the lengths of sticks available -/
def stick_lengths : List ℕ := [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

/-- The total length of all sticks -/
def total_length : ℕ := stick_lengths.sum

/-- Predicate to check if a square can be formed -/
def can_form_square (lengths : List ℕ) : Prop :=
  ∃ (side_length : ℕ), side_length > 0 ∧ 
  4 * side_length = lengths.sum ∧
  ∃ (subset : List ℕ), subset.sum = side_length ∧ subset.toFinset ⊆ lengths.toFinset

theorem no_square_possible : ¬(can_form_square stick_lengths) := by
  sorry

end no_square_possible_l463_46385


namespace weight_sum_determination_l463_46381

/-- Given the weights of four people in pairs, prove that the sum of the weights of two specific people can be determined. -/
theorem weight_sum_determination (a b c d : ℝ) 
  (h1 : a + b = 280)
  (h2 : b + c = 230)
  (h3 : c + d = 250)
  (h4 : a + d = 300) :
  a + c = 250 := by
  sorry

end weight_sum_determination_l463_46381


namespace sequence_properties_l463_46375

/-- Sequence a_n with sum S_n satisfying S_n = 2a_n - 3 for n ∈ ℕ* -/
def S (a : ℕ+ → ℝ) (n : ℕ+) : ℝ := 2 * a n - 3

/-- Sequence b_n defined as b_n = (n-1)a_n -/
def b (a : ℕ+ → ℝ) (n : ℕ+) : ℝ := (n.val - 1) * a n

/-- Sum T_n of the first n terms of sequence b_n -/
def T (b : ℕ+ → ℝ) : ℕ+ → ℝ := fun n ↦ (Finset.range n.val).sum (fun i ↦ b ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_properties (a : ℕ+ → ℝ) (k : ℝ) :
  (∀ n : ℕ+, S a n = 2 * a n - 3) →
  (∀ n : ℕ+, a n = 3 * 2^(n.val - 1)) ∧
  (∀ n : ℕ+, T (b a) n = 3 * (n.val - 2) * 2^n.val + 6) ∧
  (∀ n : ℕ+, T (b a) n > k * a n + 16 * n.val - 26 → k < 0) := by
  sorry

end sequence_properties_l463_46375


namespace megan_candy_count_l463_46350

theorem megan_candy_count (mary_initial : ℕ) (megan : ℕ) : 
  mary_initial = 3 * megan →
  mary_initial + 10 = 25 →
  megan = 5 := by
sorry

end megan_candy_count_l463_46350


namespace range_of_f_inequality_l463_46390

open Real

noncomputable def f (x : ℝ) : ℝ := 2*x + sin x

theorem range_of_f_inequality (h1 : ∀ x ∈ Set.Ioo (-2) 2, HasDerivAt f (2 + cos x) x)
                              (h2 : f 0 = 0) :
  {x : ℝ | f (1 + x) + f (x - x^2) > 0} = Set.Ioo (1 - Real.sqrt 2) 1 := by
  sorry

end range_of_f_inequality_l463_46390


namespace alcohol_mixture_problem_l463_46367

/-- Given a 1 gallon container of 75% alcohol solution, if 0.4 gallon is drained off and
    replaced with x% alcohol solution to produce a 1 gallon 65% alcohol solution,
    then x = 50%. -/
theorem alcohol_mixture_problem (x : ℝ) : 
  (0.75 * (1 - 0.4) + 0.4 * (x / 100) = 0.65) → x = 50 := by
  sorry

end alcohol_mixture_problem_l463_46367


namespace max_value_2x_minus_y_l463_46387

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x + y - 1 < 0) 
  (h2 : x - y ≤ 0) 
  (h3 : x ≥ 0) : 
  ∀ z, z = 2*x - y → z ≤ (1/2 : ℝ) :=
by sorry

end max_value_2x_minus_y_l463_46387


namespace investment_percentage_proof_l463_46383

theorem investment_percentage_proof (total_investment : ℝ) (first_investment : ℝ) 
  (second_investment : ℝ) (second_rate : ℝ) (third_rate : ℝ) (yearly_income : ℝ) :
  total_investment = 10000 ∧ 
  first_investment = 4000 ∧ 
  second_investment = 3500 ∧ 
  second_rate = 0.04 ∧ 
  third_rate = 0.064 ∧ 
  yearly_income = 500 →
  ∃ x : ℝ, 
    x = 5 ∧ 
    first_investment * (x / 100) + second_investment * second_rate + 
    (total_investment - first_investment - second_investment) * third_rate = yearly_income :=
by sorry

end investment_percentage_proof_l463_46383


namespace alpha_more_cost_effective_regular_l463_46307

/-- Represents a fitness club with a monthly fee -/
structure FitnessClub where
  name : String
  monthlyFee : ℕ

/-- Calculates the yearly cost for a fitness club -/
def yearlyCost (club : FitnessClub) : ℕ :=
  club.monthlyFee * 12

/-- Calculates the cost per visit for a given number of visits -/
def costPerVisit (club : FitnessClub) (visits : ℕ) : ℚ :=
  (yearlyCost club : ℚ) / visits

/-- Represents the two attendance scenarios -/
inductive AttendancePattern
  | Regular
  | Sporadic

/-- Calculates the number of visits per year based on the attendance pattern -/
def visitsPerYear (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | .Regular => 96
  | .Sporadic => 48

/-- The main theorem stating that Alpha is more cost-effective for regular attendance -/
theorem alpha_more_cost_effective_regular :
  let alpha : FitnessClub := ⟨"Alpha", 999⟩
  let beta : FitnessClub := ⟨"Beta", 1299⟩
  let regularVisits := visitsPerYear AttendancePattern.Regular
  costPerVisit alpha regularVisits < costPerVisit beta regularVisits :=
by sorry

end alpha_more_cost_effective_regular_l463_46307


namespace factorization_perfect_square_factorization_difference_of_cubes_l463_46348

/-- Proves the factorization of a^2 + 2a + 1 -/
theorem factorization_perfect_square (a : ℝ) : a^2 + 2*a + 1 = (a + 1)^2 := by
  sorry

/-- Proves the factorization of a^3 - ab^2 -/
theorem factorization_difference_of_cubes (a b : ℝ) : a^3 - a*b^2 = a*(a + b)*(a - b) := by
  sorry

end factorization_perfect_square_factorization_difference_of_cubes_l463_46348


namespace scientific_notation_equivalence_l463_46327

theorem scientific_notation_equivalence : 
  6390000 = 6.39 * (10 ^ 6) := by sorry

end scientific_notation_equivalence_l463_46327


namespace sector_arc_length_l463_46373

/-- Given a sector with central angle π/3 and radius 3, its arc length is π. -/
theorem sector_arc_length (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 3) :
  θ * r = π := by
  sorry

end sector_arc_length_l463_46373


namespace digits_after_decimal_point_l463_46364

theorem digits_after_decimal_point : ∃ (n : ℕ), 
  (5^7 : ℚ) / (10^5 * 15625) = (1 : ℚ) / 10^n ∧ n = 5 := by
  sorry

end digits_after_decimal_point_l463_46364


namespace point_M_in_first_quadrant_l463_46332

/-- If point P(0,m) lies on the negative half-axis of the y-axis, 
    then point M(-m,-m+1) lies in the first quadrant. -/
theorem point_M_in_first_quadrant (m : ℝ) : 
  m < 0 → -m > 0 ∧ -m + 1 > 0 := by sorry

end point_M_in_first_quadrant_l463_46332


namespace infinitely_many_planes_through_point_parallel_to_line_l463_46331

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Predicate to check if a point is outside a line -/
def isOutside (P : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Predicate to check if a plane passes through a point -/
def passesThroughPoint (plane : Plane3D) (P : Point3D) : Prop :=
  sorry

/-- Predicate to check if a plane is parallel to a line -/
def isParallelToLine (plane : Plane3D) (l : Line3D) : Prop :=
  sorry

/-- The main theorem -/
theorem infinitely_many_planes_through_point_parallel_to_line
  (P : Point3D) (l : Line3D) (h : isOutside P l) :
  ∃ (f : ℕ → Plane3D), Function.Injective f ∧
    (∀ n : ℕ, passesThroughPoint (f n) P ∧ isParallelToLine (f n) l) :=
  sorry

end infinitely_many_planes_through_point_parallel_to_line_l463_46331


namespace abundant_product_l463_46360

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- A number is abundant if the sum of its divisors is greater than twice the number -/
def abundant (n : ℕ) : Prop := sigma n > 2 * n

/-- If a is abundant, then ab is abundant for any positive integer b -/
theorem abundant_product {a b : ℕ} (ha : a > 0) (hb : b > 0) (hab : abundant a) : abundant (a * b) := by
  sorry

end abundant_product_l463_46360


namespace max_y_coordinate_l463_46322

theorem max_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end max_y_coordinate_l463_46322


namespace time_addition_theorem_l463_46305

/-- Represents time in a 12-hour format -/
structure Time12 where
  hours : Nat
  minutes : Nat
  seconds : Nat
  isPM : Bool

/-- Represents a duration in hours, minutes, and seconds -/
structure Duration where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a time and returns the resulting time -/
def addDuration (t : Time12) (d : Duration) : Time12 := sorry

/-- Computes the sum of hours, minutes, and seconds for a given time -/
def sumComponents (t : Time12) : Nat := sorry

theorem time_addition_theorem (initialTime : Time12) (duration : Duration) :
  initialTime = Time12.mk 3 0 0 true →
  duration = Duration.mk 300 55 30 →
  (addDuration initialTime duration = Time12.mk 3 55 30 true) ∧
  (sumComponents (addDuration initialTime duration) = 88) := by sorry

end time_addition_theorem_l463_46305


namespace sum_common_terms_example_l463_46336

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℕ
  diff : ℕ
  last : ℕ

/-- Calculates the sum of common terms between two arithmetic sequences -/
def sumCommonTerms (seq1 seq2 : ArithmeticSequence) : ℕ :=
  sorry

theorem sum_common_terms_example :
  let seq1 : ArithmeticSequence := ⟨2, 4, 210⟩
  let seq2 : ArithmeticSequence := ⟨2, 6, 212⟩
  sumCommonTerms seq1 seq2 = 1872 :=
by sorry

end sum_common_terms_example_l463_46336


namespace lindas_savings_l463_46384

theorem lindas_savings (furniture_fraction : Real) (tv_cost : Real) 
  (refrigerator_percent : Real) (furniture_discount : Real) (tv_tax : Real) :
  furniture_fraction = 3/4 →
  tv_cost = 210 →
  refrigerator_percent = 20/100 →
  furniture_discount = 7/100 →
  tv_tax = 6/100 →
  ∃ (savings : Real),
    savings = 1898.40 ∧
    (furniture_fraction * savings * (1 - furniture_discount) + 
     tv_cost * (1 + tv_tax) + 
     tv_cost * (1 + refrigerator_percent)) = savings :=
by
  sorry


end lindas_savings_l463_46384


namespace conference_duration_theorem_l463_46351

/-- Calculates the duration of a conference in minutes, excluding the lunch break. -/
def conference_duration (total_hours : ℕ) (total_minutes : ℕ) (lunch_break : ℕ) : ℕ :=
  total_hours * 60 + total_minutes - lunch_break

/-- Proves that a conference lasting 8 hours and 40 minutes with a 15-minute lunch break
    has an active session time of 505 minutes. -/
theorem conference_duration_theorem :
  conference_duration 8 40 15 = 505 := by
  sorry

end conference_duration_theorem_l463_46351


namespace no_absolute_winner_probability_l463_46393

/-- Represents a player in the mini-tournament -/
inductive Player : Type
| Alyosha : Player
| Borya : Player
| Vasya : Player

/-- Represents the result of a match between two players -/
def MatchResult := Player → Player → ℝ

/-- The probability that there is no absolute winner in the mini-tournament -/
def noAbsoluteWinnerProbability (matchResult : MatchResult) : ℝ :=
  let p_AB := matchResult Player.Alyosha Player.Borya
  let p_BV := matchResult Player.Borya Player.Vasya
  0.24 * (1 - p_AB) * (1 - p_BV) + 0.36 * p_AB * (1 - p_BV)

/-- The main theorem stating that the probability of no absolute winner is 0.36 -/
theorem no_absolute_winner_probability (matchResult : MatchResult) 
  (h1 : matchResult Player.Alyosha Player.Borya = 0.6)
  (h2 : matchResult Player.Borya Player.Vasya = 0.4) :
  noAbsoluteWinnerProbability matchResult = 0.36 := by
  sorry


end no_absolute_winner_probability_l463_46393


namespace rank_squared_inequality_l463_46359

theorem rank_squared_inequality (A B : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : Matrix.rank A > Matrix.rank B) : 
  Matrix.rank (A ^ 2) ≥ Matrix.rank (B ^ 2) := by
  sorry

end rank_squared_inequality_l463_46359


namespace no_valid_m_l463_46370

/-- The trajectory of point M -/
def trajectory (x y m : ℝ) : Prop :=
  x^2 / 4 - y^2 / (m^2 - 4) = 1 ∧ x ≥ 2

/-- Line L -/
def line_L (x y : ℝ) : Prop :=
  y = (1/2) * x - 3

/-- Intersection points of trajectory and line L -/
def intersection_points (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory x₁ y₁ m ∧ trajectory x₂ y₂ m ∧
    line_L x₁ y₁ ∧ line_L x₂ y₂

/-- Vector dot product condition -/
def dot_product_condition (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory x₁ y₁ m ∧ trajectory x₂ y₂ m ∧
    line_L x₁ y₁ ∧ line_L x₂ y₂ ∧
    (x₁ * x₂ + (y₁ - 1) * (y₂ - 1) = 9/2)

theorem no_valid_m :
  ¬∃ m : ℝ, m > 2 ∧ intersection_points m ∧ dot_product_condition m :=
sorry

end no_valid_m_l463_46370


namespace partial_fraction_decomposition_sum_l463_46361

theorem partial_fraction_decomposition_sum (p q r A B C : ℝ) : 
  (p ≠ q ∧ p ≠ r ∧ q ≠ r) →
  (∀ (x : ℝ), x^3 - 16*x^2 + 72*x - 27 = (x - p) * (x - q) * (x - r)) →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 16*s^2 + 72*s - 27) = A / (s - p) + B / (s - q) + C / (s - r)) →
  A + B + C = 0 := by
sorry

end partial_fraction_decomposition_sum_l463_46361


namespace line_through_origin_l463_46301

variable (m n p : ℝ)
variable (h : m ≠ 0 ∨ n ≠ 0 ∨ p ≠ 0)

def line_set : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | ∃ (k : ℝ), x = k * m ∧ y = k * n ∧ z = k * p}

theorem line_through_origin (m n p : ℝ) (h : m ≠ 0 ∨ n ≠ 0 ∨ p ≠ 0) :
  ∃ (a b c : ℝ), line_set m n p = {(x, y, z) | a * x + b * y + c * z = 0} ∧
  (0, 0, 0) ∈ line_set m n p :=
sorry

end line_through_origin_l463_46301


namespace shaded_area_theorem_l463_46357

def circle_radius : ℝ := 2
def inner_circle_radius : ℝ := 1
def num_points : ℕ := 6
def num_symmetrical_parts : ℕ := 3

theorem shaded_area_theorem :
  let sector_angle : ℝ := 2 * Real.pi / num_points
  let sector_area : ℝ := (1 / 2) * circle_radius^2 * sector_angle
  let triangle_area : ℝ := (1 / 2) * circle_radius * inner_circle_radius * Real.sin (sector_angle / 2)
  let quadrilateral_area : ℝ := 2 * triangle_area
  let part_area : ℝ := sector_area + quadrilateral_area
  num_symmetrical_parts * part_area = 2 * Real.pi + 3 := by sorry

end shaded_area_theorem_l463_46357


namespace circle_area_through_isosceles_triangle_vertices_l463_46380

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_through_isosceles_triangle_vertices (a b c : ℝ) (h_isosceles : a = c) 
  (h_sides : a = 5 ∧ c = 5) (h_base : b = 4) :
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = (13125/1764) * π := by
sorry

end circle_area_through_isosceles_triangle_vertices_l463_46380


namespace remainder_thirteen_150_mod_11_l463_46354

theorem remainder_thirteen_150_mod_11 : 13^150 % 11 = 1 := by
  sorry

end remainder_thirteen_150_mod_11_l463_46354


namespace rectangle_formation_count_l463_46315

theorem rectangle_formation_count :
  let horizontal_lines := 5
  let vertical_lines := 4
  let horizontal_choices := Nat.choose horizontal_lines 2
  let vertical_choices := Nat.choose vertical_lines 2
  horizontal_choices * vertical_choices = 60 := by
  sorry

end rectangle_formation_count_l463_46315
