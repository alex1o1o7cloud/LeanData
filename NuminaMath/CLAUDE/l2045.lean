import Mathlib

namespace NUMINAMATH_CALUDE_smallest_divisor_of_1025_l2045_204563

theorem smallest_divisor_of_1025 : 
  ∀ n : ℕ, n > 1 → n ∣ 1025 → n ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_1025_l2045_204563


namespace NUMINAMATH_CALUDE_new_average_after_adding_l2045_204598

theorem new_average_after_adding (n : ℕ) (original_avg : ℚ) (add_value : ℚ) : 
  n > 0 → 
  let original_sum := n * original_avg
  let new_sum := original_sum + n * add_value
  let new_avg := new_sum / n
  n = 15 ∧ original_avg = 40 ∧ add_value = 10 → new_avg = 50 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_adding_l2045_204598


namespace NUMINAMATH_CALUDE_pinball_spend_proof_l2045_204560

def half_dollar : ℚ := 0.5

def wednesday_spend : ℕ := 4
def thursday_spend : ℕ := 14

def total_spend : ℚ := (wednesday_spend * half_dollar) + (thursday_spend * half_dollar)

theorem pinball_spend_proof : total_spend = 9 := by
  sorry

end NUMINAMATH_CALUDE_pinball_spend_proof_l2045_204560


namespace NUMINAMATH_CALUDE_order_of_special_values_l2045_204541

/-- Given a = √(1.01), b = e^(0.01) / 1.01, and c = ln(1.01e), prove that b < a < c. -/
theorem order_of_special_values :
  let a : ℝ := Real.sqrt 1.01
  let b : ℝ := Real.exp 0.01 / 1.01
  let c : ℝ := Real.log (1.01 * Real.exp 1)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_special_values_l2045_204541


namespace NUMINAMATH_CALUDE_a_is_integer_l2045_204594

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => ((2 * n + 3) * a (n + 1) + 3 * (n + 1) * a n) / (n + 2)

theorem a_is_integer (n : ℕ) : ∃ k : ℤ, a n = k := by
  sorry

end NUMINAMATH_CALUDE_a_is_integer_l2045_204594


namespace NUMINAMATH_CALUDE_smallest_divisible_number_is_correct_l2045_204507

/-- The smallest six-digit number exactly divisible by 25, 35, 45, and 15 -/
def smallest_divisible_number : ℕ := 100800

/-- Predicate to check if a number is six digits -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

theorem smallest_divisible_number_is_correct :
  is_six_digit smallest_divisible_number ∧
  smallest_divisible_number % 25 = 0 ∧
  smallest_divisible_number % 35 = 0 ∧
  smallest_divisible_number % 45 = 0 ∧
  smallest_divisible_number % 15 = 0 ∧
  ∀ n : ℕ, is_six_digit n →
    n % 25 = 0 → n % 35 = 0 → n % 45 = 0 → n % 15 = 0 →
    n ≥ smallest_divisible_number :=
by sorry

#eval smallest_divisible_number

end NUMINAMATH_CALUDE_smallest_divisible_number_is_correct_l2045_204507


namespace NUMINAMATH_CALUDE_javier_speech_time_l2045_204592

theorem javier_speech_time (outline_time writing_time practice_time total_time : ℕ) : 
  outline_time = 30 →
  writing_time = outline_time + 28 →
  practice_time = writing_time / 2 →
  total_time = outline_time + writing_time + practice_time →
  total_time = 117 :=
by sorry

end NUMINAMATH_CALUDE_javier_speech_time_l2045_204592


namespace NUMINAMATH_CALUDE_initial_coloring_books_count_l2045_204527

/-- Proves that the initial number of coloring books in stock is 40 --/
theorem initial_coloring_books_count (books_sold : ℕ) (books_per_shelf : ℕ) (shelves_used : ℕ) : 
  books_sold = 20 → books_per_shelf = 4 → shelves_used = 5 → 
  books_sold + books_per_shelf * shelves_used = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_coloring_books_count_l2045_204527


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2045_204593

theorem inequality_proof (a b n : ℕ) (h1 : a > b) (h2 : a * b - 1 = n^2) :
  a - b ≥ Real.sqrt (4 * n - 3) := by
  sorry

theorem equality_condition (a b n : ℕ) (h1 : a > b) (h2 : a * b - 1 = n^2) :
  (a - b = Real.sqrt (4 * n - 3)) ↔ 
  (∃ u : ℕ, a = u^2 + 2*u + 2 ∧ b = u^2 + 1 ∧ n = u^2 + u + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2045_204593


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2045_204524

theorem rational_equation_solution (C D : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 →
    (D * x - 17) / (x^2 - 9*x + 20) = C / (x - 4) + 5 / (x - 5)) →
  C + D = 19/5 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2045_204524


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2045_204517

theorem arithmetic_mean_problem (a₁ a₂ a₃ a₄ a₅ a₆ A : ℝ) 
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = A)
  (h2 : (a₁ + a₂ + a₃ + a₄) / 4 = A + 10)
  (h3 : (a₃ + a₄ + a₅ + a₆) / 4 = A - 7) :
  (a₁ + a₂ + a₅ + a₆) / 4 = A - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2045_204517


namespace NUMINAMATH_CALUDE_students_taking_neither_math_nor_chemistry_l2045_204537

theorem students_taking_neither_math_nor_chemistry :
  let total_students : ℕ := 150
  let math_students : ℕ := 80
  let chemistry_students : ℕ := 60
  let both_subjects : ℕ := 15
  let neither_subject : ℕ := total_students - (math_students + chemistry_students - both_subjects)
  neither_subject = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_neither_math_nor_chemistry_l2045_204537


namespace NUMINAMATH_CALUDE_functional_equation_polynomial_l2045_204562

/-- A polynomial that satisfies the functional equation P(X^2 + 1) = P(X)^2 + 1 and P(0) = 0 is equal to the identity function. -/
theorem functional_equation_polynomial (P : Polynomial ℝ) 
  (h1 : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1)
  (h2 : P.eval 0 = 0) : 
  P = Polynomial.X :=
sorry

end NUMINAMATH_CALUDE_functional_equation_polynomial_l2045_204562


namespace NUMINAMATH_CALUDE_geometric_sequence_extreme_points_l2045_204522

/-- Given a geometric sequence {a_n} where a_3 and a_7 are extreme points of f(x) = (1/3)x^3 + 4x^2 + 9x - 1, prove a_5 = -3 -/
theorem geometric_sequence_extreme_points (a : ℕ → ℝ) (h_geometric : ∀ n, a (n+1) / a n = a (n+2) / a (n+1)) :
  (∀ x, (x^2 + 8*x + 9) * (x - a 3) * (x - a 7) ≥ 0) →
  a 3 * a 7 = 9 →
  a 3 + a 7 = -8 →
  a 5 = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_extreme_points_l2045_204522


namespace NUMINAMATH_CALUDE_smallest_B_for_divisibility_l2045_204571

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def seven_digit_number (B : ℕ) : ℕ := 4000000 + B * 80000 + 83961

theorem smallest_B_for_divisibility :
  ∀ B : ℕ, B < 10 →
    (is_divisible_by_4 (seven_digit_number B) → B ≥ 0) ∧
    is_divisible_by_4 (seven_digit_number 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_B_for_divisibility_l2045_204571


namespace NUMINAMATH_CALUDE_varya_used_discount_l2045_204555

/-- Represents the quantity of items purchased by each girl -/
structure Purchase where
  pens : ℕ
  pencils : ℕ
  notebooks : ℕ

/-- Given the purchases of three girls and the fact that they all paid equally,
    prove that the second girl (Varya) must have used a discount -/
theorem varya_used_discount (p k l : ℚ) (anya varya sasha : Purchase) 
    (h_positive : p > 0 ∧ k > 0 ∧ l > 0)
    (h_anya : anya = ⟨2, 7, 1⟩)
    (h_varya : varya = ⟨5, 6, 5⟩)
    (h_sasha : sasha = ⟨8, 4, 9⟩)
    (h_equal_payment : ∃ (x : ℚ), 
      x = p * anya.pens + k * anya.pencils + l * anya.notebooks ∧
      x = p * varya.pens + k * varya.pencils + l * varya.notebooks ∧
      x = p * sasha.pens + k * sasha.pencils + l * sasha.notebooks) :
  p * varya.pens + k * varya.pencils + l * varya.notebooks < 
  (p * anya.pens + k * anya.pencils + l * anya.notebooks + 
   p * sasha.pens + k * sasha.pencils + l * sasha.notebooks) / 2 := by
  sorry


end NUMINAMATH_CALUDE_varya_used_discount_l2045_204555


namespace NUMINAMATH_CALUDE_average_of_numbers_is_one_l2045_204552

def numbers : List Int := [-5, -2, 0, 4, 8]

theorem average_of_numbers_is_one :
  (numbers.sum : ℚ) / numbers.length = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_is_one_l2045_204552


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l2045_204558

/-- The number of sections created by n line segments in a rectangle, 
    where each new line intersects all previous lines -/
def maxSections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else maxSections (n - 1) + n

/-- Theorem stating that 5 line segments can create at most 16 sections in a rectangle -/
theorem max_sections_five_lines :
  maxSections 5 = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l2045_204558


namespace NUMINAMATH_CALUDE_grass_field_path_problem_l2045_204574

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem grass_field_path_problem (field_length field_width path_width cost_per_unit : ℝ)
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 675 ∧
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1350 := by
  sorry

#check grass_field_path_problem

end NUMINAMATH_CALUDE_grass_field_path_problem_l2045_204574


namespace NUMINAMATH_CALUDE_candy_distribution_l2045_204599

theorem candy_distribution (marta_candies carmem_candies : ℕ) : 
  (marta_candies + carmem_candies = 200) →
  (marta_candies < 100) →
  (marta_candies > (4 * carmem_candies) / 5) →
  (∃ k : ℕ, marta_candies = 8 * k) →
  (∃ l : ℕ, carmem_candies = 8 * l) →
  (marta_candies = 96 ∧ carmem_candies = 104) := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2045_204599


namespace NUMINAMATH_CALUDE_snow_probability_both_days_l2045_204595

def prob_snow_monday : ℝ := 0.4
def prob_snow_tuesday : ℝ := 0.3

theorem snow_probability_both_days :
  let prob_both_days := prob_snow_monday * prob_snow_tuesday
  prob_both_days = 0.12 := by sorry

end NUMINAMATH_CALUDE_snow_probability_both_days_l2045_204595


namespace NUMINAMATH_CALUDE_exponential_growth_dominance_l2045_204523

theorem exponential_growth_dominance (n : ℕ) (h : n ≥ 10) : 2^n ≥ n^3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_growth_dominance_l2045_204523


namespace NUMINAMATH_CALUDE_intersection_angle_zero_curve_intersects_y_axis_at_zero_angle_l2045_204568

noncomputable def f (x : ℝ) := Real.exp x - x

theorem intersection_angle_zero : 
  let slope := (deriv f) 0
  slope = 0 := by sorry

-- The angle of intersection is the arctangent of the slope
theorem curve_intersects_y_axis_at_zero_angle : 
  Real.arctan ((deriv f) 0) = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_angle_zero_curve_intersects_y_axis_at_zero_angle_l2045_204568


namespace NUMINAMATH_CALUDE_compound_interest_principal_l2045_204540

/-- Given a sum of 8820 after 2 years with an interest rate of 5% per annum compounded yearly,
    prove that the initial principal amount was 8000. -/
theorem compound_interest_principal (sum : ℝ) (years : ℕ) (rate : ℝ) (principal : ℝ) 
    (h1 : sum = 8820)
    (h2 : years = 2)
    (h3 : rate = 0.05)
    (h4 : sum = principal * (1 + rate) ^ years) :
  principal = 8000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l2045_204540


namespace NUMINAMATH_CALUDE_comparison_proofs_l2045_204559

theorem comparison_proofs :
  (-5 < -2) ∧ (-1/3 > -1/2) ∧ (abs (-5) > 0) := by
  sorry

end NUMINAMATH_CALUDE_comparison_proofs_l2045_204559


namespace NUMINAMATH_CALUDE_dressing_p_vinegar_percent_l2045_204521

/-- Represents a salad dressing with a specific percentage of vinegar -/
structure SaladDressing where
  vinegar_percent : ℝ
  oil_percent : ℝ
  vinegar_oil_sum : vinegar_percent + oil_percent = 100

/-- The percentage of dressing P in the new mixture -/
def p_mixture_percent : ℝ := 10

/-- The percentage of dressing Q in the new mixture -/
def q_mixture_percent : ℝ := 100 - p_mixture_percent

/-- Dressing Q contains 10% vinegar -/
def dressing_q : SaladDressing := ⟨10, 90, by norm_num⟩

/-- The percentage of vinegar in the new mixture -/
def new_mixture_vinegar_percent : ℝ := 12

/-- Theorem stating that dressing P contains 30% vinegar -/
theorem dressing_p_vinegar_percent :
  ∃ (dressing_p : SaladDressing),
    dressing_p.vinegar_percent = 30 ∧
    (p_mixture_percent / 100 * dressing_p.vinegar_percent +
     q_mixture_percent / 100 * dressing_q.vinegar_percent = new_mixture_vinegar_percent) :=
by sorry

end NUMINAMATH_CALUDE_dressing_p_vinegar_percent_l2045_204521


namespace NUMINAMATH_CALUDE_frequency_in_range_l2045_204557

/-- Represents an interval with its frequency -/
structure IntervalData where
  lower : ℝ
  upper : ℝ
  frequency : ℕ

/-- Calculates the frequency of a sample within a given range -/
def calculateFrequency (data : List IntervalData) (range_start range_end : ℝ) (sample_size : ℕ) : ℝ :=
  sorry

/-- The given data set -/
def sampleData : List IntervalData := [
  ⟨10, 20, 2⟩,
  ⟨20, 30, 3⟩,
  ⟨30, 40, 4⟩,
  ⟨40, 50, 5⟩,
  ⟨50, 60, 4⟩,
  ⟨60, 70, 2⟩
]

theorem frequency_in_range : calculateFrequency sampleData 15 50 20 = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_frequency_in_range_l2045_204557


namespace NUMINAMATH_CALUDE_folded_rectangle_EF_length_l2045_204509

-- Define the rectangle
structure Rectangle :=
  (AB : ℝ)
  (BC : ℝ)

-- Define the folded pentagon
structure FoldedPentagon :=
  (rect : Rectangle)
  (EF : ℝ)

-- Theorem statement
theorem folded_rectangle_EF_length 
  (rect : Rectangle) 
  (pent : FoldedPentagon) : 
  rect.AB = 4 → 
  rect.BC = 8 → 
  pent.rect = rect → 
  pent.EF = 4 := by
sorry

end NUMINAMATH_CALUDE_folded_rectangle_EF_length_l2045_204509


namespace NUMINAMATH_CALUDE_complex_multiplication_l2045_204590

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 - 2*i) = 2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2045_204590


namespace NUMINAMATH_CALUDE_lily_sees_leo_l2045_204597

/-- The time Lily can see Leo given their speeds and distances -/
theorem lily_sees_leo (lily_speed leo_speed initial_distance final_distance : ℝ) : 
  lily_speed = 15 → 
  leo_speed = 9 → 
  initial_distance = 0.75 → 
  final_distance = 0.75 → 
  (initial_distance + final_distance) / (lily_speed - leo_speed) * 60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_lily_sees_leo_l2045_204597


namespace NUMINAMATH_CALUDE_outfit_combinations_l2045_204544

def num_shirts : ℕ := 8
def num_pants : ℕ := 5
def num_jacket_options : ℕ := 3

theorem outfit_combinations :
  num_shirts * num_pants * num_jacket_options = 120 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2045_204544


namespace NUMINAMATH_CALUDE_min_trips_for_given_weights_l2045_204545

-- Define the list of people's weights
def weights : List ℕ := [130, 60, 61, 65, 68, 70, 79, 81, 83, 87, 90, 91, 95]

-- Define the elevator capacity
def capacity : ℕ := 175

-- Function to calculate the minimum number of trips
def min_trips (weights : List ℕ) (capacity : ℕ) : ℕ := sorry

-- Theorem stating that the minimum number of trips is 7
theorem min_trips_for_given_weights :
  min_trips weights capacity = 7 := by sorry

end NUMINAMATH_CALUDE_min_trips_for_given_weights_l2045_204545


namespace NUMINAMATH_CALUDE_b_work_rate_l2045_204519

/-- Given work rates for individuals and groups, prove B's work rate -/
theorem b_work_rate 
  (a_rate : ℚ)
  (b_rate : ℚ)
  (c_rate : ℚ)
  (d_rate : ℚ)
  (h1 : a_rate = 1/4)
  (h2 : b_rate + c_rate = 1/2)
  (h3 : a_rate + c_rate = 1/2)
  (h4 : d_rate = 1/8)
  (h5 : a_rate + b_rate + d_rate = 1/(8/5)) :
  b_rate = 1/4 := by
sorry

end NUMINAMATH_CALUDE_b_work_rate_l2045_204519


namespace NUMINAMATH_CALUDE_line_BC_equation_l2045_204542

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem conditions
def problem_conditions (triangle : Triangle) (altitude1 altitude2 : Line) : Prop :=
  -- First altitude: x + y = 0
  altitude1.a = 1 ∧ altitude1.b = 1 ∧ altitude1.c = 0 ∧
  -- Second altitude: 2x - 3y + 1 = 0
  altitude2.a = 2 ∧ altitude2.b = -3 ∧ altitude2.c = 1 ∧
  -- Point A is (1, 2)
  triangle.A = (1, 2)

-- Theorem statement
theorem line_BC_equation (triangle : Triangle) (altitude1 altitude2 : Line) :
  problem_conditions triangle altitude1 altitude2 →
  ∃ (line_BC : Line), line_BC.a = 2 ∧ line_BC.b = 3 ∧ line_BC.c = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_line_BC_equation_l2045_204542


namespace NUMINAMATH_CALUDE_smallest_sum_for_equation_l2045_204582

theorem smallest_sum_for_equation (m n : ℕ+) (h : 3 * m ^ 3 = 5 * n ^ 5) :
  ∀ (x y : ℕ+), 3 * x ^ 3 = 5 * y ^ 5 → m + n ≤ x + y :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_for_equation_l2045_204582


namespace NUMINAMATH_CALUDE_weaving_problem_l2045_204591

def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem weaving_problem (a₁ d : ℕ) (h₁ : a₁ > 0) (h₂ : d > 0) :
  (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + 
   arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 4 = 24) →
  (arithmetic_sequence a₁ d 7 = arithmetic_sequence a₁ d 1 * arithmetic_sequence a₁ d 2) →
  arithmetic_sequence a₁ d 10 = 21 := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l2045_204591


namespace NUMINAMATH_CALUDE_sum_15_l2045_204573

/-- An arithmetic progression with sum of first n terms S_n -/
structure ArithmeticProgression where
  S : ℕ → ℝ  -- Sum function

/-- The sum of the first 5 terms is 3 -/
axiom sum_5 (ap : ArithmeticProgression) : ap.S 5 = 3

/-- The sum of the first 10 terms is 12 -/
axiom sum_10 (ap : ArithmeticProgression) : ap.S 10 = 12

/-- Theorem: If S_5 = 3 and S_10 = 12, then S_15 = 39 -/
theorem sum_15 (ap : ArithmeticProgression) : ap.S 15 = 39 := by
  sorry


end NUMINAMATH_CALUDE_sum_15_l2045_204573


namespace NUMINAMATH_CALUDE_stairs_in_building_correct_stairs_count_l2045_204550

theorem stairs_in_building (ned_speed : ℕ) (bomb_time_left : ℕ) (time_spent_running : ℕ) (diffuse_time : ℕ) : ℕ :=
  let total_run_time := time_spent_running + (bomb_time_left - diffuse_time)
  total_run_time / ned_speed

theorem correct_stairs_count : stairs_in_building 11 72 165 17 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stairs_in_building_correct_stairs_count_l2045_204550


namespace NUMINAMATH_CALUDE_long_sleeve_shirts_count_l2045_204512

theorem long_sleeve_shirts_count (total_shirts short_sleeve_shirts : ℕ) 
  (h1 : total_shirts = 9)
  (h2 : short_sleeve_shirts = 4) :
  total_shirts - short_sleeve_shirts = 5 := by
sorry

end NUMINAMATH_CALUDE_long_sleeve_shirts_count_l2045_204512


namespace NUMINAMATH_CALUDE_walter_fish_fry_guests_l2045_204546

-- Define the constants from the problem
def hushpuppies_per_guest : ℕ := 5
def hushpuppies_per_batch : ℕ := 10
def minutes_per_batch : ℕ := 8
def total_cooking_time : ℕ := 80

-- Define the function to calculate the number of guests
def number_of_guests : ℕ :=
  (total_cooking_time / minutes_per_batch * hushpuppies_per_batch) / hushpuppies_per_guest

-- State the theorem
theorem walter_fish_fry_guests :
  number_of_guests = 20 :=
sorry

end NUMINAMATH_CALUDE_walter_fish_fry_guests_l2045_204546


namespace NUMINAMATH_CALUDE_not_divisible_by_three_l2045_204528

theorem not_divisible_by_three (n : ℤ) : ¬(3 ∣ (n^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_three_l2045_204528


namespace NUMINAMATH_CALUDE_power_division_rule_l2045_204501

theorem power_division_rule (a : ℝ) : a^4 / a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l2045_204501


namespace NUMINAMATH_CALUDE_equation_solution_l2045_204589

theorem equation_solution : 
  ∀ x : ℂ, (5 * x^2 - 3 * x + 2) / (x + 2) = 2 * x - 4 ↔ 
  x = (3 + Complex.I * Real.sqrt 111) / 6 ∨ x = (3 - Complex.I * Real.sqrt 111) / 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2045_204589


namespace NUMINAMATH_CALUDE_smallest_n_perfect_powers_l2045_204535

theorem smallest_n_perfect_powers : ∃ (n : ℕ),
  (n = 1944) ∧
  (∃ (m : ℕ), 2 * n = m^4) ∧
  (∃ (l : ℕ), 3 * n = l^6) ∧
  (∀ (k : ℕ), k < n →
    (∃ (p : ℕ), 2 * k = p^4) →
    (∃ (q : ℕ), 3 * k = q^6) →
    False) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_powers_l2045_204535


namespace NUMINAMATH_CALUDE_jerry_action_figures_l2045_204539

/-- Calculates the total number of action figures on Jerry's shelf -/
def total_action_figures (initial_figures : ℕ) (figures_per_set : ℕ) (added_sets : ℕ) : ℕ :=
  initial_figures + figures_per_set * added_sets

/-- Theorem stating that Jerry's shelf has 18 action figures in total -/
theorem jerry_action_figures :
  total_action_figures 8 5 2 = 18 := by
sorry

end NUMINAMATH_CALUDE_jerry_action_figures_l2045_204539


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2045_204538

theorem rectangular_to_polar_conversion :
  let x : ℝ := 1
  let y : ℝ := -Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 5 * Real.pi / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 2 ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2045_204538


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2045_204581

def M : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def N : Set ℝ := {x | -2 < x ∧ x ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2045_204581


namespace NUMINAMATH_CALUDE_abs_g_zero_equals_70_l2045_204547

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- Condition that g is a third-degree polynomial with specific absolute values -/
def SatisfiesCondition (g : ThirdDegreePolynomial) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧
  (|g 1| = 10) ∧ (|g 3| = 10) ∧ (|g 4| = 10) ∧
  (|g 6| = 10) ∧ (|g 8| = 10) ∧ (|g 9| = 10)

/-- Theorem: If g satisfies the condition, then |g(0)| = 70 -/
theorem abs_g_zero_equals_70 (g : ThirdDegreePolynomial) 
  (h : SatisfiesCondition g) : |g 0| = 70 := by
  sorry

end NUMINAMATH_CALUDE_abs_g_zero_equals_70_l2045_204547


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2045_204596

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 > 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2045_204596


namespace NUMINAMATH_CALUDE_unique_peg_arrangement_l2045_204536

/-- Represents a color of a peg -/
inductive PegColor
  | Yellow
  | Red
  | Green
  | Blue
  | Orange

/-- Represents a position on the triangular peg board -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the triangular peg board -/
def Board := Position → Option PegColor

/-- Checks if a given board arrangement is valid -/
def is_valid_arrangement (board : Board) : Prop :=
  (∀ r c, board ⟨r, c⟩ = some PegColor.Yellow → r < 6 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Red → r < 5 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Green → r < 4 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Blue → r < 3 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Orange → r < 2 ∧ c < 6) ∧
  (∀ r, ∃! c, board ⟨r, c⟩ = some PegColor.Yellow) ∧
  (∀ r, r < 5 → ∃! c, board ⟨r, c⟩ = some PegColor.Red) ∧
  (∀ r, r < 4 → ∃! c, board ⟨r, c⟩ = some PegColor.Green) ∧
  (∀ r, r < 3 → ∃! c, board ⟨r, c⟩ = some PegColor.Blue) ∧
  (∀ r, r < 2 → ∃! c, board ⟨r, c⟩ = some PegColor.Orange) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Yellow) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Red) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Green) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Blue) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Orange)

theorem unique_peg_arrangement :
  ∃! board : Board, is_valid_arrangement board :=
sorry

end NUMINAMATH_CALUDE_unique_peg_arrangement_l2045_204536


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_l2045_204526

/-- Tetrahedron with specific properties -/
structure Tetrahedron where
  -- Base is a square with side length 3
  baseSideLength : ℝ := 3
  -- PD length is 4
  pdLength : ℝ := 4
  -- Lateral faces PAD and PCD are perpendicular to the base
  lateralFacesPerpendicular : Prop

/-- Calculate the surface area of the tetrahedron -/
def surfaceArea (t : Tetrahedron) : ℝ :=
  -- We don't implement the actual calculation here
  sorry

/-- Theorem stating the surface area of the tetrahedron -/
theorem tetrahedron_surface_area (t : Tetrahedron) : 
  surfaceArea t = 9 + 6 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_surface_area_l2045_204526


namespace NUMINAMATH_CALUDE_f_nonnegative_implies_a_bound_f_inequality_l2045_204566

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a

theorem f_nonnegative_implies_a_bound (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) → a ≥ 1 / Real.exp 1 := by sorry

theorem f_inequality (a : ℝ) (x₁ x₂ x : ℝ) 
  (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h : x₁ < x ∧ x < x₂) :
  (f a x - f a x₁) / (x - x₁) < (f a x - f a x₂) / (x - x₂) := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_implies_a_bound_f_inequality_l2045_204566


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2045_204510

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2045_204510


namespace NUMINAMATH_CALUDE_adam_has_more_apples_l2045_204525

/-- The number of apples Adam has -/
def adam_apples : ℕ := 10

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 2

/-- The difference in apples between Adam and Jackie -/
def apple_difference : ℕ := adam_apples - jackie_apples

theorem adam_has_more_apples : apple_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_adam_has_more_apples_l2045_204525


namespace NUMINAMATH_CALUDE_inscribed_polygon_has_larger_area_l2045_204543

/-- A polygon is a set of points in the plane --/
def Polygon : Type := Set (ℝ × ℝ)

/-- A convex polygon is a polygon where all interior angles are less than or equal to 180 degrees --/
def ConvexPolygon (P : Polygon) : Prop := sorry

/-- A polygon is inscribed in a circle if all its vertices lie on the circle's circumference --/
def InscribedInCircle (P : Polygon) : Prop := sorry

/-- The area of a polygon --/
def PolygonArea (P : Polygon) : ℝ := sorry

/-- The side lengths of a polygon --/
def SideLengths (P : Polygon) : List ℝ := sorry

/-- Two polygons have the same side lengths --/
def SameSideLengths (P Q : Polygon) : Prop :=
  SideLengths P = SideLengths Q

theorem inscribed_polygon_has_larger_area 
  (N M : Polygon) 
  (h1 : ConvexPolygon N) 
  (h2 : ConvexPolygon M) 
  (h3 : InscribedInCircle N) 
  (h4 : SameSideLengths N M) :
  PolygonArea N > PolygonArea M :=
sorry

end NUMINAMATH_CALUDE_inscribed_polygon_has_larger_area_l2045_204543


namespace NUMINAMATH_CALUDE_symmetric_angle_ratio_l2045_204556

/-- 
Given a point P(x,y) on the terminal side of an angle θ (excluding the origin), 
where the terminal side of θ is symmetric to the terminal side of a 480° angle 
with respect to the x-axis, prove that xy/(x^2 + y^2) = √3/4.
-/
theorem symmetric_angle_ratio (x y : ℝ) (h1 : x ≠ 0 ∨ y ≠ 0) 
  (h2 : y = Real.sqrt 3 * x) : 
  (x * y) / (x^2 + y^2) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_angle_ratio_l2045_204556


namespace NUMINAMATH_CALUDE_flow_rate_difference_l2045_204506

/-- Proves that the difference between 0.6 times the original flow rate and the reduced flow rate is 1 gallon per minute -/
theorem flow_rate_difference (original_rate reduced_rate : ℝ) 
  (h1 : original_rate = 5.0)
  (h2 : reduced_rate = 2) : 
  0.6 * original_rate - reduced_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_flow_rate_difference_l2045_204506


namespace NUMINAMATH_CALUDE_fibonacci_identity_l2045_204518

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Statement of the problem -/
theorem fibonacci_identity (k : ℤ) :
  (fib 785 + k) * (fib 787 + k) - (fib 786 + k)^2 = -1 := by
  sorry


end NUMINAMATH_CALUDE_fibonacci_identity_l2045_204518


namespace NUMINAMATH_CALUDE_largest_n_perfect_cube_l2045_204520

theorem largest_n_perfect_cube (n : ℕ) : n = 497 ↔ 
  (n < 500 ∧ 
   ∃ m : ℕ, 6048 * 28^n = m^3 ∧ 
   ∀ k : ℕ, k < 500 ∧ k > n → ¬∃ l : ℕ, 6048 * 28^k = l^3) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_perfect_cube_l2045_204520


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2045_204572

theorem cube_root_equation_solution (x : ℝ) :
  (5 + 2 / x) ^ (1/3 : ℝ) = -3 → x = -(1/16) := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2045_204572


namespace NUMINAMATH_CALUDE_lakota_new_cd_count_l2045_204531

/-- The price of a used CD in dollars -/
def used_cd_price : ℚ := 9.99

/-- The total price of Lakota's purchase in dollars -/
def lakota_total : ℚ := 127.92

/-- The total price of Mackenzie's purchase in dollars -/
def mackenzie_total : ℚ := 133.89

/-- The number of used CDs Lakota bought -/
def lakota_used : ℕ := 2

/-- The number of new CDs Mackenzie bought -/
def mackenzie_new : ℕ := 3

/-- The number of used CDs Mackenzie bought -/
def mackenzie_used : ℕ := 8

/-- The number of new CDs Lakota bought -/
def lakota_new : ℕ := 6

theorem lakota_new_cd_count :
  ∃ (new_cd_price : ℚ),
    new_cd_price * lakota_new + used_cd_price * lakota_used = lakota_total ∧
    new_cd_price * mackenzie_new + used_cd_price * mackenzie_used = mackenzie_total :=
by sorry

end NUMINAMATH_CALUDE_lakota_new_cd_count_l2045_204531


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2045_204584

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_mersenne_prime (n : ℕ) : Prop := ∃ p : ℕ, is_prime p ∧ n = 2^p - 1

theorem largest_mersenne_prime_under_500 : 
  (∀ m : ℕ, is_mersenne_prime m → m < 500 → m ≤ 127) ∧ 
  is_mersenne_prime 127 ∧ 
  127 < 500 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2045_204584


namespace NUMINAMATH_CALUDE_gcf_of_54_and_72_l2045_204577

theorem gcf_of_54_and_72 : Nat.gcd 54 72 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_54_and_72_l2045_204577


namespace NUMINAMATH_CALUDE_equal_to_one_half_l2045_204534

theorem equal_to_one_half : 
  Real.sqrt ((1 + Real.cos (2 * Real.pi / 3)) / 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_to_one_half_l2045_204534


namespace NUMINAMATH_CALUDE_cos_sin_three_pi_eighths_l2045_204516

theorem cos_sin_three_pi_eighths : 
  Real.cos (3 * π / 8) ^ 2 - Real.sin (3 * π / 8) ^ 2 = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_three_pi_eighths_l2045_204516


namespace NUMINAMATH_CALUDE_new_york_squares_count_l2045_204529

/-- The number of squares in New York City -/
def num_squares : ℕ := 15

/-- The total number of streetlights bought by the city council -/
def total_streetlights : ℕ := 200

/-- The number of streetlights required for each square -/
def streetlights_per_square : ℕ := 12

/-- The number of unused streetlights -/
def unused_streetlights : ℕ := 20

/-- Theorem stating that the number of squares in New York City is correct -/
theorem new_york_squares_count :
  num_squares * streetlights_per_square + unused_streetlights = total_streetlights :=
by sorry

end NUMINAMATH_CALUDE_new_york_squares_count_l2045_204529


namespace NUMINAMATH_CALUDE_library_configuration_count_l2045_204505

/-- The number of different configurations for 8 identical books in a library,
    where at least one book must remain in the library and at least one must be checked out. -/
def library_configurations : ℕ := 7

/-- The total number of books in the library -/
def total_books : ℕ := 8

/-- Proposition that there are exactly 7 different configurations for the books in the library -/
theorem library_configuration_count :
  (∀ config : ℕ, 1 ≤ config ∧ config ≤ total_books - 1) →
  (∀ config : ℕ, config ≤ total_books - config) →
  library_configurations = (total_books - 1) := by
  sorry

end NUMINAMATH_CALUDE_library_configuration_count_l2045_204505


namespace NUMINAMATH_CALUDE_expand_product_l2045_204561

theorem expand_product (x : ℝ) : (x + 3) * (x + 4 + 6) = x^2 + 13*x + 30 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2045_204561


namespace NUMINAMATH_CALUDE_visible_part_of_third_mountain_l2045_204511

/-- Represents a mountain with a height and position on a great circle. -/
structure Mountain where
  height : ℝ
  position : ℝ

/-- Represents the Earth as a sphere. -/
structure Earth where
  radius : ℝ

/-- Calculates the visible height of a distant mountain. -/
def visibleHeight (earth : Earth) (m1 m2 m3 : Mountain) : ℝ :=
  sorry

theorem visible_part_of_third_mountain
  (earth : Earth)
  (m1 m2 m3 : Mountain)
  (h_earth_radius : earth.radius = 6366000) -- in meters
  (h_m1_height : m1.height = 2500)
  (h_m2_height : m2.height = 3000)
  (h_m3_height : m3.height = 8800)
  (h_m1_m2_distance : m2.position - m1.position = 1 * π / 180) -- 1 degree in radians
  (h_m2_m3_distance : m3.position - m2.position = 1.5 * π / 180) -- 1.5 degrees in radians
  : visibleHeight earth m1 m2 m3 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_visible_part_of_third_mountain_l2045_204511


namespace NUMINAMATH_CALUDE_square_area_expansion_l2045_204579

theorem square_area_expansion (a : ℝ) (h : a > 0) :
  (3 * a)^2 = 9 * a^2 := by sorry

end NUMINAMATH_CALUDE_square_area_expansion_l2045_204579


namespace NUMINAMATH_CALUDE_expression_equals_sqrt_two_l2045_204502

theorem expression_equals_sqrt_two : (-1)^2 + |-Real.sqrt 2| + (Real.pi - 3)^0 - Real.sqrt 4 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_sqrt_two_l2045_204502


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l2045_204570

/-- The shortest length of the tangent from a point on the line x - y + 2√2 = 0 to the circle x² + y² = 1 is √3 -/
theorem shortest_tangent_length (x y : ℝ) : 
  (x - y + 2 * Real.sqrt 2 = 0) →
  (x^2 + y^2 = 1) →
  ∃ (px py : ℝ), 
    (px - py + 2 * Real.sqrt 2 = 0) ∧
    Real.sqrt ((px - x)^2 + (py - y)^2) ≥ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l2045_204570


namespace NUMINAMATH_CALUDE_elberta_money_l2045_204554

theorem elberta_money (granny_smith : ℕ) (anjou : ℕ) (elberta : ℕ) : 
  granny_smith = 64 →
  anjou = granny_smith / 4 →
  elberta = anjou + 3 →
  elberta = 19 := by
  sorry

end NUMINAMATH_CALUDE_elberta_money_l2045_204554


namespace NUMINAMATH_CALUDE_brian_video_time_l2045_204504

/-- The duration of Brian's animal video watching session -/
def total_video_time (cat_video_duration : ℕ) : ℕ :=
  let dog_video_duration := 2 * cat_video_duration
  let first_two_videos_duration := cat_video_duration + dog_video_duration
  let gorilla_video_duration := 2 * first_two_videos_duration
  cat_video_duration + dog_video_duration + gorilla_video_duration

/-- Theorem stating that Brian spends 36 minutes watching animal videos -/
theorem brian_video_time : total_video_time 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_brian_video_time_l2045_204504


namespace NUMINAMATH_CALUDE_min_value_of_f_l2045_204553

-- Define the function
def f (x : ℝ) : ℝ := x^4 - 4*x + 3

-- Define the interval
def interval : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 0 ∧ ∀ y ∈ interval, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2045_204553


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_999_l2045_204530

theorem largest_prime_factor_of_999 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 999 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 999 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_999_l2045_204530


namespace NUMINAMATH_CALUDE_factor_x_sixth_plus_64_l2045_204508

theorem factor_x_sixth_plus_64 (x : ℝ) : x^6 + 64 = (x^2 + 4) * (x^4 - 4*x^2 + 16) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_sixth_plus_64_l2045_204508


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l2045_204549

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 8 ∧ 
  (∀ y : ℝ, y > 0 ∧ (⌊y^2⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 8 → x ≤ y) ∧
  x = 89 / 9 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l2045_204549


namespace NUMINAMATH_CALUDE_booklet_sheets_theorem_l2045_204500

/-- Given a stack of sheets folded into a booklet, this function calculates
    the number of sheets in the original stack based on the sum of page numbers on one sheet. -/
def calculate_original_sheets (sum_of_page_numbers : ℕ) : ℕ :=
  (sum_of_page_numbers - 2) / 4

/-- Theorem stating that if the sum of page numbers on one sheet is 74,
    then the original stack contained 9 sheets. -/
theorem booklet_sheets_theorem (sum_is_74 : calculate_original_sheets 74 = 9) :
  calculate_original_sheets 74 = 9 := by
  sorry

#eval calculate_original_sheets 74  -- Should output 9

end NUMINAMATH_CALUDE_booklet_sheets_theorem_l2045_204500


namespace NUMINAMATH_CALUDE_largest_expression_l2045_204586

theorem largest_expression : ∀ (a b c d e : ℕ),
  a = 3 + 1 + 2 + 8 →
  b = 3 * 1 + 2 + 8 →
  c = 3 + 1 * 2 + 8 →
  d = 3 + 1 + 2 * 8 →
  e = 3 * 1 * 2 * 8 →
  e ≥ a ∧ e ≥ b ∧ e ≥ c ∧ e ≥ d :=
by sorry

end NUMINAMATH_CALUDE_largest_expression_l2045_204586


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2045_204588

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 ≤ 0}
def B : Set ℝ := {x | x > 7}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = Set.Ioi 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2045_204588


namespace NUMINAMATH_CALUDE_cube_sum_eq_343_l2045_204515

theorem cube_sum_eq_343 (x y : ℝ) :
  x^3 + 21*x*y + y^3 = 343 → x + y = 7 ∨ x + y = -14 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_eq_343_l2045_204515


namespace NUMINAMATH_CALUDE_min_value_zero_at_k_eq_two_l2045_204551

/-- The quadratic function f(x, y) depending on parameter k -/
def f (k : ℝ) (x y : ℝ) : ℝ :=
  4 * x^2 - 6 * k * x * y + (3 * k^2 + 2) * y^2 - 4 * x - 4 * y + 6

/-- Theorem stating that k = 2 is the unique value for which the minimum of f is 0 -/
theorem min_value_zero_at_k_eq_two :
  ∃! k : ℝ, (∀ x y : ℝ, f k x y ≥ 0) ∧ (∃ x y : ℝ, f k x y = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_zero_at_k_eq_two_l2045_204551


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l2045_204514

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := 
  λ x => if x ≤ 1 then (4 - a) * x else a^x

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → 
  (a ∈ Set.Icc 2 4 ∧ a ≠ 4) := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l2045_204514


namespace NUMINAMATH_CALUDE_string_measurement_l2045_204583

theorem string_measurement (string_length : Real) (cut_fraction : Real) : 
  string_length = 2/3 → 
  cut_fraction = 1/4 → 
  (1 - cut_fraction) * string_length = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_string_measurement_l2045_204583


namespace NUMINAMATH_CALUDE_dark_king_game_winner_l2045_204503

/-- The dark king game on an n × m chessboard -/
def DarkKingGame (n m : ℕ) :=
  { board : Set (ℕ × ℕ) // board ⊆ (Finset.range n).product (Finset.range m) }

/-- A player in the dark king game -/
inductive Player
| First
| Second

/-- A winning strategy for a player in the dark king game -/
def WinningStrategy (n m : ℕ) (p : Player) :=
  ∃ (strategy : DarkKingGame n m → ℕ × ℕ),
    ∀ (game : DarkKingGame n m),
      (strategy game ∉ game.val) →
      (strategy game).1 < n ∧ (strategy game).2 < m

/-- The main theorem about the dark king game -/
theorem dark_king_game_winner (n m : ℕ) :
  (n % 2 = 0 ∨ m % 2 = 0) → WinningStrategy n m Player.First ∧
  (n % 2 = 1 ∧ m % 2 = 1) → WinningStrategy n m Player.Second :=
sorry

end NUMINAMATH_CALUDE_dark_king_game_winner_l2045_204503


namespace NUMINAMATH_CALUDE_odd_power_divisibility_l2045_204585

theorem odd_power_divisibility (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ∀ n : ℕ, ∃ m : ℕ, (2^n ∣ a^m * b^2 - 1) ∨ (2^n ∣ b^m * a^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_odd_power_divisibility_l2045_204585


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l2045_204532

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l2045_204532


namespace NUMINAMATH_CALUDE_regression_unit_increase_food_expenditure_increase_l2045_204564

/-- Represents a linear regression equation ŷ = ax + b -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- Calculates the predicted value for a given x -/
def LinearRegression.predict (reg : LinearRegression) (x : ℝ) : ℝ :=
  reg.a * x + reg.b

/-- The increase in ŷ when x increases by 1 is equal to the coefficient a -/
theorem regression_unit_increase (reg : LinearRegression) :
  reg.predict (x + 1) - reg.predict x = reg.a :=
by sorry

/-- The specific regression equation from the problem -/
def food_expenditure_regression : LinearRegression :=
  { a := 0.254, b := 0.321 }

/-- The increase in food expenditure when income increases by 1 is 0.254 -/
theorem food_expenditure_increase :
  food_expenditure_regression.predict (x + 1) - food_expenditure_regression.predict x = 0.254 :=
by sorry

end NUMINAMATH_CALUDE_regression_unit_increase_food_expenditure_increase_l2045_204564


namespace NUMINAMATH_CALUDE_probability_all_different_digits_l2045_204575

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_all_different_digits (n : ℕ) : Prop :=
  is_three_digit_number n ∧
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.toFinset.card = 3

def count_three_digit_numbers : ℕ := 999 - 100 + 1

def count_numbers_with_all_different_digits : ℕ := 675

theorem probability_all_different_digits :
  (count_numbers_with_all_different_digits : ℚ) / count_three_digit_numbers = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_different_digits_l2045_204575


namespace NUMINAMATH_CALUDE_encryption_3859_l2045_204587

def encrypt_digit (d : Nat) : Nat :=
  (d^3 + 1) % 10

def encrypt_number (n : List Nat) : List Nat :=
  n.map encrypt_digit

theorem encryption_3859 :
  encrypt_number [3, 8, 5, 9] = [8, 3, 6, 0] := by
  sorry

end NUMINAMATH_CALUDE_encryption_3859_l2045_204587


namespace NUMINAMATH_CALUDE_triangular_pyramid_volume_l2045_204580

/-- Given a triangular pyramid with mutually perpendicular lateral faces of areas 6, 4, and 3, 
    its volume is 4. -/
theorem triangular_pyramid_volume (a b c : ℝ) 
  (h1 : a * b / 2 = 6) 
  (h2 : a * c / 2 = 4) 
  (h3 : b * c / 2 = 3) : 
  a * b * c / 6 = 4 := by
  sorry

#check triangular_pyramid_volume

end NUMINAMATH_CALUDE_triangular_pyramid_volume_l2045_204580


namespace NUMINAMATH_CALUDE_basketball_score_proof_l2045_204578

theorem basketball_score_proof (two_points : ℕ) (three_points : ℕ) (free_throws : ℕ) :
  (3 * three_points = 2 * (2 * two_points)) →
  (free_throws = 2 * two_points) →
  (2 * two_points + 3 * three_points + free_throws = 72) →
  free_throws = 18 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l2045_204578


namespace NUMINAMATH_CALUDE_parabola_horizontal_shift_l2045_204533

/-- A parabola is defined by its coefficient and vertex -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in vertex form is y = a(x-h)^2 + k -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

theorem parabola_horizontal_shift 
  (p1 p2 : Parabola) 
  (h1 : p1.a = p2.a) 
  (h2 : p1.k = p2.k) 
  (h3 : p1.h = p2.h + 3) : 
  ∀ x, parabola_equation p1 x = parabola_equation p2 (x - 3) :=
sorry

end NUMINAMATH_CALUDE_parabola_horizontal_shift_l2045_204533


namespace NUMINAMATH_CALUDE_remainder_proof_l2045_204565

theorem remainder_proof : (7 * 10^20 + 1^20) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l2045_204565


namespace NUMINAMATH_CALUDE_sugar_per_cup_l2045_204513

def total_sugar : ℝ := 84.6
def num_cups : ℕ := 12

theorem sugar_per_cup : 
  (total_sugar / num_cups : ℝ) = 7.05 := by sorry

end NUMINAMATH_CALUDE_sugar_per_cup_l2045_204513


namespace NUMINAMATH_CALUDE_log_comparison_l2045_204576

theorem log_comparison : Real.log 2009 / Real.log 2008 > Real.log 2010 / Real.log 2009 := by sorry

end NUMINAMATH_CALUDE_log_comparison_l2045_204576


namespace NUMINAMATH_CALUDE_recipe_calculation_l2045_204548

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℚ
  flour : ℚ
  sugar : ℚ

/-- Calculates the required amount of an ingredient based on the ratio and the amount of sugar used -/
def calculateAmount (ratio : RecipeRatio) (sugarAmount : ℚ) (partRatio : ℚ) : ℚ :=
  (sugarAmount / ratio.sugar) * partRatio

/-- Proves that given a recipe with a butter:flour:sugar ratio of 1:6:4 and using 10 cups of sugar,
    the required amounts of butter and flour are 2.5 cups and 15 cups, respectively -/
theorem recipe_calculation (ratio : RecipeRatio) (sugarAmount : ℚ) :
  ratio.butter = 1 → ratio.flour = 6 → ratio.sugar = 4 → sugarAmount = 10 →
  calculateAmount ratio sugarAmount ratio.butter = 5/2 ∧
  calculateAmount ratio sugarAmount ratio.flour = 15 :=
by sorry

end NUMINAMATH_CALUDE_recipe_calculation_l2045_204548


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l2045_204567

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem stating the number of arrangements for 6 books with 3 identical -/
theorem book_arrangement_theorem :
  arrange_books 6 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l2045_204567


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2045_204569

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (5 + 2 * z) = 11 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2045_204569
