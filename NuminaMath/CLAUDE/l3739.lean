import Mathlib

namespace NUMINAMATH_CALUDE_double_average_l3739_373960

theorem double_average (n : ℕ) (original_avg : ℝ) (h1 : n = 10) (h2 : original_avg = 80) :
  let total_marks := n * original_avg
  let new_total_marks := 2 * total_marks
  let new_avg := new_total_marks / n
  new_avg = 160 := by
sorry

end NUMINAMATH_CALUDE_double_average_l3739_373960


namespace NUMINAMATH_CALUDE_calculate_gratuity_percentage_l3739_373921

/-- Calculate the gratuity percentage for a restaurant bill -/
theorem calculate_gratuity_percentage
  (num_people : ℕ)
  (total_bill : ℚ)
  (avg_cost_before_gratuity : ℚ)
  (h_num_people : num_people = 9)
  (h_total_bill : total_bill = 756)
  (h_avg_cost : avg_cost_before_gratuity = 70) :
  (total_bill - num_people * avg_cost_before_gratuity) / (num_people * avg_cost_before_gratuity) = 1/5 :=
sorry

end NUMINAMATH_CALUDE_calculate_gratuity_percentage_l3739_373921


namespace NUMINAMATH_CALUDE_least_common_denominator_l3739_373905

theorem least_common_denominator : 
  let denominators := [5, 6, 8, 9, 10, 11]
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9) 10) 11 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l3739_373905


namespace NUMINAMATH_CALUDE_largest_base5_to_base10_l3739_373909

/-- Converts a base-5 number to base-10 --/
def base5To10 (d2 d1 d0 : Nat) : Nat :=
  d2 * 5^2 + d1 * 5^1 + d0 * 5^0

/-- The largest three-digit base-5 number --/
def largestBase5 : Nat := base5To10 4 4 4

theorem largest_base5_to_base10 :
  largestBase5 = 124 := by sorry

end NUMINAMATH_CALUDE_largest_base5_to_base10_l3739_373909


namespace NUMINAMATH_CALUDE_largest_special_number_l3739_373910

/-- A number is a two-digit number if it's between 10 and 99 inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A number ends in 4 if it leaves a remainder of 4 when divided by 10 -/
def ends_in_four (n : ℕ) : Prop := n % 10 = 4

/-- The set of two-digit numbers divisible by 6 and ending in 4 -/
def special_set : Set ℕ := {n | is_two_digit n ∧ n % 6 = 0 ∧ ends_in_four n}

theorem largest_special_number : 
  ∃ (m : ℕ), m ∈ special_set ∧ ∀ (n : ℕ), n ∈ special_set → n ≤ m ∧ m = 84 :=
sorry

end NUMINAMATH_CALUDE_largest_special_number_l3739_373910


namespace NUMINAMATH_CALUDE_long_jump_difference_l3739_373903

/-- Represents the long jump event results for Ricciana and Margarita -/
structure LongJumpEvent where
  ricciana_total : ℕ
  ricciana_run : ℕ
  ricciana_jump : ℕ
  margarita_run : ℕ
  h_ricciana_total : ricciana_total = ricciana_run + ricciana_jump
  h_margarita_jump : ℕ

/-- The difference in total distance between Margarita and Ricciana is 1 foot -/
theorem long_jump_difference (event : LongJumpEvent)
  (h_ricciana_total : event.ricciana_total = 24)
  (h_ricciana_run : event.ricciana_run = 20)
  (h_ricciana_jump : event.ricciana_jump = 4)
  (h_margarita_run : event.margarita_run = 18)
  (h_margarita_jump : event.h_margarita_jump = 2 * event.ricciana_jump - 1) :
  event.margarita_run + event.h_margarita_jump - event.ricciana_total = 1 := by
  sorry

end NUMINAMATH_CALUDE_long_jump_difference_l3739_373903


namespace NUMINAMATH_CALUDE_range_of_a_l3739_373985

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def g (x : ℝ) : ℝ := -x^2 + 2*x + 1

theorem range_of_a (a : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 (Real.exp 1), ∃ x₂ ∈ Set.Icc 0 3, f a x₁ = g x₂) →
  a ∈ Set.Icc (-1 / Real.exp 1) (3 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3739_373985


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_of_squares_l3739_373950

theorem arithmetic_geometric_mean_sum_of_squares 
  (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = 15) : 
  x^2 + y^2 = 1150 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_of_squares_l3739_373950


namespace NUMINAMATH_CALUDE_unique_number_with_triple_property_l3739_373902

/-- A six-digit number with 1 as its leftmost digit -/
def sixDigitNumberStartingWith1 (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 1

/-- Function to move the leftmost digit to the rightmost position -/
def moveFirstDigitToEnd (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

/-- The main theorem statement -/
theorem unique_number_with_triple_property :
  ∃! n : ℕ, sixDigitNumberStartingWith1 n ∧ moveFirstDigitToEnd n = 3 * n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_triple_property_l3739_373902


namespace NUMINAMATH_CALUDE_students_without_A_l3739_373947

/-- Given a class of students and their grades in three subjects, calculate the number of students who didn't receive an A in any subject. -/
theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (science : ℕ) 
  (history_math : ℕ) (history_science : ℕ) (math_science : ℕ) (all_three : ℕ) : 
  total = 40 →
  history = 10 →
  math = 15 →
  science = 8 →
  history_math = 5 →
  history_science = 3 →
  math_science = 4 →
  all_three = 2 →
  total - (history + math + science - history_math - history_science - math_science + all_three) = 17 := by
sorry

end NUMINAMATH_CALUDE_students_without_A_l3739_373947


namespace NUMINAMATH_CALUDE_max_value_on_interval_max_value_at_one_l3739_373924

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

-- Define the property of f being monotonically decreasing on (-∞, 2]
def is_monotone_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 2 → f a x ≥ f a y

-- Theorem 1: Maximum value of f(x) on [3, 5] is 2
theorem max_value_on_interval (a : ℝ) 
  (h : is_monotone_decreasing_on_interval a) : 
  (∀ x, x ∈ Set.Icc 3 5 → f a x ≤ 2) ∧ (∃ x, x ∈ Set.Icc 3 5 ∧ f a x = 2) :=
sorry

-- Theorem 2: Maximum value of f(1) is -6
theorem max_value_at_one (a : ℝ) 
  (h : is_monotone_decreasing_on_interval a) : 
  f a 1 ≤ -6 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_max_value_at_one_l3739_373924


namespace NUMINAMATH_CALUDE_expression_evaluation_l3739_373948

theorem expression_evaluation (x y : ℕ) (h1 : x = 2) (h2 : y = 3) : 3 * x^y + 4 * y^x = 60 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3739_373948


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3739_373945

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a4 : a 4 = 8) :
  a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3739_373945


namespace NUMINAMATH_CALUDE_green_ball_count_l3739_373992

/-- Given a box of balls where the ratio of blue to green balls is 5:3 and there are 15 blue balls,
    prove that the number of green balls is 9. -/
theorem green_ball_count (blue : ℕ) (green : ℕ) (h1 : blue = 15) (h2 : blue * 3 = green * 5) : green = 9 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_count_l3739_373992


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3739_373914

def P : Set ℝ := {x | x^2 - 9 < 0}
def Q : Set ℝ := {y | ∃ x : ℤ, y = 2 * x}

theorem intersection_of_P_and_Q : P ∩ Q = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3739_373914


namespace NUMINAMATH_CALUDE_power_problem_l3739_373906

theorem power_problem (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3*a + b) = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_problem_l3739_373906


namespace NUMINAMATH_CALUDE_free_fall_time_l3739_373936

-- Define the relationship between height and time
def height_time_relation (t : ℝ) : ℝ := 4.9 * t^2

-- Define the initial height
def initial_height : ℝ := 490

-- Theorem statement
theorem free_fall_time : 
  ∃ (t : ℝ), t > 0 ∧ height_time_relation t = initial_height ∧ t = 10 := by
  sorry

end NUMINAMATH_CALUDE_free_fall_time_l3739_373936


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l3739_373922

theorem shaded_fraction_of_rectangle (rectangle_length rectangle_width : ℝ) 
  (h_length : rectangle_length = 15)
  (h_width : rectangle_width = 20)
  (h_triangle_area : ∃ (triangle_area : ℝ), triangle_area = (1/3) * rectangle_length * rectangle_width)
  (h_shaded_area : ∃ (shaded_area : ℝ), shaded_area = (1/2) * (1/3) * rectangle_length * rectangle_width) :
  (∃ (shaded_area : ℝ), shaded_area = (1/6) * rectangle_length * rectangle_width) :=
by sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l3739_373922


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l3739_373901

theorem fly_distance_from_ceiling (z : ℝ) : 
  3^2 + 4^2 + z^2 = 6^2 → z = Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l3739_373901


namespace NUMINAMATH_CALUDE_cos_neg_pi_third_l3739_373952

theorem cos_neg_pi_third : Real.cos (-π/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_pi_third_l3739_373952


namespace NUMINAMATH_CALUDE_overweight_condition_equiv_l3739_373990

/-- Ideal weight formula -/
def ideal_weight (h : ℝ) : ℝ := 22 * h^2

/-- Overweight threshold -/
def overweight_threshold (h : ℝ) : ℝ := 1.1 * ideal_weight h

/-- Overweight condition -/
def is_overweight (W h : ℝ) : Prop := W > overweight_threshold h

/-- Quadratic overweight condition -/
def quadratic_overweight (c d e : ℝ) (W h : ℝ) : Prop := W > c * h^2 + d * h + e

theorem overweight_condition_equiv :
  ∃ c d e : ℝ, ∀ W h : ℝ, is_overweight W h ↔ quadratic_overweight c d e W h :=
sorry

end NUMINAMATH_CALUDE_overweight_condition_equiv_l3739_373990


namespace NUMINAMATH_CALUDE_basketball_not_tabletennis_l3739_373904

theorem basketball_not_tabletennis (U A B : Finset ℕ) : 
  Finset.card U = 42 →
  Finset.card A = 20 →
  Finset.card B = 25 →
  Finset.card (U \ (A ∪ B)) = 12 →
  Finset.card (A \ B) = 5 := by
sorry

end NUMINAMATH_CALUDE_basketball_not_tabletennis_l3739_373904


namespace NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_25_l3739_373933

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def smallestPrimeBetween1And25 : ℕ := 2

def largestPrimeBetween1And25 : ℕ := 23

theorem sum_smallest_largest_prime_1_to_25 :
  isPrime smallestPrimeBetween1And25 ∧
  isPrime largestPrimeBetween1And25 ∧
  (∀ n : ℕ, 1 < n → n < 25 → isPrime n → smallestPrimeBetween1And25 ≤ n) ∧
  (∀ n : ℕ, 1 < n → n < 25 → isPrime n → n ≤ largestPrimeBetween1And25) →
  smallestPrimeBetween1And25 + largestPrimeBetween1And25 = 25 :=
by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_25_l3739_373933


namespace NUMINAMATH_CALUDE_fraction_simplification_l3739_373982

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (18 - 13 * x) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3739_373982


namespace NUMINAMATH_CALUDE_spade_calculation_l3739_373911

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 2 (spade 6 1) = -1221 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l3739_373911


namespace NUMINAMATH_CALUDE_consecutive_values_exist_l3739_373999

/-- A polynomial that takes on three consecutive integer values at three consecutive integer points -/
def polynomial (a : ℤ) (x : ℤ) : ℤ := x^3 - 18*x^2 + a*x + 1784

theorem consecutive_values_exist :
  ∃ (k n : ℤ),
    polynomial a (k-1) = n-1 ∧
    polynomial a k = n ∧
    polynomial a (k+1) = n+1 :=
sorry

end NUMINAMATH_CALUDE_consecutive_values_exist_l3739_373999


namespace NUMINAMATH_CALUDE_largest_number_after_removal_l3739_373946

def first_ten_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def concatenated_primes : Nat :=
  first_ten_primes.foldl (fun acc n => acc * (10 ^ (Nat.digits 10 n).length) + n) 0

def remove_six_digits (n : Nat) : Set Nat :=
  { m | ∃ (digits : List Nat), 
    digits.length = (Nat.digits 10 n).length - 6 ∧
    (Nat.digits 10 m) = digits ∧
    (∀ d ∈ digits, d ∈ Nat.digits 10 n) }

theorem largest_number_after_removal :
  7317192329 ∈ remove_six_digits concatenated_primes ∧
  ∀ m ∈ remove_six_digits concatenated_primes, m ≤ 7317192329 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_after_removal_l3739_373946


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l3739_373932

theorem concentric_circles_ratio (r R : ℝ) (h : r > 0) (H : R > r) :
  π * R^2 - π * r^2 = 4 * (π * r^2) → r / R = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l3739_373932


namespace NUMINAMATH_CALUDE_expression_value_l3739_373956

theorem expression_value :
  let a : ℤ := 3
  let b : ℤ := 7
  let c : ℤ := 2
  ((a * b - c) - (a + b * c)) - ((a * c - b) - (a - b * c)) = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3739_373956


namespace NUMINAMATH_CALUDE_common_root_is_negative_one_l3739_373940

/-- Given two equations with a common root and a condition on the coefficients,
    prove that the common root is -1. -/
theorem common_root_is_negative_one (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ x : ℝ, x^2 + a*x + b = 0 ∧ x^3 + b*x + a = 0 → x = -1 := by
  sorry

#check common_root_is_negative_one

end NUMINAMATH_CALUDE_common_root_is_negative_one_l3739_373940


namespace NUMINAMATH_CALUDE_seminar_selection_l3739_373939

theorem seminar_selection (boys girls : ℕ) (total_select : ℕ) : 
  boys = 4 → girls = 3 → total_select = 4 →
  (Nat.choose (boys + girls) total_select) - (Nat.choose boys total_select) = 34 := by
sorry

end NUMINAMATH_CALUDE_seminar_selection_l3739_373939


namespace NUMINAMATH_CALUDE_probability_theorem_l3739_373979

/-- The number of tiles in box A -/
def num_tiles_A : ℕ := 20

/-- The number of tiles in box B -/
def num_tiles_B : ℕ := 30

/-- The lowest number on tiles in box A -/
def min_num_A : ℕ := 1

/-- The highest number on tiles in box A -/
def max_num_A : ℕ := 20

/-- The lowest number on tiles in box B -/
def min_num_B : ℕ := 10

/-- The highest number on tiles in box B -/
def max_num_B : ℕ := 39

/-- The probability of drawing a tile less than 10 from box A -/
def prob_A : ℚ := 9 / 20

/-- The probability of drawing a tile that is either odd or greater than 35 from box B -/
def prob_B : ℚ := 17 / 30

theorem probability_theorem :
  prob_A * prob_B = 51 / 200 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l3739_373979


namespace NUMINAMATH_CALUDE_min_value_theorem_l3739_373993

theorem min_value_theorem (a k b m n : ℝ) : 
  a > 0 → a ≠ 1 → 
  (∀ x, a^(x-1) + 1 = b → x = k) → 
  m + n = b - k → m > 0 → n > 0 → 
  (9/m + 1/n ≥ 16 ∧ ∃ m n, 9/m + 1/n = 16) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3739_373993


namespace NUMINAMATH_CALUDE_sum_parity_of_nine_consecutive_naturals_l3739_373918

theorem sum_parity_of_nine_consecutive_naturals (n : ℕ) :
  Even (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) ↔ Even n :=
by sorry

end NUMINAMATH_CALUDE_sum_parity_of_nine_consecutive_naturals_l3739_373918


namespace NUMINAMATH_CALUDE_equation_solution_l3739_373978

theorem equation_solution :
  ∃ x : ℝ, (x + Real.sqrt (x^2 - x) = 2) ∧ (x = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3739_373978


namespace NUMINAMATH_CALUDE_stevens_skittles_l3739_373964

theorem stevens_skittles (erasers : ℕ) (groups : ℕ) (items_per_group : ℕ) (skittles : ℕ) :
  erasers = 4276 →
  groups = 154 →
  items_per_group = 57 →
  skittles + erasers = groups * items_per_group →
  skittles = 4502 := by
  sorry

end NUMINAMATH_CALUDE_stevens_skittles_l3739_373964


namespace NUMINAMATH_CALUDE_inverse_proportional_solution_l3739_373919

theorem inverse_proportional_solution (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 30) (h3 : x - y = 6) :
  x = 6 → y = 36 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportional_solution_l3739_373919


namespace NUMINAMATH_CALUDE_cone_base_radius_l3739_373981

theorem cone_base_radius (α : Real) (n : Nat) (r : Real) (h₁ : α = 30 * π / 180) (h₂ : n = 11) (h₃ : r = 3) :
  let R := r * (1 / Real.sin (π / n) - 1 / Real.tan (π / 4 + α / 2))
  R = r / Real.sin (π / n) - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3739_373981


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l3739_373925

/-- A color type representing red or blue --/
inductive Color
  | Red
  | Blue

/-- A type representing a complete graph with 6 vertices --/
structure CompleteGraph6 where
  /-- A function assigning a color to each pair of distinct vertices --/
  edgeColor : Fin 6 → Fin 6 → Color
  /-- Ensure the graph is undirected --/
  symm : ∀ (i j : Fin 6), i ≠ j → edgeColor i j = edgeColor j i

/-- Definition of a monochromatic triangle in the graph --/
def hasMonochromaticTriangle (g : CompleteGraph6) : Prop :=
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    g.edgeColor i j = g.edgeColor j k ∧ g.edgeColor j k = g.edgeColor k i

/-- Theorem stating that every complete graph with 6 vertices and edges colored red or blue
    contains a monochromatic triangle --/
theorem monochromatic_triangle_exists (g : CompleteGraph6) : hasMonochromaticTriangle g :=
  sorry


end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l3739_373925


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3739_373908

/-- Triangle ABC with vertices A(-2, 3), B(5, 3), and C(5, -2) is a right triangle with perimeter 12 + √74 -/
theorem triangle_abc_properties :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (5, 3)
  let C : ℝ × ℝ := (5, -2)
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  -- Triangle ABC is a right triangle with right angle at B
  AB^2 + BC^2 = AC^2 ∧
  -- The perimeter of triangle ABC is 12 + √74
  AB + BC + AC = 12 + Real.sqrt 74 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3739_373908


namespace NUMINAMATH_CALUDE_blue_shirts_count_l3739_373949

theorem blue_shirts_count (total_shirts green_shirts : ℕ) 
  (h1 : total_shirts = 23)
  (h2 : green_shirts = 17)
  (h3 : total_shirts = green_shirts + blue_shirts) :
  blue_shirts = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_shirts_count_l3739_373949


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3739_373944

theorem diophantine_equation_solution (a : ℕ+) :
  ∃ (x y : ℕ+), (x^3 + x + a^2 : ℤ) = y^2 ∧
  x = 4 * a^2 * (16 * a^4 + 2) ∧
  y = 2 * a * (16 * a^4 + 2) * (16 * a^4 + 1) - a :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3739_373944


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l3739_373989

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 10

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 3

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 10

/-- The number of times Terrell must lift the new weights to match the total weight -/
def new_lifts : ℕ := 25

theorem terrell_weight_lifting :
  num_weights * original_weight * original_lifts =
  num_weights * new_weight * new_lifts := by
  sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l3739_373989


namespace NUMINAMATH_CALUDE_brick_surface_area_l3739_373941

/-- The surface area of a rectangular prism. -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a rectangular prism with dimensions 8 cm x 4 cm x 2 cm is 112 square centimeters. -/
theorem brick_surface_area :
  surface_area 8 4 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l3739_373941


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3739_373986

theorem geometric_sequence_first_term
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r = 4) -- second term is 4
  (h2 : a * r^3 = 16) -- fourth term is 16
  : a = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3739_373986


namespace NUMINAMATH_CALUDE_company_fund_calculation_l3739_373997

theorem company_fund_calculation (n : ℕ) 
  (h1 : 80 * n - 15 = 70 * n + 155) : 
  80 * n - 15 = 1345 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_calculation_l3739_373997


namespace NUMINAMATH_CALUDE_percentage_both_languages_l3739_373976

/-- Represents the number of diplomats speaking both French and Russian -/
def both_languages (total french not_russian neither : ℕ) : ℕ :=
  french + (total - not_russian) - (total - neither)

/-- Theorem stating the percentage of diplomats speaking both French and Russian -/
theorem percentage_both_languages :
  let total := 100
  let french := 22
  let not_russian := 32
  let neither := 20
  (both_languages total french not_russian neither : ℚ) / total * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_both_languages_l3739_373976


namespace NUMINAMATH_CALUDE_base7_to_base10_equality_l3739_373966

/-- Conversion from base 7 to base 10 -/
def base7to10 (n : ℕ) : ℕ := 
  7 * 7 * (n / 100) + 7 * ((n / 10) % 10) + (n % 10)

theorem base7_to_base10_equality (c d e : ℕ) : 
  (c < 10 ∧ d < 10 ∧ e < 10) → 
  (base7to10 761 = 100 * c + 10 * d + e) → 
  (d * e : ℚ) / 15 = 48 / 15 := by
sorry

end NUMINAMATH_CALUDE_base7_to_base10_equality_l3739_373966


namespace NUMINAMATH_CALUDE_bons_win_probability_l3739_373953

theorem bons_win_probability : 
  let p : ℝ := (1 : ℝ) / 6  -- Probability of rolling a six
  let q : ℝ := 1 - p        -- Probability of not rolling a six
  ∃ (win_prob : ℝ), 
    win_prob = q * p + q * q * win_prob ∧ 
    win_prob = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_bons_win_probability_l3739_373953


namespace NUMINAMATH_CALUDE_route_ratio_is_three_l3739_373912

-- Define the grid structure
structure Grid where
  -- Add necessary fields to represent the grid

-- Define a function to count routes
def countRoutes (g : Grid) (start : Nat × Nat) (steps : Nat) : Nat :=
  sorry

-- Define points A and B
def pointA : Nat × Nat := sorry
def pointB : Nat × Nat := sorry

-- Define the specific grid
def specificGrid : Grid := sorry

-- Theorem statement
theorem route_ratio_is_three :
  let m := countRoutes specificGrid pointA 4
  let n := countRoutes specificGrid pointB 4
  n / m = 3 := by sorry

end NUMINAMATH_CALUDE_route_ratio_is_three_l3739_373912


namespace NUMINAMATH_CALUDE_max_product_constraint_l3739_373973

theorem max_product_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_constraint : 5 * x + 8 * y + 3 * z = 90) :
  x * y * z ≤ 225 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    5 * x₀ + 8 * y₀ + 3 * z₀ = 90 ∧ x₀ * y₀ * z₀ = 225 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3739_373973


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3739_373975

/-- The distance function of a particle moving in a straight line -/
def s (t : ℝ) : ℝ := 3 * t^2 + t

/-- The instantaneous velocity of the particle at time t -/
def v (t : ℝ) : ℝ := 6 * t + 1

theorem instantaneous_velocity_at_2 : v 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3739_373975


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l3739_373917

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 11/15 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 11/15) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l3739_373917


namespace NUMINAMATH_CALUDE_new_student_weight_new_student_weight_is_46_l3739_373967

/-- The weight of a new student who replaces an 86 kg student in a group of 8,
    resulting in an average weight decrease of 5 kg. -/
theorem new_student_weight : ℝ :=
  let n : ℕ := 8 -- number of students
  let avg_decrease : ℝ := 5 -- average weight decrease in kg
  let replaced_weight : ℝ := 86 -- weight of the replaced student in kg
  replaced_weight - n * avg_decrease

/-- Proof that the new student's weight is 46 kg -/
theorem new_student_weight_is_46 : new_student_weight = 46 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_new_student_weight_is_46_l3739_373967


namespace NUMINAMATH_CALUDE_fraction_value_l3739_373928

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  (a * c) / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l3739_373928


namespace NUMINAMATH_CALUDE_fraction_used_is_47_48_l3739_373994

/-- Represents the car's journey with given parameters -/
structure CarJourney where
  tankCapacity : ℚ
  firstLegDuration : ℚ
  firstLegSpeed : ℚ
  firstLegConsumptionRate : ℚ
  refillAmount : ℚ
  secondLegDuration : ℚ
  secondLegSpeed : ℚ
  secondLegConsumptionRate : ℚ

/-- Calculates the fraction of a full tank used after the entire journey -/
def fractionUsed (journey : CarJourney) : ℚ :=
  let firstLegDistance := journey.firstLegDuration * journey.firstLegSpeed
  let firstLegUsed := firstLegDistance / journey.firstLegConsumptionRate
  let secondLegDistance := journey.secondLegDuration * journey.secondLegSpeed
  let secondLegUsed := secondLegDistance / journey.secondLegConsumptionRate
  (firstLegUsed + secondLegUsed) / journey.tankCapacity

/-- The specific journey described in the problem -/
def specificJourney : CarJourney :=
  { tankCapacity := 12
  , firstLegDuration := 3
  , firstLegSpeed := 50
  , firstLegConsumptionRate := 40
  , refillAmount := 5
  , secondLegDuration := 4
  , secondLegSpeed := 60
  , secondLegConsumptionRate := 30
  }

/-- Theorem stating that the fraction of tank used in the specific journey is 47/48 -/
theorem fraction_used_is_47_48 : fractionUsed specificJourney = 47 / 48 := by
  sorry


end NUMINAMATH_CALUDE_fraction_used_is_47_48_l3739_373994


namespace NUMINAMATH_CALUDE_tan_150_degrees_l3739_373972

theorem tan_150_degrees : 
  Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l3739_373972


namespace NUMINAMATH_CALUDE_rationalize_cube_root_seven_l3739_373926

def rationalize_denominator (a b : ℕ) : ℚ × ℕ × ℕ := sorry

theorem rationalize_cube_root_seven :
  let (frac, B, C) := rationalize_denominator 4 (3 * 7^(1/3))
  frac = 4 * (49^(1/3)) / 21 ∧ 
  B = 49 ∧ 
  C = 21 ∧
  4 + B + C = 74 := by sorry

end NUMINAMATH_CALUDE_rationalize_cube_root_seven_l3739_373926


namespace NUMINAMATH_CALUDE_company_employees_l3739_373984

/-- 
If a company had 15% more employees in December than in January,
and it had 450 employees in December, then it had 391 employees in January.
-/
theorem company_employees (december_employees : ℕ) (january_employees : ℕ) : 
  december_employees = 450 → 
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 391 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l3739_373984


namespace NUMINAMATH_CALUDE_log_expression_equality_l3739_373980

theorem log_expression_equality : 2 * (Real.log 256 / Real.log 4) - (Real.log (1/16) / Real.log 4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l3739_373980


namespace NUMINAMATH_CALUDE_payment_calculation_l3739_373935

/-- The payment for C given the work rates of A and B, total payment, and total work days -/
def payment_for_C (a_rate : ℚ) (b_rate : ℚ) (total_payment : ℚ) (total_days : ℚ) : ℚ :=
  let ab_rate := a_rate + b_rate
  let ab_work := ab_rate * total_days
  let c_work := 1 - ab_work
  c_work * total_payment

theorem payment_calculation (a_rate b_rate total_payment total_days : ℚ) 
  (ha : a_rate = 1/6)
  (hb : b_rate = 1/8)
  (hp : total_payment = 3360)
  (hd : total_days = 3) :
  payment_for_C a_rate b_rate total_payment total_days = 420 := by
  sorry

#eval payment_for_C (1/6) (1/8) 3360 3

end NUMINAMATH_CALUDE_payment_calculation_l3739_373935


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3739_373974

/-- Given a point P(-3, -5) in the Cartesian coordinate system,
    its coordinates with respect to the origin are (3, 5). -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-3, -5)
  (|P.1|, |P.2|) = (3, 5) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3739_373974


namespace NUMINAMATH_CALUDE_min_distance_sum_l3739_373955

/-- Given points M(-1,3) and N(2,1), and point P on the x-axis,
    the minimum value of PM+PN is 5. -/
theorem min_distance_sum (M N P : ℝ × ℝ) : 
  M = (-1, 3) → 
  N = (2, 1) → 
  P.2 = 0 → 
  ∃ (min_val : ℝ), (∀ Q : ℝ × ℝ, Q.2 = 0 → 
    Real.sqrt ((Q.1 - M.1)^2 + (Q.2 - M.2)^2) + 
    Real.sqrt ((Q.1 - N.1)^2 + (Q.2 - N.2)^2) ≥ min_val) ∧ 
  min_val = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_sum_l3739_373955


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3739_373969

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3739_373969


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3739_373998

theorem arithmetic_equality : (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3739_373998


namespace NUMINAMATH_CALUDE_circle_radius_proof_l3739_373930

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Theorem: Given the conditions, prove that the radius of circle k is 17 -/
theorem circle_radius_proof (k k1 k2 : Circle)
  (h1 : k1.radius = 8)
  (h2 : k2.radius = 15)
  (h3 : k1.radius < k.radius)
  (h4 : (k.radius ^ 2 - k1.radius ^ 2) * Real.pi = (k2.radius ^ 2 - k.radius ^ 2) * Real.pi) :
  k.radius = 17 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l3739_373930


namespace NUMINAMATH_CALUDE_grandma_age_l3739_373965

-- Define the number of grandchildren
def num_grandchildren : ℕ := 5

-- Define the average age of the entire group
def group_average_age : ℚ := 26

-- Define the average age of the grandchildren
def grandchildren_average_age : ℚ := 7

-- Define the age difference between grandpa and grandma
def age_difference : ℕ := 1

-- Theorem statement
theorem grandma_age :
  ∃ (grandpa_age grandma_age : ℕ),
    -- The average age of the group is 26
    (grandpa_age + grandma_age + num_grandchildren * grandchildren_average_age) / (2 + num_grandchildren : ℚ) = group_average_age ∧
    -- Grandma is one year younger than grandpa
    grandpa_age = grandma_age + age_difference ∧
    -- Grandma's age is 73
    grandma_age = 73 := by
  sorry

end NUMINAMATH_CALUDE_grandma_age_l3739_373965


namespace NUMINAMATH_CALUDE_point_transformation_l3739_373970

def rotate90CounterClockwise (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

def reflectAboutYEqualX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90CounterClockwise a b 2 3
  let (x₂, y₂) := reflectAboutYEqualX x₁ y₁
  (x₂ = 5 ∧ y₂ = -1) → b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l3739_373970


namespace NUMINAMATH_CALUDE_remainder_a_sixth_mod_n_l3739_373913

theorem remainder_a_sixth_mod_n (n : ℕ+) (a : ℤ) (h : a^3 ≡ 1 [ZMOD n]) :
  a^6 ≡ 1 [ZMOD n] := by sorry

end NUMINAMATH_CALUDE_remainder_a_sixth_mod_n_l3739_373913


namespace NUMINAMATH_CALUDE_room_length_calculation_l3739_373934

/-- Given a room with width 4 meters and a paving cost of 750 per square meter,
    if the total cost of paving is 16500, then the length of the room is 5.5 meters. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (length : ℝ) :
  width = 4 →
  cost_per_sqm = 750 →
  total_cost = 16500 →
  length * width * cost_per_sqm = total_cost →
  length = 5.5 := by
  sorry

#check room_length_calculation

end NUMINAMATH_CALUDE_room_length_calculation_l3739_373934


namespace NUMINAMATH_CALUDE_exam_time_allocation_l3739_373907

theorem exam_time_allocation (total_time : ℝ) (total_questions : ℕ) (type_a_count : ℕ) :
  total_time = 180 ∧
  total_questions = 200 ∧
  type_a_count = 10 →
  let type_b_count : ℕ := total_questions - type_a_count
  let time_ratio : ℝ := 2
  let type_b_time : ℝ := total_time / (type_b_count + time_ratio * type_a_count)
  let type_a_time : ℝ := time_ratio * type_b_time
  type_a_count * type_a_time = 120 / 7 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_allocation_l3739_373907


namespace NUMINAMATH_CALUDE_roy_sports_hours_l3739_373962

/-- Calculates the total hours spent on sports in school for a week with missed days -/
def sports_hours_in_week (daily_hours : ℕ) (school_days : ℕ) (missed_days : ℕ) : ℕ :=
  (school_days - missed_days) * daily_hours

/-- Proves that Roy spent 6 hours on sports in school for the given week -/
theorem roy_sports_hours :
  let daily_hours : ℕ := 2
  let school_days : ℕ := 5
  let missed_days : ℕ := 2
  sports_hours_in_week daily_hours school_days missed_days = 6 := by
  sorry

end NUMINAMATH_CALUDE_roy_sports_hours_l3739_373962


namespace NUMINAMATH_CALUDE_lg_root_relationship_l3739_373977

theorem lg_root_relationship : ∃ (M1 M2 M3 : ℝ),
  M1 > 0 ∧ M2 > 0 ∧ M3 > 0 ∧
  Real.log M1 / Real.log 10 < M1 ^ (1/10) ∧
  Real.log M2 / Real.log 10 > M2 ^ (1/10) ∧
  Real.log M3 / Real.log 10 = M3 ^ (1/10) :=
by sorry

end NUMINAMATH_CALUDE_lg_root_relationship_l3739_373977


namespace NUMINAMATH_CALUDE_train_journey_time_l3739_373959

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 →
  usual_time > 0 →
  (6/7 * usual_speed) * (usual_time + 15/60) = usual_speed * usual_time →
  usual_time = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l3739_373959


namespace NUMINAMATH_CALUDE_square_root_power_and_increasing_l3739_373961

-- Define the function f(x) = x^(1/2) on the interval (0, +∞)
def f : ℝ → ℝ := fun x ↦ x^(1/2)

-- Define the interval (0, +∞)
def openRightHalfLine : Set ℝ := {x : ℝ | x > 0}

theorem square_root_power_and_increasing :
  (∃ r : ℝ, ∀ x ∈ openRightHalfLine, f x = x^r) ∧
  StrictMonoOn f openRightHalfLine :=
sorry

end NUMINAMATH_CALUDE_square_root_power_and_increasing_l3739_373961


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l3739_373963

/-- Given a triangle ABC with the following properties:
  1. sin C + sin(B-A) = 3 sin(2A)
  2. c = 2
  3. ∠C = π/3
  Prove that the area of triangle ABC is either 2√3/3 or 3√3/7 -/
theorem triangle_area_theorem (A B C : ℝ) (h1 : Real.sin C + Real.sin (B - A) = 3 * Real.sin (2 * A))
    (h2 : 2 = 2) (h3 : C = π / 3) :
  let S := Real.sqrt 3 / 3 * 2
  let S' := Real.sqrt 3 * 3 / 7
  let area := (Real.sin C) * 2 / 2
  area = S ∨ area = S' := by
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l3739_373963


namespace NUMINAMATH_CALUDE_three_cell_corners_count_l3739_373968

theorem three_cell_corners_count (total_cells : ℕ) (x y : ℕ) : 
  total_cells = 22 → 
  3 * x + 4 * y = total_cells → 
  (x = 2 ∨ x = 6) ∧ (y = 4 ∨ y = 1) :=
sorry

end NUMINAMATH_CALUDE_three_cell_corners_count_l3739_373968


namespace NUMINAMATH_CALUDE_wickets_before_last_match_is_55_l3739_373988

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  lastMatchWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wicketsBeforeLastMatch (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the number of wickets before the last match is 55 -/
theorem wickets_before_last_match_is_55 (stats : BowlerStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.lastMatchWickets = 4)
  (h3 : stats.lastMatchRuns = 26)
  (h4 : stats.averageDecrease = 0.4) :
  wicketsBeforeLastMatch stats = 55 := by
  sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_is_55_l3739_373988


namespace NUMINAMATH_CALUDE_system_no_solution_l3739_373900

def has_no_solution (a b c : ℤ) : Prop :=
  (a * b = 6) ∧ (b * c = 8) ∧ (c / 4 ≠ c / (4 * b))

theorem system_no_solution :
  ∀ a b c : ℤ, has_no_solution a b c ↔ 
    ((a = -6 ∧ b = -1 ∧ c = -8) ∨
     (a = -3 ∧ b = -2 ∧ c = -4) ∨
     (a = 3 ∧ b = 2 ∧ c = 4)) :=
by sorry

end NUMINAMATH_CALUDE_system_no_solution_l3739_373900


namespace NUMINAMATH_CALUDE_binary_1100111_to_decimal_l3739_373938

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1100111₂ -/
def binary_1100111 : List Bool := [true, true, false, false, true, true, true]

/-- Theorem stating that the decimal equivalent of 1100111₂ is 103 -/
theorem binary_1100111_to_decimal :
  binary_to_decimal binary_1100111 = 103 := by
  sorry

end NUMINAMATH_CALUDE_binary_1100111_to_decimal_l3739_373938


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3739_373951

theorem trigonometric_equation_solution (x : ℝ) : 
  (abs (Real.sin x) + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 ↔ 
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3739_373951


namespace NUMINAMATH_CALUDE_base4_21202_equals_610_l3739_373958

def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

theorem base4_21202_equals_610 :
  base4ToBase10 [2, 0, 2, 1, 2] = 610 := by
  sorry

end NUMINAMATH_CALUDE_base4_21202_equals_610_l3739_373958


namespace NUMINAMATH_CALUDE_rug_length_is_25_l3739_373983

/-- Represents a rectangular rug with integer dimensions -/
structure Rug where
  length : ℕ
  width : ℕ

/-- Represents a rectangular room -/
structure Room where
  width : ℕ
  length : ℕ

/-- Checks if a rug fits perfectly in a room -/
def fitsInRoom (rug : Rug) (room : Room) : Prop :=
  rug.length ^ 2 + rug.width ^ 2 = room.width ^ 2 + room.length ^ 2

theorem rug_length_is_25 :
  ∃ (rug : Rug) (room1 room2 : Room),
    room1.width = 38 ∧
    room2.width = 50 ∧
    room1.length = room2.length ∧
    fitsInRoom rug room1 ∧
    fitsInRoom rug room2 →
    rug.length = 25 := by
  sorry

end NUMINAMATH_CALUDE_rug_length_is_25_l3739_373983


namespace NUMINAMATH_CALUDE_parabola_symmetry_point_l3739_373971

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a parabola -/
def onParabola (p : Parabola) (pt : Point) : Prop :=
  pt.y = p.a * (pt.x - p.h)^2 + p.k

theorem parabola_symmetry_point (p : Parabola) :
  ∃ (m : ℝ),
    onParabola p ⟨-1, 2⟩ ∧
    onParabola p ⟨1, -2⟩ ∧
    onParabola p ⟨3, 2⟩ ∧
    onParabola p ⟨-2, m⟩ ∧
    onParabola p ⟨4, m⟩ := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_point_l3739_373971


namespace NUMINAMATH_CALUDE_chessboard_selection_divisibility_l3739_373991

theorem chessboard_selection_divisibility (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ k : ℕ, p! - p = p^5 * k := by
  sorry

end NUMINAMATH_CALUDE_chessboard_selection_divisibility_l3739_373991


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3739_373915

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 3 →
  a 4 = 24 →
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3739_373915


namespace NUMINAMATH_CALUDE_car_distance_in_yards_l3739_373931

/-- Proves the distance traveled by a car in yards over 60 minutes -/
theorem car_distance_in_yards
  (b : ℝ) (s : ℝ) (h_s_pos : s > 0) :
  let feet_per_s_seconds : ℝ := 5 * b / 12
  let seconds_in_hour : ℝ := 60 * 60
  let feet_in_yard : ℝ := 3
  let distance_feet : ℝ := feet_per_s_seconds * seconds_in_hour / s
  let distance_yards : ℝ := distance_feet / feet_in_yard
  distance_yards = 500 * b / s :=
by sorry


end NUMINAMATH_CALUDE_car_distance_in_yards_l3739_373931


namespace NUMINAMATH_CALUDE_tangent_circle_center_and_radius_l3739_373943

/-- A circle tangent to y=x, y=-x, and y=10 with center above (10,10) -/
structure TangentCircle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_gt_ten : h > 10
  k_gt_ten : k > 10
  tangent_y_eq_x : r = |h - k| / Real.sqrt 2
  tangent_y_eq_neg_x : r = |h + k| / Real.sqrt 2
  tangent_y_eq_ten : r = k - 10

/-- The center and radius of a circle tangent to y=x, y=-x, and y=10 -/
theorem tangent_circle_center_and_radius (c : TangentCircle) :
  c.h = 10 + (1 + Real.sqrt 2) * c.r ∧ c.k = 10 + c.r :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_center_and_radius_l3739_373943


namespace NUMINAMATH_CALUDE_negation_p_iff_valid_range_l3739_373929

/-- The proposition p: There exists x₀ ∈ ℝ such that x₀² + ax₀ + a < 0 -/
def p (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + a*x₀ + a < 0

/-- The range of a for which ¬p holds -/
def valid_range (a : ℝ) : Prop := a ≤ 0 ∨ a ≥ 4

theorem negation_p_iff_valid_range (a : ℝ) :
  ¬(p a) ↔ valid_range a := by sorry

end NUMINAMATH_CALUDE_negation_p_iff_valid_range_l3739_373929


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3739_373942

theorem geometric_sequence_property (a b c q : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ 
    b + c - a = (a + b + c) * q ∧
    c + a - b = (a + b + c) * q^2 ∧
    a + b - c = (a + b + c) * q^3) →
  q^3 + q^2 + q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3739_373942


namespace NUMINAMATH_CALUDE_female_democrats_count_l3739_373923

theorem female_democrats_count 
  (total : ℕ) 
  (female : ℕ) 
  (male : ℕ) 
  (h1 : total = 750)
  (h2 : female + male = total)
  (h3 : (female / 2 + male / 4 : ℚ) = total / 3) :
  female / 2 = 125 := by
sorry

end NUMINAMATH_CALUDE_female_democrats_count_l3739_373923


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_l3739_373920

theorem polygon_sides_when_interior_thrice_exterior :
  ∀ n : ℕ,
  (n ≥ 3) →
  (180 * (n - 2) = 3 * 360) →
  n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_l3739_373920


namespace NUMINAMATH_CALUDE_woodwind_to_brass_ratio_l3739_373937

/-- Represents the composition of a marching band -/
structure MarchingBand where
  total : ℕ
  percussion : ℕ
  woodwind : ℕ
  brass : ℕ

/-- Checks if the marching band satisfies the given conditions -/
def validBand (band : MarchingBand) : Prop :=
  band.total = 110 ∧
  band.percussion = 4 * band.woodwind ∧
  band.brass = 10 ∧
  band.total = band.percussion + band.woodwind + band.brass

/-- Theorem stating the ratio of woodwind to brass players -/
theorem woodwind_to_brass_ratio (band : MarchingBand) 
  (h : validBand band) : 
  band.woodwind = 2 * band.brass :=
sorry

end NUMINAMATH_CALUDE_woodwind_to_brass_ratio_l3739_373937


namespace NUMINAMATH_CALUDE_special_numbers_count_l3739_373987

/-- Sum of digits of a positive integer -/
def digit_sum (x : ℕ+) : ℕ := sorry

/-- Counts the number of three-digit positive integers satisfying the condition -/
def count_special_numbers : ℕ := sorry

/-- Main theorem -/
theorem special_numbers_count :
  count_special_numbers = 14 := by sorry

end NUMINAMATH_CALUDE_special_numbers_count_l3739_373987


namespace NUMINAMATH_CALUDE_problem_solution_l3739_373995

open Real

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b * log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2 - x

theorem problem_solution :
  (∀ x > 0, Monotone (F (1/8))) ∧
  (∀ a ≥ 1/8, ∀ x > 0, Monotone (F a)) ∧
  (∃ b : ℝ, (b < -2 ∨ b > (ℯ^2 + 2)/(ℯ - 1)) ↔
    ∃ x₀ ∈ Set.Icc 1 ℯ, x₀ - f b x₀ < -(1 + b)/x₀) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3739_373995


namespace NUMINAMATH_CALUDE_cubic_function_property_l3739_373996

/-- Given a cubic function f(x) = ax³ + bx + 1, prove that if f(m) = 6, then f(-m) = -4 -/
theorem cubic_function_property (a b m : ℝ) : 
  (fun x => a * x^3 + b * x + 1) m = 6 →
  (fun x => a * x^3 + b * x + 1) (-m) = -4 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3739_373996


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l3739_373927

theorem smallest_cube_root_with_small_remainder :
  ∃ (m : ℕ) (r : ℝ),
    (∀ (m' : ℕ) (r' : ℝ), m' < m → ¬(∃ (n' : ℕ), m'^(1/3 : ℝ) = n' + r' ∧ 0 < r' ∧ r' < 1/10000)) ∧
    m^(1/3 : ℝ) = 58 + r ∧
    0 < r ∧
    r < 1/10000 ∧
    (∀ (n : ℕ), n < 58 → 
      ¬(∃ (m' : ℕ) (r' : ℝ), m'^(1/3 : ℝ) = n + r' ∧ 0 < r' ∧ r' < 1/10000)) :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l3739_373927


namespace NUMINAMATH_CALUDE_multiples_of_four_l3739_373954

theorem multiples_of_four (n : ℕ) : n = 16 ↔ 
  (∃ (l : List ℕ), l.length = 25 ∧ 
    (∀ m ∈ l, m % 4 = 0 ∧ n ≤ m ∧ m ≤ 112) ∧
    (∀ k : ℕ, n ≤ k ∧ k ≤ 112 ∧ k % 4 = 0 → k ∈ l) ∧
    (∀ m : ℕ, n < m → 
      ¬∃ (l' : List ℕ), l'.length = 25 ∧ 
        (∀ m' ∈ l', m' % 4 = 0 ∧ m ≤ m' ∧ m' ≤ 112) ∧
        (∀ k : ℕ, m ≤ k ∧ k ≤ 112 ∧ k % 4 = 0 → k ∈ l'))) :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_four_l3739_373954


namespace NUMINAMATH_CALUDE_no_integer_solution_l3739_373957

theorem no_integer_solution (a b c d : ℤ) : 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧
   a * 62^3 + b * 62^2 + c * 62 + d = 2) → False :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3739_373957


namespace NUMINAMATH_CALUDE_medal_award_combinations_l3739_373916

/-- The number of sprinters --/
def total_sprinters : ℕ := 10

/-- The number of American sprinters --/
def american_sprinters : ℕ := 4

/-- The number of non-American sprinters --/
def non_american_sprinters : ℕ := total_sprinters - american_sprinters

/-- The number of medals to be awarded --/
def medals : ℕ := 4

/-- The maximum number of Americans that can win medals --/
def max_american_winners : ℕ := 2

/-- The function to calculate the number of ways medals can be awarded --/
def ways_to_award_medals : ℕ := sorry

/-- Theorem stating that the number of ways to award medals is 6600 --/
theorem medal_award_combinations : ways_to_award_medals = 6600 := by sorry

end NUMINAMATH_CALUDE_medal_award_combinations_l3739_373916
