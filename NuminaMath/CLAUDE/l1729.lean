import Mathlib

namespace oatmeal_cookies_count_l1729_172929

/-- The number of batches of chocolate chip cookies -/
def chocolate_chip_batches : ℕ := 2

/-- The number of cookies in each batch of chocolate chip cookies -/
def cookies_per_batch : ℕ := 3

/-- The total number of cookies baked -/
def total_cookies : ℕ := 10

/-- The number of oatmeal cookies -/
def oatmeal_cookies : ℕ := total_cookies - (chocolate_chip_batches * cookies_per_batch)

theorem oatmeal_cookies_count : oatmeal_cookies = 4 := by
  sorry

end oatmeal_cookies_count_l1729_172929


namespace tangent_point_coordinates_l1729_172974

/-- The curve y = x^4 - x -/
def f (x : ℝ) : ℝ := x^4 - x

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_point_coordinates :
  ∀ x y : ℝ,
  y = f x →
  f' x = 3 →
  x = 1 ∧ y = 0 :=
by sorry

end tangent_point_coordinates_l1729_172974


namespace johns_outfit_cost_l1729_172913

/-- The cost of John's outfit given the cost of his pants and the relative cost of his shirt. -/
theorem johns_outfit_cost (pants_cost : ℝ) (shirt_relative_cost : ℝ) : 
  pants_cost = 50 →
  shirt_relative_cost = 0.6 →
  pants_cost + (pants_cost + pants_cost * shirt_relative_cost) = 130 := by
  sorry

end johns_outfit_cost_l1729_172913


namespace base_8_to_10_conversion_l1729_172919

theorem base_8_to_10_conversion : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0 : ℕ) = 2394 := by
  sorry

end base_8_to_10_conversion_l1729_172919


namespace expansion_coefficient_l1729_172917

/-- The coefficient of x^2 in the expansion of (x - 2/x)^6 -/
def coefficient_x_squared : ℕ :=
  let n : ℕ := 6
  let k : ℕ := 2
  ((-2)^k : ℤ) * (Nat.choose n k) |>.natAbs

theorem expansion_coefficient :
  coefficient_x_squared = 60 := by
  sorry

end expansion_coefficient_l1729_172917


namespace photo_frame_perimeter_l1729_172922

theorem photo_frame_perimeter (frame_width : ℝ) (frame_area : ℝ) (outer_edge : ℝ) :
  frame_width = 2 →
  frame_area = 48 →
  outer_edge = 10 →
  ∃ (photo_length photo_width : ℝ),
    photo_length = outer_edge - 2 * frame_width ∧
    photo_width * (outer_edge - 2 * frame_width) = outer_edge * (frame_area / outer_edge) - frame_area ∧
    2 * (photo_length + photo_width) = 16 :=
by sorry

end photo_frame_perimeter_l1729_172922


namespace find_p_l1729_172909

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 2 → a n = a (n - 1) + a (n - 2)

theorem find_p (a : ℕ → ℕ) (h : fibonacci_like_sequence a) 
  (h5 : a 4 = 5) (h8 : a 5 = 8) (h13 : a 6 = 13) : a 0 = 1 := by
  sorry

end find_p_l1729_172909


namespace appropriate_sampling_methods_l1729_172946

-- Define the structure for a survey
structure Survey where
  total_population : ℕ
  sample_size : ℕ
  has_distinct_groups : Bool

-- Define the sampling methods
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling

-- Define the function to determine the appropriate sampling method
def appropriate_sampling_method (survey : Survey) : SamplingMethod :=
  if survey.has_distinct_groups && survey.total_population > survey.sample_size * 10
  then SamplingMethod.StratifiedSampling
  else SamplingMethod.SimpleRandomSampling

-- Define the surveys
def survey1 : Survey := {
  total_population := 125 + 280 + 95,
  sample_size := 100,
  has_distinct_groups := true
}

def survey2 : Survey := {
  total_population := 15,
  sample_size := 3,
  has_distinct_groups := false
}

-- Theorem to prove
theorem appropriate_sampling_methods :
  appropriate_sampling_method survey1 = SamplingMethod.StratifiedSampling ∧
  appropriate_sampling_method survey2 = SamplingMethod.SimpleRandomSampling :=
by sorry


end appropriate_sampling_methods_l1729_172946


namespace quadratic_two_roots_l1729_172991

theorem quadratic_two_roots (b : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (∀ x : ℝ, x^2 + b*x - 3 = 0 ↔ x = x₁ ∨ x = x₂) := by
sorry

end quadratic_two_roots_l1729_172991


namespace largest_number_problem_l1729_172957

theorem largest_number_problem (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 75)
  (h_diff_large : c - b = 5)
  (h_diff_small : b - a = 4) :
  c = 89 / 3 := by
sorry

end largest_number_problem_l1729_172957


namespace thirteen_fourth_mod_eight_l1729_172941

theorem thirteen_fourth_mod_eight (m : ℕ) : 
  13^4 % 8 = m ∧ 0 ≤ m ∧ m < 8 → m = 1 := by
  sorry

end thirteen_fourth_mod_eight_l1729_172941


namespace audit_options_second_week_l1729_172939

def remaining_OR : ℕ := 13 - 2
def remaining_GTU : ℕ := 15 - 3

theorem audit_options_second_week : 
  (remaining_OR.choose 2) * (remaining_GTU.choose 3) = 12100 := by
  sorry

end audit_options_second_week_l1729_172939


namespace unique_solution_3x_minus_5y_equals_z2_l1729_172983

theorem unique_solution_3x_minus_5y_equals_z2 :
  ∀ x y z : ℕ+, 3^(x : ℕ) - 5^(y : ℕ) = (z : ℕ)^2 → x = 2 ∧ y = 2 ∧ z = 4 :=
by sorry

end unique_solution_3x_minus_5y_equals_z2_l1729_172983


namespace power_equality_l1729_172972

theorem power_equality (p : ℕ) (h : (81 : ℕ)^10 = 3^p) : p = 40 := by
  sorry

end power_equality_l1729_172972


namespace parabola_translation_l1729_172992

/-- Original parabola function -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Translated parabola function -/
def g (x : ℝ) : ℝ := x^2 + 4*x + 5

/-- Translation function -/
def translate (x : ℝ) : ℝ := x + 2

theorem parabola_translation :
  ∀ x : ℝ, g x = f (translate x) :=
by sorry

end parabola_translation_l1729_172992


namespace wxyz_equals_mpwy_l1729_172920

/-- Assigns a numeric value to each letter of the alphabet -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14
  | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _ => 0

/-- The product of a four-letter list -/
def four_letter_product (a b c d : Char) : ℕ :=
  letter_value a * letter_value b * letter_value c * letter_value d

theorem wxyz_equals_mpwy :
  four_letter_product 'W' 'X' 'Y' 'Z' = four_letter_product 'M' 'P' 'W' 'Y' :=
by sorry

end wxyz_equals_mpwy_l1729_172920


namespace equation_solution_l1729_172900

theorem equation_solution (x : ℝ) : 
  |x - 3| + x^2 = 10 ↔ 
  x = (-1 + Real.sqrt 53) / 2 ∨ 
  x = (1 + Real.sqrt 29) / 2 ∨ 
  x = (1 - Real.sqrt 29) / 2 :=
by sorry

end equation_solution_l1729_172900


namespace total_fruits_l1729_172938

/-- The number of pieces of fruit in three buckets -/
structure FruitBuckets where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The conditions for the fruit bucket problem -/
def fruit_bucket_conditions (fb : FruitBuckets) : Prop :=
  fb.A = fb.B + 4 ∧ fb.B = fb.C + 3 ∧ fb.C = 9

/-- The theorem stating that the total number of fruits in all buckets is 37 -/
theorem total_fruits (fb : FruitBuckets) 
  (h : fruit_bucket_conditions fb) : fb.A + fb.B + fb.C = 37 := by
  sorry

end total_fruits_l1729_172938


namespace quadratic_equation_solution_l1729_172948

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*x - 3
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -3 := by
  sorry

end quadratic_equation_solution_l1729_172948


namespace complex_power_195_deg_36_l1729_172956

theorem complex_power_195_deg_36 :
  (Complex.exp (195 * π / 180 * Complex.I)) ^ 36 = -1 := by
  sorry

end complex_power_195_deg_36_l1729_172956


namespace abc_congruence_l1729_172935

theorem abc_congruence (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 →
  (2 * a + 3 * b + c) % 7 = 1 →
  (3 * a + b + 2 * c) % 7 = 2 →
  (a + b + c) % 7 = 3 →
  (2 * a * b * c) % 7 = 0 := by
sorry

end abc_congruence_l1729_172935


namespace equation_has_real_root_l1729_172947

theorem equation_has_real_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a > b) (hbc : b > c) : 
  ∃ x : ℝ, (1 / (x + a) + 1 / (x + b) + 1 / (x + c) - 3 / x = 0) := by
sorry

end equation_has_real_root_l1729_172947


namespace target_cube_volume_l1729_172981

-- Define the volume of the reference cube
def reference_volume : ℝ := 8

-- Define the surface area of the target cube in terms of the reference cube
def target_surface_area (reference_side : ℝ) : ℝ := 2 * (6 * reference_side^2)

-- Define the volume of a cube given its side length
def cube_volume (side : ℝ) : ℝ := side^3

-- Define the surface area of a cube given its side length
def cube_surface_area (side : ℝ) : ℝ := 6 * side^2

-- Theorem statement
theorem target_cube_volume :
  ∃ (target_side : ℝ),
    cube_surface_area target_side = target_surface_area (reference_volume^(1/3)) ∧
    cube_volume target_side = 16 * Real.sqrt 2 :=
by sorry

end target_cube_volume_l1729_172981


namespace three_digit_sum_condition_l1729_172904

/-- Represents a three-digit number abc in decimal form -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_range : a ≤ 9
  b_range : b ≤ 9
  c_range : c ≤ 9
  a_nonzero : a ≠ 0

/-- The value of a three-digit number abc in decimal form -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of all two-digit numbers formed by the digits a, b, c -/
def ThreeDigitNumber.sumTwoDigit (n : ThreeDigitNumber) : Nat :=
  2 * (10 * n.a + 10 * n.b + 10 * n.c)

/-- A three-digit number satisfies the condition if its value equals the sum of all two-digit numbers formed by its digits -/
def ThreeDigitNumber.satisfiesCondition (n : ThreeDigitNumber) : Prop :=
  n.value = n.sumTwoDigit

/-- The theorem stating that only 132, 264, and 396 satisfy the condition -/
theorem three_digit_sum_condition :
  ∀ n : ThreeDigitNumber, n.satisfiesCondition ↔ (n.value = 132 ∨ n.value = 264 ∨ n.value = 396) := by
  sorry


end three_digit_sum_condition_l1729_172904


namespace power_of_prime_squared_minus_one_l1729_172902

theorem power_of_prime_squared_minus_one (n : ℕ) : 
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ n^2 - 1 = p^k) ↔ n = 2 ∨ n = 3 :=
by sorry

end power_of_prime_squared_minus_one_l1729_172902


namespace max_distance_C_D_l1729_172994

def C : Set ℂ := {z : ℂ | z^4 - 16 = 0}
def D : Set ℂ := {z : ℂ | z^3 - 27 = 0}

theorem max_distance_C_D : 
  ∃ (c : C) (d : D), ∀ (c' : C) (d' : D), Complex.abs (c - d) ≥ Complex.abs (c' - d') ∧ 
  Complex.abs (c - d) = Real.sqrt 13 :=
sorry

end max_distance_C_D_l1729_172994


namespace consecutive_integers_product_812_sum_57_l1729_172903

theorem consecutive_integers_product_812_sum_57 (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end consecutive_integers_product_812_sum_57_l1729_172903


namespace range_of_a_for_union_complement_l1729_172968

-- Define the sets M, N, and A
def M : Set ℝ := {x | 1 < x ∧ x < 4}
def N : Set ℝ := {x | 3 < x ∧ x < 5}
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}

-- Define the theorem
theorem range_of_a_for_union_complement (a : ℝ) : 
  (A a ∪ (Set.univ \ N) = Set.univ) ↔ (2 ≤ a ∧ a ≤ 3) :=
sorry

end range_of_a_for_union_complement_l1729_172968


namespace roses_in_vase_l1729_172940

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 7

/-- The number of roses added to the vase -/
def added_roses : ℕ := 16

/-- The total number of roses after addition -/
def total_roses : ℕ := 23

theorem roses_in_vase : initial_roses + added_roses = total_roses := by
  sorry

end roses_in_vase_l1729_172940


namespace apples_per_box_l1729_172945

theorem apples_per_box (total_apples : ℕ) (num_boxes : ℕ) (leftover_apples : ℕ) 
  (h1 : total_apples = 32) 
  (h2 : num_boxes = 7) 
  (h3 : leftover_apples = 4) : 
  (total_apples - leftover_apples) / num_boxes = 4 := by
  sorry

end apples_per_box_l1729_172945


namespace parabola_shift_l1729_172908

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift :
  let original := Parabola.mk 1 (-2) 3
  let shifted := shift_parabola original 1 (-3)
  shifted = Parabola.mk 1 0 (-1) := by sorry

end parabola_shift_l1729_172908


namespace laundry_items_not_done_l1729_172928

theorem laundry_items_not_done (short_sleeve : ℕ) (long_sleeve : ℕ) (socks : ℕ) (handkerchiefs : ℕ)
  (shirts_washed : ℕ) (socks_folded : ℕ) (handkerchiefs_sorted : ℕ)
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 27)
  (h3 : socks = 50)
  (h4 : handkerchiefs = 34)
  (h5 : shirts_washed = 20)
  (h6 : socks_folded = 30)
  (h7 : handkerchiefs_sorted = 16) :
  (short_sleeve + long_sleeve - shirts_washed) + (socks - socks_folded) + (handkerchiefs - handkerchiefs_sorted) = 54 :=
by sorry

end laundry_items_not_done_l1729_172928


namespace average_price_is_52_cents_l1729_172927

/-- Represents the fruit selection problem -/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  oranges_removed : ℕ

/-- Calculates the average price of fruits kept -/
def average_price_kept (fs : FruitSelection) : ℚ :=
  sorry

/-- Theorem stating the average price of fruits kept is 52 cents -/
theorem average_price_is_52_cents (fs : FruitSelection) 
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 20)
  (h4 : fs.initial_avg_price = 56/100)
  (h5 : fs.oranges_removed = 10) :
  average_price_kept fs = 52/100 :=
by sorry

end average_price_is_52_cents_l1729_172927


namespace unique_solution_l1729_172971

-- Define the equation
def equation (p x : ℝ) : Prop := x + 1 = Real.sqrt (p * x)

-- Define the conditions
def conditions (p x : ℝ) : Prop := p * x ≥ 0 ∧ x + 1 ≥ 0

-- Theorem statement
theorem unique_solution (p : ℝ) :
  (∃! x, equation p x ∧ conditions p x) ↔ (p = 4 ∨ p ≤ 0) := by sorry

end unique_solution_l1729_172971


namespace arithmetic_sequence_x_value_l1729_172937

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_x_value
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 2 * x - 3)
  (h_a3 : a 3 = 5 * x + 4)
  : x = -11 := by
  sorry

end arithmetic_sequence_x_value_l1729_172937


namespace distance_to_x_axis_l1729_172906

def point : ℝ × ℝ := (2, -3)

theorem distance_to_x_axis : abs (point.2) = 3 := by sorry

end distance_to_x_axis_l1729_172906


namespace sum_first_50_digits_1_101_l1729_172980

/-- The decimal representation of 1/101 -/
def decimal_rep_1_101 : ℕ → ℕ
| 0 => 0  -- First digit after decimal point
| 1 => 0  -- Second digit
| 2 => 9  -- Third digit
| 3 => 9  -- Fourth digit
| n + 4 => decimal_rep_1_101 n  -- Repeating pattern

/-- Sum of the first n digits after the decimal point in 1/101 -/
def sum_digits (n : ℕ) : ℕ :=
  (List.range n).map decimal_rep_1_101 |>.sum

/-- The sum of the first 50 digits after the decimal point in the decimal representation of 1/101 is 216 -/
theorem sum_first_50_digits_1_101 : sum_digits 50 = 216 := by
  sorry

end sum_first_50_digits_1_101_l1729_172980


namespace expected_reflections_l1729_172963

open Real

/-- The expected number of reflections for a billiard ball on a rectangular table -/
theorem expected_reflections (table_length table_width ball_travel : ℝ) :
  table_length = 3 →
  table_width = 1 →
  ball_travel = 2 →
  ∃ (expected_reflections : ℝ),
    expected_reflections = (2 / π) * (3 * arccos (1/4) - arcsin (3/4) + arccos (3/4)) := by
  sorry

end expected_reflections_l1729_172963


namespace circle_equation_l1729_172977

/-- The equation of a circle with center (±2, 1) and radius 2, given specific conditions -/
theorem circle_equation (x y : ℝ) : 
  (∃ t : ℝ, t = 2 ∨ t = -2) →
  (1 : ℝ) = (1/4) * t^2 →
  (abs t = abs ((1/4) * t^2 + 1)) →
  (x^2 + y^2 + 4*x - 2*y - 1 = 0) ∨ (x^2 + y^2 - 4*x - 2*y - 1 = 0) :=
by sorry

end circle_equation_l1729_172977


namespace derivative_f_l1729_172960

noncomputable def f (x : ℝ) : ℝ := 2 / x + x

theorem derivative_f :
  (∀ x : ℝ, x ≠ 0 → HasDerivAt f ((- 2 / x ^ 2) + 1) x) ∧
  HasDerivAt f (-1) 1 ∧
  HasDerivAt f (1/2) (-2) := by sorry

end derivative_f_l1729_172960


namespace quadratic_equation_solution_l1729_172914

theorem quadratic_equation_solution (p : ℝ) (α β : ℝ) : 
  (∀ x, x^2 + p*x + p = 0 ↔ x = α ∨ x = β) →
  (α^2 + β^2 = 3) →
  p = -1 := by
  sorry

end quadratic_equation_solution_l1729_172914


namespace school_time_problem_l1729_172984

/-- Given a boy who reaches school 3 minutes early when walking at 7/6 of his usual rate,
    his usual time to reach school is 21 minutes. -/
theorem school_time_problem (usual_rate : ℝ) (usual_time : ℝ) 
  (h1 : usual_rate > 0)
  (h2 : usual_time > 0)
  (h3 : usual_rate * usual_time = (7/6 * usual_rate) * (usual_time - 3)) :
  usual_time = 21 := by
  sorry

end school_time_problem_l1729_172984


namespace total_students_amc8_l1729_172969

/-- Represents a math teacher at Pythagoras Middle School -/
inductive Teacher : Type
| Euler : Teacher
| Noether : Teacher
| Gauss : Teacher
| Riemann : Teacher

/-- Returns the number of students in a teacher's class -/
def studentsInClass (t : Teacher) : Nat :=
  match t with
  | Teacher.Euler => 13
  | Teacher.Noether => 10
  | Teacher.Gauss => 12
  | Teacher.Riemann => 7

/-- The list of all teachers at Pythagoras Middle School -/
def allTeachers : List Teacher :=
  [Teacher.Euler, Teacher.Noether, Teacher.Gauss, Teacher.Riemann]

/-- Theorem stating that the total number of students taking the AMC 8 contest is 42 -/
theorem total_students_amc8 :
  (allTeachers.map studentsInClass).sum = 42 := by
  sorry

end total_students_amc8_l1729_172969


namespace age_difference_l1729_172970

/-- Given three people A, B, and C, with B being 8 years old, B twice as old as C,
    and the total of their ages being 22, prove that A is 2 years older than B. -/
theorem age_difference (B C : ℕ) (A : ℕ) : 
  B = 8 → B = 2 * C → A + B + C = 22 → A = B + 2 :=
by
  sorry

end age_difference_l1729_172970


namespace log_8_512_l1729_172979

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_8_512 : log 8 512 = 3 := by
  sorry

end log_8_512_l1729_172979


namespace collinear_vectors_m_value_l1729_172911

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1

/-- Two vectors have the same direction if their corresponding components have the same sign -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 ≥ 0) ∧ (a.2 * b.2 ≥ 0)

theorem collinear_vectors_m_value :
  ∀ (m : ℝ),
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4, m)
  collinear a b → same_direction a b → m = 2 := by
sorry

end collinear_vectors_m_value_l1729_172911


namespace difference_of_squares_262_258_l1729_172958

theorem difference_of_squares_262_258 : 262^2 - 258^2 = 2080 := by sorry

end difference_of_squares_262_258_l1729_172958


namespace job_completion_time_B_l1729_172996

theorem job_completion_time_B (r_A r_B r_C : ℝ) : 
  (r_A + r_B = 1 / 3) →
  (r_B + r_C = 2 / 7) →
  (r_A + r_C = 1 / 4) →
  (1 / r_B = 168 / 31) :=
by sorry

end job_completion_time_B_l1729_172996


namespace inequality_proof_l1729_172933

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / Real.sqrt (b + 1 / a + 1 / 2) + 1 / Real.sqrt (c + 1 / b + 1 / 2) + 1 / Real.sqrt (a + 1 / c + 1 / 2) ≥ Real.sqrt 2 := by
  sorry

end inequality_proof_l1729_172933


namespace ruth_gave_53_stickers_l1729_172915

/-- The number of stickers Janet initially had -/
def initial_stickers : ℕ := 3

/-- The total number of stickers Janet has after receiving more from Ruth -/
def final_stickers : ℕ := 56

/-- The number of stickers Ruth gave to Janet -/
def stickers_from_ruth : ℕ := final_stickers - initial_stickers

theorem ruth_gave_53_stickers : stickers_from_ruth = 53 := by
  sorry

end ruth_gave_53_stickers_l1729_172915


namespace cherry_tomato_yield_theorem_l1729_172967

-- Define the relationship between number of plants and yield
def yield_function (x : ℕ) : ℝ := -0.5 * (x : ℝ) + 5

-- Define the total yield function
def total_yield (x : ℕ) : ℝ := (x : ℝ) * yield_function x

-- Theorem statement
theorem cherry_tomato_yield_theorem (x : ℕ) 
  (h1 : 2 ≤ x ∧ x ≤ 8) 
  (h2 : yield_function 2 = 4) 
  (h3 : ∀ n : ℕ, 2 ≤ n ∧ n < 8 → yield_function (n + 1) = yield_function n - 0.5) :
  (∀ n : ℕ, 2 ≤ n ∧ n ≤ 8 → yield_function n = -0.5 * (n : ℝ) + 5) ∧
  (∃ max_yield : ℝ, max_yield = 12.5 ∧ 
    ∀ n : ℕ, 2 ≤ n ∧ n ≤ 8 → total_yield n ≤ max_yield) ∧
  (total_yield 5 = 12.5) :=
by sorry

end cherry_tomato_yield_theorem_l1729_172967


namespace determine_counterfeit_weight_l1729_172916

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal : WeighingResult
  | LeftHeavier : WeighingResult
  | RightHeavier : WeighingResult

/-- Represents a coin -/
structure Coin :=
  (id : Nat)
  (isCounterfeit : Bool)

/-- Represents a weighing on a two-pan balance scale -/
def weighing (leftPan : List Coin) (rightPan : List Coin) : WeighingResult :=
  sorry

/-- The main theorem stating that it's possible to determine if counterfeit coins are heavier or lighter -/
theorem determine_counterfeit_weight
  (coins : List Coin)
  (h1 : coins.length = 61)
  (h2 : (coins.filter (fun c => c.isCounterfeit)).length = 2)
  (h3 : ∀ c1 c2 : Coin, ¬c1.isCounterfeit ∧ ¬c2.isCounterfeit → c1.id ≠ c2.id → weighing [c1] [c2] = WeighingResult.Equal)
  (h4 : ∃ w : WeighingResult, w ≠ WeighingResult.Equal ∧ 
    ∀ c1 c2 : Coin, c1.isCounterfeit ∧ ¬c2.isCounterfeit → weighing [c1] [c2] = w) :
  ∃ (f : List (List Coin × List Coin)), 
    f.length ≤ 3 ∧ 
    (∃ (result : Bool), 
      result = true → (∀ c1 c2 : Coin, c1.isCounterfeit ∧ ¬c2.isCounterfeit → weighing [c1] [c2] = WeighingResult.LeftHeavier) ∧
      result = false → (∀ c1 c2 : Coin, c1.isCounterfeit ∧ ¬c2.isCounterfeit → weighing [c1] [c2] = WeighingResult.RightHeavier)) :=
  sorry

end determine_counterfeit_weight_l1729_172916


namespace perpendicular_vectors_tan_theta_l1729_172959

theorem perpendicular_vectors_tan_theta (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2))
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h2 : a = (Real.cos θ, 2))
  (h3 : b = (-1, Real.sin θ))
  (h4 : a.1 * b.1 + a.2 * b.2 = 0) : 
  Real.tan θ = 1 / 2 := by
  sorry

end perpendicular_vectors_tan_theta_l1729_172959


namespace additional_oil_purchased_l1729_172926

/-- Proves that a 30% price reduction allows purchasing 9 more kgs of oil with a budget of 900 Rs. --/
theorem additional_oil_purchased (budget : ℝ) (reduced_price : ℝ) (reduction_percentage : ℝ) : 
  budget = 900 →
  reduced_price = 30 →
  reduction_percentage = 0.3 →
  ⌊budget / reduced_price - budget / (reduced_price / (1 - reduction_percentage))⌋ = 9 := by
  sorry

end additional_oil_purchased_l1729_172926


namespace disneyland_arrangement_l1729_172999

def number_of_arrangements (total : ℕ) (type_a : ℕ) (type_b : ℕ) : ℕ :=
  (Nat.factorial type_a) * (Nat.factorial type_b)

theorem disneyland_arrangement :
  let total := 6
  let type_a := 2
  let type_b := 4
  number_of_arrangements total type_a type_b = 48 := by
  sorry

end disneyland_arrangement_l1729_172999


namespace subtracted_value_proof_l1729_172923

theorem subtracted_value_proof (n : ℝ) (x : ℝ) : 
  n = 15.0 → 3 * n - x = 40 → x = 5.0 := by
  sorry

end subtracted_value_proof_l1729_172923


namespace quadratic_form_h_value_l1729_172997

theorem quadratic_form_h_value (a b c : ℝ) :
  (∃ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 7) →
  (∃ n k, ∀ x, 4 * (a * x^2 + b * x + c) = n * (x - 5)^2 + k) :=
by sorry

end quadratic_form_h_value_l1729_172997


namespace man_speed_against_current_l1729_172905

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specified speeds, 
    the man's speed against the current is 10 km/hr. -/
theorem man_speed_against_current :
  speed_against_current 15 2.5 = 10 := by
  sorry

end man_speed_against_current_l1729_172905


namespace two_digit_number_sum_l1729_172982

theorem two_digit_number_sum (tens : ℕ) (units : ℕ) : 
  tens < 10 → 
  units = 6 * tens → 
  10 * tens + units = 16 → 
  tens + units = 7 := by
  sorry

end two_digit_number_sum_l1729_172982


namespace trapezoid_median_equals_nine_inches_l1729_172930

/-- Given a triangle and a trapezoid with equal areas and the same altitude,
    if the base of the triangle is 18 inches, then the median of the trapezoid is 9 inches. -/
theorem trapezoid_median_equals_nine_inches 
  (triangle_area trapezoid_area : ℝ) 
  (triangle_altitude trapezoid_altitude : ℝ) 
  (triangle_base : ℝ) 
  (trapezoid_median : ℝ) :
  triangle_area = trapezoid_area →
  triangle_altitude = trapezoid_altitude →
  triangle_base = 18 →
  triangle_area = (1/2) * triangle_base * triangle_altitude →
  trapezoid_area = trapezoid_median * trapezoid_altitude →
  trapezoid_median = 9 := by
  sorry

end trapezoid_median_equals_nine_inches_l1729_172930


namespace simplest_radical_l1729_172973

-- Define a function to check if a number is a quadratic radical
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ y ≠ 1

-- Define a function to check if a quadratic radical is in its simplest form
def is_simplest_radical (x : ℝ) : Prop :=
  is_quadratic_radical x ∧
  ∀ (y : ℝ), y ≠ x → (is_quadratic_radical y → ¬(x^2 = y^2))

-- Theorem statement
theorem simplest_radical :
  is_simplest_radical (Real.sqrt 7) ∧
  ¬(is_simplest_radical 1) ∧
  ¬(is_simplest_radical (Real.sqrt 12)) ∧
  ¬(is_simplest_radical (1 / Real.sqrt 13)) :=
sorry

end simplest_radical_l1729_172973


namespace city_population_problem_l1729_172990

theorem city_population_problem (p : ℝ) : 
  0.85 * (p + 800) = p + 824 ↔ p = 960 :=
by sorry

end city_population_problem_l1729_172990


namespace mikes_basketball_games_l1729_172954

theorem mikes_basketball_games (points_per_game : ℕ) (total_points : ℕ) (h1 : points_per_game = 4) (h2 : total_points = 24) :
  total_points / points_per_game = 6 := by
  sorry

end mikes_basketball_games_l1729_172954


namespace comic_collection_problem_l1729_172924

/-- The number of comic books that are in either Andrew's or John's collection, but not both -/
def exclusive_comics (shared comics_andrew comics_john_exclusive : ℕ) : ℕ :=
  (comics_andrew - shared) + comics_john_exclusive

theorem comic_collection_problem (shared comics_andrew comics_john_exclusive : ℕ) 
  (h1 : shared = 15)
  (h2 : comics_andrew = 22)
  (h3 : comics_john_exclusive = 10) :
  exclusive_comics shared comics_andrew comics_john_exclusive = 17 := by
  sorry

end comic_collection_problem_l1729_172924


namespace log_100_base_10_l1729_172910

theorem log_100_base_10 : Real.log 100 / Real.log 10 = 2 := by sorry

end log_100_base_10_l1729_172910


namespace max_odd_sums_for_given_range_l1729_172989

def max_odd_sums (n : ℕ) (start : ℕ) : ℕ :=
  if n ≤ 2 then 0
  else ((n - 2) / 2) + 1

theorem max_odd_sums_for_given_range :
  max_odd_sums 998 1000 = 499 := by
  sorry

end max_odd_sums_for_given_range_l1729_172989


namespace quadratic_function_ellipse_l1729_172944

/-- Given a quadratic function y = ax^2 + bx + c where ac ≠ 0,
    with vertex (-b/(2a), -1/(4a)),
    and intersections with x-axis on opposite sides of y-axis,
    prove that (b, c) lies on the ellipse b^2 + c^2/4 = 1 --/
theorem quadratic_function_ellipse (a b c : ℝ) (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (∃ (p q : ℝ), p < 0 ∧ q > 0 ∧ a * p^2 + b * p + c = 0 ∧ a * q^2 + b * q + c = 0) →
  (∃ (m : ℝ), m = -4 ∧ (b / (2 * a))^2 + m^2 = ((b^2 - 4 * a * c) / (4 * a^2))) →
  b^2 + c^2 / 4 = 1 :=
by sorry

end quadratic_function_ellipse_l1729_172944


namespace geometric_sequence_terms_l1729_172975

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem geometric_sequence_terms
  (a₁ : ℚ) (a₂ : ℚ) (h₁ : a₁ = 4) (h₂ : a₂ = 16/3) :
  let r := a₂ / a₁
  (geometric_sequence a₁ r 10 = 1048576/19683) ∧
  (geometric_sequence a₁ r 5 = 1024/81) := by
sorry

end geometric_sequence_terms_l1729_172975


namespace problem_statement_l1729_172993

theorem problem_statement : 5 * 301 + 4 * 301 + 3 * 301 + 300 = 3912 := by
  sorry

end problem_statement_l1729_172993


namespace arithmetic_calculation_l1729_172998

theorem arithmetic_calculation : 2^3 + 4 * 5 - 6 + 7 = 29 := by
  sorry

end arithmetic_calculation_l1729_172998


namespace soldiers_per_tower_l1729_172934

/-- Proves that the number of soldiers in each tower is 2 -/
theorem soldiers_per_tower (wall_length : ℕ) (tower_interval : ℕ) (total_soldiers : ℕ)
  (h1 : wall_length = 7300)
  (h2 : tower_interval = 5)
  (h3 : total_soldiers = 2920) :
  total_soldiers / (wall_length / tower_interval) = 2 := by
  sorry

end soldiers_per_tower_l1729_172934


namespace quadratic_roots_property_l1729_172907

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 + 2*m - 5 = 0 → n^2 + 2*n - 5 = 0 → m^2 + m*n + 2*m = 0 := by
  sorry

end quadratic_roots_property_l1729_172907


namespace parabola_translation_l1729_172931

/-- Given a parabola y = -2x^2, prove that translating it upwards by 1 unit
    and to the right by 2 units results in the equation y = -2(x-2)^2 + 1 -/
theorem parabola_translation (x y : ℝ) :
  (y = -2 * x^2) →
  (y + 1 = -2 * ((x - 2)^2) + 1) :=
by sorry

end parabola_translation_l1729_172931


namespace calculate_gross_profit_l1729_172949

/-- Calculate the gross profit given the sales price, gross profit margin, sales tax, and initial discount --/
theorem calculate_gross_profit (sales_price : ℝ) (gross_profit_margin : ℝ) (sales_tax : ℝ) (initial_discount : ℝ) :
  sales_price = 81 →
  gross_profit_margin = 1.7 →
  sales_tax = 0.07 →
  initial_discount = 0.15 →
  ∃ (gross_profit : ℝ), abs (gross_profit - 56.07) < 0.01 := by
  sorry


end calculate_gross_profit_l1729_172949


namespace ryan_english_hours_l1729_172966

/-- Ryan's learning schedule --/
structure LearningSchedule where
  days : ℕ
  english_total : ℕ
  chinese_daily : ℕ

/-- Calculates the daily hours spent on learning English --/
def daily_english_hours (schedule : LearningSchedule) : ℚ :=
  schedule.english_total / schedule.days

/-- Theorem stating Ryan's daily English learning hours --/
theorem ryan_english_hours (schedule : LearningSchedule) 
  (h1 : schedule.days = 2)
  (h2 : schedule.english_total = 12)
  (h3 : schedule.chinese_daily = 5) :
  daily_english_hours schedule = 6 := by
  sorry


end ryan_english_hours_l1729_172966


namespace equation_root_implies_m_value_l1729_172965

theorem equation_root_implies_m_value (m x : ℝ) : 
  (m / (x - 3) - 1 / (3 - x) = 2) → 
  (∃ x > 0, m / (x - 3) - 1 / (3 - x) = 2) → 
  m = -1 := by
sorry

end equation_root_implies_m_value_l1729_172965


namespace minimum_buses_l1729_172942

theorem minimum_buses (max_capacity : ℕ) (total_students : ℕ) (h1 : max_capacity = 45) (h2 : total_students = 495) :
  ∃ n : ℕ, n * max_capacity ≥ total_students ∧ ∀ m : ℕ, m * max_capacity ≥ total_students → n ≤ m :=
by
  sorry

end minimum_buses_l1729_172942


namespace solution_set_max_t_value_l1729_172987

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 3|

-- Theorem for the solution of f(x) < 5
theorem solution_set (x : ℝ) : f x < 5 ↔ -7/4 < x ∧ x < 3/4 := by sorry

-- Theorem for the maximum value of t
theorem max_t_value : ∃ (a : ℝ), a = 4 ∧ ∀ (t : ℝ), (∀ (x : ℝ), f x - t ≥ 0) ↔ t ≤ a := by sorry

end solution_set_max_t_value_l1729_172987


namespace segment_distance_sum_l1729_172901

/-- Represents a line segment with a midpoint -/
structure Segment where
  length : ℝ
  midpoint : ℝ

/-- The function relating distances from midpoints -/
def distance_relation (x y : ℝ) : Prop := y / x = 5 / 3

theorem segment_distance_sum 
  (ab : Segment) 
  (a'b' : Segment) 
  (h1 : ab.length = 3) 
  (h2 : a'b'.length = 5) 
  (x : ℝ) 
  (y : ℝ) 
  (h3 : distance_relation x y) 
  (h4 : x = 2) : 
  x + y = 16 / 3 := by
  sorry

end segment_distance_sum_l1729_172901


namespace negative_square_and_subtraction_l1729_172943

theorem negative_square_and_subtraction :
  (-4^2 = -16) ∧ ((-3) - (-6) = 3) := by sorry

end negative_square_and_subtraction_l1729_172943


namespace tomato_production_relationship_l1729_172978

/-- Represents the production of three tomato plants -/
structure TomatoProduction where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the tomato production problem -/
def tomato_problem (p : TomatoProduction) : Prop :=
  p.first = 24 ∧
  p.second = (p.first / 2) + 5 ∧
  p.first + p.second + p.third = 60

theorem tomato_production_relationship (p : TomatoProduction) 
  (h : tomato_problem p) : p.third = p.second + 2 := by
  sorry

#check tomato_production_relationship

end tomato_production_relationship_l1729_172978


namespace surf_festival_attendance_l1729_172912

/-- The number of additional surfers on the second day of the Rip Curl Myrtle Beach Surf Festival --/
def additional_surfers : ℕ := 600

theorem surf_festival_attendance :
  let first_day : ℕ := 1500
  let third_day : ℕ := (2 : ℕ) * first_day / (5 : ℕ)
  let total_surfers : ℕ := first_day + (first_day + additional_surfers) + third_day
  let average_surfers : ℕ := 1400
  total_surfers / 3 = average_surfers :=
by sorry

end surf_festival_attendance_l1729_172912


namespace remaining_money_for_gas_and_maintenance_l1729_172952

def monthly_income : ℕ := 3200

def rent : ℕ := 1250
def utilities : ℕ := 150
def retirement_savings : ℕ := 400
def groceries : ℕ := 300
def insurance : ℕ := 200
def miscellaneous : ℕ := 200
def car_payment : ℕ := 350

def total_expenses : ℕ := rent + utilities + retirement_savings + groceries + insurance + miscellaneous + car_payment

theorem remaining_money_for_gas_and_maintenance :
  monthly_income - total_expenses = 350 := by sorry

end remaining_money_for_gas_and_maintenance_l1729_172952


namespace triangle_angle_problem_l1729_172921

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)
  (sum_to_180 : a + b + c = 180)

-- Define the problem
theorem triangle_angle_problem (t : Triangle) 
  (h1 : t.a = 70)
  (h2 : t.b = 40)
  (h3 : 180 - t.c = 130) :
  t.c = 60 := by
  sorry

end triangle_angle_problem_l1729_172921


namespace cube_of_negative_two_ab_squared_l1729_172964

theorem cube_of_negative_two_ab_squared (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end cube_of_negative_two_ab_squared_l1729_172964


namespace first_year_growth_rate_l1729_172985

/-- Proves that given the initial and final populations, and the second year's growth rate,
    the first year's growth rate is 22%. -/
theorem first_year_growth_rate (initial_pop : ℕ) (final_pop : ℕ) (second_year_rate : ℚ) :
  initial_pop = 800 →
  final_pop = 1220 →
  second_year_rate = 25 / 100 →
  ∃ (first_year_rate : ℚ),
    first_year_rate = 22 / 100 ∧
    final_pop = initial_pop * (1 + first_year_rate) * (1 + second_year_rate) :=
by sorry

end first_year_growth_rate_l1729_172985


namespace tip_percentage_is_twenty_percent_l1729_172976

/-- Calculates the tip percentage given the total bill, food price, and sales tax rate. -/
def calculate_tip_percentage (total_bill : ℚ) (food_price : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let sales_tax := food_price * sales_tax_rate
  let total_before_tip := food_price + sales_tax
  let tip_amount := total_bill - total_before_tip
  (tip_amount / total_before_tip) * 100

/-- Proves that the tip percentage is 20% given the specified conditions. -/
theorem tip_percentage_is_twenty_percent :
  calculate_tip_percentage 158.40 120 0.10 = 20 := by
  sorry

end tip_percentage_is_twenty_percent_l1729_172976


namespace a_greater_than_b_squared_l1729_172986

theorem a_greater_than_b_squared {a b : ℝ} (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 := by
  sorry

end a_greater_than_b_squared_l1729_172986


namespace reflection_of_circle_center_l1729_172995

/-- Reflects a point about the line y = -x -/
def reflect_about_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (9, -4)
  let reflected_center := reflect_about_y_neg_x original_center
  reflected_center = (4, -9) := by sorry

end reflection_of_circle_center_l1729_172995


namespace completing_square_equivalence_l1729_172955

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 2*x - 5 = 0) ↔ ((x + 1)^2 = 6) := by
  sorry

end completing_square_equivalence_l1729_172955


namespace range_of_sum_on_circle_l1729_172950

theorem range_of_sum_on_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 3 = 0) :
  ∃ (min max : ℝ), min = 2 - Real.sqrt 2 ∧ max = 2 + Real.sqrt 2 ∧
  min ≤ x + y ∧ x + y ≤ max :=
sorry

end range_of_sum_on_circle_l1729_172950


namespace daniels_painting_area_l1729_172962

/-- Calculate the total paintable wall area for multiple bedrooms -/
def total_paintable_area (num_rooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_rooms * paintable_area

/-- The problem statement -/
theorem daniels_painting_area :
  total_paintable_area 4 15 12 9 80 = 1624 := by
  sorry

end daniels_painting_area_l1729_172962


namespace prob_sum_div_3_correct_l1729_172953

/-- The probability of the sum of three fair six-sided dice being divisible by 3 -/
def prob_sum_div_3 : ℚ :=
  13 / 27

/-- The set of possible outcomes for a single die roll -/
def die_outcomes : Finset ℕ :=
  Finset.range 6

/-- The probability of rolling any specific number on a fair six-sided die -/
def single_roll_prob : ℚ :=
  1 / 6

/-- The set of all possible outcomes when rolling three dice -/
def all_outcomes : Finset (ℕ × ℕ × ℕ) :=
  die_outcomes.product (die_outcomes.product die_outcomes)

/-- The sum of the numbers shown on three dice -/
def sum_of_dice (roll : ℕ × ℕ × ℕ) : ℕ :=
  roll.1 + roll.2.1 + roll.2.2 + 3

/-- The set of favorable outcomes (sum divisible by 3) -/
def favorable_outcomes : Finset (ℕ × ℕ × ℕ) :=
  all_outcomes.filter (λ roll ↦ (sum_of_dice roll) % 3 = 0)

theorem prob_sum_div_3_correct :
  (favorable_outcomes.card : ℚ) / all_outcomes.card = prob_sum_div_3 := by
  sorry

end prob_sum_div_3_correct_l1729_172953


namespace sqrt_equation_solution_l1729_172918

theorem sqrt_equation_solution :
  ∃ y : ℝ, (Real.sqrt (4 - 2 * y) = 9) ∧ (y = -38.5) := by
  sorry

end sqrt_equation_solution_l1729_172918


namespace alan_roof_weight_l1729_172932

/-- The number of pine trees in Alan's backyard -/
def num_trees : ℕ := 8

/-- The number of pine cones each tree drops -/
def cones_per_tree : ℕ := 200

/-- The percentage of pine cones that fall on the roof -/
def roof_percentage : ℚ := 30 / 100

/-- The weight of each pine cone in ounces -/
def cone_weight : ℚ := 4

/-- The total weight of pine cones on Alan's roof in ounces -/
def roof_weight : ℚ := num_trees * cones_per_tree * roof_percentage * cone_weight

theorem alan_roof_weight : roof_weight = 1920 := by sorry

end alan_roof_weight_l1729_172932


namespace special_polynomial_value_l1729_172951

/-- A polynomial of the form ± x^6 ± x^5 ± x^4 ± x^3 ± x^2 ± x ± 1 -/
def SpecialPolynomial (P : ℝ → ℝ) : Prop :=
  ∃ (a b c d e f g : ℤ), ∀ x,
    P x = a * x^6 + b * x^5 + c * x^4 + d * x^3 + e * x^2 + f * x + g ∧
    (a = 1 ∨ a = -1) ∧ (b = 1 ∨ b = -1) ∧ (c = 1 ∨ c = -1) ∧
    (d = 1 ∨ d = -1) ∧ (e = 1 ∨ e = -1) ∧ (f = 1 ∨ f = -1) ∧ (g = 1 ∨ g = -1)

theorem special_polynomial_value (P : ℝ → ℝ) :
  SpecialPolynomial P → P 2 = 27 → P 3 = 439 := by
  sorry

end special_polynomial_value_l1729_172951


namespace intersection_of_A_and_B_union_of_A_and_B_range_of_p_l1729_172925

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}
def C (p : ℝ) : Set ℝ := {x | 4*x + p < 0}

-- State the theorems
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} := by sorry

theorem union_of_A_and_B : A ∪ B = Set.univ := by sorry

theorem range_of_p (p : ℝ) : C p ⊆ A → p ≥ 4 := by sorry

end intersection_of_A_and_B_union_of_A_and_B_range_of_p_l1729_172925


namespace max_profit_and_volume_l1729_172961

/-- Represents the annual profit function for a company producing a certain product. -/
noncomputable def annual_profit (x : ℝ) : ℝ :=
  if x < 80 then
    -(1/360) * x^3 + 30*x - 250
  else
    1200 - (x + 10000/x)

/-- Theorem stating the maximum annual profit and the production volume that achieves it. -/
theorem max_profit_and_volume :
  (∃ (max_profit : ℝ) (optimal_volume : ℝ),
    max_profit = 1000 ∧
    optimal_volume = 100 ∧
    ∀ x, x > 0 → annual_profit x ≤ max_profit ∧
    annual_profit optimal_volume = max_profit) :=
sorry


end max_profit_and_volume_l1729_172961


namespace power_tower_mod_500_l1729_172936

theorem power_tower_mod_500 : 7^(7^(7^7)) ≡ 343 [ZMOD 500] := by sorry

end power_tower_mod_500_l1729_172936


namespace cu_atom_count_l1729_172988

/-- Represents the number of atoms of an element in a compound -/
structure AtomCount where
  count : ℕ

/-- Represents the atomic weight of an element in amu -/
structure AtomicWeight where
  weight : ℝ

/-- Represents a chemical compound -/
structure Compound where
  cu : AtomCount
  c : AtomCount
  o : AtomCount
  molecularWeight : ℝ

def cuAtomicWeight : AtomicWeight := ⟨63.55⟩
def cAtomicWeight : AtomicWeight := ⟨12.01⟩
def oAtomicWeight : AtomicWeight := ⟨16.00⟩

def compoundWeight (cpd : Compound) : ℝ :=
  cpd.cu.count * cuAtomicWeight.weight +
  cpd.c.count * cAtomicWeight.weight +
  cpd.o.count * oAtomicWeight.weight

theorem cu_atom_count (cpd : Compound) :
  cpd.c = ⟨1⟩ ∧ cpd.o = ⟨3⟩ ∧ cpd.molecularWeight = 124 →
  cpd.cu = ⟨1⟩ :=
by sorry

end cu_atom_count_l1729_172988
