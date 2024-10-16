import Mathlib

namespace NUMINAMATH_CALUDE_correct_counting_error_l1103_110373

/-- The error in cents to be subtracted when quarters are mistakenly counted as half dollars
    and nickels are mistakenly counted as dimes. -/
def counting_error (x y : ℕ) : ℕ := 25 * x + 5 * y

/-- The value of a quarter in cents. -/
def quarter_value : ℕ := 25

/-- The value of a half dollar in cents. -/
def half_dollar_value : ℕ := 50

/-- The value of a nickel in cents. -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents. -/
def dime_value : ℕ := 10

theorem correct_counting_error (x y : ℕ) :
  counting_error x y = (half_dollar_value - quarter_value) * x + (dime_value - nickel_value) * y :=
by sorry

end NUMINAMATH_CALUDE_correct_counting_error_l1103_110373


namespace NUMINAMATH_CALUDE_max_profit_l1103_110317

noncomputable def fixed_cost : ℝ := 14000
noncomputable def variable_cost : ℝ := 210

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then (1 / 625) * x^2
  else 256

noncomputable def g (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then -(5 / 8) * x + 750
  else 500

noncomputable def c (x : ℝ) : ℝ := fixed_cost + variable_cost * x

noncomputable def Q (x : ℝ) : ℝ := f x * g x - c x

theorem max_profit (x : ℝ) : Q x ≤ 30000 ∧ Q 400 = 30000 := by sorry

end NUMINAMATH_CALUDE_max_profit_l1103_110317


namespace NUMINAMATH_CALUDE_quadratic_function_constraint_l1103_110337

/-- Given a quadratic function f(x) = ax^2 + bx + c where a ≠ 0,
    if f(-1) = 0 and x ≤ f(x) ≤ (1/2)(x^2 + 1) for all x ∈ ℝ,
    then a = 1/4 -/
theorem quadratic_function_constraint (a b c : ℝ) (ha : a ≠ 0) :
  let f := fun (x : ℝ) => a * x^2 + b * x + c
  (f (-1) = 0) →
  (∀ x : ℝ, x ≤ f x ∧ f x ≤ (1/2) * (x^2 + 1)) →
  a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_constraint_l1103_110337


namespace NUMINAMATH_CALUDE_inequality_implication_l1103_110394

theorem inequality_implication (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) :
  y * (y - 1) ≤ x^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1103_110394


namespace NUMINAMATH_CALUDE_lisa_marbles_problem_l1103_110350

def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := (num_friends * (num_friends + 1)) / 2
  if required_marbles > initial_marbles then
    required_marbles - initial_marbles
  else
    0

theorem lisa_marbles_problem (num_friends : ℕ) (initial_marbles : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_marbles = 60) :
  min_additional_marbles num_friends initial_marbles = 60 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_problem_l1103_110350


namespace NUMINAMATH_CALUDE_ninas_pet_insects_eyes_l1103_110339

/-- The total number of eyes among Nina's pet insects -/
def total_eyes (num_spiders num_ants spider_eyes ant_eyes : ℕ) : ℕ :=
  num_spiders * spider_eyes + num_ants * ant_eyes

/-- Theorem stating that the total number of eyes among Nina's pet insects is 124 -/
theorem ninas_pet_insects_eyes :
  total_eyes 3 50 8 2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_ninas_pet_insects_eyes_l1103_110339


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l1103_110314

def hyperbola_equation (x y : ℝ) : Prop := x^2 / 9 - y^2 / 7 = 1

def is_focus (F : ℝ × ℝ) (C : (ℝ × ℝ → Prop)) : Prop := sorry

def is_right_branch (P : ℝ × ℝ) (C : (ℝ × ℝ → Prop)) : Prop := sorry

def distance (P Q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_triangle_perimeter 
  (C : ℝ × ℝ → Prop)
  (F₁ F₂ P : ℝ × ℝ) :
  (∀ x y, C (x, y) ↔ hyperbola_equation x y) →
  is_focus F₁ C ∧ is_focus F₂ C →
  F₁.1 < F₂.1 →
  is_right_branch P C →
  distance P F₁ = 8 →
  distance P F₁ + distance P F₂ + distance F₁ F₂ = 18 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l1103_110314


namespace NUMINAMATH_CALUDE_sqrt_2_plus_sqrt_2_plus_l1103_110310

theorem sqrt_2_plus_sqrt_2_plus : ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (2 + x) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_plus_sqrt_2_plus_l1103_110310


namespace NUMINAMATH_CALUDE_xy_value_l1103_110318

theorem xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x / y = 81) (h4 : y = 0.2222222222222222) :
  x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1103_110318


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1103_110379

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 3) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term (a : ℚ) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → S a (3 * n) / S a n = c) →
  a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1103_110379


namespace NUMINAMATH_CALUDE_equation_solution_l1103_110307

theorem equation_solution : 
  ∀ t : ℝ, t ≠ 6 ∧ t ≠ -4 →
  ((t^2 - 3*t - 18) / (t - 6) = 2 / (t + 4)) ↔ (t = -2 ∨ t = -5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1103_110307


namespace NUMINAMATH_CALUDE_cube_surface_area_l1103_110377

/-- The surface area of a cube with edge length 2a is 24a² -/
theorem cube_surface_area (a : ℝ) : 
  6 * (2 * a)^2 = 24 * a^2 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1103_110377


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l1103_110344

theorem sqrt_mixed_number_simplification :
  Real.sqrt (12 + 9/16) = Real.sqrt 201 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l1103_110344


namespace NUMINAMATH_CALUDE_smallest_bound_for_cubic_inequality_l1103_110389

theorem smallest_bound_for_cubic_inequality :
  ∃ (M : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧
  (∀ (M' : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) →
    M ≤ M') ∧
  M = (9 * Real.sqrt 2) / 32 :=
by sorry

end NUMINAMATH_CALUDE_smallest_bound_for_cubic_inequality_l1103_110389


namespace NUMINAMATH_CALUDE_systematic_sample_sequence_l1103_110362

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  first_number : Nat

/-- Calculates the next numbers in a systematic sample sequence -/
def next_numbers (s : SystematicSample) : List Nat :=
  let step := s.total_students / s.sample_size
  [1, 2, 3, 4].map (fun i => s.first_number + i * step)

theorem systematic_sample_sequence (s : SystematicSample) 
  (h1 : s.total_students = 60)
  (h2 : s.sample_size = 5)
  (h3 : s.first_number = 4) :
  next_numbers s = [16, 28, 40, 52] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_sequence_l1103_110362


namespace NUMINAMATH_CALUDE_solve_equation_l1103_110319

theorem solve_equation : ∃! x : ℝ, 3 * x - 2 * (10 - x) = 5 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l1103_110319


namespace NUMINAMATH_CALUDE_floor_equation_unique_solution_l1103_110387

theorem floor_equation_unique_solution (n : ℤ) :
  (Int.floor (n^2 / 4) - Int.floor (n / 2)^2 = 2) ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_unique_solution_l1103_110387


namespace NUMINAMATH_CALUDE_set_operations_l1103_110399

def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

theorem set_operations :
  (A ∪ (B ∩ C) = A) ∧
  (A ∩ (A \ (B ∩ C)) = {x | x ∈ A ∧ x ≠ 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1103_110399


namespace NUMINAMATH_CALUDE_girls_in_college_l1103_110390

theorem girls_in_college (total_students : ℕ) (boys_ratio girls_ratio : ℕ) : 
  total_students = 546 →
  boys_ratio = 8 →
  girls_ratio = 5 →
  ∃ (num_girls : ℕ), num_girls = 210 ∧ 
    boys_ratio * num_girls + girls_ratio * num_girls = girls_ratio * total_students :=
by
  sorry

end NUMINAMATH_CALUDE_girls_in_college_l1103_110390


namespace NUMINAMATH_CALUDE_find_point_B_l1103_110383

/-- Given vector a, point A, and a line y = 2x, find point B on the line such that AB is parallel to a -/
theorem find_point_B (a : ℝ × ℝ) (A : ℝ × ℝ) :
  a = (1, 1) →
  A = (-3, -1) →
  ∃ B : ℝ × ℝ,
    B.2 = 2 * B.1 ∧
    ∃ k : ℝ, k • a = (B.1 - A.1, B.2 - A.2) ∧
    B = (2, 4) := by
  sorry


end NUMINAMATH_CALUDE_find_point_B_l1103_110383


namespace NUMINAMATH_CALUDE_sum_of_p_and_q_l1103_110372

theorem sum_of_p_and_q (p q : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ y : ℝ, 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 14)) →
  p + q = 69 := by
sorry

end NUMINAMATH_CALUDE_sum_of_p_and_q_l1103_110372


namespace NUMINAMATH_CALUDE_triangle_roots_range_l1103_110342

theorem triangle_roots_range (m : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁ - 2) * (x₁^2 - 4*x₁ + m) = 0 ∧
    (x₂ - 2) * (x₂^2 - 4*x₂ + m) = 0 ∧
    (x₃ - 2) * (x₃^2 - 4*x₃ + m) = 0 ∧
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    x₁ + x₂ > x₃ ∧ x₂ + x₃ > x₁ ∧ x₃ + x₁ > x₂) →
  3 < m ∧ m < 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_roots_range_l1103_110342


namespace NUMINAMATH_CALUDE_arrangement_count_l1103_110392

/-- The number of distinct arrangements of 9 indistinguishable objects and 3 indistinguishable objects in a row of 12 positions -/
def distinct_arrangements : ℕ := 220

/-- The total number of positions -/
def total_positions : ℕ := 12

/-- The number of indistinguishable objects of the first type (armchairs) -/
def first_object_count : ℕ := 9

/-- The number of indistinguishable objects of the second type (benches) -/
def second_object_count : ℕ := 3

theorem arrangement_count :
  distinct_arrangements = (total_positions.choose second_object_count) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l1103_110392


namespace NUMINAMATH_CALUDE_odd_function_properties_l1103_110334

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_shift : ∀ x, f (x - 2) = -f x) : 
  f 2 = 0 ∧ has_period f 4 ∧ ∀ x, f (x + 2) = f (-x) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_l1103_110334


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l1103_110386

/-- Given points A, B, C, and F as the midpoint of AC, prove that the sum of the slope
    and y-intercept of the line passing through F and B is 3/4 -/
theorem slope_intercept_sum (A B C F : ℝ × ℝ) : 
  A = (0, 6) →
  B = (0, 0) →
  C = (8, 0) →
  F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  let m := (F.2 - B.2) / (F.1 - B.1)
  let b := B.2
  m + b = 3/4 := by sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l1103_110386


namespace NUMINAMATH_CALUDE_hulk_jump_theorem_l1103_110323

def jump_distance (n : ℕ) : ℝ :=
  2 * (3 ^ (n - 1))

theorem hulk_jump_theorem :
  (∀ k < 8, jump_distance k ≤ 2000) ∧ jump_distance 8 > 2000 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jump_theorem_l1103_110323


namespace NUMINAMATH_CALUDE_stratified_sampling_proof_l1103_110330

/-- Represents the total population -/
def total_population : ℕ := 27 + 54 + 81

/-- Represents the number of elderly people in the population -/
def elderly_population : ℕ := 27

/-- Represents the number of elderly people in the sample -/
def elderly_sample : ℕ := 3

/-- Represents the total sample size -/
def sample_size : ℕ := 18

/-- Proves that the given sample size is correct for the stratified sampling -/
theorem stratified_sampling_proof :
  (elderly_sample : ℚ) / elderly_population = sample_size / total_population :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_proof_l1103_110330


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1103_110331

/-- A quadratic function with vertex (h, k) has the form f(x) = a(x-h)^2 + k -/
def quadratic_vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

theorem parabola_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = quadratic_vertex_form a 2 5 x) →  -- Condition 2: vertex at (2, 5)
  f 3 = 7 →  -- Condition 3: point (3, 7) lies on the graph
  a = 2 := by  -- Question: Find the value of a
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1103_110331


namespace NUMINAMATH_CALUDE_marcos_strawberries_weight_l1103_110382

theorem marcos_strawberries_weight (total_weight dad_weight : ℝ) 
  (h1 : total_weight = 20)
  (h2 : dad_weight = 17) :
  total_weight - dad_weight = 3 := by
sorry

end NUMINAMATH_CALUDE_marcos_strawberries_weight_l1103_110382


namespace NUMINAMATH_CALUDE_inequality_proof_equality_conditions_l1103_110381

theorem inequality_proof (x y : ℝ) (h1 : x ≥ y) (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) ≥
  (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1)) :=
sorry

theorem equality_conditions (x y : ℝ) (h1 : x ≥ y) (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) =
  (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1)) ↔
  (x = y ∨ x = 1 ∨ y = 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_conditions_l1103_110381


namespace NUMINAMATH_CALUDE_M_equals_interval_inequality_holds_l1103_110393

def f (x : ℝ) := |x + 2| + |x - 2|

def M : Set ℝ := {x | f x ≤ 6}

theorem M_equals_interval : M = Set.Icc (-3) 3 := by sorry

theorem inequality_holds (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  Real.sqrt 3 * |a + b| ≤ |a * b + 3| := by sorry

end NUMINAMATH_CALUDE_M_equals_interval_inequality_holds_l1103_110393


namespace NUMINAMATH_CALUDE_fraction_to_whole_number_l1103_110395

theorem fraction_to_whole_number : 
  (∃ n : ℤ, (12 : ℚ) / 2 = n) ∧
  (∀ n : ℤ, (8 : ℚ) / 6 ≠ n) ∧
  (∀ n : ℤ, (9 : ℚ) / 5 ≠ n) ∧
  (∀ n : ℤ, (10 : ℚ) / 4 ≠ n) ∧
  (∀ n : ℤ, (11 : ℚ) / 3 ≠ n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_whole_number_l1103_110395


namespace NUMINAMATH_CALUDE_rachel_total_books_l1103_110388

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 6

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 2

/-- The total number of books Rachel has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem rachel_total_books : total_books = 72 := by
  sorry

end NUMINAMATH_CALUDE_rachel_total_books_l1103_110388


namespace NUMINAMATH_CALUDE_sum_in_base6_l1103_110300

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

theorem sum_in_base6 :
  let a := base6ToDecimal [5, 5, 5, 1]
  let b := base6ToDecimal [5, 5, 1]
  let c := base6ToDecimal [5, 1]
  decimalToBase6 (a + b + c) = [2, 2, 0, 3] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base6_l1103_110300


namespace NUMINAMATH_CALUDE_area_of_specific_isosceles_triangle_l1103_110311

/-- An isosceles triangle with given heights -/
structure IsoscelesTriangle where
  /-- Height dropped to the base -/
  baseHeight : ℝ
  /-- Height dropped to the lateral side -/
  lateralHeight : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : True

/-- Calculate the area of an isosceles triangle given its heights -/
def areaOfIsoscelesTriangle (triangle : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of the specific isosceles triangle is 75 -/
theorem area_of_specific_isosceles_triangle :
  let triangle : IsoscelesTriangle := {
    baseHeight := 10,
    lateralHeight := 12,
    isIsosceles := True.intro
  }
  areaOfIsoscelesTriangle triangle = 75 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_isosceles_triangle_l1103_110311


namespace NUMINAMATH_CALUDE_stating_club_officer_selection_count_l1103_110378

/-- Represents the number of members in the club -/
def total_members : ℕ := 24

/-- Represents the number of boys in the club -/
def num_boys : ℕ := 12

/-- Represents the number of girls in the club -/
def num_girls : ℕ := 12

/-- Represents the number of offices to be filled -/
def num_offices : ℕ := 3

/-- 
Theorem stating that the number of ways to choose a president, vice-president, and secretary 
from a club of 24 members (12 boys and 12 girls) is 5808, given that the president and 
vice-president must be of the same gender, the secretary can be of any gender, and no one 
can hold more than one office.
-/
theorem club_officer_selection_count : 
  (num_boys * (num_boys - 1) + num_girls * (num_girls - 1)) * (total_members - 2) = 5808 := by
  sorry

end NUMINAMATH_CALUDE_stating_club_officer_selection_count_l1103_110378


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_39_squared_plus_52_squared_thirteen_is_prime_divisor_of_39_squared_plus_52_squared_largest_prime_divisor_is_13_l1103_110398

theorem largest_prime_divisor_of_39_squared_plus_52_squared (p : Nat) : 
  Nat.Prime p ∧ p ∣ (39^2 + 52^2) → p ≤ 13 :=
by sorry

theorem thirteen_is_prime_divisor_of_39_squared_plus_52_squared : 
  Nat.Prime 13 ∧ 13 ∣ (39^2 + 52^2) :=
by sorry

theorem largest_prime_divisor_is_13 : 
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ (39^2 + 52^2) ∧ 
  ∀ (q : Nat), Nat.Prime q ∧ q ∣ (39^2 + 52^2) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_39_squared_plus_52_squared_thirteen_is_prime_divisor_of_39_squared_plus_52_squared_largest_prime_divisor_is_13_l1103_110398


namespace NUMINAMATH_CALUDE_equal_digit_probability_l1103_110327

/-- The number of sides on each die -/
def sides : ℕ := 15

/-- The number of dice rolled -/
def dice_count : ℕ := 6

/-- The probability of rolling a one-digit number on a single die -/
def prob_one_digit : ℚ := 3/5

/-- The probability of rolling a two-digit number on a single die -/
def prob_two_digit : ℚ := 2/5

/-- The number of ways to choose 3 dice out of 6 -/
def ways_to_choose : ℕ := Nat.choose 6 3

/-- Theorem stating the probability of rolling an equal number of one-digit and two-digit numbers -/
theorem equal_digit_probability : 
  (ways_to_choose : ℚ) * (prob_one_digit ^ 3) * (prob_two_digit ^ 3) = 4320/15625 := by
  sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l1103_110327


namespace NUMINAMATH_CALUDE_exists_m_range_l1103_110374

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - a

-- Define the solution set condition
def solution_set (a : ℝ) (x₀ x₄ : ℝ) : Prop :=
  ∀ x, f a x ≤ 2 ↔ x₀ ≤ x ∧ x ≤ x₄

-- State the theorem
theorem exists_m_range (a : ℝ) (x₀ x₄ : ℝ) 
  (h : solution_set a x₀ x₄) :
  ∃ m₁ m₂ : ℝ, m₁ ≤ m₂ ∧ 
    ∀ x, -2 ≤ x ∧ x ≤ 4 → 
      ∃ m, m₁ ≤ m ∧ m ≤ m₂ ∧ -1 - f a (x + 1) ≤ m :=
sorry

end NUMINAMATH_CALUDE_exists_m_range_l1103_110374


namespace NUMINAMATH_CALUDE_cube_spheres_surface_area_ratio_l1103_110329

/-- The ratio of the surface area of a cube's inscribed sphere to its circumscribed sphere -/
theorem cube_spheres_surface_area_ratio (a : ℝ) (h : a > 0) : 
  (4 * Real.pi * (a / 2)^2) / (4 * Real.pi * (a * Real.sqrt 3 / 2)^2) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_cube_spheres_surface_area_ratio_l1103_110329


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_5_plus_10_pow_4_l1103_110349

theorem greatest_prime_factor_of_5_pow_5_plus_10_pow_4 :
  (Nat.factors (5^5 + 10^4)).maximum? = some 7 :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_5_plus_10_pow_4_l1103_110349


namespace NUMINAMATH_CALUDE_cereal_eating_time_l1103_110397

/-- The time required for two people to eat a certain amount of cereal together -/
def eating_time (quick_rate : ℚ) (slow_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (quick_rate + slow_rate)

/-- Theorem: Mr. Quick and Mr. Slow eat 5 pounds of cereal in 600/11 minutes -/
theorem cereal_eating_time :
  let quick_rate : ℚ := 1 / 15
  let slow_rate : ℚ := 1 / 40
  let total_amount : ℚ := 5
  eating_time quick_rate slow_rate total_amount = 600 / 11 := by
  sorry

#eval eating_time (1/15 : ℚ) (1/40 : ℚ) 5

end NUMINAMATH_CALUDE_cereal_eating_time_l1103_110397


namespace NUMINAMATH_CALUDE_seating_arrangement_theorem_l1103_110366

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def undesirable_arrangements (n k : ℕ) : ℕ := (Nat.factorial (n - k + 1)) * (Nat.factorial k)

def acceptable_arrangements (n k : ℕ) : ℕ := 
  (total_arrangements n) - (undesirable_arrangements n k)

theorem seating_arrangement_theorem :
  acceptable_arrangements 10 4 = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_theorem_l1103_110366


namespace NUMINAMATH_CALUDE_max_xy_given_sum_l1103_110367

theorem max_xy_given_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 18) :
  x * y ≤ 81 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 18 ∧ x * y = 81 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_given_sum_l1103_110367


namespace NUMINAMATH_CALUDE_xy_inequality_l1103_110351

theorem xy_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x*y = 3) :
  x + y ≥ 2 ∧ (x + y = 2 ↔ x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_l1103_110351


namespace NUMINAMATH_CALUDE_grandparents_count_l1103_110384

/-- Represents the amount of money each grandparent gave to John -/
def money_per_grandparent : ℕ := 50

/-- Represents the total amount of money John received -/
def total_money : ℕ := 100

/-- The number of grandparents who gave John money -/
def num_grandparents : ℕ := 2

/-- Theorem stating that the number of grandparents who gave John money is 2 -/
theorem grandparents_count :
  num_grandparents = 2 ∧ total_money = num_grandparents * money_per_grandparent :=
sorry

end NUMINAMATH_CALUDE_grandparents_count_l1103_110384


namespace NUMINAMATH_CALUDE_train_speed_l1103_110324

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 32) : 
  (train_length + bridge_length) / crossing_time = 12.5 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1103_110324


namespace NUMINAMATH_CALUDE_series_sum_after_removal_equals_neg_3026_l1103_110305

def series_sum (n : ℕ) : ℤ :=
  if n % 4 = 0 then
    (n - 3) - (n - 2) - (n - 1) + n
  else if n % 4 = 1 then
    n - (n + 1) - (n + 2)
  else
    0

def remove_multiples_of_10 (n : ℤ) : ℤ :=
  if n % 10 = 0 then 0 else n

def final_sum : ℤ :=
  (List.range 2015).foldl (λ acc i => acc + remove_multiples_of_10 (series_sum (i + 1))) 0

theorem series_sum_after_removal_equals_neg_3026 :
  final_sum = -3026 :=
sorry

end NUMINAMATH_CALUDE_series_sum_after_removal_equals_neg_3026_l1103_110305


namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l1103_110352

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l1103_110352


namespace NUMINAMATH_CALUDE_students_not_enrolled_in_french_or_german_l1103_110303

theorem students_not_enrolled_in_french_or_german 
  (total_students : ℕ) 
  (french_students : ℕ) 
  (german_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 78)
  (h2 : french_students = 41)
  (h3 : german_students = 22)
  (h4 : both_students = 9) :
  total_students - (french_students + german_students - both_students) = 24 :=
by sorry


end NUMINAMATH_CALUDE_students_not_enrolled_in_french_or_german_l1103_110303


namespace NUMINAMATH_CALUDE_unique_solution_diophantine_system_l1103_110321

theorem unique_solution_diophantine_system :
  ∀ a b c : ℕ,
  a^3 - b^3 - c^3 = 3*a*b*c →
  a^2 = 2*(b + c) →
  a = 2 ∧ b = 1 ∧ c = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_diophantine_system_l1103_110321


namespace NUMINAMATH_CALUDE_statue_weight_proof_l1103_110371

/-- Given a set of statues carved from a marble block, prove the weight of each remaining statue. -/
theorem statue_weight_proof (initial_weight discarded_weight first_statue_weight second_statue_weight : ℝ)
  (h1 : initial_weight = 80)
  (h2 : discarded_weight = 22)
  (h3 : first_statue_weight = 10)
  (h4 : second_statue_weight = 18)
  (h5 : initial_weight ≥ first_statue_weight + second_statue_weight + discarded_weight) :
  let remaining_weight := initial_weight - first_statue_weight - second_statue_weight - discarded_weight
  (remaining_weight / 2 : ℝ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_proof_l1103_110371


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1103_110368

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

theorem line_plane_perpendicularity 
  (a b : Line) (α : Plane) 
  (h1 : parallel a α) 
  (h2 : perpendicular b α) : 
  perpendicular_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1103_110368


namespace NUMINAMATH_CALUDE_circle_area_relationship_l1103_110353

theorem circle_area_relationship (A B : ℝ → ℝ → Prop) : 
  (∃ r : ℝ, (∀ x y : ℝ, A x y ↔ (x - r)^2 + (y - r)^2 = r^2) ∧ 
             (∀ x y : ℝ, B x y ↔ (x - 2*r)^2 + (y - 2*r)^2 = (2*r)^2)) →
  (π * r^2 = 16 * π) →
  (π * (2*r)^2 = 64 * π) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_relationship_l1103_110353


namespace NUMINAMATH_CALUDE_stratified_sampling_female_students_l1103_110364

/-- Calculates the number of female students selected in a stratified sampling -/
def female_students_selected (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * female_students) / total_students

/-- Theorem: In a school with 2000 total students and 800 female students,
    a stratified sampling of 50 students will select 20 female students -/
theorem stratified_sampling_female_students :
  female_students_selected 2000 800 50 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_students_l1103_110364


namespace NUMINAMATH_CALUDE_max_value_product_l1103_110346

theorem max_value_product (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + 2*y + 3*z = 1) :
  x^2 * y^2 * z ≤ 4/16807 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2*y + 3*z = 1 ∧ x^2 * y^2 * z = 4/16807 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l1103_110346


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1103_110360

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 - 3*I) / (1 - I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1103_110360


namespace NUMINAMATH_CALUDE_pieces_per_box_l1103_110325

/-- Given information about Adam's chocolate candy boxes -/
structure ChocolateBoxes where
  totalBought : ℕ
  givenAway : ℕ
  piecesLeft : ℕ

/-- Theorem stating the number of pieces in each box -/
theorem pieces_per_box (boxes : ChocolateBoxes)
  (h1 : boxes.totalBought = 13)
  (h2 : boxes.givenAway = 7)
  (h3 : boxes.piecesLeft = 36) :
  boxes.piecesLeft / (boxes.totalBought - boxes.givenAway) = 6 := by
  sorry


end NUMINAMATH_CALUDE_pieces_per_box_l1103_110325


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l1103_110301

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l1103_110301


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1103_110322

theorem sum_of_numbers (x y : ℤ) : y = 3 * x + 11 → x = 11 → x + y = 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1103_110322


namespace NUMINAMATH_CALUDE_benny_pie_price_l1103_110304

/-- Calculates the price per pie needed to achieve a desired profit given the number and cost of pumpkin and cherry pies -/
def price_per_pie (Np Nc : ℕ) (Cp Cc Pr : ℚ) : ℚ :=
  (Np * Cp + Nc * Cc + Pr) / (Np + Nc)

theorem benny_pie_price :
  let Np : ℕ := 10  -- Number of pumpkin pies
  let Nc : ℕ := 12  -- Number of cherry pies
  let Cp : ℚ := 3   -- Cost to make each pumpkin pie
  let Cc : ℚ := 5   -- Cost to make each cherry pie
  let Pr : ℚ := 20  -- Desired profit
  price_per_pie Np Nc Cp Cc Pr = 5 := by
sorry

end NUMINAMATH_CALUDE_benny_pie_price_l1103_110304


namespace NUMINAMATH_CALUDE_geometric_sequence_differences_l1103_110370

/-- The type of sequences of real numbers of length n -/
def RealSequence (n : ℕ) := Fin n → ℝ

/-- The condition that a sequence is strictly increasing -/
def StrictlyIncreasing {n : ℕ} (a : RealSequence n) : Prop :=
  ∀ i j : Fin n, i < j → a i < a j

/-- The set of differences between elements of a sequence -/
def Differences {n : ℕ} (a : RealSequence n) : Set ℝ :=
  {x : ℝ | ∃ i j : Fin n, i < j ∧ x = a j - a i}

/-- The set of powers of r from 1 to k -/
def PowerSet (r : ℝ) (k : ℕ) : Set ℝ :=
  {x : ℝ | ∃ m : ℕ, m ≤ k ∧ x = r ^ m}

/-- The main theorem -/
theorem geometric_sequence_differences (n : ℕ) (h : n ≥ 2) :
  (∃ (a : RealSequence n) (r : ℝ),
    StrictlyIncreasing a ∧
    r > 0 ∧
    Differences a = PowerSet r (n * (n - 1) / 2)) ↔
  n = 2 ∨ n = 3 ∨ n = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_differences_l1103_110370


namespace NUMINAMATH_CALUDE_sphere_only_circular_views_l1103_110355

-- Define the geometric shapes
inductive Shape
  | Cuboid
  | Cylinder
  | Cone
  | Sphere

-- Define the views
inductive View
  | Front
  | Left
  | Top

-- Define a function to check if a view is circular for a given shape
def isCircularView (s : Shape) (v : View) : Prop :=
  match s, v with
  | Shape.Sphere, _ => True
  | Shape.Cylinder, View.Top => True
  | Shape.Cone, View.Top => True
  | _, _ => False

-- Define a function to check if all views are circular for a given shape
def allViewsCircular (s : Shape) : Prop :=
  isCircularView s View.Front ∧ isCircularView s View.Left ∧ isCircularView s View.Top

-- Theorem statement
theorem sphere_only_circular_views :
  ∀ s : Shape, allViewsCircular s ↔ s = Shape.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_only_circular_views_l1103_110355


namespace NUMINAMATH_CALUDE_license_plate_count_l1103_110396

/-- The number of digits in a license plate -/
def num_digits : ℕ := 5

/-- The number of letters in a license plate -/
def num_letters : ℕ := 3

/-- The number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- The number of possible letters -/
def letter_choices : ℕ := 26

/-- The number of non-vowel letters -/
def non_vowel_choices : ℕ := 21

/-- The number of positions where the letter block can be placed -/
def block_positions : ℕ := num_digits + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  block_positions * digit_choices^num_digits * (letter_choices^num_letters - non_vowel_choices^num_letters)

theorem license_plate_count : total_license_plates = 4989000000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1103_110396


namespace NUMINAMATH_CALUDE_unique_number_l1103_110391

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def is_multiple_of_13 (n : ℕ) : Prop := ∃ k, n = 13*k

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ k, n = k^2

theorem unique_number : 
  ∃! n : ℕ, 
    is_two_digit n ∧ 
    is_odd n ∧ 
    is_multiple_of_13 n ∧ 
    is_perfect_square (sum_of_digits n) ∧ 
    n = 13 := by sorry

end NUMINAMATH_CALUDE_unique_number_l1103_110391


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1103_110354

/-- Simplification of polynomial expression -/
theorem polynomial_simplification (x : ℝ) :
  (15 * x^12 + 8 * x^9 + 5 * x^7) + (3 * x^13 + 2 * x^12 + x^11 + 6 * x^9 + 3 * x^7 + 4 * x^4 + 6 * x^2 + 9) =
  3 * x^13 + 17 * x^12 + x^11 + 14 * x^9 + 8 * x^7 + 4 * x^4 + 6 * x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1103_110354


namespace NUMINAMATH_CALUDE_min_sum_of_bases_l1103_110380

theorem min_sum_of_bases (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  (3 * a + 6 = 6 * b + 3) → (∀ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 6 = 6 * y + 3 → a + b ≤ x + y) → 
  a + b = 20 := by sorry

end NUMINAMATH_CALUDE_min_sum_of_bases_l1103_110380


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l1103_110312

theorem real_part_of_complex_product : ∃ (z : ℂ), z = (1 - Complex.I) * (2 + Complex.I) ∧ z.re = 3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l1103_110312


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l1103_110363

/-- Given vectors a and b in ℝ², prove that the cosine of the angle between them is 2√13/13 -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (5, -10)) 
  (h2 : a - b = (3, 6)) : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 2 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l1103_110363


namespace NUMINAMATH_CALUDE_smallest_n_for_reducible_fraction_l1103_110356

theorem smallest_n_for_reducible_fraction : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 13) ∧ k ∣ (5*m + 6))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 13) ∧ k ∣ (5*n + 6)) ∧
  n = 84 := by
  sorry

#check smallest_n_for_reducible_fraction

end NUMINAMATH_CALUDE_smallest_n_for_reducible_fraction_l1103_110356


namespace NUMINAMATH_CALUDE_fibonacci_150_mod_9_l1103_110308

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_150_mod_9 : fibonacci 150 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_150_mod_9_l1103_110308


namespace NUMINAMATH_CALUDE_pentagon_centroid_intersection_l1103_110348

/-- Given a convex pentagon ABCDE in a real vector space, prove that the point P defined as
    (1/5)(A + B + C + D + E) is the intersection point of all segments connecting the midpoint
    of each side to the centroid of the triangle formed by the other three vertices. -/
theorem pentagon_centroid_intersection
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
  (A B C D E : V) :
  let P := (1/5 : ℝ) • (A + B + C + D + E)
  let midpoint (X Y : V) := (1/2 : ℝ) • (X + Y)
  let centroid (X Y Z : V) := (1/3 : ℝ) • (X + Y + Z)
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 →
    (midpoint A B + t • (centroid C D E - midpoint A B) = P) ∧
    (midpoint B C + t • (centroid D E A - midpoint B C) = P) ∧
    (midpoint C D + t • (centroid E A B - midpoint C D) = P) ∧
    (midpoint D E + t • (centroid A B C - midpoint D E) = P) ∧
    (midpoint E A + t • (centroid B C D - midpoint E A) = P) :=
by sorry


end NUMINAMATH_CALUDE_pentagon_centroid_intersection_l1103_110348


namespace NUMINAMATH_CALUDE_longest_tape_measure_l1103_110359

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 600) (hb : b = 500) (hc : c = 1200) : 
  Nat.gcd a (Nat.gcd b c) = 100 := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_measure_l1103_110359


namespace NUMINAMATH_CALUDE_second_snake_length_l1103_110316

/-- Proves that the length of the second snake is 16 inches -/
theorem second_snake_length (total_snakes : Nat) (first_snake_feet : Nat) (third_snake_inches : Nat) (total_length_inches : Nat) (inches_per_foot : Nat) :
  total_snakes = 3 →
  first_snake_feet = 2 →
  third_snake_inches = 10 →
  total_length_inches = 50 →
  inches_per_foot = 12 →
  total_length_inches - (first_snake_feet * inches_per_foot + third_snake_inches) = 16 := by
  sorry

end NUMINAMATH_CALUDE_second_snake_length_l1103_110316


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1103_110369

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a^2 + b^2 = 80) : 
  a * b = 32 := by sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1103_110369


namespace NUMINAMATH_CALUDE_expenditure_estimate_l1103_110347

/-- The regression line equation for a company's expenditure (y) based on revenue (x) -/
def regression_line (x : ℝ) (a : ℝ) : ℝ := 0.8 * x + a

/-- Theorem: Given the regression line equation, when revenue is 7 billion yuan, 
    the estimated expenditure is 4.4 billion yuan -/
theorem expenditure_estimate (a : ℝ) : 
  ∃ (y : ℝ), regression_line 7 a = y ∧ y = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_estimate_l1103_110347


namespace NUMINAMATH_CALUDE_lemonade_sales_increase_l1103_110341

/-- Calculates the percentage increase in lemonade sales --/
def percentage_increase (last_week : ℕ) (total : ℕ) : ℚ :=
  let this_week := total - last_week
  ((this_week - last_week : ℚ) / last_week) * 100

/-- Theorem stating the percentage increase in lemonade sales --/
theorem lemonade_sales_increase :
  let last_week := 20
  let total := 46
  percentage_increase last_week total = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_lemonade_sales_increase_l1103_110341


namespace NUMINAMATH_CALUDE_range_of_a_l1103_110340

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - a + 3

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, -1 < x₀ ∧ x₀ < 1 ∧ f a x₀ = 0) →
  (a < -3 ∨ a > 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1103_110340


namespace NUMINAMATH_CALUDE_total_pay_is_186_l1103_110326

/-- Calculates the total pay for a worker given regular and overtime hours -/
def total_pay (regular_rate : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) : ℝ :=
  let overtime_rate := 2 * regular_rate
  regular_rate * regular_hours + overtime_rate * overtime_hours

/-- Proves that the total pay for the given conditions is $186 -/
theorem total_pay_is_186 :
  total_pay 3 40 11 = 186 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_is_186_l1103_110326


namespace NUMINAMATH_CALUDE_subtracted_number_l1103_110376

theorem subtracted_number (x : ℤ) (y : ℤ) (h1 : x = 129) (h2 : 2 * x - y = 110) : y = 148 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l1103_110376


namespace NUMINAMATH_CALUDE_nina_payment_l1103_110343

theorem nina_payment (x y z w : ℕ) : 
  x + y + z + w = 27 →  -- Total number of coins
  y = 2 * z →           -- Number of 5 kopek coins is twice the number of 2 kopek coins
  z = 2 * x →           -- Number of 2 kopek coins is twice the number of 10 kopek coins
  7 < w →               -- Number of 3 kopek coins is more than 7
  w < 20 →              -- Number of 3 kopek coins is less than 20
  10 * x + 5 * y + 2 * z + 3 * w = 107 := by
sorry

end NUMINAMATH_CALUDE_nina_payment_l1103_110343


namespace NUMINAMATH_CALUDE_move_right_3_4_eq_7_l1103_110365

/-- Moving a point on a number line to the right is equivalent to addition -/
def move_right (start : ℤ) (distance : ℤ) : ℤ := start + distance

/-- The result of moving 4 units to the right from the point 3 on a number line -/
def result : ℤ := move_right 3 4

/-- Theorem: Moving 4 units to the right from the point 3 on a number line results in the point 7 -/
theorem move_right_3_4_eq_7 : result = 7 := by sorry

end NUMINAMATH_CALUDE_move_right_3_4_eq_7_l1103_110365


namespace NUMINAMATH_CALUDE_investment_difference_l1103_110328

def initial_investment : ℝ := 500

def jackson_growth_rate : ℝ := 4

def brandon_growth_rate : ℝ := 0.2

def jackson_final_value : ℝ := initial_investment * jackson_growth_rate

def brandon_final_value : ℝ := initial_investment * brandon_growth_rate

theorem investment_difference :
  jackson_final_value - brandon_final_value = 1900 := by sorry

end NUMINAMATH_CALUDE_investment_difference_l1103_110328


namespace NUMINAMATH_CALUDE_eighth_group_sample_l1103_110302

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (m : Nat) (k : Nat) : Nat :=
  (k - 1) * 10 + (m + k) % 10

/-- The problem statement as a theorem -/
theorem eighth_group_sample :
  ∀ m : Nat,
  m = 8 →
  systematicSample m 8 = 76 := by
  sorry

end NUMINAMATH_CALUDE_eighth_group_sample_l1103_110302


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1103_110315

theorem quadratic_minimum (a b c d : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≥ d) ∧ 
  (∃ x, a * x^2 + b * x + c = d) →
  c = d + b^2 / (4 * a) := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1103_110315


namespace NUMINAMATH_CALUDE_square_of_sum_17_5_l1103_110375

theorem square_of_sum_17_5 : 17^2 + 2*(17*5) + 5^2 = 484 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_17_5_l1103_110375


namespace NUMINAMATH_CALUDE_share_difference_l1103_110306

/-- Represents the distribution of money among five people -/
structure MoneyDistribution where
  faruk : ℕ
  vasim : ℕ
  ranjith : ℕ
  priya : ℕ
  elina : ℕ

/-- Theorem stating the difference in shares based on the given conditions -/
theorem share_difference (d : MoneyDistribution) :
  d.faruk = 3 * 600 ∧
  d.vasim = 5 * 600 ∧
  d.ranjith = 9 * 600 ∧
  d.priya = 7 * 600 ∧
  d.elina = 11 * 600 ∧
  d.vasim = 3000 →
  (d.faruk + d.ranjith + d.elina) - (d.vasim + d.priya) = 6600 := by
  sorry


end NUMINAMATH_CALUDE_share_difference_l1103_110306


namespace NUMINAMATH_CALUDE_existence_of_tangent_circle_l1103_110358

/-- Given three circles with radii 1, 3, and 4 touching each other and the sides of a rectangle,
    there exists a circle touching all three circles and one side of the rectangle. -/
theorem existence_of_tangent_circle (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 1) (h₂ : r₂ = 3) (h₃ : r₃ = 4) : 
  ∃ x : ℝ, 
    (x + r₁)^2 - (x - r₁)^2 = (r₂ + x)^2 - (r₂ + r₁ - x)^2 ∧
    ∃ y : ℝ, 
      (y + r₂)^2 - (r₂ + r₁ - y)^2 = (r₃ + y)^2 - (r₃ - y)^2 ∧
      x = y := by
  sorry

end NUMINAMATH_CALUDE_existence_of_tangent_circle_l1103_110358


namespace NUMINAMATH_CALUDE_line_symmetry_l1103_110313

-- Define the original line
def original_line (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = x

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := 3 * x - 2 * y - 1 = 0

-- Theorem stating the symmetry relationship
theorem line_symmetry :
  ∀ (x y : ℝ), original_line x y ↔ symmetric_line y x :=
by sorry

end NUMINAMATH_CALUDE_line_symmetry_l1103_110313


namespace NUMINAMATH_CALUDE_halloween_costume_payment_l1103_110332

theorem halloween_costume_payment (last_year_cost : ℝ) (price_increase_percent : ℝ) (deposit_percent : ℝ) : 
  last_year_cost = 250 →
  price_increase_percent = 40 →
  deposit_percent = 10 →
  let this_year_cost := last_year_cost * (1 + price_increase_percent / 100)
  let deposit := this_year_cost * (deposit_percent / 100)
  let remaining_payment := this_year_cost - deposit
  remaining_payment = 315 :=
by
  sorry

end NUMINAMATH_CALUDE_halloween_costume_payment_l1103_110332


namespace NUMINAMATH_CALUDE_straight_line_angle_value_l1103_110309

/-- The sum of angles in a straight line is 180 degrees -/
def straight_line_angle_sum : ℝ := 180

/-- The angles along the straight line ABC -/
def angle1 (x : ℝ) : ℝ := x
def angle2 : ℝ := 21
def angle3 : ℝ := 21
def angle4 (x : ℝ) : ℝ := 2 * x
def angle5 : ℝ := 57

/-- Theorem: Given a straight line ABC with angles x°, 21°, 21°, 2x°, and 57°, the value of x is 27° -/
theorem straight_line_angle_value :
  ∀ x : ℝ, 
  angle1 x + angle2 + angle3 + angle4 x + angle5 = straight_line_angle_sum → 
  x = 27 := by
sorry


end NUMINAMATH_CALUDE_straight_line_angle_value_l1103_110309


namespace NUMINAMATH_CALUDE_special_collection_total_l1103_110338

/-- A collection of shapes consisting of circles, squares, and triangles. -/
structure ShapeCollection where
  circles : ℕ
  squares : ℕ
  triangles : ℕ

/-- The total number of shapes in the collection. -/
def ShapeCollection.total (sc : ShapeCollection) : ℕ :=
  sc.circles + sc.squares + sc.triangles

/-- A collection satisfying the given conditions. -/
def specialCollection : ShapeCollection :=
  { circles := 5, squares := 1, triangles := 9 }

theorem special_collection_total :
  (specialCollection.squares + specialCollection.triangles = 10) ∧
  (specialCollection.circles + specialCollection.triangles = 14) ∧
  (specialCollection.circles + specialCollection.squares = 6) ∧
  specialCollection.total = 15 := by
  sorry

#eval specialCollection.total

end NUMINAMATH_CALUDE_special_collection_total_l1103_110338


namespace NUMINAMATH_CALUDE_x_plus_y_range_l1103_110320

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the conditions
def conditions (x y : ℝ) : Prop :=
  y = 3 * (floor x) + 4 ∧
  y = 4 * (floor (x - 3)) + 7 ∧
  x ≠ ↑(floor x)

-- Theorem statement
theorem x_plus_y_range (x y : ℝ) :
  conditions x y → 40 < x + y ∧ x + y < 41 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_range_l1103_110320


namespace NUMINAMATH_CALUDE_work_day_ends_at_target_time_l1103_110335

-- Define the start time, lunch time, and total work hours
def start_time : Nat := 8 * 60  -- 8:00 AM in minutes
def lunch_time : Nat := 13 * 60  -- 1:00 PM in minutes
def total_work_minutes : Nat := 9 * 60  -- 9 hours in minutes
def lunch_break_minutes : Nat := 30

-- Define the end time we want to prove
def target_end_time : Nat := 17 * 60 + 30  -- 5:30 PM in minutes

-- Theorem to prove
theorem work_day_ends_at_target_time :
  start_time + total_work_minutes + lunch_break_minutes = target_end_time := by
  sorry


end NUMINAMATH_CALUDE_work_day_ends_at_target_time_l1103_110335


namespace NUMINAMATH_CALUDE_game_probability_l1103_110385

/-- The probability of a specific outcome in a game with 8 rounds -/
theorem game_probability : 
  -- Total number of rounds
  (total_rounds : ℕ) →
  -- Alex's probability of winning a round
  (alex_prob : ℚ) →
  -- Mel's probability of winning a round
  (mel_prob : ℚ) →
  -- Chelsea's probability of winning a round
  (chelsea_prob : ℚ) →
  -- Number of rounds Alex wins
  (alex_wins : ℕ) →
  -- Number of rounds Mel wins
  (mel_wins : ℕ) →
  -- Number of rounds Chelsea wins
  (chelsea_wins : ℕ) →
  -- Conditions
  total_rounds = 8 →
  alex_prob = 2/5 →
  mel_prob = 3 * chelsea_prob →
  alex_prob + mel_prob + chelsea_prob = 1 →
  alex_wins + mel_wins + chelsea_wins = total_rounds →
  alex_wins = 3 →
  mel_wins = 4 →
  chelsea_wins = 1 →
  -- Conclusion
  (Nat.choose total_rounds alex_wins * Nat.choose (total_rounds - alex_wins) mel_wins *
   alex_prob ^ alex_wins * mel_prob ^ mel_wins * chelsea_prob ^ chelsea_wins : ℚ) = 881/1000 := by
sorry

end NUMINAMATH_CALUDE_game_probability_l1103_110385


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l1103_110336

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available. -/
def num_crystals : ℕ := 6

/-- The number of crystals incompatible with one specific herb. -/
def num_incompatible : ℕ := 3

/-- The number of viable combinations for the wizard's elixir. -/
def viable_combinations : ℕ := num_herbs * num_crystals - num_incompatible

theorem wizard_elixir_combinations :
  viable_combinations = 21 :=
sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l1103_110336


namespace NUMINAMATH_CALUDE_x_value_for_given_y_z_exists_constant_k_l1103_110361

/-- Given a relationship between x, y, and z, prove that x equals 5/8 for specific values of y and z -/
theorem x_value_for_given_y_z : ∀ (x y z k : ℝ), 
  (x = k * (z / y^2)) →  -- Relationship between x, y, and z
  (1 = k * (2 / 3^2)) →  -- Initial condition
  (y = 6 ∧ z = 5) →      -- New values for y and z
  x = 5/8 := by
    sorry

/-- There exists a constant k that satisfies the given conditions -/
theorem exists_constant_k : ∃ (k : ℝ), 
  (1 = k * (2 / 3^2)) ∧
  (∀ (x y z : ℝ), x = k * (z / y^2)) := by
    sorry

end NUMINAMATH_CALUDE_x_value_for_given_y_z_exists_constant_k_l1103_110361


namespace NUMINAMATH_CALUDE_inequality_proof_l1103_110357

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * (a + b) + b * c * (b + c) + a * c * (a + c) ≥ 6 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1103_110357


namespace NUMINAMATH_CALUDE_basketball_handshakes_l1103_110333

/-- The number of handshakes in a basketball game scenario --/
def total_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let player_handshakes := team_size * team_size
  let referee_handshakes := (team_size * num_teams) * num_referees
  player_handshakes + referee_handshakes

/-- Theorem stating the total number of handshakes in the given scenario --/
theorem basketball_handshakes :
  total_handshakes 6 2 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l1103_110333


namespace NUMINAMATH_CALUDE_candy_distribution_l1103_110345

/-- 
Given the initial number of candies, the number of friends, and the additional candies bought,
prove that the number of candies each friend will receive is equal to the total number of candies
divided by the number of friends.
-/
theorem candy_distribution (initial_candies : ℕ) (friends : ℕ) (additional_candies : ℕ)
  (h1 : initial_candies = 35)
  (h2 : friends = 10)
  (h3 : additional_candies = 15)
  (h4 : friends > 0) :
  (initial_candies + additional_candies) / friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1103_110345
