import Mathlib

namespace NUMINAMATH_CALUDE_other_factor_in_product_l2043_204386

theorem other_factor_in_product (w : ℕ+) (x : ℕ+) : 
  w = 156 →
  (∃ k : ℕ+, k * w * x = 2^5 * 3^3 * 13^2) →
  x = 936 := by
sorry

end NUMINAMATH_CALUDE_other_factor_in_product_l2043_204386


namespace NUMINAMATH_CALUDE_vacation_savings_theorem_l2043_204360

/-- Calculates the number of months needed to reach a savings goal -/
def months_to_goal (goal : ℕ) (current : ℕ) (monthly : ℕ) : ℕ :=
  ((goal - current) + monthly - 1) / monthly

theorem vacation_savings_theorem (goal current monthly : ℕ) 
  (h1 : goal = 5000)
  (h2 : current = 2900)
  (h3 : monthly = 700) :
  months_to_goal goal current monthly = 3 := by
  sorry

end NUMINAMATH_CALUDE_vacation_savings_theorem_l2043_204360


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2043_204303

theorem arithmetic_sequence_common_difference
  (a : ℚ)  -- first term
  (aₙ : ℚ) -- last term
  (S : ℚ)  -- sum of all terms
  (h1 : a = 3)
  (h2 : aₙ = 50)
  (h3 : S = 318) :
  ∃ (n : ℕ) (d : ℚ), n > 1 ∧ d = 47 / 11 ∧ aₙ = a + (n - 1) * d ∧ S = (n / 2) * (a + aₙ) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2043_204303


namespace NUMINAMATH_CALUDE_equal_area_triangles_l2043_204308

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem equal_area_triangles :
  triangleArea 20 20 24 = triangleArea 20 20 32 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_triangles_l2043_204308


namespace NUMINAMATH_CALUDE_sum_of_seven_smallest_multiples_of_12_l2043_204322

theorem sum_of_seven_smallest_multiples_of_12 : 
  (Finset.range 7).sum (λ n => 12 * (n + 1)) = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_smallest_multiples_of_12_l2043_204322


namespace NUMINAMATH_CALUDE_three_digit_cubes_divisible_by_eight_l2043_204340

theorem three_digit_cubes_divisible_by_eight :
  (∃! (s : Finset Nat), 
    (∀ n ∈ s, 100 ≤ n ∧ n ≤ 999 ∧ ∃ k, n = k^3 ∧ 8 ∣ n) ∧ 
    s.card = 2) :=
sorry

end NUMINAMATH_CALUDE_three_digit_cubes_divisible_by_eight_l2043_204340


namespace NUMINAMATH_CALUDE_root_absolute_value_greater_than_four_l2043_204396

theorem root_absolute_value_greater_than_four (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + p*r₁ + 16 = 0 → 
  r₂^2 + p*r₂ + 16 = 0 → 
  (abs r₁ > 4) ∨ (abs r₂ > 4) := by
sorry

end NUMINAMATH_CALUDE_root_absolute_value_greater_than_four_l2043_204396


namespace NUMINAMATH_CALUDE_hack_represents_8634_l2043_204381

-- Define the mapping of letters to digits
def letter_to_digit : Char → Nat
| 'Q' => 0
| 'U' => 1
| 'I' => 2
| 'C' => 3
| 'K' => 4
| 'M' => 5
| 'A' => 6
| 'T' => 7
| 'H' => 8
| 'S' => 9
| _ => 0  -- Default case for other characters

-- Define the code word
def code_word : List Char := ['H', 'A', 'C', 'K']

-- Theorem to prove
theorem hack_represents_8634 :
  (code_word.map letter_to_digit).foldl (fun acc d => acc * 10 + d) 0 = 8634 := by
  sorry

end NUMINAMATH_CALUDE_hack_represents_8634_l2043_204381


namespace NUMINAMATH_CALUDE_cades_remaining_marbles_l2043_204371

/-- Represents the number of marbles Cade has left after giving some away -/
def marblesLeft (initial : Nat) (givenAway : Nat) : Nat :=
  initial - givenAway

/-- Theorem stating that Cade has 79 marbles left -/
theorem cades_remaining_marbles :
  marblesLeft 87 8 = 79 := by
  sorry

end NUMINAMATH_CALUDE_cades_remaining_marbles_l2043_204371


namespace NUMINAMATH_CALUDE_simplify_expression_l2043_204398

theorem simplify_expression (a : ℝ) : 2*a*(3*a^2 - 4*a + 3) - 3*a^2*(2*a - 4) = 4*a^2 + 6*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2043_204398


namespace NUMINAMATH_CALUDE_not_right_triangle_l2043_204348

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the ratio condition
def ratio_condition (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.A = 5 * k ∧ t.B = 12 * k ∧ t.C = 13 * k

-- Theorem statement
theorem not_right_triangle (t : Triangle) (h : ratio_condition t) : 
  t.A ≠ 90 ∧ t.B ≠ 90 ∧ t.C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l2043_204348


namespace NUMINAMATH_CALUDE_square_sum_l2043_204309

theorem square_sum (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_l2043_204309


namespace NUMINAMATH_CALUDE_find_z_l2043_204397

theorem find_z (x y z : ℚ) 
  (h1 : x = (1/3) * y) 
  (h2 : y = (1/4) * z) 
  (h3 : x + y = 16) : 
  z = 48 := by
sorry

end NUMINAMATH_CALUDE_find_z_l2043_204397


namespace NUMINAMATH_CALUDE_sphere_volume_inscribed_cylinder_l2043_204390

-- Define the radius of the base of the cylinder
def r : ℝ := 15

-- Define the radius of the sphere
def sphere_radius : ℝ := r + 2

-- Define the height of the cylinder
def cylinder_height : ℝ := r + 1

-- State the theorem
theorem sphere_volume_inscribed_cylinder :
  let volume := (4 / 3) * Real.pi * sphere_radius ^ 3
  (2 * sphere_radius) ^ 2 = (2 * r) ^ 2 + cylinder_height ^ 2 →
  volume = 6550 * (2 / 3) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_sphere_volume_inscribed_cylinder_l2043_204390


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2043_204343

theorem quadratic_equation_properties :
  ∀ (k : ℝ), 
  -- The equation has two distinct real roots
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * x₁^2 + k * x₁ - 1 = 0 ∧ 2 * x₂^2 + k * x₂ - 1 = 0) ∧
  -- When one root is -1, the other is 1/2 and k = 1
  (2 * (-1)^2 + k * (-1) - 1 = 0 → k = 1 ∧ 2 * (1/2)^2 + 1 * (1/2) - 1 = 0) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equation_properties_l2043_204343


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l2043_204334

/-- Given a line with y-intercept 3 and slope -3/2, prove that the product of its slope and y-intercept is -9/2 -/
theorem line_slope_intercept_product :
  ∀ (m b : ℚ),
    b = 3 →
    m = -3/2 →
    m * b = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l2043_204334


namespace NUMINAMATH_CALUDE_quartic_roots_l2043_204326

theorem quartic_roots : 
  let f : ℝ → ℝ := λ x ↦ 3*x^4 + 2*x^3 - 8*x^2 + 2*x + 3
  ∃ (a b c d : ℝ), 
    (a = (1 - Real.sqrt 43 + 2*Real.sqrt 34) / 6) ∧
    (b = (1 - Real.sqrt 43 - 2*Real.sqrt 34) / 6) ∧
    (c = (1 + Real.sqrt 43 + 2*Real.sqrt 34) / 6) ∧
    (d = (1 + Real.sqrt 43 - 2*Real.sqrt 34) / 6) ∧
    (∀ x : ℝ, f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_l2043_204326


namespace NUMINAMATH_CALUDE_vacation_cost_difference_l2043_204347

theorem vacation_cost_difference (total_cost : ℕ) (initial_people : ℕ) (new_people : ℕ) 
  (h1 : total_cost = 360) 
  (h2 : initial_people = 3) 
  (h3 : new_people = 4) : 
  (total_cost / initial_people) - (total_cost / new_people) = 30 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_difference_l2043_204347


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2043_204315

theorem quadratic_equal_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*a*y + a + 2 = 0 → y = x) → 
  a = -1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2043_204315


namespace NUMINAMATH_CALUDE_maggis_cupcakes_l2043_204389

theorem maggis_cupcakes (cupcakes_per_package : ℕ) (cupcakes_eaten : ℕ) (cupcakes_left : ℕ) :
  cupcakes_per_package = 4 →
  cupcakes_eaten = 5 →
  cupcakes_left = 12 →
  ∃ (initial_packages : ℕ), 
    initial_packages * cupcakes_per_package = cupcakes_left + cupcakes_eaten ∧
    initial_packages = 4 :=
by sorry

end NUMINAMATH_CALUDE_maggis_cupcakes_l2043_204389


namespace NUMINAMATH_CALUDE_quadratic_function_product_sign_l2043_204317

theorem quadratic_function_product_sign
  (a b c m n p x₁ x₂ : ℝ)
  (h_a_pos : a > 0)
  (h_roots : x₁ < x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0)
  (h_order : m < x₁ ∧ x₁ < n ∧ n < x₂ ∧ x₂ < p) :
  let f := fun x => a * x^2 + b * x + c
  f m * f n * f p < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_product_sign_l2043_204317


namespace NUMINAMATH_CALUDE_total_dolls_l2043_204300

/-- The number of dolls each person has -/
structure DollCounts where
  vera : ℕ
  sophie : ℕ
  aida : ℕ

/-- The conditions of the doll distribution -/
def doll_distribution (d : DollCounts) : Prop :=
  d.vera = 20 ∧ d.sophie = 2 * d.vera ∧ d.aida = 2 * d.sophie

/-- The theorem stating the total number of dolls -/
theorem total_dolls (d : DollCounts) (h : doll_distribution d) : 
  d.vera + d.sophie + d.aida = 140 := by
  sorry

#check total_dolls

end NUMINAMATH_CALUDE_total_dolls_l2043_204300


namespace NUMINAMATH_CALUDE_elliot_reading_rate_l2043_204384

/-- Given a book with a certain number of pages, the number of pages read before a week,
    and the number of pages left after a week of reading, calculate the number of pages read per day. -/
def pages_per_day (total_pages : ℕ) (pages_read_before : ℕ) (pages_left : ℕ) : ℕ :=
  ((total_pages - pages_left) - pages_read_before) / 7

/-- Theorem stating that for Elliot's specific reading scenario, he reads 20 pages per day. -/
theorem elliot_reading_rate : pages_per_day 381 149 92 = 20 := by
  sorry

end NUMINAMATH_CALUDE_elliot_reading_rate_l2043_204384


namespace NUMINAMATH_CALUDE_product_markup_rate_l2043_204306

theorem product_markup_rate (selling_price : ℝ) (profit_rate : ℝ) (expense_rate : ℝ) (fixed_cost : ℝ) :
  selling_price = 10 ∧ 
  profit_rate = 0.20 ∧ 
  expense_rate = 0.30 ∧ 
  fixed_cost = 1 →
  let variable_cost := selling_price * (1 - profit_rate - expense_rate) - fixed_cost
  (selling_price - variable_cost) / variable_cost = 1.5 := by sorry

end NUMINAMATH_CALUDE_product_markup_rate_l2043_204306


namespace NUMINAMATH_CALUDE_quadratic_solution_set_l2043_204341

theorem quadratic_solution_set (b c : ℝ) : 
  (∀ x, x^2 + 2*b*x + c ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1) → b + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_l2043_204341


namespace NUMINAMATH_CALUDE_expected_potato_yield_l2043_204359

/-- Calculates the expected potato yield from a rectangular garden --/
theorem expected_potato_yield
  (garden_length_steps : ℕ)
  (garden_width_steps : ℕ)
  (step_length_feet : ℝ)
  (potato_yield_per_sqft : ℝ)
  (h1 : garden_length_steps = 18)
  (h2 : garden_width_steps = 25)
  (h3 : step_length_feet = 3)
  (h4 : potato_yield_per_sqft = 1/3) :
  (garden_length_steps : ℝ) * step_length_feet *
  (garden_width_steps : ℝ) * step_length_feet *
  potato_yield_per_sqft = 1350 := by
  sorry

end NUMINAMATH_CALUDE_expected_potato_yield_l2043_204359


namespace NUMINAMATH_CALUDE_chord_length_l2043_204354

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the line l
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 11 = 0}

-- Define the intersection points A and B
def intersection_points : Set (ℝ × ℝ) :=
  circle_C ∩ line_l

-- Theorem statement
theorem chord_length :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_chord_length_l2043_204354


namespace NUMINAMATH_CALUDE_root_product_equals_27_l2043_204314

theorem root_product_equals_27 : (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_27_l2043_204314


namespace NUMINAMATH_CALUDE_half_abs_diff_cubes_20_15_l2043_204399

theorem half_abs_diff_cubes_20_15 : 
  (1/2 : ℝ) * |20^3 - 15^3| = 2312.5 := by sorry

end NUMINAMATH_CALUDE_half_abs_diff_cubes_20_15_l2043_204399


namespace NUMINAMATH_CALUDE_sinusoidal_period_l2043_204383

/-- 
Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, and d are positive constants,
if the function completes five periods over an interval of 2π, then b = 5.
-/
theorem sinusoidal_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_periods : (2 * Real.pi) / b = (2 * Real.pi) / 5) : b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_period_l2043_204383


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_l2043_204323

theorem max_consecutive_integers_sum (k : ℕ) : 
  (∃ n : ℕ, (k : ℤ) * (2 * n + k - 1) = 2 * 3^8) →
  k ≤ 108 :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_l2043_204323


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l2043_204335

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l2043_204335


namespace NUMINAMATH_CALUDE_square_roots_equation_l2043_204355

theorem square_roots_equation (a b : ℝ) :
  let f (x : ℝ) := a * b * x^2 - (a + b) * x + 1
  let g (x : ℝ) := a^2 * b^2 * x^2 - (a^2 + b^2) * x + 1
  ∀ (r : ℝ), f r = 0 → g (r^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_equation_l2043_204355


namespace NUMINAMATH_CALUDE_fraction_scaling_l2043_204362

theorem fraction_scaling (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (6 * x + 6 * y) / ((6 * x) * (6 * y)) = (1 / 6) * ((x + y) / (x * y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_scaling_l2043_204362


namespace NUMINAMATH_CALUDE_power_of_nine_l2043_204304

theorem power_of_nine (n : ℕ) (h : 3^(2*n) = 81) : 9^(n+1) = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_nine_l2043_204304


namespace NUMINAMATH_CALUDE_chicken_count_after_purchase_l2043_204363

theorem chicken_count_after_purchase (initial_count purchase_count : ℕ) 
  (h1 : initial_count = 26) 
  (h2 : purchase_count = 28) : 
  initial_count + purchase_count = 54 := by
  sorry

end NUMINAMATH_CALUDE_chicken_count_after_purchase_l2043_204363


namespace NUMINAMATH_CALUDE_count_propositions_l2043_204391

-- Define a function to check if a statement is a proposition
def isProposition (s : String) : Bool :=
  match s with
  | "|x+2|" => false
  | "-5 ∈ ℤ" => true
  | "π ∉ ℝ" => true
  | "{0} ∈ ℕ" => true
  | _ => false

-- Define the list of statements
def statements : List String := ["|x+2|", "-5 ∈ ℤ", "π ∉ ℝ", "{0} ∈ ℕ"]

-- Theorem to prove
theorem count_propositions :
  (statements.filter isProposition).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_propositions_l2043_204391


namespace NUMINAMATH_CALUDE_population_halving_time_island_l2043_204357

/-- The time it takes for a population to halve given initial population and net emigration rate -/
def time_to_halve_population (initial_population : ℕ) (net_emigration_rate_per_500 : ℚ) : ℚ :=
  let net_emigration_rate := (initial_population : ℚ) / 500 * net_emigration_rate_per_500
  (initial_population : ℚ) / (2 * net_emigration_rate)

theorem population_halving_time_island (ε : ℚ) :
  ∃ (δ : ℚ), δ > 0 ∧ |time_to_halve_population 5000 35 - 7.14| < δ → δ < ε :=
sorry

end NUMINAMATH_CALUDE_population_halving_time_island_l2043_204357


namespace NUMINAMATH_CALUDE_field_ratio_is_two_to_one_l2043_204385

/-- Proves that the ratio of length to width of a rectangular field is 2:1 given specific conditions --/
theorem field_ratio_is_two_to_one (field_length field_width pond_side : ℝ) : 
  field_length = 80 →
  pond_side = 8 →
  field_length * field_width = 50 * (pond_side * pond_side) →
  field_length / field_width = 2 := by
sorry

end NUMINAMATH_CALUDE_field_ratio_is_two_to_one_l2043_204385


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l2043_204327

theorem smaller_number_in_ratio (p q k : ℝ) (hp : p > 0) (hq : q > 0) : 
  p / q = 3 / 5 → p^2 + q^2 = 2 * k → min p q = 3 * Real.sqrt (k / 17) := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l2043_204327


namespace NUMINAMATH_CALUDE_sum_of_squares_difference_l2043_204366

theorem sum_of_squares_difference (a b : ℕ+) (h : a.val^2 - b.val^4 = 2009) : 
  a.val + b.val = 47 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_difference_l2043_204366


namespace NUMINAMATH_CALUDE_unique_prime_ending_l2043_204330

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number (A : ℕ) : ℕ := 202100 + A

theorem unique_prime_ending :
  ∃! A : ℕ, A < 10 ∧ is_prime (number A) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_ending_l2043_204330


namespace NUMINAMATH_CALUDE_tip_percentage_calculation_l2043_204332

theorem tip_percentage_calculation (total_bill : ℝ) (billy_tip : ℝ) (billy_percentage : ℝ) :
  total_bill = 50 →
  billy_tip = 8 →
  billy_percentage = 0.8 →
  (billy_tip / billy_percentage) / total_bill = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_calculation_l2043_204332


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2043_204337

theorem condition_necessary_not_sufficient :
  (∃ a b : ℝ, a + b ≠ 3 ∧ (a = 1 ∧ b = 2)) ∧
  (∀ a b : ℝ, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  (∃ a b : ℝ, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2043_204337


namespace NUMINAMATH_CALUDE_vector_b_value_l2043_204310

/-- Given a vector a and conditions on vector b, prove b equals (-3, 6) -/
theorem vector_b_value (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  (∃ k : ℝ, k < 0 ∧ b = k • a) → 
  ‖b‖ = 3 * Real.sqrt 5 → 
  b = (-3, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_b_value_l2043_204310


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l2043_204373

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : (m^2 + 2^m) % 10 = 3 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l2043_204373


namespace NUMINAMATH_CALUDE_right_triangle_trig_l2043_204311

theorem right_triangle_trig (A B C : ℝ) (h_right : A^2 + B^2 = C^2) 
  (h_hypotenuse : C = 15) (h_leg : A = 7) :
  Real.sqrt ((C^2 - A^2) / C^2) = 4 * Real.sqrt 11 / 15 ∧ A / C = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_l2043_204311


namespace NUMINAMATH_CALUDE_base_conversion_256_to_base_5_l2043_204361

def base_five_to_decimal (a b c d : ℕ) : ℕ :=
  a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

theorem base_conversion_256_to_base_5 :
  base_five_to_decimal 2 0 1 1 = 256 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_256_to_base_5_l2043_204361


namespace NUMINAMATH_CALUDE_rectangle_area_l2043_204321

/-- Theorem: Area of a rectangle with length 15 cm and width 0.9 times its length -/
theorem rectangle_area (length : ℝ) (width : ℝ) : 
  length = 15 →
  width = 0.9 * length →
  length * width = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2043_204321


namespace NUMINAMATH_CALUDE_sqrt_450_equals_15_sqrt_2_l2043_204331

theorem sqrt_450_equals_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_equals_15_sqrt_2_l2043_204331


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2043_204356

theorem age_ratio_proof (parent_age son_age : ℕ) : 
  parent_age = 45 →
  son_age = 15 →
  parent_age + 5 = (5/2) * (son_age + 5) →
  parent_age / son_age = 3 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l2043_204356


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2043_204318

theorem pure_imaginary_condition (a : ℝ) : 
  (∀ z : ℂ, z = (a^2 - 2*a - 3 : ℝ) + (a + 1 : ℝ) * I → z.re = 0 ∧ z.im ≠ 0) → 
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2043_204318


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2043_204352

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 3*x - 2*a^2 + 4 = 0}

-- State the theorem
theorem union_of_A_and_B (a : ℝ) :
  (A a ∩ B a = {1}) →
  ((a = 2 ∧ A a ∪ B a = {-4, 1}) ∨ (a = -2 ∧ A a ∪ B a = {-4, -3, 1})) :=
by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2043_204352


namespace NUMINAMATH_CALUDE_parabola_circle_theorem_l2043_204358

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℝ
  eq : (y : ℝ) → (x : ℝ) → Prop := fun y x => y^2 = 4 * a * x

/-- Represents a circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (p.a, 0)

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop := fun x => x = -p.a

/-- The chord length of the intersection between a parabola and its directrix -/
def chordLength (p : Parabola) : ℝ := sorry

/-- The standard equation of a circle -/
def standardEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem parabola_circle_theorem (p : Parabola) (c : Circle) :
  p.a = 1 →
  c.center = focus p →
  chordLength p = 6 →
  ∀ x y, standardEquation c x y ↔ (x - 1)^2 + y^2 = 13 := by sorry

end NUMINAMATH_CALUDE_parabola_circle_theorem_l2043_204358


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l2043_204387

theorem jason_pokemon_cards (initial_cards : ℕ) (bought_cards : ℕ) :
  initial_cards = 1342 →
  bought_cards = 536 →
  initial_cards - bought_cards = 806 :=
by sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l2043_204387


namespace NUMINAMATH_CALUDE_carpet_square_cost_l2043_204339

/-- The cost of each carpet square given floor and carpet dimensions and total cost -/
theorem carpet_square_cost
  (floor_length : ℝ)
  (floor_width : ℝ)
  (square_side : ℝ)
  (total_cost : ℝ)
  (h1 : floor_length = 6)
  (h2 : floor_width = 10)
  (h3 : square_side = 2)
  (h4 : total_cost = 225) :
  (total_cost / ((floor_length * floor_width) / (square_side * square_side))) = 15 := by
  sorry

#check carpet_square_cost

end NUMINAMATH_CALUDE_carpet_square_cost_l2043_204339


namespace NUMINAMATH_CALUDE_i_power_2013_l2043_204345

theorem i_power_2013 (i : ℂ) (h : i^2 = -1) : i^2013 = i := by
  sorry

end NUMINAMATH_CALUDE_i_power_2013_l2043_204345


namespace NUMINAMATH_CALUDE_seashells_needed_l2043_204333

def current_seashells : ℕ := 19
def goal_seashells : ℕ := 25

theorem seashells_needed : goal_seashells - current_seashells = 6 := by
  sorry

end NUMINAMATH_CALUDE_seashells_needed_l2043_204333


namespace NUMINAMATH_CALUDE_sequence_property_l2043_204393

theorem sequence_property (a : ℕ → ℤ) (h1 : a 2 = 4)
  (h2 : ∀ n : ℕ, n ≥ 1 → (a (n + 1) - a n : ℚ) < 2^n + 1/2)
  (h3 : ∀ n : ℕ, n ≥ 1 → (a (n + 2) - a n : ℤ) > 3 * 2^n - 1) :
  a 2018 = 2^2018 :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l2043_204393


namespace NUMINAMATH_CALUDE_min_value_equality_l2043_204378

theorem min_value_equality (x y a : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) :
  (∀ x y, x + 2*y = 1 → (3/x + a/y ≥ 6*Real.sqrt 3)) ∧
  (∃ x y, x + 2*y = 1 ∧ 3/x + a/y = 6*Real.sqrt 3) →
  (∀ x y, 1/x + 2/y = 1 → (3*x + a*y ≥ 6*Real.sqrt 3)) ∧
  (∃ x y, 1/x + 2/y = 1 ∧ 3*x + a*y = 6*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_equality_l2043_204378


namespace NUMINAMATH_CALUDE_total_students_suggestion_l2043_204375

theorem total_students_suggestion (mashed_potatoes bacon tomatoes : ℕ) 
  (h1 : mashed_potatoes = 324)
  (h2 : bacon = 374)
  (h3 : tomatoes = 128) :
  mashed_potatoes + bacon + tomatoes = 826 := by
  sorry

end NUMINAMATH_CALUDE_total_students_suggestion_l2043_204375


namespace NUMINAMATH_CALUDE_sum_of_factors_72_l2043_204394

def sum_of_factors (n : ℕ) : ℕ := sorry

theorem sum_of_factors_72 : sum_of_factors 72 = 195 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_72_l2043_204394


namespace NUMINAMATH_CALUDE_marnie_bracelets_l2043_204338

def beads_per_bracelet : ℕ := 65

def total_beads : ℕ :=
  5 * 50 + 2 * 100 + 3 * 75 + 4 * 125

theorem marnie_bracelets :
  (total_beads / beads_per_bracelet : ℕ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_marnie_bracelets_l2043_204338


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l2043_204392

theorem final_sum_after_transformation (x y S : ℝ) (h : x + y = S) :
  3 * (x + 4) + 3 * (y + 4) = 3 * S + 24 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l2043_204392


namespace NUMINAMATH_CALUDE_lars_bakery_production_l2043_204342

-- Define the baking rates and working hours
def bread_per_hour : ℕ := 10
def baguettes_per_two_hours : ℕ := 30
def working_hours : ℕ := 6

-- Define the function to calculate total breads per day
def total_breads_per_day : ℕ :=
  (bread_per_hour * working_hours) + (baguettes_per_two_hours * (working_hours / 2))

-- Theorem statement
theorem lars_bakery_production :
  total_breads_per_day = 150 := by sorry

end NUMINAMATH_CALUDE_lars_bakery_production_l2043_204342


namespace NUMINAMATH_CALUDE_monster_hunt_l2043_204325

theorem monster_hunt (x : ℕ) : 
  (x + 2*x + 4*x + 8*x + 16*x = 62) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_monster_hunt_l2043_204325


namespace NUMINAMATH_CALUDE_main_theorem_l2043_204305

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f (f y))^2)

/-- The main theorem to prove -/
theorem main_theorem (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
sorry

end NUMINAMATH_CALUDE_main_theorem_l2043_204305


namespace NUMINAMATH_CALUDE_coefficient_m4n4_in_expansion_l2043_204344

theorem coefficient_m4n4_in_expansion : ∀ m n : ℕ,
  (Nat.choose 8 4 : ℕ) = 70 := by sorry

end NUMINAMATH_CALUDE_coefficient_m4n4_in_expansion_l2043_204344


namespace NUMINAMATH_CALUDE_savings_distribution_l2043_204364

/-- Calculates the amount each child receives from the couple's savings --/
theorem savings_distribution (husband_contribution : ℕ) (wife_contribution : ℕ)
  (husband_interval : ℕ) (wife_interval : ℕ) (months : ℕ) (days_per_month : ℕ)
  (savings_percentage : ℚ) (num_children : ℕ) :
  husband_contribution = 450 →
  wife_contribution = 315 →
  husband_interval = 10 →
  wife_interval = 5 →
  months = 8 →
  days_per_month = 30 →
  savings_percentage = 3/4 →
  num_children = 6 →
  (((months * days_per_month / husband_interval) * husband_contribution +
    (months * days_per_month / wife_interval) * wife_contribution) *
    savings_percentage / num_children : ℚ) = 3240 := by
  sorry

end NUMINAMATH_CALUDE_savings_distribution_l2043_204364


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2043_204382

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2043_204382


namespace NUMINAMATH_CALUDE_custom_op_result_l2043_204388

-- Define the custom operation ⊗
def customOp (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

-- Define sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem custom_op_result :
  customOp M N = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (2 ≤ x ∧ x < 3)} := by sorry

end NUMINAMATH_CALUDE_custom_op_result_l2043_204388


namespace NUMINAMATH_CALUDE_painted_cubes_l2043_204379

theorem painted_cubes (total_cubes : ℕ) (unpainted_cubes : ℕ) (side_length : ℕ) : 
  unpainted_cubes = 24 →
  side_length = 5 →
  total_cubes = side_length^3 →
  total_cubes - unpainted_cubes = 101 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_l2043_204379


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2043_204365

theorem coefficient_x_squared_in_expansion :
  (Finset.range 5).sum (fun k => (Nat.choose 4 k : ℤ) * (-2)^(4 - k) * (if k = 2 then 1 else 0)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2043_204365


namespace NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_B_l2043_204313

-- Define sets A and B
def A : Set ℝ := {x : ℝ | (x + 1) * (4 - x) ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 2}

-- Theorem 1
theorem intersection_nonempty (a : ℝ) : 
  (A ∩ B a).Nonempty ↔ -1/2 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem 2
theorem intersection_equals_B (a : ℝ) :
  A ∩ B a = B a ↔ a ≥ 2 ∨ a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_B_l2043_204313


namespace NUMINAMATH_CALUDE_literary_society_book_exchange_l2043_204351

/-- The number of books exchanged in a Literary Society book sharing ceremony -/
def books_exchanged (x : ℕ) : ℕ := x * (x - 1)

/-- The theorem stating that for a group of x students where each student gives one book to every
    other student, and a total of 210 books are exchanged, the equation x(x-1) = 210 holds -/
theorem literary_society_book_exchange (x : ℕ) (h : books_exchanged x = 210) :
  x * (x - 1) = 210 := by sorry

end NUMINAMATH_CALUDE_literary_society_book_exchange_l2043_204351


namespace NUMINAMATH_CALUDE_parametric_curve_length_l2043_204336

/-- The parametric curve described by (x,y) = (3 sin t, 3 cos t) for t ∈ [0, 2π] -/
def parametric_curve : Set (ℝ × ℝ) :=
  {p | ∃ t ∈ Set.Icc 0 (2 * Real.pi), p = (3 * Real.sin t, 3 * Real.cos t)}

/-- The length of a curve -/
noncomputable def curve_length (c : Set (ℝ × ℝ)) : ℝ := sorry

theorem parametric_curve_length :
  curve_length parametric_curve = 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_parametric_curve_length_l2043_204336


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l2043_204329

theorem complex_multiplication_result : 
  let i : ℂ := Complex.I
  (3 - 4*i) * (2 + 6*i) * (-1 + 2*i) = -50 + 50*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l2043_204329


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l2043_204316

theorem sqrt_two_minus_one_power (n : ℕ+) :
  ∃ (a b : ℤ) (m : ℕ),
    (Real.sqrt 2 - 1) ^ (n : ℝ) = b * Real.sqrt 2 - a ∧
    m = a ^ 2 * b ^ 2 + 1 ∧
    m > 1 ∧
    (Real.sqrt 2 - 1) ^ (n : ℝ) = Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l2043_204316


namespace NUMINAMATH_CALUDE_inverse_of_f_l2043_204307

def f (x : ℝ) : ℝ := 3 - 4 * x

theorem inverse_of_f :
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ x, f (g x) = x) ∧ (∀ x, g x = (3 - x) / 4) :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_f_l2043_204307


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_l2043_204369

/-- The line equation in terms of a, x, and y -/
def line_equation (a x y : ℝ) : Prop :=
  (3*a - 1)*x + (2 - a)*y - 1 = 0

/-- The second quadrant -/
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- The line does not pass through the second quadrant -/
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y, line_equation a x y → ¬ second_quadrant x y

/-- The main theorem -/
theorem line_not_in_second_quadrant_iff (a : ℝ) :
  not_in_second_quadrant a ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_l2043_204369


namespace NUMINAMATH_CALUDE_complex_sum_real_necessary_not_sufficient_l2043_204374

theorem complex_sum_real_necessary_not_sufficient (z₁ z₂ : ℂ) :
  (∃ (a b : ℝ), z₁ = a + b * I ∧ z₂ = a - b * I) → (z₁ + z₂).im = 0 ∧
  ¬(∀ z₁ z₂ : ℂ, (z₁ + z₂).im = 0 → ∃ (a b : ℝ), z₁ = a + b * I ∧ z₂ = a - b * I) :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_real_necessary_not_sufficient_l2043_204374


namespace NUMINAMATH_CALUDE_distance_between_given_lines_l2043_204353

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ × ℝ  -- Point on first line
  b : ℝ × ℝ  -- Point on second line
  d : ℝ × ℝ  -- Direction vector (same for both lines)

/-- The distance between two parallel lines -/
def distance (lines : ParallelLines) : ℝ :=
  sorry

/-- Theorem stating that the distance between the given parallel lines is 0 -/
theorem distance_between_given_lines :
  let lines : ParallelLines := {
    a := (3, -4),
    b := (2, -1),
    d := (-1, 3)
  }
  distance lines = 0 := by sorry

end NUMINAMATH_CALUDE_distance_between_given_lines_l2043_204353


namespace NUMINAMATH_CALUDE_rectangleEnclosures_eq_100_l2043_204319

/-- The number of ways to choose 4 lines (2 horizontal and 2 vertical) from 5 horizontal and 5 vertical lines to enclose a rectangular region. -/
def rectangleEnclosures : ℕ :=
  let horizontalLines := 5
  let verticalLines := 5
  let horizontalChoices := Nat.choose horizontalLines 2
  let verticalChoices := Nat.choose verticalLines 2
  horizontalChoices * verticalChoices

/-- Theorem stating that the number of ways to choose 4 lines to enclose a rectangular region is 100. -/
theorem rectangleEnclosures_eq_100 : rectangleEnclosures = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangleEnclosures_eq_100_l2043_204319


namespace NUMINAMATH_CALUDE_profitable_after_three_years_l2043_204302

/-- Represents the financial data for the communication equipment --/
structure EquipmentData where
  initialInvestment : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrease : ℕ
  annualProfit : ℕ

/-- Calculates the cumulative profit after a given number of years --/
def cumulativeProfit (data : EquipmentData) (years : ℕ) : ℤ :=
  (data.annualProfit * years : ℤ) - 
  (data.initialInvestment + data.firstYearExpenses * years + 
   data.annualExpenseIncrease * (years * (years - 1) / 2) : ℤ)

/-- Theorem stating that the equipment becomes profitable after 3 years --/
theorem profitable_after_three_years (data : EquipmentData) 
  (h1 : data.initialInvestment = 980000)
  (h2 : data.firstYearExpenses = 120000)
  (h3 : data.annualExpenseIncrease = 40000)
  (h4 : data.annualProfit = 500000) :
  cumulativeProfit data 3 > 0 ∧ cumulativeProfit data 2 ≤ 0 := by
  sorry

#check profitable_after_three_years

end NUMINAMATH_CALUDE_profitable_after_three_years_l2043_204302


namespace NUMINAMATH_CALUDE_circumscribed_circle_diameter_l2043_204312

theorem circumscribed_circle_diameter 
  (side : ℝ) (angle : ℝ) (h1 : side = 15) (h2 : angle = π / 4) :
  (side / Real.sin angle) = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_diameter_l2043_204312


namespace NUMINAMATH_CALUDE_max_distance_point_to_line_l2043_204377

/-- The maximum distance from point A(1,1) to the line x*cos(θ) + y*sin(θ) - 2 = 0 -/
theorem max_distance_point_to_line :
  let A : ℝ × ℝ := (1, 1)
  let line (θ : ℝ) (x y : ℝ) := x * Real.cos θ + y * Real.sin θ - 2 = 0
  let distance (θ : ℝ) := |Real.cos θ + Real.sin θ - 2| / Real.sqrt (Real.cos θ ^ 2 + Real.sin θ ^ 2)
  (∀ θ : ℝ, distance θ ≤ 2 + Real.sqrt 2) ∧ (∃ θ : ℝ, distance θ = 2 + Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_max_distance_point_to_line_l2043_204377


namespace NUMINAMATH_CALUDE_length_of_AC_l2043_204301

/-- Given a quadrilateral ABCD with specific side lengths, prove the length of AC -/
theorem length_of_AC (AB DC AD : ℝ) (h1 : AB = 12) (h2 : DC = 15) (h3 : AD = 9) :
  ∃ (AC : ℝ), AC^2 = 585 := by sorry

end NUMINAMATH_CALUDE_length_of_AC_l2043_204301


namespace NUMINAMATH_CALUDE_equation_solution_l2043_204367

theorem equation_solution (x y r s : ℚ) : 
  (3 * x + 2 * y = 16) → 
  (5 * x + 3 * y = 26) → 
  (r = x) → 
  (s = y) → 
  (r - s = 2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2043_204367


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2043_204349

-- Define the universal set U
def U : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Define set A
def A : Set ℝ := {x | 1 / x ≥ 1}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = {x | -1 < x ∧ x ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2043_204349


namespace NUMINAMATH_CALUDE_optimal_choice_is_96_l2043_204368

def count_rectangles (perimeter : ℕ) : ℕ :=
  if perimeter % 2 = 0 then
    (perimeter / 2 - 1) / 2
  else
    0

theorem optimal_choice_is_96 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 97 → count_rectangles n ≤ count_rectangles 96 :=
by sorry

end NUMINAMATH_CALUDE_optimal_choice_is_96_l2043_204368


namespace NUMINAMATH_CALUDE_specific_test_result_l2043_204372

/-- Represents a test with a given number of questions, points for correct and incorrect answers --/
structure Test where
  total_questions : ℕ
  points_correct : ℤ
  points_incorrect : ℤ

/-- Represents the result of a test --/
structure TestResult where
  test : Test
  correct_answers : ℕ
  final_score : ℤ

/-- Theorem stating that for a specific test configuration, 
    if the final score is 0, then the number of correct answers is 10 --/
theorem specific_test_result (t : Test) (r : TestResult) : 
  t.total_questions = 26 ∧ 
  t.points_correct = 8 ∧ 
  t.points_incorrect = -5 ∧ 
  r.test = t ∧
  r.correct_answers + (t.total_questions - r.correct_answers) = t.total_questions ∧
  r.final_score = r.correct_answers * t.points_correct + (t.total_questions - r.correct_answers) * t.points_incorrect ∧
  r.final_score = 0 →
  r.correct_answers = 10 := by
sorry

end NUMINAMATH_CALUDE_specific_test_result_l2043_204372


namespace NUMINAMATH_CALUDE_mark_cookies_sold_l2043_204380

theorem mark_cookies_sold (n : ℕ) (mark_sold ann_sold : ℕ) : 
  n = 12 →
  mark_sold < n →
  ann_sold = n - 2 →
  mark_sold ≥ 1 →
  ann_sold ≥ 1 →
  mark_sold + ann_sold < n →
  mark_sold = n - 11 :=
by sorry

end NUMINAMATH_CALUDE_mark_cookies_sold_l2043_204380


namespace NUMINAMATH_CALUDE_integer_roots_quadratic_l2043_204395

theorem integer_roots_quadratic (a : ℤ) : 
  (∃ x y : ℤ, x^2 - a*x + 9*a = 0 ∧ y^2 - a*y + 9*a = 0 ∧ x ≠ y) ↔ 
  a ∈ ({100, -64, 48, -12, 36, 0} : Set ℤ) :=
sorry

end NUMINAMATH_CALUDE_integer_roots_quadratic_l2043_204395


namespace NUMINAMATH_CALUDE_delivery_distances_l2043_204346

/-- Represents the direction of travel --/
inductive Direction
  | North
  | South

/-- Represents a location relative to the supermarket --/
structure Location where
  distance : ℝ
  direction : Direction

/-- Calculates the distance between two locations --/
def distanceBetween (a b : Location) : ℝ :=
  match a.direction, b.direction with
  | Direction.North, Direction.North => abs (a.distance - b.distance)
  | Direction.South, Direction.South => abs (a.distance - b.distance)
  | _, _ => a.distance + b.distance

/-- Calculates the round trip distance to a location --/
def roundTripDistance (loc : Location) : ℝ :=
  2 * loc.distance

theorem delivery_distances (unitA unitB unitC : Location) 
  (hA : unitA = { distance := 30, direction := Direction.South })
  (hB : unitB = { distance := 50, direction := Direction.South })
  (hC : unitC = { distance := 15, direction := Direction.North }) :
  distanceBetween unitA unitC = 45 ∧ 
  roundTripDistance unitB + 3 * roundTripDistance unitC = 190 := by
  sorry


end NUMINAMATH_CALUDE_delivery_distances_l2043_204346


namespace NUMINAMATH_CALUDE_exists_m_n_satisfying_equation_l2043_204320

-- Define the "*" operation
def star_op (a b : ℤ) : ℤ :=
  if a = 0 ∨ b = 0 then
    max (a^2) (b^2)
  else
    (if a * b > 0 then 1 else -1) * (a^2 + b^2)

-- Theorem statement
theorem exists_m_n_satisfying_equation :
  ∃ (m n : ℤ), star_op (m - 1) (n + 2) = -2 :=
sorry

end NUMINAMATH_CALUDE_exists_m_n_satisfying_equation_l2043_204320


namespace NUMINAMATH_CALUDE_two_digit_sum_divisible_by_17_l2043_204350

/-- A function that reverses a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

/-- A predicate that checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_sum_divisible_by_17 :
  ∀ A : ℕ, is_two_digit A →
    (A + reverse_digits A) % 17 = 0 ↔ A = 89 ∨ A = 98 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_divisible_by_17_l2043_204350


namespace NUMINAMATH_CALUDE_no_perfect_squares_l2043_204370

theorem no_perfect_squares (x y z t : ℕ+) : 
  (x * y : ℤ) - (z * t : ℤ) = (x : ℤ) + y ∧ 
  (x : ℤ) + y = (z : ℤ) + t → 
  ¬(∃ (a c : ℕ+), (x * y : ℤ) = (a * a : ℤ) ∧ (z * t : ℤ) = (c * c : ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l2043_204370


namespace NUMINAMATH_CALUDE_triangle_angle_A_is_30_degrees_l2043_204376

theorem triangle_angle_A_is_30_degrees 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : a^2 - b^2 = Real.sqrt 3 * b * c)
  (h2 : Real.sin C = 2 * Real.sqrt 3 * Real.sin B)
  (h3 : 0 < A ∧ A < π)
  (h4 : 0 < B ∧ B < π)
  (h5 : 0 < C ∧ C < π)
  (h6 : A + B + C = π)
  (h7 : a / Real.sin A = b / Real.sin B)
  (h8 : b / Real.sin B = c / Real.sin C)
  : A = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_A_is_30_degrees_l2043_204376


namespace NUMINAMATH_CALUDE_largest_share_is_18000_l2043_204328

/-- Represents the profit share of a partner -/
structure Share where
  ratio : Nat
  amount : Nat

/-- Calculates the largest share given a total profit and a list of ratios -/
def largest_share (total_profit : Nat) (ratios : List Nat) : Nat :=
  let sum_ratios := ratios.sum
  let part_value := total_profit / sum_ratios
  (ratios.maximum.getD 0) * part_value

/-- The theorem stating that the largest share is $18,000 -/
theorem largest_share_is_18000 :
  largest_share 48000 [1, 2, 3, 4, 6] = 18000 := by
  sorry

end NUMINAMATH_CALUDE_largest_share_is_18000_l2043_204328


namespace NUMINAMATH_CALUDE_bush_height_after_two_years_l2043_204324

def bush_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

theorem bush_height_after_two_years 
  (h : bush_height 1 5 = 81) : 
  bush_height 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bush_height_after_two_years_l2043_204324
