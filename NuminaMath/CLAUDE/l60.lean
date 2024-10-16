import Mathlib

namespace NUMINAMATH_CALUDE_local_maximum_at_two_l60_6036

/-- The function f(x) defined as x(x-c)^2 --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) with respect to x --/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

/-- Theorem stating that the value of c for which f(x) has a local maximum at x=2 is 6 --/
theorem local_maximum_at_two (c : ℝ) : 
  (∀ x, x ≠ 2 → ∃ δ > 0, ∀ y, |y - 2| < δ → f c y ≤ f c 2) → c = 6 :=
sorry

end NUMINAMATH_CALUDE_local_maximum_at_two_l60_6036


namespace NUMINAMATH_CALUDE_cos_2x_derivative_l60_6093

theorem cos_2x_derivative (x : ℝ) : 
  deriv (λ x => Real.cos (2 * x)) x = -2 * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_derivative_l60_6093


namespace NUMINAMATH_CALUDE_range_of_a_l60_6022

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

def prop_p (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 1, ∀ y ∈ Set.Icc (-1) 1, x ≤ y → f a x ≥ f a y

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 1 ≤ 0

theorem range_of_a : 
  {a : ℝ | (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a)} = 
  Set.Iic (-2) ∪ Set.Icc 2 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l60_6022


namespace NUMINAMATH_CALUDE_canada_population_1998_l60_6099

/-- Proves that 30.3 million is equal to 30,300,000 --/
theorem canada_population_1998 : (30.3 : ℝ) * 1000000 = 30300000 := by
  sorry

end NUMINAMATH_CALUDE_canada_population_1998_l60_6099


namespace NUMINAMATH_CALUDE_f_equals_g_l60_6072

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 1
def g (t : ℝ) : ℝ := t^2 + 1

-- Theorem stating that f and g represent the same function
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l60_6072


namespace NUMINAMATH_CALUDE_james_spent_six_l60_6001

/-- Calculates the total amount spent given the cost of milk, cost of bananas, and sales tax rate. -/
def total_spent (milk_cost banana_cost tax_rate : ℚ) : ℚ :=
  let subtotal := milk_cost + banana_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Proves that James spent $6 given the costs and tax rate. -/
theorem james_spent_six :
  let milk_cost : ℚ := 3
  let banana_cost : ℚ := 2
  let tax_rate : ℚ := 1/5
  total_spent milk_cost banana_cost tax_rate = 6 := by
  sorry

#eval total_spent 3 2 (1/5)

end NUMINAMATH_CALUDE_james_spent_six_l60_6001


namespace NUMINAMATH_CALUDE_unique_solution_l60_6059

structure Grid :=
  (a b c : ℕ)
  (row_sum : ℕ)
  (col_sum : ℕ)

def is_valid_grid (g : Grid) : Prop :=
  g.row_sum = 9 ∧
  g.col_sum = 12 ∧
  g.a + g.b + g.c = g.row_sum ∧
  4 + g.a + 1 + g.b = g.col_sum ∧
  g.a + 2 + 6 = g.col_sum ∧
  3 + 1 + 6 + g.c = g.col_sum ∧
  g.b + 2 + g.c = g.row_sum

theorem unique_solution :
  ∃! g : Grid, is_valid_grid g ∧ g.a = 6 ∧ g.b = 5 ∧ g.c = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l60_6059


namespace NUMINAMATH_CALUDE_gcf_of_60_90_150_l60_6018

theorem gcf_of_60_90_150 : Nat.gcd 60 (Nat.gcd 90 150) = 30 := by sorry

end NUMINAMATH_CALUDE_gcf_of_60_90_150_l60_6018


namespace NUMINAMATH_CALUDE_group_size_is_seven_l60_6098

/-- The number of boxes one person can lift -/
def boxes_per_person : ℕ := 2

/-- The total number of boxes the group can hold -/
def total_boxes : ℕ := 14

/-- The number of people in the group -/
def group_size : ℕ := total_boxes / boxes_per_person

theorem group_size_is_seven : group_size = 7 := by
  sorry

end NUMINAMATH_CALUDE_group_size_is_seven_l60_6098


namespace NUMINAMATH_CALUDE_right_triangle_345_l60_6085

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_345 :
  is_right_triangle 3 4 5 ∧
  ¬is_right_triangle 1 2 3 ∧
  ¬is_right_triangle 2 3 4 ∧
  ¬is_right_triangle 4 5 6 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_345_l60_6085


namespace NUMINAMATH_CALUDE_fencing_cost_140m_perimeter_l60_6002

/-- The cost of fencing a rectangular plot -/
def fencing_cost (width : ℝ) (rate : ℝ) : ℝ :=
  let length : ℝ := width + 10
  let perimeter : ℝ := 2 * (length + width)
  rate * perimeter

theorem fencing_cost_140m_perimeter :
  ∃ (width : ℝ),
    (2 * (width + (width + 10)) = 140) ∧
    (fencing_cost width 6.5 = 910) := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_140m_perimeter_l60_6002


namespace NUMINAMATH_CALUDE_divisible_by_91_l60_6041

theorem divisible_by_91 (n : ℕ) : 91 ∣ (5^n * (5^n + 1) - 6^n * (3^n + 2^n)) :=
sorry

end NUMINAMATH_CALUDE_divisible_by_91_l60_6041


namespace NUMINAMATH_CALUDE_investment_ratio_l60_6081

/-- Represents the business investment scenario of Krishan and Nandan -/
structure BusinessInvestment where
  nandan_investment : ℝ
  nandan_time : ℝ
  krishan_investment_multiplier : ℝ
  total_gain : ℝ
  nandan_gain : ℝ

/-- The ratio of Krishan's investment to Nandan's investment is 4:1 -/
theorem investment_ratio (b : BusinessInvestment) : 
  b.nandan_time > 0 ∧ 
  b.nandan_investment > 0 ∧ 
  b.total_gain = 26000 ∧ 
  b.nandan_gain = 2000 ∧ 
  b.total_gain = b.nandan_gain + b.krishan_investment_multiplier * b.nandan_investment * (3 * b.nandan_time) →
  b.krishan_investment_multiplier = 4 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l60_6081


namespace NUMINAMATH_CALUDE_function_properties_l60_6060

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f x * f y = (f (x + y) + 2 * f (x - y)) / 3)
  (h2 : ∀ x : ℝ, f x ≠ 0) :
  (f 0 = 1) ∧ (∀ x : ℝ, f x = f (-x)) := by sorry

end NUMINAMATH_CALUDE_function_properties_l60_6060


namespace NUMINAMATH_CALUDE_cube_diff_prime_mod_six_l60_6080

theorem cube_diff_prime_mod_six (a b p : ℕ) : 
  0 < a → 0 < b → Prime p → a^3 - b^3 = p → p % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_diff_prime_mod_six_l60_6080


namespace NUMINAMATH_CALUDE_average_age_of_three_l60_6069

/-- Given the ages of Omi, Kimiko, and Arlette, prove their average age is 35 --/
theorem average_age_of_three (kimiko_age : ℕ) (omi_age : ℕ) (arlette_age : ℕ) 
  (h1 : kimiko_age = 28) 
  (h2 : omi_age = 2 * kimiko_age) 
  (h3 : arlette_age = 3 * kimiko_age / 4) : 
  (kimiko_age + omi_age + arlette_age) / 3 = 35 := by
  sorry

#check average_age_of_three

end NUMINAMATH_CALUDE_average_age_of_three_l60_6069


namespace NUMINAMATH_CALUDE_cos_330_degrees_l60_6076

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l60_6076


namespace NUMINAMATH_CALUDE_library_book_distribution_l60_6010

def number_of_distributions (total_books : ℕ) (min_in_library : ℕ) (min_checked_out : ℕ) : ℕ :=
  (total_books - min_in_library - min_checked_out + 1)

theorem library_book_distribution :
  number_of_distributions 8 2 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_library_book_distribution_l60_6010


namespace NUMINAMATH_CALUDE_snail_count_l60_6047

/-- The number of snails gotten rid of in Centerville -/
def snails_removed : ℕ := 3482

/-- The number of snails remaining in Centerville -/
def snails_remaining : ℕ := 8278

/-- The original number of snails in Centerville -/
def original_snails : ℕ := snails_removed + snails_remaining

theorem snail_count : original_snails = 11760 := by
  sorry

end NUMINAMATH_CALUDE_snail_count_l60_6047


namespace NUMINAMATH_CALUDE_smallest_n_with_divisibility_l60_6032

def is_divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem smallest_n_with_divisibility : ∃! N : ℕ, 
  N > 0 ∧ 
  (is_divisible N (2^2) ∨ is_divisible (N+1) (2^2) ∨ is_divisible (N+2) (2^2) ∨ is_divisible (N+3) (2^2)) ∧
  (is_divisible N (3^2) ∨ is_divisible (N+1) (3^2) ∨ is_divisible (N+2) (3^2) ∨ is_divisible (N+3) (3^2)) ∧
  (is_divisible N (5^2) ∨ is_divisible (N+1) (5^2) ∨ is_divisible (N+2) (5^2) ∨ is_divisible (N+3) (5^2)) ∧
  (is_divisible N (11^2) ∨ is_divisible (N+1) (11^2) ∨ is_divisible (N+2) (11^2) ∨ is_divisible (N+3) (11^2)) ∧
  (∀ M : ℕ, M < N → 
    ¬((is_divisible M (2^2) ∨ is_divisible (M+1) (2^2) ∨ is_divisible (M+2) (2^2) ∨ is_divisible (M+3) (2^2)) ∧
      (is_divisible M (3^2) ∨ is_divisible (M+1) (3^2) ∨ is_divisible (M+2) (3^2) ∨ is_divisible (M+3) (3^2)) ∧
      (is_divisible M (5^2) ∨ is_divisible (M+1) (5^2) ∨ is_divisible (M+2) (5^2) ∨ is_divisible (M+3) (5^2)) ∧
      (is_divisible M (11^2) ∨ is_divisible (M+1) (11^2) ∨ is_divisible (M+2) (11^2) ∨ is_divisible (M+3) (11^2)))) ∧
  N = 484 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_divisibility_l60_6032


namespace NUMINAMATH_CALUDE_stating_equilateral_triangle_condition_l60_6068

/-- 
A function that checks if a natural number n satisfies the condition
that sticks of lengths 1, 2, ..., n can form an equilateral triangle.
-/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- 
Theorem stating that sticks of lengths 1, 2, ..., n can form an equilateral triangle
if and only if n satisfies the condition defined in can_form_equilateral_triangle.
-/
theorem equilateral_triangle_condition (n : ℕ) :
  (∃ (a b c : ℕ), a + b + c = n * (n + 1) / 2 ∧ a = b ∧ b = c) ↔
  can_form_equilateral_triangle n :=
sorry

end NUMINAMATH_CALUDE_stating_equilateral_triangle_condition_l60_6068


namespace NUMINAMATH_CALUDE_equation_solutions_l60_6078

-- Define the property of being consecutive integers
def ConsecutiveIntegers (x y z : ℕ) : Prop :=
  y = x + 1 ∧ z = y + 1

-- Define the property of being consecutive even integers
def ConsecutiveEvenIntegers (x y z w : ℕ) : Prop :=
  y = x + 2 ∧ z = y + 2 ∧ w = z + 2 ∧ Even x

theorem equation_solutions :
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ ConsecutiveIntegers x y z ∧ x + y + z = 48) ∧
  (∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ ConsecutiveEvenIntegers x y z w ∧ x + y + z + w = 52) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l60_6078


namespace NUMINAMATH_CALUDE_equation_solution_l60_6004

theorem equation_solution : 
  ∃ x : ℝ, 
    (2.5 * ((3.6 * x * 2.50) / (0.12 * 0.09 * 0.5)) = 2000.0000000000002) ∧ 
    (abs (x - 0.48) < 0.00000000000001) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l60_6004


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l60_6030

theorem quadratic_equation_solution (m : ℝ) : 
  (m - 1 ≠ 0) → (m^2 - 3*m + 2 = 0) → (m = 2) := by
  sorry

#check quadratic_equation_solution

end NUMINAMATH_CALUDE_quadratic_equation_solution_l60_6030


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l60_6096

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 14 → a * b = 2562 → Nat.lcm a b = 183 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l60_6096


namespace NUMINAMATH_CALUDE_f_has_max_iff_a_ge_e_l60_6008

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ a then Real.log x else a / x

-- Theorem statement
theorem f_has_max_iff_a_ge_e (a : ℝ) :
  (∃ (M : ℝ), ∀ (x : ℝ), x > 0 → f a x ≤ M) ↔ a ≥ Real.exp 1 := by
  sorry

end

end NUMINAMATH_CALUDE_f_has_max_iff_a_ge_e_l60_6008


namespace NUMINAMATH_CALUDE_inscribed_squares_product_l60_6050

theorem inscribed_squares_product (a b : ℝ) : 
  (9 : ℝ).sqrt ^ 2 = 9 → 
  (16 : ℝ).sqrt ^ 2 = 16 → 
  a + b = (16 : ℝ).sqrt → 
  ((9 : ℝ).sqrt * Real.sqrt 2) ^ 2 = a ^ 2 + b ^ 2 → 
  a * b = -1 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_product_l60_6050


namespace NUMINAMATH_CALUDE_book_count_is_93_l60_6053

/-- Calculates the final number of books given the initial count and subsequent changes -/
def final_book_count (initial : ℕ) (given_away : ℕ) (received_later : ℕ) 
                     (traded_away : ℕ) (received_in_trade : ℕ) (additional : ℕ) : ℕ :=
  initial - given_away + received_later - traded_away + received_in_trade + additional

/-- Theorem stating that given the specific book counts and changes, the final count is 93 -/
theorem book_count_is_93 : 
  final_book_count 54 16 23 12 9 35 = 93 := by
  sorry

#eval final_book_count 54 16 23 12 9 35

end NUMINAMATH_CALUDE_book_count_is_93_l60_6053


namespace NUMINAMATH_CALUDE_cos_equality_angle_l60_6033

theorem cos_equality_angle (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (370 * π / 180) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_angle_l60_6033


namespace NUMINAMATH_CALUDE_milk_butterfat_percentage_l60_6012

theorem milk_butterfat_percentage : 
  ∀ (initial_volume initial_percentage added_volume final_volume final_percentage : ℝ),
  initial_volume > 0 →
  added_volume > 0 →
  initial_volume + added_volume = final_volume →
  initial_volume * initial_percentage + added_volume * (added_percentage / 100) = final_volume * final_percentage →
  initial_volume = 8 →
  initial_percentage = 0.4 →
  added_volume = 16 →
  final_volume = 24 →
  final_percentage = 0.2 →
  ∃ added_percentage : ℝ, added_percentage = 10 :=
by
  sorry

#check milk_butterfat_percentage

end NUMINAMATH_CALUDE_milk_butterfat_percentage_l60_6012


namespace NUMINAMATH_CALUDE_cosine_derivative_at_pi_over_two_l60_6066

theorem cosine_derivative_at_pi_over_two :
  deriv (fun x => Real.cos x) (π / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_derivative_at_pi_over_two_l60_6066


namespace NUMINAMATH_CALUDE_point_inside_circle_l60_6006

theorem point_inside_circle (r d : ℝ) (hr : r = 6) (hd : d = 4) :
  d < r → ∃ (P : ℝ × ℝ) (O : ℝ × ℝ), ‖P - O‖ = d ∧ P ∈ interior {x | ‖x - O‖ ≤ r} :=
by sorry

end NUMINAMATH_CALUDE_point_inside_circle_l60_6006


namespace NUMINAMATH_CALUDE_jacqueline_guavas_l60_6013

theorem jacqueline_guavas (plums apples given_away left : ℕ) (guavas : ℕ) : 
  plums = 16 → 
  apples = 21 → 
  given_away = 40 → 
  left = 15 → 
  plums + guavas + apples = given_away + left → 
  guavas = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_jacqueline_guavas_l60_6013


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l60_6064

/-- The line in 3D space --/
def line (t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + 2*t, -2 - 5*t, 3 - 2*t)

/-- The plane in 3D space --/
def plane (x y z : ℝ) : Prop :=
  x + 2*y - 5*z + 16 = 0

/-- The intersection point --/
def intersection_point : ℝ × ℝ × ℝ :=
  (3, -7, 1)

theorem intersection_point_is_unique :
  (∃! p : ℝ × ℝ × ℝ, ∃ t : ℝ, line t = p ∧ plane p.1 p.2.1 p.2.2) ∧
  (∃ t : ℝ, line t = intersection_point ∧ plane intersection_point.1 intersection_point.2.1 intersection_point.2.2) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l60_6064


namespace NUMINAMATH_CALUDE_triangle_reflection_area_l60_6075

/-- The area of the union of a triangle and its reflection --/
theorem triangle_reflection_area : 
  let A : ℝ × ℝ := (3, 4)
  let B : ℝ × ℝ := (5, -2)
  let C : ℝ × ℝ := (6, 2)
  let reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.1, 2 - p.2)
  let A' := reflect A
  let B' := reflect B
  let C' := reflect C
  let area (p q r : ℝ × ℝ) : ℝ := 
    (1/2) * abs (p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2))
  area A B C + area A' B' C' = 11 := by
sorry


end NUMINAMATH_CALUDE_triangle_reflection_area_l60_6075


namespace NUMINAMATH_CALUDE_sin_cos_identity_l60_6054

theorem sin_cos_identity : Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
                           Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l60_6054


namespace NUMINAMATH_CALUDE_mrs_hilt_apples_per_hour_l60_6058

/-- Given a total number of apples and hours, calculate the apples eaten per hour -/
def apples_per_hour (total_apples : ℕ) (total_hours : ℕ) : ℚ :=
  total_apples / total_hours

/-- Theorem: Mrs. Hilt ate 5 apples per hour -/
theorem mrs_hilt_apples_per_hour :
  apples_per_hour 15 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_apples_per_hour_l60_6058


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l60_6095

theorem sqrt_two_minus_one_power (n : ℕ) (hn : n > 0) :
  ∃ (k : ℕ), k > 1 ∧ (Real.sqrt 2 - 1) ^ n = Real.sqrt k - Real.sqrt (k - 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l60_6095


namespace NUMINAMATH_CALUDE_percent_relation_l60_6038

theorem percent_relation (a b c : ℝ) (h1 : c = 0.2 * a) (h2 : c = 0.1 * b) :
  b = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l60_6038


namespace NUMINAMATH_CALUDE_unique_divisor_sums_l60_6071

def divisor_sums (n : ℕ+) : Finset ℕ :=
  (Finset.powerset (Nat.divisors n.val)).image (λ s => s.sum id)

def target_sums : Finset ℕ := {4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 46, 48, 50, 54, 60}

theorem unique_divisor_sums (n : ℕ+) : divisor_sums n = target_sums → n = 45 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_sums_l60_6071


namespace NUMINAMATH_CALUDE_third_number_value_l60_6061

-- Define the proportion
def proportion (a b c d : ℚ) : Prop := a * d = b * c

-- State the theorem
theorem third_number_value : 
  ∃ (third_number : ℚ), 
    proportion (75/100) (6/5) third_number 8 ∧ third_number = 5 := by
  sorry

end NUMINAMATH_CALUDE_third_number_value_l60_6061


namespace NUMINAMATH_CALUDE_part1_part2_part3_l60_6042

-- Define the operation
def matrixOp (a b c d : ℚ) : ℚ := a * d - c * b

-- Theorem 1
theorem part1 : matrixOp (-3) (-2) 4 5 = -7 := by sorry

-- Theorem 2
theorem part2 : matrixOp 2 (-2 * x) 3 (-5 * x) = 2 → x = -1/2 := by sorry

-- Theorem 3
theorem part3 (x : ℚ) : 
  matrixOp (8 * m * x - 1) (-8/3 + 2 * x) (3/2) (-3) = matrixOp 6 (-1) (-n) x →
  m = -3/8 ∧ n = -7 := by sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l60_6042


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l60_6079

/-- The sum of 0.3̄, 0.04̄, and 0.005̄ is equal to 112386/296703 -/
theorem recurring_decimal_sum : 
  (1 : ℚ) / 3 + 4 / 99 + 5 / 999 = 112386 / 296703 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l60_6079


namespace NUMINAMATH_CALUDE_unique_colors_count_l60_6007

/-- The total number of unique colored pencils owned by Serenity, Jordan, and Alex -/
def total_unique_colors (serenity_colors jordan_colors alex_colors
                         serenity_jordan_shared serenity_alex_shared jordan_alex_shared
                         all_shared : ℕ) : ℕ :=
  serenity_colors + jordan_colors + alex_colors
  - (serenity_jordan_shared + serenity_alex_shared + jordan_alex_shared - 2 * all_shared)
  - all_shared

/-- Theorem stating the total number of unique colored pencils -/
theorem unique_colors_count :
  total_unique_colors 24 36 30 8 5 10 3 = 73 := by
  sorry

end NUMINAMATH_CALUDE_unique_colors_count_l60_6007


namespace NUMINAMATH_CALUDE_age_difference_l60_6065

/-- Given two people A and B, where B is currently 37 years old,
    and in 10 years A will be twice as old as B was 10 years ago,
    prove that A is currently 7 years older than B. -/
theorem age_difference (a b : ℕ) (h1 : b = 37) 
    (h2 : a + 10 = 2 * (b - 10)) : a - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l60_6065


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l60_6088

/-- Converts a number from base 5 to base 10 -/
def base5To10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The loot values in base 5 -/
def jewelry : List Nat := [4, 2, 1, 3]
def goldCoins : List Nat := [2, 2, 1, 3]
def rubbingAlcohol : List Nat := [4, 2, 1]

theorem pirate_loot_sum :
  base5To10 jewelry + base5To10 goldCoins + base5To10 rubbingAlcohol = 865 := by
  sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l60_6088


namespace NUMINAMATH_CALUDE_collinear_points_a_equals_4_l60_6049

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- Given three points A(a,2), B(5,1), and C(-4,2a) are collinear, prove that a = 4. -/
theorem collinear_points_a_equals_4 (a : ℝ) :
  collinear a 2 5 1 (-4) (2*a) → a = 4 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_a_equals_4_l60_6049


namespace NUMINAMATH_CALUDE_initial_players_count_l60_6082

theorem initial_players_count (players_quit : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let initial_players := 8
  let remaining_players := initial_players - players_quit
  have h1 : players_quit = 3 := by sorry
  have h2 : lives_per_player = 3 := by sorry
  have h3 : total_lives = 15 := by sorry
  have h4 : remaining_players * lives_per_player = total_lives := by sorry
  initial_players

#check initial_players_count

end NUMINAMATH_CALUDE_initial_players_count_l60_6082


namespace NUMINAMATH_CALUDE_sheila_weekly_earnings_l60_6028

/-- Represents Sheila's work schedule and earnings --/
structure SheilaWork where
  hourly_rate : ℕ
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  days_mon_wed_fri : ℕ
  days_tue_thu : ℕ

/-- Calculates Sheila's weekly earnings --/
def weekly_earnings (s : SheilaWork) : ℕ :=
  s.hourly_rate * (s.hours_mon_wed_fri * s.days_mon_wed_fri + s.hours_tue_thu * s.days_tue_thu)

/-- Theorem stating Sheila's weekly earnings --/
theorem sheila_weekly_earnings :
  ∃ (s : SheilaWork),
    s.hourly_rate = 13 ∧
    s.hours_mon_wed_fri = 8 ∧
    s.hours_tue_thu = 6 ∧
    s.days_mon_wed_fri = 3 ∧
    s.days_tue_thu = 2 ∧
    weekly_earnings s = 468 := by
  sorry

end NUMINAMATH_CALUDE_sheila_weekly_earnings_l60_6028


namespace NUMINAMATH_CALUDE_lending_interest_rate_l60_6003

/-- Proves that the lending interest rate is 6% given the specified conditions --/
theorem lending_interest_rate (borrowed_amount : ℕ) (borrowing_period : ℕ) 
  (borrowing_rate : ℚ) (gain_per_year : ℕ) (lending_rate : ℚ) : 
  borrowed_amount = 6000 →
  borrowing_period = 2 →
  borrowing_rate = 4 / 100 →
  gain_per_year = 120 →
  (borrowed_amount * borrowing_rate * borrowing_period + 
   borrowing_period * gain_per_year) / (borrowed_amount * borrowing_period) * 100 = lending_rate →
  lending_rate = 6 / 100 := by
sorry


end NUMINAMATH_CALUDE_lending_interest_rate_l60_6003


namespace NUMINAMATH_CALUDE_milk_price_calculation_l60_6055

/-- Calculates the final price of milk given wholesale price, markup percentage, and discount percentage -/
theorem milk_price_calculation (wholesale_price markup_percent discount_percent : ℝ) :
  wholesale_price = 4 →
  markup_percent = 25 →
  discount_percent = 5 →
  let retail_price := wholesale_price * (1 + markup_percent / 100)
  let final_price := retail_price * (1 - discount_percent / 100)
  final_price = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_calculation_l60_6055


namespace NUMINAMATH_CALUDE_count_integer_points_on_line_l60_6027

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- Check if a point is strictly between two other points -/
def strictly_between (p q r : IntPoint) : Prop :=
  (p.x < q.x ∧ q.x < r.x) ∨ (r.x < q.x ∧ q.x < p.x)

/-- The line passing through two points -/
def line_through (p q : IntPoint) (r : IntPoint) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- The main theorem -/
theorem count_integer_points_on_line :
  let A : IntPoint := ⟨3, 3⟩
  let B : IntPoint := ⟨120, 150⟩
  ∃! (points : Finset IntPoint),
    (∀ p ∈ points, line_through A B p ∧ strictly_between A p B) ∧
    points.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_points_on_line_l60_6027


namespace NUMINAMATH_CALUDE_cole_gum_count_l60_6070

/-- The number of people sharing the gum -/
def num_people : ℕ := 3

/-- The number of pieces of gum John has -/
def john_gum : ℕ := 54

/-- The number of pieces of gum Aubrey has -/
def aubrey_gum : ℕ := 0

/-- The number of pieces each person gets after sharing -/
def shared_gum : ℕ := 33

/-- Cole's initial number of pieces of gum -/
def cole_gum : ℕ := num_people * shared_gum - john_gum - aubrey_gum

theorem cole_gum_count : cole_gum = 45 := by
  sorry

end NUMINAMATH_CALUDE_cole_gum_count_l60_6070


namespace NUMINAMATH_CALUDE_prob_at_least_two_equals_result_l60_6005

-- Define the probabilities for each person hitting the target
def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.8
def prob_C : ℝ := 0.6

-- Define the probability of at least two people hitting the target
def prob_at_least_two : ℝ :=
  1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C) -
  (prob_A * (1 - prob_B) * (1 - prob_C) +
   (1 - prob_A) * prob_B * (1 - prob_C) +
   (1 - prob_A) * (1 - prob_B) * prob_C)

-- Theorem statement
theorem prob_at_least_two_equals_result :
  prob_at_least_two = 0.832 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_equals_result_l60_6005


namespace NUMINAMATH_CALUDE_brendan_rounds_won_all_l60_6062

/-- The number of rounds where Brendan won all matches in a kickboxing competition -/
def rounds_won_all (total_matches_won : ℕ) (matches_per_full_round : ℕ) (last_round_matches : ℕ) : ℕ :=
  ((total_matches_won - (last_round_matches / 2)) / matches_per_full_round)

/-- Theorem stating that Brendan won all matches in 2 rounds -/
theorem brendan_rounds_won_all :
  rounds_won_all 14 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_brendan_rounds_won_all_l60_6062


namespace NUMINAMATH_CALUDE_total_eyes_in_pond_l60_6073

/-- The number of snakes in the pond -/
def num_snakes : ℕ := 18

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

/-- The number of eyes each snake has -/
def eyes_per_snake : ℕ := 2

/-- The number of eyes each alligator has -/
def eyes_per_alligator : ℕ := 2

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := num_snakes * eyes_per_snake + num_alligators * eyes_per_alligator

theorem total_eyes_in_pond : total_eyes = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_eyes_in_pond_l60_6073


namespace NUMINAMATH_CALUDE_set_equality_solution_l60_6094

theorem set_equality_solution (x y : ℝ) : 
  ({x, y, x + y} : Set ℝ) = ({0, x^2, x*y} : Set ℝ) →
  ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_set_equality_solution_l60_6094


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l60_6025

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ - 2 = 0) → 
  (x₂^2 - 4*x₂ - 2 = 0) → 
  (x₁ + x₂ = 4) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l60_6025


namespace NUMINAMATH_CALUDE_first_number_in_sequence_l60_6045

def sequence_property (s : Fin 10 → ℕ) : Prop :=
  ∀ n : Fin 10, n.val ≥ 2 → s n = s (n - 1) * s (n - 2)

theorem first_number_in_sequence 
  (s : Fin 10 → ℕ) 
  (h_property : sequence_property s)
  (h_last_three : s 7 = 81 ∧ s 8 = 6561 ∧ s 9 = 43046721) :
  s 0 = 3486784401 :=
sorry

end NUMINAMATH_CALUDE_first_number_in_sequence_l60_6045


namespace NUMINAMATH_CALUDE_total_stamps_sold_l60_6092

theorem total_stamps_sold (color_stamps : ℕ) (bw_stamps : ℕ) 
  (h1 : color_stamps = 578833) 
  (h2 : bw_stamps = 523776) : 
  color_stamps + bw_stamps = 1102609 := by
  sorry

end NUMINAMATH_CALUDE_total_stamps_sold_l60_6092


namespace NUMINAMATH_CALUDE_dividend_divisor_quotient_remainder_problem_l60_6043

theorem dividend_divisor_quotient_remainder_problem 
  (y1 y2 z1 z2 r1 x1 x2 : ℤ)
  (hy1 : y1 = 2)
  (hy2 : y2 = 3)
  (hz1 : z1 = 3)
  (hz2 : z2 = 5)
  (hr1 : r1 = 1)
  (hx1 : x1 = 4)
  (hx2 : x2 = 6)
  (y : ℤ) (hy : y = 3*(y1 + y2) + 4)
  (z : ℤ) (hz : z = 2*z1^2 - z2)
  (r : ℤ) (hr : r = 3*r1 + 2)
  (x : ℤ) (hx : x = 2*x1*y1 - x2 + 10) :
  x = 20 ∧ y = 19 ∧ z = 13 ∧ r = 5 := by
sorry

end NUMINAMATH_CALUDE_dividend_divisor_quotient_remainder_problem_l60_6043


namespace NUMINAMATH_CALUDE_tuesday_pages_l60_6040

/-- Represents the number of pages read on each day of the week --/
structure PagesRead where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Represents the reading plan for the week --/
def ReadingPlan (total_pages : ℕ) (pages : PagesRead) : Prop :=
  pages.monday = 23 ∧
  pages.wednesday = 61 ∧
  pages.thursday = 12 ∧
  pages.friday = 2 * pages.thursday ∧
  total_pages = pages.monday + pages.tuesday + pages.wednesday + pages.thursday + pages.friday

theorem tuesday_pages (total_pages : ℕ) (pages : PagesRead) 
  (h : ReadingPlan total_pages pages) (h_total : total_pages = 158) : 
  pages.tuesday = 38 := by
  sorry

#check tuesday_pages

end NUMINAMATH_CALUDE_tuesday_pages_l60_6040


namespace NUMINAMATH_CALUDE_long_distance_call_cost_decrease_l60_6016

/-- The percent decrease in cost of a long-distance call --/
def percent_decrease (initial_cost final_cost : ℚ) : ℚ :=
  (initial_cost - final_cost) / initial_cost * 100

/-- Theorem: The percent decrease from 35 cents to 5 cents is approximately 86% --/
theorem long_distance_call_cost_decrease :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |percent_decrease (35/100) (5/100) - 86| < ε :=
sorry

end NUMINAMATH_CALUDE_long_distance_call_cost_decrease_l60_6016


namespace NUMINAMATH_CALUDE_sphere_radius_with_inscribed_box_l60_6020

theorem sphere_radius_with_inscribed_box (x y z r : ℝ) : 
  x > 0 → y > 0 → z > 0 → r > 0 →
  2 * (x * y + y * z + x * z) = 384 →
  4 * (x + y + z) = 112 →
  (2 * r) ^ 2 = x ^ 2 + y ^ 2 + z ^ 2 →
  r = 10 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_with_inscribed_box_l60_6020


namespace NUMINAMATH_CALUDE_lisa_marble_distribution_l60_6000

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisa_marble_distribution (num_friends : ℕ) (initial_marbles : ℕ)
  (h1 : num_friends = 14)
  (h2 : initial_marbles = 50) :
  min_additional_marbles num_friends initial_marbles = 55 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marble_distribution_l60_6000


namespace NUMINAMATH_CALUDE_prime_sum_product_l60_6086

theorem prime_sum_product (x₁ x₂ x₃ : ℕ) 
  (h_prime₁ : Nat.Prime x₁) 
  (h_prime₂ : Nat.Prime x₂) 
  (h_prime₃ : Nat.Prime x₃) 
  (h_sum : x₁ + x₂ + x₃ = 68) 
  (h_sum_prod : x₁*x₂ + x₁*x₃ + x₂*x₃ = 1121) : 
  x₁ * x₂ * x₃ = 1978 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_product_l60_6086


namespace NUMINAMATH_CALUDE_square_root_fraction_sum_l60_6046

theorem square_root_fraction_sum : 
  Real.sqrt (2/25 + 1/49 - 1/100) = 3/10 := by sorry

end NUMINAMATH_CALUDE_square_root_fraction_sum_l60_6046


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l60_6091

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4 * x^2 + y^2)).sqrt) / (x * y) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) (hx : x > 0) :
  let y := x * Real.sqrt 2
  (((x^2 + y^2) * (4 * x^2 + y^2)).sqrt) / (x * y) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l60_6091


namespace NUMINAMATH_CALUDE_unpainted_face_area_l60_6026

/-- A right circular cylinder with given dimensions -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The unpainted face created by slicing the cylinder -/
def UnpaintedFace (c : Cylinder) (arcAngle : ℝ) : ℝ := sorry

/-- Theorem stating the area of the unpainted face for the given cylinder and arc angle -/
theorem unpainted_face_area (c : Cylinder) (h1 : c.radius = 6) (h2 : c.height = 8) (h3 : arcAngle = 2 * π / 3) :
  UnpaintedFace c arcAngle = 16 * π + 27 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_face_area_l60_6026


namespace NUMINAMATH_CALUDE_counterexample_exists_l60_6039

theorem counterexample_exists (h : ∀ a b : ℝ, a > -b) : 
  ∃ a b : ℝ, a > -b ∧ (1/a) + (1/b) ≤ 0 := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l60_6039


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l60_6021

theorem single_elimination_tournament_games (initial_teams : ℕ) (preliminary_games : ℕ) 
  (eliminated_teams : ℕ) (h1 : initial_teams = 24) (h2 : preliminary_games = 4) 
  (h3 : eliminated_teams = 4) :
  preliminary_games + (initial_teams - eliminated_teams - 1) = 23 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l60_6021


namespace NUMINAMATH_CALUDE_broken_line_isoperimetric_inequality_l60_6048

/-- A non-self-intersecting broken line in a half-plane -/
structure BrokenLine where
  length : ℝ
  area : ℝ
  nonSelfIntersecting : Prop
  endsOnBoundary : Prop

/-- The isoperimetric inequality for the broken line -/
theorem broken_line_isoperimetric_inequality (b : BrokenLine) :
  b.area ≤ b.length^2 / (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_broken_line_isoperimetric_inequality_l60_6048


namespace NUMINAMATH_CALUDE_soap_cost_two_years_l60_6044

-- Define the cost of one bar of soap
def cost_per_bar : ℕ := 4

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Define the number of years
def years : ℕ := 2

-- Define the function to calculate total cost
def total_cost (cost_per_bar months_per_year years : ℕ) : ℕ :=
  cost_per_bar * months_per_year * years

-- Theorem statement
theorem soap_cost_two_years :
  total_cost cost_per_bar months_per_year years = 96 := by
  sorry

end NUMINAMATH_CALUDE_soap_cost_two_years_l60_6044


namespace NUMINAMATH_CALUDE_regular_polygon_side_length_l60_6067

theorem regular_polygon_side_length 
  (n : ℕ) 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 48) 
  (h₂ : a₂ = 55) 
  (h₃ : n > 2) 
  (h₄ : (n * a₃^2) / (4 * Real.tan (π / n)) = 
        (n * a₁^2) / (4 * Real.tan (π / n)) + 
        (n * a₂^2) / (4 * Real.tan (π / n))) : 
  a₃ = 73 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_side_length_l60_6067


namespace NUMINAMATH_CALUDE_cylinder_sphere_area_ratio_l60_6009

/-- A cylinder with a square cross-section and height equal to the diameter of a sphere -/
structure SquareCylinder where
  radius : ℝ
  height : ℝ
  isSquare : height = 2 * radius

/-- The sphere with diameter equal to the cylinder's height -/
structure MatchingSphere where
  radius : ℝ

/-- The ratio of the total surface area of the cylinder to the surface area of the sphere is 3:2 -/
theorem cylinder_sphere_area_ratio (c : SquareCylinder) (s : MatchingSphere) 
  (h : c.radius = s.radius) : 
  (2 * c.radius * c.radius + 4 * c.radius * c.height) / (4 * π * s.radius ^ 2) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_sphere_area_ratio_l60_6009


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l60_6087

theorem circle_diameter_from_area :
  ∀ (A : ℝ) (d : ℝ),
    A = 225 * Real.pi →
    d = 2 * Real.sqrt (A / Real.pi) →
    d = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l60_6087


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l60_6015

/-- The eccentricity of an ellipse with equation x^2 + y^2/4 = 1 is √3/2 -/
theorem ellipse_eccentricity : 
  let e : ℝ := (Real.sqrt 3) / 2
  ∀ x y : ℝ, x^2 + y^2/4 = 1 → e = (Real.sqrt (4 - 1)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l60_6015


namespace NUMINAMATH_CALUDE_no_guaranteed_win_strategy_l60_6011

/-- Represents a game state with the current number on the board -/
structure GameState where
  number : ℕ

/-- Represents a player's move, adding a digit to the number -/
inductive Move
| PrependDigit (d : ℕ) : Move
| AppendDigit (d : ℕ) : Move
| InsertDigit (d : ℕ) (pos : ℕ) : Move

/-- Apply a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Bool :=
  sorry

/-- Theorem stating that no player can guarantee a win -/
theorem no_guaranteed_win_strategy :
  ∀ (strategy : GameState → Move),
  ∃ (opponent_moves : List Move),
  let final_state := opponent_moves.foldl applyMove ⟨7⟩
  ¬ isPerfectSquare final_state.number :=
sorry

end NUMINAMATH_CALUDE_no_guaranteed_win_strategy_l60_6011


namespace NUMINAMATH_CALUDE_inequality_proof_l60_6017

theorem inequality_proof (x y : ℝ) (h : x * y < 0) :
  x^4 / y^4 + y^4 / x^4 - x^2 / y^2 - y^2 / x^2 + x / y + y / x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l60_6017


namespace NUMINAMATH_CALUDE_speed_ratio_of_perpendicular_paths_l60_6034

/-- The ratio of speeds of two objects moving along perpendicular paths -/
theorem speed_ratio_of_perpendicular_paths
  (vA vB : ℝ) -- Speeds of objects A and B
  (h1 : vA > 0 ∧ vB > 0) -- Both speeds are positive
  (h2 : ∃ t1 : ℝ, t1 > 0 ∧ t1 * vA = |700 - t1 * vB|) -- Equidistant at time t1
  (h3 : ∃ t2 : ℝ, t2 > t1 ∧ t2 * vA = |700 - t2 * vB|) -- Equidistant at time t2 > t1
  : vA / vB = 6 / 7 :=
sorry

end NUMINAMATH_CALUDE_speed_ratio_of_perpendicular_paths_l60_6034


namespace NUMINAMATH_CALUDE_trail_mix_weight_l60_6063

/-- The weight of peanuts in pounds -/
def peanuts : ℝ := 0.16666666666666666

/-- The weight of chocolate chips in pounds -/
def chocolate_chips : ℝ := 0.16666666666666666

/-- The weight of raisins in pounds -/
def raisins : ℝ := 0.08333333333333333

/-- The total weight of trail mix in pounds -/
def total_weight : ℝ := peanuts + chocolate_chips + raisins

/-- Theorem stating that the total weight of trail mix is equal to 0.41666666666666663 pounds -/
theorem trail_mix_weight : total_weight = 0.41666666666666663 := by sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l60_6063


namespace NUMINAMATH_CALUDE_unique_solution_l60_6077

/-- A function from positive natural numbers to positive natural numbers -/
def PositiveNatFunction := ℕ+ → ℕ+

/-- The property that f(x + y f(x)) = x f(y + 1) for all x, y ∈ ℕ⁺ -/
def SatisfiesEquation (f : PositiveNatFunction) : Prop :=
  ∀ x y : ℕ+, f (x + y * f x) = x * f (y + 1)

/-- Theorem stating that if a function satisfies the equation, it must be the identity function -/
theorem unique_solution (f : PositiveNatFunction) (h : SatisfiesEquation f) :
  ∀ x : ℕ+, f x = x := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l60_6077


namespace NUMINAMATH_CALUDE_inequality_solution_set_l60_6023

theorem inequality_solution_set (x : ℝ) : (2 * x - 1 ≤ 3) ↔ (x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l60_6023


namespace NUMINAMATH_CALUDE_lumberjack_firewood_l60_6097

/-- Calculates the total number of firewood pieces produced by a lumberjack --/
theorem lumberjack_firewood (trees : ℕ) (logs_per_tree : ℕ) (pieces_per_log : ℕ) 
  (h1 : logs_per_tree = 4)
  (h2 : pieces_per_log = 5)
  (h3 : trees = 25) :
  trees * logs_per_tree * pieces_per_log = 500 := by
  sorry

#check lumberjack_firewood

end NUMINAMATH_CALUDE_lumberjack_firewood_l60_6097


namespace NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l60_6052

theorem tangent_line_to_logarithmic_curve (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ a * x = 1 + Real.log x ∧ 
    ∀ y : ℝ, y > 0 → a * y ≤ 1 + Real.log y) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l60_6052


namespace NUMINAMATH_CALUDE_polynomial_solution_l60_6037

theorem polynomial_solution (a : ℝ) (ha : a ≠ -1) 
  (h : a^5 + 5*a^4 + 10*a^3 + 3*a^2 - 9*a - 6 = 0) : 
  (a + 1)^3 = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_solution_l60_6037


namespace NUMINAMATH_CALUDE_element_in_set_l60_6084

def U : Set ℕ := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set ℕ) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l60_6084


namespace NUMINAMATH_CALUDE_min_digits_of_m_l60_6031

theorem min_digits_of_m (n : ℤ) : 
  let m := (n - 1001) * (n - 2001) * (n - 2002) * (n - 3001) * (n - 3002) * (n - 3003)
  m > 0 → m ≥ 10^10 :=
by sorry

end NUMINAMATH_CALUDE_min_digits_of_m_l60_6031


namespace NUMINAMATH_CALUDE_katies_soccer_game_granola_boxes_l60_6089

/-- Given the number of kids, granola bars per kid, and bars per box, 
    calculate the number of boxes needed. -/
def boxes_needed (num_kids : ℕ) (bars_per_kid : ℕ) (bars_per_box : ℕ) : ℕ :=
  (num_kids * bars_per_kid + bars_per_box - 1) / bars_per_box

/-- Prove that for Katie's soccer game scenario, 5 boxes are needed. -/
theorem katies_soccer_game_granola_boxes : 
  boxes_needed 30 2 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_katies_soccer_game_granola_boxes_l60_6089


namespace NUMINAMATH_CALUDE_f_inequality_l60_6014

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

theorem f_inequality : f (Real.sqrt 3 / 2) > f (Real.sqrt 6 / 2) ∧ f (Real.sqrt 6 / 2) > f (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l60_6014


namespace NUMINAMATH_CALUDE_parabola_c_value_l60_6051

/-- A parabola with equation y = x^2 + bx + c passes through points (2,3) and (5,6) -/
def parabola_through_points (b c : ℝ) : Prop :=
  3 = 2^2 + 2*b + c ∧ 6 = 5^2 + 5*b + c

/-- The theorem stating that c = -13 for the given parabola -/
theorem parabola_c_value : ∃ b : ℝ, parabola_through_points b (-13) := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l60_6051


namespace NUMINAMATH_CALUDE_salt_fraction_in_solution_l60_6074

theorem salt_fraction_in_solution (salt_weight water_weight : ℚ) :
  salt_weight = 6 → water_weight = 30 →
  salt_weight / (salt_weight + water_weight) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_salt_fraction_in_solution_l60_6074


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l60_6035

theorem cubic_sum_problem (a b c : ℝ) 
  (sum_condition : a + b + c = 7)
  (product_sum_condition : a * b + a * c + b * c = 9)
  (product_condition : a * b * c = -18) :
  a^3 + b^3 + c^3 = 100 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l60_6035


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_repeated_digits_l60_6090

/-- The number of digits available (0 to 9) -/
def num_digits : ℕ := 10

/-- The number of digits in the numbers we're considering -/
def num_places : ℕ := 3

/-- The total number of possible three-digit numbers -/
def total_numbers : ℕ := 900

/-- The number of three-digit numbers without repeated digits -/
def non_repeating_numbers : ℕ := 9 * 9 * 8

theorem three_digit_numbers_with_repeated_digits : 
  total_numbers - non_repeating_numbers = 252 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_repeated_digits_l60_6090


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l60_6083

theorem gain_percent_calculation (C S : ℝ) (h : C > 0) :
  50 * C = 15 * S →
  (S - C) / C * 100 = 233.33 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l60_6083


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_24hour_pm_l60_6024

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours ≥ 0 ∧ hours ≤ 23
  minute_valid : minutes ≥ 0 ∧ minutes ≤ 59

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits in a Time24 -/
def sumOfDigitsTime24 (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- Checks if a Time24 is between 12:00 and 23:59 -/
def isBetween12And2359 (t : Time24) : Prop :=
  t.hours ≥ 12 ∧ t.hours ≤ 23

theorem largest_sum_of_digits_24hour_pm :
  ∃ (t : Time24), isBetween12And2359 t ∧
    ∀ (t' : Time24), isBetween12And2359 t' →
      sumOfDigitsTime24 t' ≤ sumOfDigitsTime24 t ∧
      sumOfDigitsTime24 t = 24 :=
sorry

end NUMINAMATH_CALUDE_largest_sum_of_digits_24hour_pm_l60_6024


namespace NUMINAMATH_CALUDE_find_A_l60_6056

theorem find_A : ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ A * 100 + 30 + B - 41 = 591 ∧ A = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l60_6056


namespace NUMINAMATH_CALUDE_population_decreases_below_threshold_l60_6057

/-- The annual decrease rate of the population -/
def decrease_rate : ℝ := 0.5

/-- The threshold percentage of the initial population -/
def threshold : ℝ := 0.05

/-- The number of years it takes for the population to decrease below the threshold -/
def years_to_threshold : ℕ := 5

/-- The function that calculates the population after a given number of years -/
def population_after_years (initial_population : ℝ) (years : ℕ) : ℝ :=
  initial_population * (decrease_rate ^ years)

theorem population_decreases_below_threshold :
  ∀ initial_population : ℝ,
  initial_population > 0 →
  population_after_years initial_population years_to_threshold < threshold * initial_population ∧
  population_after_years initial_population (years_to_threshold - 1) ≥ threshold * initial_population :=
by sorry

end NUMINAMATH_CALUDE_population_decreases_below_threshold_l60_6057


namespace NUMINAMATH_CALUDE_happy_dictionary_problem_l60_6019

theorem happy_dictionary_problem (a b : ℤ) (c : ℚ) : 
  (∀ n : ℤ, n > 0 → a ≤ n) → 
  (∀ n : ℤ, n < 0 → n ≤ b) → 
  (∀ q : ℚ, q ≠ 0 → |c| ≤ |q|) → 
  a - b + c = 2 := by
sorry

end NUMINAMATH_CALUDE_happy_dictionary_problem_l60_6019


namespace NUMINAMATH_CALUDE_floor_e_equals_two_l60_6029

theorem floor_e_equals_two : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_e_equals_two_l60_6029
