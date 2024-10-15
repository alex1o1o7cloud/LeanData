import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l2422_242257

theorem problem_statement (f : ℕ → ℤ) (a b : ℤ) 
  (h1 : f 1 = 7) 
  (h2 : f 2 = 11)
  (h3 : ∀ x, f x = a * x^2 + b * x + 3) :
  f 3 = 15 := 
sorry

end NUMINAMATH_GPT_problem_statement_l2422_242257


namespace NUMINAMATH_GPT_true_proposition_among_choices_l2422_242267

theorem true_proposition_among_choices (p q : Prop) (hp : p) (hq : ¬ q) :
  p ∧ ¬ q :=
by
  sorry

end NUMINAMATH_GPT_true_proposition_among_choices_l2422_242267


namespace NUMINAMATH_GPT_sum_of_intersection_coordinates_l2422_242207

noncomputable def h : ℝ → ℝ := sorry

theorem sum_of_intersection_coordinates : 
  (∃ a b : ℝ, h a = h (a + 2) ∧ h 1 = 3 ∧ h (-1) = 3 ∧ a = -1 ∧ b = 3) → -1 + 3 = 2 :=
by
  intro h_assumptions
  sorry

end NUMINAMATH_GPT_sum_of_intersection_coordinates_l2422_242207


namespace NUMINAMATH_GPT_intersection_A_B_l2422_242223

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | 0 < x ∧ x ≤ 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2422_242223


namespace NUMINAMATH_GPT_range_a_l2422_242286

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∃ x : ℝ, |x - a| + |x - 1| ≤ 3

theorem range_a (a : ℝ) : range_of_a a → -2 ≤ a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_a_l2422_242286


namespace NUMINAMATH_GPT_find_rate_percent_l2422_242238

-- Given conditions as definitions
def SI : ℕ := 128
def P : ℕ := 800
def T : ℕ := 4

-- Define the formula for Simple Interest
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Define the rate percent we need to prove
def rate_percent : ℕ := 4

-- The theorem statement we need to prove
theorem find_rate_percent (h1 : simple_interest P rate_percent T = SI) : rate_percent = 4 := 
by sorry

end NUMINAMATH_GPT_find_rate_percent_l2422_242238


namespace NUMINAMATH_GPT_average_of_three_numbers_l2422_242222

theorem average_of_three_numbers
  (a b c : ℕ)
  (h1 : 2 * a + b + c = 130)
  (h2 : a + 2 * b + c = 138)
  (h3 : a + b + 2 * c = 152) :
  (a + b + c) / 3 = 35 :=
by
  sorry

end NUMINAMATH_GPT_average_of_three_numbers_l2422_242222


namespace NUMINAMATH_GPT_bacteria_mass_at_4pm_l2422_242272

theorem bacteria_mass_at_4pm 
  (r s t u v w : ℝ)
  (x y z : ℝ)
  (h1 : x = 10.0 * (1 + r))
  (h2 : y = 15.0 * (1 + s))
  (h3 : z = 8.0 * (1 + t))
  (h4 : 28.9 = x * (1 + u))
  (h5 : 35.5 = y * (1 + v))
  (h6 : 20.1 = z * (1 + w)) :
  x = 28.9 / (1 + u) ∧ y = 35.5 / (1 + v) ∧ z = 20.1 / (1 + w) :=
by
  sorry

end NUMINAMATH_GPT_bacteria_mass_at_4pm_l2422_242272


namespace NUMINAMATH_GPT_smallest_divisor_l2422_242229

-- Define the given number and the subtracting number
def original_num : ℕ := 378461
def subtract_num : ℕ := 5

-- Define the resulting number after subtraction
def resulting_num : ℕ := original_num - subtract_num

-- Theorem stating that 47307 is the smallest divisor greater than 5 of 378456
theorem smallest_divisor : ∃ d: ℕ, d > 5 ∧ d ∣ resulting_num ∧ ∀ x: ℕ, x > 5 → x ∣ resulting_num → d ≤ x := 
sorry

end NUMINAMATH_GPT_smallest_divisor_l2422_242229


namespace NUMINAMATH_GPT_max_profit_is_45_6_l2422_242246

noncomputable def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
noncomputable def profit_B (x : ℝ) : ℝ := 2 * x

noncomputable def total_profit (x : ℝ) : ℝ :=
  profit_A x + profit_B (15 - x)

theorem max_profit_is_45_6 : 
  ∃ x, 0 ≤ x ∧ x ≤ 15 ∧ total_profit x = 45.6 :=
by
  sorry

end NUMINAMATH_GPT_max_profit_is_45_6_l2422_242246


namespace NUMINAMATH_GPT_capital_formula_minimum_m_l2422_242212

-- Define initial conditions
def initial_capital : ℕ := 50000  -- in thousand yuan
def annual_growth_rate : ℝ := 0.5
def submission_amount : ℕ := 10000  -- in thousand yuan

-- Define remaining capital after nth year
noncomputable def remaining_capital (n : ℕ) : ℝ :=
  4500 * (3 / 2)^(n - 1) + 2000  -- in thousand yuan

-- Prove the formula for a_n
theorem capital_formula (n : ℕ) : 
  remaining_capital n = 4500 * (3 / 2)^(n - 1) + 2000 := 
by
  sorry

-- Prove the minimum value of m for which a_m > 30000
theorem minimum_m (m : ℕ) : 
  remaining_capital m > 30000 ↔ m ≥ 6 := 
by
  sorry

end NUMINAMATH_GPT_capital_formula_minimum_m_l2422_242212


namespace NUMINAMATH_GPT_calculation_correct_l2422_242265

theorem calculation_correct :
  15 * ( (1/3 : ℚ) + (1/4) + (1/6) )⁻¹ = 20 := sorry

end NUMINAMATH_GPT_calculation_correct_l2422_242265


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_solve_equation3_l2422_242227

-- For equation x^2 + 2x = 5
theorem solve_equation1 (x : ℝ) : x^2 + 2 * x = 5 ↔ (x = -1 + Real.sqrt 6) ∨ (x = -1 - Real.sqrt 6) :=
sorry

-- For equation x^2 - 2x - 1 = 0
theorem solve_equation2 (x : ℝ) : x^2 - 2 * x - 1 = 0 ↔ (x = 1 + Real.sqrt 2) ∨ (x = 1 - Real.sqrt 2) :=
sorry

-- For equation 2x^2 + 3x - 5 = 0
theorem solve_equation3 (x : ℝ) : 2 * x^2 + 3 * x - 5 = 0 ↔ (x = -5 / 2) ∨ (x = 1) :=
sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_solve_equation3_l2422_242227


namespace NUMINAMATH_GPT_rectangle_area_perimeter_l2422_242276

theorem rectangle_area_perimeter (a b : ℝ) (h₁ : a * b = 6) (h₂ : a + b = 6) : a^2 + b^2 = 24 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_perimeter_l2422_242276


namespace NUMINAMATH_GPT_intercepts_correct_l2422_242254

-- Define the equation of the line
def line_eq (x y : ℝ) := 5 * x - 2 * y - 10 = 0

-- Define the intercepts
def x_intercept : ℝ := 2
def y_intercept : ℝ := -5

-- Prove that the intercepts are as stated
theorem intercepts_correct :
  (∃ x, line_eq x 0 ∧ x = x_intercept) ∧
  (∃ y, line_eq 0 y ∧ y = y_intercept) :=
by
  sorry

end NUMINAMATH_GPT_intercepts_correct_l2422_242254


namespace NUMINAMATH_GPT_largest_of_four_consecutive_integers_with_product_840_l2422_242287

theorem largest_of_four_consecutive_integers_with_product_840 
  (a b c d : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c + 1 = d) (h_pos : 0 < a) (h_prod : a * b * c * d = 840) : d = 7 :=
sorry

end NUMINAMATH_GPT_largest_of_four_consecutive_integers_with_product_840_l2422_242287


namespace NUMINAMATH_GPT_simon_change_l2422_242296

def pansy_price : ℝ := 2.50
def pansy_count : ℕ := 5
def hydrangea_price : ℝ := 12.50
def hydrangea_count : ℕ := 1
def petunia_price : ℝ := 1.00
def petunia_count : ℕ := 5
def discount_rate : ℝ := 0.10
def initial_payment : ℝ := 50.00

theorem simon_change : 
  let total_cost := (pansy_count * pansy_price) + (hydrangea_count * hydrangea_price) + (petunia_count * petunia_price)
  let discount := total_cost * discount_rate
  let cost_after_discount := total_cost - discount
  let change := initial_payment - cost_after_discount
  change = 23.00 :=
by
  sorry

end NUMINAMATH_GPT_simon_change_l2422_242296


namespace NUMINAMATH_GPT_polynomial_remainder_l2422_242264

theorem polynomial_remainder :
  (4 * (2.5 : ℝ)^5 - 9 * (2.5 : ℝ)^4 + 7 * (2.5 : ℝ)^2 - 2.5 - 35 = 45.3125) :=
by sorry

end NUMINAMATH_GPT_polynomial_remainder_l2422_242264


namespace NUMINAMATH_GPT_greatest_3_digit_base_8_divisible_by_7_l2422_242217

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end NUMINAMATH_GPT_greatest_3_digit_base_8_divisible_by_7_l2422_242217


namespace NUMINAMATH_GPT_part1_part2_part3_l2422_242211

variable {α : Type} [LinearOrderedField α]

noncomputable def f (x : α) : α := sorry  -- as we won't define it explicitly, we use sorry

axiom f_conditions : ∀ (u v : α), - 1 ≤ u ∧ u ≤ 1 ∧ - 1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v|
axiom f_endpoints : f (-1 : α) = 0 ∧ f (1 : α) = 0

theorem part1 (x : α) (hx : -1 ≤ x ∧ x ≤ 1) : x - 1 ≤ f x ∧ f x ≤ 1 - x := by
  have hf : ∀ (u v : α), -1 ≤ u ∧ u ≤ 1 ∧ -1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v| := f_conditions
  sorry

theorem part2 (u v : α) (huv : -1 ≤ u ∧ u ≤ 1 ∧ -1 ≤ v ∧ v ≤ 1) : |f u - f v| ≤ 1 := by
  have hf : ∀ (u v : α), -1 ≤ u ∧ u ≤ 1 ∧ -1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v| := f_conditions
  sorry

theorem part3 : ¬ ∃ (f : α → α), (∀ (u v : α), - 1 ≤ u ∧ u ≤ 1 ∧ - 1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v| ∧ f (-1 : α) = 0 ∧ f (1 : α) = 0 ∧
  (∀ (x : α), - 1 ≤ x ∧ x ≤ 1 → f (- x) = - f x) ∧ -- odd function condition
  (∀ (u v : α), 0 ≤ u ∧ u ≤ 1/2 ∧ 0 ≤ v ∧ v ≤ 1/2 → |f u - f v| < |u - v|) ∧
  (∀ (u v : α), 1/2 ≤ u ∧ u ≤ 1 ∧ 1/2 ≤ v ∧ v ≤ 1 → |f u - f v| = |u - v|)) := by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l2422_242211


namespace NUMINAMATH_GPT_smallest_unreachable_integer_l2422_242298

/-- The smallest positive integer that cannot be expressed in the form (2^a - 2^b) / (2^c - 2^d) where a, b, c, d are non-negative integers is 11. -/
theorem smallest_unreachable_integer : 
  ∀ (a b c d : ℕ), 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
  ∃ (n : ℕ), n = 11 ∧ ¬ ∃ (a b c d : ℕ), (2^a - 2^b) / (2^c - 2^d) = n :=
by
  sorry

end NUMINAMATH_GPT_smallest_unreachable_integer_l2422_242298


namespace NUMINAMATH_GPT_area_inequality_l2422_242260

variable {a b c : ℝ} (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a)

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) : ℝ :=
  let p := semiperimeter a b c
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

theorem area_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  (2 * (area a b c h))^3 < (a * b * c)^2 := sorry

end NUMINAMATH_GPT_area_inequality_l2422_242260


namespace NUMINAMATH_GPT_tom_read_chapters_l2422_242242

theorem tom_read_chapters (chapters pages: ℕ) (h1: pages = 8 * chapters) (h2: pages = 24):
  chapters = 3 :=
by
  sorry

end NUMINAMATH_GPT_tom_read_chapters_l2422_242242


namespace NUMINAMATH_GPT_intersection_correct_l2422_242270

open Set

def M : Set ℤ := {-1, 3, 5}
def N : Set ℤ := {-1, 0, 1, 2, 3}
def MN_intersection : Set ℤ := {-1, 3}

theorem intersection_correct : M ∩ N = MN_intersection := by
  sorry

end NUMINAMATH_GPT_intersection_correct_l2422_242270


namespace NUMINAMATH_GPT_quadratic_equation_unique_solution_l2422_242249

theorem quadratic_equation_unique_solution
  (a c : ℝ)
  (h_discriminant : 100 - 4 * a * c = 0)
  (h_sum : a + c = 12)
  (h_lt : a < c) :
  (a, c) = (6 - Real.sqrt 11, 6 + Real.sqrt 11) :=
sorry

end NUMINAMATH_GPT_quadratic_equation_unique_solution_l2422_242249


namespace NUMINAMATH_GPT_y_coordinate_of_C_range_l2422_242282

noncomputable def A : ℝ × ℝ := (0, 2)

def is_on_parabola (P : ℝ × ℝ) : Prop := (P.2)^2 = P.1 + 4

def is_perpendicular (A B C : ℝ × ℝ) : Prop := 
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  k_AB * k_BC = -1

def range_of_y_C (y_C : ℝ) : Prop := y_C ≤ 0 ∨ y_C ≥ 4

theorem y_coordinate_of_C_range (B C : ℝ × ℝ)
  (hB : is_on_parabola B) (hC : is_on_parabola C) (h_perpendicular : is_perpendicular A B C) : 
  range_of_y_C (C.2) :=
sorry

end NUMINAMATH_GPT_y_coordinate_of_C_range_l2422_242282


namespace NUMINAMATH_GPT_not_possible_total_l2422_242240

-- Definitions
variables (d r : ℕ)

-- Theorem to prove that 58 cannot be expressed as 26d + 3r
theorem not_possible_total : ¬∃ (d r : ℕ), 26 * d + 3 * r = 58 :=
sorry

end NUMINAMATH_GPT_not_possible_total_l2422_242240


namespace NUMINAMATH_GPT_ratio_y_share_to_total_l2422_242243

theorem ratio_y_share_to_total
  (total_profit : ℝ)
  (diff_share : ℝ)
  (h_total : total_profit = 800)
  (h_diff : diff_share = 160) :
  ∃ (a b : ℝ), (b / (a + b) = 2 / 5) ∧ (|a - b| = (a + b) / 5) :=
by
  sorry

end NUMINAMATH_GPT_ratio_y_share_to_total_l2422_242243


namespace NUMINAMATH_GPT_geometric_sequence_a7_l2422_242277

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a1 : a 1 = 2) (h_a3 : a 3 = 4) : a 7 = 16 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a7_l2422_242277


namespace NUMINAMATH_GPT_range_of_m_l2422_242208

theorem range_of_m (m : ℝ) : 
  (∀ x : ℤ, (x > 3 - m) ∧ (x ≤ 5) ↔ (1 ≤ x ∧ x ≤ 5)) →
  (2 < m ∧ m ≤ 3) := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2422_242208


namespace NUMINAMATH_GPT_sum_of_first_9_primes_l2422_242228

theorem sum_of_first_9_primes : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23) = 100 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_first_9_primes_l2422_242228


namespace NUMINAMATH_GPT_fraction_of_employees_laid_off_l2422_242209

theorem fraction_of_employees_laid_off
    (total_employees : ℕ)
    (salary_per_employee : ℕ)
    (total_payment_after_layoffs : ℕ)
    (h1 : total_employees = 450)
    (h2 : salary_per_employee = 2000)
    (h3 : total_payment_after_layoffs = 600000) :
    (total_employees * salary_per_employee - total_payment_after_layoffs) / (total_employees * salary_per_employee) = 1 / 3 := 
by
    sorry

end NUMINAMATH_GPT_fraction_of_employees_laid_off_l2422_242209


namespace NUMINAMATH_GPT_problem_statement_l2422_242239

def f (x : ℝ) : ℝ := 5 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 1

theorem problem_statement : g (f (g (f 1))) = 305 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2422_242239


namespace NUMINAMATH_GPT_compare_y_coordinates_l2422_242275

theorem compare_y_coordinates (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁: (x₁ = -3) ∧ (y₁ = 2 * x₁ - 1)) 
  (h₂: (x₂ = -5) ∧ (y₂ = 2 * x₂ - 1)) : 
  y₁ > y₂ := 
by 
  sorry

end NUMINAMATH_GPT_compare_y_coordinates_l2422_242275


namespace NUMINAMATH_GPT_transactions_proof_l2422_242299

def transactions_problem : Prop :=
  let mabel_transactions := 90
  let anthony_transactions := mabel_transactions + (0.10 * mabel_transactions)
  let cal_transactions := (2 / 3) * anthony_transactions
  let jade_transactions := 81
  jade_transactions - cal_transactions = 15

-- The proof is omitted (replace 'sorry' with an actual proof)
theorem transactions_proof : transactions_problem := by
  sorry

end NUMINAMATH_GPT_transactions_proof_l2422_242299


namespace NUMINAMATH_GPT_length_of_second_train_l2422_242278

/-- 
The length of the second train can be determined given the length and speed of the first train,
the speed of the second train, and the time they take to cross each other.
-/
theorem length_of_second_train (speed1_kmph : ℝ) (length1_m : ℝ) (speed2_kmph : ℝ) (time_s : ℝ) :
  (speed1_kmph = 120) →
  (length1_m = 230) →
  (speed2_kmph = 80) →
  (time_s = 9) →
  let relative_speed_m_per_s := (speed1_kmph * 1000 / 3600) + (speed2_kmph * 1000 / 3600)
  let total_distance := relative_speed_m_per_s * time_s
  let length2_m := total_distance - length1_m
  length2_m = 269.95 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  let relative_speed_m_per_s := (120 * 1000 / 3600) + (80 * 1000 / 3600)
  let total_distance := relative_speed_m_per_s * 9
  let length2_m := total_distance - 230
  exact sorry

end NUMINAMATH_GPT_length_of_second_train_l2422_242278


namespace NUMINAMATH_GPT_greater_number_is_18_l2422_242215

theorem greater_number_is_18 (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : x - y = 6) 
  (h3 : y ≥ 10) : 
  x = 18 := 
by 
  sorry

end NUMINAMATH_GPT_greater_number_is_18_l2422_242215


namespace NUMINAMATH_GPT_Tim_took_out_11_rulers_l2422_242258

-- Define the initial number of rulers
def initial_rulers := 14

-- Define the number of rulers left in the drawer
def rulers_left := 3

-- Define the number of rulers taken by Tim
def rulers_taken := initial_rulers - rulers_left

-- Statement to prove that the number of rulers taken by Tim is indeed 11
theorem Tim_took_out_11_rulers : rulers_taken = 11 := by
  sorry

end NUMINAMATH_GPT_Tim_took_out_11_rulers_l2422_242258


namespace NUMINAMATH_GPT_matrix_vector_product_l2422_242226

-- Definitions for matrix A and vector v
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-3, 4],
  ![2, -1]
]

def v : Fin 2 → ℤ := ![2, -2]

-- The theorem to prove
theorem matrix_vector_product :
  (A.mulVec v) = ![-14, 6] :=
by sorry

end NUMINAMATH_GPT_matrix_vector_product_l2422_242226


namespace NUMINAMATH_GPT_cost_price_of_product_l2422_242234

theorem cost_price_of_product (x y : ℝ)
  (h1 : 0.8 * y - x = 120)
  (h2 : 0.6 * y - x = -20) :
  x = 440 := sorry

end NUMINAMATH_GPT_cost_price_of_product_l2422_242234


namespace NUMINAMATH_GPT_problem_statement_l2422_242216

noncomputable def tan_plus_alpha_half_pi (α : ℝ) : ℝ := -1 / (Real.tan α)

theorem problem_statement (α : ℝ) (h : tan_plus_alpha_half_pi α = -1 / 2) :
  (2 * Real.sin α + Real.cos α) / (Real.cos α - Real.sin α) = -5 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2422_242216


namespace NUMINAMATH_GPT_third_grade_parts_in_batch_l2422_242284

-- Define conditions
variable (x y s : ℕ) (h_first_grade : 24 = 24) (h_second_grade : 36 = 36)
variable (h_sample_size : 20 = 20) (h_sample_third_grade : 10 = 10)

-- The problem: Prove the total number of third-grade parts in the batch is 60 and the number of second-grade parts sampled is 6
open Nat

theorem third_grade_parts_in_batch
  (h_total_parts : x - y = 60)
  (h_third_grade_proportion : y = (1 / 2) * x)
  (h_second_grade_proportion : s = (36 / 120) * 20) :
  y = 60 ∧ s = 6 := by
  sorry

end NUMINAMATH_GPT_third_grade_parts_in_batch_l2422_242284


namespace NUMINAMATH_GPT_opposite_of_neg_nine_l2422_242252

theorem opposite_of_neg_nine : -(-9) = 9 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_nine_l2422_242252


namespace NUMINAMATH_GPT_percentage_donated_l2422_242244

def income : ℝ := 1200000
def children_percentage : ℝ := 0.20
def wife_percentage : ℝ := 0.30
def remaining : ℝ := income - (children_percentage * 3 * income + wife_percentage * income)
def left_amount : ℝ := 60000
def donated : ℝ := remaining - left_amount

theorem percentage_donated : (donated / remaining) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_donated_l2422_242244


namespace NUMINAMATH_GPT_matrix_addition_l2422_242266

variable (A B : Matrix (Fin 2) (Fin 2) ℤ) -- Define matrices with integer entries

-- Define the specific matrices used in the problem
def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![ ![2, 3], ![-1, 4] ]

def matrix_B : Matrix (Fin 2) (Fin 2) ℤ := 
  ![ ![-1, 8], ![-3, 0] ]

-- Define the result matrix
def result_matrix : Matrix (Fin 2) (Fin 2) ℤ := 
  ![ ![3, 14], ![-5, 8] ]

-- The theorem to prove
theorem matrix_addition : 2 • matrix_A + matrix_B = result_matrix := by
  sorry -- Proof omitted

end NUMINAMATH_GPT_matrix_addition_l2422_242266


namespace NUMINAMATH_GPT_min_elements_in_as_l2422_242202

noncomputable def min_elems_in_A_s (n : ℕ) (S : Finset ℝ) (hS : S.card = n) : ℕ :=
  if 2 ≤ n then 2 * n - 3 else 0

theorem min_elements_in_as (n : ℕ) (S : Finset ℝ) (hS : S.card = n) (hn: 2 ≤ n) :
  ∃ (A_s : Finset ℝ), A_s.card = min_elems_in_A_s n S hS := sorry

end NUMINAMATH_GPT_min_elements_in_as_l2422_242202


namespace NUMINAMATH_GPT_calculate_expression_l2422_242261

theorem calculate_expression :
  12 * 11 + 7 * 8 - 5 * 6 + 10 * 4 = 198 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2422_242261


namespace NUMINAMATH_GPT_domain_transformation_l2422_242292

-- Definitions of conditions
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

def domain_g (x : ℝ) : Prop := 1 < x ∧ x ≤ 3

-- Theorem stating the proof problem
theorem domain_transformation : 
  (∀ x, domain_f x → 0 ≤ x+1 ∧ x+1 ≤ 4) →
  (∀ x, (0 ≤ x+1 ∧ x+1 ≤ 4) → (x-1 > 0) → domain_g x) :=
by
  intros h1 x hx
  sorry

end NUMINAMATH_GPT_domain_transformation_l2422_242292


namespace NUMINAMATH_GPT_set_inclusion_l2422_242297

def setM : Set ℝ := {θ | ∃ k : ℤ, θ = k * Real.pi / 4}

def setN : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4)}

def setP : Set ℝ := {a | ∃ k : ℤ, a = (k * Real.pi / 2) + (Real.pi / 4)}

theorem set_inclusion : setP ⊆ setN ∧ setN ⊆ setM := by
  sorry

end NUMINAMATH_GPT_set_inclusion_l2422_242297


namespace NUMINAMATH_GPT_total_capacity_is_correct_l2422_242232

-- Define small and large jars capacities
def small_jar_capacity : ℕ := 3
def large_jar_capacity : ℕ := 5

-- Define the total number of jars and the number of small jars
def total_jars : ℕ := 100
def small_jars : ℕ := 62

-- Define the number of large jars based on the total jars and small jars
def large_jars : ℕ := total_jars - small_jars

-- Calculate capacities
def small_jars_total_capacity : ℕ := small_jars * small_jar_capacity
def large_jars_total_capacity : ℕ := large_jars * large_jar_capacity

-- Define the total capacity
def total_capacity : ℕ := small_jars_total_capacity + large_jars_total_capacity

-- Prove that the total capacity is 376 liters
theorem total_capacity_is_correct : total_capacity = 376 := by
  sorry

end NUMINAMATH_GPT_total_capacity_is_correct_l2422_242232


namespace NUMINAMATH_GPT_postcards_initial_count_l2422_242274

theorem postcards_initial_count (P : ℕ) 
  (h1 : ∀ n, n = P / 2)
  (h2 : ∀ n, n = (P / 2) * 15 / 5) 
  (h3 : P / 2 + 3 * P / 2 = 36) : 
  P = 18 := 
sorry

end NUMINAMATH_GPT_postcards_initial_count_l2422_242274


namespace NUMINAMATH_GPT_isosceles_obtuse_triangle_angles_l2422_242289

def isosceles (A B C : ℝ) : Prop := A = B ∨ B = C ∨ C = A
def obtuse (A B C : ℝ) : Prop := A > 90 ∨ B > 90 ∨ C > 90

noncomputable def sixty_percent_larger_angle : ℝ := 1.6 * 90

theorem isosceles_obtuse_triangle_angles 
  (A B C : ℝ) 
  (h_iso : isosceles A B C) 
  (h_obt : obtuse A B C) 
  (h_large_angle : A = sixty_percent_larger_angle ∨ B = sixty_percent_larger_angle ∨ C = sixty_percent_larger_angle) 
  (h_sum : A + B + C = 180) : 
  (A = 18 ∨ B = 18 ∨ C = 18) := 
sorry

end NUMINAMATH_GPT_isosceles_obtuse_triangle_angles_l2422_242289


namespace NUMINAMATH_GPT_diagonal_length_of_octagon_l2422_242221

theorem diagonal_length_of_octagon 
  (r : ℝ) (s : ℝ) (has_symmetry_axes : ℕ) 
  (inscribed : r = 6) (side_length : s = 5) 
  (symmetry_condition : has_symmetry_axes = 4) : 
  ∃ (d : ℝ), d = 2 * Real.sqrt 40 := 
by 
  sorry

end NUMINAMATH_GPT_diagonal_length_of_octagon_l2422_242221


namespace NUMINAMATH_GPT_number_of_lucky_numbers_l2422_242200

-- Defining the concept of sequence with even number of digit 8
def is_lucky (seq : List ℕ) : Prop :=
  seq.count 8 % 2 = 0

-- Define S(n) recursive formula
noncomputable def S : ℕ → ℝ
| 0 => 0
| n+1 => 4 * (1 - (1 / (2 ^ (n+1))))

theorem number_of_lucky_numbers (n : ℕ) :
  ∀ (seq : List ℕ), (seq.length ≤ n) → is_lucky seq → S n = 4 * (1 - 1 / (2 ^ n)) :=
sorry

end NUMINAMATH_GPT_number_of_lucky_numbers_l2422_242200


namespace NUMINAMATH_GPT_tangent_line_parabola_k_l2422_242259

theorem tangent_line_parabola_k :
  ∃ (k : ℝ), (∀ (x y : ℝ), 4 * x + 7 * y + k = 0 → y^2 = 16 * x → (28 ^ 2 = 4 * 1 * 4 * k)) → k = 49 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_parabola_k_l2422_242259


namespace NUMINAMATH_GPT_systematic_sampling_example_l2422_242294

theorem systematic_sampling_example :
  ∃ (selected : Finset ℕ), 
    selected = {10, 30, 50, 70, 90} ∧
    ∀ n ∈ selected, 1 ≤ n ∧ n ≤ 100 ∧ 
    (∃ k, k > 0 ∧ k * 20 - 10∈ selected ∧ k * 20 - 10 ∈ Finset.range 101) := 
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_example_l2422_242294


namespace NUMINAMATH_GPT_tan_theta_half_l2422_242280

theorem tan_theta_half (θ : ℝ) (a b : ℝ × ℝ) 
  (h₀ : a = (Real.sin θ, 1)) 
  (h₁ : b = (-2, Real.cos θ)) 
  (h₂ : a.1 * b.1 + a.2 * b.2 = 0) : Real.tan θ = 1 / 2 :=
sorry

end NUMINAMATH_GPT_tan_theta_half_l2422_242280


namespace NUMINAMATH_GPT_extreme_point_property_l2422_242245

variables (f : ℝ → ℝ) (a b x x₀ x₁ : ℝ) 

-- Define the function f
def func (x : ℝ) := x^3 - a * x - b

-- The main theorem
theorem extreme_point_property (h₀ : ∃ x₀, ∃ x₁, (x₀ ≠ 0) ∧ (x₀^2 = a / 3) ∧ (x₁ ≠ x₀) ∧ (func a b x₀ = func a b x₁)) :
  x₁ + 2 * x₀ = 0 :=
sorry

end NUMINAMATH_GPT_extreme_point_property_l2422_242245


namespace NUMINAMATH_GPT_sophia_book_pages_l2422_242230

theorem sophia_book_pages:
  ∃ (P : ℕ), (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 30 ∧ P = 90 :=
by
  sorry

end NUMINAMATH_GPT_sophia_book_pages_l2422_242230


namespace NUMINAMATH_GPT_painting_area_l2422_242295

def wall_height : ℝ := 10
def wall_length : ℝ := 15
def door_height : ℝ := 3
def door_length : ℝ := 5

noncomputable def area_of_wall : ℝ :=
  wall_height * wall_length

noncomputable def area_of_door : ℝ :=
  door_height * door_length

noncomputable def area_to_paint : ℝ :=
  area_of_wall - area_of_door

theorem painting_area :
  area_to_paint = 135 := by
  sorry

end NUMINAMATH_GPT_painting_area_l2422_242295


namespace NUMINAMATH_GPT_log_sum_eq_l2422_242201

theorem log_sum_eq : ∀ (x y : ℝ), y = 2016 * x ∧ x^y = y^x → (Real.logb 2016 x + Real.logb 2016 y) = 2017 / 2015 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_log_sum_eq_l2422_242201


namespace NUMINAMATH_GPT_no_adjacent_black_balls_l2422_242220

theorem no_adjacent_black_balls (m n : ℕ) (h : m > n) : 
  (m + 1).choose n = (m + 1).factorial / (n.factorial * (m + 1 - n).factorial) := by
  sorry

end NUMINAMATH_GPT_no_adjacent_black_balls_l2422_242220


namespace NUMINAMATH_GPT_add_decimals_l2422_242205

theorem add_decimals :
  0.0935 + 0.007 + 0.2 = 0.3005 :=
by sorry

end NUMINAMATH_GPT_add_decimals_l2422_242205


namespace NUMINAMATH_GPT_circle1_correct_circle2_correct_l2422_242247

noncomputable def circle1_eq (x y : ℝ) : ℝ :=
  x^2 + y^2 + 4*x - 6*y - 12

noncomputable def circle2_eq (x y : ℝ) : ℝ :=
  36*x^2 + 36*y^2 - 24*x + 72*y + 31

theorem circle1_correct (x y : ℝ) :
  ((x + 2)^2 + (y - 3)^2 = 25) ↔ (circle1_eq x y = 0) :=
sorry

theorem circle2_correct (x y : ℝ) :
  (36 * ((x - 1/3)^2 + (y + 1)^2) = 9) ↔ (circle2_eq x y = 0) :=
sorry

end NUMINAMATH_GPT_circle1_correct_circle2_correct_l2422_242247


namespace NUMINAMATH_GPT_toys_ratio_l2422_242219

theorem toys_ratio (k A M T : ℕ) (h1 : M = 6) (h2 : A = k * M) (h3 : A = T - 2) (h4 : A + M + T = 56):
  A / M = 4 :=
by
  sorry

end NUMINAMATH_GPT_toys_ratio_l2422_242219


namespace NUMINAMATH_GPT_tetrahedron_coloring_l2422_242203

noncomputable def count_distinct_tetrahedron_colorings : ℕ :=
  sorry

theorem tetrahedron_coloring :
  count_distinct_tetrahedron_colorings = 6 :=
  sorry

end NUMINAMATH_GPT_tetrahedron_coloring_l2422_242203


namespace NUMINAMATH_GPT_max_students_equal_division_l2422_242231

theorem max_students_equal_division (pens pencils : ℕ) (h_pens : pens = 640) (h_pencils : pencils = 520) : 
  Nat.gcd pens pencils = 40 :=
by
  rw [h_pens, h_pencils]
  have : Nat.gcd 640 520 = 40 := by norm_num
  exact this

end NUMINAMATH_GPT_max_students_equal_division_l2422_242231


namespace NUMINAMATH_GPT_total_area_of_WIN_sectors_l2422_242268

theorem total_area_of_WIN_sectors (r : ℝ) (A_total : ℝ) (Prob_WIN : ℝ) (A_WIN : ℝ) : 
  r = 15 → 
  A_total = π * r^2 → 
  Prob_WIN = 3/7 → 
  A_WIN = Prob_WIN * A_total → 
  A_WIN = 3/7 * 225 * π :=
by {
  intros;
  sorry
}

end NUMINAMATH_GPT_total_area_of_WIN_sectors_l2422_242268


namespace NUMINAMATH_GPT_sum_first_five_terms_l2422_242237

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^(n-1)

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

theorem sum_first_five_terms (a1 q : ℝ) 
  (h1 : geometric_sequence a1 q 2 * geometric_sequence a1 q 3 = 2 * a1)
  (h2 : (geometric_sequence a1 q 4 + 2 * geometric_sequence a1 q 7) / 2 = 5 / 4)
  : sum_geometric_sequence a1 q 5 = 31 :=
sorry

end NUMINAMATH_GPT_sum_first_five_terms_l2422_242237


namespace NUMINAMATH_GPT_line_through_center_and_perpendicular_l2422_242269

def center_of_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 1

def perpendicular_to_line (slope : ℝ) : Prop :=
  slope = 1

theorem line_through_center_and_perpendicular (x y : ℝ) :
  center_of_circle x y →
  perpendicular_to_line 1 →
  (x - y + 1 = 0) :=
by
  intros h_center h_perpendicular
  sorry

end NUMINAMATH_GPT_line_through_center_and_perpendicular_l2422_242269


namespace NUMINAMATH_GPT_find_u_plus_v_l2422_242285

theorem find_u_plus_v (u v : ℚ) 
  (h₁ : 3 * u + 7 * v = 17) 
  (h₂ : 5 * u - 3 * v = 9) : 
  u + v = 43 / 11 :=
sorry

end NUMINAMATH_GPT_find_u_plus_v_l2422_242285


namespace NUMINAMATH_GPT_q_evaluation_at_3_point_5_l2422_242290

def q (x : ℝ) : ℝ :=
  |x - 3|^(1/3) + 2*|x - 3|^(1/5) + |x - 3|^(1/7)

theorem q_evaluation_at_3_point_5 : q 3.5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_q_evaluation_at_3_point_5_l2422_242290


namespace NUMINAMATH_GPT_range_of_a_l2422_242214

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (x > 0) ∧ (π^x = (a + 1) / (2 - a))) → (1 / 2 < a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2422_242214


namespace NUMINAMATH_GPT_advertisement_length_l2422_242235

noncomputable def movie_length : ℕ := 90
noncomputable def replay_times : ℕ := 6
noncomputable def operation_time : ℕ := 660

theorem advertisement_length : ∃ A : ℕ, 90 * replay_times + 6 * A = operation_time ∧ A = 20 :=
by
  use 20
  sorry

end NUMINAMATH_GPT_advertisement_length_l2422_242235


namespace NUMINAMATH_GPT_garden_perimeter_l2422_242236

/-- Define the dimensions of the rectangle and triangle in the garden -/
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 4
def triangle_leg1 : ℕ := 3
def triangle_leg2 : ℕ := 4
def triangle_hypotenuse : ℕ := 5 -- calculated using Pythagorean theorem

/-- Prove that the total perimeter of the combined shape is 28 units -/
theorem garden_perimeter :
  let perimeter := 2 * rectangle_length + rectangle_width + triangle_leg1 + triangle_hypotenuse
  perimeter = 28 :=
by
  let perimeter := 2 * rectangle_length + rectangle_width + triangle_leg1 + triangle_hypotenuse
  have h : perimeter = 28 := sorry
  exact h

end NUMINAMATH_GPT_garden_perimeter_l2422_242236


namespace NUMINAMATH_GPT_sum_of_reciprocals_l2422_242233

theorem sum_of_reciprocals (x y : ℝ) (h : x + y = 6 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x) + (1 / y) = 2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l2422_242233


namespace NUMINAMATH_GPT_shaded_quadrilateral_area_l2422_242283

noncomputable def area_of_shaded_quadrilateral : ℝ :=
  let side_lens : List ℝ := [3, 5, 7, 9]
  let total_base: ℝ := side_lens.sum
  let largest_square_height: ℝ := 9
  let height_base_ratio := largest_square_height / total_base
  let heights := side_lens.scanl (· + ·) 0 |>.tail.map (λ x => x * height_base_ratio)
  let a := heights.get! 0
  let b := heights.get! heights.length - 1
  (largest_square_height * (a + b)) / 2

theorem shaded_quadrilateral_area :
    let side_lens := [3, 5, 7, 9]
    let total_base := side_lens.sum
    let largest_square_height := 9
    let height_base_ratio := largest_square_height / total_base
    let heights := side_lens.scanl (· + ·) 0 |>.tail.map (λ x => x * height_base_ratio)
    let a := heights.get! 0
    let b := heights.get! heights.length - 1
    (largest_square_height * (a + b)) / 2 = 30.375 :=
by 
  sorry

end NUMINAMATH_GPT_shaded_quadrilateral_area_l2422_242283


namespace NUMINAMATH_GPT_binary_to_decimal_l2422_242255

theorem binary_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 13) :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l2422_242255


namespace NUMINAMATH_GPT_problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l2422_242256

-- Problem G6.1
theorem problem_G6_1 : (21 ^ 3 - 11 ^ 3) / (21 ^ 2 + 21 * 11 + 11 ^ 2) = 10 := 
  sorry

-- Problem G6.2
theorem problem_G6_2 (p q : ℕ) (h1 : (p : ℚ) * 6 = 4 * (q : ℚ)) : q = 3 * p / 2 := 
  sorry

-- Problem G6.3
theorem problem_G6_3 (q r : ℕ) (h1 : q % 7 = 3) (h2 : r % 7 = 5) (h3 : 18 < r) (h4 : r < 26) : r = 24 := 
  sorry

-- Problem G6.4
def star (a b : ℕ) : ℕ := a * b + 1

theorem problem_G6_4 : star (star 3 4) 2 = 27 := 
  sorry

end NUMINAMATH_GPT_problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l2422_242256


namespace NUMINAMATH_GPT_pondFishEstimate_l2422_242251

noncomputable def estimateTotalFish (initialFishMarked : ℕ) (caughtFishTenDaysLater : ℕ) (markedFishCaught : ℕ) : ℕ :=
  initialFishMarked * caughtFishTenDaysLater / markedFishCaught

theorem pondFishEstimate
    (initialFishMarked : ℕ)
    (caughtFishTenDaysLater : ℕ)
    (markedFishCaught : ℕ)
    (h1 : initialFishMarked = 30)
    (h2 : caughtFishTenDaysLater = 50)
    (h3 : markedFishCaught = 2) :
    estimateTotalFish initialFishMarked caughtFishTenDaysLater markedFishCaught = 750 := by
  sorry

end NUMINAMATH_GPT_pondFishEstimate_l2422_242251


namespace NUMINAMATH_GPT_geometric_series_sum_l2422_242293

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h_a : a = 1) (h_r : r = 1 / 2) (h_n : n = 5) :
  ((a * (1 - r^n)) / (1 - r)) = 31 / 16 := 
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l2422_242293


namespace NUMINAMATH_GPT_Heath_current_age_l2422_242241

variable (H J : ℕ) -- Declare variables for Heath's and Jude's ages
variable (h1 : J = 2) -- Jude's current age is 2
variable (h2 : H + 5 = 3 * (J + 5)) -- In 5 years, Heath will be 3 times as old as Jude

theorem Heath_current_age : H = 16 :=
by
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_Heath_current_age_l2422_242241


namespace NUMINAMATH_GPT_Arianna_time_at_work_l2422_242210

theorem Arianna_time_at_work : 
  (24 - (5 + 13)) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_Arianna_time_at_work_l2422_242210


namespace NUMINAMATH_GPT_g_at_52_l2422_242288

noncomputable def g : ℝ → ℝ := sorry

axiom g_multiplicative : ∀ (x y: ℝ), g (x * y) = y * g x
axiom g_at_1 : g 1 = 10

theorem g_at_52 : g 52 = 520 := sorry

end NUMINAMATH_GPT_g_at_52_l2422_242288


namespace NUMINAMATH_GPT_grace_have_30_pastries_l2422_242248

theorem grace_have_30_pastries (F : ℕ) :
  (2 * (F + 8) + F + (F + 13) = 97) → (F + 13 = 30) :=
by
  sorry

end NUMINAMATH_GPT_grace_have_30_pastries_l2422_242248


namespace NUMINAMATH_GPT_range_a_ff_a_eq_2_f_a_l2422_242225

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then 3 * x - 1 else 2 ^ x

theorem range_a_ff_a_eq_2_f_a :
  {a : ℝ | f (f a) = 2 ^ (f a)} = {a : ℝ | a ≥ 2/3} :=
sorry

end NUMINAMATH_GPT_range_a_ff_a_eq_2_f_a_l2422_242225


namespace NUMINAMATH_GPT_line_through_two_points_l2422_242224

-- Define the points
def p1 : ℝ × ℝ := (1, 0)
def p2 : ℝ × ℝ := (0, -2)

-- Define the equation of the line passing through the points
def line_equation (x y : ℝ) : Prop :=
  2 * x - y - 2 = 0

-- The main theorem
theorem line_through_two_points : ∀ x y, p1 = (1, 0) ∧ p2 = (0, -2) → line_equation x y :=
  by sorry

end NUMINAMATH_GPT_line_through_two_points_l2422_242224


namespace NUMINAMATH_GPT_air_conditioner_consumption_l2422_242253

theorem air_conditioner_consumption :
  ∀ (total_consumption_8_hours : ℝ)
    (hours_8 : ℝ)
    (hours_per_day : ℝ)
    (days : ℝ),
    total_consumption_8_hours / hours_8 * hours_per_day * days = 27 :=
by
  intros total_consumption_8_hours hours_8 hours_per_day days
  sorry

end NUMINAMATH_GPT_air_conditioner_consumption_l2422_242253


namespace NUMINAMATH_GPT_intersection_eq_l2422_242213

open Set

variable (A B : Set ℝ)

def setA : A = {x | -3 < x ∧ x < 2} := sorry

def setB : B = {x | x^2 + 4*x - 5 ≤ 0} := sorry

theorem intersection_eq : A ∩ B = {x | -3 < x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_GPT_intersection_eq_l2422_242213


namespace NUMINAMATH_GPT_alpha_half_quadrant_l2422_242291

open Real

theorem alpha_half_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π - π / 2 < α ∧ α < 2 * k * π) :
  (∃ k1 : ℤ, (2 * k1 + 1) * π - π / 4 < α / 2 ∧ α / 2 < (2 * k1 + 1) * π) ∨
  (∃ k2 : ℤ, 2 * k2 * π - π / 4 < α / 2 ∧ α / 2 < 2 * k2 * π) :=
sorry

end NUMINAMATH_GPT_alpha_half_quadrant_l2422_242291


namespace NUMINAMATH_GPT_min_value_sum_inverse_squares_l2422_242281

theorem min_value_sum_inverse_squares (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_sum : a + b + c = 3) :
    (1 / (a + b)^2) + (1 / (a + c)^2) + (1 / (b + c)^2) >= 3 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_sum_inverse_squares_l2422_242281


namespace NUMINAMATH_GPT_jill_bought_5_packs_of_red_bouncy_balls_l2422_242273

theorem jill_bought_5_packs_of_red_bouncy_balls
  (r : ℕ) -- number of packs of red bouncy balls
  (yellow_packs : ℕ := 4)
  (bouncy_balls_per_pack : ℕ := 18)
  (extra_red_bouncy_balls : ℕ := 18)
  (total_yellow_bouncy_balls : ℕ := yellow_packs * bouncy_balls_per_pack)
  (total_red_bouncy_balls : ℕ := total_yellow_bouncy_balls + extra_red_bouncy_balls)
  (h : r * bouncy_balls_per_pack = total_red_bouncy_balls) :
  r = 5 :=
by sorry

end NUMINAMATH_GPT_jill_bought_5_packs_of_red_bouncy_balls_l2422_242273


namespace NUMINAMATH_GPT_no_apples_info_l2422_242279

theorem no_apples_info (r d : ℕ) (condition1 : r = 79) (condition2 : d = 53) (condition3 : r = d + 26) : 
  ∀ a : ℕ, (a = a) → false :=
by
  intro a h
  sorry

end NUMINAMATH_GPT_no_apples_info_l2422_242279


namespace NUMINAMATH_GPT_rod_center_of_gravity_shift_l2422_242218

noncomputable def rod_shift (l : ℝ) (s : ℝ) : ℝ := 
  |(l / 2) - ((l - s) / 2)| 

theorem rod_center_of_gravity_shift : 
  rod_shift l 80 = 40 := by
  sorry

end NUMINAMATH_GPT_rod_center_of_gravity_shift_l2422_242218


namespace NUMINAMATH_GPT_remaining_money_after_payments_l2422_242263

-- Conditions
def initial_money : ℕ := 100
def paid_colin : ℕ := 20
def paid_helen : ℕ := 2 * paid_colin
def paid_benedict : ℕ := paid_helen / 2
def total_paid : ℕ := paid_colin + paid_helen + paid_benedict

-- Proof
theorem remaining_money_after_payments : 
  initial_money - total_paid = 20 := by
  sorry

end NUMINAMATH_GPT_remaining_money_after_payments_l2422_242263


namespace NUMINAMATH_GPT_birds_remaining_on_fence_l2422_242262

noncomputable def initial_birds : ℝ := 15.3
noncomputable def birds_flew_away : ℝ := 6.5
noncomputable def remaining_birds : ℝ := initial_birds - birds_flew_away

theorem birds_remaining_on_fence : remaining_birds = 8.8 :=
by
  -- sorry is a placeholder for the proof, which is not required
  sorry

end NUMINAMATH_GPT_birds_remaining_on_fence_l2422_242262


namespace NUMINAMATH_GPT_probability_compensation_l2422_242206

-- Define the probabilities of each vehicle getting into an accident
def p1 : ℚ := 1 / 20
def p2 : ℚ := 1 / 21

-- Define the probability of the complementary event
def comp_event : ℚ := (1 - p1) * (1 - p2)

-- Define the overall probability that at least one vehicle gets into an accident
def comp_unit : ℚ := 1 - comp_event

-- The theorem to be proved: the probability that the unit will receive compensation from this insurance within a year is 2 / 21
theorem probability_compensation : comp_unit = 2 / 21 :=
by
  -- giving the proof is not required
  sorry

end NUMINAMATH_GPT_probability_compensation_l2422_242206


namespace NUMINAMATH_GPT_exchange_5_rubles_l2422_242204

theorem exchange_5_rubles :
  ¬ ∃ n : ℕ, 1 * n + 2 * n + 3 * n + 5 * n = 500 :=
by 
  sorry

end NUMINAMATH_GPT_exchange_5_rubles_l2422_242204


namespace NUMINAMATH_GPT_alice_bush_count_l2422_242250

theorem alice_bush_count :
  let side_length := 24
  let num_sides := 3
  let bush_space := 3
  (num_sides * side_length) / bush_space = 24 :=
by
  sorry

end NUMINAMATH_GPT_alice_bush_count_l2422_242250


namespace NUMINAMATH_GPT_problem_l2422_242271

def setA : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}
def setB : Set ℝ := {x : ℝ | x ≤ 3}

theorem problem : setA ∩ setB = setA := sorry

end NUMINAMATH_GPT_problem_l2422_242271
