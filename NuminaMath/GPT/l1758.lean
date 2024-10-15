import Mathlib

namespace NUMINAMATH_GPT_sum_of_four_triangles_l1758_175831

theorem sum_of_four_triangles :
  ∀ (x y : ℝ), 3 * x + 2 * y = 27 → 2 * x + 3 * y = 23 → 4 * y = 12 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_four_triangles_l1758_175831


namespace NUMINAMATH_GPT_sequence_bound_l1758_175869

theorem sequence_bound (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_seq : ∀ n, (a n) ^ 2 ≤ a (n + 1)) :
  ∀ n, a n < 1 / n :=
by
  intros
  sorry

end NUMINAMATH_GPT_sequence_bound_l1758_175869


namespace NUMINAMATH_GPT_find_t_l1758_175809

-- Given: (1) g(x) = x^5 + px^4 + qx^3 + rx^2 + sx + t with all roots being negative integers
--        (2) p + q + r + s + t = 3024
-- Prove: t = 1600

noncomputable def poly (x : ℝ) (p q r s t : ℝ) := 
  x^5 + p*x^4 + q*x^3 + r*x^2 + s*x + t

theorem find_t
  (p q r s t : ℝ)
  (roots_neg_int : ∀ root, root ∈ [-s1, -s2, -s3, -s4, -s5] → (root : ℤ) < 0)
  (sum_coeffs : p + q + r + s + t = 3024)
  (poly_1_eq : poly 1 p q r s t = 3025) :
  t = 1600 := 
sorry

end NUMINAMATH_GPT_find_t_l1758_175809


namespace NUMINAMATH_GPT_system1_solution_l1758_175842

theorem system1_solution (x y : ℝ) (h1 : 2 * x - y = 1) (h2 : 7 * x - 3 * y = 4) : x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_GPT_system1_solution_l1758_175842


namespace NUMINAMATH_GPT_total_books_sold_amount_l1758_175837

def num_fiction_books := 60
def num_non_fiction_books := 84
def num_children_books := 42

def fiction_books_sold := 3 / 4 * num_fiction_books
def non_fiction_books_sold := 5 / 6 * num_non_fiction_books
def children_books_sold := 2 / 3 * num_children_books

def price_fiction := 5
def price_non_fiction := 7
def price_children := 3

def total_amount_fiction := fiction_books_sold * price_fiction
def total_amount_non_fiction := non_fiction_books_sold * price_non_fiction
def total_amount_children := children_books_sold * price_children

def total_amount_received := total_amount_fiction + total_amount_non_fiction + total_amount_children

theorem total_books_sold_amount :
  total_amount_received = 799 :=
sorry

end NUMINAMATH_GPT_total_books_sold_amount_l1758_175837


namespace NUMINAMATH_GPT_ratio_w_y_l1758_175899

theorem ratio_w_y (w x y z : ℚ) 
  (h1 : w / x = 5 / 4) 
  (h2 : y / z = 4 / 3)
  (h3 : z / x = 1 / 8) : 
  w / y = 15 / 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_w_y_l1758_175899


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l1758_175812

theorem angle_in_third_quadrant (θ : ℝ) (h : θ = 2010) : ((θ % 360) > 180 ∧ (θ % 360) < 270) :=
by
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l1758_175812


namespace NUMINAMATH_GPT_cubic_equation_unique_real_solution_l1758_175813

theorem cubic_equation_unique_real_solution :
  (∃ (m : ℝ), ∀ x : ℝ, x^3 - 4*x - m = 0 → x = 2) ↔ m = -8 :=
by sorry

end NUMINAMATH_GPT_cubic_equation_unique_real_solution_l1758_175813


namespace NUMINAMATH_GPT_dice_probability_l1758_175850

theorem dice_probability :
  let num_dice := 6
  let prob_one_digit := 9 / 20
  let prob_two_digit := 11 / 20
  let num_combinations := Nat.choose num_dice (num_dice / 2)
  let prob_each_combination := (prob_one_digit ^ 3) * (prob_two_digit ^ 3)
  let total_probability := num_combinations * prob_each_combination
  total_probability = 4851495 / 16000000 := by
    let num_dice := 6
    let prob_one_digit := 9 / 20
    let prob_two_digit := 11 / 20
    let num_combinations := Nat.choose num_dice (num_dice / 2)
    let prob_each_combination := (prob_one_digit ^ 3) * (prob_two_digit ^ 3)
    let total_probability := num_combinations * prob_each_combination
    sorry

end NUMINAMATH_GPT_dice_probability_l1758_175850


namespace NUMINAMATH_GPT_sumOddDivisorsOf90_l1758_175805

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end NUMINAMATH_GPT_sumOddDivisorsOf90_l1758_175805


namespace NUMINAMATH_GPT_twenty_four_game_l1758_175881

-- Definition of the cards' values
def card2 : ℕ := 2
def card5 : ℕ := 5
def cardJ : ℕ := 11
def cardQ : ℕ := 12

-- Theorem stating the proof
theorem twenty_four_game : card2 * (cardJ - card5) + cardQ = 24 :=
by
  sorry

end NUMINAMATH_GPT_twenty_four_game_l1758_175881


namespace NUMINAMATH_GPT_max_lambda_inequality_l1758_175880

theorem max_lambda_inequality 
  (a b x y : ℝ) 
  (h1 : a ≥ 0) 
  (h2 : b ≥ 0)
  (h3 : x ≥ 0)
  (h4 : y ≥ 0)
  (h5 : a + b = 27) : 
  (a * x^2 + b * y^2 + 4 * x * y)^3 ≥ 4 * (a * x^2 * y + b * x * y^2)^2 :=
sorry

end NUMINAMATH_GPT_max_lambda_inequality_l1758_175880


namespace NUMINAMATH_GPT_swap_columns_produce_B_l1758_175852

variables {n : ℕ} (A : Matrix (Fin n) (Fin n) (Fin n))

def K (B : Matrix (Fin n) (Fin n) (Fin n)) : ℕ :=
  Fintype.card {ij : (Fin n) × (Fin n) // B ij.1 ij.2 = ij.2}

theorem swap_columns_produce_B (A : Matrix (Fin n) (Fin n) (Fin n)) :
  ∃ (B : Matrix (Fin n) (Fin n) (Fin n)), (∀ i, ∃ j, B i j = A i j) ∧ K B ≤ n :=
sorry

end NUMINAMATH_GPT_swap_columns_produce_B_l1758_175852


namespace NUMINAMATH_GPT_division_remainder_l1758_175884

theorem division_remainder : 1234567 % 112 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_division_remainder_l1758_175884


namespace NUMINAMATH_GPT_max_value_l1758_175846

-- Define the vector types
structure Vector2 where
  x : ℝ
  y : ℝ

-- Define the properties given in the problem
def a_is_unit_vector (a : Vector2) : Prop :=
  a.x^2 + a.y^2 = 1

def a_plus_b (a b : Vector2) : Prop :=
  a.x + b.x = 3 ∧ a.y + b.y = 4

-- Define dot product for the vectors
def dot_product (a b : Vector2) : ℝ :=
  a.x * b.x + a.y * b.y

-- The theorem statement
theorem max_value (a b : Vector2) (h1 : a_is_unit_vector a) (h2 : a_plus_b a b) :
  ∃ m, m = 5 ∧ ∀ c : ℝ, |1 + dot_product a b| ≤ m :=
  sorry

end NUMINAMATH_GPT_max_value_l1758_175846


namespace NUMINAMATH_GPT_min_inverse_ab_l1758_175841

theorem min_inverse_ab (a b : ℝ) (h1 : a + a * b + 2 * b = 30) (h2 : a > 0) (h3 : b > 0) :
  ∃ m : ℝ, m = 1 / 18 ∧ (∀ x y : ℝ, (x + x * y + 2 * y = 30) → (x > 0) → (y > 0) → 1 / (x * y) ≥ m) :=
sorry

end NUMINAMATH_GPT_min_inverse_ab_l1758_175841


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l1758_175827

noncomputable def four_sin_40_minus_tan_40 : ℝ :=
  4 * Real.sin (40 * Real.pi / 180) - Real.tan (40 * Real.pi / 180)

theorem trigonometric_identity_proof : four_sin_40_minus_tan_40 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l1758_175827


namespace NUMINAMATH_GPT_ratio_of_areas_inequality_l1758_175806

theorem ratio_of_areas_inequality (a x m : ℝ) (h1 : a > 0) (h2 : x > 0) (h3 : x < a) :
  m = (3 * x^2 - 3 * a * x + a^2) / a^2 →
  (1 / 4 ≤ m ∧ m < 1) :=
sorry

end NUMINAMATH_GPT_ratio_of_areas_inequality_l1758_175806


namespace NUMINAMATH_GPT_one_fourth_to_fourth_power_is_decimal_l1758_175816

def one_fourth : ℚ := 1 / 4

theorem one_fourth_to_fourth_power_is_decimal :
  (one_fourth ^ 4 : ℚ) = 0.00390625 := 
by sorry

end NUMINAMATH_GPT_one_fourth_to_fourth_power_is_decimal_l1758_175816


namespace NUMINAMATH_GPT_factor_polynomial_l1758_175829

def Polynomial_Factorization (x : ℝ) : Prop := 
  let P := x^2 - 6*x + 9 - 64*x^4
  P = (8*x^2 + x - 3) * (-8*x^2 + x - 3)

theorem factor_polynomial : ∀ x : ℝ, Polynomial_Factorization x :=
by 
  intro x
  unfold Polynomial_Factorization
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1758_175829


namespace NUMINAMATH_GPT_sum_of_smallest_integers_l1758_175840

theorem sum_of_smallest_integers (x y : ℕ) (h1 : ∃ x, x > 0 ∧ (∃ n : ℕ, 720 * x = n^2) ∧ (∀ m : ℕ, m > 0 ∧ (∃ k : ℕ, 720 * m = k^2) → x ≤ m))
  (h2 : ∃ y, y > 0 ∧ (∃ p : ℕ, 720 * y = p^4) ∧ (∀ q : ℕ, q > 0 ∧ (∃ r : ℕ, 720 * q = r^4) → y ≤ q)) :
  x + y = 1130 := 
sorry

end NUMINAMATH_GPT_sum_of_smallest_integers_l1758_175840


namespace NUMINAMATH_GPT_symmetric_point_y_axis_l1758_175851

theorem symmetric_point_y_axis (B : ℝ × ℝ) (hB : B = (-3, 4)) : 
  ∃ A : ℝ × ℝ, A = (3, 4) ∧ A.2 = B.2 ∧ A.1 = -B.1 :=
by
  use (3, 4)
  sorry

end NUMINAMATH_GPT_symmetric_point_y_axis_l1758_175851


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1758_175847

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (d : ℚ) (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 4 + 1 / 2 * a 7 + a 10 = 10) : a 3 + a 11 = 8 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1758_175847


namespace NUMINAMATH_GPT_garden_perimeter_l1758_175879

theorem garden_perimeter
  (a b : ℝ)
  (h1 : a^2 + b^2 = 1156)
  (h2 : a * b = 240) :
  2 * (a + b) = 80 :=
sorry

end NUMINAMATH_GPT_garden_perimeter_l1758_175879


namespace NUMINAMATH_GPT_find_x_parallel_l1758_175823

theorem find_x_parallel (x : ℝ) 
  (a : ℝ × ℝ := (x, 2)) 
  (b : ℝ × ℝ := (2, 4)) 
  (h : a.1 * b.2 = a.2 * b.1) :
  x = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_x_parallel_l1758_175823


namespace NUMINAMATH_GPT_prob_B_hired_is_3_4_prob_at_least_two_hired_l1758_175830

-- Definitions for the conditions
def prob_A_hired : ℚ := 2 / 3
def prob_neither_A_nor_B_hired : ℚ := 1 / 12
def prob_B_and_C_hired : ℚ := 3 / 8

-- Targets to prove
theorem prob_B_hired_is_3_4 (P_A_hired : ℚ) (P_neither_A_nor_B_hired : ℚ) (P_B_and_C_hired : ℚ)
    (P_A_hired_eq : P_A_hired = prob_A_hired)
    (P_neither_A_nor_B_hired_eq : P_neither_A_nor_B_hired = prob_neither_A_nor_B_hired)
    (P_B_and_C_hired_eq : P_B_and_C_hired = prob_B_and_C_hired)
    : ∃ x y : ℚ, y = 1 / 2 ∧ x = 3 / 4 :=
by
  sorry
  
theorem prob_at_least_two_hired (P_A_hired : ℚ) (P_B_hired : ℚ) (P_C_hired : ℚ)
    (P_A_hired_eq : P_A_hired = prob_A_hired)
    (P_B_hired_eq : P_B_hired = 3 / 4)
    (P_C_hired_eq : P_C_hired = 1 / 2)
    : (P_A_hired * P_B_hired * P_C_hired) + 
      ((1 - P_A_hired) * P_B_hired * P_C_hired) + 
      (P_A_hired * (1 - P_B_hired) * P_C_hired) + 
      (P_A_hired * P_B_hired * (1 - P_C_hired)) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_prob_B_hired_is_3_4_prob_at_least_two_hired_l1758_175830


namespace NUMINAMATH_GPT_more_roses_than_orchids_l1758_175828

-- Definitions
def roses_now : Nat := 12
def orchids_now : Nat := 2

-- Theorem statement
theorem more_roses_than_orchids : (roses_now - orchids_now) = 10 := by
  sorry

end NUMINAMATH_GPT_more_roses_than_orchids_l1758_175828


namespace NUMINAMATH_GPT_annual_profit_function_correct_maximum_annual_profit_l1758_175882

noncomputable def fixed_cost : ℝ := 60

noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x < 12 then 
    0.5 * x^2 + 4 * x 
  else 
    11 * x + 100 / x - 39

noncomputable def selling_price_per_thousand : ℝ := 10

noncomputable def sales_revenue (x : ℝ) : ℝ := selling_price_per_thousand * x

noncomputable def annual_profit (x : ℝ) : ℝ := sales_revenue x - fixed_cost - variable_cost x

theorem annual_profit_function_correct : 
∀ x : ℝ, (0 < x ∧ x < 12 → annual_profit x = -0.5 * x^2 + 6 * x - fixed_cost) ∧ 
        (x ≥ 12 → annual_profit x = -x - 100 / x + 33) :=
sorry

theorem maximum_annual_profit : 
∃ x : ℝ, x = 12 ∧ annual_profit x = 38 / 3 :=
sorry

end NUMINAMATH_GPT_annual_profit_function_correct_maximum_annual_profit_l1758_175882


namespace NUMINAMATH_GPT_hot_dogs_remainder_l1758_175810

theorem hot_dogs_remainder :
  let n := 16789537
  let d := 5
  n % d = 2 :=
by
  sorry

end NUMINAMATH_GPT_hot_dogs_remainder_l1758_175810


namespace NUMINAMATH_GPT_fraction_of_managers_l1758_175896

theorem fraction_of_managers (female_managers : ℕ) (total_female_employees : ℕ)
  (total_employees: ℕ) (male_employees: ℕ) (f: ℝ) :
  female_managers = 200 →
  total_female_employees = 500 →
  total_employees = total_female_employees + male_employees →
  (f * total_employees) = female_managers + (f * male_employees) →
  f = 0.4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_fraction_of_managers_l1758_175896


namespace NUMINAMATH_GPT_least_whole_number_subtracted_l1758_175856

theorem least_whole_number_subtracted (x : ℕ) :
  ((6 - x) / (7 - x) < (16 / 21)) → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_least_whole_number_subtracted_l1758_175856


namespace NUMINAMATH_GPT_man_l1758_175873

theorem man's_age_ratio_father (M F : ℕ) (hF : F = 60)
  (h_age_relationship : M + 12 = (F + 12) / 2) :
  M / F = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_man_l1758_175873


namespace NUMINAMATH_GPT_initial_profit_percentage_l1758_175848

-- Definitions of conditions
variables {x y : ℝ} (h1 : y > x) (h2 : 2 * y - x = 1.4 * x)

-- Proof statement in Lean
theorem initial_profit_percentage (x y : ℝ) (h1 : y > x) (h2 : 2 * y - x = 1.4 * x) :
  ((y - x) / x) * 100 = 20 :=
by sorry

end NUMINAMATH_GPT_initial_profit_percentage_l1758_175848


namespace NUMINAMATH_GPT_area_of_triangle_l1758_175832

noncomputable def triangle_area (AB AC θ : ℝ) : ℝ := 
  0.5 * AB * AC * Real.sin θ

theorem area_of_triangle (AB AC : ℝ) (θ : ℝ) (hAB : AB = 1) (hAC : AC = 2) (hθ : θ = 2 * Real.pi / 3) :
  triangle_area AB AC θ = 3 * Real.sqrt 3 / 14 :=
by
  rw [triangle_area, hAB, hAC, hθ]
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1758_175832


namespace NUMINAMATH_GPT_gcd_of_A_B_l1758_175867

noncomputable def A (k : ℕ) := 2 * k
noncomputable def B (k : ℕ) := 5 * k

theorem gcd_of_A_B (k : ℕ) (h_lcm : Nat.lcm (A k) (B k) = 180) : Nat.gcd (A k) (B k) = 18 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_A_B_l1758_175867


namespace NUMINAMATH_GPT_sum_even_102_to_600_l1758_175825

def sum_first_50_even : ℕ := 2550
def sum_even_602_to_700 : ℕ := 32550

theorem sum_even_102_to_600 : sum_even_602_to_700 - sum_first_50_even = 30000 :=
by
  -- The given sum of the first 50 positive even integers is 2550
  have h1 : sum_first_50_even = 2550 := by rfl
  
  -- The given sum of the even integers from 602 to 700 inclusive is 32550
  have h2 : sum_even_602_to_700 = 32550 := by rfl
  
  -- Therefore, the sum of the even integers from 102 to 600 is:
  have h3 : sum_even_602_to_700 - sum_first_50_even = 32550 - 2550 := by
    rw [h1, h2]
  
  -- Calculate the result
  exact h3

end NUMINAMATH_GPT_sum_even_102_to_600_l1758_175825


namespace NUMINAMATH_GPT_paul_spent_81_90_l1758_175890

-- Define the original price of each racket
def originalPrice : ℝ := 60

-- Define the discount rates
def firstDiscount : ℝ := 0.20
def secondDiscount : ℝ := 0.50

-- Define the sales tax rate
def salesTax : ℝ := 0.05

-- Define the prices after discount
def firstRacketPrice : ℝ := originalPrice * (1 - firstDiscount)
def secondRacketPrice : ℝ := originalPrice * (1 - secondDiscount)

-- Define the total price before tax
def totalPriceBeforeTax : ℝ := firstRacketPrice + secondRacketPrice

-- Define the total sales tax
def totalSalesTax : ℝ := totalPriceBeforeTax * salesTax

-- Define the total amount spent
def totalAmountSpent : ℝ := totalPriceBeforeTax + totalSalesTax

-- The statement to prove
theorem paul_spent_81_90 : totalAmountSpent = 81.90 := 
by
  sorry

end NUMINAMATH_GPT_paul_spent_81_90_l1758_175890


namespace NUMINAMATH_GPT_base_angles_isosceles_triangle_l1758_175814

-- Define the conditions
def isIsoscelesTriangle (A B C : ℝ) : Prop :=
  (A = B ∨ B = C ∨ C = A)

def exteriorAngle (A B C : ℝ) (ext_angle : ℝ) : Prop :=
  ext_angle = (180 - (A + B)) ∨ ext_angle = (180 - (B + C)) ∨ ext_angle = (180 - (C + A))

-- Define the theorem
theorem base_angles_isosceles_triangle (A B C : ℝ) (ext_angle : ℝ) :
  isIsoscelesTriangle A B C ∧ exteriorAngle A B C ext_angle ∧ ext_angle = 110 →
  A = 55 ∨ A = 70 ∨ B = 55 ∨ B = 70 ∨ C = 55 ∨ C = 70 :=
by sorry

end NUMINAMATH_GPT_base_angles_isosceles_triangle_l1758_175814


namespace NUMINAMATH_GPT_arithmetic_sum_l1758_175893

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h1 : a 1 = 2)
  (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sum_l1758_175893


namespace NUMINAMATH_GPT_g_18_66_l1758_175839

def g (x y : ℕ) : ℕ := sorry

axiom g_prop1 : ∀ x, g x x = x
axiom g_prop2 : ∀ x y, g x y = g y x
axiom g_prop3 : ∀ x y, (x + 2 * y) * g x y = y * g x (x + 2 * y)

theorem g_18_66 : g 18 66 = 198 :=
by
  sorry

end NUMINAMATH_GPT_g_18_66_l1758_175839


namespace NUMINAMATH_GPT_booth_earnings_after_5_days_l1758_175886

def booth_daily_popcorn_earnings := 50
def booth_daily_cotton_candy_earnings := 3 * booth_daily_popcorn_earnings
def booth_total_daily_earnings := booth_daily_popcorn_earnings + booth_daily_cotton_candy_earnings
def booth_total_expenses := 30 + 75

theorem booth_earnings_after_5_days :
  5 * booth_total_daily_earnings - booth_total_expenses = 895 :=
by
  sorry

end NUMINAMATH_GPT_booth_earnings_after_5_days_l1758_175886


namespace NUMINAMATH_GPT_room_length_l1758_175872

theorem room_length (length width rate cost : ℝ)
    (h_width : width = 3.75)
    (h_rate : rate = 1000)
    (h_cost : cost = 20625)
    (h_eq : cost = length * width * rate) :
    length = 5.5 :=
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_room_length_l1758_175872


namespace NUMINAMATH_GPT_parabola_equation_l1758_175836

theorem parabola_equation (vertex focus : ℝ × ℝ) 
  (h_vertex : vertex = (0, 0)) 
  (h_focus_line : ∃ x y : ℝ, focus = (x, y) ∧ x - y + 2 = 0) 
  (h_symmetry_axis : ∃ axis : ℝ × ℝ → ℝ, ∀ p : ℝ × ℝ, axis p = 0): 
  ∃ k : ℝ, k > 0 ∧ (∀ x y : ℝ, y^2 = -8*x ∨ x^2 = 8*y) :=
by {
  sorry
}

end NUMINAMATH_GPT_parabola_equation_l1758_175836


namespace NUMINAMATH_GPT_cube_root_of_neg_eight_squared_is_neg_four_l1758_175833

-- Define the value of -8^2
def neg_eight_squared : ℤ := -8^2

-- Define what it means for a number to be the cube root of another number
def is_cube_root (a b : ℤ) : Prop := a^3 = b

-- The desired proof statement
theorem cube_root_of_neg_eight_squared_is_neg_four :
  neg_eight_squared = -64 → is_cube_root (-4) neg_eight_squared :=
by
  sorry

end NUMINAMATH_GPT_cube_root_of_neg_eight_squared_is_neg_four_l1758_175833


namespace NUMINAMATH_GPT_decimal_difference_l1758_175870

theorem decimal_difference : (0.650 : ℝ) - (1 / 8 : ℝ) = 0.525 := by
  sorry

end NUMINAMATH_GPT_decimal_difference_l1758_175870


namespace NUMINAMATH_GPT_sin_315_degree_l1758_175898

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end NUMINAMATH_GPT_sin_315_degree_l1758_175898


namespace NUMINAMATH_GPT_mrs_hilt_total_distance_l1758_175877

def total_distance_walked (d n : ℕ) : ℕ := 2 * d * n

theorem mrs_hilt_total_distance :
  total_distance_walked 30 4 = 240 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_mrs_hilt_total_distance_l1758_175877


namespace NUMINAMATH_GPT_problem1_problem2_l1758_175857

section ArithmeticSequence

variable {a : ℕ → ℤ} {a1 a5 a8 a6 a4 d : ℤ}

-- Problem 1: Prove that if a_5 = -1 and a_8 = 2, then a_1 = -5 and d = 1
theorem problem1 
  (h1 : a 5 = -1) 
  (h2 : a 8 = 2)
  (h3 : ∀ n, a n = a1 + n * d) : 
  a1 = -5 ∧ d = 1 := 
sorry 

-- Problem 2: Prove that if a_1 + a_6 = 12 and a_4 = 7, then a_9 = 17
theorem problem2 
  (h1 : a1 + a 6 = 12) 
  (h2 : a 4 = 7)
  (h3 : ∀ n, a n = a1 + n * d) 
  (h4 : ∀ m (hm : m ≠ 0), a1 = a 1): 
   a 9 = 17 := 
sorry

end ArithmeticSequence

end NUMINAMATH_GPT_problem1_problem2_l1758_175857


namespace NUMINAMATH_GPT_boy_lap_time_l1758_175858

noncomputable def total_time_needed
  (side_lengths : List ℝ)
  (running_speeds : List ℝ)
  (obstacle_time : ℝ) : ℝ :=
(side_lengths.zip running_speeds).foldl (λ (acc : ℝ) ⟨len, speed⟩ => acc + (len / (speed / 60))) 0
+ obstacle_time

theorem boy_lap_time
  (side_lengths : List ℝ)
  (running_speeds : List ℝ)
  (obstacle_time : ℝ) :
  side_lengths = [80, 120, 140, 100, 60] →
  running_speeds = [250, 200, 300, 166.67, 266.67] →
  obstacle_time = 5 →
  total_time_needed side_lengths running_speeds obstacle_time = 7.212 := by
  intros h_lengths h_speeds h_obstacle_time
  rw [h_lengths, h_speeds, h_obstacle_time]
  sorry

end NUMINAMATH_GPT_boy_lap_time_l1758_175858


namespace NUMINAMATH_GPT_ln_of_x_sq_sub_2x_monotonic_l1758_175887

noncomputable def ln_of_x_sq_sub_2x : ℝ → ℝ := fun x => Real.log (x^2 - 2*x)

theorem ln_of_x_sq_sub_2x_monotonic : ∀ x y : ℝ, (2 < x ∧ 2 < y ∧ x ≤ y) → ln_of_x_sq_sub_2x x ≤ ln_of_x_sq_sub_2x y :=
by
    intros x y h
    sorry

end NUMINAMATH_GPT_ln_of_x_sq_sub_2x_monotonic_l1758_175887


namespace NUMINAMATH_GPT_total_packages_sold_l1758_175808

variable (P : ℕ)

/-- An automobile parts supplier charges 25 per package of gaskets. 
    When a customer orders more than 10 packages of gaskets, the supplier charges 4/5 the price for each package in excess of 10.
    During a certain week, the supplier received 1150 in payment for the gaskets. --/
def cost (P : ℕ) : ℕ :=
  if P > 10 then 250 + (P - 10) * 20 else P * 25

theorem total_packages_sold :
  cost P = 1150 → P = 55 := by
  sorry

end NUMINAMATH_GPT_total_packages_sold_l1758_175808


namespace NUMINAMATH_GPT_math_problem_l1758_175878

theorem math_problem
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a^3 + b^3 = 2) :
  (a + b) * (a^5 + b^5) ≥ 4 ∧ a + b ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l1758_175878


namespace NUMINAMATH_GPT_intersection_A_B_l1758_175859

def A (x : ℝ) : Prop := (2 * x - 1 > 0)
def B (x : ℝ) : Prop := (x * (x - 2) < 0)

theorem intersection_A_B :
  {x : ℝ | A x ∧ B x} = {x : ℝ | 1 / 2 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1758_175859


namespace NUMINAMATH_GPT_extreme_points_l1758_175838

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem extreme_points (a b : ℝ) 
  (h1 : 3*(-2)^2 + 2*a*(-2) + b = 0) 
  (h2 : 3*(4)^2 + 2*a*(4) + b = 0) : 
  a - b = 21 :=
by sorry

end NUMINAMATH_GPT_extreme_points_l1758_175838


namespace NUMINAMATH_GPT_lineD_intersects_line1_l1758_175854

-- Define the lines based on the conditions
def line1 (x y : ℝ) := x + y - 1 = 0
def lineA (x y : ℝ) := 2 * x + 2 * y = 6
def lineB (x y : ℝ) := x + y = 0
def lineC (x y : ℝ) := y = -x - 3
def lineD (x y : ℝ) := y = x - 1

-- Define the statement that line D intersects with line1
theorem lineD_intersects_line1 : ∃ (x y : ℝ), line1 x y ∧ lineD x y :=
by
  sorry

end NUMINAMATH_GPT_lineD_intersects_line1_l1758_175854


namespace NUMINAMATH_GPT_find_ab_l1758_175845

theorem find_ab (a b c : ℤ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) (h3 : a + b + c = 21) : a * b = 10 := 
sorry

end NUMINAMATH_GPT_find_ab_l1758_175845


namespace NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l1758_175863

theorem line_through_point_with_equal_intercepts
  (P : ℝ × ℝ) (hP : P = (1, 3))
  (intercepts_equal : ∃ a : ℝ, a ≠ 0 ∧ (∀ x y : ℝ, (x/a) + (y/a) = 1 → x + y = 4 ∨ 3*x - y = 0)) :
  ∃ a b c : ℝ, (a, b, c) = (3, -1, 0) ∨ (a, b, c) = (1, 1, -4) ∧ (∀ x y : ℝ, a*x + b*y + c = 0 → (x + y = 4 ∨ 3*x - y = 0)) := 
by
  sorry

end NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l1758_175863


namespace NUMINAMATH_GPT_total_gems_in_chest_l1758_175895

theorem total_gems_in_chest (diamonds rubies : ℕ) 
  (h_diamonds : diamonds = 45)
  (h_rubies : rubies = 5110) : 
  diamonds + rubies = 5155 := 
by 
  sorry

end NUMINAMATH_GPT_total_gems_in_chest_l1758_175895


namespace NUMINAMATH_GPT_geometric_sequence_a8_l1758_175804

theorem geometric_sequence_a8 {a : ℕ → ℝ} (h1 : a 1 * a 3 = 4) (h9 : a 9 = 256) :
  a 8 = 128 ∨ a 8 = -128 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a8_l1758_175804


namespace NUMINAMATH_GPT_transport_cost_l1758_175864

theorem transport_cost (mass_g: ℕ) (cost_per_kg : ℕ) (mass_kg : ℝ) 
  (h1 : mass_g = 300) (h2 : mass_kg = (mass_g : ℝ) / 1000) 
  (h3: cost_per_kg = 18000)
  : mass_kg * cost_per_kg = 5400 := by
  sorry

end NUMINAMATH_GPT_transport_cost_l1758_175864


namespace NUMINAMATH_GPT_remainder_of_sums_modulo_l1758_175803

theorem remainder_of_sums_modulo :
  (2 * (8735 + 8736 + 8737 + 8738 + 8739)) % 11 = 8 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sums_modulo_l1758_175803


namespace NUMINAMATH_GPT_jane_ate_four_pieces_l1758_175849

def total_pieces : ℝ := 12.0
def num_people : ℝ := 3.0
def pieces_per_person : ℝ := 4.0

theorem jane_ate_four_pieces :
  total_pieces / num_people = pieces_per_person := 
  by
    sorry

end NUMINAMATH_GPT_jane_ate_four_pieces_l1758_175849


namespace NUMINAMATH_GPT_num_five_ruble_coins_l1758_175819

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end NUMINAMATH_GPT_num_five_ruble_coins_l1758_175819


namespace NUMINAMATH_GPT_CarriageSharingEquation_l1758_175811

theorem CarriageSharingEquation (x : ℕ) :
  (x / 3 + 2 = (x - 9) / 2) ↔
  (3 * ((x - 9) / 2) + 2 * 3 = x / 3 + 2) ∧ 
  (2 * ((x - 9) / 2) + 9 = x ∨ 2 * ((x - 9) / 2) + 9 < x) ∧ 
  (x / 3 + 2 < 3 * (x / 2) + 2 * 2 ∨ x / 3 + 2 = 3 * (x / 2) + 2 * 2) :=
sorry

end NUMINAMATH_GPT_CarriageSharingEquation_l1758_175811


namespace NUMINAMATH_GPT_paths_E_to_G_through_F_and_H_l1758_175815

-- Define positions of E, F, H, and G on the grid.
structure Point where
  x : ℕ
  y : ℕ

def E : Point := { x := 0, y := 0 }
def F : Point := { x := 3, y := 2 }
def H : Point := { x := 5, y := 4 }
def G : Point := { x := 8, y := 4 }

-- Function to calculate number of paths from one point to another given the number of right and down steps
def paths (start goal : Point) : ℕ :=
  let right_steps := goal.x - start.x
  let down_steps := goal.y - start.y
  Nat.choose (right_steps + down_steps) right_steps

theorem paths_E_to_G_through_F_and_H : paths E F * paths F H * paths H G = 60 := by
  sorry

end NUMINAMATH_GPT_paths_E_to_G_through_F_and_H_l1758_175815


namespace NUMINAMATH_GPT_arccos_zero_l1758_175821

theorem arccos_zero : Real.arccos 0 = Real.pi / 2 := 
by 
  sorry

end NUMINAMATH_GPT_arccos_zero_l1758_175821


namespace NUMINAMATH_GPT_milan_minutes_billed_l1758_175802

noncomputable def total_bill : ℝ := 23.36
noncomputable def monthly_fee : ℝ := 2.00
noncomputable def cost_per_minute : ℝ := 0.12

theorem milan_minutes_billed :
  (total_bill - monthly_fee) / cost_per_minute = 178 := 
sorry

end NUMINAMATH_GPT_milan_minutes_billed_l1758_175802


namespace NUMINAMATH_GPT_dinner_handshakes_l1758_175822

def num_couples := 8
def num_people_per_couple := 2
def num_attendees := num_couples * num_people_per_couple

def shakes_per_person (n : Nat) := n - 2
def total_possible_shakes (n : Nat) := (n * shakes_per_person n) / 2

theorem dinner_handshakes : total_possible_shakes num_attendees = 112 :=
by
  sorry

end NUMINAMATH_GPT_dinner_handshakes_l1758_175822


namespace NUMINAMATH_GPT_LiFangOutfitChoices_l1758_175897

variable (shirts skirts dresses : Nat) 

theorem LiFangOutfitChoices (h_shirts : shirts = 4) (h_skirts : skirts = 3) (h_dresses : dresses = 2) :
  shirts * skirts + dresses = 14 :=
by 
  -- Given the conditions and the calculations, the expected result follows.
  sorry

end NUMINAMATH_GPT_LiFangOutfitChoices_l1758_175897


namespace NUMINAMATH_GPT_seventh_grade_problem_l1758_175800

theorem seventh_grade_problem (x y : ℕ) (h1 : x + y = 12) (h2 : 6 * x = 3 * 4 * y) :
  (x + y = 12 ∧ 6 * x = 3 * 4 * y) :=
by
  apply And.intro
  . exact h1
  . exact h2

end NUMINAMATH_GPT_seventh_grade_problem_l1758_175800


namespace NUMINAMATH_GPT_find_K_l1758_175860

noncomputable def cylinder_paint (r h : ℝ) : ℝ := 2 * Real.pi * r * h
noncomputable def cube_surface_area (s : ℝ) : ℝ := 6 * s^2
noncomputable def cube_volume (s : ℝ) : ℝ := s^3

theorem find_K :
  (cylinder_paint 3 4 = 24 * Real.pi) →
  (∃ s, cube_surface_area s = 24 * Real.pi ∧ cube_volume s = 48 / Real.sqrt K) →
  K = 36 / Real.pi^3 :=
by
  sorry

end NUMINAMATH_GPT_find_K_l1758_175860


namespace NUMINAMATH_GPT_container_volumes_l1758_175862

theorem container_volumes (a r : ℝ) (h1 : (2 * a)^3 = (4 / 3) * Real.pi * r^3) :
  ((2 * a + 2)^3 > (4 / 3) * Real.pi * (r + 1)^3) :=
by sorry

end NUMINAMATH_GPT_container_volumes_l1758_175862


namespace NUMINAMATH_GPT_tetrahedron_fourth_face_possibilities_l1758_175826

theorem tetrahedron_fourth_face_possibilities :
  ∃ (S : Set String), S = {"right-angled triangle", "acute-angled triangle", "isosceles triangle", "isosceles right-angled triangle", "equilateral triangle"} :=
sorry

end NUMINAMATH_GPT_tetrahedron_fourth_face_possibilities_l1758_175826


namespace NUMINAMATH_GPT_B_alone_finishes_in_19_point_5_days_l1758_175834

-- Define the conditions
def is_half_good(A B : ℝ) : Prop := A = 1 / 2 * B
def together_finish_in_13_days(A B : ℝ) : Prop := (A + B) * 13 = 1

-- Define the statement
theorem B_alone_finishes_in_19_point_5_days (A B : ℝ) (h1 : is_half_good A B) (h2 : together_finish_in_13_days A B) :
  B * 19.5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_B_alone_finishes_in_19_point_5_days_l1758_175834


namespace NUMINAMATH_GPT_sparrows_among_non_pigeons_l1758_175844

theorem sparrows_among_non_pigeons (perc_sparrows perc_pigeons perc_parrots perc_crows : ℝ)
  (h_sparrows : perc_sparrows = 0.40)
  (h_pigeons : perc_pigeons = 0.20)
  (h_parrots : perc_parrots = 0.15)
  (h_crows : perc_crows = 0.25) :
  (perc_sparrows / (1 - perc_pigeons) * 100) = 50 :=
by
  sorry

end NUMINAMATH_GPT_sparrows_among_non_pigeons_l1758_175844


namespace NUMINAMATH_GPT_rectangle_area_y_l1758_175801

theorem rectangle_area_y (y : ℝ) (h_y_pos : y > 0)
  (h_area : (3 * y = 21)) : y = 7 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_y_l1758_175801


namespace NUMINAMATH_GPT_total_miles_walked_l1758_175824

def weekly_group_walk_miles : ℕ := 3 * 6

def Jamie_additional_walk_miles_per_week : ℕ := 2 * 6
def Sue_additional_walk_miles_per_week : ℕ := 1 * 6 -- half of Jamie's additional walk
def Laura_additional_walk_miles_per_week : ℕ := 1 * 3 -- 1 mile every two days for 6 days
def Melissa_additional_walk_miles_per_week : ℕ := 2 * 2 -- 2 miles every three days for 6 days
def Katie_additional_walk_miles_per_week : ℕ := 1 * 6

def Jamie_weekly_miles : ℕ := weekly_group_walk_miles + Jamie_additional_walk_miles_per_week
def Sue_weekly_miles : ℕ := weekly_group_walk_miles + Sue_additional_walk_miles_per_week
def Laura_weekly_miles : ℕ := weekly_group_walk_miles + Laura_additional_walk_miles_per_week
def Melissa_weekly_miles : ℕ := weekly_group_walk_miles + Melissa_additional_walk_miles_per_week
def Katie_weekly_miles : ℕ := weekly_group_walk_miles + Katie_additional_walk_miles_per_week

def weeks_in_month : ℕ := 4

def Jamie_monthly_miles : ℕ := Jamie_weekly_miles * weeks_in_month
def Sue_monthly_miles : ℕ := Sue_weekly_miles * weeks_in_month
def Laura_monthly_miles : ℕ := Laura_weekly_miles * weeks_in_month
def Melissa_monthly_miles : ℕ := Melissa_weekly_miles * weeks_in_month
def Katie_monthly_miles : ℕ := Katie_weekly_miles * weeks_in_month

def total_monthly_miles : ℕ :=
  Jamie_monthly_miles + Sue_monthly_miles + Laura_monthly_miles + Melissa_monthly_miles + Katie_monthly_miles

theorem total_miles_walked (month_has_30_days : Prop) : total_monthly_miles = 484 :=
by
  unfold total_monthly_miles
  unfold Jamie_monthly_miles Sue_monthly_miles Laura_monthly_miles Melissa_monthly_miles Katie_monthly_miles
  unfold Jamie_weekly_miles Sue_weekly_miles Laura_weekly_miles Melissa_weekly_miles Katie_weekly_miles
  unfold weekly_group_walk_miles Jamie_additional_walk_miles_per_week Sue_additional_walk_miles_per_week Laura_additional_walk_miles_per_week Melissa_additional_walk_miles_per_week Katie_additional_walk_miles_per_week
  unfold weeks_in_month
  sorry

end NUMINAMATH_GPT_total_miles_walked_l1758_175824


namespace NUMINAMATH_GPT_nat_representation_l1758_175818

theorem nat_representation (k : ℕ) : ∃ n r : ℕ, (r = 0 ∨ r = 1 ∨ r = 2) ∧ k = 3 * n + r :=
by
  sorry

end NUMINAMATH_GPT_nat_representation_l1758_175818


namespace NUMINAMATH_GPT_equivalent_proposition_l1758_175843

variable (M : Set α) (m n : α)

theorem equivalent_proposition :
  (m ∈ M → n ∉ M) ↔ (n ∈ M → m ∉ M) := by
  sorry

end NUMINAMATH_GPT_equivalent_proposition_l1758_175843


namespace NUMINAMATH_GPT_inradius_of_triangle_area_twice_perimeter_l1758_175874

theorem inradius_of_triangle_area_twice_perimeter (A p r s : ℝ) (hA : A = 2 * p) (hs : p = 2 * s)
  (hA_formula : A = r * s) : r = 4 :=
by
  sorry

end NUMINAMATH_GPT_inradius_of_triangle_area_twice_perimeter_l1758_175874


namespace NUMINAMATH_GPT_not_prime_expression_l1758_175892

theorem not_prime_expression (x y : ℕ) : ¬ Prime (x^8 - x^7 * y + x^6 * y^2 - x^5 * y^3 + x^4 * y^4 
  - x^3 * y^5 + x^2 * y^6 - x * y^7 + y^8) :=
sorry

end NUMINAMATH_GPT_not_prime_expression_l1758_175892


namespace NUMINAMATH_GPT_ned_initially_had_games_l1758_175894

variable (G : ℕ)

theorem ned_initially_had_games (h1 : (3 / 4) * (2 / 3) * G = 6) : G = 12 := by
  sorry

end NUMINAMATH_GPT_ned_initially_had_games_l1758_175894


namespace NUMINAMATH_GPT_find_a_of_exp_function_l1758_175876

theorem find_a_of_exp_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a ^ 2 = 9) : a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_of_exp_function_l1758_175876


namespace NUMINAMATH_GPT_card_giving_ratio_l1758_175853

theorem card_giving_ratio (initial_cards cards_to_Bob cards_left : ℕ) 
  (h1 : initial_cards = 18) 
  (h2 : cards_to_Bob = 3)
  (h3 : cards_left = 9) : 
  (initial_cards - cards_left - cards_to_Bob) / gcd (initial_cards - cards_left - cards_to_Bob) cards_to_Bob = 2 / 1 :=
by sorry

end NUMINAMATH_GPT_card_giving_ratio_l1758_175853


namespace NUMINAMATH_GPT_number_of_pages_500_l1758_175885

-- Define the conditions as separate constants
def cost_per_page : ℕ := 3 -- cents
def total_cents : ℕ := 1500 

-- Define the number of pages calculation
noncomputable def number_of_pages := total_cents / cost_per_page

-- Statement we want to prove
theorem number_of_pages_500 : number_of_pages = 500 :=
sorry

end NUMINAMATH_GPT_number_of_pages_500_l1758_175885


namespace NUMINAMATH_GPT_solve_inequality_l1758_175807

theorem solve_inequality (x : ℝ) (h : 5 * x - 12 ≤ 2 * (4 * x - 3)) : x ≥ -2 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l1758_175807


namespace NUMINAMATH_GPT_find_sum_of_cubes_l1758_175865

theorem find_sum_of_cubes (a b c : ℝ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : (a^3 + 9) / a = (b^3 + 9) / b)
  (h₅ : (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_cubes_l1758_175865


namespace NUMINAMATH_GPT_find_values_general_formula_l1758_175888

variable (a_n S_n : ℕ → ℝ)

-- Conditions
axiom sum_sequence (n : ℕ) (hn : n > 0) :  S_n n = (1 / 3) * (a_n n - 1)

-- Questions
theorem find_values :
  (a_n 1 = 2) ∧ (a_n 2 = 5) ∧ (a_n 3 = 8) := sorry

theorem general_formula (n : ℕ) :
  n > 0 → a_n n = n + 1 := sorry

end NUMINAMATH_GPT_find_values_general_formula_l1758_175888


namespace NUMINAMATH_GPT_positive_solution_l1758_175868

theorem positive_solution (x : ℝ) (h : (1 / 2) * (3 * x^2 - 1) = (x^2 - 50 * x - 10) * (x^2 + 25 * x + 5)) : x = 25 + Real.sqrt 159 :=
sorry

end NUMINAMATH_GPT_positive_solution_l1758_175868


namespace NUMINAMATH_GPT_negation_proposition_l1758_175883

-- Define the proposition as a Lean function
def quadratic_non_negative (x : ℝ) : Prop := x^2 - 2*x + 1 ≥ 0

-- State the theorem that we need to prove
theorem negation_proposition : ∀ x : ℝ, quadratic_non_negative x :=
by 
  sorry

end NUMINAMATH_GPT_negation_proposition_l1758_175883


namespace NUMINAMATH_GPT_m_range_l1758_175889

theorem m_range (m : ℝ) :
  (∀ x : ℝ, 1 < x → 2 * x + m + 2 / (x - 1) > 0) ↔ m > -6 :=
by
  -- The proof will be provided later
  sorry

end NUMINAMATH_GPT_m_range_l1758_175889


namespace NUMINAMATH_GPT_arithmetic_sequence_a10_l1758_175861

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (d : ℕ) 
  (h_seq : ∀ n, a (n + 1) = a n + d) 
  (h_positive : ∀ n, a n > 0) 
  (h_sum : a 1 + a 2 + a 3 = 15) 
  (h_geo : (a 1 + 2) * (a 3 + 13) = (a 2 + 5) * (a 2 + 5))  
  : a 10 = 21 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_a10_l1758_175861


namespace NUMINAMATH_GPT_journey_distance_l1758_175835

theorem journey_distance :
  ∃ D : ℝ, ((D / 2) / 21 + (D / 2) / 24 = 10) ∧ D = 224 :=
by
  use 224
  sorry

end NUMINAMATH_GPT_journey_distance_l1758_175835


namespace NUMINAMATH_GPT_circle_standard_equation_l1758_175817

theorem circle_standard_equation (a : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - 1)^2 = (x - 1 + y - 1)^2) ∧
  (∀ x y : ℝ, (x - a)^2 + (y - 1)^2 = (x - 1 + y + 2)^2) →
  (∃ x y : ℝ, (x - 2) ^ 2 + (y - 1) ^ 2 = 2) :=
sorry

end NUMINAMATH_GPT_circle_standard_equation_l1758_175817


namespace NUMINAMATH_GPT_shape_is_plane_l1758_175855

noncomputable
def cylindrical_coordinates_shape (r θ z c : ℝ) := θ = 2 * c

theorem shape_is_plane (c : ℝ) : 
  ∀ (r : ℝ) (θ : ℝ) (z : ℝ), cylindrical_coordinates_shape r θ z c → (θ = 2 * c) :=
by
  sorry

end NUMINAMATH_GPT_shape_is_plane_l1758_175855


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1758_175875

theorem sufficient_but_not_necessary (a b : ℝ) : (a > b ∧ b > 0) → (a^2 > b^2) ∧ ¬((a^2 > b^2) → (a > b ∧ b > 0)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1758_175875


namespace NUMINAMATH_GPT_Jessie_points_l1758_175891

theorem Jessie_points (total_points team_points : ℕ) (players_points : ℕ) (P Q R : ℕ) (eq1 : total_points = 311) (eq2 : players_points = 188) (eq3 : team_points - players_points = 3 * P) (eq4 : P = Q) (eq5 : Q = R) : Q = 41 :=
by
  sorry

end NUMINAMATH_GPT_Jessie_points_l1758_175891


namespace NUMINAMATH_GPT_convex_polyhedron_theorems_l1758_175866

-- Definitions for convex polyhedron and symmetric properties
structure ConvexSymmetricPolyhedron (α : Type*) :=
  (isConvex : Bool)
  (isCentrallySymmetric : Bool)
  (crossSection : α → α → α)
  (center : α)

-- Definitions for proofs required
def largest_cross_section_area
  (P : ConvexSymmetricPolyhedron ℝ) : Prop :=
  ∀ (p : ℝ), P.crossSection p P.center ≤ P.crossSection P.center P.center

def largest_radius_circle (P : ConvexSymmetricPolyhedron ℝ) : Prop :=
  ¬∀ (p : ℝ), P.crossSection p P.center = P.crossSection P.center P.center

-- The theorem combining both statements
theorem convex_polyhedron_theorems
  (P : ConvexSymmetricPolyhedron ℝ) :
  P.isConvex = true ∧ 
  P.isCentrallySymmetric = true →
  (largest_cross_section_area P) ∧ (largest_radius_circle P) :=
by 
  sorry

end NUMINAMATH_GPT_convex_polyhedron_theorems_l1758_175866


namespace NUMINAMATH_GPT_solve_inequality_l1758_175871

theorem solve_inequality (a : ℝ) : (6 * x^2 + a * x - a^2 < 0) ↔
  ((a > 0) ∧ (-a / 2 < x ∧ x < a / 3)) ∨
  ((a < 0) ∧ (a / 3 < x ∧ x < -a / 2)) ∨
  ((a = 0) ∧ false) :=
by 
  sorry

end NUMINAMATH_GPT_solve_inequality_l1758_175871


namespace NUMINAMATH_GPT_volume_of_stone_l1758_175820

theorem volume_of_stone 
  (width length initial_height final_height : ℕ)
  (h_width : width = 15)
  (h_length : length = 20)
  (h_initial_height : initial_height = 10)
  (h_final_height : final_height = 15)
  : (width * length * final_height - width * length * initial_height = 1500) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_stone_l1758_175820
