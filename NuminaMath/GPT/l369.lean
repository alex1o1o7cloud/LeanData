import Mathlib

namespace NUMINAMATH_GPT_red_marble_count_l369_36999

theorem red_marble_count (x y : ℕ) (total_yellow : ℕ) (total_diff : ℕ) 
  (jar1_ratio_red jar1_ratio_yellow : ℕ) (jar2_ratio_red jar2_ratio_yellow : ℕ) 
  (h1 : jar1_ratio_red = 7) (h2 : jar1_ratio_yellow = 2) 
  (h3 : jar2_ratio_red = 5) (h4 : jar2_ratio_yellow = 3) 
  (h5 : 2 * x + 3 * y = 50) (h6 : 8 * y = 9 * x + 20) :
  7 * x + 2 = 5 * y :=
sorry

end NUMINAMATH_GPT_red_marble_count_l369_36999


namespace NUMINAMATH_GPT_least_x_l369_36904

theorem least_x (x p : ℕ) (h1 : 0 < x) (h2: Nat.Prime p) (h3: ∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ x = 11 * p * q) : x ≥ 66 := 
sorry

end NUMINAMATH_GPT_least_x_l369_36904


namespace NUMINAMATH_GPT_NinaCalculationCorrectAnswer_l369_36920

variable (y : ℝ)

noncomputable def NinaMistakenCalculation (y : ℝ) : ℝ :=
(y + 25) * 5

noncomputable def NinaCorrectCalculation (y : ℝ) : ℝ :=
(y - 25) / 5

theorem NinaCalculationCorrectAnswer (hy : (NinaMistakenCalculation y) = 200) :
  (NinaCorrectCalculation y) = -2 := by
  sorry

end NUMINAMATH_GPT_NinaCalculationCorrectAnswer_l369_36920


namespace NUMINAMATH_GPT_symmetrical_polynomial_l369_36959

noncomputable def Q (x : ℝ) (f g h i j k : ℝ) : ℝ :=
  x^6 + f * x^5 + g * x^4 + h * x^3 + i * x^2 + j * x + k

theorem symmetrical_polynomial (f g h i j k : ℝ) :
  (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ Q 0 f g h i j k = 0 ∧
    Q x f g h i j k = x * (x - a) * (x + a) * (x - b) * (x + b) * (x - c) ∧
    Q x f g h i j k = Q (-x) f g h i j k) →
  f = 0 :=
by sorry

end NUMINAMATH_GPT_symmetrical_polynomial_l369_36959


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l369_36953

variables (a : ℕ → ℝ) (q : ℝ)
axiom h1 : a 1 = 2
axiom h2 : ∀ n : ℕ, a (n + 1) - a n ≠ 0 -- Common difference is non-zero
axiom h3 : a 3 = (a 1) * q
axiom h4 : a 11 = (a 1) * q^2
axiom h5 : a 11 = a 1 + 5 * (a 3 - a 1)

theorem common_ratio_of_geometric_sequence : q = 4 := 
by sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l369_36953


namespace NUMINAMATH_GPT_gabby_needs_more_money_l369_36941

theorem gabby_needs_more_money (cost_saved : ℕ) (initial_saved : ℕ) (additional_money : ℕ) (cost_remaining : ℕ) :
  cost_saved = 65 → initial_saved = 35 → additional_money = 20 → cost_remaining = (cost_saved - initial_saved) - additional_money → cost_remaining = 10 :=
by
  intros h_cost_saved h_initial_saved h_additional_money h_cost_remaining
  simp [h_cost_saved, h_initial_saved, h_additional_money] at h_cost_remaining
  exact h_cost_remaining

end NUMINAMATH_GPT_gabby_needs_more_money_l369_36941


namespace NUMINAMATH_GPT_determine_a_l369_36924

theorem determine_a (a : ℝ) (h : ∃ r : ℝ, (a / (1+1*I : ℂ) + (1+1*I : ℂ) / 2).im = 0) : a = 1 :=
sorry

end NUMINAMATH_GPT_determine_a_l369_36924


namespace NUMINAMATH_GPT_find_constants_l369_36947

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ -2 → 
  (x^3 - x - 4) / ((x - 1) * (x - 4) * (x + 2)) = 
  A / (x - 1) + B / (x - 4) + C / (x + 2)) →
  A = 4 / 9 ∧ B = 28 / 9 ∧ C = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l369_36947


namespace NUMINAMATH_GPT_multiplication_approximation_correct_l369_36992

noncomputable def closest_approximation (x : ℝ) : ℝ := 
  if 15700 <= x ∧ x < 15750 then 15700
  else if 15750 <= x ∧ x < 15800 then 15750
  else if 15800 <= x ∧ x < 15900 then 15800
  else if 15900 <= x ∧ x < 16000 then 15900
  else 16000

theorem multiplication_approximation_correct :
  closest_approximation (0.00525 * 3153420) = 15750 := 
by
  sorry

end NUMINAMATH_GPT_multiplication_approximation_correct_l369_36992


namespace NUMINAMATH_GPT_factor_check_l369_36921

theorem factor_check :
  ∃ (f : ℕ → ℕ) (x : ℝ), f 1 = (x^2 - 2 * x + 3) ∧ f 2 = 29 * 37 * x^4 + 2 * x^2 + 9 :=
by
  let f : ℕ → ℕ := sorry -- Define a sequence or function for the proof context
  let x : ℝ := sorry -- Define the variable x in our context
  have h₁ : f 1 = (x^2 - 2 * x + 3) := sorry -- Establish the first factor
  have h₂ : f 2 = 29 * 37 * x^4 + 2 * x^2 + 9 := sorry -- Establish the polynomial expression
  exact ⟨f, x, h₁, h₂⟩ -- Use existential quantifier to capture the required form

end NUMINAMATH_GPT_factor_check_l369_36921


namespace NUMINAMATH_GPT_total_stock_worth_is_15000_l369_36988

-- Define the total worth of the stock
variable (X : ℝ)

-- Define the conditions
def stock_condition_1 := 0.20 * X -- Worth of 20% of the stock
def stock_condition_2 := 0.10 * (0.20 * X) -- Profit from 20% of the stock
def stock_condition_3 := 0.80 * X -- Worth of 80% of the stock
def stock_condition_4 := 0.05 * (0.80 * X) -- Loss from 80% of the stock
def overall_loss := 0.04 * X - 0.02 * X

-- The question rewritten as a theorem statement
theorem total_stock_worth_is_15000 (h1 : overall_loss X = 300) : X = 15000 :=
by sorry

end NUMINAMATH_GPT_total_stock_worth_is_15000_l369_36988


namespace NUMINAMATH_GPT_new_commission_percentage_l369_36991

theorem new_commission_percentage
  (fixed_salary : ℝ)
  (total_sales : ℝ)
  (sales_threshold : ℝ)
  (previous_commission_rate : ℝ)
  (additional_earnings : ℝ)
  (prev_commission : ℝ)
  (extra_sales : ℝ)
  (new_commission : ℝ)
  (new_remuneration : ℝ) :
  fixed_salary = 1000 →
  total_sales = 12000 →
  sales_threshold = 4000 →
  previous_commission_rate = 0.05 →
  additional_earnings = 600 →
  prev_commission = previous_commission_rate * total_sales →
  extra_sales = total_sales - sales_threshold →
  new_remuneration = fixed_salary + new_commission * extra_sales →
  new_remuneration = prev_commission + additional_earnings →
  new_commission = 2.5 / 100 :=
by
  intros
  sorry

end NUMINAMATH_GPT_new_commission_percentage_l369_36991


namespace NUMINAMATH_GPT_truncated_cone_volume_l369_36938

noncomputable def volume_of_truncated_cone (R r h : ℝ) : ℝ :=
  let V_large := (1 / 3) * Real.pi * R^2 * (h + h)  -- Height of larger cone is h + x = h + h
  let V_small := (1 / 3) * Real.pi * r^2 * h       -- Height of smaller cone is h
  V_large - V_small

theorem truncated_cone_volume (R r h : ℝ) (hR : R = 8) (hr : r = 4) (hh : h = 6) :
  volume_of_truncated_cone R r h = 224 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_truncated_cone_volume_l369_36938


namespace NUMINAMATH_GPT_find_a5_l369_36973

variable {a_n : ℕ → ℤ}
variable (d : ℤ)

def arithmetic_sequence (a_n : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  a_n 1 = a1 ∧ ∀ n, a_n (n + 1) = a_n n + d

theorem find_a5 (h_seq : arithmetic_sequence a_n 6 d) (h_a3 : a_n 3 = 2) : a_n 5 = -2 :=
by
  obtain ⟨h_a1, h_arith⟩ := h_seq
  sorry

end NUMINAMATH_GPT_find_a5_l369_36973


namespace NUMINAMATH_GPT_solve_for_y_l369_36950

theorem solve_for_y (y : ℚ) : 
  y + 5 / 8 = 2 / 9 + 1 / 2 → 
  y = 7 / 72 := 
by 
  intro h1
  sorry

end NUMINAMATH_GPT_solve_for_y_l369_36950


namespace NUMINAMATH_GPT_solution_proof_l369_36989

variable (x y z : ℝ)

-- Given system of equations
def equation1 := 6 / (3 * x + 4 * y) + 4 / (5 * x - 4 * z) = 7 / 12
def equation2 := 9 / (4 * y + 3 * z) - 4 / (3 * x + 4 * y) = 1 / 3
def equation3 := 2 / (5 * x - 4 * z) + 6 / (4 * y + 3 * z) = 1 / 2

theorem solution_proof : 
  equation1 4 3 2 ∧ equation2 4 3 2 ∧ equation3 4 3 2 := by
  sorry

end NUMINAMATH_GPT_solution_proof_l369_36989


namespace NUMINAMATH_GPT_regular_polygon_num_sides_l369_36981

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_num_sides_l369_36981


namespace NUMINAMATH_GPT_count_valid_five_digit_numbers_l369_36910

-- Define the conditions
def is_five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def is_divisible_by (a b : ℕ) : Prop := b ∣ a

def quotient_remainder_sum_divisible_by (n q r : ℕ) : Prop :=
  (n = 100 * q + r) ∧ ((q + r) % 7 = 0)

-- Define the theorem
theorem count_valid_five_digit_numbers : 
  ∃ k, k = 8160 ∧ ∀ n, is_five_digit_number n ∧ 
    is_divisible_by n 13 ∧ 
    ∃ q r, quotient_remainder_sum_divisible_by n q r → 
    k = 8160 :=
sorry

end NUMINAMATH_GPT_count_valid_five_digit_numbers_l369_36910


namespace NUMINAMATH_GPT_range_of_cos_neg_alpha_l369_36906

theorem range_of_cos_neg_alpha (α : ℝ) (h : 12 * (Real.sin α)^2 + Real.cos α > 11) :
  -1 / 4 < Real.cos (-α) ∧ Real.cos (-α) < 1 / 3 := 
sorry

end NUMINAMATH_GPT_range_of_cos_neg_alpha_l369_36906


namespace NUMINAMATH_GPT_shaded_fraction_l369_36965

theorem shaded_fraction {S : ℝ} (h : 0 < S) :
  let frac_area := ∑' n : ℕ, (1/(4:ℝ)^1) * (1/(4:ℝ)^n)
  1/3 = frac_area :=
by
  sorry

end NUMINAMATH_GPT_shaded_fraction_l369_36965


namespace NUMINAMATH_GPT_gcd_160_200_360_l369_36919

theorem gcd_160_200_360 : Nat.gcd (Nat.gcd 160 200) 360 = 40 := by
  sorry

end NUMINAMATH_GPT_gcd_160_200_360_l369_36919


namespace NUMINAMATH_GPT_modulo_problem_l369_36993

theorem modulo_problem :
  (47 ^ 2051 - 25 ^ 2051) % 5 = 3 := by
  sorry

end NUMINAMATH_GPT_modulo_problem_l369_36993


namespace NUMINAMATH_GPT_no_common_points_implies_parallel_l369_36942

variable (a : Type) (P : Type) [LinearOrder P] [AddGroupWithOne P]
variable (has_no_common_point : a → P → Prop)
variable (is_parallel : a → P → Prop)

theorem no_common_points_implies_parallel (a_line : a) (a_plane : P) :
  has_no_common_point a_line a_plane ↔ is_parallel a_line a_plane :=
sorry

end NUMINAMATH_GPT_no_common_points_implies_parallel_l369_36942


namespace NUMINAMATH_GPT_combined_rent_C_D_l369_36922

theorem combined_rent_C_D :
  let rent_per_month_area_z := 100
  let rent_per_month_area_w := 120
  let months_c := 3
  let months_d := 6
  let rent_c := months_c * rent_per_month_area_z
  let rent_d := months_d * rent_per_month_area_w
  let combined_rent := rent_c + rent_d
  combined_rent = 1020 :=
by
  let rent_per_month_area_z := 100
  let rent_per_month_area_w := 120
  let months_c := 3
  let months_d := 6
  let rent_c := months_c * rent_per_month_area_z
  let rent_d := months_d * rent_per_month_area_w
  let combined_rent := rent_c + rent_d
  show combined_rent = 1020
  sorry

end NUMINAMATH_GPT_combined_rent_C_D_l369_36922


namespace NUMINAMATH_GPT_find_exercise_books_l369_36974

theorem find_exercise_books
  (pencil_ratio pen_ratio exercise_book_ratio eraser_ratio : ℕ)
  (total_pencils total_ratio_units : ℕ)
  (h1 : pencil_ratio = 10)
  (h2 : pen_ratio = 2)
  (h3 : exercise_book_ratio = 3)
  (h4 : eraser_ratio = 4)
  (h5 : total_pencils = 150)
  (h6 : total_ratio_units = pencil_ratio + pen_ratio + exercise_book_ratio + eraser_ratio) :
  (total_pencils / pencil_ratio) * exercise_book_ratio = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_exercise_books_l369_36974


namespace NUMINAMATH_GPT_adults_wearing_hats_l369_36979

theorem adults_wearing_hats (total_adults : ℕ) (percent_men : ℝ) (percent_men_hats : ℝ) 
  (percent_women_hats : ℝ) (num_hats : ℕ) 
  (h1 : total_adults = 3600) 
  (h2 : percent_men = 0.40) 
  (h3 : percent_men_hats = 0.15) 
  (h4 : percent_women_hats = 0.25) 
  (h5 : num_hats = 756) : 
  (percent_men * total_adults) * percent_men_hats + (total_adults - (percent_men * total_adults)) * percent_women_hats = num_hats := 
sorry

end NUMINAMATH_GPT_adults_wearing_hats_l369_36979


namespace NUMINAMATH_GPT_vector_dot_product_l369_36957

open Complex

def a : Complex := (1 : ℝ) + (-(2 : ℝ)) * Complex.I
def b : Complex := (-3 : ℝ) + (4 : ℝ) * Complex.I
def c : Complex := (3 : ℝ) + (2 : ℝ) * Complex.I

-- Note: Using real coordinates to simulate vector operations.
theorem vector_dot_product :
  let a_vec := (1, -2)
  let b_vec := (-3, 4)
  let c_vec := (3, 2)
  let linear_combination := (a_vec.1 + 2 * b_vec.1, a_vec.2 + 2 * b_vec.2)
  (linear_combination.1 * c_vec.1 + linear_combination.2 * c_vec.2) = -3 := 
by
  sorry

end NUMINAMATH_GPT_vector_dot_product_l369_36957


namespace NUMINAMATH_GPT_inheritance_amount_l369_36977

theorem inheritance_amount (x : ℝ) (h1 : x * 0.25 + (x * 0.75) * 0.15 + 2500 = 16500) : x = 38621 := 
by
  sorry

end NUMINAMATH_GPT_inheritance_amount_l369_36977


namespace NUMINAMATH_GPT_area_of_rectangle_l369_36997

theorem area_of_rectangle (A G Y : ℝ) 
  (hG : G = 0.15 * A) 
  (hY : Y = 21) 
  (hG_plus_Y : G + Y = 0.5 * A) : 
  A = 60 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l369_36997


namespace NUMINAMATH_GPT_simplify_expression_l369_36969

theorem simplify_expression (x : ℝ) :
  (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * (x - 1) + 1 = x^4 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l369_36969


namespace NUMINAMATH_GPT_part1_l369_36929

open Set

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

theorem part1 (U_eq : U = univ) 
  (A_eq : A = {x | (x - 5) / (x - 2) ≤ 0}) 
  (B_eq : B = {x | 1 < x ∧ x < 3}) :
  compl A ∩ compl B = {x | x ≤ 1 ∨ x > 5} := 
  sorry

end NUMINAMATH_GPT_part1_l369_36929


namespace NUMINAMATH_GPT_number_of_blue_stamps_l369_36905

theorem number_of_blue_stamps (
    red_stamps : ℕ := 20
) (
    yellow_stamps : ℕ := 7
) (
    price_per_red_stamp : ℝ := 1.1
) (
    price_per_blue_stamp : ℝ := 0.8
) (
    total_earnings : ℝ := 100
) (
    price_per_yellow_stamp : ℝ := 2
) : red_stamps = 20 ∧ yellow_stamps = 7 ∧ price_per_red_stamp = 1.1 ∧ price_per_blue_stamp = 0.8 ∧ total_earnings = 100 ∧ price_per_yellow_stamp = 2 → ∃ (blue_stamps : ℕ), blue_stamps = 80 :=
by
  sorry

end NUMINAMATH_GPT_number_of_blue_stamps_l369_36905


namespace NUMINAMATH_GPT_annual_interest_income_l369_36963

variables (totalInvestment firstBondPrincipal secondBondPrincipal firstRate secondRate : ℝ)
           (firstInterest secondInterest totalInterest : ℝ)

def investment_conditions : Prop :=
  totalInvestment = 32000 ∧
  firstRate = 0.0575 ∧
  secondRate = 0.0625 ∧
  firstBondPrincipal = 20000 ∧
  secondBondPrincipal = totalInvestment - firstBondPrincipal

def calculate_interest (principal rate : ℝ) : ℝ := principal * rate

def total_annual_interest (firstInterest secondInterest : ℝ) : ℝ :=
  firstInterest + secondInterest

theorem annual_interest_income
  (hc : investment_conditions totalInvestment firstBondPrincipal secondBondPrincipal firstRate secondRate) :
  total_annual_interest (calculate_interest firstBondPrincipal firstRate)
    (calculate_interest secondBondPrincipal secondRate) = 1900 :=
by {
  sorry
}

end NUMINAMATH_GPT_annual_interest_income_l369_36963


namespace NUMINAMATH_GPT_express_y_l369_36945

theorem express_y (x y : ℝ) (h : 3 * x + 2 * y = 1) : y = (1 - 3 * x) / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_express_y_l369_36945


namespace NUMINAMATH_GPT_abs_value_condition_l369_36933

theorem abs_value_condition (m : ℝ) (h : |m - 1| = m - 1) : m ≥ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_abs_value_condition_l369_36933


namespace NUMINAMATH_GPT_find_total_income_l369_36927

theorem find_total_income (I : ℝ)
  (h1 : 0.6 * I + 0.3 * I + 0.005 * (I - (0.6 * I + 0.3 * I)) + 50000 = I) : 
  I = 526315.79 :=
by
  sorry

end NUMINAMATH_GPT_find_total_income_l369_36927


namespace NUMINAMATH_GPT_smallest_range_of_sample_l369_36940

open Real

theorem smallest_range_of_sample {a b c d e f g : ℝ}
  (h1 : (a + b + c + d + e + f + g) / 7 = 8)
  (h2 : d = 10)
  (h3 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e ∧ e ≤ f ∧ f ≤ g) :
  ∃ r, r = g - a ∧ r = 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_range_of_sample_l369_36940


namespace NUMINAMATH_GPT_total_cost_of_repair_l369_36925

noncomputable def cost_of_repair (tire_cost: ℝ) (num_tires: ℕ) (tax: ℝ) (city_fee: ℝ) (discount: ℝ) : ℝ :=
  let total_cost := (tire_cost * num_tires : ℝ)
  let total_tax := (tax * num_tires : ℝ)
  let total_city_fee := (city_fee * num_tires : ℝ)
  (total_cost + total_tax + total_city_fee - discount)

def car_A_tire_cost : ℝ := 7
def car_A_num_tires : ℕ := 3
def car_A_tax : ℝ := 0.5
def car_A_city_fee : ℝ := 2.5
def car_A_discount : ℝ := (car_A_tire_cost * car_A_num_tires) * 0.05

def car_B_tire_cost : ℝ := 8.5
def car_B_num_tires : ℕ := 2
def car_B_tax : ℝ := 0 -- no sales tax
def car_B_city_fee : ℝ := 2.5
def car_B_discount : ℝ := 0 -- expired coupon

theorem total_cost_of_repair : 
  cost_of_repair car_A_tire_cost car_A_num_tires car_A_tax car_A_city_fee car_A_discount + 
  cost_of_repair car_B_tire_cost car_B_num_tires car_B_tax car_B_city_fee car_B_discount = 50.95 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_repair_l369_36925


namespace NUMINAMATH_GPT_problem_solution_l369_36937

theorem problem_solution :
  (19 * 19 - 12 * 12) / ((19 / 12) - (12 / 19)) = 228 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l369_36937


namespace NUMINAMATH_GPT_function_inequality_l369_36935

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_inequality (f : ℝ → ℝ) (h1 : ∀ x : ℝ, x ≥ 1 → f x ≤ x)
  (h2 : ∀ x : ℝ, x ≥ 1 → f (2 * x) / Real.sqrt 2 ≤ f x) :
  ∀ x ≥ 1, f x < Real.sqrt (2 * x) :=
sorry

end NUMINAMATH_GPT_function_inequality_l369_36935


namespace NUMINAMATH_GPT_unique_sequence_count_l369_36970

def is_valid_sequence (a : Fin 5 → ℕ) :=
  a 0 = 1 ∧
  a 1 > a 0 ∧
  a 2 > a 1 ∧
  a 3 > a 2 ∧
  a 4 = 15 ∧
  (a 1) ^ 2 ≤ a 0 * a 2 + 1 ∧
  (a 2) ^ 2 ≤ a 1 * a 3 + 1 ∧
  (a 3) ^ 2 ≤ a 2 * a 4 + 1

theorem unique_sequence_count : 
  ∃! (a : Fin 5 → ℕ), is_valid_sequence a :=
sorry

end NUMINAMATH_GPT_unique_sequence_count_l369_36970


namespace NUMINAMATH_GPT_interval_for_x_l369_36939

theorem interval_for_x (x : ℝ) 
  (hx1 : 1/x < 2) 
  (hx2 : 1/x > -3) : 
  x > 1/2 ∨ x < -1/3 :=
  sorry

end NUMINAMATH_GPT_interval_for_x_l369_36939


namespace NUMINAMATH_GPT_b_minus_d_sq_value_l369_36960

theorem b_minus_d_sq_value 
  (a b c d : ℝ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3)
  (h3 : 2 * a - 3 * b + c + 4 * d = 17) :
  (b - d) ^ 2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_b_minus_d_sq_value_l369_36960


namespace NUMINAMATH_GPT_sum_of_integers_ending_in_2_between_100_and_600_l369_36948

theorem sum_of_integers_ending_in_2_between_100_and_600 :
  let a := 102
  let d := 10
  let l := 592
  let n := (l - a) / d + 1
  ∃ S : ℤ, S = n * (a + l) / 2 ∧ S = 17350 := 
by
  let a := 102
  let d := 10
  let l := 592
  let n := (l - a) / d + 1
  use n * (a + l) / 2
  sorry

end NUMINAMATH_GPT_sum_of_integers_ending_in_2_between_100_and_600_l369_36948


namespace NUMINAMATH_GPT_Bryan_did_258_pushups_l369_36934

-- Define the conditions
def sets : ℕ := 15
def pushups_per_set : ℕ := 18
def pushups_fewer_last_set : ℕ := 12

-- Define the planned total push-ups
def planned_total_pushups : ℕ := sets * pushups_per_set

-- Define the actual push-ups in the last set
def last_set_pushups : ℕ := pushups_per_set - pushups_fewer_last_set

-- Define the total push-ups Bryan did
def total_pushups : ℕ := (sets - 1) * pushups_per_set + last_set_pushups

-- The theorem to prove
theorem Bryan_did_258_pushups :
  total_pushups = 258 := by
  sorry

end NUMINAMATH_GPT_Bryan_did_258_pushups_l369_36934


namespace NUMINAMATH_GPT_jennifer_initial_pears_l369_36903

def initialPears (P: ℕ) : Prop := (P + 20 + 2 * P - 6 = 44)

theorem jennifer_initial_pears (P: ℕ) (h : initialPears P) : P = 10 := by
  sorry

end NUMINAMATH_GPT_jennifer_initial_pears_l369_36903


namespace NUMINAMATH_GPT_smallest_positive_integer_l369_36952

theorem smallest_positive_integer (n : ℕ) (h1 : 0 < n) (h2 : ∃ k1 : ℕ, 3 * n = k1^2) (h3 : ∃ k2 : ℕ, 4 * n = k2^3) : 
  n = 54 := 
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l369_36952


namespace NUMINAMATH_GPT_a679b_multiple_of_72_l369_36983

-- Define conditions
def is_divisible_by_8 (n : Nat) : Prop :=
  n % 8 = 0

def sum_of_digits_is_divisible_by_9 (n : Nat) : Prop :=
  (n.digits 10).sum % 9 = 0

-- Define the given problem
theorem a679b_multiple_of_72 (a b : Nat) : 
  is_divisible_by_8 (7 * 100 + 9 * 10 + b) →
  sum_of_digits_is_divisible_by_9 (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) → 
  a = 3 ∧ b = 2 :=
by 
  sorry

end NUMINAMATH_GPT_a679b_multiple_of_72_l369_36983


namespace NUMINAMATH_GPT_div_by_10_3pow_l369_36982

theorem div_by_10_3pow
    (m : ℤ)
    (n : ℕ)
    (h : (3^n + m) % 10 = 0) :
    (3^(n + 4) + m) % 10 = 0 := by
  sorry

end NUMINAMATH_GPT_div_by_10_3pow_l369_36982


namespace NUMINAMATH_GPT_proof_f_3_eq_9_ln_3_l369_36964

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x

theorem proof_f_3_eq_9_ln_3 (a : ℝ) (h : deriv (deriv (f a)) 1 = 3) : f a 3 = 9 * Real.log 3 :=
by
  sorry

end NUMINAMATH_GPT_proof_f_3_eq_9_ln_3_l369_36964


namespace NUMINAMATH_GPT_polynomial_identity_l369_36936

theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 = 
  (a - b) * (b - c) * (c - a) * (a + b + c) :=
sorry

end NUMINAMATH_GPT_polynomial_identity_l369_36936


namespace NUMINAMATH_GPT_one_eighth_of_2_pow_33_eq_2_pow_x_l369_36928

theorem one_eighth_of_2_pow_33_eq_2_pow_x (x : ℕ) : (1 / 8) * (2 : ℝ) ^ 33 = (2 : ℝ) ^ x → x = 30 := by
  intro h
  sorry

end NUMINAMATH_GPT_one_eighth_of_2_pow_33_eq_2_pow_x_l369_36928


namespace NUMINAMATH_GPT_correct_calculation_l369_36958

theorem correct_calculation :
  (-7 * a * b^2 + 4 * a * b^2 = -3 * a * b^2) ∧
  ¬ (2 * x + 3 * y = 5 * x * y) ∧
  ¬ (6 * x^2 - (-x^2) = 5 * x^2) ∧
  ¬ (4 * m * n - 3 * m * n = 1) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l369_36958


namespace NUMINAMATH_GPT_power_of_power_evaluation_l369_36915

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end NUMINAMATH_GPT_power_of_power_evaluation_l369_36915


namespace NUMINAMATH_GPT_find_a_l369_36902

theorem find_a
  (x y a : ℝ)
  (h1 : x + y = 1)
  (h2 : 2 * x + y = 0)
  (h3 : a * x - 3 * y = 0) :
  a = -6 :=
sorry

end NUMINAMATH_GPT_find_a_l369_36902


namespace NUMINAMATH_GPT_complex_multiplication_value_l369_36980

theorem complex_multiplication_value (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_value_l369_36980


namespace NUMINAMATH_GPT_total_distance_dog_runs_l369_36971

-- Define the distance between Xiaoqiang's home and his grandmother's house in meters
def distance_home_to_grandma : ℕ := 1000

-- Define Xiaoqiang's walking speed in meters per minute
def xiaoqiang_speed : ℕ := 50

-- Define the dog's running speed in meters per minute
def dog_speed : ℕ := 200

-- Define the time Xiaoqiang takes to reach his grandmother's house
def xiaoqiang_time (d : ℕ) (s : ℕ) : ℕ := d / s

-- State the total distance the dog runs given the speeds and distances
theorem total_distance_dog_runs (d x_speed dog_speed : ℕ) 
  (hx : x_speed > 0) (hd : dog_speed > 0) : (d / x_speed) * dog_speed = 4000 :=
  sorry

end NUMINAMATH_GPT_total_distance_dog_runs_l369_36971


namespace NUMINAMATH_GPT_benjie_is_6_years_old_l369_36943

-- Definitions based on conditions
def margo_age_in_3_years := 4
def years_until_then := 3
def age_difference := 5

-- Current age of Margo
def margo_current_age := margo_age_in_3_years - years_until_then

-- Current age of Benjie
def benjie_current_age := margo_current_age + age_difference

-- The theorem we need to prove
theorem benjie_is_6_years_old : benjie_current_age = 6 :=
by
  -- Proof
  sorry

end NUMINAMATH_GPT_benjie_is_6_years_old_l369_36943


namespace NUMINAMATH_GPT_quadratic_solution_m_l369_36917

theorem quadratic_solution_m (m : ℝ) : (x = 2) → (x^2 - m*x + 8 = 0) → (m = 6) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_m_l369_36917


namespace NUMINAMATH_GPT_min_rectangles_to_cover_square_exactly_l369_36916

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end NUMINAMATH_GPT_min_rectangles_to_cover_square_exactly_l369_36916


namespace NUMINAMATH_GPT_larger_value_algebraic_expression_is_2_l369_36972

noncomputable def algebraic_expression (a b c d x : ℝ) : ℝ :=
  x^2 + a + b + c * d * x

theorem larger_value_algebraic_expression_is_2
  (a b c d : ℝ) (x : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1)
  (h3 : x = 1 ∨ x = -1) :
  max (algebraic_expression a b c d 1) (algebraic_expression a b c d (-1)) = 2 :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_larger_value_algebraic_expression_is_2_l369_36972


namespace NUMINAMATH_GPT_area_of_rhombus_l369_36962

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 10) : 
  1 / 2 * d1 * d2 = 30 :=
by 
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_area_of_rhombus_l369_36962


namespace NUMINAMATH_GPT_exam_results_l369_36912

variable (E F G H : Prop)

def emma_statement : Prop := E → F
def frank_statement : Prop := F → ¬G
def george_statement : Prop := G → H
def exactly_two_asing : Prop :=
  (E ∧ F ∧ ¬G ∧ ¬H) ∨ (¬E ∧ F ∧ G ∧ ¬H) ∨
  (¬E ∧ ¬F ∧ G ∧ H) ∨ (¬E ∧ F ∧ ¬G ∧ H) ∨
  (E ∧ ¬F ∧ ¬G ∧ H)

theorem exam_results :
  (E ∧ F) ∨ (G ∧ H) :=
by {
  sorry
}

end NUMINAMATH_GPT_exam_results_l369_36912


namespace NUMINAMATH_GPT_smallest_value_of_3a_plus_1_l369_36955

theorem smallest_value_of_3a_plus_1 (a : ℚ) (h : 8 * a^2 + 6 * a + 2 = 2) : 3 * a + 1 = -5/4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_3a_plus_1_l369_36955


namespace NUMINAMATH_GPT_vanessa_savings_weeks_l369_36946

-- Definitions of given conditions
def dress_cost : ℕ := 80
def vanessa_savings : ℕ := 20
def weekly_allowance : ℕ := 30
def weekly_spending : ℕ := 10

-- Required amount to save 
def required_savings : ℕ := dress_cost - vanessa_savings

-- Weekly savings calculation
def weekly_savings : ℕ := weekly_allowance - weekly_spending

-- Number of weeks needed to save the required amount
def weeks_needed_to_save (required_savings weekly_savings : ℕ) : ℕ :=
  required_savings / weekly_savings

-- Axiom representing the correctness of our calculation
theorem vanessa_savings_weeks : weeks_needed_to_save required_savings weekly_savings = 3 := 
  by
  sorry

end NUMINAMATH_GPT_vanessa_savings_weeks_l369_36946


namespace NUMINAMATH_GPT_population_increase_l369_36984

theorem population_increase (P : ℝ) (h₁ : 11000 * (1 + P / 100) * (1 + P / 100) = 13310) : 
  P = 10 :=
sorry

end NUMINAMATH_GPT_population_increase_l369_36984


namespace NUMINAMATH_GPT_betty_boxes_l369_36909

theorem betty_boxes (total_oranges boxes_capacity : ℕ) (h1 : total_oranges = 24) (h2 : boxes_capacity = 8) : total_oranges / boxes_capacity = 3 :=
by sorry

end NUMINAMATH_GPT_betty_boxes_l369_36909


namespace NUMINAMATH_GPT_inscribed_sphere_volume_l369_36956

theorem inscribed_sphere_volume
  (a : ℝ)
  (h_cube_surface_area : 6 * a^2 = 24) :
  (4 / 3) * Real.pi * (a / 2)^3 = (4 / 3) * Real.pi :=
by
  -- sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_inscribed_sphere_volume_l369_36956


namespace NUMINAMATH_GPT_solve_for_g2_l369_36918

-- Let g : ℝ → ℝ be a function satisfying the given condition
variable (g : ℝ → ℝ)

-- The given condition
def condition (x : ℝ) : Prop :=
  g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The main theorem we aim to prove
theorem solve_for_g2 (h : ∀ x, condition g x) : g 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_g2_l369_36918


namespace NUMINAMATH_GPT_greatest_divisor_under_100_l369_36914

theorem greatest_divisor_under_100 (d : ℕ) :
  d ∣ 780 ∧ d < 100 ∧ d ∣ 180 ∧ d ∣ 240 ↔ d ≤ 60 := by
  sorry

end NUMINAMATH_GPT_greatest_divisor_under_100_l369_36914


namespace NUMINAMATH_GPT_distinct_negative_real_roots_l369_36994

def poly (p : ℝ) (x : ℝ) : ℝ := x^4 + 2*p*x^3 + x^2 + 2*p*x + 1

theorem distinct_negative_real_roots (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ poly p x1 = 0 ∧ poly p x2 = 0) ↔ p > 3/4 :=
sorry

end NUMINAMATH_GPT_distinct_negative_real_roots_l369_36994


namespace NUMINAMATH_GPT_candle_height_problem_l369_36954

-- Define the conditions given in the problem
def same_initial_height (height : ℝ := 1) := height = 1

def burn_rate_first_candle := 1 / 5

def burn_rate_second_candle := 1 / 4

def height_first_candle (t : ℝ) := 1 - (burn_rate_first_candle * t)

def height_second_candle (t : ℝ) := 1 - (burn_rate_second_candle * t)

-- Define the proof problem
theorem candle_height_problem : ∃ t : ℝ, height_first_candle t = 3 * height_second_candle t ∧ t = 40 / 11 :=
by
  sorry

end NUMINAMATH_GPT_candle_height_problem_l369_36954


namespace NUMINAMATH_GPT_sum_of_plane_angles_l369_36967

theorem sum_of_plane_angles (v f p : ℕ) (h : v = p) :
    (2 * π * (v - f) = 2 * π * (p - 2)) :=
by sorry

end NUMINAMATH_GPT_sum_of_plane_angles_l369_36967


namespace NUMINAMATH_GPT_x2004_y2004_l369_36986

theorem x2004_y2004 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) : 
  x^2004 + y^2004 = 2^2004 := 
by
  sorry

end NUMINAMATH_GPT_x2004_y2004_l369_36986


namespace NUMINAMATH_GPT_line_tangent_to_circle_l369_36998

theorem line_tangent_to_circle {m : ℝ} : 
  (∀ x y : ℝ, y = m * x) → (∀ x y : ℝ, x^2 + y^2 - 4 * x + 2 = 0) → 
  (m = 1 ∨ m = -1) := 
by 
  sorry

end NUMINAMATH_GPT_line_tangent_to_circle_l369_36998


namespace NUMINAMATH_GPT_hyperbola_equation_l369_36975

-- Definition of the ellipse given in the problem
def ellipse (x y : ℝ) := y^2 / 5 + x^2 = 1

-- Definition of the conditions for the hyperbola:
-- 1. The hyperbola shares a common focus with the ellipse.
-- 2. Distance from the focus to the asymptote of the hyperbola is 1.
def hyperbola (x y : ℝ) (c : ℝ) :=
  ∃ a b : ℝ, c = 2 ∧ a^2 + b^2 = c^2 ∧
             (b = 1 ∧ y = if x = 0 then 0 else x * (a / b))

-- The statement we need to prove
theorem hyperbola_equation : 
  (∃ a b : ℝ, ellipse x y ∧ hyperbola x y 2 ∧ b = 1 ∧ a^2 = 3) → 
  (y^2 / 3 - x^2 = 1) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l369_36975


namespace NUMINAMATH_GPT_total_heads_l369_36900

variables (H C : ℕ)

theorem total_heads (h_hens: H = 22) (h_feet: 2 * H + 4 * C = 140) : H + C = 46 :=
by
  sorry

end NUMINAMATH_GPT_total_heads_l369_36900


namespace NUMINAMATH_GPT_train_cross_pole_time_l369_36966

def speed_kmh := 90 -- speed of the train in km/hr
def length_m := 375 -- length of the train in meters

/-- Convert speed from km/hr to m/s -/
def convert_speed (v_kmh : ℕ) : ℕ := v_kmh * 1000 / 3600

/-- Calculate the time it takes for the train to cross the pole -/
def time_to_cross_pole (length_m : ℕ) (speed_m_s : ℕ) : ℕ := length_m / speed_m_s

theorem train_cross_pole_time :
  time_to_cross_pole length_m (convert_speed speed_kmh) = 15 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_pole_time_l369_36966


namespace NUMINAMATH_GPT_jindra_gray_fields_counts_l369_36907

-- Definitions for the problem setup
noncomputable def initial_gray_fields: ℕ := 7
noncomputable def rotation_90_gray_fields: ℕ := 8
noncomputable def rotation_180_gray_fields: ℕ := 4

-- Statement of the theorem to be proved
theorem jindra_gray_fields_counts:
  initial_gray_fields = 7 ∧
  rotation_90_gray_fields = 8 ∧
  rotation_180_gray_fields = 4 := by
  sorry

end NUMINAMATH_GPT_jindra_gray_fields_counts_l369_36907


namespace NUMINAMATH_GPT_base10_to_base7_l369_36932

theorem base10_to_base7 : 
  ∃ a b c d : ℕ, a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 = 729 ∧ a = 2 ∧ b = 0 ∧ c = 6 ∧ d = 1 :=
sorry

end NUMINAMATH_GPT_base10_to_base7_l369_36932


namespace NUMINAMATH_GPT_problem1_problem2_l369_36931

-- Problem (1)
theorem problem1 (a : ℝ) (h : a = 1) (p q : ℝ → Prop) 
  (hp : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0) 
  (hq : ∀ x, q x ↔ (x - 3)^2 < 1) :
  (∀ x, (p x ∧ q x) ↔ (2 < x ∧ x < 3)) :=
by sorry

-- Problem (2)
theorem problem2 (a : ℝ) (p q : ℝ → Prop)
  (hp : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0)
  (hq : ∀ x, q x ↔ (x - 3)^2 < 1)
  (hnpc : ∀ x, ¬p x → ¬q x) 
  (hnpc_not_necessary : ∃ x, ¬p x ∧ q x) :
  (4 / 3 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l369_36931


namespace NUMINAMATH_GPT_simplify_expression_l369_36930

theorem simplify_expression (a b : ℝ) (h : a + b < 0) : 
  |a + b - 1| - |3 - (a + b)| = -2 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l369_36930


namespace NUMINAMATH_GPT_min_value_expression_l369_36901

theorem min_value_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 8) : 
  (x + 3 * y) * (y + 3 * z) * (3 * x * z + 1) ≥ 72 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l369_36901


namespace NUMINAMATH_GPT_min_vertical_distance_between_graphs_l369_36996

noncomputable def min_distance (x : ℝ) : ℝ :=
  |x| - (-x^2 - 4 * x - 2)

theorem min_vertical_distance_between_graphs :
  ∃ x : ℝ, ∀ y : ℝ, min_distance x ≤ min_distance y := 
    sorry

end NUMINAMATH_GPT_min_vertical_distance_between_graphs_l369_36996


namespace NUMINAMATH_GPT_sum_of_coeffs_l369_36995

theorem sum_of_coeffs (a_5 a_4 a_3 a_2 a_1 a : ℤ) (h_eq : (x - 2)^5 = a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) (h_a : a = -32) :
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coeffs_l369_36995


namespace NUMINAMATH_GPT_negation_all_nonzero_l369_36923

    theorem negation_all_nonzero (a b c : ℝ) : ¬ (¬ (a = 0 ∨ b = 0 ∨ c = 0)) → (a = 0 ∧ b = 0 ∧ c = 0) :=
    by
      sorry
    
end NUMINAMATH_GPT_negation_all_nonzero_l369_36923


namespace NUMINAMATH_GPT_range_of_x_l369_36990

theorem range_of_x (x : ℝ) (h : 2 * x + 1 ≤ 0) : x ≤ -1 / 2 := 
  sorry

end NUMINAMATH_GPT_range_of_x_l369_36990


namespace NUMINAMATH_GPT_eldest_sibling_age_correct_l369_36944

-- Definitions and conditions
def youngest_sibling_age (x : ℝ) := x
def second_youngest_sibling_age (x : ℝ) := x + 4
def third_youngest_sibling_age (x : ℝ) := x + 8
def fourth_youngest_sibling_age (x : ℝ) := x + 12
def fifth_youngest_sibling_age (x : ℝ) := x + 16
def sixth_youngest_sibling_age (x : ℝ) := x + 20
def seventh_youngest_sibling_age (x : ℝ) := x + 28
def eldest_sibling_age (x : ℝ) := x + 32

def combined_age_of_eight_siblings (x : ℝ) : ℝ := 
  youngest_sibling_age x +
  second_youngest_sibling_age x +
  third_youngest_sibling_age x +
  fourth_youngest_sibling_age x +
  fifth_youngest_sibling_age x +
  sixth_youngest_sibling_age x +
  seventh_youngest_sibling_age x +
  eldest_sibling_age x

-- Proving the combined age part
theorem eldest_sibling_age_correct (x : ℝ) (h : combined_age_of_eight_siblings x - youngest_sibling_age (x + 24) = 140) : 
  eldest_sibling_age x = 34.5 := by
  sorry

end NUMINAMATH_GPT_eldest_sibling_age_correct_l369_36944


namespace NUMINAMATH_GPT_gaussian_guardians_points_l369_36968

theorem gaussian_guardians_points :
  let Daniel := 7
  let Curtis := 8
  let Sid := 2
  let Emily := 11
  let Kalyn := 6
  let Hyojeong := 12
  let Ty := 1
  let Winston := 7
  Daniel + Curtis + Sid + Emily + Kalyn + Hyojeong + Ty + Winston = 54 :=
by
  sorry

end NUMINAMATH_GPT_gaussian_guardians_points_l369_36968


namespace NUMINAMATH_GPT_binary_to_base4_conversion_l369_36961

theorem binary_to_base4_conversion : 
  let binary := (1*2^7 + 1*2^6 + 0*2^5 + 1*2^4 + 1*2^3 + 0*2^2 + 0*2^1 + 1*2^0) 
  let base4 := (3*4^3 + 1*4^2 + 2*4^1 + 1*4^0)
  binary = base4 := by
  sorry

end NUMINAMATH_GPT_binary_to_base4_conversion_l369_36961


namespace NUMINAMATH_GPT_sum_of_squares_gt_five_l369_36908

theorem sum_of_squares_gt_five (a b c : ℝ) (h : a + b + c = 4) : a^2 + b^2 + c^2 > 5 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_gt_five_l369_36908


namespace NUMINAMATH_GPT_quadratic_radical_condition_l369_36978

variable (x : ℝ)

theorem quadratic_radical_condition : 
  (∃ (r : ℝ), r = x^2 + 1 ∧ r ≥ 0) ↔ (True) := by
  sorry

end NUMINAMATH_GPT_quadratic_radical_condition_l369_36978


namespace NUMINAMATH_GPT_cube_edge_percentage_growth_l369_36913

theorem cube_edge_percentage_growth (p : ℝ) 
  (h : (1 + p / 100) ^ 2 - 1 = 0.96) : p = 40 :=
by
  sorry

end NUMINAMATH_GPT_cube_edge_percentage_growth_l369_36913


namespace NUMINAMATH_GPT_circumference_of_circle_inscribing_rectangle_l369_36976

theorem circumference_of_circle_inscribing_rectangle (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi := by
  sorry

end NUMINAMATH_GPT_circumference_of_circle_inscribing_rectangle_l369_36976


namespace NUMINAMATH_GPT_solve_system_of_equations_l369_36985

theorem solve_system_of_equations :
  ∃ (x y z w : ℤ), 
    x - y + z - w = 2 ∧
    x^2 - y^2 + z^2 - w^2 = 6 ∧
    x^3 - y^3 + z^3 - w^3 = 20 ∧
    x^4 - y^4 + z^4 - w^4 = 66 ∧
    (x, y, z, w) = (1, 3, 0, 2) := 
  by
    sorry

end NUMINAMATH_GPT_solve_system_of_equations_l369_36985


namespace NUMINAMATH_GPT_find_f_neg2_l369_36987

-- Define the function f and the given conditions
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x - 4

theorem find_f_neg2 (a b : ℝ) (h₁ : f 2 a b = 6) : f (-2) a b = -14 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg2_l369_36987


namespace NUMINAMATH_GPT_no_infinite_monochromatic_arithmetic_progression_l369_36926

theorem no_infinite_monochromatic_arithmetic_progression : 
  ∃ (coloring : ℕ → ℕ), (∀ (q r : ℕ), ∃ (n1 n2 : ℕ), coloring (q * n1 + r) ≠ coloring (q * n2 + r)) := sorry

end NUMINAMATH_GPT_no_infinite_monochromatic_arithmetic_progression_l369_36926


namespace NUMINAMATH_GPT_specialCollectionAtEndOfMonth_l369_36949

noncomputable def specialCollectionBooksEndOfMonth (initialBooks loanedBooks returnedPercentage : ℕ) :=
  initialBooks - (loanedBooks - loanedBooks * returnedPercentage / 100)

theorem specialCollectionAtEndOfMonth :
  specialCollectionBooksEndOfMonth 150 80 65 = 122 :=
by
  sorry

end NUMINAMATH_GPT_specialCollectionAtEndOfMonth_l369_36949


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l369_36911

theorem simplify_and_evaluate_expr (x : ℤ) (h : x = -2) : 
  (2 * x + 1) * (x - 2) - (2 - x) ^ 2 = -8 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l369_36911


namespace NUMINAMATH_GPT_intersection_M_N_l369_36951

def M : Set ℝ := {y | ∃ x : ℝ, y = x - |x|}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {0} :=
  sorry

end NUMINAMATH_GPT_intersection_M_N_l369_36951
