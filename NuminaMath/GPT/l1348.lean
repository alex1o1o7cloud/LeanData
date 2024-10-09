import Mathlib

namespace div_by_seven_equiv_l1348_134848

-- Given integers a and b, prove that 10a + b is divisible by 7 if and only if a - 2b is divisible by 7.
theorem div_by_seven_equiv (a b : ℤ) : (10 * a + b) % 7 = 0 ↔ (a - 2 * b) % 7 = 0 := sorry

end div_by_seven_equiv_l1348_134848


namespace determine_x_l1348_134846

theorem determine_x (x : ℝ) (hx : 0 < x) (h : (⌊x⌋ : ℝ) * x = 120) : x = 120 / 11 := 
sorry

end determine_x_l1348_134846


namespace equal_five_digit_number_sets_l1348_134873

def five_digit_numbers_not_div_5 : ℕ :=
  9 * 10^3 * 8

def five_digit_numbers_first_two_not_5 : ℕ :=
  8 * 9 * 10^3

theorem equal_five_digit_number_sets :
  five_digit_numbers_not_div_5 = five_digit_numbers_first_two_not_5 :=
by
  repeat { sorry }

end equal_five_digit_number_sets_l1348_134873


namespace arithmetic_sequence_product_l1348_134864

theorem arithmetic_sequence_product {b : ℕ → ℤ} (d : ℤ) (h1 : ∀ n, b (n + 1) = b n + d)
    (h2 : b 5 * b 6 = 21) : b 4 * b 7 = -11 :=
  sorry

end arithmetic_sequence_product_l1348_134864


namespace ratio_of_60_to_12_l1348_134866

theorem ratio_of_60_to_12 : 60 / 12 = 5 := 
by 
  sorry

end ratio_of_60_to_12_l1348_134866


namespace largest_unreachable_integer_l1348_134849

theorem largest_unreachable_integer : ∃ n : ℕ, (¬ ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 8 * a + 11 * b = n)
  ∧ ∀ m : ℕ, m > n → (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 8 * a + 11 * b = m) := sorry

end largest_unreachable_integer_l1348_134849


namespace erica_has_correct_amount_l1348_134884

-- Definitions for conditions
def total_money : ℕ := 91
def sam_money : ℕ := 38

-- Definition for the question regarding Erica's money
def erica_money := total_money - sam_money

-- The theorem stating the proof problem
theorem erica_has_correct_amount : erica_money = 53 := sorry

end erica_has_correct_amount_l1348_134884


namespace missing_fraction_is_correct_l1348_134809

theorem missing_fraction_is_correct :
  (1 / 3 + 1 / 2 + -5 / 6 + 1 / 5 + -9 / 20 + -9 / 20) = 0.45 - (23 / 20) :=
by
  sorry

end missing_fraction_is_correct_l1348_134809


namespace max_xy_l1348_134814

theorem max_xy : 
  ∃ x y : ℕ, 5 * x + 3 * y = 100 ∧ x > 0 ∧ y > 0 ∧ x * y = 165 :=
by
  sorry

end max_xy_l1348_134814


namespace number_of_baskets_l1348_134829

-- Define the conditions
def total_peaches : Nat := 10
def red_peaches_per_basket : Nat := 4
def green_peaches_per_basket : Nat := 6
def peaches_per_basket : Nat := red_peaches_per_basket + green_peaches_per_basket

-- The goal is to prove that the number of baskets is 1 given the conditions

theorem number_of_baskets (h1 : total_peaches = 10)
                           (h2 : peaches_per_basket = red_peaches_per_basket + green_peaches_per_basket)
                           (h3 : red_peaches_per_basket = 4)
                           (h4 : green_peaches_per_basket = 6) : 
                           total_peaches / peaches_per_basket = 1 := by
                            sorry

end number_of_baskets_l1348_134829


namespace problem1_problem2_l1348_134810

variable {a b x : ℝ}

theorem problem1 (h₀ : a ≠ b) (h₁ : a ≠ -b) :
  (a / (a - b)) - (b / (a + b)) = (a^2 + b^2) / (a^2 - b^2) :=
sorry

theorem problem2 (h₀ : x ≠ 2) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  ((x - 2) / (x - 1)) / ((x^2 - 4 * x + 4) / (x^2 - 1)) + ((1 - x) / (x - 2)) = 2 / (x - 2) :=
sorry

end problem1_problem2_l1348_134810


namespace caesars_rental_fee_l1348_134824

theorem caesars_rental_fee (C : ℕ) 
  (hc : ∀ (n : ℕ), n = 60 → C + 30 * n = 500 + 35 * n) : 
  C = 800 :=
by
  sorry

end caesars_rental_fee_l1348_134824


namespace price_per_can_of_spam_l1348_134867

-- Definitions of conditions
variable (S : ℝ) -- The price per can of Spam
def cost_peanut_butter := 3 * 5 -- 3 jars of peanut butter at $5 each
def cost_bread := 4 * 2 -- 4 loaves of bread at $2 each
def total_cost := 59 -- Total amount paid

-- Proof problem to verify the price per can of Spam
theorem price_per_can_of_spam :
  12 * S + cost_peanut_butter + cost_bread = total_cost → S = 3 :=
by
  sorry

end price_per_can_of_spam_l1348_134867


namespace infinitely_many_a_not_prime_l1348_134801

theorem infinitely_many_a_not_prime (a: ℤ) (n: ℤ) : ∃ (b: ℤ), b ≥ 0 ∧ (∃ (N: ℕ) (a: ℤ), a = 4*(N:ℤ)^4 ∧ ∀ (n: ℤ), ¬Prime (n^4 + a)) :=
by { sorry }

end infinitely_many_a_not_prime_l1348_134801


namespace points_on_ellipse_l1348_134845

noncomputable def x (t : ℝ) : ℝ := (3 - t^2) / (1 + t^2)
noncomputable def y (t : ℝ) : ℝ := 4 * t / (1 + t^2)

theorem points_on_ellipse : ∀ t : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  (x t / a)^2 + (y t / b)^2 = 1 := 
sorry

end points_on_ellipse_l1348_134845


namespace multiple_7_proposition_l1348_134859

theorem multiple_7_proposition : (47 % 7 ≠ 0 ∨ 49 % 7 = 0) → True :=
by
  intros h
  sorry

end multiple_7_proposition_l1348_134859


namespace monotone_decreasing_f_find_a_value_l1348_134895

-- Condition declarations
variables (a b : ℝ) (h_a_pos : a > 0) (max_val min_val : ℝ)
noncomputable def f (x : ℝ) := x + (a / x) + b

-- Problem 1: Prove that f is monotonically decreasing in (0, sqrt(a)]
theorem monotone_decreasing_f : 
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 ≤ Real.sqrt a → f a b x1 > f a b x2) :=
sorry

-- Conditions for Problem 2
variable (hf_inc : ∀ x1 x2 : ℝ, Real.sqrt a ≤ x1 ∧ x1 < x2 → f a b x1 < f a b x2)
variable (h_max : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f a b x ≤ 5)
variable (h_min : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f a b x ≥ 3)

-- Problem 2: Find the value of a
theorem find_a_value : a = 6 :=
sorry

end monotone_decreasing_f_find_a_value_l1348_134895


namespace remainder_sum_of_integers_division_l1348_134890

theorem remainder_sum_of_integers_division (n S : ℕ) (hn_cond : n > 0) (hn_sq : n^2 + 12 * n - 3007 ≥ 0) (hn_square : ∃ m : ℕ, n^2 + 12 * n - 3007 = m^2):
  S = n → S % 1000 = 516 := 
sorry

end remainder_sum_of_integers_division_l1348_134890


namespace find_k_l1348_134839

-- Define the conditions
def parabola (k : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + k

-- Theorem statement
theorem find_k (k : ℝ) : (∀ x : ℝ, parabola k x = 0 → x = -1) → k = 1 :=
by
  sorry

end find_k_l1348_134839


namespace cost_of_one_dozen_pens_is_780_l1348_134872

-- Defining the cost of pens and pencils
def cost_of_pens (n : ℕ) := n * 65

def cost_of_pencils (m : ℕ) := m * 13

-- Given conditions
def total_cost (x y : ℕ) := cost_of_pens x + cost_of_pencils y

theorem cost_of_one_dozen_pens_is_780
  (h1 : total_cost 3 5 = 260)
  (h2 : 65 = 5 * 13)
  (h3 : 65 = 65) :
  12 * 65 = 780 := by
    sorry

end cost_of_one_dozen_pens_is_780_l1348_134872


namespace velocity_of_point_C_l1348_134825

variable (a T R L x : ℝ)
variable (a_pos : a > 0) (T_pos : T > 0) (R_pos : R > 0) (L_pos : L > 0)
variable (h_eq : a * T / (a * T - R) = (L + x) / x)

theorem velocity_of_point_C : a * (L / R) = x / T := by
  sorry

end velocity_of_point_C_l1348_134825


namespace matrix_problem_l1348_134811

def A : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![20 / 3, 4 / 3],
  ![-8 / 3, 8 / 3]
]
def B : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![0, 0], -- Correct values for B can be computed from conditions if needed
  ![0, 0]
]

theorem matrix_problem (A B : Matrix (Fin 2) (Fin 2) ℚ)
  (h1 : A + B = A * B)
  (h2 : A * B = ![
  ![20 / 3, 4 / 3],
  ![-8 / 3, 8 / 3]
]) :
  B * A = ![
    ![20 / 3, 4 / 3],
    ![-8 / 3, 8 / 3]
  ] :=
sorry

end matrix_problem_l1348_134811


namespace max_gcd_2015xy_l1348_134897

theorem max_gcd_2015xy (x y : ℤ) (coprime : Int.gcd x y = 1) :
    ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
sorry

end max_gcd_2015xy_l1348_134897


namespace age_ratio_l1348_134843

theorem age_ratio (V A : ℕ) (h1 : V - 5 = 16) (h2 : V * 2 = 7 * A) :
  (V + 4) * 2 = (A + 4) * 5 := 
sorry

end age_ratio_l1348_134843


namespace tracy_candies_l1348_134830

variable (x : ℕ) -- number of candies Tracy started with

theorem tracy_candies (h1: x % 4 = 0)
                      (h2 : 46 ≤ x / 2 - 40 ∧ x / 2 - 40 ≤ 50) 
                      (h3 : ∃ k, 2 ≤ k ∧ k ≤ 6 ∧ x / 2 - 40 - k = 4) 
                      (h4 : ∃ n, x = 4 * n) : x = 96 :=
by
  sorry

end tracy_candies_l1348_134830


namespace find_a45_l1348_134833

theorem find_a45 :
  ∃ (a : ℕ → ℝ), 
    a 0 = 11 ∧ a 1 = 11 ∧ 
    (∀ m n : ℕ, a (m + n) = (1/2) * (a (2 * m) + a (2 * n)) - (m - n)^2) ∧ 
    a 45 = 1991 := by
  sorry

end find_a45_l1348_134833


namespace evaluate_g_at_3_l1348_134883

def g (x: ℝ) := 5 * x^3 - 4 * x^2 + 3 * x - 7

theorem evaluate_g_at_3 : g 3 = 101 :=
by 
  sorry

end evaluate_g_at_3_l1348_134883


namespace mod_pow_difference_l1348_134815

theorem mod_pow_difference (a b n : ℕ) (h1 : a ≡ 47 [MOD n]) (h2 : b ≡ 22 [MOD n]) (h3 : n = 8) : (a ^ 2023 - b ^ 2023) % n = 1 :=
by
  sorry

end mod_pow_difference_l1348_134815


namespace problem1_problem2_l1348_134853

-- Problem 1 equivalent proof problem
theorem problem1 : 
  (Real.sqrt 3 * Real.sqrt 6 - (Real.sqrt (1 / 2) - Real.sqrt 8)) = (9 * Real.sqrt 2 / 2) :=
by
  sorry

-- Problem 2 equivalent proof problem
theorem problem2 (x : Real) (hx : x = Real.sqrt 5) : 
  ((1 + 1 / x) / ((x^2 + x) / x)) = (Real.sqrt 5 / 5) :=
by
  sorry

end problem1_problem2_l1348_134853


namespace billy_apples_l1348_134818

def num_apples_eaten (monday_apples tuesday_apples wednesday_apples thursday_apples friday_apples total_apples : ℕ) : Prop :=
  monday_apples = 2 ∧
  tuesday_apples = 2 * monday_apples ∧
  wednesday_apples = 9 ∧
  friday_apples = monday_apples / 2 ∧
  thursday_apples = 4 * friday_apples ∧
  total_apples = monday_apples + tuesday_apples + wednesday_apples + thursday_apples + friday_apples

theorem billy_apples : num_apples_eaten 2 4 9 4 1 20 := 
by
  unfold num_apples_eaten
  sorry

end billy_apples_l1348_134818


namespace k_range_for_two_zeros_of_f_l1348_134813

noncomputable def f (x k : ℝ) : ℝ := x^2 - x * (Real.log x) - k * (x + 2) + 2

theorem k_range_for_two_zeros_of_f :
  ∀ k : ℝ, (∃ x1 x2 : ℝ, (1/2 < x1) ∧ (x1 < x2) ∧ f x1 k = 0 ∧ f x2 k = 0) ↔ 1 < k ∧ k ≤ (9 + 2 * Real.log 2) / 10 :=
by
  sorry

end k_range_for_two_zeros_of_f_l1348_134813


namespace train_more_passengers_l1348_134804

def one_train_car_capacity : ℕ := 60
def one_airplane_capacity : ℕ := 366
def number_of_train_cars : ℕ := 16
def number_of_airplanes : ℕ := 2

theorem train_more_passengers {one_train_car_capacity : ℕ} 
                               {one_airplane_capacity : ℕ} 
                               {number_of_train_cars : ℕ} 
                               {number_of_airplanes : ℕ} :
  (number_of_train_cars * one_train_car_capacity) - (number_of_airplanes * one_airplane_capacity) = 228 :=
by
  sorry

end train_more_passengers_l1348_134804


namespace smallest_number_to_end_in_four_zeros_l1348_134820

theorem smallest_number_to_end_in_four_zeros (x : ℕ) :
  let n1 := 225
  let n2 := 525
  let factor_needed := 16
  (∃ y : ℕ, y = n1 * n2 * x) ∧ (10^4 ∣ n1 * n2 * x) ↔ x = factor_needed :=
by
  sorry

end smallest_number_to_end_in_four_zeros_l1348_134820


namespace increasing_interval_l1348_134892

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval : {x : ℝ | 2 < x} = { x : ℝ | (x - 3) * Real.exp x > 0 } :=
by
  sorry

end increasing_interval_l1348_134892


namespace total_number_of_boys_in_camp_l1348_134828

theorem total_number_of_boys_in_camp (T : ℕ)
  (hA1 : ∃ (boysA : ℕ), boysA = 20 * T / 100)
  (hA2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 30 * boysA / 100 ∧ boysM = 40 * boysA / 100)
  (hB1 : ∃ (boysB : ℕ), boysB = 30 * T / 100)
  (hB2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 25 * boysB / 100 ∧ boysM = 35 * boysB / 100)
  (hC1 : ∃ (boysC : ℕ), boysC = 50 * T / 100)
  (hC2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 15 * boysC / 100 ∧ boysM = 45 * boysC / 100)
  (hA_no_SM : 77 = 70 * boysA / 100)
  (hB_no_SM : 72 = 60 * boysB / 100)
  (hC_no_SM : 98 = 60 * boysC / 100) :
  T = 535 :=
by
  sorry

end total_number_of_boys_in_camp_l1348_134828


namespace sqrt_4_of_10000000_eq_l1348_134885

noncomputable def sqrt_4_of_10000000 : Real := Real.sqrt (Real.sqrt 10000000)

theorem sqrt_4_of_10000000_eq :
  sqrt_4_of_10000000 = 10 * Real.sqrt (Real.sqrt 10) := by
sorry

end sqrt_4_of_10000000_eq_l1348_134885


namespace unique_2_digit_cyclic_permutation_divisible_l1348_134854

def is_cyclic_permutation (n : ℕ) (M : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → M i = M j

def M (a : Fin 2 → ℕ) : ℕ := a 0 * 10 + a 1

theorem unique_2_digit_cyclic_permutation_divisible (a : Fin 2 → ℕ) (h0 : ∀ i, a i ≠ 0) :
  (M a) % (a 1 * 10 + a 0) = 0 → 
  (M a = 11) :=
by
  sorry

end unique_2_digit_cyclic_permutation_divisible_l1348_134854


namespace new_volume_l1348_134808

theorem new_volume (l w h : ℝ) 
  (h1: l * w * h = 3000) 
  (h2: l * w + w * h + l * h = 690) 
  (h3: l + w + h = 40) : 
  (l + 2) * (w + 2) * (h + 2) = 4548 := 
  sorry

end new_volume_l1348_134808


namespace chocolates_brought_by_friend_l1348_134844

-- Definitions corresponding to the conditions in a)
def total_chocolates := 50
def chocolates_not_in_box := 5
def number_of_boxes := 3
def additional_boxes := 2

-- Theorem statement: we need to prove the number of chocolates her friend brought
theorem chocolates_brought_by_friend (C : ℕ) : 
  (C + total_chocolates = total_chocolates + (chocolates_not_in_box + number_of_boxes * (total_chocolates - chocolates_not_in_box) / number_of_boxes + additional_boxes * (total_chocolates - chocolates_not_in_box) / number_of_boxes) - total_chocolates) 
  → C = 30 := 
sorry

end chocolates_brought_by_friend_l1348_134844


namespace find_x_l1348_134894

theorem find_x (x : ℤ) (h : 3 * x = (26 - x) + 10) : x = 9 :=
by
  -- proof steps would be provided here
  sorry

end find_x_l1348_134894


namespace profit_percentage_is_40_l1348_134841

-- Define the given conditions
def total_cost : ℚ := 44 * 150 + 36 * 125  -- Rs 11100
def total_weight : ℚ := 44 + 36            -- 80 kg
def selling_price_per_kg : ℚ := 194.25     -- Rs 194.25
def total_selling_price : ℚ := total_weight * selling_price_per_kg  -- Rs 15540
def profit : ℚ := total_selling_price - total_cost  -- Rs 4440

-- Define the statement about the profit percentage
def profit_percentage : ℚ := (profit / total_cost) * 100

-- State the theorem
theorem profit_percentage_is_40 :
  profit_percentage = 40 := by
  -- This is where the proof would go
  sorry

end profit_percentage_is_40_l1348_134841


namespace jack_jogging_speed_needed_l1348_134858

noncomputable def jack_normal_speed : ℝ :=
  let normal_melt_time : ℝ := 10
  let faster_melt_factor : ℝ := 0.75
  let adjusted_melt_time : ℝ := normal_melt_time * faster_melt_factor
  let adjusted_melt_time_hours : ℝ := adjusted_melt_time / 60
  let distance_to_beach : ℝ := 2
  let required_speed : ℝ := distance_to_beach / adjusted_melt_time_hours
  let slope_reduction_factor : ℝ := 0.8
  required_speed / slope_reduction_factor

theorem jack_jogging_speed_needed
  (normal_melt_time : ℝ := 10) 
  (faster_melt_factor : ℝ := 0.75) 
  (distance_to_beach : ℝ := 2) 
  (slope_reduction_factor : ℝ := 0.8) :
  jack_normal_speed = 20 := 
by
  sorry

end jack_jogging_speed_needed_l1348_134858


namespace quadratic_root_shift_l1348_134802

theorem quadratic_root_shift (d e : ℝ) :
  (∀ r s : ℝ, (r^2 - 2 * r + 0.5 = 0) → (r-3)^2 + (r-3) * (s-3) * d + e = 0) → e = 3.5 := 
by
  intros
  sorry

end quadratic_root_shift_l1348_134802


namespace cos_half_pi_plus_double_alpha_l1348_134816

theorem cos_half_pi_plus_double_alpha (α : ℝ) (h : Real.tan α = 1 / 3) : 
  Real.cos (Real.pi / 2 + 2 * α) = -3 / 5 :=
by
  sorry

end cos_half_pi_plus_double_alpha_l1348_134816


namespace inequality_am_gm_l1348_134882

theorem inequality_am_gm (a b : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : 0 < b) (h₃ : b < 1) :
  1 + a + b > 3 * Real.sqrt (a * b) :=
by
  sorry

end inequality_am_gm_l1348_134882


namespace bobby_shoes_multiple_l1348_134861

theorem bobby_shoes_multiple (B M : ℕ) (hBonny : 13 = 2 * B - 5) (hBobby : 27 = M * B) : 
  M = 3 :=
by 
  sorry

end bobby_shoes_multiple_l1348_134861


namespace symmetric_line_eq_l1348_134851

-- Defining a structure for a line using its standard equation form "ax + by + c = 0"
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Definition: A line is symmetric with respect to y-axis if it can be obtained
-- by replacing x with -x in its equation form.

def isSymmetricToYAxis (l₁ l₂ : Line) : Prop :=
  l₂.a = -l₁.a ∧ l₂.b = l₁.b ∧ l₂.c = l₁.c

-- The given condition: line1 is 4x - 3y + 5 = 0
def line1 : Line := { a := 4, b := -3, c := 5 }

-- The expected line l symmetric to y-axis should satisfy our properties
def expected_line_l : Line := { a := 4, b := 3, c := -5 }

-- The theorem we need to prove
theorem symmetric_line_eq : ∃ l : Line,
  isSymmetricToYAxis line1 l ∧ l = { a := 4, b := 3, c := -5 } :=
by
  sorry

end symmetric_line_eq_l1348_134851


namespace decimal_to_base8_conversion_l1348_134881

-- Define the base and the number in decimal.
def base : ℕ := 8
def decimal_number : ℕ := 127

-- Define the expected representation in base 8.
def expected_base8_representation : ℕ := 177

-- Theorem stating that conversion of 127 in base 10 to base 8 yields 177
theorem decimal_to_base8_conversion : Nat.ofDigits base (Nat.digits base decimal_number) = expected_base8_representation := 
by
  sorry

end decimal_to_base8_conversion_l1348_134881


namespace find_triangle_area_l1348_134863

noncomputable def triangle_area_problem
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 14) : ℝ :=
  (1 / 2) * a * b

theorem find_triangle_area
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 14) :
  triangle_area_problem a b h1 h2 = 1 / 2 := by
  sorry

end find_triangle_area_l1348_134863


namespace largest_possible_a_l1348_134875

theorem largest_possible_a (a b c e : ℕ) (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 5 * e) (h4 : e < 100) : a ≤ 2961 :=
by
  sorry

end largest_possible_a_l1348_134875


namespace probability_of_yellow_jelly_bean_l1348_134838

theorem probability_of_yellow_jelly_bean (P_red P_orange P_green P_yellow : ℝ)
  (h_red : P_red = 0.1)
  (h_orange : P_orange = 0.4)
  (h_green : P_green = 0.25)
  (h_total : P_red + P_orange + P_green + P_yellow = 1) :
  P_yellow = 0.25 :=
by
  sorry

end probability_of_yellow_jelly_bean_l1348_134838


namespace cabbages_produced_l1348_134812

theorem cabbages_produced (x y : ℕ) (h1 : y = x + 1) (h2 : x^2 + 199 = y^2) : y^2 = 10000 :=
by
  sorry

end cabbages_produced_l1348_134812


namespace greatest_value_a_plus_b_l1348_134880

theorem greatest_value_a_plus_b (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a * b = 45) : a + b = 2 * Real.sqrt 55 :=
by
  sorry

end greatest_value_a_plus_b_l1348_134880


namespace dividend_rate_correct_l1348_134860

def stock_price : ℝ := 150
def yield_percentage : ℝ := 0.08
def dividend_rate : ℝ := stock_price * yield_percentage

theorem dividend_rate_correct : dividend_rate = 12 := by
  sorry

end dividend_rate_correct_l1348_134860


namespace point_coordinates_l1348_134842

namespace CoordinateProof

structure Point where
  x : ℝ
  y : ℝ

def isSecondQuadrant (P : Point) : Prop := P.x < 0 ∧ P.y > 0
def distToXAxis (P : Point) : ℝ := |P.y|
def distToYAxis (P : Point) : ℝ := |P.x|

theorem point_coordinates (P : Point) (h1 : isSecondQuadrant P) (h2 : distToXAxis P = 3) (h3 : distToYAxis P = 7) : P = ⟨-7, 3⟩ :=
by
  sorry

end CoordinateProof

end point_coordinates_l1348_134842


namespace initial_amount_is_53_l1348_134855

variable (X : ℕ) -- Initial amount of money Olivia had
variable (ATM_collect : ℕ := 91) -- Money collected from ATM
variable (supermarket_spent_diff : ℕ := 39) -- Spent 39 dollars more at the supermarket
variable (money_left : ℕ := 14) -- Money left after supermarket

-- Define the final amount Olivia had
def final_amount (X ATM_collect supermarket_spent_diff : ℕ) : ℕ :=
  X + ATM_collect - (ATM_collect + supermarket_spent_diff)

-- Theorem stating that the initial amount X was 53 dollars
theorem initial_amount_is_53 : final_amount X ATM_collect supermarket_spent_diff = money_left → X = 53 :=
by
  intros h
  sorry

end initial_amount_is_53_l1348_134855


namespace election_votes_l1348_134862

theorem election_votes (V : ℝ) (h1 : 0.56 * V - 0.44 * V = 288) : 0.56 * V = 1344 :=
by 
  sorry

end election_votes_l1348_134862


namespace simplify_and_evaluate_expression_l1348_134837

theorem simplify_and_evaluate_expression (a b : ℝ) (h₁ : a = 2 + Real.sqrt 3) (h₂ : b = 2 - Real.sqrt 3) :
  (a^2 - b^2) / a / (a - (2 * a * b - b^2) / a) = 2 * Real.sqrt 3 / 3 :=
by
  -- Proof to be provided
  sorry

end simplify_and_evaluate_expression_l1348_134837


namespace proof_GP_product_l1348_134870

namespace GPProof

variables {a r : ℝ} {n : ℕ} (S S' P : ℝ)

def isGeometricProgression (a r : ℝ) (n : ℕ) :=
  ∀ i, 0 ≤ i ∧ i < n → ∃ k, ∃ b, b = (-1)^k * a * r^k ∧ k = i 

noncomputable def product (a r : ℝ) (n : ℕ) : ℝ :=
  a^n * r^(n*(n-1)/2) * (-1)^(n*(n-1)/2)

noncomputable def sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - (-r)^n) / (1 - (-r))

noncomputable def reciprocalSum (a r : ℝ) (n : ℕ) : ℝ :=
  (1 / a) * (1 - (-1/r)^n) / (1 + 1/r)

theorem proof_GP_product (hyp1 : isGeometricProgression a (-r) n) (hyp2 : S = sum a (-r) n) (hyp3 : S' = reciprocalSum a (-r) n) (hyp4 : P = product a (-r) n) :
  P = (S / S')^(n/2) :=
by
  sorry

end GPProof

end proof_GP_product_l1348_134870


namespace solution_l1348_134817

noncomputable def polynomial_has_real_root (a : ℝ) : Prop :=
  ∃ x : ℝ, x^4 - a * x^2 + a * x - 1 = 0

theorem solution (a : ℝ) : polynomial_has_real_root a :=
sorry

end solution_l1348_134817


namespace triangle_area_ABC_l1348_134878

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 9)
def C : ℝ × ℝ := (7, 6)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Prove that the area of the triangle with the given vertices is 15
theorem triangle_area_ABC : triangle_area A B C = 15 :=
by
  -- Proof goes here
  sorry

end triangle_area_ABC_l1348_134878


namespace sales_tax_difference_l1348_134891

theorem sales_tax_difference (price : ℝ) (tax_rate1 tax_rate2 : ℝ) :
  price = 50 → tax_rate1 = 0.075 → tax_rate2 = 0.065 →
  (price * tax_rate1 - price * tax_rate2 = 0.5) :=
by
  intros
  sorry

end sales_tax_difference_l1348_134891


namespace value_of_m_l1348_134819

theorem value_of_m (m : ℤ) : (|m| = 1) ∧ (m + 1 ≠ 0) → m = 1 := by
  sorry

end value_of_m_l1348_134819


namespace machine_loan_repaid_in_5_months_l1348_134827

theorem machine_loan_repaid_in_5_months :
  ∀ (loan cost selling_price tax_percentage products_per_month profit_per_product months : ℕ),
    loan = 22000 →
    cost = 5 →
    selling_price = 8 →
    tax_percentage = 10 →
    products_per_month = 2000 →
    profit_per_product = (selling_price - cost - (selling_price * tax_percentage / 100)) →
    (products_per_month * months * profit_per_product) ≥ loan →
    months = 5 :=
by
  intros loan cost selling_price tax_percentage products_per_month profit_per_product months
  sorry

end machine_loan_repaid_in_5_months_l1348_134827


namespace monotonic_increasing_iff_l1348_134869

noncomputable def f (x b : ℝ) : ℝ := (x - b) * Real.log x + x^2

theorem monotonic_increasing_iff (b : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → 0 ≤ (Real.log x - b/x + 1 + 2*x)) ↔ b ∈ Set.Iic (3 : ℝ) :=
by
  sorry

end monotonic_increasing_iff_l1348_134869


namespace hillary_descending_rate_is_1000_l1348_134852

-- Definitions from the conditions
def base_to_summit_distance : ℕ := 5000
def hillary_departure_time : ℕ := 6
def hillary_climbing_rate : ℕ := 800
def eddy_climbing_rate : ℕ := 500
def hillary_stop_distance_from_summit : ℕ := 1000
def hillary_and_eddy_pass_time : ℕ := 12

-- Derived definitions
def hillary_climbing_time : ℕ := (base_to_summit_distance - hillary_stop_distance_from_summit) / hillary_climbing_rate
def hillary_stop_time : ℕ := hillary_departure_time + hillary_climbing_time
def eddy_climbing_time_at_pass : ℕ := hillary_and_eddy_pass_time - hillary_departure_time
def eddy_climbed_distance : ℕ := eddy_climbing_rate * eddy_climbing_time_at_pass
def hillary_distance_descended_at_pass : ℕ := (base_to_summit_distance - hillary_stop_distance_from_summit) - eddy_climbed_distance
def hillary_descending_time : ℕ := hillary_and_eddy_pass_time - hillary_stop_time 

def hillary_descending_rate : ℕ := hillary_distance_descended_at_pass / hillary_descending_time

-- Statement to prove
theorem hillary_descending_rate_is_1000 : hillary_descending_rate = 1000 := 
by
  sorry

end hillary_descending_rate_is_1000_l1348_134852


namespace square_area_is_81_l1348_134800

def square_perimeter (s : ℕ) : ℕ := 4 * s
def square_area (s : ℕ) : ℕ := s * s

theorem square_area_is_81 (s : ℕ) (h : square_perimeter s = 36) : square_area s = 81 :=
by {
  sorry
}

end square_area_is_81_l1348_134800


namespace range_of_a_l1348_134857

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l1348_134857


namespace find_angle4_l1348_134821

theorem find_angle4 (angle1 angle2 angle3 angle4 : ℝ)
                    (h1 : angle1 + angle2 = 180)
                    (h2 : angle3 = 2 * angle4)
                    (h3 : angle1 = 50)
                    (h4 : angle3 + angle4 = 130) : 
                    angle4 = 130 / 3 := by 
    sorry

end find_angle4_l1348_134821


namespace average_eq_16_l1348_134886

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 12

theorem average_eq_16 (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : (x + y) / 2 = 16 := by
  sorry

end average_eq_16_l1348_134886


namespace arthur_money_left_l1348_134865

theorem arthur_money_left {initial_amount spent_fraction : ℝ} (h_initial : initial_amount = 200) (h_fraction : spent_fraction = 4 / 5) : 
  (initial_amount - spent_fraction * initial_amount = 40) :=
by
  sorry

end arthur_money_left_l1348_134865


namespace percentage_of_boys_to_girls_l1348_134834

theorem percentage_of_boys_to_girls
  (boys : ℕ) (girls : ℕ)
  (h1 : boys = 20)
  (h2 : girls = 26) :
  (boys / girls : ℝ) * 100 = 76.9 := by
  sorry

end percentage_of_boys_to_girls_l1348_134834


namespace total_weight_correct_l1348_134879

-- Define the constant variables as per the conditions
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def fill_percentage : ℝ := 0.7
def jug1_density : ℝ := 4
def jug2_density : ℝ := 5

-- Define the volumes of sand in each jug
def jug1_sand_volume : ℝ := fill_percentage * jug1_capacity
def jug2_sand_volume : ℝ := fill_percentage * jug2_capacity

-- Define the weights of sand in each jug
def jug1_weight : ℝ := jug1_sand_volume * jug1_density
def jug2_weight : ℝ := jug2_sand_volume * jug2_density

-- State the theorem that combines the weights
theorem total_weight_correct : jug1_weight + jug2_weight = 16.1 := sorry

end total_weight_correct_l1348_134879


namespace lines_perpendicular_l1348_134874

-- Define the conditions: lines not parallel to the coordinate planes 
-- (which translates to k_1 and k_2 not being infinite, but we can code it directly as a statement on the product being -1)
variable {k1 k2 l1 l2 : ℝ} 

-- Define the theorem statement 
theorem lines_perpendicular (hk : k1 * k2 = -1) : 
  ∀ (x : ℝ), (k1 ≠ 0) ∧ (k2 ≠ 0) → 
  (∀ (y1 y2 : ℝ), y1 = k1 * x + l1 → y2 = k2 * x + l2 → 
  (k1 * k2 = -1)) :=
sorry

end lines_perpendicular_l1348_134874


namespace avg_annual_growth_rate_optimal_selling_price_l1348_134898

-- Define the conditions and question for the first problem: average annual growth rate.
theorem avg_annual_growth_rate (initial final : ℝ) (years : ℕ) (growth_rate : ℝ) :
  initial = 200 ∧ final = 288 ∧ years = 2 ∧ (final = initial * (1 + growth_rate)^years) →
  growth_rate = 0.2 :=
by
  -- Proof will come here
  sorry

-- Define the conditions and question for the second problem: setting the selling price.
theorem optimal_selling_price (cost initial_volume : ℕ) (initial_price : ℝ) 
(additional_sales_per_dollar : ℕ) (desired_profit : ℝ) (optimal_price : ℝ) :
  cost = 50 ∧ initial_volume = 50 ∧ initial_price = 100 ∧ additional_sales_per_dollar = 5 ∧
  desired_profit = 4000 ∧ 
  (∃ p : ℝ, (p - cost) * (initial_volume + additional_sales_per_dollar * (initial_price - p)) = desired_profit ∧ p = optimal_price) →
  optimal_price = 70 :=
by
  -- Proof will come here
  sorry

end avg_annual_growth_rate_optimal_selling_price_l1348_134898


namespace geometric_sequence_a7_a8_l1348_134823

-- Define the geometric sequence {a_n}
variable {a : ℕ → ℝ}

-- {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Conditions
axiom h1 : is_geometric_sequence a
axiom h2 : a 1 + a 2 = 40
axiom h3 : a 3 + a 4 = 60

-- Proof problem: Find a_7 + a_8
theorem geometric_sequence_a7_a8 :
  a 7 + a 8 = 135 :=
by
  sorry

end geometric_sequence_a7_a8_l1348_134823


namespace range_of_m_l1348_134832

variable (m : ℝ)

/-- Proposition p: For any x in ℝ, x^2 + 1 > m -/
def p := ∀ x : ℝ, x^2 + 1 > m

/-- Proposition q: The linear function f(x) = (2 - m) * x + 1 is an increasing function -/
def q := (2 - m) > 0

theorem range_of_m (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 1 < m ∧ m < 2 := 
sorry

end range_of_m_l1348_134832


namespace savings_by_buying_gallon_l1348_134835

def gallon_to_ounces : ℕ := 128
def bottle_volume_ounces : ℕ := 16
def cost_gallon : ℕ := 8
def cost_bottle : ℕ := 3

theorem savings_by_buying_gallon :
  (cost_bottle * (gallon_to_ounces / bottle_volume_ounces)) - cost_gallon = 16 := 
by
  sorry

end savings_by_buying_gallon_l1348_134835


namespace quadratic_real_roots_a_condition_l1348_134840

theorem quadratic_real_roots_a_condition (a : ℝ) (h : ∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) :
  a ≥ 1 ∧ a ≠ 5 :=
by
  sorry

end quadratic_real_roots_a_condition_l1348_134840


namespace k_range_m_range_l1348_134807

noncomputable def f (x : ℝ) : ℝ := 1 - (2 / (2^x + 1))

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem k_range (k : ℝ) : (∃ x : ℝ, g x = (2^x + 1) * f x + k) → k < 1 :=
by
  sorry

theorem m_range (m : ℝ) : (∀ x1 : ℝ, 0 < x1 ∧ x1 < 1 → 
                        ∃ x2 : ℝ, -Real.pi / 4 ≤ x2 ∧ x2 ≤ Real.pi / 6 ∧ f x1 - m * 2^x1 > g x2) 
                       → m ≤ 7 / 6 :=
by
  sorry

end k_range_m_range_l1348_134807


namespace teresa_total_marks_l1348_134871

theorem teresa_total_marks :
  let science_marks := 70
  let music_marks := 80
  let social_studies_marks := 85
  let physics_marks := 1 / 2 * music_marks
  science_marks + music_marks + social_studies_marks + physics_marks = 275 :=
by
  sorry

end teresa_total_marks_l1348_134871


namespace sum_of_differences_of_7_in_657932657_l1348_134826

theorem sum_of_differences_of_7_in_657932657 :
  let numeral := 657932657
  let face_value (d : Nat) := d
  let local_value (d : Nat) (pos : Nat) := d * 10 ^ pos
  let indices_of_7 := [6, 0]
  let differences := indices_of_7.map (fun pos => local_value 7 pos - face_value 7)
  differences.sum = 6999993 :=
by
  sorry

end sum_of_differences_of_7_in_657932657_l1348_134826


namespace total_homework_time_l1348_134887

variable (num_math_problems num_social_studies_problems num_science_problems : ℕ)
variable (time_per_math_problem time_per_social_studies_problem time_per_science_problem : ℝ)

/-- Prove that the total time taken by Brooke to answer all his homework problems is 48 minutes -/
theorem total_homework_time :
  num_math_problems = 15 →
  num_social_studies_problems = 6 →
  num_science_problems = 10 →
  time_per_math_problem = 2 →
  time_per_social_studies_problem = 0.5 →
  time_per_science_problem = 1.5 →
  (num_math_problems * time_per_math_problem + num_social_studies_problems * time_per_social_studies_problem + num_science_problems * time_per_science_problem) = 48 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_homework_time_l1348_134887


namespace free_fall_time_l1348_134889

theorem free_fall_time (h : ℝ) (t : ℝ) (h_eq : h = 4.9 * t^2) (h_val : h = 490) : t = 10 :=
by
  sorry

end free_fall_time_l1348_134889


namespace log_domain_l1348_134831

theorem log_domain (x : ℝ) : x + 2 > 0 ↔ x ∈ Set.Ioi (-2) :=
by
  sorry

end log_domain_l1348_134831


namespace roots_exist_l1348_134899

theorem roots_exist (a : ℝ) : ∃ x : ℝ, a * x^2 - x = 0 := by
  sorry

end roots_exist_l1348_134899


namespace sum_first_11_terms_of_arithmetic_sequence_l1348_134847

noncomputable def sum_arithmetic_sequence (n : ℕ) (a1 an : ℤ) : ℤ :=
  n * (a1 + an) / 2

theorem sum_first_11_terms_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : S n = sum_arithmetic_sequence n (a 1) (a n))
  (h2 : a 3 + a 6 + a 9 = 60) : S 11 = 220 :=
sorry

end sum_first_11_terms_of_arithmetic_sequence_l1348_134847


namespace evaluate_expression_l1348_134896

theorem evaluate_expression (x : ℤ) (h : x = 2) : 20 - 2 * (3 * x^2 - 4 * x + 8) = -4 :=
by
  rw [h]
  sorry

end evaluate_expression_l1348_134896


namespace GlobalConnect_more_cost_effective_if_x_300_l1348_134806

def GlobalConnectCost (x : ℕ) : ℝ := 50 + 0.4 * x
def QuickConnectCost (x : ℕ) : ℝ := 0.6 * x

theorem GlobalConnect_more_cost_effective_if_x_300 : 
  GlobalConnectCost 300 < QuickConnectCost 300 :=
by
  sorry

end GlobalConnect_more_cost_effective_if_x_300_l1348_134806


namespace max_minutes_sleep_without_missing_happy_moment_l1348_134805

def isHappyMoment (h m : ℕ) : Prop :=
  (h = 4 * m ∨ m = 4 * h) ∧ h < 24 ∧ m < 60

def sleepDurationMax : ℕ :=
  239

theorem max_minutes_sleep_without_missing_happy_moment :
  ∀ (sleepDuration : ℕ), sleepDuration ≤ 239 :=
sorry

end max_minutes_sleep_without_missing_happy_moment_l1348_134805


namespace number_of_new_galleries_l1348_134876

-- Definitions based on conditions
def number_of_pictures_first_gallery := 9
def number_of_pictures_per_new_gallery := 2
def pencils_per_picture := 4
def pencils_per_exhibition_signature := 2
def total_pencils_used := 88

-- Theorem statement according to the correct answer
theorem number_of_new_galleries 
  (number_of_pictures_first_gallery : ℕ)
  (number_of_pictures_per_new_gallery : ℕ)
  (pencils_per_picture : ℕ)
  (pencils_per_exhibition_signature : ℕ)
  (total_pencils_used : ℕ)
  (drawing_pencils_first_gallery := number_of_pictures_first_gallery * pencils_per_picture)
  (signing_pencils_first_gallery := pencils_per_exhibition_signature)
  (total_pencils_first_gallery := drawing_pencils_first_gallery + signing_pencils_first_gallery)
  (pencils_for_new_galleries := total_pencils_used - total_pencils_first_gallery)
  (pencils_per_new_gallery := (number_of_pictures_per_new_gallery * pencils_per_picture) + pencils_per_exhibition_signature) :
  pencils_per_new_gallery > 0 → pencils_for_new_galleries / pencils_per_new_gallery = 5 :=
sorry

end number_of_new_galleries_l1348_134876


namespace problem_statement_l1348_134856

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for f

-- Theorem stating the axis of symmetry and increasing interval for the transformed function
theorem problem_statement (hf_even : ∀ x, f x = f (-x))
  (hf_increasing : ∀ x₁ x₂, 3 < x₁ → x₁ < x₂ → x₂ < 5 → f x₁ < f x₂) :
  -- For y = f(x - 1), the following holds:
  (∀ x, (f (x - 1)) = f (-(x - 1))) ∧
  (∀ x₁ x₂, 4 < x₁ → x₁ < x₂ → x₂ < 6 → f (x₁ - 1) < f (x₂ - 1)) :=
sorry

end problem_statement_l1348_134856


namespace volume_surface_area_ratio_l1348_134822

theorem volume_surface_area_ratio
  (V : ℕ := 9)
  (S : ℕ := 34)
  (shape_conditions : ∃ n : ℕ, n = 9 ∧ ∃ m : ℕ, m = 2) :
  V / S = 9 / 34 :=
by
  sorry

end volume_surface_area_ratio_l1348_134822


namespace jane_last_segment_speed_l1348_134877

theorem jane_last_segment_speed :
  let total_distance := 120  -- in miles
  let total_time := (75 / 60)  -- in hours
  let segment_time := (25 / 60)  -- in hours
  let speed1 := 75  -- in mph
  let speed2 := 80  -- in mph
  let overall_avg_speed := total_distance / total_time
  let x := (3 * overall_avg_speed) - speed1 - speed2
  x = 133 :=
by { sorry }

end jane_last_segment_speed_l1348_134877


namespace tsunami_added_sand_l1348_134836

noncomputable def dig_rate : ℝ := 8 / 4 -- feet per hour
noncomputable def sand_after_storm : ℝ := 8 / 2 -- feet
noncomputable def time_to_dig_up_treasure : ℝ := 3 -- hours
noncomputable def total_sand_dug_up : ℝ := dig_rate * time_to_dig_up_treasure -- feet

theorem tsunami_added_sand :
  total_sand_dug_up - sand_after_storm = 2 :=
by
  sorry

end tsunami_added_sand_l1348_134836


namespace solve_for_x_l1348_134893

theorem solve_for_x (x : ℝ) (h : |3990 * x + 1995| = 1995) : x = 0 ∨ x = -1 :=
by
  sorry

end solve_for_x_l1348_134893


namespace triangle_proof_l1348_134888

theorem triangle_proof (a b : ℝ) (cosA : ℝ) (ha : a = 6) (hb : b = 5) (hcosA : cosA = -4 / 5) :
  (∃ B : ℝ, B = 30) ∧ (∃ area : ℝ, area = (9 * Real.sqrt 3 - 12) / 2) :=
  by
  sorry

end triangle_proof_l1348_134888


namespace solve_system_of_equations_l1348_134868

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x * y * (x + y) = 30 ∧ x^3 + y^3 = 35 ∧ ((x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = 3)) :=
sorry

end solve_system_of_equations_l1348_134868


namespace find_n_l1348_134850

theorem find_n (n : ℕ) (hn : (Nat.choose n 2 : ℚ) / 2^n = 10 / 32) : n = 5 :=
by
  sorry

end find_n_l1348_134850


namespace mark_brings_in_148_cans_l1348_134803

-- Define the given conditions
variable (R : ℕ) (Mark Jaydon Sophie : ℕ)

-- Conditions
def jaydon_cans := 2 * R + 5
def mark_cans := 4 * jaydon_cans
def unit_ratio := mark_cans / 4
def sophie_cans := 2 * unit_ratio

-- Condition: Total cans
def total_cans := mark_cans + jaydon_cans + sophie_cans

-- Condition: Each contributes at least 5 cans
axiom each_contributes_at_least_5 : R ≥ 5

-- Condition: Total cans is an odd number not less than 250
axiom total_odd_not_less_than_250 : ∃ k : ℕ, total_cans = 2 * k + 1 ∧ total_cans ≥ 250

-- Theorem: Prove Mark brings in 148 cans under the conditions
theorem mark_brings_in_148_cans (h : R = 16) : mark_cans = 148 :=
by sorry

end mark_brings_in_148_cans_l1348_134803
