import Mathlib

namespace NUMINAMATH_GPT_terminating_decimals_count_l681_68191

theorem terminating_decimals_count :
  (∃ count : ℕ, count = 166 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ 499 → (∃ m : ℕ, n = 3 * m)) :=
sorry

end NUMINAMATH_GPT_terminating_decimals_count_l681_68191


namespace NUMINAMATH_GPT_number_of_even_ones_matrices_l681_68134

noncomputable def count_even_ones_matrices (m n : ℕ) : ℕ :=
if m = 0 ∨ n = 0 then 1 else 2^((m-1)*(n-1))

theorem number_of_even_ones_matrices (m n : ℕ) : 
  count_even_ones_matrices m n = 2^((m-1)*(n-1)) := sorry

end NUMINAMATH_GPT_number_of_even_ones_matrices_l681_68134


namespace NUMINAMATH_GPT_waiter_earning_correct_l681_68129

-- Definitions based on the conditions
def tip1 : ℝ := 25 * 0.15
def tip2 : ℝ := 22 * 0.18
def tip3 : ℝ := 35 * 0.20
def tip4 : ℝ := 30 * 0.10

def total_tips : ℝ := tip1 + tip2 + tip3 + tip4
def commission : ℝ := total_tips * 0.05
def net_tips : ℝ := total_tips - commission

-- Theorem statement
theorem waiter_earning_correct : net_tips = 16.82 := by
  sorry

end NUMINAMATH_GPT_waiter_earning_correct_l681_68129


namespace NUMINAMATH_GPT_problem_area_of_circle_l681_68138

noncomputable def circleAreaPortion : ℝ :=
  let r := Real.sqrt 59
  let theta := 135 * Real.pi / 180
  (theta / (2 * Real.pi)) * (Real.pi * r^2)

theorem problem_area_of_circle :
  circleAreaPortion = (177 / 8) * Real.pi := by
  sorry

end NUMINAMATH_GPT_problem_area_of_circle_l681_68138


namespace NUMINAMATH_GPT_trey_uses_47_nails_l681_68128

variable (D : ℕ) -- total number of decorations
variable (nails thumbtacks sticky_strips : ℕ)

-- Conditions
def uses_nails := nails = (5 * D) / 8
def uses_thumbtacks := thumbtacks = (9 * D) / 80
def uses_sticky_strips := sticky_strips = 20
def total_decorations := (21 * D) / 80 = 20

-- Question: Prove that Trey uses 47 nails when the conditions hold
theorem trey_uses_47_nails (D : ℕ) (h1 : uses_nails D nails) (h2 : uses_thumbtacks D thumbtacks) (h3 : uses_sticky_strips sticky_strips) (h4 : total_decorations D) : nails = 47 :=  
by
  sorry

end NUMINAMATH_GPT_trey_uses_47_nails_l681_68128


namespace NUMINAMATH_GPT_bottles_of_regular_soda_l681_68186

theorem bottles_of_regular_soda (R : ℕ) : 
  let apples := 36 
  let diet_soda := 54
  let total_bottles := apples + 98 
  R + diet_soda = total_bottles → R = 80 :=
by
  sorry

end NUMINAMATH_GPT_bottles_of_regular_soda_l681_68186


namespace NUMINAMATH_GPT_lighter_boxes_weight_l681_68135

noncomputable def weight_lighter_boxes (W L H : ℕ) : Prop :=
  L + H = 30 ∧
  (L * W + H * 20) / 30 = 18 ∧
  (H - 15) = 0 ∧
  (15 + L - H = 15 ∧ 15 * 16 = 15 * W)

theorem lighter_boxes_weight :
  ∃ W, ∀ L H, weight_lighter_boxes W L H → W = 16 :=
by sorry

end NUMINAMATH_GPT_lighter_boxes_weight_l681_68135


namespace NUMINAMATH_GPT_remainder_2023_mul_7_div_45_l681_68165

/-- The remainder when the product of 2023 and 7 is divided by 45 is 31. -/
theorem remainder_2023_mul_7_div_45 : 
  (2023 * 7) % 45 = 31 := 
by
  sorry

end NUMINAMATH_GPT_remainder_2023_mul_7_div_45_l681_68165


namespace NUMINAMATH_GPT_solution_set_f_l681_68153

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x^(1/2) - 1

theorem solution_set_f (x : ℝ) (hx_pos : x > 0) : 
  f x > f (2 * x - 4) ↔ 2 < x ∧ x < 4 :=
sorry

end NUMINAMATH_GPT_solution_set_f_l681_68153


namespace NUMINAMATH_GPT_min_value_x_squared_plus_6x_l681_68182

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, x^2 + 6 * x ≥ -9 := 
by
  sorry

end NUMINAMATH_GPT_min_value_x_squared_plus_6x_l681_68182


namespace NUMINAMATH_GPT_equal_real_roots_implies_m_l681_68175

theorem equal_real_roots_implies_m (m : ℝ) : (∃ (x : ℝ), x^2 + x + m = 0 ∧ ∀ y : ℝ, y^2 + y + m = 0 → y = x) → m = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_equal_real_roots_implies_m_l681_68175


namespace NUMINAMATH_GPT_max_total_weight_of_chocolates_l681_68123

theorem max_total_weight_of_chocolates 
  (A B C : ℕ)
  (hA : A ≤ 100)
  (hBC : B - C ≤ 100)
  (hC : C ≤ 100)
  (h_distribute : A ≤ 100 ∧ (B - C) ≤ 100)
  : (A + B = 300) :=
by 
  sorry

end NUMINAMATH_GPT_max_total_weight_of_chocolates_l681_68123


namespace NUMINAMATH_GPT_parallelogram_angles_l681_68144

theorem parallelogram_angles (x y : ℝ) (h_sub : y = x + 50) (h_sum : x + y = 180) : x = 65 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_angles_l681_68144


namespace NUMINAMATH_GPT_multiplication_with_mixed_number_l681_68162

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end NUMINAMATH_GPT_multiplication_with_mixed_number_l681_68162


namespace NUMINAMATH_GPT_circle_through_A_B_and_tangent_to_m_l681_68150

noncomputable def circle_equation (x y : ℚ) : Prop :=
  x^2 + (y - 1/3)^2 = 16/9

theorem circle_through_A_B_and_tangent_to_m :
  ∃ (c : ℚ × ℚ) (r : ℚ),
    (c = (0, 1/3)) ∧
    (r = 4/3) ∧
    (∀ (x y : ℚ),
      (x = 0 ∧ y = -1 ∨ x = 4/3 ∧ y = 1/3 → (x^2 + (y - 1/3)^2 = 16/9)) ∧
      (x = 4/3 → x = r)) :=
by
  sorry

end NUMINAMATH_GPT_circle_through_A_B_and_tangent_to_m_l681_68150


namespace NUMINAMATH_GPT_moles_NaHCO3_combined_l681_68131

-- Define conditions as given in the problem
def moles_HNO3_combined := 1
def moles_NaNO3_result := 1

-- The chemical equation as a definition
def balanced_reaction (moles_NaHCO3 moles_HNO3 moles_NaNO3 : ℕ) : Prop :=
  moles_HNO3 = moles_NaNO3 ∧ moles_NaHCO3 = moles_HNO3

-- The proof problem statement
theorem moles_NaHCO3_combined :
  balanced_reaction 1 moles_HNO3_combined moles_NaNO3_result → 1 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_moles_NaHCO3_combined_l681_68131


namespace NUMINAMATH_GPT_triangle_inequality_l681_68151

theorem triangle_inequality (a b c Δ : ℝ) (h_Δ: Δ = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
    a^2 + b^2 + c^2 ≥ 4 * Real.sqrt (3) * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l681_68151


namespace NUMINAMATH_GPT_average_mark_of_second_class_l681_68127

/-- 
There is a class of 30 students with an average mark of 40. 
Another class has 50 students with an unknown average mark. 
The average marks of all students combined is 65. 
Prove that the average mark of the second class is 80.
-/
theorem average_mark_of_second_class (x : ℝ) (h1 : 30 * 40 + 50 * x = 65 * (30 + 50)) : x = 80 := 
sorry

end NUMINAMATH_GPT_average_mark_of_second_class_l681_68127


namespace NUMINAMATH_GPT_comparison1_comparison2_comparison3_l681_68103

theorem comparison1 : -3.2 > -4.3 :=
by sorry

theorem comparison2 : (1 : ℚ) / 2 > -(1 / 3) :=
by sorry

theorem comparison3 : (1 : ℚ) / 4 > 0 :=
by sorry

end NUMINAMATH_GPT_comparison1_comparison2_comparison3_l681_68103


namespace NUMINAMATH_GPT_common_tangents_l681_68198

def circle1_eqn (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y = 0
def circle2_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

theorem common_tangents :
  ∃ (n : ℕ), n = 4 ∧ 
    (∀ (L : ℝ → ℝ → Prop), 
      (∀ x y, L x y → circle1_eqn x y ∧ circle2_eqn x y) → n = 4) := 
sorry

end NUMINAMATH_GPT_common_tangents_l681_68198


namespace NUMINAMATH_GPT_calc_exponent_l681_68139

theorem calc_exponent (a b : ℕ) : 1^345 + 5^7 / 5^5 = 26 := by
  sorry

end NUMINAMATH_GPT_calc_exponent_l681_68139


namespace NUMINAMATH_GPT_sum_of_sixth_powers_l681_68112

theorem sum_of_sixth_powers (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 0) 
  (h2 : α₁^2 + α₂^2 + α₃^2 = 2) 
  (h3 : α₁^3 + α₂^3 + α₃^3 = 4) : 
  α₁^6 + α₂^6 + α₃^6 = 7 :=
sorry

end NUMINAMATH_GPT_sum_of_sixth_powers_l681_68112


namespace NUMINAMATH_GPT_Mika_stickers_l681_68110

theorem Mika_stickers
  (initial_stickers : ℕ)
  (bought_stickers : ℕ)
  (received_stickers : ℕ)
  (given_stickers : ℕ)
  (used_stickers : ℕ)
  (final_stickers : ℕ) :
  initial_stickers = 45 →
  bought_stickers = 53 →
  received_stickers = 35 →
  given_stickers = 19 →
  used_stickers = 86 →
  final_stickers = initial_stickers + bought_stickers + received_stickers - given_stickers - used_stickers →
  final_stickers = 28 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Mika_stickers_l681_68110


namespace NUMINAMATH_GPT_initial_innings_count_l681_68137

theorem initial_innings_count (n T L : ℕ) 
  (h1 : T = 50 * n)
  (h2 : 174 = L + 172)
  (h3 : (T - 174 - L) = 48 * (n - 2)) :
  n = 40 :=
by 
  sorry

end NUMINAMATH_GPT_initial_innings_count_l681_68137


namespace NUMINAMATH_GPT_find_k_l681_68113

theorem find_k
  (S : ℝ)    -- Distance between the village and city
  (x : ℝ)    -- Speed of the truck in km/h
  (y : ℝ)    -- Speed of the car in km/h
  (H1 : 18 = 0.75 * x - 0.75 * x ^ 2 / (x + y))  -- Condition that truck leaving earlier meets 18 km closer to the city
  (H2 : 24 = x * y / (x + y))      -- Intermediate step from solving the first condition
  : (k = 8) :=    -- We need to show that k = 8
  sorry

end NUMINAMATH_GPT_find_k_l681_68113


namespace NUMINAMATH_GPT_ratio_of_profits_is_2_to_3_l681_68155

-- Conditions
def Praveen_initial_investment := 3220
def Praveen_investment_duration := 12
def Hari_initial_investment := 8280
def Hari_investment_duration := 7

-- Effective capital contributions
def Praveen_effective_capital : ℕ := Praveen_initial_investment * Praveen_investment_duration
def Hari_effective_capital : ℕ := Hari_initial_investment * Hari_investment_duration

-- Theorem statement to be proven
theorem ratio_of_profits_is_2_to_3 : (Praveen_effective_capital : ℚ) / Hari_effective_capital = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_profits_is_2_to_3_l681_68155


namespace NUMINAMATH_GPT_find_other_number_l681_68149

theorem find_other_number (a b : ℤ) (h1 : 2 * a + 3 * b = 100) (h2 : a = 28 ∨ b = 28) : a = 8 ∨ b = 8 :=
sorry

end NUMINAMATH_GPT_find_other_number_l681_68149


namespace NUMINAMATH_GPT_sum_of_interior_angles_n_plus_4_l681_68106

    noncomputable def sum_of_interior_angles (sides : ℕ) : ℝ :=
      180 * (sides - 2)

    theorem sum_of_interior_angles_n_plus_4 (n : ℕ) (h : sum_of_interior_angles n = 2340) :
      sum_of_interior_angles (n + 4) = 3060 :=
    by
      sorry
    
end NUMINAMATH_GPT_sum_of_interior_angles_n_plus_4_l681_68106


namespace NUMINAMATH_GPT_value_of_n_l681_68111

theorem value_of_n (n : ℤ) :
  (∀ x : ℤ, (x + n) * (x + 2) = x^2 + 2 * x + n * x + 2 * n → 2 + n = 0) → n = -2 := 
by
  intro h
  have h1 := h 0
  sorry

end NUMINAMATH_GPT_value_of_n_l681_68111


namespace NUMINAMATH_GPT_distance_between_a_and_c_l681_68166

-- Given conditions
variables (a : ℝ)

-- Statement to prove
theorem distance_between_a_and_c : |a + 1| = |a - (-1)| :=
by sorry

end NUMINAMATH_GPT_distance_between_a_and_c_l681_68166


namespace NUMINAMATH_GPT_function_not_strictly_decreasing_l681_68169

theorem function_not_strictly_decreasing (b : ℝ)
  (h : ¬ ∀ x1 x2 : ℝ, x1 < x2 → (-x1^3 + b*x1^2 - (2*b + 3)*x1 + 2 - b > -x2^3 + b*x2^2 - (2*b + 3)*x2 + 2 - b)) : 
  b < -1 ∨ b > 3 :=
by
  sorry

end NUMINAMATH_GPT_function_not_strictly_decreasing_l681_68169


namespace NUMINAMATH_GPT_train_speed_platform_man_l681_68178

theorem train_speed_platform_man (t_man t_platform : ℕ) (platform_length : ℕ) (v_train_mps : ℝ) (v_train_kmph : ℝ) 
  (h1 : t_man = 18) 
  (h2 : t_platform = 32) 
  (h3 : platform_length = 280)
  (h4 : v_train_mps = (platform_length / (t_platform - t_man)))
  (h5 : v_train_kmph = v_train_mps * 3.6) :
  v_train_kmph = 72 := 
sorry

end NUMINAMATH_GPT_train_speed_platform_man_l681_68178


namespace NUMINAMATH_GPT_find_c_l681_68120

-- Define the polynomial P(x)
def P (c : ℚ) (x : ℚ) : ℚ := x^3 + 4 * x^2 + c * x + 20

-- Given that x - 3 is a factor of P(x), prove that c = -83/3
theorem find_c (c : ℚ) (h : P c 3 = 0) : c = -83 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l681_68120


namespace NUMINAMATH_GPT_problem_proof_l681_68122

theorem problem_proof (x : ℝ) (h : x + 1/x = 3) : (x - 3) ^ 2 + 36 / (x - 3) ^ 2 = 12 :=
sorry

end NUMINAMATH_GPT_problem_proof_l681_68122


namespace NUMINAMATH_GPT_equal_numbers_l681_68167

theorem equal_numbers {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = ab + bc + cd + da) :
  a = b ∧ b = c ∧ c = d :=
by
  sorry

end NUMINAMATH_GPT_equal_numbers_l681_68167


namespace NUMINAMATH_GPT_computer_program_X_value_l681_68152

theorem computer_program_X_value : 
  ∃ (n : ℕ), (let X := 5 + 3 * (n - 1) 
               let S := (3 * n^2 + 7 * n) / 2 
               S ≥ 10500) ∧ X = 251 :=
sorry

end NUMINAMATH_GPT_computer_program_X_value_l681_68152


namespace NUMINAMATH_GPT_remaining_pictures_l681_68190

theorem remaining_pictures (first_book : ℕ) (second_book : ℕ) (third_book : ℕ) (colored_pictures : ℕ) :
  first_book = 23 → second_book = 32 → third_book = 45 → colored_pictures = 44 →
  (first_book + second_book + third_book - colored_pictures) = 56 :=
by
  sorry

end NUMINAMATH_GPT_remaining_pictures_l681_68190


namespace NUMINAMATH_GPT_determine_b_l681_68196

theorem determine_b (N a b c : ℤ) (h1 : a > 1 ∧ b > 1 ∧ c > 1) (h2 : N ≠ 1)
  (h3 : (N : ℝ) ^ (1 / a + 1 / (a * b) + 1 / (a * b * c) + 1 / (a * b * c ^ 2)) = N ^ (49 / 60)) :
  b = 4 :=
sorry

end NUMINAMATH_GPT_determine_b_l681_68196


namespace NUMINAMATH_GPT_hilt_books_transaction_difference_l681_68119

noncomputable def total_cost_paid (original_price : ℝ) (num_first_books : ℕ) (discount1 : ℝ) (num_second_books : ℕ) (discount2 : ℝ) : ℝ :=
  let cost_first_books := num_first_books * original_price * (1 - discount1)
  let cost_second_books := num_second_books * original_price * (1 - discount2)
  cost_first_books + cost_second_books

noncomputable def total_sale_amount (sale_price : ℝ) (interest_rate : ℝ) (num_books : ℕ) : ℝ :=
  let compounded_price := sale_price * (1 + interest_rate) ^ 1
  compounded_price * num_books

theorem hilt_books_transaction_difference : 
  let original_price := 11
  let num_first_books := 10
  let discount1 := 0.20
  let num_second_books := 5
  let discount2 := 0.25
  let sale_price := 25
  let interest_rate := 0.05
  let num_books := 15
  total_sale_amount sale_price interest_rate num_books - total_cost_paid original_price num_first_books discount1 num_second_books discount2 = 264.50 :=
by
  sorry

end NUMINAMATH_GPT_hilt_books_transaction_difference_l681_68119


namespace NUMINAMATH_GPT_cos_225_eq_neg_sqrt2_div_2_l681_68176

theorem cos_225_eq_neg_sqrt2_div_2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_225_eq_neg_sqrt2_div_2_l681_68176


namespace NUMINAMATH_GPT_a_n_is_perfect_square_l681_68161

theorem a_n_is_perfect_square :
  ∀ (a b : ℕ → ℤ), a 0 = 1 → b 0 = 0 →
  (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) →
  (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) →
  ∀ n, ∃ k : ℤ, a n = k * k :=
by
  sorry

end NUMINAMATH_GPT_a_n_is_perfect_square_l681_68161


namespace NUMINAMATH_GPT_find_N_l681_68195

/-- Given a row: [a, b, c, d, 2, f, g], 
    first column: [15, h, i, 14, j, k, l, 10],
    second column: [N, m, n, o, p, q, r, -21],
    where h=i+4 and i=j+4,
    b=15 and d = (2 - 15) / 3.
    The common difference c_n = -2.5.
    Prove N = -13.5.
-/
theorem find_N (a b c d f g h i j k l m n o p q r : ℝ) (N : ℝ) :
  b = 15 ∧ j = 14 ∧ l = 10 ∧ r = -21 ∧
  h = i + 4 ∧ i = j + 4 ∧
  c = (2 - 15) / 3 ∧
  g = b + 6 * c ∧
  N = g + 1 * (-2.5) →
  N = -13.5 :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_find_N_l681_68195


namespace NUMINAMATH_GPT_vehicle_value_this_year_l681_68109

variable (V_last_year : ℝ) (V_this_year : ℝ)

-- Conditions
def last_year_value : ℝ := 20000
def this_year_value : ℝ := 0.8 * last_year_value

theorem vehicle_value_this_year :
  V_last_year = last_year_value →
  V_this_year = this_year_value →
  V_this_year = 16000 := sorry

end NUMINAMATH_GPT_vehicle_value_this_year_l681_68109


namespace NUMINAMATH_GPT_weighted_average_score_l681_68189

def weight (subject_mark : Float) (weight_percentage : Float) : Float :=
    subject_mark * weight_percentage

theorem weighted_average_score :
    (weight 61 0.2) + (weight 65 0.25) + (weight 82 0.3) + (weight 67 0.15) + (weight 85 0.1) = 71.6 := by
    sorry

end NUMINAMATH_GPT_weighted_average_score_l681_68189


namespace NUMINAMATH_GPT_grasshopper_jump_l681_68126

theorem grasshopper_jump :
  ∃ (x y : ℤ), 80 * x - 50 * y = 170 ∧ x + y ≤ 7 := by
  sorry

end NUMINAMATH_GPT_grasshopper_jump_l681_68126


namespace NUMINAMATH_GPT_total_turtles_l681_68108

variable (Kristen_turtles Kris_turtles Trey_turtles : ℕ)

-- Kristen has 12 turtles
def Kristen_turtles_count : Kristen_turtles = 12 := sorry

-- Kris has 1/4 the number of turtles Kristen has
def Kris_turtles_count (hK : Kristen_turtles = 12) : Kris_turtles = Kristen_turtles / 4 := sorry

-- Trey has 5 times as many turtles as Kris
def Trey_turtles_count (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) : Trey_turtles = 5 * Kris_turtles := sorry

-- Total number of turtles
theorem total_turtles (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) 
  (hT : Trey_turtles = 5 * Kris_turtles) : Kristen_turtles + Kris_turtles + Trey_turtles = 30 := sorry

end NUMINAMATH_GPT_total_turtles_l681_68108


namespace NUMINAMATH_GPT_count_real_solutions_l681_68179

theorem count_real_solutions :
  ∃ x1 x2 : ℝ, (|x1-1| = |x1-2| + |x1-3| + |x1-4| ∧ |x2-1| = |x2-2| + |x2-3| + |x2-4|)
  ∧ (x1 ≠ x2) :=
sorry

end NUMINAMATH_GPT_count_real_solutions_l681_68179


namespace NUMINAMATH_GPT_interval_contains_root_l681_68140

noncomputable def f (x : ℝ) : ℝ := 3^x - x^2

theorem interval_contains_root : ∃ x ∈ Set.Icc (-1 : ℝ) (0 : ℝ), f x = 0 :=
by
  have f_neg : f (-1) < 0 := by sorry
  have f_zero : f 0 > 0 := by sorry
  sorry

end NUMINAMATH_GPT_interval_contains_root_l681_68140


namespace NUMINAMATH_GPT_combined_tax_rate_35_58_l681_68156

noncomputable def combined_tax_rate (john_income : ℝ) (john_tax_rate : ℝ) (ingrid_income : ℝ) (ingrid_tax_rate : ℝ) : ℝ :=
  let john_tax := john_tax_rate * john_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let total_tax := john_tax + ingrid_tax
  let total_income := john_income + ingrid_income
  (total_tax / total_income) * 100

theorem combined_tax_rate_35_58
  (john_income : ℝ) (john_tax_rate : ℝ) (ingrid_income : ℝ) (ingrid_tax_rate : ℝ)
  (h1 : john_income = 57000) (h2 : john_tax_rate = 0.3)
  (h3 : ingrid_income = 72000) (h4 : ingrid_tax_rate = 0.4) :
  combined_tax_rate john_income john_tax_rate ingrid_income ingrid_tax_rate = 35.58 :=
by
  simp [combined_tax_rate, h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_combined_tax_rate_35_58_l681_68156


namespace NUMINAMATH_GPT_maximum_value_of_f_l681_68192

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem maximum_value_of_f :
  ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = (Real.pi / 6) + Real.sqrt 3 ∧ 
  ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f (Real.pi / 6) :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_l681_68192


namespace NUMINAMATH_GPT_lcm_gcd_pairs_l681_68118

theorem lcm_gcd_pairs (a b : ℕ) :
  (lcm a b + gcd a b = (a * b) / 5) ↔
  (a = 10 ∧ b = 10) ∨ (a = 6 ∧ b = 30) ∨ (a = 30 ∧ b = 6) :=
sorry

end NUMINAMATH_GPT_lcm_gcd_pairs_l681_68118


namespace NUMINAMATH_GPT_original_side_length_l681_68143

theorem original_side_length (x : ℝ) (h1 : (x - 6) * (x - 5) = 120) : x = 15 :=
sorry

end NUMINAMATH_GPT_original_side_length_l681_68143


namespace NUMINAMATH_GPT_greatest_possible_value_y_l681_68107

theorem greatest_possible_value_y
  (x y : ℤ)
  (h : x * y + 3 * x + 2 * y = -6) : 
  y ≤ 3 :=
by sorry

end NUMINAMATH_GPT_greatest_possible_value_y_l681_68107


namespace NUMINAMATH_GPT_avg_growth_rate_eq_l681_68171

variable (x : ℝ)

theorem avg_growth_rate_eq :
  (560 : ℝ) * (1 + x)^2 = 830 :=
sorry

end NUMINAMATH_GPT_avg_growth_rate_eq_l681_68171


namespace NUMINAMATH_GPT_probability_of_at_least_two_same_rank_approx_l681_68183

noncomputable def probability_at_least_two_same_rank (cards_drawn : ℕ) (total_cards : ℕ) : ℝ :=
  let ranks := 13
  let different_ranks_comb := Nat.choose ranks cards_drawn
  let rank_suit_combinations := different_ranks_comb * (4 ^ cards_drawn)
  let total_combinations := Nat.choose total_cards cards_drawn
  let p_complement := rank_suit_combinations / total_combinations
  1 - p_complement

theorem probability_of_at_least_two_same_rank_approx (h : 5 ≤ 52) : 
  abs (probability_at_least_two_same_rank 5 52 - 0.49) < 0.01 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_at_least_two_same_rank_approx_l681_68183


namespace NUMINAMATH_GPT_solve_for_y_l681_68130

theorem solve_for_y (y : ℝ) (h_pos : y > 0) (h_eq : y^2 = 1024) : y = 32 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l681_68130


namespace NUMINAMATH_GPT_inequality_solution_range_l681_68163

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℤ, 6 - 3 * (x : ℝ) < 0 ∧ 2 * (x : ℝ) ≤ a) ∧
  (∃ x1 x2 x3 : ℤ, (x1 = 3 ∧ x2 = 4 ∧ x3 = 5) ∧
   (6 - 3 * (x1 : ℝ) < 0 ∧ 2 * (x1 : ℝ) ≤ a) ∧
   (6 - 3 * (x2 : ℝ) < 0 ∧ 2 * (x2 : ℝ) ≤ a) ∧
   (6 - 3 * (x3 : ℝ) < 0 ∧ 2 * (x3 : ℝ) ≤ a) ∧
   (∀ x : ℤ, (6 - 3 * (x : ℝ) < 0 ∧ 2 * (x : ℝ) ≤ a) → 
     (x = 3 ∨ x = 4 ∨ x = 5)))
  → 10 ≤ a ∧ a < 12 :=
sorry

end NUMINAMATH_GPT_inequality_solution_range_l681_68163


namespace NUMINAMATH_GPT_hyperbola_A_asymptote_l681_68199

-- Define the hyperbola and asymptote conditions
def hyperbola_A (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1
def asymptote_eq (y x : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Statement of the proof problem in Lean 4
theorem hyperbola_A_asymptote :
  ∀ (x y : ℝ), hyperbola_A x y → asymptote_eq y x :=
sorry

end NUMINAMATH_GPT_hyperbola_A_asymptote_l681_68199


namespace NUMINAMATH_GPT_expected_value_is_0_point_25_l681_68115

-- Define the probabilities and earnings
def prob_roll_1 := 1/4
def earning_1 := 4
def prob_roll_2 := 1/4
def earning_2 := -3
def prob_roll_3_to_6 := 1/8
def earning_3_to_6 := 0

-- Define the expected value calculation
noncomputable def expected_value : ℝ := 
  (prob_roll_1 * earning_1) + 
  (prob_roll_2 * earning_2) + 
  (prob_roll_3_to_6 * earning_3_to_6) * 4  -- For 3, 4, 5, and 6

-- The theorem to be proved
theorem expected_value_is_0_point_25 : expected_value = 0.25 := by
  sorry

end NUMINAMATH_GPT_expected_value_is_0_point_25_l681_68115


namespace NUMINAMATH_GPT_second_year_associates_l681_68172

theorem second_year_associates (not_first_year : ℝ) (more_than_two_years : ℝ) 
  (h1 : not_first_year = 0.75) (h2 : more_than_two_years = 0.5) : 
  (not_first_year - more_than_two_years) = 0.25 :=
by 
  sorry

end NUMINAMATH_GPT_second_year_associates_l681_68172


namespace NUMINAMATH_GPT_investment_ratio_proof_l681_68114

noncomputable def investment_ratio {A_invest B_invest C_invest : ℝ} (profit total_profit : ℝ) (A_times_B : ℝ) : ℝ :=
  C_invest / (A_times_B * B_invest + B_invest + C_invest)

theorem investment_ratio_proof (A_invest B_invest C_invest : ℝ)
  (profit total_profit : ℝ) (A_times_B : ℝ) 
  (h_profit : total_profit = 55000)
  (h_C_share : profit = 15000.000000000002)
  (h_A_times_B : A_times_B = 3)
  (h_ratio_eq : A_times_B * B_invest + B_invest + C_invest = 11 * B_invest / 3) :
  (A_invest / C_invest = 2) :=
by
  sorry

end NUMINAMATH_GPT_investment_ratio_proof_l681_68114


namespace NUMINAMATH_GPT_largest_c_value_l681_68187

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 3 * x + c

theorem largest_c_value (c : ℝ) :
  (∃ x : ℝ, f x c = -2) ↔ c ≤ 1/4 := by
sorry

end NUMINAMATH_GPT_largest_c_value_l681_68187


namespace NUMINAMATH_GPT_competition_winner_is_C_l681_68132

-- Define the type for singers
inductive Singer
| A | B | C | D
deriving DecidableEq

-- Assume each singer makes a statement
def statement (s : Singer) : Prop :=
  match s with
  | Singer.A => Singer.B ≠ Singer.C
  | Singer.B => Singer.A ≠ Singer.C
  | Singer.C => true
  | Singer.D => Singer.B ≠ Singer.D

-- Define that two and only two statements are true
def exactly_two_statements_are_true : Prop :=
  (statement Singer.A ∧ statement Singer.C ∧ ¬statement Singer.B ∧ ¬statement Singer.D) ∨
  (statement Singer.A ∧ statement Singer.D ∧ ¬statement Singer.B ∧ ¬statement Singer.C)

-- Define the winner
def winner : Singer := Singer.C

-- The main theorem to be proved
theorem competition_winner_is_C :
  exactly_two_statements_are_true → (winner = Singer.C) :=
by
  intro h
  exact sorry

end NUMINAMATH_GPT_competition_winner_is_C_l681_68132


namespace NUMINAMATH_GPT_files_more_than_apps_l681_68146

-- Defining the initial conditions
def initial_apps : ℕ := 11
def initial_files : ℕ := 3

-- Defining the conditions after some changes
def apps_left : ℕ := 2
def files_left : ℕ := 24

-- Statement to prove
theorem files_more_than_apps : (files_left - apps_left) = 22 := 
by
  sorry

end NUMINAMATH_GPT_files_more_than_apps_l681_68146


namespace NUMINAMATH_GPT_area_of_triangle_ABC_circumcenter_of_triangle_ABC_l681_68157

structure Point where
  x : ℚ
  y : ℚ

def A : Point := ⟨2, 1⟩
def B : Point := ⟨4, 7⟩
def C : Point := ⟨8, 3⟩

def triangle_area (A B C : Point) : ℚ := by
  -- area calculation will be filled here
  sorry

def circumcenter (A B C : Point) : Point := by
  -- circumcenter calculation will be filled here
  sorry

theorem area_of_triangle_ABC : triangle_area A B C = 16 :=
  sorry

theorem circumcenter_of_triangle_ABC : circumcenter A B C = ⟨9/2, 7/2⟩ :=
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_circumcenter_of_triangle_ABC_l681_68157


namespace NUMINAMATH_GPT_train_length_correct_l681_68193

noncomputable def length_of_train (speed_km_per_hr : ℝ) (platform_length_m : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  let total_distance := speed_m_per_s * time_s
  total_distance - platform_length_m

theorem train_length_correct :
  length_of_train 55 520 43.196544276457885 = 140 :=
by
  unfold length_of_train
  -- The conversion and calculations would be verified here
  sorry

end NUMINAMATH_GPT_train_length_correct_l681_68193


namespace NUMINAMATH_GPT_remainder_of_expression_l681_68117

theorem remainder_of_expression :
  let a := 2^206 + 206
  let b := 2^103 + 2^53 + 1
  a % b = 205 := 
sorry

end NUMINAMATH_GPT_remainder_of_expression_l681_68117


namespace NUMINAMATH_GPT_length_of_second_train_l681_68188

theorem length_of_second_train 
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_seconds : ℝ)
  (same_direction : Bool) : 
  length_first_train = 380 ∧ 
  speed_first_train_kmph = 72 ∧ 
  speed_second_train_kmph = 36 ∧ 
  time_seconds = 91.9926405887529 ∧ 
  same_direction = tt → 
  ∃ L2 : ℝ, L2 = 539.93 := by
  intro h
  rcases h with ⟨hf, sf, ss, ts, sd⟩
  use 539.926405887529 -- exact value obtained in the solution
  sorry

end NUMINAMATH_GPT_length_of_second_train_l681_68188


namespace NUMINAMATH_GPT_exist_indices_l681_68158

-- Define the sequence and the conditions.
variable (x : ℕ → ℤ)
variable (H1 : x 1 = 1)
variable (H2 : ∀ n : ℕ, x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n)

theorem exist_indices (k : ℕ) (hk : 0 < k) :
  ∃ r s : ℕ, x r - x s = k := 
sorry

end NUMINAMATH_GPT_exist_indices_l681_68158


namespace NUMINAMATH_GPT_mackenzie_new_disks_l681_68100

noncomputable def price_new (U N : ℝ) : Prop := 6 * N + 2 * U = 127.92

noncomputable def disks_mackenzie_buys (U N x : ℝ) : Prop := x * N + 8 * U = 133.89

theorem mackenzie_new_disks (U N x : ℝ) (h1 : U = 9.99) (h2 : price_new U N) (h3 : disks_mackenzie_buys U N x) :
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_mackenzie_new_disks_l681_68100


namespace NUMINAMATH_GPT_final_population_l681_68116

theorem final_population (P0 : ℕ) (r1 r2 : ℝ) (P2 : ℝ) 
  (h0 : P0 = 1000)
  (h1 : r1 = 1.20)
  (h2 : r2 = 1.30)
  (h3 : P2 = P0 * r1 * r2) : 
  P2 = 1560 := 
sorry

end NUMINAMATH_GPT_final_population_l681_68116


namespace NUMINAMATH_GPT_race_time_l681_68185

theorem race_time 
    (v_A v_B t_A t_B : ℝ)
    (h1 : v_A = 1000 / t_A) 
    (h2 : v_B = 940 / t_A)
    (h3 : v_B = 1000 / (t_A + 15)) 
    (h4 : t_B = t_A + 15) :
    t_A = 235 := 
  by
    sorry

end NUMINAMATH_GPT_race_time_l681_68185


namespace NUMINAMATH_GPT_smallest_solution_floor_equation_l681_68184

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_floor_equation_l681_68184


namespace NUMINAMATH_GPT_num_valid_four_digit_numbers_l681_68177

theorem num_valid_four_digit_numbers :
  let N (a b c d : ℕ) := 1000 * a + 100 * b + 10 * c + d
  ∃ (a b c d : ℕ), 5000 ≤ N a b c d ∧ N a b c d < 7000 ∧ (N a b c d % 5 = 0) ∧ (2 ≤ b ∧ b < c ∧ c ≤ 7) ∧
                   (60 = (if a = 5 ∨ a = 6 then (if d = 0 ∨ d = 5 then 15 else 0) else 0)) :=
sorry

end NUMINAMATH_GPT_num_valid_four_digit_numbers_l681_68177


namespace NUMINAMATH_GPT_elvins_fixed_charge_l681_68159

theorem elvins_fixed_charge (F C : ℝ) 
  (h1 : F + C = 40) 
  (h2 : F + 2 * C = 76) : F = 4 := 
by 
  sorry

end NUMINAMATH_GPT_elvins_fixed_charge_l681_68159


namespace NUMINAMATH_GPT_lana_total_spending_l681_68148

noncomputable def general_admission_cost : ℝ := 6
noncomputable def vip_cost : ℝ := 10
noncomputable def premium_cost : ℝ := 15

noncomputable def num_general_admission_tickets : ℕ := 6
noncomputable def num_vip_tickets : ℕ := 2
noncomputable def num_premium_tickets : ℕ := 1

noncomputable def discount_general_admission : ℝ := 0.10
noncomputable def discount_vip : ℝ := 0.15

noncomputable def total_spending (gen_cost : ℝ) (vip_cost : ℝ) (prem_cost : ℝ) (gen_num : ℕ) (vip_num : ℕ) (prem_num : ℕ) (gen_disc : ℝ) (vip_disc : ℝ) : ℝ :=
  let general_cost := gen_cost * gen_num
  let general_discount := general_cost * gen_disc
  let discounted_general_cost := general_cost - general_discount
  let vip_cost_total := vip_cost * vip_num
  let vip_discount := vip_cost_total * vip_disc
  let discounted_vip_cost := vip_cost_total - vip_discount
  let premium_cost_total := prem_cost * prem_num
  discounted_general_cost + discounted_vip_cost + premium_cost_total

theorem lana_total_spending : total_spending general_admission_cost vip_cost premium_cost num_general_admission_tickets num_vip_tickets num_premium_tickets discount_general_admission discount_vip = 64.40 := 
sorry

end NUMINAMATH_GPT_lana_total_spending_l681_68148


namespace NUMINAMATH_GPT_Julie_simple_interest_l681_68124

variable (S : ℝ) (r : ℝ) (A : ℝ) (C : ℝ)

def initially_savings (S : ℝ) := S = 784
def half_savings_in_each_account (S A : ℝ) := A = S / 2
def compound_interest_after_two_years (A r : ℝ) := A * (1 + r)^2 - A = 120

theorem Julie_simple_interest
  (S : ℝ) (r : ℝ) (A : ℝ)
  (h1 : initially_savings S)
  (h2 : half_savings_in_each_account S A)
  (h3 : compound_interest_after_two_years A r) :
  A * r * 2 = 112 :=
by 
  sorry

end NUMINAMATH_GPT_Julie_simple_interest_l681_68124


namespace NUMINAMATH_GPT_coefficient_x2_in_expansion_l681_68194

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the problem: Given (2x + 1)^5, find the coefficient of x^2 term
theorem coefficient_x2_in_expansion : 
  binomial 5 3 * (2 ^ 2) = 40 := by 
  sorry

end NUMINAMATH_GPT_coefficient_x2_in_expansion_l681_68194


namespace NUMINAMATH_GPT_find_a_l681_68154

theorem find_a (a : ℕ) (h_pos : a > 0) (h_quadrant : 2 - a > 0) : a = 1 := by
  sorry

end NUMINAMATH_GPT_find_a_l681_68154


namespace NUMINAMATH_GPT_depth_of_tunnel_l681_68147

theorem depth_of_tunnel (a b area : ℝ) (h := (2 * area) / (a + b)) (ht : a = 15) (hb : b = 5) (ha : area = 400) :
  h = 40 :=
by
  sorry

end NUMINAMATH_GPT_depth_of_tunnel_l681_68147


namespace NUMINAMATH_GPT_john_burritos_left_l681_68180

def total_burritos (b1 b2 b3 b4 : ℕ) : ℕ :=
  b1 + b2 + b3 + b4

def burritos_left_after_giving_away (total : ℕ) (fraction : ℕ) : ℕ :=
  total - (total / fraction)

def burritos_left_after_eating (burritos_left : ℕ) (burritos_per_day : ℕ) (days : ℕ) : ℕ :=
  burritos_left - (burritos_per_day * days)

theorem john_burritos_left :
  let b1 := 15
  let b2 := 20
  let b3 := 25
  let b4 := 5
  let total := total_burritos b1 b2 b3 b4
  let burritos_after_give_away := burritos_left_after_giving_away total 3
  let burritos_after_eating := burritos_left_after_eating burritos_after_give_away 3 10
  burritos_after_eating = 14 :=
by
  sorry

end NUMINAMATH_GPT_john_burritos_left_l681_68180


namespace NUMINAMATH_GPT_fraction_of_pizza_covered_by_pepperoni_l681_68136

theorem fraction_of_pizza_covered_by_pepperoni :
  ∀ (d_pizza d_pepperoni : ℝ) (n_pepperoni : ℕ) (overlap_fraction : ℝ),
  d_pizza = 16 ∧ d_pepperoni = d_pizza / 8 ∧ n_pepperoni = 32 ∧ overlap_fraction = 0.25 →
  (π * d_pepperoni^2 / 4 * (1 - overlap_fraction) * n_pepperoni) / (π * (d_pizza / 2)^2) = 3 / 8 :=
by
  intro d_pizza d_pepperoni n_pepperoni overlap_fraction
  intro h
  sorry

end NUMINAMATH_GPT_fraction_of_pizza_covered_by_pepperoni_l681_68136


namespace NUMINAMATH_GPT_PJ_approx_10_81_l681_68164

noncomputable def PJ_length (P Q R J : Type) (PQ PR QR : ℝ) : ℝ :=
  if PQ = 30 ∧ PR = 29 ∧ QR = 27 then 10.81 else 0

theorem PJ_approx_10_81 (P Q R J : Type) (PQ PR QR : ℝ):
  PQ = 30 ∧ PR = 29 ∧ QR = 27 → PJ_length P Q R J PQ PR QR = 10.81 :=
by sorry

end NUMINAMATH_GPT_PJ_approx_10_81_l681_68164


namespace NUMINAMATH_GPT_tod_trip_time_l681_68101

noncomputable def total_time (d1 d2 d3 d4 s1 s2 s3 s4 : ℝ) : ℝ :=
  d1 / s1 + d2 / s2 + d3 / s3 + d4 / s4

theorem tod_trip_time :
  total_time 55 95 30 75 40 50 20 60 = 6.025 :=
by 
  sorry

end NUMINAMATH_GPT_tod_trip_time_l681_68101


namespace NUMINAMATH_GPT_tangent_from_origin_l681_68104

-- Define the points A, B, and C
def A : ℝ × ℝ := (4, 5)
def B : ℝ × ℝ := (7, 10)
def C : ℝ × ℝ := (6, 14)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a function that computes the length of the tangent from O to the circle passing through A, B, and C
noncomputable def tangent_length : ℝ :=
 sorry -- Placeholder for the actual calculation

-- The theorem we need to prove: The length of the tangent from O to the circle passing through A, B, and C is as calculated
theorem tangent_from_origin (L : ℕ) : 
  tangent_length = L := 
 sorry -- Placeholder for the proof

end NUMINAMATH_GPT_tangent_from_origin_l681_68104


namespace NUMINAMATH_GPT_find_second_candy_cost_l681_68125

theorem find_second_candy_cost :
  ∃ (x : ℝ), 
    (15 * 8 + 30 * x = 45 * 6) ∧
    x = 5 := by
  sorry

end NUMINAMATH_GPT_find_second_candy_cost_l681_68125


namespace NUMINAMATH_GPT_solve_for_t_l681_68105

theorem solve_for_t (t : ℝ) (ht : (t^2 - 3*t - 70) / (t - 10) = 7 / (t + 4)) : 
  t = -3 := sorry

end NUMINAMATH_GPT_solve_for_t_l681_68105


namespace NUMINAMATH_GPT_number_of_terms_in_sequence_l681_68133

theorem number_of_terms_in_sequence :
  ∃ n : ℕ, (1 + 4 * (n - 1) = 2025) ∧ n = 507 := by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_sequence_l681_68133


namespace NUMINAMATH_GPT_cylindrical_container_depth_l681_68170

theorem cylindrical_container_depth :
    ∀ (L D A : ℝ), 
      L = 12 ∧ D = 8 ∧ A = 48 → (∃ h : ℝ, h = 4 - 2 * Real.sqrt 3) :=
by
  intros L D A h_cond
  obtain ⟨hL, hD, hA⟩ := h_cond
  sorry

end NUMINAMATH_GPT_cylindrical_container_depth_l681_68170


namespace NUMINAMATH_GPT_find_n_l681_68141

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n + 4

-- Define the condition a_n = 13
def condition (n : ℕ) : Prop := a n = 13

-- Prove that under this condition, n = 3
theorem find_n (n : ℕ) (h : condition n) : n = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_n_l681_68141


namespace NUMINAMATH_GPT_negation_necessary_but_not_sufficient_l681_68145

def P (x : ℝ) : Prop := |x - 2| ≥ 1
def Q (x : ℝ) : Prop := x^2 - 3 * x + 2 ≥ 0

theorem negation_necessary_but_not_sufficient (x : ℝ) :
  (¬ P x → ¬ Q x) ∧ ¬ (¬ Q x → ¬ P x) :=
by
  sorry

end NUMINAMATH_GPT_negation_necessary_but_not_sufficient_l681_68145


namespace NUMINAMATH_GPT_brandy_used_0_17_pounds_of_chocolate_chips_l681_68160

def weight_of_peanuts : ℝ := 0.17
def weight_of_raisins : ℝ := 0.08
def total_weight_of_trail_mix : ℝ := 0.42

theorem brandy_used_0_17_pounds_of_chocolate_chips :
  total_weight_of_trail_mix - (weight_of_peanuts + weight_of_raisins) = 0.17 :=
by
  sorry

end NUMINAMATH_GPT_brandy_used_0_17_pounds_of_chocolate_chips_l681_68160


namespace NUMINAMATH_GPT_comparison_17_pow_14_31_pow_11_l681_68121

theorem comparison_17_pow_14_31_pow_11 : 17^14 > 31^11 :=
by
  sorry

end NUMINAMATH_GPT_comparison_17_pow_14_31_pow_11_l681_68121


namespace NUMINAMATH_GPT_shaded_area_square_l681_68142

theorem shaded_area_square (s : ℝ) (r : ℝ) (A : ℝ) :
  s = 4 ∧ r = 2 * Real.sqrt 2 → A = s^2 - 4 * (π * r^2 / 2) → A = 8 - 2 * π :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_shaded_area_square_l681_68142


namespace NUMINAMATH_GPT_complex_fraction_value_l681_68168

theorem complex_fraction_value :
  1 + (1 / (2 + (1 / (2 + 2)))) = 13 / 9 :=
by
  sorry

end NUMINAMATH_GPT_complex_fraction_value_l681_68168


namespace NUMINAMATH_GPT_solve_for_y_l681_68174

theorem solve_for_y (y : ℚ) : 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_y_l681_68174


namespace NUMINAMATH_GPT_michael_initial_money_l681_68197

theorem michael_initial_money (M : ℝ) 
  (half_give_away_to_brother : ∃ (m_half : ℝ), M / 2 = m_half)
  (brother_initial_money : ℝ := 17)
  (candy_cost : ℝ := 3)
  (brother_ends_up_with : ℝ := 35) :
  brother_initial_money + M / 2 - candy_cost = brother_ends_up_with ↔ M = 42 :=
sorry

end NUMINAMATH_GPT_michael_initial_money_l681_68197


namespace NUMINAMATH_GPT_sum_coefficients_l681_68102

theorem sum_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℚ) :
  (1 - 2 * (1 : ℚ))^5 = a_0 + a_1 * (1 : ℚ) + a_2 * (1 : ℚ)^2 + a_3 * (1 : ℚ)^3 + a_4 * (1 : ℚ)^4 + a_5 * (1 : ℚ)^5 →
  (1 - 2 * (0 : ℚ))^5 = a_0 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -2 :=
by
  sorry

end NUMINAMATH_GPT_sum_coefficients_l681_68102


namespace NUMINAMATH_GPT_sheila_picnic_probability_l681_68181

theorem sheila_picnic_probability :
  let P_rain := 0.5
  let P_go_given_rain := 0.3
  let P_go_given_sunny := 0.9
  let P_remember := 0.9  -- P(remember) = 1 - P(forget)
  let P_sunny := 1 - P_rain
  
  P_rain * P_go_given_rain * P_remember + P_sunny * P_go_given_sunny * P_remember = 0.54 :=
by
  sorry

end NUMINAMATH_GPT_sheila_picnic_probability_l681_68181


namespace NUMINAMATH_GPT_find_number_l681_68173

theorem find_number (x : ℝ) (h : (1/3) * x = 12) : x = 36 :=
sorry

end NUMINAMATH_GPT_find_number_l681_68173
