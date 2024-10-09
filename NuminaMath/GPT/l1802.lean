import Mathlib

namespace proof_problem_l1802_180211

noncomputable def problem_statement (a b c d : ℝ) : Prop :=
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (a + b = 2 * c) ∧ (a * b = -5 * d) ∧ (c + d = 2 * a) ∧ (c * d = -5 * b)

theorem proof_problem (a b c d : ℝ) (h : problem_statement a b c d) : a + b + c + d = 30 :=
by
  sorry

end proof_problem_l1802_180211


namespace correct_operations_l1802_180249

theorem correct_operations :
  (∀ {a b : ℝ}, -(-a + b) = a + b → False) ∧
  (∀ {a : ℝ}, 3 * a^3 - 3 * a^2 = a → False) ∧
  (∀ {x : ℝ}, (x^6)^2 = x^8 → False) ∧
  (∀ {z : ℝ}, 1 / (2 / 3 : ℝ)⁻¹ = 2 / 3) :=
by
  sorry

end correct_operations_l1802_180249


namespace stockholm_to_malmo_distance_l1802_180269
-- Import the necessary library

-- Define the parameters for the problem.
def map_distance : ℕ := 120 -- distance in cm
def scale_factor : ℕ := 12 -- km per cm

-- The hypothesis for the map distance and the scale factor
axiom map_distance_hyp : map_distance = 120
axiom scale_factor_hyp : scale_factor = 12

-- Define the real distance function
def real_distance (d : ℕ) (s : ℕ) : ℕ := d * s

-- The problem statement: Prove that the real distance between the two city centers is 1440 km
theorem stockholm_to_malmo_distance : real_distance map_distance scale_factor = 1440 :=
by
  rw [map_distance_hyp, scale_factor_hyp]
  sorry

end stockholm_to_malmo_distance_l1802_180269


namespace stock_comparison_l1802_180292

-- Quantities of the first year depreciation or growth rates
def initial_investment : ℝ := 200.0
def dd_first_year_growth : ℝ := 1.10
def ee_first_year_decline : ℝ := 0.85
def ff_first_year_growth : ℝ := 1.05

-- Quantities of the second year depreciation or growth rates
def dd_second_year_growth : ℝ := 1.05
def ee_second_year_growth : ℝ := 1.15
def ff_second_year_decline : ℝ := 0.90

-- Mathematical expression to determine final values after first year
def dd_after_first_year := initial_investment * dd_first_year_growth
def ee_after_first_year := initial_investment * ee_first_year_decline
def ff_after_first_year := initial_investment * ff_first_year_growth

-- Mathematical expression to determine final values after second year
def dd_final := dd_after_first_year * dd_second_year_growth
def ee_final := ee_after_first_year * ee_second_year_growth
def ff_final := ff_after_first_year * ff_second_year_decline

-- Theorem representing the final comparison
theorem stock_comparison : ff_final < ee_final ∧ ee_final < dd_final :=
by {
  -- Here we would provide the proof, but as per instruction we'll place sorry
  sorry
}

end stock_comparison_l1802_180292


namespace total_transport_cost_l1802_180214

def cost_per_kg : ℝ := 25000
def mass_sensor_g : ℝ := 350
def mass_communication_g : ℝ := 150

theorem total_transport_cost : 
  (cost_per_kg * (mass_sensor_g / 1000) + cost_per_kg * (mass_communication_g / 1000)) = 12500 :=
by
  sorry

end total_transport_cost_l1802_180214


namespace smallest_solution_x_abs_x_eq_3x_plus_2_l1802_180259

theorem smallest_solution_x_abs_x_eq_3x_plus_2 : ∃ x : ℝ, (x * abs x = 3 * x + 2) ∧ (∀ y : ℝ, (y * abs y = 3 * y + 2) → x ≤ y) ∧ x = -2 :=
by
  sorry

end smallest_solution_x_abs_x_eq_3x_plus_2_l1802_180259


namespace find_length_of_CE_l1802_180215

theorem find_length_of_CE
  (triangle_ABE_right : ∀ A B E : Type, ∃ (angle_AEB : Real), angle_AEB = 45)
  (triangle_BCE_right : ∀ B C E : Type, ∃ (angle_BEC : Real), angle_BEC = 45)
  (triangle_CDE_right : ∀ C D E : Type, ∃ (angle_CED : Real), angle_CED = 45)
  (AE_is_32 : 32 = 32) :
  ∃ (CE : ℝ), CE = 16 * Real.sqrt 2 :=
by
  sorry

end find_length_of_CE_l1802_180215


namespace coffee_customers_l1802_180242

theorem coffee_customers (C : ℕ) :
  let coffee_cost := 5
  let tea_ordered := 8
  let tea_cost := 4
  let total_revenue := 67
  (coffee_cost * C + tea_ordered * tea_cost = total_revenue) → C = 7 := by
  sorry

end coffee_customers_l1802_180242


namespace increase_by_percentage_l1802_180276

theorem increase_by_percentage (a b : ℝ) (percentage : ℝ) (final : ℝ) : b = a * percentage → final = a + b → final = 437.5 :=
by
  sorry

end increase_by_percentage_l1802_180276


namespace total_bins_l1802_180210

-- Definition of the problem conditions
def road_length : ℕ := 400
def placement_interval : ℕ := 20
def bins_per_side : ℕ := (road_length / placement_interval) - 1

-- Statement of the problem
theorem total_bins : 2 * bins_per_side = 38 := by
  sorry

end total_bins_l1802_180210


namespace solve_equation_l1802_180287

theorem solve_equation (x : ℝ) (h : ((x^2 + 3*x + 4) / (x + 5)) = x + 6) : x = -13 / 4 :=
by sorry

end solve_equation_l1802_180287


namespace chips_count_l1802_180239

theorem chips_count (B G P R x : ℕ) 
  (hx1 : 5 < x) (hx2 : x < 11) 
  (h : 1^B * 5^G * x^P * 11^R = 28160) : 
  P = 2 :=
by 
  -- Hint: Prime factorize 28160 to apply constraints and identify corresponding exponents.
  have prime_factorization_28160 : 28160 = 2^6 * 5^1 * 7^2 := by sorry
  -- Given 5 < x < 11 and by prime factorization, x can only be 7 (since it factors into the count of 7)
  -- Complete the rest of the proof
  sorry

end chips_count_l1802_180239


namespace k_value_l1802_180261

theorem k_value (k : ℝ) :
    (∀ r s : ℝ, (r + s = -k ∧ r * s = 9) ∧ ((r + 3) + (s + 3) = k)) → k = -3 :=
by
    intro h
    sorry

end k_value_l1802_180261


namespace decreasing_function_l1802_180274

noncomputable def f (a x : ℝ) : ℝ := a^(1 - x)

theorem decreasing_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : ∀ x > 1, f a x < 1) :
  ∀ x y : ℝ, x < y → f a x > f a y :=
sorry

end decreasing_function_l1802_180274


namespace last_two_digits_of_squared_expression_l1802_180225

theorem last_two_digits_of_squared_expression (n : ℕ) :
  (n * 2 * 3 * 4 * 46 * 47 * 48 * 49) ^ 2 % 100 = 76 :=
by
  sorry

end last_two_digits_of_squared_expression_l1802_180225


namespace product_of_two_larger_numbers_is_115_l1802_180280

noncomputable def proofProblem : Prop :=
  ∃ (A B C : ℝ), B = 10 ∧ (C - B = B - A) ∧ (A * B = 85) ∧ (B * C = 115)

theorem product_of_two_larger_numbers_is_115 : proofProblem :=
by
  sorry

end product_of_two_larger_numbers_is_115_l1802_180280


namespace ab_equals_six_l1802_180284

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l1802_180284


namespace least_even_integer_square_l1802_180295

theorem least_even_integer_square (E : ℕ) (h_even : E % 2 = 0) (h_square : ∃ (I : ℕ), 300 * E = I^2) : E = 6 ∧ ∃ (I : ℕ), I = 30 ∧ 300 * E = I^2 :=
sorry

end least_even_integer_square_l1802_180295


namespace power_vs_square_l1802_180222

theorem power_vs_square (n : ℕ) (h : n ≥ 4) : 2^n ≥ n^2 := by
  sorry

end power_vs_square_l1802_180222


namespace initial_number_of_men_l1802_180233

def initial_average_age_increased_by_2_years_when_two_women_replace_two_men 
    (M : ℕ) (A men1 men2 women1 women2 : ℕ) : Prop :=
  (men1 = 20) ∧ (men2 = 24) ∧ (women1 = 30) ∧ (women2 = 30) ∧
  ((M * A) + 16 = (M * (A + 2)))

theorem initial_number_of_men (M : ℕ) (A : ℕ) (men1 men2 women1 women2: ℕ):
  initial_average_age_increased_by_2_years_when_two_women_replace_two_men M A men1 men2 women1 women2 → 
  2 * M = 16 → M = 8 :=
by
  sorry

end initial_number_of_men_l1802_180233


namespace sally_rum_l1802_180297

theorem sally_rum (x : ℕ) (h₁ : 3 * x = x + 12 + 8) : x = 10 := by
  sorry

end sally_rum_l1802_180297


namespace age_difference_l1802_180299

variable (A J : ℕ)
variable (h1 : A + 5 = 40)
variable (h2 : J = 31)

theorem age_difference (h1 : A + 5 = 40) (h2 : J = 31) : A - J = 4 := by
  sorry

end age_difference_l1802_180299


namespace solve_linear_system_l1802_180248

theorem solve_linear_system (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  2 * x₁ + 2 * x₂ - x₃ + x₄ + 4 * x₆ = 0 ∧
  x₁ + 2 * x₂ + 2 * x₃ + 3 * x₅ + x₆ = -2 ∧
  x₁ - 2 * x₂ + x₄ + 2 * x₅ = 0 →
  x₁ = -1 / 4 - 5 / 8 * x₄ - 9 / 8 * x₅ - 9 / 8 * x₆ ∧
  x₂ = -1 / 8 + 3 / 16 * x₄ - 7 / 16 * x₅ + 9 / 16 * x₆ ∧
  x₃ = -3 / 4 + 1 / 8 * x₄ - 11 / 8 * x₅ + 5 / 8 * x₆ :=
by
  sorry

end solve_linear_system_l1802_180248


namespace common_ratio_of_arithmetic_sequence_l1802_180270

variable {α : Type} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_ratio_of_arithmetic_sequence (a : ℕ → α) (q : α)
  (h1 : is_arithmetic_sequence a)
  (h2 : ∀ n : ℕ, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
  (h3 : a 1 > 0)
  (h4 : ∀ n : ℕ, a n < a (n + 1)) :
  q = 2 := 
sorry

end common_ratio_of_arithmetic_sequence_l1802_180270


namespace x_and_y_complete_work_in_12_days_l1802_180290

noncomputable def work_rate_x : ℚ := 1 / 24
noncomputable def work_rate_y : ℚ := 1 / 24
noncomputable def combined_work_rate : ℚ := work_rate_x + work_rate_y

theorem x_and_y_complete_work_in_12_days : (1 / combined_work_rate) = 12 :=
by
  sorry

end x_and_y_complete_work_in_12_days_l1802_180290


namespace n_divisible_by_100_l1802_180272

theorem n_divisible_by_100 (n : ℤ) (h1 : n > 101) (h2 : 101 ∣ n)
  (h3 : ∀ d : ℤ, 1 < d ∧ d < n → d ∣ n → ∃ k m : ℤ, k ∣ n ∧ m ∣ n ∧ d = k - m) : 100 ∣ n :=
sorry

end n_divisible_by_100_l1802_180272


namespace tan_960_eq_sqrt_3_l1802_180234

theorem tan_960_eq_sqrt_3 : Real.tan (960 * Real.pi / 180) = Real.sqrt 3 := by
  sorry

end tan_960_eq_sqrt_3_l1802_180234


namespace speed_of_man_in_still_water_l1802_180264

variable (v_m v_s : ℝ)

-- Conditions as definitions 
def downstream_distance_eq : Prop :=
  36 = (v_m + v_s) * 3

def upstream_distance_eq : Prop :=
  18 = (v_m - v_s) * 3

theorem speed_of_man_in_still_water (h1 : downstream_distance_eq v_m v_s) (h2 : upstream_distance_eq v_m v_s) : v_m = 9 := 
  by
  sorry

end speed_of_man_in_still_water_l1802_180264


namespace ratio_of_perimeters_l1802_180213

theorem ratio_of_perimeters (s d s' d': ℝ) (h1 : d = s * Real.sqrt 2) (h2 : d' = 2.5 * d) (h3 : d' = s' * Real.sqrt 2) : (4 * s') / (4 * s) = 5 / 2 :=
by
  -- Additional tactical details for completion, proof is omitted as per instructions
  sorry

end ratio_of_perimeters_l1802_180213


namespace unique_solution_quadratic_l1802_180277

theorem unique_solution_quadratic (n : ℕ) : (∀ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intros h
  sorry

end unique_solution_quadratic_l1802_180277


namespace parallelogram_area_is_correct_l1802_180216

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨0, 2, 3⟩
def B : Point3D := ⟨2, 5, 2⟩
def C : Point3D := ⟨-2, 3, 6⟩

noncomputable def vectorAB (A B : Point3D) : Point3D :=
  { x := B.x - A.x
  , y := B.y - A.y
  , z := B.z - A.z 
  }

noncomputable def vectorAC (A C : Point3D) : Point3D :=
  { x := C.x - A.x
  , y := C.y - A.y
  , z := C.z - A.z 
  }

noncomputable def dotProduct (u v : Point3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

noncomputable def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

noncomputable def sinAngle (u v : Point3D) : ℝ :=
  Real.sqrt (1 - (dotProduct u v / (magnitude u * magnitude v)) ^ 2)

noncomputable def parallelogramArea (u v : Point3D) : ℝ :=
  magnitude u * magnitude v * sinAngle u v

theorem parallelogram_area_is_correct :
  parallelogramArea (vectorAB A B) (vectorAC A C) = 6 * Real.sqrt 5 := by
  sorry

end parallelogram_area_is_correct_l1802_180216


namespace initial_percentage_l1802_180232

variable (P : ℝ)

theorem initial_percentage (P : ℝ) 
  (h1 : 0 ≤ P ∧ P ≤ 100)
  (h2 : (7600 * (1 - P / 100) * 0.75) = 5130) :
  P = 10 :=
by
  sorry

end initial_percentage_l1802_180232


namespace alloy_copper_percentage_l1802_180221

theorem alloy_copper_percentage 
  (x : ℝ)
  (h1 : 0 ≤ x)
  (h2 : (30 / 100) * x + (70 / 100) * 27 = 24.9) :
  x = 20 :=
sorry

end alloy_copper_percentage_l1802_180221


namespace sound_speed_temperature_l1802_180226

theorem sound_speed_temperature (v : ℝ) (T : ℝ) (h1 : v = 0.4) (h2 : T = 15 * v^2) :
  T = 2.4 :=
by {
  sorry
}

end sound_speed_temperature_l1802_180226


namespace molecular_weight_BaO_is_correct_l1802_180227

-- Define the atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of BaO as the sum of atomic weights of Ba and O
def molecular_weight_BaO := atomic_weight_Ba + atomic_weight_O

-- Theorem stating the molecular weight of BaO
theorem molecular_weight_BaO_is_correct : molecular_weight_BaO = 153.33 := by
  -- Proof can be filled in
  sorry

end molecular_weight_BaO_is_correct_l1802_180227


namespace Tanya_bought_9_apples_l1802_180289

def original_fruit_count : ℕ := 18
def remaining_fruit_count : ℕ := 9
def pears_count : ℕ := 6
def pineapples_count : ℕ := 2
def plums_basket_count : ℕ := 1

theorem Tanya_bought_9_apples : 
  remaining_fruit_count * 2 = original_fruit_count →
  original_fruit_count - (pears_count + pineapples_count + plums_basket_count) = 9 :=
by
  intros h1
  sorry

end Tanya_bought_9_apples_l1802_180289


namespace light_intensity_at_10_m_l1802_180235

theorem light_intensity_at_10_m (k : ℝ) (d1 d2 : ℝ) (I1 I2 : ℝ)
  (h1: I1 = k / d1^2) (h2: I1 = 200) (h3: d1 = 5) (h4: d2 = 10) :
  I2 = k / d2^2 → I2 = 50 :=
sorry

end light_intensity_at_10_m_l1802_180235


namespace piglets_each_ate_6_straws_l1802_180244

theorem piglets_each_ate_6_straws (total_straws : ℕ) (fraction_for_adult_pigs : ℚ) (piglets : ℕ) 
  (h1 : total_straws = 300) 
  (h2 : fraction_for_adult_pigs = 3/5) 
  (h3 : piglets = 20) :
  (total_straws * (1 - fraction_for_adult_pigs) / piglets) = 6 :=
by
  sorry

end piglets_each_ate_6_straws_l1802_180244


namespace cream_ratio_l1802_180288

noncomputable def joe_coffee_initial := 14
noncomputable def joe_coffee_drank := 3
noncomputable def joe_cream_added := 3

noncomputable def joann_coffee_initial := 14
noncomputable def joann_cream_added := 3
noncomputable def joann_mixture_stirred := 17
noncomputable def joann_amount_drank := 3

theorem cream_ratio (joe_coffee_initial joe_coffee_drank joe_cream_added 
                     joann_coffee_initial joann_cream_added joann_mixture_stirred 
                     joann_amount_drank : ℝ) : 
  (joe_coffee_initial - joe_coffee_drank + joe_cream_added) / 
  (joann_cream_added - (joann_amount_drank * (joann_cream_added / joann_mixture_stirred))) = 17 / 14 :=
by
  -- Prove the theorem statement
  sorry

end cream_ratio_l1802_180288


namespace four_digit_number_l1802_180246

theorem four_digit_number (x : ℕ) (hx : 100 ≤ x ∧ x < 1000) (unit_digit : ℕ) (hu : unit_digit = 2) :
    (10 * x + unit_digit) - (2000 + x) = 108 → 10 * x + unit_digit = 2342 :=
by
  intros h
  sorry


end four_digit_number_l1802_180246


namespace supplementary_angle_difference_l1802_180263

theorem supplementary_angle_difference (a b : ℝ) (h1 : a + b = 180) (h2 : 5 * b = 3 * a) : abs (a - b) = 45 :=
  sorry

end supplementary_angle_difference_l1802_180263


namespace coins_problem_l1802_180247

theorem coins_problem : 
  ∃ x : ℕ, 
  (x % 8 = 6) ∧ 
  (x % 7 = 5) ∧ 
  (x % 9 = 1) ∧ 
  (x % 11 = 0) := 
by
  -- Proof to be provided here
  sorry

end coins_problem_l1802_180247


namespace angle_380_in_first_quadrant_l1802_180253

theorem angle_380_in_first_quadrant : ∃ n : ℤ, 380 - 360 * n = 20 ∧ 0 ≤ 20 ∧ 20 ≤ 90 :=
by
  use 1 -- We use 1 because 380 = 20 + 360 * 1
  sorry

end angle_380_in_first_quadrant_l1802_180253


namespace seven_digit_divisible_by_11_l1802_180231

theorem seven_digit_divisible_by_11 (m n : ℕ) (h1: 0 ≤ m ∧ m ≤ 9) (h2: 0 ≤ n ∧ n ≤ 9) (h3 : 10 + n - m ≡ 0 [MOD 11])  : m + n = 1 :=
by
  sorry

end seven_digit_divisible_by_11_l1802_180231


namespace rectangle_area_l1802_180209

/-- Define a rectangle with its length being three times its breadth, and given diagonal length d = 20.
    Prove that the area of the rectangle is 120 square meters. -/
theorem rectangle_area (b : ℝ) (l : ℝ) (d : ℝ) (h1 : l = 3 * b) (h2 : d = 20) (h3 : l^2 + b^2 = d^2) : l * b = 120 :=
by
  sorry

end rectangle_area_l1802_180209


namespace arithmetic_example_l1802_180206

theorem arithmetic_example : 3889 + 12.808 - 47.80600000000004 = 3854.002 := 
by
  sorry

end arithmetic_example_l1802_180206


namespace lunch_break_duration_l1802_180286

-- Definitions based on the conditions
variables (p h1 h2 L : ℝ)
-- Monday equation
def monday_eq : Prop := (9 - L/60) * (p + h1 + h2) = 0.55
-- Tuesday equation
def tuesday_eq : Prop := (7 - L/60) * (p + h2) = 0.35
-- Wednesday equation
def wednesday_eq : Prop := (5 - L/60) * (p + h1 + h2) = 0.25
-- Thursday equation
def thursday_eq : Prop := (4 - L/60) * p = 0.15

-- Combine all conditions
def all_conditions : Prop :=
  monday_eq p h1 h2 L ∧ tuesday_eq p h2 L ∧ wednesday_eq p h1 h2 L ∧ thursday_eq p L

-- Proof that the lunch break duration is 60 minutes
theorem lunch_break_duration : all_conditions p h1 h2 L → L = 60 :=
by
  sorry

end lunch_break_duration_l1802_180286


namespace middle_integer_of_consecutive_odd_l1802_180265

theorem middle_integer_of_consecutive_odd (n : ℕ)
  (h1 : n > 2)
  (h2 : n < 8)
  (h3 : (n-2) % 2 = 1)
  (h4 : n % 2 = 1)
  (h5 : (n+2) % 2 = 1)
  (h6 : (n-2) + n + (n+2) = (n-2) * n * (n+2) / 9) :
  n = 5 :=
by sorry

end middle_integer_of_consecutive_odd_l1802_180265


namespace find_third_card_value_l1802_180236

noncomputable def point_values (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 13 ∧
  1 ≤ b ∧ b ≤ 13 ∧
  1 ≤ c ∧ c ≤ 13 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b = 25 ∧
  b + c = 13

theorem find_third_card_value :
  ∃ a b c : ℕ, point_values a b c ∧ c = 1 :=
by {
  sorry
}

end find_third_card_value_l1802_180236


namespace find_geo_prog_numbers_l1802_180208

noncomputable def geo_prog_numbers (a1 a2 a3 : ℝ) : Prop :=
a1 * a2 * a3 = 27 ∧ a1 + a2 + a3 = 13

theorem find_geo_prog_numbers :
  geo_prog_numbers 1 3 9 ∨ geo_prog_numbers 9 3 1 :=
sorry

end find_geo_prog_numbers_l1802_180208


namespace fractional_part_of_water_after_replacements_l1802_180267

theorem fractional_part_of_water_after_replacements :
  let total_quarts := 25
  let removed_quarts := 5
  (1 - removed_quarts / (total_quarts : ℚ))^3 = 64 / 125 :=
by
  sorry

end fractional_part_of_water_after_replacements_l1802_180267


namespace abc_inequality_l1802_180279

-- Define a mathematical statement to encapsulate the problem
theorem abc_inequality (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by sorry

end abc_inequality_l1802_180279


namespace randy_gave_sally_l1802_180220

-- Define the given conditions
def initial_amount_randy : ℕ := 3000
def smith_contribution : ℕ := 200
def amount_kept_by_randy : ℕ := 2000

-- The total amount Randy had after Smith's contribution
def total_amount_randy : ℕ := initial_amount_randy + smith_contribution

-- The amount of money Randy gave to Sally
def amount_given_to_sally : ℕ := total_amount_randy - amount_kept_by_randy

-- The theorem statement: Given the conditions, prove that Randy gave Sally $1,200
theorem randy_gave_sally : amount_given_to_sally = 1200 :=
by
  sorry

end randy_gave_sally_l1802_180220


namespace least_possible_sum_l1802_180201

theorem least_possible_sum (x y z : ℕ) (h1 : 2 * x = 5 * y) (h2 : 5 * y = 6 * z) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x + y + z = 26 :=
by sorry

end least_possible_sum_l1802_180201


namespace julia_played_with_kids_on_Monday_l1802_180218

theorem julia_played_with_kids_on_Monday (k_wednesday : ℕ) (k_monday : ℕ)
  (h1 : k_wednesday = 4) (h2 : k_monday = k_wednesday + 2) : k_monday = 6 := 
by
  sorry

end julia_played_with_kids_on_Monday_l1802_180218


namespace index_card_area_l1802_180212

theorem index_card_area (length width : ℕ) (h_length : length = 5) (h_width : width = 7)
  (h_area_shortened_length : (length - 2) * width = 21) : (length * (width - 2)) = 25 := by
  sorry

end index_card_area_l1802_180212


namespace arun_weight_average_l1802_180223

theorem arun_weight_average :
  (∀ w : ℝ, 65 < w ∧ w < 72 → 60 < w ∧ w < 70 → w ≤ 68 → 66 ≤ w ∧ w ≤ 69 → 64 ≤ w ∧ w ≤ 67.5 → 
    (66.75 = (66 + 67.5) / 2)) := by
  sorry

end arun_weight_average_l1802_180223


namespace total_length_of_joined_papers_l1802_180245

theorem total_length_of_joined_papers :
  let length_each_sheet := 10 -- in cm
  let number_of_sheets := 20
  let overlap_length := 0.5 -- in cm
  let total_overlapping_connections := number_of_sheets - 1
  let total_length_without_overlap := length_each_sheet * number_of_sheets
  let total_overlap_length := overlap_length * total_overlapping_connections
  let total_length := total_length_without_overlap - total_overlap_length
  total_length = 190.5 :=
by {
    sorry
}

end total_length_of_joined_papers_l1802_180245


namespace t_shirt_cost_l1802_180203

theorem t_shirt_cost
  (marked_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (selling_price : ℝ)
  (cost : ℝ)
  (h1 : marked_price = 240)
  (h2 : discount_rate = 0.20)
  (h3 : profit_rate = 0.20)
  (h4 : selling_price = 0.8 * marked_price)
  (h5 : selling_price = cost + profit_rate * cost)
  : cost = 160 := 
sorry

end t_shirt_cost_l1802_180203


namespace eccentricity_of_ellipse_l1802_180256

theorem eccentricity_of_ellipse (p q : ℕ) (hp : Nat.Coprime p q) (z : ℂ) :
  ((z - 2) * (z^2 + 3 * z + 5) * (z^2 + 5 * z + 8) = 0) →
  (∃ p q : ℕ, Nat.Coprime p q ∧ (∃ e : ℝ, e^2 = p / q ∧ p + q = 16)) :=
by
  sorry

end eccentricity_of_ellipse_l1802_180256


namespace percent_increase_jordan_alex_l1802_180298

theorem percent_increase_jordan_alex :
  let pound_to_dollar := 1.5
  let alex_dollars := 600
  let jordan_pounds := 450
  let jordan_dollars := jordan_pounds * pound_to_dollar
  let percent_increase := ((jordan_dollars - alex_dollars) / alex_dollars) * 100
  percent_increase = 12.5 := 
by
  sorry

end percent_increase_jordan_alex_l1802_180298


namespace rectangle_perimeter_l1802_180229

theorem rectangle_perimeter (b : ℕ) (h1 : 3 * b * b = 192) : 2 * ((3 * b) + b) = 64 := 
by
  sorry

end rectangle_perimeter_l1802_180229


namespace bobbie_letters_to_remove_l1802_180291

-- Definitions of the conditions
def samanthaLastNameLength := 7
def bobbieLastNameLength := samanthaLastNameLength + 3
def jamieLastNameLength := 4
def targetBobbieLastNameLength := 2 * jamieLastNameLength

-- Question: How many letters does Bobbie need to take off to have a last name twice the length of Jamie's?
theorem bobbie_letters_to_remove : 
  bobbieLastNameLength - targetBobbieLastNameLength = 2 := by 
  sorry

end bobbie_letters_to_remove_l1802_180291


namespace negation_of_exists_abs_le_two_l1802_180282

theorem negation_of_exists_abs_le_two :
  (¬ ∃ x : ℝ, |x| ≤ 2) ↔ (∀ x : ℝ, |x| > 2) :=
by
  sorry

end negation_of_exists_abs_le_two_l1802_180282


namespace triangle_sides_are_6_8_10_l1802_180238

theorem triangle_sides_are_6_8_10 (a b c r r1 r2 r3 : ℕ) (hr_even : Even r) (hr1_even : Even r1) 
(hr2_even : Even r2) (hr3_even : Even r3) (relationship : r * r1 * r2 + r * r2 * r3 + r * r3 * r1 + r1 * r2 * r3 = r * r1 * r2 * r3) :
  (a, b, c) = (6, 8, 10) :=
sorry

end triangle_sides_are_6_8_10_l1802_180238


namespace odd_and_periodic_40_l1802_180275

noncomputable def f : ℝ → ℝ := sorry

theorem odd_and_periodic_40
  (h₁ : ∀ x : ℝ, f (10 + x) = f (10 - x))
  (h₂ : ∀ x : ℝ, f (20 - x) = -f (20 + x)) :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x : ℝ, f (x + 40) = f (x)) :=
by
  sorry

end odd_and_periodic_40_l1802_180275


namespace solution_set_proof_l1802_180260

theorem solution_set_proof {a b : ℝ} :
  (∀ x, 2 < x ∧ x < 3 → x^2 - a * x - b < 0) →
  (∀ x, bx^2 - a * x - 1 > 0) →
  (∀ x, -1 / 2 < x ∧ x < -1 / 3) :=
by
  sorry

end solution_set_proof_l1802_180260


namespace like_terms_value_l1802_180252

theorem like_terms_value (a b : ℤ) (h1 : a + b = 2) (h2 : a - 1 = 1) : a - b = 2 :=
sorry

end like_terms_value_l1802_180252


namespace intersection_A_B_union_B_Ac_range_a_l1802_180273

open Set

-- Conditions
def U : Set ℝ := univ
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def Ac : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}
def Bc : Set ℝ := {x | x < -2 ∨ x > 5}

-- Questions rewritten as Lean statements

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 5} := sorry

theorem union_B_Ac :
  B ∪ Ac = {x | x ≤ 5 ∨ x ≥ 9} := sorry

theorem range_a (a : ℝ) :
  {x | a ≤ x ∧ x ≤ a + 2} ⊆ Bc → a ∈ Iio (-4) ∪ Ioi 5 := sorry

end intersection_A_B_union_B_Ac_range_a_l1802_180273


namespace work_days_together_l1802_180266

theorem work_days_together (A_days B_days : ℕ) (work_left_fraction : ℚ) 
  (hA : A_days = 15) (hB : B_days = 20) (h_fraction : work_left_fraction = 8 / 15) : 
  ∃ d : ℕ, d * (1 / 15 + 1 / 20) = 1 - 8 / 15 ∧ d = 4 :=
by
  sorry

end work_days_together_l1802_180266


namespace angle_in_third_quadrant_l1802_180281

open Real

/--
Given that 2013° can be represented as 213° + 5 * 360° and that 213° is a third quadrant angle,
we can deduce that 2013° is also a third quadrant angle.
-/
theorem angle_in_third_quadrant (h1 : 2013 = 213 + 5 * 360) (h2 : 180 < 213 ∧ 213 < 270) : 
  (540 < 2013 % 360 ∧ 2013 % 360 < 270) :=
sorry

end angle_in_third_quadrant_l1802_180281


namespace letters_written_l1802_180262

theorem letters_written (nathan_rate : ℕ) (jacob_rate : ℕ) (combined_rate : ℕ) (hours : ℕ) :
  nathan_rate = 25 →
  jacob_rate = 2 * nathan_rate →
  combined_rate = nathan_rate + jacob_rate →
  hours = 10 →
  combined_rate * hours = 750 :=
by
  intros
  sorry

end letters_written_l1802_180262


namespace david_first_six_l1802_180224

def prob_six := (1:ℚ) / 6
def prob_not_six := (5:ℚ) / 6

def prob_david_first_six_cycle : ℚ :=
  prob_not_six * prob_not_six * prob_not_six * prob_six

def prob_no_six_cycle : ℚ :=
  prob_not_six ^ 4

def infinite_series_sum (a r: ℚ) : ℚ := 
  a / (1 - r)

theorem david_first_six :
  infinite_series_sum prob_david_first_six_cycle prob_no_six_cycle = 125 / 671 :=
by
  sorry

end david_first_six_l1802_180224


namespace geometric_series_first_term_l1802_180255

theorem geometric_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h_r : r = 1/4) (h_S : S = 80)
  (h_sum : S = a / (1 - r)) : a = 60 :=
by
  -- proof steps
  sorry

end geometric_series_first_term_l1802_180255


namespace min_f_eq_2_m_n_inequality_l1802_180294

def f (x : ℝ) := abs (x + 1) + abs (x - 1)

theorem min_f_eq_2 : (∀ x, f x ≥ 2) ∧ (∃ x, f x = 2) :=
by
  sorry

theorem m_n_inequality (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m^3 + n^3 = 2) : m + n ≤ 2 :=
by
  sorry

end min_f_eq_2_m_n_inequality_l1802_180294


namespace eccentricity_range_l1802_180230

noncomputable def ellipse_eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) : Prop :=
  ∃ c : ℝ, c^2 = a^2 - b^2 ∧ e = c / a ∧ (2 * ((-a) * (c + a / 2) - (b / 2) * b) + b^2 + c^2 ≥ 0)

theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ e : ℝ, ellipse_eccentricity_range a b h e ∧ (0 < e ∧ e ≤ -1 + Real.sqrt 3) :=
sorry

end eccentricity_range_l1802_180230


namespace factorization_correct_l1802_180219

-- Define the expression
def expression (x : ℝ) : ℝ := x^2 + 2 * x

-- State the theorem to prove the factorized form is equal to the expression
theorem factorization_correct (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by {
  -- Lean will skip the proof because of sorry, ensuring the statement compiles correctly.
  sorry
}

end factorization_correct_l1802_180219


namespace range_of_m_l1802_180241

theorem range_of_m (m : ℝ) (h1 : (m - 3) < 0) (h2 : (m + 1) > 0) : -1 < m ∧ m < 3 :=
by
  sorry

end range_of_m_l1802_180241


namespace reflect_across_y_axis_l1802_180251

-- Definition of the original point A
def pointA : ℝ × ℝ := (2, 3)

-- Definition of the reflected point across the y-axis
def reflectedPoint (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- The theorem stating the reflection result
theorem reflect_across_y_axis : reflectedPoint pointA = (-2, 3) :=
by
  -- Proof (skipped)
  sorry

end reflect_across_y_axis_l1802_180251


namespace simplify_expression_l1802_180205

theorem simplify_expression : (2 * 3 * b * 4 * (b ^ 2) * 5 * (b ^ 3) * 6 * (b ^ 4)) = 720 * (b ^ 10) :=
by
  sorry

end simplify_expression_l1802_180205


namespace intersecting_lines_triangle_area_l1802_180217

theorem intersecting_lines_triangle_area :
  let line1 := { p : ℝ × ℝ | p.2 = p.1 }
  let line2 := { p : ℝ × ℝ | p.1 = -6 }
  let intersection := (-6, -6)
  let base := 6
  let height := 6
  let area := (1 / 2 : ℝ) * base * height
  area = 18 := by
  sorry

end intersecting_lines_triangle_area_l1802_180217


namespace find_positive_integer_solutions_l1802_180240

-- Define the problem conditions
variable {x y z : ℕ}

-- Main theorem statement
theorem find_positive_integer_solutions 
  (h1 : Prime y)
  (h2 : ¬ 3 ∣ z)
  (h3 : ¬ y ∣ z)
  (h4 : x^3 - y^3 = z^2) : 
  x = 8 ∧ y = 7 ∧ z = 13 := 
sorry

end find_positive_integer_solutions_l1802_180240


namespace salaries_of_a_and_b_l1802_180204

theorem salaries_of_a_and_b {x y : ℝ}
  (h1 : x + y = 5000)
  (h2 : 0.05 * x = 0.15 * y) :
  x = 3750 :=
by sorry

end salaries_of_a_and_b_l1802_180204


namespace probability_of_picking_grain_buds_l1802_180200

theorem probability_of_picking_grain_buds :
  let num_stamps := 3
  let num_grain_buds := 1
  let probability := num_grain_buds / num_stamps
  probability = 1 / 3 :=
by
  sorry

end probability_of_picking_grain_buds_l1802_180200


namespace find_a_minus_b_l1802_180268

-- Definitions based on conditions
def eq1 (a b : Int) : Prop := 2 * b + a = 5
def eq2 (a b : Int) : Prop := a * b = -12

-- Statement of the problem
theorem find_a_minus_b (a b : Int) (h1 : eq1 a b) (h2 : eq2 a b) : a - b = -7 := 
sorry

end find_a_minus_b_l1802_180268


namespace age_ratio_7_9_l1802_180285

/-- Definition of Sachin and Rahul's ages -/
def sachin_age : ℝ := 24.5
def rahul_age : ℝ := sachin_age + 7

/-- The ratio of Sachin's age to Rahul's age is 7:9 -/
theorem age_ratio_7_9 : sachin_age / rahul_age = 7 / 9 := by
  sorry

end age_ratio_7_9_l1802_180285


namespace quadratic_trinomial_has_two_roots_l1802_180202

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  4 * (a^2 - a * b + b^2 - 3 * a * c) > 0 :=
by
  sorry

end quadratic_trinomial_has_two_roots_l1802_180202


namespace minimum_value_inequality_l1802_180271

noncomputable def min_value (a b : ℝ) (ha : 0 < a) (hb : 1 < b) (hab : a + b = 2) : ℝ :=
  (4 / a) + (1 / (b - 1))

theorem minimum_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 1 < b) (hab : a + b = 2) : 
  min_value a b ha hb hab ≥ 9 :=
  sorry

end minimum_value_inequality_l1802_180271


namespace evaluate_expression_l1802_180254

theorem evaluate_expression (a : ℝ) (h : 2 * a^2 - 3 * a - 5 = 0) : 4 * a^4 - 12 * a^3 + 9 * a^2 - 10 = 15 :=
by
  sorry

end evaluate_expression_l1802_180254


namespace christopher_strolling_time_l1802_180250

theorem christopher_strolling_time
  (initial_distance : ℝ) (initial_speed : ℝ) (break_time : ℝ)
  (continuation_distance : ℝ) (continuation_speed : ℝ)
  (H1 : initial_distance = 2) (H2 : initial_speed = 4)
  (H3 : break_time = 0.25) (H4 : continuation_distance = 3)
  (H5 : continuation_speed = 6) :
  (initial_distance / initial_speed + break_time + continuation_distance / continuation_speed) = 1.25 := 
  sorry

end christopher_strolling_time_l1802_180250


namespace domain_of_inverse_l1802_180237

noncomputable def f (x : ℝ) : ℝ := 3 ^ x

theorem domain_of_inverse (x : ℝ) : f x > 0 :=
by
  sorry

end domain_of_inverse_l1802_180237


namespace regular_tetrahedron_height_eq_4r_l1802_180293

noncomputable def equilateral_triangle_inscribed_circle_height (r : ℝ) : ℝ :=
3 * r

noncomputable def regular_tetrahedron_inscribed_sphere_height (r : ℝ) : ℝ :=
4 * r

theorem regular_tetrahedron_height_eq_4r (r : ℝ) :
  regular_tetrahedron_inscribed_sphere_height r = 4 * r :=
by
  unfold regular_tetrahedron_inscribed_sphere_height
  sorry

end regular_tetrahedron_height_eq_4r_l1802_180293


namespace equation1_solution_equation2_solution_l1802_180243

theorem equation1_solution (x : ℝ) : 4 * (2 * x - 1) ^ 2 = 36 ↔ x = 2 ∨ x = -1 :=
by sorry

theorem equation2_solution (x : ℝ) : (1 / 4) * (2 * x + 3) ^ 3 - 54 = 0 ↔ x = 3 / 2 :=
by sorry

end equation1_solution_equation2_solution_l1802_180243


namespace prime_half_sum_l1802_180296

theorem prime_half_sum
  (a b c : ℕ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : Nat.Prime (a.factorial + b + c))
  (h2 : Nat.Prime (b.factorial + c + a))
  (h3 : Nat.Prime (c.factorial + a + b)) :
  Nat.Prime ((a + b + c + 1) / 2) := 
sorry

end prime_half_sum_l1802_180296


namespace inequality_proof_l1802_180258

theorem inequality_proof {k l m n : ℕ} (h_pos_k : 0 < k) (h_pos_l : 0 < l) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_klmn : k < l ∧ l < m ∧ m < n)
  (h_equation : k * n = l * m) : 
  (n - k) / 2 ^ 2 ≥ k + 2 := 
by sorry

end inequality_proof_l1802_180258


namespace order_of_fractions_l1802_180228

theorem order_of_fractions (a b c d : ℚ)
  (h₁ : a = 21/14)
  (h₂ : b = 25/18)
  (h₃ : c = 23/16)
  (h₄ : d = 27/19)
  (h₅ : a > b)
  (h₆ : a > c)
  (h₇ : a > d)
  (h₈ : b < c)
  (h₉ : b < d)
  (h₁₀ : c > d) :
  b < d ∧ d < c ∧ c < a := 
sorry

end order_of_fractions_l1802_180228


namespace cos_neg_2theta_l1802_180257

theorem cos_neg_2theta (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : Real.cos (-2 * θ) = -7 / 25 := 
by
  sorry

end cos_neg_2theta_l1802_180257


namespace remainder_when_expr_divided_by_9_l1802_180283

theorem remainder_when_expr_divided_by_9 (n m p : ℤ)
  (h1 : n % 18 = 10)
  (h2 : m % 27 = 16)
  (h3 : p % 6 = 4) :
  (2 * n + 3 * m - p) % 9 = 1 := 
sorry

end remainder_when_expr_divided_by_9_l1802_180283


namespace total_cookies_baked_l1802_180207

-- Definitions based on conditions
def pans : ℕ := 5
def cookies_per_pan : ℕ := 8

-- Statement of the theorem to be proven
theorem total_cookies_baked :
  pans * cookies_per_pan = 40 := by
  sorry

end total_cookies_baked_l1802_180207


namespace find_b_of_square_binomial_l1802_180278

theorem find_b_of_square_binomial (b : ℚ) 
  (h : ∃ c : ℚ, ∀ x : ℚ, (3 * x + c) ^ 2 = 9 * x ^ 2 + 21 * x + b) : 
  b = 49 / 4 := 
sorry

end find_b_of_square_binomial_l1802_180278
