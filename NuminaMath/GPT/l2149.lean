import Mathlib

namespace NUMINAMATH_GPT_divisible_by_7_of_sum_of_squares_l2149_214936

theorem divisible_by_7_of_sum_of_squares (a b : ℤ) (h : 7 ∣ (a^2 + b^2)) : 
    (7 ∣ a) ∧ (7 ∣ b) :=
sorry

end NUMINAMATH_GPT_divisible_by_7_of_sum_of_squares_l2149_214936


namespace NUMINAMATH_GPT_multiply_large_numbers_l2149_214907

theorem multiply_large_numbers :
  72519 * 9999 = 724817481 :=
by
  sorry

end NUMINAMATH_GPT_multiply_large_numbers_l2149_214907


namespace NUMINAMATH_GPT_find_n_divisible_by_6_l2149_214975

theorem find_n_divisible_by_6 (n : Nat) : (71230 + n) % 6 = 0 ↔ n = 2 ∨ n = 8 := by
  sorry

end NUMINAMATH_GPT_find_n_divisible_by_6_l2149_214975


namespace NUMINAMATH_GPT_votes_cast_l2149_214914

theorem votes_cast (V : ℝ) (h1 : ∃ V, (0.65 * V) = (0.35 * V + 2340)) : V = 7800 :=
by
  sorry

end NUMINAMATH_GPT_votes_cast_l2149_214914


namespace NUMINAMATH_GPT_oxygen_mass_percentage_is_58_3_l2149_214925

noncomputable def C_molar_mass := 12.01
noncomputable def H_molar_mass := 1.01
noncomputable def O_molar_mass := 16.0

noncomputable def molar_mass_C6H8O7 :=
  6 * C_molar_mass + 8 * H_molar_mass + 7 * O_molar_mass

noncomputable def O_mass := 7 * O_molar_mass

noncomputable def oxygen_mass_percentage_C6H8O7 :=
  (O_mass / molar_mass_C6H8O7) * 100

theorem oxygen_mass_percentage_is_58_3 :
  oxygen_mass_percentage_C6H8O7 = 58.3 := by
  sorry

end NUMINAMATH_GPT_oxygen_mass_percentage_is_58_3_l2149_214925


namespace NUMINAMATH_GPT_percentage_of_number_l2149_214979

variable (N P : ℝ)

theorem percentage_of_number 
  (h₁ : (1 / 4) * (1 / 3) * (2 / 5) * N = 10) 
  (h₂ : (P / 100) * N = 120) : 
  P = 40 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_of_number_l2149_214979


namespace NUMINAMATH_GPT_last_digit_of_sum_1_to_5_last_digit_of_sum_1_to_2012_l2149_214943

theorem last_digit_of_sum_1_to_5 : 
  (1 ^ 2012 + 2 ^ 2012 + 3 ^ 2012 + 4 ^ 2012 + 5 ^ 2012) % 10 = 9 :=
  sorry

theorem last_digit_of_sum_1_to_2012 : 
  (List.sum (List.map (λ k => k ^ 2012) (List.range 2012).tail)) % 10 = 0 :=
  sorry

end NUMINAMATH_GPT_last_digit_of_sum_1_to_5_last_digit_of_sum_1_to_2012_l2149_214943


namespace NUMINAMATH_GPT_remainder_1234_5678_9012_div_5_l2149_214959

theorem remainder_1234_5678_9012_div_5 : (1234 * 5678 * 9012) % 5 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_1234_5678_9012_div_5_l2149_214959


namespace NUMINAMATH_GPT_fido_reachable_area_l2149_214948

theorem fido_reachable_area (r : ℝ) (a b : ℕ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0)
  (h_leash : ∃ (r : ℝ), r > 0) (h_fraction : (a : ℝ) / b * π = π) : a * b = 1 :=
by
  sorry

end NUMINAMATH_GPT_fido_reachable_area_l2149_214948


namespace NUMINAMATH_GPT_pentagon_coloring_count_l2149_214956

-- Define the three colors
inductive Color
| Red
| Yellow
| Green

open Color

-- Define the pentagon coloring problem
def adjacent_different (color1 color2 : Color) : Prop :=
color1 ≠ color2

-- Define a coloring for the pentagon
structure PentagonColoring :=
(A B C D E : Color)
(adjAB : adjacent_different A B)
(adjBC : adjacent_different B C)
(adjCD : adjacent_different C D)
(adjDE : adjacent_different D E)
(adjEA : adjacent_different E A)

-- The main statement to prove
theorem pentagon_coloring_count :
  ∃ (colorings : Finset PentagonColoring), colorings.card = 30 := sorry

end NUMINAMATH_GPT_pentagon_coloring_count_l2149_214956


namespace NUMINAMATH_GPT_range_of_independent_variable_l2149_214934

theorem range_of_independent_variable (x : ℝ) : (x - 4) ≠ 0 ↔ x ≠ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_independent_variable_l2149_214934


namespace NUMINAMATH_GPT_root_in_interval_l2149_214953

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 2

theorem root_in_interval : ∃ x ∈ Set.Ioo (3 : ℝ) (4 : ℝ), f x = 0 := sorry

end NUMINAMATH_GPT_root_in_interval_l2149_214953


namespace NUMINAMATH_GPT_marcus_brought_30_peanut_butter_cookies_l2149_214950

/-- Jenny brought in 40 peanut butter cookies. -/
def jenny_peanut_butter_cookies := 40

/-- Jenny brought in 50 chocolate chip cookies. -/
def jenny_chocolate_chip_cookies := 50

/-- Marcus brought in 20 lemon cookies. -/
def marcus_lemon_cookies := 20

/-- The total number of non-peanut butter cookies is the sum of chocolate chip and lemon cookies. -/
def non_peanut_butter_cookies := jenny_chocolate_chip_cookies + marcus_lemon_cookies

/-- The total number of peanut butter cookies is Jenny's plus Marcus'. -/
def total_peanut_butter_cookies (marcus_peanut_butter_cookies : ℕ) := jenny_peanut_butter_cookies + marcus_peanut_butter_cookies

/-- If Renee has a 50% chance of picking a peanut butter cookie, the number of peanut butter cookies must equal the number of non-peanut butter cookies. -/
theorem marcus_brought_30_peanut_butter_cookies (x : ℕ) : total_peanut_butter_cookies x = non_peanut_butter_cookies → x = 30 :=
by
  sorry

end NUMINAMATH_GPT_marcus_brought_30_peanut_butter_cookies_l2149_214950


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l2149_214955

theorem isosceles_triangle_base_length (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 5) (h₂ : a + b + c = 17) : c = 7 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l2149_214955


namespace NUMINAMATH_GPT_factorize_m_sq_minus_one_l2149_214964

theorem factorize_m_sq_minus_one (m : ℝ) : m^2 - 1 = (m + 1) * (m - 1) := 
by
  sorry

end NUMINAMATH_GPT_factorize_m_sq_minus_one_l2149_214964


namespace NUMINAMATH_GPT_ratio_sums_is_five_sixths_l2149_214931

theorem ratio_sums_is_five_sixths
  (a b c x y z : ℝ)
  (h_positive_a : a > 0) (h_positive_b : b > 0) (h_positive_c : c > 0)
  (h_positive_x : x > 0) (h_positive_y : y > 0) (h_positive_z : z > 0)
  (h₁ : a^2 + b^2 + c^2 = 25)
  (h₂ : x^2 + y^2 + z^2 = 36)
  (h₃ : a * x + b * y + c * z = 30) :
  (a + b + c) / (x + y + z) = (5 / 6) :=
sorry

end NUMINAMATH_GPT_ratio_sums_is_five_sixths_l2149_214931


namespace NUMINAMATH_GPT_LilyUsed14Dimes_l2149_214926

variable (p n d : ℕ)

theorem LilyUsed14Dimes
  (h1 : p + n + d = 50)
  (h2 : p + 5 * n + 10 * d = 200) :
  d = 14 := by
  sorry

end NUMINAMATH_GPT_LilyUsed14Dimes_l2149_214926


namespace NUMINAMATH_GPT_factorize_expression_l2149_214930

-- Define that a and b are arbitrary real numbers
variables (a b : ℝ)

-- The theorem statement claiming that 3a²b - 12b equals the factored form 3b(a + 2)(a - 2)
theorem factorize_expression : 3 * a^2 * b - 12 * b = 3 * b * (a + 2) * (a - 2) :=
by
  sorry  -- proof omitted

end NUMINAMATH_GPT_factorize_expression_l2149_214930


namespace NUMINAMATH_GPT_fresh_grapes_weight_eq_l2149_214990

-- Definitions of the conditions from a)
def fresh_grapes_water_percent : ℝ := 0.80
def dried_grapes_water_percent : ℝ := 0.20
def dried_grapes_weight : ℝ := 10
def fresh_grapes_non_water_percent : ℝ := 1 - fresh_grapes_water_percent
def dried_grapes_non_water_percent : ℝ := 1 - dried_grapes_water_percent

-- Proving the weight of fresh grapes
theorem fresh_grapes_weight_eq :
  let F := (dried_grapes_non_water_percent * dried_grapes_weight) / fresh_grapes_non_water_percent
  F = 40 := by
  -- The proof has been omitted
  sorry

end NUMINAMATH_GPT_fresh_grapes_weight_eq_l2149_214990


namespace NUMINAMATH_GPT_savings_after_four_weeks_l2149_214971

noncomputable def hourly_wage (name : String) : ℝ :=
  match name with
  | "Robby" | "Jaylen" | "Miranda" => 10
  | "Alex" => 12
  | "Beth" => 15
  | "Chris" => 20
  | _ => 0

noncomputable def daily_hours (name : String) : ℝ :=
  match name with
  | "Robby" | "Miranda" => 10
  | "Jaylen" => 8
  | "Alex" => 6
  | "Beth" => 4
  | "Chris" => 3
  | _ => 0

noncomputable def saving_rate (name : String) : ℝ :=
  match name with
  | "Robby" => 2/5
  | "Jaylen" => 3/5
  | "Miranda" => 1/2
  | "Alex" => 1/3
  | "Beth" => 1/4
  | "Chris" => 3/4
  | _ => 0

noncomputable def weekly_earning (name : String) : ℝ :=
  hourly_wage name * daily_hours name * 5

noncomputable def weekly_saving (name : String) : ℝ :=
  weekly_earning name * saving_rate name

noncomputable def combined_savings : ℝ :=
  4 * (weekly_saving "Robby" + 
       weekly_saving "Jaylen" + 
       weekly_saving "Miranda" + 
       weekly_saving "Alex" + 
       weekly_saving "Beth" + 
       weekly_saving "Chris")

theorem savings_after_four_weeks :
  combined_savings = 4440 :=
by
  sorry

end NUMINAMATH_GPT_savings_after_four_weeks_l2149_214971


namespace NUMINAMATH_GPT_minimize_cost_l2149_214938

-- Define the prices at each salon
def GustranSalonHaircut : ℕ := 45
def GustranSalonFacial : ℕ := 22
def GustranSalonNails : ℕ := 30

def BarbarasShopHaircut : ℕ := 30
def BarbarasShopFacial : ℕ := 28
def BarbarasShopNails : ℕ := 40

def FancySalonHaircut : ℕ := 34
def FancySalonFacial : ℕ := 30
def FancySalonNails : ℕ := 20

-- Define the total cost at each salon
def GustranSalonTotal : ℕ := GustranSalonHaircut + GustranSalonFacial + GustranSalonNails
def BarbarasShopTotal : ℕ := BarbarasShopHaircut + BarbarasShopFacial + BarbarasShopNails
def FancySalonTotal : ℕ := FancySalonHaircut + FancySalonFacial + FancySalonNails

-- Prove that the minimum total cost is $84
theorem minimize_cost : min GustranSalonTotal (min BarbarasShopTotal FancySalonTotal) = 84 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_minimize_cost_l2149_214938


namespace NUMINAMATH_GPT_time_to_meet_l2149_214927

-- Definitions based on conditions
def motorboat_speed_Serezha : ℝ := 20 -- km/h
def crossing_time_Serezha : ℝ := 0.5 -- hours (30 minutes)
def running_speed_Dima : ℝ := 6 -- km/h
def running_time_Dima : ℝ := 0.25 -- hours (15 minutes)
def combined_speed : ℝ := running_speed_Dima + running_speed_Dima -- equal speeds running towards each other
def distance_meet : ℝ := (running_speed_Dima * running_time_Dima) -- The distance they need to cover towards each other

-- Prove the time for them to meet
theorem time_to_meet : (distance_meet / combined_speed) = (7.5 / 60) :=
by
  sorry

end NUMINAMATH_GPT_time_to_meet_l2149_214927


namespace NUMINAMATH_GPT_middle_number_is_12_l2149_214935

theorem middle_number_is_12 (x y z : ℕ) (h1 : x + y = 20) (h2 : x + z = 25) (h3 : y + z = 29) (h4 : x < y) (h5 : y < z) : y = 12 :=
by
  sorry

end NUMINAMATH_GPT_middle_number_is_12_l2149_214935


namespace NUMINAMATH_GPT_number_composite_l2149_214942

theorem number_composite (n : ℕ) : 
  n = 10^(2^1974 + 2^1000 - 1) + 1 →
  ∃ a b : ℕ, 1 < a ∧ a < n ∧ n = a * b :=
by sorry

end NUMINAMATH_GPT_number_composite_l2149_214942


namespace NUMINAMATH_GPT_beetle_total_distance_l2149_214977

theorem beetle_total_distance (r : ℝ) (r_eq : r = 75) : (2 * r + r + r) = 300 := 
by
  sorry

end NUMINAMATH_GPT_beetle_total_distance_l2149_214977


namespace NUMINAMATH_GPT_ellipse_general_equation_l2149_214981

theorem ellipse_general_equation (x y : ℝ) (α : ℝ) (h1 : x = 5 * Real.cos α) (h2 : y = 3 * Real.sin α) :
  x^2 / 25 + y^2 / 9 = 1 :=
sorry

end NUMINAMATH_GPT_ellipse_general_equation_l2149_214981


namespace NUMINAMATH_GPT_angle_in_triangle_l2149_214989

theorem angle_in_triangle (A B C x : ℝ) (hA : A = 40)
    (hB : B = 3 * x) (hC : C = x) (h_sum : A + B + C = 180) : x = 35 :=
by
  sorry

end NUMINAMATH_GPT_angle_in_triangle_l2149_214989


namespace NUMINAMATH_GPT_original_number_of_employees_l2149_214991

theorem original_number_of_employees (E : ℝ) :
  (E - 0.125 * E) - 0.09 * (E - 0.125 * E) = 12385 → E = 15545 := 
by  -- Start the proof
  sorry  -- Placeholder for the proof, which is not required

end NUMINAMATH_GPT_original_number_of_employees_l2149_214991


namespace NUMINAMATH_GPT_gcd_result_is_two_l2149_214958

theorem gcd_result_is_two
  (n m k j: ℕ) (hn : n > 0) (hm : m > 0) (hk : k > 0) (hj : j > 0) :
  Nat.gcd (Nat.gcd (16 * n) (20 * m)) (Nat.gcd (18 * k) (24 * j)) = 2 := 
by
  sorry

end NUMINAMATH_GPT_gcd_result_is_two_l2149_214958


namespace NUMINAMATH_GPT_remainder_div_eq_4_l2149_214997

theorem remainder_div_eq_4 {x y : ℕ} (h1 : y = 25) (h2 : (x / y : ℝ) = 96.16) : x % y = 4 := 
sorry

end NUMINAMATH_GPT_remainder_div_eq_4_l2149_214997


namespace NUMINAMATH_GPT_total_distance_l2149_214996

theorem total_distance (x : ℝ) (h : (1/2) * (x - 1) = (1/3) * x + 1) : x = 9 := 
by 
  sorry

end NUMINAMATH_GPT_total_distance_l2149_214996


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2149_214954

theorem sufficient_not_necessary (a : ℝ) : (a > 1 → a^2 > 1) ∧ ¬(a^2 > 1 → a > 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_not_necessary_l2149_214954


namespace NUMINAMATH_GPT_find_principal_amount_l2149_214963

theorem find_principal_amount :
  ∃ P : ℝ, P * (1 + 0.05) ^ 4 = 9724.05 ∧ P = 8000 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_amount_l2149_214963


namespace NUMINAMATH_GPT_solve_part_one_solve_part_two_l2149_214901

-- Define function f
def f (a x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- Prove for part (1)
theorem solve_part_one : 
  {x : ℝ | -1 / 3 ≤ x ∧ x ≤ 5} = {x : ℝ | f 2 x ≤ 1} :=
by
  -- Replace the proof with sorry
  sorry

-- Prove for part (2)
theorem solve_part_two :
  {a : ℝ | a = 1 ∨ a = -1} = {a : ℝ | ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4} :=
by
  -- Replace the proof with sorry
  sorry

end NUMINAMATH_GPT_solve_part_one_solve_part_two_l2149_214901


namespace NUMINAMATH_GPT_number_of_real_solutions_l2149_214993

theorem number_of_real_solutions :
  (∃ (x : ℝ), (x ^ 2 + x - 12) ^ 2 = 81) →
  (∃! (x : ℝ), (x ^ 2 + x - 12) ^ 2 = 81) :=
sorry

end NUMINAMATH_GPT_number_of_real_solutions_l2149_214993


namespace NUMINAMATH_GPT_arithmetic_mean_25_41_50_l2149_214944

theorem arithmetic_mean_25_41_50 :
  (25 + 41 + 50) / 3 = 116 / 3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_25_41_50_l2149_214944


namespace NUMINAMATH_GPT_find_a_plus_b_l2149_214916

theorem find_a_plus_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a^2 - b^4 = 2009) : a + b = 47 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l2149_214916


namespace NUMINAMATH_GPT_area_of_inscribed_rectangle_l2149_214960

theorem area_of_inscribed_rectangle 
    (DA : ℝ) 
    (GD HD : ℝ) 
    (rectangle_inscribed : ∀ (A B C D G H : Type), true) 
    (radius : ℝ) 
    (GH : ℝ):
    DA = 20 ∧ GD = 5 ∧ HD = 5 ∧ GH = GD + DA + HD ∧ radius = GH / 2 → 
    200 * Real.sqrt 2 = DA * (Real.sqrt (radius^2 - (GD^2))) :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_rectangle_l2149_214960


namespace NUMINAMATH_GPT_driver_net_rate_of_pay_is_25_l2149_214986

noncomputable def net_rate_of_pay_per_hour (hours_traveled : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (pay_per_mile : ℝ) (fuel_cost_per_gallon : ℝ) : ℝ :=
  let total_distance := speed * hours_traveled
  let total_fuel_used := total_distance / fuel_efficiency
  let total_earnings := pay_per_mile * total_distance
  let total_fuel_cost := fuel_cost_per_gallon * total_fuel_used
  let net_earnings := total_earnings - total_fuel_cost
  net_earnings / hours_traveled

theorem driver_net_rate_of_pay_is_25 :
  net_rate_of_pay_per_hour 3 50 25 0.6 2.5 = 25 := sorry

end NUMINAMATH_GPT_driver_net_rate_of_pay_is_25_l2149_214986


namespace NUMINAMATH_GPT_sum_c_d_eq_24_l2149_214922

theorem sum_c_d_eq_24 (c d : ℕ) (h_pos_c : c > 0) (h_pos_d : d > 1) (h_max_power : c^d < 500 ∧ ∀ ⦃x y : ℕ⦄, x^y < 500 → x^y ≤ c^d) : c + d = 24 :=
sorry

end NUMINAMATH_GPT_sum_c_d_eq_24_l2149_214922


namespace NUMINAMATH_GPT_simplify_expr_l2149_214957

-- Define the terms
def a : ℕ := 2 ^ 10
def b : ℕ := 5 ^ 6

-- Define the expression we need to simplify
def expr := (a * b : ℝ)^(1/3)

-- Define the simplified form
def c : ℕ := 200
def d : ℕ := 2
def simplified_expr := (c : ℝ) * (d : ℝ)^(1/3)

-- The statement we need to prove
theorem simplify_expr : expr = simplified_expr ∧ (c + d = 202) := by
  sorry

end NUMINAMATH_GPT_simplify_expr_l2149_214957


namespace NUMINAMATH_GPT_updated_mean_corrected_l2149_214917

theorem updated_mean_corrected (mean observations decrement : ℕ) 
  (h1 : mean = 350) (h2 : observations = 100) (h3 : decrement = 63) :
  (mean * observations + decrement * observations) / observations = 413 :=
by
  sorry

end NUMINAMATH_GPT_updated_mean_corrected_l2149_214917


namespace NUMINAMATH_GPT_max_sum_abc_l2149_214984

theorem max_sum_abc (a b c : ℝ) (h1 : 1 ≤ a) (h2 : 1 ≤ b) (h3 : 1 ≤ c) 
  (h4 : a * b * c + 2 * a^2 + 2 * b^2 + 2 * c^2 + c * a - c * b - 4 * a + 4 * b - c = 28) :
  a + b + c ≤ 6 :=
sorry

end NUMINAMATH_GPT_max_sum_abc_l2149_214984


namespace NUMINAMATH_GPT_maximum_value_l2149_214995

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

theorem maximum_value : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ f 1 :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_maximum_value_l2149_214995


namespace NUMINAMATH_GPT_part1_part2_part3_l2149_214967

namespace Problem

-- Definitions and conditions for problem 1
def f (m x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + (m - 1)

theorem part1 (m : ℝ) :
  (∀ x : ℝ, f m x < 0) ↔ m < -5/3 := sorry

-- Definitions and conditions for problem 2
theorem part2 (m : ℝ) (h : m < 0) :
  ((-1 < m ∧ m < 0) → ∀ x : ℝ, x ≤ 1 ∨ x ≥ 1 / (m + 1)) ∧
  (m = -1 → ∀ x : ℝ, x ≤ 1) ∧
  (m < -1 → ∀ x : ℝ, 1 / (m + 1) ≤ x ∧ x ≤ 1) := sorry

-- Definitions and conditions for problem 3
theorem part3 (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f m x ≥ x^2 + 2 * x) ↔ m ≥ (2 * Real.sqrt 3) / 3 + 1 := sorry

end Problem

end NUMINAMATH_GPT_part1_part2_part3_l2149_214967


namespace NUMINAMATH_GPT_max_profit_at_35_l2149_214985

-- Define the conditions
def unit_purchase_price : ℝ := 20
def base_selling_price : ℝ := 30
def base_sales_volume : ℕ := 400
def price_increase_effect : ℝ := 1
def sales_volume_decrease_per_dollar : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - unit_purchase_price) * (base_sales_volume - sales_volume_decrease_per_dollar * (x - base_selling_price))

-- Lean statement to prove that the selling price which maximizes the profit is 35
theorem max_profit_at_35 : ∃ x : ℝ, x = 35 ∧ ∀ y : ℝ, profit y ≤ profit 35 := 
  sorry

end NUMINAMATH_GPT_max_profit_at_35_l2149_214985


namespace NUMINAMATH_GPT_percentage_volume_occupied_is_100_l2149_214994

-- Define the dimensions of the box and cube
def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_side : ℕ := 2

-- Define the volumes
def box_volume : ℕ := box_length * box_width * box_height
def cube_volume : ℕ := cube_side * cube_side * cube_side

-- Define the number of cubes that fit in each dimension
def cubes_along_length : ℕ := box_length / cube_side
def cubes_along_width : ℕ := box_width / cube_side
def cubes_along_height : ℕ := box_height / cube_side

-- Define the total number of cubes and the volume they occupy
def total_cubes : ℕ := cubes_along_length * cubes_along_width * cubes_along_height
def volume_occupied_by_cubes : ℕ := total_cubes * cube_volume

-- Define the percentage of the box volume occupied by the cubes
def percentage_volume_occupied : ℕ := (volume_occupied_by_cubes * 100) / box_volume

-- Statement to prove
theorem percentage_volume_occupied_is_100 : percentage_volume_occupied = 100 := by
  sorry

end NUMINAMATH_GPT_percentage_volume_occupied_is_100_l2149_214994


namespace NUMINAMATH_GPT_find_x_l2149_214992

theorem find_x (x y : ℤ) (some_number : ℤ) (h1 : y = 2) (h2 : some_number = 14) (h3 : 2 * x - y = some_number) : x = 8 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l2149_214992


namespace NUMINAMATH_GPT_slope_of_tangent_line_l2149_214962

theorem slope_of_tangent_line 
  (center point : ℝ × ℝ) 
  (h_center : center = (5, 3)) 
  (h_point : point = (8, 8)) 
  : (∃ m : ℚ, m = -3/5) :=
sorry

end NUMINAMATH_GPT_slope_of_tangent_line_l2149_214962


namespace NUMINAMATH_GPT_closest_cube_root_l2149_214904

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end NUMINAMATH_GPT_closest_cube_root_l2149_214904


namespace NUMINAMATH_GPT_zinc_in_combined_mass_l2149_214970

def mixture1_copper_zinc_ratio : ℕ × ℕ := (13, 7)
def mixture2_copper_zinc_ratio : ℕ × ℕ := (5, 3)
def mixture1_mass : ℝ := 100
def mixture2_mass : ℝ := 50

theorem zinc_in_combined_mass :
  let zinc1 := (mixture1_copper_zinc_ratio.2 : ℝ) / (mixture1_copper_zinc_ratio.1 + mixture1_copper_zinc_ratio.2) * mixture1_mass
  let zinc2 := (mixture2_copper_zinc_ratio.2 : ℝ) / (mixture2_copper_zinc_ratio.1 + mixture2_copper_zinc_ratio.2) * mixture2_mass
  zinc1 + zinc2 = 53.75 :=
by
  sorry

end NUMINAMATH_GPT_zinc_in_combined_mass_l2149_214970


namespace NUMINAMATH_GPT_original_profit_percentage_l2149_214915

-- Our definitions based on conditions.
variables (P S : ℝ)
-- Selling at double the price results in 260% profit
axiom h : (2 * S - P) / P * 100 = 260

-- Prove that the original profit percentage is 80%
theorem original_profit_percentage : (S - P) / P * 100 = 80 := 
sorry

end NUMINAMATH_GPT_original_profit_percentage_l2149_214915


namespace NUMINAMATH_GPT_completing_the_square_l2149_214911

theorem completing_the_square (x : ℝ) : 
  x^2 - 2 * x = 9 → (x - 1)^2 = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_completing_the_square_l2149_214911


namespace NUMINAMATH_GPT_smallest_fraction_numerator_l2149_214951

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (4 * b < 9 * a) ∧ 
  (∀ (a' b' : ℕ), a' ≥ 10 ∧ a' ≤ 99 ∧ b' ≥ 10 ∧ b' ≤ 99 ∧ (4 * b' < 9 * a') → b * a' ≥ a * b') ∧ a = 41 :=
sorry

end NUMINAMATH_GPT_smallest_fraction_numerator_l2149_214951


namespace NUMINAMATH_GPT_max_length_small_stick_l2149_214910

theorem max_length_small_stick (a b c : ℕ) 
  (ha : a = 24) (hb : b = 32) (hc : c = 44) :
  Nat.gcd (Nat.gcd a b) c = 4 :=
by
  rw [ha, hb, hc]
  -- At this point, the gcd calculus will be omitted, filing it with sorry
  sorry

end NUMINAMATH_GPT_max_length_small_stick_l2149_214910


namespace NUMINAMATH_GPT_area_of_rectangle_l2149_214912

theorem area_of_rectangle (a b : ℝ) (h1 : 2 * (a + b) = 16) (h2 : 2 * a^2 + 2 * b^2 = 68) :
  a * b = 15 :=
by
  have h3 : a + b = 8 := by sorry
  have h4 : a^2 + b^2 = 34 := by sorry
  have h5 : (a + b) ^ 2 = a^2 + b^2 + 2 * a * b := by sorry
  have h6 : 64 = 34 + 2 * a * b := by sorry
  have h7 : 2 * a * b = 30 := by sorry
  exact sorry

end NUMINAMATH_GPT_area_of_rectangle_l2149_214912


namespace NUMINAMATH_GPT_union_inter_example_l2149_214929

noncomputable def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
noncomputable def B : Set ℕ := {4, 7, 8, 9}

theorem union_inter_example :
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ (A ∩ B = {4, 7, 8}) :=
by
  sorry

end NUMINAMATH_GPT_union_inter_example_l2149_214929


namespace NUMINAMATH_GPT_betty_books_l2149_214908

variable (B : ℝ)
variable (h : B + (5/4) * B = 45)

theorem betty_books : B = 20 := by
  sorry

end NUMINAMATH_GPT_betty_books_l2149_214908


namespace NUMINAMATH_GPT_paul_needs_score_to_achieve_mean_l2149_214920

theorem paul_needs_score_to_achieve_mean (x : ℤ) :
  (78 + 84 + 76 + 82 + 88 + x) / 6 = 85 → x = 102 :=
by 
  sorry

end NUMINAMATH_GPT_paul_needs_score_to_achieve_mean_l2149_214920


namespace NUMINAMATH_GPT_motorboat_speed_l2149_214987

theorem motorboat_speed 
  (c : ℝ) (h_c : c = 2.28571428571)
  (t_up : ℝ) (h_t_up : t_up = 20 / 60)
  (t_down : ℝ) (h_t_down : t_down = 15 / 60) :
  ∃ v : ℝ, v = 16 :=
by
  sorry

end NUMINAMATH_GPT_motorboat_speed_l2149_214987


namespace NUMINAMATH_GPT_value_of_x_l2149_214946

theorem value_of_x (x : ℚ) (h : (3 * x + 4) / 7 = 15) : x = 101 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l2149_214946


namespace NUMINAMATH_GPT_intersection_points_on_circle_l2149_214973

theorem intersection_points_on_circle (u : ℝ) :
  ∃ (r : ℝ), ∀ (x y : ℝ), (u * x - 3 * y - 2 * u = 0) ∧ (2 * x - 3 * u * y + u = 0) → (x^2 + y^2 = r^2) :=
sorry

end NUMINAMATH_GPT_intersection_points_on_circle_l2149_214973


namespace NUMINAMATH_GPT_Julia_total_payment_l2149_214906

namespace CarRental

def daily_rate : ℝ := 30
def mileage_rate : ℝ := 0.25
def num_days : ℝ := 3
def num_miles : ℝ := 500

def daily_cost : ℝ := daily_rate * num_days
def mileage_cost : ℝ := mileage_rate * num_miles
def total_cost : ℝ := daily_cost + mileage_cost

theorem Julia_total_payment : total_cost = 215 := by
  sorry

end CarRental

end NUMINAMATH_GPT_Julia_total_payment_l2149_214906


namespace NUMINAMATH_GPT_alex_final_silver_tokens_l2149_214902

-- Define initial conditions
def initial_red_tokens := 100
def initial_blue_tokens := 50

-- Define exchange rules
def booth1_red_cost := 3
def booth1_silver_gain := 2
def booth1_blue_gain := 1

def booth2_blue_cost := 4
def booth2_silver_gain := 1
def booth2_red_gain := 2

-- Define limits where no further exchanges are possible
def red_token_limit := 2
def blue_token_limit := 3

-- Define the number of times visiting each booth
variable (x y : ℕ)

-- Tokens left after exchanges
def remaining_red_tokens := initial_red_tokens - 3 * x + 2 * y
def remaining_blue_tokens := initial_blue_tokens + x - 4 * y

-- Define proof theorem
theorem alex_final_silver_tokens :
  (remaining_red_tokens x y ≤ red_token_limit) ∧
  (remaining_blue_tokens x y ≤ blue_token_limit) →
  (2 * x + y = 113) :=
by
  sorry

end NUMINAMATH_GPT_alex_final_silver_tokens_l2149_214902


namespace NUMINAMATH_GPT_pet_store_cages_l2149_214988

-- Definitions and conditions
def initial_puppies : ℕ := 56
def sold_puppies : ℕ := 24
def puppies_per_cage : ℕ := 4
def remaining_puppies : ℕ := initial_puppies - sold_puppies
def cages_used : ℕ := remaining_puppies / puppies_per_cage

-- Theorem statement
theorem pet_store_cages : cages_used = 8 := by sorry

end NUMINAMATH_GPT_pet_store_cages_l2149_214988


namespace NUMINAMATH_GPT_sequence_is_arithmetic_sum_of_sequence_l2149_214924

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = 3 * a n + 2 * 3 ^ (n + 1)

def arithmetic_seq (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n, (a (n + 1) / 3 ^ (n + 1)) - (a n / 3 ^ n) = c

def sum_S (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = (n - 1) * 3 ^ (n + 1) + 3

theorem sequence_is_arithmetic (a : ℕ → ℕ)
  (h : sequence_a a) : 
  arithmetic_seq a 2 :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : sequence_a a) :
  sum_S a S :=
sorry

end NUMINAMATH_GPT_sequence_is_arithmetic_sum_of_sequence_l2149_214924


namespace NUMINAMATH_GPT_storyteller_friends_house_number_l2149_214972

theorem storyteller_friends_house_number
  (x y : ℕ)
  (htotal : 50 < x ∧ x < 500)
  (hsum : 2 * y = x * (x + 1)) :
  y = 204 :=
by
  sorry

end NUMINAMATH_GPT_storyteller_friends_house_number_l2149_214972


namespace NUMINAMATH_GPT_range_of_f_l2149_214913

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.cos x) + Real.arccos (Real.sin x)

theorem range_of_f : Set.range f = Set.Icc 0 Real.pi :=
sorry

end NUMINAMATH_GPT_range_of_f_l2149_214913


namespace NUMINAMATH_GPT_trivia_team_average_points_l2149_214947

noncomputable def average_points_per_member (total_members didn't_show_up total_points : ℝ) : ℝ :=
  total_points / (total_members - didn't_show_up)

@[simp]
theorem trivia_team_average_points :
  let total_members := 8.0
  let didn't_show_up := 3.5
  let total_points := 12.5
  ∃ avg_points, avg_points = 2.78 ∧ avg_points = average_points_per_member total_members didn't_show_up total_points :=
by
  sorry

end NUMINAMATH_GPT_trivia_team_average_points_l2149_214947


namespace NUMINAMATH_GPT_not_possible_to_partition_into_groups_of_5_with_remainder_3_l2149_214941

theorem not_possible_to_partition_into_groups_of_5_with_remainder_3 (m : ℤ) :
  ¬ (m^2 % 5 = 3) :=
by sorry

end NUMINAMATH_GPT_not_possible_to_partition_into_groups_of_5_with_remainder_3_l2149_214941


namespace NUMINAMATH_GPT_find_integer_pairs_l2149_214965

theorem find_integer_pairs (a b : ℤ) : 
  (∃ d : ℤ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → d ∣ (a^n + b^n + 1)) → 
  (∃ k₁ k₂ : ℤ, ((a = 2 * k₁) ∧ (b = 2 * k₂ + 1)) ∨ ((a = 3 * k₁ + 1) ∧ (b = 3 * k₂ + 1))) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l2149_214965


namespace NUMINAMATH_GPT_find_AD_l2149_214969

-- Given conditions as definitions
def AB := 5 -- given length in meters
def angle_ABC := 85 -- given angle in degrees
def angle_BCA := 45 -- given angle in degrees
def angle_DBC := 20 -- given angle in degrees

-- Lean theorem statement to prove the result
theorem find_AD : AD = AB := by
  -- The proof will be filled in afterwards; currently, we leave it as sorry.
  sorry

end NUMINAMATH_GPT_find_AD_l2149_214969


namespace NUMINAMATH_GPT_find_ratio_l2149_214968

theorem find_ratio (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + (a + 10 * b) / (b + 10 * a) = 2) : a / b = 0.8 :=
  sorry

end NUMINAMATH_GPT_find_ratio_l2149_214968


namespace NUMINAMATH_GPT_a5_eq_neg3_l2149_214966

-- Define arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sequence with given conditions
def a (n : ℕ) : ℤ :=
  if n = 2 then -5
  else if n = 8 then 1
  else sorry  -- Placeholder for other values

axiom a3_eq_neg5 : a 2 = -5
axiom a9_eq_1 : a 8 = 1
axiom a_is_arithmetic : is_arithmetic_sequence a

-- Statement to prove
theorem a5_eq_neg3 : a 4 = -3 :=
by
  sorry

end NUMINAMATH_GPT_a5_eq_neg3_l2149_214966


namespace NUMINAMATH_GPT_compute_fraction_l2149_214905

theorem compute_fraction (x y z : ℝ) (h : x + y + z = 1) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by 
  sorry

end NUMINAMATH_GPT_compute_fraction_l2149_214905


namespace NUMINAMATH_GPT_solve_for_x_l2149_214974

theorem solve_for_x (x : ℕ) : x * 12 = 173 * 240 → x = 3460 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2149_214974


namespace NUMINAMATH_GPT_speed_on_local_roads_l2149_214998

theorem speed_on_local_roads (v : ℝ) (h1 : 60 + 120 = 180) (h2 : (60 + 120) / (60 / v + 120 / 60) = 36) : v = 20 :=
by
  sorry

end NUMINAMATH_GPT_speed_on_local_roads_l2149_214998


namespace NUMINAMATH_GPT_tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m_l2149_214937

theorem tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m (m : ℝ) (h : Real.cos (80 * Real.pi / 180) = m) :
    Real.tan (-440 * Real.pi / 180) = - (Real.sqrt (1 - m^2) / m) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m_l2149_214937


namespace NUMINAMATH_GPT_triangle_inequality_condition_l2149_214919

theorem triangle_inequality_condition (a b : ℝ) (h : a + b = 1) (ha : a ≥ 0) (hb : b ≥ 0) :
    a + b > 1 → a + 1 > b ∧ b + 1 > a := by
  sorry

end NUMINAMATH_GPT_triangle_inequality_condition_l2149_214919


namespace NUMINAMATH_GPT_find_milk_ounces_l2149_214961

def bathroom_limit : ℕ := 32
def grape_juice_ounces : ℕ := 16
def water_ounces : ℕ := 8
def total_liquid_limit : ℕ := bathroom_limit
def total_liquid_intake : ℕ := grape_juice_ounces + water_ounces
def milk_ounces := total_liquid_limit - total_liquid_intake

theorem find_milk_ounces : milk_ounces = 8 := by
  sorry

end NUMINAMATH_GPT_find_milk_ounces_l2149_214961


namespace NUMINAMATH_GPT_cos_C_in_acute_triangle_l2149_214949

theorem cos_C_in_acute_triangle 
  (a b c : ℝ) (A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_sides_angles : a * Real.cos B = 4 * c * Real.sin C - b * Real.cos A) 
  : Real.cos C = Real.sqrt 15 / 4 := 
sorry

end NUMINAMATH_GPT_cos_C_in_acute_triangle_l2149_214949


namespace NUMINAMATH_GPT_solve_arcsin_arccos_l2149_214952

open Real

theorem solve_arcsin_arccos (x : ℝ) (h_condition : - (1 / 2 : ℝ) ≤ x ∧ x ≤ 1 / 2) :
  arcsin x + arcsin (2 * x) = arccos x ↔ x = 0 :=
sorry

end NUMINAMATH_GPT_solve_arcsin_arccos_l2149_214952


namespace NUMINAMATH_GPT_points_where_star_is_commutative_are_on_line_l2149_214900

def star (a b : ℝ) : ℝ := a * b * (a - b)

theorem points_where_star_is_commutative_are_on_line :
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1} = {p : ℝ × ℝ | p.1 = p.2} :=
by
  sorry

end NUMINAMATH_GPT_points_where_star_is_commutative_are_on_line_l2149_214900


namespace NUMINAMATH_GPT_multiple_of_7_l2149_214921

theorem multiple_of_7 :
  ∃ k : ℤ, 77 = 7 * k :=
sorry

end NUMINAMATH_GPT_multiple_of_7_l2149_214921


namespace NUMINAMATH_GPT_part_a_part_b_l2149_214982

noncomputable def withdraw_rubles_after_one_year
  (initial_deposit : ℤ) (initial_rate : ℤ) (annual_yield : ℚ)
  (final_rate : ℤ) (conversion_commission : ℚ) (broker_commission : ℚ) : ℚ :=
  let deposit_in_dollars := initial_deposit / initial_rate
  let interest_earned := deposit_in_dollars * annual_yield
  let total_in_dollars := deposit_in_dollars + interest_earned
  let broker_fee := interest_earned * broker_commission
  let amount_after_fee := total_in_dollars - broker_fee
  let total_in_rubles := amount_after_fee * final_rate
  let conversion_fee := total_in_rubles * conversion_commission
  total_in_rubles - conversion_fee

theorem part_a
  (initial_deposit : ℤ) (initial_rate : ℤ) (annual_yield : ℚ)
  (final_rate : ℤ) (conversion_commission : ℚ) (broker_commission : ℚ) :
  withdraw_rubles_after_one_year initial_deposit initial_rate annual_yield final_rate conversion_commission broker_commission =
  16476.8 := sorry

def effective_yield (initial_rubles final_rubles : ℚ) : ℚ :=
  (final_rubles / initial_rubles - 1) * 100

theorem part_b
  (initial_deposit : ℤ) (final_rubles : ℚ) :
  effective_yield initial_deposit final_rubles = 64.77 := sorry

end NUMINAMATH_GPT_part_a_part_b_l2149_214982


namespace NUMINAMATH_GPT_total_cost_of_supplies_l2149_214928

variable (E P M : ℝ)

open Real

theorem total_cost_of_supplies (h1 : E + 3 * P + 2 * M = 240)
                                (h2 : 2 * E + 4 * M + 5 * P = 440)
                                : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end NUMINAMATH_GPT_total_cost_of_supplies_l2149_214928


namespace NUMINAMATH_GPT_minimum_banks_needed_l2149_214932

-- Condition definitions
def total_amount : ℕ := 10000000
def max_insurance_payout_per_bank : ℕ := 1400000

-- Theorem statement
theorem minimum_banks_needed :
  ∃ n : ℕ, n * max_insurance_payout_per_bank ≥ total_amount ∧ n = 8 :=
sorry

end NUMINAMATH_GPT_minimum_banks_needed_l2149_214932


namespace NUMINAMATH_GPT_sandwiches_count_l2149_214933

def total_sandwiches : ℕ :=
  let meats := 12
  let cheeses := 8
  let condiments := 5
  meats * (Nat.choose cheeses 2) * condiments

theorem sandwiches_count : total_sandwiches = 1680 := by
  sorry

end NUMINAMATH_GPT_sandwiches_count_l2149_214933


namespace NUMINAMATH_GPT_g_29_eq_27_l2149_214923

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ x : ℝ, g (x + g x) = 3 * g x
axiom initial_condition : g 2 = 9

theorem g_29_eq_27 : g 29 = 27 := by
  sorry

end NUMINAMATH_GPT_g_29_eq_27_l2149_214923


namespace NUMINAMATH_GPT_circle_ratio_new_diameter_circumference_l2149_214939

theorem circle_ratio_new_diameter_circumference (r : ℝ) :
  let new_radius := r + 2
  let new_diameter := 2 * new_radius
  let new_circumference := 2 * Real.pi * new_radius
  new_circumference / new_diameter = Real.pi := 
by
  sorry

end NUMINAMATH_GPT_circle_ratio_new_diameter_circumference_l2149_214939


namespace NUMINAMATH_GPT_find_monotonic_intervals_max_min_on_interval_l2149_214980

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

noncomputable def f' (x : ℝ) : ℝ := (Real.cos x - Real.sin x) * Real.exp x - 1

theorem find_monotonic_intervals (k : ℤ) : 
  ((2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi) → 0 < (f' x)) ∧
  ((2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi) → (f' x) < 0) :=
sorry

theorem max_min_on_interval : 
  (∀ x, 0 ≤ x ∧ x ≤ (2 * Real.pi / 3) → f 0 = 1 ∧ f (2 * Real.pi / 3) =  -((1/2) * Real.exp (2/3 * Real.pi)) - (2 * Real.pi / 3)) :=
sorry

end NUMINAMATH_GPT_find_monotonic_intervals_max_min_on_interval_l2149_214980


namespace NUMINAMATH_GPT_sum_of_three_digit_positive_integers_l2149_214903

noncomputable def sum_of_arithmetic_series (a l n : ℕ) : ℕ :=
  (a + l) / 2 * n

theorem sum_of_three_digit_positive_integers : 
  sum_of_arithmetic_series 100 999 900 = 494550 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_sum_of_three_digit_positive_integers_l2149_214903


namespace NUMINAMATH_GPT_socks_ratio_l2149_214983

theorem socks_ratio 
  (g : ℕ) -- number of pairs of green socks
  (y : ℝ) -- price per pair of green socks
  (h1 : y > 0) -- price per pair of green socks is positive
  (h2 : 3 * g * y + 3 * y = 1.2 * (9 * y + g * y)) -- swapping resulted in a 20% increase in the bill
  : 3 / g = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_socks_ratio_l2149_214983


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l2149_214918

theorem roots_of_quadratic_eq {x1 x2 : ℝ} (h1 : x1 * x1 - 3 * x1 - 5 = 0) (h2 : x2 * x2 - 3 * x2 - 5 = 0) 
                              (h3 : x1 + x2 = 3) (h4 : x1 * x2 = -5) : x1^2 + x2^2 = 19 := 
sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l2149_214918


namespace NUMINAMATH_GPT_train_speed_is_72_l2149_214976

def distance : ℕ := 24
def time_minutes : ℕ := 20
def time_hours : ℚ := time_minutes / 60
def speed := distance / time_hours

theorem train_speed_is_72 :
  speed = 72 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_72_l2149_214976


namespace NUMINAMATH_GPT_binomial_probability_4_l2149_214909

noncomputable def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ := 
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem binomial_probability_4 (n : ℕ) (p : ℝ) (ξ : ℕ → ℝ)
  (H1 : (ξ 0) = (n*p))
  (H2 : (ξ 1) = (n*p*(1-p))) :
  binomial_pmf n 4 p = 10 / 243 :=
by {
  sorry 
}

end NUMINAMATH_GPT_binomial_probability_4_l2149_214909


namespace NUMINAMATH_GPT_max_g_equals_sqrt3_l2149_214940

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + Real.pi / 9) + Real.sin (5 * Real.pi / 9 - x)

noncomputable def g (x : ℝ) : ℝ :=
  f (f x)

theorem max_g_equals_sqrt3 : ∀ x, g x ≤ Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_max_g_equals_sqrt3_l2149_214940


namespace NUMINAMATH_GPT_value_of_expression_l2149_214978

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 9 - 4 * x^2 - 6 * x = 7 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2149_214978


namespace NUMINAMATH_GPT_probability_both_selected_l2149_214999

theorem probability_both_selected (p_ram : ℚ) (p_ravi : ℚ) (h_ram : p_ram = 5/7) (h_ravi : p_ravi = 1/5) : 
  (p_ram * p_ravi = 1/7) := 
by
  sorry

end NUMINAMATH_GPT_probability_both_selected_l2149_214999


namespace NUMINAMATH_GPT_vector_dot_product_correct_l2149_214945

-- Definitions of the vectors
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ :=
  let x := 4 - 2 * vector_a.1
  let y := 1 - 2 * vector_a.2
  (x, y)

-- Theorem to prove the dot product is correct
theorem vector_dot_product_correct :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = 4 := by
  sorry

end NUMINAMATH_GPT_vector_dot_product_correct_l2149_214945
