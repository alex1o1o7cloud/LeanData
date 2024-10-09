import Mathlib

namespace vector_parallel_sum_l1738_173867

theorem vector_parallel_sum (m n : ℝ) (a b : ℝ × ℝ × ℝ)
  (h_a : a = (2, -1, 3))
  (h_b : b = (4, m, n))
  (h_parallel : ∃ k : ℝ, a = k • b) :
  m + n = 4 :=
sorry

end vector_parallel_sum_l1738_173867


namespace caroline_citrus_drinks_l1738_173886

-- Definitions based on problem conditions
def citrus_drinks (oranges : ℕ) : ℕ := (oranges * 8) / 3

-- Define problem statement
theorem caroline_citrus_drinks : citrus_drinks 21 = 56 :=
by
  sorry

end caroline_citrus_drinks_l1738_173886


namespace percentage_to_decimal_l1738_173823

theorem percentage_to_decimal : (5 / 100 : ℚ) = 0.05 := by
  sorry

end percentage_to_decimal_l1738_173823


namespace floor_sqrt_20_squared_eq_16_l1738_173829

theorem floor_sqrt_20_squared_eq_16 : (Int.floor (Real.sqrt 20))^2 = 16 := by
  sorry

end floor_sqrt_20_squared_eq_16_l1738_173829


namespace output_increase_percentage_l1738_173843

theorem output_increase_percentage (O : ℝ) (P : ℝ) (h : (O * (1 + P / 100) * 1.60) * 0.5682 = O) : P = 10.09 :=
by 
  sorry

end output_increase_percentage_l1738_173843


namespace number_of_terms_in_sequence_l1738_173834

def arithmetic_sequence_terms (a d l : ℕ) : ℕ :=
  (l - a) / d + 1

theorem number_of_terms_in_sequence : arithmetic_sequence_terms 1 4 57 = 15 :=
by {
  sorry
}

end number_of_terms_in_sequence_l1738_173834


namespace sin_theta_correct_l1738_173872

noncomputable def sin_theta : ℝ :=
  let d := (4, 5, 7)
  let n := (3, -4, 5)
  let d_dot_n := 4 * 3 + 5 * (-4) + 7 * 5
  let norm_d := Real.sqrt (4^2 + 5^2 + 7^2)
  let norm_n := Real.sqrt (3^2 + (-4)^2 + 5^2)
  let cos_theta := d_dot_n / (norm_d * norm_n)
  cos_theta

theorem sin_theta_correct :
  sin_theta = 27 / Real.sqrt 4500 :=
by
  sorry

end sin_theta_correct_l1738_173872


namespace equation_of_curve_t_circle_through_fixed_point_l1738_173871

noncomputable def problem (x y : ℝ) : Prop :=
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (0, -1)
  let O : ℝ × ℝ := (0, 0)
  let M : ℝ × ℝ := (x, y)
  let N : ℝ × ℝ := (0, y)
  (x + 1) * (x - 1) + y * y = y * (y + 1)

noncomputable def curve_t_equation (x : ℝ) : ℝ :=
  x^2 - 1

theorem equation_of_curve_t (x y : ℝ) 
  (h : problem x y) :
  y = curve_t_equation x := 
sorry

noncomputable def passing_through_fixed_point (x y : ℝ) : Prop :=
  let y := x^2 - 1
  let y' := 2 * x
  let P : ℝ × ℝ := (x, y)
  let Q_x := (4 * x^2 - 1) / (8 * x)
  let Q : ℝ × ℝ := (Q_x, -5 / 4)
  let H : ℝ × ℝ := (0, -3 / 4)
  (x * Q_x + (-3 / 4 - y) * ( -3 / 4 + 5 / 4)) = 0

theorem circle_through_fixed_point (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y = curve_t_equation x)
  (h : passing_through_fixed_point x y) :
  ∃ t : ℝ, passing_through_fixed_point x t ∧ t = -3 / 4 :=
sorry

end equation_of_curve_t_circle_through_fixed_point_l1738_173871


namespace cannot_form_square_with_sticks_l1738_173863

theorem cannot_form_square_with_sticks
    (num_1cm_sticks : ℕ)
    (num_2cm_sticks : ℕ)
    (num_3cm_sticks : ℕ)
    (num_4cm_sticks : ℕ)
    (len_1cm_stick : ℕ)
    (len_2cm_stick : ℕ)
    (len_3cm_stick : ℕ)
    (len_4cm_stick : ℕ)
    (sum_lengths : ℕ) :
    num_1cm_sticks = 6 →
    num_2cm_sticks = 3 →
    num_3cm_sticks = 6 →
    num_4cm_sticks = 5 →
    len_1cm_stick = 1 →
    len_2cm_stick = 2 →
    len_3cm_stick = 3 →
    len_4cm_stick = 4 →
    sum_lengths = num_1cm_sticks * len_1cm_stick + 
                  num_2cm_sticks * len_2cm_stick + 
                  num_3cm_sticks * len_3cm_stick + 
                  num_4cm_sticks * len_4cm_stick →
    ∃ (s : ℕ), sum_lengths = 4 * s → False := 
by
  intros num_1cm_sticks_eq num_2cm_sticks_eq num_3cm_sticks_eq num_4cm_sticks_eq
         len_1cm_stick_eq len_2cm_stick_eq len_3cm_stick_eq len_4cm_stick_eq
         sum_lengths_def

  sorry

end cannot_form_square_with_sticks_l1738_173863


namespace greg_pages_per_day_l1738_173827

variable (greg_pages : ℕ)
variable (brad_pages : ℕ)

theorem greg_pages_per_day :
  brad_pages = 26 → brad_pages = greg_pages + 8 → greg_pages = 18 :=
by
  intros h1 h2
  rw [h1, add_comm] at h2
  linarith

end greg_pages_per_day_l1738_173827


namespace MattRate_l1738_173805

variable (M : ℝ) (t : ℝ)

def MattRateCondition : Prop := M * t = 220
def TomRateCondition : Prop := (M + 5) * t = 275

theorem MattRate (h1 : MattRateCondition M t) (h2 : TomRateCondition M t) : M = 20 := by
  sorry

end MattRate_l1738_173805


namespace evaluate_at_points_l1738_173858

noncomputable def f (x : ℝ) : ℝ :=
if x > 3 then x^2 - 3*x + 2
else if -2 ≤ x ∧ x ≤ 3 then -3*x + 5
else 9

theorem evaluate_at_points : f (-3) + f (0) + f (4) = 20 := by
  sorry

end evaluate_at_points_l1738_173858


namespace lim_sup_eq_Union_lim_inf_l1738_173869

open Set

theorem lim_sup_eq_Union_lim_inf
  (Ω : Type*)
  (A : ℕ → Set Ω) :
  (⋂ n, ⋃ k ≥ n, A k) = ⋃ (n_infty : ℕ → ℕ) (hn : StrictMono n_infty), ⋃ n, ⋂ k ≥ n, A (n_infty k) :=
by
  sorry

end lim_sup_eq_Union_lim_inf_l1738_173869


namespace missing_number_approximately_1400_l1738_173893

theorem missing_number_approximately_1400 :
  ∃ x : ℤ, x * 54 = 75625 ∧ abs (x - Int.ofNat (75625 / 54)) ≤ 1 :=
by
  sorry

end missing_number_approximately_1400_l1738_173893


namespace chives_planted_l1738_173828

theorem chives_planted (total_rows : ℕ) (plants_per_row : ℕ)
  (parsley_rows : ℕ) (rosemary_rows : ℕ) :
  total_rows = 20 →
  plants_per_row = 10 →
  parsley_rows = 3 →
  rosemary_rows = 2 →
  (plants_per_row * (total_rows - (parsley_rows + rosemary_rows))) = 150 :=
by
  intro h1 h2 h3 h4
  sorry

end chives_planted_l1738_173828


namespace trailing_zeroes_500_fact_l1738_173879

theorem trailing_zeroes_500_fact : 
  let count_multiples (n m : ℕ) := n / m 
  let count_5 := count_multiples 500 5
  let count_25 := count_multiples 500 25
  let count_125 := count_multiples 500 125
-- We don't count multiples of 625 because 625 > 500, thus its count is 0. 
-- Therefore: total trailing zeroes = count_5 + count_25 + count_125
  count_5 + count_25 + count_125 = 124 := sorry

end trailing_zeroes_500_fact_l1738_173879


namespace find_missing_number_l1738_173859

theorem find_missing_number
  (x : ℝ)
  (h1 : (12 + x + y + 78 + 104) / 5 = 62)
  (h2 : (128 + 255 + 511 + 1023 + x) / 5 = 398.2) : 
  y = 42 :=
  sorry

end find_missing_number_l1738_173859


namespace ned_price_per_game_l1738_173865

def number_of_games : Nat := 15
def non_working_games : Nat := 6
def total_earnings : Nat := 63
def number_of_working_games : Nat := number_of_games - non_working_games
def price_per_working_game : Nat := total_earnings / number_of_working_games

theorem ned_price_per_game : price_per_working_game = 7 :=
by
  sorry

end ned_price_per_game_l1738_173865


namespace minimum_value_inequality_l1738_173812

variable {x y z : ℝ}

theorem minimum_value_inequality (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x + y + z = 4) :
  (1 / x + 4 / y + 9 / z) ≥ 9 :=
sorry

end minimum_value_inequality_l1738_173812


namespace sharon_highway_speed_l1738_173897

theorem sharon_highway_speed:
  ∀ (total_distance : ℝ) (highway_time : ℝ) (city_time: ℝ) (city_speed : ℝ),
  total_distance = 59 → highway_time = 1 / 3 → city_time = 2 / 3 → city_speed = 45 →
  (total_distance - city_speed * city_time) / highway_time = 87 :=
by
  intro total_distance highway_time city_time city_speed
  intro h_total_distance h_highway_time h_city_time h_city_speed
  rw [h_total_distance, h_highway_time, h_city_time, h_city_speed]
  sorry

end sharon_highway_speed_l1738_173897


namespace min_value_of_Box_l1738_173898

theorem min_value_of_Box (c d : ℤ) (hcd : c * d = 42) (distinct_values : c ≠ d ∧ c ≠ 85 ∧ d ≠ 85) :
  ∃ (Box : ℤ), (c^2 + d^2 = Box) ∧ (Box = 85) :=
by
  sorry

end min_value_of_Box_l1738_173898


namespace minimum_value_y_l1738_173847

theorem minimum_value_y (x : ℝ) (h : x ≥ 1) : 5*x^2 - 8*x + 20 ≥ 13 :=
by {
  sorry
}

end minimum_value_y_l1738_173847


namespace triangle_angle_measure_l1738_173857

theorem triangle_angle_measure
  (D E F : ℝ)
  (hD : D = 70)
  (hE : E = 2 * F + 18)
  (h_sum : D + E + F = 180) :
  F = 92 / 3 :=
by
  sorry

end triangle_angle_measure_l1738_173857


namespace apple_price_33_kgs_l1738_173821

theorem apple_price_33_kgs (l q : ℕ) (h1 : 30 * l + 6 * q = 366) (h2 : 15 * l = 150) : 
  30 * l + 3 * q = 333 :=
by
  sorry

end apple_price_33_kgs_l1738_173821


namespace national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l1738_173800

-- Question 5
theorem national_currency_depreciation (term : String) : term = "Devaluation" := 
sorry

-- Question 6
theorem bond_annual_coupon_income 
  (purchase_price face_value annual_yield annual_coupon : ℝ) 
  (h_price : purchase_price = 900)
  (h_face : face_value = 1000)
  (h_yield : annual_yield = 0.15) 
  (h_coupon : annual_coupon = 135) : 
  annual_coupon = annual_yield * purchase_price := 
sorry

-- Question 7
theorem dividend_yield 
  (num_shares price_per_share total_dividends dividend_yield : ℝ)
  (h_shares : num_shares = 1000000)
  (h_price : price_per_share = 400)
  (h_dividends : total_dividends = 60000000)
  (h_yield : dividend_yield = 15) : 
  dividend_yield = (total_dividends / num_shares / price_per_share) * 100 :=
sorry

-- Question 8
theorem tax_deduction 
  (insurance_premium annual_salary tax_return : ℝ)
  (h_premium : insurance_premium = 120000)
  (h_salary : annual_salary = 110000)
  (h_return : tax_return = 14300) : 
  tax_return = 0.13 * min insurance_premium annual_salary := 
sorry

end national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l1738_173800


namespace megan_initial_strawberry_jelly_beans_l1738_173880

variables (s g : ℕ)

theorem megan_initial_strawberry_jelly_beans :
  (s = 3 * g) ∧ (s - 15 = 4 * (g - 15)) → s = 135 :=
by
  sorry

end megan_initial_strawberry_jelly_beans_l1738_173880


namespace monotonicity_and_extreme_values_l1738_173801

noncomputable def f (x : ℝ) : ℝ := Real.log x - x

theorem monotonicity_and_extreme_values :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x < f (1 - x)) ∧
  (∀ x : ℝ, x > 1 → f x < f 1) ∧
  f 1 = -1 :=
by 
  sorry

end monotonicity_and_extreme_values_l1738_173801


namespace total_cost_of_vitamins_l1738_173899

-- Definitions based on the conditions
def original_price : ℝ := 15.00
def discount_percentage : ℝ := 0.20
def coupon_value : ℝ := 2.00
def num_coupons : ℕ := 3
def num_bottles : ℕ := 3

-- Lean statement to prove the final cost
theorem total_cost_of_vitamins
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (coupon_value : ℝ)
  (num_coupons : ℕ)
  (num_bottles : ℕ)
  (discounted_price_per_bottle : ℝ := original_price * (1 - discount_percentage))
  (total_coupon_value : ℝ := coupon_value * num_coupons)
  (total_cost_before_coupons : ℝ := discounted_price_per_bottle * num_bottles) :
  (total_cost_before_coupons - total_coupon_value) = 30.00 :=
by
  sorry

end total_cost_of_vitamins_l1738_173899


namespace smallest_integer_y_l1738_173868

theorem smallest_integer_y (y : ℤ) (h : 3 - 5 * y < 23) : -3 ≥ y :=
by {
  sorry
}

end smallest_integer_y_l1738_173868


namespace adult_ticket_cost_l1738_173870

-- Definitions based on the conditions
def num_adults : ℕ := 10
def num_children : ℕ := 11
def total_bill : ℝ := 124
def child_ticket_cost : ℝ := 4

-- The proof which determines the cost of one adult ticket
theorem adult_ticket_cost : ∃ (A : ℝ), A * num_adults = total_bill - (num_children * child_ticket_cost) ∧ A = 8 := 
by
  sorry

end adult_ticket_cost_l1738_173870


namespace find_r_l1738_173850

theorem find_r (r s : ℝ) (h_quadratic : ∀ y, y^2 - r * y - s = 0) (h_r_pos : r > 0) 
    (h_root_diff : ∀ (y₁ y₂ : ℝ), (y₁ = (r + Real.sqrt (r^2 + 4 * s)) / 2 
        ∧ y₂ = (r - Real.sqrt (r^2 + 4 * s)) / 2) → |y₁ - y₂| = 2) : r = 2 :=
sorry

end find_r_l1738_173850


namespace simplify_expression_l1738_173860

noncomputable def givenExpression : ℝ := 
  abs (-0.01) ^ 2 - (-5 / 8) ^ 0 - 3 ^ (Real.log 2 / Real.log 3) + 
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 5) + Real.log 5

theorem simplify_expression : givenExpression = -1.9999 := by
  sorry

end simplify_expression_l1738_173860


namespace find_B_value_l1738_173848

-- Define the polynomial and conditions
def polynomial (A B : ℤ) (z : ℤ) : ℤ := z^4 - 12 * z^3 + A * z^2 + B * z + 36

-- Define roots and their properties according to the conditions
def roots_sum_to_twelve (r1 r2 r3 r4 : ℕ) : Prop := r1 + r2 + r3 + r4 = 12

-- The final statement to prove
theorem find_B_value (r1 r2 r3 r4 : ℕ) (A B : ℤ) (h_sum : roots_sum_to_twelve r1 r2 r3 r4)
    (h_pos : r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0) 
    (h_poly : polynomial A B = (z^4 - 12*z^3 + Az^2 + Bz + 36)) :
    B = -96 :=
    sorry

end find_B_value_l1738_173848


namespace polygon_area_leq_17_point_5_l1738_173837

theorem polygon_area_leq_17_point_5 (proj_OX proj_bisector_13 proj_OY proj_bisector_24 : ℝ)
  (h1: proj_OX = 4)
  (h2: proj_bisector_13 = 3 * Real.sqrt 2)
  (h3: proj_OY = 5)
  (h4: proj_bisector_24 = 4 * Real.sqrt 2)
  (S : ℝ) :
  S ≤ 17.5 := sorry

end polygon_area_leq_17_point_5_l1738_173837


namespace blithe_toy_count_l1738_173819

-- Define the initial number of toys, the number lost, and the number found.
def initial_toys := 40
def toys_lost := 6
def toys_found := 9

-- Define the total number of toys after the changes.
def total_toys_after_changes := initial_toys - toys_lost + toys_found

-- The proof statement.
theorem blithe_toy_count : total_toys_after_changes = 43 :=
by
  -- Placeholder for the proof
  sorry

end blithe_toy_count_l1738_173819


namespace income_final_amount_l1738_173803

noncomputable def final_amount (income : ℕ) : ℕ :=
  let children_distribution := (income * 45) / 100
  let wife_deposit := (income * 30) / 100
  let remaining_after_distribution := income - children_distribution - wife_deposit
  let donation := (remaining_after_distribution * 5) / 100
  remaining_after_distribution - donation

theorem income_final_amount : final_amount 200000 = 47500 := by
  -- Proof omitted
  sorry

end income_final_amount_l1738_173803


namespace vectors_parallel_eq_l1738_173891

-- Defining the problem
variables {m : ℝ}

-- Main statement
theorem vectors_parallel_eq (h : ∃ k : ℝ, (k ≠ 0) ∧ (k * 1 = m) ∧ (k * m = 2)) :
  m = Real.sqrt 2 ∨ m = -Real.sqrt 2 :=
sorry

end vectors_parallel_eq_l1738_173891


namespace cycling_race_difference_l1738_173820

-- Define the speeds and time
def s_Chloe : ℝ := 18
def s_David : ℝ := 15
def t : ℝ := 5

-- Define the distances based on the speeds and time
def d_Chloe : ℝ := s_Chloe * t
def d_David : ℝ := s_David * t
def distance_difference : ℝ := d_Chloe - d_David

-- The theorem to prove
theorem cycling_race_difference :
  distance_difference = 15 := by
  sorry

end cycling_race_difference_l1738_173820


namespace find_n_value_l1738_173809

theorem find_n_value (n : ℤ) : (5^3 - 7 = 6^2 + n) ↔ (n = 82) :=
by
  sorry

end find_n_value_l1738_173809


namespace platform_length_l1738_173854

theorem platform_length (speed_km_hr : ℝ) (time_man : ℝ) (time_platform : ℝ) (L : ℝ) (P : ℝ) :
  speed_km_hr = 54 → time_man = 20 → time_platform = 22 → 
  L = (speed_km_hr * (1000 / 3600)) * time_man →
  L + P = (speed_km_hr * (1000 / 3600)) * time_platform → 
  P = 30 := 
by
  intros hs ht1 ht2 hL hLP
  sorry

end platform_length_l1738_173854


namespace geometric_series_sum_l1738_173895

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℚ) / 4
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geometric_series_sum_l1738_173895


namespace kylie_coins_l1738_173813

open Nat

theorem kylie_coins :
  ∀ (coins_from_piggy_bank coins_from_brother coins_from_father coins_given_to_friend total_coins_left : ℕ),
  coins_from_piggy_bank = 15 →
  coins_from_brother = 13 →
  coins_from_father = 8 →
  coins_given_to_friend = 21 →
  total_coins_left = coins_from_piggy_bank + coins_from_brother + coins_from_father - coins_given_to_friend →
  total_coins_left = 15 :=
by
  intros
  sorry

end kylie_coins_l1738_173813


namespace find_speeds_of_A_and_B_l1738_173846

noncomputable def speed_A_and_B (x y : ℕ) : Prop :=
  30 * x - 30 * y = 300 ∧ 2 * x + 2 * y = 300

theorem find_speeds_of_A_and_B : ∃ (x y : ℕ), speed_A_and_B x y ∧ x = 80 ∧ y = 70 :=
by
  sorry

end find_speeds_of_A_and_B_l1738_173846


namespace find_constants_l1738_173818

noncomputable def f (a b x : ℝ) : ℝ :=
(a * x + b) / (x + 1)

theorem find_constants (a b : ℝ) (x : ℝ) (h : x ≠ -1) : 
  (f a b (f a b x) = x) → (a = -1 ∧ ∀ b, ∃ c : ℝ, b = c) :=
by 
  sorry

end find_constants_l1738_173818


namespace total_legs_on_farm_l1738_173896

-- Define the number of each type of animal
def num_ducks : Nat := 6
def num_dogs : Nat := 5
def num_spiders : Nat := 3
def num_three_legged_dogs : Nat := 1

-- Define the number of legs for each type of animal
def legs_per_duck : Nat := 2
def legs_per_dog : Nat := 4
def legs_per_spider : Nat := 8
def legs_per_three_legged_dog : Nat := 3

-- Calculate the total number of legs
def total_duck_legs : Nat := num_ducks * legs_per_duck
def total_dog_legs : Nat := (num_dogs * legs_per_dog) - (num_three_legged_dogs * (legs_per_dog - legs_per_three_legged_dog))
def total_spider_legs : Nat := num_spiders * legs_per_spider

-- The total number of legs on the farm
def total_animal_legs : Nat := total_duck_legs + total_dog_legs + total_spider_legs

-- State the theorem to be proved
theorem total_legs_on_farm : total_animal_legs = 55 :=
by
  -- Assuming conditions and computing as per them
  sorry

end total_legs_on_farm_l1738_173896


namespace potato_yield_l1738_173875

/-- Mr. Green's gardening problem -/
theorem potato_yield
  (steps_length : ℝ)
  (steps_width : ℝ)
  (step_size : ℝ)
  (yield_rate : ℝ)
  (feet_length := steps_length * step_size)
  (feet_width := steps_width * step_size)
  (area := feet_length * feet_width)
  (yield := area * yield_rate) :
  steps_length = 18 →
  steps_width = 25 →
  step_size = 2.5 →
  yield_rate = 0.75 →
  yield = 2109.375 :=
by
  sorry

end potato_yield_l1738_173875


namespace sum_five_smallest_primes_l1738_173830

theorem sum_five_smallest_primes : (2 + 3 + 5 + 7 + 11) = 28 := by
  -- We state the sum of the known five smallest prime numbers.
  sorry

end sum_five_smallest_primes_l1738_173830


namespace mom_buys_tshirts_l1738_173808

theorem mom_buys_tshirts 
  (tshirts_per_package : ℕ := 3) 
  (num_packages : ℕ := 17) :
  tshirts_per_package * num_packages = 51 :=
by
  sorry

end mom_buys_tshirts_l1738_173808


namespace lines_intersect_at_point_l1738_173838

def ParametricLine1 (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 4 - 3 * t)

def ParametricLine2 (u : ℝ) : ℝ × ℝ :=
  (-2 + 3 * u, 5 - u)

theorem lines_intersect_at_point :
  ∃ t u : ℝ, ParametricLine1 t = ParametricLine2 u ∧ ParametricLine1 t = (-5, 13) :=
by
  sorry

end lines_intersect_at_point_l1738_173838


namespace general_term_formula_l1738_173853

/-- Define that the point (n, S_n) lies on the function y = 2x^2 + x, hence S_n = 2 * n^2 + n --/
def S_n (n : ℕ) : ℕ := 2 * n^2 + n

/-- Define the nth term of the sequence a_n --/
def a_n (n : ℕ) : ℕ := if n = 0 then 0 else 4 * n - 1

theorem general_term_formula (n : ℕ) (hn : 0 < n) :
  a_n n = S_n n - S_n (n - 1) :=
by
  sorry

end general_term_formula_l1738_173853


namespace billboard_shorter_side_length_l1738_173835

theorem billboard_shorter_side_length
  (L W : ℝ)
  (h1 : L * W = 120)
  (h2 : 2 * L + 2 * W = 46) :
  min L W = 8 :=
by
  sorry

end billboard_shorter_side_length_l1738_173835


namespace exists_x_nat_l1738_173840

theorem exists_x_nat (a c : ℕ) (b : ℤ) : ∃ x : ℕ, (a^x + x) % c = b % c :=
by
  sorry

end exists_x_nat_l1738_173840


namespace red_balls_count_l1738_173864

-- Lean 4 statement for proving the number of red balls in the bag is 336
theorem red_balls_count (x : ℕ) (total_balls red_balls : ℕ) 
  (h1 : total_balls = 60 + 18 * x) 
  (h2 : red_balls = 56 + 14 * x) 
  (h3 : (56 + 14 * x : ℚ) / (60 + 18 * x) = 4 / 5) : red_balls = 336 := 
by
  sorry

end red_balls_count_l1738_173864


namespace smallest_base_l1738_173874

-- Definitions of the conditions
def condition1 (b : ℕ) : Prop := b > 3
def condition2 (b : ℕ) : Prop := b > 7
def condition3 (b : ℕ) : Prop := b > 6
def condition4 (b : ℕ) : Prop := b > 8

-- Main theorem statement
theorem smallest_base : ∀ b : ℕ, condition1 b ∧ condition2 b ∧ condition3 b ∧ condition4 b → b = 9 := by
  sorry

end smallest_base_l1738_173874


namespace cyclist_speed_l1738_173836

theorem cyclist_speed (v : ℝ) (h : 0.7142857142857143 * (30 + v) = 50) : v = 40 :=
by
  sorry

end cyclist_speed_l1738_173836


namespace problem_l1738_173811

noncomputable def roots1 : Set ℝ := { α | α^2 - 2*α + 1 = 0 }
noncomputable def roots2 : Set ℝ := { γ | γ^2 - 3*γ + 1 = 0 }

theorem problem 
  (α β γ δ : ℝ) 
  (hαβ : α ∈ roots1 ∧ β ∈ roots1)
  (hγδ : γ ∈ roots2 ∧ δ ∈ roots2) : 
  (α - γ)^2 * (β - δ)^2 = 1 := 
sorry

end problem_l1738_173811


namespace find_a_plus_b_minus_c_l1738_173802

theorem find_a_plus_b_minus_c (a b c : ℤ) (h1 : 3 * b = 5 * a) (h2 : 7 * a = 3 * c) (h3 : 3 * a + 2 * b - 4 * c = -9) : a + b - c = 1 :=
by
  sorry

end find_a_plus_b_minus_c_l1738_173802


namespace find_x_values_for_inverse_l1738_173807

def f (x : ℝ) : ℝ := x^2 - 3 * x - 4

theorem find_x_values_for_inverse :
  ∃ (x : ℝ), (f x = 2 + 2 * Real.sqrt 2 ∨ f x = 2 - 2 * Real.sqrt 2) ∧ f x = x :=
sorry

end find_x_values_for_inverse_l1738_173807


namespace sqrt_x_minus_2_meaningful_l1738_173890

theorem sqrt_x_minus_2_meaningful (x : ℝ) (hx : x = 0 ∨ x = -1 ∨ x = -2 ∨ x = 2) : (x = 2) ↔ (x - 2 ≥ 0) :=
by
  sorry

end sqrt_x_minus_2_meaningful_l1738_173890


namespace anika_more_than_twice_reeta_l1738_173822

theorem anika_more_than_twice_reeta (R A M : ℕ) (h1 : R = 20) (h2 : A + R = 64) (h3 : A = 2 * R + M) : M = 4 :=
by
  sorry

end anika_more_than_twice_reeta_l1738_173822


namespace gcd_72_and_120_l1738_173849

theorem gcd_72_and_120 : Nat.gcd 72 120 = 24 := 
by
  sorry

end gcd_72_and_120_l1738_173849


namespace inequality_general_l1738_173881

theorem inequality_general {a b c d : ℝ} :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 :=
by
  sorry

end inequality_general_l1738_173881


namespace quadratic_function_coefficient_not_zero_l1738_173884

theorem quadratic_function_coefficient_not_zero (m : ℝ) : (∀ x : ℝ, (m-2)*x^2 + 2*x - 3 ≠ 0) → m ≠ 2 :=
by
  intro h
  by_contra h1
  exact sorry

end quadratic_function_coefficient_not_zero_l1738_173884


namespace range_of_a_for_inequality_l1738_173841

theorem range_of_a_for_inequality (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_for_inequality_l1738_173841


namespace increasing_arithmetic_sequence_l1738_173866

theorem increasing_arithmetic_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) = a n + 2) : ∀ n : ℕ, a (n + 1) > a n :=
by
  sorry

end increasing_arithmetic_sequence_l1738_173866


namespace andrea_rhinestones_needed_l1738_173888

theorem andrea_rhinestones_needed (total_needed bought_ratio found_ratio : ℝ) 
  (h1 : total_needed = 45) 
  (h2 : bought_ratio = 1 / 3) 
  (h3 : found_ratio = 1 / 5) : 
  total_needed - (bought_ratio * total_needed + found_ratio * total_needed) = 21 := 
by 
  sorry

end andrea_rhinestones_needed_l1738_173888


namespace simplify_expression_l1738_173817

variable (x y : ℝ)

-- Define the proposition
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y := 
by
  sorry

end simplify_expression_l1738_173817


namespace intersection_A_complement_B_l1738_173855

-- Definitions of sets A and B and their complement in the universal set R, which is the real numbers.
def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2 * x > 0}
def complement_R_B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- The proof statement verifying the intersection of set A with the complement of set B.
theorem intersection_A_complement_B : A ∩ complement_R_B = {0, 1, 2} := by
  sorry

end intersection_A_complement_B_l1738_173855


namespace find_a_l1738_173806

theorem find_a (a x : ℝ) (h1: a - 2 ≤ x) (h2: x ≤ a + 1) (h3 : -x^2 + 2 * x + 3 = 3) :
  a = 2 := sorry

end find_a_l1738_173806


namespace necessary_but_not_sufficient_condition_l1738_173861

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ^ 2 = a n * a (n + 2)

theorem necessary_but_not_sufficient_condition
  (a : ℕ → ℝ) :
  condition a → ¬ is_geometric_sequence a :=
sorry

end necessary_but_not_sufficient_condition_l1738_173861


namespace num_real_solutions_l1738_173814

theorem num_real_solutions (x : ℝ) (A B : Set ℝ) (hx : x ∈ A) (hx2 : x^2 ∈ A) :
  A = {0, 1, 2, x} → B = {1, x^2} → A ∪ B = A → 
  ∃! y : ℝ, y = -Real.sqrt 2 ∨ y = Real.sqrt 2 :=
by
  intro hA hB hA_union_B
  sorry

end num_real_solutions_l1738_173814


namespace find_multiple_l1738_173878

theorem find_multiple (x m : ℝ) (hx : x = 3) (h : x + 17 = m * (1 / x)) : m = 60 := 
by
  sorry

end find_multiple_l1738_173878


namespace smaller_angle_in_parallelogram_l1738_173892

theorem smaller_angle_in_parallelogram (a b : ℝ) (h1 : a + b = 180)
  (h2 : b = a + 70) : a = 55 :=
by sorry

end smaller_angle_in_parallelogram_l1738_173892


namespace find_number_of_moles_of_CaCO3_formed_l1738_173862

-- Define the molar ratios and the given condition in structures.
structure Reaction :=
  (moles_CaOH2 : ℕ)
  (moles_CO2 : ℕ)
  (moles_CaCO3 : ℕ)

-- Define a balanced reaction for Ca(OH)2 + CO2 -> CaCO3 + H2O with 1:1 molar ratio.
def balanced_reaction (r : Reaction) : Prop :=
  r.moles_CaOH2 = r.moles_CO2 ∧ r.moles_CaCO3 = r.moles_CO2

-- Define the given condition, which is we have 3 moles of CO2 and formed 3 moles of CaCO3.
def given_condition : Reaction :=
  { moles_CaOH2 := 3, moles_CO2 := 3, moles_CaCO3 := 3 }

-- Theorem: Given 3 moles of CO2, we need to prove 3 moles of CaCO3 are formed based on the balanced reaction.
theorem find_number_of_moles_of_CaCO3_formed :
  balanced_reaction given_condition :=
by {
  -- This part will contain the proof when implemented.
  sorry
}

end find_number_of_moles_of_CaCO3_formed_l1738_173862


namespace midpoint_of_line_segment_l1738_173826

theorem midpoint_of_line_segment :
  let z1 := Complex.mk (-7) 5
  let z2 := Complex.mk 5 (-3)
  (z1 + z2) / 2 = Complex.mk (-1) 1 := by sorry

end midpoint_of_line_segment_l1738_173826


namespace probability_of_snow_during_holiday_l1738_173887

theorem probability_of_snow_during_holiday
  (P_snow_Friday : ℝ)
  (P_snow_Monday : ℝ)
  (P_snow_independent : true) -- Placeholder since we assume independence
  (h_Friday: P_snow_Friday = 0.30)
  (h_Monday: P_snow_Monday = 0.45) :
  ∃ P_snow_holiday, P_snow_holiday = 0.615 :=
by
  sorry

end probability_of_snow_during_holiday_l1738_173887


namespace water_fall_amount_l1738_173824

theorem water_fall_amount (M_before J_before M_after J_after n : ℕ) 
  (h1 : M_before = 48) 
  (h2 : M_before = J_before + 32)
  (h3 : M_after = M_before + n) 
  (h4 : J_after = J_before + n)
  (h5 : M_after = 2 * J_after) : 
  n = 16 :=
by 
  -- proof omitted
  sorry

end water_fall_amount_l1738_173824


namespace total_birds_count_l1738_173873

def blackbirds_per_tree : ℕ := 3
def tree_count : ℕ := 7
def magpies : ℕ := 13

theorem total_birds_count : (blackbirds_per_tree * tree_count) + magpies = 34 := by
  sorry

end total_birds_count_l1738_173873


namespace equivalence_statements_l1738_173844

variables (P Q : Prop)

theorem equivalence_statements :
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by sorry

end equivalence_statements_l1738_173844


namespace geometric_sequence_first_term_l1738_173882

theorem geometric_sequence_first_term (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 * a 2 * a 3 = 27) (h3 : a 6 = 27) : a 0 = 1 :=
by
  sorry

end geometric_sequence_first_term_l1738_173882


namespace Q_no_negative_roots_and_at_least_one_positive_root_l1738_173845

def Q (x : ℝ) : ℝ := x^7 - 2 * x^6 - 6 * x^4 - 4 * x + 16

theorem Q_no_negative_roots_and_at_least_one_positive_root :
  (∀ x, x < 0 → Q x > 0) ∧ (∃ x, x > 0 ∧ Q x = 0) := 
sorry

end Q_no_negative_roots_and_at_least_one_positive_root_l1738_173845


namespace train_cross_tunnel_time_l1738_173839

noncomputable def train_length : ℝ := 800 -- in meters
noncomputable def train_speed : ℝ := 78 * 1000 / 3600 -- converted to meters per second
noncomputable def tunnel_length : ℝ := 500 -- in meters
noncomputable def total_distance : ℝ := train_length + tunnel_length -- total distance to travel

theorem train_cross_tunnel_time : total_distance / train_speed / 60 = 1 := by
  sorry

end train_cross_tunnel_time_l1738_173839


namespace expression_evaluation_l1738_173832

theorem expression_evaluation (x : ℝ) (h : 2 * x - 7 = 8 * x - 1) : 5 * (x - 3) = -20 :=
by
  sorry

end expression_evaluation_l1738_173832


namespace point_D_in_fourth_quadrant_l1738_173889

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (-1, -2)
def point_C : ℝ × ℝ := (-1, 2)
def point_D : ℝ × ℝ := (1, -2)

theorem point_D_in_fourth_quadrant : is_in_fourth_quadrant (point_D.1) (point_D.2) :=
by
  sorry

end point_D_in_fourth_quadrant_l1738_173889


namespace remainder_sum_abc_mod5_l1738_173883

theorem remainder_sum_abc_mod5 (a b c : ℕ) (h1 : a < 5) (h2 : b < 5) (h3 : c < 5)
  (h4 : a * b * c ≡ 1 [MOD 5])
  (h5 : 4 * c ≡ 3 [MOD 5])
  (h6 : 3 * b ≡ 2 + b [MOD 5]) :
  (a + b + c) % 5 = 1 :=
  sorry

end remainder_sum_abc_mod5_l1738_173883


namespace example_problem_l1738_173856

-- Define the numbers of students in each grade
def freshmen : ℕ := 240
def sophomores : ℕ := 260
def juniors : ℕ := 300

-- Define the total number of spots for the trip
def total_spots : ℕ := 40

-- Define the total number of students
def total_students : ℕ := freshmen + sophomores + juniors

-- Define the fraction of sophomores relative to the total number of students
def fraction_sophomores : ℚ := sophomores / total_students

-- Define the number of spots allocated to sophomores
def spots_sophomores : ℚ := fraction_sophomores * total_spots

-- The theorem we need to prove
theorem example_problem : spots_sophomores = 13 :=
by 
  sorry

end example_problem_l1738_173856


namespace decipher_proof_l1738_173831

noncomputable def decipher_message (n : ℕ) (hidden_message : String) :=
  if n = 2211169691162 then hidden_message = "Kiss me, dearest" else false

theorem decipher_proof :
  decipher_message 2211169691162 "Kiss me, dearest" = true :=
by
  -- Proof skipped
  sorry

end decipher_proof_l1738_173831


namespace smallest_K_222_multiple_of_198_l1738_173885

theorem smallest_K_222_multiple_of_198 :
  ∀ K : ℕ, (∃ x : ℕ, x = 2 * (10^K - 1) / 9 ∧ x % 198 = 0) → K = 18 :=
by
  sorry

end smallest_K_222_multiple_of_198_l1738_173885


namespace bakery_ratio_l1738_173894

theorem bakery_ratio (F B : ℕ) 
    (h1 : F = 10 * B)
    (h2 : F = 8 * (B + 60))
    (sugar : ℕ)
    (h3 : sugar = 3000) :
    sugar / F = 5 / 4 :=
by sorry

end bakery_ratio_l1738_173894


namespace six_digits_sum_l1738_173810

theorem six_digits_sum 
  (a b c d e f g : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e) (h5 : a ≠ f) (h6 : a ≠ g)
  (h7 : b ≠ c) (h8 : b ≠ d) (h9 : b ≠ e) (h10 : b ≠ f) (h11 : b ≠ g)
  (h12 : c ≠ d) (h13 : c ≠ e) (h14 : c ≠ f) (h15 : c ≠ g)
  (h16 : d ≠ e) (h17 : d ≠ f) (h18 : d ≠ g)
  (h19 : e ≠ f) (h20 : e ≠ g)
  (h21 : f ≠ g)
  (h22 : 2 ≤ a) (h23 : a ≤ 9) 
  (h24 : 2 ≤ b) (h25 : b ≤ 9) 
  (h26 : 2 ≤ c) (h27 : c ≤ 9)
  (h28 : 2 ≤ d) (h29 : d ≤ 9)
  (h30 : 2 ≤ e) (h31 : e ≤ 9)
  (h32 : 2 ≤ f) (h33 : f ≤ 9)
  (h34 : 2 ≤ g) (h35 : g ≤ 9)
  (h36 : a + b + c = 25)
  (h37 : d + e + f + g = 15)
  (h38 : b = e) :
  a + b + c + d + f + g = 31 := 
sorry

end six_digits_sum_l1738_173810


namespace matrix_product_is_zero_l1738_173842

def vec3 := (ℝ × ℝ × ℝ)

def M1 (a b c : ℝ) : vec3 × vec3 × vec3 :=
  ((0, 2 * c, -2 * b),
   (-2 * c, 0, 2 * a),
   (2 * b, -2 * a, 0))

def M2 (a b c : ℝ) : vec3 × vec3 × vec3 :=
  ((2 * a^2, a^2 + b^2, a^2 + c^2),
   (a^2 + b^2, 2 * b^2, b^2 + c^2),
   (a^2 + c^2, b^2 + c^2, 2 * c^2))

def matrix_mul (m1 m2 : vec3 × vec3 × vec3) : vec3 × vec3 × vec3 := sorry

theorem matrix_product_is_zero (a b c : ℝ) :
  matrix_mul (M1 a b c) (M2 a b c) = ((0, 0, 0), (0, 0, 0), (0, 0, 0)) := by
  sorry

end matrix_product_is_zero_l1738_173842


namespace range_of_a_l1738_173816

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l1738_173816


namespace cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7_l1738_173851

theorem cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7 (p q r : ℝ)
  (h_roots : 3*p^3 - 4*p^2 + 220*p - 7 = 0 ∧ 3*q^3 - 4*q^2 + 220*q - 7 = 0 ∧ 3*r^3 - 4*r^2 + 220*r - 7 = 0)
  (h_vieta : p + q + r = 4 / 3) :
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 64.556 :=
sorry

end cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7_l1738_173851


namespace scientific_notation_of_number_l1738_173804

theorem scientific_notation_of_number (num : ℝ) (a b: ℝ) : 
  num = 0.0000046 ∧ 
  (a = 46 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -6 ∨ 
   a = 0.46 ∧ b = -5) → 
  a = 4.6 ∧ b = -6 :=
by
  sorry

end scientific_notation_of_number_l1738_173804


namespace science_homework_is_50_minutes_l1738_173876

-- Define the times for each homework and project in minutes
def total_time : ℕ := 3 * 60  -- 3 hours converted to minutes
def math_homework : ℕ := 45
def english_homework : ℕ := 30
def history_homework : ℕ := 25
def special_project : ℕ := 30

-- Define a function to compute the time for science homework
def science_homework_time 
  (total_time : ℕ) 
  (math_time : ℕ) 
  (english_time : ℕ) 
  (history_time : ℕ) 
  (project_time : ℕ) : ℕ :=
  total_time - (math_time + english_time + history_time + project_time)

-- The theorem to prove the time Porche's science homework takes
theorem science_homework_is_50_minutes : 
  science_homework_time total_time math_homework english_homework history_homework special_project = 50 := 
sorry

end science_homework_is_50_minutes_l1738_173876


namespace diet_soda_bottles_l1738_173825

/-- Define variables for the number of bottles. -/
def total_bottles : ℕ := 38
def regular_soda : ℕ := 30

/-- Define the problem of finding the number of diet soda bottles -/
def diet_soda := total_bottles - regular_soda

/-- Claim that the number of diet soda bottles is 8 -/
theorem diet_soda_bottles : diet_soda = 8 :=
by
  sorry

end diet_soda_bottles_l1738_173825


namespace largest_rectangle_area_l1738_173877

theorem largest_rectangle_area (x y : ℝ) (h : 2*x + 2*y = 60) : x * y ≤ 225 :=
sorry

end largest_rectangle_area_l1738_173877


namespace rectangle_longer_side_length_l1738_173852

theorem rectangle_longer_side_length (r : ℝ) (h1 : r = 4) 
  (h2 : ∃ w l, w * l = 2 * (π * r^2) ∧ w = 2 * r) : 
  ∃ l, l = 4 * π :=
by 
  obtain ⟨w, l, h_area, h_shorter_side⟩ := h2
  sorry

end rectangle_longer_side_length_l1738_173852


namespace custom_op_1_neg3_l1738_173815

-- Define the custom operation as per the condition
def custom_op (a b : ℤ) : ℤ := a^2 + 2 * a * b - b^2

-- The theorem to prove that 1 * (-3) = -14 using the defined operation
theorem custom_op_1_neg3 : custom_op 1 (-3) = -14 := sorry

end custom_op_1_neg3_l1738_173815


namespace cake_has_more_calories_l1738_173833

-- Define the conditions
def cake_slices : Nat := 8
def cake_calories_per_slice : Nat := 347
def brownie_count : Nat := 6
def brownie_calories_per_brownie : Nat := 375

-- Define the total calories for the cake and the brownies
def total_cake_calories : Nat := cake_slices * cake_calories_per_slice
def total_brownie_calories : Nat := brownie_count * brownie_calories_per_brownie

-- Prove the difference in calories
theorem cake_has_more_calories : 
  total_cake_calories - total_brownie_calories = 526 :=
by
  sorry

end cake_has_more_calories_l1738_173833
