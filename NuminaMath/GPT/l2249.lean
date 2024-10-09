import Mathlib

namespace solve_for_a_minus_c_l2249_224941

theorem solve_for_a_minus_c 
  (a b c d : ℝ) 
  (h1 : a - b = c + d + 9) 
  (h2 : a + b = c - d - 3) : 
  a - c = 3 := by
  sorry

end solve_for_a_minus_c_l2249_224941


namespace num_integer_solutions_quadratic_square_l2249_224968

theorem num_integer_solutions_quadratic_square : 
  (∃ xs : Finset ℤ, 
    (∀ x ∈ xs, ∃ k : ℤ, (x^4 + 8*x^3 + 18*x^2 + 8*x + 64) = k^2) ∧ 
    xs.card = 2) := sorry

end num_integer_solutions_quadratic_square_l2249_224968


namespace statement_A_statement_B_statement_C_statement_D_statement_E_l2249_224948

-- Define a statement for each case and prove each one
theorem statement_A (x : ℝ) (h : x ≥ 0) : x^2 ≥ x :=
sorry

theorem statement_B (x : ℝ) (h : x^2 ≥ 0) : abs x ≥ 0 :=
sorry

theorem statement_C (x : ℝ) (h : x^2 ≤ x) : ¬ (x ≤ 1) :=
sorry

theorem statement_D (x : ℝ) (h : x^2 ≥ x) : ¬ (x ≤ 0) :=
sorry

theorem statement_E (x : ℝ) (h : x ≤ -1) : x^2 ≥ abs x :=
sorry

end statement_A_statement_B_statement_C_statement_D_statement_E_l2249_224948


namespace num_complementary_sets_eq_117_l2249_224939

structure Card :=
(shape : Type)
(color : Type)
(shade : Type)

def deck_condition: Prop := 
  ∃ (deck : List Card), 
  deck.length = 27 ∧
  ∀ c1 c2 c3, c1 ∈ deck ∧ c2 ∈ deck ∧ c3 ∈ deck →
  (c1.shape ≠ c2.shape ∨ c2.shape ≠ c3.shape ∨ c1.shape = c3.shape) ∧
  (c1.color ≠ c2.color ∨ c2.color ≠ c3.color ∨ c1.color = c3.color) ∧
  (c1.shade ≠ c2.shade ∨ c2.shade ≠ c3.shade ∨ c1.shade = c3.shade)

theorem num_complementary_sets_eq_117 :
  deck_condition → ∃ sets : List (List Card), sets.length = 117 := sorry

end num_complementary_sets_eq_117_l2249_224939


namespace monomials_like_terms_l2249_224911

variable (m n : ℤ)

theorem monomials_like_terms (hm : m = 3) (hn : n = 1) : m - 2 * n = 1 :=
by
  sorry

end monomials_like_terms_l2249_224911


namespace koala_fiber_intake_l2249_224994

theorem koala_fiber_intake (absorption_percentage : ℝ) (absorbed_fiber : ℝ) (total_fiber : ℝ) :
  absorption_percentage = 0.30 → absorbed_fiber = 12 → absorbed_fiber = absorption_percentage * total_fiber → total_fiber = 40 :=
by
  intros h1 h2 h3
  sorry

end koala_fiber_intake_l2249_224994


namespace f_g_of_3_l2249_224907

def g (x : ℕ) : ℕ := x^3
def f (x : ℕ) : ℕ := 3 * x - 2

theorem f_g_of_3 : f (g 3) = 79 := by
  sorry

end f_g_of_3_l2249_224907


namespace like_terms_exponent_l2249_224921

theorem like_terms_exponent (m n : ℤ) (h₁ : n = 2) (h₂ : m = 1) : m - n = -1 :=
by
  sorry

end like_terms_exponent_l2249_224921


namespace nguyen_fabric_needs_l2249_224965

def yards_to_feet (yards : ℝ) := yards * 3
def total_fabric_needed (pairs : ℝ) (fabric_per_pair : ℝ) := pairs * fabric_per_pair
def fabric_still_needed (total_needed : ℝ) (already_have : ℝ) := total_needed - already_have

theorem nguyen_fabric_needs :
  let pairs := 7
  let fabric_per_pair := 8.5
  let yards_have := 3.5
  let feet_have := yards_to_feet yards_have
  let total_needed := total_fabric_needed pairs fabric_per_pair
  fabric_still_needed total_needed feet_have = 49 :=
by
  sorry

end nguyen_fabric_needs_l2249_224965


namespace edward_spent_on_books_l2249_224904

def money_spent_on_books (initial_amount spent_on_pens amount_left : ℕ) : ℕ :=
  initial_amount - amount_left - spent_on_pens

theorem edward_spent_on_books :
  ∃ (x : ℕ), x = 6 → 
  ∀ {initial_amount spent_on_pens amount_left : ℕ},
    initial_amount = 41 →
    spent_on_pens = 16 →
    amount_left = 19 →
    x = money_spent_on_books initial_amount spent_on_pens amount_left :=
by
  sorry

end edward_spent_on_books_l2249_224904


namespace find_quadratic_function_l2249_224956

open Function

-- Define the quadratic function g(x) with parameters c and d
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- State the main theorem
theorem find_quadratic_function :
  ∃ (c d : ℝ), (∀ x : ℝ, (g c d (g c d x + x)) / (g c d x) = x^2 + 120 * x + 360) ∧ c = 119 ∧ d = 240 :=
by
  sorry

end find_quadratic_function_l2249_224956


namespace angle_terminal_side_l2249_224981

noncomputable def rad_to_deg (r : ℝ) : ℝ := r * (180 / Real.pi)

theorem angle_terminal_side :
  ∃ k : ℤ, rad_to_deg (π / 12) + 360 * k = 375 :=
sorry

end angle_terminal_side_l2249_224981


namespace find_income_separator_l2249_224949

-- Define the income and tax parameters
def income : ℝ := 60000
def total_tax : ℝ := 8000
def rate1 : ℝ := 0.10
def rate2 : ℝ := 0.20

-- Define the function for total tax calculation
def tax (I : ℝ) : ℝ := rate1 * I + rate2 * (income - I)

theorem find_income_separator (I : ℝ) (h: tax I = total_tax) : I = 40000 :=
by sorry

end find_income_separator_l2249_224949


namespace fixed_salary_new_scheme_l2249_224951

theorem fixed_salary_new_scheme :
  let old_commission_rate := 0.05
  let new_commission_rate := 0.025
  let sales_target := 4000
  let total_sales := 12000
  let remuneration_difference := 600
  let old_remuneration := old_commission_rate * total_sales
  let new_commission_earnings := new_commission_rate * (total_sales - sales_target)
  let new_remuneration := old_remuneration + remuneration_difference
  ∃ F, F + new_commission_earnings = new_remuneration :=
by
  sorry

end fixed_salary_new_scheme_l2249_224951


namespace polynomial_root_product_l2249_224931

theorem polynomial_root_product (b c : ℤ) : (∀ r : ℝ, r^2 - r - 1 = 0 → r^6 - b * r - c = 0) → b * c = 40 := 
by
  sorry

end polynomial_root_product_l2249_224931


namespace numbers_not_all_less_than_six_l2249_224975

theorem numbers_not_all_less_than_six (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬ (a + 4 / b < 6 ∧ b + 9 / c < 6 ∧ c + 16 / a < 6) :=
sorry

end numbers_not_all_less_than_six_l2249_224975


namespace matrix_determinant_equality_l2249_224952

open Complex Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_determinant_equality (A B : Matrix n n ℂ) (x : ℂ) 
  (h1 : A ^ 2 + B ^ 2 = 2 * A * B) :
  det (A - x • 1) = det (B - x • 1) :=
  sorry

end matrix_determinant_equality_l2249_224952


namespace evaluate_fraction_l2249_224960

noncomputable section

variables (u v : ℂ)
variables (h1 : u ≠ 0) (h2 : v ≠ 0) (h3 : u^2 + u * v + v^2 = 0)

theorem evaluate_fraction : (u^7 + v^7) / (u + v)^7 = -2 := by
  sorry

end evaluate_fraction_l2249_224960


namespace num_of_veg_people_l2249_224962

def only_veg : ℕ := 19
def both_veg_nonveg : ℕ := 12

theorem num_of_veg_people : only_veg + both_veg_nonveg = 31 := by 
  sorry

end num_of_veg_people_l2249_224962


namespace geometric_shape_is_sphere_l2249_224972

-- Define the spherical coordinate system conditions
def spherical_coordinates (ρ θ φ r : ℝ) : Prop :=
  ρ = r

-- The theorem we want to prove
theorem geometric_shape_is_sphere (ρ θ φ r : ℝ) (h : spherical_coordinates ρ θ φ r) : ∀ (x y z : ℝ), (x^2 + y^2 + z^2 = r^2) :=
by
  sorry

end geometric_shape_is_sphere_l2249_224972


namespace previous_year_profit_percentage_l2249_224992

variables {R P: ℝ}

theorem previous_year_profit_percentage (h1: R > 0)
    (h2: P = 0.1 * R)
    (h3: 0.7 * P = 0.07 * R) :
    (P / R) * 100 = 10 :=
by
  -- Since we have P = 0.1 * R from the conditions and definitions,
  -- it follows straightforwardly that (P / R) * 100 = 10.
  -- We'll continue the proof from here.
  sorry

end previous_year_profit_percentage_l2249_224992


namespace f1_odd_f2_even_l2249_224918

noncomputable def f1 (x : ℝ) : ℝ := x + x^3 + x^5
noncomputable def f2 (x : ℝ) : ℝ := x^2 + 1

theorem f1_odd : ∀ x : ℝ, f1 (-x) = - f1 x := 
by
  sorry

theorem f2_even : ∀ x : ℝ, f2 (-x) = f2 x := 
by
  sorry

end f1_odd_f2_even_l2249_224918


namespace decrease_percent_revenue_l2249_224963

theorem decrease_percent_revenue 
  (T C : ℝ) 
  (hT : T > 0) 
  (hC : C > 0) 
  (new_tax : ℝ := 0.65 * T) 
  (new_consumption : ℝ := 1.15 * C) 
  (original_revenue : ℝ := T * C) 
  (new_revenue : ℝ := new_tax * new_consumption) :
  100 * (original_revenue - new_revenue) / original_revenue = 25.25 :=
sorry

end decrease_percent_revenue_l2249_224963


namespace multiple_of_old_edition_l2249_224938

theorem multiple_of_old_edition 
  (new_pages: ℕ) 
  (old_pages: ℕ) 
  (difference: ℕ) 
  (m: ℕ) 
  (h1: new_pages = 450) 
  (h2: old_pages = 340) 
  (h3: 450 = 340 * m - 230) : 
  m = 2 :=
sorry

end multiple_of_old_edition_l2249_224938


namespace imaginary_part_of_z_is_sqrt2_div2_l2249_224950

open Complex

noncomputable def z : ℂ := abs (1 - I) / (1 - I)

theorem imaginary_part_of_z_is_sqrt2_div2 : z.im = Real.sqrt 2 / 2 := by
  sorry

end imaginary_part_of_z_is_sqrt2_div2_l2249_224950


namespace remainder_of_n_squared_plus_4n_plus_5_l2249_224999

theorem remainder_of_n_squared_plus_4n_plus_5 {n : ℤ} (h : n % 50 = 1) : (n^2 + 4*n + 5) % 50 = 10 :=
by
  sorry

end remainder_of_n_squared_plus_4n_plus_5_l2249_224999


namespace evaluate_expression_l2249_224996

theorem evaluate_expression :
  let x := (1 / 4 : ℚ)
  let y := (1 / 3 : ℚ)
  let z := (-2 : ℚ)
  let w := (3 : ℚ)
  (x^3 * y^2 * z^2 * w) = (1 / 48 : ℚ) :=
by
  let x := (1 / 4 : ℚ)
  let y := (1 / 3 : ℚ)
  let z := (-2 : ℚ)
  let w := (3 : ℚ)
  sorry

end evaluate_expression_l2249_224996


namespace find_prime_p_l2249_224976

theorem find_prime_p :
  ∃ p : ℕ, Prime p ∧ (∃ a b : ℤ, p = 5 ∧ 1 < p ∧ p ≤ 11 ∧ (a^2 + p * a - 720 * p = 0) ∧ (b^2 - p * b + 720 * p = 0)) :=
sorry

end find_prime_p_l2249_224976


namespace mr_li_age_l2249_224923

theorem mr_li_age (xiaofang_age : ℕ) (h1 : xiaofang_age = 5)
  (h2 : ∀ t : ℕ, (t = 3) → ∀ mr_li_age_in_3_years : ℕ, (mr_li_age_in_3_years = xiaofang_age + t + 20)) :
  ∃ mr_li_age : ℕ, mr_li_age = 25 :=
by
  sorry

end mr_li_age_l2249_224923


namespace spider_legs_is_multiple_of_human_legs_l2249_224915

def human_legs : ℕ := 2
def spider_legs : ℕ := 8

theorem spider_legs_is_multiple_of_human_legs :
  spider_legs = 4 * human_legs :=
by 
  sorry

end spider_legs_is_multiple_of_human_legs_l2249_224915


namespace amara_remaining_clothes_l2249_224937

noncomputable def remaining_clothes (initial total_donated thrown_away : ℕ) : ℕ :=
  initial - (total_donated + thrown_away)

theorem amara_remaining_clothes : 
  ∀ (initial donated_first donated_second thrown_away : ℕ), initial = 100 → donated_first = 5 → donated_second = 15 → thrown_away = 15 → 
  remaining_clothes initial (donated_first + donated_second) thrown_away = 65 := 
by 
  intros initial donated_first donated_second thrown_away hinital hdonated_first hdonated_second hthrown_away
  rw [hinital, hdonated_first, hdonated_second, hthrown_away]
  unfold remaining_clothes
  norm_num

end amara_remaining_clothes_l2249_224937


namespace video_call_cost_l2249_224985

-- Definitions based on the conditions
def charge_rate : ℕ := 30    -- Charge rate in won per ten seconds
def call_duration : ℕ := 2 * 60 + 40  -- Call duration in seconds

-- The proof statement, anticipating the solution to be a total cost calculation
theorem video_call_cost : (call_duration / 10) * charge_rate = 480 :=
by
  -- Placeholder for the proof
  sorry

end video_call_cost_l2249_224985


namespace lcm_inequality_l2249_224959

open Nat

theorem lcm_inequality (n : ℕ) (h : n ≥ 5 ∧ Odd n) :
  Nat.lcm n (n + 1) * (n + 2) > Nat.lcm (n + 1) (n + 2) * (n + 3) := by
  sorry

end lcm_inequality_l2249_224959


namespace gina_keeps_170_l2249_224991

theorem gina_keeps_170 (initial_amount : ℕ)
    (money_to_mom : ℕ)
    (money_to_clothes : ℕ)
    (money_to_charity : ℕ)
    (remaining_money : ℕ) :
  initial_amount = 400 →
  money_to_mom = (1 / 4) * initial_amount →
  money_to_clothes = (1 / 8) * initial_amount →
  money_to_charity = (1 / 5) * initial_amount →
  remaining_money = initial_amount - (money_to_mom + money_to_clothes + money_to_charity) →
  remaining_money = 170 := sorry

end gina_keeps_170_l2249_224991


namespace truck_capacities_transportation_plan_l2249_224983

-- Definitions of given conditions
def A_truck_capacity (x y : ℕ) : Prop := x + 2 * y = 50
def B_truck_capacity (x y : ℕ) : Prop := 5 * x + 4 * y = 160
def total_transport_cost (m n : ℕ) : ℕ := 500 * m + 400 * n
def most_cost_effective_plan (m n cost : ℕ) : Prop := 
  m + 2 * n = 10 ∧ (20 * m + 15 * n = 190) ∧ cost = total_transport_cost m n ∧ cost = 4800

-- Proving the capacities of trucks A and B
theorem truck_capacities : 
  ∃ x y : ℕ, A_truck_capacity x y ∧ B_truck_capacity x y ∧ x = 20 ∧ y = 15 := 
sorry

-- Proving the most cost-effective transportation plan
theorem transportation_plan : 
  ∃ m n cost, (total_transport_cost m n = cost) ∧ most_cost_effective_plan m n cost := 
sorry

end truck_capacities_transportation_plan_l2249_224983


namespace series_sum_eq_one_fourth_l2249_224998

noncomputable def sum_series : ℝ :=
  ∑' n, (3 ^ n / (1 + 3 ^ n + 3 ^ (n + 2) + 3 ^ (2 * n + 2)))

theorem series_sum_eq_one_fourth :
  sum_series = 1 / 4 :=
by
  sorry

end series_sum_eq_one_fourth_l2249_224998


namespace sum_of_roots_l2249_224932

theorem sum_of_roots (x : ℝ) :
  (x^2 = 10 * x - 13) → ∃ s, s = 10 := 
by
  sorry

end sum_of_roots_l2249_224932


namespace distance_city_A_C_l2249_224993

-- Define the conditions
def starts_simultaneously (A : Prop) (Eddy Freddy : Prop) := Eddy ∧ Freddy
def travels (A B C : Prop) (Eddy Freddy : Prop) := Eddy → 3 = 3 ∧ Freddy → 4 = 4
def distance_AB (A B : Prop) := 600
def speed_ratio (Eddy_speed Freddy_speed : ℝ) := Eddy_speed / Freddy_speed = 1.7391304347826086

noncomputable def distance_AC (Eddy_time Freddy_time : ℝ) (Eddy_speed Freddy_speed : ℝ) 
  := (Eddy_speed / 1.7391304347826086) * Freddy_time

theorem distance_city_A_C 
  (A B C Eddy Freddy : Prop)
  (Eddy_time Freddy_time : ℝ) 
  (Eddy_speed effective_Freddy_speed : ℝ)
  (h1 : starts_simultaneously A Eddy Freddy)
  (h2 : travels A B C Eddy Freddy)
  (h3 : distance_AB A B = 600)
  (h4 : speed_ratio Eddy_speed effective_Freddy_speed)
  (h5 : Eddy_speed = 200)
  (h6 : effective_Freddy_speed = 115)
  : distance_AC Eddy_time Freddy_time Eddy_speed effective_Freddy_speed = 460 := 
  by sorry

end distance_city_A_C_l2249_224993


namespace eval_expression_l2249_224954

theorem eval_expression : (-2 ^ 4) + 3 * (-1) ^ 6 - (-2) ^ 3 = -5 := by
  sorry

end eval_expression_l2249_224954


namespace lcm_18_24_eq_72_l2249_224988

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l2249_224988


namespace sufficient_but_not_necessary_condition_l2249_224928

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 3 → x^2 - 2 * x > 0) ∧ ¬ (x^2 - 2 * x > 0 → x > 3) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l2249_224928


namespace complex_square_sum_eq_five_l2249_224944

theorem complex_square_sum_eq_five (a b : ℝ) (h : (a + b * I) ^ 2 = 3 + 4 * I) : a^2 + b^2 = 5 := 
by sorry

end complex_square_sum_eq_five_l2249_224944


namespace fraction_equality_l2249_224982

def op_at (a b : ℕ) : ℕ := a * b + b^2
def op_hash (a b : ℕ) : ℕ := a + b + a * (b^2)

theorem fraction_equality : (op_at 5 3 : ℚ) / (op_hash 5 3 : ℚ) = 24 / 53 := 
by 
  sorry

end fraction_equality_l2249_224982


namespace inequality_one_system_of_inequalities_l2249_224919

theorem inequality_one (x : ℝ) : 
  (2 * x - 2) / 3 ≤ 2 - (2 * x + 2) / 2 → x ≤ 1 :=
sorry

theorem system_of_inequalities (x : ℝ) : 
  (3 * (x - 2) - 1 ≥ -4 - 2 * (x - 2) → x ≥ 7 / 5) ∧
  ((1 - 2 * x) / 3 > (3 * (2 * x - 1)) / 2 → x < 1 / 2) → false :=
sorry

end inequality_one_system_of_inequalities_l2249_224919


namespace bus_ride_duration_l2249_224910

theorem bus_ride_duration (total_hours : ℕ) (train_hours : ℕ) (walk_minutes : ℕ) (wait_factor : ℕ) 
    (h_total : total_hours = 8)
    (h_train : train_hours = 6)
    (h_walk : walk_minutes = 15)
    (h_wait : wait_factor = 2) : 
    let total_minutes := total_hours * 60
    let train_minutes := train_hours * 60
    let wait_minutes := wait_factor * walk_minutes
    let travel_minutes := total_minutes - train_minutes
    let bus_ride_minutes := travel_minutes - walk_minutes - wait_minutes
    bus_ride_minutes = 75 :=
by
  sorry

end bus_ride_duration_l2249_224910


namespace expand_expression_l2249_224900

theorem expand_expression (x : ℝ) : 
  (11 * x^2 + 5 * x - 3) * (3 * x^3) = 33 * x^5 + 15 * x^4 - 9 * x^3 :=
by 
  sorry

end expand_expression_l2249_224900


namespace constant_sum_powers_l2249_224912

theorem constant_sum_powers (n : ℕ) (x y z : ℝ) (h_sum : x + y + z = 0) (h_prod : x * y * z = 1) :
  (∀ x y z : ℝ, x + y + z = 0 → x * y * z = 1 → x^n + y^n + z^n = x^n + y^n + z^n ↔ (n = 1 ∨ n = 3)) :=
by
  sorry

end constant_sum_powers_l2249_224912


namespace complex_modulus_eq_one_l2249_224974

open Complex

theorem complex_modulus_eq_one (a b : ℝ) (h : (1 + 2 * Complex.I) / (a + b * Complex.I) = 2 - Complex.I) :
  abs (a - b * Complex.I) = 1 := by
  sorry

end complex_modulus_eq_one_l2249_224974


namespace passed_boys_count_l2249_224917

theorem passed_boys_count (total_boys avg_passed avg_failed overall_avg : ℕ) 
  (total_boys_eq : total_boys = 120) 
  (avg_passed_eq : avg_passed = 39) 
  (avg_failed_eq : avg_failed = 15) 
  (overall_avg_eq : overall_avg = 38) :
  let marks_by_passed := total_boys * overall_avg 
                         - (total_boys - passed) * avg_failed;
  let passed := marks_by_passed / avg_passed;
  passed = 115 := 
by
  sorry

end passed_boys_count_l2249_224917


namespace angle_ratio_l2249_224990

-- Definitions as per the conditions
def bisects (x y z : ℝ) : Prop := x = y / 2
def trisects (x y z : ℝ) : Prop := y = x / 3

theorem angle_ratio (ABC PBQ BM x : ℝ) (h1 : bisects PBQ ABC PQ)
                                    (h2 : trisects PBQ BM M) :
  PBQ = 2 * x →
  PBQ = ABC / 2 →
  MBQ = x →
  ABQ = 4 * x →
  MBQ / ABQ = 1 / 4 :=
by
  intros
  sorry

end angle_ratio_l2249_224990


namespace factorize_expression_find_xy_l2249_224989

-- Problem 1: Factorizing the quadratic expression
theorem factorize_expression (x : ℝ) : 
  x^2 - 120 * x + 3456 = (x - 48) * (x - 72) :=
sorry

-- Problem 2: Finding the product xy from the given equation
theorem find_xy (x y : ℝ) (h : x^2 + y^2 + 8 * x - 12 * y + 52 = 0) : 
  x * y = -24 :=
sorry

end factorize_expression_find_xy_l2249_224989


namespace maximum_k_for_ray_below_f_l2249_224958

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 3 * x - 2

theorem maximum_k_for_ray_below_f :
  let g (x : ℝ) : ℝ := (x * Real.log x + 3 * x - 2) / (x - 1)
  ∃ k : ℤ, ∀ x > 1, g x > k ∧ k = 5 :=
by sorry

end maximum_k_for_ray_below_f_l2249_224958


namespace cost_price_computer_table_l2249_224980

-- Define the variables
def cost_price : ℝ := 3840
def selling_price (CP : ℝ) := CP * 1.25

-- State the conditions and the proof problem
theorem cost_price_computer_table 
  (SP : ℝ) 
  (h1 : SP = 4800)
  (h2 : ∀ CP : ℝ, SP = selling_price CP) :
  cost_price = 3840 :=
by 
  sorry

end cost_price_computer_table_l2249_224980


namespace trihedral_angle_properties_l2249_224984

-- Definitions for the problem's conditions
variables {α β γ : ℝ}
variables {A B C S : Type}
variables (angle_ASB angle_BSC angle_CSA : ℝ)

-- Given the conditions of the trihedral angle and the dihedral angles
theorem trihedral_angle_properties 
  (h1 : angle_ASB + angle_BSC + angle_CSA < 2 * Real.pi)
  (h2 : α + β + γ > Real.pi) : 
  true := 
by
  sorry

end trihedral_angle_properties_l2249_224984


namespace pet_store_cages_l2249_224903

theorem pet_store_cages (total_puppies sold_puppies puppies_per_cage : ℕ) (h1 : total_puppies = 45) (h2 : sold_puppies = 39) (h3 : puppies_per_cage = 2) :
  (total_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l2249_224903


namespace range_of_x_of_sqrt_x_plus_3_l2249_224973

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l2249_224973


namespace compare_abc_l2249_224920

open Real

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Assuming the conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom derivative : ∀ x : ℝ, f' x = deriv f x
axiom monotonicity_condition : ∀ x > 0, x * f' x < f x

-- Definitions of a, b, and c
noncomputable def a := 2 * f (1 / 2)
noncomputable def b := - (1 / 2) * f (-2)
noncomputable def c := - (1 / log 2) * f (log (1 / 2))

theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l2249_224920


namespace ratio_upstream_downstream_l2249_224926

noncomputable def ratio_time_upstream_to_downstream
  (V_b V_s : ℕ) (T_u T_d : ℕ) : ℕ :=
(V_b + V_s) / (V_b - V_s)

theorem ratio_upstream_downstream
  (V_b V_s : ℕ) (hVb : V_b = 48) (hVs : V_s = 16) (T_u T_d : ℕ)
  (hT : ratio_time_upstream_to_downstream V_b V_s T_u T_d = 2) :
  T_u / T_d = 2 := by
  sorry

end ratio_upstream_downstream_l2249_224926


namespace greatest_t_value_l2249_224987

theorem greatest_t_value :
  ∃ t_max : ℝ, (∀ t : ℝ, ((t ≠  8) ∧ (t ≠ -7) → (t^2 - t - 90) / (t - 8) = 6 / (t + 7) → t ≤ t_max)) ∧ t_max = -1 :=
sorry

end greatest_t_value_l2249_224987


namespace g_at_10_l2249_224914

noncomputable def g : ℕ → ℝ :=
sorry

axiom g_1 : g 1 = 2
axiom g_prop (m n : ℕ) (hmn : m ≥ n) : g (m + n) + g (m - n) = 2 * (g m + g n)

theorem g_at_10 : g 10 = 200 := 
sorry

end g_at_10_l2249_224914


namespace cricket_average_increase_l2249_224995

theorem cricket_average_increase :
  ∀ (x : ℝ), (11 * (33 + x) = 407) → (x = 4) :=
  by 
  intros x hx
  sorry

end cricket_average_increase_l2249_224995


namespace original_number_is_14_l2249_224946

def two_digit_number_increased_by_2_or_4_results_fourfold (x : ℕ) : Prop :=
  (x >= 10) ∧ (x < 100) ∧ 
  (∃ (a b : ℕ), a + 2 = ((x / 10 + 2) % 10) ∧ b + 2 = (x % 10)) ∧
  (4 * x = ((x / 10 + 2) * 10 + (x % 10 + 2)) ∨ 
   4 * x = ((x / 10 + 2) * 10 + (x % 10 + 4)) ∨ 
   4 * x = ((x / 10 + 4) * 10 + (x % 10 + 2)) ∨ 
   4 * x = ((x / 10 + 4) * 10 + (x % 10 + 4)))

theorem original_number_is_14 : ∃ x : ℕ, two_digit_number_increased_by_2_or_4_results_fourfold x ∧ x = 14 :=
by
  sorry

end original_number_is_14_l2249_224946


namespace total_cantaloupes_l2249_224986

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by sorry

end total_cantaloupes_l2249_224986


namespace evaluate_product_l2249_224940

-- Define the given numerical values
def a : ℝ := 2.5
def b : ℝ := 50.5
def c : ℝ := 0.15

-- State the theorem we want to prove
theorem evaluate_product : a * (b + c) = 126.625 := by
  sorry

end evaluate_product_l2249_224940


namespace siblings_of_John_l2249_224957

structure Child :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)
  (height : String)

def John : Child := {name := "John", eyeColor := "Brown", hairColor := "Blonde", height := "Tall"}
def Emma : Child := {name := "Emma", eyeColor := "Blue", hairColor := "Black", height := "Tall"}
def Oliver : Child := {name := "Oliver", eyeColor := "Brown", hairColor := "Black", height := "Short"}
def Mia : Child := {name := "Mia", eyeColor := "Blue", hairColor := "Blonde", height := "Short"}
def Lucas : Child := {name := "Lucas", eyeColor := "Blue", hairColor := "Black", height := "Tall"}
def Sophia : Child := {name := "Sophia", eyeColor := "Blue", hairColor := "Blonde", height := "Tall"}

theorem siblings_of_John : 
  (John.hairColor = Mia.hairColor ∧ John.hairColor = Sophia.hairColor) ∧
  ((John.eyeColor = Mia.eyeColor ∨ John.eyeColor = Sophia.eyeColor) ∨
   (John.height = Mia.height ∨ John.height = Sophia.height)) ∧
  (Mia.eyeColor = Sophia.eyeColor ∨ Mia.hairColor = Sophia.hairColor ∨ Mia.height = Sophia.height) ∧
  (John.hairColor = "Blonde") ∧
  (John.height = "Tall") ∧
  (Mia.hairColor = "Blonde") ∧
  (Sophia.hairColor = "Blonde") ∧
  (Sophia.height = "Tall") 
  → True := sorry

end siblings_of_John_l2249_224957


namespace determine_n_l2249_224964

theorem determine_n (n : ℕ) (h : 9^4 = 3^n) : n = 8 :=
by {
  sorry
}

end determine_n_l2249_224964


namespace distance_proof_l2249_224936

noncomputable section

open Real

-- Define the given conditions
def AB : Real := 3 * sqrt 3
def BC : Real := 2
def theta : Real := 60 -- angle in degrees
def phi : Real := 180 - theta -- supplementary angle to use in the Law of Cosines

-- Helper function to convert degrees to radians
def deg_to_rad (d : Real) : Real := d * (π / 180)

-- Define the law of cosines to compute AC
def distance_AC (AB BC θ : Real) : Real := 
  sqrt (AB^2 + BC^2 - 2 * AB * BC * cos (deg_to_rad θ))

-- The theorem to prove
theorem distance_proof : distance_AC AB BC phi = 7 :=
by
  sorry

end distance_proof_l2249_224936


namespace average_of_six_numbers_l2249_224945

theorem average_of_six_numbers (A : ℝ) (x y z w u v : ℝ)
  (h1 : (x + y + z + w + u + v) / 6 = A)
  (h2 : (x + y) / 2 = 1.1)
  (h3 : (z + w) / 2 = 1.4)
  (h4 : (u + v) / 2 = 5) :
  A = 2.5 :=
by
  sorry

end average_of_six_numbers_l2249_224945


namespace geometric_solid_is_tetrahedron_l2249_224979

-- Definitions based on the conditions provided
def top_view_is_triangle : Prop := sorry -- Placeholder for the actual definition
def front_view_is_triangle : Prop := sorry -- Placeholder for the actual definition
def side_view_is_triangle : Prop := sorry -- Placeholder for the actual definition

-- Theorem statement to prove the geometric solid is a triangular pyramid
theorem geometric_solid_is_tetrahedron 
  (h_top : top_view_is_triangle)
  (h_front : front_view_is_triangle)
  (h_side : side_view_is_triangle) :
  -- Conclusion that the solid is a triangular pyramid (tetrahedron)
  is_tetrahedron :=
sorry

end geometric_solid_is_tetrahedron_l2249_224979


namespace difference_in_peaches_l2249_224942

-- Define the number of peaches Audrey has
def audrey_peaches : ℕ := 26

-- Define the number of peaches Paul has
def paul_peaches : ℕ := 48

-- Define the expected difference
def expected_difference : ℕ := 22

-- The theorem stating the problem
theorem difference_in_peaches : (paul_peaches - audrey_peaches = expected_difference) :=
by
  sorry

end difference_in_peaches_l2249_224942


namespace no_passing_quadrant_III_l2249_224908

def y (k x : ℝ) : ℝ := k * x - k

theorem no_passing_quadrant_III (k : ℝ) (h : k < 0) :
  ¬(∃ x y : ℝ, x < 0 ∧ y < 0 ∧ y = k * x - k) :=
sorry

end no_passing_quadrant_III_l2249_224908


namespace common_ratio_geometric_series_l2249_224977

-- Define the terms of the geometric series
def term (n : ℕ) : ℚ :=
  match n with
  | 0     => 7 / 8
  | 1     => -21 / 32
  | 2     => 63 / 128
  | _     => sorry  -- Placeholder for further terms if necessary

-- Define the common ratio
def common_ratio : ℚ := -3 / 4

-- Prove that the common ratio is consistent for the given series
theorem common_ratio_geometric_series :
  ∀ (n : ℕ), term (n + 1) / term n = common_ratio :=
by
  sorry

end common_ratio_geometric_series_l2249_224977


namespace candies_per_pack_l2249_224929

-- Conditions in Lean:
def total_candies : ℕ := 60
def packs_initially (packs_after : ℕ) : ℕ := packs_after + 1
def packs_after : ℕ := 2
def pack_count : ℕ := packs_initially packs_after

-- The statement of the proof problem:
theorem candies_per_pack : 
  total_candies / pack_count = 20 :=
by
  sorry

end candies_per_pack_l2249_224929


namespace partition_sum_condition_l2249_224966

theorem partition_sum_condition (X : Finset ℕ) (hX : X = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∀ (A B : Finset ℕ), A ∪ B = X → A ∩ B = ∅ →
  ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a + b = c := 
by
  -- sorry is here to acknowledge that no proof is required per instructions.
  sorry

end partition_sum_condition_l2249_224966


namespace quadratic_function_properties_l2249_224967

def quadratic_function (x : ℝ) : ℝ :=
  -6 * x^2 + 36 * x - 48

theorem quadratic_function_properties :
  quadratic_function 2 = 0 ∧ quadratic_function 4 = 0 ∧ quadratic_function 3 = 6 :=
by
  -- The proof is omitted
  -- Placeholder for the proof
  sorry

end quadratic_function_properties_l2249_224967


namespace non_decreasing_f_f_equal_2_at_2_addition_property_under_interval_rule_final_statement_l2249_224906

open Set

noncomputable def f : ℝ → ℝ := sorry

theorem non_decreasing_f (x y : ℝ) (h : x < y) (hx : x ∈ Icc (0 : ℝ) 2) (hy : y ∈ Icc (0 : ℝ) 2) : f x ≤ f y := sorry

theorem f_equal_2_at_2 : f 2 = 2 := sorry

theorem addition_property (x : ℝ) (hx : x ∈ Icc (0 :ℝ) 2) : f x + f (2 - x) = 2 := sorry

theorem under_interval_rule (x : ℝ) (hx : x ∈ Icc (1.5 :ℝ) 2) : f x ≤ 2 * (x - 1) := sorry

theorem final_statement : ∀ x ∈ Icc (0:ℝ) 1, f (f x) ∈ Icc (0:ℝ) 1 := sorry

end non_decreasing_f_f_equal_2_at_2_addition_property_under_interval_rule_final_statement_l2249_224906


namespace three_Y_five_l2249_224924

-- Define the operation Y
def Y (a b : ℕ) : ℕ := 3 * b + 8 * a - a^2

-- State the theorem to prove the value of 3 Y 5
theorem three_Y_five : Y 3 5 = 30 :=
by
  sorry

end three_Y_five_l2249_224924


namespace polynomial_coefficients_equivalence_l2249_224927

theorem polynomial_coefficients_equivalence
    {a0 a1 a2 a3 a4 a5 : ℤ}
    (h_poly : (2*x-1)^5 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5):
    (a0 + a1 + a2 + a3 + a4 + a5 = 1) ∧
    (|a0| + |a1| + |a2| + |a3| + |a4| + |a5| = 243) ∧
    (a1 + a3 + a5 = 122) ∧
    ((a0 + a2 + a4)^2 - (a1 + a3 + a5)^2 = -243) :=
    sorry

end polynomial_coefficients_equivalence_l2249_224927


namespace ellipse_x_intercept_l2249_224953

theorem ellipse_x_intercept :
  let F_1 := (0,3)
  let F_2 := (4,0)
  let ellipse := { P : ℝ × ℝ | (dist P F_1) + (dist P F_2) = 7 }
  ∃ x : ℝ, x ≠ 0 ∧ (x, 0) ∈ ellipse ∧ x = 56 / 11 :=
by
  sorry

end ellipse_x_intercept_l2249_224953


namespace average_of_N_l2249_224902

theorem average_of_N (N : ℤ) (h1 : (1:ℚ)/3 < N/90) (h2 : N/90 < (2:ℚ)/5) : 31 ≤ N ∧ N ≤ 35 → (N = 31 ∨ N = 32 ∨ N = 33 ∨ N = 34 ∨ N = 35) → (31 + 32 + 33 + 34 + 35) / 5 = 33 := by
  sorry

end average_of_N_l2249_224902


namespace line_through_two_points_l2249_224925

-- Define the points (2,5) and (0,3)
structure Point where
  x : ℝ
  y : ℝ

def P1 : Point := {x := 2, y := 5}
def P2 : Point := {x := 0, y := 3}

-- General form of a line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the target line equation as x - y + 3 = 0
def targetLine : Line := {a := 1, b := -1, c := 3}

-- The proof statement to show that the general equation of the line passing through the points (2, 5) and (0, 3) is x - y + 3 = 0
theorem line_through_two_points : ∃ a b c, ∀ x y : ℝ, 
    (a * x + b * y + c = 0) ↔ 
    ((∀ {P : Point}, P = P1 → targetLine.a * P.x + targetLine.b * P.y + targetLine.c = 0) ∧ 
     (∀ {P : Point}, P = P2 → targetLine.a * P.x + targetLine.b * P.y + targetLine.c = 0)) :=
sorry

end line_through_two_points_l2249_224925


namespace product_of_cosines_value_l2249_224935

noncomputable def product_of_cosines : ℝ :=
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) *
  (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12))

theorem product_of_cosines_value :
  product_of_cosines = 1 / 16 :=
by
  sorry

end product_of_cosines_value_l2249_224935


namespace meeting_point_ratio_l2249_224947

theorem meeting_point_ratio (v1 v2 : ℝ) (TA TB : ℝ)
  (h1 : TA = 45 * v2)
  (h2 : TB = 20 * v1)
  (h3 : (TA / v1) - (TB / v2) = 11) :
  TA / TB = 9 / 5 :=
by sorry

end meeting_point_ratio_l2249_224947


namespace train_length_300_l2249_224970

/-- 
Proving the length of the train given the conditions on crossing times and length of the platform.
-/
theorem train_length_300 (V L : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 200 = V * 30) : 
  L = 300 := 
by
  sorry

end train_length_300_l2249_224970


namespace necessary_but_not_sufficient_condition_l2249_224955

theorem necessary_but_not_sufficient_condition (a : ℕ → ℝ) (a1_pos : 0 < a 1) (q : ℝ) (geo_seq : ∀ n, a (n+1) = q * a n) : 
  (∀ n : ℕ, a (2*n + 1) + a (2*n + 2) < 0) → q < 0 :=
sorry

end necessary_but_not_sufficient_condition_l2249_224955


namespace no_solution_exists_l2249_224943

open Int

theorem no_solution_exists (x y z : ℕ) (hx : x > 0) (hy : y > 0)
  (hz : z = Nat.gcd x y) : x + y^2 + z^3 ≠ x * y * z := 
sorry

end no_solution_exists_l2249_224943


namespace team_total_points_l2249_224916

theorem team_total_points :
  let connor_initial := 2
  let amy_initial := connor_initial + 4
  let jason_initial := amy_initial * 2
  let emily_initial := 3 * (connor_initial + amy_initial + jason_initial)
  let team_before_bonus := connor_initial + amy_initial + jason_initial
  let connor_total := connor_initial + 3
  let amy_total := amy_initial + 5
  let jason_total := jason_initial + 1
  let emily_total := emily_initial
  let team_total := connor_total + amy_total + jason_total + emily_total
  team_total = 89 :=
by
  let connor_initial := 2
  let amy_initial := connor_initial + 4
  let jason_initial := amy_initial * 2
  let emily_initial := 3 * (connor_initial + amy_initial + jason_initial)
  let team_before_bonus := connor_initial + amy_initial + jason_initial
  let connor_total := connor_initial + 3
  let amy_total := amy_initial + 5
  let jason_total := jason_initial + 1
  let emily_total := emily_initial
  let team_total := connor_total + amy_total + jason_total + emily_total
  sorry

end team_total_points_l2249_224916


namespace james_earnings_per_subscriber_is_9_l2249_224922

/-
Problem:
James streams on Twitch. He had 150 subscribers and then someone gifted 50 subscribers. If he gets a certain amount per month per subscriber and now makes $1800 a month, how much does he make per subscriber?
-/

def initial_subscribers : ℕ := 150
def gifted_subscribers : ℕ := 50
def total_subscribers := initial_subscribers + gifted_subscribers
def total_earnings : ℤ := 1800

def earnings_per_subscriber := total_earnings / total_subscribers

/-
Theorem: James makes $9 per month for each subscriber.
-/
theorem james_earnings_per_subscriber_is_9 : earnings_per_subscriber = 9 := by
  -- to be filled in with proof steps
  sorry

end james_earnings_per_subscriber_is_9_l2249_224922


namespace duckweed_quarter_covered_l2249_224934

theorem duckweed_quarter_covered (N : ℕ) (h1 : N = 64) (h2 : ∀ n : ℕ, n < N → (n + 1 < N) → ∃ k, k = n + 1) :
  N - 2 = 62 :=
by
  sorry

end duckweed_quarter_covered_l2249_224934


namespace value_of_y_minus_x_l2249_224905

theorem value_of_y_minus_x (x y : ℝ) (h1 : abs (x + 1) = 3) (h2 : abs y = 5) (h3 : -y / x > 0) :
  y - x = -7 ∨ y - x = 9 :=
sorry

end value_of_y_minus_x_l2249_224905


namespace part_a_l2249_224909

theorem part_a (a b : ℤ) (h : a^2 - (b^2 - 4 * b + 1) * a - (b^4 - 2 * b^3) = 0) : 
  ∃ k : ℤ, b^2 + a = k^2 :=
sorry

end part_a_l2249_224909


namespace evaluate_power_l2249_224997

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l2249_224997


namespace cost_price_equals_720_l2249_224961

theorem cost_price_equals_720 (C : ℝ) :
  (0.27 * C - 0.12 * C = 108) → (C = 720) :=
by
  sorry

end cost_price_equals_720_l2249_224961


namespace range_of_m_l2249_224971

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

def increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioi 2, 0 ≤ deriv (f m) x) ↔ m ≤ 5 / 2 :=
sorry

end range_of_m_l2249_224971


namespace b2_b7_product_l2249_224930

variable {b : ℕ → ℤ}

-- Define the conditions: b is an arithmetic sequence and b_4 * b_5 = 15
def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

axiom increasing_arithmetic_sequence : is_arithmetic_sequence b
axiom b4_b5_product : b 4 * b 5 = 15

-- The target theorem to prove
theorem b2_b7_product : b 2 * b 7 = -9 :=
sorry

end b2_b7_product_l2249_224930


namespace examination_total_students_l2249_224913

theorem examination_total_students (T : ℝ) :
  (0.35 * T + 520) = T ↔ T = 800 :=
by 
  sorry

end examination_total_students_l2249_224913


namespace reciprocal_inequality_l2249_224969

open Real

theorem reciprocal_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / a) + (1 / b) > 1 / (a + b) :=
sorry

end reciprocal_inequality_l2249_224969


namespace concentration_after_removal_l2249_224933

/-- 
Given:
1. A container has 27 liters of 40% acidic liquid.
2. 9 liters of water is removed from this container.

Prove that the concentration of the acidic liquid in the container after removal is 60%.
-/
theorem concentration_after_removal :
  let initial_volume := 27
  let initial_concentration := 0.4
  let water_removed := 9
  let pure_acid := initial_concentration * initial_volume
  let new_volume := initial_volume - water_removed
  let final_concentration := (pure_acid / new_volume) * 100
  final_concentration = 60 :=
by {
  sorry
}

end concentration_after_removal_l2249_224933


namespace proposition_D_l2249_224901

theorem proposition_D (a b : ℝ) (h : a > abs b) : a^2 > b^2 :=
by {
    sorry
}

end proposition_D_l2249_224901


namespace triangle_perimeter_problem_l2249_224978

theorem triangle_perimeter_problem : 
  ∀ (c : ℝ), 20 + 15 > c ∧ 20 + c > 15 ∧ 15 + c > 20 → ¬ (35 + c = 72) :=
by
  intros c h
  sorry

end triangle_perimeter_problem_l2249_224978
