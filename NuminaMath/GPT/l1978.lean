import Mathlib

namespace problem_inequality_l1978_197841

variable (x y z : ℝ)

theorem problem_inequality (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y := by
  sorry

end problem_inequality_l1978_197841


namespace ashley_champagne_bottles_l1978_197886

theorem ashley_champagne_bottles (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) 
  (h1 : guests = 120) (h2 : glasses_per_guest = 2) (h3 : servings_per_bottle = 6) : 
  (guests * glasses_per_guest) / servings_per_bottle = 40 :=
by
  -- The proof will go here
  sorry

end ashley_champagne_bottles_l1978_197886


namespace semicircle_radius_l1978_197801

theorem semicircle_radius (b h : ℝ) (base_eq_b : b = 16) (height_eq_h : h = 15) :
  let s := (2 * 17) / 2
  let area := 240 
  s * (r : ℝ) = area → r = 120 / 17 :=
  by
  intros s area
  sorry

end semicircle_radius_l1978_197801


namespace find_coordinates_B_l1978_197867

variable (B : ℝ × ℝ)

def A : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (0, 1)
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

theorem find_coordinates_B (h : vec A B = (-2) • vec B C) : B = (-2, 5/3) :=
by
  -- Here you would provide proof steps
  sorry

end find_coordinates_B_l1978_197867


namespace quadratic_roots_square_diff_l1978_197895

theorem quadratic_roots_square_diff (α β : ℝ) (h : α ≠ β)
    (hα : α^2 - 3 * α + 2 = 0) (hβ : β^2 - 3 * β + 2 = 0) :
    (α - β)^2 = 1 :=
sorry

end quadratic_roots_square_diff_l1978_197895


namespace winner_percentage_of_votes_l1978_197800

theorem winner_percentage_of_votes (V W O : ℕ) (W_votes : W = 720) (won_by : W - O = 240) (total_votes : V = W + O) :
  (W * 100) / V = 60 :=
by
  sorry

end winner_percentage_of_votes_l1978_197800


namespace intersection_of_lines_l1978_197813

theorem intersection_of_lines :
  ∃ x y : ℚ, 12 * x - 5 * y = 8 ∧ 10 * x + 2 * y = 20 ∧ x = 58 / 37 ∧ y = 667 / 370 :=
by
  sorry

end intersection_of_lines_l1978_197813


namespace value_of_expression_l1978_197892

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 :=
by
sorry

end value_of_expression_l1978_197892


namespace cycle_selling_price_l1978_197897

theorem cycle_selling_price (initial_price : ℝ)
  (first_discount_percent : ℝ) (second_discount_percent : ℝ) (third_discount_percent : ℝ)
  (first_discounted_price : ℝ) (second_discounted_price : ℝ) :
  initial_price = 3600 →
  first_discount_percent = 15 →
  second_discount_percent = 10 →
  third_discount_percent = 5 →
  first_discounted_price = initial_price * (1 - first_discount_percent / 100) →
  second_discounted_price = first_discounted_price * (1 - second_discount_percent / 100) →
  final_price = second_discounted_price * (1 - third_discount_percent / 100) →
  final_price = 2616.30 :=
by
  intros
  sorry

end cycle_selling_price_l1978_197897


namespace fruit_vendor_sold_fruits_l1978_197812

def total_dozen_fruits_sold (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ) : ℝ :=
  (lemons_dozen * dozen) + (avocados_dozen * dozen)

theorem fruit_vendor_sold_fruits (hl : ∀ (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ), lemons_dozen = 2.5 ∧ avocados_dozen = 5 ∧ dozen = 12) :
  total_dozen_fruits_sold 2.5 5 12 = 90 :=
by
  sorry

end fruit_vendor_sold_fruits_l1978_197812


namespace tangent_line_through_M_to_circle_l1978_197882

noncomputable def M : ℝ × ℝ := (2, -1)
noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem tangent_line_through_M_to_circle :
  ∀ {x y : ℝ}, circle_eq x y → M = (2, -1) → 2*x - y - 5 = 0 :=
sorry

end tangent_line_through_M_to_circle_l1978_197882


namespace geom_seq_common_ratio_l1978_197880

theorem geom_seq_common_ratio (a₁ a₂ a₃ a₄ q : ℝ) 
  (h1 : a₁ + a₄ = 18)
  (h2 : a₂ * a₃ = 32)
  (h3 : a₂ = a₁ * q)
  (h4 : a₃ = a₁ * q^2)
  (h5 : a₄ = a₁ * q^3) : 
  q = 2 ∨ q = (1 / 2) :=
by {
  sorry
}

end geom_seq_common_ratio_l1978_197880


namespace exists_natural_number_starting_and_ending_with_pattern_l1978_197845

theorem exists_natural_number_starting_and_ending_with_pattern (n : ℕ) : 
  ∃ (m : ℕ), 
  (m % 10 = 1) ∧ 
  (∃ t : ℕ, 
    m^2 / 10^t = 10^(n - 1) * (10^n - 1) / 9) ∧ 
  (m^2 % 10^n = 1 ∨ m^2 % 10^n = 2) :=
sorry

end exists_natural_number_starting_and_ending_with_pattern_l1978_197845


namespace tommy_writing_time_l1978_197827

def numUniqueLettersTommy : Nat := 5
def numRearrangementsPerMinute : Nat := 20
def totalRearrangements : Nat := numUniqueLettersTommy.factorial
def minutesToComplete : Nat := totalRearrangements / numRearrangementsPerMinute
def hoursToComplete : Rat := minutesToComplete / 60

theorem tommy_writing_time :
  hoursToComplete = 0.1 := by
  sorry

end tommy_writing_time_l1978_197827


namespace number_of_suits_sold_l1978_197853

theorem number_of_suits_sold
  (commission_rate: ℝ)
  (price_per_suit: ℝ)
  (price_per_shirt: ℝ)
  (price_per_loafer: ℝ)
  (number_of_shirts: ℕ)
  (number_of_loafers: ℕ)
  (total_commission: ℝ)
  (suits_sold: ℕ)
  (total_sales: ℝ)
  (total_sales_from_non_suits: ℝ)
  (sales_needed_from_suits: ℝ)
  : 
  (commission_rate = 0.15) → 
  (price_per_suit = 700.0) → 
  (price_per_shirt = 50.0) → 
  (price_per_loafer = 150.0) → 
  (number_of_shirts = 6) → 
  (number_of_loafers = 2) → 
  (total_commission = 300.0) →
  (total_sales = total_commission / commission_rate) →
  (total_sales_from_non_suits = number_of_shirts * price_per_shirt + number_of_loafers * price_per_loafer) →
  (sales_needed_from_suits = total_sales - total_sales_from_non_suits) →
  (suits_sold = sales_needed_from_suits / price_per_suit) →
  suits_sold = 2 :=
by
  sorry

end number_of_suits_sold_l1978_197853


namespace DVDs_per_season_l1978_197821

theorem DVDs_per_season (total_DVDs : ℕ) (seasons : ℕ) (h1 : total_DVDs = 40) (h2 : seasons = 5) : total_DVDs / seasons = 8 :=
by
  sorry

end DVDs_per_season_l1978_197821


namespace second_expression_l1978_197833

theorem second_expression (a x : ℕ) (h₁ : (2 * a + 16 + x) / 2 = 79) (h₂ : a = 30) : x = 82 := by
  sorry

end second_expression_l1978_197833


namespace find_missing_digit_l1978_197862

theorem find_missing_digit 
  (x : Nat) 
  (h : 16 + x ≡ 0 [MOD 9]) : 
  x = 2 :=
sorry

end find_missing_digit_l1978_197862


namespace sam_hourly_rate_l1978_197858

theorem sam_hourly_rate
  (first_month_earnings : ℕ)
  (second_month_earnings : ℕ)
  (total_hours : ℕ)
  (h1 : first_month_earnings = 200)
  (h2 : second_month_earnings = first_month_earnings + 150)
  (h3 : total_hours = 55) :
  (first_month_earnings + second_month_earnings) / total_hours = 10 := 
  by
  sorry

end sam_hourly_rate_l1978_197858


namespace Jori_water_left_l1978_197814

theorem Jori_water_left (a b : ℚ) (h1 : a = 7/2) (h2 : b = 7/4) : a - b = 7/4 := by
  sorry

end Jori_water_left_l1978_197814


namespace f_neg_one_f_eq_half_l1978_197840

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then 2^(-x) else Real.log x / Real.log 2

theorem f_neg_one : f (-1) = 2 := by
  sorry

theorem f_eq_half (x : ℝ) : f x = 1 / 2 ↔ x = Real.sqrt 2 := by
  sorry

end f_neg_one_f_eq_half_l1978_197840


namespace binomial_square_b_value_l1978_197887

theorem binomial_square_b_value (b : ℝ) (h : ∃ c : ℝ, (9 * x^2 + 24 * x + b) = (3 * x + c) ^ 2) : b = 16 :=
sorry

end binomial_square_b_value_l1978_197887


namespace water_tower_excess_consumption_l1978_197852

def water_tower_problem : Prop :=
  let initial_water := 2700
  let first_neighborhood := 300
  let second_neighborhood := 2 * first_neighborhood
  let third_neighborhood := second_neighborhood + 100
  let fourth_neighborhood := 3 * first_neighborhood
  let fifth_neighborhood := third_neighborhood / 2
  let leakage := 50
  let first_neighborhood_final := first_neighborhood + 0.10 * first_neighborhood
  let second_neighborhood_final := second_neighborhood - 0.05 * second_neighborhood
  let third_neighborhood_final := third_neighborhood + 0.10 * third_neighborhood
  let fifth_neighborhood_final := fifth_neighborhood - 0.05 * fifth_neighborhood
  let total_consumption := 
    first_neighborhood_final + second_neighborhood_final + third_neighborhood_final +
    fourth_neighborhood + fifth_neighborhood_final + leakage
  let excess_consumption := total_consumption - initial_water
  excess_consumption = 252.5

theorem water_tower_excess_consumption : water_tower_problem := by
  sorry

end water_tower_excess_consumption_l1978_197852


namespace bcm_hens_count_l1978_197875

-- Propositions representing the given conditions
def total_chickens : ℕ := 100
def bcm_ratio : ℝ := 0.20
def bcm_hens_ratio : ℝ := 0.80

-- Theorem statement: proving the number of BCM hens
theorem bcm_hens_count : (total_chickens * bcm_ratio * bcm_hens_ratio = 16) := by
  sorry

end bcm_hens_count_l1978_197875


namespace problem_p_3_l1978_197871

theorem problem_p_3 (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) (hn : n = (2^(2*p) - 1) / 3) : n ∣ 2^n - 2 := by
  sorry

end problem_p_3_l1978_197871


namespace sampling_method_sequential_is_systematic_l1978_197810

def is_sequential_ids (ids : List Nat) : Prop :=
  ids = [5, 10, 15, 20, 25, 30, 35, 40]

def is_systematic_sampling (sampling_method : Prop) : Prop :=
  sampling_method

theorem sampling_method_sequential_is_systematic :
  ∀ ids, is_sequential_ids ids → 
    is_systematic_sampling (ids = [5, 10, 15, 20, 25, 30, 35, 40]) :=
by
  intros
  apply id
  sorry

end sampling_method_sequential_is_systematic_l1978_197810


namespace number_of_associates_l1978_197870

theorem number_of_associates
  (num_managers : ℕ) 
  (avg_salary_managers : ℝ) 
  (avg_salary_associates : ℝ) 
  (avg_salary_company : ℝ)
  (total_employees : ℕ := num_managers + A) -- Adding a placeholder A for the associates
  (total_salary_company : ℝ := (num_managers * avg_salary_managers) + (A * avg_salary_associates)) 
  (average_calculation : avg_salary_company = total_salary_company / total_employees) :
  ∃ A : ℕ, A = 75 :=
by
  let A : ℕ := 75
  sorry

end number_of_associates_l1978_197870


namespace find_z_plus_one_over_y_l1978_197894

variable {x y z : ℝ}

theorem find_z_plus_one_over_y (h1 : x * y * z = 1) 
                                (h2 : x + 1 / z = 7) 
                                (h3 : y + 1 / x = 31) 
                                (h4 : 0 < x ∧ 0 < y ∧ 0 < z) : 
                              z + 1 / y = 5 / 27 := 
by
  sorry

end find_z_plus_one_over_y_l1978_197894


namespace fraction_expression_l1978_197890

theorem fraction_expression :
  (1 / 4 - 1 / 6) / (1 / 3 + 1 / 2) = 1 / 10 :=
by
  sorry

end fraction_expression_l1978_197890


namespace sum_of_sequence_l1978_197830

theorem sum_of_sequence :
  3 + 15 + 27 + 53 + 65 + 17 + 29 + 41 + 71 + 83 = 404 :=
by
  sorry

end sum_of_sequence_l1978_197830


namespace sqrt_sqrt_16_l1978_197878

theorem sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := sorry

end sqrt_sqrt_16_l1978_197878


namespace horizontal_distance_travel_l1978_197855

noncomputable def radius : ℝ := 2
noncomputable def angle_degrees : ℝ := 30
noncomputable def angle_radians : ℝ := angle_degrees * (Real.pi / 180)
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def cos_theta : ℝ := Real.cos angle_radians
noncomputable def horizontal_distance (r : ℝ) (θ : ℝ) : ℝ := (circumference r) * (Real.cos θ)

theorem horizontal_distance_travel (r : ℝ) (θ : ℝ) (h_radius : r = 2) (h_angle : θ = angle_radians) :
  horizontal_distance r θ = 2 * Real.pi * Real.sqrt 3 := 
by
  sorry

end horizontal_distance_travel_l1978_197855


namespace find_quadratic_minimum_value_l1978_197818

noncomputable def quadraticMinimumPoint (a b c : ℝ) : ℝ :=
  -b / (2 * a)

theorem find_quadratic_minimum_value :
  quadraticMinimumPoint 3 6 9 = -1 :=
by
  sorry

end find_quadratic_minimum_value_l1978_197818


namespace find_a_l1978_197873

theorem find_a (a r : ℝ) (h1 : a * r = 24) (h2 : a * r^4 = 3) : a = 48 :=
sorry

end find_a_l1978_197873


namespace remainder_of_2_pow_2005_mod_7_l1978_197869

theorem remainder_of_2_pow_2005_mod_7 :
  2 ^ 2005 % 7 = 2 :=
sorry

end remainder_of_2_pow_2005_mod_7_l1978_197869


namespace a4_value_a_n_formula_l1978_197839

theorem a4_value : a_4 = 30 := 
by
    sorry

noncomputable def a_n (n : ℕ) : ℕ :=
    (n * (n + 1)^2 * (2 * n + 1)) / 6

theorem a_n_formula (n : ℕ) : a_n n = (n * (n + 1)^2 * (2 * n + 1)) / 6 := 
by
    sorry

end a4_value_a_n_formula_l1978_197839


namespace find_g2_l1978_197861

variable (g : ℝ → ℝ)

theorem find_g2 (h : ∀ x : ℝ, g (3 * x - 7) = 5 * x + 11) : g 2 = 26 := by
  sorry

end find_g2_l1978_197861


namespace actual_average_speed_l1978_197825

theorem actual_average_speed (v t : ℝ) (h1 : v > 0) (h2: t > 0) (h3 : (t / (t - (1 / 4) * t)) = ((v + 12) / v)) : v = 36 :=
by
  sorry

end actual_average_speed_l1978_197825


namespace find_num_20_paise_coins_l1978_197864

def num_20_paise_coins (x y : ℕ) : Prop :=
  x + y = 334 ∧ 20 * x + 25 * y = 7100

theorem find_num_20_paise_coins (x y : ℕ) (h : num_20_paise_coins x y) : x = 250 :=
by
  sorry

end find_num_20_paise_coins_l1978_197864


namespace find_sum_of_digits_in_base_l1978_197811

theorem find_sum_of_digits_in_base (d A B : ℕ) (hd : d > 8) (hA : A < d) (hB : B < d) (h : (A * d + B) + (A * d + A) - (B * d + A) = 1 * d^2 + 8 * d + 0) : A + B = 10 :=
sorry

end find_sum_of_digits_in_base_l1978_197811


namespace tire_circumference_l1978_197822

variable (rpm : ℕ) (car_speed_kmh : ℕ) (circumference : ℝ)

-- Define the conditions
def conditions : Prop :=
  rpm = 400 ∧ car_speed_kmh = 24

-- Define the statement to prove
theorem tire_circumference (h : conditions rpm car_speed_kmh) : circumference = 1 :=
sorry

end tire_circumference_l1978_197822


namespace find_intervals_of_monotonicity_find_value_of_a_l1978_197829

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + a + 1

theorem find_intervals_of_monotonicity (k : ℤ) (a : ℝ) :
  ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3), MonotoneOn (λ x => f x a) (Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)) :=
sorry

theorem find_value_of_a (a : ℝ) (max_value_condition : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) :
  a = 1 :=
sorry

end find_intervals_of_monotonicity_find_value_of_a_l1978_197829


namespace remainder_3_pow_1503_mod_7_l1978_197848

theorem remainder_3_pow_1503_mod_7 : 
  (3 ^ 1503) % 7 = 6 := 
by sorry

end remainder_3_pow_1503_mod_7_l1978_197848


namespace total_trophies_after_five_years_l1978_197838

theorem total_trophies_after_five_years (michael_current_trophies : ℕ) (michael_increase : ℕ) (jack_multiplier : ℕ) (h1 : michael_current_trophies = 50) (h2 : michael_increase = 150) (h3 : jack_multiplier = 15) :
  let michael_five_years : ℕ := michael_current_trophies + michael_increase
  let jack_five_years : ℕ := jack_multiplier * michael_current_trophies
  michael_five_years + jack_five_years = 950 :=
by
  sorry

end total_trophies_after_five_years_l1978_197838


namespace area_of_triangle_l1978_197856

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (h₁ : b = 2) (h₂ : c = 2 * Real.sqrt 2) (h₃ : C = Real.pi / 4) :
  1 / 2 * b * c * Real.sin (Real.pi - B - C) = Real.sqrt 3 + 1 := 
by
  sorry

end area_of_triangle_l1978_197856


namespace certain_amount_eq_3_l1978_197854

theorem certain_amount_eq_3 (x A : ℕ) (hA : A = 5) (h : A + (11 + x) = 19) : x = 3 :=
by
  sorry

end certain_amount_eq_3_l1978_197854


namespace year_when_P_costs_40_paise_more_than_Q_l1978_197804

def price_of_P (n : ℕ) : ℝ := 4.20 + 0.40 * n
def price_of_Q (n : ℕ) : ℝ := 6.30 + 0.15 * n

theorem year_when_P_costs_40_paise_more_than_Q :
  ∃ n : ℕ, price_of_P n = price_of_Q n + 0.40 ∧ 2001 + n = 2011 :=
by
  sorry

end year_when_P_costs_40_paise_more_than_Q_l1978_197804


namespace total_sales_in_december_correct_l1978_197837

def ear_muffs_sales_in_december : ℝ :=
  let typeB_sold := 3258
  let typeB_price := 6.9
  let typeC_sold := 3186
  let typeC_price := 7.4
  let total_typeB_sales := typeB_sold * typeB_price
  let total_typeC_sales := typeC_sold * typeC_price
  total_typeB_sales + total_typeC_sales

theorem total_sales_in_december_correct :
  ear_muffs_sales_in_december = 46056.6 :=
by
  sorry

end total_sales_in_december_correct_l1978_197837


namespace solveSALE_l1978_197803

namespace Sherlocked

open Nat

def areDistinctDigits (d₁ d₂ d₃ d₄ d₅ d₆ : Nat) : Prop :=
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧ 
  d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧ 
  d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧ 
  d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧ 
  d₅ ≠ d₆

theorem solveSALE :
  ∃ (S C A L E T : ℕ),
    SCALE - SALE = SLATE ∧
    areDistinctDigits S C A L E T ∧
    S < 10 ∧ C < 10 ∧ A < 10 ∧
    L < 10 ∧ E < 10 ∧ T < 10 ∧
    SALE = 1829 :=
by
  sorry

end Sherlocked

end solveSALE_l1978_197803


namespace tangent_parabola_line_l1978_197899

theorem tangent_parabola_line (a x₀ y₀ : ℝ) 
  (h_line : x₀ - y₀ - 1 = 0)
  (h_parabola : y₀ = a * x₀^2)
  (h_tangent_slope : 2 * a * x₀ = 1) : 
  a = 1 / 4 :=
sorry

end tangent_parabola_line_l1978_197899


namespace gcd_of_polynomials_l1978_197805

theorem gcd_of_polynomials (n : ℕ) (h : n > 2^5) : gcd (n^3 + 5^2) (n + 6) = 1 :=
by sorry

end gcd_of_polynomials_l1978_197805


namespace inequality_proof_l1978_197846

theorem inequality_proof (s r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by
  sorry

end inequality_proof_l1978_197846


namespace inequality_solution_l1978_197851

theorem inequality_solution (x : ℝ) : (x < -4 ∨ x > -4) → (x + 3) / (x + 4) > (2 * x + 7) / (3 * x + 12) :=
by
  intro h
  sorry

end inequality_solution_l1978_197851


namespace discount_correct_l1978_197815

-- Define the prices of items and the total amount paid
def t_shirt_price : ℕ := 30
def backpack_price : ℕ := 10
def cap_price : ℕ := 5
def total_paid : ℕ := 43

-- Define the total cost before discount
def total_cost := t_shirt_price + backpack_price + cap_price

-- Define the discount
def discount := total_cost - total_paid

-- Prove that the discount is 2 dollars
theorem discount_correct : discount = 2 :=
by
  -- We need to prove that (30 + 10 + 5) - 43 = 2
  sorry

end discount_correct_l1978_197815


namespace annalise_total_cost_correct_l1978_197876

-- Define the constants from the problem
def boxes : ℕ := 25
def packs_per_box : ℕ := 18
def tissues_per_pack : ℕ := 150
def tissue_price : ℝ := 0.06
def discount_per_box : ℝ := 0.10
def volume_discount : ℝ := 0.08
def tax_rate : ℝ := 0.05

-- Calculate the total number of tissues
def total_tissues : ℕ := boxes * packs_per_box * tissues_per_pack

-- Calculate the total cost without any discounts
def initial_cost : ℝ := total_tissues * tissue_price

-- Apply the 10% discount on the price of the total packs in each box purchased
def cost_after_box_discount : ℝ := initial_cost * (1 - discount_per_box)

-- Apply the 8% volume discount for buying 10 or more boxes
def cost_after_volume_discount : ℝ := cost_after_box_discount * (1 - volume_discount)

-- Apply the 5% tax on the final price after all discounts
def final_cost : ℝ := cost_after_volume_discount * (1 + tax_rate)

-- Define the expected final cost
def expected_final_cost : ℝ := 3521.07

-- Proof statement
theorem annalise_total_cost_correct : final_cost = expected_final_cost := by
  -- Sorry is used as placeholder for the actual proof
  sorry

end annalise_total_cost_correct_l1978_197876


namespace european_stamp_costs_l1978_197832

theorem european_stamp_costs :
  let P_Italy := 0.07
  let P_Germany := 0.03
  let N_Italy := 9
  let N_Germany := 15
  N_Italy * P_Italy + N_Germany * P_Germany = 1.08 :=
by
  sorry

end european_stamp_costs_l1978_197832


namespace mason_grandmother_age_l1978_197866

-- Defining the ages of Mason, Sydney, Mason's father, and Mason's grandmother
def mason_age : ℕ := 20

def sydney_age (S : ℕ) : Prop :=
  mason_age = S / 3

def father_age (S F : ℕ) : Prop :=
  F = S + 6

def grandmother_age (F G : ℕ) : Prop :=
  G = 2 * F

theorem mason_grandmother_age (S F G : ℕ) (h1 : sydney_age S) (h2 : father_age S F) (h3 : grandmother_age F G) : G = 132 :=
by
  -- leaving the proof as a sorry
  sorry

end mason_grandmother_age_l1978_197866


namespace deepak_current_age_l1978_197860

theorem deepak_current_age (x : ℕ) (rahul_age deepak_age : ℕ) :
  (rahul_age = 4 * x) →
  (deepak_age = 3 * x) →
  (rahul_age + 10 = 26) →
  deepak_age = 12 :=
by
  intros h1 h2 h3
  -- You would write the proof here
  sorry

end deepak_current_age_l1978_197860


namespace min_transport_cost_l1978_197819

theorem min_transport_cost :
  let large_truck_capacity := 7
  let large_truck_cost := 600
  let small_truck_capacity := 4
  let small_truck_cost := 400
  let total_goods := 20
  ∃ (n_large n_small : ℕ),
    n_large * large_truck_capacity + n_small * small_truck_capacity ≥ total_goods ∧ 
    (n_large * large_truck_cost + n_small * small_truck_cost) = 1800 :=
sorry

end min_transport_cost_l1978_197819


namespace sam_compound_interest_l1978_197884

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ := 
  P * (1 + r / n) ^ (n * t)

theorem sam_compound_interest : 
  compound_interest 3000 0.10 2 1 = 3307.50 :=
by
  sorry

end sam_compound_interest_l1978_197884


namespace terminal_side_alpha_minus_beta_nonneg_x_axis_l1978_197885

theorem terminal_side_alpha_minus_beta_nonneg_x_axis
  (α β : ℝ) (k : ℤ) (h : α = k * 360 + β) : 
  (∃ m : ℤ, α - β = m * 360) := 
sorry

end terminal_side_alpha_minus_beta_nonneg_x_axis_l1978_197885


namespace platform_length_eq_train_length_l1978_197857

noncomputable def length_of_train : ℝ := 900
noncomputable def speed_of_train_kmh : ℝ := 108
noncomputable def speed_of_train_mpm : ℝ := (speed_of_train_kmh * 1000) / 60
noncomputable def crossing_time_min : ℝ := 1
noncomputable def total_distance_covered : ℝ := speed_of_train_mpm * crossing_time_min

theorem platform_length_eq_train_length :
  total_distance_covered - length_of_train = length_of_train :=
by
  sorry

end platform_length_eq_train_length_l1978_197857


namespace tangent_line_at_point_l1978_197806

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = Real.exp x - 2 * x) (h_point : (0, 1) = (x, y)) :
  x + y - 1 = 0 := 
by 
  sorry

end tangent_line_at_point_l1978_197806


namespace number_of_cars_l1978_197888

theorem number_of_cars (people_per_car : ℝ) (total_people : ℝ) (h1 : people_per_car = 63.0) (h2 : total_people = 189) : total_people / people_per_car = 3 := by
  sorry

end number_of_cars_l1978_197888


namespace problem1_problem2_l1978_197843

variable (α : ℝ)

-- Equivalent problem 1
theorem problem1 (h : Real.tan α = 7) : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 13 := 
  sorry

-- Equivalent problem 2
theorem problem2 (h : Real.tan α = 7) : Real.sin α * Real.cos α = 7 / 50 := 
  sorry

end problem1_problem2_l1978_197843


namespace monotonic_intervals_inequality_condition_l1978_197891

noncomputable def f (x : ℝ) (m : ℝ) := Real.log x - m * x

theorem monotonic_intervals (m : ℝ) :
  (m ≤ 0 → ∀ x > 0, ∀ y > 0, x < y → f x m < f y m) ∧
  (m > 0 → (∀ x > 0, x < 1/m → ∀ y > x, y < 1/m → f x m < f y m) ∧ (∀ x ≥ 1/m, ∀ y > x, f x m > f y m)) :=
sorry

theorem inequality_condition (m : ℝ) (h : ∀ x ≥ 1, f x m ≤ (m - 1) / x - 2 * m + 1) :
  m ≥ 1/2 :=
sorry

end monotonic_intervals_inequality_condition_l1978_197891


namespace find_dividend_l1978_197826

noncomputable def quotient : ℕ := 2015
noncomputable def remainder : ℕ := 0
noncomputable def divisor : ℕ := 105

theorem find_dividend : quotient * divisor + remainder = 20685 := by
  sorry

end find_dividend_l1978_197826


namespace vending_machine_problem_l1978_197896

variable (x n : ℕ)

theorem vending_machine_problem (h : 25 * x + 10 * 15 + 5 * 30 = 25 * 25 + 10 * 5 + 5 * n) (hx : x = 25) :
  n = 50 := by
sorry

end vending_machine_problem_l1978_197896


namespace evaluate_expression_l1978_197842

theorem evaluate_expression (a b x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
    (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  sorry

end evaluate_expression_l1978_197842


namespace larger_page_sum_137_l1978_197889

theorem larger_page_sum_137 (x y : ℕ) (h1 : x + y = 137) (h2 : y = x + 1) : y = 69 :=
sorry

end larger_page_sum_137_l1978_197889


namespace fraction_a_over_b_l1978_197879

theorem fraction_a_over_b (x y a b : ℝ) (h1 : 2 * x - y = a) (h2 : 3 * y - 6 * x = b) (hb : b ≠ 0) : a / b = -1 / 3 :=
by
  sorry

end fraction_a_over_b_l1978_197879


namespace find_a10_l1978_197834

def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

theorem find_a10 
  (a1 d : ℤ)
  (h_condition : a1 + (a1 + 18 * d) = -18) :
  arithmetic_sequence a1 d 10 = -9 := 
by
  sorry

end find_a10_l1978_197834


namespace exam_time_ratio_l1978_197828

-- Lean statements to define the problem conditions and goal
theorem exam_time_ratio (x M : ℝ) (h1 : x > 0) (h2 : M = x / 18) : 
  (5 * x / 6 + 2 * M) / (x / 6 - 2 * M) = 17 := by
  sorry

end exam_time_ratio_l1978_197828


namespace tangency_point_exists_l1978_197820

theorem tangency_point_exists :
  ∃ (x y : ℝ), y = x^2 + 18 * x + 47 ∧ x = y^2 + 36 * y + 323 ∧ x = -17 / 2 ∧ y = -35 / 2 :=
by
  sorry

end tangency_point_exists_l1978_197820


namespace positive_value_of_m_l1978_197863

theorem positive_value_of_m (m : ℝ) (h : (64 * m^2 - 60 * m) = 0) : m = 15 / 16 :=
sorry

end positive_value_of_m_l1978_197863


namespace credit_card_balance_l1978_197817

theorem credit_card_balance :
  ∀ (initial_balance groceries_charge gas_charge return_credit : ℕ),
  initial_balance = 126 →
  groceries_charge = 60 →
  gas_charge = groceries_charge / 2 →
  return_credit = 45 →
  initial_balance + groceries_charge + gas_charge - return_credit = 171 :=
by
  intros initial_balance groceries_charge gas_charge return_credit
  intros h_initial h_groceries h_gas h_return
  rw [h_initial, h_groceries, h_gas, h_return]
  norm_num
  sorry

end credit_card_balance_l1978_197817


namespace range_of_m_to_satisfy_quadratic_l1978_197849

def quadratic_positive_forall_m (m : ℝ) : Prop :=
  ∀ x : ℝ, m * x^2 + m * x + 100 > 0

theorem range_of_m_to_satisfy_quadratic :
  {m : ℝ | quadratic_positive_forall_m m} = {m : ℝ | 0 ≤ m ∧ m < 400} :=
by
  sorry

end range_of_m_to_satisfy_quadratic_l1978_197849


namespace largest_integer_divides_difference_l1978_197831

theorem largest_integer_divides_difference (n : ℕ) 
  (h_composite : ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n) :
  6 ∣ (n^4 - n) :=
sorry

end largest_integer_divides_difference_l1978_197831


namespace problem1_problem2_problem3_l1978_197847

theorem problem1 : 2013^2 - 2012 * 2014 = 1 := 
by 
  sorry

variables (m n : ℤ)

theorem problem2 : ((m-n)^6 / (n-m)^4) * (m-n)^3 = (m-n)^5 :=
by 
  sorry

variables (a b c : ℤ)

theorem problem3 : (a - 2*b + 3*c) * (a - 2*b - 3*c) = a^2 - 4*a*b + 4*b^2 - 9*c^2 :=
by 
  sorry

end problem1_problem2_problem3_l1978_197847


namespace marathon_finishers_l1978_197808

-- Define the conditions
def totalParticipants : ℕ := 1250
def peopleGaveUp (F : ℕ) : ℕ := F + 124

-- Define the final statement to be proved
theorem marathon_finishers (F : ℕ) (h1 : totalParticipants = F + peopleGaveUp F) : F = 563 :=
by sorry

end marathon_finishers_l1978_197808


namespace orchestra_club_members_l1978_197859

theorem orchestra_club_members : ∃ (n : ℕ), 150 < n ∧ n < 250 ∧ n % 8 = 1 ∧ n % 6 = 2 ∧ n % 9 = 3 ∧ n = 169 := 
by {
  sorry
}

end orchestra_club_members_l1978_197859


namespace halfway_fraction_l1978_197807

theorem halfway_fraction (a b : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 7) :
  ((a + b) / 2) = 41 / 56 :=
by
  rw [h_a, h_b]
  sorry

end halfway_fraction_l1978_197807


namespace rhombus_longer_diagonal_length_l1978_197874

theorem rhombus_longer_diagonal_length (side_length : ℕ) (shorter_diagonal : ℕ) 
  (h_side : side_length = 65) (h_short_diag : shorter_diagonal = 56) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 118 :=
by
  sorry

end rhombus_longer_diagonal_length_l1978_197874


namespace imaginary_part_of_complex_num_l1978_197816

def imaginary_unit : ℂ := Complex.I

noncomputable def complex_num : ℂ := 10 * imaginary_unit / (1 - 2 * imaginary_unit)

theorem imaginary_part_of_complex_num : complex_num.im = 2 := by
  sorry

end imaginary_part_of_complex_num_l1978_197816


namespace optionC_is_correct_l1978_197868

def KalobsWindowLength : ℕ := 50
def KalobsWindowWidth : ℕ := 80
def KalobsWindowArea : ℕ := KalobsWindowLength * KalobsWindowWidth

def DoubleKalobsWindowArea : ℕ := 2 * KalobsWindowArea

def optionC_Length : ℕ := 50
def optionC_Width : ℕ := 160
def optionC_Area : ℕ := optionC_Length * optionC_Width

theorem optionC_is_correct : optionC_Area = DoubleKalobsWindowArea := by
  sorry

end optionC_is_correct_l1978_197868


namespace gear_q_revolutions_per_minute_is_40_l1978_197836

-- Definitions corresponding to conditions
def gear_p_revolutions_per_minute : ℕ := 10
def gear_q_revolutions_per_minute (r : ℕ) : Prop :=
  ∃ (r : ℕ), (r * 20 / 60) - (10 * 20 / 60) = 10

-- Statement we need to prove
theorem gear_q_revolutions_per_minute_is_40 :
  gear_q_revolutions_per_minute 40 :=
sorry

end gear_q_revolutions_per_minute_is_40_l1978_197836


namespace find_parenthesis_value_l1978_197844

theorem find_parenthesis_value (x : ℝ) (h : x * (-2/3) = 2) : x = -3 :=
by
  sorry

end find_parenthesis_value_l1978_197844


namespace carl_speed_l1978_197802

theorem carl_speed 
  (time : ℝ) (distance : ℝ) 
  (h_time : time = 5) 
  (h_distance : distance = 10) 
  : (distance / time) = 2 :=
by
  rw [h_time, h_distance]
  sorry

end carl_speed_l1978_197802


namespace optimal_years_minimize_cost_l1978_197823

noncomputable def initial_cost : ℝ := 150000
noncomputable def annual_expenses (n : ℕ) : ℝ := 15000 * n
noncomputable def maintenance_cost (n : ℕ) : ℝ := (n * (3000 + 3000 * n)) / 2
noncomputable def total_cost (n : ℕ) : ℝ := initial_cost + annual_expenses n + maintenance_cost n
noncomputable def average_annual_cost (n : ℕ) : ℝ := total_cost n / n

theorem optimal_years_minimize_cost : ∀ n : ℕ, n = 10 ↔ average_annual_cost 10 ≤ average_annual_cost n :=
by sorry

end optimal_years_minimize_cost_l1978_197823


namespace banana_price_l1978_197824

theorem banana_price (x y : ℕ) (b : ℕ) 
  (hx : x + y = 4) 
  (cost_eq : 50 * x + 60 * y + b = 275) 
  (banana_cheaper_than_pear : b < 60) 
  : b = 35 ∨ b = 45 ∨ b = 55 :=
by
  sorry

end banana_price_l1978_197824


namespace length_of_shop_proof_l1978_197893

-- Given conditions
def monthly_rent : ℝ := 1440
def width : ℝ := 20
def annual_rent_per_sqft : ℝ := 48

-- Correct answer to be proved
def length_of_shop : ℝ := 18

-- The following statement is the proof problem in Lean 4
theorem length_of_shop_proof (h1 : monthly_rent = 1440) 
                            (h2 : width = 20) 
                            (h3 : annual_rent_per_sqft = 48) : 
  length_of_shop = 18 := 
  sorry

end length_of_shop_proof_l1978_197893


namespace ratio_owners_on_horse_l1978_197881

-- Definitions based on the given conditions.
def number_of_horses : Nat := 12
def number_of_owners : Nat := 12
def total_legs_walking_on_ground : Nat := 60
def owner_leg_count : Nat := 2
def horse_leg_count : Nat := 4
def total_owners_leg_horse_count : Nat := owner_leg_count + horse_leg_count

-- Prove the ratio of the number of owners on their horses' back to the total number of owners is 1:6
theorem ratio_owners_on_horse (R W : Nat) 
  (h1 : R + W = number_of_owners)
  (h2 : total_owners_leg_horse_count * W = total_legs_walking_on_ground) :
  R = 2 → W = 10 → (R : Nat)/(number_of_owners : Nat) = (1 : Nat)/(6 : Nat) := 
sorry

end ratio_owners_on_horse_l1978_197881


namespace no_integer_solutions_system_l1978_197809

theorem no_integer_solutions_system :
  ¬∃ (x y z : ℤ), x^6 + x^3 + x^3 * y + y = 147^157 ∧ x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := 
sorry

end no_integer_solutions_system_l1978_197809


namespace fergus_entry_exit_l1978_197877

theorem fergus_entry_exit (n : ℕ) (hn : n = 8) : 
  n * (n - 1) = 56 := 
by
  sorry

end fergus_entry_exit_l1978_197877


namespace triangle_inequality_sine_three_times_equality_sine_three_times_lower_bound_equality_sine_three_times_upper_bound_l1978_197865

noncomputable def sum_sine_3A_3B_3C (A B C : ℝ) : ℝ :=
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C)

theorem triangle_inequality_sine_three_times {A B C : ℝ} (h : A + B + C = Real.pi) (hA : 0 ≤ A) (hB : 0 ≤ B) (hC : 0 ≤ C) : 
  (-2 : ℝ) ≤ sum_sine_3A_3B_3C A B C ∧ sum_sine_3A_3B_3C A B C ≤ (3 * Real.sqrt 3 / 2) :=
by
  sorry

theorem equality_sine_three_times_lower_bound {A B C : ℝ} (h : A + B + C = Real.pi) (h1: A = 0) (h2: B = Real.pi / 2) (h3: C = Real.pi / 2) :
  sum_sine_3A_3B_3C A B C = -2 :=
by
  sorry

theorem equality_sine_three_times_upper_bound {A B C : ℝ} (h : A + B + C = Real.pi) (h1: A = Real.pi / 3) (h2: B = Real.pi / 3) (h3: C = Real.pi / 3) :
  sum_sine_3A_3B_3C A B C = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_inequality_sine_three_times_equality_sine_three_times_lower_bound_equality_sine_three_times_upper_bound_l1978_197865


namespace trig_expression_zero_l1978_197850

theorem trig_expression_zero (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 :=
sorry

end trig_expression_zero_l1978_197850


namespace first_number_eq_l1978_197835

theorem first_number_eq (x y : ℝ) (h1 : x * 120 = 346) (h2 : y * 240 = 346) : x = 346 / 120 :=
by
  -- The final proof will be inserted here
  sorry

end first_number_eq_l1978_197835


namespace price_of_shoes_on_Monday_l1978_197898

noncomputable def priceOnThursday : ℝ := 50

noncomputable def increasedPriceOnFriday : ℝ := priceOnThursday * 1.2

noncomputable def discountedPriceOnMonday : ℝ := increasedPriceOnFriday * 0.85

noncomputable def finalPriceOnMonday : ℝ := discountedPriceOnMonday * 1.05

theorem price_of_shoes_on_Monday :
  finalPriceOnMonday = 53.55 :=
by
  sorry

end price_of_shoes_on_Monday_l1978_197898


namespace ratio_cher_to_gab_l1978_197883

-- Definitions based on conditions
def sammy_score : ℕ := 20
def gab_score : ℕ := 2 * sammy_score
def opponent_score : ℕ := 85
def total_points : ℕ := opponent_score + 55
def cher_score : ℕ := total_points - (sammy_score + gab_score)

-- Theorem to prove the ratio of Cher's score to Gab's score
theorem ratio_cher_to_gab : cher_score / gab_score = 2 := by
  sorry

end ratio_cher_to_gab_l1978_197883


namespace sufficiency_of_p_for_q_not_necessity_of_p_for_q_l1978_197872

noncomputable def p (m : ℝ) := ∀ x : ℝ, |x| + |x - 1| > m
noncomputable def q (m : ℝ) := ∀ x : ℝ, (- (5 - 2 * m)) ^ x < 0

theorem sufficiency_of_p_for_q : ∀ m : ℝ, (m < 1 → m < 2) :=
by sorry

theorem not_necessity_of_p_for_q : ∀ m : ℝ, ¬ (m < 2 → m < 1) :=
by sorry

end sufficiency_of_p_for_q_not_necessity_of_p_for_q_l1978_197872
