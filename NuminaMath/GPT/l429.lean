import Mathlib

namespace NUMINAMATH_GPT_alpha3_plus_8beta_plus_6_eq_30_l429_42923

noncomputable def alpha_beta_quad_roots (α β : ℝ) : Prop :=
  α^2 - 2 * α - 4 = 0 ∧ β^2 - 2 * β - 4 = 0

theorem alpha3_plus_8beta_plus_6_eq_30 (α β : ℝ) (h : alpha_beta_quad_roots α β) : 
  α^3 + 8 * β + 6 = 30 :=
sorry

end NUMINAMATH_GPT_alpha3_plus_8beta_plus_6_eq_30_l429_42923


namespace NUMINAMATH_GPT_find_x_l429_42973

theorem find_x 
  (x : ℝ) 
  (h1 : 0 < x)
  (h2 : x < π / 2)
  (h3 : 1 / (Real.sin x) = 1 / (Real.sin (2 * x)) + 1 / (Real.sin (4 * x)) + 1 / (Real.sin (8 * x))) : 
  x = π / 15 ∨ x = π / 5 ∨ x = π / 3 ∨ x = 7 * π / 15 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l429_42973


namespace NUMINAMATH_GPT_find_k_l429_42921

open BigOperators

noncomputable
def hyperbola_property (k : ℝ) (x a b c : ℝ) : Prop :=
  k > 0 ∧
  (a / 2, b / 2) = (a / 2, k / a / 2) ∧ -- midpoint condition
  abs (a * b) / 2 = 3 ∧                -- area condition
  b = k / a                            -- point B on the hyperbola

theorem find_k (k : ℝ) (x a b c : ℝ) : hyperbola_property k x a b c → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l429_42921


namespace NUMINAMATH_GPT_smallest_digit_for_divisibility_by_9_l429_42963

theorem smallest_digit_for_divisibility_by_9 : 
  ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ (18 + d) % 9 = 0 ∧ ∀ d' : ℕ, (0 ≤ d' ∧ d' ≤ 9 ∧ (18 + d') % 9 = 0) → d' ≥ d :=
sorry

end NUMINAMATH_GPT_smallest_digit_for_divisibility_by_9_l429_42963


namespace NUMINAMATH_GPT_exponential_inequality_l429_42913

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^a * b^b ≥ (a * b)^((a + b) / 2) :=
sorry

end NUMINAMATH_GPT_exponential_inequality_l429_42913


namespace NUMINAMATH_GPT_math_problem_l429_42957

theorem math_problem (a b c : ℝ) (h1 : (a + b) / 2 = 30) (h2 : (b + c) / 2 = 60) (h3 : c - a = 60) : c - a = 60 :=
by
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_math_problem_l429_42957


namespace NUMINAMATH_GPT_karl_drove_420_miles_l429_42996

theorem karl_drove_420_miles :
  ∀ (car_mileage_per_gallon : ℕ)
    (tank_capacity : ℕ)
    (initial_drive_miles : ℕ)
    (gas_purchased : ℕ)
    (destination_tank_fraction : ℚ),
    car_mileage_per_gallon = 30 →
    tank_capacity = 16 →
    initial_drive_miles = 420 →
    gas_purchased = 10 →
    destination_tank_fraction = 3 / 4 →
    initial_drive_miles + (destination_tank_fraction * tank_capacity - (tank_capacity - (initial_drive_miles / car_mileage_per_gallon)) + gas_purchased) * car_mileage_per_gallon = 420 :=
by
  intros car_mileage_per_gallon tank_capacity initial_drive_miles gas_purchased destination_tank_fraction
  intro h1 -- car_mileage_per_gallon = 30
  intro h2 -- tank_capacity = 16
  intro h3 -- initial_drive_miles = 420
  intro h4 -- gas_purchased = 10
  intro h5 -- destination_tank_fraction = 3 / 4
  sorry

end NUMINAMATH_GPT_karl_drove_420_miles_l429_42996


namespace NUMINAMATH_GPT_rosie_pies_l429_42924

-- Definition of known conditions
def apples_per_pie (apples_pies_ratio : ℕ × ℕ) : ℕ :=
  apples_pies_ratio.1 / apples_pies_ratio.2

def pies_from_apples (total_apples : ℕ) (apples_per_pie : ℕ) : ℕ :=
  total_apples / apples_per_pie

-- Theorem statement
theorem rosie_pies (apples_pies_ratio : ℕ × ℕ) (total_apples : ℕ) :
  apples_pies_ratio = (12, 3) →
  total_apples = 36 →
  pies_from_apples total_apples (apples_per_pie apples_pies_ratio) = 9 :=
by
  intros h_ratio h_apples
  rw [h_ratio, h_apples]
  sorry

end NUMINAMATH_GPT_rosie_pies_l429_42924


namespace NUMINAMATH_GPT_new_ratio_first_term_l429_42914

theorem new_ratio_first_term (x : ℕ) (r1 r2 : ℕ) (new_r1 : ℕ) :
  r1 = 4 → r2 = 15 → x = 29 → new_r1 = r1 + x → new_r1 = 33 :=
by
  intros h_r1 h_r2 h_x h_new_r1
  rw [h_r1, h_x] at h_new_r1
  exact h_new_r1

end NUMINAMATH_GPT_new_ratio_first_term_l429_42914


namespace NUMINAMATH_GPT_value_of_squares_l429_42993

-- Define the conditions
variables (p q : ℝ)

-- State the theorem with the given conditions and the proof goal
theorem value_of_squares (h1 : p * q = 12) (h2 : p + q = 8) : p ^ 2 + q ^ 2 = 40 :=
sorry

end NUMINAMATH_GPT_value_of_squares_l429_42993


namespace NUMINAMATH_GPT_inscribed_circle_radius_of_rhombus_l429_42979

theorem inscribed_circle_radius_of_rhombus (d1 d2 : ℝ) (a r : ℝ) : 
  d1 = 15 → d2 = 24 → a = Real.sqrt ((15 / 2)^2 + (24 / 2)^2) → 
  (d1 * d2) / 2 = 2 * a * r → 
  r = 60.07 / 13 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_of_rhombus_l429_42979


namespace NUMINAMATH_GPT_angle_C_in_triangle_l429_42925

open Real

noncomputable def determine_angle_C (A B C: ℝ) (AB BC: ℝ) : Prop :=
  A = (3 * π) / 4 ∧ BC = sqrt 2 * AB → C = π / 6

-- Define the theorem to state the problem
theorem angle_C_in_triangle (A B C : ℝ) (AB BC : ℝ) :
  determine_angle_C A B C AB BC := 
by
  -- Step to indicate where the proof would be
  sorry

end NUMINAMATH_GPT_angle_C_in_triangle_l429_42925


namespace NUMINAMATH_GPT_largest_perimeter_triangle_l429_42953

theorem largest_perimeter_triangle :
  ∃ (y : ℤ), 4 < y ∧ y < 20 ∧ 8 + 12 + y = 39 :=
by {
  -- we'll skip the proof steps
  sorry 
}

end NUMINAMATH_GPT_largest_perimeter_triangle_l429_42953


namespace NUMINAMATH_GPT_count_digit_2_in_range_1_to_1000_l429_42927

theorem count_digit_2_in_range_1_to_1000 :
  let count_digit_occur (digit : ℕ) (range_end : ℕ) : ℕ :=
    (range_end + 1).digits 10
    |>.count digit
  count_digit_occur 2 1000 = 300 :=
by
  sorry

end NUMINAMATH_GPT_count_digit_2_in_range_1_to_1000_l429_42927


namespace NUMINAMATH_GPT_total_amount_spent_l429_42970

def cost_per_dozen_apples : ℕ := 40
def cost_per_dozen_pears : ℕ := 50
def dozens_apples : ℕ := 14
def dozens_pears : ℕ := 14

theorem total_amount_spent : (dozens_apples * cost_per_dozen_apples + dozens_pears * cost_per_dozen_pears) = 1260 := 
  by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l429_42970


namespace NUMINAMATH_GPT_overall_profit_no_discount_l429_42905

theorem overall_profit_no_discount:
  let C_b := 100
  let C_p := 100
  let C_n := 100
  let profit_b := 42.5 / 100
  let profit_p := 35 / 100
  let profit_n := 20 / 100
  let S_b := C_b + (C_b * profit_b)
  let S_p := C_p + (C_p * profit_p)
  let S_n := C_n + (C_n * profit_n)
  let TCP := C_b + C_p + C_n
  let TSP := S_b + S_p + S_n
  let OverallProfit := TSP - TCP
  let OverallProfitPercentage := (OverallProfit / TCP) * 100
  OverallProfitPercentage = 32.5 :=
by sorry

end NUMINAMATH_GPT_overall_profit_no_discount_l429_42905


namespace NUMINAMATH_GPT_leading_digits_sum_l429_42920

-- Define the conditions
def M : ℕ := (888888888888888888888888888888888888888888888888888888888888888888888888888888) -- define the 400-digit number
-- Assume the function g(r) which finds the leading digit of the r-th root of M

/-- 
  Function g(r) definition:
  It extracts the leading digit of the r-th root of the given number M.
-/
noncomputable def g (r : ℕ) : ℕ := sorry

-- Define the problem statement in Lean 4
theorem leading_digits_sum :
  g 3 + g 4 + g 5 + g 6 + g 7 = 8 :=
sorry

end NUMINAMATH_GPT_leading_digits_sum_l429_42920


namespace NUMINAMATH_GPT_fifth_bowler_points_l429_42916

variable (P1 P2 P3 P4 P5 : ℝ)
variable (h1 : P1 = (5 / 12) * P3)
variable (h2 : P2 = (5 / 3) * P3)
variable (h3 : P4 = (5 / 3) * P3)
variable (h4 : P5 = (50 / 27) * P3)
variable (h5 : P3 ≤ 500)
variable (total_points : P1 + P2 + P3 + P4 + P5 = 2000)

theorem fifth_bowler_points : P5 = 561 :=
  sorry

end NUMINAMATH_GPT_fifth_bowler_points_l429_42916


namespace NUMINAMATH_GPT_total_apples_correct_l429_42999

variable (X : ℕ)

def Sarah_apples : ℕ := X

def Jackie_apples : ℕ := 2 * Sarah_apples X

def Adam_apples : ℕ := Jackie_apples X + 5

def total_apples : ℕ := Sarah_apples X + Jackie_apples X + Adam_apples X

theorem total_apples_correct : total_apples X = 5 * X + 5 := by
  sorry

end NUMINAMATH_GPT_total_apples_correct_l429_42999


namespace NUMINAMATH_GPT_raise_salary_to_original_l429_42992

/--
The salary of a person was reduced by 25%. By what percent should his reduced salary be raised
so as to bring it at par with his original salary?
-/
theorem raise_salary_to_original (S : ℝ) (h : S > 0) :
  ∃ P : ℝ, 0.75 * S * (1 + P / 100) = S ∧ P = 33.333333333333336 :=
sorry

end NUMINAMATH_GPT_raise_salary_to_original_l429_42992


namespace NUMINAMATH_GPT_monotone_intervals_max_floor_a_l429_42922

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + a

theorem monotone_intervals (a : ℝ) (h : a = 1) :
  (∀ x, 0 < x ∧ x < 1 → deriv (λ x => f x 1) x > 0) ∧
  (∀ x, 1 ≤ x → deriv (λ x => f x 1) x < 0) :=
by
  sorry

theorem max_floor_a (a : ℝ) (h : ∀ x > 0, f x a ≤ x) : ⌊a⌋ = 1 :=
by
  sorry

end NUMINAMATH_GPT_monotone_intervals_max_floor_a_l429_42922


namespace NUMINAMATH_GPT_decreasing_number_4312_max_decreasing_number_divisible_by_9_l429_42904

-- Definitions and conditions
def is_decreasing_number (n : ℕ) : Prop :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ d4 ≠ 0 ∧
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧
  (10 * d1 + d2 - (10 * d2 + d3) = 10 * d3 + d4)

def is_divisible_by_9 (n m : ℕ) : Prop :=
  (n + m) % 9 = 0

-- Theorem Statements
theorem decreasing_number_4312 : 
  is_decreasing_number 4312 :=
sorry

theorem max_decreasing_number_divisible_by_9 : 
  ∀ n, is_decreasing_number n ∧ is_divisible_by_9 (n / 10) (n % 1000) → n ≤ 8165 :=
sorry

end NUMINAMATH_GPT_decreasing_number_4312_max_decreasing_number_divisible_by_9_l429_42904


namespace NUMINAMATH_GPT_sum_of_pos_real_solutions_l429_42941

open Real

noncomputable def cos_equation_sum_pos_real_solutions : ℝ := 1082 * π

theorem sum_of_pos_real_solutions :
  ∃ x : ℝ, (0 < x) ∧ 
    (∀ x, 2 * cos (2 * x) * (cos (2 * x) - cos ((2016 * π ^ 2) / x)) = cos (6 * x) - 1) → 
      x = cos_equation_sum_pos_real_solutions :=
sorry

end NUMINAMATH_GPT_sum_of_pos_real_solutions_l429_42941


namespace NUMINAMATH_GPT_find_cost_price_l429_42948

theorem find_cost_price
  (cost_price : ℝ)
  (increase_rate : ℝ := 0.2)
  (decrease_rate : ℝ := 0.1)
  (profit : ℝ := 8):
  (1 + increase_rate) * cost_price * (1 - decrease_rate) - cost_price = profit → 
  cost_price = 100 := 
by 
  sorry

end NUMINAMATH_GPT_find_cost_price_l429_42948


namespace NUMINAMATH_GPT_Dan_speed_must_exceed_45_mph_l429_42966

theorem Dan_speed_must_exceed_45_mph : 
  ∀ (distance speed_Cara time_lag time_required speed_Dan : ℝ),
    distance = 180 →
    speed_Cara = 30 →
    time_lag = 2 →
    time_required = 4 →
    (distance / speed_Cara) = 6 →
    (∀ t, t = distance / speed_Dan → t < time_required) →
    speed_Dan > 45 :=
by
  intro distance speed_Cara time_lag time_required speed_Dan
  intro h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_Dan_speed_must_exceed_45_mph_l429_42966


namespace NUMINAMATH_GPT_additional_charge_per_minute_atlantic_call_l429_42951

def base_rate_U : ℝ := 11.0
def rate_per_minute_U : ℝ := 0.25
def base_rate_A : ℝ := 12.0
def call_duration : ℝ := 20.0
variable (rate_per_minute_A : ℝ)

theorem additional_charge_per_minute_atlantic_call :
  base_rate_U + rate_per_minute_U * call_duration = base_rate_A + rate_per_minute_A * call_duration →
  rate_per_minute_A = 0.20 := by
  sorry

end NUMINAMATH_GPT_additional_charge_per_minute_atlantic_call_l429_42951


namespace NUMINAMATH_GPT_restaurant_pizzas_more_than_hotdogs_l429_42947

theorem restaurant_pizzas_more_than_hotdogs
  (H P : ℕ) 
  (h1 : H = 60)
  (h2 : 30 * (P + H) = 4800) :
  P - H = 40 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_pizzas_more_than_hotdogs_l429_42947


namespace NUMINAMATH_GPT_water_remaining_45_days_l429_42981

-- Define the initial conditions and the evaporation rate
def initial_volume : ℕ := 400
def evaporation_rate : ℕ := 1
def days : ℕ := 45

-- Define a function to compute the remaining water volume
def remaining_volume (initial_volume : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_volume - (evaporation_rate * days)

-- Theorem stating that the water remaining after 45 days is 355 gallons
theorem water_remaining_45_days : remaining_volume 400 1 45 = 355 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_water_remaining_45_days_l429_42981


namespace NUMINAMATH_GPT_competition_participants_solved_all_three_l429_42931

theorem competition_participants_solved_all_three
  (p1 p2 p3 : ℕ → Prop)
  (total_participants : ℕ)
  (h1 : ∃ n, n = 85 * total_participants / 100 ∧ ∀ k, k < n → p1 k)
  (h2 : ∃ n, n = 80 * total_participants / 100 ∧ ∀ k, k < n → p2 k)
  (h3 : ∃ n, n = 75 * total_participants / 100 ∧ ∀ k, k < n → p3 k) :
  ∃ n, n ≥ 40 * total_participants / 100 ∧ ∀ k, k < n → p1 k ∧ p2 k ∧ p3 k :=
by
  sorry

end NUMINAMATH_GPT_competition_participants_solved_all_three_l429_42931


namespace NUMINAMATH_GPT_cyclists_travel_same_distance_l429_42965

-- Define constants for speeds
def v1 := 12   -- speed of the first cyclist in km/h
def v2 := 16   -- speed of the second cyclist in km/h
def v3 := 24   -- speed of the third cyclist in km/h

-- Define the known total time
def total_time := 3  -- total time in hours

-- Hypothesis: Prove that the distance traveled by each cyclist is 16 km
theorem cyclists_travel_same_distance (d : ℚ) : 
  (v1 * (total_time * 3 / 13)) = d ∧
  (v2 * (total_time * 4 / 13)) = d ∧
  (v3 * (total_time * 6 / 13)) = d ∧
  d = 16 :=
by
  sorry

end NUMINAMATH_GPT_cyclists_travel_same_distance_l429_42965


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l429_42907

theorem sufficient_but_not_necessary (x : ℝ) : (x < -2 → x ≤ 0) → ¬(x ≤ 0 → x < -2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l429_42907


namespace NUMINAMATH_GPT_minimum_species_l429_42969

theorem minimum_species (n : ℕ) (h : n = 2021) 
  (even_separation : ∀ (a b : ℕ), a ≠ b → (a ≠ b) → (a % 2 = 0)) : 
  ∃ (s : ℕ), s = 1011 :=
by
  sorry

end NUMINAMATH_GPT_minimum_species_l429_42969


namespace NUMINAMATH_GPT_range_of_a_l429_42942

def p (a : ℝ) := 0 < a ∧ a < 1
def q (a : ℝ) := a > 5 / 2 ∨ 0 < a ∧ a < 1 / 2

theorem range_of_a (a : ℝ) :
  (a > 0) ∧ (a ≠ 1) ∧ (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ (1 / 2 ≤ a ∧ a < 1) ∨ (a > 5 / 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l429_42942


namespace NUMINAMATH_GPT_sum_of_10th_degree_polynomials_is_no_higher_than_10_l429_42988

-- Given definitions of two 10th-degree polynomials
def polynomial1 := ∃p : Polynomial ℝ, p.degree = 10
def polynomial2 := ∃p : Polynomial ℝ, p.degree = 10

-- Statement to prove
theorem sum_of_10th_degree_polynomials_is_no_higher_than_10 :
  ∀ (p q : Polynomial ℝ), p.degree = 10 → q.degree = 10 → (p + q).degree ≤ 10 := by
  sorry

end NUMINAMATH_GPT_sum_of_10th_degree_polynomials_is_no_higher_than_10_l429_42988


namespace NUMINAMATH_GPT_trig_identity_l429_42971

open Real

theorem trig_identity : sin (20 * π / 180) * cos (10 * π / 180) - cos (160 * π / 180) * sin (170 * π / 180) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_l429_42971


namespace NUMINAMATH_GPT_distinguishable_arrangements_l429_42933

-- Define number of each type of tiles
def brown_tiles := 2
def purple_tiles := 1
def green_tiles := 3
def yellow_tiles := 2
def total_tiles := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem distinguishable_arrangements :
  (Nat.factorial total_tiles) / 
  ((Nat.factorial green_tiles) * 
   (Nat.factorial brown_tiles) * 
   (Nat.factorial yellow_tiles) * 
   (Nat.factorial purple_tiles)) = 1680 := by
  sorry

end NUMINAMATH_GPT_distinguishable_arrangements_l429_42933


namespace NUMINAMATH_GPT_correct_completion_l429_42986

theorem correct_completion (A B C D : String) : C = "None" :=
by
  let sentence := "Did you have any trouble with the customs officer? " ++ C ++ " to speak of."
  let correct_sentence := "Did you have any trouble with the customs officer? None to speak of."
  sorry

end NUMINAMATH_GPT_correct_completion_l429_42986


namespace NUMINAMATH_GPT_cos_225_l429_42909

theorem cos_225 (h : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2) :
    Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_225_l429_42909


namespace NUMINAMATH_GPT_scientific_notation_86400_l429_42902

theorem scientific_notation_86400 : 86400 = 8.64 * 10^4 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_86400_l429_42902


namespace NUMINAMATH_GPT_roots_sum_prod_eq_l429_42975

theorem roots_sum_prod_eq (p q : ℤ) (h1 : p / 3 = 9) (h2 : q / 3 = 20) : p + q = 87 :=
by
  sorry

end NUMINAMATH_GPT_roots_sum_prod_eq_l429_42975


namespace NUMINAMATH_GPT_taxi_fare_for_100_miles_l429_42959

theorem taxi_fare_for_100_miles
  (base_fare : ℝ := 10)
  (proportional_fare : ℝ := 140 / 80)
  (fare_for_80_miles : ℝ := 150)
  (distance_80 : ℝ := 80)
  (distance_100 : ℝ := 100) :
  let additional_fare := proportional_fare * distance_100
  let total_fare_for_100_miles := base_fare + additional_fare
  total_fare_for_100_miles = 185 :=
by
  sorry

end NUMINAMATH_GPT_taxi_fare_for_100_miles_l429_42959


namespace NUMINAMATH_GPT_food_drive_ratio_l429_42915

/-- Mark brings in 4 times as many cans as Jaydon,
Jaydon brings in 5 more cans than a certain multiple of the amount of cans that Rachel brought in,
There are 135 cans total, and Mark brought in 100 cans.
Prove that the ratio of the number of cans Jaydon brought in to the number of cans Rachel brought in is 5:2. -/
theorem food_drive_ratio (J R : ℕ) (k : ℕ)
  (h1 : 4 * J = 100)
  (h2 : J = k * R + 5)
  (h3 : 100 + J + R = 135) :
  J / Nat.gcd J R = 5 ∧ R / Nat.gcd J R = 2 := by
  sorry

end NUMINAMATH_GPT_food_drive_ratio_l429_42915


namespace NUMINAMATH_GPT_hyperbola_focal_length_l429_42908

theorem hyperbola_focal_length (m : ℝ) (h_eq : m * x^2 + 2 * y^2 = 2) (h_imag_axis : -2 / m = 4) : 
  ∃ (f : ℝ), f = 2 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_hyperbola_focal_length_l429_42908


namespace NUMINAMATH_GPT_solve_equation_l429_42980

theorem solve_equation (x : ℝ) :
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 60 → x = 12 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l429_42980


namespace NUMINAMATH_GPT_calculate_fraction_l429_42998

variable (a b : ℝ)

theorem calculate_fraction (h : a ≠ b) : (2 * a / (a - b)) + (2 * b / (b - a)) = 2 := by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l429_42998


namespace NUMINAMATH_GPT_sin_minus_cos_eq_one_sol_l429_42972

theorem sin_minus_cos_eq_one_sol (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = 1) :
  x = Real.pi / 2 ∨ x = Real.pi :=
sorry

end NUMINAMATH_GPT_sin_minus_cos_eq_one_sol_l429_42972


namespace NUMINAMATH_GPT_part1_part2_l429_42976

noncomputable def a (n : ℕ) : ℤ :=
  15 * n + 2 + (15 * n - 32) * 16^(n-1)

theorem part1 (n : ℕ) : 15^3 ∣ (a n) := by
  sorry

-- Correct answer for part (2) bundled in a formal statement:
theorem part2 (n k : ℕ) : 1991 ∣ (a n) ∧ 1991 ∣ (a (n + 1)) ∧
    1991 ∣ (a (n + 2)) ↔ n = 89595 * k := by
  sorry

end NUMINAMATH_GPT_part1_part2_l429_42976


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l429_42954

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + m > 0) ↔ m > 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l429_42954


namespace NUMINAMATH_GPT_number_of_candidates_l429_42950

theorem number_of_candidates
  (P : ℕ) (A_c A_p A_f : ℕ)
  (h_p : P = 100)
  (h_ac : A_c = 35)
  (h_ap : A_p = 39)
  (h_af : A_f = 15) :
  ∃ T : ℕ, T = 120 := 
by
  sorry

end NUMINAMATH_GPT_number_of_candidates_l429_42950


namespace NUMINAMATH_GPT_g_domain_l429_42936

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arcsin (x ^ 3))

theorem g_domain : {x : ℝ | -1 < x ∧ x < 1} = Set {x | ∃ y, g x = y} :=
by
  sorry

end NUMINAMATH_GPT_g_domain_l429_42936


namespace NUMINAMATH_GPT_folded_rectangle_perimeter_l429_42982

theorem folded_rectangle_perimeter (l : ℝ) (w : ℝ) (h_diag : ℝ)
  (h_l : l = 20) (h_w : w = 12)
  (h_diag : h_diag = Real.sqrt (l^2 + w^2)) :
  2 * (l + w) = 64 :=
by
  rw [h_l, h_w]
  simp only [mul_add, mul_two, add_mul] at *
  norm_num


end NUMINAMATH_GPT_folded_rectangle_perimeter_l429_42982


namespace NUMINAMATH_GPT_total_number_of_gifts_l429_42952

/-- Number of gifts calculation, given the distribution conditions with certain children -/
theorem total_number_of_gifts
  (n : ℕ) -- the total number of children
  (h1 : 2 * 4 + (n - 2) * 3 + 11 = 3 * n + 13) -- first scenario equation
  (h2 : 4 * 3 + (n - 4) * 6 + 10 = 6 * n - 2) -- second scenario equation
  : 3 * n + 13 = 28 := 
by 
  sorry

end NUMINAMATH_GPT_total_number_of_gifts_l429_42952


namespace NUMINAMATH_GPT_construction_days_behind_without_additional_workers_l429_42900

-- Definitions for initial and additional workers and their respective efficiencies and durations.
def initial_workers : ℕ := 100
def initial_worker_efficiency : ℕ := 1
def total_days : ℕ := 150

def additional_workers_1 : ℕ := 50
def additional_worker_efficiency_1 : ℕ := 2
def additional_worker_start_day_1 : ℕ := 30

def additional_workers_2 : ℕ := 25
def additional_worker_efficiency_2 : ℕ := 3
def additional_worker_start_day_2 : ℕ := 45

def additional_workers_3 : ℕ := 15
def additional_worker_efficiency_3 : ℕ := 4
def additional_worker_start_day_3 : ℕ := 75

-- Define the total additional work units done by the extra workers.
def total_additional_work_units : ℕ := 
  (additional_workers_1 * additional_worker_efficiency_1 * (total_days - additional_worker_start_day_1)) +
  (additional_workers_2 * additional_worker_efficiency_2 * (total_days - additional_worker_start_day_2)) +
  (additional_workers_3 * additional_worker_efficiency_3 * (total_days - additional_worker_start_day_3))

-- Define the days the initial workers would have taken to do the additional work.
def initial_days_for_additional_work : ℕ := 
  (total_additional_work_units + (initial_workers * initial_worker_efficiency) - 1) / (initial_workers * initial_worker_efficiency)

-- Define the total days behind schedule.
def days_behind_schedule : ℕ := (total_days + initial_days_for_additional_work) - total_days

-- Define the theorem to prove.
theorem construction_days_behind_without_additional_workers : days_behind_schedule = 244 := 
  by 
  -- This translates to manually verifying the outcome.
  -- A detailed proof can be added later.
  sorry

end NUMINAMATH_GPT_construction_days_behind_without_additional_workers_l429_42900


namespace NUMINAMATH_GPT_total_peaches_l429_42919

theorem total_peaches (initial_peaches_Audrey : ℕ) (multiplier_Audrey : ℕ)
                      (initial_peaches_Paul : ℕ) (multiplier_Paul : ℕ)
                      (initial_peaches_Maya : ℕ) (additional_peaches_Maya : ℕ) :
                      initial_peaches_Audrey = 26 →
                      multiplier_Audrey = 3 →
                      initial_peaches_Paul = 48 →
                      multiplier_Paul = 2 →
                      initial_peaches_Maya = 57 →
                      additional_peaches_Maya = 20 →
                      (initial_peaches_Audrey + multiplier_Audrey * initial_peaches_Audrey) +
                      (initial_peaches_Paul + multiplier_Paul * initial_peaches_Paul) +
                      (initial_peaches_Maya + additional_peaches_Maya) = 325 :=
by
  sorry

end NUMINAMATH_GPT_total_peaches_l429_42919


namespace NUMINAMATH_GPT_hyperbola_midpoint_l429_42929

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end NUMINAMATH_GPT_hyperbola_midpoint_l429_42929


namespace NUMINAMATH_GPT_adjacent_product_negative_l429_42977

-- Define the sequence
def a (n : ℕ) : ℤ := 2*n - 17

-- Define the claim about the product of adjacent terms being negative
theorem adjacent_product_negative : a 8 * a 9 < 0 :=
by sorry

end NUMINAMATH_GPT_adjacent_product_negative_l429_42977


namespace NUMINAMATH_GPT_ping_pong_ball_probability_l429_42960
open Nat 

def total_balls : ℕ := 70

def multiples_of_4_count : ℕ := 17
def multiples_of_9_count : ℕ := 7
def multiples_of_4_and_9_count : ℕ := 1

def inclusion_exclusion_principle : ℕ :=
  multiples_of_4_count + multiples_of_9_count - multiples_of_4_and_9_count

def desired_outcomes_count : ℕ := inclusion_exclusion_principle

def probability : ℚ := desired_outcomes_count / total_balls

theorem ping_pong_ball_probability : probability = 23 / 70 :=
  sorry

end NUMINAMATH_GPT_ping_pong_ball_probability_l429_42960


namespace NUMINAMATH_GPT_number_of_doubles_players_l429_42978

theorem number_of_doubles_players (x y : ℕ) 
  (h1 : x + y = 13) 
  (h2 : 4 * x - 2 * y = 4) : 
  4 * x = 20 :=
by sorry

end NUMINAMATH_GPT_number_of_doubles_players_l429_42978


namespace NUMINAMATH_GPT_nurses_quit_count_l429_42949

-- Initial Definitions
def initial_doctors : ℕ := 11
def initial_nurses : ℕ := 18
def doctors_quit : ℕ := 5
def total_remaining_staff : ℕ := 22

-- Remaining Doctors Calculation
def remaining_doctors : ℕ := initial_doctors - doctors_quit

-- Theorem to prove the number of nurses who quit
theorem nurses_quit_count : initial_nurses - (total_remaining_staff - remaining_doctors) = 2 := by
  sorry

end NUMINAMATH_GPT_nurses_quit_count_l429_42949


namespace NUMINAMATH_GPT_cylinder_height_l429_42917

theorem cylinder_height (OA OB : ℝ) (h_OA : OA = 7) (h_OB : OB = 2) :
  ∃ (h_cylinder : ℝ), h_cylinder = 3 * Real.sqrt 5 :=
by
  use (Real.sqrt (OA^2 - OB^2))
  rw [h_OA, h_OB]
  norm_num
  sorry

end NUMINAMATH_GPT_cylinder_height_l429_42917


namespace NUMINAMATH_GPT_even_function_sum_eval_l429_42934

variable (v : ℝ → ℝ)

theorem even_function_sum_eval (h_even : ∀ x : ℝ, v x = v (-x)) :
    v (-2.33) + v (-0.81) + v (0.81) + v (2.33) = 2 * (v 2.33 + v 0.81) :=
by
  sorry

end NUMINAMATH_GPT_even_function_sum_eval_l429_42934


namespace NUMINAMATH_GPT_annual_population_growth_l429_42964

noncomputable def annual_percentage_increase := 
  let P0 := 15000
  let P2 := 18150  
  exists (r : ℝ), (P0 * (1 + r)^2 = P2) ∧ (r = 0.1)

theorem annual_population_growth : annual_percentage_increase :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_annual_population_growth_l429_42964


namespace NUMINAMATH_GPT_correct_operation_l429_42918

variable {a : ℝ}

theorem correct_operation : a^4 / (-a)^2 = a^2 := by
  sorry

end NUMINAMATH_GPT_correct_operation_l429_42918


namespace NUMINAMATH_GPT_bike_shop_profit_l429_42943

theorem bike_shop_profit :
  let tire_repair_charge := 20
  let tire_repair_cost := 5
  let tire_repairs_per_month := 300
  let complex_repair_charge := 300
  let complex_repair_cost := 50
  let complex_repairs_per_month := 2
  let retail_profit := 2000
  let fixed_expenses := 4000
  let total_tire_profit := tire_repairs_per_month * (tire_repair_charge - tire_repair_cost)
  let total_complex_profit := complex_repairs_per_month * (complex_repair_charge - complex_repair_cost)
  let total_income := total_tire_profit + total_complex_profit + retail_profit
  let final_profit := total_income - fixed_expenses
  final_profit = 3000 :=
by
  sorry

end NUMINAMATH_GPT_bike_shop_profit_l429_42943


namespace NUMINAMATH_GPT_cone_volume_l429_42995

theorem cone_volume (S : ℝ) (hPos : S > 0) : 
  let R := Real.sqrt (S / 7)
  let H := Real.sqrt (5 * S)
  let V := (π * S * (Real.sqrt (5 * S))) / 21
  (π * R * R * H / 3) = V := 
sorry

end NUMINAMATH_GPT_cone_volume_l429_42995


namespace NUMINAMATH_GPT_pyramid_volume_inequality_l429_42989

theorem pyramid_volume_inequality
  (k : ℝ)
  (OA1 OB1 OC1 OA2 OB2 OC2 OA3 OB3 OC3 OB2 : ℝ)
  (V1 := k * |OA1| * |OB1| * |OC1|)
  (V2 := k * |OA2| * |OB2| * |OC2|)
  (V3 := k * |OA3| * |OB3| * |OC3|)
  (V := k * |OA1| * |OB2| * |OC3|) :
  V ≤ (V1 + V2 + V3) / 3 := 
  sorry

end NUMINAMATH_GPT_pyramid_volume_inequality_l429_42989


namespace NUMINAMATH_GPT_jane_output_increase_l429_42984

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end NUMINAMATH_GPT_jane_output_increase_l429_42984


namespace NUMINAMATH_GPT_compute_xy_l429_42939

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : x * y = 0 := 
sorry

end NUMINAMATH_GPT_compute_xy_l429_42939


namespace NUMINAMATH_GPT_find_n_l429_42935

theorem find_n (n : ℕ) (h : n * n.factorial + 2 * n.factorial = 5040) : n = 5 :=
sorry

end NUMINAMATH_GPT_find_n_l429_42935


namespace NUMINAMATH_GPT_meeting_time_when_speeds_doubled_l429_42955

noncomputable def meeting_time (x y z : ℝ) : ℝ :=
  2 * 91

theorem meeting_time_when_speeds_doubled
  (x y z : ℝ)
  (h1 : 2 * z * (x + y) = (2 * z - 56) * (2 * x + y))
  (h2 : 2 * z * (x + y) = (2 * z - 65) * (x + 2 * y))
  : meeting_time x y z = 182 := 
sorry

end NUMINAMATH_GPT_meeting_time_when_speeds_doubled_l429_42955


namespace NUMINAMATH_GPT_money_initial_amounts_l429_42906

theorem money_initial_amounts (x : ℕ) (A B : ℕ) 
  (h1 : A = 8 * x) 
  (h2 : B = 5 * x) 
  (h3 : (A - 50) = 4 * (B + 100) / 5) : 
  A = 800 ∧ B = 500 := 
sorry

end NUMINAMATH_GPT_money_initial_amounts_l429_42906


namespace NUMINAMATH_GPT_final_answer_correct_l429_42991

-- Define the initial volume V0
def V0 := 1

-- Define the volume increment ratio for new tetrahedra
def volume_ratio := (1 : ℚ) / 27

-- Define the recursive volume increments
def ΔP1 := 4 * volume_ratio
def ΔP2 := 16 * volume_ratio
def ΔP3 := 64 * volume_ratio
def ΔP4 := 256 * volume_ratio

-- Define the total volume V4
def V4 := V0 + ΔP1 + ΔP2 + ΔP3 + ΔP4

-- The target volume as a rational number
def target_volume := 367 / 27

-- Define the fraction components
def m := 367
def n := 27

-- Define the final answer
def final_answer := m + n

-- Proof statement to verify the final answer
theorem final_answer_correct :
  V4 = target_volume ∧ (Nat.gcd m n = 1) ∧ final_answer = 394 :=
by
  -- The specifics of the proof are omitted
  sorry

end NUMINAMATH_GPT_final_answer_correct_l429_42991


namespace NUMINAMATH_GPT_cubic_sum_l429_42967

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 := 
sorry

end NUMINAMATH_GPT_cubic_sum_l429_42967


namespace NUMINAMATH_GPT_quadratic_negative_root_l429_42946

theorem quadratic_negative_root (m : ℝ) : (∃ x : ℝ, (m * x^2 + 2 * x + 1 = 0 ∧ x < 0)) ↔ (m ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_negative_root_l429_42946


namespace NUMINAMATH_GPT_walking_east_of_neg_west_l429_42930

-- Define the representation of directions
def is_walking_west (d : ℕ) (x : ℤ) : Prop := x = d
def is_walking_east (d : ℕ) (x : ℤ) : Prop := x = -d

-- Given the condition and states the relationship is the proposition to prove.
theorem walking_east_of_neg_west (d : ℕ) (x : ℤ) (h : is_walking_west 2 2) : is_walking_east 5 (-5) :=
by
  sorry

end NUMINAMATH_GPT_walking_east_of_neg_west_l429_42930


namespace NUMINAMATH_GPT_probability_of_yellow_light_l429_42901

def time_red : ℕ := 30
def time_green : ℕ := 25
def time_yellow : ℕ := 5
def total_cycle_time : ℕ := time_red + time_green + time_yellow

theorem probability_of_yellow_light :
  (time_yellow : ℚ) / (total_cycle_time : ℚ) = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_yellow_light_l429_42901


namespace NUMINAMATH_GPT_exists_function_f_l429_42974

theorem exists_function_f (f : ℕ → ℕ) : (∀ n : ℕ, f (f n) = n^2) → ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n^2 :=
sorry

end NUMINAMATH_GPT_exists_function_f_l429_42974


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l429_42944

theorem necessary_but_not_sufficient_condition (p q : ℝ → Prop)
    (h₁ : ∀ x k, p x ↔ x ≥ k) 
    (h₂ : ∀ x, q x ↔ 3 / (x + 1) < 1) 
    (h₃ : ∃ k : ℝ, ∀ x, p x → q x ∧ ¬ (q x → p x)) :
  ∃ k, k > 2 :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l429_42944


namespace NUMINAMATH_GPT_probability_correct_l429_42912

-- Defining the values on the spinner
inductive SpinnerValue
| Bankrupt
| Thousand
| EightHundred
| FiveThousand
| Thousand'

open SpinnerValue

-- Function to get value in number from SpinnerValue
def value (v : SpinnerValue) : ℕ :=
  match v with
  | Bankrupt => 0
  | Thousand => 1000
  | EightHundred => 800
  | FiveThousand => 5000
  | Thousand' => 1000

-- Total number of spins
def total_spins : ℕ := 3

-- Total possible outcomes
def total_outcomes : ℕ := (5 : ℕ) ^ total_spins

-- Number of favorable outcomes (count of permutations summing to 5800)
def favorable_outcomes : ℕ :=
  12  -- This comes from solution steps

-- The probability as a ratio of favorable outcomes to total outcomes
def probability_of_5800_in_three_spins : ℚ :=
  favorable_outcomes / total_outcomes

theorem probability_correct :
  probability_of_5800_in_three_spins = 12 / 125 := by
  sorry

end NUMINAMATH_GPT_probability_correct_l429_42912


namespace NUMINAMATH_GPT_reinforcement_arrival_l429_42958

theorem reinforcement_arrival (x : ℕ) :
  (2000 * 40) = (2000 * x + 4000 * 10) → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_reinforcement_arrival_l429_42958


namespace NUMINAMATH_GPT_midpoint_AB_is_correct_l429_42956

/--
In the Cartesian coordinate system, given points A (-1, 2) and B (3, 0), prove that the coordinates of the midpoint of segment AB are (1, 1).
-/
theorem midpoint_AB_is_correct :
  let A := (-1, 2)
  let B := (3, 0)
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1 := 
by {
  let A := (-1, 2)
  let B := (3, 0)
  sorry -- this part is omitted as no proof is needed
}

end NUMINAMATH_GPT_midpoint_AB_is_correct_l429_42956


namespace NUMINAMATH_GPT_orangeade_price_per_glass_l429_42928

theorem orangeade_price_per_glass (O : ℝ) (W : ℝ) (P : ℝ) (price_1_day : ℝ) 
    (h1 : W = O) (h2 : price_1_day = 0.30) (revenue_equal : 2 * O * price_1_day = 3 * O * P) :
  P = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_orangeade_price_per_glass_l429_42928


namespace NUMINAMATH_GPT_M_is_subset_of_N_l429_42994

theorem M_is_subset_of_N : 
  ∀ (x y : ℝ), (|x| + |y| < 1) → 
    (Real.sqrt ((x - 1/2)^2 + (y + 1/2)^2) + Real.sqrt ((x + 1/2)^2 + (y - 1/2)^2) < 2 * Real.sqrt 2) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_M_is_subset_of_N_l429_42994


namespace NUMINAMATH_GPT_minimum_value_l429_42990

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 10)

theorem minimum_value (x : ℝ) (h : x > 10) : (∃ y : ℝ, (∀ x' : ℝ, x' > 10 → f x' ≥ y) ∧ y = 40) := 
sorry

end NUMINAMATH_GPT_minimum_value_l429_42990


namespace NUMINAMATH_GPT_mary_initially_selected_10_l429_42911

-- Definitions based on the conditions
def price_apple := 40
def price_orange := 60
def avg_price_initial := 54
def avg_price_after_putting_back := 48
def num_oranges_put_back := 5

-- Definition of Mary_initially_selected as the total number of pieces of fruit initially selected by Mary
def Mary_initially_selected (A O : ℕ) := A + O

-- Theorem statement
theorem mary_initially_selected_10 (A O : ℕ) 
  (h1 : (price_apple * A + price_orange * O) / (A + O) = avg_price_initial)
  (h2 : (price_apple * A + price_orange * (O - num_oranges_put_back)) / (A + O - num_oranges_put_back) = avg_price_after_putting_back) : 
  Mary_initially_selected A O = 10 := 
sorry

end NUMINAMATH_GPT_mary_initially_selected_10_l429_42911


namespace NUMINAMATH_GPT_prime_sum_diff_l429_42985

open Nat

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The problem statement
theorem prime_sum_diff (p : ℕ) (q s r t : ℕ) :
  is_prime p → is_prime q → is_prime s → is_prime r → is_prime t →
  p = q + s → p = r - t → p = 5 :=
by
  sorry

end NUMINAMATH_GPT_prime_sum_diff_l429_42985


namespace NUMINAMATH_GPT_sum_a3_a4_eq_14_l429_42983

open Nat

-- Define variables
def S (n : ℕ) : ℕ := n^2 + n
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem sum_a3_a4_eq_14 : a 3 + a 4 = 14 := by
  sorry

end NUMINAMATH_GPT_sum_a3_a4_eq_14_l429_42983


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l429_42903

theorem arithmetic_geometric_sequence (a d : ℤ) (h1 : ∃ a d, (a - d) * a * (a + d) = 1000)
  (h2 : ∃ a d, a^2 = 2 * (a - d) * ((a + d) + 7)) :
  d = 8 ∨ d = -15 :=
by sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l429_42903


namespace NUMINAMATH_GPT_remainder_div_72_l429_42968

theorem remainder_div_72 (x : ℤ) (h : x % 8 = 3) : x % 72 = 3 :=
sorry

end NUMINAMATH_GPT_remainder_div_72_l429_42968


namespace NUMINAMATH_GPT_portion_divided_equally_for_efforts_l429_42961

-- Definitions of conditions
def tom_investment : ℝ := 700
def jerry_investment : ℝ := 300
def tom_more_than_jerry : ℝ := 800
def total_profit : ℝ := 3000

-- Theorem stating what we need to prove
theorem portion_divided_equally_for_efforts (T J R E : ℝ) 
  (h1 : T = tom_investment)
  (h2 : J = jerry_investment)
  (h3 : total_profit = R)
  (h4 : (E / 2) + (7 / 10) * (R - E) - (E / 2 + (3 / 10) * (R - E)) = tom_more_than_jerry) 
  : E = 1000 :=
by
  sorry

end NUMINAMATH_GPT_portion_divided_equally_for_efforts_l429_42961


namespace NUMINAMATH_GPT_subset_implies_value_l429_42940

theorem subset_implies_value (a : ℝ) : (∀ x ∈ ({0, -a} : Set ℝ), x ∈ ({1, -1, 2 * a - 2} : Set ℝ)) → a = 1 := by
  sorry

end NUMINAMATH_GPT_subset_implies_value_l429_42940


namespace NUMINAMATH_GPT_device_works_probability_l429_42910

theorem device_works_probability (p_comp_damaged : ℝ) (two_components : Bool) :
  p_comp_damaged = 0.1 → two_components = true → (0.9 * 0.9 = 0.81) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_device_works_probability_l429_42910


namespace NUMINAMATH_GPT_kyle_car_payment_l429_42987

theorem kyle_car_payment (income rent utilities retirement groceries insurance miscellaneous gas x : ℕ)
  (h_income : income = 3200)
  (h_rent : rent = 1250)
  (h_utilities : utilities = 150)
  (h_retirement : retirement = 400)
  (h_groceries : groceries = 300)
  (h_insurance : insurance = 200)
  (h_miscellaneous : miscellaneous = 200)
  (h_gas : gas = 350)
  (h_expenses : rent + utilities + retirement + groceries + insurance + miscellaneous + gas + x = income) :
  x = 350 :=
by sorry

end NUMINAMATH_GPT_kyle_car_payment_l429_42987


namespace NUMINAMATH_GPT_sum_of_remainders_mod_30_l429_42938

theorem sum_of_remainders_mod_30 (a b c : ℕ) (h1 : a % 30 = 14) (h2 : b % 30 = 11) (h3 : c % 30 = 19) :
  (a + b + c) % 30 = 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_mod_30_l429_42938


namespace NUMINAMATH_GPT_percent_of_sales_not_pens_pencils_erasers_l429_42962

theorem percent_of_sales_not_pens_pencils_erasers :
  let percent_pens := 25
  let percent_pencils := 30
  let percent_erasers := 20
  let percent_total := 100
  percent_total - (percent_pens + percent_pencils + percent_erasers) = 25 :=
by
  -- definitions and assumptions
  let percent_pens := 25
  let percent_pencils := 30
  let percent_erasers := 20
  let percent_total := 100
  sorry

end NUMINAMATH_GPT_percent_of_sales_not_pens_pencils_erasers_l429_42962


namespace NUMINAMATH_GPT_div_by_squares_l429_42932

variables {R : Type*} [CommRing R] (a b c x y z : R)

theorem div_by_squares (a b c x y z : R) :
  (a * y - b * x) ^ 2 + (b * z - c * y) ^ 2 + (c * x - a * z) ^ 2 + (a * x + b * y + c * z) ^ 2 =
    (a ^ 2 + b ^ 2 + c ^ 2) * (x ^ 2 + y ^ 2 + z ^ 2) := sorry

end NUMINAMATH_GPT_div_by_squares_l429_42932


namespace NUMINAMATH_GPT_cards_given_by_Dan_l429_42997

def initial_cards : Nat := 27
def bought_cards : Nat := 20
def total_cards : Nat := 88

theorem cards_given_by_Dan :
  ∃ (cards_given : Nat), cards_given = total_cards - bought_cards - initial_cards :=
by
  use 41
  sorry

end NUMINAMATH_GPT_cards_given_by_Dan_l429_42997


namespace NUMINAMATH_GPT_intersection_nonempty_implies_m_eq_zero_l429_42926

theorem intersection_nonempty_implies_m_eq_zero (m : ℤ) (P Q : Set ℝ)
  (hP : P = { -1, ↑m } ) (hQ : Q = { x : ℝ | -1 < x ∧ x < 3/4 }) (h : (P ∩ Q).Nonempty) :
  m = 0 :=
by
  sorry

end NUMINAMATH_GPT_intersection_nonempty_implies_m_eq_zero_l429_42926


namespace NUMINAMATH_GPT_distance_from_P_to_origin_l429_42945

open Real -- This makes it easier to use real number functions and constants.

noncomputable def hyperbola := { P : ℝ × ℝ // (P.1^2 / 9) - (P.2^2 / 7) = 1 }

theorem distance_from_P_to_origin 
  (P : ℝ × ℝ) 
  (hP : (P.1^2 / 9) - (P.2^2 / 7) = 1)
  (d_right_focus : P.1 - 4 = -1) : 
  dist P (0, 0) = 3 :=
sorry

end NUMINAMATH_GPT_distance_from_P_to_origin_l429_42945


namespace NUMINAMATH_GPT_find_nat_numbers_l429_42937

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

theorem find_nat_numbers (n : ℕ) :
  (n + sum_of_digits n = 2021) ↔ (n = 2014 ∨ n = 1996) :=
by
  sorry

end NUMINAMATH_GPT_find_nat_numbers_l429_42937
