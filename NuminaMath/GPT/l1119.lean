import Mathlib

namespace construction_days_behind_without_additional_workers_l1119_111930

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

end construction_days_behind_without_additional_workers_l1119_111930


namespace original_number_of_men_l1119_111997

theorem original_number_of_men (x : ℕ) (h1 : x * 10 = (x - 5) * 12) : x = 30 :=
by
  sorry

end original_number_of_men_l1119_111997


namespace math_problem_l1119_111991

theorem math_problem (a b c : ℝ) (h1 : (a + b) / 2 = 30) (h2 : (b + c) / 2 = 60) (h3 : c - a = 60) : c - a = 60 :=
by
  -- Insert proof steps here
  sorry

end math_problem_l1119_111991


namespace portion_divided_equally_for_efforts_l1119_111980

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

end portion_divided_equally_for_efforts_l1119_111980


namespace sum_of_pos_real_solutions_l1119_111931

open Real

noncomputable def cos_equation_sum_pos_real_solutions : ℝ := 1082 * π

theorem sum_of_pos_real_solutions :
  ∃ x : ℝ, (0 < x) ∧ 
    (∀ x, 2 * cos (2 * x) * (cos (2 * x) - cos ((2016 * π ^ 2) / x)) = cos (6 * x) - 1) → 
      x = cos_equation_sum_pos_real_solutions :=
sorry

end sum_of_pos_real_solutions_l1119_111931


namespace minimum_species_l1119_111948

theorem minimum_species (n : ℕ) (h : n = 2021) 
  (even_separation : ∀ (a b : ℕ), a ≠ b → (a ≠ b) → (a % 2 = 0)) : 
  ∃ (s : ℕ), s = 1011 :=
by
  sorry

end minimum_species_l1119_111948


namespace cubic_sum_l1119_111986

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 := 
sorry

end cubic_sum_l1119_111986


namespace trig_identity_l1119_111989

open Real

theorem trig_identity : sin (20 * π / 180) * cos (10 * π / 180) - cos (160 * π / 180) * sin (170 * π / 180) = 1 / 2 := 
by
  sorry

end trig_identity_l1119_111989


namespace adjacent_product_negative_l1119_111983

-- Define the sequence
def a (n : ℕ) : ℤ := 2*n - 17

-- Define the claim about the product of adjacent terms being negative
theorem adjacent_product_negative : a 8 * a 9 < 0 :=
by sorry

end adjacent_product_negative_l1119_111983


namespace percent_of_sales_not_pens_pencils_erasers_l1119_111968

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

end percent_of_sales_not_pens_pencils_erasers_l1119_111968


namespace total_amount_spent_l1119_111984

def cost_per_dozen_apples : ℕ := 40
def cost_per_dozen_pears : ℕ := 50
def dozens_apples : ℕ := 14
def dozens_pears : ℕ := 14

theorem total_amount_spent : (dozens_apples * cost_per_dozen_apples + dozens_pears * cost_per_dozen_pears) = 1260 := 
  by
  sorry

end total_amount_spent_l1119_111984


namespace ping_pong_ball_probability_l1119_111959
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

end ping_pong_ball_probability_l1119_111959


namespace kyle_car_payment_l1119_111953

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

end kyle_car_payment_l1119_111953


namespace additional_charge_per_minute_atlantic_call_l1119_111961

def base_rate_U : ℝ := 11.0
def rate_per_minute_U : ℝ := 0.25
def base_rate_A : ℝ := 12.0
def call_duration : ℝ := 20.0
variable (rate_per_minute_A : ℝ)

theorem additional_charge_per_minute_atlantic_call :
  base_rate_U + rate_per_minute_U * call_duration = base_rate_A + rate_per_minute_A * call_duration →
  rate_per_minute_A = 0.20 := by
  sorry

end additional_charge_per_minute_atlantic_call_l1119_111961


namespace food_drive_ratio_l1119_111908

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

end food_drive_ratio_l1119_111908


namespace leading_digits_sum_l1119_111919

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

end leading_digits_sum_l1119_111919


namespace hyperbola_focal_length_l1119_111911

theorem hyperbola_focal_length (m : ℝ) (h_eq : m * x^2 + 2 * y^2 = 2) (h_imag_axis : -2 / m = 4) : 
  ∃ (f : ℝ), f = 2 * Real.sqrt 5 := 
sorry

end hyperbola_focal_length_l1119_111911


namespace solve_equation_l1119_111981

theorem solve_equation (x : ℝ) :
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 60 → x = 12 :=
by
  sorry

end solve_equation_l1119_111981


namespace Dan_speed_must_exceed_45_mph_l1119_111985

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

end Dan_speed_must_exceed_45_mph_l1119_111985


namespace M_is_subset_of_N_l1119_111976

theorem M_is_subset_of_N : 
  ∀ (x y : ℝ), (|x| + |y| < 1) → 
    (Real.sqrt ((x - 1/2)^2 + (y + 1/2)^2) + Real.sqrt ((x + 1/2)^2 + (y - 1/2)^2) < 2 * Real.sqrt 2) :=
by
  intro x y h
  sorry

end M_is_subset_of_N_l1119_111976


namespace inscribed_circle_radius_of_rhombus_l1119_111952

theorem inscribed_circle_radius_of_rhombus (d1 d2 : ℝ) (a r : ℝ) : 
  d1 = 15 → d2 = 24 → a = Real.sqrt ((15 / 2)^2 + (24 / 2)^2) → 
  (d1 * d2) / 2 = 2 * a * r → 
  r = 60.07 / 13 :=
by
  intros h1 h2 h3 h4
  sorry

end inscribed_circle_radius_of_rhombus_l1119_111952


namespace sum_of_10th_degree_polynomials_is_no_higher_than_10_l1119_111954

-- Given definitions of two 10th-degree polynomials
def polynomial1 := ∃p : Polynomial ℝ, p.degree = 10
def polynomial2 := ∃p : Polynomial ℝ, p.degree = 10

-- Statement to prove
theorem sum_of_10th_degree_polynomials_is_no_higher_than_10 :
  ∀ (p q : Polynomial ℝ), p.degree = 10 → q.degree = 10 → (p + q).degree ≤ 10 := by
  sorry

end sum_of_10th_degree_polynomials_is_no_higher_than_10_l1119_111954


namespace cards_given_by_Dan_l1119_111945

def initial_cards : Nat := 27
def bought_cards : Nat := 20
def total_cards : Nat := 88

theorem cards_given_by_Dan :
  ∃ (cards_given : Nat), cards_given = total_cards - bought_cards - initial_cards :=
by
  use 41
  sorry

end cards_given_by_Dan_l1119_111945


namespace karl_drove_420_miles_l1119_111982

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

end karl_drove_420_miles_l1119_111982


namespace reinforcement_arrival_l1119_111957

theorem reinforcement_arrival (x : ℕ) :
  (2000 * 40) = (2000 * x + 4000 * 10) → x = 20 :=
by
  sorry

end reinforcement_arrival_l1119_111957


namespace water_remaining_45_days_l1119_111946

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

end water_remaining_45_days_l1119_111946


namespace midpoint_AB_is_correct_l1119_111965

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

end midpoint_AB_is_correct_l1119_111965


namespace range_of_a_l1119_111921

def p (a : ℝ) := 0 < a ∧ a < 1
def q (a : ℝ) := a > 5 / 2 ∨ 0 < a ∧ a < 1 / 2

theorem range_of_a (a : ℝ) :
  (a > 0) ∧ (a ≠ 1) ∧ (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ (1 / 2 ≤ a ∧ a < 1) ∨ (a > 5 / 2) :=
sorry

end range_of_a_l1119_111921


namespace angle_C_in_triangle_l1119_111928

open Real

noncomputable def determine_angle_C (A B C: ℝ) (AB BC: ℝ) : Prop :=
  A = (3 * π) / 4 ∧ BC = sqrt 2 * AB → C = π / 6

-- Define the theorem to state the problem
theorem angle_C_in_triangle (A B C : ℝ) (AB BC : ℝ) :
  determine_angle_C A B C AB BC := 
by
  -- Step to indicate where the proof would be
  sorry

end angle_C_in_triangle_l1119_111928


namespace total_peaches_l1119_111918

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

end total_peaches_l1119_111918


namespace pyramid_volume_inequality_l1119_111962

theorem pyramid_volume_inequality
  (k : ℝ)
  (OA1 OB1 OC1 OA2 OB2 OC2 OA3 OB3 OC3 OB2 : ℝ)
  (V1 := k * |OA1| * |OB1| * |OC1|)
  (V2 := k * |OA2| * |OB2| * |OC2|)
  (V3 := k * |OA3| * |OB3| * |OC3|)
  (V := k * |OA1| * |OB2| * |OC3|) :
  V ≤ (V1 + V2 + V3) / 3 := 
  sorry

end pyramid_volume_inequality_l1119_111962


namespace total_number_of_gifts_l1119_111943

/-- Number of gifts calculation, given the distribution conditions with certain children -/
theorem total_number_of_gifts
  (n : ℕ) -- the total number of children
  (h1 : 2 * 4 + (n - 2) * 3 + 11 = 3 * n + 13) -- first scenario equation
  (h2 : 4 * 3 + (n - 4) * 6 + 10 = 6 * n - 2) -- second scenario equation
  : 3 * n + 13 = 28 := 
by 
  sorry

end total_number_of_gifts_l1119_111943


namespace cyclists_travel_same_distance_l1119_111955

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

end cyclists_travel_same_distance_l1119_111955


namespace find_n_l1119_111902

theorem find_n (n : ℕ) (h : n * n.factorial + 2 * n.factorial = 5040) : n = 5 :=
sorry

end find_n_l1119_111902


namespace div_by_squares_l1119_111917

variables {R : Type*} [CommRing R] (a b c x y z : R)

theorem div_by_squares (a b c x y z : R) :
  (a * y - b * x) ^ 2 + (b * z - c * y) ^ 2 + (c * x - a * z) ^ 2 + (a * x + b * y + c * z) ^ 2 =
    (a ^ 2 + b ^ 2 + c ^ 2) * (x ^ 2 + y ^ 2 + z ^ 2) := sorry

end div_by_squares_l1119_111917


namespace count_digit_2_in_range_1_to_1000_l1119_111906

theorem count_digit_2_in_range_1_to_1000 :
  let count_digit_occur (digit : ℕ) (range_end : ℕ) : ℕ :=
    (range_end + 1).digits 10
    |>.count digit
  count_digit_occur 2 1000 = 300 :=
by
  sorry

end count_digit_2_in_range_1_to_1000_l1119_111906


namespace intersection_nonempty_implies_m_eq_zero_l1119_111923

theorem intersection_nonempty_implies_m_eq_zero (m : ℤ) (P Q : Set ℝ)
  (hP : P = { -1, ↑m } ) (hQ : Q = { x : ℝ | -1 < x ∧ x < 3/4 }) (h : (P ∩ Q).Nonempty) :
  m = 0 :=
by
  sorry

end intersection_nonempty_implies_m_eq_zero_l1119_111923


namespace probability_of_yellow_light_l1119_111904

def time_red : ℕ := 30
def time_green : ℕ := 25
def time_yellow : ℕ := 5
def total_cycle_time : ℕ := time_red + time_green + time_yellow

theorem probability_of_yellow_light :
  (time_yellow : ℚ) / (total_cycle_time : ℚ) = 1 / 12 :=
by
  sorry

end probability_of_yellow_light_l1119_111904


namespace correct_completion_l1119_111979

theorem correct_completion (A B C D : String) : C = "None" :=
by
  let sentence := "Did you have any trouble with the customs officer? " ++ C ++ " to speak of."
  let correct_sentence := "Did you have any trouble with the customs officer? None to speak of."
  sorry

end correct_completion_l1119_111979


namespace fifth_bowler_points_l1119_111909

variable (P1 P2 P3 P4 P5 : ℝ)
variable (h1 : P1 = (5 / 12) * P3)
variable (h2 : P2 = (5 / 3) * P3)
variable (h3 : P4 = (5 / 3) * P3)
variable (h4 : P5 = (50 / 27) * P3)
variable (h5 : P3 ≤ 500)
variable (total_points : P1 + P2 + P3 + P4 + P5 = 2000)

theorem fifth_bowler_points : P5 = 561 :=
  sorry

end fifth_bowler_points_l1119_111909


namespace compute_xy_l1119_111935

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : x * y = 0 := 
sorry

end compute_xy_l1119_111935


namespace solve_for_x_l1119_111925

theorem solve_for_x : (∃ x : ℝ, (x / 18) * (x / 72) = 1) → ∃ x : ℝ, x = 36 :=
by
  sorry

end solve_for_x_l1119_111925


namespace largest_perimeter_triangle_l1119_111944

theorem largest_perimeter_triangle :
  ∃ (y : ℤ), 4 < y ∧ y < 20 ∧ 8 + 12 + y = 39 :=
by {
  -- we'll skip the proof steps
  sorry 
}

end largest_perimeter_triangle_l1119_111944


namespace quadratic_negative_root_l1119_111949

theorem quadratic_negative_root (m : ℝ) : (∃ x : ℝ, (m * x^2 + 2 * x + 1 = 0 ∧ x < 0)) ↔ (m ≤ 1) :=
by
  sorry

end quadratic_negative_root_l1119_111949


namespace orangeade_price_per_glass_l1119_111900

theorem orangeade_price_per_glass (O : ℝ) (W : ℝ) (P : ℝ) (price_1_day : ℝ) 
    (h1 : W = O) (h2 : price_1_day = 0.30) (revenue_equal : 2 * O * price_1_day = 3 * O * P) :
  P = 0.20 :=
by
  sorry

end orangeade_price_per_glass_l1119_111900


namespace jane_output_increase_l1119_111970

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end jane_output_increase_l1119_111970


namespace taxi_fare_for_100_miles_l1119_111958

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

end taxi_fare_for_100_miles_l1119_111958


namespace meeting_time_when_speeds_doubled_l1119_111964

noncomputable def meeting_time (x y z : ℝ) : ℝ :=
  2 * 91

theorem meeting_time_when_speeds_doubled
  (x y z : ℝ)
  (h1 : 2 * z * (x + y) = (2 * z - 56) * (2 * x + y))
  (h2 : 2 * z * (x + y) = (2 * z - 65) * (x + 2 * y))
  : meeting_time x y z = 182 := 
sorry

end meeting_time_when_speeds_doubled_l1119_111964


namespace exists_function_f_l1119_111988

theorem exists_function_f (f : ℕ → ℕ) : (∀ n : ℕ, f (f n) = n^2) → ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n^2 :=
sorry

end exists_function_f_l1119_111988


namespace find_k_l1119_111903

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

end find_k_l1119_111903


namespace correct_operation_l1119_111910

variable {a : ℝ}

theorem correct_operation : a^4 / (-a)^2 = a^2 := by
  sorry

end correct_operation_l1119_111910


namespace distinguishable_arrangements_l1119_111937

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

end distinguishable_arrangements_l1119_111937


namespace value_of_squares_l1119_111956

-- Define the conditions
variables (p q : ℝ)

-- State the theorem with the given conditions and the proof goal
theorem value_of_squares (h1 : p * q = 12) (h2 : p + q = 8) : p ^ 2 + q ^ 2 = 40 :=
sorry

end value_of_squares_l1119_111956


namespace restaurant_pizzas_more_than_hotdogs_l1119_111950

theorem restaurant_pizzas_more_than_hotdogs
  (H P : ℕ) 
  (h1 : H = 60)
  (h2 : 30 * (P + H) = 4800) :
  P - H = 40 :=
by
  sorry

end restaurant_pizzas_more_than_hotdogs_l1119_111950


namespace distance_from_P_to_origin_l1119_111973

open Real -- This makes it easier to use real number functions and constants.

noncomputable def hyperbola := { P : ℝ × ℝ // (P.1^2 / 9) - (P.2^2 / 7) = 1 }

theorem distance_from_P_to_origin 
  (P : ℝ × ℝ) 
  (hP : (P.1^2 / 9) - (P.2^2 / 7) = 1)
  (d_right_focus : P.1 - 4 = -1) : 
  dist P (0, 0) = 3 :=
sorry

end distance_from_P_to_origin_l1119_111973


namespace bike_shop_profit_l1119_111974

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

end bike_shop_profit_l1119_111974


namespace final_answer_correct_l1119_111987

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

end final_answer_correct_l1119_111987


namespace rosie_pies_l1119_111927

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

end rosie_pies_l1119_111927


namespace number_of_candidates_l1119_111971

theorem number_of_candidates
  (P : ℕ) (A_c A_p A_f : ℕ)
  (h_p : P = 100)
  (h_ac : A_c = 35)
  (h_ap : A_p = 39)
  (h_af : A_f = 15) :
  ∃ T : ℕ, T = 120 := 
by
  sorry

end number_of_candidates_l1119_111971


namespace find_cost_price_l1119_111938

theorem find_cost_price
  (cost_price : ℝ)
  (increase_rate : ℝ := 0.2)
  (decrease_rate : ℝ := 0.1)
  (profit : ℝ := 8):
  (1 + increase_rate) * cost_price * (1 - decrease_rate) - cost_price = profit → 
  cost_price = 100 := 
by 
  sorry

end find_cost_price_l1119_111938


namespace annual_population_growth_l1119_111947

noncomputable def annual_percentage_increase := 
  let P0 := 15000
  let P2 := 18150  
  exists (r : ℝ), (P0 * (1 + r)^2 = P2) ∧ (r = 0.1)

theorem annual_population_growth : annual_percentage_increase :=
by
  -- Placeholder proof
  sorry

end annual_population_growth_l1119_111947


namespace monotone_intervals_max_floor_a_l1119_111913

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + a

theorem monotone_intervals (a : ℝ) (h : a = 1) :
  (∀ x, 0 < x ∧ x < 1 → deriv (λ x => f x 1) x > 0) ∧
  (∀ x, 1 ≤ x → deriv (λ x => f x 1) x < 0) :=
by
  sorry

theorem max_floor_a (a : ℝ) (h : ∀ x > 0, f x a ≤ x) : ⌊a⌋ = 1 :=
by
  sorry

end monotone_intervals_max_floor_a_l1119_111913


namespace part1_part2_l1119_111992

noncomputable def a (n : ℕ) : ℤ :=
  15 * n + 2 + (15 * n - 32) * 16^(n-1)

theorem part1 (n : ℕ) : 15^3 ∣ (a n) := by
  sorry

-- Correct answer for part (2) bundled in a formal statement:
theorem part2 (n k : ℕ) : 1991 ∣ (a n) ∧ 1991 ∣ (a (n + 1)) ∧
    1991 ∣ (a (n + 2)) ↔ n = 89595 * k := by
  sorry

end part1_part2_l1119_111992


namespace mary_initially_selected_10_l1119_111916

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

end mary_initially_selected_10_l1119_111916


namespace hyperbola_midpoint_l1119_111901

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l1119_111901


namespace find_triangle_sides_l1119_111994

theorem find_triangle_sides (k : ℕ) (k_pos : k = 6) 
  {x y z : ℝ} (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) 
  (h : k * (x * y + y * z + z * x) > 5 * (x ^ 2 + y ^ 2 + z ^ 2)) :
  ∃ x' y' z', (x = x') ∧ (y = y') ∧ (z = z') ∧ ((x' + y' > z') ∧ (x' + z' > y') ∧ (y' + z' > x')) :=
by
  sorry

end find_triangle_sides_l1119_111994


namespace speed_of_third_part_l1119_111932

theorem speed_of_third_part (d : ℝ) (v : ℝ)
  (h1 : 3 * d = 3.000000000000001)
  (h2 : d / 3 + d / 4 + d / v = 47/60) :
  v = 5 := by
  sorry

end speed_of_third_part_l1119_111932


namespace find_x_l1119_111972

theorem find_x 
  (x : ℝ) 
  (h1 : 0 < x)
  (h2 : x < π / 2)
  (h3 : 1 / (Real.sin x) = 1 / (Real.sin (2 * x)) + 1 / (Real.sin (4 * x)) + 1 / (Real.sin (8 * x))) : 
  x = π / 15 ∨ x = π / 5 ∨ x = π / 3 ∨ x = 7 * π / 15 :=
by
  sorry

end find_x_l1119_111972


namespace nurses_quit_count_l1119_111939

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

end nurses_quit_count_l1119_111939


namespace scientific_notation_86400_l1119_111914

theorem scientific_notation_86400 : 86400 = 8.64 * 10^4 :=
by
  sorry

end scientific_notation_86400_l1119_111914


namespace find_x_l1119_111929

theorem find_x : ∃ x : ℚ, (3 * x + 5) / 5 = 17 ↔ x = 80 / 3 := by
  sorry

end find_x_l1119_111929


namespace call_center_agents_ratio_l1119_111933

noncomputable def fraction_of_agents (calls_A calls_B total_agents total_calls : ℕ) : ℚ :=
  let calls_A_per_agent := calls_A / total_agents
  let calls_B_per_agent := calls_B / total_agents
  let ratio_calls_A_B := (3: ℚ) / 5
  let fraction_calls_B := (8: ℚ) / 11
  let fraction_calls_A := (3: ℚ) / 11
  let ratio_of_agents := (5: ℚ) / 11
  if (calls_A_per_agent * fraction_calls_A = ratio_calls_A_B * calls_B_per_agent) then ratio_of_agents else 0

theorem call_center_agents_ratio (calls_A calls_B total_agents total_calls agents_A agents_B : ℕ) :
  (calls_A : ℚ) / (calls_B : ℚ) = (3 / 5) →
  (calls_B : ℚ) = (8 / 11) * total_calls →
  (agents_A : ℚ) = (5 / 11) * (agents_B : ℚ) :=
sorry

end call_center_agents_ratio_l1119_111933


namespace determine_constants_l1119_111995

structure Vector2D :=
(x : ℝ)
(y : ℝ)

def a := 11 / 20
def b := -7 / 20

def v1 : Vector2D := ⟨3, 2⟩
def v2 : Vector2D := ⟨-1, 6⟩
def v3 : Vector2D := ⟨2, -1⟩

def linear_combination (v1 v2 : Vector2D) (a b : ℝ) : Vector2D :=
  ⟨a * v1.x + b * v2.x, a * v1.y + b * v2.y⟩

theorem determine_constants (a b : ℝ) :
  ∃ (a b : ℝ), linear_combination v1 v2 a b = v3 :=
by
  use (11 / 20)
  use (-7 / 20)
  sorry

end determine_constants_l1119_111995


namespace cos_225_l1119_111905

theorem cos_225 (h : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2) :
    Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_l1119_111905


namespace arithmetic_geometric_sequence_l1119_111922

theorem arithmetic_geometric_sequence (a d : ℤ) (h1 : ∃ a d, (a - d) * a * (a + d) = 1000)
  (h2 : ∃ a d, a^2 = 2 * (a - d) * ((a + d) + 7)) :
  d = 8 ∨ d = -15 :=
by sorry

end arithmetic_geometric_sequence_l1119_111922


namespace alpha3_plus_8beta_plus_6_eq_30_l1119_111915

noncomputable def alpha_beta_quad_roots (α β : ℝ) : Prop :=
  α^2 - 2 * α - 4 = 0 ∧ β^2 - 2 * β - 4 = 0

theorem alpha3_plus_8beta_plus_6_eq_30 (α β : ℝ) (h : alpha_beta_quad_roots α β) : 
  α^3 + 8 * β + 6 = 30 :=
sorry

end alpha3_plus_8beta_plus_6_eq_30_l1119_111915


namespace device_works_probability_l1119_111920

theorem device_works_probability (p_comp_damaged : ℝ) (two_components : Bool) :
  p_comp_damaged = 0.1 → two_components = true → (0.9 * 0.9 = 0.81) :=
by
  intros h1 h2
  sorry

end device_works_probability_l1119_111920


namespace quadratic_inequality_solution_l1119_111963

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + m > 0) ↔ m > 1 :=
by
  sorry

end quadratic_inequality_solution_l1119_111963


namespace distinct_sequences_count_l1119_111934

def letters := ["E", "Q", "U", "A", "L", "S"]

noncomputable def count_sequences : Nat :=
  let remaining_letters := ["E", "Q", "U", "A"] -- 'L' and 'S' are already considered
  3 * (4 * 3) -- as analyzed: (LS__) + (L_S_) + (L__S)

theorem distinct_sequences_count : count_sequences = 36 := 
  by
    unfold count_sequences
    sorry

end distinct_sequences_count_l1119_111934


namespace raise_salary_to_original_l1119_111967

/--
The salary of a person was reduced by 25%. By what percent should his reduced salary be raised
so as to bring it at par with his original salary?
-/
theorem raise_salary_to_original (S : ℝ) (h : S > 0) :
  ∃ P : ℝ, 0.75 * S * (1 + P / 100) = S ∧ P = 33.333333333333336 :=
sorry

end raise_salary_to_original_l1119_111967


namespace expression_bounds_l1119_111999

theorem expression_bounds (x y z w : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ (Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) +
                     Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2)) ∧
  (Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) +
   Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2)) ≤ 4 := sorry

end expression_bounds_l1119_111999


namespace folded_rectangle_perimeter_l1119_111941

theorem folded_rectangle_perimeter (l : ℝ) (w : ℝ) (h_diag : ℝ)
  (h_l : l = 20) (h_w : w = 12)
  (h_diag : h_diag = Real.sqrt (l^2 + w^2)) :
  2 * (l + w) = 64 :=
by
  rw [h_l, h_w]
  simp only [mul_add, mul_two, add_mul] at *
  norm_num


end folded_rectangle_perimeter_l1119_111941


namespace sasha_questions_per_hour_l1119_111924

-- Define the total questions and the time she worked, and the remaining questions
def total_questions : ℕ := 60
def time_worked : ℕ := 2
def remaining_questions : ℕ := 30

-- Define the number of questions she completed
def questions_completed := total_questions - remaining_questions

-- Define the rate at which she completes questions per hour
def questions_per_hour := questions_completed / time_worked

-- The theorem to prove
theorem sasha_questions_per_hour : questions_per_hour = 15 := 
by
  -- Here we would prove the theorem, but we're using sorry to skip the proof for now
  sorry

end sasha_questions_per_hour_l1119_111924


namespace even_function_sum_eval_l1119_111926

variable (v : ℝ → ℝ)

theorem even_function_sum_eval (h_even : ∀ x : ℝ, v x = v (-x)) :
    v (-2.33) + v (-0.81) + v (0.81) + v (2.33) = 2 * (v 2.33 + v 0.81) :=
by
  sorry

end even_function_sum_eval_l1119_111926


namespace smallest_digit_for_divisibility_by_9_l1119_111969

theorem smallest_digit_for_divisibility_by_9 : 
  ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ (18 + d) % 9 = 0 ∧ ∀ d' : ℕ, (0 ≤ d' ∧ d' ≤ 9 ∧ (18 + d') % 9 = 0) → d' ≥ d :=
sorry

end smallest_digit_for_divisibility_by_9_l1119_111969


namespace prime_sum_diff_l1119_111966

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

end prime_sum_diff_l1119_111966


namespace number_of_doubles_players_l1119_111951

theorem number_of_doubles_players (x y : ℕ) 
  (h1 : x + y = 13) 
  (h2 : 4 * x - 2 * y = 4) : 
  4 * x = 20 :=
by sorry

end number_of_doubles_players_l1119_111951


namespace sin_minus_cos_eq_one_sol_l1119_111940

theorem sin_minus_cos_eq_one_sol (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = 1) :
  x = Real.pi / 2 ∨ x = Real.pi :=
sorry

end sin_minus_cos_eq_one_sol_l1119_111940


namespace remainder_div_72_l1119_111978

theorem remainder_div_72 (x : ℤ) (h : x % 8 = 3) : x % 72 = 3 :=
sorry

end remainder_div_72_l1119_111978


namespace cone_volume_l1119_111977

theorem cone_volume (S : ℝ) (hPos : S > 0) : 
  let R := Real.sqrt (S / 7)
  let H := Real.sqrt (5 * S)
  let V := (π * S * (Real.sqrt (5 * S))) / 21
  (π * R * R * H / 3) = V := 
sorry

end cone_volume_l1119_111977


namespace cylinder_height_l1119_111907

theorem cylinder_height (OA OB : ℝ) (h_OA : OA = 7) (h_OB : OB = 2) :
  ∃ (h_cylinder : ℝ), h_cylinder = 3 * Real.sqrt 5 :=
by
  use (Real.sqrt (OA^2 - OB^2))
  rw [h_OA, h_OB]
  norm_num
  sorry

end cylinder_height_l1119_111907


namespace walking_east_of_neg_west_l1119_111936

-- Define the representation of directions
def is_walking_west (d : ℕ) (x : ℤ) : Prop := x = d
def is_walking_east (d : ℕ) (x : ℤ) : Prop := x = -d

-- Given the condition and states the relationship is the proposition to prove.
theorem walking_east_of_neg_west (d : ℕ) (x : ℤ) (h : is_walking_west 2 2) : is_walking_east 5 (-5) :=
by
  sorry

end walking_east_of_neg_west_l1119_111936


namespace symmetric_circle_eq_l1119_111993

/-- Define the equation of the circle C -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Define the equation of the line l -/
def line_equation (x y : ℝ) : Prop := x + y - 1 = 0

/-- 
The symmetric circle to C with respect to line l 
has the equation (x - 1)^2 + (y - 1)^2 = 4.
-/
theorem symmetric_circle_eq (x y : ℝ) :
  (∃ x y : ℝ, circle_equation x y) → 
  (∃ x y : ℝ, line_equation x y) →
  (∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 4) :=
by
  sorry

end symmetric_circle_eq_l1119_111993


namespace roots_sum_prod_eq_l1119_111990

theorem roots_sum_prod_eq (p q : ℤ) (h1 : p / 3 = 9) (h2 : q / 3 = 20) : p + q = 87 :=
by
  sorry

end roots_sum_prod_eq_l1119_111990


namespace sum_a3_a4_eq_14_l1119_111942

open Nat

-- Define variables
def S (n : ℕ) : ℕ := n^2 + n
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem sum_a3_a4_eq_14 : a 3 + a 4 = 14 := by
  sorry

end sum_a3_a4_eq_14_l1119_111942


namespace minimum_value_l1119_111960

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 10)

theorem minimum_value (x : ℝ) (h : x > 10) : (∃ y : ℝ, (∀ x' : ℝ, x' > 10 → f x' ≥ y) ∧ y = 40) := 
sorry

end minimum_value_l1119_111960


namespace necessary_but_not_sufficient_condition_l1119_111975

theorem necessary_but_not_sufficient_condition (p q : ℝ → Prop)
    (h₁ : ∀ x k, p x ↔ x ≥ k) 
    (h₂ : ∀ x, q x ↔ 3 / (x + 1) < 1) 
    (h₃ : ∃ k : ℝ, ∀ x, p x → q x ∧ ¬ (q x → p x)) :
  ∃ k, k > 2 :=
by
  sorry

end necessary_but_not_sufficient_condition_l1119_111975


namespace additional_cost_per_pint_proof_l1119_111996

-- Definitions based on the problem conditions
def pints_sold := 54
def total_revenue_on_sale := 216
def revenue_difference := 108

-- Derived definitions
def revenue_if_not_on_sale := total_revenue_on_sale + revenue_difference
def cost_per_pint_on_sale := total_revenue_on_sale / pints_sold
def cost_per_pint_not_on_sale := revenue_if_not_on_sale / pints_sold
def additional_cost_per_pint := cost_per_pint_not_on_sale - cost_per_pint_on_sale

-- Proof statement
theorem additional_cost_per_pint_proof :
  additional_cost_per_pint = 2 :=
by
  -- Placeholder to indicate that the proof is not provided
  sorry

end additional_cost_per_pint_proof_l1119_111996


namespace find_nat_numbers_l1119_111912

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

theorem find_nat_numbers (n : ℕ) :
  (n + sum_of_digits n = 2021) ↔ (n = 2014 ∨ n = 1996) :=
by
  sorry

end find_nat_numbers_l1119_111912


namespace ratio_of_radii_l1119_111998

variables (a b : ℝ) (h : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2)

theorem ratio_of_radii (h : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) : 
  a / b = Real.sqrt 5 / 5 :=
sorry

end ratio_of_radii_l1119_111998
