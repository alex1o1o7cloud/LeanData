import Mathlib

namespace NUMINAMATH_GPT_sum_of_cubes_eq_96_over_7_l1580_158096

-- Define the conditions from the problem
variables (a r : ℝ)
axiom condition_sum : a / (1 - r) = 2
axiom condition_sum_squares : a^2 / (1 - r^2) = 6

-- Define the correct answer that we expect to prove
theorem sum_of_cubes_eq_96_over_7 :
  a^3 / (1 - r^3) = 96 / 7 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_eq_96_over_7_l1580_158096


namespace NUMINAMATH_GPT_three_digit_number_satisfies_conditions_l1580_158050

-- Definitions for the digits of the number
def x := 9
def y := 6
def z := 4

-- Define the three-digit number
def number := 100 * x + 10 * y + z

-- Define the conditions
def geometric_progression := y * y = x * z

def reverse_order_condition := (number - 495) = 100 * z + 10 * y + x

def arithmetic_progression := (z - 1) + (x - 2) = 2 * (y - 1)

-- The theorem to prove
theorem three_digit_number_satisfies_conditions :
  geometric_progression ∧ reverse_order_condition ∧ arithmetic_progression :=
by {
  sorry
}

end NUMINAMATH_GPT_three_digit_number_satisfies_conditions_l1580_158050


namespace NUMINAMATH_GPT_sarah_score_is_122_l1580_158043

-- Define the problem parameters and state the theorem
theorem sarah_score_is_122 (s g : ℝ)
  (h1 : s = g + 40)
  (h2 : (s + g) / 2 = 102) :
  s = 122 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end NUMINAMATH_GPT_sarah_score_is_122_l1580_158043


namespace NUMINAMATH_GPT_max_value_sqrt43_l1580_158026

noncomputable def max_value_expr (x y z : ℝ) : ℝ :=
  3 * x * z * Real.sqrt 2 + 5 * x * y

theorem max_value_sqrt43 (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x^2 + y^2 + z^2 = 1) :
  max_value_expr x y z ≤ Real.sqrt 43 :=
sorry

end NUMINAMATH_GPT_max_value_sqrt43_l1580_158026


namespace NUMINAMATH_GPT_total_area_of_storage_units_l1580_158053

theorem total_area_of_storage_units (total_units remaining_units : ℕ) 
    (size_8_by_4 length width unit_area_200 : ℕ)
    (h1 : total_units = 42)
    (h2 : remaining_units = 22)
    (h3 : length = 8)
    (h4 : width = 4)
    (h5 : unit_area_200 = 200) 
    (h6 : ∀ i : ℕ, i < 20 → unit_area_8_by_4 = length * width) 
    (h7 : ∀ j : ℕ, j < 22 → unit_area_200 = 200) :
    total_area_of_all_units = 5040 :=
by
  let unit_area_8_by_4 := length * width
  let total_area_20_units := 20 * unit_area_8_by_4
  let total_area_22_units := 22 * unit_area_200
  let total_area_of_all_units := total_area_20_units + total_area_22_units
  sorry

end NUMINAMATH_GPT_total_area_of_storage_units_l1580_158053


namespace NUMINAMATH_GPT_michael_choices_l1580_158088

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem michael_choices : combination 10 4 = 210 := by
  sorry

end NUMINAMATH_GPT_michael_choices_l1580_158088


namespace NUMINAMATH_GPT_arithmetic_sequence_value_of_n_l1580_158058

theorem arithmetic_sequence_value_of_n :
  ∀ (a n d : ℕ), a = 1 → d = 3 → (a + (n - 1) * d = 2005) → n = 669 :=
by
  intros a n d h_a1 h_d ha_n
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_of_n_l1580_158058


namespace NUMINAMATH_GPT_units_digit_of_expression_l1580_158038

theorem units_digit_of_expression :
  (4 ^ 101 * 5 ^ 204 * 9 ^ 303 * 11 ^ 404) % 10 = 0 := 
sorry

end NUMINAMATH_GPT_units_digit_of_expression_l1580_158038


namespace NUMINAMATH_GPT_number_of_pots_of_rosemary_l1580_158089

-- Definitions based on the conditions
def total_leaves_basil (pots_basil : ℕ) (leaves_per_basil : ℕ) : ℕ := pots_basil * leaves_per_basil
def total_leaves_rosemary (pots_rosemary : ℕ) (leaves_per_rosemary : ℕ) : ℕ := pots_rosemary * leaves_per_rosemary
def total_leaves_thyme (pots_thyme : ℕ) (leaves_per_thyme : ℕ) : ℕ := pots_thyme * leaves_per_thyme

-- The given problem conditions
def pots_basil : ℕ := 3
def leaves_per_basil : ℕ := 4
def leaves_per_rosemary : ℕ := 18
def pots_thyme : ℕ := 6
def leaves_per_thyme : ℕ := 30
def total_leaves : ℕ := 354

-- Proving the number of pots of rosemary
theorem number_of_pots_of_rosemary : 
  ∃ (pots_rosemary : ℕ), 
  total_leaves_basil pots_basil leaves_per_basil + 
  total_leaves_rosemary pots_rosemary leaves_per_rosemary + 
  total_leaves_thyme pots_thyme leaves_per_thyme = 
  total_leaves ∧ pots_rosemary = 9 :=
by
  sorry  -- proof is omitted

end NUMINAMATH_GPT_number_of_pots_of_rosemary_l1580_158089


namespace NUMINAMATH_GPT_sum_is_zero_l1580_158013

theorem sum_is_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / |a|) + (b / |b|) + (c / |c|) + ((a * b * c) / |a * b * c|) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_is_zero_l1580_158013


namespace NUMINAMATH_GPT_meeting_time_and_location_l1580_158084

/-- Define the initial conditions -/
def start_time : ℕ := 8 -- 8:00 AM
def city_distance : ℕ := 12 -- 12 kilometers
def pedestrian_speed : ℚ := 6 -- 6 km/h
def cyclist_speed : ℚ := 18 -- 18 km/h

/-- Define the conditions for meeting time and location -/
theorem meeting_time_and_location :
  ∃ (meet_time : ℕ) (meet_distance : ℚ),
    meet_time = 9 * 60 + 15 ∧   -- 9:15 AM in minutes
    meet_distance = 4.5 :=      -- 4.5 kilometers
sorry

end NUMINAMATH_GPT_meeting_time_and_location_l1580_158084


namespace NUMINAMATH_GPT_max_k_consecutive_sum_l1580_158082

theorem max_k_consecutive_sum :
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, k * (2 * n + k - 1) = 2^2 * 3^8 ∧ ∀ k' > k, ¬ ∃ n', n' > 0 ∧ k' * (2 * n' + k' - 1) = 2^2 * 3^8 := sorry

end NUMINAMATH_GPT_max_k_consecutive_sum_l1580_158082


namespace NUMINAMATH_GPT_max_sum_when_product_is_399_l1580_158091

theorem max_sum_when_product_is_399 :
  ∃ (X Y Z : ℕ), X * Y * Z = 399 ∧ X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X ∧ X + Y + Z = 29 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_when_product_is_399_l1580_158091


namespace NUMINAMATH_GPT_division_remainder_3012_97_l1580_158077

theorem division_remainder_3012_97 : 3012 % 97 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_division_remainder_3012_97_l1580_158077


namespace NUMINAMATH_GPT_number_of_valid_house_numbers_l1580_158066

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

def digit_sum_odd (n : ℕ) : Prop :=
  (n / 10 + n % 10) % 2 = 1

def valid_house_number (W X Y Z : ℕ) : Prop :=
  W ≠ 0 ∧ X ≠ 0 ∧ Y ≠ 0 ∧ Z ≠ 0 ∧
  is_two_digit_prime (10 * W + X) ∧ is_two_digit_prime (10 * Y + Z) ∧
  10 * W + X ≠ 10 * Y + Z ∧
  10 * W + X < 60 ∧ 10 * Y + Z < 60 ∧
  digit_sum_odd (10 * W + X)

theorem number_of_valid_house_numbers : ∃ n, n = 108 ∧
  (∀ W X Y Z, valid_house_number W X Y Z → valid_house_number_count = 108) :=
sorry

end NUMINAMATH_GPT_number_of_valid_house_numbers_l1580_158066


namespace NUMINAMATH_GPT_martin_big_bell_rings_l1580_158005

theorem martin_big_bell_rings (B S : ℚ) (h1 : S = B / 3 + B^2 / 4) (h2 : S + B = 52) : B = 12 :=
by
  sorry

end NUMINAMATH_GPT_martin_big_bell_rings_l1580_158005


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1580_158059

theorem simplify_and_evaluate (a : ℕ) (h : a = 2022) :
  (a - 1) / a / (a - 1 / a) = 1 / 2023 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1580_158059


namespace NUMINAMATH_GPT_ellipse_problem_part1_ellipse_problem_part2_l1580_158085

-- Statement of the problem
theorem ellipse_problem_part1 :
  ∃ k : ℝ, (∀ x y : ℝ, (x^2 / 2) + y^2 = 1 → (
    (∃ t > 0, x = t * y + 1) → k = (Real.sqrt 2) / 2)) :=
sorry

theorem ellipse_problem_part2 :
  ∃ S_max : ℝ, ∀ (t : ℝ), (t > 0 → (S_max = (4 * (t^2 + 1)^2) / ((t^2 + 2) * (2 * t^2 + 1)))) → t^2 = 1 → S_max = 16 / 9 :=
sorry

end NUMINAMATH_GPT_ellipse_problem_part1_ellipse_problem_part2_l1580_158085


namespace NUMINAMATH_GPT_total_birds_l1580_158011

theorem total_birds (g d : Nat) (h₁ : g = 58) (h₂ : d = 37) : g + d = 95 :=
by
  sorry

end NUMINAMATH_GPT_total_birds_l1580_158011


namespace NUMINAMATH_GPT_det_A_is_2_l1580_158049

-- Define the matrix A
def A (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, 2], ![-3, d]]

-- Define the inverse of matrix A 
noncomputable def A_inv (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (1 / (a * d + 6)) • ![![d, -2], ![3, a]]

-- Condition: A + A_inv = 0
def condition (a d : ℝ) : Prop := A a d + A_inv a d = 0

-- Main theorem: determinant of A under the given condition
theorem det_A_is_2 (a d : ℝ) (h : condition a d) : Matrix.det (A a d) = 2 :=
by sorry

end NUMINAMATH_GPT_det_A_is_2_l1580_158049


namespace NUMINAMATH_GPT_gas_station_constant_l1580_158079

structure GasStationData where
  amount : ℝ
  unit_price : ℝ
  price_per_yuan_per_liter : ℝ

theorem gas_station_constant (data : GasStationData) (h1 : data.amount = 116.64) (h2 : data.unit_price = 18) (h3 : data.price_per_yuan_per_liter = 6.48) : data.unit_price = 18 :=
sorry

end NUMINAMATH_GPT_gas_station_constant_l1580_158079


namespace NUMINAMATH_GPT_f_increasing_l1580_158020

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sin x

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

end NUMINAMATH_GPT_f_increasing_l1580_158020


namespace NUMINAMATH_GPT_nobel_prize_laureates_at_workshop_l1580_158081

theorem nobel_prize_laureates_at_workshop :
  ∃ (T W W_and_N N_no_W X N : ℕ), 
    T = 50 ∧ 
    W = 31 ∧ 
    W_and_N = 16 ∧ 
    (N_no_W = X + 3) ∧ 
    (T - W = 19) ∧ 
    (N_no_W + X = 19) ∧ 
    (N = W_and_N + N_no_W) ∧ 
    N = 27 :=
by
  sorry

end NUMINAMATH_GPT_nobel_prize_laureates_at_workshop_l1580_158081


namespace NUMINAMATH_GPT_initial_velocity_calculation_l1580_158068

-- Define conditions
def acceleration_due_to_gravity := 10 -- m/s^2
def time_to_highest_point := 2 -- s
def velocity_at_highest_point := 0 -- m/s
def initial_observed_acceleration := 15 -- m/s^2

-- Theorem to prove the initial velocity
theorem initial_velocity_calculation
  (a_gravity : ℝ := acceleration_due_to_gravity)
  (t_highest : ℝ := time_to_highest_point)
  (v_highest : ℝ := velocity_at_highest_point)
  (a_initial : ℝ := initial_observed_acceleration) :
  ∃ (v_initial : ℝ), v_initial = 30 := 
sorry

end NUMINAMATH_GPT_initial_velocity_calculation_l1580_158068


namespace NUMINAMATH_GPT_three_digit_integers_count_l1580_158065

theorem three_digit_integers_count (N : ℕ) :
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
            n % 7 = 4 ∧ 
            n % 8 = 3 ∧ 
            n % 10 = 2) → N = 3 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_integers_count_l1580_158065


namespace NUMINAMATH_GPT_intersection_A_B_l1580_158001

open Set

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}
def B : Set ℝ := {x : ℝ | x^2 + 2 * x - 3 ≥ 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1580_158001


namespace NUMINAMATH_GPT_gcd_101_power_l1580_158087

theorem gcd_101_power (a b : ℕ) (h1 : a = 101^6 + 1) (h2 : b = 3 * 101^6 + 101^3 + 1) (h_prime : Nat.Prime 101) : Nat.gcd a b = 1 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_gcd_101_power_l1580_158087


namespace NUMINAMATH_GPT_daily_evaporation_rate_l1580_158033

/-- A statement that verifies the daily water evaporation rate -/
theorem daily_evaporation_rate
  (initial_water : ℝ)
  (evaporation_percentage : ℝ)
  (evaporation_period : ℕ) :
  initial_water = 15 →
  evaporation_percentage = 0.05 →
  evaporation_period = 15 →
  (evaporation_percentage * initial_water / evaporation_period) = 0.05 :=
by
  intros h_water h_percentage h_period
  sorry

end NUMINAMATH_GPT_daily_evaporation_rate_l1580_158033


namespace NUMINAMATH_GPT_second_set_length_is_correct_l1580_158060

variables (first_set_length second_set_length : ℝ)

theorem second_set_length_is_correct 
  (h1 : first_set_length = 4)
  (h2 : second_set_length = 5 * first_set_length) : 
  second_set_length = 20 := 
by 
  sorry

end NUMINAMATH_GPT_second_set_length_is_correct_l1580_158060


namespace NUMINAMATH_GPT_sharon_highway_speed_l1580_158009

theorem sharon_highway_speed:
  ∀ (total_distance : ℝ) (highway_time : ℝ) (city_time: ℝ) (city_speed : ℝ),
  total_distance = 59 → highway_time = 1 / 3 → city_time = 2 / 3 → city_speed = 45 →
  (total_distance - city_speed * city_time) / highway_time = 87 :=
by
  intro total_distance highway_time city_time city_speed
  intro h_total_distance h_highway_time h_city_time h_city_speed
  rw [h_total_distance, h_highway_time, h_city_time, h_city_speed]
  sorry

end NUMINAMATH_GPT_sharon_highway_speed_l1580_158009


namespace NUMINAMATH_GPT_second_less_than_first_l1580_158023

-- Define the given conditions
def third_number : ℝ := sorry
def first_number : ℝ := 0.65 * third_number
def second_number : ℝ := 0.58 * third_number

-- Problem statement: Prove that the second number is approximately 10.77% less than the first number
theorem second_less_than_first : 
  (first_number - second_number) / first_number * 100 = 10.77 := 
sorry

end NUMINAMATH_GPT_second_less_than_first_l1580_158023


namespace NUMINAMATH_GPT_cylinder_volume_l1580_158086

-- Define the volume of the cone
def V_cone : ℝ := 18.84

-- Define the volume of the cylinder
def V_cylinder : ℝ := 3 * V_cone

-- Prove that the volume of the cylinder is 56.52 cubic meters
theorem cylinder_volume :
  V_cylinder = 56.52 := 
by 
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_cylinder_volume_l1580_158086


namespace NUMINAMATH_GPT_winnie_the_pooh_wins_l1580_158080

variable (cones : ℕ)

def can_guarantee_win (initial_cones : ℕ) : Prop :=
  ∃ strategy : (ℕ → ℕ), 
    (strategy initial_cones = 4 ∨ strategy initial_cones = 1) ∧ 
    ∀ n, (strategy n = 1 → (n = 2012 - 4 ∨ n = 2007 - 1 ∨ n = 2005 - 1)) ∧
         (strategy n = 4 → n = 2012)

theorem winnie_the_pooh_wins : can_guarantee_win 2012 :=
sorry

end NUMINAMATH_GPT_winnie_the_pooh_wins_l1580_158080


namespace NUMINAMATH_GPT_min_value_is_144_l1580_158075

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  x^2 + 4 * x * y + 4 * y^2 + 3 * z^2

theorem min_value_is_144 (x y z : ℝ) (hxyz : x * y * z = 48) : 
  ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ xyz = 48 ∧ min_value_expression x y z = 144 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_is_144_l1580_158075


namespace NUMINAMATH_GPT_count_polynomials_l1580_158037

def is_polynomial (expr : String) : Bool :=
  match expr with
  | "-7"            => true
  | "x"             => true
  | "m^2 + 1/m"     => false
  | "x^2*y + 5"     => true
  | "(x + y)/2"     => true
  | "-5ab^3c^2"     => true
  | "1/y"           => false
  | _               => false

theorem count_polynomials :
  let expressions := ["-7", "x", "m^2 + 1/m", "x^2*y + 5", "(x + y)/2", "-5ab^3c^2", "1/y"]
  List.filter is_polynomial expressions |>.length = 5 :=
by
  sorry

end NUMINAMATH_GPT_count_polynomials_l1580_158037


namespace NUMINAMATH_GPT_expression_evaluation_l1580_158097

theorem expression_evaluation : 
  (2^10 * 3^3) / (6 * 2^5) = 144 :=
by 
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1580_158097


namespace NUMINAMATH_GPT_calculate_treatment_received_l1580_158067

variable (drip_rate : ℕ) (duration_hours : ℕ) (drops_convert : ℕ) (ml_convert : ℕ)

theorem calculate_treatment_received (h1 : drip_rate = 20) (h2 : duration_hours = 2) 
    (h3 : drops_convert = 100) (h4 : ml_convert = 5) : 
    (drip_rate * (duration_hours * 60) * ml_convert) / drops_convert = 120 := 
by
  sorry

end NUMINAMATH_GPT_calculate_treatment_received_l1580_158067


namespace NUMINAMATH_GPT_period_of_time_l1580_158094

-- We define the annual expense and total amount spent as constants
def annual_expense : ℝ := 2
def total_amount_spent : ℝ := 20

-- Theorem to prove the period of time (in years)
theorem period_of_time : total_amount_spent / annual_expense = 10 :=
by 
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_period_of_time_l1580_158094


namespace NUMINAMATH_GPT_police_arrangements_l1580_158035

theorem police_arrangements (officers : Fin 5) (A B : Fin 5) (intersections : Fin 3) :
  A ≠ B →
  (∃ arrangement : Fin 5 → Fin 3, (∀ i j : Fin 3, i ≠ j → ∃ off : Fin 5, arrangement off = i ∧ arrangement off = j) ∧
    arrangement A = arrangement B) →
  ∃ arrangements_count : Nat, arrangements_count = 36 :=
by
  sorry

end NUMINAMATH_GPT_police_arrangements_l1580_158035


namespace NUMINAMATH_GPT_stories_in_building_l1580_158063

-- Definitions of the conditions
def apartments_per_floor := 4
def people_per_apartment := 2
def total_people := 200

-- Definition of people per floor
def people_per_floor := apartments_per_floor * people_per_apartment

-- The theorem stating the desired conclusion
theorem stories_in_building :
  total_people / people_per_floor = 25 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_stories_in_building_l1580_158063


namespace NUMINAMATH_GPT_pens_in_shop_l1580_158090

theorem pens_in_shop (P Pe E : ℕ) (h_ratio : 14 * Pe = 4 * P) (h_ratio2 : 14 * E = 14 * 3 + 11) (h_P : P = 140) (h_E : E = 30) : Pe = 40 :=
sorry

end NUMINAMATH_GPT_pens_in_shop_l1580_158090


namespace NUMINAMATH_GPT_hydras_never_die_l1580_158031

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end NUMINAMATH_GPT_hydras_never_die_l1580_158031


namespace NUMINAMATH_GPT_standard_equation_of_hyperbola_l1580_158044

noncomputable def ellipse_eccentricity_problem
  (e : ℚ) (a_maj : ℕ) (f_1 f_2 : ℝ × ℝ) (d : ℕ) : Prop :=
  e = 5 / 13 ∧
  a_maj = 26 ∧
  f_1 = (-5, 0) ∧
  f_2 = (5, 0) ∧
  d = 8 →
  ∃ b, (2 * b = 3) ∧ (2 * b ≠ 0) ∧
  ∃ h k : ℝ, (0 ≤  h) ∧ (0 ≤ k) ∧
  ((h^2)/(4^2)) - ((k^2)/(3^2)) = 1

-- problem statement: 
theorem standard_equation_of_hyperbola
  (e : ℚ) (a_maj : ℕ) (f_1 f_2 : ℝ × ℝ) (d : ℕ)
  (h : e = 5 / 13)
  (a_maj_length : a_maj = 26)
  (f1_coords : f_1 = (-5, 0))
  (f2_coords : f_2 = (5, 0))
  (distance_diff : d = 8) :
  ellipse_eccentricity_problem e a_maj f_1 f_2 d :=
sorry

end NUMINAMATH_GPT_standard_equation_of_hyperbola_l1580_158044


namespace NUMINAMATH_GPT_exists_two_numbers_with_gcd_quotient_ge_p_plus_one_l1580_158056

theorem exists_two_numbers_with_gcd_quotient_ge_p_plus_one (p : ℕ) (hp : Nat.Prime p)
  (l : List ℕ) (hl_len : l.length = p + 1) (hl_distinct : l.Nodup) :
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ l ∧ b ∈ l ∧ a > b ∧ a / (Nat.gcd a b) ≥ p + 1 := sorry

end NUMINAMATH_GPT_exists_two_numbers_with_gcd_quotient_ge_p_plus_one_l1580_158056


namespace NUMINAMATH_GPT_maximum_triangle_area_le_8_l1580_158095

def lengths : List ℝ := [2, 3, 4, 5, 6]

-- Function to determine if three lengths can form a valid triangle
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a 

-- Heron's formula to compute the area of a triangle given its sides
noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Statement to prove that the maximum possible area with given stick lengths is less than or equal to 8 cm²
theorem maximum_triangle_area_le_8 :
  ∃ (a b c : ℝ), a ∈ lengths ∧ b ∈ lengths ∧ c ∈ lengths ∧ 
  is_valid_triangle a b c ∧ heron_area a b c ≤ 8 :=
sorry

end NUMINAMATH_GPT_maximum_triangle_area_le_8_l1580_158095


namespace NUMINAMATH_GPT_expand_product_l1580_158057

theorem expand_product (x a : ℝ) : 2 * (x + (a + 2)) * (x + (a - 3)) = 2 * x^2 + (4 * a - 2) * x + 2 * a^2 - 2 * a - 12 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1580_158057


namespace NUMINAMATH_GPT_tip_percentage_is_20_l1580_158051

theorem tip_percentage_is_20 (total_spent price_before_tax_and_tip : ℝ) (sales_tax_rate : ℝ) (h1 : total_spent = 158.40) (h2 : price_before_tax_and_tip = 120) (h3 : sales_tax_rate = 0.10) :
  ((total_spent - (price_before_tax_and_tip * (1 + sales_tax_rate))) / (price_before_tax_and_tip * (1 + sales_tax_rate))) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_tip_percentage_is_20_l1580_158051


namespace NUMINAMATH_GPT_inequality_D_holds_l1580_158062

theorem inequality_D_holds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := 
sorry

end NUMINAMATH_GPT_inequality_D_holds_l1580_158062


namespace NUMINAMATH_GPT_possible_last_digits_count_l1580_158061

theorem possible_last_digits_count : 
  ∃ s : Finset Nat, s = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ ∀ n ∈ s, ∃ m, (m % 10 = n) ∧ (m % 3 = 0) := 
sorry

end NUMINAMATH_GPT_possible_last_digits_count_l1580_158061


namespace NUMINAMATH_GPT_barbed_wire_cost_l1580_158047

noncomputable def total_cost_barbed_wire (area : ℕ) (cost_per_meter : ℝ) (gate_width : ℕ) : ℝ :=
  let s := Real.sqrt area
  let perimeter := 4 * s - 2 * gate_width
  perimeter * cost_per_meter

theorem barbed_wire_cost :
  total_cost_barbed_wire 3136 3.5 1 = 777 := by
  sorry

end NUMINAMATH_GPT_barbed_wire_cost_l1580_158047


namespace NUMINAMATH_GPT_pentagon_diagl_sum_pentagon_diagonal_391_l1580_158092

noncomputable def diagonal_sum (AB CD BC DE AE : ℕ) 
  (AC : ℚ) (BD : ℚ) (CE : ℚ) (AD : ℚ) (BE : ℚ) : ℚ :=
  3 * AC + AD + BE

theorem pentagon_diagl_sum (AB CD BC DE AE : ℕ)
  (hAB : AB = 3) (hCD : CD = 3) 
  (hBC : BC = 10) (hDE : DE = 10) 
  (hAE : AE = 14)
  (AC BD CE AD BE : ℚ)
  (hACBC : AC = 12) 
  (hADBC: AD = 13.5)
  (hCEBE: BE = 44 / 3) :
  diagonal_sum AB CD BC DE AE AC BD CE AD BE = 385 / 6 := sorry

theorem pentagon_diagonal_391 (AB CD BC DE AE : ℕ)
  (hAB : AB = 3) (hCD : CD = 3) 
  (hBC : BC = 10) (hDE : DE = 10) 
  (hAE : AE = 14)
  (AC BD CE AD BE : ℚ)
  (hACBC : AC = 12) 
  (hADBC: AD = 13.5)
  (hCEBE: BE = 44 / 3) :
  ∃ m n : ℕ, 
    m.gcd n = 1 ∧
    m / n = 385 / 6 ∧
    m + n = 391 := sorry

end NUMINAMATH_GPT_pentagon_diagl_sum_pentagon_diagonal_391_l1580_158092


namespace NUMINAMATH_GPT_no_such_triangle_exists_l1580_158070

theorem no_such_triangle_exists (a b c : ℝ) (h1 : c = 0.2 * a) (h2 : b = 0.25 * (a + b + c)) :
  ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end NUMINAMATH_GPT_no_such_triangle_exists_l1580_158070


namespace NUMINAMATH_GPT_moles_of_water_produced_l1580_158036

-- Definitions for the chemical reaction
def moles_NaOH := 4
def moles_H₂SO₄ := 2

-- The balanced chemical equation tells us the ratio of NaOH to H₂O
def chemical_equation (moles_NaOH moles_H₂SO₄ moles_H₂O moles_Na₂SO₄: ℕ) : Prop :=
  2 * moles_NaOH = 2 * moles_H₂O ∧ moles_H₂SO₄ = 1 ∧ moles_Na₂SO₄ = 1

-- The actual proof statement
theorem moles_of_water_produced : 
  ∀ (m_NaOH m_H₂SO₄ m_Na₂SO₄ : ℕ), 
  chemical_equation m_NaOH m_H₂SO₄ 4 m_Na₂SO₄ → moles_H₂O = 4 :=
by
  intros m_NaOH m_H₂SO₄ m_Na₂SO₄ chem_eq
  -- Placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_moles_of_water_produced_l1580_158036


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1580_158074

theorem arithmetic_sequence_problem
  (a : ℕ → ℤ)
  (h1 : a 6 + a 9 = 16)
  (h2 : a 4 = 1)
  (h_arith : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q) :
  a 11 = 15 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1580_158074


namespace NUMINAMATH_GPT_find_x_l1580_158012

def star (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_x (x : ℝ) (h : star 4 x = 52) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1580_158012


namespace NUMINAMATH_GPT_skirt_price_is_13_l1580_158064

-- Definitions based on conditions
def skirts_cost (S : ℝ) : ℝ := 2 * S
def blouses_cost : ℝ := 3 * 6
def total_cost (S : ℝ) : ℝ := skirts_cost S + blouses_cost
def amount_spent : ℝ := 100 - 56

-- The statement we want to prove
theorem skirt_price_is_13 (S : ℝ) (h : total_cost S = amount_spent) : S = 13 :=
by sorry

end NUMINAMATH_GPT_skirt_price_is_13_l1580_158064


namespace NUMINAMATH_GPT_purely_imaginary_sol_l1580_158052

theorem purely_imaginary_sol {m : ℝ} (h : (m^2 - 3 * m) = 0) (h2 : (m^2 - 5 * m + 6) ≠ 0) : m = 0 :=
sorry

end NUMINAMATH_GPT_purely_imaginary_sol_l1580_158052


namespace NUMINAMATH_GPT_smallest_five_digit_multiple_of_53_l1580_158093

theorem smallest_five_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 53 = 0 ∧ n = 10017 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_multiple_of_53_l1580_158093


namespace NUMINAMATH_GPT_parabola_standard_eq_l1580_158029

theorem parabola_standard_eq (p : ℝ) (x y : ℝ) :
  (∃ x y, 3 * x - 4 * y - 12 = 0) →
  ( (p = 6 ∧ x^2 = -12 * y ∧ y = -3) ∨ (p = 8 ∧ y^2 = 16 * x ∧ x = 4)) :=
sorry

end NUMINAMATH_GPT_parabola_standard_eq_l1580_158029


namespace NUMINAMATH_GPT_first_person_work_days_l1580_158054

theorem first_person_work_days (x : ℝ) (h1 : 0 < x) :
  (1/x + 1/40 = 1/15) → x = 24 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_first_person_work_days_l1580_158054


namespace NUMINAMATH_GPT_day_in_43_days_is_wednesday_l1580_158004

-- Define a function to represent the day of the week after a certain number of days
def day_of_week (n : ℕ) : ℕ := n % 7

-- Use an enum or some notation to represent the days of the week, but this is implicit in our setup.
-- We assume the days are numbered from 0 to 6 with 0 representing Tuesday.
def Tuesday : ℕ := 0
def Wednesday : ℕ := 1

-- Theorem to prove that 43 days after Tuesday is a Wednesday
theorem day_in_43_days_is_wednesday : day_of_week (Tuesday + 43) = Wednesday :=
by
  sorry

end NUMINAMATH_GPT_day_in_43_days_is_wednesday_l1580_158004


namespace NUMINAMATH_GPT_number_of_digits_in_sum_l1580_158016

theorem number_of_digits_in_sum (C D : ℕ) (hC : C ≠ 0 ∧ C < 10) (hD : D % 2 = 0 ∧ D < 10) : 
  (Nat.digits 10 (8765 + (C * 100 + 43) + (D * 10 + 2))).length = 4 := 
by
  sorry

end NUMINAMATH_GPT_number_of_digits_in_sum_l1580_158016


namespace NUMINAMATH_GPT_total_legs_on_farm_l1580_158008

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

end NUMINAMATH_GPT_total_legs_on_farm_l1580_158008


namespace NUMINAMATH_GPT_unique_nonzero_b_l1580_158098

variable (a b m n : ℝ)
variable (h_ne : m ≠ n)
variable (h_m_nonzero : m ≠ 0)
variable (h_n_nonzero : n ≠ 0)

theorem unique_nonzero_b (h : (a * m + b * n + m)^2 - (a * m + b * n + n)^2 = (m - n)^2) : 
  a = 0 ∧ b = -1 :=
sorry

end NUMINAMATH_GPT_unique_nonzero_b_l1580_158098


namespace NUMINAMATH_GPT_solve_for_x_l1580_158006

theorem solve_for_x (x : ℝ) (h : (2 * x - 3) ^ (x + 3) = 1) : 
  x = -3 ∨ x = 2 ∨ x = 1 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l1580_158006


namespace NUMINAMATH_GPT_minimum_value_ab_l1580_158069

theorem minimum_value_ab (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h : a * b - 2 * a - b = 0) :
  8 ≤ a * b :=
by sorry

end NUMINAMATH_GPT_minimum_value_ab_l1580_158069


namespace NUMINAMATH_GPT_additional_grassy_ground_l1580_158040

theorem additional_grassy_ground (r1 r2 : ℝ) (π : ℝ) :
  r1 = 12 → r2 = 18 → π = Real.pi →
  (π * r2^2 - π * r1^2) = 180 * π := by
sorry

end NUMINAMATH_GPT_additional_grassy_ground_l1580_158040


namespace NUMINAMATH_GPT_min_shoeing_time_l1580_158041

theorem min_shoeing_time
  (num_blacksmiths : ℕ) (num_horses : ℕ) (hooves_per_horse : ℕ) (minutes_per_hoof : ℕ)
  (h_blacksmiths : num_blacksmiths = 48)
  (h_horses : num_horses = 60)
  (h_hooves_per_horse : hooves_per_horse = 4)
  (h_minutes_per_hoof : minutes_per_hoof = 5) :
  (num_horses * hooves_per_horse * minutes_per_hoof) / num_blacksmiths = 25 := 
by
  sorry

end NUMINAMATH_GPT_min_shoeing_time_l1580_158041


namespace NUMINAMATH_GPT_horizontal_asymptote_degree_l1580_158022

noncomputable def degree (p : Polynomial ℝ) : ℕ := Polynomial.natDegree p

theorem horizontal_asymptote_degree (p : Polynomial ℝ) :
  (∃ l : ℝ, ∀ ε > 0, ∃ N, ∀ x > N, |(p.eval x / (3 * x^7 - 2 * x^3 + x - 4)) - l| < ε) →
  degree p ≤ 7 :=
sorry

end NUMINAMATH_GPT_horizontal_asymptote_degree_l1580_158022


namespace NUMINAMATH_GPT_zachary_seventh_day_cans_l1580_158072

-- Define the number of cans found by Zachary every day.
def cans_found_on (day : ℕ) : ℕ :=
  if day = 1 then 4
  else if day = 2 then 9
  else if day = 3 then 14
  else 5 * (day - 1) - 1

-- The theorem to prove the number of cans found on the seventh day.
theorem zachary_seventh_day_cans : cans_found_on 7 = 34 :=
by 
  sorry

end NUMINAMATH_GPT_zachary_seventh_day_cans_l1580_158072


namespace NUMINAMATH_GPT_cost_of_brushes_and_canvas_minimum_canvases_cost_effectiveness_l1580_158042

-- Part 1: Prove the cost of one box of brushes and one canvas each.
theorem cost_of_brushes_and_canvas (x y : ℕ) 
    (h₁ : 2 * x + 4 * y = 94) (h₂ : 4 * x + 2 * y = 98) :
    x = 17 ∧ y = 15 := by
  sorry

-- Part 2: Prove the minimum number of canvases.
theorem minimum_canvases (m : ℕ) 
    (h₃ : m + (10 - m) = 10) (h₄ : 17 * (10 - m) + 15 * m ≤ 157) :
    m ≥ 7 := by
  sorry

-- Part 3: Prove the cost-effective purchasing plan.
theorem cost_effectiveness (m n : ℕ) 
    (h₃ : m + n = 10) (h₄ : 17 * n + 15 * m ≤ 157) (h₅ : m ≤ 8) :
    (m = 8 ∧ n = 2) := by
  sorry

end NUMINAMATH_GPT_cost_of_brushes_and_canvas_minimum_canvases_cost_effectiveness_l1580_158042


namespace NUMINAMATH_GPT_possible_values_of_n_l1580_158076

theorem possible_values_of_n (n : ℕ) (h1 : 0 < n)
  (h2 : 12 * n^3 = n^4 + 11 * n^2) :
  n = 1 ∨ n = 11 :=
sorry

end NUMINAMATH_GPT_possible_values_of_n_l1580_158076


namespace NUMINAMATH_GPT_sum_odd_integers_correct_l1580_158083

def sum_odd_integers_from_13_to_41 : ℕ := 
  let a := 13
  let l := 41
  let n := 15
  n * (a + l) / 2

theorem sum_odd_integers_correct : sum_odd_integers_from_13_to_41 = 405 :=
  by sorry

end NUMINAMATH_GPT_sum_odd_integers_correct_l1580_158083


namespace NUMINAMATH_GPT_different_routes_calculation_l1580_158019

-- Definitions for the conditions
def west_blocks := 3
def south_blocks := 2
def east_blocks := 3
def north_blocks := 3

-- Calculation of combinations for the number of sequences
def house_to_sw_corner_routes := Nat.choose (west_blocks + south_blocks) south_blocks
def ne_corner_to_school_routes := Nat.choose (east_blocks + north_blocks) east_blocks

-- Proving the total number of routes
theorem different_routes_calculation : 
  house_to_sw_corner_routes * 1 * ne_corner_to_school_routes = 200 :=
by
  -- Mathematical proof steps (to be filled)
  sorry

end NUMINAMATH_GPT_different_routes_calculation_l1580_158019


namespace NUMINAMATH_GPT_Carissa_ran_at_10_feet_per_second_l1580_158025

theorem Carissa_ran_at_10_feet_per_second :
  ∀ (n : ℕ), 
  (∃ (a : ℕ), 
    (2 * a + 2 * n^2 * a = 260) ∧ -- Total distance
    (a + n * a = 30)) → -- Total time spent
  (2 * n = 10) :=
by
  intro n
  intro h
  sorry

end NUMINAMATH_GPT_Carissa_ran_at_10_feet_per_second_l1580_158025


namespace NUMINAMATH_GPT_chord_bisection_l1580_158039

theorem chord_bisection {r : ℝ} (PQ RS : Set (ℝ × ℝ)) (O T P Q R S M : ℝ × ℝ)
  (radius_OP : dist O P = 6) (radius_OQ : dist O Q = 6)
  (radius_OR : dist O R = 6) (radius_OS : dist O S = 6) (radius_OT : dist O T = 6)
  (radius_OM : dist O M = 2 * Real.sqrt 13) 
  (PT_eq_8 : dist P T = 8) (TQ_eq_8 : dist T Q = 8)
  (sin_theta_eq_4_5 : Real.sin (Real.arcsin (8 / 10)) = 4 / 5) :
  4 * 5 = 20 :=
by
  sorry

end NUMINAMATH_GPT_chord_bisection_l1580_158039


namespace NUMINAMATH_GPT_sheep_to_horses_ratio_l1580_158014

-- Define the known quantities
def number_of_sheep := 32
def total_horse_food := 12880
def food_per_horse := 230

-- Calculate number of horses
def number_of_horses := total_horse_food / food_per_horse

-- Calculate and simplify the ratio of sheep to horses
def ratio_of_sheep_to_horses := (number_of_sheep : ℚ) / (number_of_horses : ℚ)

-- Define the expected simplified ratio
def expected_ratio_of_sheep_to_horses := (4 : ℚ) / (7 : ℚ)

-- The statement we want to prove
theorem sheep_to_horses_ratio : ratio_of_sheep_to_horses = expected_ratio_of_sheep_to_horses :=
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_sheep_to_horses_ratio_l1580_158014


namespace NUMINAMATH_GPT_shari_effective_distance_l1580_158024

-- Define the given conditions
def constant_rate : ℝ := 4 -- miles per hour
def wind_resistance : ℝ := 0.5 -- miles per hour
def walking_time : ℝ := 2 -- hours

-- Define the effective walking speed considering wind resistance
def effective_speed : ℝ := constant_rate - wind_resistance

-- Define the effective walking distance
def effective_distance : ℝ := effective_speed * walking_time

-- State that Shari effectively walks 7.0 miles
theorem shari_effective_distance :
  effective_distance = 7.0 :=
by
  sorry

end NUMINAMATH_GPT_shari_effective_distance_l1580_158024


namespace NUMINAMATH_GPT_man_l1580_158027

theorem man's_speed_kmph (length_train : ℝ) (time_seconds : ℝ) (speed_train_kmph : ℝ) : ℝ :=
  let speed_train_mps := speed_train_kmph * (5/18)
  let rel_speed_mps := length_train / time_seconds
  let man_speed_mps := rel_speed_mps - speed_train_mps
  man_speed_mps * (18/5)

example : man's_speed_kmph 120 6 65.99424046076315 = 6.00735873483709 := by
  sorry

end NUMINAMATH_GPT_man_l1580_158027


namespace NUMINAMATH_GPT_total_pages_written_l1580_158073

-- Define the conditions
def timeMon : ℕ := 60  -- Minutes on Monday
def rateMon : ℕ := 30  -- Minutes per page on Monday

def timeTue : ℕ := 45  -- Minutes on Tuesday
def rateTue : ℕ := 15  -- Minutes per page on Tuesday

def pagesWed : ℕ := 5  -- Pages written on Wednesday

-- Function to compute pages written based on time and rate
def pages_written (time rate : ℕ) : ℕ := time / rate

-- Define the theorem to be proved
theorem total_pages_written :
  pages_written timeMon rateMon + pages_written timeTue rateTue + pagesWed = 10 :=
sorry

end NUMINAMATH_GPT_total_pages_written_l1580_158073


namespace NUMINAMATH_GPT_sphere_surface_area_l1580_158078

theorem sphere_surface_area (a b c : ℝ)
  (h1 : a * b * c = Real.sqrt 6)
  (h2 : a * b = Real.sqrt 2)
  (h3 : b * c = Real.sqrt 3) :
  4 * Real.pi * (Real.sqrt (a^2 + b^2 + c^2) / 2) ^ 2 = 6 * Real.pi :=
sorry

end NUMINAMATH_GPT_sphere_surface_area_l1580_158078


namespace NUMINAMATH_GPT_min_troublemakers_in_class_l1580_158018

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end NUMINAMATH_GPT_min_troublemakers_in_class_l1580_158018


namespace NUMINAMATH_GPT_math_team_count_l1580_158017

open Nat

theorem math_team_count :
  let girls := 7
  let boys := 12
  let total_team := 16
  let count_ways (n k : ℕ) := choose n k
  (count_ways girls 3) * (count_ways boys 5) * (count_ways (girls - 3 + boys - 5) 8) = 456660 :=
by
  sorry

end NUMINAMATH_GPT_math_team_count_l1580_158017


namespace NUMINAMATH_GPT_lateral_surface_area_of_rotated_triangle_l1580_158015

theorem lateral_surface_area_of_rotated_triangle :
  let AC := 3
  let BC := 4
  let AB := Real.sqrt (AC ^ 2 + BC ^ 2)
  let radius := BC
  let slant_height := AB
  let lateral_surface_area := Real.pi * radius * slant_height
  lateral_surface_area = 20 * Real.pi := by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_rotated_triangle_l1580_158015


namespace NUMINAMATH_GPT_function_is_increasing_on_interval_l1580_158048

noncomputable def f (m x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * m * x^2 + 4 * x - 3

theorem function_is_increasing_on_interval {m : ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (1/3) * x^3 - (1/2) * m * x^2 + 4 * x - 3 ≥ (1/3) * (x - dx)^3 - (1/2) * m * (x - dx)^2 + 4 * (x - dx) - 3)
  ↔ m ≤ 4 :=
sorry

end NUMINAMATH_GPT_function_is_increasing_on_interval_l1580_158048


namespace NUMINAMATH_GPT_log_simplification_l1580_158032

theorem log_simplification :
  (1 / (Real.log 3 / Real.log 12 + 2))
  + (1 / (Real.log 2 / Real.log 8 + 2))
  + (1 / (Real.log 3 / Real.log 9 + 2)) = 2 :=
  sorry

end NUMINAMATH_GPT_log_simplification_l1580_158032


namespace NUMINAMATH_GPT_num_geography_books_l1580_158071

theorem num_geography_books
  (total_books : ℕ)
  (history_books : ℕ)
  (math_books : ℕ)
  (h1 : total_books = 100)
  (h2 : history_books = 32)
  (h3 : math_books = 43) :
  total_books - history_books - math_books = 25 :=
by
  sorry

end NUMINAMATH_GPT_num_geography_books_l1580_158071


namespace NUMINAMATH_GPT_largest_quantity_l1580_158030

theorem largest_quantity 
  (A := (2010 / 2009) + (2010 / 2011))
  (B := (2012 / 2011) + (2010 / 2011))
  (C := (2011 / 2010) + (2011 / 2012)) : C > A ∧ C > B := 
by {
  sorry
}

end NUMINAMATH_GPT_largest_quantity_l1580_158030


namespace NUMINAMATH_GPT_caroline_citrus_drinks_l1580_158003

-- Definitions based on problem conditions
def citrus_drinks (oranges : ℕ) : ℕ := (oranges * 8) / 3

-- Define problem statement
theorem caroline_citrus_drinks : citrus_drinks 21 = 56 :=
by
  sorry

end NUMINAMATH_GPT_caroline_citrus_drinks_l1580_158003


namespace NUMINAMATH_GPT_number_of_players_taking_mathematics_l1580_158045

-- Define the conditions
def total_players := 15
def players_physics := 10
def players_both := 4

-- Define the conclusion to be proven
theorem number_of_players_taking_mathematics : (total_players - players_physics + players_both) = 9 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_number_of_players_taking_mathematics_l1580_158045


namespace NUMINAMATH_GPT_geometric_series_sum_l1580_158007

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℚ) / 4
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1580_158007


namespace NUMINAMATH_GPT_smallest_K_222_multiple_of_198_l1580_158002

theorem smallest_K_222_multiple_of_198 :
  ∀ K : ℕ, (∃ x : ℕ, x = 2 * (10^K - 1) / 9 ∧ x % 198 = 0) → K = 18 :=
by
  sorry

end NUMINAMATH_GPT_smallest_K_222_multiple_of_198_l1580_158002


namespace NUMINAMATH_GPT_rate_percent_simple_interest_l1580_158055

theorem rate_percent_simple_interest (SI P T R : ℝ) (h₁ : SI = 500) (h₂ : P = 2000) (h₃ : T = 2)
  (h₄ : SI = (P * R * T) / 100) : R = 12.5 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_rate_percent_simple_interest_l1580_158055


namespace NUMINAMATH_GPT_inverse_proposition_equivalence_l1580_158099

theorem inverse_proposition_equivalence (x y : ℝ) :
  (x = y → abs x = abs y) ↔ (abs x = abs y → x = y) :=
sorry

end NUMINAMATH_GPT_inverse_proposition_equivalence_l1580_158099


namespace NUMINAMATH_GPT_cafeteria_extra_fruits_l1580_158028

def extra_fruits (ordered wanted : Nat) : Nat :=
  ordered - wanted

theorem cafeteria_extra_fruits :
  let red_apples_ordered := 6
  let red_apples_wanted := 5
  let green_apples_ordered := 15
  let green_apples_wanted := 8
  let oranges_ordered := 10
  let oranges_wanted := 6
  let bananas_ordered := 8
  let bananas_wanted := 7
  extra_fruits red_apples_ordered red_apples_wanted = 1 ∧
  extra_fruits green_apples_ordered green_apples_wanted = 7 ∧
  extra_fruits oranges_ordered oranges_wanted = 4 ∧
  extra_fruits bananas_ordered bananas_wanted = 1 := 
by
  sorry

end NUMINAMATH_GPT_cafeteria_extra_fruits_l1580_158028


namespace NUMINAMATH_GPT_roll_four_fair_dice_l1580_158046
noncomputable def roll_four_fair_dice_prob : ℚ :=
  let total_outcomes : ℚ := 6^4
  let favorable_outcomes : ℚ := 6
  let prob_all_same : ℚ := favorable_outcomes / total_outcomes
  let prob_not_all_same : ℚ := 1 - prob_all_same
  prob_not_all_same

theorem roll_four_fair_dice :
  roll_four_fair_dice_prob = 215 / 216 :=
by
  sorry

end NUMINAMATH_GPT_roll_four_fair_dice_l1580_158046


namespace NUMINAMATH_GPT_part1_part2_l1580_158021

variable (x k : ℝ)

-- Part (1)
theorem part1 (h1 : x = 3) : ∀ k : ℝ, (1 + k) * 3 ≤ k^2 + k + 4 := sorry

-- Part (2)
theorem part2 (h2 : ∀ k : ℝ, -4 ≤ k → (1 + k) * x ≤ k^2 + k + 4) : -5 ≤ x ∧ x ≤ 3 := sorry

end NUMINAMATH_GPT_part1_part2_l1580_158021


namespace NUMINAMATH_GPT_find_7th_term_l1580_158010

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem find_7th_term 
    (a d : ℤ) 
    (h3 : a + 2 * d = 17) 
    (h5 : a + 4 * d = 39) : 
    arithmetic_sequence a d 7 = 61 := 
sorry

end NUMINAMATH_GPT_find_7th_term_l1580_158010


namespace NUMINAMATH_GPT_mila_calculator_sum_l1580_158000

theorem mila_calculator_sum :
  let n := 60
  let calc1_start := 2
  let calc2_start := 0
  let calc3_start := -1
  calc1_start^(3^n) + calc2_start^2^(n) + (-calc3_start)^n = 2^(3^60) + 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_mila_calculator_sum_l1580_158000


namespace NUMINAMATH_GPT_percentage_reduction_l1580_158034

theorem percentage_reduction (P : ℝ) (h1 : 700 / P + 3 = 700 / 70) : 
  ((P - 70) / P) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_reduction_l1580_158034
