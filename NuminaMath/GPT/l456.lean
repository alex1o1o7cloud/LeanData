import Mathlib

namespace NUMINAMATH_GPT_ratio_as_percentage_l456_45618

theorem ratio_as_percentage (x : ℝ) (h : (x / 2) / (3 * x / 5) = 3 / 5) : 
  (3 / 5) * 100 = 60 := 
sorry

end NUMINAMATH_GPT_ratio_as_percentage_l456_45618


namespace NUMINAMATH_GPT_solution_set_l456_45616

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

variable {f : ℝ → ℝ}

-- Hypotheses
axiom odd_f : is_odd f
axiom increasing_f : is_increasing f
axiom f_of_neg_three : f (-3) = 0

-- Theorem statement
theorem solution_set (x : ℝ) : (x - 3) * f (x - 3) < 0 ↔ (0 < x ∧ x < 3) ∨ (3 < x ∧ x < 6) :=
sorry

end NUMINAMATH_GPT_solution_set_l456_45616


namespace NUMINAMATH_GPT_courier_total_travel_times_l456_45658

-- Define the conditions
variables (v1 v2 : ℝ) (t : ℝ)
axiom speed_condition_1 : v1 * (t + 16) = (v1 + v2) * t
axiom speed_condition_2 : v2 * (t + 9) = (v1 + v2) * t
axiom time_condition : t = 12

-- Define the total travel times
def total_travel_time_1 : ℝ := t + 16
def total_travel_time_2 : ℝ := t + 9

-- Proof problem statement
theorem courier_total_travel_times :
  total_travel_time_1 = 28 ∧ total_travel_time_2 = 21 :=
by
  sorry

end NUMINAMATH_GPT_courier_total_travel_times_l456_45658


namespace NUMINAMATH_GPT_taxi_ride_cost_is_five_dollars_l456_45639

def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def miles_traveled : ℝ := 10.0
def total_cost : ℝ := base_fare + (cost_per_mile * miles_traveled)

theorem taxi_ride_cost_is_five_dollars : total_cost = 5.00 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_is_five_dollars_l456_45639


namespace NUMINAMATH_GPT_estimate_fish_population_l456_45671

theorem estimate_fish_population (n m k : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) (h4 : k ≤ m) : 
  ∃ N : ℕ, N = m * n / k :=
by
  sorry

end NUMINAMATH_GPT_estimate_fish_population_l456_45671


namespace NUMINAMATH_GPT_percentage_of_400_equals_100_l456_45679

def part : ℝ := 100
def whole : ℝ := 400

theorem percentage_of_400_equals_100 : (part / whole) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_percentage_of_400_equals_100_l456_45679


namespace NUMINAMATH_GPT_correct_calculation_D_l456_45648

theorem correct_calculation_D (m : ℕ) : 
  (2 * m ^ 3) * (3 * m ^ 2) = 6 * m ^ 5 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_D_l456_45648


namespace NUMINAMATH_GPT_janet_dresses_pockets_l456_45673

theorem janet_dresses_pockets :
  ∀ (x : ℕ), (∀ (dresses_with_pockets remaining_dresses total_pockets : ℕ),
  dresses_with_pockets = 24 / 2 →
  total_pockets = 32 →
  remaining_dresses = dresses_with_pockets - dresses_with_pockets / 3 →
  (dresses_with_pockets / 3) * x + remaining_dresses * 3 = total_pockets →
  x = 2) :=
by
  intros x dresses_with_pockets remaining_dresses total_pockets h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_janet_dresses_pockets_l456_45673


namespace NUMINAMATH_GPT_total_cost_l456_45643

-- Definitions based on the problem's conditions
def cost_hamburger : ℕ := 4
def cost_milkshake : ℕ := 3

def qty_hamburgers : ℕ := 7
def qty_milkshakes : ℕ := 6

-- The proof statement
theorem total_cost :
  (qty_hamburgers * cost_hamburger + qty_milkshakes * cost_milkshake) = 46 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l456_45643


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l456_45659

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l456_45659


namespace NUMINAMATH_GPT_fraction_order_l456_45634

theorem fraction_order:
  let frac1 := (21 : ℚ) / 17
  let frac2 := (22 : ℚ) / 19
  let frac3 := (18 : ℚ) / 15
  let frac4 := (20 : ℚ) / 16
  frac2 < frac3 ∧ frac3 < frac1 ∧ frac1 < frac4 := 
sorry

end NUMINAMATH_GPT_fraction_order_l456_45634


namespace NUMINAMATH_GPT_apple_percentage_is_23_l456_45613

def total_responses := 70 + 80 + 50 + 30 + 70
def apple_responses := 70

theorem apple_percentage_is_23 :
  (apple_responses : ℝ) / (total_responses : ℝ) * 100 = 23 := 
by
  sorry

end NUMINAMATH_GPT_apple_percentage_is_23_l456_45613


namespace NUMINAMATH_GPT_right_triangle_sides_l456_45633

theorem right_triangle_sides (x y z : ℕ) (h1 : x + y + z = 30)
    (h2 : x^2 + y^2 + z^2 = 338) (h3 : x^2 + y^2 = z^2) :
    (x = 5 ∧ y = 12 ∧ z = 13) ∨ (x = 12 ∧ y = 5 ∧ z = 13) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_sides_l456_45633


namespace NUMINAMATH_GPT_area_of_right_isosceles_triangle_l456_45623

def is_right_isosceles (a b c : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = c^2

theorem area_of_right_isosceles_triangle (a b c : ℝ) (h : is_right_isosceles a b c) (h_hypotenuse : c = 10) :
  1/2 * a * b = 25 :=
by
  sorry

end NUMINAMATH_GPT_area_of_right_isosceles_triangle_l456_45623


namespace NUMINAMATH_GPT_mangoes_total_l456_45682

theorem mangoes_total (M A : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : A = 60) :
  A + M = 75 :=
by
  sorry

end NUMINAMATH_GPT_mangoes_total_l456_45682


namespace NUMINAMATH_GPT_find_k_l456_45611

-- Identifying conditions from the problem
def point (x : ℝ) : ℝ × ℝ := (x, x^3)  -- A point on the curve y = x^3
def tangent_slope (x : ℝ) : ℝ := 3 * x^2  -- The slope of the tangent to the curve y = x^3 at point (x, x^3)
def tangent_line (x k : ℝ) : ℝ := k * x + 2  -- The given tangent line equation

-- Question as a proof problem
theorem find_k (x : ℝ) (k : ℝ) (h : tangent_line x k = x^3) : k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l456_45611


namespace NUMINAMATH_GPT_count_whole_numbers_in_interval_l456_45656

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end NUMINAMATH_GPT_count_whole_numbers_in_interval_l456_45656


namespace NUMINAMATH_GPT_system_of_equations_solve_l456_45696

theorem system_of_equations_solve (x y : ℝ) 
  (h1 : 2 * x + y = 5)
  (h2 : x + 2 * y = 4) :
  x + y = 3 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solve_l456_45696


namespace NUMINAMATH_GPT_least_number_subtracted_divisible_17_l456_45601

theorem least_number_subtracted_divisible_17 :
  ∃ n : ℕ, 165826 - n % 17 = 0 ∧ n = 12 :=
by
  use 12
  sorry  -- Proof will go here.

end NUMINAMATH_GPT_least_number_subtracted_divisible_17_l456_45601


namespace NUMINAMATH_GPT_kristy_gave_to_brother_l456_45690

def total_cookies : Nat := 22
def kristy_ate : Nat := 2
def first_friend_took : Nat := 3
def second_friend_took : Nat := 5
def third_friend_took : Nat := 5
def cookies_left : Nat := 6

theorem kristy_gave_to_brother :
  kristy_ate + first_friend_took + second_friend_took + third_friend_took = 15 ∧
  total_cookies - cookies_left - (kristy_ate + first_friend_took + second_friend_took + third_friend_took) = 1 :=
by
  sorry

end NUMINAMATH_GPT_kristy_gave_to_brother_l456_45690


namespace NUMINAMATH_GPT_build_wall_time_l456_45668

theorem build_wall_time {d : ℝ} : 
  (15 * 1 + 3 * 2) * 3 = 63 ∧ 
  (25 * 1 + 5 * 2) * d = 63 → 
  d = 1.8 := 
by 
  sorry

end NUMINAMATH_GPT_build_wall_time_l456_45668


namespace NUMINAMATH_GPT_sports_minutes_in_newscast_l456_45608

-- Definitions based on the conditions
def total_newscast_minutes : ℕ := 30
def national_news_minutes : ℕ := 12
def international_news_minutes : ℕ := 5
def weather_forecasts_minutes : ℕ := 2
def advertising_minutes : ℕ := 6

-- The problem statement
theorem sports_minutes_in_newscast (t : ℕ) (n : ℕ) (i : ℕ) (w : ℕ) (a : ℕ) :
  t = 30 → n = 12 → i = 5 → w = 2 → a = 6 → t - n - i - w - a = 5 := 
by sorry

end NUMINAMATH_GPT_sports_minutes_in_newscast_l456_45608


namespace NUMINAMATH_GPT_length_of_AB_l456_45677

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem length_of_AB :
  let O := (0, 0)
  let A := (54^(1/3), 0)
  let B := (0, 54^(1/3))
  distance A B = 54^(1/3) * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l456_45677


namespace NUMINAMATH_GPT_abs_diff_eq_implies_le_l456_45647

theorem abs_diff_eq_implies_le {x y : ℝ} (h : |x - y| = y - x) : x ≤ y := 
by
  sorry

end NUMINAMATH_GPT_abs_diff_eq_implies_le_l456_45647


namespace NUMINAMATH_GPT_cyclist_average_speed_l456_45698

theorem cyclist_average_speed (v : ℝ) 
  (h1 : 8 / v + 10 / 8 = 18 / 8.78) : v = 10 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_average_speed_l456_45698


namespace NUMINAMATH_GPT_fraction_zero_implies_x_eq_neg3_l456_45646

theorem fraction_zero_implies_x_eq_neg3 (x : ℝ) (h1 : x ≠ 3) (h2 : (x^2 - 9) / (x - 3) = 0) : x = -3 :=
sorry

end NUMINAMATH_GPT_fraction_zero_implies_x_eq_neg3_l456_45646


namespace NUMINAMATH_GPT_quadratic_roots_range_l456_45669

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x^2 + (a^2 - 1) * x + a - 2 = 0 ∧ y^2 + (a^2 - 1) * y + a - 2 = 0 ∧ x ≠ y ∧ x > 1 ∧ y < 1) ↔ -2 < a ∧ a < 1 := 
sorry

end NUMINAMATH_GPT_quadratic_roots_range_l456_45669


namespace NUMINAMATH_GPT_maximum_candies_purchase_l456_45619

theorem maximum_candies_purchase (c1 : ℕ) (c4 : ℕ) (c7 : ℕ) (n : ℕ)
    (H_single : c1 = 1)
    (H_pack4  : c4 = 4)
    (H_cost4  : c4 = 3) 
    (H_pack7  : c7 = 7) 
    (H_cost7  : c7 = 4) 
    (H_budget : n = 10) :
    ∃ k : ℕ, k = 16 :=
by
    -- We'll skip the proof since the task requires only the statement
    sorry

end NUMINAMATH_GPT_maximum_candies_purchase_l456_45619


namespace NUMINAMATH_GPT_Marcy_spears_l456_45641

def makeSpears (saplings: ℕ) (logs: ℕ) (branches: ℕ) (trunks: ℕ) : ℕ :=
  3 * saplings + 9 * logs + 7 * branches + 15 * trunks

theorem Marcy_spears :
  makeSpears 12 1 6 0 - (3 * 2) + makeSpears 0 4 0 0 - (9 * 4) + makeSpears 0 0 6 1 - (7 * 0) + makeSpears 0 0 0 2 = 81 := by
  sorry

end NUMINAMATH_GPT_Marcy_spears_l456_45641


namespace NUMINAMATH_GPT_evaluate_polynomial_l456_45635

theorem evaluate_polynomial (x : ℤ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_l456_45635


namespace NUMINAMATH_GPT_gcd_of_3150_and_9800_is_350_l456_45631

-- Definition of the two numbers
def num1 : ℕ := 3150
def num2 : ℕ := 9800

-- The greatest common factor of num1 and num2 is 350
theorem gcd_of_3150_and_9800_is_350 : Nat.gcd num1 num2 = 350 := by
  sorry

end NUMINAMATH_GPT_gcd_of_3150_and_9800_is_350_l456_45631


namespace NUMINAMATH_GPT_pradeep_max_marks_l456_45626

-- conditions
variables (M : ℝ)
variable (h1 : 0.40 * M = 220)

-- question and answer
theorem pradeep_max_marks : M = 550 :=
by
  sorry

end NUMINAMATH_GPT_pradeep_max_marks_l456_45626


namespace NUMINAMATH_GPT_least_value_of_a_plus_b_l456_45685

def a_and_b (a b : ℕ) : Prop :=
  (Nat.gcd (a + b) 330 = 1) ∧ 
  (a^a % b^b = 0) ∧ 
  (¬ (a % b = 0))

theorem least_value_of_a_plus_b :
  ∃ (a b : ℕ), a_and_b a b ∧ a + b = 105 :=
sorry

end NUMINAMATH_GPT_least_value_of_a_plus_b_l456_45685


namespace NUMINAMATH_GPT_pelican_fish_count_l456_45638

theorem pelican_fish_count 
(P K F : ℕ) 
(h1: K = P + 7) 
(h2: F = 3 * (P + K)) 
(h3: F = P + 86) : P = 13 := 
by 
  sorry

end NUMINAMATH_GPT_pelican_fish_count_l456_45638


namespace NUMINAMATH_GPT_distance_walked_north_l456_45676

-- Definition of the problem parameters
def distance_west : ℝ := 10
def total_distance : ℝ := 14.142135623730951

-- The theorem stating the result
theorem distance_walked_north (x : ℝ) (h : distance_west ^ 2 + x ^ 2 = total_distance ^ 2) : x = 10 :=
by sorry

end NUMINAMATH_GPT_distance_walked_north_l456_45676


namespace NUMINAMATH_GPT_bill_milk_problem_l456_45650

theorem bill_milk_problem 
  (M : ℚ) 
  (sour_cream_milk : ℚ := M / 4)
  (butter_milk : ℚ := M / 4)
  (whole_milk : ℚ := M / 2)
  (sour_cream_gallons : ℚ := sour_cream_milk / 2)
  (butter_gallons : ℚ := butter_milk / 4)
  (butter_revenue : ℚ := butter_gallons * 5)
  (sour_cream_revenue : ℚ := sour_cream_gallons * 6)
  (whole_milk_revenue : ℚ := whole_milk * 3)
  (total_revenue : ℚ := butter_revenue + sour_cream_revenue + whole_milk_revenue)
  (h : total_revenue = 41) :
  M = 16 :=
by
  sorry

end NUMINAMATH_GPT_bill_milk_problem_l456_45650


namespace NUMINAMATH_GPT_triangle_count_l456_45627

theorem triangle_count (a b c : ℕ) (hb : b = 2008) (hab : a ≤ b) (hbc : b ≤ c) (ht : a + b > c) : 
  ∃ n, n = 2017036 :=
by
  sorry

end NUMINAMATH_GPT_triangle_count_l456_45627


namespace NUMINAMATH_GPT_sandy_earnings_correct_l456_45630

def hourly_rate : ℕ := 15
def hours_worked_friday : ℕ := 10
def hours_worked_saturday : ℕ := 6
def hours_worked_sunday : ℕ := 14

def earnings_friday : ℕ := hours_worked_friday * hourly_rate
def earnings_saturday : ℕ := hours_worked_saturday * hourly_rate
def earnings_sunday : ℕ := hours_worked_sunday * hourly_rate

def total_earnings : ℕ := earnings_friday + earnings_saturday + earnings_sunday

theorem sandy_earnings_correct : total_earnings = 450 := by
  sorry

end NUMINAMATH_GPT_sandy_earnings_correct_l456_45630


namespace NUMINAMATH_GPT_first_day_is_sunday_l456_45615

noncomputable def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem first_day_is_sunday :
  (day_of_week 18 = "Wednesday") → (day_of_week 1 = "Sunday") :=
by
  intro h
  -- proof would go here
  sorry

end NUMINAMATH_GPT_first_day_is_sunday_l456_45615


namespace NUMINAMATH_GPT_find_a_plus_c_l456_45688

noncomputable def triangle_ABC (A B C a b c : ℝ) : Prop :=
  (b * Real.cos C + c * Real.cos B = 3 * a * Real.cos B) ∧
  (b = 2) ∧
  ((1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 2) / 2)

theorem find_a_plus_c {A B C a b c : ℝ} (h : triangle_ABC A B C a b c) :
  a + c = 4 :=
by
  rcases h with ⟨hc1, hc2, hc3⟩
  sorry

end NUMINAMATH_GPT_find_a_plus_c_l456_45688


namespace NUMINAMATH_GPT_intersection_M_N_l456_45675

def M (x : ℝ) : Prop := (2 - x) / (x + 1) ≥ 0
def N (y : ℝ) : Prop := ∃ x : ℝ, y = Real.log x

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {y : ℝ | N y} = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l456_45675


namespace NUMINAMATH_GPT_teams_in_BIG_M_l456_45657

theorem teams_in_BIG_M (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
sorry

end NUMINAMATH_GPT_teams_in_BIG_M_l456_45657


namespace NUMINAMATH_GPT_water_consumed_l456_45680

theorem water_consumed (traveler_water : ℕ) (camel_multiplier : ℕ) (ounces_in_gallon : ℕ) (total_water : ℕ)
  (h_traveler : traveler_water = 32)
  (h_camel : camel_multiplier = 7)
  (h_ounces_in_gallon : ounces_in_gallon = 128)
  (h_total : total_water = traveler_water + camel_multiplier * traveler_water) :
  total_water / ounces_in_gallon = 2 :=
by
  sorry

end NUMINAMATH_GPT_water_consumed_l456_45680


namespace NUMINAMATH_GPT_prime_sequence_constant_l456_45632

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Condition: There exists a constant sequence of primes such that the given recurrence relation holds.
theorem prime_sequence_constant (p : ℕ) (k : ℤ) (n : ℕ) 
  (h1 : 1 ≤ n)
  (h2 : ∀ m ≥ 1, is_prime (p + m))
  (h3 : p + k = p + p + k) :
  ∀ m ≥ 1, p + m = p :=
sorry

end NUMINAMATH_GPT_prime_sequence_constant_l456_45632


namespace NUMINAMATH_GPT_number_writing_number_reading_l456_45640

def ten_million_place := 10^7
def hundred_thousand_place := 10^5
def ten_place := 10

def ten_million := 1 * ten_million_place
def three_hundred_thousand := 3 * hundred_thousand_place
def fifty := 5 * ten_place

def constructed_number := ten_million + three_hundred_thousand + fifty

def read_number := "ten million and thirty thousand and fifty"

theorem number_writing : constructed_number = 10300050 := by
  -- Sketch of proof goes here based on place values
  sorry

theorem number_reading : read_number = "ten million and thirty thousand and fifty" := by
  -- Sketch of proof goes here for the reading method
  sorry

end NUMINAMATH_GPT_number_writing_number_reading_l456_45640


namespace NUMINAMATH_GPT_jason_less_than_jenny_l456_45624

-- Definition of conditions

def grade_Jenny : ℕ := 95
def grade_Bob : ℕ := 35
def grade_Jason : ℕ := 2 * grade_Bob -- Bob's grade is half of Jason's grade

-- The theorem we need to prove
theorem jason_less_than_jenny : grade_Jenny - grade_Jason = 25 :=
by
  sorry

end NUMINAMATH_GPT_jason_less_than_jenny_l456_45624


namespace NUMINAMATH_GPT_ratio_of_female_democrats_l456_45651

theorem ratio_of_female_democrats 
    (M F : ℕ) 
    (H1 : M + F = 990)
    (H2 : M / 4 + 165 = 330) 
    (H3 : 165 = 165) : 
    165 / F = 1 / 2 := 
sorry

end NUMINAMATH_GPT_ratio_of_female_democrats_l456_45651


namespace NUMINAMATH_GPT_smallest_number_of_marbles_l456_45653

theorem smallest_number_of_marbles :
  ∃ N : ℕ, N > 1 ∧ (N % 9 = 1) ∧ (N % 10 = 1) ∧ (N % 11 = 1) ∧ (∀ m : ℕ, m > 1 ∧ (m % 9 = 1) ∧ (m % 10 = 1) ∧ (m % 11 = 1) → N ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_number_of_marbles_l456_45653


namespace NUMINAMATH_GPT_h_h3_eq_3568_l456_45674

def h (x : ℤ) := 3 * x ^ 2 + 3 * x - 2

theorem h_h3_eq_3568 : h (h 3) = 3568 := by
  sorry

end NUMINAMATH_GPT_h_h3_eq_3568_l456_45674


namespace NUMINAMATH_GPT_find_angle_C_l456_45683

variable (A B C : ℝ)
variable (a b c : ℝ)

theorem find_angle_C (hA : A = 39) 
                     (h_condition : (a^2 - b^2)*(a^2 + a*c - b^2) = b^2 * c^2) : 
                     C = 115 :=
sorry

end NUMINAMATH_GPT_find_angle_C_l456_45683


namespace NUMINAMATH_GPT_percentage_owning_cats_percentage_owning_birds_l456_45660

def total_students : ℕ := 500
def students_owning_cats : ℕ := 80
def students_owning_birds : ℕ := 120

theorem percentage_owning_cats : students_owning_cats * 100 / total_students = 16 := 
by 
  sorry

theorem percentage_owning_birds : students_owning_birds * 100 / total_students = 24 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_owning_cats_percentage_owning_birds_l456_45660


namespace NUMINAMATH_GPT_find_denomination_l456_45617

def denomination_of_bills (num_tumblers : ℕ) (cost_per_tumbler change num_bills amount_paid bill_denomination : ℤ) : Prop :=
  num_tumblers * cost_per_tumbler + change = amount_paid ∧
  amount_paid = num_bills * bill_denomination

theorem find_denomination :
  denomination_of_bills
    10    -- num_tumblers
    45    -- cost_per_tumbler
    50    -- change
    5     -- num_bills
    500   -- amount_paid
    100   -- bill_denomination
:=
by
  sorry

end NUMINAMATH_GPT_find_denomination_l456_45617


namespace NUMINAMATH_GPT_longest_train_length_l456_45603

theorem longest_train_length :
  ∀ (speedA : ℝ) (timeA : ℝ) (speedB : ℝ) (timeB : ℝ) (speedC : ℝ) (timeC : ℝ),
  speedA = 60 * (5 / 18) → timeA = 5 →
  speedB = 80 * (5 / 18) → timeB = 7 →
  speedC = 50 * (5 / 18) → timeC = 9 →
  speedB * timeB > speedA * timeA ∧ speedB * timeB > speedC * timeC ∧ speedB * timeB = 155.54 := by
  sorry

end NUMINAMATH_GPT_longest_train_length_l456_45603


namespace NUMINAMATH_GPT_sally_garden_area_l456_45600

theorem sally_garden_area :
  ∃ (a b : ℕ), 2 * (a + b) = 24 ∧ b + 1 = 3 * (a + 1) ∧ 
     (3 * (a - 1) * 3 * (b - 1) = 297) :=
by {
  sorry
}

end NUMINAMATH_GPT_sally_garden_area_l456_45600


namespace NUMINAMATH_GPT_polynomial_remainder_l456_45610

def f (r : ℝ) : ℝ := r^15 - r + 3

theorem polynomial_remainder :
  f 2 = 32769 := by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l456_45610


namespace NUMINAMATH_GPT_unique_prime_n_l456_45652

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_n (n : ℕ)
  (h1 : isPrime n)
  (h2 : isPrime (n^2 + 10))
  (h3 : isPrime (n^2 - 2))
  (h4 : isPrime (n^3 + 6))
  (h5 : isPrime (n^5 + 36)) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_unique_prime_n_l456_45652


namespace NUMINAMATH_GPT_probability_X_equals_Y_l456_45614

noncomputable def prob_X_equals_Y : ℚ :=
  let count_intersections : ℚ := 15
  let total_possibilities : ℚ := 15 * 15
  count_intersections / total_possibilities

theorem probability_X_equals_Y :
  (∀ (x y : ℝ), -15 * Real.pi ≤ x ∧ x ≤ 15 * Real.pi ∧ -15 * Real.pi ≤ y ∧ y ≤ 15 * Real.pi →
    (Real.cos (Real.cos x) = Real.cos (Real.cos y)) →
    prob_X_equals_Y = 1/15) :=
sorry

end NUMINAMATH_GPT_probability_X_equals_Y_l456_45614


namespace NUMINAMATH_GPT_jessica_withdraw_fraq_l456_45612

theorem jessica_withdraw_fraq {B : ℝ} (h : B - 200 + (1 / 2) * (B - 200) = 450) :
  (200 / B) = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_jessica_withdraw_fraq_l456_45612


namespace NUMINAMATH_GPT_largest_difference_rounding_l456_45661

variable (A B : ℝ)
variable (estimate_A estimate_B : ℝ)
variable (within_A within_B : ℝ)
variable (diff : ℝ)

axiom est_A : estimate_A = 55000
axiom est_B : estimate_B = 58000
axiom cond_A : within_A = 0.15
axiom cond_B : within_B = 0.10

axiom bounds_A : 46750 ≤ A ∧ A ≤ 63250
axiom bounds_B : 52727 ≤ B ∧ B ≤ 64444

noncomputable def max_possible_difference : ℝ :=
  max (abs (B - A)) (abs (A - B))

theorem largest_difference_rounding :
  max_possible_difference A B = 18000 :=
by
  sorry

end NUMINAMATH_GPT_largest_difference_rounding_l456_45661


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l456_45663

theorem parabola_focus_coordinates : 
  ∀ (x y : ℝ), x = 4 * y^2 → (∃ (y₀ : ℝ), (x, y₀) = (1/16, 0)) :=
by
  intro x y hxy
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l456_45663


namespace NUMINAMATH_GPT_remainder_div_19_l456_45636

theorem remainder_div_19 (N : ℤ) (k : ℤ) (h : N = 779 * k + 47) : N % 19 = 9 :=
sorry

end NUMINAMATH_GPT_remainder_div_19_l456_45636


namespace NUMINAMATH_GPT_eval_expr_correct_l456_45607

noncomputable def eval_expr : ℝ :=
  let a := (12:ℝ)^5 * (6:ℝ)^4
  let b := (3:ℝ)^2 * (36:ℝ)^2
  let c := Real.sqrt 9 * Real.log (27:ℝ)
  (a / b) + c

theorem eval_expr_correct : eval_expr = 27657.887510597983 := by
  sorry

end NUMINAMATH_GPT_eval_expr_correct_l456_45607


namespace NUMINAMATH_GPT_Vasya_and_Petya_no_mistake_exists_l456_45687

def is_prime (n : ℕ) : Prop := ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem Vasya_and_Petya_no_mistake_exists :
  ∃ x : ℝ, (∃ p : ℕ, is_prime p ∧ 10 * x = p) ∧ 
           (∃ q : ℕ, is_prime q ∧ 15 * x = q) :=
sorry

end NUMINAMATH_GPT_Vasya_and_Petya_no_mistake_exists_l456_45687


namespace NUMINAMATH_GPT_ellipse_parabola_common_point_l456_45670

theorem ellipse_parabola_common_point (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔  -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_parabola_common_point_l456_45670


namespace NUMINAMATH_GPT_intersection_conditions_l456_45620

-- Define the conditions
variables (c : ℝ) (k : ℝ) (m : ℝ) (n : ℝ) (p : ℝ)

-- Distance condition
def distance_condition (k : ℝ) (m : ℝ) (n : ℝ) (c : ℝ) : Prop :=
  (abs ((k^2 + 8 * k + c) - (m * k + n)) = 4)

-- Line passing through point (2, 7)
def passes_through_point (m : ℝ) (n : ℝ) : Prop :=
  (7 = 2 * m + n)

-- Definition of discriminants
def discriminant_1 (m : ℝ) (c : ℝ) (n : ℝ) : ℝ :=
  ((8 - m)^2 - 4 * (c - n - 4))

def discriminant_2 (m : ℝ) (c : ℝ) (n : ℝ) : ℝ :=
  ((8 - m)^2 - 4 * (c - n + 4))

-- Statement of the problem
theorem intersection_conditions (h₁ : n ≠ 0)
  (h₂ : passes_through_point m n)
  (h₃ : distance_condition k m n c)
  (h₄ : (discriminant_1 m c n = 0 ∨ discriminant_1 m c n < 0))
  (h₅ : (discriminant_2 m c n < 0)) :
  ∃ m n, n = 7 - 2 * m ∧ distance_condition k m n c :=
sorry

end NUMINAMATH_GPT_intersection_conditions_l456_45620


namespace NUMINAMATH_GPT_texas_california_plate_diff_l456_45693

def california_plates := 26^3 * 10^3
def texas_plates := 26^3 * 10^4
def plates_difference := texas_plates - california_plates

theorem texas_california_plate_diff :
  plates_difference = 158184000 :=
by sorry

end NUMINAMATH_GPT_texas_california_plate_diff_l456_45693


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l456_45602

/-- Given a hyperbola with the equation x^2/a^2 - y^2/b^2 = 1, point B(0, b),
the line F1B intersects with the two asymptotes at points P and Q. 
We are given that vector QP = 4 * vector PF1. Prove that the eccentricity 
of the hyperbola is 3/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (F1 : ℝ × ℝ) (B : ℝ × ℝ) (P Q : ℝ × ℝ) 
  (h_F1 : F1 = (-c, 0)) (h_B : B = (0, b)) 
  (h_int_P : P = (-a * c / (c + a), b * c / (c + a)))
  (h_int_Q : Q = (a * c / (c - a), b * c / (c - a)))
  (h_vec : (Q.1 - P.1, Q.2 - P.2) = (4 * (P.1 - F1.1), 4 * (P.2 - F1.2))) :
  (eccentricity : ℝ) = 3 / 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l456_45602


namespace NUMINAMATH_GPT_calc_expression_l456_45621

theorem calc_expression :
  let a := 3^456
  let b := 9^5 / 9^3
  a - b = 3^456 - 81 :=
by
  let a := 3^456
  let b := 9^5 / 9^3
  sorry

end NUMINAMATH_GPT_calc_expression_l456_45621


namespace NUMINAMATH_GPT_total_soccer_balls_l456_45667

theorem total_soccer_balls (boxes : ℕ) (packages_per_box : ℕ) (balls_per_package : ℕ) 
  (h1 : boxes = 10) (h2 : packages_per_box = 8) (h3 : balls_per_package = 13) : 
  (boxes * packages_per_box * balls_per_package = 1040) :=
by 
  sorry

end NUMINAMATH_GPT_total_soccer_balls_l456_45667


namespace NUMINAMATH_GPT_total_charge_for_3_6_miles_during_peak_hours_l456_45665

-- Define the initial conditions as constants
def initial_fee : ℝ := 2.05
def charge_per_half_mile_first_2_miles : ℝ := 0.45
def charge_per_two_fifth_mile_after_2_miles : ℝ := 0.35
def peak_hour_surcharge : ℝ := 1.50

-- Define the function to calculate the total charge
noncomputable def total_charge (total_distance : ℝ) (is_peak_hour : Bool) : ℝ :=
  let first_2_miles_charge := if total_distance > 2 then 4 * charge_per_half_mile_first_2_miles else (total_distance / 0.5) * charge_per_half_mile_first_2_miles
  let remaining_distance := if total_distance > 2 then total_distance - 2 else 0
  let after_2_miles_charge := if total_distance > 2 then (remaining_distance / (2 / 5)) * charge_per_two_fifth_mile_after_2_miles else 0
  let surcharge := if is_peak_hour then peak_hour_surcharge else 0
  initial_fee + first_2_miles_charge + after_2_miles_charge + surcharge

-- Prove that total charge of 3.6 miles during peak hours is 6.75
theorem total_charge_for_3_6_miles_during_peak_hours : total_charge 3.6 true = 6.75 := by
  sorry

end NUMINAMATH_GPT_total_charge_for_3_6_miles_during_peak_hours_l456_45665


namespace NUMINAMATH_GPT_ratio_of_first_to_second_ball_l456_45645

theorem ratio_of_first_to_second_ball 
  (x y z : ℕ) 
  (h1 : 3 * x = 27) 
  (h2 : y = 18) 
  (h3 : z = 3 * x) : 
  x / y = 1 / 2 := 
sorry

end NUMINAMATH_GPT_ratio_of_first_to_second_ball_l456_45645


namespace NUMINAMATH_GPT_necklace_stand_capacity_l456_45637

def necklace_stand_initial := 5
def ring_display_capacity := 30
def ring_display_current := 18
def bracelet_display_capacity := 15
def bracelet_display_current := 8
def cost_per_necklace := 4
def cost_per_ring := 10
def cost_per_bracelet := 5
def total_cost := 183

theorem necklace_stand_capacity : necklace_stand_current + (total_cost - (ring_display_capacity - ring_display_current) * cost_per_ring - (bracelet_display_capacity - bracelet_display_current) * cost_per_bracelet) / cost_per_necklace = 12 :=
by
  sorry

end NUMINAMATH_GPT_necklace_stand_capacity_l456_45637


namespace NUMINAMATH_GPT_determinant_inequality_solution_l456_45672

theorem determinant_inequality_solution (a : ℝ) :
  (∀ x : ℝ, (x > -1 → x < (4 / a))) ↔ a = -4 := by
sorry

end NUMINAMATH_GPT_determinant_inequality_solution_l456_45672


namespace NUMINAMATH_GPT_fastest_route_time_l456_45686

theorem fastest_route_time (d1 d2 : ℕ) (s1 s2 : ℕ) (h1 : d1 = 1500) (h2 : d2 = 750) (h3 : s1 = 75) (h4 : s2 = 25) :
  min (d1 / s1) (d2 / s2) = 20 := by
  sorry

end NUMINAMATH_GPT_fastest_route_time_l456_45686


namespace NUMINAMATH_GPT_div_n_by_8_eq_2_8089_l456_45694

theorem div_n_by_8_eq_2_8089
  (n : ℕ)
  (h : n = 16^2023) :
  n / 8 = 2^8089 := by
  sorry

end NUMINAMATH_GPT_div_n_by_8_eq_2_8089_l456_45694


namespace NUMINAMATH_GPT_total_students_in_class_is_15_l456_45629

noncomputable def choose (n k : ℕ) : ℕ := sorry -- Define a function for combinations
noncomputable def permute (n k : ℕ) : ℕ := sorry -- Define a function for permutations

variables (x m n : ℕ) (hx : choose x 4 = m) (hn : permute x 2 = n) (hratio : m * 2 = n * 13)

theorem total_students_in_class_is_15 : x = 15 :=
sorry

end NUMINAMATH_GPT_total_students_in_class_is_15_l456_45629


namespace NUMINAMATH_GPT_not_divisible_1978_1000_l456_45691

theorem not_divisible_1978_1000 (m : ℕ) : ¬ ∃ m : ℕ, (1000^m - 1) ∣ (1978^m - 1) := sorry

end NUMINAMATH_GPT_not_divisible_1978_1000_l456_45691


namespace NUMINAMATH_GPT_snakes_in_breeding_ball_l456_45684

theorem snakes_in_breeding_ball (x : ℕ) (h : 3 * x + 12 = 36) : x = 8 :=
by sorry

end NUMINAMATH_GPT_snakes_in_breeding_ball_l456_45684


namespace NUMINAMATH_GPT_notebook_width_l456_45664

theorem notebook_width
  (circumference : ℕ)
  (length : ℕ)
  (width : ℕ)
  (H1 : circumference = 46)
  (H2 : length = 9)
  (H3 : circumference = 2 * (length + width)) :
  width = 14 :=
by
  sorry -- proof is omitted

end NUMINAMATH_GPT_notebook_width_l456_45664


namespace NUMINAMATH_GPT_angle_sum_of_octagon_and_triangle_l456_45689

-- Define the problem setup
def is_interior_angle_of_regular_polygon (n : ℕ) (angle : ℝ) : Prop :=
  angle = 180 * (n - 2) / n

def is_regular_octagon_angle (angle : ℝ) : Prop :=
  is_interior_angle_of_regular_polygon 8 angle

def is_equilateral_triangle_angle (angle : ℝ) : Prop :=
  is_interior_angle_of_regular_polygon 3 angle

-- The statement of the problem
theorem angle_sum_of_octagon_and_triangle :
  ∃ angle_ABC angle_ABD : ℝ,
    is_regular_octagon_angle angle_ABC ∧
    is_equilateral_triangle_angle angle_ABD ∧
    angle_ABC + angle_ABD = 195 :=
sorry

end NUMINAMATH_GPT_angle_sum_of_octagon_and_triangle_l456_45689


namespace NUMINAMATH_GPT_min_visible_sum_of_4x4x4_cube_l456_45644

theorem min_visible_sum_of_4x4x4_cube (dice_capacity : ℕ) (opposite_sum : ℕ) (corner_dice edge_dice center_face_dice innermost_dice : ℕ) : 
  dice_capacity = 64 ∧ 
  opposite_sum = 7 ∧ 
  corner_dice = 8 ∧ 
  edge_dice = 24 ∧ 
  center_face_dice = 24 ∧ 
  innermost_dice = 8 → 
  ∃ min_sum, min_sum = 144 := by
  sorry

end NUMINAMATH_GPT_min_visible_sum_of_4x4x4_cube_l456_45644


namespace NUMINAMATH_GPT_alcohol_to_water_ratio_l456_45605

variable {V p q : ℚ}

def alcohol_volume_jar1 (V p : ℚ) : ℚ := (2 * p) / (2 * p + 3) * V
def water_volume_jar1 (V p : ℚ) : ℚ := 3 / (2 * p + 3) * V
def alcohol_volume_jar2 (V q : ℚ) : ℚ := q / (q + 2) * 2 * V
def water_volume_jar2 (V q : ℚ) : ℚ := 2 / (q + 2) * 2 * V

def total_alcohol_volume (V p q : ℚ) : ℚ :=
  alcohol_volume_jar1 V p + alcohol_volume_jar2 V q

def total_water_volume (V p q : ℚ) : ℚ :=
  water_volume_jar1 V p + water_volume_jar2 V q

theorem alcohol_to_water_ratio (V p q : ℚ) :
  (total_alcohol_volume V p q) / (total_water_volume V p q) = (2 * p + 2 * q) / (3 * p + q + 10) :=
by
  sorry

end NUMINAMATH_GPT_alcohol_to_water_ratio_l456_45605


namespace NUMINAMATH_GPT_larger_number_l456_45692

theorem larger_number (A B : ℝ) (h1 : A - B = 1650) (h2 : 0.075 * A = 0.125 * B) : A = 4125 :=
sorry

end NUMINAMATH_GPT_larger_number_l456_45692


namespace NUMINAMATH_GPT_sqrt_nine_eq_three_l456_45695

theorem sqrt_nine_eq_three : Real.sqrt 9 = 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_nine_eq_three_l456_45695


namespace NUMINAMATH_GPT_probability_of_B_l456_45642

variables (A B : Prop)
variables (P : Prop → ℝ) -- Probability Measure

axiom A_and_B : P (A ∧ B) = 0.15
axiom not_A_and_not_B : P (¬A ∧ ¬B) = 0.6

theorem probability_of_B : P B = 0.15 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_B_l456_45642


namespace NUMINAMATH_GPT_area_T_l456_45681

variable (T : Set (ℝ × ℝ)) -- T is a region in the plane
variable (A : Matrix (Fin 2) (Fin 2) ℝ) -- A is a 2x2 matrix
variable (detA : ℝ) -- detA is the determinant of A

-- assumptions
axiom area_T : ∃ (area : ℝ), area = 9
axiom matrix_A : A = ![![3, 2], ![-1, 4]]
axiom determinant_A : detA = 14

-- statement to prove
theorem area_T' : ∃ area_T' : ℝ, area_T' = 126 :=
sorry

end NUMINAMATH_GPT_area_T_l456_45681


namespace NUMINAMATH_GPT_find_common_difference_l456_45649

variable {a : ℕ → ℤ} 
variable {S : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def problem_conditions (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) : Prop :=
  a 3 + a 4 = 8 ∧ S 8 = 48

theorem find_common_difference :
  ∃ d, problem_conditions a S d ∧ is_arithmetic_sequence a d ∧ sum_of_first_n_terms a S ∧ d = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_common_difference_l456_45649


namespace NUMINAMATH_GPT_range_of_a_l456_45655

-- Definitions derived from conditions
def is_ellipse_with_foci_on_x_axis (a : ℝ) : Prop := a^2 > a + 6 ∧ a + 6 > 0

-- Theorem representing the proof problem
theorem range_of_a (a : ℝ) (h : is_ellipse_with_foci_on_x_axis a) :
  (a > 3) ∨ (-6 < a ∧ a < -2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l456_45655


namespace NUMINAMATH_GPT_julio_lost_15_fish_l456_45697

def fish_caught_per_hour : ℕ := 7
def hours_fished : ℕ := 9
def fish_total_without_loss : ℕ := fish_caught_per_hour * hours_fished
def fish_total_actual : ℕ := 48
def fish_lost : ℕ := fish_total_without_loss - fish_total_actual

theorem julio_lost_15_fish : fish_lost = 15 := by
  sorry

end NUMINAMATH_GPT_julio_lost_15_fish_l456_45697


namespace NUMINAMATH_GPT_circumscribed_triangle_area_relation_l456_45699

theorem circumscribed_triangle_area_relation
    (a b c: ℝ) (h₀: a = 8) (h₁: b = 15) (h₂: c = 17)
    (triangle_area: ℝ) (circle_area: ℝ) (X Y Z: ℝ)
    (hZ: Z > X) (hXY: X < Y)
    (triangle_area_calc: triangle_area = 60)
    (circle_area_calc: circle_area = π * (c / 2)^2) :
    X + Y = Z := by
  sorry

end NUMINAMATH_GPT_circumscribed_triangle_area_relation_l456_45699


namespace NUMINAMATH_GPT_find_a3_l456_45604

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_a3
  (a : ℕ → α) (q : α)
  (h_geom : geometric_sequence a q)
  (h_a1 : a 1 = 2)
  (h_cond : a 3 * a 5 = 4 * (a 6) ^ 2) :
  a 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a3_l456_45604


namespace NUMINAMATH_GPT_problem_1_problem_2_l456_45609

-- First problem: Find the solution set for the inequality |x - 1| + |x + 2| ≥ 5
theorem problem_1 (x : ℝ) : (|x - 1| + |x + 2| ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) :=
sorry

-- Second problem: Find the range of real number a such that |x - a| + |x + 2| ≤ |x + 4| for all x in [0, 1]
theorem problem_2 (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x - a| + |x + 2| ≤ |x + 4|) ↔ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l456_45609


namespace NUMINAMATH_GPT_find_subtracted_value_l456_45662

theorem find_subtracted_value (n x : ℕ) (h₁ : n = 36) (h₂ : ((n + 10) * 2 / 2 - x) = 44) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_subtracted_value_l456_45662


namespace NUMINAMATH_GPT_money_problem_l456_45628

variable (a b : ℝ)

theorem money_problem (h1 : 4 * a + b = 68) 
                      (h2 : 2 * a - b < 16) 
                      (h3 : a + b > 22) : 
                      a < 14 ∧ b > 12 := 
by 
  sorry

end NUMINAMATH_GPT_money_problem_l456_45628


namespace NUMINAMATH_GPT_simplify_sin_formula_l456_45622

theorem simplify_sin_formula : 2 * Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 2 := by
  -- Conditions and values used in the proof
  sorry

end NUMINAMATH_GPT_simplify_sin_formula_l456_45622


namespace NUMINAMATH_GPT_digit_at_1286th_position_l456_45654

def naturally_written_sequence : ℕ → ℕ := sorry

theorem digit_at_1286th_position : naturally_written_sequence 1286 = 3 :=
sorry

end NUMINAMATH_GPT_digit_at_1286th_position_l456_45654


namespace NUMINAMATH_GPT_vectors_not_coplanar_l456_45678

def a : ℝ × ℝ × ℝ := (4, 1, 1)
def b : ℝ × ℝ × ℝ := (-9, -4, -9)
def c : ℝ × ℝ × ℝ := (6, 2, 6)

def scalarTripleProduct (u v w : ℝ × ℝ × ℝ) : ℝ :=
  let (u1, u2, u3) := u
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  u1 * (v2 * w3 - v3 * w2) - u2 * (v1 * w3 - v3 * w1) + u3 * (v1 * w2 - v2 * w1)

theorem vectors_not_coplanar : scalarTripleProduct a b c = -18 := by
  sorry

end NUMINAMATH_GPT_vectors_not_coplanar_l456_45678


namespace NUMINAMATH_GPT_businessman_earnings_l456_45606

theorem businessman_earnings : 
  let P : ℝ := 1000
  let day1_stock := 1000 / P
  let day2_stock := 1000 / (P * 1.1)
  let day3_stock := 1000 / (P * 1.1^2)
  let value_on_day4 stock := stock * (P * 1.1^3)
  let total_earnings := value_on_day4 day1_stock + value_on_day4 day2_stock + value_on_day4 day3_stock
  total_earnings = 3641 := sorry

end NUMINAMATH_GPT_businessman_earnings_l456_45606


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l456_45625

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a ≠ 0) → (ab ≠ 0) ↔ (a ≠ 0) :=
by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l456_45625


namespace NUMINAMATH_GPT_solve_system_of_equations_l456_45666

theorem solve_system_of_equations (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : 1 / (x * y) = x / z + 1)
  (h2 : 1 / (y * z) = y / x + 1)
  (h3 : 1 / (z * x) = z / y + 1) :
  x = 1 / Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 ∧ z = 1 / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l456_45666
