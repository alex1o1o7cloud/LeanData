import Mathlib

namespace symmetric_point_coordinates_l1628_162830

theorem symmetric_point_coordinates (a b : ℝ) (hp : (3, 4) = (a + 3, b + 4)) :
  (a, b) = (5, 2) :=
  sorry

end symmetric_point_coordinates_l1628_162830


namespace xyz_value_l1628_162812

theorem xyz_value (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 21) :
  x * y * z = 28 / 3 :=
by
  sorry

end xyz_value_l1628_162812


namespace least_possible_value_of_z_minus_w_l1628_162871

variable (x y z w k m : Int)
variable (h1 : Even x)
variable (h2 : Odd y)
variable (h3 : Odd z)
variable (h4 : ∃ n : Int, w = - (2 * n + 1) / 3)
variable (h5 : w < x)
variable (h6 : x < y)
variable (h7 : y < z)
variable (h8 : 0 < k)
variable (h9 : (y - x) > k)
variable (h10 : 0 < m)
variable (h11 : (z - w) > m)
variable (h12 : k > m)

theorem least_possible_value_of_z_minus_w
  : z - w = 6 := sorry

end least_possible_value_of_z_minus_w_l1628_162871


namespace fraction_value_l1628_162868

theorem fraction_value : (1 - 1 / 4) / (1 - 1 / 3) = 9 / 8 := 
by sorry

end fraction_value_l1628_162868


namespace correct_equation_l1628_162841

theorem correct_equation (x : ℝ) (hx : x > 80) : 
  353 / (x - 80) - 353 / x = 5 / 3 :=
sorry

end correct_equation_l1628_162841


namespace ferris_wheel_time_l1628_162821

theorem ferris_wheel_time (R T : ℝ) (t : ℝ) (h : ℝ → ℝ) :
  R = 30 → T = 90 → (∀ t, h t = R * Real.cos ((2 * Real.pi / T) * t) + R) → h t = 45 → t = 15 :=
by
  intros hR hT hFunc hHt
  sorry

end ferris_wheel_time_l1628_162821


namespace value_of_f1_plus_g3_l1628_162811

def f (x : ℝ) := 3 * x - 4
def g (x : ℝ) := x + 2

theorem value_of_f1_plus_g3 : f (1 + g 3) = 14 := by
  sorry

end value_of_f1_plus_g3_l1628_162811


namespace ellipse_slope_ratio_l1628_162853

theorem ellipse_slope_ratio (a b : ℝ) (k1 k2 : ℝ) 
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a > 2)
  (h4 : k2 = k1 * (a^2 + 5) / (a^2 - 1)) : 
  1 < (k2 / k1) ∧ (k2 / k1) < 3 :=
by
  sorry

end ellipse_slope_ratio_l1628_162853


namespace Johnson_family_seating_l1628_162814

theorem Johnson_family_seating : 
  ∃ n : ℕ, number_of_ways_to_seat_Johnson_family = n ∧ n = 288 :=
sorry

end Johnson_family_seating_l1628_162814


namespace new_volume_proof_l1628_162819

variable (r h : ℝ)
variable (π : ℝ := Real.pi) -- Lean's notation for π
variable (original_volume : ℝ := 15) -- given original volume

-- Define original volume of the cylinder
def V := π * r^2 * h

-- Define new volume of the cylinder using new dimensions
def new_V := π * (3 * r)^2 * (2 * h)

-- Prove that new_V is 270 when V = 15
theorem new_volume_proof (hV : V = 15) : new_V = 270 :=
by
  -- Proof will go here
  sorry

end new_volume_proof_l1628_162819


namespace initial_percentage_increase_l1628_162889

theorem initial_percentage_increase 
  (W R : ℝ) 
  (P : ℝ)
  (h1 : R = W * (1 + P/100)) 
  (h2 : R * 0.70 = W * 1.18999999999999993) :
  P = 70 :=
by sorry

end initial_percentage_increase_l1628_162889


namespace quadratic_inequality_solution_set_l1628_162899

theorem quadratic_inequality_solution_set
  (a b : ℝ)
  (h1 : 2 + 3 = -a)
  (h2 : 2 * 3 = b) :
  ∀ x : ℝ, 6 * x^2 - 5 * x + 1 > 0 ↔ x < (1 / 3) ∨ x > (1 / 2) := by
  sorry

end quadratic_inequality_solution_set_l1628_162899


namespace arithmetic_sequence_fifth_term_l1628_162877

noncomputable def fifth_term_of_arithmetic_sequence (x y : ℝ)
  (h1 : 2 * x + y = 2 * x + y)
  (h2 : 2 * x - y = 2 * x + y - 2 * y)
  (h3 : 2 * x * y = 2 * x - 2 * y - 2 * y)
  (h4 : 2 * x / y = 2 * x * y - 5 * y^2 - 2 * y)
  : ℝ :=
(2 * x / y) - 2 * y

theorem arithmetic_sequence_fifth_term (x y : ℝ)
  (h1 : 2 * x + y = 2 * x + y)
  (h2 : 2 * x - y = 2 * x + y - 2 * y)
  (h3 : 2 * x * y = 2 * x - 2 * y - 2 * y)
  (h4 : 2 * x / y = 2 * x * y - 5 * y^2 - 2 * y)
  : fifth_term_of_arithmetic_sequence x y h1 h2 h3 h4 = -77 / 10 :=
sorry

end arithmetic_sequence_fifth_term_l1628_162877


namespace decreasing_function_range_l1628_162892

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else -a * x

theorem decreasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (1 / 8 ≤ a ∧ a < 1 / 3) :=
by
  sorry

end decreasing_function_range_l1628_162892


namespace problem_l1628_162849

open Set

theorem problem (M : Set ℤ) (N : Set ℤ) (hM : M = {1, 2, 3, 4}) (hN : N = {-2, 2}) : 
  M ∩ N = {2} :=
by
  sorry

end problem_l1628_162849


namespace chuck_team_leads_by_2_l1628_162816

open Nat

noncomputable def chuck_team_score_first_quarter := 9 * 2 + 5 * 1
noncomputable def yellow_team_score_first_quarter := 7 * 2 + 4 * 3

noncomputable def chuck_team_score_second_quarter := 6 * 2 + 3 * 3
noncomputable def yellow_team_score_second_quarter := 5 * 2 + 2 * 3 + 3 * 1

noncomputable def chuck_team_score_third_quarter := 4 * 2 + 2 * 3 + 6 * 1
noncomputable def yellow_team_score_third_quarter := 6 * 2 + 2 * 3

noncomputable def chuck_team_score_fourth_quarter := 8 * 2 + 1 * 3
noncomputable def yellow_team_score_fourth_quarter := 4 * 2 + 3 * 3 + 2 * 1

noncomputable def chuck_team_technical_fouls := 3
noncomputable def yellow_team_technical_fouls := 2

noncomputable def total_chuck_team_score :=
  chuck_team_score_first_quarter + chuck_team_score_second_quarter + 
  chuck_team_score_third_quarter + chuck_team_score_fourth_quarter + 
  chuck_team_technical_fouls

noncomputable def total_yellow_team_score :=
  yellow_team_score_first_quarter + yellow_team_score_second_quarter + 
  yellow_team_score_third_quarter + yellow_team_score_fourth_quarter + 
  yellow_team_technical_fouls

noncomputable def chuck_team_lead :=
  total_chuck_team_score - total_yellow_team_score

theorem chuck_team_leads_by_2 :
  chuck_team_lead = 2 :=
by
  sorry

end chuck_team_leads_by_2_l1628_162816


namespace allocation_schemes_l1628_162826

theorem allocation_schemes :
  ∃ (n : ℕ), n = 240 ∧ (∀ (volunteers : Fin 5 → Fin 4), 
    (∃ (assign : Fin 5 → Fin 4), 
      (∀ (i : Fin 4), ∃ (j : Fin 5), assign j = i)
      ∧ (∀ (k : Fin 5), assign k ≠ assign k)) 
    → true) := sorry

end allocation_schemes_l1628_162826


namespace cos_2theta_l1628_162802

theorem cos_2theta (θ : ℝ) (h : Real.tan θ = Real.sqrt 5) : Real.cos (2 * θ) = -2 / 3 :=
by
  sorry

end cos_2theta_l1628_162802


namespace resultant_after_trebled_l1628_162839

variable (x : ℕ)

theorem resultant_after_trebled (h : x = 7) : 3 * (2 * x + 9) = 69 := by
  sorry

end resultant_after_trebled_l1628_162839


namespace kira_travel_time_l1628_162896

theorem kira_travel_time :
  let time_between_stations := 2 * 60 -- converting hours to minutes
  let break_time := 30 -- in minutes
  let total_time := 2 * time_between_stations + break_time
  total_time = 270 :=
by
  let time_between_stations := 2 * 60
  let break_time := 30
  let total_time := 2 * time_between_stations + break_time
  exact rfl

end kira_travel_time_l1628_162896


namespace french_fries_cost_is_correct_l1628_162854

def burger_cost : ℝ := 5
def soft_drink_cost : ℝ := 3
def special_burger_meal_cost : ℝ := 9.5

def french_fries_cost : ℝ :=
  special_burger_meal_cost - (burger_cost + soft_drink_cost)

theorem french_fries_cost_is_correct :
  french_fries_cost = 1.5 :=
by
  unfold french_fries_cost
  unfold special_burger_meal_cost
  unfold burger_cost
  unfold soft_drink_cost
  sorry

end french_fries_cost_is_correct_l1628_162854


namespace bertha_no_daughters_count_l1628_162861

open Nat

-- Definitions for the conditions
def daughters : ℕ := 8
def total_women : ℕ := 42
def granddaughters : ℕ := total_women - daughters
def daughters_who_have_daughters := granddaughters / 6
def daughters_without_daughters := daughters - daughters_who_have_daughters
def total_without_daughters := granddaughters + daughters_without_daughters

-- The theorem to prove
theorem bertha_no_daughters_count : total_without_daughters = 37 := by
  sorry

end bertha_no_daughters_count_l1628_162861


namespace line_circle_intersect_l1628_162810

theorem line_circle_intersect (m : ℤ) :
  (∃ x y : ℝ, 4 * x + 3 * y + 2 * m = 0 ∧ (x + 3)^2 + (y - 1)^2 = 1) ↔ 2 < m ∧ m < 7 :=
by
  sorry

end line_circle_intersect_l1628_162810


namespace calculate_wholesale_price_l1628_162864

noncomputable def retail_price : ℝ := 108

noncomputable def selling_price (retail_price : ℝ) : ℝ := retail_price * 0.90

noncomputable def selling_price_alt (wholesale_price : ℝ) : ℝ := wholesale_price * 1.20

theorem calculate_wholesale_price (W : ℝ) (R : ℝ) (SP : ℝ)
  (hR : R = 108)
  (hSP1 : SP = selling_price R)
  (hSP2 : SP = selling_price_alt W) : W = 81 :=
by
  -- Proof omitted
  sorry

end calculate_wholesale_price_l1628_162864


namespace store_sells_2_kg_per_week_l1628_162823

def packets_per_week := 20
def grams_per_packet := 100
def grams_per_kg := 1000
def kg_per_week (p : Nat) (gr_per_pkt : Nat) (gr_per_kg : Nat) : Nat :=
  (p * gr_per_pkt) / gr_per_kg

theorem store_sells_2_kg_per_week :
  kg_per_week packets_per_week grams_per_packet grams_per_kg = 2 :=
  sorry

end store_sells_2_kg_per_week_l1628_162823


namespace rational_solution_l1628_162893

theorem rational_solution (a b c : ℚ) 
  (h : (3 * a - 2 * b + c - 4)^2 + (a + 2 * b - 3 * c + 6)^2 + (2 * a - b + 2 * c - 2)^2 ≤ 0) : 
  2 * a + b - 4 * c = -4 := 
by
  sorry

end rational_solution_l1628_162893


namespace find_x_l1628_162824

theorem find_x (x : ℝ) (a b : ℝ × ℝ) (h : a = (Real.cos (3 * x / 2), Real.sin (3 * x / 2)) ∧ b = (Real.cos (x / 2), -Real.sin (x / 2)) ∧ (a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2 = 1 ∧ 0 ≤ x ∧ x ≤ Real.pi)  :
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 :=
by
  sorry

end find_x_l1628_162824


namespace find_p_l1628_162806

theorem find_p (a : ℕ) (ha : a = 2030) : 
  let p := 2 * a + 1;
  let q := a * (a + 1);
  p = 4061 ∧ Nat.gcd p q = 1 := by
  sorry

end find_p_l1628_162806


namespace distinct_real_roots_l1628_162888

theorem distinct_real_roots :
  ∀ x : ℝ, (x^3 - 3*x^2 + x - 2) * (x^3 - x^2 - 4*x + 7) + 6*x^2 - 15*x + 18 = 0 ↔
  x = 1 ∨ x = -2 ∨ x = 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 1 + Real.sqrt 2 :=
by sorry

end distinct_real_roots_l1628_162888


namespace adjustments_to_equal_boys_and_girls_l1628_162850

theorem adjustments_to_equal_boys_and_girls (n : ℕ) :
  let initial_boys := 40
  let initial_girls := 0
  let boys_after_n := initial_boys - 3 * n
  let girls_after_n := initial_girls + 2 * n
  boys_after_n = girls_after_n → n = 8 :=
by
  sorry

end adjustments_to_equal_boys_and_girls_l1628_162850


namespace sqrt_15_estimate_l1628_162858

theorem sqrt_15_estimate : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 :=
by
  sorry

end sqrt_15_estimate_l1628_162858


namespace abigail_lost_money_l1628_162844

-- Conditions
def initial_money := 11
def money_spent := 2
def money_left := 3

-- Statement of the problem as a Lean theorem
theorem abigail_lost_money : initial_money - money_spent - money_left = 6 := by
  sorry

end abigail_lost_money_l1628_162844


namespace cards_from_around_country_l1628_162855

-- Define the total number of cards and the number from home
def total_cards : ℝ := 403.0
def home_cards : ℝ := 287.0

-- Define the expected number of cards from around the country
def expected_country_cards : ℝ := 116.0

-- Theorem statement
theorem cards_from_around_country :
  total_cards - home_cards = expected_country_cards :=
by
  -- Since this only requires the statement, the proof is omitted
  sorry

end cards_from_around_country_l1628_162855


namespace min_value_y_of_parabola_l1628_162885

theorem min_value_y_of_parabola :
  ∃ y : ℝ, ∃ x : ℝ, (∀ y' x', (y' + x') = (y' - x')^2 + 3 * (y' - x') + 3 → y' ≥ y) ∧
            y = -1/2 :=
by
  sorry

end min_value_y_of_parabola_l1628_162885


namespace triangle_obtuse_l1628_162881

theorem triangle_obtuse (a b c : ℝ) (A B C : ℝ) 
  (hBpos : 0 < B) 
  (hBpi : B < Real.pi) 
  (sin_C_lt_cos_A_sin_B : Real.sin C / Real.sin B < Real.cos A) 
  (hC_eq : C = A + B) 
  (ha2 : A + B + C = Real.pi) :
  B > Real.pi / 2 := 
sorry

end triangle_obtuse_l1628_162881


namespace decreased_price_correct_l1628_162808

def actual_cost : ℝ := 250
def percentage_decrease : ℝ := 0.2

theorem decreased_price_correct : actual_cost - (percentage_decrease * actual_cost) = 200 :=
by
  sorry

end decreased_price_correct_l1628_162808


namespace find_x_l1628_162874

theorem find_x (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x)
    (h3 : 1 / a + 1 / b = 1) : x = 6 :=
sorry

end find_x_l1628_162874


namespace find_primes_pqr_eq_5_sum_l1628_162803

theorem find_primes_pqr_eq_5_sum (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) :
  p * q * r = 5 * (p + q + r) → (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 7 ∧ r = 5) ∨
                                         (p = 5 ∧ q = 2 ∧ r = 7) ∨ (p = 5 ∧ q = 7 ∧ r = 2) ∨
                                         (p = 7 ∧ q = 2 ∧ r = 5) ∨ (p = 7 ∧ q = 5 ∧ r = 2) :=
by
  sorry

end find_primes_pqr_eq_5_sum_l1628_162803


namespace find_second_month_sale_l1628_162883

/-- Given sales for specific months and required sales goal -/
def sales_1 := 4000
def sales_3 := 5689
def sales_4 := 7230
def sales_5 := 6000
def sales_6 := 12557
def avg_goal := 7000
def months := 6

theorem find_second_month_sale (x2 : ℕ) :
  (sales_1 + x2 + sales_3 + sales_4 + sales_5 + sales_6) / months = avg_goal →
  x2 = 6524 :=
by
  sorry

end find_second_month_sale_l1628_162883


namespace prime_power_seven_l1628_162836

theorem prime_power_seven (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (eqn : p + 25 = q^7) : p = 103 := by
  sorry

end prime_power_seven_l1628_162836


namespace bus_stop_time_l1628_162846

theorem bus_stop_time (speed_without_stoppages speed_with_stoppages : ℕ) 
(distance : ℕ) (time_without_stoppages time_with_stoppages : ℝ) :
  speed_without_stoppages = 80 ∧ speed_with_stoppages = 40 ∧ distance = 80 ∧
  time_without_stoppages = distance / speed_without_stoppages ∧
  time_with_stoppages = distance / speed_with_stoppages →
  (time_with_stoppages - time_without_stoppages) * 60 = 30 :=
by
  sorry

end bus_stop_time_l1628_162846


namespace maximum_distance_l1628_162842

-- Defining the conditions
def highway_mileage : ℝ := 12.2
def city_mileage : ℝ := 7.6
def gasoline_amount : ℝ := 22

-- Mathematical equivalent proof statement
theorem maximum_distance (h_mileage : ℝ) (g_amount : ℝ) : h_mileage = 12.2 ∧ g_amount = 22 → g_amount * h_mileage = 268.4 :=
by
  intro h
  sorry

end maximum_distance_l1628_162842


namespace sum_of_fractions_as_decimal_l1628_162843

theorem sum_of_fractions_as_decimal : (3 / 8 : ℝ) + (5 / 32) = 0.53125 := by
  sorry

end sum_of_fractions_as_decimal_l1628_162843


namespace number_of_people_tasting_apple_pies_l1628_162832

/-- Sedrach's apple pie problem -/
def apple_pies : ℕ := 13
def halves_per_apple_pie : ℕ := 2
def bite_size_samples_per_half : ℕ := 5

theorem number_of_people_tasting_apple_pies :
    (apple_pies * halves_per_apple_pie * bite_size_samples_per_half) = 130 :=
by
  sorry

end number_of_people_tasting_apple_pies_l1628_162832


namespace average_temperature_second_to_fifth_days_l1628_162879

variable (T1 T2 T3 T4 T5 : ℝ)

theorem average_temperature_second_to_fifth_days 
  (h1 : (T1 + T2 + T3 + T4) / 4 = 58)
  (h2 : T1 / T5 = 7 / 8)
  (h3 : T5 = 32) :
  (T2 + T3 + T4 + T5) / 4 = 59 :=
by
  sorry

end average_temperature_second_to_fifth_days_l1628_162879


namespace answer_l1628_162882

def p := ∃ x : ℝ, x - 2 > Real.log x
def q := ∀ x : ℝ, Real.exp x > 1

theorem answer (hp : p) (hq : ¬ q) : p ∧ ¬ q :=
  by
    exact ⟨hp, hq⟩

end answer_l1628_162882


namespace rainy_days_l1628_162891

theorem rainy_days
  (rain_on_first_day : ℕ) (rain_on_second_day : ℕ) (rain_on_third_day : ℕ) (sum_of_first_two_days : ℕ)
  (h1 : rain_on_first_day = 4)
  (h2 : rain_on_second_day = 5 * rain_on_first_day)
  (h3 : sum_of_first_two_days = rain_on_first_day + rain_on_second_day)
  (h4 : rain_on_third_day = sum_of_first_two_days - 6) :
  rain_on_third_day = 18 :=
by
  sorry

end rainy_days_l1628_162891


namespace mr_willson_friday_work_time_l1628_162859

theorem mr_willson_friday_work_time :
  let monday := 3 / 4
  let tuesday := 1 / 2
  let wednesday := 2 / 3
  let thursday := 5 / 6
  let total_work := 4
  let time_monday_to_thursday := monday + tuesday + wednesday + thursday
  let time_friday := total_work - time_monday_to_thursday
  time_friday * 60 = 75 :=
by
  sorry

end mr_willson_friday_work_time_l1628_162859


namespace probability_at_least_one_passes_l1628_162880

theorem probability_at_least_one_passes (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 :=
by sorry

end probability_at_least_one_passes_l1628_162880


namespace point_in_second_quadrant_l1628_162886

/-- Define the quadrants in the Cartesian coordinate system -/
def quadrant (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On the axis"

theorem point_in_second_quadrant :
  quadrant (-3) 2005 = "Second quadrant" :=
by
  sorry

end point_in_second_quadrant_l1628_162886


namespace incorrect_statement_C_l1628_162870

/-- 
  Prove that the function y = -1/2 * x + 3 does not intersect the y-axis at (6,0).
-/
theorem incorrect_statement_C 
: ∀ (x y : ℝ), y = -1/2 * x + 3 → (x, y) ≠ (6, 0) :=
by
  intros x y h
  sorry

end incorrect_statement_C_l1628_162870


namespace total_GDP_l1628_162875

noncomputable def GDP_first_quarter : ℝ := 232
noncomputable def GDP_fourth_quarter : ℝ := 241

theorem total_GDP (x y : ℝ) (h1 : GDP_first_quarter < x)
                  (h2 : x < y) (h3 : y < GDP_fourth_quarter)
                  (h4 : (x + y) / 2 = (GDP_first_quarter + x + y + GDP_fourth_quarter) / 4) :
  GDP_first_quarter + x + y + GDP_fourth_quarter = 946 :=
by
  sorry

end total_GDP_l1628_162875


namespace tomato_price_l1628_162833

theorem tomato_price (P : ℝ) (W : ℝ) :
  (0.9956 * 0.9 * W = P * W + 0.12 * (P * W)) → P = 0.8 :=
by
  intro h
  sorry

end tomato_price_l1628_162833


namespace Alex_age_l1628_162873

theorem Alex_age : ∃ (x : ℕ), (∃ (y : ℕ), x - 2 = y^2) ∧ (∃ (z : ℕ), x + 2 = z^3) ∧ x = 6 := by
  sorry

end Alex_age_l1628_162873


namespace time_left_after_council_room_is_zero_l1628_162876

-- Define the conditions
def totalTimeAllowed : ℕ := 30
def travelToSchoolTime : ℕ := 25
def walkToLibraryTime : ℕ := 3
def returnBooksTime : ℕ := 4
def walkToCouncilRoomTime : ℕ := 5
def submitProjectTime : ℕ := 3

-- Calculate time spent up to the student council room
def timeSpentUpToCouncilRoom : ℕ :=
  travelToSchoolTime + walkToLibraryTime + returnBooksTime + walkToCouncilRoomTime + submitProjectTime

-- Question: How much time is left after leaving the student council room to reach the classroom without being late?
theorem time_left_after_council_room_is_zero (totalTimeAllowed travelToSchoolTime walkToLibraryTime returnBooksTime walkToCouncilRoomTime submitProjectTime : ℕ):
  totalTimeAllowed - timeSpentUpToCouncilRoom = 0 := by
  sorry

end time_left_after_council_room_is_zero_l1628_162876


namespace reciprocal_sqrt5_minus_2_l1628_162884

theorem reciprocal_sqrt5_minus_2 : 1 / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := 
by
  sorry

end reciprocal_sqrt5_minus_2_l1628_162884


namespace total_amount_paid_l1628_162865

theorem total_amount_paid (cost_of_manicure : ℝ) (tip_percentage : ℝ) (total : ℝ) 
  (h1 : cost_of_manicure = 30) (h2 : tip_percentage = 0.3) (h3 : total = cost_of_manicure + cost_of_manicure * tip_percentage) : 
  total = 39 :=
by
  sorry

end total_amount_paid_l1628_162865


namespace count_right_triangles_l1628_162835

theorem count_right_triangles: 
  ∃ n : ℕ, n = 9 ∧ ∃ (a b : ℕ), a^2 + b^2 = (b+2)^2 ∧ b < 100 ∧ a > 0 ∧ b > 0 := by
  sorry

end count_right_triangles_l1628_162835


namespace problems_per_hour_l1628_162863

theorem problems_per_hour :
  ∀ (mathProblems spellingProblems totalHours problemsPerHour : ℕ), 
    mathProblems = 36 →
    spellingProblems = 28 →
    totalHours = 8 →
    (mathProblems + spellingProblems) / totalHours = problemsPerHour →
    problemsPerHour = 8 :=
by
  intros
  subst_vars
  sorry

end problems_per_hour_l1628_162863


namespace rectangle_y_coordinate_l1628_162822

theorem rectangle_y_coordinate (x1 x2 y1 A : ℝ) (h1 : x1 = -8) (h2 : x2 = 1) (h3 : y1 = 1) (h4 : A = 72)
    (hL : x2 - x1 = 9) (hA : A = 9 * (y - y1)) :
    (y = 9) :=
by
  sorry

end rectangle_y_coordinate_l1628_162822


namespace xy_sufficient_not_necessary_l1628_162840

theorem xy_sufficient_not_necessary (x y : ℝ) :
  (xy ≠ 6) → (x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → xy ≠ 6) := by
  sorry

end xy_sufficient_not_necessary_l1628_162840


namespace magnitude_of_z_l1628_162838

theorem magnitude_of_z (z : ℂ) (h : z * (1 + 2 * Complex.I) + Complex.I = 0) : 
  Complex.abs z = Real.sqrt (5) / 5 := 
sorry

end magnitude_of_z_l1628_162838


namespace compute_combination_product_l1628_162887

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem compute_combination_product :
  combination 10 3 * combination 8 3 = 6720 :=
by
  sorry

end compute_combination_product_l1628_162887


namespace decimal_sum_difference_l1628_162820

theorem decimal_sum_difference :
  (0.5 - 0.03 + 0.007 + 0.0008 = 0.4778) :=
by
  sorry

end decimal_sum_difference_l1628_162820


namespace inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc_l1628_162829

theorem inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc (a b c : ℝ) :
  5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * a * c + 4 * b * c ∧
  (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * a * c + 4 * b * c → a = 0 ∧ b = 0 ∧ c = 0) := sorry

end inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc_l1628_162829


namespace profit_percentage_l1628_162807

theorem profit_percentage (cost_price selling_price profit_percentage : ℚ) 
  (h_cost_price : cost_price = 240) 
  (h_selling_price : selling_price = 288) 
  (h_profit_percentage : profit_percentage = 20) : 
  profit_percentage = ((selling_price - cost_price) / cost_price) * 100 := 
by 
  sorry

end profit_percentage_l1628_162807


namespace scientific_notation_example_l1628_162817

theorem scientific_notation_example :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 218000000 = a * 10 ^ n ∧ a = 2.18 ∧ n = 8 :=
by {
  -- statement of the problem conditions
  sorry
}

end scientific_notation_example_l1628_162817


namespace major_premise_wrong_l1628_162872

-- Definitions of the given conditions in Lean
def is_parallel_to_plane (line : Type) (plane : Type) : Prop := sorry -- Provide an appropriate definition
def contains_line (plane : Type) (line : Type) : Prop := sorry -- Provide an appropriate definition
def is_parallel_to_line (line1 : Type) (line2 : Type) : Prop := sorry -- Provide an appropriate definition

-- Given conditions
variables (b α a : Type)
variable (H1 : ¬ contains_line α b)  -- Line b is not contained in plane α
variable (H2 : contains_line α a)    -- Line a is contained in plane α
variable (H3 : is_parallel_to_plane b α) -- Line b is parallel to plane α

-- Proposition to prove: The major premise is wrong
theorem major_premise_wrong : ¬(∀ (a b : Type), is_parallel_to_plane b α → contains_line α a → is_parallel_to_line b a) :=
by
  sorry

end major_premise_wrong_l1628_162872


namespace range_m_l1628_162869

open Real

theorem range_m (m : ℝ)
  (hP : ¬ (∃ x : ℝ, m * x^2 + 1 ≤ 0))
  (hQ : ¬ (∃ x : ℝ, x^2 + m * x + 1 < 0)) :
  0 ≤ m ∧ m ≤ 2 := 
sorry

end range_m_l1628_162869


namespace numbers_product_l1628_162828

theorem numbers_product (x y : ℝ) (h1 : x + y = 24) (h2 : x - y = 8) : x * y = 128 := by
  sorry

end numbers_product_l1628_162828


namespace bases_to_make_equality_l1628_162866

theorem bases_to_make_equality (a b : ℕ) (h : 3 * a^2 + 4 * a + 2 = 9 * b + 7) : 
  (3 * a^2 + 4 * a + 2 = 342) ∧ (9 * b + 7 = 97) :=
by
  sorry

end bases_to_make_equality_l1628_162866


namespace no_solutions_for_specific_a_l1628_162818

theorem no_solutions_for_specific_a (a : ℝ) :
  (a < -9) ∨ (a > 0) →
  ¬ ∃ x : ℝ, 5 * |x - 4 * a| + |x - a^2| + 4 * x - 3 * a = 0 :=
by sorry

end no_solutions_for_specific_a_l1628_162818


namespace cary_needs_6_weekends_l1628_162831

variable (shoe_cost : ℕ)
variable (current_savings : ℕ)
variable (earn_per_lawn : ℕ)
variable (lawns_per_weekend : ℕ)
variable (w : ℕ)

theorem cary_needs_6_weekends
    (h1 : shoe_cost = 120)
    (h2 : current_savings = 30)
    (h3 : earn_per_lawn = 5)
    (h4 : lawns_per_weekend = 3)
    (h5 : w * (earn_per_lawn * lawns_per_weekend) = shoe_cost - current_savings) :
    w = 6 :=
by sorry

end cary_needs_6_weekends_l1628_162831


namespace rectangle_dimensions_l1628_162897

theorem rectangle_dimensions (w l : ℕ) (h : l = w + 5) (hp : 2 * l + 2 * w = 34) : w = 6 ∧ l = 11 := 
by 
  sorry

end rectangle_dimensions_l1628_162897


namespace L_shape_area_and_perimeter_l1628_162847

def rectangle1_length := 0.5
def rectangle1_width := 0.3
def rectangle2_length := 0.2
def rectangle2_width := 0.5

def area_rectangle1 := rectangle1_length * rectangle1_width
def area_rectangle2 := rectangle2_length * rectangle2_width
def total_area := area_rectangle1 + area_rectangle2

def perimeter_L_shape := rectangle1_length + rectangle1_width + rectangle1_width + rectangle2_length + rectangle2_length + rectangle2_width

theorem L_shape_area_and_perimeter :
  total_area = 0.25 ∧ perimeter_L_shape = 2.0 :=
by
  sorry

end L_shape_area_and_perimeter_l1628_162847


namespace five_person_lineup_l1628_162805

theorem five_person_lineup : 
  let total_ways := Nat.factorial 5
  let invalid_first := Nat.factorial 4
  let invalid_last := Nat.factorial 4
  let valid_ways := total_ways - (invalid_first + invalid_last)
  valid_ways = 72 :=
by
  sorry

end five_person_lineup_l1628_162805


namespace smallest_possible_n_l1628_162837

theorem smallest_possible_n (n : ℕ) (h : ∃ k : ℕ, 15 * n - 2 = 11 * k) : n % 11 = 6 :=
by
  sorry

end smallest_possible_n_l1628_162837


namespace calculate_a_plus_b_l1628_162851

theorem calculate_a_plus_b (a b : ℝ) (h1 : 3 = a + b / 2) (h2 : 2 = a + b / 4) : a + b = 5 :=
by
  sorry

end calculate_a_plus_b_l1628_162851


namespace sequence_divisibility_count_l1628_162860

theorem sequence_divisibility_count :
  ∀ (f : ℕ → ℕ), (∀ n, n ≥ 2 → f n = 10^n - 1) → 
  (∃ count, count = 504 ∧ ∀ i, 2 ≤ i ∧ i ≤ 2023 → (101 ∣ f i ↔ i % 4 = 0)) :=
by { sorry }

end sequence_divisibility_count_l1628_162860


namespace customer_bought_two_pens_l1628_162804

noncomputable def combination (n k : ℕ) : ℝ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem customer_bought_two_pens :
  ∃ n : ℕ, combination 5 n / combination 8 n = 0.3571428571428571 ↔ n = 2 := by
  sorry

end customer_bought_two_pens_l1628_162804


namespace simplifiedtown_path_difference_l1628_162845

/-- In Simplifiedtown, all streets are 30 feet wide. Each enclosed block forms a square with 
each side measuring 400 feet. Sarah runs exactly next to the block on a path that is 400 feet 
from the block's inner edge while Maude runs on the outer edge of the street opposite to 
Sarah. Prove that Maude runs 120 feet more than Sarah for each lap around the block. -/
theorem simplifiedtown_path_difference :
  let street_width := 30
  let block_side := 400
  let sarah_path := block_side
  let maude_path := block_side + street_width
  let sarah_lap := 4 * sarah_path
  let maude_lap := 4 * maude_path
  maude_lap - sarah_lap = 120 :=
by
  let street_width := 30
  let block_side := 400
  let sarah_path := block_side
  let maude_path := block_side + street_width
  let sarah_lap := 4 * sarah_path
  let maude_lap := 4 * maude_path
  show maude_lap - sarah_lap = 120
  sorry

end simplifiedtown_path_difference_l1628_162845


namespace value_of_expression_l1628_162827

theorem value_of_expression {a b c : ℝ} (h_eqn : a + b + c = 15)
  (h_ab_bc_ca : ab + bc + ca = 13) (h_abc : abc = 8)
  (h_roots : Polynomial.roots (Polynomial.X^3 - 15 * Polynomial.X^2 + 13 * Polynomial.X - 8) = {a, b, c}) :
  (a / (1/a + b*c)) + (b / (1/b + c*a)) + (c / (1/c + a*b)) = 199/9 :=
by sorry

end value_of_expression_l1628_162827


namespace probability_of_red_ball_l1628_162809

theorem probability_of_red_ball (total_balls red_balls black_balls white_balls : ℕ)
  (h1 : total_balls = 7)
  (h2 : red_balls = 2)
  (h3 : black_balls = 4)
  (h4 : white_balls = 1) :
  (red_balls / total_balls : ℚ) = 2 / 7 :=
by {
  sorry
}

end probability_of_red_ball_l1628_162809


namespace boxes_sold_l1628_162834

theorem boxes_sold (start_boxes sold_boxes left_boxes : ℕ) (h1 : start_boxes = 10) (h2 : left_boxes = 5) (h3 : start_boxes - sold_boxes = left_boxes) : sold_boxes = 5 :=
by
  sorry

end boxes_sold_l1628_162834


namespace height_difference_petronas_empire_state_l1628_162867

theorem height_difference_petronas_empire_state :
  let esb_height := 443
  let pt_height := 452
  pt_height - esb_height = 9 := by
  sorry

end height_difference_petronas_empire_state_l1628_162867


namespace men_per_table_l1628_162890

theorem men_per_table (total_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) (total_women : ℕ)
    (h1 : total_tables = 9)
    (h2 : women_per_table = 7)
    (h3 : total_customers = 90)
    (h4 : total_women = women_per_table * total_tables)
    (h5 : total_women + total_men = total_customers) :
  total_men / total_tables = 3 :=
by
  have total_women := 7 * 9
  have total_men := 90 - total_women
  exact sorry

end men_per_table_l1628_162890


namespace members_play_both_eq_21_l1628_162852

-- Given definitions
def TotalMembers := 80
def MembersPlayBadminton := 48
def MembersPlayTennis := 46
def MembersPlayNeither := 7

-- Inclusion-Exclusion Principle application to solve the problem
def MembersPlayBoth : ℕ := MembersPlayBadminton + MembersPlayTennis - (TotalMembers - MembersPlayNeither)

-- The theorem we want to prove
theorem members_play_both_eq_21 : MembersPlayBoth = 21 :=
by
  -- skipping the proof
  sorry

end members_play_both_eq_21_l1628_162852


namespace set_complement_union_l1628_162894

-- Definitions of the sets
def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

-- The statement to prove
theorem set_complement_union : (U \ A) ∪ (U \ B) = {1, 4, 5} :=
by sorry

end set_complement_union_l1628_162894


namespace walkway_area_correct_l1628_162825

/-- Define the dimensions of a single flower bed. --/
def flower_bed_length : ℝ := 8
def flower_bed_width : ℝ := 3

/-- Define the number of flower beds in rows and columns. --/
def rows : ℕ := 4
def cols : ℕ := 3

/-- Define the width of the walkways surrounding the flower beds. --/
def walkway_width : ℝ := 2

/-- Calculate the total dimensions of the garden including walkways. --/
def total_garden_width : ℝ := (cols * flower_bed_length) + ((cols + 1) * walkway_width)
def total_garden_height : ℝ := (rows * flower_bed_width) + ((rows + 1) * walkway_width)

/-- Calculate the total area of the garden including walkways. --/
def total_garden_area : ℝ := total_garden_width * total_garden_height

/-- Calculate the total area of the flower beds. --/
def flower_bed_area : ℝ := flower_bed_length * flower_bed_width
def total_flower_beds_area : ℝ := rows * cols * flower_bed_area

/-- Calculate the total area of the walkways. --/
def walkway_area := total_garden_area - total_flower_beds_area

theorem walkway_area_correct : walkway_area = 416 := 
by
  -- Proof omitted
  sorry

end walkway_area_correct_l1628_162825


namespace initial_chips_in_bag_l1628_162815

-- Definitions based on conditions
def chips_given_to_brother : ℕ := 7
def chips_given_to_sister : ℕ := 5
def chips_kept_by_nancy : ℕ := 10

-- Theorem statement
theorem initial_chips_in_bag (total_chips := chips_given_to_brother + chips_given_to_sister + chips_kept_by_nancy) : total_chips = 22 := 
by 
  -- we state the assertion
  sorry

end initial_chips_in_bag_l1628_162815


namespace lisa_flight_time_l1628_162898

theorem lisa_flight_time
  (distance : ℕ) (speed : ℕ) (time : ℕ)
  (h_distance : distance = 256)
  (h_speed : speed = 32)
  (h_time : time = distance / speed) :
  time = 8 :=
by sorry

end lisa_flight_time_l1628_162898


namespace common_difference_is_half_l1628_162800

variable (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ) (q p : ℕ)

-- Conditions
def condition1 : Prop := a p = 4
def condition2 : Prop := a q = 2
def condition3 : Prop := p = 4 + q
def arithmetic_sequence : Prop := ∀ n : ℕ, a n = a₁ + (n - 1) * d

-- Proof statement
theorem common_difference_is_half 
  (h1 : condition1 a p)
  (h2 : condition2 a q)
  (h3 : condition3 p q)
  (as : arithmetic_sequence a a₁ d)
  : d = 1 / 2 := 
sorry

end common_difference_is_half_l1628_162800


namespace high_school_sampling_problem_l1628_162813

theorem high_school_sampling_problem :
  let first_year_classes := 20
  let first_year_students_per_class := 50
  let first_year_total_students := first_year_classes * first_year_students_per_class
  let second_year_classes := 24
  let second_year_students_per_class := 45
  let second_year_total_students := second_year_classes * second_year_students_per_class
  let total_students := first_year_total_students + second_year_total_students
  let survey_students := 208
  let first_year_sample := (first_year_total_students * survey_students) / total_students
  let second_year_sample := (second_year_total_students * survey_students) / total_students
  let A_selected_probability := first_year_sample / first_year_total_students
  let B_selected_probability := second_year_sample / second_year_total_students
  (survey_students = 208) →
  (first_year_sample = 100) →
  (second_year_sample = 108) →
  (A_selected_probability = 1 / 10) →
  (B_selected_probability = 1 / 10) →
  (A_selected_probability = B_selected_probability) →
  (student_A_in_first_year : true) →
  (student_B_in_second_year : true) →
  true :=
  by sorry

end high_school_sampling_problem_l1628_162813


namespace proof_inequalities_l1628_162878

variable {R : Type} [LinearOrder R] [Ring R]

def odd_function (f : R → R) : Prop :=
∀ x : R, f (-x) = -f x

def decreasing_function (f : R → R) : Prop :=
∀ x y : R, x ≤ y → f y ≤ f x

theorem proof_inequalities (f : R → R) (a b : R) 
  (h_odd : odd_function f)
  (h_decr : decreasing_function f)
  (h : a + b ≤ 0) :
  (f a * f (-a) ≤ 0) ∧ (f a + f b ≥ f (-a) + f (-b)) :=
by
  sorry

end proof_inequalities_l1628_162878


namespace parabola_directrix_l1628_162848

theorem parabola_directrix (a : ℝ) : 
  (∃ y, (y ^ 2 = 4 * a * (-2))) → a = 2 :=
by
  sorry

end parabola_directrix_l1628_162848


namespace find_a5_l1628_162801

-- Define the sequence and its properties
def geom_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a (n + m) = (2^m) * a n

-- Define the problem statement
def sum_of_first_five_terms_is_31 (a : ℕ → ℕ) : Prop :=
a 1 + a 2 + a 3 + a 4 + a 5 = 31

-- State the theorem to prove
theorem find_a5 (a : ℕ → ℕ) (h_geom : geom_sequence a) (h_sum : sum_of_first_five_terms_is_31 a) : a 5 = 16 :=
by
  sorry

end find_a5_l1628_162801


namespace find_quadruples_l1628_162857

def quadrupleSolution (a b c d : ℝ): Prop :=
  (a * (b + c) = b * (c + d) ∧ b * (c + d) = c * (d + a) ∧ c * (d + a) = d * (a + b))

def isSolution (a b c d : ℝ): Prop :=
  (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 1 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 1 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 1) ∨
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 1 ∧ b = -1 ∧ c = 1 ∧ d = -1) ∨
  (a = 1 ∧ b = -1 + Real.sqrt 2 ∧ c = -1 ∧ d = 1 - Real.sqrt 2) ∨
  (a = 1 ∧ b = -1 - Real.sqrt 2 ∧ c = -1 ∧ d = 1 + Real.sqrt 2)

theorem find_quadruples (a b c d : ℝ) :
  quadrupleSolution a b c d ↔ isSolution a b c d :=
sorry

end find_quadruples_l1628_162857


namespace most_consistent_player_l1628_162895

section ConsistentPerformance

variables (σA σB σC σD : ℝ)
variables (σA_eq : σA = 0.023)
variables (σB_eq : σB = 0.018)
variables (σC_eq : σC = 0.020)
variables (σD_eq : σD = 0.021)

theorem most_consistent_player : σB < σC ∧ σB < σD ∧ σB < σA :=
by 
  rw [σA_eq, σB_eq, σC_eq, σD_eq]
  sorry

end ConsistentPerformance

end most_consistent_player_l1628_162895


namespace jacks_speed_is_7_l1628_162856

-- Define the constants and speeds as given in conditions
def initial_distance : ℝ := 150
def christina_speed : ℝ := 8
def lindy_speed : ℝ := 10
def lindy_total_distance : ℝ := 100

-- Hypothesis stating when the three meet
theorem jacks_speed_is_7 :
  ∃ (jack_speed : ℝ), (∃ (time : ℝ), 
    time = lindy_total_distance / lindy_speed
    ∧ christina_speed * time + jack_speed * time = initial_distance) 
  → jack_speed = 7 :=
by {
  -- Placeholder for the proof
  sorry
}

end jacks_speed_is_7_l1628_162856


namespace arithmetic_geometric_sequences_l1628_162862

theorem arithmetic_geometric_sequences :
  ∀ (a₁ a₂ b₁ b₂ b₃ : ℤ), 
  (a₂ = a₁ + (a₁ - (-1))) ∧ 
  (-4 = -1 + 3 * (a₂ - a₁)) ∧ 
  (-4 = -1 * (b₃/b₁)^4) ∧ 
  (b₂ = b₁ * (b₂/b₁)^2) →
  (a₂ - a₁) / b₂ = 1 / 2 := 
by
  intros a₁ a₂ b₁ b₂ b₃ h
  sorry

end arithmetic_geometric_sequences_l1628_162862
