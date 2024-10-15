import Mathlib

namespace NUMINAMATH_GPT_least_positive_value_l699_69921

theorem least_positive_value (x y z : ℤ) : ∃ x y z : ℤ, 0 < 72 * x + 54 * y + 36 * z ∧ ∀ (a b c : ℤ), 0 < 72 * a + 54 * b + 36 * c → 72 * x + 54 * y + 36 * z ≤ 72 * a + 54 * b + 36 * c :=
sorry

end NUMINAMATH_GPT_least_positive_value_l699_69921


namespace NUMINAMATH_GPT_product_expression_evaluation_l699_69908

theorem product_expression_evaluation :
  (1 + 2 / 1) * (1 + 2 / 2) * (1 + 2 / 3) * (1 + 2 / 4) * (1 + 2 / 5) * (1 + 2 / 6) - 1 = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_product_expression_evaluation_l699_69908


namespace NUMINAMATH_GPT_initial_southwards_distance_l699_69968

-- Define a structure that outlines the journey details
structure Journey :=
  (southwards : ℕ) 
  (westwards1 : ℕ := 10)
  (northwards : ℕ := 20)
  (westwards2 : ℕ := 20) 
  (home_distance : ℕ := 30)

-- Main theorem statement without proof
theorem initial_southwards_distance (j : Journey) : j.southwards + j.northwards = j.home_distance → j.southwards = 10 := by
  intro h
  sorry

end NUMINAMATH_GPT_initial_southwards_distance_l699_69968


namespace NUMINAMATH_GPT_gcd_A_B_l699_69975

def A : ℤ := 1989^1990 - 1988^1990
def B : ℤ := 1989^1989 - 1988^1989

theorem gcd_A_B : Int.gcd A B = 1 := 
by
  -- Conditions
  have h1 : A = 1989^1990 - 1988^1990 := rfl
  have h2 : B = 1989^1989 - 1988^1989 := rfl
  -- Conclusion
  sorry

end NUMINAMATH_GPT_gcd_A_B_l699_69975


namespace NUMINAMATH_GPT_find_p_q_l699_69936

theorem find_p_q (p q : ℚ) : 
    (∀ x, x^5 - x^4 + x^3 - p*x^2 + q*x + 9 = 0 → (x = -3 ∨ x = 2)) →
    (p, q) = (-19.5, -55.5) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_p_q_l699_69936


namespace NUMINAMATH_GPT_bleach_to_detergent_ratio_changed_factor_l699_69999

theorem bleach_to_detergent_ratio_changed_factor :
  let original_bleach : ℝ := 4
  let original_detergent : ℝ := 40
  let original_water : ℝ := 100
  let altered_detergent : ℝ := 60
  let altered_water : ℝ := 300

  -- Calculate the factor by which the volume increased
  let original_total_volume := original_detergent + original_water
  let altered_total_volume := altered_detergent + altered_water
  let volume_increase_factor := altered_total_volume / original_total_volume

  -- The calculated factor of the ratio change
  let original_ratio_bleach_to_detergent := original_bleach / original_detergent

  altered_detergent > 0 → altered_water > 0 →
  volume_increase_factor * original_ratio_bleach_to_detergent = 2.5714 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_bleach_to_detergent_ratio_changed_factor_l699_69999


namespace NUMINAMATH_GPT_senior_ticket_cost_l699_69954

theorem senior_ticket_cost (total_tickets : ℕ) (adult_ticket_cost : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) (senior_ticket_cost : ℕ) :
  total_tickets = 510 →
  adult_ticket_cost = 21 →
  total_receipts = 8748 →
  senior_tickets = 327 →
  senior_ticket_cost = 15 :=
by
  sorry

end NUMINAMATH_GPT_senior_ticket_cost_l699_69954


namespace NUMINAMATH_GPT_trader_sold_meters_l699_69905

variable (x : ℕ) (SP P CP : ℕ)

theorem trader_sold_meters (h_SP : SP = 660) (h_P : P = 5) (h_CP : CP = 5) : x = 66 :=
  by
  sorry

end NUMINAMATH_GPT_trader_sold_meters_l699_69905


namespace NUMINAMATH_GPT_xiamen_fabric_production_l699_69987

theorem xiamen_fabric_production:
  (∃ x y : ℕ, (3 * ((2 * x) / 3) + 3 * (y / 3) = 600) ∧ (2 * ((2 * x) / 3) = 3 * (y / 3))) ∧
  (∀ x y : ℕ, (3 * ((2 * x) / 3) + 3 * (y / 3) = 600) ∧ (2 * ((2 * x) / 3) = 3 * (y / 3)) →
    x = 360 ∧ y = 240 ∧ y / 3 = 240) := 
by
  sorry

end NUMINAMATH_GPT_xiamen_fabric_production_l699_69987


namespace NUMINAMATH_GPT_cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8_l699_69967

theorem cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8
  (α : ℝ) (h : Real.cos α = 2 * Real.cos (α + Real.pi / 4)) :
  Real.tan (α + Real.pi / 8) = 3 * (Real.sqrt 2 + 1) := 
sorry

end NUMINAMATH_GPT_cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8_l699_69967


namespace NUMINAMATH_GPT_toms_age_ratio_l699_69992

variables (T N : ℕ)

-- Conditions
def toms_age (T : ℕ) := T
def sum_of_children_ages (T : ℕ) := T
def years_ago (T N : ℕ) := T - N
def children_ages_years_ago (T N : ℕ) := T - 4 * N

-- Given statement
theorem toms_age_ratio (h1 : toms_age T = sum_of_children_ages T)
  (h2 : years_ago T N = 3 * children_ages_years_ago T N) :
  T / N = 11 / 2 :=
sorry

end NUMINAMATH_GPT_toms_age_ratio_l699_69992


namespace NUMINAMATH_GPT_ReuleauxTriangleFitsAll_l699_69998

-- Assume definitions for fits into various slots

def FitsTriangular (s : Type) : Prop := sorry
def FitsSquare (s : Type) : Prop := sorry
def FitsCircular (s : Type) : Prop := sorry
def ReuleauxTriangle (s : Type) : Prop := sorry

theorem ReuleauxTriangleFitsAll (s : Type) (h : ReuleauxTriangle s) : 
  FitsTriangular s ∧ FitsSquare s ∧ FitsCircular s := 
  sorry

end NUMINAMATH_GPT_ReuleauxTriangleFitsAll_l699_69998


namespace NUMINAMATH_GPT_correct_calculated_value_l699_69956

theorem correct_calculated_value (x : ℤ) 
  (h : x / 16 = 8 ∧ x % 16 = 4) : (x * 16 + 8 = 2120) := by
  sorry

end NUMINAMATH_GPT_correct_calculated_value_l699_69956


namespace NUMINAMATH_GPT_find_q_l699_69962

def Q (x : ℝ) (p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_q (p q r : ℝ) (h1 : -p = 2 * (-r)) (h2 : -p = 1 + p + q + r) (hy_intercept : r = 5) : q = -24 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l699_69962


namespace NUMINAMATH_GPT_book_price_l699_69939

theorem book_price (B P : ℝ) 
  (h1 : (1 / 3) * B = 36) 
  (h2 : (2 / 3) * B * P = 252) : 
  P = 3.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_book_price_l699_69939


namespace NUMINAMATH_GPT_compute_a1d1_a2d2_a3d3_l699_69917

noncomputable def polynomial_equation (a1 a2 a3 d1 d2 d3: ℝ) : Prop :=
  ∀ x : ℝ, (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)

theorem compute_a1d1_a2d2_a3d3 (a1 a2 a3 d1 d2 d3 : ℝ) (h : polynomial_equation a1 a2 a3 d1 d2 d3) : 
  a1 * d1 + a2 * d2 + a3 * d3 = 1 :=
  sorry

end NUMINAMATH_GPT_compute_a1d1_a2d2_a3d3_l699_69917


namespace NUMINAMATH_GPT_John_paid_total_l699_69961

def vet_cost : ℝ := 400
def num_appointments : ℕ := 3
def insurance_cost : ℝ := 100
def coverage_rate : ℝ := 0.8

def discount : ℝ := vet_cost * coverage_rate
def discounted_visits : ℕ := num_appointments - 1
def discounted_cost : ℝ := vet_cost - discount
def total_discounted_cost : ℝ := discounted_visits * discounted_cost
def J_total : ℝ := vet_cost + total_discounted_cost + insurance_cost

theorem John_paid_total : J_total = 660 := by
  sorry

end NUMINAMATH_GPT_John_paid_total_l699_69961


namespace NUMINAMATH_GPT_seating_arrangements_l699_69904

theorem seating_arrangements (p : Fin 5 → Fin 5 → Prop) :
  (∃! i j : Fin 5, p i j ∧ i = j) →
  (∃! i j : Fin 5, p i j ∧ i ≠ j) →
  ∃ ways : ℕ,
  ways = 20 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l699_69904


namespace NUMINAMATH_GPT_original_fraction_is_one_third_l699_69951

theorem original_fraction_is_one_third (a b : ℕ) 
  (coprime_ab : Nat.gcd a b = 1) 
  (h : (a + 2) * b = 3 * a * b^2) : 
  (a = 1 ∧ b = 3) := 
by 
  sorry

end NUMINAMATH_GPT_original_fraction_is_one_third_l699_69951


namespace NUMINAMATH_GPT_rectangle_area_l699_69969

theorem rectangle_area (d : ℝ) (w : ℝ) (h : (3 * w)^2 + w^2 = d^2) : 
  3 * w^2 = d^2 / 10 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l699_69969


namespace NUMINAMATH_GPT_fruit_days_l699_69912

/-
  Henry and his brother believe in the famous phrase, "An apple a day, keeps the doctor away." 
  Henry's sister, however, believes that "A banana a day makes the trouble fade away" 
  and their father thinks that "An orange a day will keep the weaknesses at bay." 
  A box of apples contains 14 apples, a box of bananas has 20 bananas, and a box of oranges contains 12 oranges. 

  If Henry and his brother eat 1 apple each a day, their sister consumes 2 bananas per day, 
  and their father eats 3 oranges per day, how many days can the family of four continue eating fruits 
  if they have 3 boxes of apples, 4 boxes of bananas, and 5 boxes of oranges? 

  However, due to seasonal changes, oranges are only available for the first 20 days. 
  Moreover, Henry's sister has decided to only eat bananas on days when the day of the month is an odd number. 
  Considering these constraints, determine the total number of days the family of four can continue eating their preferred fruits.
-/

def apples_per_box := 14
def bananas_per_box := 20
def oranges_per_box := 12

def apples_boxes := 3
def bananas_boxes := 4
def oranges_boxes := 5

def daily_apple_consumption := 2
def daily_banana_consumption := 2
def daily_orange_consumption := 3

def orange_availability_days := 20

def odd_days_in_month := 16

def total_number_of_days : ℕ :=
  let total_apples := apples_boxes * apples_per_box
  let total_bananas := bananas_boxes * bananas_per_box
  let total_oranges := oranges_boxes * oranges_per_box
  
  let days_with_apples := total_apples / daily_apple_consumption
  let days_with_bananas := (total_bananas / (odd_days_in_month * daily_banana_consumption)) * 30
  let days_with_oranges := if total_oranges / daily_orange_consumption > orange_availability_days then orange_availability_days else total_oranges / daily_orange_consumption
  min (min days_with_apples days_with_oranges) (days_with_bananas / 30 * 30)

theorem fruit_days : total_number_of_days = 20 := 
  sorry

end NUMINAMATH_GPT_fruit_days_l699_69912


namespace NUMINAMATH_GPT_group_A_percentage_l699_69933

/-!
In an examination, there are 100 questions divided into 3 groups A, B, and C such that each group contains at least one question. 
Each question in group A carries 1 mark, each question in group B carries 2 marks, and each question in group C carries 3 marks. 
It is known that:
- Group B contains 23 questions
- Group C contains 1 question.
Prove that the percentage of the total marks that the questions in group A carry is 60.8%.
-/

theorem group_A_percentage :
  ∃ (a b c : ℕ), b = 23 ∧ c = 1 ∧ (a + b + c = 100) ∧ ((a * 1) + (b * 2) + (c * 3) = 125) ∧ ((a : ℝ) / 125 * 100 = 60.8) :=
by
  sorry

end NUMINAMATH_GPT_group_A_percentage_l699_69933


namespace NUMINAMATH_GPT_jackie_phil_probability_l699_69913

noncomputable def probability_same_heads : ℚ :=
  let fair_coin := (1 + 1: ℚ)
  let p3_coin := (2 + 3: ℚ)
  let p2_coin := (1 + 2: ℚ)
  let generating_function := fair_coin * p3_coin * p2_coin
  let sum_of_coefficients := 30
  let sum_of_squares_of_coefficients := 290
  sum_of_squares_of_coefficients / (sum_of_coefficients ^ 2)

theorem jackie_phil_probability : probability_same_heads = 29 / 90 := by
  sorry

end NUMINAMATH_GPT_jackie_phil_probability_l699_69913


namespace NUMINAMATH_GPT_correct_statement_l699_69979

-- Definitions
def certain_event (P : ℝ → Prop) : Prop := P 1
def impossible_event (P : ℝ → Prop) : Prop := P 0
def uncertain_event (P : ℝ → Prop) : Prop := ∀ p, 0 < p ∧ p < 1 → P p

-- Theorem to prove
theorem correct_statement (P : ℝ → Prop) :
  (certain_event P ∧ impossible_event P ∧ uncertain_event P) →
  (∀ p, P p → p = 1)
:= by
  sorry

end NUMINAMATH_GPT_correct_statement_l699_69979


namespace NUMINAMATH_GPT_sum_of_first_11_odd_numbers_l699_69991

theorem sum_of_first_11_odd_numbers : 
  (1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19 + 21) = 121 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_11_odd_numbers_l699_69991


namespace NUMINAMATH_GPT_fifty_percent_of_number_l699_69957

-- Define the given condition
def given_condition (x : ℝ) : Prop :=
  0.6 * x = 42

-- Define the statement we need to prove
theorem fifty_percent_of_number (x : ℝ) (h : given_condition x) : 0.5 * x = 35 := by
  sorry

end NUMINAMATH_GPT_fifty_percent_of_number_l699_69957


namespace NUMINAMATH_GPT_distinct_non_zero_reals_square_rational_l699_69950

theorem distinct_non_zero_reals_square_rational
  {a : Fin 10 → ℝ}
  (distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (non_zero : ∀ i, a i ≠ 0)
  (rational_condition : ∀ i j, ∃ (q : ℚ), a i + a j = q ∨ a i * a j = q) :
  ∀ i, ∃ (q : ℚ), (a i)^2 = q :=
by
  sorry

end NUMINAMATH_GPT_distinct_non_zero_reals_square_rational_l699_69950


namespace NUMINAMATH_GPT_canoes_more_than_kayaks_l699_69916

noncomputable def canoes_difference (C K : ℕ) : Prop :=
  15 * C + 18 * K = 405 ∧ 2 * C = 3 * K → C - K = 5

theorem canoes_more_than_kayaks (C K : ℕ) : canoes_difference C K :=
by
  sorry

end NUMINAMATH_GPT_canoes_more_than_kayaks_l699_69916


namespace NUMINAMATH_GPT_prob_not_green_is_six_over_eleven_l699_69902

-- Define the odds for pulling a green marble
def odds_green : ℕ × ℕ := (5, 6)

-- Define the total number of events as the sum of both parts of the odds
def total_events : ℕ := odds_green.1 + odds_green.2

-- Define the probability of not pulling a green marble
def probability_not_green : ℚ := odds_green.2 / total_events

-- State the theorem
theorem prob_not_green_is_six_over_eleven : probability_not_green = 6 / 11 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_prob_not_green_is_six_over_eleven_l699_69902


namespace NUMINAMATH_GPT_original_apples_l699_69923

-- Define the conditions using the given data
def sells_fraction : ℝ := 0.40 -- Fraction of apples sold
def remaining_apples : ℝ := 420 -- Apples remaining after selling

-- Theorem statement for proving the original number of apples given the conditions
theorem original_apples (x : ℝ) (sells_fraction : ℝ := 0.40) (remaining_apples : ℝ := 420) : 
  420 / (1 - sells_fraction) = x :=
sorry

end NUMINAMATH_GPT_original_apples_l699_69923


namespace NUMINAMATH_GPT_jack_reads_books_in_a_year_l699_69973

/-- If Jack reads 9 books per day, how many books can he read in a year (365 days)? -/
theorem jack_reads_books_in_a_year (books_per_day : ℕ) (days_per_year : ℕ) (books_per_year : ℕ) (h1 : books_per_day = 9) (h2 : days_per_year = 365) : books_per_year = 3285 :=
by
  sorry

end NUMINAMATH_GPT_jack_reads_books_in_a_year_l699_69973


namespace NUMINAMATH_GPT_change_in_opinion_difference_l699_69985

theorem change_in_opinion_difference :
  let initially_liked_pct := 0.4;
  let initially_disliked_pct := 0.6;
  let finally_liked_pct := 0.8;
  let finally_disliked_pct := 0.2;
  let max_change := finally_liked_pct + (initially_disliked_pct - finally_disliked_pct);
  let min_change := finally_liked_pct - initially_liked_pct;
  max_change - min_change = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_change_in_opinion_difference_l699_69985


namespace NUMINAMATH_GPT_joan_first_payment_l699_69977

theorem joan_first_payment (P : ℝ) 
  (total_amount : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (h_total : total_amount = 109300)
  (h_r : r = 3)
  (h_n : n = 7)
  (h_sum : total_amount = P * (1 - r^n) / (1 - r)) : 
  P = 100 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_joan_first_payment_l699_69977


namespace NUMINAMATH_GPT_parabola_focus_directrix_distance_l699_69901

theorem parabola_focus_directrix_distance :
  ∀ {x y : ℝ}, y^2 = (1/4) * x → dist (1/16, 0) (-1/16, 0) = 1/8 := by
sorry

end NUMINAMATH_GPT_parabola_focus_directrix_distance_l699_69901


namespace NUMINAMATH_GPT_product_last_digit_l699_69934

def last_digit (n : ℕ) : ℕ := n % 10

theorem product_last_digit :
  last_digit (3^65 * 6^59 * 7^71) = 4 :=
by
  sorry

end NUMINAMATH_GPT_product_last_digit_l699_69934


namespace NUMINAMATH_GPT_cubic_sum_identity_l699_69937

theorem cubic_sum_identity (x y z : ℝ) (h1 : x + y + z = 15) (h2 : xy + yz + zx = 34) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 1845 :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_identity_l699_69937


namespace NUMINAMATH_GPT_factorial_power_of_two_iff_power_of_two_l699_69983

-- Assuming n is a positive integer
variable {n : ℕ} (h : n > 0)

theorem factorial_power_of_two_iff_power_of_two :
  (∃ k : ℕ, n = 2^k ) ↔ ∃ m : ℕ, 2^(n-1) ∣ n! :=
by {
  sorry
}

end NUMINAMATH_GPT_factorial_power_of_two_iff_power_of_two_l699_69983


namespace NUMINAMATH_GPT_calculator_unit_prices_and_min_cost_l699_69988

-- Definitions for conditions
def unit_price_type_A (x : ℕ) : Prop :=
  ∀ y : ℕ, (y = x + 10) → (550 / x = 600 / y)

def purchase_constraint (a : ℕ) : Prop :=
  25 ≤ a ∧ a ≤ 100

def total_cost (a : ℕ) (x y : ℕ) : ℕ :=
  110 * a + 120 * (100 - a)

-- Statement to prove
theorem calculator_unit_prices_and_min_cost :
  ∃ x y, unit_price_type_A x ∧ unit_price_type_A x ∧ total_cost 100 x y = 11000 :=
by
  sorry

end NUMINAMATH_GPT_calculator_unit_prices_and_min_cost_l699_69988


namespace NUMINAMATH_GPT_albums_created_l699_69976

def phone_pics : ℕ := 2
def camera_pics : ℕ := 4
def pics_per_album : ℕ := 2
def total_pics : ℕ := phone_pics + camera_pics

theorem albums_created : total_pics / pics_per_album = 3 := by
  sorry

end NUMINAMATH_GPT_albums_created_l699_69976


namespace NUMINAMATH_GPT_tomatoes_on_each_plant_l699_69960

/-- Andy harvests all the tomatoes from 18 plants that have a certain number of tomatoes each.
    He dries half the tomatoes and turns a third of the remainder into marinara sauce. He has
    42 tomatoes left. Prove that the number of tomatoes on each plant is 7.  -/
theorem tomatoes_on_each_plant (T : ℕ) (h1 : ∀ n, n = 18 * T)
  (h2 : ∀ m, m = (18 * T) / 2)
  (h3 : ∀ k, k = m / 3)
  (h4 : ∀ final, final = m - k ∧ final = 42) : T = 7 :=
by
  sorry

end NUMINAMATH_GPT_tomatoes_on_each_plant_l699_69960


namespace NUMINAMATH_GPT_rate_of_mixed_oil_l699_69978

/-- If 10 litres of an oil at Rs. 50 per litre is mixed with 5 litres of another oil at Rs. 67 per litre,
    then the rate of the mixed oil per litre is Rs. 55.67. --/
theorem rate_of_mixed_oil : 
  let volume1 := 10
  let price1 := 50
  let volume2 := 5
  let price2 := 67
  let total_cost := (volume1 * price1) + (volume2 * price2)
  let total_volume := volume1 + volume2
  (total_cost / total_volume : ℝ) = 55.67 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_mixed_oil_l699_69978


namespace NUMINAMATH_GPT_Brandy_caffeine_intake_l699_69993

theorem Brandy_caffeine_intake :
  let weight := 60
  let recommended_limit_per_kg := 2.5
  let tolerance := 50
  let coffee_cups := 2
  let coffee_per_cup := 95
  let energy_drinks := 4
  let caffeine_per_energy_drink := 120
  let max_safe_caffeine := weight * recommended_limit_per_kg + tolerance
  let caffeine_from_coffee := coffee_cups * coffee_per_cup
  let caffeine_from_energy_drinks := energy_drinks * caffeine_per_energy_drink
  let total_caffeine_consumed := caffeine_from_coffee + caffeine_from_energy_drinks
  max_safe_caffeine - total_caffeine_consumed = -470 := 
by
  sorry

end NUMINAMATH_GPT_Brandy_caffeine_intake_l699_69993


namespace NUMINAMATH_GPT_smallest_positive_integer_n_l699_69972

theorem smallest_positive_integer_n :
  ∃ (n: ℕ), n = 4 ∧ (∀ x: ℝ, (Real.sin x)^n + (Real.cos x)^n ≤ 2 / n) :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_l699_69972


namespace NUMINAMATH_GPT_total_goals_in_league_l699_69994

variables (g1 g2 T : ℕ)

-- Conditions
def equal_goals : Prop := g1 = g2
def players_goals : Prop := g1 = 30
def total_goals_percentage : Prop := (g1 + g2) * 5 = T

-- Theorem to prove: Given the conditions, the total number of goals T should be 300
theorem total_goals_in_league (h1 : equal_goals g1 g2) (h2 : players_goals g1) (h3 : total_goals_percentage g1 g2 T) : T = 300 :=
sorry

end NUMINAMATH_GPT_total_goals_in_league_l699_69994


namespace NUMINAMATH_GPT_range_of_f_l699_69914

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) * (Real.cos x) + (Real.cos x) ^ 4

theorem range_of_f : Set.Icc 0 (9 / 8) = Set.range f := 
by
  sorry

end NUMINAMATH_GPT_range_of_f_l699_69914


namespace NUMINAMATH_GPT_solution_set_of_inequality_l699_69931

theorem solution_set_of_inequality :
  {x : ℝ | 2 ≥ 1 / (x - 1)} = {x : ℝ | x < 1} ∪ {x : ℝ | x ≥ 3 / 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l699_69931


namespace NUMINAMATH_GPT_remainder_of_144_div_k_l699_69995

theorem remainder_of_144_div_k
  (k : ℕ)
  (h1 : 0 < k)
  (h2 : 120 % k^2 = 12) :
  144 % k = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_144_div_k_l699_69995


namespace NUMINAMATH_GPT_polynomial_has_roots_l699_69944

-- Define the polynomial
def polynomial (x : ℂ) : ℂ := 7 * x^4 - 48 * x^3 + 93 * x^2 - 48 * x + 7

-- Theorem to prove the existence of roots for the polynomial equation
theorem polynomial_has_roots : ∃ x : ℂ, polynomial x = 0 := by
  sorry

end NUMINAMATH_GPT_polynomial_has_roots_l699_69944


namespace NUMINAMATH_GPT_alcohol_water_ratio_l699_69981

theorem alcohol_water_ratio
  (V p q : ℝ)
  (hV : V > 0)
  (hp : p > 0)
  (hq : q > 0) :
  let total_alcohol := 3 * V * (p / (p + 1)) + V * (q / (q + 1))
  let total_water := 3 * V * (1 / (p + 1)) + V * (1 / (q + 1))
  total_alcohol / total_water = (3 * p * (q + 1) + q * (p + 1)) / (3 * (q + 1) + (p + 1)) :=
sorry

end NUMINAMATH_GPT_alcohol_water_ratio_l699_69981


namespace NUMINAMATH_GPT_number_of_red_pencils_l699_69929

theorem number_of_red_pencils (B R G : ℕ) (h1 : B + R + G = 20) (h2 : B = 6 * G) (h3 : R < B) : R = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_red_pencils_l699_69929


namespace NUMINAMATH_GPT_math_proof_problem_l699_69915

theorem math_proof_problem (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3 * x₁ * y₁^2 = 2008)
  (h₂ : y₁^3 - 3 * x₁^2 * y₁ = 2007)
  (h₃ : x₂^3 - 3 * x₂ * y₂^2 = 2008)
  (h₄ : y₂^3 - 3 * x₂^2 * y₂ = 2007)
  (h₅ : x₃^3 - 3 * x₃ * y₃^2 = 2008)
  (h₆ : y₃^3 - 3 * x₃^2 * y₃ = 2007) :
  (1 - x₁ / y₁) * (1 - x₂ / y₂) * (1 - x₃ / y₃) = 4015 / 2008 :=
by sorry

end NUMINAMATH_GPT_math_proof_problem_l699_69915


namespace NUMINAMATH_GPT_exponential_growth_equation_l699_69946

-- Define the initial and final greening areas and the years in consideration.
def initial_area : ℝ := 1000
def final_area : ℝ := 1440
def years : ℝ := 2

-- Define the average annual growth rate.
variable (x : ℝ)

-- State the theorem about the exponential growth equation.
theorem exponential_growth_equation :
  initial_area * (1 + x) ^ years = final_area :=
sorry

end NUMINAMATH_GPT_exponential_growth_equation_l699_69946


namespace NUMINAMATH_GPT_area_T_is_34_l699_69974

/-- Define the dimensions of the large rectangle -/
def width_rect : ℕ := 10
def height_rect : ℕ := 4
/-- Define the dimensions of the removed section -/
def width_removed : ℕ := 6
def height_removed : ℕ := 1

/-- Calculate the area of the large rectangle -/
def area_rect : ℕ := width_rect * height_rect

/-- Calculate the area of the removed section -/
def area_removed : ℕ := width_removed * height_removed

/-- Calculate the area of the "T" shape -/
def area_T : ℕ := area_rect - area_removed

/-- To prove that the area of the T-shape is 34 square units -/
theorem area_T_is_34 : area_T = 34 := 
by {
  sorry
}

end NUMINAMATH_GPT_area_T_is_34_l699_69974


namespace NUMINAMATH_GPT_simplify_sqrt_expression_correct_l699_69926

noncomputable def simplify_sqrt_expression (m : ℝ) (h_triangle : (2 < m + 5) ∧ (m < 2 + 5) ∧ (5 < 2 + m)) : ℝ :=
  (Real.sqrt (9 - 6 * m + m^2)) - (Real.sqrt (m^2 - 14 * m + 49))

theorem simplify_sqrt_expression_correct (m : ℝ) (h_triangle : (2 < m + 5) ∧ (m < 2 + 5) ∧ (5 < 2 + m)) :
  simplify_sqrt_expression m h_triangle = 2 * m - 10 :=
sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_correct_l699_69926


namespace NUMINAMATH_GPT_proof_problem_l699_69930

variable {a b c : ℝ}
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variable (h4 : (a+1) * (b+1) * (c+1) = 8)

theorem proof_problem :
  a + b + c ≥ 3 ∧ abc ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l699_69930


namespace NUMINAMATH_GPT_pairwise_products_same_digit_l699_69918

theorem pairwise_products_same_digit
  (a b c : ℕ)
  (h_ab : a % 10 ≠ b % 10)
  (h_ac : a % 10 ≠ c % 10)
  (h_bc : b % 10 ≠ c % 10)
  : (a * b % 10 = a * c % 10) ∧ (a * b % 10 = b * c % 10) :=
  sorry

end NUMINAMATH_GPT_pairwise_products_same_digit_l699_69918


namespace NUMINAMATH_GPT_probability_of_adjacent_vertices_in_decagon_l699_69997

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end NUMINAMATH_GPT_probability_of_adjacent_vertices_in_decagon_l699_69997


namespace NUMINAMATH_GPT_chip_cost_l699_69958

theorem chip_cost 
  (calories_per_chip : ℕ)
  (chips_per_bag : ℕ)
  (cost_per_bag : ℕ)
  (desired_calories : ℕ)
  (h1 : calories_per_chip = 10)
  (h2 : chips_per_bag = 24)
  (h3 : cost_per_bag = 2)
  (h4 : desired_calories = 480) : 
  cost_per_bag * (desired_calories / (calories_per_chip * chips_per_bag)) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_chip_cost_l699_69958


namespace NUMINAMATH_GPT_range_of_m_l699_69989

noncomputable def distance (m : ℝ) : ℝ := (|m| * Real.sqrt 2 / 2)
theorem range_of_m (m : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A.1 + A.2 + m = 0 ∧ B.1 + B.2 + m = 0) ∧
    (A.1 ^ 2 + A.2 ^ 2 = 2 ∧ B.1 ^ 2 + B.2 ^ 2 = 2) ∧
    (Real.sqrt (A.1 ^ 2 + A.2 ^ 2) + Real.sqrt (B.1 ^ 2 + B.2 ^ 2) ≥ 
     Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)) ∧ (distance m < Real.sqrt 2)) ↔ 
  m ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l699_69989


namespace NUMINAMATH_GPT_bus_speed_l699_69922

theorem bus_speed (S : ℝ) (h1 : 36 = S * (2 / 3)) : S = 54 :=
by
sorry

end NUMINAMATH_GPT_bus_speed_l699_69922


namespace NUMINAMATH_GPT_smallest_n_common_factor_l699_69927

theorem smallest_n_common_factor :
  ∃ n : ℤ, n > 0 ∧ (gcd (8 * n - 3) (5 * n + 4) > 1) ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_common_factor_l699_69927


namespace NUMINAMATH_GPT_compare_neg_one_neg_sqrt_two_l699_69925

theorem compare_neg_one_neg_sqrt_two : -1 > -Real.sqrt 2 :=
  by
    sorry

end NUMINAMATH_GPT_compare_neg_one_neg_sqrt_two_l699_69925


namespace NUMINAMATH_GPT_find_number_l699_69906

-- Define the conditions
def condition (x : ℝ) : Prop := 0.65 * x = (4/5) * x - 21

-- Prove that given the condition, x is 140.
theorem find_number (x : ℝ) (h : condition x) : x = 140 := by
  sorry

end NUMINAMATH_GPT_find_number_l699_69906


namespace NUMINAMATH_GPT_count_p_shape_points_l699_69984

-- Define the problem conditions
def side_length : ℕ := 10
def point_interval : ℕ := 1
def num_sides : ℕ := 3
def correction_corners : ℕ := 2

-- Define the total expected points
def total_expected_points : ℕ := 31

-- Proof statement
theorem count_p_shape_points :
  ((side_length / point_interval + 1) * num_sides - correction_corners) = total_expected_points := by
  sorry

end NUMINAMATH_GPT_count_p_shape_points_l699_69984


namespace NUMINAMATH_GPT_lecture_hall_rows_l699_69919

-- We define the total number of seats
def total_seats (n : ℕ) : ℕ := n * (n + 11)

-- We state the problem with the given conditions
theorem lecture_hall_rows : 
  (400 ≤ total_seats n) ∧ (total_seats n ≤ 440) → n = 16 :=
by
  sorry

end NUMINAMATH_GPT_lecture_hall_rows_l699_69919


namespace NUMINAMATH_GPT_evaluate_polynomial_l699_69990

theorem evaluate_polynomial : (99^4 - 4 * 99^3 + 6 * 99^2 - 4 * 99 + 1) = 92199816 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_l699_69990


namespace NUMINAMATH_GPT_smallest_positive_period_symmetry_axis_not_even_function_decreasing_interval_l699_69955

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (4 * Real.pi / 3))

theorem smallest_positive_period (T : ℝ) : T = Real.pi ↔ (∀ x : ℝ, f (x + T) = f x) := by
  sorry

theorem symmetry_axis (x : ℝ) : x = (7 * Real.pi / 12) ↔ (∀ y : ℝ, f (2 * x - y) = f y) := by
  sorry

theorem not_even_function : ¬ (∀ x : ℝ, f (x + (Real.pi / 3)) = f (-x - (Real.pi / 3))) := by
  sorry

theorem decreasing_interval (k : ℤ) (x : ℝ) : (k * Real.pi - (5 * Real.pi / 12) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 12)) ↔ (∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2) := by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_symmetry_axis_not_even_function_decreasing_interval_l699_69955


namespace NUMINAMATH_GPT_fractions_are_integers_l699_69932

theorem fractions_are_integers (a b c : ℤ) (h : ∃ k : ℤ, (a * b / c) + (a * c / b) + (b * c / a) = k) :
  ∃ k1 k2 k3 : ℤ, (a * b / c) = k1 ∧ (a * c / b) = k2 ∧ (b * c / a) = k3 :=
by
  sorry

end NUMINAMATH_GPT_fractions_are_integers_l699_69932


namespace NUMINAMATH_GPT_living_room_size_is_96_l699_69920

-- Define the total area of the apartment
def total_area : ℕ := 16 * 10

-- Define the number of units
def units : ℕ := 5

-- Define the size of one unit
def size_of_one_unit : ℕ := total_area / units

-- Define the size of the living room
def living_room_size : ℕ := size_of_one_unit * 3

-- Proving that the living room size is indeed 96 square feet
theorem living_room_size_is_96 : living_room_size = 96 := 
by
  -- not providing proof, thus using sorry
  sorry

end NUMINAMATH_GPT_living_room_size_is_96_l699_69920


namespace NUMINAMATH_GPT_value_of_x_l699_69970

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l699_69970


namespace NUMINAMATH_GPT_number_of_tiles_is_47_l699_69900

theorem number_of_tiles_is_47 : 
  ∃ (n : ℕ), (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 5 = 2) ∧ n = 47 :=
by
  sorry

end NUMINAMATH_GPT_number_of_tiles_is_47_l699_69900


namespace NUMINAMATH_GPT_total_cookies_l699_69996

theorem total_cookies (chris kenny glenn : ℕ) 
  (h1 : chris = kenny / 2)
  (h2 : glenn = 4 * kenny)
  (h3 : glenn = 24) : 
  chris + kenny + glenn = 33 := 
by
  -- Focusing on defining the theorem statement correct without entering the proof steps.
  sorry

end NUMINAMATH_GPT_total_cookies_l699_69996


namespace NUMINAMATH_GPT_peter_pizza_total_l699_69966

theorem peter_pizza_total (total_slices : ℕ) (whole_slice : ℕ) (shared_slice : ℚ) (shared_parts : ℕ) :
  total_slices = 16 ∧ whole_slice = 1 ∧ shared_parts = 3 ∧ shared_slice = 1 / (total_slices * shared_parts) →
  whole_slice / total_slices + shared_slice = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_peter_pizza_total_l699_69966


namespace NUMINAMATH_GPT_cadence_worked_old_company_l699_69907

theorem cadence_worked_old_company (y : ℕ) (h1 : (426000 : ℝ) = 
    5000 * 12 * y + 6000 * 12 * (y + 5 / 12)) :
    y = 3 := by
    sorry

end NUMINAMATH_GPT_cadence_worked_old_company_l699_69907


namespace NUMINAMATH_GPT_value_of_f_g_3_l699_69903

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x^2 - 2*x + 1

theorem value_of_f_g_3 : f (g 3) = 2134 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_f_g_3_l699_69903


namespace NUMINAMATH_GPT_six_lines_regions_l699_69959

def number_of_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1) / 2)

theorem six_lines_regions (h1 : 6 > 0) : 
    number_of_regions 6 = 22 :=
by 
  -- Use the formula for calculating number of regions:
  -- number_of_regions n = 1 + n + (n * (n - 1) / 2)
  sorry

end NUMINAMATH_GPT_six_lines_regions_l699_69959


namespace NUMINAMATH_GPT_range_of_a_l699_69965

-- Assuming all necessary imports and definitions are included

variable {R : Type} [LinearOrderedField R]

def satisfies_conditions (f : R → R) (a : R) : Prop :=
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∀ x y, 1 ≤ x → x < y → f x < f y) ∧
  (∀ x, (1/2 : R) ≤ x ∧ x ≤ 1 → f (a * x) < f (x - 1))

theorem range_of_a (f : R → R) (a : R) :
  satisfies_conditions f a → 0 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l699_69965


namespace NUMINAMATH_GPT_find_ratio_l699_69909

open Real

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (q : ℝ)

-- The geometric sequence conditions
def geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q

-- Sum of the first n terms for the geometric sequence
def sum_of_first_n_terms := ∀ n : ℕ, S n = (a 0) * (1 - q ^ n) / (1 - q)

-- Given conditions
def given_conditions :=
  a 0 + a 2 = 5 / 2 ∧
  a 1 + a 3 = 5 / 4

-- The goal to prove
theorem find_ratio (geo_seq : geometric_sequence a q) (sum_terms : sum_of_first_n_terms a S q) (cond : given_conditions a) :
  S 4 / a 4 = 31 :=
  sorry

end NUMINAMATH_GPT_find_ratio_l699_69909


namespace NUMINAMATH_GPT_find_x_for_parallel_vectors_l699_69964

-- Definitions for the given conditions
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- The proof statement
theorem find_x_for_parallel_vectors (x : ℝ) (h : parallel a (b x)) : x = 6 :=
  sorry

end NUMINAMATH_GPT_find_x_for_parallel_vectors_l699_69964


namespace NUMINAMATH_GPT_factorial_subtraction_l699_69949

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_subtraction : factorial 10 - factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_GPT_factorial_subtraction_l699_69949


namespace NUMINAMATH_GPT_total_sum_money_l699_69948

theorem total_sum_money (a b c : ℝ) (h1 : b = 0.65 * a) (h2 : c = 0.40 * a) (h3 : c = 64) :
  a + b + c = 328 :=
by
  sorry

end NUMINAMATH_GPT_total_sum_money_l699_69948


namespace NUMINAMATH_GPT_sin_180_eq_0_l699_69935

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sin_180_eq_0_l699_69935


namespace NUMINAMATH_GPT_inequality_example_l699_69953

theorem inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 = 3) : 
    1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + a * c) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_example_l699_69953


namespace NUMINAMATH_GPT_set_intersection_l699_69963

open Set

variable (x : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := { x | |x - 1| > 2 }
def B : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }

theorem set_intersection (x : ℝ) : x ∈ (U \ A) ∩ B ↔ 2 < x ∧ x ≤ 3 := sorry

end NUMINAMATH_GPT_set_intersection_l699_69963


namespace NUMINAMATH_GPT_back_wheel_revolutions_l699_69942

theorem back_wheel_revolutions
  (front_diameter : ℝ) (back_diameter : ℝ) (front_revolutions : ℝ) (back_revolutions : ℝ)
  (front_diameter_eq : front_diameter = 28)
  (back_diameter_eq : back_diameter = 20)
  (front_revolutions_eq : front_revolutions = 50)
  (distance_eq : ∀ {d₁ d₂}, 2 * Real.pi * d₁ / 2 * front_revolutions = back_revolutions * (2 * Real.pi * d₂ / 2)) :
  back_revolutions = 70 :=
by
  have front_circumference : ℝ := 2 * Real.pi * front_diameter / 2
  have back_circumference : ℝ := 2 * Real.pi * back_diameter / 2
  have total_distance : ℝ := front_circumference * front_revolutions
  have revolutions : ℝ := total_distance / back_circumference 
  sorry

end NUMINAMATH_GPT_back_wheel_revolutions_l699_69942


namespace NUMINAMATH_GPT_original_costs_l699_69924

theorem original_costs (P_old P_second_oldest : ℝ) (h1 : 0.9 * P_old = 1800) (h2 : 0.85 * P_second_oldest = 900) :
  P_old + P_second_oldest = 3058.82 :=
by sorry

end NUMINAMATH_GPT_original_costs_l699_69924


namespace NUMINAMATH_GPT_range_of_a_l699_69971

theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a = 0) → a < 5 := 
by sorry

end NUMINAMATH_GPT_range_of_a_l699_69971


namespace NUMINAMATH_GPT_range_of_m_l699_69945

open Classical

variable {m : ℝ}

def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * x + m ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, (3 - m) > 1 → ((3 - m) ^ x > 0)

theorem range_of_m (hm : (p m ∨ q m) ∧ ¬(p m ∧ q m)) : 1 < m ∧ m < 2 :=
  sorry

end NUMINAMATH_GPT_range_of_m_l699_69945


namespace NUMINAMATH_GPT_altered_solution_ratio_l699_69928

theorem altered_solution_ratio (initial_bleach : ℕ) (initial_detergent : ℕ) (initial_water : ℕ) :
  initial_bleach / initial_detergent = 2 / 25 ∧
  initial_detergent / initial_water = 25 / 100 →
  (initial_detergent / initial_water) / 2 = 1 / 8 →
  initial_water = 300 →
  (300 / 8) = 37.5 := 
by 
  sorry

end NUMINAMATH_GPT_altered_solution_ratio_l699_69928


namespace NUMINAMATH_GPT_area_in_square_yards_l699_69980

/-
  Given:
  - length of the classroom in feet
  - width of the classroom in feet

  Prove that the area required to cover the classroom in square yards is 30. 
-/

def classroom_length_feet : ℕ := 15
def classroom_width_feet : ℕ := 18
def feet_to_yard (feet : ℕ) : ℕ := feet / 3

theorem area_in_square_yards :
  let length_yards := feet_to_yard classroom_length_feet
  let width_yards := feet_to_yard classroom_width_feet
  length_yards * width_yards = 30 :=
by
  sorry

end NUMINAMATH_GPT_area_in_square_yards_l699_69980


namespace NUMINAMATH_GPT_probability_student_less_than_25_l699_69952

def total_students : ℕ := 100

-- Percentage conditions translated to proportions
def proportion_male : ℚ := 0.48
def proportion_female : ℚ := 0.52

def proportion_male_25_or_older : ℚ := 0.50
def proportion_female_25_or_older : ℚ := 0.20

-- Definition of probability that a randomly selected student is less than 25 years old.
def probability_less_than_25 : ℚ :=
  (proportion_male * (1 - proportion_male_25_or_older)) +
  (proportion_female * (1 - proportion_female_25_or_older))

theorem probability_student_less_than_25 :
  probability_less_than_25 = 0.656 := by
  sorry

end NUMINAMATH_GPT_probability_student_less_than_25_l699_69952


namespace NUMINAMATH_GPT_at_least_one_non_zero_l699_69911

theorem at_least_one_non_zero (a b : ℝ) : a^2 + b^2 > 0 ↔ (a ≠ 0 ∨ b ≠ 0) :=
by sorry

end NUMINAMATH_GPT_at_least_one_non_zero_l699_69911


namespace NUMINAMATH_GPT_cost_per_game_l699_69947

theorem cost_per_game 
  (x : ℝ)
  (shoe_rent : ℝ := 0.50)
  (total_money : ℝ := 12.80)
  (games : ℕ := 7)
  (h1 : total_money - shoe_rent = 12.30)
  (h2 : 7 * x = 12.30) :
  x = 1.76 := 
sorry

end NUMINAMATH_GPT_cost_per_game_l699_69947


namespace NUMINAMATH_GPT_solve_z_for_complex_eq_l699_69943

theorem solve_z_for_complex_eq (i : ℂ) (h : i^2 = -1) : ∀ (z : ℂ), 3 - 2 * i * z = -4 + 5 * i * z → z = -i :=
by
  intro z
  intro eqn
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_solve_z_for_complex_eq_l699_69943


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l699_69941

variables (a b : Int)

theorem simplify_and_evaluate_expr (ha : a = 1) (hb : b = -2) : 
  2 * (3 * a^2 * b - a * b^2) - 3 * (-a * b^2 + a^2 * b - 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l699_69941


namespace NUMINAMATH_GPT_circle_chords_integer_lengths_l699_69910

theorem circle_chords_integer_lengths (P O : ℝ) (d r : ℝ) (n : ℕ) : 
  dist P O = d → r = 20 → d = 12 → n = 9 := by
  sorry

end NUMINAMATH_GPT_circle_chords_integer_lengths_l699_69910


namespace NUMINAMATH_GPT_roots_poly_eval_l699_69982

theorem roots_poly_eval : ∀ (c d : ℝ), (c + d = 6 ∧ c * d = 8) → c^4 + c^3 * d + d^3 * c + d^4 = 432 :=
by
  intros c d h
  sorry

end NUMINAMATH_GPT_roots_poly_eval_l699_69982


namespace NUMINAMATH_GPT_mul_inv_800_mod_7801_l699_69940

theorem mul_inv_800_mod_7801 :
  ∃ x : ℕ, 0 ≤ x ∧ x < 7801 ∧ (800 * x) % 7801 = 1 := by
  use 3125
  dsimp
  norm_num1
  sorry

end NUMINAMATH_GPT_mul_inv_800_mod_7801_l699_69940


namespace NUMINAMATH_GPT_final_segment_position_correct_l699_69938

def initial_segment : ℝ × ℝ := (1, 6)
def rotate_180_about (p : ℝ) (x : ℝ) : ℝ := p - (x - p)
def first_rotation_segment : ℝ × ℝ := (rotate_180_about 2 6, rotate_180_about 2 1)
def second_rotation_segment : ℝ × ℝ := (rotate_180_about 1 3, rotate_180_about 1 (-2))

theorem final_segment_position_correct :
  second_rotation_segment = (-1, 4) :=
by
  -- This is a placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_final_segment_position_correct_l699_69938


namespace NUMINAMATH_GPT_correct_calculated_value_l699_69986

theorem correct_calculated_value (n : ℕ) (h1 : n = 32 * 3) : n / 4 = 24 := 
by
  -- proof steps will be filled here
  sorry

end NUMINAMATH_GPT_correct_calculated_value_l699_69986
