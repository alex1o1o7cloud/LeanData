import Mathlib

namespace NUMINAMATH_GPT_fold_paper_crease_length_l433_43381

theorem fold_paper_crease_length 
    (w l : ℝ) (w_pos : w = 12) (l_pos : l = 16) 
    (F G : ℝ × ℝ) (F_on_AD : F = (0, 12))
    (G_on_BC : G = (16, 12)) :
    dist F G = 20 := 
by
  sorry

end NUMINAMATH_GPT_fold_paper_crease_length_l433_43381


namespace NUMINAMATH_GPT_bus_trip_distance_l433_43345

variable (D : ℝ) (S : ℝ := 50)

theorem bus_trip_distance :
  (D / (S + 5) = D / S - 1) → D = 550 := by
  sorry

end NUMINAMATH_GPT_bus_trip_distance_l433_43345


namespace NUMINAMATH_GPT_final_passenger_count_l433_43366

def total_passengers (initial : ℕ) (first_stop : ℕ) (off_bus : ℕ) (on_bus : ℕ) : ℕ :=
  (initial + first_stop) - off_bus + on_bus

theorem final_passenger_count :
  total_passengers 50 16 22 5 = 49 := by
  sorry

end NUMINAMATH_GPT_final_passenger_count_l433_43366


namespace NUMINAMATH_GPT_clubsuit_problem_l433_43360

def clubsuit (x y : ℤ) : ℤ :=
  (x^2 + y^2) * (x - y)

theorem clubsuit_problem : clubsuit 2 (clubsuit 3 4) = 16983 := 
by 
  sorry

end NUMINAMATH_GPT_clubsuit_problem_l433_43360


namespace NUMINAMATH_GPT_sum_powers_seventh_l433_43369

/-- Given the sequence values for sums of powers of 'a' and 'b', prove the value of the sum of the 7th powers. -/
theorem sum_powers_seventh (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^7 + b^7 = 29 := 
  sorry

end NUMINAMATH_GPT_sum_powers_seventh_l433_43369


namespace NUMINAMATH_GPT_largest_expression_is_A_l433_43352

def expr_A := 1 - 2 + 3 + 4
def expr_B := 1 + 2 - 3 + 4
def expr_C := 1 + 2 + 3 - 4
def expr_D := 1 + 2 - 3 - 4
def expr_E := 1 - 2 - 3 + 4

theorem largest_expression_is_A : expr_A = 6 ∧ expr_A > expr_B ∧ expr_A > expr_C ∧ expr_A > expr_D ∧ expr_A > expr_E :=
  by sorry

end NUMINAMATH_GPT_largest_expression_is_A_l433_43352


namespace NUMINAMATH_GPT_parabola_translation_l433_43357

theorem parabola_translation :
  ∀ x : ℝ, (x^2 + 3) = ((x + 1)^2 + 3) :=
by
  skip -- proof is not needed; this is just the statement according to the instruction
  sorry

end NUMINAMATH_GPT_parabola_translation_l433_43357


namespace NUMINAMATH_GPT_ratio_area_circle_to_triangle_l433_43331

theorem ratio_area_circle_to_triangle (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
    (π * r) / (h + r) = (π * r ^ 2) / (r * (h + r)) := sorry

end NUMINAMATH_GPT_ratio_area_circle_to_triangle_l433_43331


namespace NUMINAMATH_GPT_find_square_value_l433_43324

theorem find_square_value (y : ℝ) (h : 4 * y^2 + 3 = 7 * y + 12) : (8 * y - 4)^2 = 202 := 
by
  sorry

end NUMINAMATH_GPT_find_square_value_l433_43324


namespace NUMINAMATH_GPT_equivalent_expression_l433_43336

def evaluate_expression : ℚ :=
  let part1 := (2/3) * ((35/100) * 250)
  let part2 := ((75/100) * 150) / 16
  let part3 := (1/2) * ((40/100) * 500)
  part1 - part2 + part3

theorem equivalent_expression :
  evaluate_expression = 151.3020833333 :=  
by 
  sorry

end NUMINAMATH_GPT_equivalent_expression_l433_43336


namespace NUMINAMATH_GPT_expression_simplification_l433_43370

theorem expression_simplification (a : ℝ) (h : a ≠ 1) (h_beta : 1 = 1):
  (2^(Real.log (a) / Real.log (Real.sqrt 2)) - 
   3^((Real.log (a^2+1)) / (Real.log 27)) - 
   2 * a) / 
  (7^(4 * (Real.log (a) / Real.log 49)) - 
   5^((0.5 * Real.log (a)) / (Real.log (Real.sqrt 5))) - 1) = a^2 + a + 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplification_l433_43370


namespace NUMINAMATH_GPT_area_of_inscribed_square_l433_43329

theorem area_of_inscribed_square :
  let parabola := λ x => x^2 - 10 * x + 21
  ∃ (t : ℝ), parabola (5 + t) = -2 * t ∧ (2 * t)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_area_of_inscribed_square_l433_43329


namespace NUMINAMATH_GPT_compare_fractions_l433_43310

theorem compare_fractions : -(2 / 3 : ℚ) < -(3 / 5 : ℚ) :=
by sorry

end NUMINAMATH_GPT_compare_fractions_l433_43310


namespace NUMINAMATH_GPT_sufficient_drivers_and_completion_time_l433_43371

noncomputable def one_way_trip_minutes : ℕ := 2 * 60 + 40
noncomputable def round_trip_minutes : ℕ := 2 * one_way_trip_minutes
noncomputable def rest_period_minutes : ℕ := 60
noncomputable def twelve_forty_pm : ℕ := 12 * 60 + 40 -- in minutes from midnight
noncomputable def one_forty_pm : ℕ := twelve_forty_pm + rest_period_minutes
noncomputable def thirteen_five_pm : ℕ := 13 * 60 + 5 -- 1:05 PM
noncomputable def sixteen_ten_pm : ℕ := 16 * 60 + 10 -- 4:10 PM
noncomputable def sixteen_pm : ℕ := 16 * 60 -- 4:00 PM
noncomputable def seventeen_thirty_pm : ℕ := 17 * 60 + 30 -- 5:30 PM
noncomputable def twenty_one_thirty_pm : ℕ := sixteen_ten_pm + round_trip_minutes -- 9:30 PM (21:30)

theorem sufficient_drivers_and_completion_time :
  4 = 4 ∧ twenty_one_thirty_pm = 21 * 60 + 30 := by
  sorry 

end NUMINAMATH_GPT_sufficient_drivers_and_completion_time_l433_43371


namespace NUMINAMATH_GPT_like_terms_sum_l433_43354

theorem like_terms_sum (m n : ℕ) (a b : ℝ) :
  (∀ c d : ℝ, -4 * a^(2 * m) * b^(3) = c * a^(6) * b^(n + 1)) →
  m + n = 5 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_like_terms_sum_l433_43354


namespace NUMINAMATH_GPT_min_hours_to_pass_message_ge_55_l433_43323

theorem min_hours_to_pass_message_ge_55 : 
  ∃ (n: ℕ), (∀ m: ℕ, m < n → 2^(m+1) - 2 ≤ 55) ∧ 2^(n+1) - 2 > 55 :=
by sorry

end NUMINAMATH_GPT_min_hours_to_pass_message_ge_55_l433_43323


namespace NUMINAMATH_GPT_jacqueline_has_29_percent_more_soda_than_liliane_l433_43378

variable (A : ℝ) -- A is the amount of soda Alice has

-- Define the amount of soda Jacqueline has
def J (A : ℝ) : ℝ := 1.80 * A

-- Define the amount of soda Liliane has
def L (A : ℝ) : ℝ := 1.40 * A

-- The statement that needs to be proven
theorem jacqueline_has_29_percent_more_soda_than_liliane (A : ℝ) (hA : A > 0) : 
  ((J A - L A) / L A) * 100 = 29 :=
by
  sorry

end NUMINAMATH_GPT_jacqueline_has_29_percent_more_soda_than_liliane_l433_43378


namespace NUMINAMATH_GPT_find_prime_pair_l433_43395

-- Definition of the problem
def is_integral_expression (p q : ℕ) : Prop :=
  (p + q)^(p + q) * (p - q)^(p - q) - 1 ≠ 0 ∧
  (p + q)^(p - q) * (p - q)^(p + q) - 1 ≠ 0 ∧
  ((p + q)^(p + q) * (p - q)^(p - q) - 1) % ((p + q)^(p - q) * (p - q)^(p + q) - 1) = 0

-- Mathematical theorem to be proved
theorem find_prime_pair (p q : ℕ) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) (h : p > q) :
  is_integral_expression p q → (p, q) = (3, 2) :=
by 
  sorry

end NUMINAMATH_GPT_find_prime_pair_l433_43395


namespace NUMINAMATH_GPT_number_of_zeros_l433_43351

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x - 3

theorem number_of_zeros (b : ℝ) : 
  ∃ x₁ x₂ : ℝ, f x₁ b = 0 ∧ f x₂ b = 0 ∧ x₁ ≠ x₂ := by
  sorry

end NUMINAMATH_GPT_number_of_zeros_l433_43351


namespace NUMINAMATH_GPT_required_workers_l433_43362

variable (x : ℕ) (y : ℕ)

-- Each worker can produce x units of a craft per day.
-- A craft factory needs to produce 60 units of this craft per day.

theorem required_workers (h : x > 0) : y = 60 / x ↔ x * y = 60 :=
by sorry

end NUMINAMATH_GPT_required_workers_l433_43362


namespace NUMINAMATH_GPT_intersection_A_B_l433_43377

def A (x : ℝ) : Prop := (x ≥ 2 ∧ x ≠ 3)
def B (x : ℝ) : Prop := (3 ≤ x ∧ x ≤ 5)
def C := {x : ℝ | 3 < x ∧ x ≤ 5}

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = C :=
  by sorry

end NUMINAMATH_GPT_intersection_A_B_l433_43377


namespace NUMINAMATH_GPT_malfunctioning_clock_fraction_correct_l433_43326

noncomputable def malfunctioning_clock_correct_time_fraction : ℚ := 5 / 8

theorem malfunctioning_clock_fraction_correct :
  malfunctioning_clock_correct_time_fraction = 5 / 8 := 
by
  sorry

end NUMINAMATH_GPT_malfunctioning_clock_fraction_correct_l433_43326


namespace NUMINAMATH_GPT_miriam_flowers_total_l433_43303

theorem miriam_flowers_total :
  let monday_flowers := 45
  let tuesday_flowers := 75
  let wednesday_flowers := 35
  let thursday_flowers := 105
  let friday_flowers := 0
  let saturday_flowers := 60
  (monday_flowers + tuesday_flowers + wednesday_flowers + thursday_flowers + friday_flowers + saturday_flowers) = 320 :=
by
  -- Calculations go here but we're using sorry to skip them
  sorry

end NUMINAMATH_GPT_miriam_flowers_total_l433_43303


namespace NUMINAMATH_GPT_fraction_inequality_solution_l433_43305

theorem fraction_inequality_solution (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ (4 * x + 3 > 2 * (8 - 3 * x)) → (13 / 10) < x ∧ x ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_inequality_solution_l433_43305


namespace NUMINAMATH_GPT_find_rate_percent_l433_43368

-- Define the conditions based on the problem statement
def principal : ℝ := 800
def simpleInterest : ℝ := 160
def time : ℝ := 5

-- Create the statement to prove the rate percent
theorem find_rate_percent : ∃ (rate : ℝ), simpleInterest = (principal * rate * time) / 100 := sorry

end NUMINAMATH_GPT_find_rate_percent_l433_43368


namespace NUMINAMATH_GPT_find_x_l433_43319

theorem find_x (x y z : ℕ) 
  (h1 : x + y = 74) 
  (h2 : (x + y) + y + z = 164) 
  (h3 : z - y = 16) : 
  x = 37 :=
sorry

end NUMINAMATH_GPT_find_x_l433_43319


namespace NUMINAMATH_GPT_count_three_digit_integers_divisible_by_11_and_5_l433_43306

def count_three_digit_multiples (a b: ℕ) : ℕ :=
  let lcm := Nat.lcm a b
  let first_multiple := (100 + lcm - 1) / lcm
  let last_multiple := 999 / lcm
  last_multiple - first_multiple + 1

theorem count_three_digit_integers_divisible_by_11_and_5 : 
  count_three_digit_multiples 11 5 = 17 := by 
  sorry

end NUMINAMATH_GPT_count_three_digit_integers_divisible_by_11_and_5_l433_43306


namespace NUMINAMATH_GPT_minimum_value_sqrt_m2_n2_l433_43307

theorem minimum_value_sqrt_m2_n2 
  (a b m n : ℝ)
  (h1 : a^2 + b^2 = 3)
  (h2 : m*a + n*b = 3) : 
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ Real.sqrt (m^2 + n^2) = k :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_sqrt_m2_n2_l433_43307


namespace NUMINAMATH_GPT_good_horse_catches_up_l433_43347

noncomputable def catch_up_days : ℕ := sorry

theorem good_horse_catches_up (x : ℕ) :
  (∀ (good_horse_speed slow_horse_speed head_start_duration : ℕ),
    good_horse_speed = 200 →
    slow_horse_speed = 120 →
    head_start_duration = 10 →
    200 * x = 120 * x + 120 * 10) →
  catch_up_days = x :=
by
  intro h
  have := h 200 120 10 rfl rfl rfl
  sorry

end NUMINAMATH_GPT_good_horse_catches_up_l433_43347


namespace NUMINAMATH_GPT_dot_product_is_2_l433_43302

variable (a : ℝ × ℝ) (b : ℝ × ℝ)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem dot_product_is_2 (ha : a = (1, 0)) (hb : b = (2, 1)) :
  dot_product a b = 2 := by
  sorry

end NUMINAMATH_GPT_dot_product_is_2_l433_43302


namespace NUMINAMATH_GPT_lauren_total_money_made_is_correct_l433_43384

-- Define the rate per commercial view
def rate_per_commercial_view : ℝ := 0.50
-- Define the rate per subscriber
def rate_per_subscriber : ℝ := 1.00
-- Define the number of commercial views on Tuesday
def commercial_views : ℕ := 100
-- Define the number of new subscribers on Tuesday
def subscribers : ℕ := 27
-- Calculate the total money Lauren made on Tuesday
def total_money_made (rate_com_view : ℝ) (rate_sub : ℝ) (com_views : ℕ) (subs : ℕ) : ℝ :=
  (rate_com_view * com_views) + (rate_sub * subs)

-- Theorem stating that the total money Lauren made on Tuesday is $77.00
theorem lauren_total_money_made_is_correct : total_money_made rate_per_commercial_view rate_per_subscriber commercial_views subscribers = 77.00 :=
by
  sorry

end NUMINAMATH_GPT_lauren_total_money_made_is_correct_l433_43384


namespace NUMINAMATH_GPT_diana_total_earnings_l433_43394

-- Define the earnings in each month
def july_earnings : ℕ := 150
def august_earnings : ℕ := 3 * july_earnings
def september_earnings : ℕ := 2 * august_earnings

-- State the theorem that the total earnings over the three months is $1500
theorem diana_total_earnings : july_earnings + august_earnings + september_earnings = 1500 :=
by
  have h1 : august_earnings = 3 * july_earnings := rfl
  have h2 : september_earnings = 2 * august_earnings := rfl
  sorry

end NUMINAMATH_GPT_diana_total_earnings_l433_43394


namespace NUMINAMATH_GPT_problem_statement_l433_43308

variables (P Q : Prop)

theorem problem_statement (h1 : ¬P) (h2 : ¬(P ∧ Q)) : ¬(P ∨ Q) :=
sorry

end NUMINAMATH_GPT_problem_statement_l433_43308


namespace NUMINAMATH_GPT_leo_current_weight_l433_43333

theorem leo_current_weight (L K : ℕ) 
  (h1 : L + 10 = 3 * K / 2) 
  (h2 : L + K = 160)
  : L = 92 :=
sorry

end NUMINAMATH_GPT_leo_current_weight_l433_43333


namespace NUMINAMATH_GPT_division_criterion_based_on_stroke_l433_43385

-- Definition of a drawable figure with a single stroke
def drawable_in_one_stroke (figure : Type) : Prop := sorry -- exact conditions can be detailed with figure representation

-- Example figures for the groups (types can be extended based on actual representation)
def Group1 := {fig1 : Type // drawable_in_one_stroke fig1}
def Group2 := {fig2 : Type // ¬drawable_in_one_stroke fig2}

-- Problem Statement:
theorem division_criterion_based_on_stroke (fig : Type) :
  (drawable_in_one_stroke fig ∨ ¬drawable_in_one_stroke fig) := by
  -- We state that every figure belongs to either Group1 or Group2
  sorry

end NUMINAMATH_GPT_division_criterion_based_on_stroke_l433_43385


namespace NUMINAMATH_GPT_kiki_scarves_count_l433_43376

variable (money : ℝ) (scarf_cost : ℝ) (hat_spending_ratio : ℝ) (scarves : ℕ) (hats : ℕ)

-- Condition: Kiki has $90.
axiom kiki_money : money = 90

-- Condition: Kiki spends 60% of her money on hats.
axiom kiki_hat_spending_ratio : hat_spending_ratio = 0.60

-- Condition: Each scarf costs $2.
axiom scarf_price : scarf_cost = 2

-- Condition: Kiki buys twice as many hats as scarves.
axiom hat_scarf_relationship : hats = 2 * scarves

theorem kiki_scarves_count 
  (kiki_money : money = 90)
  (kiki_hat_spending_ratio : hat_spending_ratio = 0.60)
  (scarf_price : scarf_cost = 2)
  (hat_scarf_relationship : hats = 2 * scarves)
  : scarves = 18 := 
sorry

end NUMINAMATH_GPT_kiki_scarves_count_l433_43376


namespace NUMINAMATH_GPT_chennai_to_hyderabad_distance_l433_43335

-- Definitions of the conditions
def david_speed := 50 -- mph
def lewis_speed := 70 -- mph
def meet_point := 250 -- miles from Chennai

-- Theorem statement
theorem chennai_to_hyderabad_distance :
  ∃ D T : ℝ, lewis_speed * T = D + (D - meet_point) ∧ david_speed * T = meet_point ∧ D = 300 :=
by
  sorry

end NUMINAMATH_GPT_chennai_to_hyderabad_distance_l433_43335


namespace NUMINAMATH_GPT_find_x_l433_43356

theorem find_x (a y x : ℤ) (h1 : y = 3) (h2 : a * y + x = 10) (h3 : a = 3) : x = 1 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l433_43356


namespace NUMINAMATH_GPT_linda_savings_l433_43365

theorem linda_savings (S : ℝ) 
  (h1 : ∃ f : ℝ, f = 0.9 * 1/2 * S) -- She spent half of her savings on furniture with a 10% discount
  (h2 : ∃ t : ℝ, t = 1/2 * S * 1.05) -- The rest of her savings, spent on TV, had a 5% sales tax applied
  (h3 : 1/2 * S * 1.05 = 300) -- The total cost of the TV after tax was $300
  : S = 571.42 := 
sorry

end NUMINAMATH_GPT_linda_savings_l433_43365


namespace NUMINAMATH_GPT_cindy_envelopes_left_l433_43350

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def number_of_friends : ℕ := 5

theorem cindy_envelopes_left : total_envelopes - (envelopes_per_friend * number_of_friends) = 22 :=
by
  sorry

end NUMINAMATH_GPT_cindy_envelopes_left_l433_43350


namespace NUMINAMATH_GPT_contractor_days_l433_43380

def days_engaged (days_worked days_absent : ℕ) (earnings_per_day : ℝ) (fine_per_absent_day : ℝ) : ℝ :=
  earnings_per_day * days_worked - fine_per_absent_day * days_absent

theorem contractor_days
  (days_absent : ℕ)
  (earnings_per_day : ℝ)
  (fine_per_absent_day : ℝ)
  (total_amount : ℝ)
  (days_worked : ℕ)
  (h1 : days_absent = 12)
  (h2 : earnings_per_day = 25)
  (h3 : fine_per_absent_day = 7.50)
  (h4 : total_amount = 360)
  (h5 : days_engaged days_worked days_absent earnings_per_day fine_per_absent_day = total_amount) :
  days_worked = 18 :=
by sorry

end NUMINAMATH_GPT_contractor_days_l433_43380


namespace NUMINAMATH_GPT_ball_bounces_17_times_to_reach_below_2_feet_l433_43313

theorem ball_bounces_17_times_to_reach_below_2_feet:
  ∃ k: ℕ, (∀ n, n < k → (800 * ((2: ℝ) / 3) ^ n) ≥ 2) ∧ (800 * ((2: ℝ) / 3) ^ k < 2) ∧ k = 17 :=
by
  sorry

end NUMINAMATH_GPT_ball_bounces_17_times_to_reach_below_2_feet_l433_43313


namespace NUMINAMATH_GPT_triangle_inequality_l433_43321

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) :
  1 < a / (b + c) + b / (c + a) + c / (a + b) ∧ a / (b + c) + b / (c + a) + c / (a + b) < 2 := 
sorry

end NUMINAMATH_GPT_triangle_inequality_l433_43321


namespace NUMINAMATH_GPT_smallest_number_diminished_by_2_divisible_12_16_18_21_28_l433_43375

def conditions_holds (n : ℕ) : Prop :=
  (n - 2) % 12 = 0 ∧ (n - 2) % 16 = 0 ∧ (n - 2) % 18 = 0 ∧ (n - 2) % 21 = 0 ∧ (n - 2) % 28 = 0

theorem smallest_number_diminished_by_2_divisible_12_16_18_21_28 :
  ∃ (n : ℕ), conditions_holds n ∧ (∀ m, conditions_holds m → n ≤ m) ∧ n = 1009 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_diminished_by_2_divisible_12_16_18_21_28_l433_43375


namespace NUMINAMATH_GPT_union_A_B_complement_intersect_B_intersection_sub_C_l433_43330

-- Define set A
def A : Set ℝ := {x | -5 < x ∧ x < 1}

-- Define set B
def B : Set ℝ := {x | -2 < x ∧ x < 8}

-- Define set C with variable parameter a
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Problem (1): Prove A ∪ B = { x | -5 < x < 8 }
theorem union_A_B : A ∪ B = {x | -5 < x ∧ x < 8} := 
by sorry

-- Problem (1): Prove (complement R A) ∩ B = { x | 1 ≤ x < 8 }
theorem complement_intersect_B : (Aᶜ) ∩ B = {x | 1 ≤ x ∧ x < 8} :=
by sorry

-- Problem (2): If A ∩ B ⊆ C, prove a ≥ 1
theorem intersection_sub_C (a : ℝ) (h : A ∩ B ⊆ C a) : 1 ≤ a :=
by sorry

end NUMINAMATH_GPT_union_A_B_complement_intersect_B_intersection_sub_C_l433_43330


namespace NUMINAMATH_GPT_trapezoid_area_l433_43393

-- Definitions based on the problem conditions
def Vertex := (Real × Real)

structure Triangle :=
(A : Vertex)
(B : Vertex)
(C : Vertex)
(area : Real)

structure Trapezoid :=
(AB : Real)
(CD : Real)
(M : Vertex)
(area_triangle_ABM : Real)
(area_triangle_CDM : Real)

-- The main theorem we want to prove
theorem trapezoid_area (T : Trapezoid)
  (parallel_sides : T.AB < T.CD)
  (intersect_at_M : ∃ M : Vertex, M = T.M)
  (area_ABM : T.area_triangle_ABM = 2)
  (area_CDM : T.area_triangle_CDM = 8) :
  T.AB * T.CD / (T.CD - T.AB) + T.CD * T.AB / (T.CD - T.AB) = 18 :=
sorry

end NUMINAMATH_GPT_trapezoid_area_l433_43393


namespace NUMINAMATH_GPT_total_time_to_watch_movie_l433_43312

-- Define the conditions and the question
def uninterrupted_viewing_time : ℕ := 35 + 45 + 20
def rewinding_time : ℕ := 5 + 15
def total_time : ℕ := uninterrupted_viewing_time + rewinding_time

-- Lean statement of the proof problem
theorem total_time_to_watch_movie : total_time = 120 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_total_time_to_watch_movie_l433_43312


namespace NUMINAMATH_GPT_bella_stamps_l433_43318

theorem bella_stamps :
  let snowflake_cost := 1.05
  let truck_cost := 1.20
  let rose_cost := 0.90
  let butterfly_cost := 1.15
  let snowflake_spent := 15.75
  
  let snowflake_stamps := snowflake_spent / snowflake_cost
  let truck_stamps := snowflake_stamps + 11
  let rose_stamps := truck_stamps - 17
  let butterfly_stamps := 1.5 * rose_stamps
  
  let total_stamps := snowflake_stamps + truck_stamps + rose_stamps + butterfly_stamps
  
  total_stamps = 64 := by
  sorry

end NUMINAMATH_GPT_bella_stamps_l433_43318


namespace NUMINAMATH_GPT_machine_subtract_l433_43396

theorem machine_subtract (x : ℤ) (h1 : 26 + 15 - x = 35) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_machine_subtract_l433_43396


namespace NUMINAMATH_GPT_jamie_catches_bus_probability_l433_43300

noncomputable def probability_jamie_catches_bus : ℝ :=
  let total_area := 120 * 120
  let overlap_area := 20 * 100
  overlap_area / total_area

theorem jamie_catches_bus_probability :
  probability_jamie_catches_bus = (5 / 36) :=
by
  sorry

end NUMINAMATH_GPT_jamie_catches_bus_probability_l433_43300


namespace NUMINAMATH_GPT_value_of_a5_max_sum_first_n_value_l433_43309

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_first_n (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem value_of_a5 (a d a5 : ℤ) :
  a5 = 4 ↔ (2 * a + 4 * d) + (a + 4 * d) + (a + 8 * d) = (a + 5 * d) + 8 :=
  sorry

theorem max_sum_first_n_value (a d : ℤ) (n : ℕ) (max_n : ℕ) :
  a = 16 →
  d = -3 →
  (∀ i, sum_first_n a d i ≤ sum_first_n a d max_n) →
  max_n = 6 :=
  sorry

end NUMINAMATH_GPT_value_of_a5_max_sum_first_n_value_l433_43309


namespace NUMINAMATH_GPT_lucy_fish_bought_l433_43340

def fish_bought (fish_original fish_now : ℕ) : ℕ :=
  fish_now - fish_original

theorem lucy_fish_bought : fish_bought 212 492 = 280 :=
by
  sorry

end NUMINAMATH_GPT_lucy_fish_bought_l433_43340


namespace NUMINAMATH_GPT_find_sum_of_distinct_numbers_l433_43311

variable {R : Type} [LinearOrderedField R]

theorem find_sum_of_distinct_numbers (p q r s : R) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p ∧ r * s = -13 * q)
  (h2 : p + q = 12 * r ∧ p * q = -13 * s) :
  p + q + r + s = 2028 := 
by 
  sorry

end NUMINAMATH_GPT_find_sum_of_distinct_numbers_l433_43311


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l433_43383

theorem volume_of_rectangular_prism (a b c : ℝ)
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : a * c = Real.sqrt 6) :
  a * b * c = Real.sqrt 6 := by
sorry

end NUMINAMATH_GPT_volume_of_rectangular_prism_l433_43383


namespace NUMINAMATH_GPT_lucy_first_round_cookies_l433_43315

theorem lucy_first_round_cookies (x : ℕ) : 
  (x + 27 = 61) → x = 34 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_lucy_first_round_cookies_l433_43315


namespace NUMINAMATH_GPT_coordinates_of_B_l433_43398

/--
Given point A with coordinates (2, -3) and line segment AB parallel to the x-axis,
and the length of AB being 4, prove that the coordinates of point B are either (-2, -3)
or (6, -3).
-/
theorem coordinates_of_B (x1 y1 : ℝ) (d : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : d = 4) (hx : 0 ≤ d) :
  ∃ x2 : ℝ, ∃ y2 : ℝ, (y2 = y1) ∧ ((x2 = x1 + d) ∨ (x2 = x1 - d)) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_B_l433_43398


namespace NUMINAMATH_GPT_sum_of_integers_mod_59_l433_43389

theorem sum_of_integers_mod_59 (a b c : ℕ) (h1 : a % 59 = 29) (h2 : b % 59 = 31) (h3 : c % 59 = 7)
  (h4 : a^2 % 59 = 29) (h5 : b^2 % 59 = 31) (h6 : c^2 % 59 = 7) :
  (a + b + c) % 59 = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_mod_59_l433_43389


namespace NUMINAMATH_GPT_total_distance_joseph_ran_l433_43363

-- Defining the conditions
def distance_per_day : ℕ := 900
def days_run : ℕ := 3

-- The proof problem statement
theorem total_distance_joseph_ran :
  (distance_per_day * days_run) = 2700 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_joseph_ran_l433_43363


namespace NUMINAMATH_GPT_range_of_m_l433_43342

theorem range_of_m (m : ℝ) : 
  ((0 - m)^2 + (0 + m)^2 < 4) → -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l433_43342


namespace NUMINAMATH_GPT_matrix_cube_computation_l433_43397

-- Define the original matrix
def matrix1 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -2], ![2, 0]]

-- Define the expected result matrix
def expected_matrix : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![-8, 0], ![0, -8]]

-- State the theorem to be proved
theorem matrix_cube_computation : matrix1 ^ 3 = expected_matrix :=
  by sorry

end NUMINAMATH_GPT_matrix_cube_computation_l433_43397


namespace NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l433_43317

theorem option_A : (-(-1) : ℤ) ≠ -|(-1 : ℤ)| := by
  sorry

theorem option_B : ((-3)^2 : ℤ) ≠ -(3^2 : ℤ) := by
  sorry

theorem option_C : ((-4)^3 : ℤ) = -(4^3 : ℤ) := by
  sorry

theorem option_D : ((2^2 : ℚ)/3) ≠ ((2/3)^2 : ℚ) := by
  sorry

end NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l433_43317


namespace NUMINAMATH_GPT_sequence_solution_l433_43341

theorem sequence_solution (a : ℕ → ℝ) :
  (∀ m n : ℕ, 1 ≤ m → 1 ≤ n → a (m + n) = a m + a n - m * n) ∧ 
  (∀ m n : ℕ, 1 ≤ m → 1 ≤ n → a (m * n) = m^2 * a n + n^2 * a m + 2 * a m * a n) →
    (∀ n, a n = -n*(n-1)/2) ∨ (∀ n, a n = -n^2/2) :=
  by
  sorry

end NUMINAMATH_GPT_sequence_solution_l433_43341


namespace NUMINAMATH_GPT_recipe_flour_cups_l433_43355

theorem recipe_flour_cups (F : ℕ) : 
  (exists (sugar : ℕ) (flourAdded : ℕ) (sugarExtra : ℕ), sugar = 11 ∧ flourAdded = 4 ∧ sugarExtra = 6 ∧ ((F - flourAdded) + sugarExtra = sugar)) →
  F = 9 :=
sorry

end NUMINAMATH_GPT_recipe_flour_cups_l433_43355


namespace NUMINAMATH_GPT_solve_for_x_l433_43392

theorem solve_for_x (x : ℤ) (h : 45 - (5 * 3) = x + 7) : x = 23 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l433_43392


namespace NUMINAMATH_GPT_range_of_b_l433_43301

theorem range_of_b (f g : ℝ → ℝ) (a b : ℝ)
  (hf : ∀ x, f x = Real.exp x - 1)
  (hg : ∀ x, g x = -x^2 + 4*x - 3)
  (h : f a = g b) :
  2 - Real.sqrt 2 < b ∧ b < 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_range_of_b_l433_43301


namespace NUMINAMATH_GPT_football_team_progress_l433_43367

theorem football_team_progress (loss gain : ℤ) (h_loss : loss = -5) (h_gain : gain = 8) :
  (loss + gain = 3) :=
by
  sorry

end NUMINAMATH_GPT_football_team_progress_l433_43367


namespace NUMINAMATH_GPT_ball_arrangement_problem_l433_43343

-- Defining the problem statement and conditions
theorem ball_arrangement_problem : 
  (∃ (A : ℕ), 
    (∀ (b : Fin 6 → ℕ), 
      (b 0 = 1 ∨ b 1 = 1) ∧ (b 0 = 2 ∨ b 1 = 2) ∧ -- 1 adjacent to 2
      b 4 ≠ 5 ∧ b 4 ≠ 6 ∧                 -- 5 not adjacent to 6 condition
      b 5 ≠ 5 ∧ b 5 ≠ 6     -- Add all other necessary conditions for arrangement
    ) →
    A = 144)
:= sorry

end NUMINAMATH_GPT_ball_arrangement_problem_l433_43343


namespace NUMINAMATH_GPT_probability_of_one_or_two_l433_43386

/-- Represents the number of elements in the first 20 rows of Pascal's Triangle. -/
noncomputable def total_elements : ℕ := 210

/-- Represents the number of ones in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_ones : ℕ := 39

/-- Represents the number of twos in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_twos : ℕ :=18

/-- Prove that the probability of randomly choosing an element which is either 1 or 2
from the first 20 rows of Pascal's Triangle is 57/210. -/
theorem probability_of_one_or_two (h1 : total_elements = 210)
                                  (h2 : number_of_ones = 39)
                                  (h3 : number_of_twos = 18) :
    39 + 18 = 57 ∧ (57 : ℚ) / 210 = 57 / 210 :=
by {
    sorry
}

end NUMINAMATH_GPT_probability_of_one_or_two_l433_43386


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l433_43361

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℝ) (T : ℕ → ℝ) (a b : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, S n = 3 * k * n^2)
  (h2 : ∀ n, T n = k * n * (2 * n + 1))
  (h3 : ∀ n, a n = S n - S (n - 1))
  (h4 : ∀ n, b n = T n - T (n - 1))
  (h5 : ∀ n, S n / T n = (3 * n) / (2 * n + 1)) :
  (a 1 + a 2 + a 14 + a 19) / (b 1 + b 3 + b 17 + b 19) = 17 / 13 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l433_43361


namespace NUMINAMATH_GPT_valid_range_for_b_l433_43304

noncomputable def f (x b : ℝ) : ℝ := -x^2 + 2 * x + b^2 - b + 1

theorem valid_range_for_b (b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x b > 0) → (b < -1 ∨ b > 2) :=
by
  sorry

end NUMINAMATH_GPT_valid_range_for_b_l433_43304


namespace NUMINAMATH_GPT_chimney_bricks_l433_43364

theorem chimney_bricks (x : ℝ) 
  (h1 : ∀ x, Brenda_rate = x / 8) 
  (h2 : ∀ x, Brandon_rate = x / 12) 
  (h3 : Combined_rate = (Brenda_rate + Brandon_rate - 15)) 
  (h4 : x = Combined_rate * 6) 
  : x = 360 := 
by 
  sorry

end NUMINAMATH_GPT_chimney_bricks_l433_43364


namespace NUMINAMATH_GPT_consecutive_odd_numbers_l433_43348

theorem consecutive_odd_numbers (a b c d e : ℤ) (h1 : b = a + 2) (h2 : c = a + 4) (h3 : d = a + 6) (h4 : e = a + 8) (h5 : a + c = 146) : e = 79 := 
by
  sorry

end NUMINAMATH_GPT_consecutive_odd_numbers_l433_43348


namespace NUMINAMATH_GPT_area_of_isosceles_triangle_l433_43346

theorem area_of_isosceles_triangle
  (h : ℝ)
  (s : ℝ)
  (b : ℝ)
  (altitude : h = 10)
  (perimeter : s + (s - 2) + 2 * b = 40)
  (pythagoras : b^2 + h^2 = s^2) :
  (b * h) = 81.2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_isosceles_triangle_l433_43346


namespace NUMINAMATH_GPT_bus_stop_time_l433_43353

/-- 
  We are given:
  speed_ns: speed of bus without stoppages (32 km/hr)
  speed_ws: speed of bus including stoppages (16 km/hr)
  
  We need to prove the bus stops for t = 30 minutes each hour.
-/
theorem bus_stop_time
  (speed_ns speed_ws: ℕ)
  (h_ns: speed_ns = 32)
  (h_ws: speed_ws = 16):
  ∃ t: ℕ, t = 30 := 
sorry

end NUMINAMATH_GPT_bus_stop_time_l433_43353


namespace NUMINAMATH_GPT_proof_one_third_of_seven_times_nine_subtract_three_l433_43387

def one_third_of_seven_times_nine_subtract_three : ℕ :=
  let product := 7 * 9
  let one_third := product / 3
  one_third - 3

theorem proof_one_third_of_seven_times_nine_subtract_three : one_third_of_seven_times_nine_subtract_three = 18 := by
  sorry

end NUMINAMATH_GPT_proof_one_third_of_seven_times_nine_subtract_three_l433_43387


namespace NUMINAMATH_GPT_g_five_eq_one_l433_43325

variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (x - y) = g x * g y)
variable (h_ne_zero : ∀ x : ℝ, g x ≠ 0)

theorem g_five_eq_one : g 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_g_five_eq_one_l433_43325


namespace NUMINAMATH_GPT_simplify_expression_l433_43391

theorem simplify_expression :
  ((2 + 3 + 4 + 5) / 2) + ((2 * 5 + 8) / 3) = 13 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l433_43391


namespace NUMINAMATH_GPT_simple_interest_correct_l433_43388

-- Define the parameters
def principal : ℝ := 10000
def rate_decimal : ℝ := 0.04
def time_years : ℝ := 1

-- Define the simple interest calculation function
noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Prove that the simple interest is equal to $400
theorem simple_interest_correct : simple_interest principal rate_decimal time_years = 400 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_simple_interest_correct_l433_43388


namespace NUMINAMATH_GPT_find_a_and_b_l433_43372

theorem find_a_and_b (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = x^3 - a * x^2 - b * x + a^2) →
  f 1 = 10 →
  deriv f 1 = 0 →
  (a = -4 ∧ b = 11) :=
by
  intros hf hf1 hderiv
  sorry

end NUMINAMATH_GPT_find_a_and_b_l433_43372


namespace NUMINAMATH_GPT_max_diff_six_digit_even_numbers_l433_43349

-- Definitions for six-digit numbers with all digits even
def is_6_digit_even (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000 ∧ (∀ (d : ℕ), d < 6 → (n / 10^d) % 10 % 2 = 0)

def contains_odd_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 6 ∧ (n / 10^d) % 10 % 2 = 1

-- The main theorem
theorem max_diff_six_digit_even_numbers (a b : ℕ) 
  (ha : is_6_digit_even a) 
  (hb : is_6_digit_even b)
  (h_cond : ∀ n : ℕ, a < n ∧ n < b → contains_odd_digit n) 
  : b - a = 111112 :=
sorry

end NUMINAMATH_GPT_max_diff_six_digit_even_numbers_l433_43349


namespace NUMINAMATH_GPT_parabola_directrix_l433_43359

noncomputable def equation_of_directrix (a h k : ℝ) : ℝ :=
  k - 1 / (4 * a)

theorem parabola_directrix:
  ∀ (a h k : ℝ), a = -3 ∧ h = 1 ∧ k = -2 → equation_of_directrix a h k = - 23 / 12 :=
by
  intro a h k
  intro h_ahk
  sorry

end NUMINAMATH_GPT_parabola_directrix_l433_43359


namespace NUMINAMATH_GPT_sum_of_fourth_powers_eq_174_fourth_l433_43382

theorem sum_of_fourth_powers_eq_174_fourth :
  120 ^ 4 + 97 ^ 4 + 84 ^ 4 + 27 ^ 4 = 174 ^ 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_eq_174_fourth_l433_43382


namespace NUMINAMATH_GPT_geom_seq_sum_l433_43390

noncomputable def geom_seq (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
a₁ * r^(n-1)

theorem geom_seq_sum (a₁ r : ℝ) (h_pos : 0 < a₁) (h_pos_r : 0 < r)
  (h : a₁ * (geom_seq a₁ r 5) + 2 * (geom_seq a₁ r 3) * (geom_seq a₁ r 6) + a₁ * (geom_seq a₁ r 11) = 16) :
  (geom_seq a₁ r 3 + geom_seq a₁ r 6) = 4 :=
sorry

end NUMINAMATH_GPT_geom_seq_sum_l433_43390


namespace NUMINAMATH_GPT_function_for_negative_x_l433_43314

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → f x = x * (1 - x)

theorem function_for_negative_x {f : ℝ → ℝ} :
  odd_function f → given_function f → ∀ x, x < 0 → f x = x * (1 + x) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_function_for_negative_x_l433_43314


namespace NUMINAMATH_GPT_find_other_number_l433_43399

-- Define the conditions and the theorem
theorem find_other_number (hcf lcm a b : ℕ) (hcf_def : hcf = 20) (lcm_def : lcm = 396) (a_def : a = 36) (rel : hcf * lcm = a * b) : b = 220 :=
by 
  sorry -- Proof to be provided

end NUMINAMATH_GPT_find_other_number_l433_43399


namespace NUMINAMATH_GPT_time_to_be_d_miles_apart_l433_43337

def mary_walk_rate := 4 -- Mary's walking rate in miles per hour
def sharon_walk_rate := 6 -- Sharon's walking rate in miles per hour
def time_to_be_3_miles_apart := 0.3 -- Time in hours to be 3 miles apart
def initial_distance := 3 -- They are 3 miles apart after 0.3 hours

theorem time_to_be_d_miles_apart (d: ℝ) : ∀ t: ℝ,
  (mary_walk_rate + sharon_walk_rate) * t = d ↔ 
  t = d / (mary_walk_rate + sharon_walk_rate) :=
by
  intros
  sorry

end NUMINAMATH_GPT_time_to_be_d_miles_apart_l433_43337


namespace NUMINAMATH_GPT_third_smallest_is_four_probability_l433_43322

noncomputable def probability_third_smallest_is_four : ℚ :=
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 4)
  favorable_ways / total_ways

theorem third_smallest_is_four_probability : 
  probability_third_smallest_is_four = 35 / 132 := 
sorry

end NUMINAMATH_GPT_third_smallest_is_four_probability_l433_43322


namespace NUMINAMATH_GPT_not_divisible_by_24_l433_43344

theorem not_divisible_by_24 : 
  ¬ (121416182022242628303234 % 24 = 0) := 
by
  sorry

end NUMINAMATH_GPT_not_divisible_by_24_l433_43344


namespace NUMINAMATH_GPT_exists_points_same_color_one_meter_apart_l433_43334

-- Predicate to describe points in the 2x2 square
structure Point where
  x : ℝ
  y : ℝ
  h_x : 0 ≤ x ∧ x ≤ 2
  h_y : 0 ≤ y ∧ y ≤ 2

-- Function to describe the color assignment
def color (p : Point) : Prop := sorry -- True = Black, False = White

-- The main theorem to be proven
theorem exists_points_same_color_one_meter_apart :
  ∃ p1 p2 : Point, color p1 = color p2 ∧ dist (p1.1, p1.2) (p2.1, p2.2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_points_same_color_one_meter_apart_l433_43334


namespace NUMINAMATH_GPT_matrix_power_four_l433_43320

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -1], ![1, 1]]

theorem matrix_power_four :
  (A^4) = ![![0, -9], ![9, -9]] :=
by
  sorry

end NUMINAMATH_GPT_matrix_power_four_l433_43320


namespace NUMINAMATH_GPT_range_of_m_l433_43339

theorem range_of_m (y : ℝ) (x : ℝ) (xy_ne_zero : x * y ≠ 0) :
  (x^2 + 4 * y^2 = (m^2 + 3 * m) * x * y) → -4 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l433_43339


namespace NUMINAMATH_GPT_solve_equation_l433_43327

theorem solve_equation :
  ∀ (x m n : ℕ), 
    0 < x → 0 < m → 0 < n → 
    x^m = 2^(2 * n + 1) + 2^n + 1 →
    (x = 2^(2 * n + 1) + 2^n + 1 ∧ m = 1) ∨ (x = 23 ∧ m = 2 ∧ n = 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l433_43327


namespace NUMINAMATH_GPT_exists_positive_integers_for_hexagon_area_l433_43328

theorem exists_positive_integers_for_hexagon_area (S : ℕ) (a b : ℕ) (hS : S = 2016) :
  2 * (a^2 + b^2 + a * b) = S → ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 2 * (a^2 + b^2 + a * b) = S :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_integers_for_hexagon_area_l433_43328


namespace NUMINAMATH_GPT_ordered_triples_lcm_sum_zero_l433_43358

theorem ordered_triples_lcm_sum_zero :
  ∀ (x y z : ℕ), 
    (0 < x) → 
    (0 < y) → 
    (0 < z) → 
    Nat.lcm x y = 180 →
    Nat.lcm x z = 450 →
    Nat.lcm y z = 600 →
    x + y + z = 120 →
    false := 
by
  intros x y z hx hy hz hxy hxz hyz hs
  sorry

end NUMINAMATH_GPT_ordered_triples_lcm_sum_zero_l433_43358


namespace NUMINAMATH_GPT_total_cars_produced_l433_43379

def CarCompanyA_NorthAmerica := 3884
def CarCompanyA_Europe := 2871
def CarCompanyA_Asia := 1529

def CarCompanyB_NorthAmerica := 4357
def CarCompanyB_Europe := 3690
def CarCompanyB_Asia := 1835

def CarCompanyC_NorthAmerica := 2937
def CarCompanyC_Europe := 4210
def CarCompanyC_Asia := 977

def TotalNorthAmerica :=
  CarCompanyA_NorthAmerica + CarCompanyB_NorthAmerica + CarCompanyC_NorthAmerica

def TotalEurope :=
  CarCompanyA_Europe + CarCompanyB_Europe + CarCompanyC_Europe

def TotalAsia :=
  CarCompanyA_Asia + CarCompanyB_Asia + CarCompanyC_Asia

def TotalProduction := TotalNorthAmerica + TotalEurope + TotalAsia

theorem total_cars_produced : TotalProduction = 26290 := 
by sorry

end NUMINAMATH_GPT_total_cars_produced_l433_43379


namespace NUMINAMATH_GPT_nuts_in_trail_mix_l433_43332

theorem nuts_in_trail_mix :
  let walnuts := 0.25
  let almonds := 0.25
  walnuts + almonds = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_nuts_in_trail_mix_l433_43332


namespace NUMINAMATH_GPT_man_born_year_l433_43338

theorem man_born_year (x : ℕ) : 
  (x^2 - x = 1806) ∧ (x^2 - x < 1850) ∧ (40 < x) ∧ (x < 50) → x = 43 :=
by
  sorry

end NUMINAMATH_GPT_man_born_year_l433_43338


namespace NUMINAMATH_GPT_positive_difference_sums_even_odd_l433_43373

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_sums_even_odd_l433_43373


namespace NUMINAMATH_GPT_degree_of_d_l433_43374

theorem degree_of_d (f d q r : Polynomial ℝ) (f_deg : f.degree = 17)
  (q_deg : q.degree = 10) (r_deg : r.degree = 4) 
  (remainder : r = Polynomial.C 5 * X^4 - Polynomial.C 3 * X^3 + Polynomial.C 2 * X^2 - X + 15)
  (div_relation : f = d * q + r) (r_deg_lt_d_deg : r.degree < d.degree) :
  d.degree = 7 :=
sorry

end NUMINAMATH_GPT_degree_of_d_l433_43374


namespace NUMINAMATH_GPT_part_one_solution_set_part_two_range_of_a_l433_43316

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part_one_solution_set :
  { x : ℝ | f x ≤ 9 } = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
sorry

theorem part_two_range_of_a (a : ℝ) (B := { x : ℝ | x^2 - 3 * x < 0 })
  (A := { x : ℝ | f x < 2 * x + a }) :
  B ⊆ A → 5 ≤ a :=
sorry

end NUMINAMATH_GPT_part_one_solution_set_part_two_range_of_a_l433_43316
