import Mathlib

namespace find_integers_l1583_158341

theorem find_integers (x : ℤ) (h₁ : x ≠ 3) (h₂ : (x - 3) ∣ (x ^ 3 - 3)) :
  x = -21 ∨ x = -9 ∨ x = -5 ∨ x = -3 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 4 ∨ x = 5 ∨
  x = 7 ∨ x = 9 ∨ x = 11 ∨ x = 15 ∨ x = 27 :=
sorry

end find_integers_l1583_158341


namespace proof_problem_l1583_158385

-- Define the conditions
def a : ℤ := -3
def b : ℤ := -4
def cond1 := a^4 = 81
def cond2 := b^3 = -64

-- Define the goal in terms of the conditions
theorem proof_problem : a^4 + b^3 = 17 :=
by
  have h1 : a^4 = 81 := sorry
  have h2 : b^3 = -64 := sorry
  rw [h1, h2]
  norm_num

end proof_problem_l1583_158385


namespace sarah_earnings_l1583_158325

-- Conditions
def monday_hours : ℚ := 1 + 3 / 4
def wednesday_hours : ℚ := 65 / 60
def thursday_hours : ℚ := 2 + 45 / 60
def friday_hours : ℚ := 45 / 60
def saturday_hours : ℚ := 2

def weekday_rate : ℚ := 4
def weekend_rate : ℚ := 6

-- Definition for total earnings
def total_weekday_earnings : ℚ :=
  (monday_hours + wednesday_hours + thursday_hours + friday_hours) * weekday_rate

def total_weekend_earnings : ℚ :=
  saturday_hours * weekend_rate

def total_earnings : ℚ :=
  total_weekday_earnings + total_weekend_earnings

-- Statement to prove
theorem sarah_earnings : total_earnings = 37.3332 := by
  sorry

end sarah_earnings_l1583_158325


namespace hypotenuse_square_l1583_158371

-- Define the right triangle property and the consecutive integer property
variables (a b c : ℤ)

-- Noncomputable definition will be used as we are proving a property related to integers
noncomputable def consecutive_integers (a b : ℤ) : Prop := b = a + 1

-- Define the statement to prove
theorem hypotenuse_square (h_consec : consecutive_integers a b) (h_right_triangle : a * a + b * b = c * c) : 
  c * c = 2 * a * a + 2 * a + 1 :=
by {
  -- We only need to state the theorem
  sorry
}

end hypotenuse_square_l1583_158371


namespace aunt_may_milk_left_l1583_158398

theorem aunt_may_milk_left
  (morning_milk : ℕ)
  (evening_milk : ℕ)
  (sold_milk : ℕ)
  (leftover_milk : ℕ)
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk + leftover_milk - sold_milk = 148 :=
by
  sorry

end aunt_may_milk_left_l1583_158398


namespace sum_of_arithmetic_series_l1583_158395

-- Define the conditions
def first_term := 1
def last_term := 12
def number_of_terms := 12

-- Prop statement that the sum of the arithmetic series equals 78
theorem sum_of_arithmetic_series : (number_of_terms / 2) * (first_term + last_term) = 78 := 
by
  sorry

end sum_of_arithmetic_series_l1583_158395


namespace average_speed_l1583_158391

theorem average_speed (x : ℝ) (h1 : x > 0) :
  let dist1 := x
  let speed1 := 40
  let dist2 := 4 * x
  let speed2 := 20
  let total_dist := dist1 + dist2
  let time1 := dist1 / speed1
  let time2 := dist2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_dist / total_time
  avg_speed = 200 / 9 :=
by
  -- Definitions
  let dist1 := x
  let speed1 := 40
  let dist2 := 4 * x
  let speed2 := 20
  let total_dist := dist1 + dist2
  let time1 := dist1 / speed1
  let time2 := dist2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_dist / total_time
  -- Proof structure, concluding with the correct answer.
  sorry

end average_speed_l1583_158391


namespace problem_1_l1583_158339

theorem problem_1 (α : ℝ) (k : ℤ) (n : ℕ) (hk : k > 0) (hα : α ≠ k * Real.pi) (hn : n > 0) :
  n = 1 → (0.5 + Real.cos α) = (0.5 + Real.cos α) :=
by
  sorry

end problem_1_l1583_158339


namespace rhombus_longer_diagonal_l1583_158374

theorem rhombus_longer_diagonal (d1 d2 : ℝ) (h_d1 : d1 = 11) (h_area : (d1 * d2) / 2 = 110) : d2 = 20 :=
by
  sorry

end rhombus_longer_diagonal_l1583_158374


namespace jordan_probability_l1583_158366

-- Definitions based on conditions.
def total_students := 28
def enrolled_in_french := 20
def enrolled_in_spanish := 23
def enrolled_in_both := 17

-- Calculate students enrolled only in one language.
def only_french := enrolled_in_french - enrolled_in_both
def only_spanish := enrolled_in_spanish - enrolled_in_both

-- Calculation of combinations.
def total_combinations := Nat.choose total_students 2
def only_french_combinations := Nat.choose only_french 2
def only_spanish_combinations := Nat.choose only_spanish 2

-- Probability calculations.
def prob_both_one_language := (only_french_combinations + only_spanish_combinations) / total_combinations

def prob_both_languages : ℚ := 1 - prob_both_one_language

theorem jordan_probability :
  prob_both_languages = (20 : ℚ) / 21 := by
  sorry

end jordan_probability_l1583_158366


namespace min_coins_needed_l1583_158388

-- Definitions for coins
def coins (pennies nickels dimes quarters : Nat) : Nat :=
  pennies + nickels + dimes + quarters

-- Condition: minimum number of coins to pay any amount less than a dollar
def can_pay_any_amount (pennies nickels dimes quarters : Nat) : Prop :=
  ∀ (amount : Nat), 1 ≤ amount ∧ amount < 100 →
  ∃ (p n d q : Nat), p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧ q ≤ quarters ∧
  p + 5 * n + 10 * d + 25 * q = amount

-- The main Lean 4 statement
theorem min_coins_needed :
  ∃ (pennies nickels dimes quarters : Nat),
    coins pennies nickels dimes quarters = 11 ∧
    can_pay_any_amount pennies nickels dimes quarters :=
sorry

end min_coins_needed_l1583_158388


namespace range_of_a_l1583_158353

variable (a : ℝ)

def proposition_p (a : ℝ) : Prop := 0 < a ∧ a < 1

def proposition_q (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 - x + a > 0 ∧ 1 - 4 * a^2 < 0

theorem range_of_a : (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) →
  (0 < a ∧ a ≤ 1/2 ∨ a ≥ 1) := 
by
  sorry

end range_of_a_l1583_158353


namespace no_14_non_square_rectangles_l1583_158320

theorem no_14_non_square_rectangles (side_len : ℕ) 
    (h_side_len : side_len = 9) 
    (num_rectangles : ℕ) 
    (h_num_rectangles : num_rectangles = 14) 
    (min_side_len : ℕ → ℕ → Prop) 
    (h_min_side_len : ∀ l w, min_side_len l w → l ≥ 2 ∧ w ≥ 2) : 
    ¬ (∀ l w, min_side_len l w → l ≠ w) :=
by {
    sorry
}

end no_14_non_square_rectangles_l1583_158320


namespace problem_statement_l1583_158390

theorem problem_statement :
  102^3 + 3 * 102^2 + 3 * 102 + 1 = 1092727 :=
  by sorry

end problem_statement_l1583_158390


namespace stadium_seating_and_revenue_l1583_158318

   def children := 52
   def adults := 29
   def seniors := 15
   def seats_A := 40
   def seats_B := 30
   def seats_C := 25
   def price_A := 10
   def price_B := 15
   def price_C := 20
   def total_seats := 95

   def revenue_A := seats_A * price_A
   def revenue_B := seats_B * price_B
   def revenue_C := seats_C * price_C
   def total_revenue := revenue_A + revenue_B + revenue_C

   theorem stadium_seating_and_revenue :
     (children <= seats_B + seats_C) ∧
     (adults + seniors <= seats_A + seats_C) ∧
     (children + adults + seniors > total_seats) →
     (revenue_A = 400) ∧
     (revenue_B = 450) ∧
     (revenue_C = 500) ∧
     (total_revenue = 1350) :=
   by
     sorry
   
end stadium_seating_and_revenue_l1583_158318


namespace working_mom_work_percent_l1583_158312

theorem working_mom_work_percent :
  let awake_hours := 16
  let work_hours := 8
  (work_hours / awake_hours) * 100 = 50 :=
by
  sorry

end working_mom_work_percent_l1583_158312


namespace complex_div_conjugate_l1583_158333

theorem complex_div_conjugate (a b : ℂ) (h1 : a = 2 - I) (h2 : b = 1 + 2 * I) :
    a / b = -I := by
  sorry

end complex_div_conjugate_l1583_158333


namespace scientific_notation_conversion_l1583_158381

theorem scientific_notation_conversion : 
  ∀ (n : ℝ), n = 1.8 * 10^8 → n = 180000000 :=
by
  intros n h
  sorry

end scientific_notation_conversion_l1583_158381


namespace angle_greater_difference_l1583_158355

theorem angle_greater_difference (A B C : ℕ) (h1 : B = 5 * A) (h2 : A + B + C = 180) (h3 : A = 24) 
: C - A = 12 := 
by
  -- Proof omitted
  sorry

end angle_greater_difference_l1583_158355


namespace power_C_50_l1583_158310

def matrixC : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, 1], ![-4, -1]]

theorem power_C_50 :
  matrixC ^ 50 = ![![4^49 + 1, 4^49], ![-4^50, -2 * 4^49 + 1]] :=
by
  sorry

end power_C_50_l1583_158310


namespace ratio_equivalence_l1583_158350

theorem ratio_equivalence (x : ℕ) : 
  (10 * 60 = 600) →
  (15 : ℕ) / 5 = x / 600 →
  x = 1800 :=
by
  intros h1 h2
  sorry

end ratio_equivalence_l1583_158350


namespace weight_of_first_lift_l1583_158323

-- Definitions as per conditions
variables (x y : ℝ)
def condition1 : Prop := x + y = 1800
def condition2 : Prop := 2 * x = y + 300

-- Prove that the weight of Joe's first lift is 700 pounds
theorem weight_of_first_lift (h1 : condition1 x y) (h2 : condition2 x y) : x = 700 :=
by
  sorry

end weight_of_first_lift_l1583_158323


namespace find_x_y_l1583_158361

theorem find_x_y (x y : ℝ) (h : (2 * x - 3 * y + 5) ^ 2 + |x - y + 2| = 0) : x = -1 ∧ y = 1 :=
by
  sorry

end find_x_y_l1583_158361


namespace min_value_of_c_l1583_158306

variable {a b c : ℝ}
variables (a_pos : a > 0) (b_pos : b > 0)
variable (hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
variable (semi_focal_dist : c = Real.sqrt (a^2 + b^2))
variable (distance_condition : ∀ (d : ℝ), d = a * b / c = 1 / 3 * c + 1)

theorem min_value_of_c : c = 6 := 
sorry

end min_value_of_c_l1583_158306


namespace students_not_in_biology_l1583_158360

theorem students_not_in_biology (total_students : ℕ) (percentage_in_biology : ℚ) 
  (h1 : total_students = 880) (h2 : percentage_in_biology = 27.5 / 100) : 
  total_students - (total_students * percentage_in_biology) = 638 := 
by
  sorry

end students_not_in_biology_l1583_158360


namespace trig_expression_evaluation_l1583_158324

theorem trig_expression_evaluation
  (α : ℝ)
  (h_tan_α : Real.tan α = 3) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / 
  (Real.cos (3 * π / 2 - α) + 2 * Real.cos (-π + α)) = -2 / 5 := by
  sorry

end trig_expression_evaluation_l1583_158324


namespace smallest_n_for_multiples_of_2015_l1583_158326

theorem smallest_n_for_multiples_of_2015 (n : ℕ) (hn : 0 < n)
  (h5 : (2^n - 1) % 5 = 0)
  (h13 : (2^n - 1) % 13 = 0)
  (h31 : (2^n - 1) % 31 = 0) : n = 60 := by
  sorry

end smallest_n_for_multiples_of_2015_l1583_158326


namespace sum_a6_a7_a8_is_32_l1583_158379

noncomputable def geom_seq (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem sum_a6_a7_a8_is_32 (q a₁ : ℝ) 
  (h1 : geom_seq q a₁ 1 + geom_seq q a₁ 2 + geom_seq q a₁ 3 = 1)
  (h2 : geom_seq q a₁ 2 + geom_seq q a₁ 3 + geom_seq q a₁ 4 = 2) :
  geom_seq q a₁ 6 + geom_seq q a₁ 7 + geom_seq q a₁ 8 = 32 :=
sorry

end sum_a6_a7_a8_is_32_l1583_158379


namespace asymptotic_minimal_eccentricity_l1583_158303

noncomputable def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 / (m^2 + 4) = 1

noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt (m + 4 / m + 1)

theorem asymptotic_minimal_eccentricity :
  ∃ (m : ℝ), m = 2 ∧ hyperbola m x y → ∀ x y, y = 2 * x ∨ y = -2 * x :=
by
  sorry

end asymptotic_minimal_eccentricity_l1583_158303


namespace part1_part2_l1583_158396

open Set

variable (U : Set ℤ := {x | -3 ≤ x ∧ x ≤ 3})
variable (A : Set ℤ := {1, 2, 3})
variable (B : Set ℤ := {-1, 0, 1})
variable (C : Set ℤ := {-2, 0, 2})

theorem part1 : A ∪ (B ∩ C) = {0, 1, 2, 3} := by
  sorry

theorem part2 : A ∩ Uᶜ ∪ (B ∪ C) = {3} := by
  sorry

end part1_part2_l1583_158396


namespace reduced_rate_fraction_l1583_158386

-- Definitions
def hours_in_a_week := 7 * 24
def hours_with_reduced_rates_on_weekdays := (12 * 5)
def hours_with_reduced_rates_on_weekends := (24 * 2)

-- Question in form of theorem
theorem reduced_rate_fraction :
  (hours_with_reduced_rates_on_weekdays + hours_with_reduced_rates_on_weekends) / hours_in_a_week = 9 / 14 := 
by
  sorry

end reduced_rate_fraction_l1583_158386


namespace karen_age_is_10_l1583_158392

-- Definitions for the given conditions
def ages : List ℕ := [2, 4, 6, 8, 10, 12, 14]

def to_park (a b : ℕ) : Prop := a + b = 20
def to_pool (a b : ℕ) : Prop := 3 < a ∧ a < 9 ∧ 3 < b ∧ b < 9
def stayed_home (karen_age : ℕ) : Prop := karen_age = 10

-- Theorem stating Karen's age is 10 given the conditions
theorem karen_age_is_10 :
  ∃ (a b c d e f g : ℕ),
  ages = [a, b, c, d, e, f, g] ∧
  ((to_park a b ∨ to_park a c ∨ to_park a d ∨ to_park a e ∨ to_park a f ∨ to_park a g ∨
  to_park b c ∨ to_park b d ∨ to_park b e ∨ to_park b f ∨ to_park b g ∨
  to_park c d ∨ to_park c e ∨ to_park c f ∨ to_park c g ∨
  to_park d e ∨ to_park d f ∨ to_park d g ∨
  to_park e f ∨ to_park e g ∨
  to_park f g)) ∧
  ((to_pool a b ∨ to_pool a c ∨ to_pool a d ∨ to_pool a e ∨ to_pool a f ∨ to_pool a g ∨
  to_pool b c ∨ to_pool b d ∨ to_pool b e ∨ to_pool b f ∨ to_pool b g ∨
  to_pool c d ∨ to_pool c e ∨ to_pool c f ∨
  to_pool d e ∨ to_pool d f ∨
  to_pool e f ∨
  to_pool f g)) ∧
  stayed_home 4 :=
sorry

end karen_age_is_10_l1583_158392


namespace price_reduction_for_1920_profit_maximum_profit_calculation_l1583_158356

-- Definitions based on given conditions
def cost_price : ℝ := 12
def base_price : ℝ := 20
def base_quantity_sold : ℝ := 240
def increment_per_dollar : ℝ := 40

-- Profit function
def profit (x : ℝ) : ℝ := (base_price - cost_price - x) * (base_quantity_sold + increment_per_dollar * x)

-- Prove price reduction for $1920 profit per day
theorem price_reduction_for_1920_profit : ∃ x : ℝ, profit x = 1920 ∧ x = 8 := by
  sorry

-- Prove maximum profit calculation
theorem maximum_profit_calculation : ∃ x y : ℝ, x = 4 ∧ y = 2560 ∧ ∀ z, profit z ≤ y := by
  sorry

end price_reduction_for_1920_profit_maximum_profit_calculation_l1583_158356


namespace part1_part2_l1583_158399

open Set

variable {R : Type} [OrderedRing R]

def U : Set R := univ
def A : Set R := {x | x^2 - 2*x - 3 > 0}
def B : Set R := {x | 4 - x^2 <= 0}

theorem part1 : A ∩ B = {x | -2 ≤ x ∧ x < -1} :=
sorry

theorem part2 : (U \ A) ∪ (U \ B) = {x | x < -2 ∨ x > -1} :=
sorry

end part1_part2_l1583_158399


namespace pattern_generalization_l1583_158362

theorem pattern_generalization (n : ℕ) (h : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 :=
by
  -- TODO: The proof will be filled in later
  sorry

end pattern_generalization_l1583_158362


namespace k_value_if_function_not_in_first_quadrant_l1583_158321

theorem k_value_if_function_not_in_first_quadrant : 
  ∀ k : ℝ, (∀ x : ℝ, x > 0 → (k - 2) * x ^ (|k|) + k ≤ 0) → k = -1 :=
by
  sorry

end k_value_if_function_not_in_first_quadrant_l1583_158321


namespace min_value_expression_l1583_158329

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2/x + 3/y = 1) :
  x/2 + y/3 = 4 :=
sorry

end min_value_expression_l1583_158329


namespace train_has_96_cars_l1583_158331

def train_cars_count (cars_in_15_seconds : Nat) (time_for_15_seconds : Nat) (total_time_seconds : Nat) : Nat :=
  total_time_seconds * cars_in_15_seconds / time_for_15_seconds

theorem train_has_96_cars :
  train_cars_count 8 15 180 = 96 :=
by
  sorry

end train_has_96_cars_l1583_158331


namespace juwella_read_more_last_night_l1583_158382

-- Definitions of the conditions
def pages_three_nights_ago : ℕ := 15
def book_pages : ℕ := 100
def pages_tonight : ℕ := 20
def pages_two_nights_ago : ℕ := 2 * pages_three_nights_ago
def total_pages_before_tonight : ℕ := book_pages - pages_tonight
def pages_last_night : ℕ := total_pages_before_tonight - pages_three_nights_ago - pages_two_nights_ago

theorem juwella_read_more_last_night :
  pages_last_night - pages_two_nights_ago = 5 :=
by
  sorry

end juwella_read_more_last_night_l1583_158382


namespace smallest_positive_perfect_cube_has_divisor_l1583_158302

theorem smallest_positive_perfect_cube_has_divisor (p q r s : ℕ) (hp : Prime p) (hq : Prime q)
  (hr : Prime r) (hs : Prime s) (hpqrs : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
  ∃ n : ℕ, n = (p * q * r * s^2)^3 ∧ ∀ m : ℕ, (m = p^2 * q^3 * r^4 * s^5 → m ∣ n) :=
by
  sorry

end smallest_positive_perfect_cube_has_divisor_l1583_158302


namespace reconstruct_quadrilateral_l1583_158340

def quadrilateralVectors (W W' X X' Y Y' Z Z' : ℝ) :=
  (W - Z = W/2 + Z'/2) ∧
  (X - Y = Y'/2 + X'/2) ∧
  (Y - X = Y'/2 + X'/2) ∧
  (Z - W = W/2 + Z'/2)

theorem reconstruct_quadrilateral (W W' X X' Y Y' Z Z' : ℝ) :
  quadrilateralVectors W W' X X' Y Y' Z Z' →
  W = (1/2) * W' + 0 * X' + 0 * Y' + (1/2) * Z' :=
sorry

end reconstruct_quadrilateral_l1583_158340


namespace elsa_data_remaining_l1583_158369

variable (data_total : ℕ) (data_youtube : ℕ)

def data_remaining_after_youtube (data_total data_youtube : ℕ) : ℕ := data_total - data_youtube

def data_fraction_spent_on_facebook (data_left : ℕ) : ℕ := (2 * data_left) / 5

theorem elsa_data_remaining
  (h_data_total : data_total = 500)
  (h_data_youtube : data_youtube = 300) :
  data_remaining_after_youtube data_total data_youtube
  - data_fraction_spent_on_facebook (data_remaining_after_youtube data_total data_youtube) 
  = 120 :=
by
  sorry

end elsa_data_remaining_l1583_158369


namespace xyz_value_l1583_158334

theorem xyz_value (x y z : ℚ)
  (h1 : x + y + z = 1)
  (h2 : x + y - z = 2)
  (h3 : x - y - z = 3) :
  x * y * z = 1/2 :=
by
  sorry

end xyz_value_l1583_158334


namespace purpose_of_LB_full_nutrient_medium_l1583_158315

/--
Given the experiment "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source",
which involves both experimental and control groups with the following conditions:
- The variable in the experiment is the difference in the medium used.
- The experimental group uses a medium with urea as the only nitrogen source (selective medium).
- The control group uses a full-nutrient medium.

Prove that the purpose of preparing LB full-nutrient medium is to observe the types and numbers
of soil microorganisms that can grow under full-nutrient conditions.
-/
theorem purpose_of_LB_full_nutrient_medium
  (experiment: String) (experimental_variable: String) (experimental_group: String) (control_group: String)
  (H1: experiment = "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source")
  (H2: experimental_variable = "medium")
  (H3: experimental_group = "medium with urea as the only nitrogen source (selective medium)")
  (H4: control_group = "full-nutrient medium") :
  purpose_of_preparing_LB_full_nutrient_medium = "observe the types and numbers of soil microorganisms that can grow under full-nutrient conditions" :=
sorry

end purpose_of_LB_full_nutrient_medium_l1583_158315


namespace coupon_probability_l1583_158348

theorem coupon_probability : 
  (Nat.choose 6 6 * Nat.choose 11 3) / Nat.choose 17 9 = 3 / 442 := 
by
  sorry

end coupon_probability_l1583_158348


namespace n_decomposable_form_l1583_158384

theorem n_decomposable_form (n : ℕ) (a : ℕ) (h₁ : a > 2) (h₂ : ∃ k, 1 < k ∧ n = 2^k) :
  (∀ d : ℕ, d ∣ n ∧ d ≠ n → (a^n - 2^n) % (a^d + 2^d) = 0) → ∃ k, 1 < k ∧ n = 2^k :=
by {
  sorry
}

end n_decomposable_form_l1583_158384


namespace remainder_of_3_pow_99_plus_5_mod_9_l1583_158345

theorem remainder_of_3_pow_99_plus_5_mod_9 : (3 ^ 99 + 5) % 9 = 5 := by
  -- Here we state the main goal
  sorry -- Proof to be filled in

end remainder_of_3_pow_99_plus_5_mod_9_l1583_158345


namespace class_raised_initial_amount_l1583_158347

/-- Miss Grayson's class raised some money for their field trip.
Each student contributed $5 each.
There are 20 students in her class.
The cost of the trip is $7 for each student.
After all the field trip costs were paid, there is $10 left in Miss Grayson's class fund.
Prove that the class initially raised $150 for the field trip. -/
theorem class_raised_initial_amount
  (students : ℕ)
  (contribution_per_student : ℕ)
  (cost_per_student : ℕ)
  (remaining_fund : ℕ)
  (total_students : students = 20)
  (per_student_contribution : contribution_per_student = 5)
  (per_student_cost : cost_per_student = 7)
  (remaining_amount : remaining_fund = 10) :
  (students * contribution_per_student + remaining_fund) = 150 := 
sorry

end class_raised_initial_amount_l1583_158347


namespace ratio_problem_l1583_158368

variable (a b c d : ℝ)

theorem ratio_problem (h1 : a / b = 3) (h2 : b / c = 1 / 4) (h3 : c / d = 5) : d / a = 4 / 15 := 
sorry

end ratio_problem_l1583_158368


namespace race_distance_A_beats_C_l1583_158336

variables (race_distance1 race_distance2 race_distance3 : ℕ)
           (distance_AB distance_BC distance_AC : ℕ)

theorem race_distance_A_beats_C :
  race_distance1 = 500 →
  race_distance2 = 500 →
  distance_AB = 50 →
  distance_BC = 25 →
  distance_AC = 58 →
  race_distance3 = 400 :=
by
  sorry

end race_distance_A_beats_C_l1583_158336


namespace solution_set_for_f_gt_0_l1583_158311

noncomputable def f (x : ℝ) : ℝ := sorry

theorem solution_set_for_f_gt_0
  (odd_f : ∀ x : ℝ, f (-x) = -f x)
  (f_one_eq_zero : f 1 = 0)
  (ineq_f : ∀ x : ℝ, x > 0 → (x * (deriv^[2] f x) - f x) / x^2 > 0) :
  { x : ℝ | f x > 0 } = { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x } :=
sorry

end solution_set_for_f_gt_0_l1583_158311


namespace average_lifespan_is_1013_l1583_158317

noncomputable def first_factory_lifespan : ℕ := 980
noncomputable def second_factory_lifespan : ℕ := 1020
noncomputable def third_factory_lifespan : ℕ := 1032

noncomputable def total_samples : ℕ := 100

noncomputable def first_samples : ℕ := (1 * total_samples) / 4
noncomputable def second_samples : ℕ := (2 * total_samples) / 4
noncomputable def third_samples : ℕ := (1 * total_samples) / 4

noncomputable def weighted_average_lifespan : ℕ :=
  ((first_factory_lifespan * first_samples) + (second_factory_lifespan * second_samples) + (third_factory_lifespan * third_samples)) / total_samples

theorem average_lifespan_is_1013 : weighted_average_lifespan = 1013 := by
  sorry

end average_lifespan_is_1013_l1583_158317


namespace problem_l1583_158378

variable (α : ℝ)

def setA : Set ℝ := {Real.sin α, Real.cos α, 1}
def setB : Set ℝ := {Real.sin α ^ 2, Real.sin α + Real.cos α, 0}
theorem problem (h : setA α = setB α) : Real.sin α ^ 2009 + Real.cos α ^ 2009 = -1 := 
by 
  sorry

end problem_l1583_158378


namespace find_M_l1583_158301

theorem find_M (x y z M : ℝ) 
  (h1 : x + y + z = 120) 
  (h2 : x - 10 = M) 
  (h3 : y + 10 = M) 
  (h4 : z / 10 = M) : 
  M = 10 := 
by
  sorry

end find_M_l1583_158301


namespace no_infinite_seq_pos_int_l1583_158327

theorem no_infinite_seq_pos_int : 
  ¬∃ (a : ℕ → ℕ), 
  (∀ n : ℕ, 0 < a n) ∧ 
  ∀ n : ℕ, a (n+1) ^ 2 ≥ 2 * a n * a (n+2) :=
by
  sorry

end no_infinite_seq_pos_int_l1583_158327


namespace max_three_digit_sum_l1583_158300

theorem max_three_digit_sum (A B C : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) :
  110 * A + 10 * B + 3 * C ≤ 981 :=
sorry

end max_three_digit_sum_l1583_158300


namespace expression_zero_denominator_nonzero_l1583_158397

theorem expression_zero (x : ℝ) : 
  (2 * x - 6) = 0 ↔ x = 3 :=
by {
  sorry
  }

theorem denominator_nonzero (x : ℝ) : 
  x = 3 → (5 * x + 10) ≠ 0 :=
by {
  sorry
  }

end expression_zero_denominator_nonzero_l1583_158397


namespace work_done_days_l1583_158357

theorem work_done_days (a_days : ℕ) (b_days : ℕ) (together_days : ℕ) (a_work_done : ℚ) (b_work_done : ℚ) (together_work : ℚ) : 
  a_days = 12 ∧ b_days = 15 ∧ together_days = 5 ∧ 
  a_work_done = 1/12 ∧ b_work_done = 1/15 ∧ together_work = 3/4 → 
  ∃ days : ℚ, a_days > 0 ∧ b_days > 0 ∧ together_days > 0 ∧ days = 3 := 
  sorry

end work_done_days_l1583_158357


namespace boxes_needed_l1583_158322

theorem boxes_needed (total_oranges boxes_capacity : ℕ) (h1 : total_oranges = 94) (h2 : boxes_capacity = 8) : 
  (total_oranges + boxes_capacity - 1) / boxes_capacity = 12 := 
by
  sorry

end boxes_needed_l1583_158322


namespace max_value_inequality_l1583_158342

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (abc * (a + b + c)) / ((a + b)^2 * (b + c)^3) ≤ 1 / 4 :=
sorry

end max_value_inequality_l1583_158342


namespace sum_fib_2019_eq_fib_2021_minus_1_l1583_158351

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

def sum_fib : ℕ → ℕ
| 0 => 0
| n + 1 => sum_fib n + fib (n + 1)

theorem sum_fib_2019_eq_fib_2021_minus_1 : sum_fib 2019 = fib 2021 - 1 := 
by sorry -- proof here

end sum_fib_2019_eq_fib_2021_minus_1_l1583_158351


namespace equip_20posts_with_5new_weapons_l1583_158394

/-- 
Theorem: In a line of 20 defense posts, the number of ways to equip 5 different new weapons 
such that:
1. The first and last posts are not equipped with new weapons.
2. Each set of 5 consecutive posts has at least one post equipped with a new weapon.
3. No two adjacent posts are equipped with new weapons.
is 69600. 
-/
theorem equip_20posts_with_5new_weapons : ∃ ways : ℕ, ways = 69600 :=
by
  sorry

end equip_20posts_with_5new_weapons_l1583_158394


namespace bricks_needed_to_build_wall_l1583_158376

def volume_of_brick (length_brick height_brick thickness_brick : ℤ) : ℤ :=
  length_brick * height_brick * thickness_brick

def volume_of_wall (length_wall height_wall thickness_wall : ℤ) : ℤ :=
  length_wall * height_wall * thickness_wall

def number_of_bricks_needed (length_wall height_wall thickness_wall length_brick height_brick thickness_brick : ℤ) : ℤ :=
  (volume_of_wall length_wall height_wall thickness_wall + volume_of_brick length_brick height_brick thickness_brick - 1) / 
  volume_of_brick length_brick height_brick thickness_brick

theorem bricks_needed_to_build_wall : number_of_bricks_needed 800 100 5 25 11 6 = 243 := 
  by 
    sorry

end bricks_needed_to_build_wall_l1583_158376


namespace polygon_sides_l1583_158338

theorem polygon_sides (h1 : 1260 - 360 = 900) (h2 : (n - 2) * 180 = 900) : n = 7 :=
by 
  sorry

end polygon_sides_l1583_158338


namespace maria_gave_towels_l1583_158352

def maria_towels (green_white total_left : Nat) : Nat :=
  green_white - total_left

theorem maria_gave_towels :
  ∀ (green white left given : Nat),
    green = 35 →
    white = 21 →
    left = 22 →
    given = 34 →
    maria_towels (green + white) left = given :=
by
  intros green white left given
  intros hgreen hwhite hleft hgiven
  rw [hgreen, hwhite, hleft, hgiven]
  sorry

end maria_gave_towels_l1583_158352


namespace cos_C_l1583_158387

-- Define the data and conditions of the problem
variables {A B C : ℝ}
variables (triangle_ABC : Prop)
variable (h_sinA : Real.sin A = 4 / 5)
variable (h_cosB : Real.cos B = 12 / 13)

-- Statement of the theorem
theorem cos_C (h1 : triangle_ABC)
  (h2 : Real.sin A = 4 / 5)
  (h3 : Real.cos B = 12 / 13) :
  Real.cos C = -16 / 65 :=
sorry

end cos_C_l1583_158387


namespace find_line_eq_l1583_158313

-- Define the type for the line equation
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def given_point : ℝ × ℝ := (-3, -1)
def given_parallel_line : Line := { a := 1, b := -3, c := -1 }

-- Define what it means for two lines to be parallel
def are_parallel (L1 L2 : Line) : Prop :=
  L1.a * L2.b = L1.b * L2.a

-- Define what it means for a point to lie on the line
def lies_on_line (P : ℝ × ℝ) (L : Line) : Prop :=
  L.a * P.1 + L.b * P.2 + L.c = 0

-- Define the result line we need to prove
def result_line : Line := { a := 1, b := -3, c := 0 }

-- The final theorem statement
theorem find_line_eq : 
  ∃ (L : Line), are_parallel L given_parallel_line ∧ lies_on_line given_point L ∧ L = result_line := 
sorry

end find_line_eq_l1583_158313


namespace find_angle_A_l1583_158343

theorem find_angle_A (a b c : ℝ) (A : ℝ) (h : a^2 = b^2 - b * c + c^2) : A = 60 :=
sorry

end find_angle_A_l1583_158343


namespace smallest_n_such_that_floor_eq_1989_l1583_158393

theorem smallest_n_such_that_floor_eq_1989 :
  ∃ (n : ℕ), (∀ k, k < n -> ¬(∃ x : ℤ, ⌊(10^k : ℚ) / x⌋ = 1989)) ∧ (∃ x : ℤ, ⌊(10^n : ℚ) / x⌋ = 1989) :=
sorry

end smallest_n_such_that_floor_eq_1989_l1583_158393


namespace panda_bamboo_digestion_l1583_158373

theorem panda_bamboo_digestion (h : 16 = 0.40 * x) : x = 40 :=
by sorry

end panda_bamboo_digestion_l1583_158373


namespace side_length_square_base_l1583_158344

theorem side_length_square_base 
  (height : ℕ) (volume : ℕ) (A : ℕ) (s : ℕ) 
  (h_height : height = 8) 
  (h_volume : volume = 288) 
  (h_base_area : A = volume / height) 
  (h_square_base : A = s ^ 2) :
  s = 6 :=
by
  sorry

end side_length_square_base_l1583_158344


namespace segments_in_proportion_l1583_158314

theorem segments_in_proportion (a b c d : ℝ) (ha : a = 1) (hb : b = 4) (hc : c = 2) (h : a / b = c / d) : d = 8 := 
by 
  sorry

end segments_in_proportion_l1583_158314


namespace inequality_solution_l1583_158383

theorem inequality_solution {x : ℝ} : (x + 1) / x > 1 ↔ x > 0 := 
sorry

end inequality_solution_l1583_158383


namespace probability_no_consecutive_tails_probability_no_consecutive_tails_in_five_tosses_l1583_158370

def countWays (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else countWays (n - 1) + countWays (n - 2)

theorem probability_no_consecutive_tails : countWays 5 = 13 :=
by
  sorry

theorem probability_no_consecutive_tails_in_five_tosses : 
  (countWays 5) / (2^5 : ℕ) = 13 / 32 :=
by
  sorry

end probability_no_consecutive_tails_probability_no_consecutive_tails_in_five_tosses_l1583_158370


namespace hcf_of_two_numbers_l1583_158335

theorem hcf_of_two_numbers (H L P : ℕ) (h1 : L = 160) (h2 : P = 2560) (h3 : H * L = P) : H = 16 :=
by
  sorry

end hcf_of_two_numbers_l1583_158335


namespace initial_number_of_people_l1583_158330

theorem initial_number_of_people (P : ℕ) : P * 10 = (P + 1) * 5 → P = 1 :=
by sorry

end initial_number_of_people_l1583_158330


namespace proof_problem_l1583_158364

-- Define the given condition as a constant
def condition : Prop := 213 * 16 = 3408

-- Define the statement we need to prove under the given condition
theorem proof_problem (h : condition) : 0.16 * 2.13 = 0.3408 := 
by 
  sorry

end proof_problem_l1583_158364


namespace find_total_worth_of_stock_l1583_158377

theorem find_total_worth_of_stock (X : ℝ)
  (h1 : 0.20 * X * 0.10 = 0.02 * X)
  (h2 : 0.80 * X * 0.05 = 0.04 * X)
  (h3 : 0.04 * X - 0.02 * X = 200) :
  X = 10000 :=
sorry

end find_total_worth_of_stock_l1583_158377


namespace boxes_in_carton_l1583_158389

theorem boxes_in_carton (cost_per_pack : ℕ) (packs_per_box : ℕ) (cost_dozen_cartons : ℕ) 
  (h1 : cost_per_pack = 1) (h2 : packs_per_box = 10) (h3 : cost_dozen_cartons = 1440) :
  (cost_dozen_cartons / 12) / (cost_per_pack * packs_per_box) = 12 :=
by
  sorry

end boxes_in_carton_l1583_158389


namespace books_not_sold_l1583_158319

theorem books_not_sold (X : ℕ) (H1 : (2/3 : ℝ) * X * 4 = 288) : (1 / 3 : ℝ) * X = 36 :=
by
  -- Proof goes here
  sorry

end books_not_sold_l1583_158319


namespace find_triples_l1583_158304

theorem find_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 = c^2) ∧ (a^3 + b^3 + 1 = (c-1)^3) ↔ (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10) :=
by
  sorry

end find_triples_l1583_158304


namespace sqrt_9_eq_pos_neg_3_l1583_158363

theorem sqrt_9_eq_pos_neg_3 : ∀ x : ℝ, x^2 = 9 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end sqrt_9_eq_pos_neg_3_l1583_158363


namespace john_days_to_lose_weight_l1583_158346

noncomputable def john_calories_intake : ℕ := 1800
noncomputable def john_calories_burned : ℕ := 2300
noncomputable def calories_to_lose_1_pound : ℕ := 4000
noncomputable def pounds_to_lose : ℕ := 10

theorem john_days_to_lose_weight :
  (john_calories_burned - john_calories_intake) * (pounds_to_lose * calories_to_lose_1_pound / (john_calories_burned - john_calories_intake)) = 80 :=
by
  sorry

end john_days_to_lose_weight_l1583_158346


namespace intersection_A_B_l1583_158349

def A : Set ℝ := {x | x < 3 * x - 1}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_A_B : (A ∩ B) = {x | x > 1 / 2 ∧ x < 3} :=
by sorry

end intersection_A_B_l1583_158349


namespace sin_cos_sum_l1583_158358

theorem sin_cos_sum (α x y r : ℝ) (h1 : x = 2) (h2 : y = -1) (h3 : r = Real.sqrt 5)
    (h4 : ∀ θ, x = r * Real.cos θ) (h5 : ∀ θ, y = r * Real.sin θ) : 
    Real.sin α + Real.cos α = (- 1 / Real.sqrt 5) + (2 / Real.sqrt 5) :=
by
  sorry

end sin_cos_sum_l1583_158358


namespace present_age_ratio_l1583_158308

-- Define the variables and the conditions
variable (S M : ℕ)

-- Condition 1: Sandy's present age is 84 because she was 78 six years ago
def present_age_sandy := S = 84

-- Condition 2: Sixteen years from now, the ratio of their ages is 5:2
def age_ratio_16_years := (S + 16) * 2 = 5 * (M + 16)

-- The goal: The present age ratio of Sandy to Molly is 7:2
theorem present_age_ratio {S M : ℕ} (h1 : S = 84) (h2 : (S + 16) * 2 = 5 * (M + 16)) : S / M = 7 / 2 :=
by
  -- Integrating conditions
  have hS : S = 84 := h1
  have hR : (S + 16) * 2 = 5 * (M + 16) := h2
  -- We need a proof here, but we'll skip it for now
  sorry

end present_age_ratio_l1583_158308


namespace sin_double_angle_l1583_158332

theorem sin_double_angle (α : ℝ) (h : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = -2 / 3 := 
sorry

end sin_double_angle_l1583_158332


namespace xiaozhang_participates_in_martial_arts_l1583_158372

theorem xiaozhang_participates_in_martial_arts
  (row : Prop) (shoot : Prop) (martial : Prop)
  (Zhang Wang Li: Prop → Prop)
  (H1 : ¬  Zhang row ∧ ¬ Wang row)
  (H2 : ∃ (n m : ℕ), Zhang (shoot ∨ martial) = (n > 0) ∧ Wang (shoot ∨ martial) = (m > 0) ∧ m = n + 1)
  (H3 : ¬ Li shoot ∧ (Li martial ∨ Li row)) :
  Zhang martial :=
by
  sorry

end xiaozhang_participates_in_martial_arts_l1583_158372


namespace smallest_five_digit_congruent_two_mod_seventeen_l1583_158380

theorem smallest_five_digit_congruent_two_mod_seventeen : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 2 ∧ n = 10013 :=
by
  sorry

end smallest_five_digit_congruent_two_mod_seventeen_l1583_158380


namespace leopards_points_l1583_158305

variables (x y : ℕ)

theorem leopards_points (h₁ : x + y = 50) (h₂ : x - y = 28) : y = 11 := by
  sorry

end leopards_points_l1583_158305


namespace sandra_money_left_l1583_158367

def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def candy_cost : ℚ := 0.5
def jelly_bean_cost : ℚ := 0.2
def num_candies : ℕ := 14
def num_jelly_beans : ℕ := 20

def total_money : ℕ := sandra_savings + mother_gift + father_gift
def total_candy_cost : ℚ := num_candies * candy_cost
def total_jelly_bean_cost : ℚ := num_jelly_beans * jelly_bean_cost
def total_cost : ℚ := total_candy_cost + total_jelly_bean_cost
def money_left : ℚ := total_money - total_cost

theorem sandra_money_left : money_left = 11 := by
  sorry

end sandra_money_left_l1583_158367


namespace part1_point_A_value_of_m_part1_area_ABC_part2_max_ordinate_P_l1583_158354

noncomputable def quadratic_function (m x : ℝ) : ℝ := (m - 2) * x ^ 2 - x - m ^ 2 + 6 * m - 7

theorem part1_point_A_value_of_m (m : ℝ) (h : quadratic_function m (-1) = 2) : m = 5 :=
sorry

theorem part1_area_ABC (area : ℝ) 
  (h₁ : quadratic_function 5 (1 : ℝ) = 0) 
  (h₂ : quadratic_function 5 (-2/3 : ℝ) = 0) : area = 5 / 3 :=
sorry

theorem part2_max_ordinate_P (m : ℝ) (h : - (m - 3) ^ 2 + 2 ≤ 2) : m = 3 :=
sorry

end part1_point_A_value_of_m_part1_area_ABC_part2_max_ordinate_P_l1583_158354


namespace good_deed_done_by_C_l1583_158365

def did_good (A B C : Prop) := 
  (¬A ∧ ¬B ∧ C) ∨ (¬A ∧ B ∧ ¬C) ∨ (A ∧ ¬B ∧ ¬C)

def statement_A (B : Prop) := B
def statement_B (B : Prop) := ¬B
def statement_C (C : Prop) := ¬C

theorem good_deed_done_by_C (A B C : Prop)
  (h_deed : (did_good A B C))
  (h_statement : (statement_A B ∧ ¬statement_B B ∧ ¬statement_C C) ∨ 
                      (¬statement_A B ∧ statement_B B ∧ ¬statement_C C) ∨ 
                      (¬statement_A B ∧ ¬statement_B B ∧ statement_C C)) :
  C :=
by 
  sorry

end good_deed_done_by_C_l1583_158365


namespace range_of_BD_l1583_158375

-- Define the types of points and triangle
variables {α : Type*} [MetricSpace α]

-- Hypothesis: AD is the median of triangle ABC
-- Definition of lengths AB, AC, and that BD = CD.
def isMedianOnBC (A B C D : α) : Prop :=
  dist A B = 5 ∧ dist A C = 7 ∧ dist B D = dist C D

-- The theorem to be proven
theorem range_of_BD {A B C D : α} (h : isMedianOnBC A B C D) : 
  1 < dist B D ∧ dist B D < 6 :=
by
  sorry

end range_of_BD_l1583_158375


namespace max_cylinder_volume_l1583_158307

/-- Given a rectangle with perimeter 18 cm, when rotating it around one side to form a cylinder, 
    the maximum volume of the cylinder and the corresponding side length of the rectangle. -/
theorem max_cylinder_volume (x y : ℝ) (h_perimeter : 2 * (x + y) = 18) (hx : x > 0) (hy : y > 0)
  (h_cylinder_volume : ∃ (V : ℝ), V = π * x * (y / 2)^2) :
  (x = 3 ∧ y = 6 ∧ ∀ V, V = 108 * π) := sorry

end max_cylinder_volume_l1583_158307


namespace circle_radius_c_value_l1583_158359

theorem circle_radius_c_value (x y c : ℝ) (h₁ : x^2 + 8 * x + y^2 + 10 * y + c = 0) (h₂ : (x+4)^2 + (y+5)^2 = 25) :
  c = -16 :=
by sorry

end circle_radius_c_value_l1583_158359


namespace election_winner_votes_l1583_158309

theorem election_winner_votes (V : ℝ) : (0.62 * V = 806) → (0.62 * V) - (0.38 * V) = 312 → 0.62 * V = 806 :=
by
  intro hWin hDiff
  exact hWin

end election_winner_votes_l1583_158309


namespace count_four_digit_numbers_with_digit_sum_4_l1583_158337

theorem count_four_digit_numbers_with_digit_sum_4 : 
  ∃ n : ℕ, (∀ (x1 x2 x3 x4 : ℕ), 
    x1 + x2 + x3 + x4 = 4 ∧ x1 ≥ 1 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0 →
    n = 20) :=
sorry

end count_four_digit_numbers_with_digit_sum_4_l1583_158337


namespace swimming_pool_width_l1583_158316

theorem swimming_pool_width (L D1 D2 V : ℝ) (W : ℝ) (h : L = 12) (h1 : D1 = 1) (h2 : D2 = 4) (hV : V = 270) : W = 9 :=
  by
    -- We begin by stating the formula for the volume of 
    -- a trapezoidal prism: Volume = (1/2) * (D1 + D2) * L * W
    
    -- According to the problem, we have the following conditions:
    have hVolume : V = (1/2) * (D1 + D2) * L * W :=
      by sorry

    -- Substitute the provided values into the volume equation:
    -- 270 = (1/2) * (1 + 4) * 12 * W
    
    -- Simplify and solve for W
    simp at hVolume
    exact sorry

end swimming_pool_width_l1583_158316


namespace blue_markers_count_l1583_158328

-- Definitions based on given conditions
def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

-- Main statement to prove
theorem blue_markers_count : total_markers - red_markers = 1028 := by
  sorry

end blue_markers_count_l1583_158328
