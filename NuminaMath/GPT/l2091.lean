import Mathlib

namespace ax_plus_by_equals_d_set_of_solutions_l2091_209124

theorem ax_plus_by_equals_d (a b d : ℤ) (u v : ℤ) (h_d : d = a.gcd b) (h_uv : a * u + b * v = d) :
  ∀ (x y : ℤ), (a * x + b * y = d) ↔ ∃ k : ℤ, x = u + k * b ∧ y = v - k * a :=
by
  sorry

theorem set_of_solutions (a b d : ℤ) (u v : ℤ) (h_d : d = a.gcd b) (h_uv : a * u + b * v = d) :
  {p : ℤ × ℤ | a * p.1 + b * p.2 = d} = {p : ℤ × ℤ | ∃ k : ℤ, p = (u + k * b, v - k * a)} :=
by
  sorry

end ax_plus_by_equals_d_set_of_solutions_l2091_209124


namespace cos_2beta_proof_l2091_209155

theorem cos_2beta_proof (α β : ℝ)
  (h1 : Real.sin (α - β) = 3 / 5)
  (h2 : Real.sin (α + β) = -3 / 5)
  (h3 : α - β ∈ Set.Ioo (π / 2) π)
  (h4 : α + β ∈ Set.Ioo (3 * π / 2) (2 * π)) :
  Real.cos (2 * β) = -7 / 25 :=
by
  sorry

end cos_2beta_proof_l2091_209155


namespace diet_soda_ratio_l2091_209102

def total_bottles : ℕ := 60
def diet_soda_bottles : ℕ := 14

theorem diet_soda_ratio : (diet_soda_bottles * 30) = (total_bottles * 7) :=
by {
  -- We're given that total_bottles = 60 and diet_soda_bottles = 14
  -- So to prove the ratio 14/60 is equivalent to 7/30:
  -- Multiplying both sides by 30 and 60 simplifies the arithmetic.
  sorry
}

end diet_soda_ratio_l2091_209102


namespace red_tint_percentage_new_mixture_l2091_209144

-- Definitions of the initial conditions
def original_volume : ℝ := 50
def red_tint_percentage : ℝ := 0.20
def added_red_tint : ℝ := 6

-- Definition for the proof
theorem red_tint_percentage_new_mixture : 
  let original_red_tint := red_tint_percentage * original_volume
  let new_red_tint := original_red_tint + added_red_tint
  let new_total_volume := original_volume + added_red_tint
  (new_red_tint / new_total_volume) * 100 = 28.57 :=
by
  sorry

end red_tint_percentage_new_mixture_l2091_209144


namespace largest_no_solution_l2091_209164

theorem largest_no_solution (a : ℕ) (h_odd : a % 2 = 1) (h_pos : a > 0) :
  ∃ n : ℕ, ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → a * x + (a + 1) * y + (a + 2) * z ≠ n :=
sorry

end largest_no_solution_l2091_209164


namespace largest_square_side_length_l2091_209103

noncomputable def largestInscribedSquareSide (s : ℝ) (sharedSide : ℝ) : ℝ :=
  let y := (s * Real.sqrt 2 - sharedSide * Real.sqrt 3) / (2 * Real.sqrt 2)
  y

theorem largest_square_side_length :
  let s := 12
  let t := (s * Real.sqrt 6) / 3
  largestInscribedSquareSide s t = 6 - Real.sqrt 6 :=
by
  sorry

end largest_square_side_length_l2091_209103


namespace pilot_fish_final_speed_relative_to_ocean_l2091_209171

-- Define conditions
def keanu_speed : ℝ := 20 -- Keanu's speed in mph
def wind_speed : ℝ := 5 -- Wind speed in mph
def shark_speed (initial_speed: ℝ) : ℝ := 2 * initial_speed -- Shark doubles its speed

-- The pilot fish increases its speed by half the shark's increase
def pilot_fish_speed (initial_pilot_fish_speed shark_initial_speed : ℝ) : ℝ := 
  initial_pilot_fish_speed + 0.5 * shark_initial_speed

-- Define the speed of the pilot fish relative to the ocean
def pilot_fish_speed_relative_to_ocean (initial_pilot_fish_speed shark_initial_speed : ℝ) : ℝ := 
  pilot_fish_speed initial_pilot_fish_speed shark_initial_speed - wind_speed

-- Initial assumptions
def initial_pilot_fish_speed : ℝ := keanu_speed -- Pilot fish initially swims at the same speed as Keanu
def initial_shark_speed : ℝ := keanu_speed -- Let us assume the shark initially swims at the same speed as Keanu for simplicity

-- Prove the final speed of the pilot fish relative to the ocean
theorem pilot_fish_final_speed_relative_to_ocean : 
  pilot_fish_speed_relative_to_ocean initial_pilot_fish_speed initial_shark_speed = 25 := 
by sorry

end pilot_fish_final_speed_relative_to_ocean_l2091_209171


namespace fraction_simplification_l2091_209107

theorem fraction_simplification (x y : ℚ) (h1 : x = 4) (h2 : y = 5) : 
  (1 / y) / (1 / x) = 4 / 5 :=
by
  sorry

end fraction_simplification_l2091_209107


namespace original_number_count_l2091_209130

theorem original_number_count (k S : ℕ) (M : ℚ)
  (hk : k > 0)
  (hM : M = S / k)
  (h_add15 : (S + 15) / (k + 1) = M + 2)
  (h_add1 : (S + 16) / (k + 2) = M + 1) :
  k = 6 :=
by
  -- Proof will go here
  sorry

end original_number_count_l2091_209130


namespace roots_ratio_sum_eq_six_l2091_209189

theorem roots_ratio_sum_eq_six (x1 x2 : ℝ) (h1 : 2 * x1^2 - 4 * x1 + 1 = 0) (h2 : 2 * x2^2 - 4 * x2 + 1 = 0) :
  (x1 / x2) + (x2 / x1) = 6 :=
sorry

end roots_ratio_sum_eq_six_l2091_209189


namespace find_number_l2091_209125

-- Definitions of the fractions involved
def frac_2_15 : ℚ := 2 / 15
def frac_1_5 : ℚ := 1 / 5
def frac_1_2 : ℚ := 1 / 2

-- Condition that the number is greater than the sum of frac_2_15 and frac_1_5 by frac_1_2 
def number : ℚ := frac_2_15 + frac_1_5 + frac_1_2

-- Theorem statement matching the math proof problem
theorem find_number : number = 5 / 6 :=
by
  sorry

end find_number_l2091_209125


namespace remainder_sum_1_to_12_div_9_l2091_209191

-- Define the sum of the first n natural numbers
def sum_natural (n : Nat) : Nat := n * (n + 1) / 2

-- Define the sum of the numbers from 1 to 12
def sum_1_to_12 := sum_natural 12

-- Define the remainder function
def remainder (a b : Nat) : Nat := a % b

-- Prove that the remainder when the sum of the numbers from 1 to 12 is divided by 9 is 6
theorem remainder_sum_1_to_12_div_9 : remainder sum_1_to_12 9 = 6 := by
  sorry

end remainder_sum_1_to_12_div_9_l2091_209191


namespace inequality_one_inequality_two_l2091_209132

theorem inequality_one (a b : ℝ) : 
    a^2 + b^2 ≥ (a + b)^2 / 2 := 
by
    sorry

theorem inequality_two (a b : ℝ) : 
    a^2 + b^2 ≥ 2 * (a - b - 1) := 
by
    sorry

end inequality_one_inequality_two_l2091_209132


namespace minimum_value_l2091_209135

theorem minimum_value (a b : ℝ) (h1 : 2 * a + 3 * b = 5) (h2 : a > 0) (h3 : b > 0) : 
  (1 / a) + (1 / b) = 5 + 2 * Real.sqrt 6 :=
by
  sorry

end minimum_value_l2091_209135


namespace total_toes_on_bus_l2091_209161

/-- Definition for the number of toes a Hoopit has -/
def toes_per_hoopit : ℕ := 4 * 3

/-- Definition for the number of toes a Neglart has -/
def toes_per_neglart : ℕ := 5 * 2

/-- Definition for the total number of Hoopits on the bus -/
def hoopit_students_on_bus : ℕ := 7

/-- Definition for the total number of Neglarts on the bus -/
def neglart_students_on_bus : ℕ := 8

/-- Proving that the total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus : hoopit_students_on_bus * toes_per_hoopit + neglart_students_on_bus * toes_per_neglart = 164 := by
  sorry

end total_toes_on_bus_l2091_209161


namespace evaluate_expression_l2091_209156

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 - 3 = 31 :=
by
  sorry

end evaluate_expression_l2091_209156


namespace female_democrats_l2091_209140

/-
There are 810 male and female participants in a meeting.
Half of the female participants and one-quarter of the male participants are Democrats.
One-third of all the participants are Democrats.
Prove that the number of female Democrats is 135.
-/

theorem female_democrats (F M : ℕ) (h : F + M = 810)
  (female_democrats : F / 2 = F / 2)
  (male_democrats : M / 4 = M / 4)
  (total_democrats : (F / 2 + M / 4) = 810 / 3) : 
  F / 2 = 135 := by
  sorry

end female_democrats_l2091_209140


namespace find_a2016_l2091_209157

variable {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Given conditions
def cond1 : S 1 = 6 := by sorry
def cond2 : S 2 = 4 := by sorry
def cond3 (n : ℕ) : S n > 0 := by sorry
def cond4 (n : ℕ) : S (2 * n - 1) ^ 2 = S (2 * n) * S (2 * n + 2) := by sorry
def cond5 (n : ℕ) : 2 * S (2 * n + 2) = S (2 * n - 1) + S (2 * n + 1) := by sorry

theorem find_a2016 : a 2016 = -1009 := by
  -- Use the provided conditions to prove the statement
  sorry

end find_a2016_l2091_209157


namespace infinite_solutions_l2091_209105

theorem infinite_solutions (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ (k : ℕ), x = k^3 + 1 ∧ y = (k^3 + 1) * k := 
sorry

end infinite_solutions_l2091_209105


namespace inequality_proof_l2091_209129

open Real

theorem inequality_proof
  (a b c d : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d > 0)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^2 / (b + c + d) + b^2 / (c + d + a) +
   c^2 / (d + a + b) + d^2 / (a + b + c) ≥ 2 / 3) :=
by
  sorry

end inequality_proof_l2091_209129


namespace rightmost_three_digits_of_7_pow_1997_l2091_209106

theorem rightmost_three_digits_of_7_pow_1997 :
  7^1997 % 1000 = 207 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1997_l2091_209106


namespace range_of_a_l2091_209148

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then (1/2 : ℝ) * x - 1 else 1 / x

theorem range_of_a (a : ℝ) : f a > a ↔ a < -1 :=
sorry

end range_of_a_l2091_209148


namespace ofelia_ratio_is_two_l2091_209121

noncomputable def OfeliaSavingsRatio : ℝ :=
  let january_savings := 10
  let may_savings := 160
  let x := (may_savings / january_savings)^(1/4)
  x

theorem ofelia_ratio_is_two : OfeliaSavingsRatio = 2 := by
  sorry

end ofelia_ratio_is_two_l2091_209121


namespace operation_evaluation_l2091_209104

def my_operation (x y : Int) : Int :=
  x * (y + 1) + x * y

theorem operation_evaluation :
  my_operation (-3) (-4) = 21 := by
  sorry

end operation_evaluation_l2091_209104


namespace sin_18_cos_36_eq_quarter_l2091_209131

theorem sin_18_cos_36_eq_quarter : Real.sin (Real.pi / 10) * Real.cos (Real.pi / 5) = 1 / 4 :=
by
  sorry

end sin_18_cos_36_eq_quarter_l2091_209131


namespace point_B_is_south_of_A_total_distance_traveled_fuel_consumed_l2091_209181

-- Define the travel records and the fuel consumption rate
def travel_records : List Int := [18, -9, 7, -14, -6, 13, -6, -8]
def fuel_consumption_rate : Float := 0.4

-- Question 1: Proof that point B is 5 km south of point A
theorem point_B_is_south_of_A : (travel_records.sum = -5) :=
  by sorry

-- Question 2: Proof that total distance traveled is 81 km
theorem total_distance_traveled : (travel_records.map Int.natAbs).sum = 81 :=
  by sorry

-- Question 3: Proof that the fuel consumed is 32 liters (Rounded)
theorem fuel_consumed : Float.floor (81 * fuel_consumption_rate) = 32 :=
  by sorry

end point_B_is_south_of_A_total_distance_traveled_fuel_consumed_l2091_209181


namespace cards_left_l2091_209101

def number_of_initial_cards : ℕ := 67
def number_of_cards_taken : ℕ := 9

theorem cards_left (l : ℕ) (d : ℕ) (hl : l = number_of_initial_cards) (hd : d = number_of_cards_taken) : l - d = 58 :=
by
  sorry

end cards_left_l2091_209101


namespace trip_total_hours_l2091_209175

theorem trip_total_hours
    (x : ℕ) -- additional hours of travel
    (dist_1 : ℕ := 30 * 6) -- distance for first 6 hours
    (dist_2 : ℕ := 46 * x) -- distance for additional hours
    (total_dist : ℕ := dist_1 + dist_2) -- total distance
    (total_time : ℕ := 6 + x) -- total time
    (avg_speed : ℕ := total_dist / total_time) -- average speed
    (h : avg_speed = 34) : total_time = 8 :=
by
  sorry

end trip_total_hours_l2091_209175


namespace correct_statements_l2091_209198

def problem_statements :=
  [ "The negation of the statement 'There exists an x ∈ ℝ such that x^2 - 3x + 3 = 0' is true.",
    "The statement '-1/2 < x < 0' is a necessary but not sufficient condition for '2x^2 - 5x - 3 < 0'.",
    "The negation of the statement 'If xy = 0, then at least one of x or y is equal to 0' is true.",
    "The curves x^2/25 + y^2/9 = 1 and x^2/(25 − k) + y^2/(9 − k) = 1 (9 < k < 25) share the same foci.",
    "There exists a unique line that passes through the point (1,3) and is tangent to the parabola y^2 = 4x."
  ]

theorem correct_statements :
  (∀ x : ℝ, ¬(x^2 - 3 * x + 3 = 0)) ∧ 
  ¬ (¬-1/2 < x ∧ x < 0 → 2 * x^2 - 5*x - 3 < 0) ∧ 
  (∀ x y : ℝ, xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧ 
  (∀ k : ℝ, 9 < k ∧ k < 25 → ∀ x y : ℝ, (x^2 / (25 - k) + y^2 / (9 - k) = 1) → (x^2 / 25 + y^2 / 9 = 1) → (x ≠ 0 ∨ y ≠ 0)) ∧ 
  ¬ (∃ l : ℝ, ∀ pt : ℝ × ℝ, pt = (1, 3) → ∀ y : ℝ, y^2 = 4 * pt.1 → y = 2 * pt.2)
:= 
  sorry

end correct_statements_l2091_209198


namespace least_amount_to_add_l2091_209108

theorem least_amount_to_add (current_amount : ℕ) (n : ℕ) (divisor : ℕ) [NeZero divisor]
  (current_amount_eq : current_amount = 449774) (n_eq : n = 1) (divisor_eq : divisor = 6) :
  ∃ k : ℕ, (current_amount + k) % divisor = 0 ∧ k = n := by
  sorry

end least_amount_to_add_l2091_209108


namespace correct_judgements_l2091_209169

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_period_1 : ∀ x : ℝ, f (x + 1) = -f x
axiom f_increasing_0_1 : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y

theorem correct_judgements : 
  (∀ x : ℝ, f (x + 2) = f x) ∧ 
  (∀ x : ℝ, f (1 - x) = f (1 + x)) ∧ 
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≥ f y) ∧ 
  ¬(∀ x y : ℝ, -2 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≥ f y) :=
by 
  sorry

end correct_judgements_l2091_209169


namespace factorize_problem1_factorize_problem2_l2091_209114

-- Problem 1
theorem factorize_problem1 (a b : ℝ) : 
    -3 * a^2 + 6 * a * b - 3 * b^2 = -3 * (a - b)^2 := 
by sorry

-- Problem 2
theorem factorize_problem2 (a b x y : ℝ) : 
    9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) := 
by sorry

end factorize_problem1_factorize_problem2_l2091_209114


namespace train_speed_l2091_209180

/--
Given:
  Length of the train = 500 m
  Length of the bridge = 350 m
  The train takes 60 seconds to completely cross the bridge.

Prove:
  The speed of the train is exactly 14.1667 m/s
-/
theorem train_speed (length_train length_bridge time : ℝ) (h_train : length_train = 500) (h_bridge : length_bridge = 350) (h_time : time = 60) :
  (length_train + length_bridge) / time = 14.1667 :=
by
  rw [h_train, h_bridge, h_time]
  norm_num
  sorry

end train_speed_l2091_209180


namespace simplify_and_evaluate_l2091_209185

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2) :
  ( (1 + x) / (1 - x) / (x - (2 * x / (1 - x))) = - (Real.sqrt 2 + 2) / 2) :=
by
  rw [h]
  simp
  sorry

end simplify_and_evaluate_l2091_209185


namespace log_identity_l2091_209153

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_identity
    (a b c : ℝ)
    (h1 : a ^ 2 + b ^ 2 = c ^ 2)
    (h2 : a > 0)
    (h3 : c > 0)
    (h4 : b > 0)
    (h5 : c > b) :
    log_base (c + b) a + log_base (c - b) a = 2 * log_base (c + b) a * log_base (c - b) a :=
sorry

end log_identity_l2091_209153


namespace simplify_fractions_l2091_209134

theorem simplify_fractions : 
  (150 / 225) + (90 / 135) = 4 / 3 := by 
  sorry

end simplify_fractions_l2091_209134


namespace number_of_ordered_pairs_l2091_209168

-- Define the predicate that defines the condition for the ordered pairs (m, n)
def satisfies_condition (m n : ℕ) : Prop :=
  6 % m = 0 ∧ 3 % n = 0 ∧ 6 / m + 3 / n = 1

-- Define the main theorem for the problem statement
theorem number_of_ordered_pairs : 
  (∃ count : ℕ, count = 6 ∧ 
  (∀ m n : ℕ, satisfies_condition m n → m > 0 ∧ n > 0)) :=
by {
 sorry -- Placeholder for the actual proof
}

end number_of_ordered_pairs_l2091_209168


namespace odd_function_a_b_l2091_209188

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l2091_209188


namespace sufficient_but_not_necessary_l2091_209158

theorem sufficient_but_not_necessary {a b : ℝ} (h1 : a > 2) (h2 : b > 2) : 
  a + b > 4 ∧ a * b > 4 := 
by
  sorry

end sufficient_but_not_necessary_l2091_209158


namespace find_a_plus_b_l2091_209119

def f (x a b : ℝ) := x^3 + a * x^2 + b * x + a^2

def extremum_at_one (a b : ℝ) : Prop :=
  f 1 a b = 10 ∧ (3 * 1^2 + 2 * a * 1 + b = 0)

theorem find_a_plus_b (a b : ℝ) (h : extremum_at_one a b) : a + b = -7 :=
by
  sorry

end find_a_plus_b_l2091_209119


namespace rain_at_least_one_day_l2091_209196

-- Define the probabilities
def P_A1 : ℝ := 0.30
def P_A2 : ℝ := 0.40
def P_A2_given_A1 : ℝ := 0.70

-- Define complementary probabilities
def P_not_A1 : ℝ := 1 - P_A1
def P_not_A2 : ℝ := 1 - P_A2
def P_not_A2_given_A1 : ℝ := 1 - P_A2_given_A1

-- Calculate probabilities of no rain on both days under different conditions
def P_no_rain_both_days_if_no_rain_first : ℝ := P_not_A1 * P_not_A2
def P_no_rain_both_days_if_rain_first : ℝ := P_A1 * P_not_A2_given_A1

-- Total probability of no rain on both days
def P_no_rain_both_days : ℝ := P_no_rain_both_days_if_no_rain_first + P_no_rain_both_days_if_rain_first

-- Probability of rain on at least one of the two days
def P_rain_one_or_more_days : ℝ := 1 - P_no_rain_both_days

-- Expressing the result as a percentage
def result_percentage : ℝ := P_rain_one_or_more_days * 100

-- Theorem statement
theorem rain_at_least_one_day : result_percentage = 49 := by
  -- We skip the proof
  sorry

end rain_at_least_one_day_l2091_209196


namespace area_of_triangle_l2091_209150

theorem area_of_triangle 
  (h : ∀ x y : ℝ, (x / 5 + y / 2 = 1) → ((x = 5 ∧ y = 0) ∨ (x = 0 ∧ y = 2))) : 
  ∃ t : ℝ, t = 1 / 2 * 2 * 5 := 
sorry

end area_of_triangle_l2091_209150


namespace sum_of_possible_B_is_zero_l2091_209110

theorem sum_of_possible_B_is_zero :
  ∀ B : ℕ, B < 10 → (∃ k : ℤ, 7 * k = 500 + 10 * B + 3) -> B = 0 := sorry

end sum_of_possible_B_is_zero_l2091_209110


namespace find_fourth_intersection_point_l2091_209167

theorem find_fourth_intersection_point 
  (a b r: ℝ) 
  (h4 : ∃ a b r, ∀ x y, (x - a)^2 + (y - b)^2 = r^2 → (x, y) = (4, 1) ∨ (x, y) = (-2, -2) ∨ (x, y) = (8, 1/2) ∨ (x, y) = (-1/4, -16)):
  ∃ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2 → x * y = 4 → (x, y) = (-1/4, -16) := 
sorry

end find_fourth_intersection_point_l2091_209167


namespace sum_of_primes_eq_100_l2091_209173

theorem sum_of_primes_eq_100 : 
  ∃ (S : Finset ℕ), (∀ (x : ℕ), x ∈ S → Nat.Prime x) ∧ S.sum id = 100 ∧ S.card = 9 :=
by
  sorry

end sum_of_primes_eq_100_l2091_209173


namespace bananas_indeterminate_l2091_209170

namespace RubyBananaProblem

variables (number_of_candies : ℕ) (number_of_friends : ℕ) (candies_per_friend : ℕ)
           (number_of_bananas : Option ℕ)

-- Given conditions
def Ruby_has_36_candies := number_of_candies = 36
def Ruby_has_9_friends := number_of_friends = 9
def Each_friend_gets_4_candies := candies_per_friend = 4
def Can_distribute_candies := number_of_candies = number_of_friends * candies_per_friend

-- Mathematical statement
theorem bananas_indeterminate (h1 : Ruby_has_36_candies number_of_candies)
                              (h2 : Ruby_has_9_friends number_of_friends)
                              (h3 : Each_friend_gets_4_candies candies_per_friend)
                              (h4 : Can_distribute_candies number_of_candies number_of_friends candies_per_friend) :
  number_of_bananas = none :=
by
  sorry

end RubyBananaProblem

end bananas_indeterminate_l2091_209170


namespace woman_age_multiple_l2091_209178

theorem woman_age_multiple (S : ℕ) (W : ℕ) (k : ℕ) 
  (h1 : S = 27)
  (h2 : W + S = 84)
  (h3 : W = k * S + 3) :
  k = 2 :=
by
  sorry

end woman_age_multiple_l2091_209178


namespace function_is_decreasing_l2091_209113

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 2

theorem function_is_decreasing (a b : ℝ) (f_even : ∀ x : ℝ, f a b x = f a b (-x))
  (domain_condition : 1 + a + 2 = 0) :
  ∀ x y : ℝ, 1 ≤ x → x < y → y ≤ 2 → f a 0 x > f a 0 y :=
by
  sorry

end function_is_decreasing_l2091_209113


namespace license_plate_problem_l2091_209118

noncomputable def license_plate_ways : ℕ :=
  let letters := 26
  let digits := 10
  let both_same := letters * digits * 1 * 1
  let digits_adj_same := letters * digits * 1 * letters
  let letters_adj_same := letters * digits * digits * 1
  digits_adj_same + letters_adj_same - both_same

theorem license_plate_problem :
  9100 = license_plate_ways :=
by
  -- Skipping the detailed proof for now
  sorry

end license_plate_problem_l2091_209118


namespace square_presses_exceed_1000_l2091_209174

theorem square_presses_exceed_1000:
  ∃ n : ℕ, (n = 3) ∧ (3 ^ (2^n) > 1000) :=
by
  sorry

end square_presses_exceed_1000_l2091_209174


namespace dog_food_cans_l2091_209143

theorem dog_food_cans 
  (packages_cat_food : ℕ)
  (cans_per_package_cat_food : ℕ)
  (packages_dog_food : ℕ)
  (additional_cans_cat_food : ℕ)
  (total_cans_cat_food : ℕ)
  (total_cans_dog_food : ℕ)
  (num_cans_dog_food_package : ℕ) :
  packages_cat_food = 9 →
  cans_per_package_cat_food = 10 →
  packages_dog_food = 7 →
  additional_cans_cat_food = 55 →
  total_cans_cat_food = packages_cat_food * cans_per_package_cat_food →
  total_cans_dog_food = packages_dog_food * num_cans_dog_food_package →
  total_cans_cat_food = total_cans_dog_food + additional_cans_cat_food →
  num_cans_dog_food_package = 5 :=
by
  sorry

end dog_food_cans_l2091_209143


namespace range_of_k_l2091_209177

theorem range_of_k (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_decreasing : ∀ ⦃x y⦄, 0 ≤ x → x < y → f y < f x) 
  (h_inequality : ∀ x, f (k * x ^ 2 + 2) + f (k * x + k) ≤ 0) : 0 ≤ k :=
sorry

end range_of_k_l2091_209177


namespace total_movies_correct_l2091_209199

def num_movies_Screen1 : Nat := 3
def num_movies_Screen2 : Nat := 4
def num_movies_Screen3 : Nat := 2
def num_movies_Screen4 : Nat := 3
def num_movies_Screen5 : Nat := 5
def num_movies_Screen6 : Nat := 2

def total_movies : Nat :=
  num_movies_Screen1 + num_movies_Screen2 + num_movies_Screen3 + num_movies_Screen4 + num_movies_Screen5 + num_movies_Screen6

theorem total_movies_correct :
  total_movies = 19 :=
by 
  sorry

end total_movies_correct_l2091_209199


namespace simplify_expression_l2091_209128

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l2091_209128


namespace eric_days_waited_l2091_209112

def num_chickens := 4
def eggs_per_chicken_per_day := 3
def total_eggs := 36

def eggs_per_day := num_chickens * eggs_per_chicken_per_day
def num_days := total_eggs / eggs_per_day

theorem eric_days_waited : num_days = 3 :=
by
  sorry

end eric_days_waited_l2091_209112


namespace andy_last_problem_l2091_209147

theorem andy_last_problem (s t : ℕ) (start : s = 75) (total : t = 51) : (s + t - 1) = 125 :=
by
  sorry

end andy_last_problem_l2091_209147


namespace pet_store_cages_l2091_209166

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) (h_initial : initial_puppies = 13) (h_sold : sold_puppies = 7) (h_per_cage : puppies_per_cage = 2) : (initial_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l2091_209166


namespace circle_radius_condition_l2091_209176

theorem circle_radius_condition (c : ℝ) : 
  (∃ x y : ℝ, (x^2 + 6 * x + y^2 - 4 * y + c = 0)) ∧ 
  (radius = 6) ↔ 
  c = -23 := by
  sorry

end circle_radius_condition_l2091_209176


namespace work_completion_days_l2091_209192

-- Define the work rates
def john_work_rate : ℚ := 1/8
def rose_work_rate : ℚ := 1/16
def dave_work_rate : ℚ := 1/12

-- Define the combined work rate
def combined_work_rate : ℚ := john_work_rate + rose_work_rate + dave_work_rate

-- Define the required number of days to complete the work together
def days_to_complete_work : ℚ := 1 / combined_work_rate

-- Prove that the total number of days required to complete the work is 48/13
theorem work_completion_days : days_to_complete_work = 48 / 13 :=
by 
  -- Here is where the actual proof would be, but it is not needed as per instructions
  sorry

end work_completion_days_l2091_209192


namespace increased_colored_area_l2091_209139

theorem increased_colored_area
  (P : ℝ) -- Perimeter of the original convex pentagon
  (s : ℝ) -- Distance from the points colored originally
  : 
  s * P + π * s^2 = 23.14 :=
by
  sorry

end increased_colored_area_l2091_209139


namespace div_by_13_l2091_209165

theorem div_by_13 (a b c : ℤ) (h : (a + b + c) % 13 = 0) : 
  (a^2007 + b^2007 + c^2007 + 2 * 2007 * a * b * c) % 13 = 0 :=
by
  sorry

end div_by_13_l2091_209165


namespace sufficient_but_not_necessary_pi_l2091_209163

theorem sufficient_but_not_necessary_pi (x : ℝ) : 
  (x = Real.pi → Real.sin x = 0) ∧ (Real.sin x = 0 → ∃ k : ℤ, x = k * Real.pi) → ¬(Real.sin x = 0 → x = Real.pi) :=
by
  sorry

end sufficient_but_not_necessary_pi_l2091_209163


namespace sumNats_l2091_209190

-- Define the set of natural numbers between 29 and 31 inclusive
def NatRange : List ℕ := [29, 30, 31]

-- Define the condition that checks the elements in the range
def isValidNumbers (n : ℕ) : Prop := n ≤ 31 ∧ n > 28

-- Check if all numbers in NatRange are valid
def allValidNumbers : Prop := ∀ n, n ∈ NatRange → isValidNumbers n

-- Define the sum function for the list
def sumList (lst : List ℕ) : ℕ := lst.foldr (.+.) 0

-- The main theorem
theorem sumNats : (allValidNumbers → (sumList NatRange) = 90) :=
by
  sorry

end sumNats_l2091_209190


namespace sum_of_numbers_l2091_209159

theorem sum_of_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 149)
  (h2 : ab + bc + ca = 70) : 
  a + b + c = 17 :=
sorry

end sum_of_numbers_l2091_209159


namespace correct_transformation_l2091_209162

theorem correct_transformation (x : ℝ) :
  3 + x = 7 ∧ ¬ (x = 7 + 3) ∧
  5 * x = -4 ∧ ¬ (x = -5 / 4) ∧
  (7 / 4) * x = 3 ∧ ¬ (x = 3 * (7 / 4)) ∧
  -((x - 2) / 4) = 1 ∧ (-(x - 2)) = 4 :=
by
  sorry

end correct_transformation_l2091_209162


namespace temperature_on_friday_l2091_209197

variables {M T W Th F : ℝ}

theorem temperature_on_friday
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 41) :
  F = 33 :=
  sorry

end temperature_on_friday_l2091_209197


namespace tenth_term_of_arithmetic_sequence_l2091_209126

theorem tenth_term_of_arithmetic_sequence 
  (a d : ℤ)
  (h1 : a + 2 * d = 14)
  (h2 : a + 5 * d = 32) : 
  (a + 9 * d = 56) ∧ (d = 6) := 
by
  sorry

end tenth_term_of_arithmetic_sequence_l2091_209126


namespace total_go_stones_correct_l2091_209133

-- Definitions based on the problem's conditions
def stones_per_bundle : Nat := 10
def num_bundles : Nat := 3
def white_stones : Nat := 16

-- A function that calculates the total number of go stones
def total_go_stones : Nat :=
  num_bundles * stones_per_bundle + white_stones

-- The theorem we want to prove
theorem total_go_stones_correct : total_go_stones = 46 :=
by
  sorry

end total_go_stones_correct_l2091_209133


namespace no_injective_function_exists_l2091_209122

theorem no_injective_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f (x^2) - (f x)^2 ≥ 1/4) ∧ (∀ x y, f x = f y → x = y) := 
sorry

end no_injective_function_exists_l2091_209122


namespace calculate_height_l2091_209184

def base_length : ℝ := 2 -- in cm
def base_width : ℝ := 5 -- in cm
def volume : ℝ := 30 -- in cm^3

theorem calculate_height: base_length * base_width * 3 = volume :=
by
  -- base_length * base_width = 10
  -- 10 * 3 = 30
  sorry

end calculate_height_l2091_209184


namespace non_congruent_triangles_perimeter_18_l2091_209136

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l2091_209136


namespace binomial_12_6_eq_1848_l2091_209116

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end binomial_12_6_eq_1848_l2091_209116


namespace suraj_average_increase_l2091_209137

theorem suraj_average_increase
  (A : ℝ)
  (h1 : 9 * A + 200 = 10 * 128) :
  128 - A = 8 :=
by
  sorry

end suraj_average_increase_l2091_209137


namespace least_cost_flower_bed_divisdes_l2091_209100

theorem least_cost_flower_bed_divisdes:
  let Region1 := 5 * 2
  let Region2 := 3 * 5
  let Region3 := 2 * 4
  let Region4 := 5 * 4
  let Region5 := 5 * 3
  let Cost_Dahlias := 2.70
  let Cost_Cannas := 2.20
  let Cost_Begonias := 1.70
  let Cost_Freesias := 3.20
  let total_cost := 
    Region1 * Cost_Dahlias + 
    Region2 * Cost_Cannas + 
    Region3 * Cost_Freesias + 
    Region4 * Cost_Begonias + 
    Region5 * Cost_Cannas
  total_cost = 152.60 :=
by
  sorry

end least_cost_flower_bed_divisdes_l2091_209100


namespace min_value_of_fraction_l2091_209146

theorem min_value_of_fraction 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (Real.sqrt 3) = Real.sqrt (3 ^ a * 3 ^ (2 * b))) : 
  ∃ (min : ℝ), min = (2 / a + 1 / b) ∧ min = 8 :=
by
  -- proof will be skipped using sorry
  sorry

end min_value_of_fraction_l2091_209146


namespace find_y_z_l2091_209193

def abs_diff (x y : ℝ) := abs (x - y)

noncomputable def seq_stabilize (x y z : ℝ) (n : ℕ) : Prop :=
  let x1 := abs_diff x y 
  let y1 := abs_diff y z 
  let z1 := abs_diff z x
  ∃ k : ℕ, k ≥ n ∧ abs_diff x1 y1 = x ∧ abs_diff y1 z1 = y ∧ abs_diff z1 x1 = z

theorem find_y_z (x y z : ℝ) (hx : x = 1) (hstab : ∃ n : ℕ, seq_stabilize x y z n) : y = 0 ∧ z = 0 :=
sorry

end find_y_z_l2091_209193


namespace divisor_inequality_l2091_209117

-- Definition of our main inequality theorem
theorem divisor_inequality (n : ℕ) (h1 : n > 0) (h2 : n % 8 = 4)
    (divisors : List ℕ) (h3 : divisors = (List.range (n + 1)).filter (λ x => n % x = 0)) 
    (i : ℕ) (h4 : i < divisors.length - 1) (h5 : i % 3 ≠ 0) : 
    divisors[i + 1] ≤ 2 * divisors[i] := sorry

end divisor_inequality_l2091_209117


namespace product_of_5_consecutive_numbers_not_square_l2091_209195

-- Define what it means for a product to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating the problem
theorem product_of_5_consecutive_numbers_not_square :
  ∀ (a : ℕ), 0 < a → ¬ is_perfect_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by 
  sorry

end product_of_5_consecutive_numbers_not_square_l2091_209195


namespace total_students_in_school_l2091_209160

-- Definitions and conditions
def number_of_blind_students (B : ℕ) : Prop := ∃ B, 3 * B = 180
def number_of_other_disabilities (O : ℕ) (B : ℕ) : Prop := O = 2 * B
def total_students (T : ℕ) (D : ℕ) (B : ℕ) (O : ℕ) : Prop := T = D + B + O

theorem total_students_in_school : 
  ∃ (T B O : ℕ), number_of_blind_students B ∧ 
                 number_of_other_disabilities O B ∧ 
                 total_students T 180 B O ∧ 
                 T = 360 :=
by
  sorry

end total_students_in_school_l2091_209160


namespace cost_per_amulet_is_30_l2091_209187

variable (days_sold : ℕ := 2)
variable (amulets_per_day : ℕ := 25)
variable (price_per_amulet : ℕ := 40)
variable (faire_percentage : ℕ := 10)
variable (profit : ℕ := 300)

def total_amulets_sold := days_sold * amulets_per_day
def total_revenue := total_amulets_sold * price_per_amulet
def faire_cut := total_revenue * faire_percentage / 100
def revenue_after_faire := total_revenue - faire_cut
def total_cost := revenue_after_faire - profit
def cost_per_amulet := total_cost / total_amulets_sold

theorem cost_per_amulet_is_30 : cost_per_amulet = 30 := by
  sorry

end cost_per_amulet_is_30_l2091_209187


namespace remaining_stickers_l2091_209172

def stickers_per_page : ℕ := 20
def pages : ℕ := 12
def lost_pages : ℕ := 1

theorem remaining_stickers : 
  (pages * stickers_per_page - lost_pages * stickers_per_page) = 220 :=
  by
    sorry

end remaining_stickers_l2091_209172


namespace find_f_x_squared_l2091_209151

-- Define the function f with the given condition
noncomputable def f (x : ℝ) : ℝ := (x + 1)^2

-- Theorem statement
theorem find_f_x_squared : f (x^2) = (x^2 + 1)^2 :=
by
  sorry

end find_f_x_squared_l2091_209151


namespace cubic_and_quintic_values_l2091_209182

theorem cubic_and_quintic_values (a : ℝ) (h : (a + 1/a)^2 = 11) : 
    (a^3 + 1/a^3 = 8 * Real.sqrt 11 ∧ a^5 + 1/a^5 = 71 * Real.sqrt 11) ∨ 
    (a^3 + 1/a^3 = -8 * Real.sqrt 11 ∧ a^5 + 1/a^5 = -71 * Real.sqrt 11) :=
by
  sorry

end cubic_and_quintic_values_l2091_209182


namespace inclination_angle_of_line_l2091_209142

theorem inclination_angle_of_line
  (α : ℝ) (h1 : α > 0) (h2 : α < 180)
  (hslope : Real.tan α = - (Real.sqrt 3) / 3) :
  α = 150 :=
sorry

end inclination_angle_of_line_l2091_209142


namespace x_y_z_sum_l2091_209141

theorem x_y_z_sum :
  ∃ (x y z : ℕ), (16 / 3)^x * (27 / 25)^y * (5 / 4)^z = 256 ∧ x + y + z = 6 :=
by
  -- Proof can be completed here
  sorry

end x_y_z_sum_l2091_209141


namespace trajectory_is_parabola_l2091_209138

theorem trajectory_is_parabola
  (P : ℝ × ℝ) : 
  (dist P (0, P.2 + 1) < dist P (0, 2)) -> 
  (P.1^2 = 8 * (P.2 + 2)) :=
by
  sorry

end trajectory_is_parabola_l2091_209138


namespace sum_first_eight_geom_terms_eq_l2091_209179

noncomputable def S8_geom_sum : ℚ :=
  let a := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  a * (1 - r^8) / (1 - r)

theorem sum_first_eight_geom_terms_eq :
  S8_geom_sum = 3280 / 6561 :=
by
  sorry

end sum_first_eight_geom_terms_eq_l2091_209179


namespace sqrt_calc1_sqrt_calc2_l2091_209149

-- Problem 1 proof statement
theorem sqrt_calc1 : ( (Real.sqrt 2 + Real.sqrt 3) ^ 2 - (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) = 4 + 2 * Real.sqrt 6 ) :=
  sorry

-- Problem 2 proof statement
theorem sqrt_calc2 : ( (2 - Real.sqrt 3) ^ 2023 * (2 + Real.sqrt 3) ^ 2023 - 2 * abs (-Real.sqrt 3 / 2) - (-Real.sqrt 2) ^ 0 = -Real.sqrt 3 ) :=
  sorry

end sqrt_calc1_sqrt_calc2_l2091_209149


namespace remainder_of_f_div_x_minus_2_is_48_l2091_209186

-- Define the polynomial f(x)
noncomputable def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 8 * x^3 + 25 * x^2 - 14 * x - 40

-- State the theorem to prove that the remainder of f(x) when divided by x - 2 is 48
theorem remainder_of_f_div_x_minus_2_is_48 : f 2 = 48 :=
by sorry

end remainder_of_f_div_x_minus_2_is_48_l2091_209186


namespace line_through_two_points_l2091_209120

theorem line_through_two_points :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x, y) = (1, 3) ∨ (x, y) = (3, 7) → y = m * x + b) ∧ (m + b = 3) := by
{ sorry }

end line_through_two_points_l2091_209120


namespace weight_ratio_mars_moon_l2091_209127

theorem weight_ratio_mars_moon :
  (∀ iron carbon other_elements_moon other_elements_mars wt_moon wt_mars : ℕ, 
    wt_moon = 250 ∧ 
    iron = 50 ∧ 
    carbon = 20 ∧ 
    other_elements_moon + 50 + 20 = 100 ∧ 
    other_elements_moon * wt_moon / 100 = 75 ∧ 
    other_elements_mars = 150 ∧ 
    wt_mars = (other_elements_mars * wt_moon) / other_elements_moon
  → wt_mars / wt_moon = 2) := 
sorry

end weight_ratio_mars_moon_l2091_209127


namespace solution_l2091_209194

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)

axiom periodic_f : ∀ x : ℝ, f (x - 3) = - f x

axiom increasing_f_on_interval : ∀ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ 3 ∧ 0 ≤ x2 ∧ x2 ≤ 3 ∧ x1 ≠ x2) → (f x1 - f x2) / (x1 - x2) > 0

theorem solution : f 49 < f 64 ∧ f 64 < f 81 :=
by
  sorry

end solution_l2091_209194


namespace first_loan_amount_l2091_209109

theorem first_loan_amount :
  ∃ (L₁ L₂ : ℝ) (r : ℝ),
  (L₂ = 4700) ∧
  (L₁ = L₂ + 1500) ∧
  (0.09 * L₂ + r * L₁ = 617) ∧
  (L₁ = 6200) :=
by 
  sorry

end first_loan_amount_l2091_209109


namespace angle_degree_measure_l2091_209152

theorem angle_degree_measure (x : ℝ) (h1 : (x + (90 - x) = 90)) (h2 : (x = 3 * (90 - x))) : x = 67.5 := by
  sorry

end angle_degree_measure_l2091_209152


namespace find_b_l2091_209183

noncomputable def Q (x d b e : ℝ) : ℝ := x^3 + d*x^2 + b*x + e

theorem find_b (d b e : ℝ) (h1 : -d / 3 = -e) (h2 : -e = 1 + d + b + e) (h3 : e = 6) : b = -31 :=
by sorry

end find_b_l2091_209183


namespace perpendicular_lines_a_eq_1_l2091_209145

-- Definitions for the given conditions
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + y + 3 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (2 * a - 3) * y = 4

-- Condition that the lines are perpendicular
def perpendicular_lines (a : ℝ) : Prop := a + (2 * a - 3) = 0

-- Proof problem to be solved
theorem perpendicular_lines_a_eq_1 (a : ℝ) (h : perpendicular_lines a) : a = 1 :=
by
  sorry

end perpendicular_lines_a_eq_1_l2091_209145


namespace total_students_in_class_l2091_209123

-- Definitions based on the conditions
def num_girls : ℕ := 140
def num_boys_absent : ℕ := 40
def num_boys_present := num_girls / 2
def num_boys := num_boys_present + num_boys_absent
def total_students := num_girls + num_boys

-- Theorem to be proved
theorem total_students_in_class : total_students = 250 :=
by
  sorry

end total_students_in_class_l2091_209123


namespace evaluate_expression_l2091_209115

def cube_root (x : ℝ) := x^(1/3)

theorem evaluate_expression : (cube_root (9 / 32))^2 = (3/8) := 
by
  sorry

end evaluate_expression_l2091_209115


namespace distance_between_lines_is_two_l2091_209111

noncomputable def distance_between_parallel_lines : ℝ := 
  let A1 := 3
  let B1 := 4
  let C1 := -3
  let A2 := 6
  let B2 := 8
  let C2 := 14
  (|C2 - C1| : ℝ) / Real.sqrt (A2^2 + B2^2)

theorem distance_between_lines_is_two :
  distance_between_parallel_lines = 2 := by
  sorry

end distance_between_lines_is_two_l2091_209111


namespace y_squared_range_l2091_209154

theorem y_squared_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 4) ^ (1/3) = 4) : 15 ≤ y^2 ∧ y^2 ≤ 25 :=
by
  sorry

end y_squared_range_l2091_209154
