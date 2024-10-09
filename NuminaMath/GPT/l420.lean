import Mathlib

namespace inequality_solution_set_inequality_proof_2_l420_42069

theorem inequality_solution_set : 
  { x : ℝ | |x + 1| + |x + 3| < 4 } = { x : ℝ | -4 < x ∧ x < 0 } :=
sorry

theorem inequality_proof_2 (a b : ℝ) (ha : -4 < a) (ha' : a < 0) (hb : -4 < b) (hb' : b < 0) : 
  2 * |a - b| < |a * b + 2 * a + 2 * b| :=
sorry

end inequality_solution_set_inequality_proof_2_l420_42069


namespace total_students_l420_42098

theorem total_students (S F G B N : ℕ) 
  (hF : F = 41) 
  (hG : G = 22) 
  (hB : B = 9) 
  (hN : N = 24) 
  (h_total : S = (F + G - B) + N) : 
  S = 78 :=
by
  sorry

end total_students_l420_42098


namespace cars_in_parking_lot_l420_42006

theorem cars_in_parking_lot (C : ℕ) (customers_per_car : ℕ) (total_purchases : ℕ) 
  (h1 : customers_per_car = 5)
  (h2 : total_purchases = 50)
  (h3 : C * customers_per_car = total_purchases) : 
  C = 10 := 
by
  sorry

end cars_in_parking_lot_l420_42006


namespace two_point_question_count_l420_42045

/-- Define the number of questions and points on the test,
    and prove that the number of 2-point questions is 30. -/
theorem two_point_question_count (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 := by
  sorry

end two_point_question_count_l420_42045


namespace even_and_odd_implies_zero_l420_42070

theorem even_and_odd_implies_zero (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = -f x) (h2 : ∀ x : ℝ, f (-x) = f x) : ∀ x : ℝ, f x = 0 :=
by
  sorry

end even_and_odd_implies_zero_l420_42070


namespace floor_multiple_of_floor_l420_42011

noncomputable def r : ℝ := sorry

theorem floor_multiple_of_floor (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : ∃ k, n = k * m) (hr : r ≥ 1) 
  (floor_multiple : ∀ (m n : ℕ), (∃ k : ℕ, n = k * m) → ∃ l, ⌊n * r⌋ = l * ⌊m * r⌋) :
  ∃ k : ℤ, r = k := 
sorry

end floor_multiple_of_floor_l420_42011


namespace total_chairs_agreed_proof_l420_42060

/-
Conditions:
- Carey moved 28 chairs
- Pat moved 29 chairs
- They have 17 chairs left to move
Question:
- How many chairs did they agree to move in total?
Proof Problem:
- Prove that the total number of chairs they agreed to move is equal to 74.
-/

def carey_chairs : ℕ := 28
def pat_chairs : ℕ := 29
def chairs_left : ℕ := 17
def total_chairs_agreed : ℕ := carey_chairs + pat_chairs + chairs_left

theorem total_chairs_agreed_proof : total_chairs_agreed = 74 := 
by
  sorry

end total_chairs_agreed_proof_l420_42060


namespace new_op_4_3_l420_42026

def new_op (a b : ℕ) : ℕ := a^2 - a * b + b^2

theorem new_op_4_3 : new_op 4 3 = 13 :=
by
  -- Placeholder for the proof
  sorry

end new_op_4_3_l420_42026


namespace three_powers_in_two_digit_range_l420_42051

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l420_42051


namespace max_value_func_l420_42012

noncomputable def func (x : ℝ) : ℝ :=
  Real.sin x - Real.sqrt 3 * Real.cos x

theorem max_value_func : ∃ x : ℝ, func x = 2 :=
by
  -- proof steps will be provided here
  sorry

end max_value_func_l420_42012


namespace range_of_m_l420_42034

variable (m : ℝ)
def p : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h : p m ∧ q m) : -2 < m ∧ m < 0 := sorry

end range_of_m_l420_42034


namespace probability_value_at_least_75_cents_l420_42027

-- Given conditions
def box_contains (pennies nickels quarters : ℕ) : Prop :=
  pennies = 4 ∧ nickels = 3 ∧ quarters = 5

def draw_without_replacement (total_coins : ℕ) (drawn_coins : ℕ) : Prop :=
  total_coins = 12 ∧ drawn_coins = 5

def equal_probability (chosen_probability : ℚ) (total_coins : ℕ) : Prop :=
  chosen_probability = 1/total_coins

-- Probability that the value of coins drawn is at least 75 cents
theorem probability_value_at_least_75_cents
  (pennies nickels quarters total_coins drawn_coins : ℕ)
  (chosen_probability : ℚ) :
  box_contains pennies nickels quarters →
  draw_without_replacement total_coins drawn_coins →
  equal_probability chosen_probability total_coins →
  chosen_probability = 1/792 :=
by
  intros
  sorry

end probability_value_at_least_75_cents_l420_42027


namespace isabella_more_than_giselle_l420_42080

variables (I S G : ℕ)

def isabella_has_more_than_sam : Prop := I = S + 45
def giselle_amount : Prop := G = 120
def total_amount : Prop := I + S + G = 345

theorem isabella_more_than_giselle
  (h1 : isabella_has_more_than_sam I S)
  (h2 : giselle_amount G)
  (h3 : total_amount I S G) :
  I - G = 15 :=
by
  sorry

end isabella_more_than_giselle_l420_42080


namespace greatest_possible_a_l420_42038

theorem greatest_possible_a (a : ℤ) (x : ℤ) (h_pos : 0 < a) (h_eq : x^3 + a * x^2 = -30) : 
  a ≤ 29 :=
sorry

end greatest_possible_a_l420_42038


namespace gcd_polynomial_example_l420_42040

theorem gcd_polynomial_example (b : ℕ) (h : ∃ k : ℕ, b = 2 * 7784 * k) : 
  gcd (5 * b ^ 2 + 68 * b + 143) (3 * b + 14) = 25 :=
by 
  sorry

end gcd_polynomial_example_l420_42040


namespace reverse_addition_unique_l420_42009

theorem reverse_addition_unique (k : ℤ) (h t u : ℕ) (n : ℤ)
  (hk : 100 * h + 10 * t + u = k) 
  (h_k_range : 100 < k ∧ k < 1000)
  (h_reverse_addition : 100 * u + 10 * t + h = k + n)
  (digits_range : 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9) :
  n = 99 :=
sorry

end reverse_addition_unique_l420_42009


namespace inequality_holds_for_a_l420_42083

theorem inequality_holds_for_a (a : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < -1 → (x + 1)^2 < Real.logb a (|x|)) ↔ 1 < a ∧ a ≤ 2 := by
  sorry

end inequality_holds_for_a_l420_42083


namespace scientific_notation_104000000_l420_42053

theorem scientific_notation_104000000 :
  104000000 = 1.04 * 10^8 :=
sorry

end scientific_notation_104000000_l420_42053


namespace count_numbers_divisible_by_12_not_20_l420_42030

theorem count_numbers_divisible_by_12_not_20 : 
  let N := 2017
  let a := Nat.floor (N / 12)
  let b := Nat.floor (N / 60)
  a - b = 135 := by
    -- Definitions used
    let N := 2017
    let a := Nat.floor (N / 12)
    let b := Nat.floor (N / 60)
    -- The desired statement
    show a - b = 135
    sorry

end count_numbers_divisible_by_12_not_20_l420_42030


namespace probability_three_common_books_l420_42004

-- Defining the total number of books
def total_books : ℕ := 12

-- Defining the number of books each of Harold and Betty chooses
def books_per_person : ℕ := 6

-- Assertion that the probability of choosing exactly 3 common books is 50/116
theorem probability_three_common_books :
  ((Nat.choose 12 3) * (Nat.choose 9 3) * (Nat.choose 6 3)) /
  ((Nat.choose 12 6) * (Nat.choose 12 6)) = 50 / 116 := by
  sorry

end probability_three_common_books_l420_42004


namespace max_area_rectangle_l420_42056

theorem max_area_rectangle (p : ℝ) (a b : ℝ) (h : p = 2 * (a + b)) : 
  ∃ S : ℝ, S = a * b ∧ (∀ (a' b' : ℝ), p = 2 * (a' + b') → S ≥ a' * b') → a = b :=
by
  sorry

end max_area_rectangle_l420_42056


namespace simplify_and_evaluate_problem_l420_42003

noncomputable def problem_expression (a : ℤ) : ℚ :=
  (1 - (3 : ℚ) / (a + 1)) / ((a^2 - 4 * a + 4 : ℚ) / (a + 1))

theorem simplify_and_evaluate_problem :
  ∀ (a : ℤ), -2 ≤ a ∧ a ≤ 2 → a ≠ -1 → a ≠ 2 →
  (problem_expression a = 1 / (a - 2 : ℚ)) ∧
  (a = 0 → problem_expression a = -1 / 2) ∧
  (a = 1 → problem_expression a = -1) :=
sorry

end simplify_and_evaluate_problem_l420_42003


namespace arthur_hot_dogs_first_day_l420_42075

theorem arthur_hot_dogs_first_day (H D n : ℕ) (h₀ : D = 1)
(h₁ : 3 * H + n = 10)
(h₂ : 2 * H + 3 * D = 7) : n = 4 :=
by sorry

end arthur_hot_dogs_first_day_l420_42075


namespace vans_hold_people_per_van_l420_42090

theorem vans_hold_people_per_van (students adults vans total_people people_per_van : ℤ) 
    (h1: students = 12) 
    (h2: adults = 3) 
    (h3: vans = 3) 
    (h4: total_people = students + adults) 
    (h5: people_per_van = total_people / vans) :
    people_per_van = 5 := 
by
    -- Steps will go here
    sorry

end vans_hold_people_per_van_l420_42090


namespace ten_unique_positive_odd_integers_equality_l420_42059

theorem ten_unique_positive_odd_integers_equality {x : ℕ} (h1: x = 3):
  ∃ S : Finset ℕ, S.card = 10 ∧ 
    (∀ n ∈ S, n < 100 ∧ n % 2 = 1 ∧ 
      ∃ k : ℕ, k % 2 = 1 ∧ n = k * x) :=
by
  sorry

end ten_unique_positive_odd_integers_equality_l420_42059


namespace Kendra_weekly_words_not_determined_without_weeks_l420_42081

def Kendra_goal : Nat := 60
def Kendra_already_learned : Nat := 36
def Kendra_needs_to_learn : Nat := 24

theorem Kendra_weekly_words_not_determined_without_weeks (weeks : Option Nat) : weeks = none → Kendra_needs_to_learn / weeks.getD 1 = 24 -> False := by
  sorry

end Kendra_weekly_words_not_determined_without_weeks_l420_42081


namespace problem_value_l420_42044

theorem problem_value:
  3^(1+3+4) - (3^1 + 3^3 + 3^4) = 6450 :=
by sorry

end problem_value_l420_42044


namespace percentage_of_boys_is_90_l420_42024

variables (B G : ℕ)

def total_children : ℕ := 100
def future_total_children : ℕ := total_children + 100
def percentage_girls : ℕ := 5
def girls_after_increase : ℕ := future_total_children * percentage_girls / 100
def boys_after_increase : ℕ := total_children - girls_after_increase

theorem percentage_of_boys_is_90 :
  B + G = total_children →
  G = girls_after_increase →
  B = total_children - G →
  (B:ℚ) / total_children * 100 = 90 :=
by
  sorry

end percentage_of_boys_is_90_l420_42024


namespace total_snakes_count_l420_42019

-- Define the basic conditions
def breedingBalls : Nat := 3
def snakesPerBall : Nat := 8
def pairsOfSnakes : Nat := 6
def snakesPerPair : Nat := 2

-- Define the total number of snakes
theorem total_snakes_count : 
  (breedingBalls * snakesPerBall) + (pairsOfSnakes * snakesPerPair) = 36 := 
by 
  -- we skip the proof with sorry
  sorry

end total_snakes_count_l420_42019


namespace earliest_time_meet_l420_42082

open Nat

def lap_time_anna := 5
def lap_time_bob := 8
def lap_time_carol := 10

def lcm_lap_times : ℕ :=
  Nat.lcm lap_time_anna (Nat.lcm lap_time_bob lap_time_carol)

theorem earliest_time_meet : lcm_lap_times = 40 := by
  sorry

end earliest_time_meet_l420_42082


namespace no_real_coeff_quadratic_with_roots_sum_and_product_l420_42033

theorem no_real_coeff_quadratic_with_roots_sum_and_product (a b c : ℝ) (h : a ≠ 0) :
  ¬ ∃ (α β : ℝ), (α = a + b + c) ∧ (β = a * b * c) ∧ (α + β = -b / a) ∧ (α * β = c / a) :=
by
  sorry

end no_real_coeff_quadratic_with_roots_sum_and_product_l420_42033


namespace sum_seven_consecutive_l420_42014

theorem sum_seven_consecutive (n : ℤ) : 
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 7 * n + 7 :=
by
  sorry

end sum_seven_consecutive_l420_42014


namespace find_k_l420_42041

theorem find_k 
  (x y: ℝ) 
  (h1: y = 5 * x + 3) 
  (h2: y = -2 * x - 25) 
  (h3: y = 3 * x + k) : 
  k = -5 :=
sorry

end find_k_l420_42041


namespace quadratic_roots_opposite_signs_l420_42089

theorem quadratic_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, (a * x^2 - (a + 3) * x + 2 = 0) ∧ (a * y^2 - (a + 3) * y + 2 = 0) ∧ x * y < 0) ↔ (a < 0) :=
sorry

end quadratic_roots_opposite_signs_l420_42089


namespace gini_coefficient_separate_gini_coefficient_combined_l420_42000

-- Definitions based on provided conditions
def northern_residents : ℕ := 24
def southern_residents : ℕ := 6
def price_per_set : ℝ := 2000
def northern_PPC (x : ℝ) : ℝ := 13.5 - 9 * x
def southern_PPC (x : ℝ) : ℝ := 1.5 * x^2 - 24

-- Gini Coefficient when both regions operate separately
theorem gini_coefficient_separate : 
  ∃ G : ℝ, G = 0.2 :=
  sorry

-- Gini Coefficient change when blending productions as per Northern conditions
theorem gini_coefficient_combined :
  ∃ ΔG : ℝ, ΔG = 0.001 :=
  sorry

end gini_coefficient_separate_gini_coefficient_combined_l420_42000


namespace ratio_of_refurb_to_new_tshirt_l420_42028

def cost_of_new_tshirt : ℤ := 5
def cost_of_pants : ℤ := 4
def cost_of_skirt : ℤ := 6

-- Total income from selling two new T-shirts, one pair of pants, four skirts, and six refurbished T-shirts is $53.
def total_income : ℤ := 53

-- Total income from selling new items.
def income_from_new_items : ℤ :=
  2 * cost_of_new_tshirt + cost_of_pants + 4 * cost_of_skirt

-- Income from refurbished T-shirts.
def income_from_refurb_tshirts : ℤ :=
  total_income - income_from_new_items

-- Number of refurbished T-shirts sold.
def num_refurb_tshirts_sold : ℤ := 6

-- Price of one refurbished T-shirt.
def cost_of_refurb_tshirt : ℤ :=
  income_from_refurb_tshirts / num_refurb_tshirts_sold

-- Prove the ratio of the price of a refurbished T-shirt to a new T-shirt is 0.5
theorem ratio_of_refurb_to_new_tshirt :
  (cost_of_refurb_tshirt : ℚ) / cost_of_new_tshirt = 0.5 := 
sorry

end ratio_of_refurb_to_new_tshirt_l420_42028


namespace minimum_value_2a_plus_3b_is_25_l420_42057

noncomputable def minimum_value_2a_plus_3b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (2 / a) + (3 / b) = 1) : ℝ :=
2 * a + 3 * b

theorem minimum_value_2a_plus_3b_is_25
  (a b : ℝ)
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : (2 / a) + (3 / b) = 1) :
  minimum_value_2a_plus_3b a b h₁ h₂ h₃ = 25 :=
sorry

end minimum_value_2a_plus_3b_is_25_l420_42057


namespace penny_makes_total_revenue_l420_42092

def price_per_slice : ℕ := 7
def slices_per_pie : ℕ := 6
def pies_sold : ℕ := 7

theorem penny_makes_total_revenue :
  (pies_sold * slices_per_pie) * price_per_slice = 294 := by
  sorry

end penny_makes_total_revenue_l420_42092


namespace no_real_roots_of_quadratic_l420_42036

theorem no_real_roots_of_quadratic 
  (a b c : ℝ) 
  (h1 : b - a + c > 0) 
  (h2 : b + a - c > 0) 
  (h3 : b - a - c < 0) 
  (h4 : b + a + c > 0) 
  (x : ℝ) : ¬ ∃ x : ℝ, a^2 * x^2 + (b^2 - a^2 - c^2) * x + c^2 = 0 := 
by
  sorry

end no_real_roots_of_quadratic_l420_42036


namespace present_age_of_dan_l420_42085

theorem present_age_of_dan (x : ℕ) : (x + 16 = 4 * (x - 8)) → x = 16 :=
by
  intro h
  sorry

end present_age_of_dan_l420_42085


namespace arithmetic_sequence_sum_l420_42061

noncomputable def Sn (a d n : ℕ) : ℕ :=
n * a + (n * (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a d : ℕ) (h1 : a = 3 * d) (h2 : Sn a d 5 = 50) : Sn a d 8 = 104 :=
by
/-
  From the given conditions:
  - \(a_4\) is the geometric mean of \(a_2\) and \(a_7\) implies \(a = 3d\)
  - Sum of first 5 terms is 50 implies \(S_5 = 50\)
  We need to prove \(S_8 = 104\)
-/
  sorry

end arithmetic_sequence_sum_l420_42061


namespace major_axis_range_l420_42002

theorem major_axis_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∀ x M N : ℝ, (x + (1 - x)) = 1 → x * (1 - x) = 0) 
  (e : ℝ) (h4 : (Real.sqrt 3 / 3) ≤ e ∧ e ≤ (Real.sqrt 2 / 2)) :
  ∃ a : ℝ, 2 * (Real.sqrt 5) ≤ 2 * a ∧ 2 * a ≤ 2 * (Real.sqrt 6) := 
sorry

end major_axis_range_l420_42002


namespace first_team_speed_l420_42001

theorem first_team_speed:
  ∃ v: ℝ, 
  (∀ (t: ℝ), t = 2.5 → 
  (∀ s: ℝ, s = 125 → 
  (v + 30) * t = s) ∧ v = 20) := 
  sorry

end first_team_speed_l420_42001


namespace area_of_moon_slice_l420_42017

-- Definitions of the conditions
def larger_circle_radius := 5
def larger_circle_center := (2, 0)
def smaller_circle_radius := 2
def smaller_circle_center := (0, 0)

-- Prove the area of the moon slice
theorem area_of_moon_slice : 
  (1/4) * (larger_circle_radius^2 * Real.pi) - (1/4) * (smaller_circle_radius^2 * Real.pi) = (21 * Real.pi) / 4 :=
by
  sorry

end area_of_moon_slice_l420_42017


namespace cost_of_ticket_when_Matty_was_born_l420_42020

theorem cost_of_ticket_when_Matty_was_born 
    (cost : ℕ → ℕ) 
    (h_halved : ∀ t : ℕ, cost (t + 10) = cost t / 2) 
    (h_age_30 : cost 30 = 125000) : 
    cost 0 = 1000000 := 
by 
  sorry

end cost_of_ticket_when_Matty_was_born_l420_42020


namespace mario_haircut_price_l420_42064

theorem mario_haircut_price (P : ℝ) 
  (weekend_multiplier : ℝ := 1.50)
  (sunday_price : ℝ := 27) 
  (weekend_price_eq : sunday_price = P * weekend_multiplier) : 
  P = 18 := 
by
  sorry

end mario_haircut_price_l420_42064


namespace calculate_area_of_triangle_l420_42074

theorem calculate_area_of_triangle :
  let p1 := (5, -2)
  let p2 := (5, 8)
  let p3 := (12, 8)
  let area := (1 / 2) * ((p2.2 - p1.2) * (p3.1 - p2.1))
  area = 35 := 
by
  sorry

end calculate_area_of_triangle_l420_42074


namespace simplify_expression1_simplify_expression2_l420_42094

-- Problem 1
theorem simplify_expression1 (a : ℝ) : 
  (a^2)^3 + 3 * a^4 * a^2 - a^8 / a^2 = 3 * a^6 :=
by sorry

-- Problem 2
theorem simplify_expression2 (x : ℝ) : 
  (x - 3) * (x + 4) - x * (x + 3) = -2 * x - 12 :=
by sorry

end simplify_expression1_simplify_expression2_l420_42094


namespace rise_in_water_level_correct_l420_42072

noncomputable def volume_of_rectangular_solid (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def area_of_circular_base (d : ℝ) : ℝ :=
  Real.pi * (d / 2) ^ 2

noncomputable def rise_in_water_level (solid_volume base_area : ℝ) : ℝ :=
  solid_volume / base_area

theorem rise_in_water_level_correct :
  let l := 10
  let w := 12
  let h := 15
  let d := 18
  let solid_volume := volume_of_rectangular_solid l w h
  let base_area := area_of_circular_base d
  let expected_rise := 7.07
  abs (rise_in_water_level solid_volume base_area - expected_rise) < 0.01 
:= 
by {
  sorry
}

end rise_in_water_level_correct_l420_42072


namespace count_integer_length_chords_l420_42071

/-- Point P is 9 units from the center of a circle with radius 15. -/
def point_distance_from_center : ℝ := 9

def circle_radius : ℝ := 15

/-- Correct answer to the number of different chords that contain P and have integer lengths. -/
def correct_answer : ℕ := 7

/-- Proving the number of chords containing P with integer lengths given the conditions. -/
theorem count_integer_length_chords : 
  ∀ (r_P : ℝ) (r_circle : ℝ), r_P = point_distance_from_center → r_circle = circle_radius → 
  (∃ n : ℕ, n = correct_answer) :=
by 
  intros r_P r_circle h1 h2
  use 7 
  sorry

end count_integer_length_chords_l420_42071


namespace value_of_expression_l420_42097

theorem value_of_expression (a : ℝ) (h : a^2 + a = 0) : 4*a^2 + 4*a + 2011 = 2011 :=
by
  sorry

end value_of_expression_l420_42097


namespace solitaire_game_removal_l420_42031

theorem solitaire_game_removal (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∃ moves : ℕ, ∀ i : ℕ, i < moves → (i + 1) % 2 = (i % 2) + 1) ↔ (m % 2 = 1 ∨ n % 2 = 1) :=
sorry

end solitaire_game_removal_l420_42031


namespace monotonic_increasing_interval_l420_42093

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (Real.sqrt (2 * x - x ^ 2))

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → 1 ≤ x ∧ x < 2 → ∀ x1 x2, x1 < x2 → f x1 ≤ f x2 :=
by
  sorry

end monotonic_increasing_interval_l420_42093


namespace number_of_kittens_l420_42065

-- Definitions for the given conditions.
def total_animals : ℕ := 77
def hamsters : ℕ := 15
def birds : ℕ := 30

-- The proof problem statement.
theorem number_of_kittens : total_animals - hamsters - birds = 32 := by
  sorry

end number_of_kittens_l420_42065


namespace molecular_weight_proof_l420_42021

def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_I : ℝ := 126.90

def molecular_weight (n_N n_H n_I : ℕ) : ℝ :=
  n_N * atomic_weight_N + n_H * atomic_weight_H + n_I * atomic_weight_I

theorem molecular_weight_proof : molecular_weight 1 4 1 = 144.95 :=
by {
  sorry
}

end molecular_weight_proof_l420_42021


namespace parallel_line_slope_l420_42050

theorem parallel_line_slope {x y : ℝ} (h : 3 * x + 6 * y = -24) : 
  ∀ m b : ℝ, (y = m * x + b) → m = -1 / 2 :=
sorry

end parallel_line_slope_l420_42050


namespace find_divisible_xy9z_l420_42096

-- Define a predicate for numbers divisible by 132
def divisible_by_132 (n : ℕ) : Prop :=
  n % 132 = 0

-- Define the given number form \(\overline{xy9z}\) as a number maker
def form_xy9z (x y z : ℕ) : ℕ :=
  1000 * x + 100 * y + 90 + z

-- Stating the theorem for finding all numbers of form \(\overline{xy9z}\) that are divisible by 132
theorem find_divisible_xy9z (x y z : ℕ) :
  (divisible_by_132 (form_xy9z x y z)) ↔
  form_xy9z x y z = 3696 ∨
  form_xy9z x y z = 4092 ∨
  form_xy9z x y z = 6996 ∨
  form_xy9z x y z = 7392 :=
by sorry

end find_divisible_xy9z_l420_42096


namespace y_is_multiple_of_2_y_is_multiple_of_3_y_is_multiple_of_6_y_is_multiple_of_9_l420_42073

def y : ℕ := 54 + 108 + 162 + 216 + 648 + 810 + 972

theorem y_is_multiple_of_2 : 2 ∣ y :=
sorry

theorem y_is_multiple_of_3 : 3 ∣ y :=
sorry

theorem y_is_multiple_of_6 : 6 ∣ y :=
sorry

theorem y_is_multiple_of_9 : 9 ∣ y :=
sorry

end y_is_multiple_of_2_y_is_multiple_of_3_y_is_multiple_of_6_y_is_multiple_of_9_l420_42073


namespace sally_oscillation_distance_l420_42042

noncomputable def C : ℝ := 5 / 4
noncomputable def D : ℝ := 11 / 4

theorem sally_oscillation_distance :
  abs (C - D) = 3 / 2 :=
by
  sorry

end sally_oscillation_distance_l420_42042


namespace number_of_sweet_potatoes_sold_to_mrs_adams_l420_42066

def sweet_potatoes_harvested := 80
def sweet_potatoes_sold_to_mr_lenon := 15
def sweet_potatoes_unsold := 45

def sweet_potatoes_sold_to_mrs_adams :=
  sweet_potatoes_harvested - sweet_potatoes_sold_to_mr_lenon - sweet_potatoes_unsold

theorem number_of_sweet_potatoes_sold_to_mrs_adams :
  sweet_potatoes_sold_to_mrs_adams = 20 := by
  sorry

end number_of_sweet_potatoes_sold_to_mrs_adams_l420_42066


namespace tan_diff_sin_double_l420_42048

theorem tan_diff (α : ℝ) (h : Real.tan α = 2) : 
  Real.tan (α - Real.pi / 4) = 1 / 3 := 
by 
  sorry

theorem sin_double (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4 / 5 := 
by 
  sorry

end tan_diff_sin_double_l420_42048


namespace ellipse_standard_equation_l420_42047

theorem ellipse_standard_equation (c a : ℝ) (h1 : 2 * c = 8) (h2 : 2 * a = 10) : 
  (∃ b : ℝ, b^2 = a^2 - c^2 ∧ ( ( ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ) ∨ ( ∀ x y : ℝ, x^2 / b^2 + y^2 / a^2 = 1 ) )) :=
by
  sorry

end ellipse_standard_equation_l420_42047


namespace length_of_ae_l420_42088

theorem length_of_ae
  (a b c d e : ℝ)
  (bc : ℝ)
  (cd : ℝ)
  (de : ℝ := 8)
  (ab : ℝ := 5)
  (ac : ℝ := 11)
  (h1 : bc = 2 * cd)
  (h2 : bc = ac - ab)
  : ab + bc + cd + de = 22 := 
by
  sorry

end length_of_ae_l420_42088


namespace students_wearing_specific_shirt_and_accessory_count_l420_42013

theorem students_wearing_specific_shirt_and_accessory_count :
  let total_students := 1000
  let blue_shirt_percent := 0.40
  let red_shirt_percent := 0.25
  let green_shirt_percent := 0.20
  let blue_shirt_students := blue_shirt_percent * total_students
  let red_shirt_students := red_shirt_percent * total_students
  let green_shirt_students := green_shirt_percent * total_students
  let blue_shirt_stripes_percent := 0.30
  let blue_shirt_polka_dots_percent := 0.35
  let red_shirt_stripes_percent := 0.20
  let red_shirt_polka_dots_percent := 0.40
  let green_shirt_stripes_percent := 0.25
  let green_shirt_polka_dots_percent := 0.25
  let accessory_hat_percent := 0.15
  let accessory_scarf_percent := 0.10
  let red_polka_dot_students := red_shirt_polka_dots_percent * red_shirt_students
  let red_polka_dot_hat_students := accessory_hat_percent * red_polka_dot_students
  let green_no_pattern_students := green_shirt_students - (green_shirt_stripes_percent * green_shirt_students + green_shirt_polka_dots_percent * green_shirt_students)
  let green_no_pattern_scarf_students := accessory_scarf_percent * green_no_pattern_students
  red_polka_dot_hat_students + green_no_pattern_scarf_students = 25 := by
    sorry

end students_wearing_specific_shirt_and_accessory_count_l420_42013


namespace odd_function_properties_l420_42008

theorem odd_function_properties
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 1 ≤ x ∧ x ≤ 3 ∧ 1 ≤ y ∧ y ≤ 3 ∧ x < y → f x < f y)
  (h_min_val : ∀ x, 1 ≤ x ∧ x ≤ 3 → 7 ≤ f x) :
  (∀ x y, -3 ≤ x ∧ x ≤ -1 ∧ -3 ≤ y ∧ y ≤ -1 ∧ x < y → f x < f y) ∧
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≤ -7) :=
sorry

end odd_function_properties_l420_42008


namespace sum_first_n_terms_geometric_sequence_l420_42095

def geometric_sequence_sum (n : ℕ) (k : ℝ) : ℝ :=
  if n = 0 then 0 else (3 * 2^n + k)

theorem sum_first_n_terms_geometric_sequence (k : ℝ) :
  (geometric_sequence_sum 1 k = 6 + k) ∧ 
  (∀ n > 1, geometric_sequence_sum n k - geometric_sequence_sum (n - 1) k = 3 * 2^(n-1))
  → k = -3 :=
by
  sorry

end sum_first_n_terms_geometric_sequence_l420_42095


namespace profit_shares_difference_l420_42025

theorem profit_shares_difference (total_profit : ℝ) (share_ratio_x share_ratio_y : ℝ) 
  (hx : share_ratio_x = 1/2) (hy : share_ratio_y = 1/3) (profit : ℝ):
  total_profit = 500 → profit = (total_profit * share_ratio_x) / ((share_ratio_x + share_ratio_y)) - (total_profit * share_ratio_y) / ((share_ratio_x + share_ratio_y)) → profit = 100 :=
by
  intros
  sorry

end profit_shares_difference_l420_42025


namespace range_of_f_2x_le_1_l420_42052

-- Given conditions
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

def cond_f_neg_2_eq_1 (f : ℝ → ℝ) : Prop :=
  f (-2) = 1

-- Main theorem
theorem range_of_f_2x_le_1 (f : ℝ → ℝ) 
  (h1 : is_odd f)
  (h2 : is_monotonically_decreasing f (Set.Iic 0))
  (h3 : cond_f_neg_2_eq_1 f) :
  Set.Icc (-1 : ℝ) 1 = { x | |f (2 * x)| ≤ 1 } :=
sorry

end range_of_f_2x_le_1_l420_42052


namespace divisors_large_than_8_fact_count_l420_42035

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem divisors_large_than_8_fact_count :
  let n := 9
  let factorial_n := factorial n
  let factorial_n_minus_1 := factorial (n - 1)
  ∃ (num_divisors : ℕ), num_divisors = 8 ∧
    (∀ d, d ∣ factorial_n → d > factorial_n_minus_1 ↔ ∃ k, k ∣ factorial_n ∧ k < 9) :=
by
  sorry

end divisors_large_than_8_fact_count_l420_42035


namespace value_range_of_quadratic_l420_42043

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem value_range_of_quadratic :
  ∀ x, -1 ≤ x ∧ x ≤ 2 → (2 : ℝ) ≤ quadratic_function x ∧ quadratic_function x ≤ 6 :=
by
  sorry

end value_range_of_quadratic_l420_42043


namespace sufficient_but_not_necessary_condition_for_subset_l420_42067

variable {A B : Set ℕ}
variable {a : ℕ}

theorem sufficient_but_not_necessary_condition_for_subset (hA : A = {1, a}) (hB : B = {1, 2, 3}) :
  (a = 3 → A ⊆ B) ∧ (A ⊆ B → (a = 3 ∨ a = 2)) ∧ ¬(A ⊆ B → a = 3) := by
sorry

end sufficient_but_not_necessary_condition_for_subset_l420_42067


namespace multiply_powers_same_base_l420_42091

theorem multiply_powers_same_base (a : ℝ) : a^4 * a^2 = a^6 := by
  sorry

end multiply_powers_same_base_l420_42091


namespace curve_equation_l420_42015

noncomputable def satisfies_conditions (f : ℝ → ℝ) (M₀ : ℝ × ℝ) : Prop :=
  (f M₀.1 = M₀.2) ∧ 
  (∀ (x y : ℝ) (h_tangent : ∀ x y, y = (f x) → x * y - 2 * (f x) * x = 0),
    y = f x → x * y / (y / x) = 2 * x)

theorem curve_equation (f : ℝ → ℝ) :
  satisfies_conditions f (1, 4) →
  (∀ x : ℝ, f x * x = 4) :=
by
  intro h
  sorry

end curve_equation_l420_42015


namespace distance_traveled_by_both_cars_l420_42005

def car_R_speed := 34.05124837953327
def car_P_speed := 44.05124837953327
def car_R_time := 8.810249675906654
def car_P_time := car_R_time - 2

def distance_car_R := car_R_speed * car_R_time
def distance_car_P := car_P_speed * car_P_time

theorem distance_traveled_by_both_cars :
  distance_car_R = 300 :=
by
  sorry

end distance_traveled_by_both_cars_l420_42005


namespace pennies_thrown_total_l420_42049

theorem pennies_thrown_total (rachelle_pennies gretchen_pennies rocky_pennies : ℕ) 
  (h1 : rachelle_pennies = 180)
  (h2 : gretchen_pennies = rachelle_pennies / 2)
  (h3 : rocky_pennies = gretchen_pennies / 3) : 
  rachelle_pennies + gretchen_pennies + rocky_pennies = 300 := 
by 
  sorry

end pennies_thrown_total_l420_42049


namespace school_distance_l420_42007

theorem school_distance (T D : ℝ) (h1 : 5 * (T + 6) = 630) (h2 : 7 * (T - 30) = 630) :
  D = 630 :=
sorry

end school_distance_l420_42007


namespace shaded_area_of_squares_l420_42078

theorem shaded_area_of_squares :
  let s_s := 4
  let s_L := 9
  let area_L := s_L * s_L
  let area_s := s_s * s_s
  area_L - area_s = 65 := sorry

end shaded_area_of_squares_l420_42078


namespace rectangular_field_length_l420_42022

theorem rectangular_field_length (w : ℝ) (h₁ : w * (w + 10) = 171) : w + 10 = 19 := 
by
  sorry

end rectangular_field_length_l420_42022


namespace pollen_mass_in_scientific_notation_l420_42046

theorem pollen_mass_in_scientific_notation : 
  ∃ c n : ℝ, 0.0000037 = c * 10^n ∧ 1 ≤ c ∧ c < 10 ∧ c = 3.7 ∧ n = -6 :=
sorry

end pollen_mass_in_scientific_notation_l420_42046


namespace total_flowers_purchased_l420_42077

-- Define the conditions
def sets : ℕ := 3
def pieces_per_set : ℕ := 90

-- State the proof problem
theorem total_flowers_purchased : sets * pieces_per_set = 270 :=
by
  sorry

end total_flowers_purchased_l420_42077


namespace smallest_b_greater_than_l420_42018

theorem smallest_b_greater_than (a b : ℤ) (h₁ : 9 < a) (h₂ : a < 21) (h₃ : 10 / b ≥ 2 / 3) (h₄ : b < 31) : 14 < b :=
sorry

end smallest_b_greater_than_l420_42018


namespace Loris_needs_more_books_l420_42016

noncomputable def books_needed (Loris Darryl Lamont : ℕ) :=
  (Lamont - Loris)

theorem Loris_needs_more_books
  (darryl_books: ℕ)
  (lamont_books: ℕ)
  (loris_books_total: ℕ)
  (total_books: ℕ)
  (h1: lamont_books = 2 * darryl_books)
  (h2: darryl_books = 20)
  (h3: loris_books_total + darryl_books + lamont_books = total_books)
  (h4: total_books = 97) :
  books_needed loris_books_total darryl_books lamont_books = 3 :=
sorry

end Loris_needs_more_books_l420_42016


namespace smallest_prime_with_digit_sum_23_l420_42086

-- Definition for the conditions
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The theorem stating the proof problem
theorem smallest_prime_with_digit_sum_23 : ∃ p : ℕ, Prime p ∧ sum_of_digits p = 23 ∧ p = 1993 := 
by {
 sorry
}

end smallest_prime_with_digit_sum_23_l420_42086


namespace smallest_b_for_composite_l420_42054

theorem smallest_b_for_composite (x : ℤ) : 
  ∃ b : ℕ, b > 0 ∧ Even b ∧ (∀ x : ℤ, ¬ Prime (x^4 + b^2)) ∧ b = 16 := 
by 
  sorry

end smallest_b_for_composite_l420_42054


namespace mul_add_distrib_l420_42037

theorem mul_add_distrib :
  15 * 36 + 15 * 24 = 900 := by
  sorry

end mul_add_distrib_l420_42037


namespace triangle_inequality_l420_42063

theorem triangle_inequality (a b c : ℝ) (habc_triangle : a + b > c ∧ b + c > a ∧ a + c > b) : 
  2 * (a^2 * b^2 + b^2 * c^2 + a^2 * c^2) > (a^4 + b^4 + c^4) :=
by
  sorry

end triangle_inequality_l420_42063


namespace exponent_multiplication_l420_42029

theorem exponent_multiplication :
  (10^(3/4)) * (10^(-0.25)) * (10^(1.5)) = 10^2 :=
by sorry

end exponent_multiplication_l420_42029


namespace chord_length_l420_42062

theorem chord_length (x y t : ℝ) (h₁ : x = 1 + 2 * t) (h₂ : y = 2 + t) (h_circle : x^2 + y^2 = 9) : 
  ∃ l, l = 12 / 5 * Real.sqrt 5 := 
sorry

end chord_length_l420_42062


namespace dice_sum_eight_dice_l420_42087

/--
  Given 8 fair 6-sided dice, prove that the number of ways to obtain
  a sum of 11 on the top faces of these dice, is 120.
-/
theorem dice_sum_eight_dice :
  (∃ n : ℕ, ∀ (dices : List ℕ), (dices.length = 8 ∧ (∀ d ∈ dices, 1 ≤ d ∧ d ≤ 6) 
   ∧ dices.sum = 11) → n = 120) :=
sorry

end dice_sum_eight_dice_l420_42087


namespace surcharge_X_is_2_17_percent_l420_42023

def priceX : ℝ := 575
def priceY : ℝ := 530
def surchargeY : ℝ := 0.03
def totalSaved : ℝ := 41.60

theorem surcharge_X_is_2_17_percent :
  let surchargeX := (2.17 / 100)
  let totalCostX := priceX + (priceX * surchargeX)
  let totalCostY := priceY + (priceY * surchargeY)
  (totalCostX - totalCostY = totalSaved) →
  surchargeX * 100 = 2.17 :=
by
  sorry

end surcharge_X_is_2_17_percent_l420_42023


namespace sum_place_values_of_specified_digits_l420_42010

def numeral := 95378637153370261

def place_values_of_3s := [3 * 100000000000, 3 * 10]
def place_values_of_7s := [7 * 10000000000, 7 * 1000000, 7 * 100]
def place_values_of_5s := [5 * 10000000000000, 5 * 1000, 5 * 10000, 5 * 1]

def sum_place_values (lst : List ℕ) : ℕ :=
  lst.foldl (· + ·) 0

def sum_of_place_values := 
  sum_place_values place_values_of_3s + 
  sum_place_values place_values_of_7s + 
  sum_place_values place_values_of_5s

theorem sum_place_values_of_specified_digits :
  sum_of_place_values = 350077055735 :=
by
  sorry

end sum_place_values_of_specified_digits_l420_42010


namespace pencils_in_drawer_l420_42079

theorem pencils_in_drawer (P : ℕ) 
  (h1 : 19 + 16 = 35)
  (h2 : P + 35 = 78) : 
  P = 43 := 
by
  sorry

end pencils_in_drawer_l420_42079


namespace number_of_solutions_l420_42055

open Real

theorem number_of_solutions :
  ∀ x : ℝ, (0 < x ∧ x < 3 * π) → (3 * cos x ^ 2 + 2 * sin x ^ 2 = 2) → 
  ∃ (L : Finset ℝ), L.card = 3 ∧ ∀ y ∈ L, 0 < y ∧ y < 3 * π ∧ 3 * cos y ^ 2 + 2 * sin y ^ 2 = 2 :=
by 
  sorry

end number_of_solutions_l420_42055


namespace pythagorean_triple_B_l420_42068

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_B : isPythagoreanTriple 3 4 5 :=
by
  sorry

end pythagorean_triple_B_l420_42068


namespace remainder_n_plus_1008_l420_42099

variable (n : ℕ)

theorem remainder_n_plus_1008 (h1 : n % 4 = 1) (h2 : n % 5 = 3) : (n + 1008) % 4 = 1 := by
  sorry

end remainder_n_plus_1008_l420_42099


namespace percent_larger_semicircles_l420_42039

theorem percent_larger_semicircles (r1 r2 : ℝ) (d1 d2 : ℝ)
  (hr1 : r1 = d1 / 2) (hr2 : r2 = d2 / 2)
  (hd1 : d1 = 12) (hd2 : d2 = 8) : 
  (2 * (1/2) * Real.pi * r1^2) = (9/4 * (2 * (1/2) * Real.pi * r2^2)) :=
by
  sorry

end percent_larger_semicircles_l420_42039


namespace bottle_caps_total_l420_42032

-- Define the conditions
def groups : ℕ := 7
def caps_per_group : ℕ := 5

-- State the theorem
theorem bottle_caps_total : groups * caps_per_group = 35 :=
by
  sorry

end bottle_caps_total_l420_42032


namespace prime_add_eq_2001_l420_42084

theorem prime_add_eq_2001 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^2 + b = 2003) : a + b = 2001 :=
sorry

end prime_add_eq_2001_l420_42084


namespace teresa_ahmad_equation_l420_42076

theorem teresa_ahmad_equation (b c : ℝ) :
  (∀ x : ℝ, |x - 4| = 3 ↔ x = 7 ∨ x = 1) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ x = 7 ∨ x = 1) →
  (b, c) = (-8, 7) :=
by
  intro h1 h2
  sorry

end teresa_ahmad_equation_l420_42076


namespace first_player_wins_l420_42058

def wins (sum_rows sum_cols : ℕ) : Prop := sum_rows > sum_cols

theorem first_player_wins 
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ)
  (h : a_1 > a_2 ∧ a_2 > a_3 ∧ a_3 > a_4 ∧ a_4 > a_5 ∧ a_5 > a_6 ∧ a_6 > a_7 ∧ a_7 > a_8 ∧ a_8 > a_9) :
  ∃ sum_rows sum_cols, wins sum_rows sum_cols :=
sorry

end first_player_wins_l420_42058
