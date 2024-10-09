import Mathlib

namespace coupon_value_l1081_108102

theorem coupon_value (C : ℝ) (original_price : ℝ := 120) (final_price : ℝ := 99) 
(membership_discount : ℝ := 0.1) (reduced_price : ℝ := original_price - C) :
0.9 * reduced_price = final_price → C = 10 :=
by sorry

end coupon_value_l1081_108102


namespace cos_tan_values_l1081_108191

theorem cos_tan_values (α : ℝ) (h : Real.sin α = -1 / 2) :
  (∃ (quadrant : ℕ), 
    (quadrant = 3 ∧ Real.cos α = -Real.sqrt 3 / 2 ∧ Real.tan α = Real.sqrt 3 / 3) ∨ 
    (quadrant = 4 ∧ Real.cos α = Real.sqrt 3 / 2 ∧ Real.tan α = -Real.sqrt 3 / 3)) :=
sorry

end cos_tan_values_l1081_108191


namespace necessary_but_not_sufficient_l1081_108182

theorem necessary_but_not_sufficient (x : ℝ) : ( (x + 1) * (x + 2) > 0 → (x + 1) * (x^2 + 2) > 0 ) :=
by
  intro h
  -- insert steps urther here, if proof was required
  sorry

end necessary_but_not_sufficient_l1081_108182


namespace crayon_count_l1081_108195

theorem crayon_count (initial_crayons eaten_crayons : ℕ) (h1 : initial_crayons = 62) (h2 : eaten_crayons = 52) : initial_crayons - eaten_crayons = 10 := 
by 
  sorry

end crayon_count_l1081_108195


namespace maximize_profit_l1081_108144

/-- 
The total number of rooms in the hotel 
-/
def totalRooms := 80

/-- 
The initial rent when the hotel is fully booked 
-/
def initialRent := 160

/-- 
The loss in guests for each increase in rent by 20 yuan 
-/
def guestLossPerIncrease := 3

/-- 
The increase in rent 
-/
def increasePer20Yuan := 20

/-- 
The daily service and maintenance cost per occupied room
-/
def costPerOccupiedRoom := 40

/-- 
Maximize profit given the conditions
-/
theorem maximize_profit : 
  ∃ x : ℕ, x = 360 ∧ 
            ∀ y : ℕ,
              (initialRent - costPerOccupiedRoom) * (totalRooms - guestLossPerIncrease * (x - initialRent) / increasePer20Yuan)
              ≥ (y - costPerOccupiedRoom) * (totalRooms - guestLossPerIncrease * (y - initialRent) / increasePer20Yuan) := 
sorry

end maximize_profit_l1081_108144


namespace time_calculation_correct_l1081_108131

theorem time_calculation_correct :
  let start_hour := 3
  let start_minute := 0
  let start_second := 0
  let hours_to_add := 158
  let minutes_to_add := 55
  let seconds_to_add := 32
  let total_seconds := seconds_to_add + minutes_to_add * 60 + hours_to_add * 3600
  let new_hour := (start_hour + (total_seconds / 3600) % 12) % 12
  let new_minute := (start_minute + (total_seconds / 60) % 60) % 60
  let new_second := (start_second + total_seconds % 60) % 60
  let A := new_hour
  let B := new_minute
  let C := new_second
  A + B + C = 92 :=
by
  sorry

end time_calculation_correct_l1081_108131


namespace exists_group_of_three_friends_l1081_108157

-- Defining the context of the problem
def people := Fin 10 -- a finite set of 10 people
def quarrel (x y : people) : Prop := -- a predicate indicating a quarrel between two people
sorry

-- Given conditions
axiom quarreled_pairs : ∃ S : Finset (people × people), S.card = 14 ∧ 
  ∀ {x y : people}, (x, y) ∈ S → x ≠ y ∧ quarrel x y

-- Question: Prove there exists a set of 3 friends among these 10 people
theorem exists_group_of_three_friends (p : Finset people):
  ∃ (group : Finset people), group.card = 3 ∧ ∀ {x y : people}, 
  x ∈ group → y ∈ group → x ≠ y → ¬ quarrel x y :=
sorry

end exists_group_of_three_friends_l1081_108157


namespace man_average_interest_rate_l1081_108194

noncomputable def average_rate_of_interest (total_investment : ℝ) (rate1 rate2 rate_average : ℝ) 
    (x : ℝ) (same_return : (rate1 * (total_investment - x) = rate2 * x)) : Prop :=
  (rate_average = ((rate1 * (total_investment - x) + rate2 * x) / total_investment))

theorem man_average_interest_rate
    (total_investment : ℝ) 
    (rate1 : ℝ)
    (rate2 : ℝ)
    (rate_average : ℝ)
    (x : ℝ)
    (same_return : rate1 * (total_investment - x) = rate2 * x) :
    total_investment = 4500 ∧ rate1 = 0.04 ∧ rate2 = 0.06 ∧ x = 1800 ∧ rate_average = 0.048 → 
    average_rate_of_interest total_investment rate1 rate2 rate_average x same_return := 
by
  sorry

end man_average_interest_rate_l1081_108194


namespace range_of_b_l1081_108192

theorem range_of_b (a b c m : ℝ) (h_ge_seq : c = b * b / a) (h_sum : a + b + c = m) (h_pos_a : a > 0) (h_pos_m : m > 0) : 
  (-m ≤ b ∧ b < 0) ∨ (0 < b ∧ b ≤ m / 3) :=
by
  sorry

end range_of_b_l1081_108192


namespace find_min_value_expression_l1081_108133

noncomputable def minValueExpression (θ : ℝ) : ℝ :=
  2 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 2 * Real.tan θ

theorem find_min_value_expression :
  ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 ∧ minValueExpression θ = 3 * Real.sqrt 2 :=
sorry

end find_min_value_expression_l1081_108133


namespace initial_contestants_proof_l1081_108198

noncomputable def initial_contestants (final_round : ℕ) : ℕ :=
  let fraction_remaining := 2 / 5
  let fraction_advancing := 1 / 2
  let fraction_final := fraction_remaining * fraction_advancing
  (final_round : ℕ) / fraction_final

theorem initial_contestants_proof : initial_contestants 30 = 150 :=
sorry

end initial_contestants_proof_l1081_108198


namespace boxes_of_apples_l1081_108176

theorem boxes_of_apples (n_crates apples_per_crate rotten_apples apples_per_box : ℕ) 
  (h1 : n_crates = 12) 
  (h2 : apples_per_crate = 42)
  (h3: rotten_apples = 4) 
  (h4 : apples_per_box = 10) : 
  (n_crates * apples_per_crate - rotten_apples) / apples_per_box = 50 :=
by
  sorry

end boxes_of_apples_l1081_108176


namespace check_bag_correct_l1081_108121

-- Define the conditions as variables and statements
variables (uber_to_house : ℕ) (uber_to_airport : ℕ) (check_bag : ℕ)
          (security : ℕ) (wait_for_boarding : ℕ) (wait_for_takeoff : ℕ) (total_time : ℕ)

-- Assign the given conditions
def given_conditions : Prop :=
  uber_to_house = 10 ∧
  uber_to_airport = 5 * uber_to_house ∧
  security = 3 * check_bag ∧
  wait_for_boarding = 20 ∧
  wait_for_takeoff = 2 * wait_for_boarding ∧
  total_time = 180

-- Define the question as a statement
def check_bag_time (check_bag : ℕ) : Prop :=
  check_bag = 15

-- The Lean theorem based on the problem, conditions, and answer
theorem check_bag_correct :
  given_conditions uber_to_house uber_to_airport check_bag security wait_for_boarding wait_for_takeoff total_time →
  check_bag_time check_bag :=
by
  intros h
  sorry

end check_bag_correct_l1081_108121


namespace rice_mixing_ratio_l1081_108104

theorem rice_mixing_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (4.5 * x + 8.75 * y) / (x + y) = 7.5 → y / x = 2.4 :=
by
  sorry

end rice_mixing_ratio_l1081_108104


namespace coffee_cost_l1081_108170

theorem coffee_cost :
  ∃ y : ℕ, 
  (∃ x : ℕ, 3 * x + 2 * y = 630 ∧ 2 * x + 3 * y = 690) → y = 162 :=
by
  sorry

end coffee_cost_l1081_108170


namespace different_product_l1081_108158

theorem different_product :
  let P1 := 190 * 80
  let P2 := 19 * 800
  let P3 := 19 * 8 * 10
  let P4 := 19 * 8 * 100
  P3 ≠ P1 ∧ P3 ≠ P2 ∧ P3 ≠ P4 :=
by
  sorry

end different_product_l1081_108158


namespace selection_count_l1081_108141

theorem selection_count :
  (Nat.choose 6 3) * (Nat.choose 5 2) = 200 := 
sorry

end selection_count_l1081_108141


namespace solve_sin_cos_eqn_l1081_108186

theorem solve_sin_cos_eqn (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x + Real.cos x = 1) :
  x = 0 ∨ x = Real.pi / 2 :=
sorry

end solve_sin_cos_eqn_l1081_108186


namespace math_problem_l1081_108183

theorem math_problem
  (z : ℝ)
  (hz : z = 80)
  (y : ℝ)
  (hy : y = (1/4) * z)
  (x : ℝ)
  (hx : x = (1/3) * y)
  (w : ℝ)
  (hw : w = x + y + z) :
  x = 20 / 3 ∧ w = 320 / 3 :=
by
  sorry

end math_problem_l1081_108183


namespace alice_bob_sum_proof_l1081_108179

noncomputable def alice_bob_sum_is_22 : Prop :=
  ∃ A B : ℕ, (1 ≤ A ∧ A ≤ 50) ∧ (1 ≤ B ∧ B ≤ 50) ∧ (B % 3 = 0) ∧ (∃ k : ℕ, 2 * B + A = k^2) ∧ (A + B = 22)

theorem alice_bob_sum_proof : alice_bob_sum_is_22 :=
sorry

end alice_bob_sum_proof_l1081_108179


namespace expand_expression_l1081_108106

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l1081_108106


namespace find_third_divisor_l1081_108163

theorem find_third_divisor (n : ℕ) (d : ℕ) 
  (h1 : (n - 4) % 12 = 0)
  (h2 : (n - 4) % 16 = 0)
  (h3 : (n - 4) % d = 0)
  (h4 : (n - 4) % 21 = 0)
  (h5 : (n - 4) % 28 = 0)
  (h6 : n = 1012) :
  d = 3 :=
by
  sorry

end find_third_divisor_l1081_108163


namespace ratio_of_cream_l1081_108159

theorem ratio_of_cream
  (joes_initial_coffee : ℕ := 20)
  (joe_cream_added : ℕ := 3)
  (joe_amount_drank : ℕ := 4)
  (joanns_initial_coffee : ℕ := 20)
  (joann_amount_drank : ℕ := 4)
  (joann_cream_added : ℕ := 3) :
  let joe_final_cream := (joe_cream_added - joe_amount_drank * (joe_cream_added / (joe_cream_added + joes_initial_coffee)))
  let joann_final_cream := joann_cream_added
  (joe_final_cream / joanns_initial_coffee + joann_cream_added = 15 / 23) :=
sorry

end ratio_of_cream_l1081_108159


namespace product_of_bc_l1081_108145

theorem product_of_bc
  (b c : Int)
  (h1 : ∀ r, r^2 - r - 1 = 0 → r^5 - b * r - c = 0) :
  b * c = 15 :=
by
  -- We start the proof assuming the conditions
  sorry

end product_of_bc_l1081_108145


namespace min_value_reciprocal_sum_l1081_108113

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∀ (c : ℝ), c = (1 / a) + (4 / b) → c ≥ 9 :=
by
  intros c hc
  sorry

end min_value_reciprocal_sum_l1081_108113


namespace maximum_cows_l1081_108172

theorem maximum_cows (s c : ℕ) (h1 : 30 * s + 33 * c = 1300) (h2 : c > 2 * s) : c ≤ 30 :=
by
  -- Proof would go here
  sorry

end maximum_cows_l1081_108172


namespace Alice_favorite_number_l1081_108174

theorem Alice_favorite_number :
  ∃ n : ℕ, (30 ≤ n ∧ n ≤ 70) ∧ (7 ∣ n) ∧ ¬(3 ∣ n) ∧ (4 ∣ (n / 10 + n % 10)) ∧ n = 35 :=
by
  sorry

end Alice_favorite_number_l1081_108174


namespace equal_triples_l1081_108160

theorem equal_triples (a b c x : ℝ) (h_abc : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : (xb + (1 - x) * c) / a = (x * c + (1 - x) * a) / b ∧ 
          (x * c + (1 - x) * a) / b = (x * a + (1 - x) * b) / c) : a = b ∧ b = c := by
  sorry

end equal_triples_l1081_108160


namespace find_constants_l1081_108127

theorem find_constants (A B C : ℤ) (h1 : 1 = A + B) (h2 : -2 = C) (h3 : 5 = -A) :
  A = -5 ∧ B = 6 ∧ C = -2 :=
by {
  sorry
}

end find_constants_l1081_108127


namespace harry_drank_last_mile_l1081_108117

theorem harry_drank_last_mile :
  ∀ (T D start_water end_water leak_rate drink_rate leak_time first_miles : ℕ),
    start_water = 10 →
    end_water = 2 →
    leak_rate = 1 →
    leak_time = 2 →
    drink_rate = 1 →
    first_miles = 3 →
    T = leak_rate * leak_time →
    D = drink_rate * first_miles →
    start_water - end_water = T + D + (start_water - end_water - T - D) →
    start_water - end_water - T - D = 3 :=
by
  sorry

end harry_drank_last_mile_l1081_108117


namespace instantaneous_velocity_at_1_l1081_108193

noncomputable def S (t : ℝ) : ℝ := t^2 + 2 * t

theorem instantaneous_velocity_at_1 : (deriv S 1) = 4 :=
by 
  -- The proof is left as an exercise
  sorry

end instantaneous_velocity_at_1_l1081_108193


namespace minimum_floor_sum_l1081_108185

theorem minimum_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ⌊a^2 + b^2 / c⌋ + ⌊b^2 + c^2 / a⌋ + ⌊c^2 + a^2 / b⌋ = 34 :=
sorry

end minimum_floor_sum_l1081_108185


namespace probability_top_three_same_color_l1081_108138

/-- 
  A theorem stating the probability that the top three cards from a shuffled 
  standard deck of 52 cards are all of the same color is \(\frac{12}{51}\).
-/
theorem probability_top_three_same_color : 
  let deck := 52
  let colors := 2
  let cards_per_color := 26
  let favorable_outcomes := 2 * 26 * 25 * 24
  let total_outcomes := 52 * 51 * 50
  favorable_outcomes / total_outcomes = 12 / 51 :=
by
  sorry

end probability_top_three_same_color_l1081_108138


namespace triangle_min_value_l1081_108139

open Real

theorem triangle_min_value
  (A B C : ℝ)
  (h_triangle: A + B + C = π)
  (h_sin: sin (2 * A + B) = 2 * sin B) :
  tan A + tan C + 2 / tan B ≥ 2 :=
sorry

end triangle_min_value_l1081_108139


namespace beads_cost_is_three_l1081_108153

-- Define the given conditions
def cost_of_string_per_bracelet : Nat := 1
def selling_price_per_bracelet : Nat := 6
def number_of_bracelets_sold : Nat := 25
def total_profit : Nat := 50

-- The amount spent on beads per bracelet
def amount_spent_on_beads_per_bracelet (B : Nat) : Prop :=
  B = (total_profit + number_of_bracelets_sold * (cost_of_string_per_bracelet + B) - number_of_bracelets_sold * selling_price_per_bracelet) / number_of_bracelets_sold 

-- The main goal is to prove that the amount spent on beads is 3
theorem beads_cost_is_three : amount_spent_on_beads_per_bracelet 3 :=
by sorry

end beads_cost_is_three_l1081_108153


namespace building_houses_200_people_l1081_108118

-- Define number of floors, apartments per floor, and people per apartment as constants
def numFloors := 25
def apartmentsPerFloor := 4
def peoplePerApartment := 2

-- Define the total number of apartments
def totalApartments := numFloors * apartmentsPerFloor

-- Define the total number of people
def totalPeople := totalApartments * peoplePerApartment

theorem building_houses_200_people : totalPeople = 200 :=
by
  sorry

end building_houses_200_people_l1081_108118


namespace average_percentage_15_students_l1081_108119

-- Define the average percentage of the 15 students
variable (x : ℝ)

-- Condition 1: Total percentage for the 15 students is 15 * x
def total_15_students : ℝ := 15 * x

-- Condition 2: Total percentage for the 10 students who averaged 88%
def total_10_students : ℝ := 10 * 88

-- Condition 3: Total percentage for all 25 students who averaged 79%
def total_all_students : ℝ := 25 * 79

-- Mathematical problem: Prove that x = 73 given the conditions.
theorem average_percentage_15_students (h : total_15_students x + total_10_students = total_all_students) : x = 73 := 
by
  sorry

end average_percentage_15_students_l1081_108119


namespace sqrt_conjecture_l1081_108196

theorem sqrt_conjecture (n : ℕ) (h : n ≥ 1) : 
  (Real.sqrt (n + (1 / (n + 2)))) = ((n + 1) * Real.sqrt (1 / (n + 2))) :=
sorry

end sqrt_conjecture_l1081_108196


namespace area_triangle_3_6_l1081_108146

/-
Problem: Prove that the area of a triangle with base 3 meters and height 6 meters is 9 square meters.
Definitions: 
- base: The base of the triangle is 3 meters.
- height: The height of the triangle is 6 meters.
Conditions: 
- The area of a triangle formula.
Correct Answer: 9 square meters.
-/

def area_of_triangle (base height : ℕ) : ℕ := (base * height) / 2

theorem area_triangle_3_6 : area_of_triangle 3 6 = 9 := by
  sorry

end area_triangle_3_6_l1081_108146


namespace value_of_a_27_l1081_108156

def a_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + 2 * n

theorem value_of_a_27 (a : ℕ → ℕ) (h : a_sequence a) : a 27 = 702 :=
sorry

end value_of_a_27_l1081_108156


namespace arctan_sum_l1081_108187

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l1081_108187


namespace no_real_roots_abs_eq_l1081_108129

theorem no_real_roots_abs_eq (x : ℝ) : 
  |2*x - 5| + |3*x - 7| + |5*x - 11| = 2015/2016 → false :=
by sorry

end no_real_roots_abs_eq_l1081_108129


namespace part_I_part_II_l1081_108162

def f (x : ℝ) : ℝ := abs (2 * x - 7) + 1

def g (x : ℝ) : ℝ := abs (2 * x - 7) - 2 * abs (x - 1) + 1

theorem part_I :
  {x : ℝ | f x ≤ x} = {x : ℝ | (8 / 3) ≤ x ∧ x ≤ 6} := sorry

theorem part_II (a : ℝ) :
  (∃ x : ℝ, g x ≤ a) → a ≥ -4 := sorry

end part_I_part_II_l1081_108162


namespace green_competition_l1081_108137

theorem green_competition {x : ℕ} (h : 0 ≤ x ∧ x ≤ 25) : 
  5 * x - (25 - x) ≥ 85 :=
by
  sorry

end green_competition_l1081_108137


namespace probability_no_practice_l1081_108190

def prob_has_practice : ℚ := 5 / 8

theorem probability_no_practice : 
  1 - prob_has_practice = 3 / 8 := 
by
  sorry

end probability_no_practice_l1081_108190


namespace value_of_expression_l1081_108109

theorem value_of_expression (x : ℤ) (h : x = 3) : x^6 - 3 * x = 720 := by
  sorry

end value_of_expression_l1081_108109


namespace total_votes_is_correct_l1081_108116

-- Definitions and theorem statement
theorem total_votes_is_correct (T : ℝ) 
  (votes_for_A : ℝ) 
  (candidate_A_share : ℝ) 
  (valid_vote_fraction : ℝ) 
  (invalid_vote_fraction : ℝ) 
  (votes_for_A_equals: votes_for_A = 380800) 
  (candidate_A_share_equals: candidate_A_share = 0.80) 
  (valid_vote_fraction_equals: valid_vote_fraction = 0.85) 
  (invalid_vote_fraction_equals: invalid_vote_fraction = 0.15) 
  (valid_vote_computed: votes_for_A = candidate_A_share * valid_vote_fraction * T): 
  T = 560000 := 
by 
  sorry

end total_votes_is_correct_l1081_108116


namespace fraction_sum_equals_l1081_108105

theorem fraction_sum_equals : 
    (4 / 2) + (7 / 4) + (11 / 8) + (21 / 16) + (41 / 32) + (81 / 64) - 8 = 63 / 64 :=
by 
    sorry

end fraction_sum_equals_l1081_108105


namespace probability_of_PAIR_letters_in_PROBABILITY_l1081_108128

theorem probability_of_PAIR_letters_in_PROBABILITY : 
  let total_letters := 11
  let favorable_letters := 4
  favorable_letters / total_letters = 4 / 11 :=
by
  let total_letters := 11
  let favorable_letters := 4
  show favorable_letters / total_letters = 4 / 11
  sorry

end probability_of_PAIR_letters_in_PROBABILITY_l1081_108128


namespace notes_count_l1081_108115

theorem notes_count (x : ℕ) (num_2_yuan num_5_yuan num_10_yuan total_notes total_amount : ℕ) 
    (h1 : total_amount = 160)
    (h2 : total_notes = 25)
    (h3 : num_5_yuan = x)
    (h4 : num_10_yuan = x)
    (h5 : num_2_yuan = total_notes - 2 * x)
    (h6 : 2 * num_2_yuan + 5 * num_5_yuan + 10 * num_10_yuan = total_amount) :
    num_5_yuan = 10 ∧ num_10_yuan = 10 ∧ num_2_yuan = 5 :=
by
  sorry

end notes_count_l1081_108115


namespace calculate_total_cost_l1081_108114

-- Define the cost per workbook
def cost_per_workbook (x : ℝ) : ℝ := x

-- Define the number of workbooks
def number_of_workbooks : ℝ := 400

-- Define the total cost calculation
def total_cost (x : ℝ) : ℝ := number_of_workbooks * cost_per_workbook x

-- State the theorem to prove
theorem calculate_total_cost (x : ℝ) : total_cost x = 400 * x :=
by sorry

end calculate_total_cost_l1081_108114


namespace least_five_digit_congruent_to_5_mod_15_l1081_108199

theorem least_five_digit_congruent_to_5_mod_15 : ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 15 = 5 ∧ n = 10010 := by
  sorry

end least_five_digit_congruent_to_5_mod_15_l1081_108199


namespace negation_of_proposition_l1081_108173

noncomputable def P (x : ℝ) : Prop := x^2 + 1 ≥ 0

theorem negation_of_proposition :
  (¬ ∀ x, x > 1 → P x) ↔ (∃ x, x > 1 ∧ ¬ P x) :=
sorry

end negation_of_proposition_l1081_108173


namespace adam_coins_value_l1081_108167

theorem adam_coins_value (num_coins : ℕ) (subset_value: ℕ) (subset_num: ℕ) (total_value: ℕ)
  (h1 : num_coins = 20)
  (h2 : subset_value = 16)
  (h3 : subset_num = 4)
  (h4 : total_value = num_coins * (subset_value / subset_num)) :
  total_value = 80 := 
by
  sorry

end adam_coins_value_l1081_108167


namespace fourth_term_of_geometric_sequence_l1081_108164

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) :=
  a * r ^ (n - 1)

theorem fourth_term_of_geometric_sequence 
  (a : ℝ) (r : ℝ) (ar5_eq : a * r ^ 5 = 32) 
  (a_eq : a = 81) :
  geometric_sequence a r 4 = 24 := 
by 
  sorry

end fourth_term_of_geometric_sequence_l1081_108164


namespace polynomial_value_at_2_l1081_108101

def f (x : ℝ) : ℝ := 2 * x^5 + 4 * x^4 - 2 * x^3 - 3 * x^2 + x

theorem polynomial_value_at_2 : f 2 = 102 := by
  sorry

end polynomial_value_at_2_l1081_108101


namespace sixth_graders_more_than_seventh_l1081_108154

theorem sixth_graders_more_than_seventh
  (bookstore_sells_pencils_in_whole_cents : True)
  (seventh_graders : ℕ)
  (sixth_graders : ℕ)
  (seventh_packs_payment : ℕ)
  (sixth_packs_payment : ℕ)
  (each_pack_contains_two_pencils : True)
  (seventh_graders_condition : seventh_graders = 25)
  (seventh_packs_payment_condition : seventh_packs_payment * seventh_graders = 275)
  (sixth_graders_condition : sixth_graders = 36 / 2)
  (sixth_packs_payment_condition : sixth_packs_payment * sixth_graders = 216) : 
  sixth_graders - seventh_graders = 7 := sorry

end sixth_graders_more_than_seventh_l1081_108154


namespace assign_roles_l1081_108122

def maleRoles : ℕ := 3
def femaleRoles : ℕ := 3
def eitherGenderRoles : ℕ := 4
def menCount : ℕ := 7
def womenCount : ℕ := 8

theorem assign_roles : 
  (menCount.choose maleRoles) * 
  (womenCount.choose femaleRoles) * 
  ((menCount + womenCount - maleRoles - femaleRoles).choose eitherGenderRoles) = 213955200 := 
  sorry

end assign_roles_l1081_108122


namespace find_interval_l1081_108108

theorem find_interval (x : ℝ) : x^2 + 3*x < 10 ↔ -5 < x ∧ x < 2 :=
by
  sorry

end find_interval_l1081_108108


namespace describe_graph_l1081_108107

noncomputable def points_satisfying_equation (x y : ℝ) : Prop :=
  (x - y) ^ 2 = x ^ 2 + y ^ 2

theorem describe_graph : {p : ℝ × ℝ | points_satisfying_equation p.1 p.2} = {p : ℝ × ℝ | p.1 = 0} ∪ {p : ℝ × ℝ | p.2 = 0} :=
by
  sorry

end describe_graph_l1081_108107


namespace arithmetic_progression_num_terms_l1081_108124

theorem arithmetic_progression_num_terms (a d n : ℕ) (h_even : n % 2 = 0) 
    (h_sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 30)
    (h_sum_even : (n / 2) * (2 * a + 2 * d + (n - 2) * d) = 36)
    (h_diff_last_first : (n - 1) * d = 12) :
    n = 8 := 
sorry

end arithmetic_progression_num_terms_l1081_108124


namespace solution_set_of_inequality_l1081_108148

theorem solution_set_of_inequality :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
sorry

end solution_set_of_inequality_l1081_108148


namespace number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_l1081_108197

def num_ways_to_make_125_quacks_using_coins : ℕ :=
  have h : ∃ (a b c d : ℕ), a + 5 * b + 25 * c + 125 * d = 125 := sorry
  82

theorem number_of_ways_to_make_125_quacks_using_1_5_25_125_coins : num_ways_to_make_125_quacks_using_coins = 82 := 
  sorry

end number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_l1081_108197


namespace find_a_l1081_108166

-- Given function
def quadratic_func (a x : ℝ) := a * (x - 1)^2 - a

-- Conditions
def condition1 (a : ℝ) := a ≠ 0
def condition2 (x : ℝ) := -1 ≤ x ∧ x ≤ 4
def min_value (y : ℝ) := y = -4

theorem find_a (a : ℝ) (ha : condition1 a) :
  ∃ a, (∀ x, condition2 x → quadratic_func a x = -4) → (a = 4 ∨ a = -1 / 2) :=
sorry

end find_a_l1081_108166


namespace oil_bill_increase_l1081_108110

theorem oil_bill_increase :
  ∀ (F x : ℝ), 
    (F / 120 = 5 / 4) → 
    ((F + x) / 120 = 3 / 2) → 
    x = 30 :=
by
  intros F x h1 h2
  -- proof
  sorry

end oil_bill_increase_l1081_108110


namespace integer_solution_n_l1081_108100

theorem integer_solution_n 
  (n : Int) 
  (h1 : n + 13 > 15) 
  (h2 : -6 * n > -18) : 
  n = 2 := 
sorry

end integer_solution_n_l1081_108100


namespace average_apples_per_hour_l1081_108161

theorem average_apples_per_hour (total_apples : ℝ) (total_hours : ℝ) (h1 : total_apples = 5.0) (h2 : total_hours = 3.0) : total_apples / total_hours = 1.67 :=
  sorry

end average_apples_per_hour_l1081_108161


namespace y_order_of_quadratic_l1081_108189

theorem y_order_of_quadratic (k : ℝ) (y1 y2 y3 : ℝ) :
  (y1 = (-4)^2 + 4 * (-4) + k) → 
  (y2 = (-1)^2 + 4 * (-1) + k) → 
  (y3 = (1)^2 + 4 * (1) + k) → 
  y2 < y1 ∧ y1 < y3 :=
by
  intro hy1 hy2 hy3
  sorry

end y_order_of_quadratic_l1081_108189


namespace sphere_volume_ratio_l1081_108125

theorem sphere_volume_ratio
  (r R : ℝ)
  (h : (4:ℝ) * π * r^2 / (4 * π * R^2) = (4:ℝ) / 9) : 
  (r^3 / R^3 = (8:ℝ) / 27) := by
  sorry

end sphere_volume_ratio_l1081_108125


namespace monica_tiles_l1081_108177

theorem monica_tiles (room_length : ℕ) (room_width : ℕ) (border_tile_size : ℕ) (inner_tile_size : ℕ) 
  (border_tiles : ℕ) (inner_tiles : ℕ) (total_tiles : ℕ) :
  room_length = 24 ∧ room_width = 18 ∧ border_tile_size = 2 ∧ inner_tile_size = 3 ∧ 
  border_tiles = 38 ∧ inner_tiles = 32 → total_tiles = 70 :=
by {
  sorry
}

end monica_tiles_l1081_108177


namespace x_intercept_of_l1_is_2_l1081_108147

theorem x_intercept_of_l1_is_2 (a : ℝ) (l1_perpendicular_l2 : ∀ (x y : ℝ), 
  ((a+3)*x + y - 4 = 0) -> (x + (a-1)*y + 4 = 0) -> False) : 
  ∃ b : ℝ, (2*b + 0 - 4 = 0) ∧ b = 2 := 
by
  sorry

end x_intercept_of_l1_is_2_l1081_108147


namespace correct_option_is_B_l1081_108152

theorem correct_option_is_B (a : ℝ) : 
  (¬ (-2 * a^2 * b)^3 = -6 * a^6 * b^3) ∧
  (a^7 / a = a^6) ∧
  (¬ (a + 1)^2 = a^2 + 1) ∧
  (¬ 2 * a + 3 * b = 5 * a * b) :=
by
  sorry

end correct_option_is_B_l1081_108152


namespace problem_l1081_108143

theorem problem (a k : ℕ) (h_a_pos : 0 < a) (h_a_k_pos : 0 < k) (h_div : (a^2 + k) ∣ ((a - 1) * a * (a + 1))) : k ≥ a :=
sorry

end problem_l1081_108143


namespace inscribed_square_ratio_l1081_108140

theorem inscribed_square_ratio
  (a b c : ℝ) (ha : a = 5) (hb : b = 12) (hc : c = 13) (h₁ : a^2 + b^2 = c^2)
  (x y : ℝ) (hx : x = 60 / 17) (hy : y = 144 / 17) :
  (x / y) = 5 / 12 := sorry

end inscribed_square_ratio_l1081_108140


namespace quadratic_to_square_form_l1081_108120

theorem quadratic_to_square_form (x m n : ℝ) (h : x^2 + 6 * x - 1 = 0) 
  (hm : m = 3) (hn : n = 10) : m - n = -7 :=
by 
  -- Proof steps (skipped, as per instructions)
  sorry

end quadratic_to_square_form_l1081_108120


namespace roger_earned_54_dollars_l1081_108155

-- Definitions based on problem conditions
def lawns_had : ℕ := 14
def lawns_forgot : ℕ := 8
def earn_per_lawn : ℕ := 9

-- The number of lawns actually mowed
def lawns_mowed : ℕ := lawns_had - lawns_forgot

-- The amount of money earned
def money_earned : ℕ := lawns_mowed * earn_per_lawn

-- Proof statement: Roger actually earned 54 dollars
theorem roger_earned_54_dollars : money_earned = 54 := sorry

end roger_earned_54_dollars_l1081_108155


namespace perpendicular_condition_l1081_108132

def vector_a : ℝ × ℝ := (4, 3)
def vector_b : ℝ × ℝ := (-1, 2)

def add_vector_scaled (a b : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  (a.1 + k * b.1, a.2 + k * b.2)

def sub_vector (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perpendicular_condition (k : ℝ) :
  dot_product (add_vector_scaled vector_a vector_b k) (sub_vector vector_a vector_b) = 0 ↔ k = 23 / 3 :=
by
  sorry

end perpendicular_condition_l1081_108132


namespace min_value_x_plus_y_l1081_108111

open Real

noncomputable def xy_plus_x_minus_y_minus_10_eq_zero (x y: ℝ) := x * y + x - y - 10 = 0

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : xy_plus_x_minus_y_minus_10_eq_zero x y) : 
  x + y ≥ 6 :=
by
  sorry

end min_value_x_plus_y_l1081_108111


namespace distribute_5_cousins_in_4_rooms_l1081_108168

theorem distribute_5_cousins_in_4_rooms : 
  let rooms := 4
  let cousins := 5
  ∃ ways : ℕ, ways = 67 ∧ rooms = 4 ∧ cousins = 5 := sorry

end distribute_5_cousins_in_4_rooms_l1081_108168


namespace new_average_score_l1081_108150

theorem new_average_score (n : ℕ) (initial_avg : ℕ) (grace_marks : ℕ) (h1 : n = 35) (h2 : initial_avg = 37) (h3 : grace_marks = 3) : initial_avg + grace_marks = 40 := by
  sorry

end new_average_score_l1081_108150


namespace half_angle_in_quadrant_l1081_108165

theorem half_angle_in_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 / 2) * Real.pi) :
  (π / 2 < α / 2 ∧ α / 2 < π) ∨ (3 * π / 2 < α / 2 ∧ α / 2 < 2 * π) :=
sorry

end half_angle_in_quadrant_l1081_108165


namespace circle_equation_l1081_108149

variable (x y : ℝ)

def center : ℝ × ℝ := (4, -6)
def radius : ℝ := 3

theorem circle_equation : (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ (x - 4)^2 + (y + 6)^2 = 9 :=
by
  sorry

end circle_equation_l1081_108149


namespace multiply_1546_by_100_l1081_108126

theorem multiply_1546_by_100 : 15.46 * 100 = 1546 :=
by
  sorry

end multiply_1546_by_100_l1081_108126


namespace dan_licks_l1081_108112

/-- 
Given that Michael takes 63 licks, Sam takes 70 licks, David takes 70 licks, 
Lance takes 39 licks, and the average number of licks for all five people is 60, 
prove that Dan takes 58 licks to get to the center of a lollipop.
-/
theorem dan_licks (D : ℕ) 
  (M : ℕ := 63) 
  (S : ℕ := 70) 
  (Da : ℕ := 70) 
  (L : ℕ := 39)
  (avg : ℕ := 60) :
  ((M + S + Da + L + D) / 5 = avg) → D = 58 :=
by sorry

end dan_licks_l1081_108112


namespace compare_two_sqrt_three_l1081_108123

theorem compare_two_sqrt_three : 2 > Real.sqrt 3 :=
by {
  sorry
}

end compare_two_sqrt_three_l1081_108123


namespace target_heart_rate_of_30_year_old_l1081_108184

variable (age : ℕ) (T M : ℕ)

def maximum_heart_rate (age : ℕ) : ℕ :=
  210 - age

def target_heart_rate (M : ℕ) : ℕ :=
  (75 * M) / 100

theorem target_heart_rate_of_30_year_old :
  maximum_heart_rate 30 = 180 →
  target_heart_rate (maximum_heart_rate 30) = 135 :=
by
  intros h1
  sorry

end target_heart_rate_of_30_year_old_l1081_108184


namespace S8_value_l1081_108136

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 0 + n * (a 1 - a 0)

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

def condition_a3_a6 (a : ℕ → ℝ) : Prop :=
  a 3 = 9 - a 6

theorem S8_value (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_formula : sum_of_first_n_terms S a)
  (h_condition : condition_a3_a6 a) :
  S 8 = 72 :=
by
  sorry

end S8_value_l1081_108136


namespace clock_first_ring_at_midnight_l1081_108130

theorem clock_first_ring_at_midnight (rings_every_n_hours : ℕ) (rings_per_day : ℕ) (hours_in_day : ℕ) :
  rings_every_n_hours = 3 ∧ rings_per_day = 8 ∧ hours_in_day = 24 →
  ∃ first_ring_time : Nat, first_ring_time = 0 :=
by
  sorry

end clock_first_ring_at_midnight_l1081_108130


namespace quadrant_of_angle_l1081_108178

variable (α : ℝ)

theorem quadrant_of_angle (h₁ : Real.sin α < 0) (h₂ : Real.tan α > 0) : 
  3 * (π / 2) < α ∧ α < 2 * π ∨ π < α ∧ α < 3 * (π / 2) :=
by
  sorry

end quadrant_of_angle_l1081_108178


namespace equation_equiv_product_zero_l1081_108188

theorem equation_equiv_product_zero (a b x y : ℝ) :
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1) →
  ∃ (m n p : ℤ), (a^m * x - a^n) * (a^p * y - a^3) = a^5 * b^5 ∧ m * n * p = 0 :=
by
  intros h
  sorry

end equation_equiv_product_zero_l1081_108188


namespace x_finishes_remaining_work_in_14_days_l1081_108171

-- Define the work rates of X and Y
def work_rate_X : ℚ := 1 / 21
def work_rate_Y : ℚ := 1 / 15

-- Define the amount of work Y completed in 5 days
def work_done_by_Y_in_5_days : ℚ := 5 * work_rate_Y

-- Define the remaining work after Y left
def remaining_work : ℚ := 1 - work_done_by_Y_in_5_days

-- Define the number of days needed for X to finish the remaining work
def x_days_remaining : ℚ := remaining_work / work_rate_X

-- Statement to prove
theorem x_finishes_remaining_work_in_14_days : x_days_remaining = 14 := by
  sorry

end x_finishes_remaining_work_in_14_days_l1081_108171


namespace epsilon_max_success_ratio_l1081_108151

theorem epsilon_max_success_ratio :
  ∃ (x y z w u v: ℕ), 
  (y ≠ 350) ∧
  0 < x ∧ 0 < z ∧ 0 < u ∧ 
  x < y ∧ z < w ∧ u < v ∧
  x + z + u < y + w + v ∧
  y + w + v = 800 ∧
  (x / y : ℚ) < (210 / 350 : ℚ) ∧ 
  (z / w : ℚ) < (delta_day_2_ratio) ∧ 
  (u / v : ℚ) < (delta_day_3_ratio) ∧ 
  (x + z + u) / 800 = (789 / 800 : ℚ) := 
by
  sorry

end epsilon_max_success_ratio_l1081_108151


namespace find_f3_value_l1081_108169

noncomputable def f (x : ℚ) : ℚ := (x^2 + 2*x + 1) / (4*x - 5)

theorem find_f3_value : f 3 = 16 / 7 :=
by sorry

end find_f3_value_l1081_108169


namespace find_s_l1081_108142

def f (x s : ℝ) := 3 * x^5 + 2 * x^4 - x^3 + 4 * x^2 - 5 * x + s

theorem find_s (s : ℝ) (h : f 3 s = 0) : s = -885 :=
  by sorry

end find_s_l1081_108142


namespace temperature_on_Saturday_l1081_108134

theorem temperature_on_Saturday 
  (avg_temp : ℕ)
  (sun_temp : ℕ) 
  (mon_temp : ℕ) 
  (tue_temp : ℕ) 
  (wed_temp : ℕ) 
  (thu_temp : ℕ) 
  (fri_temp : ℕ)
  (saturday_temp : ℕ)
  (h_avg : avg_temp = 53)
  (h_sun : sun_temp = 40)
  (h_mon : mon_temp = 50) 
  (h_tue : tue_temp = 65) 
  (h_wed : wed_temp = 36) 
  (h_thu : thu_temp = 82) 
  (h_fri : fri_temp = 72) 
  (h_week : 7 * avg_temp = sun_temp + mon_temp + tue_temp + wed_temp + thu_temp + fri_temp + saturday_temp) :
  saturday_temp = 26 := 
by
  sorry

end temperature_on_Saturday_l1081_108134


namespace compute_expression_l1081_108180

def sum_of_squares := 7^2 + 5^2
def square_of_sum := (7 + 5)^2
def sum_of_both := sum_of_squares + square_of_sum
def final_result := 2 * sum_of_both

theorem compute_expression : final_result = 436 := by
  sorry

end compute_expression_l1081_108180


namespace product_increase_by_13_exists_l1081_108103

theorem product_increase_by_13_exists :
  ∃ a1 a2 a3 a4 a5 a6 a7 : ℕ,
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * (a1 * a2 * a3 * a4 * a5 * a6 * a7)) :=
by
  sorry

end product_increase_by_13_exists_l1081_108103


namespace sum_q_evals_l1081_108175

noncomputable def q : ℕ → ℤ := sorry -- definition of q will be derived from conditions

theorem sum_q_evals :
  (q 0) + (q 1) + (q 2) + (q 3) + (q 4) + (q 5) + (q 6) + (q 7) + (q 8) + (q 9) +
  (q 10) + (q 11) + (q 12) + (q 13) + (q 14) + (q 15) + (q 16) + (q 17) + (q 18) = 456 :=
by
  -- Given conditions
  have h1 : q 1 = 3 := sorry
  have h6 : q 6 = 23 := sorry
  have h12 : q 12 = 17 := sorry
  have h17 : q 17 = 31 := sorry
  -- Proof outline (solved steps omitted for clarity)
  sorry

end sum_q_evals_l1081_108175


namespace find_q_l1081_108181

noncomputable def has_two_distinct_negative_roots (q : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ 
  (x₁ ^ 4 + q * x₁ ^ 3 + 2 * x₁ ^ 2 + q * x₁ + 4 = 0) ∧ 
  (x₂ ^ 4 + q * x₂ ^ 3 + 2 * x₂ ^ 2 + q * x₂ + 4 = 0)

theorem find_q (q : ℝ) : 
  has_two_distinct_negative_roots q ↔ q ≤ 3 / Real.sqrt 2 := sorry

end find_q_l1081_108181


namespace min_double_rooms_needed_min_triple_rooms_needed_with_discount_l1081_108135

-- Define the conditions 
def double_room_price : ℕ := 200
def triple_room_price : ℕ := 250
def total_students : ℕ := 50
def male_students : ℕ := 27
def female_students : ℕ := 23
def discount : ℚ := 0.2
def max_double_rooms : ℕ := 15

-- Define the property for part (1)
theorem min_double_rooms_needed (d : ℕ) (t : ℕ) : 
  2 * d + 3 * t = total_students ∧
  2 * (d - 1) + 3 * t ≠ total_students :=
sorry

-- Define the property for part (2)
theorem min_triple_rooms_needed_with_discount (d : ℕ) (t : ℕ) : 
  d + t = total_students ∧
  d ≤ max_double_rooms ∧
  2 * d + 3 * t = total_students ∧
  (1* (d - 1) + 3 * t ≠ total_students) :=
sorry

end min_double_rooms_needed_min_triple_rooms_needed_with_discount_l1081_108135
