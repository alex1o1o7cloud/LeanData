import Mathlib

namespace tan_half_angle_sum_identity_l841_84138

theorem tan_half_angle_sum_identity
  (α β γ : ℝ)
  (h : Real.sin α + Real.sin γ = 2 * Real.sin β) :
  Real.tan ((α + β) / 2) + Real.tan ((β + γ) / 2) = 2 * Real.tan ((γ + α) / 2) :=
sorry

end tan_half_angle_sum_identity_l841_84138


namespace celine_change_l841_84169

theorem celine_change :
  let laptop_price := 600
  let smartphone_price := 400
  let tablet_price := 250
  let headphone_price := 100
  let laptops_purchased := 2
  let smartphones_purchased := 4
  let tablets_purchased := 3
  let headphones_purchased := 5
  let discount_rate := 0.10
  let sales_tax_rate := 0.05
  let initial_amount := 5000
  let laptop_total := laptops_purchased * laptop_price
  let smartphone_total := smartphones_purchased * smartphone_price
  let tablet_total := tablets_purchased * tablet_price
  let headphone_total := headphones_purchased * headphone_price
  let discount := discount_rate * (laptop_total + tablet_total)
  let total_before_discount := laptop_total + smartphone_total + tablet_total + headphone_total
  let total_after_discount := total_before_discount - discount
  let sales_tax := sales_tax_rate * total_after_discount
  let final_price := total_after_discount + sales_tax
  let change := initial_amount - final_price
  change = 952.25 :=
  sorry

end celine_change_l841_84169


namespace compute_fraction_at_six_l841_84131

theorem compute_fraction_at_six (x : ℕ) (h : x = 6) : (x^6 - 16 * x^3 + 64) / (x^3 - 8) = 208 := by
  sorry

end compute_fraction_at_six_l841_84131


namespace airplane_shot_down_l841_84177

def P_A : ℝ := 0.4
def P_B : ℝ := 0.5
def P_C : ℝ := 0.8

def P_one_hit : ℝ := 0.4
def P_two_hit : ℝ := 0.7
def P_three_hit : ℝ := 1

def P_one : ℝ := (P_A * (1 - P_B) * (1 - P_C)) + ((1 - P_A) * P_B * (1 - P_C)) + ((1 - P_A) * (1 - P_B) * P_C)
def P_two : ℝ := (P_A * P_B * (1 - P_C)) + (P_A * (1 - P_B) * P_C) + ((1 - P_A) * P_B * P_C)
def P_three : ℝ := P_A * P_B * P_C

def total_probability := (P_one * P_one_hit) + (P_two * P_two_hit) + (P_three * P_three_hit)

theorem airplane_shot_down : total_probability = 0.604 := by
  sorry

end airplane_shot_down_l841_84177


namespace triangle_area_is_correct_l841_84127

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_is_correct : 
  area_of_triangle (1, 3) (5, -2) (8, 6) = 23.5 := 
by
  sorry

end triangle_area_is_correct_l841_84127


namespace ceil_sqrt_196_eq_14_l841_84113

theorem ceil_sqrt_196_eq_14 : ⌈Real.sqrt 196⌉ = 14 := 
by 
  sorry

end ceil_sqrt_196_eq_14_l841_84113


namespace min_value_sum_reciprocal_l841_84163

open Real

theorem min_value_sum_reciprocal (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
    (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
    1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 3 / 4 :=
by
  sorry

end min_value_sum_reciprocal_l841_84163


namespace a10_plus_b10_l841_84176

noncomputable def a : ℝ := sorry -- a will be a real number satisfying the conditions
noncomputable def b : ℝ := sorry -- b will be a real number satisfying the conditions

axiom ab_condition1 : a + b = 1
axiom ab_condition2 : a^2 + b^2 = 3
axiom ab_condition3 : a^3 + b^3 = 4
axiom ab_condition4 : a^4 + b^4 = 7
axiom ab_condition5 : a^5 + b^5 = 11

theorem a10_plus_b10 : a^10 + b^10 = 123 :=
by 
  sorry

end a10_plus_b10_l841_84176


namespace net_progress_l841_84114

-- Define the conditions as properties
def lost_yards : ℕ := 5
def gained_yards : ℕ := 10

-- Prove that the team's net progress is 5 yards
theorem net_progress : (gained_yards - lost_yards) = 5 :=
by
  sorry

end net_progress_l841_84114


namespace geun_bae_fourth_day_jumps_l841_84145

-- Define a function for number of jump ropes Geun-bae does on each day
def jump_ropes (n : ℕ) : ℕ :=
  match n with
  | 0     => 15
  | n + 1 => 2 * jump_ropes n

-- Theorem stating the number of jump ropes Geun-bae does on the fourth day
theorem geun_bae_fourth_day_jumps : jump_ropes 3 = 120 := 
by {
  sorry
}

end geun_bae_fourth_day_jumps_l841_84145


namespace shuai_shuai_total_words_l841_84154

-- Conditions
def words (a : ℕ) (n : ℕ) : ℕ := a + n

-- Total words memorized in 7 days
def total_memorized (a : ℕ) : ℕ := 
  (words a 0) + (words a 1) + (words a 2) + (words a 3) + (words a 4) + (words a 5) + (words a 6)

-- Condition: Sum of words memorized in the first 4 days equals sum of words in the last 3 days
def condition (a : ℕ) : Prop := 
  (words a 0) + (words a 1) + (words a 2) + (words a 3) = (words a 4) + (words a 5) + (words a 6)

-- Theorem: If condition is satisfied, then the total number of words memorized is 84.
theorem shuai_shuai_total_words : 
  ∀ a : ℕ, condition a → total_memorized a = 84 :=
by
  intro a h
  sorry

end shuai_shuai_total_words_l841_84154


namespace terminal_side_quadrant_l841_84125

theorem terminal_side_quadrant (α : ℝ) (h : α = 2) : 
  90 < α * (180 / Real.pi) ∧ α * (180 / Real.pi) < 180 := 
by
  sorry

end terminal_side_quadrant_l841_84125


namespace fraction_arithmetic_l841_84160

theorem fraction_arithmetic : ( (4 / 5 - 1 / 10) / (2 / 5) ) = 7 / 4 :=
  sorry

end fraction_arithmetic_l841_84160


namespace ratio_of_red_to_blue_beads_l841_84167

theorem ratio_of_red_to_blue_beads (red_beads blue_beads : ℕ) (h1 : red_beads = 30) (h2 : blue_beads = 20) :
    (red_beads / Nat.gcd red_beads blue_beads) = 3 ∧ (blue_beads / Nat.gcd red_beads blue_beads) = 2 := 
by 
    -- Proof will go here
    sorry

end ratio_of_red_to_blue_beads_l841_84167


namespace Nina_second_distance_l841_84158

theorem Nina_second_distance 
  (total_distance : ℝ) 
  (first_run : ℝ) 
  (second_same_run : ℝ)
  (run_twice : first_run = 0.08 ∧ second_same_run = 0.08)
  (total : total_distance = 0.83)
  : (total_distance - (first_run + second_same_run)) = 0.67 := by
  sorry

end Nina_second_distance_l841_84158


namespace dugu_team_prob_l841_84109

def game_prob (prob_win_first : ℝ) (prob_increase : ℝ) (prob_decrease : ℝ) : ℝ :=
  let p1 := prob_win_first
  let p2 := prob_win_first + prob_increase
  let p3 := prob_win_first + 2 * prob_increase
  let p4 := prob_win_first + 3 * prob_increase
  let p5 := prob_win_first + 4 * prob_increase
  let win_in_3 := p1 * p2 * p3
  let lose_first := (1 - prob_win_first)
  let win_then := prob_win_first
  let win_in_4a := lose_first * (prob_win_first - prob_decrease) * 
    prob_win_first * p2 * p3
  let win_in_4b := win_then * (1 - (prob_win_first + prob_increase)) *
    p2 * p3
  let win_in_4c := win_then * p2 * (1 - prob_win_first + prob_increase - 
    prob_decrease) * p4

  win_in_3 + win_in_4a + win_in_4b + win_in_4c

theorem dugu_team_prob : 
  game_prob 0.4 0.1 0.1 = 0.236 :=
by
  sorry

end dugu_team_prob_l841_84109


namespace sum_interior_ninth_row_l841_84100

-- Define Pascal's Triangle and the specific conditions
def pascal_sum (n : ℕ) : ℕ := 2^(n - 1)

def sum_interior_numbers (n : ℕ) : ℕ := pascal_sum n - 2

theorem sum_interior_ninth_row :
  sum_interior_numbers 9 = 254 := 
by {
  sorry
}

end sum_interior_ninth_row_l841_84100


namespace calculate_expression_l841_84175

variables {a b c : ℤ}
variable (h1 : 5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c) -- a, b, c are multiples of 5
variable (h2 : a < b ∧ b < c) -- a < b < c
variable (h3 : c = a + 10) -- c = a + 10

theorem calculate_expression :
  (a - b) * (a - c) / (b - c) = -10 :=
by
  sorry

end calculate_expression_l841_84175


namespace negation_example_l841_84152

theorem negation_example :
  (¬ (∀ x : ℝ, abs (x - 2) + abs (x - 4) > 3)) ↔ (∃ x : ℝ, abs (x - 2) + abs (x - 4) ≤ 3) :=
by
  sorry

end negation_example_l841_84152


namespace least_product_of_distinct_primes_greater_than_50_l841_84189

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  p ≠ q ∧ is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 : 
  ∃ p q, distinct_primes_greater_than_50 p q ∧ p * q = 3127 := 
sorry

end least_product_of_distinct_primes_greater_than_50_l841_84189


namespace base8_to_base10_sum_l841_84198

theorem base8_to_base10_sum (a b : ℕ) (h₁ : a = 1 * 8^3 + 4 * 8^2 + 5 * 8^1 + 3 * 8^0)
                            (h₂ : b = 5 * 8^2 + 6 * 8^1 + 7 * 8^0) :
                            ((a + b) = 2 * 8^3 + 1 * 8^2 + 4 * 8^1 + 4 * 8^0) →
                            (2 * 8^3 + 1 * 8^2 + 4 * 8^1 + 4 * 8^0 = 1124) :=
by {
  sorry
}

end base8_to_base10_sum_l841_84198


namespace number_of_distinct_stackings_l841_84164

-- Defining the conditions
def cubes : ℕ := 8
def edge_length : ℕ := 1
def valid_stackings (n : ℕ) : Prop := 
  n = 8 -- Stating that we are working with 8 cubes

-- The theorem stating the problem and expected solution
theorem number_of_distinct_stackings : 
  cubes = 8 ∧ edge_length = 1 ∧ valid_stackings cubes → ∃ (count : ℕ), count = 10 :=
by 
  sorry

end number_of_distinct_stackings_l841_84164


namespace range_of_a_l841_84180

-- Given conditions
def p (x : ℝ) : Prop := abs (4 - x) ≤ 6
def q (x : ℝ) (a : ℝ) : Prop := (x - 1)^2 - a^2 ≥ 0

-- The statement to prove
theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : ∀ x, ¬p x → q x a) : 
  0 < a ∧ a ≤ 3 :=
by
  sorry -- Proof placeholder

end range_of_a_l841_84180


namespace beijing_olympics_problem_l841_84159

theorem beijing_olympics_problem
  (M T J D: Type)
  (sports: M → Type)
  (swimming gymnastics athletics volleyball: M → Prop)
  (athlete_sits: M → M → Prop)
  (Maria Tania Juan David: M)
  (woman: M → Prop)
  (left right front next_to: M → M → Prop)
  (h1: ∀ x, swimming x → left x Maria)
  (h2: ∀ x, gymnastics x → front x Juan)
  (h3: next_to Tania David)
  (h4: ∀ x, volleyball x → ∃ y, woman y ∧ next_to y x) :
  athletics David := 
sorry

end beijing_olympics_problem_l841_84159


namespace rectangle_same_color_exists_l841_84166

theorem rectangle_same_color_exists (color : ℝ × ℝ → Prop) (red blue : Prop) (h : ∀ p : ℝ × ℝ, color p = red ∨ color p = blue) :
  ∃ (a b c d : ℝ × ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
  (color a = color b ∧ color b = color c ∧ color c = color d) :=
sorry

end rectangle_same_color_exists_l841_84166


namespace shortest_distance_phenomena_explained_l841_84116

def condition1 : Prop :=
  ∀ (a b : ℕ), (exists nail1 : ℕ, exists nail2 : ℕ, nail1 ≠ nail2) → (exists wall : ℕ, wall = a + b)

def condition2 : Prop :=
  ∀ (tree1 tree2 tree3 : ℕ), tree1 ≠ tree2 → tree2 ≠ tree3 → (tree1 + tree2 + tree3) / 3 = tree2

def condition3 : Prop :=
  ∀ (A B : ℕ), ∃ (C : ℕ), C = (B - A) → (A = B - (B - A))

def condition4 : Prop :=
  ∀ (dist : ℕ), dist = 0 → exists shortest : ℕ, shortest < dist

-- The following theorem needs to be proven to match our mathematical problem
theorem shortest_distance_phenomena_explained :
  condition3 ∧ condition4 :=
by
  sorry

end shortest_distance_phenomena_explained_l841_84116


namespace resulting_expression_l841_84107

def x : ℕ := 1000
def y : ℕ := 10

theorem resulting_expression : 
  (x + 2 * y) + x + 3 * y + x + 4 * y + x + y = 4 * x + 10 * y :=
by
  sorry

end resulting_expression_l841_84107


namespace total_distance_traveled_l841_84124

theorem total_distance_traveled :
  let car_speed1 := 90
  let car_time1 := 2
  let car_speed2 := 60
  let car_time2 := 1
  let train_speed := 100
  let train_time := 2.5
  let distance_car1 := car_speed1 * car_time1
  let distance_car2 := car_speed2 * car_time2
  let distance_train := train_speed * train_time
  distance_car1 + distance_car2 + distance_train = 490 := by
  sorry

end total_distance_traveled_l841_84124


namespace usual_time_28_l841_84130

theorem usual_time_28 (R T : ℝ) (h1 : ∀ (d : ℝ), d = R * T)
  (h2 : ∀ (d : ℝ), d = (6/7) * R * (T - 4)) : T = 28 :=
by
  -- Variables:
  -- R : Usual rate of the boy
  -- T : Usual time to reach the school
  -- h1 : Expressing distance in terms of usual rate and time
  -- h2 : Expressing distance in terms of reduced rate and time minus 4
  sorry

end usual_time_28_l841_84130


namespace arithmetic_progression_first_three_terms_l841_84153

theorem arithmetic_progression_first_three_terms 
  (S_n : ℤ) (d a_1 a_2 a_3 a_5 : ℤ)
  (h1 : S_n = 112) 
  (h2 : (a_1 + d) * d = 30)
  (h3 : (a_1 + 2 * d) + (a_1 + 4 * d) = 32) 
  (h4 : ∀ (n : ℕ), S_n = (n * (2 * a_1 + (n - 1) * d)) / 2) : 
  ((a_1 = 7 ∧ a_2 = 10 ∧ a_3 = 13) ∨ (a_1 = 1 ∧ a_2 = 6 ∧ a_3 = 11)) :=
sorry

end arithmetic_progression_first_three_terms_l841_84153


namespace find_principal_l841_84108

theorem find_principal (CI SI : ℝ) (hCI : CI = 11730) (hSI : SI = 10200)
  (P R : ℝ)
  (hSI_form : SI = P * R * 2 / 100)
  (hCI_form : CI = P * (1 + R / 100)^2 - P) :
  P = 34000 := by
  sorry

end find_principal_l841_84108


namespace wooden_block_length_l841_84185

-- Define the problem conditions
def meters_to_centimeters (m : ℕ) : ℕ := m * 100
def additional_length_cm (length_cm : ℕ) (additional_cm : ℕ) : ℕ := length_cm + additional_cm

-- Formalization of the problem
theorem wooden_block_length :
  let length_in_meters := 31
  let additional_cm := 30
  additional_length_cm (meters_to_centimeters length_in_meters) additional_cm = 3130 :=
by
  sorry

end wooden_block_length_l841_84185


namespace add_pure_chocolate_to_achieve_percentage_l841_84119

/--
Given:
    Initial amount of chocolate topping: 620 ounces.
    Initial chocolate percentage: 10%.
    Desired total weight of the final mixture: 1000 ounces.
    Desired chocolate percentage in the final mixture: 70%.
Prove:
    The amount of pure chocolate to be added to achieve the desired mixture is 638 ounces.
-/
theorem add_pure_chocolate_to_achieve_percentage :
  ∃ x : ℝ,
    0.10 * 620 + x = 0.70 * 1000 ∧
    x = 638 :=
by
  sorry

end add_pure_chocolate_to_achieve_percentage_l841_84119


namespace beavers_swimming_correct_l841_84111

variable (initial_beavers remaining_beavers beavers_swimming : ℕ)

def beavers_problem : Prop :=
  initial_beavers = 2 ∧
  remaining_beavers = 1 ∧
  beavers_swimming = initial_beavers - remaining_beavers

theorem beavers_swimming_correct :
  beavers_problem initial_beavers remaining_beavers beavers_swimming → beavers_swimming = 1 :=
by
  sorry

end beavers_swimming_correct_l841_84111


namespace reduced_price_l841_84174

-- Definitions based on given conditions
def original_price (P : ℝ) : Prop := P > 0

def condition1 (P X : ℝ) : Prop := P * X = 700

def condition2 (P X : ℝ) : Prop := 0.7 * P * (X + 3) = 700

-- Main theorem to prove the reduced price per kg is 70
theorem reduced_price (P X : ℝ) (h1 : original_price P) (h2 : condition1 P X) (h3 : condition2 P X) : 
  0.7 * P = 70 := sorry

end reduced_price_l841_84174


namespace union_of_sets_l841_84133

open Set

theorem union_of_sets :
  ∀ (P Q : Set ℕ), P = {1, 2} → Q = {2, 3} → P ∪ Q = {1, 2, 3} :=
by
  intros P Q hP hQ
  rw [hP, hQ]
  exact sorry

end union_of_sets_l841_84133


namespace car_highway_miles_per_tankful_l841_84149

-- Condition definitions
def city_miles_per_tankful : ℕ := 336
def miles_per_gallon_city : ℕ := 24
def city_to_highway_diff : ℕ := 9

-- Calculation from conditions
def miles_per_gallon_highway : ℕ := miles_per_gallon_city + city_to_highway_diff
def tank_size : ℤ := city_miles_per_tankful / miles_per_gallon_city

-- Desired result
def highway_miles_per_tankful : ℤ := miles_per_gallon_highway * tank_size

-- Proof statement
theorem car_highway_miles_per_tankful :
  highway_miles_per_tankful = 462 := by
  unfold highway_miles_per_tankful
  unfold miles_per_gallon_highway
  unfold tank_size
  -- Sorry here to skip the detailed proof steps
  sorry

end car_highway_miles_per_tankful_l841_84149


namespace compound_proposition_l841_84184

theorem compound_proposition (Sn P Q : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → Sn n = 2 * n^2 + 3 * n + 1) →
  (∀ n : ℕ, n > 0 → Sn n = 2 * P n + 1) →
  (¬(∀ n, n > 0 → ∃ d, (P (n + 1) - P n) = d)) ∧ (∀ n, n > 0 → P n = Q (n - 1)) :=
by
  sorry

end compound_proposition_l841_84184


namespace at_least_one_wins_l841_84151

def probability_A := 1 / 2
def probability_B := 1 / 4

def probability_at_least_one (pA pB : ℚ) : ℚ := 
  1 - ((1 - pA) * (1 - pB))

theorem at_least_one_wins :
  probability_at_least_one probability_A probability_B = 5 / 8 := 
by
  sorry

end at_least_one_wins_l841_84151


namespace find_q_value_l841_84172

theorem find_q_value 
  (p q r : ℕ) 
  (hp : 0 < p) 
  (hq : 0 < q) 
  (hr : 0 < r) 
  (h : p + 1 / (q + 1 / r : ℚ) = 25 / 19) : 
  q = 3 :=
by 
  sorry

end find_q_value_l841_84172


namespace smaller_cube_volume_is_correct_l841_84135

noncomputable def inscribed_smaller_cube_volume 
  (edge_length_outer_cube : ℝ)
  (h : edge_length_outer_cube = 12) : ℝ := 
  let diameter_sphere := edge_length_outer_cube
  let radius_sphere := diameter_sphere / 2
  let space_diagonal_smaller_cube := diameter_sphere
  let side_length_smaller_cube := space_diagonal_smaller_cube / (Real.sqrt 3)
  let volume_smaller_cube := side_length_smaller_cube ^ 3
  volume_smaller_cube

theorem smaller_cube_volume_is_correct 
  (h : 12 = 12) : inscribed_smaller_cube_volume 12 h = 192 * Real.sqrt 3 :=
by
  sorry

end smaller_cube_volume_is_correct_l841_84135


namespace male_teacher_classes_per_month_l841_84139

theorem male_teacher_classes_per_month (x y a : ℕ) :
  (15 * x = 6 * (x + y)) ∧ (a * y = 6 * (x + y)) → a = 10 :=
by
  sorry

end male_teacher_classes_per_month_l841_84139


namespace homework_time_decrease_l841_84101

variable (x : ℝ)
variable (initial_time final_time : ℝ)
variable (adjustments : ℕ)

def rate_of_decrease (initial_time final_time : ℝ) (adjustments : ℕ) (x : ℝ) := 
  initial_time * (1 - x)^adjustments = final_time

theorem homework_time_decrease 
  (h_initial : initial_time = 100) 
  (h_final : final_time = 70)
  (h_adjustments : adjustments = 2)
  (h_decrease : rate_of_decrease initial_time final_time adjustments x) : 
  100 * (1 - x)^2 = 70 :=
by
  sorry

end homework_time_decrease_l841_84101


namespace necessary_but_not_sufficient_l841_84150

variable (p q : Prop)

theorem necessary_but_not_sufficient (h : ¬p → q) (h1 : ¬ (q → ¬p)) : ¬q → p := 
by
  sorry

end necessary_but_not_sufficient_l841_84150


namespace people_in_each_playgroup_l841_84197

theorem people_in_each_playgroup (girls boys parents playgroups : ℕ) (hg : girls = 14) (hb : boys = 11) (hp : parents = 50) (hpg : playgroups = 3) :
  (girls + boys + parents) / playgroups = 25 := by
  sorry

end people_in_each_playgroup_l841_84197


namespace solve_congruence_l841_84161

theorem solve_congruence (n : ℤ) : 15 * n ≡ 9 [ZMOD 47] → n ≡ 18 [ZMOD 47] :=
by
  sorry

end solve_congruence_l841_84161


namespace sum_first_32_terms_bn_l841_84171

noncomputable def a_n (n : ℕ) : ℝ := 3 * n + 1

noncomputable def b_n (n : ℕ) : ℝ :=
  1 / ((a_n n) * Real.sqrt (a_n (n + 1)) + (a_n (n + 1)) * Real.sqrt (a_n n))

noncomputable def sum_bn (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) b_n

theorem sum_first_32_terms_bn : sum_bn 32 = 2 / 15 := 
sorry

end sum_first_32_terms_bn_l841_84171


namespace least_possible_value_of_x_minus_y_plus_z_l841_84192

theorem least_possible_value_of_x_minus_y_plus_z : 
  ∃ (x y z : ℕ), 3 * x = 4 * y ∧ 4 * y = 7 * z ∧ x - y + z = 19 :=
by
  sorry

end least_possible_value_of_x_minus_y_plus_z_l841_84192


namespace part1_part2_l841_84199

namespace MathProofProblem

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem part1 (x : ℝ) : f 2 * x ≤ f (x + 1) ↔ 0 ≤ x ∧ x ≤ 1 := 
by
  sorry

theorem part2 (a b : ℝ) (h₀ : a + b = 2) : f (a ^ 2) + f (b ^ 2) = 2 :=
by
  sorry

end MathProofProblem

end part1_part2_l841_84199


namespace sara_total_quarters_l841_84183

def initial_quarters : ℝ := 783.0
def given_quarters : ℝ := 271.0

theorem sara_total_quarters : initial_quarters + given_quarters = 1054.0 := 
by
  sorry

end sara_total_quarters_l841_84183


namespace ratio_of_length_to_breadth_l841_84142

theorem ratio_of_length_to_breadth (b l k : ℕ) (h1 : b = 15) (h2 : l = k * b) (h3 : l * b = 675) : l / b = 3 :=
by
  sorry

end ratio_of_length_to_breadth_l841_84142


namespace find_b_l841_84187

theorem find_b (a b c : ℝ) (A B C : ℝ) (h1 : a = 10) (h2 : c = 20) (h3 : B = 120) :
  b = 10 * Real.sqrt 7 :=
sorry

end find_b_l841_84187


namespace bisect_area_of_trapezoid_l841_84136

-- Define the vertices of the quadrilateral
structure Point :=
  (x : ℤ)
  (y : ℤ)

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 16, y := 0 }
def C : Point := { x := 8, y := 8 }
def D : Point := { x := 0, y := 8 }

-- Define the equation of a line
structure Line :=
  (slope : ℚ)
  (intercept : ℚ)

-- Define the condition for parallel lines
def parallel (L1 L2 : Line) : Prop :=
  L1.slope = L2.slope

-- Define the diagonal AC and the required line
def AC : Line := { slope := 1, intercept := 0 }
def bisecting_line : Line := { slope := 1, intercept := -4 }

-- The area of trapezoid
def trapezoid_area : ℚ := (8 * (16 + 8)) / 2

-- Proof that the required line is parallel to AC and bisects the area of the trapezoid
theorem bisect_area_of_trapezoid :
  parallel bisecting_line AC ∧ 
  (1 / 2) * (8 * (16 + bisecting_line.intercept)) = trapezoid_area / 2 :=
by
  sorry

end bisect_area_of_trapezoid_l841_84136


namespace problem_statement_l841_84104

theorem problem_statement (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 5) : 
  -a - m * c * d - b = -5 ∨ -a - m * c * d - b = 5 := 
  sorry

end problem_statement_l841_84104


namespace elois_made_3_loaves_on_Monday_l841_84165

theorem elois_made_3_loaves_on_Monday
    (bananas_per_loaf : ℕ)
    (twice_as_many : ℕ)
    (total_bananas : ℕ) 
    (h1 : bananas_per_loaf = 4) 
    (h2 : twice_as_many = 2) 
    (h3 : total_bananas = 36)
  : ∃ L : ℕ, (4 * L + 8 * L = 36) ∧ L = 3 :=
sorry

end elois_made_3_loaves_on_Monday_l841_84165


namespace min_students_in_class_l841_84186

-- Define the conditions
variables (b g : ℕ) -- number of boys and girls
variable (h1 : 3 * b = 4 * (2 * g)) -- Equal number of boys and girls passed the test

-- Define the desired minimum number of students
def min_students : ℕ := 17

-- The theorem which asserts that the total number of students in the class is at least 17
theorem min_students_in_class (b g : ℕ) (h1 : 3 * b = 4 * (2 * g)) : (b + g) ≥ min_students := 
sorry

end min_students_in_class_l841_84186


namespace sine_gamma_half_leq_c_over_a_plus_b_l841_84117

variable (a b c : ℝ) (γ : ℝ)

-- Consider a triangle with sides a, b, c, and angle γ opposite to side c.
-- We need to prove that sin(γ / 2) ≤ c / (a + b).
theorem sine_gamma_half_leq_c_over_a_plus_b (h_c_pos : 0 < c) 
  (h_g_angle : 0 < γ ∧ γ < 2 * π) : 
  Real.sin (γ / 2) ≤ c / (a + b) := 
  sorry

end sine_gamma_half_leq_c_over_a_plus_b_l841_84117


namespace find_smallest_n_modulo_l841_84144

theorem find_smallest_n_modulo :
  ∃ n : ℕ, n > 0 ∧ (2007 * n) % 1000 = 837 ∧ n = 691 :=
by
  sorry

end find_smallest_n_modulo_l841_84144


namespace desired_depth_proof_l841_84196

-- Definitions based on the conditions in Step a)
def initial_men : ℕ := 9
def initial_hours : ℕ := 8
def initial_depth : ℕ := 30
def extra_men : ℕ := 11
def total_men : ℕ := initial_men + extra_men
def new_hours : ℕ := 6

-- Total man-hours for initial setup
def initial_man_hours (days : ℕ) : ℕ := initial_men * initial_hours * days

-- Total man-hours for new setup to achieve desired depth
def new_man_hours (desired_depth : ℕ) (days : ℕ) : ℕ := total_men * new_hours * days

-- Proportional relationship between initial setup and desired depth
theorem desired_depth_proof (days : ℕ) (desired_depth : ℕ) :
  initial_man_hours days / initial_depth = new_man_hours desired_depth days / desired_depth → desired_depth = 18 :=
by
  sorry

end desired_depth_proof_l841_84196


namespace simplify_expression_l841_84123

theorem simplify_expression (y : ℝ) : 
  2 * y * (4 * y^2 - 3 * y + 1) - 6 * (y^2 - 3 * y + 4) = 8 * y^3 - 12 * y^2 + 20 * y - 24 := 
by
  sorry

end simplify_expression_l841_84123


namespace tangent_point_at_slope_one_l841_84179

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 - 3 * x

-- Define the derivative of the curve
def derivative (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem proof problem
theorem tangent_point_at_slope_one : ∃ x : ℝ, derivative x = 1 ∧ x = 2 :=
by
  sorry

end tangent_point_at_slope_one_l841_84179


namespace tan_pi_add_theta_l841_84146

theorem tan_pi_add_theta (θ : ℝ) (h : Real.tan (Real.pi + θ) = 2) : 
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + 2 * Real.cos θ) = 3 / 4 :=
by
  sorry

end tan_pi_add_theta_l841_84146


namespace speed_of_current_l841_84168

theorem speed_of_current (h_start: ∀ t: ℝ, t ≥ 0 → u ≥ 0) 
  (boat1_turn_2pm: ∀ t: ℝ, t >= 1 → t < 2 → boat1_turn_13_14) 
  (boat2_turn_3pm: ∀ t: ℝ, t >= 2 → t < 3 → boat2_turn_14_15) 
  (boats_meet: ∀ x: ℝ, x = 7.5) :
  v = 2.5 := 
sorry

end speed_of_current_l841_84168


namespace probability_green_then_blue_l841_84193

theorem probability_green_then_blue :
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := green_marbles / total_marbles
  let prob_second_blue := blue_marbles / (total_marbles - 1)
  prob_first_green * prob_second_blue = 4 / 15 :=
sorry

end probability_green_then_blue_l841_84193


namespace probability_all_boys_probability_one_girl_probability_at_least_one_girl_l841_84147

-- Assumptions and Definitions
def total_outcomes := Nat.choose 5 3
def all_boys_outcomes := Nat.choose 3 3
def one_girl_outcomes := Nat.choose 3 2 * Nat.choose 2 1
def at_least_one_girl_outcomes := one_girl_outcomes + Nat.choose 3 1 * Nat.choose 2 2

-- The probability calculation proofs
theorem probability_all_boys : all_boys_outcomes / total_outcomes = 1 / 10 := by 
  sorry

theorem probability_one_girl : one_girl_outcomes / total_outcomes = 6 / 10 := by 
  sorry

theorem probability_at_least_one_girl : at_least_one_girl_outcomes / total_outcomes = 9 / 10 := by 
  sorry

end probability_all_boys_probability_one_girl_probability_at_least_one_girl_l841_84147


namespace orange_juice_production_l841_84128

theorem orange_juice_production :
  let total_oranges := 8 -- in million tons
  let exported_oranges := total_oranges * 0.25
  let remaining_oranges := total_oranges - exported_oranges
  let juice_oranges_ratio := 0.60
  let juice_oranges := remaining_oranges * juice_oranges_ratio
  juice_oranges = 3.6  :=
by
  sorry

end orange_juice_production_l841_84128


namespace female_students_transfer_l841_84137

theorem female_students_transfer (x y z : ℕ) 
  (h1 : ∀ B : ℕ, B = x - 4) 
  (h2 : ∀ C : ℕ, C = x - 5)
  (h3 : ∀ B' : ℕ, B' = x - 4 + y - z)
  (h4 : ∀ C' : ℕ, C' = x + z - 7) 
  (h5 : x - y + 2 = x - 4 + y - z)
  (h6 : x - 4 + y - z = x + z - 7) 
  (h7 : 2 = 2) :
  y = 3 ∧ z = 4 := 
by 
  sorry

end female_students_transfer_l841_84137


namespace inequality_of_abc_l841_84170

variable (a b c : ℝ)

theorem inequality_of_abc (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c ≥ a * b * c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * a * b * c :=
sorry

end inequality_of_abc_l841_84170


namespace quadratic_solution_l841_84140

theorem quadratic_solution (x : ℝ) : 2 * x^2 - 3 * x + 1 = 0 → (x = 1 / 2 ∨ x = 1) :=
by sorry

end quadratic_solution_l841_84140


namespace find_last_score_l841_84132

/-- The list of scores in ascending order -/
def scores : List ℕ := [60, 65, 70, 75, 80, 85, 95]

/--
  The problem states that the average score after each entry is an integer.
  Given the scores in ascending order, determine the last score entered.
-/
theorem find_last_score (h : ∀ (n : ℕ) (hn : n < scores.length),
    (scores.take (n + 1) |>.sum : ℤ) % (n + 1) = 0) :
  scores.last' = some 80 :=
sorry

end find_last_score_l841_84132


namespace determine_f_zero_l841_84156

variable (f : ℝ → ℝ)

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 + y) + 4 * (f x) * y

theorem determine_f_zero (h1: functional_equation f)
    (h2 : f 2 = 4) : f 0 = 0 := 
sorry

end determine_f_zero_l841_84156


namespace simplify_expression_l841_84194

theorem simplify_expression (a b c d : ℝ) (h₁ : a + b + c + d = 0) (h₂ : a ≠ 0) (h₃ : b ≠ 0) (h₄ : c ≠ 0) (h₅ : d ≠ 0) :
  (1 / (b^2 + c^2 + d^2 - a^2) + 
   1 / (a^2 + c^2 + d^2 - b^2) + 
   1 / (a^2 + b^2 + d^2 - c^2) + 
   1 / (a^2 + b^2 + c^2 - d^2)) = 4 / d^2 := 
sorry

end simplify_expression_l841_84194


namespace find_y_value_l841_84115

theorem find_y_value : 
  (15^2 * 8^3) / y = 450 → y = 256 :=
by
  sorry

end find_y_value_l841_84115


namespace problem1_problem2_l841_84106

-- Define the conditions: f is an odd and decreasing function on [-1, 1]
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_decreasing : ∀ x y, x ≤ y → f y ≤ f x)

-- The domain of interest is [-1, 1]
variable (x1 x2 : ℝ)
variable (h_x1 : x1 ∈ Set.Icc (-1 : ℝ) 1)
variable (h_x2 : x2 ∈ Set.Icc (-1 : ℝ) 1)

-- Proof Problem 1
theorem problem1 : (f x1 + f x2) * (x1 + x2) ≤ 0 := by
  sorry

-- Assume condition for Problem 2
variable (a : ℝ)
variable (h_ineq : f (1 - a) + f (1 - a ^ 2) < 0)
variable (h_dom : ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → x ∈ Set.Icc (-1 : ℝ) 1)

-- Proof Problem 2
theorem problem2 : 0 < a ∧ a < 1 := by
  sorry

end problem1_problem2_l841_84106


namespace not_necessarily_periodic_l841_84126

-- Define the conditions of the problem
noncomputable def a : ℕ → ℕ := sorry
noncomputable def t : ℕ → ℕ := sorry
axiom h_t : ∀ k : ℕ, ∃ t_k : ℕ, ∀ n : ℕ, a (k + n * t_k) = a k

-- The theorem stating that the sequence is not necessarily periodic
theorem not_necessarily_periodic : ¬ ∃ T : ℕ, ∀ k : ℕ, a (k + T) = a k := sorry

end not_necessarily_periodic_l841_84126


namespace thomas_total_blocks_l841_84118

def stack1 := 7
def stack2 := stack1 + 3
def stack3 := stack2 - 6
def stack4 := stack3 + 10
def stack5 := stack2 * 2

theorem thomas_total_blocks : stack1 + stack2 + stack3 + stack4 + stack5 = 55 := by
  sorry

end thomas_total_blocks_l841_84118


namespace fraction_susan_can_eat_l841_84103

theorem fraction_susan_can_eat
  (v t n nf : ℕ)
  (h₁ : v = 6)
  (h₂ : n = 4)
  (h₃ : 1/3 * t = v)
  (h₄ : nf = v - n) :
  nf / t = 1 / 9 :=
sorry

end fraction_susan_can_eat_l841_84103


namespace focus_of_parabola_l841_84155

theorem focus_of_parabola (x y : ℝ) (h : y^2 + 4 * x = 0) : (x, y) = (-1, 0) := sorry

end focus_of_parabola_l841_84155


namespace coupon_savings_difference_l841_84141

-- Definitions based on conditions
def P (p : ℝ) := 120 + p
def savings_coupon_A (p : ℝ) := 24 + 0.20 * p
def savings_coupon_B := 35
def savings_coupon_C (p : ℝ) := 0.30 * p

-- Conditions
def condition_A_saves_at_least_B (p : ℝ) := savings_coupon_A p ≥ savings_coupon_B
def condition_A_saves_at_least_C (p : ℝ) := savings_coupon_A p ≥ savings_coupon_C p

-- Proof problem
theorem coupon_savings_difference :
  ∀ (p : ℝ), 55 ≤ p ∧ p ≤ 240 → (P 240 - P 55) = 185 :=
by
  sorry

end coupon_savings_difference_l841_84141


namespace sheep_remain_l841_84173

theorem sheep_remain : ∀ (total_sheep sister_share brother_share : ℕ),
  total_sheep = 400 →
  sister_share = total_sheep / 4 →
  brother_share = (total_sheep - sister_share) / 2 →
  (total_sheep - sister_share - brother_share) = 150 :=
by
  intros total_sheep sister_share brother_share h_total h_sister h_brother
  rw [h_total, h_sister, h_brother]
  sorry

end sheep_remain_l841_84173


namespace scientific_notation_of_10900_l841_84143

theorem scientific_notation_of_10900 : ∃ (x : ℝ) (n : ℤ), 10900 = x * 10^n ∧ x = 1.09 ∧ n = 4 := by
  use 1.09
  use 4
  sorry

end scientific_notation_of_10900_l841_84143


namespace charity_event_equation_l841_84191

variable (x : ℕ)

theorem charity_event_equation : x + 5 * (12 - x) = 48 :=
sorry

end charity_event_equation_l841_84191


namespace compare_abc_l841_84129

noncomputable def a : ℝ := Real.exp (Real.sqrt Real.pi)
noncomputable def b : ℝ := Real.sqrt Real.pi + 1
noncomputable def c : ℝ := (Real.log Real.pi) / Real.exp 1 + 2

theorem compare_abc : c < b ∧ b < a := by
  sorry

end compare_abc_l841_84129


namespace p_implies_q_l841_84120

def p (x : ℝ) := 0 < x ∧ x < 5
def q (x : ℝ) := -5 < x - 2 ∧ x - 2 < 5

theorem p_implies_q (x : ℝ) (h : p x) : q x :=
  by sorry

end p_implies_q_l841_84120


namespace roots_quadratic_l841_84188

theorem roots_quadratic (a b : ℝ) 
  (h1: a^2 + 3 * a - 2010 = 0) 
  (h2: b^2 + 3 * b - 2010 = 0)
  (h_roots: a + b = -3 ∧ a * b = -2010):
  a^2 - a - 4 * b = 2022 :=
by
  sorry

end roots_quadratic_l841_84188


namespace minimal_period_of_sum_l841_84178

theorem minimal_period_of_sum (A B : ℝ)
  (hA : ∃ p : ℕ, p = 6 ∧ (∃ (x : ℝ) (l : ℕ), A = x / (10 ^ l * (10 ^ p - 1))))
  (hB : ∃ p : ℕ, p = 12 ∧ (∃ (y : ℝ) (m : ℕ), B = y / (10 ^ m * (10 ^ p - 1)))) :
  ∃ p : ℕ, p = 12 ∧ (∃ (z : ℝ) (n : ℕ), A + B = z / (10 ^ n * (10 ^ p - 1))) :=
sorry

end minimal_period_of_sum_l841_84178


namespace smallest_m_div_18_l841_84134

noncomputable def smallest_multiple_18 : ℕ :=
  900

theorem smallest_m_div_18 : (∃ m: ℕ, (m % 18 = 0) ∧ (∀ d ∈ m.digits 10, d = 9 ∨ d = 0) ∧ ∀ k: ℕ, k % 18 = 0 → (∀ d ∈ k.digits 10, d = 9 ∨ d = 0) → m ≤ k) → 900 / 18 = 50 :=
by
  intro h
  sorry

end smallest_m_div_18_l841_84134


namespace mod_inverse_identity_l841_84102

theorem mod_inverse_identity : 
  (1 / 5 + 1 / 5^2) % 31 = 26 :=
by
  sorry

end mod_inverse_identity_l841_84102


namespace periodic_sequence_not_constant_l841_84181

theorem periodic_sequence_not_constant :
  ∃ (x : ℕ → ℤ), (∀ n : ℕ, x (n+1) = 2 * x n + 3 * x (n-1)) ∧ (∃ T > 0, ∀ n : ℕ, x (n+T) = x n) ∧ (∃ n m : ℕ, n ≠ m ∧ x n ≠ x m) :=
sorry

end periodic_sequence_not_constant_l841_84181


namespace find_constant_k_l841_84110

theorem find_constant_k (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ)
  (h₁ : ∀ n, S n = 3 * 2^n + k)
  (h₂ : ∀ n, 1 ≤ n → a n = S n - S (n - 1))
  (h₃ : ∃ q, ∀ n, 1 ≤ n → a (n + 1) = a n * q ) :
  k = -3 := 
sorry

end find_constant_k_l841_84110


namespace arithmetic_sequence_general_term_and_sum_l841_84121

theorem arithmetic_sequence_general_term_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 2 = 2) →
  (a 4 = 4) →
  (∀ n, a n = n) →
  (∀ n, b n = 2 ^ (a n)) →
  (∀ n, S n = 2 * (2 ^ n - 1)) :=
by
  intros h1 h2 h3 h4
  -- Proof part is skipped
  sorry

end arithmetic_sequence_general_term_and_sum_l841_84121


namespace jake_weight_l841_84148

theorem jake_weight (J S B : ℝ) (h1 : J - 8 = 2 * S)
                            (h2 : B = 2 * J + 6)
                            (h3 : J + S + B = 480)
                            (h4 : B = 1.25 * S) :
  J = 230 :=
by
  sorry

end jake_weight_l841_84148


namespace boys_count_at_table_l841_84162

-- Definitions from conditions
def children_count : ℕ := 13
def alternates (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- The problem to be proven in Lean:
theorem boys_count_at_table : ∃ b g : ℕ, b + g = children_count ∧ alternates b ∧ alternates g ∧ b = 7 :=
by
  sorry

end boys_count_at_table_l841_84162


namespace minValueExpr_ge_9_l841_84190

noncomputable def minValueExpr (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minValueExpr_ge_9 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  minValueExpr x y z ≥ 9 :=
by sorry

end minValueExpr_ge_9_l841_84190


namespace neither_odd_nor_even_and_min_value_at_one_l841_84105

def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem neither_odd_nor_even_and_min_value_at_one :
  (∀ x, f (-x) ≠ f x ∧ f (-x) ≠ - f x) ∧ ∃ x, x = 1 ∧ ∀ y, f y ≥ f x :=
by
  sorry

end neither_odd_nor_even_and_min_value_at_one_l841_84105


namespace pedro_more_squares_l841_84122

theorem pedro_more_squares (jesus_squares : ℕ) (linden_squares : ℕ) (pedro_squares : ℕ)
  (h1 : jesus_squares = 60) (h2 : linden_squares = 75) (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 :=
by
  sorry

end pedro_more_squares_l841_84122


namespace marble_distribution_l841_84195

theorem marble_distribution (x : ℚ) :
    (2 * x + 2) + (3 * x) + (x + 4) = 56 ↔ x = 25 / 3 := by
  sorry

end marble_distribution_l841_84195


namespace part_I_part_II_l841_84157

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
x^2 / a^2 + y^2 / b^2 = 1

theorem part_I (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (eccentricity : ℝ := c / a) (h3 : eccentricity = Real.sqrt 2 / 2) (vertex : ℝ × ℝ := (0, 1)) (h4 : vertex = (0, b)) 
  : ellipse_equation (Real.sqrt 2) 1 (0:ℝ) 1 :=
sorry

theorem part_II (a b k : ℝ) (x y : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = 1)
  (line_eq : ℝ → ℝ := fun x => k * x + 1) 
  (h3 : (1 + 2 * k^2) * x^2 + 4 * k * x = 0) 
  (distance_AB : ℝ := Real.sqrt 2 * 4 / 3) 
  (h4 : Real.sqrt (1 + k^2) * abs ((-4 * k) / (2 * k^2 + 1)) = distance_AB) 
  : (x, y) = (4/3, -1/3) ∨ (x, y) = (-4/3, -1/3) :=
sorry

end part_I_part_II_l841_84157


namespace total_area_correct_l841_84112

noncomputable def total_area (b l: ℝ) (h1: l = 3 * b) (h2: l * b = 588) : ℝ :=
  let rect_area := 588 -- Area of the rectangle
  let semi_circle_area := 24.5 * Real.pi -- Area of the semi-circle based on given diameter
  rect_area + semi_circle_area

theorem total_area_correct (b l: ℝ) (h1: l = 3 * b) (h2: l * b = 588) : 
  total_area b l h1 h2 = 588 + 24.5 * Real.pi :=
by
  sorry

end total_area_correct_l841_84112


namespace parabola_equation_l841_84182

noncomputable def parabola_vertex_form (x y a : ℝ) : Prop := y = a * (x - 3)^2 + 5

noncomputable def parabola_standard_form (x y : ℝ) : Prop := y = -3 * x^2 + 18 * x - 22

theorem parabola_equation (a : ℝ) (h_vertex : parabola_vertex_form 3 5 a) (h_point : parabola_vertex_form 2 2 a) :
  ∃ x y, parabola_standard_form x y :=
by
  sorry

end parabola_equation_l841_84182
