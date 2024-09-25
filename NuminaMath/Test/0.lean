namespace Jane_saves_five_dollars

import Mathlib

noncomputable def first_pair_cost : ℝ := 50
noncomputable def second_pair_cost_A : ℝ := first_pair_cost * 0.6
noncomputable def second_pair_cost_B : ℝ := first_pair_cost - 15
noncomputable def promotion_A_total_cost : ℝ := first_pair_cost + second_pair_cost_A
noncomputable def promotion_B_total_cost : ℝ := first_pair_cost + second_pair_cost_B
noncomputable def Jane_savings : ℝ := promotion_B_total_cost - promotion_A_total_cost

theorem Jane_saves_five_dollars : Jane_savings = 5 := by
  sorry

end Jane_saves_five_dollars

namespace executed_is_9

import Mathlib

-- Define the conditions based on given problem
variables (x K I : ℕ)

-- Condition 1: Number of killed
def number_killed (x : ℕ) : ℕ := 2 * x + 4

-- Condition 2: Number of injured
def number_injured (x : ℕ) : ℕ := (16 * x) / 3 + 8

-- Condition 3: Total of killed, injured, and executed is less than 98
def total_less_than_98 (x : ℕ) (k : ℕ) (i : ℕ) : Prop := k + i + x < 98

-- Condition 4: Relation between killed and executed
def killed_relation (x : ℕ) (k : ℕ) : Prop := k - 4 = 2 * x

-- The final theorem statement to prove
theorem executed_is_9 : ∃ x, number_killed x = 2 * x + 4 ∧
                       number_injured x = (16 * x) / 3 + 8 ∧
                       total_less_than_98 x (number_killed x) (number_injured x) ∧
                       killed_relation x (number_killed x) ∧
                       x = 9 :=
by
  sorry

end executed_is_9

namespace jack_last_10_shots_made

import Mathlib

theorem jack_last_10_shots_made (initial_shots : ℕ) (initial_percentage : ℚ)
  (additional_shots : ℕ) (new_percentage : ℚ)
  (initial_successful_shots : initial_shots * initial_percentage = 18)
  (total_shots : initial_shots + additional_shots = 40)
  (total_successful_shots : (initial_shots + additional_shots) * new_percentage = 25) :
  ∃ x : ℕ, x = 7 := by
sorry

end jack_last_10_shots_made

namespace tickets_to_be_sold

import Mathlib

theorem tickets_to_be_sold : 
  let total_tickets := 200
  let jude_tickets := 16
  let andrea_tickets := 4 * jude_tickets
  let sandra_tickets := 2 * jude_tickets + 8
  total_tickets - (jude_tickets + andrea_tickets + sandra_tickets) = 80 := by
  sorry

end tickets_to_be_sold

namespace remainder_is_correct

import Mathlib

def dividend : ℕ := 725
def divisor : ℕ := 36
def quotient : ℕ := 20

theorem remainder_is_correct : ∃ (remainder : ℕ), dividend = (divisor * quotient) + remainder ∧ remainder = 5 := by
  sorry

end remainder_is_correct

namespace average_speed_of_car

import Mathlib

theorem average_speed_of_car : 
  let distance1 := 30
  let speed1 := 60
  let distance2 := 35
  let speed2 := 70
  let distance3 := 36
  let speed3 := 80
  let distance4 := 20
  let speed4 := 55
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  let average_speed := total_distance / total_time
  average_speed = 66.70 := sorry

end average_speed_of_car

namespace binomial_divisible_by_prime

import Mathlib

theorem binomial_divisible_by_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  (Nat.choose n p) - (n / p) % p = 0 := 
sorry

end binomial_divisible_by_prime

namespace total_oranges

import Mathlib

theorem total_oranges (a b c : ℕ) (h1 : a = 80) (h2 : b = 60) (h3 : c = 120) : a + b + c = 260 :=
by
  sorry

end total_oranges

namespace increase_80_by_50_percent

import Mathlib

theorem increase_80_by_50_percent :
  let initial_number : ℕ := 80
  let increase_percentage : ℝ := 0.5
  initial_number + (initial_number * increase_percentage) = 120 :=
by
  sorry

end increase_80_by_50_percent

namespace inscribed_rectangle_area

import Mathlib

variables (b h x : ℝ)
variables (h_isosceles_triangle : b > 0 ∧ h > 0 ∧ x > 0 ∧ x < h)

noncomputable def rectangle_area (b h x : ℝ) : ℝ :=
  (b * x / h) * (h - x)

theorem inscribed_rectangle_area :
  rectangle_area b h x = (b * x / h) * (h - x) :=
by
  unfold rectangle_area
  sorry

end inscribed_rectangle_area

namespace cost_of_bananas_is_two

import Mathlib

variable (B : ℝ)

theorem cost_of_bananas_is_two (h : 1.20 * (3 + B) = 6) : B = 2 :=
by
  sorry

end cost_of_bananas_is_two

namespace candles_time

import Mathlib

/-- Prove that if two candles of equal length are lit at a certain time,
and by 6 PM one of the stubs is three times the length of the other,
the correct time to light the candles is 4:00 PM. -/

theorem candles_time :
  ∀ (ℓ : ℝ) (t : ℝ),
  (∀ t1 t2 : ℝ, t = t1 + t2 → 
    (180 - t1) = 3 * (300 - t2) / 3 → 
    18 <= 6 ∧ 0 <= t → ℓ / 180 * (180 - (t - 180)) = 3 * (ℓ / 300 * (300 - (6 - t))) →
    t = 4
  ) := 
by 
  sorry

end candles_time

namespace original_number_is_0_02

import Mathlib

theorem original_number_is_0_02 (x : ℝ) (h : 10000 * x = 4 / x) : x = 0.02 :=
by
  sorry

end original_number_is_0_02

namespace garden_perimeter_ratio

import Mathlib

theorem garden_perimeter_ratio (side_length : ℕ) (tripled_side_length : ℕ) (original_perimeter : ℕ) (new_perimeter : ℕ) (ratio : ℚ) :
  side_length = 50 →
  tripled_side_length = 3 * side_length →
  original_perimeter = 4 * side_length →
  new_perimeter = 4 * tripled_side_length →
  ratio = original_perimeter / new_perimeter →
  ratio = 1 / 3 :=
by
  sorry

end garden_perimeter_ratio

namespace only_solution

import Mathlib

def pythagorean_euler_theorem (p r : ℕ) : Prop :=
  ∃ (p r : ℕ), Nat.Prime p ∧ r > 0 ∧ (∑ i in Finset.range (r + 1), (p + i)^p) = (p + r + 1)^p

theorem only_solution (p r : ℕ) : pythagorean_euler_theorem p r ↔ p = 3 ∧ r = 2 :=
by
  sorry

end only_solution

namespace kangaroo_chase

import Mathlib

noncomputable def time_to_catch_up (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
  (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
  (initial_baby_jumps: ℕ): ℕ :=
  let jump_dist_baby := jump_dist_mother / jump_dist_reduction_factor
  let distance_mother := jumps_mother * jump_dist_mother
  let distance_baby := jumps_baby * jump_dist_baby
  let relative_velocity := distance_mother - distance_baby
  let initial_distance := initial_baby_jumps * jump_dist_baby
  (initial_distance / relative_velocity) * time_period

theorem kangaroo_chase :
 ∀ (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
   (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
   (initial_baby_jumps: ℕ),
  jumps_baby = 5 ∧ jumps_mother = 3 ∧ time_period = 2 ∧ 
  jump_dist_mother = 6 ∧ jump_dist_reduction_factor = 3 ∧ 
  initial_baby_jumps = 12 →
  time_to_catch_up jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps = 6 := 
by
  intros jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps _; sorry

end kangaroo_chase

namespace new_monthly_savings

import Mathlib

-- Definitions based on conditions
def monthly_salary := 4166.67
def initial_savings_percent := 0.20
def expense_increase_percent := 0.10

-- Calculations
def initial_savings := initial_savings_percent * monthly_salary
def initial_expenses := (1 - initial_savings_percent) * monthly_salary
def increased_expenses := initial_expenses + expense_increase_percent * initial_expenses
def new_savings := monthly_salary - increased_expenses

-- Lean statement to prove the question equals the answer given conditions
theorem new_monthly_savings :
  new_savings = 499.6704 := 
by
  sorry

end new_monthly_savings

namespace aiyanna_more_cookies_than_alyssa

import Mathlib

-- Definitions of the conditions
def alyssa_cookies : ℕ := 129
def aiyanna_cookies : ℕ := 140

-- The proof problem statement
theorem aiyanna_more_cookies_than_alyssa : (aiyanna_cookies - alyssa_cookies) = 11 := sorry

end aiyanna_more_cookies_than_alyssa

namespace additional_men_joined

import Mathlib

theorem additional_men_joined
    (M : ℕ) (X : ℕ)
    (h1 : M = 20)
    (h2 : M * 50 = (M + X) * 25) :
    X = 20 := by
  sorry

end additional_men_joined

namespace number_of_cirrus_clouds

import Mathlib

def C_cb := 3
def C_cu := 12 * C_cb
def C_ci := 4 * C_cu

theorem number_of_cirrus_clouds : C_ci = 144 :=
by
  sorry

end number_of_cirrus_clouds

namespace fourth_term_of_geometric_sequence_is_320

import Mathlib

theorem fourth_term_of_geometric_sequence_is_320
  (a : ℕ) (r : ℕ)
  (h_a : a = 5)
  (h_fifth_term : a * r^4 = 1280) :
  a * r^3 = 320 := 
by
  sorry

end fourth_term_of_geometric_sequence_is_320

namespace parallel_lines_m_values

import Mathlib

theorem parallel_lines_m_values (m : ℝ) :
  (∀ (x y : ℝ), (m - 2) * x - y + 5 = 0) ∧ 
  (∀ (x y : ℝ), (m - 2) * x + (3 - m) * y + 2 = 0) → 
  (m = 2 ∨ m = 4) :=
sorry

end parallel_lines_m_values

namespace shaded_region_area

import Mathlib

def area_of_shaded_region (grid_height grid_width triangle_base triangle_height : ℝ) : ℝ :=
  let total_area := grid_height * grid_width
  let triangle_area := 0.5 * triangle_base * triangle_height
  total_area - triangle_area

theorem shaded_region_area :
  area_of_shaded_region 3 15 5 3 = 37.5 :=
by 
  sorry

end shaded_region_area

namespace total_rooms_count

import Mathlib

noncomputable def apartment_area : ℕ := 160
noncomputable def living_room_area : ℕ := 60
noncomputable def other_room_area : ℕ := 20

theorem total_rooms_count (A : apartment_area = 160) (L : living_room_area = 60) (O : other_room_area = 20) :
  1 + (apartment_area - living_room_area) / other_room_area = 6 :=
by
  sorry

end total_rooms_count

namespace greatest_sum_consecutive_integers

import Mathlib

theorem greatest_sum_consecutive_integers (n : ℤ) (h : n * (n + 1) < 360) : n + (n + 1) ≤ 37 := by
  sorry

end greatest_sum_consecutive_integers

namespace part1_solution_part2_solution

import Mathlib

noncomputable def find_prices (price_peanuts price_tea : ℝ) : Prop :=
price_peanuts + 40 = price_tea ∧
50 * price_peanuts = 10 * price_tea

theorem part1_solution :
  ∃ (price_peanuts price_tea : ℝ), find_prices price_peanuts price_tea :=
by
  sorry

def cost_function (m : ℝ) : ℝ :=
6 * m + 36 * (60 - m)

def profit_function (m : ℝ) : ℝ :=
(10 - 6) * m + (50 - 36) * (60 - m)

noncomputable def max_profit := 540

theorem part2_solution :
  ∃ (m t : ℝ), 30 ≤ m ∧ m ≤ 40 ∧ cost_function m ≤ 1260 ∧ profit_function m = max_profit :=
by
  sorry

end part1_solution_part2_solution

namespace cost_per_tissue

import Mathlib

-- Annalise conditions
def boxes : ℕ := 10
def packs_per_box : ℕ := 20
def tissues_per_pack : ℕ := 100
def total_spent : ℝ := 1000

-- Definition for total packs and total tissues
def total_packs : ℕ := boxes * packs_per_box
def total_tissues : ℕ := total_packs * tissues_per_pack

-- The math problem: Prove the cost per tissue
theorem cost_per_tissue : (total_spent / total_tissues) = 0.05 := by
  sorry

end cost_per_tissue

namespace number_of_combinations

import Mathlib

-- Define the binomial coefficient (combinations) function
def C (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Our main theorem statement
theorem number_of_combinations (n k m : ℕ) (h1 : 1 ≤ n) (h2 : m > 1) :
  let valid_combinations := C (n - (k - 1) * (m - 1)) k;
  let invalid_combinations := n - (k - 1) * m;
  valid_combinations - invalid_combinations = 
  C (n - (k - 1) * (m - 1)) k - (n - (k - 1) * m) := by
  let valid_combinations := C (n - (k - 1) * (m - 1)) k
  let invalid_combinations := n - (k - 1) * m
  sorry

end number_of_combinations

namespace gcd_of_6Tn2_and_nplus1_eq_2

import Mathlib

theorem gcd_of_6Tn2_and_nplus1_eq_2 (n : ℕ) (h_pos : 0 < n) :
  Nat.gcd (6 * ((n * (n + 1) / 2)^2)) (n + 1) = 2 :=
sorry

end gcd_of_6Tn2_and_nplus1_eq_2

namespace geometric_sequence_problem

import Mathlib

variable (a_n : ℕ → ℝ)

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := λ n => a₁ * q^(n-1)

theorem geometric_sequence_problem (q a_1 : ℝ) (a_1_pos : a_1 = 9)
  (h : ∀ n, a_n n = geometric_sequence a_1 q n)
  (h5 : a_n 5 = a_n 3 * (a_n 4)^2) : 
  a_n 4 = 1/3 ∨ a_n 4 = -1/3 := by 
  sorry

end geometric_sequence_problem

namespace triangle_sets

import Mathlib

def forms_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_sets :
  ¬ forms_triangle 1 2 3 ∧ forms_triangle 20 20 30 ∧ forms_triangle 30 10 15 ∧ forms_triangle 4 15 7 :=
by
  sorry

end triangle_sets

namespace brownies_in_pan

import Mathlib

theorem brownies_in_pan : 
    ∀ (pan_length pan_width brownie_length brownie_width : ℕ), 
    pan_length = 24 -> 
    pan_width = 20 -> 
    brownie_length = 3 -> 
    brownie_width = 2 -> 
    (pan_length * pan_width) / (brownie_length * brownie_width) = 80 := 
by
  intros pan_length pan_width brownie_length brownie_width h1 h2 h3 h4
  sorry

end brownies_in_pan

namespace avg_weight_B_correct

import Mathlib

-- Definitions of the conditions
def students_A : ℕ := 24
def students_B : ℕ := 16
def avg_weight_A : ℝ := 40
def avg_weight_class : ℝ := 38

-- Definition of the total weight calculation for sections A and B
def total_weight_A : ℝ := students_A * avg_weight_A
def total_weight_class : ℝ := (students_A + students_B) * avg_weight_class

-- Defining the average weight of section B as the unknown to be proven
noncomputable def avg_weight_B : ℝ := 35

-- The theorem to prove that the average weight of section B is 35 kg
theorem avg_weight_B_correct : 
  total_weight_A + students_B * avg_weight_B = total_weight_class :=
by
  sorry

end avg_weight_B_correct

namespace participation_schemes_count

import Mathlib

-- Define the conditions
def num_people : ℕ := 6
def num_selected : ℕ := 4
def subjects : List String := ["math", "physics", "chemistry", "english"]
def not_in_english : List String := ["A", "B"]

-- Define the problem 
theorem participation_schemes_count : 
  ∃ total_schemes : ℕ , (total_schemes = 240) :=
by {
  sorry
}

end participation_schemes_count

namespace principal_amount_borrowed

import Mathlib

theorem principal_amount_borrowed (P R T SI : ℕ) (h₀ : SI = (P * R * T) / 100) (h₁ : SI = 5400) (h₂ : R = 12) (h₃ : T = 3) : P = 15000 :=
by
  sorry

end principal_amount_borrowed

namespace equilateral_triangle_side_length

import Mathlib

variable (R : ℝ)

theorem equilateral_triangle_side_length (R : ℝ) :
  (∃ (s : ℝ), s = R * Real.sqrt 3) :=
sorry

end equilateral_triangle_side_length

namespace time_addition_correct

import Mathlib

def start_time := (3, 0, 0) -- Representing 3:00:00 PM as (hours, minutes, seconds)
def additional_time := (315, 78, 30) -- Representing additional time as (hours, minutes, seconds)

noncomputable def resulting_time (start add : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (sh, sm, ss) := start -- start hours, minutes, seconds
  let (ah, am, as) := add -- additional hours, minutes, seconds
  let total_seconds := ss + as
  let extra_minutes := total_seconds / 60
  let remaining_seconds := total_seconds % 60
  let total_minutes := sm + am + extra_minutes
  let extra_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let total_hours := sh + ah + extra_hours
  let resulting_hours := (total_hours % 12) -- Modulo 12 for wrap-around
  (resulting_hours, remaining_minutes, remaining_seconds)

theorem time_addition_correct :
  let (A, B, C) := resulting_time start_time additional_time
  A + B + C = 55 := by
  sorry

end time_addition_correct

namespace factorization1_factorization2_factorization3

import Mathlib

-- Problem 1
theorem factorization1 (a x : ℝ) : a * x^2 - 4 * a = a * (x + 2) * (x - 2) :=
sorry

-- Problem 2
theorem factorization2 (m x y : ℝ) : m * x^2 + 2 * m * x * y + m * y^2 = m * (x + y)^2 :=
sorry

-- Problem 3
theorem factorization3 (a b : ℝ) : (1 / 2) * a^2 - a * b + (1 / 2) * b^2 = (1 / 2) * (a - b)^2 :=
sorry

end factorization1_factorization2_factorization3

namespace sqrt_domain

import Mathlib

def inequality_holds (x : ℝ) : Prop := x + 5 ≥ 0

theorem sqrt_domain (x : ℝ) : inequality_holds x ↔ x ≥ -5 := by
  sorry

end sqrt_domain

namespace uncle_zhang_age

import Mathlib

theorem uncle_zhang_age (z l : ℕ) (h1 : z + l = 56) (h2 : z = l - (l / 2)) : z = 24 :=
by sorry

end uncle_zhang_age

namespace price_of_shares

import Mathlib

variable (share_value : ℝ) (dividend_rate : ℝ) (tax_rate : ℝ) (effective_return : ℝ) (price : ℝ)

-- Given conditions
axiom H1 : share_value = 50
axiom H2 : dividend_rate = 0.185
axiom H3 : tax_rate = 0.05
axiom H4 : effective_return = 0.25
axiom H5 : 0.25 * price = 0.185 * 50 - (0.05 * (0.185 * 50))

-- Prove that the price at which the investor bought the shares is Rs. 35.15
theorem price_of_shares : price = 35.15 :=
by
  sorry

end price_of_shares

namespace total_new_students

import Mathlib

-- Given conditions
def number_of_schools : ℝ := 25.0
def average_students_per_school : ℝ := 9.88

-- Problem statement
theorem total_new_students : number_of_schools * average_students_per_school = 247 :=
by sorry

end total_new_students

namespace sam_has_75_dollars

import Mathlib

variable (S B : ℕ)

def condition1 := B = 2 * S - 25
def condition2 := S + B = 200

theorem sam_has_75_dollars (h1 : condition1 S B) (h2 : condition2 S B) : S = 75 := by
  sorry

end sam_has_75_dollars

namespace hexagon_coloring_count

import Mathlib

def num_possible_colorings : Nat :=
by
  /- There are 7 choices for first vertex A.
     Once A is chosen, there are 6 choices for the remaining vertices B, C, D, E, F considering the diagonal restrictions. -/
  let total_colorings := 7 * 6 ^ 5
  let restricted_colorings := 7 * 6 ^ 3
  let valid_colorings := total_colorings - restricted_colorings
  exact valid_colorings

theorem hexagon_coloring_count : num_possible_colorings = 52920 :=
  by
    /- Computation steps above show that the number of valid colorings is 52920 -/
    sorry   -- Proof computation already indicated

end hexagon_coloring_count

namespace halfway_between_3_4_and_5_7

import Mathlib

-- Define the two fractions
def frac1 := 3/4
def frac2 := 5/7

-- Define the average function for two fractions
def halfway_fract (a b : ℚ) : ℚ := (a + b) / 2

-- Prove that the halfway fraction between 3/4 and 5/7 is 41/56
theorem halfway_between_3_4_and_5_7 : 
  halfway_fract frac1 frac2 = 41/56 := 
by 
  sorry

end halfway_between_3_4_and_5_7

namespace abc_value

import Mathlib

-- Define constants for the problem
variable (a b c k : ℕ)

-- Assumptions based on the given conditions
axiom h1 : a - b = 3
axiom h2 : a^2 + b^2 = 29
axiom h3 : a^2 + b^2 + c^2 = k
axiom pos_k : k > 0
axiom pos_a : a > 0

-- The goal is to prove that abc = 10
theorem abc_value : a * b * c = 10 :=
by
  sorry

end abc_value

namespace repeating_decimal_base

import Mathlib

theorem repeating_decimal_base (k : ℕ) (h_pos : 0 < k) (h_repr : (9 : ℚ) / 61 = (3 * k + 4) / (k^2 - 1)) : k = 21 :=
  sorry

end repeating_decimal_base

namespace projection_ratio_zero

import Mathlib

variables (v w u p q : ℝ → ℝ) -- Assuming vectors are functions from ℝ to ℝ
variables (norm : (ℝ → ℝ) → ℝ) -- norm is a function from vectors to ℝ
variables (proj : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ)) -- proj is the projection function

-- Assume the conditions
axiom proj_p : p = proj v w
axiom proj_q : q = proj p u
axiom perp_uv : ∀ t, v t * u t = 0 -- u is perpendicular to v
axiom norm_ratio : norm p / norm v = 3 / 8

theorem projection_ratio_zero : norm q / norm v = 0 :=
by sorry

end projection_ratio_zero

namespace cost_of_large_fries

import Mathlib

noncomputable def cost_of_cheeseburger : ℝ := 3.65
noncomputable def cost_of_milkshake : ℝ := 2
noncomputable def cost_of_coke : ℝ := 1
noncomputable def cost_of_cookie : ℝ := 0.5
noncomputable def tax : ℝ := 0.2
noncomputable def toby_initial_amount : ℝ := 15
noncomputable def toby_remaining_amount : ℝ := 7
noncomputable def split_bill : ℝ := 2

theorem cost_of_large_fries : 
  let total_meal_cost := (split_bill * (toby_initial_amount - toby_remaining_amount))
  let total_cost_so_far := (2 * cost_of_cheeseburger) + cost_of_milkshake + cost_of_coke + (3 * cost_of_cookie) + tax
  total_meal_cost - total_cost_so_far = 4 := 
by
  sorry

end cost_of_large_fries

namespace men_absent_is_5

import Mathlib

-- Define the given conditions
def original_number_of_men : ℕ := 30
def planned_days : ℕ := 10
def actual_days : ℕ := 12

-- Prove the number of men absent (x) is 5, under given conditions
theorem men_absent_is_5 : ∃ x : ℕ, 30 * planned_days = (original_number_of_men - x) * actual_days ∧ x = 5 :=
by
  sorry

end men_absent_is_5

namespace female_managers_count

import Mathlib

def total_employees : ℕ := sorry
def female_employees : ℕ := 700
def managers : ℕ := (2 * total_employees) / 5
def male_employees : ℕ := total_employees - female_employees
def male_managers : ℕ := (2 * male_employees) / 5

theorem female_managers_count :
  ∃ (fm : ℕ), managers = fm + male_managers ∧ fm = 280 := by
  sorry

end female_managers_count

namespace cookies_indeterminate

import Mathlib

theorem cookies_indeterminate (bananas : ℕ) (boxes : ℕ) (bananas_per_box : ℕ) (cookies : ℕ)
  (h1 : bananas = 40)
  (h2 : boxes = 8)
  (h3 : bananas_per_box = 5)
  : ∃ c : ℕ, c = cookies :=
by sorry

end cookies_indeterminate

namespace ratio_pentagon_side_length_to_rectangle_width

import Mathlib

def pentagon_side_length (p : ℕ) (n : ℕ) := p / n
def rectangle_width (p : ℕ) (ratio : ℕ) := p / (2 * (1 + ratio))

theorem ratio_pentagon_side_length_to_rectangle_width :
  pentagon_side_length 60 5 / rectangle_width 80 3 = (6 : ℚ) / 5 :=
by {
  sorry
}

end ratio_pentagon_side_length_to_rectangle_width

namespace find_a8

import Mathlib

theorem find_a8 (a : ℕ → ℤ) (x : ℤ) :
  (1 + x)^10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 +
               a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 +
               a 7 * (1 - x)^7 + a 8 * (1 - x)^8 + a 9 * (1 - x)^9 +
               a 10 * (1 - x)^10 → a 8 = 180 := by
  sorry

end find_a8

namespace root_expression_value

import Mathlib

theorem root_expression_value 
  (m : ℝ) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2021 = 2024 := 
by 
  sorry

end root_expression_value

namespace algebraic_expression_value

import Mathlib

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m - 2 = 0) : 2 * m^2 - 2 * m = 4 := by
  sorry

end algebraic_expression_value

namespace probability_A_B_C_adjacent

import Mathlib

theorem probability_A_B_C_adjacent (students : Fin 5 → Prop) (A B C : Fin 5) :
  (students A ∧ students B ∧ students C) →
  (∃ n m : ℕ, n = 48 ∧ m = 12 ∧ m / n = (1 : ℚ) / 4) :=
by
  sorry

end probability_A_B_C_adjacent

namespace geometric_sequence_sum

import Mathlib

variable {a : ℕ → ℝ} -- Sequence terms
variable {S : ℕ → ℝ} -- Sum of the first n terms

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, S n = a 0 * (1 - (a n)) / (1 - a 1)
def is_arithmetic_sequence (x y z : ℝ) := 2 * y = x + z
def term_1_equals_1 (a : ℕ → ℝ) := a 0 = 1

-- Question: Prove that given the conditions, S_5 = 31
theorem geometric_sequence_sum (q : ℝ) (h_geom : is_geometric_sequence a q) 
  (h_sum : sum_of_first_n_terms a S) (h_arith : is_arithmetic_sequence (4 * a 0) (2 * a 1) (a 2)) 
  (h_a1 : term_1_equals_1 a) : S 5 = 31 :=
sorry

end geometric_sequence_sum

namespace correct_phone_call_sequence

import Mathlib

-- Define the six steps as an enumerated type.
inductive Step
| Dial
| WaitDialTone
| PickUpHandset
| StartConversationOrHangUp
| WaitSignal
| EndCall

open Step

-- Define the problem as a theorem.
theorem correct_phone_call_sequence : 
  ∃ sequence : List Step, sequence = [PickUpHandset, WaitDialTone, Dial, WaitSignal, StartConversationOrHangUp, EndCall] :=
sorry

end correct_phone_call_sequence

namespace minimum_value_expression

import Mathlib

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a = 1 ∧ b = 1 ∧ c = 1) :
  (a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2) / (a * b * c) = 48 * Real.sqrt 6 := 
by
  sorry

end minimum_value_expression

namespace abc_eq_1

import Mathlib

theorem abc_eq_1 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
(h7 : a + 1 / b^2 = b + 1 / c^2) (h8 : b + 1 / c^2 = c + 1 / a^2) :
  |a * b * c| = 1 :=
sorry

end abc_eq_1

namespace num_five_digit_integers

import Mathlib

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

theorem num_five_digit_integers : 
  let num_ways := factorial 5 / (factorial 2 * factorial 3)
  num_ways = 10 :=
by 
  sorry

end num_five_digit_integers

namespace total_hens_and_cows

import Mathlib

theorem total_hens_and_cows (H C : ℕ) (hH : H = 28) (h_feet : 2 * H + 4 * C = 136) : H + C = 48 :=
by
  -- Proof goes here 
  sorry

end total_hens_and_cows

namespace find_interest_rate

import Mathlib

noncomputable def amount : ℝ := 896
noncomputable def principal : ℝ := 799.9999999999999
noncomputable def time : ℝ := 2 + 2 / 5
noncomputable def interest : ℝ := amount - principal
noncomputable def rate : ℝ := interest / (principal * time)

theorem find_interest_rate :
  rate * 100 = 5 := by
  sorry

end find_interest_rate

namespace CD_is_b_minus_a_minus_c

import Mathlib

variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (A B C D : V) (a b c : V)

def AB : V := a
def AD : V := b
def BC : V := c

theorem CD_is_b_minus_a_minus_c (h1 : A + a = B) (h2 : A + b = D) (h3 : B + c = C) :
  D - C = b - a - c :=
by sorry

end CD_is_b_minus_a_minus_c

namespace probability_not_touch_outer_edge

import Mathlib

def checkerboard : ℕ := 10

def total_squares : ℕ := checkerboard * checkerboard

def perimeter_squares : ℕ := 4 * checkerboard - 4

def inner_squares : ℕ := total_squares - perimeter_squares

def probability : ℚ := inner_squares / total_squares

theorem probability_not_touch_outer_edge : probability = 16 / 25 :=
by
  sorry

end probability_not_touch_outer_edge

namespace LengthRatiosSimultaneous_LengthRatiosNonSimultaneous

import Mathlib

noncomputable section

-- Problem 1: Prove length ratios for simultaneous ignition
def LengthRatioSimultaneous (t : ℝ) : Prop :=
  let LA := 1 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosSimultaneous (t : ℝ) : LengthRatioSimultaneous t := sorry

-- Problem 2: Prove length ratios when one candle is lit 30 minutes earlier
def LengthRatioNonSimultaneous (t : ℝ) : Prop :=
  let LA := 5 / 6 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosNonSimultaneous (t : ℝ) : LengthRatioNonSimultaneous t := sorry

end LengthRatiosSimultaneous_LengthRatiosNonSimultaneous

namespace consecutive_odds_base_eqn

import Mathlib

-- Given conditions
def isOdd (n : ℕ) : Prop := n % 2 = 1

variables {C D : ℕ}

theorem consecutive_odds_base_eqn (C_odd : isOdd C) (D_odd : isOdd D) (consec : D = C + 2)
    (base_eqn : 2 * C^2 + 4 * C + 3 + 6 * D + 5 = 10 * (C + D) + 7) :
    C + D = 16 :=
sorry

end consecutive_odds_base_eqn

namespace count_perfect_squares

import Mathlib

theorem count_perfect_squares (N : Nat) :
  ∃ k : Nat, k = 1666 ∧ ∀ m, (∃ n, m = n * n ∧ m < 10^8 ∧ 36 ∣ m) ↔ (m = 36 * k ^ 2 ∧ k < 10^4) :=
sorry

end count_perfect_squares

namespace ratio_matt_fem_4_1

import Mathlib

-- Define Fem's current age
def FemCurrentAge : ℕ := 11

-- Define the condition about the sum of their ages in two years
def AgeSumInTwoYears (MattCurrentAge : ℕ) : Prop :=
  (FemCurrentAge + 2) + (MattCurrentAge + 2) = 59

-- Define the desired ratio as a property
def DesiredRatio (MattCurrentAge : ℕ) : Prop :=
  MattCurrentAge / FemCurrentAge = 4

-- Create the theorem statement
theorem ratio_matt_fem_4_1 (M : ℕ) (h : AgeSumInTwoYears M) : DesiredRatio M :=
  sorry

end ratio_matt_fem_4_1

namespace abs_m_plus_one

import Mathlib

theorem abs_m_plus_one (m : ℝ) (h : |m| = m + 1) : (4 * m - 1) ^ 4 = 81 := by
  sorry

end abs_m_plus_one

namespace total_earnings_correct

import Mathlib

noncomputable def total_earnings : ℝ :=
  let earnings1 := 12 * (2 + 15 / 60)
  let earnings2 := 15 * (1 + 40 / 60)
  let earnings3 := 10 * (3 + 10 / 60)
  earnings1 + earnings2 + earnings3

theorem total_earnings_correct : total_earnings = 83.75 := by
  sorry

end total_earnings_correct

namespace find_c

import Mathlib

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 15 = 3) : c = 9 :=
by sorry

end find_c

namespace total_pepper_weight

import Mathlib

theorem total_pepper_weight :
  let green_peppers := 2.8333333333333335
  let red_peppers := 3.254
  let yellow_peppers := 1.375
  let orange_peppers := 0.567
  (green_peppers + red_peppers + yellow_peppers + orange_peppers) = 8.029333333333333 := 
by
  sorry

end total_pepper_weight

namespace johns_number_is_1500

import Mathlib

def is_multiple_of (a b : Nat) : Prop := ∃ k, a = k * b

theorem johns_number_is_1500 (n : ℕ) (h1 : is_multiple_of n 125) (h2 : is_multiple_of n 30) (h3 : 1000 ≤ n ∧ n ≤ 3000) : n = 1500 :=
by
  -- proof structure goes here
  sorry

end johns_number_is_1500

namespace four_digit_sum

import Mathlib

theorem four_digit_sum (A B : ℕ) (hA : 1000 ≤ A ∧ A < 10000) (hB : 1000 ≤ B ∧ B < 10000) (h : A * B = 16^5 + 2^10) : A + B = 2049 := 
by sorry

end four_digit_sum

namespace loan_amount_is_900

import Mathlib

theorem loan_amount_is_900 (P R T SI : ℕ) (hR : R = 9) (hT : T = 9) (hSI : SI = 729)
    (h_simple_interest : SI = (P * R * T) / 100) : P = 900 := by
  sorry

end loan_amount_is_900

namespace buses_needed

import Mathlib

def total_students : ℕ := 111
def seats_per_bus : ℕ := 3

theorem buses_needed : total_students / seats_per_bus = 37 :=
by
  sorry

end buses_needed

namespace probability_XiaoCong_project_A_probability_same_project_not_C

import Mathlib

-- Definition of projects and conditions
inductive Project
| A | B | C

def XiaoCong : Project := sorry
def XiaoYing : Project := sorry

-- (1) Probability of Xiao Cong assigned to project A
theorem probability_XiaoCong_project_A : 
  (1 / 3 : ℝ) = 1 / 3 := 
by sorry

-- (2) Probability of Xiao Cong and Xiao Ying being assigned to the same project, given Xiao Ying not assigned to C
theorem probability_same_project_not_C : 
  (2 / 6 : ℝ) = 1 / 3 :=
by sorry

end probability_XiaoCong_project_A_probability_same_project_not_C

namespace tape_needed_for_large_box

import Mathlib

-- Definition of the problem conditions
def tape_per_large_box (L : ℕ) : Prop :=
  -- Each large box takes L feet of packing tape to seal
  -- Each medium box takes 2 feet of packing tape to seal
  -- Each small box takes 1 foot of packing tape to seal
  -- Each box also takes 1 foot of packing tape to stick the address label on
  -- Debbie packed two large boxes this afternoon
  -- Debbie packed eight medium boxes this afternoon
  -- Debbie packed five small boxes this afternoon
  -- Debbie used 44 feet of tape in total
  2 * L + 2 + 24 + 10 = 44

theorem tape_needed_for_large_box : ∃ L : ℕ, tape_per_large_box L ∧ L = 4 :=
by {
  -- Proof goes here
  sorry
}

end tape_needed_for_large_box

namespace expression_of_quadratic_function_coordinates_of_vertex

import Mathlib

def quadratic_function_through_points (a b : ℝ) : Prop :=
  (0 = a * (-3)^2 + b * (-3) + 3) ∧ (-5 = a * 2^2 + b * 2 + 3)

theorem expression_of_quadratic_function :
  ∃ a b : ℝ, quadratic_function_through_points a b ∧ ∀ x : ℝ, -x^2 - 2 * x + 3 = a * x^2 + b * x + 3 :=
by
  sorry

theorem coordinates_of_vertex :
  - (1 : ℝ) * (1 : ℝ) = (-1) / (2 * (-1)) ∧ 4 = -(1 - (-1) + 3) + 4 :=
by
  sorry

end expression_of_quadratic_function_coordinates_of_vertex

namespace water_on_wednesday

import Mathlib

-- Define the total water intake for the week.
def total_water : ℕ := 60

-- Define the water intake amounts for specific days.
def water_on_mon_thu_sat : ℕ := 9
def water_on_tue_fri_sun : ℕ := 8

-- Define the number of days for each intake.
def days_mon_thu_sat : ℕ := 3
def days_tue_fri_sun : ℕ := 3

-- Define the water intake calculated for specific groups of days.
def total_water_mon_thu_sat := water_on_mon_thu_sat * days_mon_thu_sat
def total_water_tue_fri_sun := water_on_tue_fri_sun * days_tue_fri_sun

-- Define the total water intake for these days combined.
def total_water_other_days := total_water_mon_thu_sat + total_water_tue_fri_sun

-- Define the water intake for Wednesday, which we need to prove to be 9 liters.
theorem water_on_wednesday : total_water - total_water_other_days = 9 := by
  -- Proof omitted.
  sorry

end water_on_wednesday

namespace total_weight_is_correct

import Mathlib

-- Define the weight of apples
def weight_of_apples : ℕ := 240

-- Define the multiplier for pears
def pears_multiplier : ℕ := 3

-- Define the weight of pears
def weight_of_pears := pears_multiplier * weight_of_apples

-- Define the total weight of apples and pears
def total_weight : ℕ := weight_of_apples + weight_of_pears

-- The theorem that states the total weight calculation
theorem total_weight_is_correct : total_weight = 960 := by
  sorry

end total_weight_is_correct

namespace eleven_hash_five

import Mathlib

def my_op (r s : ℝ) : ℝ := sorry

axiom op_cond1 : ∀ r : ℝ, my_op r 0 = r
axiom op_cond2 : ∀ r s : ℝ, my_op r s = my_op s r
axiom op_cond3 : ∀ r s : ℝ, my_op (r + 1) s = (my_op r s) + s + 1

theorem eleven_hash_five : my_op 11 5 = 71 :=
by {
    sorry
}

end eleven_hash_five

namespace number_of_items

import Mathlib

variable (s d : ℕ)
variable (total_money cost_sandwich cost_drink discount : ℝ)
variable (s_purchase_criterion : s > 5)
variable (total_money_value : total_money = 50.00)
variable (cost_sandwich_value : cost_sandwich = 6.00)
variable (cost_drink_value : cost_drink = 1.50)
variable (discount_value : discount = 5.00)

theorem number_of_items (h1 : total_money = 50.00)
(h2 : cost_sandwich = 6.00)
(h3 : cost_drink = 1.50)
(h4 : discount = 5.00)
(h5 : s > 5) :
  s + d = 9 :=
by
  sorry

end number_of_items

namespace B_pow_5_eq_rB_plus_sI

import Mathlib

def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 4, 5]

def I : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 0, 1]

theorem B_pow_5_eq_rB_plus_sI : 
  ∃ (r s : ℤ), r = 1169 ∧ s = -204 ∧ B^5 = r • B + s • I := 
by
  use 1169
  use -204
  sorry

end B_pow_5_eq_rB_plus_sI

namespace pieces_left

import Mathlib

def pieces_initial : ℕ := 900
def pieces_used : ℕ := 156

theorem pieces_left : pieces_initial - pieces_used = 744 := by
  sorry

end pieces_left

namespace salaries_proof

import Mathlib

-- Define salaries as real numbers
variables (a b c d : ℝ)

-- Define assumptions
def conditions := 
  (a + b + c + d = 4000) ∧
  (0.05 * a + 0.15 * b = c) ∧ 
  (0.25 * d = 0.3 * b) ∧
  (b = 3 * c)

-- Define the solution as found
def solution :=
  (a = 2365.55) ∧
  (b = 645.15) ∧
  (c = 215.05) ∧
  (d = 774.18)

-- Prove that given the conditions, the solution holds
theorem salaries_proof : 
  (conditions a b c d) → (solution a b c d) := by
  sorry

end salaries_proof

namespace max_quotient

import Mathlib

theorem max_quotient (x y : ℝ) (h1 : -5 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 6) : 
  ∃ z, z = (x + y) / x ∧ ∀ w, w = (x + y) / x → w ≤ 0 :=
by
  sorry

end max_quotient

namespace triangle_equilateral_if_condition

import Mathlib

-- Define the given conditions
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Opposite sides

-- Assume the condition that a/ cos(A) = b/ cos(B) = c/ cos(C)
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.cos C

-- The theorem to prove under these conditions
theorem triangle_equilateral_if_condition (A B C a b c : ℝ) 
  (h : triangle_condition A B C a b c) : 
  A = B ∧ B = C :=
sorry

end triangle_equilateral_if_condition

namespace problem_a_problem_d

import Mathlib

theorem problem_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : (1 / (a * b)) ≥ 1 / 4 :=
by
  sorry

theorem problem_d (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : a^2 + b^2 ≥ 8 :=
by
  sorry

end problem_a_problem_d

namespace max_min_x_plus_inv_x

import Mathlib

-- We're assuming existence of 101 positive numbers with given conditions.
variable {x : ℝ}
variable {y : Fin 100 → ℝ}

-- Conditions given in the problem
def cumulative_sum (x : ℝ) (y : Fin 100 → ℝ) : Prop :=
  0 < x ∧ (∀ i, 0 < y i) ∧ x + (∑ i, y i) = 102 ∧ 1 / x + (∑ i, 1 / y i) = 102

-- The theorem to prove the maximum and minimum value of x + 1/x
theorem max_min_x_plus_inv_x (x : ℝ) (y : Fin 100 → ℝ) (h : cumulative_sum x y) : 
  (x + 1 / x ≤ 405 / 102) ∧ (x + 1 / x ≥ 399 / 102) := 
  sorry

end max_min_x_plus_inv_x

namespace min_value_of_function_product_inequality

import Mathlib

-- Part (1) Lean 4 statement
theorem min_value_of_function (x : ℝ) (hx : x > -1) : 
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 := 
by 
  sorry

-- Part (2) Lean 4 statement
theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) : 
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := 
by 
  sorry

end min_value_of_function_product_inequality

namespace harry_morning_ratio

import Mathlib

-- Define the total morning routine time
def total_morning_routine_time : ℕ := 45

-- Define the time taken to buy coffee and a bagel
def time_buying_coffee_and_bagel : ℕ := 15

-- Calculate the time spent reading the paper and eating
def time_reading_and_eating : ℕ :=
  total_morning_routine_time - time_buying_coffee_and_bagel

-- Define the ratio of the time spent reading and eating to buying coffee and a bagel
def ratio_reading_eating_to_buying_coffee_bagel : ℚ :=
  (time_reading_and_eating : ℚ) / (time_buying_coffee_and_bagel : ℚ)

-- State the theorem
theorem harry_morning_ratio : ratio_reading_eating_to_buying_coffee_bagel = 2 := 
by
  sorry

end harry_morning_ratio

namespace quadratic_real_roots_m_range

import Mathlib

theorem quadratic_real_roots_m_range :
  ∀ (m : ℝ), (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 :=
by sorry

end quadratic_real_roots_m_range

namespace find_r_s

import Mathlib

theorem find_r_s (r s : ℚ) :
  (-3)^5 - 2*(-3)^4 + 3*(-3)^3 - r*(-3)^2 + s*(-3) - 8 = 0 ∧
  2^5 - 2*(2^4) + 3*(2^3) - r*(2^2) + s*2 - 8 = 0 →
  (r, s) = (-482/15, -1024/15) :=
by
  sorry

end find_r_s

namespace suff_but_not_nec

import Mathlib

def M (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0
def N (a : ℝ) : Prop := ∃ x : ℝ, (a - 3) * x + 1 = 0

theorem suff_but_not_nec (a : ℝ) : M a → N a ∧ ¬(N a → M a) := by
  sorry

end suff_but_not_nec

namespace problem_statement

import Mathlib

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 2)

def S : Set ℝ := {y | ∃ x ≥ 0, y = f x}

theorem problem_statement :
  (∀ y ∈ S, y ≤ 2) ∧ (¬ (2 ∈ S)) ∧ (∀ y ∈ S, y ≥ 3 / 2) ∧ (3 / 2 ∈ S) :=
by
  sorry

end problem_statement

namespace jerry_remaining_money

import Mathlib

-- Define initial money
def initial_money := 18

-- Define amount spent on video games
def spent_video_games := 6

-- Define amount spent on a snack
def spent_snack := 3

-- Define total amount spent
def total_spent := spent_video_games + spent_snack

-- Define remaining money after spending
def remaining_money := initial_money - total_spent

theorem jerry_remaining_money : remaining_money = 9 :=
by
  sorry

end jerry_remaining_money

namespace 

import Mathlib

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

theorem find_pairs {a b : ℝ} :
  (0 < b) → (b ≤ 1) → (0 < a) → (a < 1) → (2 * a + b ≤ 2) →
  (∀ x y : ℝ, f a b (x * y) + f a b (x + y) ≥ f a b x * f a b y) :=
by
  intros h_b_gt_zero h_b_le_one h_a_gt_zero h_a_lt_one h_2a_b_le_2
  sorry

end 

namespace inequality_order

import Mathlib

theorem inequality_order (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) : 
  b > (a^4 - b^4) / (a - b) ∧ (a^4 - b^4) / (a - b) > (a + b) / 2 ∧ (a + b) / 2 > 2 * a * b :=
by 
  sorry

end inequality_order

namespace divisor_inequality

import Mathlib

variable (k p : Nat)

-- Conditions
def is_prime (p : Nat) : Prop := Nat.Prime p
def is_divisor_of (k d : Nat) : Prop := ∃ m : Nat, d = k * m

-- Given conditions and the theorem to be proved
theorem divisor_inequality (h1 : k > 3) (h2 : is_prime p) (h3 : is_divisor_of k (2^p + 1)) : k ≥ 2 * p + 1 :=
  sorry

end divisor_inequality

namespace area_of_park

import Mathlib

-- Definitions of conditions
def ratio_length_breadth (L B : ℝ) : Prop := L / B = 1 / 3
def cycling_time_distance (speed time perimeter : ℝ) : Prop := perimeter = speed * time

theorem area_of_park :
  ∃ (L B : ℝ),
    ratio_length_breadth L B ∧
    cycling_time_distance 12 (8 / 60) (2 * (L + B)) ∧
    L * B = 120000 := by
  sorry

end area_of_park

namespace match_Tile_C_to_Rectangle_III

import Mathlib

-- Define the structure for a Tile
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the given tiles
def Tile_A : Tile := { top := 5, right := 3, bottom := 7, left := 2 }
def Tile_B : Tile := { top := 3, right := 6, bottom := 2, left := 8 }
def Tile_C : Tile := { top := 7, right := 9, bottom := 1, left := 3 }
def Tile_D : Tile := { top := 1, right := 8, bottom := 5, left := 9 }

-- The proof problem: Prove that Tile C should be matched to Rectangle III
theorem match_Tile_C_to_Rectangle_III : (Tile_C = { top := 7, right := 9, bottom := 1, left := 3 }) → true := 
by
  intros
  sorry

end match_Tile_C_to_Rectangle_III

namespace A_investment

import Mathlib

theorem A_investment (x : ℝ) (hx : 0 < x) :
  (∃ a b c d e : ℝ,
    a = x ∧ b = 12 ∧ c = 200 ∧ d = 6 ∧ e = 60 ∧ 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    ((a * b) / (a * b + c * d)) * 100 = e)
  → x = 150 :=
by
  sorry

end A_investment

namespace expected_return_correct

import Mathlib

-- Define the probabilities
def p1 := 1/4
def p2 := 1/4
def p3 := 1/6
def p4 := 1/3

-- Define the payouts
def payout (n : ℕ) (previous_odd : Bool) : ℝ :=
  match n with
  | 1 => 2
  | 2 => if previous_odd then -3 else 0
  | 3 => 0
  | 4 => 5
  | _ => 0

-- Define the expected values of one throw
def E1 : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

def E2_odd : ℝ :=
  p1 * payout 1 true + p2 * payout 2 true + p3 * payout 3 true + p4 * payout 4 true

def E2_even : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

-- Define the probability of throwing an odd number first
def p_odd : ℝ := p1 + p3

-- Define the probability of not throwing an odd number first
def p_even : ℝ := 1 - p_odd

-- Define the total expected return
def total_expected_return : ℝ :=
  E1 + (p_odd * E2_odd + p_even * E2_even)


theorem expected_return_correct :
  total_expected_return = 4.18 :=
  by
    -- The proof is omitted
    sorry

end expected_return_correct

namespace Thomas_speed_greater_than_Jeremiah

import Mathlib

-- Define constants
def Thomas_passes_kilometers_per_hour := 5
def Jeremiah_passes_kilometers_per_hour := 6

-- Define speeds (in meters per hour)
def Thomas_speed := Thomas_passes_kilometers_per_hour * 1000
def Jeremiah_speed := Jeremiah_passes_kilometers_per_hour * 1000

-- Define hypothetical additional distances
def Thomas_hypothetical_additional_distance := 600 * 2
def Jeremiah_hypothetical_additional_distance := 50 * 2

-- Define effective distances traveled
def Thomas_effective_distance := Thomas_speed + Thomas_hypothetical_additional_distance
def Jeremiah_effective_distance := Jeremiah_speed + Jeremiah_hypothetical_additional_distance

-- Theorem to prove
theorem Thomas_speed_greater_than_Jeremiah : Thomas_effective_distance > Jeremiah_effective_distance := by
  -- Placeholder for the proof
  sorry

end Thomas_speed_greater_than_Jeremiah

namespace apprentice_daily_output

import Mathlib.Data.Real.Basic

namespace Production

variables (x y : ℝ)

theorem apprentice_daily_output
  (h1 : 4 * x + 7 * y = 765)
  (h2 : 6 * x + 2 * y = 765) :
  y = 45 :=
sorry

end Production

end apprentice_daily_output

namespace nishita_common_shares

import Mathlib

def annual_dividend_preferred_shares (num_preferred_shares : ℕ) (par_value : ℕ) (dividend_rate_preferred : ℕ) : ℕ :=
  (dividend_rate_preferred * par_value * num_preferred_shares) / 100

def annual_dividend_common_shares (total_dividend : ℕ) (dividend_preferred : ℕ) : ℕ :=
  total_dividend - dividend_preferred

def number_of_common_shares (annual_dividend_common : ℕ) (par_value : ℕ) (annual_rate_common : ℕ) : ℕ :=
  annual_dividend_common / ((annual_rate_common * par_value) / 100)

theorem nishita_common_shares (total_annual_dividend : ℕ) (num_preferred_shares : ℕ)
                             (par_value : ℕ) (dividend_rate_preferred : ℕ)
                             (semi_annual_rate_common : ℕ) : 
                             (number_of_common_shares (annual_dividend_common_shares total_annual_dividend 
                             (annual_dividend_preferred_shares num_preferred_shares par_value dividend_rate_preferred)) 
                             par_value (semi_annual_rate_common * 2)) = 3000 :=
by
  -- Provide values specific to the problem
  let total_annual_dividend := 16500
  let num_preferred_shares := 1200
  let par_value := 50
  let dividend_rate_preferred := 10
  let semi_annual_rate_common := 3.5
  sorry

end nishita_common_shares

namespace unique_x1_exists

import Mathlib

theorem unique_x1_exists (x : ℕ → ℝ) :
  (∀ n : ℕ+, x (n+1) = x n * (x n + 1 / n)) →
  ∃! (x1 : ℝ), (∀ n : ℕ+, 0 < x n ∧ x n < x (n+1) ∧ x (n+1) < 1) :=
sorry

end unique_x1_exists

namespace charity_donation_correct

import Mathlib

-- Define each donation series for Suzanne, Maria, and James
def suzanne_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 10
  | (n+1)  => 2 * suzanne_donation_per_km n

def maria_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 15
  | (n+1)  => 1.5 * maria_donation_per_km n

def james_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 20
  | (n+1)  => 2 * james_donation_per_km n

-- Total donations after 5 kilometers
def total_donation_suzanne : ℝ := (List.range 5).map suzanne_donation_per_km |>.sum
def total_donation_maria : ℝ := (List.range 5).map maria_donation_per_km |>.sum
def total_donation_james : ℝ := (List.range 5).map james_donation_per_km |>.sum

def total_donation_charity : ℝ :=
  total_donation_suzanne + total_donation_maria + total_donation_james

-- Statement to be proven
theorem charity_donation_correct : total_donation_charity = 1127.81 := by
  sorry

end charity_donation_correct

namespace negation_proof

import Mathlib

open Classical

variable {x : ℝ}

theorem negation_proof :
  (∀ x : ℝ, (x + 1) ≥ 0 ∧ (x^2 - x) ≤ 0) ↔ ¬ (∃ x_0 : ℝ, (x_0 + 1) < 0 ∨ (x_0^2 - x_0) > 0) := 
by
  sorry

end negation_proof

namespace carsProducedInEurope

import Mathlib

-- Definitions of the conditions
def carsProducedInNorthAmerica : ℕ := 3884
def totalCarsProduced : ℕ := 6755

-- Theorem statement
theorem carsProducedInEurope : ∃ (carsProducedInEurope : ℕ), totalCarsProduced = carsProducedInNorthAmerica + carsProducedInEurope ∧ carsProducedInEurope = 2871 := by
  sorry

end carsProducedInEurope

namespace tangent_line_inv_g_at_0

import Mathlib

noncomputable def g (x : ℝ) := Real.log x

theorem tangent_line_inv_g_at_0 
  (h₁ : ∀ x, g x = Real.log x) 
  (h₂ : ∀ x, x > 0): 
  ∃ m b, (∀ x y, y = g⁻¹ x → y - m * x = b) ∧ 
         (m = 1) ∧ 
         (b = 1) ∧ 
         (∀ x y, x - y + 1 = 0) := 
by
  sorry

end tangent_line_inv_g_at_0

namespace salary_january

import Mathlib

variable (J F M A May : ℝ)

theorem salary_january 
  (h1 : J + F + M + A = 32000) 
  (h2 : F + M + A + May = 33600) 
  (h3 : May = 6500) : 
  J = 4900 := 
by {
 sorry 
}

end salary_january

namespace find_central_angle

import Mathlib

theorem find_central_angle
  (θ r : ℝ)
  (h1 : r * θ = 2 * π)
  (h2 : (1 / 2) * r^2 * θ = 3 * π) :
  θ = 2 * π / 3 := 
sorry

end find_central_angle

namespace union_of_A_and_B

import Mathlib

namespace SetProof

def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x > 0}
def expectedUnion : Set ℝ := {x | -2 ≤ x}

theorem union_of_A_and_B : (A ∪ B) = expectedUnion := by
  sorry

end SetProof

end union_of_A_and_B

namespace max_triangles_formed

import Mathlib

-- Define the triangles and their properties
structure EquilateralTriangle (α : Type) :=
(midpoint_segment : α) -- Each triangle has a segment connecting the midpoints of two sides

variables {α : Type} [OrderedSemiring α]

-- Define the condition of being mirrored horizontally
def areMirroredHorizontally (A B : EquilateralTriangle α) : Prop := 
  -- Placeholder for any formalization needed to specify mirrored horizontally
  sorry

-- Movement conditions and number of smaller triangles
def numberOfSmallerTrianglesAtMaxOverlap (A B : EquilateralTriangle α) (move_horizontally : α) : ℕ :=
  -- Placeholder function/modeling for counting triangles during movement
  sorry

-- Statement of our main theorem
theorem max_triangles_formed (A B : EquilateralTriangle α) (move_horizontally : α) 
  (h_mirrored : areMirroredHorizontally A B) :
  numberOfSmallerTrianglesAtMaxOverlap A B move_horizontally = 11 :=
sorry

end max_triangles_formed

namespace daughter_and_child_weight

import Mathlib

variables (M D C : ℝ)

-- Conditions
def condition1 : Prop := M + D + C = 160
def condition2 : Prop := D = 40
def condition3 : Prop := C = (1/5) * M

-- Goal (Question)
def goal : Prop := D + C = 60

theorem daughter_and_child_weight
  (h1 : condition1 M D C)
  (h2 : condition2 D)
  (h3 : condition3 M C) : goal D C :=
by
  sorry

end daughter_and_child_weight

namespace sum_of_integers_is_18

import Mathlib

theorem sum_of_integers_is_18 (a b : ℕ) (h1 : b = 2 * a) (h2 : a * b + a + b = 156) (h3 : Nat.gcd a b = 1) (h4 : a < 25) : a + b = 18 :=
by
  sorry

end sum_of_integers_is_18

namespace exponent_equality

import Mathlib

theorem exponent_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 :=
by
  sorry

end exponent_equality

namespace find_a_in_triangle

import Mathlib

theorem find_a_in_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : c = 3)
  (h2 : C = Real.pi / 3)
  (h3 : Real.sin B = 2 * Real.sin A)
  (h4 : a = 3) :
  a = Real.sqrt 3 := by
  sorry

end find_a_in_triangle

namespace bottles_in_cups

import Mathlib

-- Defining the given conditions
variables (BOTTLE GLASS CUP JUG : ℕ)

axiom h1 : JUG = BOTTLE + GLASS
axiom h2 : 2 * JUG = 7 * GLASS
axiom h3 : BOTTLE = CUP + 2 * GLASS

theorem bottles_in_cups : BOTTLE = 5 * CUP :=
sorry

end bottles_in_cups

namespace simplified_expression_evaluates_to_2

import Mathlib

-- Definitions based on given conditions:
def x := 2 -- where x = (1/2)^(-1)
def y := 1 -- where y = (-2023)^0

-- Main statement to prove:
theorem simplified_expression_evaluates_to_2 :
  ((2 * x - y) / (x + y) - (x * x - 2 * x * y + y * y) / (x * x - y * y)) / (x - y) / (x + y) = 2 :=
by
  sorry

end simplified_expression_evaluates_to_2

namespace positive_even_integers_less_than_1000_not_divisible_by_3_or_11

import Mathlib

theorem positive_even_integers_less_than_1000_not_divisible_by_3_or_11 :
  ∃ n : ℕ, n = 108 ∧
    (∀ m : ℕ, 0 < m → 2 ∣ m → m < 1000 → (¬ (3 ∣ m) ∧ ¬ (11 ∣ m) ↔ m ≤ n)) :=
sorry

end positive_even_integers_less_than_1000_not_divisible_by_3_or_11

namespace total_growth_of_trees

import Mathlib

theorem total_growth_of_trees :
  let t1_growth_rate := 1 -- first tree grows 1 meter/day
  let t2_growth_rate := 2 -- second tree grows 2 meters/day
  let t3_growth_rate := 2 -- third tree grows 2 meters/day
  let t4_growth_rate := 3 -- fourth tree grows 3 meters/day
  let days := 4
  t1_growth_rate * days + t2_growth_rate * days + t3_growth_rate * days + t4_growth_rate * days = 32 :=
by
  let t1_growth_rate := 1
  let t2_growth_rate := 2
  let t3_growth_rate := 2
  let t4_growth_rate := 3
  let days := 4
  sorry

end total_growth_of_trees

namespace find_breadth_of_cuboid

import Mathlib

variable (l : ℝ) (h : ℝ) (surface_area : ℝ) (b : ℝ)

theorem find_breadth_of_cuboid (hL : l = 10) (hH : h = 6) (hSA : surface_area = 480) 
  (hFormula : surface_area = 2 * (l * b + b * h + h * l)) : b = 11.25 := by
  sorry

end find_breadth_of_cuboid

namespace star_value

import Mathlib

-- Define the operation a star b
def star (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

-- We want to prove that 5 star 3 = 4
theorem star_value : star 5 3 = 4 := by
  sorry

end star_value

namespace file_size

import Mathlib

-- Definitions based on conditions
def upload_speed : ℕ := 8 -- megabytes per minute
def upload_time : ℕ := 20 -- minutes

-- Goal to prove
theorem file_size:
  (upload_speed * upload_time = 160) :=
by sorry

end file_size

namespace g_of_3_equals_5

import Mathlib

def g (x : ℝ) : ℝ := 2 * (x - 2) + 3

theorem g_of_3_equals_5 :
  g 3 = 5 :=
by
  sorry

end g_of_3_equals_5

namespace ratio_of_age_difference

import Mathlib

theorem ratio_of_age_difference (R J K : ℕ) 
  (h1 : R = J + 6) 
  (h2 : R + 4 = 2 * (J + 4)) 
  (h3 : (R + 4) * (K + 4) = 108) : 
  (R - J) / (R - K) = 2 :=
by 
  sorry

end ratio_of_age_difference

namespace integer_sided_triangle_with_60_degree_angle_exists

import Mathlib

theorem integer_sided_triangle_with_60_degree_angle_exists 
  (m n t : ℤ) : 
  ∃ (x y z : ℤ), (x = (m^2 - n^2) * t) ∧ 
                  (y = m * (m - 2 * n) * t) ∧ 
                  (z = (m^2 - m * n + n^2) * t) := by
  sorry

end integer_sided_triangle_with_60_degree_angle_exists

namespace area_of_circle_with_given_circumference

import Mathlib

-- Defining the given problem's conditions as variables
variables (C : ℝ) (r : ℝ) (A : ℝ)
  
-- The condition that circumference is 12π meters
def circumference_condition : Prop := C = 12 * Real.pi
  
-- The relationship between circumference and radius
def radius_relationship : Prop := C = 2 * Real.pi * r
  
-- The formula to calculate the area of the circle
def area_formula : Prop := A = Real.pi * r^2
  
-- The proof goal that we need to establish
theorem area_of_circle_with_given_circumference :
  circumference_condition C ∧ radius_relationship C r ∧ area_formula A r → A = 36 * Real.pi :=
by
  intros
  sorry -- Skipping the proof, to be done later

end area_of_circle_with_given_circumference

namespace eval_expr

import Mathlib

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr

namespace x_sq_plus_3x_eq_1

import Mathlib

theorem x_sq_plus_3x_eq_1 (x : ℝ) (h : (x^2 + 3*x)^2 + 2*(x^2 + 3*x) - 3 = 0) : x^2 + 3*x = 1 :=
sorry

end x_sq_plus_3x_eq_1

namespace average_salary_correct

import Mathlib

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 15000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 9000 := by
  -- proof is skipped
  sorry

end average_salary_correct

namespace first_month_sale

import Mathlib

def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029
def sale6 : ℕ := 4937
def average_sale : ℕ := 5600

theorem first_month_sale :
  let total_sales := average_sale * 6
  let known_sales := sale2 + sale3 + sale4 + sale5 + sale6
  let sale1 := total_sales - known_sales
  sale1 = 5266 :=
by
  sorry

end first_month_sale

namespace valid_schedule_count

import Mathlib

theorem valid_schedule_count :
  ∃ (valid_schedules : Finset (Fin 8 → Option (Fin 4))),
    valid_schedules.card = 488 ∧
    (∀ (schedule : Fin 8 → Option (Fin 4)), schedule ∈ valid_schedules →
      (∀ i : Fin 7, schedule i ≠ none ∧ schedule (i + 1) ≠ schedule i) ∧
      schedule 4 = none) :=
sorry

end valid_schedule_count

namespace function_has_exactly_one_zero

import Mathlib

open Set

-- Conditions
def a_gt_3 (a : ℝ) : Prop := a > 3
def f (x a : ℝ) : ℝ := x^2 - a * x + 1

-- Theorem Statement
theorem function_has_exactly_one_zero (a : ℝ) (h : a_gt_3 a) :
  ∃! x ∈ Ioo 0 2, f x a = 0 := sorry

end function_has_exactly_one_zero

namespace min_value_expression

import Mathlib

theorem min_value_expression (x : ℝ) (h : x > 1) : 
  ∃ min_val, min_val = 6 ∧ ∀ y > 1, 2 * y + 2 / (y - 1) ≥ min_val :=
by  
  use 6
  sorry

end min_value_expression

namespace region_area

import Mathlib

noncomputable def area_of_region_outside_hexagon_inside_semicircles (s : ℝ) : ℝ :=
  let area_hexagon := (3 * Real.sqrt 3 / 2) * s^2
  let area_semicircle := (1/2) * Real.pi * (s/2)^2
  let total_area_semicircles := 6 * area_semicircle
  let total_area_circles := 6 * Real.pi * (s/2)^2
  total_area_circles - area_hexagon

theorem region_area (s := 2) : area_of_region_outside_hexagon_inside_semicircles s = (6 * Real.pi - 6 * Real.sqrt 3) :=
by
  sorry  -- Proof is skipped.

end region_area

namespace prove_scientific_notation

import Mathlib

def scientific_notation_correct : Prop :=
  340000 = 3.4 * (10 ^ 5)

theorem prove_scientific_notation : scientific_notation_correct :=
  by
    sorry

end prove_scientific_notation

namespace original_perimeter_not_necessarily_multiple_of_four

import Mathlib

/-
Define the conditions given in the problem:
1. A rectangle is divided into several smaller rectangles.
2. The perimeter of each of these smaller rectangles is a multiple of 4.
-/
structure Rectangle where
  length : ℕ
  width : ℕ

def perimeter (r : Rectangle) : ℕ :=
  2 * (r.length + r.width)

def is_multiple_of_four (n : ℕ) : Prop :=
  n % 4 = 0

def smaller_rectangles (rs : List Rectangle) : Prop :=
  ∀ r ∈ rs, is_multiple_of_four (perimeter r)

-- Define the main statement to be proved
theorem original_perimeter_not_necessarily_multiple_of_four (original : Rectangle) (rs : List Rectangle)
  (h1 : smaller_rectangles rs) (h2 : ∀ r ∈ rs, r.length * r.width = original.length * original.width) :
  ¬ is_multiple_of_four (perimeter original) :=
by
  sorry

end original_perimeter_not_necessarily_multiple_of_four

namespace cost_price_USD

import Mathlib

-- Assume the conditions in Lean as given:
variable {C_USD : ℝ}

def condition1 (C_USD : ℝ) : Prop := 0.9 * C_USD + 200 = 1.04 * C_USD

theorem cost_price_USD (h : condition1 C_USD) : C_USD = 200 / 0.14 :=
by
  sorry

end cost_price_USD

namespace 

import Mathlib

theorem log_one_plus_x_sq_lt_x_sq {x : ℝ} (hx : 0 < x) : 
  Real.log (1 + x^2) < x^2 := 
sorry

end 

namespace sqrt_7_estimate

import Mathlib

theorem sqrt_7_estimate (h1 : 4 < 7) (h2 : 7 < 9) (h3 : Nat.sqrt 4 = 2) (h4 : Nat.sqrt 9 = 3) : 2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 :=
  by {
    -- the proof would go here, but use 'sorry' to omit it
    sorry
  }

end sqrt_7_estimate

namespace complement_of_P_in_U

import Mathlib

/-- Definitions of sets U and P -/
def U := { y : ℝ | ∃ x : ℝ, x > 1 ∧ y = Real.log x / Real.log 2 }
def P := { y : ℝ | ∃ x : ℝ, x > 2 ∧ y = 1 / x }

/-- The complement of P in U -/
def complement_U_P := { y : ℝ | y = 0 ∨ y ≥ 1 / 2 }

/-- Proving the complement of P in U is as expected -/
theorem complement_of_P_in_U : { y : ℝ | y ∈ U ∧ y ∉ P } = complement_U_P := by
  sorry

end complement_of_P_in_U

namespace number_of_participants

import Mathlib

-- Define the conditions and theorem
theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 231) : n = 22 :=
  sorry

end number_of_participants

namespace proof_problem

import Mathlib

axiom sqrt (x : ℝ) : ℝ
axiom cbrt (x : ℝ) : ℝ
noncomputable def sqrtValue : ℝ :=
  sqrt 81

theorem proof_problem (m n : ℝ) (hm : sqrt m = 3) (hn : cbrt n = -4) : sqrt (2 * m - n - 1) = 9 ∨ sqrt (2 * m - n - 1) = -9 :=
by
  sorry

end proof_problem

namespace starting_number_of_sequence

import Mathlib

theorem starting_number_of_sequence :
  ∃ (start : ℤ), 
    (∀ n, 0 ≤ n ∧ n < 8 → start + n * 11 ≤ 119) ∧ 
    (∃ k, 1 ≤ k ∧ k ≤ 8 ∧ 119 = start + (k - 1) * 11) ↔ start = 33 :=
by
  sorry

end starting_number_of_sequence

namespace find_point_P

import Mathlib

structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def vector (P Q : Point) : Point :=
⟨Q.x - P.x, Q.y - P.y⟩

def magnitude_ratio (P A B : Point) (r : ℝ) : Prop :=
  let AP := vector A P
  let PB := vector P B
  (AP.x, AP.y) = (r * PB.x, r * PB.y)

theorem find_point_P (P : Point) : 
  magnitude_ratio P A B (4/3) → (P.x = 10 ∧ P.y = -21) :=
sorry

end find_point_P

namespace fill_tank_with_leak

import Mathlib

theorem fill_tank_with_leak (A L : ℝ) (h1 : A = 1 / 6) (h2 : L = 1 / 18) : (1 / (A - L)) = 9 :=
by
  sorry

end fill_tank_with_leak

namespace inequality_proof

import Mathlib

variable (a b c d : ℝ)

theorem inequality_proof (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (1 / (1 / a + 1 / b)) + (1 / (1 / c + 1 / d)) ≤ (1 / (1 / (a + c) + 1 / (b + d))) :=
by
  sorry

end inequality_proof

namespace total_cost_is_18

import Mathlib

-- Definitions based on the conditions
def cost_soda : ℕ := 1
def cost_3_sodas := 3 * cost_soda
def cost_soup := cost_3_sodas
def cost_2_soups := 2 * cost_soup
def cost_sandwich := 3 * cost_soup
def total_cost := cost_3_sodas + cost_2_soups + cost_sandwich

-- The proof statement
theorem total_cost_is_18 : total_cost = 18 := by
  -- proof will go here
  sorry

end total_cost_is_18

namespace triangle_inequality

import Mathlib

theorem triangle_inequality
  (a b c : ℝ)
  (habc : ¬(a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a)) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := 
by {
  sorry
}

end triangle_inequality

namespace valentines_given

import Mathlib

theorem valentines_given (x y : ℕ) (h : x * y = x + y + 40) : x * y = 84 :=
by
  -- solving for x, y based on the factors of 41
  sorry

end valentines_given

namespace find_lower_rate

import Mathlib

-- Definitions
def total_investment : ℝ := 20000
def total_interest : ℝ := 1440
def higher_rate : ℝ := 0.09
def fraction_higher : ℝ := 0.55

-- The amount invested at the higher rate
def x := fraction_higher * total_investment
-- The amount invested at the lower rate
def y := total_investment - x

-- The interest contributions
def interest_higher := x * higher_rate
def interest_lower (r : ℝ) := y * r

-- The equation we need to solve to find the lower interest rate
theorem find_lower_rate (r : ℝ) : interest_higher + interest_lower r = total_interest → r = 0.05 :=
by
  sorry

end find_lower_rate

namespace ceil_of_neg_frac_squared

import Mathlib

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared

namespace convex_parallelogram_faces_1992

import Mathlib

theorem convex_parallelogram_faces_1992 (n : ℕ) (h : n > 0) : (n * (n - 1) ≠ 1992) := 
by
  sorry

end convex_parallelogram_faces_1992

namespace integer_a_conditions

import Mathlib

theorem integer_a_conditions (a : ℤ) :
  (∃ (x y : ℕ), x ≠ y ∧ (a * x * y + 1) ∣ (a * x^2 + 1) ^ 2) → a ≥ -1 :=
sorry

end integer_a_conditions

namespace number_of_proper_subsets_of_P

import Mathlib

theorem number_of_proper_subsets_of_P (P : Set ℝ) (hP : P = {x | x^2 = 1}) : 
  (∃ n, n = 2 ∧ ∃ k, k = 2 ^ n - 1 ∧ k = 3) :=
by
  sorry

end number_of_proper_subsets_of_P

namespace projectile_height_reaches_49

import Mathlib

theorem projectile_height_reaches_49 (t : ℝ) :
  (∃ t : ℝ, 49 = -20 * t^2 + 100 * t) → t = 0.7 :=
by
  sorry

end projectile_height_reaches_49

namespace correct_operation

import Mathlib

theorem correct_operation (x y a b : ℝ) :
  (-2 * x) * (3 * y) = -6 * x * y :=
by
  sorry

end correct_operation

namespace initial_salt_percentage

import Mathlib

theorem initial_salt_percentage (P : ℕ) : 
  let initial_solution := 100 
  let added_salt := 20 
  let final_solution := initial_solution + added_salt 
  (P / 100) * initial_solution + added_salt = (25 / 100) * final_solution → 
  P = 10 := 
by
  sorry

end initial_salt_percentage

namespace gcd_2023_1991

import Mathlib

theorem gcd_2023_1991 : Nat.gcd 2023 1991 = 1 :=
by
  sorry

end gcd_2023_1991

namespace max_area_trapezoid

import Mathlib

theorem max_area_trapezoid :
  ∀ {AB CD : ℝ}, 
    AB = 6 → CD = 14 → 
    (∃ (r1 r2 : ℝ), r1 = AB / 2 ∧ r2 = CD / 2 ∧ r1 + r2 = 10) → 
    (1 / 2 * (AB + CD) * 10 = 100) :=
by
  intros AB CD hAB hCD hExist
  sorry

end max_area_trapezoid

namespace smaller_balloon_radius_is_correct

import Mathlib

-- Condition: original balloon radius
def original_balloon_radius : ℝ := 2

-- Condition: number of smaller balloons
def num_smaller_balloons : ℕ := 64

-- Question (to be proved): Radius of each smaller balloon
theorem smaller_balloon_radius_is_correct :
  ∃ r : ℝ, (4/3) * Real.pi * (original_balloon_radius^3) = num_smaller_balloons * (4/3) * Real.pi * (r^3) ∧ r = 1/2 := 
by {
  sorry
}

end smaller_balloon_radius_is_correct

namespace other_asymptote

import Mathlib

-- Define the conditions
def C1 := ∀ x y, y = -2 * x
def C2 := ∀ x, x = -3

-- Formulate the problem
theorem other_asymptote :
  (∃ y m b, y = m * x + b ∧ m = 2 ∧ b = 12) :=
by
  sorry

end other_asymptote

namespace ratio_of_pieces

import Mathlib

def total_length (len: ℕ) := len = 35
def longer_piece (len: ℕ) := len = 20

theorem ratio_of_pieces (shorter len_shorter : ℕ) : 
  total_length 35 →
  longer_piece 20 →
  shorter = 35 - 20 →
  len_shorter = 15 →
  (20:ℚ) / (len_shorter:ℚ) = (4:ℚ) / (3:ℚ) :=
by
  sorry

end ratio_of_pieces

namespace polynomial_characterization

import Mathlib

noncomputable def homogeneous_polynomial (P : ℝ → ℝ → ℝ) (n : ℕ) :=
  ∀ t x y : ℝ, P (t * x) (t * y) = t^n * P x y

def polynomial_condition (P : ℝ → ℝ → ℝ) :=
  ∀ a b c : ℝ, P (a + b) c + P (b + c) a + P (c + a) b = 0

def P_value (P : ℝ → ℝ → ℝ) :=
  P 1 0 = 1

theorem polynomial_characterization (P : ℝ → ℝ → ℝ) (n : ℕ) :
  homogeneous_polynomial P n →
  polynomial_condition P →
  P_value P →
  ∃ A : ℝ → ℝ → ℝ, ∀ x y : ℝ, P x y = (x + y)^(n - 1) * (x - 2 * y) :=
by
  sorry

end polynomial_characterization

namespace triangle_area

import Mathlib

variable (a b c : ℕ)
variable (s : ℕ := 21)
variable (area : ℕ := 84)

theorem triangle_area 
(h1 : c = a + b - 12) 
(h2 : (a + b + c) / 2 = s) 
(h3 : c - a = 2) 
: (21 * (21 - a) * (21 - b) * (21 - c)).sqrt = area := 
sorry

end triangle_area

namespace symmetric_points_power

import Mathlib

variables (m n : ℝ)

def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetric_points_power 
  (h : symmetric_y_axis (m, 3) (4, n)) : 
  (m + n) ^ 2023 = -1 :=
by 
  sorry

end symmetric_points_power

namespace janet_needs_9_dog_collars

import Mathlib

variable (D : ℕ)

theorem janet_needs_9_dog_collars (h1 : ∀ d : ℕ, d = 18)
  (h2 : ∀ c : ℕ, c = 10)
  (h3 : (18 * D) + (3 * 10) = 192) :
  D = 9 :=
by
  sorry

end janet_needs_9_dog_collars

namespace pumps_time_to_empty_pool

import Mathlib

theorem pumps_time_to_empty_pool :
  (1 / (1 / 6 + 1 / 9) * 60) = 216 :=
by
  norm_num
  sorry

end pumps_time_to_empty_pool

namespace evaluate_expression

import Mathlib

noncomputable def expression (a b : ℕ) := (a + b)^2 - (a - b)^2

theorem evaluate_expression:
  expression (5^500) (6^501) = 24 * 30^500 := by
sorry

end evaluate_expression

namespace profit_percent_is_25

import Mathlib

-- Define the cost price (CP) and selling price (SP) based on the given ratio.
def CP (x : ℝ) := 4 * x
def SP (x : ℝ) := 5 * x

-- Calculate the profit percent based on the given conditions.
noncomputable def profitPercent (x : ℝ) := ((SP x - CP x) / CP x) * 100

-- Prove that the profit percent is 25% given the ratio of CP to SP is 4:5.
theorem profit_percent_is_25 (x : ℝ) : profitPercent x = 25 := by
  sorry

end profit_percent_is_25

namespace twelve_edge_cubes_painted_faces

import Mathlib

theorem twelve_edge_cubes_painted_faces :
  let painted_faces_per_edge_cube := 2
  let num_edge_cubes := 12
  painted_faces_per_edge_cube * num_edge_cubes = 24 :=
by
  sorry

end twelve_edge_cubes_painted_faces

namespace area_of_quadrilateral

import Mathlib

theorem area_of_quadrilateral (A B C : ℝ) (triangle1 triangle2 triangle3 quadrilateral : ℝ)
  (hA : A = 5) (hB : B = 9) (hC : C = 9)
  (h_sum : quadrilateral = triangle1 + triangle2 + triangle3)
  (h1 : triangle1 = A)
  (h2 : triangle2 = B)
  (h3 : triangle3 = C) :
  quadrilateral = 40 :=
by
  sorry

end area_of_quadrilateral

namespace factorization_correct

import Mathlib

theorem factorization_correct (x : ℝ) : 
  (hxA : x^2 + 2*x + 1 ≠ x*(x + 2) + 1) → 
  (hxB : x^2 + 2*x + 1 ≠ (x + 1)*(x - 1)) → 
  (hxC : x^2 + x ≠ (x + 1/2)^2 - 1/4) →
  x^2 + x = x * (x + 1) := 
by sorry

end factorization_correct

namespace total_amount_is_2500

import Mathlib

noncomputable def total_amount_divided (P1 : ℝ) (annual_income : ℝ) : ℝ :=
  let P2 := 2500 - P1
  let income_from_P1 := (5 / 100) * P1
  let income_from_P2 := (6 / 100) * P2
  income_from_P1 + income_from_P2

theorem total_amount_is_2500 : 
  (total_amount_divided 2000 130) = 130 :=
by
  sorry

end total_amount_is_2500

namespace combined_marbles

import Mathlib

def Rhonda_marbles : ℕ := 80
def Amon_marbles : ℕ := Rhonda_marbles + 55

theorem combined_marbles : Amon_marbles + Rhonda_marbles = 215 :=
by
  sorry

end combined_marbles

namespace opposite_of_neg_six

import Mathlib

theorem opposite_of_neg_six : -(-6) = 6 := 
by
  sorry

end opposite_of_neg_six

namespace unique_five_digit_integers

import Mathlib

-- Define the problem conditions
def digits := [2, 2, 3, 9, 9]
def total_spots := 5
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Compute the number of five-digit integers that can be formed
noncomputable def num_unique_permutations : Nat :=
  factorial total_spots / (factorial 2 * factorial 1 * factorial 2)

-- Proof statement
theorem unique_five_digit_integers : num_unique_permutations = 30 := by
  sorry

end unique_five_digit_integers

namespace intersection_M_N

import Mathlib

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x <2} := by
  sorry

end intersection_M_N

namespace simplify_expression

import Mathlib

variable (a : ℝ)

theorem simplify_expression (h1 : 0 < a ∨ a < 0) : a * Real.sqrt (-(1 / a)) = -Real.sqrt (-a) :=
sorry

end simplify_expression

namespace product_range

import Mathlib

theorem product_range (m b : ℚ) (h₀ : m = 3 / 4) (h₁ : b = 6 / 5) : 0 < m * b ∧ m * b < 1 :=
by
  sorry

end product_range

namespace olivia_savings

import Mathlib

noncomputable def compound_amount 
  (P : ℝ) -- Initial principal
  (r : ℝ) -- Annual interest rate
  (n : ℕ) -- Number of times interest is compounded per year
  (t : ℕ) -- Number of years
  : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem olivia_savings :
  compound_amount 2500 0.045 2 21 = 5077.14 :=
by
  sorry

end olivia_savings

namespace gerald_added_crayons

import Mathlib

namespace Proof

variable (original_crayons : ℕ) (total_crayons : ℕ)

theorem gerald_added_crayons (h1 : original_crayons = 7) (h2 : total_crayons = 13) : 
  total_crayons - original_crayons = 6 := by
  sorry

end Proof

end gerald_added_crayons

namespace condition_for_positive_expression

import Mathlib

theorem condition_for_positive_expression (a b c : ℝ) :
  (∀ x y : ℝ, x^2 + x * y + y^2 + a * x + b * y + c > 0) ↔ a^2 - a * b + b^2 < 3 * c :=
by
  -- Proof should be provided here
  sorry

end condition_for_positive_expression

namespace black_region_area_is_correct

import Mathlib

noncomputable def area_of_black_region : ℕ :=
  let area_large_square := 10 * 10
  let area_first_smaller_square := 4 * 4
  let area_second_smaller_square := 2 * 2
  area_large_square - (area_first_smaller_square + area_second_smaller_square)

theorem black_region_area_is_correct :
  area_of_black_region = 80 :=
by
  sorry

end black_region_area_is_correct

namespace min_tip_percentage

import Mathlib

noncomputable def meal_cost : ℝ := 37.25
noncomputable def total_paid : ℝ := 40.975
noncomputable def tip_percentage (P : ℝ) : Prop := P > 0 ∧ P < 15 ∧ (meal_cost + (P/100) * meal_cost = total_paid)

theorem min_tip_percentage : ∃ P : ℝ, tip_percentage P ∧ P = 10 := by
  sorry

end min_tip_percentage

namespace bead_necklaces_sold

import Mathlib

def cost_per_necklace : ℕ := 7
def total_earnings : ℕ := 70
def gemstone_necklaces_sold : ℕ := 7

theorem bead_necklaces_sold (B : ℕ) 
  (h1 : total_earnings = cost_per_necklace * (B + gemstone_necklaces_sold))  :
  B = 3 :=
by {
  sorry
}

end bead_necklaces_sold

namespace square_area_from_diagonal

import Mathlib

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  ∃ (A : ℝ), A = 72 :=
by
  sorry

end square_area_from_diagonal

namespace sum_of_n_terms

import Mathlib

noncomputable def S : ℕ → ℕ :=
sorry -- We define S, but its exact form is not used in the statement directly

noncomputable def a : ℕ → ℕ := 
sorry -- We define a, but its exact form is not used in the statement directly

-- Conditions
axiom S3_eq : S 3 = 1
axiom a_rec : ∀ n : ℕ, 0 < n → a (n + 3) = 2 * (a n)

-- Proof problem
theorem sum_of_n_terms : S 2019 = 2^673 - 1 :=
sorry

end sum_of_n_terms

namespace steve_halfway_time_longer

import Mathlib

theorem steve_halfway_time_longer :
  ∀ (Td: ℝ) (Ts: ℝ),
  Td = 33 →
  Ts = 2 * Td →
  (Ts / 2) - (Td / 2) = 16.5 :=
by
  intros Td Ts hTd hTs
  rw [hTd, hTs]
  sorry

end steve_halfway_time_longer

namespace deepak_profit_share

import Mathlib

theorem deepak_profit_share (anand_investment : ℕ) (deepak_investment : ℕ) (total_profit : ℕ) 
  (h₁ : anand_investment = 22500) 
  (h₂ : deepak_investment = 35000) 
  (h₃ : total_profit = 13800) : 
  (14 * total_profit / (9 + 14)) = 8400 := 
by
  sorry

end deepak_profit_share

namespace find_angle

import Mathlib.Data.Real.Basic

-- Given definitions:
def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

-- Condition:
def condition (α : ℝ) : Prop :=
  supplement α = 3 * complement α + 10

-- Statement to prove:
theorem find_angle (α : ℝ) (h : condition α) : α = 50 :=
sorry

end find_angle

namespace variance_of_data_set

import Mathlib

theorem variance_of_data_set (a : ℝ) (ha : (1 + a + 3 + 6 + 7) / 5 = 4) : 
  (1 / 5) * ((1 - 4)^2 + (a - 4)^2 + (3 - 4)^2 + (6 - 4)^2 + (7 - 4)^2) = 24 / 5 :=
by
  sorry

end variance_of_data_set

namespace condition_for_y_exists

import Mathlib

theorem condition_for_y_exists (n : ℕ) (hn : n ≥ 2) (x y : Fin (n + 1) → ℝ)
  (z : Fin (n + 1) → ℂ)
  (hz : ∀ k, z k = x k + Complex.I * y k)
  (heq : z 0 ^ 2 = ∑ k in Finset.range n, z (k + 1) ^ 2) :
  x 0 ^ 2 ≤ ∑ k in Finset.range n, x (k + 1) ^ 2 :=
by
  sorry

end condition_for_y_exists

namespace units_digit_of_k_squared_plus_2_to_the_k

import Mathlib

def k : ℕ := 2021^2 + 2^2021 + 3

theorem units_digit_of_k_squared_plus_2_to_the_k :
    (k^2 + 2^k) % 10 = 0 :=
by
    sorry

end units_digit_of_k_squared_plus_2_to_the_k

namespace probability_red_or_white_is_11_over_13

import Mathlib

-- Given data
def total_marbles : ℕ := 60
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - blue_marbles - red_marbles

def blue_size : ℕ := 2
def red_size : ℕ := 1
def white_size : ℕ := 1

-- Total size value of all marbles
def total_size_value : ℕ := (blue_size * blue_marbles) + (red_size * red_marbles) + (white_size * white_marbles)

-- Probability of selecting a red or white marble
def probability_red_or_white : ℚ := (red_size * red_marbles + white_size * white_marbles) / total_size_value

-- Theorem to prove
theorem probability_red_or_white_is_11_over_13 : probability_red_or_white = 11 / 13 :=
by sorry

end probability_red_or_white_is_11_over_13

namespace triangle_side_relation

import Mathlib

theorem triangle_side_relation 
  (a b c : ℝ) 
  (A : ℝ) 
  (h : b^2 + c^2 = a * ((√3 / 3) * b * c + a)) : 
  a = 2 * √3 * Real.cos A := 
sorry

end triangle_side_relation

namespace total_spent_is_195

import Mathlib

def hoodie_cost : ℝ := 80
def flashlight_cost : ℝ := 0.2 * hoodie_cost
def boots_original_cost : ℝ := 110
def boots_discount : ℝ := 0.1
def boots_discounted_cost : ℝ := boots_original_cost * (1 - boots_discount)
def total_cost : ℝ := hoodie_cost + flashlight_cost + boots_discounted_cost

theorem total_spent_is_195 : total_cost = 195 := by
  sorry

end total_spent_is_195

namespace triangle_inequality

import Mathlib

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by
  sorry

end triangle_inequality

namespace polynomial_simplification

import Mathlib

theorem polynomial_simplification (x : ℤ) :
  (5 * x ^ 12 + 8 * x ^ 11 + 10 * x ^ 9) + (3 * x ^ 13 + 2 * x ^ 12 + x ^ 11 + 6 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9) =
  3 * x ^ 13 + 7 * x ^ 12 + 9 * x ^ 11 + 16 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9 :=
by
  sorry

end polynomial_simplification

namespace 

import Mathlib

theorem max_eq_zero_max_two_solutions {a b : Fin 10 → ℝ}
  (h : ∀ i, a i ≠ 0) : 
  ∃ (solution_count : ℕ), solution_count <= 2 ∧
  ∃ (solutions : Fin solution_count → ℝ), 
    ∀ (x : ℝ), (∀ i, max (a i * x + b i) = 0) ↔ ∃ j, x = solutions j := sorry

end 

namespace natural_number_squares

import Mathlib

theorem natural_number_squares (n : ℕ) (h : ∃ k : ℕ, n^2 + 492 = k^2) :
    n = 122 ∨ n = 38 :=
by
  sorry

end natural_number_squares

namespace evaluate_expression

import Mathlib

-- Given conditions
def a : ℕ := 3
def b : ℕ := 2

-- Proof problem statement
theorem evaluate_expression : (1 / 3 : ℝ) ^ (b - a) = 3 := sorry

end evaluate_expression

namespace initial_girls_count

import Mathlib

variable (p : ℝ) (g : ℝ) (b : ℝ) (initial_girls : ℝ)

-- Conditions
def initial_percentage_of_girls (p g : ℝ) : Prop := g / p = 0.6
def final_percentage_of_girls (g : ℝ) (p : ℝ) : Prop := (g - 3) / p = 0.5

-- Statement only (no proof)
theorem initial_girls_count (p : ℝ) (h1 : initial_percentage_of_girls p (0.6 * p)) (h2 : final_percentage_of_girls (0.6 * p) p) :
  initial_girls = 18 :=
by
  sorry

end initial_girls_count

namespace polynomial_coefficients

import Mathlib

theorem polynomial_coefficients (a : Fin 10 → ℤ) :
  (1 - X) ^ 9 = ∑ i in Finset.range 10, (a i) * X ^ i →
  a 0 = 1 ∧
  a 1 + a 3 + a 5 + a 7 + a 9 = -256 ∧
  (2 : ℤ) * a 1 + (2 : ℤ)^2 * a 2 + (2 : ℤ)^3 * a 3 + (2 : ℤ)^4 * a 4 + (2 : ℤ)^5 * a 5 + 
  (2 : ℤ)^6 * a 6 + (2 : ℤ)^7 * a 7 + (2 : ℤ)^8 * a 8 + (2 : ℤ)^9 * a 9 = -2 := by
  sorry

end polynomial_coefficients

namespace john_made_money

import Mathlib

theorem john_made_money 
  (repair_cost : ℕ := 20000) 
  (discount_percentage : ℕ := 20) 
  (prize_money : ℕ := 70000) 
  (keep_percentage : ℕ := 90) : 
  (prize_money * keep_percentage / 100) - (repair_cost - (repair_cost * discount_percentage / 100)) = 47000 := 
by 
  sorry

end john_made_money

namespace houses_with_neither

import Mathlib

theorem houses_with_neither (T G P GP N : ℕ) (hT : T = 65) (hG : G = 50) (hP : P = 40) (hGP : GP = 35) (hN : N = T - (G + P - GP)) :
  N = 10 :=
by
  rw [hT, hG, hP, hGP] at hN
  exact hN

-- Proof is not required, just the statement is enough.

end houses_with_neither

namespace abs_difference

import Mathlib

theorem abs_difference (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 8) : 
  |a - b| = 2 * Real.sqrt 10 :=
by
  sorry

end abs_difference

namespace problem_solution

import Mathlib

def eq_A (x : ℝ) : Prop := 2 * x = 7
def eq_B (x y : ℝ) : Prop := x^2 + y = 5
def eq_C (x : ℝ) : Prop := x = 1 / x + 1
def eq_D (x : ℝ) : Prop := x^2 + x = 4

def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ a * x^2 + b * x + c = 0

theorem problem_solution : is_quadratic eq_D := by
  sorry

end problem_solution

namespace calculate_expression

import Mathlib

theorem calculate_expression :
  ((2000000000000 - 1234567890123) * 3 = 2296296329631) :=
by 
  sorry

end calculate_expression

namespace quadratic_to_standard_form_div

import Mathlib

theorem quadratic_to_standard_form_div (b c : ℤ)
  (h : ∀ x : ℤ, x^2 - 2100 * x - 8400 = (x + b)^2 + c) :
  c / b = 1058 :=
sorry

end quadratic_to_standard_form_div

namespace problem

import Mathlib

variable (α : ℝ)

def setA : Set ℝ := {Real.sin α, Real.cos α, 1}
def setB : Set ℝ := {Real.sin α ^ 2, Real.sin α + Real.cos α, 0}
theorem problem (h : setA α = setB α) : Real.sin α ^ 2009 + Real.cos α ^ 2009 = -1 := 
by 
  sorry

end problem

namespace geometric_arithmetic_seq_unique_ratio

import Mathlib

variable (d : ℚ) (q : ℚ) (k : ℤ)
variable (h_d_nonzero : d ≠ 0)
variable (h_q_pos : 0 < q) (h_q_lt_one : q < 1)
variable (h_integer : 14 / (1 + q + q^2) = k)

theorem geometric_arithmetic_seq_unique_ratio :
  q = 1 / 2 :=
by
  sorry

end geometric_arithmetic_seq_unique_ratio

namespace compare_fractions

import Mathlib

theorem compare_fractions (a : ℝ) : 
  (a = 0 → (1 / (1 - a)) = (1 + a)) ∧ 
  (0 < a ∧ a < 1 → (1 / (1 - a)) > (1 + a)) ∧ 
  (a > 1 → (1 / (1 - a)) < (1 + a)) := by
  sorry

end compare_fractions

namespace exponent_division

import Mathlib

variable (a : ℝ) (m n : ℝ)
-- Conditions
def condition1 : Prop := a^m = 2
def condition2 : Prop := a^n = 16

-- Theorem Statement
theorem exponent_division (h1 : condition1 a m) (h2 : condition2 a n) : a^(m - n) = 1 / 8 := by
  sorry

end exponent_division

namespace total_messages

import Mathlib

theorem total_messages (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  sorry

end total_messages

namespace sequence_pattern

import Mathlib

theorem sequence_pattern (a b c d e f : ℕ) 
  (h1 : a + b = 12)
  (h2 : 8 + 9 = 16)
  (h3 : 5 + 6 = 10)
  (h4 : 7 + 8 = 14)
  (h5 : 3 + 3 = 5) : 
  ∀ x, ∃ y, x + y = 2 * x := by
  intros x
  use 0
  sorry

end sequence_pattern

namespace percentage_increase_second_movie

import Mathlib

def length_first_movie : ℕ := 2
def total_length_marathon : ℕ := 9
def length_last_movie (F S : ℕ) := S + F - 1

theorem percentage_increase_second_movie :
  ∀ (S : ℕ), 
  length_first_movie + S + length_last_movie length_first_movie S = total_length_marathon →
  ((S - length_first_movie) * 100) / length_first_movie = 50 :=
by
  sorry

end percentage_increase_second_movie

namespace euler_totient_bound

import Mathlib

theorem euler_totient_bound (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : (Nat.totient^[k]) n = 1) :
  n ≤ 3^k :=
sorry

end euler_totient_bound

namespace optimal_years_minimize_cost

import Mathlib

noncomputable def initial_cost : ℝ := 150000
noncomputable def annual_expenses (n : ℕ) : ℝ := 15000 * n
noncomputable def maintenance_cost (n : ℕ) : ℝ := (n * (3000 + 3000 * n)) / 2
noncomputable def total_cost (n : ℕ) : ℝ := initial_cost + annual_expenses n + maintenance_cost n
noncomputable def average_annual_cost (n : ℕ) : ℝ := total_cost n / n

theorem optimal_years_minimize_cost : ∀ n : ℕ, n = 10 ↔ average_annual_cost 10 ≤ average_annual_cost n :=
by sorry

end optimal_years_minimize_cost

namespace find_pairs

import Mathlib

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem find_pairs (a n : ℕ) (h1 : a ≥ n) (h2 : is_power_of_two ((a + 1)^n + a - 1)) :
  (a = 4 ∧ n = 3) ∨ (∃ k : ℕ, a = 2^k ∧ n = 1) :=
by
  sorry

end find_pairs

namespace stools_chopped_up

import Mathlib

variable (chairs tables stools : ℕ)
variable (sticks_per_chair sticks_per_table sticks_per_stool : ℕ)
variable (sticks_per_hour hours total_sticks_from_chairs tables_sticks required_sticks : ℕ)

theorem stools_chopped_up (h1 : sticks_per_chair = 6)
                         (h2 : sticks_per_table = 9)
                         (h3 : sticks_per_stool = 2)
                         (h4 : sticks_per_hour = 5)
                         (h5 : chairs = 18)
                         (h6 : tables = 6)
                         (h7 : hours = 34)
                         (h8 : total_sticks_from_chairs = chairs * sticks_per_chair)
                         (h9 : tables_sticks = tables * sticks_per_table)
                         (h10 : required_sticks = hours * sticks_per_hour)
                         (h11 : total_sticks_from_chairs + tables_sticks = 162) :
                         stools = 4 := by
  sorry

end stools_chopped_up

namespace tan_15_degrees_theta_range_valid_max_f_value

import Mathlib

-- Define the dot product condition
def dot_product_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  AB * BC * (Real.cos θ) = 6

-- Define the sine inequality condition
def sine_inequality_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  6 * (2 - Real.sqrt 3) ≤ AB * BC * (Real.sin θ) ∧ AB * BC * (Real.sin θ) ≤ 6 * Real.sqrt 3

-- Define the maximum value function
noncomputable def f (θ : ℝ) : ℝ :=
  (1 - Real.sqrt 2 * Real.cos (2 * θ - Real.pi / 4)) / (Real.sin θ)

-- Proof that tan 15 degrees is equal to 2 - sqrt(3)
theorem tan_15_degrees : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 := 
  by sorry

-- Proof for the range of θ
theorem theta_range_valid (AB BC : ℝ) (θ : ℝ) 
  (h1 : dot_product_condition AB BC θ)
  (h2 : sine_inequality_condition AB BC θ) : 
  (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3) := 
  by sorry

-- Proof for the maximum value of the function
theorem max_f_value (θ : ℝ) 
  (h : (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3)) : 
  f θ ≤ Real.sqrt 3 - 1 := 
  by sorry

end tan_15_degrees_theta_range_valid_max_f_value

namespace inequality_x_y

import Mathlib

theorem inequality_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + x * y = 3) : x + y ≥ 2 := 
  sorry

end inequality_x_y

namespace simplify_and_evaluate_expr

import Mathlib

theorem simplify_and_evaluate_expr (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3) - x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  rw [h]
  sorry

end simplify_and_evaluate_expr

namespace total_weight_correct_weight_difference_correct

import Mathlib

variables (baskets_of_apples baskets_of_pears : ℕ) (kg_per_basket_of_apples kg_per_basket_of_pears : ℕ)

def total_weight_apples_ppears (baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears : ℕ) : ℕ :=
  (baskets_of_apples * kg_per_basket_of_apples) + (baskets_of_pears * kg_per_basket_of_pears)

def weight_difference_pears_apples (baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears : ℕ) : ℕ :=
  (baskets_of_pears * kg_per_basket_of_pears) - (baskets_of_apples * kg_per_basket_of_apples)

theorem total_weight_correct (h_apples: baskets_of_apples = 120) (h_pears: baskets_of_pears = 130) (h_kg_apples: kg_per_basket_of_apples = 40) (h_kg_pears: kg_per_basket_of_pears = 50) : 
  total_weight_apples_ppears baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears = 11300 :=
by
  rw [h_apples, h_pears, h_kg_apples, h_kg_pears]
  sorry

theorem weight_difference_correct (h_apples: baskets_of_apples = 120) (h_pears: baskets_of_pears = 130) (h_kg_apples: kg_per_basket_of_apples = 40) (h_kg_pears: kg_per_basket_of_pears = 50) : 
  weight_difference_pears_apples baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears = 1700 :=
by
  rw [h_apples, h_pears, h_kg_apples, h_kg_pears]
  sorry

end total_weight_correct_weight_difference_correct

namespace expand_polynomial

import Mathlib

variable {x y z : ℝ}

theorem expand_polynomial : (x + 10 * z + 5) * (2 * y + 15) = 2 * x * y + 20 * y * z + 15 * x + 10 * y + 150 * z + 75 :=
  sorry

end expand_polynomial

namespace quadrant_of_P

import Mathlib

theorem quadrant_of_P (m n : ℝ) (h1 : m * n > 0) (h2 : m + n < 0) : (m < 0 ∧ n < 0) :=
by
  sorry

end quadrant_of_P

namespace Julie_can_print_complete_newspapers

import Mathlib

def sheets_in_box_A : ℕ := 4 * 200
def sheets_in_box_B : ℕ := 3 * 350
def total_sheets : ℕ := sheets_in_box_A + sheets_in_box_B

def front_section_sheets : ℕ := 10
def sports_section_sheets : ℕ := 7
def arts_section_sheets : ℕ := 5
def events_section_sheets : ℕ := 3

def sheets_per_newspaper : ℕ := front_section_sheets + sports_section_sheets + arts_section_sheets + events_section_sheets

theorem Julie_can_print_complete_newspapers : total_sheets / sheets_per_newspaper = 74 := by
  sorry

end Julie_can_print_complete_newspapers

namespace B_share_correct

import Mathlib

noncomputable def total_share : ℕ := 120
noncomputable def B_share : ℕ := 20
noncomputable def A_share (x : ℕ) : ℕ := x + 20
noncomputable def C_share (x : ℕ) : ℕ := x + 40

theorem B_share_correct : ∃ x : ℕ, total_share = (A_share x) + x + (C_share x) ∧ x = B_share := by
  sorry

end B_share_correct

namespace radio_lowest_price_rank

import Mathlib

-- Definitions based on the conditions
def total_items : ℕ := 38
def radio_highest_rank : ℕ := 16

-- The theorem statement
theorem radio_lowest_price_rank : (total_items - (radio_highest_rank - 1)) = 24 := by
  sorry

end radio_lowest_price_rank

namespace quadratic_function_conditions

import Mathlib

noncomputable def quadratic_function_example (x : ℝ) : ℝ :=
  -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_conditions :
  quadratic_function_example 1 = 0 ∧
  quadratic_function_example 5 = 0 ∧
  quadratic_function_example 3 = 10 :=
by
  sorry

end quadratic_function_conditions

namespace part1_part2

import Mathlib

-- Conditions and the equation of the circle
def circleCenterLine (a : ℝ) : Prop := ∃ y, y = a + 2
def circleRadius : ℝ := 2
def pointOnCircle (A : ℝ × ℝ) (a : ℝ) : Prop := (A.1 - a)^2 + (A.2 - (a + 2))^2 = circleRadius^2
def tangentToYAxis (a : ℝ) : Prop := abs a = circleRadius

-- Problem 1: Proving the equation of the circle C
def circleEq (x y a : ℝ) : Prop := (x - a)^2 + (y - (a + 2))^2 = circleRadius^2

theorem part1 (a : ℝ) (h : abs a = circleRadius) (h1 : pointOnCircle (2, 2) a) 
    (h2 : circleCenterLine a) : circleEq 2 0 2 := 
sorry

-- Conditions and the properties for Problem 2
def distSquared (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
def QCondition (Q : ℝ × ℝ) : Prop := 
  distSquared Q (1, 3) - distSquared Q (1, 1) = 32
def onCircle (Q : ℝ × ℝ) (a : ℝ) : Prop := (Q.1 - a)^2 + (Q.2 - (a + 2))^2 = circleRadius^2

-- Problem 2: Proving the range of the abscissa a
theorem part2 (Q : ℝ × ℝ) (a : ℝ) 
    (hQ : QCondition Q) (hCircle : onCircle Q a) : 
    -3 ≤ a ∧ a ≤ 1 := 
sorry

end part1_part2

namespace how_many_trucks

import Mathlib

-- Define the conditions given in the problem
def people_to_lift_car : ℕ := 5
def people_to_lift_truck : ℕ := 2 * people_to_lift_car

-- Set up the problem conditions
def total_people_needed (cars : ℕ) (trucks : ℕ) : ℕ :=
  cars * people_to_lift_car + trucks * people_to_lift_truck

-- Now state the precise theorem we need to prove
theorem how_many_trucks (cars trucks total_people : ℕ) 
  (h1 : cars = 6)
  (h2 : trucks = 3)
  (h3 : total_people = total_people_needed cars trucks) :
  trucks = 3 :=
by
  sorry

end how_many_trucks

namespace fraction_inequality

import Mathlib

theorem fraction_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (1 / a) + (1 / b) ≥ (4 / (a + b)) :=
by 
-- Skipping the proof using 'sorry'
sorry

end fraction_inequality

namespace find_largest_cos_x

import Mathlib

theorem find_largest_cos_x (x y z : ℝ) 
  (h1 : Real.sin x = Real.cot y)
  (h2 : Real.sin y = Real.cot z)
  (h3 : Real.sin z = Real.cot x) :
  Real.cos x ≤ Real.sqrt ((3 - Real.sqrt 5) / 2) := sorry

end find_largest_cos_x

namespace toms_age

import Mathlib

theorem toms_age (T S : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 :=
sorry

end toms_age

namespace no_integer_solution_xyz

import Mathlib

theorem no_integer_solution_xyz : ¬ ∃ (x y z : ℤ),
  x^6 + x^3 + x^3 * y + y = 147^157 ∧
  x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := by
  sorry

end no_integer_solution_xyz

namespace find_max_value

import Mathlib

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  2 * x * y * Real.sqrt 3 + 3 * y * z * Real.sqrt 2 + 3 * z * x

theorem find_max_value (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z)
  (h₃ : x^2 + y^2 + z^2 = 1) : 
  maximum_value x y z ≤ Real.sqrt 3 := sorry

end find_max_value

namespace sum_odd_numbers_to_2019_is_correct

import Mathlib

-- Define the sequence sum
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Define the specific problem
theorem sum_odd_numbers_to_2019_is_correct : sum_first_n_odd 1010 = 1020100 :=
by
  -- Sorry placeholder for the proof
  sorry

end sum_odd_numbers_to_2019_is_correct

namespace minimum_dwarfs

import Mathlib

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs

namespace 

import Mathlib

theorem range_of_a_minus_b {a b : ℝ} (h1 : -2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) : 
  -4 < a - b ∧ a - b < 2 := 
by
  sorry

end 

namespace propA_propB_relation

import Mathlib

variable (x y : ℤ)

theorem propA_propB_relation :
  (x + y ≠ 5 → x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → x + y ≠ 5) :=
by
  sorry

end propA_propB_relation

namespace distance_correct

import Mathlib

-- Define geometry entities and properties
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Sphere where
  center : Point
  radius : ℝ

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define conditions
def sphere_center : Point := { x := 0, y := 0, z := 0 }
def sphere : Sphere := { center := sphere_center, radius := 5 }
def triangle : Triangle := { a := 13, b := 13, c := 10 }

-- Define the distance calculation
noncomputable def distance_from_sphere_center_to_plane (O : Point) (T : Triangle) : ℝ :=
  let h := 12  -- height calculation based on given triangle sides
  let A := 60  -- area of the triangle
  let s := 18  -- semiperimeter
  let r := 10 / 3  -- inradius calculation
  let x := 5 * (Real.sqrt 5) / 3  -- final distance calculation
  x

-- Prove the obtained distance matches expected value
theorem distance_correct :
  distance_from_sphere_center_to_plane sphere_center triangle = 5 * (Real.sqrt 5) / 3 :=
by
  sorry

end distance_correct

namespace ticket_1000_wins_probability

import Mathlib

-- Define the total number of tickets
def n_tickets := 1000

-- Define the number of odd tickets
def n_odd_tickets := 500

-- Define the number of relevant tickets (ticket 1000 + odd tickets)
def n_relevant_tickets := 501

-- Define the probability that ticket number 1000 wins a prize
def win_probability : ℚ := 1 / n_relevant_tickets

-- State the theorem
theorem ticket_1000_wins_probability : win_probability = 1 / 501 :=
by
  -- The proof would go here
  sorry

end ticket_1000_wins_probability

namespace choir_row_lengths

import Mathlib

theorem choir_row_lengths (x : ℕ) : 
  ((x ∈ [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) ∧ (90 % x = 0)) → (x = 5 ∨ x = 6 ∨ x = 9 ∨ x = 10 ∨ x = 15) :=
by
  intro h
  cases h
  sorry

end choir_row_lengths

namespace final_velocity

import Mathlib

variable (u a t : ℝ)

-- Defining the conditions
def initial_velocity := u = 0
def acceleration := a = 1.2
def time := t = 15

-- Statement of the theorem
theorem final_velocity : initial_velocity u ∧ acceleration a ∧ time t → (u + a * t = 18) := by
  sorry

end final_velocity

namespace sequence_is_increasing

import Mathlib

variable (a_n : ℕ → ℝ)

def sequence_positive_numbers (a_n : ℕ → ℝ) : Prop :=
∀ n, 0 < a_n n

def sequence_condition (a_n : ℕ → ℝ) : Prop :=
∀ n, a_n (n + 1) = 2 * a_n n

theorem sequence_is_increasing 
  (h1 : sequence_positive_numbers a_n) 
  (h2 : sequence_condition a_n) : 
  ∀ n, a_n (n + 1) > a_n n :=
by
  sorry

end sequence_is_increasing

namespace quadrilateral_side_squares_inequality

import Mathlib

theorem quadrilateral_side_squares_inequality :
  ∀ (x1 y1 x2 y2 : ℝ),
    0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ y1 ∧ y1 ≤ 1 ∧
    0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ y2 ∧ y2 ≤ 1 →
    2 ≤ (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ∧ 
          (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ≤ 4 :=
by
  intro x1 y1 x2 y2 h
  sorry

end quadrilateral_side_squares_inequality

namespace largest_n_for_crates

import Mathlib

theorem largest_n_for_crates (total_crates : ℕ) (min_oranges max_oranges : ℕ)
  (h1 : total_crates = 145)
  (h2 : min_oranges = 110)
  (h3 : max_oranges = 140) : 
  ∃ n : ℕ, n = 5 ∧ ∀ k : ℕ, k ≤ max_oranges - min_oranges + 1 → total_crates / k ≤ n :=
  by {
    sorry
  }

end largest_n_for_crates

namespace arithmetic_sequence_sum

import Mathlib

theorem arithmetic_sequence_sum :
  ∀ (a₁ : ℕ) (d : ℕ) (a_n : ℕ) (n : ℕ),
    a₁ = 1 →
    d = 2 →
    a_n = 29 →
    a_n = a₁ + (n - 1) * d →
    (n : ℕ) = 15 →
    (∑ k in Finset.range n, a₁ + k * d) = 225 :=
by
  intros a₁ d a_n n h₁ h_d hₐ h_an h_n
  sorry

end arithmetic_sequence_sum

namespace sum_of_distinct_integers_eq_zero

import Mathlib

theorem sum_of_distinct_integers_eq_zero 
  (a b c d : ℤ) 
  (distinct : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  (prod_eq_25 : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end sum_of_distinct_integers_eq_zero

namespace minimum_value_of_sum

import Mathlib

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
    1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) >= 3 :=
by
  sorry

end minimum_value_of_sum

namespace dino_dolls_count

import Mathlib

theorem dino_dolls_count (T : ℝ) (H : 0.7 * T = 140) : T = 200 :=
sorry

end dino_dolls_count

namespace sellingPrice_is_459

import Mathlib

-- Definitions based on conditions
def costPrice : ℝ := 540
def markupPercentage : ℝ := 0.15
def discountPercentage : ℝ := 0.2608695652173913

-- Calculating the marked price based on the given conditions
def markedPrice (cp : ℝ) (markup : ℝ) : ℝ := cp + (markup * cp)

-- Calculating the discount amount based on the marked price and the discount percentage
def discount (mp : ℝ) (discountPct : ℝ) : ℝ := discountPct * mp

-- Calculating the selling price
def sellingPrice (mp : ℝ) (discountAmt : ℝ) : ℝ := mp - discountAmt

-- Stating the final proof problem
theorem sellingPrice_is_459 :
  sellingPrice (markedPrice costPrice markupPercentage) (discount (markedPrice costPrice markupPercentage) discountPercentage) = 459 :=
by
  sorry

end sellingPrice_is_459

namespace additional_savings_in_cents

import Mathlib

/-
The book has a cover price of $30.
There are two discount methods to compare:
1. First $5 off, then 25% off.
2. First 25% off, then $5 off.
Prove that the difference in final costs (in cents) between these two discount methods is 125 cents.
-/
def book_price : ℝ := 30
def discount_cash : ℝ := 5
def discount_percentage : ℝ := 0.25

def final_price_apply_cash_first (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (price - cash_discount) * (1 - percentage_discount)

def final_price_apply_percentage_first (price : ℝ) (percentage_discount : ℝ) (cash_discount : ℝ) : ℝ :=
  (price * (1 - percentage_discount)) - cash_discount

def savings_comparison (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (final_price_apply_cash_first price cash_discount percentage_discount) - 
  (final_price_apply_percentage_first price percentage_discount cash_discount)

theorem additional_savings_in_cents : 
  savings_comparison book_price discount_cash discount_percentage * 100 = 125 :=
  by sorry

end additional_savings_in_cents

namespace find_length_of_rod

import Mathlib

-- Constants representing the given conditions
def weight_6m_rod : ℝ := 6.1
def length_6m_rod : ℝ := 6
def weight_unknown_rod : ℝ := 12.2

-- Proof statement ensuring the length of the rod that weighs 12.2 kg is 12 meters
theorem find_length_of_rod (L : ℝ) (h : weight_6m_rod / length_6m_rod = weight_unknown_rod / L) : 
  L = 12 := by
  sorry

end find_length_of_rod

namespace maximize_inscribed_polygons

import Mathlib

theorem maximize_inscribed_polygons : 
  ∃ (n : ℕ) (m : ℕ → ℕ), 
    (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → m i < m j) ∧ 
    (∑ i in Finset.range n, m i = 1996) ∧ 
    (n = 61) ∧ 
    (∀ k, 0 ≤ k ∧ k < n → m k = k + 2) :=
by
  sorry

end maximize_inscribed_polygons

namespace mrs_sheridan_final_cats

import Mathlib

def initial_cats : ℝ := 17.5
def given_away_cats : ℝ := 6.2
def returned_cats : ℝ := 2.8
def additional_given_away_cats : ℝ := 1.3

theorem mrs_sheridan_final_cats : 
  initial_cats - given_away_cats + returned_cats - additional_given_away_cats = 12.8 :=
by
  sorry

end mrs_sheridan_final_cats

namespace length_of_platform

import Mathlib

-- Definitions for conditions
def train_length : ℕ := 300
def time_cross_platform : ℕ := 39
def time_cross_signal : ℕ := 12

-- Speed calculation
def train_speed := train_length / time_cross_signal

-- Total distance calculation while crossing the platform
def total_distance := train_speed * time_cross_platform

-- Length of the platform
def platform_length : ℕ := total_distance - train_length

-- Theorem stating the length of the platform
theorem length_of_platform :
  platform_length = 675 := by
  sorry

end length_of_platform

namespace find_remainder

import Mathlib

theorem find_remainder (y : ℕ) (hy : 7 * y % 31 = 1) : (17 + 2 * y) % 31 = 4 :=
sorry

end find_remainder

namespace problem1

import Mathlib

   theorem problem1 : (Real.sqrt (9 / 4) + |2 - Real.sqrt 3| - (64 : ℝ) ^ (1 / 3) + 2⁻¹) = -Real.sqrt 3 :=
   by
     sorry
   
end problem1

namespace consistent_scale

import Mathlib

-- Conditions definitions

def dist_gardensquare_newtonsville : ℕ := 3  -- in inches
def dist_newtonsville_madison : ℕ := 4  -- in inches
def speed_gardensquare_newtonsville : ℕ := 50  -- mph
def time_gardensquare_newtonsville : ℕ := 2  -- hours
def speed_newtonsville_madison : ℕ := 60  -- mph
def time_newtonsville_madison : ℕ := 3  -- hours

-- Actual distances calculated
def actual_distance_gardensquare_newtonsville : ℕ := speed_gardensquare_newtonsville * time_gardensquare_newtonsville
def actual_distance_newtonsville_madison : ℕ := speed_newtonsville_madison * time_newtonsville_madison

-- Prove the scale is consistent across the map
theorem consistent_scale :
  actual_distance_gardensquare_newtonsville / dist_gardensquare_newtonsville =
  actual_distance_newtonsville_madison / dist_newtonsville_madison :=
by
  sorry

end consistent_scale

namespace inverse_value

import Mathlib

def f (x : ℤ) : ℤ := 5 * x ^ 3 - 3

theorem inverse_value : ∀ y, (f y) = 4 → y = 317 :=
by
  intros
  sorry

end inverse_value

namespace find_y

import Mathlib

open Complex

theorem find_y (y : ℝ) (h₁ : (3 : ℂ) + (↑y : ℂ) * I = z₁) 
  (h₂ : (2 : ℂ) - I = z₂) 
  (h₃ : z₁ / z₂ = 1 + I) 
  (h₄ : z₁ = (3 : ℂ) + (↑y : ℂ) * I) 
  (h₅ : z₂ = (2 : ℂ) - I)
  : y = 1 :=
sorry


end find_y

namespace miles_walked_on_Tuesday

import Mathlib

theorem miles_walked_on_Tuesday (monday_miles total_miles : ℕ) (hmonday : monday_miles = 9) (htotal : total_miles = 18) :
  total_miles - monday_miles = 9 :=
by
  sorry

end miles_walked_on_Tuesday

namespace percentage_of_left_handed_women

import Mathlib

variable (x y : Nat) (h_ratio_rh_lh : 3 * x = 1 * x)
variable (h_ratio_men_women : 3 * y = 2 * y)
variable (h_rh_men_max : True)

theorem percentage_of_left_handed_women :
  (x / (4 * x)) * 100 = 25 :=
by sorry

end percentage_of_left_handed_women

namespace find_a_pow_b

import Mathlib

theorem find_a_pow_b (a b : ℝ) (h : (a - 2)^2 + |b + 1| = 0) : a^b = 1 / 2 := 
sorry

end find_a_pow_b

namespace fizz_preference_count

import Mathlib

-- Definitions from conditions
def total_people : ℕ := 500
def fizz_angle : ℕ := 270
def total_angle : ℕ := 360
def fizz_fraction : ℚ := fizz_angle / total_angle

-- The target proof statement
theorem fizz_preference_count (hp : total_people = 500) 
                              (ha : fizz_angle = 270) 
                              (ht : total_angle = 360)
                              (hf : fizz_fraction = 3 / 4) : 
    total_people * fizz_fraction = 375 := by
    sorry

end fizz_preference_count

namespace smallest_next_divisor

import Mathlib

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

noncomputable def has_divisor_323 (n : ℕ) : Prop := 323 ∣ n

theorem smallest_next_divisor (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : has_divisor_323 n) :
  ∃ m : ℕ, m > 323 ∧ m ∣ n ∧ (∀ k : ℕ, k > 323 ∧ k < m → ¬ k ∣ n) ∧ m = 340 :=
sorry

end smallest_next_divisor

namespace smallest_k_for_sixty_four_gt_four_nineteen

import Mathlib

-- Definitions of the conditions
def sixty_four (k : ℕ) : ℕ := 64^k
def four_nineteen : ℕ := 4^19

-- The theorem to prove
theorem smallest_k_for_sixty_four_gt_four_nineteen (k : ℕ) : sixty_four k > four_nineteen ↔ k ≥ 7 := 
by
  sorry

end smallest_k_for_sixty_four_gt_four_nineteen

namespace number_of_members

import Mathlib

theorem number_of_members (n : ℕ) (h : n * n = 2025) : n = 45 :=
sorry

end number_of_members

namespace sum_mod_18

import Mathlib

theorem sum_mod_18 :
  (65 + 66 + 67 + 68 + 69 + 70 + 71 + 72) % 18 = 8 :=
by
  sorry

end sum_mod_18

namespace find_m

import Mathlib

def triangle (x y : ℤ) := x * y + x + y

theorem find_m (m : ℤ) (h : triangle 2 m = -16) : m = -6 :=
by
  sorry

end find_m

namespace number_less_than_one_is_correct

import Mathlib

theorem number_less_than_one_is_correct : (1 - 5 = -4) :=
by
  sorry

end number_less_than_one_is_correct

namespace largest_corner_sum

import Mathlib

-- Define the cube and its properties
structure Cube :=
  (faces : ℕ → ℕ)
  (opposite_faces_sum_to_8 : ∀ i, faces i + faces (7 - i) = 8)

-- Prove that the largest sum of three numbers whose faces meet at one corner is 16
theorem largest_corner_sum (c : Cube) : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  (c.faces i + c.faces j + c.faces k = 16) :=
sorry

end largest_corner_sum

namespace math_proof_problem

import Mathlib

-- Definitions for conditions:
def condition1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 2) = -f x
def condition2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x - 3 / 4) = -f (- (x - 3 / 4))

-- Statements to prove:
def statement1 (f : ℝ → ℝ) : Prop := ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x
def statement2 (f : ℝ → ℝ) : Prop := ∀ x, f (-(3 / 4) - x) = f (-(3 / 4) + x)
def statement3 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def statement4 (f : ℝ → ℝ) : Prop := ¬(∀ x y : ℝ, x < y → f x ≤ f y)

theorem math_proof_problem (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) :
  statement1 f ∧ statement2 f ∧ statement3 f ∧ statement4 f :=
by
  sorry

end math_proof_problem

namespace min_value_square_distance

import Mathlib

theorem min_value_square_distance (x y : ℝ) (h : x^2 + y^2 - 4*x + 2 = 0) : 
  ∃ c, (∀ x y : ℝ, x^2 + y^2 - 4*x + 2 = 0 → x^2 + (y - 2)^2 ≥ c) ∧ c = 2 :=
sorry

end min_value_square_distance

namespace expression_range

import Mathlib

theorem expression_range (a b c x : ℝ) (h : a^2 + b^2 + c^2 ≠ 0) :
  ∃ y : ℝ, y = (a * Real.cos x + b * Real.sin x + c) / (Real.sqrt (a^2 + b^2 + c^2)) 
           ∧ y ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end expression_range

namespace time_to_cross_pole_is_correct

import Mathlib

-- Define the conversion factor to convert km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_per_hr : ℕ) : ℕ := speed_km_per_hr * 1000 / 3600

-- Define the speed of the train in m/s
def train_speed_m_per_s : ℕ := km_per_hr_to_m_per_s 216

-- Define the length of the train
def train_length_m : ℕ := 480

-- Define the time to cross an electric pole
def time_to_cross_pole : ℕ := train_length_m / train_speed_m_per_s

-- Theorem stating that the computed time to cross the pole is 8 seconds
theorem time_to_cross_pole_is_correct :
  time_to_cross_pole = 8 := by
  sorry

end time_to_cross_pole_is_correct

namespace third_discount_is_five_percent

import Mathlib

theorem third_discount_is_five_percent (P F : ℝ) (D : ℝ)
  (h1: P = 9356.725146198829)
  (h2: F = 6400)
  (h3: F = (1 - D / 100) * (0.9 * (0.8 * P))) : 
  D = 5 := by
  sorry

end third_discount_is_five_percent

namespace zoey_holidays_in_a_year

import Mathlib

-- Definitions based on the conditions
def holidays_per_month := 2
def months_in_year := 12

-- Lean statement representing the proof problem
theorem zoey_holidays_in_a_year : (holidays_per_month * months_in_year) = 24 :=
by sorry

end zoey_holidays_in_a_year

namespace range_of_a

import Mathlib

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a

namespace expected_groups

import Mathlib

/-- Given a sequence of k zeros and m ones arranged in random order and divided into 
alternating groups of identical digits, the expected value of the total number of 
groups is 1 + 2 * k * m / (k + m). -/
theorem expected_groups (k m : ℕ) (h_pos : k + m > 0) :
  let total_groups := 1 + 2 * k * m / (k + m)
  total_groups = (1 + 2 * k * m / (k + m)) := by
  sorry

end expected_groups

namespace trajectory_of_A

import Mathlib

def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

def sin_B : ℝ := sorry
def sin_C : ℝ := sorry
def sin_A : ℝ := sorry

axiom sin_relation : sin_B - sin_C = (3/5) * sin_A

theorem trajectory_of_A :
  ∃ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1 ∧ x < -3 :=
sorry

end trajectory_of_A

namespace remainder_y_div_13

import Mathlib

def x (k : ℤ) : ℤ := 159 * k + 37
def y (x : ℤ) : ℤ := 5 * x^2 + 18 * x + 22

theorem remainder_y_div_13 (k : ℤ) : (y (x k)) % 13 = 8 := by
  sorry

end remainder_y_div_13

namespace fewerEmployeesAbroadThanInKorea

import Mathlib

def totalEmployees : Nat := 928
def employeesInKorea : Nat := 713
def employeesAbroad : Nat := totalEmployees - employeesInKorea

theorem fewerEmployeesAbroadThanInKorea :
  employeesInKorea - employeesAbroad = 498 :=
by
  sorry

end fewerEmployeesAbroadThanInKorea

namespace factor_y6_plus_64

import Mathlib

theorem factor_y6_plus_64 : (y^2 + 4) ∣ (y^6 + 64) :=
sorry

end factor_y6_plus_64

namespace sum_in_base7

import Mathlib

-- An encoder function for base 7 integers
def to_base7 (n : ℕ) : string :=
sorry -- skipping the implementation for brevity

-- Decoding the string representation back to a natural number
def from_base7 (s : string) : ℕ :=
sorry -- skipping the implementation for brevity

-- The provided numbers in base 7
def x : ℕ := from_base7 "666"
def y : ℕ := from_base7 "66"
def z : ℕ := from_base7 "6"

-- The expected sum in base 7
def expected_sum : ℕ := from_base7 "104"

-- The statement to be proved
theorem sum_in_base7 : x + y + z = expected_sum :=
sorry -- The proof is omitted

end sum_in_base7

namespace find_15th_term

import Mathlib

-- Define the initial terms and the sequence properties
def first_term := 4
def second_term := 13
def third_term := 22

-- Define the common difference
def common_difference := second_term - first_term

-- Define the nth term formula for arithmetic sequence
def nth_term (a d : ℕ) (n : ℕ) := a + (n - 1) * d

-- State the theorem
theorem find_15th_term : nth_term first_term common_difference 15 = 130 := by
  -- The proof will come here
  sorry

end find_15th_term

namespace perpendicular_bisector_eq

import Mathlib

theorem perpendicular_bisector_eq (A B: (ℝ × ℝ)) (hA: A = (1, 3)) (hB: B = (-5, 1)) :
  ∃ m c, (m = -3) ∧ (c = 4) ∧ (∀ x y, y = m * x + c ↔ 3 * x + y + 4 = 0) := 
by
  sorry

end perpendicular_bisector_eq

namespace interest_rate_A

import Mathlib

-- Given conditions
variables (Principal : ℝ := 4000)
variables (interestRate_C : ℝ := 11.5 / 100)
variables (gain_B : ℝ := 180)
variables (time : ℝ := 3)
variables (interest_from_C : ℝ := Principal * interestRate_C * time)
variables (interest_to_A : ℝ := interest_from_C - gain_B)

-- The proof goal
theorem interest_rate_A (R : ℝ) : 
  1200 = Principal * (R / 100) * time → 
  R = 10 :=
by
  sorry

end interest_rate_A

namespace proof_problem

import Mathlib

variable {a b c : ℝ}

-- Condition: a < 0
variable (ha : a < 0)
-- Condition: b > 0
variable (hb : b > 0)
-- Condition: c > 0
variable (hc : c > 0)
-- Condition: a < b < c
variable (hab : a < b) (hbc : b < c)

-- Proof statement
theorem proof_problem :
  (ab * b < b * c) ∧
  (a * c < b * c) ∧
  (a + c < b + c) ∧
  (c / a < 1) :=
  by
    sorry

end proof_problem

namespace common_difference

import Mathlib

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Conditions
axiom h1 : a 3 + a 7 = 10
axiom h2 : a 8 = 8

-- Statement to prove
theorem common_difference (h : ∀ n, a (n + 1) = a n + d) : d = 1 :=
  sorry

end common_difference

namespace problem_statement

import Mathlib

def U : Set Int := {x | |x| < 5}
def A : Set Int := {-2, 1, 3, 4}
def B : Set Int := {0, 2, 4}

theorem problem_statement : (A ∩ (U \ B)) = {-2, 1, 3} := by
  sorry

end problem_statement

namespace investor_receives_7260

import Mathlib

-- Define the initial conditions
def principal : ℝ := 6000
def annual_rate : ℝ := 0.10
def compoundings_per_year : ℝ := 1
def years : ℝ := 2

-- Define the compound interest formula
noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- State the theorem: The investor will receive $7260 after two years
theorem investor_receives_7260 : compound_interest principal annual_rate compoundings_per_year years = 7260 := by
  sorry

end investor_receives_7260

namespace geese_count

import Mathlib

theorem geese_count (initial : ℕ) (flown_away : ℕ) (left : ℕ) 
  (h₁ : initial = 51) (h₂ : flown_away = 28) : 
  left = initial - flown_away → left = 23 := 
by
  sorry

end geese_count

namespace x_and_y_complete_work_in_12_days

import Mathlib

noncomputable def work_rate_x : ℚ := 1 / 24
noncomputable def work_rate_y : ℚ := 1 / 24
noncomputable def combined_work_rate : ℚ := work_rate_x + work_rate_y

theorem x_and_y_complete_work_in_12_days : (1 / combined_work_rate) = 12 :=
by
  sorry

end x_and_y_complete_work_in_12_days

namespace find_d_value

import Mathlib

open Nat

variable {PA BC PB : ℕ}
noncomputable def d (PA BC PB : ℕ) := PB

theorem find_d_value (h₁ : PA = 6) (h₂ : BC = 9) (h₃ : PB = d PA BC PB) : d PA BC PB = 3 := by
  sorry

end find_d_value

namespace fourth_throw_probability

import Mathlib

-- Define a fair dice where each face has an equal probability.
def fair_dice (n : ℕ) : Prop := (n >= 1 ∧ n <= 6)

-- Define the probability of rolling a 6 on a fair dice.
noncomputable def probability_of_6 : ℝ := 1 / 6

/-- 
  Prove that the probability of getting a "6" on the 4th throw is 1/6 
  given that the dice is fair and the first three throws result in "6".
-/
theorem fourth_throw_probability : 
  (∀ (n1 n2 n3 : ℕ), fair_dice n1 ∧ fair_dice n2 ∧ fair_dice n3 ∧ n1 = 6 ∧ n2 = 6 ∧ n3 = 6) 
  → (probability_of_6 = 1 / 6) :=
by 
  sorry

end fourth_throw_probability

namespace sec_120_eq_neg_2

import Mathlib

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_120_eq_neg_2 : sec 120 = -2 := by
  sorry

end sec_120_eq_neg_2

namespace series_equality

import Mathlib

theorem series_equality :
  (∑ n in Finset.range 200, (-1)^(n+1) * (1:ℚ) / (n+1)) = (∑ n in Finset.range 100, 1 / (100 + n + 1)) :=
by sorry

end series_equality

namespace scientific_notation_example

import Mathlib

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * 10^b

theorem scientific_notation_example : 
  scientific_notation 0.00519 5.19 (-3) :=
by 
  sorry

end scientific_notation_example

namespace car_highway_mileage

import Mathlib

theorem car_highway_mileage :
  (∀ (H : ℝ), 
    (H > 0) → 
    (4 / H + 4 / 20 = (8 / H) * 1.4000000000000001) → 
    (H = 36)) :=
by
  intros H H_pos h_cond
  have : H = 36 := 
    sorry
  exact this

end car_highway_mileage

namespace find_sale_month_4

import Mathlib

-- Definitions based on the given conditions
def avg_sale_per_month : ℕ := 6500
def num_months : ℕ := 6
def sale_month_1 : ℕ := 6435
def sale_month_2 : ℕ := 6927
def sale_month_3 : ℕ := 6855
def sale_month_5 : ℕ := 6562
def sale_month_6 : ℕ := 4991

theorem find_sale_month_4 : 
  (avg_sale_per_month * num_months) - (sale_month_1 + sale_month_2 + sale_month_3 + sale_month_5 + sale_month_6) = 7230 :=
by
  -- The proof will be provided below
  sorry

end find_sale_month_4

namespace total_number_of_people

import Mathlib

variables (A : ℕ) -- Number of adults in the group

-- Conditions
-- Each adult meal costs $8 and the total cost was $72
def cost_per_adult_meal : ℕ := 8
def total_cost : ℕ := 72
def number_of_kids : ℕ := 2

-- Proof problem: Given the conditions, prove the total number of people in the group is 11
theorem total_number_of_people (h : A * cost_per_adult_meal = total_cost) : A + number_of_kids = 11 :=
sorry

end total_number_of_people

namespace line_intersects_plane_at_angle

import Mathlib

def direction_vector : ℝ × ℝ × ℝ := (1, -1, 2)
def normal_vector : ℝ × ℝ × ℝ := (-2, 2, -4)

theorem line_intersects_plane_at_angle :
  let a := direction_vector
  let u := normal_vector
  a ≠ (0, 0, 0) → u ≠ (0, 0, 0) →
  ∃ θ : ℝ, 0 < θ ∧ θ < π :=
by
  sorry

end line_intersects_plane_at_angle

namespace cups_of_flour_put_in

import Mathlib

-- Conditions
def recipeSugar : ℕ := 3
def recipeFlour : ℕ := 10
def neededMoreFlourThanSugar : ℕ := 5

-- Question: How many cups of flour did she put in?
-- Answer: 5 cups of flour
theorem cups_of_flour_put_in : (recipeSugar + neededMoreFlourThanSugar = recipeFlour) → recipeFlour - neededMoreFlourThanSugar = 5 := 
by
  intros h
  sorry

end cups_of_flour_put_in

namespace find_fraction

import Mathlib

variable (N : ℕ) (F : ℚ)
theorem find_fraction (h1 : N = 90) (h2 : 3 + (1/2 : ℚ) * (1/3 : ℚ) * (1/5 : ℚ) * N = F * N) : F = 1 / 15 :=
sorry

end find_fraction

namespace malvina_correct

import Mathlib
noncomputable def angle (x : ℝ) : Prop := 0 < x ∧ x < 180
noncomputable def malvina_identifies (x : ℝ) : Prop := x > 90

noncomputable def sum_of_values := (Real.sqrt 5 + Real.sqrt 2) / 2

theorem malvina_correct (x : ℝ) (h1 : angle x) (h2 : malvina_identifies x) :
  sum_of_values = (Real.sqrt 5 + Real.sqrt 2) / 2 :=
by sorry

end malvina_correct

namespace add_in_base_7

import Mathlib

theorem add_in_base_7 (X Y : ℕ) (h1 : (X + 5) % 7 = 0) (h2 : (Y + 2) % 7 = X) : X + Y = 2 :=
by
  sorry

end add_in_base_7

namespace probability_participation_on_both_days

import Mathlib

-- Definitions based on conditions
def total_students := 5
def total_combinations := 2^total_students
def same_day_scenarios := 2
def favorable_outcomes := total_combinations - same_day_scenarios

-- Theorem statement
theorem probability_participation_on_both_days :
  (favorable_outcomes / total_combinations : ℚ) = 15 / 16 :=
by
  sorry

end probability_participation_on_both_days

namespace new_rectangle_area_eq_a_squared

import Mathlib

theorem new_rectangle_area_eq_a_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let d := Real.sqrt (a^2 + b^2)
  let base := 2 * (d + b)
  let height := (d - b) / 2
  base * height = a^2 := by
  sorry

end new_rectangle_area_eq_a_squared

namespace division_theorem

import Mathlib

theorem division_theorem (k : ℕ) (h : k = 6) : 24 / k = 4 := by
  sorry

end division_theorem

namespace cost_price_of_apple_is_18

import Mathlib

noncomputable def cp (sp : ℝ) (loss_fraction : ℝ) : ℝ := sp / (1 - loss_fraction)

theorem cost_price_of_apple_is_18 :
  cp 15 (1/6) = 18 :=
by
  sorry

end cost_price_of_apple_is_18

namespace outfit_count_correct

import Mathlib

def total_shirts : ℕ := 8
def total_pants : ℕ := 4
def total_hats : ℕ := 6
def shirt_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def conflict_free_outfits (total_shirts total_pants total_hats : ℕ) : ℕ :=
  let total_outfits := total_shirts * total_pants * total_hats
  let matching_outfits := (2 * 1 * 4) * total_pants
  total_outfits - matching_outfits

theorem outfit_count_correct :
  conflict_free_outfits total_shirts total_pants total_hats = 160 :=
by
  unfold conflict_free_outfits
  norm_num
  sorry

end outfit_count_correct

namespace exists_unique_continuous_extension

import Mathlib

noncomputable def F (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) : ℝ → ℝ :=
  sorry

theorem exists_unique_continuous_extension (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) :
  ∃! F : ℝ → ℝ, Continuous F ∧ ∀ x : ℚ, F x = f x :=
sorry

end exists_unique_continuous_extension

namespace sum_of_angles_of_roots_eq_1020

import Mathlib

noncomputable def sum_of_angles_of_roots : ℝ :=
  60 + 132 + 204 + 276 + 348

theorem sum_of_angles_of_roots_eq_1020 :
  (∑ θ in {60, 132, 204, 276, 348}, θ) = 1020 := by
  sorry

end sum_of_angles_of_roots_eq_1020

namespace Ryan_spit_distance_correct

import Mathlib

-- Definitions of given conditions
def Billy_spit_distance : ℝ := 30
def Madison_spit_distance : ℝ := Billy_spit_distance * 1.20
def Ryan_spit_distance : ℝ := Madison_spit_distance * 0.50

-- Goal statement
theorem Ryan_spit_distance_correct : Ryan_spit_distance = 18 := by
  -- proof would go here
  sorry

end Ryan_spit_distance_correct

namespace min_y_value

import Mathlib

theorem min_y_value :
  ∃ c : ℝ, ∀ x : ℝ, (5 * x^2 + 20 * x + 25) >= c ∧ (∀ x : ℝ, (5 * x^2 + 20 * x + 25 = c) → x = -2) ∧ c = 5 :=
by
  sorry

end min_y_value

namespace cupric_cyanide_formation

import Mathlib

/--
Given:
1 mole of CuSO₄ 
2 moles of HCN

Prove:
The number of moles of Cu(CN)₂ formed is 0.
-/
theorem cupric_cyanide_formation (CuSO₄ HCN : ℕ) (h₁ : CuSO₄ = 1) (h₂ : HCN = 2) : 0 = 0 :=
by
  -- Proof goes here
  sorry

end cupric_cyanide_formation

namespace incorrect_expression_D

import Mathlib

noncomputable def E : ℝ := sorry
def R : ℕ := sorry
def S : ℕ := sorry
def m : ℕ := sorry
def t : ℕ := sorry

-- E is a repeating decimal
-- R is the non-repeating part of E with m digits
-- S is the repeating part of E with t digits

theorem incorrect_expression_D : ¬ (10^m * (10^t - 1) * E = S * (R - 1)) :=
sorry

end incorrect_expression_D

namespace multiply_identity

import Mathlib

variable (x y : ℝ)

theorem multiply_identity :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 := by
  sorry

end multiply_identity

namespace solution_set_of_abs_inequality

import Mathlib

theorem solution_set_of_abs_inequality :
  { x : ℝ | |x^2 - 2| < 2 } = { x : ℝ | -2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2 } :=
sorry

end solution_set_of_abs_inequality

namespace scientific_notation_correct

import Mathlib

noncomputable def scientific_notation_139000 : Prop :=
  139000 = 1.39 * 10^5

theorem scientific_notation_correct : scientific_notation_139000 :=
by
  -- The proof would be included here, but we add sorry to skip it
  sorry

end scientific_notation_correct

namespace irreducible_fraction_for_any_n

import Mathlib

theorem irreducible_fraction_for_any_n (n : ℤ) : Int.gcd (14 * n + 3) (21 * n + 4) = 1 := 
by {
  sorry
}

end irreducible_fraction_for_any_n

namespace contractor_absent_days

import Mathlib

noncomputable def solve_contractor_problem : Prop :=
  ∃ (x y : ℕ), 
    x + y = 30 ∧ 
    25 * x - 750 / 100 * y = 555 ∧
    y = 6

theorem contractor_absent_days : solve_contractor_problem :=
  sorry

end contractor_absent_days

namespace proof_problem

import Mathlib

-- Given conditions for propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

-- Combined proposition p and q
def p_and_q (a : ℝ) := p a ∧ q a

-- Statement of the proof problem: Prove that p_and_q a → a ≤ -1
theorem proof_problem (a : ℝ) : p_and_q a → (a ≤ -1) :=
by
  sorry

end proof_problem

namespace triangle_area

import Mathlib

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def has_perimeter (a b c p : ℝ) : Prop :=
  a + b + c = p

def has_altitude (base side altitude : ℝ) : Prop :=
  (base / 2) ^ 2 + altitude ^ 2 = side ^ 2

def area_of_triangle (a base altitude : ℝ) : ℝ :=
  0.5 * base * altitude

theorem triangle_area (a b c : ℝ)
  (h_iso : is_isosceles a b c)
  (h_p : has_perimeter a b c 40)
  (h_alt : has_altitude (2 * a) b 12) :
  area_of_triangle a (2 * a) 12 = 76.8 :=
by
  sorry

end triangle_area

namespace perpendicular_vectors_parallel_vectors

import Mathlib

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (2, x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x - 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (x : ℝ) :
  dot_product (vector_a x) (vector_b x) = 0 ↔ x = 2 / 3 :=
by sorry

theorem parallel_vectors (x : ℝ) :
  (2 / (x - 1) = x) ∨ (x - 1 = 0) ∨ (2 = 0) ↔ (x = 2 ∨ x = -1) :=
by sorry

end perpendicular_vectors_parallel_vectors

namespace find_a6

import Mathlib

variable (a : ℕ → ℝ)

-- condition: a_2 + a_8 = 16
axiom h1 : a 2 + a 8 = 16

-- condition: a_4 = 1
axiom h2 : a 4 = 1

-- question: Prove that a_6 = 15
theorem find_a6 : a 6 = 15 :=
sorry

end find_a6

namespace complement_of_B_in_A

import Mathlib

def complement (A B : Set Int) := { x ∈ A | x ∉ B }

theorem complement_of_B_in_A (A B : Set Int) (a : Int) (h1 : A = {2, 3, 4}) (h2 : B = {a + 2, a}) (h3 : A ∩ B = B)
: complement A B = {3} :=
  sorry

end complement_of_B_in_A

namespace a_eq_bn

import Mathlib

theorem a_eq_bn (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → ∃ m : ℕ, a - k^n = m * (b - k)) → a = b^n :=
by
  sorry

end a_eq_bn

namespace solution_for_system

import Mathlib
open Real

noncomputable def solve_system (a b x y : ℝ) : Prop :=
  (a * x + b * y = 7 ∧ b * x + a * y = 8)

noncomputable def solve_linear (a b m n : ℝ) : Prop :=
  (a * (m + n) + b * (m - n) = 7 ∧ b * (m + n) + a * (m - n) = 8)

theorem solution_for_system (a b : ℝ) : solve_system a b 2 3 → solve_linear a b (5/2) (-1/2) :=
by {
  sorry
}

end solution_for_system

namespace count_f_compositions

import Mathlib

noncomputable def count_special_functions : Nat :=
  let A := Finset.range 6
  let f := (Set.univ : Set (A → A))
  sorry

theorem count_f_compositions (f : Fin 6 → Fin 6) 
  (h : ∀ x : Fin 6, (f ∘ f ∘ f) x = x) :
  count_special_functions = 81 :=
sorry

end count_f_compositions

namespace largest_even_integer_sum_12000

import Mathlib

theorem largest_even_integer_sum_12000 : 
  ∃ y, (∑ k in (Finset.range 30), (2 * y + 2 * k) = 12000) ∧ (y + 29) * 2 + 58 = 429 :=
by
  sorry

end largest_even_integer_sum_12000

namespace eggs_left_in_box

import Mathlib

theorem eggs_left_in_box (initial_eggs : ℕ) (taken_eggs : ℕ) (remaining_eggs : ℕ) : 
  initial_eggs = 47 → taken_eggs = 5 → remaining_eggs = initial_eggs - taken_eggs → remaining_eggs = 42 :=
by
  sorry

end eggs_left_in_box

namespace minimize_triangle_expression

import Mathlib

theorem minimize_triangle_expression :
  ∃ (a b c : ℤ), a < b ∧ b < c ∧ a + b + c = 30 ∧
  ∀ (x y z : ℤ), x < y ∧ y < z ∧ x + y + z = 30 → (z^2 + 18*x + 18*y - 446) ≥ 17 ∧ 
  ∃ (p q r : ℤ), p < q ∧ q < r ∧ p + q + r = 30 ∧ (r^2 + 18*p + 18*q - 446 = 17) := 
sorry

end minimize_triangle_expression

namespace slope_intercept_equivalence

import Mathlib

-- Define the given equation in Lean
def given_line_equation (x y : ℝ) : Prop := 3 * x - 2 * y = 4

-- Define the slope-intercept form as extracted from the given line equation
def slope_intercept_form (x y : ℝ) : Prop := y = (3/2) * x - 2

-- Prove that the given line equation is equivalent to its slope-intercept form
theorem slope_intercept_equivalence (x y : ℝ) :
  given_line_equation x y ↔ slope_intercept_form x y :=
by sorry

end slope_intercept_equivalence

namespace function_relation

import Mathlib

theorem function_relation (f : ℝ → ℝ) 
  (h0 : ∀ x, f (-x) = f x)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y) :
  f 0 < f (-6.5) ∧ f (-6.5) < f (-1) := 
by
  sorry

end function_relation

namespace negation_of_p

import Mathlib

variable (x : ℝ)

-- Define the original proposition p
def p := ∀ x, x^2 < 1 → x < 1

-- Define the negation of p
def neg_p := ∃ x₀, x₀^2 ≥ 1 ∧ x₀ < 1

-- State the theorem that negates p
theorem negation_of_p : ¬ p ↔ neg_p :=
by
  sorry

end negation_of_p

namespace mutually_exclusive_events

import Mathlib

-- Define the conditions
variable (redBalls greenBalls : ℕ)
variable (n : ℕ) -- Number of balls drawn
variable (event_one_red_ball event_two_green_balls : Prop)

-- Assumptions: more than two red balls and more than two green balls
axiom H1 : 2 < redBalls
axiom H2 : 2 < greenBalls

-- Assume that exactly one red ball and exactly two green balls are events
axiom H3 : event_one_red_ball = (n = 2 ∧ 1 ≤ redBalls ∧ 1 ≤ greenBalls)
axiom H4 : event_two_green_balls = (n = 2 ∧ greenBalls ≥ 2)

-- Definition of mutually exclusive events
def mutually_exclusive (A B : Prop) : Prop :=
  A ∧ B → false

-- Statement of the theorem
theorem mutually_exclusive_events :
  mutually_exclusive event_one_red_ball event_two_green_balls :=
by {
  sorry
}

end mutually_exclusive_events

namespace sin_B_value_triangle_area

import Mathlib

-- Problem 1: sine value of angle B given the conditions
theorem sin_B_value (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C) :
  Real.sin B = (4 * Real.sqrt 5) / 9 :=
sorry

-- Problem 2: Area of triangle ABC given the conditions and b = 4
theorem triangle_area (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C)
  (h3 : b = 4) :
  (1 / 2) * b * c * Real.sin A = (14 * Real.sqrt 5) / 9 :=
sorry

end sin_B_value_triangle_area

namespace num_partitions_of_staircase

import Mathlib

-- Definition of a staircase
def is_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ j → j ≤ i → i ≤ n → cells (i, j)

-- Number of partitions of a staircase of height n
def num_partitions (n : ℕ) : ℕ :=
  2^(n-1)

theorem num_partitions_of_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) :
  is_staircase n cells → (∃ p : ℕ, p = num_partitions n) :=
by
  intro h
  use (2^(n-1))
  sorry

end num_partitions_of_staircase

namespace grasshopper_total_distance

import Mathlib

theorem grasshopper_total_distance :
  let initial := 2
  let first_jump := -3
  let second_jump := 8
  let final_jump := -1
  abs (first_jump - initial) + abs (second_jump - first_jump) + abs (final_jump - second_jump) = 25 :=
by
  sorry

end grasshopper_total_distance

namespace tray_height

import Mathlib

-- Declare the main theorem with necessary given conditions.
theorem tray_height (a b c : ℝ) (side_length : ℝ) (cut_distance : ℝ) (angle : ℝ) : 
  (side_length = 150) →
  (cut_distance = Real.sqrt 50) →
  (angle = 45) →
  a^2 + b^2 = c^2 → -- Condition from Pythagorean theorem
  a = side_length * Real.sqrt 2 / 2 - cut_distance → -- Calculation for half diagonal minus cut distance
  b = (side_length * Real.sqrt 2 / 2 - cut_distance) / 2 → -- Perpendicular from R to the side
  side_length = 150 → -- Ensure consistency of side length
  b^2 + c^2 = side_length^2 → -- Ensure we use another Pythagorean relation
  c = Real.sqrt 7350 → -- Derived c value
  c = Real.sqrt 1470 := -- Simplified form of c.
  sorry

end tray_height

namespace sum_mod_9

import Mathlib

theorem sum_mod_9 (h1 : 34125 % 9 = 1) (h2 : 34126 % 9 = 2) (h3 : 34127 % 9 = 3)
                  (h4 : 34128 % 9 = 4) (h5 : 34129 % 9 = 5) (h6 : 34130 % 9 = 6)
                  (h7 : 34131 % 9 = 7) :
  (34125 + 34126 + 34127 + 34128 + 34129 + 34130 + 34131) % 9 = 1 :=
by
  sorry

end sum_mod_9

namespace xiaohui_pe_score

import Mathlib

-- Define the conditions
def morning_score : ℝ := 95
def midterm_score : ℝ := 90
def final_score : ℝ := 85

def morning_weight : ℝ := 0.2
def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.5

-- The problem is to prove that Xiaohui's physical education score for the semester is 88.5 points.
theorem xiaohui_pe_score :
  morning_score * morning_weight +
  midterm_score * midterm_weight +
  final_score * final_weight = 88.5 :=
by
  sorry

end xiaohui_pe_score

namespace fisher_needed_score

import Mathlib

-- Condition 1: To have an average of at least 85% over all four quarters
def average_score_threshold := 85
def total_score := 4 * average_score_threshold

-- Condition 2: Fisher's scores for the first three quarters
def first_three_scores := [82, 77, 75]
def current_total_score := first_three_scores.sum

-- Define the Lean statement to prove
theorem fisher_needed_score : ∃ x, current_total_score + x = total_score ∧ x = 106 := by
  sorry

end fisher_needed_score

namespace Tom_age_ratio

import Mathlib

variable (T N : ℕ)
variable (a : ℕ)
variable (c3 c4 : ℕ)

-- conditions
def condition1 : Prop := T = 4 * a + 5
def condition2 : Prop := T - N = 3 * (4 * a + 5 - 4 * N)

theorem Tom_age_ratio (h1 : condition1 T a) (h2 : condition2 T N a) : (T = 6 * N) :=
by sorry

end Tom_age_ratio

namespace minimal_volume_block

import Mathlib

theorem minimal_volume_block (l m n : ℕ) (h : (l - 1) * (m - 1) * (n - 1) = 297) : l * m * n = 192 :=
sorry

end minimal_volume_block

namespace number_of_true_propositions

import Mathlib

theorem number_of_true_propositions : 
  (∃ x y : ℝ, (x * y = 1) ↔ (x = y⁻¹ ∨ y = x⁻¹)) ∧
  (¬(∀ x : ℝ, (x > -3) → x^2 - x - 6 ≤ 0)) ∧
  (¬(∀ a b : ℝ, (a > b) → (a^2 < b^2))) ∧
  (¬(∀ x : ℝ, (x - 1/x > 0) → (x > -1))) →
  True := by
  sorry

end number_of_true_propositions

namespace find_abc_solutions

import Mathlib

theorem find_abc_solutions
    (a b c : ℕ)
    (h_pos : (a > 0) ∧ (b > 0) ∧ (c > 0))
    (h1 : a < b)
    (h2 : a < 4 * c)
    (h3 : b * c ^ 3 ≤ a * c ^ 3 + b) :
    ((a = 7) ∧ (b = 8) ∧ (c = 2)) ∨
    ((a = 1 ∨ a = 2 ∨ a = 3) ∧ (b > a) ∧ (c = 1)) :=
by
  sorry

end find_abc_solutions

namespace total_tea_cups

import Mathlib

def num_cupboards := 8
def num_compartments_per_cupboard := 5
def num_tea_cups_per_compartment := 85

theorem total_tea_cups :
  num_cupboards * num_compartments_per_cupboard * num_tea_cups_per_compartment = 3400 :=
by
  sorry

end total_tea_cups

namespace x_pow_n_plus_inv_x_pow_n

import Mathlib

theorem x_pow_n_plus_inv_x_pow_n (θ : ℝ) (x : ℝ) (n : ℕ) (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : x + 1 / x = 2 * Real.sin θ) (hn_pos : 0 < n) : 
  x^n + (1 / x)^n = 2 * Real.cos (n * θ) := 
by
  sorry

end x_pow_n_plus_inv_x_pow_n

namespace perimeter_of_face_given_volume

import Mathlib

-- Definitions based on conditions
def volume_of_cube (v : ℝ) := v = 512

def side_of_cube (s : ℝ) := s^3 = 512

def perimeter_of_face (p s : ℝ) := p = 4 * s

-- Lean 4 statement: prove that the perimeter of one face of the cube is 32 cm given the volume is 512 cm³.
theorem perimeter_of_face_given_volume :
  ∃ s : ℝ, volume_of_cube (s^3) ∧ perimeter_of_face 32 s :=
by sorry

end perimeter_of_face_given_volume

namespace zhou_yu_age_equation

import Mathlib

variable (x : ℕ)

theorem zhou_yu_age_equation (h : x + 3 < 10) : 10 * x + (x + 3) = (x + 3) ^ 2 :=
  sorry

end zhou_yu_age_equation

namespace percentage_of_women_picnic

import Mathlib

theorem percentage_of_women_picnic (E : ℝ) (h1 : 0.20 * 0.55 * E + W * 0.45 * E = 0.29 * E) : 
  W = 0.4 := 
  sorry

end percentage_of_women_picnic

namespace probability_of_log_ge_than_1

import Mathlib

noncomputable def probability_log_greater_than_one : ℝ := sorry

theorem probability_of_log_ge_than_1 :
  probability_log_greater_than_one = 1 / 2 :=
sorry

end probability_of_log_ge_than_1

namespace tim_tasks_per_day

import Mathlib

theorem tim_tasks_per_day (earnings_per_task : ℝ) (days_per_week : ℕ) (weekly_earnings : ℝ) :
  earnings_per_task = 1.2 ∧ days_per_week = 6 ∧ weekly_earnings = 720 → (weekly_earnings / days_per_week / earnings_per_task = 100) :=
by
  sorry

end tim_tasks_per_day

namespace area_square_field

import Mathlib

-- Define the side length of the square
def side_length : ℕ := 12

-- Define the area of the square with the given side length
def area_of_square (side : ℕ) : ℕ := side * side

-- The theorem to state and prove
theorem area_square_field : area_of_square side_length = 144 :=
by
  sorry

end area_square_field

namespace hair_ratio

import Mathlib

theorem hair_ratio (washed : ℕ) (grow_back : ℕ) (brushed : ℕ) (n : ℕ)
  (hwashed : washed = 32)
  (hgrow_back : grow_back = 49)
  (heq : washed + brushed + 1 = grow_back) :
  (brushed : ℚ) / washed = 1 / 2 := 
by 
  sorry

end hair_ratio

namespace least_n_exceeds_product

import Mathlib

def product_exceeds (n : ℕ) : Prop :=
  10^(n * (n + 1) / 18) > 10^6

theorem least_n_exceeds_product (n : ℕ) (h : n = 12) : product_exceeds n :=
by
  rw [h]
  sorry

end least_n_exceeds_product

namespace hard_candy_food_colouring

import Mathlib

noncomputable def food_colouring_per_hard_candy (lollipop_use : ℕ) (gummy_use : ℕ)
    (lollipops_per_day : ℕ) (gummies_per_day : ℕ) (hard_candies_per_day : ℕ)
    (total_food_colouring : ℕ) : ℕ := 
by
  -- Let ml_lollipops be the total amount needed for lollipops
  let ml_lollipops := lollipop_use * lollipops_per_day
  -- Let ml_gummy be the total amount needed for gummy candies
  let ml_gummy := gummy_use * gummies_per_day
  -- Let ml_non_hard be the amount for lollipops and gummy candies combined
  let ml_non_hard := ml_lollipops + ml_gummy
  -- Let ml_hard be the amount used for hard candies alone
  let ml_hard := total_food_colouring - ml_non_hard
  -- Compute the food colouring used per hard candy
  exact ml_hard / hard_candies_per_day

theorem hard_candy_food_colouring :
  food_colouring_per_hard_candy 8 3 150 50 20 1950 = 30 :=
by
  unfold food_colouring_per_hard_candy
  sorry

end hard_candy_food_colouring

namespace spadesuit_eval

import Mathlib

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 6 3) = -704 := by
  sorry

end spadesuit_eval

namespace triangle_angle_sum

import Mathlib

theorem triangle_angle_sum (CD CB : ℝ) 
    (isosceles_triangle: CD = CB)
    (interior_pentagon_angle: 108 = 180 * (5 - 2) / 5)
    (interior_triangle_angle: 60 = 180 / 3)
    (triangle_angle_sum: ∀ (a b c : ℝ), a + b + c = 180) :
    mangle_CDB = 6 :=
by
  have x : ℝ := 6
  sorry

end triangle_angle_sum

namespace parcels_division

import Mathlib

theorem parcels_division (x y n : ℕ) (h : 5 + 2 * x + 3 * y = 4 * n) (hn : n = x + y) :
    n = 3 ∨ n = 4 ∨ n = 5 := 
sorry

end parcels_division

namespace plane_through_intersection

import Mathlib.Data.Real.Basic

def plane1 (x y z : ℝ) : Prop := x + y + 5 * z - 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x + 3 * y - z + 2 = 0
def pointM (x y z : ℝ) : Prop := (x, y, z) = (3, 2, 1)

theorem plane_through_intersection (x y z : ℝ) :
  plane1 x y z ∧ plane2 x y z ∧ pointM x y z → 5 * x + 14 * y - 74 * z + 31 = 0 := by
  intro h
  sorry

end plane_through_intersection

namespace average_score

import Mathlib

theorem average_score (s1 s2 s3 : ℕ) (n : ℕ) (h1 : s1 = 115) (h2 : s2 = 118) (h3 : s3 = 115) (h4 : n = 3) :
    (s1 + s2 + s3) / n = 116 :=
by
    sorry

end average_score

namespace lunch_break_duration

import Mathlib

theorem lunch_break_duration (m a : ℝ) (L : ℝ) :
  (9 - L) * (m + a) = 0.6 → 
  (7 - L) * a = 0.3 → 
  (5 - L) * m = 0.1 → 
  L = 42 / 60 :=
by sorry

end lunch_break_duration

namespace oplus_self_twice

import Mathlib

def my_oplus (x y : ℕ) := 3^x - y

theorem oplus_self_twice (a : ℕ) : my_oplus a (my_oplus a a) = a := by
  sorry

end oplus_self_twice

namespace Eddy_travel_time

import Mathlib

theorem Eddy_travel_time (T V_e V_f : ℝ) 
  (dist_AB dist_AC : ℝ) 
  (time_Freddy : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : dist_AB = 600) 
  (h2 : dist_AC = 300) 
  (h3 : time_Freddy = 3) 
  (h4 : speed_ratio = 2)
  (h5 : V_f = dist_AC / time_Freddy)
  (h6 : V_e = speed_ratio * V_f)
  (h7 : T = dist_AB / V_e) :
  T = 3 :=
by
  sorry

end Eddy_travel_time

namespace least_positive_integer_condition

import Mathlib

theorem least_positive_integer_condition
  (a : ℤ) (ha1 : a % 4 = 1) (ha2 : a % 5 = 2) (ha3 : a % 6 = 3) :
  a > 0 → a = 57 :=
by
  intro ha_pos
  -- Proof omitted for brevity
  sorry

end least_positive_integer_condition

namespace grill_burns_fifteen_coals_in_twenty_minutes

import Mathlib

-- Define the problem conditions
def total_coals (bags : ℕ) (coals_per_bag : ℕ) : ℕ :=
  bags * coals_per_bag

def burning_ratio (total_coals : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / total_coals

-- Given conditions
def bags := 3
def coals_per_bag := 60
def total_minutes := 240
def fifteen_coals := 15

-- Problem statement
theorem grill_burns_fifteen_coals_in_twenty_minutes :
  total_minutes / total_coals bags coals_per_bag * fifteen_coals = 20 :=
by
  sorry

end grill_burns_fifteen_coals_in_twenty_minutes

namespace amount_paid_to_Y

import Mathlib

-- Definition of the conditions.
def total_payment (X Y : ℕ) : Prop := X + Y = 330
def payment_relation (X Y : ℕ) : Prop := X = 12 * Y / 10

-- The theorem we want to prove.
theorem amount_paid_to_Y (X Y : ℕ) (h1 : total_payment X Y) (h2 : payment_relation X Y) : Y = 150 := 
by 
  sorry

end amount_paid_to_Y

namespace frank_hamburger_goal

import Mathlib

theorem frank_hamburger_goal:
  let price_per_hamburger := 5
  let group1_hamburgers := 2 * 4
  let group2_hamburgers := 2 * 2
  let current_hamburgers := group1_hamburgers + group2_hamburgers
  let extra_hamburgers_needed := 4
  let total_hamburgers := current_hamburgers + extra_hamburgers_needed
  price_per_hamburger * total_hamburgers = 80 :=
by
  sorry

end frank_hamburger_goal

namespace total_pumped_volume

import Mathlib

def powerJetA_rate : ℕ := 360
def powerJetB_rate : ℕ := 540
def powerJetA_time : ℕ := 30
def powerJetB_time : ℕ := 45

def pump_volume (rate : ℕ) (minutes : ℕ) : ℕ :=
  rate * (minutes / 60)

theorem total_pumped_volume : 
  pump_volume powerJetA_rate powerJetA_time + pump_volume powerJetB_rate powerJetB_time = 585 := 
by
  sorry

end total_pumped_volume

namespace subsets_neither_A_nor_B

import Mathlib

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {4, 5, 6, 7, 8}

theorem subsets_neither_A_nor_B : 
  (U.powerset.card - A.powerset.card - B.powerset.card + (A ∩ B).powerset.card) = 196 := by 
  sorry

end subsets_neither_A_nor_B

namespace union_complement_eq

import Mathlib

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq

namespace union_complement_eq

import Mathlib

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq

namespace union_complement_set

import Mathlib

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set

namespace union_complement_set

import Mathlib

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set

namespace quadratic_least_value

import Mathlib

variable (a b c : ℝ)

theorem quadratic_least_value (h_a_pos : a > 0)
  (h_c_eq : ∀ x : ℝ, a * x^2 + b * x + c ≥ 9) :
  c = 9 + b^2 / (4 * a) :=
by
  sorry

end quadratic_least_value

namespace trig_identity

import Mathlib

theorem trig_identity (α : ℝ) :
  (Real.cos (α - 35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180 + α) +
   Real.sin (α - 35 * Real.pi / 180) * Real.sin (25 * Real.pi / 180 + α)) = 1 / 2 :=
by
  sorry

end trig_identity

namespace emerie_dimes_count

import Mathlib

variables (zain_coins emerie_coins num_quarters num_nickels : ℕ)
variable (emerie_dimes : ℕ)

-- Conditions as per part a)
axiom zain_has_more_coins : ∀ (e z : ℕ), z = e + 10
axiom total_zain_coins : zain_coins = 48
axiom emerie_coins_from_quarters_and_nickels : num_quarters = 6 ∧ num_nickels = 5
axiom emerie_known_coins : ∀ q n : ℕ, emerie_coins = q + n + emerie_dimes

-- The statement to prove
theorem emerie_dimes_count : emerie_coins = 38 → emerie_dimes = 27 := 
by 
  sorry

end emerie_dimes_count

namespace tank_empty_time_when_inlet_open

import Mathlib

-- Define the conditions
def leak_empty_time : ℕ := 6
def tank_capacity : ℕ := 4320
def inlet_rate_per_minute : ℕ := 6

-- Calculate rates from conditions
def leak_rate_per_hour : ℕ := tank_capacity / leak_empty_time
def inlet_rate_per_hour : ℕ := inlet_rate_per_minute * 60

-- Proof Problem: Prove the time for the tank to empty when both leak and inlet are open
theorem tank_empty_time_when_inlet_open :
  tank_capacity / (leak_rate_per_hour - inlet_rate_per_hour) = 12 :=
by
  sorry

end tank_empty_time_when_inlet_open

namespace bank_exceeds_1600cents_in_9_days_after_Sunday

import Mathlib

theorem bank_exceeds_1600cents_in_9_days_after_Sunday
  (a : ℕ)
  (r : ℕ)
  (initial_deposit : ℕ)
  (days_after_sunday : ℕ)
  (geometric_series : ℕ -> ℕ)
  (sum_geometric_series : ℕ -> ℕ)
  (geo_series_definition : ∀(n : ℕ), geometric_series n = 5 * 2^n)
  (sum_geo_series_definition : ∀(n : ℕ), sum_geometric_series n = 5 * (2^n - 1))
  (exceeds_condition : ∀(n : ℕ), sum_geometric_series n > 1600 -> n >= 9) :
  days_after_sunday = 9 → a = 5 → r = 2 → initial_deposit = 5 → days_after_sunday = 9 → geometric_series 1 = 10 → sum_geometric_series 9 > 1600 :=
by sorry

end bank_exceeds_1600cents_in_9_days_after_Sunday

namespace repeated_process_pure_alcohol

import Mathlib

theorem repeated_process_pure_alcohol : 
  ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, 2 * (1 / 2 : ℝ)^(m : ℝ) ≥ 0.2 := by
  sorry

end repeated_process_pure_alcohol

namespace total_spent_correct

import Mathlib

def shorts : ℝ := 13.99
def shirt : ℝ := 12.14
def jacket : ℝ := 7.43
def total_spent : ℝ := 33.56

theorem total_spent_correct : shorts + shirt + jacket = total_spent :=
by
  sorry

end total_spent_correct

namespace amount_of_salmon_sold_first_week

import Mathlib

-- Define the conditions
def fish_sold_in_two_weeks (x : ℝ) := x + 3 * x = 200

-- Define the theorem we want to prove
theorem amount_of_salmon_sold_first_week (x : ℝ) (h : fish_sold_in_two_weeks x) : x = 50 :=
by
  sorry

end amount_of_salmon_sold_first_week

namespace factor_difference_of_squares_example

import Mathlib

theorem factor_difference_of_squares_example :
    (m : ℝ) → (m ^ 2 - 4 = (m + 2) * (m - 2)) :=
by
    intro m
    sorry

end factor_difference_of_squares_example

namespace distance_between_trees

import Mathlib

def yard_length : ℝ := 1530
def number_of_trees : ℝ := 37
def number_of_gaps := number_of_trees - 1

theorem distance_between_trees :
  number_of_gaps ≠ 0 →
  (yard_length / number_of_gaps) = 42.5 :=
by
  sorry

end distance_between_trees

namespace shortest_wire_length

import Mathlib

theorem shortest_wire_length (d1 d2 : ℝ) (r1 r2 : ℝ) (t : ℝ) :
  d1 = 8 ∧ d2 = 20 ∧ r1 = 4 ∧ r2 = 10 ∧ t = 8 * Real.sqrt 10 + 17.4 * Real.pi → 
  ∃ l : ℝ, l = t :=
by 
  sorry

end shortest_wire_length

namespace geometric_series_sum

import Mathlib

theorem geometric_series_sum :
  let a := -3
  let r := -2
  let n := 9
  let term := a * r^(n-1)
  let Sn := (a * (r^n - 1)) / (r - 1)
  term = -768 → Sn = 514 := by
  intros a r n term Sn h_term
  sorry

end geometric_series_sum

namespace impossible_event

import Mathlib

noncomputable def EventA := ∃ (ω : ℕ), ω = 0 ∨ ω = 1
noncomputable def EventB := ∃ (t : ℤ), t >= 0
noncomputable def Bag := {b : String // b = "White"}
noncomputable def EventC := ∀ (x : Bag), x.val ≠ "Red"
noncomputable def EventD := ∀ (a b : ℤ), (a > 0 ∧ b < 0) → a > b

theorem impossible_event:
  (EventA ∧ EventB ∧ EventD) →
  EventC :=
by
  sorry

end impossible_event

namespace magnitude_of_z

import Mathlib

theorem magnitude_of_z (z : ℂ) (h : z * (2 - Complex.I) = Complex.I) : Complex.abs z = 1 / Real.sqrt 5 := by
  sorry

end magnitude_of_z

namespace find_c

import Mathlib

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x : ℝ, g_inv (g x c) = x) -> c = 3 :=
by 
  intro h
  sorry

end find_c

namespace find_number

import Mathlib

-- Define the conditions
def satisfies_condition (x : ℝ) : Prop := x * 4 * 25 = 812

-- The main theorem stating that the number satisfying the condition is 8.12
theorem find_number (x : ℝ) (h : satisfies_condition x) : x = 8.12 :=
by
  sorry

end find_number

namespace polynomial_division_remainder_zero

import Mathlib

theorem polynomial_division_remainder_zero (x : ℂ) (hx : x^5 + x^4 + x^3 + x^2 + x + 1 = 0)
  : (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end polynomial_division_remainder_zero

namespace cubed_difference

import Mathlib

theorem cubed_difference (x : ℝ) (h : x - 1/x = 3) : (x^3 - 1/x^3 = 36) := 
by
  sorry

end cubed_difference

namespace percentage_of_girls_after_change

import Mathlib

variables (initial_total_children initial_boys initial_girls additional_boys : ℕ)
variables (percentage_boys : ℚ)

-- Initial conditions
def initial_conditions : Prop :=
  initial_total_children = 50 ∧
  percentage_boys = 90 / 100 ∧
  initial_boys = initial_total_children * percentage_boys ∧
  initial_girls = initial_total_children - initial_boys ∧
  additional_boys = 50

-- Statement to prove
theorem percentage_of_girls_after_change :
  initial_conditions initial_total_children initial_boys initial_girls additional_boys percentage_boys →
  (initial_girls / (initial_total_children + additional_boys) * 100 = 5) :=
by
  sorry

end percentage_of_girls_after_change

namespace percentage_increase_each_job

import Mathlib

-- Definitions of original and new amounts for each job as given conditions
def original_first_job : ℝ := 65
def new_first_job : ℝ := 70

def original_second_job : ℝ := 240
def new_second_job : ℝ := 315

def original_third_job : ℝ := 800
def new_third_job : ℝ := 880

-- Proof problem statement
theorem percentage_increase_each_job :
  (new_first_job - original_first_job) / original_first_job * 100 = 7.69 ∧
  (new_second_job - original_second_job) / original_second_job * 100 = 31.25 ∧
  (new_third_job - original_third_job) / original_third_job * 100 = 10 := by
  sorry

end percentage_increase_each_job

namespace A_plus_B_eq_one_fourth

import Mathlib

noncomputable def A : ℚ := 1 / 3
noncomputable def B : ℚ := -1 / 12

theorem A_plus_B_eq_one_fourth :
  A + B = 1 / 4 := by
  sorry

end A_plus_B_eq_one_fourth

namespace alice_bob_task

import Mathlib

theorem alice_bob_task (t : ℝ) (h₁ : 1/4 + 1/6 = 5/12) (h₂ : t - 1/2 ≠ 0) :
    (5/12) * (t - 1/2) = 1 :=
sorry

end alice_bob_task

namespace samantha_coins_worth

import Mathlib

-- Define the conditions and the final question with an expected answer.
theorem samantha_coins_worth (n d : ℕ) (h1 : n + d = 30)
  (h2 : 10 * n + 5 * d = 5 * n + 10 * d + 120) :
  (5 * n + 10 * d) = 165 := 
sorry

end samantha_coins_worth

namespace rate_per_kg_for_mangoes

import Mathlib

theorem rate_per_kg_for_mangoes (quantity_grapes : ℕ)
    (rate_grapes : ℕ)
    (quantity_mangoes : ℕ)
    (total_payment : ℕ)
    (rate_mangoes : ℕ) :
    quantity_grapes = 8 →
    rate_grapes = 70 →
    quantity_mangoes = 9 →
    total_payment = 1055 →
    8 * 70 + 9 * rate_mangoes = 1055 →
    rate_mangoes = 55 := by
  intros h1 h2 h3 h4 h5
  have h6 : 8 * 70 = 560 := by norm_num
  have h7 : 560 + 9 * rate_mangoes = 1055 := by rw [h5]
  have h8 : 1055 - 560 = 495 := by norm_num
  have h9 : 9 * rate_mangoes = 495 := by linarith
  have h10 : rate_mangoes = 55 := by linarith
  exact h10

end rate_per_kg_for_mangoes

namespace intersection_A_B

import Mathlib

-- Define the sets A and B based on given conditions
def A : Set ℝ := { x | x^2 ≤ 1 }
def B : Set ℝ := { x | (x - 2) / x ≤ 0 }

-- State the proof problem
theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_A_B

namespace arithmetic_sequence_a3

import Mathlib

theorem arithmetic_sequence_a3 (a : ℕ → ℤ) (h1 : a 1 = 4) (h10 : a 10 = 22) (d : ℤ) (hd : ∀ n, a n = a 1 + (n - 1) * d) :
  a 3 = 8 :=
by
  -- Skipping the proof
  sorry

end arithmetic_sequence_a3

namespace sin_identity

import Mathlib

theorem sin_identity (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
  by 
  sorry

end sin_identity

namespace part1_part2

import Mathlib

-- Definitions from part (a)
def a_n (n : ℕ) : ℕ := 2 * n - 1
def b_n (n : ℕ) : ℕ := 2 ^ (a_n n + 1)

-- Specification from the given problem
def S_n (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), a_n i
def c_n (n : ℕ) : ℕ := a_n n * b_n n
def T_n (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), c_n i

-- Theorem to be proven (part (c))
theorem part1 (n : ℕ) : S_n n = n ^ 2 := by
  sorry

theorem part2 (n : ℕ) : T_n n = (24 * n - 20) * 4 ^ n / 9 + 20 / 9 := by
  sorry

end part1_part2

namespace determine_x_y

import Mathlib

-- Definitions from the conditions
def cond1 (x y : ℚ) : Prop := 12 * x + 198 = 12 * y + 176
def cond2 (x y : ℚ) : Prop := x + y = 29

-- Statement to prove
theorem determine_x_y : ∃ x y : ℚ, cond1 x y ∧ cond2 x y ∧ x = 163 / 12 ∧ y = 185 / 12 := 
by 
  sorry

end determine_x_y

namespace probability_red_second_draw

import Mathlib

theorem probability_red_second_draw 
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (after_first_draw_balls : ℕ)
  (after_first_draw_red : ℕ)
  (probability : ℚ) :
  total_balls = 5 →
  red_balls = 2 →
  white_balls = 3 →
  after_first_draw_balls = 4 →
  after_first_draw_red = 2 →
  probability = after_first_draw_red / after_first_draw_balls →
  probability = 0.5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end probability_red_second_draw

namespace daughter_work_alone_12_days

import Mathlib

/-- Given a man, his wife, and their daughter working together on a piece of work. The man can complete the work in 4 days, the wife in 6 days, and together with their daughter, they can complete it in 2 days. Prove that the daughter alone would take 12 days to complete the work. -/
theorem daughter_work_alone_12_days (h1 : (1/4 : ℝ) + (1/6) + D = 1/2) : D = 1/12 :=
by
  sorry

end daughter_work_alone_12_days

namespace simplify_expression

import Mathlib

theorem simplify_expression :
  (1 / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 6 - 2)))) =
  ((3 * Real.sqrt 5 + 2 * Real.sqrt 6 + 2) / 29) :=
  sorry

end simplify_expression

namespace no_values_satisfy_equation

import Mathlib

-- Define the sum of the digits function S
noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the sum of digits of the sum of the digits function S(S(n))
noncomputable def sum_of_sum_of_digits (n : ℕ) : ℕ :=
  sum_of_digits (sum_of_digits n)

-- Theorem statement about the number of n satisfying n + S(n) + S(S(n)) = 2099
theorem no_values_satisfy_equation :
  (∃ n : ℕ, n > 0 ∧ n + sum_of_digits n + sum_of_sum_of_digits n = 2099) ↔ False := sorry

end no_values_satisfy_equation

namespace Johnson_farm_budget

import Mathlib

variable (total_land : ℕ) (corn_cost_per_acre : ℕ) (wheat_cost_per_acre : ℕ)
variable (acres_wheat : ℕ) (acres_corn : ℕ)

def total_money (total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn : ℕ) : ℕ :=
  acres_corn * corn_cost_per_acre + acres_wheat * wheat_cost_per_acre

theorem Johnson_farm_budget :
  total_land = 500 ∧
  corn_cost_per_acre = 42 ∧
  wheat_cost_per_acre = 30 ∧
  acres_wheat = 200 ∧
  acres_corn = total_land - acres_wheat →
  total_money total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn = 18600 := by
  sorry

end Johnson_farm_budget

namespace regression_analysis_incorrect_statement

import Mathlib

theorem regression_analysis_incorrect_statement
  (y : ℕ → ℝ) (x : ℕ → ℝ) (b a : ℝ)
  (r : ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ)
  (H1 : ∀ i, y i = b * x i + a)
  (H2 : abs r = 1 → ∀ x1 x2, l x1 = l x2 → x1 = x2)
  (H3 : ∃ m k, ∀ x, l x = m * x + k)
  (H4 : P.1 = b → l P.1 = P.2)
  (cond_A : ∀ i, y i ≠ b * x i + a) : false := 
sorry

end regression_analysis_incorrect_statement

namespace circle_center_and_radius_sum

import Mathlib

theorem circle_center_and_radius_sum :
  let a := -4
  let b := -8
  let r := Real.sqrt 17
  a + b + r = -12 + Real.sqrt 17 :=
by
  sorry

end circle_center_and_radius_sum

namespace oldest_child_age

import Mathlib

def avg (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem oldest_child_age (a b : ℕ) (h1 : avg a b x = 10) (h2 : a = 8) (h3 : b = 11) : x = 11 :=
by
  sorry

end oldest_child_age

namespace total_worth_is_correct

import Mathlib

-- Define the conditions
def rows : ℕ := 4
def gold_bars_per_row : ℕ := 20
def worth_per_gold_bar : ℕ := 20000

-- Define the calculated values
def total_gold_bars : ℕ := rows * gold_bars_per_row
def total_worth_of_gold_bars : ℕ := total_gold_bars * worth_per_gold_bar

-- Theorem statement to prove the correct total worth
theorem total_worth_is_correct : total_worth_of_gold_bars = 1600000 := by
  sorry

end total_worth_is_correct

namespace remainder_when_xyz_divided_by_9_is_0

import Mathlib

theorem remainder_when_xyz_divided_by_9_is_0
  (x y z : ℕ)
  (hx : x < 9)
  (hy : y < 9)
  (hz : z < 9)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z) % 9 = 0 := by
  sorry

end remainder_when_xyz_divided_by_9_is_0

namespace gcd_expression

import Mathlib

theorem gcd_expression (n : ℕ) (h : n > 2) : Nat.gcd (n^5 - 5 * n^3 + 4 * n) 120 = 120 :=
by
  sorry

end gcd_expression

namespace sum_a_b_is_nine

import Mathlib

theorem sum_a_b_is_nine (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
    (h3 : (b + 2 - a)^2 + (a - b)^2 + (b + 2 + a)^2 + (a + b)^2 = 324) 
    (h4 : ∃ a' b', a' = a ∧ b' = b ∧ (b + 2 - a) * 1 = -(b + 2 - a)) : 
  a + b = 9 :=
sorry

end sum_a_b_is_nine

namespace jack_jogging_speed_needed

import Mathlib

noncomputable def jack_normal_speed : ℝ :=
  let normal_melt_time : ℝ := 10
  let faster_melt_factor : ℝ := 0.75
  let adjusted_melt_time : ℝ := normal_melt_time * faster_melt_factor
  let adjusted_melt_time_hours : ℝ := adjusted_melt_time / 60
  let distance_to_beach : ℝ := 2
  let required_speed : ℝ := distance_to_beach / adjusted_melt_time_hours
  let slope_reduction_factor : ℝ := 0.8
  required_speed / slope_reduction_factor

theorem jack_jogging_speed_needed
  (normal_melt_time : ℝ := 10) 
  (faster_melt_factor : ℝ := 0.75) 
  (distance_to_beach : ℝ := 2) 
  (slope_reduction_factor : ℝ := 0.8) :
  jack_normal_speed = 20 := 
by
  sorry

end jack_jogging_speed_needed

namespace horner_v3_value

import Mathlib

-- Define constants
def a_n : ℤ := 2 -- Leading coefficient of x^5
def a_3 : ℤ := -3 -- Coefficient of x^3
def a_2 : ℤ := 5 -- Coefficient of x^2
def a_0 : ℤ := -4 -- Constant term
def x : ℤ := 2 -- Given value of x

-- Horner's method sequence for the coefficients
def v_0 : ℤ := a_n -- Initial value v_0
def v_1 : ℤ := v_0 * x -- Calculated as v_0 * x
def v_2 : ℤ := v_1 * x + a_3 -- Calculated as v_1 * x + a_3 (coefficient of x^3)
def v_3 : ℤ := v_2 * x + a_2 -- Calculated as v_2 * x + a_2 (coefficient of x^2)

theorem horner_v3_value : v_3 = 15 := 
by
  -- Formal proof would go here, skipped due to problem specifications
  sorry

end horner_v3_value

namespace original_population_correct

import Mathlib

def original_population_problem :=
  let original_population := 6731
  let final_population := 4725
  let initial_disappeared := 0.10 * original_population
  let remaining_after_disappearance := original_population - initial_disappeared
  let left_after_remaining := 0.25 * remaining_after_disappearance
  let remaining_after_leaving := remaining_after_disappearance - left_after_remaining
  let disease_affected := 0.05 * original_population
  let disease_died := 0.02 * disease_affected
  let disease_migrated := 0.03 * disease_affected
  let remaining_after_disease := remaining_after_leaving - (disease_died + disease_migrated)
  let moved_to_village := 0.04 * remaining_after_disappearance
  let total_after_moving := remaining_after_disease + moved_to_village
  let births := 0.008 * original_population
  let deaths := 0.01 * original_population
  let final_population_calculated := total_after_moving + (births - deaths)
  final_population_calculated = final_population

theorem original_population_correct :
  original_population_problem ↔ True :=
by
  sorry

end original_population_correct

namespace arithmetic_mean

import Mathlib

variable {x b c : ℝ}

theorem arithmetic_mean (hx : x ≠ 0) (hb : b ≠ c) : 
  (1 / 2) * ((x + b) / x + (x - c) / x) = 1 + (b - c) / (2 * x) :=
by
  sorry

end arithmetic_mean

namespace units_digit_of_7_pow_6_cubed

import Mathlib

-- Define the repeating cycle of unit digits for powers of 7
def unit_digit_of_power_of_7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0 -- This case is actually unreachable given the modulus operation

-- Define the main problem statement
theorem units_digit_of_7_pow_6_cubed : unit_digit_of_power_of_7 (6 ^ 3) = 1 :=
by
  sorry

end units_digit_of_7_pow_6_cubed

namespace only_zero_function_satisfies_inequality

import Mathlib

noncomputable def f (x : ℝ) : ℝ := sorry

theorem only_zero_function_satisfies_inequality (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  (∀ x y : ℝ, 0 < x → 0 < y →
    f x * f y ≥ (y^α / (x^α + x^β)) * (f x)^2 + (x^β / (y^α + y^β)) * (f y)^2) →
  ∀ x : ℝ, 0 < x → f x = 0 :=
sorry

end only_zero_function_satisfies_inequality

namespace isosceles_trapezoid_height

import Mathlib

/-- Given an isosceles trapezoid with area 100 and diagonals that are mutually perpendicular,
    we want to prove that the height of the trapezoid is 10. -/
theorem isosceles_trapezoid_height (BC AD h : ℝ) 
    (area_eq_100 : 100 = (1 / 2) * (BC + AD) * h)
    (height_eq_half_sum : h = (1 / 2) * (BC + AD)) :
    h = 10 :=
by
  sorry

end isosceles_trapezoid_height

namespace max_candies_theorem

import Mathlib

-- Defining constants: the number of students and the total number of candies.
def n : ℕ := 40
def T : ℕ := 200

-- Defining the condition that each student takes at least 2 candies.
def min_candies_per_student : ℕ := 2

-- Calculating the minimum total number of candies taken by 39 students.
def min_total_for_39_students := min_candies_per_student * (n - 1)

-- The maximum number of candies one student can take.
def max_candies_one_student_can_take := T - min_total_for_39_students

-- The statement to prove.
theorem max_candies_theorem : max_candies_one_student_can_take = 122 :=
by
  sorry

end max_candies_theorem

namespace jerky_remaining_after_giving_half

import Mathlib

-- Define the main conditions as variables
def days := 5
def initial_jerky := 40
def jerky_per_day := 1 + 1 + 2

-- Calculate total consumption
def total_consumption := jerky_per_day * days

-- Calculate remaining jerky
def remaining_jerky := initial_jerky - total_consumption

-- Calculate final jerky after giving half to her brother
def jerky_left := remaining_jerky / 2

-- Statement to be proved
theorem jerky_remaining_after_giving_half :
  jerky_left = 10 :=
by
  -- Proof will go here
  sorry

end jerky_remaining_after_giving_half

namespace smallest_x_for_non_prime_expression

import Mathlib

/-- The smallest positive integer x for which x^2 + x + 41 is not a prime number is 40. -/
theorem smallest_x_for_non_prime_expression : ∃ x : ℕ, x > 0 ∧ x^2 + x + 41 = 41 * 41 ∧ (∀ y : ℕ, 0 < y ∧ y < x → Prime (y^2 + y + 41)) := 
sorry

end smallest_x_for_non_prime_expression

namespace probability_no_correct_letter_for_7_envelopes

import Mathlib

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement (n - 1) + derangement (n - 2))

noncomputable def probability_no_correct_letter (n : ℕ) : ℚ :=
  derangement n / factorial n

theorem probability_no_correct_letter_for_7_envelopes :
  probability_no_correct_letter 7 = 427 / 1160 :=
by sorry

end probability_no_correct_letter_for_7_envelopes

namespace increasing_interval

import Mathlib

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x^2

theorem increasing_interval :
  ∃ a b : ℝ, (0 < a) ∧ (a < b) ∧ (b = 1/2) ∧ (∀ x : ℝ, a < x ∧ x < b → (deriv f x > 0)) :=
by
  sorry

end increasing_interval

namespace harper_water_duration

import Mathlib

theorem harper_water_duration
  (half_bottle_per_day : ℝ)
  (bottles_per_case : ℕ)
  (cost_per_case : ℝ)
  (total_spending : ℝ)
  (cases_bought : ℕ)
  (days_per_case : ℕ)
  (total_days : ℕ) :
  half_bottle_per_day = 1/2 →
  bottles_per_case = 24 →
  cost_per_case = 12 →
  total_spending = 60 →
  cases_bought = total_spending / cost_per_case →
  days_per_case = bottles_per_case * 2 →
  total_days = days_per_case * cases_bought →
  total_days = 240 :=
by
  -- The proof will be added here
  sorry

end harper_water_duration

namespace sandy_spent_on_repairs

import Mathlib

theorem sandy_spent_on_repairs (initial_cost : ℝ) (selling_price : ℝ) (gain_percent : ℝ) (repair_cost : ℝ) :
  initial_cost = 800 → selling_price = 1400 → gain_percent = 40 → selling_price = 1.4 * (initial_cost + repair_cost) → repair_cost = 200 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_spent_on_repairs

namespace problem

import Mathlib

variable (R S : Prop)

theorem problem (h1 : R → S) :
  ((¬S → ¬R) ∧ (¬R ∨ S)) :=
by
  sorry

end problem

namespace rectangular_prism_volume

import Mathlib

variables (a b c : ℝ)

theorem rectangular_prism_volume
  (h1 : a * b = 24)
  (h2 : b * c = 8)
  (h3 : c * a = 3) :
  a * b * c = 24 :=
by
  sorry

end rectangular_prism_volume

namespace mickey_horses_per_week

import Mathlib

variable (days_in_week : ℕ := 7)
variable (minnie_horses_per_day : ℕ := days_in_week + 3)
variable (mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6)

theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week

namespace prob1_prob2_max_area_prob3_circle_diameter

import Mathlib

-- Definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def line_through_center (x y : ℝ) : Prop := x - y - 3 = 0
def line_eq (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Problem 1: Line passes through the center of the circle
theorem prob1 (x y : ℝ) : line_through_center x y ↔ circle_eq x y :=
sorry

-- Problem 2: Maximum area of triangle CAB
theorem prob2_max_area (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 0 ∨ m = -6) :=
sorry

-- Problem 3: Circle with diameter AB passes through origin
theorem prob3_circle_diameter (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 1 ∨ m = -4) :=
sorry

end prob1_prob2_max_area_prob3_circle_diameter

namespace average_sitting_time_per_student

import Mathlib

def total_travel_time_in_minutes : ℕ := 152
def number_of_seats : ℕ := 5
def number_of_students : ℕ := 8

theorem average_sitting_time_per_student :
  (total_travel_time_in_minutes * number_of_seats) / number_of_students = 95 := 
by
  sorry

end average_sitting_time_per_student

namespace valid_integer_values_of_x

import Mathlib

theorem valid_integer_values_of_x (x : ℤ) 
  (h1 : 3 < x) (h2 : x < 10)
  (h3 : 5 < x) (h4 : x < 18)
  (h5 : -2 < x) (h6 : x < 9)
  (h7 : 0 < x) (h8 : x < 8) 
  (h9 : x + 1 < 9) : x = 6 ∨ x = 7 :=
by
  sorry

end valid_integer_values_of_x

namespace total_pencils_correct

import Mathlib

def pencils_per_child := 4
def num_children := 8
def total_pencils := pencils_per_child * num_children

theorem total_pencils_correct : total_pencils = 32 := by
  sorry

end total_pencils_correct

namespace sum_square_divisors_positive

import Mathlib

theorem sum_square_divisors_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b > 0 := 
by 
  sorry

end sum_square_divisors_positive

namespace ratio_of_voters

import Mathlib

theorem ratio_of_voters (V_X V_Y : ℝ) 
  (h1 : 0.62 * V_X + 0.38 * V_Y = 0.54 * (V_X + V_Y)) : V_X / V_Y = 2 :=
by
  sorry

end ratio_of_voters

namespace angles_relation

import Mathlib

/-- Given angles α and β from two right-angled triangles in a 3x3 grid such that α + β = 90°,
    prove that 2α + β = 90°. -/
theorem angles_relation (α β : ℝ) (h1 : α + β = 90) : 2 * α + β = 90 := by
  sorry

end angles_relation

namespace square_fold_distance

import Mathlib

noncomputable def distance_from_A (area : ℝ) (visible_equal : Bool) : ℝ :=
  if area = 18 ∧ visible_equal then 2 * Real.sqrt 6 else 0

theorem square_fold_distance (area : ℝ) (visible_equal : Bool) :
  area = 18 → visible_equal → distance_from_A area visible_equal = 2 * Real.sqrt 6 :=
by
  sorry

end square_fold_distance

namespace factorize_a3_minus_ab2

import Mathlib

theorem factorize_a3_minus_ab2 (a b: ℝ) : 
  a^3 - a * b^2 = a * (a + b) * (a - b) :=
by
  sorry

end factorize_a3_minus_ab2

namespace max_value_min_4x_y_4y_x2_5y2

import Mathlib

theorem max_value_min_4x_y_4y_x2_5y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, t = min (4 * x + y) (4 * y / (x^2 + 5 * y^2)) ∧ t ≤ 2 :=
by
  sorry

end max_value_min_4x_y_4y_x2_5y2

namespace sum_of_five_integers

import Mathlib

theorem sum_of_five_integers :
  ∃ (n m : ℕ), (n * (n + 1) = 336) ∧ ((m - 1) * m * (m + 1) = 336) ∧ ((n + (n + 1) + (m - 1) + m + (m + 1)) = 51) := 
sorry

end sum_of_five_integers

namespace train_speed_correct

import Mathlib

def train_length : ℝ := 250  -- length of the train in meters
def time_to_pass : ℝ := 18  -- time to pass a tree in seconds
def speed_of_train_km_hr : ℝ := 50  -- speed of the train in km/hr

theorem train_speed_correct :
  (train_length / time_to_pass) * (3600 / 1000) = speed_of_train_km_hr :=
by
  sorry

end train_speed_correct

namespace find_the_triplet

import Mathlib

theorem find_the_triplet (x y z : ℕ) (h : x + y + z = x * y * z) : (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
by
  sorry

end find_the_triplet

namespace fibonacci_series_sum

import Mathlib

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n-1) + fibonacci (n-2)

noncomputable def sum_fibonacci_fraction : ℚ :=
  ∑' (n : ℕ), (fibonacci n : ℚ) / (5^n : ℚ)

theorem fibonacci_series_sum : sum_fibonacci_fraction = 5 / 19 := by
  sorry

end fibonacci_series_sum

namespace rationalize_denominator_correct

import Mathlib

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalize_denominator_correct :
  (let A := 5
   let B := 49
   let C := 21
   (A + B + C) = 75) :=
by
  sorry

end rationalize_denominator_correct

namespace cat_toy_cost

import Mathlib

-- Define the conditions
def cost_of_cage : ℝ := 11.73
def total_cost_of_purchases : ℝ := 21.95

-- Define the proof statement
theorem cat_toy_cost : (total_cost_of_purchases - cost_of_cage) = 10.22 := by
  sorry

end cat_toy_cost

namespace find_b

import Mathlib

theorem find_b (b n : ℝ) (h_neg : b < 0) :
  (∀ x, x^2 + b * x + 1 / 4 = (x + n)^2 + 1 / 18) → b = - (Real.sqrt 7) / 3 :=
by
  sorry

end find_b

namespace m_div_x_eq_4_div_5

import Mathlib

variable (a b : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_ratio : a / b = 4 / 5)

def x := a * 1.25

def m := b * 0.80

theorem m_div_x_eq_4_div_5 : m / x = 4 / 5 :=
by
  sorry

end m_div_x_eq_4_div_5

namespace find_four_consecutive_odd_numbers

import Mathlib

noncomputable def four_consecutive_odd_numbers (a b c d : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧
  (a = b + 2 ∨ a = b - 2) ∧ (b = c + 2 ∨ b = c - 2) ∧ (c = d + 2 ∨ c = d - 2)

def numbers_sum_to_26879 (a b c d : ℤ) : Prop :=
  1 + (a + b + c + d) +
  (a * b + a * c + a * d + b * c + b * d + c * d) +
  (a * b * c + a * b * d + a * c * d + b * c * d) +
  (a * b * c * d) = 26879

theorem find_four_consecutive_odd_numbers (a b c d : ℤ) :
  four_consecutive_odd_numbers a b c d ∧ numbers_sum_to_26879 a b c d →
  ((a, b, c, d) = (9, 11, 13, 15) ∨ (a, b, c, d) = (-17, -15, -13, -11)) :=
by {
  sorry
}

end find_four_consecutive_odd_numbers

namespace discount_for_multiple_rides

import Mathlib

-- Definitions based on given conditions
def ferris_wheel_cost : ℝ := 2.0
def roller_coaster_cost : ℝ := 7.0
def coupon_value : ℝ := 1.0
def total_tickets_needed : ℝ := 7.0

-- The proof problem
theorem discount_for_multiple_rides : 
  (ferris_wheel_cost + roller_coaster_cost) - (total_tickets_needed - coupon_value) = 2.0 :=
by
  sorry

end discount_for_multiple_rides

namespace molecular_weight_of_compound

import Mathlib

noncomputable def atomic_weight_carbon : ℝ := 12.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.008
noncomputable def atomic_weight_oxygen : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def num_oxygen_atoms : ℕ := 1

noncomputable def molecular_weight (num_C num_H num_O : ℕ) : ℝ :=
  (num_C * atomic_weight_carbon) + (num_H * atomic_weight_hydrogen) + (num_O * atomic_weight_oxygen)

theorem molecular_weight_of_compound :
  molecular_weight num_carbon_atoms num_hydrogen_atoms num_oxygen_atoms = 65.048 :=
by
  sorry

end molecular_weight_of_compound

namespace perpendicular_lines_a_eq_1

import Mathlib

-- Definitions for the given conditions
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + y + 3 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (2 * a - 3) * y = 4

-- Condition that the lines are perpendicular
def perpendicular_lines (a : ℝ) : Prop := a + (2 * a - 3) = 0

-- Proof problem to be solved
theorem perpendicular_lines_a_eq_1 (a : ℝ) (h : perpendicular_lines a) : a = 1 :=
by
  sorry

end perpendicular_lines_a_eq_1

namespace find_expression_value

import Mathlib

theorem find_expression_value (x : ℝ) (h : x^2 - 3 * x - 1 = 0) : -3 * x^2 + 9 * x + 4 = 1 :=
by sorry

end find_expression_value

namespace gcf_3465_10780

import Mathlib

theorem gcf_3465_10780 : Nat.gcd 3465 10780 = 385 := by
  sorry

end gcf_3465_10780

namespace find_a8_a12_sum

import Mathlib

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a8_a12_sum
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end find_a8_a12_sum

namespace biology_marks

import Mathlib

theorem biology_marks 
  (e m p c : ℤ) 
  (avg : ℚ) 
  (marks_biology : ℤ)
  (h1 : e = 70) 
  (h2 : m = 63) 
  (h3 : p = 80)
  (h4 : c = 63)
  (h5 : avg = 68.2) 
  (h6 : avg * 5 = (e + m + p + c + marks_biology)) : 
  marks_biology = 65 :=
sorry

end biology_marks

namespace solution_set_f

import Mathlib

def f (x a b : ℝ) : ℝ := (x - 2) * (a * x + b)

theorem solution_set_f (a b : ℝ) (h1 : b = 2 * a) (h2 : 0 < a) :
  {x | f (2 - x) a b > 0} = {x | x < 0 ∨ 4 < x} :=
by
  sorry

end solution_set_f

namespace num_pos_int_values

import Mathlib

theorem num_pos_int_values
  (N : ℕ) 
  (h₀ : 0 < N)
  (h₁ : ∃ (k : ℕ), 0 < k ∧ 48 = k * (N + 3)) :
  ∃ (n : ℕ), n = 7 :=
sorry

end num_pos_int_values

namespace sequence_general_term

import Mathlib

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, 2 * a n = 3 * a (n + 1)) ∧ 
  (a 2 * a 5 = 8 / 27) ∧ 
  (∀ n, 0 < a n)

theorem sequence_general_term (a : ℕ → ℝ) (h : sequence_condition a) : 
  ∀ n, a n = (2 / 3)^(n - 2) :=
by 
  sorry

end sequence_general_term

namespace perimeter_C_correct

import Mathlib

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct

namespace tommy_profit

import Mathlib

noncomputable def total_cost : ℝ := 220 + 375 + 180 + 50 + 30

noncomputable def tomatoes_A : ℝ := 2 * (20 - 4)
noncomputable def oranges_A : ℝ := 2 * (10 - 2)

noncomputable def tomatoes_B : ℝ := 3 * (25 - 5)
noncomputable def oranges_B : ℝ := 3 * (15 - 3)
noncomputable def apples_B : ℝ := 3 * (5 - 1)

noncomputable def tomatoes_C : ℝ := 1 * (30 - 3)
noncomputable def apples_C : ℝ := 1 * (20 - 2)

noncomputable def revenue_A : ℝ := tomatoes_A * 5 + oranges_A * 4
noncomputable def revenue_B : ℝ := tomatoes_B * 6 + oranges_B * 4.5 + apples_B * 3
noncomputable def revenue_C : ℝ := tomatoes_C * 7 + apples_C * 3.5

noncomputable def total_revenue : ℝ := revenue_A + revenue_B + revenue_C

noncomputable def profit : ℝ := total_revenue - total_cost

theorem tommy_profit : profit = 179 :=
by
    sorry

end tommy_profit

namespace solution_set_equiv

import Mathlib

def solution_set (x : ℝ) : Prop := 2 * x - 6 < 0

theorem solution_set_equiv (x : ℝ) : solution_set x ↔ x < 3 := by
  sorry

end solution_set_equiv

namespace factorize1_factorize2_factorize3_factorize4

import Mathlib

-- 1. Factorize 3x - 12x^3
theorem factorize1 (x : ℝ) : 3 * x - 12 * x^3 = 3 * x * (1 - 2 * x) * (1 + 2 * x) := 
sorry

-- 2. Factorize 9m^2 - 4n^2
theorem factorize2 (m n : ℝ) : 9 * m^2 - 4 * n^2 = (3 * m + 2 * n) * (3 * m - 2 * n) := 
sorry

-- 3. Factorize a^2(x - y) + b^2(y - x)
theorem factorize3 (a b x y : ℝ) : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := 
sorry

-- 4. Factorize x^2 - 4xy + 4y^2 - 1
theorem factorize4 (x y : ℝ) : x^2 - 4 * x * y + 4 * y^2 - 1 = (x - y + 1) * (x - y - 1) := 
sorry

end factorize1_factorize2_factorize3_factorize4

namespace frequency_of_middle_group

import Mathlib

theorem frequency_of_middle_group (sample_size : ℕ) (x : ℝ) (h : sample_size = 160) (h_rel_freq : x = 0.2) 
  (h_relation : x = (1 / 4) * (10 * x)) : 
  sample_size * x = 32 :=
by
  sorry

end frequency_of_middle_group

namespace integral_value

import Mathlib

theorem integral_value : ∫ x in (1:ℝ)..(2:ℝ), (x^2 + 1) / x = (3 / 2) + Real.log 2 :=
by sorry

end integral_value

namespace volume_hemisphere_from_sphere

import Mathlib

theorem volume_hemisphere_from_sphere (r : ℝ) (V_sphere : ℝ) (V_hemisphere : ℝ) 
  (h1 : V_sphere = 150 * Real.pi) 
  (h2 : V_sphere = (4 / 3) * Real.pi * r^3) : 
  V_hemisphere = 75 * Real.pi :=
by
  sorry

end volume_hemisphere_from_sphere

namespace points_distance_le_sqrt5

import Mathlib

theorem points_distance_le_sqrt5 :
  ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (0 ≤ (points i).1 ∧ (points i).1 ≤ 4) ∧ (0 ≤ (points i).2 ∧ (points i).2 ≤ 3)) →
  ∃ (i j : Fin 6), i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end points_distance_le_sqrt5

namespace Lyle_friends_sandwich_juice

import Mathlib

/-- 
Lyle wants to buy himself and his friends a sandwich and a pack of juice. 
A sandwich costs $0.30 while a pack of juice costs $0.20. Given Lyle has $2.50, 
prove that he can buy sandwiches and juice for 4 of his friends.
-/
theorem Lyle_friends_sandwich_juice :
  let sandwich_cost := 0.30
  let juice_cost := 0.20
  let total_money := 2.50
  let total_cost_one_set := sandwich_cost + juice_cost
  let total_sets := total_money / total_cost_one_set
  total_sets - 1 = 4 :=
by
  sorry

end Lyle_friends_sandwich_juice

namespace area_of_triangle_intercepts

import Mathlib

theorem area_of_triangle_intercepts :
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  area = 168 :=
by
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  show area = 168
  sorry

end area_of_triangle_intercepts

namespace ratio_eliminated_to_remaining

import Mathlib

theorem ratio_eliminated_to_remaining (initial_racers : ℕ) (final_racers : ℕ)
  (eliminations_1st_segment : ℕ) (eliminations_2nd_segment : ℕ) :
  initial_racers = 100 →
  final_racers = 30 →
  eliminations_1st_segment = 10 →
  eliminations_2nd_segment = initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3 - final_racers →
  (eliminations_2nd_segment / (initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3)) = 1 / 2 :=
by
  sorry

end ratio_eliminated_to_remaining

namespace gcd_884_1071

import Mathlib

theorem gcd_884_1071 : Nat.gcd 884 1071 = 17 := by
  sorry

end gcd_884_1071

namespace contrapositive_proof

import Mathlib

theorem contrapositive_proof (m : ℕ) (h_pos : 0 < m) :
  (¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

end contrapositive_proof

namespace fraction_of_number_is_one_fifth

import Mathlib

theorem fraction_of_number_is_one_fifth (N : ℕ) (f : ℚ) 
    (hN : N = 90) 
    (h : 3 + (1 / 2) * (1 / 3) * f * N = (1 / 15) * N) : 
  f = 1 / 5 := by 
  sorry

end fraction_of_number_is_one_fifth

namespace midpoint_3d

import Mathlib

/-- Midpoint calculation in 3D space -/
theorem midpoint_3d (x1 y1 z1 x2 y2 z2 : ℝ) : 
  (x1, y1, z1) = (2, -3, 6) → 
  (x2, y2, z2) = (8, 5, -4) → 
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2) = (5, 1, 1) := 
by
  intros
  sorry

end midpoint_3d

namespace brownies_per_person

import Mathlib

-- Define the conditions as constants
def columns : ℕ := 6
def rows : ℕ := 3
def people : ℕ := 6

-- Define the total number of brownies
def total_brownies : ℕ := columns * rows

-- Define the theorem to be proved
theorem brownies_per_person : total_brownies / people = 3 :=
by sorry

end brownies_per_person

namespace daily_production

import Mathlib

-- Define the conditions
def bottles_per_case : ℕ := 9
def num_cases : ℕ := 8000

-- State the theorem with the question and the calculated answer
theorem daily_production : bottles_per_case * num_cases = 72000 :=
by
  sorry

end daily_production

namespace stratified_sampling_third_grade

import Mathlib

theorem stratified_sampling_third_grade 
  (N : ℕ) (N3 : ℕ) (S : ℕ) (x : ℕ)
  (h1 : N = 1600)
  (h2 : N3 = 400)
  (h3 : S = 80)
  (h4 : N3 / N = x / S) :
  x = 20 := 
by {
  sorry
}

end stratified_sampling_third_grade

namespace value_of_a5_max_sum_first_n_value

import Mathlib

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

end value_of_a5_max_sum_first_n_value

namespace trip_duration_is_6_hours

import Mathlib

def distance_1 := 55 * 4
def total_distance (A : ℕ) := distance_1 + 70 * A
def total_time (A : ℕ) := 4 + A
def average_speed (A : ℕ) := total_distance A / total_time A

theorem trip_duration_is_6_hours (A : ℕ) (h : 60 = average_speed A) : total_time A = 6 :=
by
  sorry

end trip_duration_is_6_hours

namespace cost_price_of_watch

import Mathlib

variable (CP : ℝ)
variable (SP_loss SP_gain : ℝ)
variable (h1 : SP_loss = CP * 0.725)
variable (h2 : SP_gain = CP * 1.125)
variable (h3 : SP_gain - SP_loss = 275)

theorem cost_price_of_watch : CP = 687.50 :=
by
  sorry

end cost_price_of_watch

namespace abc_divisibility

import Mathlib

theorem abc_divisibility (a b c : ℕ) (h₁ : a ∣ (b * c - 1)) (h₂ : b ∣ (c * a - 1)) (h₃ : c ∣ (a * b - 1)) : 
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 1 ∧ b = 1 ∧ ∃ n : ℕ, n ≥ 1 ∧ c = n) :=
by
  sorry

end abc_divisibility

namespace rhombus_perimeter

import Mathlib

-- Define the conditions for the rhombus
variable (d1 d2 : ℝ) (a b s : ℝ)

-- State the condition that the diagonals of a rhombus measure 24 cm and 10 cm
def diagonal_condition := (d1 = 24) ∧ (d2 = 10)

-- State the Pythagorean theorem for the lengths of half-diagonals
def pythagorean_theorem := a^2 + b^2 = s^2

-- State the relationship of diagonals bisecting each other at right angles
def bisect_condition := (a = d1 / 2) ∧ (b = d2 / 2)

-- State the definition of the perimeter for a rhombus
def perimeter (s : ℝ) : ℝ := 4 * s

-- The theorem we want to prove
theorem rhombus_perimeter : diagonal_condition d1 d2 →
                            bisect_condition d1 d2 a b →
                            pythagorean_theorem a b s →
                            perimeter s = 52 :=
by
  intros h1 h2 h3
  -- Proof would go here, but it is omitted
  sorry

end rhombus_perimeter

namespace combin_sum

import Mathlib

def combin (n m : ℕ) : ℕ := Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))

theorem combin_sum (n : ℕ) (h₁ : n = 99) : combin n 2 + combin n 3 = 161700 := by
  sorry

end combin_sum

namespace can_cut_rectangle_with_area_300_cannot_cut_rectangle_with_ratio_3_2

import Mathlib

-- Question and conditions
def side_length_of_square (A : ℝ) := A = 400
def area_of_rect (A : ℝ) := A = 300
def ratio_of_rect (length width : ℝ) := 3 * width = 2 * length

-- Prove that Li can cut a rectangle with area 300 from the square with area 400
theorem can_cut_rectangle_with_area_300 
  (a : ℝ) (h1 : side_length_of_square a)
  (length width : ℝ)
  (ha : a ^ 2 = 400) (har : length * width = 300) :
  length ≤ a ∧ width ≤ a :=
by
  sorry

-- Prove that Li cannot cut a rectangle with ratio 3:2 from the square
theorem cannot_cut_rectangle_with_ratio_3_2 (a : ℝ)
  (h1 : side_length_of_square a)
  (length width : ℝ)
  (har : area_of_rect (length * width))
  (hratio : ratio_of_rect length width)
  (ha : a ^ 2 = 400) :
  ¬(length ≤ a ∧ width ≤ a) :=
by
  sorry

end can_cut_rectangle_with_area_300_cannot_cut_rectangle_with_ratio_3_2

namespace mary_remaining_money

import Mathlib

variable (p : ℝ) -- p is the price per drink in dollars

def drinks_cost : ℝ := 3 * p
def medium_pizzas_cost : ℝ := 2 * (2 * p)
def large_pizza_cost : ℝ := 3 * p

def total_cost : ℝ := drinks_cost p + medium_pizzas_cost p + large_pizza_cost p

theorem mary_remaining_money : 
  30 - total_cost p = 30 - 10 * p := 
by
  sorry

end mary_remaining_money

namespace point_B_in_first_quadrant

import Mathlib

theorem point_B_in_first_quadrant 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : -b > 0) : 
  (a > 0) ∧ (b > 0) := 
by 
  sorry

end point_B_in_first_quadrant

namespace range_of_a

import Mathlib

-- Define the inequality condition
def condition (a : ℝ) (x : ℝ) : Prop := abs (a - 2 * x) > x - 1

-- Define the range for x
def in_range (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the main theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, in_range x → condition a x) ↔ (a < 2 ∨ 5 < a) := 
by
  sorry

end range_of_a

namespace seats_not_occupied

import Mathlib

def seats_per_row : ℕ := 8
def total_rows : ℕ := 12
def seat_utilization_ratio : ℚ := 3 / 4

theorem seats_not_occupied : 
  (seats_per_row * total_rows) - (seats_per_row * seat_utilization_ratio * total_rows) = 24 := 
by
  sorry

end seats_not_occupied

namespace julia_age_correct

import Mathlib

def julia_age_proof : Prop :=
  ∃ (j : ℚ) (m : ℚ), m = 15 * j ∧ m - j = 40 ∧ j = 20 / 7

theorem julia_age_correct : julia_age_proof :=
by
  sorry

end julia_age_correct

namespace passing_marks

import Mathlib

theorem passing_marks (T P : ℝ) (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) : P = 120 := 
by
  sorry

end passing_marks

namespace weight_of_rod

import Mathlib

theorem weight_of_rod (w₆ : ℝ) (h₁ : w₆ = 6.1) : 
  w₆ / 6 * 12 = 12.2 := by
  sorry

end weight_of_rod

namespace simplify_div

import Mathlib

theorem simplify_div : (27 * 10^12) / (9 * 10^4) = 3 * 10^8 := 
by
  sorry

end simplify_div

namespace num_boys_is_22

import Mathlib

variable (girls boys total_students : ℕ)

-- Conditions
axiom h1 : total_students = 41
axiom h2 : boys = girls + 3
axiom h3 : total_students = girls + boys

-- Goal: Prove that the number of boys is 22
theorem num_boys_is_22 : boys = 22 :=
by
  sorry

end num_boys_is_22

namespace relationship_between_roses_and_total_flowers

import Mathlib

variables (C V T R F : ℝ)
noncomputable def F_eq_64_42376521116678_percent_of_C := 
  C = 0.6442376521116678 * F

def V_eq_one_third_of_C := 
  V = (1 / 3) * C

def T_eq_one_ninth_of_C := 
  T = (1 / 9) * C

def F_eq_C_plus_V_plus_T_plus_R := 
  F = C + V + T + R

theorem relationship_between_roses_and_total_flowers (C V T R F : ℝ) 
    (h1 : C = 0.6442376521116678 * F)
    (h2 : V = 1 / 3 * C)
    (h3 : T = 1 / 9 * C)
    (h4 : F = C + V + T + R) :
    R = F - 13 / 9 * C := 
  by sorry

end relationship_between_roses_and_total_flowers

namespace sum_of_squares_with_signs

import Mathlib

theorem sum_of_squares_with_signs (n : ℤ) : 
  ∃ (k : ℕ) (s : Fin k → ℤ), (∀ i : Fin k, s i = 1 ∨ s i = -1) ∧ n = ∑ i : Fin k, s i * ((i + 1) * (i + 1)) := sorry

end sum_of_squares_with_signs

namespace ice_cream_total_volume

import Mathlib

/-- 
  The interior of a right, circular cone is 12 inches tall with a 3-inch radius at the opening.
  The interior of the cone is filled with ice cream.
  The cone has a hemisphere of ice cream exactly covering the opening of the cone.
  On top of this hemisphere, there is a cylindrical layer of ice cream of height 2 inches 
  and the same radius as the hemisphere (3 inches).
  Prove that the total volume of ice cream is 72π cubic inches.
-/
theorem ice_cream_total_volume :
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  V_cone + V_hemisphere + V_cylinder = 72 * Real.pi :=
by {
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  sorry
}

end ice_cream_total_volume

namespace mixture_alcohol_quantity

import Mathlib

theorem mixture_alcohol_quantity:
  ∀ (A W : ℝ), 
    A / W = 4 / 3 ∧ A / (W + 7) = 4 / 5 → A = 14 :=
by
  intros A W h
  sorry

end mixture_alcohol_quantity

namespace julia_drove_214_miles

import Mathlib

def daily_rate : ℝ := 29
def cost_per_mile : ℝ := 0.08
def total_cost : ℝ := 46.12

theorem julia_drove_214_miles :
  (total_cost - daily_rate) / cost_per_mile = 214 :=
by
  sorry

end julia_drove_214_miles

namespace children_absent

import Mathlib

theorem children_absent (A : ℕ) (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ) :
  total_children = 660 →
  bananas_per_child = 2 →
  extra_bananas_per_child = 2 →
  (total_children * bananas_per_child) = 1320 →
  ((total_children - A) * (bananas_per_child + extra_bananas_per_child)) = 1320 →
  A = 330 :=
by
  intros
  sorry

end children_absent

namespace expected_value_of_12_sided_die_is_6_5

import Mathlib

noncomputable def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  n * (a + l) / 2

noncomputable def expected_value_12_sided_die : ℚ :=
  (sum_arithmetic_series 12 1 12 : ℚ) / 12

theorem expected_value_of_12_sided_die_is_6_5 :
  expected_value_12_sided_die = 6.5 :=
by
  sorry

end expected_value_of_12_sided_die_is_6_5

namespace max_value_g_eq_3_in_interval

import Mathlib

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_g_eq_3_in_interval : 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3) ∧ (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3) :=
by
  sorry

end max_value_g_eq_3_in_interval

namespace sum_of_faces_of_rectangular_prism

import Mathlib

/-- Six positive integers are written on the faces of a rectangular prism.
Each vertex is labeled with the product of the three numbers on the faces adjacent to that vertex.
If the sum of the numbers on the eight vertices is equal to 720, 
prove that the sum of the numbers written on the faces is equal to 27. -/
theorem sum_of_faces_of_rectangular_prism (a b c d e f : ℕ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
(h_vertex_sum : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 720) :
  (a + d) + (b + e) + (c + f) = 27 :=
by
  sorry

end sum_of_faces_of_rectangular_prism

namespace equilateral_triangle_perimeter

import Mathlib

-- Definitions based on conditions
def equilateral_triangle_side : ℕ := 8

-- The statement we need to prove
theorem equilateral_triangle_perimeter : 3 * equilateral_triangle_side = 24 := by
  sorry

end equilateral_triangle_perimeter

namespace 

import Mathlib

-- Definitions for the problem conditions
def frame_thickness : ℕ := 3
def frame_area : ℕ := 108

-- Declaration that expresses the given conditions and the problem's conclusion
theorem painting_perimeter {w h : ℕ} (h_frame : (w + 2 * frame_thickness) * (h + 2 * frame_thickness) - w * h = frame_area) :
  2 * (w + h) = 24 :=
by
  sorry

end 

namespace commercial_break_duration

import Mathlib

theorem commercial_break_duration (n1 n2 t1 t2 : ℕ) (h1 : n1 = 3) (h2: t1 = 5) (h3 : n2 = 11) (h4 : t2 = 2) : 
  n1 * t1 + n2 * t2 = 37 := 
by 
  sorry

end commercial_break_duration

namespace k_bounds_inequality

import Mathlib

open Real

theorem k_bounds_inequality (k : ℝ) :
  (∀ x : ℝ, abs ((x^2 - k * x + 1) / (x^2 + x + 1)) < 3) ↔ -5 ≤ k ∧ k ≤ 1 := 
sorry

end k_bounds_inequality

namespace box_height_at_least_2_sqrt_15

import Mathlib

def box_height (x : ℝ) : ℝ := 2 * x
def surface_area (x : ℝ) : ℝ := 10 * x ^ 2

theorem box_height_at_least_2_sqrt_15 (x : ℝ) (h : ℝ) :
  h = box_height x →
  surface_area x ≥ 150 →
  h ≥ 2 * Real.sqrt 15 :=
by
  intros h_eq sa_ge_150
  sorry

end box_height_at_least_2_sqrt_15

namespace hiking_committee_selection

import Mathlib

def comb (n k : ℕ) : ℕ := n.choose k

theorem hiking_committee_selection :
  comb 10 3 = 120 :=
by
  sorry

end hiking_committee_selection

namespace right_triangle_midpoints_distances

import Mathlib

theorem right_triangle_midpoints_distances (a b : ℝ) 
  (hXON : 19^2 = a^2 + (b/2)^2)
  (hYOM : 22^2 = b^2 + (a/2)^2) :
  a^2 + b^2 = 676 :=
by
  sorry

end right_triangle_midpoints_distances

namespace exists_arith_prog_5_primes_exists_arith_prog_6_primes

import Mathlib

-- Define the condition of being an arithmetic progression
def is_arith_prog (seq : List ℕ) : Prop :=
  ∀ (i : ℕ), i < seq.length - 1 → seq.get! (i + 1) - seq.get! i = seq.get! 1 - seq.get! 0

-- Define the condition of being prime
def all_prime (seq : List ℕ) : Prop :=
  ∀ (n : ℕ), n ∈ seq → Nat.Prime n

-- The main statements
theorem exists_arith_prog_5_primes :
  ∃ (seq : List ℕ), seq.length = 5 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

theorem exists_arith_prog_6_primes :
  ∃ (seq : List ℕ), seq.length = 6 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

end exists_arith_prog_5_primes_exists_arith_prog_6_primes

namespace cos_double_angle

import Mathlib
open Real

theorem cos_double_angle (α : ℝ) (h : tan (α - π / 4) = 2) : cos (2 * α) = -4 / 5 := 
sorry

end cos_double_angle

namespace binomial_10_3_eq_120

import Mathlib

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120

namespace player_B_wins

import Mathlib

variable {R : Type*} [Ring R]

noncomputable def polynomial_game (n : ℕ) (f : Polynomial R) : Prop :=
  (f.degree = 2 * n) ∧ (∃ (a b : R) (x y : R), f.eval x = 0 ∨ f.eval y = 0)

theorem player_B_wins (n : ℕ) (f : Polynomial ℝ)
  (h1 : n ≥ 2)
  (h2 : f.degree = 2 * n) :
  polynomial_game n f :=
by
  sorry

end player_B_wins

namespace monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value

import Mathlib

-- Proof Problem I
noncomputable def f1 (x : ℝ) := x^2 + x - Real.log x

theorem monotonic_intervals_a1 : 
  (∀ x, 0 < x ∧ x < 1 / 2 → f1 x < 0) ∧ (∀ x, 1 / 2 < x → f1 x > 0) := 
sorry

-- Proof Problem II
noncomputable def f2 (x : ℝ) (a : ℝ) := x^2 + a * x - Real.log x

theorem decreasing_on_1_to_2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f2 x a ≤ 0) → a ≤ -7 / 2 :=
sorry

-- Proof Problem III
noncomputable def g (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem exists_a_for_minimum_value :
  ∃ a : ℝ, (∀ x, 0 < x ∧ x ≤ Real.exp 1 → g x a = 3) ∧ a = Real.exp 2 :=
sorry

end monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value

namespace simplify_polynomial

import Mathlib

variable {x : ℝ} -- Assume x is a real number

theorem simplify_polynomial :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 6 * x - 15) = x^2 + 2 * x + 10 :=
sorry

end simplify_polynomial

namespace positive_integer_product_divisibility

import Mathlib

theorem positive_integer_product_divisibility (x : ℕ → ℕ) (n p k : ℕ)
    (P : ℕ) (hx : ∀ i, 1 ≤ i → i ≤ n → x i < 2 * x 1)
    (hpos : ∀ i, 1 ≤ i → i ≤ n → 0 < x i)
    (hstrict : ∀ i j, 1 ≤ i → i < j → j ≤ n → x i < x j)
    (hn : 3 ≤ n)
    (hp : Nat.Prime p)
    (hk : 0 < k)
    (hP : P = ∏ i in Finset.range n, x (i + 1))
    (hdiv : p ^ k ∣ P) : 
  (P / p^k) ≥ Nat.factorial n := by
  sorry

end positive_integer_product_divisibility

namespace origin_movement_by_dilation

import Mathlib

/-- Given a dilation of the plane that maps a circle with radius 4 centered at (3,3) 
to a circle of radius 6 centered at (7,9), calculate the distance the origin (0,0)
moves under this transformation to be 0.5 * sqrt(10). -/
theorem origin_movement_by_dilation :
  let B := (3, 3)
  let B' := (7, 9)
  let radius_B := 4
  let radius_B' := 6
  let dilation_factor := radius_B' / radius_B
  let center_of_dilation := (-1, -3)
  let initial_distance := Real.sqrt ((-1)^2 + (-3)^2) 
  let moved_distance := dilation_factor * initial_distance
  moved_distance - initial_distance = 0.5 * Real.sqrt (10) := 
by
  sorry

end origin_movement_by_dilation

namespace time_to_walk_against_walkway_150

import Mathlib

def v_p := 4 / 3
def v_w := 2 - v_p
def distance := 100
def time_against_walkway := distance / (v_p - v_w)

theorem time_to_walk_against_walkway_150 :
  time_against_walkway = 150 := by
  -- Note: Proof goes here (not required)
  sorry

end time_to_walk_against_walkway_150

namespace total_heads

import Mathlib

theorem total_heads (D P : ℕ) (h1 : D = 9) (h2 : 4 * D + 2 * P = 42) : D + P = 12 :=
by
  sorry

end total_heads

namespace addition_of_two_odds_is_even_subtraction_of_two_odds_is_even

import Mathlib

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem addition_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a + b) :=
sorry

theorem subtraction_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a - b) :=
sorry

end addition_of_two_odds_is_even_subtraction_of_two_odds_is_even

namespace trig_identity_example

import Mathlib

theorem trig_identity_example :
  256 * (Real.sin (10 * Real.pi / 180)) * (Real.sin (30 * Real.pi / 180)) *
    (Real.sin (50 * Real.pi / 180)) * (Real.sin (70 * Real.pi / 180)) = 16 := by
  sorry

end trig_identity_example

namespace factorize_difference_of_squares

import Mathlib

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 81 = (a + 9) * (a - 9) :=
by
  sorry

end factorize_difference_of_squares

namespace probability_correct

import Mathlib

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct

namespace min_value_of_quadratic_function

import Mathlib

def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 2 * x - 5

theorem min_value_of_quadratic_function :
  ∃ x : ℝ, quadratic_function x = -1 :=
by
  sorry

end min_value_of_quadratic_function

namespace anthony_pencils

import Mathlib

def initial_pencils : ℝ := 56.0  -- Condition 1
def pencils_left : ℝ := 47.0     -- Condition 2
def pencils_given : ℝ := 9.0     -- Correct Answer

theorem anthony_pencils :
  initial_pencils - pencils_left = pencils_given :=
by
  sorry

end anthony_pencils

namespace part_I_part_II

import Mathlib

noncomputable def A : Set ℝ := {x | 2*x^2 - 5*x - 3 <= 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - (2*a + 1)) * (x - (a - 1)) < 0}

theorem part_I :
  (A ∪ B 0 = {x : ℝ | -1 < x ∧ x ≤ 3}) :=
by sorry

theorem part_II (a : ℝ) :
  (A ∩ B a = ∅) →
  (a ≤ -3/4 ∨ a ≥ 4) ∧ a ≠ -2 :=
by sorry


end part_I_part_II

namespace right_triangle_third_angle

import Mathlib

-- Define the problem
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Define the given angles
def is_right_angle (a : ℝ) : Prop := a = 90
def given_angle (b : ℝ) : Prop := b = 25

-- Define the third angle
def third_angle (a b c : ℝ) : Prop := a + b + c = 180

-- The theorem to prove 
theorem right_triangle_third_angle : ∀ (a b c : ℝ), 
  is_right_angle a → given_angle b → third_angle a b c → c = 65 :=
by
  intros a b c ha hb h_triangle
  sorry

end right_triangle_third_angle

namespace construction_company_total_weight

import Mathlib

noncomputable def total_weight_of_materials_in_pounds : ℝ :=
  let weight_of_concrete := 12568.3
  let weight_of_bricks := 2108 * 2.20462
  let weight_of_stone := 7099.5
  let weight_of_wood := 3778 * 2.20462
  let weight_of_steel := 5879 * (1 / 16)
  let weight_of_glass := 12.5 * 2000
  let weight_of_sand := 2114.8
  weight_of_concrete + weight_of_bricks + weight_of_stone + weight_of_wood + weight_of_steel + weight_of_glass + weight_of_sand

theorem construction_company_total_weight : total_weight_of_materials_in_pounds = 60129.72 :=
by
  sorry

end construction_company_total_weight

namespace fraction_of_groups_with_a_and_b

import Mathlib

/- Definitions based on the conditions -/
def total_persons : ℕ := 6
def group_size : ℕ := 3
def person_a : ℕ := 1  -- arbitrary assignment for simplicity
def person_b : ℕ := 2  -- arbitrary assignment for simplicity

/- Hypotheses based on conditions -/
axiom six_persons (n : ℕ) : n = total_persons
axiom divided_into_two_groups (grp_size : ℕ) : grp_size = group_size
axiom a_and_b_included (a b : ℕ) : a = person_a ∧ b = person_b

/- The theorem to prove -/
theorem fraction_of_groups_with_a_and_b
    (total_groups : ℕ := Nat.choose total_persons group_size)
    (groups_with_a_b : ℕ := Nat.choose 4 1) :
    groups_with_a_b / total_groups = 1 / 5 :=
by
    sorry

end fraction_of_groups_with_a_and_b

namespace car_b_speed

import Mathlib

theorem car_b_speed :
  ∀ (v : ℕ),
    (232 - 4 * v = 32) →
    v = 50 :=
  by
  sorry

end car_b_speed

namespace difference_divisible_by_9

import Mathlib

-- Define the integers a and b
variables (a b : ℤ)

-- Define the theorem statement
theorem difference_divisible_by_9 (a b : ℤ) : 9 ∣ ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) :=
sorry

end difference_divisible_by_9

namespace quadratic_has_real_roots

import Mathlib

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, x^2 - 4 * x - 2 * k + 8 = 0) ->
  k ≥ 2 :=
by
  sorry

end quadratic_has_real_roots

namespace largest_lcm

import Mathlib

def lcm_list : List ℕ := [
  Nat.lcm 15 3,
  Nat.lcm 15 5,
  Nat.lcm 15 9,
  Nat.lcm 15 10,
  Nat.lcm 15 12,
  Nat.lcm 15 15
]

theorem largest_lcm : List.maximum lcm_list = 60 := by
  sorry

end largest_lcm

namespace area_of_fourth_rectangle

import Mathlib

-- The conditions provided in the problem
variables (x y z w : ℝ)
variables (h1 : x * y = 24) (h2 : x * w = 12) (h3 : z * w = 8)

-- The problem statement with the conclusion
theorem area_of_fourth_rectangle :
  (∃ (x y z w : ℝ), ((x * y = 24 ∧ x * w = 12 ∧ z * w = 8) ∧ y * z = 16)) :=
sorry

end area_of_fourth_rectangle

namespace trigonometric_identity

import Mathlib

variable (α : Real)

theorem trigonometric_identity 
  (h : Real.sin (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (π / 3 - α) = Real.sqrt 3 / 3 :=
sorry

end trigonometric_identity

namespace probability_recruitment

import Mathlib

-- Definitions for conditions
def P_A : ℚ := 2/3
def P_A_not_and_B_not : ℚ := 1/12
def P_B_and_C : ℚ := 3/8

-- Independence of A, B, and C
axiom independence_A_B_C : ∀ {P_A P_B P_C : Prop}, 
  (P_A ∧ P_B ∧ P_C) → (P_A ∧ P_B) ∧ (P_A ∧ P_C) ∧ (P_B ∧ P_C)

-- Definition of probabilities of B and C
def P_B : ℚ := 3/4
def P_C : ℚ := 1/2

-- Main theorem
theorem probability_recruitment : 
  P_A = 2/3 ∧ 
  P_A_not_and_B_not = 1/12 ∧ 
  P_B_and_C = 3/8 ∧ 
  (∀ {P_A P_B P_C : Prop}, 
    (P_A ∧ P_B ∧ P_C) → (P_A ∧ P_B) ∧ (P_A ∧ P_C) ∧ (P_B ∧ P_C)) → 
  (P_B = 3/4 ∧ P_C = 1/2) ∧ 
  (2/3 * 3/4 * 1/2 + 1/3 * 3/4 * 1/2 + 2/3 * 1/4 * 1/2 + 2/3 * 3/4 * 1/2 = 17/24) := 
by sorry

end probability_recruitment

namespace madeline_flower_count

import Mathlib

theorem madeline_flower_count 
    (r w : ℕ) 
    (b_percent : ℝ) 
    (total : ℕ) 
    (h_r : r = 4)
    (h_w : w = 2)
    (h_b_percent : b_percent = 0.40)
    (h_total : r + w + (b_percent * total) = total) : 
    total = 10 :=
by 
    sorry

end madeline_flower_count

namespace base_r_representation_26_eq_32

import Mathlib

theorem base_r_representation_26_eq_32 (r : ℕ) : 
  26 = 3 * r + 6 → r = 8 :=
by
  sorry

end base_r_representation_26_eq_32

namespace books_not_sold

import Mathlib

variable {B : ℕ} -- Total number of books

-- Conditions
def two_thirds_books_sold (B : ℕ) : ℕ := (2 * B) / 3
def price_per_book : ℕ := 2
def total_amount_received : ℕ := 144
def remaining_books_sold : ℕ := 0
def two_thirds_by_price (B : ℕ) : ℕ := two_thirds_books_sold B * price_per_book

-- Main statement to prove
theorem books_not_sold (h : two_thirds_by_price B = total_amount_received) : (B / 3) = 36 :=
by
  sorry

end books_not_sold

namespace DiagonalsOfShapesBisectEachOther

import Mathlib

structure Shape where
  bisect_diagonals : Prop

def is_parallelogram (s : Shape) : Prop := s.bisect_diagonals
def is_rectangle (s : Shape) : Prop := s.bisect_diagonals
def is_rhombus (s : Shape) : Prop := s.bisect_diagonals
def is_square (s : Shape) : Prop := s.bisect_diagonals

theorem DiagonalsOfShapesBisectEachOther (s : Shape) :
  is_parallelogram s ∨ is_rectangle s ∨ is_rhombus s ∨ is_square s → s.bisect_diagonals := by
  sorry

end DiagonalsOfShapesBisectEachOther

namespace kenny_total_liquid

import Mathlib

def total_liquid (oil_per_recipe water_per_recipe : ℚ) (times : ℕ) : ℚ :=
  (oil_per_recipe + water_per_recipe) * times

theorem kenny_total_liquid :
  total_liquid 0.17 1.17 12 = 16.08 := by
  sorry

end kenny_total_liquid

namespace avg_price_pen_is_correct

import Mathlib

-- Definitions for the total numbers and expenses:
def number_of_pens : ℕ := 30
def number_of_pencils : ℕ := 75
def total_cost : ℕ := 630
def avg_price_pencil : ℝ := 2.00

-- Calculation of total cost for pencils and pens
def total_cost_pencils : ℝ := number_of_pencils * avg_price_pencil
def total_cost_pens : ℝ := total_cost - total_cost_pencils

-- Statement to prove:
theorem avg_price_pen_is_correct :
  total_cost_pens / number_of_pens = 16 :=
by
  sorry

end avg_price_pen_is_correct

namespace non_similar_triangles_with_arithmetic_angles

import Mathlib

theorem non_similar_triangles_with_arithmetic_angles : 
  ∃! (d : ℕ), d > 0 ∧ d ≤ 50 := 
sorry

end non_similar_triangles_with_arithmetic_angles

namespace move_right_by_three_units

import Mathlib

theorem move_right_by_three_units :
  (-1 + 3 = 2) :=
  by { sorry }

end move_right_by_three_units

namespace interval_of_increase

import Mathlib

noncomputable def u (x : ℝ) : ℝ := x^2 - 5*x + 6

def increasing_interval (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ interval → y ∈ interval → x < y → f x < f y

noncomputable def f (x : ℝ) : ℝ := Real.log (u x)

theorem interval_of_increase :
  increasing_interval f {x : ℝ | 3 < x} :=
sorry

end interval_of_increase

namespace max_areas_in_disk

import Mathlib

noncomputable def max_non_overlapping_areas (n : ℕ) : ℕ := 5 * n + 1

theorem max_areas_in_disk (n : ℕ) : 
  let disk_divided_by_2n_radii_and_two_secant_lines_areas  := (5 * n + 1)
  disk_divided_by_2n_radii_and_two_secant_lines_areas = max_non_overlapping_areas n := by sorry

end max_areas_in_disk

namespace prime_quadratic_root_range

import Mathlib

theorem prime_quadratic_root_range (p : ℕ) (hprime : Prime p) 
  (hroots : ∃ x1 x2 : ℤ, x1 * x2 = -580 * p ∧ x1 + x2 = p) : 20 < p ∧ p < 30 :=
by
  sorry

end prime_quadratic_root_range

namespace pears_value_equivalence

import Mathlib

-- Condition: $\frac{3}{4}$ of $16$ apples are worth $12$ pears
def apples_to_pears (a p : ℕ) : Prop :=
  (3 * 16 / 4 * a = 12 * p)

-- Question: How many pears (p) are equivalent in value to $\frac{2}{3}$ of $9$ apples?
def pears_equivalent_to_apples (p : ℕ) : Prop :=
  (2 * 9 / 3 * p = 6)

theorem pears_value_equivalence (p : ℕ) (a : ℕ) (h1 : apples_to_pears a p) (h2 : pears_equivalent_to_apples p) : 
  p = 6 :=
sorry

end pears_value_equivalence

namespace warehouse_bins_total

import Mathlib

theorem warehouse_bins_total (x : ℕ) (h1 : 12 * 20 + x * 15 = 510) : 12 + x = 30 :=
by
  sorry

end warehouse_bins_total

namespace quadratic_expression_positive_intervals

import Mathlib

noncomputable def quadratic_expression (x : ℝ) : ℝ := (x + 3) * (x - 1)
def interval_1 (x : ℝ) : Prop := x < (1 - Real.sqrt 13) / 2
def interval_2 (x : ℝ) : Prop := x > (1 + Real.sqrt 13) / 2

theorem quadratic_expression_positive_intervals (x : ℝ) :
  quadratic_expression x > 0 ↔ interval_1 x ∨ interval_2 x :=
by {
  sorry
}

end quadratic_expression_positive_intervals

namespace max_value_7a_9b

import Mathlib

theorem max_value_7a_9b 
    (r_1 r_2 r_3 a b : ℝ) 
    (h_eq : ∀ x, x^3 - x^2 + a * x - b = 0 → (x = r_1 ∨ x = r_2 ∨ x = r_3))
    (h_root_sum : r_1 + r_2 + r_3 = 1)
    (h_root_prod : r_1 * r_2 * r_3 = b)
    (h_root_sumprod : r_1 * r_2 + r_2 * r_3 + r_3 * r_1 = a)
    (h_bounds : ∀ i, i = r_1 ∨ i = r_2 ∨ i = r_3 → 0 < i ∧ i < 1) :
        7 * a - 9 * b ≤ 2 := 
sorry

end max_value_7a_9b

namespace phones_left_is_7500

import Mathlib

def last_year_production : ℕ := 5000
def this_year_production : ℕ := 2 * last_year_production
def sold_phones : ℕ := this_year_production / 4
def phones_left : ℕ := this_year_production - sold_phones

theorem phones_left_is_7500 : phones_left = 7500 :=
by
  sorry

end phones_left_is_7500

namespace angle_A_is_60_degrees

import Mathlib

theorem angle_A_is_60_degrees
  (a b c : ℝ) (A : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : 0 < A) (h3 : A < 180) : 
  A = 60 := 
  sorry

end angle_A_is_60_degrees

namespace largest_consecutive_positive_elements

import Mathlib

theorem largest_consecutive_positive_elements (a : ℕ → ℝ)
  (h₁ : ∀ n ≥ 2, a n = a (n-1) + a (n+2)) :
  ∃ m, m = 5 ∧ ∀ k < m, a k > 0 :=
sorry

end largest_consecutive_positive_elements

namespace jerry_task_duration

import Mathlib

def earnings_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7
def total_earnings : ℕ := 1400

theorem jerry_task_duration :
  (10 * 7 = 70) →
  (1400 / 40 = 35) →
  (70 / 35 = 2) →
  (total_earnings / earnings_per_task = (hours_per_day * days_per_week) / h) →
  h = 2 :=
by
  intros h1 h2 h3 h4
  -- proof steps (omitted)
  sorry

end jerry_task_duration

namespace childSupportOwed

import Mathlib

def annualIncomeBeforeRaise : ℕ := 30000
def yearsBeforeRaise : ℕ := 3
def raisePercentage : ℕ := 20
def annualIncomeAfterRaise (incomeBeforeRaise raisePercentage : ℕ) : ℕ :=
  incomeBeforeRaise + (incomeBeforeRaise * raisePercentage / 100)
def yearsAfterRaise : ℕ := 4
def childSupportPercentage : ℕ := 30
def amountPaid : ℕ := 1200

def calculateChildSupport (incomeYears : ℕ → ℕ → ℕ) (supportPercentage : ℕ) (years : ℕ) : ℕ :=
  (incomeYears years supportPercentage) * supportPercentage / 100 * years

def totalChildSupportOwed : ℕ :=
  (calculateChildSupport (λ _ _ => annualIncomeBeforeRaise) childSupportPercentage yearsBeforeRaise) +
  (calculateChildSupport (λ _ _ => annualIncomeAfterRaise annualIncomeBeforeRaise raisePercentage) childSupportPercentage yearsAfterRaise)

theorem childSupportOwed : totalChildSupportOwed - amountPaid = 69000 :=
by trivial

end childSupportOwed

namespace triangle_max_area_in_quarter_ellipse

import Mathlib

theorem triangle_max_area_in_quarter_ellipse (a b c : ℝ) (h : c^2 = a^2 - b^2) :
  ∃ (T_max : ℝ), T_max = b / 2 :=
by sorry

end triangle_max_area_in_quarter_ellipse

namespace original_denominator_is_21

import Mathlib

theorem original_denominator_is_21 (d : ℕ) : (3 + 6) / (d + 6) = 1 / 3 → d = 21 :=
by
  intros h
  sorry

end original_denominator_is_21

namespace mother_age_when_harry_born

import Mathlib

variable (harry_age father_age mother_age : ℕ)

-- Conditions
def harry_is_50 (harry_age : ℕ) : Prop := harry_age = 50
def father_is_24_years_older (harry_age father_age : ℕ) : Prop := father_age = harry_age + 24
def mother_younger_by_1_25_of_harry_age (harry_age father_age mother_age : ℕ) : Prop := mother_age = father_age - harry_age / 25

-- Proof Problem
theorem mother_age_when_harry_born (harry_age father_age mother_age : ℕ) 
  (h₁ : harry_is_50 harry_age) 
  (h₂ : father_is_24_years_older harry_age father_age)
  (h₃ : mother_younger_by_1_25_of_harry_age harry_age father_age mother_age) :
  mother_age - harry_age = 22 :=
by
  sorry

end mother_age_when_harry_born

namespace geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder

import Mathlib

-- Define geometric body type
inductive GeometricBody
  | rectangularPrism
  | cylinder

-- Define the condition where both front and left views are rectangles
def hasRectangularViews (body : GeometricBody) : Prop :=
  body = GeometricBody.rectangularPrism ∨ body = GeometricBody.cylinder

-- The theorem statement
theorem geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder (body : GeometricBody) :
  hasRectangularViews body :=
sorry

end geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder

namespace probability_exactly_two_heads_and_two_tails

import Mathlib

noncomputable def probability_two_heads_two_tails (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ n)

theorem probability_exactly_two_heads_and_two_tails
  (tosses : ℕ) (k : ℕ) (p : ℚ) (h_tosses : tosses = 4) (h_k : k = 2) (h_p : p = 1/2) :
  probability_two_heads_two_tails tosses k p = 3 / 8 := by
  sorry

end probability_exactly_two_heads_and_two_tails

namespace series_sum_is_6_over_5

import Mathlib

noncomputable def series_sum : ℝ := ∑' n : ℕ, if n % 4 == 0 then 1 / (4^(n/4)) else 
                                          if n % 4 == 1 then 1 / (2 * 4^(n/4)) else 
                                          if n % 4 == 2 then -1 / (4^(n/4) * 4^(1/2)) else 
                                          -1 / (2 * 4^(n/4 + 1/2))

theorem series_sum_is_6_over_5 : series_sum = 6 / 5 := 
  sorry

end series_sum_is_6_over_5

namespace g_triple_composition

import Mathlib

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n + 3

theorem g_triple_composition : g (g (g 3)) = 49 :=
by
  sorry

end g_triple_composition

namespace sampling_interval_is_9

import Mathlib

-- Conditions
def books_per_hour : ℕ := 362
def sampled_books_per_hour : ℕ := 40

-- Claim to prove
theorem sampling_interval_is_9 : (360 / sampled_books_per_hour = 9) := by
  sorry

end sampling_interval_is_9

namespace total_sum_of_money

import Mathlib

theorem total_sum_of_money (x : ℝ) (A B C D E : ℝ) (hA : A = x) (hB : B = 0.75 * x) 
  (hC : C = 0.60 * x) (hD : D = 0.50 * x) (hE1 : E = 0.40 * x) (hE2 : E = 84) : 
  A + B + C + D + E = 682.50 := 
by sorry

end total_sum_of_money

namespace petya_time_comparison

import Mathlib

variables (D V : ℝ) (hD_pos : D > 0) (hV_pos : V > 0)

theorem petya_time_comparison (hD_pos : D > 0) (hV_pos : V > 0) :
  (41 * D / (40 * V)) > (D / V) :=
by
  sorry

end petya_time_comparison

namespace sqrt_D_always_irrational

import Mathlib

-- Definitions for consecutive even integers and D
def is_consecutive_even (p q : ℤ) : Prop :=
  ∃ k : ℤ, p = 2 * k ∧ q = 2 * k + 2

def D (p q : ℤ) : ℤ :=
  p^2 + q^2 + p * q^2

-- The main statement to prove
theorem sqrt_D_always_irrational (p q : ℤ) (h : is_consecutive_even p q) :
  ¬ ∃ r : ℤ, r * r = D p q :=
sorry

end sqrt_D_always_irrational

namespace bags_of_chips_count

import Mathlib

theorem bags_of_chips_count :
  ∃ n : ℕ, n * 400 + 4 * 50 = 2200 ∧ n = 5 :=
by {
  sorry
}

end bags_of_chips_count

namespace subtraction_division

import Mathlib

theorem subtraction_division : 3550 - (1002 / 20.04) = 3499.9501 := by
  sorry

end subtraction_division

namespace remainder_of_max_6_multiple_no_repeated_digits

import Mathlib

theorem remainder_of_max_6_multiple_no_repeated_digits (M : ℕ) 
  (hM : ∃ n, M = 6 * n) 
  (h_unique_digits : ∀ (d : ℕ), d ∈ (M.digits 10) → (M.digits 10).count d = 1) 
  (h_max_M : ∀ (k : ℕ), (∃ n, k = 6 * n) ∧ (∀ (d : ℕ), d ∈ (k.digits 10) → (k.digits 10).count d = 1) → k ≤ M) :
  M % 100 = 78 := 
sorry

end remainder_of_max_6_multiple_no_repeated_digits

namespace maximum_profit

import Mathlib

def cost_price_per_unit : ℕ := 40
def initial_selling_price_per_unit : ℕ := 50
def units_sold_per_month : ℕ := 210
def price_increase_effect (x : ℕ) : ℕ := units_sold_per_month - 10 * x
def profit_function (x : ℕ) : ℕ := (price_increase_effect x) * (initial_selling_price_per_unit + x - cost_price_per_unit)

theorem maximum_profit :
  profit_function 5 = 2400 ∧ profit_function 6 = 2400 :=
by
  sorry

end maximum_profit

namespace smallest_unreachable_integer

import Mathlib

/-- The smallest positive integer that cannot be expressed in the form (2^a - 2^b) / (2^c - 2^d) where a, b, c, d are non-negative integers is 11. -/
theorem smallest_unreachable_integer : 
  ∀ (a b c d : ℕ), 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
  ∃ (n : ℕ), n = 11 ∧ ¬ ∃ (a b c d : ℕ), (2^a - 2^b) / (2^c - 2^d) = n :=
by
  sorry

end smallest_unreachable_integer

namespace find_common_difference

import Mathlib

def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, S n = n * a 1 + (n * (n - 1)) / 2 * (a 1 - a 0)

noncomputable def quadratic_roots (c : ℚ) (x1 x2 : ℚ) : Prop :=
2 * x1^2 - 12 * x1 + c = 0 ∧ 2 * x2^2 - 12 * x2 + c = 0

theorem find_common_difference
  (a : ℕ → ℚ) (S : ℕ → ℚ) (c : ℚ)
  (h_arith_seq: is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_roots : quadratic_roots c (a 3) (a 7))
  (h_S13 : S 13 = c) :
  (a 1 - a 0 = -3/2) ∨ (a 1 - a 0 = -7/4) :=
sorry

end find_common_difference

namespace total_circles

import Mathlib

theorem total_circles (n : ℕ) (h1 : ∀ k : ℕ, k = n + 14 → n^2 = (k * (k + 1) / 2)) : 
  n = 35 → n^2 = 1225 :=
by
  sorry

end total_circles

namespace distance_to_workplace

import Mathlib

def driving_speed : ℕ := 40
def driving_time : ℕ := 3
def total_distance := driving_speed * driving_time
def one_way_distance := total_distance / 2

theorem distance_to_workplace : one_way_distance = 60 := by
  sorry

end distance_to_workplace

namespace binomial_12_6_eq_924

import Mathlib.Data.Nat.Choose.Basic

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924

namespace ratio_of_saramago_readers

import Mathlib

theorem ratio_of_saramago_readers 
  (W : ℕ) (S K B N : ℕ)
  (h1 : W = 42)
  (h2 : K = W / 6)
  (h3 : B = 3)
  (h4 : N = (S - B) - 1)
  (h5 : W = (S - B) + (K - B) + B + N) :
  S / W = 1 / 2 :=
by
  sorry

end ratio_of_saramago_readers

namespace step_count_initial

import Mathlib

theorem step_count_initial :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (11 * y - x = 64) ∧ (10 * x + y = 26) :=
by
  sorry

end step_count_initial

namespace planes_are_perpendicular

import Mathlib

-- Define the normal vectors
def N1 : List ℝ := [2, 3, -4]
def N2 : List ℝ := [5, -2, 1]

-- Define the dot product function
def dotProduct (v1 v2 : List ℝ) : ℝ :=
  List.zipWith (fun a b => a * b) v1 v2 |>.sum

-- State the theorem
theorem planes_are_perpendicular :
  dotProduct N1 N2 = 0 :=
by
  sorry

end planes_are_perpendicular

namespace find_theta_in_interval

import Mathlib

variable (θ : ℝ)

def angle_condition (θ : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ (x^3 * Real.cos θ - x * (1 - x) + (1 - x)^3 * Real.tan θ > 0)

theorem find_theta_in_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → angle_condition θ x) →
  0 < θ ∧ θ < Real.pi / 2 :=
by
  sorry

end find_theta_in_interval

namespace number_x_is_divided_by

import Mathlib

-- Define the conditions
variable (x y n : ℕ)
variable (cond1 : x = n * y + 4)
variable (cond2 : 2 * x = 8 * 3 * y + 3)
variable (cond3 : 13 * y - x = 1)

-- Define the statement to be proven
theorem number_x_is_divided_by : n = 11 :=
by
  sorry

end number_x_is_divided_by

namespace wood_burned_afternoon

import Mathlib

theorem wood_burned_afternoon (burned_morning burned_afternoon bundles_start bundles_end : ℕ) 
  (h_burned_morning : burned_morning = 4)
  (h_bundles_start : bundles_start = 10) 
  (h_bundles_end : bundles_end = 3)
  (total_burned : bundles_start - bundles_end = burned_morning + burned_afternoon) :
  burned_afternoon = 3 :=
by {
  -- Proof placeholder
  sorry
}

end wood_burned_afternoon

namespace maximum_abc_827

import Mathlib

noncomputable def maximum_abc (a b c : ℝ) := (a * b * c)

theorem maximum_abc_827 (a b c : ℝ) 
  (h1: a > 0) 
  (h2: b > 0) 
  (h3: c > 0) 
  (h4: (a * b) + c = (a + c) * (b + c)) 
  (h5: a + b + c = 2) : 
  maximum_abc a b c = 8 / 27 := 
by 
  sorry

end maximum_abc_827

namespace grid_covering_impossible

import Mathlib

theorem grid_covering_impossible :
  ∀ (x y : ℕ), x + y = 19 → 6 * x + 7 * y = 132 → False :=
by
  intros x y h1 h2
  -- Proof would go here.
  sorry

end grid_covering_impossible

namespace abs_eq_1_solution_set

import Mathlib

theorem abs_eq_1_solution_set (x : ℝ) : (|x| + |x + 1| = 1) ↔ (x ∈ Set.Icc (-1 : ℝ) 0) := by
  sorry

end abs_eq_1_solution_set

namespace domain_of_f

import Mathlib

-- Define the conditions
def sqrt_domain (x : ℝ) : Prop := x + 1 ≥ 0
def log_domain (x : ℝ) : Prop := 3 - x > 0

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.log (3 - x)

-- Statement of the theorem
theorem domain_of_f : ∀ x, sqrt_domain x ∧ log_domain x ↔ -1 ≤ x ∧ x < 3 := by
  sorry

end domain_of_f

namespace dice_probability_five_or_six

import Mathlib

theorem dice_probability_five_or_six :
  let outcomes := 36
  let favorable := 18
  let probability := favorable / outcomes
  probability = 1 / 2 :=
by
  sorry

end dice_probability_five_or_six

namespace eight_bees_have_48_legs

import Mathlib

  def legs_per_bee : ℕ := 6
  def number_of_bees : ℕ := 8
  def total_legs : ℕ := 48

  theorem eight_bees_have_48_legs :
    number_of_bees * legs_per_bee = total_legs :=
  by
    sorry
  
end eight_bees_have_48_legs

namespace ratio_of_parallel_vectors

import Mathlib

theorem ratio_of_parallel_vectors (m n : ℝ) 
  (h1 : ∃ k : ℝ, (m, 1, 3) = (k * 2, k * n, k)) : (m / n) = 18 :=
by
  sorry

end ratio_of_parallel_vectors

namespace constant_term_of_expansion

import Mathlib

noncomputable def constant_term := 
  (20: ℕ) * (216: ℕ) * (1/27: ℚ) = (160: ℕ)

theorem constant_term_of_expansion : constant_term :=
  by sorry

end constant_term_of_expansion

namespace evaluate_expression

import Mathlib

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem evaluate_expression :
  (4 / log_base 5 (2500^3) + 2 / log_base 2 (2500^3) = 1 / 3) := by
  sorry

end evaluate_expression

namespace best_scrap_year_limit

import Mathlib

theorem best_scrap_year_limit
    (purchase_cost : ℝ)
    (annual_expenses : ℝ)
    (base_maintenance_cost : ℝ)
    (annual_maintenance_increase : ℝ)
    (n : ℕ)
    (n_min_avg : ℝ) :
    purchase_cost = 150000 ∧
    annual_expenses = 15000 ∧
    base_maintenance_cost = 3000 ∧
    annual_maintenance_increase = 3000 ∧
    n = 10 →
    n_min_avg = 10 := by
  sorry

end best_scrap_year_limit

namespace geometric_series_first_term

import Mathlib

theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 18) 
  (h2 : a^2 / (1 - r^2) = 72) : 
  a = 7.2 :=
by
  sorry

end geometric_series_first_term

namespace trigonometric_identity

import Mathlib

theorem trigonometric_identity (x : ℝ) (h : Real.tan (3 * π - x) = 2) :
    (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 := by
  sorry

end trigonometric_identity

namespace problem

import Mathlib

theorem problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1 / 2) :
  (1 - x) / (1 + x) * (1 - y) / (1 + y) * (1 - z) / (1 + z) ≥ 1 / 3 :=
by
  sorry

end problem

namespace proof_problem

import Mathlib

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem proof_problem (a b : ℝ) :
  f 2016 a b + f (-2016) a b + f' 2017 a b - f' (-2017) a b = 8 := by
  sorry

end proof_problem

namespace james_marbles_left

import Mathlib

def total_initial_marbles : Nat := 28
def marbles_in_bag_A : Nat := 4
def marbles_in_bag_B : Nat := 6
def marbles_in_bag_C : Nat := 2
def marbles_in_bag_D : Nat := 8
def marbles_in_bag_E : Nat := 4
def marbles_in_bag_F : Nat := 4

theorem james_marbles_left : total_initial_marbles - marbles_in_bag_D = 20 := by
  -- James has 28 marbles initially.
  -- He gives away Bag D which has 8 marbles.
  -- 28 - 8 = 20
  sorry

end james_marbles_left

namespace question_true

import Mathlib
noncomputable def a := (1/2) * Real.cos (7 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (7 * Real.pi / 180)
noncomputable def b := (2 * Real.tan (12 * Real.pi / 180)) / (1 + Real.tan (12 * Real.pi / 180)^2)
noncomputable def c := Real.sqrt ((1 - Real.cos (44 * Real.pi / 180)) / 2)

theorem question_true :
  b > a ∧ a > c :=
by
  sorry

end question_true

namespace max_pies_without_ingredients

import Mathlib

theorem max_pies_without_ingredients :
  let total_pies := 36
  let chocolate_pies := total_pies / 3
  let marshmallow_pies := total_pies / 4
  let cayenne_pies := total_pies / 2
  let soy_nuts_pies := total_pies / 8
  let max_ingredient_pies := max (max chocolate_pies marshmallow_pies) (max cayenne_pies soy_nuts_pies)
  total_pies - max_ingredient_pies = 18 :=
by
  sorry

end max_pies_without_ingredients

namespace krista_driving_hours_each_day

import Mathlib

-- Define the conditions as constants
def road_trip_days : ℕ := 3
def jade_hours_per_day : ℕ := 8
def total_hours : ℕ := 42

-- Define the function to calculate Krista's hours per day
noncomputable def krista_hours_per_day : ℕ :=
  (total_hours - road_trip_days * jade_hours_per_day) / road_trip_days

-- State the theorem to prove Krista drove 6 hours each day
theorem krista_driving_hours_each_day : krista_hours_per_day = 6 := by
  sorry

end krista_driving_hours_each_day

namespace nth_inequality

import Mathlib

theorem nth_inequality (n : ℕ) : 
  (∑ k in Finset.range (2^(n+1) - 1), (1/(k+1))) > (↑(n+1) / 2) := 
by sorry

end nth_inequality

namespace only_function

import Mathlib

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, divides (f m + f n) (m + n)

theorem only_function (f : ℕ → ℕ) (h : satisfies_condition f) : f = id :=
by
  -- Proof goes here.
  sorry

end only_function

namespace relationship_A_B

import Mathlib

variable (x y : ℝ)

noncomputable def A : ℝ := (x + y) / (1 + x + y)

noncomputable def B : ℝ := (x / (1 + x)) + (y / (1 + y))

theorem relationship_A_B (hx : 0 < x) (hy : 0 < y) : A x y < B x y := sorry

end relationship_A_B

namespace miley_total_cost

import Mathlib

-- Define the cost per cellphone
def cost_per_cellphone : ℝ := 800

-- Define the number of cellphones
def number_of_cellphones : ℝ := 2

-- Define the discount rate
def discount_rate : ℝ := 0.05

-- Define the total cost without discount
def total_cost_without_discount : ℝ := cost_per_cellphone * number_of_cellphones

-- Define the discount amount
def discount_amount : ℝ := total_cost_without_discount * discount_rate

-- Define the total cost with discount
def total_cost_with_discount : ℝ := total_cost_without_discount - discount_amount

-- Prove that the total amount Miley paid is $1520
theorem miley_total_cost : total_cost_with_discount = 1520 := by
  sorry

end miley_total_cost

namespace survival_rate_is_98

import Mathlib

def total_flowers := 150
def unsurviving_flowers := 3
def surviving_flowers := total_flowers - unsurviving_flowers

theorem survival_rate_is_98 : (surviving_flowers : ℝ) / total_flowers * 100 = 98 := by
  sorry

end survival_rate_is_98

namespace total_surface_area_of_three_face_painted_cubes

import Mathlib

def cube_side_length : ℕ := 9
def small_cube_side_length : ℕ := 1
def num_small_cubes_with_three_faces_painted : ℕ := 8
def surface_area_of_each_painted_face : ℕ := 6

theorem total_surface_area_of_three_face_painted_cubes :
  num_small_cubes_with_three_faces_painted * surface_area_of_each_painted_face = 48 := by
  sorry

end total_surface_area_of_three_face_painted_cubes

namespace final_lives_equals_20

import Mathlib

def initial_lives : ℕ := 30
def lives_lost : ℕ := 12
def bonus_lives : ℕ := 5
def penalty_lives : ℕ := 3

theorem final_lives_equals_20 : (initial_lives - lives_lost + bonus_lives - penalty_lives) = 20 :=
by 
  sorry

end final_lives_equals_20

namespace find_unit_price_B

import Mathlib

variable (x : ℕ)

def unit_price_B := x
def unit_price_A := x + 50

theorem find_unit_price_B (h : (2000 / unit_price_A x = 1500 / unit_price_B x)) : unit_price_B x = 150 :=
by
  sorry

end find_unit_price_B

namespace problem1_problem2

import Mathlib

-- Define the conditions
variable (a x : ℝ)
variable (h_gt_zero : x > 0) (a_gt_zero : a > 0)

-- Problem 1: Prove that 0 < x ≤ 300
theorem problem1 (h: 12 * (500 - x) * (1 + 0.005 * x) ≥ 12 * 500) : 0 < x ∧ x ≤ 300 := 
sorry

-- Problem 2: Prove that 0 < a ≤ 5.5 given the conditions
theorem problem2 (h1 : 12 * (a - 13 / 1000 * x) * x ≤ 12 * (500 - x) * (1 + 0.005 * x))
                (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := 
sorry

end problem1_problem2

namespace Chloe_final_points

import Mathlib

-- Define the points scored (or lost) in each round
def round1_points : ℤ := 40
def round2_points : ℤ := 50
def round3_points : ℤ := 60
def round4_points : ℤ := 70
def round5_points : ℤ := -4
def round6_points : ℤ := 80
def round7_points : ℤ := -6

-- Statement to prove: Chloe's total points at the end of the game
theorem Chloe_final_points : 
  round1_points + round2_points + round3_points + round4_points + round5_points + round6_points + round7_points = 290 :=
by
  sorry

end Chloe_final_points

namespace inequality_solution

import Mathlib

theorem inequality_solution (a x : ℝ) : 
  (x^2 - (a + 1) * x + a) ≤ 0 ↔ 
  (a > 1 → (1 ≤ x ∧ x ≤ a)) ∧ 
  (a = 1 → x = 1) ∧ 
  (a < 1 → (a ≤ x ∧ x ≤ 1)) :=
by 
  sorry

end inequality_solution

namespace ram_ravi_selected_probability

import Mathlib

noncomputable def probability_both_selected : ℝ := 
  let probability_ram_80 := (1 : ℝ) / 7
  let probability_ravi_80 := (1 : ℝ) / 5
  let probability_both_80 := probability_ram_80 * probability_ravi_80
  let num_applicants := 200
  let num_spots := 4
  let probability_single_selection := (num_spots : ℝ) / (num_applicants : ℝ)
  let probability_both_selected_given_80 := probability_single_selection * probability_single_selection
  probability_both_80 * probability_both_selected_given_80

theorem ram_ravi_selected_probability :
  probability_both_selected = 1 / 87500 := 
by
  sorry

end ram_ravi_selected_probability

namespace two_digit_numbers

import Mathlib

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem two_digit_numbers (a b : ℕ) (h1 : is_digit a) (h2 : is_digit b) 
  (h3 : a ≠ b) (h4 : (a + b) = 11) : 
  (∃ n m : ℕ, (n = 10 * a + b) ∧ (m = 10 * b + a) ∧ (∃ k : ℕ, (10 * a + b)^2 - (10 * b + a)^2 = k^2)) := 
sorry

end two_digit_numbers

namespace initial_contribution_amount

import Mathlib

variable (x : ℕ)
variable (workers : ℕ := 1200)
variable (total_with_extra_contribution: ℕ := 360000)
variable (extra_contribution_each: ℕ := 50)

theorem initial_contribution_amount :
  (workers * x = total_with_extra_contribution - workers * extra_contribution_each) →
  workers * x = 300000 :=
by
  intro h
  sorry

end initial_contribution_amount

namespace solve_eq

import Mathlib

theorem solve_eq (x y : ℕ) (h : x^2 - 2 * x * y + y^2 + 5 * x + 5 * y = 1500) :
  (x = 150 ∧ y = 150) ∨ (x = 150 ∧ y = 145) ∨ (x = 145 ∧ y = 135) ∨
  (x = 135 ∧ y = 120) ∨ (x = 120 ∧ y = 100) ∨ (x = 100 ∧ y = 75) ∨
  (x = 75 ∧ y = 45) ∨ (x = 45 ∧ y = 10) ∨ (x = 145 ∧ y = 150) ∨
  (x = 135 ∧ y = 145) ∨ (x = 120 ∧ y = 135) ∨ (x = 100 ∧ y = 120) ∨
  (x = 75 ∧ y = 100) ∨ (x = 45 ∧ y = 75) ∨ (x = 10 ∧ y = 45) :=
sorry

end solve_eq

namespace total_number_of_members

import Mathlib

-- Define the basic setup
def committees := Fin 5
def members := {m : Finset committees // m.card = 2}

-- State the theorem
theorem total_number_of_members :
  (∃ s : Finset members, s.card = 10) :=
sorry

end total_number_of_members

namespace lambda_range

import Mathlib  -- To avoid import errors

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 1 else
  sequence_a (n - 1) / (sequence_a (n - 1) + 2)

noncomputable def sequence_b (lambda : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then -3/2 * lambda else
  (n - 2 * lambda) * (1 / sequence_a (n - 1) + 1)

def is_monotonically_increasing (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → seq (n+1) > seq n

theorem lambda_range (lambda : ℝ) (hn : is_monotonically_increasing (sequence_b lambda)) : lambda < 4/5 := sorry

end lambda_range

namespace horner_example

import Mathlib

def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldr (λ a acc => a + x * acc) 0

theorem horner_example : horner [12, 35, -8, 79, 6, 5, 3] (-4) = 220 := by
  sorry

end horner_example

namespace neg_distance_represents_west

import Mathlib

def represents_east (distance : Int) : Prop :=
  distance > 0

def represents_west (distance : Int) : Prop :=
  distance < 0

theorem neg_distance_represents_west (pos_neg : represents_east 30) :
  represents_west (-50) :=
by
  sorry

end neg_distance_represents_west

namespace books_of_jason

import Mathlib

theorem books_of_jason (M J : ℕ) (hM : M = 42) (hTotal : M + J = 60) : J = 18 :=
by
  sorry

end books_of_jason

namespace max_possible_x

import Mathlib

noncomputable section

def tan_deg (x : ℕ) : ℝ := Real.tan (x * Real.pi / 180)

theorem max_possible_x (x y : ℕ) (h₁ : tan_deg x - tan_deg y = 1 + tan_deg x * tan_deg y)
  (h₂ : tan_deg x * tan_deg y = 1) (h₃ : x = 98721) : x = 98721 := sorry

end max_possible_x

namespace geometric_sequence_seventh_term

import Mathlib

theorem geometric_sequence_seventh_term
  (a r : ℝ)
  (h1 : a * r^4 = 16)
  (h2 : a * r^10 = 4) :
  a * r^6 = 4 * (2^(2/3)) :=
by
  sorry

end geometric_sequence_seventh_term

namespace find_g7

import Mathlib

namespace ProofProblem

variable (g : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, g (x + y) = g x + g y)
variable (h2 : g 6 = 8)

theorem find_g7 : g 7 = 28 / 3 := by
  sorry

end ProofProblem

end find_g7

namespace part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b

import Mathlib

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (2 * (Real.cos (x/2))^2 + Real.sin x) + b

theorem part1_monotonically_increasing_interval (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, x ∈ Set.Icc (-3 * Real.pi / 4 + 2 * k * Real.pi) (Real.pi / 4 + 2 * k * Real.pi) ->
    f x <= f (x + Real.pi) :=
sorry

theorem part1_symmetry_axis (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, f x = f (2 * (Real.pi / 4 + k * Real.pi) - x) :=
sorry

theorem part2_find_a_b (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi)
  (h2 : ∃ (a b : ℝ), ∀ x, x ∈ Set.Icc 0 Real.pi → (a > 0 ∧ 3 ≤ f a b x ∧ f a b x ≤ 4)) :
  (1 - Real.sqrt 2 < a ∧ a < 1 + Real.sqrt 2) ∧ b = 3 :=
sorry

end part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b

namespace problem_remainder_3

import Mathlib

theorem problem_remainder_3 :
  88 % 5 = 3 :=
by
  sorry

end problem_remainder_3

namespace average_speed_of_car

import Mathlib

/-- The car's average speed given it travels 65 km in the first hour and 45 km in the second hour. -/
theorem average_speed_of_car (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 65) (h2 : d2 = 45) (h3 : t = 2) :
  (d1 + d2) / t = 55 :=
by
  sorry

end average_speed_of_car

namespace largest_number_is_correct

import Mathlib

theorem largest_number_is_correct (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 3) : c = 33.25 :=
by
  sorry

end largest_number_is_correct

namespace playgroup_count

import Mathlib

-- Definitions based on the conditions
def total_people (girls boys parents : ℕ) := girls + boys + parents
def playgroups (total size_per_group : ℕ) := total / size_per_group

-- Statement of the problem
theorem playgroup_count (girls boys parents size_per_group : ℕ)
  (h_girls : girls = 14)
  (h_boys : boys = 11)
  (h_parents : parents = 50)
  (h_size_per_group : size_per_group = 25) :
  playgroups (total_people girls boys parents) size_per_group = 3 :=
by {
  -- This is just the statement, the proof is skipped with sorry
  sorry
}

end playgroup_count

namespace find_c

import Mathlib

-- Define the polynomial P(x)
def P (c : ℚ) (x : ℚ) : ℚ := x^3 + 4 * x^2 + c * x + 20

-- Given that x - 3 is a factor of P(x), prove that c = -83/3
theorem find_c (c : ℚ) (h : P c 3 = 0) : c = -83 / 3 :=
by
  sorry

end find_c

namespace triangle_height_and_segments

import Mathlib

-- Define the sides of the triangle
noncomputable def a : ℝ := 13
noncomputable def b : ℝ := 14
noncomputable def c : ℝ := 15

-- Define the height h and the segments m and 15 - m
noncomputable def m : ℝ := 6.6
noncomputable def h : ℝ := 11.2
noncomputable def base_segment_left : ℝ := m
noncomputable def base_segment_right : ℝ := c - m

-- The height and segments calculation theorem
theorem triangle_height_and_segments :
  h = 11.2 ∧ m = 6.6 ∧ (c - m) = 8.4 :=
by {
  sorry
}

end triangle_height_and_segments

namespace boys_count_at_table

import Mathlib

-- Definitions from conditions
def children_count : ℕ := 13
def alternates (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- The problem to be proven in Lean:
theorem boys_count_at_table : ∃ b g : ℕ, b + g = children_count ∧ alternates b ∧ alternates g ∧ b = 7 :=
by
  sorry

end boys_count_at_table

namespace cube_volume

import Mathlib
import Mathlib.Data.Real.Basic -- Import necessary library for real number operations

-- Define the surface area constant
def surface_area : ℝ := 725.9999999999998

-- Define the formula for surface area of a cube and solve for volume given the conditions
theorem cube_volume (SA : ℝ) (h : SA = surface_area) : 11^3 = 1331 :=
by sorry

end cube_volume

namespace base8_operations

import Mathlib

def add_base8 (a b : ℕ) : ℕ :=
  let sum := (a + b) % 8
  sum

def subtract_base8 (a b : ℕ) : ℕ :=
  let diff := (a + 8 - b) % 8
  diff

def step1 := add_base8 672 156
def step2 := subtract_base8 step1 213

theorem base8_operations :
  step2 = 0645 :=
by
  sorry

end base8_operations

namespace anne_total_bottle_caps

import Mathlib

def initial_bottle_caps_anne : ℕ := 10
def found_bottle_caps_anne : ℕ := 5

theorem anne_total_bottle_caps : initial_bottle_caps_anne + found_bottle_caps_anne = 15 := 
by
  sorry

end anne_total_bottle_caps

namespace pow_two_grows_faster_than_square

import Mathlib

theorem pow_two_grows_faster_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := sorry

end pow_two_grows_faster_than_square

namespace average_brown_MnMs

import Mathlib

theorem average_brown_MnMs 
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 9)
  (h2 : a2 = 12)
  (h3 : a3 = 8)
  (h4 : a4 = 8)
  (h5 : a5 = 3) : 
  (a1 + a2 + a3 + a4 + a5) / 5 = 8 :=
by
  sorry

end average_brown_MnMs

namespace find_LCM_of_numbers

import Mathlib

def HCF (a b : ℕ) : ℕ := sorry  -- A placeholder definition for HCF
def LCM (a b : ℕ) : ℕ := sorry  -- A placeholder definition for LCM

theorem find_LCM_of_numbers (a b : ℕ) 
  (h1 : a + b = 55) 
  (h2 : HCF a b = 5) 
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b = 0.09166666666666666) : 
  LCM a b = 120 := 
by 
  sorry

end find_LCM_of_numbers

namespace find_whole_number_M

import Mathlib

-- Define the conditions
def condition (M : ℕ) : Prop :=
  21 < M ∧ M < 23

-- Define the main theorem to be proven
theorem find_whole_number_M (M : ℕ) (h : condition M) : M = 22 := by
  sorry

end find_whole_number_M

namespace digit_theta

import Mathlib

noncomputable def theta : ℕ := 7

theorem digit_theta (Θ : ℕ) (h1 : 378 / Θ = 40 + Θ + Θ) : Θ = theta :=
by {
  sorry
}

end digit_theta

namespace soccer_league_fraction_female_proof

import Mathlib

variable (m f : ℝ)

def soccer_league_fraction_female : Prop :=
  let males_last_year := m
  let females_last_year := f
  let males_this_year := 1.05 * m
  let females_this_year := 1.2 * f
  let total_this_year := 1.1 * (m + f)
  (1.05 * m + 1.2 * f = 1.1 * (m + f)) → ((0.6 * m) / (1.65 * m) = 4 / 11)

theorem soccer_league_fraction_female_proof (m f : ℝ) : soccer_league_fraction_female m f :=
by {
  sorry
}

end soccer_league_fraction_female_proof

namespace zero_in_interval

import Mathlib

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^(1/3)

theorem zero_in_interval : ∃ x ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ), f x = 0 :=
by
  -- The correct statement only
  sorry

end zero_in_interval

namespace k5_possibility

import Mathlib

noncomputable def possible_k5 : Prop :=
  ∃ (intersections : Fin 5 → Fin 5 × Fin 10), 
    ∀ i j : Fin 5, i ≠ j → intersections i ≠ intersections j

theorem k5_possibility : possible_k5 := 
by
  sorry

end k5_possibility

namespace intersection_complement

import Mathlib

def set_M : Set ℝ := {x : ℝ | x^2 - x = 0}

def set_N : Set ℝ := {x : ℝ | ∃ n : ℤ, x = 2 * n + 1}

theorem intersection_complement (h : UniversalSet = Set.univ) :
  set_M ∩ (UniversalSet \ set_N) = {0} := 
sorry

end intersection_complement

namespace num_paths_from_E_to_G_pass_through_F

import Mathlib

-- Definitions for the positions on the grid.
def E := (0, 4)
def G := (5, 0)
def F := (3, 3)

-- Function to calculate the number of combinations.
def binom (n k: ℕ) : ℕ := Nat.choose n k

-- The mathematical statement to be proven.
theorem num_paths_from_E_to_G_pass_through_F :
  (binom 4 1) * (binom 5 2) = 40 :=
by
  -- Placeholder for the proof.
  sorry

end num_paths_from_E_to_G_pass_through_F

namespace choosing_top_cases

import Mathlib

def original_tops : Nat := 2
def bought_tops : Nat := 4
def total_tops : Nat := original_tops + bought_tops

theorem choosing_top_cases : total_tops = 6 := by
  sorry

end choosing_top_cases

namespace compare_fractions

import Mathlib

theorem compare_fractions : (-2 / 7) > (-3 / 10) :=
sorry

end compare_fractions

namespace johns_uncommon_cards

import Mathlib

def packs_bought : ℕ := 10
def cards_per_pack : ℕ := 20
def uncommon_fraction : ℚ := 1 / 4

theorem johns_uncommon_cards : packs_bought * (cards_per_pack * uncommon_fraction) = (50 : ℚ) := 
by 
  sorry

end johns_uncommon_cards

namespace square_side_length_is_10

import Mathlib

-- Define the side lengths of the original squares
def side_length1 : ℝ := 8
def side_length2 : ℝ := 6

-- Define the areas of the original squares
def area1 : ℝ := side_length1^2
def area2 : ℝ := side_length2^2

-- Define the total area of the combined squares
def total_area : ℝ := area1 + area2

-- Define the side length of the new square
def side_length_new_square : ℝ := 10

-- Theorem statement to prove that the side length of the new square is 10 cm
theorem square_side_length_is_10 : side_length_new_square^2 = total_area := by
  sorry

end square_side_length_is_10

namespace triangle_exists_among_single_color_sticks

import Mathlib

theorem triangle_exists_among_single_color_sticks
  (red yellow green : ℕ)
  (k y g K Y G : ℕ)
  (hk : k + y > G)
  (hy : y + g > K)
  (hg : g + k > Y)
  (hred : red = 100)
  (hyellow : yellow = 100)
  (hgreen : green = 100) :
  ∃ color : string, ∀ a b c : ℕ, (a = k ∨ a = K) → (b = k ∨ b = K) → (c = k ∨ c = K) → a + b > c :=
sorry

end triangle_exists_among_single_color_sticks

namespace range_cos_2alpha_cos_2beta

import Mathlib

variable (α β : ℝ)
variable (h : Real.sin α + Real.cos β = 3 / 2)

theorem range_cos_2alpha_cos_2beta :
  -3/2 ≤ Real.cos (2 * α) + Real.cos (2 * β) ∧ Real.cos (2 * α) + Real.cos (2 * β) ≤ 3/2 :=
sorry

end range_cos_2alpha_cos_2beta

namespace sphere_surface_area_ratio

import Mathlib

axiom prism_has_circumscribed_sphere : Prop
axiom prism_has_inscribed_sphere : Prop

theorem sphere_surface_area_ratio 
  (h1 : prism_has_circumscribed_sphere)
  (h2 : prism_has_inscribed_sphere) : 
  ratio_surface_area_of_circumscribed_to_inscribed_sphere = 5 :=
sorry

end sphere_surface_area_ratio

namespace total_time_to_complete_work

import Mathlib

noncomputable def mahesh_work_rate (W : ℕ) := W / 40
noncomputable def mahesh_work_done_in_20_days (W : ℕ) := 20 * (mahesh_work_rate W)
noncomputable def remaining_work (W : ℕ) := W - (mahesh_work_done_in_20_days W)
noncomputable def rajesh_work_rate (W : ℕ) := (remaining_work W) / 30

theorem total_time_to_complete_work (W : ℕ) :
    (mahesh_work_rate W) + (rajesh_work_rate W) = W / 24 →
    (mahesh_work_done_in_20_days W) = W / 2 →
    (remaining_work W) = W / 2 →
    (rajesh_work_rate W) = W / 60 →
    20 + 30 = 50 :=
by 
  intros _ _ _ _
  sorry

end total_time_to_complete_work

namespace sum_digits_500

import Mathlib

noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_500 (k : ℕ) (h : k = 55) :
  sum_digits (63 * 10^k - 64) = 500 :=
by
  sorry

end sum_digits_500

namespace sum_of_squares

import Mathlib

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 :=
by
  sorry

end sum_of_squares

namespace fraction_of_people_under_21_correct

import Mathlib

variable (P : ℕ) (frac_over_65 : ℚ) (num_under_21 : ℕ) (frac_under_21 : ℚ)

def total_people_in_range (P : ℕ) : Prop := 50 < P ∧ P < 100

def fraction_of_people_over_65 (frac_over_65 : ℚ) : Prop := frac_over_65 = 5/12

def number_of_people_under_21 (num_under_21 : ℕ) : Prop := num_under_21 = 36

def fraction_of_people_under_21 (frac_under_21 : ℚ) : Prop := frac_under_21 = 3/7

theorem fraction_of_people_under_21_correct :
  ∀ (P : ℕ),
  total_people_in_range P →
  fraction_of_people_over_65 (5 / 12) →
  number_of_people_under_21 36 →
  P = 84 →
  fraction_of_people_under_21 (36 / P) :=
by
  intros P h_range h_over_65 h_under_21 h_P
  sorry

end fraction_of_people_under_21_correct

namespace math_problem

import Mathlib

theorem math_problem 
  (x y : ℝ) 
  (h1 : 4 * x + y = 12) 
  (h2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := 
by
  sorry

end math_problem

namespace minimum_value_of_a

import Mathlib

noncomputable def inequality_valid_for_all_x (a : ℝ) : Prop :=
  ∀ (x : ℝ), 1 < x → x + a * Real.log x - x^a + 1 / Real.exp x ≥ 0

theorem minimum_value_of_a : ∃ a, inequality_valid_for_all_x a ∧ a = -Real.exp 1 := sorry

end minimum_value_of_a

namespace mod_graph_sum

import Mathlib

theorem mod_graph_sum (x₀ y₀ : ℕ) (h₁ : 2 * x₀ ≡ 1 [MOD 11]) (h₂ : 3 * y₀ ≡ 10 [MOD 11]) : x₀ + y₀ = 13 :=
by
  sorry

end mod_graph_sum

namespace desiree_age

import Mathlib

variables (D C : ℕ)
axiom condition1 : D = 2 * C
axiom condition2 : D + 30 = (2 * (C + 30)) / 3 + 14

theorem desiree_age : D = 6 :=
by
  sorry

end desiree_age

namespace number_of_pears_in_fruit_gift_set

import Mathlib

theorem number_of_pears_in_fruit_gift_set 
  (F : ℕ) 
  (h1 : (2 / 9) * F = 10) 
  (h2 : 2 / 5 * F = 18) : 
  (2 / 5) * F = 18 :=
by 
  -- Sorry is used to skip the actual proof for now
  sorry

end number_of_pears_in_fruit_gift_set

namespace andrey_boris_denis_eat_candies

import Mathlib

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies

namespace log_inequality_solution

import Mathlib

noncomputable def log_a (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (log_a a (3 / 5) < 1) ↔ (a ∈ Set.Ioo 0 (3 / 5) ∪ Set.Ioi 1) := 
by
  sorry

end log_inequality_solution

namespace female_democrats

import Mathlib

theorem female_democrats (F M : ℕ) 
    (h₁ : F + M = 990)
    (h₂ : F / 2 + M / 4 = 330) : F / 2 = 275 := 
by sorry

end female_democrats

namespace decreasing_interval

import Mathlib

def f (a x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

theorem decreasing_interval (a : ℝ) : (∀ x y : ℝ, x ≤ y → y ≤ 4 → f a y ≤ f a x) ↔ a < -3 := 
by
  sorry

end decreasing_interval

namespace harkamal_total_amount

import Mathlib

-- Define the conditions as constants
def quantity_grapes : ℕ := 10
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost of grapes and mangoes based on the given conditions
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Define the total amount paid
def total_amount_paid : ℕ := cost_grapes + cost_mangoes

-- The theorem stating the problem and the solution
theorem harkamal_total_amount : total_amount_paid = 1195 := by
  -- Proof goes here (omitted)
  sorry

end harkamal_total_amount

namespace penguin_fish_consumption

import Mathlib

-- Definitions based on the conditions
def initial_penguins : ℕ := 158
def total_fish_per_day : ℕ := 237
def fish_per_penguin_per_day : ℚ := 1.5

-- Lean statement for the conditional problem
theorem penguin_fish_consumption
  (P : ℕ)
  (h_initial_penguins : P = initial_penguins)
  (h_total_fish_per_day : total_fish_per_day = 237)
  (h_current_penguins : P * 2 * 3 + 129 = 1077)
  : total_fish_per_day / P = fish_per_penguin_per_day := by
  sorry

end penguin_fish_consumption

namespace arrangement_plans_count

import Mathlib

noncomputable def number_of_arrangement_plans (num_teachers : ℕ) (num_students : ℕ) : ℕ :=
if num_teachers = 2 ∧ num_students = 4 then 12 else 0

theorem arrangement_plans_count :
  number_of_arrangement_plans 2 4 = 12 :=
by 
  sorry

end arrangement_plans_count

namespace find_remainder

import Mathlib

noncomputable def q (x : ℝ) : ℝ := (x^2010 + x^2009 + x^2008 + x + 1)
noncomputable def s (x : ℝ) := (q x) % (x^3 + 2*x^2 + 3*x + 1)

theorem find_remainder (x : ℝ) : (|s 2011| % 500) = 357 := by
    sorry

end find_remainder

namespace find_a4

import Mathlib

-- Define the sequence
noncomputable def a : ℕ → ℝ := sorry

-- Define the initial term a1 and common difference d
noncomputable def a1 : ℝ := sorry
noncomputable def d : ℝ := sorry

-- The conditions from the problem
def condition1 : Prop := a 2 + a 6 = 10 * Real.sqrt 3
def condition2 : Prop := a 3 + a 7 = 14 * Real.sqrt 3

-- Using the conditions to prove a4
theorem find_a4 (h1 : condition1) (h2 : condition2) : a 4 = 5 * Real.sqrt 3 :=
by
  sorry

end find_a4

namespace stream_speed

import Mathlib

variable (v : ℝ)

def effective_speed_downstream (v : ℝ) : ℝ := 7.5 + v
def effective_speed_upstream (v : ℝ) : ℝ := 7.5 - v 

theorem stream_speed : (7.5 - v) / (7.5 + v) = 1 / 2 → v = 2.5 :=
by
  intro h
  -- Proof will be resolved here
  sorry

end stream_speed

namespace obtuse_triangle_sum_range

import Mathlib

variable (a b c : ℝ)

theorem obtuse_triangle_sum_range (h1 : b^2 + c^2 - a^2 = b * c)
                                   (h2 : a = (Real.sqrt 3) / 2)
                                   (h3 : (b * c) * (Real.cos (Real.pi - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))) < 0) :
    (b + c) ∈ Set.Ioo ((Real.sqrt 3) / 2) (3 / 2) :=
sorry

end obtuse_triangle_sum_range

namespace probability_all_black_after_rotation

import Mathlib -- Import the math library

-- Define the conditions
def num_unit_squares : ℕ := 16
def num_colors : ℕ := 3
def prob_per_color : ℚ := 1 / 3

-- Define the type for probabilities
def prob_black_grid : ℚ := (1 / 81) * (11 / 27) ^ 12

-- The statement to be proven
theorem probability_all_black_after_rotation :
  (prob_black_grid =
    ((1 / 3) ^ 4) * ((11 / 27) ^ 12)) :=
sorry

end probability_all_black_after_rotation

namespace line_circle_intersection

import Mathlib

theorem line_circle_intersection (a : ℝ) : 
  (∀ x y : ℝ, (4 * x + 3 * y + a = 0) → ((x - 1)^2 + (y - 2)^2 = 9)) ∧
  (∃ A B : ℝ, dist A B = 4 * Real.sqrt 2) →
  (a = -5 ∨ a = -15) :=
by 
  sorry

end line_circle_intersection

namespace dogs_prevent_wolf_escape

import Mathlib

theorem dogs_prevent_wolf_escape
  (wolf_speed dog_speed : ℝ)
  (at_center: True)
  (dogs_at_vertices: True)
  (wolf_all_over_field: True)
  (dogs_on_perimeter: True)
  (wolf_handles_one_dog: ∀ (d : ℕ), d = 1 → True)
  (wolf_handles_two_dogs: ∀ (d : ℕ), d = 2 → False)
  (dog_faster_than_wolf: dog_speed = 1.5 * wolf_speed) : 
  ∀ (wolf_position : ℝ × ℝ) (boundary_position : ℝ × ℝ), 
  wolf_position != boundary_position → dog_speed > wolf_speed → 
  False := 
by sorry

end dogs_prevent_wolf_escape

namespace papers_left

import Mathlib

def total_papers_bought : ℕ := 20
def pictures_drawn_today : ℕ := 6
def pictures_drawn_yesterday_before_work : ℕ := 6
def pictures_drawn_yesterday_after_work : ℕ := 6

theorem papers_left :
  total_papers_bought - (pictures_drawn_today + pictures_drawn_yesterday_before_work + pictures_drawn_yesterday_after_work) = 2 := 
by 
  sorry

end papers_left

namespace max_area_ABC

import Mathlib

noncomputable def q (p : ℝ) : ℝ := p^2 - 7*p + 10

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem max_area_ABC : ∃ p : ℝ, 2 ≤ p ∧ p ≤ 5 ∧ 
  triangle_area (2, 0) (5, 4) (p, q p) = 0.536625 := sorry

end max_area_ABC

namespace max_min_x2_minus_xy_plus_y2

import Mathlib

theorem max_min_x2_minus_xy_plus_y2 (x y: ℝ) (h : |5 * x + y| + |5 * x - y| = 20) : 
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end max_min_x2_minus_xy_plus_y2

namespace work_completion_days

import Mathlib

theorem work_completion_days (A_days B_days : ℕ) (hA : A_days = 3) (hB : B_days = 6) : 
  (1 / ((1 / (A_days : ℚ)) + (1 / (B_days : ℚ)))) = 2 := 
by
  sorry

end work_completion_days

namespace prove_3a_3b_3c

import Mathlib

variable (a b c : ℝ)

def condition1 := b + c = 15 - 2 * a
def condition2 := a + c = -18 - 3 * b
def condition3 := a + b = 8 - 4 * c
def condition4 := a - b + c = 3

theorem prove_3a_3b_3c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) (h4 : condition4 a b c) :
  3 * a + 3 * b + 3 * c = 24 / 5 :=
sorry

end prove_3a_3b_3c

namespace number_of_friends

import Mathlib

-- Define the initial amount of money John had
def initial_money : ℝ := 20.10 

-- Define the amount spent on sweets
def sweets_cost : ℝ := 1.05 

-- Define the amount given to each friend
def money_per_friend : ℝ := 1.00 

-- Define the amount of money left after giving to friends
def final_money : ℝ := 17.05 

-- Define a theorem to find the number of friends John gave money to
theorem number_of_friends (init_money sweets_cost money_per_friend final_money : ℝ) : 
  (init_money - sweets_cost - final_money) / money_per_friend = 2 :=
by
  sorry

end number_of_friends

namespace jane_paints_correct_area

import Mathlib

def height_of_wall : ℕ := 10
def length_of_wall : ℕ := 15
def width_of_door : ℕ := 3
def height_of_door : ℕ := 5

def area_of_wall := height_of_wall * length_of_wall
def area_of_door := width_of_door * height_of_door
def area_to_be_painted := area_of_wall - area_of_door

theorem jane_paints_correct_area : area_to_be_painted = 135 := by
  sorry

end jane_paints_correct_area

namespace least_number_subtracted

import Mathlib

theorem least_number_subtracted (n k : ℕ) (h₁ : n = 123457) (h₂ : k = 79) : ∃ r, n % k = r ∧ r = 33 :=
by
  sorry

end least_number_subtracted

namespace a_n_formula_T_n_formula

import Mathlib

variable (a : Nat → Int) (b : Nat → Int)
variable (S : Nat → Int) (T : Nat → Int)
variable (d a_1 : Int)

-- Conditions:
axiom a_seq_arith : ∀ n, a (n + 1) = a n + d
axiom S_arith : ∀ n, S n = n * (a 1 + a n) / 2
axiom S_10 : S 10 = 110
axiom geo_seq : (a 2) ^ 2 = a 1 * a 4
axiom b_def : ∀ n, b n = 1 / ((a n - 1) * (a n + 1))

-- Goals: 
-- 1. Find the general formula for the terms of sequence {a_n}
theorem a_n_formula : ∀ n, a n = 2 * n := sorry

-- 2. Find the sum of the first n terms T_n of the sequence {b_n} given b_n
theorem T_n_formula : ∀ n, T n = 1 / 2 - 1 / (4 * n + 2) := sorry

end a_n_formula_T_n_formula

namespace proof_statement

import Mathlib

open Classical

variable (Person : Type) (Nationality : Type) (Occupation : Type)

variable (A B C D : Person)
variable (UnitedKingdom UnitedStates Germany France : Nationality)
variable (Doctor Teacher : Occupation)

variable (nationality : Person → Nationality)
variable (occupation : Person → Occupation)
variable (can_swim : Person → Prop)
variable (play_sports_together : Person → Person → Prop)

noncomputable def proof :=
  (nationality A = UnitedKingdom ∧ nationality D = Germany)

axiom condition1 : occupation A = Doctor ∧ ∃ x : Person, nationality x = UnitedStates ∧ occupation x = Doctor
axiom condition2 : occupation B = Teacher ∧ ∃ x : Person, nationality x = Germany ∧ occupation x = Teacher 
axiom condition3 : can_swim C ∧ ∀ x : Person, nationality x = Germany → ¬ can_swim x
axiom condition4 : ∃ x : Person, nationality x = France ∧ play_sports_together A x

theorem proof_statement : 
  (nationality A = UnitedKingdom ∧ nationality D = Germany) :=
by {
  sorry
}

end proof_statement

namespace souvenirs_total_cost

import Mathlib

theorem souvenirs_total_cost (T : ℝ) (H1 : 347 = T + 146) : T + 347 = 548 :=
by
  -- To ensure the validity of the Lean statement but without the proof.
  sorry

end souvenirs_total_cost

namespace sum_of_fractions

import Mathlib

theorem sum_of_fractions : (1 / 3 : ℚ) + (2 / 7) = 13 / 21 :=
by
  sorry

end sum_of_fractions

namespace female_members_count

import Mathlib

theorem female_members_count (M F : ℕ) (h1 : F = 2 * M) (h2 : F + M = 18) : F = 12 :=
by
  -- the proof will go here
  sorry

end female_members_count

namespace probability_of_MATHEMATICS_letter

import Mathlib

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_MATHEMATICS_letter :
  let total_letters := 26
  let unique_letters_count := unique_letters_in_mathematics.card
  (unique_letters_count / total_letters : ℝ) = 8 / 26 := by
  sorry

end probability_of_MATHEMATICS_letter

namespace Caden_total_money

import Mathlib

theorem Caden_total_money (p n d q : ℕ) (hp : p = 120)
    (hn : p = 3 * n) 
    (hd : n = 5 * d)
    (hq : q = 2 * d) :
    (p * 1 / 100 + n * 5 / 100 + d * 10 / 100 + q * 25 / 100) = 8 := 
by
  sorry

end Caden_total_money

namespace vasya_has_more_fanta

import Mathlib

-- Definitions based on the conditions:
def initial_fanta_vasya (a : ℝ) : ℝ := a
def initial_fanta_petya (a : ℝ) : ℝ := 1.1 * a
def remaining_fanta_vasya (a : ℝ) : ℝ := a * 0.98
def remaining_fanta_petya (a : ℝ) : ℝ := 1.1 * a * 0.89

-- The theorem to prove Vasya has more Fanta left than Petya.
theorem vasya_has_more_fanta (a : ℝ) (h : 0 < a) : remaining_fanta_vasya a > remaining_fanta_petya a := by
  sorry

end vasya_has_more_fanta

namespace find_tangent_line

import Mathlib

def is_perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

def is_tangent_to_circle (a b c : ℝ) : Prop :=
  let d := abs c / (Real.sqrt (a^2 + b^2))
  d = 1

def in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem find_tangent_line :
  ∀ (k b : ℝ),
    is_perpendicular k 1 →
    is_tangent_to_circle 1 1 b →
    ∃ (x y : ℝ), in_first_quadrant x y ∧ x + y - b = 0 →
    b = Real.sqrt 2 := sorry

end find_tangent_line

namespace swimming_time

import Mathlib

theorem swimming_time (c t : ℝ) 
  (h1 : 10.5 + c ≠ 0)
  (h2 : 10.5 - c ≠ 0)
  (h3 : t = 45 / (10.5 + c))
  (h4 : t = 18 / (10.5 - c)) :
  t = 3 := 
by
  sorry

end swimming_time

namespace Liu_Wei_parts_per_day

import Mathlib

theorem Liu_Wei_parts_per_day :
  ∀ (total_parts days_needed parts_per_day_worked initial_days days_remaining : ℕ), 
  total_parts = 190 →
  parts_per_day_worked = 15 →
  initial_days = 2 →
  days_needed = 10 →
  days_remaining = days_needed - initial_days →
  (total_parts - (initial_days * parts_per_day_worked)) / days_remaining = 20 :=
by
  intros total_parts days_needed parts_per_day_worked initial_days days_remaining h1 h2 h3 h4 h5
  sorry

end Liu_Wei_parts_per_day

namespace part1_part2_part3

import Mathlib

-- Definitions based on conditions
def fractional_eq (x a : ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Part (1): Proof statement for a == -1 if x == 5 is a root
theorem part1 (x : ℝ) (a : ℝ) (h : x = 5) (heq : fractional_eq x a) : a = -1 :=
sorry

-- Part (2): Proof statement for a == 2 if the equation has a double root
theorem part2 (a : ℝ) (h_double_root : ∀ x, fractional_eq x a → x = 0 ∨ x = 2) : a = 2 :=
sorry

-- Part (3): Proof statement for a == -3 or == 2 if the equation has no solution
theorem part3 (a : ℝ) (h_no_solution : ¬∃ x, fractional_eq x a) : a = -3 ∨ a = 2 :=
sorry

end part1_part2_part3

namespace factor_theorem_solution

import Mathlib

theorem factor_theorem_solution (t : ℝ) :
  (∃ p q : ℝ, 10 * p * q = 10 * t * t + 21 * t - 10 ∧ (x - q) = (x - t)) →
  t = 2 / 5 ∨ t = -5 / 2 := by
  sorry

end factor_theorem_solution

namespace problem_statement

import Mathlib

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end problem_statement

namespace minimum_shirts_for_saving_money

import Mathlib

-- Define the costs for Acme and Gamma
def acme_cost (x : ℕ) : ℕ := 60 + 10 * x
def gamma_cost (x : ℕ) : ℕ := 15 * x

-- Prove that the minimum number of shirts x for which a customer saves money by using Acme is 13
theorem minimum_shirts_for_saving_money : ∃ (x : ℕ), 60 + 10 * x < 15 * x ∧ x = 13 := by
  sorry

end minimum_shirts_for_saving_money

namespace final_inventory_is_correct

import Mathlib

def initial_inventory : ℕ := 4500
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_remaining_week : ℕ := 50
def supplier_delivery : ℕ := 650

def bottles_sold_first_two_days : ℕ := bottles_sold_monday + bottles_sold_tuesday
def days_remaining : ℕ := 5
def bottles_sold_remaining_week : ℕ := days_remaining * bottles_sold_per_day_remaining_week
def total_bottles_sold_week : ℕ := bottles_sold_first_two_days + bottles_sold_remaining_week
def remaining_inventory : ℕ := initial_inventory - total_bottles_sold_week
def final_inventory : ℕ := remaining_inventory + supplier_delivery

theorem final_inventory_is_correct :
  final_inventory = 1555 :=
by
  sorry

end final_inventory_is_correct

namespace focus_of_hyperbola

import Mathlib

-- Define the given hyperbola equation and its conversion to standard form
def hyperbola_eq (x y : ℝ) : Prop := -2 * (x - 2)^2 + 3 * (y + 3)^2 - 28 = 0

-- Define the standard form equation of the hyperbola
def standard_form (x y : ℝ) : Prop :=
  ((y + 3)^2 / (28 / 3)) - ((x - 2)^2 / 14) = 1

-- Define the coordinates of one of the foci of the hyperbola
def focus (x y : ℝ) : Prop :=
  x = 2 ∧ y = -3 + Real.sqrt (70 / 3)

-- The theorem statement proving the given coordinates is a focus of the hyperbola
theorem focus_of_hyperbola :
  ∃ x y, hyperbola_eq x y ∧ standard_form x y → focus x y :=
by
  existsi 2, (-3 + Real.sqrt (70 / 3))
  sorry -- Proof is required to substantiate it, placeholder here.

end focus_of_hyperbola

namespace convex_polygon_sides

import Mathlib

theorem convex_polygon_sides (n : ℕ) (h : ∀ angle, angle = 45 → angle * n = 360) : n = 8 :=
  sorry

end convex_polygon_sides

namespace joe_total_paint_used

import Mathlib

-- Define the initial amount of paint Joe buys.
def initial_paint : ℕ := 360

-- Define the fraction of paint used during the first week.
def first_week_fraction := 1 / 4

-- Define the fraction of remaining paint used during the second week.
def second_week_fraction := 1 / 2

-- Define the total paint used by Joe in the first week.
def paint_used_first_week := first_week_fraction * initial_paint

-- Define the remaining paint after the first week.
def remaining_paint_after_first_week := initial_paint - paint_used_first_week

-- Define the total paint used by Joe in the second week.
def paint_used_second_week := second_week_fraction * remaining_paint_after_first_week

-- Define the total paint used by Joe.
def total_paint_used := paint_used_first_week + paint_used_second_week

-- The theorem to be proven: the total amount of paint Joe has used is 225 gallons.
theorem joe_total_paint_used : total_paint_used = 225 := by
  sorry

end joe_total_paint_used

namespace raised_bed_section_area

import Mathlib

theorem raised_bed_section_area :
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  area_of_raised_beds = 8800 :=
by 
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  show area_of_raised_beds = 8800
  sorry

end raised_bed_section_area

namespace raised_bed_area_correct

import Mathlib

def garden_length : ℝ := 220
def garden_width : ℝ := 120
def garden_area : ℝ := garden_length * garden_width
def tilled_land_area : ℝ := garden_area / 2
def remaining_area : ℝ := garden_area - tilled_land_area
def trellis_area : ℝ := remaining_area / 3
def raised_bed_area : ℝ := remaining_area - trellis_area

theorem raised_bed_area_correct : raised_bed_area = 8800 := by
  sorry

end raised_bed_area_correct

namespace crayons_lost_or_given_away

import Mathlib

theorem crayons_lost_or_given_away (given_away lost : ℕ) (H_given_away : given_away = 213) (H_lost : lost = 16) :
  given_away + lost = 229 :=
by
  sorry

end crayons_lost_or_given_away

namespace bagel_pieces_after_10_cuts

import Mathlib

def bagel_pieces_after_cuts (initial_pieces : ℕ) (cuts : ℕ) : ℕ :=
  initial_pieces + cuts

theorem bagel_pieces_after_10_cuts : bagel_pieces_after_cuts 1 10 = 11 := by
  sorry

end bagel_pieces_after_10_cuts

namespace no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10

import Mathlib

theorem no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10 :
  ¬ ∃ x : ℝ, x^4 + (x + 1)^4 + (x + 2)^4 = (x + 3)^4 + 10 :=
by {
  sorry
}

end no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10

namespace determine_marriages

import Mathlib

-- Definitions of the items each person bought
variable (a_items b_items c_items : ℕ) -- Number of items bought by wives a, b, and c
variable (A_items B_items C_items : ℕ) -- Number of items bought by husbands A, B, and C

-- Conditions
variable (spend_eq_square_a : a_items * a_items = a_spend) -- Spending equals square of items
variable (spend_eq_square_b : b_items * b_items = b_spend)
variable (spend_eq_square_c : c_items * c_items = c_spend)
variable (spend_eq_square_A : A_items * A_items = A_spend)
variable (spend_eq_square_B : B_items * B_items = B_spend)
variable (spend_eq_square_C : C_items * C_items = C_spend)

variable (A_spend_eq : A_spend = a_spend + 48) -- Husbands spent 48 yuan more than wives
variable (B_spend_eq : B_spend = b_spend + 48)
variable (C_spend_eq : C_spend = c_spend + 48)

variable (A_bought_9_more : A_items = b_items + 9) -- A bought 9 more items than b
variable (B_bought_7_more : B_items = a_items + 7) -- B bought 7 more items than a

-- Theorem statement
theorem determine_marriages (hA : A_items ≥ b_items + 9) (hB : B_items ≥ a_items + 7) :
  (A_spend = A_items * A_items) ∧ (B_spend = B_items * B_items) ∧ (C_spend = C_items * C_items) ∧
  (a_spend = a_items * a_items) ∧ (b_spend = b_items * b_items) ∧ (c_spend = c_items * c_items) →
  (A_spend = a_spend + 48) ∧ (B_spend = b_spend + 48) ∧ (C_spend = c_spend + 48) →
  (A_items = b_items + 9) ∧ (B_items = a_items + 7) →
  (A_items = 13 ∧ c_items = 11) ∧ (B_items = 8 ∧ b_items = 4) ∧ (C_items = 7 ∧ a_items = 1) :=
by
  sorry

end determine_marriages

namespace smallest_odd_digit_number_gt_1000_mult_5

import Mathlib

def is_odd_digit (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def valid_number (n : ℕ) : Prop :=
  n > 1000 ∧ (∃ d1 d2 d3 d4, n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
  is_odd_digit d1 ∧ is_odd_digit d2 ∧ is_odd_digit d3 ∧ is_odd_digit d4 ∧ 
  d4 = 5)

theorem smallest_odd_digit_number_gt_1000_mult_5 : ∃ n : ℕ, valid_number n ∧ 
  ∀ m : ℕ, valid_number m → m ≥ n := 
by
  use 1115
  simp [valid_number, is_odd_digit]
  sorry

end smallest_odd_digit_number_gt_1000_mult_5

namespace sum_of_x_values

import Mathlib

theorem sum_of_x_values (y x : ℝ) (h1 : y = 6) (h2 : x^2 + y^2 = 144) : x + (-x) = 0 :=
by
  sorry

end sum_of_x_values

namespace find_rth_term

import Mathlib

def S (n : ℕ) : ℕ := 2 * n + 3 * (n^3)

def a (r : ℕ) : ℕ := S r - S (r - 1)

theorem find_rth_term (r : ℕ) : a r = 9 * r^2 - 9 * r + 5 := by
  sorry

end find_rth_term

namespace problem

import Mathlib

variables {b1 b2 b3 a1 a2 : ℤ}

-- Condition: five numbers -9, b1, b2, b3, -1 form a geometric sequence.
def is_geometric_seq (b1 b2 b3 : ℤ) : Prop :=
b1^2 = -9 * b2 ∧ b2^2 = b1 * b3 ∧ b1 * b3 = 9

-- Condition: four numbers -9, a1, a2, -3 form an arithmetic sequence.
def is_arithmetic_seq (a1 a2 : ℤ) : Prop :=
2 * a1 = -9 + a2 ∧ 2 * a2 = a1 - 3

-- Proof problem: prove that b2(a2 - a1) = -6
theorem problem (h_geom : is_geometric_seq b1 b2 b3) (h_arith : is_arithmetic_seq a1 a2) : 
  b2 * (a2 - a1) = -6 :=
by sorry

end problem

namespace probability_adjacent_vertices_decagon

import Mathlib

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon

namespace length_of_second_platform

import Mathlib

/-- 
Let L be the length of the second platform.
A train crosses a platform of 100 m in 15 sec.
The same train crosses another platform in 20 sec.
The length of the train is 350 m.
Prove that the length of the second platform is 250 meters.
-/
theorem length_of_second_platform (L : ℕ) (train_length : ℕ) (platform1_length : ℕ) (time1 : ℕ) (time2 : ℕ):
  train_length = 350 → platform1_length = 100 → time1 = 15 → time2 = 20 → L = 250 :=
by
  sorry

end length_of_second_platform

namespace divisibility_equivalence_distinct_positive

import Mathlib

variable (a b c : ℕ)

theorem divisibility_equivalence_distinct_positive (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c) ∣ (a^3 * b + b^3 * c + c^3 * a)) ↔ ((a + b + c) ∣ (a * b^3 + b * c^3 + c * a^3)) :=
by sorry

end divisibility_equivalence_distinct_positive

namespace most_convincing_method_for_relationship

import Mathlib

-- Definitions from conditions
def car_owners : ℕ := 300
def car_owners_opposed_policy : ℕ := 116
def non_car_owners : ℕ := 200
def non_car_owners_opposed_policy : ℕ := 121

-- The theorem statement
theorem most_convincing_method_for_relationship : 
  (owning_a_car_related_to_opposing_policy : Bool) :=
by
  -- Proof of the statement
  sorry

end most_convincing_method_for_relationship

namespace negative_column_exists

import Mathlib

theorem negative_column_exists
  (table : Fin 1999 → Fin 2001 → ℤ)
  (H : ∀ i : Fin 1999, (∏ j : Fin 2001, table i j) < 0) :
  ∃ j : Fin 2001, (∏ i : Fin 1999, table i j) < 0 :=
sorry

end negative_column_exists

namespace recommendation_plans_count

import Mathlib

def num_male : ℕ := 3
def num_female : ℕ := 2
def num_recommendations : ℕ := 5

def num_spots_russian : ℕ := 2
def num_spots_japanese : ℕ := 2
def num_spots_spanish : ℕ := 1

def condition_russian (males : ℕ) : Prop := males > 0
def condition_japanese (males : ℕ) : Prop := males > 0

theorem recommendation_plans_count : 
  (∃ (males_r : ℕ) (males_j : ℕ), condition_russian males_r ∧ condition_japanese males_j ∧ 
  num_male - males_r - males_j >= 0 ∧ males_r + males_j ≤ num_male ∧ 
  num_female + (num_male - males_r - males_j) >= num_recommendations - (num_spots_russian + num_spots_japanese + num_spots_spanish)) →
  (∃ (x : ℕ), x = 24) := by
  sorry

end recommendation_plans_count

namespace day_of_20th_is_Thursday

import Mathlib

noncomputable def day_of_week (d : ℕ) : String :=
  match d % 7 with
  | 0 => "Saturday"
  | 1 => "Sunday"
  | 2 => "Monday"
  | 3 => "Tuesday"
  | 4 => "Wednesday"
  | 5 => "Thursday"
  | 6 => "Friday"
  | _ => "Unknown"

theorem day_of_20th_is_Thursday (s1 s2 s3: ℕ) (h1: 2 ≤ s1) (h2: s1 ≤ 30) (h3: s2 = s1 + 14) (h4: s3 = s2 + 14) (h5: s3 ≤ 30) (h6: day_of_week s1 = "Sunday") : 
  day_of_week 20 = "Thursday" :=
by
  sorry

end day_of_20th_is_Thursday

namespace sqrt_of_4_eq_2

import Mathlib

theorem sqrt_of_4_eq_2 : Real.sqrt 4 = 2 := by
  sorry

end sqrt_of_4_eq_2

namespace probability_mass_range

import Mathlib

/-- Let ξ be a random variable representing the mass of a badminton product. 
    Suppose P(ξ < 4.8) = 0.3 and P(ξ ≥ 4.85) = 0.32. 
    We want to prove that the probability that the mass is in the range [4.8, 4.85) is 0.38. -/
theorem probability_mass_range (P : ℝ → ℝ) (h1 : P (4.8) = 0.3) (h2 : P (4.85) = 0.32) :
  P (4.8) - P (4.85) = 0.38 :=
by 
  sorry

end probability_mass_range

namespace triangle_is_isosceles_right

import Mathlib

theorem triangle_is_isosceles_right (a b S : ℝ) (h : S = (1/4) * (a^2 + b^2)) :
  ∃ C : ℝ, C = 90 ∧ a = b :=
by
  sorry

end triangle_is_isosceles_right

namespace hypotenuse_length

import Mathlib

theorem hypotenuse_length (x y : ℝ) 
  (h1 : (1/3) * Real.pi * y^2 * x = 1080 * Real.pi) 
  (h2 : (1/3) * Real.pi * x^2 * y = 2430 * Real.pi) : 
  Real.sqrt (x^2 + y^2) = 6 * Real.sqrt 13 := 
  sorry

end hypotenuse_length

namespace train_crossing_time

import Mathlib

namespace TrainCrossingProblem

def length_of_train : ℕ := 250
def length_of_bridge : ℕ := 300
def speed_of_train_kmph : ℕ := 36
def speed_of_train_mps : ℕ := 10 -- conversion from 36 kmph to m/s
def total_distance : ℕ := length_of_train + length_of_bridge -- 250 + 300
def expected_time : ℕ := 55

theorem train_crossing_time : 
  (total_distance / speed_of_train_mps) = expected_time :=
by
  sorry
end TrainCrossingProblem

end train_crossing_time

namespace algebraic_sum_parity

import Mathlib

theorem algebraic_sum_parity :
  ∀ (f : Fin 2006 → ℤ),
    (∀ i, f i = i ∨ f i = -i) →
    (∑ i, f i) % 2 = 1 := by
  sorry

end algebraic_sum_parity

namespace find_x

import Mathlib

variable (x : ℕ)  -- we'll use natural numbers to avoid negative values

-- initial number of children
def initial_children : ℕ := 21

-- number of children who got off
def got_off : ℕ := 10

-- total children after some got on
def total_children : ℕ := 16

-- statement to prove x is the number of children who got on the bus
theorem find_x : initial_children - got_off + x = total_children → x = 5 :=
by
  sorry

end find_x

namespace servant_leaving_months

import Mathlib

-- The given conditions
def total_salary_year : ℕ := 90 + 110
def monthly_salary (months: ℕ) : ℕ := (months * total_salary_year) / 12
def total_received : ℕ := 40 + 110

-- The theorem to prove
theorem servant_leaving_months (months : ℕ) (h : monthly_salary months = total_received) : months = 9 :=
by {
    sorry
}

end servant_leaving_months

namespace blocks_left

import Mathlib

theorem blocks_left (initial_blocks used_blocks : ℕ) (h_initial : initial_blocks = 59) (h_used : used_blocks = 36) : initial_blocks - used_blocks = 23 :=
by
  -- proof here
  sorry

end blocks_left

namespace prove_range_of_a

import Mathlib

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + Real.log (abs (a + 2))

def is_increasing (f : ℝ → ℝ) (interval : Set ℝ) :=
 ∀ ⦃x y⦄, x ∈ interval → y ∈ interval → x ≤ y → f x ≤ f y

def g (x a : ℝ) := (a + 1) * x
def is_decreasing (g : ℝ → ℝ) :=
 ∀ ⦃x y⦄, x ≤ y → g y ≤ g x

def proposition_p (a : ℝ) : Prop :=
  is_increasing (f a) (Set.Ici ((a + 1)^2))

def proposition_q (a : ℝ) : Prop :=
  is_decreasing (g a)

theorem prove_range_of_a (a : ℝ) (h : ¬ (proposition_p a ↔ proposition_q a)) :
  a > -3 / 2 :=
sorry

end prove_range_of_a

namespace angle_sum_x_y

import Mathlib

def angle_A := 36
def angle_B := 80
def angle_C := 24

def target_sum : ℕ := 140

theorem angle_sum_x_y (angle_A angle_B angle_C : ℕ) (x y : ℕ) : 
  angle_A = 36 → angle_B = 80 → angle_C = 24 → x + y = 140 := by 
  intros _ _ _
  sorry

end angle_sum_x_y

namespace weight_of_new_person

import Mathlib

/-- The average weight of 10 persons increases by 7.2 kg when a new person
replaces one who weighs 65 kg. Prove that the weight of the new person is 137 kg. -/
theorem weight_of_new_person (W_new : ℝ) (W_old : ℝ) (n : ℝ) (increase : ℝ) 
  (h1 : W_old = 65) (h2 : n = 10) (h3 : increase = 7.2) 
  (h4 : W_new = W_old + n * increase) : W_new = 137 := 
by
  -- proof to be done later
  sorry

end weight_of_new_person

namespace blocks_combination_count

import Mathlib

-- Definition statements reflecting all conditions in the problem
def select_4_blocks_combinations : ℕ :=
  let choose (n k : ℕ) := Nat.choose n k
  let factorial (n : ℕ) := Nat.factorial n
  choose 6 4 * choose 6 4 * factorial 4

-- Theorem stating the result we want to prove
theorem blocks_combination_count : select_4_blocks_combinations = 5400 :=
by
  -- We will provide the proof steps here
  sorry

end blocks_combination_count

namespace a1_lt_a3_iff_an_lt_an1

import Mathlib

-- Define arithmetic sequence and required properties
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℝ)

-- Define the necessary and sufficient condition theorem
theorem a1_lt_a3_iff_an_lt_an1 (h_arith : is_arithmetic_sequence a) :
  (a 1 < a 3) ↔ (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end a1_lt_a3_iff_an_lt_an1

namespace infinitely_many_c_exist

import Mathlib

theorem infinitely_many_c_exist :
  ∃ c: ℕ, ∃ x y z: ℕ, (x^2 - c) * (y^2 - c) = z^2 - c ∧ (x^2 + c) * (y^2 - c) = z^2 - c :=
by
  sorry

end infinitely_many_c_exist

namespace perimeter_of_similar_triangle

import Mathlib

def is_isosceles (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

theorem perimeter_of_similar_triangle (a b c : ℕ) (d e f : ℕ) 
  (h1 : is_isosceles a b c) (h2 : min a b = 15) (h3 : min (min a b) c = 15)
  (h4 : d = 75) (h5 : (d / 15) = e / b) (h6 : f = e) :
  d + e + f = 375 :=
by
  sorry

end perimeter_of_similar_triangle

namespace polynomial_solution

import Mathlib

theorem polynomial_solution (P : ℝ → ℝ) (hP : ∀ x : ℝ, (x + 1) * P (x - 1) + (x - 1) * P (x + 1) = 2 * x * P x) :
  ∃ (a d : ℝ), ∀ x : ℝ, P x = a * x^3 - a * x + d := 
sorry

end polynomial_solution

namespace cost_to_fill_pool

import Mathlib

-- Definitions based on the conditions
def cubic_foot_to_liters: ℕ := 25
def pool_depth: ℕ := 10
def pool_width: ℕ := 6
def pool_length: ℕ := 20
def cost_per_liter: ℕ := 3

-- Statement to be proved
theorem cost_to_fill_pool : 
  (pool_depth * pool_width * pool_length * cubic_foot_to_liters * cost_per_liter) = 90000 := 
by 
  sorry

end cost_to_fill_pool

namespace johns_commute_distance

import Mathlib

theorem johns_commute_distance
  (y : ℝ)  -- distance in miles
  (h1 : 200 * (y / 200) = y)  -- John usually takes 200 minutes, so usual speed is y/200 miles per minute
  (h2 : 320 = (y / (2 * (y / 200))) + (y / (2 * ((y / 200) - 15/60)))) -- Total journey time on the foggy day
  : y = 92 :=
sorry

end johns_commute_distance

namespace find_a2

import Mathlib

variable (S a : ℕ → ℕ)

-- Define the condition S_n = 2a_n - 2 for all n
axiom sum_first_n_terms (n : ℕ) : S n = 2 * a n - 2

-- Define the specific lemma for n = 1 to find a_1
axiom a1 : a 1 = 2

-- State the proof problem for a_2
theorem find_a2 : a 2 = 4 := 
by 
  sorry

end find_a2

namespace Annabelle_saved_12_dollars

import Mathlib

def weekly_allowance : ℕ := 30
def spent_on_junk_food : ℕ := weekly_allowance / 3
def spent_on_sweets : ℕ := 8
def total_spent : ℕ := spent_on_junk_food + spent_on_sweets
def saved_amount : ℕ := weekly_allowance - total_spent

theorem Annabelle_saved_12_dollars : saved_amount = 12 := by
  -- proof goes here
  sorry

end Annabelle_saved_12_dollars

namespace fraction_simplest_sum

import Mathlib

theorem fraction_simplest_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (3975 : ℚ) / 10000 = (a : ℚ) / b) 
  (simp : ∀ (c : ℕ), c ∣ a ∧ c ∣ b → c = 1) : a + b = 559 :=
sorry

end fraction_simplest_sum

namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites

import Mathlib

-- Definition: A number is composite if it has more than two distinct positive divisors
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (d : ℕ), d > 1 ∧ d < n ∧ n % d = 0

-- There exists a sequence of nine consecutive composite numbers within the first 500
theorem exists_nine_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 8 < 500 ∧
                           (∀ i, i < 9 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

-- There exists a sequence of eleven consecutive composite numbers within the first 500
theorem exists_eleven_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 10 < 500 ∧
                           (∀ i, i < 11 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites

namespace count_multiples_5_or_7_but_not_35

import Mathlib

def count_multiples (n d : ℕ) : ℕ :=
  n / d

def inclusion_exclusion (a b c : ℕ) : ℕ :=
  a + b - c

theorem count_multiples_5_or_7_but_not_35 : 
  count_multiples 3000 5 + count_multiples 3000 7 - count_multiples 3000 35 = 943 :=
by
  sorry

end count_multiples_5_or_7_but_not_35

namespace certain_number_is_50

import Mathlib

theorem certain_number_is_50 (x : ℝ) (h : 4 = 0.08 * x) : x = 50 :=
by {
    sorry
}

end certain_number_is_50

namespace interval_of_increase_find_side_c

import Mathlib

noncomputable def f (x : ℝ) : ℝ := 
  √3 * Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2

def interval_increasing (k : ℤ) : Set ℝ := 
  { x | k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 }

theorem interval_of_increase (k : ℤ) : 
  ∀ x : ℝ, (interval_increasing k) x ↔ f x > f (x - Real.pi / 3) ∧ f x < f (x + Real.pi / 6) :=
sorry

theorem find_side_c (A : ℝ) (a b : ℝ) (hA : f A = 1 / 2) (ha : a = √17) (hb : b = 4) : 
  ∃ c : ℝ, c = 2 + √5 :=
sorry

end interval_of_increase_find_side_c

namespace find_a

import Mathlib

-- Problem 1
theorem find_a (f : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x * (a + Real.log x))
  (hmin : ∃ x, f x = -Real.exp (-2) ∧ ∀ y, f y ≥ f x) : a = 1 := 
sorry

-- Problem 2
theorem find_k_max {k : ℤ} : 
  (∀ x > 1, k < (x * (1 + Real.log x)) / (x - 1)) → k ≤ 3 :=
sorry

end find_a

namespace percentage_of_males

import Mathlib

theorem percentage_of_males (P : ℝ) (total_employees : ℝ) (below_50_male_count : ℝ) :
  total_employees = 2800 →
  0.70 * (P / 100 * total_employees) = below_50_male_count →
  below_50_male_count = 490 →
  P = 25 :=
by
  intros h_total h_eq h_below_50
  sorry

end percentage_of_males

namespace right_triangle_has_one_right_angle

import Mathlib

def is_right_angle (θ : ℝ) : Prop := θ = 90

def sum_of_triangle_angles (α β γ : ℝ) : Prop := α + β + γ = 180

def right_triangle (α β γ : ℝ) : Prop := is_right_angle α ∨ is_right_angle β ∨ is_right_angle γ

theorem right_triangle_has_one_right_angle (α β γ : ℝ) :
  right_triangle α β γ → sum_of_triangle_angles α β γ →
  (is_right_angle α ∧ ¬is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ ¬is_right_angle β ∧ is_right_angle γ) :=
by
  sorry

end right_triangle_has_one_right_angle

namespace exists_disk_of_radius_one_containing_1009_points

import Mathlib

theorem exists_disk_of_radius_one_containing_1009_points
  (points : Fin 2017 → ℝ × ℝ)
  (h : ∀ (a b c : Fin 2017), (dist (points a) (points b) < 1) ∨ (dist (points b) (points c) < 1) ∨ (dist (points c) (points a) < 1)) :
  ∃ (center : ℝ × ℝ), ∃ (sub_points : Finset (Fin 2017)), sub_points.card ≥ 1009 ∧ ∀ p ∈ sub_points, dist (center) (points p) ≤ 1 :=
sorry

end exists_disk_of_radius_one_containing_1009_points

namespace karlson_wins_with_optimal_play

import Mathlib

def game_win_optimal_play: Prop :=
  ∀ (total_moves: ℕ), 
  (total_moves % 2 = 1) 

theorem karlson_wins_with_optimal_play: game_win_optimal_play :=
by sorry

end karlson_wins_with_optimal_play

namespace 

import Mathlib

theorem sequence_inequality {a : ℕ → ℝ} (h₁ : a 1 > 1)
  (h₂ : ∀ n : ℕ, a (n+1) = (a n ^ 2 + 1) / (2 * a n)) :
  ∀ n : ℕ, (∑ i in Finset.range n, a (i + 1)) < n + 2 * (a 1 - 1) := by
  sorry

end 

namespace decreasing_function_iff_m_eq_2

import Mathlib

theorem decreasing_function_iff_m_eq_2 
    (m : ℝ) : 
    (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(-5*m - 3) < (m^2 - m - 1) * (x + 1)^(-5*m - 3)) ↔ m = 2 := 
sorry

end decreasing_function_iff_m_eq_2

namespace area_of_circle_segment

import Mathlib

-- Definitions for the conditions in the problem
def circle_eq (x y : ℝ) : Prop := x^2 - 10 * x + y^2 = 9
def line_eq (x y : ℝ) : Prop := y = x - 5

-- The area of the portion of the circle that lies above the x-axis and to the left of the line y = x - 5
theorem area_of_circle_segment :
  let area_of_circle := 34 * Real.pi
  let portion_fraction := 1 / 8
  portion_fraction * area_of_circle = 4.25 * Real.pi :=
by
  sorry

end area_of_circle_segment

namespace avg_height_eq_61

import Mathlib

-- Define the constants and conditions
def Brixton : ℕ := 64
def Zara : ℕ := 64
def Zora := Brixton - 8
def Itzayana := Zora + 4

-- Define the total height of the four people
def total_height := Brixton + Zara + Zora + Itzayana

-- Define the average height
def average_height := total_height / 4

-- Theorem stating that the average height is 61 inches
theorem avg_height_eq_61 : average_height = 61 := by
  sorry

end avg_height_eq_61

namespace fuse_length_must_be_80

import Mathlib

-- Define the basic conditions
def distanceToSafeArea : ℕ := 400
def personSpeed : ℕ := 5
def fuseBurnSpeed : ℕ := 1

-- Calculate the time required to reach the safe area
def timeToSafeArea (distance speed : ℕ) : ℕ := distance / speed

-- Calculate the minimum length of the fuse based on the time to reach the safe area
def minFuseLength (time burnSpeed : ℕ) : ℕ := time * burnSpeed

-- The main problem statement: The fuse must be at least 80 meters long.
theorem fuse_length_must_be_80:
  minFuseLength (timeToSafeArea distanceToSafeArea personSpeed) fuseBurnSpeed = 80 :=
by
  sorry

end fuse_length_must_be_80

namespace arithmetic_sequence_sum

import Mathlib

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S_9 : ℚ) 
  (h_arith : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a2_a8 : a 2 + a 8 = 4 / 3) :
  S_9 = 6 :=
by
  sorry

end arithmetic_sequence_sum

namespace min_ratio_ax

import Mathlib

theorem min_ratio_ax (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) 
: y^2 - 1 = a^2 * (x^2 - 1) → ∃ (k : ℕ), k = 2 ∧ (a = k * x) := 
sorry

end min_ratio_ax

namespace fill_time

import Mathlib

-- Definition of the conditions
def faster_pipe_rate (t : ℕ) := 1 / t
def slower_pipe_rate (t : ℕ) := 1 / (4 * t)
def combined_rate (t : ℕ) := faster_pipe_rate t + slower_pipe_rate t
def time_to_fill_tank (t : ℕ) := 1 / combined_rate t

-- Given t = 50, prove the combined fill time is 40 minutes which is equal to the target time to fill the tank.
theorem fill_time (t : ℕ) (h : 4 * t = 200) : t = 50 → time_to_fill_tank t = 40 :=
by
  intros ht
  rw [ht]
  sorry

end fill_time

namespace max_num_triangles_for_right_triangle

import Mathlib

-- Define a right triangle on graph paper
def right_triangle (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ n ∧ 0 ≤ b ∧ b ≤ n

-- Define maximum number of triangles that can be formed within the triangle
def max_triangles (n : ℕ) : ℕ :=
  if h : n = 7 then 28 else 0  -- Given n = 7, the max number is 28

-- Define the theorem to be proven
theorem max_num_triangles_for_right_triangle :
  right_triangle 7 → max_triangles 7 = 28 :=
by
  intro h
  -- Proof goes here
  sorry

end max_num_triangles_for_right_triangle

namespace common_divisor

import Mathlib

theorem common_divisor (d : ℕ) (h1 : 30 % d = 3) (h2 : 40 % d = 4) : d = 9 :=
by 
  sorry

end common_divisor

namespace three_point_one_two_six_as_fraction

import Mathlib

theorem three_point_one_two_six_as_fraction : (3126 / 1000 : ℚ) = 1563 / 500 := 
by 
  sorry

end three_point_one_two_six_as_fraction

namespace rectangle_midpoints_sum

import Mathlib

theorem rectangle_midpoints_sum (A B C D M N O P : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (4, 0))
  (hC : C = (4, 3))
  (hD : D = (0, 3))
  (hM : M = (2, 0))
  (hN : N = (4, 1.5))
  (hO : O = (2, 3))
  (hP : P = (0, 1.5)) :
  (Real.sqrt ((2 - 0) ^ 2 + (0 - 0) ^ 2) + 
  Real.sqrt ((4 - 0) ^ 2 + (1.5 - 0) ^ 2) + 
  Real.sqrt ((2 - 0) ^ 2 + (3 - 0) ^ 2) + 
  Real.sqrt ((0 - 0) ^ 2 + (1.5 - 0) ^ 2)) = 11.38 :=
by
  sorry

end rectangle_midpoints_sum

namespace hyperbolas_same_asymptotes

import Mathlib

theorem hyperbolas_same_asymptotes :
  (∀ x y, (x^2 / 4 - y^2 / 9 = 1) → (∃ k, y = k * x)) →
  (∀ x y, (y^2 / 18 - x^2 / N = 1) → (∃ k, y = k * x)) →
  N = 8 :=
by sorry

end hyperbolas_same_asymptotes

namespace negation_of_existential_statement

import Mathlib

theorem negation_of_existential_statement (x : ℚ) :
  ¬ (∃ x : ℚ, x^2 = 3) ↔ ∀ x : ℚ, x^2 ≠ 3 :=
by sorry

end negation_of_existential_statement

namespace hyperbola_asymptote_equation

import Mathlib

variable (a b : ℝ)
variable (x y : ℝ)

def arithmetic_mean := (a + b) / 2 = 5
def geometric_mean := (a * b) ^ (1 / 2) = 4
def a_greater_b := a > b
def hyperbola_asymptote := (y = (1 / 2) * x) ∨ (y = -(1 / 2) * x)

theorem hyperbola_asymptote_equation :
  arithmetic_mean a b ∧ geometric_mean a b ∧ a_greater_b a b → hyperbola_asymptote x y :=
by
  sorry

end hyperbola_asymptote_equation

namespace car_speed

import Mathlib

-- Definitions from conditions
def distance : ℝ := 360
def time : ℝ := 4.5

-- Statement to prove
theorem car_speed : (distance / time) = 80 := by
  sorry

end car_speed

namespace jack_keeps_deers_weight_is_correct

import Mathlib

-- Define conditions
def monthly_hunt_count : Float := 7.5
def fraction_of_year_hunting_season : Float := 1 / 3
def deers_per_hunt : Float := 2.5
def weight_per_deer : Float := 600
def weight_kept_per_deer : Float := 0.65

-- Prove the total weight of the deer Jack keeps
theorem jack_keeps_deers_weight_is_correct :
  (12 * fraction_of_year_hunting_season) * monthly_hunt_count * deers_per_hunt * weight_per_deer * weight_kept_per_deer = 29250 :=
by
  sorry

end jack_keeps_deers_weight_is_correct

namespace shaded_region_area

import Mathlib

open Real

noncomputable def area_of_shaded_region (r : ℝ) (s : ℝ) (d : ℝ) : ℝ := 
  (1/4) * π * r^2 + (1/2) * (d - s)^2

theorem shaded_region_area :
  let r := 3
  let s := 2
  let d := sqrt 5
  area_of_shaded_region r s d = 9 * π / 4 + (9 - 4 * sqrt 5) / 2 :=
by
  sorry

end shaded_region_area

namespace unique_triplet

import Mathlib

theorem unique_triplet (a b p : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) :
  (1 / (p : ℚ) = 1 / (a^2 : ℚ) + 1 / (b^2 : ℚ)) → (a = 2 ∧ b = 2 ∧ p = 2) :=
by
  sorry

end unique_triplet

namespace 

import Mathlib

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem functional_eq {f : ℝ → ℝ} (h1 : ∀ x, x * (f (x + 1) - f x) = f x) (h2 : ∀ x y, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, ∀ x > 0, f x = k * x :=
sorry

end 

namespace relationship_between_a_b_c

import Mathlib

noncomputable def a := (3 / 5 : ℝ) ^ (2 / 5)
noncomputable def b := (2 / 5 : ℝ) ^ (3 / 5)
noncomputable def c := (2 / 5 : ℝ) ^ (2 / 5)

theorem relationship_between_a_b_c :
  a > c ∧ c > b :=
by
  sorry

end relationship_between_a_b_c

namespace participation_increase_closest_to_10

import Mathlib

def percentage_increase (old new : ℕ) : ℚ := ((new - old) / old) * 100

theorem participation_increase_closest_to_10 :
  (percentage_increase 80 88 = 10) ∧ 
  (percentage_increase 90 99 = 10) := by
  sorry

end participation_increase_closest_to_10

namespace det_2x2_matrix

import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Determinant

open Matrix

theorem det_2x2_matrix : 
  det ![![7, -2], ![-3, 5]] = 29 := by
  sorry

end det_2x2_matrix

namespace positive_integer_k

import Mathlib

theorem positive_integer_k (k x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = k * x * y * z) :
  k = 1 ∨ k = 3 :=
sorry

end positive_integer_k

namespace range_of_m_three_zeros

import Mathlib

noncomputable def f (x m : ℝ) : ℝ :=
if h : x < 0 then -x + m else x^2 - 1

theorem range_of_m_three_zeros (h : 0 < m) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f (f x1 m) m - 1 = 0 ∧ f (f x2 m) m - 1 = 0 ∧ f (f x3 m) m - 1 = 0) ↔ (0 < m ∧ m < 1) :=
by
  sorry

end range_of_m_three_zeros

namespace part_1_part_2

import Mathlib

-- Part (Ⅰ)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 + (m - 2) * x + 1

theorem part_1 (m : ℝ) : (∀ x : ℝ, ¬ f x m < 0) ↔ (-2 ≤ m ∧ m ≤ 6) :=
by sorry

-- Part (Ⅱ)
theorem part_2 (m : ℝ) (h_even : ∀ ⦃x : ℝ⦄, f x m = f (-x) m) :
  (m = 2) → 
  ((∀ x : ℝ, x ≤ 0 → f x 2 ≥ f 0 2) ∧ (∀ x : ℝ, x ≥ 0 → f x 2 ≥ f 0 2)) :=
by sorry

end part_1_part_2

namespace total_matches_played

import Mathlib

-- Definitions
def victories_points := 3
def draws_points := 1
def defeats_points := 0
def points_after_5_games := 8
def games_played := 5
def target_points := 40
def remaining_wins_required := 9

-- Statement to prove
theorem total_matches_played :
  ∃ M : ℕ, points_after_5_games + victories_points * remaining_wins_required < target_points -> M = games_played + remaining_wins_required + 1 :=
sorry

end total_matches_played

namespace largest_fraction

import Mathlib

variable {a b c d e f g h : ℝ}
variable {w x y z : ℝ}

/-- Given real numbers w, x, y, z such that w < x < y < z,
    the fraction z/w represents the largest value among the given fractions. -/
theorem largest_fraction (hwx : w < x) (hxy : x < y) (hyz : y < z) :
  (z / w) > (x / w) ∧ (z / w) > (y / x) ∧ (z / w) > (y / w) ∧ (z / w) > (z / x) :=
by
  sorry

end largest_fraction

namespace ratio_of_men_to_women

import Mathlib

def num_cannoneers : ℕ := 63
def num_people : ℕ := 378
def num_women (C : ℕ) : ℕ := 2 * C
def num_men (total : ℕ) (women : ℕ) : ℕ := total - women

theorem ratio_of_men_to_women : 
  let C := num_cannoneers
  let total := num_people
  let W := num_women C
  let M := num_men total W
  M / W = 2 :=
by
  sorry

end ratio_of_men_to_women

namespace percentage_error_in_area

import Mathlib

-- Definitions based on conditions
def actual_side (s : ℝ) := s
def measured_side (s : ℝ) := s * 1.01
def actual_area (s : ℝ) := s^2
def calculated_area (s : ℝ) := (measured_side s)^2

-- Theorem statement of the proof problem
theorem percentage_error_in_area (s : ℝ) : 
  (calculated_area s - actual_area s) / actual_area s * 100 = 2.01 := 
by 
  -- Proof is omitted
  sorry

end percentage_error_in_area

namespace combined_weight_is_correct

import Mathlib

def EvanDogWeight := 63
def IvanDogWeight := EvanDogWeight / 7
def CombinedWeight := EvanDogWeight + IvanDogWeight

theorem combined_weight_is_correct 
: CombinedWeight = 72 :=
by 
  sorry

end combined_weight_is_correct

namespace statues_ratio

import Mathlib.Data.Rat.Defs

theorem statues_ratio :
  let y1 := 4                  -- Number of statues after first year.
  let y2 := 4 * y1             -- Number of statues after second year.
  let y3 := (y2 + 12) - 3      -- Number of statues after third year.
  let y4 := 31                 -- Number of statues after fourth year.
  let added_fourth_year := y4 - y3  -- Statues added in the fourth year.
  let broken_third_year := 3        -- Statues broken in the third year.
  added_fourth_year / broken_third_year = 2 :=
by
  sorry

end statues_ratio

namespace plus_one_eq_next_plus

import Mathlib

theorem plus_one_eq_next_plus (m : ℕ) (h : m > 1) : (m^2 + m) + 1 = ((m + 1)^2 + (m + 1)) := by
  sorry

end plus_one_eq_next_plus

namespace learn_at_least_537_words

import Mathlib

theorem learn_at_least_537_words (total_words : ℕ) (guess_percentage : ℝ) (required_percentage : ℝ) :
  total_words = 600 → guess_percentage = 0.05 → required_percentage = 0.90 → 
  ∀ (words_learned : ℕ), words_learned ≥ 537 → 
  (words_learned + guess_percentage * (total_words - words_learned)) / total_words ≥ required_percentage :=
by
  intros h_total_words h_guess_percentage h_required_percentage words_learned h_words_learned
  sorry

end learn_at_least_537_words

namespace solution_set_of_inequality

import Mathlib

theorem solution_set_of_inequality (x : ℝ) : x * (9 - x) > 0 ↔ 0 < x ∧ x < 9 := by
  sorry

end solution_set_of_inequality

namespace correct_option_B

import Mathlib

theorem correct_option_B (a b : ℝ) : (-a^2 * b^3)^2 = a^4 * b^6 := 
  sorry

end correct_option_B

namespace unique_nonneg_sequence

import Mathlib

theorem unique_nonneg_sequence (a : List ℝ) (h_sum : 0 < a.sum) :
  ∃ b : List ℝ, (∀ x ∈ b, 0 ≤ x) ∧ 
                (∃ f : List ℝ → List ℝ, (f a = b) ∧ (∀ x y z, f (x :: y :: z :: tl) = (x + y) :: (-y) :: (z + y) :: tl)) :=
sorry

end unique_nonneg_sequence

namespace banana_price

import Mathlib

theorem banana_price (x y : ℕ) (b : ℕ) 
  (hx : x + y = 4) 
  (cost_eq : 50 * x + 60 * y + b = 275) 
  (banana_cheaper_than_pear : b < 60) 
  : b = 35 ∨ b = 45 ∨ b = 55 :=
by
  sorry

end banana_price

namespace problem_proof

import Mathlib

variable {x y z : ℝ}

theorem problem_proof (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) : 2 * (x + y + z) ≤ 3 := 
sorry

end problem_proof

namespace sum_xyz

import Mathlib 

variables {x y z : ℝ}

theorem sum_xyz (hx : x * y = 30) (hy : x * z = 60) (hz : y * z = 90) : 
  x + y + z = 11 * Real.sqrt 5 :=
sorry

end sum_xyz

namespace triangle_is_equilateral

import Mathlib

-- Define a triangle with angles A, B, and C
variables (A B C : ℝ)

-- The conditions of the problem
def log_sin_arithmetic_sequence : Prop :=
  Real.log (Real.sin A) + Real.log (Real.sin C) = 2 * Real.log (Real.sin B)

def angles_arithmetic_sequence : Prop :=
  2 * B = A + C

-- The theorem that the triangle is equilateral given these conditions
theorem triangle_is_equilateral :
  log_sin_arithmetic_sequence A B C → angles_arithmetic_sequence A B C → 
  A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end triangle_is_equilateral

namespace sum_first_m_terms_inequality_always_holds

import Mathlib

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

-- Define the sequence {a_n}
noncomputable def a_n (n m : ℕ) : ℝ := f (n / m)

-- Define the sum S_m
noncomputable def S_m (m : ℕ) : ℝ := ∑ n in Finset.range m, a_n n m

theorem sum_first_m_terms (m : ℕ) : S_m m = (1 / 12) * (3 * m - 1) := sorry

theorem inequality_always_holds (m : ℕ) (a : ℝ) (h : ∀ m : ℕ, (a^m / S_m m) < (a^(m+1) / S_m (m+1))) : a > 5/2 := sorry

end sum_first_m_terms_inequality_always_holds

namespace maximum_value_of_piecewise_function

import Mathlib

noncomputable def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + 3 else 
  if 0 < x ∧ x ≤ 1 then x + 3 else 
  -x + 5

theorem maximum_value_of_piecewise_function : ∃ M, ∀ x, piecewise_function x ≤ M ∧ (∀ y, (∀ x, piecewise_function x ≤ y) → M ≤ y) := 
by
  use 4
  sorry

end maximum_value_of_piecewise_function

namespace division_of_decimals

import Mathlib

theorem division_of_decimals : 0.18 / 0.003 = 60 :=
by
  sorry

end division_of_decimals

namespace minimum_third_highest_score

import Mathlib

theorem minimum_third_highest_score (scores : Fin 6 → ℕ) (h_uniq : Function.Injective scores)
  (h_avg : (∑ i, scores i) = 555) (h_max : ∃ i, scores i = 99) 
  (h_min : ∃ i, scores i = 76) : 
  ∃ s, s = 95 ∧ 
    ∃ (i : Fin 6), scores i = s ∧ 
    ∃ (j : Fin 6), (i ≠ j) ∧ (scores j < scores i) ∧ 
    ∃ (k : Fin 6), (i ≠ k) ∧ (j ≠ k) ∧ (scores k < scores j) :=
  sorry

end minimum_third_highest_score

namespace sum_and_difference_repeating_decimals

import Mathlib

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_9 : ℚ := 1
noncomputable def repeating_decimal_3 : ℚ := 1 / 3

theorem sum_and_difference_repeating_decimals :
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_9 + repeating_decimal_3 = 2 / 9 := 
by 
  sorry

end sum_and_difference_repeating_decimals

namespace largest_k_exists

import Mathlib

noncomputable def largest_k := 3

theorem largest_k_exists :
  ∃ (k : ℕ), (k = largest_k) ∧ ∀ m : ℕ, 
    (∀ n : ℕ, ∃ a b : ℕ, m + n = a^2 + b^2) ∧ 
    (∀ n : ℕ, ∃ seq : ℕ → ℕ,
      (∀ i : ℕ, seq i = a^2 + b^2) ∧
      (∀ j : ℕ, m ≤ j → a^2 + b^2 ≠ 3 + 4 * j)
    ) := ⟨3, rfl, sorry⟩

end largest_k_exists

namespace analytical_expression_range_of_t

import Mathlib

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem analytical_expression (x : ℝ) :
  (f (x + 1) - f x = 2 * x - 2) ∧ (f 1 = -2) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x > 0 ∧ f (x + t) < 0 → x = 1) ↔ (-2 <= t ∧ t < -1) :=
by
  sorry

end analytical_expression_range_of_t

namespace expression_equals_five

import Mathlib

theorem expression_equals_five (a : ℝ) (h : 2 * a^2 - 3 * a + 4 = 5) : 7 + 6 * a - 4 * a^2 = 5 :=
by
  sorry

end expression_equals_five

namespace solve_equation

import Mathlib

theorem solve_equation (x : ℚ) : 3 * (x - 2) = 2 - 5 * (x - 2) ↔ x = 9 / 4 := by
  sorry

end solve_equation

namespace fraction_value

import Mathlib

variable (x y : ℝ)

theorem fraction_value (h : 1/x + 1/y = 2) : (2*x + 5*x*y + 2*y) / (x - 3*x*y + y) = -9 := by
  sorry

end fraction_value

namespace soda_cost

import Mathlib

-- Definitions of the given conditions
def initial_amount : ℝ := 40
def cost_pizza : ℝ := 2.75
def cost_jeans : ℝ := 11.50
def quarters_left : ℝ := 97
def value_per_quarter : ℝ := 0.25

-- Calculate amount left in dollars
def amount_left : ℝ := quarters_left * value_per_quarter

-- Statement we want to prove: the cost of the soda
theorem soda_cost :
  initial_amount - amount_left - (cost_pizza + cost_jeans) = 1.5 :=
by
  sorry

end soda_cost

namespace cubic_sum

import Mathlib

theorem cubic_sum (p q r : ℝ) (h1 : p + q + r = 4) (h2 : p * q + q * r + r * p = 7) (h3 : p * q * r = -10) :
  p ^ 3 + q ^ 3 + r ^ 3 = 154 := 
by sorry

end cubic_sum

namespace harmonica_value

import Mathlib

theorem harmonica_value (x : ℕ) (h1 : ∃ k : ℕ, ∃ r : ℕ, x = 12 * k + r ∧ r ≠ 0 
                                                   ∧ r ≠ 6 ∧ r ≠ 9 
                                                   ∧ r ≠ 10 ∧ r ≠ 11)
                         (h2 : ¬ (x * x % 12 = 0)) : 
                         4 = 4 :=
by 
  sorry

end harmonica_value

namespace relationship_between_a_and_b

import Mathlib

open Real

theorem relationship_between_a_and_b
   (a b : ℝ)
   (ha : 0 < a ∧ a < 1)
   (hb : 0 < b ∧ b < 1)
   (hab : (1 - a) * b > 1 / 4) :
   a < b := 
sorry

end relationship_between_a_and_b

namespace f_sum_zero

import Mathlib

noncomputable def f : ℝ → ℝ := sorry

axiom f_property_1 : ∀ x : ℝ, f (x ^ 3) = (f x) ^ 3
axiom f_property_2 : ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2

theorem f_sum_zero : f 0 + f (-1) + f 1 = 0 := by
  sorry

end f_sum_zero

namespace squares_difference

import Mathlib

theorem squares_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 :=
by sorry

end squares_difference

namespace creative_sum

import Mathlib

def letterValue (ch : Char) : Int :=
  let n := (ch.toNat - 'a'.toNat + 1) % 12
  if n = 0 then 2
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 2
  else if n = 5 then 1
  else if n = 6 then 0
  else if n = 7 then -1
  else if n = 8 then -2
  else if n = 9 then -3
  else if n = 10 then -2
  else if n = 11 then -1
  else 0 -- this should never happen

def wordValue (word : String) : Int :=
  word.foldl (λ acc ch => acc + letterValue ch) 0

theorem creative_sum : wordValue "creative" = -2 :=
  by
    sorry

end creative_sum

namespace sqrt_47_minus_2_range

import Mathlib -- Avoid import errors

theorem sqrt_47_minus_2_range (h : 6 < Real.sqrt 47 ∧ Real.sqrt 47 < 7) : 4 < Real.sqrt 47 - 2 ∧ Real.sqrt 47 - 2 < 5 := by
  sorry

end sqrt_47_minus_2_range

namespace alberto_more_than_bjorn_and_charlie

import Mathlib

theorem alberto_more_than_bjorn_and_charlie (time : ℕ) 
  (alberto_speed bjorn_speed charlie_speed: ℕ) 
  (alberto_distance bjorn_distance charlie_distance : ℕ) :
  time = 6 ∧ alberto_speed = 10 ∧ bjorn_speed = 8 ∧ charlie_speed = 9
  ∧ alberto_distance = alberto_speed * time
  ∧ bjorn_distance = bjorn_speed * time
  ∧ charlie_distance = charlie_speed * time
  → (alberto_distance - bjorn_distance = 12) ∧ (alberto_distance - charlie_distance = 6) :=
by
  sorry

end alberto_more_than_bjorn_and_charlie

namespace problem_incorrect_statement_D

import Mathlib

theorem problem_incorrect_statement_D :
  (∀ x y, x = -y → x + y = 0) ∧
  (∃ x : ℕ, x^2 + 2 * x = 0) ∧
  (∀ x y : ℝ, x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  (¬ (∀ x y : ℝ, (x > 1 ∧ y > 1) ↔ (x + y > 2))) :=
by sorry

end problem_incorrect_statement_D

namespace distance_between_intersections

import Mathlib

def ellipse_eq (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 25 = 1

def is_focus_of_ellipse (fx fy : ℝ) : Prop := (fx = 0 ∧ (fy = 4 ∨ fy = -4))

def parabola_eq (x y : ℝ) : Prop := y = x^2 / 8 + 2

theorem distance_between_intersections :
  let d := 12 * Real.sqrt 2 / 5
  ∃ x1 x2 y1 y2 : ℝ, 
    ellipse_eq x1 y1 ∧ 
    parabola_eq x1 y1 ∧
    ellipse_eq x2 y2 ∧
    parabola_eq x2 y2 ∧ 
    (x2 - x1)^2 + (y2 - y1)^2 = d^2 :=
by
  sorry

end distance_between_intersections

namespace volleyballs_basketballs_difference

import Mathlib

variable (V B : ℕ)

theorem volleyballs_basketballs_difference :
  (V + B = 14) →
  (4 * V + 5 * B = 60) →
  V - B = 6 :=
by
  intros h1 h2
  sorry

end volleyballs_basketballs_difference

namespace fit_jack_apples_into_jill_basket

import Mathlib

-- Conditions:
def jack_basket_full : ℕ := 12
def jack_basket_space : ℕ := 4
def jack_current_apples : ℕ := jack_basket_full - jack_basket_space
def jill_basket_capacity : ℕ := 2 * jack_basket_full

-- Proof statement:
theorem fit_jack_apples_into_jill_basket : jill_basket_capacity / jack_current_apples = 3 :=
by {
  sorry
}

end fit_jack_apples_into_jill_basket

namespace factor_expression

import Mathlib

theorem factor_expression (x y z : ℝ) :
  x^3 * (y^2 - z^2) - y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x * y + z^2 - z * x) :=
by
  sorry

end factor_expression

namespace ordered_pair_solution

import Mathlib

theorem ordered_pair_solution :
  ∃ x y : ℤ, (x + y = (3 - x) + (3 - y)) ∧ (x - y = (x - 2) + (y - 2)) ∧ (x = 2) ∧ (y = 1) :=
by
  use 2, 1
  repeat { sorry }

end ordered_pair_solution

namespace cube_inscribed_circumscribed_volume_ratio

import Mathlib

theorem cube_inscribed_circumscribed_volume_ratio
  (S_1 S_2 V_1 V_2 : ℝ)
  (h : S_1 / S_2 = (1 / Real.sqrt 2) ^ 2) :
  V_1 / V_2 = (Real.sqrt 3 / 3) ^ 3 :=
sorry

end cube_inscribed_circumscribed_volume_ratio

namespace john_umbrella_in_car

import Mathlib

variable (UmbrellasInHouse : Nat)
variable (CostPerUmbrella : Nat)
variable (TotalAmountPaid : Nat)

theorem john_umbrella_in_car
  (h1 : UmbrellasInHouse = 2)
  (h2 : CostPerUmbrella = 8)
  (h3 : TotalAmountPaid = 24) :
  (TotalAmountPaid / CostPerUmbrella) - UmbrellasInHouse = 1 := by
  sorry

end john_umbrella_in_car

namespace find_x_collinear

import Mathlib

def vec := ℝ × ℝ

def collinear (u v: vec): Prop :=
  ∃ k: ℝ, u = (k * v.1, k * v.2)

theorem find_x_collinear:
  ∀ (x: ℝ), (let a : vec := (1, 2)
              let b : vec := (x, 1)
              collinear a (a.1 - b.1, a.2 - b.2)) → x = 1 / 2 :=
by
  intros x h
  sorry

end find_x_collinear

namespace cos_sin_value

import Mathlib

theorem cos_sin_value (α : ℝ) (h : Real.tan α = Real.sqrt 2) : Real.cos α * Real.sin α = Real.sqrt 2 / 3 :=
sorry

end cos_sin_value

namespace triangle_formation_and_acuteness

import Mathlib

variables {a b c : ℝ} {k n : ℕ}

theorem triangle_formation_and_acuteness (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 2 ≤ n) (hk : k < n) (hp : a^n + b^n = c^n) : 
  (a^k + b^k > c^k ∧ b^k + c^k > a^k ∧ c^k + a^k > b^k) ∧ (k < n / 2 → (a^k)^2 + (b^k)^2 > (c^k)^2) :=
sorry

end triangle_formation_and_acuteness

namespace fraction_identity

import Mathlib

theorem fraction_identity (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 := 
by 
  sorry

end fraction_identity

namespace minimum_sum_of_distances_squared

import Mathlib

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 0 }
def B : Point := { x := 2, y := 0 }

-- Define the moving point P on the circle
def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

-- Distance squared between two points
def dist_squared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the sum of squared distances from P to points A and B
def sum_distances_squared (P : Point) : ℝ :=
  dist_squared P A + dist_squared P B

-- Statement of the proof problem
theorem minimum_sum_of_distances_squared :
  ∃ P : Point, on_circle P ∧ sum_distances_squared P = 26 :=
sorry

end minimum_sum_of_distances_squared

namespace solve_basketball_points

import Mathlib

noncomputable def y_points_other_members (x : ℕ) : ℕ :=
  let d_points := (1 / 3) * x
  let e_points := (3 / 8) * x
  let f_points := 18
  let total := x
  total - d_points - e_points - f_points

theorem solve_basketball_points (x : ℕ) (h1: x > 0) (h2: ∃ y ≤ 24, y = y_points_other_members x) :
  ∃ y, y = 21 :=
by
  sorry

end solve_basketball_points

namespace oak_trees_remaining_is_7

import Mathlib

-- Define the number of oak trees initially in the park
def initial_oak_trees : ℕ := 9

-- Define the number of oak trees cut down by workers
def oak_trees_cut_down : ℕ := 2

-- Define the remaining oak trees calculation
def remaining_oak_trees : ℕ := initial_oak_trees - oak_trees_cut_down

-- Prove that the remaining oak trees is equal to 7
theorem oak_trees_remaining_is_7 : remaining_oak_trees = 7 := by
  sorry

end oak_trees_remaining_is_7

namespace general_admission_tickets

import Mathlib

variable (x y : ℕ)

theorem general_admission_tickets (h1 : x + y = 525) (h2 : 4 * x + 6 * y = 2876) : y = 388 := by
  sorry

end general_admission_tickets

namespace maximize_profit_marginal_profit_monotonic_decreasing

import Mathlib

-- Definition of revenue function R
def R (x : ℕ) : ℤ := 3700 * x + 45 * x^2 - 10 * x^3

-- Definition of cost function C
def C (x : ℕ) : ℤ := 460 * x + 500

-- Definition of profit function p
def p (x : ℕ) : ℤ := R x - C x

-- Lemma for the solution
theorem maximize_profit (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 20) : 
  p x = -10 * x^3 + 45 * x^2 + 3240 * x - 500 ∧ 
  (∀ y, 1 ≤ y ∧ y ≤ 20 → p y ≤ p 12) :=
by
  sorry

-- Definition of marginal profit function Mp
def Mp (x : ℕ) : ℤ := p (x + 1) - p x

-- Lemma showing Mp is monotonically decreasing
theorem marginal_profit_monotonic_decreasing (x : ℕ) (h2 : 1 ≤ x ∧ x ≤ 19) : 
  Mp x = -30 * x^2 + 60 * x + 3275 ∧ 
  ∀ y, 1 ≤ y ∧ y ≤ 19 → (Mp y ≥ Mp (y + 1)) :=
by
  sorry

end maximize_profit_marginal_profit_monotonic_decreasing

namespace scientific_notation_of_130944000000

import Mathlib

theorem scientific_notation_of_130944000000 :
  130944000000 = 1.30944 * 10^11 :=
by sorry

end scientific_notation_of_130944000000

namespace find_arrays

import Mathlib

-- Defines a condition where positive integers satisfy the given properties
def satisfies_conditions (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a ∣ b * c * d - 1 ∧ 
  b ∣ a * c * d - 1 ∧ 
  c ∣ a * b * d - 1 ∧ 
  d ∣ a * b * c - 1

-- The theorem that any four positive integers satisfying the conditions are either (2, 3, 7, 11) or (2, 3, 11, 13)
theorem find_arrays :
  ∀ a b c d : ℕ, satisfies_conditions a b c d → 
    (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨ 
    (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) :=
by
  intro a b c d h
  sorry

end find_arrays

namespace marbles_per_boy

import Mathlib

theorem marbles_per_boy (boys marbles : ℕ) (h1 : boys = 5) (h2 : marbles = 35) : marbles / boys = 7 := by
  sorry

end marbles_per_boy

namespace find_original_radius

import Mathlib

theorem find_original_radius (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 2) / 2 :=
by
  sorry

end find_original_radius

namespace intersection_of_M_and_N

import Mathlib

def M : Set ℤ := {0, 1}
def N : Set ℤ := {-1, 0}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end intersection_of_M_and_N

namespace tom_total_spent_correct

import Mathlib

-- Definitions for discount calculations
def original_price_skateboard : ℝ := 9.46
def discount_rate_skateboard : ℝ := 0.10
def discounted_price_skateboard : ℝ := original_price_skateboard * (1 - discount_rate_skateboard)

def original_price_marbles : ℝ := 9.56
def discount_rate_marbles : ℝ := 0.10
def discounted_price_marbles : ℝ := original_price_marbles * (1 - discount_rate_marbles)

def price_shorts : ℝ := 14.50

def original_price_action_figures : ℝ := 12.60
def discount_rate_action_figures : ℝ := 0.20
def discounted_price_action_figures : ℝ := original_price_action_figures * (1 - discount_rate_action_figures)

-- Total for all discounted items
def total_discounted_items : ℝ := 
  discounted_price_skateboard + discounted_price_marbles + price_shorts + discounted_price_action_figures

-- Currency conversion for video game
def price_video_game_eur : ℝ := 20.50
def exchange_rate_eur_to_usd : ℝ := 1.12
def price_video_game_usd : ℝ := price_video_game_eur * exchange_rate_eur_to_usd

-- Total amount spent including the video game
def total_spent : ℝ := total_discounted_items + price_video_game_usd

-- Lean proof statement
theorem tom_total_spent_correct :
  total_spent = 64.658 :=
by {
  -- This is a placeholder "by sorry" which means the proof is missing.
  sorry
}

end tom_total_spent_correct

namespace determine_8_genuine_coins

import Mathlib

-- Assume there are 11 coins and one may be counterfeit.
variable (coins : Fin 11 → ℝ)
variable (is_counterfeit : Fin 11 → Prop)
variable (genuine_weight : ℝ)
variable (balance : (Fin 11 → ℝ) → (Fin 11 → ℝ) → Prop)

-- The weight of genuine coins.
axiom genuine_coins_weight : ∀ i, ¬ is_counterfeit i → coins i = genuine_weight

-- The statement of the mathematical problem in Lean 4.
theorem determine_8_genuine_coins :
  ∃ (genuine_set : Finset (Fin 11)), genuine_set.card ≥ 8 ∧ ∀ i ∈ genuine_set, ¬ is_counterfeit i :=
sorry

end determine_8_genuine_coins

namespace compute_expression

import Mathlib

theorem compute_expression : 1013^2 - 987^2 - 1007^2 + 993^2 = 24000 := by
  sorry

end compute_expression

namespace find_smaller_number

import Mathlib

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) : x = 10 :=
sorry

end find_smaller_number

namespace subtraction_like_terms

import Mathlib

variable (a : ℝ)

theorem subtraction_like_terms : 3 * a ^ 2 - 2 * a ^ 2 = a ^ 2 :=
by
  sorry

end subtraction_like_terms

namespace not_solvable_det_three_times

import Mathlib

theorem not_solvable_det_three_times (a b c d : ℝ) (h : a * d - b * c = 5) :
  ¬∃ (x : ℝ), (3 * a + 1) * (3 * d + 1) - (3 * b + 1) * (3 * c + 1) = x :=
by {
  -- This is where the proof would go, but the problem states that it's not solvable with the given information.
  sorry
}

end not_solvable_det_three_times

namespace problem_1_problem_2

import Mathlib

def p (x : ℝ) : Prop := -x^2 + 6*x + 16 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 4*x + 4 - m^2 ≤ 0 ∧ m > 0

theorem problem_1 (x : ℝ) : p x → -2 ≤ x ∧ x ≤ 8 :=
by
  -- Proof goes here
  sorry

theorem problem_2 (m : ℝ) : (∀ x, p x → q x m) ∧ (∃ x, ¬ p x ∧ q x m) → m ≥ 6 :=
by
  -- Proof goes here
  sorry

end problem_1_problem_2

namespace smallest_common_term_larger_than_2023

import Mathlib

noncomputable def a_seq (n : ℕ) : ℤ :=
  3 * n - 2

noncomputable def b_seq (m : ℕ) : ℤ :=
  10 * m - 8

theorem smallest_common_term_larger_than_2023 :
  ∃ (n m : ℕ), a_seq n = b_seq m ∧ a_seq n > 2023 ∧ a_seq n = 2032 :=
by {
  sorry
}

end smallest_common_term_larger_than_2023

namespace ellipse_foci_distance

import Mathlib

theorem ellipse_foci_distance 
  (h : ∀ x y : ℝ, 9 * x^2 + y^2 = 144) : 
  ∃ c : ℝ, c = 16 * Real.sqrt 2 :=
  sorry

end ellipse_foci_distance

namespace slope_angle_at_point

import Mathlib

def f (x : ℝ) : ℝ := 2 * x^3 - 7 * x + 2

theorem slope_angle_at_point :
  let deriv_f := fun x : ℝ => 6 * x^2 - 7
  let slope := deriv_f 1
  let angle := Real.arctan slope
  angle = (3 * Real.pi) / 4 :=
by
  sorry

end slope_angle_at_point

namespace cube_volume_is_64

import Mathlib

theorem cube_volume_is_64 (a : ℕ) (h : (a - 2) * (a + 3) * a = a^3 + 12) : a^3 = 64 := 
  sorry

end cube_volume_is_64

namespace supermarket_spent_more_than_collected

import Mathlib

-- Given conditions
def initial_amount : ℕ := 53
def collected_amount : ℕ := 91
def amount_left : ℕ := 14

-- Finding the total amount before shopping and amount spent in supermarket
def total_amount : ℕ := initial_amount + collected_amount
def spent_amount : ℕ := total_amount - amount_left

-- Prove that the difference between spent amount and collected amount is 39
theorem supermarket_spent_more_than_collected : (spent_amount - collected_amount) = 39 := by
  -- The proof will go here
  sorry

end supermarket_spent_more_than_collected

namespace no_solutions_exist

import Mathlib

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 2 :=
by sorry

end no_solutions_exist

namespace average_rate_of_change

import Mathlib

def f (x : ℝ) : ℝ := x^2 - 1

theorem average_rate_of_change : (f 1.1) - (f 1) / (1.1 - 1) = 2.1 :=
by
  sorry

end average_rate_of_change

namespace f_max_a_zero_f_zero_range

import Mathlib

-- Part 1: Proving the maximum value when a = 0
theorem f_max_a_zero : ∀ (f : ℝ → ℝ) (x : ℝ),
  (f x = (-1 / x) - Real.log x) ∧ x > 0 → 
  ∃ x_max : ℝ, f x_max = -1 ∧ 
  (∀ x > 0, f x ≤ -1) := 
sorry

-- Part 2: Proving the range of a for exactly one zero of f(x)
theorem f_zero_range (a : ℝ) : (0 < a) → 
  ∀ (f : ℝ → ℝ) (x : ℝ), 
  (f x = a * x - 1 / x - (a + 1) * Real.log x) ∧ x > 0 →
  ∃! (x_zero : ℝ), f x_zero = 0 :=
sorry

end f_max_a_zero_f_zero_range

namespace octal_addition_correct

import Mathlib

def octal_to_decimal (n : ℕ) : ℕ := 
  /- function to convert an octal number to decimal goes here -/
  sorry

def decimal_to_octal (n : ℕ) : ℕ :=
  /- function to convert a decimal number to octal goes here -/
  sorry

theorem octal_addition_correct :
  let a := 236 
  let b := 521
  let c := 74
  let sum_decimal := octal_to_decimal a + octal_to_decimal b + octal_to_decimal c
  decimal_to_octal sum_decimal = 1063 :=
by
  sorry

end octal_addition_correct

namespace base_seven_sum

import Mathlib

def base_seven_to_ten (n : ℕ) : ℕ := 3 * 7^1 + 5 * 7^0   -- Converts 35_7 to base 10
def base_seven_to_ten' (m : ℕ) : ℕ := 1 * 7^1 + 2 * 7^0  -- Converts 12_7 to base 10

noncomputable def base_ten_product (a b : ℕ) : ℕ := (a * b) -- Computes product in base 10

noncomputable def base_ten_to_seven (p : ℕ) : ℕ :=        -- Converts base 10 to base 7
  let p1 := (p / 7 / 7) % 7
  let p2 := (p / 7) % 7
  let p3 := p % 7
  p1 * 100 + p2 * 10 + p3

noncomputable def sum_of_digits (a : ℕ) : ℕ :=             -- Sums digits in base 7
  let d1 := (a / 100) % 10
  let d2 := (a / 10) % 10
  let d3 := a % 10
  d1 + d2 + d3

noncomputable def base_ten_to_seven' (s : ℕ) : ℕ :=        -- Converts sum back to base 7
  let s1 := s / 7
  let s2 := s % 7
  s1 * 10 + s2

theorem base_seven_sum (n m : ℕ) : base_ten_to_seven' (sum_of_digits (base_ten_to_seven (base_ten_product (base_seven_to_ten n) (base_seven_to_ten' m)))) = 15 :=
by
  sorry

end base_seven_sum

namespace purchase_costs_10

import Mathlib

def total_cost (a b c d e : ℝ) := a + b + c + d + e
def cost_dates (a : ℝ) := 3 * a
def cost_cantaloupe (a b : ℝ) := a - b
def cost_eggs (b c : ℝ) := b + c

theorem purchase_costs_10 (a b c d e : ℝ) 
  (h_total_cost : total_cost a b c d e = 30)
  (h_cost_dates : d = cost_dates a)
  (h_cost_cantaloupe : c = cost_cantaloupe a b)
  (h_cost_eggs : e = cost_eggs b c) :
  b + c + e = 10 :=
by
  have := h_total_cost
  have := h_cost_dates
  have := h_cost_cantaloupe
  have := h_cost_eggs
  sorry

end purchase_costs_10

namespace graveling_cost_is_969

import Mathlib

-- Definitions for lawn dimensions
def lawn_length : ℝ := 75
def lawn_breadth : ℝ := 45

-- Definitions for road widths and costs
def road1_width : ℝ := 6
def road1_cost_per_sq_meter : ℝ := 0.90

def road2_width : ℝ := 5
def road2_cost_per_sq_meter : ℝ := 0.85

def road3_width : ℝ := 4
def road3_cost_per_sq_meter : ℝ := 0.80

def road4_width : ℝ := 3
def road4_cost_per_sq_meter : ℝ := 0.75

-- Calculate the area of each road
def road1_area : ℝ := road1_width * lawn_length
def road2_area : ℝ := road2_width * lawn_length
def road3_area : ℝ := road3_width * lawn_breadth
def road4_area : ℝ := road4_width * lawn_breadth

-- Calculate the cost of graveling each road
def road1_graveling_cost : ℝ := road1_area * road1_cost_per_sq_meter
def road2_graveling_cost : ℝ := road2_area * road2_cost_per_sq_meter
def road3_graveling_cost : ℝ := road3_area * road3_cost_per_sq_meter
def road4_graveling_cost : ℝ := road4_area * road4_cost_per_sq_meter

-- Calculate the total cost
def total_graveling_cost : ℝ := 
  road1_graveling_cost + road2_graveling_cost + road3_graveling_cost + road4_graveling_cost

-- Statement to be proved
theorem graveling_cost_is_969 : total_graveling_cost = 969 := by
  sorry

end graveling_cost_is_969

namespace binomial_theorem_problem_statement

import Mathlib

-- Binomial Coefficient definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Binomial Theorem
theorem binomial_theorem (a b : ℝ) (n : ℕ) : (a + b) ^ n = ∑ k in Finset.range (n + 1), binom n k • (a ^ (n - k) * b ^ k) := sorry

-- Problem Statement
theorem problem_statement (n : ℕ) : (∑ k in Finset.filter (λ x => x % 2 = 0) (Finset.range (2 * n + 1)), binom (2 * n) k * 9 ^ (k / 2)) = 2^(2*n-1) + 8^(2*n-1) := sorry

end binomial_theorem_problem_statement

namespace inequality_solution

import Mathlib

theorem inequality_solution (x : ℝ) : 
    (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
    sorry

end inequality_solution

namespace perp_lines_iff_m_values

import Mathlib

section
variables (m x y : ℝ)

def l1 := (m * x + y - 2 = 0)
def l2 := ((m + 1) * x - 2 * m * y + 1 = 0)

theorem perp_lines_iff_m_values (h1 : l1 m x y) (h2 : l2 m x y) (h_perp : (m * (m + 1) + (-2 * m) = 0)) : m = 0 ∨ m = 1 :=
by {
  sorry
}
end

end perp_lines_iff_m_values

namespace compute_fg

import Mathlib

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 3 * x + 4

theorem compute_fg : f (g (-3)) = 25 := by
  sorry

end compute_fg

namespace johns_father_age

import Mathlib

variable {Age : Type} [OrderedRing Age]
variables (J M F : Age)

def john_age := J
def mother_age := M
def father_age := F

def john_younger_than_father (F J : Age) : Prop := F = 2 * J
def father_older_than_mother (F M : Age) : Prop := F = M + 4
def age_difference_between_john_and_mother (M J : Age) : Prop := M = J + 16

-- The question to be proved in Lean:
theorem johns_father_age :
  john_younger_than_father F J →
  father_older_than_mother F M →
  age_difference_between_john_and_mother M J →
  F = 40 := 
by
  intros h1 h2 h3
  sorry

end johns_father_age

namespace polynomial_g

import Mathlib

def f (x : ℝ) : ℝ := x^2

theorem polynomial_g (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by
  sorry

end polynomial_g

namespace number_of_ways_to_select_starting_lineup

import Mathlib

noncomputable def choose (n k : ℕ) : ℕ := 
if h : k ≤ n then Nat.choose n k else 0

theorem number_of_ways_to_select_starting_lineup (n k : ℕ) (h : n = 12) (h1 : k = 5) : 
  12 * choose 11 4 = 3960 := 
by sorry

end number_of_ways_to_select_starting_lineup

namespace distance_rowed_upstream

import Mathlib

noncomputable def speed_of_boat_in_still_water := 18 -- from solution step; b = 18 km/h
def speed_of_stream := 3 -- given
def time := 4 -- given
def distance_downstream := 84 -- given

theorem distance_rowed_upstream 
  (b : ℕ) (s : ℕ) (t : ℕ) (d_down : ℕ) (d_up : ℕ)
  (h_stream : s = 3) 
  (h_time : t = 4)
  (h_distance_downstream : d_down = 84) 
  (h_speed_boat : b = 18) 
  (h_effective_downstream_speed : b + s = d_down / t) :
  d_up = 60 := by
  sorry

end distance_rowed_upstream

namespace polar_coordinates_of_point

import Mathlib

theorem polar_coordinates_of_point (x y : ℝ) (hx : x = 2) (hy : y = -2 * √3) : 
  ∃ (ρ θ : ℝ), ρ = 4 ∧ θ = -2 * Real.pi / 3 ∧ (ρ * Real.cos θ, ρ * Real.sin θ) = (x, y) :=
by 
  use 4
  use -2 * Real.pi / 3
  sorry

end polar_coordinates_of_point

namespace li_to_zhang

import Mathlib

theorem li_to_zhang :
  (∀ (meter chi : ℕ), 3 * meter = chi) →
  (∀ (zhang chi : ℕ), 10 * zhang = chi) →
  (∀ (kilometer li : ℕ), 2 * li = kilometer) →
  (1 * lin = 150 * zhang) :=
by
  intro h_meter h_zhang h_kilometer
  sorry

end li_to_zhang

namespace opposite_sides_line

import Mathlib

theorem opposite_sides_line (a : ℝ) : (0 + 0 - a) * (1 + 1 - a) < 0 → 0 < a ∧ a < 2 := by
  sorry

end opposite_sides_line

namespace calculate_expression

import Mathlib

theorem calculate_expression :
  -1 ^ 4 + ((-1 / 2) ^ 2 * |(-5 + 3)|) / ((-1 / 2) ^ 3) = -5 := by
  sorry

end calculate_expression

namespace remainder_when_divided_by_2000

import Mathlib

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

noncomputable def count_disjoint_subsets (S : Set ℕ) : ℕ :=
  let totalWays := 3^12
  let emptyACases := 2*2^12
  let bothEmptyCase := 1
  (totalWays - emptyACases + bothEmptyCase) / 2

theorem remainder_when_divided_by_2000 : count_disjoint_subsets S % 2000 = 1625 := by
  sorry

end remainder_when_divided_by_2000

namespace soja_book_page_count

import Mathlib

theorem soja_book_page_count (P : ℕ) (h1 : P > 0) (h2 : (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 100) : P = 300 :=
by
  -- The Lean proof is not required, so we just add sorry to skip the proof
  sorry

end soja_book_page_count

namespace regular_bike_wheels_eq_two

import Mathlib

-- Conditions
def regular_bikes : ℕ := 7
def childrens_bikes : ℕ := 11
def wheels_per_childrens_bike : ℕ := 4
def total_wheels_seen : ℕ := 58

-- Define the problem
theorem regular_bike_wheels_eq_two 
  (w : ℕ)
  (h1 : total_wheels_seen = regular_bikes * w + childrens_bikes * wheels_per_childrens_bike) :
  w = 2 :=
by
  -- Proof steps would go here
  sorry

end regular_bike_wheels_eq_two

namespace age_difference

import Mathlib

theorem age_difference (a b : ℕ) (ha : a < 10) (hb : b < 10)
  (h1 : 10 * a + b + 10 = 3 * (10 * b + a + 10)) :
  10 * a + b - (10 * b + a) = 54 :=
by sorry

end age_difference

namespace ring_stack_distance

import Mathlib

noncomputable def vertical_distance (rings : Nat) : Nat :=
  let diameters := List.range rings |>.map (λ i => 15 - 2 * i)
  let thickness := 1 * rings
  thickness

theorem ring_stack_distance :
  vertical_distance 7 = 58 :=
by 
  sorry

end ring_stack_distance

namespace min_value_of_expression

import Mathlib

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 1) :
  (2 / x + 5 / y) ≥ 2 := sorry

end min_value_of_expression

namespace part_a_part_b_part_c_part_d_part_e_part_f

import Mathlib

-- Part (a)
theorem part_a (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^2 = 5 * k + 1 ∨ n^2 = 5 * k - 1 := 
sorry

-- Part (b)
theorem part_b (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^4 - 1 = 5 * k := 
sorry

-- Part (c)
theorem part_c (n : ℤ) : n^5 % 10 = n % 10 := 
sorry

-- Part (d)
theorem part_d (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k := 
sorry

-- Part (e)
theorem part_e (k n : ℤ) (h1 : ¬ ∃ j : ℤ, k = 5 * j) (h2 : ¬ ∃ j : ℤ, n = 5 * j) : ∃ j : ℤ, k^4 - n^4 = 5 * j := 
sorry

-- Part (f)
theorem part_f (k m n : ℤ) (h : k^2 + m^2 = n^2) : ∃ j : ℤ, k = 5 * j ∨ ∃ r : ℤ, m = 5 * r ∨ ∃ s : ℤ, n = 5 * s := 
sorry

end part_a_part_b_part_c_part_d_part_e_part_f

namespace number_of_arrangements

import Mathlib

theorem number_of_arrangements (P : Fin 5 → Type) (youngest : Fin 5) 
  (h_in_not_first_last : ∀ (i : Fin 5), i ≠ 0 → i ≠ 4 → i ≠ youngest) : 
  ∃ n, n = 72 := 
by
  sorry

end number_of_arrangements

namespace gate_distance_probability_correct

import Mathlib

-- Define the number of gates
def num_gates : ℕ := 15

-- Define the distance between adjacent gates
def distance_between_gates : ℕ := 80

-- Define the maximum distance Dave can walk
def max_distance : ℕ := 320

-- Define the function that calculates the probability
def calculate_probability (num_gates : ℕ) (distance_between_gates : ℕ) (max_distance : ℕ) : ℚ :=
  let total_pairs := num_gates * (num_gates - 1)
  let valid_pairs :=
    2 * (4 + 5 + 6 + 7) + 7 * 8
  valid_pairs / total_pairs

-- Assert the relevant result and stated answer
theorem gate_distance_probability_correct :
  let m := 10
  let n := 21
  let probability := calculate_probability num_gates distance_between_gates max_distance
  m + n = 31 ∧ probability = (10 / 21 : ℚ) :=
by
  sorry

end gate_distance_probability_correct

namespace smallest_k_for_min_period_15

import Mathlib

/-- Rational number with minimal period -/
def is_minimal_period (r : ℚ) (n : ℕ) : Prop :=
  ∃ m : ℤ, r = m / (10^n - 1)

variables (a b : ℚ)

-- Conditions for a and b
axiom ha : is_minimal_period a 30
axiom hb : is_minimal_period b 30

-- Condition for a - b
axiom hab_min_period : is_minimal_period (a - b) 15

-- Conclusion
theorem smallest_k_for_min_period_15 : ∃ k : ℕ, k = 6 ∧ is_minimal_period (a + k * b) 15 :=
by sorry

end smallest_k_for_min_period_15

namespace complement_of_A_in_U

import Mathlib

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | |x - 1| > 2 }

theorem complement_of_A_in_U : 
  ∀ x, x ∈ U → x ∈ U \ A ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

end complement_of_A_in_U

namespace total_amount_paid

import Mathlib

theorem total_amount_paid :
  let pizzas := 3
  let cost_per_pizza := 8
  let total_cost := pizzas * cost_per_pizza
  total_cost = 24 :=
by
  sorry

end total_amount_paid

namespace arithmetic_geometric_relation

import Mathlib

variable (a₁ a₂ b₁ b₂ b₃ : ℝ)

-- Conditions
def is_arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ (d : ℝ), -2 + d = a₁ ∧ a₁ + d = a₂ ∧ a₂ + d = -8

def is_geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ (r : ℝ), -2 * r = b₁ ∧ b₁ * r = b₂ ∧ b₂ * r = b₃ ∧ b₃ * r = -8

-- The problem statement
theorem arithmetic_geometric_relation (h₁ : is_arithmetic_sequence a₁ a₂) (h₂ : is_geometric_sequence b₁ b₂ b₃) :
  (a₂ - a₁) / b₂ = 1 / 2 := by
    sorry

end arithmetic_geometric_relation

namespace train_length_proof

import Mathlib

-- Define the conditions
def time_to_cross := 12 -- Time in seconds
def speed_km_per_h := 75 -- Speed in km/h

-- Convert the speed to m/s
def speed_m_per_s := speed_km_per_h * (5 / 18 : ℚ)

-- The length of the train using the formula: length = speed * time
def length_of_train := speed_m_per_s * (time_to_cross : ℚ)

-- The theorem to prove
theorem train_length_proof : length_of_train = 250 := by
  sorry

end train_length_proof

namespace race_distance

import Mathlib

theorem race_distance (D : ℝ)
  (A_speed : ℝ := D / 20)
  (B_speed : ℝ := D / 25)
  (A_beats_B_by : ℝ := 18)
  (h1 : A_speed * 25 = D + A_beats_B_by)
  : D = 72 := 
by
  sorry

end race_distance

namespace minimum_norm_of_v

import Mathlib

open Real 

-- Define the vector v and condition
noncomputable def v : ℝ × ℝ := sorry

-- Define the condition
axiom v_condition : ‖(v.1 + 4, v.2 + 2)‖ = 10

-- The statement that we need to prove
theorem minimum_norm_of_v : ‖v‖ = 10 - 2 * sqrt 5 :=
by
  sorry

end minimum_norm_of_v

namespace stations_visited

import Mathlib

-- Define the total number of nails
def total_nails : ℕ := 560

-- Define the number of nails left at each station
def nails_per_station : ℕ := 14

-- Main theorem statement
theorem stations_visited : total_nails / nails_per_station = 40 := by
  sorry

end stations_visited

namespace value_of_a

import Mathlib

theorem value_of_a (a b c : ℕ) (h1 : a + b = 12) (h2 : b + c = 16) (h3 : c = 7) : a = 3 := by
  sorry

end value_of_a

namespace coloring_15_segments_impossible

import Mathlib

theorem coloring_15_segments_impossible :
  ¬ ∃ (colors : Fin 15 → Fin 3) (adj : Fin 15 → Fin 2),
    ∀ i j, adj i = adj j → i ≠ j → colors i ≠ colors j :=
by
  sorry

end coloring_15_segments_impossible

namespace jenny_hours_left

import Mathlib

theorem jenny_hours_left 
    (h_research : ℕ := 10)
    (h_proposal : ℕ := 2)
    (h_visual_aids : ℕ := 5)
    (h_editing : ℕ := 3)
    (h_total : ℕ := 25) :
    h_total - (h_research + h_proposal + h_visual_aids + h_editing) = 5 := by
  sorry

end jenny_hours_left

namespace total_amount_is_4200

import Mathlib

variables (p q r : ℕ)
variable (total_amount : ℕ)
variable (r_has_two_thirds : total_amount / 3 * 2 = 2800)
variable (r_value : r = 2800)

theorem total_amount_is_4200 (h1 : total_amount / 3 * 2 = 2800) (h2 : r = 2800) : total_amount = 4200 :=
by
  sorry

end total_amount_is_4200

namespace part1_part2

import Mathlib

-- Define all given conditions
variable {A B C AC BC : ℝ}
variable (A_in_range : 0 < A ∧ A < π/2)
variable (B_in_range : 0 < B ∧ B < π/2)
variable (C_in_range : 0 < C ∧ C < π/2)
variable (m_perp_n : (Real.cos (A + π/3) * Real.cos B) + (Real.sin (A + π/3) * Real.sin B) = 0)
variable (cos_B : Real.cos B = 3/5)
variable (AC_value : AC = 8)

-- First part: Prove A - B = π/6
theorem part1 : A - B = π / 6 :=
by
  sorry

-- Second part: Prove BC = 4√3 + 3 given additional conditions
theorem part2 : BC = 4 * Real.sqrt 3 + 3 :=
by
  sorry

end part1_part2

namespace election_votes_total

import Mathlib

-- Definitions representing the conditions
def CandidateAVotes (V : ℕ) := 45 * V / 100
def CandidateBVotes (V : ℕ) := 35 * V / 100
def CandidateCVotes (V : ℕ) := 20 * V / 100

-- Main theorem statement
theorem election_votes_total (V : ℕ) (h1: CandidateAVotes V = 45 * V / 100) (h2: CandidateBVotes V = 35 * V / 100) (h3: CandidateCVotes V = 20 * V / 100)
  (h4: CandidateAVotes V - CandidateBVotes V = 1800) : V = 18000 :=
  sorry

end election_votes_total

namespace jellybeans_problem

import Mathlib

theorem jellybeans_problem (n : ℕ) (h : n ≥ 100) (h_mod : n % 13 = 11) : n = 102 :=
sorry

end jellybeans_problem

namespace apples_per_sandwich

import Mathlib

-- Define the conditions
def sam_sandwiches_per_day : Nat := 10
def days_in_week : Nat := 7
def total_apples_in_week : Nat := 280

-- Calculate total sandwiches in a week
def total_sandwiches_in_week := sam_sandwiches_per_day * days_in_week

-- Prove that Sam eats 4 apples for each sandwich
theorem apples_per_sandwich : total_apples_in_week / total_sandwiches_in_week = 4 :=
  by
    sorry

end apples_per_sandwich

namespace number_of_people

import Mathlib

theorem number_of_people (x : ℕ) (h1 : 175 = 175) (h2: 2 = 2) (h3 : ∀ (p : ℕ), p * x = 175 + p * 10) : x = 7 :=
sorry

end number_of_people

namespace sin_alpha_neg_point_two

import Mathlib

theorem sin_alpha_neg_point_two (a : ℝ) (h : Real.sin (Real.pi + a) = 0.2) : Real.sin a = -0.2 := 
by
  sorry

end sin_alpha_neg_point_two

namespace area_triangle_MNR

import Mathlib

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

/-- Given the quadrilateral PQRS with the midpoints M and N of PQ and QR 
and specified lengths, prove the calculated area of triangle MNR. -/
theorem area_triangle_MNR : 
  let P : (ℝ × ℝ) := (0, 5)
  let Q : (ℝ × ℝ) := (10, 5)
  let R : (ℝ × ℝ) := (14, 0)
  let S : (ℝ × ℝ) := (7, 0)
  let M : (ℝ × ℝ) := (5, 5)  -- Midpoint of PQ
  let N : (ℝ × ℝ) := (12, 2.5) -- Midpoint of QR
  distance M.fst M.snd N.fst N.snd = 7.435 →
  ((5 - 0 : ℝ) / 2 = 2.5) →
  (1 / 2 * 7.435 * 2.5) = 9.294375 :=
by
  sorry

end area_triangle_MNR

namespace min_value_a_plus_3b

import Mathlib

theorem min_value_a_plus_3b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 * a * b - 3 = a + 3 * b) :
  ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, y = a + 3 * b → y ≥ 6 :=
sorry

end min_value_a_plus_3b

namespace students_per_bench

import Mathlib

-- Definitions based on conditions
def num_male_students : ℕ := 29
def num_female_students : ℕ := 4 * num_male_students
def num_benches : ℕ := 29
def total_students : ℕ := num_male_students + num_female_students

-- Theorem to prove
theorem students_per_bench : total_students / num_benches = 5 := by
  sorry

end students_per_bench

namespace pqrs_sum

import Mathlib

/--
Given two pairs of real numbers (x, y) satisfying the equations:
1. x + y = 6
2. 2xy = 6

Prove that the solutions for x in the form x = (p ± q * sqrt(r)) / s give p + q + r + s = 11.
-/
theorem pqrs_sum : ∃ (p q r s : ℕ), (∀ (x y : ℝ), x + y = 6 ∧ 2*x*y = 6 → 
  (x = (p + q * Real.sqrt r) / s) ∨ (x = (p - q * Real.sqrt r) / s)) ∧ 
  p + q + r + s = 11 := 
sorry

end pqrs_sum

namespace total_distance

import Mathlib

theorem total_distance (x : ℝ) (h : (1/2) * (x - 1) = (1/3) * x + 1) : x = 9 := 
by 
  sorry

end total_distance

namespace cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths

import Mathlib

theorem cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths 
  (b : ℝ)
  (h : ∀ x : ℝ, 4 * x^3 + 3 * x^2 + b * x + 27 = 0 → ∃! r : ℝ, r = x) :
  b = 3 / 4 := 
by
  sorry

end cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths

namespace marek_sequence_sum

import Mathlib

theorem marek_sequence_sum (x : ℝ) :
  let a := x
  let b := (a + 4) / 4 - 4
  let c := (b + 4) / 4 - 4
  let d := (c + 4) / 4 - 4
  (a + 4) / 4 * 4 + (b + 4) / 4 * 4 + (c + 4) / 4 * 4 + (d + 4) / 4 * 4 = 80 →
  x = 38 :=
by
  sorry

end marek_sequence_sum

namespace tommys_family_members

import Mathlib

-- Definitions
def ounces_per_member : ℕ := 16
def ounces_per_steak : ℕ := 20
def steaks_needed : ℕ := 4

-- Theorem statement
theorem tommys_family_members : (steaks_needed * ounces_per_steak) / ounces_per_member = 5 :=
by
  -- Proof goes here
  sorry

end tommys_family_members

namespace units_digit_sum_factorials

import Mathlib

-- Definitions based on the conditions
def units_digit (n : ℕ) : ℕ := n % 10

-- Lean statement to represent the proof problem
theorem units_digit_sum_factorials :
  units_digit (∑ n in Finset.range 2024, n.factorial) = 3 :=
by 
  sorry

end units_digit_sum_factorials

namespace travel_time_total

import Mathlib

theorem travel_time_total (dist1 dist2 dist3 speed1 speed2 speed3 : ℝ)
  (h_dist1 : dist1 = 50) (h_dist2 : dist2 = 100) (h_dist3 : dist3 = 150)
  (h_speed1 : speed1 = 50) (h_speed2 : speed2 = 80) (h_speed3 : speed3 = 120) :
  dist1 / speed1 + dist2 / speed2 + dist3 / speed3 = 3.5 :=
by
  sorry

end travel_time_total

namespace jordan_trapezoid_height

import Mathlib

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

theorem jordan_trapezoid_height :
  ∀ (h : ℕ),
    rectangle_area 5 24 = trapezoid_area 2 6 h →
    h = 30 :=
by
  intro h
  intro h_eq
  sorry

end jordan_trapezoid_height

namespace cat_clothing_probability

import Mathlib.Data.Real.Basic

-- Define the conditions as Lean definitions
def n_items : ℕ := 3
def total_legs : ℕ := 4
def favorable_outcomes_per_leg : ℕ := 1
def possible_outcomes_per_leg : ℕ := (n_items.factorial : ℕ)
def probability_per_leg : ℚ := favorable_outcomes_per_leg / possible_outcomes_per_leg

-- Theorem statement to show the combined probability for all legs
theorem cat_clothing_probability
    (n_items_eq : n_items = 3)
    (total_legs_eq : total_legs = 4)
    (fact_n_items : (n_items.factorial) = 6)
    (prob_leg_eq : probability_per_leg = 1 / 6) :
    (probability_per_leg ^ total_legs = 1 / 1296) := by
    sorry

end cat_clothing_probability

namespace triangle_area_is_2

import Mathlib

noncomputable def area_of_triangle_OAB {x₀ : ℝ} (h₀ : 0 < x₀) : ℝ :=
  let y₀ := 1 / x₀
  let slope := -1 / x₀^2
  let tangent_line (x : ℝ) := y₀ + slope * (x - x₀)
  let A : ℝ × ℝ := (2 * x₀, 0) -- Intersection with x-axis
  let B : ℝ × ℝ := (0, 2 * y₀) -- Intersection with y-axis
  1 / 2 * abs (2 * y₀ * 2 * x₀)

theorem triangle_area_is_2 (x₀ : ℝ) (h₀ : 0 < x₀) : area_of_triangle_OAB h₀ = 2 :=
by
  sorry

end triangle_area_is_2

namespace christian_age_in_eight_years

import Mathlib

-- Definitions from the conditions
def christian_current_age : ℕ := 72
def brian_age_in_eight_years : ℕ := 40

-- Theorem to prove
theorem christian_age_in_eight_years : ∃ (age : ℕ), age = christian_current_age + 8 ∧ age = 80 := by
  sorry

end christian_age_in_eight_years

namespace linear_function_passing_origin

import Mathlib

theorem linear_function_passing_origin (m : ℝ) :
  (∃ (y x : ℝ), y = -2 * x + (m - 5) ∧ y = 0 ∧ x = 0) → m = 5 :=
by
  sorry

end linear_function_passing_origin

namespace math_proof

import Mathlib

variable {a b c A B C : ℝ}
variable {S : ℝ}

noncomputable def problem_statement (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) : Prop :=
    (∃ A B : ℝ, (A = 2 * B) ∧ (A = 90)) 

theorem math_proof (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) :
    problem_statement h1 h2 :=
    sorry

end math_proof

namespace ratio_a_c

import Mathlib

variables (a b c d : ℚ)

axiom ratio_a_b : a / b = 5 / 4
axiom ratio_c_d : c / d = 4 / 3
axiom ratio_d_b : d / b = 1 / 8

theorem ratio_a_c : a / c = 15 / 2 :=
by sorry

end ratio_a_c

namespace prob_draw

import Mathlib

-- Define the probabilities as constants
def prob_A_winning : ℝ := 0.4
def prob_A_not_losing : ℝ := 0.9

-- Prove that the probability of a draw is 0.5
theorem prob_draw : prob_A_not_losing - prob_A_winning = 0.5 :=
by sorry

end prob_draw

namespace num_marked_cells_at_least_num_cells_in_one_square

import Mathlib

-- Defining the total number of squares
def num_squares : ℕ := 2009

-- A square covers a cell if it is within its bounds.
-- A cell is marked if it is covered by an odd number of squares.
-- We have to show that the number of marked cells is at least the number of cells in one square.
theorem num_marked_cells_at_least_num_cells_in_one_square (side_length : ℕ) : 
  side_length * side_length ≤ (num_squares : ℕ) :=
sorry

end num_marked_cells_at_least_num_cells_in_one_square

namespace find_N

import Mathlib

theorem find_N (N x : ℝ) (h1 : N / (1 + 4 / x) = 1) (h2 : x = 0.5) : N = 9 := 
by 
  sorry

end find_N

namespace alpha_beta_value

import Mathlib

theorem alpha_beta_value :
  ∃ α β : ℝ, (α^2 - 2 * α - 4 = 0) ∧ (β^2 - 2 * β - 4 = 0) ∧ (α + β = 2) ∧ (α^3 + 8 * β + 6 = 30) :=
by
  sorry

end alpha_beta_value

namespace find_angle_C

import Mathlib

variable {A B C : ℝ} -- Angles of triangle ABC
variable {a b c : ℝ} -- Sides opposite the respective angles

theorem find_angle_C
  (h1 : 2 * c * Real.cos B = 2 * a + b) : 
  C = 120 :=
  sorry

end find_angle_C

namespace div_by_133

import Mathlib

theorem div_by_133 (n : ℕ) : 133 ∣ 11^(n+2) + 12^(2*n+1) :=
by sorry

end div_by_133

namespace find_principal

import Mathlib

theorem find_principal (P : ℝ) (r : ℝ) (t : ℝ) (CI SI : ℝ) 
  (h_r : r = 0.20) 
  (h_t : t = 2) 
  (h_diff : CI - SI = 144) 
  (h_CI : CI = P * (1 + r)^t - P) 
  (h_SI : SI = P * r * t) : 
  P = 3600 :=
by
  sorry

end find_principal

namespace increasing_interval

import Mathlib

-- Define the function f(x) = x^2 + 2*(a - 1)*x
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*(a - 1)*x

-- Define the condition for f(x) being increasing on [4, +∞)
def is_increasing_on_interval (a : ℝ) : Prop := 
  ∀ x y : ℝ, 4 ≤ x → x ≤ y → 
    f x a ≤ f y a

-- Define the main theorem that we need to prove
theorem increasing_interval (a : ℝ) (h : is_increasing_on_interval a) : -3 ≤ a :=
by 
  sorry -- proof is required, but omitted as per the instruction.

end increasing_interval

namespace perpendicular_lines

import Mathlib

theorem perpendicular_lines (a : ℝ) : 
  (2 * (a + 1) * a + a * 2 = 0) ↔ (a = -2 ∨ a = 0) :=
by 
  sorry

end perpendicular_lines

namespace only_exprC_cannot_be_calculated_with_square_of_binomial

import Mathlib

-- Definitions of our expressions using their variables
def exprA (a b : ℝ) := (a + b) * (a - b)
def exprB (x : ℝ) := (-x + 1) * (-x - 1)
def exprC (y : ℝ) := (y + 1) * (-y - 1)
def exprD (m : ℝ) := (m - 1) * (-1 - m)

-- Statement that only exprC cannot be calculated using the square of a binomial formula
theorem only_exprC_cannot_be_calculated_with_square_of_binomial :
  (∀ a b : ℝ, ∃ (u v : ℝ), exprA a b = u^2 - v^2) ∧
  (∀ x : ℝ, ∃ (u v : ℝ), exprB x = u^2 - v^2) ∧
  (forall m : ℝ, ∃ (u v : ℝ), exprD m = u^2 - v^2) 
  ∧ (∀ v : ℝ, ¬ ∃ (u : ℝ), exprC v = u^2 ∨ (exprC v = - (u^2))) := sorry

end only_exprC_cannot_be_calculated_with_square_of_binomial

namespace thalassa_population_2050

import Mathlib

def population_in_2000 : ℕ := 250

def population_doubling_interval : ℕ := 20

def population_linear_increase_interval : ℕ := 10

def linear_increase_amount : ℕ := 500

noncomputable def population_in_2050 : ℕ :=
  let double1 := population_in_2000 * 2
  let double2 := double1 * 2
  double2 + linear_increase_amount

theorem thalassa_population_2050 : population_in_2050 = 1500 := by
  sorry

end thalassa_population_2050

namespace john_spends_40_dollars

import Mathlib

-- Definitions based on conditions
def cost_per_loot_box : ℝ := 5
def average_value_per_loot_box : ℝ := 3.5
def average_loss : ℝ := 12

-- Prove the amount spent on loot boxes is $40
theorem john_spends_40_dollars :
  ∃ S : ℝ, (S * (cost_per_loot_box - average_value_per_loot_box) / cost_per_loot_box = average_loss) ∧ S = 40 :=
by
  sorry

end john_spends_40_dollars

namespace simplify_expression

import Mathlib

variable (y : ℝ)

theorem simplify_expression : (3 * y)^3 + (4 * y) * (y^2) - 2 * y^3 = 29 * y^3 :=
by
  sorry

end simplify_expression

namespace ratio_of_packets_to_tent_stakes

import Mathlib

-- Definitions based on the conditions provided
def total_items (D T W : ℕ) : Prop := D + T + W = 22
def tent_stakes (T : ℕ) : Prop := T = 4
def bottles_of_water (W T : ℕ) : Prop := W = T + 2

-- The goal is to prove the ratio of packets of drink mix to tent stakes
theorem ratio_of_packets_to_tent_stakes (D T W : ℕ) :
  total_items D T W →
  tent_stakes T →
  bottles_of_water W T →
  D = 3 * T :=
by
  sorry

end ratio_of_packets_to_tent_stakes

namespace yolanda_walking_rate

import Mathlib

theorem yolanda_walking_rate 
  (d_xy : ℕ) (bob_start_after_yolanda : ℕ) (bob_distance_walked : ℕ) 
  (bob_rate : ℕ) (y : ℕ) 
  (bob_distance_to_time : bob_rate ≠ 0 ∧ bob_distance_walked / bob_rate = 2) 
  (yolanda_distance_walked : d_xy - bob_distance_walked = 9 ∧ y = 9 / 3) : 
  y = 3 :=
by 
  sorry

end yolanda_walking_rate

namespace factorize_ab_factorize_x

import Mathlib

-- Problem 1: Factorization of a^3 b - 2 a^2 b^2 + a b^3
theorem factorize_ab (a b : ℤ) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = a * b * (a - b)^2 := 
by sorry

-- Problem 2: Factorization of (x^2 + 4)^2 - 16 x^2
theorem factorize_x (x : ℤ) : (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 :=
by sorry

end factorize_ab_factorize_x

namespace mary_principal_amount

import Mathlib

theorem mary_principal_amount (t1 t2 t3 t4:ℕ) (P R:ℕ) :
  (t1 = 2) →
  (t2 = 260) →
  (t3 = 5) →
  (t4 = 350) →
  (P + 2 * P * R = t2) →
  (P + 5 * P * R = t4) →
  P = 200 :=
by
  intros
  sorry

end mary_principal_amount

namespace min_value_f

import Mathlib

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x)^2 / (Real.cos x * Real.sin x - (Real.sin x)^2)

theorem min_value_f :
  ∃ x : ℝ, 0 < x ∧ x < Real.pi / 4 ∧ f x = 4 := 
sorry

end min_value_f

namespace tommy_gum_given

import Mathlib

variable (original_gum : ℕ) (luis_gum : ℕ) (final_total_gum : ℕ)

-- Defining the conditions
def conditions := original_gum = 25 ∧ luis_gum = 20 ∧ final_total_gum = 61

-- The theorem stating that Tommy gave Maria 16 pieces of gum
theorem tommy_gum_given (t_gum : ℕ) (h : conditions original_gum luis_gum final_total_gum) :
  t_gum = final_total_gum - (original_gum + luis_gum) → t_gum = 16 :=
by
  intros h
  sorry

end tommy_gum_given

namespace find_g_of_3

import Mathlib

theorem find_g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 2 * g x - 5 * g (1 / x) = 2 * x) : g 3 = -32 / 63 :=
by sorry

end find_g_of_3

namespace kamal_average_marks

import Mathlib

theorem kamal_average_marks :
  let total_marks_obtained := 66 + 65 + 77 + 62 + 75 + 58
  let total_max_marks := 150 + 120 + 180 + 140 + 160 + 90
  (total_marks_obtained / total_max_marks.toFloat) * 100 = 48.0 :=
by
  sorry

end kamal_average_marks

namespace problem

import Mathlib

theorem problem (a : ℝ) (h : a^2 - 2 * a - 1 = 0) : -3 * a^2 + 6 * a + 5 = 2 := by
  sorry

end problem

namespace area_of_fig_between_x1_and_x2

import Mathlib

noncomputable def area_under_curve_x2 (a b : ℝ) : ℝ :=
∫ x in a..b, x^2

theorem area_of_fig_between_x1_and_x2 :
  area_under_curve_x2 1 2 = 7 / 3 := by
  sorry

end area_of_fig_between_x1_and_x2

namespace calculate_bmw_sales_and_revenue

import Mathlib

variable (total_cars : ℕ) (percentage_ford percentage_toyota percentage_nissan percentage_audi : ℕ) (avg_price_bmw : ℕ)
variable (h_total_cars : total_cars = 300) (h_percentage_ford : percentage_ford = 10)
variable (h_percentage_toyota : percentage_toyota = 25) (h_percentage_nissan : percentage_nissan = 20)
variable (h_percentage_audi : percentage_audi = 15) (h_avg_price_bmw : avg_price_bmw = 35000)

theorem calculate_bmw_sales_and_revenue :
  let percentage_non_bmw := percentage_ford + percentage_toyota + percentage_nissan + percentage_audi
  let percentage_bmw := 100 - percentage_non_bmw
  let number_bmw := total_cars * percentage_bmw / 100
  let total_revenue := number_bmw * avg_price_bmw
  (number_bmw = 90) ∧ (total_revenue = 3150000) := by
  -- Definitions are taken from conditions and used directly in the theorem statement
  sorry

end calculate_bmw_sales_and_revenue

namespace conference_games

import Mathlib

theorem conference_games (teams_per_division : ℕ) (divisions : ℕ) 
  (intradivision_games_per_team : ℕ) (interdivision_games_per_team : ℕ) 
  (total_teams : ℕ) (total_games : ℕ) : 
  total_teams = teams_per_division * divisions →
  intradivision_games_per_team = (teams_per_division - 1) * 2 →
  interdivision_games_per_team = teams_per_division →
  total_games = (total_teams * (intradivision_games_per_team + interdivision_games_per_team)) / 2 →
  total_games = 133 :=
by
  intros
  sorry

end conference_games

namespace factor_complete_polynomial

import Mathlib

theorem factor_complete_polynomial :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x^2 - 20 * x + 77) :=
sorry

end factor_complete_polynomial

namespace union_of_sets

import Mathlib

def A : Set ℝ := {x | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x | 4 < x ∧ x ≤ 10}

theorem union_of_sets :
  A ∪ B = {x | 3 < x ∧ x ≤ 10} :=
by
  sorry

end union_of_sets

namespace three_monotonic_intervals_iff_a_lt_zero

import Mathlib

-- Definition of the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Definition of the first derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Main statement: Prove that f(x) has exactly three monotonic intervals if and only if a < 0.
theorem three_monotonic_intervals_iff_a_lt_zero (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' a x1 = 0 ∧ f' a x2 = 0) ↔ a < 0 :=
by
  sorry

end three_monotonic_intervals_iff_a_lt_zero

namespace junk_items_count

import Mathlib

variable (total_items : ℕ)
variable (useful_percentage : ℚ := 0.20)
variable (heirloom_percentage : ℚ := 0.10)
variable (junk_percentage : ℚ := 0.70)
variable (useful_items : ℕ := 8)

theorem junk_items_count (huseful : useful_percentage * total_items = useful_items) : 
  junk_percentage * total_items = 28 :=
by
  sorry

end junk_items_count

namespace probability_of_second_ball_white_is_correct

import Mathlib

-- Definitions based on the conditions
def initial_white_balls : ℕ := 8
def initial_black_balls : ℕ := 7
def total_initial_balls : ℕ := initial_white_balls + initial_black_balls
def white_balls_after_first_draw : ℕ := initial_white_balls
def black_balls_after_first_draw : ℕ := initial_black_balls - 1
def total_balls_after_first_draw : ℕ := white_balls_after_first_draw + black_balls_after_first_draw
def probability_second_ball_white : ℚ := white_balls_after_first_draw / total_balls_after_first_draw

-- The proof problem
theorem probability_of_second_ball_white_is_correct :
  probability_second_ball_white = 4 / 7 :=
by
  sorry

end probability_of_second_ball_white_is_correct

namespace move_point_right_3_units

import Mathlib

theorem move_point_right_3_units (x y : ℤ) (hx : x = 2) (hy : y = -1) :
  (x + 3, y) = (5, -1) :=
by
  sorry

end move_point_right_3_units

namespace power_mean_inequality

import Mathlib

variables {a b c : ℝ}
variables {n p q r : ℕ}

theorem power_mean_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 0 < n)
  (hpqr_nonneg : 0 ≤ p ∧ 0 ≤ q ∧ 0 ≤ r)
  (sum_pqr : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p :=
sorry

end power_mean_inequality

namespace painting_time

import Mathlib

variable (a d e : ℕ)

theorem painting_time (h : a * e * d = a * d * e) : (d * x = a^2 * e) := 
by
   sorry

end painting_time

namespace eval_expression

import Mathlib

-- Define the redefined operation
def red_op (a b : ℝ) : ℝ := (a + b)^2

-- Define the target expression to be evaluated
def expr (x y : ℝ) : ℝ := red_op ((x + y)^2) ((x - y)^2)

-- State the theorem
theorem eval_expression (x y : ℝ) : expr x y = 4 * (x^2 + y^2)^2 := by
  sorry

end eval_expression

namespace domain_f_log2_x_to_domain_f_x

import Mathlib

variable {f : ℝ → ℝ}

-- Condition: The domain of y = f(log₂ x) is [1/2, 4]
def domain_f_log2_x : Set ℝ := Set.Icc (1 / 2) 4

-- Proof statement
theorem domain_f_log2_x_to_domain_f_x
  (h : ∀ x, x ∈ domain_f_log2_x → f (Real.log x / Real.log 2) = f x) :
  Set.Icc (-1) 2 = {x : ℝ | ∃ y ∈ domain_f_log2_x, Real.log y / Real.log 2 = x} :=
by
  sorry

end domain_f_log2_x_to_domain_f_x

namespace part1_part2

import Mathlib

section PartOne

variables (x y : ℕ)
def condition1 := x + y = 360
def condition2 := x - y = 110

theorem part1 (h1 : condition1 x y) (h2 : condition2 x y) : x = 235 ∧ y = 125 := by {
  sorry
}

end PartOne

section PartTwo

variables (t W : ℕ)
def tents_capacity (t : ℕ) := 40 * t + 20 * (9 - t)
def food_capacity (t : ℕ) := 10 * t + 20 * (9 - t)
def transportation_cost (t : ℕ) := 4000 * t + 3600 * (9 - t)

theorem part2 
  (htents : tents_capacity t ≥ 235) 
  (hfood : food_capacity t ≥ 125) : 
  W = transportation_cost t → t = 3 ∧ W = 33600 := by {
  sorry
}

end PartTwo

end part1_part2

namespace distinct_convex_polygons_of_four_or_more_sides

import Mathlib

noncomputable def total_subsets (n : Nat) : Nat := 2^n

noncomputable def subsets_with_fewer_than_four_members (n : Nat) : Nat := 
  (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)

noncomputable def valid_subsets (n : Nat) : Nat := 
  total_subsets n - subsets_with_fewer_than_four_members n

theorem distinct_convex_polygons_of_four_or_more_sides (n : Nat) (h : n = 15) : valid_subsets n = 32192 := by
  sorry

end distinct_convex_polygons_of_four_or_more_sides

namespace intersection_A_C_U_B

import Mathlib

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | Real.log x / Real.log 2 > 0}
def C_U_B : Set ℝ := {x | ¬ (Real.log x / Real.log 2 > 0)}

theorem intersection_A_C_U_B :
  A ∩ C_U_B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_A_C_U_B

namespace parabola_equation

import Mathlib

noncomputable def parabola_focus : (ℝ × ℝ) := (5, -2)

noncomputable def parabola_directrix (x y : ℝ) : Prop := 4 * x - 5 * y = 20

theorem parabola_equation (x y : ℝ) :
  (parabola_focus = (5, -2)) →
  (parabola_directrix x y) →
  25 * x^2 + 40 * x * y + 16 * y^2 - 650 * x + 184 * y + 1009 = 0 :=
by
  sorry

end parabola_equation

namespace calculate_fraction_value

import Mathlib

theorem calculate_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 11 / 7 := 
  sorry

end calculate_fraction_value

namespace tangent_polar_equation

import Mathlib

theorem tangent_polar_equation :
  (∀ t : ℝ, ∃ (x y : ℝ), x = √2 * Real.cos t ∧ y = √2 * Real.sin t) →
  ∃ ρ θ : ℝ, (x = 1) ∧ (y = 1) → 
  ρ * Real.cos θ + ρ * Real.sin θ = 2 := 
by
  sorry

end tangent_polar_equation

namespace max_quotient

import Mathlib

-- Define the given conditions
def conditions (a b : ℝ) :=
  100 ≤ a ∧ a ≤ 250 ∧ 700 ≤ b ∧ b ≤ 1400

-- State the theorem for the largest value of the quotient b / a
theorem max_quotient (a b : ℝ) (h : conditions a b) : b / a ≤ 14 :=
by
  sorry

end max_quotient

namespace pool_width

import Mathlib

variable (length : ℝ) (depth : ℝ) (chlorine_per_cubic_foot : ℝ) (chlorine_cost_per_quart : ℝ) (total_spent : ℝ)
variable (w : ℝ)

-- defining the conditions
def pool_conditions := length = 10 ∧ depth = 6 ∧ chlorine_per_cubic_foot = 120 ∧ chlorine_cost_per_quart = 3 ∧ total_spent = 12

-- goal statement
theorem pool_width : pool_conditions length depth chlorine_per_cubic_foot chlorine_cost_per_quart total_spent →
  w = 8 :=
by
  sorry

end pool_width

namespace david_wins_2011th_even

import Mathlib

theorem david_wins_2011th_even :
  ∃ n : ℕ, (∃ k : ℕ, k = 2011 ∧ n = 2 * k) ∧ (∀ a b : ℕ, a < b → a + b < b * a) ∧ (n % 2 = 0) := 
sorry

end david_wins_2011th_even

namespace math_competition_rankings

import Mathlib

noncomputable def rankings (n : ℕ) : ℕ → Prop := sorry

theorem math_competition_rankings :
  (∀ (A B C D E : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    
    -- A's guesses
    (rankings A 1 → rankings B 3 ∧ rankings C 5) →
    -- B's guesses
    (rankings B 2 → rankings E 4 ∧ rankings D 5) →
    -- C's guesses
    (rankings C 3 → rankings A 1 ∧ rankings E 4) →
    -- D's guesses
    (rankings D 4 → rankings C 1 ∧ rankings D 2) →
    -- E's guesses
    (rankings E 5 → rankings A 3 ∧ rankings D 4) →
    -- Condition that each position is guessed correctly by someone
    (∃ i, rankings A i) ∧
    (∃ i, rankings B i) ∧
    (∃ i, rankings C i) ∧
    (∃ i, rankings D i) ∧
    (∃ i, rankings E i) →
    
    -- The actual placing according to derived solution
    rankings A 1 ∧ 
    rankings D 2 ∧ 
    rankings B 3 ∧ 
    rankings E 4 ∧ 
    rankings C 5) :=
sorry

end math_competition_rankings

namespace simplify_expression

import Mathlib

variable (b : ℝ)

theorem simplify_expression (b : ℝ) : 
  (3 * b + 7 - 5 * b) / 3 = (-2 / 3) * b + (7 / 3) :=
by
  sorry

end simplify_expression

namespace man_speed_against_current

import Mathlib

-- Definitions for the problem conditions
def man_speed_with_current : ℝ := 21
def current_speed : ℝ := 4.3

-- Main proof statement
theorem man_speed_against_current : man_speed_with_current - 2 * current_speed = 12.4 :=
by
  sorry

end man_speed_against_current

namespace length_of_GH

import Mathlib

theorem length_of_GH (AB FE CD : ℕ) (side_large side_second side_third side_small : ℕ) 
  (h1 : AB = 11) (h2 : FE = 13) (h3 : CD = 5)
  (h4 : side_large = side_second + AB)
  (h5 : side_second = side_third + CD)
  (h6 : side_third = side_small + FE) :
  GH = 29 :=
by
  -- Proof steps would follow here based on the problem's solution
  -- Using the given conditions and transformations.
  sorry

end length_of_GH

namespace proof_remove_terms_sum_is_one

import Mathlib

noncomputable def remove_terms_sum_is_one : Prop :=
  let initial_sum := (1/2) + (1/4) + (1/6) + (1/8) + (1/10) + (1/12)
  let terms_to_remove := (1/8) + (1/10)
  initial_sum - terms_to_remove = 1

theorem proof_remove_terms_sum_is_one : remove_terms_sum_is_one :=
by
  -- proof will go here but is not required
  sorry

end proof_remove_terms_sum_is_one

namespace Faye_total_pencils

import Mathlib

def pencils_per_row : ℕ := 8
def number_of_rows : ℕ := 4
def total_pencils : ℕ := pencils_per_row * number_of_rows

theorem Faye_total_pencils : total_pencils = 32 := by
  sorry

end Faye_total_pencils

namespace min_value_expression

import Mathlib

noncomputable def log (base : ℝ) (num : ℝ) := Real.log num / Real.log base

theorem min_value_expression (a b : ℝ) (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * log a b + 6 * log b a = 11) : 
  a^3 + (2 / (b - 1)) ≥ 2 * Real.sqrt 2 + 1 :=
by
  sorry

end min_value_expression

namespace weight_lift_equality

import Mathlib

-- Definitions based on conditions
def total_weight_25_pounds_lifted_times := 750
def total_weight_20_pounds_lifted_per_time (n : ℝ) := 60 * n

-- Statement of the proof problem
theorem weight_lift_equality : ∃ n, total_weight_20_pounds_lifted_per_time n = total_weight_25_pounds_lifted_times :=
  sorry

end weight_lift_equality

namespace additional_track_length_needed

import Mathlib

theorem additional_track_length_needed
  (vertical_rise : ℝ) (initial_grade final_grade : ℝ) (initial_horizontal_length final_horizontal_length : ℝ) : 
  vertical_rise = 400 →
  initial_grade = 0.04 →
  final_grade = 0.03 →
  initial_horizontal_length = (vertical_rise / initial_grade) →
  final_horizontal_length = (vertical_rise / final_grade) →
  final_horizontal_length - initial_horizontal_length = 3333 :=
by
  intros h_vertical_rise h_initial_grade h_final_grade h_initial_horizontal_length h_final_horizontal_length
  sorry

end additional_track_length_needed

namespace option_C_correct

import Mathlib

theorem option_C_correct (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := 
by 
  sorry

end option_C_correct

namespace prove_m_value

import Mathlib

theorem prove_m_value (m : ℕ) : 8^4 = 4^m → m = 6 := by
  sorry

end prove_m_value

namespace horizontal_shift_equivalence

import Mathlib

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (x - Real.pi / 6)
noncomputable def resulting_function (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

theorem horizontal_shift_equivalence :
  ∀ x : ℝ, resulting_function x = original_function (x + Real.pi / 3) :=
by sorry

end horizontal_shift_equivalence

namespace find_numbers

import Mathlib

variables {x y : ℤ}

theorem find_numbers (x y : ℤ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 185) (h3 : (x - y)^2 = 121) :
  (x = 13 ∧ y = 2) ∨ (x = -5 ∧ y = -16) :=
sorry

end find_numbers

namespace two_b_squared_eq_a_squared_plus_c_squared

import Mathlib

theorem two_b_squared_eq_a_squared_plus_c_squared (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 2 / (c + a)) : 
  2 * b^2 = a^2 + c^2 := 
sorry

end two_b_squared_eq_a_squared_plus_c_squared

namespace negation_of_proposition

import Mathlib

theorem negation_of_proposition (x : ℝ) (h : 2 * x + 1 ≤ 0) : ¬ (2 * x + 1 ≤ 0) ↔ 2 * x + 1 > 0 := 
by
  sorry

end negation_of_proposition

namespace smallest_y_value

import Mathlib

theorem smallest_y_value (y : ℝ) : 3 * y ^ 2 + 33 * y - 105 = y * (y + 16) → y = -21 / 2 ∨ y = 5 := sorry

end smallest_y_value

namespace ScarlettsDishCost

import Mathlib

theorem ScarlettsDishCost (L P : ℝ) (tip_rate tip_amount : ℝ) (x : ℝ) 
  (hL : L = 10) (hP : P = 17) (htip_rate : tip_rate = 0.10) (htip_amount : tip_amount = 4) 
  (h : tip_rate * (L + P + x) = tip_amount) : x = 13 :=
by
  sorry

end ScarlettsDishCost

namespace not_divisible_by_1000_pow_m_minus_1

import Mathlib

theorem not_divisible_by_1000_pow_m_minus_1 (m : ℕ) : ¬ (1000^m - 1 ∣ 1998^m - 1) :=
sorry

end not_divisible_by_1000_pow_m_minus_1

namespace sum_ge_3_implies_one_ge_2

import Mathlib

theorem sum_ge_3_implies_one_ge_2 (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by
  sorry

end sum_ge_3_implies_one_ge_2

namespace evaluate_g_at_neg1

import Mathlib

def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem evaluate_g_at_neg1 : g (-1) = -5 := by
  sorry

end evaluate_g_at_neg1

namespace paperback_copies_sold

import Mathlib

theorem paperback_copies_sold
  (H P : ℕ)
  (h1 : H = 36000)
  (h2 : H + P = 440000) :
  P = 404000 :=
by
  rw [h1] at h2
  sorry

end paperback_copies_sold

namespace find_the_number

import Mathlib

theorem find_the_number (x : ℝ) : (3 * x - 1 = 2 * x^2) ∧ (2 * x = (3 * x - 1) / x) → x = 1 := 
by sorry

end find_the_number

namespace power_sum_eq

import Mathlib

theorem power_sum_eq : (-2)^2011 + (-2)^2012 = 2^2011 := by
  sorry

end power_sum_eq

namespace power_multiplication

import Mathlib

variable (x y m n : ℝ)

-- Establishing our initial conditions
axiom h1 : 10^x = m
axiom h2 : 10^y = n

theorem power_multiplication : 10^(2*x + 3*y) = m^2 * n^3 :=
by
  sorry

end power_multiplication

namespace sum_possible_values

import Mathlib

theorem sum_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 4) :
  (x - 2) * (y - 2) = 4 ∨ (x - 2) * (y - 2) = 0 → (4 + 0 = 4) :=
by
  sorry

end sum_possible_values

namespace coin_flip_probability

import Mathlib

theorem coin_flip_probability : 
  ∀ (prob_tails : ℚ) (seq : List (Bool × ℚ)),
    prob_tails = 1/2 →
    seq = [(true, 1/2), (true, 1/2), (false, 1/2), (false, 1/2)] →
    (seq.map Prod.snd).prod = 0.0625 :=
by 
  intros prob_tails seq htails hseq 
  sorry

end coin_flip_probability

namespace mary_age_proof

import Mathlib

theorem mary_age_proof (suzy_age_now : ℕ) (H1 : suzy_age_now = 20) (H2 : ∀ (years : ℕ), years = 4 → (suzy_age_now + years) = 2 * (mary_age + years)) : mary_age = 8 :=
by
  sorry

end mary_age_proof

namespace population_ratio

import Mathlib

-- Definitions
def population_z (Z : ℕ) : ℕ := Z
def population_y (Z : ℕ) : ℕ := 2 * population_z Z
def population_x (Z : ℕ) : ℕ := 6 * population_y Z

-- Theorem stating the ratio
theorem population_ratio (Z : ℕ) : (population_x Z) / (population_z Z) = 12 :=
  by 
  unfold population_x population_y population_z
  sorry

end population_ratio

namespace rectangle_area

import Mathlib

theorem rectangle_area (length diagonal : ℝ) (h_length : length = 16) (h_diagonal : diagonal = 20) : 
  ∃ width : ℝ, (length * width = 192) :=
by 
  sorry

end rectangle_area

namespace 

import Mathlib

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

def is_valid_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem isosceles_triangle_perimeter {a b : ℕ} (h₁ : is_isosceles_triangle a b b) (h₂ : is_valid_triangle a b b) : a + b + b = 15 :=
  sorry

end 

namespace maximum_profit

import Mathlib

def radioactive_marble_problem : ℕ :=
    let total_marbles := 100
    let radioactive_marbles := 1
    let non_radioactive_profit := 1
    let measurement_cost := 1
    let max_profit := 92 
    max_profit

theorem maximum_profit 
    (total_marbles : ℕ := 100) 
    (radioactive_marbles : ℕ := 1) 
    (non_radioactive_profit : ℕ := 1) 
    (measurement_cost : ℕ := 1) :
    radioactive_marble_problem = 92 :=
by sorry

end maximum_profit

namespace min_S_min_S_values_range_of_c

import Mathlib

-- Part 1
theorem min_S (a b c : ℝ) (h : a + b + c = 1) : 
  2 * a^2 + 3 * b^2 + c^2 ≥ (6 / 11) :=
sorry

-- Part 1, finding exact values of a, b, c where minimum is reached
theorem min_S_values (a b c : ℝ) (h : a + b + c = 1) :
  2 * a^2 + 3 * b^2 + c^2 = (6 / 11) ↔ a = (3 / 11) ∧ b = (2 / 11) ∧ c = (6 / 11) :=
sorry
  
-- Part 2
theorem range_of_c (a b c : ℝ) (h1 : 2 * a^2 + 3 * b^2 + c^2 = 1) : 
  (1 / 11) ≤ c ∧ c ≤ 1 :=
sorry

end min_S_min_S_values_range_of_c

namespace machine_A_sprockets_per_hour

import Mathlib

theorem machine_A_sprockets_per_hour :
  ∀ (A T : ℝ),
    (T > 0 ∧
    (∀ P Q, P = 1.1 * A ∧ Q = 330 / P ∧ Q = 330 / A + 10) →
      A = 3) := 
by
  intro A T
  intro h
  sorry

end machine_A_sprockets_per_hour

namespace solve_for_x

import Mathlib

theorem solve_for_x (x : ℝ) (h : (5 - 3 * x)^5 = -1) : x = 2 := by
sorry

end solve_for_x

namespace g_is_odd

import Mathlib

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) - (1 / 2)

theorem g_is_odd (x : ℝ) : g (-x) = -g x :=
by sorry

end g_is_odd