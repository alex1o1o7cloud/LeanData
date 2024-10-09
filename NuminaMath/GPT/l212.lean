import Mathlib

namespace algebraic_expression_evaluation_l212_21255

theorem algebraic_expression_evaluation (a b : ℝ) (h₁ : a ≠ b) 
  (h₂ : a^2 - 8 * a + 5 = 0) (h₃ : b^2 - 8 * b + 5 = 0) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 :=
by
  sorry

end algebraic_expression_evaluation_l212_21255


namespace inequality_solution_l212_21273

theorem inequality_solution
  : {x : ℝ | (x^2 / (x + 2)^2) ≥ 0} = {x : ℝ | x ≠ -2} :=
by
  sorry

end inequality_solution_l212_21273


namespace union_is_correct_l212_21222

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}
def union_set : Set ℤ := {-1, 0, 1, 2}

theorem union_is_correct : M ∪ N = union_set :=
  by sorry

end union_is_correct_l212_21222


namespace op_4_neg3_eq_neg28_l212_21232

def op (x y : Int) : Int := x * (y + 2) + 2 * x * y

theorem op_4_neg3_eq_neg28 : op 4 (-3) = -28 := by
  sorry

end op_4_neg3_eq_neg28_l212_21232


namespace cost_of_filling_all_pots_l212_21250

def cost_palm_fern : ℝ := 15.00
def cost_creeping_jenny_per_plant : ℝ := 4.00
def num_creeping_jennies : ℝ := 4
def cost_geranium_per_plant : ℝ := 3.50
def num_geraniums : ℝ := 4
def cost_elephant_ear_per_plant : ℝ := 7.00
def num_elephant_ears : ℝ := 2
def cost_purple_fountain_grass_per_plant : ℝ := 6.00
def num_purple_fountain_grasses : ℝ := 3
def num_pots : ℝ := 4

def total_cost_per_pot : ℝ := 
  cost_palm_fern +
  (num_creeping_jennies * cost_creeping_jenny_per_plant) +
  (num_geraniums * cost_geranium_per_plant) +
  (num_elephant_ears * cost_elephant_ear_per_plant) +
  (num_purple_fountain_grasses * cost_purple_fountain_grass_per_plant)

def total_cost : ℝ := total_cost_per_pot * num_pots

theorem cost_of_filling_all_pots : total_cost = 308.00 := by
  sorry

end cost_of_filling_all_pots_l212_21250


namespace find_pairs_l212_21259

theorem find_pairs (x y p : ℕ)
  (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h3 : x ≤ y) (h4 : Prime p) :
  (x = 3 ∧ y = 5 ∧ p = 7) ∨ (x = 1 ∧ ∃ q, Prime q ∧ y = q + 1 ∧ p = q ∧ q ≠ 7) ↔
  (x + y) * (x * y - 1) / (x * y + 1) = p := 
sorry

end find_pairs_l212_21259


namespace ratio_and_tangent_l212_21290

-- Definitions for the problem
def acute_triangle (A B C : Point) : Prop := 
  -- acute angles condition
  sorry

def is_diameter (A B C D : Point) : Prop := 
  -- D is midpoint of BC condition
  sorry

def divide_in_half (A B C : Point) (D : Point) : Prop := 
  -- D divides BC in half condition
  sorry

def divide_in_ratio (A B C : Point) (D : Point) (ratio : ℚ) : Prop := 
  -- D divides AC in the given ratio condition
  sorry

def tan (angle : ℝ) : ℝ := 
  -- Tangent function
  sorry

def angle (A B C : Point) : ℝ := 
  -- Angle at B of triangle ABC
  sorry

-- The statement of the problem in Lean
theorem ratio_and_tangent (A B C D : Point) :
  acute_triangle A B C →
  is_diameter A B C D →
  divide_in_half A B C D →
  (divide_in_ratio A B C D (1 / 3) ↔ tan (angle A B C) = 2 * tan (angle A C B)) :=
by sorry

end ratio_and_tangent_l212_21290


namespace total_cars_for_sale_l212_21291

-- Define the conditions given in the problem
def salespeople : Nat := 10
def cars_per_salesperson_per_month : Nat := 10
def months : Nat := 5

-- Statement to prove the total number of cars for sale
theorem total_cars_for_sale : (salespeople * cars_per_salesperson_per_month) * months = 500 := by
  -- Proof goes here
  sorry

end total_cars_for_sale_l212_21291


namespace least_subtracted_correct_second_num_correct_l212_21297

-- Define the given numbers
def given_num : ℕ := 1398
def remainder : ℕ := 5
def num1 : ℕ := 7
def num2 : ℕ := 9
def num3 : ℕ := 11

-- Least number to subtract to satisfy the condition
def least_subtracted : ℕ := 22

-- Second number in the sequence
def second_num : ℕ := 2069

-- Define the hypotheses and statements to be proved
theorem least_subtracted_correct : given_num - least_subtracted ≡ remainder [MOD num1]
∧ given_num - least_subtracted ≡ remainder [MOD num2]
∧ given_num - least_subtracted ≡ remainder [MOD num3] := sorry

theorem second_num_correct : second_num ≡ remainder [MOD num1 * num2 * num3] := sorry

end least_subtracted_correct_second_num_correct_l212_21297


namespace P_has_real_root_l212_21243

def P : ℝ → ℝ := sorry
variables (a1 a2 a3 b1 b2 b3 : ℝ)

axiom a1_nonzero : a1 ≠ 0
axiom a2_nonzero : a2 ≠ 0
axiom a3_nonzero : a3 ≠ 0

axiom functional_eq (x : ℝ) :
  P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)

theorem P_has_real_root :
  ∃ x : ℝ, P x = 0 :=
sorry

end P_has_real_root_l212_21243


namespace monthly_increase_per_ticket_l212_21240

variable (x : ℝ)

theorem monthly_increase_per_ticket
    (initial_premium : ℝ := 50)
    (percent_increase_per_accident : ℝ := 0.10)
    (tickets : ℕ := 3)
    (final_premium : ℝ := 70) :
    initial_premium * (1 + percent_increase_per_accident) + tickets * x = final_premium → x = 5 :=
by
  intro h
  sorry

end monthly_increase_per_ticket_l212_21240


namespace four_four_four_digits_eight_eight_eight_digits_l212_21277

theorem four_four_four_digits_eight_eight_eight_digits (n : ℕ) :
  (4 * (10 ^ (n + 1) - 1) * (10 ^ n) + 8 * (10^n - 1) + 9) = 
  (6 * 10^n + 7) * (6 * 10^n + 7) :=
sorry

end four_four_four_digits_eight_eight_eight_digits_l212_21277


namespace sum_of_primes_is_prime_l212_21205

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

theorem sum_of_primes_is_prime (P Q : ℕ) :
  is_prime P → is_prime Q → is_prime (P - Q) → is_prime (P + Q) →
  ∃ n : ℕ, n = P + Q + (P - Q) + (P + Q) ∧ is_prime n := by
  sorry

end sum_of_primes_is_prime_l212_21205


namespace whole_number_N_l212_21285

theorem whole_number_N (N : ℤ) : (9 < N / 4 ∧ N / 4 < 10) ↔ (N = 37 ∨ N = 38 ∨ N = 39) := 
by sorry

end whole_number_N_l212_21285


namespace max_distinct_numbers_example_l212_21225

def max_distinct_numbers (a b c d e : ℕ) : ℕ := sorry

theorem max_distinct_numbers_example
  (A B : ℕ) :
  max_distinct_numbers 100 200 400 A B = 64 := sorry

end max_distinct_numbers_example_l212_21225


namespace determine_x_l212_21215

theorem determine_x (p q : ℝ) (hpq : p ≠ q) : 
  ∃ (c d : ℝ), (x = c*p + d*q) ∧ c = 2 ∧ d = -2 :=
by 
  sorry

end determine_x_l212_21215


namespace decreasing_interval_of_logarithm_derived_function_l212_21221

theorem decreasing_interval_of_logarithm_derived_function :
  ∀ (x : ℝ), 1 < x → ∃ (f : ℝ → ℝ), (f x = x / (x - 1)) ∧ (∀ (h : x ≠ 1), deriv f x < 0) :=
by
  sorry

end decreasing_interval_of_logarithm_derived_function_l212_21221


namespace solution_set_f_pos_l212_21219

noncomputable def f : ℝ → ℝ := sorry -- Definition of the function f(x)

-- Conditions
axiom h1 : ∀ x, f (-x) = -f x     -- f(x) is odd
axiom h2 : f 2 = 0                -- f(2) = 0
axiom h3 : ∀ x > 0, 2 * f x + x * (deriv f x) > 0 -- 2f(x) + xf'(x) > 0 for x > 0

-- Theorem to prove
theorem solution_set_f_pos : { x : ℝ | f x > 0 } = { x : ℝ | x > 2 ∨ (-2 < x ∧ x < 0) } :=
sorry

end solution_set_f_pos_l212_21219


namespace smallest_four_digit_divisible_by_6_l212_21292

-- Define the smallest four-digit number
def smallest_four_digit_number := 1000

-- Define divisibility conditions
def divisible_by_2 (n : Nat) := n % 2 = 0
def divisible_by_3 (n : Nat) := n % 3 = 0
def divisible_by_6 (n : Nat) := divisible_by_2 n ∧ divisible_by_3 n

-- Prove that the smallest four-digit number divisible by 6 is 1002
theorem smallest_four_digit_divisible_by_6 : ∃ n : Nat, n ≥ smallest_four_digit_number ∧ divisible_by_6 n ∧ ∀ m : Nat, m ≥ smallest_four_digit_number ∧ divisible_by_6 m → n ≤ m :=
  sorry

end smallest_four_digit_divisible_by_6_l212_21292


namespace total_logs_in_both_stacks_l212_21238

-- Define the number of logs in the first stack
def first_stack_logs : Nat :=
  let bottom_row := 15
  let top_row := 4
  let number_of_terms := bottom_row - top_row + 1
  let average_logs := (bottom_row + top_row) / 2
  average_logs * number_of_terms

-- Define the number of logs in the second stack
def second_stack_logs : Nat :=
  let bottom_row := 5
  let top_row := 10
  let number_of_terms := top_row - bottom_row + 1
  let average_logs := (bottom_row + top_row) / 2
  average_logs * number_of_terms

-- Prove the total number of logs in both stacks
theorem total_logs_in_both_stacks : first_stack_logs + second_stack_logs = 159 := by
  sorry

end total_logs_in_both_stacks_l212_21238


namespace forgot_to_take_capsules_l212_21262

theorem forgot_to_take_capsules (total_days : ℕ) (days_taken : ℕ) 
  (h1 : total_days = 31) 
  (h2 : days_taken = 29) : 
  total_days - days_taken = 2 := 
by 
  sorry

end forgot_to_take_capsules_l212_21262


namespace relatively_prime_pair_count_l212_21272

theorem relatively_prime_pair_count :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m + n = 190 ∧ Nat.gcd m n = 1) →
  (∃! k : ℕ, k = 26) :=
by
  sorry

end relatively_prime_pair_count_l212_21272


namespace circle_through_three_points_l212_21200

open Real

structure Point where
  x : ℝ
  y : ℝ

def circle_equation (D E F : ℝ) (P : Point) : Prop :=
  P.x^2 + P.y^2 + D * P.x + E * P.y + F = 0

theorem circle_through_three_points :
  ∃ (D E F : ℝ), 
    (circle_equation D E F ⟨1, 12⟩) ∧ 
    (circle_equation D E F ⟨7, 10⟩) ∧ 
    (circle_equation D E F ⟨-9, 2⟩) ∧
    (D = -2) ∧ (E = -4) ∧ (F = -95) :=
by
  sorry

end circle_through_three_points_l212_21200


namespace probability_no_adjacent_birch_l212_21218

theorem probability_no_adjacent_birch (m n : ℕ):
  let maple_trees := 5
  let oak_trees := 4
  let birch_trees := 6
  let total_trees := maple_trees + oak_trees + birch_trees
  (∀ (prob : ℚ), prob = (2 : ℚ) / 45) → (m + n = 47) := by
  sorry

end probability_no_adjacent_birch_l212_21218


namespace total_games_l212_21284

-- Define the conditions
def games_this_year : ℕ := 4
def games_last_year : ℕ := 9

-- Define the proposition that we want to prove
theorem total_games : games_this_year + games_last_year = 13 := by
  sorry

end total_games_l212_21284


namespace y_at_x_eq_120_l212_21201

@[simp] def custom_op (a b : ℕ) : ℕ := List.prod (List.map (λ i => a + i) (List.range b))

theorem y_at_x_eq_120 {x y : ℕ}
  (h1 : custom_op x (custom_op y 2) = 420)
  (h2 : x = 4)
  (h3 : y = 2) :
  custom_op y x = 120 := by
  sorry

end y_at_x_eq_120_l212_21201


namespace speed_ratio_correct_l212_21206

noncomputable def boat_speed_still_water := 12 -- Boat's speed in still water (in mph)
noncomputable def current_speed := 4 -- Current speed of the river (in mph)

-- Calculate the downstream speed
noncomputable def downstream_speed := boat_speed_still_water + current_speed

-- Calculate the upstream speed
noncomputable def upstream_speed := boat_speed_still_water - current_speed

-- Assume a distance for the trip (1 mile each up and down)
noncomputable def distance := 1

-- Calculate time for downstream
noncomputable def time_downstream := distance / downstream_speed

-- Calculate time for upstream
noncomputable def time_upstream := distance / upstream_speed

-- Calculate total time for the round trip
noncomputable def total_time := time_downstream + time_upstream

-- Calculate total distance for the round trip
noncomputable def total_distance := 2 * distance

-- Calculate the average speed for the round trip
noncomputable def avg_speed_trip := total_distance / total_time

-- Calculate the ratio of average speed to speed in still water
noncomputable def speed_ratio := avg_speed_trip / boat_speed_still_water

theorem speed_ratio_correct : speed_ratio = 8/9 := by
  sorry

end speed_ratio_correct_l212_21206


namespace fraction_of_salary_on_rent_l212_21220

theorem fraction_of_salary_on_rent
  (S : ℝ) (food_fraction : ℝ) (clothes_fraction : ℝ) (remaining_amount : ℝ) (approx_salary : ℝ)
  (food_fraction_eq : food_fraction = 1 / 5)
  (clothes_fraction_eq : clothes_fraction = 3 / 5)
  (remaining_amount_eq : remaining_amount = 19000)
  (approx_salary_eq : approx_salary = 190000) :
  ∃ (H : ℝ), H = 1 / 10 :=
by
  sorry

end fraction_of_salary_on_rent_l212_21220


namespace geometric_sequence_ratio_l212_21229

variable {α : Type*} [Field α]

def geometric_sequence (a_1 q : α) (n : ℕ) : α :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_ratio (a1 q a4 a14 a5 a13 : α)
  (h_seq : ∀ n, geometric_sequence a1 q (n + 1) = a_5) 
  (h0 : geometric_sequence a1 q 5 * geometric_sequence a1 q 13 = 6) 
  (h1 : geometric_sequence a1 q 4 + geometric_sequence a1 q 14 = 5) :
  (∃ (k : α), k = 2 / 3 ∨ k = 3 / 2) → 
  geometric_sequence a1 q 80 / geometric_sequence a1 q 90 = k :=
by
  sorry

end geometric_sequence_ratio_l212_21229


namespace fruit_salad_cherries_l212_21227

theorem fruit_salad_cherries (b r g c : ℕ) 
(h1 : b + r + g + c = 360)
(h2 : r = 3 * b) 
(h3 : g = 4 * c)
(h4 : c = 5 * r) :
c = 68 := 
sorry

end fruit_salad_cherries_l212_21227


namespace solution_set_of_inequality_l212_21235

variable (f : ℝ → ℝ)

theorem solution_set_of_inequality :
  (∀ x, f (x) = f (-x)) →               -- f(x) is even
  (∀ x y, 0 < x → x < y → f y ≤ f x) →   -- f(x) is monotonically decreasing on (0, +∞)
  f 2 = 0 →                              -- f(2) = 0
  {x : ℝ | (f x + f (-x)) / (3 * x) < 0} = 
    {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 2 < x} :=
by sorry

end solution_set_of_inequality_l212_21235


namespace find_number_l212_21226

theorem find_number (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 := 
sorry

end find_number_l212_21226


namespace map_area_ratio_l212_21258

theorem map_area_ratio (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  ¬ ((l * w) / ((500 * l) * (500 * w)) = 1 / 500) :=
by
  -- The proof will involve calculations showing the true ratio is 1/250000
  sorry

end map_area_ratio_l212_21258


namespace abs_frac_sqrt_l212_21214

theorem abs_frac_sqrt (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a^2 + b^2 = 9 * a * b) : 
  abs ((a + b) / (a - b)) = Real.sqrt (11 / 7) :=
by
  sorry

end abs_frac_sqrt_l212_21214


namespace scientific_notation_of_3300000000_l212_21209

theorem scientific_notation_of_3300000000 :
  3300000000 = 3.3 * 10^9 :=
sorry

end scientific_notation_of_3300000000_l212_21209


namespace balls_in_boxes_l212_21213

theorem balls_in_boxes : (3^4 = 81) :=
by
  sorry

end balls_in_boxes_l212_21213


namespace find_m_value_l212_21253

theorem find_m_value :
  62519 * 9999 = 625127481 :=
  by sorry

end find_m_value_l212_21253


namespace dan_violet_marbles_l212_21261

def InitMarbles : ℕ := 128
def MarblesGivenMary : ℕ := 24
def MarblesGivenPeter : ℕ := 16
def MarblesReceived : ℕ := 10

def FinalMarbles : ℕ := InitMarbles - MarblesGivenMary - MarblesGivenPeter + MarblesReceived

theorem dan_violet_marbles : FinalMarbles = 98 := 
by 
  sorry

end dan_violet_marbles_l212_21261


namespace Connie_savings_l212_21266

theorem Connie_savings (cost_of_watch : ℕ) (extra_needed : ℕ) (saved_amount : ℕ) : 
  cost_of_watch = 55 → 
  extra_needed = 16 → 
  saved_amount = cost_of_watch - extra_needed → 
  saved_amount = 39 := 
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end Connie_savings_l212_21266


namespace k_sq_geq_25_over_4_l212_21256

theorem k_sq_geq_25_over_4
  (a1 a2 a3 a4 a5 k : ℝ)
  (h1 : |a1 - a2| ≥ 1 ∧ |a1 - a3| ≥ 1 ∧ |a1 - a4| ≥ 1 ∧ |a1 - a5| ≥ 1 ∧
       |a2 - a3| ≥ 1 ∧ |a2 - a4| ≥ 1 ∧ |a2 - a5| ≥ 1 ∧
       |a3 - a4| ≥ 1 ∧ |a3 - a5| ≥ 1 ∧
       |a4 - a5| ≥ 1)
  (h2 : a1 + a2 + a3 + a4 + a5 = 2 * k)
  (h3 : a1^2 + a2^2 + a3^2 + a4^2 + a5^2 = 2 * k^2) :
  k^2 ≥ 25 / 4 :=
sorry

end k_sq_geq_25_over_4_l212_21256


namespace xy_eq_one_l212_21248

theorem xy_eq_one (x y : ℝ) (h : x + y = (1 / x) + (1 / y) ∧ x + y ≠ 0) : x * y = 1 := by
  sorry

end xy_eq_one_l212_21248


namespace lipstick_cost_is_correct_l212_21264

noncomputable def cost_of_lipstick (palette_cost : ℝ) (num_palettes : ℝ) (hair_color_cost : ℝ) (num_hair_colors : ℝ) (total_paid : ℝ) (num_lipsticks : ℝ) : ℝ :=
  let total_palette_cost := num_palettes * palette_cost
  let total_hair_color_cost := num_hair_colors * hair_color_cost
  let remaining_amount := total_paid - (total_palette_cost + total_hair_color_cost)
  remaining_amount / num_lipsticks

theorem lipstick_cost_is_correct :
  cost_of_lipstick 15 3 4 3 67 4 = 2.5 :=
by
  sorry

end lipstick_cost_is_correct_l212_21264


namespace cuboid_properties_l212_21234

-- Given definitions from conditions
variables (l w h : ℝ)
variables (h_edge_length : 4 * (l + w + h) = 72)
variables (h_ratio : l / w = 3 / 2 ∧ w / h = 2 / 1)

-- Define the surface area and volume based on the given conditions
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)
def volume (l w h : ℝ) : ℝ := l * w * h

-- Theorem statement
theorem cuboid_properties :
  surface_area l w h = 198 ∧ volume l w h = 162 :=
by
  -- Code to provide the proof goes here
  sorry

end cuboid_properties_l212_21234


namespace adoption_event_l212_21298

theorem adoption_event (c : ℕ) 
  (h1 : ∀ d : ℕ, d = 8) 
  (h2 : ∀ fees_dog : ℕ, fees_dog = 15) 
  (h3 : ∀ fees_cat : ℕ, fees_cat = 13)
  (h4 : ∀ donation : ℕ, donation = 53)
  (h5 : fees_dog * 8 + fees_cat * c = 159) :
  c = 3 :=
by 
  sorry

end adoption_event_l212_21298


namespace xy_computation_l212_21211

theorem xy_computation (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : 
  x * y = 21 := by
  sorry

end xy_computation_l212_21211


namespace sum_product_of_pairs_l212_21203

theorem sum_product_of_pairs (x y z : ℝ) 
  (h1 : x + y + z = 20) 
  (h2 : x^2 + y^2 + z^2 = 200) :
  x * y + x * z + y * z = 100 := 
by
  sorry

end sum_product_of_pairs_l212_21203


namespace simplify_expression_l212_21224

def a : ℕ := 1050
def p : ℕ := 2101
def q : ℕ := 1050 * 1051

theorem simplify_expression : 
  (1051 / 1050) - (1050 / 1051) = (p : ℚ) / (q : ℚ) ∧ Nat.gcd p a = 1 ∧ Nat.gcd p (a + 1) = 1 :=
by 
  sorry

end simplify_expression_l212_21224


namespace initial_sum_l212_21239

theorem initial_sum (P : ℝ) (compound_interest : ℝ) (r1 r2 r3 r4 r5 : ℝ) 
  (h1 : r1 = 0.06) (h2 : r2 = 0.08) (h3 : r3 = 0.07) (h4 : r4 = 0.09) (h5 : r5 = 0.10)
  (interest_sum : compound_interest = 4016.25) :
  P = 4016.25 / ((1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5) - 1) :=
by
  sorry

end initial_sum_l212_21239


namespace evaluate_expression_l212_21244

def operation (x y : ℚ) : ℚ := x^2 / y

theorem evaluate_expression : 
  (operation (operation 3 4) 2) - (operation 3 (operation 4 2)) = 45 / 32 :=
by
  sorry

end evaluate_expression_l212_21244


namespace distribution_problem_distribution_problem_variable_distribution_problem_equal_l212_21246

def books_distribution_fixed (n : ℕ) (a b c : ℕ) : ℕ :=
  if h : a + b + c = n then n.factorial / (a.factorial * b.factorial * c.factorial) else 0

theorem distribution_problem (n a b c : ℕ) (h : a + b + c = n) : 
  books_distribution_fixed n a b c = 1260 :=
sorry

def books_distribution_variable (n : ℕ) (a b c : ℕ) : ℕ :=
  if h : a + b + c = n then (n.factorial / (a.factorial * b.factorial * c.factorial)) * 6 else 0

theorem distribution_problem_variable (n a b c : ℕ) (h : a + b + c = n) : 
  books_distribution_variable n a b c = 7560 :=
sorry

def books_distribution_equal (n : ℕ) (k : ℕ) : ℕ :=
  if h : 3 * k = n then n.factorial / (k.factorial * k.factorial * k.factorial) else 0

theorem distribution_problem_equal (n k : ℕ) (h : 3 * k = n) : 
  books_distribution_equal n k = 1680 :=
sorry

end distribution_problem_distribution_problem_variable_distribution_problem_equal_l212_21246


namespace gcd_three_digit_numbers_l212_21228

theorem gcd_three_digit_numbers (a b c : ℕ) (h1 : b = a + 1) (h2 : c = a + 2) :
  ∃ k, (∀ n, n = 100 * a + 10 * b + c + 100 * c + 10 * b + a → n = 212 * k) :=
by
  sorry

end gcd_three_digit_numbers_l212_21228


namespace quadratic_real_roots_iff_range_of_a_l212_21296

theorem quadratic_real_roots_iff_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a = 0) ↔ a ≤ 1 :=
by
  sorry

end quadratic_real_roots_iff_range_of_a_l212_21296


namespace line_slope_intercept_l212_21230

theorem line_slope_intercept :
  ∃ k b, (∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 → y = k * x + b) ∧ k = 2/3 ∧ b = 2 :=
by
  sorry

end line_slope_intercept_l212_21230


namespace each_person_pays_l212_21223

def numPeople : ℕ := 6
def rentalDays : ℕ := 4
def weekdayRate : ℕ := 420
def weekendRate : ℕ := 540
def numWeekdays : ℕ := 2
def numWeekends : ℕ := 2

theorem each_person_pays : 
  (numWeekdays * weekdayRate + numWeekends * weekendRate) / numPeople = 320 :=
by
  sorry

end each_person_pays_l212_21223


namespace black_marbles_count_l212_21299

theorem black_marbles_count :
  ∀ (white_marbles total_marbles : ℕ), 
  white_marbles = 19 → total_marbles = 37 → total_marbles - white_marbles = 18 :=
by
  intros white_marbles total_marbles h_white h_total
  sorry

end black_marbles_count_l212_21299


namespace radius_ratio_of_circumscribed_truncated_cone_l212_21212

theorem radius_ratio_of_circumscribed_truncated_cone 
  (R r ρ : ℝ) 
  (h : ℝ) 
  (Vcs Vg : ℝ) 
  (h_eq : h = 2 * ρ)
  (Vcs_eq : Vcs = (π / 3) * h * (R^2 + r^2 + R * r))
  (Vg_eq : Vg = (4 * π * (ρ^3)) / 3)
  (Vcs_Vg_eq : Vcs = 2 * Vg) :
  (R / r) = (3 + Real.sqrt 5) / 2 := 
sorry

end radius_ratio_of_circumscribed_truncated_cone_l212_21212


namespace repeating_sequence_length_1_over_221_l212_21275

theorem repeating_sequence_length_1_over_221 : ∃ n : ℕ, (10 ^ n ≡ 1 [MOD 221]) ∧ (∀ m : ℕ, (10 ^ m ≡ 1 [MOD 221]) → (n ≤ m)) ∧ n = 48 :=
by
  sorry

end repeating_sequence_length_1_over_221_l212_21275


namespace difference_between_numbers_l212_21293

theorem difference_between_numbers :
  ∃ X Y : ℕ, 
    100 ≤ X ∧ X < 1000 ∧
    100 ≤ Y ∧ Y < 1000 ∧
    X + Y = 999 ∧
    1000 * X + Y = 6 * (1000 * Y + X) ∧
    (X - Y = 715 ∨ Y - X = 715) :=
by
  sorry

end difference_between_numbers_l212_21293


namespace value_of_m_l212_21242

theorem value_of_m : (∀ x : ℝ, (1 + 2 * x) ^ 3 = 1 + 6 * x + m * x ^ 2 + 8 * x ^ 3 → m = 12) := 
by {
  -- This is where the proof would go
  sorry
}

end value_of_m_l212_21242


namespace vector_sum_l212_21271

-- Define the vectors a and b according to the conditions.
def a : (ℝ × ℝ) := (2, 1)
def b : (ℝ × ℝ) := (-3, 4)

-- Prove that the vector sum a + b is (-1, 5).
theorem vector_sum : (a.1 + b.1, a.2 + b.2) = (-1, 5) :=
by
  -- include the proof later
  sorry

end vector_sum_l212_21271


namespace find_divisor_l212_21276

variable (dividend quotient remainder divisor : ℕ)

theorem find_divisor (h1 : dividend = 52) (h2 : quotient = 16) (h3 : remainder = 4) (h4 : dividend = divisor * quotient + remainder) : 
  divisor = 3 := by
  sorry

end find_divisor_l212_21276


namespace nancy_marks_home_economics_l212_21217

-- Definitions from conditions
def marks_american_lit := 66
def marks_history := 75
def marks_physical_ed := 68
def marks_art := 89
def average_marks := 70
def num_subjects := 5
def total_marks := average_marks * num_subjects
def marks_other_subjects := marks_american_lit + marks_history + marks_physical_ed + marks_art

-- Statement to prove
theorem nancy_marks_home_economics : 
  (total_marks - marks_other_subjects = 52) := by 
  sorry

end nancy_marks_home_economics_l212_21217


namespace find_amplitude_l212_21249

theorem find_amplitude (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : ∀ x, a * Real.cos (b * x - c) ≤ 3) 
  (h5 : ∀ x, abs (a * Real.cos (b * x - c) - a * Real.cos (b * (x + 2 * π / b) - c)) = 0) :
  a = 3 := 
sorry

end find_amplitude_l212_21249


namespace max_reflections_l212_21288

theorem max_reflections (angle_increase : ℕ := 10) (max_angle : ℕ := 90) :
  ∃ n : ℕ, 10 * n ≤ max_angle ∧ ∀ m : ℕ, (10 * (m + 1) > max_angle → m < n) := 
sorry

end max_reflections_l212_21288


namespace range_of_a_if_odd_symmetric_points_l212_21257

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a

theorem range_of_a_if_odd_symmetric_points (a : ℝ): 
  (∃ x₀ : ℝ, x₀ ≠ 0 ∧ f x₀ a = -f (-x₀) a) → (1 < a) :=
by 
  sorry

end range_of_a_if_odd_symmetric_points_l212_21257


namespace gold_copper_alloy_ratio_l212_21251

theorem gold_copper_alloy_ratio {G C A : ℝ} (hC : C = 9) (hA : A = 18) (hG : 9 < G ∧ G < 18) :
  ∃ x : ℝ, 18 = x * G + (1 - x) * 9 :=
by
  sorry

end gold_copper_alloy_ratio_l212_21251


namespace range_of_2a_plus_3b_inequality_between_expressions_l212_21236

-- First proof problem
theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b) (h2 : a + b ≤ 1) (h3 : -1 ≤ a - b) (h4 : a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
sorry

-- Second proof problem
theorem inequality_between_expressions (a b c : ℝ) (h : a^2 + b^2 + c^2 = 6) :
  (1 / (a^2 + 1) + 1 / (b^2 + 2)) > (1 / 2 - 1 / (c^2 + 3)) :=
sorry

end range_of_2a_plus_3b_inequality_between_expressions_l212_21236


namespace negation_of_proposition_l212_21245

theorem negation_of_proposition (a b : ℝ) : 
  ¬ (∀ a b : ℝ, (a = 1 → a + b = 1)) ↔ (∃ a b : ℝ, a = 1 ∧ a + b ≠ 1) :=
by
  sorry

end negation_of_proposition_l212_21245


namespace groceries_spent_l212_21260

/-- Defining parameters from the conditions provided -/
def rent : ℝ := 5000
def milk : ℝ := 1500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 700
def savings_rate : ℝ := 0.10
def savings : ℝ := 1800

/-- Adding an assertion for the total spent on groceries -/
def groceries : ℝ := 4500

theorem groceries_spent (total_salary total_expenses : ℝ) :
  total_salary = savings / savings_rate →
  total_expenses = rent + milk + education + petrol + miscellaneous →
  groceries = total_salary - (total_expenses + savings) :=
by
  intros h_salary h_expenses
  sorry

end groceries_spent_l212_21260


namespace fido_leash_yard_reach_area_product_l212_21207

noncomputable def fido_leash_yard_fraction : ℝ :=
  let a := 2 + Real.sqrt 2
  let b := 8
  a * b

theorem fido_leash_yard_reach_area_product :
  ∃ (a b : ℝ), 
  (fido_leash_yard_fraction = (a * b)) ∧ 
  (1 > a) ∧ -- Regular Octagon computation constraints
  (b = 8) ∧ 
  a = 2 + Real.sqrt 2 :=
sorry

end fido_leash_yard_reach_area_product_l212_21207


namespace coffee_processing_completed_l212_21252

-- Define the initial conditions
def CoffeeBeansProcessed (m n : ℕ) : Prop :=
  let mass: ℝ := 1
  let days_single_machine: ℕ := 5
  let days_both_machines: ℕ := 4
  let half_mass: ℝ := mass / 2
  let total_ground_by_June_10 := (days_single_machine * m + days_both_machines * (m + n)) = half_mass
  total_ground_by_June_10

-- Define the final proof problem
theorem coffee_processing_completed (m n : ℕ) (h: CoffeeBeansProcessed m n) : ∃ d : ℕ, d = 15 := by
  -- Processed in 15 working days
  sorry

end coffee_processing_completed_l212_21252


namespace abs_sum_lt_abs_l212_21254

theorem abs_sum_lt_abs (a b : ℝ) (h : a * b < 0) : |a + b| < |a| + |b| :=
sorry

end abs_sum_lt_abs_l212_21254


namespace polygon_sides_l212_21270

theorem polygon_sides (h : 1440 = (n - 2) * 180) : n = 10 := 
by {
  -- Here, the proof would show the steps to solve the equation h and confirm n = 10
  sorry
}

end polygon_sides_l212_21270


namespace necessary_but_not_sufficient_l212_21295

theorem necessary_but_not_sufficient (x: ℝ) :
  (1 < x ∧ x < 4) → (1 < x ∧ x < 3) := by
sorry

end necessary_but_not_sufficient_l212_21295


namespace percent_of_ducks_among_non_swans_l212_21210

theorem percent_of_ducks_among_non_swans
  (total_birds : ℕ) 
  (percent_ducks percent_swans percent_eagles percent_sparrows : ℕ)
  (h1 : percent_ducks = 40) 
  (h2 : percent_swans = 20) 
  (h3 : percent_eagles = 15) 
  (h4 : percent_sparrows = 25)
  (h_sum : percent_ducks + percent_swans + percent_eagles + percent_sparrows = 100) :
  (percent_ducks * 100) / (100 - percent_swans) = 50 :=
by
  sorry

end percent_of_ducks_among_non_swans_l212_21210


namespace remaining_budget_correct_l212_21231

def cost_item1 := 13
def cost_item2 := 24
def last_year_remaining_budget := 6
def this_year_budget := 50

theorem remaining_budget_correct :
    (last_year_remaining_budget + this_year_budget - (cost_item1 + cost_item2) = 19) :=
by
  -- This is the statement only, with the proof omitted
  sorry

end remaining_budget_correct_l212_21231


namespace find_b_l212_21233

noncomputable def P (x a b c : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + c

theorem find_b (a b c : ℝ) (h1: P 0 a b c = 12)
  (h2: (-c / 2) * 1 = -6)
  (h3: (2 + a + b + c) = -6)
  (h4: a + b + 14 = -6) : b = -56 :=
sorry

end find_b_l212_21233


namespace tulips_for_each_eye_l212_21286

theorem tulips_for_each_eye (R : ℕ) : 2 * R + 18 + 9 * 18 = 196 → R = 8 :=
by
  intro h
  sorry

end tulips_for_each_eye_l212_21286


namespace remainders_sum_l212_21263

theorem remainders_sum (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 20) 
  (h3 : c % 30 = 10) : 
  (a + b + c) % 30 = 15 := 
by
  sorry

end remainders_sum_l212_21263


namespace sum_of_coefficients_proof_l212_21202

-- Problem statement: Define the expressions and prove the sum of the coefficients
def expr1 (c : ℝ) : ℝ := -(3 - c) * (c + 2 * (3 - c))
def expanded_form (c : ℝ) : ℝ := -c^2 + 9 * c - 18
def sum_of_coefficients (p : ℝ) := -1 + 9 - 18

theorem sum_of_coefficients_proof (c : ℝ) : sum_of_coefficients (expr1 c) = -10 := by
  sorry

end sum_of_coefficients_proof_l212_21202


namespace remainder_div_2468135790_101_l212_21294

theorem remainder_div_2468135790_101 : 2468135790 % 101 = 50 :=
by
  sorry

end remainder_div_2468135790_101_l212_21294


namespace proof_calculate_expr_l212_21278

def calculate_expr : Prop :=
  (4 + 4 + 6) / 3 - 2 / 3 = 4

theorem proof_calculate_expr : calculate_expr := 
by 
  sorry

end proof_calculate_expr_l212_21278


namespace solve_for_x_l212_21282

theorem solve_for_x (x y : ℤ) (h1 : x + 2 * y = 10) (h2 : y = 3) : x = 4 := by
  sorry

end solve_for_x_l212_21282


namespace solve_equation_l212_21287

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x^2 - 1 ≠ 0) : (x / (x - 1) = 2 / (x^2 - 1)) → (x = -2) :=
by
  intro h
  sorry

end solve_equation_l212_21287


namespace b_n_expression_l212_21265

-- Define sequence a_n as an arithmetic sequence with given conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + d * (n - 1)

-- Define the conditions for the sequence a_n
def a_conditions (a : ℕ → ℤ) : Prop :=
  a 2 = 8 ∧ a 8 = 26

-- Define the new sequence b_n based on the terms of a_n
def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  a (3^n)

theorem b_n_expression (a : ℕ → ℤ) (n : ℕ)
  (h_arith : is_arithmetic_sequence a)
  (h_conditions : a_conditions a) :
  b a n = 3^(n + 1) + 2 := 
sorry

end b_n_expression_l212_21265


namespace find_other_number_l212_21274

theorem find_other_number (a b lcm hcf : ℕ) (h_lcm : lcm = 2310) (h_hcf : hcf = 61) (h_first_number : a = 210) :
  a * b = lcm * hcf → b = 671 :=
by 
  -- setup
  sorry

end find_other_number_l212_21274


namespace benny_kids_l212_21280

theorem benny_kids (total_money : ℕ) (cost_per_apple : ℕ) (apples_per_kid : ℕ) (total_apples : ℕ) (kids : ℕ) :
  total_money = 360 →
  cost_per_apple = 4 →
  apples_per_kid = 5 →
  total_apples = total_money / cost_per_apple →
  kids = total_apples / apples_per_kid →
  kids = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end benny_kids_l212_21280


namespace discount_percentage_l212_21283

variable (P : ℝ) (r : ℝ) (S : ℝ)

theorem discount_percentage (hP : P = 20) (hr : r = 30 / 100) (hS : S = 13) :
  (P * (1 + r) - S) / (P * (1 + r)) * 100 = 50 := 
sorry

end discount_percentage_l212_21283


namespace eval_expression_l212_21216

theorem eval_expression : 9^9 * 3^3 / 3^30 = 1 / 19683 := by
  sorry

end eval_expression_l212_21216


namespace arithmetic_sequence_sum_l212_21204

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
    (h₁ : ∀ n, a (n + 1) = a n + d)
    (h₂ : a 3 + a 5 + a 7 + a 9 + a 11 = 20) : a 1 + a 13 = 8 := 
by 
  sorry

end arithmetic_sequence_sum_l212_21204


namespace product_xyz_equals_one_l212_21208

theorem product_xyz_equals_one (x y z : ℝ) (h1 : x + (1/y) = 2) (h2 : y + (1/z) = 2) : x * y * z = 1 := 
by
  sorry

end product_xyz_equals_one_l212_21208


namespace count_possible_third_side_lengths_l212_21281

theorem count_possible_third_side_lengths : ∀ (n : ℤ), 2 < n ∧ n < 14 → ∃ s : Finset ℤ, s.card = 11 ∧ ∀ x ∈ s, 2 < x ∧ x < 14 := by
  sorry

end count_possible_third_side_lengths_l212_21281


namespace volume_of_solid_rotation_l212_21237

noncomputable def volume_of_solid := 
  (∫ y in (0:ℝ)..(1:ℝ), (y^(2/3) - y^2)) * Real.pi 

theorem volume_of_solid_rotation :
  volume_of_solid = (4 * Real.pi / 15) :=
by
  sorry

end volume_of_solid_rotation_l212_21237


namespace rachel_baked_brownies_l212_21247

theorem rachel_baked_brownies (b : ℕ) (h : 3 * b / 5 = 18) : b = 30 :=
by
  sorry

end rachel_baked_brownies_l212_21247


namespace range_of_a_over_b_l212_21267

variable (a b : ℝ)

theorem range_of_a_over_b (ha : 1 < a ∧ a < 2) (hb : -2 < b ∧ b < -1) :
  -2 < a / b ∧ a / b < -1 / 2 :=
by
  sorry

end range_of_a_over_b_l212_21267


namespace range_of_k_for_distinct_real_roots_l212_21241

theorem range_of_k_for_distinct_real_roots (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 - 2 * x + 1 = 0) → (k < 2 ∧ k ≠ 1) :=
by
  sorry

end range_of_k_for_distinct_real_roots_l212_21241


namespace volleyball_team_starters_l212_21279

-- Define the team and the triplets
def total_players : ℕ := 14
def triplet_count : ℕ := 3
def remaining_players : ℕ := total_players - triplet_count

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the problem
theorem volleyball_team_starters : 
  C total_players 6 - C remaining_players 3 = 2838 :=
by sorry

end volleyball_team_starters_l212_21279


namespace white_square_area_l212_21269

theorem white_square_area
    (edge_length : ℝ)
    (total_paint : ℝ)
    (total_surface_area : ℝ)
    (green_paint_per_face : ℝ)
    (white_square_area_per_face: ℝ) :
    edge_length = 12 →
    total_paint = 432 →
    total_surface_area = 6 * (edge_length ^ 2) →
    green_paint_per_face = total_paint / 6 →
    white_square_area_per_face = (edge_length ^ 2) - green_paint_per_face →
    white_square_area_per_face = 72
:= sorry

end white_square_area_l212_21269


namespace rem_sum_a_b_c_l212_21289

theorem rem_sum_a_b_c (a b c : ℤ) (h1 : a * b * c ≡ 1 [ZMOD 5]) (h2 : 3 * c ≡ 1 [ZMOD 5]) (h3 : 4 * b ≡ 1 + b [ZMOD 5]) : 
  (a + b + c) % 5 = 3 := by 
  sorry

end rem_sum_a_b_c_l212_21289


namespace P_and_S_could_not_be_fourth_l212_21268

-- Define the relationships between the runners using given conditions
variables (P Q R S T U : ℕ)

axiom P_beats_Q : P < Q
axiom Q_beats_R : Q < R
axiom R_beats_S : R < S
axiom T_after_P_before_R : P < T ∧ T < R
axiom U_before_R_after_S : S < U ∧ U < R

-- Prove that P and S could not be fourth
theorem P_and_S_could_not_be_fourth : ¬((Q < U ∧ U < P) ∨ (Q > S ∧ S < P)) :=
by sorry

end P_and_S_could_not_be_fourth_l212_21268
