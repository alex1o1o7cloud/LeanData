import Mathlib

namespace sum_prime_factors_143_l147_147007

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 143 = p * q ∧ p + q = 24 :=
begin
  use 13,
  use 11,
  repeat { split },
  { exact nat.prime_of_four_divisors 13 (by norm_num) },
  { exact nat.prime_of_four_divisors 11 (by norm_num) },
  { norm_num },
  { norm_num }
end

end sum_prime_factors_143_l147_147007


namespace asha_remaining_money_l147_147341

-- Define the borrowed amounts, gift, and savings
def borrowed_from_brother : ℤ := 20
def borrowed_from_father : ℤ := 40
def borrowed_from_mother : ℤ := 30
def gift_from_granny : ℤ := 70
def savings : ℤ := 100

-- Total amount of money Asha has
def total_amount : ℤ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings

-- Amount spent by Asha
def amount_spent : ℤ := (3 * total_amount) / 4

-- Amount of money Asha remains with
def amount_left : ℤ := total_amount - amount_spent

-- The proof statement
theorem asha_remaining_money : amount_left = 65 := by
  sorry

end asha_remaining_money_l147_147341


namespace A_time_to_cover_distance_is_45_over_y_l147_147321

variable (y : ℝ)
variable (h0 : y > 0)
variable (h1 : (45 : ℝ) / (y - 2 / 3) - (45 : ℝ) / y = 3 / 4)

theorem A_time_to_cover_distance_is_45_over_y :
  45 / y = 45 / y :=
by
  sorry

end A_time_to_cover_distance_is_45_over_y_l147_147321


namespace log_domain_l147_147784

theorem log_domain (x : ℝ) : 3 - 2 * x > 0 ↔ x < 3 / 2 :=
by
  sorry

end log_domain_l147_147784


namespace problem_l147_147113

def f (x : ℝ) (a b c d : ℝ) : ℝ := a * x^7 + b * x^5 - c * x^3 + d * x + 3

theorem problem (a b c d : ℝ) (h : f 92 a b c d = 2) : f 92 a b c d + f (-92) a b c d = 6 :=
by
  sorry

end problem_l147_147113


namespace domino_arrangement_possible_l147_147740

def set_contains {α : Type*} (s : set α) (a : α) : Prop := a ∈ s

noncomputable def domino_set : Type :=
  {s : set (ℕ × ℕ) // ∀ a b, a ≤ b → (a, b) ∈ s → a ≤ 9 ∧ b ≤ 9}

def is_single_line_arrangement (dominoes : set (ℕ × ℕ)) : Prop :=
  ∃ (path : list (ℕ × ℕ)), ∀ (p : (ℕ × ℕ)), p ∈ dominoes ↔ p ∈ path.to_finset

def remaining_dominoes : set (ℕ × ℕ) :=
  (domino_set \ {(7, 6), (5, 4), (3, 2), (1, 0)} : set (ℕ × ℕ))

theorem domino_arrangement_possible :
  is_single_line_arrangement remaining_dominoes :=
sorry

end domino_arrangement_possible_l147_147740


namespace seq_a_eval_a4_l147_147086

theorem seq_a_eval_a4 (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 2, a n = 2 * a (n - 1) + 1) : a 4 = 15 :=
sorry

end seq_a_eval_a4_l147_147086


namespace coin_flip_prob_nickel_halfdollar_heads_l147_147531

def coin_prob : ℚ :=
  let total_outcomes := 2^5
  let successful_outcomes := 2^3
  successful_outcomes / total_outcomes

theorem coin_flip_prob_nickel_halfdollar_heads :
  coin_prob = 1 / 4 :=
by
  sorry

end coin_flip_prob_nickel_halfdollar_heads_l147_147531


namespace carlos_improved_lap_time_l147_147343

-- Define the initial condition using a function to denote time per lap initially
def initial_lap_time : ℕ := (45 * 60) / 15

-- Define the later condition using a function to denote time per lap later on
def current_lap_time : ℕ := (42 * 60) / 18

-- Define the proof that calculates the improvement in seconds
theorem carlos_improved_lap_time : initial_lap_time - current_lap_time = 40 := by
  sorry

end carlos_improved_lap_time_l147_147343


namespace compare_expression_l147_147822

variable (m x : ℝ)

theorem compare_expression : x^2 - x + 1 > -2 * m^2 - 2 * m * x := 
sorry

end compare_expression_l147_147822


namespace focus_of_parabola_l147_147785

def parabola (x : ℝ) : ℝ := (x - 3) ^ 2

theorem focus_of_parabola :
  ∃ f : ℝ × ℝ, f = (3, 1 / 4) ∧
  ∀ x : ℝ, parabola x = (x - 3)^2 :=
sorry

end focus_of_parabola_l147_147785


namespace point_B_number_l147_147259

theorem point_B_number (A B : ℤ) (hA : A = -2) (hB : abs (B - A) = 3) : B = 1 ∨ B = -5 :=
sorry

end point_B_number_l147_147259


namespace find_k_l147_147929

theorem find_k :
  ∀ (k : ℤ),
    (∃ a1 a2 a3 : ℤ,
        a1 = 49 + k ∧
        a2 = 225 + k ∧
        a3 = 484 + k ∧
        2 * a2 = a1 + a3) →
    k = 324 :=
by
  sorry

end find_k_l147_147929


namespace maximum_root_l147_147212

noncomputable def max_root (α β γ : ℝ) : ℝ := 
  if α ≥ β ∧ α ≥ γ then α 
  else if β ≥ α ∧ β ≥ γ then β 
  else γ

theorem maximum_root :
  ∃ α β γ : ℝ, α + β + γ = 14 ∧ α^2 + β^2 + γ^2 = 84 ∧ α^3 + β^3 + γ^3 = 584 ∧ max_root α β γ = 8 :=
by
  sorry

end maximum_root_l147_147212


namespace number_of_players_sold_eq_2_l147_147330

def initial_balance : ℕ := 100
def selling_price_per_player : ℕ := 10
def buying_cost_per_player : ℕ := 15
def number_of_players_bought : ℕ := 4
def final_balance : ℕ := 60

theorem number_of_players_sold_eq_2 :
  ∃ x : ℕ, (initial_balance + selling_price_per_player * x - buying_cost_per_player * number_of_players_bought = final_balance) ∧ (x = 2) :=
by
  sorry

end number_of_players_sold_eq_2_l147_147330


namespace mean_is_not_51_l147_147888

def frequencies : List Nat := [5, 8, 7, 13, 7]
def pH_values : List Float := [4.8, 4.9, 5.0, 5.2, 5.3]

def total_observations : Nat := List.sum frequencies

def mean (freqs : List Nat) (values : List Float) : Float :=
  let weighted_sum := List.sum (List.zipWith (· * ·) values (List.map (Float.ofNat) freqs))
  weighted_sum / (Float.ofNat total_observations)

theorem mean_is_not_51 : mean frequencies pH_values ≠ 5.1 := by
  -- Proof skipped
  sorry

end mean_is_not_51_l147_147888


namespace repeating_decimal_to_fraction_l147_147772

theorem repeating_decimal_to_fraction (x : ℝ) (h : x = 0.3 + 0.0666...) : x = 11 / 30 := by
  sorry

end repeating_decimal_to_fraction_l147_147772


namespace herman_days_per_week_l147_147375

-- Defining the given conditions as Lean definitions
def total_meals : ℕ := 4
def cost_per_meal : ℕ := 4
def total_weeks : ℕ := 16
def total_cost : ℕ := 1280

-- Calculating derived facts based on given conditions
def cost_per_day : ℕ := total_meals * cost_per_meal
def cost_per_week : ℕ := total_cost / total_weeks

-- Our main theorem that states Herman buys breakfast combos 5 days per week
theorem herman_days_per_week : cost_per_week / cost_per_day = 5 :=
by
  -- Skipping the proof
  sorry

end herman_days_per_week_l147_147375


namespace findQuadraticFunctionAndVertex_l147_147218

noncomputable section

def quadraticFunction (x : ℝ) (b c : ℝ) : ℝ :=
  (1 / 2) * x^2 + b * x + c

theorem findQuadraticFunctionAndVertex :
  (∃ b c : ℝ, quadraticFunction 0 b c = -1 ∧ quadraticFunction 2 b c = -3) →
  (quadraticFunction x (-2) (-1) = (1 / 2) * x^2 - 2 * x - 1) ∧
  (∃ (vₓ vᵧ : ℝ), vₓ = 2 ∧ vᵧ = -3 ∧ quadraticFunction vₓ (-2) (-1) = vᵧ)  :=
by
  sorry

end findQuadraticFunctionAndVertex_l147_147218


namespace find_m_l147_147144

-- Define the conditions
def ellipse_eq (x y : ℝ) (m : ℝ) : Prop := (x^2) / m + (y^2) / 4 = 1
def eccentricity (e : ℝ) : Prop := e = 2

-- Statement of the problem
theorem find_m (m : ℝ) (e : ℝ) (h1 : eccentricity e) (h2 : ∀ x y : ℝ, ellipse_eq x y m) :
  m = 3 ∨ m = 5 :=
sorry

end find_m_l147_147144


namespace sum_of_solutions_sum_of_solutions_is_16_l147_147629

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l147_147629


namespace selling_price_of_radio_l147_147533

theorem selling_price_of_radio (CP LP : ℝ) (hCP : CP = 1500) (hLP : LP = 14.000000000000002) : 
  CP - (LP / 100 * CP) = 1290 :=
by
  -- Given definitions
  have h1 : CP - (LP / 100 * CP) = 1290 := sorry
  exact h1

end selling_price_of_radio_l147_147533


namespace problem_trapezoid_l147_147814

noncomputable def ratio_of_areas (AB CD : ℝ) (h : ℝ) (ratio : ℝ) :=
  let area_trapezoid := (AB + CD) * h / 2
  let area_triangle_AZW := (4 * h) / 15
  ratio = area_triangle_AZW / area_trapezoid

theorem problem_trapezoid :
  ratio_of_areas 2 5 h (8 / 105) :=
by
  sorry

end problem_trapezoid_l147_147814


namespace karen_cases_picked_up_l147_147110

theorem karen_cases_picked_up (total_boxes : ℤ) (boxes_per_case : ℤ) (h1 : total_boxes = 36) (h2 : boxes_per_case = 12) : (total_boxes / boxes_per_case) = 3 := by
  sorry

end karen_cases_picked_up_l147_147110


namespace intersecting_lines_l147_147159

theorem intersecting_lines (p q r s t : ℝ) : (∃ u v : ℝ, p * u^2 + q * v^2 + r * u + s * v + t = 0) →
  ( ∃ p q : ℝ, p * q < 0 ∧ 4 * t = r^2 / p + s^2 / q ) :=
sorry

end intersecting_lines_l147_147159


namespace probability_same_color_correct_l147_147225

-- conditions
def sides := ["maroon", "teal", "cyan", "sparkly"]
def die : Type := {v // v ∈ sides}
def maroon_count := 6
def teal_count := 9
def cyan_count := 10
def sparkly_count := 5
def total_sides := 30

-- calculate probabilities
def prob (count : ℕ) : ℚ := (count ^ 2) / (total_sides ^ 2)
def prob_same_color : ℚ :=
  prob maroon_count +
  prob teal_count +
  prob cyan_count +
  prob sparkly_count

-- statement
theorem probability_same_color_correct :
  prob_same_color = 121 / 450 :=
sorry

end probability_same_color_correct_l147_147225


namespace line_circle_intersect_l147_147982

theorem line_circle_intersect {a : ℝ} :
  ∃ P : ℝ × ℝ, (P.1, P.2) = (-2, 0) ∧ (a * P.1 - P.2 + 2 * a = 0) ∧ (P.1^2 + P.2^2 < 9) :=
by
  use (-2, 0)
  sorry

end line_circle_intersect_l147_147982


namespace runner_time_second_half_l147_147020

theorem runner_time_second_half (v : ℝ) (h1 : 20 / v + 4 = 40 / v) : 40 / v = 8 :=
by
  sorry

end runner_time_second_half_l147_147020


namespace six_digit_palindromes_count_l147_147382

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def is_non_zero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem six_digit_palindromes_count : 
  (∃a b c : ℕ, is_non_zero_digit a ∧ is_digit b ∧ is_digit c) → 
  (∃ n : ℕ, n = 900) :=
by
  sorry

end six_digit_palindromes_count_l147_147382


namespace solve_equation1_solve_equation2_l147_147138

open Real

theorem solve_equation1 (x : ℝ) : (x - 2)^2 = 9 → (x = 5 ∨ x = -1) :=
by
  intro h
  sorry -- Proof would go here

theorem solve_equation2 (x : ℝ) : (2 * x^2 - 3 * x - 1 = 0) → (x = (3 + sqrt 17) / 4 ∨ x = (3 - sqrt 17) / 4) :=
by
  intro h
  sorry -- Proof would go here

end solve_equation1_solve_equation2_l147_147138


namespace intersection_points_l147_147764

noncomputable def h (x : ℝ) : ℝ := -x^2 - 4 * x + 1
noncomputable def j (x : ℝ) : ℝ := -h x
noncomputable def k (x : ℝ) : ℝ := h (-x)

def c : ℕ := 2 -- Number of intersections of y = h(x) and y = j(x)
def d : ℕ := 1 -- Number of intersections of y = h(x) and y = k(x)

theorem intersection_points :
  10 * c + d = 21 := by
  sorry

end intersection_points_l147_147764


namespace Nicole_has_69_clothes_l147_147513

def clothingDistribution : Prop :=
  let nicole_clothes := 15
  let first_sister_clothes := nicole_clothes / 3
  let second_sister_clothes := nicole_clothes + 5
  let third_sister_clothes := 2 * first_sister_clothes
  let average_clothes := (nicole_clothes + first_sister_clothes + second_sister_clothes + third_sister_clothes) / 4
  let oldest_sister_clothes := 1.5 * average_clothes
  let total_clothes := nicole_clothes + first_sister_clothes + second_sister_clothes + third_sister_clothes + oldest_sister_clothes
  total_clothes = 69

theorem Nicole_has_69_clothes : clothingDistribution :=
by
  -- Proof omitted
  sorry

end Nicole_has_69_clothes_l147_147513


namespace volume_of_rectangular_prism_l147_147680

-- Given conditions translated into Lean definitions
variables (AB AD AC1 AA1 : ℕ)

def rectangular_prism_properties : Prop :=
  AB = 2 ∧ AD = 2 ∧ AC1 = 3 ∧ AA1 = 1

-- The mathematical volume of the rectangular prism
def volume (AB AD AA1 : ℕ) := AB * AD * AA1

-- Prove that given the conditions, the volume of the rectangular prism is 4
theorem volume_of_rectangular_prism (h : rectangular_prism_properties AB AD AC1 AA1) : volume AB AD AA1 = 4 :=
by
  sorry

#check volume_of_rectangular_prism

end volume_of_rectangular_prism_l147_147680


namespace at_least_one_not_less_than_one_l147_147207

theorem at_least_one_not_less_than_one (x : ℝ) (a b c : ℝ) 
  (ha : a = x^2 + 1/2) 
  (hb : b = 2 - x) 
  (hc : c = x^2 - x + 1) : 
  (1 ≤ a) ∨ (1 ≤ b) ∨ (1 ≤ c) := 
sorry

end at_least_one_not_less_than_one_l147_147207


namespace exercise_l147_147673

-- Define the given expression.
def f (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 5

-- Define the general form expression.
def g (x h k : ℝ) (a : ℝ) := a * (x - h)^2 + k

-- Prove that a + h + k = 6 when expressing f(x) in the form a(x-h)^2 + k.
theorem exercise : ∃ a h k : ℝ, (∀ x : ℝ, f x = g x h k a) ∧ (a + h + k = 6) :=
by
  sorry

end exercise_l147_147673


namespace cost_of_meal_l147_147049

noncomputable def total_cost (hamburger_cost fry_cost drink_cost : ℕ) (num_hamburgers num_fries num_drinks : ℕ) (discount_rate : ℕ) : ℕ :=
  let initial_cost := (hamburger_cost * num_hamburgers) + (fry_cost * num_fries) + (drink_cost * num_drinks)
  let discount := initial_cost * discount_rate / 100
  initial_cost - discount

theorem cost_of_meal :
  total_cost 5 3 2 3 4 6 10 = 35 := by
  sorry

end cost_of_meal_l147_147049


namespace sufficient_but_not_necessary_condition_for_intersections_l147_147921

theorem sufficient_but_not_necessary_condition_for_intersections
  (k : ℝ) (h : 0 < k ∧ k < 3) :
  ∃ x y : ℝ, (x - y - k = 0) ∧ ((x - 1)^2 + y^2 = 2) :=
sorry

end sufficient_but_not_necessary_condition_for_intersections_l147_147921


namespace tank_depth_l147_147580

open Real

theorem tank_depth :
  ∃ d : ℝ, (0.75 * (2 * 25 * d + 2 * 12 * d + 25 * 12) = 558) ∧ d = 6 :=
sorry

end tank_depth_l147_147580


namespace geometric_seq_tenth_term_l147_147562

theorem geometric_seq_tenth_term :
  let a := 12
  let r := (1 / 2 : ℝ)
  (a * r^9) = (3 / 128 : ℝ) :=
by
  let a := 12
  let r := (1 / 2 : ℝ)
  show a * r^9 = 3 / 128
  sorry

end geometric_seq_tenth_term_l147_147562


namespace chocoBites_mod_l147_147200

theorem chocoBites_mod (m : ℕ) (hm : m % 8 = 5) : (4 * m) % 8 = 4 :=
by
  sorry

end chocoBites_mod_l147_147200


namespace common_ratio_of_geometric_series_l147_147217

theorem common_ratio_of_geometric_series (a₁ q : ℝ) 
  (S_3 : ℝ) (S_2 : ℝ) 
  (hS3 : S_3 = a₁ * (1 - q^3) / (1 - q)) 
  (hS2 : S_2 = a₁ * (1 - q^2) / (1 - q)) 
  (h_ratio : S_3 / S_2 = 3 / 2) :
  q = 1 ∨ q = -1/2 :=
by
  -- Proof goes here.
  sorry

end common_ratio_of_geometric_series_l147_147217


namespace find_x_l147_147914

variable (x : ℝ)

theorem find_x (h : 2 * x - 12 = -(x + 3)) : x = 3 := 
sorry

end find_x_l147_147914


namespace marbles_solution_l147_147154

def marbles_problem : Prop :=
  ∃ (marbles : Finset (Finset ℕ)), 
    marbles.card = 28 ∧ ∀ m ∈ marbles,
    (∃ (c ∈ {0, 1, 2, 3}), m = {c, c} ∨
     ∃ (c1 c2 ∈ {0, 1, 2, 3}), c1 ≠ c2 ∧ m = {c1, c2})

theorem marbles_solution : marbles_problem :=
sorry

end marbles_solution_l147_147154


namespace numberOfBoysInClass_l147_147047

-- Define the problem condition: students sit in a circle and boy at 5th position is opposite to boy at 20th position
def studentsInCircle (n : ℕ) : Prop :=
  (n > 5) ∧ (n > 20) ∧ ((20 - 5) * 2 + 2 = n)

-- The main theorem: Given the conditions, prove the total number of boys equals 32
theorem numberOfBoysInClass : ∀ n : ℕ, studentsInCircle n → n = 32 :=
by
  intros n hn
  sorry

end numberOfBoysInClass_l147_147047


namespace fibonacci_sequence_x_l147_147124

theorem fibonacci_sequence_x {a : ℕ → ℕ} 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2) 
  (h3 : a 3 = 3) 
  (h_fib : ∀ n, n ≥ 3 → a (n + 1) = a n + a (n - 1)) : 
  a 5 = 8 := 
sorry

end fibonacci_sequence_x_l147_147124


namespace triangle_equi_if_sides_eq_sum_of_products_l147_147645

theorem triangle_equi_if_sides_eq_sum_of_products (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + bc + ac) : a = b ∧ b = c :=
by sorry

end triangle_equi_if_sides_eq_sum_of_products_l147_147645


namespace incorrect_proposition_statement_l147_147861

theorem incorrect_proposition_statement (p q : Prop) : (p ∨ q) → ¬ (p ∧ q) := 
sorry

end incorrect_proposition_statement_l147_147861


namespace rahul_matches_played_l147_147841

theorem rahul_matches_played
  (current_avg : ℕ)
  (runs_today : ℕ)
  (new_avg : ℕ)
  (m: ℕ)
  (h1 : current_avg = 51)
  (h2 : runs_today = 78)
  (h3 : new_avg = 54)
  (h4 : (51 * m + runs_today) / (m + 1) = new_avg) :
  m = 8 :=
by
  sorry

end rahul_matches_played_l147_147841


namespace multiple_of_weight_lifted_l147_147059

variable (F : ℝ) (M : ℝ)

theorem multiple_of_weight_lifted 
  (H1: ∀ (B : ℝ), B = 2 * F) 
  (H2: ∀ (B : ℝ), ∀ (W : ℝ), W = 3 * B) 
  (H3: ∃ (B : ℝ), (3 * B = 600)) 
  (H4: M * F = 150) : 
  M = 1.5 :=
by
  sorry

end multiple_of_weight_lifted_l147_147059


namespace asha_remaining_money_l147_147339

-- Define the borrowed amounts, gift, and savings
def borrowed_from_brother : ℤ := 20
def borrowed_from_father : ℤ := 40
def borrowed_from_mother : ℤ := 30
def gift_from_granny : ℤ := 70
def savings : ℤ := 100

-- Total amount of money Asha has
def total_amount : ℤ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings

-- Amount spent by Asha
def amount_spent : ℤ := (3 * total_amount) / 4

-- Amount of money Asha remains with
def amount_left : ℤ := total_amount - amount_spent

-- The proof statement
theorem asha_remaining_money : amount_left = 65 := by
  sorry

end asha_remaining_money_l147_147339


namespace eating_time_l147_147825

-- Define the eating rates of Mr. Fat, Mr. Thin, and Mr. Medium
def mrFat_rate := 1 / 15
def mrThin_rate := 1 / 35
def mrMedium_rate := 1 / 25

-- Define the combined eating rate
def combined_rate := mrFat_rate + mrThin_rate + mrMedium_rate

-- Define the amount of cereal to be eaten
def amount_cereal := 5

-- Prove that the time taken to eat the cereal is 2625 / 71 minutes
theorem eating_time : amount_cereal / combined_rate = 2625 / 71 :=
by 
  -- Here should be the proof, but it is skipped
  sorry

end eating_time_l147_147825


namespace original_weight_l147_147334

variable (W : ℝ) -- Let W be the original weight of the side of beef

-- Conditions
def condition1 : ℝ := 0.80 * W -- Weight after first stage
def condition2 : ℝ := 0.70 * condition1 W -- Weight after second stage
def condition3 : ℝ := 0.75 * condition2 W -- Weight after third stage

-- Final weight is given as 570 pounds
theorem original_weight (h : condition3 W = 570) : W = 1357.14 :=
by 
  sorry

end original_weight_l147_147334


namespace regular_pyramid_sufficient_condition_l147_147150

-- Define the basic structure of a pyramid
structure Pyramid :=
  (lateral_face_is_equilateral_triangle : Prop)  
  (base_is_square : Prop)  
  (apex_angles_of_lateral_face_are_45_deg : Prop)  
  (projection_of_vertex_at_intersection_of_base_diagonals : Prop)
  (is_regular : Prop)

-- Define the hypothesis conditions
variables 
  (P : Pyramid)
  (h1 : P.lateral_face_is_equilateral_triangle)
  (h2 : P.base_is_square)
  (h3 : P.apex_angles_of_lateral_face_are_45_deg)
  (h4 : P.projection_of_vertex_at_intersection_of_base_diagonals)

-- Define the statement of the proof
theorem regular_pyramid_sufficient_condition :
  (P.lateral_face_is_equilateral_triangle → P.is_regular) ∧ 
  (¬(P.lateral_face_is_equilateral_triangle) → ¬P.is_regular) ↔
  (P.lateral_face_is_equilateral_triangle ∧ ¬P.base_is_square ∧ ¬P.apex_angles_of_lateral_face_are_45_deg ∧ ¬P.projection_of_vertex_at_intersection_of_base_diagonals) := 
by { sorry }


end regular_pyramid_sufficient_condition_l147_147150


namespace purely_imaginary_implies_m_eq_neg_half_simplify_z_squared_over_z_add_5_plus_2i_l147_147475

def z (m : ℝ) : Complex := Complex.mk (2 * m^2 - 3 * m - 2) (m^2 - 3 * m + 2)

theorem purely_imaginary_implies_m_eq_neg_half (m : ℝ) : 
  (z m).re = 0 ↔ m = -1 / 2 := sorry

theorem simplify_z_squared_over_z_add_5_plus_2i (z_zero : ℂ) :
  z 0 = ⟨-2, 2⟩ →
  (z 0)^2 / (z 0 + Complex.mk 5 2) = ⟨-32 / 25, -24 / 25⟩ := sorry

end purely_imaginary_implies_m_eq_neg_half_simplify_z_squared_over_z_add_5_plus_2i_l147_147475


namespace find_m_l147_147918

-- Definitions based on conditions
def is_eccentricity (a b c : ℝ) (e : ℝ) : Prop :=
  e = c / a

def ellipse_relation (a b m : ℝ) : Prop :=
  a ^ 2 = 3 ∧ b ^ 2 = m

def eccentricity_square_relation (c a : ℝ) : Prop :=
  (c / a) ^ 2 = 1 / 4

-- Main theorem statement
theorem find_m (m : ℝ) :
  (∀ (a b c : ℝ), ellipse_relation a b m → is_eccentricity a b c (1 / 2) → eccentricity_square_relation c a)
  → (m = 9 / 4 ∨ m = 4) := sorry

end find_m_l147_147918


namespace yo_yos_collected_l147_147817

-- Define the given conditions
def stuffed_animals : ℕ := 14
def frisbees : ℕ := 18
def total_prizes : ℕ := 50

-- Define the problem to prove that the number of yo-yos is 18
theorem yo_yos_collected : (total_prizes - (stuffed_animals + frisbees) = 18) :=
by
  sorry

end yo_yos_collected_l147_147817


namespace smallest_number_am_median_l147_147722

theorem smallest_number_am_median :
  ∃ (a b c : ℕ), a + b + c = 90 ∧ b = 28 ∧ c = b + 6 ∧ (a ≤ b ∧ b ≤ c) ∧ a = 28 :=
by
  sorry

end smallest_number_am_median_l147_147722


namespace index_difference_l147_147864

theorem index_difference (n f m : ℕ) (h_n : n = 25) (h_f : f = 8) (h_m : m = 25 - 8) :
  (n - f) / n - (n - m) / n = 9 / 25 :=
by
  -- The proof is to be completed here.
  sorry

end index_difference_l147_147864


namespace difference_students_guinea_pigs_l147_147461

-- Define the conditions as constants
def students_per_classroom : Nat := 20
def guinea_pigs_per_classroom : Nat := 3
def number_of_classrooms : Nat := 6

-- Calculate the total number of students
def total_students : Nat := students_per_classroom * number_of_classrooms

-- Calculate the total number of guinea pigs
def total_guinea_pigs : Nat := guinea_pigs_per_classroom * number_of_classrooms

-- Define the theorem to prove the equality
theorem difference_students_guinea_pigs :
  total_students - total_guinea_pigs = 102 :=
by
  sorry -- Proof to be filled in

end difference_students_guinea_pigs_l147_147461


namespace coin_toss_5_times_same_side_l147_147318

noncomputable def probability_of_same_side (n : ℕ) : ℝ :=
  (1 / 2) ^ n

theorem coin_toss_5_times_same_side :
  probability_of_same_side 5 = 1 / 32 :=
by 
  -- The goal is to prove (1/2)^5 = 1/32
  sorry

end coin_toss_5_times_same_side_l147_147318


namespace distance_A_B_l147_147152

theorem distance_A_B (d : ℝ)
  (speed_A : ℝ := 100) (speed_B : ℝ := 90) (speed_C : ℝ := 75)
  (location_A location_B : point) (is_at_A : location_A = point_A) (is_at_B : location_B = point_B)
  (t_meet_AB : ℝ := d / (speed_A + speed_B))
  (t_meet_AC : ℝ := t_meet_AB + 3)
  (distance_AC : ℝ := speed_A * 3)
  (distance_C : ℝ := speed_C * t_meet_AC) :
  d = 650 :=
by {
  sorry
}

end distance_A_B_l147_147152


namespace range_of_f_l147_147206

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |x| - 1 else Real.sin x ^ 2

theorem range_of_f : Set.range f = Set.Ioi (-1) := 
  sorry

end range_of_f_l147_147206


namespace probability_losing_ticket_l147_147386

theorem probability_losing_ticket (winning : ℕ) (losing : ℕ)
  (h_odds : winning = 5 ∧ losing = 8) :
  (losing : ℚ) / (winning + losing : ℚ) = 8 / 13 := by
  sorry

end probability_losing_ticket_l147_147386


namespace hilltop_high_students_l147_147583

theorem hilltop_high_students : 
  ∀ (n_sophomore n_freshman n_junior : ℕ), 
  (n_sophomore : ℚ) / n_freshman = 7 / 4 ∧ (n_junior : ℚ) / n_sophomore = 6 / 7 → 
  n_sophomore + n_freshman + n_junior = 17 :=
by
  sorry

end hilltop_high_students_l147_147583


namespace total_bugs_eaten_l147_147241

theorem total_bugs_eaten :
  let gecko_bugs := 12
  let lizard_bugs := gecko_bugs / 2
  let frog_bugs := lizard_bugs * 3
  let toad_bugs := frog_bugs + (frog_bugs / 2)
  gecko_bugs + lizard_bugs + frog_bugs + toad_bugs = 63 :=
by
  sorry

end total_bugs_eaten_l147_147241


namespace probability_of_purple_l147_147410

def total_faces := 10
def purple_faces := 3

theorem probability_of_purple : (purple_faces : ℚ) / (total_faces : ℚ) = 3 / 10 := 
by 
  sorry

end probability_of_purple_l147_147410


namespace find_y_l147_147487

theorem find_y (y : ℚ) (h : 1/3 - 1/4 = 4/y) : y = 48 := sorry

end find_y_l147_147487


namespace smallest_sum_Q_lt_7_9_l147_147344

def Q (N k : ℕ) : ℚ := (N + 1) / (N + k + 1)

theorem smallest_sum_Q_lt_7_9 : 
    ∃ N k : ℕ, (N + k) % 4 = 0 ∧ Q N k < 7 / 9 ∧ (∀ N' k' : ℕ, (N' + k') % 4 = 0 ∧ Q N' k' < 7 / 9 → N' + k' ≥ N + k) ∧ N + k = 4 :=
by
  sorry

end smallest_sum_Q_lt_7_9_l147_147344


namespace square_remainder_is_square_l147_147966

theorem square_remainder_is_square (a : ℤ) : ∃ b : ℕ, (a^2 % 16 = b) ∧ (∃ c : ℕ, b = c^2) :=
by
  sorry

end square_remainder_is_square_l147_147966


namespace mitzi_money_left_l147_147694

theorem mitzi_money_left :
  let A := 75
  let T := 30
  let F := 13
  let S := 23
  let total_spent := T + F + S
  let money_left := A - total_spent
  money_left = 9 :=
by
  sorry

end mitzi_money_left_l147_147694


namespace seashells_left_sam_seashells_now_l147_147842

-- Problem conditions
def initial_seashells : ℕ := 35
def seashells_given : ℕ := 18

-- Proof problem statement
theorem seashells_left (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

-- The required statement
theorem sam_seashells_now : seashells_left initial_seashells seashells_given = 17 := by 
  sorry

end seashells_left_sam_seashells_now_l147_147842


namespace no_real_roots_of_quadratic_l147_147856

theorem no_real_roots_of_quadratic (a b : ℝ) : (∀ x : ℝ, x^2 + a * x + b ≠ 0) ↔ ¬ (∃ x : ℝ, x^2 + a * x + b = 0) := sorry

end no_real_roots_of_quadratic_l147_147856


namespace distinct_fractions_exists_l147_147833

theorem distinct_fractions_exists : 
  ∃ (fractions : Finset (ℚ → ℚ)), 
  (∀ (f : ℚ → ℚ), f ∈ fractions ↔ 
    ∃ (a b c d : ℚ), 
      (Set.mem (perm (2, 1, 0, 0)) (a, b, c, d) ∧ 
      (c ≠ 0 ∨ d ≠ 0) ∧ 
      f = λ x, (a * x + b) / (c * x + d))) ∧
  fractions.card = 7 :=
begin
  sorry
end

end distinct_fractions_exists_l147_147833


namespace third_divisor_l147_147424

/-- 
Given that the new number after subtracting 7 from 3,381 leaves a remainder of 8 when divided by 9 
and 11, prove that the third divisor that also leaves a remainder of 8 is 17.
-/
theorem third_divisor (x : ℕ) (h1 : x = 3381 - 7)
                      (h2 : x % 9 = 8)
                      (h3 : x % 11 = 8) :
  ∃ (d : ℕ), d = 17 ∧ x % d = 8 := sorry

end third_divisor_l147_147424


namespace a_b_condition_l147_147421

theorem a_b_condition (a b : ℂ) (h : (a + b) / a = b / (a + b)) :
  (∃ x y : ℂ, x = a ∧ y = b ∧ ((¬ x.im = 0 ∧ y.im = 0) ∨ (x.im = 0 ∧ ¬ y.im = 0) ∨ (¬ x.im = 0 ∧ ¬ y.im = 0))) :=
by
  sorry

end a_b_condition_l147_147421


namespace numCounterexamplesCorrect_l147_147463

-- Define a function to calculate the sum of digits of a number
def digitSum (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Predicate to check if a number is prime
def isPrime (n : Nat) : Prop := 
  Nat.Prime n

-- Set definition where the sum of digits must be 5 and all digits are non-zero
def validSet (n : Nat) : Prop :=
  digitSum n = 5 ∧ ∀ d ∈ n.digits 10, d ≠ 0

-- Define the number of counterexamples
def numCounterexamples : Nat := 6

-- The final theorem stating the number of counterexamples
theorem numCounterexamplesCorrect :
  (∃ ns : Finset Nat, 
    (∀ n ∈ ns, validSet n) ∧ 
    (∀ n ∈ ns, ¬ isPrime n) ∧ 
    ns.card = numCounterexamples) :=
sorry

end numCounterexamplesCorrect_l147_147463


namespace find_x_l147_147715

theorem find_x (x : ℝ) (h : (1 / 2) * x + (1 / 3) * x = (1 / 4) * x + 7) : x = 12 :=
by
  sorry

end find_x_l147_147715


namespace band_member_share_l147_147075

def num_people : ℕ := 500
def ticket_price : ℝ := 30
def band_share_percent : ℝ := 0.70
def num_band_members : ℕ := 4

theorem band_member_share : 
  (num_people * ticket_price * band_share_percent) / num_band_members = 2625 := by
  sorry

end band_member_share_l147_147075


namespace flea_reach_B_with_7_jumps_flea_reach_C_with_9_jumps_flea_cannot_reach_D_with_2028_jumps_l147_147870

-- Problem (a)
theorem flea_reach_B_with_7_jumps (A B : ℤ) (jumps : ℤ) (distance : ℤ) (ways : ℕ) :
  B = A + 5 → jumps = 7 → distance = 5 → 
  ways = Nat.choose (7) (1) := 
sorry

-- Problem (b)
theorem flea_reach_C_with_9_jumps (A C : ℤ) (jumps : ℤ) (distance : ℤ) (ways : ℕ) :
  C = A + 5 → jumps = 9 → distance = 5 → 
  ways = Nat.choose (9) (2) :=
sorry

-- Problem (c)
theorem flea_cannot_reach_D_with_2028_jumps (A D : ℤ) (jumps : ℤ) (distance : ℤ) :
  D = A + 2013 → jumps = 2028 → distance = 2013 → 
  ∃ x y : ℤ, x + y = 2028 ∧ x - y = 2013 → false :=
sorry

end flea_reach_B_with_7_jumps_flea_reach_C_with_9_jumps_flea_cannot_reach_D_with_2028_jumps_l147_147870


namespace six_digit_number_reversed_by_9_l147_147460

-- Hypothetical function to reverse digits of a number
def reverseDigits (n : ℕ) : ℕ := sorry

theorem six_digit_number_reversed_by_9 :
  ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n * 9 = reverseDigits n :=
by
  sorry

end six_digit_number_reversed_by_9_l147_147460


namespace evaluate_polynomial_at_2_l147_147762

def polynomial (x : ℕ) : ℕ := 3 * x^4 + x^3 + 2 * x^2 + x + 4

def horner_method (x : ℕ) : ℕ :=
  let v_0 := x
  let v_1 := 3 * v_0 + 1
  let v_2 := v_1 * v_0 + 2
  v_2

theorem evaluate_polynomial_at_2 :
  horner_method 2 = 16 :=
by
  sorry

end evaluate_polynomial_at_2_l147_147762


namespace non_talking_birds_count_l147_147576

def total_birds : ℕ := 77
def talking_birds : ℕ := 64

theorem non_talking_birds_count : total_birds - talking_birds = 13 := by
  sorry

end non_talking_birds_count_l147_147576


namespace min_value_of_M_l147_147844

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

noncomputable def M : ℝ :=
  (Real.rpow (a / (b + c)) (1 / 4)) + (Real.rpow (b / (c + a)) (1 / 4)) + (Real.rpow (c / (b + a)) (1 / 4)) +
  Real.sqrt ((b + c) / a) + Real.sqrt ((a + c) / b) + Real.sqrt ((a + b) / c)

theorem min_value_of_M : M a b c = 3 * Real.sqrt 2 + (3 * Real.rpow 8 (1 / 4)) / 2 := sorry

end min_value_of_M_l147_147844


namespace repeating_decimal_to_fraction_l147_147774

theorem repeating_decimal_to_fraction : (0.36 : ℝ) = (11 / 30 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l147_147774


namespace smallest_positive_period_minimum_value_of_f_center_of_symmetry_interval_of_increasing_l147_147479

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x - Real.pi / 6)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem minimum_value_of_f :
  ∃ x, f x = -3 :=
sorry

theorem center_of_symmetry (k : ℤ) :
  ∃ p, (∀ x, f (p + x) = f (p - x)) ∧ p = (Real.pi / 12) + (k * Real.pi / 2) :=
sorry

theorem interval_of_increasing (k : ℤ) :
  ∃ a b, a = -(Real.pi / 6) + k * Real.pi ∧ b = (Real.pi / 3) + k * Real.pi ∧
  ∀ x, (a <= x ∧ x <= b) → StrictMonoOn f (Set.Icc a b) :=
sorry

end smallest_positive_period_minimum_value_of_f_center_of_symmetry_interval_of_increasing_l147_147479


namespace factorization_option_D_l147_147162

-- Define variables
variables (x y : ℝ)

-- Define the expressions
def left_side_D := -4 * x^2 + 12 * x * y - 9 * y^2
def right_side_D := -(2 * x - 3 * y)^2

-- Theorem statement
theorem factorization_option_D : left_side_D x y = right_side_D x y :=
sorry

end factorization_option_D_l147_147162


namespace sum_prime_factors_of_143_l147_147010

theorem sum_prime_factors_of_143 : 
  let primes := {p : ℕ | p.prime ∧ p ∣ 143} in
  ∑ p in primes, p = 24 := 
by
  sorry

end sum_prime_factors_of_143_l147_147010


namespace find_n_for_arithmetic_sequence_l147_147208

variable {a : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (a₁ : ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a n = a₁ + n * d

theorem find_n_for_arithmetic_sequence (h_arith : is_arithmetic_sequence a (-1) 2)
  (h_nth_term : ∃ n : ℕ, a n = 15) : ∃ n : ℕ, n = 9 :=
by
  sorry

end find_n_for_arithmetic_sequence_l147_147208


namespace initial_quarters_l147_147258

-- Define the conditions
def quartersAfterLoss (x : ℕ) : ℕ := (4 * x) / 3
def quartersAfterThirdYear (x : ℕ) : ℕ := x - 4
def quartersAfterSecondYear (x : ℕ) : ℕ := x - 36
def quartersAfterFirstYear (x : ℕ) : ℕ := x * 2

-- The main theorem
theorem initial_quarters (x : ℕ) (h1 : quartersAfterLoss x = 140)
    (h2 : quartersAfterThirdYear 140 = 136)
    (h3 : quartersAfterSecondYear 136 = 100)
    (h4 : quartersAfterFirstYear 50 = 100) :
  x = 50 := by
  simp [quartersAfterFirstYear, quartersAfterSecondYear,
        quartersAfterThirdYear, quartersAfterLoss] at *
  sorry

end initial_quarters_l147_147258


namespace machines_job_completion_time_l147_147034

theorem machines_job_completion_time (t : ℕ) 
  (hR_rate : ∀ t, 1 / t = 1 / 216) 
  (hS_rate : ∀ t, 1 / t = 1 / 216) 
  (same_num_machines : ∀ R S, R = 9 ∧ S = 9) 
  (total_time : 12 = 12) 
  (jobs_completed : 1 = (18 / t) * 12) : 
  t = 216 := 
sorry

end machines_job_completion_time_l147_147034


namespace finishing_order_l147_147939

-- Definitions of conditions
def athletes := ["Grisha", "Sasha", "Lena"]

def overtakes : (String → ℕ) := 
  fun athlete =>
    if athlete = "Grisha" then 10
    else if athlete = "Sasha" then 4
    else if athlete = "Lena" then 6
    else 0

-- All three were never at the same point at the same time
def never_same_point_at_same_time : Prop := True -- Simplified for translation purpose

-- The main theorem stating the finishing order given the provided conditions
theorem finishing_order :
  never_same_point_at_same_time →
  (overtakes "Grisha" = 10) →
  (overtakes "Sasha" = 4) →
  (overtakes "Lena" = 6) →
  athletes = ["Grisha", "Sasha", "Lena"] :=
  by
    intro h1 h2 h3 h4
    exact sorry -- The proof is not required, just ensuring the statement is complete.


end finishing_order_l147_147939


namespace students_in_class_l147_147973

theorem students_in_class (n : ℕ) (h1 : (n : ℝ) * 100 = (n * 100 + 60 - 10)) 
  (h2 : (n : ℝ) * 98 = ((n : ℝ) * 100 - 50)) : n = 25 :=
sorry

end students_in_class_l147_147973


namespace expected_balls_in_original_position_proof_l147_147845

-- Define the problem conditions as Lean definitions
def n_balls : ℕ := 10

def probability_not_moved_by_one_rotation : ℚ := 7 / 10

def probability_not_moved_by_two_rotations : ℚ := (7 / 10) * (7 / 10)

def expected_balls_in_original_position : ℚ := n_balls * probability_not_moved_by_two_rotations

-- The statement representing the proof problem
theorem expected_balls_in_original_position_proof :
  expected_balls_in_original_position = 4.9 :=
  sorry

end expected_balls_in_original_position_proof_l147_147845


namespace find_b_find_area_of_ABC_l147_147809

variable {a b c : ℝ}
variable {B : ℝ}

-- Given Conditions
def given_conditions (a b c B : ℝ) := a = 4 ∧ c = 3 ∧ B = Real.arccos (1 / 8)

-- Proving b = sqrt(22)
theorem find_b (h : given_conditions a b c B) : b = Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) :=
by
  sorry

-- Proving the area of triangle ABC
theorem find_area_of_ABC (h : given_conditions a b c B) 
  (sinB : Real.sin B = 3 * Real.sqrt 7 / 8) : 
  (1 / 2) * a * c * Real.sin B = 9 * Real.sqrt 7 / 4 :=
by
  sorry

end find_b_find_area_of_ABC_l147_147809


namespace unique_triple_solution_l147_147347

theorem unique_triple_solution (a b c : ℝ) 
  (h1 : a * (b ^ 2 + c) = c * (c + a * b))
  (h2 : b * (c ^ 2 + a) = a * (a + b * c))
  (h3 : c * (a ^ 2 + b) = b * (b + c * a)) : 
  a = b ∧ b = c := 
sorry

end unique_triple_solution_l147_147347


namespace max_chips_with_constraints_l147_147262

theorem max_chips_with_constraints (n : ℕ) (h1 : n > 0) 
  (h2 : ∀ i j : ℕ, (i < n) → (j = i + 10 ∨ j = i + 15) → ((i % 25) = 0 ∨ (j % 25) = 0)) :
  n ≤ 25 := 
sorry

end max_chips_with_constraints_l147_147262


namespace molecular_weight_l147_147560

theorem molecular_weight (w8 : ℝ) (n : ℝ) (w1 : ℝ) (h1 : w8 = 2376) (h2 : n = 8) : w1 = 297 :=
by
  sorry

end molecular_weight_l147_147560


namespace percentage_disliked_by_both_l147_147835

theorem percentage_disliked_by_both 
  (total_comic_books : ℕ) 
  (percentage_females_like : ℕ) 
  (comic_books_males_like : ℕ) :
  total_comic_books = 300 →
  percentage_females_like = 30 →
  comic_books_males_like = 120 →
  ((total_comic_books - (total_comic_books * percentage_females_like / 100) - comic_books_males_like) * 100 / total_comic_books) = 30 :=
by
  intros h1 h2 h3
  sorry

end percentage_disliked_by_both_l147_147835


namespace consumption_increase_percentage_l147_147544

theorem consumption_increase_percentage (T C : ℝ) (T_pos : 0 < T) (C_pos : 0 < C) :
  (0.7 * (1 + x / 100) * T * C = 0.84 * T * C) → x = 20 :=
by sorry

end consumption_increase_percentage_l147_147544


namespace square_land_plot_area_l147_147429

theorem square_land_plot_area (side_length : ℕ) (h1 : side_length = 40) : side_length * side_length = 1600 :=
by
  sorry

end square_land_plot_area_l147_147429


namespace parabola_directrix_l147_147977

theorem parabola_directrix (x y : ℝ) (h : x^2 + 12 * y = 0) : y = 3 :=
sorry

end parabola_directrix_l147_147977


namespace second_job_hourly_wage_l147_147256

-- Definitions based on conditions
def total_wages : ℕ := 160
def first_job_wages : ℕ := 52
def second_job_hours : ℕ := 12

-- Proof statement
theorem second_job_hourly_wage : 
  (total_wages - first_job_wages) / second_job_hours = 9 :=
by
  sorry

end second_job_hourly_wage_l147_147256


namespace smallest_number_l147_147721

theorem smallest_number (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : (a + b + c) = 90) (h4 : b = 28) (h5 : b = c - 6) : a = 28 :=
by 
  sorry

end smallest_number_l147_147721


namespace sum_of_solutions_eq_16_l147_147626

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l147_147626


namespace cos_C_value_triangle_perimeter_l147_147500

variables (A B C a b c : ℝ)
variables (cos_B : ℝ) (A_eq_2B : A = 2 * B) (cos_B_val : cos_B = 2 / 3)
variables (dot_product_88 : a * b * (Real.cos C) = 88)

theorem cos_C_value (A B : ℝ) (a b : ℝ) (cos_B : ℝ) (cos_C : ℝ) (dot_product_88 : a * b * cos_C = 88) :
  A = 2 * B →
  cos_B = 2 / 3 →
  cos_C = 22 / 27 :=
sorry

theorem triangle_perimeter (A B C a b c : ℝ) (cos_B : ℝ)
  (A_eq_2B : A = 2 * B) (cos_B_val : cos_B = 2 / 3) (dot_product_88 : a * b * (Real.cos C) = 88)
  (a_val : a = 12) (b_val : b = 9) (c_val : c = 7) :
  a + b + c = 28 :=
sorry

end cos_C_value_triangle_perimeter_l147_147500


namespace total_dogs_at_center_l147_147758

structure PawsitiveTrainingCenter :=
  (sit : Nat)
  (stay : Nat)
  (fetch : Nat)
  (roll_over : Nat)
  (sit_stay : Nat)
  (sit_fetch : Nat)
  (sit_roll_over : Nat)
  (stay_fetch : Nat)
  (stay_roll_over : Nat)
  (fetch_roll_over : Nat)
  (sit_stay_fetch : Nat)
  (sit_stay_roll_over : Nat)
  (sit_fetch_roll_over : Nat)
  (stay_fetch_roll_over : Nat)
  (all_four : Nat)
  (none : Nat)

def PawsitiveTrainingCenter.total_dogs (p : PawsitiveTrainingCenter) : Nat :=
  p.sit + p.stay + p.fetch + p.roll_over
  - p.sit_stay - p.sit_fetch - p.sit_roll_over - p.stay_fetch - p.stay_roll_over - p.fetch_roll_over
  + p.sit_stay_fetch + p.sit_stay_roll_over + p.sit_fetch_roll_over + p.stay_fetch_roll_over
  - p.all_four + p.none

theorem total_dogs_at_center (p : PawsitiveTrainingCenter) (h : 
  p.sit = 60 ∧
  p.stay = 35 ∧
  p.fetch = 45 ∧
  p.roll_over = 40 ∧
  p.sit_stay = 20 ∧
  p.sit_fetch = 15 ∧
  p.sit_roll_over = 10 ∧
  p.stay_fetch = 5 ∧
  p.stay_roll_over = 8 ∧
  p.fetch_roll_over = 6 ∧
  p.sit_stay_fetch = 4 ∧
  p.sit_stay_roll_over = 3 ∧
  p.sit_fetch_roll_over = 2 ∧
  p.stay_fetch_roll_over = 1 ∧
  p.all_four = 2 ∧
  p.none = 12
) : PawsitiveTrainingCenter.total_dogs p = 135 := by
  sorry

end total_dogs_at_center_l147_147758


namespace twelfth_term_arithmetic_sequence_l147_147160

-- Given conditions
def first_term : ℚ := 1 / 4
def common_difference : ℚ := 1 / 2

-- Statement to prove
theorem twelfth_term_arithmetic_sequence :
  (first_term + 11 * common_difference) = 23 / 4 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l147_147160


namespace band_member_share_l147_147073

def num_people : ℕ := 500
def ticket_price : ℝ := 30
def band_share_percent : ℝ := 0.70
def num_band_members : ℕ := 4

theorem band_member_share : 
  (num_people * ticket_price * band_share_percent) / num_band_members = 2625 := by
  sorry

end band_member_share_l147_147073


namespace inequality_proof_l147_147399

theorem inequality_proof 
  (a b c : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c > 0) :
  (a^2 - b^2) / c + (c^2 - b^2) / a + (a^2 - c^2) / b ≥ 3 * a - 4 * b + c :=
  sorry

end inequality_proof_l147_147399


namespace field_trip_savings_l147_147329

-- Define the parameters given in the conditions
def num_students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_per_month : ℕ := 4
def num_months : ℕ := 2

-- Define the weekly savings for the class
def weekly_savings : ℕ := num_students * contribution_per_student_per_week

-- Define the total weeks in the given number of months
def total_weeks : ℕ := num_months * weeks_per_month

-- Define the total savings in the given number of months
def total_savings : ℕ := weekly_savings * total_weeks

-- Now, we state the theorem
theorem field_trip_savings : total_savings = 480 :=
by {
  -- calculations are skipped
  sorry
}

end field_trip_savings_l147_147329


namespace cost_of_toast_l147_147757

theorem cost_of_toast (egg_cost : ℕ) (toast_cost : ℕ)
  (dale_toasts : ℕ) (dale_eggs : ℕ)
  (andrew_toasts : ℕ) (andrew_eggs : ℕ)
  (total_cost : ℕ)
  (h1 : egg_cost = 3)
  (h2 : dale_toasts = 2)
  (h3 : dale_eggs = 2)
  (h4 : andrew_toasts = 1)
  (h5 : andrew_eggs = 2)
  (h6 : 2 * toast_cost + dale_eggs * egg_cost 
        + andrew_toasts * toast_cost + andrew_eggs * egg_cost = total_cost) :
  total_cost = 15 → toast_cost = 1 :=
by
  -- Proof not needed
  sorry

end cost_of_toast_l147_147757


namespace infinite_product_equals_nine_l147_147450

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, ite (n = 0) 1 (3^(n * (1 / 2^n)))

theorem infinite_product_equals_nine : infinite_product = 9 := sorry

end infinite_product_equals_nine_l147_147450


namespace sqrt_multiplication_l147_147826

theorem sqrt_multiplication :
  (Real.sqrt 8 - Real.sqrt 2) * (Real.sqrt 7 - Real.sqrt 3) = Real.sqrt 14 - Real.sqrt 6 :=
by
  -- statement follows
  sorry

end sqrt_multiplication_l147_147826


namespace lily_jog_time_l147_147959

theorem lily_jog_time :
  (∃ (max_time : ℕ) (lily_miles_max : ℕ) (max_distance : ℕ) (lily_time_ratio : ℕ) (distance_wanted : ℕ)
      (expected_time : ℕ),
    max_time = 36 ∧
    lily_miles_max = 4 ∧
    max_distance = 6 ∧
    lily_time_ratio = 3 ∧
    distance_wanted = 7 ∧
    expected_time = 21 ∧
    lily_miles_max * lily_time_ratio = max_time ∧
    max_distance * lily_time_ratio = distance_wanted * expected_time) := 
sorry

end lily_jog_time_l147_147959


namespace victoria_donuts_cost_l147_147300

theorem victoria_donuts_cost (n : ℕ) (cost_per_dozen : ℝ) (total_donuts_needed : ℕ) 
  (dozens_needed : ℕ) (actual_total_donuts : ℕ) (total_cost : ℝ) :
  total_donuts_needed ≥ 550 ∧ cost_per_dozen = 7.49 ∧ (total_donuts_needed = 12 * dozens_needed) ∧
  (dozens_needed = Nat.ceil (total_donuts_needed / 12)) ∧ 
  (actual_total_donuts = 12 * dozens_needed) ∧ actual_total_donuts ≥ 550 ∧ 
  (total_cost = dozens_needed * cost_per_dozen) →
  total_cost = 344.54 :=
by
  sorry

end victoria_donuts_cost_l147_147300


namespace problem1_problem2_l147_147453

-- Problem 1: Prove that (a/(a - b)) + (b/(b - a)) = 1
theorem problem1 (a b : ℝ) (h : a ≠ b) : (a / (a - b)) + (b / (b - a)) = 1 := 
sorry

-- Problem 2: Prove that (a^2 / (b^2 * c)) * (- (b * c^2) / (2 * a)) / (a / b) = -c
theorem problem2 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a^2 / (b^2 * c)) * (- (b * c^2) / (2 * a)) / (a / b) = -c :=
sorry

end problem1_problem2_l147_147453


namespace simplified_fraction_of_num_l147_147306

def num : ℚ := 368 / 100

theorem simplified_fraction_of_num : num = 92 / 25 := by
  sorry

end simplified_fraction_of_num_l147_147306


namespace num_factors_of_M_l147_147458

theorem num_factors_of_M (M : ℕ) 
  (hM : M = (2^5) * (3^4) * (5^3) * (11^2)) : ∃ n : ℕ, n = 360 ∧ M = (2^5) * (3^4) * (5^3) * (11^2) := 
by
  sorry

end num_factors_of_M_l147_147458


namespace steve_travel_time_l147_147143

theorem steve_travel_time :
  ∀ (d : ℕ) (v_back : ℕ) (v_to : ℕ),
  d = 20 →
  v_back = 10 →
  v_to = v_back / 2 →
  d / v_to + d / v_back = 6 := 
by
  intros d v_back v_to h1 h2 h3
  sorry

end steve_travel_time_l147_147143


namespace calculate_expression_l147_147053

theorem calculate_expression : 56.8 * 35.7 + 56.8 * 28.5 + 64.2 * 43.2 = 6420 := 
by sorry

end calculate_expression_l147_147053


namespace sum_of_solutions_eq_16_l147_147620

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l147_147620


namespace negation_proof_l147_147273

theorem negation_proof :
  ¬ (∀ x : ℝ, 0 < x ∧ x < (π / 2) → x > Real.sin x) ↔ 
  ∃ x : ℝ, 0 < x ∧ x < (π / 2) ∧ x ≤ Real.sin x := 
sorry

end negation_proof_l147_147273


namespace male_contestants_l147_147749

theorem male_contestants (total_contestants : ℕ) (female_proportion : ℕ) (total_females : ℕ) :
  female_proportion = 3 ∧ total_contestants = 18 ∧ total_females = total_contestants / female_proportion →
  (total_contestants - total_females) = 12 :=
by
  sorry

end male_contestants_l147_147749


namespace intersection_A_B_l147_147219

def set_A : Set ℝ := { x | abs (x - 1) < 2 }
def set_B : Set ℝ := { x | Real.log x / Real.log 2 > Real.log x / Real.log 3 }

theorem intersection_A_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l147_147219


namespace most_likely_number_of_red_balls_l147_147992

-- Define the conditions
def total_balls : ℕ := 20
def red_ball_frequency : ℝ := 0.8

-- Define the statement we want to prove
theorem most_likely_number_of_red_balls : red_ball_frequency * total_balls = 16 :=
by sorry

end most_likely_number_of_red_balls_l147_147992


namespace max_value_of_f_l147_147063

def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x

theorem max_value_of_f : ∃ m, (∀ x, f(x) ≤ m) ∧ m = 5 :=
by
  sorry

end max_value_of_f_l147_147063


namespace solve_equations_l147_147529

theorem solve_equations :
  (∃ x1 x2 : ℝ, (x1 = 1 ∧ x2 = 3) ∧ (x1^2 - 4 * x1 + 3 = 0) ∧ (x2^2 - 4 * x2 + 3 = 0)) ∧
  (∃ y1 y2 : ℝ, (y1 = 9 ∧ y2 = 11 / 7) ∧ (4 * (2 * y1 - 5)^2 = (3 * y1 - 1)^2) ∧ (4 * (2 * y2 - 5)^2 = (3 * y2 - 1)^2)) :=
by
  sorry

end solve_equations_l147_147529


namespace value_of_expression_l147_147716

theorem value_of_expression :
  4 * 5 + 5 * 4 = 40 :=
sorry

end value_of_expression_l147_147716


namespace election_required_percentage_l147_147097

def votes_cast : ℕ := 10000

def geoff_percentage : ℕ := 5
def geoff_received_votes := (geoff_percentage * votes_cast) / 1000

def extra_votes_needed : ℕ := 5000
def total_votes_needed := geoff_received_votes + extra_votes_needed

def required_percentage := (total_votes_needed * 100) / votes_cast

theorem election_required_percentage : required_percentage = 505 / 10 :=
by
  sorry

end election_required_percentage_l147_147097


namespace Brittany_second_test_grade_is_83_l147_147052

theorem Brittany_second_test_grade_is_83
  (first_test_score : ℝ) (first_test_weight : ℝ) 
  (second_test_weight : ℝ) (final_weighted_average : ℝ) : 
  first_test_score = 78 → 
  first_test_weight = 0.40 →
  second_test_weight = 0.60 →
  final_weighted_average = 81 →
  ∃ G : ℝ, 0.40 * first_test_score + 0.60 * G = final_weighted_average ∧ G = 83 :=
by
  sorry

end Brittany_second_test_grade_is_83_l147_147052


namespace solve_equation_l147_147319

theorem solve_equation : ∃! x : ℕ, 3^x = x + 2 := by
  sorry

end solve_equation_l147_147319


namespace sphere_radius_l147_147745

theorem sphere_radius (tree_height sphere_shadow tree_shadow : ℝ) 
  (h_tree_shadow_pos : tree_shadow > 0) 
  (h_sphere_shadow_pos : sphere_shadow > 0) 
  (h_tree_height_pos : tree_height > 0)
  (h_tangent : (tree_height / tree_shadow) = (sphere_shadow / 15)) : 
  sphere_shadow = 11.25 :=
by
  sorry

end sphere_radius_l147_147745


namespace minimum_value_of_f_l147_147925

noncomputable def f (x : ℝ) : ℝ := x + 4 / (x - 1)

theorem minimum_value_of_f (x : ℝ) (hx : x > 1) : (∃ y : ℝ, f x = 5 ∧ ∀ y > 1, f y ≥ 5) :=
sorry

end minimum_value_of_f_l147_147925


namespace molecular_weight_of_compound_l147_147561

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00
def num_C : ℕ := 4
def num_H : ℕ := 1
def num_O : ℕ := 1

theorem molecular_weight_of_compound : 
  (num_C * atomic_weight_C + num_H * atomic_weight_H + num_O * atomic_weight_O) = 65.048 := 
  by 
  -- proof skipped
  sorry

end molecular_weight_of_compound_l147_147561


namespace sum_of_roots_eq_seventeen_l147_147617

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l147_147617


namespace original_fish_count_l147_147511

def initial_fish_count (fish_taken_out : ℕ) (current_fish : ℕ) : ℕ :=
  fish_taken_out + current_fish

theorem original_fish_count :
  initial_fish_count 16 3 = 19 :=
by
  sorry

end original_fish_count_l147_147511


namespace solve_inequality_l147_147971

open Set

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - (a + 1) * x + 1 < 0

-- Define the solution sets for different cases of a
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 1}
  else if a < 0 then {x | x < 1 / a ∨ x > 1}
  else if 0 < a ∧ a < 1 then {x | 1 < x ∧ x < 1 / a}
  else if a > 1 then {x | 1 / a < x ∧ x < 1}
  else ∅

-- State the theorem
theorem solve_inequality (a : ℝ) : 
  {x : ℝ | inequality a x} = solution_set a :=
by
  sorry

end solve_inequality_l147_147971


namespace average_goals_l147_147893

theorem average_goals (c s j : ℕ) (h1 : c = 4) (h2 : s = c / 2) (h3 : j = 2 * s - 3) :
  c + s + j = 7 :=
sorry

end average_goals_l147_147893


namespace simplify_expr_l147_147132

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l147_147132


namespace sum_of_solutions_l147_147633

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l147_147633


namespace total_expense_l147_147129

noncomputable def sandys_current_age : ℕ := 36 - 2
noncomputable def sandys_monthly_expense : ℕ := 10 * sandys_current_age
noncomputable def alexs_current_age : ℕ := sandys_current_age / 2
noncomputable def alexs_next_month_expense : ℕ := 2 * sandys_monthly_expense

theorem total_expense : 
  sandys_monthly_expense + alexs_next_month_expense = 1020 := 
by 
  sorry

end total_expense_l147_147129


namespace Sandy_fingernails_reach_world_record_in_20_years_l147_147968

-- Definitions for the conditions of the problem
def world_record_len : ℝ := 26
def current_len : ℝ := 2
def growth_rate : ℝ := 0.1

-- Proof goal
theorem Sandy_fingernails_reach_world_record_in_20_years :
  (world_record_len - current_len) / growth_rate / 12 = 20 :=
by
  sorry

end Sandy_fingernails_reach_world_record_in_20_years_l147_147968


namespace find_angle_ACB_l147_147106

-- Definitions corresponding to the conditions
def angleABD : ℝ := 145
def angleBAC : ℝ := 105
def supplementary (a b : ℝ) : Prop := a + b = 180
def triangleAngleSum (a b c : ℝ) : Prop := a + b + c = 180

theorem find_angle_ACB :
  ∃ (angleACB : ℝ), 
    supplementary angleABD angleABC ∧
    triangleAngleSum angleBAC angleABC angleACB ∧
    angleACB = 40 := 
sorry

end find_angle_ACB_l147_147106


namespace pages_torn_l147_147285

theorem pages_torn (n : ℕ) (H1 : n = 185) (H2 : ∃ m, m = 518 ∧ (digits 10 m = digits 10 n) ∧ (m % 2 = 0)) : 
  ∃ k, k = ((518 - 185 + 1) / 2) ∧ k = 167 :=
by sorry

end pages_torn_l147_147285


namespace find_principal_l147_147867

theorem find_principal (r t1 t2 ΔI : ℝ) (h_r : r = 0.15) (h_t1 : t1 = 3.5) (h_t2 : t2 = 5) (h_ΔI : ΔI = 144) :
  ∃ P : ℝ, P = 640 :=
by
  sorry

end find_principal_l147_147867


namespace vacuum_pump_operations_l147_147751

theorem vacuum_pump_operations (n : ℕ) (h : n ≥ 10) : 
  ∀ a : ℝ, 
  a > 0 → 
  (0.5 ^ n) * a < 0.001 * a :=
by
  intros a h_a
  sorry

end vacuum_pump_operations_l147_147751


namespace abs_expression_eq_6500_l147_147253

def given_expression (x : ℝ) : ℝ := 
  abs (abs x - x - abs x + 500) - x

theorem abs_expression_eq_6500 (x : ℝ) (h : x = -3000) : given_expression x = 6500 := by
  sorry

end abs_expression_eq_6500_l147_147253


namespace find_width_of_lawn_l147_147039

noncomputable def width_of_lawn
    (length : ℕ)
    (cost : ℕ)
    (cost_per_sq_m : ℕ)
    (road_width : ℕ) : ℕ :=
  let total_area := cost / cost_per_sq_m
  let road_area_length := road_width * length
  let eq_area := total_area - road_area_length
  eq_area / road_width

theorem find_width_of_lawn :
  width_of_lawn 110 4800 3 10 = 50 :=
by
  sorry

end find_width_of_lawn_l147_147039


namespace longest_side_of_triangle_l147_147076

theorem longest_side_of_triangle (y : ℝ) 
  (side1 : ℝ := 8) (side2 : ℝ := y + 5) (side3 : ℝ := 3 * y + 2)
  (h_perimeter : side1 + side2 + side3 = 47) :
  max side1 (max side2 side3) = 26 :=
sorry

end longest_side_of_triangle_l147_147076


namespace minimum_days_to_pay_back_l147_147512

theorem minimum_days_to_pay_back (x : ℕ) : 
  (50 + 5 * x ≥ 150) → x = 20 :=
sorry

end minimum_days_to_pay_back_l147_147512


namespace rectangle_square_division_l147_147884

theorem rectangle_square_division (n : ℕ) 
  (a b c d : ℕ) 
  (h1 : a * b = n) 
  (h2 : c * d = n + 76)
  (h3 : ∃ u v : ℕ, gcd a c = u ∧ gcd b d = v ∧ u * v * a^2 = u * v * c^2 ∧ u * v * b^2 = u * v * d^2) : 
  n = 324 := sorry

end rectangle_square_division_l147_147884


namespace total_yards_in_marathons_eq_495_l147_147177

-- Definitions based on problem conditions
def marathon_miles : ℕ := 26
def marathon_yards : ℕ := 385
def yards_in_mile : ℕ := 1760
def marathons_run : ℕ := 15

-- Main proof statement
theorem total_yards_in_marathons_eq_495
  (miles_per_marathon : ℕ := marathon_miles)
  (yards_per_marathon : ℕ := marathon_yards)
  (yards_per_mile : ℕ := yards_in_mile)
  (marathons : ℕ := marathons_run) :
  let total_yards := marathons * yards_per_marathon
  let remaining_yards := total_yards % yards_per_mile
  remaining_yards = 495 :=
by
  sorry

end total_yards_in_marathons_eq_495_l147_147177


namespace number_of_six_digit_palindromes_l147_147384

def is_six_digit_palindrome (n : ℕ) : Prop := 
  100000 ≤ n ∧ n ≤ 999999 ∧ (∀ a b c : ℕ, 
    n = 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a → a ≠ 0)

theorem number_of_six_digit_palindromes : 
  ∃ (count : ℕ), (count = 900 ∧ 
  ∀ n : ℕ, is_six_digit_palindrome n → true) 
:= 
by 
  use 900 
  sorry

end number_of_six_digit_palindromes_l147_147384


namespace distance_from_integer_l147_147956

theorem distance_from_integer (a : ℝ) (h : a > 0) (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, ∃ m : ℕ, 1 ≤ m ∧ m < n ∧ abs (m * a - k) ≤ (1 / n) :=
by
  sorry

end distance_from_integer_l147_147956


namespace minValue_l147_147823

noncomputable def minValueOfExpression (a b c : ℝ) : ℝ :=
  (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a))

theorem minValue (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 2 * a + 2 * b + 2 * c = 3) : 
  minValueOfExpression a b c = 2 :=
  sorry

end minValue_l147_147823


namespace bicycle_cost_correct_l147_147188

def pay_rate : ℕ := 5
def hours_p_week : ℕ := 2 + 1 + 3
def weeks : ℕ := 6
def bicycle_cost : ℕ := 180

theorem bicycle_cost_correct :
  pay_rate * hours_p_week * weeks = bicycle_cost :=
by
  sorry

end bicycle_cost_correct_l147_147188


namespace exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012_l147_147459

theorem exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012 :
  ∃ (a b c : ℕ), 
    a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧ 
    a ∣ (a * b * c + 2012) ∧ b ∣ (a * b * c + 2012) ∧ c ∣ (a * b * c + 2012) :=
by
  sorry

end exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012_l147_147459


namespace inverse_of_g_at_1_over_32_l147_147924

noncomputable def g (x : ℝ) : ℝ := (x^5 + 2) / 4

theorem inverse_of_g_at_1_over_32 :
  g⁻¹ (1/32) = (-15 / 8)^(1/5) :=
sorry

end inverse_of_g_at_1_over_32_l147_147924


namespace calf_probability_l147_147890

theorem calf_probability 
  (P_B1 : ℝ := 0.6)  -- Proportion of calves from the first farm
  (P_B2 : ℝ := 0.3)  -- Proportion of calves from the second farm
  (P_B3 : ℝ := 0.1)  -- Proportion of calves from the third farm
  (P_B1_A : ℝ := 0.15)  -- Conditional probability of a calf weighing more than 300 kg given it is from the first farm
  (P_B2_A : ℝ := 0.25)  -- Conditional probability of a calf weighing more than 300 kg given it is from the second farm
  (P_B3_A : ℝ := 0.35)  -- Conditional probability of a calf weighing more than 300 kg given it is from the third farm)
  (P_A : ℝ := P_B1 * P_B1_A + P_B2 * P_B2_A + P_B3 * P_B3_A) : 
  P_B3 * P_B3_A / P_A = 0.175 := 
by
  sorry

end calf_probability_l147_147890


namespace problem_statement_l147_147908

theorem problem_statement (x y : ℝ) (h₁ : |x| = 3) (h₂ : |y| = 4) (h₃ : x > y) : 2 * x - y = 10 := 
by {
  sorry
}

end problem_statement_l147_147908


namespace evaluate_expression_l147_147355

theorem evaluate_expression : (10^9) / ((2 * 10^6) * 3) = 500 / 3 :=
by sorry

end evaluate_expression_l147_147355


namespace simplify_expression_l147_147969

theorem simplify_expression :
  2 * Real.sqrt (1 + Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4 - 4 * Real.cos 4 :=
by
  sorry

end simplify_expression_l147_147969


namespace option_B_correct_l147_147426

theorem option_B_correct (a b : ℝ) (h : a < b) : a^3 < b^3 := sorry

end option_B_correct_l147_147426


namespace number_of_ensembles_sold_l147_147123

-- Define the prices
def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earring_price : ℕ := 10
def ensemble_price : ℕ := 45

-- Define the quantities sold
def necklaces_sold : ℕ := 5
def bracelets_sold : ℕ := 10
def earrings_sold : ℕ := 20

-- Define the total income
def total_income : ℕ := 565

-- Define the function or theorem that determines the number of ensembles sold
theorem number_of_ensembles_sold : 
  (total_income = (necklaces_sold * necklace_price) + (bracelets_sold * bracelet_price) + (earrings_sold * earring_price) + (2 * ensemble_price)) :=
sorry

end number_of_ensembles_sold_l147_147123


namespace sum_of_solutions_equation_l147_147612

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l147_147612


namespace perp_vec_m_l147_147652

theorem perp_vec_m (m : ℝ) : (1 : ℝ) * (-1 : ℝ) + 2 * m = 0 → m = 1 / 2 :=
by 
  intro h
  -- Translate the given condition directly
  sorry

end perp_vec_m_l147_147652


namespace log_relation_l147_147683

noncomputable def a := Real.log 3 / Real.log 4
noncomputable def b := Real.log 3 / Real.log 0.4
def c := (1 / 2) ^ 2

theorem log_relation (h1 : a = Real.log 3 / Real.log 4)
                     (h2 : b = Real.log 3 / Real.log 0.4)
                     (h3 : c = (1 / 2) ^ 2) : a > c ∧ c > b :=
by
  sorry

end log_relation_l147_147683


namespace total_pokemon_cards_l147_147961

-- Definitions based on conditions
def dozen := 12
def amount_per_person := 9 * dozen
def num_people := 4

-- Proposition to prove
theorem total_pokemon_cards :
  num_people * amount_per_person = 432 :=
by sorry

end total_pokemon_cards_l147_147961


namespace problem_divisibility_l147_147964

theorem problem_divisibility (n : ℕ) : ∃ k : ℕ, 2 ^ (3 ^ n) + 1 = 3 ^ (n + 1) * k :=
sorry

end problem_divisibility_l147_147964


namespace solve_quadratic_roots_l147_147288

theorem solve_quadratic_roots (x : ℝ) : (x - 3) ^ 2 = 3 - x ↔ x = 3 ∨ x = 2 :=
by
  sorry

end solve_quadratic_roots_l147_147288


namespace magic_8_ball_probability_l147_147247

def probability_positive (p_pos : ℚ) (questions : ℕ) (positive_responses : ℕ) : ℚ :=
  (Nat.choose questions positive_responses : ℚ) * (p_pos ^ positive_responses) * ((1 - p_pos) ^ (questions - positive_responses))

theorem magic_8_ball_probability :
  probability_positive (1/3) 7 3 = 560 / 2187 :=
by
  sorry

end magic_8_ball_probability_l147_147247


namespace part1_part2_l147_147084

def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

theorem part1 (a : ℝ) :
  (Set.Icc (a - 7) (a - 3)) = (Set.Icc (-5 : ℝ) (-1 : ℝ)) -> a = 2 :=
by
  intro h
  sorry

theorem part2 (m : ℝ) :
  (∃ x_0 : ℝ, f x_0 2 < 4 * m + m^2) -> (m < -5 ∨ m > 1) :=
by
  intro h
  sorry

end part1_part2_l147_147084


namespace ratio_of_fractions_l147_147865

theorem ratio_of_fractions (x y : ℝ) (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) : 
  (1 / 5 * x) / (1 / 6 * y) = 0.72 :=
sorry

end ratio_of_fractions_l147_147865


namespace product_of_roots_l147_147902

theorem product_of_roots :
  let a := 18
  let b := 45
  let c := -500
  let prod_roots := c / a
  prod_roots = -250 / 9 := 
by
  -- Define coefficients
  let a := 18
  let c := -500

  -- Calculate product of roots
  let prod_roots := c / a

  -- Statement to prove
  have : prod_roots = -250 / 9 := sorry
  exact this

-- Adding sorry since the proof is not required according to the problem statement.

end product_of_roots_l147_147902


namespace largest_integer_base8_square_l147_147954

theorem largest_integer_base8_square :
  ∃ (N : ℕ), (N^2 >= 8^3) ∧ (N^2 < 8^4) ∧ (N = 63 ∧ N % 8 = 7) := sorry

end largest_integer_base8_square_l147_147954


namespace B_contribution_l147_147863

-- Define the conditions
def capitalA : ℝ := 3500
def monthsA : ℕ := 12
def monthsB : ℕ := 7
def profit_ratio_A : ℕ := 2
def profit_ratio_B : ℕ := 3

-- Statement: B's contribution to the capital
theorem B_contribution :
  (capitalA * monthsA * profit_ratio_B) / (monthsB * profit_ratio_A) = 4500 := by
  sorry

end B_contribution_l147_147863


namespace mitzi_money_left_l147_147695

theorem mitzi_money_left :
  let A := 75
  let T := 30
  let F := 13
  let S := 23
  let total_spent := T + F + S
  let money_left := A - total_spent
  money_left = 9 :=
by
  sorry

end mitzi_money_left_l147_147695


namespace right_triangle_possible_third_side_l147_147098

theorem right_triangle_possible_third_side (a b : ℕ) (h : a = 5 ∧ b = 12 ∨ a = 12 ∧ b = 5) :
  ∃ c : ℝ, (c = sqrt (a^2 + b^2) ∨ c = sqrt (b^2 - a^2)) :=
by {
  sorry
}

end right_triangle_possible_third_side_l147_147098


namespace crayons_given_to_friends_l147_147838

def initial_crayons : ℕ := 440
def lost_crayons : ℕ := 106
def remaining_crayons : ℕ := 223

theorem crayons_given_to_friends :
  initial_crayons - remaining_crayons - lost_crayons = 111 := 
by
  sorry

end crayons_given_to_friends_l147_147838


namespace largest_a_l147_147805

theorem largest_a (a b : ℕ) (x : ℕ) (h_a_range : 2 < a ∧ a < x) (h_b_range : 4 < b ∧ b < 13) (h_fraction_range : 7 * a = 57) : a = 8 :=
sorry

end largest_a_l147_147805


namespace isosceles_triangle_perimeter_l147_147238

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 6) (h₂ : b = 3) (h₃ : a > b) : a + a + b = 15 :=
by
  sorry

end isosceles_triangle_perimeter_l147_147238


namespace number_of_children_l147_147703

theorem number_of_children (x : ℕ) : 3 * x + 12 = 5 * x - 10 → x = 11 :=
by
  intros h
  have : 3 * x + 12 = 5 * x - 10 := h
  sorry

end number_of_children_l147_147703


namespace theatre_fraction_l147_147937

noncomputable def fraction_theatre_took_elective_last_year (T P Th M : ℕ) : Prop :=
  (P = 1 / 2 * T) ∧
  (Th + M = T - P) ∧
  (1 / 3 * P + M = 2 / 3 * T) ∧
  (Th = 1 / 6 * T)

theorem theatre_fraction (T P Th M : ℕ) :
  fraction_theatre_took_elective_last_year T P Th M →
  Th / T = 1 / 6 :=
by
  intro h
  cases h
  sorry

end theatre_fraction_l147_147937


namespace subset_A_has_only_one_element_l147_147292

theorem subset_A_has_only_one_element (m : ℝ) :
  (∀ x y, (mx^2 + 2*x + 1 = 0) → (mx*y^2 + 2*y + 1 = 0) → x = y) →
  (m = 0 ∨ m = 1) :=
by
  sorry

end subset_A_has_only_one_element_l147_147292


namespace calculate_three_Z_five_l147_147765

def Z (a b : ℤ) : ℤ := b + 15 * a - a^3

theorem calculate_three_Z_five : Z 3 5 = 23 :=
by
  -- The proof goes here
  sorry

end calculate_three_Z_five_l147_147765


namespace min_value_is_144_l147_147958

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  x^2 + 4 * x * y + 4 * y^2 + 3 * z^2

theorem min_value_is_144 (x y z : ℝ) (hxyz : x * y * z = 48) : 
  ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ xyz = 48 ∧ min_value_expression x y z = 144 :=
by 
  sorry

end min_value_is_144_l147_147958


namespace connectivity_within_square_l147_147871

theorem connectivity_within_square (side_length : ℝ) (highway1 highway2 : ℝ) 
  (A1 A2 A3 A4 : ℝ → ℝ → Prop) : 
  side_length = 10 → 
  highway1 ≠ highway2 → 
  (∀ x y, (0 ≤ x ∧ x ≤ side_length ∧ 0 ≤ y ∧ y ≤ side_length) → 
    (A1 x y ∨ A2 x y ∨ A3 x y ∨ A4 x y)) →
  ∃ (road_length : ℝ), road_length ≤ 25 := 
sorry

end connectivity_within_square_l147_147871


namespace incorrect_statement_l147_147242

def population : ℕ := 13000
def sample_size : ℕ := 500
def academic_performance (n : ℕ) : Type := sorry

def statement_A (ap : Type) : Prop := 
  ap = academic_performance population

def statement_B (ap : Type) : Prop := 
  ∀ (u : ℕ), u ≤ population → ap = academic_performance 1

def statement_C (ap : Type) : Prop := 
  ap = academic_performance sample_size

def statement_D : Prop := 
  sample_size = 500

theorem incorrect_statement : ¬ (statement_B (academic_performance 1)) :=
sorry

end incorrect_statement_l147_147242


namespace car_fewer_minutes_than_bus_l147_147186

-- Conditions translated into Lean definitions
def bus_time_to_beach : ℕ := 40
def car_round_trip_time : ℕ := 70

-- Derived condition
def car_one_way_time : ℕ := car_round_trip_time / 2

-- Theorem statement to be proven
theorem car_fewer_minutes_than_bus : car_one_way_time = bus_time_to_beach - 5 := by
  -- This is the placeholder for the proof
  sorry

end car_fewer_minutes_than_bus_l147_147186


namespace average_growth_rate_l147_147714

theorem average_growth_rate (x : ℝ) (hx : (1 + x)^2 = 1.44) : x < 0.22 :=
sorry

end average_growth_rate_l147_147714


namespace no_daily_coverage_l147_147100

theorem no_daily_coverage (ranks : Nat → Nat)
  (h_ranks_ordered : ∀ i, ranks (i+1) ≥ 3 * ranks i)
  (h_cycle : ∀ i, ∃ N : Nat, ranks i = N ∧ ∃ k : Nat, k = N ∧ ∀ m, m % (2 * N) < N → (¬ ∃ j, ranks j ≤ N))
  : ¬ (∀ d : Nat, ∃ j : Nat, (∃ k : Nat, d % (2 * (ranks j)) < ranks j))
  := sorry

end no_daily_coverage_l147_147100


namespace shaded_area_possible_values_l147_147514

variable (AB BC PQ SC : ℕ)

-- Conditions:
def dimensions_correct : Prop := AB * BC = 33 ∧ AB < 7 ∧ BC < 7
def length_constraint : Prop := PQ < SC

-- Theorem statement
theorem shaded_area_possible_values (h1 : dimensions_correct AB BC) (h2 : length_constraint PQ SC) :
  (AB = 3 ∧ BC = 11 ∧ (PQ = 1 ∧ SC = 6 ∧ (33 - 1 * 4 - 2 * 6 = 17) ∨
                      (33 - 2 * 3 - 1 * 6 = 21) ∨
                      (33 - 2 * 4 - 1 * 5 = 20))) ∨ 
  (AB = 11 ∧ BC = 3 ∧ (PQ = 1 ∧ SC = 6 ∧ (33 - 1 * 4 - 2 * 6 = 17))) :=
sorry

end shaded_area_possible_values_l147_147514


namespace area_of_intersection_l147_147299

-- Define the circle centered at (3, 0) with radius 3
def circle1 (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 9

-- Define the circle centered at (0, 3) with radius 3
def circle2 (x y : ℝ) : Prop := x ^ 2 + (y - 3) ^ 2 = 9

-- Defining the theorem to prove the area of intersection of these circles
theorem area_of_intersection : 
  let r := 3 in
  let a := (3, 0) in
  let b := (0, 3) in
  area_intersection (circle1) (circle2) = (9 * π - 18) / 2 := 
sorry

end area_of_intersection_l147_147299


namespace Kimberley_collected_10_pounds_l147_147502

variable (K H E total : ℝ)

theorem Kimberley_collected_10_pounds (h_total : total = 35) (h_Houston : H = 12) (h_Ela : E = 13) :
    K + H + E = total → K = 10 :=
by
  intros h_sum
  rw [h_Houston, h_Ela] at h_sum
  linarith

end Kimberley_collected_10_pounds_l147_147502


namespace vector_subtraction_correct_l147_147792

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (-4, 2)

theorem vector_subtraction_correct :
  vector_a - 2 • vector_b = (10, -5) :=
sorry

end vector_subtraction_correct_l147_147792


namespace num_new_students_l147_147267

-- Definitions based on the provided conditions
def original_class_strength : ℕ := 10
def original_average_age : ℕ := 40
def new_students_avg_age : ℕ := 32
def decrease_in_average_age : ℕ := 4
def new_average_age : ℕ := original_average_age - decrease_in_average_age
def new_class_strength (n : ℕ) : ℕ := original_class_strength + n

-- The proof statement
theorem num_new_students (n : ℕ) :
  (original_class_strength * original_average_age + n * new_students_avg_age) 
  = new_class_strength n * new_average_age → n = 10 :=
by
  sorry

end num_new_students_l147_147267


namespace initial_ratio_l147_147832

-- Define the initial number of horses and cows
def initial_horses (H : ℕ) : Prop := H = 120
def initial_cows (C : ℕ) : Prop := C = 20

-- Define the conditions of the problem
def condition1 (H C : ℕ) : Prop := H - 15 = 3 * (C + 15)
def condition2 (H C : ℕ) : Prop := H - 15 = C + 15 + 70

-- The statement that initial ratio is 6:1
theorem initial_ratio (H C : ℕ) (h1 : condition1 H C) (h2 : condition2 H C) : 
  H = 6 * C :=
by {
  sorry
}

end initial_ratio_l147_147832


namespace average_minutes_correct_l147_147495

variable (s : ℕ)
def sixth_graders := 3 * s
def seventh_graders := s
def eighth_graders := s / 2

def minutes_sixth_graders := 18 * sixth_graders s
def minutes_seventh_graders := 20 * seventh_graders s
def minutes_eighth_graders := 22 * eighth_graders s

def total_minutes := minutes_sixth_graders s + minutes_seventh_graders s + minutes_eighth_graders s
def total_students := sixth_graders s + seventh_graders s + eighth_graders s

def average_minutes := total_minutes s / total_students s

theorem average_minutes_correct : average_minutes s = 170 / 9 := sorry

end average_minutes_correct_l147_147495


namespace ceil_minus_floor_eq_one_imp_ceil_minus_x_l147_147263

variable {x : ℝ}

theorem ceil_minus_floor_eq_one_imp_ceil_minus_x (H : ⌈x⌉ - ⌊x⌋ = 1) : ∃ (n : ℤ) (f : ℝ), (x = n + f) ∧ (0 < f) ∧ (f < 1) ∧ (⌈x⌉ - x = 1 - f) := sorry

end ceil_minus_floor_eq_one_imp_ceil_minus_x_l147_147263


namespace number_of_six_digit_palindromes_l147_147383

def is_six_digit_palindrome (n : ℕ) : Prop := 
  100000 ≤ n ∧ n ≤ 999999 ∧ (∀ a b c : ℕ, 
    n = 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a → a ≠ 0)

theorem number_of_six_digit_palindromes : 
  ∃ (count : ℕ), (count = 900 ∧ 
  ∀ n : ℕ, is_six_digit_palindrome n → true) 
:= 
by 
  use 900 
  sorry

end number_of_six_digit_palindromes_l147_147383


namespace ladder_of_twos_l147_147818

theorem ladder_of_twos (n : ℕ) (h : n ≥ 3) : 
  ∃ N_n : ℕ, N_n = 2 ^ (n - 3) :=
by
  sorry

end ladder_of_twos_l147_147818


namespace distance_squared_between_intersections_l147_147422

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + (y + 2)^2 = 9

-- Hypothesis about points of intersection
def pointC (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y
def pointD (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y ∧ (x, y) ≠ (4, -2)

-- The requirement to prove
theorem distance_squared_between_intersections :
  ∃ (x1 y1 x2 y2 : ℝ),
  pointC x1 y1 ∧ pointD x2 y2 ∧
  ((x1 - x2)^2 + (y1 - y2)^2 = 396.8) :=
sorry

end distance_squared_between_intersections_l147_147422


namespace problem_2_l147_147920

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x ^ 2 + a * Real.log (1 - x)

theorem problem_2 (a : ℝ) (x₁ x₂ : ℝ) (h₀ : 0 < a) (h₁ : a < 1/4) (h₂ : f x₂ a = 0) 
  (h₃ : f x₁ a = 0) (hx₁ : 0 < x₁) (hx₂ : x₁ < 1/2) (h₄ : x₁ < x₂) :
  f x₂ a - x₁ > - (3 + Real.log 4) / 8 := sorry

end problem_2_l147_147920


namespace no_integers_for_sum_of_squares_l147_147753

theorem no_integers_for_sum_of_squares :
  ¬ ∃ a b : ℤ, a^2 + b^2 = 10^100 + 3 :=
by
  sorry

end no_integers_for_sum_of_squares_l147_147753


namespace infinite_solutions_for_equation_l147_147126

theorem infinite_solutions_for_equation :
  ∃ (x y z : ℤ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ ∀ (k : ℤ), (x^2 + y^5 = z^3) :=
sorry

end infinite_solutions_for_equation_l147_147126


namespace final_payment_order_450_l147_147449

noncomputable def finalPayment (orderAmount : ℝ) : ℝ :=
  let serviceCharge := if orderAmount < 500 then 0.04 * orderAmount
                      else if orderAmount < 1000 then 0.05 * orderAmount
                      else 0.06 * orderAmount
  let salesTax := if orderAmount < 500 then 0.05 * orderAmount
                  else if orderAmount < 1000 then 0.06 * orderAmount
                  else 0.07 * orderAmount
  let totalBeforeDiscount := orderAmount + serviceCharge + salesTax
  let discount := if totalBeforeDiscount < 600 then 0.05 * totalBeforeDiscount
                  else if totalBeforeDiscount < 800 then 0.10 * totalBeforeDiscount
                  else 0.15 * totalBeforeDiscount
  totalBeforeDiscount - discount

theorem final_payment_order_450 :
  finalPayment 450 = 465.98 := by
  sorry

end final_payment_order_450_l147_147449


namespace number_of_suits_sold_l147_147353

theorem number_of_suits_sold
  (commission_rate: ℝ)
  (price_per_suit: ℝ)
  (price_per_shirt: ℝ)
  (price_per_loafer: ℝ)
  (number_of_shirts: ℕ)
  (number_of_loafers: ℕ)
  (total_commission: ℝ)
  (suits_sold: ℕ)
  (total_sales: ℝ)
  (total_sales_from_non_suits: ℝ)
  (sales_needed_from_suits: ℝ)
  : 
  (commission_rate = 0.15) → 
  (price_per_suit = 700.0) → 
  (price_per_shirt = 50.0) → 
  (price_per_loafer = 150.0) → 
  (number_of_shirts = 6) → 
  (number_of_loafers = 2) → 
  (total_commission = 300.0) →
  (total_sales = total_commission / commission_rate) →
  (total_sales_from_non_suits = number_of_shirts * price_per_shirt + number_of_loafers * price_per_loafer) →
  (sales_needed_from_suits = total_sales - total_sales_from_non_suits) →
  (suits_sold = sales_needed_from_suits / price_per_suit) →
  suits_sold = 2 :=
by
  sorry

end number_of_suits_sold_l147_147353


namespace gallons_left_l147_147509

theorem gallons_left (initial_gallons : ℚ) (gallons_given : ℚ) (gallons_left : ℚ) : 
  initial_gallons = 4 ∧ gallons_given = 16/3 → gallons_left = -4/3 :=
by
  sorry

end gallons_left_l147_147509


namespace sea_lions_count_l147_147022

theorem sea_lions_count (S P : ℕ) (h1 : 11 * S = 4 * P) (h2 : P = S + 84) : S = 48 := 
by {
  sorry
}

end sea_lions_count_l147_147022


namespace male_contestants_l147_147750

theorem male_contestants (total_contestants : ℕ) (female_proportion : ℕ) (total_females : ℕ) :
  female_proportion = 3 ∧ total_contestants = 18 ∧ total_females = total_contestants / female_proportion →
  (total_contestants - total_females) = 12 :=
by
  sorry

end male_contestants_l147_147750


namespace limit_of_f_at_1_l147_147432

noncomputable def f (x : ℝ) : ℝ := (3 - real.sqrt (10 - x)) / real.sin (3 * real.pi * x)

theorem limit_of_f_at_1 :
  filter.tendsto (λ x : ℝ, f x) (nhds 1) (nhds (-1 / (18 * real.pi))) := sorry

end limit_of_f_at_1_l147_147432


namespace universal_proposition_is_B_l147_147304

theorem universal_proposition_is_B :
  (∀ n : ℤ, (2 * n % 2 = 0)) = True :=
sorry

end universal_proposition_is_B_l147_147304


namespace monotonic_increasing_interval_l147_147348

noncomputable def f (x : ℝ) : ℝ :=
  Real.logb 0.5 (x^2 + 2 * x - 3)

theorem monotonic_increasing_interval :
  ∀ x, f x = Real.logb 0.5 (x^2 + 2 * x - 3) → 
  (∀ x₁ x₂, x₁ < x₂ ∧ x₁ < -3 ∧ x₂ < -3 → f x₁ ≤ f x₂) :=
sorry

end monotonic_increasing_interval_l147_147348


namespace mean_of_set_l147_147858

theorem mean_of_set {m : ℝ} 
  (median_condition : (m + 8 + m + 11) / 2 = 19) : 
  (m + (m + 6) + (m + 8) + (m + 11) + (m + 18) + (m + 20)) / 6 = 20 := 
by 
  sorry

end mean_of_set_l147_147858


namespace exposed_surface_area_l147_147416

theorem exposed_surface_area (r h : ℝ) (π : ℝ) (sphere_surface_area : ℝ) (cylinder_lateral_surface_area : ℝ) 
  (cond1 : r = 10) (cond2 : h = 5) (cond3 : sphere_surface_area = 4 * π * r^2) 
  (cond4 : cylinder_lateral_surface_area = 2 * π * r * h) :
  let hemisphere_curved_surface_area := sphere_surface_area / 2
  let hemisphere_base_area := π * r^2
  let total_surface_area := hemisphere_curved_surface_area + hemisphere_base_area + cylinder_lateral_surface_area
  total_surface_area = 400 * π :=
by
  sorry

end exposed_surface_area_l147_147416


namespace repeating_decimal_to_fraction_l147_147779

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + 0.066666... ) : x = 11 / 30 :=
sorry

end repeating_decimal_to_fraction_l147_147779


namespace find_g_neg1_l147_147916

-- Define that f(x) is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Given conditions
variables (f g : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_eq : ∀ x : ℝ, f x = g x + x^2)
variable (h_g1 : g 1 = 1)

-- The statement to prove
theorem find_g_neg1 : g (-1) = -3 :=
sorry

end find_g_neg1_l147_147916


namespace no_lighter_sentence_for_liar_l147_147852

theorem no_lighter_sentence_for_liar
  (total_eggs : ℕ)
  (stolen_eggs1 stolen_eggs2 stolen_eggs3 : ℕ)
  (different_stolen_eggs : stolen_eggs1 ≠ stolen_eggs2 ∧ stolen_eggs2 ≠ stolen_eggs3 ∧ stolen_eggs1 ≠ stolen_eggs3)
  (stolen_eggs1_max : stolen_eggs1 > stolen_eggs2 ∧ stolen_eggs1 > stolen_eggs3)
  (stole_7 : stolen_eggs1 = 7)
  (total_eq_20 : stolen_eggs1 + stolen_eggs2 + stolen_eggs3 = 20) :
  false :=
by
  sorry

end no_lighter_sentence_for_liar_l147_147852


namespace find_plane_equation_l147_147901

def point := ℝ × ℝ × ℝ

def plane_equation (A B C D : ℝ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

def points := (0, 3, -1) :: (4, 7, 1) :: (2, 5, 0) :: []

def correct_plane_equation : Prop :=
  ∃ A B C D : ℝ, plane_equation A B C D = fun x y z => A * x + B * y + C * z + D = 0 ∧ 
  (A, B, C, D) = (0, 1, -2, -5) ∧ ∀ x y z, (x, y, z) ∈ points → plane_equation A B C D x y z

theorem find_plane_equation : correct_plane_equation :=
sorry

end find_plane_equation_l147_147901


namespace distance_between_cars_after_third_checkpoint_l147_147553

theorem distance_between_cars_after_third_checkpoint
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (speed_after_first : ℝ)
  (speed_after_second : ℝ)
  (speed_after_third : ℝ)
  (distance_travelled : ℝ) :
  initial_distance = 100 →
  initial_speed = 60 →
  speed_after_first = 80 →
  speed_after_second = 100 →
  speed_after_third = 120 →
  distance_travelled = 200 :=
by
  sorry

end distance_between_cars_after_third_checkpoint_l147_147553


namespace band_member_share_l147_147074

def num_people : ℕ := 500
def ticket_price : ℝ := 30
def band_share_percent : ℝ := 0.70
def num_band_members : ℕ := 4

theorem band_member_share : 
  (num_people * ticket_price * band_share_percent) / num_band_members = 2625 := by
  sorry

end band_member_share_l147_147074


namespace locus_of_vertices_locus_of_vertices_l147_147201

-- Definitions based on the given problem
def O : Point := sorry
def H : Point := sorry
def C0 : Point := midpoint (A, B)

-- Given conditions
axiom condition_CH_eq_2OC0 : dist C H = 2 * dist O C0
axiom condition_C0_in_circumcircle : C0 ∈ circumcircle O A B
axiom condition_distance_relationship : dist C H < 2 * dist O C

-- We now state our theorem
theorem locus_of_vertices_locus_of_vertices (O H : Point) (C : Point) : 
  ∃ M : Point, ∃ M' : Point,
  M = midpoint(O, H) ∧
  M' = point_symmetric H O ∧
  ∀ C, C ∉ circle_with_diameter M M' ∧ C ∉ circumference_with_diameter M H ∧ C = H :=
sorry

end locus_of_vertices_locus_of_vertices_l147_147201


namespace problem_abc_l147_147519

theorem problem_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by
  sorry

end problem_abc_l147_147519


namespace probability_of_two_boys_l147_147743

def club_has_15_members := 15
def number_of_boys := 8
def number_of_girls := 7

def total_ways_to_choose_2_members : ℕ :=
  (Nat.choose club_has_15_members 2)

def ways_to_choose_2_boys : ℕ :=
  (Nat.choose number_of_boys 2)

theorem probability_of_two_boys : (ways_to_choose_2_boys : ℚ) / (total_ways_to_choose_2_members : ℚ) = (4 / 15 : ℚ) :=
  by
  sorry

end probability_of_two_boys_l147_147743


namespace area_of_intersection_l147_147298

-- Define the circle centered at (3, 0) with radius 3
def circle1 (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 9

-- Define the circle centered at (0, 3) with radius 3
def circle2 (x y : ℝ) : Prop := x ^ 2 + (y - 3) ^ 2 = 9

-- Defining the theorem to prove the area of intersection of these circles
theorem area_of_intersection : 
  let r := 3 in
  let a := (3, 0) in
  let b := (0, 3) in
  area_intersection (circle1) (circle2) = (9 * π - 18) / 2 := 
sorry

end area_of_intersection_l147_147298


namespace centroid_of_triangle_l147_147151

-- Definitions and conditions
def is_lattice_point (p : ℤ × ℤ) : Prop := 
  true -- Placeholder for a more specific definition if necessary

def triangle (A B C : ℤ × ℤ) : Prop := 
  true -- Placeholder for defining a triangle with vertices at integer grid points

def no_other_nodes_on_sides (A B C : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert no other integer grid points on the sides

def exactly_one_node_inside (A B C O : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert exactly one integer grid point inside the triangle

def medians_intersection_is_point_O (A B C O : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert \(O\) is the intersection point of the medians

-- Theorem statement
theorem centroid_of_triangle 
  (A B C O : ℤ × ℤ)
  (h1 : is_lattice_point A)
  (h2 : is_lattice_point B)
  (h3 : is_lattice_point C)
  (h4 : triangle A B C)
  (h5 : no_other_nodes_on_sides A B C)
  (h6 : exactly_one_node_inside A B C O) : 
  medians_intersection_is_point_O A B C O :=
sorry

end centroid_of_triangle_l147_147151


namespace square_minus_sqrt_l147_147761

-- Variables and conditions
variable {y : ℝ}

-- The theorem to be proven
theorem square_minus_sqrt (y : ℝ) : (7 - real.sqrt (y^2 - 49))^2 = y^2 - 14 * real.sqrt (y^2 - 49) :=
sorry

end square_minus_sqrt_l147_147761


namespace find_k_l147_147361

-- Define the variables and conditions
variables (x y k : ℤ)

-- State the theorem
theorem find_k (h1 : x = 2) (h2 : y = 1) (h3 : k * x - y = 3) : k = 2 :=
sorry

end find_k_l147_147361


namespace train_cross_time_proof_l147_147556

noncomputable def train_cross_time_opposite (L : ℝ) (v1 v2 : ℝ) (t_same : ℝ) : ℝ :=
  let speed_same := (v1 - v2) * (5/18)
  let dist_same := speed_same * t_same
  let speed_opposite := (v1 + v2) * (5/18)
  dist_same / speed_opposite

theorem train_cross_time_proof : 
  train_cross_time_opposite 69.444 50 40 50 = 5.56 :=
by
  sorry

end train_cross_time_proof_l147_147556


namespace division_result_l147_147024

theorem division_result : 3486 / 189 = 18.444444444444443 := 
by sorry

end division_result_l147_147024


namespace max_f_5_value_l147_147477

noncomputable def f (x : ℝ) : ℝ := x ^ 2 + 2 * x

noncomputable def f_1 (x : ℝ) : ℝ := f x
noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0       => x -- Not used, as n starts from 1
  | (n + 1) => f (f_n n x)

noncomputable def max_f_5 : ℝ := 3 ^ 32 - 1

theorem max_f_5_value : ∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f_n 5 x ≤ max_f_5 :=
by
  intro x hx
  have := hx
  -- The detailed proof would go here,
  -- but for the statement, we end with sorry.
  sorry

end max_f_5_value_l147_147477


namespace total_workers_l147_147457

theorem total_workers (h_beavers : ℕ := 318) (h_spiders : ℕ := 544) :
  h_beavers + h_spiders = 862 :=
by
  sorry

end total_workers_l147_147457


namespace inversely_proportional_ratio_l147_147139

theorem inversely_proportional_ratio (x y x1 x2 y1 y2 : ℝ) 
  (h_inv_prop : x * y = x1 * y2) 
  (h_ratio : x1 / x2 = 3 / 5) 
  (x1_nonzero : x1 ≠ 0) 
  (x2_nonzero : x2 ≠ 0) 
  (y1_nonzero : y1 ≠ 0) 
  (y2_nonzero : y2 ≠ 0) : 
  y1 / y2 = 5 / 3 := 
sorry

end inversely_proportional_ratio_l147_147139


namespace tree_growth_rate_l147_147271

-- Given conditions
def currentHeight : ℝ := 52
def futureHeightInches : ℝ := 1104
def oneFootInInches : ℝ := 12
def years : ℝ := 8

-- Prove the yearly growth rate in feet
theorem tree_growth_rate:
  (futureHeightInches / oneFootInInches - currentHeight) / years = 5 := 
by
  sorry

end tree_growth_rate_l147_147271


namespace smallest_class_size_l147_147938

theorem smallest_class_size (n : ℕ) (h : 5 * n + 2 > 40) : 5 * n + 2 ≥ 42 :=
by
  sorry

end smallest_class_size_l147_147938


namespace sum_of_solutions_equation_l147_147613

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l147_147613


namespace tennis_players_l147_147099

theorem tennis_players (total_members badminton_players neither_players both_players : ℕ)
  (h1 : total_members = 80)
  (h2 : badminton_players = 48)
  (h3 : neither_players = 7)
  (h4 : both_players = 21) :
  total_members - neither_players = badminton_players - both_players + (total_members - neither_players - badminton_players + both_players) + both_players →
  ((total_members - neither_players) - (badminton_players - both_players) - both_players) + both_players = 46 :=
by
  intros h
  sorry

end tennis_players_l147_147099


namespace smallest_sum_of_15_consecutive_positive_integers_is_375_l147_147850

noncomputable def sum_of_15_consecutive_integers_is_perfect_square
  (m : ℕ) : Prop :=
  let sum := 15 * (m + 7) in
  (∃ k : ℕ, sum = k * k)

theorem smallest_sum_of_15_consecutive_positive_integers_is_375 :
  ∃ m : ℕ, m > 0 ∧ sum_of_15_consecutive_integers_is_perfect_square m ∧ 15 * (m + 7) = 375 :=
by
  sorry

end smallest_sum_of_15_consecutive_positive_integers_is_375_l147_147850


namespace band_member_earnings_l147_147068

-- Define conditions
def n_people : ℕ := 500
def p_ticket : ℚ := 30
def r_earnings : ℚ := 0.7
def n_members : ℕ := 4

-- Definition of total earnings and share per band member
def total_earnings : ℚ := n_people * p_ticket
def band_share : ℚ := total_earnings * r_earnings
def amount_per_member : ℚ := band_share / n_members

-- Statement to be proved
theorem band_member_earnings : amount_per_member = 2625 := 
by
  -- Proof goes here
  sorry

end band_member_earnings_l147_147068


namespace value_of_x_l147_147013

theorem value_of_x : (2015^2 + 2015 - 1) / (2015 : ℝ) = 2016 - 1 / 2015 := 
  sorry

end value_of_x_l147_147013


namespace probability_three_specific_cards_l147_147420

theorem probability_three_specific_cards :
  let deck_size := 52
  let diamonds := 13
  let spades := 13
  let hearts := 13
  let p1 := diamonds / deck_size
  let p2 := spades / (deck_size - 1)
  let p3 := hearts / (deck_size - 2)
  p1 * p2 * p3 = 169 / 5100 :=
by
  sorry

end probability_three_specific_cards_l147_147420


namespace minimum_small_droppers_l147_147746

/-
Given:
1. A total volume to be filled: V = 265 milliliters.
2. Small droppers can hold: s = 19 milliliters each.
3. No large droppers are used.

Prove:
The minimum number of small droppers required to fill the container completely is 14.
-/

theorem minimum_small_droppers (V s: ℕ) (hV: V = 265) (hs: s = 19) : 
  ∃ n: ℕ, n = 14 ∧ n * s ≥ V ∧ (n - 1) * s < V :=
by
  sorry  -- proof to be provided

end minimum_small_droppers_l147_147746


namespace total_pages_read_l147_147120

-- Define the average pages read by Lucas for the first four days.
def day1_4_avg : ℕ := 42

-- Define the average pages read by Lucas for the next two days.
def day5_6_avg : ℕ := 50

-- Define the pages read on the last day.
def day7 : ℕ := 30

-- Define the total number of days for which measurement is provided.
def total_days : ℕ := 7

-- Prove that the total number of pages Lucas read is 298.
theorem total_pages_read : 
  4 * day1_4_avg + 2 * day5_6_avg + day7 = 298 := 
by 
  sorry

end total_pages_read_l147_147120


namespace bus_driver_total_hours_l147_147741

variables (R OT : ℕ)

-- Conditions
def regular_rate := 16
def overtime_rate := 28
def max_regular_hours := 40
def total_compensation := 864

-- Proof goal: total hours worked is 48
theorem bus_driver_total_hours :
  (regular_rate * R + overtime_rate * OT = total_compensation) →
  (R ≤ max_regular_hours) →
  (R + OT = 48) :=
by
  sorry

end bus_driver_total_hours_l147_147741


namespace three_digit_multiples_of_36_eq_25_l147_147223

-- Definition: A three-digit number is between 100 and 999
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Definition: A number is a multiple of both 4 and 9 if and only if it's a multiple of 36
def is_multiple_of_36 (n : ℕ) : Prop := n % 36 = 0

-- Definition: Count of three-digit integers that are multiples of 36
def count_multiples_of_36 : ℕ :=
  (999 / 36) - (100 / 36) + 1

-- Theorem: There are 25 three-digit integers that are multiples of 36
theorem three_digit_multiples_of_36_eq_25 : count_multiples_of_36 = 25 := by
  sorry

end three_digit_multiples_of_36_eq_25_l147_147223


namespace arithmetic_sequence_sum_l147_147272

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ) (d : ℤ),
    a 1 = 1 →
    d ≠ 0 →
    (a 2 = a 1 + d) →
    (a 3 = a 1 + 2 * d) →
    (a 6 = a 1 + 5 * d) →
    (a 3)^2 = (a 2) * (a 6) →
    (1 + 2 * d)^2 = (1 + d) * (1 + 5 * d) →
    (6 / 2) * (2 * a 1 + (6 - 1) * d) = -24 := 
by intros a d h1 h2 h3 h4 h5 h6 h7
   sorry

end arithmetic_sequence_sum_l147_147272


namespace correct_group_l147_147335

def atomic_number (element : String) : Nat :=
  match element with
  | "Be" => 4
  | "C" => 6
  | "B" => 5
  | "Cl" => 17
  | "O" => 8
  | "Li" => 3
  | "Al" => 13
  | "S" => 16
  | "Si" => 14
  | "Mg" => 12
  | _ => 0

def is_descending (lst : List Nat) : Bool :=
  match lst with
  | [] => true
  | [x] => true
  | x :: y :: xs => if x > y then is_descending (y :: xs) else false

theorem correct_group : is_descending [atomic_number "Cl", atomic_number "O", atomic_number "Li"] = true ∧
                        is_descending [atomic_number "Be", atomic_number "C", atomic_number "B"] = false ∧
                        is_descending [atomic_number "Al", atomic_number "S", atomic_number "Si"] = false ∧
                        is_descending [atomic_number "C", atomic_number "S", atomic_number "Mg"] = false :=
by
  -- Prove the given theorem based on the atomic number function and is_descending condition
  sorry

end correct_group_l147_147335


namespace value_of_f_at_2_l147_147397

def f (x : ℤ) : ℤ := x^3 - x

theorem value_of_f_at_2 : f 2 = 6 := by
  sorry

end value_of_f_at_2_l147_147397


namespace find_b_l147_147148

theorem find_b (b : ℝ) (tangent_condition : ∀ x y : ℝ, y = -2 * x + b → y^2 = 8 * x) : b = -1 :=
sorry

end find_b_l147_147148


namespace calc_f_7_2_l147_147364

variable {f : ℝ → ℝ}

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_sqrt_on_interval : ∀ x, 0 < x ∧ x ≤ 1 → f x = Real.sqrt x

theorem calc_f_7_2 : f (7 / 2) = -Real.sqrt 2 / 2 := by
  sorry

end calc_f_7_2_l147_147364


namespace tens_digit_of_23_pow_2057_l147_147770

theorem tens_digit_of_23_pow_2057 : (23^2057 % 100) / 10 % 10 = 6 := 
by
  sorry

end tens_digit_of_23_pow_2057_l147_147770


namespace faye_has_62_pieces_of_candy_l147_147789

-- Define initial conditions
def initialCandy : Nat := 47
def eatenCandy : Nat := 25
def receivedCandy : Nat := 40

-- Define the resulting number of candies after eating and receiving more candies
def resultingCandy : Nat := initialCandy - eatenCandy + receivedCandy

-- State the theorem and provide the proof
theorem faye_has_62_pieces_of_candy :
  resultingCandy = 62 :=
by
  -- proof goes here
  sorry

end faye_has_62_pieces_of_candy_l147_147789


namespace average_of_25_results_is_24_l147_147532

theorem average_of_25_results_is_24 
  (first12_sum : ℕ)
  (last12_sum : ℕ)
  (result13 : ℕ)
  (n1 n2 n3 : ℕ)
  (h1 : n1 = 12)
  (h2 : n2 = 12)
  (h3 : n3 = 25)
  (avg_first12 : first12_sum = 14 * n1)
  (avg_last12 : last12_sum = 17 * n2)
  (res_13 : result13 = 228) :
  (first12_sum + last12_sum + result13) / n3 = 24 :=
by
  sorry

end average_of_25_results_is_24_l147_147532


namespace cost_price_of_product_is_100_l147_147577

theorem cost_price_of_product_is_100 
  (x : ℝ) 
  (h : x * 1.2 * 0.9 - x = 8) : 
  x = 100 := 
sorry

end cost_price_of_product_is_100_l147_147577


namespace aliyah_more_phones_l147_147046

theorem aliyah_more_phones (vivi_phones : ℕ) (phone_price : ℕ) (total_money : ℕ) (aliyah_more : ℕ) : 
  vivi_phones = 40 → 
  phone_price = 400 → 
  total_money = 36000 → 
  40 + 40 + aliyah_more = total_money / phone_price → 
  aliyah_more = 10 :=
sorry

end aliyah_more_phones_l147_147046


namespace cosine_quartic_representation_l147_147768

theorem cosine_quartic_representation :
  ∃ (a b c : ℝ), (∀ θ : ℝ, cos θ = cos (θ % (2 * π))) → 
    (∀ θ : ℝ, cos θ = cos ((-θ) % (2 * π))) → 
    a = 1/8 ∧ b = 1/2 ∧ c = 0 ∧
    (∀ θ : ℝ, cos (4*θ) = 8 * (cos θ)^4 - 8 * (cos θ)^2 + 1) ∧ 
    (∀ θ : ℝ, cos (2*θ) = 2 * (cos θ)^2 - 1) ∧
    (∀ θ : ℝ, (cos θ)^4 = a * cos (4*θ) + b * cos (2*θ) + c * cos θ) :=
by
  use 1/8, 1/2, 0
  split
  · refl  -- a = 1/8
  split
  · refl  -- b = 1/2
  split
  · refl  -- c = 0
  split
  · assume θ
    sorry  -- proof for cos(4θ) formula
  split
  · assume θ
    sorry  -- proof for cos(2θ) formula
  assume θ
  sorry  -- proof for the main goal

end cosine_quartic_representation_l147_147768


namespace remainder_of_7529_div_by_9_is_not_divisible_by_11_l147_147301

theorem remainder_of_7529_div_by_9 : 7529 % 9 = 5 := by
  sorry

theorem is_not_divisible_by_11 : ¬ (7529 % 11 = 0) := by
  sorry

end remainder_of_7529_div_by_9_is_not_divisible_by_11_l147_147301


namespace torn_sheets_count_l147_147274

noncomputable def first_page_num : ℕ := 185
noncomputable def last_page_num : ℕ := 518
noncomputable def pages_per_sheet : ℕ := 2

theorem torn_sheets_count :
  last_page_num > first_page_num ∧
  last_page_num.digits = first_page_num.digits.rotate 1 ∧
  pages_per_sheet = 2 →
  (last_page_num - first_page_num + 1)/pages_per_sheet = 167 :=
by {
  sorry
}

end torn_sheets_count_l147_147274


namespace pages_torn_and_sheets_calculation_l147_147282

theorem pages_torn_and_sheets_calculation : 
  (∀ (n : ℕ), (sheet_no n) = (n + 1) / 2 → (2 * (n + 1) / 2) - 1 = n ∨ 2 * (n + 1) / 2 = n) →
  let first_page := 185 in
  let last_page := 518 in
  last_page = 518 → 
  ((last_page - first_page + 1) / 2) = 167 := 
by
  sorry

end pages_torn_and_sheets_calculation_l147_147282


namespace factorization_left_to_right_l147_147729

-- Definitions (conditions)
def exprD_lhs : ℝ → ℝ := λ x, x^2 - 9
def exprD_rhs : ℝ → ℝ := λ x, (x + 3) * (x - 3)

-- Statement
theorem factorization_left_to_right (x : ℝ) :
  exprD_lhs x = exprD_rhs x := 
by sorry

end factorization_left_to_right_l147_147729


namespace solve_trig_equation_l147_147409

open Real

theorem solve_trig_equation (x : ℝ) (n : ℤ) :
  (2 * tan (6 * x) ^ 4 + 4 * sin (4 * x) * sin (8 * x) - cos (8 * x) - cos (16 * x) + 2) / sqrt (cos x - sqrt 3 * sin x) = 0 
  ∧ cos x - sqrt 3 * sin x > 0 →
  ∃ (k : ℤ), x = 2 * π * k ∨ x = -π / 6 + 2 * π * k ∨ x = -π / 3 + 2 * π * k ∨ x = -π / 2 + 2 * π * k ∨ x = -2 * π / 3 + 2 * π * k :=
sorry

end solve_trig_equation_l147_147409


namespace calculate_expression_l147_147191

theorem calculate_expression (a : ℤ) (h : a = -2) : a^3 - a^2 = -12 := 
by
  sorry

end calculate_expression_l147_147191


namespace largest_expression_is_d_l147_147199

def expr_a := 3 + 0 + 4 + 8
def expr_b := 3 * 0 + 4 + 8
def expr_c := 3 + 0 * 4 + 8
def expr_d := 3 + 0 + 4 * 8
def expr_e := 3 * 0 * 4 * 8
def expr_f := (3 + 0 + 4) / 8

theorem largest_expression_is_d : 
  expr_d = 35 ∧ 
  expr_a = 15 ∧ 
  expr_b = 12 ∧ 
  expr_c = 11 ∧ 
  expr_e = 0 ∧ 
  expr_f = 7 / 8 ∧
  35 > 15 ∧ 
  35 > 12 ∧ 
  35 > 11 ∧ 
  35 > 0 ∧ 
  35 > 7 / 8 := 
by
  sorry

end largest_expression_is_d_l147_147199


namespace symmetric_point_of_A_is_correct_l147_147707

def symmetric_point_with_respect_to_x_axis (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, -A.2)

theorem symmetric_point_of_A_is_correct :
  symmetric_point_with_respect_to_x_axis (3, 4) = (3, -4) :=
by
  sorry

end symmetric_point_of_A_is_correct_l147_147707


namespace S_shaped_growth_curve_varied_growth_rate_l147_147497

theorem S_shaped_growth_curve_varied_growth_rate :
  ∀ (population_growth : ℝ → ℝ), 
    (∃ t1 t2 : ℝ, t1 < t2 ∧ 
      (∃ r : ℝ, r = population_growth t1 / t1 ∧ r ≠ population_growth t2 / t2)) 
    → 
    ∀ t3 t4 : ℝ, t3 < t4 → (population_growth t3 / t3) ≠ (population_growth t4 / t4) :=
by
  sorry

end S_shaped_growth_curve_varied_growth_rate_l147_147497


namespace valid_transformation_b_l147_147731

theorem valid_transformation_b (a b : ℚ) : ((-a - b) / (a + b) = -1) := sorry

end valid_transformation_b_l147_147731


namespace sum_prime_factors_143_l147_147001

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24 := 
by
  let p := 11
  let q := 13
  have h1 : 143 = p * q := by norm_num
  have h2 : prime p := by norm_num
  have h3 : prime q := by norm_num
  have h4 : p + q = 24 := by norm_num
  exact ⟨p, q, h2, h3, h1, h4⟩  

end sum_prime_factors_143_l147_147001


namespace range_of_a_l147_147493

theorem range_of_a (a : ℝ) : 
  (∃ x : ℤ, 2 < (x : ℝ) ∧ (x : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ y : ℤ, 2 < (y : ℝ) ∧ (y : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ z : ℤ, 2 < (z : ℝ) ∧ (z : ℝ) ≤ 2 * a - 1) ∧ 
  (∀ w : ℤ, 2 < (w : ℝ) ∧ (w : ℝ) ≤ 2 * a - 1 → w = 3 ∨ w = 4 ∨ w = 5) :=
  by
    sorry

end range_of_a_l147_147493


namespace four_digit_numbers_neither_5_nor_7_l147_147922

-- Define the range of four-digit numbers
def four_digit_numbers : Set ℕ := {x | 1000 ≤ x ∧ x ≤ 9999}

-- Define the predicates for multiples of 5, 7, and 35
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0
def is_multiple_of_7 (n : ℕ) : Prop := n % 7 = 0
def is_multiple_of_35 (n : ℕ) : Prop := n % 35 = 0

-- Using set notation to define the sets of multiples
def multiples_of_5 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_5 n}
def multiples_of_7 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_7 n}
def multiples_of_35 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_35 n}

-- Total count of 4-digit numbers
def total_four_digit_numbers : ℕ := 9000

-- Count of multiples of 5, 7, and 35 within 4-digit numbers
def count_multiples_of_5 : ℕ := 1800
def count_multiples_of_7 : ℕ := 1286
def count_multiples_of_35 : ℕ := 257

-- Count of multiples of 5 or 7 using the principle of inclusion-exclusion
def count_multiples_of_5_or_7 : ℕ := count_multiples_of_5 + count_multiples_of_7 - count_multiples_of_35

-- Prove that the number of 4-digit numbers which are multiples of neither 5 nor 7 is 6171
theorem four_digit_numbers_neither_5_nor_7 : 
  (total_four_digit_numbers - count_multiples_of_5_or_7) = 6171 := 
by 
  sorry

end four_digit_numbers_neither_5_nor_7_l147_147922


namespace decimal_to_fraction_simplify_l147_147316

theorem decimal_to_fraction_simplify (d : ℚ) (h : d = 3.68) : d = 92 / 25 :=
by
  rw h
  sorry

end decimal_to_fraction_simplify_l147_147316


namespace pages_torn_l147_147284

theorem pages_torn (n : ℕ) (H1 : n = 185) (H2 : ∃ m, m = 518 ∧ (digits 10 m = digits 10 n) ∧ (m % 2 = 0)) : 
  ∃ k, k = ((518 - 185 + 1) / 2) ∧ k = 167 :=
by sorry

end pages_torn_l147_147284


namespace correct_time_fraction_l147_147033

theorem correct_time_fraction :
  let hours := 24
  let correct_hours := 10
  let minutes_per_hour := 60
  let correct_minutes_per_hour := 20
  (correct_hours * correct_minutes_per_hour : ℝ) / (hours * minutes_per_hour) = (5 / 36 : ℝ) :=
by
  let hours := 24
  let correct_hours := 10
  let minutes_per_hour := 60
  let correct_minutes_per_hour := 20
  sorry

end correct_time_fraction_l147_147033


namespace determine_digit_l147_147195

theorem determine_digit (Θ : ℕ) (hΘ : Θ > 0 ∧ Θ < 10) (h : 630 / Θ = 40 + 3 * Θ) : Θ = 9 :=
sorry

end determine_digit_l147_147195


namespace intersection_point_l147_147357

theorem intersection_point :
  ∃ (x y : ℝ), (2 * x + 3 * y + 8 = 0) ∧ (x - y - 1 = 0) ∧ (x = -1) ∧ (y = -2) := 
by
  sorry

end intersection_point_l147_147357


namespace root_exists_in_interval_l147_147322

def f (x : ℝ) : ℝ := 2 * x + x - 2

theorem root_exists_in_interval :
  (∃ x ∈ (Set.Ioo 0 1), f x = 0) :=
by
  sorry

end root_exists_in_interval_l147_147322


namespace baseball_team_grouping_l147_147320

theorem baseball_team_grouping (new_players returning_players : ℕ) (group_size : ℕ) 
  (h_new : new_players = 4) (h_returning : returning_players = 6) (h_group : group_size = 5) : 
  (new_players + returning_players) / group_size = 2 := 
  by 
  sorry

end baseball_team_grouping_l147_147320


namespace repeating_decimal_to_fraction_l147_147775

theorem repeating_decimal_to_fraction : (0.36 : ℝ) = (11 / 30 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l147_147775


namespace train_speed_l147_147999

theorem train_speed (v : ℝ) (d : ℝ) : 
  (v > 0) →
  (d > 0) →
  (d + (d - 55) = 495) →
  (d / v = (d - 55) / 25) →
  v = 31.25 := 
by
  intros hv hd hdist heqn
  -- We can leave the proof part out because we only need the statement
  sorry

end train_speed_l147_147999


namespace cosine_of_acute_angle_l147_147215

theorem cosine_of_acute_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α = 4 / 5) : Real.cos α = 3 / 5 :=
by
  sorry

end cosine_of_acute_angle_l147_147215


namespace distance_between_intersection_points_l147_147351

noncomputable def C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def l (t : ℝ) : ℝ × ℝ :=
  (-2 * t + 2, 3 * t)

theorem distance_between_intersection_points :
  ∃ (A B : ℝ × ℝ), 
    (∃ θ : ℝ, C θ = A) ∧
    (∃ t : ℝ, l t = A) ∧
    (∃ θ : ℝ, C θ = B) ∧
    (∃ t : ℝ, l t = B) ∧
    dist A B = Real.sqrt 13 / 2 :=
sorry

end distance_between_intersection_points_l147_147351


namespace James_vegetable_intake_in_third_week_l147_147816

noncomputable def third_week_vegetable_intake : ℝ :=
  let asparagus_per_day_first_week : ℝ := 0.25
  let broccoli_per_day_first_week : ℝ := 0.25
  let cauliflower_per_day_first_week : ℝ := 0.5

  let asparagus_per_day_second_week := 2 * asparagus_per_day_first_week
  let broccoli_per_day_second_week := 3 * broccoli_per_day_first_week
  let cauliflower_per_day_second_week := cauliflower_per_day_first_week * 1.75
  let spinach_per_day_second_week : ℝ := 0.5
  
  let daily_intake_second_week := asparagus_per_day_second_week +
                                  broccoli_per_day_second_week +
                                  cauliflower_per_day_second_week +
                                  spinach_per_day_second_week
  
  let kale_per_day_third_week : ℝ := 0.5
  let zucchini_per_day_third_week : ℝ := 0.15
  
  let daily_intake_third_week := asparagus_per_day_second_week +
                                 broccoli_per_day_second_week +
                                 cauliflower_per_day_second_week +
                                 spinach_per_day_second_week +
                                 kale_per_day_third_week +
                                 zucchini_per_day_third_week
  
  daily_intake_third_week * 7

theorem James_vegetable_intake_in_third_week : 
  third_week_vegetable_intake = 22.925 :=
  by
    sorry

end James_vegetable_intake_in_third_week_l147_147816


namespace percentage_increase_after_decrease_and_increase_l147_147092

theorem percentage_increase_after_decrease_and_increase 
  (P : ℝ) 
  (h : 0.8 * P + (x / 100) * (0.8 * P) = 1.16 * P) : 
  x = 45 :=
by
  sorry

end percentage_increase_after_decrease_and_increase_l147_147092


namespace decimal_to_fraction_l147_147312

theorem decimal_to_fraction :
  (368 / 100 : ℚ) = (92 / 25 : ℚ) := by
  sorry

end decimal_to_fraction_l147_147312


namespace sum_of_solutions_eq_16_l147_147622

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l147_147622


namespace cheezit_bag_weight_l147_147243

-- Definitions based on the conditions of the problem
def cheezit_bags : ℕ := 3
def calories_per_ounce : ℕ := 150
def run_minutes : ℕ := 40
def calories_per_minute : ℕ := 12
def excess_calories : ℕ := 420

-- Main theorem stating the question with the solution
theorem cheezit_bag_weight (x : ℕ) : 
  (calories_per_ounce * cheezit_bags * x) - (run_minutes * calories_per_minute) = excess_calories → 
  x = 2 :=
by
  sorry

end cheezit_bag_weight_l147_147243


namespace Diaz_age_20_years_from_now_l147_147172

open Nat

theorem Diaz_age_20_years_from_now:
  (∃ (diaz_age : ℕ) (sierra_age : ℕ),
    sierra_age = 30 ∧
    40 + 10 * diaz_age = 20 + 10 * sierra_age ∧
    diaz_age + 20 = 56) :=
begin
  sorry
end

end Diaz_age_20_years_from_now_l147_147172


namespace range_of_a_l147_147931

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a-1) * x^2 + 2 * (a-1) * x - 4 ≥ 0 -> false) ↔ -3 < a ∧ a ≤ 1 := by
  sorry

end range_of_a_l147_147931


namespace bucket_full_weight_l147_147564

variable {a b x y : ℝ}

theorem bucket_full_weight (h1 : x + 2/3 * y = a) (h2 : x + 1/2 * y = b) : 
  (x + y) = 3 * a - 2 * b := 
sorry

end bucket_full_weight_l147_147564


namespace max_value_fraction_l147_147664

theorem max_value_fraction (x : ℝ) : 
  ∃ (n : ℤ), n = 3 ∧ 
  ∃ (y : ℝ), y = (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ∧ 
  y ≤ n := 
sorry

end max_value_fraction_l147_147664


namespace sum_of_solutions_eqn_l147_147637

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l147_147637


namespace cos_expression_l147_147473

-- Define the condition for the line l and its relationship
def slope_angle_of_line_l (α : ℝ) : Prop :=
  ∃ l : ℝ, l = 2

-- Given the tangent condition for α
def tan_alpha (α : ℝ) : Prop :=
  Real.tan α = 2

theorem cos_expression (α : ℝ) (h1 : slope_angle_of_line_l α) (h2 : tan_alpha α) :
  Real.cos (2015 * Real.pi / 2 - 2 * α) = -4/5 :=
by sorry

end cos_expression_l147_147473


namespace new_solid_edges_l147_147590

-- Definitions based on conditions
def original_vertices : ℕ := 8
def original_edges : ℕ := 12
def new_edges_per_vertex : ℕ := 3
def number_of_vertices : ℕ := original_vertices

-- Conclusion to prove
theorem new_solid_edges : 
  (original_edges + new_edges_per_vertex * number_of_vertices) = 36 := 
by
  sorry

end new_solid_edges_l147_147590


namespace f_2019_is_zero_l147_147503

noncomputable def f : ℝ → ℝ := sorry

axiom f_is_non_negative
  (x : ℝ) : 0 ≤ f x

axiom f_satisfies_condition
  (a b c : ℝ) : f (a^3) + f (b^3) + f (c^3) = 3 * f a * f b * f c

axiom f_one_not_one : f 1 ≠ 1

theorem f_2019_is_zero : f 2019 = 0 := 
  sorry

end f_2019_is_zero_l147_147503


namespace fold_creates_bisector_l147_147698

-- Define an angle α with its vertex located outside the drawing (hence inaccessible)
structure Angle :=
  (theta1 theta2 : ℝ) -- theta1 and theta2 are the measures of the two angle sides

-- Define the condition: there exists an angle on transparent paper
variable (a: Angle)

-- Prove that folding such that the sides of the angle coincide results in the crease formed being the bisector
theorem fold_creates_bisector (a: Angle) :
  ∃ crease, crease = (a.theta1 + a.theta2) / 2 := 
sorry

end fold_creates_bisector_l147_147698


namespace compare_sqrt_difference_minimize_materials_plan_compare_a_inv_l147_147127

-- Problem 1
theorem compare_sqrt_difference : 3 - Real.sqrt 2 > 4 - 2 * Real.sqrt 2 := 
  sorry

-- Problem 2
theorem minimize_materials_plan (x y : ℝ) (h : x > y) : 
  4 * x + 6 * y > 3 * x + 7 * y := 
  sorry

-- Problem 3
theorem compare_a_inv (a : ℝ) (h : a > 0) : 
  (0 < a ∧ a < 1) → a < 1 / a ∧ (a = 1 → a = 1 / a) ∧ (a > 1 → a > 1 / a) :=
  sorry

end compare_sqrt_difference_minimize_materials_plan_compare_a_inv_l147_147127


namespace cost_price_eq_l147_147333

variable (SP : Real) (profit_percentage : Real)

theorem cost_price_eq : SP = 100 → profit_percentage = 0.15 → (100 / (1 + profit_percentage)) = 86.96 :=
by
  intros hSP hProfit
  sorry

end cost_price_eq_l147_147333


namespace loan_duration_l147_147175

theorem loan_duration (P R SI : ℝ) (hP : P = 20000) (hR : R = 12) (hSI : SI = 7200) : 
  ∃ T : ℝ, T = 3 :=
by
  sorry

end loan_duration_l147_147175


namespace sum_of_solutions_eq_16_l147_147627

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l147_147627


namespace sequence_sum_periodic_l147_147679

theorem sequence_sum_periodic (a : ℕ → ℕ) (a1 a8 : ℕ) :
  a 1 = 11 →
  a 8 = 12 →
  (∀ i, 1 ≤ i → i ≤ 6 → a i + a (i + 1) + a (i + 2) = 50) →
  (a 1 = 11 ∧ a 2 = 12 ∧ a 3 = 27 ∧ a 4 = 11 ∧ a 5 = 12 ∧ a 6 = 27 ∧ a 7 = 11 ∧ a 8 = 12) :=
by
  intros h1 h8 hsum
  sorry

end sequence_sum_periodic_l147_147679


namespace angle_in_third_quadrant_l147_147795

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α * Real.cos α > 0) (h2 : Real.sin α * Real.tan α < 0) : 
  (π < α ∧ α < 3 * π / 2) :=
by
  sorry

end angle_in_third_quadrant_l147_147795


namespace time_to_save_for_downpayment_l147_147962

def annual_salary : ℝ := 120000
def savings_percentage : ℝ := 0.15
def house_cost : ℝ := 550000
def downpayment_percentage : ℝ := 0.25

def annual_savings : ℝ := savings_percentage * annual_salary
def downpayment_needed : ℝ := downpayment_percentage * house_cost

theorem time_to_save_for_downpayment :
  (downpayment_needed / annual_savings) = 7.64 :=
by
  -- Proof to be provided
  sorry

end time_to_save_for_downpayment_l147_147962


namespace num_other_adults_l147_147524

-- Define the variables and conditions
def num_baskets : ℕ := 15
def eggs_per_basket : ℕ := 12
def eggs_per_person : ℕ := 9
def shonda_kids : ℕ := 2
def kids_friends : ℕ := 10
def num_participants : ℕ := (num_baskets * eggs_per_basket) / eggs_per_person

-- Prove the number of other adults at the Easter egg hunt
theorem num_other_adults : (num_participants - (shonda_kids + kids_friends + 1)) = 7 := by
  sorry

end num_other_adults_l147_147524


namespace subtract_vectors_l147_147642

def vec_a : ℤ × ℤ × ℤ := (5, -3, 2)
def vec_b : ℤ × ℤ × ℤ := (-2, 4, 1)
def vec_result : ℤ × ℤ × ℤ := (9, -11, 0)

theorem subtract_vectors :
  vec_a - 2 • vec_b = vec_result :=
by sorry

end subtract_vectors_l147_147642


namespace exists_n_l147_147819

theorem exists_n (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k > 0 → ¬(2^n ∣ a^k + b^k + c^k) :=
by
  sorry

end exists_n_l147_147819


namespace geometric_sequence_n_value_l147_147392

theorem geometric_sequence_n_value (a₁ : ℕ) (q : ℕ) (a_n : ℕ) (n : ℕ) (h1 : a₁ = 1) (h2 : q = 2) (h3 : a_n = 64) (h4 : a_n = a₁ * q^(n-1)) : n = 7 :=
by
  sorry

end geometric_sequence_n_value_l147_147392


namespace max_elements_of_S_l147_147109

-- Define the relation on set S and the conditions given
variable {S : Type} (R : S → S → Prop)

-- Lean translation of the conditions
def condition_1 (a b : S) : Prop :=
  (R a b ∨ R b a) ∧ ¬ (R a b ∧ R b a)

def condition_2 (a b c : S) : Prop :=
  R a b ∧ R b c → R c a

-- Define the problem statement:
theorem max_elements_of_S (h1 : ∀ a b : S, condition_1 R a b)
                          (h2 : ∀ a b c : S, condition_2 R a b c) :
  ∃ (n : ℕ), (∀ T : Finset S, T.card ≤ n) ∧ (∃ T : Finset S, T.card = 3) :=
sorry

end max_elements_of_S_l147_147109


namespace evaluate_fraction_l147_147354

theorem evaluate_fraction :
  (20 - 18 + 16 - 14 + 12 - 10 + 8 - 6 + 4 - 2) / (2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18) = 1 :=
by
  sorry

end evaluate_fraction_l147_147354


namespace average_of_first_12_l147_147846

theorem average_of_first_12 (avg25 : ℝ) (avg12 : ℝ) (avg_last12 : ℝ) (result_13th : ℝ) : 
  (avg25 = 18) → (avg_last12 = 17) → (result_13th = 78) → 
  25 * avg25 = (12 * avg12) + result_13th + (12 * avg_last12) → avg12 = 14 :=
by 
  sorry

end average_of_first_12_l147_147846


namespace emt_selection_ways_l147_147802

theorem emt_selection_ways :
  let groupA_nurses := 4
  let groupA_doctors := 1
  let groupB_nurses := 6
  let groupB_doctors := 2
  let ways_scenario1 := (nat.choose groupA_doctors 1) * (nat.choose groupA_nurses 1) * 
                        (nat.choose groupB_nurses 2) * (nat.choose groupB_doctors 0)
  let ways_scenario2 := (nat.choose groupA_doctors 0) * (nat.choose groupA_nurses 2) * 
                        (nat.choose groupB_doctors 1) * (nat.choose groupB_nurses 1)
  ways_scenario1 + ways_scenario2 = 132 := by 
let groupA_nurses := 4
let groupA_doctors := 1
let groupB_nurses := 6
let groupB_doctors := 2
let ways_scenario1 := (nat.choose groupA_doctors 1) * (nat.choose groupA_nurses 1) * 
                      (nat.choose groupB_nurses 2) * (nat.choose groupB_doctors 0)
let ways_scenario2 := (nat.choose groupA_doctors 0) * (nat.choose groupA_nurses 2) * 
                      (nat.choose groupB_doctors 1) * (nat.choose groupB_nurses 1)
have : ways_scenario1 = 60 :=
  by norm_num [nat.choose] 
have : ways_scenario2 = 72 := 
  by norm_num [nat.choose] 
show (60 + 72 = 132) from rfl

end emt_selection_ways_l147_147802


namespace sum_of_ages_l147_147855

theorem sum_of_ages (X_c Y_c : ℕ) (h1 : X_c = 45) 
  (h2 : X_c - 3 = 2 * (Y_c - 3)) : 
  (X_c + 7) + (Y_c + 7) = 83 := 
by
  sorry

end sum_of_ages_l147_147855


namespace number_of_triples_l147_147065

noncomputable def count_valid_triples : ℕ :=
  ∑ a in Finset.range 2017, 2016 - a + 1

theorem number_of_triples : count_valid_triples = 2031120 := by
  sorry

end number_of_triples_l147_147065


namespace num_diagonals_octagon_l147_147190

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem num_diagonals_octagon : num_diagonals 8 = 20 :=
by
  sorry

end num_diagonals_octagon_l147_147190


namespace volume_of_mixture_l147_147989

theorem volume_of_mixture
    (weight_a : ℝ) (weight_b : ℝ) (ratio_a_b : ℝ) (total_weight : ℝ)
    (h1 : weight_a = 900) (h2 : weight_b = 700)
    (h3 : ratio_a_b = 3/2) (h4 : total_weight = 3280) :
    ∃ Va Vb : ℝ, (Va / Vb = ratio_a_b) ∧ (weight_a * Va + weight_b * Vb = total_weight) ∧ (Va + Vb = 4) := 
by
  sorry

end volume_of_mixture_l147_147989


namespace max_non_managers_l147_147936

theorem max_non_managers (N : ℕ) : (8 / N : ℚ) > 7 / 32 → N ≤ 36 :=
by sorry

end max_non_managers_l147_147936


namespace exists_n_for_m_l147_147252

def π (x : ℕ) : ℕ := sorry -- Placeholder for the prime counting function

theorem exists_n_for_m (m : ℕ) (hm : m > 1) : ∃ n : ℕ, n > 1 ∧ n / π n = m :=
by sorry

end exists_n_for_m_l147_147252


namespace log_inequality_l147_147950

noncomputable def a := Real.log 6 / Real.log 3
noncomputable def b := Real.log 10 / Real.log 5
noncomputable def c := Real.log 14 / Real.log 7

theorem log_inequality :
  a > b ∧ b > c :=
by
  sorry

end log_inequality_l147_147950


namespace simplify_sqrt_expression_l147_147134

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l147_147134


namespace expression_multiple_l147_147644

theorem expression_multiple :
  let a : ℚ := 1/2
  let b : ℚ := 1/3
  (a - b) / (1/78) = 13 :=
by
  sorry

end expression_multiple_l147_147644


namespace simplify_expr_l147_147702

variable (a b c : ℤ)

theorem simplify_expr :
  (15 * a + 45 * b + 20 * c) + (25 * a - 35 * b - 10 * c) - (10 * a + 55 * b + 30 * c) = 30 * a - 45 * b - 20 * c := 
by
  sorry

end simplify_expr_l147_147702


namespace correct_multiplication_l147_147566

variable {a : ℕ} -- Assume 'a' to be a natural number for simplicity in this example

theorem correct_multiplication : (3 * a) * (4 * a^2) = 12 * a^3 := by
  sorry

end correct_multiplication_l147_147566


namespace probability_A_given_B_probability_A_or_B_l147_147554

-- Definitions of the given conditions
def PA : ℝ := 0.2
def PB : ℝ := 0.18
def PAB : ℝ := 0.12

-- Theorem to prove the probability that city A also experiences rain when city B is rainy
theorem probability_A_given_B : PA * PB = PAB -> PA = 2 / 3 := by
  sorry

-- Theorem to prove the probability that at least one of the two cities experiences rain
theorem probability_A_or_B (PA PB PAB : ℝ) : (PA + PB - PAB) = 0.26 := by
  sorry

end probability_A_given_B_probability_A_or_B_l147_147554


namespace walkway_area_correct_l147_147496

/-- Definitions as per problem conditions --/
def bed_length : ℕ := 8
def bed_width : ℕ := 3
def walkway_bed_width : ℕ := 2
def walkway_row_width : ℕ := 1
def num_beds_in_row : ℕ := 3
def num_rows : ℕ := 4

/-- Total dimensions including walkways --/
def total_width := num_beds_in_row * bed_length + (num_beds_in_row + 1) * walkway_bed_width
def total_height := num_rows * bed_width + (num_rows + 1) * walkway_row_width

/-- Total areas --/
def total_area := total_width * total_height
def bed_area := bed_length * bed_width
def total_bed_area := num_beds_in_row * num_rows * bed_area
def walkway_area := total_area - total_bed_area

theorem walkway_area_correct : walkway_area = 256 := by
  /- Import necessary libraries and skip the proof -/
  sorry

end walkway_area_correct_l147_147496


namespace allison_not_lowest_probability_l147_147443

open ProbabilityTheory

-- Definitions based on conditions
def allisonRolls : ℕ := 3

def brianRollDistribution : Pmf ℕ :=
Pmf.ofFinset (finset.range 6) (λ n, if n > 0 ∧ n ≤ 6 then 1 else 0)

def noahRollDistribution : Pmf ℕ :=
Pmf.ofFinset (finset.range 2) (λ n, if n = 0 then 3 else if n = 1 then 3 else 0)

-- The proof problem: proving the probability calculation
theorem allison_not_lowest_probability :
  P (λ (a b n : ℕ), allisonRolls ≥ b ∧ allisonRolls ≥ n) = 5 / 6 :=
by
  sorry

end allison_not_lowest_probability_l147_147443


namespace possible_values_of_C_l147_147972

theorem possible_values_of_C {a b C : ℤ} :
  (C = a * (a - 5) ∧ C = b * (b - 8)) ↔ (C = 0 ∨ C = 84) :=
sorry

end possible_values_of_C_l147_147972


namespace painting_cost_l147_147250

-- Define contributions
def JudsonContrib := 500
def KennyContrib := JudsonContrib + 0.20 * JudsonContrib
def CamiloContrib := KennyContrib + 200

-- Define total cost
def TotalCost := JudsonContrib + KennyContrib + CamiloContrib

-- Theorem to prove
theorem painting_cost : TotalCost = 1900 :=
by 
  -- Calculate Kenny's contribution
  have hK : KennyContrib = 600 := by 
    simp [KennyContrib, JudsonContrib]
    sorry -- additional steps would go here, we use sorry to skip details

  -- Calculate Camilo's contribution
  have hC : CamiloContrib = 800 := by 
    simp [CamiloContrib, hK]
    sorry -- additional steps would go here, we use sorry to skip details

  -- Calculate total cost
  simp [TotalCost, JudsonContrib, hK, hC]
  sorry -- additional steps would go here, we use sorry to skip details

end painting_cost_l147_147250


namespace correct_option_e_l147_147428

theorem correct_option_e : 15618 = 1 + 5^6 - 1 * 8 :=
by sorry

end correct_option_e_l147_147428


namespace band_member_earnings_l147_147070

theorem band_member_earnings :
  let attendees := 500
  let ticket_price := 30
  let band_share_percentage := 70 / 100
  let band_members := 4
  let total_earnings := attendees * ticket_price
  let band_earnings := total_earnings * band_share_percentage
  let earnings_per_member := band_earnings / band_members
  earnings_per_member = 2625 := 
by {
  sorry
}

end band_member_earnings_l147_147070


namespace probability_of_one_triplet_without_any_pairs_l147_147462

noncomputable def probability_one_triplet_no_pairs : ℚ :=
  let total_outcomes := 6^5
  let choices_for_triplet := 6
  let ways_to_choose_triplet_dice := Nat.choose 5 3
  let choices_for_remaining_dice := 5 * 4
  let successful_outcomes := choices_for_triplet * ways_to_choose_triplet_dice * choices_for_remaining_dice
  successful_outcomes / total_outcomes

theorem probability_of_one_triplet_without_any_pairs :
  probability_one_triplet_no_pairs = 25 / 129 := by
  sorry

end probability_of_one_triplet_without_any_pairs_l147_147462


namespace cosine_equation_solution_count_l147_147769

open Real

noncomputable def number_of_solutions : ℕ := sorry

theorem cosine_equation_solution_count :
  number_of_solutions = 2 :=
by
  -- Let x be an angle in [0, 2π].
  sorry

end cosine_equation_solution_count_l147_147769


namespace inequality_proof_equality_conditions_l147_147080

theorem inequality_proof
  (x y : ℝ)
  (h1 : x ≥ y)
  (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y) + y / Real.sqrt (y + 1) + 1 / Real.sqrt (x + 1) ≥
   y / Real.sqrt (x + y) + x / Real.sqrt (x + 1) + 1 / Real.sqrt (y + 1)) :=
by
  sorry

theorem equality_conditions
  (x y : ℝ) :
  (x = y ∨ x = 1 ∨ y = 1) ↔
  (x / Real.sqrt (x + y) + y / Real.sqrt (y + 1) + 1 / Real.sqrt (x + 1) =
   y / Real.sqrt (x + y) + x / Real.sqrt (x + 1) + 1 / Real.sqrt (y + 1)) :=
by
  sorry

end inequality_proof_equality_conditions_l147_147080


namespace avg_percentage_decrease_l147_147436

theorem avg_percentage_decrease (x : ℝ) 
  (h : 16 * (1 - x)^2 = 9) : x = 0.25 :=
sorry

end avg_percentage_decrease_l147_147436


namespace remainder_when_divided_by_x_plus_2_l147_147014

variable (D E F : ℝ)

def q (x : ℝ) := D * x^4 + E * x^2 + F * x + 7

theorem remainder_when_divided_by_x_plus_2 :
  q D E F (-2) = 21 - 2 * F :=
by
  have hq2 : q D E F 2 = 21 := sorry
  sorry

end remainder_when_divided_by_x_plus_2_l147_147014


namespace school_population_proof_l147_147391

variables (x y z: ℕ)
variable (B: ℕ := (50 * y) / 100)

theorem school_population_proof (h1 : 162 = (x * B) / 100)
                               (h2 : B = (50 * y) / 100)
                               (h3 : z = 100 - 50) :
  z = 50 :=
  sorry

end school_population_proof_l147_147391


namespace sum_of_squares_correct_l147_147997

-- Define the three incorrect entries
def incorrect_entry_1 : Nat := 52
def incorrect_entry_2 : Nat := 81
def incorrect_entry_3 : Nat := 111

-- Define the sum of the squares of these entries
def sum_of_squares : Nat := incorrect_entry_1 ^ 2 + incorrect_entry_2 ^ 2 + incorrect_entry_3 ^ 2

-- State that this sum of squares equals 21586
theorem sum_of_squares_correct : sum_of_squares = 21586 := by
  sorry

end sum_of_squares_correct_l147_147997


namespace find_p_q_r_l147_147895

theorem find_p_q_r : 
  ∃ (p q r : ℕ), 
  p > 0 ∧ q > 0 ∧ r > 0 ∧ 
  4 * (Real.sqrt (Real.sqrt 7) - Real.sqrt (Real.sqrt 6)) 
  = Real.sqrt (Real.sqrt p) + Real.sqrt (Real.sqrt q) - Real.sqrt (Real.sqrt r) 
  ∧ p + q + r = 99 := 
sorry

end find_p_q_r_l147_147895


namespace cos_five_theta_l147_147227

theorem cos_five_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (5 * θ) = (125 * Real.sqrt 15 - 749) / 1024 := 
  sorry

end cos_five_theta_l147_147227


namespace sequence_integers_l147_147986

theorem sequence_integers (a : ℕ → ℤ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 1) 
  (h₃ : ∀ n, n ≥ 3 → a n = (a (n-1)) ^ 2 + 2 / a (n-2)) : 
  ∀ n, ∃ k : ℤ, a n = k := 
by 
  sorry

end sequence_integers_l147_147986


namespace sum_of_solutions_l147_147615

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l147_147615


namespace necessary_but_not_sufficient_condition_l147_147213

open Real

-- Define α as an internal angle of a triangle
def is_internal_angle (α : ℝ) : Prop := (0 < α ∧ α < π)

-- Given conditions
axiom α : ℝ
axiom h1 : is_internal_angle α

-- Prove: if (α ≠ π / 6) then (sin α ≠ 1 / 2) is a necessary but not sufficient condition 
theorem necessary_but_not_sufficient_condition : 
  (α ≠ π / 6) ∧ ¬((α ≠ π / 6) → (sin α ≠ 1 / 2)) ∧ ((sin α ≠ 1 / 2) → (α ≠ π / 6)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l147_147213


namespace followers_after_one_year_l147_147601

theorem followers_after_one_year :
  let initial_followers := 100000
  let daily_new_followers := 1000
  let unfollowers_per_year := 20000
  let days_per_year := 365
  initial_followers + (daily_new_followers * days_per_year - unfollowers_per_year) = 445000 :=
by
  sorry

end followers_after_one_year_l147_147601


namespace difference_between_numbers_l147_147412

theorem difference_between_numbers (x y : ℕ) (h : x - y = 9) :
  (10 * x + y) - (10 * y + x) = 81 :=
by
  sorry

end difference_between_numbers_l147_147412


namespace sum_prime_factors_143_l147_147012

open Nat

theorem sum_prime_factors_143 : (11 + 13) = 24 :=
by
  have h1 : Prime 11 := by sorry
  have h2 : Prime 13 := by sorry
  have h3 : 143 = 11 * 13 := by sorry
  exact add_eq_of_eq h3 (11 + 13) 24 sorry

end sum_prime_factors_143_l147_147012


namespace band_member_earnings_l147_147072

theorem band_member_earnings :
  let attendees := 500
  let ticket_price := 30
  let band_share_percentage := 70 / 100
  let band_members := 4
  let total_earnings := attendees * ticket_price
  let band_earnings := total_earnings * band_share_percentage
  let earnings_per_member := band_earnings / band_members
  earnings_per_member = 2625 := 
by {
  sorry
}

end band_member_earnings_l147_147072


namespace added_number_is_five_l147_147161

variable (n x : ℤ)

theorem added_number_is_five (h1 : n % 25 = 4) (h2 : (n + x) % 5 = 4) : x = 5 :=
by
  sorry

end added_number_is_five_l147_147161


namespace geometric_sequence_nec_suff_l147_147268

theorem geometric_sequence_nec_suff (a b c : ℝ) : (b^2 = a * c) ↔ (∃ r : ℝ, b = a * r ∧ c = b * r) :=
sorry

end geometric_sequence_nec_suff_l147_147268


namespace allocate_teaching_positions_l147_147224

theorem allocate_teaching_positions :
  ∃ (ways : ℕ), ways = 10 ∧ 
    (∃ (a b c : ℕ), a + b + c = 8 ∧ 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 2 ≤ a) := 
sorry

end allocate_teaching_positions_l147_147224


namespace f_20_value_l147_147400

noncomputable def f (n : ℕ) : ℚ := sorry

axiom f_initial : f 1 = 3 / 2
axiom f_eq : ∀ x y : ℕ, 
  f (x + y) = (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2

theorem f_20_value : f 20 = 4305 := 
by {
  sorry 
}

end f_20_value_l147_147400


namespace rational_sum_is_negative_then_at_most_one_positive_l147_147859

theorem rational_sum_is_negative_then_at_most_one_positive (a b : ℚ) (h : a + b < 0) :
  (a > 0 ∧ b ≤ 0) ∨ (a ≤ 0 ∧ b > 0) ∨ (a ≤ 0 ∧ b ≤ 0) :=
by
  sorry

end rational_sum_is_negative_then_at_most_one_positive_l147_147859


namespace arithmetic_sequence_ratio_l147_147794

theorem arithmetic_sequence_ratio
  (d : ℕ) (h₀ : d ≠ 0)
  (a : ℕ → ℕ)
  (h₁ : ∀ n, a (n + 1) = a n + d)
  (h₂ : (a 3)^2 = (a 1) * (a 9)) :
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5 / 8 :=
  sorry

end arithmetic_sequence_ratio_l147_147794


namespace sum_integers_30_to_50_subtract_15_l147_147452

-- Definitions and proof problem based on conditions
def sumIntSeries (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_30_to_50_subtract_15 : sumIntSeries 30 50 - 15 = 825 := by
  -- We are stating that the sum of the integers from 30 to 50 minus 15 is equal to 825
  sorry


end sum_integers_30_to_50_subtract_15_l147_147452


namespace sum_prime_factors_of_143_l147_147005

theorem sum_prime_factors_of_143 :
  let is_prime (n : ℕ) := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0 in
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ 143 = a * b ∧ a ≠ b ∧ (a + b = 24) :=
by
  sorry

end sum_prime_factors_of_143_l147_147005


namespace count_multiples_6_not_12_l147_147653

theorem count_multiples_6_not_12 (n: ℕ) : 
  ∃ (count : ℕ), count = 25 ∧ 
                  count = (finset.filter (λ m, (m < 300) ∧ (6 ∣ m) ∧ ¬ (12 ∣ m)) (finset.range 300)).card :=
by
  sorry

end count_multiples_6_not_12_l147_147653


namespace determine_c_l147_147490

theorem determine_c (c : ℚ) : (∀ x : ℝ, (x + 7) * (x^2 * c * x + 19 * x^2 - c * x - 49) = 0) → c = 21 / 8 :=
by
  sorry

end determine_c_l147_147490


namespace cos_difference_l147_147897

theorem cos_difference (a b : ℝ) : 
  cos (a + b) - cos (a - b) = -2 * sin a * sin b := sorry

end cos_difference_l147_147897


namespace band_member_earnings_l147_147067

-- Define conditions
def n_people : ℕ := 500
def p_ticket : ℚ := 30
def r_earnings : ℚ := 0.7
def n_members : ℕ := 4

-- Definition of total earnings and share per band member
def total_earnings : ℚ := n_people * p_ticket
def band_share : ℚ := total_earnings * r_earnings
def amount_per_member : ℚ := band_share / n_members

-- Statement to be proved
theorem band_member_earnings : amount_per_member = 2625 := 
by
  -- Proof goes here
  sorry

end band_member_earnings_l147_147067


namespace incorrect_score_modulo_l147_147588

theorem incorrect_score_modulo (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9) 
  (hb : 0 ≤ b ∧ b ≤ 9) 
  (hc : 0 ≤ c ∧ c ≤ 9) : 
  ∃ remainder : ℕ, remainder = (90 * a + 9 * b + c) % 9 ∧ 0 ≤ remainder ∧ remainder ≤ 9 := 
by
  sorry

end incorrect_score_modulo_l147_147588


namespace lele_dongdong_meet_probability_l147_147104

-- Define the conditions: distances and speeds
def segment_length : ℕ := 500
def n : ℕ := sorry
def d : ℕ := segment_length * n
def lele_speed : ℕ := 18
def dongdong_speed : ℕ := 24

-- Define times to traverse distance d
def t_L : ℚ := d / lele_speed
def t_D : ℚ := d / dongdong_speed

-- Define the time t when they meet
def t : ℚ := d / (lele_speed + dongdong_speed)

-- Define the maximum of t_L and t_D
def max_t_L_t_D : ℚ := max t_L t_D

-- Define the probability they meet on their way
def P_meet : ℚ := t / max_t_L_t_D

-- The theorem to prove the probability of meeting is 97/245
theorem lele_dongdong_meet_probability : P_meet = 97 / 245 :=
sorry

end lele_dongdong_meet_probability_l147_147104


namespace combined_cost_is_correct_l147_147044

-- Definitions based on the conditions
def dryer_cost : ℕ := 150
def washer_cost : ℕ := 3 * dryer_cost
def combined_cost : ℕ := dryer_cost + washer_cost

-- Statement to be proved
theorem combined_cost_is_correct : combined_cost = 600 :=
by
  sorry

end combined_cost_is_correct_l147_147044


namespace most_likely_number_of_red_balls_l147_147990

-- Define the conditions
def total_balls : ℕ := 20
def red_ball_frequency : ℝ := 0.8

-- Define the statement we want to prove
theorem most_likely_number_of_red_balls : red_ball_frequency * total_balls = 16 :=
by sorry

end most_likely_number_of_red_balls_l147_147990


namespace faye_coloring_books_l147_147058

theorem faye_coloring_books (initial_books : ℕ) (gave_away : ℕ) (bought_more : ℕ) (h1 : initial_books = 34) (h2 : gave_away = 3) (h3 : bought_more = 48) : 
  initial_books - gave_away + bought_more = 79 :=
by
  sorry

end faye_coloring_books_l147_147058


namespace sum_consecutive_integers_l147_147813

theorem sum_consecutive_integers (S : ℕ) (hS : S = 221) :
  ∃ (k : ℕ) (hk : k ≥ 2) (n : ℕ), 
    (S = k * n + (k * (k - 1)) / 2) → k = 2 := sorry

end sum_consecutive_integers_l147_147813


namespace diana_probability_l147_147604

open Classical

noncomputable def probability_diana_greater_or_double_apollo : ℚ :=
  let outcomes_diana := {1, 2, 3, 4, 5, 6, 7, 8}
  let outcomes_apollo := {1, 2, 3, 4}
  let successful_pairs := 
    { (a, b) | a ∈ outcomes_diana ∧ b ∈ outcomes_apollo ∧ 
               (a > b ∨ a = 2 * b) }
  let successful_outcomes := successful_pairs.card
  let total_outcomes := outcomes_diana.card * outcomes_apollo.card
  (successful_outcomes : ℚ) / (total_outcomes : ℚ)

theorem diana_probability : probability_diana_greater_or_double_apollo = 13 / 16 := by
  sorry

end diana_probability_l147_147604


namespace polygon_encloses_250_square_units_l147_147198

def vertices : List (ℕ × ℕ) := [(0, 0), (20, 0), (20, 20), (10, 20), (10, 10), (0, 10)]

def polygon_area (vertices : List (ℕ × ℕ)) : ℕ :=
  -- Function to calculate the area of the given polygon
  sorry

theorem polygon_encloses_250_square_units : polygon_area vertices = 250 := by
  -- Proof that the area of the polygon is 250 square units
  sorry

end polygon_encloses_250_square_units_l147_147198


namespace correct_transformation_option_c_l147_147303

theorem correct_transformation_option_c (x : ℝ) (h : (x / 2) - (x / 3) = 1) : 3 * x - 2 * x = 6 :=
by
  sorry

end correct_transformation_option_c_l147_147303


namespace number_exceeds_fraction_80_l147_147824

theorem number_exceeds_fraction_80 (x : ℝ) (h : x = (3 / 7) * x + 0.8 * (3 / 7) * x) : x = 0 := 
by
  sorry

end number_exceeds_fraction_80_l147_147824


namespace simplify_expression_l147_147130

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l147_147130


namespace undefined_expression_l147_147771

theorem undefined_expression (y : ℝ) : (y^2 - 16 * y + 64 = 0) ↔ (y = 8) := by
  sorry

end undefined_expression_l147_147771


namespace calvin_total_insects_l147_147586

-- Definitions based on the conditions
def roaches := 12
def scorpions := 3
def crickets := roaches / 2
def caterpillars := scorpions * 2

-- Statement of the problem
theorem calvin_total_insects : 
  roaches + scorpions + crickets + caterpillars = 27 :=
  by
    sorry

end calvin_total_insects_l147_147586


namespace closest_to_zero_is_neg_1001_l147_147427

-- Definitions used in the conditions
def list_of_integers : List Int := [-1101, 1011, -1010, -1001, 1110]

-- Problem statement
theorem closest_to_zero_is_neg_1001 (x : Int) (H : x ∈ list_of_integers) :
  x = -1001 ↔ ∀ y ∈ list_of_integers, abs x ≤ abs y :=
sorry

end closest_to_zero_is_neg_1001_l147_147427


namespace factorization_correct_l147_147728

theorem factorization_correct (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by 
  sorry

end factorization_correct_l147_147728


namespace paths_E_to_G_through_F_and_H_l147_147088

-- Define positions of E, F, H, and G on the grid.
structure Point where
  x : ℕ
  y : ℕ

def E : Point := { x := 0, y := 0 }
def F : Point := { x := 3, y := 2 }
def H : Point := { x := 5, y := 4 }
def G : Point := { x := 8, y := 4 }

-- Function to calculate number of paths from one point to another given the number of right and down steps
def paths (start goal : Point) : ℕ :=
  let right_steps := goal.x - start.x
  let down_steps := goal.y - start.y
  Nat.choose (right_steps + down_steps) right_steps

theorem paths_E_to_G_through_F_and_H : paths E F * paths F H * paths H G = 60 := by
  sorry

end paths_E_to_G_through_F_and_H_l147_147088


namespace matthew_hotdogs_needed_l147_147689

def total_hotdogs (ella_hotdogs emma_hotdogs : ℕ) (luke_multiplier hunter_multiplier : ℕ) : ℕ :=
  let total_sisters := ella_hotdogs + emma_hotdogs
  let luke := luke_multiplier * total_sisters
  let hunter := hunter_multiplier * total_sisters / 2  -- because 1.5 = 3/2 and hunter_multiplier = 3
  total_sisters + luke + hunter

theorem matthew_hotdogs_needed :
  total_hotdogs 2 2 2 3 = 18 := 
by
  -- This proof is correct given the calculations above
  sorry

end matthew_hotdogs_needed_l147_147689


namespace problem_equivalent_l147_147090

theorem problem_equivalent :
  500 * 2019 * 0.0505 * 20 = 2019^2 :=
by
  sorry

end problem_equivalent_l147_147090


namespace S_n_formula_l147_147448

def P (n : ℕ) : Type := sorry -- The type representing the nth polygon, not fully defined here.
def S : ℕ → ℝ := sorry -- The sequence S_n defined recursively.

-- Recursive definition of S_n given
axiom S_0 : S 0 = 1

-- This axiom represents the recursive step mentioned in the problem.
axiom S_rec : ∀ (k : ℕ), S (k + 1) = S k + (4^k / 3^(2*k + 2))

-- The main theorem we need to prove
theorem S_n_formula (n : ℕ) : 
  S n = (8 / 5) - (3 / 5) * (4 / 9)^n := sorry

end S_n_formula_l147_147448


namespace activity_popularity_order_l147_147756

theorem activity_popularity_order :
  let dodgeball := 13 / 40
  let movie_screening := 9 / 25
  let quiz_bowl := 7 / 20
  let gardening := 6 / 15
  min gardening (min movie_screening (min quiz_bowl dodgeball)) == gardening ∧
  min dodgeball (min movie_screening (min quiz_bowl gardening)) == dodgeball ∧
  min quiz_bowl (min gardening (min movie_screening dodgeball)) == quiz_bowl ∧
  min movie_screening (min gardening (min quiz_bowl dodgeball)) == movie_screening :=
by
  sorry

end activity_popularity_order_l147_147756


namespace unique_functional_equation_solution_l147_147602

theorem unique_functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end unique_functional_equation_solution_l147_147602


namespace melissa_remaining_bananas_l147_147255

theorem melissa_remaining_bananas :
  let initial_bananas := 88
  let shared_bananas := 4
  initial_bananas - shared_bananas = 84 :=
by
  sorry

end melissa_remaining_bananas_l147_147255


namespace decimal_to_fraction_simplify_l147_147314

theorem decimal_to_fraction_simplify (d : ℚ) (h : d = 3.68) : d = 92 / 25 :=
by
  rw h
  sorry

end decimal_to_fraction_simplify_l147_147314


namespace ratio_between_two_numbers_l147_147331

noncomputable def first_number : ℕ := 48
noncomputable def lcm_value : ℕ := 432
noncomputable def second_number : ℕ := 9 * 24  -- Derived from the given conditions in the problem

def ratio (a b : ℕ) : ℚ := (a : ℚ) / (b : ℚ)

theorem ratio_between_two_numbers 
  (A B : ℕ) 
  (hA : A = first_number) 
  (hLCM : Nat.lcm A B = lcm_value) 
  (hB : B = 9 * 24) : 
  ratio A B = 1 / 4.5 :=
by
  -- Proof would go here
  sorry

end ratio_between_two_numbers_l147_147331


namespace sum_prime_factors_143_l147_147008

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 143 = p * q ∧ p + q = 24 :=
begin
  use 13,
  use 11,
  repeat { split },
  { exact nat.prime_of_four_divisors 13 (by norm_num) },
  { exact nat.prime_of_four_divisors 11 (by norm_num) },
  { norm_num },
  { norm_num }
end

end sum_prime_factors_143_l147_147008


namespace simon_project_score_l147_147094

-- Define the initial conditions
def num_students_before : Nat := 20
def num_students_total : Nat := 21
def avg_before : ℕ := 86
def avg_after : ℕ := 88

-- Calculate total score before Simon's addition
def total_score_before : ℕ := num_students_before * avg_before

-- Calculate total score after Simon's addition
def total_score_after : ℕ := num_students_total * avg_after

-- Definition to represent Simon's score
def simon_score : ℕ := total_score_after - total_score_before

-- Theorem that we need to prove
theorem simon_project_score : simon_score = 128 :=
by
  sorry

end simon_project_score_l147_147094


namespace repeating_decimal_to_fraction_l147_147773

theorem repeating_decimal_to_fraction (x : ℝ) (h : x = 0.3 + 0.0666...) : x = 11 / 30 := by
  sorry

end repeating_decimal_to_fraction_l147_147773


namespace find_g2_l147_147979

-- Given conditions:
variables (g : ℝ → ℝ) 
axiom cond1 : ∀ (x y : ℝ), x * g y = 2 * y * g x
axiom cond2 : g 10 = 5

-- Proof to show g(2) = 2
theorem find_g2 : g 2 = 2 := 
by
  -- Skipping the actual proof
  sorry

end find_g2_l147_147979


namespace standard_equation_hyperbola_l147_147363

-- Define necessary conditions
def condition_hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

def condition_asymptote (a b : ℝ) :=
  b / a = Real.sqrt 3

def condition_focus_hyperbola_parabola (a b : ℝ) :=
  (a^2 + b^2).sqrt = 4

-- Define the proof problem
theorem standard_equation_hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h_asymptote : condition_asymptote a b)
  (h_focus : condition_focus_hyperbola_parabola a b) :
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1) :=
sorry

end standard_equation_hyperbola_l147_147363


namespace coin_toss_problem_l147_147767

def coin_toss_sequences : Nat :=
  let T_combinations := binomial 12 5  -- 792
  let H_combinations := binomial 7 4   -- 35
  T_combinations * H_combinations      -- 27720

theorem coin_toss_problem :
  coin_toss_sequences = 27720 :=
by
  unfold coin_toss_sequences
  sorry

end coin_toss_problem_l147_147767


namespace percent_of_employed_females_l147_147866

theorem percent_of_employed_females (p e m f : ℝ) (h1 : e = 0.60 * p) (h2 : m = 0.15 * p) (h3 : f = e - m):
  (f / e) * 100 = 75 :=
by
  -- We place the proof here
  sorry

end percent_of_employed_females_l147_147866


namespace asha_remaining_money_l147_147340

-- Define the borrowed amounts, gift, and savings
def borrowed_from_brother : ℤ := 20
def borrowed_from_father : ℤ := 40
def borrowed_from_mother : ℤ := 30
def gift_from_granny : ℤ := 70
def savings : ℤ := 100

-- Total amount of money Asha has
def total_amount : ℤ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings

-- Amount spent by Asha
def amount_spent : ℤ := (3 * total_amount) / 4

-- Amount of money Asha remains with
def amount_left : ℤ := total_amount - amount_spent

-- The proof statement
theorem asha_remaining_money : amount_left = 65 := by
  sorry

end asha_remaining_money_l147_147340


namespace widow_share_l147_147440

theorem widow_share (w d s : ℝ) (h_sum : w + 5 * s + 4 * d = 8000)
  (h1 : d = 2 * w)
  (h2 : s = 3 * d) :
  w = 8000 / 39 := by
sorry

end widow_share_l147_147440


namespace ram_money_l147_147984

variable (R G K : ℕ)

theorem ram_money (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : K = 2890) : R = 490 :=
by
  sorry

end ram_money_l147_147984


namespace cos_diff_to_product_l147_147898

open Real

theorem cos_diff_to_product (a b : ℝ) : 
  cos (a + b) - cos (a - b) = -2 * sin a * sin b := 
  sorry

end cos_diff_to_product_l147_147898


namespace most_likely_number_of_red_balls_l147_147995

-- Define the total number of balls and the frequency of picking red balls as given in the conditions
def total_balls : ℕ := 20
def frequency_red : ℝ := 0.8

-- State the equivalent proof problem: Prove that the most likely number of red balls is 16
theorem most_likely_number_of_red_balls : frequency_red * (total_balls : ℝ) = 16 := by
  sorry

end most_likely_number_of_red_balls_l147_147995


namespace land_for_crop_production_l147_147945

-- Conditions as Lean definitions
def total_land : ℕ := 150
def house_and_machinery : ℕ := 25
def future_expansion : ℕ := 15
def cattle_rearing : ℕ := 40

-- Proof statement defining the goal
theorem land_for_crop_production : 
  total_land - (house_and_machinery + future_expansion + cattle_rearing) = 70 := 
by
  sorry

end land_for_crop_production_l147_147945


namespace smallest_lucky_number_exists_l147_147515

theorem smallest_lucky_number_exists :
  ∃ (a b c d N: ℕ), 
  N = a^2 + b^2 ∧ 
  N = c^2 + d^2 ∧ 
  a - c = 7 ∧ 
  d - b = 13 ∧ 
  N = 545 := 
by {
  sorry
}

end smallest_lucky_number_exists_l147_147515


namespace irrational_of_sqrt_3_l147_147016

noncomputable def is_irritational (x : ℝ) : Prop :=
  ¬ (∃ p q : ℤ, q ≠ 0 ∧ x = p / q)

theorem irrational_of_sqrt_3 :
  is_irritational 0 = false ∧
  is_irritational 3.14 = false ∧
  is_irritational (-1) = false ∧
  is_irritational (Real.sqrt 3) = true := 
by
  -- Proof omitted
  sorry

end irrational_of_sqrt_3_l147_147016


namespace completion_time_l147_147879

variables {P E : ℝ}
theorem completion_time (h1 : (20 : ℝ) * P * E / 2 = D * (2.5 * P * E)) : D = 4 :=
by
  -- Given h1 as the condition
  sorry

end completion_time_l147_147879


namespace count_three_digit_with_f_l147_147955

open Nat

def f : ℕ → ℕ := sorry 

axiom f_add_add (a b : ℕ) : f (a + b) = f (f a + b)
axiom f_add_small (a b : ℕ) (h : a + b < 10) : f (a + b) = f a + f b
axiom f_10 : f 10 = 1

theorem count_three_digit_with_f (hN : ∀ n : ℕ, f 2^(3^(4^5)) = f n):
  ∃ k, k = 100 ∧ ∀ n, 100 ≤ n ∧ n < 1000 → (f n = f 2^(3^(4^5))) :=
sorry

end count_three_digit_with_f_l147_147955


namespace original_price_double_value_l147_147387

theorem original_price_double_value :
  ∃ (P : ℝ), P + 0.30 * P = 351 ∧ 2 * P = 540 :=
by
  sorry

end original_price_double_value_l147_147387


namespace circular_garden_area_l147_147885

theorem circular_garden_area (AD DB DC R : ℝ) 
  (h1 : AD = 10) 
  (h2 : DB = 10) 
  (h3 : DC = 12) 
  (h4 : AD^2 + DC^2 = R^2) : 
  π * R^2 = 244 * π := 
  by 
    sorry

end circular_garden_area_l147_147885


namespace max_value_of_quadratic_l147_147857

theorem max_value_of_quadratic :
  ∀ z : ℝ, -6*z^2 + 24*z - 12 ≤ 12 :=
by
  sorry

end max_value_of_quadratic_l147_147857


namespace repeating_decimal_to_fraction_l147_147777

theorem repeating_decimal_to_fraction : (0.3666666 : ℚ) = 11 / 30 :=
by sorry

end repeating_decimal_to_fraction_l147_147777


namespace sum_of_solutions_eq_16_l147_147628

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l147_147628


namespace pages_torn_and_sheets_calculation_l147_147283

theorem pages_torn_and_sheets_calculation : 
  (∀ (n : ℕ), (sheet_no n) = (n + 1) / 2 → (2 * (n + 1) / 2) - 1 = n ∨ 2 * (n + 1) / 2 = n) →
  let first_page := 185 in
  let last_page := 518 in
  last_page = 518 → 
  ((last_page - first_page + 1) / 2) = 167 := 
by
  sorry

end pages_torn_and_sheets_calculation_l147_147283


namespace river_width_l147_147040

theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) 
  (h1 : depth = 2) 
  (h2 : flow_rate = 4000 / 60)  -- Flow rate in meters per minute
  (h3 : volume_per_minute = 6000) :
  volume_per_minute / (flow_rate * depth) = 45 :=
by
  sorry

end river_width_l147_147040


namespace sum_of_solutions_sum_of_solutions_is_16_l147_147631

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l147_147631


namespace simplify_radical_expression_l147_147133

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l147_147133


namespace smallest_number_l147_147720

theorem smallest_number (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : (a + b + c) = 90) (h4 : b = 28) (h5 : b = c - 6) : a = 28 :=
by 
  sorry

end smallest_number_l147_147720


namespace original_number_l147_147882

theorem original_number (x : ℝ) (h : x * 1.5 = 105) : x = 70 :=
sorry

end original_number_l147_147882


namespace range_of_a_l147_147708

def quadratic_function (a x : ℝ) : ℝ := a * x ^ 2 + a * x - 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic_function a x < 0) ↔ -4 < a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l147_147708


namespace floor_length_l147_147967

theorem floor_length (tile_length tile_width : ℕ) (floor_width max_tiles : ℕ)
  (h_tile : tile_length = 25 ∧ tile_width = 16)
  (h_floor_width : floor_width = 120)
  (h_max_tiles : max_tiles = 54) :
  ∃ floor_length : ℕ, 
    (∃ num_cols num_rows : ℕ, 
      num_cols * tile_width = floor_width ∧ 
      num_cols * num_rows = max_tiles ∧ 
      num_rows * tile_length = floor_length) ∧
    floor_length = 175 := 
by
  sorry

end floor_length_l147_147967


namespace necessary_condition_of_and_is_or_l147_147202

variable (p q : Prop)

theorem necessary_condition_of_and_is_or (hpq : p ∧ q) : p ∨ q :=
by {
    sorry
}

end necessary_condition_of_and_is_or_l147_147202


namespace faucet_open_duration_l147_147546

-- Initial definitions based on conditions in the problem
def init_water : ℕ := 120
def flow_rate : ℕ := 4
def rem_water : ℕ := 20

-- The equivalent Lean 4 statement to prove
theorem faucet_open_duration (t : ℕ) (H1: init_water - rem_water = flow_rate * t) : t = 25 :=
sorry

end faucet_open_duration_l147_147546


namespace total_guitars_l147_147051

theorem total_guitars (Barbeck_guitars Steve_guitars Davey_guitars : ℕ) (h1 : Barbeck_guitars = 2 * Steve_guitars) (h2 : Davey_guitars = 3 * Barbeck_guitars) (h3 : Davey_guitars = 18) : Barbeck_guitars + Steve_guitars + Davey_guitars = 27 :=
by sorry

end total_guitars_l147_147051


namespace fraction_evaporated_l147_147360

theorem fraction_evaporated (x : ℝ) (h : (1 - x) * (1/4) = 1/6) : x = 1/3 :=
by
  sorry

end fraction_evaporated_l147_147360


namespace unit_digit_14_pow_100_l147_147165

theorem unit_digit_14_pow_100 : (14 ^ 100) % 10 = 6 :=
by
  sorry

end unit_digit_14_pow_100_l147_147165


namespace find_blue_chips_l147_147324

def num_chips_satisfies (n m : ℕ) : Prop :=
  (n > m) ∧ (n + m > 2) ∧ (n + m < 50) ∧
  (n * (n - 1) + m * (m - 1)) = 2 * n * m

theorem find_blue_chips (n : ℕ) :
  (∃ m : ℕ, num_chips_satisfies n m) → 
  n = 3 ∨ n = 6 ∨ n = 10 ∨ n = 15 ∨ n = 21 ∨ n = 28 :=
by
  sorry

end find_blue_chips_l147_147324


namespace number_of_sides_l147_147036

def side_length : ℕ := 16
def perimeter : ℕ := 80

theorem number_of_sides (h1: side_length = 16) (h2: perimeter = 80) : (perimeter / side_length = 5) :=
by
  -- Proof should be inserted here.
  sorry

end number_of_sides_l147_147036


namespace reeya_third_subject_score_l147_147522

theorem reeya_third_subject_score
  (score1 score2 score4 : ℕ)
  (avg_score : ℕ)
  (num_subjects : ℕ)
  (total_score : ℕ)
  (score3 : ℕ) :
  score1 = 65 →
  score2 = 67 →
  score4 = 85 →
  avg_score = 75 →
  num_subjects = 4 →
  total_score = avg_score * num_subjects →
  score1 + score2 + score3 + score4 = total_score →
  score3 = 83 :=
by
  intros h1 h2 h4 h5 h6 h7 h8
  sorry

end reeya_third_subject_score_l147_147522


namespace tan_rewrite_l147_147907

open Real

theorem tan_rewrite (α β : ℝ) 
  (h1 : tan (α + β) = 2 / 5)
  (h2 : tan (β - π / 4) = 1 / 4) : 
  (1 + tan α) / (1 - tan α) = 3 / 22 := 
by
  sorry

end tan_rewrite_l147_147907


namespace cos_15_eq_l147_147735

theorem cos_15_eq :
  real.cos (15 * real.pi / 180) = (real.sqrt 6 + real.sqrt 2) / 4 :=
by
  sorry

end cos_15_eq_l147_147735


namespace fourth_root_of_expression_l147_147659

theorem fourth_root_of_expression (x : ℝ) (h : 0 < x) : Real.sqrt (x^3 * Real.sqrt (x^2)) ^ (1 / 4) = x := sorry

end fourth_root_of_expression_l147_147659


namespace sum_of_solutions_sum_of_all_solutions_l147_147639

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l147_147639


namespace average_age_of_choir_l147_147847

theorem average_age_of_choir 
  (num_females : ℕ) (avg_age_females : ℝ)
  (num_males : ℕ) (avg_age_males : ℝ)
  (total_people : ℕ) (total_people_eq : total_people = num_females + num_males) :
  num_females = 12 → avg_age_females = 28 → num_males = 18 → avg_age_males = 38 → total_people = 30 →
  (num_females * avg_age_females + num_males * avg_age_males) / total_people = 34 := by
  intros
  sorry

end average_age_of_choir_l147_147847


namespace percentage_disliked_by_both_l147_147837

theorem percentage_disliked_by_both (total_comics liked_by_females liked_by_males disliked_by_both : ℕ) 
  (total_comics_eq : total_comics = 300)
  (liked_by_females_eq : liked_by_females = 30 * total_comics / 100)
  (liked_by_males_eq : liked_by_males = 120)
  (disliked_by_both_eq : disliked_by_both = total_comics - (liked_by_females + liked_by_males)) :
  (disliked_by_both * 100 / total_comics) = 30 := by
  sorry

end percentage_disliked_by_both_l147_147837


namespace inequality_C_l147_147228

variable (a b : ℝ)
variable (h : a > b)
variable (h' : b > 0)

theorem inequality_C : a + b > 2 * b := by
  sorry

end inequality_C_l147_147228


namespace intersection_area_two_circles_l147_147296

theorem intersection_area_two_circles :
  let r : ℝ := 3
  let center1 : ℝ × ℝ := (3, 0)
  let center2 : ℝ × ℝ := (0, 3)
  let intersection_area := (9 * Real.pi - 18) / 2
  (∃ x y : ℝ, (x - center1.1)^2 + y^2 = r^2 ∧ x^2 + (y - center2.2)^2 = r^2) →
  (∃ (a : ℝ), a = intersection_area) :=
by
  sorry

end intersection_area_two_circles_l147_147296


namespace paths_inequality_l147_147455
open Nat

-- Definitions
def m : ℕ := sorry -- m represents the number of rows.
def n : ℕ := sorry -- n represents the number of columns.
def N : ℕ := sorry -- N is the number of ways to color the grid such that there is a path composed of black cells from the left edge to the right edge.
def M : ℕ := sorry -- M is the number of ways to color the grid such that there are two non-intersecting paths composed of black cells from the left edge to the right edge.

-- Theorem statement
theorem paths_inequality : (N ^ 2) ≥ 2 ^ (m * n) * M := 
by
  sorry

end paths_inequality_l147_147455


namespace probability_range_l147_147388

theorem probability_range (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1)
  (h3 : (4 * p * (1 - p)^3) ≤ (6 * p^2 * (1 - p)^2)) : 
  2 / 5 ≤ p ∧ p ≤ 1 :=
by {
  sorry
}

end probability_range_l147_147388


namespace exists_function_f_l147_147521

theorem exists_function_f (f : ℕ → ℕ) : (∀ n : ℕ, f (f n) = n^2) → ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n^2 :=
sorry

end exists_function_f_l147_147521


namespace matthew_hotdogs_l147_147692

-- Definitions based on conditions
def hotdogs_ella_emma : ℕ := 2 + 2
def hotdogs_luke : ℕ := 2 * hotdogs_ella_emma
def hotdogs_hunter : ℕ := (3 * hotdogs_ella_emma) / 2  -- Multiplying by 1.5 

-- Theorem statement to prove the total number of hotdogs
theorem matthew_hotdogs : hotdogs_ella_emma + hotdogs_luke + hotdogs_hunter = 18 := by
  sorry

end matthew_hotdogs_l147_147692


namespace value_range_l147_147717

-- Step to ensure proofs about sine and real numbers are within scope
open Real

noncomputable def y (x : ℝ) : ℝ := 2 * sin x * cos x - 1

theorem value_range (x : ℝ) : -2 ≤ y x ∧ y x ≤ 0 :=
by sorry

end value_range_l147_147717


namespace area_of_sector_AOB_l147_147975

-- Definitions for the conditions
def circumference_sector_AOB : Real := 6 -- Circumference of sector AOB
def central_angle_AOB : Real := 1 -- Central angle of sector AOB

-- Theorem stating the area of the sector is 2 cm²
theorem area_of_sector_AOB (C : Real) (θ : Real) (hC : C = circumference_sector_AOB) (hθ : θ = central_angle_AOB) : 
    ∃ S : Real, S = 2 :=
by
  sorry

end area_of_sector_AOB_l147_147975


namespace train_length_is_correct_l147_147042

noncomputable def length_of_train (train_speed : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) : ℝ :=
  let speed_m_s := train_speed * (1000 / 3600)
  let total_distance := speed_m_s * time_to_cross
  total_distance - bridge_length

theorem train_length_is_correct :
  length_of_train 36 24.198064154867613 132 = 109.98064154867613 :=
by
  sorry

end train_length_is_correct_l147_147042


namespace magic_8_ball_probability_l147_147244

open ProbabilityTheory
noncomputable theory

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (choose n k) * (p^k) * ((1 - p)^(n - k))

theorem magic_8_ball_probability :
  binomial_probability 7 3 (1/3) = 560 / 2187 :=
by
  sorry

end magic_8_ball_probability_l147_147244


namespace isosceles_triangle_perimeter_l147_147209

-- Define the conditions
def isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a) ∧ (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

-- Define the side lengths
def side1 := 2
def side2 := 2
def base := 5

-- Define the perimeter
def perimeter (a b c : ℝ) := a + b + c

-- State the theorem
theorem isosceles_triangle_perimeter : isosceles_triangle side1 side2 base → perimeter side1 side2 base = 9 :=
  by sorry

end isosceles_triangle_perimeter_l147_147209


namespace power_sum_ge_three_l147_147687

theorem power_sum_ge_three {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 3) :
  a ^ a + b ^ b + c ^ c ≥ 3 :=
by
  sorry

end power_sum_ge_three_l147_147687


namespace find_distance_between_foci_l147_147719

noncomputable def distance_between_foci (pts : List (ℝ × ℝ)) : ℝ :=
  let c := (1, -1)  -- center of the ellipse
  let x1 := (1, 3)
  let x2 := (1, -5)
  let y := (7, -5)
  let b := 4       -- semi-minor axis length
  let a := 2 * Real.sqrt 13  -- semi-major axis length
  let foci_distance := 2 * Real.sqrt (a^2 - b^2)
  foci_distance

theorem find_distance_between_foci :
  distance_between_foci [(1, 3), (7, -5), (1, -5)] = 12 :=
by
  sorry

end find_distance_between_foci_l147_147719


namespace number_of_clients_l147_147021

theorem number_of_clients (num_cars num_selections_per_car num_cars_per_client total_selections num_clients : ℕ)
  (h1 : num_cars = 15)
  (h2 : num_selections_per_car = 3)
  (h3 : num_cars_per_client = 3)
  (h4 : total_selections = num_cars * num_selections_per_car)
  (h5 : num_clients = total_selections / num_cars_per_client) :
  num_clients = 15 := 
by
  sorry

end number_of_clients_l147_147021


namespace line_intersects_circle_l147_147349

theorem line_intersects_circle : 
  ∀ (x y : ℝ), 
  (2 * x + y = 0) ∧ (x^2 + y^2 + 2 * x - 4 * y - 4 = 0) ↔
    ∃ (x0 y0 : ℝ), (2 * x0 + y0 = 0) ∧ ((x0 + 1)^2 + (y0 - 2)^2 = 9) :=
by
  sorry

end line_intersects_circle_l147_147349


namespace coefficient_x4_expansion_eq_7_l147_147670

theorem coefficient_x4_expansion_eq_7 (a : ℝ) : 
  (∀ r : ℕ, 8 - (4 * r) / 3 = 4 → (a ^ r) * (Nat.choose 8 r) = 7) → a = 1 / 2 :=
by
  sorry

end coefficient_x4_expansion_eq_7_l147_147670


namespace sum_of_solutions_l147_147625
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l147_147625


namespace average_minutes_proof_l147_147037

noncomputable def average_minutes_heard (total_minutes : ℕ) (total_attendees : ℕ) (full_listened_fraction : ℚ) (none_listened_fraction : ℚ) (half_remainder_fraction : ℚ) : ℚ := 
  let full_listeners := full_listened_fraction * total_attendees
  let none_listeners := none_listened_fraction * total_attendees
  let remaining_listeners := total_attendees - full_listeners - none_listeners
  let half_listeners := half_remainder_fraction * remaining_listeners
  let quarter_listeners := remaining_listeners - half_listeners
  let total_heard := (full_listeners * total_minutes) + (none_listeners * 0) + (half_listeners * (total_minutes / 2)) + (quarter_listeners * (total_minutes / 4))
  total_heard / total_attendees

theorem average_minutes_proof : 
  average_minutes_heard 120 100 (30/100) (15/100) (40/100) = 59.1 := 
by
  sorry

end average_minutes_proof_l147_147037


namespace chocolate_bars_remaining_l147_147501

theorem chocolate_bars_remaining (total_bars sold_week1 sold_week2 : ℕ) (h_total : total_bars = 18) (h_sold1 : sold_week1 = 5) (h_sold2 : sold_week2 = 7) : total_bars - (sold_week1 + sold_week2) = 6 :=
by {
  sorry
}

end chocolate_bars_remaining_l147_147501


namespace no_solution_if_and_only_if_l147_147230

theorem no_solution_if_and_only_if (n : ℝ) : 
  ¬ ∃ (x y z : ℝ), 
    (n * x + y = 1) ∧ 
    (n * y + z = 1) ∧ 
    (x + n * z = 1) ↔ n = -1 :=
by
  sorry

end no_solution_if_and_only_if_l147_147230


namespace combinations_eq_765_l147_147803

theorem combinations_eq_765 : 
  ∃ C,
  C = (∑ a in Finset.range 7, (Nat.choose 6 a) * (Nat.choose 9 (16 - 2 * a))) ∧ 
  C = 765 :=
by
  sorry

end combinations_eq_765_l147_147803


namespace cryptarithmetic_puzzle_sol_l147_147107

theorem cryptarithmetic_puzzle_sol (A B C D : ℕ) 
  (h1 : A + B + C = D) 
  (h2 : B + C = 7) 
  (h3 : A - B = 1) : D = 9 := 
by 
  sorry

end cryptarithmetic_puzzle_sol_l147_147107


namespace circle_equation_range_l147_147269

theorem circle_equation_range (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a + 1 = 0) → a < 4 := 
by 
  sorry

end circle_equation_range_l147_147269


namespace part1_part2_part3_l147_147169

-- Definitions for the conditions
def not_divisible_by_2_or_3 (k : ℤ) : Prop :=
  ¬(k % 2 = 0 ∨ k % 3 = 0)

def form_6n1_or_6n5 (k : ℤ) : Prop :=
  ∃ (n : ℤ), k = 6 * n + 1 ∨ k = 6 * n + 5

-- Part 1
theorem part1 (k : ℤ) (h : not_divisible_by_2_or_3 k) : form_6n1_or_6n5 k :=
sorry

-- Part 2
def form_6n1 (a : ℤ) : Prop :=
  ∃ (n : ℤ), a = 6 * n + 1

def form_6n5 (a : ℤ) : Prop :=
  ∃ (n : ℤ), a = 6 * n + 5

theorem part2 (a b : ℤ) (ha : form_6n1 a ∨ form_6n5 a) (hb : form_6n1 b ∨ form_6n5 b) :
  form_6n1 (a * b) :=
sorry

-- Part 3
theorem part3 (a b : ℤ) (ha : form_6n1 a) (hb : form_6n5 b) :
  form_6n5 (a * b) :=
sorry

end part1_part2_part3_l147_147169


namespace dividend_calculation_l147_147610

theorem dividend_calculation :
  let divisor := 12
  let quotient := 909809
  let dividend := divisor * quotient
  dividend = 10917708 :=
by
  let divisor := 12
  let quotient := 909809
  let dividend := divisor * quotient
  show dividend = 10917708
  sorry

end dividend_calculation_l147_147610


namespace group_scores_analysis_l147_147032

def group1_scores : List ℕ := [92, 90, 91, 96, 96]
def group2_scores : List ℕ := [92, 96, 90, 95, 92]

def median (l : List ℕ) : ℕ := sorry
def mode (l : List ℕ) : ℕ := sorry
def mean (l : List ℕ) : ℕ := sorry
def variance (l : List ℕ) : ℕ := sorry

theorem group_scores_analysis :
  median group2_scores = 92 ∧
  mode group1_scores = 96 ∧
  mean group2_scores = 93 ∧
  variance group1_scores = 64 / 10 ∧
  variance group2_scores = 48 / 10 ∧
  variance group2_scores < variance group1_scores :=
by
  sorry

end group_scores_analysis_l147_147032


namespace work_completion_time_l147_147023

-- Definitions for work rates
def work_rate_B : ℚ := 1 / 7
def work_rate_A : ℚ := 1 / 10

-- Statement to prove
theorem work_completion_time (W : ℚ) : 
  (1 / work_rate_A + 1 / work_rate_B) = 70 / 17 := 
by 
  sorry

end work_completion_time_l147_147023


namespace sum_prime_factors_143_is_24_l147_147004

def is_not_divisible (n k : ℕ) : Prop := ¬ (n % k = 0)

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_factors_sum_143 : ℕ :=
  if is_not_divisible 143 2 ∧
     is_not_divisible 143 3 ∧
     is_not_divisible 143 5 ∧
     is_not_divisible 143 7 ∧
     (143 % 11 = 0) ∧
     (143 / 11 = 13) ∧
     is_prime 11 ∧
     is_prime 13 then 11 + 13 else 0

theorem sum_prime_factors_143_is_24 :
  prime_factors_sum_143 = 24 :=
by
  sorry

end sum_prime_factors_143_is_24_l147_147004


namespace sqrt_9_minus_2_pow_0_plus_abs_neg1_l147_147342

theorem sqrt_9_minus_2_pow_0_plus_abs_neg1 :
  (Real.sqrt 9 - 2^0 + abs (-1) = 3) :=
by
  -- Proof omitted for brevity
  sorry

end sqrt_9_minus_2_pow_0_plus_abs_neg1_l147_147342


namespace find_m_for_split_l147_147464

theorem find_m_for_split (m : ℕ) (h1 : m > 1) (h2 : ∃ k, k < m ∧ 2023 = (m^2 - m + 1) + 2*k) : m = 45 :=
sorry

end find_m_for_split_l147_147464


namespace find_number_l147_147323

theorem find_number (x : ℝ) (h : 5020 - 502 / x = 5015) : x = 100.4 :=
by
  sorry

end find_number_l147_147323


namespace number_of_papers_l147_147185

-- Define the conditions
def folded_pieces (folds : ℕ) : ℕ := 2 ^ folds
def notes_per_day : ℕ := 10
def days_per_notepad : ℕ := 4
def notes_per_notepad : ℕ := notes_per_day * days_per_notepad
def notes_per_paper (folds : ℕ) : ℕ := folded_pieces folds

-- Lean statement for the proof problem
theorem number_of_papers (folds : ℕ) (h_folds : folds = 3) :
  (notes_per_notepad / notes_per_paper folds) = 5 :=
by
  rw [h_folds]
  simp [notes_per_notepad, notes_per_paper, folded_pieces]
  sorry

end number_of_papers_l147_147185


namespace computer_price_problem_l147_147492

theorem computer_price_problem (x : ℝ) (h : x + 0.30 * x = 351) : x + 351 = 621 :=
by
  sorry

end computer_price_problem_l147_147492


namespace simplified_fraction_of_num_l147_147307

def num : ℚ := 368 / 100

theorem simplified_fraction_of_num : num = 92 / 25 := by
  sorry

end simplified_fraction_of_num_l147_147307


namespace walkway_area_296_l147_147239

theorem walkway_area_296 :
  let bed_length := 4
  let bed_width := 3
  let num_rows := 4
  let num_columns := 3
  let walkway_width := 2
  let total_bed_area := num_rows * num_columns * bed_length * bed_width
  let total_garden_width := num_columns * bed_length + (num_columns + 1) * walkway_width
  let total_garden_height := num_rows * bed_width + (num_rows + 1) * walkway_width
  let total_garden_area := total_garden_width * total_garden_height
  let total_walkway_area := total_garden_area - total_bed_area
  total_walkway_area = 296 :=
by 
  sorry

end walkway_area_296_l147_147239


namespace lateral_area_cone_l147_147235

-- Define the cone problem with given conditions
def radius : ℝ := 5
def slant_height : ℝ := 10

-- Given these conditions, prove the lateral area is 50π
theorem lateral_area_cone (r : ℝ) (l : ℝ) (h_r : r = 5) (h_l : l = 10) : (1/2) * 2 * Real.pi * r * l = 50 * Real.pi :=
by 
  -- import useful mathematical tools
  sorry

end lateral_area_cone_l147_147235


namespace shaded_area_calculation_l147_147456

noncomputable section

-- Definition of the total area of the grid
def total_area (rows columns : ℕ) : ℝ :=
  rows * columns

-- Definition of the area of a right triangle
def triangle_area (base height : ℕ) : ℝ :=
  1 / 2 * base * height

-- Definition of the shaded area in the grid
def shaded_area (total_area triangle_area : ℝ) : ℝ :=
  total_area - triangle_area

-- Theorem stating the shaded area
theorem shaded_area_calculation :
  let rows := 4
  let columns := 13
  let height := 3
  shaded_area (total_area rows columns) (triangle_area columns height) = 32.5 :=
  sorry

end shaded_area_calculation_l147_147456


namespace walking_west_is_negative_l147_147093

-- Definitions based on conditions
def east (m : Int) : Int := m
def west (m : Int) : Int := -m

-- Proof statement (no proof required, so use "sorry")
theorem walking_west_is_negative (m : Int) (h : east 8 = 8) : west 10 = -10 :=
by
  sorry

end walking_west_is_negative_l147_147093


namespace ratio_of_cube_sides_l147_147980

theorem ratio_of_cube_sides 
  (a b : ℝ) 
  (h : (6 * a^2) / (6 * b^2) = 49) :
  a / b = 7 :=
by
  sorry

end ratio_of_cube_sides_l147_147980


namespace amount_p_l147_147734

variable (P : ℚ)

/-- p has $42 more than what q and r together would have had if both q and r had 1/8 of what p has.
    We need to prove that P = 56. -/
theorem amount_p (h : P = (1/8 : ℚ) * P + (1/8) * P + 42) : P = 56 :=
by
  sorry

end amount_p_l147_147734


namespace problem_statement_l147_147706

-- Mathematical Conditions
variables (a : ℝ)

-- Sufficient but not necessary condition proof statement
def sufficient_but_not_necessary : Prop :=
  (∀ a : ℝ, a > 0 → a^2 + a ≥ 0) ∧ ¬(∀ a : ℝ, a^2 + a ≥ 0 → a > 0)

-- Main problem to be proved
theorem problem_statement : sufficient_but_not_necessary :=
by
  sorry

end problem_statement_l147_147706


namespace distance_travelled_l147_147183

theorem distance_travelled
  (d : ℝ)                   -- distance in kilometers
  (train_speed : ℝ)         -- train speed in km/h
  (ship_speed : ℝ)          -- ship speed in km/h
  (time_difference : ℝ)     -- time difference in hours
  (h1 : train_speed = 48)
  (h2 : ship_speed = 60)
  (h3 : time_difference = 2) :
  d = 480 := 
by
  sorry

end distance_travelled_l147_147183


namespace social_studies_score_l147_147264

-- Step d): Translate to Lean 4
theorem social_studies_score 
  (K E S SS : ℝ)
  (h1 : (K + E + S) / 3 = 89)
  (h2 : (K + E + S + SS) / 4 = 90) :
  SS = 93 :=
by
  -- We'll leave the mathematics formal proof details to Lean.
  sorry

end social_studies_score_l147_147264


namespace sum_items_l147_147947

theorem sum_items (A B : ℕ) (h1 : A = 585) (h2 : A = B + 249) : A + B = 921 :=
by
  -- Proof step skipped
  sorry

end sum_items_l147_147947


namespace lcm_of_4_9_10_27_l147_147726

theorem lcm_of_4_9_10_27 : Nat.lcm (Nat.lcm 4 9) (Nat.lcm 10 27) = 540 :=
by
  sorry

end lcm_of_4_9_10_27_l147_147726


namespace intersection_area_two_circles_l147_147297

theorem intersection_area_two_circles :
  let r : ℝ := 3
  let center1 : ℝ × ℝ := (3, 0)
  let center2 : ℝ × ℝ := (0, 3)
  let intersection_area := (9 * Real.pi - 18) / 2
  (∃ x y : ℝ, (x - center1.1)^2 + y^2 = r^2 ∧ x^2 + (y - center2.2)^2 = r^2) →
  (∃ (a : ℝ), a = intersection_area) :=
by
  sorry

end intersection_area_two_circles_l147_147297


namespace proof_problem_l147_147996

theorem proof_problem
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2010)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2009)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2010)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2009)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2010)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2009) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -1 :=
by
  sorry

end proof_problem_l147_147996


namespace solution_set_of_inequality_l147_147415

theorem solution_set_of_inequality :
  { x : ℝ | |x^2 - 3 * x| > 4 } = { x : ℝ | x < -1 ∨ x > 4 } :=
sorry

end solution_set_of_inequality_l147_147415


namespace compute_expression_l147_147732

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 + 1012^2 - 988^2 = 68000 := 
by
  sorry

end compute_expression_l147_147732


namespace real_numbers_inequality_l147_147115

theorem real_numbers_inequality (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1 / 3) * (a + b + c)^2 :=
by
  sorry

end real_numbers_inequality_l147_147115


namespace min_students_in_class_l147_147811

theorem min_students_in_class (b g : ℕ) (hb : 3 * b = 4 * g) : b + g = 7 :=
sorry

end min_students_in_class_l147_147811


namespace age_difference_l147_147904

theorem age_difference (a1 a2 a3 a4 x y : ℕ) 
  (h1 : (a1 + a2 + a3 + a4 + x) / 5 = 28)
  (h2 : ((a1 + 1) + (a2 + 1) + (a3 + 1) + (a4 + 1) + y) / 5 = 30) : 
  y - (x + 1) = 5 := 
by
  sorry

end age_difference_l147_147904


namespace time_after_hours_l147_147141

def current_time := 9
def total_hours := 2023
def clock_cycle := 12

theorem time_after_hours : (current_time + total_hours) % clock_cycle = 8 := by
  sorry

end time_after_hours_l147_147141


namespace problem_statement_l147_147799

def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1 else sequence (n - 1) / (sequence (n - 1) + 1)

noncomputable def sum_of_squares (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), (sequence i) ^ 2

theorem problem_statement : (⌊ sum_of_squares 2017 ⌋ = 1) :=
  sorry

end problem_statement_l147_147799


namespace nicky_profit_l147_147828

theorem nicky_profit (value_traded_away value_received : ℤ)
  (h1 : value_traded_away = 2 * 8)
  (h2 : value_received = 21) :
  value_received - value_traded_away = 5 :=
by
  sorry

end nicky_profit_l147_147828


namespace male_contestants_count_l147_147748

-- Define the total number of contestants.
def total_contestants : ℕ := 18

-- A third of the contestants are female.
def female_contestants : ℕ := total_contestants / 3

-- The rest are male.
def male_contestants : ℕ := total_contestants - female_contestants

-- The theorem to prove
theorem male_contestants_count : male_contestants = 12 := by
  -- Proof goes here.
  sorry

end male_contestants_count_l147_147748


namespace geometric_series_sum_l147_147585

theorem geometric_series_sum (a r : ℚ) (ha : a = 1) (hr : r = 1/4) : 
  (∑' n:ℕ, a * r^n) = 4/3 :=
by
  rw [ha, hr]
  sorry

end geometric_series_sum_l147_147585


namespace scientific_notation_example_l147_147103

theorem scientific_notation_example : (8485000 : ℝ) = 8.485 * 10 ^ 6 := 
by 
  sorry

end scientific_notation_example_l147_147103


namespace dave_files_left_l147_147056

theorem dave_files_left 
  (initial_apps : ℕ) 
  (initial_files : ℕ) 
  (apps_left : ℕ)
  (files_more_than_apps : ℕ) 
  (h1 : initial_apps = 11) 
  (h2 : initial_files = 3) 
  (h3 : apps_left = 2)
  (h4 : files_more_than_apps = 22) 
  : ∃ (files_left : ℕ), files_left = apps_left + files_more_than_apps :=
by
  use 24
  sorry

end dave_files_left_l147_147056


namespace capsules_per_bottle_l147_147976

-- Translating conditions into Lean definitions
def days := 180
def daily_serving_size := 2
def total_bottles := 6
def total_capsules_required := days * daily_serving_size

-- The statement to prove
theorem capsules_per_bottle : total_capsules_required / total_bottles = 60 :=
by
  sorry

end capsules_per_bottle_l147_147976


namespace percent_in_second_part_l147_147661

-- Defining the conditions and the proof statement
theorem percent_in_second_part (x y P : ℝ) 
  (h1 : 0.25 * (x - y) = (P / 100) * (x + y))
  (h2 : y = 0.25 * x) : 
  P = 15 :=
by
  sorry

end percent_in_second_part_l147_147661


namespace total_spending_l147_147596

-- Conditions used as definitions
def price_pants : ℝ := 110.00
def discount_pants : ℝ := 0.30
def number_of_pants : ℕ := 4

def price_socks : ℝ := 60.00
def discount_socks : ℝ := 0.30
def number_of_socks : ℕ := 2

-- Lean 4 statement to prove the total spending
theorem total_spending :
  (number_of_pants : ℝ) * (price_pants * (1 - discount_pants)) +
  (number_of_socks : ℝ) * (price_socks * (1 - discount_socks)) = 392.00 :=
by
  sorry

end total_spending_l147_147596


namespace megan_initial_cupcakes_l147_147693

noncomputable def initial_cupcakes (packages : Nat) (cupcakes_per_package : Nat) (cupcakes_eaten : Nat) : Nat :=
  packages * cupcakes_per_package + cupcakes_eaten

theorem megan_initial_cupcakes (packages : Nat) (cupcakes_per_package : Nat) (cupcakes_eaten : Nat) :
  packages = 4 → cupcakes_per_package = 7 → cupcakes_eaten = 43 →
  initial_cupcakes packages cupcakes_per_package cupcakes_eaten = 71 :=
by
  intros
  simp [initial_cupcakes]
  sorry

end megan_initial_cupcakes_l147_147693


namespace work_done_together_in_six_days_l147_147575

theorem work_done_together_in_six_days (A B : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 18) :
  1 / (A + B) = 6 :=
by
  sorry

end work_done_together_in_six_days_l147_147575


namespace linear_function_no_pass_quadrant_I_l147_147710

theorem linear_function_no_pass_quadrant_I (x y : ℝ) (h : y = -2 * x - 1) : 
  ¬ (0 < x ∧ 0 < y) :=
by 
  sorry

end linear_function_no_pass_quadrant_I_l147_147710


namespace trigonometric_identities_l147_147905

theorem trigonometric_identities (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : Real.sin α = 4 / 5) :
    (Real.tan α = 4 / 3) ∧ 
    ((Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) :=
by
  sorry

end trigonometric_identities_l147_147905


namespace sandy_grew_6_carrots_l147_147405

theorem sandy_grew_6_carrots (sam_grew : ℕ) (total_grew : ℕ) (h1 : sam_grew = 3) (h2 : total_grew = 9) : ∃ sandy_grew : ℕ, sandy_grew = total_grew - sam_grew ∧ sandy_grew = 6 :=
by
  sorry

end sandy_grew_6_carrots_l147_147405


namespace max_wickets_in_innings_l147_147163

-- Define the max wickets a bowler can take per over
def max_wickets_per_over : ℕ := 3

-- Define the number of overs bowled by the bowler
def overs_bowled : ℕ := 6

-- Assume the total players in a cricket team
def total_players : ℕ := 11

-- Lean statement that proves the maximum number of wickets the bowler can take in an innings
theorem max_wickets_in_innings :
  3 * 6 ≥ total_players - 1 →
  max_wickets_per_over * overs_bowled ≥ total_players - 1 :=
by
  sorry

end max_wickets_in_innings_l147_147163


namespace probability_no_shaded_rectangle_l147_147738

theorem probability_no_shaded_rectangle :
  let n := (1002 * 1001) / 2
  let m := 501 * 501
  (1 - (m / n) = 500 / 1001) := sorry

end probability_no_shaded_rectangle_l147_147738


namespace problem_l147_147530

noncomputable def f (x : ℝ) : ℝ := 5 * x - 7
noncomputable def g (x : ℝ) : ℝ := x / 5 + 3

theorem problem : ∀ x : ℝ, f (g x) - g (f x) = 6.4 :=
by
  intro x
  sorry

end problem_l147_147530


namespace pentagon_area_l147_147193

theorem pentagon_area {a b c d e : ℕ} (split: ℕ) (non_parallel1 non_parallel2 parallel1 parallel2 : ℕ)
  (h1 : a = 16) (h2 : b = 25) (h3 : c = 30) (h4 : d = 26) (h5 : e = 25)
  (split_condition : a + b + c + d + e = 5 * split)
  (np_condition1: non_parallel1 = c) (np_condition2: non_parallel2 = a)
  (p_condition1: parallel1 = d) (p_condition2: parallel2 = e)
  (area_triangle: 1 / 2 * b * a = 200)
  (area_trapezoid: 1 / 2 * (parallel1 + parallel2) * non_parallel1 = 765) :
  a + b + c + d + e = 965 := by
  sorry

end pentagon_area_l147_147193


namespace male_contestants_count_l147_147747

-- Define the total number of contestants.
def total_contestants : ℕ := 18

-- A third of the contestants are female.
def female_contestants : ℕ := total_contestants / 3

-- The rest are male.
def male_contestants : ℕ := total_contestants - female_contestants

-- The theorem to prove
theorem male_contestants_count : male_contestants = 12 := by
  -- Proof goes here.
  sorry

end male_contestants_count_l147_147747


namespace surface_area_rectangular_solid_l147_147869

def length := 5
def width := 4
def depth := 1

def surface_area (l w d : ℕ) := 2 * (l * w) + 2 * (l * d) + 2 * (w * d)

theorem surface_area_rectangular_solid : surface_area length width depth = 58 := 
by 
sorry

end surface_area_rectangular_solid_l147_147869


namespace intersection_area_l147_147853

-- Define the square vertices
def vertex1 : (ℝ × ℝ) := (2, 8)
def vertex2 : (ℝ × ℝ) := (13, 8)
def vertex3 : (ℝ × ℝ) := (13, -3)
def vertex4 : (ℝ × ℝ) := (2, -3)  -- Derived from the conditions

-- Define the circle with center and radius
def circle_center : (ℝ × ℝ) := (2, -3)
def circle_radius : ℝ := 4

-- Define the square side length
def square_side_length : ℝ := 11  -- From vertex (2, 8) to vertex (2, -3)

-- Prove the intersection area
theorem intersection_area :
  let area := (1 / 4) * Real.pi * (circle_radius^2)
  area = 4 * Real.pi :=
by
  sorry

end intersection_area_l147_147853


namespace emptying_tank_time_l147_147182

theorem emptying_tank_time :
  let V := 30 * 12^3 -- volume of the tank in cubic inches
  let r_in := 3 -- rate of inlet pipe in cubic inches per minute
  let r_out1 := 12 -- rate of first outlet pipe in cubic inches per minute
  let r_out2 := 6 -- rate of second outlet pipe in cubic inches per minute
  let net_rate := r_out1 + r_out2 - r_in
  V / net_rate = 3456 := by
sorry

end emptying_tank_time_l147_147182


namespace largest_power_dividing_factorial_l147_147454

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 2015) : ∃ k : ℕ, (2015^k ∣ n!) ∧ k = 67 :=
by
  sorry

end largest_power_dividing_factorial_l147_147454


namespace decimal_to_fraction_simplify_l147_147315

theorem decimal_to_fraction_simplify (d : ℚ) (h : d = 3.68) : d = 92 / 25 :=
by
  rw h
  sorry

end decimal_to_fraction_simplify_l147_147315


namespace new_problem_l147_147704

theorem new_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 3 * y) / (3 * x - y) = 16 / 13 := 
by
  sorry

end new_problem_l147_147704


namespace fraction_to_percentage_l147_147932

theorem fraction_to_percentage (y : ℝ) (h : y > 0) : ((7 * y) / 20 + (3 * y) / 10) = 0.65 * y :=
by
  -- the proof steps will go here
  sorry

end fraction_to_percentage_l147_147932


namespace inequality_proof_l147_147362

open Real

theorem inequality_proof {x y : ℝ} (hx : x < 0) (hy : y < 0) : 
    (x ^ 4 / y ^ 4) + (y ^ 4 / x ^ 4) - (x ^ 2 / y ^ 2) - (y ^ 2 / x ^ 2) + (x / y) + (y / x) >= 2 := 
by
    sorry

end inequality_proof_l147_147362


namespace denny_followers_l147_147599

theorem denny_followers (initial_followers: ℕ) (new_followers_per_day: ℕ) (unfollowers_in_year: ℕ) (days_in_year: ℕ)
  (h_initial: initial_followers = 100000)
  (h_new_per_day: new_followers_per_day = 1000)
  (h_unfollowers: unfollowers_in_year = 20000)
  (h_days: days_in_year = 365):
  initial_followers + (new_followers_per_day * days_in_year) - unfollowers_in_year = 445000 :=
by
  sorry

end denny_followers_l147_147599


namespace min_value_expression_l147_147365

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2*x + y = 1) : 
  ∃ (xy_min : ℝ), xy_min = 9 ∧ (∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 2*x + y = 1 → (x + 2*y)/(x*y) ≥ xy_min) :=
sorry

end min_value_expression_l147_147365


namespace part_a_part_b_l147_147801

variable {p q n : ℕ}

-- Conditions
def coprime (a b : ℕ) : Prop := gcd a b = 1
def differ_by_more_than_one (p q : ℕ) : Prop := (q > p + 1) ∨ (p > q + 1)

-- Part (a): Prove there exists a natural number n such that p + n and q + n are not coprime
theorem part_a (coprime_pq : coprime p q) (diff : differ_by_more_than_one p q) : 
  ∃ n : ℕ, ¬ coprime (p + n) (q + n) :=
sorry

-- Part (b): Prove the smallest such n is 41 for p = 2 and q = 2023
theorem part_b (h : p = 2) (h1 : q = 2023) : 
  ∃ n : ℕ, (n = 41) ∧ (¬ coprime (2 + n) (2023 + n)) :=
sorry

end part_a_part_b_l147_147801


namespace parabola_problem_l147_147176

theorem parabola_problem (a x1 x2 y1 y2 : ℝ)
  (h1 : y1^2 = a * x1)
  (h2 : y2^2 = a * x2)
  (h3 : x1 + x2 = 8)
  (h4 : (x2 - x1)^2 + (y2 - y1)^2 = 144) : 
  a = 8 := 
sorry

end parabola_problem_l147_147176


namespace total_cost_is_15_l147_147592

def toast_cost : ℕ := 1
def egg_cost : ℕ := 3

def dale_toast : ℕ := 2
def dale_eggs : ℕ := 2

def andrew_toast : ℕ := 1
def andrew_eggs : ℕ := 2

def dale_breakfast_cost := dale_toast * toast_cost + dale_eggs * egg_cost
def andrew_breakfast_cost := andrew_toast * toast_cost + andrew_eggs * egg_cost

def total_breakfast_cost := dale_breakfast_cost + andrew_breakfast_cost

theorem total_cost_is_15 : total_breakfast_cost = 15 := by
  sorry

end total_cost_is_15_l147_147592


namespace value_of_b_div_a_l147_147660

theorem value_of_b_div_a (a b : ℝ) (h : |5 - a| + (b + 3)^2 = 0) : b / a = -3 / 5 :=
by
  sorry

end value_of_b_div_a_l147_147660


namespace four_r_eq_sum_abcd_l147_147116

theorem four_r_eq_sum_abcd (a b c d r : ℤ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : (r - a) * (r - b) * (r - c) * (r - d) = 4) : 
  4 * r = a + b + c + d :=
by 
  sorry

end four_r_eq_sum_abcd_l147_147116


namespace max_whole_nine_one_number_l147_147668

def is_non_zero_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def whole_nine_one_number (a b c d : ℕ) : Prop :=
  is_non_zero_digit a ∧ is_non_zero_digit b ∧ is_non_zero_digit c ∧ is_non_zero_digit d ∧ 
  (a + c = 9) ∧ (b = d + 1) ∧ ((2 * (2 * a + d) : ℚ) / (2 * b + c : ℚ)).denom = 1

def M (a b c d : ℕ) := 1000 * a + 100 * b + 10 * c + d

theorem max_whole_nine_one_number : 
  ∃ (a b c d : ℕ), whole_nine_one_number a b c d ∧ M a b c d = 7524 :=
begin
  sorry
end

end max_whole_nine_one_number_l147_147668


namespace problem_equivalent_l147_147806

variable (p : ℤ) 

theorem problem_equivalent (h : p = (-2023) * 100) : (-2023) * 99 = p + 2023 :=
by sorry

end problem_equivalent_l147_147806


namespace half_AB_equals_l147_147221

-- Define vectors OA and OB
def vector_OA : ℝ × ℝ := (3, 2)
def vector_OB : ℝ × ℝ := (4, 7)

-- Prove that (1 / 2) * (OB - OA) = (1 / 2, 5 / 2)
theorem half_AB_equals :
  (1 / 2 : ℝ) • ((vector_OB.1 - vector_OA.1), (vector_OB.2 - vector_OA.2)) = (1 / 2, 5 / 2) := 
  sorry

end half_AB_equals_l147_147221


namespace triangle_third_side_l147_147808

noncomputable def length_of_third_side
  (a b : ℝ) (θ : ℝ) (cosθ : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * cosθ)

theorem triangle_third_side : 
  length_of_third_side 8 15 (Real.pi / 6) (Real.cos (Real.pi / 6)) = Real.sqrt (289 - 120 * Real.sqrt 3) :=
by
  sorry

end triangle_third_side_l147_147808


namespace triangle_angle_A_l147_147933

theorem triangle_angle_A (C : ℝ) (c : ℝ) (a : ℝ) 
  (hC : C = 45) (hc : c = Real.sqrt 2) (ha : a = Real.sqrt 3) :
  (∃ A : ℝ, A = 60 ∨ A = 120) :=
by
  sorry

end triangle_angle_A_l147_147933


namespace reeya_third_subject_score_l147_147261

theorem reeya_third_subject_score (s1 s2 s3 s4 : ℝ) (average : ℝ) (num_subjects : ℝ) (total_score : ℝ) :
    s1 = 65 → s2 = 67 → s4 = 95 → average = 76.6 → num_subjects = 4 → total_score = 306.4 →
    (s1 + s2 + s3 + s4) / num_subjects = average → s3 = 79.4 :=
by
  intros h1 h2 h4 h_average h_num_subjects h_total_score h_avg_eq
  -- Proof steps can be added here
  sorry

end reeya_third_subject_score_l147_147261


namespace walking_rate_ratio_l147_147173

theorem walking_rate_ratio :
  let T := 16
  let T' := 12
  (T : ℚ) / (T' : ℚ) = (4 : ℚ) / (3 : ℚ) := 
by
  sorry

end walking_rate_ratio_l147_147173


namespace sum_prime_factors_143_is_24_l147_147003

def is_not_divisible (n k : ℕ) : Prop := ¬ (n % k = 0)

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_factors_sum_143 : ℕ :=
  if is_not_divisible 143 2 ∧
     is_not_divisible 143 3 ∧
     is_not_divisible 143 5 ∧
     is_not_divisible 143 7 ∧
     (143 % 11 = 0) ∧
     (143 / 11 = 13) ∧
     is_prime 11 ∧
     is_prime 13 then 11 + 13 else 0

theorem sum_prime_factors_143_is_24 :
  prime_factors_sum_143 = 24 :=
by
  sorry

end sum_prime_factors_143_is_24_l147_147003


namespace general_term_arithmetic_sequence_l147_147214

variable {α : Type*}
variables (a_n a : ℕ → ℕ) (d a_1 a_2 a_3 a_4 n : ℕ)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a_n : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a_n (n + 1) = a_n n + d

-- Define the inequality solution condition 
def inequality_solution_set (a_1 a_2 : ℕ) (x : ℕ) :=
  a_1 ≤ x ∧ x ≤ a_2

theorem general_term_arithmetic_sequence :
  arithmetic_sequence a_n d ∧ (d ≠ 0) ∧ 
  (∀ x, x^2 - a_3 * x + a_4 ≤ 0 ↔ inequality_solution_set a_1 a_2 x) →
  a_n = 2 * n :=
by
  sorry

end general_term_arithmetic_sequence_l147_147214


namespace geometric_seq_increasing_condition_not_sufficient_nor_necessary_l147_147105

-- Definitions based on conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = q * a n
def monotonically_increasing (a : ℕ → ℝ) := ∀ n : ℕ, a n ≤ a (n + 1)
def common_ratio_gt_one (q : ℝ) := q > 1

-- Proof statement of the problem
theorem geometric_seq_increasing_condition_not_sufficient_nor_necessary 
    (a : ℕ → ℝ) (q : ℝ) 
    (h1 : geometric_sequence a q) : 
    ¬(common_ratio_gt_one q ↔ monotonically_increasing a) :=
sorry

end geometric_seq_increasing_condition_not_sufficient_nor_necessary_l147_147105


namespace magic_8_ball_probability_l147_147246

def probability_positive (p_pos : ℚ) (questions : ℕ) (positive_responses : ℕ) : ℚ :=
  (Nat.choose questions positive_responses : ℚ) * (p_pos ^ positive_responses) * ((1 - p_pos) ^ (questions - positive_responses))

theorem magic_8_ball_probability :
  probability_positive (1/3) 7 3 = 560 / 2187 :=
by
  sorry

end magic_8_ball_probability_l147_147246


namespace positive_integer_pairs_l147_147781

theorem positive_integer_pairs (m n : ℕ) (p : ℕ) (hp_prime : Prime p) (h_diff : m - n = p) (h_square : ∃ k : ℕ, m * n = k^2) :
  ∃ p' : ℕ, (Prime p') ∧ m = (p' + 1) / 2 ^ 2 ∧ n = (p' - 1) / 2 ^ 2 :=
sorry

end positive_integer_pairs_l147_147781


namespace inradius_of_triangle_l147_147286

theorem inradius_of_triangle (P A : ℝ) (hP : P = 40) (hA : A = 50) : 
  ∃ r : ℝ, r = 2.5 ∧ A = r * (P / 2) :=
by
  sorry

end inradius_of_triangle_l147_147286


namespace range_of_b_l147_147478

noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 / 2 then (2 * x + 1) / (x ^ 2) else x + 1

def g (x : ℝ) : ℝ := x ^ 2 - 4 * x - 4

-- The main theorem to prove the range of b
theorem range_of_b (a b : ℝ) (h : f a + g b = 0) : b ∈ Set.Icc (-1) 5 := by
  sorry

end range_of_b_l147_147478


namespace avg_of_five_consecutive_from_b_l147_147359

-- Conditions
def avg_of_five_even_consecutive (a : ℕ) : ℕ := (2 * a + (2 * a + 2) + (2 * a + 4) + (2 * a + 6) + (2 * a + 8)) / 5

-- The main theorem
theorem avg_of_five_consecutive_from_b (a : ℕ) : 
  avg_of_five_even_consecutive a = 2 * a + 4 → 
  ((2 * a + 4 + (2 * a + 4 + 1) + (2 * a + 4 + 2) + (2 * a + 4 + 3) + (2 * a + 4 + 4)) / 5) = 2 * a + 6 :=
by
  sorry

end avg_of_five_consecutive_from_b_l147_147359


namespace henry_final_price_l147_147574

-- Definitions based on the conditions in the problem
def price_socks : ℝ := 5
def price_tshirt : ℝ := price_socks + 10
def price_jeans : ℝ := 2 * price_tshirt
def discount_jeans : ℝ := 0.15 * price_jeans
def discounted_price_jeans : ℝ := price_jeans - discount_jeans
def sales_tax_jeans : ℝ := 0.08 * discounted_price_jeans
def final_price_jeans : ℝ := discounted_price_jeans + sales_tax_jeans

-- Statement to prove
theorem henry_final_price : final_price_jeans = 27.54 := by
  sorry

end henry_final_price_l147_147574


namespace exists_k_square_congruent_neg_one_iff_l147_147953

theorem exists_k_square_congruent_neg_one_iff (p : ℕ) [Fact p.Prime] :
  (∃ k : ℤ, (k^2 ≡ -1 [ZMOD p])) ↔ (p = 2 ∨ p % 4 = 1) :=
sorry

end exists_k_square_congruent_neg_one_iff_l147_147953


namespace angle_BDC_is_55_l147_147031

def right_triangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] : Prop :=
  ∃ (angle_A angle_B angle_C : ℝ), angle_A + angle_B + angle_C = 180 ∧
  angle_A = 20 ∧ angle_C = 90

def bisector (B D : Type) [Inhabited B] [Inhabited D] (angle_ABC : ℝ) : Prop :=
  ∃ (angle_DBC : ℝ), angle_DBC = angle_ABC / 2

theorem angle_BDC_is_55 (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] :
  right_triangle A B C →
  bisector B D 70 →
  ∃ angle_BDC : ℝ, angle_BDC = 55 :=
by sorry

end angle_BDC_is_55_l147_147031


namespace torn_out_sheets_count_l147_147279

theorem torn_out_sheets_count :
  ∃ (sheets : ℕ), (first_page = 185 ∧
                   last_page = 518 ∧
                   pages_torn_out = last_page - first_page + 1 ∧ 
                   sheets = pages_torn_out / 2 ∧
                   sheets = 167) :=
by
  sorry

end torn_out_sheets_count_l147_147279


namespace number_of_six_digit_palindromes_l147_147378

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 100000 % 10
  let d2 := n / 10000 % 10
  let d3 := n / 1000 % 10
  let d4 := n / 100 % 10
  let d5 := n / 10 % 10
  let d6 := n % 10
  n >= 100000 ∧ n < 1000000 ∧ d1 > 0 ∧ d1 = d6 ∧ d2 = d5 ∧ d3 = d4

theorem number_of_six_digit_palindromes : 
  {n : ℕ | is_six_digit_palindrome n}.card = 900 := 
sorry

end number_of_six_digit_palindromes_l147_147378


namespace ways_to_draw_at_least_two_defective_l147_147444

-- Definitions based on the conditions of the problem
def total_products : ℕ := 100
def defective_products : ℕ := 3
def selected_products : ℕ := 5

-- Binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to prove
theorem ways_to_draw_at_least_two_defective :
  C defective_products 2 * C (total_products - defective_products) 3 + C defective_products 3 * C (total_products - defective_products) 2 =
  (C total_products selected_products - C defective_products 1 * C (total_products - defective_products) 4) :=
sorry

end ways_to_draw_at_least_two_defective_l147_147444


namespace solve_equation_l147_147988

theorem solve_equation : ∀ x : ℝ, (2 / (x + 5) = 1 / (3 * x)) → x = 1 :=
by
  intro x
  intro h
  -- The proof would go here
  sorry

end solve_equation_l147_147988


namespace functional_equation_divisibility_l147_147356

theorem functional_equation_divisibility (f : ℕ+ → ℕ+) :
  (∀ x y : ℕ+, (f x)^2 + y ∣ f y + x^2) → (∀ x : ℕ+, f x = x) :=
by
  sorry

end functional_equation_divisibility_l147_147356


namespace gcd_102_238_eq_34_l147_147557

theorem gcd_102_238_eq_34 :
  Int.gcd 102 238 = 34 :=
sorry

end gcd_102_238_eq_34_l147_147557


namespace sum_of_cubes_mod_4_l147_147117

theorem sum_of_cubes_mod_4 :
  let b := 2
  let n := 2010
  ( (n * (n + 1) / 2) ^ 2 ) % (b ^ 2) = 1 :=
by
  let b := 2
  let n := 2010
  sorry

end sum_of_cubes_mod_4_l147_147117


namespace root_implies_quadratic_eq_l147_147484

theorem root_implies_quadratic_eq (m : ℝ) (h : (m + 2) - 2 + m^2 - 2 * m - 6 = 0) : 
  2 * m^2 - m - 6 = 0 :=
sorry

end root_implies_quadratic_eq_l147_147484


namespace Cubs_home_runs_third_inning_l147_147675

variable (X : ℕ)

theorem Cubs_home_runs_third_inning 
  (h : X + 1 + 2 = 2 + 3) : 
  X = 2 :=
by 
  sorry

end Cubs_home_runs_third_inning_l147_147675


namespace p_suff_not_necess_q_l147_147467

def proposition_p (a : ℝ) : Prop := ∀ (x : ℝ), x > 0 → (3*a - 1)^x < 1
def proposition_q (a : ℝ) : Prop := a > (1 / 3)

theorem p_suff_not_necess_q : 
  (∀ (a : ℝ), proposition_p a → proposition_q a) ∧ (¬∀ (a : ℝ), proposition_q a → proposition_p a) :=
  sorry

end p_suff_not_necess_q_l147_147467


namespace no_int_solutions_for_cubic_eqn_l147_147605

theorem no_int_solutions_for_cubic_eqn :
  ¬ ∃ (m n : ℤ), m^3 = 3 * n^2 + 3 * n + 7 := by
  sorry

end no_int_solutions_for_cubic_eqn_l147_147605


namespace sum_of_roots_eq_seventeen_l147_147618

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l147_147618


namespace tan_A_minus_B_l147_147815

theorem tan_A_minus_B (A B : ℝ) (h1: Real.cos A = -Real.sqrt 2 / 2) (h2 : Real.tan B = 1 / 3) : 
  Real.tan (A - B) = -2 := by
  sorry

end tan_A_minus_B_l147_147815


namespace division_of_expressions_l147_147545

theorem division_of_expressions : 
  (2 * 3 + 4) / (2 + 3) = 2 :=
by
  sorry

end division_of_expressions_l147_147545


namespace positive_iff_sum_and_product_positive_l147_147079

theorem positive_iff_sum_and_product_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) :=
by
  sorry

end positive_iff_sum_and_product_positive_l147_147079


namespace area_ratio_of_shapes_l147_147538

theorem area_ratio_of_shapes (l w r : ℝ) (h1 : 2 * l + 2 * w = 2 * π * r) (h2 : l = 3 * w) :
  (l * w) / (π * r^2) = (3 * π) / 16 :=
by sorry

end area_ratio_of_shapes_l147_147538


namespace range_of_a_if_monotonic_l147_147480

theorem range_of_a_if_monotonic :
  (∀ x : ℝ, 1 < x ∧ x < 2 → 3 * a * x^2 - 2 * x + 1 ≥ 0) → a > 1 / 3 :=
by
  sorry

end range_of_a_if_monotonic_l147_147480


namespace Laurent_number_greater_than_Chloe_l147_147894

theorem Laurent_number_greater_than_Chloe :
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 2000) →
    let y := (λ (x : ℝ), (uniform [0, 2 * x]).sample) in
    P(y > x) = 1 / 2 :=
by
  let Chloe_distribution := uniform [0, 2000]
  let Laurent_distribution := λ (x : ℝ), uniform [0, 2 * x]
  sorry

end Laurent_number_greater_than_Chloe_l147_147894


namespace xy_divides_x2_plus_y2_plus_one_l147_147949

theorem xy_divides_x2_plus_y2_plus_one 
    (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : (x * y) ∣ (x^2 + y^2 + 1)) :
  (x^2 + y^2 + 1) / (x * y) = 3 := by
  sorry

end xy_divides_x2_plus_y2_plus_one_l147_147949


namespace total_cost_of_fruits_l147_147232

theorem total_cost_of_fruits (h_orange_weight : 12 * 2 = 24)
                             (h_apple_weight : 8 * 3.75 = 30)
                             (price_orange : ℝ := 1.5)
                             (price_apple : ℝ := 2.0) :
  (5 * 2 * price_orange + 4 * 3.75 * price_apple) = 45 :=
by
  sorry

end total_cost_of_fruits_l147_147232


namespace field_trip_savings_l147_147328

-- Define the parameters given in the conditions
def num_students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_per_month : ℕ := 4
def num_months : ℕ := 2

-- Define the weekly savings for the class
def weekly_savings : ℕ := num_students * contribution_per_student_per_week

-- Define the total weeks in the given number of months
def total_weeks : ℕ := num_months * weeks_per_month

-- Define the total savings in the given number of months
def total_savings : ℕ := weekly_savings * total_weeks

-- Now, we state the theorem
theorem field_trip_savings : total_savings = 480 :=
by {
  -- calculations are skipped
  sorry
}

end field_trip_savings_l147_147328


namespace find_x_l147_147641

theorem find_x (x : ℝ) : (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - y^2 - 4.5 = 0) → x = 1.5 := 
by 
  sorry

end find_x_l147_147641


namespace n_plus_5_divisible_by_6_l147_147038

theorem n_plus_5_divisible_by_6 (n : ℕ) (h1 : (n + 2) % 3 = 0) (h2 : (n + 3) % 4 = 0) : (n + 5) % 6 = 0 := 
sorry

end n_plus_5_divisible_by_6_l147_147038


namespace emily_age_proof_l147_147896

theorem emily_age_proof (e m : ℕ) (h1 : e = m - 18) (h2 : e + m = 54) : e = 18 :=
by
  sorry

end emily_age_proof_l147_147896


namespace max_n_minus_m_l147_147216

/-- The function defined with given parameters. -/
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

theorem max_n_minus_m (a b : ℝ) (h1 : -a / 2 = 1)
    (h2 : ∀ x, f x a b ≥ 2)
    (h3 : ∃ m n, (∀ x, f x a b ≤ 6 → m ≤ x ∧ x ≤ n) ∧ (n = 3 ∧ m = -1)) : 
    (∀ m n, (m ≤ n) → (n - m ≤ 4)) :=
by sorry

end max_n_minus_m_l147_147216


namespace total_distance_dog_runs_l147_147567

-- Define the distance between Xiaoqiang's home and his grandmother's house in meters
def distance_home_to_grandma : ℕ := 1000

-- Define Xiaoqiang's walking speed in meters per minute
def xiaoqiang_speed : ℕ := 50

-- Define the dog's running speed in meters per minute
def dog_speed : ℕ := 200

-- Define the time Xiaoqiang takes to reach his grandmother's house
def xiaoqiang_time (d : ℕ) (s : ℕ) : ℕ := d / s

-- State the total distance the dog runs given the speeds and distances
theorem total_distance_dog_runs (d x_speed dog_speed : ℕ) 
  (hx : x_speed > 0) (hd : dog_speed > 0) : (d / x_speed) * dog_speed = 4000 :=
  sorry

end total_distance_dog_runs_l147_147567


namespace height_to_top_floor_l147_147265

def total_height : ℕ := 1454
def antenna_spire_height : ℕ := 204

theorem height_to_top_floor : (total_height - antenna_spire_height) = 1250 := by
  sorry

end height_to_top_floor_l147_147265


namespace evaluate_expression_l147_147077

theorem evaluate_expression (x : ℝ) : x * (x * (x * (x - 3) - 5) + 12) + 2 = x^4 - 3 * x^3 - 5 * x^2 + 12 * x + 2 :=
by
  sorry

end evaluate_expression_l147_147077


namespace original_profit_percentage_is_10_l147_147584

-- Define the conditions and the theorem
theorem original_profit_percentage_is_10
  (original_selling_price : ℝ)
  (price_reduction: ℝ)
  (additional_profit: ℝ)
  (profit_percentage: ℝ)
  (new_profit_percentage: ℝ)
  (new_selling_price: ℝ) :
  original_selling_price = 659.9999999999994 →
  price_reduction = 0.10 →
  additional_profit = 42 →
  profit_percentage = 30 →
  new_profit_percentage = 1.30 →
  new_selling_price = original_selling_price + additional_profit →
  ((original_selling_price / (original_selling_price / (new_profit_percentage * (1 - price_reduction)))) - 1) * 100 = 10 :=
by
  sorry

end original_profit_percentage_is_10_l147_147584


namespace alpha_bound_l147_147057

theorem alpha_bound (α : ℝ) (x : ℕ → ℝ) (h_x_inc : ∀ n, x n < x (n + 1))
    (x0_one : x 0 = 1) (h_alpha : α = ∑' n, x (n + 1) / (x n)^3) :
    α ≥ 3 * Real.sqrt 3 / 2 := 
sorry

end alpha_bound_l147_147057


namespace initial_oak_trees_l147_147418

theorem initial_oak_trees (n : ℕ) (h : n - 2 = 7) : n = 9 := 
by
  sorry

end initial_oak_trees_l147_147418


namespace remainder_of_22_divided_by_3_l147_147963

theorem remainder_of_22_divided_by_3 : ∃ (r : ℕ), 22 = 3 * 7 + r ∧ r = 1 := by
  sorry

end remainder_of_22_divided_by_3_l147_147963


namespace range_of_c_l147_147470

noncomputable def p (c : ℝ) : Prop := ∀ x : ℝ, (2 * c - 1) ^ x = (2 * c - 1) ^ x

def q (c : ℝ) : Prop := ∀ x : ℝ, x + |x - 2 * c| > 1

theorem range_of_c (c : ℝ) (h1 : c > 0)
  (h2 : p c ∨ q c) (h3 : ¬ (p c ∧ q c)) : c ≥ 1 :=
sorry

end range_of_c_l147_147470


namespace election_votes_l147_147101

theorem election_votes (V : ℝ) 
  (h1 : 0.15 * V = 0.15 * V)
  (h2 : 0.85 * V = 309400 / 0.65)
  (h3 : 0.65 * (0.85 * V) = 309400) : 
  V = 560000 :=
by {
  sorry
}

end election_votes_l147_147101


namespace count_multiples_of_6_not_12_lt_300_l147_147654

theorem count_multiples_of_6_not_12_lt_300 : 
  {N : ℕ // 0 < N ∧ N < 300 ∧ (6 ∣ N) ∧ ¬(12 ∣ N)}.toFinset.card = 25 := sorry

end count_multiples_of_6_not_12_lt_300_l147_147654


namespace molecular_weight_N2O5_l147_147060

theorem molecular_weight_N2O5 :
  let atomic_weight_N := 14.01
  let atomic_weight_O := 16.00
  let molecular_weight_N2O5 := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  molecular_weight_N2O5 = 108.02 := 
by
  sorry

end molecular_weight_N2O5_l147_147060


namespace term_10_of_sequence_l147_147474

theorem term_10_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n * (2 * n + 1)) →
  (∀ n, a n = S n - S (n - 1)) →
  a 10 = 39 :=
by
  intros hS ha
  sorry

end term_10_of_sequence_l147_147474


namespace average_age_of_omi_kimiko_arlette_l147_147831

theorem average_age_of_omi_kimiko_arlette (Kimiko Omi Arlette : ℕ) (hK : Kimiko = 28) (hO : Omi = 2 * Kimiko) (hA : Arlette = (3 * Kimiko) / 4) : 
  (Omi + Kimiko + Arlette) / 3 = 35 := 
by
  sorry

end average_age_of_omi_kimiko_arlette_l147_147831


namespace rectangle_remainder_condition_l147_147948

theorem rectangle_remainder_condition
    (n a b : ℕ) (hn : 2 ≤ n)
    (ha : 1 ≤ a) (hb : 1 ≤ b) :
    (n ∣ (a - 1) ∨ n ∣ (b - 1)) ∧ (n ∣ (a + 1) ∨ n ∣ (b + 1)) :=
sorry

end rectangle_remainder_condition_l147_147948


namespace at_most_one_perfect_square_l147_147688

theorem at_most_one_perfect_square (a : ℕ → ℕ) :
  (∀ n, a (n + 1) = a n ^ 3 + 103) →
  (∃ n1, ∃ n2, a n1 = k1^2 ∧ a n2 = k2^2) → n1 = n2 
    ∨ (∀ n, a n ≠ k1^2) 
    ∨ (∀ n, a n ≠ k2^2) :=
sorry

end at_most_one_perfect_square_l147_147688


namespace angle_in_quadrant_l147_147468

-- Define the problem statement as a theorem to prove
theorem angle_in_quadrant (α : ℝ) (k : ℤ) 
  (hα : 2 * (k:ℝ) * Real.pi + Real.pi < α ∧ α < 2 * (k:ℝ) * Real.pi + 3 * Real.pi / 2) :
  (k:ℝ) * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < (k:ℝ) * Real.pi + 3 * Real.pi / 4 := 
sorry

end angle_in_quadrant_l147_147468


namespace matthew_hotdogs_l147_147691

-- Definitions based on conditions
def hotdogs_ella_emma : ℕ := 2 + 2
def hotdogs_luke : ℕ := 2 * hotdogs_ella_emma
def hotdogs_hunter : ℕ := (3 * hotdogs_ella_emma) / 2  -- Multiplying by 1.5 

-- Theorem statement to prove the total number of hotdogs
theorem matthew_hotdogs : hotdogs_ella_emma + hotdogs_luke + hotdogs_hunter = 18 := by
  sorry

end matthew_hotdogs_l147_147691


namespace right_triangle_area_l147_147393

theorem right_triangle_area (a b c : ℝ) (h1 : a + b = 21) (h2 : c = 15) (h3 : a^2 + b^2 = c^2):
  (1/2) * a * b = 54 :=
by
  sorry

end right_triangle_area_l147_147393


namespace part1_part2_l147_147499

namespace VectorProblem

def vector_a : ℝ × ℝ := (3, 2)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (4, 1)

def m := 5 / 9
def n := 8 / 9

def k := -16 / 13

-- Statement 1: Prove vectors satisfy the linear combination
theorem part1 : vector_a = (m * vector_b.1 + n * vector_c.1, m * vector_b.2 + n * vector_c.2) :=
by {
  sorry
}

-- Statement 2: Prove vectors are parallel
theorem part2 : (3 + 4 * k) * 2 + (2 + k) * 5 = 0 :=
by {
  sorry
}

end VectorProblem

end part1_part2_l147_147499


namespace count_correct_conclusions_l147_147766

structure Point where
  x : ℝ
  y : ℝ

def isDoublingPoint (P Q : Point) : Prop :=
  2 * (P.x + Q.x) = P.y + Q.y

def P1 : Point := {x := 2, y := 0}

def Q1 : Point := {x := 2, y := 8}
def Q2 : Point := {x := -3, y := -2}

def onLine (P : Point) : Prop :=
  P.y = P.x + 2

def onParabola (P : Point) : Prop :=
  P.y = P.x ^ 2 - 2 * P.x - 3

def dist (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

theorem count_correct_conclusions :
  (isDoublingPoint P1 Q1) ∧
  (isDoublingPoint P1 Q2) ∧
  (∃ A : Point, onLine A ∧ isDoublingPoint P1 A ∧ A = {x := -2, y := 0}) ∧
  (∃ B₁ B₂ : Point, onParabola B₁ ∧ onParabola B₂ ∧ isDoublingPoint P1 B₁ ∧ isDoublingPoint P1 B₂) ∧
  (∃ B : Point, isDoublingPoint P1 B ∧
   ∀ P : Point, isDoublingPoint P1 P → dist P1 P ≥ dist P1 B ∧
   dist P1 B = 8 * (5:ℝ)^(1/2) / 5) :=
by sorry

end count_correct_conclusions_l147_147766


namespace total_words_story_l147_147111

def words_per_line : ℕ := 10
def lines_per_page : ℕ := 20
def pages_filled : ℚ := 1.5
def words_left : ℕ := 100

theorem total_words_story : 
    words_per_line * lines_per_page * pages_filled + words_left = 400 := 
by
sorry

end total_words_story_l147_147111


namespace three_point_sixty_eight_as_fraction_l147_147309

theorem three_point_sixty_eight_as_fraction : 3.68 = 92 / 25 := 
by 
  sorry

end three_point_sixty_eight_as_fraction_l147_147309


namespace problem_1_problem_2_l147_147369

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + Real.log x)
def g (x : ℝ) : ℝ := x^2

theorem problem_1 (a : ℝ) (ha : a ≠ 0) : 
  (∀ (x : ℝ), f a x = a * (x + Real.log x)) →
  deriv (f a) 1 = deriv g 1 → a = 1 := 
by 
  sorry

theorem problem_2 (a : ℝ) (ha : 0 < a) (hb : a < 1) (x1 x2 : ℝ) 
  (hx1 : 1 ≤ x1) (hx2 : x2 ≤ 2) (hx12 : x1 ≠ x2) : 
  |f a x1 - f a x2| < |g x1 - g x2| := 
by 
  sorry

end problem_1_problem_2_l147_147369


namespace part1_part2_l147_147911

def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y
def B (x y : ℝ) : ℝ := x^2 - x * y + x

def difference (x y : ℝ) : ℝ := A x y - 2 * B x y

theorem part1 : difference (-2) 3 = -20 :=
by
  -- Proving that difference (-2) 3 = -20
  sorry

theorem part2 (y : ℝ) : (∀ (x : ℝ), difference x y = 2 * y) → y = 2 / 5 :=
by
  -- Proving that if difference x y is independent of x, then y = 2 / 5
  sorry

end part1_part2_l147_147911


namespace Petya_tore_out_sheets_l147_147280

theorem Petya_tore_out_sheets (n m : ℕ) (h1 : n = 185) (h2 : m = 518)
  (h3 : m.digits = n.digits) : (m - n + 1) / 2 = 167 :=
by
  sorry

end Petya_tore_out_sheets_l147_147280


namespace part1_part2_part3_l147_147909

def f (a x : ℝ) : ℝ := log a x + a * x + 1 / (x + 1)

-- Part (1)
theorem part1 (h : 2 = 2) : f 2 (1 / 4) = -7 / 10 := 
  sorry

-- Part (2)
theorem part2 (a : ℝ) (ha : 1 < a) : ∃! x : ℝ, 0 < x ∧ f a x = 0 := 
  sorry

-- Part (3)
theorem part3 (a x0 : ℝ) (ha : 1 < a) (hfx0 : f a x0 = 0) : 
  1 / 2 < f a (sqrt x0) ∧ f a (sqrt x0) < (a + 1) / 2 := 
  sorry

end part1_part2_part3_l147_147909


namespace exists_a_b_l147_147965

theorem exists_a_b (S : Finset ℕ) (hS : S.card = 43) :
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (a^2 - b^2) % 100 = 0 := 
by
  sorry

end exists_a_b_l147_147965


namespace hyperbola_eccentricity_cond_l147_147167

def hyperbola_eccentricity_condition (m : ℝ) : Prop :=
  let a := Real.sqrt m
  let b := Real.sqrt 3
  let c := Real.sqrt (m + 3)
  let e := 2
  (e * e) = (c * c) / (a * a)

theorem hyperbola_eccentricity_cond (m : ℝ) :
  hyperbola_eccentricity_condition m ↔ m = 1 :=
by
  sorry

end hyperbola_eccentricity_cond_l147_147167


namespace max_value_of_3sinx_4cosx_is_5_l147_147062

def max_value_of_function (a b : ℝ) : ℝ :=
  (sqrt (a^2 + b^2))

theorem max_value_of_3sinx_4cosx_is_5 :
  max_value_of_function 3 4 = 5 :=
by
  sorry

end max_value_of_3sinx_4cosx_is_5_l147_147062


namespace repeating_decimal_to_fraction_l147_147776

theorem repeating_decimal_to_fraction : (0.3666666 : ℚ) = 11 / 30 :=
by sorry

end repeating_decimal_to_fraction_l147_147776


namespace widgets_unloaded_l147_147317
-- We import the necessary Lean library for general mathematical purposes.

-- We begin the lean statement for our problem.
theorem widgets_unloaded (n_doo n_geegaw n_widget n_yamyam : ℕ) :
  (2^n_doo) * (11^n_geegaw) * (5^n_widget) * (7^n_yamyam) = 104350400 →
  n_widget = 2 := by
  -- Placeholder for proof
  sorry

end widgets_unloaded_l147_147317


namespace intersection_of_sets_l147_147372

noncomputable def A : Set ℤ := {x | x^2 - 1 = 0}
def B : Set ℤ := {-1, 2, 5}

theorem intersection_of_sets : A ∩ B = {-1} :=
by
  sorry

end intersection_of_sets_l147_147372


namespace pool_ratio_l147_147404

theorem pool_ratio 
  (total_pools : ℕ)
  (ark_athletic_wear_pools : ℕ)
  (total_pools_eq : total_pools = 800)
  (ark_athletic_wear_pools_eq : ark_athletic_wear_pools = 200)
  : ((total_pools - ark_athletic_wear_pools) / ark_athletic_wear_pools) = 3 :=
by
  sorry

end pool_ratio_l147_147404


namespace odd_function_decreasing_l147_147210

theorem odd_function_decreasing (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x y, x < y → y < 0 → f x > f y) :
  ∀ x y, 0 < x → x < y → f y < f x :=
by
  sorry

end odd_function_decreasing_l147_147210


namespace carl_took_4_pink_hard_hats_l147_147237

-- Define the initial number of hard hats
def initial_pink : ℕ := 26
def initial_green : ℕ := 15
def initial_yellow : ℕ := 24

-- Define the number of hard hats John took
def john_pink : ℕ := 6
def john_green : ℕ := 2 * john_pink
def john_total : ℕ := john_pink + john_green

-- Define the total initial number of hard hats
def total_initial : ℕ := initial_pink + initial_green + initial_yellow

-- Define the number of hard hats remaining after John's removal
def remaining_after_john : ℕ := total_initial - john_total

-- Define the total number of hard hats that remained in the truck
def total_remaining : ℕ := 43

-- Define the number of pink hard hats Carl took away
def carl_pink : ℕ := remaining_after_john - total_remaining

-- State the proof problem
theorem carl_took_4_pink_hard_hats : carl_pink = 4 := by
  sorry

end carl_took_4_pink_hard_hats_l147_147237


namespace dennis_total_cost_l147_147594

-- Define the cost of items and quantities
def cost_pants : ℝ := 110.0
def cost_socks : ℝ := 60.0
def quantity_pants : ℝ := 4
def quantity_socks : ℝ := 2
def discount_rate : ℝ := 0.30

-- Define the total costs before and after discount
def total_cost_pants_before_discount : ℝ := cost_pants * quantity_pants
def total_cost_socks_before_discount : ℝ := cost_socks * quantity_socks
def total_cost_before_discount : ℝ := total_cost_pants_before_discount + total_cost_socks_before_discount
def total_discount : ℝ := total_cost_before_discount * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - total_discount

-- Theorem asserting the total amount after discount
theorem dennis_total_cost : total_cost_after_discount = 392 := by 
  sorry

end dennis_total_cost_l147_147594


namespace quadratic_inequality_solution_l147_147804

open Real

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 15 < 0) : 3 < x ∧ x < 5 :=
sorry

end quadratic_inequality_solution_l147_147804


namespace problem_abc_l147_147518

theorem problem_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by
  sorry

end problem_abc_l147_147518


namespace sum_of_prime_factors_of_143_l147_147000

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_of_143 :
  let pfs : List ℕ := [11, 13] in
  (∀ p ∈ pfs, is_prime p) → pfs.sum = 24 → pfs.product = 143  :=
by
  sorry

end sum_of_prime_factors_of_143_l147_147000


namespace inequality_proof_l147_147505

theorem inequality_proof (p : ℝ) (x y z v : ℝ) (hp : p ≥ 2) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hv : v ≥ 0) :
  (x + y) ^ p + (z + v) ^ p + (x + z) ^ p + (y + v) ^ p ≤ x ^ p + y ^ p + z ^ p + v ^ p + (x + y + z + v) ^ p := 
by sorry

end inequality_proof_l147_147505


namespace curve_crossing_l147_147755

structure Point where
  x : ℝ
  y : ℝ

def curve (t : ℝ) : Point :=
  { x := 2 * t^2 - 3, y := 2 * t^4 - 9 * t^2 + 6 }

theorem curve_crossing : ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve 1 = { x := -1, y := -1 } := by
  sorry

end curve_crossing_l147_147755


namespace solve_for_y_l147_147528

theorem solve_for_y (y : ℝ) (h : 5 * (y ^ (1/3)) - 3 * (y / (y ^ (2/3))) = 10 + (y ^ (1/3))) :
  y = 1000 :=
by {
  sorry
}

end solve_for_y_l147_147528


namespace sum_of_roots_eq_seventeen_l147_147619

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l147_147619


namespace snail_total_distance_l147_147578

-- Conditions
def initial_pos : ℤ := 0
def pos1 : ℤ := 4
def pos2 : ℤ := -3
def pos3 : ℤ := 6

-- Total distance traveled by the snail
def distance_traveled : ℤ :=
  abs (pos1 - initial_pos) +
  abs (pos2 - pos1) +
  abs (pos3 - pos2)

-- Theorem statement
theorem snail_total_distance : distance_traveled = 20 :=
by
  -- Proof is omitted, as per request
  sorry

end snail_total_distance_l147_147578


namespace maximilian_wealth_greater_than_national_wealth_l147_147158

theorem maximilian_wealth_greater_than_national_wealth (x y z : ℝ) (h1 : 2 * x > z) (h2 : y < z) :
    x > (2 * x + y) - (x + z) :=
by
  sorry

end maximilian_wealth_greater_than_national_wealth_l147_147158


namespace officers_selection_count_l147_147050

theorem officers_selection_count :
  (nat.choose 20 6) - (nat.choose 12 6 + (nat.choose 8 1 * nat.choose 12 5)) = 31500 :=
by
  sorry

end officers_selection_count_l147_147050


namespace total_spending_l147_147597

-- Conditions used as definitions
def price_pants : ℝ := 110.00
def discount_pants : ℝ := 0.30
def number_of_pants : ℕ := 4

def price_socks : ℝ := 60.00
def discount_socks : ℝ := 0.30
def number_of_socks : ℕ := 2

-- Lean 4 statement to prove the total spending
theorem total_spending :
  (number_of_pants : ℝ) * (price_pants * (1 - discount_pants)) +
  (number_of_socks : ℝ) * (price_socks * (1 - discount_socks)) = 392.00 :=
by
  sorry

end total_spending_l147_147597


namespace arithmetic_sequence_sum_condition_l147_147254

noncomputable def sum_first_n_terms (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a_1 + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_sum_condition (a_1 d : ℤ) :
  sum_first_n_terms a_1 d 3 = 3 →
  sum_first_n_terms a_1 d 6 = 15 →
  (a_1 + 9 * d) + (a_1 + 10 * d) + (a_1 + 11 * d) = 30 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sum_condition_l147_147254


namespace sum_of_solutions_eqn_l147_147636

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l147_147636


namespace trapezoid_PQRS_perimeter_l147_147345

noncomputable def trapezoid_perimeter (PQ RS : ℝ) (height : ℝ) (PS QR : ℝ) : ℝ :=
  PQ + RS + PS + QR

theorem trapezoid_PQRS_perimeter :
  ∀ (PQ RS : ℝ) (height : ℝ)
  (PS QR : ℝ),
  PQ = 6 →
  RS = 10 →
  height = 5 →
  PS = Real.sqrt (5^2 + 4^2) →
  QR = Real.sqrt (5^2 + 4^2) →
  trapezoid_perimeter PQ RS height PS QR = 16 + 2 * Real.sqrt 41 :=
by
  intros
  sorry

end trapezoid_PQRS_perimeter_l147_147345


namespace minimum_occupied_seats_l147_147417

theorem minimum_occupied_seats (total_seats : ℕ) (min_empty_seats : ℕ) (occupied_seats : ℕ)
  (h1 : total_seats = 150)
  (h2 : min_empty_seats = 2)
  (h3 : occupied_seats = 2 * (total_seats / (occupied_seats + min_empty_seats + min_empty_seats)))
  : occupied_seats = 74 := by
  sorry

end minimum_occupied_seats_l147_147417


namespace n_divisible_by_100_l147_147891

theorem n_divisible_by_100 (n : ℤ) (h1 : n > 101) (h2 : 101 ∣ n)
  (h3 : ∀ d : ℤ, 1 < d ∧ d < n → d ∣ n → ∃ k m : ℤ, k ∣ n ∧ m ∣ n ∧ d = k - m) : 100 ∣ n :=
sorry

end n_divisible_by_100_l147_147891


namespace intersection_eq_singleton_l147_147491

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_eq_singleton :
  A ∩ B = {1} :=
sorry

end intersection_eq_singleton_l147_147491


namespace tangent_ellipse_hyperbola_l147_147187

-- Definitions of the curves
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y+3)^2 = 1

-- Condition for tangency: the curves must meet and the discriminant must be zero
noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Prove the given curves are tangent at some x and y for m = 8/9
theorem tangent_ellipse_hyperbola : 
    (∃ x y : ℝ, ellipse x y ∧ hyperbola x y (8 / 9)) ∧ 
    quadratic_discriminant ((8 / 9) + 9) (6 * (8 / 9)) ((-8/9) * (8 * (8/9)) - 8) = 0 :=
sorry

end tangent_ellipse_hyperbola_l147_147187


namespace torn_sheets_count_l147_147275

noncomputable def first_page_num : ℕ := 185
noncomputable def last_page_num : ℕ := 518
noncomputable def pages_per_sheet : ℕ := 2

theorem torn_sheets_count :
  last_page_num > first_page_num ∧
  last_page_num.digits = first_page_num.digits.rotate 1 ∧
  pages_per_sheet = 2 →
  (last_page_num - first_page_num + 1)/pages_per_sheet = 167 :=
by {
  sorry
}

end torn_sheets_count_l147_147275


namespace simplify_expression_l147_147131

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l147_147131


namespace functional_eq_f800_l147_147396

theorem functional_eq_f800
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y)
  (h2 : f 1000 = 6)
  : f 800 = 7.5 := by
  sorry

end functional_eq_f800_l147_147396


namespace sum_of_solutions_sum_of_all_solutions_l147_147640

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l147_147640


namespace range_of_m_l147_147087

variable {x y m : ℝ}

theorem range_of_m (hx : 0 < x) (hy : 0 < y) (h_eq : 1/x + 4/y = 1) (h_ineq : ∃ x y, x + y/4 < m^2 - 3*m) : m < -1 ∨ m > 4 :=
sorry

end range_of_m_l147_147087


namespace gage_needs_to_skate_l147_147791

noncomputable def gage_average_skating_time (d1 d2: ℕ) (t1 t2 t8: ℕ) : ℕ :=
  let total_time := (d1 * t1) + (d2 * t2) + t8
  (total_time / (d1 + d2 + 1))

theorem gage_needs_to_skate (t1 t2: ℕ) (d1 d2: ℕ) (avg: ℕ) 
  (t1_minutes: t1 = 80) (t2_minutes: t2 = 105) 
  (days1: d1 = 4) (days2: d2 = 3) (avg_goal: avg = 95) :
  gage_average_skating_time d1 d2 t1 t2 125 = avg :=
by
  sorry

end gage_needs_to_skate_l147_147791


namespace max_ballpoint_pens_l147_147325

theorem max_ballpoint_pens (x y z : ℕ) (hx : x + y + z = 15)
  (hy : 10 * x + 40 * y + 60 * z = 500) (hz : x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1) :
  x ≤ 6 :=
sorry

end max_ballpoint_pens_l147_147325


namespace probability_one_head_one_tail_l147_147565

def toss_outcomes : List (String × String) := [("head", "head"), ("head", "tail"), ("tail", "head"), ("tail", "tail")]

def favorable_outcomes (outcomes : List (String × String)) : List (String × String) :=
  outcomes.filter (fun x => (x = ("head", "tail")) ∨ (x = ("tail", "head")))

theorem probability_one_head_one_tail :
  (favorable_outcomes toss_outcomes).length / toss_outcomes.length = 1 / 2 :=
by
  -- Proof will be filled in here
  sorry

end probability_one_head_one_tail_l147_147565


namespace cashier_can_satisfy_request_l147_147153

theorem cashier_can_satisfy_request (k : ℕ) (h : k > 8) : ∃ m n : ℕ, k = 3 * m + 5 * n :=
sorry

end cashier_can_satisfy_request_l147_147153


namespace Mitzi_leftover_money_l147_147696

def A := 75
def T := 30
def F := 13
def S := 23
def R := A - (T + F + S)

theorem Mitzi_leftover_money : R = 9 := by
  sorry

end Mitzi_leftover_money_l147_147696


namespace matthew_hotdogs_needed_l147_147690

def total_hotdogs (ella_hotdogs emma_hotdogs : ℕ) (luke_multiplier hunter_multiplier : ℕ) : ℕ :=
  let total_sisters := ella_hotdogs + emma_hotdogs
  let luke := luke_multiplier * total_sisters
  let hunter := hunter_multiplier * total_sisters / 2  -- because 1.5 = 3/2 and hunter_multiplier = 3
  total_sisters + luke + hunter

theorem matthew_hotdogs_needed :
  total_hotdogs 2 2 2 3 = 18 := 
by
  -- This proof is correct given the calculations above
  sorry

end matthew_hotdogs_needed_l147_147690


namespace asha_remaining_money_l147_147338

theorem asha_remaining_money :
  let brother := 20
  let father := 40
  let mother := 30
  let granny := 70
  let savings := 100
  let total_money := brother + father + mother + granny + savings
  let spent := (3 / 4) * total_money
  let remaining := total_money - spent
  remaining = 65 :=
by
  sorry

end asha_remaining_money_l147_147338


namespace find_N_l147_147666

theorem find_N (N p q : ℝ) 
  (h1 : N / p = 4) 
  (h2 : N / q = 18) 
  (h3 : p - q = 0.5833333333333334) :
  N = 3 := 
sorry

end find_N_l147_147666


namespace denny_followers_l147_147598

theorem denny_followers (initial_followers: ℕ) (new_followers_per_day: ℕ) (unfollowers_in_year: ℕ) (days_in_year: ℕ)
  (h_initial: initial_followers = 100000)
  (h_new_per_day: new_followers_per_day = 1000)
  (h_unfollowers: unfollowers_in_year = 20000)
  (h_days: days_in_year = 365):
  initial_followers + (new_followers_per_day * days_in_year) - unfollowers_in_year = 445000 :=
by
  sorry

end denny_followers_l147_147598


namespace xiaoliang_prob_correct_l147_147293

def initial_box_setup : List (Nat × Nat) := [(1, 2), (2, 2), (3, 2), (4, 2)]

def xiaoming_draw : List Nat := [1, 1, 3]

def remaining_balls_after_xiaoming : List (Nat × Nat) := [(1, 0), (2, 2), (3, 1), (4, 2)]

def remaining_ball_count (balls : List (Nat × Nat)) : Nat :=
  balls.foldl (λ acc ⟨_, count⟩ => acc + count) 0

theorem xiaoliang_prob_correct :
  (1 : ℚ) / (remaining_ball_count remaining_balls_after_xiaoming) = 1 / 5 :=
by
  sorry

end xiaoliang_prob_correct_l147_147293


namespace new_trailers_added_l147_147549

theorem new_trailers_added (n : ℕ) :
  let original_trailers := 15
  let original_age := 12
  let years_passed := 3
  let current_total_trailers := original_trailers + n
  let current_average_age := 10
  let total_age_three_years_ago := original_trailers * original_age
  let new_trailers_age := 3
  let total_current_age := (original_trailers * (original_age + years_passed)) + (n * new_trailers_age)
  (total_current_age / current_total_trailers = current_average_age) ↔ (n = 10) :=
by
  sorry

end new_trailers_added_l147_147549


namespace rectangle_width_l147_147711

-- Conditions
def length (w : Real) : Real := 4 * w
def area (w : Real) : Real := w * length w

-- Theorem stating that the width of the rectangle is 5 inches if the area is 100 square inches
theorem rectangle_width (h : area w = 100) : w = 5 :=
sorry

end rectangle_width_l147_147711


namespace fraction_of_total_amount_l147_147026

theorem fraction_of_total_amount (p q r : ℕ) (h1 : p + q + r = 4000) (h2 : r = 1600) :
  r / (p + q + r) = 2 / 5 :=
by
  sorry

end fraction_of_total_amount_l147_147026


namespace solve_inequality_l147_147970

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if h : a > -1 then { x : ℝ | -1 < x ∧ x < a }
  else if h : a < -1 then { x : ℝ | a < x ∧ x < -1 }
  else ∅

theorem solve_inequality (x a : ℝ) :
  (x^2 + (1 - a)*x - a < 0) ↔ (
    (a > -1 → x ∈ { x : ℝ | -1 < x ∧ x < a }) ∧
    (a < -1 → x ∈ { x : ℝ | a < x ∧ x < -1 }) ∧
    (a = -1 → False)
  ) :=
sorry

end solve_inequality_l147_147970


namespace least_number_to_add_l147_147559

theorem least_number_to_add (LCM : ℕ) (a : ℕ) (x : ℕ) :
  LCM = 23 * 29 * 31 →
  a = 1076 →
  x = LCM - a →
  (a + x) % LCM = 0 :=
by
  sorry

end least_number_to_add_l147_147559


namespace problem_part_I_problem_part_II_l147_147919

noncomputable def f (x : ℝ) : ℝ := (√3 / 2) * Real.sin (2 * x) - (Real.sin (2 * x / 2))^2 + 1 / 2

-- Given the conditions and correct answer
def omega_conditions (f : ℝ → ℝ) (omega : ℝ) : Prop :=
  (f x = (√3 / 2) * Real.sin (omega * x) - (Real.sin (omega * x / 2))^2 + 1 / 2) ∧ omega > 0 ∧
  (∃ (p : ℝ), p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = π)

theorem problem_part_I : ∃ ω : ℝ, omega_conditions f ω → ω = 2 ∧ 
  ∀ k : ℤ, ∀ x, k*π - π/3 ≤ x → x ≤ k*π + π/6 →
  strict_increasing_on f (set.Icc (k*π-π/3) (k*π+π/6)) :=
sorry

theorem problem_part_II : ∀ x ∈ set.Icc (0 : ℝ) (π / 2),
  -1 / 2 ≤ f x ∧ f x ≤ 1 :=
sorry

end problem_part_I_problem_part_II_l147_147919


namespace sum_prime_factors_143_l147_147011

open Nat

theorem sum_prime_factors_143 : (11 + 13) = 24 :=
by
  have h1 : Prime 11 := by sorry
  have h2 : Prime 13 := by sorry
  have h3 : 143 = 11 * 13 := by sorry
  exact add_eq_of_eq h3 (11 + 13) 24 sorry

end sum_prime_factors_143_l147_147011


namespace probability_at_least_one_first_class_part_l147_147547

-- Define the problem constants
def total_parts : ℕ := 6
def first_class_parts : ℕ := 4
def second_class_parts : ℕ := 2
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Define the target probability
def target_probability : ℚ := 14 / 15

-- Statement of the problem as a Lean theorem
theorem probability_at_least_one_first_class_part :
  (1 - (choose second_class_parts 2 : ℚ) / (choose total_parts 2 : ℚ)) = target_probability :=
by
  -- the proof is omitted
  sorry

end probability_at_least_one_first_class_part_l147_147547


namespace solution_interval_l147_147609

noncomputable def set_of_solutions : Set ℝ :=
  {x : ℝ | 4 * x - 3 < (x - 2) ^ 2 ∧ (x - 2) ^ 2 < 6 * x - 5}

theorem solution_interval :
  set_of_solutions = {x : ℝ | 7 < x ∧ x < 9} := by
  sorry

end solution_interval_l147_147609


namespace milk_owed_l147_147508

theorem milk_owed (initial_milk : ℚ) (given_milk : ℚ) (h_initial : initial_milk = 4) (h_given : given_milk = 16 / 3) :
  initial_milk - given_milk = -4 / 3 :=
by {
  rw [h_initial, h_given],
  norm_num,
}

end milk_owed_l147_147508


namespace square_root_and_quadratic_solution_l147_147917

theorem square_root_and_quadratic_solution
  (a b : ℤ)
  (h1 : 2 * a + b = 0)
  (h2 : 3 * b + 12 = 0) :
  (2 * a - 3 * b = 16) ∧ (a * x^2 + 4 * b - 2 = 0 → x^2 = 9) :=
by {
  -- Placeholder for proof
  sorry
}

end square_root_and_quadratic_solution_l147_147917


namespace selection_representatives_count_l147_147718

theorem selection_representatives_count (boys girls : ℕ) (specific_girl specific_boy : ℕ) 
  (chinese_rep : specific_girl ∈ girls)
  (math_rep : specific_boy ∈ boys ∧ specific_boy ≠ specific_girl)
  (girls_less_than_boys : ∀ g b, g ∈ girls → b ∈ boys → specific_girl ≠ g ∧ specific_boy ≠ b) :
  ∑ (g : ℕ) (h : g < boys), binom girls 2 * binom boys 3 * perm (boys - 1) 5 + 
  ∑ (g : ℕ) (h : g < boys), (binom girls 1 * binom boys 4 + binom girls 2 * binom boys 3) * perm boys 4 = 360 := 
by
  sorry

end selection_representatives_count_l147_147718


namespace inequality_proof_l147_147517

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_proof_l147_147517


namespace chewing_gum_company_revenue_l147_147389

theorem chewing_gum_company_revenue (R : ℝ) :
  let projected_revenue := 1.25 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100 = 60 := 
by
  sorry

end chewing_gum_company_revenue_l147_147389


namespace sum_of_reciprocals_eq_one_l147_147543

theorem sum_of_reciprocals_eq_one {x y : ℝ} (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x + y = (x * y) ^ 2) : (1/x) + (1/y) = 1 :=
sorry

end sum_of_reciprocals_eq_one_l147_147543


namespace equal_volume_rect_parallelepipeds_decomposable_equal_volume_prisms_decomposable_l147_147030

-- Definition of volumes for rectangular parallelepipeds
def volume_rect_parallelepiped (a b c: ℝ) : ℝ := a * b * c

-- Definition of volumes for prisms
def volume_prism (base_area height: ℝ) : ℝ := base_area * height

-- Definition of decomposability of rectangular parallelepipeds
def decomposable_rect_parallelepipeds (a1 b1 c1 a2 b2 c2: ℝ) : Prop :=
  (volume_rect_parallelepiped a1 b1 c1) = (volume_rect_parallelepiped a2 b2 c2)

-- Lean statement for part (a)
theorem equal_volume_rect_parallelepipeds_decomposable (a1 b1 c1 a2 b2 c2: ℝ) (h: decomposable_rect_parallelepipeds a1 b1 c1 a2 b2 c2) :
  True := sorry

-- Definition of decomposability of prisms
def decomposable_prisms (base_area1 height1 base_area2 height2: ℝ) : Prop :=
  (volume_prism base_area1 height1) = (volume_prism base_area2 height2)

-- Lean statement for part (b)
theorem equal_volume_prisms_decomposable (base_area1 height1 base_area2 height2: ℝ) (h: decomposable_prisms base_area1 height1 base_area2 height2) :
  True := sorry

end equal_volume_rect_parallelepipeds_decomposable_equal_volume_prisms_decomposable_l147_147030


namespace ounces_of_wax_for_car_l147_147681

noncomputable def ounces_wax_for_SUV : ℕ := 4
noncomputable def initial_wax_amount : ℕ := 11
noncomputable def wax_spilled : ℕ := 2
noncomputable def wax_left_after_detailing : ℕ := 2
noncomputable def total_wax_used : ℕ := initial_wax_amount - wax_spilled - wax_left_after_detailing

theorem ounces_of_wax_for_car :
  (initial_wax_amount - wax_spilled - wax_left_after_detailing) - ounces_wax_for_SUV = 3 :=
by
  sorry

end ounces_of_wax_for_car_l147_147681


namespace probability_A_more_than_B_sum_m_n_l147_147701

noncomputable def prob_A_more_than_B : ℚ :=
  0.6 + 0.4 * (1 / 2) * (1 - (63 / 512))

theorem probability_A_more_than_B : prob_A_more_than_B = 779 / 1024 := sorry

theorem sum_m_n : 779 + 1024 = 1803 := sorry

end probability_A_more_than_B_sum_m_n_l147_147701


namespace complement_of_M_in_U_l147_147220

noncomputable def U : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }
noncomputable def M : Set ℝ := { y | ∃ x, x^2 + y^2 = 1 }

theorem complement_of_M_in_U :
  (U \ M) = { x | 1 < x ∧ x ≤ 3 } :=
by
  sorry

end complement_of_M_in_U_l147_147220


namespace massive_crate_chocolate_bars_l147_147441

theorem massive_crate_chocolate_bars :
  (54 * 24 * 37 = 47952) :=
by
  sorry

end massive_crate_chocolate_bars_l147_147441


namespace three_hundred_percent_of_x_equals_seventy_five_percent_of_y_l147_147226

theorem three_hundred_percent_of_x_equals_seventy_five_percent_of_y
  (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 20) : y = 80 := by
  sorry

end three_hundred_percent_of_x_equals_seventy_five_percent_of_y_l147_147226


namespace sheets_torn_out_l147_147277

-- Define the conditions as given in the problem
def first_torn_page : Nat := 185
def last_torn_page : Nat := 518
def pages_per_sheet : Nat := 2

-- Calculate the total number of pages torn out
def total_pages_torn_out : Nat :=
  last_torn_page - first_torn_page + 1

-- Calculate the number of sheets torn out
def number_of_sheets_torn_out : Nat :=
  total_pages_torn_out / pages_per_sheet

-- Prove that the number of sheets torn out is 167
theorem sheets_torn_out :
  number_of_sheets_torn_out = 167 :=
by
  unfold number_of_sheets_torn_out total_pages_torn_out
  rw [Nat.sub_add_cancel (Nat.le_of_lt (Nat.lt_of_le_of_ne
    (Nat.le_add_left _ _) (Nat.ne_of_lt (Nat.lt_add_one 184))))]
  rw [Nat.div_eq_of_lt (Nat.lt.base 333)] 
  sorry -- proof steps are omitted

end sheets_torn_out_l147_147277


namespace lottery_probability_prizes_l147_147445

theorem lottery_probability_prizes :
  let total_tickets := 3
  let first_prize_tickets := 1
  let second_prize_tickets := 1
  let non_prize_tickets := 1
  let person_a_wins_first := (2 / 3 : ℝ)
  let person_b_wins_from_remaining := (1 / 2 : ℝ)
  (2 / 3 * 1 / 2) = (1 / 3 : ℝ) := sorry

end lottery_probability_prizes_l147_147445


namespace find_minimum_a_l147_147650

theorem find_minimum_a (a x : ℤ) : 
  (x - a < 0) → 
  (x > -3 / 2) → 
  (∃ n : ℤ, ∀ y : ℤ, y ∈ {k | -1 ≤ k ∧ k ≤ n} ∧ y < a) → 
  a = 3 := sorry

end find_minimum_a_l147_147650


namespace solve_for_y_l147_147525

theorem solve_for_y : ∃ y : ℝ, (5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3)) ↔ y = 1000 := by
  sorry

end solve_for_y_l147_147525


namespace three_point_sixty_eight_as_fraction_l147_147310

theorem three_point_sixty_eight_as_fraction : 3.68 = 92 / 25 := 
by 
  sorry

end three_point_sixty_eight_as_fraction_l147_147310


namespace total_savings_in_2_months_l147_147327

def students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_in_month : ℕ := 4
def months : ℕ := 2

def total_contribution_per_week : ℕ := students * contribution_per_student_per_week
def total_weeks : ℕ := months * weeks_in_month
def total_savings : ℕ := total_contribution_per_week * total_weeks

theorem total_savings_in_2_months : total_savings = 480 := by
  -- Proof goes here
  sorry

end total_savings_in_2_months_l147_147327


namespace three_point_sixty_eight_as_fraction_l147_147308

theorem three_point_sixty_eight_as_fraction : 3.68 = 92 / 25 := 
by 
  sorry

end three_point_sixty_eight_as_fraction_l147_147308


namespace root_implies_m_values_l147_147483

theorem root_implies_m_values (m : ℝ) :
  (∃ x : ℝ, x = 1 ∧ (m + 2) * x^2 - 2 * x + m^2 - 2 * m - 6 = 0) →
  (m = 3 ∨ m = -2) :=
by
  sorry

end root_implies_m_values_l147_147483


namespace neg_ex_iff_forall_geq_0_l147_147651

theorem neg_ex_iff_forall_geq_0 :
  ¬(∃ x_0 : ℝ, x_0^2 - x_0 + 1 < 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≥ 0 :=
by
  sorry

end neg_ex_iff_forall_geq_0_l147_147651


namespace beth_total_crayons_l147_147433

theorem beth_total_crayons :
  let packs := 4
  let crayons_per_pack := 10
  let extra_crayons := 6
  packs * crayons_per_pack + extra_crayons = 46 :=
by
  let packs := 4
  let crayons_per_pack := 10
  let extra_crayons := 6
  show packs * crayons_per_pack + extra_crayons = 46
  sorry

end beth_total_crayons_l147_147433


namespace larger_number_of_ratio_and_lcm_l147_147236

theorem larger_number_of_ratio_and_lcm (x : ℕ) (h1 : (2 * x) % (5 * x) = 160) : (5 * x) = 160 := by
  sorry

end larger_number_of_ratio_and_lcm_l147_147236


namespace ratio_of_Carla_to_Cosima_l147_147122

variables (C M : ℝ)

-- Natasha has 3 times as much money as Carla
axiom h1 : 3 * C = 60

-- Carla has the same amount of money as Cosima
axiom h2 : C = M

-- Prove: the ratio of Carla's money to Cosima's money is 1:1
theorem ratio_of_Carla_to_Cosima : C / M = 1 :=
by sorry

end ratio_of_Carla_to_Cosima_l147_147122


namespace find_number_l147_147889

theorem find_number (x : ℝ) (h : (((x + 1.4) / 3 - 0.7) * 9 = 5.4)) : x = 2.5 :=
by 
  sorry

end find_number_l147_147889


namespace circle_equation_bisected_and_tangent_l147_147018

theorem circle_equation_bisected_and_tangent :
  (∃ x0 y0 r : ℝ, x0 = y0 ∧ (x0 + y0 - 2 * r) = 0 ∧ (∀ x y : ℝ, (x - x0)^2 + (y - y0)^2 = r^2 → (x - 1)^2 + (y - 1)^2 = 2)) := sorry

end circle_equation_bisected_and_tangent_l147_147018


namespace magic_8_ball_probability_l147_147245

open ProbabilityTheory
noncomputable theory

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (choose n k) * (p^k) * ((1 - p)^(n - k))

theorem magic_8_ball_probability :
  binomial_probability 7 3 (1/3) = 560 / 2187 :=
by
  sorry

end magic_8_ball_probability_l147_147245


namespace solve_for_x_l147_147934

variable (x y z a b w : ℝ)
variable (angle_DEB : ℝ)

def angle_sum_D (x y z angle_DEB : ℝ) : Prop := x + y + z + angle_DEB = 360
def angle_sum_E (a b w angle_DEB : ℝ) : Prop := a + b + w + angle_DEB = 360

theorem solve_for_x 
  (h1 : angle_sum_D x y z angle_DEB) 
  (h2 : angle_sum_E a b w angle_DEB) : 
  x = a + b + w - y - z :=
by
  -- Proof not required
  sorry

end solve_for_x_l147_147934


namespace find_larger_number_l147_147570

theorem find_larger_number (L S : ℕ) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 :=
sorry

end find_larger_number_l147_147570


namespace basketball_classes_l147_147096

theorem basketball_classes (x : ℕ) : (x * (x - 1)) / 2 = 10 :=
sorry

end basketball_classes_l147_147096


namespace min_flight_routes_l147_147376

-- Defining a problem of connecting cities with flight routes such that 
-- every city can be reached from any other city with no more than two layovers.
theorem min_flight_routes (n : ℕ) (h : n = 50) : ∃ (r : ℕ), (r = 49) ∧
  (∀ (c1 c2 : ℕ), c1 ≠ c2 → c1 < n → c2 < n → ∃ (a b : ℕ),
    a < n ∧ b < n ∧ (a = c1 ∨ a = c2) ∧ (b = c1 ∨ b = c2) ∧
    ((c1 = a ∧ c2 = b) ∨ (c1 = a ∧ b = c2) ∨ (a = c2 ∧ b = c1))) :=
by {
  sorry
}

end min_flight_routes_l147_147376


namespace intersection_of_A_and_B_l147_147646

-- Definitions of the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

-- Statement to prove the intersection of sets A and B is {3}
theorem intersection_of_A_and_B : A ∩ B = {3} :=
sorry

end intersection_of_A_and_B_l147_147646


namespace find_m_l147_147763

open Real

/-- Define Circle C1 and C2 as having the given equations
and verify their internal tangency to find the possible m values -/
theorem find_m (m : ℝ) :
  (∃ (x y : ℝ), (x - m)^2 + (y + 2)^2 = 9) ∧ 
  (∃ (x y : ℝ), (x + 1)^2 + (y - m)^2 = 4) ∧ 
  (by exact (sqrt ((m + 1)^2 + (-2 - m)^2)) = 3 - 2) → 
  m = -2 ∨ m = -1 := 
sorry -- Proof is omitted

end find_m_l147_147763


namespace find_point_B_l147_147240

def line_segment_parallel_to_x_axis (A B : (ℝ × ℝ)) : Prop :=
  A.snd = B.snd

def length_3 (A B : (ℝ × ℝ)) : Prop :=
  abs (A.fst - B.fst) = 3

theorem find_point_B (A B : (ℝ × ℝ))
  (h₁ : A = (3, 2))
  (h₂ : line_segment_parallel_to_x_axis A B)
  (h₃ : length_3 A B) :
  B = (0, 2) ∨ B = (6, 2) :=
sorry

end find_point_B_l147_147240


namespace terminating_decimal_expansion_of_7_over_72_l147_147787

theorem terminating_decimal_expansion_of_7_over_72 : (7 / 72) = 0.175 := 
sorry

end terminating_decimal_expansion_of_7_over_72_l147_147787


namespace number_of_nickels_l147_147705

-- Define the conditions
variable (m : ℕ) -- Total number of coins initially
variable (v : ℕ) -- Total value of coins initially in cents
variable (n : ℕ) -- Number of nickels

-- State the conditions in terms of mathematical equations
-- Condition 1: Average value is 25 cents
axiom avg_value_initial : v = 25 * m

-- Condition 2: Adding one half-dollar (50 cents) results in average of 26 cents
axiom avg_value_after_half_dollar : v + 50 = 26 * (m + 1)

-- Define the relationship between the number of each type of coin and the total value
-- We sum the individual products of the count of each type and their respective values
axiom total_value_definition : v = 5 * n  -- since the problem already validates with total_value == 25m

-- Question to prove
theorem number_of_nickels : n = 30 :=
by
  -- Since we are not providing proof, we will use sorry to indicate the proof is omitted
  sorry

end number_of_nickels_l147_147705


namespace range_of_a_l147_147371

-- Function definition
def f (x a : ℝ) : ℝ := -x^3 + 3 * a^2 * x - 4 * a

-- Main theorem statement
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, f x a = 0) ↔ (a ∈ Set.Ioi (Real.sqrt 2)) :=
sorry

end range_of_a_l147_147371


namespace original_price_doubled_l147_147234

variable (P : ℝ)

-- Given condition: Original price plus 20% equals 351
def price_increased (P : ℝ) : Prop :=
  P + 0.20 * P = 351

-- The goal is to prove that 2 times the original price is 585
theorem original_price_doubled (P : ℝ) (h : price_increased P) : 2 * P = 585 :=
sorry

end original_price_doubled_l147_147234


namespace gain_percent_is_150_l147_147927

variable (C S : ℝ)
variable (h : 50 * C = 20 * S)

theorem gain_percent_is_150 (h : 50 * C = 20 * S) : ((S - C) / C) * 100 = 150 :=
by
  sorry

end gain_percent_is_150_l147_147927


namespace find_num_yoYos_l147_147682

variables (x y z w : ℕ)

def stuffed_animals_frisbees_puzzles := x + y + w = 80
def total_prizes := x + y + z + w + 180 + 60
def cars_and_robots := 180 + 60 = x + y + z + w + 15

theorem find_num_yoYos 
(h1 : stuffed_animals_frisbees_puzzles x y w)
(h2 : total_prizes = 300)
(h3 : cars_and_robots x y z w) : z = 145 :=
sorry

end find_num_yoYos_l147_147682


namespace maximize_angle_distance_l147_147737

noncomputable def f (x : ℝ) : ℝ :=
  40 * x / (x * x + 500)

theorem maximize_angle_distance :
  ∃ x : ℝ, x = 10 * Real.sqrt 5 ∧ ∀ y : ℝ, y ≠ x → f y < f x :=
sorry

end maximize_angle_distance_l147_147737


namespace decimal_to_fraction_l147_147313

theorem decimal_to_fraction :
  (368 / 100 : ℚ) = (92 / 25 : ℚ) := by
  sorry

end decimal_to_fraction_l147_147313


namespace circle_equation_of_diameter_l147_147211

theorem circle_equation_of_diameter (A B : ℝ × ℝ) (hA : A = (-4, -5)) (hB : B = (6, -1)) :
  ∃ h k r : ℝ, (x - h)^2 + (y - k)^2 = r ∧ h = 1 ∧ k = -3 ∧ r = 29 := 
by
  sorry

end circle_equation_of_diameter_l147_147211


namespace inclination_angle_of_line_l147_147147

theorem inclination_angle_of_line (α : ℝ) (h_eq : ∀ x y, x - y + 1 = 0 ↔ y = x + 1) (h_range : 0 < α ∧ α < 180) :
  α = 45 :=
by
  -- α is the inclination angle satisfying tan α = 1 and 0 < α < 180
  sorry

end inclination_angle_of_line_l147_147147


namespace simplify_expression_l147_147736

theorem simplify_expression : (Real.sqrt 12 - |1 - Real.sqrt 3| + (7 + Real.pi)^0) = (Real.sqrt 3 + 2) :=
by
  sorry

end simplify_expression_l147_147736


namespace problem_solution_l147_147302

theorem problem_solution :
  (12345 * 5 + 23451 * 4 + 34512 * 3 + 45123 * 2 + 51234 * 1 = 400545) :=
by
  sorry

end problem_solution_l147_147302


namespace amount_in_paise_l147_147231

theorem amount_in_paise (a : ℝ) (h_a : a = 170) (percentage_value : ℝ) (h_percentage : percentage_value = 0.5 / 100) : 
  (percentage_value * a * 100) = 85 := 
by
  sorry

end amount_in_paise_l147_147231


namespace min_c_for_expression_not_min_abs_c_for_expression_l147_147197

theorem min_c_for_expression :
  ∀ c : ℝ,
  (c - 3)^2 + (c - 4)^2 + (c - 8)^2 ≥ (5 - 3)^2 + (5 - 4)^2 + (5 - 8)^2 := 
by sorry

theorem not_min_abs_c_for_expression :
  ∃ c : ℝ, |c - 3| + |c - 4| + |c - 8| < |5 - 3| + |5 - 4| + |5 - 8| := 
by sorry

end min_c_for_expression_not_min_abs_c_for_expression_l147_147197


namespace time_for_c_l147_147569

   variable (A B C : ℚ)

   -- Conditions
   def condition1 : Prop := (A + B = 1/6)
   def condition2 : Prop := (B + C = 1/8)
   def condition3 : Prop := (C + A = 1/12)

   -- Theorem to be proved
   theorem time_for_c (h1 : condition1 A B) (h2 : condition2 B C) (h3 : condition3 C A) :
     1 / C = 48 :=
   sorry
   
end time_for_c_l147_147569


namespace cyclist_north_speed_l147_147998

variable {v : ℝ} -- Speed of the cyclist going north.

-- Conditions: 
def speed_south := 15 -- Speed of the cyclist going south (15 kmph).
def time := 2 -- The time after which they are 50 km apart (2 hours).
def distance := 50 -- The distance they are apart after 2 hours (50 km).

-- Theorem statement:
theorem cyclist_north_speed :
    (v + speed_south) * time = distance → v = 10 := by
  intro h
  sorry

end cyclist_north_speed_l147_147998


namespace parabola_focus_distance_l147_147881

theorem parabola_focus_distance (A : ℝ × ℝ) (F : ℝ × ℝ := (1, 0)) 
    (h_parabola : A.2^2 = 4 * A.1) (h_distance : dist A F = 3) :
    A = (2, 2 * Real.sqrt 2) ∨ A = (2, -2 * Real.sqrt 2) :=
by
  sorry

end parabola_focus_distance_l147_147881


namespace total_pencils_l147_147447

def initial_pencils : ℕ := 9
def additional_pencils : ℕ := 56

theorem total_pencils : initial_pencils + additional_pencils = 65 :=
by
  -- proof steps are not required, so we use sorry
  sorry

end total_pencils_l147_147447


namespace third_median_length_l147_147043

-- Proposition stating the problem with conditions and the conclusion
theorem third_median_length (m1 m2 : ℝ) (area : ℝ) (h1 : m1 = 4) (h2 : m2 = 5) (h_area : area = 10 * Real.sqrt 3) : 
  ∃ m3 : ℝ, m3 = 3 * Real.sqrt 10 :=
by
  sorry  -- proof is not included

end third_median_length_l147_147043


namespace power_function_monotonic_incr_l147_147672

theorem power_function_monotonic_incr (m : ℝ) (h₁ : m^2 - 5 * m + 7 = 1) (h₂ : m^2 - 6 > 0) : m = 3 := 
by
  sorry

end power_function_monotonic_incr_l147_147672


namespace diaz_age_twenty_years_later_l147_147171

theorem diaz_age_twenty_years_later (D S : ℕ) (h₁ : 10 * D - 40 = 10 * S + 20) (h₂ : S = 30) : D + 20 = 56 :=
sorry

end diaz_age_twenty_years_later_l147_147171


namespace find_M_l147_147926

theorem find_M :
  (∃ M: ℕ, (10 + 11 + 12) / 3 = (2022 + 2023 + 2024) / M) → M = 551 :=
by
  sorry

end find_M_l147_147926


namespace sum_of_solutions_l147_147634

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l147_147634


namespace quadratic_inequality_solution_l147_147648

theorem quadratic_inequality_solution (a m : ℝ) (h : a < 0) :
  (∀ x : ℝ, ax^2 + 6*x - a^2 < 0 ↔ (x < 1 ∨ x > m)) → m = 2 :=
by
  sorry

end quadratic_inequality_solution_l147_147648


namespace square_area_relation_l147_147930

variable {lA lB : ℝ}

theorem square_area_relation (h : lB = 4 * lA) : lB^2 = 16 * lA^2 :=
by sorry

end square_area_relation_l147_147930


namespace sin_equals_cos_630_l147_147061

open Real

theorem sin_equals_cos_630 (n : ℤ) (h_range : -180 ≤ n ∧ n ≤ 180) (h_eq : sin (n * (π / 180)) = cos (630 * (π / 180))): 
  n = 0 ∨ n = 180 ∨ n = -180 :=
by
  sorry

end sin_equals_cos_630_l147_147061


namespace logarithmic_expression_range_l147_147196

theorem logarithmic_expression_range (a : ℝ) : 
  (a - 2 > 0) ∧ (5 - a > 0) ∧ (a - 2 ≠ 1) ↔ (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) := 
by
  sorry

end logarithmic_expression_range_l147_147196


namespace innkeeper_room_assignments_l147_147442

open Finset

theorem innkeeper_room_assignments :
  let scholars := 6
  let rooms := 6
  let max_per_room := 2
  let unoccupied := 1
  ∃ (ways : ℕ), ways = 9720 ∧
    (ways = (choose rooms (rooms - unoccupied)) * (nat.perm scholars (rooms - unoccupied)) +
            (choose rooms (rooms - unoccupied - 1)) * 
            (nat.perm scholars (rooms - unoccupied - 1)) *
            (choose (rooms - unoccupied - 1) ((rooms - unoccupied - 1) / max_per_room))) :=
sorry

end innkeeper_room_assignments_l147_147442


namespace harry_books_l147_147222

theorem harry_books : ∀ (H : ℝ), 
  (H + 2 * H + H / 2 = 175) → 
  H = 50 :=
by
  intros H h_sum
  sorry

end harry_books_l147_147222


namespace proof_problem_l147_147025

-- Define the conditions: n is a positive integer and (n(n + 1) / 3) is a square
def problem_condition (n : ℕ) : Prop :=
  ∃ m : ℕ, n > 0 ∧ (n * (n + 1)) = 3 * m^2

-- Define the proof problem: given the condition, n is a multiple of 3, n+1 and n/3 are squares
theorem proof_problem (n : ℕ) (h : problem_condition n) : 
  (∃ a : ℕ, n = 3 * a^2) ∧ 
  (∃ b : ℕ, n + 1 = b^2) ∧ 
  (∃ c : ℕ, n = 3 * c^2) :=
sorry

end proof_problem_l147_147025


namespace range_of_s_l147_147233

def double_value_point (s t : ℝ) (ht : t ≠ -1) :
  Prop := 
  ∀ k : ℝ, (t + 1) * k^2 + t * k + s = 0 →
  (t^2 - 4 * s * (t + 1) > 0)

theorem range_of_s (s t : ℝ) (ht : t ≠ -1) :
  double_value_point s t ht ↔ -1 < s ∧ s < 0 :=
sorry

end range_of_s_l147_147233


namespace solve_for_y_l147_147527

theorem solve_for_y (y : ℝ) (h : 5 * (y ^ (1/3)) - 3 * (y / (y ^ (2/3))) = 10 + (y ^ (1/3))) :
  y = 1000 :=
by {
  sorry
}

end solve_for_y_l147_147527


namespace tan_difference_l147_147906

open Real

noncomputable def tan_difference_intermediate (θ : ℝ) : ℝ :=
  (tan θ - tan (π / 4)) / (1 + tan θ * tan (π / 4))

theorem tan_difference (θ : ℝ) (h1 : cos θ = -12 / 13) (h2 : π < θ ∧ θ < 3 * π / 2) :
  tan (θ - π / 4) = -7 / 17 :=
by
  sorry

end tan_difference_l147_147906


namespace determine_some_number_l147_147983

theorem determine_some_number (x : ℝ) (n : ℝ) (hx : x = 1.5) (h : (3 + 2 * x)^5 = (1 + n * x)^4) : n = 10 / 3 :=
by {
  sorry
}

end determine_some_number_l147_147983


namespace inequality_solution_l147_147790

theorem inequality_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 6) : x^3 - 12 * x^2 + 36 * x > 0 :=
sorry

end inequality_solution_l147_147790


namespace nicky_profit_l147_147827

theorem nicky_profit (value_traded_away value_received : ℤ)
  (h1 : value_traded_away = 2 * 8)
  (h2 : value_received = 21) :
  value_received - value_traded_away = 5 :=
by
  sorry

end nicky_profit_l147_147827


namespace closure_of_M_is_closed_interval_l147_147821

noncomputable def U : Set ℝ := Set.univ

noncomputable def M : Set ℝ := {a | a^2 - 2 * a > 0}

theorem closure_of_M_is_closed_interval :
  closure M = {a | 0 ≤ a ∧ a ≤ 2} :=
by
  sorry

end closure_of_M_is_closed_interval_l147_147821


namespace max_value_fraction_l147_147665

theorem max_value_fraction (x : ℝ) : 
  ∃ (n : ℤ), n = 3 ∧ 
  ∃ (y : ℝ), y = (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ∧ 
  y ≤ n := 
sorry

end max_value_fraction_l147_147665


namespace g_84_value_l147_147978

-- Define the function g with the given conditions
def g (x : ℝ) : ℝ := sorry

-- Conditions given in the problem
axiom g_property1 : ∀ x y : ℝ, g (x * y) = y * g x
axiom g_property2 : g 2 = 48

-- Statement to prove
theorem g_84_value : g 84 = 2016 :=
by
  sorry

end g_84_value_l147_147978


namespace sum_of_solutions_eqn_l147_147635

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l147_147635


namespace a_n_formula_S_n_formula_T_n_formula_l147_147466

noncomputable def a_sequence (n : ℕ) : ℕ := 2 * n
noncomputable def S (n : ℕ) : ℕ := n * (n + 1)
noncomputable def b_sequence (n : ℕ) : ℕ := a_sequence (3 ^ n)
noncomputable def T (n : ℕ) : ℕ := 3^(n + 1) - 3

theorem a_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → a_sequence n = 2 * n :=
sorry

theorem S_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → S n = n * (n + 1) :=
sorry

theorem T_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → T n = 3^(n + 1) - 3 :=
sorry

end a_n_formula_S_n_formula_T_n_formula_l147_147466


namespace emily_subtracts_99_from_50_squared_l147_147295

theorem emily_subtracts_99_from_50_squared :
  (50 - 1) ^ 2 = 50 ^ 2 - 99 := by
  sorry

end emily_subtracts_99_from_50_squared_l147_147295


namespace original_number_of_men_l147_147019

theorem original_number_of_men (x : ℕ) 
  (h1 : 17 * x = 21 * (x - 8)) : x = 42 := 
by {
   -- proof steps can be filled in here
   sorry
}

end original_number_of_men_l147_147019


namespace asha_remaining_money_l147_147336

theorem asha_remaining_money :
  let brother := 20
  let father := 40
  let mother := 30
  let granny := 70
  let savings := 100
  let total_money := brother + father + mother + granny + savings
  let spent := (3 / 4) * total_money
  let remaining := total_money - spent
  remaining = 65 :=
by
  sorry

end asha_remaining_money_l147_147336


namespace mean_of_all_students_is_65_l147_147403

def morn_mean : ℚ := 88
def aft_mean : ℚ := 74
def eve_mean : ℚ := 80
def morn_aft_ratio : ℚ := 4 / 5
def morn_eve_ratio : ℚ := 2 / 3

theorem mean_of_all_students_is_65 :
  let m := 1     -- Assume m is 1 for simplicity of calculation
      a := 5 / 4 * m
      e := 3 / 2 * m
      total_students := m + a + e
      total_scores := morn_mean * m + aft_mean * a + eve_mean * e
  in (total_scores / total_students) = 65 := by
  sorry

end mean_of_all_students_is_65_l147_147403


namespace smallest_number_am_median_l147_147723

theorem smallest_number_am_median :
  ∃ (a b c : ℕ), a + b + c = 90 ∧ b = 28 ∧ c = b + 6 ∧ (a ≤ b ∧ b ≤ c) ∧ a = 28 :=
by
  sorry

end smallest_number_am_median_l147_147723


namespace max_profit_under_budget_max_profit_no_budget_l147_147437

-- Definitions from conditions
def sales_revenue (x1 x2 : ℝ) : ℝ :=
  -2 * x1^2 - x2^2 + 13 * x1 + 11 * x2 - 28

def profit (x1 x2 : ℝ) : ℝ :=
  sales_revenue x1 x2 - x1 - x2

-- Statements for the conditions
theorem max_profit_under_budget :
  (∀ x1 x2 : ℝ, x1 + x2 = 5 → profit x1 x2 ≤ 9) ∧
  (profit 2 3 = 9) :=
by sorry

theorem max_profit_no_budget :
  (∀ x1 x2 : ℝ, profit x1 x2 ≤ 15) ∧
  (profit 3 5 = 15) :=
by sorry

end max_profit_under_budget_max_profit_no_budget_l147_147437


namespace number_of_square_free_odd_integers_between_1_and_200_l147_147658

def count_square_free_odd_integers (a b : ℕ) (squares : List ℕ) : ℕ :=
  (b - (a + 1)) / 2 + 1 - List.foldl (λ acc sq => acc + ((b - 1) / sq).div 2 + 1) 0 squares

theorem number_of_square_free_odd_integers_between_1_and_200 :
  count_square_free_odd_integers 1 200 [9, 25, 49, 81, 121] = 81 :=
by
  apply sorry

end number_of_square_free_odd_integers_between_1_and_200_l147_147658


namespace parabola_directrix_x_eq_neg1_eqn_l147_147540

theorem parabola_directrix_x_eq_neg1_eqn :
  (∀ y : ℝ, ∃ x : ℝ, x = -1 → y^2 = 4 * x) :=
by
  sorry

end parabola_directrix_x_eq_neg1_eqn_l147_147540


namespace findSolutions_l147_147608

-- Define the given mathematical problem
def originalEquation (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 5) * (x - 4) * (x - 3)) / ((x - 4) * (x - 6) * (x - 4)) = 1

-- Define the conditions where the equation is valid
def validCondition (x : ℝ) : Prop :=
  x ≠ 4 ∧ x ≠ 6

-- Define the set of solutions
def solutions (x : ℝ) : Prop :=
  x = 4 + Real.sqrt 2 ∨ x = 4 - Real.sqrt 2

-- The theorem stating the correct set of solutions
theorem findSolutions (x : ℝ) : originalEquation x ∧ validCondition x ↔ solutions x :=
by sorry

end findSolutions_l147_147608


namespace fraction_equality_l147_147366

variable (a_n b_n : ℕ → ℝ)
variable (S_n T_n : ℕ → ℝ)

-- Conditions
axiom S_T_ratio (n : ℕ) : T_n n ≠ 0 → S_n n / T_n n = (2 * n + 1) / (4 * n - 2)
axiom Sn_def (n : ℕ) : S_n n = n / 2 * (2 * a_n 0 + (n - 1) * (a_n 1 - a_n 0))
axiom Tn_def (n : ℕ) : T_n n = n / 2 * (2 * b_n 0 + (n - 1) * (b_n 1 - b_n 0))
axiom an_def (n : ℕ) : a_n n = a_n 0 + n * (a_n 1 - a_n 0)
axiom bn_def (n : ℕ) : b_n n = b_n 0 + n * (b_n 1 - b_n 0)

-- Proof statement
theorem fraction_equality :
  (b_n 3 + b_n 18) ≠ 0 → (b_n 6 + b_n 15) ≠ 0 →
  (a_n 10 / (b_n 3 + b_n 18) + a_n 11 / (b_n 6 + b_n 15)) = (41 / 78) :=
by
  sorry

end fraction_equality_l147_147366


namespace total_water_hold_l147_147946

variables
  (first : ℕ := 100)
  (second : ℕ := 150)
  (third : ℕ := 75)
  (total : ℕ := 325)

theorem total_water_hold :
  first + second + third = total := by
  sorry

end total_water_hold_l147_147946


namespace max_value_of_trig_function_l147_147064

theorem max_value_of_trig_function : 
  ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 5 := sorry


end max_value_of_trig_function_l147_147064


namespace maximum_of_fraction_l147_147662

theorem maximum_of_fraction (x : ℝ) : (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 := by
  sorry

end maximum_of_fraction_l147_147662


namespace cross_out_number_l147_147408

theorem cross_out_number (n : ℤ) (h1 : 5 * n + 10 = 10085) : n = 2015 → (n + 5 = 2020) :=
by
  sorry

end cross_out_number_l147_147408


namespace coprime_composite_lcm_l147_147555

theorem coprime_composite_lcm (a b : ℕ) (ha : a > 1) (hb : b > 1) (hcoprime : Nat.gcd a b = 1) (hlcm : Nat.lcm a b = 120) : 
  Nat.gcd a b = 1 ∧ min a b = 8 := 
by 
  sorry

end coprime_composite_lcm_l147_147555


namespace terminating_decimal_expansion_l147_147903

theorem terminating_decimal_expansion : (15 / 625 : ℝ) = 0.024 :=
by
  -- Lean requires a justification for non-trivial facts
  -- Provide math reasoning here if necessary
  sorry

end terminating_decimal_expansion_l147_147903


namespace find_first_term_and_ratio_l147_147913

variable (b1 q : ℝ)

-- Conditions
def infinite_geometric_series (q : ℝ) : Prop := |q| < 1

def sum_odd_even_difference (b1 q : ℝ) : Prop := 
  b1 / (1 - q^2) = 2 + (b1 * q) / (1 - q^2)

def sum_square_odd_even_difference (b1 q : ℝ) : Prop :=
  b1^2 / (1 - q^4) - (b1^2 * q^2) / (1 - q^4) = 36 / 5

-- Proof problem
theorem find_first_term_and_ratio (b1 q : ℝ) 
  (h1 : infinite_geometric_series q) 
  (h2 : sum_odd_even_difference b1 q)
  (h3 : sum_square_odd_even_difference b1 q) : 
  b1 = 3 ∧ q = 1 / 2 := by
  sorry

end find_first_term_and_ratio_l147_147913


namespace number_of_elderly_employees_in_sample_l147_147742

variables (total_employees young_employees sample_young_employees elderly_employees : ℕ)
variables (sample_total : ℕ)

def conditions (total_employees young_employees sample_young_employees elderly_employees : ℕ) :=
  total_employees = 430 ∧
  young_employees = 160 ∧
  sample_young_employees = 32 ∧
  (∃ M, M = 2 * elderly_employees ∧ elderly_employees + M + young_employees = total_employees)

theorem number_of_elderly_employees_in_sample
  (total_employees young_employees sample_young_employees elderly_employees : ℕ)
  (sample_total : ℕ) :
  conditions total_employees young_employees sample_young_employees elderly_employees →
  sample_total = 430 * 32 / 160 →
  sample_total = 90 * 32 / 430 :=
by
  sorry

end number_of_elderly_employees_in_sample_l147_147742


namespace chord_segments_division_l147_147591

theorem chord_segments_division (O : Point) (r r0 : ℝ) (h : r0 < r) : 
  3 * r0 ≥ r :=
sorry

end chord_segments_division_l147_147591


namespace sum_of_acute_angles_l147_147118

open Real

theorem sum_of_acute_angles (θ₁ θ₂ : ℝ)
  (h1 : 0 < θ₁ ∧ θ₁ < π / 2)
  (h2 : 0 < θ₂ ∧ θ₂ < π / 2)
  (h_eq : (sin θ₁) ^ 2020 / (cos θ₂) ^ 2018 + (cos θ₁) ^ 2020 / (sin θ₂) ^ 2018 = 1) :
  θ₁ + θ₂ = π / 2 := sorry

end sum_of_acute_angles_l147_147118


namespace units_digit_17_pow_27_l147_147350

-- Define the problem: the units digit of 17^27
theorem units_digit_17_pow_27 : (17 ^ 27) % 10 = 3 :=
sorry

end units_digit_17_pow_27_l147_147350


namespace kabob_cubes_calculation_l147_147041

-- Define the properties of a slab of beef
def cubes_per_slab := 80
def cost_per_slab := 25

-- Define Simon's usage and expenditure
def simons_budget := 50
def number_of_kabob_sticks := 40

-- Auxiliary calculations for proofs (making noncomputable if necessary)
noncomputable def cost_per_cube := cost_per_slab / cubes_per_slab
noncomputable def cubes_per_kabob_stick := (2 * cubes_per_slab) / number_of_kabob_sticks

-- The theorem we want to prove
theorem kabob_cubes_calculation :
  cubes_per_kabob_stick = 4 := by
  sorry

end kabob_cubes_calculation_l147_147041


namespace total_volume_of_cubes_l147_147563

theorem total_volume_of_cubes (s : ℕ) (n : ℕ) (h_s : s = 5) (h_n : n = 4) : 
  n * s^3 = 500 :=
by
  sorry

end total_volume_of_cubes_l147_147563


namespace largest_natural_divisible_power_l147_147872

theorem largest_natural_divisible_power (p q : ℤ) (hp : p % 5 = 0) (hq : q % 5 = 0) (hdiscr : p^2 - 4*q > 0) :
  ∀ (α β : ℂ), (α^2 + p*α + q = 0 ∧ β^2 + p*β + q = 0) → (α^100 + β^100) % 5^50 = 0 :=
sorry

end largest_natural_divisible_power_l147_147872


namespace even_function_a_value_l147_147370

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (a * (-x)^2 + (2 * a + 1) * (-x) - 1) = (a * x^2 + (2 * a + 1) * x - 1)) →
  a = - 1 / 2 :=
by sorry

end even_function_a_value_l147_147370


namespace complex_number_quadrant_l147_147438

def inSecondQuadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_quadrant : inSecondQuadrant (i / (1 - i)) :=
by
  sorry

end complex_number_quadrant_l147_147438


namespace first_guard_hours_l147_147744

-- Define conditions
def total_hours := 9
def last_guard_hours := 2
def each_middle_guard_hours := 2

-- Define the proof problem
theorem first_guard_hours : 
  (total_hours - last_guard_hours - 2 * each_middle_guard_hours) = 3 :=
by
  -- sorry is used to skip the proof
  sorry

end first_guard_hours_l147_147744


namespace negation_universal_proposition_l147_147981

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negation_universal_proposition_l147_147981


namespace range_of_a_l147_147287
noncomputable def exponential_quadratic (a : ℝ) : Prop :=
  ∃ x : ℝ, 0 < x ∧ (1/4)^x + (1/2)^(x-1) + a = 0

theorem range_of_a (a : ℝ) : exponential_quadratic a ↔ -3 < a ∧ a < 0 :=
sorry

end range_of_a_l147_147287


namespace Mitzi_leftover_money_l147_147697

def A := 75
def T := 30
def F := 13
def S := 23
def R := A - (T + F + S)

theorem Mitzi_leftover_money : R = 9 := by
  sorry

end Mitzi_leftover_money_l147_147697


namespace direct_proportion_graph_is_straight_line_l147_147270

-- Defining the direct proportion function
def direct_proportion_function (k x : ℝ) : ℝ := k * x

-- Theorem statement
theorem direct_proportion_graph_is_straight_line (k : ℝ) :
  ∀ x : ℝ, ∃ y : ℝ, y = direct_proportion_function k x ∧ 
    ∀ (x1 x2 : ℝ), 
    ∃ a b : ℝ, b ≠ 0 ∧ 
    (a * x1 + b * (direct_proportion_function k x1)) = (a * x2 + b * (direct_proportion_function k x2)) :=
by
  sorry

end direct_proportion_graph_is_straight_line_l147_147270


namespace tim_change_l147_147550

theorem tim_change :
  ∀ (initial_amount : ℕ) (amount_paid : ℕ),
  initial_amount = 50 →
  amount_paid = 45 →
  initial_amount - amount_paid = 5 :=
by
  intros
  sorry

end tim_change_l147_147550


namespace opposite_of_2021_l147_147537

theorem opposite_of_2021 : ∃ y : ℝ, 2021 + y = 0 ∧ y = -2021 :=
by
  sorry

end opposite_of_2021_l147_147537


namespace find_m_l147_147942

noncomputable def slope_at_one (m : ℝ) := 2 + m

noncomputable def tangent_line_eq (m : ℝ) (x : ℝ) := (slope_at_one m) * x - 2 * m

noncomputable def y_intercept (m : ℝ) := tangent_line_eq m 0

noncomputable def x_intercept (m : ℝ) := - (y_intercept m) / (slope_at_one m)

noncomputable def intercept_sum_eq (m : ℝ) := (x_intercept m) + (y_intercept m)

theorem find_m (m : ℝ) (h : m ≠ -2) (h2 : intercept_sum_eq m = 12) : m = -3 ∨ m = -4 := 
sorry

end find_m_l147_147942


namespace no_solutions_for_a3_plus_5b3_eq_2016_l147_147114

theorem no_solutions_for_a3_plus_5b3_eq_2016 (a b : ℤ) : a^3 + 5 * b^3 ≠ 2016 :=
by sorry

end no_solutions_for_a3_plus_5b3_eq_2016_l147_147114


namespace intersection_of_sets_l147_147373

noncomputable def universal_set (x : ℝ) := true

def set_A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

def set_B (x : ℝ) : Prop := ∃ y, y = Real.log (1 - x)

def complement_U_B (x : ℝ) : Prop := ¬ set_B x

theorem intersection_of_sets :
  { x : ℝ | set_A x } ∩ { x | complement_U_B x } = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_of_sets_l147_147373


namespace subtraction_property_l147_147435

theorem subtraction_property : (12.56 - (5.56 - 2.63)) = (12.56 - 5.56 + 2.63) := 
by 
  sorry

end subtraction_property_l147_147435


namespace possible_values_of_c_l147_147686

-- Definition of c(S) based on the problem conditions
def c (S : String) (m : ℕ) : ℕ := sorry

-- Condition: m > 1
variable {m : ℕ} (hm : m > 1)

-- Goal: To prove the possible values that c(S) can take
theorem possible_values_of_c (S : String) : ∃ n : ℕ, c S m = 0 ∨ c S m = 2^n :=
sorry

end possible_values_of_c_l147_147686


namespace coins_after_tenth_hour_l147_147551

-- Given variables representing the number of coins added or removed each hour.
def coins_put_in : ℕ :=
  20 + 30 + 30 + 40 + 50 + 60 + 70

def coins_taken_out : ℕ :=
  20 + 15 + 25

-- Definition of the full proof problem
theorem coins_after_tenth_hour :
  coins_put_in - coins_taken_out = 240 :=
by
  sorry

end coins_after_tenth_hour_l147_147551


namespace assign_teachers_to_classes_l147_147851

theorem assign_teachers_to_classes :
  let num_classes := 6
  let num_teachers := 3
  let classes_per_teacher := 2
  (num_classes.choose classes_per_teacher) * ((num_classes - classes_per_teacher).choose classes_per_teacher) * ((num_classes - 2 * classes_per_teacher).choose classes_per_teacher) / num_teachers.factorial = 15 :=
by
  let num_classes := 6
  let num_teachers := 3
  let classes_per_teacher := 2
  have h1 : (num_classes.choose classes_per_teacher) = 15 := by sorry
  have h2 : ((num_classes - classes_per_teacher).choose classes_per_teacher) = 6 := by sorry
  have h3 : ((num_classes - 2 * classes_per_teacher).choose classes_per_teacher) = 1 := by sorry
  calc
    (num_classes.choose classes_per_teacher) * ((num_classes - classes_per_teacher).choose classes_per_teacher) * ((num_classes - 2 * classes_per_teacher).choose classes_per_teacher) / num_teachers.factorial
        = 15 * 6 * 1 / 6 : by rw [h1, h2, h3]
    ... = 15 : by norm_num

end assign_teachers_to_classes_l147_147851


namespace hyperbola_asymptotes_m_value_l147_147066

theorem hyperbola_asymptotes_m_value : 
    (∀ x y : ℝ, (x^2 / 144 - y^2 / 81 = 1) → (y = (3/4) * x ∨ y = -(3/4) * x)) := 
by sorry

end hyperbola_asymptotes_m_value_l147_147066


namespace workers_time_to_complete_job_l147_147203

theorem workers_time_to_complete_job (D E Z H k : ℝ) (h1 : 1 / D + 1 / E + 1 / Z + 1 / H = 1 / (D - 8))
  (h2 : 1 / D + 1 / E + 1 / Z + 1 / H = 1 / (E - 2))
  (h3 : 1 / D + 1 / E + 1 / Z + 1 / H = 3 / Z) :
  E = 10 → Z = 3 * (E - 2) → k = 120 / 19 :=
by
  intros hE hZ
  sorry

end workers_time_to_complete_job_l147_147203


namespace sum_of_four_consecutive_integers_is_even_l147_147419

theorem sum_of_four_consecutive_integers_is_even (n : ℤ) : 2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by sorry

end sum_of_four_consecutive_integers_is_even_l147_147419


namespace sum_of_solutions_l147_147623
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l147_147623


namespace shortest_path_octahedron_l147_147752

theorem shortest_path_octahedron 
  (edge_length : ℝ) (h : edge_length = 2) 
  (d : ℝ) : d = 2 :=
by
  sorry

end shortest_path_octahedron_l147_147752


namespace find_some_number_l147_147873

theorem find_some_number :
  ∃ (some_number : ℝ), (0.0077 * 3.6) / (some_number * 0.1 * 0.007) = 990.0000000000001 ∧ some_number = 0.04 :=
  sorry

end find_some_number_l147_147873


namespace simplify_and_evaluate_l147_147137

theorem simplify_and_evaluate (x y : ℤ) (h1 : x = -1) (h2 : y = -2) :
  ((x + y) ^ 2 - (3 * x - y) * (3 * x + y) - 2 * y ^ 2) / (-2 * x) = -2 :=
by 
  sorry

end simplify_and_evaluate_l147_147137


namespace Juan_has_498_marbles_l147_147589

def ConnieMarbles : Nat := 323
def JuanMoreMarbles : Nat := 175
def JuanMarbles : Nat := ConnieMarbles + JuanMoreMarbles

theorem Juan_has_498_marbles : JuanMarbles = 498 := by
  sorry

end Juan_has_498_marbles_l147_147589


namespace weight_of_sugar_is_16_l147_147875

def weight_of_sugar_bag (weight_of_sugar weight_of_salt remaining_weight weight_removed : ℕ) : Prop :=
  weight_of_sugar + weight_of_salt - weight_removed = remaining_weight

theorem weight_of_sugar_is_16 :
  ∃ (S : ℕ), weight_of_sugar_bag S 30 42 4 ∧ S = 16 :=
by
  sorry

end weight_of_sugar_is_16_l147_147875


namespace original_salary_l147_147985

theorem original_salary (x : ℝ)
  (h1 : x * 1.10 * 0.95 = 3135) : x = 3000 :=
by
  sorry

end original_salary_l147_147985


namespace t50_mod_7_l147_147535

def T (n : ℕ) : ℕ :=
  match n with
  | 0     => 3
  | n + 1 => 3 ^ T n

theorem t50_mod_7 : T 50 % 7 = 6 := sorry

end t50_mod_7_l147_147535


namespace ellipse_properties_l147_147476

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), a = 2 ∧ b = sqrt 3 ∧ ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

noncomputable def triangle_area : Prop :=
  ∀ (F1 F2 : ℝ×ℝ) (P : ℝ×ℝ), F1 = (-1, 0) ∧ F2 = (1, 0) ∧ 
  P ∈ { (x, y) | x^2 / 4 + y^2 / 3 = 1 } ∧
  ∃ (θ : ℝ), θ = π / 3 ∧
  let PF1 := dist P F1,
      PF2 := dist P F2 in
  ∃ (S : ℝ), S = (1 / 2) * PF1 * PF2 * sin θ ∧ S = sqrt 3

theorem ellipse_properties :
  ellipse_equation ∧ triangle_area :=
by
  -- The proofs are omitted as required by the problem statement
  sorry

end ellipse_properties_l147_147476


namespace marble_ratio_l147_147119

theorem marble_ratio
  (L_b : ℕ) (J_y : ℕ) (A : ℕ)
  (A_b : ℕ) (A_y : ℕ) (R : ℕ)
  (h1 : L_b = 4)
  (h2 : J_y = 22)
  (h3 : A = 19)
  (h4 : A_y = J_y / 2)
  (h5 : A = A_b + A_y)
  (h6 : A_b = L_b * R) :
  R = 2 := by
  sorry

end marble_ratio_l147_147119


namespace arithmetic_sequence_15th_term_l147_147145

theorem arithmetic_sequence_15th_term :
  ∀ (a d n : ℕ), a = 3 → d = 13 - a → n = 15 → 
  a + (n - 1) * d = 143 :=
by
  intros a d n ha hd hn
  rw [ha, hd, hn]
  sorry

end arithmetic_sequence_15th_term_l147_147145


namespace approx_average_sqft_per_person_l147_147048

noncomputable def average_sqft_per_person 
  (population : ℕ) 
  (land_area_sqmi : ℕ) 
  (sqft_per_sqmi : ℕ) : ℕ :=
(sqft_per_sqmi * land_area_sqmi) / population

theorem approx_average_sqft_per_person :
  average_sqft_per_person 331000000 3796742 (5280 ^ 2) = 319697 := 
sorry

end approx_average_sqft_per_person_l147_147048


namespace sheets_torn_out_l147_147276

-- Define the conditions as given in the problem
def first_torn_page : Nat := 185
def last_torn_page : Nat := 518
def pages_per_sheet : Nat := 2

-- Calculate the total number of pages torn out
def total_pages_torn_out : Nat :=
  last_torn_page - first_torn_page + 1

-- Calculate the number of sheets torn out
def number_of_sheets_torn_out : Nat :=
  total_pages_torn_out / pages_per_sheet

-- Prove that the number of sheets torn out is 167
theorem sheets_torn_out :
  number_of_sheets_torn_out = 167 :=
by
  unfold number_of_sheets_torn_out total_pages_torn_out
  rw [Nat.sub_add_cancel (Nat.le_of_lt (Nat.lt_of_le_of_ne
    (Nat.le_add_left _ _) (Nat.ne_of_lt (Nat.lt_add_one 184))))]
  rw [Nat.div_eq_of_lt (Nat.lt.base 333)] 
  sorry -- proof steps are omitted

end sheets_torn_out_l147_147276


namespace parabola_vertex_value_of_a_l147_147709

-- Define the conditions as given in the math problem
variables (a b c : ℤ)
def quadratic_fun (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

-- Given conditions about the vertex and a point on the parabola
def vertex_condition : Prop := (quadratic_fun a b c 2 = 3)
def point_condition : Prop := (quadratic_fun a b c 1 = 0)

-- Statement to prove
theorem parabola_vertex_value_of_a : vertex_condition a b c ∧ point_condition a b c → a = -3 :=
sorry

end parabola_vertex_value_of_a_l147_147709


namespace diamond_value_l147_147346

def diamond (a b : ℕ) : ℕ := 4 * a - 2 * b

theorem diamond_value : diamond 6 3 = 18 :=
by
  sorry

end diamond_value_l147_147346


namespace no_real_x_satisfies_quadratic_ineq_l147_147860

theorem no_real_x_satisfies_quadratic_ineq :
  ¬ ∃ x : ℝ, x^2 + 3 * x + 3 ≤ 0 :=
sorry

end no_real_x_satisfies_quadratic_ineq_l147_147860


namespace pow_equation_sum_l147_147489

theorem pow_equation_sum (x y : ℕ) (hx : 2 ^ 11 * 6 ^ 5 = 4 ^ x * 3 ^ y) : x + y = 13 :=
  sorry

end pow_equation_sum_l147_147489


namespace probability_first_or_second_l147_147581

/-- Define the events and their probabilities --/
def prob_hit_first_sector : ℝ := 0.4
def prob_hit_second_sector : ℝ := 0.3
def prob_hit_first_or_second : ℝ := 0.7

/-- The proof that these probabilities add up as mutually exclusive events --/
theorem probability_first_or_second (P_A : ℝ) (P_B : ℝ) (P_A_or_B : ℝ) (hP_A : P_A = prob_hit_first_sector) (hP_B : P_B = prob_hit_second_sector) (hP_A_or_B : P_A_or_B = prob_hit_first_or_second) :
  P_A_or_B = P_A + P_B := 
  by
    rw [hP_A, hP_B, hP_A_or_B]
    sorry

end probability_first_or_second_l147_147581


namespace torn_out_sheets_count_l147_147278

theorem torn_out_sheets_count :
  ∃ (sheets : ℕ), (first_page = 185 ∧
                   last_page = 518 ∧
                   pages_torn_out = last_page - first_page + 1 ∧ 
                   sheets = pages_torn_out / 2 ∧
                   sheets = 167) :=
by
  sorry

end torn_out_sheets_count_l147_147278


namespace transform_circle_to_ellipse_l147_147155

theorem transform_circle_to_ellipse (x y x'' y'' : ℝ) (h_circle : x^2 + y^2 = 1)
  (hx_trans : x = x'' / 2) (hy_trans : y = y'' / 3) :
  (x''^2 / 4) + (y''^2 / 9) = 1 :=
by {
  sorry
}

end transform_circle_to_ellipse_l147_147155


namespace part1_part2_l147_147401

-- Define set A
def set_A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }

-- Define set B depending on m
def set_B (m : ℝ) : Set ℝ := { x | 2 * m - 1 ≤ x ∧ x ≤ m + 1 }

-- Part 1: When m = -3, find A ∩ B
theorem part1 : set_B (-3) ∩ set_A = { x | -3 ≤ x ∧ x ≤ -2 } := 
sorry

-- Part 2: Find the range of m such that B ⊆ A
theorem part2 (m : ℝ) : set_B m ⊆ set_A ↔ m ≥ -1 :=
sorry

end part1_part2_l147_147401


namespace percentage_disliked_by_both_l147_147836

theorem percentage_disliked_by_both (total_comics liked_by_females liked_by_males disliked_by_both : ℕ) 
  (total_comics_eq : total_comics = 300)
  (liked_by_females_eq : liked_by_females = 30 * total_comics / 100)
  (liked_by_males_eq : liked_by_males = 120)
  (disliked_by_both_eq : disliked_by_both = total_comics - (liked_by_females + liked_by_males)) :
  (disliked_by_both * 100 / total_comics) = 30 := by
  sorry

end percentage_disliked_by_both_l147_147836


namespace root_implies_quadratic_eq_l147_147485

theorem root_implies_quadratic_eq (m : ℝ) (h : (m + 2) - 2 + m^2 - 2 * m - 6 = 0) : 
  2 * m^2 - m - 6 = 0 :=
sorry

end root_implies_quadratic_eq_l147_147485


namespace sum_of_solutions_l147_147616

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l147_147616


namespace amusing_permutations_formula_l147_147883

-- Definition of amusing permutations and their count
def amusing_permutations_count (n : ℕ) : ℕ :=
  2^(n-1)

-- Theorem statement: The number of amusing permutations of the set {1, 2, ..., n} is 2^(n-1)
theorem amusing_permutations_formula (n : ℕ) : 
  -- The number of amusing permutations should be equal to 2^(n-1)
  amusing_permutations_count n = 2^(n-1) :=
by
  sorry

end amusing_permutations_formula_l147_147883


namespace band_member_earnings_l147_147071

theorem band_member_earnings :
  let attendees := 500
  let ticket_price := 30
  let band_share_percentage := 70 / 100
  let band_members := 4
  let total_earnings := attendees * ticket_price
  let band_earnings := total_earnings * band_share_percentage
  let earnings_per_member := band_earnings / band_members
  earnings_per_member = 2625 := 
by {
  sorry
}

end band_member_earnings_l147_147071


namespace miles_traveled_correct_l147_147257

def initial_odometer_reading := 212.3
def odometer_reading_at_lunch := 372.0
def miles_traveled := odometer_reading_at_lunch - initial_odometer_reading

theorem miles_traveled_correct : miles_traveled = 159.7 :=
by
  sorry

end miles_traveled_correct_l147_147257


namespace sum_of_solutions_sum_of_all_solutions_l147_147638

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l147_147638


namespace odd_square_free_count_l147_147657

theorem odd_square_free_count : 
  ∃ n : ℕ, n = 64 ∧ ∀ k : ℕ, (1 < k ∧ k < 200 ∧ k % 2 = 1 ∧ 
  (∀ m : ℕ, m * m ∣ k → m = 1)) ↔ k ∈ {3, 5, 7, ..., 199} :=
sorry

end odd_square_free_count_l147_147657


namespace sum_prime_factors_of_143_l147_147006

theorem sum_prime_factors_of_143 :
  let is_prime (n : ℕ) := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0 in
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ 143 = a * b ∧ a ≠ b ∧ (a + b = 24) :=
by
  sorry

end sum_prime_factors_of_143_l147_147006


namespace find_cosine_of_angle_subtraction_l147_147465

variable (α : ℝ)
variable (h : Real.sin ((Real.pi / 6) - α) = 1 / 3)

theorem find_cosine_of_angle_subtraction :
  Real.cos ((2 * Real.pi / 3) - α) = -1 / 3 :=
by
  exact sorry

end find_cosine_of_angle_subtraction_l147_147465


namespace sq_diff_eq_binom_identity_l147_147451

variable (a b : ℝ)

theorem sq_diff_eq_binom_identity : (a - b) ^ 2 = a ^ 2 - 2 * a * b + b ^ 2 :=
by
  sorry

end sq_diff_eq_binom_identity_l147_147451


namespace find_B_find_sin_A_find_sin_2A_minus_B_l147_147108

open Real

noncomputable def triangle_conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a * cos C + c * cos A = 2 * b * cos B) ∧ (7 * a = 5 * b)

theorem find_B (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) :
  B = π / 3 :=
sorry

theorem find_sin_A (a b c A B C : ℝ) (h : triangle_conditions a b c A B C)
  (hB : B = π / 3) :
  sin A = 3 * sqrt 3 / 14 :=
sorry

theorem find_sin_2A_minus_B (a b c A B C : ℝ) (h : triangle_conditions a b c A B C)
  (hB : B = π / 3) (hA : sin A = 3 * sqrt 3 / 14) :
  sin (2 * A - B) = 8 * sqrt 3 / 49 :=
sorry

end find_B_find_sin_A_find_sin_2A_minus_B_l147_147108


namespace circle_through_point_and_same_center_l147_147358

theorem circle_through_point_and_same_center :
  ∃ (x_0 y_0 r : ℝ),
    (∀ (x y : ℝ), (x - x_0)^2 + (y - y_0)^2 = r^2 ↔
      x^2 + y^2 - 4 * x + 6 * y - 3 = 0)
    ∧
    ∀ (x y : ℝ), (x - x_0)^2 + (y - y_0)^2 = r^2 ↔
      (x - 2)^2 + (y + 3)^2 = 25 := sorry

end circle_through_point_and_same_center_l147_147358


namespace find_k_shelf_life_at_11_22_l147_147987

noncomputable def food_shelf_life (k b x : ℝ) : ℝ := Real.exp (k * x + b)

-- Given conditions
def condition1 : food_shelf_life k b 0 = 192 := by sorry
def condition2 : food_shelf_life k b 33 = 24 := by sorry

-- Prove that k = - (Real.log 2) / 11
theorem find_k (k b : ℝ) (h1 : food_shelf_life k b 0 = 192) (h2 : food_shelf_life k b 33 = 24) : 
  k = - (Real.log 2) / 11 :=
by sorry

-- Use the found value of k to determine the shelf life at 11°C and 22°C
theorem shelf_life_at_11_22 (k b : ℝ) (h1 : food_shelf_life k b 0 = 192) (h2 : food_shelf_life k b 33 = 24) 
  (hk : k = - (Real.log 2) / 11) : 
  food_shelf_life k b 11 = 96 ∧ food_shelf_life k b 22 = 48 :=
by sorry

end find_k_shelf_life_at_11_22_l147_147987


namespace find_s_l147_147900

theorem find_s : ∃ s : ℝ, ⌊s⌋ + s = 18.3 ∧ s = 9.3 :=
by
  sorry

end find_s_l147_147900


namespace sally_took_home_pens_l147_147700

theorem sally_took_home_pens
    (initial_pens : ℕ)
    (students : ℕ)
    (pens_per_student : ℕ)
    (locker_fraction : ℕ)
    (total_pens_given : ℕ)
    (remainder : ℕ)
    (locker_pens : ℕ)
    (home_pens : ℕ) :
    initial_pens = 5230 →
    students = 89 →
    pens_per_student = 58 →
    locker_fraction = 2 →
    total_pens_given = students * pens_per_student →
    remainder = initial_pens - total_pens_given →
    locker_pens = remainder / locker_fraction →
    home_pens = locker_pens →
    home_pens = 34 :=
by {
  sorry
}

end sally_took_home_pens_l147_147700


namespace josh_and_fred_age_l147_147248

theorem josh_and_fred_age
    (a b k : ℕ)
    (h1 : 10 * a + b > 10 * b + a)
    (h2 : 99 * (a^2 - b^2) = k^2)
    (ha : a ≥ 0 ∧ a ≤ 9)
    (hb : b ≥ 0 ∧ b ≤ 9) : 
    10 * a + b = 65 ∧ 
    10 * b + a = 56 := 
sorry

end josh_and_fred_age_l147_147248


namespace number_of_six_digit_palindromes_l147_147379

theorem number_of_six_digit_palindromes : 
  let count_palindromes : ℕ := 9 * 10 * 10 in
  count_palindromes = 900 :=
by
  sorry

end number_of_six_digit_palindromes_l147_147379


namespace rank_from_left_l147_147887

theorem rank_from_left (total_students rank_from_right rank_from_left : ℕ) 
  (h_total : total_students = 31) (h_right : rank_from_right = 21) : 
  rank_from_left = 11 := by
  sorry

end rank_from_left_l147_147887


namespace min_value_x_3y_min_value_x_3y_iff_l147_147797

theorem min_value_x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 4 * y = x * y) : x + 3 * y ≥ 25 :=
sorry

theorem min_value_x_3y_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 4 * y = x * y) : x + 3 * y = 25 ↔ x = 10 ∧ y = 5 :=
sorry

end min_value_x_3y_min_value_x_3y_iff_l147_147797


namespace line_through_intersection_points_l147_147374

noncomputable def circle1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 10 }
noncomputable def circle2 := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 10 }

theorem line_through_intersection_points (p : ℝ × ℝ) (hp1 : p ∈ circle1) (hp2 : p ∈ circle2) :
  p.1 + 3 * p.2 - 5 = 0 :=
sorry

end line_through_intersection_points_l147_147374


namespace find_d_l147_147189

theorem find_d
  (a b c d : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (h_d_pos : d > 0)
  (h_max : a * 1 + d = 5)
  (h_min : a * (-1) + d = -3) :
  d = 1 := 
sorry

end find_d_l147_147189


namespace decision_making_system_reliability_l147_147140

theorem decision_making_system_reliability (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  (10 * p^3 - 15 * p^4 + 6 * p^5 > 3 * p^2 - 2 * p^3) -> (1 / 2 < p) ∧ (p < 1) :=
by
  sorry

end decision_making_system_reliability_l147_147140


namespace subtraction_example_l147_147727

theorem subtraction_example : 6102 - 2016 = 4086 := by
  sorry

end subtraction_example_l147_147727


namespace outer_perimeter_l147_147142

theorem outer_perimeter (F G H I J K L M N : ℕ) 
  (h_outer : F + G + H + I + J = 42) 
  (h_inner : K + L + M = 20) 
  (h_adjustment : N = 4) : 
  F + G + H + I + J - K - L - M + N = 26 := 
by 
  sorry

end outer_perimeter_l147_147142


namespace example_equation_l147_147848

-- Define what it means to be an equation in terms of containing an unknown and being an equality
def is_equation (expr : Prop) (contains_unknown : Prop) : Prop :=
  (contains_unknown ∧ expr)

-- Prove that 4x + 2 = 10 is an equation
theorem example_equation : is_equation (4 * x + 2 = 10) (∃ x : ℝ, true) :=
  by sorry

end example_equation_l147_147848


namespace roots_cube_reciprocal_eqn_l147_147910

variable (a b c r s : ℝ)

def quadratic_eqn (r s : ℝ) : Prop :=
  3 * a * r ^ 2 + 5 * b * r + 7 * c = 0 ∧ 
  3 * a * s ^ 2 + 5 * b * s + 7 * c = 0

theorem roots_cube_reciprocal_eqn (h : quadratic_eqn a b c r s) :
  (1 / r^3 + 1 / s^3) = (-5 * b * (25 * b ^ 2 - 63 * c) / (343 * c^3)) :=
sorry

end roots_cube_reciprocal_eqn_l147_147910


namespace sum_of_first_nine_terms_l147_147912

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def sum_of_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (n / 2) * (a 0 + a (n - 1))

theorem sum_of_first_nine_terms 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_terms a S)
  (h_sum_terms : a 2 + a 3 + a 4 + a 5 + a 6 = 20) :
  S 9 = 36 :=
sorry

end sum_of_first_nine_terms_l147_147912


namespace sum_lent_l147_147862

theorem sum_lent (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ) 
  (h1 : r = 0.06) (h2 : t = 8) (h3 : I = P - 520) : P * r * t = I → P = 1000 :=
by
  -- Given conditions
  intros
  -- Sorry placeholder
  sorry

end sum_lent_l147_147862


namespace closest_fraction_to_team_aus_medals_l147_147892

theorem closest_fraction_to_team_aus_medals 
  (won_medals : ℕ) (total_medals : ℕ) 
  (choices : List ℚ)
  (fraction_won : ℚ)
  (c1 : won_medals = 28)
  (c2 : total_medals = 150)
  (c3 : choices = [1/4, 1/5, 1/6, 1/7, 1/8])
  (c4 : fraction_won = 28 / 150) :
  abs (fraction_won - 1/5) < abs (fraction_won - 1/4) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/6) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/7) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/8) := 
sorry

end closest_fraction_to_team_aus_medals_l147_147892


namespace part1_solution_part2_solution_l147_147017

-- Conditions
variables (x y : ℕ) -- Let x be the number of parcels each person sorts manually per hour,
                     -- y be the number of machines needed

def machine_efficiency : ℕ := 20 * x
def time_machines (parcels : ℕ) (machines : ℕ) : ℕ := parcels / (machines * machine_efficiency x)
def time_people (parcels : ℕ) (people : ℕ) : ℕ := parcels / (people * x)
def parcels_per_day : ℕ := 100000

-- Problem 1: Find x
axiom problem1 : (time_people 6000 20) - (time_machines 6000 5) = 4

-- Problem 2: Find y to sort 100000 parcels in a day with machines working 16 hours/day
axiom problem2 : 16 * machine_efficiency x * y ≥ parcels_per_day

-- Correct answers:
theorem part1_solution : x = 60 := by sorry
theorem part2_solution : y = 6 := by sorry

end part1_solution_part2_solution_l147_147017


namespace candice_spending_l147_147402

variable (total_budget : ℕ) (remaining_money : ℕ) (mildred_spending : ℕ)

theorem candice_spending 
  (h1 : total_budget = 100)
  (h2 : remaining_money = 40)
  (h3 : mildred_spending = 25) :
  (total_budget - remaining_money) - mildred_spending = 35 := 
by
  sorry

end candice_spending_l147_147402


namespace area_of_triangle_l147_147112

noncomputable def segment_length_AB : ℝ := 10
noncomputable def point_AP : ℝ := 2
noncomputable def point_PB : ℝ := segment_length_AB - point_AP -- PB = AB - AP 
noncomputable def radius_omega1 : ℝ := point_AP / 2 -- radius of ω1
noncomputable def radius_omega2 : ℝ := point_PB / 2 -- radius of ω2
noncomputable def distance_centers : ℝ := 5 -- given directly
noncomputable def length_XY : ℝ := 4 -- given directly
noncomputable def altitude_PZ : ℝ := 8 / 5 -- given directly
noncomputable def area_triangle_XPY : ℝ := (1 / 2) * length_XY * altitude_PZ

theorem area_of_triangle : area_triangle_XPY = 16 / 5 := by
  sorry

end area_of_triangle_l147_147112


namespace find_f2_l147_147083

noncomputable def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 :=
by
  sorry

end find_f2_l147_147083


namespace hyperbola_parabola_intersection_l147_147146

open Real

theorem hyperbola_parabola_intersection :
  let A := (4, 4)
  let B := (4, -4)
  |dist A B| = 8 :=
by
  let hyperbola_asymptote (x y: ℝ) := x^2 - y^2 = 1
  let parabola_equation (x y : ℝ) := y^2 = 4 * x
  sorry

end hyperbola_parabola_intersection_l147_147146


namespace total_savings_in_2_months_l147_147326

def students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_in_month : ℕ := 4
def months : ℕ := 2

def total_contribution_per_week : ℕ := students * contribution_per_student_per_week
def total_weeks : ℕ := months * weeks_in_month
def total_savings : ℕ := total_contribution_per_week * total_weeks

theorem total_savings_in_2_months : total_savings = 480 := by
  -- Proof goes here
  sorry

end total_savings_in_2_months_l147_147326


namespace sum_prime_factors_143_l147_147002

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24 := 
by
  let p := 11
  let q := 13
  have h1 : 143 = p * q := by norm_num
  have h2 : prime p := by norm_num
  have h3 : prime q := by norm_num
  have h4 : p + q = 24 := by norm_num
  exact ⟨p, q, h2, h3, h1, h4⟩  

end sum_prime_factors_143_l147_147002


namespace negation_forall_pos_l147_147149

theorem negation_forall_pos (h : ∀ x : ℝ, 0 < x → x^2 + x + 1 > 0) :
  ∃ x : ℝ, 0 < x ∧ x^2 + x + 1 ≤ 0 :=
sorry

end negation_forall_pos_l147_147149


namespace y_coordinate_in_fourth_quadrant_l147_147671
-- Importing the necessary libraries

-- Definition of the problem statement
theorem y_coordinate_in_fourth_quadrant (x y : ℝ) (h : x = 5 ∧ y < 0) : y < 0 :=
by 
  sorry

end y_coordinate_in_fourth_quadrant_l147_147671


namespace period_cosine_l147_147603

noncomputable def period_of_cosine_function : ℝ := 2 * Real.pi / 3

theorem period_cosine (x : ℝ) : ∃ T, ∀ x, Real.cos (3 * x - Real.pi) = Real.cos (3 * (x + T) - Real.pi) :=
  ⟨period_of_cosine_function, by sorry⟩

end period_cosine_l147_147603


namespace percentage_disliked_by_both_l147_147834

theorem percentage_disliked_by_both 
  (total_comic_books : ℕ) 
  (percentage_females_like : ℕ) 
  (comic_books_males_like : ℕ) :
  total_comic_books = 300 →
  percentage_females_like = 30 →
  comic_books_males_like = 120 →
  ((total_comic_books - (total_comic_books * percentage_females_like / 100) - comic_books_males_like) * 100 / total_comic_books) = 30 :=
by
  intros h1 h2 h3
  sorry

end percentage_disliked_by_both_l147_147834


namespace painting_house_cost_l147_147251

theorem painting_house_cost 
  (judson_contrib : ℕ := 500)
  (kenny_contrib : ℕ := judson_contrib + (judson_contrib * 20) / 100)
  (camilo_contrib : ℕ := kenny_contrib + 200) :
  judson_contrib + kenny_contrib + camilo_contrib = 1900 :=
by
  sorry

end painting_house_cost_l147_147251


namespace L_of_specific_set_L_of_arithmetic_sequence_l147_147800

-- Definitions for set and arithmetic sequence properties
variable {α : Type*} [Add α] [LT α] [DecidableEq α] 

def sum_of_two_elements (A : Finset α) : Finset α := 
  (A.product A).filter (λ ⟨x, y⟩, x < y).image (λ ⟨x, y⟩, x + y)

def L (A : Finset α) : ℕ := (sum_of_two_elements A).card

-- Problem 1: Specific set A = {2, 4, 6, 8}
def specific_set_A := ({2, 4, 6, 8} : Finset ℕ)

-- Problem 2: Arithmetic sequence
def is_arithmetic_sequence (A : Finset ℕ) : Prop := 
  ∃ (a d : ℕ), A = Finset.image (λ n, a + n * d) (Finset.range A.card)

theorem L_of_specific_set : L specific_set_A = 5 := 
  sorry

theorem L_of_arithmetic_sequence (m : ℕ) (h : 2 < m) (A : Finset ℕ) (ha : A.card = m) (has : is_arithmetic_sequence A) : 
  L A = 2 * m - 3 :=
  sorry

end L_of_specific_set_L_of_arithmetic_sequence_l147_147800


namespace solution_l147_147607

open Set

theorem solution (A B : Set ℤ) :
  (∀ x, x ∈ A ∨ x ∈ B) →
  (∀ x, x ∈ A → (x - 1) ∈ B) →
  (∀ x y, x ∈ B ∧ y ∈ B → (x + y) ∈ A) →
  A = { z | ∃ n, z = 2 * n } ∧ B = { z | ∃ n, z = 2 * n + 1 } :=
by
  sorry

end solution_l147_147607


namespace mass_of_man_l147_147733

-- Definitions of the given conditions
def boat_length : Float := 3.0
def boat_breadth : Float := 2.0
def sink_depth : Float := 0.01 -- 1 cm converted to meters
def water_density : Float := 1000.0 -- Density of water in kg/m³

-- Define the proof goal as the mass of the man
theorem mass_of_man : Float :=
by
  let volume_displaced := boat_length * boat_breadth * sink_depth
  let weight_displaced := volume_displaced * water_density
  exact weight_displaced

end mass_of_man_l147_147733


namespace aria_analysis_time_l147_147754

-- Definitions for the number of bones in each section
def skull_bones : ℕ := 29
def spine_bones : ℕ := 33
def thorax_bones : ℕ := 37
def upper_limb_bones : ℕ := 64
def lower_limb_bones : ℕ := 62

-- Definitions for the time spent per bone in each section (in minutes)
def time_per_skull_bone : ℕ := 15
def time_per_spine_bone : ℕ := 10
def time_per_thorax_bone : ℕ := 12
def time_per_upper_limb_bone : ℕ := 8
def time_per_lower_limb_bone : ℕ := 10

-- Definition for the total time needed in minutes
def total_time_in_minutes : ℕ :=
  (skull_bones * time_per_skull_bone) +
  (spine_bones * time_per_spine_bone) +
  (thorax_bones * time_per_thorax_bone) +
  (upper_limb_bones * time_per_upper_limb_bone) +
  (lower_limb_bones * time_per_lower_limb_bone)

-- Definition for the total time needed in hours
def total_time_in_hours : ℚ := total_time_in_minutes / 60

-- Theorem to prove the total time needed in hours is approximately 39.02
theorem aria_analysis_time : abs (total_time_in_hours - 39.02) < 0.01 :=
by
  sorry

end aria_analysis_time_l147_147754


namespace count_multiples_l147_147655

theorem count_multiples (n : ℕ) (h_n : n = 300) : 
  let m := 6 in 
  let m' := 12 in 
  (finset.card (finset.filter (λ x, x % m = 0 ∧ x % m' ≠ 0) (finset.range n))) = 24 :=
by
  sorry

end count_multiples_l147_147655


namespace sum_of_15_consecutive_integers_perfect_square_l147_147849

open Nat

-- statement that defines the conditions and the objective of the problem
theorem sum_of_15_consecutive_integers_perfect_square :
  ∃ n k : ℕ, 15 * (n + 7) = k^2 ∧ 15 * (n + 7) ≥ 225 := 
sorry

end sum_of_15_consecutive_integers_perfect_square_l147_147849


namespace proof_problem_l147_147081

variable {a b c d e f : ℝ}

theorem proof_problem :
  (a * b * c = 130) →
  (b * c * d = 65) →
  (d * e * f = 250) →
  (a * f / (c * d) = 0.5) →
  (c * d * e = 1000) :=
by
  intros h1 h2 h3 h4
  sorry

end proof_problem_l147_147081


namespace problem1_problem2_l147_147793

theorem problem1 (x : ℝ) : (4 * x ^ 2 + 12 * x - 7 ≤ 0) ∧ (a = 0) ∧ (x < -3 ∨ x > 3) → (-7/2 ≤ x ∧ x < -3) := by
  sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, 4 * x ^ 2 + 12 * x - 7 ≤ 0 → a - 3 ≤ x ∧ x ≤ a + 3) → (-5/2 ≤ a ∧ a ≤ -1/2) := by
  sorry

end problem1_problem2_l147_147793


namespace angle_C_is_100_l147_147552

-- Define the initial measures in the equilateral triangle
def initial_angle (A B C : ℕ) (h_equilateral : A = B ∧ B = C ∧ C = 60) : ℕ := C

-- Definition to capture the increase in angle C
def increased_angle (C : ℕ) : ℕ := C + 40

-- Now, we need to state the theorem assuming the given conditions
theorem angle_C_is_100
  (A B C : ℕ)
  (h_equilateral : A = 60 ∧ B = 60 ∧ C = 60)
  (h_increase : C = 60 + 40)
  : C = 100 := 
sorry

end angle_C_is_100_l147_147552


namespace xiaoguang_advances_l147_147289

theorem xiaoguang_advances (x1 x2 x3 x4 : ℝ) (h1 : 96 ≤ (x1 + x2 + x3 + x4) / 4) (hx1 : x1 = 95) (hx2 : x2 = 97) (hx3 : x3 = 94) : 
  98 ≤ x4 := 
by 
  sorry

end xiaoguang_advances_l147_147289


namespace max_M_is_7524_l147_147667

-- Define the conditions
def is_valid_t (t : ℕ) : Prop :=
  let a := t / 1000
  let b := (t % 1000) / 100
  let c := (t % 100) / 10
  let d := t % 10
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + c = 9 ∧
  b - d = 1 ∧
  (2 * (2 * a + d)) % (2 * b + c) = 0

-- Define function M
def M (a b c d : ℕ) : ℕ := 2000 * a + 100 * b + 10 * c + d

-- Define the maximum value of M
def max_valid_M : ℕ :=
  let m_values := [5544, 7221, 7322, 7524]
  m_values.foldl max 0

theorem max_M_is_7524 : max_valid_M = 7524 := by
  -- The proof would be written here. For now, we indicate the theorem as
  -- not yet proven.
  sorry

end max_M_is_7524_l147_147667


namespace sum_of_digits_of_fraction_repeating_decimal_l147_147713

theorem sum_of_digits_of_fraction_repeating_decimal :
  (exists (c d : ℕ), (4 / 13 : ℚ) = c * 0.1 + d * 0.01 ∧ (c + d) = 3) :=
sorry

end sum_of_digits_of_fraction_repeating_decimal_l147_147713


namespace unique_integer_solution_l147_147899

-- Define the problem statement and the conditions: integers x, y such that x^4 - 2y^2 = 1
theorem unique_integer_solution (x y: ℤ) (h: x^4 - 2 * y^2 = 1) : (x = 1 ∧ y = 0) :=
sorry

end unique_integer_solution_l147_147899


namespace find_y_l147_147486

theorem find_y (y : ℚ) (h : 1/3 - 1/4 = 4/y) : y = 48 := sorry

end find_y_l147_147486


namespace tan_sin_cos_ratio_l147_147469

open Real

variable {α β : ℝ}

theorem tan_sin_cos_ratio (h1 : tan (α + β) = 2) (h2 : tan (α - β) = 3) :
  sin (2 * α) / cos (2 * β) = 5 / 7 := sorry

end tan_sin_cos_ratio_l147_147469


namespace train_length_l147_147164

theorem train_length (v_kmh : ℝ) (p_len : ℝ) (t_sec : ℝ) (l_train : ℝ) 
  (h_v : v_kmh = 72) (h_p : p_len = 250) (h_t : t_sec = 26) :
  l_train = 270 :=
by
  sorry

end train_length_l147_147164


namespace minimum_value_y_range_of_a_l147_147798

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

theorem minimum_value_y (x : ℝ) 
  (hx_pos : x > 0) : (f x 2 / x) = -2 :=
by sorry

theorem range_of_a : 
  ∀ a : ℝ, ∀ x ∈ (Set.Icc 0 2), (f x a) ≤ a ↔ a ≥ 3 / 4 :=
by sorry

end minimum_value_y_range_of_a_l147_147798


namespace find_a_l147_147928

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (a^2 + a = 6)) : a = 2 :=
sorry

end find_a_l147_147928


namespace student_error_difference_l147_147676

theorem student_error_difference (num : ℤ) (num_val : num = 480) : 
  (5 / 6 * num - 5 / 16 * num) = 250 := 
by 
  sorry

end student_error_difference_l147_147676


namespace third_quadrant_to_first_third_fourth_l147_147923

theorem third_quadrant_to_first_third_fourth (k : ℤ) (α : ℝ) 
  (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2) : 
  ∃ n : ℤ, (2 * k / 3 % 2) * Real.pi + Real.pi / 3 < α / 3 ∧ α / 3 < (2 * k / 3 % 2) * Real.pi + Real.pi / 2 ∨
            (2 * (3 * n + 1) % 2) * Real.pi + Real.pi < α / 3 ∧ α / 3 < (2 * (3 * n + 1) % 2) * Real.pi + 7 * Real.pi / 6 ∨
            (2 * (3 * n + 2) % 2) * Real.pi + 5 * Real.pi / 3 < α / 3 ∧ α / 3 < (2 * (3 * n + 2) % 2) * Real.pi + 11 * Real.pi / 6 :=
sorry

end third_quadrant_to_first_third_fourth_l147_147923


namespace completing_square_l147_147558

theorem completing_square (x : ℝ) (h : x^2 - 6 * x - 7 = 0) : (x - 3)^2 = 16 := 
sorry

end completing_square_l147_147558


namespace Calvin_insect_collection_l147_147587

def Calvin_has_insects (num_roaches num_scorpions num_crickets num_caterpillars total_insects : ℕ) : Prop :=
  total_insects = num_roaches + num_scorpions + num_crickets + num_caterpillars

theorem Calvin_insect_collection
  (roach_count : ℕ)
  (scorpion_count : ℕ)
  (cricket_count : ℕ)
  (caterpillar_count : ℕ)
  (total_count : ℕ)
  (h1 : roach_count = 12)
  (h2 : scorpion_count = 3)
  (h3 : cricket_count = roach_count / 2)
  (h4 : caterpillar_count = scorpion_count * 2)
  (h5 : total_count = roach_count + scorpion_count + cricket_count + caterpillar_count) :
  Calvin_has_insects roach_count scorpion_count cricket_count caterpillar_count total_count :=
by
  rw [h1, h2, h3, h4]
  norm_num
  exact h5

end Calvin_insect_collection_l147_147587


namespace time_to_upload_file_l147_147430

-- Define the conditions
def file_size : ℕ := 160
def upload_speed : ℕ := 8

-- Define the question as a proof goal
theorem time_to_upload_file :
  file_size / upload_speed = 20 := 
sorry

end time_to_upload_file_l147_147430


namespace area_of_quadrilateral_AXYD_l147_147506

open Real

noncomputable def area_quadrilateral_AXYD: ℝ :=
  let A := (0, 0)
  let B := (20, 0)
  let C := (20, 12)
  let D := (0, 12)
  let Z := (20, 30)
  let E := (6, 6)
  let X := (2.5, 0)
  let Y := (9.5, 12)
  let base1 := (B.1 - X.1)  -- Length from B to X
  let base2 := (Y.1 - A.1)  -- Length from D to Y
  let height := (C.2 - A.2) -- Height common for both bases
  (base1 + base2) * height / 2

theorem area_of_quadrilateral_AXYD : area_quadrilateral_AXYD = 72 :=
by
  sorry

end area_of_quadrilateral_AXYD_l147_147506


namespace proof_l147_147606

noncomputable def proof_problem (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem proof (
  a b c : ℝ
) (h1 : (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3 = 3 * (a^3 - b^3) * (b^3 - c^3) * (c^3 - a^3))
  (h2 : (a - b)^3 + (b - c)^3 + (c - a)^3 = 3 * (a - b) * (b - c) * (c - a)) :
  proof_problem a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end proof_l147_147606


namespace sum_of_solutions_sum_of_solutions_is_16_l147_147630

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l147_147630


namespace minimal_rooms_l147_147125

-- Definitions
def numTourists := 100

def roomsAvailable (n k : Nat) : Prop :=
  ∀ k_even : k % 2 = 0, 
    ∃ m : Nat, k = 2 * m ∧ n = 100 * (m + 1) ∨
    ∀ k_odd : k % 2 = 1, k = 2 * m + 1 ∧ n = 100 * (m + 1) + 1

-- Proof statement
theorem minimal_rooms (k n : Nat) : roomsAvailable n k :=
by 
  -- The proof is provided in the solution steps
  sorry

end minimal_rooms_l147_147125


namespace number_of_stanzas_is_correct_l147_147523

-- Define the total number of words in the poem
def total_words : ℕ := 1600

-- Define the number of lines per stanza
def lines_per_stanza : ℕ := 10

-- Define the number of words per line
def words_per_line : ℕ := 8

-- Calculate the number of words per stanza
def words_per_stanza : ℕ := lines_per_stanza * words_per_line

-- Define the number of stanzas
def stanzas (total_words words_per_stanza : ℕ) := total_words / words_per_stanza

-- Theorem: Prove that given the conditions, the number of stanzas is 20
theorem number_of_stanzas_is_correct : stanzas total_words words_per_stanza = 20 :=
by
  -- Insert the proof here
  sorry

end number_of_stanzas_is_correct_l147_147523


namespace range_of_a_l147_147385

theorem range_of_a (x a : ℝ) : (∃ x : ℝ,  |x + 2| + |x - 3| ≤ |a - 1| ) ↔ (a ≤ -4 ∨ a ≥ 6) :=
by
  sorry

end range_of_a_l147_147385


namespace ticket_cost_difference_l147_147157

theorem ticket_cost_difference (num_prebuy : ℕ) (price_prebuy : ℕ) (num_gate : ℕ) (price_gate : ℕ)
  (h_prebuy : num_prebuy = 20) (h_price_prebuy : price_prebuy = 155)
  (h_gate : num_gate = 30) (h_price_gate : price_gate = 200) :
  num_gate * price_gate - num_prebuy * price_prebuy = 2900 :=
by
  sorry

end ticket_cost_difference_l147_147157


namespace range_of_expression_l147_147078

variable {a b : ℝ}

theorem range_of_expression 
  (h₁ : -1 < a + b) (h₂ : a + b < 3)
  (h₃ : 2 < a - b) (h₄ : a - b < 4) :
  -9 / 2 < 2 * a + 3 * b ∧ 2 * a + 3 * b < 13 / 2 := 
sorry

end range_of_expression_l147_147078


namespace ratio_Bipin_Alok_l147_147760

-- Definitions based on conditions
def Alok_age : Nat := 5
def Chandan_age : Nat := 10
def Bipin_age : Nat := 30
def Bipin_age_condition (B C : Nat) : Prop := B + 10 = 2 * (C + 10)

-- Statement to prove
theorem ratio_Bipin_Alok : 
  Bipin_age_condition Bipin_age Chandan_age -> 
  Alok_age = 5 -> 
  Chandan_age = 10 -> 
  Bipin_age / Alok_age = 6 :=
by
  sorry

end ratio_Bipin_Alok_l147_147760


namespace line_intersects_circle_l147_147712

theorem line_intersects_circle (k : ℝ) (h1 : k = 2) (radius : ℝ) (center_distance : ℝ) (eq_roots : ∀ x, x^2 - k * x + 1 = 0) :
  radius = 5 → center_distance = k → k < radius :=
by
  intros hradius hdistance
  have h_root_eq : k = 2 := h1
  have h_rad : radius = 5 := hradius
  have h_dist : center_distance = k := hdistance
  have kval : k = 2 := h1
  simp [kval, hradius, hdistance, h_rad, h_dist]
  sorry

end line_intersects_circle_l147_147712


namespace tan_of_angle_in_fourth_quadrant_l147_147915

theorem tan_of_angle_in_fourth_quadrant (α : ℝ) (h1 : Real.sin α = -5 / 13) (h2 : α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) :
  Real.tan α = -5 / 12 :=
sorry

end tan_of_angle_in_fourth_quadrant_l147_147915


namespace band_member_earnings_l147_147069

-- Define conditions
def n_people : ℕ := 500
def p_ticket : ℚ := 30
def r_earnings : ℚ := 0.7
def n_members : ℕ := 4

-- Definition of total earnings and share per band member
def total_earnings : ℚ := n_people * p_ticket
def band_share : ℚ := total_earnings * r_earnings
def amount_per_member : ℚ := band_share / n_members

-- Statement to be proved
theorem band_member_earnings : amount_per_member = 2625 := 
by
  -- Proof goes here
  sorry

end band_member_earnings_l147_147069


namespace probability_of_two_digit_number_l147_147178

def total_elements_in_set : ℕ := 961
def two_digit_elements_in_set : ℕ := 60

theorem probability_of_two_digit_number :
  (two_digit_elements_in_set : ℚ) / total_elements_in_set = 60 / 961 := by
  sorry

end probability_of_two_digit_number_l147_147178


namespace find_2alpha_minus_beta_l147_147204

theorem find_2alpha_minus_beta (α β : ℝ) (tan_diff : Real.tan (α - β) = 1 / 2) 
  (cos_β : Real.cos β = -7 * Real.sqrt 2 / 10) (α_range : 0 < α ∧ α < Real.pi) 
  (β_range : 0 < β ∧ β < Real.pi) : 2 * α - β = -3 * Real.pi / 4 :=
sorry

end find_2alpha_minus_beta_l147_147204


namespace Nicky_profit_l147_147829

-- Definitions for conditions
def card1_value : ℕ := 8
def card2_value : ℕ := 8
def received_card_value : ℕ := 21

-- The statement to be proven
theorem Nicky_profit : (received_card_value - (card1_value + card2_value)) = 5 :=
by
  sorry

end Nicky_profit_l147_147829


namespace total_time_correct_l147_147943

-- Conditions
def minutes_per_story : Nat := 7
def weeks : Nat := 20

-- Total time calculation
def total_minutes : Nat := minutes_per_story * weeks

-- Conversion to hours and minutes
def total_hours : Nat := total_minutes / 60
def remaining_minutes : Nat := total_minutes % 60

-- The proof problem
theorem total_time_correct :
  total_minutes = 140 ∧ total_hours = 2 ∧ remaining_minutes = 20 := by
  sorry

end total_time_correct_l147_147943


namespace sum_of_solutions_eq_16_l147_147621

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l147_147621


namespace r_p_q_sum_l147_147471

theorem r_p_q_sum (t p q r : ℕ) (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4)
    (h2 : (1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r)
    (h3 : r > 0) (h4 : p > 0) (h5 : q > 0)
    (h6 : Nat.gcd p q = 1) : r + p + q = 5 := 
sorry

end r_p_q_sum_l147_147471


namespace david_ate_more_than_emma_l147_147677

-- Definitions and conditions
def contestants : Nat := 8
def pies_david_ate : Nat := 8
def pies_emma_ate : Nat := 2
def pies_by_david (contestants pies_david_ate: Nat) : Prop := pies_david_ate = 8
def pies_by_emma (contestants pies_emma_ate: Nat) : Prop := pies_emma_ate = 2

-- Theorem statement
theorem david_ate_more_than_emma (contestants pies_david_ate pies_emma_ate : Nat) (h_david : pies_by_david contestants pies_david_ate) (h_emma : pies_by_emma contestants pies_emma_ate) : pies_david_ate - pies_emma_ate = 6 :=
by
  sorry

end david_ate_more_than_emma_l147_147677


namespace problem_1_problem_2_problem_3_l147_147507

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.log (x + 1) - (a*x / (x + 1))

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := (1 + k) ^ x - k * x - 1

theorem problem_1 {a : ℝ} : 
  (a ≤ 0 → ∀ x > -1, deriv (λ x, f x a) x > 0) ∧
  (a > 0 → ∀ x ∈ Ioo (-1) (a-1), deriv (λ x, f x a) x < 0 ∧ ∀ x ∈ Ioo (a-1) 1, deriv (λ x, f x a) x > 0) :=
sorry

theorem problem_2 {k : ℝ} (hk : k ∈ Ioi (-1)) : 
  ∀ x ∈ Icc (0 : ℝ) 1, g x k = 0 :=
sorry

theorem problem_3 (n : ℕ) (hn : 0 < n) : 
  ∑ k in finset.range n, (1 : ℝ) / (k + 2) < real.log (n+1) ∧ 
  real.log (n+1) < ∑ k in finset.range n, (1 : ℝ) / (k + 1) :=
sorry

end problem_1_problem_2_problem_3_l147_147507


namespace studentsInBandOrSports_l147_147571

-- conditions definitions
def totalStudents : ℕ := 320
def studentsInBand : ℕ := 85
def studentsInSports : ℕ := 200
def studentsInBoth : ℕ := 60

-- theorem statement
theorem studentsInBandOrSports : studentsInBand + studentsInSports - studentsInBoth = 225 :=
by
  sorry

end studentsInBandOrSports_l147_147571


namespace priyas_age_l147_147260

/-- 
  Let P be Priya's current age, and F be her father's current age. 
  Given:
  1. F = P + 31
  2. (P + 8) + (F + 8) = 69
  Prove: Priya's current age P is 11.
-/
theorem priyas_age 
  (P F : ℕ) 
  (h1 : F = P + 31) 
  (h2 : (P + 8) + (F + 8) = 69) 
  : P = 11 :=
by
  sorry

end priyas_age_l147_147260


namespace Petya_tore_out_sheets_l147_147281

theorem Petya_tore_out_sheets (n m : ℕ) (h1 : n = 185) (h2 : m = 518)
  (h3 : m.digits = n.digits) : (m - n + 1) / 2 = 167 :=
by
  sorry

end Petya_tore_out_sheets_l147_147281


namespace solve_for_y_l147_147526

theorem solve_for_y : ∃ y : ℝ, (5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3)) ↔ y = 1000 := by
  sorry

end solve_for_y_l147_147526


namespace total_stickers_used_l147_147352

-- Define all the conditions as given in the problem
def initially_water_bottles : ℕ := 20
def lost_at_school : ℕ := 5
def found_at_park : ℕ := 3
def stolen_at_dance : ℕ := 4
def misplaced_at_library : ℕ := 2
def acquired_from_friend : ℕ := 6
def stickers_per_bottle_school : ℕ := 4
def stickers_per_bottle_dance : ℕ := 3
def stickers_per_bottle_library : ℕ := 2

-- Prove the total number of stickers used
theorem total_stickers_used : 
  (lost_at_school * stickers_per_bottle_school)
  + (stolen_at_dance * stickers_per_bottle_dance)
  + (misplaced_at_library * stickers_per_bottle_library)
  = 36 := 
by
  sorry

end total_stickers_used_l147_147352


namespace pete_total_blocks_traveled_l147_147840

theorem pete_total_blocks_traveled : 
    ∀ (walk_to_garage : ℕ) (bus_to_post_office : ℕ), 
    walk_to_garage = 5 → bus_to_post_office = 20 → 
    ((walk_to_garage + bus_to_post_office) * 2) = 50 :=
by
  intros walk_to_garage bus_to_post_office h_walk h_bus
  sorry

end pete_total_blocks_traveled_l147_147840


namespace geom_seq_value_l147_147647

variable (a_n : ℕ → ℝ)
variable (r : ℝ)
variable (π : ℝ)

-- Define the conditions
axiom geom_seq : ∀ n, a_n (n + 1) = a_n n * r
axiom sum_pi : a_n 3 + a_n 5 = π

-- Statement to prove
theorem geom_seq_value : a_n 4 * (a_n 2 + 2 * a_n 4 + a_n 6) = π^2 :=
by
  sorry

end geom_seq_value_l147_147647


namespace C_investment_is_20000_l147_147045

-- Definitions of investments and profits
def A_investment : ℕ := 12000
def B_investment : ℕ := 16000
def total_profit : ℕ := 86400
def C_share_of_profit : ℕ := 36000

-- The proof problem statement
theorem C_investment_is_20000 (X : ℕ) (hA : A_investment = 12000) (hB : B_investment = 16000)
  (h_total_profit : total_profit = 86400) (h_C_share_of_profit : C_share_of_profit = 36000) :
  X = 20000 :=
sorry

end C_investment_is_20000_l147_147045


namespace not_function_of_x_l147_147730

theorem not_function_of_x : 
  ∃ x : ℝ, ∃ y1 y2 : ℝ, (|y1| = 2 * x ∧ |y2| = 2 * x ∧ y1 ≠ y2) := sorry

end not_function_of_x_l147_147730


namespace value_of_f_f_f_2_l147_147951

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem value_of_f_f_f_2 : f (f (f 2)) = 2 :=
by {
  sorry
}

end value_of_f_f_f_2_l147_147951


namespace sum_of_numbers_l147_147534

theorem sum_of_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 130) : x + y = -7 :=
by
  sorry

end sum_of_numbers_l147_147534


namespace roman_coins_left_l147_147128

theorem roman_coins_left (X Y : ℕ) (h1 : X * Y = 50) (h2 : (X - 7) * Y = 28) : X - 7 = 8 :=
by
  sorry

end roman_coins_left_l147_147128


namespace pamTotalApples_l147_147699

-- Define the given conditions
def applesPerGeraldBag : Nat := 40
def applesPerPamBag := 3 * applesPerGeraldBag
def pamBags : Nat := 10

-- Statement to prove
theorem pamTotalApples : pamBags * applesPerPamBag = 1200 :=
by
  sorry

end pamTotalApples_l147_147699


namespace functional_eqn_even_function_l147_147957

variable {R : Type*} [AddGroup R] (f : R → ℝ)

theorem functional_eqn_even_function
  (h_not_zero : ∃ x, f x ≠ 0)
  (h_func_eq : ∀ a b, f (a + b) + f (a - b) = 2 * f a + 2 * f b) :
  ∀ x, f (-x) = f x :=
by
  sorry

end functional_eqn_even_function_l147_147957


namespace probability_of_odd_sum_given_even_product_l147_147788

-- Define a function to represent the probability of an event given the conditions
noncomputable def conditional_probability_odd_sum_even_product (dice : Fin 5 → Fin 8) : ℚ :=
  if h : (∃ i, (dice i).val % 2 = 0)  -- At least one die is even (product is even)
  then (1/2) / (31/32)  -- Probability of odd sum given even product
  else 0  -- If product is not even (not possible under conditions)

theorem probability_of_odd_sum_given_even_product :
  ∀ (dice : Fin 5 → Fin 8),
  conditional_probability_odd_sum_even_product dice = 16/31 :=
sorry  -- Proof omitted

end probability_of_odd_sum_given_even_product_l147_147788


namespace beads_initial_state_repeats_l147_147759

-- Define the setup of beads on a circular wire
structure BeadConfig (n : ℕ) :=
(beads : Fin n → ℝ)  -- Each bead's position indexed by a finite set, ℝ denotes angular position

-- Define the instantaneous collision swapping function
def swap (n : ℕ) (i j : Fin n) (config : BeadConfig n) : BeadConfig n :=
⟨fun k => if k = i then config.beads j else if k = j then config.beads i else config.beads k⟩

-- Define what it means for a configuration to return to its initial state
def returns_to_initial (n : ℕ) (initial : BeadConfig n) (t : ℝ) : Prop :=
  ∃ (config : BeadConfig n), (∀ k, config.beads k = initial.beads k) ∧ (config = initial)

-- Specification of the problem
theorem beads_initial_state_repeats (n : ℕ) (initial : BeadConfig n) (ω : Fin n → ℝ) :
  (∀ k, ω k > 0) →  -- condition that all beads have positive angular speed, either clockwise or counterclockwise
  ∃ t : ℝ, t > 0 ∧ returns_to_initial n initial t := 
by
  sorry

end beads_initial_state_repeats_l147_147759


namespace water_pump_rate_l147_147411

theorem water_pump_rate (hourly_rate : ℕ) (minutes : ℕ) (calculated_gallons : ℕ) : 
  hourly_rate = 600 → minutes = 30 → calculated_gallons = (hourly_rate * (minutes / 60)) → 
  calculated_gallons = 300 :=
by 
  sorry

end water_pump_rate_l147_147411


namespace part_I_part_II_part_III_l147_147085

noncomputable def f (x : ℝ) := x / (x^2 - 1)

-- (I) Prove that f(2) = 2/3.
theorem part_I : f 2 = 2 / 3 :=
by sorry

-- (II) Prove that f(x) is decreasing on the interval (-1, 1).
theorem part_II : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 > f x2 :=
by sorry

-- (III) Prove that f(x) is an odd function.
theorem part_III : ∀ x : ℝ, f (-x) = -f x :=
by sorry

end part_I_part_II_part_III_l147_147085


namespace total_games_equal_684_l147_147170

-- Define the number of players
def n : Nat := 19

-- Define the formula to calculate the total number of games played
def total_games (n : Nat) : Nat := n * (n - 1) * 2

-- The proposition asserting the total number of games equals 684
theorem total_games_equal_684 : total_games n = 684 :=
by
  sorry

end total_games_equal_684_l147_147170


namespace find_a_b_l147_147205

theorem find_a_b (a b x y : ℝ) (h₀ : a + b = 10) (h₁ : a / x + b / y = 1) (h₂ : x + y = 16) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
    (a = 1 ∧ b = 9) ∨ (a = 9 ∧ b = 1) :=
by
  sorry

end find_a_b_l147_147205


namespace swimmer_path_min_time_l147_147579

theorem swimmer_path_min_time (k : ℝ) :
  (k > Real.sqrt 2 → ∀ x y : ℝ, x = 0 ∧ y = 0 ∧ t = 2/k) ∧
  (k < Real.sqrt 2 → x = 1 ∧ y = 1 ∧ t = Real.sqrt 2) ∧
  (k = Real.sqrt 2 → ∀ x y : ℝ, x = y ∧ t = (1 / Real.sqrt 2) + Real.sqrt 2 + (1 / Real.sqrt 2)) :=
by sorry

end swimmer_path_min_time_l147_147579


namespace cube_roll_sums_l147_147395

def opposite_faces_sum_to_seven (a b : ℕ) : Prop := a + b = 7

def valid_cube_faces : Prop := 
  opposite_faces_sum_to_seven 1 6 ∧
  opposite_faces_sum_to_seven 2 5 ∧
  opposite_faces_sum_to_seven 3 4

def max_min_sums : ℕ × ℕ := (342, 351)

theorem cube_roll_sums (faces_sum_seven : valid_cube_faces) : 
  ∃ cube_sums : ℕ × ℕ, cube_sums = max_min_sums := sorry

end cube_roll_sums_l147_147395


namespace find_track_circumference_l147_147434

noncomputable def track_circumference : ℝ := 720

theorem find_track_circumference
  (A B : ℝ)
  (uA uB : ℝ)
  (h1 : A = 0)
  (h2 : B = track_circumference / 2)
  (h3 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = 150 / uB)
  (h4 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = (track_circumference - 90) / uA)
  (h5 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = 1.5 * track_circumference / uA) :
  track_circumference = 720 :=
by sorry

end find_track_circumference_l147_147434


namespace max_m_sq_plus_n_sq_l147_147796

theorem max_m_sq_plus_n_sq (m n : ℤ) (h1 : 1 ≤ m ∧ m ≤ 1981) (h2 : 1 ≤ n ∧ n ≤ 1981) (h3 : (n^2 - m*n - m^2)^2 = 1) : m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_sq_plus_n_sq_l147_147796


namespace second_number_less_than_twice_first_l147_147290

theorem second_number_less_than_twice_first (x y z : ℤ) (h1 : y = 37) (h2 : x + y = 57) (h3 : y = 2 * x - z) : z = 3 :=
by
  sorry

end second_number_less_than_twice_first_l147_147290


namespace max_black_balls_C_is_22_l147_147548

-- Define the given parameters
noncomputable def balls_A : ℕ := 100
noncomputable def black_balls_A : ℕ := 15
noncomputable def balls_B : ℕ := 50
noncomputable def balls_C : ℕ := 80
noncomputable def probability : ℚ := 101 / 600

-- Define the maximum number of black balls in box C given the conditions
theorem max_black_balls_C_is_22 (y : ℕ) (h : (1/3 * (black_balls_A / balls_A) + 1/3 * (y / balls_B) + 1/3 * (22 / balls_C)) = probability  ) :
  ∃ (x : ℕ), x ≤ 22 := sorry

end max_black_balls_C_is_22_l147_147548


namespace bryden_receives_22_50_dollars_l147_147877

-- Define the face value of a regular quarter
def face_value_regular : ℝ := 0.25

-- Define the number of regular quarters Bryden has
def num_regular_quarters : ℕ := 4

-- Define the face value of the special quarter
def face_value_special : ℝ := face_value_regular * 2

-- The collector pays 15 times the face value for regular quarters
def multiplier : ℝ := 15

-- Calculate the total face value of all quarters
def total_face_value : ℝ := (num_regular_quarters * face_value_regular) + face_value_special

-- Calculate the total amount Bryden will receive
def total_amount_received : ℝ := multiplier * total_face_value

-- Prove that the total amount Bryden will receive is $22.50
theorem bryden_receives_22_50_dollars : total_amount_received = 22.50 :=
by
  sorry

end bryden_receives_22_50_dollars_l147_147877


namespace original_sheets_count_is_115_l147_147582

def find_sheets_count (S P : ℕ) : Prop :=
  -- Ann's condition: all papers are used leaving 100 flyers
  S - P = 100 ∧
  -- Bob's condition: all bindings used leaving 35 sheets of paper
  5 * P = S - 35

theorem original_sheets_count_is_115 (S P : ℕ) (h : find_sheets_count S P) : S = 115 :=
by
  sorry

end original_sheets_count_is_115_l147_147582


namespace root_implies_m_values_l147_147482

theorem root_implies_m_values (m : ℝ) :
  (∃ x : ℝ, x = 1 ∧ (m + 2) * x^2 - 2 * x + m^2 - 2 * m - 6 = 0) →
  (m = 3 ∨ m = -2) :=
by
  sorry

end root_implies_m_values_l147_147482


namespace simplified_fraction_of_num_l147_147305

def num : ℚ := 368 / 100

theorem simplified_fraction_of_num : num = 92 / 25 := by
  sorry

end simplified_fraction_of_num_l147_147305


namespace sum_of_solutions_l147_147614

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l147_147614


namespace dennis_total_cost_l147_147595

-- Define the cost of items and quantities
def cost_pants : ℝ := 110.0
def cost_socks : ℝ := 60.0
def quantity_pants : ℝ := 4
def quantity_socks : ℝ := 2
def discount_rate : ℝ := 0.30

-- Define the total costs before and after discount
def total_cost_pants_before_discount : ℝ := cost_pants * quantity_pants
def total_cost_socks_before_discount : ℝ := cost_socks * quantity_socks
def total_cost_before_discount : ℝ := total_cost_pants_before_discount + total_cost_socks_before_discount
def total_discount : ℝ := total_cost_before_discount * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - total_discount

-- Theorem asserting the total amount after discount
theorem dennis_total_cost : total_cost_after_discount = 392 := by 
  sorry

end dennis_total_cost_l147_147595


namespace most_likely_number_of_red_balls_l147_147994

-- Define the total number of balls and the frequency of picking red balls as given in the conditions
def total_balls : ℕ := 20
def frequency_red : ℝ := 0.8

-- State the equivalent proof problem: Prove that the most likely number of red balls is 16
theorem most_likely_number_of_red_balls : frequency_red * (total_balls : ℝ) = 16 := by
  sorry

end most_likely_number_of_red_balls_l147_147994


namespace four_people_complete_task_in_18_days_l147_147854

theorem four_people_complete_task_in_18_days :
  (forall r : ℝ, (3 * 24 * r = 1) → (4 * 18 * r = 1)) :=
by
  intro r
  intro h
  sorry

end four_people_complete_task_in_18_days_l147_147854


namespace circle_intersection_solution_count_l147_147874

variables {S1 S2 : Circle} {A B : Point}
variables (O1 O2 : Point) (a : ℝ)

theorem circle_intersection_solution_count :
  let O1O2 := dist O1 O2 in
  O1 ∈ S1 ∧ O2 ∈ S2 ∧ A ∈ S1 ∩ S2 ∧ B ∈ S1 ∩ S2 →
  (if O1O2 > a / 2 then 2
   else if O1O2 = a / 2 then 1
   else 0) = ?n := sorry

end circle_intersection_solution_count_l147_147874


namespace probability_non_expired_bags_l147_147446

theorem probability_non_expired_bags :
  let total_bags := 5
  let expired_bags := 2
  let selected_bags := 2
  let total_combinations := Nat.choose total_bags selected_bags
  let non_expired_bags := total_bags - expired_bags
  let favorable_outcomes := Nat.choose non_expired_bags selected_bags
  (favorable_outcomes : ℚ) / (total_combinations : ℚ) = 3 / 10 := by
  sorry

end probability_non_expired_bags_l147_147446


namespace sufficient_not_necessary_condition_l147_147229

theorem sufficient_not_necessary_condition (a b : ℝ) (h1 : a > 1) (h2 : b > 2) : a + b > 3 :=
by
  sorry

end sufficient_not_necessary_condition_l147_147229


namespace sum_of_solutions_equation_l147_147611

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l147_147611


namespace Nicky_profit_l147_147830

-- Definitions for conditions
def card1_value : ℕ := 8
def card2_value : ℕ := 8
def received_card_value : ℕ := 21

-- The statement to be proven
theorem Nicky_profit : (received_card_value - (card1_value + card2_value)) = 5 :=
by
  sorry

end Nicky_profit_l147_147830


namespace larger_number_hcf_lcm_l147_147431

theorem larger_number_hcf_lcm (a b : ℕ) (hcf : ℕ) (factor1 factor2 : ℕ) 
  (h_hcf : hcf = 20) 
  (h_factor1 : factor1 = 13) 
  (h_factor2 : factor2 = 14) 
  (h_ab_hcf : Nat.gcd a b = hcf)
  (h_ab_lcm : Nat.lcm a b = hcf * factor1 * factor2) :
  max a b = 280 :=
by 
  sorry

end larger_number_hcf_lcm_l147_147431


namespace q_polynomial_sum_roots_l147_147820

-- Given: The polynomial Q(x) = x^3 + ax^2 + bx + c has real coefficients a, b, and c.
-- There exists a complex number u such that the roots of Q(z) are u + 2i, u - 2i, and 2u + 3.

theorem q_polynomial_sum_roots {a b c m n : ℝ} :
  let u := m + n * complex.I in
  let Q := λ x: ℝ, x^3 + a * x^2 + b * x + c in
  (Q (u + 2 * complex.I)).im = 0 ∧ (Q (u - 2 * complex.I)).im = 0 ∧ 
  (Q (2 * u + 3)).im = 0 →
  a + b + c = -2 * m^3 - 2 * m^2 + 7 * m - 4 * n^2 + 1 :=
sorry

end q_polynomial_sum_roots_l147_147820


namespace find_parallelogram_base_length_l147_147783

variable (A h b : ℕ)
variable (parallelogram_area : A = 240)
variable (parallelogram_height : h = 10)
variable (area_formula : A = b * h)

theorem find_parallelogram_base_length : b = 24 :=
by
  have h₁ : A = 240 := parallelogram_area
  have h₂ : h = 10 := parallelogram_height
  have h₃ : A = b * h := area_formula
  sorry

end find_parallelogram_base_length_l147_147783


namespace breadth_of_rectangular_plot_l147_147868

theorem breadth_of_rectangular_plot (b l : ℝ) (h1 : l = 3 * b) (h2 : l * b = 432) : b = 12 := 
sorry

end breadth_of_rectangular_plot_l147_147868


namespace most_likely_number_of_red_balls_l147_147991

-- Define the conditions
def total_balls : ℕ := 20
def red_ball_frequency : ℝ := 0.8

-- Define the statement we want to prove
theorem most_likely_number_of_red_balls : red_ball_frequency * total_balls = 16 :=
by sorry

end most_likely_number_of_red_balls_l147_147991


namespace correct_calculation_l147_147425

theorem correct_calculation (a b c d : ℤ) (h1 : a = -1) (h2 : b = -3) (h3 : c = 3) (h4 : d = -3) :
  a * b = c :=
by 
  rw [h1, h2]
  exact h3.symm

end correct_calculation_l147_147425


namespace monotonicity_of_f_sum_of_squares_of_roots_l147_147368

noncomputable def f (x a : Real) : Real := Real.log x - a * x^2

theorem monotonicity_of_f (a : Real) :
  (a ≤ 0 → ∀ x y : Real, 0 < x → x < y → f x a < f y a) ∧
  (a > 0 → ∀ x y : Real, 0 < x → x < Real.sqrt (1/(2 * a)) → Real.sqrt (1/(2 * a)) < y → f x a < f (Real.sqrt (1/(2 * a))) a ∧ f (Real.sqrt (1/(2 * a))) a > f y a) :=
by sorry

theorem sum_of_squares_of_roots (a x1 x2 : Real) (h1 : f x1 a = 0) (h2 : f x2 a = 0) (h3 : x1 ≠ x2) :
  x1^2 + x2^2 > 2 * Real.exp 1 :=
by sorry

end monotonicity_of_f_sum_of_squares_of_roots_l147_147368


namespace vasya_read_entire_book_l147_147725

theorem vasya_read_entire_book :
  let day1 := 1 / 2
  let day2 := 1 / 3 * (1 - day1)
  let days12 := day1 + day2
  let day3 := 1 / 2 * days12
  (days12 + day3) = 1 :=
by
  sorry

end vasya_read_entire_book_l147_147725


namespace decimal_to_fraction_l147_147311

theorem decimal_to_fraction :
  (368 / 100 : ℚ) = (92 / 25 : ℚ) := by
  sorry

end decimal_to_fraction_l147_147311


namespace determine_s_l147_147504

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem determine_s (s : ℝ) (h : g (-3) s = 0) : s = -192 :=
by
  sorry

end determine_s_l147_147504


namespace followers_after_one_year_l147_147600

theorem followers_after_one_year :
  let initial_followers := 100000
  let daily_new_followers := 1000
  let unfollowers_per_year := 20000
  let days_per_year := 365
  initial_followers + (daily_new_followers * days_per_year - unfollowers_per_year) = 445000 :=
by
  sorry

end followers_after_one_year_l147_147600


namespace original_denominator_is_two_l147_147181

theorem original_denominator_is_two (d : ℕ) : 
  (∃ d : ℕ, 2 * (d + 4) = 6) → d = 2 :=
by sorry

end original_denominator_is_two_l147_147181


namespace polynomial_sum_correct_l147_147398

def f (x : ℝ) : ℝ := -4 * x^3 + 2 * x^2 - x - 5
def g (x : ℝ) : ℝ := -6 * x^3 - 7 * x^2 + 4 * x - 2
def h (x : ℝ) : ℝ := 2 * x^3 + 8 * x^2 + 6 * x + 3
def sum_polynomials (x : ℝ) : ℝ := -8 * x^3 + 3 * x^2 + 9 * x - 4

theorem polynomial_sum_correct (x : ℝ) : f x + g x + h x = sum_polynomials x :=
by sorry

end polynomial_sum_correct_l147_147398


namespace max_cookies_l147_147394

-- Definitions for the conditions
def John_money : ℕ := 2475
def cookie_cost : ℕ := 225

-- Statement of the problem
theorem max_cookies (x : ℕ) : cookie_cost * x ≤ John_money → x ≤ 11 :=
sorry

end max_cookies_l147_147394


namespace simplify_vector_expression_l147_147407

-- Definitions for vectors
variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Defining the vectors
variables (AB CA BD CD : A)

-- A definition using the head-to-tail addition of vectors.
def vector_add (v1 v2 : A) : A := v1 + v2

-- Statement to prove
theorem simplify_vector_expression :
  vector_add (vector_add AB CA) BD = CD :=
sorry

end simplify_vector_expression_l147_147407


namespace maria_high_school_students_l147_147960

variable (M D : ℕ)

theorem maria_high_school_students (h1 : M = 4 * D) (h2 : M - D = 1800) : M = 2400 :=
by
  sorry

end maria_high_school_students_l147_147960


namespace total_legs_in_park_l147_147944

theorem total_legs_in_park :
  let dogs := 109
  let cats := 37
  let birds := 52
  let spiders := 19
  let dog_legs := 4
  let cat_legs := 4
  let bird_legs := 2
  let spider_legs := 8
  dogs * dog_legs + cats * cat_legs + birds * bird_legs + spiders * spider_legs = 840 := by
  sorry

end total_legs_in_park_l147_147944


namespace minimum_passing_rate_l147_147102

-- Define the conditions as hypotheses
variable (total_students : ℕ)
variable (correct_q1 : ℕ)
variable (correct_q2 : ℕ)
variable (correct_q3 : ℕ)
variable (correct_q4 : ℕ)
variable (correct_q5 : ℕ)
variable (pass_threshold : ℕ)

-- Assume all percentages are converted to actual student counts based on total_students
axiom students_answered_q1_correctly : correct_q1 = total_students * 81 / 100
axiom students_answered_q2_correctly : correct_q2 = total_students * 91 / 100
axiom students_answered_q3_correctly : correct_q3 = total_students * 85 / 100
axiom students_answered_q4_correctly : correct_q4 = total_students * 79 / 100
axiom students_answered_q5_correctly : correct_q5 = total_students * 74 / 100
axiom passing_criteria : pass_threshold = 3

-- Define the main theorem statement to be proven
theorem minimum_passing_rate (total_students : ℕ) :
  (total_students - (total_students * 19 / 100 + total_students * 9 / 100 + 
  total_students * 15 / 100 + total_students * 21 / 100 + 
  total_students * 26 / 100) / pass_threshold) / total_students * 100 ≥ 70 :=
  by sorry

end minimum_passing_rate_l147_147102


namespace probability_different_colors_l147_147095

-- Define the total number of chips
def total_chips : ℕ := 6 + 5 + 4 + 3

-- Define the probabilities of drawing each color and then another color
def prob_blue_not_blue := (6 / total_chips : ℚ) * (12 / total_chips)
def prob_red_not_red := (5 / total_chips : ℚ) * (13 / total_chips)
def prob_yellow_not_yellow := (4 / total_chips : ℚ) * (14 / total_chips)
def prob_green_not_green := (3 / total_chips : ℚ) * (15 / total_chips)

-- Sum of probabilities for drawing two chips of different colors
def prob_different_colors := prob_blue_not_blue + prob_red_not_red + prob_yellow_not_yellow + prob_green_not_green

-- The theorem to be proved
theorem probability_different_colors :
  prob_different_colors = 137 / 162 :=
by
  sorry

end probability_different_colors_l147_147095


namespace unique_triple_l147_147780

theorem unique_triple (x y z : ℤ) (h₁ : x + y = z) (h₂ : y + z = x) (h₃ : z + x = y) :
  (x = 0) ∧ (y = 0) ∧ (z = 0) :=
sorry

end unique_triple_l147_147780


namespace simplify_fraction_l147_147406

theorem simplify_fraction :
  (45 * (14 / 25) * (1 / 18) * (5 / 11) : ℚ) = 7 / 11 := 
by sorry

end simplify_fraction_l147_147406


namespace time_to_pass_l147_147028
-- Import the Mathlib library

-- Define the lengths of the trains
def length_train1 := 150 -- meters
def length_train2 := 150 -- meters

-- Define the speeds of the trains in km/h
def speed_train1_kmh := 95 -- km/h
def speed_train2_kmh := 85 -- km/h

-- Convert speeds to m/s
def speed_train1_ms := (speed_train1_kmh * 1000) / 3600 -- meters per second
def speed_train2_ms := (speed_train2_kmh * 1000) / 3600 -- meters per second

-- Calculate the relative speed in m/s (since they move in opposite directions, the relative speed is additive)
def relative_speed_ms := speed_train1_ms + speed_train2_ms -- meters per second

-- Calculate the total distance to be covered (sum of the lengths of the trains)
def total_length := length_train1 + length_train2 -- meters

-- State the theorem: the time taken for the trains to pass each other
theorem time_to_pass :
  total_length / relative_speed_ms = 6 := by
  sorry

end time_to_pass_l147_147028


namespace solution_set_inequality_l147_147291

theorem solution_set_inequality (x : ℝ) : 
  (∃ x, (x-1)/((x^2) - x - 30) > 0) ↔ (x > -5 ∧ x < 1) ∨ (x > 6) :=
by
  sorry

end solution_set_inequality_l147_147291


namespace inequality_solutions_l147_147843

theorem inequality_solutions (n : ℕ) (h : n > 0) : n^3 - n < n! ↔ (n = 1 ∨ n ≥ 6) := 
by
  sorry

end inequality_solutions_l147_147843


namespace calc_factorial_sum_l147_147054

theorem calc_factorial_sum : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 2 * Nat.factorial 6 = 40320 := 
sorry

end calc_factorial_sum_l147_147054


namespace complex_number_in_second_quadrant_l147_147439

-- We must prove that the complex number z = i / (1 - i) lies in the second quadrant
theorem complex_number_in_second_quadrant : let z := (Complex.I / (1 - Complex.I)) in (z.re < 0) ∧ (z.im > 0) :=
by
  let z := (Complex.I / (1 - Complex.I))
  have h1 : (1 - Complex.I).conj = 1 + Complex.I := by sorry
  have h2 : (Complex.I * (1 + Complex.I)) = -1 + Complex.I := by sorry
  have h3 : (1 - Complex.I) * (1 + Complex.I) = 2 := by sorry
  have h4 : z = (-1 + Complex.I) / 2 := by sorry
  have h5 : z.re = -1 / 2 := by sorry
  have h6 : z.im = 1 / 2 := by sorry
  show (z.re < 0) ∧ (z.im > 0), from and.intro (by norm_num) (by norm_num)

end complex_number_in_second_quadrant_l147_147439


namespace most_likely_number_of_red_balls_l147_147993

-- Define the total number of balls and the frequency of picking red balls as given in the conditions
def total_balls : ℕ := 20
def frequency_red : ℝ := 0.8

-- State the equivalent proof problem: Prove that the most likely number of red balls is 16
theorem most_likely_number_of_red_balls : frequency_red * (total_balls : ℝ) = 16 := by
  sorry

end most_likely_number_of_red_balls_l147_147993


namespace num_multiples_6_not_12_lt_300_l147_147656

theorem num_multiples_6_not_12_lt_300 : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k < 300 ∧ k % 6 = 0 ∧ k % 12 ≠ 0 → ∃ m : ℕ, k = 6 * (2 * m - 1) ∧ 1 ≤ m ∧ m ≤ 25 := 
by
  sorry

end num_multiples_6_not_12_lt_300_l147_147656


namespace find_x_l147_147089

theorem find_x (x : ℝ) (h : 2 * x - 3 * x + 5 * x = 80) : x = 20 :=
by 
  -- placeholder for proof
  sorry 

end find_x_l147_147089


namespace number_of_six_digit_palindromes_l147_147380

theorem number_of_six_digit_palindromes : 
  let count_palindromes : ℕ := 9 * 10 * 10 in
  count_palindromes = 900 :=
by
  sorry

end number_of_six_digit_palindromes_l147_147380


namespace books_sold_l147_147839

theorem books_sold (initial_books sold_books remaining_books : ℕ) 
  (h_initial : initial_books = 242) 
  (h_remaining : remaining_books = 105)
  (h_relation : sold_books = initial_books - remaining_books) :
  sold_books = 137 := 
by
  sorry

end books_sold_l147_147839


namespace estimated_students_in_sport_A_correct_l147_147876

noncomputable def total_students_surveyed : ℕ := 80
noncomputable def students_in_sport_A_surveyed : ℕ := 30
noncomputable def total_school_population : ℕ := 800
noncomputable def proportion_sport_A : ℚ := students_in_sport_A_surveyed / total_students_surveyed
noncomputable def estimated_students_in_sport_A : ℚ := total_school_population * proportion_sport_A

theorem estimated_students_in_sport_A_correct :
  estimated_students_in_sport_A = 300 :=
by
  sorry

end estimated_students_in_sport_A_correct_l147_147876


namespace slower_train_passing_time_l147_147572

/--
Two goods trains, each 500 meters long, are running in opposite directions on parallel tracks. 
Their respective speeds are 45 kilometers per hour and 15 kilometers per hour. 
Prove that the time taken by the slower train to pass the driver of the faster train is 30 seconds.
-/
theorem slower_train_passing_time : 
  ∀ (distance length_speed : ℝ), 
    distance = 500 →
    ∃ (v1 v2 : ℝ), 
      v1 = 45 * (1000 / 3600) → 
      v2 = 15 * (1000 / 3600) →
      (distance / ((v1 + v2) * (3/50)) = 30) :=
by
  sorry

end slower_train_passing_time_l147_147572


namespace range_of_m_l147_147643

variable {x m : ℝ}

def condition_p (x : ℝ) : Prop := |x - 3| ≤ 2
def condition_q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

theorem range_of_m (m : ℝ) :
  (∀ x, ¬(condition_p x) → ¬(condition_q x m)) ∧ ¬(∀ x, ¬(condition_q x m) → ¬(condition_p x)) →
  2 < m ∧ m < 4 := 
sorry

end range_of_m_l147_147643


namespace rectangle_area_l147_147332

theorem rectangle_area (w l : ℕ) (h1 : l = w + 8) (h2 : 2 * l + 2 * w = 176) :
  l * w = 1920 :=
by
  sorry

end rectangle_area_l147_147332


namespace sum_of_solutions_l147_147632

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l147_147632


namespace max_sector_area_l147_147472

theorem max_sector_area (r θ : ℝ) (h₁ : 2 * r + r * θ = 16) : 
  (∃ A : ℝ, A = 1/2 * r^2 * θ ∧ A ≤ 16) ∧ (∃ r θ, r = 4 ∧ θ = 2 ∧ 1/2 * r^2 * θ = 16) := 
by
  sorry

end max_sector_area_l147_147472


namespace simplify_expression_calculate_expression_l147_147136

-- Problem 1
theorem simplify_expression (x : ℝ) : 
  (x + 1) * (x + 1) - x * (x + 1) = x + 1 := by
  sorry

-- Problem 2
theorem calculate_expression : 
  (-1 : ℝ) ^ 2023 + 2 ^ (-2 : ℝ) + 4 * (Real.cos (Real.pi / 6))^2 = 9 / 4 := by
  sorry

end simplify_expression_calculate_expression_l147_147136


namespace sum_prime_factors_of_143_l147_147009

theorem sum_prime_factors_of_143 : 
  let primes := {p : ℕ | p.prime ∧ p ∣ 143} in
  ∑ p in primes, p = 24 := 
by
  sorry

end sum_prime_factors_of_143_l147_147009


namespace arithmetic_sequence_sum_l147_147498

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_sum :
  (a 3 = 5) →
  (a 4 + a 8 = 22) →
  ( ∑ i in range 8, a (i + 1) = 64) :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sum_l147_147498


namespace ticket_price_difference_l147_147156

def pre_bought_payment (number_pre : ℕ) (price_pre : ℕ) : ℕ :=
  number_pre * price_pre

def gate_payment (number_gate : ℕ) (price_gate : ℕ) : ℕ :=
  number_gate * price_gate

theorem ticket_price_difference :
  ∀ (number_pre number_gate price_pre price_gate : ℕ),
  number_pre = 20 →
  price_pre = 155 →
  number_gate = 30 →
  price_gate = 200 →
  gate_payment number_gate price_gate - pre_bought_payment number_pre price_pre = 2900 :=
by {
  intros,
  sorry
}

end ticket_price_difference_l147_147156


namespace root_ratios_equal_l147_147179

theorem root_ratios_equal (a : ℝ) (ha : 0 < a)
  (hroots : ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁^3 + 1 = a * x₁ ∧ x₂^3 + 1 = a * x₂ ∧ x₂ / x₁ = 2018) :
  ∃ y₁ y₂ : ℝ, 0 < y₁ ∧ 0 < y₂ ∧ y₁^3 + 1 = a * y₁^2 ∧ y₂^3 + 1 = a * y₂^2 ∧ y₂ / y₁ = 2018 :=
sorry

end root_ratios_equal_l147_147179


namespace find_f2_l147_147413

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder function definition

theorem find_f2 (h : ∀ x : ℝ, f x + 2 * f (1 - x) = x^3 + 1) : f 2 = -3 :=
by
  -- Lean proof goes here
  sorry

end find_f2_l147_147413


namespace rr_sr_sum_le_one_l147_147542

noncomputable def rr_sr_le_one (r s : ℝ) (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_sum : r + s = 1) : Prop :=
  r^r * s^s + r^s * s^r ≤ 1

theorem rr_sr_sum_le_one {r s : ℝ} (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_sum : r + s = 1) : rr_sr_le_one r s h_pos_r h_pos_s h_sum :=
  sorry

end rr_sr_sum_le_one_l147_147542


namespace calc_expression_eq_3_solve_quadratic_eq_l147_147168

-- Problem 1
theorem calc_expression_eq_3 :
  (-1 : ℝ) ^ 2020 + (- (1 / 2)⁻¹) - (3.14 - Real.pi) ^ 0 + abs (-3) = 3 :=
by
  sorry

-- Problem 2
theorem solve_quadratic_eq {x : ℝ} :
  (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2 / 3) :=
by
  sorry

end calc_expression_eq_3_solve_quadratic_eq_l147_147168


namespace maximum_of_fraction_l147_147663

theorem maximum_of_fraction (x : ℝ) : (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 := by
  sorry

end maximum_of_fraction_l147_147663


namespace not_sum_six_odd_squares_l147_147192

-- Definition stating that a number is odd.
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- Given that the square of any odd number is 1 modulo 8.
lemma odd_square_mod_eight (n : ℕ) (h : is_odd n) : (n^2) % 8 = 1 :=
sorry

-- Main theorem stating that 1986 cannot be the sum of six squares of odd numbers.
theorem not_sum_six_odd_squares : ¬ ∃ n1 n2 n3 n4 n5 n6 : ℕ, 
    is_odd n1 ∧ is_odd n2 ∧ is_odd n3 ∧ is_odd n4 ∧ is_odd n5 ∧ is_odd n6 ∧
    n1^2 + n2^2 + n3^2 + n4^2 + n5^2 + n6^2 = 1986 :=
sorry

end not_sum_six_odd_squares_l147_147192


namespace dividend_percentage_l147_147878

theorem dividend_percentage (face_value : ℝ) (investment : ℝ) (roi : ℝ) (dividend_percentage : ℝ) 
    (h1 : face_value = 40) 
    (h2 : investment = 20) 
    (h3 : roi = 0.25) : dividend_percentage = 12.5 := 
  sorry

end dividend_percentage_l147_147878


namespace rectangle_area_l147_147568

theorem rectangle_area
  (width : ℕ) (length : ℕ)
  (h1 : width = 7)
  (h2 : length = 4 * width) :
  length * width = 196 := by
  sorry

end rectangle_area_l147_147568


namespace range_of_a_plus_2014b_l147_147481

theorem range_of_a_plus_2014b (a b : ℝ) (h1 : a < b) (h2 : |(Real.log a) / (Real.log 2)| = |(Real.log b) / (Real.log 2)|) :
  ∃ c : ℝ, c > 2015 ∧ ∀ x : ℝ, a + 2014 * b = x → x > 2015 := by
  sorry

end range_of_a_plus_2014b_l147_147481


namespace green_ish_count_l147_147940

theorem green_ish_count (total : ℕ) (blue_ish : ℕ) (both : ℕ) (neither : ℕ) (green_ish : ℕ) :
  total = 150 ∧ blue_ish = 90 ∧ both = 40 ∧ neither = 30 → green_ish = 70 :=
by
  sorry

end green_ish_count_l147_147940


namespace smallest_portion_is_five_thirds_l147_147166

theorem smallest_portion_is_five_thirds
    (a1 a2 a3 a4 a5 : ℚ)
    (h1 : a2 = a1 + 1)
    (h2 : a3 = a1 + 2)
    (h3 : a4 = a1 + 3)
    (h4 : a5 = a1 + 4)
    (h_sum : a1 + a2 + a3 + a4 + a5 = 100)
    (h_cond : (1 / 7) * (a3 + a4 + a5) = a1 + a2) :
    a1 = 5 / 3 :=
by
  sorry

end smallest_portion_is_five_thirds_l147_147166


namespace sin_sum_bound_l147_147520

theorem sin_sum_bound (x : ℝ) : 
  |(Real.sin x) + (Real.sin (Real.sqrt 2 * x))| < 2 - 1 / (100 * (x^2 + 1)) :=
by sorry

end sin_sum_bound_l147_147520


namespace prob_grid_completely_black_l147_147935

theorem prob_grid_completely_black :
  let prob_center_black := (1 / 2 : ℝ)
  let prob_pair_half_black := (3 / 4 : ℝ)
  let prob_all_pairs :=
        (prob_pair_half_black * prob_pair_half_black * prob_pair_half_black * prob_pair_half_black)
  in
  (prob_center_black * prob_all_pairs) = (81 / 512 : ℝ) :=
by
  let prob_center_black := (1 / 2 : ℝ)
  let prob_pair_half_black := (3 / 4 : ℝ)
  let prob_all_pairs :=
        (prob_pair_half_black * prob_pair_half_black * prob_pair_half_black * prob_pair_half_black)
  let result := prob_center_black * prob_all_pairs
  have calculation : result = (1 / 2 * (3 / 4) ^ 4) := rfl
  have expected : (81 / 512 : ℝ) = (1 / 2 * (3 / 4) ^ 4) := by norm_num
  exact expected.symm ▸ calculation

end prob_grid_completely_black_l147_147935


namespace ratio_A_B_l147_147541

theorem ratio_A_B (A B C : ℕ) (h1 : A + B + C = 98) (h2 : B = 30) (h3 : 5 * C = 8 * B) : A / B = 2 / 3 := 
by sorry

end ratio_A_B_l147_147541


namespace equal_area_of_second_square_l147_147941

/-- 
In an isosceles right triangle with legs of length 25√2 cm, if a square is inscribed such that two 
of its vertices lie on one leg and one vertex on each of the hypotenuse and the other leg, 
and the area of the square is 625 cm², prove that the area of another inscribed square 
(with one vertex each on the hypotenuse and one leg, and two vertices on the other leg) is also 625 cm².
-/
theorem equal_area_of_second_square 
  (a b : ℝ) (h1 : a = 25 * Real.sqrt 2)  
  (h2 : b = 625) :
  ∃ c : ℝ, c = 625 :=
by
  sorry

end equal_area_of_second_square_l147_147941


namespace periodicity_iff_condition_l147_147367

-- Define the given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x, f (-x) = f x)

-- State the problem
theorem periodicity_iff_condition :
  (∀ x, f (1 - x) = f (1 + x)) ↔ (∀ x, f (x + 2) = f x) :=
sorry

end periodicity_iff_condition_l147_147367


namespace range_of_a_has_three_integer_solutions_l147_147494

theorem range_of_a_has_three_integer_solutions (a : ℝ) :
  (∃ (x : ℤ → ℝ), (2 * x - 1 > 3) ∧ (x ≤ 2 * a - 1) ∧ (x = 3 ∨ x = 4 ∨ x = 5)) → (3 ≤ a ∧ a < 3.5) :=
sorry

end range_of_a_has_three_integer_solutions_l147_147494


namespace quadratic_roots_equation_l147_147669

theorem quadratic_roots_equation (α β : ℝ) 
  (h1 : (α + β) / 2 = 8) 
  (h2 : Real.sqrt (α * β) = 15) : 
  x^2 - (α + β) * x + α * β = 0 ↔ x^2 - 16 * x + 225 = 0 := 
by
  sorry

end quadratic_roots_equation_l147_147669


namespace least_positive_number_of_linear_combination_of_24_20_l147_147674

-- Define the conditions as integers
def problem_statement (x y : ℤ) : Prop := 24 * x + 20 * y = 4

theorem least_positive_number_of_linear_combination_of_24_20 :
  ∃ (x y : ℤ), (24 * x + 20 * y = 4) := 
by
  sorry

end least_positive_number_of_linear_combination_of_24_20_l147_147674


namespace remainder_of_h_x6_l147_147952

def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

noncomputable def remainder_when_h_x6_divided_by_h (x : ℝ) : ℝ :=
  let hx := h x
  let hx6 := h (x^6)
  hx6 - 6 * hx

theorem remainder_of_h_x6 (x : ℝ) : remainder_when_h_x6_divided_by_h x = 6 :=
  sorry

end remainder_of_h_x6_l147_147952


namespace least_subtraction_divisibility_l147_147029

theorem least_subtraction_divisibility :
  ∃ k : ℕ, 427398 - k = 14 * n ∧ k = 6 :=
by
  use 6
  sorry

end least_subtraction_divisibility_l147_147029


namespace perfect_square_unique_n_l147_147782

theorem perfect_square_unique_n (n : ℕ) (hn : n > 0) : 
  (∃ m : ℕ, 2^n + 12^n + 2011^n = m^2) ↔ n = 1 := by
  sorry

end perfect_square_unique_n_l147_147782


namespace sqrt_simplification_l147_147135

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l147_147135


namespace sum_of_solutions_l147_147624
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l147_147624


namespace find_values_f_l147_147649

open Real

noncomputable def f (ω A x : ℝ) : ℝ := 2 * sin (ω * x) * cos (ω * x) + 2 * A * (cos (ω * x))^2 - A

theorem find_values_f (θ : ℝ) (A : ℝ) (ω : ℝ) (hA : A > 0) (hω : ω = 1)
  (h1 : π / 6 < θ) (h2 : θ < π / 3) (h3 : f ω A θ = 2 / 3) :
  f ω A (π / 3 - θ) = (1 + 2 * sqrt 6) / 3 :=
  sorry

end find_values_f_l147_147649


namespace find_digit_A_l147_147685

theorem find_digit_A (A M C : ℕ) (h1 : A < 10) (h2 : M < 10) (h3 : C < 10) (h4 : (100 * A + 10 * M + C) * (A + M + C) = 2008) : 
  A = 2 :=
sorry

end find_digit_A_l147_147685


namespace sum_of_three_consecutive_is_50_l147_147678

-- Define a sequence of integers
def seq : Fin 8 → ℕ
| 0 => 11
| 1 => 12
| 2 => 27
| 3 => 11
| 4 => 12
| 5 => 27
| 6 => 11
| 7 => 12

-- Theorem: The sum of any three consecutive numbers in the sequence is 50
theorem sum_of_three_consecutive_is_50 (n : Fin (8 - 2)) : 
  seq n + seq (n + 1) + seq (n + 2) = 50 :=
by
  fin_cases n
  case 0 => simp [seq]
  case 1 => simp [seq]
  case 2 => simp [seq]
  case 3 => simp [seq]
  case 4 => simp [seq]
  case 5 => simp [seq]
  -- This skips the proof details; the proof should show the sum matches 50 in all cases.
  sorry

-- Use sorry to skip the detailed proof

end sum_of_three_consecutive_is_50_l147_147678


namespace area_region_inside_but_outside_l147_147880

noncomputable def area_diff (side_large side_small : ℝ) : ℝ :=
  (side_large ^ 2) - (side_small ^ 2)

theorem area_region_inside_but_outside (h_large : 10 > 0) (h_small : 4 > 0) :
  area_diff 10 4 = 84 :=
by
  -- The proof steps would go here
  sorry

end area_region_inside_but_outside_l147_147880


namespace number_of_female_students_l147_147027

theorem number_of_female_students
    (F : ℕ)  -- Number of female students
    (avg_all : ℝ)  -- Average score for all students
    (avg_male : ℝ)  -- Average score for male students
    (avg_female : ℝ)  -- Average score for female students
    (num_male : ℕ)  -- Number of male students
    (h_avg_all : avg_all = 90)
    (h_avg_male : avg_male = 82)
    (h_avg_female : avg_female = 92)
    (h_num_male : num_male = 8)
    (h_avg : avg_all * (num_male + F) = avg_male * num_male + avg_female * F) :
  F = 32 :=
by
  sorry

end number_of_female_students_l147_147027


namespace first_year_with_digit_sum_seven_l147_147810

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_with_digit_sum_seven : ∃ y, y > 2023 ∧ sum_of_digits y = 7 ∧ ∀ z, z > 2023 ∧ z < y → sum_of_digits z ≠ 7 :=
by
  use 2032
  sorry

end first_year_with_digit_sum_seven_l147_147810


namespace Carol_mother_carrots_l147_147055

theorem Carol_mother_carrots (carol_picked : ℕ) (total_good : ℕ) (total_bad : ℕ) (total_carrots : ℕ) (mother_picked : ℕ) :
  carol_picked = 29 → total_good = 38 → total_bad = 7 → total_carrots = total_good + total_bad → mother_picked = total_carrots - carol_picked → mother_picked = 16 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  sorry

end Carol_mother_carrots_l147_147055


namespace sum_of_integers_70_to_85_l147_147423

theorem sum_of_integers_70_to_85 :
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sum = 1240 :=
by
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sorry

end sum_of_integers_70_to_85_l147_147423


namespace lateral_surface_area_base_area_ratio_correct_l147_147886

noncomputable def lateral_surface_area_to_base_area_ratio
  (S P Q R : Type)
  (angle_PSR angle_SQR angle_PSQ : ℝ)
  (h_PSR : angle_PSR = π / 2)
  (h_SQR : angle_SQR = π / 4)
  (h_PSQ : angle_PSQ = 7 * π / 12)
  : ℝ :=
  π * (4 * Real.sqrt 3 - 3) / 13

theorem lateral_surface_area_base_area_ratio_correct
  {S P Q R : Type}
  (angle_PSR angle_SQR angle_PSQ : ℝ)
  (h_PSR : angle_PSR = π / 2)
  (h_SQR : angle_SQR = π / 4)
  (h_PSQ : angle_PSQ = 7 * π / 12) :
  lateral_surface_area_to_base_area_ratio S P Q R angle_PSR angle_SQR angle_PSQ
    h_PSR h_SQR h_PSQ = π * (4 * Real.sqrt 3 - 3) / 13 :=
  by sorry

end lateral_surface_area_base_area_ratio_correct_l147_147886


namespace clothing_order_equation_l147_147174

open Real

-- Definitions and conditions
def total_pieces : ℕ := 720
def initial_rate : ℕ := 48
def days_earlier : ℕ := 5

-- Statement that we need to prove
theorem clothing_order_equation (x : ℕ) :
    (720 / 48 : ℝ) - (720 / (x + 48) : ℝ) = 5 := 
sorry

end clothing_order_equation_l147_147174


namespace total_students_l147_147294

noncomputable def total_students_in_gym (F : ℕ) (T : ℕ) : Prop :=
  T = 26

theorem total_students (F T : ℕ) (h1 : 4 = T - F) (h2 : F / (F + 4) = 11 / 13) : total_students_in_gym F T :=
by sorry

end total_students_l147_147294


namespace repeating_decimal_to_fraction_l147_147778

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + 0.066666... ) : x = 11 / 30 :=
sorry

end repeating_decimal_to_fraction_l147_147778


namespace minimum_red_chips_l147_147739

variable (w b r : ℕ)

-- Define the conditions
def condition1 : Prop := b ≥ 3 * w / 4
def condition2 : Prop := b ≤ r / 4
def condition3 : Prop := 60 ≤ w + b ∧ w + b ≤ 80

-- Prove the minimum number of red chips r is 108
theorem minimum_red_chips (H1 : condition1 w b) (H2 : condition2 b r) (H3 : condition3 w b) : r ≥ 108 := 
sorry

end minimum_red_chips_l147_147739


namespace asha_remaining_money_l147_147337

theorem asha_remaining_money :
  let brother := 20
  let father := 40
  let mother := 30
  let granny := 70
  let savings := 100
  let total_money := brother + father + mother + granny + savings
  let spent := (3 / 4) * total_money
  let remaining := total_money - spent
  remaining = 65 :=
by
  sorry

end asha_remaining_money_l147_147337


namespace common_chord_length_l147_147724

theorem common_chord_length (r : ℝ) (h : r = 12) 
  (condition : ∀ (C₁ C₂ : Set (ℝ × ℝ)), 
      ((C₁ = {p : ℝ × ℝ | dist p (0, 0) = r}) ∧ 
       (C₂ = {p : ℝ × ℝ | dist p (12, 0) = r}) ∧
       (C₂ ∩ C₁ ≠ ∅))) : 
  ∃ chord_len : ℝ, chord_len = 12 * Real.sqrt 3 :=
by
  sorry

end common_chord_length_l147_147724


namespace miles_collection_height_l147_147121

-- Definitions based on conditions
def pages_per_inch_miles : ℕ := 5
def pages_per_inch_daphne : ℕ := 50
def daphne_height_inches : ℕ := 25
def longest_collection_pages : ℕ := 1250

-- Theorem to prove the height of Miles's book collection.
theorem miles_collection_height :
  (longest_collection_pages / pages_per_inch_miles) = 250 := by sorry

end miles_collection_height_l147_147121


namespace initial_avg_weight_proof_l147_147974

open Classical

variable (A B C D E : ℝ) (W : ℝ)

-- Given conditions
def initial_avg_weight_A_B_C : Prop := W = (A + B + C) / 3
def avg_with_D : Prop := (A + B + C + D) / 4 = 80
def E_weighs_D_plus_8 : Prop := E = D + 8
def avg_with_E_replacing_A : Prop := (B + C + D + E) / 4 = 79
def weight_of_A : Prop := A = 80

-- Question to prove
theorem initial_avg_weight_proof (h1 : initial_avg_weight_A_B_C W A B C)
                                 (h2 : avg_with_D A B C D)
                                 (h3 : E_weighs_D_plus_8 D E)
                                 (h4 : avg_with_E_replacing_A B C D E)
                                 (h5 : weight_of_A A) :
  W = 84 := by
  sorry

end initial_avg_weight_proof_l147_147974


namespace x_over_y_l147_147488

theorem x_over_y (x y : ℝ) (h : 16 * x = 0.24 * 90 * y) : x / y = 1.35 :=
sorry

end x_over_y_l147_147488


namespace six_digit_palindromes_count_l147_147381

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def is_non_zero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem six_digit_palindromes_count : 
  (∃a b c : ℕ, is_non_zero_digit a ∧ is_digit b ∧ is_digit c) → 
  (∃ n : ℕ, n = 900) :=
by
  sorry

end six_digit_palindromes_count_l147_147381


namespace inequality_proof_l147_147516

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_proof_l147_147516


namespace inspection_team_combinations_l147_147184

theorem inspection_team_combinations (total_members : ℕ)
                                     (men : ℕ)
                                     (women : ℕ)
                                     (team_size : ℕ)
                                     (men_in_team : ℕ)
                                     (women_in_team : ℕ)
                                     (h_total : total_members = 15)
                                     (h_men : men = 10)
                                     (h_women : women = 5)
                                     (h_team_size : team_size = 6)
                                     (h_men_in_team : men_in_team = 4)
                                     (h_women_in_team : women_in_team = 2) :
  nat.choose men men_in_team * nat.choose women women_in_team = nat.choose 10 4 * nat.choose 5 2 := 
by {
    rw [h_men, h_women, h_men_in_team, h_women_in_team],
    sorry
}

end inspection_team_combinations_l147_147184


namespace cakes_served_today_l147_147180

def lunch_cakes := 6
def dinner_cakes := 9
def total_cakes := lunch_cakes + dinner_cakes

theorem cakes_served_today : total_cakes = 15 := by
  sorry

end cakes_served_today_l147_147180


namespace find_f_of_conditions_l147_147414

theorem find_f_of_conditions (f : ℝ → ℝ) :
  (f 1 = 1) →
  (∀ x y : ℝ, f (x + y) = 3^y * f x + 2^x * f y) →
  (∀ x : ℝ, f x = 3^x - 2^x) :=
by
  intros h1 h2
  sorry

end find_f_of_conditions_l147_147414


namespace rotation_locus_l147_147194

-- Definitions for points and structure of the cube
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
(A : Point3D) (B : Point3D) (C : Point3D) (D : Point3D)
(E : Point3D) (F : Point3D) (G : Point3D) (H : Point3D)

-- Function to perform the required rotations and return the locus geometrical representation
noncomputable def locus_points_on_surface (c : Cube) : Set Point3D :=
sorry

-- Mathematical problem rephrased in Lean 4 statement
theorem rotation_locus (c : Cube) :
  locus_points_on_surface c = {c.D, c.A} ∪ {c.A, c.C} ∪ {c.C, c.D} :=
sorry

end rotation_locus_l147_147194


namespace paint_rate_l147_147536

theorem paint_rate (l b : ℝ) (cost : ℕ) (rate_per_sq_m : ℝ) 
  (h1 : l = 3 * b) 
  (h2 : cost = 300) 
  (h3 : l = 13.416407864998739) 
  (area : ℝ := l * b) : 
  rate_per_sq_m = 5 :=
by
  sorry

end paint_rate_l147_147536


namespace DebateClubOfficerSelection_l147_147266

-- Definitions based on the conditions
def members : Finset ℕ := Finset.range 25 -- Members are indexed from 0 to 24
def Simon := 0
def Rachel := 1
def John := 2

-- Conditions regarding the officers
def is_officer (x : ℕ) (pres sec tre : ℕ) : Prop := 
  x = pres ∨ x = sec ∨ x = tre

def Simon_condition (pres sec tre : ℕ) : Prop :=
  (is_officer Simon pres sec tre) → (is_officer Rachel pres sec tre)

def Rachel_condition (pres sec tre : ℕ) : Prop :=
  (is_officer Rachel pres sec tre) → (is_officer Simon pres sec tre) ∨ (is_officer John pres sec tre)

-- Statement of the problem in Lean
theorem DebateClubOfficerSelection : ∃ (pres sec tre : ℕ), 
  pres ≠ sec ∧ sec ≠ tre ∧ pres ≠ tre ∧ 
  pres ∈ members ∧ sec ∈ members ∧ tre ∈ members ∧ 
  Simon_condition pres sec tre ∧
  Rachel_condition pres sec tre :=
sorry

end DebateClubOfficerSelection_l147_147266


namespace length_PC_l147_147390

noncomputable def length_PA : ℝ := 8
noncomputable def length_PB : ℝ := 5
noncomputable def angle_APB : ℝ := 120
noncomputable def angle_BPC : ℝ := 120
noncomputable def angle_CPA : ℝ := 120

theorem length_PC 
  (a b c : ℝ)  -- lengths of sides of triangle ABC
  (right_triangle_ABC : a^2 + b^2 = c^2) -- ABC is right triangle with right angle at B
  (PA : ℝ := 8) (PB : ℝ := 5) (APB : ℝ := 120) (BPC : ℝ := 120) (CPA : ℝ := 120) 
  (in_triangle_condition : ∃ P, P inside triangle ABC ∧ PA = 8 ∧ PB = 5 
   ∧ angle APB = 120 ∧ angle BPC = 120 ∧ angle CPA = 120) :
  PC = 12.17 := 
sorry

end length_PC_l147_147390


namespace digit_at_position_2021_l147_147684

def sequence_digit (n : ℕ) : ℕ :=
  let seq := (List.range' 1 999).bind (λ i => i.toString.data.toList)
  seq.nth! (n - 1)

theorem digit_at_position_2021 : sequence_digit 2021 = 1 := 
by
  -- We skip the proof details for now
  sorry

end digit_at_position_2021_l147_147684


namespace geometric_mean_l147_147807

theorem geometric_mean (a b c : ℝ) (h1 : a = 5 + 2 * Real.sqrt 6) (h2 : c = 5 - 2 * Real.sqrt 6) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h6 : b^2 = a * c) : b = 1 :=
sorry

end geometric_mean_l147_147807


namespace painting_area_l147_147510

theorem painting_area (wall_height wall_length bookshelf_height bookshelf_length : ℝ)
  (h_wall_height : wall_height = 10)
  (h_wall_length : wall_length = 15)
  (h_bookshelf_height : bookshelf_height = 3)
  (h_bookshelf_length : bookshelf_length = 5) :
  wall_height * wall_length - bookshelf_height * bookshelf_length = 135 := 
by
  sorry

end painting_area_l147_147510


namespace range_of_x_l147_147539

theorem range_of_x (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x / (Real.sqrt (x + 2)) :=
by {
  sorry
}

end range_of_x_l147_147539


namespace probability_A_not_losing_l147_147015

theorem probability_A_not_losing (P_draw P_win : ℚ) (h1 : P_draw = 1/2) (h2 : P_win = 1/3) : 
  P_draw + P_win = 5/6 :=
by
  sorry

end probability_A_not_losing_l147_147015


namespace customer_total_payment_l147_147035

def Riqing_Beef_Noodles_quantity : ℕ := 24
def Riqing_Beef_Noodles_price_per_bag : ℝ := 1.80
def Riqing_Beef_Noodles_discount : ℝ := 0.8

def Kang_Shifu_Ice_Red_Tea_quantity : ℕ := 6
def Kang_Shifu_Ice_Red_Tea_price_per_box : ℝ := 1.70
def Kang_Shifu_Ice_Red_Tea_discount : ℝ := 0.8

def Shanlin_Purple_Cabbage_Soup_quantity : ℕ := 5
def Shanlin_Purple_Cabbage_Soup_price_per_bag : ℝ := 3.40

def Shuanghui_Ham_Sausage_quantity : ℕ := 3
def Shuanghui_Ham_Sausage_price_per_bag : ℝ := 11.20
def Shuanghui_Ham_Sausage_discount : ℝ := 0.9

def total_price : ℝ :=
  (Riqing_Beef_Noodles_quantity * Riqing_Beef_Noodles_price_per_bag * Riqing_Beef_Noodles_discount) +
  (Kang_Shifu_Ice_Red_Tea_quantity * Kang_Shifu_Ice_Red_Tea_price_per_box * Kang_Shifu_Ice_Red_Tea_discount) +
  (Shanlin_Purple_Cabbage_Soup_quantity * Shanlin_Purple_Cabbage_Soup_price_per_bag) +
  (Shuanghui_Ham_Sausage_quantity * Shuanghui_Ham_Sausage_price_per_bag * Shuanghui_Ham_Sausage_discount)

theorem customer_total_payment :
  total_price = 89.96 :=
by
  unfold total_price
  sorry

end customer_total_payment_l147_147035


namespace inverse_function_log_base_two_l147_147082

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_log_base_two (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : f a (a^2) = a) : f a = fun x => Real.log x / Real.log 2 := 
by
  sorry

end inverse_function_log_base_two_l147_147082


namespace variance_of_remaining_scores_l147_147812

def scores : List ℕ := [91, 89, 91, 96, 94, 95, 94]

def remaining_scores : List ℕ := [91, 91, 94, 95, 94]

def mean (l : List ℕ) : ℚ := l.sum / l.length

def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

theorem variance_of_remaining_scores :
  variance remaining_scores = 2.8 := by
  sorry

end variance_of_remaining_scores_l147_147812


namespace eq_sin_intersect_16_solutions_l147_147786

theorem eq_sin_intersect_16_solutions :
  ∃ S : Finset ℝ, (∀ x ∈ S, 0 ≤ x ∧ x ≤ 50 ∧ (x / 50 = Real.sin x)) ∧ (S.card = 16) :=
  sorry

end eq_sin_intersect_16_solutions_l147_147786


namespace remainder_divided_by_82_l147_147091

theorem remainder_divided_by_82 (x : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) ↔ (∃ m : ℤ, x + 13 = 41 * m + 18) :=
by
  sorry

end remainder_divided_by_82_l147_147091


namespace find_ab_l147_147573

theorem find_ab 
(a b : ℝ) 
(h1 : a + b = 2) 
(h2 : a * b = 1 ∨ a * b = -1) :
(a = 1 ∧ b = 1) ∨
(a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨
(a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2) :=
sorry

end find_ab_l147_147573


namespace intersecting_lines_l147_147593

def diamondsuit (a b : ℝ) : ℝ := a^2 + a * b - b^2

theorem intersecting_lines (x y : ℝ) : 
  (diamondsuit x y = diamondsuit y x) ↔ (y = x ∨ y = -x) := by
  sorry

end intersecting_lines_l147_147593


namespace jovana_shells_l147_147249

variable (initial_shells : Nat) (additional_shells : Nat)

theorem jovana_shells (h1 : initial_shells = 5) (h2 : additional_shells = 12) : initial_shells + additional_shells = 17 := 
by 
  sorry

end jovana_shells_l147_147249


namespace number_of_six_digit_palindromes_l147_147377

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 100000 % 10
  let d2 := n / 10000 % 10
  let d3 := n / 1000 % 10
  let d4 := n / 100 % 10
  let d5 := n / 10 % 10
  let d6 := n % 10
  n >= 100000 ∧ n < 1000000 ∧ d1 > 0 ∧ d1 = d6 ∧ d2 = d5 ∧ d3 = d4

theorem number_of_six_digit_palindromes : 
  {n : ℕ | is_six_digit_palindrome n}.card = 900 := 
sorry

end number_of_six_digit_palindromes_l147_147377
