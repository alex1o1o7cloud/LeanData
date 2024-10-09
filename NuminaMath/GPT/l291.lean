import Mathlib

namespace smallest_x_l291_29165

theorem smallest_x (x : ℕ) (h₁ : x % 3 = 2) (h₂ : x % 4 = 3) (h₃ : x % 5 = 4) : x = 59 :=
by
  sorry

end smallest_x_l291_29165


namespace both_selected_prob_l291_29186

def ram_prob : ℚ := 6 / 7
def ravi_prob : ℚ := 1 / 5

theorem both_selected_prob : ram_prob * ravi_prob = 6 / 35 := 
by
  sorry

end both_selected_prob_l291_29186


namespace find_c_d_l291_29141

theorem find_c_d (y c d : ℕ) (H1 : y = c + Real.sqrt d) (H2 : y^2 + 4 * y + 4 / y + 1 / (y^2) = 30) :
  c + d = 5 :=
sorry

end find_c_d_l291_29141


namespace find_fourth_number_l291_29166

variable (x : ℝ)

theorem find_fourth_number
  (h : 3 + 33 + 333 + x = 399.6) :
  x = 30.6 :=
sorry

end find_fourth_number_l291_29166


namespace living_room_curtain_length_l291_29115

theorem living_room_curtain_length :
  let length_bolt := 16
  let width_bolt := 12
  let area_bolt := length_bolt * width_bolt
  let area_left := 160
  let area_cut := area_bolt - area_left
  let length_bedroom := 2
  let width_bedroom := 4
  let area_bedroom := length_bedroom * width_bedroom
  let area_living_room := area_cut - area_bedroom
  let width_living_room := 4
  area_living_room / width_living_room = 6 :=
by
  sorry

end living_room_curtain_length_l291_29115


namespace eval_expression_l291_29147

theorem eval_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end eval_expression_l291_29147


namespace sum_last_two_digits_l291_29114

theorem sum_last_two_digits (n : ℕ) (h1 : n = 20) : (9^n + 11^n) % 100 = 1 :=
by
  sorry

end sum_last_two_digits_l291_29114


namespace greatest_three_digit_multiple_23_l291_29195

theorem greatest_three_digit_multiple_23 : 
  ∃ n : ℕ, n < 1000 ∧ n % 23 = 0 ∧ (∀ m : ℕ, m < 1000 ∧ m % 23 = 0 → m ≤ n) ∧ n = 989 :=
sorry

end greatest_three_digit_multiple_23_l291_29195


namespace problem1_problem2_l291_29185

theorem problem1 : (- (2 : ℤ) ^ 3 / 8 - (1 / 4 : ℚ) * ((-2)^2)) = -2 :=
by {
    sorry
}

theorem problem2 : ((- (1 / 12 : ℚ) - 1 / 16 + 3 / 4 - 1 / 6) * -48) = -21 :=
by {
    sorry
}

end problem1_problem2_l291_29185


namespace minimize_tank_construction_cost_l291_29180

noncomputable def minimum_cost (l w h : ℝ) (P_base P_wall : ℝ) : ℝ :=
  P_base * (l * w) + P_wall * (2 * h * (l + w))

theorem minimize_tank_construction_cost :
  ∃ l w : ℝ, l * w = 9 ∧ l = w ∧
  minimum_cost l w 2 200 150 = 5400 :=
by
  sorry

end minimize_tank_construction_cost_l291_29180


namespace white_tile_count_l291_29163

theorem white_tile_count (total_tiles yellow_tiles blue_tiles purple_tiles white_tiles : ℕ)
  (h_total : total_tiles = 20)
  (h_yellow : yellow_tiles = 3)
  (h_blue : blue_tiles = yellow_tiles + 1)
  (h_purple : purple_tiles = 6)
  (h_sum : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  white_tiles = 7 :=
sorry

end white_tile_count_l291_29163


namespace grain_milling_l291_29145

theorem grain_milling (W : ℝ) (h : 0.9 * W = 100) : W = 111.1 :=
sorry

end grain_milling_l291_29145


namespace frank_remaining_money_l291_29108

theorem frank_remaining_money
  (cheapest_lamp : ℕ)
  (most_expensive_factor : ℕ)
  (frank_money : ℕ)
  (cheapest_lamp_cost : cheapest_lamp = 20)
  (most_expensive_lamp_cost : most_expensive_factor = 3)
  (frank_current_money : frank_money = 90) :
  frank_money - (most_expensive_factor * cheapest_lamp) = 30 :=
by {
  sorry
}

end frank_remaining_money_l291_29108


namespace inequality_holds_for_all_x_l291_29103

theorem inequality_holds_for_all_x (x : ℝ) : 3 * x^2 + 9 * x ≥ -12 := by
  sorry

end inequality_holds_for_all_x_l291_29103


namespace like_terms_monomials_l291_29118

theorem like_terms_monomials (a b : ℕ) : (5 * (m^8) * (n^6) = -(3/4) * (m^(2*a)) * (n^(2*b))) → (a = 4 ∧ b = 3) := by
  sorry

end like_terms_monomials_l291_29118


namespace find_S15_l291_29160

-- Define the arithmetic progression series
variable {S : ℕ → ℕ}

-- Given conditions
axiom S5 : S 5 = 3
axiom S10 : S 10 = 12

-- We need to prove the final statement
theorem find_S15 : S 15 = 39 := 
by
  sorry

end find_S15_l291_29160


namespace largest_divisor_of_n4_minus_n2_l291_29188

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 12 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_n4_minus_n2_l291_29188


namespace expression_value_l291_29126

theorem expression_value :
  3 * ((18 + 7)^2 - (7^2 + 18^2)) = 756 := 
sorry

end expression_value_l291_29126


namespace password_correct_l291_29142

-- conditions
def poly1 (x y : ℤ) : ℤ := x ^ 4 - y ^ 4
def factor1 (x y : ℤ) : ℤ := (x - y) * (x + y) * (x ^ 2 + y ^ 2)

def poly2 (x y : ℤ) : ℤ := x ^ 3 - x * y ^ 2
def factor2 (x y : ℤ) : ℤ := x * (x - y) * (x + y)

-- given values
def x := 18
def y := 5

-- goal
theorem password_correct : factor2 x y = 18 * 13 * 23 :=
by
  -- We setup the goal with the equivalent sequence of the password generation
  sorry

end password_correct_l291_29142


namespace proof_problem_l291_29156

theorem proof_problem 
  (a1 a2 b2 : ℚ)
  (ha1 : a1 = -9 + (8/3))
  (ha2 : a2 = -9 + 2 * (8/3))
  (hb2 : b2 = -3) :
  b2 * (a1 + a2) = 30 :=
by
  sorry

end proof_problem_l291_29156


namespace john_must_solve_at_least_17_correct_l291_29174

theorem john_must_solve_at_least_17_correct :
  ∀ (x : ℕ), 25 = 20 + 5 → 7 * x - (20 - x) + 2 * 5 ≥ 120 → x ≥ 17 :=
by
  intros x h1 h2
  -- Remaining steps will be included in the proof
  sorry

end john_must_solve_at_least_17_correct_l291_29174


namespace fib_ratio_bound_l291_29181

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fib_ratio_bound {a b n : ℕ} (h1: b > 0) (h2: fib (n-1) > 0)
  (h3: (fib n) * b > (fib (n-1)) * a)
  (h4: (fib (n+1)) * b < (fib n) * a) :
  b ≥ fib (n+1) :=
sorry

end fib_ratio_bound_l291_29181


namespace part1_part2_l291_29167

noncomputable def f (x : ℝ) : ℝ := abs (x + 20) - abs (16 - x)

theorem part1 (x : ℝ) : f x ≥ 0 ↔ x ≥ -2 := 
by sorry

theorem part2 (m : ℝ) (x_exists : ∃ x : ℝ, f x ≥ m) : m ≤ 36 := 
by sorry

end part1_part2_l291_29167


namespace problem1_problem2_l291_29158

def f (x : ℝ) := |x - 1| + |x + 2|

def T (a : ℝ) := -Real.sqrt 3 < a ∧ a < Real.sqrt 3

theorem problem1 (a : ℝ) : (∀ x : ℝ, f x > a^2) ↔ T a :=
by
  sorry

theorem problem2 (m n : ℝ) (h1 : T m) (h2 : T n) : Real.sqrt 3 * |m + n| < |m * n + 3| :=
by
  sorry

end problem1_problem2_l291_29158


namespace find_number_l291_29157

theorem find_number : ∃ x, x - 0.16 * x = 126 ↔ x = 150 :=
by 
  sorry

end find_number_l291_29157


namespace perfect_square_trinomial_l291_29138

theorem perfect_square_trinomial (k : ℝ) :
  ∃ k, (∀ x, (4 * x^2 - 2 * k * x + 1) = (2 * x + 1)^2 ∨ (4 * x^2 - 2 * k * x + 1) = (2 * x - 1)^2) → 
  (k = 2 ∨ k = -2) := by
  sorry

end perfect_square_trinomial_l291_29138


namespace width_of_plot_is_correct_l291_29116

-- Definitions based on the given conditions
def cost_per_acre_per_month : ℝ := 60
def total_monthly_rent : ℝ := 600
def length_of_plot : ℝ := 360
def sq_feet_per_acre : ℝ := 43560

-- Theorems to be proved based on the conditions and the correct answer
theorem width_of_plot_is_correct :
  let number_of_acres := total_monthly_rent / cost_per_acre_per_month
  let total_sq_footage := number_of_acres * sq_feet_per_acre
  let width_of_plot := total_sq_footage / length_of_plot
  width_of_plot = 1210 :=
by 
  sorry

end width_of_plot_is_correct_l291_29116


namespace circle_center_coordinates_l291_29191

theorem circle_center_coordinates :
  ∃ c : ℝ × ℝ, (∀ x y : ℝ, x^2 + y^2 - x + 2*y = 0 ↔ (x-c.1)^2 + (y-c.2)^2 = (5/4)) ∧ c = (1/2, -1) :=
sorry

end circle_center_coordinates_l291_29191


namespace when_was_p_turned_off_l291_29136

noncomputable def pipe_p_rate := (1/12 : ℚ)  -- Pipe p rate
noncomputable def pipe_q_rate := (1/15 : ℚ)  -- Pipe q rate
noncomputable def combined_rate := (3/20 : ℚ) -- Combined rate of p and q when both are open
noncomputable def time_after_p_off := (1.5 : ℚ)  -- Time for q to fill alone after p is off
noncomputable def fill_cistern (t : ℚ) := combined_rate * t + pipe_q_rate * time_after_p_off

theorem when_was_p_turned_off (t : ℚ) : fill_cistern t = 1 ↔ t = 6 := sorry

end when_was_p_turned_off_l291_29136


namespace solve_for_a_l291_29148

theorem solve_for_a (x a : ℝ) (h : x = 3) (eq : 5 * x - a = 8) : a = 7 :=
by
  -- sorry to skip the proof as instructed
  sorry

end solve_for_a_l291_29148


namespace price_of_first_variety_l291_29127

theorem price_of_first_variety
  (P : ℝ)
  (H1 : 1 * P + 1 * 135 + 2 * 175.5 = 4 * 153) :
  P = 126 :=
by
  sorry

end price_of_first_variety_l291_29127


namespace triangle_area_l291_29137

namespace MathProof

theorem triangle_area (y_eq_6 y_eq_2_plus_x y_eq_2_minus_x : ℝ → ℝ)
  (h1 : ∀ x, y_eq_6 x = 6)
  (h2 : ∀ x, y_eq_2_plus_x x = 2 + x)
  (h3 : ∀ x, y_eq_2_minus_x x = 2 - x) :
  let a := (4, 6)
  let b := (-4, 6)
  let c := (0, 2)
  let base := dist a b
  let height := (6 - 2:ℝ)
  (1 / 2 * base * height = 16) := by
    sorry

end MathProof

end triangle_area_l291_29137


namespace quadrilateral_interior_angle_not_greater_90_l291_29159

-- Definition of the quadrilateral interior angle property
def quadrilateral_interior_angles := ∀ (a b c d : ℝ), (a + b + c + d = 360) → (a > 90 → b > 90 → c > 90 → d > 90 → false)

-- Proposition: There is at least one interior angle in a quadrilateral that is not greater than 90 degrees.
theorem quadrilateral_interior_angle_not_greater_90 :
  (∀ (a b c d : ℝ), (a + b + c + d = 360) → (a > 90 ∧ b > 90 ∧ c > 90 ∧ d > 90) → false) →
  (∃ (a b c d : ℝ), a + b + c + d = 360 ∧ (a ≤ 90 ∨ b ≤ 90 ∨ c ≤ 90 ∨ d ≤ 90)) :=
sorry

end quadrilateral_interior_angle_not_greater_90_l291_29159


namespace problem_statement_l291_29170

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem problem_statement : f (f (1/2)) = 1 :=
by
    sorry

end problem_statement_l291_29170


namespace find_angle_C_l291_29144

-- Given conditions
variable {A B C : ℝ}
variable (h_triangle : A + B + C = π)
variable (h_tanA : Real.tan A = 1/2)
variable (h_cosB : Real.cos B = 3 * Real.sqrt 10 / 10)

-- The proof statement
theorem find_angle_C :
  C = 3 * π / 4 := by
  sorry

end find_angle_C_l291_29144


namespace chris_packed_percentage_l291_29173

theorem chris_packed_percentage (K C : ℕ) (h : K / (C : ℝ) = 2 / 3) :
  (C / (K + C : ℝ)) * 100 = 60 :=
by
  sorry

end chris_packed_percentage_l291_29173


namespace dvd_cost_packs_l291_29154

theorem dvd_cost_packs (cost_per_pack : ℕ) (number_of_packs : ℕ) (total_money : ℕ) :
  cost_per_pack = 12 → number_of_packs = 11 → total_money = (cost_per_pack * number_of_packs) → total_money = 132 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dvd_cost_packs_l291_29154


namespace fraction_simplification_l291_29152

-- Definitions based on conditions and question
def lcm_462_42 : ℕ := 462
def prime_factors_462 : List ℕ := [2, 3, 7, 11]
def prime_factors_42 : List ℕ := [2, 3, 7]

-- Main theorem statement
theorem fraction_simplification :
  (1 / 462) + (17 / 42) = 94 / 231 :=
by
  sorry

end fraction_simplification_l291_29152


namespace ratio_melina_alma_age_l291_29132

theorem ratio_melina_alma_age
  (A M : ℕ)
  (alma_score : ℕ)
  (h1 : M = 60)
  (h2 : alma_score = 40)
  (h3 : A + M = 2 * alma_score)
  : M / A = 3 :=
by
  sorry

end ratio_melina_alma_age_l291_29132


namespace factor_theorem_solution_l291_29187

theorem factor_theorem_solution (t : ℝ) :
  (x - t ∣ 3 * x^2 + 10 * x - 8) ↔ (t = 2 / 3 ∨ t = -4) :=
by
  sorry

end factor_theorem_solution_l291_29187


namespace sport_formulation_water_l291_29104

theorem sport_formulation_water (corn_syrup_ounces : ℕ) (h_cs : corn_syrup_ounces = 3) : 
  ∃ water_ounces : ℕ, water_ounces = 45 :=
by
  -- The ratios for the "sport" formulation: Flavoring : Corn Syrup : Water = 1 : 4 : 60
  let flavoring_ratio := 1
  let corn_syrup_ratio := 4
  let water_ratio := 60
  -- The given corn syrup is 3 ounces which corresponds to corn_syrup_ratio parts
  have h_ratio : corn_syrup_ratio = 4 := rfl
  have h_flavoring_to_corn_syrup : flavoring_ratio / corn_syrup_ratio = 1 / 4 := by sorry
  have h_flavoring_to_water : flavoring_ratio / water_ratio = 1 / 60 := by sorry
  -- Set up the proportion
  have h_proportion : corn_syrup_ratio / corn_syrup_ounces = water_ratio / 45 := by sorry 
  -- Cross-multiply to solve for the water
  have h_cross_mul : 4 * 45 = 3 * 60 := by sorry
  exact ⟨45, rfl⟩

end sport_formulation_water_l291_29104


namespace ordering_PQR_l291_29139

noncomputable def P := Real.sqrt 2
noncomputable def Q := Real.sqrt 7 - Real.sqrt 3
noncomputable def R := Real.sqrt 6 - Real.sqrt 2

theorem ordering_PQR : P > R ∧ R > Q := by
  sorry

end ordering_PQR_l291_29139


namespace determine_sold_cakes_l291_29143

def initial_cakes := 121
def new_cakes := 170
def remaining_cakes := 186
def sold_cakes (S : ℕ) : Prop := initial_cakes - S + new_cakes = remaining_cakes

theorem determine_sold_cakes : ∃ S, sold_cakes S ∧ S = 105 :=
by
  use 105
  unfold sold_cakes
  simp
  sorry

end determine_sold_cakes_l291_29143


namespace distinct_real_roots_max_abs_gt_2_l291_29162

theorem distinct_real_roots_max_abs_gt_2 
  (r1 r2 r3 q : ℝ)
  (h_distinct : r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h_sum : r1 + r2 + r3 = -q)
  (h_product : r1 * r2 * r3 = -9)
  (h_sum_prod : r1 * r2 + r2 * r3 + r3 * r1 = 6)
  (h_nonzero_discriminant : q^2 * 6^2 - 4 * 6^3 - 4 * q^3 * 9 - 27 * 9^2 + 18 * q * 6 * (-9) ≠ 0) :
  max (|r1|) (max (|r2|) (|r3|)) > 2 :=
sorry

end distinct_real_roots_max_abs_gt_2_l291_29162


namespace billy_questions_third_hour_l291_29133

variable (x : ℝ)
variable (questions_in_first_hour : ℝ := x)
variable (questions_in_second_hour : ℝ := 1.5 * x)
variable (questions_in_third_hour : ℝ := 3 * x)
variable (total_questions_solved : ℝ := 242)

theorem billy_questions_third_hour (h : questions_in_first_hour + questions_in_second_hour + questions_in_third_hour = total_questions_solved) :
  questions_in_third_hour = 132 :=
by
  sorry

end billy_questions_third_hour_l291_29133


namespace boat_speed_in_still_water_l291_29150

variables (V_b V_c V_w : ℝ)

-- Conditions from the problem
def speed_upstream (V_b V_c V_w : ℝ) : ℝ := V_b - V_c - V_w
def water_current_range (V_c : ℝ) : Prop := 2 ≤ V_c ∧ V_c ≤ 4
def wind_resistance_range (V_w : ℝ) : Prop := -1 ≤ V_w ∧ V_w ≤ 1
def upstream_speed : Prop := speed_upstream V_b 4 (2 - (-1)) + (2 - -1) = 4

-- Statement of the proof problem
theorem boat_speed_in_still_water :
  (∀ V_c V_w, water_current_range V_c → wind_resistance_range V_w → speed_upstream V_b V_c V_w = 4) → V_b = 7 :=
by
  sorry

end boat_speed_in_still_water_l291_29150


namespace percentage_mutant_frogs_is_33_l291_29190

def num_extra_legs_frogs := 5
def num_two_heads_frogs := 2
def num_bright_red_frogs := 2
def num_normal_frogs := 18

def total_mutant_frogs := num_extra_legs_frogs + num_two_heads_frogs + num_bright_red_frogs
def total_frogs := total_mutant_frogs + num_normal_frogs

theorem percentage_mutant_frogs_is_33 :
  Float.round (100 * total_mutant_frogs.toFloat / total_frogs.toFloat) = 33 :=
by 
  -- placeholder for the proof
  sorry

end percentage_mutant_frogs_is_33_l291_29190


namespace tan_of_log_conditions_l291_29153

theorem tan_of_log_conditions (x : ℝ) (h1 : 0 < x ∧ x < (Real.pi / 2))
  (h2 : Real.log (Real.sin (2 * x)) - Real.log (Real.sin x) = Real.log (1 / 2)) :
  Real.tan x = Real.sqrt 15 :=
sorry

end tan_of_log_conditions_l291_29153


namespace operation_is_addition_l291_29100

theorem operation_is_addition : (5 + (-5) = 0) :=
by
  sorry

end operation_is_addition_l291_29100


namespace marthas_bedroom_size_l291_29161

-- Define the variables and conditions
def total_square_footage := 300
def additional_square_footage := 60
def Martha := 120
def Jenny := Martha + additional_square_footage

-- The main theorem stating the requirement 
theorem marthas_bedroom_size : (Martha + (Martha + additional_square_footage) = total_square_footage) -> Martha = 120 :=
by 
  sorry

end marthas_bedroom_size_l291_29161


namespace instantaneous_velocity_at_1_l291_29168

noncomputable def particle_displacement (t : ℝ) : ℝ := t + Real.log t

theorem instantaneous_velocity_at_1 : 
  let v := fun t => deriv (particle_displacement) t
  v 1 = 2 :=
by
  sorry

end instantaneous_velocity_at_1_l291_29168


namespace infinitely_many_primes_congruent_3_mod_4_l291_29121

def is_congruent_3_mod_4 (p : ℕ) : Prop :=
  p % 4 = 3

def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

def S (p : ℕ) : Prop :=
  is_prime p ∧ is_congruent_3_mod_4 p

theorem infinitely_many_primes_congruent_3_mod_4 :
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ S p :=
sorry

end infinitely_many_primes_congruent_3_mod_4_l291_29121


namespace Angelina_speeds_l291_29101

def distance_home_to_grocery := 960
def distance_grocery_to_gym := 480
def distance_gym_to_library := 720
def time_diff_grocery_to_gym := 40
def time_diff_gym_to_library := 20

noncomputable def initial_speed (v : ℝ) :=
  (distance_home_to_grocery : ℝ) = (v * (960 / v)) ∧
  (distance_grocery_to_gym : ℝ) = (2 * v * (240 / v)) ∧
  (distance_gym_to_library : ℝ) = (3 * v * (720 / v))

theorem Angelina_speeds (v : ℝ) :
  initial_speed v →
  v = 18 ∧ 2 * v = 36 ∧ 3 * v = 54 :=
by
  sorry

end Angelina_speeds_l291_29101


namespace greatest_possible_sum_of_roots_l291_29102

noncomputable def quadratic_roots (c b : ℝ) : Prop :=
  ∃ α β : ℝ, α ≠ β ∧ α + β = c ∧ α * β = b ∧ |α - β| = 1

theorem greatest_possible_sum_of_roots :
  ∃ (c : ℝ), ( ∃ b : ℝ, quadratic_roots c b) ∧
             ( ∀ (d : ℝ), ( ∃ b : ℝ, quadratic_roots d b) → d ≤ 11 ) ∧ c = 11 :=
sorry

end greatest_possible_sum_of_roots_l291_29102


namespace magnitude_of_T_l291_29192

def i : Complex := Complex.I

def T : Complex := (1 + i) ^ 18 - (1 - i) ^ 18

theorem magnitude_of_T : Complex.abs T = 1024 := by
  sorry

end magnitude_of_T_l291_29192


namespace volume_of_region_l291_29155

theorem volume_of_region :
  ∃ (V : ℝ), V = 9 ∧
  ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 6 :=
sorry

end volume_of_region_l291_29155


namespace min_value_of_xy_ratio_l291_29124

theorem min_value_of_xy_ratio :
  ∃ t : ℝ,
    (t = 2 ∨
    t = ((-1 + Real.sqrt 217) / 12) ∨
    t = ((-1 - Real.sqrt 217) / 12)) ∧
    min (min 2 ((-1 + Real.sqrt 217) / 12)) ((-1 - Real.sqrt 217) / 12) = -1.31 :=
sorry

end min_value_of_xy_ratio_l291_29124


namespace actual_cost_of_article_l291_29122

theorem actual_cost_of_article (x : ℝ) (hx : 0.76 * x = 988) : x = 1300 :=
sorry

end actual_cost_of_article_l291_29122


namespace complement_union_eq_complement_l291_29128

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end complement_union_eq_complement_l291_29128


namespace smallest_multiple_9_11_13_l291_29178

theorem smallest_multiple_9_11_13 : ∃ n : ℕ, n > 0 ∧ (9 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) ∧ n = 1287 := 
 by {
   sorry
 }

end smallest_multiple_9_11_13_l291_29178


namespace reading_minutes_per_disc_l291_29199

-- Define the total reading time
def total_reading_time := 630

-- Define the maximum capacity per disc
def max_capacity_per_disc := 80

-- Define the allowable unused space
def max_unused_space := 4

-- Define the effective capacity of each disc
def effective_capacity_per_disc := max_capacity_per_disc - max_unused_space

-- Define the number of discs needed, rounded up as a ceiling function
def number_of_discs := Nat.ceil (total_reading_time / effective_capacity_per_disc)

-- Theorem statement: Each disc will contain 70 minutes of reading if all conditions are met
theorem reading_minutes_per_disc : ∀ (total_reading_time : ℕ) (max_capacity_per_disc : ℕ) (max_unused_space : ℕ)
  (effective_capacity_per_disc := max_capacity_per_disc - max_unused_space) 
  (number_of_discs := Nat.ceil (total_reading_time / effective_capacity_per_disc)), 
  number_of_discs = 9 → total_reading_time / number_of_discs = 70 :=
by
  sorry

end reading_minutes_per_disc_l291_29199


namespace probability_of_selecting_male_is_three_fifths_l291_29140

-- Define the number of male and female students
def num_male_students : ℕ := 6
def num_female_students : ℕ := 4

-- Define the total number of students
def total_students : ℕ := num_male_students + num_female_students

-- Define the probability of selecting a male student's ID
def probability_male_student : ℚ := num_male_students / total_students

-- Theorem: The probability of selecting a male student's ID is 3/5
theorem probability_of_selecting_male_is_three_fifths : probability_male_student = 3 / 5 :=
by
  -- Proof to be filled in
  sorry

end probability_of_selecting_male_is_three_fifths_l291_29140


namespace range_of_m_l291_29125

-- Definitions of propositions and their negations
def p (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x - 10 ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0
def not_p (x : ℝ) : Prop := x < -2 ∨ x > 10
def not_q (x m : ℝ) : Prop := x < (1 - m) ∨ x > (1 + m) ∧ m > 0

-- Statement that \neg p is a necessary but not sufficient condition for \neg q
def necessary_but_not_sufficient (x m : ℝ) : Prop := 
  (∀ x, not_q x m → not_p x) ∧ ¬(∀ x, not_p x → not_q x m)

-- The main theorem to prove
theorem range_of_m (m : ℝ) : (∀ x, necessary_but_not_sufficient x m) ↔ 9 ≤ m :=
by
  sorry

end range_of_m_l291_29125


namespace average_speed_train_l291_29176

theorem average_speed_train (d1 d2 : ℝ) (t1 t2 : ℝ) 
  (h_d1 : d1 = 325) (h_d2 : d2 = 470)
  (h_t1 : t1 = 3.5) (h_t2 : t2 = 4) :
  (d1 + d2) / (t1 + t2) = 106 :=
by
  sorry

end average_speed_train_l291_29176


namespace least_value_of_x_l291_29189

theorem least_value_of_x 
  (x : ℕ) (p : ℕ) 
  (h1 : x > 0) 
  (h2 : Prime p) 
  (h3 : ∃ q, Prime q ∧ q % 2 = 1 ∧ x = 9 * p * q) : 
  x = 90 := 
sorry

end least_value_of_x_l291_29189


namespace find_ax5_by5_l291_29111

variable (a b x y : ℝ)

theorem find_ax5_by5 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := 
sorry

end find_ax5_by5_l291_29111


namespace locus_eq_l291_29134

noncomputable def locus_of_centers (a b : ℝ) : Prop :=
  5 * a^2 + 9 * b^2 + 80 * a - 400 = 0

theorem locus_eq (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 2)^2) ∧ ((a - 1)^2 + b^2 = (5 - r)^2)) →
  locus_of_centers a b :=
by
  intro h
  sorry

end locus_eq_l291_29134


namespace ratio_circumscribed_circle_area_triangle_area_l291_29119

open Real

theorem ratio_circumscribed_circle_area_triangle_area (h R : ℝ) (h_eq : R = h / 2) :
  let circle_area := π * R^2
  let triangle_area := (h^2) / 4
  (circle_area / triangle_area) = π :=
by
  sorry

end ratio_circumscribed_circle_area_triangle_area_l291_29119


namespace gena_encoded_numbers_unique_l291_29109

theorem gena_encoded_numbers_unique : 
  ∃ (B AN AX NO FF d : ℕ), (AN - B = d) ∧ (AX - AN = d) ∧ (NO - AX = d) ∧ (FF - NO = d) ∧ 
  [B, AN, AX, NO, FF] = [5, 12, 19, 26, 33] := sorry

end gena_encoded_numbers_unique_l291_29109


namespace quadratic_zeros_l291_29196

theorem quadratic_zeros : ∀ x : ℝ, (x = 3 ∨ x = -1) ↔ (x^2 - 2*x - 3 = 0) := by
  intro x
  sorry

end quadratic_zeros_l291_29196


namespace not_support_either_l291_29123

theorem not_support_either (total_attendance supporters_first supporters_second : ℕ) 
  (h1 : total_attendance = 50) 
  (h2 : supporters_first = 50 * 40 / 100) 
  (h3 : supporters_second = 50 * 34 / 100) : 
  total_attendance - (supporters_first + supporters_second) = 13 :=
by
  sorry

end not_support_either_l291_29123


namespace intersection_of_A_and_B_l291_29106

noncomputable def A := {x : ℝ | Real.log x ≤ 0}
noncomputable def B := {x : ℝ | abs (x^2 - 1) ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = A :=
sorry

end intersection_of_A_and_B_l291_29106


namespace repeating_decimal_sum_l291_29175

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l291_29175


namespace speed_man_l291_29107

noncomputable def speedOfMan : ℝ := 
  let d := 437.535 / 1000  -- distance in kilometers
  let t := 25 / 3600      -- time in hours
  d / t                    -- speed in kilometers per hour

theorem speed_man : speedOfMan = 63 := by
  sorry

end speed_man_l291_29107


namespace find_max_number_l291_29149

noncomputable def increasing_sequence (a : ℕ → ℝ) := ∀ n m, n < m → a n < a m

noncomputable def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) (n : ℕ) := 
  (a n + d = a (n+1)) ∧ (a (n+1) + d = a (n+2)) ∧ (a (n+2) + d = a (n+3))

noncomputable def geometric_progression (a : ℕ → ℝ) (r : ℝ) (n : ℕ) := 
  (a (n+1) = a n * r) ∧ (a (n+2) = a (n+1) * r) ∧ (a (n+3) = a (n+2) * r)

theorem find_max_number (a : ℕ → ℝ):
  increasing_sequence a → 
  (∃ n, arithmetic_progression a 4 n) →
  (∃ n, arithmetic_progression a 36 n) →
  (∃ n, geometric_progression a (a (n+1) / a n) n) →
  a 7 = 126 := sorry

end find_max_number_l291_29149


namespace rectangle_area_ratio_l291_29130

theorem rectangle_area_ratio (x d : ℝ) (h_ratio : 5 * x / (2 * x) = 5 / 2) (h_diag : d = 13) :
  ∃ k : ℝ, 10 * x^2 = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_ratio_l291_29130


namespace min_value_expression_l291_29197

theorem min_value_expression : 
  ∃ x : ℝ, ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ (15 - x) * (8 - x) * (15 + x) * (8 + x) ∧ 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6480.25 :=
by sorry

end min_value_expression_l291_29197


namespace value_of_a_l291_29146

open Set

theorem value_of_a (a : ℝ) (h : {1, 2} ∪ {x | x^2 - a * x + a - 1 = 0} = {1, 2}) : a = 3 :=
by
  sorry

end value_of_a_l291_29146


namespace employee_salaries_l291_29183

theorem employee_salaries 
  (x y z : ℝ)
  (h1 : x + y + z = 638)
  (h2 : x = 1.20 * y)
  (h3 : z = 0.80 * y) :
  x = 255.20 ∧ y = 212.67 ∧ z = 170.14 :=
sorry

end employee_salaries_l291_29183


namespace sum_of_series_l291_29172

noncomputable def seriesSum : ℝ := ∑' n : ℕ, (4 * (n + 1) + 1) / (3 ^ (n + 1))

theorem sum_of_series : seriesSum = 7 / 2 := by
  sorry

end sum_of_series_l291_29172


namespace white_tiles_count_l291_29151

-- Definitions from conditions
def total_tiles : ℕ := 20
def yellow_tiles : ℕ := 3
def blue_tiles : ℕ := yellow_tiles + 1
def purple_tiles : ℕ := 6

-- We need to prove that number of white tiles is 7
theorem white_tiles_count : total_tiles - (yellow_tiles + blue_tiles + purple_tiles) = 7 := by
  -- Placeholder for the actual proof
  sorry

end white_tiles_count_l291_29151


namespace miner_distance_when_explosion_heard_l291_29179

-- Distance function for the miner (in feet)
def miner_distance (t : ℕ) : ℕ := 30 * t

-- Distance function for the sound after the explosion (in feet)
def sound_distance (t : ℕ) : ℕ := 1100 * (t - 45)

theorem miner_distance_when_explosion_heard :
  ∃ t : ℕ, miner_distance t / 3 = 463 ∧ miner_distance t = sound_distance t :=
sorry

end miner_distance_when_explosion_heard_l291_29179


namespace number_of_solutions_l291_29129

theorem number_of_solutions :
  ∃ (sols : Finset (ℝ × ℝ × ℝ × ℝ)), 
  (∀ (x y z w : ℝ), ((x, y, z, w) ∈ sols) ↔ (x = z + w + z * w * x ∧ y = w + x + w * x * y ∧ z = x + y + x * y * z ∧ w = y + z + y * z * w ∧ x * y + y * z + z * w + w * x = 2)) ∧ 
  sols.card = 5 :=
sorry

end number_of_solutions_l291_29129


namespace initial_oranges_per_tree_l291_29169

theorem initial_oranges_per_tree (x : ℕ) (h1 : 8 * (5 * x - 2 * x) / 5 = 960) : x = 200 :=
sorry

end initial_oranges_per_tree_l291_29169


namespace required_run_rate_is_correct_l291_29117

-- Define the initial conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_10 : ℝ := 10
def target_runs : ℝ := 282
def remaining_overs : ℝ := 40

-- Given total runs in the first 10 overs
def total_runs_first_10_overs : ℝ := run_rate_first_10_overs * overs_first_10
-- Given runs needed in the remaining 40 overs
def runs_needed_remaining_overs : ℝ := target_runs - total_runs_first_10_overs

-- Lean statement to prove the required run rate in the remaining 40 overs
theorem required_run_rate_is_correct (h1 : run_rate_first_10_overs = 3.2)
                                     (h2 : overs_first_10 = 10)
                                     (h3 : target_runs = 282)
                                     (h4 : remaining_overs = 40) :
  (runs_needed_remaining_overs / remaining_overs) = 6.25 :=
by sorry


end required_run_rate_is_correct_l291_29117


namespace values_of_a_and_b_l291_29120

theorem values_of_a_and_b (a b : ℝ) :
  (∀ x : ℝ, x ≥ -1 → a * x^2 + b * x + a^2 - 1 ≤ 0) →
  a = 0 ∧ b = -1 :=
sorry

end values_of_a_and_b_l291_29120


namespace room_width_l291_29113

theorem room_width (W : ℝ) (L : ℝ := 17) (veranda_width : ℝ := 2) (veranda_area : ℝ := 132) :
  (21 * (W + veranda_width) - L * W = veranda_area) → W = 12 :=
by
  -- setup of the problem
  have total_length := L + 2 * veranda_width
  have total_width := W + 2 * veranda_width
  have area_room_incl_veranda := total_length * total_width - (L * W)
  -- the statement is already provided in the form of the theorem to be proven
  sorry

end room_width_l291_29113


namespace complement_intersection_l291_29182

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A : Set ℕ := {1, 2}

-- Define the set B
def B : Set ℕ := {2, 3, 4}

-- Statement to be proven
theorem complement_intersection :
  (U \ A) ∩ B = {3, 4} :=
sorry

end complement_intersection_l291_29182


namespace L_shaped_region_area_l291_29194

noncomputable def area_L_shaped_region (length full_width : ℕ) (sub_length sub_width : ℕ) : ℕ :=
  let area_full_rect := length * full_width
  let small_width := length - sub_length
  let small_height := full_width - sub_width
  let area_small_rect := small_width * small_height
  area_full_rect - area_small_rect

theorem L_shaped_region_area :
  area_L_shaped_region 10 7 3 4 = 49 :=
by sorry

end L_shaped_region_area_l291_29194


namespace triangle_base_length_l291_29112

theorem triangle_base_length (height : ℝ) (area : ℝ) (base : ℝ) 
  (h_height : height = 6) (h_area : area = 9) 
  (h_formula : area = (1/2) * base * height) : 
  base = 3 :=
by
  sorry

end triangle_base_length_l291_29112


namespace quadratic_pos_in_interval_l291_29184

theorem quadratic_pos_in_interval (m n : ℤ)
  (h2014 : (2014:ℤ)^2 + m * 2014 + n > 0)
  (h2015 : (2015:ℤ)^2 + m * 2015 + n > 0) :
  ∀ x : ℝ, 2014 ≤ x ∧ x ≤ 2015 → (x^2 + (m:ℝ) * x + (n:ℝ)) > 0 :=
by
  sorry

end quadratic_pos_in_interval_l291_29184


namespace angle_in_triangle_l291_29177

theorem angle_in_triangle
  (A B C : Type)
  (a b c : ℝ)
  (angle_ABC : ℝ)
  (h1 : a = 15)
  (h2 : angle_ABC = π/3 ∨ angle_ABC = 2 * π / 3)
  : angle_ABC = π/3 ∨ angle_ABC = 2 * π / 3 := 
  sorry

end angle_in_triangle_l291_29177


namespace khalil_paid_correct_amount_l291_29193

-- Defining the charges for dogs and cats
def cost_per_dog : ℕ := 60
def cost_per_cat : ℕ := 40

-- Defining the number of dogs and cats Khalil took to the clinic
def num_dogs : ℕ := 20
def num_cats : ℕ := 60

-- The total amount Khalil paid
def total_amount_paid : ℕ := 3600

-- The theorem to prove the total amount Khalil paid
theorem khalil_paid_correct_amount :
  (cost_per_dog * num_dogs + cost_per_cat * num_cats) = total_amount_paid :=
by
  sorry

end khalil_paid_correct_amount_l291_29193


namespace student_average_vs_true_average_l291_29135

theorem student_average_vs_true_average (w x y z : ℝ) (h : w < x ∧ x < y ∧ y < z) : 
  (2 * w + 2 * x + y + z) / 6 < (w + x + y + z) / 4 :=
by
  sorry

end student_average_vs_true_average_l291_29135


namespace hyperbola_foci_distance_l291_29110

theorem hyperbola_foci_distance :
  (∀ (x y : ℝ), (y = 2 * x + 3) ∨ (y = -2 * x + 7)) →
  (∃ (x y : ℝ), x = 4 ∧ y = 5 ∧ ((y = 2 * x + 3) ∨ (y = -2 * x + 7))) →
  (∃ h : ℝ, h = 6 * Real.sqrt 2) :=
by
  sorry

end hyperbola_foci_distance_l291_29110


namespace probability_red_ball_10th_draw_l291_29164

-- Definitions for conditions in the problem
def total_balls : ℕ := 10
def red_balls : ℕ := 2

-- Probability calculation function
def probability_of_red_ball (total : ℕ) (red : ℕ) : ℚ :=
  red / total

-- Theorem statement: Given the conditions, the probability of drawing a red ball on the 10th attempt is 1/5
theorem probability_red_ball_10th_draw :
  probability_of_red_ball total_balls red_balls = 1 / 5 :=
by
  sorry

end probability_red_ball_10th_draw_l291_29164


namespace probability_of_rectangle_area_greater_than_32_l291_29171

-- Definitions representing the problem conditions
def segment_length : ℝ := 12
def point_C (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ segment_length
def rectangle_area (x : ℝ) : ℝ := x * (segment_length - x)

-- The probability we need to prove. 
noncomputable def desired_probability : ℝ := 1 / 3

theorem probability_of_rectangle_area_greater_than_32 :
  (∀ x, point_C x → rectangle_area x > 32) → (desired_probability = 1 / 3) :=
by
  sorry

end probability_of_rectangle_area_greater_than_32_l291_29171


namespace restaurant_meal_cost_l291_29131

def cost_of_group_meal (total_people : Nat) (kids : Nat) (adult_meal_cost : Nat) : Nat :=
  let adults := total_people - kids
  adults * adult_meal_cost

theorem restaurant_meal_cost :
  cost_of_group_meal 9 2 2 = 14 := by
  sorry

end restaurant_meal_cost_l291_29131


namespace tetrahedron_distance_sum_l291_29198

theorem tetrahedron_distance_sum (S₁ S₂ S₃ S₄ H₁ H₂ H₃ H₄ V k : ℝ) 
  (h1 : S₁ = k) (h2 : S₂ = 2 * k) (h3 : S₃ = 3 * k) (h4 : S₄ = 4 * k)
  (V_eq : (1 / 3) * S₁ * H₁ + (1 / 3) * S₂ * H₂ + (1 / 3) * S₃ * H₃ + (1 / 3) * S₄ * H₄ = V) :
  1 * H₁ + 2 * H₂ + 3 * H₃ + 4 * H₄ = (3 * V) / k :=
by
  sorry

end tetrahedron_distance_sum_l291_29198


namespace number_of_ways_to_fold_cube_with_one_face_missing_l291_29105

-- Definitions:
-- The polygon is initially in the shape of a cross with 5 congruent squares.
-- One additional square can be attached to any of the 12 possible edge positions around this polygon.
-- Define what it means for the resulting figure to fold into a cube with one face missing.

-- Statement:
theorem number_of_ways_to_fold_cube_with_one_face_missing 
  (initial_squares : ℕ)
  (additional_positions : ℕ)
  (valid_folding_positions : ℕ) : 
  initial_squares = 5 ∧ additional_positions = 12 → valid_folding_positions = 8 :=
by
  sorry

end number_of_ways_to_fold_cube_with_one_face_missing_l291_29105
