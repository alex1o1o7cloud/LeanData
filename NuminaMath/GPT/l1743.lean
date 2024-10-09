import Mathlib

namespace probability_same_flavor_l1743_174349

theorem probability_same_flavor (num_flavors : ℕ) (num_bags : ℕ) (h1 : num_flavors = 4) (h2 : num_bags = 2) :
  let total_outcomes := num_flavors ^ num_bags
  let favorable_outcomes := num_flavors
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 4 :=
by
  sorry

end probability_same_flavor_l1743_174349


namespace dimes_difference_l1743_174358

theorem dimes_difference
  (a b c d : ℕ)
  (h1 : a + b + c + d = 150)
  (h2 : 5 * a + 10 * b + 25 * c + 50 * d = 1500) :
  (b = 150 ∨ ∃ c d : ℕ, b = 0 ∧ 4 * c + 9 * d = 150) →
  ∃ b₁ b₂ : ℕ, (b₁ = 150 ∧ b₂ = 0 ∧ b₁ - b₂ = 150) :=
by
  sorry

end dimes_difference_l1743_174358


namespace simplify_fraction_144_1008_l1743_174347

theorem simplify_fraction_144_1008 :
  (144 : ℤ) / (1008 : ℤ) = (1 : ℤ) / (7 : ℤ) :=
by
  sorry

end simplify_fraction_144_1008_l1743_174347


namespace pills_supply_duration_l1743_174369

open Nat

-- Definitions based on conditions
def one_third_pill_every_three_days : ℕ := 1 / 3 * 3
def pills_in_bottle : ℕ := 90
def days_per_pill : ℕ := 9
def days_per_month : ℕ := 30

-- The Lean statement to prove the question == answer given conditions
theorem pills_supply_duration : (pills_in_bottle * days_per_pill) / days_per_month = 27 := by
  sorry

end pills_supply_duration_l1743_174369


namespace kohen_apples_l1743_174331

theorem kohen_apples (B : ℕ) (h1 : 300 * B = 4 * 750) : B = 10 :=
by
  -- proof goes here
  sorry

end kohen_apples_l1743_174331


namespace gcd_765432_654321_l1743_174319

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l1743_174319


namespace a_sequence_arithmetic_sum_of_bn_l1743_174366

   noncomputable def a (n : ℕ) : ℕ := 1 + n

   def S (n : ℕ) : ℕ := n * (n + 1) / 2

   def b (n : ℕ) : ℚ := 1 / S n

   def T (n : ℕ) : ℚ := (Finset.range n).sum b

   theorem a_sequence_arithmetic (n : ℕ) (a_n_positive : ∀ n, a n > 0)
     (a₁_is_one : a 0 = 1) :
     (a (n+1)) - a n = 1 := by
     sorry

   theorem sum_of_bn (n : ℕ) :
     T n = 2 * n / (n + 1) := by
     sorry
   
end a_sequence_arithmetic_sum_of_bn_l1743_174366


namespace xy_squares_l1743_174336

theorem xy_squares (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 := 
by 
  sorry

end xy_squares_l1743_174336


namespace dennis_initial_money_l1743_174315

theorem dennis_initial_money :
  let cost_of_shirts := 27
  let change_bills := 2 * 10
  let change_coins := 3
  let total_change := change_bills + change_coins
  cost_of_shirts + total_change = 50 :=
by
  let cost_of_shirts := 27
  let change_bills := 2 * 10
  let change_coins := 3
  let total_change := change_bills + change_coins
  show cost_of_shirts + total_change = 50
  sorry

end dennis_initial_money_l1743_174315


namespace no_such_constant_l1743_174321

noncomputable def f : ℚ → ℚ := sorry

theorem no_such_constant (h : ∀ x y : ℚ, ∃ k : ℤ, f (x + y) - f x - f y = k) :
  ¬ ∃ c : ℚ, ∀ x : ℚ, ∃ k : ℤ, f x - c * x = k := 
sorry

end no_such_constant_l1743_174321


namespace select_best_player_l1743_174329

theorem select_best_player : 
  (average_A = 9.6 ∧ variance_A = 0.25) ∧ 
  (average_B = 9.5 ∧ variance_B = 0.27) ∧ 
  (average_C = 9.5 ∧ variance_C = 0.30) ∧ 
  (average_D = 9.6 ∧ variance_D = 0.23) → 
  best_player = D := 
by 
  sorry

end select_best_player_l1743_174329


namespace initial_men_l1743_174335

variable (P M : ℕ) -- P represents the provisions and M represents the initial number of men.

-- Conditons
def provision_lasts_20_days : Prop := P / (M * 20) = P / ((M + 200) * 15)

-- The proof problem
theorem initial_men (h : provision_lasts_20_days P M) : M = 600 :=
sorry

end initial_men_l1743_174335


namespace mean_value_of_pentagon_angles_l1743_174375

theorem mean_value_of_pentagon_angles : 
  let n := 5 
  let interior_angle_sum := (n - 2) * 180 
  mean_angle = interior_angle_sum / n :=
  sorry

end mean_value_of_pentagon_angles_l1743_174375


namespace intersection_of_A_and_B_l1743_174328

variable (A : Set ℝ)
variable (B : Set ℝ)
variable (C : Set ℝ)

theorem intersection_of_A_and_B (hA : A = { x | -1 < x ∧ x < 3 })
                                (hB : B = { -1, 1, 2 })
                                (hC : C = { 1, 2 }) :
  A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l1743_174328


namespace guilt_proof_l1743_174395

variables (E F G : Prop)

theorem guilt_proof
  (h1 : ¬G → F)
  (h2 : ¬E → G)
  (h3 : G → E)
  (h4 : E → ¬F)
  : E ∧ G :=
by
  sorry

end guilt_proof_l1743_174395


namespace time_for_a_and_b_together_l1743_174333

variable (R_a R_b : ℝ)
variable (T_ab : ℝ)

-- Given conditions
def condition_1 : Prop := R_a = 3 * R_b
def condition_2 : Prop := R_a * 28 = 1  -- '1' denotes the entire work

-- Proof goal
theorem time_for_a_and_b_together (h1 : condition_1 R_a R_b) (h2 : condition_2 R_a) : T_ab = 21 := 
by
  sorry

end time_for_a_and_b_together_l1743_174333


namespace problem_statement_l1743_174373

theorem problem_statement (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) : y * (y - 1) ≤ x^2 := 
sorry

end problem_statement_l1743_174373


namespace max_xyz_squared_l1743_174317

theorem max_xyz_squared 
  (x y z : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (h1 : x * y * z = (14 - x) * (14 - y) * (14 - z)) 
  (h2 : x + y + z < 28) : 
  x^2 + y^2 + z^2 ≤ 219 :=
sorry

end max_xyz_squared_l1743_174317


namespace score_of_tenth_game_must_be_at_least_l1743_174324

variable (score_5 average_9 average_10 score_10 : ℤ)
variable (H1 : average_9 > score_5 / 5)
variable (H2 : average_10 > 18)
variable (score_6 score_7 score_8 score_9 : ℤ)
variable (H3 : score_6 = 23)
variable (H4 : score_7 = 14)
variable (H5 : score_8 = 11)
variable (H6 : score_9 = 20)
variable (H7 : average_9 = (score_5 + score_6 + score_7 + score_8 + score_9) / 9)
variable (H8 : average_10 = (score_5 + score_6 + score_7 + score_8 + score_9 + score_10) / 10)

theorem score_of_tenth_game_must_be_at_least :
  score_10 ≥ 29 :=
by
  sorry

end score_of_tenth_game_must_be_at_least_l1743_174324


namespace marks_age_more_than_thrice_aarons_l1743_174379

theorem marks_age_more_than_thrice_aarons :
  ∃ (A : ℕ)(X : ℕ), 28 = A + 17 ∧ 25 = 3 * (A - 3) + X ∧ 32 = 2 * (A + 4) + 2 ∧ X = 1 :=
by
  sorry

end marks_age_more_than_thrice_aarons_l1743_174379


namespace simplify_fraction_l1743_174386

theorem simplify_fraction (a : ℝ) (h : a ≠ 0) : (a + 1) / a - 1 / a = 1 := by
  sorry

end simplify_fraction_l1743_174386


namespace division_pairs_l1743_174387

theorem division_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (ab^2 + b + 7) ∣ (a^2 * b + a + b) →
  (∃ k : ℕ, k ≥ 1 ∧ a = 7 * k^2 ∧ b = 7 * k) ∨ (a, b) = (11, 1) ∨ (a, b) = (49, 1) :=
sorry

end division_pairs_l1743_174387


namespace fraction_decomposition_l1743_174330

theorem fraction_decomposition :
  ∃ (A B : ℚ), 
  (A = 27 / 10) ∧ (B = -11 / 10) ∧ 
  (∀ x : ℚ, 
    7 * x - 13 = A * (3 * x - 4) + B * (x + 2)) := 
  sorry

end fraction_decomposition_l1743_174330


namespace quadratic_equation_in_one_variable_l1743_174301

def is_quadratic_in_one_variable (eq : String) : Prop :=
  match eq with
  | "2x^2 + 5y + 1 = 0" => False
  | "ax^2 + bx - c = 0" => ∃ (a b c : ℝ), a ≠ 0
  | "1/x^2 + x = 2" => False
  | "x^2 = 0" => True
  | _ => False

theorem quadratic_equation_in_one_variable :
  is_quadratic_in_one_variable "x^2 = 0" := by
  sorry

end quadratic_equation_in_one_variable_l1743_174301


namespace combined_volleyball_percentage_l1743_174337

theorem combined_volleyball_percentage (students_north: ℕ) (students_south: ℕ)
(percent_volleyball_north percent_volleyball_south: ℚ)
(H1: students_north = 1800) (H2: percent_volleyball_north = 0.25)
(H3: students_south = 2700) (H4: percent_volleyball_south = 0.35):
  (((students_north * percent_volleyball_north) + (students_south * percent_volleyball_south))
  / (students_north + students_south) * 100) = 31 := 
  sorry

end combined_volleyball_percentage_l1743_174337


namespace value_of_a_l1743_174398

-- Define the variables and conditions as lean definitions/constants
variable (a b c : ℝ)
variable (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
variable (h2 : a * 15 * 11 = 1)

-- Statement to prove
theorem value_of_a : a = 6 :=
by
  sorry

end value_of_a_l1743_174398


namespace algebraic_expression_l1743_174363

-- Definition for the problem expressed in Lean
def number_one_less_than_three_times (a : ℝ) : ℝ :=
  3 * a - 1

-- Theorem stating the proof problem
theorem algebraic_expression (a : ℝ) : number_one_less_than_three_times a = 3 * a - 1 :=
by
  -- Proof steps would go here; omitted as per instructions
  sorry

end algebraic_expression_l1743_174363


namespace find_arc_length_of_sector_l1743_174378

variable (s r p : ℝ)
variable (h_s : s = 4)
variable (h_r : r = 2)
variable (h_area : 2 * s = r * p)

theorem find_arc_length_of_sector 
  (h_s : s = 4) (h_r : r = 2) (h_area : 2 * s = r * p) :
  p = 4 :=
sorry

end find_arc_length_of_sector_l1743_174378


namespace man_l1743_174384

theorem man's_rate_in_still_water 
  (V_s V_m : ℝ)
  (with_stream : V_m + V_s = 24)  -- Condition 1
  (against_stream : V_m - V_s = 10) -- Condition 2
  : V_m = 17 := 
by
  sorry

end man_l1743_174384


namespace exists_three_with_gcd_d_l1743_174390

theorem exists_three_with_gcd_d (n : ℕ) (nums : Fin n.succ → ℕ) (d : ℕ)
  (h1 : n ≥ 2)  -- because n+1 (number of elements nums : Fin n.succ) ≥ 3 given that n ≥ 2
  (h2 : ∀ i, nums i > 0) 
  (h3 : ∀ i, nums i ≤ 100) 
  (h4 : Nat.gcd (nums 0) (Nat.gcd (nums 1) (nums 2)) = d) : 
  ∃ i j k : Fin n.succ, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ Nat.gcd (nums i) (Nat.gcd (nums j) (nums k)) = d :=
by
  sorry

end exists_three_with_gcd_d_l1743_174390


namespace part3_l1743_174364

noncomputable def f (x a : ℝ) : ℝ := x^2 - (2*a + 1)*x + a * Real.log x

theorem part3 (a : ℝ) : 
  (∀ x > 1, f x a > 0) ↔ a ∈ Set.Iic 0 := 
sorry

end part3_l1743_174364


namespace simplify_polynomial_l1743_174368

theorem simplify_polynomial :
  (3 * x ^ 5 - 2 * x ^ 3 + 5 * x ^ 2 - 8 * x + 6) + (7 * x ^ 4 + x ^ 3 - 3 * x ^ 2 + x - 9) =
  3 * x ^ 5 + 7 * x ^ 4 - x ^ 3 + 2 * x ^ 2 - 7 * x - 3 :=
by
  sorry

end simplify_polynomial_l1743_174368


namespace find_missing_number_l1743_174338

theorem find_missing_number (x : ℝ) (h : 1 / ((1 / 0.03) + (1 / x)) = 0.02775) : abs (x - 0.370) < 0.001 := by
  sorry

end find_missing_number_l1743_174338


namespace anna_pays_total_l1743_174320

-- Define the conditions
def daily_rental_cost : ℝ := 35
def cost_per_mile : ℝ := 0.25
def rental_days : ℝ := 3
def miles_driven : ℝ := 300

-- Define the total cost function
def total_cost (daily_rental_cost cost_per_mile rental_days miles_driven : ℝ) : ℝ :=
  (daily_rental_cost * rental_days) + (cost_per_mile * miles_driven)

-- The statement to be proved
theorem anna_pays_total : total_cost daily_rental_cost cost_per_mile rental_days miles_driven = 180 :=
by
  sorry

end anna_pays_total_l1743_174320


namespace ac_plus_bd_eq_neg_10_l1743_174340

theorem ac_plus_bd_eq_neg_10 (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 3)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 6) :
  a * c + b * d = -10 :=
by
  sorry

end ac_plus_bd_eq_neg_10_l1743_174340


namespace cos_double_angle_l1743_174397

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 5) = Real.sqrt 3 / 3) :
  Real.cos (2 * α + 2 * Real.pi / 5) = 1 / 3 :=
by
  sorry

end cos_double_angle_l1743_174397


namespace john_new_weekly_earnings_l1743_174339

theorem john_new_weekly_earnings :
  let original_earnings : ℝ := 40
  let percentage_increase : ℝ := 37.5 / 100
  let raise_amount : ℝ := original_earnings * percentage_increase
  let new_weekly_earnings : ℝ := original_earnings + raise_amount
  new_weekly_earnings = 55 := 
by
  sorry

end john_new_weekly_earnings_l1743_174339


namespace poly_roots_arith_progression_l1743_174344

theorem poly_roots_arith_progression (a b c : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, -- There exist roots x₁, x₂, x₃
    (x₁ + x₃ = 2 * x₂) ∧ -- Roots form an arithmetic progression
    (x₁ * x₂ * x₃ = -c) ∧ -- Roots satisfy polynomial's product condition
    (x₁ + x₂ + x₃ = -a) ∧ -- Roots satisfy polynomial's sum condition
    ((x₁ * x₂) + (x₂ * x₃) + (x₃ * x₁) = b)) -- Roots satisfy polynomial's sum of products condition
  → (2 * a^3 / 27 - a * b / 3 + c = 0) := 
sorry -- proof is not required

end poly_roots_arith_progression_l1743_174344


namespace enjoyable_gameplay_l1743_174371

theorem enjoyable_gameplay (total_hours : ℕ) (boring_percentage : ℕ) (expansion_hours : ℕ)
  (h_total : total_hours = 100)
  (h_boring : boring_percentage = 80)
  (h_expansion : expansion_hours = 30) :
  ((1 - boring_percentage / 100) * total_hours + expansion_hours) = 50 := 
by
  sorry

end enjoyable_gameplay_l1743_174371


namespace find_b_of_roots_condition_l1743_174370

theorem find_b_of_roots_condition
  (α β : ℝ)
  (h1 : α * β = -1)
  (h2 : α + β = -b)
  (h3 : α * β - 2 * α - 2 * β = -11) :
  b = -5 := 
  sorry

end find_b_of_roots_condition_l1743_174370


namespace union_of_sets_l1743_174350

def set_M : Set ℕ := {0, 1, 3}
def set_N : Set ℕ := {x | ∃ (a : ℕ), a ∈ set_M ∧ x = 3 * a}

theorem union_of_sets :
  set_M ∪ set_N = {0, 1, 3, 9} :=
by
  sorry

end union_of_sets_l1743_174350


namespace line_equation_and_inclination_l1743_174399

variable (t : ℝ)
variable (x y : ℝ)
variable (α : ℝ)
variable (l : x = -3 + t ∧ y = 1 + sqrt 3 * t)

theorem line_equation_and_inclination 
  (H : l) : 
  (∃ a b c : ℝ, a = sqrt 3 ∧ b = -1 ∧ c = 3 * sqrt 3 + 1 ∧ a * x + b * y + c = 0) ∧
  α = Real.pi / 3 :=
by
  sorry

end line_equation_and_inclination_l1743_174399


namespace op_neg2_3_l1743_174383

def op (a b : ℤ) : ℤ := a^2 + 2 * a * b

theorem op_neg2_3 : op (-2) 3 = -8 :=
by
  -- proof
  sorry

end op_neg2_3_l1743_174383


namespace gold_bars_total_worth_l1743_174303

theorem gold_bars_total_worth :
  let rows := 4
  let bars_per_row := 20
  let worth_per_bar : ℕ := 20000
  let total_bars := rows * bars_per_row
  let total_worth := total_bars * worth_per_bar
  total_worth = 1600000 :=
by
  sorry

end gold_bars_total_worth_l1743_174303


namespace actor_A_constraints_l1743_174354

-- Definitions corresponding to the conditions.
def numberOfActors : Nat := 6
def positionConstraints : Nat := 4
def permutations (n : Nat) : Nat := Nat.factorial n

-- Lean statement for the proof problem.
theorem actor_A_constraints : 
  (positionConstraints * permutations (numberOfActors - 1)) = 480 := by
sorry

end actor_A_constraints_l1743_174354


namespace Iain_pennies_left_l1743_174392

theorem Iain_pennies_left (initial_pennies older_pennies : ℕ) (percentage : ℝ)
  (h_initial : initial_pennies = 200)
  (h_older : older_pennies = 30)
  (h_percentage : percentage = 0.20) :
  initial_pennies - older_pennies - (percentage * (initial_pennies - older_pennies)) = 136 :=
by
  sorry

end Iain_pennies_left_l1743_174392


namespace parallelogram_area_l1743_174381

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) (hb : b = 15) (hh : h = 2 * b) (hA : A = b * h) : A = 450 := 
by
  rw [hb, hh] at hA
  rw [hA]
  sorry

end parallelogram_area_l1743_174381


namespace total_work_completed_in_days_l1743_174311

-- Define the number of days Amit can complete the work
def amit_days : ℕ := 15

-- Define the number of days Ananthu can complete the work
def ananthu_days : ℕ := 90

-- Define the number of days Amit worked
def amit_work_days : ℕ := 3

-- Calculate the amount of work Amit can do in one day
def amit_work_day_rate : ℚ := 1 / amit_days

-- Calculate the amount of work Ananthu can do in one day
def ananthu_work_day_rate : ℚ := 1 / ananthu_days

-- Calculate the total work completed
theorem total_work_completed_in_days :
  amit_work_days * amit_work_day_rate + (1 - amit_work_days * amit_work_day_rate) / ananthu_work_day_rate = 75 :=
by
  -- Placeholder for the proof
  sorry

end total_work_completed_in_days_l1743_174311


namespace three_digit_number_l1743_174307

theorem three_digit_number (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 1 ≤ b) (h4 : b ≤ 9) (h5 : 0 ≤ c) (h6 : c ≤ 9) 
  (h : 100 * a + 10 * b + c = 3 * (10 * (a + b) + c)) : 100 * a + 10 * b + c = 135 :=
  sorry

end three_digit_number_l1743_174307


namespace kangaroo_can_jump_1000_units_l1743_174389

noncomputable def distance (x y : ℕ) : ℕ := x + y

def valid_small_jump (x y : ℕ) : Prop :=
  x + 1 ≥ 0 ∧ y - 1 ≥ 0

def valid_big_jump (x y : ℕ) : Prop :=
  x - 5 ≥ 0 ∧ y + 7 ≥ 0

theorem kangaroo_can_jump_1000_units (x y : ℕ) (h : x + y > 6) :
  distance x y ≥ 1000 :=
sorry

end kangaroo_can_jump_1000_units_l1743_174389


namespace gcd_linear_combination_l1743_174388

theorem gcd_linear_combination (a b : ℤ) : Int.gcd (5 * a + 3 * b) (13 * a + 8 * b) = Int.gcd a b := by
  sorry

end gcd_linear_combination_l1743_174388


namespace quadricycles_count_l1743_174332

theorem quadricycles_count (s q : ℕ) (hsq : s + q = 9) (hw : 2 * s + 4 * q = 30) : q = 6 :=
by
  sorry

end quadricycles_count_l1743_174332


namespace no_sum_of_cubes_eq_2002_l1743_174357

theorem no_sum_of_cubes_eq_2002 :
  ¬ ∃ (a b c : ℕ), (a ^ 3 + b ^ 3 + c ^ 3 = 2002) :=
sorry

end no_sum_of_cubes_eq_2002_l1743_174357


namespace cindy_dress_discount_l1743_174343

theorem cindy_dress_discount (P D : ℝ) 
  (h1 : P * (1 - D) * 1.25 = 61.2) 
  (h2 : P - 61.2 = 4.5) : D = 0.255 :=
sorry

end cindy_dress_discount_l1743_174343


namespace percentage_of_good_fruits_l1743_174316

theorem percentage_of_good_fruits (total_oranges : ℕ) (total_bananas : ℕ) 
    (rotten_oranges_percent : ℝ) (rotten_bananas_percent : ℝ) :
    total_oranges = 600 ∧ total_bananas = 400 ∧ 
    rotten_oranges_percent = 0.15 ∧ rotten_bananas_percent = 0.03 →
    (510 + 388) / (600 + 400) * 100 = 89.8 :=
by
  intros
  sorry

end percentage_of_good_fruits_l1743_174316


namespace find_x_minus_y_l1743_174345

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  -- Proof omitted
  sorry

end find_x_minus_y_l1743_174345


namespace log_three_div_square_l1743_174380

theorem log_three_div_square (x y : ℝ) (h₁ : x ≠ 1) (h₂ : y ≠ 1) (h₃ : Real.log x / Real.log 3 = Real.log 81 / Real.log y) (h₄ : x * y = 243) :
  (Real.log (x / y) / Real.log 3) ^ 2 = 9 := 
sorry

end log_three_div_square_l1743_174380


namespace verify_conditions_l1743_174365

-- Define the conditions as expressions
def condition_A (a : ℝ) : Prop := 2 * a * 3 * a = 6 * a
def condition_B (a b : ℝ) : Prop := 3 * a^2 * b - 3 * a * b^2 = 0
def condition_C (a : ℝ) : Prop := 6 * a / (2 * a) = 3
def condition_D (a : ℝ) : Prop := (-2 * a) ^ 3 = -6 * a^3

-- Prove which condition is correct
theorem verify_conditions (a b : ℝ) (h : a ≠ 0) : 
  ¬ condition_A a ∧ ¬ condition_B a b ∧ condition_C a ∧ ¬ condition_D a :=
by 
  sorry

end verify_conditions_l1743_174365


namespace lcm_5_6_8_18_l1743_174362

/-- The least common multiple of the numbers 5, 6, 8, and 18 is 360. -/
theorem lcm_5_6_8_18 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 18) = 360 := by
  sorry

end lcm_5_6_8_18_l1743_174362


namespace green_peaches_per_basket_l1743_174351

-- Definitions based on given conditions
def total_peaches : ℕ := 10
def red_peaches_per_basket : ℕ := 4

-- Theorem statement based on the question and correct answer
theorem green_peaches_per_basket :
  (total_peaches - red_peaches_per_basket) = 6 := 
by
  sorry

end green_peaches_per_basket_l1743_174351


namespace coprime_integer_pairs_sum_285_l1743_174334

theorem coprime_integer_pairs_sum_285 : 
  (∃ s : Finset (ℕ × ℕ), 
    ∀ p ∈ s, p.1 + p.2 = 285 ∧ Nat.gcd p.1 p.2 = 1 ∧ s.card = 72) := sorry

end coprime_integer_pairs_sum_285_l1743_174334


namespace percent_democrats_is_60_l1743_174313
-- Import the necessary library

-- Define the problem conditions
variables (D R : ℝ)
variables (h1 : D + R = 100)
variables (h2 : 0.70 * D + 0.20 * R = 50)

-- State the theorem to be proved
theorem percent_democrats_is_60 (D R : ℝ) (h1 : D + R = 100) (h2 : 0.70 * D + 0.20 * R = 50) : D = 60 :=
by
  sorry

end percent_democrats_is_60_l1743_174313


namespace tomatoes_ruined_and_discarded_l1743_174322

theorem tomatoes_ruined_and_discarded 
  (W : ℝ)
  (C : ℝ)
  (P : ℝ)
  (S : ℝ)
  (profit_percentage : ℝ)
  (initial_cost : C = 0.80 * W)
  (remaining_tomatoes : S = 0.9956)
  (desired_profit : profit_percentage = 0.12)
  (final_cost : 0.896 = 0.80 + 0.096) :
  0.9956 * (1 - P / 100) = 0.896 :=
by
  sorry

end tomatoes_ruined_and_discarded_l1743_174322


namespace distance_between_trains_l1743_174326

def speed_train1 : ℝ := 11 -- Speed of the first train in mph
def speed_train2 : ℝ := 31 -- Speed of the second train in mph
def time_travelled : ℝ := 8 -- Time in hours

theorem distance_between_trains : 
  (speed_train2 * time_travelled) - (speed_train1 * time_travelled) = 160 := by
  sorry

end distance_between_trains_l1743_174326


namespace fixed_point_of_line_l1743_174376

theorem fixed_point_of_line (m : ℝ) : 
  ∀ (x y : ℝ), (3 * x - 2 * y + 7 = 0) ∧ (4 * x + 5 * y - 6 = 0) → x = -1 ∧ y = 2 :=
sorry

end fixed_point_of_line_l1743_174376


namespace square_perimeter_l1743_174352

theorem square_perimeter (a : ℝ) (side : ℝ) (perimeter : ℝ) (h1 : a = 144) (h2 : side = Real.sqrt a) (h3 : perimeter = 4 * side) : perimeter = 48 := by
  sorry

end square_perimeter_l1743_174352


namespace max_gcd_is_2_l1743_174325

-- Define the sequence
def a (n : ℕ) : ℕ := 101 + (n + 1)^2 + 3 * n

-- Define the gcd of consecutive terms
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_is_2 : ∀ n : ℕ, n > 0 → d n = 2 :=
by
  intros n hn
  dsimp [d]
  sorry

end max_gcd_is_2_l1743_174325


namespace scientific_notation_of_190_million_l1743_174341

theorem scientific_notation_of_190_million : (190000000 : ℝ) = 1.9 * 10^8 :=
sorry

end scientific_notation_of_190_million_l1743_174341


namespace man_l1743_174318

theorem man's_rowing_speed_in_still_water
  (river_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (H_river_speed : river_speed = 2)
  (H_total_time : total_time = 1)
  (H_total_distance : total_distance = 5.333333333333333) :
  ∃ (v : ℝ), 
    v = 7.333333333333333 ∧
    ∀ d,
    d = total_distance / 2 →
    d = (v - river_speed) * (total_time / 2) ∧
    d = (v + river_speed) * (total_time / 2) := 
by
  sorry

end man_l1743_174318


namespace triangle_inequality_min_diff_l1743_174306

theorem triangle_inequality_min_diff
  (DE EF FD : ℕ) 
  (h1 : DE + EF + FD = 398)
  (h2 : DE < EF ∧ EF ≤ FD) : 
  EF - DE = 1 :=
by
  sorry

end triangle_inequality_min_diff_l1743_174306


namespace prob_zhang_nings_wins_2_1_correct_prob_ξ_minus_2_correct_prob_ξ_minus_1_correct_prob_ξ_1_correct_prob_ξ_2_correct_expected_value_ξ_correct_l1743_174377

noncomputable def prob_zhang_nings_wins_2_1 :=
  2 * 0.4 * 0.6 * 0.6 = 0.288

theorem prob_zhang_nings_wins_2_1_correct : prob_zhang_nings_wins_2_1 := sorry

def prob_ξ_minus_2 := 0.4 * 0.4 = 0.16
def prob_ξ_minus_1 := 2 * 0.4 * 0.6 * 0.4 = 0.192
def prob_ξ_1 := 2 * 0.4 * 0.6 * 0.6 = 0.288
def prob_ξ_2 := 0.6 * 0.6 = 0.36

theorem prob_ξ_minus_2_correct : prob_ξ_minus_2 := sorry
theorem prob_ξ_minus_1_correct : prob_ξ_minus_1 := sorry
theorem prob_ξ_1_correct : prob_ξ_1 := sorry
theorem prob_ξ_2_correct : prob_ξ_2 := sorry

noncomputable def expected_value_ξ :=
  (-2 * 0.16) + (-1 * 0.192) + (1 * 0.288) + (2 * 0.36) = 0.496

theorem expected_value_ξ_correct : expected_value_ξ := sorry

end prob_zhang_nings_wins_2_1_correct_prob_ξ_minus_2_correct_prob_ξ_minus_1_correct_prob_ξ_1_correct_prob_ξ_2_correct_expected_value_ξ_correct_l1743_174377


namespace Ki_tae_pencils_l1743_174361

theorem Ki_tae_pencils (P B : ℤ) (h1 : P + B = 12) (h2 : 1000 * P + 1300 * B = 15000) : P = 2 :=
sorry

end Ki_tae_pencils_l1743_174361


namespace liam_markers_liam_first_markers_over_500_l1743_174308

def seq (n : ℕ) : ℕ := 5 * 3^n

theorem liam_markers (n : ℕ) (h1 : seq 0 = 5) (h2 : seq 1 = 10) (h3 : ∀ k < n, 5 * 3^k ≤ 500) : 
  seq n > 500 := by sorry

theorem liam_first_markers_over_500 (h1 : seq 0 = 5) (h2 : seq 1 = 10) :
  ∃ n, seq n > 500 ∧ ∀ k < n, seq k ≤ 500 := by sorry

end liam_markers_liam_first_markers_over_500_l1743_174308


namespace number_of_integers_with_abs_val_conditions_l1743_174323

theorem number_of_integers_with_abs_val_conditions : 
  (∃ n : ℕ, n = 8) :=
by sorry

end number_of_integers_with_abs_val_conditions_l1743_174323


namespace odd_function_property_l1743_174356

theorem odd_function_property {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = - f x) (h2 : ∀ x, f (1 + x) = f (-x)) (h3 : f (-1 / 3) = 1 / 3) : f (5 / 3) = 1 / 3 := 
sorry

end odd_function_property_l1743_174356


namespace skill_of_passing_through_walls_l1743_174359

theorem skill_of_passing_through_walls (k n : ℕ) (h : k = 8) (h_eq : k * Real.sqrt (k / (k * k - 1)) = Real.sqrt (k * k / (k * k - 1))) : n = k * k - 1 :=
by sorry

end skill_of_passing_through_walls_l1743_174359


namespace average_increase_l1743_174374

-- Define the conditions as Lean definitions
def runs_in_17th_inning : ℕ := 50
def average_after_17th_inning : ℕ := 18

-- The condition about the average increase can be written as follows
theorem average_increase 
  (initial_average: ℕ) -- The batsman's average after the 16th inning
  (h1: runs_in_17th_inning = 50)
  (h2: average_after_17th_inning = 18)
  (h3: 16 * initial_average + runs_in_17th_inning = 17 * average_after_17th_inning) :
  average_after_17th_inning - initial_average = 2 := 
sorry

end average_increase_l1743_174374


namespace running_laps_l1743_174302

theorem running_laps (A B : ℕ)
  (h_ratio : ∀ t : ℕ, (A * t) = 5 * (B * t) / 3)
  (h_start : A = 5 ∧ B = 3 ∧ ∀ t : ℕ, (A * t) - (B * t) = 4) :
  (B * 2 = 6) ∧ (A * 2 = 10) :=
by
  sorry

end running_laps_l1743_174302


namespace distinct_integers_sum_of_three_elems_l1743_174305

-- Define the set S and the property of its elements
def S : Set ℕ := {1, 4, 7, 10, 13, 16, 19}

-- Define the property that each element in S is of the form 3k + 1
def is_form_3k_plus_1 (x : ℕ) : Prop := ∃ k : ℤ, x = 3 * k + 1

theorem distinct_integers_sum_of_three_elems (h₁ : ∀ x ∈ S, is_form_3k_plus_1 x) :
  (∃! n, n = 13) :=
by
  sorry

end distinct_integers_sum_of_three_elems_l1743_174305


namespace x_pow_4_plus_inv_x_pow_4_l1743_174327

theorem x_pow_4_plus_inv_x_pow_4 (x : ℝ) (h : x^2 - 15 * x + 1 = 0) : x^4 + (1 / x^4) = 49727 :=
by
  sorry

end x_pow_4_plus_inv_x_pow_4_l1743_174327


namespace housewife_more_kgs_l1743_174300

theorem housewife_more_kgs (P R money more_kgs : ℝ)
  (hR: R = 40)
  (hReduction: R = P - 0.25 * P)
  (hMoney: money = 800)
  (hMoreKgs: more_kgs = (money / R) - (money / P)) :
  more_kgs = 5 :=
  by
    sorry

end housewife_more_kgs_l1743_174300


namespace tom_travel_time_to_virgo_island_l1743_174394

-- Definitions based on conditions
def boat_trip_time : ℕ := 2
def plane_trip_time : ℕ := 4 * boat_trip_time
def total_trip_time : ℕ := plane_trip_time + boat_trip_time

-- Theorem we need to prove
theorem tom_travel_time_to_virgo_island : total_trip_time = 10 := by
  sorry

end tom_travel_time_to_virgo_island_l1743_174394


namespace smallest_top_block_number_l1743_174396

-- Define the pyramid structure and number assignment problem
def block_pyramid : Type := sorry

-- Given conditions:
-- 4 layers, specific numberings, and block support structure.
structure Pyramid :=
  (Layer1 : Fin 16 → ℕ)
  (Layer2 : Fin 9 → ℕ)
  (Layer3 : Fin 4 → ℕ)
  (TopBlock : ℕ)

-- Constraints on block numbers
def is_valid (P : Pyramid) : Prop :=
  -- base layer numbers are from 1 to 16
  (∀ i, 1 ≤ P.Layer1 i ∧ P.Layer1 i ≤ 16) ∧
  -- each above block is the sum of directly underlying neighboring blocks
  (∀ i, P.Layer2 i = P.Layer1 (i * 3) + P.Layer1 (i * 3 + 1) + P.Layer1 (i * 3 + 2)) ∧
  (∀ i, P.Layer3 i = P.Layer2 (i * 3) + P.Layer2 (i * 3 + 1) + P.Layer2 (i * 3 + 2)) ∧
  P.TopBlock = P.Layer3 0 + P.Layer3 1 + P.Layer3 2 + P.Layer3 3

-- Statement of the theorem
theorem smallest_top_block_number : ∃ P : Pyramid, is_valid P ∧ P.TopBlock = ComputedValue := sorry

end smallest_top_block_number_l1743_174396


namespace cost_of_softball_l1743_174360

theorem cost_of_softball 
  (original_budget : ℕ)
  (dodgeball_cost : ℕ)
  (num_dodgeballs : ℕ)
  (increase_rate : ℚ)
  (num_softballs : ℕ)
  (new_budget : ℕ)
  (softball_cost : ℕ)
  (h0 : original_budget = num_dodgeballs * dodgeball_cost)
  (h1 : increase_rate = 0.20)
  (h2 : new_budget = original_budget + increase_rate * original_budget)
  (h3 : new_budget = num_softballs * softball_cost) :
  softball_cost = 9 :=
by
  sorry

end cost_of_softball_l1743_174360


namespace distance_between_centers_l1743_174314

-- Define the points P, Q, R in the plane
variable (P Q R : ℝ × ℝ)

-- Define the lengths PQ, PR, and QR
variable (PQ PR QR : ℝ)
variable (is_right_triangle : ∃ (a b c : ℝ), PQ = a ∧ PR = b ∧ QR = c ∧ a^2 + b^2 = c^2)

-- Define the inradii r1, r2, r3 for triangles PQR, RST, and QUV respectively
variable (r1 r2 r3 : ℝ)

-- Assume PQ = 90, PR = 120, and QR = 150
axiom PQ_length : PQ = 90
axiom PR_length : PR = 120
axiom QR_length : QR = 150

-- Define the centers O2 and O3 of the circles C2 and C3 respectively
variable (O2 O3 : ℝ × ℝ)

-- Assume the inradius length is 30 for the initial triangle
axiom inradius_PQR : r1 = 30

-- Assume the positions of the centers of C2 and C3
axiom O2_position : O2 = (15, 75)
axiom O3_position : O3 = (70, 10)

-- Use the distance formula to express the final result
theorem distance_between_centers : ∃ n : ℕ, dist O2 O3 = Real.sqrt (10 * n) ∧ n = 725 :=
by
  sorry

end distance_between_centers_l1743_174314


namespace sale_in_fifth_month_l1743_174304

theorem sale_in_fifth_month (sale1 sale2 sale3 sale4 sale6 avg_sale num_months total_sales known_sales_five_months sale5: ℕ) :
  sale1 = 6400 →
  sale2 = 7000 →
  sale3 = 6800 →
  sale4 = 7200 →
  sale6 = 5100 →
  avg_sale = 6500 →
  num_months = 6 →
  total_sales = avg_sale * num_months →
  known_sales_five_months = sale1 + sale2 + sale3 + sale4 + sale6 →
  sale5 = total_sales - known_sales_five_months →
  sale5 = 6500 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end sale_in_fifth_month_l1743_174304


namespace minimum_value_of_y_exists_l1743_174346

theorem minimum_value_of_y_exists :
  ∃ (y : ℝ), (∀ (x : ℝ), (y + x) = (y - x)^2 + 3 * (y - x) + 3) ∧ y = -1/2 :=
by sorry

end minimum_value_of_y_exists_l1743_174346


namespace factor_expr_l1743_174353

def expr1 (x : ℝ) := 16 * x^6 + 49 * x^4 - 9
def expr2 (x : ℝ) := 4 * x^6 - 14 * x^4 - 9

theorem factor_expr (x : ℝ) :
  (expr1 x - expr2 x) = 3 * x^4 * (4 * x^2 + 21) := 
by
  sorry

end factor_expr_l1743_174353


namespace remainder_of_sum_div_10_l1743_174391

theorem remainder_of_sum_div_10 : (5000 + 5001 + 5002 + 5003 + 5004) % 10 = 0 :=
by
  sorry

end remainder_of_sum_div_10_l1743_174391


namespace intersection_M_N_l1743_174309

-- Define the set M
def M : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the condition for set N
def N : Set ℤ := {x | x + 2 ≥ x^2}

-- State the theorem to prove the intersection
theorem intersection_M_N :
  M ∩ N = {-1, 0, 1, 2} :=
sorry

end intersection_M_N_l1743_174309


namespace permutation_value_l1743_174367

theorem permutation_value : ∀ (n r : ℕ), n = 5 → r = 3 → (n.choose r) * r.factorial = 60 := 
by
  intros n r hn hr 
  rw [hn, hr]
  -- We use the permutation formula A_{n}^{r} = n! / (n-r)!
  -- A_{5}^{3} = 5! / 2!
  -- Simplifies to 5 * 4 * 3 = 60.
  sorry

end permutation_value_l1743_174367


namespace negation_proposition_l1743_174355

theorem negation_proposition : 
  ¬(∀ x : ℝ, 0 ≤ x → 2^x > x^2) ↔ ∃ x : ℝ, 0 ≤ x ∧ 2^x ≤ x^2 := by
  sorry

end negation_proposition_l1743_174355


namespace maximum_PM_minus_PN_l1743_174312

noncomputable def x_squared_over_9_minus_y_squared_over_16_eq_1 (x y : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / 16) = 1

noncomputable def circle1 (x y : ℝ) : Prop :=
  (x + 5)^2 + y^2 = 4

noncomputable def circle2 (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 1

theorem maximum_PM_minus_PN :
  ∀ (P M N : ℝ × ℝ),
    x_squared_over_9_minus_y_squared_over_16_eq_1 P.1 P.2 →
    circle1 M.1 M.2 →
    circle2 N.1 N.2 →
    (|dist P M - dist P N| ≤ 9) := sorry

end maximum_PM_minus_PN_l1743_174312


namespace task_completion_time_l1743_174348

theorem task_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ t : ℝ, t = (a * b) / (a + b) := 
sorry

end task_completion_time_l1743_174348


namespace root_conditions_imply_sum_l1743_174382

-- Define the variables a and b in the context that their values fit the given conditions.
def a : ℝ := 5
def b : ℝ := -6

-- Define the quadratic equation and conditions on roots.
def quadratic_eq (x : ℝ) := x^2 - a * x - b

-- Given that 2 and 3 are the roots of the quadratic equation.
def roots_condition := (quadratic_eq 2 = 0) ∧ (quadratic_eq 3 = 0)

-- The theorem to prove.
theorem root_conditions_imply_sum :
  roots_condition → a + b = -1 :=
by
sorry

end root_conditions_imply_sum_l1743_174382


namespace compute_expression_l1743_174393

theorem compute_expression :
  20 * ((144 / 3) + (36 / 6) + (16 / 32) + 2) = 1130 := sorry

end compute_expression_l1743_174393


namespace unit_squares_in_50th_ring_l1743_174372

-- Definitions from the conditions
def unit_squares_in_first_ring : ℕ := 12

def unit_squares_in_nth_ring (n : ℕ) : ℕ :=
  32 * n - 16

-- Prove the specific instance for the 50th ring
theorem unit_squares_in_50th_ring : unit_squares_in_nth_ring 50 = 1584 :=
by
  sorry

end unit_squares_in_50th_ring_l1743_174372


namespace crayons_left_l1743_174310

def initial_crayons : ℕ := 253
def lost_or_given_away_crayons : ℕ := 70
def remaining_crayons : ℕ := 183

theorem crayons_left (initial_crayons : ℕ) (lost_or_given_away_crayons : ℕ) (remaining_crayons : ℕ) :
  initial_crayons - lost_or_given_away_crayons = remaining_crayons :=
by {
  sorry
}

end crayons_left_l1743_174310


namespace eighth_day_of_april_2000_is_saturday_l1743_174342

noncomputable def april_2000_eight_day_is_saturday : Prop :=
  (∃ n : ℕ, (1 ≤ n ∧ n ≤ 7) ∧
            ((n + 0 * 7) = 2 ∨ (n + 1 * 7) = 2 ∨ (n + 2 * 7) = 2 ∨
             (n + 3 * 7) = 2 ∨ (n + 4 * 7) = 2) ∧
            ((n + 0 * 7) % 2 = 0 ∨ (n + 1 * 7) % 2 = 0 ∨
             (n + 2 * 7) % 2 = 0 ∨ (n + 3 * 7) % 2 = 0 ∨
             (n + 4 * 7) % 2 = 0) ∧
            (∃ k : ℕ, k ≤ 4 ∧ (n + k * 7 = 8))) ∧
            (8 % 7) = 1 ∧ (1 ≠ 0)

theorem eighth_day_of_april_2000_is_saturday :
  april_2000_eight_day_is_saturday := 
sorry

end eighth_day_of_april_2000_is_saturday_l1743_174342


namespace ellipse_equation_minimum_distance_l1743_174385

-- Define the conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2) / (a^2) + (y^2) / (b^2) = 1)

def eccentricity (a c : ℝ) : Prop :=
  c = a / 2

def focal_distance (c : ℝ) : Prop :=
  2 * c = 4

def foci_parallel (F1 A B C D : ℝ × ℝ) : Prop :=
  let ⟨x1, y1⟩ := F1;
  let ⟨xA, yA⟩ := A;
  let ⟨xC, yC⟩ := C;
  let ⟨xB, yB⟩ := B;
  let ⟨xD, yD⟩ := D;
  (yA - y1) / (xA - x1) = (yC - y1) / (xC - x1) ∧ 
  (yB - y1) / (xB - x1) = (yD - y1) / (xD - x1)

def orthogonal_vectors (A C B D : ℝ × ℝ) : Prop :=
  let ⟨xA, yA⟩ := A;
  let ⟨xC, yC⟩ := C;
  let ⟨xB, yB⟩ := B;
  let ⟨xD, yD⟩ := D;
  (xC - xA) * (xD - xB) + (yC - yA) * (yD - yB) = 0

-- Prove equation of ellipse E
theorem ellipse_equation (a b : ℝ) (x y : ℝ) (c : ℝ)
  (h1 : ellipse a b x y)
  (h2 : eccentricity a c)
  (h3 : focal_distance c) :
  (a = 4) ∧ (b^2 = 12) ∧ (x^2 / 16 + y^2 / 12 = 1) :=
sorry

-- Prove minimum value of |AC| + |BD|
theorem minimum_distance (A B C D : ℝ × ℝ)
  (F1 : ℝ × ℝ)
  (h1 : foci_parallel F1 A B C D)
  (h2 : orthogonal_vectors A C B D) :
  |(AC : ℝ)| + |(BD : ℝ)| = 96 / 7 :=
sorry

end ellipse_equation_minimum_distance_l1743_174385
