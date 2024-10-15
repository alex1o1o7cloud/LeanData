import Mathlib

namespace NUMINAMATH_GPT_k_h_of_3_eq_79_l1221_122103

def h (x : ℝ) : ℝ := x^3
def k (x : ℝ) : ℝ := 3 * x - 2

theorem k_h_of_3_eq_79 : k (h 3) = 79 := by
  sorry

end NUMINAMATH_GPT_k_h_of_3_eq_79_l1221_122103


namespace NUMINAMATH_GPT_train_speed_l1221_122137

theorem train_speed (L1 L2: ℕ) (V2: ℕ) (T: ℕ) (V1: ℕ) : 
  L1 = 120 -> 
  L2 = 280 -> 
  V2 = 30 -> 
  T = 20 -> 
  (L1 + L2) * 18 = (V1 + V2) * T * 100 -> 
  V1 = 42 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_train_speed_l1221_122137


namespace NUMINAMATH_GPT_hand_position_at_8PM_yesterday_l1221_122158

-- Define the conditions of the problem
def positions : ℕ := 20
def jump_interval_min : ℕ := 7
def jump_positions : ℕ := 9
def start_position : ℕ := 0
def end_position : ℕ := 8 -- At 8:00 AM, the hand is at position 9, hence moving forward 8 positions from position 0

-- Define the total time from 8:00 PM yesterday to 8:00 AM today
def total_minutes : ℕ := 720

-- Calculate the number of full jumps
def num_full_jumps : ℕ := total_minutes / jump_interval_min

-- Calculate the hand's final position from 8:00 PM yesterday
def final_hand_position : ℕ := (start_position + num_full_jumps * jump_positions) % positions

-- Prove that the final hand position is 2
theorem hand_position_at_8PM_yesterday : final_hand_position = 2 :=
by
  sorry

end NUMINAMATH_GPT_hand_position_at_8PM_yesterday_l1221_122158


namespace NUMINAMATH_GPT_min_value_l1221_122161

theorem min_value (a b : ℝ) (h : a * b > 0) : (∃ x, x = a^2 + 4 * b^2 + 1 / (a * b) ∧ ∀ y, y = a^2 + 4 * b^2 + 1 / (a * b) → y ≥ 4) :=
sorry

end NUMINAMATH_GPT_min_value_l1221_122161


namespace NUMINAMATH_GPT_rest_area_location_l1221_122119

theorem rest_area_location : 
  ∃ (rest_area_milepost : ℕ), 
    let first_exit := 23
    let seventh_exit := 95
    let distance := seventh_exit - first_exit
    let halfway_distance := distance / 2
    rest_area_milepost = first_exit + halfway_distance :=
by
  sorry

end NUMINAMATH_GPT_rest_area_location_l1221_122119


namespace NUMINAMATH_GPT_find_b_l1221_122104

theorem find_b (b : ℤ) (h1 : ∀ (a b : ℤ), a * b = (a - 1) * (b - 1)) (h2 : 21 * b = 160) : b = 9 := by
  sorry

end NUMINAMATH_GPT_find_b_l1221_122104


namespace NUMINAMATH_GPT_continuity_f_at_1_l1221_122114

theorem continuity_f_at_1 (f : ℝ → ℝ) (x0 : ℝ)
  (h1 : f x0 = -12)
  (h2 : ∀ x : ℝ, f x = -5 * x^2 - 7)
  (h3 : x0 = 1) :
  ∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, |x - x0| < δ → |f x - f x0| < ε :=
by
  sorry

end NUMINAMATH_GPT_continuity_f_at_1_l1221_122114


namespace NUMINAMATH_GPT_proportion_equiv_l1221_122149

theorem proportion_equiv (X : ℕ) (h : 8 / 4 = X / 240) : X = 480 :=
by
  sorry

end NUMINAMATH_GPT_proportion_equiv_l1221_122149


namespace NUMINAMATH_GPT_an_plus_an_minus_1_eq_two_pow_n_l1221_122156

def a_n (n : ℕ) : ℕ := sorry -- Placeholder for the actual function a_n

theorem an_plus_an_minus_1_eq_two_pow_n (n : ℕ) (h : n ≥ 4) : a_n (n - 1) + a_n n = 2^n := 
by
  sorry

end NUMINAMATH_GPT_an_plus_an_minus_1_eq_two_pow_n_l1221_122156


namespace NUMINAMATH_GPT_quadratic_fraction_formula_l1221_122170

theorem quadratic_fraction_formula (p q α β : ℝ) 
  (h1 : α + β = p) 
  (h2 : α * β = 6) 
  (h3 : p^2 ≠ 12) 
  (h4 : ∃ x : ℝ, x^2 - p * x + q = 0) :
  (α + β) / (α^2 + β^2) = p / (p^2 - 12) :=
sorry

end NUMINAMATH_GPT_quadratic_fraction_formula_l1221_122170


namespace NUMINAMATH_GPT_frank_worked_days_l1221_122147

def total_hours : ℝ := 8.0
def hours_per_day : ℝ := 2.0

theorem frank_worked_days :
  (total_hours / hours_per_day = 4.0) :=
by sorry

end NUMINAMATH_GPT_frank_worked_days_l1221_122147


namespace NUMINAMATH_GPT_hcf_two_numbers_l1221_122105

theorem hcf_two_numbers
  (x y : ℕ) 
  (h_lcm : Nat.lcm x y = 560)
  (h_prod : x * y = 42000) : Nat.gcd x y = 75 :=
by
  sorry

end NUMINAMATH_GPT_hcf_two_numbers_l1221_122105


namespace NUMINAMATH_GPT_find_a21_l1221_122172

def seq_a (n : ℕ) : ℝ := sorry  -- This should define the sequence a_n
def seq_b (n : ℕ) : ℝ := sorry  -- This should define the sequence b_n

theorem find_a21 (h1 : seq_a 1 = 2)
  (h2 : ∀ n, seq_b n = seq_a (n + 1) / seq_a n)
  (h3 : ∀ n m, seq_b n = seq_b m * r^(n - m)) 
  (h4 : seq_b 10 * seq_b 11 = 2) :
  seq_a 21 = 2 ^ 11 :=
sorry

end NUMINAMATH_GPT_find_a21_l1221_122172


namespace NUMINAMATH_GPT_min_value_of_expression_l1221_122102

noncomputable def min_val_expr (x y : ℝ) : ℝ :=
  (8 / (x + 1)) + (1 / y)

theorem min_value_of_expression
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hcond : 2 * x + y = 1) :
  min_val_expr x y = (25 / 3) :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1221_122102


namespace NUMINAMATH_GPT_hiker_walks_18_miles_on_first_day_l1221_122148

noncomputable def miles_walked_first_day (h : ℕ) : ℕ := 3 * h

def total_miles_walked (h : ℕ) : ℕ := (3 * h) + (4 * (h - 1)) + (4 * h)

theorem hiker_walks_18_miles_on_first_day :
  (∃ h : ℕ, total_miles_walked h = 62) → miles_walked_first_day 6 = 18 :=
by
  sorry

end NUMINAMATH_GPT_hiker_walks_18_miles_on_first_day_l1221_122148


namespace NUMINAMATH_GPT_intersection_m_zero_range_of_m_l1221_122182

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def B (x : ℝ) (m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

theorem intersection_m_zero : 
  ∀ x : ℝ, A x → B x 0 ↔ (1 ≤ x ∧ x < 3) :=
sorry

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, A x → B x m) ∧ (∃ x : ℝ, B x m ∧ ¬A x) → (m ≤ -2 ∨ m ≥ 4) :=
sorry

end NUMINAMATH_GPT_intersection_m_zero_range_of_m_l1221_122182


namespace NUMINAMATH_GPT_cubics_sum_l1221_122117

theorem cubics_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : x^3 + y^3 = 640 :=
by
  sorry

end NUMINAMATH_GPT_cubics_sum_l1221_122117


namespace NUMINAMATH_GPT_acuteAngleAt725_l1221_122133

noncomputable def hourHandPosition (h : ℝ) (m : ℝ) : ℝ :=
  h * 30 + m / 60 * 30

noncomputable def minuteHandPosition (m : ℝ) : ℝ :=
  m / 60 * 360

noncomputable def angleBetweenHands (h m : ℝ) : ℝ :=
  abs (hourHandPosition h m - minuteHandPosition m)

theorem acuteAngleAt725 : angleBetweenHands 7 25 = 72.5 :=
  sorry

end NUMINAMATH_GPT_acuteAngleAt725_l1221_122133


namespace NUMINAMATH_GPT_joe_spent_255_minutes_l1221_122187

-- Define the time taken to cut hair for women, men, and children
def time_per_woman : Nat := 50
def time_per_man : Nat := 15
def time_per_child : Nat := 25

-- Define the number of haircuts for each category
def women_haircuts : Nat := 3
def men_haircuts : Nat := 2
def children_haircuts : Nat := 3

-- Compute the total time spent cutting hair
def total_time_spent : Nat :=
  (women_haircuts * time_per_woman) +
  (men_haircuts * time_per_man) +
  (children_haircuts * time_per_child)

-- The theorem stating the total time spent is equal to 255 minutes
theorem joe_spent_255_minutes : total_time_spent = 255 := by
  sorry

end NUMINAMATH_GPT_joe_spent_255_minutes_l1221_122187


namespace NUMINAMATH_GPT_cylinder_radius_l1221_122110

theorem cylinder_radius (h r: ℝ) (S: ℝ) (S_eq: S = 130 * Real.pi) (h_eq: h = 8) 
    (surface_area_eq: S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) : 
    r = 5 :=
by {
  -- Placeholder for proof steps.
  sorry
}

end NUMINAMATH_GPT_cylinder_radius_l1221_122110


namespace NUMINAMATH_GPT_initial_rows_of_chairs_l1221_122164

theorem initial_rows_of_chairs (x : ℕ) (h1 : 12 * x + 11 = 95) : x = 7 := 
by
  sorry

end NUMINAMATH_GPT_initial_rows_of_chairs_l1221_122164


namespace NUMINAMATH_GPT_smiths_bakery_multiple_l1221_122145

theorem smiths_bakery_multiple (x : ℤ) (mcgee_pies : ℤ) (smith_pies : ℤ) 
  (h1 : smith_pies = x * mcgee_pies + 6)
  (h2 : mcgee_pies = 16)
  (h3 : smith_pies = 70) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_smiths_bakery_multiple_l1221_122145


namespace NUMINAMATH_GPT_compute_sum_pq_pr_qr_l1221_122121

theorem compute_sum_pq_pr_qr (p q r : ℝ) (h : 5 * (p + q + r) = p^2 + q^2 + r^2) : 
  let N := 150
  let n := -12.5
  N + 15 * n = -37.5 := 
by {
  sorry
}

end NUMINAMATH_GPT_compute_sum_pq_pr_qr_l1221_122121


namespace NUMINAMATH_GPT_ab_plus_a_plus_b_l1221_122138

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^4 - 6 * x^2 - x + 2
-- Define the conditions on a and b
def is_root (x : ℝ) : Prop := poly x = 0

-- State the theorem
theorem ab_plus_a_plus_b (a b : ℝ) (ha : is_root a) (hb : is_root b) : a * b + a + b = 1 :=
sorry

end NUMINAMATH_GPT_ab_plus_a_plus_b_l1221_122138


namespace NUMINAMATH_GPT_unique_positive_x_for_volume_l1221_122181

variable (x : ℕ)

def prism_volume (x : ℕ) : ℕ :=
  (x + 5) * (x - 5) * (x ^ 2 + 25)

theorem unique_positive_x_for_volume {x : ℕ} (h : prism_volume x < 700) (h_pos : 0 < x) :
  ∃! x, (prism_volume x < 700) ∧ (x - 5 > 0) :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_x_for_volume_l1221_122181


namespace NUMINAMATH_GPT_total_crayons_correct_l1221_122188

-- Define the number of crayons each child has
def crayons_per_child : ℕ := 12

-- Define the number of children
def number_of_children : ℕ := 18

-- Define the total number of crayons
def total_crayons : ℕ := crayons_per_child * number_of_children

-- State the theorem
theorem total_crayons_correct : total_crayons = 216 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_crayons_correct_l1221_122188


namespace NUMINAMATH_GPT_total_cost_correct_l1221_122185

-- Define the individual costs and quantities
def pumpkin_cost : ℝ := 2.50
def tomato_cost : ℝ := 1.50
def chili_pepper_cost : ℝ := 0.90

def pumpkin_quantity : ℕ := 3
def tomato_quantity : ℕ := 4
def chili_pepper_quantity : ℕ := 5

-- Define the total cost calculation
def total_cost : ℝ :=
  pumpkin_quantity * pumpkin_cost +
  tomato_quantity * tomato_cost +
  chili_pepper_quantity * chili_pepper_cost

-- Prove the total cost is $18.00
theorem total_cost_correct : total_cost = 18.00 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1221_122185


namespace NUMINAMATH_GPT_average_speed_correct_l1221_122154

noncomputable def initial_odometer := 12321
noncomputable def final_odometer := 12421
noncomputable def time_hours := 4
noncomputable def distance := final_odometer - initial_odometer
noncomputable def avg_speed := distance / time_hours

theorem average_speed_correct : avg_speed = 25 := by
  sorry

end NUMINAMATH_GPT_average_speed_correct_l1221_122154


namespace NUMINAMATH_GPT_trey_more_turtles_than_kristen_l1221_122122

theorem trey_more_turtles_than_kristen (kristen_turtles : ℕ) 
  (H1 : kristen_turtles = 12) 
  (H2 : ∀ kris_turtles, kris_turtles = (1 / 4) * kristen_turtles)
  (H3 : ∀ kris_turtles trey_turtles, trey_turtles = 7 * kris_turtles) :
  ∃ trey_turtles, trey_turtles - kristen_turtles = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_trey_more_turtles_than_kristen_l1221_122122


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1221_122130

-- Define the quadratic equation
def quadratic_eq (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + a

-- State the necessary but not sufficient condition proof statement
theorem necessary_but_not_sufficient (a : ℝ) :
  (∃ x y : ℝ, quadratic_eq a x = 0 ∧ quadratic_eq a y = 0 ∧ x > 0 ∧ y < 0) → a < 1 :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1221_122130


namespace NUMINAMATH_GPT_Tyler_cucumbers_and_grapes_l1221_122101

theorem Tyler_cucumbers_and_grapes (a b c g : ℝ) (h1 : 10 * a = 5 * b) (h2 : 3 * b = 4 * c) (h3 : 4 * c = 6 * g) :
  (20 * a = (40 / 3) * c) ∧ (20 * a = 20 * g) :=
by
  sorry

end NUMINAMATH_GPT_Tyler_cucumbers_and_grapes_l1221_122101


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1221_122146

variable (f : ℝ → ℝ)

-- Define even function
def even_function : Prop := ∀ x, f x = f (-x)

-- Define periodic function with period 2
def periodic_function : Prop := ∀ x, f (x + 2) = f x

-- Define increasing function on [0, 1]
def increasing_on_0_1 : Prop := ∀ x y, 0 ≤ x → x ≤ y → y ≤ 1 → f x ≤ f y

-- Define decreasing function on [3, 4]
def decreasing_on_3_4 : Prop := ∀ x y, 3 ≤ x → x ≤ y → y ≤ 4 → f x ≥ f y

theorem necessary_and_sufficient_condition :
  even_function f →
  periodic_function f →
  (increasing_on_0_1 f ↔ decreasing_on_3_4 f) :=
by
  intros h_even h_periodic
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1221_122146


namespace NUMINAMATH_GPT_pheromone_effect_on_population_l1221_122108

-- Definitions of conditions
def disrupt_sex_ratio (uses_pheromones : Bool) : Bool :=
  uses_pheromones = true

def decrease_birth_rate (disrupt_sex_ratio : Bool) : Bool :=
  disrupt_sex_ratio = true

def decrease_population_density (decrease_birth_rate : Bool) : Bool :=
  decrease_birth_rate = true

-- Problem Statement for Lean 4
theorem pheromone_effect_on_population (uses_pheromones : Bool) :
  disrupt_sex_ratio uses_pheromones = true →
  decrease_birth_rate (disrupt_sex_ratio uses_pheromones) = true →
  decrease_population_density (decrease_birth_rate (disrupt_sex_ratio uses_pheromones)) = true :=
sorry

end NUMINAMATH_GPT_pheromone_effect_on_population_l1221_122108


namespace NUMINAMATH_GPT_evaluate_g_h_2_l1221_122150

def g (x : ℝ) : ℝ := 3 * x^2 - 4 
def h (x : ℝ) : ℝ := -2 * x^3 + 2 

theorem evaluate_g_h_2 : g (h 2) = 584 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_h_2_l1221_122150


namespace NUMINAMATH_GPT_find_unique_positive_integers_l1221_122169

theorem find_unique_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  3 ^ x + 7 = 2 ^ y → x = 2 ∧ y = 4 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_find_unique_positive_integers_l1221_122169


namespace NUMINAMATH_GPT_zoo_ticket_sales_l1221_122165

theorem zoo_ticket_sales (A K : ℕ) (h1 : A + K = 254) (h2 : 28 * A + 12 * K = 3864) : K = 202 :=
by {
  sorry
}

end NUMINAMATH_GPT_zoo_ticket_sales_l1221_122165


namespace NUMINAMATH_GPT_find_page_words_l1221_122141
open Nat

-- Define the conditions
def condition1 : Nat := 150
def condition2 : Nat := 221
def total_words_modulo : Nat := 220
def upper_bound_words : Nat := 120

-- Define properties
def is_solution (p : Nat) : Prop :=
  Nat.Prime p ∧ p ≤ upper_bound_words ∧ (condition1 * p) % condition2 = total_words_modulo

-- The theorem to prove
theorem find_page_words (p : Nat) (hp : is_solution p) : p = 67 :=
by
  sorry

end NUMINAMATH_GPT_find_page_words_l1221_122141


namespace NUMINAMATH_GPT_gum_left_after_sharing_l1221_122120

-- Define the initial state of Adrianna's gum and the changes to it
def initial_gum : Nat := 10
def additional_gum : Nat := 3
def given_out_gum : Nat := 11

-- Define the final state of Adrianna's gum
def final_gum : Nat := initial_gum + additional_gum - given_out_gum

-- Prove that Adrianna ends up with 2 pieces of gum under the given conditions
theorem gum_left_after_sharing :
  final_gum = 2 :=
by 
  -- Since this is just the statement and not the proof, we end with sorry.
  sorry

end NUMINAMATH_GPT_gum_left_after_sharing_l1221_122120


namespace NUMINAMATH_GPT_hours_of_work_l1221_122123

variables (M W X : ℝ)

noncomputable def work_rate := 
  (2 * M + 3 * W) * X * 5 = 1 ∧ 
  (4 * M + 4 * W) * 3 * 7 = 1 ∧ 
  7 * M * 4 * 5.000000000000001 = 1

theorem hours_of_work (M W : ℝ) (h : work_rate M W 7) : X = 7 :=
sorry

end NUMINAMATH_GPT_hours_of_work_l1221_122123


namespace NUMINAMATH_GPT_solve_x_perpendicular_l1221_122135

def vec_a : ℝ × ℝ := (1, 3)
def vec_b (x : ℝ) : ℝ × ℝ := (3, x)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem solve_x_perpendicular (x : ℝ) (h : perpendicular vec_a (vec_b x)) : x = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_x_perpendicular_l1221_122135


namespace NUMINAMATH_GPT_symmetric_line_equation_l1221_122160

theorem symmetric_line_equation 
  (L : ℝ → ℝ → Prop)
  (H : ∀ x y, L x y ↔ x - 2 * y + 1 = 0) : 
  ∃ L' : ℝ → ℝ → Prop, 
    (∀ x y, L' x y ↔ x + 2 * y - 3 = 0) ∧ 
    ( ∀ x y, L (2 - x) y ↔ L' x y ) := 
sorry

end NUMINAMATH_GPT_symmetric_line_equation_l1221_122160


namespace NUMINAMATH_GPT_Jimmy_earns_229_l1221_122118

-- Definitions based on conditions from the problem
def number_of_type_A : ℕ := 5
def number_of_type_B : ℕ := 4
def number_of_type_C : ℕ := 3

def value_of_type_A : ℕ := 20
def value_of_type_B : ℕ := 30
def value_of_type_C : ℕ := 40

def discount_type_A : ℕ := 7
def discount_type_B : ℕ := 10
def discount_type_C : ℕ := 12

-- Calculation of the total amount Jimmy will earn
def total_earnings : ℕ :=
  let price_A := value_of_type_A - discount_type_A
  let price_B := value_of_type_B - discount_type_B
  let price_C := value_of_type_C - discount_type_C
  (number_of_type_A * price_A) +
  (number_of_type_B * price_B) +
  (number_of_type_C * price_C)

-- The statement to be proved
theorem Jimmy_earns_229 : total_earnings = 229 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_Jimmy_earns_229_l1221_122118


namespace NUMINAMATH_GPT_negation_proposition_equivalence_l1221_122113

theorem negation_proposition_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2 * x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_equivalence_l1221_122113


namespace NUMINAMATH_GPT_inverse_proportion_m_range_l1221_122171

theorem inverse_proportion_m_range (m : ℝ) :
  (∀ x : ℝ, x < 0 → ∀ y1 y2 : ℝ, y1 = (1 - 2 * m) / x → y2 = (1 - 2 * m) / (x + 1) → y1 < y2) 
  ↔ (m > 1 / 2) :=
by sorry

end NUMINAMATH_GPT_inverse_proportion_m_range_l1221_122171


namespace NUMINAMATH_GPT_ziggy_rap_requests_l1221_122151

variables (total_songs electropop dance rock oldies djs_choice rap : ℕ)

-- Given conditions
axiom total_songs_eq : total_songs = 30
axiom electropop_eq : electropop = total_songs / 2
axiom dance_eq : dance = electropop / 3
axiom rock_eq : rock = 5
axiom oldies_eq : oldies = rock - 3
axiom djs_choice_eq : djs_choice = oldies / 2

-- Proof statement
theorem ziggy_rap_requests : rap = total_songs - electropop - dance - rock - oldies - djs_choice :=
by
  -- Apply the axioms and conditions to prove the resulting rap count
  sorry

end NUMINAMATH_GPT_ziggy_rap_requests_l1221_122151


namespace NUMINAMATH_GPT_cookies_left_correct_l1221_122163

def cookies_left (cookies_per_dozen : ℕ) (flour_per_dozen_lb : ℕ) (bag_count : ℕ) (flour_per_bag_lb : ℕ) (cookies_eaten : ℕ) : ℕ :=
  let total_flour_lb := bag_count * flour_per_bag_lb
  let total_cookies := (total_flour_lb / flour_per_dozen_lb) * cookies_per_dozen
  total_cookies - cookies_eaten

theorem cookies_left_correct :
  cookies_left 12 2 4 5 15 = 105 :=
by sorry

end NUMINAMATH_GPT_cookies_left_correct_l1221_122163


namespace NUMINAMATH_GPT_gcd_8251_6105_l1221_122178

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end NUMINAMATH_GPT_gcd_8251_6105_l1221_122178


namespace NUMINAMATH_GPT_hotel_loss_l1221_122198

theorem hotel_loss (operations_expenses : ℝ) (payment_fraction : ℝ) (total_payment : ℝ) (loss : ℝ) 
  (hOpExp : operations_expenses = 100) 
  (hPayFr : payment_fraction = 3 / 4)
  (hTotalPay : total_payment = payment_fraction * operations_expenses) 
  (hLossCalc : loss = operations_expenses - total_payment) : 
  loss = 25 := 
by 
  sorry

end NUMINAMATH_GPT_hotel_loss_l1221_122198


namespace NUMINAMATH_GPT_socks_probability_l1221_122173

theorem socks_probability :
  let total_socks := 18
  let total_pairs := (total_socks.choose 2)
  let gray_socks := 12
  let white_socks := 6
  let gray_pairs := (gray_socks.choose 2)
  let white_pairs := (white_socks.choose 2)
  let same_color_pairs := gray_pairs + white_pairs
  same_color_pairs / total_pairs = (81 / 153) :=
by
  sorry

end NUMINAMATH_GPT_socks_probability_l1221_122173


namespace NUMINAMATH_GPT_count_three_element_arithmetic_mean_subsets_l1221_122139
open Nat

theorem count_three_element_arithmetic_mean_subsets (n : ℕ) (h : n ≥ 3) :
    ∃ a_n : ℕ, a_n = (n / 2) * ((n - 1) / 2) :=
by
  sorry

end NUMINAMATH_GPT_count_three_element_arithmetic_mean_subsets_l1221_122139


namespace NUMINAMATH_GPT_cheaper_fuji_shimla_l1221_122153

variable (S R F : ℝ)
variable (h : 1.05 * (S + R) = R + 0.90 * F + 250)

theorem cheaper_fuji_shimla : S - F = (-0.15 * S - 0.05 * R) / 0.90 + 250 / 0.90 :=
by
  sorry

end NUMINAMATH_GPT_cheaper_fuji_shimla_l1221_122153


namespace NUMINAMATH_GPT_smaller_root_of_equation_l1221_122136

theorem smaller_root_of_equation : 
  ∀ x : ℝ, (x - 3 / 4) * (x - 3 / 4) + (x - 3 / 4) * (x - 1 / 4) = 0 → x = 1 / 2 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_smaller_root_of_equation_l1221_122136


namespace NUMINAMATH_GPT_quadratic_equation_is_D_l1221_122112

theorem quadratic_equation_is_D (x a b c : ℝ) : 
  (¬ (∃ b' : ℝ, (x^2 - 2) * x = b' * x + 2)) ∧
  (¬ ((a ≠ 0) ∧ (ax^2 + bx + c = 0))) ∧
  (¬ (x + (1 / x) = 5)) ∧
  ((x^2 = 0) ↔ true) :=
by sorry

end NUMINAMATH_GPT_quadratic_equation_is_D_l1221_122112


namespace NUMINAMATH_GPT_f_is_odd_range_of_x_l1221_122197

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂
axiom f_3 : f 3 = 1
axiom f_increase_nonneg : ∀ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ x₂) → f x₁ ≤ f x₂
axiom f_lt_2 : ∀ x : ℝ, f (x - 1) < 2

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem range_of_x : {x : ℝ | f (x - 1) < 2} =
{s : ℝ | sorry } :=
sorry

end NUMINAMATH_GPT_f_is_odd_range_of_x_l1221_122197


namespace NUMINAMATH_GPT_supplement_of_angle_l1221_122195

theorem supplement_of_angle (θ : ℝ) 
  (h_complement: θ = 90 - 30) : 180 - θ = 120 :=
by
  sorry

end NUMINAMATH_GPT_supplement_of_angle_l1221_122195


namespace NUMINAMATH_GPT_TwentyFifthMultipleOfFour_l1221_122186

theorem TwentyFifthMultipleOfFour (n : ℕ) (h : ∀ k, 0 <= k ∧ k <= 24 → n = 16 + 4 * k) : n = 112 :=
by
  sorry

end NUMINAMATH_GPT_TwentyFifthMultipleOfFour_l1221_122186


namespace NUMINAMATH_GPT_find_certain_number_l1221_122142

noncomputable def certain_number (x : ℝ) : Prop :=
  3005 - 3000 + x = 2705

theorem find_certain_number : ∃ x : ℝ, certain_number x ∧ x = 2700 :=
by
  use 2700
  unfold certain_number
  sorry

end NUMINAMATH_GPT_find_certain_number_l1221_122142


namespace NUMINAMATH_GPT_intersection_A_B_l1221_122127

def A : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 2 * x + 5}
def B : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 1 - 2 * x}
def inter : Set (ℝ × ℝ) := {(x, y) | x = -1 ∧ y = 3}

theorem intersection_A_B :
  A ∩ B = inter :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1221_122127


namespace NUMINAMATH_GPT_area_of_garden_l1221_122167

variable (P : ℝ) (A : ℝ)

theorem area_of_garden (hP : P = 38) (hA : A = 2 * P + 14.25) : A = 90.25 :=
by
  sorry

end NUMINAMATH_GPT_area_of_garden_l1221_122167


namespace NUMINAMATH_GPT_jonessa_take_home_pay_l1221_122115

noncomputable def tax_rate : ℝ := 0.10
noncomputable def pay : ℝ := 500
noncomputable def tax_amount : ℝ := pay * tax_rate
noncomputable def take_home_pay : ℝ := pay - tax_amount

theorem jonessa_take_home_pay : take_home_pay = 450 := by
  have h1 : tax_amount = 50 := by
    sorry
  have h2 : take_home_pay = 450 := by
    sorry
  exact h2

end NUMINAMATH_GPT_jonessa_take_home_pay_l1221_122115


namespace NUMINAMATH_GPT_solution_l1221_122176

theorem solution (y q : ℝ) (h1 : |y - 3| = q) (h2 : y < 3) : y - 2 * q = 3 - 3 * q :=
by
  sorry

end NUMINAMATH_GPT_solution_l1221_122176


namespace NUMINAMATH_GPT_fred_initial_dimes_l1221_122100

theorem fred_initial_dimes (current_dimes borrowed_dimes initial_dimes : ℕ)
  (hc : current_dimes = 4)
  (hb : borrowed_dimes = 3)
  (hi : current_dimes + borrowed_dimes = initial_dimes) :
  initial_dimes = 7 := 
by
  sorry

end NUMINAMATH_GPT_fred_initial_dimes_l1221_122100


namespace NUMINAMATH_GPT_geometric_mean_of_4_and_9_l1221_122134

theorem geometric_mean_of_4_and_9 : ∃ G : ℝ, (4 / G = G / 9) ∧ (G = 6 ∨ G = -6) := 
by
  sorry

end NUMINAMATH_GPT_geometric_mean_of_4_and_9_l1221_122134


namespace NUMINAMATH_GPT_sum_n_k_l1221_122180

theorem sum_n_k (n k : ℕ) (h1 : 3 * (k + 1) = n - k) (h2 : 2 * (k + 2) = n - k - 1) : n + k = 13 := by
  sorry

end NUMINAMATH_GPT_sum_n_k_l1221_122180


namespace NUMINAMATH_GPT_probability_of_at_most_3_heads_l1221_122190

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end NUMINAMATH_GPT_probability_of_at_most_3_heads_l1221_122190


namespace NUMINAMATH_GPT_additional_people_needed_l1221_122194

theorem additional_people_needed (h₁ : ∀ p h : ℕ, (p * h = 40)) (h₂ : 5 * 8 = 40) : 7 - 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_additional_people_needed_l1221_122194


namespace NUMINAMATH_GPT_find_missing_number_l1221_122157

theorem find_missing_number (x : ℕ) (h : 10111 - x * 2 * 5 = 10011) : x = 5 := 
sorry

end NUMINAMATH_GPT_find_missing_number_l1221_122157


namespace NUMINAMATH_GPT_cubic_roots_sum_of_cubes_l1221_122189

def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem cubic_roots_sum_of_cubes :
  let α := cube_root 17
  let β := cube_root 73
  let γ := cube_root 137
  ∀ (a b c : ℝ),
    (a - α) * (a - β) * (a - γ) = 1/2 ∧
    (b - α) * (b - β) * (b - γ) = 1/2 ∧
    (c - α) * (c - β) * (c - γ) = 1/2 →
    a^3 + b^3 + c^3 = 228.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_cubic_roots_sum_of_cubes_l1221_122189


namespace NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l1221_122162

def opposite_of (x : Int) : Int := -x

theorem opposite_of_2023_is_neg_2023 : opposite_of 2023 = -2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l1221_122162


namespace NUMINAMATH_GPT_line_equation_l1221_122199

theorem line_equation (a b : ℝ) (h_intercept_eq : a = b) (h_pass_through : 3 * a + 2 * b = 2 * a + 5) : (3 + 2 = 5) ↔ (a = 5 ∧ b = 5) :=
sorry

end NUMINAMATH_GPT_line_equation_l1221_122199


namespace NUMINAMATH_GPT_jim_out_of_pocket_cost_l1221_122155

theorem jim_out_of_pocket_cost {price1 price2 sale : ℕ} 
    (h1 : price1 = 10000)
    (h2 : price2 = 2 * price1)
    (h3 : sale = price1 / 2) :
    (price1 + price2 - sale = 25000) :=
by
  sorry

end NUMINAMATH_GPT_jim_out_of_pocket_cost_l1221_122155


namespace NUMINAMATH_GPT_cows_in_group_l1221_122131

theorem cows_in_group (c h : ℕ) (L H: ℕ) 
  (legs_eq : L = 4 * c + 2 * h)
  (heads_eq : H = c + h)
  (legs_heads_relation : L = 2 * H + 14) 
  : c = 7 :=
by
  sorry

end NUMINAMATH_GPT_cows_in_group_l1221_122131


namespace NUMINAMATH_GPT_sunny_weather_prob_correct_l1221_122192

def rain_prob : ℝ := 0.45
def cloudy_prob : ℝ := 0.20
def sunny_prob : ℝ := 1 - rain_prob - cloudy_prob

theorem sunny_weather_prob_correct : sunny_prob = 0.35 := by
  sorry

end NUMINAMATH_GPT_sunny_weather_prob_correct_l1221_122192


namespace NUMINAMATH_GPT_find_a_c_l1221_122168

theorem find_a_c (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_neg : c < 0)
    (h_max : c + a = 3) (h_min : c - a = -5) :
  a = 4 ∧ c = -1 := 
sorry

end NUMINAMATH_GPT_find_a_c_l1221_122168


namespace NUMINAMATH_GPT_triangle_base_and_height_l1221_122166

theorem triangle_base_and_height (h b : ℕ) (A : ℕ) (hb : b = h - 4) (hA : A = 96) 
  (hArea : A = (1 / 2) * b * h) : (b = 12 ∧ h = 16) :=
by
  sorry

end NUMINAMATH_GPT_triangle_base_and_height_l1221_122166


namespace NUMINAMATH_GPT_find_y_l1221_122143

theorem find_y
  (x y : ℝ)
  (h1 : x^(3*y) = 8)
  (h2 : x = 2) :
  y = 1 :=
sorry

end NUMINAMATH_GPT_find_y_l1221_122143


namespace NUMINAMATH_GPT_team_leader_and_deputy_choice_l1221_122174

def TeamLeaderSelection : Type := {x : Fin 5 // true}
def DeputyLeaderSelection (TL : TeamLeaderSelection) : Type := {x : Fin 5 // x ≠ TL.val}

theorem team_leader_and_deputy_choice : 
  (Σ TL : TeamLeaderSelection, DeputyLeaderSelection TL) → Fin 20 :=
by sorry

end NUMINAMATH_GPT_team_leader_and_deputy_choice_l1221_122174


namespace NUMINAMATH_GPT_find_f_of_f_l1221_122126

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else (4 * x + 1 - 2 / x) / 3

theorem find_f_of_f (h : ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 2 * x + 1) : 
  f 2 = -1/3 :=
sorry

end NUMINAMATH_GPT_find_f_of_f_l1221_122126


namespace NUMINAMATH_GPT_exists_indices_non_decreasing_l1221_122109

theorem exists_indices_non_decreasing
    (a b c : ℕ → ℕ) :
    ∃ p q : ℕ, p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
  sorry

end NUMINAMATH_GPT_exists_indices_non_decreasing_l1221_122109


namespace NUMINAMATH_GPT_circle_intersection_range_l1221_122124

theorem circle_intersection_range (m : ℝ) :
  (x^2 + y^2 - 4*x + 2*m*y + m + 6 = 0) ∧ 
  (∀ A B : ℝ, 
    (A - y = 0) ∧ (B - y = 0) → A * B > 0
  ) → 
  (m > 2 ∨ (-6 < m ∧ m < -2)) :=
by 
  sorry

end NUMINAMATH_GPT_circle_intersection_range_l1221_122124


namespace NUMINAMATH_GPT_cubic_polynomial_p_value_l1221_122116

noncomputable def p (x : ℝ) : ℝ := sorry

theorem cubic_polynomial_p_value :
  (∀ n ∈ ({1, 2, 3, 5} : Finset ℝ), p n = 1 / n ^ 2) →
  p 4 = 1 / 150 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_cubic_polynomial_p_value_l1221_122116


namespace NUMINAMATH_GPT_proof_problem_l1221_122129

-- Definitions for the arithmetic and geometric sequences
def a_n (n : ℕ) : ℚ := 2 * n - 4
def b_n (n : ℕ) : ℚ := 2^(n - 2)

-- Conditions based on initial problem statements
axiom a_2 : a_n 2 = 0
axiom b_2 : b_n 2 = 1
axiom a_3_eq_b_3 : a_n 3 = b_n 3
axiom a_4_eq_b_4 : a_n 4 = b_n 4

-- Sum of first n terms of the sequence {n * b_n}
def S_n (n : ℕ) : ℚ := (n-1) * 2^(n-1) + 1/2

-- The main theorem to prove
theorem proof_problem (n : ℕ) : ∃ a_n b_n S_n, 
    (a_n = 2 * n - 4) ∧
    (b_n = 2^(n - 2)) ∧
    (S_n = (n-1) * 2^(n-1) + 1/2) :=
by {
    sorry
}

end NUMINAMATH_GPT_proof_problem_l1221_122129


namespace NUMINAMATH_GPT_small_cubes_with_two_faces_painted_red_l1221_122132

theorem small_cubes_with_two_faces_painted_red (edge_length : ℕ) (small_cube_edge_length : ℕ)
  (h1 : edge_length = 4) (h2 : small_cube_edge_length = 1) :
  ∃ n, n = 24 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_small_cubes_with_two_faces_painted_red_l1221_122132


namespace NUMINAMATH_GPT_cricket_innings_count_l1221_122144

theorem cricket_innings_count (n : ℕ) (h_avg_current : ∀ (total_runs : ℕ), total_runs = 32 * n)
  (h_runs_needed : ∀ (total_runs : ℕ), total_runs + 116 = 36 * (n + 1)) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_cricket_innings_count_l1221_122144


namespace NUMINAMATH_GPT_volunteer_recommendations_l1221_122152

def num_recommendations (boys girls : ℕ) (total_choices chosen : ℕ) : ℕ :=
  let total_combinations := Nat.choose total_choices chosen
  let invalid_combinations := Nat.choose boys chosen
  total_combinations - invalid_combinations

theorem volunteer_recommendations : num_recommendations 4 3 7 4 = 34 := by
  sorry

end NUMINAMATH_GPT_volunteer_recommendations_l1221_122152


namespace NUMINAMATH_GPT_number_cooking_and_weaving_l1221_122140

section CurriculumProblem

variables {total_yoga total_cooking total_weaving : ℕ}
variables {cooking_only cooking_and_yoga all_curriculums CW : ℕ}

-- Given conditions
def yoga (total_yoga : ℕ) := total_yoga = 35
def cooking (total_cooking : ℕ) := total_cooking = 20
def weaving (total_weaving : ℕ) := total_weaving = 15
def cookingOnly (cooking_only : ℕ) := cooking_only = 7
def cookingAndYoga (cooking_and_yoga : ℕ) := cooking_and_yoga = 5
def allCurriculums (all_curriculums : ℕ) := all_curriculums = 3

-- Prove that CW (number of people studying both cooking and weaving) is 8
theorem number_cooking_and_weaving : 
  yoga total_yoga → cooking total_cooking → weaving total_weaving → 
  cookingOnly cooking_only → cookingAndYoga cooking_and_yoga → 
  allCurriculums all_curriculums → CW = 8 := 
by 
  intros h_yoga h_cooking h_weaving h_cookingOnly h_cookingAndYoga h_allCurriculums
  -- Placeholder for the actual proof
  sorry

end CurriculumProblem

end NUMINAMATH_GPT_number_cooking_and_weaving_l1221_122140


namespace NUMINAMATH_GPT_sum_of_roots_l1221_122128

theorem sum_of_roots (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - 2 * x₁ - 8 = 0) 
  (h₂ : x₂^2 - 2 * x₂ - 8 = 0)
  (h_distinct : x₁ ≠ x₂) : 
  x₁ + x₂ = 2 := 
sorry

end NUMINAMATH_GPT_sum_of_roots_l1221_122128


namespace NUMINAMATH_GPT_sqrt_mixed_number_simplified_l1221_122111

theorem sqrt_mixed_number_simplified : 
  (Real.sqrt (12 + 1 / 9) = Real.sqrt 109 / 3) := by
  sorry

end NUMINAMATH_GPT_sqrt_mixed_number_simplified_l1221_122111


namespace NUMINAMATH_GPT_total_drink_volume_l1221_122107

-- Define the percentages of the various juices
def grapefruit_percentage : ℝ := 0.20
def lemon_percentage : ℝ := 0.25
def pineapple_percentage : ℝ := 0.10
def mango_percentage : ℝ := 0.15

-- Define the volume of orange juice in ounces
def orange_juice_volume : ℝ := 24

-- State the total percentage of all juices other than orange juice
def non_orange_percentage : ℝ := grapefruit_percentage + lemon_percentage + pineapple_percentage + mango_percentage

-- Calculate the percentage of orange juice
def orange_percentage : ℝ := 1 - non_orange_percentage

-- State that the total volume of the drink is such that 30% of it is 24 ounces
theorem total_drink_volume : ∃ (total_volume : ℝ), (orange_percentage * total_volume = orange_juice_volume) ∧ (total_volume = 80) := by
  use 80
  sorry

end NUMINAMATH_GPT_total_drink_volume_l1221_122107


namespace NUMINAMATH_GPT_sum_of_possible_values_l1221_122159

theorem sum_of_possible_values
  (x : ℝ)
  (h : (x + 3) * (x - 4) = 22) :
  ∃ s : ℝ, s = 1 :=
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l1221_122159


namespace NUMINAMATH_GPT_sin_A_value_l1221_122179

variables {A B C a b c : ℝ}
variables {sin cos : ℝ → ℝ}

-- Conditions
axiom triangle_sides : ∀ (A B C: ℝ), ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0
axiom sin_cos_conditions : 3 * b * sin A = c * cos A + a * cos C

-- Proof statement
theorem sin_A_value (h : 3 * b * sin A = c * cos A + a * cos C) : sin A = 1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_sin_A_value_l1221_122179


namespace NUMINAMATH_GPT_num_ways_to_distribute_balls_into_boxes_l1221_122193

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end NUMINAMATH_GPT_num_ways_to_distribute_balls_into_boxes_l1221_122193


namespace NUMINAMATH_GPT_coin_ratio_l1221_122177

theorem coin_ratio (coins_1r coins_50p coins_25p : ℕ) (value_1r value_50p value_25p : ℕ) :
  coins_1r = 120 → coins_50p = 120 → coins_25p = 120 →
  value_1r = coins_1r * 1 → value_50p = coins_50p * 50 → value_25p = coins_25p * 25 →
  value_1r + value_50p + value_25p = 210 →
  (coins_1r : ℚ) / (coins_50p + coins_25p : ℚ) = (1 / 1) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_coin_ratio_l1221_122177


namespace NUMINAMATH_GPT_count_ordered_pairs_l1221_122196

theorem count_ordered_pairs (d n : ℕ) (h₁ : d ≥ 35) (h₂ : n > 0) 
    (h₃ : 45 + 2 * n < 120)
    (h₄ : ∃ a b : ℕ, 10 * a + b = 30 + n ∧ 10 * b + a = 35 + n ∧ a ≤ 9 ∧ b ≤ 9) :
    ∃ k : ℕ, -- number of valid ordered pairs (d, n)
    sorry := sorry

end NUMINAMATH_GPT_count_ordered_pairs_l1221_122196


namespace NUMINAMATH_GPT_toys_profit_l1221_122175

theorem toys_profit (sp cp : ℕ) (x : ℕ) (h1 : sp = 25200) (h2 : cp = 1200) (h3 : 18 * cp + x * cp = sp) :
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_toys_profit_l1221_122175


namespace NUMINAMATH_GPT_probability_of_winning_plan1_is_2_over_5_probability_of_winning_plan2_is_11_over_36_choose_plan1_l1221_122184

-- Definition of the total number of outcomes and outcomes where a player wins for Plan 1
def total_outcomes_plan1 := 15
def winning_outcomes_plan1 := 6
def probability_plan1 : ℚ := winning_outcomes_plan1 / total_outcomes_plan1

-- Definition of the total number of outcomes and outcomes where a player wins for Plan 2
def total_outcomes_plan2 := 36
def winning_outcomes_plan2 := 11
def probability_plan2 : ℚ := winning_outcomes_plan2 / total_outcomes_plan2

-- Statements to prove
theorem probability_of_winning_plan1_is_2_over_5 : probability_plan1 = 2 / 5 :=
by sorry

theorem probability_of_winning_plan2_is_11_over_36 : probability_plan2 = 11 / 36 :=
by sorry

theorem choose_plan1 : probability_plan1 > probability_plan2 :=
by sorry

end NUMINAMATH_GPT_probability_of_winning_plan1_is_2_over_5_probability_of_winning_plan2_is_11_over_36_choose_plan1_l1221_122184


namespace NUMINAMATH_GPT_exists_y_lt_p_div2_py_plus1_not_product_of_greater_y_l1221_122106

theorem exists_y_lt_p_div2_py_plus1_not_product_of_greater_y (p : ℕ) [hp : Fact (Nat.Prime p)] (h3 : 3 < p) :
  ∃ y : ℕ, y < p / 2 ∧ ∀ a b : ℕ, py + 1 ≠ a * b ∨ a ≤ y ∨ b ≤ y :=
by
  sorry

end NUMINAMATH_GPT_exists_y_lt_p_div2_py_plus1_not_product_of_greater_y_l1221_122106


namespace NUMINAMATH_GPT_factory_earns_8100_per_day_l1221_122125

-- Define the conditions
def working_hours_machines := 23
def working_hours_fourth_machine := 12
def production_per_hour := 2
def price_per_kg := 50
def number_of_machines := 3

-- Calculate earnings
def total_earnings : ℕ :=
  let total_runtime_machines := number_of_machines * working_hours_machines
  let production_machines := total_runtime_machines * production_per_hour
  let production_fourth_machine := working_hours_fourth_machine * production_per_hour
  let total_production := production_machines + production_fourth_machine
  total_production * price_per_kg

theorem factory_earns_8100_per_day : total_earnings = 8100 :=
by
  sorry

end NUMINAMATH_GPT_factory_earns_8100_per_day_l1221_122125


namespace NUMINAMATH_GPT_cos_value_l1221_122183

theorem cos_value (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) : Real.cos (2 * π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_value_l1221_122183


namespace NUMINAMATH_GPT_cross_number_puzzle_hundreds_digit_l1221_122191

theorem cross_number_puzzle_hundreds_digit :
  ∃ a b : ℕ, a ≥ 5 ∧ a ≤ 6 ∧ b = 3 ∧ (3^a / 100 = 7 ∨ 7^b / 100 = 7) :=
sorry

end NUMINAMATH_GPT_cross_number_puzzle_hundreds_digit_l1221_122191
