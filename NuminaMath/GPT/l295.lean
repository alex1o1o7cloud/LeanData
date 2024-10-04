import Mathlib

namespace selling_price_of_radio_l295_295917

theorem selling_price_of_radio
  (cost_price : ℝ)
  (loss_percentage : ℝ) :
  loss_percentage = 13 → cost_price = 1500 → 
  (cost_price - (loss_percentage / 100) * cost_price) = 1305 :=
by
  intros h1 h2
  sorry

end selling_price_of_radio_l295_295917


namespace trigonometric_identity_l295_295569

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
    Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = -4 / 3 :=
sorry

end trigonometric_identity_l295_295569


namespace max_side_length_triangle_l295_295213

def triangle_with_max_side_length (a b c : ℕ) (ha : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hper : a + b + c = 30) : Prop :=
  a > b ∧ a > c ∧ a = 14

theorem max_side_length_triangle : ∃ a b c : ℕ, 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 30 ∧ a > b ∧ a > c ∧ a = 14 :=
sorry

end max_side_length_triangle_l295_295213


namespace negation_exists_or_l295_295313

theorem negation_exists_or (x : ℝ) :
  ¬ (∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 :=
by sorry

end negation_exists_or_l295_295313


namespace restore_fractions_l295_295512

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l295_295512


namespace maximum_side_length_of_triangle_l295_295203

theorem maximum_side_length_of_triangle (a b c : ℕ) (h_diff: a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_perimeter: a + b + c = 30)
  (h_triangle_inequality_1: a + b > c) 
  (h_triangle_inequality_2: a + c > b) 
  (h_triangle_inequality_3: b + c > a) : 
  c ≤ 14 :=
sorry

end maximum_side_length_of_triangle_l295_295203


namespace probability_both_selected_l295_295646

theorem probability_both_selected 
  (p_jamie : ℚ) (p_tom : ℚ) 
  (h1 : p_jamie = 2/3) 
  (h2 : p_tom = 5/7) : 
  (p_jamie * p_tom = 10/21) :=
by
  sorry

end probability_both_selected_l295_295646


namespace option_d_is_quadratic_equation_l295_295938

theorem option_d_is_quadratic_equation (x y : ℝ) : 
  (x^2 + x - 4 = 0) ↔ (x^2 + x = 4) := 
by
  sorry

end option_d_is_quadratic_equation_l295_295938


namespace geometric_sequence_a3_l295_295733

theorem geometric_sequence_a3 (
  a : ℕ → ℝ
) 
(h1 : a 1 = 1)
(h5 : a 5 = 16)
(h_geometric : ∀ (n : ℕ), a (n + 1) / a n = a 2 / a 1) :
a 3 = 4 := by
  sorry

end geometric_sequence_a3_l295_295733


namespace probability_of_hitting_10_or_9_probability_of_hitting_at_least_7_probability_of_hitting_less_than_8_l295_295820

-- Definitions of the probabilities
def P_A := 0.24
def P_B := 0.28
def P_C := 0.19
def P_D := 0.16
def P_E := 0.13

-- Prove that the probability of hitting the 10 or 9 rings is 0.52
theorem probability_of_hitting_10_or_9 : P_A + P_B = 0.52 :=
  by sorry

-- Prove that the probability of hitting at least the 7 ring is 0.87
theorem probability_of_hitting_at_least_7 : P_A + P_B + P_C + P_D = 0.87 :=
  by sorry

-- Prove that the probability of hitting less than 8 rings is 0.29
theorem probability_of_hitting_less_than_8 : P_D + P_E = 0.29 :=
  by sorry

end probability_of_hitting_10_or_9_probability_of_hitting_at_least_7_probability_of_hitting_less_than_8_l295_295820


namespace problem1_problem2_l295_295834

-- Problem 1
theorem problem1 : ((2 / 3 - 1 / 12 - 1 / 15) * -60) = -31 := by
  sorry

-- Problem 2
theorem problem2 : ((-7 / 8) / ((7 / 4) - 7 / 8 - 7 / 12)) = -3 := by
  sorry

end problem1_problem2_l295_295834


namespace hyperbola_focal_length_l295_295709

theorem hyperbola_focal_length (x y : ℝ) : 
  (∃ h : x^2 / 9 - y^2 / 4 = 1, 
   ∀ a b : ℝ, a^2 = 9 → b^2 = 4 → 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 13) :=
by sorry

end hyperbola_focal_length_l295_295709


namespace mean_of_remaining_two_l295_295982

def seven_numbers := [1865, 1990, 2015, 2023, 2105, 2120, 2135]

def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem mean_of_remaining_two
  (h : mean (seven_numbers.take 5) = 2043) :
  mean (seven_numbers.drop 5) = 969 :=
by
  sorry

end mean_of_remaining_two_l295_295982


namespace find_A_l295_295068

variable (x ω φ b A : ℝ)

-- Given conditions
axiom cos_squared_eq : 2 * (Real.cos (x + Real.sin (2 * x)))^2 = A * Real.sin (ω * x + φ) + b
axiom A_gt_zero : A > 0

-- Lean 4 statement to prove
theorem find_A : A = Real.sqrt 2 :=
by
  sorry

end find_A_l295_295068


namespace odd_log_function_eval_at_neg_four_l295_295383

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then log 2 x else -log 2 (-x)

theorem odd_log_function_eval_at_neg_four :
  (f (-4) = -2) :=
by
sorry

end odd_log_function_eval_at_neg_four_l295_295383


namespace quadratic_rewrite_sum_l295_295777

theorem quadratic_rewrite_sum :
  let a : ℝ := -3
  let b : ℝ := 9 / 2
  let c : ℝ := 567 / 4
  a + b + c = 143.25 :=
by 
  let a : ℝ := -3
  let b : ℝ := 9 / 2
  let c : ℝ := 567 / 4
  sorry

end quadratic_rewrite_sum_l295_295777


namespace find_a_l295_295591

theorem find_a (a : ℝ) (x₁ x₂ : ℝ) :
  (2 * x₁ + 1 = 3) →
  (2 - (a - x₂) / 3 = 1) →
  (x₁ = x₂) →
  a = 4 :=
by
  intros h₁ h₂ h₃
  sorry

end find_a_l295_295591


namespace arithmetic_sequence_multiples_l295_295773

theorem arithmetic_sequence_multiples (a1 a8 : ℤ) (n : ℕ) (f : ℤ → ℤ) (d : ℤ) :
  a1 = 9 →
  a8 = 12 →
  ∀ n, f n = a1 + (n - 1) * d →
  ∃ k, ∀ m, (1 ≤ m ∧ m ≤ 2015) → f m = 3 * k ∧ k ≥ 0 ∧ k ≤ 287 →
  count_multiples_3 (first_2015_terms (f)) = 288 :=
by
  sorry

end arithmetic_sequence_multiples_l295_295773


namespace intersection_M_N_l295_295391

open Set

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = 2 * x + 1}

theorem intersection_M_N : M ∩ N = {-1, 1} :=
by
  sorry

end intersection_M_N_l295_295391


namespace geometric_sequence_increasing_condition_l295_295721

noncomputable def is_geometric (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (h_geo : is_geometric a) (h_cond : a 0 < a 1 ∧ a 1 < a 2) :
  ¬(∀ n : ℕ, a n < a (n + 1)) → (a 0 < a 1 ∧ a 1 < a 2) :=
sorry

end geometric_sequence_increasing_condition_l295_295721


namespace lucy_packs_of_cake_l295_295901

theorem lucy_packs_of_cake (total_groceries cookies : ℕ) (h1 : total_groceries = 27) (h2 : cookies = 23) :
  total_groceries - cookies = 4 :=
by
  -- In Lean, we would provide the actual proof here, but we'll use sorry to skip the proof as instructed
  sorry

end lucy_packs_of_cake_l295_295901


namespace points_per_other_player_l295_295368

-- Define the conditions as variables
variables (total_points : ℕ) (faye_points : ℕ) (total_players : ℕ)

-- Assume the given conditions
def conditions : Prop :=
  total_points = 68 ∧ faye_points = 28 ∧ total_players = 5

-- Define the proof problem: Prove that the points scored by each of the other players is 10
theorem points_per_other_player :
  conditions total_points faye_points total_players →
  (total_points - faye_points) / (total_players - 1) = 10 :=
by
  sorry

end points_per_other_player_l295_295368


namespace monotonic_has_at_most_one_solution_l295_295079

def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y ∨ f y ≤ f x

theorem monotonic_has_at_most_one_solution (f : ℝ → ℝ) (c : ℝ) 
  (hf : monotonic f) : ∃! x : ℝ, f x = c :=
sorry

end monotonic_has_at_most_one_solution_l295_295079


namespace min_value_frac_l295_295984

theorem min_value_frac (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  ∃ (min : ℝ), min = 9 / 2 ∧ (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 2 → 4 / x + 1 / y ≥ min) :=
by
  sorry

end min_value_frac_l295_295984


namespace chemistry_more_than_physics_l295_295128

variables (M P C x : ℤ)

-- Condition 1: The total marks in mathematics and physics is 50
def condition1 : Prop := M + P = 50

-- Condition 2: The average marks in mathematics and chemistry together is 35
def condition2 : Prop := (M + C) / 2 = 35

-- Condition 3: The score in chemistry is some marks more than that in physics
def condition3 : Prop := C = P + x

theorem chemistry_more_than_physics :
  condition1 M P ∧ condition2 M C ∧ (∃ x : ℤ, condition3 P C x ∧ x = 20) :=
sorry

end chemistry_more_than_physics_l295_295128


namespace marty_combinations_l295_295746

theorem marty_combinations : 
  let C := 5
  let P := 4
  C * P = 20 :=
by
  sorry

end marty_combinations_l295_295746


namespace max_side_of_triangle_with_perimeter_30_l295_295207

theorem max_side_of_triangle_with_perimeter_30 
  (a b c : ℕ) 
  (h1 : a + b + c = 30) 
  (h2 : a ≥ b) 
  (h3 : b ≥ c) 
  (h4 : a < b + c) 
  (h5 : b < a + c) 
  (h6 : c < a + b) 
  : a ≤ 14 :=
sorry

end max_side_of_triangle_with_perimeter_30_l295_295207


namespace largest_four_digit_number_l295_295457

theorem largest_four_digit_number
  (n : ℕ) (hn1 : n % 8 = 2) (hn2 : n % 7 = 4) (hn3 : 1000 ≤ n) (hn4 : n ≤ 9999) :
  n = 9990 :=
sorry

end largest_four_digit_number_l295_295457


namespace original_amount_of_money_l295_295431

variable (took : ℕ) (now : ℕ) (initial : ℕ)

-- conditions from the problem
def conditions := (took = 2) ∧ (now = 3)

-- the statement to prove
theorem original_amount_of_money {took now initial : ℕ} (h : conditions took now) :
  initial = now + took ↔ initial = 5 :=
by {
  sorry
}

end original_amount_of_money_l295_295431


namespace restore_fractions_l295_295511

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l295_295511


namespace percentage_increase_third_year_l295_295067

theorem percentage_increase_third_year
  (initial_price : ℝ)
  (price_2007 : ℝ := initial_price * (1 + 20 / 100))
  (price_2008 : ℝ := price_2007 * (1 - 25 / 100))
  (price_end_third_year : ℝ := initial_price * (108 / 100)) :
  ((price_end_third_year - price_2008) / price_2008) * 100 = 20 :=
by
  sorry

end percentage_increase_third_year_l295_295067


namespace simplify_eq_neg_one_l295_295897

variable (a b c : ℝ)

noncomputable def simplify_expression := 
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)

theorem simplify_eq_neg_one 
  (a_ne_zero : a ≠ 0) 
  (b_ne_zero : b ≠ 0) 
  (c_ne_zero : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) 
  : simplify_expression a b c = -1 :=
by sorry

end simplify_eq_neg_one_l295_295897


namespace solve_fractions_l295_295487

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l295_295487


namespace total_workers_construction_l295_295824

def number_of_monkeys : Nat := 239
def number_of_termites : Nat := 622
def total_workers (m : Nat) (t : Nat) : Nat := m + t

theorem total_workers_construction : total_workers number_of_monkeys number_of_termites = 861 := by
  sorry

end total_workers_construction_l295_295824


namespace cost_per_gift_l295_295255

theorem cost_per_gift (a b c : ℕ) (hc : c = 70) (ha : a = 3) (hb : b = 4) :
  c / (a + b) = 10 :=
by sorry

end cost_per_gift_l295_295255


namespace rectangle_area_l295_295821

theorem rectangle_area (sqr_area : ℕ) (rect_width rect_length : ℕ) (h1 : sqr_area = 25)
    (h2 : rect_width = Int.sqrt sqr_area) (h3 : rect_length = 2 * rect_width) :
    rect_width * rect_length = 50 := by
  sorry

end rectangle_area_l295_295821


namespace Lara_age_in_10_years_l295_295412

theorem Lara_age_in_10_years (current_age: ℕ) (years_ago: ℕ) (years_from_now: ℕ) (age_years_ago: ℕ) (h1: current_age = age_years_ago + years_ago) (h2: age_years_ago = 9) (h3: years_ago = 7) (h4: years_from_now = 10) : current_age + years_from_now = 26 := 
by 
  rw [h2, h3] at h1
  rw [← h1, h4]
  exact rfl

end Lara_age_in_10_years_l295_295412


namespace running_speed_l295_295795

theorem running_speed
  (walking_speed : Float)
  (walking_time : Float)
  (running_time : Float)
  (distance : Float) :
  walking_speed = 8 → walking_time = 3 → running_time = 1.5 → distance = walking_speed * walking_time → 
  (distance / running_time) = 16 :=
by
  intros h_walking_speed h_walking_time h_running_time h_distance
  sorry

end running_speed_l295_295795


namespace percents_multiplication_l295_295010

theorem percents_multiplication :
  let p1 := 0.40
  let p2 := 0.35
  let p3 := 0.60
  let p4 := 0.70
  (p1 * p2 * p3 * p4) * 100 = 5.88 := 
by
  let p1 := 0.40
  let p2 := 0.35
  let p3 := 0.60
  let p4 := 0.70
  sorry

end percents_multiplication_l295_295010


namespace probability_sum_greater_than_six_l295_295595

variable (A : Finset ℕ) (B : Finset ℕ)
variable (balls_in_A : A = {1, 2}) (balls_in_B : B = {3, 4, 5, 6})

theorem probability_sum_greater_than_six : 
  (∃ selected_pair ∈ (A.product B), selected_pair.1 + selected_pair.2 > 6) →
  (Finset.filter (λ pair => pair.1 + pair.2 > 6) (A.product B)).card / 
  (A.product B).card = 3 / 8 := sorry

end probability_sum_greater_than_six_l295_295595


namespace triangle_base_length_l295_295767

theorem triangle_base_length (A h b : ℝ) 
  (h1 : A = 30) 
  (h2 : h = 5) 
  (h3 : A = (b * h) / 2) : 
  b = 12 :=
by
  sorry

end triangle_base_length_l295_295767


namespace cone_ratio_l295_295346

noncomputable def cone_height_ratio : ℚ :=
  let original_height := 40
  let circumference := 24 * Real.pi
  let original_radius := 12
  let new_volume := 432 * Real.pi
  let new_height := 9
  new_height / original_height

theorem cone_ratio (h : cone_height_ratio = 9 / 40) : (9 : ℚ) / 40 = 9 / 40 := by
  sorry

end cone_ratio_l295_295346


namespace b_in_terms_of_a_l295_295606

noncomputable def a (k : ℝ) : ℝ := 3 + 3^k
noncomputable def b (k : ℝ) : ℝ := 3 + 3^(-k)

theorem b_in_terms_of_a (k : ℝ) :
  b k = (3 * (a k) - 8) / ((a k) - 3) := 
sorry

end b_in_terms_of_a_l295_295606


namespace max_triangle_side_length_l295_295210

theorem max_triangle_side_length:
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a + b + c = 30 ∧
    a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 14 :=
  sorry

end max_triangle_side_length_l295_295210


namespace cone_volume_l295_295385

theorem cone_volume (r h l : ℝ) (π := Real.pi)
  (slant_height : l = 5)
  (lateral_area : π * r * l = 20 * π) :
  (1 / 3) * π * r^2 * h = 16 * π :=
by
  -- Definitions based on conditions
  let slant_height_definition := slant_height
  let lateral_area_definition := lateral_area
  
  -- Need actual proof steps which are omitted using sorry
  sorry

end cone_volume_l295_295385


namespace valid_n_value_l295_295243

theorem valid_n_value (n : ℕ) (a : ℕ → ℕ)
    (h1 : ∀ k : ℕ, 1 ≤ k ∧ k < n → k ∣ a k)
    (h2 : ¬ n ∣ a n)
    (h3 : 2 ≤ n) :
    ∃ (p : ℕ) (α : ℕ), (Nat.Prime p) ∧ (n = p ^ α) ∧ (α ≥ 1) :=
by sorry

end valid_n_value_l295_295243


namespace mixed_fraction_product_l295_295522

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l295_295522


namespace additional_cars_needed_to_make_multiple_of_8_l295_295053

theorem additional_cars_needed_to_make_multiple_of_8 (current_cars : ℕ) (rows_of_cars : ℕ) (next_multiple : ℕ)
  (h1 : current_cars = 37)
  (h2 : rows_of_cars = 8)
  (h3 : next_multiple = 40)
  (h4 : next_multiple ≥ current_cars)
  (h5 : next_multiple % rows_of_cars = 0) :
  (next_multiple - current_cars) = 3 :=
by { sorry }

end additional_cars_needed_to_make_multiple_of_8_l295_295053


namespace mixed_fractions_product_l295_295503

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l295_295503


namespace b_should_pay_360_l295_295803

theorem b_should_pay_360 :
  let total_cost : ℝ := 870
  let a_horses  : ℝ := 12
  let a_months  : ℝ := 8
  let b_horses  : ℝ := 16
  let b_months  : ℝ := 9
  let c_horses  : ℝ := 18
  let c_months  : ℝ := 6
  let a_horse_months := a_horses * a_months
  let b_horse_months := b_horses * b_months
  let c_horse_months := c_horses * c_months
  let total_horse_months := a_horse_months + b_horse_months + c_horse_months
  let cost_per_horse_month := total_cost / total_horse_months
  let b_cost := b_horse_months * cost_per_horse_month
  b_cost = 360 :=
by sorry

end b_should_pay_360_l295_295803


namespace multiplication_72515_9999_l295_295551

theorem multiplication_72515_9999 : 72515 * 9999 = 725077485 :=
by
  sorry

end multiplication_72515_9999_l295_295551


namespace max_tickets_l295_295703

theorem max_tickets (ticket_price normal_discounted_price budget : ℕ) (h1 : ticket_price = 15) (h2 : normal_discounted_price = 13) (h3 : budget = 180) :
  ∃ n : ℕ, ((n ≤ 10 → ticket_price * n ≤ budget) ∧ (n > 10 → normal_discounted_price * n ≤ budget)) ∧ ∀ m : ℕ, ((m ≤ 10 → ticket_price * m ≤ budget) ∧ (m > 10 → normal_discounted_price * m ≤ budget)) → m ≤ 13 :=
by
  sorry

end max_tickets_l295_295703


namespace negate_universal_proposition_l295_295633

theorem negate_universal_proposition : 
  (¬ (∀ x : ℝ, x^2 - 2 * x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) :=
by sorry

end negate_universal_proposition_l295_295633


namespace sandy_change_correct_l295_295302

def football_cost : ℚ := 914 / 100
def baseball_cost : ℚ := 681 / 100
def payment : ℚ := 20

def total_cost : ℚ := football_cost + baseball_cost
def change_received : ℚ := payment - total_cost

theorem sandy_change_correct :
  change_received = 405 / 100 :=
by
  -- The proof should go here
  sorry

end sandy_change_correct_l295_295302


namespace min_sum_of_a_and_b_l295_295794

theorem min_sum_of_a_and_b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > 4 * b) : a + b ≥ 6 :=
by
  sorry

end min_sum_of_a_and_b_l295_295794


namespace sin_390_eq_half_l295_295363

theorem sin_390_eq_half : Real.sin (390 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_390_eq_half_l295_295363


namespace smallest_four_digit_divisible_by_35_l295_295151

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l295_295151


namespace solve_problem_l295_295229

noncomputable def problem_statement : Prop :=
  ∀ (a b c : ℕ),
    (a ≤ b) →
    (b ≤ c) →
    Nat.gcd (Nat.gcd a b) c = 1 →
    (a^2 * b) ∣ (a^3 + b^3 + c^3) →
    (b^2 * c) ∣ (a^3 + b^3 + c^3) →
    (c^2 * a) ∣ (a^3 + b^3 + c^3) →
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 2 ∧ c = 3)

-- Here we declare the main theorem but skip the proof.
theorem solve_problem : problem_statement :=
by sorry

end solve_problem_l295_295229


namespace standard_deviation_distance_l295_295438

-- Definitions and assumptions based on the identified conditions
def mean : ℝ := 12
def std_dev : ℝ := 1.2
def value : ℝ := 9.6

-- Statement to prove
theorem standard_deviation_distance : (value - mean) / std_dev = -2 :=
by sorry

end standard_deviation_distance_l295_295438


namespace ron_tickets_sold_l295_295757

theorem ron_tickets_sold 
  (R K : ℕ) 
  (h1 : R + K = 20) 
  (h2 : 2 * R + 9 / 2 * K = 60) : 
  R = 12 := 
by 
  sorry

end ron_tickets_sold_l295_295757


namespace interest_calculation_years_l295_295590

theorem interest_calculation_years (P n : ℝ) (r : ℝ) (SI CI : ℝ)
  (h₁ : SI = P * r * n / 100)
  (h₂ : r = 5)
  (h₃ : SI = 50)
  (h₄ : CI = P * ((1 + r / 100)^n - 1))
  (h₅ : CI = 51.25) :
  n = 2 := by
  sorry

end interest_calculation_years_l295_295590


namespace min_value_circles_tangents_l295_295138

theorem min_value_circles_tangents (a b : ℝ) (h1 : (∃ x y : ℝ, x^2 + y^2 + 2 * a * x + a^2 - 4 = 0) ∧ 
  (∃ x y : ℝ, x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0))
  (h2 : ∃ k : ℕ, k = 3) (h3 : a ≠ 0) (h4 : b ≠ 0) : 
  (∃ m : ℝ, m = 1 ∧  ∀ x : ℝ, (x = (1 / a^2) + (1 / b^2)) → x ≥ m) :=
  sorry

end min_value_circles_tangents_l295_295138


namespace selection_count_l295_295377

noncomputable def choose (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_count :
  let boys := 4
  let girls := 3
  let total := boys + girls
  let choose_boys_girls : ℕ := (choose 4 2) * (choose 3 1) + (choose 4 1) * (choose 3 2)
  choose_boys_girls = 30 := 
by
  sorry

end selection_count_l295_295377


namespace area_PTR_l295_295883

-- Define points P, Q, R, S, and T
variables (P Q R S T : Type)

-- Assume QR is divided by points S and T in the given ratio
variables (QS ST TR : ℕ)
axiom ratio_condition : QS = 2 ∧ ST = 5 ∧ TR = 3

-- Assume the area of triangle PQS is given as 60 square centimeters
axiom area_PQS : ℕ
axiom area_PQS_value : area_PQS = 60

-- State the problem
theorem area_PTR : ∃ (area_PTR : ℕ), area_PTR = 90 :=
by
  sorry

end area_PTR_l295_295883


namespace sum_of_relatively_prime_integers_l295_295430

theorem sum_of_relatively_prime_integers (n : ℕ) (h : n ≥ 7) :
  ∃ a b : ℕ, n = a + b ∧ a > 1 ∧ b > 1 ∧ Nat.gcd a b = 1 :=
by
  sorry

end sum_of_relatively_prime_integers_l295_295430


namespace common_difference_of_arithmetic_seq_l295_295707

variable (a_1 d : ℤ) (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Condition 1: definition of general term in an arithmetic sequence
axiom general_term_arith_sequence (n : ℕ) : a n = a_1 + (n - 1) * d

-- Condition 2: sum of the first n terms of the arithmetic sequence
axiom sum_first_n_arith_sequence (n : ℕ) : S n = n * (2 * a_1 + (n - 1) * d) / 2

-- Condition 3: given condition S_4 = 3 * S_2
axiom S4_eq_3S2 : S 4 = 3 * S 2

-- Condition 4: given condition a_7 = 15
axiom a7_eq_15 : a 7 = 15

-- Goal: prove that the common difference d is 2
theorem common_difference_of_arithmetic_seq : d = 2 := by
  sorry

end common_difference_of_arithmetic_seq_l295_295707


namespace restore_fractions_l295_295514

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l295_295514


namespace elevator_height_after_20_seconds_l295_295400

-- Conditions
def starting_height : ℕ := 120
def descending_speed : ℕ := 4
def time_elapsed : ℕ := 20

-- Statement to prove
theorem elevator_height_after_20_seconds : 
  starting_height - descending_speed * time_elapsed = 40 := 
by 
  sorry

end elevator_height_after_20_seconds_l295_295400


namespace area_of_quadrilateral_l295_295851

theorem area_of_quadrilateral (d h1 h2 : ℝ) (h1_pos : h1 = 9) (h2_pos : h2 = 6) (d_pos : d = 30) : 
  let area1 := (1/2 : ℝ) * d * h1
  let area2 := (1/2 : ℝ) * d * h2
  (area1 + area2) = 225 :=
by
  sorry

end area_of_quadrilateral_l295_295851


namespace mixed_fractions_product_l295_295508

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l295_295508


namespace decreasing_on_transformed_interval_l295_295384

theorem decreasing_on_transformed_interval
  (f : ℝ → ℝ)
  (h : ∀ ⦃x₁ x₂ : ℝ⦄, 1 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 2 → f x₁ ≤ f x₂) :
  ∀ ⦃x₁ x₂ : ℝ⦄, -1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 0 → f (1 - x₂) ≤ f (1 - x₁) :=
sorry

end decreasing_on_transformed_interval_l295_295384


namespace carol_invitations_l295_295045

-- Definitions: each package has 3 invitations, Carol bought 2 packs, and Carol needs 3 extra invitations.
def invitations_per_pack : ℕ := 3
def packs_bought : ℕ := 2
def extra_invitations : ℕ := 3

-- Total number of invitations Carol will have
def total_invitations : ℕ := (packs_bought * invitations_per_pack) + extra_invitations

-- Statement to prove: Carol wants to invite 9 friends.
theorem carol_invitations : total_invitations = 9 := by
  sorry  -- Proof omitted

end carol_invitations_l295_295045


namespace hari_joins_l295_295618

theorem hari_joins {x : ℕ} :
  let praveen_start := 3500
  let hari_start := 9000
  let total_months := 12
  (praveen_start * total_months) * 3 = (hari_start * (total_months - x)) * 2
  → x = 5 :=
by
  intros
  sorry

end hari_joins_l295_295618


namespace average_of_values_l295_295699

theorem average_of_values (z : ℝ) : 
  (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end average_of_values_l295_295699


namespace additional_bags_at_max_weight_l295_295792

/-
Constants representing the problem conditions.
-/
def num_people : Nat := 6
def bags_per_person : Nat := 5
def max_weight_per_bag : Nat := 50
def total_weight_capacity : Nat := 6000

/-
Calculate the total existing luggage weight.
-/
def total_existing_bags : Nat := num_people * bags_per_person
def total_existing_weight : Nat := total_existing_bags * max_weight_per_bag
def remaining_weight_capacity : Nat := total_weight_capacity - total_existing_weight

/-
The proof statement asserting that given the conditions, 
the airplane can hold 90 more bags at maximum weight.
-/
theorem additional_bags_at_max_weight : remaining_weight_capacity / max_weight_per_bag = 90 := by
  sorry

end additional_bags_at_max_weight_l295_295792


namespace min_value_quadratic_l295_295545

theorem min_value_quadratic :
  ∃ (x y : ℝ), (∀ (a b : ℝ), (3*a^2 + 4*a*b + 2*b^2 - 6*a - 8*b + 6 ≥ 0)) ∧ 
  (3*x^2 + 4*x*y + 2*y^2 - 6*x - 8*y + 6 = 0) := 
sorry

end min_value_quadratic_l295_295545


namespace sum_odd_multiples_of_5_from_1_to_60_l295_295808

def isOdd (n : ℤ) : Prop := ¬ Even n

def isMultipleOf5 (n : ℤ) : Prop := ∃ k : ℤ, n = 5 * k

def oddMultiplesOf5 (n : ℤ) : Prop := isOdd n ∧ isMultipleOf5 n

def filterOddMultiplesOf5UpTo60 : List ℤ :=
  List.filter oddMultiplesOf5 (List.range' 1 60)

theorem sum_odd_multiples_of_5_from_1_to_60 :
  ∑ n in filterOddMultiplesOf5UpTo60, n = 180 := by
  sorry

end sum_odd_multiples_of_5_from_1_to_60_l295_295808


namespace contradictory_statement_of_p_l295_295055

-- Given proposition p
def p : Prop := ∀ (x : ℝ), x + 3 ≥ 0 → x ≥ -3

-- Contradictory statement of p
noncomputable def contradictory_p : Prop := ∀ (x : ℝ), x + 3 < 0 → x < -3

-- Proof statement
theorem contradictory_statement_of_p : contradictory_p :=
sorry

end contradictory_statement_of_p_l295_295055


namespace restore_original_problem_l295_295501

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l295_295501


namespace simplify_complex_expression_l295_295434

open Complex

theorem simplify_complex_expression : (1 + 2 * I) / I = -2 + I :=
by
  sorry

end simplify_complex_expression_l295_295434


namespace sum_of_first_five_terms_l295_295099

theorem sum_of_first_five_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q) -- geometric sequence definition
  (h3 : a 2 + a 5 = 2 * (a 4 + 2)) : 
  S 5 = 62 :=
by
  -- lean tactics would go here to provide the proof
  sorry

end sum_of_first_five_terms_l295_295099


namespace vanessa_score_record_l295_295139

theorem vanessa_score_record 
  (team_total_points : ℕ) 
  (other_players_average : ℕ) 
  (num_other_players : ℕ) 
  (total_game_points : team_total_points = 55) 
  (average_points_per_player : other_players_average = 4) 
  (number_of_other_players : num_other_players = 7) 
  : 
  ∃ vanessa_points : ℕ, vanessa_points = 27 :=
by
  sorry

end vanessa_score_record_l295_295139


namespace divides_power_of_odd_l295_295418

theorem divides_power_of_odd (k : ℕ) (hk : k % 2 = 1) (n : ℕ) (hn : n ≥ 1) : 2^(n + 2) ∣ (k^(2^n) - 1) :=
by
  sorry

end divides_power_of_odd_l295_295418


namespace restore_original_problem_l295_295492

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l295_295492


namespace complement_union_l295_295877

namespace SetTheory

def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

theorem complement_union (U A B: Set ℕ) (hU : U = {1, 3, 5, 9}) (hA : A = {1, 3, 9}) (hB : B = {1, 9}) :
  U \ (A ∪ B) = {5} :=
by
  sorry

end SetTheory

end complement_union_l295_295877


namespace units_digit_of_p_is_6_l295_295078

-- Given conditions
variable (p : ℕ)
variable (h1 : p % 2 = 0)                -- p is a positive even integer
variable (h2 : (p^3 % 10) - (p^2 % 10) = 0)  -- The units digit of p^3 minus the units digit of p^2 is 0
variable (h3 : (p + 2) % 10 = 8)         -- The units digit of p + 2 is 8

-- Prove the units digit of p is 6
theorem units_digit_of_p_is_6 : p % 10 = 6 :=
sorry

end units_digit_of_p_is_6_l295_295078


namespace num_multiples_of_three_in_ap_l295_295772

variable (a : ℕ → ℚ)  -- Defining the arithmetic sequence

def first_term (a1 : ℚ) := a 1 = a1
def eighth_term (a8 : ℚ) := a 8 = a8
def general_term (d : ℚ) := ∀ n : ℕ, a n = 9 + (n - 1) * d
def multiple_of_three (n : ℕ) := ∃ k : ℕ, a n = 3 * k

theorem num_multiples_of_three_in_ap 
  (a : ℕ → ℚ)
  (h1 : first_term a 9)
  (h2 : eighth_term a 12) :
  ∃ n : ℕ, n = 288 ∧ ∃ l : ℕ → Prop, ∀ k : ℕ, l k → multiple_of_three a (k * 7 + 1) :=
sorry

end num_multiples_of_three_in_ap_l295_295772


namespace spherical_coordinate_conversion_l295_295097

theorem spherical_coordinate_conversion (ρ θ φ : ℝ) 
  (h_ρ : ρ > 0) 
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h_φ : 0 ≤ φ): 
  (ρ, θ, φ - 2 * Real.pi * ⌊φ / (2 * Real.pi)⌋) = (5, 3 * Real.pi / 4, Real.pi / 4) :=
  by 
  sorry

end spherical_coordinate_conversion_l295_295097


namespace domain_of_f_l295_295118

-- Define the function f(x) = 1/(x+1) + ln(x)
noncomputable def f (x : ℝ) : ℝ := (1 / (x + 1)) + Real.log x

-- The domain of the function is all x such that x > 0
theorem domain_of_f :
  ∀ x : ℝ, (x > 0) ↔ (f x = (1 / (x + 1)) + Real.log x) := 
by sorry

end domain_of_f_l295_295118


namespace total_cost_of_stamps_is_correct_l295_295039

-- Define the costs of each type of stamp
def cost_of_stamp_A : ℕ := 34 -- cost in cents
def cost_of_stamp_B : ℕ := 52 -- cost in cents
def cost_of_stamp_C : ℕ := 73 -- cost in cents

-- Define the number of stamps Alice needs to buy
def num_stamp_A : ℕ := 4
def num_stamp_B : ℕ := 6
def num_stamp_C : ℕ := 2

-- Define the expected total cost in dollars
def expected_total_cost : ℝ := 5.94

-- State the theorem about the total cost
theorem total_cost_of_stamps_is_correct :
  ((num_stamp_A * cost_of_stamp_A) + (num_stamp_B * cost_of_stamp_B) + (num_stamp_C * cost_of_stamp_C)) / 100 = expected_total_cost :=
by
  sorry

end total_cost_of_stamps_is_correct_l295_295039


namespace initial_avg_height_l295_295627

-- Lean 4 statement for the given problem
theorem initial_avg_height (A : ℝ) (n : ℕ) (wrong_height correct_height actual_avg init_diff : ℝ)
  (h_class_size : n = 35)
  (h_wrong_height : wrong_height = 166)
  (h_correct_height : correct_height = 106)
  (h_actual_avg : actual_avg = 183)
  (h_init_diff : init_diff = wrong_height - correct_height)
  (h_total_height_actual : n * actual_avg = 35 * 183)
  (h_total_height_wrong : n * A = 35 * actual_avg - init_diff) :
  A = 181 :=
by {
  -- The problem and conditions are correctly stated. The proof is skipped with sorry.
  sorry
}

end initial_avg_height_l295_295627


namespace required_oranges_for_juice_l295_295019

theorem required_oranges_for_juice (oranges quarts : ℚ) (h : oranges = 36 ∧ quarts = 48) :
  ∃ x, ((oranges / quarts) = (x / 6) ∧ x = 4.5) := 
by sorry

end required_oranges_for_juice_l295_295019


namespace trigonometric_identity_l295_295791

theorem trigonometric_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (70 * Real.pi / 180) +
   Real.sin (10 * Real.pi / 180) * Real.sin (50 * Real.pi / 180)) = 1 / 4 :=
by sorry

end trigonometric_identity_l295_295791


namespace parabola_intersection_diff_l295_295843

theorem parabola_intersection_diff (a b c d : ℝ) 
  (h₁ : ∀ x y, (3 * x^2 - 2 * x + 1 = y) → (c = x ∨ a = x))
  (h₂ : ∀ x y, (-2 * x^2 + 4 * x + 1 = y) → (c = x ∨ a = x))
  (h₃ : c ≥ a) :
  c - a = 6 / 5 :=
by sorry

end parabola_intersection_diff_l295_295843


namespace solve_fractions_l295_295482

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l295_295482


namespace range_of_m_three_zeros_l295_295995

noncomputable def f (x m : ℝ) : ℝ :=
if h : x < 0 then -x + m else x^2 - 1

theorem range_of_m_three_zeros (h : 0 < m) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f (f x1 m) m - 1 = 0 ∧ f (f x2 m) m - 1 = 0 ∧ f (f x3 m) m - 1 = 0) ↔ (0 < m ∧ m < 1) :=
by
  sorry

end range_of_m_three_zeros_l295_295995


namespace value_of_ratio_l295_295077

theorem value_of_ratio (x y : ℝ)
    (hx : x > 0)
    (hy : y > 0)
    (h : 2 * x + 3 * y = 8) :
    (2 / x + 3 / y) = 25 / 8 := 
by
  sorry

end value_of_ratio_l295_295077


namespace option_d_is_quadratic_equation_l295_295937

theorem option_d_is_quadratic_equation (x y : ℝ) : 
  (x^2 + x - 4 = 0) ↔ (x^2 + x = 4) := 
by
  sorry

end option_d_is_quadratic_equation_l295_295937


namespace largest_multiple_of_8_less_than_100_l295_295007

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), (n % 8 = 0) ∧ (n < 100) ∧ (∀ m : ℕ, (m % 8 = 0) ∧ (m < 100) → m ≤ n) :=
begin
  use 96,
  split,
  { -- 96 is a multiple of 8
    exact nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 96),
  },
  split,
  { -- 96 is less than 100
    norm_num,
  },
  { -- 96 is the largest multiple of 8 less than 100
    intros m hm,
    obtain ⟨k, rfl⟩ := (nat.dvd_iff_mod_eq_zero.mp hm.1),
    have : k ≤ 12, by linarith,
    linarith [mul_le_mul (zero_le _ : (0 : ℕ) ≤ 8) this (zero_le _ : (0 : ℕ) ≤ 12) (zero_le _ : (0 : ℕ) ≤ 8)],
  },
end

end largest_multiple_of_8_less_than_100_l295_295007


namespace distinct_four_digit_integers_with_digit_product_eight_l295_295584

theorem distinct_four_digit_integers_with_digit_product_eight : 
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (∀ (a b c d : ℕ), 10 > a ∧ 10 > b ∧ 10 > c ∧ 10 > d ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ a * b * c * d = 8) ∧ (∃ (count : ℕ), count = 20 ) :=
sorry

end distinct_four_digit_integers_with_digit_product_eight_l295_295584


namespace min_fraction_value_l295_295120

theorem min_fraction_value
    (a x y : ℕ)
    (h1 : a > 100)
    (h2 : x > 100)
    (h3 : y > 100)
    (h4 : y^2 - 1 = a^2 * (x^2 - 1))
    : a / x ≥ 2 := 
sorry

end min_fraction_value_l295_295120


namespace workshopA_more_stable_than_B_l295_295953

-- Given data sets for workshops A and B
def workshopA_data := [102, 101, 99, 98, 103, 98, 99]
def workshopB_data := [110, 115, 90, 85, 75, 115, 110]

-- Define stability of a product in terms of the standard deviation or similar metric
def is_more_stable (dataA dataB : List ℕ) : Prop :=
  sorry -- Replace with a definition comparing stability based on a chosen metric, e.g., standard deviation

-- Prove that Workshop A's product is more stable than Workshop B's product
theorem workshopA_more_stable_than_B : is_more_stable workshopA_data workshopB_data :=
  sorry

end workshopA_more_stable_than_B_l295_295953


namespace percentage_of_second_solution_correct_l295_295130

noncomputable def percentage_of_alcohol_in_second_solution : ℝ :=
  let total_liters := 80
  let percentage_final_solution := 0.49
  let volume_first_solution := 24
  let percentage_first_solution := 0.4
  let volume_second_solution := 56
  let total_alcohol_in_final_solution := total_liters * percentage_final_solution
  let total_alcohol_first_solution := volume_first_solution * percentage_first_solution
  let x := (total_alcohol_in_final_solution - total_alcohol_first_solution) / volume_second_solution
  x

theorem percentage_of_second_solution_correct : 
  percentage_of_alcohol_in_second_solution = 0.5285714286 := by sorry

end percentage_of_second_solution_correct_l295_295130


namespace express_in_standard_form_l295_295845

theorem express_in_standard_form (x : ℝ) : x^2 - 6 * x = (x - 3)^2 - 9 :=
by
  sorry

end express_in_standard_form_l295_295845


namespace Lara_age_10_years_from_now_l295_295413

theorem Lara_age_10_years_from_now (current_year_age : ℕ) (age_7_years_ago : ℕ)
  (h1 : age_7_years_ago = 9) (h2 : current_year_age = age_7_years_ago + 7) :
  current_year_age + 10 = 26 :=
by
  sorry

end Lara_age_10_years_from_now_l295_295413


namespace find_a_minus_c_l295_295655

theorem find_a_minus_c (a b c : ℝ) (h1 : (a + b) / 2 = 80) (h2 : (b + c) / 2 = 180) : a - c = -200 :=
by 
  sorry

end find_a_minus_c_l295_295655


namespace largest_multiple_of_8_less_than_100_l295_295000

theorem largest_multiple_of_8_less_than_100 : ∃ n, n < 100 ∧ n % 8 = 0 ∧ ∀ m, m < 100 ∧ m % 8 = 0 → m ≤ n :=
begin
  use 96,
  split,
  { -- prove 96 < 100
    norm_num,
  },
  split,
  { -- prove 96 is a multiple of 8
    norm_num,
  },
  { -- prove 96 is the largest such multiple
    intros m hm,
    cases hm with h1 h2,
    have h3 : m / 8 < 100 / 8,
    { exact_mod_cast h1 },
    interval_cases (m / 8) with H,
    all_goals { 
      try { norm_num, exact le_refl _ },
    },
  },
end

end largest_multiple_of_8_less_than_100_l295_295000


namespace cos_half_pi_minus_2alpha_l295_295380

open Real

theorem cos_half_pi_minus_2alpha (α : ℝ) (h : sin α - cos α = 1 / 3) : cos (π / 2 - 2 * α) = 8 / 9 :=
sorry

end cos_half_pi_minus_2alpha_l295_295380


namespace axis_of_symmetry_parabola_l295_295629

theorem axis_of_symmetry_parabola : 
  (∃ a b c : ℝ, ∀ x : ℝ, (y = x^2 + 4 * x - 5) ∧ (a = 1) ∧ (b = 4) → ( x = -b / (2 * a) ) → ( x = -2 ) ) :=
by
  sorry

end axis_of_symmetry_parabola_l295_295629


namespace takeoff_run_length_l295_295126

theorem takeoff_run_length
  (t : ℕ) (h_t : t = 15)
  (v_kmh : ℕ) (h_v : v_kmh = 100)
  (uniform_acc : Prop) :
  ∃ S : ℕ, S = 208 := by
  sorry

end takeoff_run_length_l295_295126


namespace p_squared_plus_one_over_p_squared_plus_six_l295_295396

theorem p_squared_plus_one_over_p_squared_plus_six (p : ℝ) (h : p + 1/p = 10) : p^2 + 1/p^2 + 6 = 104 :=
by {
  sorry
}

end p_squared_plus_one_over_p_squared_plus_six_l295_295396


namespace smallest_four_digit_div_by_35_l295_295145

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l295_295145


namespace local_minimum_at_one_iff_l295_295578

open Real

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x => x^3 - 2*a*x^2 + a^2*x + 1

theorem local_minimum_at_one_iff (a : ℝ) : (∀ f', f' = deriv (f a) → f' 1 = 0 ∧ (forall x, f' x = 3*x^2 - 4*a*x + a^2) → (∀ x, 1 < x → f' x > 0) ∧ (∀ x, x < 1 → f' x < 0)) ↔ a = 1 :=
by
  sorry

end local_minimum_at_one_iff_l295_295578


namespace team_total_points_l295_295729

-- Definitions based on conditions
def chandra_points (akiko_points : ℕ) := 2 * akiko_points
def akiko_points (michiko_points : ℕ) := michiko_points + 4
def michiko_points (bailey_points : ℕ) := bailey_points / 2
def bailey_points := 14

-- Total points scored by the team
def total_points :=
  let michiko := michiko_points bailey_points
  let akiko := akiko_points michiko
  let chandra := chandra_points akiko
  bailey_points + michiko + akiko + chandra

theorem team_total_points : total_points = 54 := by
  sorry

end team_total_points_l295_295729


namespace restore_original_problem_l295_295488

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l295_295488


namespace geom_seq_general_term_sum_geometric_arithmetic_l295_295570

noncomputable def a_n (n : ℕ) : ℕ := 2^n
def b_n (n : ℕ) : ℕ := 2*n - 1

theorem geom_seq_general_term (a : ℕ → ℕ) (a1 : a 1 = 2)
  (a2 : a 3 = (a 2) + 4) : ∀ n, a n = a_n n :=
by
  sorry

theorem sum_geometric_arithmetic (a b : ℕ → ℕ) 
  (a_def : ∀ n, a n = 2 ^ n) (b_def : ∀ n, b n = 2 * n - 1) : 
  ∀ n, (Finset.range n).sum (λ i => (a (i + 1) + b (i + 1))) = 2^(n+1) + n^2 - 2 :=
by
  sorry

end geom_seq_general_term_sum_geometric_arithmetic_l295_295570


namespace largest_repeating_number_l295_295268

theorem largest_repeating_number :
  ∃ n, n * 365 = 273863 * 365 := sorry

end largest_repeating_number_l295_295268


namespace parabola_points_relationship_l295_295710

theorem parabola_points_relationship (c y1 y2 y3 : ℝ)
  (h1 : y1 = -0^2 + 2 * 0 + c)
  (h2 : y2 = -1^2 + 2 * 1 + c)
  (h3 : y3 = -3^2 + 2 * 3 + c) :
  y2 > y1 ∧ y1 > y3 := by
  sorry

end parabola_points_relationship_l295_295710


namespace total_loss_is_correct_l295_295288

-- Definitions for each item's purchase conditions
def paintings_cost : ℕ := 18 * 75
def toys_cost : ℕ := 25 * 30
def hats_cost : ℕ := 12 * 20
def wallets_cost : ℕ := 10 * 50
def mugs_cost : ℕ := 35 * 10

def paintings_loss_percentage : ℝ := 0.22
def toys_loss_percentage : ℝ := 0.27
def hats_loss_percentage : ℝ := 0.15
def wallets_loss_percentage : ℝ := 0.05
def mugs_loss_percentage : ℝ := 0.12

-- Calculation of loss on each item
def paintings_loss : ℝ := paintings_cost * paintings_loss_percentage
def toys_loss : ℝ := toys_cost * toys_loss_percentage
def hats_loss : ℝ := hats_cost * hats_loss_percentage
def wallets_loss : ℝ := wallets_cost * wallets_loss_percentage
def mugs_loss : ℝ := mugs_cost * mugs_loss_percentage

-- Total loss calculation
def total_loss : ℝ := paintings_loss + toys_loss + hats_loss + wallets_loss + mugs_loss

-- Lean statement to verify the total loss
theorem total_loss_is_correct : total_loss = 602.50 := by
  sorry

end total_loss_is_correct_l295_295288


namespace device_failure_probability_l295_295022

noncomputable def probability_fail_device (p1 p2 p3 : ℝ) (p_one p_two p_three : ℝ) : ℝ :=
  0.006 * p3 + 0.092 * p_two + 0.398 * p_one

theorem device_failure_probability
  (p1 p2 p3 : ℝ) (p_one p_two p_three : ℝ)
  (h1 : p1 = 0.1)
  (h2 : p2 = 0.2)
  (h3 : p3 = 0.3)
  (h4 : p_one = 0.25)
  (h5 : p_two = 0.6)
  (h6 : p_three = 0.9) :
  probability_fail_device p1 p2 p3 p_one p_two p_three = 0.1601 :=
by
  sorry

end device_failure_probability_l295_295022


namespace tan_alpha_tan_beta_value_l295_295089

theorem tan_alpha_tan_beta_value
  (α β : ℝ)
  (h1 : Real.cos (α + β) = 1 / 5)
  (h2 : Real.cos (α - β) = 3 / 5) :
  Real.tan α * Real.tan β = 1 / 2 :=
by
  sorry

end tan_alpha_tan_beta_value_l295_295089


namespace correct_statement_is_B_l295_295012

def coefficient_of_x : Int := 1
def is_monomial (t : String) : Bool := t = "1x^0"
def coefficient_of_neg_3x : Int := -3
def degree_of_5x2y : Int := 3

theorem correct_statement_is_B :
  (coefficient_of_x = 0) = false ∧ 
  (is_monomial "1x^0" = true) ∧ 
  (coefficient_of_neg_3x = 3) = false ∧ 
  (degree_of_5x2y = 2) = false ∧ 
  (B = "1 is a monomial") :=
by {
  sorry
}

end correct_statement_is_B_l295_295012


namespace ratio_of_wire_pieces_l295_295672

theorem ratio_of_wire_pieces (a b : ℝ) (h_equal_areas : (a / 4) ^ 2 = 2 * (1 + real.sqrt 2) * (b / 8) ^ 2) :
  a / b = real.sqrt (2 + real.sqrt 2) / 2 := 
by
  sorry

end ratio_of_wire_pieces_l295_295672


namespace problem_solution_l295_295094

-- Definitions for given conditions
variables {a_n b_n : ℕ → ℝ} -- Sequences {a_n} and {b_n}
variables {S T : ℕ → ℝ} -- Sums of the first n terms of {a_n} and {b_n}
variables (h1 : ∀ n, S n = (n * (a_n 1 + a_n n)) / 2)
variables (h2 : ∀ n, T n = (n * (b_n 1 + b_n n)) / 2)
variables (h3 : ∀ n, n > 0 → S n / T n = (2 * n + 1) / (n + 2))

-- The goal
theorem problem_solution :
  (a_n 7) / (b_n 7) = 9 / 5 :=
sorry

end problem_solution_l295_295094


namespace minimize_function_l295_295701

noncomputable def f (x : ℝ) : ℝ := x - 4 + 9 / (x + 1)

theorem minimize_function : 
  (∀ x : ℝ, x > -1 → f x ≥ 1) ∧ (f 2 = 1) :=
by 
  sorry

end minimize_function_l295_295701


namespace combined_salaries_BCDE_l295_295635

-- Define the given conditions
def salary_A : ℕ := 10000
def average_salary : ℕ := 8400
def num_individuals : ℕ := 5

-- Define the total salary of all individuals
def total_salary_all : ℕ := average_salary * num_individuals

-- Define the proof problem
theorem combined_salaries_BCDE : (total_salary_all - salary_A) = 32000 := by
  sorry

end combined_salaries_BCDE_l295_295635


namespace median_of_36_consecutive_integers_l295_295781

theorem median_of_36_consecutive_integers (f : ℕ → ℤ) (h_consecutive : ∀ n : ℕ, f (n + 1) = f n + 1) 
(h_size : ∃ k, f 36 = f 0 + 35) (h_sum : ∑ i in finset.range 36, f i = 6^4) : 
(∃ m, m = f (36 / 2 - 1) ∧ m = 36) :=
by
  sorry

end median_of_36_consecutive_integers_l295_295781


namespace nina_spends_70_l295_295747

-- Definitions of the quantities and prices
def toys := 3
def toy_price := 10
def basketball_cards := 2
def card_price := 5
def shirts := 5
def shirt_price := 6

-- Calculate the total amount spent
def total_spent := (toys * toy_price) + (basketball_cards * card_price) + (shirts * shirt_price)

-- Problem statement: Prove that the total amount spent is $70
theorem nina_spends_70 : total_spent = 70 := by
  sorry

end nina_spends_70_l295_295747


namespace restore_original_problem_l295_295494

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l295_295494


namespace certain_number_correct_l295_295263

theorem certain_number_correct : 
  (h1 : 29.94 / 1.45 = 17.9) -> (2994 / 14.5 = 1790) :=
by 
  sorry

end certain_number_correct_l295_295263


namespace rectangular_plot_area_l295_295631

theorem rectangular_plot_area (Breadth Length Area : ℕ): 
  (Length = 3 * Breadth) → 
  (Breadth = 30) → 
  (Area = Length * Breadth) → 
  Area = 2700 :=
by 
  intros h_length h_breadth h_area
  rw [h_breadth] at h_length
  rw [h_length, h_breadth] at h_area
  exact h_area

end rectangular_plot_area_l295_295631


namespace count_routes_from_A_to_B_l295_295836

-- Define cities as an inductive type
inductive City
| A
| B
| C
| D
| E

-- Define roads as a list of pairs of cities
def roads : List (City × City) := [
  (City.A, City.B),
  (City.A, City.D),
  (City.B, City.D),
  (City.C, City.D),
  (City.D, City.E),
  (City.B, City.E)
]

-- Define the problem statement
noncomputable def route_count : ℕ :=
  3  -- This should be proven

theorem count_routes_from_A_to_B : route_count = 3 :=
  by
    sorry  -- Proof goes here

end count_routes_from_A_to_B_l295_295836


namespace find_range_t_l295_295084

noncomputable def f (x t : ℝ) : ℝ :=
  if x < t then -6 + Real.exp (x - 1) else x^2 - 4 * x

theorem find_range_t (f : ℝ → ℝ → ℝ)
  (h : ∀ t : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ t = x₁ - 6 ∧ f x₂ t = x₂ - 6 ∧ f x₃ t = x₃ - 6)) :
  ∀ t : ℝ, 1 < t ∧ t ≤ 2 := sorry

end find_range_t_l295_295084


namespace distinct_intersection_points_l295_295546

theorem distinct_intersection_points :
  let S1 := { p : ℝ × ℝ | (p.1 + p.2 - 7) * (2 * p.1 - 3 * p.2 + 9) = 0 }
  let S2 := { p : ℝ × ℝ | (p.1 - p.2 - 2) * (4 * p.1 + 3 * p.2 - 18) = 0 }
  ∃! (p1 p2 p3 : ℝ × ℝ), p1 ∈ S1 ∧ p1 ∈ S2 ∧ p2 ∈ S1 ∧ p2 ∈ S2 ∧ p3 ∈ S1 ∧ p3 ∈ S2 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 :=
sorry

end distinct_intersection_points_l295_295546


namespace sqrt_diff_approx_l295_295832

noncomputable def x : ℝ := Real.sqrt 50 - Real.sqrt 48

theorem sqrt_diff_approx : abs (x - 0.14) < 0.01 :=
by
  sorry

end sqrt_diff_approx_l295_295832


namespace find_a_integer_condition_l295_295230

theorem find_a_integer_condition (a : ℚ) :
  (∀ n : ℕ, (a * (n * (n+2) * (n+3) * (n+4)) : ℚ).den = 1) ↔ ∃ k : ℤ, a = k / 6 := 
sorry

end find_a_integer_condition_l295_295230


namespace total_points_scored_l295_295732

theorem total_points_scored
    (Bailey_points Chandra_points Akiko_points Michiko_points : ℕ)
    (h1 : Bailey_points = 14)
    (h2 : Michiko_points = Bailey_points / 2)
    (h3 : Akiko_points = Michiko_points + 4)
    (h4 : Chandra_points = 2 * Akiko_points) :
  Bailey_points + Michiko_points + Akiko_points + Chandra_points = 54 := by
  sorry

end total_points_scored_l295_295732


namespace max_sqrt_distance_l295_295073

theorem max_sqrt_distance (x y : ℝ) 
  (h : x^2 + y^2 - 4 * x - 4 * y + 6 = 0) : 
  ∃ z, z = 3 * Real.sqrt 2 ∧ ∀ w, w = Real.sqrt (x^2 + y^2) → w ≤ z :=
sorry

end max_sqrt_distance_l295_295073


namespace max_side_of_triangle_with_perimeter_30_l295_295204

theorem max_side_of_triangle_with_perimeter_30 
  (a b c : ℕ) 
  (h1 : a + b + c = 30) 
  (h2 : a ≥ b) 
  (h3 : b ≥ c) 
  (h4 : a < b + c) 
  (h5 : b < a + c) 
  (h6 : c < a + b) 
  : a ≤ 14 :=
sorry

end max_side_of_triangle_with_perimeter_30_l295_295204


namespace conclusion_A_conclusion_B_conclusion_C_conclusion_D_l295_295181

open Real

theorem conclusion_A (A B : ℝ) (h_triangle : A > B) : sin A > sin B :=
sorry

theorem conclusion_B (a b c : ℝ) (h_acute : a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2) : b^2 + c^2 - a^2 > 0 :=
sorry

theorem conclusion_C (A B : ℝ) (h_sin : sin (2 * A) = sin (2 * B)) : ¬(∀ a b c : ℝ, isosceles_triangle a b c) :=
sorry

theorem conclusion_D (b : ℝ) (A : ℝ) (S : ℝ) (h_b : b = 3) (h_A : A = π / 3) (h_S : S = 3 * sqrt 3) : let R := (sqrt 3 / 3) in
not (let a c : ℝ := sorry in sorry) := sorry

end conclusion_A_conclusion_B_conclusion_C_conclusion_D_l295_295181


namespace field_area_proof_l295_295028

-- Define the length of the uncovered side
def L : ℕ := 20

-- Define the total amount of fencing used for the other three sides
def total_fence : ℕ := 26

-- Define the field area function
def field_area (length width : ℕ) : ℕ := length * width

-- Statement: Prove that the area of the field is 60 square feet
theorem field_area_proof : 
  ∃ W : ℕ, (2 * W + L = total_fence) ∧ (field_area L W = 60) :=
  sorry

end field_area_proof_l295_295028


namespace construction_company_total_weight_l295_295195

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

end construction_company_total_weight_l295_295195


namespace gcd_105_88_l295_295975

-- Define the numbers as constants
def a : ℕ := 105
def b : ℕ := 88

-- State the theorem: gcd(a, b) = 1
theorem gcd_105_88 : Nat.gcd a b = 1 := by
  sorry

end gcd_105_88_l295_295975


namespace subset_complU_N_l295_295867

variable {U : Type} {M N : Set U}

-- Given conditions
axiom non_empty_M : ∃ x, x ∈ M
axiom non_empty_N : ∃ y, y ∈ N
axiom subset_complU_M : N ⊆ Mᶜ

-- Prove the statement that M is a subset of the complement of N
theorem subset_complU_N : M ⊆ Nᶜ := by
  sorry

end subset_complU_N_l295_295867


namespace line_through_fixed_point_and_parabola_l295_295177

theorem line_through_fixed_point_and_parabola :
  (∀ (a : ℝ), ∃ (P : ℝ × ℝ), 
    (a - 1) * P.1 - P.2 + 2 * a + 1 = 0 ∧ 
    (∀ (x y : ℝ), (y^2 = - ((9:ℝ) / 2) * x ∧ x = -2 ∧ y = 3) ∨ (x^2 = (4:ℝ) / 3 * y ∧ x = -2 ∧ y = 3))) :=
by
  sorry

end line_through_fixed_point_and_parabola_l295_295177


namespace find_a100_l295_295737

noncomputable def sequence : ℕ → ℕ
| 0     := 1
| (n+1) := sequence n + n

theorem find_a100 : sequence 100 = 4951 :=
by
  sorry

end find_a100_l295_295737


namespace general_formula_of_geometric_seq_term_in_arithmetic_seq_l295_295734

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Condition: Geometric sequence {a_n} with a_1 = 2 and a_4 = 16
def geometric_seq (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n, a (n + 1) = a n * q

-- General formula for the sequence {a_n}
theorem general_formula_of_geometric_seq 
  (ha : geometric_seq a) (h1 : a 1 = 2) (h4 : a 4 = 16) :
  ∀ n, a n = 2^n :=
sorry

-- Condition: Arithmetic sequence {b_n} with b_3 = a_3 and b_5 = a_5
def arithmetic_seq (b : ℕ → ℝ) := ∃ d : ℝ, ∀ n, b (n + 1) = b n + d

-- Check if a_9 is a term in the sequence {b_n} and find its term number
theorem term_in_arithmetic_seq 
  (ha : geometric_seq a) (hb : arithmetic_seq b)
  (h1 : a 1 = 2) (h4 : a 4 = 16)
  (hb3 : b 3 = a 3) (hb5 : b 5 = a 5) :
  ∃ n, b n = a 9 ∧ n = 45 :=
sorry

end general_formula_of_geometric_seq_term_in_arithmetic_seq_l295_295734


namespace pies_sold_each_day_l295_295033

theorem pies_sold_each_day (total_pies : ℕ) (days_in_week : ℕ) (h1 : total_pies = 56) (h2 : days_in_week = 7) :
  (total_pies / days_in_week = 8) :=
by
exact sorry

end pies_sold_each_day_l295_295033


namespace ratio_wx_l295_295871

theorem ratio_wx (w x y : ℚ) (h1 : w / y = 3 / 4) (h2 : (x + y) / y = 13 / 4) : w / x = 1 / 3 :=
  sorry

end ratio_wx_l295_295871


namespace smaller_rectangle_ratio_l295_295196

theorem smaller_rectangle_ratio
  (length_large : ℝ) (width_large : ℝ) (area_small : ℝ)
  (h_length : length_large = 40)
  (h_width : width_large = 20)
  (h_area : area_small = 200) : 
  ∃ r : ℝ, (length_large * r) * (width_large * r) = area_small ∧ r = 0.5 :=
by
  sorry

end smaller_rectangle_ratio_l295_295196


namespace johns_age_is_15_l295_295317

-- Definitions from conditions
variables (J F : ℕ) -- J is John's age, F is his father's age
axiom sum_of_ages : J + F = 77
axiom father_age : F = 2 * J + 32

-- Target statement to prove
theorem johns_age_is_15 : J = 15 :=
by
  sorry

end johns_age_is_15_l295_295317


namespace quadratic_with_roots_1_and_2_l295_295429

theorem quadratic_with_roots_1_and_2 : ∃ (a b c : ℝ), (a = 1 ∧ b = 2) ∧ (∀ x : ℝ, x ≠ 1 → x ≠ 2 → a * x^2 + b * x + c = 0) ∧ (a * x^2 + b * x + c = x^2 - 3 * x + 2) :=
by
  sorry

end quadratic_with_roots_1_and_2_l295_295429


namespace value_of_f2_l295_295249

noncomputable def f : ℕ → ℕ :=
  sorry

axiom f_condition : ∀ x : ℕ, f (x + 1) = 2 * x + 3

theorem value_of_f2 : f 2 = 5 :=
by sorry

end value_of_f2_l295_295249


namespace volume_of_triangular_pyramid_l295_295298

variable (a b : ℝ)

noncomputable def volume_of_pyramid (a b : ℝ) : ℝ :=
  (a * b / 12) * Real.sqrt (3 * b ^ 2 - a ^ 2)

theorem volume_of_triangular_pyramid (a b : ℝ) :
  volume_of_pyramid a b = (a * b / 12) * Real.sqrt (3 * b ^ 2 - a ^ 2) :=
by
  sorry

end volume_of_triangular_pyramid_l295_295298


namespace is_linear_equation_l295_295327

def quadratic_equation (x y : ℝ) : Prop := x * y + 2 * x = 7
def fractional_equation (x y : ℝ) : Prop := (1 / x) + y = 5
def quadratic_equation_2 (x y : ℝ) : Prop := x^2 + y = 2

def linear_equation (x y : ℝ) : Prop := 2 * x - y = 2

theorem is_linear_equation (x y : ℝ) (h1 : quadratic_equation x y) (h2 : fractional_equation x y) (h3 : quadratic_equation_2 x y) : linear_equation x y :=
  sorry

end is_linear_equation_l295_295327


namespace find_a_l295_295070

noncomputable def f (a x : ℝ) : ℝ := 2^x / (2^x + a * x)

variables (a p q : ℝ)

theorem find_a
  (h1 : f a p = 6 / 5)
  (h2 : f a q = -1 / 5)
  (h3 : 2^(p + q) = 16 * p * q)
  (h4 : a > 0) :
  a = 4 :=
  sorry

end find_a_l295_295070


namespace population_in_terms_of_t_l295_295884

noncomputable def boys_girls_teachers_total (b g t : ℕ) : ℕ :=
  b + g + t

theorem population_in_terms_of_t (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) :
  boys_girls_teachers_total b g t = 26 * t :=
by
  sorry

end population_in_terms_of_t_l295_295884


namespace remaining_money_is_83_l295_295291

noncomputable def OliviaMoney : ℕ := 112
noncomputable def NigelMoney : ℕ := 139
noncomputable def TicketCost : ℕ := 28
noncomputable def TicketsBought : ℕ := 6

def TotalMoney : ℕ := OliviaMoney + NigelMoney
def TotalCost : ℕ := TicketsBought * TicketCost
def RemainingMoney : ℕ := TotalMoney - TotalCost

theorem remaining_money_is_83 : RemainingMoney = 83 := by
  sorry

end remaining_money_is_83_l295_295291


namespace national_education_fund_expenditure_l295_295623

theorem national_education_fund_expenditure (gdp_2012 : ℝ) (h : gdp_2012 = 43.5 * 10^12) : 
  (0.04 * gdp_2012) = 1.74 * 10^13 := 
by sorry

end national_education_fund_expenditure_l295_295623


namespace max_true_statements_l295_295894

theorem max_true_statements (c d : ℝ) : 
  (∃ n, 1 ≤ n ∧ n ≤ 5 ∧ 
    (n = (if (1/c > 1/d) then 1 else 0) +
          (if (c^2 < d^2) then 1 else 0) +
          (if (c > d) then 1 else 0) +
          (if (c > 0) then 1 else 0) +
          (if (d > 0) then 1 else 0))) → 
  n ≤ 3 := 
sorry

end max_true_statements_l295_295894


namespace square_area_from_diagonal_l295_295369

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : (d^2 / 2) = 72 :=
by sorry

end square_area_from_diagonal_l295_295369


namespace solve_for_x_l295_295174
-- Import the entire Mathlib library

-- Define the condition
def condition (x : ℝ) := (72 - x)^2 = x^2

-- State the theorem
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 36 :=
by {
  -- The proof will be provided here
  sorry
}

end solve_for_x_l295_295174


namespace binomial_sum_eval_l295_295844

theorem binomial_sum_eval :
  (Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 5)) +
  (Nat.factorial 6 / (Nat.factorial 4 * Nat.factorial 2)) = 36 := by
sorry

end binomial_sum_eval_l295_295844


namespace intersection_eq_l295_295743

noncomputable def U : Set ℝ := Set.univ
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- Complement of B in U
def complement_B : Set ℝ := {x | x < 2 ∨ x ≥ 3}

-- Intersection of A and complement of B
def intersection : Set ℕ := {x ∈ A | ↑x < 2 ∨ ↑x ≥ 3}

theorem intersection_eq : intersection = {1, 3, 4} :=
by
  sorry

end intersection_eq_l295_295743


namespace participation_increase_closest_to_10_l295_295962

def percentage_increase (old new : ℕ) : ℚ := ((new - old) / old) * 100

theorem participation_increase_closest_to_10 :
  (percentage_increase 80 88 = 10) ∧ 
  (percentage_increase 90 99 = 10) := by
  sorry

end participation_increase_closest_to_10_l295_295962


namespace money_left_l295_295289

theorem money_left (olivia_money nigel_money ticket_cost tickets_purchased : ℕ) 
  (h1 : olivia_money = 112) 
  (h2 : nigel_money = 139) 
  (h3 : ticket_cost = 28) 
  (h4 : tickets_purchased = 6) : 
  olivia_money + nigel_money - tickets_purchased * ticket_cost = 83 := 
by 
  sorry

end money_left_l295_295289


namespace color_change_probability_l295_295523

theorem color_change_probability :
  let cycle_duration := 45 + 5 + 35,
      total_favorable_duration := 4 + 4 + 4 in
  (total_favorable_duration : ℝ) / cycle_duration = (12 : ℝ) / (85 : ℝ) :=
by
  let cycle_duration := 45 + 5 + 35
  let total_favorable_duration := 4 + 4 + 4
  sorry

end color_change_probability_l295_295523


namespace divides_expression_l295_295724

theorem divides_expression (x : ℕ) (hx : Even x) : 90 ∣ (15 * x + 3) * (15 * x + 9) * (5 * x + 10) :=
sorry

end divides_expression_l295_295724


namespace tiles_needed_to_cover_floor_l295_295468

/-- 
A floor 10 feet by 15 feet is to be tiled with 3-inch-by-9-inch tiles. 
This theorem verifies that the necessary number of tiles is 800. 
-/
theorem tiles_needed_to_cover_floor
  (floor_length : ℝ)
  (floor_width : ℝ)
  (tile_length_inch : ℝ)
  (tile_width_inch : ℝ)
  (conversion_factor : ℝ)
  (num_tiles : ℕ) 
  (h_floor_length : floor_length = 10)
  (h_floor_width : floor_width = 15)
  (h_tile_length_inch : tile_length_inch = 3)
  (h_tile_width_inch : tile_width_inch = 9)
  (h_conversion_factor : conversion_factor = 12)
  (h_num_tiles : num_tiles = 800) :
  (floor_length * floor_width) / ((tile_length_inch / conversion_factor) * (tile_width_inch / conversion_factor)) = num_tiles :=
by
  -- The proof is not included, using sorry to mark this part
  sorry

end tiles_needed_to_cover_floor_l295_295468


namespace sum_of_prime_h_l295_295554

def h (n : ℕ) := n^4 - 380 * n^2 + 600

theorem sum_of_prime_h (S : Finset ℕ) (hS : S = { n | Nat.Prime (h n) }) :
  S.sum h = 0 :=
by
  sorry

end sum_of_prime_h_l295_295554


namespace outdoor_section_area_l295_295345

theorem outdoor_section_area :
  ∀ (width length : ℕ), width = 4 → length = 6 → (width * length = 24) :=
by
  sorry

end outdoor_section_area_l295_295345


namespace region_diff_correct_l295_295111

noncomputable def hexagon_area : ℝ := (3 * Real.sqrt 3) / 2
noncomputable def one_triangle_area : ℝ := (Real.sqrt 3) / 4
noncomputable def triangles_area : ℝ := 18 * one_triangle_area
noncomputable def R_area : ℝ := hexagon_area + triangles_area
noncomputable def S_area : ℝ := 4 * (1 + Real.sqrt 2)
noncomputable def region_diff : ℝ := S_area - R_area

theorem region_diff_correct :
  region_diff = 4 + 4 * Real.sqrt 2 - 6 * Real.sqrt 3 :=
by
  sorry

end region_diff_correct_l295_295111


namespace cubic_polynomial_has_three_real_roots_l295_295925

open Polynomial

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry
noncomputable def R : Polynomial ℝ := sorry

axiom P_degree : degree P = 2
axiom Q_degree : degree Q = 3
axiom R_degree : degree R = 3
axiom PQR_relationship : ∀ x : ℝ, P.eval x ^ 2 + Q.eval x ^ 2 = R.eval x ^ 2

theorem cubic_polynomial_has_three_real_roots : 
  (∃ x : ℝ, Q.eval x = 0 ∧ ∃ y : ℝ, Q.eval y = 0 ∧ ∃ z : ℝ, Q.eval z = 0) ∨
  (∃ x : ℝ, R.eval x = 0 ∧ ∃ y : ℝ, R.eval y = 0 ∧ ∃ z : ℝ, R.eval z = 0) :=
sorry

end cubic_polynomial_has_three_real_roots_l295_295925


namespace ellipse_eq_l295_295726

theorem ellipse_eq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : a^2 - b^2 = 4)
  (h4 : ∃ (line_eq : ℝ → ℝ), ∀ (x : ℝ), line_eq x = 3 * x + 7)
  (h5 : ∃ (mid_y : ℝ), mid_y = 1 ∧ ∃ (x1 y1 x2 y2 : ℝ), 
    ((y1 = 3 * x1 + 7) ∧ (y2 = 3 * x2 + 7)) ∧ 
    (y1 + y2) / 2 = mid_y): 
  (∀ x y : ℝ, (y^2 / (a^2 - 4) + x^2 / b^2 = 1) ↔ 
  (x^2 / 8 + y^2 / 12 = 1)) :=
by { sorry }

end ellipse_eq_l295_295726


namespace three_digit_number_five_times_product_of_digits_l295_295695

theorem three_digit_number_five_times_product_of_digits :
  ∃ (a b c : ℕ), a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ (100 * a + 10 * b + c = 5 * a * b * c) ∧ (100 * a + 10 * b + c = 175) := 
begin
  existsi 1,
  existsi 7,
  existsi 5,
  split, { norm_num }, -- a > 0
  split, { norm_num }, -- a < 10
  split, { norm_num }, -- b < 10
  split, { norm_num }, -- c < 10
  split,
  { calc 100 * 1 + 10 * 7 + 5 = 100 + 70 + 5 : by norm_num
                        ... = 175 : by norm_num
                        ... = 5 * 1 * 7 * 5 : by norm_num [1*7*5] },
  { norm_num }
end

end three_digit_number_five_times_product_of_digits_l295_295695


namespace sum_of_digits_of_product_l295_295965

noncomputable def N : ℚ := (10^100 - 1) / 9
noncomputable def M : ℚ := 4 * (10^50 - 1) / 9

def sum_of_digits (n : ℚ) : ℕ :=
  n.to_digit_list.sum

theorem sum_of_digits_of_product :
  sum_of_digits (N * M) = S := sorry

end sum_of_digits_of_product_l295_295965


namespace function_classification_l295_295688

theorem function_classification (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) := by
  sorry

end function_classification_l295_295688


namespace dot_product_example_l295_295250

def vector := ℝ × ℝ

-- Define the dot product function
def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_example : dot_product (-1, 0) (0, 2) = 0 := by
  sorry

end dot_product_example_l295_295250


namespace no_consecutive_integers_square_difference_2000_l295_295401

theorem no_consecutive_integers_square_difference_2000 :
  ¬ ∃ a : ℤ, (a + 1) ^ 2 - a ^ 2 = 2000 :=
by {
  -- some detailed steps might go here in a full proof
  sorry
}

end no_consecutive_integers_square_difference_2000_l295_295401


namespace largest_multiple_of_8_less_than_100_l295_295003

theorem largest_multiple_of_8_less_than_100 : ∃ n, n < 100 ∧ n % 8 = 0 ∧ ∀ m, m < 100 ∧ m % 8 = 0 → m ≤ n :=
begin
  use 96,
  split,
  { -- prove 96 < 100
    norm_num,
  },
  split,
  { -- prove 96 is a multiple of 8
    norm_num,
  },
  { -- prove 96 is the largest such multiple
    intros m hm,
    cases hm with h1 h2,
    have h3 : m / 8 < 100 / 8,
    { exact_mod_cast h1 },
    interval_cases (m / 8) with H,
    all_goals { 
      try { norm_num, exact le_refl _ },
    },
  },
end

end largest_multiple_of_8_less_than_100_l295_295003


namespace solve_system_l295_295115

theorem solve_system (x y z : ℝ) 
  (h1 : 19 * (x + y) + 17 = 19 * (-x + y) - 21)
  (h2 : 5 * x - 3 * z = 11 * y - 7) : 
  x = -1 ∧ z = -11 * y / 3 + 2 / 3 :=
by sorry

end solve_system_l295_295115


namespace seedling_prices_l295_295133

theorem seedling_prices (x y : ℝ) (a b : ℝ) 
  (h1 : 3 * x + 2 * y = 12)
  (h2 : x + 3 * y = 11) 
  (h3 : a + b = 200) 
  (h4 : 2 * 100 * a + 3 * 100 * b ≥ 50000) :
  x = 2 ∧ y = 3 ∧ b ≥ 100 := 
sorry

end seedling_prices_l295_295133


namespace refrigerator_profit_l295_295193

theorem refrigerator_profit 
  (marked_price : ℝ) 
  (cost_price : ℝ) 
  (profit_margin : ℝ ) 
  (discount1 : ℝ) 
  (profit1 : ℝ)
  (discount2 : ℝ):
  profit_margin = 0.1 → 
  profit1 = 200 → 
  cost_price = 2000 → 
  discount1 = 0.8 → 
  discount2 = 0.85 → 
  discount1 * marked_price - cost_price = profit1 → 

  (discount2 * marked_price - cost_price) = 337.5 := 
by 
  intros; 
  let marked_price := 2750; 
  sorry

end refrigerator_profit_l295_295193


namespace limit_bounds_Cn_l295_295065

theorem limit_bounds_Cn :
  (∀ n : ℕ, n ≥ 2 → (2 ^ (0.1887 * n^2) ≤ C(n) ∧ C(n) ≤ 2 ^ (0.6571 * n^2))) →
  (∀ n : ℕ, n ≥ 2 → (0.1887 ≤ (Real.log (C(n)) / n^2) / Real.log 2 ∧ (Real.log (C(n)) / n^2) / Real.log 2 ≤ 0.6571)) :=
by
  -- proof here
  sorry

end limit_bounds_Cn_l295_295065


namespace amount_after_two_years_l295_295691

theorem amount_after_two_years (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ)
  (hP : P = 64000) (hr : r = 1 / 6) (hn : n = 2) : 
  A = P * (1 + r) ^ n := by
  sorry

end amount_after_two_years_l295_295691


namespace lambda_range_if_u_distributed_in_all_quadrants_l295_295072

noncomputable def f (x : ℝ) (λ : ℝ) : ℝ := x^2 - λ * x + 2 * λ
noncomputable def g (x : ℝ) : ℝ := Real.log (x + 1)
noncomputable def u (x : ℝ) (λ : ℝ) : ℝ := f x λ * g x

theorem lambda_range_if_u_distributed_in_all_quadrants :
  (∀ x : ℝ, (x^2 - λ * x + 2 * λ) * Real.log (x + 1) < 0)
  ↔ λ ∈ set.Ioo (-1 / 3 : ℝ) 0 := sorry

end lambda_range_if_u_distributed_in_all_quadrants_l295_295072


namespace angles_on_x_axis_eq_l295_295121

open Set

def S1 : Set ℝ := { β | ∃ k : ℤ, β = k * 360 }
def S2 : Set ℝ := { β | ∃ k : ℤ, β = 180 + k * 360 }
def S_total : Set ℝ := S1 ∪ S2
def S_target : Set ℝ := { β | ∃ n : ℤ, β = n * 180 }

theorem angles_on_x_axis_eq : S_total = S_target := 
by 
  sorry

end angles_on_x_axis_eq_l295_295121


namespace spongebob_earnings_l295_295304

-- Define the conditions as variables and constants
def burgers_sold : ℕ := 30
def price_per_burger : ℝ := 2
def fries_sold : ℕ := 12
def price_per_fries : ℝ := 1.5

-- Define total earnings calculation
def earnings_from_burgers := burgers_sold * price_per_burger
def earnings_from_fries := fries_sold * price_per_fries
def total_earnings := earnings_from_burgers + earnings_from_fries

-- State the theorem we need to prove
theorem spongebob_earnings :
  total_earnings = 78 := by
    sorry

end spongebob_earnings_l295_295304


namespace part_1_part_2_l295_295997

-- Part (Ⅰ)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 + (m - 2) * x + 1

theorem part_1 (m : ℝ) : (∀ x : ℝ, ¬ f x m < 0) ↔ (-2 ≤ m ∧ m ≤ 6) :=
by sorry

-- Part (Ⅱ)
theorem part_2 (m : ℝ) (h_even : ∀ ⦃x : ℝ⦄, f x m = f (-x) m) :
  (m = 2) → 
  ((∀ x : ℝ, x ≤ 0 → f x 2 ≥ f 0 2) ∧ (∀ x : ℝ, x ≥ 0 → f x 2 ≥ f 0 2)) :=
by sorry

end part_1_part_2_l295_295997


namespace eta_properties_l295_295251

noncomputable def Bernoulli (n : ℕ) (p : ℝ) : distribution :=
  sorry -- Assume Bernoulli distribution, define it appropriately

open probability_theory

variables (ξ η : ℝ → ℝ)
variables (ξ_dist : distribution)
variable (given_sum : ℝ)

axiom bernoulli_properties :
  ξ_dist = Bernoulli 10 0.6 ∧ (ξ + η =ᵐ[ξ_dist] given_sum)

-- Abstract conditions
axiom exi_properties :
  expectation ξ = 6 ∧ variance ξ = 2.4

-- The theorem to prove
theorem eta_properties : expectation η = 2 ∧ variance η = 2.4 :=
by {
  -- To be proven
  sorry
}

end eta_properties_l295_295251


namespace general_term_formula_l295_295713

def seq (n : ℕ) : ℤ :=
  match n with
  | 1     => 2
  | 2     => -6
  | 3     => 12
  | 4     => -20
  | 5     => 30
  | 6     => -42
  | _     => 0 -- We match only the first few elements as given

theorem general_term_formula (n : ℕ) :
  seq n = (-1)^(n+1) * n * (n + 1) := by
  sorry

end general_term_formula_l295_295713


namespace max_triangle_side_length_l295_295209

theorem max_triangle_side_length:
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a + b + c = 30 ∧
    a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 14 :=
  sorry

end max_triangle_side_length_l295_295209


namespace profit_at_15_percent_off_l295_295192

theorem profit_at_15_percent_off 
    (cost_price marked_price : ℝ) 
    (cost_price_eq : cost_price = 2000)
    (marked_price_eq : marked_price = (200 + cost_price) / 0.8) :
    (0.85 * marked_price - cost_price) = 337.5 := by
  sorry

end profit_at_15_percent_off_l295_295192


namespace probability_prime_sum_l295_295952

-- Noncomputable theory to leverage classical probability calculations
noncomputable theory

-- Function to compute the possible sums and determine if they are prime
def count_prime_sums (n : ℕ) (sides : Finset ℕ) : ℕ :=
  sides.sum (λ a, sides.filter (λ b, (a + b).prime).card)

-- The main theorem stating that the probability that the sum of two rolls of a cube is prime
theorem probability_prime_sum (sides : Finset ℕ) (h : sides = {1, 2, 3, 4, 5, 6}) :
  (count_prime_sums 2 sides : ℚ) / (sides.card * sides.card : ℚ) = 5 / 12 :=
begin
  sorry
end

end probability_prime_sum_l295_295952


namespace zachary_crunches_more_than_pushups_l295_295182

def push_ups_zachary : ℕ := 46
def crunches_zachary : ℕ := 58

theorem zachary_crunches_more_than_pushups : crunches_zachary - push_ups_zachary = 12 := by
  sorry

end zachary_crunches_more_than_pushups_l295_295182


namespace g_domain_l295_295337

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arccos (x^3))

theorem g_domain : { x : ℝ | -1 ≤ x ∧ x ≤ 1 ∧ x ≠ 0 } = (Set.Icc (-1) 0 ∪ Set.Icc 0 1) \ {0} :=
by
  sorry

end g_domain_l295_295337


namespace restore_fractions_l295_295510

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l295_295510


namespace maximum_value_fraction_l295_295992

theorem maximum_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2 * x + y) + y / (x + 2 * y)) ≤ 2 / 3 :=
sorry

end maximum_value_fraction_l295_295992


namespace restore_original_problem_l295_295500

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l295_295500


namespace intersection_angle_parabola_circle_l295_295963

-- Define the conditions for the parabola and the circle.
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def circle_shifted (p : ℝ) (x y : ℝ) := (x - p / 2)^2 + y^2 = 4 * p^2

-- Define the problem statement in Lean 4
theorem intersection_angle_parabola_circle (p x y : ℝ) :
  parabola p x y ∧ circle_shifted p x y → ∃ (θ : ℝ), θ = 60 :=
by
  sorry

end intersection_angle_parabola_circle_l295_295963


namespace units_digit_3m_squared_plus_2m_l295_295740

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_3m_squared_plus_2m : (3 * (m^2 + 2^m)) % 10 = 9 := by
  sorry

end units_digit_3m_squared_plus_2m_l295_295740


namespace new_drug_effectiveness_expectation_of_X_company_claim_doubt_l295_295817

section
variables {n a b c d : ℕ}
variables (ta tb tc td : ℕ) (K : ℝ)

-- Part 1: Given the conditions, prove the calculated K^2 value is less than the critical value at 90% confidence
theorem new_drug_effectiveness :
  let K2 := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  in n = 200 → a = 60 → b = 40 → c = 50 → d = 50 →
     K2 < 2.706 :=
sorry

-- Part 2: Given the conditions, find the expectation of X, where X is the number of cured patients out of a sample of 3
theorem expectation_of_X :
  let p_cured := 0.6
  in ∀ X : ℕ → ℝ, (binomial C_3(X) * (p_cured)^X * (1 - p_cured)^(3 - X)) * X in $[0,1,2,3]$
  → ∑ P(X) = 1 → E[X] = 1.8 :=
sorry

-- Part 3: Evaluate the company's claim that the efficacy is 90%
theorem company_claim_doubt :
  let claim_eff := 0.9
  in ∑ P(X ≤ 6 out of 10 patients) ≈ 0.013 
  -- Probability calculated under binomial distribution
  → this probability is very small
  → we should doubt the company's claim :=
sorry
end

end new_drug_effectiveness_expectation_of_X_company_claim_doubt_l295_295817


namespace max_modulus_l295_295381

open Complex

noncomputable def max_modulus_condition (z : ℂ) : Prop :=
  abs (z - (0 + 2*Complex.I)) = 1

theorem max_modulus : ∀ z : ℂ, max_modulus_condition z → abs z ≤ 3 :=
  by sorry

end max_modulus_l295_295381


namespace negation_proof_l295_295775

theorem negation_proof :
  (¬ ∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) :=
by
  sorry

end negation_proof_l295_295775


namespace total_customers_in_line_l295_295839

-- Define the number of people behind the first person
def people_behind := 11

-- Define the total number of people in line
def people_in_line : Nat := people_behind + 1

-- Prove the total number of people in line is 12
theorem total_customers_in_line : people_in_line = 12 :=
by
  sorry

end total_customers_in_line_l295_295839


namespace fraction_zero_l295_295334

theorem fraction_zero (x : ℝ) (h : x ≠ 1) (h₁ : (x + 1) / (x - 1) = 0) : x = -1 :=
sorry

end fraction_zero_l295_295334


namespace triangle_area_l295_295370

theorem triangle_area :
  let line1 (x : ℝ) := 2 * x + 1
  let line2 (x : ℝ) := (16 + x) / 4
  ∃ (base height : ℝ), height = (16 + 2 * base) / 7 ∧ base * height / 2 = 18 / 7 :=
  by
    sorry

end triangle_area_l295_295370


namespace cube_edge_ratio_l295_295332

theorem cube_edge_ratio (a b : ℕ) (h : a^3 = 27 * b^3) : a = 3 * b :=
sorry

end cube_edge_ratio_l295_295332


namespace mixed_fraction_product_example_l295_295478

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l295_295478


namespace bag_cost_is_2_l295_295583

-- Define the inputs and conditions
def carrots_per_day := 1
def days_per_year := 365
def carrots_per_bag := 5
def yearly_spending := 146

-- The final goal is to find the cost per bag
def cost_per_bag := yearly_spending / ((carrots_per_day * days_per_year) / carrots_per_bag)

-- Prove that the cost per bag is $2
theorem bag_cost_is_2 : cost_per_bag = 2 := by
  -- Using sorry to complete the proof
  sorry

end bag_cost_is_2_l295_295583


namespace product_ab_l295_295354

noncomputable def a : ℝ := 3
noncomputable def b : ℝ := 5 / 2
def tan_function (x : ℝ) : ℝ := a * Real.tan (b * x)

theorem product_ab :
  (∀ x, x = -2*Real.pi/5 ∨ x = 0 ∨ x = 2*Real.pi/5 → (Real.tan (b * x) = 0 ∨ Real.tan (b * x) = 1)) ∧
  tan_function (Real.pi / 10) = 3 →
  a * b = 7.5 := 
by
  sorry

end product_ab_l295_295354


namespace inequality_proof_l295_295572

variable {x₁ x₂ x₃ x₄ : ℝ}

theorem inequality_proof
  (h₁ : x₁ ≥ x₂) (h₂ : x₂ ≥ x₃) (h₃ : x₃ ≥ x₄) (h₄ : x₄ ≥ 2)
  (h₅ : x₂ + x₃ + x₄ ≥ x₁) 
  : (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := 
by {
  sorry
}

end inequality_proof_l295_295572


namespace no_such_p_l295_295723

theorem no_such_p : ¬ ∃ p : ℕ, p > 0 ∧ (∃ k : ℤ, 4 * p + 35 = k * (3 * p - 7)) :=
by
  sorry

end no_such_p_l295_295723


namespace gcd_18_24_l295_295233

theorem gcd_18_24 : Int.gcd 18 24 = 6 :=
by
  sorry

end gcd_18_24_l295_295233


namespace simplify_expression_l295_295833

theorem simplify_expression : 1 - (1 / (1 + Real.sqrt 2)) + (1 / (1 - Real.sqrt 2)) = 1 - 2 * Real.sqrt 2 := by
  sorry

end simplify_expression_l295_295833


namespace solve_for_n_l295_295852

theorem solve_for_n :
  ∃ n : ℤ, -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.sin (750 * Real.pi / 180) ∧ n = 30 :=
by
  sorry

end solve_for_n_l295_295852


namespace mixed_fraction_product_example_l295_295477

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l295_295477


namespace union_M_N_eq_l295_295253

open Set

-- Define set M and set N according to the problem conditions
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {y | ∃ x ∈ M, y = x^2}

-- The theorem we need to prove
theorem union_M_N_eq : M ∪ N = {0, 1, 2, 4} :=
by
  -- Just assert the theorem without proving it
  sorry

end union_M_N_eq_l295_295253


namespace dealership_sales_l295_295555

theorem dealership_sales :
  (∀ (n : ℕ), 3 * n ≤ 36 → 5 * n ≤ x) →
  (36 / 3) * 5 = 60 :=
by
  sorry

end dealership_sales_l295_295555


namespace marbles_left_calculation_l295_295470

/-- A magician starts with 20 red marbles and 30 blue marbles.
    He removes 3 red marbles and 12 blue marbles. We need to 
    prove that he has 35 marbles left in total. -/
theorem marbles_left_calculation (initial_red : ℕ) (initial_blue : ℕ) (removed_red : ℕ) 
    (removed_blue : ℕ) (H1 : initial_red = 20) (H2 : initial_blue = 30) 
    (H3 : removed_red = 3) (H4 : removed_blue = 4 * removed_red) :
    (initial_red - removed_red) + (initial_blue - removed_blue) = 35 :=
by
   -- sorry to skip the proof
   sorry

end marbles_left_calculation_l295_295470


namespace factorize_first_poly_factorize_second_poly_l295_295228

variable (x m n : ℝ)

-- Proof statement for the first polynomial
theorem factorize_first_poly : x^2 + 14*x + 49 = (x + 7)^2 := 
by sorry

-- Proof statement for the second polynomial
theorem factorize_second_poly : (m - 1) + n^2 * (1 - m) = (m - 1) * (1 - n) * (1 + n) := 
by sorry

end factorize_first_poly_factorize_second_poly_l295_295228


namespace union_P_Q_l295_295580

-- Definition of sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 4 }
def Q : Set ℝ := { x | -3 < x ∧ x < 3 }

-- Statement to prove
theorem union_P_Q :
  P ∪ Q = { x : ℝ | -3 < x ∧ x ≤ 4 } :=
sorry

end union_P_Q_l295_295580


namespace smallest_four_digit_divisible_by_35_l295_295156

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l295_295156


namespace mixed_fraction_product_example_l295_295475

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l295_295475


namespace jellybeans_needed_l295_295350

theorem jellybeans_needed (n : ℕ) : (n ≥ 120 ∧ n % 15 = 14) → n = 134 :=
by sorry

end jellybeans_needed_l295_295350


namespace part1_relationship_range_part2_maximize_profit_l295_295112

variables {x y a : ℝ}
noncomputable def zongzi_profit (x : ℝ) : ℝ := -5 * x + 6000

-- Given conditions
def conditions (x : ℝ) : Prop :=
  100 ≤ x ∧ x ≤ 150

-- Part 1: Prove the functional relationship and range of x
theorem part1_relationship_range (x : ℝ) (h : conditions x) :
  zongzi_profit x = -5 * x + 6000 :=
  sorry

-- Part 2: Profit maximization given modified purchase price condition
noncomputable def modified_zongzi_profit (x : ℝ) (a : ℝ) : ℝ :=
  (a - 5) * x + 6000

def maximize_strategy (x a : ℝ) : Prop :=
  (0 < a ∧ a < 5 → x = 100) ∧ (5 ≤ a ∧ a < 10 → x = 150)

theorem part2_maximize_profit (a : ℝ) (ha : 0 < a ∧ a < 10) :
  ∃ x, conditions x ∧ maximize_strategy x a :=
  sorry

end part1_relationship_range_part2_maximize_profit_l295_295112


namespace derivative_ln_div_x_l295_295768

noncomputable def f (x : ℝ) := (Real.log x) / x

theorem derivative_ln_div_x (x : ℝ) (h : x ≠ 0) : deriv f x = (1 - Real.log x) / (x^2) :=
by
  sorry

end derivative_ln_div_x_l295_295768


namespace at_least_one_angle_ge_60_l295_295639

theorem at_least_one_angle_ge_60 (A B C : ℝ) (hA : A < 60) (hB : B < 60) (hC : C < 60) (h_sum : A + B + C = 180) : false :=
sorry

end at_least_one_angle_ge_60_l295_295639


namespace chris_babysitting_hours_l295_295357

theorem chris_babysitting_hours (h : ℕ) (video_game_cost candy_cost earn_per_hour leftover total_cost : ℕ) :
  video_game_cost = 60 ∧
  candy_cost = 5 ∧
  earn_per_hour = 8 ∧
  leftover = 7 ∧
  total_cost = video_game_cost + candy_cost ∧
  earn_per_hour * h = total_cost + leftover
  → h = 9 := by
  intros
  sorry

end chris_babysitting_hours_l295_295357


namespace Ben_shirts_is_15_l295_295673

variable (Alex_shirts Joe_shirts Ben_shirts : Nat)

def Alex_has_4 : Alex_shirts = 4 := by sorry

def Joe_has_more_than_Alex : Joe_shirts = Alex_shirts + 3 := by sorry

def Ben_has_more_than_Joe : Ben_shirts = Joe_shirts + 8 := by sorry

theorem Ben_shirts_is_15 (h1 : Alex_shirts = 4) (h2 : Joe_shirts = Alex_shirts + 3) (h3 : Ben_shirts = Joe_shirts + 8) : Ben_shirts = 15 := by
  sorry

end Ben_shirts_is_15_l295_295673


namespace books_sold_correct_l295_295600

-- Definitions of the conditions
def initial_books : ℕ := 33
def remaining_books : ℕ := 7
def books_sold : ℕ := initial_books - remaining_books

-- The statement to be proven (with proof omitted)
theorem books_sold_correct : books_sold = 26 := by
  -- Proof omitted
  sorry

end books_sold_correct_l295_295600


namespace max_side_length_triangle_l295_295215

def triangle_with_max_side_length (a b c : ℕ) (ha : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hper : a + b + c = 30) : Prop :=
  a > b ∧ a > c ∧ a = 14

theorem max_side_length_triangle : ∃ a b c : ℕ, 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 30 ∧ a > b ∧ a > c ∧ a = 14 :=
sorry

end max_side_length_triangle_l295_295215


namespace product_of_binaries_l295_295058

-- Step a) Define the binary numbers as Lean 4 terms.
def bin_11011 : ℕ := 0b11011
def bin_111 : ℕ := 0b111
def bin_101 : ℕ := 0b101

-- Step c) Define the goal to be proven.
theorem product_of_binaries :
  bin_11011 * bin_111 * bin_101 = 0b1110110001 :=
by
  -- proof goes here
  sorry

end product_of_binaries_l295_295058


namespace union_complement_A_when_a_eq_1_A_cap_B_eq_A_range_of_a_l295_295568

def setA (a : ℝ) : Set ℝ := { x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3 }
def setB : Set ℝ := { x | -1 / 2 < x ∧ x < 2 }
def complementB : Set ℝ := { x | x ≤ -1 / 2 ∨ x ≥ 2 }

theorem union_complement_A_when_a_eq_1 :
  (complementB ∪ setA 1) = { x | x ≤ 1 ∨ x ≥ 2 } :=
by
  sorry

theorem A_cap_B_eq_A_range_of_a (a : ℝ) :
  (setA a ∩ setB = setA a) → (-1 < a ∧ a ≤ 1) :=
by
  sorry

end union_complement_A_when_a_eq_1_A_cap_B_eq_A_range_of_a_l295_295568


namespace pure_imaginary_product_imaginary_part_fraction_l295_295706

-- Part 1
theorem pure_imaginary_product (m : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : z1 = m + i) (h3 : z2 = 2 + m * i) :
  (z1 * z2).re = 0 ↔ m = 0 := 
sorry

-- Part 2
theorem imaginary_part_fraction (m : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : z1 = m + i) (h3 : z2 = 2 + m * i)
  (h4 : z1^2 - 2 * z1 + 2 = 0) :
  (z2 / z1).im = -1 / 2 :=
sorry

end pure_imaginary_product_imaginary_part_fraction_l295_295706


namespace nicole_answers_correctly_l295_295907

theorem nicole_answers_correctly :
  ∀ (C K N : ℕ), C = 17 → K = C + 8 → N = K - 3 → N = 22 :=
by
  intros C K N hC hK hN
  sorry

end nicole_answers_correctly_l295_295907


namespace line_y_intercept_l295_295677

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 9)) (h2 : (x2, y2) = (5, 21)) :
    ∃ b : ℝ, (∀ x : ℝ, y = 4 * x + b) ∧ (b = 1) :=
by
  use 1
  sorry

end line_y_intercept_l295_295677


namespace restore_fractions_l295_295509

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l295_295509


namespace rate_per_meter_eq_2_5_l295_295700

-- Definitions of the conditions
def diameter : ℝ := 14
def total_cost : ℝ := 109.96

-- The theorem to be proven
theorem rate_per_meter_eq_2_5 (π : ℝ) (hπ : π = 3.14159) : 
  diameter = 14 ∧ total_cost = 109.96 → (109.96 / (π * 14)) = 2.5 :=
by
  sorry

end rate_per_meter_eq_2_5_l295_295700


namespace smallest_four_digit_divisible_by_35_l295_295167

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l295_295167


namespace expected_balls_original_positions_l295_295540

noncomputable def expected_original_positions : ℝ :=
  8 * ((3/4:ℝ)^3)

theorem expected_balls_original_positions :
  expected_original_positions = 3.375 := by
  sorry

end expected_balls_original_positions_l295_295540


namespace restore_original_problem_l295_295490

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l295_295490


namespace correct_option_is_B_l295_295014

-- Definitions and conditions based on the problem
def is_monomial (t : String) : Prop :=
  t = "1"

def coefficient (expr : String) : Int :=
  if expr = "x" then 1
  else if expr = "-3x" then -3
  else 0

def degree (term : String) : Int :=
  if term = "5x^2y" then 3
  else 0

-- Proof statement
theorem correct_option_is_B : 
  is_monomial "1" ∧ ¬ (coefficient "x" = 0) ∧ ¬ (coefficient "-3x" = 3) ∧ ¬ (degree "5x^2y" = 2) := 
by
  -- Proof steps will go here
  sorry

end correct_option_is_B_l295_295014


namespace village_population_l295_295663

theorem village_population (P : ℝ) (h : 0.9 * P = 45000) : P = 50000 :=
by
  sorry

end village_population_l295_295663


namespace find_m_value_l295_295375

-- Defining the hyperbola equation and the conditions
def hyperbola_eq (x y : ℝ) (m : ℝ) : Prop :=
  (x^2 / m) - (y^2 / 4) = 1

-- Definition of the focal distance
def focal_distance (c : ℝ) :=
  2 * c = 6

-- Definition of the relationship c^2 = a^2 + b^2 for hyperbolas
def hyperbola_focal_distance_eq (m : ℝ) (c b : ℝ) : Prop :=
  c^2 = m + b^2

-- Stating that the hyperbola has the given focal distance
def given_focal_distance : Prop :=
  focal_distance 3

-- Stating the given condition on b²
def given_b_squared : Prop :=
  4 = 4

-- The main theorem stating that m = 5 given the conditions.
theorem find_m_value (m : ℝ) : 
  (hyperbola_eq 1 1 m) → given_focal_distance → given_b_squared → m = 5 :=
by
  sorry

end find_m_value_l295_295375


namespace polar_not_one_to_one_correspondence_l295_295801

theorem polar_not_one_to_one_correspondence :
  ¬ ∃ f : ℝ × ℝ → ℝ × ℝ, (∀ p1 p2 : ℝ × ℝ, f p1 = f p2 → p1 = p2) ∧
  (∀ q : ℝ × ℝ, ∃ p : ℝ × ℝ, q = f p) :=
by
  sorry

end polar_not_one_to_one_correspondence_l295_295801


namespace machine_tasks_l295_295450

theorem machine_tasks (y : ℕ) 
  (h1 : (1 : ℚ)/(y + 4) + (1 : ℚ)/(y + 3) + (1 : ℚ)/(4 * y) = (1 : ℚ)/y) : y = 1 :=
sorry

end machine_tasks_l295_295450


namespace median_of_consecutive_integers_sum_eq_6_pow_4_l295_295780

theorem median_of_consecutive_integers_sum_eq_6_pow_4 :
  ∀ (s : ℕ) (n : ℕ), s = 36 → ∑ i in finset.range 36, (n + i) = 6^4 → 36 / 2 = 36 :=
by
  sorry

end median_of_consecutive_integers_sum_eq_6_pow_4_l295_295780


namespace double_point_quadratic_l295_295398

theorem double_point_quadratic (m x1 x2 : ℝ) 
  (H1 : x1 < 1) (H2 : 1 < x2)
  (H3 : ∃ (y1 y2 : ℝ), y1 = 2 * x1 ∧ y2 = 2 * x2 ∧ y1 = x1^2 + 2 * m * x1 - m ∧ y2 = x2^2 + 2 * m * x2 - m)
  : m < 1 :=
sorry

end double_point_quadratic_l295_295398


namespace probability_neither_red_nor_purple_l295_295340

theorem probability_neither_red_nor_purple :
  let total_balls := 100 
  let white_balls := 50 
  let green_balls := 30 
  let yellow_balls := 8 
  let red_balls := 9 
  let purple_balls := 3 
  (100 - (red_balls + purple_balls)) / total_balls = 0.88 := 
by 
  -- Definitions based on conditions
  let total_balls := 100 
  let white_balls := 50 
  let green_balls := 30 
  let yellow_balls := 8 
  let red_balls := 9 
  let purple_balls := 3 
  -- Compute the probability
  sorry

end probability_neither_red_nor_purple_l295_295340


namespace restore_original_problem_l295_295499

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l295_295499


namespace part1_part2_l295_295874

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (1 - a) * x + (1 - a)

theorem part1 (x : ℝ) : f x 4 ≥ 7 ↔ x ≥ 5 ∨ x ≤ -2 :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, -1 < x → f x a ≥ 0) ↔ a ≤ 1 :=
sorry

end part1_part2_l295_295874


namespace evaluate_polynomial_l295_295543

noncomputable def polynomial_evaluation : Prop :=
∀ (x : ℝ), x^2 - 3*x - 9 = 0 ∧ 0 < x → (x^4 - 3*x^3 - 9*x^2 + 27*x - 8) = (65 + 81*(Real.sqrt 5))/2

theorem evaluate_polynomial : polynomial_evaluation :=
sorry

end evaluate_polynomial_l295_295543


namespace Randy_trip_distance_l295_295432

noncomputable def total_distance (x : ℝ) :=
  (x / 4) + 40 + 10 + (x / 6)

theorem Randy_trip_distance (x : ℝ) (h : total_distance x = x) : x = 600 / 7 :=
by
  sorry

end Randy_trip_distance_l295_295432


namespace smallest_positive_integer_n_l295_295168

theorem smallest_positive_integer_n (n : ℕ) (h : 5 * n ≡ 1463 [MOD 26]) : n = 23 :=
sorry

end smallest_positive_integer_n_l295_295168


namespace poly_perfect_fourth_l295_295446

theorem poly_perfect_fourth (a b c : ℤ) (h : ∀ x : ℤ, ∃ k : ℤ, (a * x^2 + b * x + c) = k^4) : 
  a = 0 ∧ b = 0 :=
sorry

end poly_perfect_fourth_l295_295446


namespace abs_neg_three_l295_295680

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l295_295680


namespace geometric_sequence_formula_l295_295594

variable {q : ℝ} -- Common ratio
variable {m n : ℕ} -- Positive natural numbers
variable {b : ℕ → ℝ} -- Geometric sequence

-- This is only necessary if importing Mathlib didn't bring it in
noncomputable def geom_sequence (m n : ℕ) (b : ℕ → ℝ) (q : ℝ) : Prop :=
  b n = b m * q^(n - m)

theorem geometric_sequence_formula (q : ℝ) (m n : ℕ) (b : ℕ → ℝ) 
  (hmn : 0 < m ∧ 0 < n) :
  geom_sequence m n b q :=
by sorry

end geometric_sequence_formula_l295_295594


namespace mixed_fraction_product_l295_295517

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l295_295517


namespace fraction_addition_l295_295720

theorem fraction_addition (x y : ℚ) (h : x / y = 2 / 3) : (x + y) / y = 5 / 3 := 
by 
  sorry

end fraction_addition_l295_295720


namespace max_a4_l295_295399

variable {a_n : ℕ → ℝ}

-- Assume a_n is a positive geometric sequence
def is_geometric_seq (a_n : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a_n (n + 1) = a_n n * r

-- Given conditions
def condition1 (a_n : ℕ → ℝ) : Prop := is_geometric_seq a_n
def condition2 (a_n : ℕ → ℝ) : Prop := a_n 3 + a_n 5 = 4

theorem max_a4 (a_n : ℕ → ℝ) (h1 : condition1 a_n) (h2 : condition2 a_n) :
    ∃ max_a4 : ℝ, max_a4 = 2 :=
  sorry

end max_a4_l295_295399


namespace tracy_initial_balloons_l295_295533

theorem tracy_initial_balloons (T : ℕ) : 
  (12 + 8 + (T + 24) / 2 = 35) → T = 6 :=
by
  sorry

end tracy_initial_balloons_l295_295533


namespace find_term_in_sequence_l295_295252

theorem find_term_in_sequence (n : ℕ) (k : ℕ) (term_2020: ℚ) : 
  (3^7 = 2187) → 
  (2020 : ℕ) / (2187 : ℕ) = term_2020 → 
  (term_2020 = 2020 / 2187) →
  (∃ (k : ℕ), k = 2020 ∧ (2 ≤ k ∧ k < 2187 ∧ k % 3 ≠ 0)) → 
  (2020 / 2187 = (1347 / 2187 : ℚ)) :=
by {
  sorry
}

end find_term_in_sequence_l295_295252


namespace solve_fractions_l295_295484

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l295_295484


namespace total_books_l295_295644

-- Define the number of books Victor originally had and the number he bought
def original_books : ℕ := 9
def bought_books : ℕ := 3

-- The proof problem statement: Prove Victor has a total of original_books + bought_books books
theorem total_books : original_books + bought_books = 12 := by
  -- proof will go here, using sorry to indicate it's omitted
  sorry

end total_books_l295_295644


namespace money_left_l295_295293

noncomputable def olivia_money : ℕ := 112
noncomputable def nigel_money : ℕ := 139
noncomputable def ticket_cost : ℕ := 28
noncomputable def num_tickets : ℕ := 6

theorem money_left : (olivia_money + nigel_money - ticket_cost * num_tickets) = 83 :=
by
  sorry

end money_left_l295_295293


namespace correct_calculation_l295_295934

-- Definitions of calculations based on conditions
def calc_A (a : ℝ) := a^2 + a^2 = a^4
def calc_B (a : ℝ) := (a^2)^3 = a^5
def calc_C (a : ℝ) := a + 2 = 2 * a
def calc_D (a b : ℝ) := (a * b)^3 = a^3 * b^3

-- Theorem stating that only the fourth calculation is correct
theorem correct_calculation (a b : ℝ) :
  ¬(calc_A a) ∧ ¬(calc_B a) ∧ ¬(calc_C a) ∧ calc_D a b :=
by sorry

end correct_calculation_l295_295934


namespace vector_sum_correct_l295_295374

-- Define the three vectors
def v1 : ℝ × ℝ := (5, -3)
def v2 : ℝ × ℝ := (-4, 6)
def v3 : ℝ × ℝ := (2, -8)

-- Define the expected result
def expected_sum : ℝ × ℝ := (3, -5)

-- Define vector addition (component-wise)
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- The theorem statement
theorem vector_sum_correct : vector_add (vector_add v1 v2) v3 = expected_sum := by
  sorry

end vector_sum_correct_l295_295374


namespace total_crayons_l295_295318

def box1_crayons := 3 * (8 + 4 + 5)
def box2_crayons := 4 * (7 + 6 + 3)
def box3_crayons := 2 * (11 + 5 + 2)
def unique_box_crayons := 9 + 2 + 7

theorem total_crayons : box1_crayons + box2_crayons + box3_crayons + unique_box_crayons = 169 := by
  sorry

end total_crayons_l295_295318


namespace count_squares_with_center_55_25_l295_295296

noncomputable def number_of_squares_with_natural_number_coordinates : ℕ :=
  600

theorem count_squares_with_center_55_25 :
  ∀ (x y : ℕ), (x = 55) ∧ (y = 25) → number_of_squares_with_natural_number_coordinates = 600 :=
by
  intros x y h
  cases h
  sorry

end count_squares_with_center_55_25_l295_295296


namespace arithmetic_sequence_geometric_sequence_l295_295662

-- Arithmetic sequence proof
theorem arithmetic_sequence (d n : ℕ) (a_n a_1 : ℤ) (s_n : ℤ) :
  d = 2 → n = 15 → a_n = -10 →
  a_1 = -38 ∧ s_n = -360 :=
sorry

-- Geometric sequence proof
theorem geometric_sequence (a_1 a_4 q s_3 : ℤ) :
  a_1 = -1 → a_4 = 64 →
  q = -4 ∧ s_3 = -13 :=
sorry

end arithmetic_sequence_geometric_sequence_l295_295662


namespace angle_range_between_lines_l295_295248

-- Defining the equation and problem statement
theorem angle_range_between_lines (b : ℝ) :
  ∃ θ : ℝ, (arctan (2 * sqrt 5 / 5) ≤ θ ∧ θ ≤ π / 2) ∧
          (∃ (x y : ℝ), x^2 + (b+2)*x*y + b*y^2 = 0) :=
sorry

end angle_range_between_lines_l295_295248


namespace subset_sum_bounds_l295_295421

theorem subset_sum_bounds (M m n : ℕ) (A : Finset ℕ)
  (h1 : 1 ≤ m) (h2 : m ≤ n) (h3 : 1 ≤ M) (h4 : M ≤ (m * (m + 1)) / 2) (hA : A.card = m) (hA_subset : ∀ x ∈ A, x ∈ Finset.range (n + 1)) :
  ∃ B ⊆ A, 0 ≤ (B.sum id) - M ∧ (B.sum id) - M ≤ n - m :=
by
  sorry

end subset_sum_bounds_l295_295421


namespace angelaAgeInFiveYears_l295_295675

namespace AgeProblem

variables (A B : ℕ) -- Define Angela's and Beth's current age as natural numbers.

-- Condition 1: Angela is four times as old as Beth.
axiom angelaAge : A = 4 * B

-- Condition 2: Five years ago, the sum of their ages was 45 years.
axiom ageSumFiveYearsAgo : (A - 5) + (B - 5) = 45

-- Theorem: Prove that Angela's age in 5 years will be 49.
theorem angelaAgeInFiveYears : A + 5 = 49 :=
by {
  -- proof goes here
  sorry
}

end AgeProblem

end angelaAgeInFiveYears_l295_295675


namespace relationship_between_a_b_c_l295_295943

noncomputable def a := (3 / 5 : ℝ) ^ (2 / 5)
noncomputable def b := (2 / 5 : ℝ) ^ (3 / 5)
noncomputable def c := (2 / 5 : ℝ) ^ (2 / 5)

theorem relationship_between_a_b_c :
  a > c ∧ c > b :=
by
  sorry

end relationship_between_a_b_c_l295_295943


namespace find_A_students_l295_295217

variables (Alan Beth Carlos Diana : Prop)
variable (num_As : ℕ)

def Alan_implies_Beth := Alan → Beth
def Beth_implies_no_Carlos_A := Beth → ¬Carlos
def Carlos_implies_Diana := Carlos → Diana
def Beth_implies_Diana := Beth → Diana

theorem find_A_students 
  (h1 : Alan_implies_Beth Alan Beth)
  (h2 : Beth_implies_no_Carlos_A Beth Carlos)
  (h3 : Carlos_implies_Diana Carlos Diana)
  (h4 : Beth_implies_Diana Beth Diana)
  (h_cond : num_As = 2) :
  (Alan ∧ Beth) ∨ (Beth ∧ Diana) ∨ (Carlos ∧ Diana) :=
by sorry

end find_A_students_l295_295217


namespace amelia_wins_probability_is_expected_l295_295040

noncomputable def amelia_wins_probability : ℚ :=
  let P_am_head := (1:ℚ) / 4
  let P_bl_head := (1:ℚ) / 3
  let P_am_tail := 1 - P_am_head
  let P_bl_tail := 1 - P_bl_head
  let P_both_tails := P_am_tail * P_bl_tail
  let P_not_both_tails := 1 - P_both_tails
  let sum_geom_prob (n : ℕ) : ℚ := if n = 0 then 1 else sum (λ k, (P_both_tails^k)) (fin (n+1))
  P_am_head * sum_geom_prob 4

theorem amelia_wins_probability_is_expected : amelia_wins_probability = (15:ℚ) / (32:ℚ) :=
by sorry

end amelia_wins_probability_is_expected_l295_295040


namespace total_games_l295_295276

variable (Ken_games Dave_games Jerry_games : ℕ)

-- The conditions from the problem.
def condition1 : Prop := Ken_games = Dave_games + 5
def condition2 : Prop := Dave_games = Jerry_games + 3
def condition3 : Prop := Jerry_games = 7

-- The final statement to prove
theorem total_games (h1 : condition1 Ken_games Dave_games) 
                    (h2 : condition2 Dave_games Jerry_games) 
                    (h3 : condition3 Jerry_games) : 
  Ken_games + Dave_games + Jerry_games = 32 :=
by
  sorry

end total_games_l295_295276


namespace restore_original_problem_l295_295495

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l295_295495


namespace total_hours_worked_l295_295921

-- Define the number of hours worked on Saturday
def hours_saturday : ℕ := 6

-- Define the number of hours worked on Sunday
def hours_sunday : ℕ := 4

-- Define the total number of hours worked on both days
def total_hours : ℕ := hours_saturday + hours_sunday

-- The theorem to prove the total number of hours worked on Saturday and Sunday
theorem total_hours_worked : total_hours = 10 := by
  sorry

end total_hours_worked_l295_295921


namespace select_student_for_performance_and_stability_l295_295559

def average_score_A : ℝ := 6.2
def average_score_B : ℝ := 6.0
def average_score_C : ℝ := 5.8
def average_score_D : ℝ := 6.2

def variance_A : ℝ := 0.32
def variance_B : ℝ := 0.58
def variance_C : ℝ := 0.12
def variance_D : ℝ := 0.25

theorem select_student_for_performance_and_stability :
  (average_score_A ≤ average_score_D ∧ variance_D < variance_A) →
  (average_score_B < average_score_A ∧ average_score_B < average_score_D) →
  (average_score_C < average_score_A ∧ average_score_C < average_score_D) →
  "D" = "D" :=
by
  intros h₁ h₂ h₃
  exact rfl

end select_student_for_performance_and_stability_l295_295559


namespace find_first_term_and_common_difference_l295_295271

variable (a d : ℕ)
variable (S_even S_odd S_total : ℕ)

-- Conditions
axiom condition1 : S_total = 354
axiom condition2 : S_even = 192
axiom condition3 : S_odd = 162
axiom condition4 : 12*(2*a + 11*d) = 2*S_total
axiom condition5 : 6*(a + 6*d) = S_even
axiom condition6 : 6*(a + 5*d) = S_odd

-- Theorem to prove
theorem find_first_term_and_common_difference (a d S_even S_odd S_total : ℕ)
  (h1 : S_total = 354)
  (h2 : S_even = 192)
  (h3 : S_odd = 162)
  (h4 : 12*(2*a + 11*d) = 2*S_total)
  (h5 : 6*(a + 6*d) = S_even)
  (h6 : 6*(a + 5*d) = S_odd) : a = 2 ∧ d = 5 := by
  sorry

end find_first_term_and_common_difference_l295_295271


namespace min_value_AN_plus_2BM_l295_295379

open Real

def point_on_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 4
def not_on_axes (x y : ℝ) : Prop := x ≠ 0 ∧ y ≠ 0
def point_A : ℝ × ℝ := (2, 0)
def point_B : ℝ × ℝ := (0, 2)
def on_y_axis (x y : ℝ) : Prop := y = 0
def on_x_axis (x y : ℝ) : Prop := x = 0

theorem min_value_AN_plus_2BM (P : ℝ × ℝ) 
  (h1 : point_on_circle P.1 P.2)
  (h2 : not_on_axes P.1 P.2) :
  ∃ M N : ℝ × ℝ, 
    on_y_axis M.1 M.2 ∧ on_x_axis N.1 N.2 ∧ 
    |2 - N.2| + 2 * |2 - M.1| = 8 := 
sorry

end min_value_AN_plus_2BM_l295_295379


namespace slope_of_line_l295_295854

theorem slope_of_line (x y : ℝ) (h : 4 * x - 7 * y = 28) : (∃ m b : ℝ, y = m * x + b ∧ m = 4 / 7) :=
by
  -- Proof omitted
  sorry

end slope_of_line_l295_295854


namespace largest_sphere_radius_l295_295348

-- Define the conditions
def inner_radius : ℝ := 3
def outer_radius : ℝ := 7
def circle_center_x := 5
def circle_center_z := 2
def circle_radius := 2

-- Define the question into a statement
noncomputable def radius_of_largest_sphere : ℝ :=
  (29 : ℝ) / 4

-- Prove the required radius given the conditions
theorem largest_sphere_radius:
  ∀ (r : ℝ),
  r = radius_of_largest_sphere → r * r = inner_radius * inner_radius + (circle_center_x * circle_center_x + (r - circle_center_z) * (r - circle_center_z))
:=
by
  sorry

end largest_sphere_radius_l295_295348


namespace no_positive_integer_satisfies_l295_295231

theorem no_positive_integer_satisfies : ¬ ∃ n : ℕ, 0 < n ∧ (20 * n + 2) ∣ (2003 * n + 2002) :=
by sorry

end no_positive_integer_satisfies_l295_295231


namespace true_statements_about_f_l295_295996

noncomputable def f (x : ℝ) := 2 * abs (Real.cos x) * Real.sin x + Real.sin (2 * x)

theorem true_statements_about_f :
  (∀ x y : ℝ, -π/4 ≤ x ∧ x < y ∧ y ≤ π/4 → f x < f y) ∧
  (∀ y : ℝ, -2 ≤ y ∧ y ≤ 2 → (∃ x : ℝ, f x = y)) :=
by
  sorry

end true_statements_about_f_l295_295996


namespace proportion_not_necessarily_correct_l295_295259

theorem proportion_not_necessarily_correct
  (a b c d : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : b ≠ 0)
  (h₃ : c ≠ 0)
  (h₄ : d ≠ 0)
  (h₅ : a * d = b * c) :
  ¬ ((a + 1) / b = (c + 1) / d) :=
by 
  sorry

end proportion_not_necessarily_correct_l295_295259


namespace mandy_total_cost_after_discount_l295_295610

-- Define the conditions
def packs_black_shirts : ℕ := 6
def packs_yellow_shirts : ℕ := 8
def packs_green_socks : ℕ := 5

def items_per_pack_black_shirts : ℕ := 7
def items_per_pack_yellow_shirts : ℕ := 4
def items_per_pack_green_socks : ℕ := 5

def cost_per_pack_black_shirts : ℕ := 25
def cost_per_pack_yellow_shirts : ℕ := 15
def cost_per_pack_green_socks : ℕ := 10

def discount_rate : ℚ := 0.10

-- Calculate the total number of each type of item
def total_black_shirts : ℕ := packs_black_shirts * items_per_pack_black_shirts
def total_yellow_shirts : ℕ := packs_yellow_shirts * items_per_pack_yellow_shirts
def total_green_socks : ℕ := packs_green_socks * items_per_pack_green_socks

-- Calculate the total cost before discount
def total_cost_before_discount : ℕ :=
  (packs_black_shirts * cost_per_pack_black_shirts) +
  (packs_yellow_shirts * cost_per_pack_yellow_shirts) +
  (packs_green_socks * cost_per_pack_green_socks)

-- Calculate the total cost after discount
def discount_amount : ℚ := discount_rate * total_cost_before_discount
def total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount

-- Problem to prove: Total cost after discount is $288
theorem mandy_total_cost_after_discount : total_cost_after_discount = 288 := by
  sorry

end mandy_total_cost_after_discount_l295_295610


namespace candies_per_friend_l295_295047

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (friends : ℕ) 
  (h_initial : initial_candies = 10)
  (h_additional : additional_candies = 4)
  (h_friends : friends = 7) : initial_candies + additional_candies = 14 ∧ 14 / friends = 2 :=
by
  sorry

end candies_per_friend_l295_295047


namespace unit_vector_parallel_to_a_l295_295447

theorem unit_vector_parallel_to_a (x y : ℝ) (h1 : x^2 + y^2 = 1) (h2 : 12 * y = 5 * x) :
  (x = 12 / 13 ∧ y = 5 / 13) ∨ (x = -12 / 13 ∧ y = -5 / 13) := by
  sorry

end unit_vector_parallel_to_a_l295_295447


namespace product_of_roots_of_t_squared_equals_49_l295_295061

theorem product_of_roots_of_t_squared_equals_49 : 
  ∃ t : ℝ, (t^2 = 49) ∧ (t = 7 ∨ t = -7) ∧ (t * (7 + -7)) = -49 := 
by
  sorry

end product_of_roots_of_t_squared_equals_49_l295_295061


namespace leak_time_l295_295300

theorem leak_time (A L : ℝ) (PipeA_filling_rate : A = 1 / 6) (Combined_rate : A - L = 1 / 10) : 
  1 / L = 15 :=
by
  sorry

end leak_time_l295_295300


namespace x_y_solution_l295_295541

variable (x y : ℕ)

noncomputable def x_wang_speed : ℕ := x - 6

theorem x_y_solution (hx : (5 : ℚ) / 6 * x = y) (hy : (2 : ℚ) / 3 * (x - 6) = y - 10) : x = 36 ∧ y = 30 :=
by {
  sorry
}

end x_y_solution_l295_295541


namespace LukaNeeds24CupsOfWater_l295_295609

theorem LukaNeeds24CupsOfWater
  (L S W : ℕ)
  (h1 : S = 2 * L)
  (h2 : W = 4 * S)
  (h3 : L = 3) :
  W = 24 := by
  sorry

end LukaNeeds24CupsOfWater_l295_295609


namespace largest_multiple_of_8_less_than_100_l295_295006

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), (n % 8 = 0) ∧ (n < 100) ∧ (∀ m : ℕ, (m % 8 = 0) ∧ (m < 100) → m ≤ n) :=
begin
  use 96,
  split,
  { -- 96 is a multiple of 8
    exact nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 96),
  },
  split,
  { -- 96 is less than 100
    norm_num,
  },
  { -- 96 is the largest multiple of 8 less than 100
    intros m hm,
    obtain ⟨k, rfl⟩ := (nat.dvd_iff_mod_eq_zero.mp hm.1),
    have : k ≤ 12, by linarith,
    linarith [mul_le_mul (zero_le _ : (0 : ℕ) ≤ 8) this (zero_le _ : (0 : ℕ) ≤ 12) (zero_le _ : (0 : ℕ) ≤ 8)],
  },
end

end largest_multiple_of_8_less_than_100_l295_295006


namespace eighteenth_entry_l295_295553

def r_8 (n : ℕ) : ℕ := n % 8

theorem eighteenth_entry (n : ℕ) (h : r_8 (3 * n) ≤ 3) : n = 17 :=
sorry

end eighteenth_entry_l295_295553


namespace a_2n_is_perfect_square_l295_295279

-- Define the sequence a_n as per the problem's conditions
def a (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 4
  else a (n - 1) + a (n - 3) + a (n - 4)

-- Define the Fibonacci sequence for comparison
def fib (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

-- Key theorem to prove: a_{2n} is a perfect square
theorem a_2n_is_perfect_square (n : ℕ) : 
  ∃ k : ℕ, a (2 * n) = k * k :=
sorry

end a_2n_is_perfect_square_l295_295279


namespace calculate_f2_f_l295_295589

variable {f : ℝ → ℝ}

-- Definition of the conditions
def tangent_line_at_x2 (f : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ → ℝ), (∀ x, L x = -x + 1) ∧ (∀ x, f x = L x + (f x - L 2))

theorem calculate_f2_f'2 (h : tangent_line_at_x2 f) :
  f 2 + deriv f 2 = -2 :=
sorry

end calculate_f2_f_l295_295589


namespace james_total_time_l295_295888

def time_to_play_main_game : ℕ := 
  let download_time := 10
  let install_time := download_time / 2
  let update_time := download_time * 2
  let account_time := 5
  let internet_issues_time := 15
  let before_tutorial_time := download_time + install_time + update_time + account_time + internet_issues_time
  let tutorial_time := before_tutorial_time * 3
  before_tutorial_time + tutorial_time

theorem james_total_time : time_to_play_main_game = 220 := by
  sorry

end james_total_time_l295_295888


namespace correct_factorization_l295_295650

-- Definitions from conditions
def A: Prop := ∀ x y: ℝ, x^2 - 4*y^2 = (x + y) * (x - 4*y)
def B: Prop := ∀ x: ℝ, (x + 4) * (x - 4) = x^2 - 16
def C: Prop := ∀ x: ℝ, x^2 - 2*x + 1 = (x - 1)^2
def D: Prop := ∀ x: ℝ, x^2 - 8*x + 9 = (x - 4)^2 - 7

-- Goal is to prove that C is a correct factorization
theorem correct_factorization: C := by
  sorry

end correct_factorization_l295_295650


namespace bob_twice_alice_l295_295038

open ProbabilityTheory

theorem bob_twice_alice (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1000) (hb : 0 ≤ b ∧ b ≤ 3000) :
  P (λ (x : ℝ × ℝ), x.2 ≥ 2 * x.1) = 1 / 2 := by
sorry

end bob_twice_alice_l295_295038


namespace product_of_roots_of_t_squared_equals_49_l295_295062

theorem product_of_roots_of_t_squared_equals_49 : 
  ∃ t : ℝ, (t^2 = 49) ∧ (t = 7 ∨ t = -7) ∧ (t * (7 + -7)) = -49 := 
by
  sorry

end product_of_roots_of_t_squared_equals_49_l295_295062


namespace cube_remainder_l295_295932

theorem cube_remainder (n : ℤ) (h : n % 13 = 5) : (n^3) % 17 = 6 :=
by
  sorry

end cube_remainder_l295_295932


namespace new_average_weight_l295_295659

theorem new_average_weight (original_players : ℕ) (new_players : ℕ) 
  (average_weight_original : ℝ) (weight_new_player1 : ℝ) (weight_new_player2 : ℝ) : 
  original_players = 7 → 
  new_players = 2 →
  average_weight_original = 76 → 
  weight_new_player1 = 110 → 
  weight_new_player2 = 60 → 
  (original_players * average_weight_original + weight_new_player1 + weight_new_player2) / (original_players + new_players) = 78 :=
by 
  intros h1 h2 h3 h4 h5;
  sorry

end new_average_weight_l295_295659


namespace sufficient_but_not_necessary_l295_295528

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (a > b + 1) → (a > b) ∧ (¬(a > b) → ¬(a > b + 1)) :=
by
  sorry

end sufficient_but_not_necessary_l295_295528


namespace johns_original_earnings_l295_295275

def JohnsEarningsBeforeRaise (currentEarnings: ℝ) (percentageIncrease: ℝ) := 
  ∀ x, currentEarnings = x + x * percentageIncrease → x = 50

theorem johns_original_earnings : 
  JohnsEarningsBeforeRaise 80 0.60 :=
by
  intro x
  intro h
  sorry

end johns_original_earnings_l295_295275


namespace pies_sold_each_day_l295_295031

theorem pies_sold_each_day (total_pies: ℕ) (days_in_week: ℕ) 
  (h1: total_pies = 56) (h2: days_in_week = 7) : 
  total_pies / days_in_week = 8 :=
by
  sorry

end pies_sold_each_day_l295_295031


namespace hyperbola_properties_l295_295581

theorem hyperbola_properties :
  (∃ x y : Real,
    (x^2 / 4 - y^2 / 2 = 1) ∧
    (∃ a b c e : Real,
      2 * a = 4 ∧
      2 * b = 2 * Real.sqrt 2 ∧
      c = Real.sqrt (a^2 + b^2) ∧
      2 * c = 2 * Real.sqrt 6 ∧
      e = c / a)) :=
by
  sorry

end hyperbola_properties_l295_295581


namespace ratio_of_volumes_l295_295008

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1/3) * Real.pi * r^2 * h

theorem ratio_of_volumes :
  let r_C := 10
  let h_C := 20
  let r_D := 18
  let h_D := 12
  volume_cone r_C h_C / volume_cone r_D h_D = 125 / 243 :=
by
  sorry

end ratio_of_volumes_l295_295008


namespace smallest_vertical_distance_l295_295441

def f (x : ℝ) : ℝ := abs x
def g (x : ℝ) : ℝ := -x^2 - 4 * x - 3

theorem smallest_vertical_distance : 
  ∃ x : ℝ, (∀ y : ℝ, abs (f y - g y) ≥ abs (f x - g x)) ∧ abs (f x - g x) = 3 / 4 :=
begin
  sorry,
end

end smallest_vertical_distance_l295_295441


namespace alice_has_largest_result_l295_295526

def initial_number : ℕ := 15

def alice_transformation (x : ℕ) : ℕ := (x * 3 - 2 + 4)
def bob_transformation (x : ℕ) : ℕ := (x * 2 + 3 - 5)
def charlie_transformation (x : ℕ) : ℕ := (x + 5) / 2 * 4

def alice_final := alice_transformation initial_number
def bob_final := bob_transformation initial_number
def charlie_final := charlie_transformation initial_number

theorem alice_has_largest_result :
  alice_final > bob_final ∧ alice_final > charlie_final := by
  sorry

end alice_has_largest_result_l295_295526


namespace gcd_of_gx_and_x_l295_295741

theorem gcd_of_gx_and_x (x : ℕ) (h : 7200 ∣ x) : Nat.gcd ((5 * x + 3) * (11 * x + 2) * (17 * x + 5) * (4 * x + 7)) x = 30 := 
by 
  sorry

end gcd_of_gx_and_x_l295_295741


namespace find_second_number_l295_295016

def problem (a b c d : ℚ) : Prop :=
  a + b + c + d = 280 ∧
  a = 2 * b ∧
  c = 2 / 3 * a ∧
  d = b + c

theorem find_second_number (a b c d : ℚ) (h : problem a b c d) : b = 52.5 :=
by
  -- Proof will go here.
  sorry

end find_second_number_l295_295016


namespace find_socks_cost_l295_295408

variable (S : ℝ)
variable (socks_cost : ℝ := 9.5)
variable (shoe_cost : ℝ := 92)
variable (jack_has : ℝ := 40)
variable (needs_more : ℝ := 71)
variable (total_funds : ℝ := jack_has + needs_more)

theorem find_socks_cost (h : 2 * S + shoe_cost = total_funds) : S = socks_cost :=
by 
  sorry

end find_socks_cost_l295_295408


namespace price_reduction_l295_295190

theorem price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h : original_price = 289) (h2 : final_price = 256) :
  289 * (1 - x) ^ 2 = 256 := sorry

end price_reduction_l295_295190


namespace value_of_square_l295_295957

theorem value_of_square (z : ℝ) (h : 3 * z^2 + 2 * z = 5 * z + 11) : (6 * z - 5)^2 = 141 := by
  sorry

end value_of_square_l295_295957


namespace mary_money_left_l295_295903

def drink_price (p : ℕ) : ℕ := p
def medium_pizza_price (p : ℕ) : ℕ := 2 * p
def large_pizza_price (p : ℕ) : ℕ := 3 * p
def drinks_cost (n : ℕ) (p : ℕ) : ℕ := n * drink_price p
def medium_pizzas_cost (n : ℕ) (p : ℕ) : ℕ := n * medium_pizza_price p
def large_pizza_cost (n : ℕ) (p : ℕ) : ℕ := n * large_pizza_price p
def total_cost (p : ℕ) : ℕ := drinks_cost 5 p + medium_pizzas_cost 2 p + large_pizza_cost 1 p
def money_left (initial_money : ℕ) (p : ℕ) : ℕ := initial_money - total_cost p

theorem mary_money_left (p : ℕ) : money_left 50 p = 50 - 12 * p := sorry

end mary_money_left_l295_295903


namespace annual_interest_rate_equivalent_l295_295905

noncomputable def quarterly_compound_rate : ℝ := 1 + 0.02
noncomputable def annual_compound_amount : ℝ := quarterly_compound_rate ^ 4

theorem annual_interest_rate_equivalent : 
  (annual_compound_amount - 1) * 100 = 8.24 := 
by
  sorry

end annual_interest_rate_equivalent_l295_295905


namespace sym_axis_of_curve_eq_zero_b_plus_d_l295_295311

theorem sym_axis_of_curve_eq_zero_b_plus_d
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (h_symm : ∀ x : ℝ, 2 * x = (a * ((a * x + b) / (c * x + d)) + b) / (c * ((a * x + b) / (c * x + d)) + d)) :
  b + d = 0 :=
sorry

end sym_axis_of_curve_eq_zero_b_plus_d_l295_295311


namespace round_to_nearest_whole_l295_295758

theorem round_to_nearest_whole (x : ℝ) (hx : x = 7643.498201) : Int.floor (x + 0.5) = 7643 := 
by
  -- To prove
  sorry

end round_to_nearest_whole_l295_295758


namespace largest_visits_is_four_l295_295941

noncomputable def largest_num_visits (stores people visits : ℕ) (eight_people_two_stores : ℕ) 
  (one_person_min : ℕ) : ℕ := 4 -- This represents the largest number of stores anyone could have visited.

theorem largest_visits_is_four 
  (stores : ℕ) (total_visits : ℕ) (people_shopping : ℕ) 
  (eight_people_two_stores : ℕ) (each_one_store : ℕ) 
  (H1 : stores = 8) 
  (H2 : total_visits = 23) 
  (H3 : people_shopping = 12) 
  (H4 : eight_people_two_stores = 8)
  (H5 : each_one_store = 1) :
  largest_num_visits stores people_shopping total_visits eight_people_two_stores each_one_store = 4 :=
by
  sorry

end largest_visits_is_four_l295_295941


namespace largest_integer_k_for_distinct_real_roots_l295_295083

theorem largest_integer_k_for_distinct_real_roots :
  ∀ (k : ℤ), ((k < 3) ∧ (k ≠ 2)) → k ≤ 1 :=
by
  intros k h_conditions,
  let a := k - 2,
  let b := -4,
  let c := 4,
  have discriminant_pos : 48 - 16 * k > 0,
  {
    -- The proof for 48 - 16k > 0 which implies k < 3 is assumed from the conditions
    sorry,
  },
  have k_not_2 : k ≠ 2,
  {
    -- The proof for k ≠ 2 is assumed from the conditions
    sorry,
  },
  -- Since k < 3 and k ≠ 2, the largest integer satisfying this is 1
  sorry

end largest_integer_k_for_distinct_real_roots_l295_295083


namespace restore_original_problem_l295_295496

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l295_295496


namespace conditional_probabilities_l295_295433

noncomputable def total_outcomes := 6 * 6 * 6

noncomputable def event_A_outcomes := 6 * 5 * 4

noncomputable def event_B_outcomes := total_outcomes - 5 * 5 * 5

noncomputable def event_A_and_B_outcomes := 3 * 5 * 4

theorem conditional_probabilities :
  (event_A_and_B_outcomes.toRat / event_B_outcomes.toRat = 60 / 91) ∧ 
  (event_A_and_B_outcomes.toRat / event_A_outcomes.toRat = 1 / 2) :=
by
  have total_outcomes_value : total_outcomes = 216 := by decide
  have event_A_value : event_A_outcomes = 120 := by decide
  have event_B_value : event_B_outcomes = 91 := by decide
  have event_A_and_B_value : event_A_and_B_outcomes = 60 := by decide
  rw [event_A_value, event_B_value, event_A_and_B_value]
  split
  case left => exact (by norm_num)
  case right => exact (by norm_num)
  sorry

end conditional_probabilities_l295_295433


namespace arithmetic_sequence_twentieth_term_l295_295051

theorem arithmetic_sequence_twentieth_term :
  ∀ (a_1 d : ℕ), a_1 = 3 → d = 4 → (a_1 + (20 - 1) * d) = 79 := by
  intros a_1 d h1 h2
  rw [h1, h2]
  simp
  sorry

end arithmetic_sequence_twentieth_term_l295_295051


namespace martha_jar_spices_cost_l295_295108

def price_per_jar_spices (p_beef p_fv p_oj : ℕ) (price_spices : ℕ) :=
  let total_spent := (3 * p_beef) + (8 * p_fv) + p_oj + (3 * price_spices)
  let total_points := (total_spent / 10) * 50 + if total_spent > 100 then 250 else 0
  total_points

theorem martha_jar_spices_cost (price_spices : ℕ) :
  price_per_jar_spices 11 4 37 price_spices = 850 → price_spices = 6 := by
  sorry

end martha_jar_spices_cost_l295_295108


namespace carson_circles_theorem_l295_295046

-- Define the dimensions of the warehouse
def warehouse_length : ℕ := 600
def warehouse_width : ℕ := 400

-- Define the perimeter calculation
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- Define the distance Carson walked
def distance_walked : ℕ := 16000

-- Define the number of circles Carson skipped
def circles_skipped : ℕ := 2

-- Define the expected number of circles Carson was supposed to circle
def expected_circles :=
  let actual_circles := distance_walked / (perimeter warehouse_length warehouse_width)
  actual_circles + circles_skipped

-- The theorem we want to prove
theorem carson_circles_theorem : expected_circles = 10 := by
  sorry

end carson_circles_theorem_l295_295046


namespace candle_burning_time_l295_295793

theorem candle_burning_time :
  ∃ T : ℝ, 
    (∀ T, 0 ≤ T ∧ T ≤ 4 → thin_candle_length = 24 - 6 * T) ∧
    (∀ T, 0 ≤ T ∧ T ≤ 6 → thick_candle_length = 24 - 4 * T) ∧
    (2 * (24 - 6 * T) = 24 - 4 * T) →
    T = 3 :=
by
  sorry

end candle_burning_time_l295_295793


namespace gcd_840_1764_evaluate_polynomial_at_2_l295_295186

-- Define the Euclidean algorithm steps and prove the gcd result
theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

-- Define the polynomial and evaluate it using Horner's method
def polynomial := λ x : ℕ => 2 * (x ^ 4) + 3 * (x ^ 3) + 5 * x - 4

theorem evaluate_polynomial_at_2 : polynomial 2 = 62 := by
  sorry

end gcd_840_1764_evaluate_polynomial_at_2_l295_295186


namespace max_g_value_l295_295842

def g : Nat → Nat
| n => if n < 15 then n + 15 else g (n - 6)

theorem max_g_value : ∀ n, g n ≤ 29 := by
  sorry

end max_g_value_l295_295842


namespace x_can_be_any_sign_l295_295999

theorem x_can_be_any_sign
  (x y z w : ℤ)
  (h1 : (y - 1) * (w - 2) ≠ 0)
  (h2 : (x + 2)/(y - 1) < - (z + 3)/(w - 2)) :
  ∃ x : ℤ, True :=
by
  sorry

end x_can_be_any_sign_l295_295999


namespace find_a_l295_295880

noncomputable def f (a x : ℝ) : ℝ := Real.log (x + 1) / Real.log a

theorem find_a (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) 
  (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f a x ∧ f a x ≤ 1) : a = 2 :=
sorry

end find_a_l295_295880


namespace gcd_le_sqrt_sum_l295_295378

theorem gcd_le_sqrt_sum {a b : ℕ} (h : ∃ k : ℕ, (a + 1) / b + (b + 1) / a = k) :
  ↑(Nat.gcd a b) ≤ Real.sqrt (a + b) := sorry

end gcd_le_sqrt_sum_l295_295378


namespace same_yield_among_squares_l295_295239

-- Define the conditions
def rectangular_schoolyard (length : ℝ) (width : ℝ) := length = 70 ∧ width = 35

def total_harvest (harvest : ℝ) := harvest = 1470 -- in kilograms (14.7 quintals)

def smaller_square (side : ℝ) := side = 0.7

-- Define the proof problem
theorem same_yield_among_squares :
  ∃ side : ℝ, smaller_square side ∧
  ∃ length width harvest : ℝ, rectangular_schoolyard length width ∧ total_harvest harvest →
  ∃ (yield1 yield2 : ℝ), yield1 = yield2 ∧ yield1 ≠ 0 ∧ yield2 ≠ 0 :=
by sorry

end same_yield_among_squares_l295_295239


namespace pure_imaginary_solution_l295_295280

theorem pure_imaginary_solution (m : ℝ) (z : ℂ)
  (h1 : z = (m^2 - 1) + (m - 1) * I)
  (h2 : z.re = 0) : m = -1 :=
sorry

end pure_imaginary_solution_l295_295280


namespace xy_system_l295_295394

theorem xy_system (x y : ℚ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 :=
by
  sorry

end xy_system_l295_295394


namespace how_many_years_older_l295_295805

-- Definitions of the conditions
variables (a b c : ℕ)
def b_is_16 : Prop := b = 16
def b_is_twice_c : Prop := b = 2 * c
def sum_is_42 : Prop := a + b + c = 42

-- Statement of the proof problem
theorem how_many_years_older (h1 : b_is_16 b) (h2 : b_is_twice_c b c) (h3 : sum_is_42 a b c) : a - b = 2 :=
by
  sorry

end how_many_years_older_l295_295805


namespace bus_stop_time_per_hour_l295_295807

theorem bus_stop_time_per_hour
  (speed_no_stops : ℝ)
  (speed_with_stops : ℝ)
  (h1 : speed_no_stops = 50)
  (h2 : speed_with_stops = 35) : 
  18 = (60 * (1 - speed_with_stops / speed_no_stops)) :=
by
  sorry

end bus_stop_time_per_hour_l295_295807


namespace sum_of_powers_modulo_l295_295603

theorem sum_of_powers_modulo (R : Finset ℕ) (S : ℕ) :
  (∀ n < 100, ∃ r, r ∈ R ∧ r = 3^n % 500) →
  S = R.sum id →
  (S % 500) = 0 :=
by {
  -- Proof would go here
  sorry
}

end sum_of_powers_modulo_l295_295603


namespace problem_solution_l295_295565

theorem problem_solution (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : a + b = 1) :
  (a + 1 / b) ^ 2 + (b + 1 / a) ^ 2 ≥ 25 / 2 :=
sorry

end problem_solution_l295_295565


namespace min_abs_E_value_l295_295645

theorem min_abs_E_value (x E : ℝ) (h : |x - 4| + |E| + |x - 5| = 10) : |E| = 9 :=
sorry

end min_abs_E_value_l295_295645


namespace derivative_f_at_0_l295_295660

def f (x : ℝ) : ℝ := if x ≠ 0 then tan (x^3 + x^2 * sin (2/x)) else 0

noncomputable def f_prime_at_0 : ℝ :=
deriv f 0

theorem derivative_f_at_0 : f_prime_at_0 = 0 :=
by
  have h0 : f 0 = 0 := rfl
  have h_diff : differentiable_at ℝ f 0 := 
    sorry
  have h_deriv : deriv f 0 = 0 :=
    sorry
  exact h_deriv

end derivative_f_at_0_l295_295660


namespace sum_coordinates_point_C_l295_295909

/-
Let point A = (0,0), point B is on the line y = 6, and the slope of AB is 3/4.
Point C lies on the y-axis with a slope of 1/2 from B to C.
We need to prove that the sum of the coordinates of point C is 2.
-/
theorem sum_coordinates_point_C : 
  ∃ (A B C : ℝ × ℝ), 
  A = (0, 0) ∧ 
  B.2 = 6 ∧ 
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 ∧ 
  C.1 = 0 ∧ 
  (C.2 - B.2) / (C.1 - B.1) = 1 / 2 ∧ 
  C.1 + C.2 = 2 :=
by
  sorry

end sum_coordinates_point_C_l295_295909


namespace no_n_gt_1_divisibility_l295_295847

theorem no_n_gt_1_divisibility (n : ℕ) (h : n > 1) : ¬ (3 ^ (n - 1) + 5 ^ (n - 1)) ∣ (3 ^ n + 5 ^ n) :=
by
  sorry

end no_n_gt_1_divisibility_l295_295847


namespace socks_problem_l295_295409

/-
  Theorem: Given x + y + z = 15, 2x + 4y + 5z = 36, and x, y, z ≥ 1, 
  the number of $2 socks Jack bought is x = 4.
-/

theorem socks_problem
  (x y z : ℕ)
  (h1 : x + y + z = 15)
  (h2 : 2 * x + 4 * y + 5 * z = 36)
  (h3 : 1 ≤ x)
  (h4 : 1 ≤ y)
  (h5 : 1 ≤ z) :
  x = 4 :=
  sorry

end socks_problem_l295_295409


namespace initially_calculated_average_height_l295_295624

theorem initially_calculated_average_height
    (A : ℕ)
    (initial_total_height : ℕ)
    (real_total_height : ℕ)
    (height_error : ℕ := 60)
    (num_boys : ℕ := 35)
    (actual_average_height : ℕ := 183)
    (initial_total_height_eq : initial_total_height = num_boys * A)
    (real_total_height_eq : real_total_height = num_boys * actual_average_height)
    (height_discrepancy : initial_total_height = real_total_height + height_error) :
    A = 181 :=
by
  sorry

end initially_calculated_average_height_l295_295624


namespace marcus_baseball_cards_l295_295424

/-- 
Marcus initially has 210.0 baseball cards.
Carter gives Marcus 58.0 more baseball cards.
Prove that Marcus now has 268.0 baseball cards.
-/
theorem marcus_baseball_cards (initial_cards : ℝ) (additional_cards : ℝ) 
  (h_initial : initial_cards = 210.0) (h_additional : additional_cards = 58.0) : 
  initial_cards + additional_cards = 268.0 :=
  by
    sorry

end marcus_baseball_cards_l295_295424


namespace median_of_36_consecutive_integers_l295_295787

theorem median_of_36_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) 
  (h1 : num_of_integers = 36) (h2 : sum_of_integers = 6 ^ 4) : 
  (sum_of_integers / num_of_integers) = 36 := 
by 
  sorry

end median_of_36_consecutive_integers_l295_295787


namespace cross_section_is_rectangle_l295_295800

def RegularTetrahedron : Type := sorry

def Plane : Type := sorry

variable (T : RegularTetrahedron) (P : Plane)

-- Conditions
axiom regular_tetrahedron (T : RegularTetrahedron) : Prop
axiom plane_intersects_tetrahedron (P : Plane) (T : RegularTetrahedron) : Prop
axiom plane_parallel_opposite_edges (P : Plane) (T : RegularTetrahedron) : Prop

-- The cross-section formed by intersecting a regular tetrahedron with a plane
-- that is parallel to two opposite edges is a rectangle.
theorem cross_section_is_rectangle (T : RegularTetrahedron) (P : Plane) 
  (hT : regular_tetrahedron T) 
  (hI : plane_intersects_tetrahedron P T) 
  (hP : plane_parallel_opposite_edges P T) :
  ∃ (shape : Type), shape = Rectangle := 
  sorry

end cross_section_is_rectangle_l295_295800


namespace smallest_four_digit_divisible_by_35_l295_295154

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l295_295154


namespace maximum_weight_truck_can_carry_l295_295036

-- Definitions for the conditions.
def weight_boxes : Nat := 100 * 100
def weight_crates : Nat := 10 * 60
def weight_sacks : Nat := 50 * 50
def weight_additional_bags : Nat := 10 * 40

-- Summing up all the weights.
def total_weight : Nat :=
  weight_boxes + weight_crates + weight_sacks + weight_additional_bags

-- The theorem stating the maximum weight.
theorem maximum_weight_truck_can_carry : total_weight = 13500 := by
  sorry

end maximum_weight_truck_can_carry_l295_295036


namespace find_current_listens_l295_295101

theorem find_current_listens (x : ℕ) (h : 15 * x = 900000) : x = 60000 :=
by
  sorry

end find_current_listens_l295_295101


namespace fraction_of_orange_juice_in_mixture_l295_295642

theorem fraction_of_orange_juice_in_mixture
  (capacity_pitcher : ℕ)
  (fraction_first_pitcher : ℚ)
  (fraction_second_pitcher : ℚ)
  (condition1 : capacity_pitcher = 500)
  (condition2 : fraction_first_pitcher = 1/4)
  (condition3 : fraction_second_pitcher = 3/7) :
  (125 + 500 * (3/7)) / (2 * 500) = 95 / 280 :=
by
  sorry

end fraction_of_orange_juice_in_mixture_l295_295642


namespace debby_drink_days_l295_295841

theorem debby_drink_days :
  ∀ (total_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ),
  total_bottles = 301 →
  bottles_per_day = 144 →
  remaining_bottles = 157 →
  (total_bottles - remaining_bottles) / bottles_per_day = 1 :=
by
  intros total_bottles bottles_per_day remaining_bottles ht he hb
  sorry

end debby_drink_days_l295_295841


namespace only_one_solution_l295_295548

theorem only_one_solution (n : ℕ) (h : 0 < n ∧ ∃ a : ℕ, a * a = 5^n + 4) : n = 1 :=
sorry

end only_one_solution_l295_295548


namespace smallest_four_digit_divisible_by_35_l295_295160

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l295_295160


namespace solve_for_x_l295_295173

theorem solve_for_x (x : ℕ) : (8^3 + 8^3 + 8^3 + 8^3 = 2^x) → x = 11 :=
by
  intro h
  sorry

end solve_for_x_l295_295173


namespace area_of_rhombus_perimeter_of_rhombus_l295_295918

-- Definitions and conditions for the area of the rhombus
def d1 : ℕ := 18
def d2 : ℕ := 16

-- Definition for the side length of the rhombus
def side_length : ℕ := 10

-- Statement for the area of the rhombus
theorem area_of_rhombus : (d1 * d2) / 2 = 144 := by
  sorry

-- Statement for the perimeter of the rhombus
theorem perimeter_of_rhombus : 4 * side_length = 40 := by
  sorry

end area_of_rhombus_perimeter_of_rhombus_l295_295918


namespace total_points_scored_l295_295731

theorem total_points_scored
    (Bailey_points Chandra_points Akiko_points Michiko_points : ℕ)
    (h1 : Bailey_points = 14)
    (h2 : Michiko_points = Bailey_points / 2)
    (h3 : Akiko_points = Michiko_points + 4)
    (h4 : Chandra_points = 2 * Akiko_points) :
  Bailey_points + Michiko_points + Akiko_points + Chandra_points = 54 := by
  sorry

end total_points_scored_l295_295731


namespace solution_set_inequality_l295_295549

theorem solution_set_inequality : {x : ℝ | (x + 3) * (1 - x) ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end solution_set_inequality_l295_295549


namespace find_a_from_quadratic_inequality_l295_295236

theorem find_a_from_quadratic_inequality :
  ∀ (a : ℝ), (∀ x : ℝ, (x > - (1 / 2)) ∧ (x < 1 / 3) → a * x^2 - 2 * x + 2 > 0) → a = -12 :=
by
  intros a h
  have h1 := h (-1 / 2)
  have h2 := h (1 / 3)
  sorry

end find_a_from_quadratic_inequality_l295_295236


namespace problem_l295_295587

theorem problem (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) 
: (5 * m * r - 2 * n * t) / (7 * n * t - 10 * m * r) = -31 / 56 := 
sorry

end problem_l295_295587


namespace xyz_logarithm_sum_l295_295770

theorem xyz_logarithm_sum :
  ∃ (X Y Z : ℕ), X > 0 ∧ Y > 0 ∧ Z > 0 ∧
  Nat.gcd X (Nat.gcd Y Z) = 1 ∧ 
  (↑X * Real.log 3 / Real.log 180 + ↑Y * Real.log 5 / Real.log 180 = ↑Z) ∧ 
  (X + Y + Z = 4) :=
by
  sorry

end xyz_logarithm_sum_l295_295770


namespace probability_largest_ball_is_six_l295_295238

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_largest_ball_is_six : 
  (choose 6 4 : ℝ) / (choose 10 4 : ℝ) = (15 : ℝ) / (210 : ℝ) :=
by
  sorry

end probability_largest_ball_is_six_l295_295238


namespace y_intercept_of_line_l295_295797

theorem y_intercept_of_line : 
  ∀ (x y : ℝ), 3 * x - 5 * y = 7 → y = -7 / 5 :=
by
  intro x y h
  sorry

end y_intercept_of_line_l295_295797


namespace mixed_fraction_product_example_l295_295480

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l295_295480


namespace translation_coordinates_l295_295393

variable (A B A1 B1 : ℝ × ℝ)

theorem translation_coordinates
  (hA : A = (-1, 0))
  (hB : B = (1, 2))
  (hA1 : A1 = (2, -1))
  (translation_A : A1 = (A.1 + 3, A.2 - 1))
  (translation_B : B1 = (B.1 + 3, B.2 - 1)) :
  B1 = (4, 1) :=
sorry

end translation_coordinates_l295_295393


namespace find_f_2006_l295_295309

-- Assuming an odd periodic function f with period 3(3x+1), defining the conditions.
def f : ℤ → ℤ := sorry -- Definition of f is not provided.

-- Conditions
axiom odd_function : ∀ x : ℤ, f (-x) = -f x
axiom period_3_function : ∀ x : ℤ, f (3 * x + 1) = f (3 * (x + 1) + 1)
axiom value_at_1 : f 1 = -1

-- Question: What is f(2006)?
theorem find_f_2006 : f 2006 = 1 := sorry

end find_f_2006_l295_295309


namespace union_complement_eq_l295_295715

open Set

variable (U A B : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

theorem union_complement_eq (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) :
  (complement U A) ∪ B = {0, 2, 4} :=
by
  rw [hU, hA, hB]
  sorry

end union_complement_eq_l295_295715


namespace necessary_sufficient_condition_l295_295258

theorem necessary_sufficient_condition 
  (a b : ℝ) : 
  a * |a + b| < |a| * (a + b) ↔ (a < 0 ∧ b > -a) :=
sorry

end necessary_sufficient_condition_l295_295258


namespace max_sides_of_convex_polygon_l295_295692

theorem max_sides_of_convex_polygon (n : ℕ) 
  (h_convex : n ≥ 3) 
  (h_angles: ∀ (a : Fin 4), (100 : ℝ) ≤ a.val) 
  : n ≤ 8 :=
sorry

end max_sides_of_convex_polygon_l295_295692


namespace find_m_l295_295865

-- Definitions for the lines and the condition of parallelism
def line1 (m : ℝ) (x y : ℝ): Prop := x + m * y + 6 = 0
def line2 (m : ℝ) (x y : ℝ): Prop := 3 * x + (m - 2) * y + 2 * m = 0

-- Condition for lines being parallel
def parallel_lines (m : ℝ) : Prop := 1 * (m - 2) - 3 * m = 0

-- Main formal statement
theorem find_m (m : ℝ) (h1 : ∀ x y, line1 m x y)
                (h2 : ∀ x y, line2 m x y)
                (h_parallel : parallel_lines m) : m = -1 :=
sorry

end find_m_l295_295865


namespace extreme_point_distance_number_of_roots_l295_295873

noncomputable def f (x : ℝ) : ℝ := real.log (x ^ 2 + 1)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 1 / (x ^ 2 - 1) + a
noncomputable def l (x y a : ℝ) : Prop := 2 * real.sqrt 2 * x + y + a + 5 = 0

theorem extreme_point_distance (a : ℝ) :
  let x := 0 in
  |2 * real.sqrt 2 * x + a + 5| = 3 →
  a = -2 ∨ a = -8 := sorry

theorem number_of_roots (a : ℝ) :
  let h (x : ℝ) := f x - g x a in
  if a < 1 then 
    set.count_roots h = 2 
  else if a = 1 then 
    set.count_roots h = 3 
  else 
    set.count_roots h = 4 := sorry

end extreme_point_distance_number_of_roots_l295_295873


namespace ramsey_example_l295_295620

theorem ramsey_example (P : Fin 10 → Fin 10 → Prop) :
  (∀ (i j k : Fin 10), i ≠ j → i ≠ k → j ≠ k → ¬(¬P i j ∧ ¬P j k ∧ ¬P k i))
  ∨ (∀ (i j k : Fin 10), i ≠ j → i ≠ k → j ≠ k → ¬(P i j ∧ P j k ∧ P k i)) →
  (∃ (i j k l : Fin 10), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ (P i j ∧ P j k ∧ P k l ∧ P i k ∧ P j l ∧ P i l))
  ∨ (∃ (i j k l : Fin 10), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ (¬P i j ∧ ¬P j k ∧ ¬P k l ∧ ¬P i k ∧ ¬P j l ∧ ¬P i l)) :=
by
  sorry

end ramsey_example_l295_295620


namespace median_of_consecutive_integers_l295_295783

theorem median_of_consecutive_integers (sum_n : ℤ) (n : ℤ) 
  (h1 : sum_n = 6^4) (h2 : n = 36) : 
  (sum_n / n) = 36 :=
by
  sorry

end median_of_consecutive_integers_l295_295783


namespace michael_watermelon_weight_l295_295904

theorem michael_watermelon_weight (m c j : ℝ) (h1 : c = 3 * m) (h2 : j = c / 2) (h3 : j = 12) : m = 8 :=
by
  sorry

end michael_watermelon_weight_l295_295904


namespace power_congruence_l295_295420

theorem power_congruence (a b n : ℕ) (h : a ≡ b [MOD n]) : a^n ≡ b^n [MOD n^2] :=
sorry

end power_congruence_l295_295420


namespace part1_purchase_price_part2_minimum_A_l295_295462

section
variables (x y m : ℝ)

-- Part 1: Purchase price per piece
theorem part1_purchase_price (h1 : 10 * x + 15 * y = 3600) (h2 : 25 * x + 30 * y = 8100) :
  x = 180 ∧ y = 120 :=
sorry

-- Part 2: Minimum number of model A bamboo mats
theorem part2_minimum_A (h3 : x = 180) (h4 : y = 120) 
    (h5 : (260 - x) * m + (180 - y) * (60 - m) ≥ 4400) : 
  m ≥ 40 :=
sorry
end

end part1_purchase_price_part2_minimum_A_l295_295462


namespace correct_calculation_l295_295460

-- Definitions of the equations
def option_A (a : ℝ) : Prop := a + 2 * a = 3 * a^2
def option_B (a b : ℝ) : Prop := (a^2 * b)^3 = a^6 * b^3
def option_C (a : ℝ) (m : ℕ) : Prop := (a^m)^2 = a^(m+2)
def option_D (a : ℝ) : Prop := a^3 * a^2 = a^6

-- The theorem that states option B is correct and others are incorrect
theorem correct_calculation (a b : ℝ) (m : ℕ) : 
  ¬ option_A a ∧ 
  option_B a b ∧ 
  ¬ option_C a m ∧ 
  ¬ option_D a :=
by sorry

end correct_calculation_l295_295460


namespace find_number_l295_295041

def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def first_digit_is_three (n : ℕ) : Prop :=
  n / 1000 = 3

def last_digit_is_five (n : ℕ) : Prop :=
  n % 10 = 5

theorem find_number :
  ∃ (x : ℕ), four_digit_number (x^2) ∧ first_digit_is_three (x^2) ∧ last_digit_is_five (x^2) ∧ x = 55 :=
sorry

end find_number_l295_295041


namespace find_c_l295_295980

/-
Given:
1. c and d are integers.
2. x^2 - x - 1 is a factor of cx^{18} + dx^{17} + x^2 + 1.
Show that c = -1597 under these conditions.

Assume we have the following Fibonacci number definitions:
F_16 = 987,
F_17 = 1597,
F_18 = 2584,
then:
Proof that c = -1597.
-/

noncomputable def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

theorem find_c (c d : ℤ) (h1 : c * 2584 + d * 1597 + 1 = 0) (h2 : c * 1597 + d * 987 + 2 = 0) :
  c = -1597 :=
by
  sorry

end find_c_l295_295980


namespace smallest_four_digit_div_by_35_l295_295146

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l295_295146


namespace Mike_ride_distance_l295_295613

theorem Mike_ride_distance 
  (M : ℕ)
  (total_cost_Mike : ℝ)
  (total_cost_Annie : ℝ)
  (h1 : total_cost_Mike = 4.50 + 0.30 * M)
  (h2: total_cost_Annie = 15.00)
  (h3: total_cost_Mike = total_cost_Annie) : 
  M = 35 := 
by
  sorry

end Mike_ride_distance_l295_295613


namespace find_a_l295_295863

theorem find_a (a x y : ℤ) (h_x : x = 1) (h_y : y = -3) (h_eq : a * x - y = 1) : a = -2 := by
  -- Proof skipped
  sorry

end find_a_l295_295863


namespace smallest_four_digit_divisible_by_35_l295_295153

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l295_295153


namespace algebraic_expression_value_l295_295564

theorem algebraic_expression_value {x : ℝ} (h : x * (x + 2) = 2023) : 2 * (x + 3) * (x - 1) - 2018 = 2022 := 
by 
  sorry

end algebraic_expression_value_l295_295564


namespace problem_statement_l295_295661

variable (F : ℕ → Prop)

theorem problem_statement (h1 : ∀ k : ℕ, F k → F (k + 1)) (h2 : ¬F 7) : ¬F 6 ∧ ¬F 5 := by
  sorry

end problem_statement_l295_295661


namespace steven_seeds_l295_295761

def average_seeds (fruit: String) : Nat :=
  match fruit with
  | "apple" => 6
  | "pear" => 2
  | "grape" => 3
  | "orange" => 10
  | "watermelon" => 300
  | _ => 0

def fruits := [("apple", 2), ("pear", 3), ("grape", 5), ("orange", 1), ("watermelon", 2)]

def required_seeds := 420

def total_seeds (fruit_list : List (String × Nat)) : Nat :=
  fruit_list.foldr (fun (fruit_qty : String × Nat) acc =>
    acc + (average_seeds fruit_qty.fst) * fruit_qty.snd) 0

theorem steven_seeds : total_seeds fruits - required_seeds = 223 := by
  sorry

end steven_seeds_l295_295761


namespace max_side_length_triangle_l295_295214

def triangle_with_max_side_length (a b c : ℕ) (ha : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hper : a + b + c = 30) : Prop :=
  a > b ∧ a > c ∧ a = 14

theorem max_side_length_triangle : ∃ a b c : ℕ, 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 30 ∧ a > b ∧ a > c ∧ a = 14 :=
sorry

end max_side_length_triangle_l295_295214


namespace g_decreasing_on_neg1_0_l295_295562

noncomputable def f (x : ℝ) : ℝ := 8 + 2 * x - x^2 
noncomputable def g (x : ℝ) : ℝ := f (2 - x^2)

theorem g_decreasing_on_neg1_0 : 
  ∀ x y : ℝ, -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ x < y → g y < g x :=
sorry

end g_decreasing_on_neg1_0_l295_295562


namespace find_remainder_l295_295372

-- Define the numbers
def a := 98134
def b := 98135
def c := 98136
def d := 98137
def e := 98138
def f := 98139

-- Theorem statement
theorem find_remainder :
  (a + b + c + d + e + f) % 9 = 3 :=
by {
  sorry
}

end find_remainder_l295_295372


namespace max_andy_l295_295451

def max_cookies_eaten_by_andy (total : ℕ) (k1 k2 a b c : ℤ) : Prop :=
  a + b + c = total ∧ b = 2 * a + 2 ∧ c = a - 3

theorem max_andy (total : ℕ) (a : ℤ) :
  (∀ b c, max_cookies_eaten_by_andy total 2 (-3) a b c) → a ≤ 7 :=
by
  intros H
  sorry

end max_andy_l295_295451


namespace alex_growth_rate_l295_295826

noncomputable def growth_rate_per_hour_hanging_upside_down
  (current_height : ℝ)
  (required_height : ℝ)
  (normal_growth_per_month : ℝ)
  (hanging_hours_per_month : ℝ)
  (answer : ℝ) : Prop :=
  current_height + 12 * normal_growth_per_month + 12 * hanging_hours_per_month * answer = required_height

theorem alex_growth_rate 
  (current_height : ℝ) 
  (required_height : ℝ)
  (normal_growth_per_month : ℝ)
  (hanging_hours_per_month : ℝ)
  (answer : ℝ) :
  current_height = 48 → 
  required_height = 54 → 
  normal_growth_per_month = 1/3 → 
  hanging_hours_per_month = 2 → 
  growth_rate_per_hour_hanging_upside_down current_height required_height normal_growth_per_month hanging_hours_per_month answer ↔ answer = 1/12 :=
by sorry

end alex_growth_rate_l295_295826


namespace fraction_of_married_men_l295_295042

theorem fraction_of_married_men (prob_single_woman : ℚ) (H : prob_single_woman = 3 / 7) :
  ∃ (fraction_married_men : ℚ), fraction_married_men = 4 / 11 :=
by
  -- Further proof steps would go here if required
  sorry

end fraction_of_married_men_l295_295042


namespace find_x_for_dot_product_l295_295716

theorem find_x_for_dot_product :
  let a : (ℝ × ℝ) := (1, -1)
  let b : (ℝ × ℝ) := (2, x)
  (a.1 * b.1 + a.2 * b.2 = 1) ↔ x = 1 :=
by
  sorry

end find_x_for_dot_product_l295_295716


namespace max_a_value_l295_295592

theorem max_a_value :
  ∀ (a x : ℝ), 
  (x - 1) * x - (a - 2) * (a + 1) ≥ 1 → a ≤ 3 / 2 := sorry

end max_a_value_l295_295592


namespace smallest_four_digit_divisible_by_35_l295_295150

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l295_295150


namespace range_of_a_l295_295390

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def B (x : ℝ) (a : ℝ) : Prop := x > a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, A x → B x a) → a < -2 :=
by
  sorry

end range_of_a_l295_295390


namespace third_month_sale_l295_295342

theorem third_month_sale (s3 : ℝ)
  (s1 s2 s4 s5 s6 : ℝ)
  (h1 : s1 = 2435)
  (h2 : s2 = 2920)
  (h4 : s4 = 3230)
  (h5 : s5 = 2560)
  (h6 : s6 = 1000)
  (average : (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 2500) :
  s3 = 2855 := 
by sorry

end third_month_sale_l295_295342


namespace tangent_line_circle_l295_295284

theorem tangent_line_circle (r : ℝ) (h : 0 < r) :
  (∀ x y : ℝ, x + y = r → x * x + y * y ≠ 4 * r) →
  r = 8 :=
by
  sorry

end tangent_line_circle_l295_295284


namespace range_of_m_l295_295064

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x - 2 < 0) → -8 < m ∧ m ≤ 0 :=
sorry

end range_of_m_l295_295064


namespace maximum_side_length_of_triangle_l295_295202

theorem maximum_side_length_of_triangle (a b c : ℕ) (h_diff: a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_perimeter: a + b + c = 30)
  (h_triangle_inequality_1: a + b > c) 
  (h_triangle_inequality_2: a + c > b) 
  (h_triangle_inequality_3: b + c > a) : 
  c ≤ 14 :=
sorry

end maximum_side_length_of_triangle_l295_295202


namespace min_major_axis_length_l295_295923

theorem min_major_axis_length (a b c : ℝ) (h_area : b * c = 1) (h_focal_relation : 2 * a = 2 * Real.sqrt (b^2 + c^2)) :
  2 * a = 2 * Real.sqrt 2 :=
by
  sorry

end min_major_axis_length_l295_295923


namespace theta1_gt_theta2_l295_295085

theorem theta1_gt_theta2 (a : ℝ) (b : ℝ) (θ1 θ2 : ℝ)
  (h_range_θ1 : 0 ≤ θ1 ∧ θ1 ≤ π) (h_range_θ2 : 0 ≤ θ2 ∧ θ2 ≤ π)
  (x1 x2 : ℝ) (hx1 : x1 = a * Real.cos θ1) (hx2 : x2 = a * Real.cos θ2)
  (h_less : x1 < x2) : θ1 > θ2 :=
by
  sorry

end theta1_gt_theta2_l295_295085


namespace store_credit_card_discount_proof_l295_295322

def full_price : ℕ := 125
def sale_discount_percentage : ℕ := 20
def coupon_discount : ℕ := 10
def total_savings : ℕ := 44

def sale_discount := full_price * sale_discount_percentage / 100
def price_after_sale_discount := full_price - sale_discount
def price_after_coupon := price_after_sale_discount - coupon_discount
def store_credit_card_discount := total_savings - sale_discount - coupon_discount
def discount_percentage_of_store_credit := (store_credit_card_discount * 100) / price_after_coupon

theorem store_credit_card_discount_proof : discount_percentage_of_store_credit = 10 := by
  sorry

end store_credit_card_discount_proof_l295_295322


namespace red_blue_beads_ratio_l295_295831

-- Definitions based on the conditions
def has_red_beads (betty : Type) := betty → ℕ
def has_blue_beads (betty : Type) := betty → ℕ

def betty : Type := Unit

-- Given conditions
def num_red_beads : has_red_beads betty := λ _ => 30
def num_blue_beads : has_blue_beads betty := λ _ => 20
def red_to_blue_ratio := 3 / 2

-- Theorem to prove the ratio
theorem red_blue_beads_ratio (R B: ℕ) (h_red : R = 30) (h_blue : B = 20) :
  (R / gcd R B) / (B / gcd R B ) = red_to_blue_ratio :=
by sorry

end red_blue_beads_ratio_l295_295831


namespace proposition_does_not_hold_at_2_l295_295667

variable (P : ℕ+ → Prop)
open Nat

theorem proposition_does_not_hold_at_2
  (h₁ : ¬ P 3)
  (h₂ : ∀ k : ℕ+, P k → P (k + 1)) :
  ¬ P 2 :=
by
  sorry

end proposition_does_not_hold_at_2_l295_295667


namespace sqrt_squared_l295_295222

theorem sqrt_squared (n : ℕ) (hn : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by
  sorry

example : (Real.sqrt 987654) ^ 2 = 987654 := 
  sqrt_squared 987654 (by norm_num)

end sqrt_squared_l295_295222


namespace tradesman_gain_l295_295823

-- Let's define a structure representing the tradesman's buying and selling operation.
structure Trade where
  true_value : ℝ
  defraud_rate : ℝ
  buy_price : ℕ
  sell_price : ℕ

theorem tradesman_gain (T : Trade) (H1 : T.defraud_rate = 0.2) (H2 : T.true_value = 100)
  (H3 : T.buy_price = T.true_value * (1 - T.defraud_rate))
  (H4 : T.sell_price = T.true_value * (1 + T.defraud_rate)) :
  ((T.sell_price - T.buy_price) / T.buy_price) * 100 = 50 := 
by
  sorry

end tradesman_gain_l295_295823


namespace part_A_part_C_part_D_l295_295241

noncomputable def f : ℝ → ℝ := sorry -- define f with given properties

-- Given conditions
axiom mono_incr_on_neg1_0 : ∀ x y : ℝ, -1 ≤ x → x ≤ 0 → -1 ≤ y → y ≤ 0 → x < y → f x < f y
axiom symmetry_about_1 : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom symmetry_about_2_0 : ∀ x : ℝ, f (2 + x) = -f (2 - x)

-- Prove the statements
theorem part_A : f 0 = f (-2) := sorry
theorem part_C : ∀ x y : ℝ, 2 < x → x < 3 → 2 < y → y < 3 → x < y → f x > f y := sorry
theorem part_D : f 2021 > f 2022 ∧ f 2022 > f 2023 := sorry

end part_A_part_C_part_D_l295_295241


namespace number_of_ah_tribe_residents_l295_295728

theorem number_of_ah_tribe_residents 
  (P A U : Nat) 
  (H1 : 16 < P) 
  (H2 : P ≤ 17) 
  (H3 : A + U = P) 
  (H4 : U = 2) : 
  A = 15 := 
by
  sorry

end number_of_ah_tribe_residents_l295_295728


namespace ellipse_problem_l295_295281

theorem ellipse_problem
  (a b : ℝ)
  (h₀ : 0 < a)
  (h₁ : 0 < b)
  (h₂ : a > b)
  (P Q : ℝ × ℝ)
  (ellipse_eq : ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1})
  (A : ℝ × ℝ)
  (hA : A = (a, 0))
  (R : ℝ × ℝ)
  (O : ℝ × ℝ)
  (hO : O = (0, 0))
  (AQ_OP_parallels : ∀ (x y : ℝ) (Qx Qy Px Py : ℝ), 
    x = a ∧ y = 0  ∧ (Qx, Qy) = (x, y) ↔ (O.1, O.2) = (Px, Py)
    ) :
  ∀ (AQ AR OP : ℝ), 
  AQ = dist (a, 0) Q → 
  AR = dist A R → 
  OP = dist O P → 
  |AQ * AR| / (OP ^ 2) = 2 :=
  sorry

end ellipse_problem_l295_295281


namespace estimated_probability_l295_295269

noncomputable def needle_intersection_probability : ℝ := 0.4

structure NeedleExperimentData :=
(distance_between_lines : ℝ)
(length_of_needle : ℝ)
(num_trials_intersections : List (ℕ × ℕ))
(intersection_frequencies : List ℝ)

def experiment_data : NeedleExperimentData :=
{ distance_between_lines := 5,
  length_of_needle := 3,
  num_trials_intersections := [(50, 23), (100, 48), (200, 83), (500, 207), (1000, 404), (2000, 802)],
  intersection_frequencies := [0.460, 0.480, 0.415, 0.414, 0.404, 0.401] }

theorem estimated_probability (data : NeedleExperimentData) :
  ∀ P : ℝ, (∀ n m, (n, m) ∈ data.num_trials_intersections → abs (m / n - P) < 0.1) → P = needle_intersection_probability :=
by
  intro P hP
  sorry

end estimated_probability_l295_295269


namespace sin_405_eq_sqrt_2_div_2_l295_295683

theorem sin_405_eq_sqrt_2_div_2 : sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt_2_div_2_l295_295683


namespace number_of_terms_in_arithmetic_sequence_l295_295717

-- Define the necessary conditions
def a := 2
def d := 5
def l := 1007  -- last term

-- Prove the number of terms in the sequence
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 202 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l295_295717


namespace octagon_area_half_l295_295071

theorem octagon_area_half (parallelogram : ℝ) (h_parallelogram : parallelogram = 1) : 
  (octagon_area : ℝ) =
  1 / 2 := 
  sorry

end octagon_area_half_l295_295071


namespace madeline_part_time_hours_l295_295285

theorem madeline_part_time_hours :
  let hours_in_class := 18
  let days_in_week := 7
  let hours_homework_per_day := 4
  let hours_sleeping_per_day := 8
  let leftover_hours := 46
  let hours_per_day := 24
  let total_hours_per_week := hours_per_day * days_in_week
  let total_homework_hours := hours_homework_per_day * days_in_week
  let total_sleeping_hours := hours_sleeping_per_day * days_in_week
  let total_other_activities := hours_in_class + total_homework_hours + total_sleeping_hours
  let available_hours := total_hours_per_week - total_other_activities
  available_hours - leftover_hours = 20 := by
  sorry

end madeline_part_time_hours_l295_295285


namespace shifted_parabola_expression_l295_295764

theorem shifted_parabola_expression (x y x' y' : ℝ) 
  (h_initial : y = (x + 2)^2 + 3)
  (h_shift_right : x' = x - 3)
  (h_shift_down : y' = y - 2)
  : y' = (x' - 1)^2 + 1 := 
sorry

end shifted_parabola_expression_l295_295764


namespace flea_returns_to_0_l295_295024

noncomputable def flea_return_probability (p : ℝ) : ℝ :=
if p = 1 then 0 else 1

theorem flea_returns_to_0 (p : ℝ) : 
  flea_return_probability p = (if p = 1 then 0 else 1) :=
by
  sorry

end flea_returns_to_0_l295_295024


namespace tunnel_length_correct_l295_295806

noncomputable def length_of_tunnel
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time_min : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let crossing_time_s := crossing_time_min * 60
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem tunnel_length_correct :
  length_of_tunnel 800 78 1 = 500.2 :=
by
  -- The proof will be filled later.
  sorry

end tunnel_length_correct_l295_295806


namespace minimum_distance_proof_l295_295573

noncomputable def minimum_distance_AB : ℝ :=
  let f (x : ℝ) := x^2 - Real.log x
  let x_min := Real.sqrt 2 / 2
  let min_dist := (5 + Real.log 2) / 4
  min_dist

theorem minimum_distance_proof :
  ∃ a : ℝ, a = minimum_distance_AB :=
by
  use (5 + Real.log 2) / 4
  sorry

end minimum_distance_proof_l295_295573


namespace exam_score_impossible_l295_295406

theorem exam_score_impossible (x y : ℕ) : 
  (5 * x + y = 97) ∧ (x + y ≤ 20) → false :=
by
  sorry

end exam_score_impossible_l295_295406


namespace sqrt_squared_l295_295223

theorem sqrt_squared (n : ℕ) (hn : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by
  sorry

example : (Real.sqrt 987654) ^ 2 = 987654 := 
  sqrt_squared 987654 (by norm_num)

end sqrt_squared_l295_295223


namespace determine_a_l295_295830

-- Given conditions
variable {a b : ℝ}
variable (h_neg : a < 0) (h_pos : b > 0) (h_max : ∀ x, -2 ≤ a * sin (b * x) ∧ a * sin (b * x) ≤ 2)

-- Statement to prove
theorem determine_a : a = -2 := by
  sorry

end determine_a_l295_295830


namespace bridge_length_l295_295524

theorem bridge_length
  (train_length : ℝ)
  (train_speed_km_hr : ℝ)
  (crossing_time_sec : ℝ)
  (train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600)
  (total_distance : ℝ := train_speed_m_s * crossing_time_sec)
  (bridge_length : ℝ := total_distance - train_length)
  (train_length_val : train_length = 110)
  (train_speed_km_hr_val : train_speed_km_hr = 36)
  (crossing_time_sec_val : crossing_time_sec = 24.198064154867613) :
  bridge_length = 131.98064154867613 :=
by
  sorry

end bridge_length_l295_295524


namespace quadratic_second_root_l295_295100

noncomputable def second_root (p q : ℝ) : ℝ :=
  -2 * p / (p - 2)

theorem quadratic_second_root (p q : ℝ) (h1 : (p + q) * 1^2 + (p - q) * 1 + p * q = 0) :
  ∃ r : ℝ, r = second_root p q :=
by 
  sorry

end quadratic_second_root_l295_295100


namespace correct_factorization_l295_295649

-- Definitions from conditions
def A: Prop := ∀ x y: ℝ, x^2 - 4*y^2 = (x + y) * (x - 4*y)
def B: Prop := ∀ x: ℝ, (x + 4) * (x - 4) = x^2 - 16
def C: Prop := ∀ x: ℝ, x^2 - 2*x + 1 = (x - 1)^2
def D: Prop := ∀ x: ℝ, x^2 - 8*x + 9 = (x - 4)^2 - 7

-- Goal is to prove that C is a correct factorization
theorem correct_factorization: C := by
  sorry

end correct_factorization_l295_295649


namespace avg_values_l295_295696

theorem avg_values (z : ℝ) : (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end avg_values_l295_295696


namespace mixed_fraction_product_l295_295521

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l295_295521


namespace large_pyramid_tiers_l295_295790

def surface_area_pyramid (n : ℕ) : ℕ :=
  4 * n^2 + 2 * n

theorem large_pyramid_tiers :
  (∃ n : ℕ, surface_area_pyramid n = 42) →
  (∃ n : ℕ, surface_area_pyramid n = 2352) →
  ∃ n : ℕ, surface_area_pyramid n = 2352 ∧ n = 24 :=
by
  sorry

end large_pyramid_tiers_l295_295790


namespace eight_natural_numbers_exist_l295_295407

theorem eight_natural_numbers_exist :
  ∃ (n : Fin 8 → ℕ), (∀ i j : Fin 8, i ≠ j → ¬(n i ∣ n j)) ∧ (∀ i j : Fin 8, i ≠ j → n i ∣ (n j * n j)) :=
by 
  sorry

end eight_natural_numbers_exist_l295_295407


namespace num_multiples_of_three_in_ap_l295_295771

variable (a : ℕ → ℚ)  -- Defining the arithmetic sequence

def first_term (a1 : ℚ) := a 1 = a1
def eighth_term (a8 : ℚ) := a 8 = a8
def general_term (d : ℚ) := ∀ n : ℕ, a n = 9 + (n - 1) * d
def multiple_of_three (n : ℕ) := ∃ k : ℕ, a n = 3 * k

theorem num_multiples_of_three_in_ap 
  (a : ℕ → ℚ)
  (h1 : first_term a 9)
  (h2 : eighth_term a 12) :
  ∃ n : ℕ, n = 288 ∧ ∃ l : ℕ → Prop, ∀ k : ℕ, l k → multiple_of_three a (k * 7 + 1) :=
sorry

end num_multiples_of_three_in_ap_l295_295771


namespace solve_fractions_l295_295485

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l295_295485


namespace bob_favorite_number_is_correct_l295_295532

def bob_favorite_number : ℕ :=
  99

theorem bob_favorite_number_is_correct :
  50 < bob_favorite_number ∧
  bob_favorite_number < 100 ∧
  bob_favorite_number % 11 = 0 ∧
  bob_favorite_number % 2 ≠ 0 ∧
  (bob_favorite_number / 10 + bob_favorite_number % 10) % 3 = 0 :=
by
  sorry

end bob_favorite_number_is_correct_l295_295532


namespace area_ratio_and_sum_l295_295273

def triangle_area_ratio (XY YZ ZX s t u : ℝ) :=
  1 - s * (1 - u) - t * (1 - s) - u * (1 - t)

theorem area_ratio_and_sum (s t u : ℝ) (XY YZ ZX : ℝ) (h1 : s + t + u = 3/4)
  (h2 : s^2 + t^2 + u^2 = 3/7) (hXY : XY = 14) (hYZ : YZ = 16) (hZX : ZX = 18) :
  let ratio := triangle_area_ratio XY YZ ZX s t u in
  ratio = 59/112 ∧ (59 + 112 = 171) :=
by
  sorry

end area_ratio_and_sum_l295_295273


namespace find_a_l295_295860

theorem find_a 
  (x y a : ℝ) 
  (hx : x = 1) 
  (hy : y = -3) 
  (h : a * x - y = 1) : 
  a = -2 := 
  sorry

end find_a_l295_295860


namespace angle_ratio_l295_295887

theorem angle_ratio (BP BQ BM: ℝ) (ABC: ℝ) (quadrisect : BP = ABC/4 ∧ BQ = ABC)
  (bisect : BM = (3/4) * ABC / 2):
  (BM / (ABC / 4 + ABC / 4)) = 1 / 6 := by
    sorry

end angle_ratio_l295_295887


namespace weight_measurement_l295_295358

theorem weight_measurement :
  ∀ (w : Set ℕ), w = {1, 3, 9, 27} → (∀ n ∈ w, ∃ k, k = n ∧ k ∈ w) →
  ∃ (num_sets : ℕ), num_sets = 41 := by
  intros w hw hcomb
  sorry

end weight_measurement_l295_295358


namespace number_of_sequences_l295_295197

theorem number_of_sequences : 
  let n : ℕ := 7
  let ones : ℕ := 5
  let twos : ℕ := 2
  let comb := Nat.choose
  (ones + twos = n) ∧  
  comb (ones + 1) twos + comb (ones + 1) (twos - 1) = 21 := 
  by sorry

end number_of_sequences_l295_295197


namespace tangerines_count_l295_295928

theorem tangerines_count (apples pears tangerines : ℕ)
  (h1 : apples = 45)
  (h2 : pears = apples - 21)
  (h3 : tangerines = pears + 18) :
  tangerines = 42 :=
by
  sorry

end tangerines_count_l295_295928


namespace coloring_count_in_3x3_grid_l295_295536

theorem coloring_count_in_3x3_grid (n m : ℕ) (h1 : n = 3) (h2 : m = 3) : 
  ∃ count : ℕ, count = 15 ∧ ∀ (cells : Finset (Fin n × Fin m)),
  (cells.card = 3 ∧ ∀ (c1 c2 : Fin n × Fin m), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 → 
  (c1.fst ≠ c2.fst ∧ c1.snd ≠ c2.snd)) → cells.card ∣ count :=
sorry

end coloring_count_in_3x3_grid_l295_295536


namespace find_integer_pairs_l295_295849

theorem find_integer_pairs :
  {p : ℤ × ℤ | p.1 * (p.1 + 1) * (p.1 + 7) * (p.1 + 8) = p.2^2} =
  {(1, 12), (1, -12), (-9, 12), (-9, -12), (0, 0), (-8, 0), (-4, -12), (-4, 12), (-1, 0), (-7, 0)} :=
sorry

end find_integer_pairs_l295_295849


namespace find_circle_radius_l295_295366

/-- Eight congruent copies of the parabola y = x^2 are arranged in the plane so that each vertex 
is tangent to a circle, and each parabola is tangent to its two neighbors at an angle of 45°.
Find the radius of the circle. -/

theorem find_circle_radius
  (r : ℝ)
  (h_tangent_to_circle : ∀ (x : ℝ), (x^2 + r) = x → x^2 - x + r = 0)
  (h_single_tangent_point : ∀ (x : ℝ), (x^2 - x + r = 0) → ((1 : ℝ)^2 - 4 * 1 * r = 0)) :
  r = 1/4 :=
by
  -- the proof would go here
  sorry

end find_circle_radius_l295_295366


namespace gloria_money_left_l295_295879

theorem gloria_money_left 
  (cost_of_cabin : ℕ) (cash : ℕ)
  (num_cypress_trees num_pine_trees num_maple_trees : ℕ)
  (price_per_cypress_tree price_per_pine_tree price_per_maple_tree : ℕ)
  (money_left : ℕ)
  (h_cost_of_cabin : cost_of_cabin = 129000)
  (h_cash : cash = 150)
  (h_num_cypress_trees : num_cypress_trees = 20)
  (h_num_pine_trees : num_pine_trees = 600)
  (h_num_maple_trees : num_maple_trees = 24)
  (h_price_per_cypress_tree : price_per_cypress_tree = 100)
  (h_price_per_pine_tree : price_per_pine_tree = 200)
  (h_price_per_maple_tree : price_per_maple_tree = 300)
  (h_money_left : money_left = (num_cypress_trees * price_per_cypress_tree + 
                                num_pine_trees * price_per_pine_tree + 
                                num_maple_trees * price_per_maple_tree + 
                                cash) - cost_of_cabin)
  : money_left = 350 :=
by
  sorry

end gloria_money_left_l295_295879


namespace paco_initial_cookies_l295_295616

-- Define the given conditions
def cookies_given : ℕ := 14
def cookies_eaten : ℕ := 10
def cookies_left : ℕ := 12

-- Proposition to prove: Paco initially had 36 cookies
theorem paco_initial_cookies : (cookies_given + cookies_eaten + cookies_left = 36) :=
by
  sorry

end paco_initial_cookies_l295_295616


namespace probability_correct_l295_295929

noncomputable def probability_of_at_least_one_black_ball 
  (black white total drawn : ℕ) : ℚ :=
  (Finset.card (Finset.Icc 1 black) * Finset.card (Finset.Icc 1 white) +
  Finset.card (Finset.Icc 1 black) * (Finset.card (Finset.Icc 1 black) - 1) / 2) / 
  (Finset.card (Finset.Icc 1 total) * (Finset.card (Finset.Icc 1 total) - 1) / 2)

theorem probability_correct :
  probability_of_at_least_one_black_ball 5 3 8 2 = (25/28 : ℚ) :=
  sorry

end probability_correct_l295_295929


namespace distance_interval_l295_295674

def distance_to_town (d : ℝ) : Prop :=
  ¬(d ≥ 8) ∧ ¬(d ≤ 7) ∧ ¬(d ≤ 6) ∧ ¬(d ≥ 9)

theorem distance_interval (d : ℝ) : distance_to_town d → d ∈ Set.Ioo 7 8 :=
by
  intro h
  have h1 : d < 8 := by sorry
  have h2 : d > 7 := by sorry
  rw [Set.mem_Ioo]
  exact ⟨h2, h1⟩

end distance_interval_l295_295674


namespace arithmetic_expression_equality_l295_295169

theorem arithmetic_expression_equality : 
  (1/4 : ℝ) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 * (1/4096) * 8192 = 64 := 
by
  sorry

end arithmetic_expression_equality_l295_295169


namespace kitty_cleaning_weeks_l295_295614

def time_spent_per_week (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust_furniture: ℕ) : ℕ :=
  pick_up + vacuum + clean_windows + dust_furniture

def total_weeks (total_time: ℕ) (time_per_week: ℕ) : ℕ :=
  total_time / time_per_week

theorem kitty_cleaning_weeks
  (pick_up_time : ℕ := 5)
  (vacuum_time : ℕ := 20)
  (clean_windows_time : ℕ := 15)
  (dust_furniture_time : ℕ := 10)
  (total_cleaning_time : ℕ := 200)
  : total_weeks total_cleaning_time (time_spent_per_week pick_up_time vacuum_time clean_windows_time dust_furniture_time) = 4 :=
by
  sorry

end kitty_cleaning_weeks_l295_295614


namespace water_volume_in_B_when_A_is_0_point_4_l295_295910

noncomputable def pool_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

noncomputable def valve_rate (volume time : ℝ) : ℝ :=
  volume / time

theorem water_volume_in_B_when_A_is_0_point_4 :
  ∀ (length width depth : ℝ)
    (time_A_fill time_A_to_B : ℝ)
    (depth_A_target : ℝ),
    length = 3 → width = 2 → depth = 1.2 →
    time_A_fill = 18 → time_A_to_B = 24 →
    depth_A_target = 0.4 →
    pool_volume length width depth = 7.2 →
    valve_rate 7.2 time_A_fill = 0.4 →
    valve_rate 7.2 time_A_to_B = 0.3 →
    ∃ (time_required : ℝ),
    time_required = 24 →
    (valve_rate 7.2 time_A_to_B * time_required = 7.2) :=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end water_volume_in_B_when_A_is_0_point_4_l295_295910


namespace mart_income_percentage_juan_l295_295287

-- Define the conditions
def TimIncomeLessJuan (J T : ℝ) : Prop := T = 0.40 * J
def MartIncomeMoreTim (T M : ℝ) : Prop := M = 1.60 * T

-- Define the proof problem
theorem mart_income_percentage_juan (J T M : ℝ) 
  (h1 : TimIncomeLessJuan J T) 
  (h2 : MartIncomeMoreTim T M) :
  M = 0.64 * J := 
  sorry

end mart_income_percentage_juan_l295_295287


namespace sphere_volume_l295_295471

theorem sphere_volume (A : ℝ) (d : ℝ) (V : ℝ) : 
    (A = 2 * Real.pi) →  -- Cross-sectional area is 2π cm²
    (d = 1) →            -- Distance from center to cross-section is 1 cm
    (V = 4 * Real.sqrt 3 * Real.pi) :=  -- Volume of sphere is 4√3 π cm³
by 
  intros hA hd
  sorry

end sphere_volume_l295_295471


namespace Ashutosh_time_to_complete_job_l295_295915

noncomputable def SureshWorkRate : ℝ := 1 / 15
noncomputable def AshutoshWorkRate (A : ℝ) : ℝ := 1 / A
noncomputable def SureshWorkIn9Hours : ℝ := 9 * SureshWorkRate

theorem Ashutosh_time_to_complete_job (A : ℝ) :
  (1 - SureshWorkIn9Hours) * AshutoshWorkRate A = 14 / 35 →
  A = 35 :=
by
  sorry

end Ashutosh_time_to_complete_job_l295_295915


namespace Ariana_running_time_l295_295622

theorem Ariana_running_time
  (time_Sadie : ℝ)
  (speed_Sadie : ℝ)
  (speed_Ariana : ℝ)
  (speed_Sarah : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (distance_Sadie := speed_Sadie * time_Sadie)
  (time_Ariana_Sarah := total_time - time_Sadie)
  (distance_Ariana_Sarah := total_distance - distance_Sadie) :
  (6 * (time_Ariana_Sarah - (11 - 6 * (time_Ariana_Sarah / (speed_Ariana + (4 / speed_Sarah)))))
  = (0.5 : ℝ)) :=
by
  sorry

end Ariana_running_time_l295_295622


namespace sin_double_angle_l295_295561

theorem sin_double_angle (α : ℝ) (h : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = -2 / 3 := 
sorry

end sin_double_angle_l295_295561


namespace ellipse_equation_l295_295735

theorem ellipse_equation {a b : ℝ} 
  (center_origin : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → x + y = 0)
  (foci_on_x : ∀ c : ℝ, c = a / 2)
  (perimeter_triangle : ∀ A B : ℝ, A + B + 2 * c = 16) :
  a = 4 ∧ b^2 = 12 → (∀ x y : ℝ, x^2/16 + y^2/12 = 1) :=
by
  sorry

end ellipse_equation_l295_295735


namespace B_days_solve_l295_295189

noncomputable def combined_work_rate (A_rate B_rate C_rate : ℝ) : ℝ := A_rate + B_rate + C_rate
noncomputable def A_rate : ℝ := 1 / 6
noncomputable def C_rate : ℝ := 1 / 7.5
noncomputable def combined_rate : ℝ := 1 / 2

theorem B_days_solve : ∃ (B_days : ℝ), combined_work_rate A_rate (1 / B_days) C_rate = combined_rate ∧ B_days = 5 :=
by
  use 5
  rw [←inv_div] -- simplifying the expression of 1/B_days
  have : ℝ := sorry -- steps to cancel and simplify, proving the equality
  sorry

end B_days_solve_l295_295189


namespace squirrel_travel_distance_l295_295034

theorem squirrel_travel_distance
  (height: ℝ)
  (circumference: ℝ)
  (vertical_rise: ℝ)
  (num_circuits: ℝ):
  height = 25 →
  circumference = 3 →
  vertical_rise = 5 →
  num_circuits = height / vertical_rise →
  (num_circuits * circumference) ^ 2 + height ^ 2 = 850 :=
by
  sorry

end squirrel_travel_distance_l295_295034


namespace largest_multiple_of_8_less_than_100_l295_295005

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), (n % 8 = 0) ∧ (n < 100) ∧ (∀ m : ℕ, (m % 8 = 0) ∧ (m < 100) → m ≤ n) :=
begin
  use 96,
  split,
  { -- 96 is a multiple of 8
    exact nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 96),
  },
  split,
  { -- 96 is less than 100
    norm_num,
  },
  { -- 96 is the largest multiple of 8 less than 100
    intros m hm,
    obtain ⟨k, rfl⟩ := (nat.dvd_iff_mod_eq_zero.mp hm.1),
    have : k ≤ 12, by linarith,
    linarith [mul_le_mul (zero_le _ : (0 : ℕ) ≤ 8) this (zero_le _ : (0 : ℕ) ≤ 12) (zero_le _ : (0 : ℕ) ≤ 8)],
  },
end

end largest_multiple_of_8_less_than_100_l295_295005


namespace find_geometric_sequence_values_l295_295859

structure GeometricSequence (a b c d : ℝ) : Prop where
  ratio1 : b / a = c / b
  ratio2 : c / b = d / c

theorem find_geometric_sequence_values (x u v y : ℝ)
    (h1 : x + y = 20)
    (h2 : u + v = 34)
    (h3 : x^2 + u^2 + v^2 + y^2 = 1300) :
    (GeometricSequence x u v y ∧ ((x = 16 ∧ u = 4 ∧ v = 32 ∧ y = 2) ∨ (x = 4 ∧ u = 16 ∧ v = 2 ∧ y = 32))) :=
by
  sorry

end find_geometric_sequence_values_l295_295859


namespace find_larger_number_l295_295615

theorem find_larger_number (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : y = 33 :=
by
  sorry

end find_larger_number_l295_295615


namespace abs_difference_l295_295898

theorem abs_difference (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 8) : 
  |a - b| = 2 * Real.sqrt 10 :=
by
  sorry

end abs_difference_l295_295898


namespace mike_gave_4_marbles_l295_295612

noncomputable def marbles_given (original_marbles : ℕ) (remaining_marbles : ℕ) : ℕ :=
  original_marbles - remaining_marbles

theorem mike_gave_4_marbles (original_marbles remaining_marbles given_marbles : ℕ) 
  (h1 : original_marbles = 8) (h2 : remaining_marbles = 4) (h3 : given_marbles = marbles_given original_marbles remaining_marbles) : given_marbles = 4 :=
by
  sorry

end mike_gave_4_marbles_l295_295612


namespace value_a7_l295_295593

variables {a : ℕ → ℝ}

-- Condition 1: Arithmetic sequence where each term is non-zero
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variable (h1 : arithmetic_sequence a)
-- Condition 2: 2a_3 - a_1^2 + 2a_11 = 0
variable (h2 : 2 * a 3 - (a 1)^2 + 2 * a 11 = 0)
-- Condition 3: a_3 + a_11 = 2a_7
variable (h3 : a 3 + a 11 = 2 * a 7)

theorem value_a7 : a 7 = 4 := by
  sorry

end value_a7_l295_295593


namespace cricket_innings_l295_295916

theorem cricket_innings (n : ℕ) (h1 : (36 * n) / n = 36) (h2 : (36 * n + 80) / (n + 1) = 40) : n = 10 := by
  -- The proof goes here
  sorry

end cricket_innings_l295_295916


namespace problem_proof_l295_295172

def problem_statement : Prop :=
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 * (1 / 4096) * 8192 = 64

theorem problem_proof : problem_statement := by
  sorry

end problem_proof_l295_295172


namespace reflected_ray_eq_l295_295343

theorem reflected_ray_eq:
  ∀ (x y : ℝ), 
    (3 * x + 4 * y - 18 = 0) ∧ (3 * x + 2 * y - 12 = 0) →
    63 * x + 16 * y - 174 = 0 :=
by
  intro x y
  intro h
  sorry

end reflected_ray_eq_l295_295343


namespace semicircle_radius_l295_295316

noncomputable def radius_of_semicircle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem semicircle_radius (P : ℝ) (hP : P = 180) : radius_of_semicircle P = 180 / (Real.pi + 2) :=
by
  sorry

end semicircle_radius_l295_295316


namespace find_X_l295_295652

theorem find_X :
  (15.2 * 0.25 - 48.51 / 14.7) / X = ((13 / 44 - 2 / 11 - 5 / 66) / (5 / 2) * (6 / 5)) / (3.2 + 0.8 * (5.5 - 3.25)) ->
  X = 137.5 :=
by
  intro h
  sorry

end find_X_l295_295652


namespace roots_reciprocal_l295_295009

theorem roots_reciprocal (a b c x1 x2 x3 x4 : ℝ) 
  (h1 : a ≠ 0)
  (h2 : c ≠ 0)
  (hx1 : a * x1^2 + b * x1 + c = 0)
  (hx2 : a * x2^2 + b * x2 + c = 0)
  (hx3 : c * x3^2 + b * x3 + a = 0)
  (hx4 : c * x4^2 + b * x4 + a = 0) :
  (x3 = 1/x1 ∧ x4 = 1/x2) :=
  sorry

end roots_reciprocal_l295_295009


namespace minimal_total_distance_l295_295405

variable (A B : ℝ) -- Coordinates of houses A and B on a straight road
variable (h_dist : B - A = 50) -- The distance between A and B is 50 meters

-- Define a point X on the road
variable (X : ℝ)

-- Define the function that calculates the total distance from point X to A and B
def total_distance (A B X : ℝ) := abs (X - A) + abs (X - B)

-- The theorem stating that the total distance is minimized if X lies on the line segment AB
theorem minimal_total_distance : A ≤ X ∧ X ≤ B ↔ total_distance A B X = B - A :=
by
  sorry

end minimal_total_distance_l295_295405


namespace largest_common_term_lt_300_l295_295439

theorem largest_common_term_lt_300 :
  ∃ a : ℕ, a < 300 ∧ (∃ n : ℤ, a = 4 + 5 * n) ∧ (∃ m : ℤ, a = 3 + 7 * m) ∧ ∀ b : ℕ, b < 300 → (∃ n : ℤ, b = 4 + 5 * n) → (∃ m : ℤ, b = 3 + 7 * m) → b ≤ a :=
sorry

end largest_common_term_lt_300_l295_295439


namespace train_speed_kmph_l295_295199

def train_length : ℝ := 360
def bridge_length : ℝ := 140
def time_to_pass : ℝ := 40
def mps_to_kmph (speed : ℝ) : ℝ := speed * 3.6

theorem train_speed_kmph : mps_to_kmph ((train_length + bridge_length) / time_to_pass) = 45 := 
by {
  sorry
}

end train_speed_kmph_l295_295199


namespace Lara_age_in_10_years_l295_295414

theorem Lara_age_in_10_years (Lara_age_7_years_ago : ℕ) (h1 : Lara_age_7_years_ago = 9) : 
  Lara_age_7_years_ago + 7 + 10 = 26 :=
by
  rw [h1]
  norm_num
  sorry

end Lara_age_in_10_years_l295_295414


namespace dress_shirt_cost_l295_295299

theorem dress_shirt_cost (x : ℝ) :
  let total_cost_before_discounts := 4 * x + 2 * 40 + 150 + 2 * 30
  let total_cost_after_store_discount := total_cost_before_discounts * 0.8
  let total_cost_after_coupon := total_cost_after_store_discount * 0.9
  total_cost_after_coupon = 252 → x = 15 :=
by
  let total_cost_before_discounts := 4 * x + 2 * 40 + 150 + 2 * 30
  let total_cost_after_store_discount := total_cost_before_discounts * 0.8
  let total_cost_after_coupon := total_cost_after_store_discount * 0.9
  intro h
  sorry

end dress_shirt_cost_l295_295299


namespace line_passes_through_circle_center_l295_295864

theorem line_passes_through_circle_center (a : ℝ) : 
  ∀ x y : ℝ, (x, y) = (a, 2*a) → (x - a)^2 + (y - 2*a)^2 = 1 → 2*x - y = 0 :=
by
  sorry

end line_passes_through_circle_center_l295_295864


namespace impossible_rearrange_reverse_l295_295636

theorem impossible_rearrange_reverse :
  ∀ (tokens : ℕ → ℕ), 
    (∀ i, (i % 2 = 1 ∧ i < 99 → tokens i = tokens (i + 2)) 
      ∧ (i % 2 = 0 ∧ i < 99 → tokens i = tokens (i + 2))) → ¬(∀ i, tokens i = 100 + 1 - tokens (i - 1)) :=
by
  intros tokens h
  sorry

end impossible_rearrange_reverse_l295_295636


namespace sin_405_eq_sqrt2_div_2_l295_295686

theorem sin_405_eq_sqrt2_div_2 :
  Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt2_div_2_l295_295686


namespace isosceles_triangle_perimeter_l295_295272

theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h1 : (a = 2 ∧ b = 4 ∧ c = 4) ∨ (a = 4 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) 
  (h2 : a + b > c ∧ a + c > b ∧ b + c > a) : a + b + c = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l295_295272


namespace calories_per_cookie_l295_295020

theorem calories_per_cookie :
  ∀ (cookies_per_bag bags_per_box total_calories total_number_cookies : ℕ),
  cookies_per_bag = 20 →
  bags_per_box = 4 →
  total_calories = 1600 →
  total_number_cookies = cookies_per_bag * bags_per_box →
  (total_calories / total_number_cookies) = 20 :=
by sorry

end calories_per_cookie_l295_295020


namespace mixed_fraction_product_example_l295_295479

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l295_295479


namespace find_m_value_l295_295283

theorem find_m_value (a m : ℤ) (h : a ≠ 1) (hx : ∀ x y : ℤ, (x > 0) ∧ (y > 0) ∧ (a - 1) * x^2 - m * x + a = 0 ∧ (a - 1) * y^2 - m * y + a = 0) : m = 3 :=
sorry

end find_m_value_l295_295283


namespace original_proposition_contrapositive_converse_inverse_negation_false_l295_295914

variable {a b c : ℝ}

-- Original Proposition
theorem original_proposition (h : a < b) : a + c < b + c :=
sorry

-- Contrapositive
theorem contrapositive (h : a + c >= b + c) : a >= b :=
sorry

-- Converse
theorem converse (h : a + c < b + c) : a < b :=
sorry

-- Inverse
theorem inverse (h : a >= b) : a + c >= b + c :=
sorry

-- Negation is false
theorem negation_false (h : a < b) : ¬ (a + c >= b + c) :=
sorry

end original_proposition_contrapositive_converse_inverse_negation_false_l295_295914


namespace unique_natural_number_l295_295846

theorem unique_natural_number (n a b : ℕ) (h1 : a ≠ b) 
(h2 : digits_in_reverse_order (n^a + 1) (n^b + 1)) : n = 3 := sorry

end unique_natural_number_l295_295846


namespace Q_has_exactly_one_negative_root_l295_295225

def Q (x : ℝ) : ℝ := x^7 + 5 * x^5 + 5 * x^4 - 6 * x^3 - 2 * x^2 - 10 * x + 12

theorem Q_has_exactly_one_negative_root :
  ∃! r : ℝ, r < 0 ∧ Q r = 0 := sorry

end Q_has_exactly_one_negative_root_l295_295225


namespace sufficient_condition_of_square_inequality_l295_295257

variables (a b : ℝ)

theorem sufficient_condition_of_square_inequality (ha : a > 0) (hb : b > 0) (h : a > b) : a^2 > b^2 :=
by {
  sorry
}

end sufficient_condition_of_square_inequality_l295_295257


namespace fg_of_3_l295_295395

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := x^2 - 3 * x

theorem fg_of_3 : f (g 3) = -2 := by
  sorry

end fg_of_3_l295_295395


namespace find_numbers_l295_295437

theorem find_numbers :
  ∃ (a b c d : ℕ), 
  (a + 2 = 22) ∧ 
  (b - 2 = 22) ∧ 
  (c * 2 = 22) ∧ 
  (d / 2 = 22) ∧ 
  (a + b + c + d = 99) :=
sorry

end find_numbers_l295_295437


namespace meaningful_expression_range_l295_295321

theorem meaningful_expression_range (x : ℝ) : (3 * x + 9 ≥ 0) ∧ (x ≠ 2) ↔ (x ≥ -3 ∧ x ≠ 2) := by
  sorry

end meaningful_expression_range_l295_295321


namespace number_of_pickup_trucks_l295_295066

theorem number_of_pickup_trucks 
  (cars : ℕ) (bicycles : ℕ) (tricycles : ℕ) (total_tires : ℕ)
  (tires_per_car : ℕ) (tires_per_bicycle : ℕ) (tires_per_tricycle : ℕ) (tires_per_pickup : ℕ) :
  cars = 15 →
  bicycles = 3 →
  tricycles = 1 →
  total_tires = 101 →
  tires_per_car = 4 →
  tires_per_bicycle = 2 →
  tires_per_tricycle = 3 →
  tires_per_pickup = 4 →
  ((total_tires - (cars * tires_per_car + bicycles * tires_per_bicycle + tricycles * tires_per_tricycle)) / tires_per_pickup) = 8 :=
by
  sorry

end number_of_pickup_trucks_l295_295066


namespace Danica_additional_cars_l295_295970

theorem Danica_additional_cars (num_cars : ℕ) (cars_per_row : ℕ) (current_cars : ℕ) 
  (h_cars_per_row : cars_per_row = 8) (h_current_cars : current_cars = 35) :
  ∃ n, num_cars = 5 ∧ n = 40 ∧ n - current_cars = num_cars := 
by
  sorry

end Danica_additional_cars_l295_295970


namespace restore_original_problem_l295_295491

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l295_295491


namespace no_repetition_five_digit_count_l295_295643

theorem no_repetition_five_digit_count (digits : Finset ℕ) (count : Nat) :
  digits = {0, 1, 2, 3, 4, 5} →
  (∀ n ∈ digits, 0 ≤ n ∧ n ≤ 5) →
  (∃ numbers : Finset ℕ, 
    (∀ x ∈ numbers, (x / 100) % 10 ≠ 3 ∧ x % 5 = 0 ∧ x < 100000 ∧ x ≥ 10000) ∧
    (numbers.card = count)) →
  count = 174 :=
by
  sorry

end no_repetition_five_digit_count_l295_295643


namespace trigonometric_expression_l295_295585

open Real

theorem trigonometric_expression (α β : ℝ) (h : cos α ^ 2 = cos β ^ 2) :
  (sin β ^ 2 / sin α + cos β ^ 2 / cos α = sin α + cos α ∨ sin β ^ 2 / sin α + cos β ^ 2 / cos α = -sin α + cos α) :=
sorry

end trigonometric_expression_l295_295585


namespace x_1998_mod_1998_l295_295103

theorem x_1998_mod_1998 (λ : ℝ) (x : ℕ → ℝ)
  (hλ_eq : λ^2 - 1998 * λ - 1 = 0)
  (hx0 : x 0 = 1)
  (hx : ∀ n, x (n + 1) = Real.floor (λ * x n)) :
  x 1998 % 1998 = 0 :=
sorry

end x_1998_mod_1998_l295_295103


namespace log_ordering_l295_295416

noncomputable def P : ℝ := Real.log 3 / Real.log 2
noncomputable def Q : ℝ := Real.log 2 / Real.log 3
noncomputable def R : ℝ := Real.log (Real.log 2 / Real.log 3) / Real.log 2

theorem log_ordering (P Q R : ℝ) (h₁ : P = Real.log 3 / Real.log 2)
  (h₂ : Q = Real.log 2 / Real.log 3) (h₃ : R = Real.log (Real.log 2 / Real.log 3) / Real.log 2) :
  R < Q ∧ Q < P := by
  sorry

end log_ordering_l295_295416


namespace no_integer_solutions_for_mn_squared_eq_1980_l295_295113

theorem no_integer_solutions_for_mn_squared_eq_1980 :
  ¬ ∃ m n : ℤ, m^2 + n^2 = 1980 := 
sorry

end no_integer_solutions_for_mn_squared_eq_1980_l295_295113


namespace sequence_v_20_l295_295537

noncomputable def sequence_v : ℕ → ℝ → ℝ
| 0, b => b
| (n + 1), b => - (2 / (sequence_v n b + 2))

theorem sequence_v_20 (b : ℝ) (hb : 0 < b) : sequence_v 20 b = -(2 / (b + 2)) :=
by
  sorry

end sequence_v_20_l295_295537


namespace projections_of_opposite_sides_equal_l295_295428

-- The theorem statement requires us to define cyclic quadrilateral, its sides, projections and then show their equality.
variable (A B C D O P Q : Point)
variable (circle : Circle) (cyclic_quadrilateral : Quadrilateral)
variable [is_cyclic_quadrilateral : isCyclicQuadrilateral cyclic_quadrilateral]
variable (diameter_AC : isDiameter circle A C)
variable (proj_AB_on_BD : projectionLength A B D P)
variable (proj_CD_on_BD : projectionLength C D B Q)

theorem projections_of_opposite_sides_equal :
  isCyclicQuadrilateral cyclic_quadrilateral →
  isDiameter circle A C →
  projectionLength A B D P →
  projectionLength C D B Q →
  A = C → -- points A and C are endpoints of the diameter
  P = Q := sorry

end projections_of_opposite_sides_equal_l295_295428


namespace restore_fractions_l295_295513

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l295_295513


namespace triangle_altitude_l295_295588

theorem triangle_altitude
  (base : ℝ) (height : ℝ) (side : ℝ)
  (h_base : base = 6)
  (h_side : side = 6)
  (area_triangle : ℝ) (area_square : ℝ)
  (h_area_square : area_square = side ^ 2)
  (h_area_equal : area_triangle = area_square)
  (h_area_triangle : area_triangle = (base * height) / 2) :
  height = 12 := 
by
  sorry

end triangle_altitude_l295_295588


namespace original_area_of_circle_l295_295814

theorem original_area_of_circle
  (A₀ : ℝ) -- original area
  (r₀ r₁ : ℝ) -- original and new radius
  (π : ℝ := 3.14)
  (h_area : A₀ = π * r₀^2)
  (h_area_increase : π * r₁^2 = 9 * A₀)
  (h_circumference_increase : 2 * π * r₁ - 2 * π * r₀ = 50.24) :
  A₀ = 50.24 :=
by
  sorry

end original_area_of_circle_l295_295814


namespace bicycle_helmet_lock_costs_l295_295187

-- Given total cost, relationships between costs, and the specific costs
theorem bicycle_helmet_lock_costs (H : ℝ) (bicycle helmet lock : ℝ) 
  (h1 : bicycle = 5 * H) 
  (h2 : helmet = H) 
  (h3 : lock = H / 2)
  (total_cost : bicycle + helmet + lock = 360) :
  H = 55.38 ∧ bicycle = 276.90 ∧ lock = 27.72 :=
by 
  -- The proof would go here
  sorry

end bicycle_helmet_lock_costs_l295_295187


namespace intersection_conditions_l295_295131

-- Define the conditions
variables (c : ℝ) (k : ℝ) (m : ℝ) (n : ℝ) (p : ℝ)

-- Distance condition
def distance_condition (k : ℝ) (m : ℝ) (n : ℝ) (c : ℝ) : Prop :=
  (abs ((k^2 + 8 * k + c) - (m * k + n)) = 4)

-- Line passing through point (2, 7)
def passes_through_point (m : ℝ) (n : ℝ) : Prop :=
  (7 = 2 * m + n)

-- Definition of discriminants
def discriminant_1 (m : ℝ) (c : ℝ) (n : ℝ) : ℝ :=
  ((8 - m)^2 - 4 * (c - n - 4))

def discriminant_2 (m : ℝ) (c : ℝ) (n : ℝ) : ℝ :=
  ((8 - m)^2 - 4 * (c - n + 4))

-- Statement of the problem
theorem intersection_conditions (h₁ : n ≠ 0)
  (h₂ : passes_through_point m n)
  (h₃ : distance_condition k m n c)
  (h₄ : (discriminant_1 m c n = 0 ∨ discriminant_1 m c n < 0))
  (h₅ : (discriminant_2 m c n < 0)) :
  ∃ m n, n = 7 - 2 * m ∧ distance_condition k m n c :=
sorry

end intersection_conditions_l295_295131


namespace arithmetic_expression_equality_l295_295170

theorem arithmetic_expression_equality : 
  (1/4 : ℝ) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 * (1/4096) * 8192 = 64 := 
by
  sorry

end arithmetic_expression_equality_l295_295170


namespace prob_diff_fruit_correct_l295_295676

noncomputable def prob_same_all_apple : ℝ := (0.4)^3
noncomputable def prob_same_all_orange : ℝ := (0.3)^3
noncomputable def prob_same_all_banana : ℝ := (0.2)^3
noncomputable def prob_same_all_grape : ℝ := (0.1)^3

noncomputable def prob_same_fruit_all_day : ℝ := 
  prob_same_all_apple + prob_same_all_orange + prob_same_all_banana + prob_same_all_grape

noncomputable def prob_diff_fruit (prob_same : ℝ) : ℝ := 1 - prob_same

theorem prob_diff_fruit_correct :
  prob_diff_fruit prob_same_fruit_all_day = 0.9 :=
by
  sorry

end prob_diff_fruit_correct_l295_295676


namespace tangent_line_to_parabola_l295_295987

theorem tangent_line_to_parabola (l : ℝ → ℝ) (y : ℝ) (x : ℝ)
  (passes_through_P : l (-2) = 0)
  (intersects_once : ∃! x, (l x)^2 = 8*x) :
  (l = fun x => 0) ∨ (l = fun x => x + 2) ∨ (l = fun x => -x - 2) :=
sorry

end tangent_line_to_parabola_l295_295987


namespace eq_three_div_x_one_of_eq_l295_295586

theorem eq_three_div_x_one_of_eq (x : ℝ) (hx : 1 - 6 / x + 9 / (x ^ 2) = 0) : (3 / x) = 1 :=
sorry

end eq_three_div_x_one_of_eq_l295_295586


namespace problem_solution_l295_295935

def eq_A (x : ℝ) : Prop := 2 * x = 7
def eq_B (x y : ℝ) : Prop := x^2 + y = 5
def eq_C (x : ℝ) : Prop := x = 1 / x + 1
def eq_D (x : ℝ) : Prop := x^2 + x = 4

def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ a * x^2 + b * x + c = 0

theorem problem_solution : is_quadratic eq_D := by
  sorry

end problem_solution_l295_295935


namespace exists_sum_of_two_squares_l295_295754

theorem exists_sum_of_two_squares (n : ℤ) (h : n > 10000) : ∃ m : ℤ, (∃ a b : ℤ, m = a^2 + b^2) ∧ 0 < m - n ∧ m - n < 3 * n^(1/4) :=
by
  sorry

end exists_sum_of_two_squares_l295_295754


namespace sara_bought_cards_l295_295912

-- Definition of the given conditions
def initial_cards : ℕ := 39
def torn_cards : ℕ := 9
def remaining_cards_after_sale : ℕ := 15

-- Derived definition: Number of good cards before selling to Sara
def good_cards_before_selling : ℕ := initial_cards - torn_cards

-- The statement we need to prove
theorem sara_bought_cards : good_cards_before_selling - remaining_cards_after_sale = 15 :=
by
  sorry

end sara_bought_cards_l295_295912


namespace min_vertical_segment_length_l295_295440

noncomputable def vertical_segment_length (x : ℝ) : ℝ :=
  abs (|x| - (-x^2 - 4*x - 3))

theorem min_vertical_segment_length :
  ∃ x : ℝ, vertical_segment_length x = 3 / 4 :=
by
  sorry

end min_vertical_segment_length_l295_295440


namespace maximize_area_l295_295819

variable (x : ℝ)
def fence_length : ℝ := 240 - 2 * x
def area (x : ℝ) : ℝ := x * fence_length x

theorem maximize_area : fence_length 60 = 120 :=
  sorry

end maximize_area_l295_295819


namespace solve_fractions_l295_295481

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l295_295481


namespace sum_of_digits_l295_295266

variable {w x y z : ℕ}

theorem sum_of_digits :
  (w + x + y + z = 20) ∧ w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  (y + w = 11) ∧ (x + y = 9) ∧ (w + z = 10) :=
by
  sorry

end sum_of_digits_l295_295266


namespace median_of_36_consecutive_integers_l295_295785

theorem median_of_36_consecutive_integers (x : ℤ) (sum_eq : (∑ i in finset.range 36, (x + i)) = 6^4) : (17 + 18) / 2 = 36 :=
by
  -- Proof goes here
  sorry

end median_of_36_consecutive_integers_l295_295785


namespace mixed_fraction_product_l295_295518

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l295_295518


namespace least_number_to_subtract_l295_295057

theorem least_number_to_subtract (x : ℕ) (h : x = 1234567890) : ∃ n, x - n = 5 := 
  sorry

end least_number_to_subtract_l295_295057


namespace largest_multiple_of_8_less_than_100_l295_295002

theorem largest_multiple_of_8_less_than_100 : ∃ n, n < 100 ∧ n % 8 = 0 ∧ ∀ m, m < 100 ∧ m % 8 = 0 → m ≤ n :=
begin
  use 96,
  split,
  { -- prove 96 < 100
    norm_num,
  },
  split,
  { -- prove 96 is a multiple of 8
    norm_num,
  },
  { -- prove 96 is the largest such multiple
    intros m hm,
    cases hm with h1 h2,
    have h3 : m / 8 < 100 / 8,
    { exact_mod_cast h1 },
    interval_cases (m / 8) with H,
    all_goals { 
      try { norm_num, exact le_refl _ },
    },
  },
end

end largest_multiple_of_8_less_than_100_l295_295002


namespace puzzle_pieces_missing_l295_295134

/-- Trevor and Joe were working together to finish a 500 piece puzzle. 
They put the border together first and that was 75 pieces. 
Trevor was able to place 105 pieces of the puzzle.
Joe was able to place three times the number of puzzle pieces as Trevor. 
Prove that the number of puzzle pieces missing is 5. -/
theorem puzzle_pieces_missing :
  let total_pieces := 500
  let border_pieces := 75
  let trevor_pieces := 105
  let joe_pieces := 3 * trevor_pieces
  let placed_pieces := trevor_pieces + joe_pieces
  let remaining_pieces := total_pieces - border_pieces
  remaining_pieces - placed_pieces = 5 :=
by
  sorry

end puzzle_pieces_missing_l295_295134


namespace sunset_duration_l295_295690

theorem sunset_duration (changes : ℕ) (interval : ℕ) (total_changes : ℕ) (h1 : total_changes = 12) (h2 : interval = 10) : ∃ hours : ℕ, hours = 2 :=
by
  sorry

end sunset_duration_l295_295690


namespace find_a_l295_295862

theorem find_a (a x y : ℤ) (h_x : x = 1) (h_y : y = -3) (h_eq : a * x - y = 1) : a = -2 := by
  -- Proof skipped
  sorry

end find_a_l295_295862


namespace transformed_parabola_correct_l295_295765

def f (x : ℝ) : ℝ := (x + 2)^2 + 3
def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem transformed_parabola_correct :
  ∀ x : ℝ, g x = f (x - 3) - 2 := by
  sorry

end transformed_parabola_correct_l295_295765


namespace intersection_eq_union_eq_l295_295893

def A := { x : ℝ | x ≥ 2 }
def B := { x : ℝ | 1 < x ∧ x ≤ 4 }

theorem intersection_eq : A ∩ B = { x : ℝ | 2 ≤ x ∧ x ≤ 4 } :=
by sorry

theorem union_eq : A ∪ B = { x : ℝ | 1 < x } :=
by sorry

end intersection_eq_union_eq_l295_295893


namespace circumscribedCircleDiameter_is_10sqrt2_l295_295670

noncomputable def circumscribedCircleDiameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem circumscribedCircleDiameter_is_10sqrt2 :
  circumscribedCircleDiameter 10 (Real.pi / 4) = 10 * Real.sqrt 2 :=
by
  sorry

end circumscribedCircleDiameter_is_10sqrt2_l295_295670


namespace maximize_revenue_l295_295665

noncomputable def revenue (p : ℝ) : ℝ :=
  p * (150 - 6 * p)

theorem maximize_revenue : ∃ (p : ℝ), p = 12.5 ∧ p ≤ 30 ∧ ∀ q ≤ 30, revenue q ≤ revenue 12.5 := by 
  sorry

end maximize_revenue_l295_295665


namespace contractor_engaged_days_l295_295951

theorem contractor_engaged_days
  (earnings_per_day : ℤ)
  (fine_per_day : ℤ)
  (total_earnings : ℤ)
  (absent_days : ℤ)
  (days_worked : ℤ) 
  (h1 : earnings_per_day = 25)
  (h2 : fine_per_day = 15 / 2)
  (h3 : total_earnings = 620)
  (h4 : absent_days = 4)
  (h5 : total_earnings = earnings_per_day * days_worked - fine_per_day * absent_days) :
  days_worked = 26 := 
by {
  -- Proof goes here
  sorry
}

end contractor_engaged_days_l295_295951


namespace sum_of_first_8_terms_of_geom_seq_l295_295247

theorem sum_of_first_8_terms_of_geom_seq :
  let q : ℝ := 2
  let a_1 := (1 - q^4) / (1 - q)
  let S4 := a_1 + a_1 * q + a_1 * q^2 + a_1 * q^3
  S4 = 1 →
  let a_5 := a_1 * q^4
  let a_6 := a_1 * q^5
  let a_7 := a_1 * q^6
  let a_8 := a_1 * q^7
  let S8 := S4 + a_5 + a_6 + a_7 + a_8
  S8 = 17 :=
by
  sorry

end sum_of_first_8_terms_of_geom_seq_l295_295247


namespace shoes_produced_min_pairs_for_profit_l295_295081

-- given conditions
def production_cost (n : ℕ) : ℕ := 4000 + 50 * n

-- Question (1)
theorem shoes_produced (C : ℕ) (h : C = 36000) : ∃ n : ℕ, production_cost n = C :=
by sorry

-- given conditions for part (2)
def selling_price (price_per_pair : ℕ) (n : ℕ) : ℕ := price_per_pair * n
def profit (price_per_pair : ℕ) (n : ℕ) : ℕ := selling_price price_per_pair n - production_cost n

-- Question (2)
theorem min_pairs_for_profit (price_per_pair profit_goal : ℕ) (h : price_per_pair = 90) (h1 : profit_goal = 8500) :
  ∃ n : ℕ, profit price_per_pair n ≥ profit_goal :=
by sorry

end shoes_produced_min_pairs_for_profit_l295_295081


namespace marks_in_social_studies_l295_295759

def shekar_marks : ℕ := 82

theorem marks_in_social_studies 
  (marks_math : ℕ := 76)
  (marks_science : ℕ := 65)
  (marks_english : ℕ := 67)
  (marks_biology : ℕ := 55)
  (average_marks : ℕ := 69)
  (num_subjects : ℕ := 5) :
  marks_math + marks_science + marks_english + marks_biology + shekar_marks = average_marks * num_subjects :=
by
  sorry

end marks_in_social_studies_l295_295759


namespace larry_daily_dog_time_l295_295415

-- Definitions from the conditions
def half_hour_in_minutes : ℕ := 30
def twice_a_day (minutes : ℕ) : ℕ := 2 * minutes
def one_fifth_hour_in_minutes : ℕ := 60 / 5

-- Hypothesis resulting from the conditions
def time_walking_and_playing : ℕ := twice_a_day half_hour_in_minutes
def time_feeding : ℕ := one_fifth_hour_in_minutes

-- The theorem to prove
theorem larry_daily_dog_time : time_walking_and_playing + time_feeding = 72 := by
  show time_walking_and_playing + time_feeding = 72
  sorry

end larry_daily_dog_time_l295_295415


namespace sophomores_in_program_l295_295270

-- Define variables
variable (P S : ℕ)

-- Conditions for the problem
def total_students (P S : ℕ) : Prop := P + S = 36
def percent_sophomores_club (P S : ℕ) (x : ℕ) : Prop := x = 3 * P / 10
def percent_seniors_club (P S : ℕ) (y : ℕ) : Prop := y = S / 4
def equal_club_members (x y : ℕ) : Prop := x = y

-- Theorem stating the problem and proof goal
theorem sophomores_in_program
  (x y : ℕ)
  (h1 : total_students P S)
  (h2 : percent_sophomores_club P S x)
  (h3 : percent_seniors_club P S y)
  (h4 : equal_club_members x y) :
  P = 15 := 
sorry

end sophomores_in_program_l295_295270


namespace max_triangle_side_length_l295_295211

theorem max_triangle_side_length:
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a + b + c = 30 ∧
    a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 14 :=
  sorry

end max_triangle_side_length_l295_295211


namespace cakes_served_during_lunch_today_l295_295959

-- Define the conditions as parameters
variables
  (L : ℕ)   -- Number of cakes served during lunch today
  (D : ℕ := 6)  -- Number of cakes served during dinner today
  (Y : ℕ := 3)  -- Number of cakes served yesterday
  (T : ℕ := 14)  -- Total number of cakes served

-- Define the theorem to prove L = 5
theorem cakes_served_during_lunch_today : L + D + Y = T → L = 5 :=
by
  sorry

end cakes_served_during_lunch_today_l295_295959


namespace A_salary_less_than_B_by_20_percent_l295_295037

theorem A_salary_less_than_B_by_20_percent (A B : ℝ) (h1 : B = 1.25 * A) : 
  (B - A) / B * 100 = 20 :=
by
  sorry

end A_salary_less_than_B_by_20_percent_l295_295037


namespace refrigerator_profit_l295_295194

theorem refrigerator_profit 
  (marked_price : ℝ) 
  (cost_price : ℝ) 
  (profit_margin : ℝ ) 
  (discount1 : ℝ) 
  (profit1 : ℝ)
  (discount2 : ℝ):
  profit_margin = 0.1 → 
  profit1 = 200 → 
  cost_price = 2000 → 
  discount1 = 0.8 → 
  discount2 = 0.85 → 
  discount1 * marked_price - cost_price = profit1 → 

  (discount2 * marked_price - cost_price) = 337.5 := 
by 
  intros; 
  let marked_price := 2750; 
  sorry

end refrigerator_profit_l295_295194


namespace store_A_total_cost_store_B_total_cost_cost_effective_store_l295_295596

open Real

def total_cost_store_A (x : ℝ) : ℝ :=
  110 * x + 210 * (100 - x)

def total_cost_store_B (x : ℝ) : ℝ :=
  120 * x + 202 * (100 - x)

theorem store_A_total_cost (x : ℝ) :
  total_cost_store_A x = -100 * x + 21000 :=
by
  sorry

theorem store_B_total_cost (x : ℝ) :
  total_cost_store_B x = -82 * x + 20200 :=
by
  sorry

theorem cost_effective_store (x : ℝ) (h : x = 60) :
  total_cost_store_A x < total_cost_store_B x :=
by
  rw [h]
  sorry

end store_A_total_cost_store_B_total_cost_cost_effective_store_l295_295596


namespace steps_in_staircase_l295_295750

theorem steps_in_staircase (h1 : 120 / 20 = 6) (h2 : 180 / 6 = 30) : 
  ∃ n : ℕ, n = 30 :=
by
  -- the proof is omitted
  sorry

end steps_in_staircase_l295_295750


namespace determine_y_l295_295364

theorem determine_y (x y : ℤ) (h1 : x^2 + 4 * x - 1 = y - 2) (h2 : x = -3) : y = -2 := by
  intros
  sorry

end determine_y_l295_295364


namespace median_of_36_consecutive_integers_l295_295782

theorem median_of_36_consecutive_integers (f : ℕ → ℤ) (h_consecutive : ∀ n : ℕ, f (n + 1) = f n + 1) 
(h_size : ∃ k, f 36 = f 0 + 35) (h_sum : ∑ i in finset.range 36, f i = 6^4) : 
(∃ m, m = f (36 / 2 - 1) ∧ m = 36) :=
by
  sorry

end median_of_36_consecutive_integers_l295_295782


namespace age_ratio_l295_295529

noncomputable def ratio_of_ages (A M : ℕ) : ℕ × ℕ :=
if A = 30 ∧ (A + 15 + (M + 15)) / 2 = 50 then
  (A / Nat.gcd A M, M / Nat.gcd A M)
else
  (0, 0)

theorem age_ratio :
  (45 + (40 + 15)) / 2 = 50 → 30 = 3 * 10 ∧ 40 = 4 * 10 →
  ratio_of_ages 30 40 = (3, 4) :=
by
  sorry

end age_ratio_l295_295529


namespace proof_problem_l295_295419

theorem proof_problem
  (n : ℕ)
  (h : n = 16^3018) :
  n / 8 = 2^9032 := by
  sorry

end proof_problem_l295_295419


namespace smallest_part_when_divided_l295_295088

theorem smallest_part_when_divided (total : ℝ) (a b c : ℝ) (h_total : total = 150)
                                   (h_a : a = 3) (h_b : b = 5) (h_c : c = 7/2) :
                                   min (min (3 * (total / (a + b + c))) (5 * (total / (a + b + c)))) ((7/2) * (total / (a + b + c))) = 3 * (total / (a + b + c)) :=
by
  -- Mathematical steps have been omitted
  sorry

end smallest_part_when_divided_l295_295088


namespace solve_equation_l295_295760

open Real

noncomputable def f (x : ℝ) := 2017 * x ^ 2017 - 2017 + x
noncomputable def g (x : ℝ) := (2018 - 2017 * x) ^ (1 / 2017 : ℝ)

theorem solve_equation :
  ∀ x : ℝ, 2017 * x ^ 2017 - 2017 + x = (2018 - 2017 * x) ^ (1 / 2017 : ℝ) → x = 1 :=
by
  sorry

end solve_equation_l295_295760


namespace geometric_seq_comparison_l295_295926

def geometric_seq_positive (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ (n : ℕ), a (n+1) = a n * q

theorem geometric_seq_comparison (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_seq_positive a q) (h2 : q ≠ 1) (h3 : ∀ n, a n > 0) (h4 : q > 0) :
  a 0 + a 7 > a 3 + a 4 :=
sorry

end geometric_seq_comparison_l295_295926


namespace tulip_to_remaining_ratio_l295_295301

theorem tulip_to_remaining_ratio (total_flowers daisies sunflowers tulips remaining_tulips remaining_flowers : ℕ) 
  (h1 : total_flowers = 12) 
  (h2 : daisies = 2) 
  (h3 : sunflowers = 4) 
  (h4 : tulips = total_flowers - (daisies + sunflowers))
  (h5 : remaining_tulips = tulips)
  (h6 : remaining_flowers = remaining_tulips + sunflowers)
  (h7 : remaining_flowers = 10) : 
  tulips / remaining_flowers = 3 / 5 := 
by
  sorry

end tulip_to_remaining_ratio_l295_295301


namespace willam_farm_tax_l295_295367

theorem willam_farm_tax
  (T : ℝ)
  (h1 : 0.4 * T * (3840 / (0.4 * T)) = 3840)
  (h2 : 0 < T) :
  0.3125 * T * (3840 / (0.4 * T)) = 3000 := by
  sorry

end willam_farm_tax_l295_295367


namespace convert_8pi_over_5_to_degrees_l295_295538

noncomputable def radian_to_degree (rad : ℝ) : ℝ := rad * (180 / Real.pi)

theorem convert_8pi_over_5_to_degrees : radian_to_degree (8 * Real.pi / 5) = 288 := by
  sorry

end convert_8pi_over_5_to_degrees_l295_295538


namespace rabbit_count_l295_295454

theorem rabbit_count (r1 r2 : ℕ) (h1 : r1 = 8) (h2 : r2 = 5) : r1 + r2 = 13 := 
by 
  sorry

end rabbit_count_l295_295454


namespace acute_triangle_sin_sum_gt_two_l295_295911

theorem acute_triangle_sin_sum_gt_two 
  {α β γ : ℝ} 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : 0 < γ ∧ γ < π / 2) 
  (h4 : α + β + γ = π) :
  (Real.sin α + Real.sin β + Real.sin γ > 2) :=
sorry

end acute_triangle_sin_sum_gt_two_l295_295911


namespace cos_6theta_l295_295264

theorem cos_6theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (6 * θ) = -3224/4096 := 
by
  sorry

end cos_6theta_l295_295264


namespace find_honeydews_left_l295_295361

theorem find_honeydews_left 
  (cantaloupe_price : ℕ)
  (honeydew_price : ℕ)
  (initial_cantaloupes : ℕ)
  (initial_honeydews : ℕ)
  (dropped_cantaloupes : ℕ)
  (rotten_honeydews : ℕ)
  (end_cantaloupes : ℕ)
  (total_revenue : ℕ)
  (honeydews_left : ℕ) :
  cantaloupe_price = 2 →
  honeydew_price = 3 →
  initial_cantaloupes = 30 →
  initial_honeydews = 27 →
  dropped_cantaloupes = 2 →
  rotten_honeydews = 3 →
  end_cantaloupes = 8 →
  total_revenue = 85 →
  honeydews_left = 9 :=
by
  sorry

end find_honeydews_left_l295_295361


namespace cost_per_dvd_l295_295423

theorem cost_per_dvd (total_cost : ℝ) (num_dvds : ℕ) 
  (h1 : total_cost = 4.8) (h2 : num_dvds = 4) : (total_cost / num_dvds) = 1.2 :=
by
  sorry

end cost_per_dvd_l295_295423


namespace optimal_purchase_interval_discount_advantage_l295_295812

/- The functions and assumptions used here. -/
def purchase_feed_days (feed_per_day : ℕ) (price_per_kg : ℝ) 
  (storage_cost_per_kg_per_day : ℝ) (transportation_fee : ℝ) : ℕ :=
-- Implementation omitted
sorry

def should_use_discount (feed_per_day : ℕ) (price_per_kg : ℝ) 
  (storage_cost_per_kg_per_day : ℝ) (transportation_fee : ℝ) 
  (discount_threshold : ℕ) (discount_rate : ℝ) : Prop :=
-- Implementation omitted
sorry

/- Conditions -/
def conditions : Prop :=
  let feed_per_day := 200
  let price_per_kg := 1.8
  let storage_cost_per_kg_per_day := 0.03
  let transportation_fee := 300
  let discount_threshold := 5000 -- in kg, since 5 tons = 5000 kg
  let discount_rate := 0.85
  True -- We apply these values in the proofs below.

/- Main statements -/
theorem optimal_purchase_interval : conditions → 
  purchase_feed_days 200 1.8 0.03 300 = 10 :=
by
  intros
  -- Proof is omitted.
  sorry

theorem discount_advantage : conditions →
  should_use_discount 200 1.8 0.03 300 5000 0.85 :=
by
  intros
  -- Proof is omitted.
  sorry

end optimal_purchase_interval_discount_advantage_l295_295812


namespace ratio_pat_to_mark_l295_295751

theorem ratio_pat_to_mark (K P M : ℕ) 
  (h1 : P + K + M = 117) 
  (h2 : P = 2 * K) 
  (h3 : M = K + 65) : 
  P / Nat.gcd P M = 1 ∧ M / Nat.gcd P M = 3 := 
by
  sorry

end ratio_pat_to_mark_l295_295751


namespace knight_min_moves_l295_295326

theorem knight_min_moves (n : ℕ) (h : n ≥ 4) : 
  ∃ k : ℕ, k = 2 * (Nat.floor ((n + 1 : ℚ) / 3)) ∧
  (∀ m, (3 * m) ≥ (2 * (n - 1)) → ∃ l, l = 2 * m ∧ l ≥ k) :=
by
  sorry

end knight_min_moves_l295_295326


namespace unknown_angles_are_80_l295_295885

theorem unknown_angles_are_80 (y : ℝ) (h1 : y + y + 200 = 360) : y = 80 :=
by
  sorry

end unknown_angles_are_80_l295_295885


namespace solve_equation1_solve_equation2_l295_295436

theorem solve_equation1 (x : ℝ) (h1 : 5 * x - 2 * (x - 1) = 3) : x = 1 / 3 := 
sorry

theorem solve_equation2 (x : ℝ) (h2 : (x + 3) / 2 - 1 = (2 * x - 1) / 3) : x = 5 :=
sorry

end solve_equation1_solve_equation2_l295_295436


namespace slant_asymptote_and_sum_of_slope_and_intercept_l295_295838

noncomputable def f (x : ℚ) : ℚ := (3 * x^2 + 5 * x + 1) / (x + 2)

theorem slant_asymptote_and_sum_of_slope_and_intercept :
  (∀ x : ℚ, ∃ (m b : ℚ), (∃ r : ℚ, (r = f x ∧ (r + (m * x + b)) = f x)) ∧ m = 3 ∧ b = -1) →
  3 - 1 = 2 :=
by
  sorry

end slant_asymptote_and_sum_of_slope_and_intercept_l295_295838


namespace smallest_four_digit_divisible_by_35_l295_295152

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧ 
  ∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0) → n ≤ m := 
begin
  use 1200,
  split,
  { exact le_refl 1200 }, -- 1200 ≥ 1000
  split,
  { exact nat.lt_succ_self 9999 }, -- 1200 < 10000
  split,
  { norm_num }, -- 1200 % 35 = 0 is verified by calculation
  { intros m h, cases h, cases h_right, cases h_right_right, -- split through conditions
    exact nat.le_of_lt_succ (by norm_num at h_right_right_right_lhs.right 
    : 1200 % 35 = 0 ) -- it verifies our final smallest number is indeed 1200.
    sorry 
end

end smallest_four_digit_divisible_by_35_l295_295152


namespace exponent_of_four_l295_295262

theorem exponent_of_four (n : ℕ) (k : ℕ) (h : n = 21) 
  (eq : (↑(4 : ℕ) * 2 ^ (2 * n) = 4 ^ k)) : k = 22 :=
by
  sorry

end exponent_of_four_l295_295262


namespace inequality_relationship_l295_295080

noncomputable def even_function_periodic_decreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧
  (∀ x, f (x + 2) = f x) ∧
  (∀ x1 x2, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 1 → f x1 > f x2)

theorem inequality_relationship (f : ℝ → ℝ) (h : even_function_periodic_decreasing f) : 
  f (-1) < f (2.5) ∧ f (2.5) < f 0 :=
by 
  sorry

end inequality_relationship_l295_295080


namespace expand_polynomial_l295_295693

variable (x : ℝ)

theorem expand_polynomial :
  (7 * x - 3) * (2 * x ^ 3 + 5 * x ^ 2 - 4) = 14 * x ^ 4 + 29 * x ^ 3 - 15 * x ^ 2 - 28 * x + 12 := by
  sorry

end expand_polynomial_l295_295693


namespace find_smaller_number_l295_295855

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 45) (h2 : b = 4 * a) : a = 9 :=
by
  sorry

end find_smaller_number_l295_295855


namespace smallest_four_digit_divisible_by_35_l295_295141

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l295_295141


namespace difference_of_same_prime_factors_l295_295753

theorem difference_of_same_prime_factors (n : ℕ) :
  ∃ a b : ℕ, a - b = n ∧ (a.primeFactors.card = b.primeFactors.card) :=
by
  sorry

end difference_of_same_prime_factors_l295_295753


namespace number_of_tickets_l295_295295

-- Define the given conditions
def initial_premium := 50 -- dollars per month
def premium_increase_accident (initial_premium : ℕ) := initial_premium / 10 -- 10% increase
def premium_increase_ticket := 5 -- dollars per month per ticket
def num_accidents := 1
def new_premium := 70 -- dollars per month

-- Define the target question
theorem number_of_tickets (tickets : ℕ) :
  initial_premium + premium_increase_accident initial_premium * num_accidents + premium_increase_ticket * tickets = new_premium → 
  tickets = 3 :=
by
   sorry

end number_of_tickets_l295_295295


namespace retirement_total_l295_295949

/-- A company retirement plan allows an employee to retire when their age plus years of employment total a specific number.
A female employee was hired in 1990 on her 32nd birthday. She could first be eligible to retire under this provision in 2009. -/
def required_total_age_years_of_employment : ℕ :=
  let hire_year := 1990
  let retirement_year := 2009
  let age_when_hired := 32
  let years_of_employment := retirement_year - hire_year
  let age_at_retirement := age_when_hired + years_of_employment
  age_at_retirement + years_of_employment

theorem retirement_total :
  required_total_age_years_of_employment = 70 :=
by
  sorry

end retirement_total_l295_295949


namespace real_number_representation_l295_295336

theorem real_number_representation (x : ℝ) 
  (h₀ : 0 < x) (h₁ : x ≤ 1) :
  ∃ (n : ℕ → ℕ), (∀ k, n k > 0) ∧ (∀ k, n (k + 1) = n k * 2 ∨ n (k + 1) = n k * 3 ∨ n (k + 1) = n k * 4) ∧ 
  (x = ∑' k, 1 / (n k)) :=
sorry

end real_number_representation_l295_295336


namespace sequence_4951_l295_295736

theorem sequence_4951 :
  (∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = a n + n) ∧ a 100 = 4951) :=
sorry

end sequence_4951_l295_295736


namespace proof_of_acdb_l295_295234

theorem proof_of_acdb
  (x a b c d : ℤ)
  (hx_eq : 7 * x - 8 * x = 20)
  (hx_form : (a + b * Real.sqrt c) / d = x)
  (hints : x = (4 + 2 * Real.sqrt 39) / 7)
  (int_cond : a = 4 ∧ b = 2 ∧ c = 39 ∧ d = 7) :
  a * c * d / b = 546 := by
sorry

end proof_of_acdb_l295_295234


namespace distinct_positions_24_l295_295972

open Finset

theorem distinct_positions_24 : 
  ∃ (x y z : ℕ), x + y + z = 24 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (∃! (n : ℕ), n = 265) :=
by 
  sorry

end distinct_positions_24_l295_295972


namespace tv_price_reduction_l295_295443

theorem tv_price_reduction (x : ℝ) (Q : ℝ) (P : ℝ) (h1 : Q > 0) (h2 : P > 0) (h3 : P*(1 - x/100) * 1.85 * Q = 1.665 * P * Q) : x = 10 :=
by 
  sorry

end tv_price_reduction_l295_295443


namespace smallest_four_digit_div_by_35_l295_295144

theorem smallest_four_digit_div_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  let n := 1015,
  use n,
  split,
  { exact nat.le_of_lt (nat.lt_of_succ_le 1000) },
  split,
  { exact nat.lt_succ_self 10000 },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (by norm_num)) },
  { intros m hm hbound hmod,
    exact le_of_lt hbound },
  sorry,
end

end smallest_four_digit_div_by_35_l295_295144


namespace octal_to_binary_of_55_equals_101101_l295_295359

def octalToDecimal (n : ℕ) : ℕ :=
  (n / 10) * 8^(1 : ℕ) + (n % 10) * 8^(0 : ℕ)

def divideBy2Remainders (n : ℕ) : List ℕ :=
  if n = 0 then [] else (n % 2) :: divideBy2Remainders (n / 2)

def listToBinary (l : List ℕ) : ℕ :=
  l.foldr (λ x acc, acc * 10 + x) 0

noncomputable def octalToBinary (n : ℕ) : ℕ :=
  listToBinary (divideBy2Remainders (octalToDecimal n))

theorem octal_to_binary_of_55_equals_101101 : octalToBinary 55 = 101101 :=
by
  sorry

end octal_to_binary_of_55_equals_101101_l295_295359


namespace trigonometric_identity_l295_295335

theorem trigonometric_identity : 
  Real.cos 6 * Real.cos 36 + Real.sin 6 * Real.cos 54 = Real.sqrt 3 / 2 :=
sorry

end trigonometric_identity_l295_295335


namespace connect_5_points_four_segments_l295_295986

theorem connect_5_points_four_segments (A B C D E : Type) (h : ∀ (P Q R : Type), P ≠ Q ∧ Q ≠ R ∧ R ≠ P)
: ∃ (n : ℕ), n = 135 := 
  sorry

end connect_5_points_four_segments_l295_295986


namespace milo_cash_reward_l295_295109

theorem milo_cash_reward : 
  let three_2s := [2, 2, 2]
  let four_3s := [3, 3, 3, 3]
  let one_4 := [4]
  let one_5 := [5]
  let all_grades := three_2s ++ four_3s ++ one_4 ++ one_5
  let total_grades := all_grades.length
  let total_sum := all_grades.sum
  let average_grade := total_sum / total_grades
  5 * average_grade = 15 := by
  sorry

end milo_cash_reward_l295_295109


namespace prob_two_white_balls_l295_295188

open Nat

def total_balls : ℕ := 8 + 10

def prob_first_white : ℚ := 8 / total_balls

def prob_second_white (total_balls_minus_one : ℕ) : ℚ := 7 / total_balls_minus_one

theorem prob_two_white_balls : 
  ∃ (total_balls_minus_one : ℕ) (p_first p_second : ℚ), 
    total_balls_minus_one = total_balls - 1 ∧
    p_first = prob_first_white ∧
    p_second = prob_second_white total_balls_minus_one ∧
    p_first * p_second = 28 / 153 := 
by
  sorry

end prob_two_white_balls_l295_295188


namespace present_cost_after_discount_l295_295942

theorem present_cost_after_discount 
  (X : ℝ) (P : ℝ) 
  (h1 : X - 4 = (0.80 * P) / 3) 
  (h2 : P = 3 * X)
  :
  0.80 * P = 48 :=
by
  sorry

end present_cost_after_discount_l295_295942


namespace acute_angles_relation_l295_295387

theorem acute_angles_relation (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : Real.sin α = (1 / 2) * Real.sin (α + β)) : α < β :=
sorry

end acute_angles_relation_l295_295387


namespace mod_equiv_example_l295_295762

theorem mod_equiv_example : (185 * 944) % 60 = 40 := by
  sorry

end mod_equiv_example_l295_295762


namespace find_three_digit_number_l295_295694

theorem find_three_digit_number (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) 
  : 100 * a + 10 * b + c = 5 * a * b * c → a = 1 ∧ b = 7 ∧ c = 5 :=
by
  sorry

end find_three_digit_number_l295_295694


namespace parabola_point_value_l295_295875

variable {x₀ y₀ : ℝ}

theorem parabola_point_value
  (h₁ : y₀^2 = 4 * x₀)
  (h₂ : (Real.sqrt ((x₀ - 1)^2 + y₀^2) = 5/4 * x₀)) :
  x₀ = 4 := by
  sorry

end parabola_point_value_l295_295875


namespace can_capacity_l295_295183

-- Definitions of the conditions
variable (M W : ℕ) -- initial amounts of milk and water
variable (M' : ℕ := M + 2) -- new amount of milk after adding 2 liters
variable (ratio_initial : M / W = 1 / 5)
variable (ratio_new : M' / W = 3 / 5)

theorem can_capacity (M W : ℕ) (h_ratio_initial : M / W = 1 / 5) (h_ratio_new : (M + 2) / W = 3 / 5) : (M + W + 2) = 8 := 
by
  sorry

end can_capacity_l295_295183


namespace remainder_of_p_div_x_plus_2_l295_295853

def p (x : ℝ) : ℝ := x^4 - x^2 + 3 * x + 4

theorem remainder_of_p_div_x_plus_2 : p (-2) = 10 := by
  sorry

end remainder_of_p_div_x_plus_2_l295_295853


namespace min_tosses_one_head_l295_295815

theorem min_tosses_one_head (n : ℕ) (P : ℝ) (h₁ : P = 1 - (1 / 2) ^ n) (h₂ : P ≥ 15 / 16) : n ≥ 4 :=
by
  sorry -- Proof to be filled in.

end min_tosses_one_head_l295_295815


namespace journey_time_l295_295338

noncomputable def journey_time_proof : Prop :=
  ∃ t1 t2 t3 : ℝ,
    25 * t1 - 25 * t2 + 25 * t3 = 100 ∧
    5 * t1 + 5 * t2 + 25 * t3 = 100 ∧
    25 * t1 + 5 * t2 + 5 * t3 = 100 ∧
    t1 + t2 + t3 = 8

theorem journey_time : journey_time_proof := by sorry

end journey_time_l295_295338


namespace solve_problem_l295_295365

open Classical

-- Definition of the problem conditions
def problem_conditions (x y : ℝ) : Prop :=
  5 * y^2 + 3 * y + 2 = 2 * (10 * x^2 + 3 * y + 3) ∧ y = 3 * x + 1

-- Definition of the quadratic solution considering the quadratic formula
def quadratic_solution (x : ℝ) : Prop :=
  x = (-21 + Real.sqrt 641) / 50 ∨ x = (-21 - Real.sqrt 641) / 50

-- Main theorem statement
theorem solve_problem :
  ∃ x y : ℝ, problem_conditions x y ∧ quadratic_solution x :=
by
  sorry

end solve_problem_l295_295365


namespace algebraic_expression_value_l295_295563

theorem algebraic_expression_value {x : ℝ} (h : x * (x + 2) = 2023) : 2 * (x + 3) * (x - 1) - 2018 = 2022 := 
by 
  sorry

end algebraic_expression_value_l295_295563


namespace smallest_four_digit_divisible_by_35_l295_295158

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l295_295158


namespace remainder_x_plus_3uy_div_y_l295_295180

theorem remainder_x_plus_3uy_div_y (x y u v : ℕ) (hx : x = u * y + v) (hv_range : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
by
  sorry

end remainder_x_plus_3uy_div_y_l295_295180


namespace find_a_and_b_l295_295095

theorem find_a_and_b (a b : ℝ) 
  (curve : ∀ x : ℝ, y = x^2 + a * x + b) 
  (tangent : ∀ x : ℝ, y - b = a * x) 
  (tangent_line : ∀ x y : ℝ, x + y = 1) :
  a = -1 ∧ b = 1 := 
by 
  sorry

end find_a_and_b_l295_295095


namespace area_of_ABCD_is_196_l295_295449

-- Define the shorter side length of the smaller rectangles
def shorter_side : ℕ := 7

-- Define the longer side length of the smaller rectangles
def longer_side : ℕ := 2 * shorter_side

-- Define the width of rectangle ABCD
def width_ABCD : ℕ := 2 * shorter_side

-- Define the length of rectangle ABCD
def length_ABCD : ℕ := longer_side

-- Define the area of rectangle ABCD
def area_ABCD : ℕ := length_ABCD * width_ABCD

-- Statement of the problem
theorem area_of_ABCD_is_196 : area_ABCD = 196 :=
by
  -- insert proof here
  sorry

end area_of_ABCD_is_196_l295_295449


namespace negation_of_tan_one_l295_295712

theorem negation_of_tan_one :
  (∃ x : ℝ, Real.tan x = 1) ↔ ¬ (∀ x : ℝ, Real.tan x ≠ 1) :=
by
  sorry

end negation_of_tan_one_l295_295712


namespace money_left_l295_295294

noncomputable def olivia_money : ℕ := 112
noncomputable def nigel_money : ℕ := 139
noncomputable def ticket_cost : ℕ := 28
noncomputable def num_tickets : ℕ := 6

theorem money_left : (olivia_money + nigel_money - ticket_cost * num_tickets) = 83 :=
by
  sorry

end money_left_l295_295294


namespace correct_flag_positions_l295_295102

-- Definitions for the gears and their relations
structure Gear where
  flag_position : ℝ -- position of the flag in degrees

-- Condition: Two identical gears
def identical_gears (A B : Gear) : Prop := true

-- Conditions: Initial positions and gear interaction
def initial_position_A (A : Gear) : Prop := A.flag_position = 0
def initial_position_B (B : Gear) : Prop := B.flag_position = 180
def gear_interaction (A B : Gear) (theta : ℝ) : Prop :=
  A.flag_position = -theta ∧ B.flag_position = theta

-- Definition for the final positions given a rotation angle θ
def final_position (A B : Gear) (theta : ℝ) : Prop :=
  identical_gears A B ∧ initial_position_A A ∧ initial_position_B B ∧ gear_interaction A B theta

-- Theorem stating the positions after some rotation θ
theorem correct_flag_positions (A B : Gear) (theta : ℝ) : final_position A B theta → 
  A.flag_position = -theta ∧ B.flag_position = theta :=
by
  intro h
  cases h
  sorry

end correct_flag_positions_l295_295102


namespace fraction_product_equivalence_l295_295679

theorem fraction_product_equivalence :
  (1 / 3) * (1 / 2) * (2 / 5) * (3 / 7) = 6 / 35 := 
by 
  sorry

end fraction_product_equivalence_l295_295679


namespace sum_first_11_terms_l295_295988

variable (a : ℕ → ℤ) -- The arithmetic sequence
variable (d : ℤ) -- Common difference
variable (S : ℕ → ℤ) -- Sum of the arithmetic sequence

-- The properties of the arithmetic sequence and sum
axiom arith_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom sum_arith_seq (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Given condition
axiom given_condition : a 1 + a 5 + a 8 = a 2 + 12

-- To prove
theorem sum_first_11_terms : S 11 = 66 := by
  sorry

end sum_first_11_terms_l295_295988


namespace simplify_expression_l295_295837

theorem simplify_expression (b : ℝ) (h : b ≠ -1) : 
  1 - (1 / (1 - (b / (1 + b)))) = -b :=
by {
  sorry
}

end simplify_expression_l295_295837


namespace gcd_105_88_l295_295978

-- Define the numbers as constants
def a : ℕ := 105
def b : ℕ := 88

-- State the theorem: gcd(a, b) = 1
theorem gcd_105_88 : Nat.gcd a b = 1 := by
  sorry

end gcd_105_88_l295_295978


namespace brooke_kent_ratio_l295_295351

theorem brooke_kent_ratio :
  ∀ (alison brooke brittany kent : ℕ),
  (kent = 1000) →
  (alison = 4000) →
  (alison = brittany / 2) →
  (brittany = 4 * brooke) →
  brooke / kent = 2 :=
by
  intros alison brooke brittany kent kent_val alison_val alison_brittany brittany_brooke
  sorry

end brooke_kent_ratio_l295_295351


namespace sum_sequence_arithmetic_l295_295989

noncomputable def arithmetic_sequence (a2 a7 : ℕ → ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a2 + (n - 2 : ℕ) * d

noncomputable def sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  1 / (a n * a (n + 1))

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, sequence a i

theorem sum_sequence_arithmetic :
  let a2 := 3
  let a7 := 13
  let d  := (a7 - a2) / (7 - 2)
  let a  := arithmetic_sequence a2 a7 d
  ∀ n : ℕ, sum_first_n_terms a n = n / (2 * n + 1) :=
begin
  intros,
  sorry
end

end sum_sequence_arithmetic_l295_295989


namespace sector_area_correct_l295_295882

noncomputable def sector_area (r θ : ℝ) : ℝ := 0.5 * θ * r^2

theorem sector_area_correct (r θ : ℝ) (hr : r = 2) (hθ : θ = 2 * Real.pi / 3) :
  sector_area r θ = 4 * Real.pi / 3 :=
by
  subst hr
  subst hθ
  sorry

end sector_area_correct_l295_295882


namespace pies_sold_each_day_l295_295032

theorem pies_sold_each_day (total_pies : ℕ) (days_in_week : ℕ) (h1 : total_pies = 56) (h2 : days_in_week = 7) :
  (total_pies / days_in_week = 8) :=
by
exact sorry

end pies_sold_each_day_l295_295032


namespace product_of_values_of_t_squared_eq_49_l295_295059

theorem product_of_values_of_t_squared_eq_49 :
  (∀ t, t^2 = 49 → t = 7 ∨ t = -7) →
  (7 * -7 = -49) :=
by
  intros h
  sorry

end product_of_values_of_t_squared_eq_49_l295_295059


namespace log_inequality_l295_295076

theorem log_inequality (n : ℕ) (h1 : n > 1) : 
  (1 : ℝ) / (n : ℝ) > Real.log ((n + 1 : ℝ) / n) ∧ 
  Real.log ((n + 1 : ℝ) / n) > (1 : ℝ) / (n + 1) := 
by
  sorry

end log_inequality_l295_295076


namespace function_is_quadratic_l295_295651

-- Definitions for the conditions
def is_quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0) ∧ ∀ (x : ℝ), f x = a * x^2 + b * x + c

-- The function to be proved as a quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- The theorem statement: f must be a quadratic function
theorem function_is_quadratic : is_quadratic_function f :=
  sorry

end function_is_quadratic_l295_295651


namespace points_on_opposite_sides_of_line_l295_295870

theorem points_on_opposite_sides_of_line 
  (a : ℝ) 
  (h : (3 * -3 - 2 * -1 - a) * (3 * 4 - 2 * -6 - a) < 0) : 
  -7 < a ∧ a < 24 :=
sorry

end points_on_opposite_sides_of_line_l295_295870


namespace domain_of_fraction_is_all_real_l295_295226

theorem domain_of_fraction_is_all_real (k : ℝ) :
  (∀ x : ℝ, -7 * x^2 + 3 * x + 4 * k ≠ 0) ↔ k < -9 / 112 :=
by sorry

end domain_of_fraction_is_all_real_l295_295226


namespace sum_derivatives_positive_l295_295576

noncomputable def f (x : ℝ) : ℝ := -x^2 - x^4 - x^6
noncomputable def f' (x : ℝ) : ℝ := -2*x - 4*x^3 - 6*x^5

theorem sum_derivatives_positive (x1 x2 x3 : ℝ) (h1 : x1 + x2 < 0) (h2 : x2 + x3 < 0) (h3 : x3 + x1 < 0) :
  f' x1 + f' x2 + f' x3 > 0 := 
sorry

end sum_derivatives_positive_l295_295576


namespace isosceles_right_triangle_area_l295_295442

-- Define the isosceles right triangle and its properties

theorem isosceles_right_triangle_area 
  (h : ℝ)
  (hyp : h = 6) :
  let l : ℝ := h / Real.sqrt 2
  let A : ℝ := (l^2) / 2
  A = 9 :=
by
  -- The proof steps are skipped with sorry
  sorry

end isosceles_right_triangle_area_l295_295442


namespace sector_area_l295_295347

theorem sector_area (r : ℝ) (h1 : r = 2) (h2 : 2 * r + r * ((2 * π * r - 2) / r) = 4 * π) :
  (1 / 2) * r^2 * ((4 * π - 2) / r) = 4 * π - 2 :=
by
  sorry

end sector_area_l295_295347


namespace kimmie_earnings_l295_295411

theorem kimmie_earnings (K : ℚ) (h : (1/2 : ℚ) * K + (1/3 : ℚ) * K = 375) : K = 450 := 
by
  sorry

end kimmie_earnings_l295_295411


namespace sum_of_scores_with_three_ways_correct_l295_295453

def is_valid_combination (c u: ℕ) (S: ℚ) : Prop :=
  (c + u <= 25) ∧
  (6 * c + 2.5 * u = S)

def count_ways_to_achieve_score (S: ℚ) : ℕ :=
  (List.range 26).countp (λ c => ∃ u, is_valid_combination c u S)

def scores_with_three_ways : List ℚ :=
  (List.range 151).map (λ n => (n : ℚ)).filter (λ S => count_ways_to_achieve_score S = 3)

def sum_of_scores_with_three_ways : ℚ :=
  scores_with_three_ways.sum

theorem sum_of_scores_with_three_ways_correct :
  sum_of_scores_with_three_ways = 182 := -- The actual sum to be specified based on the calculation
  sorry

end sum_of_scores_with_three_ways_correct_l295_295453


namespace distance_between_A_and_B_l295_295752

theorem distance_between_A_and_B
  (vA vB D : ℝ)
  (hvB : vB = (3/2) * vA)
  (second_meeting_distance : 20 = D * 2 / 5) : 
  D = 50 := 
by
  sorry

end distance_between_A_and_B_l295_295752


namespace solution_l295_295866

def p (x : ℝ) : Prop := x^2 + 2 * x - 3 < 0
def q (x : ℝ) : Prop := x ∈ Set.univ

theorem solution (x : ℝ) (hx : p x ∧ q x) : x = -2 ∨ x = -1 ∨ x = 0 := 
by
  sorry

end solution_l295_295866


namespace total_amount_after_interest_l295_295981

-- Define the constants
def principal : ℝ := 979.0209790209791
def rate : ℝ := 0.06
def time : ℝ := 2.4

-- Define the formula for interest calculation
def interest (P R T : ℝ) : ℝ := P * R * T

-- Define the formula for the total amount after interest is added
def total_amount (P I : ℝ) : ℝ := P + I

-- State the theorem
theorem total_amount_after_interest : 
    total_amount principal (interest principal rate time) = 1120.0649350649352 :=
by
    -- placeholder for the proof
    sorry

end total_amount_after_interest_l295_295981


namespace missing_number_geometric_sequence_l295_295632

theorem missing_number_geometric_sequence : 
  ∃ (x : ℤ), (x = 162) ∧ 
  (x = 54 * 3 ∧ 
  486 = x * 3 ∧ 
  ∀ a b : ℤ, (b = 2 * 3) ∧ 
              (a = 2 * 3) ∧ 
              (18 = b * 3) ∧ 
              (54 = 18 * 3) ∧ 
              (54 * 3 = x)) := 
by sorry

end missing_number_geometric_sequence_l295_295632


namespace train_length_is_correct_l295_295669

noncomputable def speed_kmph_to_mps (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

noncomputable def distance_crossed (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def train_length (speed_kmph crossing_time bridge_length : ℝ) : ℝ :=
  distance_crossed (speed_kmph_to_mps speed_kmph) crossing_time - bridge_length

theorem train_length_is_correct :
  ∀ (crossing_time bridge_length speed_kmph : ℝ),
    crossing_time = 26.997840172786177 →
    bridge_length = 150 →
    speed_kmph = 36 →
    train_length speed_kmph crossing_time bridge_length = 119.97840172786177 :=
by
  intros crossing_time bridge_length speed_kmph h1 h2 h3
  rw [h1, h2, h3]
  simp only [speed_kmph_to_mps, distance_crossed, train_length]
  sorry

end train_length_is_correct_l295_295669


namespace cylinder_height_percentage_l295_295341

-- Lean 4 statement for the problem
theorem cylinder_height_percentage (h : ℝ) (r : ℝ) (H : ℝ) :
  (7 / 8) * h = (3 / 5) * (1.25 * r)^2 * H → H = 0.9333 * h :=
by 
  sorry

end cylinder_height_percentage_l295_295341


namespace problem_l295_295096

def seq (a : ℕ → ℤ) : Prop :=
∀ n, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

theorem problem (a : ℕ → ℤ) (h₁ : a 1 = 2010) (h₂ : a 2 = 2011) (h₃ : seq a) : a 1000 = 2343 :=
sorry

end problem_l295_295096


namespace directrix_of_parabola_l295_295919

-- Define the condition given in the problem
def parabola_eq (x y : ℝ) : Prop := x^2 = 2 * y

-- Define the directrix equation property we want to prove
theorem directrix_of_parabola (x : ℝ) :
  (∃ y : ℝ, parabola_eq x y) → (∃ y : ℝ, y = -1 / 2) :=
by sorry

end directrix_of_parabola_l295_295919


namespace batsman_average_after_11th_inning_l295_295653

theorem batsman_average_after_11th_inning 
  (x : ℝ) 
  (h1 : (10 * x + 95) / 11 = x + 5) : 
  x + 5 = 45 :=
by 
  sorry

end batsman_average_after_11th_inning_l295_295653


namespace cara_between_friends_l295_295355

theorem cara_between_friends (n : ℕ) (h : n = 6) : ∃ k : ℕ, k = 15 :=
by {
  sorry
}

end cara_between_friends_l295_295355


namespace factorize_expression_l295_295544

theorem factorize_expression (a : ℝ) : a^3 + 2*a^2 + a = a*(a+1)^2 :=
  sorry

end factorize_expression_l295_295544


namespace correct_statement_is_B_l295_295011

def coefficient_of_x : Int := 1
def is_monomial (t : String) : Bool := t = "1x^0"
def coefficient_of_neg_3x : Int := -3
def degree_of_5x2y : Int := 3

theorem correct_statement_is_B :
  (coefficient_of_x = 0) = false ∧ 
  (is_monomial "1x^0" = true) ∧ 
  (coefficient_of_neg_3x = 3) = false ∧ 
  (degree_of_5x2y = 2) = false ∧ 
  (B = "1 is a monomial") :=
by {
  sorry
}

end correct_statement_is_B_l295_295011


namespace total_shirts_made_l295_295352

def shirtsPerMinute := 6
def minutesWorkedYesterday := 12
def shirtsMadeToday := 14

theorem total_shirts_made : shirtsPerMinute * minutesWorkedYesterday + shirtsMadeToday = 86 := by
  sorry

end total_shirts_made_l295_295352


namespace matrix_hall_property_l295_295739

open Finset

variable {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℕ)

theorem matrix_hall_property (m_ne_n : m ≠ n)
  (h : ∀ (f : Fin m → Fin n) (hf : Function.Injective f), ∃ i, A i (f i) = 0) :
  ∃ S : Finset (Fin m), ∃ T : Finset (Fin n),
    (∀ i ∈ S, ∀ j ∈ T, A i j = 0) ∧ S.card + T.card > n := 
sorry

end matrix_hall_property_l295_295739


namespace product_of_values_of_t_squared_eq_49_l295_295060

theorem product_of_values_of_t_squared_eq_49 :
  (∀ t, t^2 = 49 → t = 7 ∨ t = -7) →
  (7 * -7 = -49) :=
by
  intros h
  sorry

end product_of_values_of_t_squared_eq_49_l295_295060


namespace cube_surface_area_l295_295445

/-- Given a cube with a space diagonal of 6, the surface area is 72. -/
theorem cube_surface_area (s : ℝ) (h : s * Real.sqrt 3 = 6) : 6 * s^2 = 72 :=
by
  sorry

end cube_surface_area_l295_295445


namespace mixed_fractions_product_l295_295506

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l295_295506


namespace students_in_ms_delmont_class_l295_295621

-- Let us define the necessary conditions

def total_cupcakes : Nat := 40
def students_mrs_donnelly_class : Nat := 16
def adults_count : Nat := 4 -- Ms. Delmont, Mrs. Donnelly, the school nurse, and the school principal
def leftover_cupcakes : Nat := 2

-- Define the number of students in Ms. Delmont's class
def students_ms_delmont_class : Nat := 18

-- The statement to prove
theorem students_in_ms_delmont_class :
  total_cupcakes - adults_count - students_mrs_donnelly_class - leftover_cupcakes = students_ms_delmont_class :=
by
  sorry

end students_in_ms_delmont_class_l295_295621


namespace marked_cells_in_grid_l295_295427

theorem marked_cells_in_grid :
  ∀ (grid : Matrix (Fin 5) (Fin 5) Bool), 
  (∀ (i j : Fin 3), ∃! (a b : Fin 3), grid (i + a + 1) (j + b + 1) = true) → ∃ (n : ℕ), 1 ≤ n ∧ n ≤ 4 :=
by
  sorry

end marked_cells_in_grid_l295_295427


namespace find_takeoff_run_distance_l295_295124

-- Define the conditions
def time_of_takeoff : ℝ := 15 -- seconds
def takeoff_speed_kmh : ℝ := 100 -- km/h

-- Define the conversions and proof problem
noncomputable def takeoff_speed_ms : ℝ := takeoff_speed_kmh * 1000 / 3600 -- conversion from km/h to m/s
noncomputable def acceleration : ℝ := takeoff_speed_ms / time_of_takeoff -- a = v / t

theorem find_takeoff_run_distance : 
  (1/2) * acceleration * (time_of_takeoff ^ 2) = 208 := by
  sorry

end find_takeoff_run_distance_l295_295124


namespace daniel_spent_2290_l295_295840

theorem daniel_spent_2290 (total_games: ℕ) (price_12_games count_price_12: ℕ) 
  (price_7_games frac_price_7: ℕ) (price_3_games: ℕ) 
  (count_price_7: ℕ) (h1: total_games = 346)
  (h2: count_price_12 = 80) (h3: price_12_games = 12)
  (h4: frac_price_7 = 50) (h5: price_7_games = 7)
  (h6: price_3_games = 3) (h7: count_price_7 = (frac_price_7 * (total_games - count_price_12)) / 100):
  (count_price_12 * price_12_games) + (count_price_7 * price_7_games) + ((total_games - count_price_12 - count_price_7) * price_3_games) = 2290 := 
by
  sorry

end daniel_spent_2290_l295_295840


namespace probability_of_coprime_l295_295810

open Finset
open Rat

noncomputable def set_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def pairs := (set_numbers.val.to_list.product set_numbers.val.to_list).filter (λ x, x.1 < x.2)

def gcd_is_one (x : ℕ × ℕ) : Prop := Nat.gcd x.1 x.2 = 1

def num_coprime_pairs : ℕ := (pairs.filter gcd_is_one).length

def num_total_pairs : ℕ := pairs.length

def probability_coprime_pairs : ℚ := num_coprime_pairs /. num_total_pairs

theorem probability_of_coprime :
  probability_coprime_pairs = 17/21 :=
sorry

end probability_of_coprime_l295_295810


namespace remainder_of_S_mod_500_eq_zero_l295_295604

open Function

def R : Set ℕ := { r | ∃ n : ℕ, r = (3^n % 500) }

def S : ℕ := ∑ r in R.toFinset, r

theorem remainder_of_S_mod_500_eq_zero :
  (S % 500) = 0 := by
  sorry

end remainder_of_S_mod_500_eq_zero_l295_295604


namespace probability_three_dice_same_number_is_1_div_36_l295_295798

noncomputable def probability_same_number_three_dice : ℚ :=
  let first_die := 1
  let second_die := 1 / 6
  let third_die := 1 / 6
  first_die * second_die * third_die

theorem probability_three_dice_same_number_is_1_div_36 : probability_same_number_three_dice = 1 / 36 :=
  sorry

end probability_three_dice_same_number_is_1_div_36_l295_295798


namespace simplify_and_evaluate_l295_295114

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 2 - Real.sqrt 2

theorem simplify_and_evaluate : 
  let expr := (a / (a^2 - b^2) - 1 / (a + b)) / (b / (b - a))
  expr = -1 / 2 := by
  sorry

end simplify_and_evaluate_l295_295114


namespace series_sum_proof_l295_295052

noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, if n % 3 = 0 then 1 / (27 ^ (n / 3)) * (5 / 9) else 0

theorem series_sum_proof : infinite_series_sum = 15 / 26 :=
  sorry

end series_sum_proof_l295_295052


namespace circle_center_radius_l295_295630

theorem circle_center_radius (x y : ℝ) :
  (x ^ 2 + y ^ 2 + 2 * x - 4 * y - 6 = 0) →
  ((x + 1) ^ 2 + (y - 2) ^ 2 = 11) :=
by sorry

end circle_center_radius_l295_295630


namespace triangle_area_correct_l295_295456

-- Define the points (vertices) of the triangle
def point1 : ℝ × ℝ := (2, 1)
def point2 : ℝ × ℝ := (8, -3)
def point3 : ℝ × ℝ := (2, 7)

-- Function to calculate the area of the triangle given three points (shoelace formula)
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - B.2 * C.1 - C.2 * A.1 - A.2 * B.1)

-- Prove that the area of the triangle with the given vertices is 18 square units
theorem triangle_area_correct : triangle_area point1 point2 point3 = 18 :=
by
  sorry

end triangle_area_correct_l295_295456


namespace takeoff_run_length_l295_295125

theorem takeoff_run_length
  (t : ℕ) (h_t : t = 15)
  (v_kmh : ℕ) (h_v : v_kmh = 100)
  (uniform_acc : Prop) :
  ∃ S : ℕ, S = 208 := by
  sorry

end takeoff_run_length_l295_295125


namespace arithmetic_sequence_twentieth_term_l295_295050

theorem arithmetic_sequence_twentieth_term :
  ∀ (a_1 d : ℕ), a_1 = 3 → d = 4 → (a_1 + (20 - 1) * d) = 79 := by
  intros a_1 d h1 h2
  rw [h1, h2]
  simp
  sorry

end arithmetic_sequence_twentieth_term_l295_295050


namespace pies_sold_each_day_l295_295030

theorem pies_sold_each_day (total_pies: ℕ) (days_in_week: ℕ) 
  (h1: total_pies = 56) (h2: days_in_week = 7) : 
  total_pies / days_in_week = 8 :=
by
  sorry

end pies_sold_each_day_l295_295030


namespace average_of_values_l295_295698

theorem average_of_values (z : ℝ) : 
  (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end average_of_values_l295_295698


namespace find_k_l295_295107

theorem find_k (k : ℝ) : 
  (∀ x : ℝ, y = 2 * x + 3) ∧ 
  (∀ x : ℝ, y = k * x + 4) ∧ 
  (1, 5) ∈ { p | ∃ x, p = (x, 2 * x + 3) } ∧ 
  (1, 5) ∈ { q | ∃ x, q = (x, k * x + 4) } → 
  k = 1 :=
by
  sorry

end find_k_l295_295107


namespace difference_of_numbers_l295_295927

theorem difference_of_numbers 
  (a b : ℕ) 
  (h1 : a + b = 23976)
  (h2 : b % 8 = 0)
  (h3 : a = 7 * b / 8) : 
  b - a = 1598 :=
sorry

end difference_of_numbers_l295_295927


namespace convert_cylindrical_to_rectangular_l295_295360

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 7 (Real.pi / 4) 8 = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2, 8) :=
by
  sorry

end convert_cylindrical_to_rectangular_l295_295360


namespace Balint_claim_impossible_l295_295534

-- Declare the lengths of the ladders and the vertical projection distance
def AC : ℝ := 3
def BD : ℝ := 2
def E_proj : ℝ := 1

-- State the problem conditions and what we need to prove
theorem Balint_claim_impossible (h1 : AC = 3) (h2 : BD = 2) (h3 : E_proj = 1) :
  False :=
  sorry

end Balint_claim_impossible_l295_295534


namespace arsh_eq_arch_pos_eq_arch_neg_eq_arth_eq_l295_295619

noncomputable def arsh (x : ℝ) := Real.log (x + Real.sqrt (x^2 + 1))
noncomputable def arch_pos (x : ℝ) := Real.log (x + Real.sqrt (x^2 - 1))
noncomputable def arch_neg (x : ℝ) := Real.log (x - Real.sqrt (x^2 - 1))
noncomputable def arth (x : ℝ) := (1 / 2) * Real.log ((1 + x) / (1 - x))

theorem arsh_eq (x : ℝ) : arsh x = Real.log (x + Real.sqrt (x^2 + 1)) := by
  sorry

theorem arch_pos_eq (x : ℝ) : arch_pos x = Real.log (x + Real.sqrt (x^2 - 1)) := by
  sorry

theorem arch_neg_eq (x : ℝ) : arch_neg x = Real.log (x - Real.sqrt (x^2 - 1)) := by
  sorry

theorem arth_eq (x : ℝ) : arth x = (1 / 2) * Real.log ((1 + x) / (1 - x)) := by
  sorry

end arsh_eq_arch_pos_eq_arch_neg_eq_arth_eq_l295_295619


namespace find_c_l295_295175

def P (x : ℝ) (c : ℝ) : ℝ :=
  x^3 + 3*x^2 + c*x + 15

theorem find_c (c : ℝ) : (x - 3 = P x c → c = -23) := by
  sorry

end find_c_l295_295175


namespace gcd_fib_consecutive_rel_prime_gcd_fib_l295_295654

/-- 
Prove that the gcd of two consecutive Fibonacci numbers is 1.
-/
theorem gcd_fib_consecutive_rel_prime (n : ℕ) : Nat.gcd (Nat.fib n) (Nat.fib (n + 1)) = 1 :=
sorry

/-- 
Prove that gcd(F_m, F_n) = F_gcd(m, n) where F is Fibonacci sequence.
-/
theorem gcd_fib (m n : ℕ) : Nat.gcd (Nat.fib m) (Nat.fib n) = Nat.fib (Nat.gcd m n) :=
sorry

end gcd_fib_consecutive_rel_prime_gcd_fib_l295_295654


namespace percentage_increase_of_cars_l295_295829

theorem percentage_increase_of_cars :
  ∀ (initial final : ℕ), initial = 24 → final = 48 → ((final - initial) * 100 / initial) = 100 :=
by
  intros
  sorry

end percentage_increase_of_cars_l295_295829


namespace problem_solution_l295_295936

def eq_A (x : ℝ) : Prop := 2 * x = 7
def eq_B (x y : ℝ) : Prop := x^2 + y = 5
def eq_C (x : ℝ) : Prop := x = 1 / x + 1
def eq_D (x : ℝ) : Prop := x^2 + x = 4

def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ a * x^2 + b * x + c = 0

theorem problem_solution : is_quadratic eq_D := by
  sorry

end problem_solution_l295_295936


namespace carrots_total_l295_295286

-- Define the initial number of carrots Maria picked
def initial_carrots : ℕ := 685

-- Define the number of carrots Maria threw out
def thrown_out : ℕ := 156

-- Define the number of carrots Maria picked the next day
def picked_next_day : ℕ := 278

-- Define the total number of carrots Maria has after these actions
def total_carrots : ℕ :=
  initial_carrots - thrown_out + picked_next_day

-- The proof statement
theorem carrots_total : total_carrots = 807 := by
  sorry

end carrots_total_l295_295286


namespace sacks_after_6_days_l295_295745

theorem sacks_after_6_days (sacks_per_day : ℕ) (days : ℕ) 
  (h1 : sacks_per_day = 83) (h2 : days = 6) : 
  sacks_per_day * days = 498 :=
by
  sorry

end sacks_after_6_days_l295_295745


namespace correct_option_is_B_l295_295013

-- Definitions and conditions based on the problem
def is_monomial (t : String) : Prop :=
  t = "1"

def coefficient (expr : String) : Int :=
  if expr = "x" then 1
  else if expr = "-3x" then -3
  else 0

def degree (term : String) : Int :=
  if term = "5x^2y" then 3
  else 0

-- Proof statement
theorem correct_option_is_B : 
  is_monomial "1" ∧ ¬ (coefficient "x" = 0) ∧ ¬ (coefficient "-3x" = 3) ∧ ¬ (degree "5x^2y" = 2) := 
by
  -- Proof steps will go here
  sorry

end correct_option_is_B_l295_295013


namespace sin_405_eq_sqrt_2_div_2_l295_295682

theorem sin_405_eq_sqrt_2_div_2 : sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt_2_div_2_l295_295682


namespace percentage_of_720_equals_356_point_4_l295_295933

theorem percentage_of_720_equals_356_point_4 : 
  let part := 356.4
  let whole := 720
  (part / whole) * 100 = 49.5 :=
by
  sorry

end percentage_of_720_equals_356_point_4_l295_295933


namespace max_side_of_triangle_with_perimeter_30_l295_295206

theorem max_side_of_triangle_with_perimeter_30 
  (a b c : ℕ) 
  (h1 : a + b + c = 30) 
  (h2 : a ≥ b) 
  (h3 : b ≥ c) 
  (h4 : a < b + c) 
  (h5 : b < a + c) 
  (h6 : c < a + b) 
  : a ≤ 14 :=
sorry

end max_side_of_triangle_with_perimeter_30_l295_295206


namespace hiker_walking_speed_l295_295954

theorem hiker_walking_speed (v : ℝ) :
  (∃ (hiker_shares_cyclist_distance : 20 / 60 * v = 25 * (5 / 60)), v = 6.25) :=
by
  sorry

end hiker_walking_speed_l295_295954


namespace mixed_fractions_product_l295_295504

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l295_295504


namespace shifted_parabola_expression_l295_295763

theorem shifted_parabola_expression (x y x' y' : ℝ) 
  (h_initial : y = (x + 2)^2 + 3)
  (h_shift_right : x' = x - 3)
  (h_shift_down : y' = y - 2)
  : y' = (x' - 1)^2 + 1 := 
sorry

end shifted_parabola_expression_l295_295763


namespace units_digit_n_is_7_l295_295235

def units_digit (x : ℕ) : ℕ := x % 10

theorem units_digit_n_is_7 (m n : ℕ) (h1 : m * n = 31 ^ 4) (h2 : units_digit m = 6) :
  units_digit n = 7 :=
sorry

end units_digit_n_is_7_l295_295235


namespace problem1_problem2_problem3_problem4_l295_295535

theorem problem1 : -20 - (-14) + (-18) - 13 = -37 := by
  sorry

theorem problem2 : (-3/4 + 1/6 - 5/8) / (-1/24) = 29 := by
  sorry

theorem problem3 : -3^2 + (-3)^2 + 3 * 2 + |(-4)| = 10 := by
  sorry

theorem problem4 : 16 / (-2)^3 - (-1/6) * (-4) + (-1)^2024 = -5/3 := by
  sorry

end problem1_problem2_problem3_problem4_l295_295535


namespace max_ahn_achieve_max_ahn_achieve_attained_l295_295825

def is_two_digit_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem max_ahn_achieve :
  ∀ (n : ℕ), is_two_digit_integer n → 3 * (300 - n) ≤ 870 := 
by sorry

theorem max_ahn_achieve_attained :
  3 * (300 - 10) = 870 := 
by norm_num

end max_ahn_achieve_max_ahn_achieve_attained_l295_295825


namespace y_paisa_for_each_rupee_x_l295_295822

theorem y_paisa_for_each_rupee_x (p : ℕ) (x : ℕ) (y_share total_amount : ℕ) 
  (h₁ : y_share = 2700) 
  (h₂ : total_amount = 10500) 
  (p_condition : (130 + p) * x = total_amount) 
  (y_condition : p * x = y_share) : 
  p = 45 := 
by
  sorry

end y_paisa_for_each_rupee_x_l295_295822


namespace different_purchasing_methods_l295_295947

noncomputable def number_of_purchasing_methods (n_two_priced : ℕ) (n_one_priced : ℕ) (total_price : ℕ) : ℕ :=
  let combinations_two_price (k : ℕ) := Nat.choose n_two_priced k
  let combinations_one_price (k : ℕ) := Nat.choose n_one_priced k
  combinations_two_price 5 + (combinations_two_price 4 * combinations_one_price 2)

theorem different_purchasing_methods :
  number_of_purchasing_methods 8 3 10 = 266 :=
by
  sorry

end different_purchasing_methods_l295_295947


namespace coloring_problem_l295_295525

def condition (m n : ℕ) : Prop :=
  2 ≤ m ∧ m ≤ 31 ∧ 2 ≤ n ∧ n ≤ 31 ∧ m ≠ n ∧ m % n = 0

def color (f : ℕ → ℕ) : Prop :=
  ∀ m n, condition m n → f m ≠ f n

theorem coloring_problem :
  ∃ (k : ℕ) (f : ℕ → ℕ), (∀ n, 2 ≤ n ∧ n ≤ 31 → f n ≤ k) ∧ color f ∧ k = 4 :=
by
  sorry

end coloring_problem_l295_295525


namespace miles_driven_l295_295611

theorem miles_driven (rental_fee charge_per_mile total_amount_paid : ℝ) (h₁ : rental_fee = 20.99) (h₂ : charge_per_mile = 0.25) (h₃ : total_amount_paid = 95.74) :
  (total_amount_paid - rental_fee) / charge_per_mile = 299 :=
by
  -- Placeholder for proof
  sorry

end miles_driven_l295_295611


namespace point_p_locus_equation_l295_295789

noncomputable def locus_point_p (x y : ℝ) : Prop :=
  ∀ (k b x1 y1 x2 y2 : ℝ), 
  (x1^2 + y1^2 = 1) ∧ 
  (x2^2 + y2^2 = 1) ∧ 
  (3 * x1 * x + 4 * y1 * y = 12) ∧ 
  (3 * x2 * x + 4 * y2 * y = 12) ∧ 
  (1 + k^2 = b^2) ∧ 
  (y = 3 / b) ∧ 
  (x = -4 * k / (3 * b)) → 
  x^2 / 16 + y^2 / 9 = 1

theorem point_p_locus_equation :
  ∀ (x y : ℝ), locus_point_p x y → (x^2 / 16 + y^2 / 9 = 1) :=
by
  intros x y h
  sorry

end point_p_locus_equation_l295_295789


namespace determine_a_b_l295_295087

-- Definitions
def num (a b : ℕ) := 10000*a + 1000*6 + 100*7 + 10*9 + b

def divisible_by_72 (n : ℕ) : Prop := n % 72 = 0

noncomputable def a : ℕ := 3
noncomputable def b : ℕ := 2

-- Theorem statement
theorem determine_a_b : divisible_by_72 (num a b) :=
by
  -- The proof will be inserted here
  sorry

end determine_a_b_l295_295087


namespace money_left_l295_295290

theorem money_left (olivia_money nigel_money ticket_cost tickets_purchased : ℕ) 
  (h1 : olivia_money = 112) 
  (h2 : nigel_money = 139) 
  (h3 : ticket_cost = 28) 
  (h4 : tickets_purchased = 6) : 
  olivia_money + nigel_money - tickets_purchased * ticket_cost = 83 := 
by 
  sorry

end money_left_l295_295290


namespace bugs_diagonally_at_least_9_unoccupied_l295_295749

theorem bugs_diagonally_at_least_9_unoccupied (bugs : ℕ × ℕ → Prop) :
  let board_size := 9
  let cells := (board_size * board_size)
  let black_cells := 45
  let white_cells := 36
  ∃ unoccupied_cells ≥ 9, true := 
sorry

end bugs_diagonally_at_least_9_unoccupied_l295_295749


namespace min_area_triangle_l295_295881

theorem min_area_triangle (m n : ℝ) (h1 : (1 : ℝ) / m + (2 : ℝ) / n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∃ A B C : ℝ, 
  ((0 < A) ∧ (0 < B) ∧ ((1 : ℝ) / A + (2 : ℝ) / B = 1) ∧ (A * B = C) ∧ (2 / C = mn)) ∧ (C = 4) :=
by
  sorry

end min_area_triangle_l295_295881


namespace empty_one_container_l295_295638

theorem empty_one_container (a b c : ℕ) :
  ∃ a' b' c', (a' = 0 ∨ b' = 0 ∨ c' = 0) ∧
    (a' = a ∧ b' = b ∧ c' = c ∨
     (a' ≤ a ∧ b' ≤ b ∧ c' ≤ c ∧ (a + b + c = a' + b' + c')) ∧
     (∀ i j, i ≠ j → (i = 1 ∨ i = 2 ∨ i = 3) →
              (j = 1 ∨ j = 2 ∨ j = 3) →
              (if i = 1 then (if j = 2 then a' = a - a ∨ a' = a else (if j = 3 then a' = a - a ∨ a' = a else false))
               else if i = 2 then (if j = 1 then b' = b - b ∨ b' = b else (if j = 3 then b' = b - b ∨ b' = b else false))
               else (if j = 1 then c' = c - c ∨ c' = c else (if j = 2 then c' = c - c ∨ c' = c else false))))) :=
by
  sorry

end empty_one_container_l295_295638


namespace arithmetic_mean_frac_l295_295307

theorem arithmetic_mean_frac (y b : ℝ) (h : y ≠ 0) : 
  (1 / 2 : ℝ) * ((y + b) / y + (2 * y - b) / y) = 1.5 := 
by 
  sorry

end arithmetic_mean_frac_l295_295307


namespace positive_integer_k_l295_295974

theorem positive_integer_k (k x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = k * x * y * z) :
  k = 1 ∨ k = 3 :=
sorry

end positive_integer_k_l295_295974


namespace closest_perfect_square_to_273_l295_295176

theorem closest_perfect_square_to_273 : ∃ n : ℕ, (n^2 = 289) ∧ 
  ∀ m : ℕ, (m^2 < 273 → 273 - m^2 ≥ 1) ∧ (m^2 > 273 → m^2 - 273 ≥ 16) :=
by
  sorry

end closest_perfect_square_to_273_l295_295176


namespace sin_405_eq_sqrt2_div2_l295_295684

theorem sin_405_eq_sqrt2_div2 : Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_405_eq_sqrt2_div2_l295_295684


namespace participants_won_more_than_lost_l295_295778

-- Define the conditions given in the problem
def total_participants := 64
def rounds := 6

-- Define a function that calculates the number of participants reaching a given round
def participants_after_round (n : Nat) (r : Nat) : Nat :=
  n / (2 ^ r)

-- The theorem we need to prove
theorem participants_won_more_than_lost :
  participants_after_round total_participants 2 = 16 :=
by 
  -- Provide a placeholder for the proof
  sorry

end participants_won_more_than_lost_l295_295778


namespace period_change_l295_295727

theorem period_change {f : ℝ → ℝ} (T : ℝ) (hT : 0 < T) (h_period : ∀ x, f (x + T) = f x) (α : ℝ) (hα : 0 < α) :
  ∀ x, f (α * (x + T / α)) = f (α * x) :=
by
  sorry

end period_change_l295_295727


namespace frac_equiv_l295_295983

theorem frac_equiv (a b : ℚ) (h : a / b = 3 / 4) : (a - b) / (a + b) = -1 / 7 := by
  sorry

end frac_equiv_l295_295983


namespace fraction_power_multiplication_l295_295796

theorem fraction_power_multiplication :
  ( (8 / 9)^3 * (5 / 3)^3 ) = (64000 / 19683) :=
by
  sorry

end fraction_power_multiplication_l295_295796


namespace correct_answer_l295_295719

-- Define the sentence structure and the requirement for a formal object
structure SentenceStructure where
  subject : String := "I"
  verb : String := "like"
  object_placeholder : String := "_"
  clause : String := "when the weather is clear and bright"

-- Correct choices provided
inductive Choice
  | this
  | that
  | it
  | one

-- Problem formulation: Based on SentenceStructure, prove that 'it' is the correct choice
theorem correct_answer {S : SentenceStructure} : Choice.it = Choice.it :=
by
  -- Proof omitted
  sorry

end correct_answer_l295_295719


namespace b_range_given_conditions_l295_295575

theorem b_range_given_conditions 
    (b c : ℝ)
    (roots_in_interval : ∀ x, x^2 + b * x + c = 0 → -1 ≤ x ∧ x ≤ 1)
    (ineq : 0 ≤ 3 * b + c ∧ 3 * b + c ≤ 3) :
    0 ≤ b ∧ b ≤ 2 :=
sorry

end b_range_given_conditions_l295_295575


namespace binomial_510_510_l295_295048

theorem binomial_510_510 : binomial 510 510 = 1 :=
by sorry

end binomial_510_510_l295_295048


namespace solve_for_A_l295_295459

theorem solve_for_A (A B : ℕ) (h1 : 4 * 10 + A + 10 * B + 3 = 68) (h2 : 10 ≤ 4 * 10 + A) (h3 : 4 * 10 + A < 100) (h4 : 10 ≤ 10 * B + 3) (h5 : 10 * B + 3 < 100) (h6 : A < 10) (h7 : B < 10) : A = 5 := 
by
  sorry

end solve_for_A_l295_295459


namespace percentage_less_than_l295_295658

theorem percentage_less_than (p j t : ℝ) (h1 : j = 0.75 * p) (h2 : j = 0.80 * t) : 
  t = (1 - 0.0625) * p := 
by 
  sorry

end percentage_less_than_l295_295658


namespace inequality_proof_l295_295566

theorem inequality_proof
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 ≥ x2)
  (h2 : x2 ≥ x3)
  (h3 : x3 ≥ x4)
  (h4 : x2 + x3 + x4 ≥ x1) :
  (x1 + x2 + x3 + x4)^2 ≤ 4 * x1 * x2 * x3 * x4 := 
by
  sorry

end inequality_proof_l295_295566


namespace monotonic_intervals_l295_295577

open Set

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * a * x^3 + x^2 + 1

theorem monotonic_intervals (a : ℝ) (h : a ≤ 0) :
  (a = 0 → (∀ x : ℝ, (x < 0 → deriv (f a) x < 0) ∧ (0 < x → deriv (f a) x > 0))) ∧
  (a < 0 → (∀ x : ℝ, (x < 2 / a → deriv (f a) x > 0 ∨ deriv (f a) x = 0) ∧ 
                     (2 / a < x → deriv (f a) x < 0 ∨ deriv (f a) x = 0))) :=
by
  sorry

end monotonic_intervals_l295_295577


namespace badArrangementsCount_l295_295314

noncomputable def numberIsBadArrangements (A : Finset (Finset ℕ)) :=
  (∀ n ∈ (Finset.range 21).erase 0, ∃ S ∈ A, S.sum = n) = false

noncomputable def countBadArrangements :=
  let numbers := {1, 2, 3, 4, 5, 6}
  let circularArrangements := Finset.univ.filter (λ s : Finset ℕ, s.card = numbers.card ∧ s ⊆ numbers)
  (circularArrangements.filter numberIsBadArrangements).card / 2

theorem badArrangementsCount : countBadArrangements = 5 :=
by
  sorry

end badArrangementsCount_l295_295314


namespace complementary_angles_positive_difference_l295_295312

theorem complementary_angles_positive_difference
  (x : ℝ)
  (h1 : 3 * x + x = 90): 
  |(3 * x) - x| = 45 := 
by
  -- Proof would go here (details skipped)
  sorry

end complementary_angles_positive_difference_l295_295312


namespace expression_value_l295_295920

theorem expression_value : 2 + 3 * 5 + 2 = 19 := by
  sorry

end expression_value_l295_295920


namespace Ana_age_eight_l295_295891

theorem Ana_age_eight (A B n : ℕ) (h1 : A - 1 = 7 * (B - 1)) (h2 : A = 4 * B) (h3 : A - B = n) : A = 8 :=
by
  sorry

end Ana_age_eight_l295_295891


namespace profit_at_15_percent_off_l295_295191

theorem profit_at_15_percent_off 
    (cost_price marked_price : ℝ) 
    (cost_price_eq : cost_price = 2000)
    (marked_price_eq : marked_price = (200 + cost_price) / 0.8) :
    (0.85 * marked_price - cost_price) = 337.5 := by
  sorry

end profit_at_15_percent_off_l295_295191


namespace roger_and_friend_fraction_l295_295127

theorem roger_and_friend_fraction 
  (total_distance : ℝ) 
  (fraction_driven_before_lunch : ℝ) 
  (lunch_time : ℝ) 
  (total_time : ℝ) 
  (same_speed : Prop) 
  (driving_time_before_lunch : ℝ)
  (driving_time_after_lunch : ℝ) :
  total_distance = 200 ∧
  lunch_time = 1 ∧
  total_time = 5 ∧
  driving_time_before_lunch = 1 ∧
  driving_time_after_lunch = (total_time - lunch_time - driving_time_before_lunch) ∧
  same_speed = (total_distance * fraction_driven_before_lunch / driving_time_before_lunch = total_distance * (1 - fraction_driven_before_lunch) / driving_time_after_lunch) →
  fraction_driven_before_lunch = 1 / 4 :=
sorry

end roger_and_friend_fraction_l295_295127


namespace smallest_d_for_inverse_l295_295417

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse (d : ℝ) : 
  (∀ x1 x2 : ℝ, d ≤ x1 → d ≤ x2 → g x1 = g x2 → x1 = x2) → d = 3 :=
by
  sorry

end smallest_d_for_inverse_l295_295417


namespace length_of_DF_l295_295425

theorem length_of_DF
  (D E F P Q: Type)
  (DP: ℝ)
  (EQ: ℝ)
  (h1: DP = 27)
  (h2: EQ = 36)
  (perp: ∀ (u v: Type), u ≠ v):
  ∃ (DF: ℝ), DF = 4 * Real.sqrt 117 :=
by
  sorry

end length_of_DF_l295_295425


namespace number_of_representations_l295_295105

-- Definitions of the conditions
def is_valid_b (b : ℕ) : Prop :=
  b ≤ 99

def is_representation (b3 b2 b1 b0 : ℕ) : Prop :=
  3152 = b3 * 10^3 + b2 * 10^2 + b1 * 10 + b0

-- The theorem to prove
theorem number_of_representations : 
  ∃ (N' : ℕ), (N' = 316) ∧ 
  (∀ (b3 b2 b1 b0 : ℕ), is_representation b3 b2 b1 b0 → is_valid_b b0 → is_valid_b b1 → is_valid_b b2 → is_valid_b b3) :=
sorry

end number_of_representations_l295_295105


namespace find_x_approx_l295_295811

theorem find_x_approx :
  ∀ (x : ℝ), 3639 + 11.95 - x^2 = 3054 → abs (x - 24.43) < 0.01 :=
by
  intro x
  sorry

end find_x_approx_l295_295811


namespace sales_tax_percentage_l295_295818

noncomputable def original_price : ℝ := 200
noncomputable def discount : ℝ := 0.25 * original_price
noncomputable def sale_price : ℝ := original_price - discount
noncomputable def total_paid : ℝ := 165
noncomputable def sales_tax : ℝ := total_paid - sale_price

theorem sales_tax_percentage : (sales_tax / sale_price) * 100 = 10 := by
  sorry

end sales_tax_percentage_l295_295818


namespace smallest_four_digit_divisible_by_35_l295_295143

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l295_295143


namespace avg_values_l295_295697

theorem avg_values (z : ℝ) : (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end avg_values_l295_295697


namespace maximum_side_length_of_triangle_l295_295200

theorem maximum_side_length_of_triangle (a b c : ℕ) (h_diff: a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_perimeter: a + b + c = 30)
  (h_triangle_inequality_1: a + b > c) 
  (h_triangle_inequality_2: a + c > b) 
  (h_triangle_inequality_3: b + c > a) : 
  c ≤ 14 :=
sorry

end maximum_side_length_of_triangle_l295_295200


namespace problem_statement_l295_295090

def p (x y : ℝ) : Prop :=
  (x^2 + y^2 ≠ 0) → ¬ (x = 0 ∧ y = 0)

def q (m : ℝ) : Prop :=
  (m > -2) → ∃ x : ℝ, x^2 + 2*x - m = 0

theorem problem_statement : ∀ (x y m : ℝ), p x y ∨ q m :=
sorry

end problem_statement_l295_295090


namespace ratio_of_volumes_l295_295473

theorem ratio_of_volumes (s : ℝ) (hs : s > 0) :
  let r_s := s / 2
  let r_c := s / 2
  let V_sphere := (4 / 3) * π * (r_s ^ 3)
  let V_cylinder := π * (r_c ^ 2) * s
  let V_total := V_sphere + V_cylinder
  let V_cube := s ^ 3
  V_total / V_cube = (5 * π) / 12 := by {
    -- Given the conditions and expressions
    sorry
  }

end ratio_of_volumes_l295_295473


namespace necessary_but_not_sufficient_condition_for_x_equals_0_l295_295991

theorem necessary_but_not_sufficient_condition_for_x_equals_0 (x : ℝ) :
  ((2 * x - 1) * x = 0 → x = 0 ∨ x = 1 / 2) ∧ (x = 0 → (2 * x - 1) * x = 0) ∧ ¬((2 * x - 1) * x = 0 → x = 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_for_x_equals_0_l295_295991


namespace infinitely_many_n_l295_295913

theorem infinitely_many_n (S : Set ℕ) :
  (∀ n ∈ S, n > 0 ∧ (n ∣ 2 ^ (2 ^ n + 1) + 1) ∧ ¬ (n ∣ 2 ^ n + 1)) ∧ S.Infinite :=
sorry

end infinitely_many_n_l295_295913


namespace max_modulus_l295_295382

theorem max_modulus (z : ℂ) (h : |z - 2 * complex.I| = 1) : |z| ≤ 3 :=
begin
  sorry
end

end max_modulus_l295_295382


namespace remainder_sum_remainders_mod_500_l295_295605

open Nat

/-- Define the set of remainders of 3^n mod 500 for nonnegative integers n -/
def remainders_mod_500 : Set ℕ := {r | ∃ n : ℕ, r = 3^n % 500}

/-- Define the sum of the elements in the set of remainders -/
def S : ℕ := remainders_mod_500.sum (λ x, x)

theorem remainder_sum_remainders_mod_500 (x : ℕ)
  (hx : S % 500 = x) :
  S % 500 = x := by
  sorry

end remainder_sum_remainders_mod_500_l295_295605


namespace smallest_four_digit_divisible_by_35_l295_295163

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l295_295163


namespace domain_of_f_l295_295117

def f (x : ℝ) : ℝ := (1 / (x + 1)) + Real.log x

theorem domain_of_f :
  ∀ x : ℝ, (1 / (x + 1)) + Real.log x ∈ ℝ → x > 0 :=
by
  intro x
  split
  sorry

end domain_of_f_l295_295117


namespace bullet_train_time_pass_man_l295_295813

variable (length_of_train : ℝ) (speed_of_train_kmph : ℝ) (speed_of_man_kmph : ℝ)

def relative_speed_kmph : ℝ :=
  speed_of_train_kmph + speed_of_man_kmph

def relative_speed_mps : ℝ :=
  relative_speed_kmph * (1000 / 3600)

def time_to_pass : ℝ :=
  length_of_train / relative_speed_mps

theorem bullet_train_time_pass_man :
  length_of_train = 120 →
  speed_of_train_kmph = 50 →
  speed_of_man_kmph = 4 →
  time_to_pass length_of_train speed_of_train_kmph speed_of_man_kmph = 8 :=
by
  intros
  sorry

end bullet_train_time_pass_man_l295_295813


namespace sum_first_ten_terms_arithmetic_sequence_l295_295799

theorem sum_first_ten_terms_arithmetic_sequence (a₁ d : ℤ) (h₁ : a₁ = -3) (h₂ : d = 4) : 
  let a₁₀ := a₁ + (9 * d)
  let S := ((a₁ + a₁₀) / 2) * 10
  S = 150 :=
by
  subst h₁
  subst h₂
  let a₁₀ := -3 + (9 * 4)
  let S := ((-3 + a₁₀) / 2) * 10
  sorry

end sum_first_ten_terms_arithmetic_sequence_l295_295799


namespace total_balls_l295_295129

theorem total_balls (S V B Total : ℕ) (hS : S = 68) (hV : S = V - 12) (hB : S = B + 23) : 
  Total = S + V + B := by
  sorry

end total_balls_l295_295129


namespace proof_problem_l295_295869

noncomputable def problem_conditions : Type :=
  Σ α : ℝ, (π / 2 < α) ∧ (α < 3 * π / 2) ∧ (∀ (A B : ℝ × ℝ) (hA : A = (3, 0)) (hB : B = (0, 3)),
    let C := (Real.cos α, Real.sin α),
    let AC := ((Real.cos α) - 3, (Real.sin α)),
    let BC := ((Real.cos α) - 0, (Real.sin α) - 3),
    ‖AC‖ = ‖BC‖ → α = 5 * π / 4)

noncomputable def second_problem_conditions : Type :=
  Σ α : ℝ, (α = 5 * π / 4) ∧ (let C := (Real.cos α, Real.sin α),
    let AC := ((Real.cos α) - 3, (Real.sin α)),
    let BC := ((Real.cos α) - 0, (Real.sin α) - 3),
    ACdotBC : ℝ := (AC.1 * BC.1 + AC.2 * BC.2),
    ACdotBC = -1 →
    (2 * (Real.sin α)^2 + 2 * (Real.sin α) * (Real.cos α)) / (1 + (Real.tan α)) = -(5 / 9))

-- The final theorem
theorem proof_problem :
  ∃ α : ℝ, (π / 2 < α) ∧ (α < 3 * π / 2) ∧
  (let C := (Real.cos α, Real.sin α),
   let AC := ((Real.cos α) - 3, (Real.sin α)),
   let BC := ((Real.cos α) - 0, (Real.sin α) - 3),
   ‖AC‖ = ‖BC‖) → (α = 5 * π / 4) ∧
  (let AC := ((Real.cos α) - 3, (Real.sin α)),
   let BC := ((Real.cos α) - 0, (Real.sin α) - 3),
   (AC.1 * BC.1 + AC.2 * BC.2 = -1) →
   ((2 * (Real.sin α)^2 + 2 * (Real.sin α) * (Real.cos α)) / (1 + (Real.tan α)) = -(5 / 9))) :=
begin
  sorry
end

end proof_problem_l295_295869


namespace find_real_pairs_l295_295850

theorem find_real_pairs (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by
  sorry

end find_real_pairs_l295_295850


namespace correct_factorization_l295_295647

theorem correct_factorization :
  (x^2 - 2 * x + 1 = (x - 1)^2) ∧ 
  (¬ (x^2 - 4 * y^2 = (x + y) * (x - 4 * y))) ∧ 
  (¬ ((x + 4) * (x - 4) = x^2 - 16)) ∧ 
  (¬ (x^2 - 8 * x + 9 = (x - 4)^2 - 7)) :=
by
  sorry

end correct_factorization_l295_295647


namespace find_unique_positive_integer_pair_l295_295556

theorem find_unique_positive_integer_pair :
  ∃! (b c : ℕ), b > 0 ∧ c > 0 ∧ c > b^2 ∧ b > c^2 :=
sorry

end find_unique_positive_integer_pair_l295_295556


namespace puzzle_pieces_missing_l295_295135

/-- Trevor and Joe were working together to finish a 500 piece puzzle. 
They put the border together first and that was 75 pieces. 
Trevor was able to place 105 pieces of the puzzle.
Joe was able to place three times the number of puzzle pieces as Trevor. 
Prove that the number of puzzle pieces missing is 5. -/
theorem puzzle_pieces_missing :
  let total_pieces := 500
  let border_pieces := 75
  let trevor_pieces := 105
  let joe_pieces := 3 * trevor_pieces
  let placed_pieces := trevor_pieces + joe_pieces
  let remaining_pieces := total_pieces - border_pieces
  remaining_pieces - placed_pieces = 5 :=
by
  sorry

end puzzle_pieces_missing_l295_295135


namespace option_A_sufficient_not_necessary_l295_295560

variable (a b : ℝ)

def A : Set ℝ := { x | x^2 - x + a ≤ 0 }
def B : Set ℝ := { x | x^2 - x + b ≤ 0 }

theorem option_A_sufficient_not_necessary : (A = B → a = b) ∧ (a = b → A = B) :=
by
  sorry

end option_A_sufficient_not_necessary_l295_295560


namespace pencils_ordered_l295_295472

theorem pencils_ordered (pencils_per_student : ℕ) (number_of_students : ℕ) (total_pencils : ℕ) :
  pencils_per_student = 3 →
  number_of_students = 65 →
  total_pencils = pencils_per_student * number_of_students →
  total_pencils = 195 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end pencils_ordered_l295_295472


namespace find_cd_l295_295458

noncomputable def g (x : ℝ) (c : ℝ) (d : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

theorem find_cd (c d : ℝ) :
  g 2 c d = -9 ∧ g (-1) c d = -19 ↔
  (c = 19/3 ∧ d = -7/3) :=
by
  sorry

end find_cd_l295_295458


namespace bd_squared_l295_295261

theorem bd_squared (a b c d : ℤ) (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 9) : 
  (b - d) ^ 2 = 4 := 
sorry

end bd_squared_l295_295261


namespace jellybeans_red_l295_295816

-- Define the individual quantities of each color of jellybean.
def b := 14
def p := 26
def o := 40
def pk := 7
def y := 21
def T := 237

-- Prove that the number of red jellybeans is 129.
theorem jellybeans_red : T - (b + p + o + pk + y) = 129 := by
  -- (optional: you can include intermediate steps if needed, but it's not required here)
  sorry

end jellybeans_red_l295_295816


namespace area_of_bounded_region_l295_295310

theorem area_of_bounded_region (x y : ℝ) (h : y^2 + 2 * x * y + 50 * abs x = 500) : 
  ∃ A, A = 1250 :=
sorry

end area_of_bounded_region_l295_295310


namespace inverse_function_evaluation_l295_295998

def g (x : ℕ) : ℕ :=
  if x = 1 then 4
  else if x = 2 then 5
  else if x = 3 then 2
  else if x = 4 then 3
  else if x = 5 then 1
  else 0  -- default case, though it shouldn't be used given the conditions

noncomputable def g_inv (y : ℕ) : ℕ :=
  if y = 4 then 1
  else if y = 5 then 2
  else if y = 2 then 3
  else if y = 3 then 4
  else if y = 1 then 5
  else 0  -- default case, though it shouldn't be used given the conditions

theorem inverse_function_evaluation : g_inv (g_inv (g_inv 4)) = 2 := by
  sorry

end inverse_function_evaluation_l295_295998


namespace det_2x2_matrix_l295_295967

open Matrix

theorem det_2x2_matrix : 
  det ![![7, -2], ![-3, 5]] = 29 := by
  sorry

end det_2x2_matrix_l295_295967


namespace chairs_left_to_move_l295_295681

theorem chairs_left_to_move (total_chairs : ℕ) (carey_chairs : ℕ) (pat_chairs : ℕ) (h1 : total_chairs = 74)
  (h2 : carey_chairs = 28) (h3 : pat_chairs = 29) : total_chairs - carey_chairs - pat_chairs = 17 :=
by 
  sorry

end chairs_left_to_move_l295_295681


namespace median_of_consecutive_integers_l295_295784

theorem median_of_consecutive_integers (sum_n : ℤ) (n : ℤ) 
  (h1 : sum_n = 6^4) (h2 : n = 36) : 
  (sum_n / n) = 36 :=
by
  sorry

end median_of_consecutive_integers_l295_295784


namespace value_of_m_l295_295402

theorem value_of_m
  (x y m : ℝ)
  (h1 : 2 * x + 3 * y = 4)
  (h2 : 3 * x + 2 * y = 2 * m - 3)
  (h3 : x + y = -3/5) :
  m = -2 :=
sorry

end value_of_m_l295_295402


namespace largest_multiple_of_8_less_than_100_l295_295004

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), (n % 8 = 0) ∧ (n < 100) ∧ (∀ m : ℕ, (m % 8 = 0) ∧ (m < 100) → m ≤ n) :=
begin
  use 96,
  split,
  { -- 96 is a multiple of 8
    exact nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 96),
  },
  split,
  { -- 96 is less than 100
    norm_num,
  },
  { -- 96 is the largest multiple of 8 less than 100
    intros m hm,
    obtain ⟨k, rfl⟩ := (nat.dvd_iff_mod_eq_zero.mp hm.1),
    have : k ≤ 12, by linarith,
    linarith [mul_le_mul (zero_le _ : (0 : ℕ) ≤ 8) this (zero_le _ : (0 : ℕ) ≤ 12) (zero_le _ : (0 : ℕ) ≤ 8)],
  },
end

end largest_multiple_of_8_less_than_100_l295_295004


namespace cost_of_western_european_postcards_before_1980s_l295_295944

def germany_cost_1950s : ℝ := 5 * 0.07
def france_cost_1950s : ℝ := 8 * 0.05

def germany_cost_1960s : ℝ := 6 * 0.07
def france_cost_1960s : ℝ := 9 * 0.05

def germany_cost_1970s : ℝ := 11 * 0.07
def france_cost_1970s : ℝ := 10 * 0.05

def total_germany_cost : ℝ := germany_cost_1950s + germany_cost_1960s + germany_cost_1970s
def total_france_cost : ℝ := france_cost_1950s + france_cost_1960s + france_cost_1970s

def total_western_europe_cost : ℝ := total_germany_cost + total_france_cost

theorem cost_of_western_european_postcards_before_1980s :
  total_western_europe_cost = 2.89 := by
  sorry

end cost_of_western_european_postcards_before_1980s_l295_295944


namespace gcd_105_88_l295_295979

-- Define the numbers as constants
def a : ℕ := 105
def b : ℕ := 88

-- State the theorem: gcd(a, b) = 1
theorem gcd_105_88 : Nat.gcd a b = 1 := by
  sorry

end gcd_105_88_l295_295979


namespace problem_proof_l295_295171

def problem_statement : Prop :=
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 * (1 / 4096) * 8192 = 64

theorem problem_proof : problem_statement := by
  sorry

end problem_proof_l295_295171


namespace cos_105_degree_value_l295_295550

noncomputable def cos105 : ℝ := Real.cos (105 * Real.pi / 180)

theorem cos_105_degree_value :
  cos105 = (Real.sqrt 2 - Real.sqrt 6) / 4 :=
by
  sorry

end cos_105_degree_value_l295_295550


namespace family_four_children_includes_at_least_one_boy_one_girl_l295_295530

-- Specification of the probability function
def prob_event (n : ℕ) (event : fin n → bool) : ℚ := 
  (Real.to_rat (Real.exp (- (Real.nat_to_real (nat.log2 n)))) : ℚ)

-- Predicate that checks if there is at least one boy and one girl in the list
def has_boy_and_girl (children : fin 4 → bool) : Prop :=
  ∃ i j, children i ≠ children j

theorem family_four_children_includes_at_least_one_boy_one_girl : 
  (∑ event in (finset.univ : finset (fin 4 → bool)), 
     if has_boy_and_girl event then prob_event 4 event else 0) = 7 / 8 :=
by
  sorry

end family_four_children_includes_at_least_one_boy_one_girl_l295_295530


namespace inequality_holds_for_n_ge_0_l295_295848

theorem inequality_holds_for_n_ge_0
  (n : ℤ)
  (h : n ≥ 0)
  (a b c x y z : ℝ)
  (Habc : 0 < a ∧ 0 < b ∧ 0 < c)
  (Hxyz : 0 < x ∧ 0 < y ∧ 0 < z)
  (Hmax : max a (max b (max c (max x (max y z)))) = a)
  (Hsum : a + b + c = x + y + z)
  (Hprod : a * b * c = x * y * z) : a^n + b^n + c^n ≥ x^n + y^n + z^n := 
sorry

end inequality_holds_for_n_ge_0_l295_295848


namespace smallest_four_digit_divisible_by_35_l295_295166

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l295_295166


namespace solve_x_l295_295320

-- Define the function f with the given properties
axiom f : ℝ → ℝ → ℝ
axiom f_assoc : ∀ (a b c : ℝ), f a (f b c) = f (f a b) c
axiom f_inv : ∀ (a : ℝ), f a a = 1

-- Define x and the equation to be solved
theorem solve_x : ∃ (x : ℝ), f x 36 = 216 :=
  sorry

end solve_x_l295_295320


namespace min_value_expression_l295_295722

variable {a b : ℝ}

theorem min_value_expression
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : a + b = 4) : 
  (∃ C, (∀ a b, a > 0 → b > 0 → a + b = 4 → (b / a + 4 / b) ≥ C) ∧ 
         (∀ a b, a > 0 → b > 0 → a + b = 4 → (b / a + 4 / b) = C)) ∧ 
         C = 3 :=
  by sorry

end min_value_expression_l295_295722


namespace arrangement_correct_l295_295018

def A := 4
def B := 1
def C := 2
def D := 5
def E := 6
def F := 3

def sum1 := A + B + C
def sum2 := A + D + F
def sum3 := B + E + D
def sum4 := C + F + E
def sum5 := A + E + F
def sum6 := B + D + C
def sum7 := B + C + F

theorem arrangement_correct :
  sum1 = 15 ∧ sum2 = 15 ∧ sum3 = 15 ∧ sum4 = 15 ∧ sum5 = 15 ∧ sum6 = 15 ∧ sum7 = 15 := 
by
  unfold sum1 sum2 sum3 sum4 sum5 sum6 sum7 
  unfold A B C D E F
  sorry

end arrangement_correct_l295_295018


namespace number_of_students_l295_295602

theorem number_of_students (T : ℕ) (n : ℕ) (h1 : (T + 20) / n = T / n + 1 / 2) : n = 40 :=
  sorry

end number_of_students_l295_295602


namespace maximum_side_length_range_l295_295242

variable (P : ℝ)
variable (a b c : ℝ)
variable (h1 : a + b + c = P)
variable (h2 : a ≤ b)
variable (h3 : b ≤ c)
variable (h4 : a + b > c)

theorem maximum_side_length_range : 
  (P / 3) ≤ c ∧ c < (P / 2) :=
by
  sorry

end maximum_side_length_range_l295_295242


namespace chess_team_selection_l295_295908

theorem chess_team_selection
  (players : Finset ℕ) (twin1 twin2 : ℕ)
  (H1 : players.card = 10)
  (H2 : twin1 ∈ players)
  (H3 : twin2 ∈ players) :
  ∃ n : ℕ, n = 182 ∧ 
  (∃ team : Finset ℕ, team.card = 4 ∧
    (twin1 ∉ team ∨ twin2 ∉ team)) ∧
  n = (players.card.choose 4 - 
      ((players.erase twin1).erase twin2).card.choose 2) := sorry

end chess_team_selection_l295_295908


namespace find_total_cows_l295_295464

-- Definitions as per the conditions
variables (D C L H : ℕ)

-- Condition 1: Total number of legs
def total_legs : ℕ := 2 * D + 4 * C

-- Condition 2: Total number of heads
def total_heads : ℕ := D + C

-- Condition 3: Legs are 28 more than twice the number of heads
def legs_heads_relation : Prop := total_legs D C = 2 * total_heads D C + 28

-- The theorem to prove
theorem find_total_cows (h : legs_heads_relation D C) : C = 14 :=
sorry

end find_total_cows_l295_295464


namespace born_in_1890_l295_295955

theorem born_in_1890 (x : ℕ) (h1 : x^2 - x - 2 = 1890) (h2 : x^2 < 1950) : x = 44 :=
by {
    sorry
}

end born_in_1890_l295_295955


namespace composite_expr_l295_295858

open Nat

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

theorem composite_expr (n : ℕ) : n ≥ 2 ↔ is_composite (3^(2*n + 1) - 2^(2*n + 1) - 6^n) :=
sorry

end composite_expr_l295_295858


namespace Karl_miles_driven_l295_295890

theorem Karl_miles_driven
  (gas_per_mile : ℝ)
  (tank_capacity : ℝ)
  (initial_gas : ℝ)
  (first_leg_miles : ℝ)
  (refuel_gallons : ℝ)
  (final_gas_fraction : ℝ)
  (total_miles_driven : ℝ) :
  gas_per_mile = 30 →
  tank_capacity = 16 →
  initial_gas = 16 →
  first_leg_miles = 420 →
  refuel_gallons = 10 →
  final_gas_fraction = 3 / 4 →
  total_miles_driven = 420 :=
by
  sorry

end Karl_miles_driven_l295_295890


namespace Jed_cards_after_4_weeks_l295_295599

theorem Jed_cards_after_4_weeks :
  (∀ n: ℕ, (if n % 2 = 0 then 20 + 4*n - 2*n else 20 + 4*n - 2*(n-1)) = 40) :=
by {
  sorry
}

end Jed_cards_after_4_weeks_l295_295599


namespace binary_multiplication_addition_l295_295678

-- Define the binary representation of the given numbers
def b1101 : ℕ := 0b1101
def b111 : ℕ := 0b111
def b1011 : ℕ := 0b1011
def b1011010 : ℕ := 0b1011010

-- State the theorem
theorem binary_multiplication_addition :
  (b1101 * b111 + b1011) = b1011010 := 
sorry

end binary_multiplication_addition_l295_295678


namespace sum_mod_9_l295_295373

theorem sum_mod_9 : (7155 + 7156 + 7157 + 7158 + 7159) % 9 = 1 :=
by sorry

end sum_mod_9_l295_295373


namespace bet_strategy_possible_l295_295333

def betting_possibility : Prop :=
  (1 / 6 + 1 / 2 + 1 / 9 + 1 / 8 <= 1)

theorem bet_strategy_possible : betting_possibility :=
by
  -- Proof is intentionally omitted
  sorry

end bet_strategy_possible_l295_295333


namespace area_inequality_l295_295856

theorem area_inequality (a b c d S : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (hS : 0 ≤ S) :
  S ≤ (a + c) / 2 * (b + d) / 2 :=
by
  sorry

end area_inequality_l295_295856


namespace remaining_money_is_83_l295_295292

noncomputable def OliviaMoney : ℕ := 112
noncomputable def NigelMoney : ℕ := 139
noncomputable def TicketCost : ℕ := 28
noncomputable def TicketsBought : ℕ := 6

def TotalMoney : ℕ := OliviaMoney + NigelMoney
def TotalCost : ℕ := TicketsBought * TicketCost
def RemainingMoney : ℕ := TotalMoney - TotalCost

theorem remaining_money_is_83 : RemainingMoney = 83 := by
  sorry

end remaining_money_is_83_l295_295292


namespace complement_union_l295_295878

namespace SetTheory

def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

theorem complement_union (U A B: Set ℕ) (hU : U = {1, 3, 5, 9}) (hA : A = {1, 3, 9}) (hB : B = {1, 9}) :
  U \ (A ∪ B) = {5} :=
by
  sorry

end SetTheory

end complement_union_l295_295878


namespace median_of_36_consecutive_integers_l295_295786

theorem median_of_36_consecutive_integers (x : ℤ) (sum_eq : (∑ i in finset.range 36, (x + i)) = 6^4) : (17 + 18) / 2 = 36 :=
by
  -- Proof goes here
  sorry

end median_of_36_consecutive_integers_l295_295786


namespace average_production_l295_295939

theorem average_production (n : ℕ) :
  let total_past_production := 50 * n
  let total_production_including_today := 100 + total_past_production
  let average_production := total_production_including_today / (n + 1)
  average_production = 55
  -> n = 9 :=
by
  sorry

end average_production_l295_295939


namespace questionnaire_visitors_l295_295940

theorem questionnaire_visitors
  (V : ℕ)
  (E U : ℕ)
  (h1 : ∀ v : ℕ, v ∈ { x : ℕ | x ≠ E ∧ x ≠ U } → v = 110)
  (h2 : E = U)
  (h3 : 3 * V = 4 * (E + U - 110))
  : V = 440 :=
by
  sorry

end questionnaire_visitors_l295_295940


namespace good_numbers_10_70_l295_295958

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def no_repeating_digits (n : ℕ) : Prop :=
  (n / 10 ≠ n % 10)

def is_good_number (n : ℕ) : Prop :=
  no_repeating_digits n ∧ (n % sum_of_digits n = 0)

theorem good_numbers_10_70 :
  is_good_number 10 ∧ is_good_number (10 + 11) ∧
  is_good_number 70 ∧ is_good_number (70 + 11) :=
by {
  -- Check that 10 is a good number
  -- Check that 21 is a good number
  -- Check that 70 is a good number
  -- Check that 81 is a good number
  sorry
}

end good_numbers_10_70_l295_295958


namespace find_diameter_C_l295_295835

noncomputable def diameter_of_circle_C (diameter_of_D : ℝ) (ratio_shaded_to_C : ℝ) : ℝ :=
  let radius_D := diameter_of_D / 2
  let radius_C := radius_D / (2 * Real.sqrt ratio_shaded_to_C)
  2 * radius_C

theorem find_diameter_C :
  let diameter_D := 20
  let ratio_shaded_area_to_C := 7
  diameter_of_circle_C diameter_D ratio_shaded_area_to_C = 5 * Real.sqrt 2 :=
by
  -- The proof is omitted.
  sorry

end find_diameter_C_l295_295835


namespace rectangle_perimeter_of_triangle_area_l295_295452

theorem rectangle_perimeter_of_triangle_area
  (h_right : ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a = 9 ∧ b = 12 ∧ c = 15)
  (rect_length : ℕ) 
  (rect_area_eq_triangle_area : ∃ (area : ℕ), area = 1/2 * 9 * 12 ∧ area = rect_length * rect_width ) 
  : ∃ (perimeter : ℕ), perimeter = 2 * (6 + rect_width) ∧ perimeter = 30 :=
sorry

end rectangle_perimeter_of_triangle_area_l295_295452


namespace smallest_four_digit_number_divisible_by_35_l295_295148

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l295_295148


namespace freshmen_more_than_sophomores_l295_295404

theorem freshmen_more_than_sophomores :
  ∀ (total_students juniors not_sophomores not_freshmen seniors adv_grade freshmen sophomores : ℕ),
    total_students = 1200 →
    juniors = 264 →
    not_sophomores = 660 →
    not_freshmen = 300 →
    seniors = 240 →
    adv_grade = 20 →
    freshmen = total_students - not_freshmen - seniors - adv_grade →
    sophomores = total_students - not_sophomores - seniors - adv_grade →
    freshmen - sophomores = 360 :=
by
  intros total_students juniors not_sophomores not_freshmen seniors adv_grade freshmen sophomores
  intros h_total h_juniors h_not_sophomores h_not_freshmen h_seniors h_adv_grade h_freshmen h_sophomores
  sorry

end freshmen_more_than_sophomores_l295_295404


namespace possible_values_of_n_l295_295082

open Nat

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b (n : ℕ) : ℕ := 2 ^ (n - 1)

noncomputable def c (n : ℕ) : ℕ := a (b n)

noncomputable def T (n : ℕ) : ℕ := (Finset.range n).sum (λ i => c (i + 1))

theorem possible_values_of_n (n : ℕ) :
  T n < 2021 → n = 8 ∨ n = 9 := by
  sorry

end possible_values_of_n_l295_295082


namespace seating_arrangements_l295_295448

/-- There are 7 seats on a long bench, and 4 people are to be seated such that 
exactly 2 of the 3 empty seats are adjacent. -/
theorem seating_arrangements : 
  let seats := 7
  let people := 4
  let empty_seats := seats - people
  in (∃ adj_empty_seats: ℕ, adj_empty_seats = 2 ∧ empty_seats - adj_empty_seats = 1) 
     ∧ fintype.card (finset.perms_of_multiset (finset.range seats).val) = 480 := 
by
  sorry

end seating_arrangements_l295_295448


namespace number_of_band_students_l295_295886

noncomputable def total_students := 320
noncomputable def sports_students := 200
noncomputable def both_activities_students := 60
noncomputable def either_activity_students := 225

theorem number_of_band_students : 
  ∃ B : ℕ, either_activity_students = B + sports_students - both_activities_students ∧ B = 85 :=
by
  sorry

end number_of_band_students_l295_295886


namespace tan_add_l295_295362

open Real

-- Define positive acute angles
def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

-- Theorem: Tangent addition formula
theorem tan_add (α β : ℝ) (hα : acute_angle α) (hβ : acute_angle β) :
  tan (α + β) = (tan α + tan β) / (1 - tan α * tan β) :=
  sorry

end tan_add_l295_295362


namespace find_original_price_l295_295945

variable (P : ℝ)

def final_price (discounted_price : ℝ) (discount_rate : ℝ) (original_price : ℝ) : Prop :=
  discounted_price = (1 - discount_rate) * original_price

theorem find_original_price (h1 : final_price 120 0.4 P) : P = 200 := 
by
  sorry

end find_original_price_l295_295945


namespace m_value_if_Q_subset_P_l295_295582

noncomputable def P : Set ℝ := {x | x^2 = 1}
def Q (m : ℝ) : Set ℝ := {x | m * x = 1}
def m_values (m : ℝ) : Prop := Q m ⊆ P → m = 0 ∨ m = 1 ∨ m = -1

theorem m_value_if_Q_subset_P (m : ℝ) : m_values m :=
sorry

end m_value_if_Q_subset_P_l295_295582


namespace roots_squared_sum_l295_295224

theorem roots_squared_sum {x y : ℝ} (hx : 3 * x^2 - 7 * x + 5 = 0) (hy : 3 * y^2 - 7 * y + 5 = 0) (hxy : x ≠ y) :
  x^2 + y^2 = 19 / 9 :=
sorry

end roots_squared_sum_l295_295224


namespace complex_pow_sub_eq_zero_l295_295260

namespace complex_proof

open Complex

def i : ℂ := Complex.I -- Defining i to be the imaginary unit

-- Stating the conditions as definitions
def condition := i^2 = -1

-- Stating the goal as a theorem
theorem complex_pow_sub_eq_zero (cond : condition) :
  (1 + 2 * i) ^ 24 - (1 - 2 * i) ^ 24 = 0 := 
by
  sorry

end complex_proof

end complex_pow_sub_eq_zero_l295_295260


namespace simon_fraction_of_alvin_l295_295527

theorem simon_fraction_of_alvin (alvin_age simon_age : ℕ) (h_alvin : alvin_age = 30)
  (h_simon : simon_age = 10) (h_fraction : ∃ f : ℚ, simon_age + 5 = f * (alvin_age + 5)) :
  ∃ f : ℚ, f = 3 / 7 := by
  sorry

end simon_fraction_of_alvin_l295_295527


namespace count_valid_a_l295_295971

variable (a : ℕ) (x : ℕ)

def valid_a (a : ℕ) : Prop :=
  0 < a ∧ a < 18 ∧ ∃ x, a * x ≡ 1 [MOD 18]

theorem count_valid_a :
  Nat.card {a : ℕ | valid_a a} = 6 :=
sorry

end count_valid_a_l295_295971


namespace book_prices_purchasing_plans_l295_295748

theorem book_prices (x y : ℕ) (h1 : 20 * x + 40 * y = 1600) (h2 : 20 * x = 30 * y + 200) : x = 40 ∧ y = 20 :=
by
  sorry

theorem purchasing_plans (m : ℕ) (h3 : 2 * m + 20 ≥ 70) (h4 : 40 * m + 20 * (m + 20) ≤ 2000) :
  (m = 25 ∧ m + 20 = 45) ∨ (m = 26 ∧ m + 20 = 46) :=
by
  -- proof steps
  sorry

end book_prices_purchasing_plans_l295_295748


namespace area_under_arccos_cos_l295_295220

noncomputable def func (x : ℝ) : ℝ := Real.arccos (Real.cos x)

theorem area_under_arccos_cos :
  ∫ x in (0:ℝ)..3 * Real.pi, func x = 3 * Real.pi ^ 2 / 2 :=
by
  sorry

end area_under_arccos_cos_l295_295220


namespace points_lie_on_line_l295_295557

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
    let x := (t + 2) / t
    let y := (t - 2) / t
    x + y = 2 :=
by
  let x := (t + 2) / t
  let y := (t - 2) / t
  sorry

end points_lie_on_line_l295_295557


namespace problem1_problem2_l295_295245

-- Definitions of vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-2, 3)
def vec_c (m : ℝ) : ℝ × ℝ := (-2, m)

-- Problem Part 1: Prove m = -1 given a ⊥ (b + c)
theorem problem1 (m : ℝ) (h : vec_a.1 * (vec_b + vec_c m).1 + vec_a.2 * (vec_b + vec_c m).2 = 0) : m = -1 :=
sorry

-- Problem Part 2: Prove k = -2 given k*a + b is collinear with 2*a - b
theorem problem2 (k : ℝ) (h : (k * vec_a.1 + vec_b.1) / (2 * vec_a.1 - vec_b.1) = (k * vec_a.2 + vec_b.2) / (2 * vec_a.2 - vec_b.2)) : k = -2 :=
sorry

end problem1_problem2_l295_295245


namespace smallest_X_divisible_by_15_l295_295896

theorem smallest_X_divisible_by_15 (T : ℕ) (h_pos : T > 0) (h_digits : ∀ (d : ℕ), d ∈ (Nat.digits 10 T) → d = 0 ∨ d = 1)
  (h_div15 : T % 15 = 0) : ∃ X : ℕ, X = T / 15 ∧ X = 74 :=
sorry

end smallest_X_divisible_by_15_l295_295896


namespace wrapping_paper_cost_l295_295254

theorem wrapping_paper_cost :
  let cost_design1 := 4 * 4 -- 20 shirt boxes / 5 shirt boxes per roll * $4.00 per roll
  let cost_design2 := 3 * 8 -- 12 XL boxes / 4 XL boxes per roll * $8.00 per roll
  let cost_design3 := 3 * 12-- 6 XXL boxes / 2 XXL boxes per roll * $12.00 per roll
  cost_design1 + cost_design2 + cost_design3 = 76
:= by
  -- Definitions
  let cost_design1 := 4 * 4
  let cost_design2 := 3 * 8
  let cost_design3 := 3 * 12
  -- Proof (To be implemented)
  sorry

end wrapping_paper_cost_l295_295254


namespace maximum_ab_expression_l295_295278

open Function Real

theorem maximum_ab_expression {a b : ℝ} (h : 0 < a ∧ 0 < b ∧ 5 * a + 6 * b < 110) :
  ab * (110 - 5 * a - 6 * b) ≤ 1331000 / 810 :=
sorry

end maximum_ab_expression_l295_295278


namespace sum_of_roots_of_abs_quadratic_is_zero_l295_295969

theorem sum_of_roots_of_abs_quadratic_is_zero : 
  ∀ x : ℝ, (|x|^2 + |x| - 6 = 0) → (x = 2 ∨ x = -2) → (2 + (-2) = 0) :=
by
  intros x h h1
  sorry

end sum_of_roots_of_abs_quadratic_is_zero_l295_295969


namespace area_of_triangle_is_3_l295_295637

noncomputable def area_of_triangle_ABC (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_is_3 : 
  ∀ (A B C : ℝ × ℝ), 
  A = (-5, -2) → 
  B = (0, 0) → 
  C = (7, -4) →
  area_of_triangle_ABC A B C = 3 :=
by
  intros A B C hA hB hC
  rw [hA, hB, hC]
  sorry

end area_of_triangle_is_3_l295_295637


namespace fraction_of_25_exact_value_l295_295930

-- Define the conditions
def eighty_percent_of_sixty : ℝ := 0.80 * 60
def smaller_by_twenty_eight (x : ℝ) : Prop := x * 25 = eighty_percent_of_sixty - 28

-- The proof problem
theorem fraction_of_25_exact_value (x : ℝ) : smaller_by_twenty_eight x → x = 4 / 5 := by
  intro h
  sorry

end fraction_of_25_exact_value_l295_295930


namespace smallest_four_digit_divisible_by_35_l295_295164

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l295_295164


namespace range_of_a_l295_295388

-- Define proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

-- Define the main theorem
theorem range_of_a (a : ℝ) : (p a ∧ ¬q a) → -1 ≤ a ∧ a < 0 :=
by
  sorry

end range_of_a_l295_295388


namespace complex_division_l295_295809

def i_units := Complex.I

def numerator := (3 : ℂ) + i_units
def denominator := (1 : ℂ) + i_units
def expected_result := (2 : ℂ) - i_units

theorem complex_division :
  numerator / denominator = expected_result :=
by sorry

end complex_division_l295_295809


namespace walls_per_room_is_8_l295_295601

-- Definitions and conditions
def total_rooms : Nat := 10
def green_rooms : Nat := 3 * total_rooms / 5
def purple_rooms : Nat := total_rooms - green_rooms
def purple_walls : Nat := 32
def walls_per_room : Nat := purple_walls / purple_rooms

-- Theorem to prove
theorem walls_per_room_is_8 : walls_per_room = 8 := by
  sorry

end walls_per_room_is_8_l295_295601


namespace combined_weight_of_jake_and_sister_l295_295725

theorem combined_weight_of_jake_and_sister (j s : ℕ) (h1 : j = 188) (h2 : j - 8 = 2 * s) : j + s = 278 :=
sorry

end combined_weight_of_jake_and_sister_l295_295725


namespace percentage_by_which_x_is_less_than_y_l295_295265

noncomputable def percentageLess (x y : ℝ) : ℝ :=
  ((y - x) / y) * 100

theorem percentage_by_which_x_is_less_than_y :
  ∀ (x y : ℝ),
  y = 125 + 0.10 * 125 →
  x = 123.75 →
  percentageLess x y = 10 :=
by
  intros x y h1 h2
  rw [h1, h2]
  unfold percentageLess
  sorry

end percentage_by_which_x_is_less_than_y_l295_295265


namespace num_valid_subset_pairs_l295_295422

open Finset

/-- I is the set {1, 2, 3, 4} -/
def I : Finset ℕ := {1, 2, 3, 4}

/-- Conditions for subsets A and B, where A and B are non-empty subsets of I and the largest element of A
    is not greater than the smallest element of B -/
def valid_subset_pair (A B : Finset ℕ) : Prop :=
  A ≠ ∅ ∧ B ≠ ∅ ∧ ∀ a ∈ A, ∀ b ∈ B, a ≤ b

/-- The number of different valid subset pairs (A, B) -/
theorem num_valid_subset_pairs : ∃ n, n = 49 ∧
  ∃ s : Finset (Finset ℕ × Finset ℕ), 
  (∀ p ∈ s, valid_subset_pair p.fst p.snd) ∧
  s.card = 49 :=
sorry

end num_valid_subset_pairs_l295_295422


namespace topsoil_cost_is_112_l295_295323

noncomputable def calculate_topsoil_cost (length width depth_in_inches : ℝ) (cost_per_cubic_foot : ℝ) : ℝ :=
  let depth_in_feet := depth_in_inches / 12
  let volume := length * width * depth_in_feet
  volume * cost_per_cubic_foot

theorem topsoil_cost_is_112 :
  calculate_topsoil_cost 8 4 6 7 = 112 :=
by
  sorry

end topsoil_cost_is_112_l295_295323


namespace probability_age_less_than_20_l295_295656

theorem probability_age_less_than_20 (total_people : ℕ) (over_30_years : ℕ) 
  (less_than_20_years : ℕ) (h1 : total_people = 120) (h2 : over_30_years = 90) 
  (h3 : less_than_20_years = total_people - over_30_years) : 
  (less_than_20_years : ℚ) / total_people = 1 / 4 :=
by {
  sorry
}

end probability_age_less_than_20_l295_295656


namespace calculate_length_l295_295467

theorem calculate_length (rent_per_acre_per_month : ℝ)
                         (total_rent_per_month : ℝ)
                         (width_of_plot : ℝ)
                         (square_feet_per_acre : ℝ) :
  rent_per_acre_per_month = 60 →
  total_rent_per_month = 600 →
  width_of_plot = 1210 →
  square_feet_per_acre = 43560 →
  (total_rent_per_month / rent_per_acre_per_month) * square_feet_per_acre / width_of_plot = 360 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    (600 / 60) * 43560 / 1210 = 10 * 43560 / 1210  : by rw div_eq_mul_inv
                            ... = 435600 / 1210    : by norm_num
                            ... = 360              : by norm_num
  -- Proof using calc block to show step-by-step result
  sorry

end calculate_length_l295_295467


namespace irrational_sqrt_10_l295_295328

theorem irrational_sqrt_10 : Irrational (Real.sqrt 10) :=
sorry

end irrational_sqrt_10_l295_295328


namespace θ_values_l295_295705

-- Define the given conditions
def terminal_side_coincides (θ : ℝ) : Prop :=
  ∃ k : ℤ, 7 * θ = θ + 360 * k

def θ_in_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

-- The main theorem
theorem θ_values (θ : ℝ) (h_terminal : terminal_side_coincides θ) (h_range : θ_in_range θ) :
  θ = 0 ∨ θ = 60 ∨ θ = 120 ∨ θ = 180 ∨ θ = 240 ∨ θ = 300 :=
sorry

end θ_values_l295_295705


namespace value_of_a_l295_295579

-- Definitions based on conditions
def A (a : ℝ) : Set ℝ := {1, 2, a}
def B : Set ℝ := {1, 7}

-- Theorem statement
theorem value_of_a (a : ℝ) (h : B ⊆ A a) : a = 7 :=
sorry

end value_of_a_l295_295579


namespace mixed_fractions_product_l295_295505

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l295_295505


namespace min_value_proof_l295_295994

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  4 / (x + 3 * y) + 1 / (x - y)

theorem min_value_proof (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 2) : 
  min_value_expr x y = 9 / 4 := 
sorry

end min_value_proof_l295_295994


namespace restore_original_problem_l295_295497

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l295_295497


namespace find_P_l295_295232

-- We start by defining the cubic polynomial
def cubic_eq (P : ℝ) (x : ℝ) := 5 * x^3 - 5 * (P + 1) * x^2 + (71 * P - 1) * x + 1

-- Define the condition that all roots are natural numbers
def has_three_natural_roots (P : ℝ) : Prop :=
  ∃ a b c : ℕ, 
    cubic_eq P a = 66 * P ∧ cubic_eq P b = 66 * P ∧ cubic_eq P c = 66 * P

-- Prove the value of P that satisfies the condition
theorem find_P : ∀ P : ℝ, has_three_natural_roots P → P = 76 := 
by
  -- We start the proof here
  sorry

end find_P_l295_295232


namespace range_of_m_l295_295872

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt (1 + x) + Real.sqrt (1 - x)) * (2 * Real.sqrt (1 - x^2) - 1)

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x = m) ↔ -Real.sqrt 2 ≤ m ∧ m ≤ Real.sqrt 2 :=
sorry

end range_of_m_l295_295872


namespace rational_solutions_zero_l295_295756

theorem rational_solutions_zero (x y z : ℚ) (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by 
  sorry

end rational_solutions_zero_l295_295756


namespace average_output_assembly_line_l295_295219

theorem average_output_assembly_line
  (initial_rate : ℕ) (initial_cogs : ℕ) 
  (increased_rate : ℕ) (increased_cogs : ℕ)
  (h1 : initial_rate = 15)
  (h2 : initial_cogs = 60)
  (h3 : increased_rate = 60)
  (h4 : increased_cogs = 60) :
  (initial_cogs + increased_cogs) / (initial_cogs / initial_rate + increased_cogs / increased_rate) = 24 := 
by sorry

end average_output_assembly_line_l295_295219


namespace even_iff_b_eq_zero_l295_295899

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

def f' (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- Given that f' is an even function, prove that b = 0.
theorem even_iff_b_eq_zero (h : ∀ x : ℝ, f' x = f' (-x)) : b = 0 :=
  sorry

end even_iff_b_eq_zero_l295_295899


namespace set_intersection_subset_condition_l295_295714

-- Define the sets A and B
def A (x : ℝ) : Prop := 1 < x - 1 ∧ x - 1 ≤ 4
def B (a : ℝ) (x : ℝ) : Prop := x < a

-- First proof problem: A ∩ B = {x | 2 < x < 3}
theorem set_intersection (a : ℝ) (x : ℝ) (h_a : a = 3) :
  A x ∧ B a x ↔ 2 < x ∧ x < 3 :=
by
  sorry

-- Second proof problem: a > 5 given A ⊆ B
theorem subset_condition (a : ℝ) :
  (∀ x, A x → B a x) ↔ a > 5 :=
by
  sorry

end set_intersection_subset_condition_l295_295714


namespace gcd_105_88_l295_295976

-- Define the numbers as constants
def a : ℕ := 105
def b : ℕ := 88

-- State the theorem: gcd(a, b) = 1
theorem gcd_105_88 : Nat.gcd a b = 1 := by
  sorry

end gcd_105_88_l295_295976


namespace sin_405_eq_sqrt2_div2_l295_295685

theorem sin_405_eq_sqrt2_div2 : Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_405_eq_sqrt2_div2_l295_295685


namespace binomial_510_510_l295_295049

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem binomial_510_510 : binomial 510 510 = 1 :=
  by
    -- Skip the proof with sorry
    sorry

end binomial_510_510_l295_295049


namespace candle_duration_1_hour_per_night_l295_295356

-- Definitions based on the conditions
def burn_rate_2_hours (candles: ℕ) (nights: ℕ) : ℕ := nights / candles -- How long each candle lasts when burned for 2 hours per night

-- Given conditions provided
def nights_24 : ℕ := 24
def candles_6 : ℕ := 6

-- The duration a candle lasts when burned for 2 hours every night
def candle_duration_2_hours_per_night : ℕ := burn_rate_2_hours candles_6 nights_24 -- => 4 (not evaluated here)

-- Theorem to prove the duration a candle lasts when burned for 1 hour every night
theorem candle_duration_1_hour_per_night : candle_duration_2_hours_per_night * 2 = 8 :=
by
  sorry -- The proof is omitted, only the statement is required

-- Note: candle_duration_2_hours_per_night = 4 by the given conditions 
-- This leads to 4 * 2 = 8, which matches the required number of nights the candle lasts when burned for 1 hour per night.

end candle_duration_1_hour_per_night_l295_295356


namespace jakes_class_boys_count_l295_295403

theorem jakes_class_boys_count 
    (ratio_girls_boys : ℕ → ℕ → Prop)
    (students_total : ℕ)
    (ratio_condition : ratio_girls_boys 3 4)
    (total_condition : students_total = 35) :
    ∃ boys : ℕ, boys = 20 :=
by
  sorry

end jakes_class_boys_count_l295_295403


namespace no_solution_for_equation_l295_295435

theorem no_solution_for_equation : 
  ∀ x : ℝ, (x ≠ 3) → (x-1)/(x-3) = 2 - 2/(3-x) → False :=
by
  intro x hx heq
  sorry

end no_solution_for_equation_l295_295435


namespace profit_percentage_l295_295827

theorem profit_percentage (SP CP : ℝ) (h_SP : SP = 150) (h_CP : CP = 120) : 
  ((SP - CP) / CP) * 100 = 25 :=
by {
  sorry
}

end profit_percentage_l295_295827


namespace smallest_four_digit_divisible_by_35_l295_295161

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l295_295161


namespace pony_jeans_discount_rate_l295_295331

noncomputable def fox_price : ℝ := 15
noncomputable def pony_price : ℝ := 18

-- Define the conditions
def total_savings (F P : ℝ) : Prop :=
  3 * (F / 100 * fox_price) + 2 * (P / 100 * pony_price) = 9

def discount_sum (F P : ℝ) : Prop :=
  F + P = 22

-- Main statement to be proven
theorem pony_jeans_discount_rate (F P : ℝ) (h1 : total_savings F P) (h2 : discount_sum F P) : P = 10 :=
by
  -- Proof goes here
  sorry

end pony_jeans_discount_rate_l295_295331


namespace ln_inequality_complex_ln_inequality_l295_295608

noncomputable def C (α : ℝ) := 1 / α

theorem ln_inequality (α : ℝ) (x : ℝ)
  (hα : 0 < α ∧ α ≤ 1) (hx : 0 ≤ x) :
  Real.log (1 + x) ≤ C(α) * x^α :=
sorry

theorem complex_ln_inequality (α : ℝ) (z1 z2 : ℂ)
  (hα : 0 < α ∧ α ≤ 1) (hz1 : z1 ≠ 0) (hz2 : z2 ≠ 0) :
  Complex.abs (Complex.log (Complex.abs (z1 / z2))) ≤
    C(α) * ((Complex.abs ((z1 - z2) / z2))^α + (Complex.abs ((z2 - z1) / z1))^α) :=
sorry

end ln_inequality_complex_ln_inequality_l295_295608


namespace mixed_fractions_product_l295_295507

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l295_295507


namespace cricket_bat_profit_percentage_l295_295015

theorem cricket_bat_profit_percentage 
  (selling_price profit : ℝ) 
  (h_sp: selling_price = 850) 
  (h_p: profit = 230) : 
  (profit / (selling_price - profit) * 100) = 37.10 :=
by
  sorry

end cricket_bat_profit_percentage_l295_295015


namespace towel_bleach_percentage_decrease_l295_295349

theorem towel_bleach_percentage_decrease :
  ∀ (L B : ℝ), (L > 0) → (B > 0) → 
  let L' := 0.70 * L 
  let B' := 0.75 * B 
  let A := L * B 
  let A' := L' * B' 
  (A - A') / A * 100 = 47.5 :=
by sorry

end towel_bleach_percentage_decrease_l295_295349


namespace problem1_problem2_problem3_problem4_l295_295240

variable (f : ℝ → ℝ)
variables (H1 : f (-1) = 2) 
          (H2 : ∀ x, x < 0 → f x > 1)
          (H3 : ∀ x y, f (x + y) = f x * f y)

-- (1) Prove f(0) = 1
theorem problem1 : f 0 = 1 := sorry

-- (2) Prove f(-4) = 16
theorem problem2 : f (-4) = 16 := sorry

-- (3) Prove f(x) is strictly decreasing
theorem problem3 : ∀ x y, x < y → f x > f y := sorry

-- (4) Solve f(-4x^2)f(10x) ≥ 1/16
theorem problem4 : { x : ℝ | f (-4 * x ^ 2) * f (10 * x) ≥ 1 / 16 } = { x | x ≤ 1 / 2 ∨ 2 ≤ x } := sorry

end problem1_problem2_problem3_problem4_l295_295240


namespace wire_cut_square_octagon_area_l295_295671

theorem wire_cut_square_octagon_area (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (equal_area : (a / 4)^2 = (2 * (b / 8)^2 * (1 + Real.sqrt 2))) : 
  a / b = Real.sqrt ((1 + Real.sqrt 2) / 2) := 
  sorry

end wire_cut_square_octagon_area_l295_295671


namespace johns_coin_collection_value_l295_295889

theorem johns_coin_collection_value :
  ∀ (n : ℕ) (value : ℕ), n = 24 → value = 20 → 
  ((n/3) * (value/8)) = 60 :=
by
  intro n value n_eq value_eq
  sorry

end johns_coin_collection_value_l295_295889


namespace find_a_l295_295861

theorem find_a 
  (x y a : ℝ) 
  (hx : x = 1) 
  (hy : y = -3) 
  (h : a * x - y = 1) : 
  a = -2 := 
  sorry

end find_a_l295_295861


namespace integer_roots_and_composite_l295_295755

theorem integer_roots_and_composite (a b : ℤ) (h1 : ∃ x1 x2 : ℤ, x1 * x2 = 1 - b ∧ x1 + x2 = -a) (h2 : b ≠ 1) : 
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ m * n = (a^2 + b^2) := 
sorry

end integer_roots_and_composite_l295_295755


namespace sandwiches_count_l295_295218

theorem sandwiches_count (M : ℕ) (C : ℕ) (S : ℕ) (hM : M = 12) (hC : C = 12) (hS : S = 5) :
  M * (C * (C - 1) / 2) * S = 3960 := 
  by sorry

end sandwiches_count_l295_295218


namespace smallest_four_digit_divisible_by_35_l295_295157

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l295_295157


namespace part_a_part_b_l295_295330

theorem part_a (A B : ℕ) (hA : 1 ≤ A) (hB : 1 ≤ B) : 
  (A + B = 70) → 
  (A * (4 : ℚ) / 35 + B * (4 : ℚ) / 35 = 8) :=
  by
    sorry

theorem part_b (C D : ℕ) (r : ℚ) (hC : C > 1) (hD : D > 1) (hr : r > 1) :
  (C + D = 8 / r) → 
  (C * r + D * r = 8) → 
  (∃ ki : ℕ, (C + D = (70 : ℕ) / ki ∧ 1 < ki ∧ ki ∣ 70)) :=
  by
    sorry

end part_a_part_b_l295_295330


namespace correlation_graph_is_scatter_plot_l295_295922

/-- The definition of a scatter plot graph -/
def scatter_plot_graph (x y : ℝ → ℝ) : Prop := 
  ∃ f : ℝ → ℝ, ∀ t : ℝ, (x t, y t) = (t, f t)

/-- Prove that the graph representing a set of data for two variables with a correlation is called a "scatter plot" -/
theorem correlation_graph_is_scatter_plot (x y : ℝ → ℝ) :
  (∃ f : ℝ → ℝ, ∀ t : ℝ, (x t, y t) = (t, f t)) → 
  (scatter_plot_graph x y) :=
by
  sorry

end correlation_graph_is_scatter_plot_l295_295922


namespace arithmetic_sequence_multiples_l295_295774

theorem arithmetic_sequence_multiples (a1 a8 : ℤ) (n : ℕ) (f : ℤ → ℤ) (d : ℤ) :
  a1 = 9 →
  a8 = 12 →
  ∀ n, f n = a1 + (n - 1) * d →
  ∃ k, ∀ m, (1 ≤ m ∧ m ≤ 2015) → f m = 3 * k ∧ k ≥ 0 ∧ k ≤ 287 →
  count_multiples_3 (first_2015_terms (f)) = 288 :=
by
  sorry

end arithmetic_sequence_multiples_l295_295774


namespace sin_405_eq_sqrt2_div_2_l295_295687

theorem sin_405_eq_sqrt2_div_2 :
  Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt2_div_2_l295_295687


namespace find_x_in_list_l295_295704

theorem find_x_in_list :
  ∃ x : ℕ, x > 0 ∧ x ≤ 120 ∧ (45 + 76 + 110 + x + x) / 5 = 2 * x ∧ x = 29 :=
by
  sorry

end find_x_in_list_l295_295704


namespace constant_term_expansion_eq_sixty_l295_295091

theorem constant_term_expansion_eq_sixty (a : ℝ) (h : 15 * a = 60) : a = 4 :=
by
  sorry

end constant_term_expansion_eq_sixty_l295_295091


namespace smallest_four_digit_divisible_by_35_l295_295155

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l295_295155


namespace greatest_integer_not_exceeding_a_l295_295708

theorem greatest_integer_not_exceeding_a (a : ℝ) (h : 3^a + a^3 = 123) : ⌊a⌋ = 4 :=
sorry

end greatest_integer_not_exceeding_a_l295_295708


namespace polynomial_equal_roots_l295_295185

open Complex

theorem polynomial_equal_roots (n : ℕ) (a : ℕ → ℂ) 
  (h : ∀ (x : ℂ), (∑ k in finset.range (n+1), (-1)^k * (binom n k) * (a k)^k * x^(n-k)) = 0) :
  ∀ i j, a i = a j :=
by 
  -- Proof steps would go here
  sorry

end polynomial_equal_roots_l295_295185


namespace interval_of_decrease_l295_295689

noncomputable def f : ℝ → ℝ := fun x => x^2 - 2 * x

theorem interval_of_decrease : 
  ∃ a b : ℝ, a = -2 ∧ b = 1 ∧ ∀ x1 x2 : ℝ, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 ≥ f x2 :=
by 
  use -2, 1
  sorry

end interval_of_decrease_l295_295689


namespace minimum_value_l295_295895

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 6) :
  37.5 ≤ (9 / x + 25 / y + 49 / z) :=
sorry

end minimum_value_l295_295895


namespace sin_cos_identity_l295_295074

theorem sin_cos_identity (θ : ℝ) (h : Real.tan (θ + (Real.pi / 4)) = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = -7/5 := 
by 
  sorry

end sin_cos_identity_l295_295074


namespace new_ratio_first_term_less_than_implied_l295_295119

-- Define the original and new ratios
def original_ratio := (6, 7)
def subtracted_value := 3
def new_ratio := (original_ratio.1 - subtracted_value, original_ratio.2 - subtracted_value)

-- Prove the required property
theorem new_ratio_first_term_less_than_implied {r1 r2 : ℕ} (h : new_ratio = (3, 4))
  (h_less : r1 > 3) :
  new_ratio.1 < r1 := 
sorry

end new_ratio_first_term_less_than_implied_l295_295119


namespace mn_sum_value_l295_295315

-- Definition of the problem conditions
def faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_consecutive (a b : ℕ) : Prop :=
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨
  (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3) ∨
  (a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) ∨
  (a = 5 ∧ b = 6) ∨ (a = 6 ∧ b = 5) ∨
  (a = 6 ∧ b = 7) ∨ (a = 7 ∧ b = 6) ∨
  (a = 7 ∧ b = 8) ∨ (a = 8 ∧ b = 7) ∨
  (a = 8 ∧ b = 9) ∨ (a = 9 ∧ b = 8) ∨
  (a = 9 ∧ b = 1) ∨ (a = 1 ∧ b = 9)

noncomputable def m_n_sum : ℕ :=
  let total_permutations := 5040
  let valid_permutations := 60
  let probability := valid_permutations / total_permutations
  let m := 1
  let n := total_permutations / valid_permutations
  m + n

theorem mn_sum_value : m_n_sum = 85 :=
  sorry

end mn_sum_value_l295_295315


namespace find_totally_damaged_cartons_l295_295900

def jarsPerCarton : ℕ := 20
def initialCartons : ℕ := 50
def reducedCartons : ℕ := 30
def damagedJarsPerCarton : ℕ := 3
def damagedCartons : ℕ := 5
def totalGoodJars : ℕ := 565

theorem find_totally_damaged_cartons :
  (initialCartons * jarsPerCarton - ((initialCartons - reducedCartons) * jarsPerCarton + damagedJarsPerCarton * damagedCartons - totalGoodJars)) / jarsPerCarton = 1 := by
  sorry

end find_totally_damaged_cartons_l295_295900


namespace probability_at_least_one_boy_and_one_girl_in_four_children_l295_295531

theorem probability_at_least_one_boy_and_one_girl_in_four_children :
  ∀ (n : ℕ), n = 4 → 
  (∀ (p : ℚ), p = 1 / 2 →
  ((1 : ℚ) - ((p ^ n) + (p ^ n)) = 7 / 8)) :=
by
  intro n hn p hp
  rw [hn, hp]
  norm_num
  sorry

end probability_at_least_one_boy_and_one_girl_in_four_children_l295_295531


namespace find_certain_number_l295_295178

theorem find_certain_number (x : ℕ) (h : (55 * x) % 8 = 7) : x = 1 := 
sorry

end find_certain_number_l295_295178


namespace pencil_distribution_l295_295237

theorem pencil_distribution (n : ℕ) (friends : ℕ): 
  (friends = 4) → (n = 8) → 
  (∃ A B C D : ℕ, A ≥ 2 ∧ B ≥ 1 ∧ C ≥ 1 ∧ D ≥ 1 ∧ A + B + C + D = n) →
  (∃! k : ℕ, k = 20) :=
by
  intros friends_eq n_eq h
  use 20
  sorry

end pencil_distribution_l295_295237


namespace monthly_interest_rate_l295_295738

-- Define the principal amount (initial amount).
def principal : ℝ := 200

-- Define the final amount after 2 months (A).
def amount_after_two_months : ℝ := 222

-- Define the number of months (n).
def months : ℕ := 2

-- Define the monthly interest rate (r) we need to prove.
def interest_rate : ℝ := 0.053

-- Main statement to prove
theorem monthly_interest_rate :
  amount_after_two_months = principal * (1 + interest_rate)^months :=
sorry

end monthly_interest_rate_l295_295738


namespace ingrid_income_l295_295410

theorem ingrid_income (I : ℝ) (h1 : 0.30 * 56000 = 16800) 
  (h2 : ∀ (I : ℝ), 0.40 * I = 0.4 * I) 
  (h3 : 0.35625 * (56000 + I) = 16800 + 0.4 * I) : 
  I = 49142.86 := 
by 
  sorry

end ingrid_income_l295_295410


namespace rachel_bella_total_distance_l295_295216

theorem rachel_bella_total_distance:
  ∀ (distance_land distance_sea total_distance: ℕ), 
  distance_land = 451 → 
  distance_sea = 150 → 
  total_distance = distance_land + distance_sea → 
  total_distance = 601 := 
by 
  intros distance_land distance_sea total_distance h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end rachel_bella_total_distance_l295_295216


namespace range_of_x_when_a_is_1_range_of_a_for_necessity_l295_295868

-- Define the statements p and q based on the conditions
def p (x a : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := (x - 3) / (x - 2) ≤ 0

-- (1) Prove the range of x when a = 1 and p ∧ q is true
theorem range_of_x_when_a_is_1 {x : ℝ} (h1 : ∀ x, p x 1) (h2 : q x) : 2 < x ∧ x < 3 :=
  sorry

-- (2) Prove the range of a for p to be necessary but not sufficient for q
theorem range_of_a_for_necessity : ∀ a, (∀ x, p x a → q x) → (1 ≤ a ∧ a ≤ 2) :=
  sorry

end range_of_x_when_a_is_1_range_of_a_for_necessity_l295_295868


namespace number_of_B_eq_l295_295634

variable (a b : ℝ)
variable (B : ℝ)

theorem number_of_B_eq : 3 * B = a + b → B = (a + b) / 3 :=
by sorry

end number_of_B_eq_l295_295634


namespace volume_of_rectangular_prism_l295_295993

theorem volume_of_rectangular_prism (a b c : ℝ)
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : a * c = Real.sqrt 6) :
  a * b * c = Real.sqrt 6 := by
sorry

end volume_of_rectangular_prism_l295_295993


namespace boris_number_of_bowls_l295_295044

-- Definitions from the conditions
def total_candies : ℕ := 100
def daughter_eats : ℕ := 8
def candies_per_bowl_after_removal : ℕ := 20
def candies_removed_per_bowl : ℕ := 3

-- Derived definitions
def remaining_candies : ℕ := total_candies - daughter_eats
def candies_per_bowl_orig : ℕ := candies_per_bowl_after_removal + candies_removed_per_bowl

-- Statement to prove
theorem boris_number_of_bowls : remaining_candies / candies_per_bowl_orig = 4 :=
by sorry

end boris_number_of_bowls_l295_295044


namespace find_takeoff_run_distance_l295_295123

-- Define the conditions
def time_of_takeoff : ℝ := 15 -- seconds
def takeoff_speed_kmh : ℝ := 100 -- km/h

-- Define the conversions and proof problem
noncomputable def takeoff_speed_ms : ℝ := takeoff_speed_kmh * 1000 / 3600 -- conversion from km/h to m/s
noncomputable def acceleration : ℝ := takeoff_speed_ms / time_of_takeoff -- a = v / t

theorem find_takeoff_run_distance : 
  (1/2) * acceleration * (time_of_takeoff ^ 2) = 208 := by
  sorry

end find_takeoff_run_distance_l295_295123


namespace expression_evaluation_l295_295063

theorem expression_evaluation (a : ℕ) (h : a = 1580) : 
  2 * a - ((2 * a - 3) / (a + 1) - (a + 1) / (2 - 2 * a) - (a^2 + 3) / 2) * ((a^3 + 1) / (a^2 - a)) + 2 / a = 2 := 
sorry

end expression_evaluation_l295_295063


namespace probability_of_majors_around_table_l295_295324

-- Defining the set of people
structure People where
  math_major : Nat
  physics_major : Nat
  biology_major : Nat
  total_people : Nat

def conditions : People :=
  { math_major := 5, physics_major := 4, biology_major := 3, total_people := 12 }

def round_table_probability (p : People) : ℚ :=
  if p.total_people = 12 ∧ p.math_major = 5 ∧ p.physics_major = 4 ∧ p.biology_major = 3 then
    18/175
  else
    0

theorem probability_of_majors_around_table :
  round_table_probability conditions = 18 / 175 :=
by
  sorry

end probability_of_majors_around_table_l295_295324


namespace cube_surface_area_difference_l295_295026

theorem cube_surface_area_difference :
  let large_cube_volume := 8
  let small_cube_volume := 1
  let num_small_cubes := 8
  let large_cube_side := (large_cube_volume : ℝ) ^ (1 / 3)
  let small_cube_side := (small_cube_volume : ℝ) ^ (1 / 3)
  let large_cube_surface_area := 6 * (large_cube_side ^ 2)
  let small_cube_surface_area := 6 * (small_cube_side ^ 2)
  let total_small_cubes_surface_area := num_small_cubes * small_cube_surface_area
  total_small_cubes_surface_area - large_cube_surface_area = 24 :=
by
  sorry

end cube_surface_area_difference_l295_295026


namespace tom_helicopter_hours_l295_295640

theorem tom_helicopter_hours (total_cost : ℤ) (cost_per_hour : ℤ) (days : ℤ) (h : total_cost = 450) (c : cost_per_hour = 75) (d : days = 3) :
  total_cost / cost_per_hour / days = 2 := by
  -- Proof goes here
  sorry

end tom_helicopter_hours_l295_295640


namespace apples_remain_correct_l295_295329

def total_apples : ℕ := 15
def apples_eaten : ℕ := 7
def apples_remaining : ℕ := total_apples - apples_eaten

theorem apples_remain_correct : apples_remaining = 8 :=
by
  -- Initial number of apples
  let total := total_apples
  -- Number of apples eaten
  let eaten := apples_eaten
  -- Remaining apples
  let remain := total - eaten
  -- Assertion
  have h : remain = 8 := by
      sorry
  exact h

end apples_remain_correct_l295_295329


namespace mixed_fraction_product_example_l295_295474

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l295_295474


namespace range_of_m_l295_295397

-- Define the conditions in Lean 4

def double_point (x y : ℝ) : Prop := y = 2 * x

def quadratic_function (x m : ℝ) : ℝ := x^2 + 2 * m * x - m

noncomputable def M := (x1 : ℝ) (hM : double_point x1 (quadratic_function x1 m)) 
def N := (x2 : ℝ) (hN : double_point x2 (quadratic_function x2 m))
def x1_lt_1_lt_x2 (x1 x2 : ℝ) : Prop := x1 < 1 ∧ 1 < x2

-- Lean 4 theorem statement

theorem range_of_m (x1 x2 m : ℝ) 
  (h_double_point_M : double_point x1 (quadratic_function x1 m))
  (h_double_point_N : double_point x2 (quadratic_function x2 m))
  (h_x1_lt_1_lt_x2 : x1_lt_1_lt_x2 x1 x2) 
: m < 1 := 
sorry

end range_of_m_l295_295397


namespace unique_functional_equation_l295_295702

theorem unique_functional_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x + y :=
sorry

end unique_functional_equation_l295_295702


namespace max_value_x2_plus_y2_l295_295742

theorem max_value_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 + 4 * y^2 = 10 * x) : 
  x^2 + y^2 ≤ 4 :=
sorry

end max_value_x2_plus_y2_l295_295742


namespace probability_MAME_top_l295_295325

-- Conditions
def paper_parts : ℕ := 8
def desired_top : ℕ := 1

-- Question and Proof Problem (Probability calculation)
theorem probability_MAME_top : (1 : ℚ) / paper_parts = 1 / 8 :=
by
  sorry

end probability_MAME_top_l295_295325


namespace max_triangle_side_length_l295_295208

theorem max_triangle_side_length:
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a + b + c = 30 ∧
    a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 14 :=
  sorry

end max_triangle_side_length_l295_295208


namespace range_of_k_l295_295093

noncomputable def meets_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 - y^2 = 4 ∧ y = k * x - 1

theorem range_of_k : 
  { k : ℝ | meets_hyperbola k } = { k : ℝ | k = 1 ∨ k = -1 ∨ - (Real.sqrt 5) / 2 ≤ k ∧ k ≤ (Real.sqrt 5) / 2 } :=
by
  sorry

end range_of_k_l295_295093


namespace sum_reciprocals_lt_seven_sixths_l295_295711

open scoped BigOperators

variable (a : Fin 1009 → ℕ)
variable (a_sorted : StrictMono a)
variable (a_bounds : ∀ i, 1 < a i ∧ a i < 2018)
variable (lcm_condition : ∀ i j, i ≠ j → Nat.lcm (a i) (a j) > 2018)

theorem sum_reciprocals_lt_seven_sixths :
  ∑ i in Finset.univ, (1 : ℚ) / (a i) < 7 / 6 :=
sorry

end sum_reciprocals_lt_seven_sixths_l295_295711


namespace missing_pieces_l295_295136

-- Definitions based on the conditions.
def total_pieces : ℕ := 500
def border_pieces : ℕ := 75
def trevor_pieces : ℕ := 105
def joe_pieces : ℕ := 3 * trevor_pieces

-- Prove the number of missing pieces is 5.
theorem missing_pieces : total_pieces - (border_pieces + trevor_pieces + joe_pieces) = 5 := by
  sorry

end missing_pieces_l295_295136


namespace div_count_27n5_l295_295376

theorem div_count_27n5 
  (n : ℕ) 
  (h : (120 * n^3).divisors.card = 120) 
  : (27 * n^5).divisors.card = 324 :=
sorry

end div_count_27n5_l295_295376


namespace sum_of_intersection_coordinates_l295_295043

noncomputable def h : ℝ → ℝ := sorry

theorem sum_of_intersection_coordinates : 
  (∃ a b : ℝ, h a = h (a + 2) ∧ h 1 = 3 ∧ h (-1) = 3 ∧ a = -1 ∧ b = 3) → -1 + 3 = 2 :=
by
  intro h_assumptions
  sorry

end sum_of_intersection_coordinates_l295_295043


namespace problem_l295_295985

noncomputable def a : Real := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : Real := Real.sin (16 * Real.pi / 180) + Real.cos (16 * Real.pi / 180)
noncomputable def c : Real := Real.sqrt 6 / 2

theorem problem :
  a < c ∧ c < b := by
  sorry

end problem_l295_295985


namespace square_triangle_same_area_l295_295598

theorem square_triangle_same_area (perimeter_square height_triangle : ℤ) (same_area : ℚ) 
  (h_perimeter_square : perimeter_square = 64) 
  (h_height_triangle : height_triangle = 64)
  (h_same_area : same_area = 256) :
  ∃ x : ℚ, x = 8 :=
by
  sorry

end square_triangle_same_area_l295_295598


namespace fish_total_count_l295_295132

theorem fish_total_count :
  let num_fishermen : ℕ := 20
  let fish_caught_per_fisherman : ℕ := 400
  let fish_caught_by_twentieth_fisherman : ℕ := 2400
  (19 * fish_caught_per_fisherman + fish_caught_by_twentieth_fisherman) = 10000 :=
by
  sorry

end fish_total_count_l295_295132


namespace cost_of_bench_eq_150_l295_295025

theorem cost_of_bench_eq_150 (B : ℕ) (h : B + 2 * B = 450) : B = 150 :=
sorry

end cost_of_bench_eq_150_l295_295025


namespace smallest_four_digit_divisible_by_35_l295_295142

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 35 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
begin
  use 1050,
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    have h35m: m % 35 = 0 := hm.right.right,
    have hm0: m ≥ 1000 := hm.left,
    have hm1: m < 10000 := hm.right.left,
    sorry, -- this is where the detailed proof steps would go
  }
end

end smallest_four_digit_divisible_by_35_l295_295142


namespace remove_terms_l295_295056

-- Define the fractions
def f1 := 1 / 3
def f2 := 1 / 6
def f3 := 1 / 9
def f4 := 1 / 12
def f5 := 1 / 15
def f6 := 1 / 18

-- Define the total sum
def total_sum := f1 + f2 + f3 + f4 + f5 + f6

-- Define the target sum after removal
def target_sum := 2 / 3

-- Define the condition to be proven
theorem remove_terms {x y : Real} (h1 : (x = f4) ∧ (y = f5)) : 
  total_sum - (x + y) = target_sum := by
  sorry

end remove_terms_l295_295056


namespace quadratic_minimum_value_proof_l295_295857

-- Define the quadratic function and its properties
def quadratic_function (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

-- Define the condition that the coefficient of the squared term is positive
def coefficient_positive : Prop := (2 : ℝ) > 0

-- Define the axis of symmetry
def axis_of_symmetry (h : ℝ) : Prop := h = 3

-- Define the minimum value of the quadratic function
def minimum_value (y_min : ℝ) : Prop := ∀ x : ℝ, y_min ≤ quadratic_function x 

-- Define the correct answer choice
def correct_answer : Prop := minimum_value 2

-- The theorem stating the proof problem
theorem quadratic_minimum_value_proof :
  coefficient_positive ∧ axis_of_symmetry 3 → correct_answer :=
sorry

end quadratic_minimum_value_proof_l295_295857


namespace speed_of_policeman_l295_295198

theorem speed_of_policeman 
  (d_initial : ℝ) 
  (v_thief : ℝ) 
  (d_thief : ℝ)
  (d_policeman : ℝ)
  (h_initial : d_initial = 100) 
  (h_v_thief : v_thief = 8) 
  (h_d_thief : d_thief = 400) 
  (h_d_policeman : d_policeman = 500) 
  : ∃ (v_p : ℝ), v_p = 10 :=
by
  -- Use the provided conditions
  sorry

end speed_of_policeman_l295_295198


namespace joshua_finishes_after_malcolm_l295_295902

-- Definitions based on conditions.
def malcolm_speed : ℕ := 6 -- Malcolm's speed in minutes per mile
def joshua_speed : ℕ := 8 -- Joshua's speed in minutes per mile
def race_distance : ℕ := 10 -- Race distance in miles

-- Theorem: How many minutes after Malcolm crosses the finish line will Joshua cross the finish line?
theorem joshua_finishes_after_malcolm :
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 20 :=
by
  -- sorry is a placeholder for the proof
  sorry

end joshua_finishes_after_malcolm_l295_295902


namespace smallest_four_digit_divisible_by_35_l295_295165

theorem smallest_four_digit_divisible_by_35 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m := 
begin
  use 1015,
  split,
  { exact le_of_eq (by simp) },
  split,
  { exact le_trans (by simp) (by norm_num) },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    exact le_of_lt (by norm_num), 
    use sorry },
end

end smallest_four_digit_divisible_by_35_l295_295165


namespace initial_avg_height_l295_295626

-- Lean 4 statement for the given problem
theorem initial_avg_height (A : ℝ) (n : ℕ) (wrong_height correct_height actual_avg init_diff : ℝ)
  (h_class_size : n = 35)
  (h_wrong_height : wrong_height = 166)
  (h_correct_height : correct_height = 106)
  (h_actual_avg : actual_avg = 183)
  (h_init_diff : init_diff = wrong_height - correct_height)
  (h_total_height_actual : n * actual_avg = 35 * 183)
  (h_total_height_wrong : n * A = 35 * actual_avg - init_diff) :
  A = 181 :=
by {
  -- The problem and conditions are correctly stated. The proof is skipped with sorry.
  sorry
}

end initial_avg_height_l295_295626


namespace angle_AEC_invariant_and_30_degrees_l295_295297

theorem angle_AEC_invariant_and_30_degrees 
  (A B C D E : Type) [linear_ordered_field A] [metric_space A]
  [ordered_smetric_space A] (triangle_ABC : triangle A B C)
  (equilateral : triangle.equilateral triangle_ABC)
  (D_on_BC : ∃ d, d ∈ segment B C) 
  (E_on_AD : ∃ e, e ∈ line_through A D ∧ dist B E = dist B A) : 
  ∃ E : A, ∠ A E C = 30 :=
by
  sorry

end angle_AEC_invariant_and_30_degrees_l295_295297


namespace right_angled_triangle_k_values_l295_295098

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

def AB : ℝ × ℝ := (2, 1)
def AC (k : ℝ) : ℝ × ℝ := (3, k)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def BC (k : ℝ) : ℝ × ℝ := (1, k - 1)

theorem right_angled_triangle_k_values (k : ℝ) :
  (dot_product AB (AC k) = 0 ∨ dot_product AB (BC k) = 0 ∨ dot_product (BC k) (AC k) = 0) ↔ (k = -6 ∨ k = -1) :=
sorry

end right_angled_triangle_k_values_l295_295098


namespace sufficient_but_not_necessary_l295_295552

theorem sufficient_but_not_necessary (x: ℝ) (hx: 0 < x ∧ x < 1) : 0 < x^2 ∧ x^2 < 1 ∧ (∀ y, 0 < y^2 ∧ y^2 < 1 → (y > 0 ∧ y < 1 ∨ y < 0 ∧ y > -1)) :=
by {
  sorry
}

end sufficient_but_not_necessary_l295_295552


namespace smallest_possible_number_of_apples_l295_295179

theorem smallest_possible_number_of_apples :
  ∃ (M : ℕ), M > 2 ∧ M % 9 = 2 ∧ M % 10 = 2 ∧ M % 11 = 2 ∧ M = 200 :=
by
  sorry

end smallest_possible_number_of_apples_l295_295179


namespace restore_fractions_l295_295515

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l295_295515


namespace sugar_percentage_in_new_solution_l295_295664

open Real

noncomputable def original_volume : ℝ := 450
noncomputable def original_sugar_percentage : ℝ := 20 / 100
noncomputable def added_sugar : ℝ := 7.5
noncomputable def added_water : ℝ := 20
noncomputable def added_kola : ℝ := 8.1
noncomputable def added_flavoring : ℝ := 2.3

noncomputable def original_sugar_amount : ℝ := original_volume * original_sugar_percentage
noncomputable def total_sugar_amount : ℝ := original_sugar_amount + added_sugar
noncomputable def new_total_volume : ℝ := original_volume + added_water + added_kola + added_flavoring + added_sugar
noncomputable def new_sugar_percentage : ℝ := (total_sugar_amount / new_total_volume) * 100

theorem sugar_percentage_in_new_solution : abs (new_sugar_percentage - 19.97) < 0.01 := sorry

end sugar_percentage_in_new_solution_l295_295664


namespace batsman_average_proof_l295_295466

noncomputable def batsman_average_after_17th_inning (A : ℝ) : ℝ :=
  (A * 16 + 87) / 17

theorem batsman_average_proof (A : ℝ) (h1 : 16 * A + 87 = 17 * (A + 2)) : batsman_average_after_17th_inning 53 = 55 :=
by
  sorry

end batsman_average_proof_l295_295466


namespace empty_cistern_time_l295_295023

variable (t_fill : ℝ) (t_empty₁ : ℝ) (t_empty₂ : ℝ) (t_empty₃ : ℝ)

theorem empty_cistern_time
  (h_fill : t_fill = 3.5)
  (h_empty₁ : t_empty₁ = 14)
  (h_empty₂ : t_empty₂ = 16)
  (h_empty₃ : t_empty₃ = 18) :
  1008 / (1/t_empty₁ + 1/t_empty₂ + 1/t_empty₃) = 1.31979 := by
  sorry

end empty_cistern_time_l295_295023


namespace intersection_A_B_l295_295744

def A := {x : ℝ | 2 ≤ x ∧ x ≤ 8}
def B := {x : ℝ | x^2 - 3 * x - 4 < 0}
def expected := {x : ℝ | 2 ≤ x ∧ x < 4 }

theorem intersection_A_B : (A ∩ B) = expected := 
by 
  sorry

end intersection_A_B_l295_295744


namespace find_b_of_quadratic_eq_l295_295267

theorem find_b_of_quadratic_eq (a b c y1 y2 : ℝ) 
    (h1 : y1 = a * (2:ℝ)^2 + b * (2:ℝ) + c) 
    (h2 : y2 = a * (-2:ℝ)^2 + b * (-2:ℝ) + c) 
    (h_diff : y1 - y2 = 4) : b = 1 :=
by
  sorry

end find_b_of_quadratic_eq_l295_295267


namespace sum_of_consecutive_even_numbers_l295_295017

theorem sum_of_consecutive_even_numbers (x : ℤ) (h : (x + 2)^2 - x^2 = 84) : x + (x + 2) = 42 :=
by 
  sorry

end sum_of_consecutive_even_numbers_l295_295017


namespace nell_initial_cards_l295_295906

theorem nell_initial_cards (cards_given cards_left total_cards : ℕ)
  (h1 : cards_given = 301)
  (h2 : cards_left = 154)
  (h3 : total_cards = cards_given + cards_left) :
  total_cards = 455 := by
  rw [h1, h2] at h3
  exact h3

end nell_initial_cards_l295_295906


namespace max_side_of_triangle_with_perimeter_30_l295_295205

theorem max_side_of_triangle_with_perimeter_30 
  (a b c : ℕ) 
  (h1 : a + b + c = 30) 
  (h2 : a ≥ b) 
  (h3 : b ≥ c) 
  (h4 : a < b + c) 
  (h5 : b < a + c) 
  (h6 : c < a + b) 
  : a ≤ 14 :=
sorry

end max_side_of_triangle_with_perimeter_30_l295_295205


namespace total_water_intake_l295_295426

def morning_water : ℝ := 1.5
def afternoon_water : ℝ := 3 * morning_water
def evening_water : ℝ := 0.5 * afternoon_water

theorem total_water_intake : 
  (morning_water + afternoon_water + evening_water) = 8.25 :=
by
  sorry

end total_water_intake_l295_295426


namespace median_of_36_consecutive_integers_l295_295788

theorem median_of_36_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) 
  (h1 : num_of_integers = 36) (h2 : sum_of_integers = 6 ^ 4) : 
  (sum_of_integers / num_of_integers) = 36 := 
by 
  sorry

end median_of_36_consecutive_integers_l295_295788


namespace parabola_focus_distance_area_l295_295776

theorem parabola_focus_distance_area (p : ℝ) (hp : p > 0)
  (A : ℝ × ℝ) (hA : A.2^2 = 2 * p * A.1)
  (hDist : A.1 + p / 2 = 2 * A.1)
  (hArea : 1/2 * (p / 2) * |A.2| = 1) :
  p = 2 :=
sorry

end parabola_focus_distance_area_l295_295776


namespace weight_7_moles_AlI3_l295_295931

-- Definitions from the conditions
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_I : ℝ := 126.90
def molecular_weight_AlI3 : ℝ := atomic_weight_Al + 3 * atomic_weight_I
def weight_of_compound (moles : ℝ) (molecular_weight : ℝ) : ℝ := moles * molecular_weight

-- Theorem stating the weight of 7 moles of AlI3
theorem weight_7_moles_AlI3 : 
  weight_of_compound 7 molecular_weight_AlI3 = 2853.76 :=
by
  -- Proof will be added here
  sorry

end weight_7_moles_AlI3_l295_295931


namespace restore_original_problem_l295_295498

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l295_295498


namespace book_cost_price_l295_295946

theorem book_cost_price 
  (M : ℝ) (hM : M = 64.54) 
  (h1 : ∃ L : ℝ, 0.92 * L = M ∧ L = 1.25 * 56.12) :
  ∃ C : ℝ, C = 56.12 :=
by
  sorry

end book_cost_price_l295_295946


namespace p_is_sufficient_but_not_necessary_l295_295990

-- Definitions based on conditions
def p (x y : Int) : Prop := x + y ≠ -2
def q (x y : Int) : Prop := ¬(x = -1 ∧ y = -1)

theorem p_is_sufficient_but_not_necessary (x y : Int) : 
  (p x y → q x y) ∧ ¬(q x y → p x y) :=
by
  sorry

end p_is_sufficient_but_not_necessary_l295_295990


namespace phase_shift_of_sine_l295_295227

theorem phase_shift_of_sine (b c : ℝ) (h_b : b = 4) (h_c : c = - (Real.pi / 2)) :
  (-c / b) = Real.pi / 8 :=
by
  rw [h_b, h_c]
  sorry

end phase_shift_of_sine_l295_295227


namespace percentage_less_than_l295_295657

theorem percentage_less_than (p j t : ℝ) (h1 : j = 0.75 * p) (h2 : j = 0.80 * t) : 
  t = (1 - 0.0625) * p := 
by 
  sorry

end percentage_less_than_l295_295657


namespace product_of_fractions_l295_295455

theorem product_of_fractions (a b c d e f : ℚ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 3) 
  (h_d : d = 2) (h_e : e = 3) (h_f : f = 4) :
  (a / b) * (d / e) * (c / f) = 1 / 4 :=
by
  sorry

end product_of_fractions_l295_295455


namespace probability_square_or_triangle_l295_295968

theorem probability_square_or_triangle :
  let total_figures := 10
  let number_of_triangles := 4
  let number_of_squares := 3
  let number_of_favorable_outcomes := number_of_triangles + number_of_squares
  let probability := number_of_favorable_outcomes / total_figures
  probability = 7 / 10 :=
sorry

end probability_square_or_triangle_l295_295968


namespace mean_equality_l295_295924

theorem mean_equality (z : ℝ) :
  (8 + 15 + 24) / 3 = (16 + z) / 2 → z = 15.34 :=
by
  intro h
  sorry

end mean_equality_l295_295924


namespace median_of_consecutive_integers_sum_eq_6_pow_4_l295_295779

theorem median_of_consecutive_integers_sum_eq_6_pow_4 :
  ∀ (s : ℕ) (n : ℕ), s = 36 → ∑ i in finset.range 36, (n + i) = 6^4 → 36 / 2 = 36 :=
by
  sorry

end median_of_consecutive_integers_sum_eq_6_pow_4_l295_295779


namespace mixed_fraction_product_l295_295516

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l295_295516


namespace solve_fractions_l295_295483

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l295_295483


namespace expectation_of_transformed_binomial_l295_295876

def binomial_expectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

def linear_property_of_expectation (a b : ℚ) (E_ξ : ℚ) : ℚ :=
  a * E_ξ + b

theorem expectation_of_transformed_binomial (ξ : ℚ) :
  ξ = binomial_expectation 5 (2/5) →
  linear_property_of_expectation 5 2 ξ = 12 :=
by
  intros h
  rw [h]
  unfold linear_property_of_expectation binomial_expectation
  sorry

end expectation_of_transformed_binomial_l295_295876


namespace no_unique_solution_for_c_l295_295558

theorem no_unique_solution_for_c (k : ℕ) (hk : k = 9) (c : ℕ) :
  (∀ x y : ℕ, 9 * x + c * y = 30 → 3 * x + 4 * y = 12) → c = 12 :=
by
  sorry

end no_unique_solution_for_c_l295_295558


namespace remainder_of_expression_l295_295539

theorem remainder_of_expression :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by {
  -- Prove the expression step by step
  -- sorry
  sorry
}

end remainder_of_expression_l295_295539


namespace f_periodic_if_is_bounded_and_satisfies_fe_l295_295892

variable {f : ℝ → ℝ}

-- Condition 1: f is a bounded real function, i.e., it is bounded above and below
def is_bounded (f : ℝ → ℝ) : Prop := ∃ M, ∀ x, |f x| ≤ M

-- Condition 2: The functional equation given for all x.
def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1/3) + f (x + 1/2) = f x + f (x + 5/6)

-- We need to show that f is periodic with period 1.
theorem f_periodic_if_is_bounded_and_satisfies_fe (h_bounded : is_bounded f) (h_fe : functional_eq f) : 
  ∀ x, f (x + 1) = f x :=
sorry

end f_periodic_if_is_bounded_and_satisfies_fe_l295_295892


namespace inequality_proof_l295_295075

noncomputable def a : Real := (1 / 3) ^ Real.pi
noncomputable def b : Real := (1 / 3) ^ (1 / 2 : Real)
noncomputable def c : Real := Real.pi ^ (1 / 2 : Real)

theorem inequality_proof : a < b ∧ b < c :=
by
  -- Proof will be provided here
  sorry

end inequality_proof_l295_295075


namespace shorter_side_length_l295_295344

theorem shorter_side_length (a b : ℕ) (h1 : 2 * a + 2 * b = 42) (h2 : a * b = 108) : b = 9 :=
by
  sorry

end shorter_side_length_l295_295344


namespace fill_blank_1_fill_blank_2_l295_295973

theorem fill_blank_1 (x : ℤ) (h : 1 + x = -10) : x = -11 := sorry

theorem fill_blank_2 (y : ℝ) (h : y - 4.5 = -4.5) : y = 0 := sorry

end fill_blank_1_fill_blank_2_l295_295973


namespace blue_paint_cans_needed_l295_295617

-- Definitions of the conditions
def blue_to_green_ratio : ℕ × ℕ := (4, 3)
def total_cans : ℕ := 42
def expected_blue_cans : ℕ := 24

-- Proof statement
theorem blue_paint_cans_needed (r : ℕ × ℕ) (total : ℕ) (expected : ℕ) 
  (h1: r = (4, 3)) (h2: total = 42) : expected = 24 :=
by
  sorry

end blue_paint_cans_needed_l295_295617


namespace evaluate_expression_l295_295542

theorem evaluate_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ( (1 / a^2 + 1 / b^2)⁻¹ = a^2 * b^2 / (a^2 + b^2) ) :=
by
  sorry

end evaluate_expression_l295_295542


namespace min_value_f_l295_295069

theorem min_value_f
  (a b c : ℝ)
  (α β γ : ℤ)
  (hα : α = 1 ∨ α = -1)
  (hβ : β = 1 ∨ β = -1)
  (hγ : γ = 1 ∨ γ = -1)
  (h : a * α + b * β + c * γ = 0) :
  (∃ f_min : ℝ, f_min = ( ((a ^ 3 + b ^ 3 + c ^ 3) / (a * b * c)) ^ 2) ∧ f_min = 9) :=
sorry

end min_value_f_l295_295069


namespace exists_zero_in_interval_minus3_minus2_l295_295106

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x - x

theorem exists_zero_in_interval_minus3_minus2 : 
  ∃ x ∈ Set.Icc (-3 : ℝ) (-2), f x = 0 :=
by
  sorry

end exists_zero_in_interval_minus3_minus2_l295_295106


namespace orchard_yield_correct_l295_295597

-- Definitions for conditions
def gala3YrTreesYield : ℕ := 10 * 120
def gala2YrTreesYield : ℕ := 10 * 150
def galaTotalYield : ℕ := gala3YrTreesYield + gala2YrTreesYield

def fuji4YrTreesYield : ℕ := 5 * 180
def fuji5YrTreesYield : ℕ := 5 * 200
def fujiTotalYield : ℕ := fuji4YrTreesYield + fuji5YrTreesYield

def redhaven6YrTreesYield : ℕ := 15 * 50
def redhaven4YrTreesYield : ℕ := 15 * 60
def redhavenTotalYield : ℕ := redhaven6YrTreesYield + redhaven4YrTreesYield

def elberta2YrTreesYield : ℕ := 5 * 70
def elberta3YrTreesYield : ℕ := 5 * 75
def elberta5YrTreesYield : ℕ := 5 * 80
def elbertaTotalYield : ℕ := elberta2YrTreesYield + elberta3YrTreesYield + elberta5YrTreesYield

def appleTotalYield : ℕ := galaTotalYield + fujiTotalYield
def peachTotalYield : ℕ := redhavenTotalYield + elbertaTotalYield
def orchardTotalYield : ℕ := appleTotalYield + peachTotalYield

-- Theorem to prove
theorem orchard_yield_correct : orchardTotalYield = 7375 := 
by sorry

end orchard_yield_correct_l295_295597


namespace cubic_sum_of_roots_l295_295571

theorem cubic_sum_of_roots (r s a b : ℝ) (h1 : r + s = a) (h2 : r * s = b) : 
  r^3 + s^3 = a^3 - 3 * a * b :=
by
  sorry

end cubic_sum_of_roots_l295_295571


namespace convert_to_scientific_notation_l295_295961

theorem convert_to_scientific_notation :
  (1670000000 : ℝ) = 1.67 * 10 ^ 9 := 
by
  sorry

end convert_to_scientific_notation_l295_295961


namespace mixed_fraction_product_l295_295520

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l295_295520


namespace team_total_points_l295_295730

-- Definitions based on conditions
def chandra_points (akiko_points : ℕ) := 2 * akiko_points
def akiko_points (michiko_points : ℕ) := michiko_points + 4
def michiko_points (bailey_points : ℕ) := bailey_points / 2
def bailey_points := 14

-- Total points scored by the team
def total_points :=
  let michiko := michiko_points bailey_points
  let akiko := akiko_points michiko
  let chandra := chandra_points akiko
  bailey_points + michiko + akiko + chandra

theorem team_total_points : total_points = 54 := by
  sorry

end team_total_points_l295_295730


namespace carnival_days_l295_295444

-- Define the given conditions
def total_money := 3168
def daily_income := 144

-- Define the main theorem statement
theorem carnival_days : (total_money / daily_income) = 22 := by
  sorry

end carnival_days_l295_295444


namespace batsman_average_after_17_matches_l295_295021

theorem batsman_average_after_17_matches (A : ℕ) (h : (17 * (A + 3) = 16 * A + 87)) : A + 3 = 39 := by
  sorry

end batsman_average_after_17_matches_l295_295021


namespace net_progress_l295_295804

-- Definitions based on conditions in the problem
def loss := 5
def gain := 9

-- Theorem: Proving the team's net progress
theorem net_progress : (gain - loss) = 4 :=
by
  -- Placeholder for proof
  sorry

end net_progress_l295_295804


namespace sum_with_extra_five_l295_295319

theorem sum_with_extra_five 
  (a b c : ℕ)
  (h1 : a + b = 31)
  (h2 : b + c = 48)
  (h3 : c + a = 55) : 
  a + b + c + 5 = 72 :=
by
  sorry

end sum_with_extra_five_l295_295319


namespace smallest_four_digit_divisible_by_35_l295_295162

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m :=
by {
  use 1015,
  split; try {norm_num},
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    have h5 : m = 1015 ∨ m > 1015, from sorry,
    cases h5, { exact le_of_eq h5 },
    exact h5
  }
}

end smallest_four_digit_divisible_by_35_l295_295162


namespace numbers_to_be_left_out_l295_295274

axiom problem_conditions :
  let numbers := [2, 3, 4, 7, 10, 11, 12, 13, 15]
  let grid_numbers := [1, 9, 14, 5]
  numbers.sum + grid_numbers.sum = 106 ∧
  ∃ (left_out : ℕ) (remaining_numbers : List ℕ),
    numbers.erase left_out = remaining_numbers ∧
    (numbers.sum + grid_numbers.sum - left_out) = 96 ∧
    remaining_numbers.length = 8

theorem numbers_to_be_left_out :
  let numbers := [2, 3, 4, 7, 10, 11, 12, 13, 15]
  10 ∈ numbers ∧
  let grid_numbers := [1, 9, 14, 5]
  let total_sum := numbers.sum + grid_numbers.sum
  let grid_sum := total_sum - 10
  grid_sum % 12 = 0 ∧
  grid_sum = 96 :=
sorry

end numbers_to_be_left_out_l295_295274


namespace card_worth_l295_295110

theorem card_worth (value_per_card : ℕ) (num_cards_traded : ℕ) (profit : ℕ) (value_traded : ℕ) (worth_received : ℕ) :
  value_per_card = 8 →
  num_cards_traded = 2 →
  profit = 5 →
  value_traded = num_cards_traded * value_per_card →
  worth_received = value_traded + profit →
  worth_received = 21 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end card_worth_l295_295110


namespace c_share_of_rent_l295_295463

/-- 
Given the conditions:
- a puts 10 oxen for 7 months,
- b puts 12 oxen for 5 months,
- c puts 15 oxen for 3 months,
- The rent of the pasture is Rs. 210,
Prove that C should pay Rs. 54 as his share of rent.
-/
noncomputable def total_rent : ℝ := 210
noncomputable def oxen_months_a : ℝ := 10 * 7
noncomputable def oxen_months_b : ℝ := 12 * 5
noncomputable def oxen_months_c : ℝ := 15 * 3
noncomputable def total_oxen_months : ℝ := oxen_months_a + oxen_months_b + oxen_months_c

theorem c_share_of_rent : (total_rent / total_oxen_months) * oxen_months_c = 54 :=
by
  sorry

end c_share_of_rent_l295_295463


namespace sulfuric_acid_reaction_l295_295547

theorem sulfuric_acid_reaction (SO₃ H₂O H₂SO₄ : ℕ) 
  (reaction : SO₃ + H₂O = H₂SO₄)
  (H₂O_eq : H₂O = 2)
  (H₂SO₄_eq : H₂SO₄ = 2) :
  SO₃ = 2 :=
by
  sorry

end sulfuric_acid_reaction_l295_295547


namespace remainder_of_127_div_25_is_2_l295_295666

theorem remainder_of_127_div_25_is_2 : ∃ r, 127 = 25 * 5 + r ∧ r = 2 := by
  have h1 : 127 = 25 * 5 + (127 - 25 * 5) := by rw [mul_comm 25 5, mul_comm 5 25]
  have h2 : 127 - 25 * 5 = 2 := by norm_num
  exact ⟨127 - 25 * 5, h1, h2⟩

end remainder_of_127_div_25_is_2_l295_295666


namespace middle_number_l295_295122

theorem middle_number {a b c : ℕ} (h1 : a + b = 12) (h2 : a + c = 17) (h3 : b + c = 19) (h4 : a < b) (h5 : b < c) : b = 7 :=
sorry

end middle_number_l295_295122


namespace commute_weeks_per_month_l295_295966

variable (total_commute_one_way : ℕ)
variable (gas_cost_per_gallon : ℝ)
variable (car_mileage : ℝ)
variable (commute_days_per_week : ℕ)
variable (individual_monthly_payment : ℝ)
variable (number_of_people : ℕ)

theorem commute_weeks_per_month :
  total_commute_one_way = 21 →
  gas_cost_per_gallon = 2.5 →
  car_mileage = 30 →
  commute_days_per_week = 5 →
  individual_monthly_payment = 14 →
  number_of_people = 5 →
  (individual_monthly_payment * number_of_people) / 
  ((total_commute_one_way * 2 / car_mileage) * gas_cost_per_gallon * commute_days_per_week) = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end commute_weeks_per_month_l295_295966


namespace no_common_root_l295_295465

variables {R : Type*} [OrderedRing R]

def f (x m n : R) := x^2 + m*x + n
def p (x k l : R) := x^2 + k*x + l

theorem no_common_root (k m n l : R) (h1 : k > m) (h2 : m > n) (h3 : n > l) (h4 : l > 0) :
  ¬ ∃ x : R, (f x m n = 0 ∧ p x k l = 0) :=
by
  sorry

end no_common_root_l295_295465


namespace geometric_sequence_a2_l295_295567

theorem geometric_sequence_a2 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h1 : a 1 = 1/4) 
  (h3_h5 : a 3 * a 5 = 4 * (a 4 - 1)) 
  (h_seq : ∀ n : ℕ, a n = a 1 * q ^ (n - 1)) :
  a 2 = 1/2 :=
sorry

end geometric_sequence_a2_l295_295567


namespace corrected_average_l295_295628

theorem corrected_average (incorrect_avg : ℕ) (correct_val incorrect_val number_of_values : ℕ) (avg := 17) (n := 10) (inc := 26) (cor := 56) :
  incorrect_avg = 17 →
  number_of_values = 10 →
  correct_val = 56 →
  incorrect_val = 26 →
  correct_avg = (incorrect_avg * number_of_values + (correct_val - incorrect_val)) / number_of_values →
  correct_avg = 20 := by
  sorry

end corrected_average_l295_295628


namespace least_integer_square_double_condition_l295_295140

theorem least_integer_square_double_condition : ∃ x : ℤ, x^2 = 2 * x + 75 ∧ ∀ y : ℤ, y^2 = 2 * y + 75 → x ≤ y :=
by
  use -8
  sorry

end least_integer_square_double_condition_l295_295140


namespace problem_statement_l295_295104

-- Given conditions
variables {p q r t n : ℕ}

axiom prime_p : Nat.Prime p
axiom prime_q : Nat.Prime q
axiom prime_r : Nat.Prime r

axiom nat_n : n ≥ 1
axiom nat_t : t ≥ 1

axiom eqn1 : p^2 + q * t = (p + t)^n
axiom eqn2 : p^2 + q * r = t^4

-- Statement to prove
theorem problem_statement : n < 3 ∧ (p = 2 ∧ q = 7 ∧ r = 11 ∧ t = 3 ∧ n = 2) :=
by
  sorry

end problem_statement_l295_295104


namespace gcd_105_88_l295_295977

-- Define the numbers as constants
def a : ℕ := 105
def b : ℕ := 88

-- State the theorem: gcd(a, b) = 1
theorem gcd_105_88 : Nat.gcd a b = 1 := by
  sorry

end gcd_105_88_l295_295977


namespace smallest_four_digit_number_divisible_by_35_l295_295147

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l295_295147


namespace annika_total_distance_l295_295353

/--
Annika hikes at a constant rate of 12 minutes per kilometer. She has hiked 2.75 kilometers
east from the start of a hiking trail when she realizes that she has to be back at the start
of the trail in 51 minutes. Prove that the total distance Annika hiked east is 3.5 kilometers.
-/
theorem annika_total_distance :
  (hike_rate : ℝ) = 12 → 
  (initial_distance_east : ℝ) = 2.75 → 
  (total_time : ℝ) = 51 → 
  (total_distance_east : ℝ) = 3.5 :=
by 
  intro hike_rate initial_distance_east total_time 
  sorry

end annika_total_distance_l295_295353


namespace restore_original_problem_l295_295489

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l295_295489


namespace asymptotes_of_hyperbola_l295_295769

theorem asymptotes_of_hyperbola : 
  (∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1 → y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intro x y h
  sorry

end asymptotes_of_hyperbola_l295_295769


namespace marguerites_fraction_l295_295956

variable (x r b s : ℕ)

theorem marguerites_fraction
  (h1 : r = 5 * (x - r))
  (h2 : b = (x - b) / 5)
  (h3 : r + b + s = x) : s = 0 := by sorry

end marguerites_fraction_l295_295956


namespace wire_length_between_poles_l295_295308

theorem wire_length_between_poles :
  let d := 18  -- distance between the bottoms of the poles
  let h1 := 6 + 3  -- effective height of the shorter pole
  let h2 := 20  -- height of the taller pole
  let vertical_distance := h2 - h1 -- vertical distance between the tops of the poles
  let hypotenuse := Real.sqrt (d^2 + vertical_distance^2)
  hypotenuse = Real.sqrt 445 :=
by
  sorry

end wire_length_between_poles_l295_295308


namespace interval_contains_zeros_l295_295389

-- Define the conditions and the function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c 

theorem interval_contains_zeros (a b c : ℝ) (h1 : 2 * a + c / 2 > b) (h2 : c < 0) : 
  ∃ x ∈ Set.Ioc (-2 : ℝ) 0, quadratic a b c x = 0 :=
by
  -- Problem Statement: given conditions, interval (-2, 0) contains a zero
  sorry

end interval_contains_zeros_l295_295389


namespace a_plus_b_eq_l295_295386

-- Define the sets A and B
def A := { x : ℝ | -1 < x ∧ x < 3 }
def B := { x : ℝ | -3 < x ∧ x < 2 }

-- Define the intersection set A ∩ B
def A_inter_B := { x : ℝ | -1 < x ∧ x < 2 }

-- Define a condition
noncomputable def is_solution_set (a b : ℝ) : Prop :=
  ∀ x : ℝ, (-1 < x ∧ x < 2) ↔ (x^2 + a * x + b < 0)

-- The proof statement
theorem a_plus_b_eq : ∃ a b : ℝ, is_solution_set a b ∧ a + b = -3 := by
  sorry

end a_plus_b_eq_l295_295386


namespace ellipse_standard_equation_l295_295574

theorem ellipse_standard_equation
  (a b : ℝ) (P : ℝ × ℝ) (h_center : P = (3, 0))
  (h_a_eq_3b : a = 3 * b) 
  (h1 : a = 3) 
  (h2 : b = 1) : 
  (∀ (x y : ℝ), (x = 3 → y = 0) → (x = 0 → y = 3)) → 
  ((x^2 / a^2) + y^2 = 1 ∨ (x^2 / b^2) + (y^2 / a^2) = 1) := 
by sorry

end ellipse_standard_equation_l295_295574


namespace complement_union_correct_l295_295392

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = {2, 4})
variable (hB : B = {3, 4})

theorem complement_union_correct : ((U \ A) ∪ B) = {1, 3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end complement_union_correct_l295_295392


namespace impossible_to_save_one_minute_for_60kmh_l295_295948

theorem impossible_to_save_one_minute_for_60kmh (v : ℝ) (h : v = 60) :
  ¬ ∃ (new_v : ℝ), 1 / new_v = (1 / 60) - 1 :=
by
  sorry

end impossible_to_save_one_minute_for_60kmh_l295_295948


namespace solve_system_equations_l295_295303

theorem solve_system_equations (x y : ℝ) :
  (5 * x^2 + 14 * x * y + 10 * y^2 = 17 ∧ 4 * x^2 + 10 * x * y + 6 * y^2 = 8) ↔
  (x = -1 ∧ y = 2) ∨ (x = 11 ∧ y = -7) ∨ (x = -11 ∧ y = 7) ∨ (x = 1 ∧ y = -2) := 
sorry

end solve_system_equations_l295_295303


namespace solve_fractions_l295_295486

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l295_295486


namespace correct_factorization_l295_295648

theorem correct_factorization :
  (x^2 - 2 * x + 1 = (x - 1)^2) ∧ 
  (¬ (x^2 - 4 * y^2 = (x + y) * (x - 4 * y))) ∧ 
  (¬ ((x + 4) * (x - 4) = x^2 - 16)) ∧ 
  (¬ (x^2 - 8 * x + 9 = (x - 4)^2 - 7)) :=
by
  sorry

end correct_factorization_l295_295648


namespace town_council_original_plan_count_l295_295960

theorem town_council_original_plan_count (planned_trees current_trees : ℕ) (leaves_per_tree total_leaves : ℕ)
  (h1 : leaves_per_tree = 100)
  (h2 : total_leaves = 1400)
  (h3 : current_trees = total_leaves / leaves_per_tree)
  (h4 : current_trees = 2 * planned_trees) : 
  planned_trees = 7 :=
by
  sorry

end town_council_original_plan_count_l295_295960


namespace determine_d_l295_295607

theorem determine_d (f g : ℝ → ℝ) (c d : ℝ) (h1 : ∀ x, f x = 5 * x + c) (h2 : ∀ x, g x = c * x + 3) (h3 : ∀ x, f (g x) = 15 * x + d) : d = 18 := 
  sorry

end determine_d_l295_295607


namespace smallest_four_digit_divisible_by_35_l295_295159

theorem smallest_four_digit_divisible_by_35 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n → m % 35 ≠ 0) := 
begin 
    use 1170, 
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { intro m,
      contrapose,
      norm_num,
      intro h,
      exact h,
    },
end

end smallest_four_digit_divisible_by_35_l295_295159


namespace missing_pieces_l295_295137

-- Definitions based on the conditions.
def total_pieces : ℕ := 500
def border_pieces : ℕ := 75
def trevor_pieces : ℕ := 105
def joe_pieces : ℕ := 3 * trevor_pieces

-- Prove the number of missing pieces is 5.
theorem missing_pieces : total_pieces - (border_pieces + trevor_pieces + joe_pieces) = 5 := by
  sorry

end missing_pieces_l295_295137


namespace strips_overlap_area_l295_295029

theorem strips_overlap_area :
  ∀ (length_left length_right area_only_left area_only_right : ℕ) (S : ℚ),
    length_left = 9 →
    length_right = 7 →
    area_only_left = 27 →
    area_only_right = 18 →
    (area_only_left + S) / (area_only_right + S) = 9 / 7 →
    S = 13.5 :=
by
  intros length_left length_right area_only_left area_only_right S
  intro h1 h2 h3 h4 h5
  sorry

end strips_overlap_area_l295_295029


namespace part1_part2_part3_l295_295950

-- Problem Definitions
def air_conditioner_cost (A B : ℕ → ℕ) :=
  A 3 + B 2 = 39000 ∧ 4 * A 1 - 5 * B 1 = 6000

def possible_schemes (A B : ℕ → ℕ) :=
  ∀ a b, a ≥ b / 2 ∧ 9000 * a + 6000 * b ≤ 217000 ∧ a + b = 30

def minimize_cost (A B : ℕ → ℕ) :=
  ∃ a, (a = 10 ∧ 9000 * a + 6000 * (30 - a) = 210000) ∧
  ∀ b, b ≥ 10 → b ≤ 12 → 9000 * b + 6000 * (30 - b) ≥ 210000

-- Theorem Statements
theorem part1 (A B : ℕ → ℕ) : air_conditioner_cost A B → A 1 = 9000 ∧ B 1 = 6000 :=
by sorry

theorem part2 (A B : ℕ → ℕ) : air_conditioner_cost A B →
  possible_schemes A B :=
by sorry

theorem part3 (A B : ℕ → ℕ) : air_conditioner_cost A B ∧ possible_schemes A B →
  minimize_cost A B :=
by sorry

end part1_part2_part3_l295_295950


namespace john_recreation_percent_l295_295277

theorem john_recreation_percent (W : ℝ) (P : ℝ) (H1 : 0 ≤ P ∧ P ≤ 1) (H2 : 0 ≤ W) (H3 : 0.15 * W = 0.50 * (P * W)) :
  P = 0.30 :=
by
  sorry

end john_recreation_percent_l295_295277


namespace distinct_arrangements_STARS_l295_295718

def num_letters : ℕ := 5
def freq_S : ℕ := 2

theorem distinct_arrangements_STARS :
  (num_letters.factorial / freq_S.factorial) = 60 := 
by
  sorry

end distinct_arrangements_STARS_l295_295718


namespace number_of_pencil_boxes_l295_295256

-- Define the total number of pencils and pencils per box as given conditions
def total_pencils : ℝ := 2592
def pencils_per_box : ℝ := 648.0

-- Problem statement: To prove the number of pencil boxes is 4
theorem number_of_pencil_boxes : total_pencils / pencils_per_box = 4 := by
  sorry

end number_of_pencil_boxes_l295_295256


namespace total_annual_interest_l295_295668

def total_amount : ℝ := 4000
def P1 : ℝ := 2800
def Rate1 : ℝ := 0.03
def Rate2 : ℝ := 0.05

def P2 : ℝ := total_amount - P1
def I1 : ℝ := P1 * Rate1
def I2 : ℝ := P2 * Rate2
def I_total : ℝ := I1 + I2

theorem total_annual_interest : I_total = 144 := by
  sorry

end total_annual_interest_l295_295668


namespace part1_part2_1_part2_2_l295_295246

-- Define the operation
def mul_op (x y : ℚ) : ℚ := x ^ 2 - 3 * y + 3

-- Part 1: Prove (-4) * 2 = 13 given the operation definition
theorem part1 : mul_op (-4) 2 = 13 := sorry

-- Part 2.1: Simplify (a - b) * (a - b)^2
theorem part2_1 (a b : ℚ) : mul_op (a - b) ((a - b) ^ 2) = -2 * a ^ 2 - 2 * b ^ 2 + 4 * a * b + 3 := sorry

-- Part 2.2: Find the value of the expression when a = -2 and b = 1/2
theorem part2_2 : mul_op (-2 - 1/2) ((-2 - 1/2) ^ 2) = -13 / 2 := sorry

end part1_part2_1_part2_2_l295_295246


namespace balloon_arrangements_l295_295086

theorem balloon_arrangements : 
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / (Nat.factorial k1 * Nat.factorial k2) = 1260 := 
by
  let n := 7
  let k1 := 2
  let k2 := 2
  sorry

end balloon_arrangements_l295_295086


namespace manufacturing_section_degrees_l295_295306

def circle_total_degrees : ℕ := 360
def percentage_to_degree (percentage : ℕ) : ℕ := (circle_total_degrees / 100) * percentage
def manufacturing_percentage : ℕ := 60

theorem manufacturing_section_degrees : percentage_to_degree manufacturing_percentage = 216 :=
by
  -- Proof goes here
  sorry

end manufacturing_section_degrees_l295_295306


namespace series_sum_eq_one_fourth_l295_295221

noncomputable def sum_series : ℝ :=
  ∑' n, (3 ^ n / (1 + 3 ^ n + 3 ^ (n + 2) + 3 ^ (2 * n + 2)))

theorem series_sum_eq_one_fourth :
  sum_series = 1 / 4 :=
by
  sorry

end series_sum_eq_one_fourth_l295_295221


namespace maximum_side_length_of_triangle_l295_295201

theorem maximum_side_length_of_triangle (a b c : ℕ) (h_diff: a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_perimeter: a + b + c = 30)
  (h_triangle_inequality_1: a + b > c) 
  (h_triangle_inequality_2: a + c > b) 
  (h_triangle_inequality_3: b + c > a) : 
  c ≤ 14 :=
sorry

end maximum_side_length_of_triangle_l295_295201


namespace eccentricity_range_l295_295244

-- We start with the given problem and conditions
variables {a c b : ℝ}
def C1 := ∀ x y, x^2 + 2 * c * x + y^2 = 0
def C2 := ∀ x y, x^2 - 2 * c * x + y^2 = 0
def ellipse := ∀ x y, x^2 / a^2 + y^2 / b^2 = 1

-- Ellipse semi-latus rectum condition and circles inside the ellipse
axiom h1 : c = b^2 / a
axiom h2 : a > 2 * c

-- Proving the range of the eccentricity
theorem eccentricity_range : 0 < c / a ∧ c / a < 1 / 2 :=
by
  sorry

end eccentricity_range_l295_295244


namespace prove_frac_addition_l295_295964

def frac_addition_correct : Prop :=
  (3 / 8 + 9 / 12 = 9 / 8)

theorem prove_frac_addition : frac_addition_correct :=
  by
  -- We assume the necessary fractions and their properties.
  sorry

end prove_frac_addition_l295_295964


namespace mixed_fractions_product_l295_295502

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l295_295502


namespace abs_x_plus_7_eq_0_has_no_solution_l295_295802

theorem abs_x_plus_7_eq_0_has_no_solution : ¬∃ x : ℝ, |x| + 7 = 0 :=
by
  sorry

end abs_x_plus_7_eq_0_has_no_solution_l295_295802


namespace mixed_fraction_product_l295_295519

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l295_295519


namespace minimum_value_fraction_l295_295282

theorem minimum_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  ∃ (x : ℝ), (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 → x ≤ (a + b) / (a * b * c)) ∧ x = 16 / 9 := 
sorry

end minimum_value_fraction_l295_295282


namespace sqrt_meaningful_range_l295_295092

theorem sqrt_meaningful_range (x : ℝ) : x + 2 ≥ 0 → x ≥ -2 :=
by 
  intro h
  linarith [h]

end sqrt_meaningful_range_l295_295092


namespace max_side_length_triangle_l295_295212

def triangle_with_max_side_length (a b c : ℕ) (ha : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hper : a + b + c = 30) : Prop :=
  a > b ∧ a > c ∧ a = 14

theorem max_side_length_triangle : ∃ a b c : ℕ, 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 30 ∧ a > b ∧ a > c ∧ a = 14 :=
sorry

end max_side_length_triangle_l295_295212


namespace largest_multiple_of_8_less_than_100_l295_295001

theorem largest_multiple_of_8_less_than_100 : ∃ n, n < 100 ∧ n % 8 = 0 ∧ ∀ m, m < 100 ∧ m % 8 = 0 → m ≤ n :=
begin
  use 96,
  split,
  { -- prove 96 < 100
    norm_num,
  },
  split,
  { -- prove 96 is a multiple of 8
    norm_num,
  },
  { -- prove 96 is the largest such multiple
    intros m hm,
    cases hm with h1 h2,
    have h3 : m / 8 < 100 / 8,
    { exact_mod_cast h1 },
    interval_cases (m / 8) with H,
    all_goals { 
      try { norm_num, exact le_refl _ },
    },
  },
end

end largest_multiple_of_8_less_than_100_l295_295001


namespace orange_juice_percentage_l295_295469

theorem orange_juice_percentage 
  (V : ℝ) 
  (W : ℝ) 
  (G : ℝ)
  (hV : V = 300)
  (hW: W = 0.4 * V)
  (hG: G = 105) : 
  (V - W - G) / V * 100 = 25 := 
by 
  -- We will need to use sorry to skip the proof and focus just on the statement
  sorry

end orange_juice_percentage_l295_295469


namespace transformed_parabola_correct_l295_295766

def f (x : ℝ) : ℝ := (x + 2)^2 + 3
def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem transformed_parabola_correct :
  ∀ x : ℝ, g x = f (x - 3) - 2 := by
  sorry

end transformed_parabola_correct_l295_295766


namespace initial_dozens_of_doughnuts_l295_295339

theorem initial_dozens_of_doughnuts (doughnuts_eaten doughnuts_left : ℕ)
  (h_eaten : doughnuts_eaten = 8)
  (h_left : doughnuts_left = 16) :
  (doughnuts_eaten + doughnuts_left) / 12 = 2 := by
  sorry

end initial_dozens_of_doughnuts_l295_295339


namespace exists_divisible_triangle_l295_295305

theorem exists_divisible_triangle (p : ℕ) (n : ℕ) (m : ℕ) (points : Fin m → ℤ × ℤ) 
  (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_pos : 0 < n) (hm_eight : m = 8) 
  (on_circle : ∀ k : Fin m, (points k).fst ^ 2 + (points k).snd ^ 2 = (p ^ n) ^ 2) :
  ∃ (i j k : Fin m), (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ (∃ d : ℕ, (points i).fst - (points j).fst = p ^ d ∧ 
  (points i).snd - (points j).snd = p ^ d ∧ d ≥ n + 1) :=
sorry

end exists_divisible_triangle_l295_295305


namespace m_range_l295_295054

noncomputable def otimes (a b : ℝ) : ℝ := 
if a > b then a else b

theorem m_range (m : ℝ) : (otimes (2 * m - 5) 3 = 3) ↔ (m ≤ 4) := by
  sorry

end m_range_l295_295054


namespace restore_original_problem_l295_295493

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l295_295493


namespace speed_of_current_l295_295027

-- Definitions
def downstream_speed (m current : ℝ) := m + current
def upstream_speed (m current : ℝ) := m - current

-- Theorem
theorem speed_of_current 
  (m : ℝ) (current : ℝ) 
  (h1 : downstream_speed m current = 20) 
  (h2 : upstream_speed m current = 14) : 
  current = 3 :=
by
  -- proof goes here
  sorry

end speed_of_current_l295_295027


namespace mixed_fraction_product_example_l295_295476

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l295_295476


namespace tom_initial_foreign_exchange_l295_295641

theorem tom_initial_foreign_exchange (x : ℝ) (y₀ y₁ y₂ y₃ y₄ : ℝ) :
  y₀ = x / 2 - 5 ∧
  y₁ = y₀ / 2 - 5 ∧
  y₂ = y₁ / 2 - 5 ∧
  y₃ = y₂ / 2 - 5 ∧
  y₄ = y₃ / 2 - 5 ∧
  y₄ - 5 = 100
  → x = 3355 :=
by
  intro h
  sorry

end tom_initial_foreign_exchange_l295_295641


namespace area_sum_eq_l295_295828

-- Define the conditions given in the problem
variables {A B C P Q R M N : Type*}

-- Define the properties of the points
variables (triangle_ABC : Triangle A B C)
          (point_P : OnSegment P A B)
          (point_Q : OnSegment Q B C)
          (point_R : OnSegment R A C)
          (parallelogram_PQCR : Parallelogram P Q C R)
          (intersection_M : Intersection M (LineSegment AQ) (LineSegment PR))
          (intersection_N : Intersection N (LineSegment BR) (LineSegment PQ))

-- Define the areas of the triangles involved
variables (area_AMP area_BNP area_CQR : ℝ)

-- Define the conditions for the areas of the triangles
variables (h_area_AMP : area_AMP = Area (Triangle A M P))
          (h_area_BNP : area_BNP = Area (Triangle B N P))
          (h_area_CQR : area_CQR = Area (Triangle C Q R))

-- The theorem to be proved
theorem area_sum_eq :
  area_AMP + area_BNP = area_CQR :=
sorry

end area_sum_eq_l295_295828


namespace smallest_four_digit_number_divisible_by_35_l295_295149

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l295_295149


namespace initially_calculated_average_height_l295_295625

theorem initially_calculated_average_height
    (A : ℕ)
    (initial_total_height : ℕ)
    (real_total_height : ℕ)
    (height_error : ℕ := 60)
    (num_boys : ℕ := 35)
    (actual_average_height : ℕ := 183)
    (initial_total_height_eq : initial_total_height = num_boys * A)
    (real_total_height_eq : real_total_height = num_boys * actual_average_height)
    (height_discrepancy : initial_total_height = real_total_height + height_error) :
    A = 181 :=
by
  sorry

end initially_calculated_average_height_l295_295625


namespace time_to_fill_pool_l295_295035

theorem time_to_fill_pool :
  let R1 := 1
  let R2 := 1 / 2
  let R3 := 1 / 3
  let R4 := 1 / 4
  let R_total := R1 + R2 + R3 + R4
  let T := 1 / R_total
  T = 12 / 25 := 
by
  sorry

end time_to_fill_pool_l295_295035


namespace probability_not_within_square_b_l295_295184

noncomputable def prob_not_within_square_b : Prop :=
  let area_A := 121
  let side_length_B := 16 / 4
  let area_B := side_length_B * side_length_B
  let area_not_covered := area_A - area_B
  let prob := area_not_covered / area_A
  prob = (105 / 121)

theorem probability_not_within_square_b : prob_not_within_square_b :=
by
  sorry

end probability_not_within_square_b_l295_295184


namespace max_value_l295_295371

noncomputable def max_expression (x : ℝ) : ℝ :=
  3^x - 2 * 9^x

theorem max_value : ∃ x : ℝ, max_expression x = 1 / 8 :=
sorry

end max_value_l295_295371


namespace average_weight_all_children_l295_295116

theorem average_weight_all_children (avg_boys_weight avg_girls_weight : ℝ) (num_boys num_girls : ℕ)
    (hb : avg_boys_weight = 155) (nb : num_boys = 8)
    (hg : avg_girls_weight = 125) (ng : num_girls = 7) :
    (num_boys + num_girls = 15) → (avg_boys_weight * num_boys + avg_girls_weight * num_girls) / (num_boys + num_girls) = 141 := by
  intro h_sum
  sorry

end average_weight_all_children_l295_295116


namespace problem_correctness_l295_295461

theorem problem_correctness
  (correlation_A : ℝ)
  (correlation_B : ℝ)
  (chi_squared : ℝ)
  (P_chi_squared_5_024 : ℝ)
  (P_chi_squared_6_635 : ℝ)
  (P_X_leq_2 : ℝ)
  (P_X_lt_0 : ℝ) :
  correlation_A = 0.66 →
  correlation_B = -0.85 →
  chi_squared = 6.352 →
  P_chi_squared_5_024 = 0.025 →
  P_chi_squared_6_635 = 0.01 →
  P_X_leq_2 = 0.68 →
  P_X_lt_0 = 0.32 →
  (abs correlation_B > abs correlation_A) ∧
  (1 - P_chi_squared_5_024 < 0.99) ∧
  (P_X_lt_0 = 1 - P_X_leq_2) ∧
  (false) := sorry

end problem_correctness_l295_295461
