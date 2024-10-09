import Mathlib

namespace math_problem_l4_439

theorem math_problem :
  -50 * 3 - (-2.5) / 0.1 = -125 := by
sorry

end math_problem_l4_439


namespace price_of_second_candy_l4_403

variables (X P : ℝ)

-- Conditions
def total_weight (X : ℝ) := X + 6.25 = 10
def total_value (X P : ℝ) := 3.50 * X + 6.25 * P = 40

-- Proof problem
theorem price_of_second_candy (h1 : total_weight X) (h2 : total_value X P) : P = 4.30 :=
by 
  sorry

end price_of_second_candy_l4_403


namespace meaningful_fraction_l4_418

theorem meaningful_fraction (x : ℝ) : (x + 5 ≠ 0) → (x ≠ -5) :=
by
  sorry

end meaningful_fraction_l4_418


namespace midpoint_trajectory_l4_423

theorem midpoint_trajectory (x y : ℝ) :
  (∃ B C : ℝ × ℝ, B ≠ C ∧ (B.1^2 + B.2^2 = 25) ∧ (C.1^2 + C.2^2 = 25) ∧ 
                   (x, y) = ((B.1 + C.1)/2, (B.2 + C.2)/2) ∧ 
                   (B.1 - C.1)^2 + (B.2 - C.2)^2 = 36) →
  x^2 + y^2 = 16 :=
sorry

end midpoint_trajectory_l4_423


namespace find_a_l4_479

def star (a b : ℝ) : ℝ := 2 * a - b^3

theorem find_a (a : ℝ) : star a 3 = 15 → a = 21 :=
by
  intro h
  sorry

end find_a_l4_479


namespace domain_of_f_l4_433

-- Define the function f(x) = 1/(x+1) + ln(x)
noncomputable def f (x : ℝ) : ℝ := (1 / (x + 1)) + Real.log x

-- The domain of the function is all x such that x > 0
theorem domain_of_f :
  ∀ x : ℝ, (x > 0) ↔ (f x = (1 / (x + 1)) + Real.log x) := 
by sorry

end domain_of_f_l4_433


namespace total_games_eq_64_l4_482

def games_attended : ℕ := 32
def games_missed : ℕ := 32
def total_games : ℕ := games_attended + games_missed

theorem total_games_eq_64 : total_games = 64 := by
  sorry

end total_games_eq_64_l4_482


namespace choices_of_N_l4_411

def base7_representation (N : ℕ) : ℕ := 
  (N / 49) * 100 + ((N % 49) / 7) * 10 + (N % 7)

def base8_representation (N : ℕ) : ℕ := 
  (N / 64) * 100 + ((N % 64) / 8) * 10 + (N % 8)

theorem choices_of_N : 
  ∃ (N_set : Finset ℕ), 
    (∀ N ∈ N_set, 100 ≤ N ∧ N < 1000 ∧ 
      ((base7_representation N * base8_representation N) % 100 = (3 * N) % 100)) 
    ∧ N_set.card = 15 :=
by
  sorry

end choices_of_N_l4_411


namespace rectangle_no_shaded_square_l4_402

noncomputable def total_rectangles (cols : ℕ) : ℕ :=
  (cols + 1) * (cols + 1 - 1) / 2

noncomputable def shaded_rectangles (cols : ℕ) : ℕ :=
  cols + 1 - 1

noncomputable def probability_no_shaded (cols : ℕ) : ℚ :=
  let n := total_rectangles cols
  let m := shaded_rectangles cols
  1 - (m / n)

theorem rectangle_no_shaded_square :
  probability_no_shaded 2003 = 2002 / 2003 :=
by
  sorry

end rectangle_no_shaded_square_l4_402


namespace solve_custom_operation_l4_408

theorem solve_custom_operation (x : ℤ) (h : ((4 * 3 - (12 - x)) = 2)) : x = -2 :=
by
  sorry

end solve_custom_operation_l4_408


namespace degree_f_x2_mul_g_x4_l4_434

open Polynomial

theorem degree_f_x2_mul_g_x4 {f g : Polynomial ℝ} (hf : degree f = 4) (hg : degree g = 5) :
  degree (f.comp (X ^ 2) * g.comp (X ^ 4)) = 28 :=
sorry

end degree_f_x2_mul_g_x4_l4_434


namespace thirty_percent_more_than_80_is_one_fourth_less_l4_400

-- Translating the mathematical equivalency conditions into Lean definitions and theorems

def thirty_percent_more (n : ℕ) : ℕ :=
  n + (n * 30 / 100)

def one_fourth_less (x : ℕ) : ℕ :=
  x - (x / 4)

theorem thirty_percent_more_than_80_is_one_fourth_less (x : ℕ) :
  thirty_percent_more 80 = one_fourth_less x → x = 139 :=
by
  sorry

end thirty_percent_more_than_80_is_one_fourth_less_l4_400


namespace solve_quadratic_eq_l4_446

theorem solve_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := sorry

end solve_quadratic_eq_l4_446


namespace onions_on_scale_l4_416

theorem onions_on_scale (N : ℕ) (W_total : ℕ) (W_removed : ℕ) (avg_remaining : ℕ) (avg_removed : ℕ) :
  W_total = 7680 →
  W_removed = 5 * 206 →
  avg_remaining = 190 →
  avg_removed = 206 →
  N = 40 :=
by
  sorry

end onions_on_scale_l4_416


namespace mod_inverse_sum_l4_454

theorem mod_inverse_sum :
  ∃ a b : ℕ, (5 * a ≡ 1 [MOD 21]) ∧ (b = (a * a) % 21) ∧ ((a + b) % 21 = 9) :=
by
  sorry

end mod_inverse_sum_l4_454


namespace find_k_l4_427

theorem find_k (k m : ℝ) : (m^2 - 8*m) ∣ (m^3 - k*m^2 - 24*m + 16) → k = 8 := by
  sorry

end find_k_l4_427


namespace find_m_l4_441

theorem find_m (m : ℝ) 
    (h1 : ∃ (m: ℝ), ∀ x y : ℝ, x - m * y + 2 * m = 0) 
    (h2 : ∃ (m: ℝ), ∀ x y : ℝ, x + 2 * y - m = 0) 
    (perpendicular : (1/m) * (-1/2) = -1) : m = 1/2 :=
sorry

end find_m_l4_441


namespace largest_product_is_168_l4_450

open Set

noncomputable def largest_product_from_set (s : Set ℤ) (n : ℕ) (result : ℤ) : Prop :=
  ∃ (a b c : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∀ (x y z : ℤ), x ∈ s → y ∈ s → z ∈ s → x ≠ y → y ≠ z → x ≠ z →
  x * y * z ≤ a * b * c ∧ a * b * c = result

theorem largest_product_is_168 :
  largest_product_from_set {-4, -3, 1, 3, 7, 8} 3 168 :=
sorry

end largest_product_is_168_l4_450


namespace max_area_of_garden_l4_474

theorem max_area_of_garden (l w : ℝ) (h : l + 2*w = 270) : l * w ≤ 9112.5 :=
sorry

end max_area_of_garden_l4_474


namespace expression_eq_l4_459

variable {α β γ δ p q : ℝ}

-- Conditions from the problem
def roots_eq1 (α β p : ℝ) : Prop := ∀ x : ℝ, (x - α) * (x - β) = x^2 + p*x - 1
def roots_eq2 (γ δ q : ℝ) : Prop := ∀ x : ℝ, (x - γ) * (x - δ) = x^2 + q*x + 1

-- The proof statement where the expression is equated to p^2 - q^2
theorem expression_eq (h1: roots_eq1 α β p) (h2: roots_eq2 γ δ q) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = p^2 - q^2 := sorry

end expression_eq_l4_459


namespace find_b_l4_409

noncomputable def func (x a b : ℝ) := (1 / 12) * x^2 + a * x + b

theorem  find_b (a b : ℝ) (x1 x2 : ℝ):
    (func x1 a b = 0) →
    (func x2 a b = 0) →
    (b = (x1 * x2) / 12) →
    ((3 - x1) = (x2 - 3)) →
    (b = -6) :=
by
    sorry

end find_b_l4_409


namespace share_of_B_in_profit_l4_442

variable {D : ℝ} (hD_pos : 0 < D)

def investment (D : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let C := 2.5 * D
  let B := 1.25 * D
  let A := 5 * B
  (A, B, C, D)

def totalInvestment (A B C D : ℝ) : ℝ :=
  A + B + C + D

theorem share_of_B_in_profit (D : ℝ) (profit : ℝ) (hD : 0 < D)
  (h_profit : profit = 8000) :
  let ⟨A, B, C, D⟩ := investment D
  B / totalInvestment A B C D * profit = 1025.64 :=
by
  sorry

end share_of_B_in_profit_l4_442


namespace unique_integer_solution_l4_431

theorem unique_integer_solution (x : ℤ) : x^3 + (x + 1)^3 + (x + 2)^3 = (x + 3)^3 ↔ x = 3 := by
  sorry

end unique_integer_solution_l4_431


namespace a3_equals_neg7_l4_489

-- Definitions based on given conditions
noncomputable def a₁ := -11
noncomputable def d : ℤ := sorry -- this is derived but unknown presently
noncomputable def a(n : ℕ) : ℤ := a₁ + (n - 1) * d

axiom condition : a 4 + a 6 = -6

-- The proof problem statement
theorem a3_equals_neg7 : a 3 = -7 :=
by
  have h₁ : a₁ = -11 := rfl
  have h₂ : a 4 + a 6 = -6 := condition
  sorry

end a3_equals_neg7_l4_489


namespace p_range_l4_464

def h (x : ℝ) : ℝ := 2 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem p_range :
  ∀ x, -1 ≤ x ∧ x ≤ 3 → 29 ≤ p x ∧ p x ≤ 93 :=
by
  intros x hx
  sorry

end p_range_l4_464


namespace angle_ACD_l4_492

theorem angle_ACD {α β δ : Type*} [LinearOrderedField α] [CharZero α] (ABC DAB DBA : α)
  (h1 : ABC = 60) (h2 : BAC = 80) (h3 : DAB = 10) (h4 : DBA = 20):
  ACD = 30 := by
  sorry

end angle_ACD_l4_492


namespace find_expression_value_l4_480

theorem find_expression_value 
  (m : ℝ) 
  (hroot : m^2 - 3 * m + 1 = 0) : 
  (m - 3)^2 + (m + 2) * (m - 2) = 3 := 
sorry

end find_expression_value_l4_480


namespace prime_ratio_sum_l4_463

theorem prime_ratio_sum (p q m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
(h_roots : ∀ x : ℝ, x^2 - 99 * x + m = 0 → x = p ∨ x = q) :
  (p : ℚ) / q + q / p = 9413 / 194 :=
sorry

end prime_ratio_sum_l4_463


namespace find_k_value_l4_462

variables {R : Type*} [Field R] {a b x k : R}

-- Definitions for the conditions in the problem
def f (x : R) (a b : R) : R := (b * x + 1) / (2 * x + a)

-- Statement of the problem
theorem find_k_value (h_ab : a * b ≠ 2)
  (h_k : ∀ (x : R), x ≠ 0 → f x a b * f (x⁻¹) a b = k) :
  k = (1 : R) / 4 :=
by
  sorry

end find_k_value_l4_462


namespace circle_center_line_condition_l4_445

theorem circle_center_line_condition (a : ℝ) :
    (∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 4 * y - 6 = 0 → (a, -2) = (x, y) → x + 2 * y + 1 = 0) → a = 3 :=
by
  sorry

end circle_center_line_condition_l4_445


namespace tennis_balls_ordered_originally_l4_490

-- Definitions according to the conditions in a)
def retailer_ordered_equal_white_yellow_balls (W Y : ℕ) : Prop :=
  W = Y

def dispatch_error (Y : ℕ) : ℕ :=
  Y + 90

def ratio_white_to_yellow (W Y : ℕ) : Prop :=
  W / dispatch_error Y = 8 / 13

-- Main statement
theorem tennis_balls_ordered_originally (W Y : ℕ) (h1 : retailer_ordered_equal_white_yellow_balls W Y)
  (h2 : ratio_white_to_yellow W Y) : W + Y = 288 :=
by
  sorry    -- Placeholder for the actual proof

end tennis_balls_ordered_originally_l4_490


namespace abc_zero_l4_444

-- Define the given conditions as hypotheses
theorem abc_zero (a b c : ℚ) 
  (h1 : (a^2 + 1)^3 = b + 1)
  (h2 : (b^2 + 1)^3 = c + 1)
  (h3 : (c^2 + 1)^3 = a + 1) : 
  a = 0 ∧ b = 0 ∧ c = 0 := 
sorry

end abc_zero_l4_444


namespace base_rate_second_telephone_company_l4_456

theorem base_rate_second_telephone_company : 
  ∃ B : ℝ, (11 + 20 * 0.25 = B + 20 * 0.20) ∧ B = 12 := by
  sorry

end base_rate_second_telephone_company_l4_456


namespace Tony_change_l4_438

structure Conditions where
  bucket_capacity : ℕ := 2
  sandbox_depth : ℕ := 2
  sandbox_width : ℕ := 4
  sandbox_length : ℕ := 5
  sand_weight_per_cubic_foot : ℕ := 3
  water_per_4_trips : ℕ := 3
  water_bottle_cost : ℕ := 2
  water_bottle_volume: ℕ := 15
  initial_money : ℕ := 10

theorem Tony_change (conds : Conditions) : 
  let volume_sandbox := conds.sandbox_depth * conds.sandbox_width * conds.sandbox_length
  let total_weight_sand := volume_sandbox * conds.sand_weight_per_cubic_foot
  let trips := total_weight_sand / conds.bucket_capacity
  let drinks := trips / 4
  let total_water := drinks * conds.water_per_4_trips
  let bottles_needed := total_water / conds.water_bottle_volume
  let total_cost := bottles_needed * conds.water_bottle_cost
  let change := conds.initial_money - total_cost
  change = 4 := by
  /- calculations corresponding to steps that are translated from the solution to show that the 
     change indeed is $4 -/
  sorry

end Tony_change_l4_438


namespace functional_equation_solution_l4_452

noncomputable def f (t : ℝ) (x : ℝ) := (t * (x - t)) / (t + 1)

noncomputable def g (t : ℝ) (x : ℝ) := t * (x - t)

theorem functional_equation_solution (t : ℝ) (ht : t ≠ -1) :
  ∀ x y : ℝ, f t (x + g t y) = x * f t y - y * f t x + g t x :=
by
  intros x y
  let fx := f t
  let gx := g t
  sorry

end functional_equation_solution_l4_452


namespace steel_ingot_weight_l4_460

theorem steel_ingot_weight 
  (initial_weight : ℕ)
  (percent_increase : ℚ)
  (ingot_cost : ℚ)
  (discount_threshold : ℕ)
  (discount_percent : ℚ)
  (total_cost : ℚ)
  (added_weight : ℚ)
  (number_of_ingots : ℕ)
  (ingot_weight : ℚ)
  (h1 : initial_weight = 60)
  (h2 : percent_increase = 0.6)
  (h3 : ingot_cost = 5)
  (h4 : discount_threshold = 10)
  (h5 : discount_percent = 0.2)
  (h6 : total_cost = 72)
  (h7 : added_weight = initial_weight * percent_increase)
  (h8 : added_weight = ingot_weight * number_of_ingots)
  (h9 : total_cost = (ingot_cost * number_of_ingots) * (1 - discount_percent)) :
  ingot_weight = 2 := 
by
  sorry

end steel_ingot_weight_l4_460


namespace juice_profit_eq_l4_478

theorem juice_profit_eq (x : ℝ) :
  (70 - x) * (160 + 8 * x) = 16000 :=
sorry

end juice_profit_eq_l4_478


namespace minimum_value_of_v_l4_426

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

noncomputable def g (x u v : ℝ) : ℝ := (x - u)^3 - 3 * (x - u) - v

theorem minimum_value_of_v (u v : ℝ) (h_pos_u : u > 0) :
  ∀ u > 0, ∀ x : ℝ, f x = g x u v → v ≥ 4 :=
by
  sorry

end minimum_value_of_v_l4_426


namespace total_handshakes_l4_471

variable (n : ℕ) (h : n = 12)

theorem total_handshakes (H : ∀ (b : ℕ), b = n → (n * (n - 1)) / 2 = 66) : 
  (12 * 11) / 2 = 66 := 
by
  sorry

end total_handshakes_l4_471


namespace mark_buttons_l4_414

/-- Mark started the day with some buttons. His friend Shane gave him 3 times that amount of buttons.
    Then his other friend Sam asked if he could have half of Mark’s buttons. 
    Mark ended up with 28 buttons. How many buttons did Mark start the day with? --/
theorem mark_buttons (B : ℕ) (h1 : 2 * B = 28) : B = 14 := by
  sorry

end mark_buttons_l4_414


namespace read_both_books_l4_461

theorem read_both_books (B S K N : ℕ) (TOTAL : ℕ)
  (h1 : S = 1/4 * 72)
  (h2 : K = 5/8 * 72)
  (h3 : N = (S - B) - 1)
  (h4 : TOTAL = 72)
  (h5 : TOTAL = (S - B) + (K - B) + B + N)
  : B = 8 :=
by
  sorry

end read_both_books_l4_461


namespace trigonometric_identity_l4_412

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = -(2 + Real.sqrt 3) / 3 := 
sorry

end trigonometric_identity_l4_412


namespace find_multiple_of_y_l4_484

noncomputable def multiple_of_y (q m : ℝ) : Prop :=
  ∀ x y : ℝ, (x = 5 - q) → (y = m * q - 1) → (q = 1) → (x = 3 * y) → (m = 7 / 3)

theorem find_multiple_of_y :
  multiple_of_y 1 (7 / 3) :=
by
  sorry

end find_multiple_of_y_l4_484


namespace boat_distance_along_stream_l4_420

-- Define the conditions
def boat_speed_still_water := 15 -- km/hr
def distance_against_stream_one_hour := 9 -- km

-- Define the speed of the stream
def stream_speed := boat_speed_still_water - distance_against_stream_one_hour -- km/hr

-- Define the effective speed along the stream
def effective_speed_along_stream := boat_speed_still_water + stream_speed -- km/hr

-- Define the proof statement
theorem boat_distance_along_stream : effective_speed_along_stream = 21 :=
by
  -- Given conditions and definitions, the steps are assumed logically correct
  sorry

end boat_distance_along_stream_l4_420


namespace Gwen_remaining_homework_l4_493

def initial_problems_math := 18
def completed_problems_math := 12
def remaining_problems_math := initial_problems_math - completed_problems_math

def initial_problems_science := 11
def completed_problems_science := 6
def remaining_problems_science := initial_problems_science - completed_problems_science

def initial_questions_history := 15
def completed_questions_history := 10
def remaining_questions_history := initial_questions_history - completed_questions_history

def initial_questions_english := 7
def completed_questions_english := 4
def remaining_questions_english := initial_questions_english - completed_questions_english

def total_remaining_problems := remaining_problems_math 
                               + remaining_problems_science 
                               + remaining_questions_history 
                               + remaining_questions_english

theorem Gwen_remaining_homework : total_remaining_problems = 19 :=
by
  sorry

end Gwen_remaining_homework_l4_493


namespace sum_of_odds_square_l4_443

theorem sum_of_odds_square (n : ℕ) (h : 0 < n) : (Finset.range n).sum (λ i => 2 * i + 1) = n ^ 2 :=
sorry

end sum_of_odds_square_l4_443


namespace caesars_charge_l4_453

theorem caesars_charge :
  ∃ (C : ℕ), (C + 30 * 60 = 500 + 35 * 60) ↔ (C = 800) :=
by
  sorry

end caesars_charge_l4_453


namespace new_average_l4_497

theorem new_average (n : ℕ) (a : ℕ) (multiplier : ℕ) (average : ℕ) :
  (n = 35) →
  (a = 25) →
  (multiplier = 5) →
  (average = 125) →
  ((n * a * multiplier) / n = average) :=
by
  intros hn ha hm havg
  rw [hn, ha, hm]
  norm_num
  sorry

end new_average_l4_497


namespace least_even_perimeter_l4_435

def triangle_perimeter (a b c : ℕ) : ℕ := a + b + c

theorem least_even_perimeter
  (a b : ℕ) (h1 : a = 24) (h2 : b = 37) (c : ℕ)
  (h3 : c > b) (h4 : a + b > c)
  (h5 : ∃ k : ℕ, k * 2 = triangle_perimeter a b c) :
  triangle_perimeter a b c = 100 :=
sorry

end least_even_perimeter_l4_435


namespace pirate_overtakes_at_8pm_l4_491

noncomputable def pirate_overtake_trade : Prop :=
  let initial_distance := 15
  let pirate_speed_before_damage := 14
  let trade_speed := 10
  let time_before_damage := 3
  let pirate_distance_before_damage := pirate_speed_before_damage * time_before_damage
  let trade_distance_before_damage := trade_speed * time_before_damage
  let remaining_distance := initial_distance + trade_distance_before_damage - pirate_distance_before_damage
  let pirate_speed_after_damage := (18 / 17) * 10
  let relative_speed_after_damage := pirate_speed_after_damage - trade_speed
  let time_to_overtake_after_damage := remaining_distance / relative_speed_after_damage
  let total_time := time_before_damage + time_to_overtake_after_damage
  total_time = 8

theorem pirate_overtakes_at_8pm : pirate_overtake_trade :=
by
  sorry

end pirate_overtakes_at_8pm_l4_491


namespace circumcenter_coords_l4_401

-- Define the given points A, B, and C
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (-5, 1)
def C : ℝ × ℝ := (3, -5)

-- The target statement to prove
theorem circumcenter_coords :
  ∃ x y : ℝ, (x - 2)^2 + (y - 2)^2 = (x + 5)^2 + (y - 1)^2 ∧
             (x - 2)^2 + (y - 2)^2 = (x - 3)^2 + (y + 5)^2 ∧
             x = -1 ∧ y = -2 :=
by
  sorry

end circumcenter_coords_l4_401


namespace m_greater_than_p_l4_486

theorem m_greater_than_p (p m n : ℕ) (hp : Nat.Prime p) (hm : 0 < m) (hn : 0 < n) (eq : p^2 + m^2 = n^2) : m > p :=
sorry

end m_greater_than_p_l4_486


namespace fraction_to_decimal_l4_473

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end fraction_to_decimal_l4_473


namespace perimeter_non_shaded_region_l4_472

def shaded_area : ℤ := 78
def large_rect_area : ℤ := 80
def small_rect_area : ℤ := 8
def total_area : ℤ := large_rect_area + small_rect_area
def non_shaded_area : ℤ := total_area - shaded_area
def non_shaded_width : ℤ := 2
def non_shaded_length : ℤ := non_shaded_area / non_shaded_width
def non_shaded_perimeter : ℤ := 2 * (non_shaded_length + non_shaded_width)

theorem perimeter_non_shaded_region : non_shaded_perimeter = 14 := 
by
  exact rfl

end perimeter_non_shaded_region_l4_472


namespace team_selection_count_l4_451

-- The problem's known conditions
def boys : ℕ := 10
def girls : ℕ := 12
def team_size : ℕ := 8

-- The number of ways to select a team of 8 members with at least 2 boys and no more than 4 boys
noncomputable def count_ways : ℕ :=
  (Nat.choose boys 2) * (Nat.choose girls 6) +
  (Nat.choose boys 3) * (Nat.choose girls 5) +
  (Nat.choose boys 4) * (Nat.choose girls 4)

-- The main statement to prove
theorem team_selection_count : count_ways = 238570 := by
  sorry

end team_selection_count_l4_451


namespace ellipse_foci_on_y_axis_l4_419

theorem ellipse_foci_on_y_axis (m : ℝ) : 
  (∃ (x y : ℝ), (x^2 / (m + 2)) - (y^2 / (m + 1)) = 1) ↔ (-2 < m ∧ m < -3/2) := 
by
  sorry

end ellipse_foci_on_y_axis_l4_419


namespace smallest_number_divisible_l4_415

theorem smallest_number_divisible
  (x : ℕ)
  (h : (x - 2) % 12 = 0 ∧ (x - 2) % 16 = 0 ∧ (x - 2) % 18 = 0 ∧ (x - 2) % 21 = 0 ∧ (x - 2) % 28 = 0) :
  x = 1010 :=
by
  sorry

end smallest_number_divisible_l4_415


namespace josh_remaining_marbles_l4_405

theorem josh_remaining_marbles : 
  let initial_marbles := 19 
  let lost_marbles := 11
  initial_marbles - lost_marbles = 8 := by
  sorry

end josh_remaining_marbles_l4_405


namespace equilateral_triangle_area_perimeter_l4_422

theorem equilateral_triangle_area_perimeter (altitude : ℝ) : 
  altitude = Real.sqrt 12 →
  (exists area perimeter : ℝ, area = 4 * Real.sqrt 3 ∧ perimeter = 12) :=
by
  intro h_alt
  sorry

end equilateral_triangle_area_perimeter_l4_422


namespace boys_from_clay_l4_465

theorem boys_from_clay (total_students jonas_students clay_students pine_students total_boys total_girls : ℕ)
  (jonas_girls pine_boys : ℕ) 
  (H1 : total_students = 150)
  (H2 : jonas_students = 50)
  (H3 : clay_students = 60)
  (H4 : pine_students = 40)
  (H5 : total_boys = 80)
  (H6 : total_girls = 70)
  (H7 : jonas_girls = 30)
  (H8 : pine_boys = 15):
  ∃ (clay_boys : ℕ), clay_boys = 45 :=
by
  have jonas_boys : ℕ := jonas_students - jonas_girls
  have boys_from_clay := total_boys - pine_boys - jonas_boys
  exact ⟨boys_from_clay, by sorry⟩

end boys_from_clay_l4_465


namespace distance_to_right_focus_l4_436

variable (F1 F2 P : ℝ × ℝ)
variable (a : ℝ)
variable (h_ellipse : ∀ P : ℝ × ℝ, P ∈ { P : ℝ × ℝ | (P.1^2 / 9) + (P.2^2 / 8) = 1 })
variable (h_foci_dist : (P : ℝ × ℝ) → (F1 : ℝ × ℝ) → (F2 : ℝ × ℝ) → (dist P F1) = 2)
variable (semi_major_axis : a = 3)

theorem distance_to_right_focus (h : dist F1 F2 = 2 * a) : dist P F2 = 4 := 
sorry

end distance_to_right_focus_l4_436


namespace polygon_with_20_diagonals_is_octagon_l4_449

theorem polygon_with_20_diagonals_is_octagon :
  ∃ (n : ℕ), n ≥ 3 ∧ (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end polygon_with_20_diagonals_is_octagon_l4_449


namespace jelly_cost_l4_466

theorem jelly_cost (N B J : ℕ) (hN_gt_1 : N > 1) (h_cost_eq : N * (3 * B + 7 * J) = 252) : 7 * N * J = 168 := by
  sorry

end jelly_cost_l4_466


namespace arithmetic_sequence_sum_l4_476

variable (a : ℕ → ℝ) (d : ℝ)

-- Condition: The sequence {a_n} is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom a1 : a 1 = 2
axiom a2_a3_sum : a 2 + a 3 = 13

-- The theorem to be proved
theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a d) : a (4) + a (5) + a (6) = 42 :=
sorry

end arithmetic_sequence_sum_l4_476


namespace range_of_m_l4_437

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem range_of_m (m : ℝ) (h : second_quadrant (m-3) (m-2)) : 2 < m ∧ m < 3 :=
sorry

end range_of_m_l4_437


namespace sum_of_five_integers_l4_455

theorem sum_of_five_integers :
  ∃ (n m : ℕ), (n * (n + 1) = 336) ∧ ((m - 1) * m * (m + 1) = 336) ∧ ((n + (n + 1) + (m - 1) + m + (m + 1)) = 51) := 
sorry

end sum_of_five_integers_l4_455


namespace solve_for_q_l4_483

theorem solve_for_q (k l q : ℕ) (h1 : (2 : ℚ) / 3 = k / 45) (h2 : (2 : ℚ) / 3 = (k + l) / 75) (h3 : (2 : ℚ) / 3 = (q - l) / 105) : q = 90 :=
sorry

end solve_for_q_l4_483


namespace percent_of_b_l4_407

variables (a b c : ℝ)

theorem percent_of_b (h1 : c = 0.30 * a) (h2 : b = 1.20 * a) : c = 0.25 * b :=
by sorry

end percent_of_b_l4_407


namespace number_of_people_l4_406

-- Definitions based on the conditions
def average_age (T : ℕ) (n : ℕ) := T / n = 30
def youngest_age := 3
def average_age_when_youngest_born (T : ℕ) (n : ℕ) := (T - youngest_age) / (n - 1) = 27

theorem number_of_people (T n : ℕ) (h1 : average_age T n) (h2 : average_age_when_youngest_born T n) : n = 7 :=
by
  sorry

end number_of_people_l4_406


namespace factorization_of_polynomial_l4_429

theorem factorization_of_polynomial (x : ℝ) : 12 * x^2 - 40 * x + 25 = (2 * Real.sqrt 3 * x - 5)^2 :=
  sorry

end factorization_of_polynomial_l4_429


namespace ac_length_l4_432

theorem ac_length (a b c d e : ℝ)
  (h1 : b - a = 5)
  (h2 : c - b = 2 * (d - c))
  (h3 : e - d = 4)
  (h4 : e - a = 18) :
  d - a = 11 :=
by
  sorry

end ac_length_l4_432


namespace log_domain_l4_428

theorem log_domain (x : ℝ) : 3 - 2 * x > 0 ↔ x < 3 / 2 :=
by
  sorry

end log_domain_l4_428


namespace factor_2210_two_digit_l4_469

theorem factor_2210_two_digit :
  (∃ (a b : ℕ), a * b = 2210 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99) ∧
  (∃ (c d : ℕ), c * d = 2210 ∧ 10 ≤ c ∧ c ≤ 99 ∧ 10 ≤ d ∧ d ≤ 99) ∧
  (∀ (x y : ℕ), x * y = 2210 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 → 
   ((x = c ∧ y = d) ∨ (x = d ∧ y = c) ∨ (x = a ∧ y = b) ∨ (x = b ∧ y = a))) :=
sorry

end factor_2210_two_digit_l4_469


namespace side_length_of_S2_l4_425

variables (s r : ℕ)

-- Conditions
def combined_width_eq : Prop := 3 * s + 100 = 4000
def combined_height_eq : Prop := 2 * r + s = 2500

-- Conclusion we want to prove
theorem side_length_of_S2 : combined_width_eq s → combined_height_eq s r → s = 1300 :=
by
  intros h_width h_height
  sorry

end side_length_of_S2_l4_425


namespace total_letters_sent_l4_470

-- Define the number of letters sent in each month
def letters_in_January : ℕ := 6
def letters_in_February : ℕ := 9
def letters_in_March : ℕ := 3 * letters_in_January

-- Theorem statement: the total number of letters sent across the three months
theorem total_letters_sent :
  letters_in_January + letters_in_February + letters_in_March = 33 := by
  sorry

end total_letters_sent_l4_470


namespace jamie_avg_is_correct_l4_477

-- Declare the set of test scores and corresponding sums
def test_scores : List ℤ := [75, 78, 82, 85, 88, 91]

-- Alex's average score
def alex_avg : ℤ := 82

-- Total test score sum
def total_sum : ℤ := test_scores.sum

theorem jamie_avg_is_correct (alex_sum : ℤ) :
    alex_sum = 3 * alex_avg →
    (total_sum - alex_sum) / 3 = 253 / 3 :=
by
  sorry

end jamie_avg_is_correct_l4_477


namespace coefficient_of_linear_term_l4_457

def polynomial (x : ℝ) := x^2 - 2 * x - 3

theorem coefficient_of_linear_term : (∀ x : ℝ, polynomial x = x^2 - 2 * x - 3) → -2 = -2 := by
  intro h
  sorry

end coefficient_of_linear_term_l4_457


namespace vectors_not_coplanar_l4_410

def vector_a : Fin 3 → ℤ := ![1, 5, 2]
def vector_b : Fin 3 → ℤ := ![-1, 1, -1]
def vector_c : Fin 3 → ℤ := ![1, 1, 1]

def scalar_triple_product (a b c : Fin 3 → ℤ) : ℤ :=
  a 0 * (b 1 * c 2 - b 2 * c 1) -
  a 1 * (b 0 * c 2 - b 2 * c 0) +
  a 2 * (b 0 * c 1 - b 1 * c 0)

theorem vectors_not_coplanar :
  scalar_triple_product vector_a vector_b vector_c ≠ 0 :=
by
  sorry

end vectors_not_coplanar_l4_410


namespace max_quadratic_function_l4_495

def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 12 * x - 5

theorem max_quadratic_function : ∃ x, quadratic_function x = 7 ∧ ∀ x', quadratic_function x' ≤ 7 :=
by
  sorry

end max_quadratic_function_l4_495


namespace initial_speed_increase_l4_498

variables (S : ℝ) (P : ℝ)

/-- Prove that the initial percentage increase in speed P is 0.3 based on the given conditions: 
1. After the first increase by P, the speed becomes S + PS.
2. After the second increase by 10%, the final speed is (S + PS) * 1.10.
3. The total increase results in a speed that is 1.43 times the original speed S. -/
theorem initial_speed_increase (h : (S + P * S) * 1.1 = 1.43 * S) : P = 0.3 :=
sorry

end initial_speed_increase_l4_498


namespace rope_length_in_cm_l4_417

-- Define the given conditions
def num_equal_pieces : ℕ := 150
def length_equal_piece_mm : ℕ := 75
def num_remaining_pieces : ℕ := 4
def length_remaining_piece_mm : ℕ := 100

-- Prove that the total length of the rope in centimeters is 1165
theorem rope_length_in_cm : (num_equal_pieces * length_equal_piece_mm + num_remaining_pieces * length_remaining_piece_mm) / 10 = 1165 :=
by
  sorry

end rope_length_in_cm_l4_417


namespace least_non_lucky_multiple_of_8_l4_487

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (· + ·) 0

def lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def multiple_of_8 (n : ℕ) : Prop :=
  n % 8 = 0

theorem least_non_lucky_multiple_of_8 : ∃ n > 0, multiple_of_8 n ∧ ¬ lucky n ∧ n = 16 :=
by
  -- Proof goes here.
  sorry

end least_non_lucky_multiple_of_8_l4_487


namespace rectangle_aspect_ratio_l4_485

theorem rectangle_aspect_ratio (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x / y = 2 * y / x) : x / y = Real.sqrt 2 :=
by
  sorry

end rectangle_aspect_ratio_l4_485


namespace large_pretzel_cost_l4_404

theorem large_pretzel_cost : 
  ∀ (P S : ℕ), 
  P = 3 * S ∧ 7 * P + 4 * S = 4 * P + 7 * S + 12 → 
  P = 6 :=
by sorry

end large_pretzel_cost_l4_404


namespace Michael_pizza_fraction_l4_413

theorem Michael_pizza_fraction (T : ℚ) (L : ℚ) (total : ℚ) (M : ℚ) 
  (hT : T = 1 / 2) (hL : L = 1 / 6) (htotal : total = 1) (hM : total - (T + L) = M) :
  M = 1 / 3 := 
sorry

end Michael_pizza_fraction_l4_413


namespace right_triangle_area_l4_496

theorem right_triangle_area (A B C : ℝ) (hA : A = 64) (hB : B = 49) (hC : C = 225) :
  let a := Real.sqrt A
  let b := Real.sqrt B
  let c := Real.sqrt C
  ∃ (area : ℝ), area = (1 / 2) * a * b ∧ area = 28 :=
by
  sorry

end right_triangle_area_l4_496


namespace no_real_roots_of_quadratic_l4_467

theorem no_real_roots_of_quadratic (k : ℝ) (hk : k ≠ 0) : ¬ ∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 = 0 :=
by
  sorry

end no_real_roots_of_quadratic_l4_467


namespace correct_option_l4_430

theorem correct_option : 
  (-(2:ℤ))^3 ≠ -6 ∧ 
  (-(1:ℤ))^10 ≠ -10 ∧ 
  (-(1:ℚ)/3)^3 ≠ -1/9 ∧ 
  -(2:ℤ)^2 = -4 :=
by 
  sorry

end correct_option_l4_430


namespace comparison_of_exponential_values_l4_421

theorem comparison_of_exponential_values : 
  let a := 0.3^3
  let b := 3^0.3
  let c := 0.2^3
  c < a ∧ a < b := 
by 
  let a := 0.3^3
  let b := 3^0.3
  let c := 0.2^3
  sorry

end comparison_of_exponential_values_l4_421


namespace sum_of_first_20_terms_arithmetic_sequence_l4_447

theorem sum_of_first_20_terms_arithmetic_sequence 
  (a : ℕ → ℤ)
  (h_arith : ∃ d : ℤ, ∀ n, a n = a 0 + n * d)
  (h_sum_first_three : a 0 + a 1 + a 2 = -24)
  (h_sum_eighteen_nineteen_twenty : a 17 + a 18 + a 19 = 78) :
  (20 / 2 * (a 0 + (a 0 + 19 * d))) = 180 :=
by
  sorry

end sum_of_first_20_terms_arithmetic_sequence_l4_447


namespace longer_leg_smallest_triangle_l4_448

noncomputable def length_of_longer_leg_of_smallest_triangle (n : ℕ) (a : ℝ) : ℝ :=
  if n = 0 then a 
  else if n = 1 then (a / 2) * Real.sqrt 3
  else if n = 2 then ((a / 2) * Real.sqrt 3 / 2) * Real.sqrt 3
  else ((a / 2) * Real.sqrt 3 / 2 * Real.sqrt 3 / 2) * Real.sqrt 3

theorem longer_leg_smallest_triangle : 
  length_of_longer_leg_of_smallest_triangle 3 10 = 45 / 8 := 
sorry

end longer_leg_smallest_triangle_l4_448


namespace triangle_inequality_check_l4_475

theorem triangle_inequality_check (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  (a = 5 ∧ b = 8 ∧ c = 12) → (a + b > c ∧ b + c > a ∧ c + a > b) :=
by 
  intros h
  rcases h with ⟨rfl, rfl, rfl⟩
  exact ⟨h1, h2, h3⟩

end triangle_inequality_check_l4_475


namespace g_does_not_pass_second_quadrant_l4_468

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x+5) + 4

def M : ℝ × ℝ := (-5, 5)

noncomputable def g (x : ℝ) : ℝ := -5 + (5 : ℝ)^(x)

theorem g_does_not_pass_second_quadrant (a : ℝ) (x : ℝ) 
  (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hM : f a (-5) = 5) : 
  ∀ x < 0, g x < 0 :=
by
  sorry

end g_does_not_pass_second_quadrant_l4_468


namespace least_five_digit_congruent_to_7_mod_17_l4_424

theorem least_five_digit_congruent_to_7_mod_17 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 17 = 7 ∧ (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 7 → n ≤ m) :=
sorry

end least_five_digit_congruent_to_7_mod_17_l4_424


namespace additional_charge_l4_440

variable (charge_first : ℝ) -- The charge for the first 1/5 of a mile
variable (total_charge : ℝ) -- Total charge for an 8-mile ride
variable (distance : ℝ) -- Total distance of the ride

theorem additional_charge 
  (h1 : charge_first = 3.50) 
  (h2 : total_charge = 19.1) 
  (h3 : distance = 8) :
  ∃ x : ℝ, x = 0.40 :=
  sorry

end additional_charge_l4_440


namespace first_system_solution_second_system_solution_l4_481

theorem first_system_solution (x y : ℝ) (h₁ : 3 * x - y = 8) (h₂ : 3 * x - 5 * y = -20) : 
  x = 5 ∧ y = 7 := 
by
  sorry

theorem second_system_solution (x y : ℝ) (h₁ : x / 3 - y / 2 = -1) (h₂ : 3 * x - 2 * y = 1) : 
  x = 3 ∧ y = 4 := 
by
  sorry

end first_system_solution_second_system_solution_l4_481


namespace euclid_1976_part_a_problem_4_l4_488

theorem euclid_1976_part_a_problem_4
  (p q y1 y2 : ℝ)
  (h1 : y1 = p * 1^2 + q * 1 + 5)
  (h2 : y2 = p * (-1)^2 + q * (-1) + 5)
  (h3 : y1 + y2 = 14) :
  p = 2 :=
by
  sorry

end euclid_1976_part_a_problem_4_l4_488


namespace pct_three_petals_is_75_l4_499

-- Given Values
def total_clovers : Nat := 200
def pct_two_petals : Nat := 24
def pct_four_petals : Nat := 1

-- Statement: Prove that the percentage of clovers with three petals is 75%
theorem pct_three_petals_is_75 :
  (100 - pct_two_petals - pct_four_petals) = 75 := by
  sorry

end pct_three_petals_is_75_l4_499


namespace bug_total_distance_l4_458

/-!
# Problem Statement
A bug starts crawling on a number line from position -3. It first moves to -7, then turns around and stops briefly at 0 before continuing on to 8. Prove that the total distance the bug crawls is 19 units.
-/

def bug_initial_position : ℤ := -3
def bug_position_1 : ℤ := -7
def bug_position_2 : ℤ := 0
def bug_final_position : ℤ := 8

theorem bug_total_distance : 
  |bug_position_1 - bug_initial_position| + 
  |bug_position_2 - bug_position_1| + 
  |bug_final_position - bug_position_2| = 19 :=
by 
  sorry

end bug_total_distance_l4_458


namespace geometric_sequence_problem_l4_494

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

-- Define the statement for the roots of the quadratic function
def is_root (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f a = 0

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ :=
  x^2 - x - 2013

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : is_root quadratic_function (a 2)) 
  (h3 : is_root quadratic_function (a 3)) : 
  a 1 * a 4 = -2013 :=
sorry

end geometric_sequence_problem_l4_494
