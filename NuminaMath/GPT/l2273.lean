import Mathlib

namespace multiplication_approximation_correct_l2273_227304

noncomputable def closest_approximation (x : ℝ) : ℝ := 
  if 15700 <= x ∧ x < 15750 then 15700
  else if 15750 <= x ∧ x < 15800 then 15750
  else if 15800 <= x ∧ x < 15900 then 15800
  else if 15900 <= x ∧ x < 16000 then 15900
  else 16000

theorem multiplication_approximation_correct :
  closest_approximation (0.00525 * 3153420) = 15750 := 
by
  sorry

end multiplication_approximation_correct_l2273_227304


namespace investment_years_l2273_227330

def principal (P : ℝ) := P = 1200
def rate (r : ℝ) := r = 0.10
def interest_diff (P r : ℝ) (t : ℝ) :=
  let SI := P * r * t
  let CI := P * (1 + r)^t - P
  CI - SI = 12

theorem investment_years (P r : ℝ) (t : ℝ) 
  (h_principal : principal P) 
  (h_rate : rate r) 
  (h_diff : interest_diff P r t) : 
  t = 2 := 
sorry

end investment_years_l2273_227330


namespace cos_arith_prog_impossible_l2273_227362

noncomputable def sin_arith_prog (x y z : ℝ) : Prop :=
  (2 * Real.sin y = Real.sin x + Real.sin z) ∧ (Real.sin x < Real.sin y) ∧ (Real.sin y < Real.sin z)

theorem cos_arith_prog_impossible (x y z : ℝ) (h : sin_arith_prog x y z) : 
  ¬(2 * Real.cos y = Real.cos x + Real.cos z) := 
by 
  sorry

end cos_arith_prog_impossible_l2273_227362


namespace max_odd_integers_l2273_227394

theorem max_odd_integers (a b c d e f : ℕ) 
  (hprod : a * b * c * d * e * f % 2 = 0) 
  (hpos_a : 0 < a) (hpos_b : 0 < b) 
  (hpos_c : 0 < c) (hpos_d : 0 < d) 
  (hpos_e : 0 < e) (hpos_f : 0 < f) : 
  ∃ x : ℕ, x ≤ 5 ∧ x = 5 :=
by sorry

end max_odd_integers_l2273_227394


namespace battery_current_l2273_227339

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l2273_227339


namespace best_in_district_round_l2273_227333

-- Assume a structure that lets us refer to positions
inductive Position
| first
| second
| third
| last

open Position

-- Definitions of the statements
def Eva (p : Position → Prop) := ¬ (p first) ∧ ¬ (p last)
def Mojmir (p : Position → Prop) := ¬ (p last)
def Karel (p : Position → Prop) := p first
def Peter (p : Position → Prop) := p last

-- The main hypothesis
def exactly_one_lie (p : Position → Prop) :=
  (Eva p ∧ Mojmir p ∧ Karel p ∧ ¬ (Peter p)) ∨
  (Eva p ∧ Mojmir p ∧ ¬ (Karel p) ∧ Peter p) ∨
  (Eva p ∧ ¬ (Mojmir p) ∧ Karel p ∧ Peter p) ∨
  (¬ (Eva p) ∧ Mojmir p ∧ Karel p ∧ Peter p)

theorem best_in_district_round :
  ∃ (p : Position → Prop),
    (Eva p ∧ Mojmir p ∧ ¬ (Karel p) ∧ Peter p) ∧ exactly_one_lie p :=
by
  sorry

end best_in_district_round_l2273_227333


namespace geometric_series_problem_l2273_227312

theorem geometric_series_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ)
  (h_seq : ∀ n, a n + a (n + 1) = 3 * 2^n) :
  S (k + 2) - 2 * S (k + 1) + S k = 2^(k + 1) :=
sorry

end geometric_series_problem_l2273_227312


namespace solution_proof_l2273_227302

variable (x y z : ℝ)

-- Given system of equations
def equation1 := 6 / (3 * x + 4 * y) + 4 / (5 * x - 4 * z) = 7 / 12
def equation2 := 9 / (4 * y + 3 * z) - 4 / (3 * x + 4 * y) = 1 / 3
def equation3 := 2 / (5 * x - 4 * z) + 6 / (4 * y + 3 * z) = 1 / 2

theorem solution_proof : 
  equation1 4 3 2 ∧ equation2 4 3 2 ∧ equation3 4 3 2 := by
  sorry

end solution_proof_l2273_227302


namespace find_f_2018_l2273_227387

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_functional_eq : ∀ x : ℝ, f x = - (1 / f (x + 3))
axiom f_at_4 : f 4 = -2018

theorem find_f_2018 : f 2018 = -2018 :=
  sorry

end find_f_2018_l2273_227387


namespace net_salary_change_l2273_227390

variable (S : ℝ)

theorem net_salary_change (h1 : S > 0) : 
  (1.3 * S - 0.3 * (1.3 * S)) - S = -0.09 * S := by
  sorry

end net_salary_change_l2273_227390


namespace smallest_b_l2273_227356

noncomputable def Q (b : ℤ) (x : ℤ) : ℤ := sorry -- Q is a polynomial, will be defined in proof

theorem smallest_b (b : ℤ) 
  (h1 : b > 0) 
  (h2 : ∀ x, x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 → Q b x = b) 
  (h3 : ∀ x, x = 1 ∨ x = 3 ∨ x = 5 ∨ x = 7 → Q b x = -b) 
  : b = 315 := sorry

end smallest_b_l2273_227356


namespace dasha_rectangle_l2273_227350

theorem dasha_rectangle:
  ∃ (a b c : ℤ), a * (2 * b + 2 * c - a) = 43 ∧ a = 1 ∧ b + c = 22 :=
by
  sorry

end dasha_rectangle_l2273_227350


namespace sum_of_present_ages_l2273_227345

def Jed_age_future (current_Jed: ℕ) (years: ℕ) : ℕ := 
  current_Jed + years

def Matt_age (current_Jed: ℕ) : ℕ := 
  current_Jed - 10

def sum_ages (jed_age: ℕ) (matt_age: ℕ) : ℕ := 
  jed_age + matt_age

theorem sum_of_present_ages :
  ∃ jed_curr_age matt_curr_age : ℕ, 
  (Jed_age_future jed_curr_age 10 = 25) ∧ 
  (jed_curr_age = matt_curr_age + 10) ∧ 
  (sum_ages jed_curr_age matt_curr_age = 20) :=
sorry

end sum_of_present_ages_l2273_227345


namespace saving_percentage_l2273_227398

variable (I S : Real)

-- Conditions
def cond1 : Prop := S = 0.3 * I -- Man saves 30% of his income

def cond2 : Prop := let income_next_year := 1.3 * I
                    let savings_next_year := 2 * S
                    let expenditure_first_year := I - S
                    let expenditure_second_year := income_next_year - savings_next_year
                    expenditure_first_year + expenditure_second_year = 2 * expenditure_first_year

-- Question
theorem saving_percentage :
  cond1 I S →
  cond2 I S →
  S = 0.3 * I :=
by
  intros
  sorry

end saving_percentage_l2273_227398


namespace wooden_toy_price_l2273_227344

noncomputable def price_of_hat : ℕ := 10
noncomputable def total_money : ℕ := 100
noncomputable def hats_bought : ℕ := 3
noncomputable def change_received : ℕ := 30
noncomputable def total_spent := total_money - change_received
noncomputable def cost_of_hats := hats_bought * price_of_hat

theorem wooden_toy_price :
  ∃ (W : ℕ), total_spent = 2 * W + cost_of_hats ∧ W = 20 := 
by 
  sorry

end wooden_toy_price_l2273_227344


namespace total_nails_l2273_227358

-- Definitions based on the conditions
def Violet_nails : ℕ := 27
def Tickletoe_nails : ℕ := (27 - 3) / 2

-- Theorem to prove the total number of nails
theorem total_nails : Violet_nails + Tickletoe_nails = 39 := by
  sorry

end total_nails_l2273_227358


namespace temperature_difference_l2273_227308

theorem temperature_difference (high low : ℝ) (h_high : high = 5) (h_low : low = -3) :
  high - low = 8 :=
by {
  -- Proof goes here
  sorry
}

end temperature_difference_l2273_227308


namespace prime_product_sum_l2273_227331

theorem prime_product_sum (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (h : (p * q * r = 101 * (p + q + r))) : 
  p = 101 ∧ q = 2 ∧ r = 103 :=
sorry

end prime_product_sum_l2273_227331


namespace row_trip_time_example_l2273_227332

noncomputable def round_trip_time
    (rowing_speed : ℝ)
    (current_speed : ℝ)
    (total_distance : ℝ) : ℝ :=
  let downstream_speed := rowing_speed + current_speed
  let upstream_speed := rowing_speed - current_speed
  let one_way_distance := total_distance / 2
  let time_to_place := one_way_distance / downstream_speed
  let time_back := one_way_distance / upstream_speed
  time_to_place + time_back

theorem row_trip_time_example :
  round_trip_time 10 2 96 = 10 := by
  sorry

end row_trip_time_example_l2273_227332


namespace greatest_4_digit_base7_divisible_by_7_l2273_227307

-- Definitions and conditions
def is_base7_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 7, d < 7

def is_4_digit_base7 (n : ℕ) : Prop :=
  is_base7_number n ∧ 343 ≤ n ∧ n < 2401 -- 343 = 7^3 (smallest 4-digit base 7) and 2401 = 7^4

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

-- Proof problem statement
theorem greatest_4_digit_base7_divisible_by_7 :
  ∃ (n : ℕ), is_4_digit_base7 n ∧ is_divisible_by_7 n ∧ n = 2346 :=
sorry

end greatest_4_digit_base7_divisible_by_7_l2273_227307


namespace value_of_a_add_b_l2273_227327

theorem value_of_a_add_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
sorry

end value_of_a_add_b_l2273_227327


namespace distinct_triangle_areas_l2273_227384

variables (A B C D E F G : ℝ) (h : ℝ)
variables (AB BC CD EF FG AC BD AD EG : ℝ)

def is_valid_points := AB = 2 ∧ BC = 1 ∧ CD = 3 ∧ EF = 1 ∧ FG = 2 ∧ AC = AB + BC ∧ BD = BC + CD ∧ AD = AB + BC + CD ∧ EG = EF + FG

theorem distinct_triangle_areas (h_pos : 0 < h) (valid : is_valid_points AB BC CD EF FG AC BD AD EG) : 
  ∃ n : ℕ, n = 5 := 
by
  sorry

end distinct_triangle_areas_l2273_227384


namespace bus_A_speed_l2273_227391

variable (v_A v_B : ℝ)
variable (h1 : v_A - v_B = 15)
variable (h2 : v_A + v_B = 75)

theorem bus_A_speed : v_A = 45 := sorry

end bus_A_speed_l2273_227391


namespace binom_divisibility_l2273_227313

theorem binom_divisibility (p : ℕ) (h₀ : Nat.Prime p) (h₁ : p % 2 = 1) : 
  (Nat.choose (2 * p - 1) (p - 1) - 1) % (p^2) = 0 := 
by 
  sorry

end binom_divisibility_l2273_227313


namespace maximum_mark_for_paper_i_l2273_227378

noncomputable def maximum_mark (pass_percentage: ℝ) (secured_marks: ℝ) (failed_by: ℝ) : ℝ :=
  (secured_marks + failed_by) / pass_percentage

theorem maximum_mark_for_paper_i :
  maximum_mark 0.35 42 23 = 186 :=
by
  sorry

end maximum_mark_for_paper_i_l2273_227378


namespace parallel_to_a_perpendicular_to_a_l2273_227305

-- Definition of vectors a and b and conditions
def a : ℝ × ℝ := (3, 4)
def b (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Mathematical statement for Problem (1)
theorem parallel_to_a (x y : ℝ) (h : b x y) (h_parallel : 3 * y - 4 * x = 0) :
  (x = 3/5 ∧ y = 4/5) ∨ (x = -3/5 ∧ y = -4/5) := 
sorry

-- Mathematical statement for Problem (2)
theorem perpendicular_to_a (x y : ℝ) (h : b x y) (h_perpendicular : 3 * x + 4 * y = 0) :
  (x = -4/5 ∧ y = 3/5) ∨ (x = 4/5 ∧ y = -3/5) := 
sorry

end parallel_to_a_perpendicular_to_a_l2273_227305


namespace find_a5_l2273_227309

variable {a_n : ℕ → ℤ}
variable (d : ℤ)

def arithmetic_sequence (a_n : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  a_n 1 = a1 ∧ ∀ n, a_n (n + 1) = a_n n + d

theorem find_a5 (h_seq : arithmetic_sequence a_n 6 d) (h_a3 : a_n 3 = 2) : a_n 5 = -2 :=
by
  obtain ⟨h_a1, h_arith⟩ := h_seq
  sorry

end find_a5_l2273_227309


namespace larger_value_algebraic_expression_is_2_l2273_227315

noncomputable def algebraic_expression (a b c d x : ℝ) : ℝ :=
  x^2 + a + b + c * d * x

theorem larger_value_algebraic_expression_is_2
  (a b c d : ℝ) (x : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1)
  (h3 : x = 1 ∨ x = -1) :
  max (algebraic_expression a b c d 1) (algebraic_expression a b c d (-1)) = 2 :=
by
  -- Proof is omitted.
  sorry

end larger_value_algebraic_expression_is_2_l2273_227315


namespace f_increasing_f_t_range_l2273_227337

noncomputable def f : Real → Real :=
  sorry

axiom f_prop1 : f 2 = 1
axiom f_prop2 : ∀ x, x > 1 → f x > 0
axiom f_prop3 : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

theorem f_increasing (x1 x2 : Real) (hx1 : x1 > 0) (hx2 : x2 > 0) (h : x1 < x2) : f x1 < f x2 := by
  sorry

theorem f_t_range (t : Real) (ht : t > 0) (ht3 : t - 3 > 0) (hf : f t + f (t - 3) ≤ 2) : 3 < t ∧ t ≤ 4 := by
  sorry

end f_increasing_f_t_range_l2273_227337


namespace player1_wins_game_533_player1_wins_game_1000_l2273_227300

-- Defining a structure for the game conditions
structure Game :=
  (target_sum : ℕ)
  (player1_wins_optimal : Bool)

-- Definition of the game scenarios
def game_533 := Game.mk 533 true
def game_1000 := Game.mk 1000 true

-- Theorem statements for the respective games
theorem player1_wins_game_533 : game_533.player1_wins_optimal :=
by sorry

theorem player1_wins_game_1000 : game_1000.player1_wins_optimal :=
by sorry

end player1_wins_game_533_player1_wins_game_1000_l2273_227300


namespace distinct_quadrilateral_areas_l2273_227351

theorem distinct_quadrilateral_areas (A B C D E F : ℝ) 
  (h : A + B + C + D + E + F = 156) :
  ∃ (Q1 Q2 Q3 : ℝ), Q1 = 78 ∧ Q2 = 104 ∧ Q3 = 104 :=
sorry

end distinct_quadrilateral_areas_l2273_227351


namespace solve_system_of_equations_l2273_227388

theorem solve_system_of_equations (a1 a2 a3 a4 : ℝ) (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (x1 x2 x3 x4 : ℝ)
  (h1 : |a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1)
  (h2 : |a2 - a1| * x1 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1)
  (h3 : |a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a4| * x4 = 1)
  (h4 : |a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 = 1) :
  x1 = 1 / (a4 - a1) ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 1 / (a4 - a1) := 
sorry

end solve_system_of_equations_l2273_227388


namespace median_possible_values_l2273_227364

variable {ι : Type} -- Representing the set S as a type
variable (S : Finset ℤ) -- S is a finite set of integers

def conditions (S: Finset ℤ) : Prop :=
  S.card = 9 ∧
  {5, 7, 10, 13, 17, 21} ⊆ S

theorem median_possible_values :
  ∀ S : Finset ℤ, conditions S → ∃ medians : Finset ℤ, medians.card = 7 :=
by
  sorry

end median_possible_values_l2273_227364


namespace six_digit_number_representation_l2273_227386

-- Defining that a is a two-digit number
def isTwoDigitNumber (a : ℕ) : Prop := a >= 10 ∧ a < 100

-- Defining that b is a four-digit number
def isFourDigitNumber (b : ℕ) : Prop := b >= 1000 ∧ b < 10000

-- The statement that placing a to the left of b forms the number 10000*a + b
theorem six_digit_number_representation (a b : ℕ) 
  (ha : isTwoDigitNumber a) 
  (hb : isFourDigitNumber b) : 
  (10000 * a + b) = (10^4 * a + b) :=
by
  sorry

end six_digit_number_representation_l2273_227386


namespace discount_percentage_is_10_l2273_227357

-- Definitions of the conditions directly translated
def CP (MP : ℝ) : ℝ := 0.7 * MP
def GainPercent : ℝ := 0.2857142857142857
def SP (MP : ℝ) : ℝ := CP MP * (1 + GainPercent)

-- Using the alternative expression for selling price involving discount percentage
def DiscountSP (MP : ℝ) (D : ℝ) : ℝ := MP * (1 - D)

-- The theorem to prove the discount percentage is 10%
theorem discount_percentage_is_10 (MP : ℝ) : ∃ D : ℝ, DiscountSP MP D = SP MP ∧ D = 0.1 := 
by
  use 0.1
  sorry

end discount_percentage_is_10_l2273_227357


namespace prove_angle_A_l2273_227361

-- Definitions and conditions in triangle ABC
variables (A B C : ℝ) (a b c : ℝ) (h₁ : a^2 - b^2 = 3 * b * c) (h₂ : sin C = 2 * sin B)

-- Objective: Prove that angle A is 120 degrees
theorem prove_angle_A : A = 120 :=
sorry

end prove_angle_A_l2273_227361


namespace rectangles_perimeter_l2273_227383

theorem rectangles_perimeter : 
  let base := 4
  let top := 4
  let left_side := 4
  let right_side := 6
  base + top + left_side + right_side = 18 := 
by {
  let base := 4
  let top := 4
  let left_side := 4
  let right_side := 6
  sorry
}

end rectangles_perimeter_l2273_227383


namespace angle_quadrant_l2273_227360

def same_terminal_side (θ α : ℝ) (k : ℤ) : Prop :=
  θ = α + 360 * k

def in_first_quadrant (α : ℝ) : Prop :=
  0 < α ∧ α < 90

theorem angle_quadrant (θ : ℝ) (k : ℤ) (h : same_terminal_side θ 12 k) : in_first_quadrant 12 :=
  by
    sorry

end angle_quadrant_l2273_227360


namespace zoo_structure_l2273_227370

theorem zoo_structure (P : ℕ) (h1 : ∃ (snakes monkeys elephants zebras : ℕ),
  snakes = 3 * P ∧
  monkeys = 6 * P ∧
  elephants = (P + snakes) / 2 ∧
  zebras = elephants - 3 ∧
  monkeys - zebras = 35) : P = 8 :=
sorry

end zoo_structure_l2273_227370


namespace olivia_spent_amount_l2273_227340

noncomputable def initial_amount : ℕ := 100
noncomputable def collected_amount : ℕ := 148
noncomputable def final_amount : ℕ := 159

theorem olivia_spent_amount :
  initial_amount + collected_amount - final_amount = 89 :=
by
  sorry

end olivia_spent_amount_l2273_227340


namespace michael_weight_loss_in_may_l2273_227367

-- Defining the conditions
def weight_loss_goal : ℕ := 10
def weight_loss_march : ℕ := 3
def weight_loss_april : ℕ := 4

-- Statement of the problem to prove
theorem michael_weight_loss_in_may (weight_loss_goal weight_loss_march weight_loss_april : ℕ) :
  weight_loss_goal - (weight_loss_march + weight_loss_april) = 3 :=
by
  sorry

end michael_weight_loss_in_may_l2273_227367


namespace least_possible_area_of_square_l2273_227381

theorem least_possible_area_of_square :
  (∃ (side_length : ℝ), 3.5 ≤ side_length ∧ side_length < 4.5 ∧ 
    (∃ (area : ℝ), area = side_length * side_length ∧ 
    (∀ (side : ℝ), 3.5 ≤ side ∧ side < 4.5 → side * side ≥ 12.25))) :=
sorry

end least_possible_area_of_square_l2273_227381


namespace minimize_distance_midpoint_Q5_Q6_l2273_227366

theorem minimize_distance_midpoint_Q5_Q6 
  (Q : ℝ → ℝ)
  (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10 : ℝ)
  (h1 : Q2 = Q1 + 1)
  (h2 : Q3 = Q2 + 1)
  (h3 : Q4 = Q3 + 1)
  (h4 : Q5 = Q4 + 1)
  (h5 : Q6 = Q5 + 2)
  (h6 : Q7 = Q6 + 2)
  (h7 : Q8 = Q7 + 2)
  (h8 : Q9 = Q8 + 2)
  (h9 : Q10 = Q9 + 2) :
  Q ((Q5 + Q6) / 2) = (Q ((Q1 + Q2) / 2) + Q ((Q3 + Q4) / 2) + Q ((Q7 + Q8) / 2) + Q ((Q9 + Q10) / 2)) :=
sorry

end minimize_distance_midpoint_Q5_Q6_l2273_227366


namespace proof_f_f_f_3_l2273_227323

def f (n : ℤ) : ℤ :=
  if n < 5
  then n^2 + 1
  else 2 * n - 3

theorem proof_f_f_f_3 :
  f (f (f 3)) = 31 :=
by 
  -- Here, we skip the proof as instructed
  sorry

end proof_f_f_f_3_l2273_227323


namespace min_value_5_l2273_227355

theorem min_value_5 (x y : ℝ) : ∃ x y : ℝ, (xy - 2)^2 + (x + y + 1)^2 = 5 :=
sorry

end min_value_5_l2273_227355


namespace average_sales_six_months_l2273_227321

theorem average_sales_six_months :
  let sales1 := 4000
  let sales2 := 6524
  let sales3 := 5689
  let sales4 := 7230
  let sales5 := 6000
  let sales6 := 12557
  let total_sales_first_five := sales1 + sales2 + sales3 + sales4 + sales5
  let total_sales_six := total_sales_first_five + sales6
  let average_sales := total_sales_six / 6
  average_sales = 7000 :=
by
  let sales1 := 4000
  let sales2 := 6524
  let sales3 := 5689
  let sales4 := 7230
  let sales5 := 6000
  let sales6 := 12557
  let total_sales_first_five := sales1 + sales2 + sales3 + sales4 + sales5
  let total_sales_six := total_sales_first_five + sales6
  let average_sales := total_sales_six / 6
  have h : total_sales_first_five = 29443 := by sorry
  have h1 : total_sales_six = 42000 := by sorry
  have h2 : average_sales = 7000 := by sorry
  exact h2

end average_sales_six_months_l2273_227321


namespace range_of_m_l2273_227346

noncomputable def condition_p (x : ℝ) : Prop := -2 < x ∧ x < 10
noncomputable def condition_q (x m : ℝ) : Prop := (x - 1)^2 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) :
  (∀ x, condition_p x → condition_q x m) ∧ (∃ x, ¬ condition_p x ∧ condition_q x m) ↔ 9 ≤ m := sorry

end range_of_m_l2273_227346


namespace greatest_divisor_of_three_consecutive_odds_l2273_227369

theorem greatest_divisor_of_three_consecutive_odds (n : ℕ) : 
  ∃ (d : ℕ), (∀ (k : ℕ), k = 2*n + 1 ∨ k = 2*n + 3 ∨ k = 2*n + 5 → d ∣ (2*n + 1) * (2*n + 3) * (2*n + 5)) ∧ d = 3 :=
by
  sorry

end greatest_divisor_of_three_consecutive_odds_l2273_227369


namespace tank_full_capacity_is_72_l2273_227303

theorem tank_full_capacity_is_72 (x : ℝ) 
  (h1 : 0.9 * x - 0.4 * x = 36) : 
  x = 72 := 
sorry

end tank_full_capacity_is_72_l2273_227303


namespace area_ratio_l2273_227374

-- Definitions for the geometric entities
structure Point where
  x : ℚ
  y : ℚ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨0, 4⟩
def C : Point := ⟨2, 4⟩
def D : Point := ⟨2, 0⟩
def E : Point := ⟨1, 2⟩  -- Midpoint of BD
def F : Point := ⟨6 / 5, 0⟩  -- Given DF = 2/5 DA

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P Q R : Point) : ℚ :=
  (1 / 2) * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

-- Function to calculate the sum of the area of two triangles
def quadrilateral_area (P Q R S : Point) : ℚ :=
  triangle_area P Q R + triangle_area P R S

-- Prove the ratio of the areas
theorem area_ratio : 
  triangle_area D F E / quadrilateral_area A B E F = 4 / 13 := 
by {
  sorry
}

end area_ratio_l2273_227374


namespace xy_extrema_l2273_227319

noncomputable def xy_product (a : ℝ) : ℝ := a^2 - 1

theorem xy_extrema (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^2 + y^2 = -a^2 + 2) : 
  -1 ≤ xy_product a ∧ xy_product a ≤ 1/3 :=
by
  sorry

end xy_extrema_l2273_227319


namespace integral_eval_l2273_227375

noncomputable def integral_problem : ℝ :=
  ∫ x in - (Real.pi / 2)..(Real.pi / 2), (x + Real.cos x)

theorem integral_eval : integral_problem = 2 :=
  by 
  sorry

end integral_eval_l2273_227375


namespace sum_of_angles_is_correct_l2273_227349

noncomputable def hexagon_interior_angle : ℝ := 180 * (6 - 2) / 6
noncomputable def pentagon_interior_angle : ℝ := 180 * (5 - 2) / 5
noncomputable def sum_of_hexagon_and_pentagon_angles (A B C D : Type) 
  (hexagon_interior_angle : ℝ) 
  (pentagon_interior_angle : ℝ) : ℝ := 
  hexagon_interior_angle + pentagon_interior_angle

theorem sum_of_angles_is_correct (A B C D : Type) : 
  sum_of_hexagon_and_pentagon_angles A B C D hexagon_interior_angle pentagon_interior_angle = 228 := 
by
  simp [hexagon_interior_angle, pentagon_interior_angle]
  sorry

end sum_of_angles_is_correct_l2273_227349


namespace total_fencing_cost_l2273_227314

theorem total_fencing_cost
  (length : ℝ) 
  (breadth : ℝ)
  (cost_per_meter : ℝ)
  (h1 : length = 61)
  (h2 : length = breadth + 22)
  (h3 : cost_per_meter = 26.50) : 
  2 * (length + breadth) * cost_per_meter = 5300 := 
by 
  sorry

end total_fencing_cost_l2273_227314


namespace percent_calculation_l2273_227354

theorem percent_calculation (x : ℝ) (h : 0.40 * x = 160) : 0.30 * x = 120 :=
by
  sorry

end percent_calculation_l2273_227354


namespace committee_count_l2273_227320

-- Definitions based on conditions
def num_males := 15
def num_females := 10

-- Define the binomial coefficient
def binomial (n k : ℕ) := Nat.choose n k

-- Define the total number of committees
def num_committees_with_at_least_two_females : ℕ :=
  binomial num_females 2 * binomial num_males 3 +
  binomial num_females 3 * binomial num_males 2 +
  binomial num_females 4 * binomial num_males 1 +
  binomial num_females 5 * binomial num_males 0

theorem committee_count : num_committees_with_at_least_two_females = 36477 :=
by {
  sorry
}

end committee_count_l2273_227320


namespace triangle_cos_identity_l2273_227338

variable {A B C : ℝ} -- Angle A, B, C are real numbers representing the angles of the triangle
variable {a b c : ℝ} -- Sides a, b, c are real numbers representing the lengths of the sides of the triangle

theorem triangle_cos_identity (h : 2 * b = a + c) : 5 * (Real.cos A) - 4 * (Real.cos A) * (Real.cos C) + 5 * (Real.cos C) = 4 :=
by
  sorry

end triangle_cos_identity_l2273_227338


namespace uniqueFlavors_l2273_227368

-- Definitions for the conditions
def numRedCandies : ℕ := 6
def numGreenCandies : ℕ := 4
def numBlueCandies : ℕ := 5

-- Condition stating each flavor must use at least two candies and no more than two colors
def validCombination (x y z : ℕ) : Prop :=
  (x = 0 ∨ y = 0 ∨ z = 0) ∧ (x + y ≥ 2 ∨ x + z ≥ 2 ∨ y + z ≥ 2)

-- The main theorem statement
theorem uniqueFlavors : 
  ∃ n : ℕ, n = 30 ∧ 
  (∀ x y z : ℕ, validCombination x y z → (x ≤ numRedCandies) ∧ (y ≤ numGreenCandies) ∧ (z ≤ numBlueCandies)) :=
sorry

end uniqueFlavors_l2273_227368


namespace larger_triangle_perimeter_is_126_l2273_227363

noncomputable def smaller_triangle_side1 : ℝ := 12
noncomputable def smaller_triangle_side2 : ℝ := 12
noncomputable def smaller_triangle_base : ℝ := 18
noncomputable def larger_triangle_longest_side : ℝ := 54
noncomputable def similarity_ratio : ℝ := larger_triangle_longest_side / smaller_triangle_base
noncomputable def larger_triangle_side1 : ℝ := smaller_triangle_side1 * similarity_ratio
noncomputable def larger_triangle_side2 : ℝ := smaller_triangle_side2 * similarity_ratio
noncomputable def larger_triangle_perimeter : ℝ := larger_triangle_side1 + larger_triangle_side2 + larger_triangle_longest_side

theorem larger_triangle_perimeter_is_126 :
  larger_triangle_perimeter = 126 := by
  sorry

end larger_triangle_perimeter_is_126_l2273_227363


namespace height_of_bottom_step_l2273_227352

variable (h l w : ℝ)

theorem height_of_bottom_step
  (h l w : ℝ)
  (eq1 : l + h - w / 2 = 42)
  (eq2 : 2 * l + h = 38)
  (w_value : w = 4) : h = 34 := by
sorry

end height_of_bottom_step_l2273_227352


namespace polynomial_expansion_l2273_227393

theorem polynomial_expansion :
  let x := 1 
  let y := -1 
  let a_0 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_1 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  let a_2 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_3 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  let a_4 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_5 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3 + a_5)^2 = 3125 := by
sorry

end polynomial_expansion_l2273_227393


namespace union_of_sets_l2273_227396

def setA := { x : ℝ | -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3 }
def setB := { x : ℝ | (x - 2) / x ≤ 0 }

theorem union_of_sets :
  { x : ℝ | -1 ≤ x ∧ x ≤ 2 } = setA ∪ setB :=
by
  sorry

end union_of_sets_l2273_227396


namespace integer_with_exactly_12_integers_to_its_left_l2273_227382

theorem integer_with_exactly_12_integers_to_its_left :
  let initial_list := List.range' 1 20
  let first_half := initial_list.take 10
  let second_half := initial_list.drop 10
  let new_list := second_half ++ first_half
  new_list.get! 12 = 3 :=
by
  let initial_list := List.range' 1 20
  let first_half := initial_list.take 10
  let second_half := initial_list.drop 10
  let new_list := second_half ++ first_half
  sorry

end integer_with_exactly_12_integers_to_its_left_l2273_227382


namespace perfect_square_difference_of_solutions_l2273_227395

theorem perfect_square_difference_of_solutions
  (x y : ℤ) (hx : x > 0) (hy : y > 0) 
  (h : 3 * x^2 + x = 4 * y^2 + y) : ∃ k : ℤ, k^2 = x - y := 
sorry

end perfect_square_difference_of_solutions_l2273_227395


namespace monotonically_increasing_range_a_l2273_227347

theorem monotonically_increasing_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (x^3 + a * x) ≤ (y^3 + a * y)) → a ≥ 0 := 
by
  sorry

end monotonically_increasing_range_a_l2273_227347


namespace total_weight_all_bags_sold_l2273_227376

theorem total_weight_all_bags_sold (morning_potatoes afternoon_potatoes morning_onions afternoon_onions morning_carrots afternoon_carrots : ℕ)
  (weight_potatoes weight_onions weight_carrots total_weight : ℕ)
  (h_morning_potatoes : morning_potatoes = 29)
  (h_afternoon_potatoes : afternoon_potatoes = 17)
  (h_morning_onions : morning_onions = 15)
  (h_afternoon_onions : afternoon_onions = 22)
  (h_morning_carrots : morning_carrots = 12)
  (h_afternoon_carrots : afternoon_carrots = 9)
  (h_weight_potatoes : weight_potatoes = 7)
  (h_weight_onions : weight_onions = 5)
  (h_weight_carrots : weight_carrots = 4)
  (h_total_weight : total_weight = 591) :
  morning_potatoes + afternoon_potatoes * weight_potatoes +
  morning_onions + afternoon_onions * weight_onions +
  morning_carrots + afternoon_carrots * weight_carrots = total_weight :=
by {
  sorry
}

end total_weight_all_bags_sold_l2273_227376


namespace machines_together_work_time_l2273_227317

theorem machines_together_work_time :
  let rate_A := 1 / 4
  let rate_B := 1 / 12
  let rate_C := 1 / 6
  let rate_D := 1 / 8
  let rate_E := 1 / 18
  let total_rate := rate_A + rate_B + rate_C + rate_D + rate_E
  total_rate ≠ 0 → 
  let total_time := 1 / total_rate
  total_time = 72 / 49 :=
by
  sorry

end machines_together_work_time_l2273_227317


namespace ac_lt_bd_l2273_227334

theorem ac_lt_bd (a b c d : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) : a * c < b * d :=
by
  sorry

end ac_lt_bd_l2273_227334


namespace proof_problem_l2273_227377

noncomputable def p : Prop := ∃ x : ℝ, x^2 + 1 / x^2 ≤ 2
def q : Prop := ¬ p

theorem proof_problem : q ∧ (p ∨ q) :=
by
  -- Insert proof here
  sorry

end proof_problem_l2273_227377


namespace quadratic_radical_condition_l2273_227311

variable (x : ℝ)

theorem quadratic_radical_condition : 
  (∃ (r : ℝ), r = x^2 + 1 ∧ r ≥ 0) ↔ (True) := by
  sorry

end quadratic_radical_condition_l2273_227311


namespace solution_set_inequality_l2273_227336

theorem solution_set_inequality (x : ℝ) (h1 : 2 < 1 / (x - 1)) (h2 : 1 / (x - 1) < 3) (h3 : x - 1 > 0) :
  4 / 3 < x ∧ x < 3 / 2 :=
sorry

end solution_set_inequality_l2273_227336


namespace scooped_water_amount_l2273_227328

variables (x : ℝ)

def initial_water_amount : ℝ := 10
def total_amount : ℝ := initial_water_amount
def alcohol_concentration : ℝ := 0.75

theorem scooped_water_amount (h : x / total_amount = alcohol_concentration) : x = 7.5 :=
by sorry

end scooped_water_amount_l2273_227328


namespace find_other_endpoint_l2273_227343

theorem find_other_endpoint 
    (Mx My : ℝ) (x1 y1 : ℝ) 
    (hx_Mx : Mx = 3) (hy_My : My = 1)
    (hx1 : x1 = 7) (hy1 : y1 = -3) : 
    ∃ (x2 y2 : ℝ), Mx = (x1 + x2) / 2 ∧ My = (y1 + y2) / 2 ∧ x2 = -1 ∧ y2 = 5 :=
by
    sorry

end find_other_endpoint_l2273_227343


namespace soybeans_to_oil_kg_l2273_227348

-- Define initial data
def kgSoybeansToTofu : ℕ := 3
def kgSoybeansToOil : ℕ := 6
def kgTofuCostPerKg : ℕ := 3
def kgOilCostPerKg : ℕ := 15
def batchSoybeansKg : ℕ := 460
def totalRevenue : ℕ := 1800

-- Define problem statement
theorem soybeans_to_oil_kg (x y : ℕ) (h : x + y = batchSoybeansKg) 
  (hRevenue : 3 * kgTofuCostPerKg * x + (kgOilCostPerKg * y) / (kgSoybeansToOil) = totalRevenue) : 
  y = 360 :=
sorry

end soybeans_to_oil_kg_l2273_227348


namespace percentage_loss_l2273_227329

theorem percentage_loss (SP_loss SP_profit CP : ℝ) 
  (h₁ : SP_loss = 9) 
  (h₂ : SP_profit = 11.8125) 
  (h₃ : SP_profit = CP * 1.05) : 
  (CP - SP_loss) / CP * 100 = 20 :=
by sorry

end percentage_loss_l2273_227329


namespace exists_monomials_l2273_227353

theorem exists_monomials (a b : ℕ) :
  ∃ x y : ℕ → ℕ → ℤ,
  (x 2 1 * y 2 1 = -12) ∧
  (∀ m n : ℕ, m ≠ 2 ∨ n ≠ 1 → x m n = 0 ∧ y m n = 0) ∧
  (∃ k l : ℤ, x 2 1 = k * (a ^ 2 * b ^ 1) ∧ y 2 1 = l * (a ^ 2 * b ^ 1) ∧ k + l = 1) :=
by
  sorry

end exists_monomials_l2273_227353


namespace A_work_days_l2273_227341

theorem A_work_days (x : ℝ) :
  (1 / x + 1 / 6 + 1 / 12 = 7 / 24) → x = 24 :=
by
  intro h
  sorry

end A_work_days_l2273_227341


namespace problem_statement_l2273_227326

theorem problem_statement
  (x y : ℝ)
  (h1 : x * y = 1 / 9)
  (h2 : x * (y + 1) = 7 / 9)
  (h3 : y * (x + 1) = 5 / 18) :
  (x + 1) * (y + 1) = 35 / 18 :=
by
  sorry

end problem_statement_l2273_227326


namespace value_of_a_l2273_227301

theorem value_of_a (a : ℝ) (h : (a ^ 3) * ((5).choose (2)) = 80) : a = 2 :=
  sorry

end value_of_a_l2273_227301


namespace ducks_in_smaller_pond_l2273_227316

theorem ducks_in_smaller_pond (x : ℝ) (h1 : 50 > 0) 
  (h2 : 0.20 * x > 0) (h3 : 0.12 * 50 > 0) (h4 : 0.15 * (x + 50) = 0.20 * x + 0.12 * 50) 
  : x = 30 := 
sorry

end ducks_in_smaller_pond_l2273_227316


namespace smallest_area_of_right_triangle_l2273_227379

noncomputable def right_triangle_area (a b : ℝ) : ℝ :=
  if a^2 + b^2 = 6^2 then (1/2) * a * b else 12

theorem smallest_area_of_right_triangle :
  min (right_triangle_area 4 (2 * Real.sqrt 5)) 12 = 4 * Real.sqrt 5 :=
by
  -- Proof goes here
  sorry

end smallest_area_of_right_triangle_l2273_227379


namespace complex_number_sum_l2273_227397

variable (ω : ℂ)
variable (h1 : ω^9 = 1)
variable (h2 : ω ≠ 1)

theorem complex_number_sum :
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80 = ω^2 :=
by sorry

end complex_number_sum_l2273_227397


namespace inequality_solution_l2273_227359

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := |2 - 3 * x| ≥ 4

-- Define the solution set
def solution_set (x : ℝ) : Prop := x ≤ -2/3 ∨ x ≥ 2

-- The theorem that we need to prove
theorem inequality_solution : {x : ℝ | inequality_condition x} = {x : ℝ | solution_set x} :=
by sorry

end inequality_solution_l2273_227359


namespace find_abscissa_of_P_l2273_227318

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem find_abscissa_of_P (x_P : ℝ) :
  (x + 2*y - 1 = 0 -> 
  (f' x_P = 2 -> 
  (f x_P - 2) * (x_P^2 - 1) = 0)) := by
  sorry

end find_abscissa_of_P_l2273_227318


namespace g_of_f_of_3_is_217_l2273_227389

-- Definitions of f and g
def f (x : ℝ) : ℝ := x^2 - 4
def g (x : ℝ) : ℝ := x^3 + 3 * x^2 + 3 * x + 2

-- The theorem we need to prove
theorem g_of_f_of_3_is_217 : g (f 3) = 217 := by
  sorry

end g_of_f_of_3_is_217_l2273_227389


namespace sum_of_numerator_and_denominator_of_decimal_0_345_l2273_227372

def repeating_decimal_to_fraction_sum (x : ℚ) : ℕ :=
if h : x = 115 / 333 then 115 + 333 else 0

theorem sum_of_numerator_and_denominator_of_decimal_0_345 :
  repeating_decimal_to_fraction_sum 345 / 999 = 448 :=
by {
  -- Given: 0.\overline{345} = 345 / 999, simplified to 115 / 333
  -- hence the sum of numerator and denominator = 115 + 333
  -- We don't need the proof steps here, just conclude with the sum
  sorry }

end sum_of_numerator_and_denominator_of_decimal_0_345_l2273_227372


namespace length_of_first_train_is_correct_l2273_227392

noncomputable def length_of_first_train (speed1_km_hr speed2_km_hr : ℝ) (time_cross_sec : ℝ) (length2_m : ℝ) : ℝ :=
  let speed1_m_s := speed1_km_hr * (5 / 18)
  let speed2_m_s := speed2_km_hr * (5 / 18)
  let relative_speed_m_s := speed1_m_s + speed2_m_s
  let total_distance_m := relative_speed_m_s * time_cross_sec
  total_distance_m - length2_m

theorem length_of_first_train_is_correct : 
  length_of_first_train 60 40 11.879049676025918 160 = 170 := by
  sorry

end length_of_first_train_is_correct_l2273_227392


namespace wage_difference_seven_l2273_227325

-- Define the parameters and conditions
variables (P Q h : ℝ)

-- Given conditions
def condition1 : Prop := P = 1.5 * Q
def condition2 : Prop := P * h = 420
def condition3 : Prop := Q * (h + 10) = 420

-- Theorem to be proved
theorem wage_difference_seven (h : ℝ) (P Q : ℝ) 
  (h_condition1 : condition1 P Q)
  (h_condition2 : condition2 P h)
  (h_condition3 : condition3 Q h) :
  (P - Q) = 7 :=
  sorry

end wage_difference_seven_l2273_227325


namespace find_b_l2273_227342

theorem find_b (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 20)
  (h3 : (5 + 4 * 83 + 6 * 83^2 + 3 * 83^3 + 7 * 83^4 + 5 * 83^5 + 2 * 83^6 - b) % 17 = 0) :
  b = 8 :=
sorry

end find_b_l2273_227342


namespace part1_correct_part2_correct_l2273_227335

-- Definitions for conditions
def total_students := 200
def likes_employment := 140
def dislikes_employment := 60
def p_likes : ℚ := likes_employment / total_students

def male_likes := 60
def male_dislikes := 40
def female_likes := 80
def female_dislikes := 20
def n := total_students
def alpha := 0.005
def chi_squared_critical_value := 7.879

-- Part 1: Estimate the probability of selecting at least 2 students who like employment
def probability_at_least_2_of_3 : ℚ :=
  3 * ((7/10) ^ 2) * (3/10) + ((7/10) ^ 3)

-- Proof goal for Part 1
theorem part1_correct : probability_at_least_2_of_3 = 98 / 125 := by
  sorry

-- Part 2: Chi-squared test for independence between intention and gender
def a := male_likes
def b := male_dislikes
def c := female_likes
def d := female_dislikes
def chi_squared_statistic : ℚ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof goal for Part 2
theorem part2_correct : chi_squared_statistic = 200 / 21 ∧ 200 / 21 > chi_squared_critical_value := by
  sorry

end part1_correct_part2_correct_l2273_227335


namespace plan_A_is_cost_effective_l2273_227322

-- Definitions of the costs considering the problem's conditions
def cost_plan_A (days_A : ℕ) (rate_A : ℕ) : ℕ := days_A * rate_A
def cost_plan_C (days_AB : ℕ) (rate_A : ℕ) (rate_B : ℕ) (remaining_B : ℕ) : ℕ :=
  (days_AB * (rate_A + rate_B)) + (remaining_B * rate_B)

-- Specification of the days and rates from the conditions
def days_A := 12
def rate_A := 10000
def rate_B := 6000
def days_AB := 3
def remaining_B := 13

-- Costs for each plan
def A_cost := cost_plan_A days_A rate_A
def C_cost := cost_plan_C days_AB rate_A rate_B remaining_B

-- Theorem stating that Plan A is more cost-effective
theorem plan_A_is_cost_effective : A_cost < C_cost := by
  unfold A_cost
  unfold C_cost
  sorry

end plan_A_is_cost_effective_l2273_227322


namespace gain_per_year_is_120_l2273_227373

def principal := 6000
def rate_borrow := 4
def rate_lend := 6
def time := 2

def simple_interest (P R T : Nat) : Nat := P * R * T / 100

def interest_earned := simple_interest principal rate_lend time
def interest_paid := simple_interest principal rate_borrow time
def gain_in_2_years := interest_earned - interest_paid
def gain_per_year := gain_in_2_years / 2

theorem gain_per_year_is_120 : gain_per_year = 120 :=
by
  sorry

end gain_per_year_is_120_l2273_227373


namespace mona_cookie_count_l2273_227306

theorem mona_cookie_count {M : ℕ} (h1 : (M - 5) + (M - 5 + 10) + M = 60) : M = 20 :=
by
  sorry

end mona_cookie_count_l2273_227306


namespace jill_arrives_15_minutes_before_jack_l2273_227365

theorem jill_arrives_15_minutes_before_jack
  (distance : ℝ) (jill_speed : ℝ) (jack_speed : ℝ) (start_same_time : true)
  (h_distance : distance = 2) (h_jill_speed : jill_speed = 8) (h_jack_speed : jack_speed = 4) :
  (2 / 4 * 60) - (2 / 8 * 60) = 15 :=
by
  sorry

end jill_arrives_15_minutes_before_jack_l2273_227365


namespace retailer_profit_percentage_l2273_227371

theorem retailer_profit_percentage 
  (CP MP SP : ℝ)
  (hCP : CP = 100)
  (hMP : MP = CP + 0.65 * CP)
  (hSP : SP = MP - 0.25 * MP)
  : ((SP - CP) / CP) * 100 = 23.75 := 
sorry

end retailer_profit_percentage_l2273_227371


namespace store_earnings_correct_l2273_227324

theorem store_earnings_correct :
  let graphics_cards_qty := 10
  let hard_drives_qty := 14
  let cpus_qty := 8
  let rams_qty := 4
  let psus_qty := 12
  let monitors_qty := 6
  let keyboards_qty := 18
  let mice_qty := 24

  let graphics_card_price := 600
  let hard_drive_price := 80
  let cpu_price := 200
  let ram_price := 60
  let psu_price := 90
  let monitor_price := 250
  let keyboard_price := 40
  let mouse_price := 20

  let total_earnings := graphics_cards_qty * graphics_card_price +
                        hard_drives_qty * hard_drive_price +
                        cpus_qty * cpu_price +
                        rams_qty * ram_price +
                        psus_qty * psu_price +
                        monitors_qty * monitor_price +
                        keyboards_qty * keyboard_price +
                        mice_qty * mouse_price
  total_earnings = 12740 :=
by
  -- definitions and calculations here
  sorry

end store_earnings_correct_l2273_227324


namespace inheritance_amount_l2273_227310

theorem inheritance_amount (x : ℝ) (h1 : x * 0.25 + (x * 0.75) * 0.15 + 2500 = 16500) : x = 38621 := 
by
  sorry

end inheritance_amount_l2273_227310


namespace bear_problem_l2273_227380

variables (w b br : ℕ)

theorem bear_problem 
    (h1 : b = 2 * w)
    (h2 : br = b + 40)
    (h3 : w + b + br = 190) :
    b = 60 :=
by
  sorry

end bear_problem_l2273_227380


namespace inequality_proof_l2273_227385

variable (a b c d : ℝ)

theorem inequality_proof (ha : 0 ≤ a ∧ a ≤ 1)
                       (hb : 0 ≤ b ∧ b ≤ 1)
                       (hc : 0 ≤ c ∧ c ≤ 1)
                       (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1) ^ 2 ≥ 4 * (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2) :=
sorry

end inequality_proof_l2273_227385


namespace floor_neg_7_over_4_l2273_227399

theorem floor_neg_7_over_4 : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_7_over_4_l2273_227399
