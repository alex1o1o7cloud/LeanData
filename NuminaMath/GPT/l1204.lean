import Mathlib

namespace NUMINAMATH_GPT_work_problem_l1204_120440

theorem work_problem (x : ℝ) (hx : x > 0)
    (hB : B_work_rate = 1 / 18)
    (hTogether : together_work_rate = 1 / 7.2)
    (hCombined : together_work_rate = 1 / x + B_work_rate) :
    x = 2 := by
    sorry

end NUMINAMATH_GPT_work_problem_l1204_120440


namespace NUMINAMATH_GPT_area_union_of_reflected_triangles_l1204_120416

def point : Type := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def reflect_y_eq_1 (P : point) : point := (P.1, 2 * 1 - P.2)

def area_of_union (A B C : point) (f : point → point) : ℝ :=
  let A' := f A
  let B' := f B
  let C' := f C
  triangle_area A B C + triangle_area A' B' C'

theorem area_union_of_reflected_triangles :
  area_of_union (3, 4) (5, -2) (6, 2) reflect_y_eq_1 = 11 :=
  sorry

end NUMINAMATH_GPT_area_union_of_reflected_triangles_l1204_120416


namespace NUMINAMATH_GPT_initial_amount_is_825_l1204_120482

theorem initial_amount_is_825 (P R : ℝ) 
    (h1 : 956 = P * (1 + 3 * R / 100))
    (h2 : 1055 = P * (1 + 3 * (R + 4) / 100)) : 
    P = 825 := 
by 
  sorry

end NUMINAMATH_GPT_initial_amount_is_825_l1204_120482


namespace NUMINAMATH_GPT_A_3_2_eq_29_l1204_120484

def A : ℕ → ℕ → ℕ
| 0, n     => n + 1
| (m + 1), 0 => A m 1
| (m + 1), (n + 1) => A m (A (m + 1) n)

theorem A_3_2_eq_29 : A 3 2 = 29 := by
  sorry

end NUMINAMATH_GPT_A_3_2_eq_29_l1204_120484


namespace NUMINAMATH_GPT_solve_quadratic_and_linear_equations_l1204_120478

theorem solve_quadratic_and_linear_equations :
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 → x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) ∧
  (∀ x : ℝ, (x + 3) * (x - 3) = 3 * (x + 3) → x = -3 ∨ x = 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_and_linear_equations_l1204_120478


namespace NUMINAMATH_GPT_chef_bought_kilograms_of_almonds_l1204_120443

def total_weight_of_nuts : ℝ := 0.52
def weight_of_pecans : ℝ := 0.38
def weight_of_almonds : ℝ := total_weight_of_nuts - weight_of_pecans

theorem chef_bought_kilograms_of_almonds : weight_of_almonds = 0.14 := by
  sorry

end NUMINAMATH_GPT_chef_bought_kilograms_of_almonds_l1204_120443


namespace NUMINAMATH_GPT_parallel_lines_solution_l1204_120436

theorem parallel_lines_solution (m : ℝ) :
  (∀ x y : ℝ, (x + (1 + m) * y + (m - 2) = 0) → (m * x + 2 * y + 8 = 0)) → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_solution_l1204_120436


namespace NUMINAMATH_GPT_room_breadth_l1204_120428

theorem room_breadth (length height diagonal : ℕ) (h_length : length = 12) (h_height : height = 9) (h_diagonal : diagonal = 17) : 
  ∃ breadth : ℕ, breadth = 8 :=
by
  -- Using the three-dimensional Pythagorean theorem:
  -- d² = length² + breadth² + height²
  -- 17² = 12² + b² + 9²
  -- 289 = 144 + b² + 81
  -- 289 = 225 + b²
  -- b² = 289 - 225
  -- b² = 64
  -- Taking the square root of both sides, we find:
  -- b = √64
  -- b = 8
  let b := 8
  existsi b
  -- This is a skip step, where we assert the breadth equals 8
  sorry

end NUMINAMATH_GPT_room_breadth_l1204_120428


namespace NUMINAMATH_GPT_walter_exceptional_days_l1204_120402

theorem walter_exceptional_days :
  ∃ (w b : ℕ), 
  b + w = 10 ∧ 
  3 * b + 5 * w = 36 ∧ 
  w = 3 :=
by
  sorry

end NUMINAMATH_GPT_walter_exceptional_days_l1204_120402


namespace NUMINAMATH_GPT_hillary_sunday_spend_l1204_120453

noncomputable def spend_per_sunday (total_spent : ℕ) (weeks : ℕ) (weekday_price : ℕ) (weekday_papers : ℕ) : ℕ :=
  (total_spent - weeks * weekday_papers * weekday_price) / weeks

theorem hillary_sunday_spend :
  spend_per_sunday 2800 8 50 3 = 200 :=
sorry

end NUMINAMATH_GPT_hillary_sunday_spend_l1204_120453


namespace NUMINAMATH_GPT_Ramya_reads_total_124_pages_l1204_120469

theorem Ramya_reads_total_124_pages :
  let total_pages : ℕ := 300
  let pages_read_monday := (1/5 : ℚ) * total_pages
  let pages_remaining := total_pages - pages_read_monday
  let pages_read_tuesday := (4/15 : ℚ) * pages_remaining
  pages_read_monday + pages_read_tuesday = 124 := 
by
  sorry

end NUMINAMATH_GPT_Ramya_reads_total_124_pages_l1204_120469


namespace NUMINAMATH_GPT_equal_phrases_impossible_l1204_120475

-- Define the inhabitants and the statements they make.
def inhabitants : ℕ := 1234

-- Define what it means to be a knight or a liar.
inductive Person
| knight : Person
| liar : Person

-- Define the statements "He is a knight!" and "He is a liar!"
inductive Statement
| is_knight : Statement
| is_liar : Statement

-- Define the pairings and types of statements 
def pairings (inhabitant1 inhabitant2 : Person) : Statement :=
match inhabitant1, inhabitant2 with
| Person.knight, Person.knight => Statement.is_knight
| Person.liar, Person.liar => Statement.is_knight
| Person.knight, Person.liar => Statement.is_liar
| Person.liar, Person.knight => Statement.is_knight

-- Define the total number of statements
def total_statements (pairs : ℕ) : ℕ := 2 * pairs

-- Theorem stating the mathematical equivalent proof problem
theorem equal_phrases_impossible :
  ¬ ∃ n : ℕ, n = inhabitants / 2 ∧ total_statements n = inhabitants ∧
    (pairings Person.knight Person.liar = Statement.is_knight ∧
     pairings Person.liar Person.knight = Statement.is_knight ∧
     (pairings Person.knight Person.knight = Statement.is_knight ∧
      pairings Person.liar Person.liar = Statement.is_knight) ∨
      (pairings Person.knight Person.liar = Statement.is_liar ∧
       pairings Person.liar Person.knight = Statement.is_liar)) :=
sorry

end NUMINAMATH_GPT_equal_phrases_impossible_l1204_120475


namespace NUMINAMATH_GPT_no_integers_solution_l1204_120408

theorem no_integers_solution (k : ℕ) (x y z : ℤ) (hx1 : 0 < x) (hx2 : x < k) (hy1 : 0 < y) (hy2 : y < k) (hz : z > 0) :
  x^k + y^k ≠ z^k :=
sorry

end NUMINAMATH_GPT_no_integers_solution_l1204_120408


namespace NUMINAMATH_GPT_smallest_possible_Y_l1204_120473

def digits (n : ℕ) : List ℕ := -- hypothetical function to get the digits of a number
  sorry

def is_divisible (n d : ℕ) : Prop := d ∣ n

theorem smallest_possible_Y :
  ∃ (U : ℕ), (∀ d ∈ digits U, d = 0 ∨ d = 1) ∧ is_divisible U 18 ∧ U / 18 = 61728395 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_Y_l1204_120473


namespace NUMINAMATH_GPT_total_weight_of_nuts_l1204_120420

theorem total_weight_of_nuts (weight_almonds weight_pecans : ℝ) (h1 : weight_almonds = 0.14) (h2 : weight_pecans = 0.38) : weight_almonds + weight_pecans = 0.52 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_nuts_l1204_120420


namespace NUMINAMATH_GPT_total_number_of_balls_l1204_120400

def number_of_yellow_balls : Nat := 6
def probability_yellow_ball : Rat := 1 / 9

theorem total_number_of_balls (N : Nat) (h1 : number_of_yellow_balls = 6) (h2 : probability_yellow_ball = 1 / 9) :
    6 / N = 1 / 9 → N = 54 := 
by
  sorry

end NUMINAMATH_GPT_total_number_of_balls_l1204_120400


namespace NUMINAMATH_GPT_find_value_of_expression_l1204_120465

-- Define non-negative variables
variables (x y z : ℝ) 

-- Conditions
def cond1 := x ^ 2 + x * y + y ^ 2 / 3 = 25
def cond2 := y ^ 2 / 3 + z ^ 2 = 9
def cond3 := z ^ 2 + z * x + x ^ 2 = 16

-- Target statement to be proven
theorem find_value_of_expression (h1 : cond1 x y) (h2 : cond2 y z) (h3 : cond3 z x) : 
  x * y + 2 * y * z + 3 * z * x = 24 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_value_of_expression_l1204_120465


namespace NUMINAMATH_GPT_bode_law_planet_9_l1204_120419

theorem bode_law_planet_9 :
  ∃ (a b : ℝ),
    (a + b = 0.7) ∧ (a + 2 * b = 1) ∧ 
    (70 < a + b * 2^8) ∧ (a + b * 2^8 < 80) :=
by
  -- Define variables and equations based on given conditions
  let a : ℝ := 0.4
  let b : ℝ := 0.3
  
  have h1 : a + b = 0.7 := by 
    sorry  -- Proof that a + b = 0.7
  
  have h2 : a + 2 * b = 1 := by
    sorry  -- Proof that a + 2 * b = 1
  
  have hnine : 70 < a + b * 2^8 ∧ a + b * 2^8 < 80 := by
    -- Calculate a + b * 2^8 and then check the range
    sorry  -- Proof that 70 < a + b * 2^8 < 80

  exact ⟨a, b, h1, h2, hnine⟩

end NUMINAMATH_GPT_bode_law_planet_9_l1204_120419


namespace NUMINAMATH_GPT_quadrangular_pyramid_edge_length_l1204_120429

theorem quadrangular_pyramid_edge_length :
  ∃ e : ℝ, 8 * e = 14.8 ∧ e = 1.85 :=
  sorry

end NUMINAMATH_GPT_quadrangular_pyramid_edge_length_l1204_120429


namespace NUMINAMATH_GPT_aba_div_by_7_l1204_120433

theorem aba_div_by_7 (a b : ℕ) (h : (a + b) % 7 = 0) : (101 * a + 10 * b) % 7 = 0 := 
sorry

end NUMINAMATH_GPT_aba_div_by_7_l1204_120433


namespace NUMINAMATH_GPT_towels_after_a_week_l1204_120406

theorem towels_after_a_week 
  (initial_green : ℕ) (initial_white : ℕ) (initial_blue : ℕ) 
  (daily_green : ℕ) (daily_white : ℕ) (daily_blue : ℕ) 
  (days : ℕ) 
  (H1 : initial_green = 35)
  (H2 : initial_white = 21)
  (H3 : initial_blue = 15)
  (H4 : daily_green = 3)
  (H5 : daily_white = 1)
  (H6 : daily_blue = 1)
  (H7 : days = 7) :
  (initial_green - daily_green * days) + (initial_white - daily_white * days) + (initial_blue - daily_blue * days) = 36 :=
by 
  sorry

end NUMINAMATH_GPT_towels_after_a_week_l1204_120406


namespace NUMINAMATH_GPT_range_of_y0_l1204_120483

theorem range_of_y0
  (y0 : ℝ)
  (h_tangent : ∃ N : ℝ × ℝ, (N.1^2 + N.2^2 = 1) ∧ ((↑(Real.sqrt 3 - N.1)^2 + (y0 - N.2)^2) = 1))
  (h_angle : ∀ N : ℝ × ℝ, (N.1^2 + N.2^2 = 1) ∧ ((↑(Real.sqrt 3 - N.1)^2 + (y0 - N.2)^2 = 1)) → (Real.arccos ((Real.sqrt 3 - N.1)/Real.sqrt ((3 - 2 * N.1 * Real.sqrt 3 + N.1^2) + (y0 - N.2)^2)) ≥ π / 6)) :
  -1 ≤ y0 ∧ y0 ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_y0_l1204_120483


namespace NUMINAMATH_GPT_ratio_of_mustang_models_length_l1204_120461

theorem ratio_of_mustang_models_length :
  ∀ (full_size_length mid_size_length smallest_model_length : ℕ),
    full_size_length = 240 →
    mid_size_length = full_size_length / 10 →
    smallest_model_length = 12 →
    smallest_model_length / mid_size_length = 1/2 :=
by
  intros full_size_length mid_size_length smallest_model_length h1 h2 h3
  sorry

end NUMINAMATH_GPT_ratio_of_mustang_models_length_l1204_120461


namespace NUMINAMATH_GPT_ones_digit_of_prime_p_l1204_120442

theorem ones_digit_of_prime_p (p q r s : ℕ) (hp : p > 5) (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q) (prime_r : Nat.Prime r) (prime_s : Nat.Prime s)
  (hseq1 : q = p + 8) (hseq2 : r = p + 16) (hseq3 : s = p + 24) 
  : p % 10 = 3 := 
sorry

end NUMINAMATH_GPT_ones_digit_of_prime_p_l1204_120442


namespace NUMINAMATH_GPT_distance_from_origin_12_5_l1204_120458

def distance_from_origin (x y : ℕ) : ℕ := 
  Int.natAbs (Nat.sqrt (x * x + y * y))

theorem distance_from_origin_12_5 : distance_from_origin 12 5 = 13 := by
  sorry

end NUMINAMATH_GPT_distance_from_origin_12_5_l1204_120458


namespace NUMINAMATH_GPT_n_cubed_plus_5n_divisible_by_6_l1204_120454

theorem n_cubed_plus_5n_divisible_by_6 (n : ℕ) : ∃ k : ℤ, n^3 + 5 * n = 6 * k :=
by
  sorry

end NUMINAMATH_GPT_n_cubed_plus_5n_divisible_by_6_l1204_120454


namespace NUMINAMATH_GPT_julia_baking_days_l1204_120485

variable (bakes_per_day : ℕ)
variable (clifford_eats_per_two_days : ℕ)
variable (final_cakes : ℕ)

def number_of_baking_days : ℕ :=
  2 * (final_cakes / (bakes_per_day * 2 - clifford_eats_per_two_days))

theorem julia_baking_days (h1 : bakes_per_day = 4)
                        (h2 : clifford_eats_per_two_days = 1)
                        (h3 : final_cakes = 21) :
  number_of_baking_days bakes_per_day clifford_eats_per_two_days final_cakes = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_julia_baking_days_l1204_120485


namespace NUMINAMATH_GPT_no_real_solution_for_pairs_l1204_120449

theorem no_real_solution_for_pairs (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬ (1 / a + 1 / b = 1 / (a + b)) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_for_pairs_l1204_120449


namespace NUMINAMATH_GPT_doug_lost_marbles_l1204_120489

-- Definitions based on the conditions
variables (D D' : ℕ) -- D is the number of marbles Doug originally had, D' is the number Doug has now

-- Condition 1: Ed had 10 more marbles than Doug originally.
def ed_marble_initial (D : ℕ) : ℕ := D + 10

-- Condition 2: Ed had 45 marbles originally.
axiom ed_initial_marble_count : ed_marble_initial D = 45

-- Solve for D from condition 2
noncomputable def doug_initial_marble_count : ℕ := 45 - 10

-- Condition 3: Ed now has 21 more marbles than Doug.
axiom ed_current_marble_difference : 45 = D' + 21

-- Translate what we need to prove
theorem doug_lost_marbles : (doug_initial_marble_count - D') = 11 :=
by
    -- Insert math proof steps here
    sorry

end NUMINAMATH_GPT_doug_lost_marbles_l1204_120489


namespace NUMINAMATH_GPT_sum_of_values_of_x_l1204_120422

noncomputable def g (x : ℝ) : ℝ :=
if x < 3 then 7 * x + 10 else 3 * x - 18

theorem sum_of_values_of_x (h : ∃ x : ℝ, g x = 5) :
  (∃ x1 x2 : ℝ, g x1 = 5 ∧ g x2 = 5) → (x1 + x2 = 18 / 7) :=
sorry

end NUMINAMATH_GPT_sum_of_values_of_x_l1204_120422


namespace NUMINAMATH_GPT_domain_range_of_p_l1204_120450

variable (h : ℝ → ℝ)
variable (h_domain : ∀ x, -1 ≤ x ∧ x ≤ 3)
variable (h_range : ∀ x, 0 ≤ h x ∧ h x ≤ 2)

def p (x : ℝ) : ℝ := 2 - h (x - 1)

theorem domain_range_of_p :
  (∀ x, 0 ≤ x ∧ x ≤ 4) ∧ (∀ y, 0 ≤ y ∧ y ≤ 2) :=
by
  -- Proof to show that the domain of p(x) is [0, 4] and the range is [0, 2]
  sorry

end NUMINAMATH_GPT_domain_range_of_p_l1204_120450


namespace NUMINAMATH_GPT_total_frisbees_l1204_120438

-- Let x be the number of $3 frisbees and y be the number of $4 frisbees.
variables (x y : ℕ)

-- Condition 1: Total sales amount is 200 dollars.
def condition1 : Prop := 3 * x + 4 * y = 200

-- Condition 2: At least 8 $4 frisbees were sold.
def condition2 : Prop := y >= 8

-- Prove that the total number of frisbees sold is 64.
theorem total_frisbees (h1 : condition1 x y) (h2 : condition2 y) : x + y = 64 :=
by
  sorry

end NUMINAMATH_GPT_total_frisbees_l1204_120438


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1204_120434

def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x ^ 2

theorem necessary_but_not_sufficient_condition :
  (∀ x, q x → p x) ∧ (¬ ∀ x, p x → q x) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1204_120434


namespace NUMINAMATH_GPT_part1_l1204_120411

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end NUMINAMATH_GPT_part1_l1204_120411


namespace NUMINAMATH_GPT_area_of_park_l1204_120487

theorem area_of_park (L B : ℝ) (h1 : L / B = 1 / 3) (h2 : 12 * 1000 / 60 * 4 = 2 * (L + B)) : 
  L * B = 30000 :=
by
  sorry

end NUMINAMATH_GPT_area_of_park_l1204_120487


namespace NUMINAMATH_GPT_response_rate_increase_l1204_120405

theorem response_rate_increase :
  let original_customers := 70
  let original_responses := 7
  let redesigned_customers := 63
  let redesigned_responses := 9
  let original_response_rate := (original_responses : ℝ) / original_customers
  let redesigned_response_rate := (redesigned_responses : ℝ) / redesigned_customers
  let percentage_increase := ((redesigned_response_rate - original_response_rate) / original_response_rate) * 100
  abs (percentage_increase - 42.86) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_response_rate_increase_l1204_120405


namespace NUMINAMATH_GPT_roots_reciprocal_sum_l1204_120459

theorem roots_reciprocal_sum
  {a b c : ℂ}
  (h_roots : ∀ x : ℂ, (x - a) * (x - b) * (x - c) = x^3 - x + 1) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_roots_reciprocal_sum_l1204_120459


namespace NUMINAMATH_GPT_find_N_l1204_120414

theorem find_N (N : ℕ) (hN : N > 1) (h1 : 2019 ≡ 1743 [MOD N]) (h2 : 3008 ≡ 2019 [MOD N]) : N = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l1204_120414


namespace NUMINAMATH_GPT_solution_interval_l1204_120448

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem solution_interval :
  ∃ x_0, f x_0 = 0 ∧ 2 < x_0 ∧ x_0 < 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_interval_l1204_120448


namespace NUMINAMATH_GPT_roots_eq_202_l1204_120466

theorem roots_eq_202 (p q : ℝ) 
  (h1 : ∀ x : ℝ, ((x + p) * (x + q) * (x + 10) = 0 ↔ (x = -p ∨ x = -q ∨ x = -10)) ∧ 
       ∀ x : ℝ, ((x + 5) ^ 2 = 0 ↔ x = -5)) 
  (h2 : ∀ x : ℝ, ((x + 2 * p) * (x + 4) * (x + 8) = 0 ↔ (x = -2 * p ∨ x = -4 ∨ x = -8)) ∧ 
       ∀ x : ℝ, ((x + q) * (x + 10) = 0 ↔ (x = -q ∨ x = -10))) 
  (hpq : p = q) (neq_5 : q ≠ 5) (p_2 : p = 2):
  100 * p + q = 202 := sorry

end NUMINAMATH_GPT_roots_eq_202_l1204_120466


namespace NUMINAMATH_GPT_paula_remaining_money_l1204_120441

-- Define the given conditions
def given_amount : ℕ := 109
def cost_shirt : ℕ := 11
def number_shirts : ℕ := 2
def cost_pants : ℕ := 13

-- Calculate total spending
def total_spent : ℕ := (cost_shirt * number_shirts) + cost_pants

-- Define the remaining amount Paula has
def remaining_amount : ℕ := given_amount - total_spent

-- State the theorem
theorem paula_remaining_money : remaining_amount = 74 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_paula_remaining_money_l1204_120441


namespace NUMINAMATH_GPT_det_B_squared_minus_3IB_l1204_120426

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 4], ![3, 1]]
def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem det_B_squared_minus_3IB :
  det (B * B - 3 * I * B) = 100 := by
  sorry

end NUMINAMATH_GPT_det_B_squared_minus_3IB_l1204_120426


namespace NUMINAMATH_GPT_manager_salary_l1204_120467

theorem manager_salary 
    (avg_salary_18 : ℕ)
    (new_avg_salary : ℕ)
    (num_employees : ℕ)
    (num_employees_with_manager : ℕ)
    (old_total_salary : ℕ := num_employees * avg_salary_18)
    (new_total_salary : ℕ := num_employees_with_manager * new_avg_salary) :
    (new_avg_salary = avg_salary_18 + 200) →
    (old_total_salary = 18 * 2000) →
    (new_total_salary = 19 * (2000 + 200)) →
    new_total_salary - old_total_salary = 5800 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_manager_salary_l1204_120467


namespace NUMINAMATH_GPT_intersection_points_zero_l1204_120432

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem intersection_points_zero
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h_gp : geometric_sequence a b c)
  (h_ac_pos : a * c > 0) :
  ∃ x : ℝ, quadratic_function a b c x = 0 → false :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_intersection_points_zero_l1204_120432


namespace NUMINAMATH_GPT_derivative_of_y_l1204_120462

noncomputable def y (x : ℝ) : ℝ := (Real.log x) / x + x * Real.exp x

theorem derivative_of_y (x : ℝ) (hx : x > 0) : 
  deriv y x = (1 - Real.log x) / (x^2) + (x + 1) * Real.exp x := by
  sorry

end NUMINAMATH_GPT_derivative_of_y_l1204_120462


namespace NUMINAMATH_GPT_income_increase_correct_l1204_120463

noncomputable def income_increase_percentage (I1 : ℝ) (S1 : ℝ) (E1 : ℝ) (I2 : ℝ) (S2 : ℝ) (E2 : ℝ) (P : ℝ) :=
  S1 = 0.5 * I1 ∧
  S2 = 2 * S1 ∧
  E1 = 0.5 * I1 ∧
  E2 = I2 - S2 ∧
  I2 = I1 * (1 + P / 100) ∧
  E1 + E2 = 2 * E1

theorem income_increase_correct (I1 : ℝ) (S1 : ℝ) (E1 : ℝ) (I2 : ℝ) (S2 : ℝ) (E2 : ℝ) (P : ℝ)
  (h1 : income_increase_percentage I1 S1 E1 I2 S2 E2 P) : P = 50 :=
sorry

end NUMINAMATH_GPT_income_increase_correct_l1204_120463


namespace NUMINAMATH_GPT_find_cost_of_books_l1204_120425

theorem find_cost_of_books
  (C_L C_G1 C_G2 : ℝ)
  (h1 : C_L + C_G1 + C_G2 = 1080)
  (h2 : 0.9 * C_L = 1.15 * C_G1 + 1.25 * C_G2)
  (h3 : C_G1 + C_G2 = 1080 - C_L) :
  C_L = 784 :=
sorry

end NUMINAMATH_GPT_find_cost_of_books_l1204_120425


namespace NUMINAMATH_GPT_spend_on_candy_l1204_120451

variable (initial_money spent_on_oranges spent_on_apples remaining_money spent_on_candy : ℕ)

-- Conditions
axiom initial_amount : initial_money = 95
axiom spent_on_oranges_value : spent_on_oranges = 14
axiom spent_on_apples_value : spent_on_apples = 25
axiom remaining_amount : remaining_money = 50

-- Question as a theorem
theorem spend_on_candy :
  spent_on_candy = initial_money - (spent_on_oranges + spent_on_apples) - remaining_money :=
by sorry

end NUMINAMATH_GPT_spend_on_candy_l1204_120451


namespace NUMINAMATH_GPT_determine_y_increase_volume_l1204_120444

noncomputable def volume_increase_y (r h y : ℝ) : Prop :=
  (1/3) * Real.pi * (r + y)^2 * h = (1/3) * Real.pi * r^2 * (h + y)

theorem determine_y_increase_volume (y : ℝ) :
  volume_increase_y 5 12 y ↔ y = 31 / 12 :=
by
  sorry

end NUMINAMATH_GPT_determine_y_increase_volume_l1204_120444


namespace NUMINAMATH_GPT_exponential_equality_l1204_120472

theorem exponential_equality (n : ℕ) (h : 4 ^ n = 64 ^ 2) : n = 6 :=
  sorry

end NUMINAMATH_GPT_exponential_equality_l1204_120472


namespace NUMINAMATH_GPT_tourists_went_free_l1204_120412

theorem tourists_went_free (x : ℕ) : 
  (13 + 4 * x = x + 100) → x = 29 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_tourists_went_free_l1204_120412


namespace NUMINAMATH_GPT_combined_weight_l1204_120494

theorem combined_weight (x y z : ℕ) (h1 : x + y = 110) (h2 : y + z = 130) (h3 : z + x = 150) : x + y + z = 195 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_l1204_120494


namespace NUMINAMATH_GPT_sin_cos_ratio_l1204_120410

theorem sin_cos_ratio (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2)
  (h2 : Real.tan (α - β) = 3) : 
  Real.sin (2 * α) / Real.cos (2 * β) = (Real.sqrt 5 + 3 * Real.sqrt 2) / 20 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_ratio_l1204_120410


namespace NUMINAMATH_GPT_FG_square_l1204_120497

def trapezoid_EFGH (EF FG GH EH : ℝ) : Prop :=
  ∃ x y : ℝ, 
  EF = 4 ∧
  EH = 31 ∧
  FG = x ∧
  GH = y ∧
  x^2 + (y - 4)^2 = 961 ∧
  x^2 = 4 * y

theorem FG_square (EF EH FG GH x y : ℝ) (h : trapezoid_EFGH EF FG GH EH) :
  FG^2 = 132 :=
by
  obtain ⟨x, y, h1, h2, h3, h4, h5, h6⟩ := h
  exact sorry

end NUMINAMATH_GPT_FG_square_l1204_120497


namespace NUMINAMATH_GPT_scouts_earnings_over_weekend_l1204_120430

def base_pay_per_hour : ℝ := 10.00
def tip_per_customer : ℝ := 5.00
def hours_worked_saturday : ℝ := 4.0
def customers_served_saturday : ℝ := 5.0
def hours_worked_sunday : ℝ := 5.0
def customers_served_sunday : ℝ := 8.0

def earnings_saturday : ℝ := (hours_worked_saturday * base_pay_per_hour) + (customers_served_saturday * tip_per_customer)
def earnings_sunday : ℝ := (hours_worked_sunday * base_pay_per_hour) + (customers_served_sunday * tip_per_customer)

def total_earnings : ℝ := earnings_saturday + earnings_sunday

theorem scouts_earnings_over_weekend : total_earnings = 155.00 := by
  sorry

end NUMINAMATH_GPT_scouts_earnings_over_weekend_l1204_120430


namespace NUMINAMATH_GPT_formula1_correct_formula2_correct_formula3_correct_l1204_120417

noncomputable def formula1 (n : ℕ) := (Real.sqrt 2 / 2) * (1 - (-1 : ℝ) ^ n)
noncomputable def formula2 (n : ℕ) := Real.sqrt (1 - (-1 : ℝ) ^ n)
noncomputable def formula3 (n : ℕ) := if (n % 2 = 1) then Real.sqrt 2 else 0

theorem formula1_correct (n : ℕ) : 
  (n % 2 = 1 → formula1 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula1 n = 0) := 
by
  sorry

theorem formula2_correct (n : ℕ) : 
  (n % 2 = 1 → formula2 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula2 n = 0) := 
by
  sorry
  
theorem formula3_correct (n : ℕ) : 
  (n % 2 = 1 → formula3 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula3 n = 0) := 
by
  sorry

end NUMINAMATH_GPT_formula1_correct_formula2_correct_formula3_correct_l1204_120417


namespace NUMINAMATH_GPT_carter_cheesecakes_l1204_120488

theorem carter_cheesecakes (C : ℕ) (nm : ℕ) (nr : ℕ) (increase : ℕ) (this_week_cakes : ℕ) (usual_cakes : ℕ) :
  nm = 5 → nr = 8 → increase = 38 → 
  this_week_cakes = 3 * C + 3 * nm + 3 * nr → 
  usual_cakes = C + nm + nr → 
  this_week_cakes = usual_cakes + increase → 
  C = 6 :=
by
  intros hnm hnr hinc htw husual hcakes
  sorry

end NUMINAMATH_GPT_carter_cheesecakes_l1204_120488


namespace NUMINAMATH_GPT_crackers_count_l1204_120480

theorem crackers_count (crackers_Marcus crackers_Mona crackers_Nicholas : ℕ) 
  (h1 : crackers_Marcus = 3 * crackers_Mona)
  (h2 : crackers_Nicholas = crackers_Mona + 6)
  (h3 : crackers_Marcus = 27) : crackers_Nicholas = 15 := 
by 
  sorry

end NUMINAMATH_GPT_crackers_count_l1204_120480


namespace NUMINAMATH_GPT_quadratic_transformed_correct_l1204_120495

noncomputable def quadratic_transformed (a b c : ℝ) (r s : ℝ) (h1 : a ≠ 0) 
  (h_roots : r + s = -b / a ∧ r * s = c / a) : Polynomial ℝ :=
Polynomial.C (a * b * c) + Polynomial.C ((-(a + b) * b)) * Polynomial.X + Polynomial.X^2

-- The theorem statement
theorem quadratic_transformed_correct (a b c r s : ℝ) (h1 : a ≠ 0)
  (h_roots : r + s = -b / a ∧ r * s = c / a) :
  (quadratic_transformed a b c r s h1 h_roots).roots = {a * (r + b), a * (s + b)} :=
sorry

end NUMINAMATH_GPT_quadratic_transformed_correct_l1204_120495


namespace NUMINAMATH_GPT_new_line_length_l1204_120427

/-- Eli drew a line that was 1.5 meters long and then erased 37.5 centimeters of it.
    We need to prove that the length of the line now is 112.5 centimeters. -/
theorem new_line_length (initial_length_m : ℝ) (erased_length_cm : ℝ) 
    (h1 : initial_length_m = 1.5) (h2 : erased_length_cm = 37.5) :
    initial_length_m * 100 - erased_length_cm = 112.5 :=
by
  sorry

end NUMINAMATH_GPT_new_line_length_l1204_120427


namespace NUMINAMATH_GPT_delta_discount_percentage_l1204_120447

theorem delta_discount_percentage (original_delta : ℝ) (original_united : ℝ)
  (united_discount_percent : ℝ) (savings : ℝ) (delta_discounted : ℝ) : 
  original_delta - delta_discounted = 0.2 * original_delta := by
  -- Given conditions
  let discounted_united := original_united * (1 - united_discount_percent / 100)
  have : delta_discounted = discounted_united - savings := sorry
  let delta_discount_amount := original_delta - delta_discounted
  have : delta_discount_amount = 0.2 * original_delta := sorry
  exact this

end NUMINAMATH_GPT_delta_discount_percentage_l1204_120447


namespace NUMINAMATH_GPT_number_of_ways_to_assign_roles_l1204_120490

theorem number_of_ways_to_assign_roles : 
  let male_roles := 3
  let female_roles := 2
  let either_gender_roles := 1
  let men := 4
  let women := 5
  let total_roles := male_roles + female_roles + either_gender_roles
  let ways_to_assign_males := men * (men-1) * (men-2)
  let ways_to_assign_females := women * (women-1)
  let remaining_actors := men + women - male_roles - female_roles
  let ways_to_assign_either_gender := remaining_actors
  let total_ways := ways_to_assign_males * ways_to_assign_females * ways_to_assign_either_gender

  total_ways = 1920 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_assign_roles_l1204_120490


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l1204_120499

-- Definitions derived directly from the conditions
def first_term : ℕ := 2
def common_difference : ℕ := 4
def last_term : ℕ := 2010

-- Lean statement for the proof problem
theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, last_term = first_term + (n - 1) * common_difference ∧ n = 503 :=
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l1204_120499


namespace NUMINAMATH_GPT_parabola_ratio_l1204_120401

noncomputable def ratio_AF_BF (p : ℝ) (h_pos : p > 0) : ℝ :=
  let y1 := (Real.sqrt (2 * p * (3 / 2 * p)))
  let y2 := (Real.sqrt (2 * p * (1 / 6 * p)))
  let dist1 := Real.sqrt ((3 / 2 * p - (p / 2))^2 + y1^2)
  let dist2 := Real.sqrt ((1 / 6 * p - p / 2)^2 + y2^2)
  dist1 / dist2

theorem parabola_ratio (p : ℝ) (h_pos : p > 0) : ratio_AF_BF p h_pos = 3 :=
  sorry

end NUMINAMATH_GPT_parabola_ratio_l1204_120401


namespace NUMINAMATH_GPT_find_other_number_l1204_120479

theorem find_other_number
  (B : ℕ)
  (hcf_condition : Nat.gcd 24 B = 12)
  (lcm_condition : Nat.lcm 24 B = 396) :
  B = 198 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l1204_120479


namespace NUMINAMATH_GPT_perpendicular_lines_l1204_120424

theorem perpendicular_lines :
  (∀ (x y : ℝ), (4 * y - 3 * x = 16)) ∧ 
  (∀ (x y : ℝ), (3 * y + 4 * x = 15)) → 
  (∃ (m1 m2 : ℝ), m1 * m2 = -1) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l1204_120424


namespace NUMINAMATH_GPT_robotics_club_neither_l1204_120498

theorem robotics_club_neither (n c e b neither : ℕ) (h1 : n = 80) (h2 : c = 50) (h3 : e = 40) (h4 : b = 25) :
  neither = n - (c - b + e - b + b) :=
by 
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_robotics_club_neither_l1204_120498


namespace NUMINAMATH_GPT_tangent_line_eq_bounded_area_l1204_120474

-- Given two parabolas and a tangent line, and a positive constant a
variables (a : ℝ)
variables (y1 y2 l : ℝ → ℝ)

-- Conditions:
def parabola1 := ∀ (x : ℝ), y1 x = x^2 + a * x
def parabola2 := ∀ (x : ℝ), y2 x = x^2 - 2 * a * x
def tangent_line := ∀ (x : ℝ), l x = - (a / 2) * x - (9 * a^2 / 16)
def a_positive := a > 0

-- Proof goals:
theorem tangent_line_eq : 
  parabola1 a y1 ∧ parabola2 a y2 ∧ tangent_line a l ∧ a_positive a 
  → ∀ x, (y1 x = l x ∨ y2 x = l x) :=
sorry

theorem bounded_area : 
  parabola1 a y1 ∧ parabola2 a y2 ∧ tangent_line a l ∧ a_positive a 
  → ∫ (x : ℝ) in (-3 * a / 4)..(3 * a / 4), (y1 x - l x) + (y2 x - l x) = 9 * a^3 / 8 :=
sorry

end NUMINAMATH_GPT_tangent_line_eq_bounded_area_l1204_120474


namespace NUMINAMATH_GPT_final_price_of_pencil_l1204_120455

-- Define the initial constants
def initialCost : ℝ := 4.00
def christmasDiscount : ℝ := 0.63
def seasonalDiscountRate : ℝ := 0.07
def finalDiscountRate : ℝ := 0.05
def taxRate : ℝ := 0.065

-- Define the steps of the problem concisely
def priceAfterChristmasDiscount := initialCost - christmasDiscount
def priceAfterSeasonalDiscount := priceAfterChristmasDiscount * (1 - seasonalDiscountRate)
def priceAfterFinalDiscount := priceAfterSeasonalDiscount * (1 - finalDiscountRate)
def finalPrice := priceAfterFinalDiscount * (1 + taxRate)

-- The theorem to be proven
theorem final_price_of_pencil :
  abs (finalPrice - 3.17) < 0.01 := by
  sorry

end NUMINAMATH_GPT_final_price_of_pencil_l1204_120455


namespace NUMINAMATH_GPT_boxes_needed_to_pack_all_muffins_l1204_120403

theorem boxes_needed_to_pack_all_muffins
  (total_muffins : ℕ := 95)
  (muffins_per_box : ℕ := 5)
  (available_boxes : ℕ := 10) :
  (total_muffins / muffins_per_box) - available_boxes = 9 :=
by
  sorry

end NUMINAMATH_GPT_boxes_needed_to_pack_all_muffins_l1204_120403


namespace NUMINAMATH_GPT_seashells_given_to_Jessica_l1204_120404

-- Define the initial number of seashells Dan had
def initialSeashells : ℕ := 56

-- Define the number of seashells Dan has left
def seashellsLeft : ℕ := 22

-- Define the number of seashells Dan gave to Jessica
def seashellsGiven : ℕ := initialSeashells - seashellsLeft

-- State the theorem to prove
theorem seashells_given_to_Jessica :
  seashellsGiven = 34 :=
by
  -- Begin the proof here
  sorry

end NUMINAMATH_GPT_seashells_given_to_Jessica_l1204_120404


namespace NUMINAMATH_GPT_hexagon_angle_sum_l1204_120435

theorem hexagon_angle_sum (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) :
  a1 + a2 + a3 + a4 = 360 ∧ b1 + b2 + b3 + b4 = 360 → 
  a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 = 720 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_angle_sum_l1204_120435


namespace NUMINAMATH_GPT_quadratic_to_general_form_l1204_120457

theorem quadratic_to_general_form (x : ℝ) :
  ∃ b : ℝ, (∀ a c : ℝ, (a = 3) ∧ (c = 1) → (a * x^2 + c = 6 * x) → b = -6) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_to_general_form_l1204_120457


namespace NUMINAMATH_GPT_grading_ratio_l1204_120415

noncomputable def num_questions : ℕ := 100
noncomputable def correct_answers : ℕ := 91
noncomputable def score_received : ℕ := 73
noncomputable def incorrect_answers : ℕ := num_questions - correct_answers
noncomputable def total_points_subtracted : ℕ := correct_answers - score_received
noncomputable def points_per_incorrect : ℚ := total_points_subtracted / incorrect_answers

theorem grading_ratio (h: (points_per_incorrect : ℚ) = 2) :
  2 / 1 = points_per_incorrect / 1 :=
by sorry

end NUMINAMATH_GPT_grading_ratio_l1204_120415


namespace NUMINAMATH_GPT_probability_green_jelly_bean_l1204_120496

theorem probability_green_jelly_bean :
  let red := 10
  let green := 9
  let yellow := 5
  let blue := 7
  let total := red + green + yellow + blue
  (green : ℚ) / (total : ℚ) = 9 / 31 := by
  sorry

end NUMINAMATH_GPT_probability_green_jelly_bean_l1204_120496


namespace NUMINAMATH_GPT_cylinder_ratio_l1204_120437

theorem cylinder_ratio
  (V : ℝ) (r h : ℝ)
  (h_volume : π * r^2 * h = V)
  (h_surface_area : 2 * π * r * h = 2 * (V / r)) :
  h / r = 2 :=
sorry

end NUMINAMATH_GPT_cylinder_ratio_l1204_120437


namespace NUMINAMATH_GPT_total_games_single_elimination_l1204_120468

theorem total_games_single_elimination (teams : ℕ) (h_teams : teams = 24)
  (preliminary_matches : ℕ) (h_preliminary_matches : preliminary_matches = 8)
  (preliminary_teams : ℕ) (h_preliminary_teams : preliminary_teams = 16)
  (idle_teams : ℕ) (h_idle_teams : idle_teams = 8)
  (main_draw_teams : ℕ) (h_main_draw_teams : main_draw_teams = 16) :
  (games : ℕ) -> games = 23 :=
by
  sorry

end NUMINAMATH_GPT_total_games_single_elimination_l1204_120468


namespace NUMINAMATH_GPT_abs_expression_not_positive_l1204_120439

theorem abs_expression_not_positive (x : ℝ) (h : |2 * x - 7| = 0) : x = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_expression_not_positive_l1204_120439


namespace NUMINAMATH_GPT_sum_of_products_l1204_120471

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62)
  (h2 : a + b + c = 18) : 
  a * b + b * c + c * a = 131 :=
sorry

end NUMINAMATH_GPT_sum_of_products_l1204_120471


namespace NUMINAMATH_GPT_sum_of_possible_two_digit_values_l1204_120456

theorem sum_of_possible_two_digit_values (d : ℕ) (h1 : 0 < d) (h2 : d < 100) (h3 : 137 % d = 6) : d = 131 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_two_digit_values_l1204_120456


namespace NUMINAMATH_GPT_fraction_budget_paid_l1204_120452

variable (B : ℝ) (b k : ℝ)

-- Conditions
def condition1 : b = 0.30 * (B - k) := by sorry
def condition2 : k = 0.10 * (B - b) := by sorry

-- Proof that Jenny paid 35% of her budget for her book and snack
theorem fraction_budget_paid :
  b + k = 0.35 * B :=
by
  -- use condition1 and condition2 to prove the theorem
  sorry

end NUMINAMATH_GPT_fraction_budget_paid_l1204_120452


namespace NUMINAMATH_GPT_jenna_remaining_money_l1204_120460

theorem jenna_remaining_money (m c : ℝ) (h : (1 / 4) * m = (1 / 2) * c) : (m - c) / m = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_jenna_remaining_money_l1204_120460


namespace NUMINAMATH_GPT_complex_purely_imaginary_l1204_120446

theorem complex_purely_imaginary (m : ℝ) :
  (m^2 - 3*m + 2 = 0) ∧ (m^2 - 2*m ≠ 0) → m = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_complex_purely_imaginary_l1204_120446


namespace NUMINAMATH_GPT_find_f2_l1204_120493

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + a^(-x)

theorem find_f2 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 3) : f a 2 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_find_f2_l1204_120493


namespace NUMINAMATH_GPT_sin_theta_plus_2pi_div_3_cos_theta_minus_5pi_div_6_l1204_120481

variable (θ : ℝ)

theorem sin_theta_plus_2pi_div_3 (h : Real.sin (θ - Real.pi / 3) = 1 / 3) :
  Real.sin (θ + 2 * Real.pi / 3) = -1 / 3 :=
  sorry

theorem cos_theta_minus_5pi_div_6 (h : Real.sin (θ - Real.pi / 3) = 1 / 3) :
  Real.cos (θ - 5 * Real.pi / 6) = 1 / 3 :=
  sorry

end NUMINAMATH_GPT_sin_theta_plus_2pi_div_3_cos_theta_minus_5pi_div_6_l1204_120481


namespace NUMINAMATH_GPT_range_of_m_l1204_120491

def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (m : ℝ) := {x : ℝ | x^2 - 2*x + m = 0}

theorem range_of_m (m : ℝ) : (A ∪ B m = A) ↔ m ∈ Set.Ici 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1204_120491


namespace NUMINAMATH_GPT_incorrect_expression_l1204_120423

theorem incorrect_expression (x y : ℝ) (h : x > y) : ¬ (3 - x > 3 - y) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_expression_l1204_120423


namespace NUMINAMATH_GPT_example_problem_l1204_120407

theorem example_problem (a b : ℕ) : a = 1 → a * (a + b) + 1 ∣ (a + b) * (b + 1) - 1 :=
by
  sorry

end NUMINAMATH_GPT_example_problem_l1204_120407


namespace NUMINAMATH_GPT_present_age_of_son_l1204_120431

theorem present_age_of_son (S F : ℕ) (h1 : F = S + 22) (h2 : F + 2 = 2 * (S + 2)) : S = 20 :=
by
  sorry

end NUMINAMATH_GPT_present_age_of_son_l1204_120431


namespace NUMINAMATH_GPT_average_first_set_eq_3_more_than_second_set_l1204_120413

theorem average_first_set_eq_3_more_than_second_set (x : ℤ) :
  let avg_first_set := (14 + 32 + 53) / 3
  let avg_second_set := (x + 47 + 22) / 3
  avg_first_set = avg_second_set + 3 → x = 21 := by
  sorry

end NUMINAMATH_GPT_average_first_set_eq_3_more_than_second_set_l1204_120413


namespace NUMINAMATH_GPT_infinite_n_exists_r_s_t_l1204_120464

noncomputable def a (n : ℕ) : ℝ := n^(1/3 : ℝ)
noncomputable def b (n : ℕ) : ℝ := 1 / (a n - ⌊a n⌋)
noncomputable def c (n : ℕ) : ℝ := 1 / (b n - ⌊b n⌋)

theorem infinite_n_exists_r_s_t :
  ∃ (n : ℕ) (r s t : ℤ), (0 < n ∧ ¬∃ k : ℕ, n = k^3) ∧ (¬(r = 0 ∧ s = 0 ∧ t = 0)) ∧ (r * a n + s * b n + t * c n = 0) :=
sorry

end NUMINAMATH_GPT_infinite_n_exists_r_s_t_l1204_120464


namespace NUMINAMATH_GPT_find_a10_l1204_120477

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def a2_eq_4 (a : ℕ → ℝ) := a 2 = 4

def a6_eq_6 (a : ℕ → ℝ) := a 6 = 6

theorem find_a10 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h2 : a2_eq_4 a) (h6 : a6_eq_6 a) : 
  a 10 = 9 :=
sorry

end NUMINAMATH_GPT_find_a10_l1204_120477


namespace NUMINAMATH_GPT_find_x_l1204_120470

theorem find_x (x : ℝ) (a : ℝ × ℝ := (1, 2)) (b : ℝ × ℝ := (x, 1)) :
  ((2 * a.fst - x, 2 * a.snd + 1) • b = 0) → x = -1 ∨ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1204_120470


namespace NUMINAMATH_GPT_common_difference_arithmetic_geometric_sequence_l1204_120418

theorem common_difference_arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_geom : ∃ r, ∀ n, a (n+1) = a n * r)
  (h_a1 : a 1 = 1) :
  d = 0 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_geometric_sequence_l1204_120418


namespace NUMINAMATH_GPT_number_of_boxes_on_pallet_l1204_120409

-- Define the total weight of the pallet.
def total_weight_of_pallet : ℤ := 267

-- Define the weight of each box.
def weight_of_each_box : ℤ := 89

-- The theorem states that given the total weight of the pallet and the weight of each box,
-- the number of boxes on the pallet is 3.
theorem number_of_boxes_on_pallet : total_weight_of_pallet / weight_of_each_box = 3 :=
by sorry

end NUMINAMATH_GPT_number_of_boxes_on_pallet_l1204_120409


namespace NUMINAMATH_GPT_value_of_unknown_number_l1204_120476

theorem value_of_unknown_number (x n : ℤ) 
  (h1 : x = 88320) 
  (h2 : x + n + 9211 - 1569 = 11901) : 
  n = -84061 :=
by
  sorry

end NUMINAMATH_GPT_value_of_unknown_number_l1204_120476


namespace NUMINAMATH_GPT_airline_num_airplanes_l1204_120421

-- Definitions based on the conditions
def rows_per_airplane : ℕ := 20
def seats_per_row : ℕ := 7
def flights_per_day_per_airplane : ℕ := 2
def total_passengers_per_day : ℕ := 1400

-- The theorem to prove the number of airplanes owned by the company
theorem airline_num_airplanes : 
  (total_passengers_per_day = 
   rows_per_airplane * seats_per_row * flights_per_day_per_airplane * n) → 
  n = 5 := 
by 
  sorry

end NUMINAMATH_GPT_airline_num_airplanes_l1204_120421


namespace NUMINAMATH_GPT_remainder_mod_of_a_squared_subtract_3b_l1204_120486

theorem remainder_mod_of_a_squared_subtract_3b (a b : ℕ) (h₁ : a % 7 = 2) (h₂ : b % 7 = 5) (h₃ : a^2 > 3 * b) : 
  (a^2 - 3 * b) % 7 = 3 := 
sorry

end NUMINAMATH_GPT_remainder_mod_of_a_squared_subtract_3b_l1204_120486


namespace NUMINAMATH_GPT_inequality_inequality_always_holds_l1204_120445

theorem inequality_inequality_always_holds (x y : ℝ) (h : x > y) : |x| > y :=
sorry

end NUMINAMATH_GPT_inequality_inequality_always_holds_l1204_120445


namespace NUMINAMATH_GPT_negative_square_inequality_l1204_120492

theorem negative_square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end NUMINAMATH_GPT_negative_square_inequality_l1204_120492
