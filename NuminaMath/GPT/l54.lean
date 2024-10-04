import Mathlib

namespace perpendicular_slope_of_line_l54_54293

theorem perpendicular_slope_of_line (x y : ℤ) : 
    (5 * x - 4 * y = 20) → 
    ∃ m : ℚ, m = -4 / 5 := 
by 
    sorry

end perpendicular_slope_of_line_l54_54293


namespace divisible_sum_or_difference_l54_54882

theorem divisible_sum_or_difference (a : Fin 52 → ℤ) :
  ∃ i j, (i ≠ j) ∧ (a i + a j) % 100 = 0 ∨ (a i - a j) % 100 = 0 :=
by
  sorry

end divisible_sum_or_difference_l54_54882


namespace carrots_total_l54_54515

variables (initiallyPicked : Nat) (thrownOut : Nat) (pickedNextDay : Nat)

def totalCarrots (initiallyPicked : Nat) (thrownOut : Nat) (pickedNextDay : Nat) :=
  initiallyPicked - thrownOut + pickedNextDay

theorem carrots_total (h1 : initiallyPicked = 19)
                     (h2 : thrownOut = 4)
                     (h3 : pickedNextDay = 46) :
  totalCarrots initiallyPicked thrownOut pickedNextDay = 61 :=
by
  sorry

end carrots_total_l54_54515


namespace probability_of_neither_event_l54_54885

theorem probability_of_neither_event (P_A P_B P_A_and_B : ℝ) (h1 : P_A = 0.25) (h2 : P_B = 0.40) (h3 : P_A_and_B = 0.15) : 
  1 - (P_A + P_B - P_A_and_B) = 0.50 :=
by
  rw [h1, h2, h3]
  sorry

end probability_of_neither_event_l54_54885


namespace minimum_participants_l54_54203

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l54_54203


namespace discount_limit_l54_54568

theorem discount_limit {cost_price selling_price : ℕ} (x : ℚ)
  (h1: cost_price = 100)
  (h2: selling_price = 150)
  (h3: ∃ p : ℚ, p = 1.2 * cost_price) : selling_price * (x / 10) - cost_price ≥ 0.2 * cost_price ↔ x ≤ 8 :=
by {
  sorry
}

end discount_limit_l54_54568


namespace cn_squared_eq_28_l54_54014

theorem cn_squared_eq_28 (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end cn_squared_eq_28_l54_54014


namespace N_divisible_by_9_l54_54837

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem N_divisible_by_9 (N : ℕ) (h : sum_of_digits N = sum_of_digits (5 * N)) : N % 9 = 0 := 
sorry

end N_divisible_by_9_l54_54837


namespace triangle_PQR_area_l54_54402

-- Define the points P, Q, and R
def P : (ℝ × ℝ) := (-2, 2)
def Q : (ℝ × ℝ) := (8, 2)
def R : (ℝ × ℝ) := (4, -4)

-- Define a function to calculate the area of triangle
def triangle_area (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Lean statement to prove the area of triangle PQR is 30 square units
theorem triangle_PQR_area : triangle_area P Q R = 30 := by
  sorry

end triangle_PQR_area_l54_54402


namespace total_salmon_count_l54_54154

def chinook_males := 451228
def chinook_females := 164225
def sockeye_males := 212001
def sockeye_females := 76914
def coho_males := 301008
def coho_females := 111873
def pink_males := 518001
def pink_females := 182945
def chum_males := 230023
def chum_females := 81321

theorem total_salmon_count : 
  chinook_males + chinook_females + 
  sockeye_males + sockeye_females + 
  coho_males + coho_females + 
  pink_males + pink_females + 
  chum_males + chum_females = 2329539 := 
by
  sorry

end total_salmon_count_l54_54154


namespace perpendicular_slope_l54_54602

theorem perpendicular_slope (x y : ℝ) (h : 3 * x - 4 * y = 8) :
  ∃ m : ℝ, m = - (4 / 3) :=
by
  sorry

end perpendicular_slope_l54_54602


namespace volleyball_shotput_cost_l54_54551

theorem volleyball_shotput_cost (x y : ℝ) :
  (2*x + 3*y = 95) ∧ (5*x + 7*y = 230) :=
  sorry

end volleyball_shotput_cost_l54_54551


namespace binomial_square_formula_l54_54558

theorem binomial_square_formula (a b : ℝ) :
  let e1 := (4 * a + b) * (4 * a - 2 * b)
  let e2 := (a - 2 * b) * (2 * b - a)
  let e3 := (2 * a - b) * (-2 * a + b)
  let e4 := (a - b) * (a + b)
  (e4 = a^2 - b^2) :=
by
  sorry

end binomial_square_formula_l54_54558


namespace evaluate_expression_l54_54287

theorem evaluate_expression (x : ℝ) (h1 : x^3 + 2 ≠ 0) (h2 : x^3 - 2 ≠ 0) :
  (( (x+2)^3 * (x^2-x+2)^3 / (x^3+2)^3 )^3 * ( (x-2)^3 * (x^2+x+2)^3 / (x^3-2)^3 )^3 ) = 1 :=
by
  sorry

end evaluate_expression_l54_54287


namespace suma_work_rate_l54_54883

theorem suma_work_rate (W : ℕ) : 
  (∀ W, (W / 6) + (W / S) = W / 4) → S = 24 :=
by
  intro h
  -- detailed proof would actually go here
  sorry

end suma_work_rate_l54_54883


namespace range_of_a_l54_54933

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h0 : a > 0) (h1 : a ≠ 1) 
  (h2 : log_base a (a^2 + 1) < log_base a (2 * a))
  (h3 : log_base a (2 * a) < 0) : a ∈ Set.Ioo (0.5) 1 := 
sorry

end range_of_a_l54_54933


namespace oliver_learning_vowels_l54_54968

theorem oliver_learning_vowels : 
  let learn := 5
  let rest_days (n : Nat) := n
  let total_days :=
    (learn + rest_days 1) + -- For 'A'
    (learn + rest_days 2) + -- For 'E'
    (learn + rest_days 3) + -- For 'I'
    (learn + rest_days 4) + -- For 'O'
    (rest_days 5 + learn)  -- For 'U' and 'Y'
  total_days = 40 :=
by
  sorry

end oliver_learning_vowels_l54_54968


namespace tan_3theta_l54_54789

-- Let θ be an angle such that tan θ = 3.
variable (θ : ℝ)
noncomputable def tan_theta : ℝ := 3

-- Claim: tan(3 * θ) = 9/13
theorem tan_3theta :
  Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_3theta_l54_54789


namespace next_time_10_10_11_15_l54_54569

noncomputable def next_time_angle_x (current_time : ℕ × ℕ) (x : ℕ) : ℕ × ℕ := sorry

theorem next_time_10_10_11_15 :
  ∀ (x : ℕ), next_time_angle_x (10, 10) 115 = (11, 15) := sorry

end next_time_10_10_11_15_l54_54569


namespace soda_cost_proof_l54_54827

-- Define the main facts about the weeds
def weeds_flower_bed : ℕ := 11
def weeds_vegetable_patch : ℕ := 14
def weeds_grass : ℕ := 32 / 2  -- Only half the weeds in the grass

-- Define the earning rate
def earning_per_weed : ℕ := 6

-- Define the total earnings and the remaining money conditions
def total_earnings : ℕ := (weeds_flower_bed + weeds_vegetable_patch + weeds_grass) * earning_per_weed
def remaining_money : ℕ := 147

-- Define the cost of the soda
def cost_of_soda : ℕ := total_earnings - remaining_money

-- Problem statement: Prove that the cost of the soda is 99 cents
theorem soda_cost_proof : cost_of_soda = 99 := by
  sorry

end soda_cost_proof_l54_54827


namespace find_a_for_even_function_l54_54766

theorem find_a_for_even_function (a : ℝ) : 
  (∀ x : ℝ, (x + a) * (x - 4) = ((-x) + a) * ((-x) - 4)) → a = 4 :=
by sorry

end find_a_for_even_function_l54_54766


namespace minimum_value_2sqrt5_l54_54597

theorem minimum_value_2sqrt5 : ∀ x : ℝ, 
  ∃ m : ℝ, (∀ x : ℝ, m ≤ (x^2 + 10) / (Real.sqrt (x^2 + 5))) ∧ (m = 2 * Real.sqrt 5) := by
  sorry

end minimum_value_2sqrt5_l54_54597


namespace positive_number_is_25_l54_54212

theorem positive_number_is_25 {a x : ℝ}
(h1 : x = (3 * a + 1)^2)
(h2 : x = (-a - 3)^2)
(h_sum : 3 * a + 1 + (-a - 3) = 0) :
x = 25 :=
sorry

end positive_number_is_25_l54_54212


namespace min_x_plus_2y_l54_54757

theorem min_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / (2 * x + y) + 1 / (y + 1) = 1) : x + 2 * y ≥ (1 / 2) + Real.sqrt 3 :=
sorry

end min_x_plus_2y_l54_54757


namespace race_participants_least_number_l54_54205

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l54_54205


namespace area_of_circle_l54_54057

theorem area_of_circle (C : ℝ) (hC : C = 30 * Real.pi) : ∃ k : ℝ, (Real.pi * (C / (2 * Real.pi))^2 = k * Real.pi) ∧ k = 225 :=
by
  sorry

end area_of_circle_l54_54057


namespace odd_function_value_at_neg2_l54_54145

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_ge_one : ∀ x, 1 ≤ x → f x = 3 * x - 7)

theorem odd_function_value_at_neg2 : f (-2) = 1 :=
by
  -- Proof goes here
  sorry

end odd_function_value_at_neg2_l54_54145


namespace markup_rate_l54_54734

variable (S : ℝ) (C : ℝ)
variable (profit_percent : ℝ := 0.12) (expense_percent : ℝ := 0.18)
variable (selling_price : ℝ := 8.00)

theorem markup_rate (h1 : C + profit_percent * S + expense_percent * S = S)
                    (h2 : S = selling_price) :
  ((S - C) / C) * 100 = 42.86 := by
  sorry

end markup_rate_l54_54734


namespace total_number_of_animals_is_304_l54_54941

theorem total_number_of_animals_is_304
    (dogs frogs : ℕ) 
    (h1 : frogs = 160) 
    (h2 : frogs = 2 * dogs) 
    (cats : ℕ) 
    (h3 : cats = dogs - (dogs / 5)) :
  cats + dogs + frogs = 304 :=
by
  sorry

end total_number_of_animals_is_304_l54_54941


namespace volume_box_l54_54879

theorem volume_box (x y : ℝ) :
  (16 - 2 * x) * (12 - 2 * y) * y = 4 * x * y ^ 2 - 24 * x * y + 192 * y - 32 * y ^ 2 :=
by sorry

end volume_box_l54_54879


namespace tangent_line_intersects_x_axis_at_minus_3_l54_54863

open Real

noncomputable def curve (x : ℝ) : ℝ := x^3 + 11

def point_p : ℝ × ℝ := (1, 12)

theorem tangent_line_intersects_x_axis_at_minus_3 :
  let tangent_slope := deriv curve 1 in
  let tangent_line (x : ℝ) : ℝ := tangent_slope * (x - point_p.1) + point_p.2 in
  ∃ x₀, tangent_line x₀ = 0 ∧ x₀ = -3 :=
by
  sorry

end tangent_line_intersects_x_axis_at_minus_3_l54_54863


namespace discount_price_l54_54449

theorem discount_price (P P_d : ℝ) 
  (h1 : P_d = 0.85 * P) 
  (P_final : ℝ) 
  (h2 : P_final = 1.25 * P_d) 
  (h3 : P - P_final = 5.25) :
  P_d = 71.4 :=
by
  sorry

end discount_price_l54_54449


namespace eugene_pencils_after_giving_l54_54286

-- Define Eugene's initial number of pencils and the number of pencils given away.
def initial_pencils : ℝ := 51.0
def pencils_given : ℝ := 6.0

-- State the theorem that should be proved.
theorem eugene_pencils_after_giving : initial_pencils - pencils_given = 45.0 :=
by
  -- We would normally provide the proof steps here, but as per instructions, we'll use "sorry" to skip it.
  sorry

end eugene_pencils_after_giving_l54_54286


namespace distance_between_parallel_lines_l54_54379

-- Definition of the first line l1
def line1 (x y : ℝ) (c1 : ℝ) : Prop := 3 * x + 4 * y + c1 = 0

-- Definition of the second line l2
def line2 (x y : ℝ) (c2 : ℝ) : Prop := 6 * x + 8 * y + c2 = 0

-- The problem statement in Lean:
theorem distance_between_parallel_lines (c1 c2 : ℝ) :
  ∃ d : ℝ, d = |2 * c1 - c2| / 10 :=
sorry

end distance_between_parallel_lines_l54_54379


namespace problem_l54_54384

def P (x : ℝ) : Prop := x^2 - 2*x + 1 > 0

theorem problem (h : ¬ ∀ x : ℝ, P x) : ∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0 :=
by {
  sorry
}

end problem_l54_54384


namespace dice_probability_l54_54425

theorem dice_probability :
  let prob_one_digit := (9:ℚ) / 20
  let prob_two_digit := (11:ℚ) / 20
  let prob := 10 * (prob_two_digit^2) * (prob_one_digit^3)
  prob = 1062889 / 128000000 := 
by 
  sorry

end dice_probability_l54_54425


namespace find_other_number_l54_54393

theorem find_other_number (x y : ℕ) (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) (h3 : x = 7) : y = 3 :=
by
  sorry

end find_other_number_l54_54393


namespace lowest_point_in_fourth_quadrant_l54_54939

theorem lowest_point_in_fourth_quadrant (k : ℝ) (h : k < -1) :
    let x := - (k + 1) / 2
    let y := (4 * k - (k + 1) ^ 2) / 4
    y < 0 ∧ x > 0 :=
by
  let x := - (k + 1) / 2
  let y := (4 * k - (k + 1) ^ 2) / 4
  sorry

end lowest_point_in_fourth_quadrant_l54_54939


namespace max_f_max_g_pow_f_l54_54621

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4 * x + 3) / (x^2 + 7 * x + 14)
noncomputable def g (x : ℝ) : ℝ := (x^2 - 5 * x + 10) / (x^2 + 5 * x + 20)

theorem max_f : ∀ x : ℝ, f x ≤ 2 := by
  intro x
  sorry

theorem max_g_pow_f : ∀ x : ℝ, g x ^ f x ≤ 9 := by
  intro x
  sorry

end max_f_max_g_pow_f_l54_54621


namespace problem_statement_l54_54306

noncomputable def f (x : ℝ) := 2 * x + 3
noncomputable def g (x : ℝ) := 3 * x - 2

theorem problem_statement : (f (g (f 3)) / g (f (g 3))) = 53 / 49 :=
by
  -- The proof is not provided as requested.
  sorry

end problem_statement_l54_54306


namespace sum_and_product_of_white_are_white_l54_54518

-- Definitions based on the conditions
def is_colored_black_or_white (n : ℕ) : Prop :=
  true -- This is a simplified assumption since this property is always true.

def is_black (n : ℕ) : Prop := (n % 2 = 0)
def is_white (n : ℕ) : Prop := (n % 2 = 1)

-- Conditions given in the problem
axiom sum_diff_colors_is_black (a b : ℕ) (ha : is_black a) (hb : is_white b) : is_black (a + b)
axiom infinitely_many_whites : ∀ n, ∃ m ≥ n, is_white m

-- Statement to prove that the sum and product of two white numbers are white
theorem sum_and_product_of_white_are_white (a b : ℕ) (ha : is_white a) (hb : is_white b) : 
  is_white (a + b) ∧ is_white (a * b) :=
sorry

end sum_and_product_of_white_are_white_l54_54518


namespace multiple_of_bees_l54_54965

theorem multiple_of_bees (b₁ b₂ : ℕ) (h₁ : b₁ = 144) (h₂ : b₂ = 432) : b₂ / b₁ = 3 := 
by
  sorry

end multiple_of_bees_l54_54965


namespace min_value_proof_l54_54656

noncomputable def min_value (α γ : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin γ - 7)^2 + (3 * Real.sin α + 4 * Real.cos γ - 12)^2

theorem min_value_proof (α γ : ℝ) : ∃ α γ : ℝ, min_value α γ = 36 :=
by
  use (Real.arcsin 12/13), (Real.pi/2 - Real.arcsin 12/13)
  sorry

end min_value_proof_l54_54656


namespace find_number_l54_54542

theorem find_number (x : ℝ) (h : 100 - x = x + 40) : x = 30 :=
sorry

end find_number_l54_54542


namespace math_problem_l54_54003

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l54_54003


namespace unique_triangle_exists_l54_54051

-- Definition of a triangle with consecutive side lengths and angle constraints
structure TriangularProps (a b c : ℕ) (A B C : ℝ) where
  sides_consecutive : a + 1 = b ∧ b + 1 = c
  angles_relation : A = 2 * B

theorem unique_triangle_exists :
  ∃! (a b c : ℕ) (A B C : ℝ), TriangularProps a b c A B C := sorry

end unique_triangle_exists_l54_54051


namespace lions_min_games_for_90_percent_wins_l54_54262

theorem lions_min_games_for_90_percent_wins : 
  ∀ N : ℕ, (N ≥ 26) ↔ 1 + N ≥ (9 * (4 + N)) / 10 := 
by 
  sorry

end lions_min_games_for_90_percent_wins_l54_54262


namespace f_expr_for_nonneg_l54_54135

-- Define the function f piecewise as per the given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then
    Real.exp (-x) + 2 * x - 1
  else
    -Real.exp x + 2 * x + 1

-- Prove that for x > 0, f(x) = -e^x + 2x + 1 given the conditions
theorem f_expr_for_nonneg (x : ℝ) (h : x ≥ 0) : f x = -Real.exp x + 2 * x + 1 := by
  sorry

end f_expr_for_nonneg_l54_54135


namespace cloth_length_l54_54989

theorem cloth_length (L : ℕ) (x : ℕ) :
  32 + x = L ∧ 20 + 3 * x = L → L = 38 :=
by
  sorry

end cloth_length_l54_54989


namespace negation_prop_l54_54538

variable {U : Type} (A B : Set U)
variable (x : U)

theorem negation_prop (h : x ∈ A ∩ B) : (x ∉ A ∩ B) → (x ∉ A ∧ x ∉ B) :=
sorry

end negation_prop_l54_54538


namespace ceil_neg_sqrt_fraction_l54_54120

theorem ceil_neg_sqrt_fraction :
  (⌈-real.sqrt (64 / 9)⌉ = -2) :=
by
  -- Define the necessary conditions
  have h1 : real.sqrt (64 / 9) = 8 / 3 := by sorry,
  have h2 : -real.sqrt (64 / 9) = -8 / 3 := by sorry,
  -- Apply the ceiling function and prove the result
  exact sorry

end ceil_neg_sqrt_fraction_l54_54120


namespace cabbage_price_l54_54039

theorem cabbage_price
  (earnings_wednesday : ℕ)
  (earnings_friday : ℕ)
  (earnings_today : ℕ)
  (total_weight : ℕ)
  (h1 : earnings_wednesday = 30)
  (h2 : earnings_friday = 24)
  (h3 : earnings_today = 42)
  (h4 : total_weight = 48) :
  (earnings_wednesday + earnings_friday + earnings_today) / total_weight = 2 := by
  sorry

end cabbage_price_l54_54039


namespace jill_runs_more_than_jack_l54_54335

noncomputable def streetWidth : ℝ := 15 -- Street width in feet
noncomputable def blockSide : ℝ := 300 -- Side length of the block in feet

noncomputable def jacksPerimeter : ℝ := 4 * blockSide -- Perimeter of Jack's running path
noncomputable def jillsPerimeter : ℝ := 4 * (blockSide + 2 * streetWidth) -- Perimeter of Jill's running path on the opposite side of the street

theorem jill_runs_more_than_jack :
  jillsPerimeter - jacksPerimeter = 120 :=
by
  sorry

end jill_runs_more_than_jack_l54_54335


namespace ham_block_cut_mass_distribution_l54_54086

theorem ham_block_cut_mass_distribution
  (length width height : ℝ) (mass : ℝ)
  (parallelogram_side1 parallelogram_side2 : ℝ)
  (condition1 : length = 12) 
  (condition2 : width = 12) 
  (condition3 : height = 35)
  (condition4 : mass = 5)
  (condition5 : parallelogram_side1 = 15) 
  (condition6 : parallelogram_side2 = 20) :
  ∃ (mass_piece1 mass_piece2 : ℝ),
    mass_piece1 = 1.7857 ∧ mass_piece2 = 3.2143 :=
by
  sorry

end ham_block_cut_mass_distribution_l54_54086


namespace find_x_l54_54644

theorem find_x (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y + x * y = 104) : x = 34 :=
sorry

end find_x_l54_54644


namespace car_discount_l54_54260

variable (P D : ℝ)

theorem car_discount (h1 : 0 < P)
                     (h2 : (P - D) * 1.45 = 1.16 * P) :
                     D = 0.2 * P := by
  sorry

end car_discount_l54_54260


namespace tangent_line_circle_l54_54937

theorem tangent_line_circle (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, x + y = 2 ↔ x^2 + y^2 = m) → m = 2 :=
by
  intro h_tangent
  sorry

end tangent_line_circle_l54_54937


namespace find_x_if_opposites_l54_54130

theorem find_x_if_opposites (x : ℝ) (h : 2 * (x - 3) = - 4 * (1 - x)) : x = -1 := 
by
  sorry

end find_x_if_opposites_l54_54130


namespace find_central_angle_l54_54615

noncomputable def sector := 
  {R : ℝ // R > 0}

noncomputable def central_angle (R : ℝ) : ℝ := 
  (6 - 2 * R) / R

theorem find_central_angle :
  ∃ α : ℝ, (α = 1 ∨ α = 4) ∧ 
  (∃ R : ℝ, 
    (2 * R + α * R = 6) ∧ 
    (1 / 2 * R^2 * α = 2)) := 
by {
  sorry
}

end find_central_angle_l54_54615


namespace petya_must_have_photo_files_on_portable_hard_drives_l54_54842

theorem petya_must_have_photo_files_on_portable_hard_drives 
    (H F P T : ℕ) 
    (h1 : H > F) 
    (h2 : P > T) 
    : ∃ x, x ≠ 0 ∧ x ≤ H :=
by
  sorry

end petya_must_have_photo_files_on_portable_hard_drives_l54_54842


namespace arithmetic_sequence_problem_l54_54153

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (d : ℚ) (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 4 + 1 / 2 * a 7 + a 10 = 10) : a 3 + a 11 = 8 :=
sorry

end arithmetic_sequence_problem_l54_54153


namespace tan_triple_angle_l54_54778

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l54_54778


namespace find_judy_rotation_l54_54273

-- Definition of the problem
def CarlaRotation := 480 % 360 -- This effectively becomes 120
def JudyRotation (y : ℕ) := (360 - 120) % 360 -- This should effectively be 240

-- Theorem stating the problem and solution
theorem find_judy_rotation (y : ℕ) (h : y < 360) : 360 - CarlaRotation = y :=
by 
  dsimp [CarlaRotation, JudyRotation] 
  sorry

end find_judy_rotation_l54_54273


namespace find_m_l54_54011

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (-1, -1)
noncomputable def a_minus_b : ℝ × ℝ := (2, 3)
noncomputable def m_a_plus_b (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m - 1)

theorem find_m (m : ℝ) : (a_minus_b.1 * (m_a_plus_b m).1 + a_minus_b.2 * (m_a_plus_b m).2) = 0 → m = 5 / 8 := 
by
  sorry

end find_m_l54_54011


namespace problem1_problem2_l54_54270

-- First Problem Statement:
theorem problem1 :  12 - (-18) + (-7) - 20 = 3 := 
by 
  sorry

-- Second Problem Statement:
theorem problem2 : -4 / (1 / 2) * 8 = -64 := 
by 
  sorry

end problem1_problem2_l54_54270


namespace minimum_bats_examined_l54_54422

theorem minimum_bats_examined 
  (bats : Type) 
  (R L : bats → Prop) 
  (total_bats : ℕ)
  (right_eye_bats : ∀ {b: bats}, R b → Fin 2)
  (left_eye_bats : ∀ {b: bats}, L b → Fin 3)
  (not_left_eye_bats: ∀ {b: bats}, ¬ L b → Fin 4)
  (not_right_eye_bats: ∀ {b: bats}, ¬ R b → Fin 5)
  : total_bats ≥ 7 := sorry

end minimum_bats_examined_l54_54422


namespace total_numbers_in_first_30_rows_of_Pascals_Triangle_l54_54315

-- Define the number of elements in the nth row of Pascal's Triangle
def num_elements_in_row (n : ℕ) : ℕ := n + 1

-- Define the sum of numbers from 0 to n, inclusive
def sum_of_first_n_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the total number of elements in the first 30 rows (0th to 29th)
def total_elements_in_first_30_rows : ℕ := sum_of_first_n_numbers 30

-- The main statement to prove
theorem total_numbers_in_first_30_rows_of_Pascals_Triangle :
  total_elements_in_first_30_rows = 465 :=
by
  simp [total_elements_in_first_30_rows, sum_of_first_n_numbers]
  sorry

end total_numbers_in_first_30_rows_of_Pascals_Triangle_l54_54315


namespace highest_power_of_5_dividing_S_l54_54294

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def f (n : ℕ) : ℤ :=
  if sum_of_digits n % 2 = 0 then n ^ 100 else -n ^ 100

def S : ℤ :=
  (Finset.range (10 ^ 100)).sum (λ n => f n)

theorem highest_power_of_5_dividing_S :
  ∃ m : ℕ, 5 ^ m ∣ S ∧ ∀ k : ℕ, 5 ^ (k + 1) ∣ S → k < 24 :=
by
  sorry

end highest_power_of_5_dividing_S_l54_54294


namespace remainder_problem_l54_54560

theorem remainder_problem (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) : N % 15 = 10 :=
by
  sorry

end remainder_problem_l54_54560


namespace minimum_participants_l54_54204

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l54_54204


namespace school_student_monthly_earnings_l54_54704

theorem school_student_monthly_earnings :
  let daily_rate := 1250
  let days_per_week := 4
  let weeks_per_month := 4
  let tax_rate := 0.13
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings := weekly_earnings * weeks_per_month
  let tax := monthly_earnings * tax_rate
  let earnings_after_tax := monthly_earnings - tax
  earnings_after_tax = 17400 :=
by
  let daily_rate := 1250
  let days_per_week := 4
  let weeks_per_month := 4
  let tax_rate := 0.13
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings := weekly_earnings * weeks_per_month
  let tax := monthly_earnings * tax_rate
  let earnings_after_tax := monthly_earnings - tax
  sorry

end school_student_monthly_earnings_l54_54704


namespace square_plot_area_l54_54415

theorem square_plot_area (cost_per_foot total_cost : ℕ) (hcost_per_foot : cost_per_foot = 60) (htotal_cost : total_cost = 4080) :
  ∃ (A : ℕ), A = 289 :=
by
  have h : 4 * 60 * 17 = 4080 := by rfl
  have s : 17 = 4080 / (4 * 60) := by sorry
  use 17 ^ 2
  have hsquare : 17 ^ 2 = 289 := by rfl
  exact hsquare

end square_plot_area_l54_54415


namespace simplify_expression_l54_54348

theorem simplify_expression (a c b : ℝ) (h1 : a > c) (h2 : c ≥ 0) (h3 : b > 0) :
  (a * b^2 * (1 / (a + c)^2 + 1 / (a - c)^2) = a - b) → (2 * a * b = a^2 - c^2) :=
by
  sorry

end simplify_expression_l54_54348


namespace interest_rate_proof_l54_54858

noncomputable def compound_interest_rate (P A : ℝ) (t n : ℕ) : ℝ :=
  (((A / P)^(1 / (n * t))) - 1) * n

theorem interest_rate_proof :
  ∀ P A : ℝ, ∀ t n : ℕ, P = 1093.75 → A = 1183 → t = 2 → n = 1 →
  compound_interest_rate P A t n = 0.0399 :=
by
  intros P A t n hP hA ht hn
  rw [hP, hA, ht, hn]
  unfold compound_interest_rate
  sorry

end interest_rate_proof_l54_54858


namespace contrapositive_correct_l54_54213

-- Conditions and the proposition
def prop1 (a : ℝ) : Prop := a = -1 → a^2 = 1

-- The contrapositive of the proposition
def contrapositive (a : ℝ) : Prop := a^2 ≠ 1 → a ≠ -1

-- The proof problem statement
theorem contrapositive_correct (a : ℝ) : prop1 a ↔ contrapositive a :=
by sorry

end contrapositive_correct_l54_54213


namespace water_addition_to_achieve_concentration_l54_54623

theorem water_addition_to_achieve_concentration :
  ∀ (w1 w2 : ℝ), 
  (60 * 0.25 = 15) →              -- initial amount of acid
  (15 / (60 + w1) = 0.15) →       -- first dilution to 15%
  (15 / (100 + w2) = 0.10) →      -- second dilution to 10%
  w1 + w2 = 90 :=                 -- total water added to achieve final concentration
by
  intros w1 w2 h_initial h_first h_second
  sorry

end water_addition_to_achieve_concentration_l54_54623


namespace rooms_count_l54_54513

theorem rooms_count (total_paintings : ℕ) (paintings_per_room : ℕ) (h1 : total_paintings = 32) (h2 : paintings_per_room = 8) : (total_paintings / paintings_per_room) = 4 := by
  sorry

end rooms_count_l54_54513


namespace complete_square_solution_l54_54691

theorem complete_square_solution :
  ∀ x : ℝ, x^2 - 4 * x - 22 = 0 → (x - 2)^2 = 26 :=
by
  intro x h
  sorry

end complete_square_solution_l54_54691


namespace cistern_filling_time_with_leak_l54_54713

theorem cistern_filling_time_with_leak (T : ℝ) (h1 : 1 / T - 1 / 4 = 1 / (T + 2)) : T = 4 :=
by
  sorry

end cistern_filling_time_with_leak_l54_54713


namespace ball_bounce_height_l54_54709

theorem ball_bounce_height
  (k : ℕ) 
  (h1 : 20 * (2 / 3 : ℝ)^k < 2) : 
  k = 7 :=
sorry

end ball_bounce_height_l54_54709


namespace arithmetic_sequence_common_difference_l54_54639

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a6 : a 6 = 5) (h_a10 : a 10 = 6) : 
  (a 10 - a 6) / 4 = 1 / 4 := 
by
  sorry

end arithmetic_sequence_common_difference_l54_54639


namespace inequalities_proof_l54_54047

theorem inequalities_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  (a < (c / 2)) ∧ (b < a + c / 2) ∧ ¬(b < c / 2) :=
by
  constructor
  { sorry }
  { constructor
    { sorry }
    { sorry } }

end inequalities_proof_l54_54047


namespace sum_of_integers_eq_17_l54_54986

theorem sum_of_integers_eq_17 (a b : ℕ) (h1 : a * b + a + b = 87) 
  (h2 : Nat.gcd a b = 1) (h3 : a < 15) (h4 : b < 15) (h5 : Even a ∨ Even b) :
  a + b = 17 := 
sorry

end sum_of_integers_eq_17_l54_54986


namespace race_participants_least_number_l54_54206

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l54_54206


namespace determine_c_l54_54324

theorem determine_c (c : ℝ) 
  (h : ∃ a : ℝ, (∀ x : ℝ, x^2 + 200 * x + c = (x + a)^2)) : c = 10000 :=
sorry

end determine_c_l54_54324


namespace ceil_neg_sqrt_frac_l54_54119

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := 
sorry

end ceil_neg_sqrt_frac_l54_54119


namespace find_T5_l54_54612

variables (a b x y : ℝ)

def T (n : ℕ) : ℝ := a * x^n + b * y^n

theorem find_T5
  (h1 : T a b x y 1 = 3)
  (h2 : T a b x y 2 = 7)
  (h3 : T a b x y 3 = 6)
  (h4 : T a b x y 4 = 42) :
  T a b x y 5 = -360 :=
sorry

end find_T5_l54_54612


namespace logarithmic_product_l54_54617

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem logarithmic_product (a b : ℝ) (h1 : a ≠ b) (h2 : f a = f b) : a * b = 1 := by
  sorry

end logarithmic_product_l54_54617


namespace number_of_cats_l54_54239

theorem number_of_cats (c d : ℕ) (h1 : c = 20 + d) (h2 : c + d = 60) : c = 40 :=
sorry

end number_of_cats_l54_54239


namespace no_solution_fractional_eq_l54_54525

theorem no_solution_fractional_eq :
  ¬∃ x : ℝ, (1 - x) / (x - 2) = 1 / (2 - x) + 1 :=
by
  -- The proof is intentionally omitted.
  sorry

end no_solution_fractional_eq_l54_54525


namespace cube_volume_surface_area_l54_54465

variable (x : ℝ)

theorem cube_volume_surface_area (h1 : s^3 = 8 * x) (h2 : 6 * s^2 = 2 * x) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_l54_54465


namespace fibers_below_20_count_l54_54715

variable (fibers : List ℕ)

-- Conditions
def total_fibers := fibers.length = 100
def length_interval (f : ℕ) := 5 ≤ f ∧ f ≤ 40
def fibers_within_interval := ∀ f ∈ fibers, length_interval f

-- Question
def fibers_less_than_20 (fibers : List ℕ) : Nat :=
  (fibers.filter (λ f => f < 20)).length

theorem fibers_below_20_count (h_total : total_fibers fibers)
  (h_interval : fibers_within_interval fibers)
  (histogram_data : fibers_less_than_20 fibers = 30) :
  fibers_less_than_20 fibers = 30 :=
by
  sorry

end fibers_below_20_count_l54_54715


namespace curve_y_all_real_l54_54059

theorem curve_y_all_real (y : ℝ) : ∃ (x : ℝ), 2 * x * |x| + y^2 = 1 :=
sorry

end curve_y_all_real_l54_54059


namespace divide_19_degree_angle_into_19_equal_parts_l54_54552

/-- Divide a 19° angle into 19 equal parts, resulting in each part being 1° -/
theorem divide_19_degree_angle_into_19_equal_parts
  (α : ℝ) (hα : α = 19) :
  α / 19 = 1 :=
by
  sorry

end divide_19_degree_angle_into_19_equal_parts_l54_54552


namespace problem_statement_a_problem_statement_b_problem_statement_c_l54_54156

-- Define θ2n as described in the problem statement
def θ2n (S : ℕ → ℤ) (n : ℕ) : ℕ :=
if h : ∃ k > 1, (∀ i < k, S i < S k) ∧ (∀ i > k, i ≤ 2 * n → S i ≤ S k) then
  Classical.choose h
else
  0

-- Define the probability measure P
noncomputable def P {Ω : Type*} [ProbabilitySpace Ω] (event : set Ω) : ℝ :=
Probability.measure_space.unnormalized_measure event

-- Define the sequences of random variables S_0, S_1, ..., S_{2n}
variable {Ω : Type*} [ProbabilitySpace Ω] (S : ℕ → Ω → ℤ)

-- Define the indicators u2n, u2k, u2n_2k
variable {u2n u2k u2n_2k : ℕ → ℝ}

-- Prove the three conditions as Lean statements
theorem problem_statement_a (S : ℕ → Ω → ℤ) (u2n : ℕ → ℝ) (n : ℕ) :
  P {ω | θ2n (λ i, S i ω) n = 0} = u2n (2 * n) :=
sorry

theorem problem_statement_b (S : ℕ → Ω → ℤ) (u2n : ℕ → ℝ) (n : ℕ) :
  P {ω | θ2n (λ i, S i ω) n = 2 * n} = (1 / 2) * u2n (2 * n) :=
sorry

theorem problem_statement_c (S : ℕ → Ω → ℤ) (u2k u2n_2k : ℕ → ℝ) (n k : ℕ) (hkn : 0 < k ∧ k < n) :
  P {ω | θ2n (λ i, S i ω) n = 2 * k ∨ θ2n (λ i, S i ω) n = 2 * k + 1} = (1 / 2) * u2k (2 * k) * u2n_2k (2 * (n - k)) :=
sorry

end problem_statement_a_problem_statement_b_problem_statement_c_l54_54156


namespace negation_prop_l54_54539

variable {U : Type} (A B : Set U)
variable (x : U)

theorem negation_prop (h : x ∈ A ∩ B) : (x ∉ A ∩ B) → (x ∉ A ∧ x ∉ B) :=
sorry

end negation_prop_l54_54539


namespace prove_M_squared_l54_54614

noncomputable def M (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -1, 2], ![ (5/2:ℝ), x]]

def eigenvalue_condition (x : ℝ) : Prop :=
  let A := M x
  ∃ v : ℝ, (A - (-2) • (1 : Matrix (Fin 2) (Fin 2) ℝ)).det = 0

theorem prove_M_squared (x : ℝ) (h : eigenvalue_condition x) :
  (M x * M x) = ![![ 6, -9], ![ - (45/4:ℝ), 69/4]] :=
sorry

end prove_M_squared_l54_54614


namespace distinct_real_roots_range_of_m_l54_54920

theorem distinct_real_roots_range_of_m (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + x₁ - m = 0) ∧ (x₂^2 + x₂ - m = 0)) → m > -1/4 := 
sorry

end distinct_real_roots_range_of_m_l54_54920


namespace roots_cubic_polynomial_l54_54657

theorem roots_cubic_polynomial (a b c : ℝ) 
  (h1 : a^3 - 2*a - 2 = 0) 
  (h2 : b^3 - 2*b - 2 = 0) 
  (h3 : c^3 - 2*c - 2 = 0) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -18 :=
by
  sorry

end roots_cubic_polynomial_l54_54657


namespace new_group_size_l54_54540

theorem new_group_size (N : ℕ) (h1 : 20 < N) (h2 : N < 50) (h3 : (N - 5) % 6 = 0) (h4 : (N - 5) % 7 = 0) (h5 : (N % (N - 7)) = 7) : (N - 7).gcd (N) = 8 :=
by
  sorry

end new_group_size_l54_54540


namespace div_poly_l54_54050

theorem div_poly (m n p : ℕ) : 
  (X^2 + X + 1) ∣ (X^(3*m) + X^(3*n + 1) + X^(3*p + 2)) := 
sorry

end div_poly_l54_54050


namespace youngest_child_age_l54_54421

theorem youngest_child_age (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 55) : x = 7 := 
by
  sorry

end youngest_child_age_l54_54421


namespace largest_perfect_square_factor_of_1764_l54_54079

theorem largest_perfect_square_factor_of_1764 : ∃ m, m * m = 1764 ∧ ∀ n, n * n ∣ 1764 → n * n ≤ 1764 :=
by
  sorry

end largest_perfect_square_factor_of_1764_l54_54079


namespace cyclic_quadrilateral_KRLQ_l54_54960

noncomputable def is_cyclic_quadrilateral (a b c d : Point) : Prop :=
∃ (circle : Circle), a ∈ circle ∧ b ∈ circle ∧ c ∈ circle ∧ d ∈ circle

theorem cyclic_quadrilateral_KRLQ
  (A S T X Y R P Q K L : Point)
  (circle_omega : Circle)
  (hA : ¬ A ∈ circle_omega)
  (hS : S ∈ circle_omega)
  (hT : T ∈ circle_omega)
  (hST : tangent A S ∧ tangent A T)
  (hX : midpoint X A T)
  (hY : midpoint Y A S)
  (hR : tangent X R ∧ R ∈ circle_omega)
  (hP : midpoint P X T)
  (hQ : midpoint Q X R)
  (hK : line_intersection P Q X Y = some K)
  (hL : line_intersection S X T K = some L) :
  is_cyclic_quadrilateral K R L Q :=
sorry

end cyclic_quadrilateral_KRLQ_l54_54960


namespace negate_neg_two_l54_54367

theorem negate_neg_two : -(-2) = 2 := by
  -- The proof goes here
  sorry

end negate_neg_two_l54_54367


namespace tan_theta_3_l54_54782

noncomputable def tan_triple_angle (θ : ℝ) : ℝ := (3 * (Real.tan θ) - ((Real.tan θ) ^ 3)) / (1 - 3 * (Real.tan θ)^2)

theorem tan_theta_3 (θ : ℝ) (h : Real.tan θ = 3) : tan_triple_angle θ = 9 / 13 :=
by
  sorry

end tan_theta_3_l54_54782


namespace number_of_deluxe_volumes_l54_54546

theorem number_of_deluxe_volumes (d s : ℕ) 
  (h1 : d + s = 15)
  (h2 : 30 * d + 20 * s = 390) : 
  d = 9 :=
by
  sorry

end number_of_deluxe_volumes_l54_54546


namespace hours_per_batch_l54_54945

noncomputable section

def gallons_per_batch : ℕ := 3 / 2   -- 1.5 gallons expressed as a rational number
def ounces_per_gallon : ℕ := 128
def jack_consumption_per_2_days : ℕ := 96
def total_days : ℕ := 24
def time_spent_hours : ℕ := 120

def total_ounces : ℕ := gallons_per_batch * ounces_per_gallon
def total_ounces_consumed_24_days : ℕ := jack_consumption_per_2_days * (total_days / 2)
def number_of_batches : ℕ := total_ounces_consumed_24_days / total_ounces

theorem hours_per_batch :
  (time_spent_hours / number_of_batches) = 20 := by
  sorry

end hours_per_batch_l54_54945


namespace number_of_cows_l54_54152

-- Definitions
variables (c h : ℕ)

-- Conditions
def condition1 : Prop := 4 * c + 2 * h = 20 + 2 * (c + h)
def condition2 : Prop := c + h = 12

-- Theorem
theorem number_of_cows : condition1 c h → condition2 c h → c = 10 :=
  by 
  intros h1 h2
  sorry

end number_of_cows_l54_54152


namespace unfolded_paper_has_eight_holes_l54_54738

theorem unfolded_paper_has_eight_holes
  (T : Type)
  (equilateral_triangle : T)
  (midpoint : T → T → T)
  (vertex_fold : T → T → T)
  (holes_punched : T → ℕ)
  (first_fold_vertex midpoint_1 : T)
  (second_fold_vertex midpoint_2 : T)
  (holes_near_first_fold holes_near_second_fold : ℕ) :
  holes_punched (vertex_fold second_fold_vertex midpoint_2)
    = 8 := 
by sorry

end unfolded_paper_has_eight_holes_l54_54738


namespace negation_proof_l54_54773

theorem negation_proof :
  (∃ x₀ : ℝ, x₀ < 2) → ¬ (∀ x : ℝ, x < 2) :=
by
  sorry

end negation_proof_l54_54773


namespace functions_eq_l54_54660

open Function

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

theorem functions_eq (h_surj : Surjective f) (h_inj : Injective g) (h_ge : ∀ n : ℕ, f n ≥ g n) : ∀ n : ℕ, f n = g n :=
sorry

end functions_eq_l54_54660


namespace teacher_age_l54_54847

theorem teacher_age (avg_student_age : ℕ) (num_students : ℕ) (new_avg_age : ℕ) (num_total : ℕ) (total_student_age : ℕ) (total_age_with_teacher : ℕ) :
  avg_student_age = 22 → 
  num_students = 23 → 
  new_avg_age = 23 → 
  num_total = 24 → 
  total_student_age = avg_student_age * num_students → 
  total_age_with_teacher = new_avg_age * num_total → 
  total_age_with_teacher - total_student_age = 46 :=
by
  intros
  sorry

end teacher_age_l54_54847


namespace arrow_sequence_correct_l54_54423

variable (A B C D E F G : ℕ)
variable (square : ℕ → ℕ)

-- Definitions based on given conditions
def conditions : Prop :=
  square 1 = 1 ∧ square 9 = 9 ∧
  square A = 6 ∧ square B = 2 ∧ square C = 4 ∧
  square D = 5 ∧ square E = 3 ∧ square F = 8 ∧ square G = 7 ∧
  (∀ x, (x = 1 → square 2 = B) ∧ (x = 2 → square 3 = E) ∧
       (x = 3 → square 4 = C) ∧ (x = 4 → square 5 = D) ∧
       (x = 5 → square 6 = A) ∧ (x = 6 → square 7 = G) ∧
       (x = 7 → square 8 = F) ∧ (x = 8 → square 9 = 9))

theorem arrow_sequence_correct :
  conditions A B C D E F G square → 
  ∀ x, square (x + 1) = 1 + x :=
by sorry

end arrow_sequence_correct_l54_54423


namespace map_a_distance_map_b_distance_miles_map_b_distance_km_l54_54719

theorem map_a_distance (distance_cm : ℝ) (scale_cm : ℝ) (scale_km : ℝ) (actual_distance : ℝ) : 
  distance_cm = 80.5 → scale_cm = 0.6 → scale_km = 6.6 → actual_distance = (distance_cm * scale_km) / scale_cm → actual_distance = 885.5 :=
by
  intros h1 h2 h3 h4
  sorry

theorem map_b_distance_miles (distance_cm : ℝ) (scale_cm : ℝ) (scale_miles : ℝ) (actual_distance_miles : ℝ) : 
  distance_cm = 56.3 → scale_cm = 1.1 → scale_miles = 7.7 → actual_distance_miles = (distance_cm * scale_miles) / scale_cm → actual_distance_miles = 394.1 :=
by
  intros h1 h2 h3 h4
  sorry

theorem map_b_distance_km (distance_miles : ℝ) (conversion_factor : ℝ) (actual_distance_km : ℝ) :
  conversion_factor = 1.60934 → distance_miles = 394.1 → actual_distance_km = distance_miles * conversion_factor → actual_distance_km = 634.3 :=
by
  intros h1 h2 h3
  sorry

end map_a_distance_map_b_distance_miles_map_b_distance_km_l54_54719


namespace tan_3theta_l54_54787

-- Let θ be an angle such that tan θ = 3.
variable (θ : ℝ)
noncomputable def tan_theta : ℝ := 3

-- Claim: tan(3 * θ) = 9/13
theorem tan_3theta :
  Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_3theta_l54_54787


namespace top_card_is_queen_probability_l54_54259

theorem top_card_is_queen_probability :
  let num_queens := 4
  let total_cards := 52
  let prob := num_queens / total_cards
  prob = 1 / 13 :=
by 
  sorry

end top_card_is_queen_probability_l54_54259


namespace sufficient_not_necessary_condition_l54_54755

variable (a : ℝ)

theorem sufficient_not_necessary_condition (h1 : a > 2) : (1 / a < 1 / 2) ↔ (a > 2 ∨ a < 0) :=
by
  sorry

end sufficient_not_necessary_condition_l54_54755


namespace g_of_neg_two_l54_54509

def f (x : ℝ) : ℝ := 4 * x - 9

def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_of_neg_two : g (-2) = 227 / 16 :=
by
  sorry

end g_of_neg_two_l54_54509


namespace acres_of_flax_l54_54716

-- Let F be the number of acres of flax
variable (F : ℕ)

-- Condition: The total farm size is 240 acres
def total_farm_size (F : ℕ) := F + (F + 80) = 240

-- Proof statement
theorem acres_of_flax (h : total_farm_size F) : F = 80 :=
sorry

end acres_of_flax_l54_54716


namespace solve_fraction_eq_l54_54524

theorem solve_fraction_eq (x : ℝ) 
  (h₁ : x ≠ -9) 
  (h₂ : x ≠ -7) 
  (h₃ : x ≠ -10) 
  (h₄ : x ≠ -6) 
  (h₅ : 1 / (x + 9) + 1 / (x + 7) = 1 / (x + 10) + 1 / (x + 6)) : 
  x = -8 := 
sorry

end solve_fraction_eq_l54_54524


namespace smallest_positive_period_f_max_min_f_on_interval_l54_54304

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi / 6) + 1

theorem smallest_positive_period_f : 
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ Real.pi) :=
sorry

theorem max_min_f_on_interval :
  let a := Real.pi / 4
  let b := 2 * Real.pi / 3
  ∃ M m, (∀ x, a ≤ x ∧ x ≤ b → f x ≤ M ∧ f x ≥ m) ∧ (M = 2) ∧ (m = -1) :=
sorry

end smallest_positive_period_f_max_min_f_on_interval_l54_54304


namespace diff_of_squares_535_465_l54_54406

theorem diff_of_squares_535_465 : (535^2 - 465^2) = 70000 :=
sorry

end diff_of_squares_535_465_l54_54406


namespace mental_math_quiz_l54_54223

theorem mental_math_quiz : ∃ (q_i q_c : ℕ), q_c + q_i = 100 ∧ 10 * q_c - 5 * q_i = 850 ∧ q_i = 10 :=
by
  sorry

end mental_math_quiz_l54_54223


namespace red_star_team_wins_l54_54031

theorem red_star_team_wins (x y : ℕ) (h1 : x + y = 9) (h2 : 3 * x + y = 23) : x = 7 := by
  sorry

end red_star_team_wins_l54_54031


namespace convert_spherical_to_rectangular_correct_l54_54453

-- Define the spherical to rectangular conversion functions
noncomputable def spherical_to_rectangular (rho θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin φ * Real.cos θ, rho * Real.sin φ * Real.sin θ, rho * Real.cos φ)

-- Define the given spherical coordinates
def given_spherical_coords : ℝ × ℝ × ℝ :=
  (5, 7 * Real.pi / 4, Real.pi / 3)

-- Define the expected rectangular coordinates
def expected_rectangular_coords : ℝ × ℝ × ℝ :=
  (-5 * Real.sqrt 6 / 4, -5 * Real.sqrt 6 / 4, 5 / 2)

-- The proof statement
theorem convert_spherical_to_rectangular_correct (ρ θ φ : ℝ)
  (h_ρ : ρ = 5) (h_θ : θ = 7 * Real.pi / 4) (h_φ : φ = Real.pi / 3) :
  spherical_to_rectangular ρ θ φ = expected_rectangular_coords :=
by
  -- Proof omitted
  sorry

end convert_spherical_to_rectangular_correct_l54_54453


namespace base_k_eq_26_l54_54019

theorem base_k_eq_26 (k : ℕ) (h : 3 * k + 2 = 26) : k = 8 :=
by {
  -- The actual proof will go here.
  sorry
}

end base_k_eq_26_l54_54019


namespace sum_of_three_numbers_l54_54224

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : ab + bc + ca = 131) : 
  a + b + c = 20 := 
by sorry

end sum_of_three_numbers_l54_54224


namespace y_neither_directly_nor_inversely_proportional_l54_54588

theorem y_neither_directly_nor_inversely_proportional (x y : ℝ) :
  ¬((∃ k : ℝ, x = k * y) ∨ (∃ k : ℝ, x * y = k)) ↔ 2 * x + 3 * y = 6 :=
by 
  sorry

end y_neither_directly_nor_inversely_proportional_l54_54588


namespace same_terminal_side_l54_54994

theorem same_terminal_side (k : ℤ) : ∃ k : ℤ, (2 * k * Real.pi - Real.pi / 6) = 11 * Real.pi / 6 := by
  sorry

end same_terminal_side_l54_54994


namespace mary_income_percentage_l54_54514

-- Declare noncomputable as necessary
noncomputable def calculate_percentage_more
    (J : ℝ) -- Juan's income
    (T : ℝ) (M : ℝ)
    (hT : T = 0.70 * J) -- Tim's income is 30% less than Juan's income
    (hM : M = 1.12 * J) -- Mary's income is 112% of Juan's income
    : ℝ :=
  ((M - T) / T) * 100

theorem mary_income_percentage
    (J T M : ℝ)
    (hT : T = 0.70 * J)
    (hM : M = 1.12 * J) :
    calculate_percentage_more J T M hT hM = 60 :=
by sorry

end mary_income_percentage_l54_54514


namespace distance_between_foci_of_hyperbola_l54_54748

theorem distance_between_foci_of_hyperbola :
  (let a_squared := 32
       b_squared := 8
       c_squared := a_squared + b_squared
       c := Real.sqrt c_squared
       distance := 2 * c
   in distance = 4 * Real.sqrt 10) :=
by
  sorry

end distance_between_foci_of_hyperbola_l54_54748


namespace fifteen_balls_three_boxes_l54_54053

theorem fifteen_balls_three_boxes :
  ∃ (ways : ℕ), ways = 91 ∧
    (∀ (x y z : ℕ), x ≥ 1 ∧ y ≥ 2 ∧ z ≥ 3 ∧ x + y + z = 15 → ways = 91) :=
begin
  existsi 91,
  split,
  { refl },
  { intros x y z,
    sorry }
end

end fifteen_balls_three_boxes_l54_54053


namespace ab_bc_ca_max_le_l54_54961

theorem ab_bc_ca_max_le (a b c : ℝ) :
  ab + bc + ca + max (abs (a - b)) (max (abs (b - c)) (abs (c - a))) ≤
  1 + (1 / 3) * (a + b + c)^2 :=
sorry

end ab_bc_ca_max_le_l54_54961


namespace distance_circle_center_to_line_l54_54944

def line := {p : ℝ × ℝ | p.1 + p.2 = 6}
def circle_center := (0, 2)
def distance_from_center_to_line := Real.sqrt ((circle_center.1 + circle_center.2 - 6)^2 / ((1 : ℝ)^2 + (1 : ℝ)^2))

theorem distance_circle_center_to_line : distance_from_center_to_line = 2 * Real.sqrt 2 := by
  sorry

end distance_circle_center_to_line_l54_54944


namespace initial_people_count_l54_54268

-- Definitions from conditions
def initial_people (W : ℕ) : ℕ := W
def net_increase : ℕ := 5 - 2
def current_people : ℕ := 19

-- Theorem to prove: initial_people == 16 given conditions
theorem initial_people_count (W : ℕ) (h1 : W + net_increase = current_people) : initial_people W = 16 :=
by
  sorry

end initial_people_count_l54_54268


namespace barium_oxide_moles_l54_54747

noncomputable def moles_of_bao_needed (mass_H2O : ℝ) (molar_mass_H2O : ℝ) : ℝ :=
  mass_H2O / molar_mass_H2O

theorem barium_oxide_moles :
  moles_of_bao_needed 54 18.015 = 3 :=
by
  unfold moles_of_bao_needed
  norm_num
  sorry

end barium_oxide_moles_l54_54747


namespace no_real_solution_exists_l54_54174

theorem no_real_solution_exists:
  ¬ ∃ (x y z : ℝ), (x ^ 2 + 4 * y * z + 2 * z = 0) ∧
                   (x + 2 * x * y + 2 * z ^ 2 = 0) ∧
                   (2 * x * z + y ^ 2 + y + 1 = 0) :=
by
  sorry

end no_real_solution_exists_l54_54174


namespace perpendicular_vectors_x_value_l54_54010

theorem perpendicular_vectors_x_value 
  (x : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (x, -1)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : x = 2 :=
by
  sorry

end perpendicular_vectors_x_value_l54_54010


namespace non_upgraded_sensor_ratio_l54_54256

theorem non_upgraded_sensor_ratio 
  (N U S : ℕ) 
  (units : ℕ := 24) 
  (fraction_upgraded : ℚ := 1 / 7) 
  (fraction_non_upgraded : ℚ := 6 / 7)
  (h1 : U / S = fraction_upgraded)
  (h2 : units * N = (fraction_non_upgraded * S)) : 
  N / U = 1 / 4 := 
by 
  sorry

end non_upgraded_sensor_ratio_l54_54256


namespace triangle_inequality_not_true_l54_54475

theorem triangle_inequality_not_true (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : ¬ (b + c > 2 * a) :=
by {
  -- assume (b + c > 2 * a)
  -- we need to reach a contradiction
  sorry
}

end triangle_inequality_not_true_l54_54475


namespace shaded_area_of_rectangle_l54_54033

theorem shaded_area_of_rectangle :
  let length := 5   -- Length of the rectangle in cm
  let width := 12   -- Width of the rectangle in cm
  let base := 2     -- Base of each triangle in cm
  let height := 5   -- Height of each triangle in cm
  let rect_area := length * width
  let triangle_area := (1 / 2) * base * height
  let unshaded_area := 2 * triangle_area
  let shaded_area := rect_area - unshaded_area
  shaded_area = 50 :=
by
  -- Calculation follows solution steps.
  sorry

end shaded_area_of_rectangle_l54_54033


namespace max_food_cost_l54_54416

theorem max_food_cost (total_cost : ℝ) (food_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (max_allowable : ℝ)
  (h1 : tax_rate = 0.07) (h2 : tip_rate = 0.15) (h3 : max_allowable = 75) (h4 : total_cost = food_cost * (1 + tax_rate + tip_rate)) :
  food_cost ≤ 61.48 :=
sorry

end max_food_cost_l54_54416


namespace max_least_integer_l54_54675

theorem max_least_integer (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 2160) (h_order : x ≤ y ∧ y ≤ z) : x ≤ 10 :=
by
  sorry

end max_least_integer_l54_54675


namespace company_bought_gravel_l54_54714

def weight_of_gravel (total_weight_of_materials : ℝ) (weight_of_sand : ℝ) : ℝ :=
  total_weight_of_materials - weight_of_sand

theorem company_bought_gravel :
  weight_of_gravel 14.02 8.11 = 5.91 := 
by
  sorry

end company_bought_gravel_l54_54714


namespace unique_n_divisible_by_210_l54_54311

theorem unique_n_divisible_by_210 
  (n : ℤ)
  (h1 : 1 ≤ n) 
  (h2 : n ≤ 210)
  (h3 : ∀ k, 1 ≤ k ∧ k ≤ 2013 → (n + k.factorial) % 210 = 0) :
  n = 1 := 
by 
  sorry

end unique_n_divisible_by_210_l54_54311


namespace average_of_three_l54_54671

theorem average_of_three (y : ℝ) (h : (15 + 24 + y) / 3 = 20) : y = 21 :=
by
  sorry

end average_of_three_l54_54671


namespace basketball_game_l54_54634

/-- Given the conditions of the basketball game:
  * a, ar, ar^2, ar^3 form the Dragons' scores
  * b, b + d, b + 2d, b + 3d form the Lions' scores
  * The game was tied at halftime: a + ar = b + (b + d)
  * The Dragons won by three points at the end: a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3
  * Neither team scored more than 100 points
Prove that the total number of points scored by the two teams in the first half is 30.
-/
theorem basketball_game (a r b d : ℕ) (h1 : a + a * r = b + (b + d))
  (h2 : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3)
  (h3 : a * (1 + r + r^2 + r^3) < 100)
  (h4 : 4 * b + 6 * d < 100) :
  a + a * r + b + (b + d) = 30 :=
by
  sorry

end basketball_game_l54_54634


namespace sean_total_apples_l54_54054

-- Define initial apples
def initial_apples : Nat := 9

-- Define the number of apples Susan gives each day
def apples_per_day : Nat := 8

-- Define the number of days Susan gives apples
def number_of_days : Nat := 5

-- Calculate total apples given by Susan
def total_apples_given : Nat := apples_per_day * number_of_days

-- Define the final total apples
def total_apples : Nat := initial_apples + total_apples_given

-- Prove the number of total apples is 49
theorem sean_total_apples : total_apples = 49 := by
  sorry

end sean_total_apples_l54_54054


namespace remainder_17_pow_1499_mod_23_l54_54872

theorem remainder_17_pow_1499_mod_23 : (17 ^ 1499) % 23 = 11 :=
by
  sorry

end remainder_17_pow_1499_mod_23_l54_54872


namespace difference_of_quarters_l54_54170

variables (n d q : ℕ)

theorem difference_of_quarters :
  (n + d + q = 150) ∧ (5 * n + 10 * d + 25 * q = 1425) →
  (∃ qmin qmax : ℕ, q = qmax - qmin ∧ qmax - qmin = 30) :=
by
  sorry

end difference_of_quarters_l54_54170


namespace ticket_cost_difference_l54_54516

noncomputable def total_cost_adults (tickets : ℕ) (price : ℝ) : ℝ := tickets * price
noncomputable def total_cost_children (tickets : ℕ) (price : ℝ) : ℝ := tickets * price
noncomputable def total_tickets (adults : ℕ) (children : ℕ) : ℕ := adults + children
noncomputable def discount (threshold : ℕ) (discount_rate : ℝ) (cost : ℝ) (tickets : ℕ) : ℝ :=
  if tickets > threshold then cost * discount_rate else 0
noncomputable def final_cost (initial_cost : ℝ) (discount : ℝ) : ℝ := initial_cost - discount
noncomputable def proportional_discount (partial_cost : ℝ) (total_cost : ℝ) (total_discount : ℝ) : ℝ :=
  (partial_cost / total_cost) * total_discount
noncomputable def difference (cost1 : ℝ) (cost2 : ℝ) : ℝ := cost1 - cost2

theorem ticket_cost_difference :
  let adult_tickets := 9
  let children_tickets := 7
  let adult_price := 11
  let children_price := 7
  let discount_rate := 0.15
  let discount_threshold := 10
  let total_adult_cost := total_cost_adults adult_tickets adult_price
  let total_children_cost := total_cost_children children_tickets children_price
  let all_tickets := total_tickets adult_tickets children_tickets
  let initial_total_cost := total_adult_cost + total_children_cost
  let total_discount := discount discount_threshold discount_rate initial_total_cost all_tickets
  let final_total_cost := final_cost initial_total_cost total_discount
  let adult_discount := proportional_discount total_adult_cost initial_total_cost total_discount
  let children_discount := proportional_discount total_children_cost initial_total_cost total_discount
  let final_adult_cost := final_cost total_adult_cost adult_discount
  let final_children_cost := final_cost total_children_cost children_discount
  difference final_adult_cost final_children_cost = 42.52 := by
  sorry

end ticket_cost_difference_l54_54516


namespace remainder_is_20_l54_54721

def N := 220020
def a := 555
def b := 445
def d := a + b
def q := 2 * (a - b)

theorem remainder_is_20 : N % d = 20 := by
  sorry

end remainder_is_20_l54_54721


namespace saturday_price_is_correct_l54_54168

-- Define Thursday's price
def thursday_price : ℝ := 50

-- Define the price increase rate on Friday
def friday_increase_rate : ℝ := 0.2

-- Define the discount rate on Saturday
def saturday_discount_rate : ℝ := 0.15

-- Calculate the price on Friday
def friday_price : ℝ := thursday_price * (1 + friday_increase_rate)

-- Calculate the discount amount on Saturday
def saturday_discount : ℝ := friday_price * saturday_discount_rate

-- Calculate the price on Saturday
def saturday_price : ℝ := friday_price - saturday_discount

-- Theorem stating the price on Saturday
theorem saturday_price_is_correct : saturday_price = 51 := by
  -- Definitions are already embedded into the conditions
  -- so here we only state the property to be proved.
  sorry

end saturday_price_is_correct_l54_54168


namespace polar_to_rectangular_l54_54903

theorem polar_to_rectangular (r θ : ℝ) (x y : ℝ) 
  (hr : r = 10) 
  (hθ : θ = (3 * Real.pi) / 4) 
  (hx : x = r * Real.cos θ) 
  (hy : y = r * Real.sin θ) 
  :
  x = -5 * Real.sqrt 2 ∧ y = 5 * Real.sqrt 2 := 
by
  -- We assume that the problem is properly stated
  -- Proof omitted here
  sorry

end polar_to_rectangular_l54_54903


namespace distinct_schedules_l54_54741

-- Define the problem setting and assumptions
def subjects := ["Chinese", "Mathematics", "Politics", "English", "Physical Education", "Art"]

-- Given conditions
def math_in_first_three_periods (schedule : List String) : Prop :=
  ∃ k, (k < 3) ∧ (schedule.get! k = "Mathematics")

def english_not_in_sixth_period (schedule : List String) : Prop :=
  schedule.get! 5 ≠ "English"

-- Define the proof problem
theorem distinct_schedules : 
  ∃! (schedules : List (List String)), 
  (∀ schedule ∈ schedules, 
    math_in_first_three_periods schedule ∧ 
    english_not_in_sixth_period schedule) ∧
  schedules.length = 288 :=
by
  sorry

end distinct_schedules_l54_54741


namespace person_B_work_days_l54_54686

-- Let a be the work rate for person A, and b be the work rate for person B.
-- a completes the work in 20 days
-- b completes the work in x days
-- When working together, a and b complete 0.375 of the work in 5 days


theorem person_B_work_days (x : ℝ) :
  ((5 : ℝ) * ((1 / 20) + 1 / x) = 0.375) -> x = 40 := 
by 
  sorry

end person_B_work_days_l54_54686


namespace hall_100_guests_67_friends_find_clique_l54_54636

theorem hall_100_guests_67_friends_find_clique :
  ∀ (P : Fin 100 → Fin 100 → Prop) (n : Fin 100),
    (∀ i : Fin 100, ∃ S : Finset (Fin 100), (S.card ≥ 67) ∧ (∀ j ∈ S, P i j)) →
    (∃ (A B C D : Fin 100), P A B ∧ P A C ∧ P A D ∧ P B C ∧ P B D ∧ P C D) :=
by
  sorry

end hall_100_guests_67_friends_find_clique_l54_54636


namespace average_speed_of_bike_l54_54247

theorem average_speed_of_bike (distance : ℕ) (time : ℕ) (h1 : distance = 21) (h2 : time = 7) : distance / time = 3 := by
  sorry

end average_speed_of_bike_l54_54247


namespace students_registered_for_course_l54_54668

theorem students_registered_for_course :
  ∀ (students_present_yesterday students_absent_today: ℕ),
    students_present_yesterday = 70 →
    students_absent_today = 30 →
    let students_attended_today := 0.9 * 2 * students_present_yesterday in
    students_registered = students_attended_today + students_absent_today →
    students_registered = 156 :=
by
  intros students_present_yesterday students_absent_today h1 h2 h3
  rw [h1, h2, h3]
  sorry

end students_registered_for_course_l54_54668


namespace total_hours_charged_l54_54517

variable (K P M : ℕ)

theorem total_hours_charged (h1 : P = 2 * K) (h2 : P = M / 3) (h3 : M = K + 80) : K + P + M = 144 := 
by
  sorry

end total_hours_charged_l54_54517


namespace math_problem_l54_54008

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l54_54008


namespace square_of_chord_length_l54_54075

/--
Given two circles with radii 10 and 7, and centers 15 units apart, if they intersect at a point P such that the chords QP and PR are of equal lengths, then the square of the length of chord QP is 289.
-/
theorem square_of_chord_length :
  ∀ (r1 r2 d x : ℝ), r1 = 10 → r2 = 7 → d = 15 →
  let cos_theta1 := (x^2 + r1^2 - r2^2) / (2 * r1 * x)
  let cos_theta2 := (x^2 + r2^2 - r1^2) / (2 * r2 * x)
  cos_theta1 = cos_theta2 →
  x^2 = 289 := 
by
  intros r1 r2 d x h1 h2 h3
  let cos_theta1 := (x^2 + r1^2 - r2^2) / (2 * r1 * x)
  let cos_theta2 := (x^2 + r2^2 - r1^2) / (2 * r2 * x)
  intro h4
  sorry

end square_of_chord_length_l54_54075


namespace art_gallery_total_pieces_l54_54731

theorem art_gallery_total_pieces :
  ∃ T : ℕ, 
    (1/3 : ℝ) * T + (2/3 : ℝ) * (1/3 : ℝ) * T + 400 + 3 * (1/18 : ℝ) * T + 2 * (1/18 : ℝ) * T = T :=
sorry

end art_gallery_total_pieces_l54_54731


namespace brick_laying_days_l54_54016

theorem brick_laying_days (a m n d : ℕ) (hm : 0 < m) (hn : 0 < n) (hd : 0 < d) :
  let rate_M := m / (a * d)
  let rate_N := n / (a * (2 * d))
  let total_days := 3 * a^2 / (m + n)
  (a * rate_M * (d * total_days) + 2 * a * rate_N * (d * total_days)) = (a + 2 * a) :=
by
  -- Definitions from the problem conditions
  let rate_M := m / (a * d)
  let rate_N := n / (a * (2 * d))
  let total_days := 3 * a^2 / (m + n)
  have h0 : a * rate_M * (d * total_days) = a := sorry
  have h1 : 2 * a * rate_N * (d * total_days) = 2 * a := sorry
  exact sorry

end brick_laying_days_l54_54016


namespace find_t_eq_l54_54344

variable (a V V_0 S t : ℝ)

theorem find_t_eq (h1 : V = a * t + V_0) (h2 : S = (1/3) * a * t^3 + V_0 * t) : t = (V - V_0) / a :=
sorry

end find_t_eq_l54_54344


namespace eq_neg2_multi_l54_54081

theorem eq_neg2_multi {m n : ℝ} (h : m = n) : -2 * m = -2 * n :=
by sorry

end eq_neg2_multi_l54_54081


namespace Jason_4week_visits_l54_54648

-- Definitions
def William_weekly_visits : ℕ := 2
def Jason_weekly_multiplier : ℕ := 4
def weeks_period : ℕ := 4

-- We need to prove that Jason goes to the library 32 times in 4 weeks.
theorem Jason_4week_visits : William_weekly_visits * Jason_weekly_multiplier * weeks_period = 32 := 
by sorry

end Jason_4week_visits_l54_54648


namespace david_more_pushups_than_zachary_l54_54241

def zacharyPushUps : ℕ := 59
def davidPushUps : ℕ := 78

theorem david_more_pushups_than_zachary :
  davidPushUps - zacharyPushUps = 19 :=
by
  sorry

end david_more_pushups_than_zachary_l54_54241


namespace polyhedron_equation_l54_54593

variables (V E F H T : ℕ)

-- Euler's formula for convex polyhedra
axiom euler_formula : V - E + F = 2
-- Number of faces is 50, and each face is either a triangle or a hexagon
axiom faces_count : F = 50
-- At each vertex, 3 triangles and 2 hexagons meet
axiom triangles_meeting : T = 3
axiom hexagons_meeting : H = 2

-- Prove that 100H + 10T + V = 230
theorem polyhedron_equation : 100 * H + 10 * T + V = 230 :=
  sorry

end polyhedron_equation_l54_54593


namespace real_nums_inequality_l54_54839

theorem real_nums_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : a ^ 2000 + b ^ 2000 = a ^ 1998 + b ^ 1998) :
  a ^ 2 + b ^ 2 ≤ 2 :=
sorry

end real_nums_inequality_l54_54839


namespace sum_of_decimals_l54_54582

theorem sum_of_decimals :
  (2 / 100 : ℝ) + (5 / 1000) + (8 / 10000) + (6 / 100000) = 0.02586 :=
by
  sorry

end sum_of_decimals_l54_54582


namespace math_problem_l54_54381

theorem math_problem (c d : ℝ) (hc : c^2 - 6 * c + 15 = 27) (hd : d^2 - 6 * d + 15 = 27) (h_cd : c ≥ d) : 
  3 * c + 2 * d = 15 + Real.sqrt 21 :=
by
  sorry

end math_problem_l54_54381


namespace triangle_area_l54_54148

noncomputable def area_of_triangle (a b c α β γ : ℝ) :=
  (1 / 2) * a * b * Real.sin γ

theorem triangle_area 
  (a b c A B C : ℝ)
  (h1 : b * Real.cos C = 3 * a * Real.cos B - c * Real.cos B)
  (h2 : (a * b * Real.cos C) / (a * b) = 2) :
  area_of_triangle a b c A B C = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_area_l54_54148


namespace sum_first_seven_terms_geometric_sequence_l54_54753

noncomputable def sum_geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := 
  a * (1 - r^n) / (1 - r)

theorem sum_first_seven_terms_geometric_sequence : 
  sum_geometric_sequence (1/4) (1/4) 7 = 16383 / 49152 := 
by
  sorry

end sum_first_seven_terms_geometric_sequence_l54_54753


namespace Victor_can_carry_7_trays_at_a_time_l54_54553

-- Define the conditions
def trays_from_first_table : Nat := 23
def trays_from_second_table : Nat := 5
def number_of_trips : Nat := 4

-- Define the total number of trays
def total_trays : Nat := trays_from_first_table + trays_from_second_table

-- Prove that the number of trays Victor can carry at a time is 7
theorem Victor_can_carry_7_trays_at_a_time :
  total_trays / number_of_trips = 7 :=
by
  sorry

end Victor_can_carry_7_trays_at_a_time_l54_54553


namespace bees_multiple_l54_54966

theorem bees_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 :=
by
  sorry

end bees_multiple_l54_54966


namespace socks_choice_count_l54_54143

variable (white_socks : ℕ) (brown_socks : ℕ) (blue_socks : ℕ) (black_socks : ℕ)

theorem socks_choice_count :
  white_socks = 5 →
  brown_socks = 4 →
  blue_socks = 2 →
  black_socks = 2 →
  (white_socks.choose 2) + (brown_socks.choose 2) + (blue_socks.choose 2) + (black_socks.choose 2) = 18 :=
by
  -- Here the proof would be elaborated
  sorry

end socks_choice_count_l54_54143


namespace delta_value_l54_54322

theorem delta_value (Δ : ℝ) (h : 4 * 3 = Δ - 6) : Δ = 18 :=
sorry

end delta_value_l54_54322


namespace range_of_n_l54_54926

noncomputable def parabola (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + b

variable {a b n y1 y2 : ℝ}

theorem range_of_n (h_a : a > 0) 
  (hA : parabola a b (2*n + 3) = y1) 
  (hB : parabola a b (n - 1) = y2)
  (h_sym : y1 < y2) 
  (h_opposite_sides : (2*n + 3 - 1) * (n - 1 - 1) < 0) :
  -1 < n ∧ n < 0 :=
sorry

end range_of_n_l54_54926


namespace fold_paper_crease_length_l54_54916

theorem fold_paper_crease_length 
    (w l : ℝ) (w_pos : w = 12) (l_pos : l = 16) 
    (F G : ℝ × ℝ) (F_on_AD : F = (0, 12))
    (G_on_BC : G = (16, 12)) :
    dist F G = 20 := 
by
  sorry

end fold_paper_crease_length_l54_54916


namespace minimum_participants_l54_54192

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l54_54192


namespace derek_alice_pair_l54_54740

-- Variables and expressions involved
variable (x b c : ℝ)

-- Definitions of the conditions
def derek_eq := |x + 3| = 5 
def alice_eq := ∀ a, (a - 2) * (a + 8) = a^2 + b * a + c

-- The theorem to prove
theorem derek_alice_pair : derek_eq x → alice_eq b c → (b, c) = (6, -16) :=
by
  intros h1 h2
  sorry

end derek_alice_pair_l54_54740


namespace largest_integer_divisor_l54_54234

theorem largest_integer_divisor (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
sorry

end largest_integer_divisor_l54_54234


namespace hal_paul_difference_l54_54139

def halAnswer : Int := 12 - (3 * 2) + 4
def paulAnswer : Int := (12 - 3) * 2 + 4

theorem hal_paul_difference :
  halAnswer - paulAnswer = -12 := by
  sorry

end hal_paul_difference_l54_54139


namespace range_of_a_l54_54331

theorem range_of_a (a : ℝ) (h : ¬ ∃ x0 : ℝ, a * x0^2 - a * x0 - 2 ≥ 0) 
: a ∈ Icc (-8 : ℝ) 0 := sorry

end range_of_a_l54_54331


namespace original_price_l54_54265

theorem original_price (x : ℝ) (h : 0.9504 * x = 108) : x = 10800 / 9504 :=
by
  sorry

end original_price_l54_54265


namespace rectangle_area_error_l54_54440

theorem rectangle_area_error
  (L W : ℝ)
  (measured_length : ℝ := 1.15 * L)
  (measured_width : ℝ := 1.20 * W)
  (true_area : ℝ := L * W)
  (measured_area : ℝ := measured_length * measured_width)
  (percentage_error : ℝ := ((measured_area - true_area) / true_area) * 100) :
  percentage_error = 38 :=
by
  sorry

end rectangle_area_error_l54_54440


namespace work_done_isothermal_l54_54049

variable (n : ℕ) (R T : ℝ) (P DeltaV : ℝ)

-- Definitions based on the conditions
def isobaric_work (P DeltaV : ℝ) := P * DeltaV

noncomputable def isobaric_heat (P DeltaV : ℝ) : ℝ :=
  (5 / 2) * P * DeltaV

noncomputable def isothermal_work (Q_iso : ℝ) : ℝ := Q_iso

theorem work_done_isothermal :
  ∃ (n R : ℝ) (P DeltaV : ℝ),
    isobaric_work P DeltaV = 20 ∧
    isothermal_work (isobaric_heat P DeltaV) = 50 :=
by 
  sorry

end work_done_isothermal_l54_54049


namespace investment_amount_correct_l54_54696

-- Lean statement definitions based on conditions
def cost_per_tshirt : ℕ := 3
def selling_price_per_tshirt : ℕ := 20
def tshirts_sold : ℕ := 83
def total_revenue : ℕ := tshirts_sold * selling_price_per_tshirt
def total_cost_of_tshirts : ℕ := tshirts_sold * cost_per_tshirt
def investment_in_equipment : ℕ := total_revenue - total_cost_of_tshirts

-- Theorem statement
theorem investment_amount_correct : investment_in_equipment = 1411 := by
  sorry

end investment_amount_correct_l54_54696


namespace friday_can_determine_arrival_date_l54_54811

-- Define the conditions
def Robinson_crusoe (day : ℕ) : Prop := day % 365 = 0

-- Goal: Within 183 days, Friday can determine his arrival date.
theorem friday_can_determine_arrival_date : 
  (∀ day : ℕ, day < 183 → (Robinson_crusoe day ↔ ¬ Robinson_crusoe (day + 1)) ∨ (day % 365 = 0)) :=
sorry

end friday_can_determine_arrival_date_l54_54811


namespace complex_expression_is_none_of_the_above_l54_54746

-- We define the problem in Lean, stating that the given complex expression is not equal to any of the simplified forms
theorem complex_expression_is_none_of_the_above (x : ℝ) :
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x+1)^4 ) ∧
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x^3+1)^4 ) ∧
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x-1)^4 ) :=
sorry

end complex_expression_is_none_of_the_above_l54_54746


namespace no_real_solutions_iff_k_gt_4_l54_54917

theorem no_real_solutions_iff_k_gt_4 (k : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + k ≠ 0) ↔ k > 4 :=
sorry

end no_real_solutions_iff_k_gt_4_l54_54917


namespace race_participants_minimum_l54_54176

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l54_54176


namespace work_done_in_isothermal_process_l54_54048

variable (n : ℕ) (R : ℝ) (P : ℝ) (ΔV : ℝ) (Q_iso : ℝ)

-- Conditions
def ideal_monoatomic_gas : Prop := (n = 1) ∧ (P * ΔV = 20)

-- Work done in isobaric process
def W_isobaric : ℝ := P * ΔV

-- Heat added in isobaric process
def Q_isobaric : ℝ := (5 / 2) * P * ΔV

-- Heat added in isothermal process (equal to isobaric heat)
def Q_isothermal : ℝ := Q_isobaric

-- Work done in isothermal process is equal to the heat added
def W_isothermal : ℝ := Q_isothermal

-- Proposition for the proof
theorem work_done_in_isothermal_process :
  ideal_monoatomic_gas n R P ΔV →
  Q_isothermal = 50 :=
by
  intro h,
  sorry

end work_done_in_isothermal_process_l54_54048


namespace quadratic_function_min_value_l54_54387

theorem quadratic_function_min_value (x : ℝ) (y : ℝ) :
  (y = x^2 - 2 * x + 6) →
  (∃ x_min, x_min = 1 ∧ y = (1 : ℝ)^2 - 2 * (1 : ℝ) + 6 ∧ (∀ x, y ≥ x^2 - 2 * x + 6)) :=
by
  sorry

end quadratic_function_min_value_l54_54387


namespace angle_no_complement_greater_than_90_l54_54015

-- Definition of angle
def angle (A : ℝ) : Prop := 
  A = 100 + (15 / 60)

-- Definition of complement
def has_complement (A : ℝ) : Prop :=
  A < 90

-- Theorem: Angles greater than 90 degrees do not have complements
theorem angle_no_complement_greater_than_90 {A : ℝ} (h: angle A) : ¬ has_complement A :=
by sorry

end angle_no_complement_greater_than_90_l54_54015


namespace gcd_779_209_589_l54_54750

theorem gcd_779_209_589 : Int.gcd (Int.gcd 779 209) 589 = 19 := 
by 
  sorry

end gcd_779_209_589_l54_54750


namespace hamburger_cost_l54_54164

variable (H : ℝ)

theorem hamburger_cost :
  (H + 2 + 3 = 20 - 11) → (H = 4) :=
by
  sorry

end hamburger_cost_l54_54164


namespace sector_area_l54_54330

theorem sector_area (radius area : ℝ) (θ : ℝ) (h1 : 2 * radius + θ * radius = 16) (h2 : θ = 2) : area = 16 :=
  sorry

end sector_area_l54_54330


namespace minimum_participants_l54_54193

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l54_54193


namespace ratio_abc_xyz_l54_54345

theorem ratio_abc_xyz
  (a b c x y z : ℝ)
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z) 
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a * x + b * y + c * z = 56) :
  (a + b + c) / (x + y + z) = 7 / 8 := 
sorry

end ratio_abc_xyz_l54_54345


namespace frank_spent_on_mower_blades_l54_54296

def money_made := 19
def money_spent_on_games := 4 * 2
def money_left := money_made - money_spent_on_games

theorem frank_spent_on_mower_blades : money_left = 11 :=
by
  -- we are providing the proof steps here in comments, but in the actual code, it's just sorry
  -- calc money_left
  --    = money_made - money_spent_on_games : by refl
  --    = 19 - 8 : by norm_num
  --    = 11 : by norm_num
  sorry

end frank_spent_on_mower_blades_l54_54296


namespace min_overlap_percent_l54_54662

theorem min_overlap_percent
  (M S : ℝ)
  (hM : M = 0.9)
  (hS : S = 0.85) :
  ∃ x, x = 0.75 ∧ (M + S - 1 ≤ x ∧ x ≤ min M S ∧ x = M + S - 1) :=
by
  sorry

end min_overlap_percent_l54_54662


namespace simplify_sqrt_expression_l54_54667

theorem simplify_sqrt_expression :
  (3 * (Real.sqrt (4 * 3)) - 2 * (Real.sqrt (1 / 3)) +
     Real.sqrt (16 * 3)) / (2 * Real.sqrt 3) = 14 / 3 := by
sorry

end simplify_sqrt_expression_l54_54667


namespace karen_group_size_l54_54881

theorem karen_group_size (total_students : ℕ) (zack_group_size number_of_groups : ℕ) (karen_group_size : ℕ) (h1 : total_students = 70) (h2 : zack_group_size = 14) (h3 : number_of_groups = total_students / zack_group_size) (h4 : number_of_groups = total_students / karen_group_size) : karen_group_size = 14 :=
by
  sorry

end karen_group_size_l54_54881


namespace relationship_among_a_b_c_l54_54956

noncomputable def a : ℝ := (Real.sqrt 2) / 2 * (Real.sin (17 * Real.pi / 180) + Real.cos (17 * Real.pi / 180))
noncomputable def b : ℝ := 2 * (Real.cos (13 * Real.pi / 180))^2 - 1
noncomputable def c : ℝ := (Real.sqrt 3) / 2

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l54_54956


namespace joeys_votes_l54_54635

theorem joeys_votes
  (M B J : ℕ) 
  (h1 : M = 66) 
  (h2 : M = 3 * B) 
  (h3 : B = 2 * (J + 3)) : 
  J = 8 := 
by 
  sorry

end joeys_votes_l54_54635


namespace right_triangle_can_form_isosceles_l54_54759

-- Definitions for the problem
structure RightTriangle :=
  (a b : ℝ) -- The legs of the right triangle
  (c : ℝ)  -- The hypotenuse of the right triangle
  (h1 : c = Real.sqrt (a ^ 2 + b ^ 2)) -- Pythagoras theorem

-- The triangle attachment requirement definition
def IsoscelesTriangleAttachment (rightTriangle : RightTriangle) : Prop :=
  ∃ (b1 b2 : ℝ), -- Two base sides of the new triangle sharing one side with the right triangle
    (b1 ≠ b2) ∧ -- They should be different to not overlap
    (b1 = rightTriangle.a ∨ b1 = rightTriangle.b) ∧ -- Share one side with the right triangle
    (b2 ≠ rightTriangle.a ∧ b2 ≠ rightTriangle.b) ∧ -- Ensure non-overlapping
    (b1^2 + b2^2 = rightTriangle.c^2)

-- The statement to prove
theorem right_triangle_can_form_isosceles (T : RightTriangle) : IsoscelesTriangleAttachment T :=
sorry

end right_triangle_can_form_isosceles_l54_54759


namespace radius_of_sphere_l54_54252

-- Define the conditions.
def radius_wire : ℝ := 8
def length_wire : ℝ := 36

-- Given the volume of the metallic sphere is equal to the volume of the wire,
-- Prove that the radius of the sphere is 12 cm.
theorem radius_of_sphere (r_wire : ℝ) (h_wire : ℝ) (r_sphere : ℝ) : 
    r_wire = radius_wire → h_wire = length_wire →
    (π * r_wire^2 * h_wire = (4/3) * π * r_sphere^3) → 
    r_sphere = 12 :=
by
  intros h₁ h₂ h₃
  -- Add proof steps here.
  sorry

end radius_of_sphere_l54_54252


namespace tory_video_games_l54_54398

theorem tory_video_games (T J: ℕ) :
    (3 * J + 5 = 11) → (J = T / 3) → T = 6 :=
by
  sorry

end tory_video_games_l54_54398


namespace sum_of_ages_is_26_l54_54240

def Yoongi_aunt_age := 38
def Yoongi_age := Yoongi_aunt_age - 23
def Hoseok_age := Yoongi_age - 4
def sum_of_ages := Yoongi_age + Hoseok_age

theorem sum_of_ages_is_26 : sum_of_ages = 26 :=
by
  sorry

end sum_of_ages_is_26_l54_54240


namespace distinct_real_numbers_condition_l54_54954

theorem distinct_real_numbers_condition (a b c : ℝ) (h_abc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a / (b - c)) + (b / (c - a)) + (c / (a - b)) = 1) :
  (a / (b - c)^2) + (b / (c - a)^2) + (c / (a - b)^2) = 1 := 
by sorry

end distinct_real_numbers_condition_l54_54954


namespace total_amount_received_l54_54884

theorem total_amount_received (CI : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (P : ℝ) (A : ℝ) 
  (hCI : CI = P * ((1 + r / n) ^ (n * t) - 1))
  (hCI_value : CI = 370.80)
  (hr : r = 0.06)
  (hn : n = 1)
  (ht : t = 2)
  (hP : P = 3000)
  (hP_value : P = CI / 0.1236) :
  A = P + CI := 
by 
sorry

end total_amount_received_l54_54884


namespace equivalent_discount_l54_54891

variable (P d1 d2 d : ℝ)

-- Given conditions:
def original_price : ℝ := 50
def first_discount_rate : ℝ := 0.25
def second_discount_rate : ℝ := 0.10
def equivalent_single_discount_rate : ℝ := 0.325

-- Final conclusion:
theorem equivalent_discount :
  let final_price_after_first_discount := (original_price * (1 - first_discount_rate))
  let final_price_after_second_discount := (final_price_after_first_discount * (1 - second_discount_rate))
  final_price_after_second_discount = (original_price * (1 - equivalent_single_discount_rate)) :=
by
  sorry

end equivalent_discount_l54_54891


namespace non_right_triangle_option_l54_54034

-- Definitions based on conditions
def optionA (A B C : ℝ) : Prop := A + B = C
def optionB (A B C : ℝ) : Prop := A - B = C
def optionC (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ B / C = 2 / 3
def optionD (A B C : ℝ) : Prop := A = B ∧ A = 3 * C

-- Given conditions for a right triangle
def is_right_triangle (A B C : ℝ) : Prop := ∃(θ : ℝ), θ = 90 ∧ (A = θ ∨ B = θ ∨ C = θ)

-- The proof problem
theorem non_right_triangle_option (A B C : ℝ) :
  optionD A B C ∧ ¬(is_right_triangle A B C) := sorry

end non_right_triangle_option_l54_54034


namespace total_time_taken_l54_54390

theorem total_time_taken (speed_boat : ℕ) (speed_stream : ℕ) (distance : ℕ) 
    (h1 : speed_boat = 12) (h2 : speed_stream = 4) (h3 : distance = 480) : 
    ((distance / (speed_boat + speed_stream)) + (distance / (speed_boat - speed_stream)) = 90) :=
by
  -- Sorry is used to skip the proof
  sorry

end total_time_taken_l54_54390


namespace probability_of_type_A_probability_of_different_type_l54_54725

def total_questions : ℕ := 6
def type_A_questions : ℕ := 4
def type_B_questions : ℕ := 2
def select_questions : ℕ := 2

def total_combinations := Nat.choose total_questions select_questions
def type_A_combinations := Nat.choose type_A_questions select_questions
def different_type_combinations := Nat.choose type_A_questions 1 * Nat.choose type_B_questions 1

theorem probability_of_type_A : (type_A_combinations : ℚ) / total_combinations = 2 / 5 := by
  sorry

theorem probability_of_different_type : (different_type_combinations : ℚ) / total_combinations = 8 / 15 := by
  sorry

end probability_of_type_A_probability_of_different_type_l54_54725


namespace desk_height_l54_54400

variables (h l w : ℝ)

theorem desk_height
  (h_eq_2l_50 : h + 2 * l = 50)
  (h_eq_2w_40 : h + 2 * w = 40)
  (l_minus_w_eq_5 : l - w = 5) :
  h = 30 :=
by {
  sorry
}

end desk_height_l54_54400


namespace black_lambs_count_l54_54592

/-- Definition of the total number of lambs. -/
def total_lambs : Nat := 6048

/-- Definition of the number of white lambs. -/
def white_lambs : Nat := 193

/-- Prove that the number of black lambs is 5855. -/
theorem black_lambs_count : total_lambs - white_lambs = 5855 := by
  sorry

end black_lambs_count_l54_54592


namespace inequality_solution_set_l54_54604

theorem inequality_solution_set (x : ℝ) (h : x ≠ 2) :
  (1 / (x - 2) ≤ 1) ↔ (x < 2 ∨ 3 ≤ x) :=
sorry

end inequality_solution_set_l54_54604


namespace journey_distance_l54_54432

theorem journey_distance (t : ℝ) : 
  t = 20 →
  ∃ D : ℝ, (D / 20 + D / 30 = t) ∧ D = 240 :=
by
  sorry

end journey_distance_l54_54432


namespace problem_solution_l54_54909

theorem problem_solution :
  { x : ℝ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 5) } 
  = { x : ℝ | x < 0 } ∪ { x : ℝ | 1 < x ∧ x < 2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end problem_solution_l54_54909


namespace find_value_of_2_times_x_minus_y_squared_minus_3_l54_54012

-- Define the conditions as noncomputable variables
variables (x y : ℝ)

-- State the main theorem
theorem find_value_of_2_times_x_minus_y_squared_minus_3 :
  (x^2 - x*y = 12) →
  (y^2 - y*x = 15) →
  2 * (x - y)^2 - 3 = 51 :=
by
  intros h1 h2
  sorry

end find_value_of_2_times_x_minus_y_squared_minus_3_l54_54012


namespace max_total_balls_l54_54901

theorem max_total_balls
  (r₁ : ℕ := 89)
  (t₁ : ℕ := 90)
  (r₂ : ℕ := 8)
  (t₂ : ℕ := 9)
  (y : ℕ)
  (h₁ : t₁ > 0)
  (h₂ : t₂ > 0)
  (h₃ : 92 ≤ (r₁ + r₂ * y) * 100 / (t₁ + t₂ * y))
  : y ≤ 22 → 90 + 9 * y = 288 :=
by sorry

end max_total_balls_l54_54901


namespace eccentricity_of_ellipse_l54_54897

noncomputable def problem_conditions (a b c e : ℝ) :=
  a > b ∧ b > 0 ∧
  c = real.sqrt (a^2 - b^2) ∧
  (∃ (x y: ℝ), x^2 / a^2 + y^2 / b^2 = 1) ∧
  ∃ n, isosceles_obtuse_triangle (c, 0) (14 * a^2 / (9 * c), 0) n

theorem eccentricity_of_ellipse (a b c e : ℝ) (h : problem_conditions a b c e) :
  e = real.sqrt (1 - (b^2 / a^2)) :=
begin
  sorry
end

end eccentricity_of_ellipse_l54_54897


namespace ratio_of_costs_l54_54645

-- Definitions based on conditions
def quilt_length : Nat := 16
def quilt_width : Nat := 20
def patch_area : Nat := 4
def first_10_patch_cost : Nat := 10
def total_cost : Nat := 450

-- Theorem we need to prove
theorem ratio_of_costs : (total_cost - 10 * first_10_patch_cost) / (10 * first_10_patch_cost) = 7 / 2 := by
  sorry

end ratio_of_costs_l54_54645


namespace expression_value_l54_54001

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l54_54001


namespace solve_system_of_equations_l54_54682

-- Conditions from the problem
variables (x y : ℚ)

-- Definitions (the original equations)
def equation1 := x + 2 * y = 3
def equation2 := 9 * x - 8 * y = 5

-- Correct answer
def solution_x := 17 / 13
def solution_y := 11 / 13

-- The final proof statement
theorem solve_system_of_equations (h1 : equation1 solution_x solution_y) (h2 : equation2 solution_x solution_y) :
  x = solution_x ∧ y = solution_y := sorry

end solve_system_of_equations_l54_54682


namespace cos_seven_pi_over_six_l54_54459

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := 
by
  sorry

end cos_seven_pi_over_six_l54_54459


namespace charlie_cortland_apples_l54_54457

/-- Given that Charlie picked 0.17 bags of Golden Delicious apples, 0.17 bags of Macintosh apples, 
   and a total of 0.67 bags of fruit, prove that the number of bags of Cortland apples picked by Charlie is 0.33. -/
theorem charlie_cortland_apples :
  let golden_delicious := 0.17
  let macintosh := 0.17
  let total_fruit := 0.67
  total_fruit - (golden_delicious + macintosh) = 0.33 :=
by
  sorry

end charlie_cortland_apples_l54_54457


namespace total_votes_cast_l54_54808

/-- Define the conditions for Elvis's votes and percentage representation -/
def elvis_votes : ℕ := 45
def percentage_representation : ℚ := 1 / 4

/-- The main theorem that proves the total number of votes cast -/
theorem total_votes_cast : (elvis_votes: ℚ) / percentage_representation = 180 := by
  sorry

end total_votes_cast_l54_54808


namespace batsman_highest_score_l54_54702

theorem batsman_highest_score (H L : ℕ) 
  (h₁ : (40 * 50 = 2000)) 
  (h₂ : (H = L + 172))
  (h₃ : (38 * 48 = 1824)) :
  (2000 = 1824 + H + L) → H = 174 :=
by 
  sorry

end batsman_highest_score_l54_54702


namespace double_neg_cancel_l54_54365

theorem double_neg_cancel (a : ℤ) : - (-2) = 2 :=
sorry

end double_neg_cancel_l54_54365


namespace tan3theta_l54_54796

theorem tan3theta (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 := 
by
  sorry

end tan3theta_l54_54796


namespace ceil_neg_sqrt_frac_l54_54118

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := 
sorry

end ceil_neg_sqrt_frac_l54_54118


namespace math_problem_l54_54380

theorem math_problem (c d : ℝ) (hc : c^2 - 6 * c + 15 = 27) (hd : d^2 - 6 * d + 15 = 27) (h_cd : c ≥ d) : 
  3 * c + 2 * d = 15 + Real.sqrt 21 :=
by
  sorry

end math_problem_l54_54380


namespace total_pencils_given_out_l54_54394

theorem total_pencils_given_out (n p : ℕ) (h1 : n = 10) (h2 : p = 5) : n * p = 50 :=
by
  sorry

end total_pencils_given_out_l54_54394


namespace braids_each_dancer_l54_54501

-- Define the conditions
def num_dancers := 8
def time_per_braid := 30 -- seconds per braid
def total_time := 20 * 60 -- convert 20 minutes into seconds

-- Define the total number of braids Jill makes
def total_braids := total_time / time_per_braid

-- Define the number of braids per dancer
def braids_per_dancer := total_braids / num_dancers

-- Theorem: Prove that each dancer has 5 braids
theorem braids_each_dancer : braids_per_dancer = 5 := 
by sorry

end braids_each_dancer_l54_54501


namespace find_c_l54_54774

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : 
  c = 7 := 
by {
  sorry
}

end find_c_l54_54774


namespace gcd_71_19_l54_54854

theorem gcd_71_19 : Int.gcd 71 19 = 1 := by
  sorry

end gcd_71_19_l54_54854


namespace poster_height_proportion_l54_54254

-- Defining the given conditions
def original_width : ℕ := 3
def original_height : ℕ := 2
def new_width : ℕ := 12
def scale_factor := new_width / original_width

-- The statement to prove the new height
theorem poster_height_proportion :
  scale_factor = 4 → (original_height * scale_factor) = 8 :=
by
  sorry

end poster_height_proportion_l54_54254


namespace three_digit_number_count_l54_54126

theorem three_digit_number_count :
  ∃ n : ℕ, n = 15 ∧
  (∀ a b c : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) →
    (100 * a + 10 * b + c = 37 * (a + b + c) → ∃ k : ℕ, k = n)) :=
sorry

end three_digit_number_count_l54_54126


namespace initial_men_in_fort_l54_54091

theorem initial_men_in_fort (M : ℕ) 
  (h1 : ∀ N : ℕ, M * 35 = (N - 25) * 42) 
  (h2 : 10 + 42 = 52) : M = 150 :=
sorry

end initial_men_in_fort_l54_54091


namespace probability_of_meeting_at_cafe_l54_54264

noncomputable def alice_charlie_meet_probability : ℝ :=
  let meet_event_area : ℝ :=
    let total_area : ℝ := 1
    let nonmeet_area_triangles : ℝ := 2 * (1 / 2 * (2 / 3) ^ 2)
    total_area - nonmeet_area_triangles
  meet_event_area

theorem probability_of_meeting_at_cafe :
  alice_charlie_meet_probability = 5 / 9 :=
by
  sorry

end probability_of_meeting_at_cafe_l54_54264


namespace correct_option_l54_54522

variable (a : ℤ)

theorem correct_option :
  (-2 * a^2)^3 = -8 * a^6 :=
by
  sorry

end correct_option_l54_54522


namespace anne_trip_shorter_l54_54502

noncomputable def john_walk_distance : ℝ := 2 + 1

noncomputable def anne_walk_distance : ℝ := Real.sqrt (2^2 + 1^2)

noncomputable def distance_difference : ℝ := john_walk_distance - anne_walk_distance

noncomputable def percentage_reduction : ℝ := (distance_difference / john_walk_distance) * 100

theorem anne_trip_shorter :
  20 ≤ percentage_reduction ∧ percentage_reduction < 30 :=
by
  sorry

end anne_trip_shorter_l54_54502


namespace janet_earnings_per_hour_l54_54038

def rate_per_post := 0.25  -- Janet’s rate per post in dollars
def time_per_post := 10    -- Time to check one post in seconds
def seconds_per_hour := 3600  -- Seconds in one hour

theorem janet_earnings_per_hour :
  let posts_per_hour := seconds_per_hour / time_per_post
  let earnings_per_hour := rate_per_post * posts_per_hour
  earnings_per_hour = 90 := sorry

end janet_earnings_per_hour_l54_54038


namespace kelly_held_longest_l54_54654

variable (K : ℕ)

-- Conditions
def Brittany_held (K : ℕ) : ℕ := K - 20
def Buffy_held : ℕ := 120

-- Theorem to prove
theorem kelly_held_longest (h : K > Buffy_held) : K > 120 :=
by sorry

end kelly_held_longest_l54_54654


namespace campaign_donation_ratio_l54_54865

theorem campaign_donation_ratio (max_donation : ℝ) 
  (total_money : ℝ) 
  (percent_donations : ℝ) 
  (num_max_donors : ℕ) 
  (half_max_donation : ℝ) 
  (total_raised : ℝ) 
  (half_donation : ℝ) :
  total_money = total_raised * percent_donations →
  half_donation = max_donation / 2 →
  half_max_donation = num_max_donors * max_donation →
  total_money - half_max_donation = 1500 * half_donation →
  (1500 : ℝ) / (num_max_donors : ℝ) = 3 :=
sorry

end campaign_donation_ratio_l54_54865


namespace range_of_m_l54_54021

open Real

theorem range_of_m (m : ℝ) : (¬ ∃ x₀ : ℝ, m * x₀^2 + m * x₀ + 1 ≤ 0) ↔ (0 ≤ m ∧ m < 4) := by
  sorry

end range_of_m_l54_54021


namespace proof_intersection_complement_l54_54351

open Set

variable (U : Set ℝ) (A B : Set ℝ)

theorem proof_intersection_complement:
  U = univ ∧ A = {x | -1 < x ∧ x ≤ 5} ∧ B = {x | x < 2} →
  A ∩ (U \ B) = {x | 2 ≤ x ∧ x ≤ 5} :=
by
  intros h
  rcases h with ⟨hU, hA, hB⟩
  simp [hU, hA, hB]
  sorry

end proof_intersection_complement_l54_54351


namespace problem_proof_l54_54279

def delta (a b : ℕ) : ℕ := a^2 + b

theorem problem_proof :
  let x := 6
  let y := 8
  let z := 4
  let w := 2
  let u := 5^delta x y
  let v := 7^delta z w
  delta u v = 5^88 + 7^18 :=
by
  let x := 6
  let y := 8
  let z := 4
  let w := 2
  let u := 5^delta x y
  let v := 7^delta z w
  have h1: delta x y = 44 := by sorry
  have h2: delta z w = 18 := by sorry
  have hu: u = 5^44 := by sorry
  have hv: v = 7^18 := by sorry
  have hdelta: delta u v = 5^88 + 7^18 := by sorry
  exact hdelta

end problem_proof_l54_54279


namespace log_a_interval_l54_54769

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem log_a_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  {a | log_a a 3 - log_a a 1 = 2} = {Real.sqrt 3, Real.sqrt 3 / 3} :=
by
  sorry

end log_a_interval_l54_54769


namespace solve_cubic_eq_with_geo_prog_coeff_l54_54368

variables {a q x : ℝ}

theorem solve_cubic_eq_with_geo_prog_coeff (h_a_nonzero : a ≠ 0) 
    (h_b : b = a * q) (h_c : c = a * q^2) (h_d : d = a * q^3) :
    (a * x^3 + b * x^2 + c * x + d = 0) → (x = -q) :=
by
  intros h_cubic_eq
  have h_b' : b = a * q := h_b
  have h_c' : c = a * q^2 := h_c
  have h_d' : d = a * q^3 := h_d
  sorry

end solve_cubic_eq_with_geo_prog_coeff_l54_54368


namespace alexa_pages_left_l54_54263

theorem alexa_pages_left 
  (total_pages : ℕ) 
  (first_day_read : ℕ) 
  (next_day_read : ℕ) 
  (total_pages_val : total_pages = 95) 
  (first_day_read_val : first_day_read = 18) 
  (next_day_read_val : next_day_read = 58) : 
  total_pages - (first_day_read + next_day_read) = 19 := by
  sorry

end alexa_pages_left_l54_54263


namespace graph_translation_l54_54633

theorem graph_translation (f : ℝ → ℝ) (x : ℝ) (h : f 1 = -1) :
  f (x - 1) - 1 = -2 :=
by
  sorry

end graph_translation_l54_54633


namespace pascal_triangle_sum_first_30_rows_l54_54318

theorem pascal_triangle_sum_first_30_rows :
  (Finset.range 30).sum (λ n, n + 1) = 465 :=
begin
  sorry
end

end pascal_triangle_sum_first_30_rows_l54_54318


namespace mice_meet_after_three_days_l54_54155

theorem mice_meet_after_three_days 
  (thickness : ℕ) 
  (first_day_distance : ℕ) 
  (big_mouse_double_progress : ℕ → ℕ) 
  (small_mouse_half_remain_distance : ℕ → ℕ) 
  (days : ℕ) 
  (big_mouse_distance : ℚ) : 
  thickness = 5 ∧ 
  first_day_distance = 1 ∧ 
  (∀ n, big_mouse_double_progress n = 2 ^ (n - 1)) ∧ 
  (∀ n, small_mouse_half_remain_distance n = 5 - (5 / 2 ^ (n - 1))) ∧ 
  days = 3 → 
  big_mouse_distance = 3 + 8 / 17 := 
by
  sorry

end mice_meet_after_three_days_l54_54155


namespace meaningful_expression_l54_54399

theorem meaningful_expression (m : ℝ) :
  (2 - m ≥ 0) ∧ (m + 2 ≠ 0) ↔ (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end meaningful_expression_l54_54399


namespace factor_of_7_l54_54360

theorem factor_of_7 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 7 ∣ (a + 2 * b)) : 7 ∣ (100 * a + 11 * b) :=
by sorry

end factor_of_7_l54_54360


namespace simplify_expression_l54_54840

theorem simplify_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 - 2) / x) * ((y^2 - 2) / y) - ((x^2 + 2) / y) * ((y^2 + 2) / x) = -4 * (x / y + y / x) :=
by
  sorry

end simplify_expression_l54_54840


namespace base_area_functional_relationship_base_area_when_height_4_8_l54_54082

noncomputable def cylinder_base_area (h : ℝ) : ℝ := 24 / h

theorem base_area_functional_relationship (h : ℝ) (H : h ≠ 0) :
  cylinder_base_area h = 24 / h := by
  unfold cylinder_base_area
  rfl

theorem base_area_when_height_4_8 :
  cylinder_base_area 4.8 = 5 := by
  unfold cylinder_base_area
  norm_num

end base_area_functional_relationship_base_area_when_height_4_8_l54_54082


namespace ceil_neg_sqrt_eq_neg_two_l54_54114

noncomputable def x : ℝ := -Real.sqrt (64 / 9)

theorem ceil_neg_sqrt_eq_neg_two : Real.ceil x = -2 := by
  exact sorry

end ceil_neg_sqrt_eq_neg_two_l54_54114


namespace tan_at_max_value_l54_54936

theorem tan_at_max_value : 
  ∃ x₀, (∀ x, 3 * Real.sin x₀ - 4 * Real.cos x₀ ≥ 3 * Real.sin x - 4 * Real.cos x) → Real.tan x₀ = 3/4 := 
sorry

end tan_at_max_value_l54_54936


namespace thabo_books_l54_54084

theorem thabo_books :
  ∃ (H P F : ℕ), 
    P = H + 20 ∧ 
    F = 2 * P ∧ 
    H + P + F = 200 ∧ 
    H = 35 :=
by
  sorry

end thabo_books_l54_54084


namespace function_relationship_area_60_maximum_area_l54_54868

-- Definitions and conditions
def perimeter := 32
def side_length (x : ℝ) : ℝ := 16 - x  -- One side of the rectangle
def area (x : ℝ) : ℝ := x * (16 - x)

-- Theorem 1: Function relationship between y and x
theorem function_relationship (x : ℝ) (hx : 0 < x ∧ x < 16) : area x = -x^2 + 16 * x :=
by
  sorry

-- Theorem 2: Values of x when the area is 60 square meters
theorem area_60 (x : ℝ) (hx1 : area x = 60) : x = 6 ∨ x = 10 :=
by
  sorry

-- Theorem 3: Maximum area
theorem maximum_area : ∃ x, area x = 64 ∧ x = 8 :=
by
  sorry

end function_relationship_area_60_maximum_area_l54_54868


namespace bees_multiple_l54_54967

theorem bees_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 :=
by
  sorry

end bees_multiple_l54_54967


namespace value_expression_eq_zero_l54_54952

theorem value_expression_eq_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
    (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
    a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 :=
by
  sorry

end value_expression_eq_zero_l54_54952


namespace conditional_probability_l54_54061

def P (event : ℕ → Prop) : ℝ := sorry

def A (n : ℕ) : Prop := n = 10000
def B (n : ℕ) : Prop := n = 15000

theorem conditional_probability :
  P A = 0.80 →
  P B = 0.60 →
  P B / P A = 0.75 :=
by
  intros hA hB
  sorry

end conditional_probability_l54_54061


namespace find_y_l54_54976

theorem find_y (y : ℚ) (h : Real.sqrt (1 + Real.sqrt (3 * y - 4)) = Real.sqrt 9) : y = 68 / 3 := 
by
  sorry

end find_y_l54_54976


namespace sum_of_youngest_and_oldest_cousins_l54_54577

theorem sum_of_youngest_and_oldest_cousins 
  (a1 a2 a3 a4 : ℕ) 
  (h_order : a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4) 
  (h_mean : a1 + a2 + a3 + a4 = 36) 
  (h_median : a2 + a3 = 14) : 
  a1 + a4 = 22 :=
by sorry

end sum_of_youngest_and_oldest_cousins_l54_54577


namespace max_marks_equals_l54_54171

/-
  Pradeep has to obtain 45% of the total marks to pass.
  He got 250 marks and failed by 50 marks.
  Prove that the maximum marks is 667.
-/

-- Define the passing percentage
def passing_percentage : ℝ := 0.45

-- Define Pradeep's marks and the marks he failed by
def pradeep_marks : ℝ := 250
def failed_by : ℝ := 50

-- Passing marks is the sum of Pradeep's marks and the marks he failed by
def passing_marks : ℝ := pradeep_marks + failed_by

-- Prove that the maximum marks M is 667
theorem max_marks_equals : ∃ M : ℝ, passing_percentage * M = passing_marks ∧ M = 667 :=
sorry

end max_marks_equals_l54_54171


namespace number_of_plants_l54_54066

--- The given problem conditions and respective proof setup
axiom green_leaves_per_plant : ℕ
axiom yellow_turn_fall_off : ℕ
axiom green_leaves_total : ℕ

def one_third (n : ℕ) : ℕ := n / 3

-- Specify the given conditions
axiom leaves_per_plant_cond : green_leaves_per_plant = 18
axiom fall_off_cond : yellow_turn_fall_off = one_third green_leaves_per_plant
axiom total_leaves_cond : green_leaves_total = 36

-- Proof statement for the number of tea leaf plants
theorem number_of_plants : 
  (green_leaves_per_plant - yellow_turn_fall_off) * 3 = green_leaves_total :=
by
  sorry

end number_of_plants_l54_54066


namespace avg_first_six_results_l54_54977

theorem avg_first_six_results (average_11 : ℕ := 52) (average_last_6 : ℕ := 52) (sixth_result : ℕ := 34) :
  ∃ A : ℕ, (6 * A + 6 * average_last_6 - sixth_result = 11 * average_11) ∧ A = 49 :=
by
  sorry

end avg_first_six_results_l54_54977


namespace discount_percentage_l54_54140

theorem discount_percentage (wm_cost dryer_cost after_discount before_discount discount_amount : ℝ)
    (h0 : wm_cost = 100) 
    (h1 : dryer_cost = wm_cost - 30) 
    (h2 : after_discount = 153) 
    (h3 : before_discount = wm_cost + dryer_cost) 
    (h4 : discount_amount = before_discount - after_discount) 
    (h5 : (discount_amount / before_discount) * 100 = 10) : 
    True := sorry

end discount_percentage_l54_54140


namespace find_E_coordinates_l54_54969

structure Point :=
(x : ℚ)
(y : ℚ)

def A : Point := { x := -2, y := 1 }
def B : Point := { x := 1, y := 4 }
def C : Point := { x := 4, y := -3 }

def D : Point := 
  let m : ℚ := 1
  let n : ℚ := 2
  let x1 := A.x
  let y1 := A.y
  let x2 := B.x
  let y2 := B.y
  { x := (m * x2 + n * x1) / (m + n), y := (m * y2 + n * y1) / (m + n) }

theorem find_E_coordinates : 
  let k : ℚ := 4
  let x_E : ℚ := (k * C.x + D.x) / (k + 1)
  let y_E : ℚ := (k * C.y + D.y) / (k + 1)
  ∃ E : Point, E.x = (17:ℚ) / 3 ∧ E.y = -(14:ℚ) / 3 :=
sorry

end find_E_coordinates_l54_54969


namespace angle_in_triangle_l54_54494

theorem angle_in_triangle (A B C x : ℝ) (hA : A = 40)
    (hB : B = 3 * x) (hC : C = x) (h_sum : A + B + C = 180) : x = 35 :=
by
  sorry

end angle_in_triangle_l54_54494


namespace minimum_value_function_inequality_ln_l54_54609

noncomputable def f (x : ℝ) := x * Real.log x

theorem minimum_value_function (t : ℝ) (ht : 0 < t) :
  ∃ (xmin : ℝ), xmin = if (0 < t ∧ t < 1 / Real.exp 1) then -1 / Real.exp 1 else t * Real.log t :=
sorry

theorem inequality_ln (x : ℝ) (hx : 0 < x) : 
  Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end minimum_value_function_inequality_ln_l54_54609


namespace minimum_participants_l54_54201

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l54_54201


namespace length_of_rectangular_garden_l54_54244

-- Define the perimeter and breadth conditions
def perimeter : ℕ := 950
def breadth : ℕ := 100

-- The formula for the perimeter of a rectangle
def formula (L B : ℕ) : ℕ := 2 * (L + B)

-- State the theorem
theorem length_of_rectangular_garden (L : ℕ) 
  (h1 : perimeter = 2 * (L + breadth)) : 
  L = 375 := 
by
  sorry

end length_of_rectangular_garden_l54_54244


namespace value_of_a_l54_54629

-- Definition of the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

-- Definition of the derivative f'(-1)
def f_prime_at_neg1 (a : ℝ) : ℝ := 3 * a - 6

-- The theorem to prove the value of a
theorem value_of_a (a : ℝ) (h : f_prime_at_neg1 a = 3) : a = 3 :=
by
  sorry

end value_of_a_l54_54629


namespace find_first_number_l54_54845

theorem find_first_number (HCF LCM num2 num1 : ℕ) (hcf_cond : HCF = 20) (lcm_cond : LCM = 396) (num2_cond : num2 = 220) 
    (relation_cond : HCF * LCM = num1 * num2) : num1 = 36 :=
by
  sorry

end find_first_number_l54_54845


namespace find_y_value_l54_54243

theorem find_y_value : (12^3 * 6^3 / 432) = 864 := by
  sorry

end find_y_value_l54_54243


namespace coin_toss_probability_l54_54898

-- Defining the problem conditions
def unfair_coin_probability : ℝ := 3 / 4
def num_tosses : ℕ := 60
def successful_heads : ℕ := 45

-- Lean proof statement
theorem coin_toss_probability :
  Pr { outcomes : ℕ // outcomes > successful_heads | binomial num_tosses unfair_coin_probability } = 0.9524 :=
sorry

end coin_toss_probability_l54_54898


namespace remainder_of_98_mult_102_div_12_l54_54688

theorem remainder_of_98_mult_102_div_12 : (98 * 102) % 12 = 0 := by
    sorry

end remainder_of_98_mult_102_div_12_l54_54688


namespace stream_speed_l54_54544

variable (D v : ℝ)

/--
The time taken by a man to row his boat upstream is twice the time taken by him to row the same distance downstream.
If the speed of the boat in still water is 63 kmph, prove that the speed of the stream is 21 kmph.
-/
theorem stream_speed (h : D / (63 - v) = 2 * (D / (63 + v))) : v = 21 := 
sorry

end stream_speed_l54_54544


namespace david_marks_in_english_l54_54904

theorem david_marks_in_english : 
  ∀ (E : ℕ), 
  let math_marks := 85 
  let physics_marks := 82 
  let chemistry_marks := 87 
  let biology_marks := 85 
  let avg_marks := 85 
  let total_subjects := 5 
  let total_marks := avg_marks * total_subjects 
  let total_known_subject_marks := math_marks + physics_marks + chemistry_marks + biology_marks 
  total_marks = total_known_subject_marks + E → 
  E = 86 :=
by 
  intros
  sorry

end david_marks_in_english_l54_54904


namespace mean_of_five_numbers_l54_54022

theorem mean_of_five_numbers (a b c d e : ℚ) (h : a + b + c + d + e = 2/3) : 
  (a + b + c + d + e) / 5 = 2 / 15 := 
by 
  -- This is where the proof would go, but we'll omit it as per instructions
  sorry

end mean_of_five_numbers_l54_54022


namespace circumference_difference_l54_54087

theorem circumference_difference (r : ℝ) (width : ℝ) (hp : width = 10.504226244065093) : 
  2 * Real.pi * (r + width) - 2 * Real.pi * r = 66.00691339889247 := by
  sorry

end circumference_difference_l54_54087


namespace evaluate_expression_l54_54594

theorem evaluate_expression : 2^(Real.log 5 / Real.log 2) + Real.log 25 / Real.log 5 = 7 := by
  sorry

end evaluate_expression_l54_54594


namespace inequality_solution_set_range_of_k_l54_54768

variable {k m x : ℝ}

theorem inequality_solution_set (k_pos : k > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = k * x / (x^2 + 3 * k)) 
  (sol_set_f_x_gt_m : ∀ x, f x > m ↔ (x < -3 ∨ x > -2)) :
  -1 < x ∧ x < 3 / 2 := 
sorry

theorem range_of_k (k_pos : k > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = k * x / (x^2 + 3 * k))
  (exists_f_x_gt_1 : ∃ x > 3, f x > 1) : 
  k > 12 :=
sorry

end inequality_solution_set_range_of_k_l54_54768


namespace cell_phone_total_cost_l54_54427

def base_cost : ℕ := 25
def text_cost_per_message : ℕ := 3
def extra_minute_cost_per_minute : ℕ := 15
def included_hours : ℕ := 40
def messages_sent_in_february : ℕ := 200
def hours_talked_in_february : ℕ := 41

theorem cell_phone_total_cost :
  base_cost + (messages_sent_in_february * text_cost_per_message) / 100 + 
  ((hours_talked_in_february - included_hours) * 60 * extra_minute_cost_per_minute) / 100 = 40 :=
by
  sorry

end cell_phone_total_cost_l54_54427


namespace parabola_vertex_coordinates_l54_54533

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = 3 * (x - 7)^2 + 5 → (7, 5) = (7, 5) :=
by
  intros x y h
  exact rfl

end parabola_vertex_coordinates_l54_54533


namespace triangle_inequality_l54_54512

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b) / (a + b + c) > 1 / 2 :=
sorry

end triangle_inequality_l54_54512


namespace ceil_of_neg_sqrt_frac_64_over_9_l54_54113

theorem ceil_of_neg_sqrt_frac_64_over_9 :
  ⌈-Real.sqrt (64 / 9)⌉ = -2 :=
by
  sorry

end ceil_of_neg_sqrt_frac_64_over_9_l54_54113


namespace product_of_xy_l54_54329

theorem product_of_xy : 
  ∃ (x y : ℝ), 3 * x + 4 * y = 60 ∧ 6 * x - 4 * y = 12 ∧ x * y = 72 :=
by
  sorry

end product_of_xy_l54_54329


namespace second_watermelon_correct_weight_l54_54503

-- Define various weights involved as given in the conditions
def first_watermelon_weight : ℝ := 9.91
def total_watermelon_weight : ℝ := 14.02

-- Define the weight of the second watermelon
def second_watermelon_weight : ℝ :=
  total_watermelon_weight - first_watermelon_weight

-- State the theorem to prove that the weight of the second watermelon is 4.11 pounds
theorem second_watermelon_correct_weight : second_watermelon_weight = 4.11 :=
by
  -- This ensures the statement can be built successfully in Lean 4
  sorry

end second_watermelon_correct_weight_l54_54503


namespace combined_forgotten_angles_l54_54284

-- Define primary conditions
def initial_angle_sum : ℝ := 2873
def correct_angle_sum : ℝ := 16 * 180

-- The theorem to prove
theorem combined_forgotten_angles : correct_angle_sum - initial_angle_sum = 7 :=
by sorry

end combined_forgotten_angles_l54_54284


namespace ellipse_reflection_symmetry_l54_54744

theorem ellipse_reflection_symmetry :
  (∀ x y, (x = -y ∧ y = -x) →
  (∀ a b : ℝ, 
    (a - 3)^2 / 9 + (b - 2)^2 / 4 = 1 ↔
    (b - 3)^2 / 4 + (a - 2)^2 / 9 = 1)
  )
  →
  (∀ x y, 
    ((x + 2)^2 / 9 + (y + 3)^2 / 4 = 1) = 
    (∃ a b : ℝ, 
      (a - 3)^2 / 9 + (b - 2)^2 / 4 = 1 ∧ 
      (a = -y ∧ b = -x))
  ) :=
by
  intros
  sorry

end ellipse_reflection_symmetry_l54_54744


namespace bob_daily_work_hours_l54_54908

theorem bob_daily_work_hours
  (total_hours_in_month : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_working_days : ℕ)
  (daily_working_hours : ℕ)
  (h1 : total_hours_in_month = 200)
  (h2 : days_per_week = 5)
  (h3 : weeks_per_month = 4)
  (h4 : total_working_days = days_per_week * weeks_per_month)
  (h5 : daily_working_hours = total_hours_in_month / total_working_days) :
  daily_working_hours = 10 := 
sorry

end bob_daily_work_hours_l54_54908


namespace no_solution_range_of_a_l54_54803

theorem no_solution_range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 5| + |x + 3| < a)) → a ≤ 8 :=
by
  sorry

end no_solution_range_of_a_l54_54803


namespace find_c_l54_54806

theorem find_c (y : ℝ) (c : ℝ) (h1 : y > 0) (h2 : (6 * y / 20) + (c * y / 10) = 0.6 * y) : c = 3 :=
by 
  -- Skipping the proof
  sorry

end find_c_l54_54806


namespace probability_of_TTH_sequence_l54_54163

-- Define the probability of each event in a fair coin flip
noncomputable def fair_coin_flip : ProbabilityMassFunction (Bool) :=
  { support := { tt, ff },
    mass := λ b, if b then 1 / 2 else 1 / 2,
    mass_nonneg := by intros; split_ifs; norm_num,
    mass_sum := by norm_num }

-- Define the sequence we are interested in
def sequence_TTH (flips : list Bool) : Prop :=
  flips = [ff, ff, tt]

-- Define the probability of observing a specific sequence in three independent flips of a fair coin
def probability_TTH : ℚ :=
  (fair_coin_flip.mass false) * (fair_coin_flip.mass false) * (fair_coin_flip.mass true)

-- State the theorem to be proved
theorem probability_of_TTH_sequence : probability_TTH = 1 / 8 :=
by
  unfold probability_TTH fair_coin_flip.mass
  simp
  norm_num

end probability_of_TTH_sequence_l54_54163


namespace kira_memory_space_is_140_l54_54821

def kira_songs_memory_space 
  (n_m : ℕ) -- number of songs downloaded in the morning
  (n_d : ℕ) -- number of songs downloaded later that day
  (n_n : ℕ) -- number of songs downloaded at night
  (s : ℕ) -- size of each song in MB
  : ℕ := (n_m + n_d + n_n) * s

theorem kira_memory_space_is_140 :
  kira_songs_memory_space 10 15 3 5 = 140 := 
by
  sorry

end kira_memory_space_is_140_l54_54821


namespace arithmetic_seq_a4_l54_54302

theorem arithmetic_seq_a4 (a : ℕ → ℕ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 4) 
  (h3 : a 3 = 6) : 
  a 4 = 8 :=
by
  sorry

end arithmetic_seq_a4_l54_54302


namespace distinct_int_divisible_by_12_l54_54505

variable {a b c d : ℤ}

theorem distinct_int_divisible_by_12 (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) :=
by
  sorry

end distinct_int_divisible_by_12_l54_54505


namespace work_completed_together_l54_54697

theorem work_completed_together (A_days B_days : ℕ) (hA : A_days = 40) (hB : B_days = 60) : 
  1 / (1 / (A_days: ℝ) + 1 / (B_days: ℝ)) = 24 :=
by
  sorry

end work_completed_together_l54_54697


namespace tan_triple_angle_l54_54794

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l54_54794


namespace regular_tetrahedron_of_angle_l54_54383

-- Definition and condition from the problem
def angle_between_diagonals (shape : Type _) (adj_sides_diag_angle : ℝ) : Prop :=
  adj_sides_diag_angle = 60

-- Theorem stating the problem in Lean 4
theorem regular_tetrahedron_of_angle (shape : Type _) (adj_sides_diag_angle : ℝ) 
  (h : angle_between_diagonals shape adj_sides_diag_angle) : 
  shape = regular_tetrahedron :=
sorry

end regular_tetrahedron_of_angle_l54_54383


namespace charge_difference_percentage_l54_54378

-- Given definitions
variables (G R P : ℝ)
def hotelR := 1.80 * G
def hotelP := 0.90 * G

-- Theorem statement
theorem charge_difference_percentage (G : ℝ) (hR : R = 1.80 * G) (hP : P = 0.90 * G) :
  (R - P) / R * 100 = 50 :=
by sorry

end charge_difference_percentage_l54_54378


namespace vacation_trip_l54_54045

theorem vacation_trip (airbnb_cost : ℕ) (car_rental_cost : ℕ) (share_per_person : ℕ) (total_people : ℕ) :
  airbnb_cost = 3200 → car_rental_cost = 800 → share_per_person = 500 → airbnb_cost + car_rental_cost / share_per_person = 8 :=
by
  intros h1 h2 h3
  sorry

end vacation_trip_l54_54045


namespace find_x_if_opposites_l54_54129

theorem find_x_if_opposites (x : ℝ) (h : 2 * (x - 3) = - 4 * (1 - x)) : x = -1 := 
by
  sorry

end find_x_if_opposites_l54_54129


namespace race_participants_minimum_l54_54197

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l54_54197


namespace problem1_problem2_l54_54771

-- Define Set A
def SetA : Set ℝ := { y | ∃ x, (2 ≤ x ∧ x ≤ 3) ∧ y = -2^x }

-- Define Set B parameterized by a
def SetB (a : ℝ) : Set ℝ := { x | x^2 + 3 * x - a^2 - 3 * a > 0 }

-- Problem 1: Prove that when a = 4, A ∩ B = {-8 < x < -7}
theorem problem1 : A ∩ SetB 4 = { x | -8 < x ∧ x < -7 } :=
sorry

-- Problem 2: Prove the range of a for which "x ∈ A" is a sufficient but not necessary condition for "x ∈ B"
theorem problem2 : ∀ a : ℝ, (∀ x, x ∈ SetA → x ∈ SetB a) → -4 < a ∧ a < 1 :=
sorry

end problem1_problem2_l54_54771


namespace probability_of_negative_l54_54896

def set_of_numbers : Set ℤ := {-2, 1, 4, -3, 0}
def negative_numbers : Set ℤ := {-2, -3}
def total_numbers : ℕ := 5
def total_negative_numbers : ℕ := 2

theorem probability_of_negative :
  (total_negative_numbers : ℚ) / (total_numbers : ℚ) = 2 / 5 := 
by 
  sorry

end probability_of_negative_l54_54896


namespace find_x_l54_54689

theorem find_x :
  ∀ (x y z w : ℕ), 
    x = y + 5 →
    y = z + 10 →
    z = w + 20 →
    w = 80 →
    x = 115 :=
by
  intros x y z w h1 h2 h3 h4
  sorry

end find_x_l54_54689


namespace shaded_area_represents_correct_set_l54_54928

theorem shaded_area_represents_correct_set :
  ∀ (U A B : Set ℕ), 
    U = {0, 1, 2, 3, 4} → 
    A = {1, 2, 3} → 
    B = {2, 4} → 
    (U \ (A ∪ B)) ∪ (A ∩ B) = {0, 2} :=
by
  intros U A B hU hA hB
  -- The rest of the proof would go here
  sorry

end shaded_area_represents_correct_set_l54_54928


namespace Jane_Hector_meet_point_C_l54_54646

theorem Jane_Hector_meet_point_C (s t : ℝ) (h_start : ℝ) (j_start : ℝ) (loop_length : ℝ) 
  (h_speed : ℝ) (j_speed : ℝ) (h_dest : ℝ) (j_dest : ℝ)
  (h_speed_eq : h_speed = s) (j_speed_eq : j_speed = 3 * s) (loop_len_eq : loop_length = 30)
  (start_point_eq : h_start = 0 ∧ j_start = 0)
  (opposite_directions : h_dest + j_dest = loop_length)
  (meet_time_eq : t = 15 / (2 * s)) :
  h_dest = 7.5 ∧ j_dest = 22.5 → (h_dest = 7.5 ∧ j_dest = 22.5) :=
by
  sorry

end Jane_Hector_meet_point_C_l54_54646


namespace apple_capacity_l54_54970

/-- Question: What is the largest possible number of apples that can be held by the 6 boxes and 4 extra trays?
 Conditions:
 - Paul has 6 boxes.
 - Each box contains 12 trays.
 - Paul has 4 extra trays.
 - Each tray can hold 8 apples.
 Answer: 608 apples
-/
theorem apple_capacity :
  let boxes := 6
  let trays_per_box := 12
  let extra_trays := 4
  let apples_per_tray := 8
  let total_trays := (boxes * trays_per_box) + extra_trays
  let total_apples_capacity := total_trays * apples_per_tray
  total_apples_capacity = 608 := 
by
  sorry

end apple_capacity_l54_54970


namespace opposite_of_2023_l54_54678

theorem opposite_of_2023 : ∃ y : ℤ, 2023 + y = 0 ∧ y = -2023 :=
by
  use -2023
  split
  · linarith
  · refl

end opposite_of_2023_l54_54678


namespace find_a_l54_54133

variable (a : ℝ)

def p (a : ℝ) : Set ℝ := {x | a-1 < x ∧ x < a+1}
def q : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}
def q_negation : Set ℝ := {x | 1 < x ∧ x < 3}

theorem find_a :
  (∀ x, q_negation x → p a x) → a = 2 := by
  sorry

end find_a_l54_54133


namespace probability_product_odd_prob_lt_eighth_l54_54228

theorem probability_product_odd_prob_lt_eighth:
  let total_numbers := 2020
  let odd_numbers := 1010
  let first_odd_prob := (odd_numbers : ℚ) / total_numbers
  let second_odd_prob := (odd_numbers - 1 : ℚ) / (total_numbers - 1)
  let third_odd_prob := (odd_numbers - 2 : ℚ) / (total_numbers - 2)
  let p := first_odd_prob * second_odd_prob * third_odd_prob
  p < 1 / 8 :=
by
  sorry

end probability_product_odd_prob_lt_eighth_l54_54228


namespace tan_theta_3_l54_54785

noncomputable def tan_triple_angle (θ : ℝ) : ℝ := (3 * (Real.tan θ) - ((Real.tan θ) ^ 3)) / (1 - 3 * (Real.tan θ)^2)

theorem tan_theta_3 (θ : ℝ) (h : Real.tan θ = 3) : tan_triple_angle θ = 9 / 13 :=
by
  sorry

end tan_theta_3_l54_54785


namespace melted_ice_cream_depth_l54_54258

theorem melted_ice_cream_depth :
  ∀ (r_sphere r_cylinder : ℝ) (h_cylinder : ℝ),
    r_sphere = 3 ∧ r_cylinder = 12 ∧
    (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h_cylinder →
    h_cylinder = 1 / 4 :=
by
  intros r_sphere r_cylinder h_cylinder h
  have r_sphere_eq : r_sphere = 3 := h.1
  have r_cylinder_eq : r_cylinder = 12 := h.2.1
  have volume_eq : (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h_cylinder := h.2.2
  sorry

end melted_ice_cream_depth_l54_54258


namespace g_g_g_g_15_eq_3_l54_54043

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 5 * x + 2

theorem g_g_g_g_15_eq_3 : g (g (g (g 15))) = 3 := 
by
  sorry

end g_g_g_g_15_eq_3_l54_54043


namespace solution_set_of_inequality_system_l54_54681

theorem solution_set_of_inequality_system (x : ℝ) :
  (x + 2 ≤ 3 ∧ 1 + x > -2) ↔ (-3 < x ∧ x ≤ 1) :=
by
  sorry

end solution_set_of_inequality_system_l54_54681


namespace staff_discount_l54_54089

open Real

theorem staff_discount (d : ℝ) (h : d > 0) (final_price_eq : 0.14 * d = 0.35 * d * (1 - 0.6)) : 0.6 * 100 = 60 :=
by
  sorry

end staff_discount_l54_54089


namespace difference_of_squares_l54_54410

theorem difference_of_squares :
  535^2 - 465^2 = 70000 :=
by
  sorry

end difference_of_squares_l54_54410


namespace xiaoLiangComprehensiveScore_l54_54567

-- Define the scores for the three aspects
def contentScore : ℝ := 88
def deliveryAbilityScore : ℝ := 95
def effectivenessScore : ℝ := 90

-- Define the weights for the three aspects
def contentWeight : ℝ := 0.5
def deliveryAbilityWeight : ℝ := 0.4
def effectivenessWeight : ℝ := 0.1

-- Define the comprehensive score
def comprehensiveScore : ℝ :=
  (contentScore * contentWeight) +
  (deliveryAbilityScore * deliveryAbilityWeight) +
  (effectivenessScore * effectivenessWeight)

-- The theorem stating that the comprehensive score equals 91
theorem xiaoLiangComprehensiveScore : comprehensiveScore = 91 := by
  -- proof here (omitted)
  sorry

end xiaoLiangComprehensiveScore_l54_54567


namespace remainder_of_square_l54_54826

theorem remainder_of_square (n : ℤ) (h : n % 5 = 3) : n^2 % 5 = 4 := 
by 
  sorry

end remainder_of_square_l54_54826


namespace slope_perpendicular_to_given_line_l54_54601

-- Define the given line
def given_line (x y : ℝ) := 3 * x - 4 * y = 8

-- Define the slope-intercept form of the line and its slope
def slope_of_given_line := (3 : ℝ) / (4 : ℝ)

-- Define the slope of the line perpendicular to the given line
def perpendicular_slope := -1 / slope_of_given_line

-- Theorem: The slope of the line perpendicular to the line 3x - 4y = 8 is -4/3
theorem slope_perpendicular_to_given_line : perpendicular_slope = - (4 : ℝ) / (3 : ℝ) :=
by sorry

end slope_perpendicular_to_given_line_l54_54601


namespace total_students_registered_l54_54669

theorem total_students_registered 
  (students_yesterday : ℕ) (absent_today : ℕ) 
  (attended_today : ℕ)
  (h1 : students_yesterday = 70)
  (h2 : absent_today = 30)
  (h3 : attended_today = (2 * students_yesterday) - (10 * (2 * students_yesterday) / 100)) :
  students_yesterday + absent_today = 156 := 
by
  sorry

end total_students_registered_l54_54669


namespace range_of_m_l54_54134

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * x > m
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * m * x + 2 - m ≤ 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ m ∈ Set.Ioo (-2:ℝ) (-1) ∪ Set.Ici 1 :=
sorry

end range_of_m_l54_54134


namespace pat_kate_ratio_l54_54833

theorem pat_kate_ratio 
  (P K M : ℕ)
  (h1 : P + K + M = 117)
  (h2 : ∃ r : ℕ, P = r * K)
  (h3 : P = M / 3)
  (h4 : M = K + 65) : 
  P / K = 2 :=
by
  sorry

end pat_kate_ratio_l54_54833


namespace area_ratio_of_similar_isosceles_triangles_l54_54685

theorem area_ratio_of_similar_isosceles_triangles
  (b1 b2 h1 h2 : ℝ)
  (h_ratio : h1 / h2 = 2 / 3)
  (similar_tri : b1 / b2 = 2 / 3) :
  (1 / 2 * b1 * h1) / (1 / 2 * b2 * h2) = 4 / 9 :=
by
  sorry

end area_ratio_of_similar_isosceles_triangles_l54_54685


namespace tan_theta_3_l54_54784

noncomputable def tan_triple_angle (θ : ℝ) : ℝ := (3 * (Real.tan θ) - ((Real.tan θ) ^ 3)) / (1 - 3 * (Real.tan θ)^2)

theorem tan_theta_3 (θ : ℝ) (h : Real.tan θ = 3) : tan_triple_angle θ = 9 / 13 :=
by
  sorry

end tan_theta_3_l54_54784


namespace total_students_playing_one_sport_l54_54334

noncomputable def students_playing_at_least_one_sport (total_students B S Ba C B_S B_Ba B_C S_Ba C_S C_Ba B_C_S: ℕ) : ℕ :=
  B + S + Ba + C - B_S - B_Ba - B_C - S_Ba - C_S - C_Ba + B_C_S

theorem total_students_playing_one_sport : 
  students_playing_at_least_one_sport 200 50 60 35 80 10 15 20 25 30 5 10 = 130 := by
  sorry

end total_students_playing_one_sport_l54_54334


namespace determine_alpha_l54_54307

theorem determine_alpha (α : ℝ) (y : ℝ → ℝ) (h : ∀ x, y x = x^α) (hp : y 2 = Real.sqrt 2) : α = 1 / 2 :=
sorry

end determine_alpha_l54_54307


namespace race_participants_minimum_l54_54195

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l54_54195


namespace third_offense_percentage_increase_l54_54353

theorem third_offense_percentage_increase 
    (base_per_5000 : ℕ)
    (goods_stolen : ℕ)
    (additional_years : ℕ)
    (total_sentence : ℕ) :
    base_per_5000 = 1 →
    goods_stolen = 40000 →
    additional_years = 2 →
    total_sentence = 12 →
    100 * (total_sentence - additional_years - goods_stolen / 5000) / (goods_stolen / 5000) = 25 :=
by
  intros h_base h_goods h_additional h_total
  sorry

end third_offense_percentage_increase_l54_54353


namespace symmetric_probability_l54_54572

-- Definitions based on the problem conditions
def total_points : ℕ := 121
def central_point : ℕ × ℕ := (6, 6)
def remaining_points : ℕ := total_points - 1
def symmetric_points : ℕ := 40

-- Predicate for the probability that line PQ is a line of symmetry
def is_symmetrical_line (p q : (ℕ × ℕ)) : Prop := 
  (q.fst = 11 - p.fst ∧ q.snd = p.snd) ∨
  (q.fst = p.fst ∧ q.snd = 11 - p.snd) ∨
  (q.fst + q.snd = 12) ∨ 
  (q.fst - q.snd = 0)

-- The theorem stating the probability is 1/3
theorem symmetric_probability :
  ∃ (total_points : ℕ) (remaining_points : ℕ) (symmetric_points : ℕ),
    total_points = 121 ∧
    remaining_points = total_points - 1 ∧
    symmetric_points = 40 ∧
    (symmetric_points : ℚ) / (remaining_points : ℚ) = 1 / 3 :=
by
  sorry

end symmetric_probability_l54_54572


namespace cost_difference_per_square_inch_l54_54996

theorem cost_difference_per_square_inch (width1 height1 width2 height2 : ℕ) (cost1 cost2 : ℕ)
  (h_size1 : width1 = 24 ∧ height1 = 16)
  (h_cost1 : cost1 = 672)
  (h_size2 : width2 = 48 ∧ height2 = 32)
  (h_cost2 : cost2 = 1152) :
  (cost1 / (width1 * height1) : ℚ) - (cost2 / (width2 * height2) : ℚ) = 1 := 
by
  sorry

end cost_difference_per_square_inch_l54_54996


namespace problem_solution_l54_54931

theorem problem_solution :
  (∑ n in Finset.range 1000 + 1, n * (1001 - n)) = 1000 * 500 * (2 / 3) :=
by
  sorry

end problem_solution_l54_54931


namespace tan_ratio_l54_54608

theorem tan_ratio (α β : ℝ) (h : Real.sin (2 * α) = 3 * Real.sin (2 * β)) :
  (Real.tan (α - β) / Real.tan (α + β)) = 1 / 2 :=
sorry

end tan_ratio_l54_54608


namespace min_number_of_participants_l54_54180

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l54_54180


namespace problem_solution_l54_54347

theorem problem_solution (n : ℕ) (h : n^3 - n = 5814) : (n % 2 = 0) :=
by sorry

end problem_solution_l54_54347


namespace probability_product_odd_prob_lt_eighth_l54_54229

theorem probability_product_odd_prob_lt_eighth:
  let total_numbers := 2020
  let odd_numbers := 1010
  let first_odd_prob := (odd_numbers : ℚ) / total_numbers
  let second_odd_prob := (odd_numbers - 1 : ℚ) / (total_numbers - 1)
  let third_odd_prob := (odd_numbers - 2 : ℚ) / (total_numbers - 2)
  let p := first_odd_prob * second_odd_prob * third_odd_prob
  p < 1 / 8 :=
by
  sorry

end probability_product_odd_prob_lt_eighth_l54_54229


namespace wrapping_paper_area_correct_l54_54430

-- Given conditions:
variables (w h : ℝ) -- base length and height of the box

-- Definition of the area of the wrapping paper given the problem's conditions
def wrapping_paper_area (w h : ℝ) : ℝ :=
  2 * (w + h) ^ 2

-- Theorem statement to prove the area of the wrapping paper
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  -- proof to be provided
  sorry

end wrapping_paper_area_correct_l54_54430


namespace integer_product_zero_l54_54216

theorem integer_product_zero (a : ℤ) (x : Fin 13 → ℤ)
  (h : a = ∏ i, (1 + x i) ∧ a = ∏ i, (1 - x i)) :
  a * ∏ i, x i = 0 :=
sorry

end integer_product_zero_l54_54216


namespace molly_takes_180_minutes_more_l54_54559

noncomputable def xanthia_speed : ℕ := 120
noncomputable def molly_speed : ℕ := 60
noncomputable def first_book_pages : ℕ := 360

-- Time taken by Xanthia to read the first book in hours
noncomputable def xanthia_time_first_book : ℕ := first_book_pages / xanthia_speed

-- Time taken by Molly to read the first book in hours
noncomputable def molly_time_first_book : ℕ := first_book_pages / molly_speed

-- Difference in time taken to read the first book in minutes
noncomputable def time_diff_minutes : ℕ := (molly_time_first_book - xanthia_time_first_book) * 60

theorem molly_takes_180_minutes_more : time_diff_minutes = 180 := by
  sorry

end molly_takes_180_minutes_more_l54_54559


namespace find_y_l54_54677

def custom_op (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem find_y (y : ℤ) (h : custom_op y 10 = 90) : y = 11 :=
by
  sorry

end find_y_l54_54677


namespace race_participants_minimum_l54_54196

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l54_54196


namespace probability_x_plus_y_lt_3_in_rectangle_l54_54434

noncomputable def probability_problem : ℚ :=
let rect_area := (4 : ℚ) * 3
let tri_area := (1 / 2 : ℚ) * 3 * 3
tri_area / rect_area

theorem probability_x_plus_y_lt_3_in_rectangle :
  probability_problem = 3 / 8 :=
sorry

end probability_x_plus_y_lt_3_in_rectangle_l54_54434


namespace probability_sum_of_5_l54_54867

noncomputable def tetrahedral_dice_probability : ℚ :=
  let outcomes := [(1, 4), (2, 3), (3, 2), (4, 1)]
  let favorable_outcomes := 4 * 2
  let total_outcomes := 4 * 4
  favorable_outcomes / total_outcomes

theorem probability_sum_of_5 : tetrahedral_dice_probability = 1 / 2 := by
  sorry

end probability_sum_of_5_l54_54867


namespace percentage_spent_on_hats_l54_54504

def total_money : ℕ := 90
def cost_per_scarf : ℕ := 2
def number_of_scarves : ℕ := 18
def cost_of_scarves : ℕ := number_of_scarves * cost_per_scarf
def money_left_for_hats : ℕ := total_money - cost_of_scarves
def number_of_hats : ℕ := 2 * number_of_scarves

theorem percentage_spent_on_hats : 
  (money_left_for_hats : ℝ) / (total_money : ℝ) * 100 = 60 :=
by
  sorry

end percentage_spent_on_hats_l54_54504


namespace range_of_a_l54_54480

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → x ≤ 4 → (a * x^2 - 2 * x + 2) > 0) ↔ (a > 1 / 2) :=
by
  sorry

end range_of_a_l54_54480


namespace three_digit_number_value_l54_54706

theorem three_digit_number_value (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
    (h4 : a > b) (h5 : b > c)
    (h6 : (10 * a + b) + (10 * b + a) = 55)  
    (h7 : 1300 < 222 * (a + b + c) ∧ 222 * (a + b + c) < 1400) : 
    (100 * a + 10 * b + c) = 321 := 
sorry

end three_digit_number_value_l54_54706


namespace minutes_past_midnight_l54_54814

-- Definitions for the problem

def degree_per_tick : ℝ := 30
def degree_per_minute_hand : ℝ := 6
def degree_per_hour_hand_hourly : ℝ := 30
def degree_per_hour_hand_minutes : ℝ := 0.5

def condition_minute_hand_degree := 300
def condition_hour_hand_degree := 70

-- Main theorem statement
theorem minutes_past_midnight :
  ∃ (h m: ℝ),
    degree_per_hour_hand_hourly * h + degree_per_hour_hand_minutes * m = condition_hour_hand_degree ∧
    degree_per_minute_hand * m = condition_minute_hand_degree ∧
    h * 60 + m = 110 :=
by
  sorry

end minutes_past_midnight_l54_54814


namespace no_b_satisfies_143b_square_of_integer_l54_54456

theorem no_b_satisfies_143b_square_of_integer :
  ∀ b : ℤ, b > 4 → ¬ ∃ k : ℤ, b^2 + 4 * b + 3 = k^2 :=
by
  intro b hb
  by_contra h
  obtain ⟨k, hk⟩ := h
  have : b^2 + 4 * b + 3 = k ^ 2 := hk
  sorry

end no_b_satisfies_143b_square_of_integer_l54_54456


namespace weight_difference_l54_54101

variable (W_A W_D : Nat)

theorem weight_difference : W_A - W_D = 15 :=
by
  -- Given conditions
  have h1 : W_A = 67 := sorry
  have h2 : W_D = 52 := sorry
  -- Proof
  sorry

end weight_difference_l54_54101


namespace race_minimum_participants_l54_54187

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l54_54187


namespace driver_total_distance_is_148_l54_54088

-- Definitions of the distances traveled according to the given conditions
def distance_MWF : ℕ := 12 * 3
def total_distance_MWF : ℕ := distance_MWF * 3
def distance_T : ℕ := 9 * 5 / 2  -- using ℕ for 2.5 hours as 5/2
def distance_Th : ℕ := 7 * 5 / 2

-- Statement of the total distance calculation
def total_distance_week : ℕ :=
  total_distance_MWF + distance_T + distance_Th

-- Theorem stating the total distance traveled during the week
theorem driver_total_distance_is_148 : total_distance_week = 148 := by
  sorry

end driver_total_distance_is_148_l54_54088


namespace kan_krao_park_walkways_l54_54653

-- Definitions for the given conditions
structure Park (α : Type*) := 
  (entrances : Finset α)
  (walkways : α → α → Prop)
  (brick_paved : α → α → Prop)
  (asphalt_paved : α → α → Prop)
  (no_three_intersections : ∀ (x y z w : α), x ≠ y → y ≠ z → z ≠ w → w ≠ x → (walkways x y ∧ walkways z w) → ¬ (walkways x z ∧ walkways y w))

-- Conditions based on the given problem
variables {α : Type*} [Finite α] [DecidableRel (@walkways α)]
variable (p : Park α)
variables [Fintype α]

-- Translate conditions to definitions
def has_lotuses (p : α → α → Prop) (q : α → α → Prop) (x y : α) : Prop := p x y ∧ p x y
def has_waterlilies (p : α → α → Prop) (q : α → α → Prop) (x y : α) : Prop := (p x y ∧ q x y) ∨ (q x y ∧ p x y)
def is_lit (p : α → α → Prop) (q : α → α → Prop) : Prop := ∃ (x y : α), x ≠ y ∧ (has_lotuses p q x y ∧ has_lotuses p q x y ∧ ∃ sz, sz ≥ 45)

-- Mathematically equivalent proof problem
theorem kan_krao_park_walkways (p : Park α) :
  (∃ walkways_same_material : α → α → Prop, ∃ (lit_walkways : Finset (α × α)), lit_walkways.card ≥ 11) :=
sorry

end kan_krao_park_walkways_l54_54653


namespace vertical_asymptote_values_l54_54295

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := (x^2 - x + c) / (x^2 + x - 20)

theorem vertical_asymptote_values (c : ℝ) :
  (∃ x : ℝ, (x^2 + x - 20 = 0 ∧ x^2 - x + c = 0) ↔
   (c = -12 ∨ c = -30)) := sorry

end vertical_asymptote_values_l54_54295


namespace simplify_expression_l54_54257

def a : ℕ := 1050
def p : ℕ := 2101
def q : ℕ := 1050 * 1051

theorem simplify_expression : 
  (1051 / 1050) - (1050 / 1051) = (p : ℚ) / (q : ℚ) ∧ Nat.gcd p a = 1 ∧ Nat.gcd p (a + 1) = 1 :=
by 
  sorry

end simplify_expression_l54_54257


namespace emptying_rate_l54_54712

theorem emptying_rate (fill_time1 : ℝ) (total_fill_time : ℝ) (T : ℝ) 
  (h1 : fill_time1 = 4) 
  (h2 : total_fill_time = 20) 
  (h3 : 1 / fill_time1 - 1 / T = 1 / total_fill_time) :
  T = 5 :=
by
  sorry

end emptying_rate_l54_54712


namespace division_of_negatives_l54_54451

theorem division_of_negatives : (-500 : ℤ) / (-50 : ℤ) = 10 := by
  sorry

end division_of_negatives_l54_54451


namespace heloise_total_pets_l54_54485

-- Define initial data
def ratio_dogs_to_cats := (10, 17)
def dogs_given_away := 10
def dogs_remaining := 60

-- Definition of initial number of dogs based on conditions
def initial_dogs := dogs_remaining + dogs_given_away

-- Definition based on ratio of dogs to cats
def dogs_per_set := ratio_dogs_to_cats.1
def cats_per_set := ratio_dogs_to_cats.2

-- Compute the number of sets of dogs
def sets_of_dogs := initial_dogs / dogs_per_set

-- Compute the number of cats
def initial_cats := sets_of_dogs * cats_per_set

-- Definition of the total number of pets
def total_pets := dogs_remaining + initial_cats

-- Lean statement for the proof
theorem heloise_total_pets :
  initial_dogs = 70 ∧
  sets_of_dogs = 7 ∧
  initial_cats = 119 ∧
  total_pets = 179 :=
by
  -- The statements to be proved are listed as conjunctions (∧)
  sorry

end heloise_total_pets_l54_54485


namespace area_of_square_l54_54573

-- Define the conditions given in the problem
def radius_circle := 7 -- radius of each circle in inches

def diameter_circle := 2 * radius_circle -- diameter of each circle

def side_length_square := 2 * diameter_circle -- side length of the square

-- State the theorem we want to prove
theorem area_of_square : side_length_square ^ 2 = 784 := 
by
  sorry

end area_of_square_l54_54573


namespace problem1_problem2_l54_54271

-- First Problem Statement:
theorem problem1 :  12 - (-18) + (-7) - 20 = 3 := 
by 
  sorry

-- Second Problem Statement:
theorem problem2 : -4 / (1 / 2) * 8 = -64 := 
by 
  sorry

end problem1_problem2_l54_54271


namespace problem_statement_l54_54478

theorem problem_statement (x : ℝ) (h : x^2 + 4 * x - 2 = 0) : 3 * x^2 + 12 * x - 23 = -17 :=
sorry

end problem_statement_l54_54478


namespace solve_x_eq_l54_54932

theorem solve_x_eq : ∃ x : ℚ, -3 * x - 12 = 6 * x + 9 ∧ x = -7 / 3 :=
by 
  sorry

end solve_x_eq_l54_54932


namespace last_two_digits_7_pow_2017_l54_54663

noncomputable def last_two_digits_of_pow :=
  ∀ n : ℕ, ∃ (d : ℕ), d < 100 ∧ 7^n % 100 = d

theorem last_two_digits_7_pow_2017 : ∃ (d : ℕ), d = 7 ∧ 7^2017 % 100 = d :=
by
  sorry

end last_two_digits_7_pow_2017_l54_54663


namespace find_fraction_of_number_l54_54888

theorem find_fraction_of_number (N : ℚ) (h : (3/10 : ℚ) * N - 8 = 12) :
  (1/5 : ℚ) * N = 40 / 3 :=
by
  sorry

end find_fraction_of_number_l54_54888


namespace negation_of_existence_lt_zero_l54_54359

theorem negation_of_existence_lt_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by sorry

end negation_of_existence_lt_zero_l54_54359


namespace min_value_expression_l54_54914

theorem min_value_expression (n : ℕ) (h : 0 < n) : 
  ∃ (m : ℕ), (m = n) ∧ (∀ k > 0, (k = n) -> (n / 3 + 27 / n) = 6) := 
sorry

end min_value_expression_l54_54914


namespace triangle_ABC_area_l54_54642

-- We define the basic structure of a triangle and its properties
structure Triangle :=
(base : ℝ)
(height : ℝ)
(right_angled_at : ℝ)

-- Define the specific triangle ABC with given properties
def triangle_ABC : Triangle := {
  base := 12,
  height := 15,
  right_angled_at := 90 -- since right-angled at C
}

-- Given conditions, we need to prove the area is 90 square cm
theorem triangle_ABC_area : 1/2 * triangle_ABC.base * triangle_ABC.height = 90 := 
by 
  sorry

end triangle_ABC_area_l54_54642


namespace distance_between_skew_lines_l54_54807

-- Definitions for the geometric configuration
def AB : ℝ := 4
def AA1 : ℝ := 4
def AD : ℝ := 3

-- Theorem statement to prove the distance between skew lines A1D and B1D1
theorem distance_between_skew_lines:
  ∃ d : ℝ, d = (6 * Real.sqrt 34) / 17 :=
sorry

end distance_between_skew_lines_l54_54807


namespace num_and_sum_of_divisors_of_36_l54_54320

noncomputable def num_divisors_and_sum (n : ℕ) : ℕ × ℕ :=
  let divisors := (List.range (n + 1)).filter (λ x => n % x = 0)
  (divisors.length, divisors.sum)

theorem num_and_sum_of_divisors_of_36 : num_divisors_and_sum 36 = (9, 91) := by
  sorry

end num_and_sum_of_divisors_of_36_l54_54320


namespace work_completion_time_l54_54561

-- Define the rate of work done by a, b, and c.
def rate_a := 1 / 4
def rate_b := 1 / 12
def rate_c := 1 / 6

-- Define the time each person starts working and the cycle pattern.
def start_time : ℕ := 6 -- in hours
def cycle_pattern := [rate_a, rate_b, rate_c]

-- Calculate the total amount of work done in one cycle of 3 hours.
def work_per_cycle := (rate_a + rate_b + rate_c)

-- Calculate the total time to complete the work.
def total_time_to_complete_work := 2 * 3 -- number of cycles times 3 hours per cycle

-- Calculate the time of completion.
def completion_time := start_time + total_time_to_complete_work

-- Theorem to prove the work completion time.
theorem work_completion_time : completion_time = 12 := 
by
  -- Proof can be filled in here
  sorry

end work_completion_time_l54_54561


namespace math_problem_l54_54007

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l54_54007


namespace race_participants_minimum_l54_54175

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l54_54175


namespace dot_product_range_l54_54506

theorem dot_product_range (a b : ℝ) (θ : ℝ) (h1 : a = 8) (h2 : b = 12)
  (h3 : 30 * (Real.pi / 180) ≤ θ ∧ θ ≤ 60 * (Real.pi / 180)) :
  48 * Real.sqrt 3 ≤ a * b * Real.cos θ ∧ a * b * Real.cos θ ≤ 48 :=
by
  sorry

end dot_product_range_l54_54506


namespace positive_difference_of_solutions_l54_54599

theorem positive_difference_of_solutions:
  ∀ (s : ℝ), s ≠ -3 → (s^2 - 5*s - 24) / (s + 3) = 3*s + 10 →
  abs (-1 - (-27)) = 26 :=
by
  sorry

end positive_difference_of_solutions_l54_54599


namespace tangency_point_of_parabolas_l54_54913

theorem tangency_point_of_parabolas :
  ∃ (x y : ℝ), y = x^2 + 17 * x + 40 ∧ x = y^2 + 51 * y + 650 ∧ x = -7 ∧ y = -25 :=
by
  sorry

end tangency_point_of_parabolas_l54_54913


namespace probability_one_defective_item_l54_54607

theorem probability_one_defective_item : 
  let total_number_of_ways := Nat.choose 6 3,
      ways_with_one_defective := Nat.choose 4 2 * Nat.choose 2 1,
      probability := (ways_with_one_defective : ℚ) / total_number_of_ways
  in probability = 3 / 5 := 
by
  let total_number_of_ways := Nat.choose 6 3
  let ways_with_one_defective := Nat.choose 4 2 * Nat.choose 2 1
  let probability := (ways_with_one_defective : ℚ) / total_number_of_ways
  have h1 : total_number_of_ways = 20 := by sorry
  have h2 : ways_with_one_defective = 12 := by sorry
  have h3 : probability = 12 / 20 := by sorry
  have h4 : (12 / 20 : ℚ) = 3 / 5 := by sorry
  exact h4

end probability_one_defective_item_l54_54607


namespace probability_distance_ge_one_l54_54950

theorem probability_distance_ge_one (S : set ℝ) (side_length_S : ∀ x ∈ S, x = 2)
  (P : ℝ) : 
  -- Assuming two points are chosen independently at random on the sides of a square S of side length 2
  let prob := (26 - Real.pi) / 32 in
    P = prob := 
sorry

end probability_distance_ge_one_l54_54950


namespace solve_eqn_in_integers_l54_54369

theorem solve_eqn_in_integers :
  ∃ (x y : ℤ), xy + 3*x - 5*y = -3 ∧ 
  ((x, y) = (6, 9) ∨ (x, y) = (7, 3) ∨ (x, y) = (8, 1) ∨ 
  (x, y) = (9, 0) ∨ (x, y) = (11, -1) ∨ (x, y) = (17, -2) ∨ 
  (x, y) = (4, -15) ∨ (x, y) = (3, -9) ∨ (x, y) = (2, -7) ∨ 
  (x, y) = (1, -6) ∨ (x, y) = (-1, -5) ∨  (x, y) = (-7, -4)) :=
sorry

end solve_eqn_in_integers_l54_54369


namespace xiao_li_more_stable_l54_54584

def average_xiao_li : ℝ := 95
def average_xiao_zhang : ℝ := 95

def variance_xiao_li : ℝ := 0.55
def variance_xiao_zhang : ℝ := 1.35

theorem xiao_li_more_stable : 
  variance_xiao_li < variance_xiao_zhang :=
by
  sorry

end xiao_li_more_stable_l54_54584


namespace vector_addition_result_l54_54929

-- Define the given vectors
def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-3, 4)

-- Statement to prove that the sum of the vectors is (-1, 5)
theorem vector_addition_result : vector_a + vector_b = (-1, 5) :=
by
  -- Use the fact that vector addition in ℝ^2 is component-wise
  sorry

end vector_addition_result_l54_54929


namespace alice_questions_wrong_l54_54575

theorem alice_questions_wrong (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a + d = b + c + 3) 
  (h3 : c = 7) : 
  a = 8.5 := 
by
  sorry

end alice_questions_wrong_l54_54575


namespace proof_problem_l54_54131

def pos_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
∀ n, 4 * S n = (a n + 1) ^ 2

def sequence_condition (a : ℕ → ℝ) : Prop :=
a 0 = 1 ∧ ∀ n, a (n + 1) - a n = 2

def sum_sequence_T (a : ℕ → ℝ) (T : ℕ → ℝ) :=
∀ n, T n = (1 - 1 / (2 * n + 1))

def range_k (T : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n, T n ≥ k → k ≤ 2 / 3

theorem proof_problem (a : ℕ → ℝ) (S T : ℕ → ℝ) (k : ℝ) :
  pos_sequence a S → sequence_condition a → sum_sequence_T a T → range_k T k :=
by sorry

end proof_problem_l54_54131


namespace root_equation_solution_l54_54802

theorem root_equation_solution (a : ℝ) (h : 3 * a^2 - 5 * a - 2 = 0) : 6 * a^2 - 10 * a = 4 :=
by 
  sorry

end root_equation_solution_l54_54802


namespace money_returned_l54_54735

theorem money_returned (individual group taken : ℝ)
  (h1 : individual = 12000)
  (h2 : group = 16000)
  (h3 : taken = 26400) :
  (individual + group - taken) = 1600 :=
by
  -- The proof has been omitted
  sorry

end money_returned_l54_54735


namespace pascal_triangle_count_30_rows_l54_54316

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end pascal_triangle_count_30_rows_l54_54316


namespace difference_of_squares_l54_54411

theorem difference_of_squares :
  535^2 - 465^2 = 70000 :=
by
  sorry

end difference_of_squares_l54_54411


namespace stuart_initial_marbles_is_56_l54_54096

-- Define the initial conditions
def betty_initial_marbles : ℕ := 60
def percentage_given_to_stuart : ℚ := 40 / 100
def stuart_marbles_after_receiving : ℕ := 80

-- Define the calculation of how many marbles Betty gave to Stuart
def marbles_given_to_stuart := (percentage_given_to_stuart * betty_initial_marbles)

-- Define the target: Stuart's initial number of marbles
def stuart_initial_marbles := stuart_marbles_after_receiving - marbles_given_to_stuart

-- Main theorem stating the problem
theorem stuart_initial_marbles_is_56 : stuart_initial_marbles = 56 :=
by 
  sorry

end stuart_initial_marbles_is_56_l54_54096


namespace modulus_of_complex_l54_54756

-- Define the conditions
variables {x y : ℝ}
def i := Complex.I

-- State the conditions of the problem
def condition1 : 1 + x * i = (2 - y) - 3 * i :=
by sorry

-- State the hypothesis and the goal
theorem modulus_of_complex (h : 1 + x * i = (2 - y) - 3 * i) : Complex.abs (x + y * i) = Real.sqrt 10 :=
sorry

end modulus_of_complex_l54_54756


namespace factor_polynomial_l54_54291

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end factor_polynomial_l54_54291


namespace modular_expression_problem_l54_54957

theorem modular_expression_problem
  (m : ℕ)
  (hm : 0 ≤ m ∧ m < 29)
  (hmod : 4 * m % 29 = 1) :
  (5^m % 29)^4 - 3 % 29 = 13 % 29 :=
by
  sorry

end modular_expression_problem_l54_54957


namespace systematic_sampling_first_group_draw_l54_54991

noncomputable def index_drawn_from_group (x n : ℕ) : ℕ := x + 8 * (n - 1)

theorem systematic_sampling_first_group_draw (k : ℕ) (fifteenth_group : index_drawn_from_group k 15 = 116) :
  index_drawn_from_group k 1 = 4 := 
sorry

end systematic_sampling_first_group_draw_l54_54991


namespace hyperbola_focal_length_l54_54058

theorem hyperbola_focal_length (m : ℝ) : 
  (∀ x y : ℝ, (x^2 / m - y^2 / 4 = 1)) ∧ (∀ f : ℝ, f = 6) → m = 5 := 
  by 
    -- Using the condition that the focal length is 6
    sorry

end hyperbola_focal_length_l54_54058


namespace find_function_l54_54461

theorem find_function (f : ℕ → ℕ)
  (h1 : ∀ m n : ℕ, f (m^2 + n^2) = f m ^ 2 + f n ^ 2)
  (h2 : f 1 > 0) : ∀ n : ℕ, f n = n := 
sorry

end find_function_l54_54461


namespace Elizabeth_More_Revenue_Than_Banks_l54_54166

theorem Elizabeth_More_Revenue_Than_Banks : 
  let banks_investments := 8
  let banks_revenue_per_investment := 500
  let elizabeth_investments := 5
  let elizabeth_revenue_per_investment := 900
  let banks_total_revenue := banks_investments * banks_revenue_per_investment
  let elizabeth_total_revenue := elizabeth_investments * elizabeth_revenue_per_investment
  elizabeth_total_revenue - banks_total_revenue = 500 :=
by
  sorry

end Elizabeth_More_Revenue_Than_Banks_l54_54166


namespace min_value_x2_y2_z2_l54_54959

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  4 ≤ x^2 + y^2 + z^2 :=
sorry

end min_value_x2_y2_z2_l54_54959


namespace probability_odd_product_lt_one_eighth_l54_54226

theorem probability_odd_product_lt_one_eighth :
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  p < 1 / 8 :=
by
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  sorry

end probability_odd_product_lt_one_eighth_l54_54226


namespace speed_in_still_water_l54_54892

-- Definitions for the conditions
def upstream_speed : ℕ := 30
def downstream_speed : ℕ := 60

-- Prove that the speed of the man in still water is 45 kmph
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 45 := by
  sorry

end speed_in_still_water_l54_54892


namespace estimate_3_sqrt_2_range_l54_54906

theorem estimate_3_sqrt_2_range :
  4 < 3 * Real.sqrt 2 ∧ 3 * Real.sqrt 2 < 5 :=
by
  sorry

end estimate_3_sqrt_2_range_l54_54906


namespace find_triangle_lengths_l54_54231

-- Conditions:
-- 1. Two right-angled triangles are similar.
-- 2. Bigger triangle sides: x + 1 and y + 5, Area larger by 8 cm^2

def triangle_lengths (x y : ℝ) : Prop := 
  (y = 5 * x ∧ 
  (5 / 2) * (x + 1) ^ 2 - (5 / 2) * x ^ 2 = 8)

theorem find_triangle_lengths (x y : ℝ) : triangle_lengths x y ↔ (x = 1.1 ∧ y = 5.5) :=
sorry

end find_triangle_lengths_l54_54231


namespace arithmetic_sequence_integers_l54_54028

theorem arithmetic_sequence_integers (a3 a18 : ℝ) (d : ℝ) (n : ℕ)
  (h3 : a3 = 14) (h18 : a18 = 23) (hd : d = 0.6)
  (hn : n = 2010) : 
  (∃ (k : ℕ), n = 5 * (k + 1) - 2) ∧ (k ≤ 401) :=
by
  sorry

end arithmetic_sequence_integers_l54_54028


namespace activity_support_probabilities_l54_54711

theorem activity_support_probabilities :
  let boys_support_A := 200 / (200 + 400) in
  let girls_support_A := 300 / (300 + 100) in
  let P_boys_support_A := 1 / 3 in
  let P_girls_support_A := 3 / 4 in
  ∀ (total_boys_total_girls total_boys total_girls : ℕ) 
    (two_boys_support_A one_girl_support_A : ℚ),
    two_boys_support_A = P_boys_support_A^2 * (1 - P_girls_support_A) ∧
    one_girl_support_A = (2 * P_boys_support_A * (1 - P_boys_support_A) * P_girls_support_A) ∧
    (two_boys_support_A + one_girl_support_A = 13 / 36) ∧
    (total_boys_total_girls = 500 + 300) ∧
    (total_boys = 500) ∧
    (total_girls = 300) ∧
    (P_b0 = (350 + 150) / (350 + 250 + 150 + 250)) ∧
    (p0 = 1 / 2) →
    ∃ (a : ℕ) (p0 p1 : ℚ), 
      p0 = 1 / 2 ∧
      p1 = (a - 808) / (2 * (a - 800)) ∧
      p0 > p1
| boys_support_A girls_support_A P_boys_support_A P_girls_support_A 
  total_boys_total_girls total_boys total_girls two_boys_support_A one_girl_support_A P_b0 p0 :=
sorry

end activity_support_probabilities_l54_54711


namespace estimate_households_above_320_units_proved_l54_54024

noncomputable def estimate_households_above_320_units
    (households : ℕ)
    (μ σ : ℝ)
    (prob_interval : ℝ)
    (prob_above : ℝ)
    (h : households = 1000 ∧ μ = 300 ∧ σ = 10 ∧ prob_interval = 0.9544 ∧ prob_above = 0.0228)
    : Prop :=
  let expected_households := households * prob_above in
  expected_households = 23

theorem estimate_households_above_320_units_proved
  : estimate_households_above_320_units 1000 300 10 0.9544 0.0228 :=
by 
  sorry

end estimate_households_above_320_units_proved_l54_54024


namespace problem_statement_l54_54214

theorem problem_statement (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ) 
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) : 
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := 
sorry

end problem_statement_l54_54214


namespace difference_of_roots_l54_54463

theorem difference_of_roots :
  ∀ (x : ℝ), (x^2 - 5*x + 6 = 0) → (∃ r1 r2 : ℝ, r1 > 2 ∧ r2 < r1 ∧ r1 - r2 = 1) :=
by
  sorry

end difference_of_roots_l54_54463


namespace scrabble_middle_letter_value_l54_54497

theorem scrabble_middle_letter_value 
  (triple_word_score : ℕ) (single_letter_value : ℕ) (middle_letter_value : ℕ)
  (h1 : triple_word_score = 30)
  (h2 : single_letter_value = 1)
  : 3 * (2 * single_letter_value + middle_letter_value) = triple_word_score → middle_letter_value = 8 :=
by
  sorry

end scrabble_middle_letter_value_l54_54497


namespace rectangle_bounds_product_l54_54856

theorem rectangle_bounds_product (b : ℝ) :
  (∃ b, y = 3 ∧ y = 7 ∧ x = -1 ∧ (x = b) 
   → (b = 3 ∨ b = -5) 
    ∧ (3 * -5 = -15)) :=
sorry

end rectangle_bounds_product_l54_54856


namespace cos_7pi_over_6_l54_54458

noncomputable def cos_seven_pi_six : ℝ := -real.cos (real.pi / 6)

theorem cos_7pi_over_6 : real.cos (7 * real.pi / 6) = cos_seven_pi_six := by
  -- skipped proof
  sorry

end cos_7pi_over_6_l54_54458


namespace complete_square_solution_l54_54692

theorem complete_square_solution :
  ∀ x : ℝ, x^2 - 4 * x - 22 = 0 → (x - 2)^2 = 26 :=
by
  intro x h
  sorry

end complete_square_solution_l54_54692


namespace find_third_side_l54_54255

theorem find_third_side
  (cubes : ℕ) (cube_volume : ℚ) (side1 side2 : ℚ)
  (fits : cubes = 24) (vol_cube : cube_volume = 27)
  (dim1 : side1 = 8) (dim2 : side2 = 9) :
  (side1 * side2 * (cube_volume * cubes) / (side1 * side2)) = 9 := by
  sorry

end find_third_side_l54_54255


namespace Stuart_initial_marbles_l54_54100

variable (Betty_marbles Stuart_final increased_by: ℤ) 

-- Conditions as definitions
def Betty_has : Betty_marbles = 60 := sorry 
def Stuart_collect_increase : Stuart_final = 80 := sorry 
def percentage_given : ∃ x, x = (40 * Betty_marbles) / 100 := sorry 

-- Theorem to prove Stuart had 56 marbles initially
theorem Stuart_initial_marbles 
  (h1 : Betty_has)
  (h2 : Stuart_collect_increase)
  (h3 : percentage_given) :
  ∃ y, y = Stuart_final - 24 := 
sorry

end Stuart_initial_marbles_l54_54100


namespace diff_of_squares_535_465_l54_54408

theorem diff_of_squares_535_465 : (535^2 - 465^2) = 70000 :=
sorry

end diff_of_squares_535_465_l54_54408


namespace tan_triple_angle_l54_54791

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l54_54791


namespace tennis_tournament_rounds_l54_54726

/-- Defining the constants and conditions stated in the problem -/
def first_round_games : ℕ := 8
def second_round_games : ℕ := 4
def third_round_games : ℕ := 2
def finals_games : ℕ := 1
def cans_per_game : ℕ := 5
def balls_per_can : ℕ := 3
def total_balls_used : ℕ := 225

/-- Theorem stating the number of rounds in the tennis tournament -/
theorem tennis_tournament_rounds : 
  first_round_games + second_round_games + third_round_games + finals_games = 15 ∧
  15 * cans_per_game = 75 ∧
  75 * balls_per_can = total_balls_used →
  4 = 4 :=
by sorry

end tennis_tournament_rounds_l54_54726


namespace k_cannot_be_zero_l54_54467

theorem k_cannot_be_zero (k : ℝ) (h₁ : k ≠ 0) (h₂ : 4 - 2 * k > 0) : k ≠ 0 :=
by 
  exact h₁

end k_cannot_be_zero_l54_54467


namespace john_total_skateboarded_distance_l54_54817

noncomputable def total_skateboarded_distance (to_park: ℕ) (back_home: ℕ) : ℕ :=
  to_park + back_home

theorem john_total_skateboarded_distance :
  total_skateboarded_distance 10 10 = 20 :=
by
  sorry

end john_total_skateboarded_distance_l54_54817


namespace contradiction_proof_l54_54869

theorem contradiction_proof (a b c d : ℝ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 1) (h4 : d = 1) (h5 : a * c + b * d > 1) : ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by 
  sorry

end contradiction_proof_l54_54869


namespace pascal_triangle_elements_count_l54_54314

theorem pascal_triangle_elements_count :
  ∑ n in finset.range 30, (n + 1) = 465 :=
by 
  sorry

end pascal_triangle_elements_count_l54_54314


namespace a_gt_abs_b_suff_not_necc_l54_54391

theorem a_gt_abs_b_suff_not_necc (a b : ℝ) (h : a > |b|) : 
  a^2 > b^2 ∧ ∀ a b : ℝ, (a^2 > b^2 → |a| > |b|) → ¬ (a < -|b|) := 
by
  sorry

end a_gt_abs_b_suff_not_necc_l54_54391


namespace xiaoqiang_xiaolin_stamps_l54_54238

-- Definitions for initial conditions and constraints
noncomputable def x : ℤ := 227
noncomputable def y : ℤ := 221
noncomputable def k : ℤ := sorry

-- Proof problem as a theorem
theorem xiaoqiang_xiaolin_stamps:
  x + y > 400 ∧
  x - k = (13 / 19) * (y + k) ∧
  y - k = (11 / 17) * (x + k) ∧
  x = 227 ∧ 
  y = 221 :=
by
  sorry

end xiaoqiang_xiaolin_stamps_l54_54238


namespace factorization_correct_l54_54288

-- Define noncomputable to deal with the natural arithmetic operations
noncomputable def a : ℕ := 66
noncomputable def b : ℕ := 231

-- Define the given expressions
noncomputable def lhs (x : ℕ) : ℤ := ((a : ℤ) * x^6) - ((b : ℤ) * x^12)
noncomputable def rhs (x : ℕ) : ℤ := (33 : ℤ) * x^6 * (2 - 7 * x^6)

-- The theorem to prove the equality
theorem factorization_correct (x : ℕ) : lhs x = rhs x :=
by sorry

end factorization_correct_l54_54288


namespace rowing_time_ratio_l54_54860

theorem rowing_time_ratio
  (V_b : ℝ) (V_s : ℝ) (V_upstream : ℝ) (V_downstream : ℝ) (T_upstream T_downstream : ℝ)
  (h1 : V_b = 39) (h2 : V_s = 13)
  (h3 : V_upstream = V_b - V_s) (h4 : V_downstream = V_b + V_s)
  (h5 : T_upstream * V_upstream = T_downstream * V_downstream) :
  T_upstream / T_downstream = 2 := by
  sorry

end rowing_time_ratio_l54_54860


namespace probability_x_lt_2y_l54_54433

noncomputable def rectangle_area : ℝ := 5 * 2

noncomputable def triangle_area : ℝ := 1/2 * 2 * 4

theorem probability_x_lt_2y :
  (triangle_area / rectangle_area) = 2 / 5 :=
by
  sorry

end probability_x_lt_2y_l54_54433


namespace incorrect_number_read_as_l54_54376

theorem incorrect_number_read_as (n a_incorrect a_correct correct_number incorrect_number : ℕ) 
(hn : n = 10) (h_inc_avg : a_incorrect = 18) (h_cor_avg : a_correct = 22) (h_cor_num : correct_number = 66) :
incorrect_number = 26 := by
  sorry

end incorrect_number_read_as_l54_54376


namespace min_value_of_k_l54_54027

-- Given definitions based on conditions
variable (students : Finset ℕ) (clubs : Finset (Finset ℕ))
variable (student_count : ℕ := 1200)
variable (club_membership_count : ℕ := 23)
variable (k : ℕ)

-- Condition: There are 1200 students
axiom h1 : students.card = student_count

-- Condition: Each student must join exactly k clubs
axiom h2 : ∀ s ∈ students, (Finset.filter (λ c, s ∈ c) clubs).card = k

-- Condition: Each club is joined by exactly 23 students
axiom h3 : ∀ c ∈ clubs, c.card = club_membership_count

-- Condition: No club is joined by all 1200 students
axiom h4 : ∀ c ∈ clubs, ¬(students ⊆ c)

-- Prove the smallest possible value of k is 23
theorem min_value_of_k : k = 23 :=
sorry

end min_value_of_k_l54_54027


namespace tan_triple_angle_l54_54779

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l54_54779


namespace asymptote_slope_of_hyperbola_l54_54452

theorem asymptote_slope_of_hyperbola :
  ∀ (x y : ℝ), (x ≠ 0) ∧ (y/x = 3/4 ∨ y/x = -3/4) ↔ (x^2 / 144 - y^2 / 81 = 1) := 
by
  sorry

end asymptote_slope_of_hyperbola_l54_54452


namespace inequality_proof_l54_54173

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x / (y + z)) + (y / (z + x)) + (z / (x + y)) ≥ (3 / 2) :=
sorry

end inequality_proof_l54_54173


namespace other_acute_angle_in_right_triangle_l54_54332

theorem other_acute_angle_in_right_triangle (a : ℝ) (h : a = 25) :
    ∃ b : ℝ, b = 65 :=
by
  sorry

end other_acute_angle_in_right_triangle_l54_54332


namespace proof_expression_value_l54_54277

noncomputable def a : ℝ := 0.15
noncomputable def b : ℝ := 0.06
noncomputable def x : ℝ := a^3
noncomputable def y : ℝ := b^3
noncomputable def z : ℝ := a^2
noncomputable def w : ℝ := b^2

theorem proof_expression_value :
  ( (x - y) / (z + w) ) + 0.009 + w^4 = 0.1300341679616 := sorry

end proof_expression_value_l54_54277


namespace tan_triple_angle_l54_54780

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l54_54780


namespace arithmetic_to_geometric_find_original_numbers_l54_54547

theorem arithmetic_to_geometric (k : ℕ) 
  (h1 : ∃ k, (3 * k + 1) * 4 * k = 5 * k * (3 * k + 1))
  : k = 5 :=
begin
  sorry,
end

theorem find_original_numbers :
  ∃ (a b c : ℕ), a = 15 ∧ b = 20 ∧ c = 25 :=
begin
  use [15, 20, 25],
  exact ⟨rfl, rfl, rfl⟩
end

end arithmetic_to_geometric_find_original_numbers_l54_54547


namespace right_triangle_acute_angle_l54_54333

/-- In a right triangle, if one acute angle is 25 degrees, then the measure of the other acute angle is 65 degrees. -/
theorem right_triangle_acute_angle (α : ℝ) (hα : α = 25) : ∃ β : ℝ, β = 65 := 
by
  have h_sum : α + 65 = 90 := by
    rw hα
    norm_num
  use 65
  exact h_sum.symm
  sorry

end right_triangle_acute_angle_l54_54333


namespace probability_product_multiple_of_4_l54_54875

theorem probability_product_multiple_of_4 :
  (∃ (cards : Finset ℕ) (h : cards = {1, 2, 3, 4, 5, 6}) (drawn : Finset (ℕ × ℕ))
     (h2 : drawn = {⟨1,2⟩, ⟨1,3⟩, ⟨1,4⟩, ⟨1,5⟩, ⟨1,6⟩,
                    ⟨2,3⟩, ⟨2,4⟩, ⟨2,5⟩, ⟨2,6⟩,
                    ⟨3,4⟩, ⟨3,5⟩, ⟨3,6⟩,
                    ⟨4,5⟩, ⟨4,6⟩, ⟨5,6⟩}),
   filter (λ (pair : ℕ × ℕ), (pair.fst * pair.snd) % 4 = 0) drawn).card / drawn.card = 2 / 5 :=
sorry

end probability_product_multiple_of_4_l54_54875


namespace solve_abs_eqn_l54_54211

theorem solve_abs_eqn (y : ℝ) : (|y - 4| + 3 * y = 15) ↔ (y = 19 / 4) :=
by
  sorry

end solve_abs_eqn_l54_54211


namespace min_number_of_participants_l54_54183

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l54_54183


namespace parabola_vertex_coordinates_l54_54532

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, (3 * (x - 7) ^ 2 + 5) = 3 * (x - 7) ^ 2 + 5 := by
  sorry

end parabola_vertex_coordinates_l54_54532


namespace conditional_probability_l54_54060

def P (event : ℕ → Prop) : ℝ := sorry

def A (n : ℕ) : Prop := n = 10000
def B (n : ℕ) : Prop := n = 15000

theorem conditional_probability :
  P A = 0.80 →
  P B = 0.60 →
  P B / P A = 0.75 :=
by
  intros hA hB
  sorry

end conditional_probability_l54_54060


namespace find_sum_invested_l54_54083

theorem find_sum_invested (P : ℝ)
  (h1 : P * 18 / 100 * 2 - P * 12 / 100 * 2 = 504) :
  P = 4200 := 
sorry

end find_sum_invested_l54_54083


namespace specialist_time_l54_54739

def hospital_bed_charge (days : ℕ) (rate : ℕ) : ℕ := days * rate

def total_known_charges (bed_charge : ℕ) (ambulance_charge : ℕ) : ℕ := bed_charge + ambulance_charge

def specialist_minutes (total_bill : ℕ) (known_charges : ℕ) (spec_rate_per_hour : ℕ) : ℕ := 
  ((total_bill - known_charges) / spec_rate_per_hour) * 60 / 2

theorem specialist_time (days : ℕ) (bed_rate : ℕ) (ambulance_charge : ℕ) (spec_rate_per_hour : ℕ) 
(total_bill : ℕ) (known_charges := total_known_charges (hospital_bed_charge days bed_rate) ambulance_charge)
(hospital_days := 3) (bed_charge_per_day := 900) (specialist_rate := 250) 
(ambulance_cost := 1800) (total_cost := 4625) :
  specialist_minutes total_cost known_charges specialist_rate = 15 :=
sorry

end specialist_time_l54_54739


namespace jane_exercises_40_hours_l54_54815

-- Define the conditions
def hours_per_day : ℝ := 1
def days_per_week : ℝ := 5
def weeks : ℝ := 8

-- Define total_hours using the conditions
def total_hours : ℝ := (hours_per_day * days_per_week) * weeks

-- The theorem stating the result
theorem jane_exercises_40_hours :
  total_hours = 40 := by
  sorry

end jane_exercises_40_hours_l54_54815


namespace race_minimum_participants_l54_54185

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l54_54185


namespace real_function_as_sum_of_symmetric_graphs_l54_54835

theorem real_function_as_sum_of_symmetric_graphs (f : ℝ → ℝ) :
  ∃ (g h : ℝ → ℝ), (∀ x, g x + h x = f x) ∧ (∀ x, g x = g (-x)) ∧ (∀ x, h (1 + x) = h (1 - x)) :=
sorry

end real_function_as_sum_of_symmetric_graphs_l54_54835


namespace enemy_defeat_points_l54_54563

theorem enemy_defeat_points 
    (points_per_enemy : ℕ) (total_enemies : ℕ) (undefeated_enemies : ℕ) (defeated : ℕ) (points_earned : ℕ) :
    points_per_enemy = 8 →
    total_enemies = 7 →
    undefeated_enemies = 2 →
    defeated = total_enemies - undefeated_enemies →
    points_earned = defeated * points_per_enemy →
    points_earned = 40 :=
by
  intros
  sorry

end enemy_defeat_points_l54_54563


namespace part_I_part_II_l54_54305

def f (x a : ℝ) : ℝ := abs (x - a) + abs (2 * x + 1)

-- Part (I)
theorem part_I (x : ℝ) : f x 1 ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

-- Part (II)
theorem part_II (a : ℝ) : (∃ x ∈ Set.Ici a, f x a ≤ 2 * a + x) ↔ a ≥ 1 :=
by sorry

end part_I_part_II_l54_54305


namespace count_5_numbers_after_996_l54_54455

theorem count_5_numbers_after_996 : 
  ∃ a b c d e, a = 997 ∧ b = 998 ∧ c = 999 ∧ d = 1000 ∧ e = 1001 :=
sorry

end count_5_numbers_after_996_l54_54455


namespace John_needs_more_days_l54_54343

theorem John_needs_more_days (days_worked : ℕ) (amount_earned : ℕ) :
  days_worked = 10 ∧ amount_earned = 250 ∧ 
  (∀ d : ℕ, d < days_worked → amount_earned / days_worked = amount_earned / 10) →
  ∃ more_days : ℕ, more_days = 10 ∧ amount_earned * 2 = (days_worked + more_days) * (amount_earned / days_worked) :=
sorry

end John_needs_more_days_l54_54343


namespace minimize_fraction_l54_54065

theorem minimize_fraction (n : ℕ) (h : 0 < n) : 
  (n = 9) → (∀ m : ℕ, 0 < m → (n = m) → (3 * m + 27 / m ≥ 6)) := sorry

end minimize_fraction_l54_54065


namespace ceil_neg_sqrt_fraction_l54_54121

theorem ceil_neg_sqrt_fraction :
  (⌈-real.sqrt (64 / 9)⌉ = -2) :=
by
  -- Define the necessary conditions
  have h1 : real.sqrt (64 / 9) = 8 / 3 := by sorry,
  have h2 : -real.sqrt (64 / 9) = -8 / 3 := by sorry,
  -- Apply the ceiling function and prove the result
  exact sorry

end ceil_neg_sqrt_fraction_l54_54121


namespace find_constants_l54_54122

theorem find_constants (P Q R : ℚ) 
  (h : ∀ x : ℚ, x ≠ 4 → x ≠ 2 → (5 * x + 1) / ((x - 4) * (x - 2) ^ 2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) :
  P = 21 / 4 ∧ Q = 15 ∧ R = -11 / 2 :=
by
  sorry

end find_constants_l54_54122


namespace runners_meetings_on_track_l54_54232

def number_of_meetings (speed1 speed2 laps : ℕ) : ℕ := ((speed1 + speed2) * laps) / (2 * (speed2 - speed1))

theorem runners_meetings_on_track 
  (speed1 speed2 : ℕ) 
  (start_laps : ℕ)
  (speed1_spec : speed1 = 4) 
  (speed2_spec : speed2 = 10) 
  (laps_spec : start_laps = 28) : 
  number_of_meetings speed1 speed2 start_laps = 77 := 
by
  rw [speed1_spec, speed2_spec, laps_spec]
  -- Add further necessary steps or lemmas if required to reach the final proving statement
  sorry

end runners_meetings_on_track_l54_54232


namespace incorrect_statement_D_l54_54080

-- Definitions based on conditions
def length_of_spring (x : ℝ) : ℝ := 8 + 0.5 * x

-- Incorrect Statement (to be proved as incorrect)
def statement_D_incorrect : Prop :=
  ¬ (length_of_spring 30 = 23)

-- Main theorem statement
theorem incorrect_statement_D : statement_D_incorrect :=
by
  sorry

end incorrect_statement_D_l54_54080


namespace determinant_zero_l54_54373

variables (a b c d : ℝ^3) (α β γ : ℝ) (D : ℝ)

-- Define the determinant condition
def matrix_det (M : Matrix (Fin 3) (Fin 3) ℝ) : ℝ :=
  Matrix.det M

theorem determinant_zero (a b c d : ℝ^3) (α β γ : ℝ) : 
  matrix_det ![![α * (a × b), β * (b × c), γ * (c × (a × d))]] = 0 := 
by {
  -- providing the proof is not necessary
  sorry,
}

end determinant_zero_l54_54373


namespace initial_cell_count_l54_54396

theorem initial_cell_count (f : ℕ → ℕ) (h₁ : ∀ n, f (n + 1) = 2 * (f n - 2)) (h₂ : f 5 = 164) : f 0 = 9 :=
sorry

end initial_cell_count_l54_54396


namespace find_p7_value_l54_54278

def quadratic (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem find_p7_value (d e f : ℝ)
  (h1 : quadratic d e f 1 = 4)
  (h2 : quadratic d e f 2 = 4) :
  quadratic d e f 7 = 5 := by
  sorry

end find_p7_value_l54_54278


namespace number_of_odd_positive_integer_triples_sum_25_l54_54486

theorem number_of_odd_positive_integer_triples_sum_25 :
  ∃ n : ℕ, (
    n = 78 ∧
    ∃ (a b c : ℕ), 
      (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 25
  ) := 
sorry

end number_of_odd_positive_integer_triples_sum_25_l54_54486


namespace intersection_roots_l54_54527

theorem intersection_roots :
  x^2 - 4*x - 5 = 0 → (x = 5 ∨ x = -1) := by
  sorry

end intersection_roots_l54_54527


namespace GCF_LCM_18_30_10_45_eq_90_l54_54161

-- Define LCM and GCF functions
def LCM (a b : ℕ) := a / Nat.gcd a b * b
def GCF (a b : ℕ) := Nat.gcd a b

-- Define the problem
theorem GCF_LCM_18_30_10_45_eq_90 : 
  GCF (LCM 18 30) (LCM 10 45) = 90 := by
sorry

end GCF_LCM_18_30_10_45_eq_90_l54_54161


namespace find_multiplier_l54_54801

theorem find_multiplier (x n : ℤ) (h : 2 * n + 20 = x * n - 4) (hn : n = 4) : x = 8 :=
by
  sorry

end find_multiplier_l54_54801


namespace range_of_function_l54_54751

open Real

noncomputable def f (x : ℝ) : ℝ := -cos x ^ 2 - 4 * sin x + 6

theorem range_of_function : 
  ∀ y, (∃ x, y = f x) ↔ 2 ≤ y ∧ y ≤ 10 :=
by
  sorry

end range_of_function_l54_54751


namespace c_value_l54_54326

theorem c_value (c : ℝ) : (∃ a : ℝ, (x : ℝ) → x^2 + 200 * x + c = (x + a)^2) → c = 10000 := 
by
  intro h
  sorry

end c_value_l54_54326


namespace unpainted_unit_cubes_l54_54426

theorem unpainted_unit_cubes (total_cubes painted_faces edge_overlaps corner_overlaps : ℕ) :
  total_cubes = 6 * 6 * 6 ∧
  painted_faces = 6 * (2 * 6) ∧
  edge_overlaps = 12 * 3 / 2 ∧
  corner_overlaps = 8 ∧
  total_cubes - (painted_faces - edge_overlaps - corner_overlaps) = 170 :=
by
  sorry

end unpainted_unit_cubes_l54_54426


namespace cheapest_shipping_option_l54_54448

/-- Defines the cost options for shipping, given a weight of 5 pounds. -/
def cost_A (weight : ℕ) : ℝ := 5.00 + 0.80 * weight
def cost_B (weight : ℕ) : ℝ := 4.50 + 0.85 * weight
def cost_C (weight : ℕ) : ℝ := 3.00 + 0.95 * weight

/-- Proves that for a package weighing 5 pounds, the cheapest shipping option is Option C costing $7.75. -/
theorem cheapest_shipping_option : cost_C 5 < cost_A 5 ∧ cost_C 5 < cost_B 5 ∧ cost_C 5 = 7.75 :=
by
  -- Calculation is omitted
  sorry

end cheapest_shipping_option_l54_54448


namespace kira_memory_space_is_140_l54_54820

def kira_songs_memory_space 
  (n_m : ℕ) -- number of songs downloaded in the morning
  (n_d : ℕ) -- number of songs downloaded later that day
  (n_n : ℕ) -- number of songs downloaded at night
  (s : ℕ) -- size of each song in MB
  : ℕ := (n_m + n_d + n_n) * s

theorem kira_memory_space_is_140 :
  kira_songs_memory_space 10 15 3 5 = 140 := 
by
  sorry

end kira_memory_space_is_140_l54_54820


namespace correct_option_C_l54_54874

variable (x : ℝ)
variable (hx : 0 < x ∧ x < 1)

theorem correct_option_C : 0 < 1 - x^2 ∧ 1 - x^2 < 1 :=
by
  sorry

end correct_option_C_l54_54874


namespace find_n_from_lcm_gcf_l54_54044

open scoped Classical

noncomputable def LCM (a b : ℕ) : ℕ := sorry
noncomputable def GCF (a b : ℕ) : ℕ := sorry

theorem find_n_from_lcm_gcf (n m : ℕ) (h1 : LCM n m = 48) (h2 : GCF n m = 18) (h3 : m = 16) : n = 54 :=
by sorry

end find_n_from_lcm_gcf_l54_54044


namespace unique_triangle_with_consecutive_sides_and_angle_condition_l54_54052

theorem unique_triangle_with_consecutive_sides_and_angle_condition
    (a b c : ℕ) (A B C : ℝ) (h1 : a < b ∧ b < c)
    (h2 : b = a + 1 ∧ c = a + 2)
    (h3 : C = 2 * B)
    (h4 : ∀ x y z : ℕ, x < y ∧ y < z → y = x + 1 ∧ z = x + 2 → 2 * B = C)
    : ∃! (a b c : ℕ) (A B C : ℝ), (a < b ∧ b < c) ∧ (b = a + 1 ∧ c = a + 2) ∧ (C = 2 * B) :=
  sorry

end unique_triangle_with_consecutive_sides_and_angle_condition_l54_54052


namespace pascal_triangle_sum_l54_54313

theorem pascal_triangle_sum (n : ℕ) (h₀ : n = 29) :
  (∑ k in Finset.range (n + 1), k + 1) = 465 := sorry

end pascal_triangle_sum_l54_54313


namespace skittles_problem_l54_54595

def initial_skittles : ℕ := 76
def shared_skittles : ℕ := 72
def final_skittles (initial shared : ℕ) : ℕ := initial - shared

theorem skittles_problem : final_skittles initial_skittles shared_skittles = 4 := by
  sorry

end skittles_problem_l54_54595


namespace car_distance_l54_54578

theorem car_distance (time_am_18 : ℕ) (time_car_48 : ℕ) (h : time_am_18 = time_car_48) : 
  let distance_am_18 := 18
  let distance_car_48 := 48
  let total_distance_am := 675
  let distance_ratio := (distance_am_18 : ℝ) / (distance_car_48 : ℝ)
  let distance_car := (total_distance_am : ℝ) * (distance_car_48 : ℝ) / (distance_am_18 : ℝ)
  distance_car = 1800 :=
by
  sorry

end car_distance_l54_54578


namespace find_polar_equations_and_distance_l54_54492

noncomputable def polar_equation_C1 (rho theta : ℝ) : Prop :=
  rho^2 * Real.cos (2 * theta) = 1

noncomputable def polar_equation_C2 (rho theta : ℝ) : Prop :=
  rho = 2 * Real.cos theta

theorem find_polar_equations_and_distance :
  (∀ rho theta, polar_equation_C1 rho theta ↔ rho^2 * Real.cos (2 * theta) = 1) ∧
  (∀ rho theta, polar_equation_C2 rho theta ↔ rho = 2 * Real.cos theta) ∧
  let theta := Real.pi / 6
  let rho_A := Real.sqrt 2
  let rho_B := Real.sqrt 3
  (|rho_A - rho_B| = |Real.sqrt 3 - Real.sqrt 2|) :=
  by sorry

end find_polar_equations_and_distance_l54_54492


namespace find_y1_l54_54298

theorem find_y1
  (y1 y2 y3 : ℝ)
  (h1 : 0 ≤ y3)
  (h2 : y3 ≤ y2)
  (h3 : y2 ≤ y1)
  (h4 : y1 ≤ 1)
  (h5 : (1 - y1)^2 + 2 * (y1 - y2)^2 + 2 * (y2 - y3)^2 + y3^2 = 1 / 2) :
  y1 = 3 / 4 :=
sorry

end find_y1_l54_54298


namespace minimum_value_of_function_l54_54912

noncomputable def target_function (x : ℝ) := x^2 + 12 * x + 108 / x^4

theorem minimum_value_of_function : ∃ x > 0, target_function x = 36 :=
by
  use 3
  split
  -- Prove x > 0
  linarith
  -- Prove target_function x = 36
  unfold target_function
  calc
    3^2 + 12 * 3 + 108 / 3^4 = 9 + 36 + 108 / 81 : by norm_num
    ...                       = 9 + 36 + 108 / 81 : by norm_num
    ...                       = 45                : by norm_num
    ...                       = 45                : by norm_num
    ...                       = 36                : sorry

end minimum_value_of_function_l54_54912


namespace pascal_triangle_rows_sum_l54_54317

theorem pascal_triangle_rows_sum :
  ∑ k in finset.range 30, (k + 1) = 465 := by
  sorry

end pascal_triangle_rows_sum_l54_54317


namespace area_of_triangle_tangent_line_l54_54292

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def tangent_line_at_1 (y x : ℝ) : Prop := y = x - 1

theorem area_of_triangle_tangent_line :
  let tangent_intercept_x : ℝ := 1
  let tangent_intercept_y : ℝ := -1
  let area_of_triangle : ℝ := 1 / 2 * tangent_intercept_x * -tangent_intercept_y
  area_of_triangle = 1 / 2 :=
by
  sorry

end area_of_triangle_tangent_line_l54_54292


namespace wrapping_paper_area_correct_l54_54429

variable (w h : ℝ) -- Define the base length and height of the box.

-- Lean statement for the problem asserting that the area of the wrapping paper is \(2(w+h)^2\).
def wrapping_paper_area (w h : ℝ) : ℝ := 2 * (w + h) ^ 2

-- Theorem stating that the derived formula for the area of the wrapping paper is correct.
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  sorry -- Proof is omitted

end wrapping_paper_area_correct_l54_54429


namespace wrapping_paper_area_correct_l54_54428

variable (w h : ℝ) -- Define the base length and height of the box.

-- Lean statement for the problem asserting that the area of the wrapping paper is \(2(w+h)^2\).
def wrapping_paper_area (w h : ℝ) : ℝ := 2 * (w + h) ^ 2

-- Theorem stating that the derived formula for the area of the wrapping paper is correct.
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  sorry -- Proof is omitted

end wrapping_paper_area_correct_l54_54428


namespace convert_spherical_to_rectangular_correct_l54_54454

-- Define the spherical to rectangular conversion functions
noncomputable def spherical_to_rectangular (rho θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin φ * Real.cos θ, rho * Real.sin φ * Real.sin θ, rho * Real.cos φ)

-- Define the given spherical coordinates
def given_spherical_coords : ℝ × ℝ × ℝ :=
  (5, 7 * Real.pi / 4, Real.pi / 3)

-- Define the expected rectangular coordinates
def expected_rectangular_coords : ℝ × ℝ × ℝ :=
  (-5 * Real.sqrt 6 / 4, -5 * Real.sqrt 6 / 4, 5 / 2)

-- The proof statement
theorem convert_spherical_to_rectangular_correct (ρ θ φ : ℝ)
  (h_ρ : ρ = 5) (h_θ : θ = 7 * Real.pi / 4) (h_φ : φ = Real.pi / 3) :
  spherical_to_rectangular ρ θ φ = expected_rectangular_coords :=
by
  -- Proof omitted
  sorry

end convert_spherical_to_rectangular_correct_l54_54454


namespace two_p_plus_q_l54_54417

variable {p q : ℚ}

theorem two_p_plus_q (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 :=
by
  sorry

end two_p_plus_q_l54_54417


namespace tan_triple_angle_l54_54776

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l54_54776


namespace determine_c_l54_54325

theorem determine_c (c : ℝ) 
  (h : ∃ a : ℝ, (∀ x : ℝ, x^2 + 200 * x + c = (x + a)^2)) : c = 10000 :=
sorry

end determine_c_l54_54325


namespace expression_value_l54_54000

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l54_54000


namespace brendan_yards_per_week_l54_54269

def original_speed_flat : ℝ := 8  -- Brendan's speed on flat terrain in yards/day
def improvement_flat : ℝ := 0.5   -- Lawn mower improvement on flat terrain (50%)
def reduction_uneven : ℝ := 0.35  -- Speed reduction on uneven terrain (35%)
def days_flat : ℝ := 4            -- Days on flat terrain
def days_uneven : ℝ := 3          -- Days on uneven terrain

def improved_speed_flat : ℝ := original_speed_flat * (1 + improvement_flat)
def speed_uneven : ℝ := improved_speed_flat * (1 - reduction_uneven)

def total_yards_week : ℝ := (improved_speed_flat * days_flat) + (speed_uneven * days_uneven)

theorem brendan_yards_per_week : total_yards_week = 71.4 :=
sorry

end brendan_yards_per_week_l54_54269


namespace sequence_all_ones_l54_54526

theorem sequence_all_ones (k : ℕ) (n : ℕ → ℕ) (h_k : 2 ≤ k)
  (h1 : ∀ i, 1 ≤ i → i ≤ k → 1 ≤ n i) 
  (h2 : n 2 ∣ 2^(n 1) - 1) 
  (h3 : n 3 ∣ 2^(n 2) - 1) 
  (h4 : n 4 ∣ 2^(n 3) - 1)
  (h5 : ∀ i, 2 ≤ i → i < k → n (i + 1) ∣ 2^(n i) - 1)
  (h6 : n 1 ∣ 2^(n k) - 1) : 
  ∀ i, 1 ≤ i → i ≤ k → n i = 1 := 
by 
  sorry

end sequence_all_ones_l54_54526


namespace factorize_expression_l54_54852

theorem factorize_expression (a b : ℤ) (h1 : 3 * b + a = -1) (h2 : a * b = -18) : a - b = -11 :=
by
  sorry

end factorize_expression_l54_54852


namespace race_participants_least_number_l54_54208

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l54_54208


namespace math_problem_l54_54005

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l54_54005


namespace cauliflower_difference_is_401_l54_54093

-- Definitions using conditions from part a)
def garden_area_this_year : ℕ := 40401
def side_length_this_year : ℕ := Nat.sqrt garden_area_this_year
def side_length_last_year : ℕ := side_length_this_year - 1
def garden_area_last_year : ℕ := side_length_last_year ^ 2
def cauliflowers_difference : ℕ := garden_area_this_year - garden_area_last_year

-- Problem statement claiming that the difference in cauliflowers produced is 401
theorem cauliflower_difference_is_401 :
  garden_area_this_year = 40401 →
  side_length_this_year = 201 →
  side_length_last_year = 200 →
  garden_area_last_year = 40000 →
  cauliflowers_difference = 401 :=
by
  intros
  sorry

end cauliflower_difference_is_401_l54_54093


namespace Roy_height_l54_54520

theorem Roy_height (Sara_height Joe_height Roy_height : ℕ) 
  (h1 : Sara_height = 45)
  (h2 : Sara_height = Joe_height + 6)
  (h3 : Joe_height = Roy_height + 3) :
  Roy_height = 36 :=
by
  sorry

end Roy_height_l54_54520


namespace solve_system_l54_54370

theorem solve_system :
  ∃ x y : ℝ, 3^x * 2^y = 972 ∧ log (sqrt 3) (x - y) = 2 ∧ x = 5 ∧ y = 2 :=
by { sorry }

end solve_system_l54_54370


namespace problem1_problem2_l54_54484

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := 
  vector_dot v1 v2 = 0

def parallel (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.2 = v1.2 * v2.1

-- Given vectors in the problem
def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, -1)
def n (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
def v : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

-- Problem 1: Find k when n is perpendicular to v
theorem problem1 (k : ℝ) : perpendicular (n k) v → k = 5 / 3 := 
by sorry

-- Problem 2: Find k when n is parallel to c + k * b
theorem problem2 (k : ℝ) : parallel (n k) (c.1 + k * b.1, c.2 + k * b.2) → k = -1 / 3 := 
by sorry

end problem1_problem2_l54_54484


namespace sum_of_B_and_C_in_base_6_l54_54627

def digit_base_6 (n: Nat) : Prop :=
  n > 0 ∧ n < 6

theorem sum_of_B_and_C_in_base_6
  (A B C : Nat)
  (hA : digit_base_6 A)
  (hB : digit_base_6 B)
  (hC : digit_base_6 C)
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : 43 * (A + B + C) = 216 * A) :
  B + C = 5 := by
  sorry

end sum_of_B_and_C_in_base_6_l54_54627


namespace min_table_sum_l54_54640

theorem min_table_sum (A : Fin 5 → Fin 5 → ℕ)
  (distinct_sums : ∀ (i j : Fin 5), i ≠ j → (∑ k : Fin 5, A i k) ≠ (∑ k : Fin 5, A j k) ∧ (∑ k : Fin 5, A k i) ≠ (∑ k : Fin 5, A k j)) :
  ∑ i j, A i j ≥ 48 :=
sorry

end min_table_sum_l54_54640


namespace tan_3theta_l54_54788

-- Let θ be an angle such that tan θ = 3.
variable (θ : ℝ)
noncomputable def tan_theta : ℝ := 3

-- Claim: tan(3 * θ) = 9/13
theorem tan_3theta :
  Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_3theta_l54_54788


namespace arcsin_arccos_add_eq_pi6_l54_54523

noncomputable def arcsin (x : Real) : Real := sorry
noncomputable def arccos (x : Real) : Real := sorry

theorem arcsin_arccos_add_eq_pi6 (x : Real) (hx_range : -1 ≤ x ∧ x ≤ 1)
    (h3x_range : -1 ≤ 3 * x ∧ 3 * x ≤ 1) 
    (h : arcsin x + arccos (3 * x) = Real.pi / 6) :
    x = Real.sqrt (3 / 124) := 
  sorry

end arcsin_arccos_add_eq_pi6_l54_54523


namespace min_neg_signs_to_zero_sum_l54_54664

-- Definition of the set of numbers on the clock face
def clock_face_numbers : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Sum of the clock face numbers
def sum_clock_face_numbers := clock_face_numbers.sum

-- Given condition that the sum of clock face numbers is 78
axiom sum_clock_face_numbers_is_78 : sum_clock_face_numbers = 78

-- Definition of the function to calculate the minimum number of negative signs needed
def min_neg_signs_needed (numbers : List ℤ) (target : ℤ) : ℕ :=
  sorry -- The implementation is omitted

-- Theorem stating the goal of our problem
theorem min_neg_signs_to_zero_sum : min_neg_signs_needed clock_face_numbers 39 = 4 :=
by
  -- Proof is omitted
  sorry

end min_neg_signs_to_zero_sum_l54_54664


namespace problem_statement_l54_54009

-- Define proposition p
def prop_p : Prop := ∃ x : ℝ, Real.exp x ≥ x + 1

-- Define proposition q
def prop_q : Prop := ∀ (a b : ℝ), a^2 < b^2 → a < b

-- The final statement we want to prove
theorem problem_statement : (prop_p ∧ ¬prop_q) :=
by
  sorry

end problem_statement_l54_54009


namespace intersection_A_B_l54_54659

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end intersection_A_B_l54_54659


namespace red_sea_glass_pieces_l54_54581

theorem red_sea_glass_pieces (R : ℕ) 
    (h_bl : ∃ g : ℕ, g = 12) 
    (h_rose_red : ∃ r_b : ℕ, r_b = 9)
    (h_rose_blue : ∃ b : ℕ, b = 11) 
    (h_dorothy_red : 2 * (R + 9) + 3 * 11 = 57) : R = 3 :=
  by
    sorry

end red_sea_glass_pieces_l54_54581


namespace sum_of_m_and_n_l54_54032

theorem sum_of_m_and_n :
  ∃ m n : ℝ, (∀ x : ℝ, (x = 2 → m = 6 / x) ∧ (x = -2 → n = 6 / x)) ∧ (m + n = 0) :=
by
  let m := 6 / 2
  let n := 6 / (-2)
  use m, n
  simp
  sorry -- Proof omitted

end sum_of_m_and_n_l54_54032


namespace total_skateboarded_distance_l54_54816

/-- Define relevant conditions as variables for input values in Lean -/
variables
  (distance_skateboard_to_park : ℕ) -- Distance skateboarded to the park
  (distance_walk_to_park : ℕ) -- Distance walked to the park
  (distance_skateboard_park_to_home : ℕ) -- Distance skateboarded from the park to home

/-- Constraint conditions as hypotheses -/
variables
  (H1 : distance_skateboard_to_park = 10)
  (H2 : distance_walk_to_park = 4)
  (H3 : distance_skateboard_park_to_home = distance_skateboard_to_park + distance_walk_to_park)

/-- The theorem we intend to prove -/
theorem total_skateboarded_distance : 
  distance_skateboard_to_park + distance_skateboard_park_to_home - distance_walk_to_park = 24 := 
by 
  rw [H1, H2, H3] 
  sorry

end total_skateboarded_distance_l54_54816


namespace number_of_girls_on_playground_l54_54072

theorem number_of_girls_on_playground (boys girls total : ℕ) 
  (h1 : boys = 44) (h2 : total = 97) (h3 : total = boys + girls) : 
  girls = 53 :=
by sorry

end number_of_girls_on_playground_l54_54072


namespace opposite_of_2023_l54_54680

-- Define the opposite (additive inverse) function
def additive_inverse (a : ℤ) : ℤ := -a

-- Define the specific problem condition
def condition (n : ℤ) : Prop := 2023 + n = 0

-- Prove that the additive inverse of 2023 satisfies the condition
theorem opposite_of_2023 : condition (additive_inverse 2023) :=
by
  unfold condition additive_inverse
  simp
  exact rfl

end opposite_of_2023_l54_54680


namespace geometric_seq_term_positive_l54_54718

theorem geometric_seq_term_positive :
  ∃ (b : ℝ), 81 * (b / 81) = b ∧ b * (b / 81) = (8 / 27) ∧ b > 0 ∧ b = 2 * Real.sqrt 6 :=
by 
  use 2 * Real.sqrt 6
  sorry

end geometric_seq_term_positive_l54_54718


namespace exam_students_count_l54_54699

theorem exam_students_count (failed_students : ℕ) (failed_percentage : ℝ) (total_students : ℕ) 
    (h1 : failed_students = 260) 
    (h2 : failed_percentage = 0.65) 
    (h3 : (failed_percentage * total_students : ℝ) = (failed_students : ℝ)) : 
    total_students = 400 := 
by 
    sorry

end exam_students_count_l54_54699


namespace tan_triple_angle_l54_54795

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l54_54795


namespace find_roots_of_polynomial_l54_54596

noncomputable def polynomial := Polynomial ℝ

theorem find_roots_of_polynomial :
  (∃ (x : ℝ), x^3 + 3 * x^2 - 6 * x - 8 = 0) ↔ (x = -1 ∨ x = 2 ∨ x = -4) :=
sorry

end find_roots_of_polynomial_l54_54596


namespace equivalent_expression_l54_54136

theorem equivalent_expression (a : ℝ) (h1 : a ≠ -2) (h2 : a ≠ -1) :
  ( (a^2 + a - 2) / (a^2 + 3*a + 2) * 5 * (a + 1)^2 = 5*a^2 - 5 ) :=
by {
  sorry
}

end equivalent_expression_l54_54136


namespace find_a_minus_3b_l54_54851

theorem find_a_minus_3b (a b : ℤ) (h1 : a - 2 * b + 3 = 0) (h2 : -a - b + 3 = 0) : a - 3 * b = -5 :=  
by
  -- Proof steps go here
  sorry

end find_a_minus_3b_l54_54851


namespace total_cost_verification_l54_54873

def sandwich_cost : ℝ := 2.45
def soda_cost : ℝ := 0.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4
def total_cost : ℝ := 8.38

theorem total_cost_verification 
  (sc : sandwich_cost = 2.45)
  (sd : soda_cost = 0.87)
  (ns : num_sandwiches = 2)
  (nd : num_sodas = 4) :
  num_sandwiches * sandwich_cost + num_sodas * soda_cost = total_cost := 
sorry

end total_cost_verification_l54_54873


namespace expression_eq_49_l54_54443

theorem expression_eq_49 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by 
  sorry

end expression_eq_49_l54_54443


namespace intersection_of_complements_l54_54482

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := U \ A

theorem intersection_of_complements :
  U = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
  A = {0, 1, 3, 5, 8} →
  B = {2, 4, 5, 6, 8} →
  (complement U A ∩ complement U B) = {7, 9} :=
by
  intros hU hA hB
  sorry

end intersection_of_complements_l54_54482


namespace boxes_with_neither_l54_54141

-- Definitions based on the conditions given
def total_boxes : Nat := 12
def boxes_with_markers : Nat := 8
def boxes_with_erasers : Nat := 5
def boxes_with_both : Nat := 4

-- The statement we want to prove
theorem boxes_with_neither :
  total_boxes - (boxes_with_markers + boxes_with_erasers - boxes_with_both) = 3 :=
by
  sorry

end boxes_with_neither_l54_54141


namespace quadratic_inequality_solution_set_l54_54301

/- Given a quadratic function with specific roots and coefficients, prove a quadratic inequality. -/
theorem quadratic_inequality_solution_set :
  ∀ (a b : ℝ),
    (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 + a*x + b < 0) →
    a = -3 →
    b = 2 →
    ∀ x : ℝ, (x < 1/2 ∨ x > 1) ↔ (2*x^2 - 3*x + 1 > 0) :=
by
  intros a b h cond_a cond_b x
  sorry

end quadratic_inequality_solution_set_l54_54301


namespace largest_sum_distinct_factors_l54_54026

theorem largest_sum_distinct_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
  (h4 : A * B * C = 2023) : A + B + C = 297 :=
sorry

end largest_sum_distinct_factors_l54_54026


namespace exists_c_gt_zero_l54_54655

theorem exists_c_gt_zero (a b : ℕ) (h_a_square_free : ¬ ∃ (k : ℕ), k^2 ∣ a)
    (h_b_square_free : ¬ ∃ (k : ℕ), k^2 ∣ b) (h_a_b_distinct : a ≠ b) :
    ∃ c > 0, ∀ n : ℕ, n > 0 →
    |(n * Real.sqrt a % 1) - (n * Real.sqrt b % 1)| > c / n^3 := sorry

end exists_c_gt_zero_l54_54655


namespace negate_neg_two_l54_54366

theorem negate_neg_two : -(-2) = 2 := by
  -- The proof goes here
  sorry

end negate_neg_two_l54_54366


namespace cost_of_double_room_l54_54580

theorem cost_of_double_room (total_rooms : ℕ) (cost_single_room : ℕ) (total_revenue : ℕ) 
  (double_rooms_booked : ℕ) (single_rooms_booked := total_rooms - double_rooms_booked) 
  (total_single_revenue := single_rooms_booked * cost_single_room) : 
  total_rooms = 260 → cost_single_room = 35 → total_revenue = 14000 → double_rooms_booked = 196 → 
  196 * 60 + 64 * 35 = total_revenue :=
by
  intros h1 h2 h3 h4
  sorry

end cost_of_double_room_l54_54580


namespace singleBase12Digit_l54_54760

theorem singleBase12Digit (n : ℕ) : 
  (7 ^ 6 ^ 5 ^ 3 ^ 2 ^ 1) % 11 = 4 :=
sorry

end singleBase12Digit_l54_54760


namespace probability_B_given_A_l54_54063

variables {Ω : Type*} {P : MeasureTheory.Measure Ω} [ProbabilityMeasure P]
variables (A B : Set Ω)

theorem probability_B_given_A (hA : P(A) = 0.80) (hB : P(B) = 0.60) :
  (Probability.cond A B) = 0.75 :=
by
  sorry

end probability_B_given_A_l54_54063


namespace two_point_five_one_million_in_scientific_notation_l54_54591

theorem two_point_five_one_million_in_scientific_notation :
  (2.51 * 10^6 : ℝ) = 2.51e6 := 
sorry

end two_point_five_one_million_in_scientific_notation_l54_54591


namespace sum_reciprocal_of_roots_l54_54762

variables {m n : ℝ}

-- Conditions: m and n are real roots of the quadratic equation x^2 + 4x - 1 = 0
def is_root (a : ℝ) : Prop := a^2 + 4 * a - 1 = 0

theorem sum_reciprocal_of_roots (hm : is_root m) (hn : is_root n) : 
  (1 / m) + (1 / n) = 4 :=
by sorry

end sum_reciprocal_of_roots_l54_54762


namespace initial_discount_l54_54352

theorem initial_discount (total_amount price_after_initial_discount additional_disc_percent : ℝ)
  (H1 : total_amount = 1000)
  (H2 : price_after_initial_discount = total_amount - 280)
  (H3 : additional_disc_percent = 0.20) :
  let additional_discount := additional_disc_percent * price_after_initial_discount
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  let total_discount := total_amount - price_after_additional_discount
  let initial_discount := total_discount - additional_discount
  initial_discount = 280 := by
  sorry

end initial_discount_l54_54352


namespace domain_condition_implies_m_range_range_condition_implies_m_range_l54_54616

noncomputable def f (x m : ℝ) : ℝ := Real.log (x^2 - 2 * m * x + m + 2)

def condition1 (m : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 - 2 * m * x + m + 2 > 0)

def condition2 (m : ℝ) : Prop :=
  ∃ y : ℝ, (∀ x : ℝ, y = Real.log (x^2 - 2 * m * x + m + 2))

theorem domain_condition_implies_m_range (m : ℝ) :
  condition1 m → -1 < m ∧ m < 2 :=
sorry

theorem range_condition_implies_m_range (m : ℝ) :
  condition2 m → (m ≤ -1 ∨ m ≥ 2) :=
sorry

end domain_condition_implies_m_range_range_condition_implies_m_range_l54_54616


namespace trigonometric_identity_l54_54469

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α - Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 1 / 13 := 
by
-- The proof goes here
sorry

end trigonometric_identity_l54_54469


namespace problem_1_problem_2_l54_54707

-- Define the factorial and permutation functions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def permutation (n k : ℕ) : ℕ :=
  factorial n / factorial (n - k)

-- Problem 1 statement
theorem problem_1 : permutation 6 6 - permutation 5 5 = 600 := by
  sorry

-- Problem 2 statement
theorem problem_2 : 
  15 * permutation 5 5 * (10^5) + 15 * permutation 4 4 * 11111 =
  15 * permutation 5 5 * (10^5) + 15 * permutation 4 4 * 11111 := by
  sorry

end problem_1_problem_2_l54_54707


namespace pascals_triangle_total_numbers_l54_54319

theorem pascals_triangle_total_numbers (N : ℕ) (hN : N = 29) :
  (∑ n in Finset.range (N + 1), (n + 1)) = 465 :=
by
  rw hN
  calc (∑ n in Finset.range 30, (n + 1))
      = ∑ k in Finset.range 30, (k + 1) : rfl
  -- Here we are calculating the sum of the first 30 terms of the sequence (n + 1)
  ... = 465 : sorry

end pascals_triangle_total_numbers_l54_54319


namespace oranges_to_pears_l54_54650

-- Define the equivalence relation between oranges and pears
def equivalent_weight (orange pear : ℕ) : Prop := 4 * pear = 3 * orange

-- Given:
-- 1. 4 oranges weigh the same as 3 pears
-- 2. Jimmy has 36 oranges
-- Prove that 27 pears are required to balance the weight of 36 oranges
theorem oranges_to_pears (orange pear : ℕ) (h : equivalent_weight 1 1) :
  (4 * pear = 3 * orange) → equivalent_weight 36 27 :=
by
  sorry

end oranges_to_pears_l54_54650


namespace total_ducks_l54_54067

-- Definitions based on the given conditions
def Muscovy : ℕ := 39
def Cayuga : ℕ := Muscovy - 4
def KhakiCampbell : ℕ := (Cayuga - 3) / 2

-- Proof statement
theorem total_ducks : Muscovy + Cayuga + KhakiCampbell = 90 := by
  sorry

end total_ducks_l54_54067


namespace point_on_line_l54_54222

theorem point_on_line (s : ℝ) : 
  (∃ b : ℝ, ∀ x y : ℝ, (y = 3 * x + b) → 
    ((2 = x ∧ y = 8) ∨ (4 = x ∧ y = 14) ∨ (6 = x ∧ y = 20) ∨ (35 = x ∧ y = s))) → s = 107 :=
by
  sorry

end point_on_line_l54_54222


namespace c_value_l54_54327

theorem c_value (c : ℝ) : (∃ a : ℝ, (x : ℝ) → x^2 + 200 * x + c = (x + a)^2) → c = 10000 := 
by
  intro h
  sorry

end c_value_l54_54327


namespace roll_in_second_round_roll_B_two_times_four_rounds_l54_54357

-- Part (1)
theorem roll_in_second_round (total_outcomes : ℕ) (favorable_outcomes : ℕ)
  (prob_sum_multiple_of_3 : ℚ) : 
  (favorable_outcomes = 12) → 
  (total_outcomes = 36) → 
  (prob_sum_multiple_of_3 = 1/3) → 
  (prob_sum_multiple_of_3 = favorable_outcomes / total_outcomes) → 
  prob_sum_multiple_of_3 = 1/3 :=
by
  intros favorable_outcomes_eq total_outcomes_eq prob_eq
  rw [favorable_outcomes_eq, total_outcomes_eq]
  exact prob_eq

-- Part (2)
theorem roll_B_two_times_four_rounds (non_multiple_prob : ℚ) 
  (multiple_prob : ℚ) 
  (comb1 : ℚ) (comb2 : ℚ) (comb3 : ℚ)
  (total_prob : ℚ) : 
  (multiple_prob = 1/3) → 
  (non_multiple_prob = 2/3) → 
  (comb1 = (2/3 * 1/3 * 2/3)) → 
  (comb2 = (2/3)^3) → 
  (comb3 = (1/3 * 2/3 * 1/3)) → 
  (total_prob = comb1 + comb2 + comb3) → 
  total_prob = 14/27 :=
by
  intros multiple_prob_eq non_multiple_prob_eq comb1_eq comb2_eq comb3_eq total_prob_eq
  rw [multiple_prob_eq, non_multiple_prob_eq, comb1_eq, comb2_eq, comb3_eq]
  exact total_prob_eq

end roll_in_second_round_roll_B_two_times_four_rounds_l54_54357


namespace pipe_A_fill_time_l54_54990

theorem pipe_A_fill_time (x : ℝ) (h₁ : x > 0) (h₂ : 1 / x + 1 / 15 = 1 / 6) : x = 10 :=
by
  sorry

end pipe_A_fill_time_l54_54990


namespace angle_A_measure_l54_54276

theorem angle_A_measure (A B C D E : ℝ) 
(h1 : A = 3 * B)
(h2 : A = 4 * C)
(h3 : A = 5 * D)
(h4 : A = 6 * E)
(h5 : A + B + C + D + E = 540) : 
A = 277 :=
by
  sorry

end angle_A_measure_l54_54276


namespace ratio_of_years_taught_l54_54107

-- Definitions based on given conditions
def C : ℕ := 4
def A : ℕ := 2 * C
def total_years (S : ℕ) : Prop := C + A + S = 52

-- Proof statement
theorem ratio_of_years_taught (S : ℕ) (h : total_years S) : 
  S / A = 5 / 1 :=
by
  sorry

end ratio_of_years_taught_l54_54107


namespace jet_bar_sales_difference_l54_54439

variable (monday_sales : ℕ) (total_target : ℕ) (remaining_target : ℕ)
variable (sales_so_far : ℕ) (tuesday_sales : ℕ)
def JetBarsDifference : Prop :=
  monday_sales = 45 ∧ total_target = 90 ∧ remaining_target = 16 ∧
  sales_so_far = total_target - remaining_target ∧
  tuesday_sales = sales_so_far - monday_sales ∧
  (monday_sales - tuesday_sales = 16)

theorem jet_bar_sales_difference :
  JetBarsDifference 45 90 16 (90 - 16) (90 - 16 - 45) :=
by
  sorry

end jet_bar_sales_difference_l54_54439


namespace janet_acres_l54_54647

-- Defining the variables and conditions
variable (horse_production : ℕ) (acre_requirement : ℕ) (spread_rate : ℕ) (num_horses : ℕ) (days : ℕ)

-- Assigning the given values to the variables
def horseFertilizer := 5
def acreFertilizer := 400
def janetSpreadRate := 4
def janetHorses := 80
def fertilizingDays := 25

-- Main theorem stating the question and proving the answer
theorem janet_acres : 
  ∀ (horse_production : ℕ) (acre_requirement : ℕ) (spread_rate : ℕ) (num_horses : ℕ) (days : ℕ),
  horse_production = 5 → 
  acre_requirement = 400 →
  spread_rate = 4 →
  num_horses = 80 →
  days = 25 →
  (spread_rate * days = 100) := 
by
  intros
  -- Proof would be inserted here
  sorry

end janet_acres_l54_54647


namespace student_monthly_earnings_l54_54705

theorem student_monthly_earnings :
  let daily_rate := 1250
  let days_per_week := 4
  let weeks_per_month := 4
  let income_tax_rate := 0.13
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings_before_tax := weekly_earnings * weeks_per_month
  let income_tax_amount := monthly_earnings_before_tax * income_tax_rate
  let monthly_earnings_after_tax := monthly_earnings_before_tax - income_tax_amount
  monthly_earnings_after_tax = 17400 := by
  -- Proof steps here
  sorry

end student_monthly_earnings_l54_54705


namespace probability_selecting_A_l54_54992

theorem probability_selecting_A :
  let total_people := 4
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_people
  probability = 1 / 4 :=
by
  sorry

end probability_selecting_A_l54_54992


namespace squirrel_acorns_left_l54_54436

noncomputable def acorns_per_winter_month (total_acorns : ℕ) (months : ℕ) (acorns_taken_total : ℕ) : ℕ :=
  let per_month := total_acorns / months
  let acorns_taken_per_month := acorns_taken_total / months
  per_month - acorns_taken_per_month

theorem squirrel_acorns_left (total_acorns : ℕ) (months : ℕ) (acorns_taken_total : ℕ) :
  total_acorns = 210 → months = 3 → acorns_taken_total = 30 → acorns_per_winter_month total_acorns months acorns_taken_total = 60 :=
by intros; sorry

end squirrel_acorns_left_l54_54436


namespace difference_of_squares_l54_54409

theorem difference_of_squares :
  535^2 - 465^2 = 70000 :=
by
  sorry

end difference_of_squares_l54_54409


namespace isosceles_triangle_perimeter_l54_54673

theorem isosceles_triangle_perimeter :
  ∀ x y : ℝ, x^2 - 7*x + 10 = 0 → y^2 - 7*y + 10 = 0 → x ≠ y → x + x + y = 12 :=
by
  intros x y hx hy hxy
  -- Place for proof
  sorry

end isosceles_triangle_perimeter_l54_54673


namespace average_shifted_data_is_7_l54_54632

variable (x1 x2 x3 : ℝ)

theorem average_shifted_data_is_7 (h : (x1 + x2 + x3) / 3 = 5) : 
  ((x1 + 2) + (x2 + 2) + (x3 + 2)) / 3 = 7 :=
by
  sorry

end average_shifted_data_is_7_l54_54632


namespace total_stamps_received_l54_54521

theorem total_stamps_received
  (initial_stamps : ℕ)
  (final_stamps : ℕ)
  (received_stamps : ℕ)
  (h_initial : initial_stamps = 34)
  (h_final : final_stamps = 61)
  (h_received : received_stamps = final_stamps - initial_stamps) :
  received_stamps = 27 :=
by 
  sorry

end total_stamps_received_l54_54521


namespace pencils_multiple_of_10_l54_54220

theorem pencils_multiple_of_10 (pens : ℕ) (students : ℕ) (pencils : ℕ) 
  (h_pens : pens = 1230) 
  (h_students : students = 10) 
  (h_max_distribute : ∀ s, s ≤ students → (∃ pens_per_student, pens = pens_per_student * s ∧ ∃ pencils_per_student, pencils = pencils_per_student * s)) :
  ∃ n, pencils = 10 * n :=
by
  sorry

end pencils_multiple_of_10_l54_54220


namespace tan3theta_l54_54797

theorem tan3theta (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 := 
by
  sorry

end tan3theta_l54_54797


namespace total_students_surveyed_l54_54092

variable (F E S FE FS ES FES N T : ℕ)

def only_one_language := 230
def exactly_two_languages := 190
def all_three_languages := 40
def no_language := 60

-- Summing up all categories
def total_students := only_one_language + exactly_two_languages + all_three_languages + no_language

theorem total_students_surveyed (h1 : F + E + S = only_one_language) 
    (h2 : FE + FS + ES = exactly_two_languages) 
    (h3 : FES = all_three_languages) 
    (h4 : N = no_language) 
    (h5 : T = F + E + S + FE + FS + ES + FES + N) : 
    T = total_students :=
by
  rw [total_students, only_one_language, exactly_two_languages, all_three_languages, no_language]
  sorry

end total_students_surveyed_l54_54092


namespace simple_interest_years_l54_54094

theorem simple_interest_years (P R : ℝ) (T : ℝ) :
  P = 2500 → (2500 * (R + 2) / 100 * T = 2500 * R / 100 * T + 250) → T = 5 :=
by
  intro hP h
  -- Note: Actual proof details would go here
  sorry

end simple_interest_years_l54_54094


namespace middle_letter_value_l54_54498

theorem middle_letter_value 
  (final_score : ℕ) 
  (first_letter_value : ℕ) 
  (third_letter_value : ℕ)
  (word_length : ℕ)
  (triple_score : ℕ)
  (total_points : ℕ)
  (middle_letter_value : ℕ)
  (h1 : final_score = 30)
  (h2 : first_letter_value = 1)
  (h3 : third_letter_value = 1)
  (h4 : word_length = 3)
  (h5 : triple_score = 3)
  (h6 : total_points = final_score / triple_score)
  (h7 : total_points = 10)
  (h8 : middle_letter_value = total_points - first_letter_value - third_letter_value) :
  middle_letter_value = 8 := 
by sorry

end middle_letter_value_l54_54498


namespace smallest_n_l54_54556

theorem smallest_n (n : ℕ) : 
  (∃ k : ℕ, 4 * n = k^2) ∧ (∃ l : ℕ, 5 * n = l^5) ↔ n = 625 :=
by sorry

end smallest_n_l54_54556


namespace lattice_points_count_l54_54570

-- A definition of lattice points and bounded region
def is_lattice_point (p : ℤ × ℤ) : Prop := true

def in_region (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  (y = abs x ∨ y = -x^2 + 4*x + 6) ∧ (y ≤ abs x ∧ y ≤ -x^2 + 4*x + 6)

-- The target statement to prove
theorem lattice_points_count : ∃ n, n = 23 ∧ ∀ p : ℤ × ℤ, is_lattice_point p → in_region p := sorry

end lattice_points_count_l54_54570


namespace solution_set_ineq_l54_54761

open Set

theorem solution_set_ineq (a x : ℝ) (h : 0 < a ∧ a < 1) : 
 (a < x ∧ x < 1/a) ↔ ((x - a) * (x - 1/a) > 0) :=
by
  sorry

end solution_set_ineq_l54_54761


namespace find_t_l54_54857

theorem find_t:
  (∃ t, (∀ (x y: ℝ), (x = 2 ∧ y = 8) ∨ (x = 4 ∧ y = 14) ∨ (x = 6 ∧ y = 20) → 
                (∀ (m b: ℝ), y = m * x + b) ∧ 
                (∀ (m b: ℝ), y = 3 * x + b ∧ b = 2 ∧ (t = 3 * 50 + 2) ∧ t = 152))) := by
  sorry

end find_t_l54_54857


namespace probability_odd_product_lt_one_eighth_l54_54227

theorem probability_odd_product_lt_one_eighth :
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  p < 1 / 8 :=
by
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  sorry

end probability_odd_product_lt_one_eighth_l54_54227


namespace donut_selection_count_l54_54832

-- Define the variables and their sum condition
def donutSelections : ℕ := ∑ (g c p : ℕ) in {g // g + c + p = 5}, 1

-- Main theorem statement: The count of selections is 21
theorem donut_selection_count : donutSelections = 21 :=
by
  sorry

end donut_selection_count_l54_54832


namespace janet_earnings_per_hour_l54_54037

theorem janet_earnings_per_hour :
  let P := 0.25
  let T := 10
  3600 / T * P = 90 :=
by
  let P := 0.25
  let T := 10
  sorry

end janet_earnings_per_hour_l54_54037


namespace factor_expression_l54_54450

theorem factor_expression (x : ℝ) :
  (12 * x^3 + 45 * x - 3) - (-3 * x^3 + 5 * x - 2) = 5 * x * (3 * x^2 + 8) - 1 :=
by
  sorry

end factor_expression_l54_54450


namespace find_B_l54_54975

theorem find_B (A B C : ℝ) (h : ∀ (x : ℝ), x ≠ 7 ∧ x ≠ -1 → 
    2 / ((x-7)*(x+1)^2) = A / (x-7) + B / (x+1) + C / (x+1)^2) : 
  B = 1 / 16 :=
sorry

end find_B_l54_54975


namespace integer_values_b_for_three_integer_solutions_l54_54070

theorem integer_values_b_for_three_integer_solutions (b : ℤ) :
  ¬ ∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1^2 + b * x1 + 5 ≤ 0) ∧
                     (x2^2 + b * x2 + 5 ≤ 0) ∧ (x3^2 + b * x3 + 5 ≤ 0) ∧
                     (∀ x : ℤ, x1 < x ∧ x < x3 → x^2 + b * x + 5 > 0) :=
by
  sorry

end integer_values_b_for_three_integer_solutions_l54_54070


namespace probability_one_excellence_A_probability_one_excellence_B_range_n_for_A_l54_54974

def probability_of_excellence_A : ℚ := 2/5
def probability_of_excellence_B1 : ℚ := 1/4
def probability_of_excellence_B2 : ℚ := 2/5
def probability_of_excellence_B3 (n : ℚ) : ℚ := n

def one_excellence_A : ℚ := 3 * (2/5) * (3/5)^2
def one_excellence_B (n : ℚ) : ℚ := 
    (probability_of_excellence_B1 * (3/5) * (1 - n)) + 
    ((1 - probability_of_excellence_B1) * (2/5) * (1 - n)) + 
    ((1 - probability_of_excellence_B1) * (3/5) * n)

theorem probability_one_excellence_A : one_excellence_A = 54/125 := sorry

theorem probability_one_excellence_B (n : ℚ) (hn : n = 1/3) : one_excellence_B n = 9/20 := sorry

def expected_excellence_A : ℚ := 3 * (2/5)
def expected_excellence_B (n : ℚ) : ℚ := (13/20) + n

theorem range_n_for_A (n : ℚ) (hn1 : 0 < n) (hn2 : n < 11/20): 
    expected_excellence_A > expected_excellence_B n := sorry

end probability_one_excellence_A_probability_one_excellence_B_range_n_for_A_l54_54974


namespace fuel_tank_capacity_l54_54579

theorem fuel_tank_capacity
  (ethanol_A_ethanol : ∀ {x : Float}, x = 0.12 * 49.99999999999999)
  (ethanol_B_ethanol : ∀ {C : Float}, x = 0.16 * (C - 49.99999999999999))
  (total_ethanol : ∀ {C : Float}, 0.12 * 49.99999999999999 + 0.16 * (C - 49.99999999999999) = 30) :
  (C = 162.5) :=
sorry

end fuel_tank_capacity_l54_54579


namespace inscribed_circle_radius_l54_54841

/-- Define a square SEAN with side length 2. -/
def square_side_length : ℝ := 2

/-- Define a quarter-circle of radius 1. -/
def quarter_circle_radius : ℝ := 1

/-- Hypothesis: The radius of the largest circle that can be inscribed in the remaining figure. -/
theorem inscribed_circle_radius :
  let S : ℝ := square_side_length
  let R : ℝ := quarter_circle_radius
  ∃ (r : ℝ), (r = 5 - 3 * Real.sqrt 2) := 
sorry

end inscribed_circle_radius_l54_54841


namespace total_amount_collected_l54_54830

theorem total_amount_collected 
  (num_members : ℕ)
  (annual_fee : ℕ)
  (cost_hardcover : ℕ)
  (num_hardcovers : ℕ)
  (cost_paperback : ℕ)
  (num_paperbacks : ℕ)
  (total_collected : ℕ) :
  num_members = 6 →
  annual_fee = 150 →
  cost_hardcover = 30 →
  num_hardcovers = 6 →
  cost_paperback = 12 →
  num_paperbacks = 6 →
  total_collected = (annual_fee + cost_hardcover * num_hardcovers + cost_paperback * num_paperbacks) * num_members →
  total_collected = 2412 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end total_amount_collected_l54_54830


namespace wax_initial_amount_l54_54308

def needed : ℕ := 17
def total : ℕ := 574
def initial : ℕ := total - needed

theorem wax_initial_amount :
  initial = 557 :=
by
  sorry

end wax_initial_amount_l54_54308


namespace average_of_three_l54_54672

theorem average_of_three (y : ℝ) (h : (15 + 24 + y) / 3 = 20) : y = 21 :=
by
  sorry

end average_of_three_l54_54672


namespace total_memory_space_l54_54818

def morning_songs : Nat := 10
def afternoon_songs : Nat := 15
def night_songs : Nat := 3
def song_size : Nat := 5

theorem total_memory_space : (morning_songs + afternoon_songs + night_songs) * song_size = 140 := by
  sorry

end total_memory_space_l54_54818


namespace brownies_count_l54_54342

theorem brownies_count (pan_length : ℕ) (pan_width : ℕ) (piece_side : ℕ) 
  (h1 : pan_length = 24) (h2 : pan_width = 15) (h3 : piece_side = 3) : 
  (pan_length * pan_width) / (piece_side * piece_side) = 40 :=
by {
  sorry
}

end brownies_count_l54_54342


namespace solve_inequality_solve_system_of_inequalities_l54_54564

-- Inequality proof problem
theorem solve_inequality (x : ℝ) (h : (2*x - 3)/3 > (3*x + 1)/6 - 1) : x > 1 := by
  sorry

-- System of inequalities proof problem
theorem solve_system_of_inequalities (x : ℝ) (h1 : x ≤ 3*x - 6) (h2 : 3*x + 1 > 2*(x - 1)) : x ≥ 3 := by
  sorry

end solve_inequality_solve_system_of_inequalities_l54_54564


namespace a_x1_x2_x13_eq_zero_l54_54218

theorem a_x1_x2_x13_eq_zero {a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ}
  (h1: a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) *
             (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2: a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) *
             (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := by
  sorry

end a_x1_x2_x13_eq_zero_l54_54218


namespace tan_theta_correct_l54_54471

noncomputable def cos_double_angle (θ : ℝ) : ℝ := 2 * Real.cos θ ^ 2 - 1

theorem tan_theta_correct (θ : ℝ) (hθ₁ : θ > 0) (hθ₂ : θ < Real.pi / 2) 
  (h : 15 * cos_double_angle θ - 14 * Real.cos θ + 11 = 0) : Real.tan θ = Real.sqrt 5 / 2 :=
sorry

end tan_theta_correct_l54_54471


namespace factorization_correct_l54_54289

-- Define noncomputable to deal with the natural arithmetic operations
noncomputable def a : ℕ := 66
noncomputable def b : ℕ := 231

-- Define the given expressions
noncomputable def lhs (x : ℕ) : ℤ := ((a : ℤ) * x^6) - ((b : ℤ) * x^12)
noncomputable def rhs (x : ℕ) : ℤ := (33 : ℤ) * x^6 * (2 - 7 * x^6)

-- The theorem to prove the equality
theorem factorization_correct (x : ℕ) : lhs x = rhs x :=
by sorry

end factorization_correct_l54_54289


namespace solution_exists_l54_54519

theorem solution_exists (a b c : ℝ) : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 := 
sorry

end solution_exists_l54_54519


namespace factor_polynomial_l54_54290

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end factor_polynomial_l54_54290


namespace divides_polynomial_l54_54041

theorem divides_polynomial (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  ∀ x : ℂ, (x^2 + x + 1) ∣ (x^(3 * m + 1) + x^(3 * n + 2) + 1) :=
by
  sorry

end divides_polynomial_l54_54041


namespace replace_batteries_in_December_16_years_later_l54_54930

theorem replace_batteries_in_December_16_years_later :
  ∀ (n : ℕ), n = 30 → ∃ (years : ℕ) (months : ℕ), years = 16 ∧ months = 11 :=
by
  sorry

end replace_batteries_in_December_16_years_later_l54_54930


namespace reflect_ellipse_l54_54745

theorem reflect_ellipse :
  let A : ℝ × ℝ → ℝ := λ p, ((p.1 + 2)^2 / 9) + ((p.2 + 3)^2 / 4)
  let B := (x : ℝ) → (y : ℝ) → ((x - 3)^2 / 9) + ((y - 2)^2 / 4) = 1
  (∀ x y, B (−y) (−x) = 1 ↔ A (x, y) = 1) :=
by
  sorry

end reflect_ellipse_l54_54745


namespace roots_quadratic_eq_k_l54_54589

theorem roots_quadratic_eq_k (k : ℝ) :
  (∀ x : ℝ, (5 * x^2 + 20 * x + k = 0) ↔ (x = (-20 + Real.sqrt 60) / 10 ∨ x = (-20 - Real.sqrt 60) / 10)) →
  k = 17 := by
  intro h
  sorry

end roots_quadratic_eq_k_l54_54589


namespace find_values_of_a_and_b_l54_54246

theorem find_values_of_a_and_b (a b x y : ℝ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < x) (h4: 0 < y) 
  (h5 : a + b = 10) (h6 : a / x + b / y = 1) (h7 : x + y = 18) : 
  (a = 2 ∧ b = 8) ∨ (a = 8 ∧ b = 2) := 
sorry

end find_values_of_a_and_b_l54_54246


namespace power_evaluation_l54_54625

theorem power_evaluation (x : ℕ) (h1 : 3^x = 81) : 3^(x+2) = 729 := by
  sorry

end power_evaluation_l54_54625


namespace employed_males_percentage_l54_54493

theorem employed_males_percentage (P : ℕ) (H1: P > 0)
    (employed_pct : ℝ) (female_pct : ℝ)
    (H_employed_pct : employed_pct = 0.64)
    (H_female_pct : female_pct = 0.140625) :
    (0.859375 * employed_pct * 100) = 54.96 :=
by
  sorry

end employed_males_percentage_l54_54493


namespace solve_system_l54_54371

noncomputable def log_base_sqrt_3 (z : ℝ) : ℝ := Real.log z / Real.log (Real.sqrt 3)

theorem solve_system :
  ∃ x y : ℝ, (3^x * 2^y = 972) ∧ (log_base_sqrt_3 (x - y) = 2) ∧ (x = 5 ∧ y = 2) :=
by
  sorry

end solve_system_l54_54371


namespace slope_reciprocal_and_a_bounds_l54_54481

theorem slope_reciprocal_and_a_bounds (x : ℝ) (f g : ℝ → ℝ) 
    (h1 : ∀ x, f x = Real.log x - a * (x - 1)) 
    (h2 : ∀ x, g x = Real.exp x) :
    ((∀ k₁ k₂, (∃ x₁, k₁ = deriv f x₁) ∧ (∃ x₂, k₂ = deriv g x₂) ∧ k₁ * k₂ = 1) 
    ↔ (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp 2 - 1) / Real.exp 1 ∨ a = 0) :=
by
  sorry

end slope_reciprocal_and_a_bounds_l54_54481


namespace triangle_circumscribed_circle_diameter_l54_54934

noncomputable def circumscribed_circle_diameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem triangle_circumscribed_circle_diameter :
  let a := 16
  let A := Real.pi / 4   -- 45 degrees in radians
  circumscribed_circle_diameter a A = 16 * Real.sqrt 2 :=
by
  sorry

end triangle_circumscribed_circle_diameter_l54_54934


namespace Stuart_initial_marbles_l54_54099

variable (Betty_marbles Stuart_final increased_by: ℤ) 

-- Conditions as definitions
def Betty_has : Betty_marbles = 60 := sorry 
def Stuart_collect_increase : Stuart_final = 80 := sorry 
def percentage_given : ∃ x, x = (40 * Betty_marbles) / 100 := sorry 

-- Theorem to prove Stuart had 56 marbles initially
theorem Stuart_initial_marbles 
  (h1 : Betty_has)
  (h2 : Stuart_collect_increase)
  (h3 : percentage_given) :
  ∃ y, y = Stuart_final - 24 := 
sorry

end Stuart_initial_marbles_l54_54099


namespace profit_inequality_solution_l54_54090

theorem profit_inequality_solution (x : ℝ) (h₁ : 1 ≤ x) (h₂ : x ≤ 10) :
  100 * 2 * (5 * x + 1 - 3 / x) ≥ 3000 ↔ 3 ≤ x ∧ x ≤ 10 :=
by
  sorry

end profit_inequality_solution_l54_54090


namespace find_a_l54_54477

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x ^ 2 + a * Real.cos (Real.pi * x) else 2

theorem find_a (a : ℝ) :
  (∀ x, f (-x) a = -f x a) → f 1 a = 2 → a = - 3 :=
by
  sorry

end find_a_l54_54477


namespace c_plus_d_l54_54825

theorem c_plus_d (c d : ℝ)
  (h1 : c^3 - 12 * c^2 + 15 * c - 36 = 0)
  (h2 : 6 * d^3 - 36 * d^2 - 150 * d + 1350 = 0) :
  c + d = 7 := 
  sorry

end c_plus_d_l54_54825


namespace winning_jackpot_is_event_l54_54695

-- Definitions based on the conditions
def has_conditions (experiment : String) : Prop :=
  experiment = "A" ∨ experiment = "B" ∨ experiment = "C" ∨ experiment = "D"

def has_outcomes (experiment : String) : Prop :=
  experiment = "D"

def is_event (experiment : String) : Prop :=
  has_conditions experiment ∧ has_outcomes experiment

-- Statement to prove
theorem winning_jackpot_is_event : is_event "D" :=
by
  -- Trivial step to show that D meets both conditions and outcomes
  exact sorry

end winning_jackpot_is_event_l54_54695


namespace triangular_number_30_l54_54442

theorem triangular_number_30 : 
  (∃ (T : ℕ), T = 30 * (30 + 1) / 2 ∧ T = 465) :=
by 
  sorry

end triangular_number_30_l54_54442


namespace solve_for_y_l54_54972

theorem solve_for_y (y : ℝ) (h : (5 - 1 / y)^(1/3) = -3) : y = 1 / 32 :=
by
  sorry

end solve_for_y_l54_54972


namespace bleach_contains_chlorine_l54_54911

noncomputable def element_in_bleach (mass_percentage : ℝ) (substance : String) : String :=
  if mass_percentage = 31.08 ∧ substance = "sodium hypochlorite" then "Chlorine"
  else "unknown"

theorem bleach_contains_chlorine : element_in_bleach 31.08 "sodium hypochlorite" = "Chlorine" :=
by
  sorry

end bleach_contains_chlorine_l54_54911


namespace coeff_x6_in_expansion_l54_54281

noncomputable def binom_coeff : ℕ → ℕ → ℕ
| n, k => Nat.choose n k

def term_coeff (n k : ℕ) : ℤ :=
  (-2)^k * binom_coeff n k

theorem coeff_x6_in_expansion : term_coeff 8 2 = 112 := by
  sorry

end coeff_x6_in_expansion_l54_54281


namespace find_number_l54_54064

-- Given conditions
variables (x y : ℕ)

-- The conditions from the problem statement
def digit_sum : Prop := x + y = 12
def reverse_condition : Prop := (10 * x + y) + 36 = 10 * y + x

-- The final statement
theorem find_number (h1 : digit_sum x y) (h2 : reverse_condition x y) : 10 * x + y = 48 :=
sorry

end find_number_l54_54064


namespace odd_function_property_l54_54767

noncomputable def odd_function := {f : ℝ → ℝ // ∀ x : ℝ, f (-x) = -f x}

theorem odd_function_property (f : odd_function) (h1 : f.1 1 = -2) : f.1 (-1) + f.1 0 = 2 := by
  sorry

end odd_function_property_l54_54767


namespace find_a_l54_54610

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1)
  (h_diff : |a^2 - a| = 6) : a = 3 :=
sorry

end find_a_l54_54610


namespace number_of_combinations_for_Xiao_Wang_l54_54810

-- Definitions for the subjects
inductive Subject
| Physics
| Chemistry
| Biology
| Politics
| History
| Geography

-- List of all available subjects
def all_subjects : List Subject := [
  Subject.Physics,
  Subject.Chemistry,
  Subject.Biology,
  Subject.Politics,
  Subject.History,
  Subject.Geography
]

-- Definitions for science and humanities subjects
def science_subjects : List Subject := [
  Subject.Physics,
  Subject.Chemistry,
  Subject.Biology
]

def humanities_subjects : List Subject := [
  Subject.Politics,
  Subject.History,
  Subject.Geography
]

-- Theorem statement
theorem number_of_combinations_for_Xiao_Wang : ∃ n: ℕ, n = 10 ∧ (
  let combinations := (Finset.powersetLen 2 (Finset.ofList science_subjects)).card * 
                      (Finset.powersetLen 1 (Finset.ofList humanities_subjects)).card +
                      (Finset.powersetLen 3 (Finset.ofList science_subjects)).card
  in combinations = 10) :=
by
  sorry

end number_of_combinations_for_Xiao_Wang_l54_54810


namespace required_sixth_quiz_score_l54_54108

theorem required_sixth_quiz_score :
  let scores := [91, 94, 88, 90, 101] in
  let sum_scores := scores.sum in
  let num_quizzes := 6 in
  let desired_mean := 95 in
  let required_total := desired_mean * num_quizzes in
  let sixth_score := required_total - sum_scores in
  sixth_score = 106 :=
by
  sorry

end required_sixth_quiz_score_l54_54108


namespace part1_part2_l54_54923

-- Definitions
def A (x : ℝ) : Prop := (x + 2) / (x - 3 / 2) < 0
def B (x : ℝ) (m : ℝ) : Prop := x^2 - (m + 1) * x + m ≤ 0

-- Part (1): when m = 2, find A ∪ B
theorem part1 :
  (∀ x, A x ∨ B x 2) ↔ ∀ x, -2 < x ∧ x ≤ 2 := sorry

-- Part (2): find the range of real number m
theorem part2 :
  (∀ x, A x → B x m) ↔ (-2 < m ∧ m < 3 / 2) := sorry

end part1_part2_l54_54923


namespace tan3theta_l54_54799

theorem tan3theta (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 := 
by
  sorry

end tan3theta_l54_54799


namespace minimum_value_l54_54466

open Real

theorem minimum_value (x : ℝ) (hx : x > 2) : 
  ∃ y ≥ 4 * Real.sqrt 2, ∀ z, (z = (x + 6) / (Real.sqrt (x - 2)) → y ≤ z) := 
sorry

end minimum_value_l54_54466


namespace total_lives_l54_54886

theorem total_lives :
  ∀ (num_friends num_new_players lives_per_friend lives_per_new_player : ℕ),
  num_friends = 2 →
  lives_per_friend = 6 →
  num_new_players = 2 →
  lives_per_new_player = 6 →
  (num_friends * lives_per_friend + num_new_players * lives_per_new_player) = 24 :=
by
  intros num_friends num_new_players lives_per_friend lives_per_new_player
  intro h1 h2 h3 h4
  sorry

end total_lives_l54_54886


namespace total_cash_realized_correct_l54_54576

structure Stock where
  value : ℝ
  return_rate : ℝ
  brokerage_fee_rate : ℝ

def stockA : Stock := { value := 10000, return_rate := 0.14, brokerage_fee_rate := 0.0025 }
def stockB : Stock := { value := 20000, return_rate := 0.10, brokerage_fee_rate := 0.005 }
def stockC : Stock := { value := 30000, return_rate := 0.07, brokerage_fee_rate := 0.0075 }

def cash_realized (s : Stock) : ℝ :=
  let total_with_return := s.value * (1 + s.return_rate)
  total_with_return - (total_with_return * s.brokerage_fee_rate)

noncomputable def total_cash_realized : ℝ :=
  cash_realized stockA + cash_realized stockB + cash_realized stockC

theorem total_cash_realized_correct :
  total_cash_realized = 65120.75 :=
    sorry

end total_cash_realized_correct_l54_54576


namespace polynomial_minimal_degree_l54_54598

noncomputable def minimal_polynomial : Polynomial ℚ :=
  Polynomial.X^4 - 4*Polynomial.X^3 + 4*Polynomial.X^2 + 16*Polynomial.X - 8

theorem polynomial_minimal_degree
  (x : ℚ)
  (root1 : x = 1 + real.sqrt 2 + real.sqrt 3)
  (root2 : x = 1 + real.sqrt 2 - real.sqrt 3)
  (root3 : x = 1 - real.sqrt 2 + real.sqrt 3)
  (root4 : x = 1 - real.sqrt 2 - real.sqrt 3)
  : (minimal_polynomial.eval x = 0) :=
by
  sorry

end polynomial_minimal_degree_l54_54598


namespace inequality_solution_minimum_value_l54_54605

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem inequality_solution :
  {x : ℝ | f x > 7} = {x | x > 4 ∨ x < -3} :=
by
  sorry

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : ∀ x, f x ≥ m + n) :
  m + n = 3 →
  (m^2 + n^2 ≥ 9 / 2 ∧ (m = 3 / 2 ∧ n = 3 / 2)) :=
by
  sorry

end inequality_solution_minimum_value_l54_54605


namespace intersection_A_B_l54_54658

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end intersection_A_B_l54_54658


namespace original_three_numbers_are_arith_geo_seq_l54_54550

theorem original_three_numbers_are_arith_geo_seq
  (x y z : ℕ) (h1 : ∃ k : ℕ, x = 3*k ∧ y = 4*k ∧ z = 5*k)
  (h2 : ∃ r : ℝ, (x + 1) / y = r ∧ y / z = r ∧ r^2 = z / y):
  x = 15 ∧ y = 20 ∧ z = 25 :=
by 
  sorry

end original_three_numbers_are_arith_geo_seq_l54_54550


namespace championship_winner_is_902_l54_54388

namespace BasketballMatch

inductive Class : Type
| c901
| c902
| c903
| c904

open Class

def A_said (champ third : Class) : Prop :=
  champ = c902 ∧ third = c904

def B_said (fourth runner_up : Class) : Prop :=
  fourth = c901 ∧ runner_up = c903

def C_said (third champ : Class) : Prop :=
  third = c903 ∧ champ = c904

def half_correct (P Q : Prop) : Prop := 
  (P ∧ ¬Q) ∨ (¬P ∧ Q)

theorem championship_winner_is_902 (A_third B_fourth B_runner_up C_third : Class) 
  (H_A : half_correct (A_said c902 A_third) (A_said A_third c902))
  (H_B : half_correct (B_said B_fourth B_runner_up) (B_said B_runner_up B_fourth))
  (H_C : half_correct (C_said C_third c904) (C_said c904 C_third)) :
  ∃ winner, winner = c902 :=
sorry

end BasketballMatch

end championship_winner_is_902_l54_54388


namespace line_through_longest_chord_l54_54473

-- Define the point M and the circle equation
def M : ℝ × ℝ := (3, -1)
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + y - 2 = 0

-- Define the standard form of the circle equation
def standard_circle_eqn (x y : ℝ) : Prop := (x - 2)^2 + (y + 1/2)^2 = 25/4

-- Define the line equation
def line_eqn (x y : ℝ) : Prop := x + 2 * y - 2 = 0

-- Theorem: Equation of the line containing the longest chord passing through M
theorem line_through_longest_chord : 
  (circle_eqn 3 (-1)) → 
  ∀ (x y : ℝ), standard_circle_eqn x y → ∃ (k b : ℝ), line_eqn x y :=
by
  -- Proof goes here
  intro h1 x y h2
  sorry

end line_through_longest_chord_l54_54473


namespace total_stickers_l54_54838

def stickers_in_first_box : ℕ := 23
def stickers_in_second_box : ℕ := stickers_in_first_box + 12

theorem total_stickers :
  stickers_in_first_box + stickers_in_second_box = 58 := 
by
  sorry

end total_stickers_l54_54838


namespace slower_pipe_fills_tank_in_200_minutes_l54_54419

noncomputable def slower_pipe_filling_time (F S : ℝ) (h1 : F = 4 * S) (h2 : F + S = 1 / 40) : ℝ :=
  1 / S

theorem slower_pipe_fills_tank_in_200_minutes (F S : ℝ) (h1 : F = 4 * S) (h2 : F + S = 1 / 40) :
  slower_pipe_filling_time F S h1 h2 = 200 :=
sorry

end slower_pipe_fills_tank_in_200_minutes_l54_54419


namespace min_lines_to_separate_points_l54_54068

theorem min_lines_to_separate_points (m n : ℕ) (h_m : m = 8) (h_n : n = 8) : 
  (m - 1) + (n - 1) = 14 := by
  sorry

end min_lines_to_separate_points_l54_54068


namespace cos_five_pi_over_three_l54_54245

theorem cos_five_pi_over_three : Real.cos (5 * Real.pi / 3) = 1 / 2 := 
by 
  sorry

end cos_five_pi_over_three_l54_54245


namespace diff_of_squares_example_l54_54413

theorem diff_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end diff_of_squares_example_l54_54413


namespace main_theorem_l54_54447

noncomputable def main_expr := (Real.pi - 2019) ^ 0 + |Real.sqrt 3 - 1| + (-1 / 2)⁻¹ - 2 * Real.tan (Real.pi / 6)

theorem main_theorem : main_expr = -2 + Real.sqrt 3 / 3 := by
  sorry

end main_theorem_l54_54447


namespace minimum_participants_l54_54194

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l54_54194


namespace jesse_initial_blocks_l54_54500

def total_blocks_initial (blocks_cityscape blocks_farmhouse blocks_zoo blocks_first_area blocks_second_area blocks_third_area blocks_left : ℕ) : ℕ :=
  blocks_cityscape + blocks_farmhouse + blocks_zoo + blocks_first_area + blocks_second_area + blocks_third_area + blocks_left

theorem jesse_initial_blocks :
  total_blocks_initial 80 123 95 57 43 62 84 = 544 :=
sorry

end jesse_initial_blocks_l54_54500


namespace winter_spending_l54_54853

-- Define the total spending by the end of November
def total_spending_end_november : ℝ := 3.3

-- Define the total spending by the end of February
def total_spending_end_february : ℝ := 7.0

-- Formalize the problem: prove that the spending during December, January, and February is 3.7 million dollars
theorem winter_spending : total_spending_end_february - total_spending_end_november = 3.7 := by
  sorry

end winter_spending_l54_54853


namespace no_solution_for_xx_plus_yy_eq_9z_l54_54971

theorem no_solution_for_xx_plus_yy_eq_9z (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ¬ (x^x + y^y = 9^z) :=
sorry

end no_solution_for_xx_plus_yy_eq_9z_l54_54971


namespace range_of_f_l54_54138

noncomputable def f (x : ℝ) := Real.log (2 - x^2) / Real.log (1 / 2)

theorem range_of_f : Set.range f = Set.Icc (-1 : ℝ) 0 := by
  sorry

end range_of_f_l54_54138


namespace power_function_value_at_3_l54_54299

theorem power_function_value_at_3
  (f : ℝ → ℝ)
  (h1 : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α)
  (h2 : f 2 = 1 / 4) :
  f 3 = 1 / 9 := 
sorry

end power_function_value_at_3_l54_54299


namespace tan_add_pi_over_4_l54_54638

theorem tan_add_pi_over_4 (α : ℝ)
  (O : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM_coords : M = (-1, real.sqrt 3))
  (hO_origin : O = (0, 0))
  (hM_on_terminal_side : ∃ α, M.1 = -1 ∧ M.2 = real.sqrt 3 ∧ M = (cos α, sin α)) :
  real.tan (α + real.pi / 4) = real.sqrt 3 - 2 :=
by
  sorry

end tan_add_pi_over_4_l54_54638


namespace savings_correct_l54_54855

-- Define the conditions
def in_store_price : ℝ := 320
def discount_rate : ℝ := 0.05
def monthly_payment : ℝ := 62
def monthly_payments : ℕ := 5
def shipping_handling : ℝ := 10

-- Prove that the savings from buying in-store is 16 dollars.
theorem savings_correct : 
  (monthly_payments * monthly_payment + shipping_handling) - (in_store_price * (1 - discount_rate)) = 16 := 
by
  sorry

end savings_correct_l54_54855


namespace roots_of_poly_l54_54763

theorem roots_of_poly (a b c : ℂ) :
  ∀ x, x = a ∨ x = b ∨ x = c → x^4 - a*x^3 - b*x + c = 0 :=
sorry

end roots_of_poly_l54_54763


namespace probability_unit_square_not_touch_central_2x2_square_l54_54356

-- Given a 6x6 checkerboard with a marked 2x2 square at the center,
-- prove that the probability of choosing a unit square that does not touch
-- the marked 2x2 square is 2/3.

theorem probability_unit_square_not_touch_central_2x2_square : 
    let total_squares := 36
    let touching_squares := 12
    let squares_not_touching := total_squares - touching_squares
    (squares_not_touching : ℚ) / (total_squares : ℚ) = 2 / 3 := by
  sorry

end probability_unit_square_not_touch_central_2x2_square_l54_54356


namespace train_length_l54_54438

/-- 
Given that a train can cross an electric pole in 200 seconds and its speed is 18 km/h,
prove that the length of the train is 1000 meters.
-/
theorem train_length
  (time_to_cross : ℕ)
  (speed_kmph : ℕ)
  (h_time : time_to_cross = 200)
  (h_speed : speed_kmph = 18)
  : (speed_kmph * 1000 / 3600 * time_to_cross = 1000) :=
by
  sorry

end train_length_l54_54438


namespace scientific_notation_of_600000_l54_54355

theorem scientific_notation_of_600000 : 600000 = 6 * 10^5 :=
by
  sorry

end scientific_notation_of_600000_l54_54355


namespace distance_greater_than_school_l54_54850

-- Let d1, d2, and d3 be the distances given as the conditions
def distance_orchard_to_house : ℕ := 800
def distance_house_to_pharmacy : ℕ := 1300
def distance_pharmacy_to_school : ℕ := 1700

-- The total distance from orchard to pharmacy via the house
def total_distance_orchard_to_pharmacy : ℕ :=
  distance_orchard_to_house + distance_house_to_pharmacy

-- The difference between the total distance from orchard to pharmacy and the distance from pharmacy to school
def distance_difference : ℕ :=
  total_distance_orchard_to_pharmacy - distance_pharmacy_to_school

-- The theorem to prove
theorem distance_greater_than_school :
  distance_difference = 400 := sorry

end distance_greater_than_school_l54_54850


namespace stock_price_rise_l54_54297

theorem stock_price_rise {P : ℝ} (h1 : P > 0)
    (h2007 : P * 1.20 = 1.20 * P)
    (h2008 : 1.20 * P * 0.75 = P * 0.90)
    (hCertainYear : P * 1.17 = P * 0.90 * (1 + 30 / 100)) :
  30 = 30 :=
by sorry

end stock_price_rise_l54_54297


namespace vasya_mushrooms_l54_54077

-- Lean definition of the problem based on the given conditions
theorem vasya_mushrooms :
  ∃ (N : ℕ), 
    N ≥ 100 ∧ N < 1000 ∧
    (∃ (a b c : ℕ), a ≠ 0 ∧ N = 100 * a + 10 * b + c ∧ a + b + c = 14) ∧
    N % 50 = 0 ∧ 
    N = 950 :=
by
  sorry

end vasya_mushrooms_l54_54077


namespace race_minimum_participants_l54_54188

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l54_54188


namespace arithmetic_to_geometric_find_original_numbers_l54_54548

theorem arithmetic_to_geometric (k : ℕ) 
  (h1 : ∃ k, (3 * k + 1) * 4 * k = 5 * k * (3 * k + 1))
  : k = 5 :=
begin
  sorry,
end

theorem find_original_numbers :
  ∃ (a b c : ℕ), a = 15 ∧ b = 20 ∧ c = 25 :=
begin
  use [15, 20, 25],
  exact ⟨rfl, rfl, rfl⟩
end

end arithmetic_to_geometric_find_original_numbers_l54_54548


namespace least_value_of_d_l54_54843

theorem least_value_of_d (c d : ℕ) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (hc_factors : (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a ≠ b ∧ c = a * b) ∨ (∃ p : ℕ, p > 1 ∧ c = p^3))
  (hd_factors : ∃ factors : ℕ, factors = c ∧ ∃ divisors : Finset ℕ, divisors.card = factors ∧ ∀ k ∈ divisors, d % k = 0)
  (div_cd : d % c = 0) : d = 18 :=
sorry

end least_value_of_d_l54_54843


namespace f_monotonically_decreasing_in_interval_l54_54303

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

theorem f_monotonically_decreasing_in_interval :
  ∀ x y : ℝ, -2 < x ∧ x < 1 → -2 < y ∧ y < 1 → (y > x → f y < f x) :=
by
  sorry

end f_monotonically_decreasing_in_interval_l54_54303


namespace diff_of_squares_535_465_l54_54407

theorem diff_of_squares_535_465 : (535^2 - 465^2) = 70000 :=
sorry

end diff_of_squares_535_465_l54_54407


namespace race_minimum_participants_l54_54189

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l54_54189


namespace planting_area_correct_l54_54717

def garden_area : ℕ := 18 * 14
def pond_area : ℕ := 4 * 2
def flower_bed_area : ℕ := (1 / 2) * 3 * 2
def planting_area : ℕ := garden_area - pond_area - flower_bed_area

theorem planting_area_correct : planting_area = 241 := by
  -- proof would go here
  sorry

end planting_area_correct_l54_54717


namespace black_to_brown_ratio_l54_54543

-- Definitions of the given conditions
def total_shoes : ℕ := 66
def brown_shoes : ℕ := 22
def black_shoes : ℕ := total_shoes - brown_shoes

-- Lean 4 problem statement: Prove the ratio of black shoes to brown shoes is 2:1
theorem black_to_brown_ratio :
  (black_shoes / Nat.gcd black_shoes brown_shoes) = 2 ∧ (brown_shoes / Nat.gcd black_shoes brown_shoes) = 1 := by
sorry

end black_to_brown_ratio_l54_54543


namespace distinct_real_numbers_condition_l54_54955

theorem distinct_real_numbers_condition (a b c : ℝ) (h_abc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a / (b - c)) + (b / (c - a)) + (c / (a - b)) = 1) :
  (a / (b - c)^2) + (b / (c - a)^2) + (c / (a - b)^2) = 1 := 
by sorry

end distinct_real_numbers_condition_l54_54955


namespace gcd_72_120_168_l54_54983

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  -- Each step would be proven individually here.
  sorry

end gcd_72_120_168_l54_54983


namespace largest_odd_digit_multiple_of_11_l54_54404

theorem largest_odd_digit_multiple_of_11 (n : ℕ) (h1 : n < 10000) (h2 : ∀ d ∈ (n.digits 10), d % 2 = 1) (h3 : 11 ∣ n) : n ≤ 9559 :=
sorry

end largest_odd_digit_multiple_of_11_l54_54404


namespace sunflower_height_A_l54_54963

-- Define the height of sunflowers from Packet B
def height_B : ℝ := 160

-- Define that Packet A sunflowers are 20% taller than Packet B sunflowers
def height_A : ℝ := 1.2 * height_B

-- State the theorem to show that height_A equals 192 inches
theorem sunflower_height_A : height_A = 192 := by
  sorry

end sunflower_height_A_l54_54963


namespace student_net_monthly_earnings_l54_54703

theorem student_net_monthly_earnings : 
  (∀ (days_per_week : ℕ) (rate_per_day : ℕ) (weeks_per_month : ℕ) (tax_rate : ℚ), 
      days_per_week = 4 → 
      rate_per_day = 1250 → 
      weeks_per_month = 4 → 
      tax_rate = 0.13 →  
      (days_per_week * rate_per_day * weeks_per_month * (1 - tax_rate)).toInt) = 17400 := 
by {
  sorry
}

end student_net_monthly_earnings_l54_54703


namespace multiple_of_bees_l54_54964

theorem multiple_of_bees (b₁ b₂ : ℕ) (h₁ : b₁ = 144) (h₂ : b₂ = 432) : b₂ / b₁ = 3 := 
by
  sorry

end multiple_of_bees_l54_54964


namespace cost_per_meter_l54_54382

-- Definitions of the conditions
def length_of_plot : ℕ := 63
def breadth_of_plot : ℕ := length_of_plot - 26
def perimeter_of_plot := 2 * length_of_plot + 2 * breadth_of_plot
def total_cost : ℕ := 5300

-- Statement to prove
theorem cost_per_meter : (total_cost : ℚ) / perimeter_of_plot = 26.5 :=
by sorry

end cost_per_meter_l54_54382


namespace tangency_condition_intersection_condition_l54_54758

-- Definitions of the circle and line for the given conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 8 * y + 12 = 0
def line_eq (a x y : ℝ) : Prop := a * x + y + 2 * a = 0

/-- Theorem for the tangency condition -/
theorem tangency_condition (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y ↔ (x^2 + (y + 4)^2 = 4)) →
  (|(-4 + 2 * a)| / Real.sqrt (a^2 + 1) = 2) →
  a = 3 / 4 :=
by
  sorry

/-- Theorem for the intersection condition -/
theorem intersection_condition (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y ↔ (x^2 + (y + 4)^2 = 4)) →
  (|(-4 + 2 * a)| / Real.sqrt (a^2 + 1) = Real.sqrt 2) →
  (a = 1 ∨ a = 7) →
  (∀ x y : ℝ,
    (line_eq 1 x y ∧ line_eq 7 x y ↔ 
    (7 * x + y + 14 = 0 ∨ x + y + 2 = 0))) :=
by
  sorry

end tangency_condition_intersection_condition_l54_54758


namespace distinct_placements_of_two_pieces_l54_54023

-- Definitions of the conditions
def grid_size : ℕ := 3
def cell_count : ℕ := grid_size * grid_size
def pieces_count : ℕ := 2

-- The theorem statement
theorem distinct_placements_of_two_pieces : 
  (number_of_distinct_placements : ℕ) = 10 := by
  -- Proof goes here with calculations and accounting for symmetry
  sorry

end distinct_placements_of_two_pieces_l54_54023


namespace fraction_of_single_female_students_l54_54698

variables (total_students : ℕ) (male_students married_students married_male_students female_students single_female_students : ℕ)

-- Given conditions
def condition1 : male_students = (7 * total_students) / 10 := sorry
def condition2 : married_students = (3 * total_students) / 10 := sorry
def condition3 : married_male_students = male_students / 7 := sorry

-- Derived conditions
def condition4 : female_students = total_students - male_students := sorry
def condition5 : married_female_students = married_students - married_male_students := sorry
def condition6 : single_female_students = female_students - married_female_students := sorry

-- The proof goal
theorem fraction_of_single_female_students 
  (h1 : male_students = (7 * total_students) / 10)
  (h2 : married_students = (3 * total_students) / 10)
  (h3 : married_male_students = male_students / 7)
  (h4 : female_students = total_students - male_students)
  (h5 : married_female_students = married_students - married_male_students)
  (h6 : single_female_students = female_students - married_female_students) :
  (single_female_students : ℚ) / (female_students : ℚ) = 1 / 3 :=
sorry

end fraction_of_single_female_students_l54_54698


namespace factorization_correct_l54_54536

theorem factorization_correct :
  ∀ (m a b x y : ℝ), 
    (m^2 - 4 = (m + 2) * (m - 2)) ∧
    ((a + 3) * (a - 3) = a^2 - 9) ∧
    (a^2 - b^2 + 1 = (a + b) * (a - b) + 1) ∧
    (6 * x^2 * y^3 = 2 * x^2 * 3 * y^3) →
    (m^2 - 4 = (m + 2) * (m - 2)) :=
by
  intros m a b x y h
  have ⟨hA, hB, hC, hD⟩ := h
  exact hA

end factorization_correct_l54_54536


namespace greatest_value_a_plus_b_l54_54871

theorem greatest_value_a_plus_b (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a * b = 45) : a + b = 2 * Real.sqrt 55 :=
by
  sorry

end greatest_value_a_plus_b_l54_54871


namespace correct_statements_l54_54620

theorem correct_statements (f : ℝ → ℝ) (t : ℝ)
  (h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0)
  (h2 : (∀ x : ℝ, f x = f (-x)) ∧ (∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 > f x2) ∧ f (-2) = 0)
  (h3 : ∀ x : ℝ, f (-x) = -f x)
  (h4 : ∀ x : ℝ, f (x - t) = f (x + t)) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 > f x2 ↔ x1 < x2) ∧
  (∀ x : ℝ, f x - f (|x|) = - (f (-x) - f (|x|))) :=
by
  sorry

end correct_statements_l54_54620


namespace minimum_letters_for_grid_coloring_l54_54071

theorem minimum_letters_for_grid_coloring : 
  ∀ (grid_paper : Type) 
  (is_node : grid_paper → Prop) 
  (marked : grid_paper → Prop)
  (mark_with_letter : grid_paper → ℕ) 
  (connected : grid_paper → grid_paper → Prop), 
  (∀ n₁ n₂ : grid_paper, is_node n₁ → is_node n₂ → mark_with_letter n₁ = mark_with_letter n₂ → 
  (n₁ ≠ n₂ → ∃ n₃ : grid_paper, is_node n₃ ∧ connected n₁ n₃ ∧ connected n₃ n₂ ∧ mark_with_letter n₃ ≠ mark_with_letter n₁)) → 
  ∃ (k : ℕ), k = 2 :=
by
  sorry

end minimum_letters_for_grid_coloring_l54_54071


namespace reflection_correct_l54_54848

def point := (ℝ × ℝ)

def reflect_x_axis (p : point) : point :=
  (p.1, -p.2)

def M : point := (3, 2)

theorem reflection_correct : reflect_x_axis M = (3, -2) :=
  sorry

end reflection_correct_l54_54848


namespace largest_last_digit_in_string_l54_54537

theorem largest_last_digit_in_string :
  ∃ (s : Nat → Fin 10), 
    (s 0 = 1) ∧ 
    (∀ k, k < 99 → (∃ m, (s k * 10 + s (k + 1)) = 17 * m ∨ (s k * 10 + s (k + 1)) = 23 * m)) ∧
    (∃ l, l < 10 ∧ (s 99 = l)) ∧
    (forall last, (last < 10 ∧ (s 99 = last))) ∧
    (∀ m n, s 99 = m → s 99 = n → m ≤ n → n = 9) :=
sorry

end largest_last_digit_in_string_l54_54537


namespace quadratic_sum_r_s_l54_54973

/-- Solve the quadratic equation and identify the sum of r and s 
from the equivalent completed square form (x + r)^2 = s. -/
theorem quadratic_sum_r_s (r s : ℤ) :
  (∃ r s : ℤ, (x - r)^2 = s → r + s = 11) :=
sorry

end quadratic_sum_r_s_l54_54973


namespace stuart_initial_marbles_l54_54098

theorem stuart_initial_marbles (B S : ℝ) (h1 : B = 60) (h2 : 0.40 * B = 24) (h3 : S + 24 = 80) : S = 56 :=
by
  sorry

end stuart_initial_marbles_l54_54098


namespace speed_first_hour_l54_54683

theorem speed_first_hour (x : ℝ) :
  (∃ x, (x + 45) / 2 = 65) → x = 85 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  sorry

end speed_first_hour_l54_54683


namespace prob_top_three_cards_all_hearts_l54_54895

-- Define the total numbers of cards and suits
def total_cards := 52
def hearts_count := 13

-- Define the probability calculation as per the problem statement
def prob_top_three_hearts : ℚ :=
  (13 * 12 * 11 : ℚ) / (52 * 51 * 50 : ℚ)

-- The theorem states that the probability of the top three cards being all hearts is 11/850
theorem prob_top_three_cards_all_hearts : prob_top_three_hearts = 11 / 850 := by
  -- The proof details are not required, just stating the structure
  sorry

end prob_top_three_cards_all_hearts_l54_54895


namespace number_of_representatives_from_companyA_l54_54586

-- Define conditions
def companyA_representatives : ℕ := 120
def companyB_representatives : ℕ := 100
def total_selected : ℕ := 11

-- Define the theorem
theorem number_of_representatives_from_companyA : 120 * (11 / (120 + 100)) = 6 := by
  sorry

end number_of_representatives_from_companyA_l54_54586


namespace angle_between_clock_hands_at_7_25_l54_54554

theorem angle_between_clock_hands_at_7_25 : 
  let degrees_per_hour := 30
  let minute_hand_position := (25 / 60 * 360 : ℝ)
  let hour_hand_position := (7 * degrees_per_hour + (25 / 60 * degrees_per_hour) : ℝ)
  abs (hour_hand_position - minute_hand_position) = 72.5 
  := by
  let degrees_per_hour := 30
  let minute_hand_position := (25 / 60 * 360 : ℝ)
  let hour_hand_position := (7 * degrees_per_hour + (25 / 60 * degrees_per_hour) : ℝ)
  sorry

end angle_between_clock_hands_at_7_25_l54_54554


namespace students_transferred_l54_54149

theorem students_transferred (students_before : ℕ) (total_students : ℕ) (students_equal : ℕ) 
  (h1 : students_before = 23) (h2 : total_students = 50) (h3 : students_equal = total_students / 2) : 
  (∃ x : ℕ, students_equal = students_before + x) → (∃ x : ℕ, x = 2) :=
by
  -- h1: students_before = 23
  -- h2: total_students = 50
  -- h3: students_equal = total_students / 2
  -- to prove: ∃ x : ℕ, students_equal = students_before + x → ∃ x : ℕ, x = 2
  sorry

end students_transferred_l54_54149


namespace product_of_two_large_integers_l54_54210

theorem product_of_two_large_integers :
  ∃ a b : ℕ, a > 2009^182 ∧ b > 2009^182 ∧ 3^2008 + 4^2009 = a * b :=
by { sorry }

end product_of_two_large_integers_l54_54210


namespace three_pow_y_plus_two_l54_54626

theorem three_pow_y_plus_two (y : ℕ) (h : 3^y = 81) : 3^(y+2) = 729 := sorry

end three_pow_y_plus_two_l54_54626


namespace world_book_day_l54_54266

theorem world_book_day
  (x y : ℕ)
  (h1 : x + y = 22)
  (h2 : x = 2 * y + 1) :
  x = 15 ∧ y = 7 :=
by {
  -- The proof is omitted as per the instructions
  sorry
}

end world_book_day_l54_54266


namespace negation_statement_l54_54056

variables {a b c : ℝ}

theorem negation_statement (h : a * b * c = 0) : ¬(a = 0 ∨ b = 0 ∨ c = 0) :=
sorry

end negation_statement_l54_54056


namespace polygon_sides_from_angle_sum_l54_54392

-- Let's define the problem
theorem polygon_sides_from_angle_sum : 
  ∀ (n : ℕ), (n - 2) * 180 = 900 → n = 7 :=
by
  intros n h
  sorry

end polygon_sides_from_angle_sum_l54_54392


namespace hypotenuse_length_l54_54435

theorem hypotenuse_length {a b c : ℕ} (ha : a = 8) (hb : b = 15) (hc : c = (8^2 + 15^2).sqrt) : c = 17 :=
by
  sorry

end hypotenuse_length_l54_54435


namespace ceil_neg_sqrt_eq_neg_two_l54_54115

noncomputable def x : ℝ := -Real.sqrt (64 / 9)

theorem ceil_neg_sqrt_eq_neg_two : Real.ceil x = -2 := by
  exact sorry

end ceil_neg_sqrt_eq_neg_two_l54_54115


namespace train_probability_at_station_l54_54730

-- Define time intervals
def t0 := 0 -- Train arrival start time in minutes after 1:00 PM
def t1 := 60 -- Train arrival end time in minutes after 1:00 PM
def a0 := 0 -- Alex arrival start time in minutes after 1:00 PM
def a1 := 120 -- Alex arrival end time in minutes after 1:00 PM

-- Define the probability calculation problem
theorem train_probability_at_station :
  let total_area := (t1 - t0) * (a1 - a0)
  let overlap_area := (1/2 * 50 * 50) + (10 * 55)
  (overlap_area / total_area) = 1/4 := 
by
  sorry

end train_probability_at_station_l54_54730


namespace oil_bill_for_January_l54_54541

variables (J F : ℝ)

-- Conditions
def condition1 := F = (5 / 4) * J
def condition2 := (F + 45) / J = 3 / 2

theorem oil_bill_for_January (h1 : condition1 J F) (h2 : condition2 J F) : J = 180 :=
by sorry

end oil_bill_for_January_l54_54541


namespace minimum_participants_l54_54202

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l54_54202


namespace find_x_plus_y_l54_54013

theorem find_x_plus_y (x y : ℚ) (h1 : 5 * x - 3 * y = 27) (h2 : 3 * x + 5 * y = 1) : x + y = 31 / 17 :=
by
  sorry

end find_x_plus_y_l54_54013


namespace smallest_y_divisible_l54_54998

theorem smallest_y_divisible (y : ℕ) : 
  (y % 3 = 2) ∧ (y % 5 = 4) ∧ (y % 7 = 6) → y = 104 :=
by
  sorry

end smallest_y_divisible_l54_54998


namespace second_train_start_time_l54_54729

theorem second_train_start_time :
  let start_time_first_train := 14 -- 2:00 pm in 24-hour format
  let catch_up_time := 22          -- 10:00 pm in 24-hour format
  let speed_first_train := 70      -- km/h
  let speed_second_train := 80     -- km/h
  let travel_time_first_train := catch_up_time - start_time_first_train
  let distance_first_train := speed_first_train * travel_time_first_train
  let t := distance_first_train / speed_second_train
  let start_time_second_train := catch_up_time - t
  start_time_second_train = 15 := -- 3:00 pm in 24-hour format
by
  sorry

end second_train_start_time_l54_54729


namespace vertical_distance_rotated_square_l54_54468

-- Lean 4 statement for the mathematically equivalent proof problem
theorem vertical_distance_rotated_square
  (side_length : ℝ)
  (n : ℕ)
  (rot_angle : ℝ)
  (orig_line_height before_rotation : ℝ)
  (diagonal_length : ℝ)
  (lowered_distance : ℝ)
  (highest_point_drop : ℝ)
  : side_length = 2 →
    n = 4 →
    rot_angle = 45 →
    orig_line_height = 1 →
    diagonal_length = side_length * (2:ℝ)^(1/2) →
    lowered_distance = (diagonal_length / 2) - orig_line_height →
    highest_point_drop = lowered_distance →
    2 = 2 :=
    sorry

end vertical_distance_rotated_square_l54_54468


namespace Elizabeth_More_Revenue_Than_Banks_l54_54165

theorem Elizabeth_More_Revenue_Than_Banks : 
  let banks_investments := 8
  let banks_revenue_per_investment := 500
  let elizabeth_investments := 5
  let elizabeth_revenue_per_investment := 900
  let banks_total_revenue := banks_investments * banks_revenue_per_investment
  let elizabeth_total_revenue := elizabeth_investments * elizabeth_revenue_per_investment
  elizabeth_total_revenue - banks_total_revenue = 500 :=
by
  sorry

end Elizabeth_More_Revenue_Than_Banks_l54_54165


namespace remainder_1234_mul_5678_mod_1000_l54_54555

theorem remainder_1234_mul_5678_mod_1000 :
  (1234 * 5678) % 1000 = 652 := by
  sorry

end remainder_1234_mul_5678_mod_1000_l54_54555


namespace largest_class_students_l54_54418

theorem largest_class_students (x : ℕ)
  (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 95) :
  x = 23 :=
by
  sorry

end largest_class_students_l54_54418


namespace tan_theta_3_l54_54783

noncomputable def tan_triple_angle (θ : ℝ) : ℝ := (3 * (Real.tan θ) - ((Real.tan θ) ^ 3)) / (1 - 3 * (Real.tan θ)^2)

theorem tan_theta_3 (θ : ℝ) (h : Real.tan θ = 3) : tan_triple_angle θ = 9 / 13 :=
by
  sorry

end tan_theta_3_l54_54783


namespace scientific_notation_101_49_billion_l54_54285

-- Define the term "one hundred and one point four nine billion"
def billion (n : ℝ) := n * 10^9

-- Axiomatization of the specific number in question
def hundredOnePointFourNineBillion := billion 101.49

-- Theorem stating that the scientific notation for 101.49 billion is 1.0149 × 10^10
theorem scientific_notation_101_49_billion : hundredOnePointFourNineBillion = 1.0149 * 10^10 :=
by
  sorry

end scientific_notation_101_49_billion_l54_54285


namespace compute_expression_l54_54900

theorem compute_expression :
    ( (2 / 3) * Real.sqrt 15 - Real.sqrt 20 ) / ( (1 / 3) * Real.sqrt 5 ) = 2 * Real.sqrt 3 - 6 :=
by
  sorry

end compute_expression_l54_54900


namespace actual_diameter_of_tissue_is_0_03_mm_l54_54690

-- Defining necessary conditions
def magnified_diameter_meters : ℝ := 0.15
def magnification_factor : ℝ := 5000
def meters_to_millimeters : ℝ := 1000

-- Prove that the actual diameter of the tissue is 0.03 millimeters
theorem actual_diameter_of_tissue_is_0_03_mm :
  (magnified_diameter_meters * meters_to_millimeters) / magnification_factor = 0.03 := 
  sorry

end actual_diameter_of_tissue_is_0_03_mm_l54_54690


namespace jerry_sister_increase_temp_l54_54341

theorem jerry_sister_increase_temp :
  let T0 := 40
  let T1 := 2 * T0
  let T2 := T1 - 30
  let T3 := T2 - 0.3 * T2
  let T4 := 59
  T4 - T3 = 24 := by
  sorry

end jerry_sister_increase_temp_l54_54341


namespace oranges_equivalency_l54_54649

theorem oranges_equivalency :
  ∀ (w_orange w_apple w_pear : ℕ), 
  (9 * w_orange = 6 * w_apple + w_pear) →
  (36 * w_orange = 24 * w_apple + 4 * w_pear) :=
by
  -- The proof will go here; for now, we'll use sorry to skip it
  sorry

end oranges_equivalency_l54_54649


namespace minimizing_reciprocal_sum_l54_54030

theorem minimizing_reciprocal_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a + 4 * b = 30) :
  a = 10 ∧ b = 5 :=
by
  sorry

end minimizing_reciprocal_sum_l54_54030


namespace prime_sum_value_l54_54483

theorem prime_sum_value (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 2019) : 
  (p - 1)^(q - 1) = 1 ∨ (p - 1)^(q - 1) = 2016 :=
by
  sorry

end prime_sum_value_l54_54483


namespace donuts_per_box_l54_54340

-- Define the conditions and the theorem
theorem donuts_per_box :
  (10 * 12 - 12 - 8) / 10 = 10 :=
by
  sorry

end donuts_per_box_l54_54340


namespace probability_one_male_correct_probability_atleast_one_female_correct_l54_54362

def total_students := 5
def female_students := 2
def male_students := 3
def number_of_selections := 2

noncomputable def probability_only_one_male : ℚ :=
  (6 : ℚ) / 10

noncomputable def probability_atleast_one_female : ℚ :=
  (7 : ℚ) / 10

theorem probability_one_male_correct :
  (6 / 10 : ℚ) = 3 / 5 :=
by
  sorry

theorem probability_atleast_one_female_correct :
  (7 / 10 : ℚ) = 7 / 10 :=
by
  sorry

end probability_one_male_correct_probability_atleast_one_female_correct_l54_54362


namespace cat_mouse_position_258_l54_54025

-- Define the cycle positions for the cat
def cat_position (n : ℕ) : String :=
  match n % 4 with
  | 0 => "top left"
  | 1 => "top right"
  | 2 => "bottom right"
  | _ => "bottom left"

-- Define the cycle positions for the mouse
def mouse_position (n : ℕ) : String :=
  match n % 8 with
  | 0 => "top middle"
  | 1 => "top right"
  | 2 => "right middle"
  | 3 => "bottom right"
  | 4 => "bottom middle"
  | 5 => "bottom left"
  | 6 => "left middle"
  | _ => "top left"

theorem cat_mouse_position_258 : 
  cat_position 258 = "top right" ∧ mouse_position 258 = "top right" := by
  sorry

end cat_mouse_position_258_l54_54025


namespace total_paintable_area_correct_l54_54035

-- Bedroom dimensions and unoccupied wall space
def bedroom1_length : ℕ := 14
def bedroom1_width : ℕ := 12
def bedroom1_height : ℕ := 9
def bedroom1_unoccupied : ℕ := 70

def bedroom2_length : ℕ := 12
def bedroom2_width : ℕ := 11
def bedroom2_height : ℕ := 9
def bedroom2_unoccupied : ℕ := 65

def bedroom3_length : ℕ := 13
def bedroom3_width : ℕ := 12
def bedroom3_height : ℕ := 9
def bedroom3_unoccupied : ℕ := 68

-- Total paintable area calculation
def calculate_paintable_area (length width height unoccupied : ℕ) : ℕ :=
  2 * (length * height + width * height) - unoccupied

-- Total paintable area of all bedrooms
def total_paintable_area : ℕ :=
  calculate_paintable_area bedroom1_length bedroom1_width bedroom1_height bedroom1_unoccupied +
  calculate_paintable_area bedroom2_length bedroom2_width bedroom2_height bedroom2_unoccupied +
  calculate_paintable_area bedroom3_length bedroom3_width bedroom3_height bedroom3_unoccupied

theorem total_paintable_area_correct : 
  total_paintable_area = 1129 :=
by
  unfold total_paintable_area
  unfold calculate_paintable_area
  norm_num
  sorry

end total_paintable_area_correct_l54_54035


namespace gabriel_pages_correct_l54_54106

-- Given conditions
def beatrix_pages : ℕ := 704

def cristobal_pages (b : ℕ) : ℕ := 3 * b + 15

def gabriel_pages (c b : ℕ) : ℕ := 3 * (c + b)

-- Problem statement
theorem gabriel_pages_correct : gabriel_pages (cristobal_pages beatrix_pages) beatrix_pages = 8493 :=
by 
  sorry

end gabriel_pages_correct_l54_54106


namespace total_shoes_l54_54890

variables (people : ℕ) (shoes_per_person : ℕ)

-- There are 10 people
axiom h1 : people = 10
-- Each person has 2 shoes
axiom h2 : shoes_per_person = 2

-- The total number of shoes kept outside the library is 10 * 2 = 20
theorem total_shoes (people shoes_per_person : ℕ) (h1 : people = 10) (h2 : shoes_per_person = 2) : people * shoes_per_person = 20 :=
by sorry

end total_shoes_l54_54890


namespace area_Q1RQ3Q5_of_regular_hexagon_l54_54737

noncomputable def area_quadrilateral (s : ℝ) (θ : ℝ) : ℝ := s^2 * Real.sin θ / 2

theorem area_Q1RQ3Q5_of_regular_hexagon :
  let apothem := 3
  let side_length := 6 * Real.sqrt 3
  let θ := Real.pi / 3  -- 60 degrees in radians
  area_quadrilateral (3 * Real.sqrt 3) θ = 27 * Real.sqrt 3 / 2 :=
by
  sorry

end area_Q1RQ3Q5_of_regular_hexagon_l54_54737


namespace fred_paid_amount_l54_54167

def ticket_price : ℝ := 5.92
def number_of_tickets : ℕ := 2
def borrowed_movie_price : ℝ := 6.79
def change_received : ℝ := 1.37

def total_cost : ℝ := (number_of_tickets : ℝ) * ticket_price + borrowed_movie_price
def amount_paid : ℝ := total_cost + change_received

theorem fred_paid_amount : amount_paid = 20.00 := sorry

end fred_paid_amount_l54_54167


namespace work_efficiency_l54_54172

theorem work_efficiency (orig_time : ℝ) (new_time : ℝ) (work : ℝ) 
  (h1 : orig_time = 1)
  (h2 : new_time = orig_time * (1 - 0.20))
  (h3 : work = 1) :
  (orig_time / new_time) * 100 = 125 :=
by
  sorry

end work_efficiency_l54_54172


namespace simplify_negative_exponents_l54_54557

theorem simplify_negative_exponents (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + y)⁻¹ * (x⁻¹ + y⁻¹) = x⁻¹ * y⁻¹ :=
  sorry

end simplify_negative_exponents_l54_54557


namespace land_tax_calculation_l54_54104

theorem land_tax_calculation
  (area : ℝ)
  (value_per_acre : ℝ)
  (tax_rate : ℝ)
  (total_cadastral_value : ℝ := area * value_per_acre)
  (land_tax : ℝ := total_cadastral_value * tax_rate) :
  area = 15 → value_per_acre = 100000 → tax_rate = 0.003 → land_tax = 4500 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end land_tax_calculation_l54_54104


namespace diff_of_squares_example_l54_54414

theorem diff_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end diff_of_squares_example_l54_54414


namespace completing_square_solution_l54_54694

theorem completing_square_solution (x : ℝ) : x^2 - 4 * x - 22 = 0 ↔ (x - 2)^2 = 26 := sorry

end completing_square_solution_l54_54694


namespace total_green_peaches_l54_54545

-- Define the known conditions
def baskets : ℕ := 7
def green_peaches_per_basket : ℕ := 2

-- State the problem and the proof goal
theorem total_green_peaches : baskets * green_peaches_per_basket = 14 := by
  -- Provide a proof here
  sorry

end total_green_peaches_l54_54545


namespace smallest_d_for_divisibility_by_3_l54_54742

def sum_of_digits (d : ℕ) : ℕ := 5 + 4 + 7 + d + 0 + 6

theorem smallest_d_for_divisibility_by_3 (d : ℕ) :
  (sum_of_digits 2) % 3 = 0 ∧ ∀ k, k < 2 → sum_of_digits k % 3 ≠ 0 := 
sorry

end smallest_d_for_divisibility_by_3_l54_54742


namespace Yoongi_stack_taller_than_Taehyung_l54_54899

theorem Yoongi_stack_taller_than_Taehyung :
  let height_A := 3
  let height_B := 3.5
  let count_A := 16
  let count_B := 14
  let total_height_A := height_A * count_A
  let total_height_B := height_B * count_B
  total_height_B > total_height_A ∧ (total_height_B - total_height_A = 1) :=
by
  sorry

end Yoongi_stack_taller_than_Taehyung_l54_54899


namespace jellybeans_in_jar_now_l54_54225

def initial_jellybeans : ℕ := 90
def samantha_takes : ℕ := 24
def shelby_takes : ℕ := 12
def scarlett_takes : ℕ := 2 * shelby_takes
def scarlett_returns : ℕ := scarlett_takes / 2
def shannon_refills : ℕ := (samantha_takes + shelby_takes) / 2

theorem jellybeans_in_jar_now : 
  initial_jellybeans 
  - samantha_takes 
  - shelby_takes 
  + scarlett_returns
  + shannon_refills 
  = 84 := by
  sorry

end jellybeans_in_jar_now_l54_54225


namespace max_m_value_l54_54479

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem max_m_value 
  (t : ℝ) 
  (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) <= x) : m ≤ 4 :=
sorry

end max_m_value_l54_54479


namespace no_valid_placement_for_digits_on_45gon_l54_54812

theorem no_valid_placement_for_digits_on_45gon (f : Fin 45 → Fin 10) :
  ¬ ∀ (a b : Fin 10), a ≠ b → ∃ (i j : Fin 45), i ≠ j ∧ f i = a ∧ f j = b :=
by {
  sorry
}

end no_valid_placement_for_digits_on_45gon_l54_54812


namespace problem_statement_l54_54215

theorem problem_statement (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ) 
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) : 
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := 
sorry

end problem_statement_l54_54215


namespace cone_base_circumference_l54_54249

theorem cone_base_circumference (r : ℝ) (sector_angle : ℝ) (total_angle : ℝ) (C : ℝ) (h1 : r = 6) (h2 : sector_angle = 180) (h3 : total_angle = 360) (h4 : C = 2 * r * Real.pi) :
  (sector_angle / total_angle) * C = 6 * Real.pi :=
by
  -- Skipping proof
  sorry

end cone_base_circumference_l54_54249


namespace problem_i_problem_ii_l54_54470

noncomputable def f (m x : ℝ) := (Real.log x / Real.log m) ^ 2 + 2 * (Real.log x / Real.log m) - 3

theorem problem_i (x : ℝ) : f 2 x < 0 ↔ (1 / 8) < x ∧ x < 2 :=
by sorry

theorem problem_ii (m : ℝ) (H : ∀ x, 2 ≤ x ∧ x ≤ 4 → f m x < 0) : 
  (0 < m ∧ m < 4^(1/3)) ∨ (4 < m) :=
by sorry

end problem_i_problem_ii_l54_54470


namespace tan3theta_l54_54798

theorem tan3theta (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 := 
by
  sorry

end tan3theta_l54_54798


namespace julie_aaron_age_l54_54160

variables {J A m : ℕ}

theorem julie_aaron_age : (J = 4 * A) → (J + 10 = m * (A + 10)) → (m = 4) :=
by
  intros h1 h2
  sorry

end julie_aaron_age_l54_54160


namespace directrix_of_parabola_l54_54464

theorem directrix_of_parabola (y x : ℝ) (p : ℝ) (h₁ : y = 8 * x ^ 2) (h₂ : y = 4 * p * x) : 
  p = 2 ∧ (y = -p ↔ y = -2) :=
by
  sorry

end directrix_of_parabola_l54_54464


namespace train_crosses_signal_post_in_40_seconds_l54_54708

noncomputable def time_to_cross_signal_post : Nat := 40

theorem train_crosses_signal_post_in_40_seconds
  (train_length : Nat) -- Length of the train in meters
  (bridge_length_km : Nat) -- Length of the bridge in kilometers
  (bridge_cross_time_min : Nat) -- Time to cross the bridge in minutes
  (constant_speed : Prop) -- Assumption that the speed is constant
  (h1 : train_length = 600) -- Train is 600 meters long
  (h2 : bridge_length_km = 9) -- Bridge is 9 kilometers long
  (h3 : bridge_cross_time_min = 10) -- Time to cross the bridge is 10 minutes
  (h4 : constant_speed) -- The train's speed is constant
  : time_to_cross_signal_post = 40 :=
sorry

end train_crosses_signal_post_in_40_seconds_l54_54708


namespace part1_part1_decreasing_part2_part3_l54_54919

noncomputable def f (x : ℝ) (a : ℝ) := (Real.log (1-x) + a * x ^ 2 + x)

theorem part1 (x : ℝ) : f x (1/2) = Real.log (1-x) + (1/2) * x ^ 2 + x := by
  sorry

theorem part1_decreasing (x : ℝ) : x ∈ Ioo (-∞) 1 → ((Real.log (1-x) + (1/2) * x ^ 2 + x) < (Real.log (1-y) + (1/2) * y ^ 2 + y)) -> x > y := by
  sorry

theorem part2 (a : ℝ) (x : ℝ) : 0 < a ∧ a ≤ 1 / 2 ∧ x ∈ Ioo (0 : ℝ) (1 : ℝ) → (Real.log (1-x) + a * x ^ 2 + x < 0) := by
  sorry

theorem part3 (n : ℕ) : 0 < n → (Real.log (1 + n) - (Finset.sum (Finset.range (n + 1)) (λ i, 1 / (i + 1 : ℝ))) > 1 - 1 / (2 * n)) := by
  sorry

end part1_part1_decreasing_part2_part3_l54_54919


namespace max_marks_l54_54724

theorem max_marks (M : ℝ) :
  (0.33 * M = 125 + 73) → M = 600 := by
  intro h
  sorry

end max_marks_l54_54724


namespace arrangement_plans_l54_54395

theorem arrangement_plans (classes students: ℕ) (selectClasses: ℕ): 
  classes = 6 → students = 4 → selectClasses = 2 →
  (nat.choose classes selectClasses) * (nat.fact students / (nat.fact selectClasses * nat.fact (students - selectClasses))) = 90 :=
by
  intros h_classes h_students h_selectClasses
  rw [h_classes, h_students, h_selectClasses]
  norm_num
  sorry

end arrangement_plans_l54_54395


namespace min_number_of_participants_l54_54184

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l54_54184


namespace activities_equally_popular_l54_54859

def Dodgeball_prefers : ℚ := 10 / 25
def ArtWorkshop_prefers : ℚ := 12 / 30
def MovieScreening_prefers : ℚ := 18 / 45
def QuizBowl_prefers : ℚ := 16 / 40

theorem activities_equally_popular :
  Dodgeball_prefers = ArtWorkshop_prefers ∧
  ArtWorkshop_prefers = MovieScreening_prefers ∧
  MovieScreening_prefers = QuizBowl_prefers :=
by
  sorry

end activities_equally_popular_l54_54859


namespace square_prob_distance_l54_54951

noncomputable def probability_distance_ge_1 (S : set (ℝ × ℝ)) (side_len : ℝ) : ℝ :=
  let a := 28 in
  let b := 1 in
  let c := 32 in
  (a - b * Real.pi) / c

theorem square_prob_distance (side_len : ℝ) (hS : side_len = 2) :
  probability_distance_ge_1 {p | p.1 ≥ 0 ∧ p.1 ≤ side_len ∧ p.2 ≥ 0 ∧ p.2 ≤ side_len} side_len = (28 - Real.pi) / 32 :=
by {
  rw hS,
  unfold probability_distance_ge_1,
  sorry
}

end square_prob_distance_l54_54951


namespace LCM_4_6_15_is_60_l54_54995

def prime_factors (n : ℕ) : List ℕ :=
  [] -- placeholder, definition of prime_factor is not necessary for the problem statement, so we leave it abstract

def LCM (a b : ℕ) : ℕ := 
  sorry -- placeholder, definition of LCM not directly necessary for the statement

theorem LCM_4_6_15_is_60 : LCM (LCM 4 6) 15 = 60 := 
  sorry

end LCM_4_6_15_is_60_l54_54995


namespace simplify_polynomial_l54_54363

theorem simplify_polynomial (s : ℝ) :
  (2*s^2 + 5*s - 3) - (2*s^2 + 9*s - 7) = -4*s + 4 :=
by
  sorry

end simplify_polynomial_l54_54363


namespace distance_between_first_and_last_bushes_l54_54374

theorem distance_between_first_and_last_bushes 
  (bushes : Nat)
  (spaces_per_bush : ℕ) 
  (distance_first_to_fifth : ℕ) 
  (total_bushes : bushes = 10)
  (fifth_bush_distance : distance_first_to_fifth = 100)
  : ∃ (d : ℕ), d = 225 :=
by
  sorry

end distance_between_first_and_last_bushes_l54_54374


namespace average_percentage_of_first_20_percent_l54_54631

theorem average_percentage_of_first_20_percent (X : ℝ) 
  (h1 : 0.20 * X + 0.50 * 60 + 0.30 * 40 = 58) : 
  X = 80 :=
sorry

end average_percentage_of_first_20_percent_l54_54631


namespace notebook_cost_l54_54151

theorem notebook_cost
  (students : ℕ)
  (majority_students : ℕ)
  (cost : ℕ)
  (notebooks : ℕ)
  (h1 : students = 36)
  (h2 : majority_students > 18)
  (h3 : notebooks > 1)
  (h4 : cost > notebooks)
  (h5 : majority_students * cost * notebooks = 2079) :
  cost = 11 :=
by
  sorry

end notebook_cost_l54_54151


namespace probability_of_stopping_on_corner_l54_54606

def grid_size : ℕ := 4

inductive Direction
| up
| down
| left
| right

structure Position :=
(x : ℕ)
(y : ℕ)
deriving DecidableEq

def initial_position : Position := { x := 0, y := 0 }

def is_corner (pos : Position) : Bool :=
  (pos.x = 0 ∧ pos.y = 0) ∨
  (pos.x = grid_size - 1 ∧ pos.y = 0) ∨
  (pos.x = 0 ∧ pos.y = grid_size - 1) ∨
  (pos.x = grid_size - 1 ∧ pos.y = grid_size - 1)

def move (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.up    => { p with y := (p.y + 1) % grid_size }
  | Direction.down  => { p with y := (p.y + grid_size - 1) % grid_size }
  | Direction.left  => { p with x := (p.x + grid_size - 1) % grid_size }
  | Direction.right => { p with x := (p.x + 1) % grid_size }

def random_walk (p : Position) (steps : ℕ) : list Position :=
  if steps = 0 then [p]
  else
    let next_positions := Direction.all.map (move p)
    p :: (next_positions.choose random_walk (steps - 1)).flatten

def probability_stopping_on_corner : ℚ := 89 / 256

theorem probability_of_stopping_on_corner :
  ∀ p, p = initial_position →
  (∃ steps ≤ 5, is_corner (random_walk p steps).last = true) →
  probability_stopping_on_corner = (89 / 256) := by
    sorry

end probability_of_stopping_on_corner_l54_54606


namespace sum_of_first_3n_terms_l54_54921

theorem sum_of_first_3n_terms (n a d : ℝ) 
  (h1 : n > 0)
  (S_n : ∑ i in finset.range n, (a + i * d) = 48)
  (S_2n : ∑ i in finset.range (2 * n), (a + i * d) = 60) :
  ∑ i in finset.range (3 * n), (a + i * d) = 36 :=
by
  sorry

end sum_of_first_3n_terms_l54_54921


namespace goose_eggs_l54_54267

theorem goose_eggs (E : ℝ) :
  (E / 2 * 3 / 4 * 2 / 5 + (1 / 3 * (E / 2)) * 2 / 3 * 3 / 4 + (1 / 6 * (E / 2 + E / 6)) * 1 / 2 * 2 / 3 = 150) →
  E = 375 :=
by
  sorry

end goose_eggs_l54_54267


namespace slope_of_perpendicular_line_l54_54603

noncomputable def perpendicular_slope : ℚ :=
  let slope := (3/4 : ℚ) in
  - (1 / slope)

theorem slope_of_perpendicular_line (a b c : ℚ) (h : a = 3 ∧ b = -4 ∧ c = 8) : 
  perpendicular_slope = - (4 / 3) :=
by {
  sorry
}

end slope_of_perpendicular_line_l54_54603


namespace scrabble_middle_letter_value_l54_54496

theorem scrabble_middle_letter_value 
  (triple_word_score : ℕ) (single_letter_value : ℕ) (middle_letter_value : ℕ)
  (h1 : triple_word_score = 30)
  (h2 : single_letter_value = 1)
  : 3 * (2 * single_letter_value + middle_letter_value) = triple_word_score → middle_letter_value = 8 :=
by
  sorry

end scrabble_middle_letter_value_l54_54496


namespace brownies_shared_l54_54073

theorem brownies_shared
  (total_brownies : ℕ)
  (tina_brownies : ℕ)
  (husband_brownies : ℕ)
  (remaining_brownies : ℕ)
  (shared_brownies : ℕ)
  (h1 : total_brownies = 24)
  (h2 : tina_brownies = 10)
  (h3 : husband_brownies = 5)
  (h4 : remaining_brownies = 5) :
  shared_brownies = total_brownies - (tina_brownies + husband_brownies + remaining_brownies) → shared_brownies = 4 :=
by
  sorry

end brownies_shared_l54_54073


namespace vasya_mushrooms_l54_54076

-- Lean definition of the problem based on the given conditions
theorem vasya_mushrooms :
  ∃ (N : ℕ), 
    N ≥ 100 ∧ N < 1000 ∧
    (∃ (a b c : ℕ), a ≠ 0 ∧ N = 100 * a + 10 * b + c ∧ a + b + c = 14) ∧
    N % 50 = 0 ∧ 
    N = 950 :=
by
  sorry

end vasya_mushrooms_l54_54076


namespace nails_needed_for_house_wall_l54_54754

theorem nails_needed_for_house_wall
    (large_planks : ℕ)
    (small_planks : ℕ)
    (nails_for_large_planks : ℕ)
    (nails_for_small_planks : ℕ)
    (H1 : large_planks = 12)
    (H2 : small_planks = 10)
    (H3 : nails_for_large_planks = 15)
    (H4 : nails_for_small_planks = 5) :
    (nails_for_large_planks + nails_for_small_planks) = 20 := by
  sorry

end nails_needed_for_house_wall_l54_54754


namespace race_minimum_participants_l54_54186

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l54_54186


namespace locus_of_points_is_straight_line_l54_54942

theorem locus_of_points_is_straight_line 
  (a R1 R2 : ℝ) 
  (h_nonzero_a : a ≠ 0)
  (h_positive_R1 : R1 > 0)
  (h_positive_R2 : R2 > 0) :
  ∃ x : ℝ, ∀ (y : ℝ),
  ((x + a)^2 + y^2 - R1^2 = (x - a)^2 + y^2 - R2^2) ↔ 
  x = (R1^2 - R2^2) / (4 * a) :=
by
  sorry

end locus_of_points_is_straight_line_l54_54942


namespace tan_theta_3_l54_54781

noncomputable def tan_triple_angle (θ : ℝ) : ℝ := (3 * (Real.tan θ) - ((Real.tan θ) ^ 3)) / (1 - 3 * (Real.tan θ)^2)

theorem tan_theta_3 (θ : ℝ) (h : Real.tan θ = 3) : tan_triple_angle θ = 9 / 13 :=
by
  sorry

end tan_theta_3_l54_54781


namespace find_y_given_x_l54_54862

-- Let x and y be real numbers
variables (x y : ℝ)

-- Assume x and y are inversely proportional, so their product is a constant C
variable (C : ℝ)

-- Additional conditions from the problem statement
variable (h1 : x + y = 40) (h2 : x - y = 10) (hx : x = 7)

-- Define the goal: y = 375 / 7
theorem find_y_given_x : y = 375 / 7 :=
sorry

end find_y_given_x_l54_54862


namespace area_of_rectangle_with_diagonal_length_l54_54894

variable (x : ℝ)

def rectangle_area_given_diagonal_length (x : ℝ) : Prop :=
  ∃ (w l : ℝ), l = 3 * w ∧ w^2 + l^2 = x^2 ∧ (w * l = (3 / 10) * x^2)

theorem area_of_rectangle_with_diagonal_length (x : ℝ) :
  rectangle_area_given_diagonal_length x :=
sorry

end area_of_rectangle_with_diagonal_length_l54_54894


namespace combined_resistance_parallel_l54_54029

theorem combined_resistance_parallel (R1 R2 R3 : ℝ) (r : ℝ) (h1 : R1 = 2) (h2 : R2 = 5) (h3 : R3 = 6) :
  (1 / r) = (1 / R1) + (1 / R2) + (1 / R3) → r = 15 / 13 :=
by
  sorry

end combined_resistance_parallel_l54_54029


namespace abs_inequality_example_l54_54600

theorem abs_inequality_example (x : ℝ) : abs (5 - x) < 6 ↔ -1 < x ∧ x < 11 :=
by 
  sorry

end abs_inequality_example_l54_54600


namespace bananas_unit_measurement_l54_54710

-- Definition of given conditions
def units_per_day : ℕ := 13
def total_bananas : ℕ := 9828
def total_weeks : ℕ := 9
def days_per_week : ℕ := 7
def total_days : ℕ := total_weeks * days_per_week
def bananas_per_day : ℕ := total_bananas / total_days
def bananas_per_unit : ℕ := bananas_per_day / units_per_day

-- Main theorem statement
theorem bananas_unit_measurement :
  bananas_per_unit = 12 := sorry

end bananas_unit_measurement_l54_54710


namespace circle_condition_l54_54489

theorem circle_condition (k : ℝ) : (∃ x y : ℝ, x^2 + y^2 + 4 * x + 2 * y + 4 * k + 1 = 0) → (k < 1) :=
by
  sorry

end circle_condition_l54_54489


namespace calen_more_pencils_l54_54583

def calen_pencils (C B D: ℕ) :=
  D = 9 ∧
  B = 2 * D - 3 ∧
  C - 10 = 10

theorem calen_more_pencils (C B D : ℕ) (h : calen_pencils C B D) : C = B + 5 :=
by
  obtain ⟨hD, hB, hC⟩ := h
  simp only [hD, hB, hC]
  sorry

end calen_more_pencils_l54_54583


namespace total_lives_l54_54248

noncomputable def C : ℝ := 9.5
noncomputable def D : ℝ := C - 3.25
noncomputable def M : ℝ := D + 7.75
noncomputable def E : ℝ := 2 * C - 5.5
noncomputable def F : ℝ := 2/3 * E

theorem total_lives : C + D + M + E + F = 52.25 :=
by
  sorry

end total_lives_l54_54248


namespace max_regular_hours_correct_l54_54565

-- Define the conditions
def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
def total_hours_worked : ℝ := 57
def total_compensation : ℝ := 1116

-- Define the maximum regular hours per week
def max_regular_hours : ℝ := 40

-- Define the compensation equation
def compensation (H : ℝ) : ℝ :=
  regular_rate * H + overtime_rate * (total_hours_worked - H)

-- The theorem that needs to be proved
theorem max_regular_hours_correct :
  compensation max_regular_hours = total_compensation :=
by
  -- skolemize the proof
  sorry

end max_regular_hours_correct_l54_54565


namespace inner_square_area_l54_54372

theorem inner_square_area (side_ABCD : ℝ) (dist_BI : ℝ) (area_IJKL : ℝ) :
  side_ABCD = Real.sqrt 72 →
  dist_BI = 2 →
  area_IJKL = 39 :=
by
  sorry

end inner_square_area_l54_54372


namespace variance_is_4_l54_54765

variable {datapoints : List ℝ}

noncomputable def variance (datapoints : List ℝ) : ℝ :=
  let n := datapoints.length
  let mean := (datapoints.sum / n : ℝ)
  (1 / n : ℝ) * ((datapoints.map (λ x => x ^ 2)).sum - n * mean ^ 2)

theorem variance_is_4 :
  (datapoints.length = 20)
  → ((datapoints.map (λ x => x ^ 2)).sum = 800)
  → (datapoints.sum / 20 = 6)
  → variance datapoints = 4 := by
  intros length_cond sum_squares_cond mean_cond
  sorry

end variance_is_4_l54_54765


namespace fraction_division_l54_54233

theorem fraction_division : (3 / 4) / (2 / 5) = 15 / 8 := 
by
  -- We need to convert this division into multiplication by the reciprocal
  -- (3 / 4) / (2 / 5) = (3 / 4) * (5 / 2)
  -- Now perform the multiplication of the numerators and denominators
  -- (3 * 5) / (4 * 2) = 15 / 8
  sorry

end fraction_division_l54_54233


namespace coords_of_P_max_PA_distance_l54_54924

open Real

noncomputable def A : (ℝ × ℝ) := (0, -5)

def on_circle (P : ℝ × ℝ) : Prop :=
  ∃ x y, x = P.1 ∧ y = P.2 ∧ (x - 2)^2 + (y + 3)^2 = 2

def max_PA_distance (P : (ℝ × ℝ)) : Prop :=
  dist P A = max (dist (3, -2) A) (dist (1, -4) A)

theorem coords_of_P_max_PA_distance (P : (ℝ × ℝ)) :
  on_circle P →
  max_PA_distance P →
  P = (3, -2) :=
  sorry

end coords_of_P_max_PA_distance_l54_54924


namespace eggs_given_by_Andrew_l54_54309

variable (total_eggs := 222)
variable (eggs_to_buy := 67)
variable (eggs_given : ℕ)

theorem eggs_given_by_Andrew :
  eggs_given = total_eggs - eggs_to_buy ↔ eggs_given = 155 := 
by 
  sorry

end eggs_given_by_Andrew_l54_54309


namespace milk_left_l54_54661

theorem milk_left (initial_milk : ℝ) (milk_james : ℝ) (milk_maria : ℝ) :
  initial_milk = 5 → milk_james = 15 / 4 → milk_maria = 3 / 4 → 
  initial_milk - (milk_james + milk_maria) = 1 / 2 :=
by
  intros h_initial h_james h_maria
  rw [h_initial, h_james, h_maria]
  -- The calculation would be performed here.
  sorry

end milk_left_l54_54661


namespace stuart_initial_marbles_l54_54097

theorem stuart_initial_marbles (B S : ℝ) (h1 : B = 60) (h2 : 0.40 * B = 24) (h3 : S + 24 = 80) : S = 56 :=
by
  sorry

end stuart_initial_marbles_l54_54097


namespace find_number_l54_54491

theorem find_number (x : ℕ) (h : (x + 720) / 125 = 7392 / 462) : x = 1280 :=
sorry

end find_number_l54_54491


namespace evaluate_ceil_of_neg_sqrt_l54_54111

-- Define the given expression and its value computation
def given_expression : ℚ := -real.sqrt (64 / 9)

-- Define the expected answer
def expected_answer : ℤ := -2

-- State the theorem to be proven
theorem evaluate_ceil_of_neg_sqrt : (Int.ceil given_expression) = expected_answer := sorry

end evaluate_ceil_of_neg_sqrt_l54_54111


namespace road_trip_total_miles_l54_54230

theorem road_trip_total_miles (tracy_miles michelle_miles katie_miles : ℕ) (h_michelle : michelle_miles = 294)
    (h_tracy : tracy_miles = 2 * michelle_miles + 20) (h_katie : michelle_miles = 3 * katie_miles):
  tracy_miles + michelle_miles + katie_miles = 1000 :=
by
  sorry

end road_trip_total_miles_l54_54230


namespace area_arccos_cos_l54_54462

open Real

theorem area_arccos_cos (a b : ℝ) (h : a = 0 ∧ b = 3 * π) :
  ∫ x in a..b, arccos (cos x) = (3 / 2) * π^2 :=
by
  have h1 : ∫ x in 0..π, x = (1 / 2) * π^2, by sorry
  have h2 : ∫ x in π..2 * π, 2 * π - x = (1 / 2) * π^2, by sorry
  have h3 : ∫ x in 2 * π..3 * π, x - 2 * π = (1 / 2) * π^2, by sorry
  calc
    ∫ x in a..b, arccos (cos x)
        = ∫ x in 0..π, arccos (cos x) + ∫ x in π..2 * π, arccos (cos x) +
          ∫ x in 2 * π..3 * π, arccos (cos x) : by sorry
    ... = ∫ x in 0..π, x + ∫ x in π..2 * π, 2 * π - x + ∫ x in 2 * π..3 * π, x - 2 * π : by sorry
    ... = (1 / 2) * π^2 + (1 / 2) * π^2 + (1 / 2) * π^2 : by rw [h1, h2, h3]
    ... = (3 / 2) * π^2 : by norm_num

end area_arccos_cos_l54_54462


namespace problem_solve_l54_54511

theorem problem_solve (x y : ℝ) (h1 : x ≠ y) (h2 : x / y + (x + 6 * y) / (y + 6 * x) = 3) : 
    x / y = (8 + Real.sqrt 46) / 6 := 
  sorry

end problem_solve_l54_54511


namespace grunters_win_all_5_games_grunters_win_at_least_one_game_l54_54375

/-- Given that the Grunters have an independent probability of 3/4 of winning any single game, 
and they play 5 games, the probability that the Grunters will win all 5 games is 243/1024. --/
theorem grunters_win_all_5_games :
  (3/4)^5 = 243 / 1024 :=
sorry

/-- Given that the Grunters have an independent probability of 3/4 of winning any single game, 
and they play 5 games, the probability that the Grunters will win at least one game is 1023/1024. --/
theorem grunters_win_at_least_one_game :
  1 - (1/4)^5 = 1023 / 1024 :=
sorry

end grunters_win_all_5_games_grunters_win_at_least_one_game_l54_54375


namespace quadratic_roots_condition_l54_54938

theorem quadratic_roots_condition (m : ℝ) :
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1^2 - 3 * x1 + 2 * m = 0 ∧ x2^2 - 3 * x2 + 2 * m = 0) →
  m < 9 / 8 :=
by
  sorry

end quadratic_roots_condition_l54_54938


namespace total_people_on_bus_l54_54102

def initial_people := 4
def added_people := 13

theorem total_people_on_bus : initial_people + added_people = 17 := by
  sorry

end total_people_on_bus_l54_54102


namespace trigonometric_identity_l54_54476

theorem trigonometric_identity 
  (α m : ℝ) 
  (h : Real.tan (α / 2) = m) :
  (1 - 2 * (Real.sin (α / 2))^2) / (1 + Real.sin α) = (1 - m) / (1 + m) :=
by
  sorry

end trigonometric_identity_l54_54476


namespace KatieMarbles_l54_54948

variable {O P : ℕ}

theorem KatieMarbles :
  13 + O + P = 33 → P = 4 * O → 13 - O = 9 :=
by
  sorry

end KatieMarbles_l54_54948


namespace students_in_college_l54_54637

variable (P S : ℕ)

def condition1 : Prop := S = 15 * P
def condition2 : Prop := S + P = 40000

theorem students_in_college (h1 : condition1 S P) (h2 : condition2 S P) : S = 37500 := by
  sorry

end students_in_college_l54_54637


namespace man_l54_54261

theorem man's_speed_kmph (length_train : ℝ) (time_seconds : ℝ) (speed_train_kmph : ℝ) : ℝ :=
  let speed_train_mps := speed_train_kmph * (5/18)
  let rel_speed_mps := length_train / time_seconds
  let man_speed_mps := rel_speed_mps - speed_train_mps
  man_speed_mps * (18/5)

example : man's_speed_kmph 120 6 65.99424046076315 = 6.00735873483709 := by
  sorry

end man_l54_54261


namespace inequality_triangle_areas_l54_54169

theorem inequality_triangle_areas (a b c α β γ : ℝ) (hα : α = 2 * Real.sqrt (b * c)) (hβ : β = 2 * Real.sqrt (a * c)) (hγ : γ = 2 * Real.sqrt (a * b)) : 
  a / α + b / β + c / γ ≥ 3 / 2 := 
by
  sorry

end inequality_triangle_areas_l54_54169


namespace max_XG_l54_54358

theorem max_XG :
  ∀ (G X Y Z : ℝ),
    Y - X = 5 ∧ Z - Y = 3 ∧ (1 / G + 1 / (G - 5) + 1 / (G - 8) = 0) →
    G = 20 / 3 :=
by
  sorry

end max_XG_l54_54358


namespace pascal_triangle_row_sum_l54_54312

theorem pascal_triangle_row_sum : (∑ n in Finset.range 30, n + 1) = 465 := by
  sorry

end pascal_triangle_row_sum_l54_54312


namespace consistent_price_per_kg_l54_54441

theorem consistent_price_per_kg (m₁ m₂ : ℝ) (p₁ p₂ : ℝ)
  (h₁ : p₁ = 6) (h₂ : m₁ = 2)
  (h₃ : p₂ = 36) (h₄ : m₂ = 12) :
  (p₁ / m₁ = p₂ / m₂) := 
by 
  sorry

end consistent_price_per_kg_l54_54441


namespace exists_distinct_group_and_country_selection_l54_54993

theorem exists_distinct_group_and_country_selection 
  (n m : ℕ) 
  (h_nm1 : n > m) 
  (h_m1 : m > 1) 
  (groups : Fin n → Fin m → Fin n → Prop) 
  (group_conditions : ∀ i j : Fin n, ∀ k : Fin m, ∀ l : Fin m, (i ≠ j) → (groups i k j = false)) 
  : 
  ∃ (selected : Fin n → Fin (m * n)), 
    (∀ i j: Fin n, i ≠ j → selected i ≠ selected j) ∧ 
    (∀ i j: Fin n, selected i / m ≠ selected j / m) := sorry

end exists_distinct_group_and_country_selection_l54_54993


namespace mt_product_l54_54346

def g : ℝ → ℝ := sorry

axiom func_eqn (x y : ℝ) : g (x * g y + 2 * x) = 2 * x * y + g x

axiom g3_value : g 3 = 6

def m : ℕ := 1

def t : ℝ := 6

theorem mt_product : m * t = 6 :=
by 
  sorry

end mt_product_l54_54346


namespace coeff_exists_l54_54338

theorem coeff_exists :
  ∃ (A B C : ℕ), 
    ¬(8 ∣ A) ∧ ¬(8 ∣ B) ∧ ¬(8 ∣ C) ∧ 
    (∀ (n : ℕ), 8 ∣ (A * 5^n + B * 3^(n-1) + C))
    :=
sorry

end coeff_exists_l54_54338


namespace geometric_sequence_properties_l54_54611

theorem geometric_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) (h1 : ∀ n, S n = 3^n + t) (h2 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 2 = 6 ∧ t = -1 :=
by
  sorry

end geometric_sequence_properties_l54_54611


namespace plywood_long_side_length_l54_54893

theorem plywood_long_side_length (L : ℕ) (h1 : 2 * (L + 5) = 22) : L = 6 :=
by
  sorry

end plywood_long_side_length_l54_54893


namespace no_extreme_value_min_int_k_l54_54770

-- Define the function f(x) = e^x(-x + ln x + a)
noncomputable def f (x a : ℝ) : ℝ := Real.exp x * (-x + Real.log x + a)

-- Define the condition: a ≤ 1
axiom a_leq_one {a : ℝ} : a ≤ 1

-- Problem (I): Prove that f(x) has no extreme value point in the interval (1, e)
theorem no_extreme_value (a_leq_one : a ≤ 1) : 
  ¬ ∃ x ∈ (1 : ℝ, Real.exp 1), IsLocalMin (f x a) x ∨ IsLocalMax (f x a) x :=
begin
  sorry
end

-- Problem (II): Prove that minimum integer k such that f(x) < k for all x when a = ln 2 is 0
theorem min_int_k (a_ln2 : (a = Real.log 2)) : 
  ∃ k ∈ Int, (∀ x > 0, f x (Real.log 2) < k) ∧ k = 0 :=
begin
  sorry
end

end no_extreme_value_min_int_k_l54_54770


namespace johns_out_of_pocket_l54_54651

noncomputable def total_cost_after_discounts (computer_cost gaming_chair_cost accessories_cost : ℝ) 
  (comp_discount gaming_discount : ℝ) (tax : ℝ) : ℝ :=
  let comp_price := computer_cost * (1 - comp_discount)
  let chair_price := gaming_chair_cost * (1 - gaming_discount)
  let pre_tax_total := comp_price + chair_price + accessories_cost
  pre_tax_total * (1 + tax)

noncomputable def total_selling_price (playstation_value playstation_discount bicycle_price : ℝ) (exchange_rate : ℝ) : ℝ :=
  let playstation_price := playstation_value * (1 - playstation_discount)
  (playstation_price * exchange_rate) / exchange_rate + bicycle_price

theorem johns_out_of_pocket (computer_cost gaming_chair_cost accessories_cost comp_discount gaming_discount tax 
  playstation_value playstation_discount bicycle_price exchange_rate : ℝ) :
  computer_cost = 1500 →
  gaming_chair_cost = 400 →
  accessories_cost = 300 →
  comp_discount = 0.2 →
  gaming_discount = 0.1 →
  tax = 0.05 →
  playstation_value = 600 →
  playstation_discount = 0.2 →
  bicycle_price = 200 →
  exchange_rate = 100 →
  total_cost_after_discounts computer_cost gaming_chair_cost accessories_cost comp_discount gaming_discount tax -
  total_selling_price playstation_value playstation_discount bicycle_price exchange_rate = 1273 := by
  intros
  sorry

end johns_out_of_pocket_l54_54651


namespace roots_quartic_ab_plus_a_plus_b_l54_54823

theorem roots_quartic_ab_plus_a_plus_b (a b : ℝ) 
  (h1 : Polynomial.eval a (Polynomial.C 1 + Polynomial.C (-4) * Polynomial.X + Polynomial.C (-6) * (Polynomial.X ^ 2) + Polynomial.C 1 * (Polynomial.X ^ 4)) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1 + Polynomial.C (-4) * Polynomial.X + Polynomial.C (-6) * (Polynomial.X ^ 2) + Polynomial.C 1 * (Polynomial.X ^ 4)) = 0) :
  a * b + a + b = -1 := 
sorry

end roots_quartic_ab_plus_a_plus_b_l54_54823


namespace quadratic_min_value_l54_54147

theorem quadratic_min_value (p r : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x^2 + 2 * p * x + r) (h₁ : ∃ x₀, f x₀ = 1 ∧ ∀ x, f x₀ ≤ f x) : r = p^2 + 1 :=
by
  sorry

end quadratic_min_value_l54_54147


namespace minimum_participants_l54_54191

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l54_54191


namespace profit_made_after_two_years_l54_54861

variable (present_value : ℝ) (depreciation_rate : ℝ) (selling_price : ℝ) 

def value_after_one_year (present_value depreciation_rate : ℝ) : ℝ :=
  present_value - (depreciation_rate * present_value)

def value_after_two_years (value_after_one_year : ℝ) (depreciation_rate : ℝ) : ℝ :=
  value_after_one_year - (depreciation_rate * value_after_one_year)

def profit (selling_price value_after_two_years : ℝ) : ℝ :=
  selling_price - value_after_two_years

theorem profit_made_after_two_years
  (h_present_value : present_value = 150000)
  (h_depreciation_rate : depreciation_rate = 0.22)
  (h_selling_price : selling_price = 115260) :
  profit selling_price (value_after_two_years (value_after_one_year present_value depreciation_rate) depreciation_rate) = 24000 := 
by
  sorry

end profit_made_after_two_years_l54_54861


namespace completing_square_solution_l54_54693

theorem completing_square_solution (x : ℝ) : x^2 - 4 * x - 22 = 0 ↔ (x - 2)^2 = 26 := sorry

end completing_square_solution_l54_54693


namespace path_count_l54_54889

theorem path_count (f : ℕ → (ℤ × ℤ)) :
  (∀ n, (f (n + 1)).1 = (f n).1 + 1 ∨ (f (n + 1)).2 = (f n).2 + 1) ∧
  f 0 = (-6, -6) ∧ f 24 = (6, 6) ∧
  (∀ n, ¬(-3 ≤ (f n).1 ∧ (f n).1 ≤ 3 ∧ -3 ≤ (f n).2 ∧ (f n).2 ≤ 3)) →
  ∃ N, N = 2243554 :=
by {
  sorry
}

end path_count_l54_54889


namespace original_three_numbers_are_arith_geo_seq_l54_54549

theorem original_three_numbers_are_arith_geo_seq
  (x y z : ℕ) (h1 : ∃ k : ℕ, x = 3*k ∧ y = 4*k ∧ z = 5*k)
  (h2 : ∃ r : ℝ, (x + 1) / y = r ∧ y / z = r ∧ r^2 = z / y):
  x = 15 ∧ y = 20 ∧ z = 25 :=
by 
  sorry

end original_three_numbers_are_arith_geo_seq_l54_54549


namespace simplify_tan_cot_60_l54_54055

theorem simplify_tan_cot_60 :
  let tan60 := Real.sqrt 3
  let cot60 := 1 / Real.sqrt 3
  (tan60^3 + cot60^3) / (tan60 + cot60) = 7 / 3 :=
by
  let tan60 := Real.sqrt 3
  let cot60 := 1 / Real.sqrt 3
  sorry

end simplify_tan_cot_60_l54_54055


namespace math_problem_l54_54004

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l54_54004


namespace gcd_of_three_numbers_l54_54980

theorem gcd_of_three_numbers (a b c d : ℕ) (ha : a = 72) (hb : b = 120) (hc : c = 168) (hd : d = 24) : 
  Nat.gcd (Nat.gcd a b) c = d :=
by
  rw [ha, hb, hc, hd]
  -- Placeholder for the actual proof
  exact sorry

end gcd_of_three_numbers_l54_54980


namespace minimum_participants_l54_54200

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l54_54200


namespace card_prob_ace_of_hearts_l54_54437

def problem_card_probability : Prop :=
  let deck_size := 52
  let draw_size := 2
  let ace_hearts := 1
  let total_combinations := Nat.choose deck_size draw_size
  let favorable_combinations := deck_size - ace_hearts
  let probability := favorable_combinations / total_combinations
  probability = 1 / 26

theorem card_prob_ace_of_hearts : problem_card_probability := by
  sorry

end card_prob_ace_of_hearts_l54_54437


namespace perpendicular_lines_k_value_l54_54124

theorem perpendicular_lines_k_value (k : ℝ) : 
  k * (k - 1) + (1 - k) * (2 * k + 3) = 0 ↔ k = -3 ∨ k = 1 :=
by
  sorry

end perpendicular_lines_k_value_l54_54124


namespace relationship_of_products_l54_54323

theorem relationship_of_products
  {a1 a2 b1 b2 : ℝ}
  (h1 : a1 < a2)
  (h2 : b1 < b2) :
  a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 :=
sorry

end relationship_of_products_l54_54323


namespace sufficient_but_not_necessary_l54_54328

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 0): (x = 1 → x > 0) ∧ ¬(x > 0 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_l54_54328


namespace simplify_175_sub_57_sub_43_simplify_128_sub_64_sub_36_simplify_156_sub_49_sub_51_l54_54272

theorem simplify_175_sub_57_sub_43 : 175 - 57 - 43 = 75 :=
by
  sorry

theorem simplify_128_sub_64_sub_36 : 128 - 64 - 36 = 28 :=
by
  sorry

theorem simplify_156_sub_49_sub_51 : 156 - 49 - 51 = 56 :=
by
  sorry

end simplify_175_sub_57_sub_43_simplify_128_sub_64_sub_36_simplify_156_sub_49_sub_51_l54_54272


namespace double_neg_cancel_l54_54364

theorem double_neg_cancel (a : ℤ) : - (-2) = 2 :=
sorry

end double_neg_cancel_l54_54364


namespace number_of_people_is_8_l54_54979

noncomputable def find_number_of_people (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) (weight_diff : ℝ) (n : ℕ) :=
  avg_increase = weight_diff / n ∧ old_weight = 70 ∧ new_weight = 90 ∧ weight_diff = new_weight - old_weight → n = 8

theorem number_of_people_is_8 :
  ∃ n : ℕ, find_number_of_people 2.5 70 90 20 n :=
by
  use 8
  sorry

end number_of_people_is_8_l54_54979


namespace opposite_of_2023_is_neg_2023_l54_54679

-- Definitions based on conditions
def is_additive_inverse (x y : Int) : Prop := x + y = 0

-- The proof statement
theorem opposite_of_2023_is_neg_2023 : is_additive_inverse 2023 (-2023) :=
by
  -- This is where the proof would go, but it is marked as sorry for now
  sorry

end opposite_of_2023_is_neg_2023_l54_54679


namespace cube_side_length_equals_six_l54_54221

theorem cube_side_length_equals_six {s : ℝ} (h : 6 * s ^ 2 = s ^ 3) : s = 6 :=
by
  sorry

end cube_side_length_equals_six_l54_54221


namespace race_participants_least_number_l54_54207

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l54_54207


namespace math_problem_l54_54006

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l54_54006


namespace geometric_seq_an_minus_2_l54_54472

-- Definitions of conditions based on given problem
def seq_a : ℕ → ℝ := sorry -- The sequence {a_n}
def sum_s : ℕ → ℝ := sorry -- The sum of the first n terms {s_n}

axiom cond1 (n : ℕ) (hn : n > 0) : seq_a (n + 1) ≠ seq_a n
axiom cond2 (n : ℕ) (hn : n > 0) : sum_s n + seq_a n = 2 * n

-- Theorem statement
theorem geometric_seq_an_minus_2 (n : ℕ) (hn : n > 0) : 
  ∃ r : ℝ, ∀ k : ℕ, seq_a (k + 1) - 2 = r * (seq_a k - 2) := 
sorry

end geometric_seq_an_minus_2_l54_54472


namespace g_of_f_roots_reciprocal_l54_54040

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 3 * b * x + 4 * c

theorem g_of_f_roots_reciprocal
  (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  ∃ g : ℝ → ℝ, g 1 = (4 - a) / (4 * c) :=
sorry

end g_of_f_roots_reciprocal_l54_54040


namespace largest_angle_in_ratio_triangle_l54_54528

theorem largest_angle_in_ratio_triangle (a b c : ℕ) (h_ratios : 2 * c = 3 * b ∧ 3 * b = 4 * a)
  (h_sum : a + b + c = 180) : max a (max b c) = 80 :=
by
  sorry

end largest_angle_in_ratio_triangle_l54_54528


namespace number_of_matches_in_first_set_l54_54978

theorem number_of_matches_in_first_set
  (x : ℕ)
  (h1 : (30 : ℚ) * x + 15 * 10 = 25 * (x + 10)) :
  x = 20 :=
by
  -- The proof will be filled in here
  sorry

end number_of_matches_in_first_set_l54_54978


namespace quadratic_has_one_real_root_positive_value_of_m_l54_54490

theorem quadratic_has_one_real_root (m : ℝ) (h : (4 * m) * (4 * m) - 4 * 1 * m = 0) : m = 0 ∨ m = 1/4 := by
  sorry

theorem positive_value_of_m (m : ℝ) (h : (4 * m) * (4 * m) - 4 * 1 * m = 0) : m = 1/4 := by
  have root_cases := quadratic_has_one_real_root m h
  cases root_cases
  · exfalso
    -- We know m = 0 cannot be the positive m we are looking for.
    sorry
  · assumption

end quadratic_has_one_real_root_positive_value_of_m_l54_54490


namespace sin_cos_inequality_for_any_x_l54_54687

noncomputable def largest_valid_n : ℕ := 8

theorem sin_cos_inequality_for_any_x (n : ℕ) (h : n = largest_valid_n) :
  ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / n :=
sorry

end sin_cos_inequality_for_any_x_l54_54687


namespace total_rainfall_january_l54_54905

theorem total_rainfall_january (R1 R2 T : ℝ) (h1 : R2 = 1.5 * R1) (h2 : R2 = 21) : T = 35 :=
by 
  let R1 := 14
  let R2 := 21
  let T := R1 + R2
  sorry

end total_rainfall_january_l54_54905


namespace square_side_factor_l54_54020

theorem square_side_factor (k : ℝ) (h : k^2 = 1) : k = 1 :=
sorry

end square_side_factor_l54_54020


namespace sodium_hydride_reaction_l54_54142

theorem sodium_hydride_reaction (H2O NaH NaOH H2 : ℕ) 
  (balanced_eq : NaH + H2O = NaOH + H2) 
  (stoichiometry : NaH = H2O → NaOH = H2 → NaH = H2) 
  (h : H2O = 2) : NaH = 2 :=
sorry

end sodium_hydride_reaction_l54_54142


namespace paint_left_l54_54310

-- Define the conditions
def total_paint_needed : ℕ := 333
def paint_needed_to_buy : ℕ := 176

-- State the theorem
theorem paint_left : total_paint_needed - paint_needed_to_buy = 157 := 
by 
  sorry

end paint_left_l54_54310


namespace cost_price_computer_table_l54_54386

variable (C : ℝ) -- Cost price of the computer table
variable (S : ℝ) -- Selling price of the computer table

-- Conditions based on the problem
axiom h1 : S = 1.10 * C
axiom h2 : S = 8800

-- The theorem to be proven
theorem cost_price_computer_table : C = 8000 :=
by
  -- Proof will go here
  sorry

end cost_price_computer_table_l54_54386


namespace polynomial_value_at_4_l54_54445

def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

theorem polynomial_value_at_4 : f 4 = 371 := by
  sorry

end polynomial_value_at_4_l54_54445


namespace ceil_of_neg_sqrt_frac_64_over_9_l54_54112

theorem ceil_of_neg_sqrt_frac_64_over_9 :
  ⌈-Real.sqrt (64 / 9)⌉ = -2 :=
by
  sorry

end ceil_of_neg_sqrt_frac_64_over_9_l54_54112


namespace Ellen_strawberries_used_l54_54743

theorem Ellen_strawberries_used :
  let yogurt := 0.1
  let orange_juice := 0.2
  let total_ingredients := 0.5
  let strawberries := total_ingredients - (yogurt + orange_juice)
  strawberries = 0.2 :=
by
  sorry

end Ellen_strawberries_used_l54_54743


namespace three_digit_even_with_sum_twelve_l54_54624

theorem three_digit_even_with_sum_twelve :
  ∃ n: ℕ, n = 36 ∧ 
    (∀ x, 100 ≤ x ∧ x ≤ 999 ∧ x % 2 = 0 ∧ 
          ((x / 10) % 10 + x % 10 = 12) → x = n) :=
sorry

end three_digit_even_with_sum_twelve_l54_54624


namespace value_of_f_l54_54349

noncomputable
def f (k l m x : ℝ) : ℝ := k + m / (x - l)

theorem value_of_f (k l m : ℝ) (hk : k = -2) (hl : l = 2.5) (hm : m = 12) :
  f k l m (k + l + m) = -4 / 5 :=
by
  sorry

end value_of_f_l54_54349


namespace sqrt_22_gt_4_l54_54275

theorem sqrt_22_gt_4 : Real.sqrt 22 > 4 := 
sorry

end sqrt_22_gt_4_l54_54275


namespace find_5_minus_a_l54_54018

-- Define the problem conditions as assumptions
variable (a b : ℤ)
variable (h1 : 5 + a = 6 - b)
variable (h2 : 3 + b = 8 + a)

-- State the theorem we want to prove
theorem find_5_minus_a : 5 - a = 7 :=
by
  sorry

end find_5_minus_a_l54_54018


namespace janet_earnings_per_hour_l54_54036

theorem janet_earnings_per_hour : 
  (∃ (rate_per_post : ℝ) (time_per_post : ℝ), 
    rate_per_post = 0.25 ∧ 
    time_per_post = 10 ∧ 
    (let posts_per_hour := 3600 / time_per_post in
     let earnings_per_hour := rate_per_post * posts_per_hour in
     earnings_per_hour = 90)) :=
by
  use 0.25
  use 10
  split
  rfl
  split
  rfl
  let posts_per_hour := 3600 / 10
  let earnings_per_hour := 0.25 * posts_per_hour
  have h : earnings_per_hour = 90, by sorry
  exact h

end janet_earnings_per_hour_l54_54036


namespace race_participants_minimum_l54_54177

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l54_54177


namespace minimum_participants_l54_54190

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l54_54190


namespace noel_baked_dozens_l54_54354

theorem noel_baked_dozens (total_students : ℕ) (percent_like_donuts : ℝ)
    (donuts_per_student : ℕ) (dozen : ℕ) (h_total_students : total_students = 30)
    (h_percent_like_donuts : percent_like_donuts = 0.80)
    (h_donuts_per_student : donuts_per_student = 2)
    (h_dozen : dozen = 12) :
    total_students * percent_like_donuts * donuts_per_student / dozen = 4 := 
by
  sorry

end noel_baked_dozens_l54_54354


namespace a_x1_x2_x13_eq_zero_l54_54219

theorem a_x1_x2_x13_eq_zero {a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ}
  (h1: a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) *
             (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2: a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) *
             (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := by
  sorry

end a_x1_x2_x13_eq_zero_l54_54219


namespace length_of_train_l54_54728

theorem length_of_train (speed_kmh : ℕ) (time_s : ℕ) (length_bridge_m : ℕ) (length_train_m : ℕ) :
  speed_kmh = 45 → time_s = 30 → length_bridge_m = 275 → length_train_m = 475 :=
by
  intros h1 h2 h3
  sorry

end length_of_train_l54_54728


namespace measure_of_angle_l54_54530

theorem measure_of_angle (x : ℝ) (h1 : 90 - x = 3 * x - 10) : x = 25 :=
by
  sorry

end measure_of_angle_l54_54530


namespace partI_solution_partII_solution_l54_54137

-- Part (I)
theorem partI_solution (x : ℝ) (a : ℝ) (h : a = 5) : (|x + a| + |x - 2| > 9) ↔ (x < -6 ∨ x > 3) :=
by
  sorry

-- Part (II)
theorem partII_solution (a : ℝ) :
  (∀ x : ℝ, (|2*x - 1| ≤ 3) → (|x + a| + |x - 2| ≤ |x - 4|)) → (-1 ≤ a ∧ a ≤ 0) :=
by
  sorry

end partI_solution_partII_solution_l54_54137


namespace quadratic_real_roots_and_value_l54_54925

theorem quadratic_real_roots_and_value (m x1 x2: ℝ) 
  (h1: ∀ (a: ℝ), ∃ (b c: ℝ), a = x^2 - 4 * x - 2 * m + 5) 
  (h2: x1 * x2 + x1 + x2 = m^2 + 6):
  m ≥ 1/2 ∧ m = 1 := 
by
  sorry

end quadratic_real_roots_and_value_l54_54925


namespace pharmacy_incurs_loss_l54_54566

variable (a b : ℝ)
variable (h : a < b)

theorem pharmacy_incurs_loss 
  (H : (41 * a + 59 * b) > 100 * (a + b) / 2) : true :=
by
  sorry

end pharmacy_incurs_loss_l54_54566


namespace larger_segment_length_l54_54943

open Real

theorem larger_segment_length (a b c : ℝ) (h : a = 50 ∧ b = 110 ∧ c = 120) :
  ∃ x : ℝ, x = 100 ∧ (∃ h : ℝ, a^2 = x^2 + h^2 ∧ b^2 = (c - x)^2 + h^2) :=
by
  sorry

end larger_segment_length_l54_54943


namespace steven_owes_jeremy_l54_54946

theorem steven_owes_jeremy (rate_per_room rooms_cleaned : ℚ) (h1 : rate_per_room = 11 / 2) (h2 : rooms_cleaned = 7 / 3) :
    (rate_per_room * rooms_cleaned) = 77 / 6 :=
by
  rw [h1, h2]
  norm_num

end steven_owes_jeremy_l54_54946


namespace remainder_101_mul_103_mod_11_l54_54405

theorem remainder_101_mul_103_mod_11 : (101 * 103) % 11 = 8 :=
by
  sorry

end remainder_101_mul_103_mod_11_l54_54405


namespace count_isosceles_numbers_correct_l54_54350

open Finset

def count_isosceles_numbers : ℕ :=
  let digits := range 1 10
  let equilateral_count := digits.card
  let isosceles_count :=
    (digits.product digits).filter (λ ab : ℕ × ℕ,
      let (a, b) := ab in a ≠ b ∧ 
      let aa := {a, a, b}
      ab.2 > 0 ∧ (2 * a > b)).card * 3
  equilateral_count + isosceles_count - 20

theorem count_isosceles_numbers_correct :
  count_isosceles_numbers = 165 :=
by
  sorry

end count_isosceles_numbers_correct_l54_54350


namespace least_possible_value_of_smallest_integer_l54_54701

theorem least_possible_value_of_smallest_integer 
  (A B C D : ℤ) 
  (H_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (H_avg : (A + B + C + D) / 4 = 74)
  (H_max : D = 90) :
  A ≥ 31 :=
by sorry

end least_possible_value_of_smallest_integer_l54_54701


namespace evaluate_ceil_of_neg_sqrt_l54_54110

-- Define the given expression and its value computation
def given_expression : ℚ := -real.sqrt (64 / 9)

-- Define the expected answer
def expected_answer : ℤ := -2

-- State the theorem to be proven
theorem evaluate_ceil_of_neg_sqrt : (Int.ceil given_expression) = expected_answer := sorry

end evaluate_ceil_of_neg_sqrt_l54_54110


namespace triangle_DEF_rotate_180_D_l54_54665

def rotate_180_degrees_clockwise (E D : (ℝ × ℝ)) : (ℝ × ℝ) :=
  let ED := (D.1 - E.1, D.2 - E.2)
  (E.1 - ED.1, E.2 - ED.2)

theorem triangle_DEF_rotate_180_D (D E F : (ℝ × ℝ))
  (hD : D = (3, 2)) (hE : E = (6, 5)) (hF : F = (6, 2)) :
  rotate_180_degrees_clockwise E D = (9, 8) :=
by
  rw [hD, hE, rotate_180_degrees_clockwise]
  sorry

end triangle_DEF_rotate_180_D_l54_54665


namespace diagonal_BD_eq_diagonal_AD_eq_l54_54809

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-1, 2⟩
def C : Point := ⟨5, 4⟩
def line_AB (p : Point) : Prop := p.x - p.y + 3 = 0

theorem diagonal_BD_eq :
  (∃ M : Point, M = ⟨2, 3⟩) →
  (∃ BD : Point → Prop, (BD = fun p => 3*p.x + p.y - 9 = 0)) :=
by
  sorry

theorem diagonal_AD_eq :
  (∃ M : Point, M = ⟨2, 3⟩) →
  (∃ AD : Point → Prop, (AD = fun p => p.x + 7*p.y - 13 = 0)) :=
by
  sorry

end diagonal_BD_eq_diagonal_AD_eq_l54_54809


namespace gcd_of_three_numbers_l54_54981

theorem gcd_of_three_numbers (a b c d : ℕ) (ha : a = 72) (hb : b = 120) (hc : c = 168) (hd : d = 24) : 
  Nat.gcd (Nat.gcd a b) c = d :=
by
  rw [ha, hb, hc, hd]
  -- Placeholder for the actual proof
  exact sorry

end gcd_of_three_numbers_l54_54981


namespace num_players_in_chess_tournament_l54_54150

theorem num_players_in_chess_tournament (p1 points : Nat) (h1 : points = 1979 ∨ points = 1980 ∨ points = 1984 ∨ points = 1985)
    (h2 : ∃ (n : ℕ), 2*n*(n-1)=1984 ∨ 2*n*(n-1)=1980): (p1 = 45) :=
begin
  sorry
end

end num_players_in_chess_tournament_l54_54150


namespace base6_subtraction_proof_l54_54752

-- Define the operations needed
def base6_add (a b : Nat) : Nat := sorry
def base6_subtract (a b : Nat) : Nat := sorry

axiom base6_add_correct : ∀ (a b : Nat), base6_add a b = (a + b)
axiom base6_subtract_correct : ∀ (a b : Nat), base6_subtract a b = (if a ≥ b then a - b else 0)

-- Define the problem conditions in base 6
def a := 5*6^2 + 5*6^1 + 5*6^0
def b := 5*6^1 + 5*6^0
def c := 2*6^2 + 0*6^1 + 2*6^0

-- Define the expected result
def result := 6*6^2 + 1*6^1 + 4*6^0

-- State the proof problem
theorem base6_subtraction_proof : base6_subtract (base6_add a b) c = result :=
by
  rw [base6_add_correct, base6_subtract_correct]
  sorry

end base6_subtraction_proof_l54_54752


namespace diff_of_squares_example_l54_54412

theorem diff_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end diff_of_squares_example_l54_54412


namespace cost_price_proof_l54_54700

noncomputable def selling_price : Real := 12000
noncomputable def discount_rate : Real := 0.10
noncomputable def new_selling_price : Real := selling_price * (1 - discount_rate)
noncomputable def profit_rate : Real := 0.08

noncomputable def cost_price : Real := new_selling_price / (1 + profit_rate)

theorem cost_price_proof : cost_price = 10000 := by sorry

end cost_price_proof_l54_54700


namespace sum_alternating_binomial_l54_54907

-- Define the problem
theorem sum_alternating_binomial :
  (∑ k in Finset.range 51, (-1:ℤ)^k * Nat.choose 100 (2 * k)) = 2^50 := sorry

end sum_alternating_binomial_l54_54907


namespace ratio_a7_b7_l54_54772

-- Definitions of the conditions provided in the problem
variables {a b : ℕ → ℝ}   -- Arithmetic sequences {a_n} and {b_n}
variables {S T : ℕ → ℝ}   -- Sums of the first n terms of {a_n} and {b_n}

-- Condition: For any positive integer n, S_n / T_n = (3n + 5) / (2n + 3)
axiom condition_S_T (n : ℕ) (hn : 0 < n) : S n / T n = (3 * n + 5) / (2 * n + 3)

-- Goal: Prove that a_7 / b_7 = 44 / 29
theorem ratio_a7_b7 : a 7 / b 7 = 44 / 29 := 
sorry

end ratio_a7_b7_l54_54772


namespace not_equivalent_to_0_0000375_l54_54877

theorem not_equivalent_to_0_0000375 : 
    ¬ (3 / 8000000 = 3.75 * 10 ^ (-5)) :=
by sorry

end not_equivalent_to_0_0000375_l54_54877


namespace smallest_number_of_marbles_l54_54237

theorem smallest_number_of_marbles :
  ∃ N : ℕ, N > 1 ∧ (N % 9 = 1) ∧ (N % 10 = 1) ∧ (N % 11 = 1) ∧ (∀ m : ℕ, m > 1 ∧ (m % 9 = 1) ∧ (m % 10 = 1) ∧ (m % 11 = 1) → N ≤ m) :=
sorry

end smallest_number_of_marbles_l54_54237


namespace remainder_29_169_1990_mod_11_l54_54235

theorem remainder_29_169_1990_mod_11 :
  (29 * 169 ^ 1990) % 11 = 7 :=
by
  sorry

end remainder_29_169_1990_mod_11_l54_54235


namespace sum_of_possible_values_l54_54962

theorem sum_of_possible_values (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := 
sorry

end sum_of_possible_values_l54_54962


namespace race_participants_minimum_l54_54178

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l54_54178


namespace fermats_little_theorem_l54_54836

theorem fermats_little_theorem (p : ℕ) (a : ℤ) (hp : Nat.Prime p) : 
  (a^p - a) % p = 0 :=
sorry

end fermats_little_theorem_l54_54836


namespace find_q_of_quadratic_with_roots_ratio_l54_54385

theorem find_q_of_quadratic_with_roots_ratio {q : ℝ} :
  (∃ r1 r2 : ℝ, r1 ≠ 0 ∧ r2 ≠ 0 ∧ r1 / r2 = 3 / 1 ∧ r1 + r2 = -10 ∧ r1 * r2 = q) →
  q = 18.75 :=
by
  sorry

end find_q_of_quadratic_with_roots_ratio_l54_54385


namespace math_problem_l54_54002

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l54_54002


namespace slope_of_intersection_points_l54_54125

theorem slope_of_intersection_points :
  ∀ s : ℝ, ∃ k b : ℝ, (∀ (x y : ℝ), (2 * x - 3 * y = 4 * s + 6) ∧ (2 * x + y = 3 * s + 1) → y = k * x + b) ∧ k = -2/13 := 
by
  intros s
  -- Proof to be provided here
  sorry

end slope_of_intersection_points_l54_54125


namespace eldest_boy_age_l54_54529

theorem eldest_boy_age (a b c : ℕ) (h1 : a + b + c = 45) (h2 : 3 * c = 7 * a) (h3 : 5 * c = 7 * b) : c = 21 := 
sorry

end eldest_boy_age_l54_54529


namespace red_blue_tile_difference_is_15_l54_54831

def num_blue_tiles : ℕ := 17
def num_red_tiles_initial : ℕ := 8
def additional_red_tiles : ℕ := 24
def num_red_tiles_new : ℕ := num_red_tiles_initial + additional_red_tiles
def tile_difference : ℕ := num_red_tiles_new - num_blue_tiles

theorem red_blue_tile_difference_is_15 : tile_difference = 15 :=
by
  sorry

end red_blue_tile_difference_is_15_l54_54831


namespace total_fish_is_22_l54_54361

def gold_fish : ℕ := 15
def blue_fish : ℕ := 7
def total_fish : ℕ := gold_fish + blue_fish

theorem total_fish_is_22 : total_fish = 22 :=
by
  -- the proof should be written here
  sorry

end total_fish_is_22_l54_54361


namespace boat_b_takes_less_time_l54_54684

theorem boat_b_takes_less_time (A_speed_still : ℝ) (B_speed_still : ℝ)
  (A_current : ℝ) (B_current : ℝ) (distance_downstream : ℝ)
  (A_speed_downstream : A_speed_still + A_current = 26)
  (B_speed_downstream : B_speed_still + B_current = 28)
  (A_time : A_speed_still + A_current = 26 → distance_downstream / (A_speed_still + A_current) = 4.6154)
  (B_time : B_speed_still + B_current = 28 → distance_downstream / (B_speed_still + B_current) = 4.2857) :
  distance_downstream / (B_speed_still + B_current) < distance_downstream / (A_speed_still + A_current) :=
by sorry

end boat_b_takes_less_time_l54_54684


namespace parabola_vertex_on_x_axis_l54_54940

theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + c = 0) → c = 1 := by
  sorry

end parabola_vertex_on_x_axis_l54_54940


namespace total_memory_space_l54_54819

def morning_songs : Nat := 10
def afternoon_songs : Nat := 15
def night_songs : Nat := 3
def song_size : Nat := 5

theorem total_memory_space : (morning_songs + afternoon_songs + night_songs) * song_size = 140 := by
  sorry

end total_memory_space_l54_54819


namespace integer_values_b_l54_54069

theorem integer_values_b (h : ∃ b : ℤ, ∀ x : ℤ, (x^2 + b * x + 5 ≤ 0) → x ∈ {x | true}):
  {b : ℤ | ∃! x : ℤ, x^2 + b * x + 5 ≤ 0}.size = 2 :=
sorry

end integer_values_b_l54_54069


namespace counting_unit_of_0_75_l54_54535

def decimal_places (n : ℝ) : ℕ := 
  by sorry  -- Assume this function correctly calculates the number of decimal places of n

def counting_unit (n : ℝ) : ℝ :=
  by sorry  -- Assume this function correctly determines the counting unit based on decimal places

theorem counting_unit_of_0_75 : counting_unit 0.75 = 0.01 :=
  by sorry


end counting_unit_of_0_75_l54_54535


namespace tv_price_comparison_l54_54997

def area (width height : ℕ) : ℕ := width * height

def cost_per_square_inch (cost area : ℕ) : ℚ := cost.toRat / area.toRat

def price_difference (cost1 cost2 area1 area2 : ℕ) : ℚ :=
  cost_per_square_inch(cost1, area1) - cost_per_square_inch(cost2, area2)

theorem tv_price_comparison :
  price_difference 672 1152 (area 24 16) (area 48 32) = 1 := by
  sorry

end tv_price_comparison_l54_54997


namespace cylinder_height_relationship_l54_54866

variable (r₁ h₁ r₂ h₂ : ℝ)

def radius_relationship := r₂ = 1.1 * r₁

def volume_equal := π * r₁^2 * h₁ = π * r₂^2 * h₂

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) 
     (h_radius : radius_relationship r₁ r₂) 
     (h_volume : volume_equal r₁ h₁ r₂ h₂) : 
     h₁ = 1.21 * h₂ :=
by
  unfold radius_relationship at h_radius
  unfold volume_equal at h_volume
  sorry

end cylinder_height_relationship_l54_54866


namespace probability_correct_l54_54127

open Finset

variables (s : Finset ℕ) (n m : ℕ)

-- Define the set of numbers
def set_of_numbers : Finset ℕ := {1, 2, 3, 4}

-- Condition: Two numbers are drawn without replacement
def possible_outcomes (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s \ s.diag

-- Calculate the number of total outcomes
def total_outcomes : ℕ := (possible_outcomes (set_of_numbers)).card

-- Condition: The two numbers should both be even
def even_numbers : Finset ℕ := {2, 4}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (possible_outcomes (even_numbers)).filter (λ (p : ℕ × ℕ), p.1 ∈ even_numbers ∧ p.2 ∈ even_numbers)

def number_of_favorable_outcomes : ℕ := favorable_outcomes.card

-- Final probability calculation
def probability_of_drawing_two_even_numbers := (number_of_favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_correct : probability_of_drawing_two_even_numbers == (1 : ℚ) / (6 : ℚ) :=
  by sorry

end probability_correct_l54_54127


namespace width_of_property_l54_54733

theorem width_of_property (W : ℝ) 
  (h1 : ∃ w l, (w = W / 8) ∧ (l = 2250 / 10) ∧ (w * l = 28125)) : W = 1000 :=
by
  -- Formal proof here
  sorry

end width_of_property_l54_54733


namespace total_lambs_l54_54829

-- Defining constants
def Merry_lambs : ℕ := 10
def Brother_lambs : ℕ := Merry_lambs + 3

-- Proving the total number of lambs
theorem total_lambs : Merry_lambs + Brother_lambs = 23 :=
  by
    -- The actual proof is omitted and a placeholder is put instead
    sorry

end total_lambs_l54_54829


namespace number_of_wins_in_first_9_matches_highest_possible_points_minimum_wins_in_remaining_matches_l54_54949

-- Define the conditions
def total_matches := 16
def played_matches := 9
def lost_matches := 2
def current_points := 19
def max_points_per_win := 3
def draw_points := 1
def remaining_matches := total_matches - played_matches
def required_points := 34

-- Statements to prove
theorem number_of_wins_in_first_9_matches :
  ∃ wins_in_first_9, 3 * wins_in_first_9 + draw_points * (played_matches - lost_matches - wins_in_first_9) = current_points :=
sorry

theorem highest_possible_points :
  current_points + remaining_matches * max_points_per_win = 40 :=
sorry

theorem minimum_wins_in_remaining_matches :
  ∃ min_wins_in_remaining_7, (min_wins_in_remaining_7 = 4 ∧ 3 * min_wins_in_remaining_7 + current_points + (remaining_matches - min_wins_in_remaining_7) * draw_points ≥ required_points) :=
sorry

end number_of_wins_in_first_9_matches_highest_possible_points_minimum_wins_in_remaining_matches_l54_54949


namespace jesse_carpet_problem_l54_54947

noncomputable def area_rectangle (length width : ℝ) : ℝ :=
  length * width

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

noncomputable def area_circle (radius : ℝ) : ℝ :=
  Real.pi * radius^2

noncomputable def total_area (rect_area tri_area circ_area carpet_bought : ℝ) : ℝ :=
  rect_area + tri_area + circ_area - carpet_bought

noncomputable def carpet_needed (total_area : ℝ) : ℝ :=
  total_area

noncomputable def max_area (budget price_per_sqft : ℝ) : ℝ :=
  budget / price_per_sqft

theorem jesse_carpet_problem :
  let rect_area := area_rectangle 11 15 in
  let tri_area := area_triangle 12 8 in
  let circ_area := area_circle 6 in
  let initial_carpet := 16 in
  let budget := 800 in
  let total_needed := total_area rect_area tri_area circ_area initial_carpet in
  let needed_carpet := carpet_needed total_needed in
  let max_regular := max_area budget 5 in
  let max_deluxe := max_area budget 7.5 in
  let max_luxury := max_area budget 10 in
  (needed_carpet ≈ 310.097) ∧ (max_regular < 310.097) ∧ (max_deluxe < 310.097) ∧ (max_luxury < 310.097) :=
sorry

end jesse_carpet_problem_l54_54947


namespace parabola_vertex_coordinates_l54_54534

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = 3 * (x - 7)^2 + 5 → (7, 5) = (7, 5) :=
by
  intros x y h
  exact rfl

end parabola_vertex_coordinates_l54_54534


namespace race_participants_minimum_l54_54198

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l54_54198


namespace pentagon_AEDCB_area_l54_54571

-- Definitions based on the given conditions
def rectangle_ABCD (AB BC : ℕ) : Prop :=
AB = 12 ∧ BC = 10

def triangle_ADE (AE ED : ℕ) : Prop :=
AE = 9 ∧ ED = 6 ∧ AE * ED ≠ 0 ∧ (AE^2 + ED^2 = (AE^2 + ED^2))

def area_of_rectangle (AB BC : ℕ) : ℕ :=
AB * BC

def area_of_triangle (AE ED : ℕ) : ℕ :=
(AE * ED) / 2

-- The theorem to be proved
theorem pentagon_AEDCB_area (AB BC AE ED : ℕ) (h_rect : rectangle_ABCD AB BC) (h_tri : triangle_ADE AE ED) :
  area_of_rectangle AB BC - area_of_triangle AE ED = 93 :=
sorry

end pentagon_AEDCB_area_l54_54571


namespace problem_conditions_and_inequalities_l54_54764

open Real

theorem problem_conditions_and_inequalities (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + 2 * b = a * b) :
  (a + 2 * b ≥ 8) ∧ (2 * a + b ≥ 9) ∧ (a ^ 2 + 4 * b ^ 2 + 5 * a * b ≥ 72) ∧ ¬(logb 2 a + logb 2 b < 3) :=
by
  sorry

end problem_conditions_and_inequalities_l54_54764


namespace race_participants_minimum_l54_54179

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l54_54179


namespace evaluate_expression_l54_54887

theorem evaluate_expression : 
    8 * 7 / 8 * 7 = 49 := 
by sorry

end evaluate_expression_l54_54887


namespace perp_bisector_of_AB_l54_54676

noncomputable def perpendicular_bisector_eq : Prop :=
  ∀ (x y : ℝ), (x - y + 1 = 0) ∧ (x^2 + y^2 = 1) → (x + y = 0)

-- The proof is omitted
theorem perp_bisector_of_AB : perpendicular_bisector_eq :=
sorry

end perp_bisector_of_AB_l54_54676


namespace total_eggs_l54_54987

noncomputable def total_eggs_in_all_containers (n : ℕ) (f l : ℕ) : ℕ :=
  n * (f * l)

theorem total_eggs (f l : ℕ) :
  (f = 14 + 20 - 1) →
  (l = 3 + 2 - 1) →
  total_eggs_in_all_containers 28 f l = 3696 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end total_eggs_l54_54987


namespace only_element_in_intersection_l54_54736

theorem only_element_in_intersection :
  ∃! (n : ℕ), n = 2500 ∧ ∃ (r : ℚ), r ≠ 2 ∧ r ≠ -2 ∧ 404 / (r^2 - 4) = n := sorry

end only_element_in_intersection_l54_54736


namespace arithmetic_sequence_product_l54_54508

theorem arithmetic_sequence_product (a : ℕ → ℤ) (d : ℤ) (h_inc : ∀ n m, n < m → a n < a m) 
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_prod : a 4 * a 5 = 12) : a 2 * a 7 = 6 :=
sorry

end arithmetic_sequence_product_l54_54508


namespace calculate_expression_l54_54103

theorem calculate_expression (y : ℝ) : (20 * y^3) * (7 * y^2) * (1 / (2 * y)^3) = 17.5 * y^2 :=
by
  sorry

end calculate_expression_l54_54103


namespace condition_iff_inequality_l54_54918

theorem condition_iff_inequality (a b : ℝ) (h : a * b ≠ 0) : (0 < a ∧ 0 < b) ↔ ((a + b) / 2 ≥ Real.sqrt (a * b)) :=
by
  -- Proof goes here
  sorry 

end condition_iff_inequality_l54_54918


namespace tan_ratio_l54_54958

theorem tan_ratio (p q : Real) (hpq1 : Real.sin (p + q) = 0.6) (hpq2 : Real.sin (p - q) = 0.3) : 
  Real.tan p / Real.tan q = 3 := 
by
  sorry

end tan_ratio_l54_54958


namespace ceil_neg_sqrt_64_over_9_l54_54117

theorem ceil_neg_sqrt_64_over_9 : Real.ceil (-Real.sqrt (64 / 9)) = -2 := 
by
  sorry

end ceil_neg_sqrt_64_over_9_l54_54117


namespace total_count_not_47_l54_54643

theorem total_count_not_47 (h c : ℕ) : 11 * h + 6 * c ≠ 47 := by
  sorry

end total_count_not_47_l54_54643


namespace correct_answers_count_l54_54727

theorem correct_answers_count (total_questions correct_pts incorrect_pts final_score : ℤ)
  (h1 : total_questions = 26)
  (h2 : correct_pts = 8)
  (h3 : incorrect_pts = -5)
  (h4 : final_score = 0) :
  ∃ c i : ℤ, c + i = total_questions ∧ correct_pts * c + incorrect_pts * i = final_score ∧ c = 10 :=
by
  use 10, (26 - 10)
  simp
  sorry

end correct_answers_count_l54_54727


namespace explicit_formula_l54_54922

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem explicit_formula (x1 x2 : ℝ) (h1 : x1 ∈ Set.Icc (-1 : ℝ) 1) (h2 : x2 ∈ Set.Icc (-1 : ℝ) 1) :
  f x = x^3 - 3 * x ∧ |f x1 - f x2| ≤ 4 :=
by
  sorry

end explicit_formula_l54_54922


namespace parabola_vertex_coordinates_l54_54531

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, (3 * (x - 7) ^ 2 + 5) = 3 * (x - 7) ^ 2 + 5 := by
  sorry

end parabola_vertex_coordinates_l54_54531


namespace find_y_l54_54984

theorem find_y (y : ℝ) : (∃ y : ℝ, (4, y) ≠ (2, -3) ∧ ((-3 - y) / (2 - 4) = 1)) → y = -1 :=
by
  sorry

end find_y_l54_54984


namespace camel_cost_l54_54085

theorem camel_cost :
  ∀ (C H O E : ℝ),
    (10 * C = 24 * H) →
    (16 * H = 4 * O) →
    (6 * O = 4 * E) →
    (10 * E = 120000) →
    C = 4800 :=
by
  intros C H O E h1 h2 h3 h4
  -- Allow the proof process to be skipped for now
  sorry

end camel_cost_l54_54085


namespace middle_letter_value_l54_54499

theorem middle_letter_value 
  (final_score : ℕ) 
  (first_letter_value : ℕ) 
  (third_letter_value : ℕ)
  (word_length : ℕ)
  (triple_score : ℕ)
  (total_points : ℕ)
  (middle_letter_value : ℕ)
  (h1 : final_score = 30)
  (h2 : first_letter_value = 1)
  (h3 : third_letter_value = 1)
  (h4 : word_length = 3)
  (h5 : triple_score = 3)
  (h6 : total_points = final_score / triple_score)
  (h7 : total_points = 10)
  (h8 : middle_letter_value = total_points - first_letter_value - third_letter_value) :
  middle_letter_value = 8 := 
by sorry

end middle_letter_value_l54_54499


namespace solve_equation_l54_54389

theorem solve_equation : ∀ x : ℝ, 4 * x - 2 * x + 1 - 3 = 0 → x = 1 :=
by
  intro x
  intro h
  sorry

end solve_equation_l54_54389


namespace tan_3theta_l54_54790

-- Let θ be an angle such that tan θ = 3.
variable (θ : ℝ)
noncomputable def tan_theta : ℝ := 3

-- Claim: tan(3 * θ) = 9/13
theorem tan_3theta :
  Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_3theta_l54_54790


namespace Emily_sixth_score_l54_54109

theorem Emily_sixth_score :
  let scores := [91, 94, 88, 90, 101]
  let current_sum := scores.sum
  let desired_average := 95
  let num_quizzes := 6
  let total_score_needed := num_quizzes * desired_average
  let sixth_score := total_score_needed - current_sum
  sixth_score = 106 :=
by
  sorry

end Emily_sixth_score_l54_54109


namespace perfect_square_trinomial_l54_54144

theorem perfect_square_trinomial (k x y : ℝ) :
  (∃ a b : ℝ, 9 * x^2 - k * x * y + 4 * y^2 = (a * x + b * y)^2) ↔ (k = 12 ∨ k = -12) :=
by
  sorry

end perfect_square_trinomial_l54_54144


namespace class_student_numbers_l54_54274

theorem class_student_numbers (a b c d : ℕ) 
    (h_avg : (a + b + c + d) / 4 = 46)
    (h_diff_ab : a - b = 4)
    (h_diff_bc : b - c = 3)
    (h_diff_cd : c - d = 2)
    (h_max_a : a > b ∧ a > c ∧ a > d) : 
    a = 51 ∧ b = 47 ∧ c = 44 ∧ d = 42 := 
by 
  sorry

end class_student_numbers_l54_54274


namespace log_comparison_l54_54824

/-- Assuming a = log base 3 of 2, b = natural log of 3, and c = log base 2 of 3,
    prove that c > b > a. -/
theorem log_comparison (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 3)
                                (h2 : b = Real.log 3)
                                (h3 : c = Real.log 3 / Real.log 2) :
  c > b ∧ b > a :=
by {
  sorry
}

end log_comparison_l54_54824


namespace probability_B_given_A_l54_54062

variables {Ω : Type*} {P : MeasureTheory.Measure Ω} [ProbabilityMeasure P]
variables (A B : Set Ω)

theorem probability_B_given_A (hA : P(A) = 0.80) (hB : P(B) = 0.60) :
  (Probability.cond A B) = 0.75 :=
by
  sorry

end probability_B_given_A_l54_54062


namespace calculate_expression_l54_54446

theorem calculate_expression : 
  (π - 2019)^0 + |real.sqrt 3 - 1| + (-1/2 : ℝ)^(-1) - 2 * real.tan (real.pi / 6) = -2 + real.sqrt 3 / 3 :=
by sorry

end calculate_expression_l54_54446


namespace tan_triple_angle_l54_54777

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l54_54777


namespace cole_drive_to_work_time_l54_54242

variables (D : ℝ) (T_work T_home : ℝ)

def speed_work : ℝ := 75
def speed_home : ℝ := 105
def total_time : ℝ := 2

theorem cole_drive_to_work_time :
  (T_work = D / speed_work) ∧
  (T_home = D / speed_home) ∧
  (T_work + T_home = total_time) →
  T_work * 60 = 70 :=
by
  sorry

end cole_drive_to_work_time_l54_54242


namespace largest_digit_M_divisible_by_6_l54_54403

theorem largest_digit_M_divisible_by_6 (M : ℕ) (h1 : 5172 * 10 + M % 2 = 0) (h2 : (5 + 1 + 7 + 2 + M) % 3 = 0) : M = 6 := by
  sorry

end largest_digit_M_divisible_by_6_l54_54403


namespace day_of_50th_day_l54_54495

theorem day_of_50th_day (days_250_N days_150_N1 : ℕ) 
  (h₁ : days_250_N % 7 = 5) (h₂ : days_150_N1 % 7 = 5) : 
  ((50 + 315 - 150 + 365 * 2) % 7) = 4 := 
  sorry

end day_of_50th_day_l54_54495


namespace tan3theta_l54_54800

theorem tan3theta (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 := 
by
  sorry

end tan3theta_l54_54800


namespace red_flowers_count_l54_54828

-- Let's define the given conditions
def total_flowers : ℕ := 10
def white_flowers : ℕ := 2
def blue_percentage : ℕ := 40

-- Calculate the number of blue flowers
def blue_flowers : ℕ := (blue_percentage * total_flowers) / 100

-- The property we want to prove is the number of red flowers
theorem red_flowers_count :
  total_flowers - (blue_flowers + white_flowers) = 4 :=
by
  sorry

end red_flowers_count_l54_54828


namespace range_for_m_l54_54927

def A := { x : ℝ | x^2 - 3 * x - 10 < 0 }
def B (m : ℝ) := { x : ℝ | m + 1 < x ∧ x < 1 - 3 * m }

theorem range_for_m (m : ℝ) (h : ∀ x, x ∈ A ∪ B m ↔ x ∈ B m) : m ≤ -3 := sorry

end range_for_m_l54_54927


namespace room_length_l54_54849

def area_four_walls (L: ℕ) (w: ℕ) (h: ℕ) : ℕ :=
  2 * (L * h) + 2 * (w * h)

def area_door (d_w: ℕ) (d_h: ℕ) : ℕ :=
  d_w * d_h

def area_windows (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) : ℕ :=
  num_windows * (win_w * win_h)

def total_area_to_whitewash (L: ℕ) (w: ℕ) (h: ℕ) (d_w: ℕ) (d_h: ℕ) (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) : ℕ :=
  area_four_walls L w h - area_door d_w d_h - area_windows win_w win_h num_windows

theorem room_length (cost: ℕ) (rate: ℕ) (w: ℕ) (h: ℕ) (d_w: ℕ) (d_h: ℕ) (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) (L: ℕ) :
  cost = rate * total_area_to_whitewash L w h d_w d_h win_w win_h num_windows →
  L = 25 :=
by
  have h1 : total_area_to_whitewash 25 15 12 6 3 4 3 3 = 24 * 25 + 306 := sorry
  have h2 : rate * (24 * 25 + 306) = 5436 := sorry
  sorry

end room_length_l54_54849


namespace ones_mult_palindrome_l54_54720

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 
  digits = digits.reverse

def ones (k : ℕ) : ℕ := (10 ^ k - 1) / 9

theorem ones_mult_palindrome (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  is_palindrome (ones m * ones n) ↔ (m = n ∧ m ≤ 9 ∧ n ≤ 9) := 
sorry

end ones_mult_palindrome_l54_54720


namespace sum_of_uv_l54_54510

theorem sum_of_uv (u v : ℕ) (hu : 0 < u) (hv : 0 < v) (hv_lt_hu : v < u)
  (area_pent : 6 * u * v = 500) : u + v = 19 :=
by
  sorry

end sum_of_uv_l54_54510


namespace smallest_root_of_equation_l54_54902

theorem smallest_root_of_equation :
  let a := (x : ℝ) - 4 / 5
  let b := (x : ℝ) - 2 / 5
  let c := (x : ℝ) - 1 / 2
  (a^2 + a * b + c^2 = 0) → (x = 4 / 5 ∨ x = 14 / 15) ∧ (min (4 / 5) (14 / 15) = 14 / 15) :=
by
  sorry

end smallest_root_of_equation_l54_54902


namespace min_number_of_participants_l54_54181

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l54_54181


namespace hyperbola_asymptote_solution_l54_54935

theorem hyperbola_asymptote_solution (b : ℝ) (hb : b > 0)
  (h_asym : ∀ x y, (∀ y : ℝ, y = (1 / 2) * x ∨ y = - (1 / 2) * x) → (x^2 / 4 - y^2 / b^2 = 1)) :
  b = 1 :=
sorry

end hyperbola_asymptote_solution_l54_54935


namespace hemisphere_surface_area_ratio_l54_54722

theorem hemisphere_surface_area_ratio 
  (r : ℝ) (sphere_surface_area : ℝ) (hemisphere_surface_area : ℝ) 
  (eq1 : sphere_surface_area = 4 * π * r^2) 
  (eq2 : hemisphere_surface_area = 3 * π * r^2) : 
  hemisphere_surface_area / sphere_surface_area = 3 / 4 :=
by sorry

end hemisphere_surface_area_ratio_l54_54722


namespace arithmetic_sequence_a17_l54_54132

theorem arithmetic_sequence_a17 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : S 13 = 78)
  (h2 : a 7 + a 12 = 10)
  (h_sum : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 1 + (a 2 - a 1) / (2 - 1)))
  (h_term : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1) / (2 - 1)) :
  a 17 = 2 :=
by
  sorry

end arithmetic_sequence_a17_l54_54132


namespace find_common_ratio_l54_54300

-- Define the variables and constants involved.
variables (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)

-- Define the conditions of the problem.
def is_geometric_sequence := ∀ n, a (n + 1) = q * a n
def sum_of_first_n_terms := ∀ n, S n = a 0 * (1 - q^(n + 1)) / (1 - q)
def condition1 := a 5 = 4 * S 4 + 3
def condition2 := a 6 = 4 * S 5 + 3

-- The main statement that needs to be proved.
theorem find_common_ratio
  (h1: is_geometric_sequence a q)
  (h2: sum_of_first_n_terms a S q)
  (h3: condition1 a S)
  (h4: condition2 a S) : 
  q = 5 :=
sorry -- proof to be provided

end find_common_ratio_l54_54300


namespace problem_statement_l54_54613

theorem problem_statement (x : ℝ) (h1 : x = 3 ∨ x = -3) : 6 * x^2 + 4 * x - 2 * (x^2 - 1) - 2 * (2 * x + x^2) = 20 := 
by {
  sorry
}

end problem_statement_l54_54613


namespace total_earnings_l54_54732

-- Define the constants and conditions.
def regular_hourly_rate : ℕ := 5
def overtime_hourly_rate : ℕ := 6
def regular_hours_per_week : ℕ := 40
def first_week_hours : ℕ := 44
def second_week_hours : ℕ := 48

-- Define the proof problem in Lean 4.
theorem total_earnings : (regular_hours_per_week * 2 * regular_hourly_rate + 
                         ((first_week_hours - regular_hours_per_week) + 
                          (second_week_hours - regular_hours_per_week)) * overtime_hourly_rate) = 472 := 
by 
  exact sorry -- Detailed proof steps would go here.

end total_earnings_l54_54732


namespace find_k_l54_54915

variables {α : Type*} [CommRing α]

theorem find_k (a b c : α) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - 2 * a * b * c :=
by sorry

end find_k_l54_54915


namespace intersection_A_B_l54_54474

open Set

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def B : Set ℕ := {x | 2 ≤ x ∧ x < 6}

theorem intersection_A_B : A ∩ B = {2, 3, 4, 5} := by
  sorry

end intersection_A_B_l54_54474


namespace import_tax_excess_amount_l54_54251

theorem import_tax_excess_amount (X : ℝ)
  (total_value : ℝ) (tax_paid : ℝ)
  (tax_rate : ℝ) :
  total_value = 2610 → tax_paid = 112.70 → tax_rate = 0.07 → 0.07 * (2610 - X) = 112.70 → X = 1000 :=
by
  intros h1 h2 h3 h4
  sorry

end import_tax_excess_amount_l54_54251


namespace integer_product_zero_l54_54217

theorem integer_product_zero (a : ℤ) (x : Fin 13 → ℤ)
  (h : a = ∏ i, (1 + x i) ∧ a = ∏ i, (1 - x i)) :
  a * ∏ i, x i = 0 :=
sorry

end integer_product_zero_l54_54217


namespace total_students_registered_l54_54670

theorem total_students_registered (num_present_yesterday : ℕ) (num_absent_today : ℕ) 
  (percent_less : ℝ) (num_students : ℕ) 
  (h1 : num_present_yesterday = 70)
  (h2 : num_absent_today = 30)
  (h3 : percent_less = 0.1)
  (h4 : num_students = 156) :
  let twice_present_yesterday := 2 * num_present_yesterday in
  let reduction := percent_less * twice_present_yesterday in
  let num_present_today := twice_present_yesterday - reduction in
  num_students = num_present_today + num_absent_today :=
by
  -- Using the conditions provided to show the equivalence.
  sorry

end total_students_registered_l54_54670


namespace mixed_doubles_pairing_l54_54988

def num_ways_to_pair (men women : ℕ) (select_men select_women : ℕ) : ℕ :=
  (Nat.choose men select_men) * (Nat.choose women select_women) * 2

theorem mixed_doubles_pairing : num_ways_to_pair 5 4 2 2 = 120 := by
  sorry

end mixed_doubles_pairing_l54_54988


namespace foci_distance_of_hyperbola_l54_54749

theorem foci_distance_of_hyperbola : 
  let a_squared := 32
  let b_squared := 8
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  2 * c = 4 * Real.sqrt 10 :=
by
  -- Definitions based on conditions
  let a_squared := 32
  let b_squared := 8
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  
  -- Proof outline here (using sorry to skip proof details)
  sorry

end foci_distance_of_hyperbola_l54_54749


namespace numbers_from_five_threes_l54_54401

theorem numbers_from_five_threes :
  (∃ (a b c d e : ℤ), (3*a + 3*b + 3*c + 3*d + 3*e = 11 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 12 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 13 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 14 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 15) ) :=
by
  -- Proof provided by the problem statement steps, using:
  -- 11 = (33/3)
  -- 12 = 3 * 3 + 3 + 3 - 3
  -- 13 = 3 * 3 + 3 + 3/3
  -- 14 = (33 + 3 * 3) / 3
  -- 15 = 3 + 3 + 3 + 3 + 3
  sorry

end numbers_from_five_threes_l54_54401


namespace min_number_of_participants_l54_54182

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l54_54182


namespace solve_for_a_l54_54804

theorem solve_for_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 16 - 6 * a + a ^ 2) : 
  a = -5 + Real.sqrt 41 ∨ a = -5 - Real.sqrt 41 := by
  sorry

end solve_for_a_l54_54804


namespace find_k_range_l54_54619

noncomputable def f (k x : ℝ) : ℝ := (k * x + 1 / 3) * Real.exp x - x

theorem find_k_range : 
  (∃ (k : ℝ), ∀ (x : ℕ), x > 0 → (f k (x : ℝ) < 0 ↔ x = 1)) ↔
  (k ≥ 1 / (Real.exp 2) - 1 / 6 ∧ k < 1 / Real.exp 1 - 1 / 3) :=
sorry

end find_k_range_l54_54619


namespace tan_triple_angle_l54_54792

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l54_54792


namespace triplet_sum_not_zero_l54_54878

def sum_triplet (a b c : ℝ) : ℝ := a + b + c

theorem triplet_sum_not_zero :
  ¬ (sum_triplet 3 (-5) 2 = 0) ∧
  (sum_triplet (1/4) (1/4) (-1/2) = 0) ∧
  (sum_triplet 0.3 (-0.1) (-0.2) = 0) ∧
  (sum_triplet 0.5 (-0.3) (-0.2) = 0) ∧
  (sum_triplet (1/3) (-1/6) (-1/6) = 0) :=
by 
  sorry

end triplet_sum_not_zero_l54_54878


namespace problem_3000_mod_1001_l54_54834

theorem problem_3000_mod_1001 : (300 ^ 3000 - 1) % 1001 = 0 := 
by
  have h1: (300 ^ 3000) % 7 = 1 := sorry
  have h2: (300 ^ 3000) % 11 = 1 := sorry
  have h3: (300 ^ 3000) % 13 = 1 := sorry
  sorry

end problem_3000_mod_1001_l54_54834


namespace hyperbola_equation_l54_54339

noncomputable def hyperbola (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

def parabola_focus_same_as_hyperbola_focus (c : ℝ) : Prop :=
  ∃ x y : ℝ, y^2 = 4 * (10:ℝ).sqrt * x ∧ (c, 0) = ((10:ℝ).sqrt, 0)

def hyperbola_eccentricity (c a : ℝ) := (c / a) = (10:ℝ).sqrt / 3

theorem hyperbola_equation :
  ∃ a b : ℝ, (hyperbola a b) ∧
  (parabola_focus_same_as_hyperbola_focus ((10:ℝ).sqrt)) ∧
  (hyperbola_eccentricity ((10:ℝ).sqrt) a) ∧
  ((a = 3) ∧ (b = 1)) :=
sorry

end hyperbola_equation_l54_54339


namespace juan_faster_than_peter_l54_54652

theorem juan_faster_than_peter (J : ℝ) :
  (Peter_speed : ℝ) = 5.0 →
  (time : ℝ) = 1.5 →
  (distance_apart : ℝ) = 19.5 →
  (J + 5.0) * time = distance_apart →
  J - 5.0 = 3 := 
by
  intros Peter_speed_eq time_eq distance_apart_eq relative_speed_eq
  sorry

end juan_faster_than_peter_l54_54652


namespace find_m_l54_54321

theorem find_m (m : ℕ) :
  (2022 ^ 2 - 4) * (2021 ^ 2 - 4) = 2024 * 2020 * 2019 * m → 
  m = 2023 :=
by
  sorry

end find_m_l54_54321


namespace unique_solution_for_star_l54_54280

def star (x y : ℝ) : ℝ := 4 * x - 5 * y + 2 * x * y

theorem unique_solution_for_star :
  ∃! y : ℝ, star 2 y = 5 :=
by
  -- We know the definition of star and we need to verify the condition.
  sorry

end unique_solution_for_star_l54_54280


namespace digit_sum_26_l54_54157

theorem digit_sum_26 
  (A B C D E : ℕ)
  (h1 : 1 ≤ A ∧ A ≤ 9)
  (h2 : 0 ≤ B ∧ B ≤ 9)
  (h3 : 0 ≤ C ∧ C ≤ 9)
  (h4 : 0 ≤ D ∧ D ≤ 9)
  (h5 : 0 ≤ E ∧ E ≤ 9)
  (h6 : 100000 + 10000 * A + 1000 * B + 100 * C + 10 * D + E * 3 = 100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + 1):
  A + B + C + D + E = 26 
  := 
  by
    sorry

end digit_sum_26_l54_54157


namespace total_cookies_needed_l54_54880

-- Define the conditions
def cookies_per_person : ℝ := 24.0
def number_of_people : ℝ := 6.0

-- Define the goal
theorem total_cookies_needed : cookies_per_person * number_of_people = 144.0 :=
by
  sorry

end total_cookies_needed_l54_54880


namespace total_polled_votes_proof_l54_54562

-- Define the conditions
variables (V : ℕ) -- total number of valid votes
variables (invalid_votes : ℕ) -- number of invalid votes
variables (total_polled_votes : ℕ) -- total polled votes
variables (candidateA_votes candidateB_votes : ℕ) -- votes for candidate A and B respectively

-- Assume the known conditions
variable (h1 : candidateA_votes = 45 * V / 100) -- candidate A got 45% of valid votes
variable (h2 : candidateB_votes = 55 * V / 100) -- candidate B got 55% of valid votes
variable (h3 : candidateB_votes - candidateA_votes = 9000) -- candidate A was defeated by 9000 votes
variable (h4 : invalid_votes = 83) -- there are 83 invalid votes
variable (h5 : total_polled_votes = V + invalid_votes) -- total polled votes is sum of valid and invalid votes

-- Define the theorem to prove
theorem total_polled_votes_proof : total_polled_votes = 90083 :=
by 
  -- Placeholder for the proof
  sorry

end total_polled_votes_proof_l54_54562


namespace find_Q_plus_R_l54_54488

-- P, Q, R must be digits in base 8 (distinct and non-zero)
def is_valid_digit (d : Nat) : Prop :=
  d > 0 ∧ d < 8

def digits_distinct (P Q R : Nat) : Prop :=
  P ≠ Q ∧ Q ≠ R ∧ R ≠ P

-- Define the base 8 number from its digits
def base8_number (P Q R : Nat) : Nat :=
  8^2 * P + 8 * Q + R

-- Define the given condition
def condition (P Q R : Nat) : Prop :=
  is_valid_digit P ∧ is_valid_digit Q ∧ is_valid_digit R ∧ digits_distinct P Q R ∧ 
  (base8_number P Q R + base8_number Q R P + base8_number R P Q = 8^3 * P + 8^2 * P + 8 * P + 8)

-- The result: Q + R in base 8 is 10_8 which is 8 + 2 (in decimal is 10)
theorem find_Q_plus_R (P Q R : Nat) (h : condition P Q R) : Q + R = 8 + 2 :=
sorry

end find_Q_plus_R_l54_54488


namespace melanie_missed_games_l54_54397

-- Define the total number of games and the number of games attended by Melanie
def total_games : ℕ := 7
def games_attended : ℕ := 3

-- Define the number of games missed as total games minus games attended
def games_missed : ℕ := total_games - games_attended

-- Theorem stating the number of games missed by Melanie
theorem melanie_missed_games : games_missed = 4 := by
  -- The proof is omitted
  sorry

end melanie_missed_games_l54_54397


namespace number_of_rectangles_on_3x3_grid_l54_54487

-- Define the grid and its properties
structure Grid3x3 where
  sides_are_2_units_apart : Bool
  diagonal_connections_allowed : Bool
  condition : sides_are_2_units_apart = true ∧ diagonal_connections_allowed = true

-- Define the number_rectangles function
def number_rectangles (g : Grid3x3) : Nat := 60

-- Define the theorem to prove the number of rectangles
theorem number_of_rectangles_on_3x3_grid : ∀ (g : Grid3x3), g.sides_are_2_units_apart = true ∧ g.diagonal_connections_allowed = true → number_rectangles g = 60 := by
  intro g
  intro h
  -- proof goes here
  sorry

end number_of_rectangles_on_3x3_grid_l54_54487


namespace find_x_orthogonal_l54_54460

theorem find_x_orthogonal :
  ∃ x : ℝ, (2 * x + 5 * (-3) = 0) ∧ x = 15 / 2 :=
by
  sorry

end find_x_orthogonal_l54_54460


namespace Tracy_sold_paintings_l54_54074

theorem Tracy_sold_paintings (num_people : ℕ) (group1_customers : ℕ) (group1_paintings : ℕ)
    (group2_customers : ℕ) (group2_paintings : ℕ) (group3_customers : ℕ) (group3_paintings : ℕ) 
    (total_paintings : ℕ) :
    num_people = 20 →
    group1_customers = 4 →
    group1_paintings = 2 →
    group2_customers = 12 →
    group2_paintings = 1 →
    group3_customers = 4 →
    group3_paintings = 4 →
    total_paintings = (group1_customers * group1_paintings) + (group2_customers * group2_paintings) + 
                      (group3_customers * group3_paintings) →
    total_paintings = 36 :=
by
  intros 
  -- including this to ensure the lean code passes syntax checks
  sorry

end Tracy_sold_paintings_l54_54074


namespace four_digit_even_and_multiple_of_7_sum_l54_54822

def num_four_digit_even_numbers : ℕ := 4500
def num_four_digit_multiples_of_7 : ℕ := 1286
def C : ℕ := num_four_digit_even_numbers
def D : ℕ := num_four_digit_multiples_of_7

theorem four_digit_even_and_multiple_of_7_sum :
  C + D = 5786 := by
  sorry

end four_digit_even_and_multiple_of_7_sum_l54_54822


namespace arithmetic_sequence_5th_term_l54_54337

theorem arithmetic_sequence_5th_term :
  let a1 := 3
  let d := 4
  a1 + 4 * (5 - 1) = 19 :=
by
  sorry

end arithmetic_sequence_5th_term_l54_54337


namespace find_positive_k_l54_54283

noncomputable def cubic_roots (a b k : ℝ) : Prop :=
  (3 * a * a * a + 9 * a * a - 135 * a + k = 0) ∧
  (a * a * b = -45 / 2)

theorem find_positive_k :
  ∃ (a b : ℝ), ∃ (k : ℝ) (pos : k > 0), (cubic_roots a b k) ∧ (k = 525) :=
by
  sorry

end find_positive_k_l54_54283


namespace wrapping_paper_area_correct_l54_54431

-- Given conditions:
variables (w h : ℝ) -- base length and height of the box

-- Definition of the area of the wrapping paper given the problem's conditions
def wrapping_paper_area (w h : ℝ) : ℝ :=
  2 * (w + h) ^ 2

-- Theorem statement to prove the area of the wrapping paper
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  -- proof to be provided
  sorry

end wrapping_paper_area_correct_l54_54431


namespace stamps_on_last_page_l54_54158

theorem stamps_on_last_page (total_books : ℕ) (pages_per_book : ℕ) (stamps_per_page_initial : ℕ) (stamps_per_page_new : ℕ)
    (full_books_new : ℕ) (pages_filled_seventh_book : ℕ) (total_stamps : ℕ) (stamps_in_seventh_book : ℕ) 
    (remaining_stamps : ℕ) :
    total_books = 10 →
    pages_per_book = 50 →
    stamps_per_page_initial = 8 →
    stamps_per_page_new = 12 →
    full_books_new = 6 →
    pages_filled_seventh_book = 37 →
    total_stamps = total_books * pages_per_book * stamps_per_page_initial →
    stamps_in_seventh_book = 4000 - (600 * full_books_new) →
    remaining_stamps = stamps_in_seventh_book - (pages_filled_seventh_book * stamps_per_page_new) →
    remaining_stamps = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end stamps_on_last_page_l54_54158


namespace Ahmad_eight_steps_l54_54574

def reach_top (n : Nat) (holes : List Nat) : Nat := sorry

theorem Ahmad_eight_steps (h : reach_top 8 [6] = 8) : True := by 
  trivial

end Ahmad_eight_steps_l54_54574


namespace stadium_length_l54_54123

theorem stadium_length
  (W : ℝ) (H : ℝ) (P : ℝ) (L : ℝ)
  (h1 : W = 18)
  (h2 : H = 16)
  (h3 : P = 34)
  (h4 : P^2 = L^2 + W^2 + H^2) :
  L = 24 :=
by
  sorry

end stadium_length_l54_54123


namespace value_expression_eq_zero_l54_54953

theorem value_expression_eq_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
    (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
    a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 :=
by
  sorry

end value_expression_eq_zero_l54_54953


namespace gcd_72_120_168_l54_54982

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  -- Each step would be proven individually here.
  sorry

end gcd_72_120_168_l54_54982


namespace total_lines_correct_l54_54590

-- Define the shapes and their corresponding lines
def triangles := 12
def squares := 8
def pentagons := 4
def hexagons := 6
def octagons := 2

def triangle_sides := 3
def square_sides := 4
def pentagon_sides := 5
def hexagon_sides := 6
def octagon_sides := 8

def lines_in_triangles := triangles * triangle_sides
def lines_in_squares := squares * square_sides
def lines_in_pentagons := pentagons * pentagon_sides
def lines_in_hexagons := hexagons * hexagon_sides
def lines_in_octagons := octagons * octagon_sides

def shared_lines_ts := 5
def shared_lines_ph := 3
def shared_lines_ho := 1

def total_lines_triangles := lines_in_triangles - shared_lines_ts
def total_lines_squares := lines_in_squares - shared_lines_ts
def total_lines_pentagons := lines_in_pentagons - shared_lines_ph
def total_lines_hexagons := lines_in_hexagons - shared_lines_ph - shared_lines_ho
def total_lines_octagons := lines_in_octagons - shared_lines_ho

-- The statement to prove
theorem total_lines_correct :
  total_lines_triangles = 31 ∧
  total_lines_squares = 27 ∧
  total_lines_pentagons = 17 ∧
  total_lines_hexagons = 32 ∧
  total_lines_octagons = 15 :=
by sorry

end total_lines_correct_l54_54590


namespace sum_of_three_rel_prime_pos_integers_l54_54864

theorem sum_of_three_rel_prime_pos_integers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h_rel_prime_ab : Nat.gcd a b = 1) (h_rel_prime_ac : Nat.gcd a c = 1) (h_rel_prime_bc : Nat.gcd b c = 1)
  (h_product : a * b * c = 2700) :
  a + b + c = 56 := by
  sorry

end sum_of_three_rel_prime_pos_integers_l54_54864


namespace Shinyoung_ate_most_of_cake_l54_54666

noncomputable def Shinyoung_portion := (1 : ℚ) / 3
noncomputable def Seokgi_portion := (1 : ℚ) / 4
noncomputable def Woong_portion := (1 : ℚ) / 5

theorem Shinyoung_ate_most_of_cake :
  Shinyoung_portion > Seokgi_portion ∧ Shinyoung_portion > Woong_portion := by
  sorry

end Shinyoung_ate_most_of_cake_l54_54666


namespace tan_3theta_l54_54786

-- Let θ be an angle such that tan θ = 3.
variable (θ : ℝ)
noncomputable def tan_theta : ℝ := 3

-- Claim: tan(3 * θ) = 9/13
theorem tan_3theta :
  Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_3theta_l54_54786


namespace solve_fraction_equation_l54_54805

theorem solve_fraction_equation (x : ℝ) (h : x ≠ 3) : (x^2 - 9) / (x - 3) = 0 ↔ x = -3 :=
sorry

end solve_fraction_equation_l54_54805


namespace sum_of_roots_l54_54017

theorem sum_of_roots (m n : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : ∀ x : ℝ, x^2 + m * x + n = 0 → (x = m ∨ x = n)) :
  m + n = -1 :=
sorry

end sum_of_roots_l54_54017


namespace find_angle_APB_l54_54641

-- Definitions based on conditions
def r1 := 2 -- Radius of semicircle SAR
def r2 := 3 -- Radius of semicircle RBT

def angle_AO1S := 70
def angle_BO2T := 40

def angle_AO1R := 180 - angle_AO1S
def angle_BO2R := 180 - angle_BO2T

def angle_PA := 90
def angle_PB := 90

-- Statement of the theorem
theorem find_angle_APB : angle_PA + angle_AO1R + angle_BO2R + angle_PB + 110 = 540 :=
by
  -- Unused in proof: added only to state theorem 
  have _ := angle_PA
  have _ := angle_AO1R
  have _ := angle_BO2R
  have _ := angle_PB
  have _ := 110
  sorry

end find_angle_APB_l54_54641


namespace stuart_initial_marbles_is_56_l54_54095

-- Define the initial conditions
def betty_initial_marbles : ℕ := 60
def percentage_given_to_stuart : ℚ := 40 / 100
def stuart_marbles_after_receiving : ℕ := 80

-- Define the calculation of how many marbles Betty gave to Stuart
def marbles_given_to_stuart := (percentage_given_to_stuart * betty_initial_marbles)

-- Define the target: Stuart's initial number of marbles
def stuart_initial_marbles := stuart_marbles_after_receiving - marbles_given_to_stuart

-- Main theorem stating the problem
theorem stuart_initial_marbles_is_56 : stuart_initial_marbles = 56 :=
by 
  sorry

end stuart_initial_marbles_is_56_l54_54095


namespace range_of_f_l54_54162

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) * (Real.cos x) + (Real.cos x) ^ 4

theorem range_of_f : Set.Icc 0 (9 / 8) = Set.range f := 
by
  sorry

end range_of_f_l54_54162


namespace eggs_in_each_basket_l54_54159

theorem eggs_in_each_basket :
  ∃ x : ℕ, x ∣ 30 ∧ x ∣ 42 ∧ x ≥ 5 ∧ x = 6 :=
by
  sorry

end eggs_in_each_basket_l54_54159


namespace base_addition_l54_54336

theorem base_addition (R1 R3 : ℕ) (F1 F2 : ℚ)
    (hF1_baseR1 : F1 = 45 / (R1^2 - 1))
    (hF2_baseR1 : F2 = 54 / (R1^2 - 1))
    (hF1_baseR3 : F1 = 36 / (R3^2 - 1))
    (hF2_baseR3 : F2 = 63 / (R3^2 - 1)) :
  R1 + R3 = 20 :=
sorry

end base_addition_l54_54336


namespace statement_a_statement_b_statement_c_statement_d_l54_54628

open Real

-- Statement A (incorrect)
theorem statement_a (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬ (a*c > b*d) := sorry

-- Statement B (correct)
theorem statement_b (a b : ℝ) (h1 : b < a) (h2 : a < 0) : (1 / a < 1 / b) := sorry

-- Statement C (incorrect)
theorem statement_c (a b : ℝ) (h : 1 / (a^2) < 1 / (b^2)) : ¬ (a > abs b) := sorry

-- Statement D (correct)
theorem statement_d (a b m : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : (a + m) / (b + m) > a / b := sorry

end statement_a_statement_b_statement_c_statement_d_l54_54628


namespace domain_of_function_l54_54910

def quadratic_inequality (x : ℝ) : Prop := -8 * x^2 - 14 * x + 9 ≥ 0

theorem domain_of_function :
  {x : ℝ | quadratic_inequality x} = {x : ℝ | x ≤ -1} ∪ {x : ℝ | x ≥ 9 / 8} :=
by
  -- The detailed proof would go here, but we're focusing on the statement structure.
  sorry

end domain_of_function_l54_54910


namespace graph_of_equation_is_two_lines_l54_54282

theorem graph_of_equation_is_two_lines :
  ∀ (x y : ℝ), (2 * x - y)^2 = 4 * x^2 - y^2 ↔ (y = 0 ∨ y = 2 * x) :=
by
  sorry

end graph_of_equation_is_two_lines_l54_54282


namespace min_colors_needed_is_3_l54_54870

noncomputable def min_colors_needed (S : Finset (Fin 7)) : Nat :=
  -- function to determine the minimum number of colors needed
  if ∀ (f : Finset (Fin 7) → Fin 3), ∀ (A B : Finset (Fin 7)), A.card = 3 ∧ B.card = 3 →
    A ∩ B = ∅ → f A ≠ f B then
    3
  else
    sorry

theorem min_colors_needed_is_3 :
  ∀ S : Finset (Fin 7), min_colors_needed S = 3 :=
by
  sorry

end min_colors_needed_is_3_l54_54870


namespace usual_time_to_school_l54_54078

variables (R T : ℝ)

theorem usual_time_to_school (h₁ : T > 0) (h₂ : R > 0) (h₃ : R / T = (5 / 4 * R) / (T - 4)) :
  T = 20 :=
by
  sorry

end usual_time_to_school_l54_54078


namespace fraction_sequence_calc_l54_54444

theorem fraction_sequence_calc : 
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) * (8 / 9) - 1 = -(7 / 9) := 
by 
  sorry

end fraction_sequence_calc_l54_54444


namespace average_score_all_test_takers_l54_54377

def avg (scores : List ℕ) : ℕ := scores.sum / scores.length

theorem average_score_all_test_takers (s_avg u_avg n : ℕ) 
  (H1 : s_avg = 42) (H2 : u_avg = 38) (H3 : n = 20) : avg ([s_avg * n, u_avg * n]) / (2 * n) = 40 := 
by sorry

end average_score_all_test_takers_l54_54377


namespace tan_triple_angle_l54_54793

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l54_54793


namespace percentage_of_other_sales_l54_54846

theorem percentage_of_other_sales :
  let pensPercentage := 20
  let pencilsPercentage := 15
  let notebooksPercentage := 30
  let totalPercentage := 100
  totalPercentage - (pensPercentage + pencilsPercentage + notebooksPercentage) = 35 :=
by
  sorry

end percentage_of_other_sales_l54_54846


namespace clarissa_copies_needed_l54_54585

-- Define the given conditions
def manuscript_pages : ℕ := 400
def cost_per_page : ℚ := 0.05
def cost_per_binding : ℚ := 5.00
def total_cost : ℚ := 250.00

-- Calculate the total cost for one manuscript
def cost_per_copy_and_bind : ℚ := cost_per_page * manuscript_pages + cost_per_binding

-- Define number of copies needed
def number_of_copies_needed : ℚ := total_cost / cost_per_copy_and_bind

-- Prove number of copies needed is 10
theorem clarissa_copies_needed : number_of_copies_needed = 10 := 
by 
  -- Implementing the proof steps would go here
  sorry

end clarissa_copies_needed_l54_54585


namespace vertex_of_parabola_l54_54674

theorem vertex_of_parabola :
  ∃ (x y : ℝ), (∀ x : ℝ, y = x^2 - 12 * x + 9) → (x, y) = (6, -27) :=
sorry

end vertex_of_parabola_l54_54674


namespace puppy_cost_l54_54813

variable (P : ℕ)

theorem puppy_cost (hc : 2 * 50 = 100) (hd : 3 * 100 = 300) (htotal : 2 * 50 + 3 * 100 + 2 * P = 700) : P = 150 :=
by
  sorry

end puppy_cost_l54_54813


namespace wage_difference_l54_54420

variable (P Q : ℝ)
variable (h : ℝ)
axiom wage_relation : P = 1.5 * Q
axiom time_relation : 360 = P * h
axiom time_relation_q : 360 = Q * (h + 10)

theorem wage_difference : P - Q = 6 :=
  by
  sorry

end wage_difference_l54_54420


namespace percent_of_x_is_y_minus_z_l54_54775

variable (x y z : ℝ)

axiom condition1 : 0.60 * (x - y) = 0.30 * (x + y + z)
axiom condition2 : 0.40 * (y - z) = 0.20 * (y + x - z)

theorem percent_of_x_is_y_minus_z :
  (y - z) = x := by
  sorry

end percent_of_x_is_y_minus_z_l54_54775


namespace men_in_second_group_l54_54424

theorem men_in_second_group (M : ℕ) (h1 : 16 * 30 = 480) (h2 : M * 24 = 480) : M = 20 :=
by
  sorry

end men_in_second_group_l54_54424


namespace find_fraction_l54_54507

theorem find_fraction (a b : ℝ) (h₁ : a ≠ b) (h₂ : a / b + (a + 6 * b) / (b + 6 * a) = 2) :
  a / b = 1 / 2 :=
sorry

end find_fraction_l54_54507


namespace trigonometric_relationship_l54_54128

-- Given conditions
variables (x : ℝ) (a b c : ℝ)

-- Required conditions
variables (h1 : π / 4 < x) (h2 : x < π / 2)
variables (ha : a = Real.sin x)
variables (hb : b = Real.cos x)
variables (hc : c = Real.tan x)

-- Proof goal
theorem trigonometric_relationship : b < a ∧ a < c :=
by
  -- Proof will go here
  sorry

end trigonometric_relationship_l54_54128


namespace rain_all_three_days_is_six_percent_l54_54985

-- Definitions based on conditions from step a)
def P_rain_friday : ℚ := 2 / 5
def P_rain_saturday : ℚ := 1 / 2
def P_rain_sunday : ℚ := 3 / 10

-- The probability it will rain on all three days
def P_rain_all_three_days : ℚ := P_rain_friday * P_rain_saturday * P_rain_sunday

-- The Lean 4 theorem statement
theorem rain_all_three_days_is_six_percent : P_rain_all_three_days * 100 = 6 := by
  sorry

end rain_all_three_days_is_six_percent_l54_54985


namespace find_y_l54_54630

theorem find_y {x y : ℤ} (h1 : x - y = 12) (h2 : x + y = 6) : y = -3 := 
by
  sorry

end find_y_l54_54630


namespace problem1_problem2_l54_54618

-- Define the function f(x)
def f (x m : ℝ) : ℝ := abs (x - m) - abs (x + 3 * m)

-- Condition that m must be greater than 0
variable {m : ℝ} (hm : m > 0)

-- First problem statement: When m=1, the solution set for f(x) ≥ 1 is x ≤ -3/2.
theorem problem1 (x : ℝ) (h : f x 1 ≥ 1) : x ≤ -3 / 2 :=
sorry

-- Second problem statement: The range of values for m such that f(x) < |2 + t| + |t - 1| holds for all x and t is 0 < m < 3/4.
theorem problem2 (m : ℝ) : (∀ (x t : ℝ), f x m < abs (2 + t) + abs (t - 1)) ↔ (0 < m ∧ m < 3 / 4) :=
sorry

end problem1_problem2_l54_54618


namespace factor_expression_l54_54105

theorem factor_expression (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = (a - b) * (b - c) * (c - a) * (a * b^2 + a * c^2) :=
by 
  sorry

end factor_expression_l54_54105


namespace ceil_neg_sqrt_64_over_9_l54_54116

theorem ceil_neg_sqrt_64_over_9 : Real.ceil (-Real.sqrt (64 / 9)) = -2 := 
by
  sorry

end ceil_neg_sqrt_64_over_9_l54_54116


namespace rectangle_perimeter_is_3y_l54_54723

noncomputable def congruent_rectangle_perimeter (y : ℝ) (h1 : y > 0) : ℝ :=
  let side_length := 2 * y
  let center_square_side := y
  let width := (side_length - center_square_side) / 2
  let length := center_square_side
  2 * (length + width)

theorem rectangle_perimeter_is_3y (y : ℝ) (h1 : y > 0) :
  congruent_rectangle_perimeter y h1 = 3 * y :=
sorry

end rectangle_perimeter_is_3y_l54_54723


namespace find_fibonacci_x_l54_54236

def is_fibonacci (a b c : ℕ) : Prop :=
  c = a + b

theorem find_fibonacci_x (a b x : ℕ)
  (h₁ : a = 8)
  (h₂ : b = 13)
  (h₃ : is_fibonacci a b x) :
  x = 21 :=
by
  sorry

end find_fibonacci_x_l54_54236


namespace race_participants_least_number_l54_54209

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l54_54209


namespace sum_consecutive_even_l54_54844

theorem sum_consecutive_even (m : ℤ) : m + (m + 2) + (m + 4) + (m + 6) + (m + 8) + (m + 10) = 6 * m + 30 :=
by
  sorry

end sum_consecutive_even_l54_54844


namespace find_a_l54_54622

def point_of_tangency (x0 y0 a : ℝ) : Prop :=
  (x0 - y0 - 1 = 0) ∧ (y0 = a * x0^2) ∧ (2 * a * x0 = 1)

theorem find_a (x0 y0 a : ℝ) (h : point_of_tangency x0 y0 a) : a = 1/4 :=
by
  sorry

end find_a_l54_54622


namespace banana_difference_l54_54250

theorem banana_difference (d : ℕ) :
  (8 + (8 + d) + (8 + 2 * d) + (8 + 3 * d) + (8 + 4 * d) = 100) →
  d = 6 :=
by
  sorry

end banana_difference_l54_54250


namespace price_of_movie_ticket_l54_54146

theorem price_of_movie_ticket
  (M F : ℝ)
  (h1 : 8 * M = 2 * F)
  (h2 : 8 * M + 5 * F = 840) :
  M = 30 :=
by
  sorry

end price_of_movie_ticket_l54_54146


namespace minimum_highways_l54_54046

open SimpleGraph

theorem minimum_highways (V : Finset ℕ) (hV : V.card = 10)
  (H : ∀ (u v w : V), u ≠ v → u ≠ w → v ≠ w →
   (G.adj u v ∧ G.adj v w ∧ G.adj w u) ∨ (¬G.adj u v ∧ ¬G.adj v w ∧ G.adj w u)): 
  ∃ (G : SimpleGraph V), G.edge_finset.card = 40 := by 
  sorry

end minimum_highways_l54_54046


namespace trajectory_of_midpoint_l54_54253

theorem trajectory_of_midpoint 
  (x y : ℝ)
  (P : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : (M.fst - 4)^2 + M.snd^2 = 16)
  (hP : P = (x, y))
  (h_mid : M = (2 * P.1 + 4, 2 * P.2 - 8)) :
  x^2 + (y - 4)^2 = 4 :=
by
  sorry

end trajectory_of_midpoint_l54_54253


namespace a_value_l54_54042

-- Definition of the operation
def star (x y : ℝ) : ℝ := x + y - x * y

-- Main theorem to prove
theorem a_value :
  let a := star 1 (star 0 1)
  a = 1 :=
by
  sorry

end a_value_l54_54042


namespace gcd_lcm_product_360_distinct_gcd_values_l54_54999

/-- 
  Given two integers a and b, such that the product of their gcd and lcm is 360,
  we need to prove that the number of distinct possible values for their gcd is 9.
--/
theorem gcd_lcm_product_360_distinct_gcd_values :
  ∀ (a b : ℕ), gcd a b * lcm a b = 360 → 
  (∃ gcd_values : Finset ℕ, gcd_values.card = 9 ∧ ∀ g, g ∈ gcd_values ↔ g = gcd a b) :=
by
  sorry

end gcd_lcm_product_360_distinct_gcd_values_l54_54999


namespace suitable_for_comprehensive_survey_l54_54876

-- Define the conditions
def is_comprehensive_survey (group_size : ℕ) (is_specific_group : Bool) : Bool :=
  is_specific_group ∧ (group_size < 100)  -- assuming "small" means fewer than 100 individuals/items

def is_sampling_survey (group_size : ℕ) (is_specific_group : Bool) : Bool :=
  ¬is_comprehensive_survey group_size is_specific_group

-- Define the surveys
def option_A (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_comprehensive_survey group_size is_specific_group

def option_B (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

def option_C (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

def option_D (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

-- Question: Which of the following surveys is suitable for a comprehensive survey given conditions
theorem suitable_for_comprehensive_survey :
  ∀ (group_size_A group_size_B group_size_C group_size_D : ℕ) 
    (is_specific_group_A is_specific_group_B is_specific_group_C is_specific_group_D : Bool),
  option_A group_size_A is_specific_group_A ↔ 
  ((option_B group_size_B is_specific_group_B = false) ∧ 
   (option_C group_size_C is_specific_group_C = false) ∧ 
   (option_D group_size_D is_specific_group_D = false)) :=
by
  sorry

end suitable_for_comprehensive_survey_l54_54876


namespace race_participants_minimum_l54_54199

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l54_54199


namespace removing_zeros_changes_value_l54_54587

noncomputable def a : ℝ := 7.0800
noncomputable def b : ℝ := 7.8

theorem removing_zeros_changes_value : a ≠ b :=
by
  -- proof goes here
  sorry

end removing_zeros_changes_value_l54_54587
