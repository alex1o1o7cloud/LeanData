import Mathlib

namespace average_weight_of_abc_l1129_112928

theorem average_weight_of_abc 
  (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 46)
  (h3 : B = 37) :
  (A + B + C) / 3 = 45 := 
by
  sorry

end average_weight_of_abc_l1129_112928


namespace newspaper_spending_over_8_weeks_l1129_112924

theorem newspaper_spending_over_8_weeks :
  (3 * 0.50 + 2.00) * 8 = 28 := by
  sorry

end newspaper_spending_over_8_weeks_l1129_112924


namespace problem_I_l1129_112959

theorem problem_I (x m : ℝ) (h1 : |x - m| < 1) (h2 : (1/3 : ℝ) < x ∧ x < (1/2 : ℝ)) : (-1/2 : ℝ) ≤ m ∧ m ≤ (4/3 : ℝ) :=
sorry

end problem_I_l1129_112959


namespace area_of_triangle_l1129_112919

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

end area_of_triangle_l1129_112919


namespace exists_k_undecided_l1129_112908

def tournament (n : ℕ) : Type :=
  { T : Fin n → Fin n → Prop // ∀ i j, T i j = ¬T j i }

def k_undecided (n k : ℕ) (T : tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k → ∃ (p : Fin n), ∀ (a : Fin n), a ∈ A → T.1 p a

theorem exists_k_undecided (k : ℕ) (hk : 0 < k) : ∃ (n : ℕ), n > k ∧ ∃ (T : tournament n), k_undecided n k T :=
by
  sorry

end exists_k_undecided_l1129_112908


namespace binary_addition_l1129_112957

theorem binary_addition (M : ℕ) (hM : M = 0b101110) :
  let M_plus_five := M + 5 
  let M_plus_five_binary := 0b110011
  let M_plus_five_predecessor := 0b110010
  M_plus_five = M_plus_five_binary ∧ M_plus_five - 1 = M_plus_five_predecessor :=
by
  sorry

end binary_addition_l1129_112957


namespace parabola_vertex_l1129_112916

noncomputable def is_vertex (x y : ℝ) : Prop :=
  y^2 + 8 * y + 4 * x + 5 = 0 ∧ (∀ y₀, y₀^2 + 8 * y₀ + 4 * x + 5 ≥ 0)

theorem parabola_vertex : is_vertex (11 / 4) (-4) :=
by
  sorry

end parabola_vertex_l1129_112916


namespace count_ball_box_arrangements_l1129_112966

theorem count_ball_box_arrangements :
  ∃ (arrangements : ℕ), arrangements = 20 ∧
  (∃ f : Fin 5 → Fin 5,
    (∃! i1, f i1 = i1) ∧ (∃! i2, f i2 = i2) ∧
    ∀ i, ∃! j, f i = j) :=
sorry

end count_ball_box_arrangements_l1129_112966


namespace complete_square_monomials_l1129_112930

theorem complete_square_monomials (x : ℝ) :
  ∃ (m : ℝ), (m = 4 * x ^ 4 ∨ m = 4 * x ∨ m = -4 * x ∨ m = -1 ∨ m = -4 * x ^ 2) ∧
              (∃ (a b : ℝ), (4 * x ^ 2 + 1 + m = a ^ 2 + b ^ 2)) :=
sorry

-- Note: The exact formulation of the problem might vary based on the definition
-- of perfect squares and corresponding polynomials in the Lean environment.

end complete_square_monomials_l1129_112930


namespace hex_B2F_to_dec_l1129_112902

theorem hex_B2F_to_dec : 
  let A := 10
  let B := 11
  let C := 12
  let D := 13
  let E := 14
  let F := 15
  let base := 16
  let b2f := B * base^2 + 2 * base^1 + F * base^0
  b2f = 2863 :=
by {
  sorry
}

end hex_B2F_to_dec_l1129_112902


namespace floor_sqrt_120_l1129_112979

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120)) = 10 := by
  have h1 : 10^2 < 120 := by norm_num
  have h2 : 120 < 11^2 := by norm_num
  -- Additional steps to show that Int.floor (Real.sqrt 120) = 10
  sorry

end floor_sqrt_120_l1129_112979


namespace p_evaluation_l1129_112972

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else 2 * x + 2 * y

theorem p_evaluation : p (p 3 (-4)) (p (-7) 0) = 40 := by
  sorry

end p_evaluation_l1129_112972


namespace nontrivial_solution_exists_l1129_112901

theorem nontrivial_solution_exists
  (a b c : ℝ) :
  (∃ x y z : ℝ, (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ 
    a * x + b * y + c * z = 0 ∧ 
    b * x + c * y + a * z = 0 ∧ 
    c * x + a * y + b * z = 0) ↔ (a + b + c = 0 ∨ a = b ∧ b = c) := 
sorry

end nontrivial_solution_exists_l1129_112901


namespace find_irrational_satisfying_conditions_l1129_112944

-- Define a real number x which is irrational
def is_irrational (x : ℝ) : Prop := ¬∃ (q : ℚ), (x : ℝ) = q

-- Define that x satisfies the given conditions
def rational_conditions (x : ℝ) : Prop :=
  (∃ (r1 : ℚ), x^3 - 17 * x = r1) ∧ (∃ (r2 : ℚ), x^2 + 4 * x = r2)

-- The main theorem statement
theorem find_irrational_satisfying_conditions (x : ℝ) 
  (hx_irr : is_irrational x) 
  (hx_cond : rational_conditions x) : x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5 :=
by
  sorry

end find_irrational_satisfying_conditions_l1129_112944


namespace couples_at_prom_l1129_112977

theorem couples_at_prom (total_students attending_alone attending_with_partners couples : ℕ) 
  (h1 : total_students = 123) 
  (h2 : attending_alone = 3) 
  (h3 : attending_with_partners = total_students - attending_alone) 
  (h4 : couples = attending_with_partners / 2) : 
  couples = 60 := 
by 
  sorry

end couples_at_prom_l1129_112977


namespace tan_315_deg_l1129_112989

theorem tan_315_deg : Real.tan (315 * Real.pi / 180) = -1 := sorry

end tan_315_deg_l1129_112989


namespace range_of_k_l1129_112973

theorem range_of_k (k n : ℝ) (h : k ≠ 0) (h_pass : k - n^2 - 2 = k / 2) : k ≥ 4 :=
sorry

end range_of_k_l1129_112973


namespace sum_of_gcd_and_lcm_l1129_112917

-- Definitions of gcd and lcm for the conditions
def gcd_of_42_and_56 : ℕ := Nat.gcd 42 56
def lcm_of_24_and_18 : ℕ := Nat.lcm 24 18

-- Lean statement that the sum of the gcd and lcm is 86
theorem sum_of_gcd_and_lcm : gcd_of_42_and_56 + lcm_of_24_and_18 = 86 := by
  sorry

end sum_of_gcd_and_lcm_l1129_112917


namespace powers_of_2_not_powers_of_4_l1129_112932

theorem powers_of_2_not_powers_of_4 (n : ℕ) (h1 : n < 500000) (h2 : ∃ k : ℕ, n = 2^k) (h3 : ∀ m : ℕ, n ≠ 4^m) : n = 9 := 
by
  sorry

end powers_of_2_not_powers_of_4_l1129_112932


namespace probability_in_interval_l1129_112974

theorem probability_in_interval (a b c d : ℝ) (h1 : a = 2) (h2 : b = 10) (h3 : c = 5) (h4 : d = 7) :
  (d - c) / (b - a) = 1 / 4 :=
by
  sorry

end probability_in_interval_l1129_112974


namespace inequality_count_l1129_112984

theorem inequality_count {a b : ℝ} (h : 1/a < 1/b ∧ 1/b < 0) :
  (if (|a| > |b|) then 0 else 1) + 
  (if (a + b > ab) then 1 else 0) +
  (if (a / b + b / a > 2) then 1 else 0) + 
  (if (a^2 / b < 2 * a - b) then 1 else 0) = 2 :=
sorry

end inequality_count_l1129_112984


namespace cosine_in_third_quadrant_l1129_112920

theorem cosine_in_third_quadrant (B : Real) 
  (h1 : Real.sin B = -5/13) 
  (h2 : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 := 
sorry

end cosine_in_third_quadrant_l1129_112920


namespace domain_of_g_l1129_112970

theorem domain_of_g (t : ℝ) : (t - 1)^2 + (t + 1)^2 + t ≠ 0 :=
  by
  sorry

end domain_of_g_l1129_112970


namespace maximize_profit_l1129_112990

-- Definitions
def initial_employees := 320
def profit_per_employee := 200000
def profit_increase_per_layoff := 20000
def expense_per_laid_off_employee := 60000
def min_employees := (3 * initial_employees) / 4
def profit_function (x : ℝ) := -0.2 * x^2 + 38 * x + 6400

-- The main statement
theorem maximize_profit : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 80 ∧ (∀ y : ℝ, 0 ≤ y ∧ y ≤ 80 → profit_function y ≤ profit_function x) ∧ x = 80 :=
by
  sorry

end maximize_profit_l1129_112990


namespace perimeter_pentagon_ABCD_l1129_112948

noncomputable def AB : ℝ := 2
noncomputable def BC : ℝ := Real.sqrt 8
noncomputable def CD : ℝ := Real.sqrt 18
noncomputable def DE : ℝ := Real.sqrt 32
noncomputable def AE : ℝ := Real.sqrt 62

theorem perimeter_pentagon_ABCD : 
  AB + BC + CD + DE + AE = 2 + 9 * Real.sqrt 2 + Real.sqrt 62 := by
  -- Note: The proof has been skipped as per instruction.
  sorry

end perimeter_pentagon_ABCD_l1129_112948


namespace identity_proof_l1129_112976

theorem identity_proof (a b : ℝ) : a^4 + b^4 + (a + b)^4 = 2 * (a^2 + a * b + b^2)^2 := 
sorry

end identity_proof_l1129_112976


namespace relationship_f_neg2_f_expr_l1129_112906

noncomputable def f : ℝ → ℝ := sorry  -- f is some function ℝ → ℝ, the exact definition is not provided

axiom even_function : ∀ x : ℝ, f (-x) = f x -- f is an even function
axiom increasing_on_negatives : ∀ x y : ℝ, x < y ∧ y < 0 → f x < f y -- f is increasing on (-∞, 0)

theorem relationship_f_neg2_f_expr (a : ℝ) : f (-2) ≥ f (a^2 - 4 * a + 6) := by
  -- proof omitted
  sorry

end relationship_f_neg2_f_expr_l1129_112906


namespace find_number_l1129_112992

theorem find_number (N : ℚ) (h : (5 / 6) * N = (5 / 16) * N + 150) : N = 288 := by
  sorry

end find_number_l1129_112992


namespace problem_statement_l1129_112904

-- Definitions for the conditions in the problem
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p
def has_three_divisors (k : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ k = p^2

-- Given conditions
def m : ℕ := 3 -- the smallest odd prime
def n : ℕ := 49 -- the largest integer less than 50 with exactly three positive divisors

-- The proof statement
theorem problem_statement : m + n = 52 :=
by sorry

end problem_statement_l1129_112904


namespace fraction_to_decimal_l1129_112965

theorem fraction_to_decimal : (53 : ℚ) / (4 * 5^7) = 1325 / 10^7 := sorry

end fraction_to_decimal_l1129_112965


namespace no_solution_for_system_l1129_112922

theorem no_solution_for_system (x y z : ℝ) 
  (h1 : |x| < |y - z|) 
  (h2 : |y| < |z - x|) 
  (h3 : |z| < |x - y|) : 
  false :=
sorry

end no_solution_for_system_l1129_112922


namespace simplified_expr_eval_l1129_112912

theorem simplified_expr_eval
  (x : ℚ) (y : ℚ) (h_x : x = -1/2) (h_y : y = 1) :
  (5*x^2 - 10*y^2) = -35/4 := 
by
  subst h_x
  subst h_y
  sorry

end simplified_expr_eval_l1129_112912


namespace find_a_l1129_112905

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53^2017 + a) % 13 = 0) : a = 12 :=
sorry

end find_a_l1129_112905


namespace gcd_consecutive_triplets_l1129_112911

theorem gcd_consecutive_triplets : ∀ i : ℕ, 1 ≤ i → gcd (i * (i + 1) * (i + 2)) 6 = 6 :=
by
  sorry

end gcd_consecutive_triplets_l1129_112911


namespace james_lifting_ratio_correct_l1129_112997

theorem james_lifting_ratio_correct :
  let lt_initial := 2200
  let bw_initial := 245
  let lt_gain_percentage := 0.15
  let bw_gain := 8
  let lt_final := lt_initial + lt_initial * lt_gain_percentage
  let bw_final := bw_initial + bw_gain
  (lt_final / bw_final) = 10 :=
by
  sorry

end james_lifting_ratio_correct_l1129_112997


namespace rita_bought_4_pounds_l1129_112991

variable (total_amount : ℝ) (cost_per_pound : ℝ) (amount_left : ℝ)

theorem rita_bought_4_pounds (h1 : total_amount = 70)
                             (h2 : cost_per_pound = 8.58)
                             (h3 : amount_left = 35.68) :
  (total_amount - amount_left) / cost_per_pound = 4 := 
  by
  sorry

end rita_bought_4_pounds_l1129_112991


namespace john_sales_percentage_l1129_112926

noncomputable def percentage_buyers (houses_visited_per_day : ℕ) (work_days_per_week : ℕ) (weekly_sales : ℝ) (low_price : ℝ) (high_price : ℝ) : ℝ :=
  let total_houses_per_week := houses_visited_per_day * work_days_per_week
  let average_sale_per_customer := (low_price + high_price) / 2
  let total_customers := weekly_sales / average_sale_per_customer
  (total_customers / total_houses_per_week) * 100

theorem john_sales_percentage :
  percentage_buyers 50 5 5000 50 150 = 20 := 
by 
  sorry

end john_sales_percentage_l1129_112926


namespace jane_waiting_time_l1129_112949

-- Given conditions as constants for readability
def base_coat_drying_time := 2
def first_color_coat_drying_time := 3
def second_color_coat_drying_time := 3
def top_coat_drying_time := 5

-- Total drying time calculation
def total_drying_time := base_coat_drying_time 
                       + first_color_coat_drying_time 
                       + second_color_coat_drying_time 
                       + top_coat_drying_time

-- The theorem to prove
theorem jane_waiting_time : total_drying_time = 13 := 
by
  sorry

end jane_waiting_time_l1129_112949


namespace polynomial_expansion_l1129_112999

theorem polynomial_expansion : (x + 3) * (x - 6) * (x + 2) = x^3 - x^2 - 24 * x - 36 := 
by
  sorry

end polynomial_expansion_l1129_112999


namespace sum_lent_borrowed_l1129_112933

-- Define the given conditions and the sum lent
def sum_lent (P r t : ℝ) (I : ℝ) : Prop :=
  I = P * r * t / 100 ∧ I = P - 1540

-- Define the main theorem to be proven
theorem sum_lent_borrowed : 
  ∃ P : ℝ, sum_lent P 8 10 ((4 * P) / 5) ∧ P = 7700 :=
by
  sorry

end sum_lent_borrowed_l1129_112933


namespace ashwin_rental_hours_l1129_112988

theorem ashwin_rental_hours (x : ℕ) 
  (h1 : 25 + 10 * x = 125) : 1 + x = 11 :=
by
  sorry

end ashwin_rental_hours_l1129_112988


namespace find_x_l1129_112993

-- Let x be a real number such that x > 0 and the area of the given triangle is 180.
theorem find_x (x : ℝ) (h_pos : x > 0) (h_area : 3 * x^2 = 180) : x = 2 * Real.sqrt 15 :=
by
  -- Placeholder for the actual proof
  sorry

end find_x_l1129_112993


namespace intersection_of_A_and_B_l1129_112900

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 1 ≤ 0}
noncomputable def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_of_A_and_B_l1129_112900


namespace complement_of_A_in_U_l1129_112998

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x ≥ 2}
def complement_U_A : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_of_A_in_U :
  (U \ A) = complement_U_A :=
sorry

end complement_of_A_in_U_l1129_112998


namespace canoe_kayak_rental_l1129_112941

theorem canoe_kayak_rental:
  ∀ (C K : ℕ), 
    12 * C + 18 * K = 504 → 
    C = (3 * K) / 2 → 
    C - K = 7 :=
  by
    intro C K
    intros h1 h2
    sorry

end canoe_kayak_rental_l1129_112941


namespace a10_b10_l1129_112954

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem a10_b10 : a^10 + b^10 = 123 :=
by
  sorry

end a10_b10_l1129_112954


namespace smaug_copper_coins_l1129_112929

def copper_value_of_silver (silver_coins silver_to_copper : ℕ) : ℕ :=
  silver_coins * silver_to_copper

def copper_value_of_gold (gold_coins gold_to_silver silver_to_copper : ℕ) : ℕ :=
  gold_coins * gold_to_silver * silver_to_copper

def total_copper_value (gold_coins silver_coins gold_to_silver silver_to_copper : ℕ) : ℕ :=
  copper_value_of_gold gold_coins gold_to_silver silver_to_copper +
  copper_value_of_silver silver_coins silver_to_copper

def actual_copper_coins (total_value gold_value silver_value : ℕ) : ℕ :=
  total_value - (gold_value + silver_value)

theorem smaug_copper_coins :
  let gold_coins := 100
  let silver_coins := 60
  let silver_to_copper := 8
  let gold_to_silver := 3
  let total_copper_value := 2913
  let gold_value := copper_value_of_gold gold_coins gold_to_silver silver_to_copper
  let silver_value := copper_value_of_silver silver_coins silver_to_copper
  actual_copper_coins total_copper_value gold_value silver_value = 33 :=
by
  sorry

end smaug_copper_coins_l1129_112929


namespace problem_l1129_112945

variable {a b c d : ℝ}

theorem problem (h1 : a > b) (h2 : c > d) : a - d > b - c := sorry

end problem_l1129_112945


namespace total_time_spent_l1129_112937

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2
def total_time : ℕ := outlining_time + writing_time + practicing_time

theorem total_time_spent : total_time = 117 := by
  sorry

end total_time_spent_l1129_112937


namespace f_strictly_decreasing_intervals_f_max_min_on_interval_l1129_112935

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 6 * x^2 - 9 * x + 3

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := -3 * x^2 - 12 * x - 9

-- Statement for part (I)
theorem f_strictly_decreasing_intervals :
  (∀ x : ℝ, x < -3 → f_deriv x < 0) ∧ (∀ x : ℝ, x > -1 → f_deriv x < 0) := by
  sorry

-- Statement for part (II)
theorem f_max_min_on_interval :
  (∀ x ∈ Set.Icc (-4 : ℝ) (2 : ℝ), f x ≤ 7) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) (2 : ℝ), f x ≥ -47) :=
  sorry

end f_strictly_decreasing_intervals_f_max_min_on_interval_l1129_112935


namespace maximum_area_right_triangle_in_rectangle_l1129_112910

theorem maximum_area_right_triangle_in_rectangle :
  ∃ (area : ℕ), 
  (∀ (a b : ℕ), a = 12 ∧ b = 5 → area = 1 / 2 * a * b) :=
by
  use 30
  sorry

end maximum_area_right_triangle_in_rectangle_l1129_112910


namespace simplified_evaluation_eq_half_l1129_112967

theorem simplified_evaluation_eq_half :
  ∃ x y : ℝ, (|x - 2| + (y + 1)^2 = 0) → 
             (3 * x - 2 * (x^2 - (1/2) * y^2) + (x - (1/2) * y^2) = 1/2) :=
by
  sorry

end simplified_evaluation_eq_half_l1129_112967


namespace programs_produce_same_output_l1129_112996

def sum_program_a : ℕ :=
  let S := (Finset.range 1000).sum (λ i => i + 1)
  S

def sum_program_b : ℕ :=
  let S := (Finset.range 1000).sum (λ i => 1000 - i)
  S

theorem programs_produce_same_output :
  sum_program_a = sum_program_b := by
  sorry

end programs_produce_same_output_l1129_112996


namespace determine_divisors_l1129_112942

theorem determine_divisors (n : ℕ) (h_pos : n > 0) (d : ℕ) (h_div : d ∣ 3 * n^2) (h_exists : ∃ k : ℤ, n^2 + d = k^2) : d = 3 * n^2 := 
sorry

end determine_divisors_l1129_112942


namespace hotdog_cost_l1129_112921

theorem hotdog_cost
  (h s : ℕ) -- Make sure to assume that the cost in cents is a natural number 
  (h1 : 3 * h + 2 * s = 360)
  (h2 : 2 * h + 3 * s = 390) :
  h = 60 :=

sorry

end hotdog_cost_l1129_112921


namespace distance_between_intersections_l1129_112936

-- Given conditions
def line_eq (x : ℝ) : ℝ := 5
def quad_eq (x : ℝ) : ℝ := 5 * x^2 + 2 * x - 2

-- The proof statement
theorem distance_between_intersections : 
  ∃ (C D : ℝ), line_eq C = quad_eq C ∧ line_eq D = quad_eq D ∧ abs (C - D) = 2.4 :=
by
  -- We will later fill in the proof here
  sorry

end distance_between_intersections_l1129_112936


namespace smallest_N_l1129_112938

theorem smallest_N (N : ℕ) : (N * 3 ≥ 75) ∧ (N * 2 < 75) → N = 25 :=
by {
  sorry
}

end smallest_N_l1129_112938


namespace tickets_per_candy_l1129_112915

theorem tickets_per_candy (tickets_whack_a_mole : ℕ) (tickets_skee_ball : ℕ) (candies_bought : ℕ)
    (h1 : tickets_whack_a_mole = 26) (h2 : tickets_skee_ball = 19) (h3 : candies_bought = 5) :
    (tickets_whack_a_mole + tickets_skee_ball) / candies_bought = 9 := by
  sorry

end tickets_per_candy_l1129_112915


namespace fraction_division_l1129_112968

theorem fraction_division :
  (5 / 4) / (8 / 15) = 75 / 32 :=
sorry

end fraction_division_l1129_112968


namespace set_intersection_l1129_112982

def A (x : ℝ) : Prop := -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3
def B (x : ℝ) : Prop := (x + 1) / x ≤ 0
def C_x_B (x : ℝ) : Prop := x < -1 ∨ x ≥ 0

theorem set_intersection :
  {x : ℝ | A x} ∩ {x : ℝ | C_x_B x} = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
sorry

end set_intersection_l1129_112982


namespace solve_for_x_l1129_112927

theorem solve_for_x (x : ℕ) 
  (h : 225 + 2 * 15 * 4 + 16 = x) : x = 361 := 
by 
  sorry

end solve_for_x_l1129_112927


namespace relationship_of_products_l1129_112946

theorem relationship_of_products
  {a1 a2 b1 b2 : ℝ}
  (h1 : a1 < a2)
  (h2 : b1 < b2) :
  a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 :=
sorry

end relationship_of_products_l1129_112946


namespace problem_one_problem_two_l1129_112958

variables (a₁ a₂ a₃ : ℤ) (n : ℕ)
def arith_sequence : Prop :=
  a₁ + a₂ + a₃ = 21 ∧ a₁ * a₂ * a₃ = 231

theorem problem_one (h : arith_sequence a₁ a₂ a₃) : a₂ = 7 :=
sorry

theorem problem_two (h : arith_sequence a₁ a₂ a₃) :
  (∃ d : ℤ, (d = -4 ∨ d = 4) ∧ (a_n = a₁ + (n - 1) * d ∨ a_n = a₃ + (n - 1) * d)) :=
sorry

end problem_one_problem_two_l1129_112958


namespace solve_polynomial_l1129_112987

theorem solve_polynomial (z : ℂ) : z^6 - 9 * z^3 + 8 = 0 ↔ z = 1 ∨ z = 2 := 
by
  sorry

end solve_polynomial_l1129_112987


namespace P_ne_77_for_integers_l1129_112980

def P (x y : ℤ) : ℤ :=
  x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_ne_77_for_integers (x y : ℤ) : P x y ≠ 77 :=
by
  sorry

end P_ne_77_for_integers_l1129_112980


namespace smallest_n_inequality_l1129_112909

theorem smallest_n_inequality :
  ∃ (n : ℕ), ∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4) ∧ n = 4 :=
by
  -- Proof steps would go here
  sorry

end smallest_n_inequality_l1129_112909


namespace integral_problem1_integral_problem2_integral_problem3_l1129_112925

open Real

noncomputable def integral1 := ∫ x in (0 : ℝ)..1, x * exp (-x) = 1 - 2 / exp 1
noncomputable def integral2 := ∫ x in (1 : ℝ)..2, x * log x / log 2 = 2 - 3 / (4 * log 2)
noncomputable def integral3 := ∫ x in (1 : ℝ)..Real.exp 1, (log x) ^ 2 = exp 1 - 2

theorem integral_problem1 : integral1 := sorry
theorem integral_problem2 : integral2 := sorry
theorem integral_problem3 : integral3 := sorry

end integral_problem1_integral_problem2_integral_problem3_l1129_112925


namespace number_of_integers_l1129_112981

theorem number_of_integers (n : ℤ) : 
    25 < n^2 ∧ n^2 < 144 → ∃ l, l = 12 :=
by
  sorry

end number_of_integers_l1129_112981


namespace greatest_five_consecutive_odd_integers_l1129_112961

theorem greatest_five_consecutive_odd_integers (A B C D E : ℤ) (x : ℤ) 
  (h1 : B = x + 2) 
  (h2 : C = x + 4)
  (h3 : D = x + 6)
  (h4 : E = x + 8)
  (h5 : A + B + C + D + E = 148) :
  E = 33 :=
by {
  sorry -- proof not required
}

end greatest_five_consecutive_odd_integers_l1129_112961


namespace product_of_roots_quadratic_eq_l1129_112950

theorem product_of_roots_quadratic_eq : 
  ∀ (x1 x2 : ℝ), 
  (∀ x : ℝ, x^2 - 2 * x - 3 = 0 → (x = x1 ∨ x = x2)) → 
  x1 * x2 = -3 :=
by
  intros x1 x2 h
  sorry

end product_of_roots_quadratic_eq_l1129_112950


namespace trapezoid_distances_l1129_112903

-- Define the problem parameters
variables (AB CD AD BC : ℝ)
-- Assume given conditions
axiom h1 : AD > BC
noncomputable def k := AD / BC

-- Formalizing the proof problem in Lean 4
theorem trapezoid_distances (M : Type) (BM AM CM DM : ℝ) :
  BM = AB * BC / (AD - BC) →
  AM = AB * AD / (AD - BC) →
  CM = CD * BC / (AD - BC) →
  DM = CD * AD / (AD - BC) →
  true :=
sorry

end trapezoid_distances_l1129_112903


namespace floor_sum_min_value_l1129_112907

theorem floor_sum_min_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end floor_sum_min_value_l1129_112907


namespace river_flow_rate_l1129_112962

theorem river_flow_rate
  (h : ℝ) (h_eq : h = 3)
  (w : ℝ) (w_eq : w = 36)
  (V : ℝ) (V_eq : V = 3600)
  (conversion_factor : ℝ) (conversion_factor_eq : conversion_factor = 3.6) :
  (60 / (w * h)) * conversion_factor = 2 := by
  sorry

end river_flow_rate_l1129_112962


namespace angle_ne_iff_cos2angle_ne_l1129_112986

theorem angle_ne_iff_cos2angle_ne (A B : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  (A ≠ B) ↔ (Real.cos (2 * A) ≠ Real.cos (2 * B)) :=
sorry

end angle_ne_iff_cos2angle_ne_l1129_112986


namespace part1_part2_l1129_112923

variable (x y : ℤ) (A B : ℤ)

def A_def : ℤ := 3 * x^2 - 5 * x * y - 2 * y^2
def B_def : ℤ := x^2 - 3 * y

theorem part1 : A_def x y - 2 * B_def x y = x^2 - 5 * x * y - 2 * y^2 + 6 * y := by
  sorry

theorem part2 : A_def 2 (-1) - 2 * B_def 2 (-1) = 6 := by
  sorry

end part1_part2_l1129_112923


namespace always_true_inequality_l1129_112985

theorem always_true_inequality (x : ℝ) : x^2 + 1 ≥ 2 * |x| := 
sorry

end always_true_inequality_l1129_112985


namespace quadratic_function_analysis_l1129_112914

theorem quadratic_function_analysis (a b c : ℝ) :
  (a - b + c = -1) →
  (c = 2) →
  (4 * a + 2 * b + c = 2) →
  (16 * a + 4 * b + c = -6) →
  (¬ ∃ x > 3, a * x^2 + b * x + c = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end quadratic_function_analysis_l1129_112914


namespace number_of_boys_l1129_112940

theorem number_of_boys (n : ℕ) (h1 : (n * 182 - 60) / n = 180): n = 30 :=
by
  sorry

end number_of_boys_l1129_112940


namespace gribblean_words_count_l1129_112943

universe u

-- Define the Gribblean alphabet size
def alphabet_size : Nat := 3

-- Words of length 1 to 4
def words_of_length (n : Nat) : Nat :=
  alphabet_size ^ n

-- All possible words count
def total_words : Nat :=
  (words_of_length 1) + (words_of_length 2) + (words_of_length 3) + (words_of_length 4)

-- Theorem statement
theorem gribblean_words_count : total_words = 120 :=
by
  sorry

end gribblean_words_count_l1129_112943


namespace multiply_increase_by_196_l1129_112918

theorem multiply_increase_by_196 (x : ℕ) (h : 14 * x = 14 + 196) : x = 15 :=
sorry

end multiply_increase_by_196_l1129_112918


namespace entrance_fee_increase_l1129_112995

theorem entrance_fee_increase
  (entrance_fee_under_18 : ℕ)
  (rides_cost : ℕ)
  (num_rides : ℕ)
  (total_spent : ℕ)
  (total_cost_twins : ℕ)
  (total_ride_cost_twins : ℕ)
  (amount_spent_joe : ℕ)
  (total_ride_cost_joe : ℕ)
  (joe_entrance_fee : ℕ)
  (increase : ℕ)
  (percentage_increase : ℕ)
  (h1 : entrance_fee_under_18 = 5)
  (h2 : rides_cost = 50) -- representing $0.50 as 50 cents to maintain integer calculations
  (h3 : num_rides = 3)
  (h4 : total_spent = 2050) -- representing $20.5 as 2050 cents
  (h5 : total_cost_twins = 1300) -- combining entrance fees and cost of rides for the twins in cents
  (h6 : total_ride_cost_twins = 300) -- cost of rides for twins in cents
  (h7 : amount_spent_joe = 750) -- representing $7.5 as 750 cents
  (h8 : total_ride_cost_joe = 150) -- cost of rides for Joe in cents
  (h9 : joe_entrance_fee = 600) -- representing $6 as 600 cents
  (h10 : increase = 100) -- increase in entrance fee in cents
  (h11 : percentage_increase = 20) :
  percentage_increase = ((increase * 100) / entrance_fee_under_18) :=
sorry

end entrance_fee_increase_l1129_112995


namespace corrected_mean_l1129_112964

theorem corrected_mean (mean : ℝ) (n : ℕ) (wrong_ob : ℝ) (correct_ob : ℝ) 
(h1 : mean = 36) (h2 : n = 50) (h3 : wrong_ob = 23) (h4 : correct_ob = 34) : 
(mean * n + (correct_ob - wrong_ob)) / n = 36.22 :=
by
  sorry

end corrected_mean_l1129_112964


namespace find_analytical_expression_and_a_l1129_112956

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)

theorem find_analytical_expression_and_a :
  (A > 0) → (ω > 0) → (0 < φ ∧ φ < π / 2) →
  (∀ x, ∃ k : ℤ, f (x + k * π / 2) = f (x)) →
  (∃ A, ∀ x, A * sin (ω * x + φ) ≤ 2) →
  ((∀ x, f (x - π / 6) = -f (-x + π / 6)) ∨ f 0 = sqrt 3 ∨ (∃ x, 2 * x + φ = k * π + π / 2)) →
  (∀ x, f x = 2 * sin (2 * x + π / 3)) ∧
  (∀ (A : ℝ), (0 < A ∧ A < π) → (f A = sqrt 3) →
  (c = 3 ∧ S = 3 * sqrt 3) →
  (a ^ 2 = ((4 * sqrt 3) ^ 2 + 3 ^ 2 - 2 * (4 * sqrt 3) * 3 * cos (π / 6))) → a = sqrt 21) :=
  sorry

end find_analytical_expression_and_a_l1129_112956


namespace triangle_inradius_l1129_112947

theorem triangle_inradius (A s r : ℝ) (h₁ : A = 3 * s) (h₂ : A = r * s) (h₃ : s ≠ 0) : r = 3 :=
by
  -- Proof omitted
  sorry

end triangle_inradius_l1129_112947


namespace even_function_expression_l1129_112931

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x then 2*x + 1 else -2*x + 1

theorem even_function_expression (x : ℝ) (hx : x < 0) :
  f x = -2*x + 1 :=
by sorry

end even_function_expression_l1129_112931


namespace no_x_intersect_one_x_intersect_l1129_112975

variable (m : ℝ)

-- Define the original quadratic function
def quadratic_function (x : ℝ) := x^2 - 2 * m * x + m^2 + 3

-- 1. Prove the function does not intersect the x-axis
theorem no_x_intersect : ∀ m, ∀ x : ℝ, quadratic_function m x ≠ 0 := by
  intros
  unfold quadratic_function
  sorry

-- 2. Prove that translating down by 3 units intersects the x-axis at one point
def translated_quadratic (x : ℝ) := (x - m)^2

theorem one_x_intersect : ∃ x : ℝ, translated_quadratic m x = 0 := by
  unfold translated_quadratic
  sorry

end no_x_intersect_one_x_intersect_l1129_112975


namespace problem_1_problem_2_l1129_112963

section Problem1

variable (x a : ℝ)

-- Proposition p
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0

-- Proposition q
def q (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Problem 1
theorem problem_1 : p 1 x ∧ q x → 2 < x ∧ x < 3 :=
by { sorry }

end Problem1

section Problem2

variable (a : ℝ)

-- Proposition p with a as a variable
def p_a (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0

-- Proposition q with x as a variable
def q_x (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Problem 2
theorem problem_2 : (∀ (x : ℝ), ¬p_a a x → ¬q_x x) → (1 < a ∧ a ≤ 2) :=
by { sorry }

end Problem2

end problem_1_problem_2_l1129_112963


namespace B_holds_32_l1129_112971

variable (x y z : ℝ)

-- Conditions
def condition1 : Prop := x + 1/2 * (y + z) = 90
def condition2 : Prop := y + 1/2 * (x + z) = 70
def condition3 : Prop := z + 1/2 * (x + y) = 56

-- Theorem to prove
theorem B_holds_32 (h1 : condition1 x y z) (h2 : condition2 x y z) (h3 : condition3 x y z) : y = 32 :=
sorry

end B_holds_32_l1129_112971


namespace find_n_l1129_112969

noncomputable def problem_statement (m n : ℤ) : Prop :=
  (∀ x : ℝ, x^2 - (m + 2) * x + (m - 2) = 0 → ∃ x1 x2 : ℝ, x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 * x2 < 0 ∧ x1 > |x2|) ∧
  (∃ r1 r2 : ℚ, r1 * r2 = 2 ∧ m * (r1 * r1 + r2 * r2) = (n - 2) * (r1 + r2) + m^2 - 3)

theorem find_n (m : ℤ) (hm : -2 < m ∧ m < 2) : 
  problem_statement m 5 ∨ problem_statement m (-1) :=
sorry

end find_n_l1129_112969


namespace arithmetic_progression_rth_term_l1129_112983

variable (n r : ℕ)

def S (n : ℕ) : ℕ := 2 * n + 3 * n^2

theorem arithmetic_progression_rth_term : (S r) - (S (r - 1)) = 6 * r - 1 :=
by
  sorry

end arithmetic_progression_rth_term_l1129_112983


namespace weight_of_each_soda_crate_l1129_112955

-- Definitions based on conditions
def bridge_weight_limit := 20000
def empty_truck_weight := 12000
def number_of_soda_crates := 20
def dryer_weight := 3000
def number_of_dryers := 3
def fully_loaded_truck_weight := 24000
def soda_weight := 1000
def produce_weight := 2 * soda_weight
def total_cargo_weight := fully_loaded_truck_weight - empty_truck_weight

-- Lean statement to prove the weight of each soda crate
theorem weight_of_each_soda_crate :
  number_of_soda_crates * ((total_cargo_weight - (number_of_dryers * dryer_weight)) / 3) / number_of_soda_crates = 50 :=
by
  sorry

end weight_of_each_soda_crate_l1129_112955


namespace nialls_children_ages_l1129_112978

theorem nialls_children_ages : ∃ (a b c d : ℕ), 
  a < 18 ∧ b < 18 ∧ c < 18 ∧ d < 18 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = 882 ∧ a + b + c + d = 32 :=
by
  sorry

end nialls_children_ages_l1129_112978


namespace widget_production_difference_l1129_112939

variable (w t : ℕ)
variable (h_wt : w = 2 * t)

theorem widget_production_difference (w t : ℕ)
    (h_wt : w = 2 * t) :
  (w * t) - ((w + 5) * (t - 3)) = t + 15 :=
by 
  sorry

end widget_production_difference_l1129_112939


namespace sum_of_fractions_l1129_112951

theorem sum_of_fractions : 
  (1/12 + 2/12 + 3/12 + 4/12 + 5/12 + 6/12 + 7/12 + 8/12 + 9/12 + 65/12 + 3/4) = 119 / 12 :=
by
  sorry

end sum_of_fractions_l1129_112951


namespace maximum_value_of_f_intervals_of_monotonic_increase_l1129_112952

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 
  let a1 := a x
  let b1 := b x
  a1.1 * (a1.1 + b1.1) + a1.2 * (a1.2 + b1.2)

theorem maximum_value_of_f :
  ∃ x : ℝ, f x = 3 / 2 + Real.sqrt 2 / 2 := sorry

theorem intervals_of_monotonic_increase :
  ∃ I1 I2 : Set ℝ, 
  I1 = Set.Icc 0 (Real.pi / 8) ∧ 
  I2 = Set.Icc (5 * Real.pi / 8) Real.pi ∧ 
  (∀ x ∈ I1, ∀ y ∈ I2, x ≤ y ∧ f x ≤ f y) ∧
  (∀ x y, x ∈ I1 → y ∈ I1 → x < y → f x < f y) ∧
  (∀ x y, x ∈ I2 → y ∈ I2 → x < y → f x < f y) := sorry

end maximum_value_of_f_intervals_of_monotonic_increase_l1129_112952


namespace product_remainder_l1129_112994

-- Define the product of the consecutive numbers
def product := 86 * 87 * 88 * 89 * 90 * 91 * 92

-- Lean statement to state the problem
theorem product_remainder :
  product % 7 = 0 :=
by
  sorry

end product_remainder_l1129_112994


namespace student_thought_six_is_seven_l1129_112953

theorem student_thought_six_is_seven
  (n : ℕ → ℕ)
  (h1 : (n 1 + n 3) / 2 = 2)
  (h2 : (n 2 + n 4) / 2 = 3)
  (h3 : (n 3 + n 5) / 2 = 4)
  (h4 : (n 4 + n 6) / 2 = 5)
  (h5 : (n 5 + n 7) / 2 = 6)
  (h6 : (n 6 + n 8) / 2 = 7)
  (h7 : (n 7 + n 9) / 2 = 8)
  (h8 : (n 8 + n 10) / 2 = 9)
  (h9 : (n 9 + n 1) / 2 = 10)
  (h10 : (n 10 + n 2) / 2 = 1) : 
  n 6 = 7 := 
  sorry

end student_thought_six_is_seven_l1129_112953


namespace frog_eggs_ratio_l1129_112960

theorem frog_eggs_ratio
    (first_day : ℕ)
    (second_day : ℕ)
    (third_day : ℕ)
    (total_eggs : ℕ)
    (h1 : first_day = 50)
    (h2 : second_day = first_day * 2)
    (h3 : third_day = second_day + 20)
    (h4 : total_eggs = 810) :
    (total_eggs - (first_day + second_day + third_day)) / (first_day + second_day + third_day) = 2 :=
by
    sorry

end frog_eggs_ratio_l1129_112960


namespace initial_paintings_l1129_112934

theorem initial_paintings (x : ℕ) (h : x - 3 = 95) : x = 98 :=
sorry

end initial_paintings_l1129_112934


namespace required_additional_coins_l1129_112913

-- Summing up to the first 15 natural numbers
def sum_first_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Given: Alex has 15 friends and 90 coins
def number_of_friends := 15
def initial_coins := 90

-- The total number of coins required
def total_coins_required := sum_first_natural_numbers number_of_friends

-- Calculate the additional coins needed
theorem required_additional_coins : total_coins_required - initial_coins = 30 :=
by
  -- Placeholder for proof
  sorry

end required_additional_coins_l1129_112913
