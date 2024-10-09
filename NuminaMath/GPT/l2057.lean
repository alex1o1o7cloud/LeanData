import Mathlib

namespace max_ab_l2057_205762

theorem max_ab {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 6) : ab ≤ 9 :=
sorry

end max_ab_l2057_205762


namespace power_of_two_divides_sub_one_l2057_205748

theorem power_of_two_divides_sub_one (k : ℕ) (h_odd : k % 2 = 1) : ∀ n ≥ 1, 2^(n+2) ∣ k^(2^n) - 1 :=
by
  sorry

end power_of_two_divides_sub_one_l2057_205748


namespace nacho_will_be_three_times_older_in_future_l2057_205786

variable (N D x : ℕ)
variable (h1 : D = 5)
variable (h2 : N + D = 40)
variable (h3 : N + x = 3 * (D + x))

theorem nacho_will_be_three_times_older_in_future :
  x = 10 :=
by {
  -- Given conditions
  sorry
}

end nacho_will_be_three_times_older_in_future_l2057_205786


namespace minimum_value_of_m_minus_n_l2057_205792

def f (x : ℝ) : ℝ := (x - 1) ^ 2

theorem minimum_value_of_m_minus_n 
  (f_even : ∀ x : ℝ, f x = f (-x))
  (condition1 : n ≤ f (-2))
  (condition2 : n ≤ f (-1 / 2))
  (condition3 : f (-2) ≤ m)
  (condition4 : f (-1 / 2) ≤ m)
  : ∃ n m, m - n = 1 :=
by
  sorry

end minimum_value_of_m_minus_n_l2057_205792


namespace star_operation_l2057_205775

def star (a b : ℚ) : ℚ := 2 * a - b + 1

theorem star_operation :
  star 1 (star 2 (-3)) = -5 :=
by
  -- Calcualtion follows the steps given in the solution, 
  -- but this line is here just to satisfy the 'rewrite the problem' instruction.
  sorry

end star_operation_l2057_205775


namespace composite_rate_proof_l2057_205789

noncomputable def composite_rate (P A : ℝ) (T : ℕ) (X Y Z : ℝ) (R : ℝ) : Prop :=
  let factor := (1 + X / 100) * (1 + Y / 100) * (1 + Z / 100)
  1.375 = factor ∧ (A = P * (1 + R / 100) ^ T)

theorem composite_rate_proof :
  composite_rate 4000 5500 3 X Y Z 11.1 :=
by sorry

end composite_rate_proof_l2057_205789


namespace solution_m_plus_n_l2057_205798

variable (m n : ℝ)

theorem solution_m_plus_n 
  (h₁ : m ≠ 0)
  (h₂ : m^2 + m * n - m = 0) :
  m + n = 1 := by
  sorry

end solution_m_plus_n_l2057_205798


namespace sum_first_10_log_a_l2057_205728

-- Given sum of the first n terms of the sequence
def S (n : ℕ) : ℕ := 2^n - 1

-- Function to get general term log_2 a_n
def log_a (n : ℕ) : ℕ := n - 1

-- The statement to prove
theorem sum_first_10_log_a : (List.range 10).sum = 45 := by 
  sorry

end sum_first_10_log_a_l2057_205728


namespace min_value_of_expression_l2057_205785

theorem min_value_of_expression (a b : ℝ) (h₁ : 0 < a) (h₂ : 1 < b) (h₃ : a + b = 2) :
  4 / a + 1 / (b - 1) = 9 := 
sorry

end min_value_of_expression_l2057_205785


namespace divide_decimals_l2057_205776

theorem divide_decimals : (0.24 / 0.006) = 40 := by
  sorry

end divide_decimals_l2057_205776


namespace solve_equation_l2057_205717

noncomputable def f (x : ℝ) : ℝ :=
  2 * x + 1 + Real.arctan x * Real.sqrt (x^2 + 1)

theorem solve_equation : ∃ x : ℝ, f x + f (x + 1) = 0 ∧ x = -1/2 :=
  by
    use -1/2
    simp [f]
    sorry

end solve_equation_l2057_205717


namespace ursula_hourly_wage_l2057_205759

def annual_salary : ℝ := 16320
def hours_per_day : ℝ := 8
def days_per_month : ℝ := 20
def months_per_year : ℝ := 12

theorem ursula_hourly_wage : 
  (annual_salary / months_per_year) / (hours_per_day * days_per_month) = 8.50 := by 
  sorry

end ursula_hourly_wage_l2057_205759


namespace nonneg_reals_sum_to_one_implies_ineq_l2057_205770

theorem nonneg_reals_sum_to_one_implies_ineq
  (x y z : ℝ)
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 :=
sorry

end nonneg_reals_sum_to_one_implies_ineq_l2057_205770


namespace number_decomposition_l2057_205743

theorem number_decomposition (n : ℕ) : n = 6058 → (n / 1000 = 6) ∧ ((n % 100) / 10 = 5) ∧ (n % 10 = 8) :=
by
  -- Actual proof will go here
  sorry

end number_decomposition_l2057_205743


namespace range_of_x_div_y_l2057_205724

theorem range_of_x_div_y {x y : ℝ} (hx : 1 < x ∧ x < 6) (hy : 2 < y ∧ y < 8) : 
  (1/8 < x / y) ∧ (x / y < 3) :=
sorry

end range_of_x_div_y_l2057_205724


namespace total_red_and_green_peaches_l2057_205795

-- Define the number of red peaches and green peaches.
def red_peaches : ℕ := 6
def green_peaches : ℕ := 16

-- Theorem stating the sum of red and green peaches is 22.
theorem total_red_and_green_peaches : red_peaches + green_peaches = 22 := 
by
  -- Proof would go here but is not required
  sorry

end total_red_and_green_peaches_l2057_205795


namespace evaluate_expression_l2057_205742

theorem evaluate_expression : (16 ^ 24) / (64 ^ 8) = 16 ^ 12 :=
by sorry

end evaluate_expression_l2057_205742


namespace proportion_x_l2057_205764

theorem proportion_x (x : ℝ) (h : 0.60 / x = 6 / 4) : x = 0.4 :=
sorry

end proportion_x_l2057_205764


namespace find_f_lg_lg_2_l2057_205718

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * (Real.sin x) + 4

theorem find_f_lg_lg_2 (a b : ℝ) (m : ℝ) 
  (h1 : f a b (Real.logb 10 2) = 5) 
  (h2 : m = Real.logb 10 2) : 
  f a b (Real.logb 2 m) = 3 :=
sorry

end find_f_lg_lg_2_l2057_205718


namespace find_alpha_l2057_205799

variable (α β k : ℝ)

axiom h1 : α * β = k
axiom h2 : α = -4
axiom h3 : β = -8
axiom k_val : k = 32
axiom β_val : β = 12

theorem find_alpha (h1 : α * β = k) (h2 : α = -4) (h3 : β = -8) (k_val : k = 32) (β_val : β = 12) :
  α = 8 / 3 :=
sorry

end find_alpha_l2057_205799


namespace num_ways_product_72_l2057_205715

def num_ways_product (n : ℕ) : ℕ := sorry  -- Definition for D(n), the number of ways to write n as a product of integers greater than 1

def example_integer := 72  -- Given integer n

theorem num_ways_product_72 : num_ways_product example_integer = 67 := by 
  sorry

end num_ways_product_72_l2057_205715


namespace ring_revolutions_before_stopping_l2057_205744

variable (R ω μ m g : ℝ) -- Declare the variables as real numbers

-- Statement of the theorem
theorem ring_revolutions_before_stopping
  (h_positive_R : 0 < R)
  (h_positive_ω : 0 < ω)
  (h_positive_μ : 0 < μ)
  (h_positive_m : 0 < m)
  (h_positive_g : 0 < g) :
  let N1 := m * g / (1 + μ^2)
  let N2 := μ * m * g / (1 + μ^2)
  let K_initial := (1 / 2) * m * R^2 * ω^2
  let A_friction := -2 * π * R * n * μ * (N1 + N2)
  ∃ n : ℝ, n = ω^2 * R * (1 + μ^2) / (4 * π * g * μ * (1 + μ)) :=
by sorry

end ring_revolutions_before_stopping_l2057_205744


namespace factor_poly_l2057_205716

theorem factor_poly (P Q : ℝ) (h1 : ∃ b c : ℝ, 
  (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + Q)
  : P + Q = 50 :=
sorry

end factor_poly_l2057_205716


namespace region_area_l2057_205731

theorem region_area : 
  (∃ (x y : ℝ), abs (4 * x - 16) + abs (3 * y + 9) ≤ 6) →
  (∀ (A : ℝ), (∀ x y : ℝ, abs (4 * x - 16) + abs (3 * y + 9) ≤ 6 → 0 ≤ A ∧ A = 6)) :=
by
  intro h exist_condtion
  sorry

end region_area_l2057_205731


namespace value_S3_S2_S5_S3_l2057_205768

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
variable {d : ℝ}
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (d_ne_zero : d ≠ 0)
variable (h_geom_seq : (a 1 + 2 * d) ^ 2 = (a 1) * (a 1 + 3 * d))
variable (S_def : ∀ n, S n = n * a 1 + d * (n * (n - 1)) / 2)

theorem value_S3_S2_S5_S3 : (S 3 - S 2) / (S 5 - S 3) = 2 := by
  sorry

end value_S3_S2_S5_S3_l2057_205768


namespace minimum_value_expr_l2057_205778

noncomputable def expr (x : ℝ) : ℝ := 9 * x + 3 / (x ^ 3)

theorem minimum_value_expr : (∀ x : ℝ, x > 0 → expr x ≥ 12) ∧ (∃ x : ℝ, x > 0 ∧ expr x = 12) :=
by
  sorry

end minimum_value_expr_l2057_205778


namespace exist_distinct_indices_l2057_205760

theorem exist_distinct_indices (n : ℕ) (h1 : n > 3)
  (a : Fin n.succ → ℕ) 
  (h2 : StrictMono a) 
  (h3 : a n ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n.succ), 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ 
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ 
    k ≠ l ∧ k ≠ m ∧ l ≠ m ∧ 
    a i + a j = a k + a l ∧ 
    a k + a l = a m := 
sorry

end exist_distinct_indices_l2057_205760


namespace oldest_brother_age_ratio_l2057_205711

-- Define the ages
def rick_age : ℕ := 15
def youngest_brother_age : ℕ := 3
def smallest_brother_age : ℕ := youngest_brother_age + 2
def middle_brother_age : ℕ := smallest_brother_age * 2
def oldest_brother_age : ℕ := middle_brother_age * 3

-- Define the ratio
def expected_ratio : ℕ := oldest_brother_age / rick_age

theorem oldest_brother_age_ratio : expected_ratio = 2 := by
  sorry 

end oldest_brother_age_ratio_l2057_205711


namespace solve_system_of_equations_l2057_205783

theorem solve_system_of_equations (x y : ℚ)
  (h1 : 15 * x + 24 * y = 18)
  (h2 : 24 * x + 15 * y = 63) :
  x = 46 / 13 ∧ y = -19 / 13 := 
sorry

end solve_system_of_equations_l2057_205783


namespace Gloria_pine_tree_price_l2057_205755

theorem Gloria_pine_tree_price :
  ∀ (cabin_cost cash cypress_count pine_count maple_count cypress_price maple_price left_over_price : ℕ)
  (cypress_total maple_total total_required total_from_cypress_and_maple total_needed amount_per_pine : ℕ),
    cabin_cost = 129000 →
    cash = 150 →
    cypress_count = 20 →
    pine_count = 600 →
    maple_count = 24 →
    cypress_price = 100 →
    maple_price = 300 →
    left_over_price = 350 →
    cypress_total = cypress_count * cypress_price →
    maple_total = maple_count * maple_price →
    total_required = cabin_cost - cash + left_over_price →
    total_from_cypress_and_maple = cypress_total + maple_total →
    total_needed = total_required - total_from_cypress_and_maple →
    amount_per_pine = total_needed / pine_count →
    amount_per_pine = 200 :=
by
  intros
  sorry

end Gloria_pine_tree_price_l2057_205755


namespace smallest_n_for_three_nested_rectangles_l2057_205777

/-- Rectangle represented by its side lengths -/
structure Rectangle where
  x : ℕ
  y : ℕ
  h1 : 1 ≤ x
  h2 : x ≤ y
  h3 : y ≤ 100

/-- Define the nesting relation between rectangles -/
def nested (R1 R2 : Rectangle) : Prop :=
  R1.x < R2.x ∧ R1.y < R2.y

/-- Prove the smallest n such that there exist 3 nested rectangles out of n rectangles where n = 101 -/
theorem smallest_n_for_three_nested_rectangles (n : ℕ) (h : n ≥ 101) :
  ∀ (rectangles : Fin n → Rectangle), 
    ∃ (R1 R2 R3 : Fin n), nested (rectangles R1) (rectangles R2) ∧ nested (rectangles R2) (rectangles R3) :=
  sorry

end smallest_n_for_three_nested_rectangles_l2057_205777


namespace number_of_quadruplets_l2057_205709

variables (a b c : ℕ)

theorem number_of_quadruplets (h1 : 2 * a + 3 * b + 4 * c = 1200)
                             (h2 : b = 3 * c)
                             (h3 : a = 2 * b) :
  4 * c = 192 :=
by
  sorry

end number_of_quadruplets_l2057_205709


namespace inequality_proof_l2057_205754

theorem inequality_proof (a b c : ℝ) (hab : a * b < 0) : 
  a^2 + b^2 + c^2 > 2 * a * b + 2 * b * c + 2 * c * a := 
by 
  sorry

end inequality_proof_l2057_205754


namespace num_ways_arrange_passengers_l2057_205719

theorem num_ways_arrange_passengers 
  (seats : ℕ) (passengers : ℕ) (consecutive_empty : ℕ)
  (h1 : seats = 10) (h2 : passengers = 4) (h3 : consecutive_empty = 5) :
  ∃ ways, ways = 480 := by
  sorry

end num_ways_arrange_passengers_l2057_205719


namespace problem_arith_sequences_l2057_205772

theorem problem_arith_sequences (a b : ℕ → ℕ) 
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = b n + e)
  (h1 : a 1 = 25)
  (h2 : b 1 = 75)
  (h3 : a 2 + b 2 = 100) : 
  a 37 + b 37 = 100 := 
sorry

end problem_arith_sequences_l2057_205772


namespace remainder_when_101_divided_by_7_is_3_l2057_205703

theorem remainder_when_101_divided_by_7_is_3
    (A : ℤ)
    (h : 9 * A + 1 = 10 * A - 100) : A % 7 = 3 := by
  -- Mathematical steps are omitted as instructed
  sorry

end remainder_when_101_divided_by_7_is_3_l2057_205703


namespace choir_females_correct_l2057_205781

noncomputable def number_of_females_in_choir : ℕ :=
  let orchestra_males := 11
  let orchestra_females := 12
  let orchestra_musicians := orchestra_males + orchestra_females
  let band_males := 2 * orchestra_males
  let band_females := 2 * orchestra_females
  let band_musicians := 2 * orchestra_musicians
  let total_musicians := 98
  let choir_males := 12
  let choir_musicians := total_musicians - (orchestra_musicians + band_musicians)
  let choir_females := choir_musicians - choir_males
  choir_females

theorem choir_females_correct : number_of_females_in_choir = 17 := by
  sorry

end choir_females_correct_l2057_205781


namespace problem_sequence_sum_l2057_205710

theorem problem_sequence_sum (a : ℤ) (h : 14 * a^2 + 7 * a = 135) : 7 * a + (a - 1) = 23 :=
by {
  sorry
}

end problem_sequence_sum_l2057_205710


namespace radius_of_larger_circle_is_25_over_3_l2057_205701

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := (5 / 2) * r 

theorem radius_of_larger_circle_is_25_over_3
  (rAB rBD : ℝ)
  (h_ratio : 2 * rBD = 5 * rBD / 2)
  (h_ab : rAB = 8)
  (h_tangent : ∀ rBD, (5 * rBD / 2 - 8) ^ 2 = 64 + rBD ^ 2) :
  radius_of_larger_circle (10 / 3) = 25 / 3 :=
  by
  sorry

end radius_of_larger_circle_is_25_over_3_l2057_205701


namespace inequality_problem_l2057_205782

theorem inequality_problem
  (a b c d e : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : c ≤ d)
  (h4 : d ≤ e)
  (h5 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1 / 5 :=
by
  sorry

end inequality_problem_l2057_205782


namespace train_length_is_correct_l2057_205791

-- Define the given conditions and the expected result.
def train_speed_kmph : ℝ := 270
def time_seconds : ℝ := 5
def expected_length_meters : ℝ := 375

-- State the theorem to be proven.
theorem train_length_is_correct :
  (train_speed_kmph * 1000 / 3600) * time_seconds = expected_length_meters := by
  sorry -- Proof is not required, so we use 'sorry'

end train_length_is_correct_l2057_205791


namespace product_squared_inequality_l2057_205747

theorem product_squared_inequality (n : ℕ) (a : Fin n → ℝ) (h : (Finset.univ.prod (λ i => a i)) = 1) :
    (Finset.univ.prod (λ i => (1 + (a i)^2))) ≥ 2^n := 
sorry

end product_squared_inequality_l2057_205747


namespace utilities_cost_l2057_205736

theorem utilities_cost
    (rent1 : ℝ) (utility1 : ℝ) (rent2 : ℝ) (utility2 : ℝ)
    (distance1 : ℝ) (distance2 : ℝ) 
    (cost_per_mile : ℝ) 
    (drive_days : ℝ) (cost_difference : ℝ)
    (h1 : rent1 = 800)
    (h2 : rent2 = 900)
    (h3 : utility2 = 200)
    (h4 : distance1 = 31)
    (h5 : distance2 = 21)
    (h6 : cost_per_mile = 0.58)
    (h7 : drive_days = 20)
    (h8 : cost_difference = 76)
    : utility1 = 259.60 := 
by
  sorry

end utilities_cost_l2057_205736


namespace rickshaw_distance_l2057_205704

theorem rickshaw_distance :
  ∃ (distance : ℝ), 
  (13.5 + (distance - 1) * (2.50 / (1 / 3))) = 103.5 ∧ distance = 13 :=
by
  sorry

end rickshaw_distance_l2057_205704


namespace inequality_holds_if_b_greater_than_2_l2057_205712

variable (x : ℝ) (b : ℝ)

theorem inequality_holds_if_b_greater_than_2  :
  (b > 0) → (∃ x, |x-5| + |x-7| < b) ↔ (b > 2) := sorry

end inequality_holds_if_b_greater_than_2_l2057_205712


namespace constant_condition_for_quadrant_I_solution_l2057_205750

-- Define the given conditions
def equations (c : ℚ) (x y : ℚ) : Prop :=
  (x - 2 * y = 5) ∧ (c * x + 3 * y = 2)

-- Define the condition for the solution to be in Quadrant I
def isQuadrantI (x y : ℚ) : Prop :=
  (x > 0) ∧ (y > 0)

-- The theorem to be proved
theorem constant_condition_for_quadrant_I_solution (c : ℚ) :
  (∃ x y : ℚ, equations c x y ∧ isQuadrantI x y) ↔ (-3/2 < c ∧ c < 2/5) :=
by
  sorry

end constant_condition_for_quadrant_I_solution_l2057_205750


namespace annulus_area_l2057_205797

variables {R r d : ℝ}
variables (h1 : R > r) (h2 : d < R)

theorem annulus_area :
  π * (R^2 - r^2 - d^2 / (R - r)) = π * ((R - r)^2 - d^2) :=
sorry

end annulus_area_l2057_205797


namespace B_is_criminal_l2057_205767

-- Introduce the conditions
variable (A B C : Prop)  -- A, B, and C represent whether each individual is the criminal.

-- A says they did not commit the crime
axiom A_says_innocent : ¬A

-- Exactly one of A_says_innocent must hold true (A says ¬A, so B or C must be true)
axiom exactly_one_assertion_true : (¬A ∨ B ∨ C)

-- Problem Statement: Prove that B is the criminal
theorem B_is_criminal : B :=
by
  -- Solution steps would go here
  sorry

end B_is_criminal_l2057_205767


namespace oliver_final_amount_l2057_205714

variable (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ)

def final_amount_after_transactions (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ) : ℕ :=
  initial_amount + saved_amount - (spent_on_frisbee + spent_on_puzzle) + gift

theorem oliver_final_amount :
  final_amount_after_transactions 9 5 4 3 8 = 15 :=
by
  -- We can fill in the exact calculations here to provide the proof.
  sorry

end oliver_final_amount_l2057_205714


namespace find_radius_l2057_205758

theorem find_radius
  (r_1 r_2 r_3 : ℝ)
  (h_cone : r_2 = 2 * r_1 ∧ r_3 = 3 * r_1 ∧ r_1 + r_2 + r_3 = 18) :
  r_1 = 3 :=
by
  sorry

end find_radius_l2057_205758


namespace range_of_m_l2057_205779

def one_root_condition (m : ℝ) : Prop :=
  (4 - 4 * m) * (2 * m + 4) ≤ 0 ∧ m ≠ 0

theorem range_of_m : {m : ℝ | (4 - 4 * m) * (2 * m + 4) ≤ 0 ∧ m ≠ 0} = {m | m ≤ -2 ∨ m ≥ 1} :=
by
  sorry

end range_of_m_l2057_205779


namespace decimal_equiv_of_one_fourth_cubed_l2057_205707

theorem decimal_equiv_of_one_fourth_cubed : (1 / 4 : ℝ) ^ 3 = 0.015625 := 
by sorry

end decimal_equiv_of_one_fourth_cubed_l2057_205707


namespace oranges_in_bin_l2057_205733

theorem oranges_in_bin (initial_oranges thrown_out new_oranges : ℕ) (h1 : initial_oranges = 34) (h2 : thrown_out = 20) (h3 : new_oranges = 13) :
  (initial_oranges - thrown_out + new_oranges = 27) :=
by
  sorry

end oranges_in_bin_l2057_205733


namespace problem_statement_l2057_205729

def diamond (x y : ℝ) : ℝ := (x + y) ^ 2 * (x - y) ^ 2

theorem problem_statement : diamond 2 (diamond 3 4) = 5745329 := by
  sorry

end problem_statement_l2057_205729


namespace sqrt_four_eq_two_or_neg_two_l2057_205732

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 :=
by 
  sorry

end sqrt_four_eq_two_or_neg_two_l2057_205732


namespace lemons_needed_l2057_205787

theorem lemons_needed (initial_lemons : ℝ) (initial_gallons : ℝ) 
  (reduced_ratio : ℝ) (first_gallons : ℝ) (total_gallons : ℝ) :
  initial_lemons / initial_gallons * first_gallons 
  + (initial_lemons / initial_gallons * reduced_ratio) * (total_gallons - first_gallons) = 56.25 :=
by 
  let initial_ratio := initial_lemons / initial_gallons
  let reduced_ratio_amount := initial_ratio * reduced_ratio 
  let lemons_first := initial_ratio * first_gallons
  let lemons_remaining := reduced_ratio_amount * (total_gallons - first_gallons)
  let total_lemons := lemons_first + lemons_remaining
  show total_lemons = 56.25
  sorry

end lemons_needed_l2057_205787


namespace train_speed_is_144_l2057_205746

-- Definitions for the conditions
def length_of_train_passing_pole (S : ℝ) := S * 8
def length_of_train_passing_stationary_train (S : ℝ) := S * 18 - 400

-- The main theorem to prove the speed of the train
theorem train_speed_is_144 (S : ℝ) :
  (length_of_train_passing_pole S = length_of_train_passing_stationary_train S) →
  (S * 3.6 = 144) :=
by
  sorry

end train_speed_is_144_l2057_205746


namespace greatest_root_of_gx_l2057_205705

theorem greatest_root_of_gx :
  ∃ x : ℝ, (10 * x^4 - 16 * x^2 + 3 = 0) ∧ (∀ y : ℝ, (10 * y^4 - 16 * y^2 + 3 = 0) → x ≥ y) ∧ x = Real.sqrt (3 / 5) := 
sorry

end greatest_root_of_gx_l2057_205705


namespace min_square_side_length_l2057_205756

theorem min_square_side_length (s : ℝ) (h : s^2 ≥ 625) : s ≥ 25 :=
sorry

end min_square_side_length_l2057_205756


namespace alcohol_water_ratio_l2057_205796

theorem alcohol_water_ratio (V : ℝ) (hV_pos : V > 0) :
  let jar1_alcohol := (2 / 3) * V
  let jar1_water := (1 / 3) * V
  let jar2_alcohol := (3 / 2) * V
  let jar2_water := (1 / 2) * V
  let total_alcohol := jar1_alcohol + jar2_alcohol
  let total_water := jar1_water + jar2_water
  (total_alcohol / total_water) = (13 / 5) :=
by
  -- Placeholder for the proof
  sorry

end alcohol_water_ratio_l2057_205796


namespace part_one_part_two_range_l2057_205738

/-
Definitions based on conditions from the problem:
- Given vectors ax = (\cos x, \sin x), bx = (3, - sqrt(3))
- Domain for x is [0, π]
--
- Prove if a + b is parallel to b, then x = 5π / 6
- Definition of function f(x), and g(x) based on problem requirements.
- Prove the range of g(x) is [-3, sqrt(3)]
-/

/-
Part (1):
Given ax + bx = (cos x + 3, sin x - sqrt(3)) is parallel to bx =  (3, - sqrt(3));
Prove that x = 5π / 6 under x ∈ [0, π].
-/
noncomputable def vector_ax (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_bx : ℝ × ℝ := (3, - Real.sqrt 3)

theorem part_one (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) 
  (h_parallel : (vector_ax x).1 + vector_bx.1 = (vector_ax x).2 + vector_bx.2) :
  x = 5 * Real.pi / 6 :=
  sorry

/-
Part (2):
Let f(x) = 3 cos x - sqrt(3) sin x.
The function g(x) = -2 sqrt(3) sin(1/2 x - 2π/3) is defined by shifting f(x) right by π/3 and doubling the horizontal coordinate.
Prove the range of g(x) is [-3, sqrt(3)].
-/
noncomputable def f (x : ℝ) := 3 * Real.cos x - Real.sqrt 3 * Real.sin x
noncomputable def g (x : ℝ) := -2 * Real.sqrt 3 * Real.sin (0.5 * x - 2 * Real.pi / 3)

theorem part_two_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) : 
  -3 ≤ g x ∧ g x ≤ Real.sqrt 3 :=
  sorry

end part_one_part_two_range_l2057_205738


namespace seq_b_is_geometric_l2057_205788

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence {a_n} with first term a_1 and common ratio q
def a_n (a₁ q : α) (n : ℕ) : α := a₁ * q^(n-1)

-- Define the sequence {b_n}
def b_n (a₁ q : α) (n : ℕ) : α :=
  a_n a₁ q (3*n - 2) + a_n a₁ q (3*n - 1) + a_n a₁ q (3*n)

-- Theorem stating {b_n} is a geometric sequence with common ratio q^3
theorem seq_b_is_geometric (a₁ q : α) (h : q ≠ 1) :
  ∀ n : ℕ, b_n a₁ q (n + 1) = q^3 * b_n a₁ q n :=
by
  sorry

end seq_b_is_geometric_l2057_205788


namespace gym_hours_per_week_l2057_205706

-- Definitions for conditions
def timesAtGymEachWeek : ℕ := 3
def weightliftingTimeEachDay : ℕ := 1
def warmupCardioFraction : ℚ := 1 / 3

-- The theorem to prove
theorem gym_hours_per_week : (timesAtGymEachWeek * (weightliftingTimeEachDay + weightliftingTimeEachDay * warmupCardioFraction) = 4) := 
by
  sorry

end gym_hours_per_week_l2057_205706


namespace num_distinguishable_octahedrons_l2057_205780

-- Define the given conditions
def num_faces : ℕ := 8
def num_colors : ℕ := 8
def total_permutations : ℕ := Nat.factorial num_colors
def distinct_orientations : ℕ := 24

-- Prove the main statement
theorem num_distinguishable_octahedrons : total_permutations / distinct_orientations = 1680 :=
by
  sorry

end num_distinguishable_octahedrons_l2057_205780


namespace value_at_17pi_over_6_l2057_205765

variable (f : Real → Real)

-- Defining the conditions
def period (f : Real → Real) (T : Real) := ∀ x, f (x + T) = f x
def specific_value (f : Real → Real) (x : Real) (v : Real) := f x = v

-- The main theorem statement
theorem value_at_17pi_over_6 : 
  period f (π / 2) →
  specific_value f (π / 3) 1 →
  specific_value f (17 * π / 6) 1 :=
by
  intros h_period h_value
  sorry

end value_at_17pi_over_6_l2057_205765


namespace parallelogram_rectangle_l2057_205771

/-- A quadrilateral is a parallelogram if both pairs of opposite sides are equal,
and it is a rectangle if its diagonals are equal. -/
structure Quadrilateral :=
  (side1 side2 side3 side4 : ℝ)
  (diag1 diag2 : ℝ)

structure Parallelogram extends Quadrilateral :=
  (opposite_sides_equal : side1 = side3 ∧ side2 = side4)

def is_rectangle (p : Parallelogram) : Prop :=
  p.diag1 = p.diag2 → (p.side1^2 + p.side2^2 = p.side3^2 + p.side4^2)

theorem parallelogram_rectangle (p : Parallelogram) : is_rectangle p :=
  sorry

end parallelogram_rectangle_l2057_205771


namespace completing_square_solution_l2057_205721

theorem completing_square_solution (x : ℝ) : x^2 - 4 * x - 22 = 0 ↔ (x - 2)^2 = 26 := sorry

end completing_square_solution_l2057_205721


namespace ratio_p_q_l2057_205794

theorem ratio_p_q 
  (total_amount : ℕ) 
  (amount_r : ℕ) 
  (ratio_q_r : ℕ × ℕ) 
  (total_amount_eq : total_amount = 1210) 
  (amount_r_eq : amount_r = 400) 
  (ratio_q_r_eq : ratio_q_r = (9, 10)) :
  ∃ (amount_p amount_q : ℕ), 
    total_amount = amount_p + amount_q + amount_r ∧ 
    (amount_q : ℕ) = 9 * (amount_r / 10) ∧ 
    (amount_p : ℕ) / (amount_q : ℕ) = 5 / 4 := 
by sorry

end ratio_p_q_l2057_205794


namespace find_x_l2057_205793

theorem find_x (x : ℝ) : 
  0.65 * x = 0.20 * 682.50 → x = 210 := 
by 
  sorry

end find_x_l2057_205793


namespace min_value_expression_l2057_205757

noncomputable def expression (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x)

theorem min_value_expression : ∃ x : ℝ, expression x = -6480.25 :=
sorry

end min_value_expression_l2057_205757


namespace sin_2phi_l2057_205773

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l2057_205773


namespace dissection_impossible_l2057_205727

theorem dissection_impossible :
  ∀ (n m : ℕ), n = 1000 → m = 2016 → ¬(∃ (k l : ℕ), k * (n * m) = 1 * 2015 + l * 3) :=
by
  intros n m hn hm
  sorry

end dissection_impossible_l2057_205727


namespace sum_a_c_eq_l2057_205766

theorem sum_a_c_eq
  (a b c d : ℝ)
  (h1 : a * b + a * c + b * c + b * d + c * d + a * d = 40)
  (h2 : b^2 + d^2 = 29) :
  a + c = 8.4 :=
by
  sorry

end sum_a_c_eq_l2057_205766


namespace min_abs_sum_l2057_205735

theorem min_abs_sum (x y : ℝ) : (|x - 1| + |x| + |y - 1| + |y + 1|) ≥ 3 :=
sorry

end min_abs_sum_l2057_205735


namespace inequality_xy_gt_xz_l2057_205769

theorem inequality_xy_gt_xz (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 1) : 
  x * y > x * z := 
by
  sorry  -- Proof is not required as per the instructions

end inequality_xy_gt_xz_l2057_205769


namespace abi_suji_age_ratio_l2057_205740

theorem abi_suji_age_ratio (A S : ℕ) (h1 : S = 24) 
  (h2 : (A + 3) / (S + 3) = 11 / 9) : A / S = 5 / 4 := 
by 
  sorry

end abi_suji_age_ratio_l2057_205740


namespace geometric_sequence_a3a5_l2057_205745

theorem geometric_sequence_a3a5 (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r) (h2 : a 4 = 5) : a 3 * a 5 = 25 :=
by
  sorry

end geometric_sequence_a3a5_l2057_205745


namespace range_of_m_l2057_205720

theorem range_of_m (m x1 x2 y1 y2 : ℝ) (h₁ : x1 < x2) (h₂ : y1 < y2)
  (A_on_line : y1 = (2 * m - 1) * x1 + 1)
  (B_on_line : y2 = (2 * m - 1) * x2 + 1) :
  m > 0.5 :=
sorry

end range_of_m_l2057_205720


namespace min_x_squared_plus_y_squared_l2057_205734

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 3) * (y - 3) = 0) : x^2 + y^2 = 18 :=
sorry

end min_x_squared_plus_y_squared_l2057_205734


namespace parabola_intersection_at_1_2003_l2057_205737

theorem parabola_intersection_at_1_2003 (p q : ℝ) (h : p + q = 2002) :
  (1, (1 : ℝ)^2 + p * 1 + q) = (1, 2003) :=
by
  sorry

end parabola_intersection_at_1_2003_l2057_205737


namespace edward_original_amount_l2057_205722

-- Given conditions
def spent : ℝ := 16
def remaining : ℝ := 6

-- Question: How much did Edward have before he spent his money?
-- Correct answer: 22
theorem edward_original_amount : (spent + remaining) = 22 :=
by sorry

end edward_original_amount_l2057_205722


namespace length_of_qr_l2057_205751

theorem length_of_qr (Q : ℝ) (PQ QR : ℝ) 
  (h1 : Real.sin Q = 0.6)
  (h2 : PQ = 15) :
  QR = 18.75 :=
by
  sorry

end length_of_qr_l2057_205751


namespace eduardo_frankie_classes_total_l2057_205790

theorem eduardo_frankie_classes_total (eduardo_classes : ℕ) (h₁ : eduardo_classes = 3) 
                                       (h₂ : ∀ frankie_classes, frankie_classes = 2 * eduardo_classes) :
  ∃ total_classes : ℕ, total_classes = eduardo_classes + 2 * eduardo_classes := 
by
  use 3 + 2 * 3
  sorry

end eduardo_frankie_classes_total_l2057_205790


namespace mean_of_combined_sets_l2057_205749

theorem mean_of_combined_sets 
  (mean1 mean2 mean3 : ℚ)
  (count1 count2 count3 : ℕ)
  (h1 : mean1 = 15)
  (h2 : mean2 = 20)
  (h3 : mean3 = 12)
  (hc1 : count1 = 7)
  (hc2 : count2 = 8)
  (hc3 : count3 = 5) :
  ((count1 * mean1 + count2 * mean2 + count3 * mean3) / (count1 + count2 + count3)) = 16.25 :=
by
  sorry

end mean_of_combined_sets_l2057_205749


namespace same_color_points_exist_l2057_205752

theorem same_color_points_exist (d : ℝ) (colored_plane : ℝ × ℝ → Prop) :
  (∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ colored_plane p1 = colored_plane p2 ∧ dist p1 p2 = d) := 
sorry

end same_color_points_exist_l2057_205752


namespace promotional_pricing_plan_l2057_205713

theorem promotional_pricing_plan (n : ℕ) : 
  (8 * 100 = 800) ∧ 
  (∀ n > 100, 6 * n < 640) :=
by
  sorry

end promotional_pricing_plan_l2057_205713


namespace henry_present_age_l2057_205763

theorem henry_present_age (H J : ℕ) (h1 : H + J = 41) (h2 : H - 7 = 2 * (J - 7)) : H = 25 :=
sorry

end henry_present_age_l2057_205763


namespace largest_of_A_B_C_l2057_205726

noncomputable def A : ℝ := (2010 / 2009) + (2010 / 2011)
noncomputable def B : ℝ := (2010 / 2011) + (2012 / 2011)
noncomputable def C : ℝ := (2011 / 2010) + (2011 / 2012)

theorem largest_of_A_B_C : B > A ∧ B > C := by
  sorry

end largest_of_A_B_C_l2057_205726


namespace slower_train_time_to_pass_driver_faster_one_l2057_205784

noncomputable def convert_speed (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def relative_speed (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let speed1 := convert_speed speed1_kmh
  let speed2 := convert_speed speed2_kmh
  speed1 + speed2

noncomputable def time_to_pass (length1_m length2_m speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let relative_speed := relative_speed speed1_kmh speed2_kmh
  (length1_m + length2_m) / relative_speed

theorem slower_train_time_to_pass_driver_faster_one :
  ∀ (length1 length2 speed1 speed2 : ℝ),
    length1 = 900 → length2 = 900 →
    speed1 = 45 → speed2 = 30 →
    time_to_pass length1 length2 speed1 speed2 = 86.39 :=
by
  intros
  simp only [time_to_pass, relative_speed, convert_speed]
  sorry

end slower_train_time_to_pass_driver_faster_one_l2057_205784


namespace lindy_distance_traveled_l2057_205708

/-- Jack and Christina are standing 240 feet apart on a level surface. 
Jack walks in a straight line toward Christina at a constant speed of 5 feet per second. 
Christina walks in a straight line toward Jack at a constant speed of 3 feet per second. 
Lindy runs at a constant speed of 9 feet per second from Christina to Jack, back to Christina, back to Jack, and so forth. 
The total distance Lindy travels when the three meet at one place is 270 feet. -/
theorem lindy_distance_traveled
    (initial_distance : ℝ)
    (jack_speed : ℝ)
    (christina_speed : ℝ)
    (lindy_speed : ℝ)
    (time_to_meet : ℝ)
    (total_distance_lindy : ℝ) :
    initial_distance = 240 ∧
    jack_speed = 5 ∧
    christina_speed = 3 ∧
    lindy_speed = 9 ∧
    time_to_meet = (initial_distance / (jack_speed + christina_speed)) ∧
    total_distance_lindy = lindy_speed * time_to_meet →
    total_distance_lindy = 270 :=
by
  sorry

end lindy_distance_traveled_l2057_205708


namespace no_solutions_l2057_205730

theorem no_solutions (x : ℝ) (h : x ≠ 0) : 4 * Real.sin x - 3 * Real.cos x ≠ 5 + 1 / |x| := 
by
  sorry

end no_solutions_l2057_205730


namespace find_values_of_a_and_b_l2057_205702

theorem find_values_of_a_and_b (a b x y : ℝ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < x) (h4: 0 < y) 
  (h5 : a + b = 10) (h6 : a / x + b / y = 1) (h7 : x + y = 18) : 
  (a = 2 ∧ b = 8) ∨ (a = 8 ∧ b = 2) := 
sorry

end find_values_of_a_and_b_l2057_205702


namespace cos_triple_angle_l2057_205774

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 :=
by
  sorry

end cos_triple_angle_l2057_205774


namespace tan_pi_minus_alpha_l2057_205725

theorem tan_pi_minus_alpha 
  (α : ℝ) 
  (h1 : Real.sin α = 1 / 3) 
  (h2 : π / 2 < α) 
  (h3 : α < π) :
  Real.tan (π - α) = Real.sqrt 2 / 4 :=
by
  sorry

end tan_pi_minus_alpha_l2057_205725


namespace problem_l2057_205723

theorem problem (n : ℕ) (h : n ∣ (2^n - 2)) : (2^n - 1) ∣ (2^(2^n - 1) - 2) :=
by
  sorry

end problem_l2057_205723


namespace complement_union_l2057_205700

def is_pos_int_less_than_9 (x : ℕ) : Prop := x > 0 ∧ x < 9

def U : Set ℕ := {x | is_pos_int_less_than_9 x}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_union :
  (U \ (M ∪ N)) = {2, 4, 8} :=
by
  sorry

end complement_union_l2057_205700


namespace train_length_55_meters_l2057_205739

noncomputable def V_f := 47 * 1000 / 3600 -- Speed of the faster train in m/s
noncomputable def V_s := 36 * 1000 / 3600 -- Speed of the slower train in m/s
noncomputable def t := 36 -- Time in seconds

theorem train_length_55_meters (L : ℝ) (Vf : ℝ := V_f) (Vs : ℝ := V_s) (time : ℝ := t) :
  (2 * L = (Vf - Vs) * time) → L = 55 :=
by
  sorry

end train_length_55_meters_l2057_205739


namespace x12_is_1_l2057_205753

noncomputable def compute_x12 (x : ℝ) (h : x + 1 / x = Real.sqrt 5) : ℝ :=
  x ^ 12

theorem x12_is_1 (x : ℝ) (h : x + 1 / x = Real.sqrt 5) : compute_x12 x h = 1 :=
  sorry

end x12_is_1_l2057_205753


namespace polynomial_roots_to_determinant_l2057_205761

noncomputable def determinant_eq (a b c m p q : ℂ) : Prop :=
  (Matrix.det ![
    ![a, 1, 1],
    ![1, b, 1],
    ![1, 1, c]
  ] = 2 - m - q)

theorem polynomial_roots_to_determinant (a b c m p q : ℂ) 
  (h1 : Polynomial.eval a (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  (h2 : Polynomial.eval b (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  (h3 : Polynomial.eval c (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  : determinant_eq a b c m p q :=
by sorry

end polynomial_roots_to_determinant_l2057_205761


namespace power_sums_equal_l2057_205741

theorem power_sums_equal (x y a b : ℝ)
  (h1 : x + y = a + b)
  (h2 : x^2 + y^2 = a^2 + b^2) :
  ∀ n : ℕ, x^n + y^n = a^n + b^n :=
by
  sorry

end power_sums_equal_l2057_205741
