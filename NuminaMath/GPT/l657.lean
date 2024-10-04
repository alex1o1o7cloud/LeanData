import Mathlib

namespace intersection_M_N_l657_657036

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := { x | 2 * x > 7 }

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657036


namespace min_value_f_l657_657705

noncomputable def f (x : ℝ) : ℝ := 4 * real.sqrt x + 4 / x + 1 / (x^2)

theorem min_value_f : ∃ x > 0, f x = 9 :=
by
  use 1
  split
  · exact one_pos
  · unfold f
    rw [real.sqrt_one, one_pow, one_pow]
    norm_num

end min_value_f_l657_657705


namespace ab_bc_ca_fraction_l657_657484

theorem ab_bc_ca_fraction (a b c : ℝ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 10) (h3 : a * b * c = 12) :
    (a * b / c) + (b * c / a) + (c * a / b) = -17 / 3 := 
    sorry

end ab_bc_ca_fraction_l657_657484


namespace quadratic_factorization_l657_657917

theorem quadratic_factorization (a b : ℕ) (h1 : x^2 - 20 * x + 96 = (x - a) * (x - b)) (h2 : a > b) : 2 * b - a = 4 :=
sorry

end quadratic_factorization_l657_657917


namespace typing_time_together_l657_657886

-- Define the times it takes for Randy and Candy individually.
def T_R : ℕ := 30
def T_C : ℕ := 45

-- Define the inverse work rate for Randy and Candy, and then the combined work rate.
def randyWorkRate := 1 / (T_R : ℝ)
def candyWorkRate := 1 / (T_C : ℝ)
def combinedWorkRate := randyWorkRate + candyWorkRate

-- Define the total time it should take for them to type the paper together.
def T_total := 1 / combinedWorkRate

-- The theorem stating the total time to type the paper together.
theorem typing_time_together :
  T_total = 18 :=
by
  sorry

end typing_time_together_l657_657886


namespace complement_of_M_in_U_is_1_4_l657_657856

-- Define U
def U : Set ℕ := {x | x < 5 ∧ x ≠ 0}

-- Define M
def M : Set ℕ := {x | x^2 - 5 * x + 6 = 0}

-- The complement of M in U
def complement_U_M : Set ℕ := U \ M

-- The theorem to prove
theorem complement_of_M_in_U_is_1_4 : complement_U_M = {1, 4} := 
by sorry

end complement_of_M_in_U_is_1_4_l657_657856


namespace intersection_eq_l657_657028

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_eq : M ∩ N = {5, 7, 9} := sorry

end intersection_eq_l657_657028


namespace probability_of_one_common_point_l657_657376

def f1 (x : ℝ) : ℝ := -x
def f2 (x : ℝ) : ℝ := -1 / x
def f3 (x : ℝ) : ℝ := x^3
def f4 (x : ℝ) : ℝ := x^(1 / 2)

noncomputable def event_probability : ℚ :=
  let functions := [f1, f2, f3, f4]
  let pairs := functions.product functions |>.filter (λ pair, pair.1 ≠ pair.2)
  let valid_pairs := pairs.filter (λ pair, (∃ x : ℝ, pair.1 x = pair.2 x) ∧
    (¬∃ x y : ℝ, x ≠ y ∧ pair.1 x = pair.2 x ∧ pair.1 y = pair.2 y))
  ⟨valid_pairs.length, pairs.length⟩

theorem probability_of_one_common_point :
    event_probability = 1 / 3 :=
by
  sorry

end probability_of_one_common_point_l657_657376


namespace pipes_fill_tank_in_10_hours_l657_657200

noncomputable def time_to_fill_tank (rateA rateB rateC : ℝ) (timeB : ℝ) : ℝ :=
  (rateA + rateB + rateC)⁻¹

theorem pipes_fill_tank_in_10_hours :
  ∀ (rateB : ℝ),
    rateB = (1/35) →
    time_to_fill_tank (2 * rateB) rateB (rateB / 2) 35 = 10 :=
by
  intros rateB h_rateB
  rw [time_to_fill_tank, h_rateB]
  have h_rateA := 2 * (1 / 35)
  have h_rateC := 1 / 70
  rw [h_rateA, h_rateB, h_rateC]
  norm_num
  sorry

end pipes_fill_tank_in_10_hours_l657_657200


namespace find_a_l657_657858

def set_U (a : ℤ) : Set ℤ := {3, a, a^2 + 2*a - 3}
def set_A : Set ℤ := {2, 3}
def complement_of_A_in_U (a : ℤ) : Set ℤ := {5}

theorem find_a : ∀ (a : ℤ), 5 ∈ complement_of_A_in_U a → a = 2 :=
begin
  sorry
end

end find_a_l657_657858


namespace count_negative_values_of_x_l657_657309

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l657_657309


namespace area_of_circle_from_polar_eq_l657_657113

theorem area_of_circle_from_polar_eq : 
  let r (θ : ℝ) := 4 * Real.cos θ - 3 * Real.sin θ in
  (∃ c : ℝ × ℝ, (∀ θ : ℝ, 
    (let x := r θ * Real.cos θ in
     let y := r θ * Real.sin θ in
     (x - c.1)^2 + (y - c.2)^2 = (5 / 2)^2)) ∧ 
  (π * (5 / 2)^2 = 25 / 4 * π)) :=
by {
  sorry
}

end area_of_circle_from_polar_eq_l657_657113


namespace collinear_condition_l657_657192

theorem collinear_condition {a b c d : ℝ} (h₁ : a < b) (h₂ : c < d) (h₃ : a < d) (h₄ : c < b) :
  (a / d) + (c / b) = 1 := 
sorry

end collinear_condition_l657_657192


namespace distance_between_A_and_B_l657_657459

noncomputable def polar_equation_C1 : (ρ θ : ℝ) → Prop :=
  ρ = 2 * Real.cos θ

noncomputable def polar_equation_C2 : (ρ θ : ℝ) → Prop :=
  ρ^2 * (1 + 2 * (Real.sin θ)^2) = 3

def intersection_distance (θ : ℝ) : ℝ :=
  let ρ1 := 2 * Real.cos θ in
  let ρ2 := Real.sqrt (3 / (1 + 2 * (Real.sin θ)^2)) in
  |ρ1 - ρ2|

theorem distance_between_A_and_B :
  intersection_distance (Real.pi / 3) = |1 - Real.sqrt (30) / 5| :=
by
  have ρ1 : ℝ := 2 * Real.cos (Real.pi / 3)
  have ρ2 : ℝ := Real.sqrt(3 / (1 + 2 * (Real.sin (Real.pi / 3))^2))
  rw [intersection_distance, ρ1, ρ2]
  sorry

end distance_between_A_and_B_l657_657459


namespace num_pairs_le_l657_657433

-- Define the context and conditions
variables {m n : ℕ}
variables (G : Type) [Fintype G] (boys girls : G → bool)

-- Define the edges based on knowing relationship
variable (knows : G → G → Prop)

-- Define conditions given in the problem
def condition_1 := Fintype.card {g | girls g} = m
def condition_2 := Fintype.card {g | boys g} = n
def condition_3 := ∀ g₁ g₂ b₁ b₂, 
  girls g₁ → girls g₂ → boys b₁ → boys b₂ →
  (knows g₁ b₁ ∧ knows g₂ b₂) ∨ (knows g₁ b₂ ∧ knows g₂ b₁) → false

-- Define the statement to prove
theorem num_pairs_le (knows : G → G → Prop) (condition_1 : condition_1) 
   (condition_2 : condition_2) (condition_3 : condition_3 knows) :
   ∃ E, (∀ x y, knows x y → (boys x ∧ girls y) ∨ (boys y ∧ girls x)) ∧ Fintype.card E ≤ m + (n * (n - 1)) / 2 :=
sorry

end num_pairs_le_l657_657433


namespace altitudes_not_form_triangle_l657_657596

theorem altitudes_not_form_triangle (h₁ h₂ h₃ : ℝ) :
  ¬(h₁ = 5 ∧ h₂ = 12 ∧ h₃ = 13 ∧ ∃ a b c : ℝ, a * h₁ = b * h₂ ∧ b * h₂ = c * h₃ ∧
    a < b + c ∧ b < a + c ∧ c < a + b) :=
by sorry

end altitudes_not_form_triangle_l657_657596


namespace derivative_y_l657_657537

variable (x : ℝ)
def y : ℝ := x^2 * Real.cos x

theorem derivative_y : deriv y x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by
  sorry

end derivative_y_l657_657537


namespace min_c_reach_max_at_zero_l657_657216

-- Define the general cosine function and conditions
def cos_function (a b c x : ℝ) : ℝ := a * Real.cos (b * x + c)

-- Mathematical statement to prove
theorem min_c_reach_max_at_zero :
  ∀ (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0),
  ∃ (c : ℝ), c = 0 ∧ 
    (∀ x : ℝ, x = 0 → cos_function a b c x = a) :=
by
  sorry

end min_c_reach_max_at_zero_l657_657216


namespace polynomial_roots_sum_l657_657498

noncomputable def roots (p : Polynomial ℚ) : Set ℚ := {r | p.eval r = 0}

theorem polynomial_roots_sum :
  ∀ a b c : ℚ, (a ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  (b ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  (c ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  a ≠ b → b ≠ c → a ≠ c →
  (a + b + c = 8) →
  (a * b + a * c + b * c = 7) →
  (a * b * c = -3) →
  (a / (b * c + 1) + b / (a * c + 1) + c / (a * b + 1) = 17 / 2) := by
    intros a b c ha hb hc hab habc hac sum_nums sum_prods prod_roots
    sorry

#check polynomial_roots_sum

end polynomial_roots_sum_l657_657498


namespace equal_and_real_roots_l657_657679

theorem equal_and_real_roots 
  (a c : ℝ) 
  (h : ∆ := (-4 * √3)^2 - 4 * a * c = 0) 
  (discriminant_zero : h = 0) : 
  let roots := λ (a b c : ℝ), ((-b + Real.sqrt h) / (2 * a), (-b - Real.sqrt h) / (2 * a)) in 
  (roots a (-4 * √3) c) = (roots a (-4 * √3) c) ∧ 
  ∃ x : ℝ, (roots a (-4 * √3) c).1 = x ∧ (roots a (-4 * √3) c).2 = x :=
by {
  sorry
}

end equal_and_real_roots_l657_657679


namespace Sn_formula_l657_657969

def T (k : ℕ) : ℕ := -- A function to count the number of "11" pairs in binary representation of k.
  if k = 1 ∨ k = 2 ∨ k = 4 ∨ k = 5 then
    0
  else if k = 3 ∨ k = 6 then
    1
  else if k = 7 then
    2
  else
    0 -- Given initial values; a full implementation would require a function for binary pair counting.

def S : ℕ → ℕ
| 0 := 0
| (n + 1) := ∑ k in Finset.range (2^(n+1)), T k

theorem Sn_formula (n : ℕ) (h : 2 ≤ n) : S n = 2^(n-2) * (n-1) := by
  sorry

end Sn_formula_l657_657969


namespace range_of_fx_a_eq_2_range_of_a_increasing_fx_l657_657762

-- Part (1)
theorem range_of_fx_a_eq_2 (x : ℝ) (h : x ∈ Set.Icc (-2 : ℝ) (3 : ℝ)) :
  ∃ y ∈ Set.Icc (-21 / 4 : ℝ) (15 : ℝ), y = x^2 + 3 * x - 3 :=
sorry

-- Part (2)
theorem range_of_a_increasing_fx (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) (3 : ℝ) → 2 * x + 2 * a - 1 ≥ 0) ↔ a ∈ Set.Ici (3 / 2 : ℝ) :=
sorry

end range_of_fx_a_eq_2_range_of_a_increasing_fx_l657_657762


namespace chord_length_is_2_l657_657364

-- Define the circle M's center moving along the parabola
def circle_center (a : ℝ) : ℝ × ℝ := (a, (1/2) * a^2)

-- The radius of the circle M passing through the fixed point (0, 1)
def radius (a : ℝ) : ℝ := real.sqrt (a^2 + ((1/2) * a^2 - 1)^2)

-- The equation of the circle M centered at (a, 1/2 * a^2)
def circle_eq (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - (1/2) * a^2)^2 = radius a ^ 2

-- Finding the x-intercepts when y = 0
def x_intercepts (a : ℝ) : ℝ × ℝ :=
  let b := a^2 + ((1/2) * a^2 - 1)^2 in
  (real.sqrt b + a, a - real.sqrt b)

-- Condition for chord length |PQ| intercepted by the x-axis
def chord_length (a : ℝ) : ℝ :=
  let (x1, x2) := x_intercepts a in
  real.abs (x1 - x2)

-- The main theorem stating the chord length
theorem chord_length_is_2 (a : ℝ) :
  chord_length a = 2 :=
sorry

end chord_length_is_2_l657_657364


namespace regular_polygon_sides_l657_657641

theorem regular_polygon_sides (ex_angle : ℝ) (hne_zero : ex_angle ≠ 0)
  (sum_ext_angles : ∀ (n : ℕ), n > 2 → n * ex_angle = 360) :
  ∃ (n : ℕ), n * 15 = 360 ∧ n = 24 :=
by 
  sorry

end regular_polygon_sides_l657_657641


namespace train_speed_l657_657647

noncomputable def speed_of_train_kmh (time_to_pass_pole: ℝ) (platform_length: ℝ) 
(time_to_pass_platform: ℝ) : ℝ := 
let train_speed_mps := platform_length / (time_to_pass_platform - time_to_pass_pole) in
train_speed_mps * 3.6

theorem train_speed (h1 : time_to_pass_pole = 15) 
(h2 : platform_length = 380) 
(h3 : time_to_pass_platform = 52.99696024318054) : 
speed_of_train_kmh time_to_pass_pole platform_length time_to_pass_platform = 36.0037908 := 
by
  simp [speed_of_train_kmh, h1, h2, h3]
  /- calculation proof here providing the exact steps above to verify
  10.001053 * 3.6 = 36.0037908
  -/
  sorry

end train_speed_l657_657647


namespace find_a_l657_657116

-- Define the prime number theorem, essential for the proof
theorem find_a (a b p : ℤ) (hp : p.prime)
  (h1 : p^2 + a * p + b = 0)
  (h2 : p^2 + b * p + 1100 = 0) :
  a = 274 ∨ a = 40 :=
by
  sorry

end find_a_l657_657116


namespace probability_red_ball_l657_657429

-- Let P_red be the probability of drawing a red ball.
-- Let P_white be the probability of drawing a white ball.
-- Let P_black be the probability of drawing a black ball.
-- Let P_red_or_white be the probability of drawing a red or white ball.
-- Let P_red_or_black be the probability of drawing a red or black ball.

variable (P_red P_white P_black : ℝ)
variable (P_red_or_white P_red_or_black : ℝ)

-- Given conditions
axiom P_red_or_white_condition : P_red_or_white = 0.58
axiom P_red_or_black_condition : P_red_or_black = 0.62

-- The total probability must sum to 1.
axiom total_probability_condition : P_red + P_white + P_black = 1

-- Prove that the probability of drawing a red ball is 0.2.
theorem probability_red_ball : P_red = 0.2 :=
by
  -- To be proven
  sorry

end probability_red_ball_l657_657429


namespace p_necessary_for_q_q_not_necessary_for_p_l657_657499

variable (f : ℝ → ℝ) (a : ℝ)

def p : Prop := ∀ x > 0, f x = ln x + x^2 + a * x + 1 → deriv f x ≥ 0
def q : Prop := a ≥ -2

theorem p_necessary_for_q : p f a → q a := 
by sorry

theorem q_not_necessary_for_p : q a → ¬ p f a := 
by sorry

end p_necessary_for_q_q_not_necessary_for_p_l657_657499


namespace dog_cat_food_difference_l657_657188

theorem dog_cat_food_difference :
  let dogFood := 600
  let catFood := 327
  dogFood - catFood = 273 :=
by
  let dogFood := 600
  let catFood := 327
  show dogFood - catFood = 273
  sorry

end dog_cat_food_difference_l657_657188


namespace michael_truck_meetings_2_times_l657_657868

/-- Michael walks at a rate of 6 feet per second on a straight path. 
Trash pails are placed every 240 feet along the path. 
A garbage truck traveling at 12 feet per second in the same direction stops for 40 seconds at each pail. 
When Michael passes a pail, he sees the truck, which is 240 feet ahead, just leaving the next pail. 
Prove that Michael and the truck will meet exactly 2 times. -/

def michael_truck_meetings (v_michael v_truck d_pail t_stop init_michael init_truck : ℕ) : ℕ := sorry

theorem michael_truck_meetings_2_times :
  michael_truck_meetings 6 12 240 40 0 240 = 2 := 
  sorry

end michael_truck_meetings_2_times_l657_657868


namespace cost_of_fencing_l657_657131
-- Importing necessary Lean libraries

-- Definitions based on given conditions
def length (x : ℝ) := 5 * x
def width (x : ℝ) := 3 * x
def area (x : ℝ) := length x * width x
def perimeter (x : ℝ) := 2 * (length x + width x)
def cost_per_meter : ℝ := 0.75

-- Given condition for the area of the park
def x : ℝ := real.sqrt (9000 / 15)

-- Theorem to prove the cost of fencing the park
theorem cost_of_fencing : perimeter x * cost_per_meter ≈ 293.88 := by
  sorry

end cost_of_fencing_l657_657131


namespace find_third_vertex_l657_657968

variables {y : ℝ}

def is_obtuse_triangle (v1 v2 v3 : ℝ × ℝ) : Prop :=
  let ⟨x1, y1⟩ := v1
  let ⟨x2, y2⟩ := v2
  let ⟨x3, y3⟩ := v3 in
  (x1 = 8 ∧ y1 = 6) ∧ (x2 = 0 ∧ y2 = 0) ∧ ∃ y, y < 0 ∧ (x3 = 0 ∧ y3 = y) ∧ 
  (1/2 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) = 48)

theorem find_third_vertex (y : ℝ) (h : is_obtuse_triangle (8, 6) (0, 0) (0, y)) : y = -12 :=
sorry

end find_third_vertex_l657_657968


namespace find_x_for_h_eq_20_l657_657850

def h (x : ℝ) : ℝ := 2 * real.inv_fun f x
def f (x : ℝ) : ℝ := 30 / (x + 4)

theorem find_x_for_h_eq_20 : ∃ x : ℝ, h x = 20 ∧ x = 15 / 7 :=
begin
  sorry
end

end find_x_for_h_eq_20_l657_657850


namespace system_of_equations_solution_l657_657896

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (2 * x - 3 * y = 1) ∧ 
    (5 * x + 4 * y = 6) ∧ 
    (x + 2 * y = 2) ∧
    x = 2 / 3 ∧ y = 2 / 3 :=
by {
  sorry
}

end system_of_equations_solution_l657_657896


namespace sum_of_sequence_l657_657244

theorem sum_of_sequence (n : ℕ) : 
  (∑ k in finset.range n, (∑ j in finset.range k, 10 ^ j)) = (10 * (10^n - 1) - 9 * n) / 81 := 
sorry

end sum_of_sequence_l657_657244


namespace ln_x_gt_1_sufficient_but_not_necessary_l657_657239

theorem ln_x_gt_1_sufficient_but_not_necessary (x : ℝ) (h1 : x > 0) : 
  (ln x > 1 → x > 1) ∧ ¬ (x > 1 → ln x > 1) :=
by {
  sorry
}

end ln_x_gt_1_sufficient_but_not_necessary_l657_657239


namespace sum_of_qualified_primes_eq_388_l657_657151

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def digits_reversed (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

noncomputable def qualified_primes : List ℕ :=
  List.filter (λ n, 20 < n ∧ n < 99 ∧ is_prime n ∧ is_prime (digits_reversed n))
    (List.range' 21 78)

theorem sum_of_qualified_primes_eq_388 :
  List.sum qualified_primes = 388 := 
sorry

end sum_of_qualified_primes_eq_388_l657_657151


namespace ellipse_properties_l657_657924

theorem ellipse_properties {a c b : ℝ} (h_a : 2 * a = 6) (h_e : c / a = 2 / 3) (h_b : b ^ 2 = a ^ 2 - c ^ 2) :
  (∀ x y, (x^2 / 9 + y^2 / 5 = 1) ↔ (a = 3 ∧ c = 2 ∧ b^2 = 5 ∧ ellipse_eq : ∀ x y, x^2 / 9 + y^2 / 5 = 1)) ∧ 
  (chord_length : |(((-9 : ℝ)/7) + sqrt(130)/7) - ((-9 : ℝ)/7) - sqrt(130)/7)| = 3 / 7 * sqrt(130)) :=
by
  sorry

end ellipse_properties_l657_657924


namespace remainder_of_arithmetic_sequence_sum_l657_657575

theorem remainder_of_arithmetic_sequence_sum (a d l : ℕ) (h1 : a = 1) (h2 : d = 8) (h3 : l = 313) :
  let n := (l - a) / d + 1 in
  let S := n * (2 * a + (n - 1) * d) / 2 in
  S % 8 = 0 :=
by
  -- Proof will go here
  sorry

end remainder_of_arithmetic_sequence_sum_l657_657575


namespace probability_closer_to_eight_l657_657635

noncomputable def probability_point_closer_to_eight (x : ℝ) : ℚ :=
if 0 ≤ x ∧ x ≤ 8 then 
  if x > 4 then 1 else 0
else 0

theorem probability_closer_to_eight : 
  (∫ x in 0..8, probability_point_closer_to_eight x) / ∫ x in 0..8, 1 = (1 : ℚ) / 2 :=
sorry

end probability_closer_to_eight_l657_657635


namespace correct_answer_l657_657722

noncomputable def f : ℝ → ℝ := sorry

variables {x : ℝ}

-- Condition 1: Symmetry
axiom f_symm : ∀ x : ℝ, f(x) = f(4 - x)

-- Condition 2: f is increasing on [2, + ∞)
axiom f_increasing : ∀ {a b : ℝ}, 2 ≤ a → a ≤ b → f(a) ≤ f(b)

-- Proof goal
theorem correct_answer : f(4) > f(0.5) ∧ f(0.5) > f(1) :=
by
  sorry

end correct_answer_l657_657722


namespace tax_percent_is_4_8_l657_657513

def tax_percent_of_total_amount (total_amount : ℝ) : ℝ :=
  let clothing_spent := 0.60 * total_amount
  let food_spent := 0.10 * total_amount
  let other_items_spent := 0.30 * total_amount
  let clothing_tax := 0.04 * clothing_spent
  let food_tax := 0
  let other_items_tax := 0.08 * other_items_spent
  let total_tax := clothing_tax + food_tax + other_items_tax
  (total_tax / total_amount) * 100

theorem tax_percent_is_4_8 (total_amount : ℝ) : tax_percent_of_total_amount total_amount = 4.8 :=
by
  sorry

end tax_percent_is_4_8_l657_657513


namespace lines_divide_plane_l657_657680

theorem lines_divide_plane (n : ℕ) :
  let R : ℕ → ℕ := λ k, k * (k - 1) / 2 + 1 in
  R n ∧ R (n + 1) :=
by
  sorry

end lines_divide_plane_l657_657680


namespace location_of_A_total_fuel_consumption_l657_657999

def travel_record : List Int := [+9, -8, +6, -15, +6, -14, +4, -3]

def fuel_consumption (a : Int) : Int :=
  65 * a

theorem location_of_A :
  List.sum travel_record = -15 := sorry

theorem total_fuel_consumption (a : Int) :
  List.sum (List.map Int.natAbs travel_record) * a = 65 * a := sorry

end location_of_A_total_fuel_consumption_l657_657999


namespace A_alone_work_days_l657_657175

noncomputable def A_and_B_together : ℕ := 40
noncomputable def A_and_B_worked_together_days : ℕ := 10
noncomputable def B_left_and_C_joined_after_days : ℕ := 6
noncomputable def A_and_C_finish_remaining_work_days : ℕ := 15
noncomputable def C_alone_work_days : ℕ := 60

theorem A_alone_work_days (h1 : A_and_B_together = 40)
                          (h2 : A_and_B_worked_together_days = 10)
                          (h3 : B_left_and_C_joined_after_days = 6)
                          (h4 : A_and_C_finish_remaining_work_days = 15)
                          (h5 : C_alone_work_days = 60) : ∃ (n : ℕ), n = 30 :=
by {
  sorry -- Proof goes here
}

end A_alone_work_days_l657_657175


namespace total_grocery_packs_l657_657862

variable {packs_cookies : Nat} {indiv_per_cookie_pack : Nat} 
variable {packs_noodles : Nat} {indiv_per_noodle_pack : Nat}
variable {packs_juice : Nat} {bottles_per_juice_pack : Nat}
variable {packs_snacks : Nat} {indiv_per_snack_pack : Nat}

theorem total_grocery_packs 
  (packs_cookies = 3) (indiv_per_cookie_pack = 4) 
  (packs_noodles = 4) (indiv_per_noodle_pack = 8)
  (packs_juice = 5) (bottles_per_juice_pack = 6)
  (packs_snacks = 2) (indiv_per_snack_pack = 10) :
  packs_cookies * indiv_per_cookie_pack + 
  packs_noodles * indiv_per_noodle_pack + 
  packs_juice * bottles_per_juice_pack + 
  packs_snacks * indiv_per_snack_pack = 94 := by
  sorry

end total_grocery_packs_l657_657862


namespace translate_parabola_l657_657960

theorem translate_parabola :
  (∀ x, y = 1/2 * x^2 + 1 → y = 1/2 * (x - 1)^2 - 2) :=
by
  sorry

end translate_parabola_l657_657960


namespace probability_at_least_four_same_value_l657_657276

-- Define the event of rolling five fair eight-sided dice
def roll_outcomes : ℕ := 8 -- There are 8 possible outcomes per die

-- Calculate the probability
theorem probability_at_least_four_same_value :
  let at_least_four_same_value := (35 / (8^4)) + (1 / (8^4))
  in at_least_four_same_value = (9 / 1024) :=
by
  let p1 := 1 / roll_outcomes^4
  let p2 := 5 * (1 / roll_outcomes^3) * (7 / roll_outcomes)
  have h : p1 + p2 = 36 / 4096 := sorry
  have h_uniform := (9 : ℚ) / 1024
  -- Ensure that the calculated probability matches the expected value
  show h_uniform = (p1 + p2)
  sorry

end probability_at_least_four_same_value_l657_657276


namespace imaginary_part_of_complex_number_l657_657921

open Complex

-- Problem Statement
theorem imaginary_part_of_complex_number : 
  ∀ (i : ℂ), i^2 = -1 → let z := i^2 * (1 + i) in Complex.im z = -1 :=
by
  intros i hi h
  -- conditions are directly used from the problem statement
  sorry

end imaginary_part_of_complex_number_l657_657921


namespace num_of_negative_x_l657_657294

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l657_657294


namespace group_count_4_men_5_women_l657_657454

theorem group_count_4_men_5_women : 
  let men := 4
  let women := 5
  let groups := List.replicate 3 (3, true)
  ∃ (m_w_combinations : List (ℕ × ℕ)),
    m_w_combinations = [(1, 2), (2, 1)] ∧
    ((men.choose m_w_combinations.head.fst * women.choose m_w_combinations.head.snd) * (men - m_w_combinations.head.fst).choose m_w_combinations.tail.head.fst * (women - m_w_combinations.head.snd).choose m_w_combinations.tail.head.snd) = 360 :=
by
  sorry

end group_count_4_men_5_women_l657_657454


namespace median_inequality_l657_657000

variables {α : ℝ} (A B C M : Point) (a b c : ℝ)

-- Definitions and conditions
def isTriangle (A B C : Point) : Prop := -- definition of triangle
sorry

def isMedian (A B C M : Point) : Prop := -- definition of median
sorry

-- Statement we want to prove
theorem median_inequality (h1 : isTriangle A B C) (h2 : isMedian A B C M) :
  2 * AM ≥ (b + c) * Real.cos (α / 2) :=
sorry

end median_inequality_l657_657000


namespace balance_in_equilibrium_l657_657144

-- Definitions of the conditions
def is_integer (n : ℕ) := ∃ m : ℤ, n = m.nat_abs

variable (weights : List ℕ)
variable (total_mass : ℕ)
variable (num_weights : ℕ)

-- Statement of the problem
theorem balance_in_equilibrium 
  (h_mass : total_mass = 200)
  (h_num_weights : num_weights = 101)
  (h_weights_sum : weights.sum = 200)
  (h_weights_integer : ∀ (w : ℕ), w ∈ weights → is_integer w)
  (descending_order : ∀ (i j : ℕ), i < j → weights.get i > weights.get j) :
  ∃ (left_pan right_pan : List ℕ), 
    (left_pan.sum = right_pan.sum) ∧ (left_pan ++ right_pan = weights) :=
sorry

end balance_in_equilibrium_l657_657144


namespace solve_root_equation_l657_657261

noncomputable def sqrt4 (x : ℝ) : ℝ := x^(1/4)

theorem solve_root_equation (x : ℝ) :
  sqrt4 (43 - 2 * x) + sqrt4 (39 + 2 * x) = 4 ↔ x = 21 ∨ x = -13.5 :=
by
  sorry

end solve_root_equation_l657_657261


namespace randy_money_left_after_expenses_l657_657087

theorem randy_money_left_after_expenses : 
  ∀ (initial_money lunch_cost : ℕ) (ice_cream_fraction : ℚ), 
  initial_money = 30 → 
  lunch_cost = 10 → 
  ice_cream_fraction = 1 / 4 → 
  let post_lunch_money := initial_money - lunch_cost in
  let ice_cream_cost := ice_cream_fraction * post_lunch_money in
  let money_left := post_lunch_money - ice_cream_cost in
  money_left = 15 :=
by
  intros initial_money lunch_cost ice_cream_fraction
  assume h_initial h_lunch h_fraction
  let post_lunch_money := initial_money - lunch_cost
  let ice_cream_cost := ice_cream_fraction * post_lunch_money
  let money_left := post_lunch_money - ice_cream_cost
  sorry

end randy_money_left_after_expenses_l657_657087


namespace group_count_4_men_5_women_l657_657453

theorem group_count_4_men_5_women : 
  let men := 4
  let women := 5
  let groups := List.replicate 3 (3, true)
  ∃ (m_w_combinations : List (ℕ × ℕ)),
    m_w_combinations = [(1, 2), (2, 1)] ∧
    ((men.choose m_w_combinations.head.fst * women.choose m_w_combinations.head.snd) * (men - m_w_combinations.head.fst).choose m_w_combinations.tail.head.fst * (women - m_w_combinations.head.snd).choose m_w_combinations.tail.head.snd) = 360 :=
by
  sorry

end group_count_4_men_5_women_l657_657453


namespace trapezoid_segment_length_l657_657911

theorem trapezoid_segment_length (a b : ℝ) : 
  ∃ x : ℝ, x = Real.sqrt ((a^2 + b^2) / 2) :=
sorry

end trapezoid_segment_length_l657_657911


namespace number_of_regions_l657_657252

theorem number_of_regions (V E F S : ℕ) (hV : V = 52) (hE : E = 300) (hF : F = 432) (hEuler : V - E + F - S = -1) :
  S = 185 := 
by
  -- proof goes here
  sorry

end number_of_regions_l657_657252


namespace graph_of_equation_l657_657267

theorem graph_of_equation :
  let eqn := (x + 2) ^ 2 * (y - 1) * (x + y + 2) = (y + 2) ^ 2 * (x - 1) * (x + y + 2)
  in ∃ l1 l2 l3 : affine_plane ℝ,
  (∃ a b : ℝ, l1 = {z | z.2 = -z.1 - 4}) ∧ 
  (∃ a b : ℝ, l2 = {z | z.2 = z.1}) ∧ 
  (∃ a b : ℝ, l3 = {z | z.2 = -z.1 - 2}) ∧ 
  (∀ l : affine_plane ℝ, (l = l1 ∨ l = l2 ∨ l = l3) → 
  (∃ p : point ℝ, p ∉ l1 ∧ p ∉ l2 ∧ p ∉ l3)) :=
by
  sorry

end graph_of_equation_l657_657267


namespace find_solution_l657_657259

open Nat

def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

noncomputable def expression (n : ℕ) : ℕ :=
  1 + binomial n 1 + binomial n 2 + binomial n 3

theorem find_solution (n : ℕ) (h : n > 3) :
  expression n ∣ 2 ^ 2000 ↔ n = 7 ∨ n = 23 :=
by
  sorry

end find_solution_l657_657259


namespace calculateBooksRemaining_l657_657140

noncomputable def totalBooksRemaining
    (initialBooks : ℕ)
    (n : ℕ)
    (a₁ : ℕ)
    (d : ℕ)
    (borrowedBooks : ℕ)
    (returnedBooks : ℕ) : ℕ :=
  let sumDonations := n * (2 * a₁ + (n - 1) * d) / 2
  let totalAfterDonations := initialBooks + sumDonations
  totalAfterDonations - borrowedBooks + returnedBooks

theorem calculateBooksRemaining :
  totalBooksRemaining 1000 15 2 2 350 270 = 1160 :=
by
  sorry

end calculateBooksRemaining_l657_657140


namespace residue_bound_l657_657829

noncomputable def numResiduesMax (n : ℕ) (h : n > 1) : ℕ :=
  card { m | ∃ x y, x < n ∧ y < n ∧ m = (x^n + y^n) % n^2 }

theorem residue_bound (n : ℕ) (h : n > 1) :
  numResiduesMax n h ≤ n * (n + 1) / 2 :=
sorry

end residue_bound_l657_657829


namespace ways_to_place_people_into_groups_l657_657441

theorem ways_to_place_people_into_groups :
  let men := 4
  let women := 5
  ∃ (groups : Nat), groups = 2 ∧
  ∀ (g : Nat → (Fin 3 → (Bool → Nat → Nat))),
    (∀ i, i < group_counts → ∃ m w, g i m w < people ∧ g i m (1 - w) < people ∧ m + 1 - w + (1 - m) + w = 3) →
    let groups : List (List (Fin 2)) := [
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)],
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)]
    ] in
    g.mk 1 dec_trivial * g.mk 2 dec_trivial = 360 :=
sorry

end ways_to_place_people_into_groups_l657_657441


namespace t_f_6_eq_l657_657838

def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)

def f (x : ℝ) : ℝ := 6 - t x

theorem t_f_6_eq : t (f 6) = Real.sqrt 26 - 2 := by
  sorry

end t_f_6_eq_l657_657838


namespace probability_neither_squares_cubes_factorials_l657_657933

-- Define the set of numbers from 1 to 200
def numbers := Finset.range 201

-- Define the sets of perfect squares, perfect cubes, and factorials within the range
def perfect_squares := numbers.filter (λ n, ∃ k, k * k = n)
def perfect_cubes := numbers.filter (λ n, ∃ k, k * k * k = n)
def factorials := numbers.filter (λ n, ∃ k, Nat.factorial k = n)

-- Define the set of numbers that are neither perfect squares, perfect cubes, nor factorials
def neither_squares_cubes_factorials := 
  numbers.filter (λ n, n ∉ perfect_squares ∧ n ∉ perfect_cubes ∧ n ∉ factorials)

-- Total count of numbers
def total_count := numbers.card

-- Count of numbers that are neither perfect squares, perfect cubes, nor factorials
def count_neither := neither_squares_cubes_factorials.card

-- Statement of the problem
theorem probability_neither_squares_cubes_factorials :
  (count_neither : ℚ) / total_count = 179 / 200 :=
sorry

end probability_neither_squares_cubes_factorials_l657_657933


namespace g_of_x_eq_15_implies_x_eq_3_l657_657103

theorem g_of_x_eq_15_implies_x_eq_3 (g f : ℝ → ℝ) 
  (h₁ : ∀ x, g x = 3 * (f⁻¹' x))
  (h₂ : ∀ x, f x = 24 / (x + 3)) :
  ∃ x, g x = 15 ∧ x = 3 :=
begin
  sorry
end

end g_of_x_eq_15_implies_x_eq_3_l657_657103


namespace kaleb_balance_l657_657170

theorem kaleb_balance (springEarnings : ℕ) (summerEarnings : ℕ) (suppliesCost : ℕ) (totalBalance : ℕ)
  (h1 : springEarnings = 4)
  (h2 : summerEarnings = 50)
  (h3 : suppliesCost = 4)
  (h4 : totalBalance = (springEarnings + summerEarnings) - suppliesCost) : totalBalance = 50 := by
  sorry

end kaleb_balance_l657_657170


namespace probability_different_suits_l657_657864

theorem probability_different_suits (h : ∀ (c1 c2 c3 : ℕ), c1 ≠ c2 ∧ c1 ≠ c3 ∧ c2 ≠ c3 ∧ 
                                    ∀ {x}, x ∈ {c1, c2, c3} → x ∈ finset.range 52) : 
  let prob := (13 / 17) * (13 / 25) in
  prob = (169 / 425) := 
by
  sorry

end probability_different_suits_l657_657864


namespace sum_of_a_and_b_l657_657191

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

noncomputable def perimeter : ℝ :=
  distance (1, 2) (4, 5) + distance (4, 5) (5, 4) + distance (5, 4) (4, 1) + distance (4, 1) (1, 2)

theorem sum_of_a_and_b : 
  ∃ a b : ℤ, perimeter = a * real.sqrt 2 + b * real.sqrt 10 ∧ a + b = 6 :=
by
  sorry

end sum_of_a_and_b_l657_657191


namespace area_of_region_l657_657263

theorem area_of_region :
  (let closed_region := {p : ℝ × ℝ | |p.1 - 80| + |p.2| = |p.1 / 5|} in
   let horizontal_span := 100 - 80 in
   let vertical_span := 20 - (-20) in
   (1/2) * horizontal_span * vertical_span = 400) := 
sorry

end area_of_region_l657_657263


namespace numeral_in_150th_decimal_place_l657_657592

noncomputable def decimal_representation_13_17 : String :=
  "7647058823529411"

theorem numeral_in_150th_decimal_place :
  (decimal_representation_13_17.get (150 % 17)).iget = '1' :=
by
  sorry

end numeral_in_150th_decimal_place_l657_657592


namespace negative_values_count_l657_657331

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l657_657331


namespace sufficient_but_not_necessary_l657_657988

theorem sufficient_but_not_necessary (x : ℝ) : ((0 < x) → (|x-1| - |x| ≤ 1)) ∧ ((|x-1| - |x| ≤ 1) → True) ∧ ¬((|x-1| - |x| ≤ 1) → (0 < x)) := sorry

end sufficient_but_not_necessary_l657_657988


namespace inverse_eq_4_l657_657760

def f (x : ℝ) := logBase 3 ((4/x) + 2)

theorem inverse_eq_4 (h : f(4) = 1) : f⁻¹ 1 = 4 :=
by {
  unfold f at h, 
  sorry
}

end inverse_eq_4_l657_657760


namespace correct_operation_l657_657154

theorem correct_operation :
  (sqrt ((-6 : ℝ) ^ 2) = -6 = false) ∧
  (((a - b) ^ 2 = a ^ 2 - a * b + b ^ 2) = false) ∧
  (((-2 * x ^ 2) ^ 3 = -6 * x ^ 6) = false) ∧
  ((x ^ 2 * x ^ 3 = x ^ 5) = true) :=
by {
  sorry
}

end correct_operation_l657_657154


namespace negative_values_count_l657_657317

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l657_657317


namespace profit_per_unit_maximize_profit_l657_657610

-- Define profits per unit for type A and B phones
variables (a b : ℝ)

-- Define the conditions as lean statements for the given problem
def cond1 : Prop := a + b = 600
def cond2 : Prop := 3 * a + 2 * b = 1400

-- Define the statement to find profit per unit for type A and B phones
theorem profit_per_unit (h1 : cond1 a b) (h2 : cond2 a b) :
  a = 200 ∧ b = 400 := by
sorry

-- Define the number of phones purchased
variables (x y : ℕ)

-- Define conditions for total number of phones and number of type B phones
def total_phones : Prop := x + y = 20
def cond_phones : Prop := y ≤ (2 * x) / 3

-- Define the profit function
def profit := 200 * x + 400 * y

-- Define the statement to maximize profit with purchasing plan
theorem maximize_profit (h3 : total_phones x y) (h4 : cond_phones x) :
  profit x y = 5600 ∧ x = 12 ∧ y = 8 := by
sorry

end profit_per_unit_maximize_profit_l657_657610


namespace findDivisor_l657_657633

def addDivisorProblem : Prop :=
  ∃ d : ℕ, ∃ n : ℕ, n = 172835 + 21 ∧ d ∣ n ∧ d = 21

theorem findDivisor : addDivisorProblem :=
by
  sorry

end findDivisor_l657_657633


namespace train_passes_man_in_8_seconds_l657_657202

def length_of_train : ℝ := 186
def length_of_platform : ℝ := 279
def time_to_cross_platform : ℝ := 20
def total_distance := length_of_train + length_of_platform
def speed_of_train := total_distance / time_to_cross_platform
def time_to_pass_man := length_of_train / speed_of_train

theorem train_passes_man_in_8_seconds : time_to_pass_man = 8 := by
  sorry

end train_passes_man_in_8_seconds_l657_657202


namespace group_division_l657_657446

theorem group_division (men women : ℕ) : 
  men = 4 → women = 5 →
  (∃ g1 g2 g3 : set (fin $ men + women), 
    g1.card = 3 ∧ g2.card = 3 ∧ g3.card = 3 ∧ 
    (∀ g, g ∈ [g1, g2, g3] → (∃ m w : ℕ, 1 ≤ m ∧ 1 ≤ w ∧ 
      finset.card (finset.filter (λ x, x < men) g) = m ∧ 
      finset.card (finset.filter (λ x, x ≥ men) g) = w)) 
    ∧ finset.disjoint g1 g2 ∧ finset.disjoint g2 g3 ∧ finset.disjoint g3 g1 
    ∧ g1 ∪ g2 ∪ g3 = finset.univ (fin $ men + women)) → 
  finset.card (finset.powerset' 3 (finset.univ (fin $ men + women))) / 2 = 180 :=
begin
  intros hmen hwomen,
  sorry
end

end group_division_l657_657446


namespace find_s_l657_657835

noncomputable def is_monic (p : Polynomial ℝ) : Prop :=
  p.leadingCoeff = 1

variables (f g : Polynomial ℝ) (s : ℝ)
variables (r1 r2 r3 r4 r5 r6 : ℝ)

-- Conditions
def conditions : Prop :=
  is_monic f ∧ is_monic g ∧
  (f.roots = [s + 2, s + 8, r1] ∨ f.roots = [s + 8, s + 2, r1] ∨ f.roots = [s + 2, r1, s + 8] ∨
   f.roots = [r1, s + 2, s + 8] ∨ f.roots = [r1, s + 8, s + 2]) ∧
  (g.roots = [s + 4, s + 10, r2] ∨ g.roots = [s + 10, s + 4, r2] ∨ g.roots = [s + 4, r2, s + 10] ∨
   g.roots = [r2, s + 4, s + 10] ∨ g.roots = [r2, s + 10, s + 4]) ∧
  ∀ (x : ℝ), f.eval x - g.eval x = 2 * s

-- Theorem statement

theorem find_s (h : conditions f g r1 r2 s) : s = 288 / 14 :=
sorry

end find_s_l657_657835


namespace sqrt_meaningful_range_iff_l657_657788

noncomputable def sqrt_meaningful_range (x : ℝ) : Prop :=
  (∃ r : ℝ, r ≥ 0 ∧ r * r = x - 2023)

theorem sqrt_meaningful_range_iff {x : ℝ} : sqrt_meaningful_range x ↔ x ≥ 2023 :=
by
  sorry

end sqrt_meaningful_range_iff_l657_657788


namespace last_two_nonzero_digits_of_80_fact_eq_8_l657_657930

theorem last_two_nonzero_digits_of_80_fact_eq_8 : 
  let m := lastTwoNonZeroDigits (Nat.factorial 80)
  in m = 8 := 
by
  sorry

end last_two_nonzero_digits_of_80_fact_eq_8_l657_657930


namespace possible_values_of_a_l657_657387

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_g : ∀ x : ℝ, g (-x) = g x
axiom f_g_relation : ∀ x : ℝ, f x + g x = a * x ^ 2 + x + 2
axiom g_condition : ∀ (x1 x2 : ℝ), 1 < x1 → x1 < x2 → x2 < 2 → (g (x1) - g (x2)) / (x1 - x2) > -2

theorem possible_values_of_a : a ≥ -1 / 2 := 
sorry

end possible_values_of_a_l657_657387


namespace avg_age_calculation_correct_l657_657909

noncomputable def avg_age_before_new_students (N : ℕ) (new_students : ℕ) (new_avg_age : ℕ) (decrease_in_avg : ℕ) : ℕ :=
let orig_avg_age := (((N + new_students) * (new_avg_age + decrease_in_avg)) - (new_students * new_avg_age)) / N in
orig_avg_age

theorem avg_age_calculation_correct:
  avg_age_before_new_students 12 12 32 4 = 40 :=
by
  sorry -- The proof goes here

end avg_age_calculation_correct_l657_657909


namespace measure_angle_B_l657_657002

-- Assume the necessary geometric entities and conditions stated in the problem
variables {A B C A1 C1 H M : Type} [Geometry]

-- Conditions given in the problem
axiom triangle_ABC : Triangle A B C
axiom altitude_AA1 : Altitude A A1 B C
axiom altitude_CC1 : Altitude C C1 A B
axiom intersect_at_H : Intersect AA1 CC1 = {H}
axiom H_inside_triangle : LiesInside H (Triangle A B C)
axiom H_midpoint_AA1 : Midpoint H A A1
axiom CH_HC1_2_1 : SegmentRatio (CH) (HC1) = 2 / 1

-- Theorem statement: Measure of angle B is 45 degrees
theorem measure_angle_B : angle B = 45 := 
by sorry

end measure_angle_B_l657_657002


namespace sum_of_valid_primes_is_2462_l657_657974

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_valid_prime (p : ℕ) : Prop :=
  is_prime p ∧ 100 ≤ p ∧ p < 500 ∧ 
  (let swapped := (p % 10) * 100 + (p / 10 % 10) * 10 + (p / 100) in is_prime swapped)

theorem sum_of_valid_primes_is_2462 : ∑ p in Finset.filter is_valid_prime (Finset.filter is_prime (Finset.range 500)), p = 2462 :=
by
  sorry

end sum_of_valid_primes_is_2462_l657_657974


namespace complement_of_A_cap_B_l657_657854

def set_A (x : ℝ) : Prop := x ≤ -4 ∨ x ≥ 2
def set_B (x : ℝ) : Prop := |x - 1| ≤ 3

def A_cap_B (x : ℝ) : Prop := set_A x ∧ set_B x

def complement_A_cap_B (x : ℝ) : Prop := ¬A_cap_B x

theorem complement_of_A_cap_B :
  {x : ℝ | complement_A_cap_B x} = {x : ℝ | x < 2 ∨ x > 4} :=
by
  sorry

end complement_of_A_cap_B_l657_657854


namespace triangle_area_l657_657941

theorem triangle_area (a b c : ℝ) (h₀ : a = 18) (h₁ : b = 80) (h₂ : c = 82) (h₃ : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 720 :=
by
  rw [h₀, h₁, h₂]
  sorry

end triangle_area_l657_657941


namespace metal_bar_weight_loss_l657_657616

theorem metal_bar_weight_loss :
  ∃ T S : ℝ, 
  T + S = 50 ∧ 
  T / S = 2 / 3 ∧ 
  ((T / 10) * 1.375) + ((S / 5) * 0.375) = 5 :=
begin
  sorry
end

end metal_bar_weight_loss_l657_657616


namespace range_of_m_l657_657733

open Set

-- Definitions and conditions
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0
def neg_p (x : ℝ) : Prop := ¬ p x
def neg_q (x m : ℝ) : Prop := ¬ q x m

-- Theorem statement
theorem range_of_m (x m : ℝ) (h₁ : ¬ p x → ¬ q x m) (h₂ : m > 0) : m ≥ 9 :=
  sorry

end range_of_m_l657_657733


namespace problem_statement_l657_657749

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := λ x => f (x - 1)

theorem problem_statement :
  (∀ x : ℝ, f (-x) = f x) →  -- Condition: f is an even function.
  (∀ x : ℝ, g (-x) = -g x) → -- Condition: g is an odd function.
  (g 1 = 3) →                -- Condition: g passes through (1,3).
  (f 2012 + g 2013 = 6) :=   -- Statement to prove.
by
  sorry

end problem_statement_l657_657749


namespace permutations_without_patterns_l657_657816

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def num_permutations_excluding_patterns : ℕ :=
  let total := factorial 9 / (factorial 4 * factorial 3 * factorial 2)
  let A1 := factorial 6 / (factorial 3 * factorial 2)
  let A2 := factorial 7 / (factorial 4 * factorial 2)
  let A3 := factorial 8 / (factorial 4 * factorial 3)
  let A1_A2 := factorial 4 / factorial 2
  let A1_A3 := factorial 5 / factorial 3
  let A2_A3 := factorial 6 / factorial 4
  let A1_A2_A3 := factorial 3
  total - (A1 + A2 + A3) + (A1_A2 + A1_A3 + A2_A3) - A1_A2_A3

theorem permutations_without_patterns : num_permutations_excluding_patterns = 871 :=
by
  sorry

end permutations_without_patterns_l657_657816


namespace compound_proposition_p_or_q_l657_657520

theorem compound_proposition_p_or_q : 
  (∃ (n : ℝ), ∀ (m : ℝ), m * n = m) ∨ 
  (∀ (n : ℝ), ∃ (m : ℝ), m^2 < n) := 
by
  sorry

end compound_proposition_p_or_q_l657_657520


namespace find_target_function_l657_657169

open Real

/-- Definition of the target function y given constant a and expression for x + 2. -/
def target_function (a : ℝ) (x : ℝ) : ℝ :=
  a * (x + 2)^2 + 2

/-- Condition that ensures the tangents intersect at a right angle. -/
def tangents_intersect_right_angle (x0 y0 : ℝ) : Prop :=
  ∀ x y : ℝ, log (x - x^2 + 3) (y - 6) = log (x - x^2 + 3) ((abs (2 * x + 6) - abs (2 * x + 3)) / (3 * x + 7.5) * sqrt (x^2 + 5 * x + 6.25))

theorem find_target_function (x0 y0 : ℝ) (h_condition : tangents_intersect_right_angle x0 y0) :
  target_function (-0.15625) x0 = y0 :=
sorry

end find_target_function_l657_657169


namespace limit_a_n_l657_657521

noncomputable def a_n (n: ℕ) : ℝ := (23 - 4 * (n : ℝ)) / (2 - (n : ℝ))

theorem limit_a_n : filter.tendsto (λ n: ℕ, a_n n) filter.at_top (nhds 4) :=
sorry

end limit_a_n_l657_657521


namespace count_negative_x_with_sqrt_pos_int_l657_657350

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l657_657350


namespace amount_to_give_is_14_4_l657_657277

def earnings : List ℕ := [18, 22, 30, 38, 45]
def totalEarnings : ℕ := earnings.sum
def numberOfFriends : ℕ := earnings.length
def equalShare : ℕ := totalEarnings / numberOfFriends
def amountGivenByHighestEarner : ℕ := earnings.last' earnings - equalShare

theorem amount_to_give_is_14_4 :
  amountGivenByHighestEarner = 14.4 := by
  sorry

end amount_to_give_is_14_4_l657_657277


namespace number_of_lockers_is_3129_l657_657556

noncomputable def total_cost : ℝ :=
  let cost_per_digit := 0.03
  let cost_3_digit := 900 * 3 * cost_per_digit
  let cost_total := 336.57
  let cost_4_digit (n : ℕ) := (n - 1000 + 1) * 4 * cost_per_digit
  in cost_3_digit + cost_4_digit 3129

theorem number_of_lockers_is_3129 :
  let n := 3129 in
  total_cost = 336.57 := 
sorry

end number_of_lockers_is_3129_l657_657556


namespace evaluate_expression_at_3_l657_657975

-- Define the expression
def expression (x : ℕ) : ℕ := x^2 - 3*x + 2

-- Statement of the problem
theorem evaluate_expression_at_3 : expression 3 = 2 := by
    sorry -- Proof is omitted

end evaluate_expression_at_3_l657_657975


namespace possible_values_of_a_l657_657119

theorem possible_values_of_a 
  (p : ℕ) (hp : p.prime) 
  (a b k : ℤ) 
  (h1 : -p^2 * (1 + k) = 1100)
  (hb : b = k * p) 
  (ha : a = -((b + p^2) / p)) :
  (a = 274 ∨ a = 40) :=
sorry

end possible_values_of_a_l657_657119


namespace intersection_M_N_l657_657052

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657052


namespace count_negative_values_correct_l657_657283

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l657_657283


namespace kaleb_chocolate_boxes_l657_657008

theorem kaleb_chocolate_boxes :
  ∀ (total_boxes given_boxes current_pieces box_pieces : ℕ),
    total_boxes = 14 →
    given_boxes = 5 →
    current_pieces = 54 →
    total_boxes - given_boxes = 9 →
    current_pieces / (total_boxes - given_boxes) = 6 :=
by
  intros total_boxes given_boxes current_pieces box_pieces
  assume h1 h2 h3 h4
  sorry

end kaleb_chocolate_boxes_l657_657008


namespace probability_of_divisible_by_3_is_one_third_l657_657278

noncomputable def probability_sum_divisible_by_3 : ℚ :=
  let s := finset.range 16 \ {0}  -- Numbers 1 to 15
  let choices := finset.powersetLen 5 s
  let favorable := finset.filter (λ t, (((t.sum id) % 3) = 0)) choices
  favorable.card / choices.card

theorem probability_of_divisible_by_3_is_one_third :
  probability_sum_divisible_by_3 = 1 / 3 :=
sorry

end probability_of_divisible_by_3_is_one_third_l657_657278


namespace find_m_for_asymptotes_l657_657677

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 9 = 1

-- Definition of the asymptotes form
def asymptote_form (m : ℝ) (x y : ℝ) : Prop :=
  y - 1 = m * x + 2 * m ∨ y - 1 = -m * x - 2 * m

-- The main theorem to prove
theorem find_m_for_asymptotes :
  (∀ x y : ℝ, hyperbola x y → asymptote_form (4 / 3) x y) :=
sorry

end find_m_for_asymptotes_l657_657677


namespace orthocenter_is_circumcenter_l657_657561

open EuclideanGeometry

variables {A B C A₁ B₁ C₁ : Point}
variables (triangle_Similar : Triangle_Similar ABC A₁B₁C₁)
variables (A₁_on_BC : On_Side A₁ (BC))
variables (B₁_on_CA : On_Side B₁ (CA))
variables (C₁_on_AB : On_Side C₁ (AB))

theorem orthocenter_is_circumcenter:
  orthocenter A₁ B₁ C₁ = circumcenter A B C :=
sorry

end orthocenter_is_circumcenter_l657_657561


namespace gcd_of_repeated_three_digit_numbers_l657_657689

theorem gcd_of_repeated_three_digit_numbers :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → Int.gcd 1001001 n = 1001001 :=
by
  -- proof omitted
  sorry

end gcd_of_repeated_three_digit_numbers_l657_657689


namespace count_negative_x_with_sqrt_pos_int_l657_657344

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l657_657344


namespace linear_eq_a_value_l657_657423

theorem linear_eq_a_value (a : ℤ) (x : ℝ) 
  (h : x^(a-1) - 5 = 3) 
  (h_lin : ∃ b c : ℝ, x^(a-1) * b + c = 0 ∧ b ≠ 0):
  a = 2 :=
sorry

end linear_eq_a_value_l657_657423


namespace moon_non_consecutive_Os_l657_657241

theorem moon_non_consecutive_Os : 
  let letters := ['M', 'O', 'O', 'N'] in
  let total_arrangements := (Fact 4) / (Fact 2) in
  let consecutive_Os := Fact 3 in
  total_arrangements - consecutive_Os = 6 :=
by
  let letters := ['M', 'O', 'O', 'N']
  let total_arrangements := (Fact 4) / (Fact 2)
  let consecutive_Os := Fact 3
  show total_arrangements - consecutive_Os = 6
  sorry

end moon_non_consecutive_Os_l657_657241


namespace part_a_part_b_l657_657994

-- Part (a): Proving at most one integer solution for general k
theorem part_a (k : ℤ) : 
  ∀ (x1 x2 : ℤ), (x1^3 - 24*x1 + k = 0 ∧ x2^3 - 24*x2 + k = 0) → x1 = x2 :=
sorry

-- Part (b): Proving exactly one integer solution for k = -2016
theorem part_b :
  ∃! (x : ℤ), x^3 + 24*x - 2016 = 0 :=
sorry

end part_a_part_b_l657_657994


namespace group_division_l657_657447

theorem group_division (men women : ℕ) : 
  men = 4 → women = 5 →
  (∃ g1 g2 g3 : set (fin $ men + women), 
    g1.card = 3 ∧ g2.card = 3 ∧ g3.card = 3 ∧ 
    (∀ g, g ∈ [g1, g2, g3] → (∃ m w : ℕ, 1 ≤ m ∧ 1 ≤ w ∧ 
      finset.card (finset.filter (λ x, x < men) g) = m ∧ 
      finset.card (finset.filter (λ x, x ≥ men) g) = w)) 
    ∧ finset.disjoint g1 g2 ∧ finset.disjoint g2 g3 ∧ finset.disjoint g3 g1 
    ∧ g1 ∪ g2 ∪ g3 = finset.univ (fin $ men + women)) → 
  finset.card (finset.powerset' 3 (finset.univ (fin $ men + women))) / 2 = 180 :=
begin
  intros hmen hwomen,
  sorry
end

end group_division_l657_657447


namespace transylvanian_response_l657_657594

-- Define the types of beings in Transylvanian
inductive Being
| SaneHuman
| InsaneHuman
| SaneVampire
| InsaneVampire

def responds_yes_to_human (b : Being) : Prop :=
  b = Being.SaneHuman ∨ b = Being.InsaneHuman ∨ b = Being.SaneVampire ∨ b = Being.InsaneVampire

def responds_yes_to_reliable (b : Being) : Prop :=
  b = Being.SaneHuman ∨ b = Being.InsaneHuman ∨ b = Being.SaneVampire ∨ b = Being.InsaneVampire

theorem transylvanian_response (b : Being) : responds_yes_to_human b ∧ responds_yes_to_reliable b :=
  by {
    cases b;
    simp [responds_yes_to_human, responds_yes_to_reliable];
    sorry
  }

end transylvanian_response_l657_657594


namespace find_angle_A_and_sum_Sn_l657_657383

theorem find_angle_A_and_sum_Sn
  (a b c A B C : Real)
  (a₁ a₂ a₄ a₈ : Real)
  (an : ℕ → Real)
  (d : Real)
  (h0 : a₁ ≠ 0)
  (h1 : a₁ * Real.cos A = 1)
  (h2 : a₂ * a₈ = a₄^2)
  (h3 : ∀ n, an (n+1) = an n + d)
  (h4 : a * Real.sin A = b * Real.sin B + (c - b) * Real.sin C) :
  A = Real.pi / 3 ∧
  (∑ k in Finset.range n, 4 / (an k * an (k+1)) = n / (n + 1)) :=
by
  sorry

end find_angle_A_and_sum_Sn_l657_657383


namespace intersection_M_N_l657_657045

def M := {1, 3, 5, 7, 9}

def N := {x : ℤ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} := by
  sorry

end intersection_M_N_l657_657045


namespace angle_ECD_22_5_l657_657425

variables {A B C D E : Type*} [euclidean_geometry A] [euclidean_geometry B]
  [euclidean_geometry C] [euclidean_geometry D] [euclidean_geometry E]

/-- Given a right triangle ABC with ∠ACB = 90 degrees, CD is the height to AB, and CE is the angle bisector of ∠C.
    If the triangles CED and ABC are similar, then ∠ECD is 22.5 degrees. -/
theorem angle_ECD_22_5 (h1 : right_triangle ABC)
  (hACB : ∠ACB = 90)
  (hCD : altitude CD AB)
  (hCE : angle_bisector CE ∠ACB)
  (h_sim : similar_triangles CED ABC) :
  ∠ECD = 22.5 :=
sorry

end angle_ECD_22_5_l657_657425


namespace profit_percentage_before_decrease_l657_657681

-- Defining the conditions as Lean definitions
def newManufacturingCost : ℝ := 50
def oldManufacturingCost : ℝ := 80
def profitPercentageNew : ℝ := 0.5

-- Defining the problem as a theorem in Lean
theorem profit_percentage_before_decrease
  (P : ℝ)
  (hP : profitPercentageNew * P = P - newManufacturingCost) :
  ((P - oldManufacturingCost) / P) * 100 = 20 := 
by
  sorry

end profit_percentage_before_decrease_l657_657681


namespace inequality_solution_l657_657534

theorem inequality_solution :
  {x : ℝ | ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0} = 
  {x : ℝ | (1 < x ∧ x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7)} :=
sorry

end inequality_solution_l657_657534


namespace num_of_negative_x_l657_657289

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l657_657289


namespace rod_center_of_gravity_shift_l657_657354

noncomputable def rod_shift (l : ℝ) (s : ℝ) : ℝ := 
  |(l / 2) - ((l - s) / 2)| 

theorem rod_center_of_gravity_shift : 
  rod_shift l 80 = 40 := by
  sorry

end rod_center_of_gravity_shift_l657_657354


namespace tan_double_angle_l657_657789

theorem tan_double_angle (x y : ℝ) (h₁ : Real.tan x + Real.tan y = 10) (h₂ : Real.cot x + Real.cot y = 12) : 
  Real.tan (2 * x + 2 * y) = -120 / 3599 :=
sorry

end tan_double_angle_l657_657789


namespace smallest_four_digit_unique_l657_657576

def has_unique_digits (n : ℕ) : Prop :=
  (n % 10 ≠ (n / 10) % 10) ∧ (n % 10 ≠ (n / 100) % 10) ∧ (n % 10 ≠ (n / 1000) % 10) ∧ 
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧ ((n / 10) % 10 ≠ (n / 1000) % 10) ∧ 
  ((n / 100) % 10 ≠ (n / 1000) % 10)

theorem smallest_four_digit_unique : 1023 = 
  if 1032 < 1023 ∧ has_unique_digits 1032 then 1032 else
  if 1234 < 1023 ∧ has_unique_digits 1234 then 1234 else
  1023 := sorry

end smallest_four_digit_unique_l657_657576


namespace annual_interest_rate_l657_657704

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  ((A / P) ^ (1 / t)) - 1

-- Define the given parameters
def P : ℝ := 1200
def A : ℝ := 2488.32
def n : ℕ := 1
def t : ℕ := 4

theorem annual_interest_rate : compound_interest_rate P A n t = 0.25 :=
by
  sorry

end annual_interest_rate_l657_657704


namespace value_of_nested_fraction_l657_657559

def nested_fraction : ℚ :=
  2 - (1 / (2 - (1 / (2 - 1 / 2))))

theorem value_of_nested_fraction : nested_fraction = 3 / 4 :=
by
  sorry

end value_of_nested_fraction_l657_657559


namespace mod_inverse_3_35_l657_657700

theorem mod_inverse_3_35 : ∃ x : ℤ, 3 * x ≡ 1 [MOD 35] ∧ 0 ≤ x ∧ x ≤ 34 ∧ x = 12 :=
by
  use 12  -- proposed solution
  sorry  -- place holder for the proof

end mod_inverse_3_35_l657_657700


namespace statements_correct_l657_657949

theorem statements_correct :
  (∃ x y : ℝ, irrational x ∧ irrational y ∧ x ≠ y ∧ (∃ z : ℤ, x - y = ↑z)) ∧
  (∃ x y : ℝ, irrational x ∧ irrational y ∧ x ≠ y ∧ (∃ z : ℤ, x * y = ↑z)) ∧
  (∃ x y : ℚ, ¬integer x ∧ ¬integer y ∧ x ≠ y ∧ (∃ z w : ℤ, x + y = z ∧ x / y = ↑w)) :=
by
  sorry

end statements_correct_l657_657949


namespace num_of_negative_x_l657_657292

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l657_657292


namespace calculate_area_l657_657830

noncomputable def area {a b : ℝ} (f : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, f x

theorem calculate_area :
  let f1 := (λ x : ℝ, x^2),
      f2 := (λ x : ℝ, 3 * x - 10) in
  area f1 0 6 + area f2 6 10 = 128 :=
by
  let f1 := (λ x : ℝ, x^2)
  let f2 := (λ x : ℝ, 3 * x - 10)
  have h1 : area f1 0 6 = 72 := sorry
  have h2 : area f2 6 10 = 56 := sorry
  calc
    area f1 0 6 + area f2 6 10 = 72 + 56 := by rw [h1, h2]
                            ...           = 128 := by norm_num

end calculate_area_l657_657830


namespace constant_speed_l657_657898

open Real

def total_trip_time := 50.0
def total_distance := 2790.0
def break_interval := 5.0
def break_duration := 0.5
def hotel_search_time := 0.5

theorem constant_speed :
  let number_of_breaks := total_trip_time / break_interval
  let total_break_time := number_of_breaks * break_duration
  let actual_driving_time := total_trip_time - total_break_time - hotel_search_time
  let constant_speed := total_distance / actual_driving_time
  constant_speed = 62.7 :=
by
  -- Provide proof here
  sorry

end constant_speed_l657_657898


namespace distance_walked_by_friend_p_l657_657962

variable (v : ℝ) -- Speed of Friend Q in km/h
variable (d : ℝ) (d = 33) -- Total trail length in km

axiom friends_walking_conditions :
  ∀ t : ℝ, (1.2 * v * t) + (v * t) = d

theorem distance_walked_by_friend_p :
  ∀ t : ℝ, 
    t = d / (2.2 * v) →
    (1.2 * v * t) = 18 :=
by
  intros t ht
  rw ht
  have h : 1.2 * v * (d / (2.2 * v)) = (1.2 * d) / 2.2,
  { field_simp [mul_comm, mul_assoc, div_eq_mul_inv, mul_inv_cancel],
    linarith },
  rw [h],
  norm_num,
  sorry

end distance_walked_by_friend_p_l657_657962


namespace box_volume_l657_657976

theorem box_volume (l w h V : ℝ) 
  (h1 : l * w = 30) 
  (h2 : w * h = 18) 
  (h3 : l * h = 10) 
  : V = l * w * h → V = 90 :=
by 
  intro volume_eq
  sorry

end box_volume_l657_657976


namespace group_count_4_men_5_women_l657_657457

theorem group_count_4_men_5_women : 
  let men := 4
  let women := 5
  let groups := List.replicate 3 (3, true)
  ∃ (m_w_combinations : List (ℕ × ℕ)),
    m_w_combinations = [(1, 2), (2, 1)] ∧
    ((men.choose m_w_combinations.head.fst * women.choose m_w_combinations.head.snd) * (men - m_w_combinations.head.fst).choose m_w_combinations.tail.head.fst * (women - m_w_combinations.head.snd).choose m_w_combinations.tail.head.snd) = 360 :=
by
  sorry

end group_count_4_men_5_women_l657_657457


namespace volume_of_solid_of_revolution_l657_657851

noncomputable def piecewise_f (x : ℝ) : ℝ :=
  if x < 0 then real.sqrt (4 - x ^ 2) else 2 - x

theorem volume_of_solid_of_revolution :
  let f := piecewise_f in
  ∫ x in -2..2, π * (f x) ^ 2 = 8 * π :=
by
  sorry

end volume_of_solid_of_revolution_l657_657851


namespace function_reciprocal_points_l657_657723

def has_reciprocal_point (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, x * f x = 1

theorem function_reciprocal_points :
  has_reciprocal_point (λ x, -2 * x + 2 * Real.sqrt 2) ∧
  has_reciprocal_point (λ x, Real.sin x) ∧
  ¬ has_reciprocal_point (λ x, x + 1 / x) ∧
  has_reciprocal_point (λ x, Real.exp x) ∧
  ¬ has_reciprocal_point (λ x, -2 * Real.log x) :=
by
  sorry

end function_reciprocal_points_l657_657723


namespace birthday_number_l657_657514

theorem birthday_number (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 1 ≤ y ∧ y ≤ 9) :
    ∃ x y, (40000 + 1000 * x + 100 * y + 29 = 2379^2) :=
by {
  have hT : ∃ T, T = 2379 := Exists.intro 2379 rfl,
  sorry
}

end birthday_number_l657_657514


namespace sunset_time_l657_657873

theorem sunset_time (daylight_hours : ℕ) (daylight_minutes : ℕ) (sunrise_hour : ℕ) (sunrise_minutes : ℕ) :
  daylight_hours = 10 → daylight_minutes = 24 → sunrise_hour = 6 → sunrise_minutes = 57 →
  let sunset_hour := (sunrise_hour + daylight_hours + (sunrise_minutes + daylight_minutes) / 60) in
  let sunset_minutes := (sunrise_minutes + daylight_minutes) % 60 in
  (sunset_hour, sunset_minutes) = (17, 21) →
  (sunset_hour - 12, sunset_minutes) = (5, 21) :=
by
  intros h_daylight_hours h_daylight_minutes h_sunrise_hour h_sunrise_minutes
  let sunset_hour := (sunrise_hour + daylight_hours + (sunrise_minutes + daylight_minutes) / 60)
  let sunset_minutes := (sunrise_minutes + daylight_minutes) % 60
  have h_sunset_calc : (sunset_hour, sunset_minutes) = (17, 21) := by sorry
  show (sunset_hour - 12, sunset_minutes) = (5, 21) from h_sunset_calc

end sunset_time_l657_657873


namespace minimum_sum_value_l657_657732

def x : ℕ → ℤ
| 0 := 0
| n + 1 := | x n + 1 |

theorem minimum_sum_value : | (Finset.range 2004).sum x | = 34 :=
sorry

end minimum_sum_value_l657_657732


namespace omega_range_monotonically_decreasing_l657_657720

-- Definition of the function f(x)
def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 4)

-- The theorem to be proved
theorem omega_range_monotonically_decreasing (ω : ℝ) :
  ω > 0 →
  (∀ x, π / 2 < x ∧ x < π → f ω x ≤ f ω (x + ε))) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 :=
sorry

end omega_range_monotonically_decreasing_l657_657720


namespace polygon_sides_l657_657558

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
by sorry

end polygon_sides_l657_657558


namespace problem65_solution_l657_657997

def div1 : ℕ := 180 / 3
def mult1 : ℕ := 5 * 12
def div2 : ℕ := mult1 / div1

theorem problem65_solution : 65 + div2 = 66 := by
  have h1 : div1 = 60 := by sorry
  have h2 : mult1 = 60 := by sorry
  have h3 : div2 = 1 := by 
    rw [←h2, ←h1]
    exact Nat.div_self (by norm_num : 60 ≠ 0)
  rw [h3]
  rfl


end problem65_solution_l657_657997


namespace smallest_n_integer_sum_l657_657056

theorem smallest_n_integer_sum : ∃ n : ℕ, (n > 0) ∧ (ℚ.isInteger (1/3 + 1/4 + 1/6 + 1/n)) ∧ ∀ m : ℕ, (m > 0) ∧ (ℚ.isInteger (1/3 + 1/4 + 1/6 + 1/m)) → n ≤ m :=
sorry

end smallest_n_integer_sum_l657_657056


namespace percentage_error_in_side_length_l657_657208

theorem percentage_error_in_side_length 
  (A A' s s' : ℝ) (h₁ : A = s^2)
  (h₂ : A' = s'^2)
  (h₃ : ((A' - A) / A * 100) = 12.36) :
  ∃ E : ℝ, (s' = s * (1 + E / 100)) ∧ (E = (real.sqrt 1.1236 - 1) * 100) := 
by
  sorry

end percentage_error_in_side_length_l657_657208


namespace mushroom_distribution_l657_657778

theorem mushroom_distribution :
  let total_mushrooms := 94 + 85
  let rabbits := 8 
  total_mushrooms = 179 ∧ total_mushrooms / rabbits = 22 ∧ total_mushrooms % rabbits = 3 :=
by
  let total_mushrooms := 94 + 85
  let rabbits := 8 
  have h1: total_mushrooms = 179 := rfl
  have h2: total_mushrooms / rabbits = 22 := by norm_num
  have h3: total_mushrooms % rabbits = 3 := by norm_num
  exact ⟨h1, h2, h3⟩

end mushroom_distribution_l657_657778


namespace lucy_initial_fish_count_l657_657861

def initial_fish_count (final_fish : ℕ) (extra_fish : ℕ) : ℕ := final_fish - extra_fish

theorem lucy_initial_fish_count (final_fish : ℕ) (extra_fish : ℕ) (initial_fish : ℕ) :
  final_fish = 280 → extra_fish = 68 → initial_fish = initial_fish_count final_fish extra_fish → initial_fish = 212 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

#eval initial_fish_count 280 68 -- It should return 212


end lucy_initial_fish_count_l657_657861


namespace inequality_solution_l657_657531

def f (x : ℝ) : ℝ := ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution :
  { x : ℝ | f x > 0 } = {x : ℝ | x < 1} ∪ {x : ℝ | 2 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry


end inequality_solution_l657_657531


namespace group_count_4_men_5_women_l657_657456

theorem group_count_4_men_5_women : 
  let men := 4
  let women := 5
  let groups := List.replicate 3 (3, true)
  ∃ (m_w_combinations : List (ℕ × ℕ)),
    m_w_combinations = [(1, 2), (2, 1)] ∧
    ((men.choose m_w_combinations.head.fst * women.choose m_w_combinations.head.snd) * (men - m_w_combinations.head.fst).choose m_w_combinations.tail.head.fst * (women - m_w_combinations.head.snd).choose m_w_combinations.tail.head.snd) = 360 :=
by
  sorry

end group_count_4_men_5_women_l657_657456


namespace num_literary_readers_l657_657434

def total_readers : ℕ := 400
def scifi_readers : ℕ := 250
def both_readers : ℕ := 80

theorem num_literary_readers :
  ∃ L : ℕ, total_readers = scifi_readers + L - both_readers ∧ L = 230 :=
by {
  let L := total_readers - scifi_readers + both_readers,
  use L,
  sorry
}

end num_literary_readers_l657_657434


namespace intersection_M_N_l657_657040

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := { x | 2 * x > 7 }

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657040


namespace no_tetrahedron_with_all_faces_congruent_right_triangles_l657_657823

theorem no_tetrahedron_with_all_faces_congruent_right_triangles :
  ¬ (∃ (A B C D : ℝ × ℝ × ℝ), -- ∃ four points in ℝ³
    let T := { A, B, C, D } in 
    -- T contains 4 points and forms a tetrahedron
    (∀ {X Y Z : ℝ × ℝ × ℝ}, X ∈ T → Y ∈ T → Z ∈ T → 
      (X ≠ Y → Y ≠ Z → Z ≠ X → 
        (right_triangle X Y Z) ∧
        (∀ {X' Y' Z' : ℝ × ℝ × ℝ}, right_triangle X' Y' Z' → 
          congruent_triangles X Y Z X' Y' Z')))) := 
sorry

end no_tetrahedron_with_all_faces_congruent_right_triangles_l657_657823


namespace common_circumcircle_point_l657_657883

/-
Prove that the circumcircles of the triangles formed by the medians of an acute-angled triangle
have a common point.
-/

theorem common_circumcircle_point 
  {A B C A1 B1 C1 : Type}
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space A1] [metric_space B1] [metric_space C1]
  (H_acute : ∀ (α β γ : ℝ), 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π → α < π/2 ∧ β < π/2 ∧ γ < π/2)
  (H_midpoints : ∀ (A B C A1 B1 C1 P : Type), (A1 = midpoint B C) ∧ (B1 = midpoint C A) ∧ (C1 = midpoint A B))
  (H_circumcircle : ∀ (ABC A1B1C1 : Type), is_circumcircle ABC A1B1C1) :
  ∃ M : Type, (M ∈ circumcircle (triangle A1 B C1) ∧ M ∈ circumcircle (triangle A C1 B1) ∧ M ∈ circumcircle (triangle B1 C A1)) :=
sorry

end common_circumcircle_point_l657_657883


namespace age_composition_is_decline_l657_657621

-- Define the population and age groups
variable (P : Type)
variable (Y E : P → ℕ) -- Functions indicating the number of young and elderly individuals

-- Assumptions as per the conditions
axiom fewer_young_more_elderly (p : P) : Y p < E p

-- Conclusion: Prove that the population is of Decline type.
def age_composition_decline (p : P) : Prop :=
  Y p < E p

theorem age_composition_is_decline (p : P) : age_composition_decline P Y E p := by
  sorry

end age_composition_is_decline_l657_657621


namespace lattice_point_inequalities_l657_657780

theorem lattice_point_inequalities :
  let S := {p : ℤ × ℤ | p.1^2 + p.2^2 < 25 ∧ p.1^2 + p.2^2 < 10 * p.1 ∧ p.1^2 + p.2^2 < 10 * p.2} in
  S.to_finset.card = 8 :=
by sorry

end lattice_point_inequalities_l657_657780


namespace negative_values_count_l657_657327

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l657_657327


namespace f_0_eq_sqrt3_ω_eq_2_inequ_solve_l657_657761

def (ω : ℝ) > 0
def f (x : ℝ) : ℝ := 2 * cos (π / 2 - ω * x) + 2 * sin (π / 3 - ω * x)
def f_cond : Prop := f (π / 6) + f (π / 2) = 0
def decreasing (a b : ℝ) (h : a < b) : Prop := ∀ x1 x2, a < x1 → x1 < x2 → x2 < b → f x1 > f x2
def decr_cond : Prop := decreasing (π / 6) (π / 2) (pi_div_two_pos : π / 6 < π / 2)

theorem f_0_eq_sqrt3 (h₀ : ω > 0) (h₁ : f_cond) (h₂ : decr_cond) : f 0 = sqrt 3 := sorry

theorem ω_eq_2 (h₀ : ω > 0) (h₁ : f_cond) (h₂ : decr_cond) : ω = 2 := sorry

theorem inequ_solve (h₀ : ω > 0) (h₁ : f_cond) (h₂ : decr_cond) : 
  ∀ x, f x ≥ 1 ↔ ∃ k : ℤ, (π / 12 + k * π ≤ x ∧ x ≤ π / 4 + k * π) := sorry

end f_0_eq_sqrt3_ω_eq_2_inequ_solve_l657_657761


namespace smallest_integer_with_divisors_l657_657973

theorem smallest_integer_with_divisors :
  ∃ n : ℤ, (odd_divisors n = 12) ∧ (even_divisors n = 18) ∧ (∀ m : ℤ, (odd_divisors m = 12) ∧ (even_divisors m = 18) → n ≤ m) ∧ n = 900 :=
sorry

end smallest_integer_with_divisors_l657_657973


namespace determine_k_l657_657242

variable (x y z k : ℝ)

theorem determine_k (h : 9 / (x + y) = k / (y + z) ∧ k / (y + z) = 15 / (x - z)) : k = 0 := by
  sorry

end determine_k_l657_657242


namespace exists_n_sum_of_three_squares_l657_657248

variable (n : ℕ)
def is_sum_of_three_squares (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ n = a^2 + b^2 + c^2

theorem exists_n_sum_of_three_squares : 
  ∃ (n : ℕ), n < 10^9 ∧ (∃! (a b c : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0), n = a^2 + b^2 + c^2) 1000 := 
sorry

end exists_n_sum_of_three_squares_l657_657248


namespace intersection_proof_l657_657034

noncomputable def M : Set ℕ := {1, 3, 5, 7, 9}

noncomputable def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_proof : M ∩ (N ∩ Set.univ) = {5, 7, 9} :=
by sorry

end intersection_proof_l657_657034


namespace work_faster_with_additional_workers_l657_657620

variable (W : ℕ) (D : ℕ) (X : ℝ)

-- Definitions based on the conditions
def original_days : ℕ := 45
def additional_workers : ℕ := 10
def original_workers : ℕ := 35
def total_workers := original_workers + additional_workers

-- Equation reflecting the constant amount of work
def work_equation : Prop :=
  original_workers * original_days = total_workers * (original_days - X)

-- Statement to prove that adding 10 more workers reduces the completion time by approximately 16.36 days.
theorem work_faster_with_additional_workers :
  work_equation X → X ≈ 16.36 :=
sorry

end work_faster_with_additional_workers_l657_657620


namespace find_tan_Y_l657_657258

def right_triangle (X Y Z : Type) [metric_space X] :=
  ∃ (hypotenuse opposite adjacent : ℝ), hypotenuse = 25 ∧ adjacent = 24 
  ∧ sqrt(hypotenuse^2 - adjacent^2) = 7

theorem find_tan_Y (X Y Z : Type) [metric_space X] (hypotenuse opposite adjacent : ℝ)
  (h_hypotenuse : hypotenuse = 25)
  (h_adjacent : adjacent = 24)
  (h_opposite : opposite = sqrt(25^2 - 24^2)):
  opposite / adjacent = 7 / 24 :=
by
  intro X Y Z
  have h1 : 25 ^ 2 = 625 by norm_num
  have h2 : 24 ^ 2 = 576 by norm_num
  have h3 : 625 - 576 = 49 by norm_num
  have h4 : sqrt(49) = 7 by norm_num
  have h5 : opposite = 7 by rwa [h1, h2, h3, h4] at h_opposite
  have h6 : 7 / 24 = 7 / 24 by norm_num
  exact h6
  

end find_tan_Y_l657_657258


namespace cos_A_value_and_area_l657_657427
noncomputable theory

-- Define the conditions
variables {A B C : ℝ} {a b c : ℝ}
variables (sin cos sqrt : ℝ → ℝ)
axiom sin2_cos2_add : ∀ x, sin x ^ 2 + cos x ^ 2 = 1
axiom sin_of_angle_sum : ∀ {A B C : ℝ}, sin (A + B + C) = sin A + sin B + sin C
axiom solving_quadratic : ∀ {a b c : ℝ}, a * x^2 + b * x + c = 0 → (x = (-b + sqrt (b^2 - 4 * a * c)) / (2 * a) ∨ x = (-b - sqrt (b^2 - 4 * a * c)) / (2 * a))

-- Conditions from the problem
axiom condition1 : c * sin A - 2 * b * sin C = 0
axiom condition2 : a^2 - b^2 - c^2 = sqrt 5 / 5 * a * c
axiom b_value : b = sqrt 5

-- Target
theorem cos_A_value_and_area
    (h1 : c * sin A - 2 * b * sin C = 0)
    (h2 : a^2 - b^2 - c^2 = sqrt 5 / 5 * a * c)
    (h3 : b = sqrt 5)
    : cos A = - sqrt 5 / 5 
    ∧ (1/2) * b * c * sin A = 3 :=
sorry

end cos_A_value_and_area_l657_657427


namespace two_points_determine_a_line_l657_657155

-- Let A and B be points in a 2D plane, representing the positions of the front and back desks respectively.
-- The principle states that there exists a unique line passing through points A and B.
-- We aim to prove this principle.
theorem two_points_determine_a_line (A B : Point) (h : A ≠ B) :
    ∃! l : Line, A ∈ l ∧ B ∈ l := by
  sorry

end two_points_determine_a_line_l657_657155


namespace find_certain_number_l657_657179

theorem find_certain_number (D S X : ℕ): 
  D = 20 → 
  S = 55 → 
  X + (D - S) = 3 * D - 90 →
  X = 5 := 
by
  sorry

end find_certain_number_l657_657179


namespace seq_contains_122_l657_657655

def generate_next_digit (a b c : ℕ) : ℕ :=
  (a + b + c) % 10

def sequence_contains_subsequence 
  (initial : List ℕ) (target : List ℕ) : Prop := 
  ∃ n, initial.append (list.replicate n 0) where
    list.replicate n 0 gives n zeros to extend the initial list

theorem seq_contains_122 :
  sequence_contains_subsequence [2, 0, 1] [1, 2, 2] :=
sorry

end seq_contains_122_l657_657655


namespace three_different_suits_probability_l657_657866

def probability_three_different_suits := (39 / 51) * (35 / 50) = 91 / 170

theorem three_different_suits_probability (deck : Finset (Fin 52)) (h : deck.card = 52) :
  probability_three_different_suits :=
sorry

end three_different_suits_probability_l657_657866


namespace max_min_distance_to_line_l657_657390

def curve (x y : ℝ) := (x^2 / 4) + (y^2 / 9) = 1
def param_line (t : ℝ) : ℝ × ℝ := (2 * t, 2 - 2 * t)
def point_on_curve (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 3 * Real.sin θ)

noncomputable def distance (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ :=
  let (x, y) := P
  abs (4 * Real.cos θ + 3 * Real.sin θ - 6) / Real.sqrt 5

theorem max_min_distance_to_line :
  ∀ θ : ℝ, P : ℝ × ℝ,
  curve (2 * Real.cos θ) (3 * Real.sin θ) →
  (distance (2 * Real.cos θ, 3 * Real.sin θ) param_line) = max_iff (θ : ℝ) (P: ℝ × ℝ) :=
  sorry

end max_min_distance_to_line_l657_657390


namespace theta_25_degrees_l657_657416

theorem theta_25_degrees (θ : ℝ) (h1 : 0 < θ ∧ θ < 90) (h2 : cos θ - sin θ = sqrt 2 * sin (20 * real.pi / 180)) :
  θ = 25 * real.pi / 180 :=
sorry

end theta_25_degrees_l657_657416


namespace combined_perimeter_l657_657643

-- Define the given areas and height
def area_square : ℝ := 144
def area_rectangle : ℝ := 100
def height_rectangle : ℝ := 25

-- State the theorem for the combined perimeter
theorem combined_perimeter : 
  let side_square := Real.sqrt area_square in
  let length_rectangle := area_rectangle / height_rectangle in
  4 * side_square + 2 * length_rectangle + 2 * height_rectangle - 2 * (side_square + length_rectangle) = 74 := 
by
  -- sqrts and basic arithmetic can be used in the proof here, but the proof itself is skipped according to the instructions
  sorry

end combined_perimeter_l657_657643


namespace polynomial_equivalence_l657_657979

variable (x : ℝ) -- Define variable x

-- Define the expressions.
def expr1 := (3 * x^2 + 5 * x + 8) * (x + 2)
def expr2 := (x + 2) * (x^2 + 5 * x - 72)
def expr3 := (4 * x - 15) * (x + 2) * (x + 6)

-- Define the expression to be proved.
def original_expr := expr1 - expr2 + expr3
def simplified_expr := 6 * x^3 + 21 * x^2 + 18 * x

-- The theorem to prove the equivalence of the original and simplified expressions.
theorem polynomial_equivalence : original_expr = simplified_expr :=
by sorry -- proof to be filled in

end polynomial_equivalence_l657_657979


namespace impact_point_l657_657135

variable (R g α : ℝ)
variable (V t T : ℝ)
variable (cos_alpha sqrt : ℝ → ℝ)

-- Conditions
def V_def : V = sqrt (2 * g * R * cos_alpha α)
def x_t : x t = R * sin α + V * cos α * t
def y_t : y t = R * (1 - cos α) + V * sin α * t - (g * t^2) / 2
def T_def : T = sqrt (2 * R / g) * (sin α * sqrt (cos α) + sqrt (1 - (cos α)^3))

-- Desired proof
theorem impact_point :
  x T = R * (sin α + sin (2 * α) + sqrt (cos α * (1 - (cos α)^3))) :=
  sorry

end impact_point_l657_657135


namespace count_negative_x_with_sqrt_pos_int_l657_657345

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l657_657345


namespace parallelogram_area_l657_657075

theorem parallelogram_area :
  ∀ (A B C D : Type) [EuclideanGeometry A B C D],
    ∀ (AB AD : ℝ) (angle_BAD : ℝ) (area : ℝ),
      AB = 12 ∧ AD = 10 ∧ angle_BAD = 150 ∧ parallelogram A B C D →
      area = 60 :=
by sorry

end parallelogram_area_l657_657075


namespace equal_segments_AP_BQ_l657_657059

variables (A B C K L P Q : Point)
variables (ABA : Line) (triangleABC : Triangle)
variables (altitudeAK : Line) (altitudeBL : Line)
variables (excircleOmega : Circle) (circumcircleOmega : Circle)
variables (tangentPoints : Tangent) (homothetyCenter : Point)

-- Conditions
axiom altitudes_of_ABC : altitudeAK.perpendicular_to (ABA) ∧ altitudeBL.perpendicular_to(ABA)
axiom excircle_tangent_AB : excircleOmega.is_tangent_to_side(triangleABC, ABA)
axiom circumscribed_triangle_CKL : circumcircleOmega.is_circumscribing(triangleABC)
axiom tangents_intersect_AB_at_P_and_Q :
  (tangentPoints.is_internal_common_tangent(circumcircleOmega, excircleOmega)) ∧
  (tangentPoints.intersect_line_at(circumcircleOmega, excircleOmega, ABA, P, Q))

-- Theorem statement
theorem equal_segments_AP_BQ : dist A P = dist B Q :=
  by
    sorry

end equal_segments_AP_BQ_l657_657059


namespace numeral_150th_decimal_place_l657_657584

theorem numeral_150th_decimal_place (k : ℕ) (h : k = 150) : 
  (decimal_place (13 / 17) k) = 5 :=
sorry

end numeral_150th_decimal_place_l657_657584


namespace profit_percentage_correct_l657_657634

theorem profit_percentage_correct :
  ∀ (cost_price selling_price : ℕ), 
  (cost_price = 620 ∧ selling_price = 775) →
  let profit := selling_price - cost_price in
  let profit_percentage := (profit * 100) / cost_price in
  profit_percentage = 25 :=
by
  intros cost_price selling_price h,
  cases h with h_cost h_selling,
  rw [h_cost, h_selling],
  let profit := 775 - 620,
  have h_profit : profit = 155 := by norm_num,
  let profit_percentage := (profit * 100) / 620,
  have h_pp_eq : profit_percentage = (155 * 100) / 620 := by rw h_profit,
  have h_pp_norm : profit_percentage = 25 := by norm_num,
  exact h_pp_norm

end profit_percentage_correct_l657_657634


namespace work_completion_l657_657615

-- Define the basic variables and conditions
variables (men1 men2 days1 days2: ℕ)
variable (h1 : men1 = 36)
variable (h2 : days1 = 18)
variable (h3 : men2 = 81)

-- Define the main theorem
theorem work_completion (D : ℕ) (h4 : days2 = D) :
  (men1 * days1 = men2 * days2) → (D = 41) :=
by {
  intros h_eq,
  rw [h1, h2, h3] at h_eq,
  have h_calculation : 36 * 18 = 81 * D := h_eq,
  norm_num at h_calculation,
  linarith,
  -- Proof can be filled in here
  sorry
}

end work_completion_l657_657615


namespace cannot_lie_on_line_l657_657229

open Real

theorem cannot_lie_on_line (m b : ℝ) (h1 : m * b > 0) (h2 : b > 0) :
  (0, -2023) ≠ (0, b) :=
by
  sorry

end cannot_lie_on_line_l657_657229


namespace jana_walk_distance_l657_657469

-- Define the time taken to walk one mile and the rest period
def walk_time_per_mile : ℕ := 24
def rest_time_per_mile : ℕ := 6

-- Define the total time spent per mile (walking + resting)
def total_time_per_mile : ℕ := walk_time_per_mile + rest_time_per_mile

-- Define the total available time
def total_available_time : ℕ := 78

-- Define the number of complete cycles of walking and resting within the total available time
def complete_cycles : ℕ := total_available_time / total_time_per_mile

-- Define the distance walked per cycle (in miles)
def distance_per_cycle : ℝ := 1.0

-- Define the total distance walked
def total_distance_walked : ℝ := complete_cycles * distance_per_cycle

-- The proof statement
theorem jana_walk_distance : total_distance_walked = 2.0 := by
  sorry

end jana_walk_distance_l657_657469


namespace probability_of_selection_matching_ticket_l657_657080

noncomputable def is_power_of_ten (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 10^k

noncomputable def is_divisible_by_eleven (n : ℕ) : Prop :=
  n % 11 = 0

theorem probability_of_selection_matching_ticket :
  ∀ (S : Finset ℕ),
  (∀ x ∈ S, x ∈ (Finset.range 60).erase 0) →
  S.card = 6 →
  is_power_of_ten (S.prod id) →
  is_divisible_by_eleven (S.sum id) →
  (1 / 2 : ℝ) =
  -- Here, we leave the proof of the probability
  sorry

end probability_of_selection_matching_ticket_l657_657080


namespace number_of_negative_x_l657_657338

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l657_657338


namespace count_negative_values_of_x_l657_657311

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l657_657311


namespace average_revenue_fall_is_correct_l657_657953

-- Define the initial and final revenues for each company
def initial_revenue_A : ℝ := 72.0
def final_revenue_A : ℝ := 48.0
def initial_revenue_B : ℝ := 30.0
def final_revenue_B : ℝ := 25.0
def initial_revenue_C : ℝ := 65.0
def final_revenue_C : ℝ := 57.0

-- Calculate the revenue fall for each company
def revenue_fall_A := initial_revenue_A - final_revenue_A
def revenue_fall_B := initial_revenue_B - final_revenue_B
def revenue_fall_C := initial_revenue_C - final_revenue_C

-- Calculate the percent revenue fall for each company
def percent_fall_A := (revenue_fall_A / initial_revenue_A) * 100
def percent_fall_B := (revenue_fall_B / initial_revenue_B) * 100
def percent_fall_C := (revenue_fall_C / initial_revenue_C) * 100

-- Calculate the average percent of revenue fall
def average_percent_fall := (percent_fall_A + percent_fall_B + percent_fall_C) / 3

-- The theorem statement
theorem average_revenue_fall_is_correct : average_percent_fall = 20.77 := by
  sorry

end average_revenue_fall_is_correct_l657_657953


namespace sum_fx_eq_sqrt_3_l657_657360

theorem sum_fx_eq_sqrt_3 :
  (∑ k in Finset.range 2008, (λ x : ℕ, sin (π / 3 * (x + 1)) - sqrt 3 * cos (π / 3 * (x + 1))) (k + 1)) = sqrt 3 :=
by
  sorry

end sum_fx_eq_sqrt_3_l657_657360


namespace time_to_reach_ship_l657_657195

-- Define the conditions
def rate_of_descent := 30 -- feet per minute
def depth_to_ship := 2400 -- feet

-- Define the proof statement
theorem time_to_reach_ship : (depth_to_ship / rate_of_descent) = 80 :=
by
  -- The proof will be inserted here in practice
  sorry

end time_to_reach_ship_l657_657195


namespace equivalent_fraction_l657_657254

theorem equivalent_fraction : (8 / (5 * 46)) = (0.8 / 23) := 
by sorry

end equivalent_fraction_l657_657254


namespace intersection_A_B_l657_657737

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_A_B : A ∩ B = {1, 2} :=
  sorry

end intersection_A_B_l657_657737


namespace decimal_representation_150th_digit_l657_657587

theorem decimal_representation_150th_digit : 
  ∃ s : ℕ → ℤ, 
    (∀ n, s n = (13/17 : ℚ)^n % 10) ∧ 
    let decSeq := [7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1].cycle in
    ∀ n, decSeq.get n = s n → decSeq.get 149 = 7 := 
begin
  sorry
end

end decimal_representation_150th_digit_l657_657587


namespace lemon_juice_quarts_l657_657564

theorem lemon_juice_quarts (ratio_water_juice : ℕ × ℕ) (gallons : ℚ) (quarts_per_gallon : ℕ) : 
  ratio_water_juice = (5, 3) → gallons = 1.5 → quarts_per_gallon = 4 → 
  let total_parts := ratio_water_juice.1 + ratio_water_juice.2 in
  let total_quarts := gallons * quarts_per_gallon in
  let quarts_per_part := total_quarts / total_parts in
  let lemon_juice_quarts := ratio_water_juice.2 * quarts_per_part in
  lemon_juice_quarts = 9 / 4 :=
by
  -- Insert proof steps here as needed
  sorry

end lemon_juice_quarts_l657_657564


namespace dice_game_probability_l657_657617

theorem dice_game_probability (n : ℕ) : 
  let Pn := (1 / 2 : ℝ) + (1 / 2 : ℝ) * ( - (1 / 3 : ℝ))^(n - 1)
  in Pn = (1 / 2 : ℝ) + (1 / 2 : ℝ) * ( - (1 / 3 : ℝ))^(n - 1) := 
by
  sorry

end dice_game_probability_l657_657617


namespace not_necessarily_divisible_by_66_l657_657902

theorem not_necessarily_divisible_by_66 (m : ℤ) (h1 : ∃ k : ℤ, m = k * (k + 1) * (k + 2) * (k + 3) * (k + 4)) (h2 : 11 ∣ m) : ¬ (66 ∣ m) :=
sorry

end not_necessarily_divisible_by_66_l657_657902


namespace mutual_greetings_l657_657945

theorem mutual_greetings (students : ℕ) (letters_sent : ∀ s : ℕ, letters_sent_to_classmates s ≥ 16)
  (h : students = 30) : ∃ pairs : ℕ, pairs ≥ 45 :=
by
  sorry

end mutual_greetings_l657_657945


namespace dart_board_prob_odd_score_l657_657876

-- Define the conditions
def outer_circle_radius : ℕ := 8
def inner_circle_radius : ℕ := 4
def inner_point_values : List ℕ := [3, 5, 5]
def outer_point_values : List ℕ := [4, 3, 3]

-- Auxiliary calculations (without steps, as required)
def inner_area : ℝ := π * (inner_circle_radius ^ 2)
def outer_area : ℝ := π * (outer_circle_radius ^ 2) - inner_area
def total_area : ℝ := inner_area + outer_area

def probability_inner_region : ℝ := inner_area / total_area
def probability_outer_region : ℝ := outer_area / total_area

-- Correct answer (probability of getting an odd score)
def correct_probability : ℚ := 24 / 49

-- Problem statement
theorem dart_board_prob_odd_score :
  (((probability_inner_region / 3) * 2 + (probability_inner_region / 3)) * 3) +
  (((probability_outer_region / 3) * 2 + (probability_outer_region / 3)) * 3) = correct_probability := sorry

end dart_board_prob_odd_score_l657_657876


namespace average_age_of_new_men_is_30_l657_657908

noncomputable def average_age_of_two_new_men (A : ℝ) : ℝ :=
  let total_age_before : ℝ := 8 * A
  let total_age_after : ℝ := 8 * (A + 2)
  let age_of_replaced_men : ℝ := 21 + 23
  let total_age_of_new_men : ℝ := total_age_after - total_age_before + age_of_replaced_men
  total_age_of_new_men / 2

theorem average_age_of_new_men_is_30 (A : ℝ) : 
  average_age_of_two_new_men A = 30 :=
by 
  sorry

end average_age_of_new_men_is_30_l657_657908


namespace yolkino_palkino_l657_657071

open Nat

/-- On every kilometer of the highway between the villages Yolkino and Palkino, there is a post with a sign.
    On one side of the sign, the distance to Yolkino is written, and on the other side, the distance to Palkino is written.
    The sum of all the digits on each post equals 13.
    Prove that the distance from Yolkino to Palkino is 49 kilometers. -/
theorem yolkino_palkino (n : ℕ) (h : ∀ k : ℕ, k ≤ n → (digits 10 k).sum + (digits 10 (n - k)).sum = 13) : n = 49 :=
by
  sorry

end yolkino_palkino_l657_657071


namespace k_value_l657_657794

theorem k_value {x y k : ℝ} (h : ∃ c : ℝ, (x ^ 2 + k * x * y + 49 * y ^ 2) = c ^ 2) : k = 14 ∨ k = -14 :=
by sorry

end k_value_l657_657794


namespace number_of_negative_x_l657_657342

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l657_657342


namespace diagonals_intersect_probability_l657_657996

theorem diagonals_intersect_probability (n : ℕ) (hn : n = 8) :
  let total_points := n,
      total_combinations := Nat.choose total_points 2,
      total_sides := total_points,
      total_diagonals := total_combinations - total_sides,
      diagonal_pairs := Nat.choose total_diagonals 2,
      quad_points_combinations := Nat.choose total_points 4,
      probability := (quad_points_combinations : ℚ) / diagonal_pairs
  in probability = 7 / 19 := 
by
  sorry

end diagonals_intersect_probability_l657_657996


namespace find_constant_term_l657_657266

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def expansion_constant_term : ℤ :=
  let term1 := (-1)^4 * binomial_coeff 8 4
  let term2 := 2 * (-1)^5 * binomial_coeff 8 5
  term1 + term2

theorem find_constant_term :
  expansion_constant_term = -42 :=
by
  exactly sorry

end find_constant_term_l657_657266


namespace find_ellipse_eq_max_area_ABC_l657_657759

noncomputable def ellipse_eq (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x = 2 ∧ y = sqrt 2) ∨ (x = -2 ∧ y = -sqrt 2)

theorem find_ellipse_eq (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : dist (0, -b) (sqrt 2, 1) = 2)
  (h4 : ∃ x y, (x = sqrt 2 ∧ y = 1)):
  ellipse_eq a b :=
by sorry

theorem max_area_ABC (m : ℝ) (h1 : ∃ A B C : ℝ×ℝ, A = (sqrt 2, 1) ∧ B ≠ C)
  (h2 : ∀ x y, (y = sqrt 2 / 2 * x + m) → line_through A B C)
  (h3 : ∃ k: ℝ, ∀ t : ℝ, 0 < k^2 < 4):
  maximum_area (triangle A B C) = sqrt 2 :=
by sorry

end find_ellipse_eq_max_area_ABC_l657_657759


namespace ellipse_equation_line_equation_l657_657391

-- Conditions for the ellipse
variables {a b : ℝ}
variable (h_a_pos : a > b > 0)
variable (h_eccentricity : (c : ℝ) → a = 2 * c)
variable (h_point_on_ellipse : ∃ c : ℝ, 1 / (4 * c^2) + 3 / (4 * c^2) = 1)

-- Definition of the ellipse and the point
def ellipse : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ c : ℝ, let x := p.1, y := p.2 in (x^2) / (4 * c^2) + (y^2) / (3 * c^2) = 1}

def point_P := (1 : ℝ, 3 / 2 : ℝ)

-- Conditions for the line and the areas of triangles
variables {l : ℝ → (ℝ × ℝ)} -- Definition of line l
variable (h_line : ∃ k : ℝ, ∀ x, l x = (x, k * (x - 4)))

variables {A M N F : ℝ × ℝ}
variable (h_A : A = (4, 0))
variable (h_M_between_A_N : M.1 < N.1 ∧ M.1 > A.1)
variable (h_equal_area : area_triangle A M F = area_triangle M N F)

open_locale real

-- Goal: prove the equation of the ellipse and the line
theorem ellipse_equation :
  (∀ c, ∃ x y : ℝ, (x^2) / (4 * c^2) + (y^2) / (3 * c^2) = 1) →
  ellipse = {p : ℝ × ℝ | let x := p.1, y := p.2 in (x^2) / 4 + (y^2) / 3 = 1} :=
sorry

theorem line_equation :
  h_point_on_ellipse →
  (∀ k, ∃ x y : ℝ, l x = (x, k * (x - 4))) →
  (h_equal_area → l = {p : ℝ × ℝ | let x := p.1, y := p.2 in y = ±(sqrt 5 / 6) * (x - 4)}) :=
sorry

end ellipse_equation_line_equation_l657_657391


namespace numeral_150th_decimal_place_l657_657582

theorem numeral_150th_decimal_place (k : ℕ) (h : k = 150) : 
  (decimal_place (13 / 17) k) = 5 :=
sorry

end numeral_150th_decimal_place_l657_657582


namespace dot_product_zero_l657_657771

-- Define vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

-- Define the dot product operation for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the scalar multiplication and vector subtraction for 2D vectors
def scalar_mul_vec (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Now we state the theorem we want to prove
theorem dot_product_zero : dot_product a (vec_sub (scalar_mul_vec 2 a) b) = 0 := 
by
  sorry

end dot_product_zero_l657_657771


namespace intersection_M_N_l657_657051

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657051


namespace nth_equation_l657_657067

theorem nth_equation (n : ℕ) (hn: n > 0) : 
  (finset.range (2 * n - 1)).sum (λ k, n + k) = (2 * n - 1) ^ 2 := 
sorry

end nth_equation_l657_657067


namespace side_of_beef_weight_after_processing_l657_657197

theorem side_of_beef_weight_after_processing (initial_weight : ℝ) (lost_percentage : ℝ) (final_weight : ℝ) 
  (h1 : initial_weight = 400) 
  (h2 : lost_percentage = 0.4) 
  (h3 : final_weight = initial_weight * (1 - lost_percentage)) : 
  final_weight = 240 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end side_of_beef_weight_after_processing_l657_657197


namespace side_length_percentage_error_l657_657210

variable (s s' : Real)
-- Conditions
-- s' = s * 1.06 (measured side length is 6% more than actual side length)
-- (s'^2 - s^2) / s^2 * 100% = 12.36% (percentage error in area)

theorem side_length_percentage_error 
    (h1 : s' = s * 1.06)
    (h2 : (s'^2 - s^2) / s^2 * 100 = 12.36) :
    ((s' - s) / s) * 100 = 6 := 
sorry

end side_length_percentage_error_l657_657210


namespace distinct_management_subcommittees_count_l657_657129

theorem distinct_management_subcommittees_count
  (total_members : ℕ)
  (managers : ℕ)
  (subcommittee_size : ℕ)
  (non_managers := total_members - managers)
  (at_least_two_managers : ℕ) :
  total_members = 12 ∧ managers = 5 ∧ subcommittee_size = 5 ∧ at_least_two_managers = 2 →
  (nat.choose total_members subcommittee_size) -
  ((nat.choose non_managers subcommittee_size) +
  (nat.choose managers 1 * nat.choose non_managers (subcommittee_size - 1)))
  = 596 :=
by {
  intros,
  sorry
}

end distinct_management_subcommittees_count_l657_657129


namespace experiment_probabilities_l657_657541

noncomputable def probability_left_urn_emptied (a b : ℕ) : ℝ :=
  b / (a + b : ℝ)

noncomputable def probability_right_urn_emptied (a b : ℕ) : ℝ :=
  a / (a + b : ℝ)

noncomputable def probability_experiment_never_ends : ℝ :=
  0

theorem experiment_probabilities (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  probability_left_urn_emptied a b = b / (a + b : ℝ) ∧
  probability_right_urn_emptied a b = a / (a + b : ℝ) ∧
  probability_experiment_never_ends = 0 :=
by
  split
  · -- Prove that the left urn probability is correct
    unfold probability_left_urn_emptied
    sorry
  split
  · -- Prove that the right urn probability is correct
    unfold probability_right_urn_emptied
    sorry
  · -- Prove that the probability of the experiment never ending is 0
    unfold probability_experiment_never_ends
    rfl

end experiment_probabilities_l657_657541


namespace value_of_expression_l657_657384

def f (x : ℝ) : ℝ := x^2 - 3*x + 7
def g (x : ℝ) : ℝ := x + 2

theorem value_of_expression : f (g 3) - g (f 3) = 8 :=
by
  sorry

end value_of_expression_l657_657384


namespace thirteen_percent_greater_than_80_l657_657160

theorem thirteen_percent_greater_than_80 (x : ℝ) (h : x = 1.13 * 80) : x = 90.4 :=
sorry

end thirteen_percent_greater_than_80_l657_657160


namespace moles_CO₂_formed_l657_657240

-- Define the initial amounts of each substance
def initial_moles_HNO₃ : ℕ := 2
def initial_moles_NaHCO₃ : ℕ := 2
def initial_moles_NH₄Cl : ℕ := 1

-- Define the balanced chemical equations
def first_reaction : ℕ → ℕ → ℕ := λ moles_NaHCO₃ moles_HNO₃, min moles_NaHCO₃ moles_HNO₃
def second_reaction : ℕ → ℕ := λ moles_NH₄Cl, moles_NH₄Cl

-- Proof problem statement
theorem moles_CO₂_formed : first_reaction initial_moles_NaHCO₃ initial_moles_HNO₃ = 2 :=
by sorry

end moles_CO₂_formed_l657_657240


namespace decimal_rep_150th_l657_657581

theorem decimal_rep_150th (num : ℚ := 13/17) (dec_repr : String := "7647058823529411") (length_seq : Nat := 16) :
  (dec_repr[(150 % length_seq) - 1] = '5') :=
by
  sorry

end decimal_rep_150th_l657_657581


namespace find_angles_of_triangle_ABC_l657_657820
noncomputable def angles_of_triangle_ABC : Prop :=
  ∃ (α β γ : ℝ), α + β + γ = 180 ∧
    α = 72 ∧ β = 72 ∧ γ = 36 ∧
    let O := circumcenter (triangle.mk A B C) in
    incenter (triangle.mk A D C) = O ∧
    D ∈ (bisector (angle A B C)) ∧
    B ≠ C 

theorem find_angles_of_triangle_ABC (A B C D : Point) 
  (h1 : D ∈ (line_segment B C))
  (h2 : D ∈ (bisector (angle A B C))) 
  (h3 : circumcenter (triangle.mk A B C) = incenter (triangle.mk A D C))
  : angles_of_triangle_ABC :=
sorry

end find_angles_of_triangle_ABC_l657_657820


namespace intersection_proof_l657_657033

noncomputable def M : Set ℕ := {1, 3, 5, 7, 9}

noncomputable def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_proof : M ∩ (N ∩ Set.univ) = {5, 7, 9} :=
by sorry

end intersection_proof_l657_657033


namespace min_participants_correct_l657_657189

variable {a b : ℕ}

-- Define input set A
def A := {a | a > 0 ∧ 6 ∣ a}

-- Function to calculate minimum participants
def min_participants (s : set ℕ) : ℕ :=
  let max_a := s.max' (sorry : s.nonempty) in
  max_a / 2 + 3

-- Theorem statement to prove
theorem min_participants_correct (s : set ℕ) (hA : ∀ a ∈ s, a ∈ A) :
  ∃ n, min_participants s = n :=
by
  sorry

end min_participants_correct_l657_657189


namespace evaluate_f_neg3_l657_657991

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

theorem evaluate_f_neg3 (a b c : ℝ) (h : f 3 a b c = 11) : f (-3) a b c = -9 := by
  sorry

end evaluate_f_neg3_l657_657991


namespace number_of_guests_l657_657143

def cook_per_minute : ℕ := 10
def time_to_cook : ℕ := 80
def guests_ate_per_guest : ℕ := 5
def guests_to_serve : ℕ := 20 -- This is what we'll prove.

theorem number_of_guests 
    (cook_per_8min : cook_per_minute = 10)
    (total_time : time_to_cook = 80)
    (eat_rate : guests_ate_per_guest = 5) :
    (time_to_cook * cook_per_minute) / guests_ate_per_guest = guests_to_serve := 
by 
  sorry

end number_of_guests_l657_657143


namespace part_a_part_b_l657_657985

open Nat

theorem part_a (n: ℕ) (h_pos: 0 < n) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, k > 0 ∧ n = 3 * k :=
sorry

theorem part_b (n: ℕ) (h_pos: 0 < n) : (2^n + 1) % 7 ≠ 0 :=
sorry

end part_a_part_b_l657_657985


namespace number_of_negative_x_l657_657339

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l657_657339


namespace annual_interest_proof_l657_657088

variable (total_amount : ℝ)
variable (first_part : ℝ)
variable (second_part : ℝ)
variable (interest_rate_first : ℝ)
variable (interest_rate_second : ℝ)
variable (annual_interest_first : ℝ)
variable (annual_interest_second : ℝ)

-- Given conditions
def conditions := total_amount = 3000 ∧
                  first_part = 299.99999999999994 ∧ 
                  second_part = total_amount - first_part ∧ 
                  interest_rate_first = 0.03 ∧
                  interest_rate_second = 0.05 ∧
                  annual_interest_first = first_part * interest_rate_first ∧
                  annual_interest_second = second_part * interest_rate_second

-- To prove
theorem annual_interest_proof (h : conditions) : 
  (annual_interest_first + annual_interest_second) = 144 :=
by
  sorry

end annual_interest_proof_l657_657088


namespace fraction_of_students_participated_l657_657998

theorem fraction_of_students_participated (total_students : ℕ) (did_not_participate : ℕ)
  (h_total : total_students = 39) (h_did_not_participate : did_not_participate = 26) :
  (total_students - did_not_participate) / total_students = 1 / 3 :=
by
  sorry

end fraction_of_students_participated_l657_657998


namespace area_of_triangle_PCF_l657_657821

theorem area_of_triangle_PCF
  {A B C P D E F : Type}
  [equilateral_triangle A B C]
  (P_inside : point_inside_triangle P A B C)
  (perpendiculars : feet_of_perpendiculars P A B C D E F)
  (area_ABC : area A B C = 2028)
  (area_PAD : area P A D = 192)
  (area_PBE : area P B E = 192) :
  area P C F = 1644 := by
  sorry

end area_of_triangle_PCF_l657_657821


namespace triangle_area_l657_657204

theorem triangle_area (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) 
  (h₄ : a^2 + b^2 = c^2) : (1 / 2 : ℝ) * a * b = 30 := 
by 
  rw [h₁, h₂]
  norm_num

end triangle_area_l657_657204


namespace rectangular_solid_surface_area_l657_657987

theorem rectangular_solid_surface_area
  (length : ℕ) (width : ℕ) (depth : ℕ)
  (h_length : length = 9) (h_width : width = 8) (h_depth : depth = 5) :
  2 * (length * width + width * depth + length * depth) = 314 := 
  by
  sorry

end rectangular_solid_surface_area_l657_657987


namespace existence_n0_xn_convergent_nxn_l657_657993

noncomputable theory
open Real

-- First part
theorem existence_n0_xn {f : ℝ → ℝ} (hf : ∀ x, 0 ≤ x → 0 < f x) (hf_cont : ContinuousOn f (Set.Ici 0)) :
  ∃ (n0 : ℕ) (x : ℕ → ℝ), 
    (∀ n > n0, 0 < x n ∧ (n : ℝ) * (∫ t in 0..(x n), f t) = 1) := sorry

-- Second part
theorem convergent_nxn {f : ℝ → ℝ} {n0 : ℕ} {x : ℕ → ℝ} (hf : ∀ x, 0 ≤ x → 0 < f x)
  (hf_cont : ContinuousOn f (Set.Ici 0))
  (hx : ∀ n > n0, 0 < x n ∧ (n : ℝ) * (∫ t in 0..(x n), f t) = 1) :
  ∃ l : ℝ, (∀ n > n0, (n : ℝ) * x n → l) ∧ l = 1 / f 0 := sorry

end existence_n0_xn_convergent_nxn_l657_657993


namespace gary_money_after_sale_l657_657356

theorem gary_money_after_sale :
  let initial_money := 73.0
  let sale_amount := 55.0
  initial_money + sale_amount = 128.0 :=
by
  let initial_money := 73.0
  let sale_amount := 55.0
  show initial_money + sale_amount = 128.0
  sorry

end gary_money_after_sale_l657_657356


namespace count_negative_values_correct_l657_657284

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l657_657284


namespace find_k_l657_657628

theorem find_k (k : ℝ) (h1 : (5 - 3) ≠ 0) (h2 : (11 - 3) ≠ 0)
    (h3 : (15 - 7) / (11 - 3) = 1) : k = 9 :=
by
  have slope1 := (15 - 7) / (11 - 3)
  have slope2 := (k - 7) / (5 - 3)
  have h : slope1 = slope2, from h3
  -- Using the given slopes, we solve for \( k \)
  have h_solve : k - 7 = 2, from sorry
  exact (eq_add_of_sub_eq h_solve).symm

end find_k_l657_657628


namespace exists_consecutive_integers_with_unique_prime_exponents_l657_657374
-- Step: Import necessary library

-- Step: Declare the theorem statement
theorem exists_consecutive_integers_with_unique_prime_exponents (n : ℕ) (hn : 0 < n) :
  ∃ x : ℕ, ∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → 
  (∀ p : ℕ, p.prime → p ∣ (x + i) → (¬ p^2 ∣ (x + i))) :=
sorry

end exists_consecutive_integers_with_unique_prime_exponents_l657_657374


namespace product_all_possible_values_of_c_l657_657684

noncomputable def g (c : ℝ) (x : ℝ) : ℝ := c / (3 * x - 4)

theorem product_all_possible_values_of_c : 
  let c_values := {c : ℝ | g c 3 = g⁻¹ c (c+2)} in
  ∏ c in c_values, c = -8 / 3 :=
sorry

end product_all_possible_values_of_c_l657_657684


namespace set_intersection_example_l657_657740

theorem set_intersection_example (A : Set ℝ) (B : Set ℝ):
  A = { -1, 1, 2, 4 } → 
  B = { x | |x - 1| ≤ 1 } → 
  A ∩ B = {1, 2} :=
by
  intros hA hB
  sorry

end set_intersection_example_l657_657740


namespace square_of_1027_l657_657226

theorem square_of_1027 :
  1027 * 1027 = 1054729 :=
by
  sorry

end square_of_1027_l657_657226


namespace number_of_negative_x_l657_657336

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l657_657336


namespace negative_values_of_x_l657_657305

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l657_657305


namespace dihedral_angle_measure_l657_657490

theorem dihedral_angle_measure
  (A B C D E F G : Point)
  (is_regular_tetrahedron : regular_tetrahedron A B C D)
  (E_is_midpoint_AB : midpoint E A B)
  (F_is_midpoint_BC : midpoint F B C)
  (G_is_midpoint_CD : midpoint G C D) :
  dihedral_angle C F G E = π - arccot (sqrt 2 / 2) :=
sorry

end dihedral_angle_measure_l657_657490


namespace KochCurve_MinkowskiDimension_l657_657660

noncomputable def minkowskiDimensionOfKochCurve : ℝ :=
  let N (n : ℕ) := 3 * (4 ^ (n - 1))
  (Real.log 4) / (Real.log 3)

theorem KochCurve_MinkowskiDimension : minkowskiDimensionOfKochCurve = (Real.log 4) / (Real.log 3) := by
  sorry

end KochCurve_MinkowskiDimension_l657_657660


namespace percentage_discount_l657_657657

-- Names and definitions based on conditions
def original_price : ℝ := 120
def total_cost : ℝ := 63
def sales_tax_rate : ℝ := 0.05

-- We need to prove the percentage discount D is 50
theorem percentage_discount : ∃ (D : ℝ), 
  let P := original_price - (D / 100) * original_price in
  let T := P * (1 + sales_tax_rate) in
  T = total_cost ∧ D = 50 := by
  sorry

end percentage_discount_l657_657657


namespace total_cost_after_discount_l657_657625

theorem total_cost_after_discount:
  let num_children := 6
  let num_adults := 10
  let num_seniors := 4
  let cost_per_child := 10
  let cost_per_adult := 16
  let cost_per_senior := 12
  let discount_rate := 0.15
  let total_cost_before_discount := (num_children * cost_per_child) + (num_adults * cost_per_adult) + (num_seniors * cost_per_senior)
  let discount := discount_rate * total_cost_before_discount 
  let final_cost := total_cost_before_discount - discount
  in final_cost = 227.80 :=
by
  -- sorry to skip the proof
  sorry

end total_cost_after_discount_l657_657625


namespace incorrect_statement_c_l657_657990

-- Definitions from the given problem
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def is_increasing (S : ℕ → ℝ) : Prop := ∀ n, S (n + 1) > S n

-- The actual math proof problem
theorem incorrect_statement_c (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a d)
  (h_S : ∀ n, S n = sum_of_first_n_terms a n)
  (h_increasing : is_increasing S) :
  ¬ ∀ n : ℕ, n > 0 → S n > 0 := 
sorry

end incorrect_statement_c_l657_657990


namespace new_solution_liquid_x_percentage_l657_657603

theorem new_solution_liquid_x_percentage:
  (initial_solution_y : ℝ) (liquid_x_percent : ℝ) (water_percent : ℝ) (evaporated_water : ℝ) (added_solution_y : ℝ)
  (liquid_x_mass_initial := initial_solution_y * liquid_x_percent / 100)
  (water_mass_initial := initial_solution_y * water_percent / 100)
  (remaining_liquid_x_mass := liquid_x_mass_initial)
  (remaining_water_mass := water_mass_initial - evaporated_water)
  (added_liquid_x_mass := added_solution_y * liquid_x_percent / 100)
  (added_water_mass := added_solution_y * water_percent / 100)
  (total_liquid_x_mass := remaining_liquid_x_mass + added_liquid_x_mass)
  (total_water_mass := remaining_water_mass + added_water_mass)
  (total_new_solution_mass := total_liquid_x_mass + total_water_mass)
  (new_solution_liquid_x_percent := (total_liquid_x_mass / total_new_solution_mass) * 100):
  ∀ (initial_solution_y = 8) (liquid_x_percent = 30) (water_percent = 70)
    (evaporated_water = 4) (added_solution_y = 4),
  new_solution_liquid_x_percent = 45 :=
by
  sorry

end new_solution_liquid_x_percentage_l657_657603


namespace combined_score_of_three_students_left_l657_657907

variable (T S : ℕ) (avg16 avg13 : ℝ) (N16 N13 : ℕ)

theorem combined_score_of_three_students_left (h_avg16 : avg16 = 62.5) 
  (h_avg13 : avg13 = 62.0) (h_N16 : N16 = 16) (h_N13 : N13 = 13) 
  (h_total16 : T = avg16 * N16) (h_total13 : T - S = avg13 * N13) :
  S = 194 :=
by
  sorry

end combined_score_of_three_students_left_l657_657907


namespace solution_set_of_inequality_l657_657501

variable {x : ℝ}
variable {f : ℝ → ℝ}

-- Conditions
axiom differentiable_f : ∀ x ∈ set.Iio 0, differentiable_at ℝ f x
axiom condition_on_f : ∀ x ∈ set.Iio 0, (2 * f x / x + deriv f x) < 0

-- Question: Prove that the inequality ((x + 2015)^2 * f (x + 2015) - 4 * f (-2) > 0) has the solution set (-∞, -2017).
theorem solution_set_of_inequality :
  (∀ x, (x + 2015) ^ 2 * f (x + 2015) - 4 * f (-2) > 0 ↔ x < -2017)
:= by
  sorry

end solution_set_of_inequality_l657_657501


namespace incorrect_rounding_l657_657570

theorem incorrect_rounding :
  (∀ x : ℝ, round_nearest x 0.1 ≠ 0.05019 → false) ∧
  (∀ x : ℝ, round_nearest x 0.01 ≠ 0.05019 → false) ∧
  (∀ x : ℝ, round_nearest x 0.001 ≠ 0.05019 → true) ∧
  (∀ x : ℝ, round_nearest x 0.0001 ≠ 0.05019 → false) := by
  sorry

def round_nearest (x : ℝ) (t : ℝ) : ℝ := 
by sorry

end incorrect_rounding_l657_657570


namespace cyclic_quadrilateral_area_ineq_l657_657477

variable {A B C D E F : Type*}
variable [OrderedCommRing A]
variable [OrderedCommRing B]
variable [OrderedCommRing C]
variable [HMetricSpace D]
variable [HMetricSpace E]
variable [HMetricSpace F]

noncomputable def cyclic_quadrilateral (A B C D : Type*) : Prop :=
  -- Definition for cyclic quadrilateral
  sorry

theorem cyclic_quadrilateral_area_ineq
  (D E F : Type*)
  (on_sides : D ≠ E ∧ E ≠ F ∧ F ≠ D)
  (cyclic_quad : cyclic_quadrilateral A F D E) :
  (4 * \mathcal{A}[D E F]) / \mathcal{A}[A B C] ≤ (\frac{E F}{A D})^2 :=
sorry

end cyclic_quadrilateral_area_ineq_l657_657477


namespace numeral_150th_decimal_place_l657_657585

theorem numeral_150th_decimal_place (k : ℕ) (h : k = 150) : 
  (decimal_place (13 / 17) k) = 5 :=
sorry

end numeral_150th_decimal_place_l657_657585


namespace h3_is_328_minus_52sqrt34_l657_657010

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1
def g (x : ℝ) : ℝ := real.sqrt (f x) - x^2
def h (x : ℝ) : ℝ := f (g x)

theorem h3_is_328_minus_52sqrt34 : h 3 = 328 - 52 * real.sqrt 34 :=
by
  sorry

end h3_is_328_minus_52sqrt34_l657_657010


namespace range_of_x_l657_657397

theorem range_of_x (x : ℝ) (h : Real.log (x - 1) < 1) : 1 < x ∧ x < Real.exp 1 + 1 :=
by
  sorry

end range_of_x_l657_657397


namespace inequality_solution_set_l657_657554

def solution_set (a b x : ℝ) : Set ℝ := {x | |a - b * x| - 5 ≤ 0}

theorem inequality_solution_set (x : ℝ) :
  solution_set 4 3 x = {x | - (1 : ℝ) / 3 ≤ x ∧ x ≤ 3} :=
by {
  sorry
}

end inequality_solution_set_l657_657554


namespace initial_packs_l657_657666

def num_invitations_per_pack := 3
def num_friends := 9
def extra_invitations := 3
def total_invitations := num_friends + extra_invitations

theorem initial_packs (h : total_invitations = 12) : (total_invitations / num_invitations_per_pack) = 4 :=
by
  have h1 : total_invitations = 12 := by exact h
  have h2 : num_invitations_per_pack = 3 := by exact rfl
  have H_pack : total_invitations / num_invitations_per_pack = 4 := by sorry
  exact H_pack

end initial_packs_l657_657666


namespace polar_to_rectangular_coordinates_l657_657233

noncomputable def r : ℝ := 4
noncomputable def theta : ℝ := Real.pi / 4

theorem polar_to_rectangular_coordinates :
  ∃ x y : ℝ, x = r * Real.cos theta ∧ y = r * Real.sin theta ∧ x = 2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 := 
by
  use 2 * Real.sqrt 2
  use 2 * Real.sqrt 2
  split;
  {
    sorry,
    sorry
  }


end polar_to_rectangular_coordinates_l657_657233


namespace max_profit_l657_657180

def ProductA := ℕ  -- Tons of Product A
def ProductB := ℕ  -- Tons of Product B
def MaterialA := ℕ  -- Tons of Material A
def MaterialB := ℕ  -- Tons of Material B

-- Material constraints
def MaterialA_used (x : ProductA) (y : ProductB) : MaterialA := 3 * x + y
def MaterialB_used (x : ProductA) (y : ProductB) : MaterialB := 2 * x + 3 * y

-- Profit function
def profit (x : ProductA) (y : ProductB) : ℕ := 50000 * x + 30000 * y

-- Constraints
def materialA_constraint (x : ProductA) (y : ProductB) : Prop := MaterialA_used x y ≤ 13
def materialB_constraint (x : ProductA) (y : ProductB) : Prop := MaterialB_used x y ≤ 18

-- Main Theorem to Prove:
theorem max_profit : ∃ (x y : ℕ), materialA_constraint x y ∧ materialB_constraint x y ∧ profit x y = 270000 := by
    sorry

end max_profit_l657_657180


namespace slower_train_speed_l657_657966

noncomputable def speed_of_slower_train (v: ℝ) : Prop :=
  let faster_train_speed: ℝ := 46 -- speed in km/hr
  let passing_time: ℝ := 45 -- time in seconds
  let train_length: ℝ := 62.5 -- length in meters
  let relative_distance: ℝ := 2 * train_length -- relative distance in meters
  let convert_to_mps: ℝ := 5 / 18 -- conversion factor from km/hr to m/s
  in 
    relative_distance = ((faster_train_speed - v) * convert_to_mps * passing_time)

theorem slower_train_speed : speed_of_slower_train 37 := 
  sorry

end slower_train_speed_l657_657966


namespace parallel_vector_magnitude_l657_657772

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem parallel_vector_magnitude:
  ∀ (x : ℝ),
  let a := (1, 2)
  let b := (x, 6)
  (1 * 6 - 2 * x = 0) →
  x = 3 ∧ vector_magnitude (a.1 - b.1, a.2 - b.2) = 2 * Real.sqrt 5 :=
begin
  intros x a b h,
  have x_val : x = 3,
  { linarith },
  have mag_val : vector_magnitude (1 - x, 2 - 6) = 2 * Real.sqrt 5,
  { calc
      vector_magnitude (1 - x, 2 - 6) = vector_magnitude (-2, -4) : by simp [x_val]
      ... = Real.sqrt ((-2) ^ 2 + (-4) ^ 2) : by refl
      ... = Real.sqrt (4 + 16) : by norm_num
      ... = Real.sqrt 20 : by refl
      ... = 2 * Real.sqrt 5 : by linarith [Real.sqrt_eq_rpow] },
  exact ⟨x_val, mag_val⟩,
end

end parallel_vector_magnitude_l657_657772


namespace segment_less_than_largest_side_l657_657196

variables {A B C M : Type} [MetricSpace A]

theorem segment_less_than_largest_side {A B C M : Point} 
  (h_triangle: IsTriangle A B C) (h_M_on_BC: M ∈ Segment B C):
  dist A M < max (dist A B) (dist A C) :=
sorry

end segment_less_than_largest_side_l657_657196


namespace largest_multiple_of_7_smaller_than_neg_50_l657_657148

theorem largest_multiple_of_7_smaller_than_neg_50 : ∃ n, (∃ k : ℤ, n = 7 * k) ∧ n < -50 ∧ ∀ m, (∃ j : ℤ, m = 7 * j) ∧ m < -50 → m ≤ n :=
by
  sorry

end largest_multiple_of_7_smaller_than_neg_50_l657_657148


namespace sequence_not_arithmetic_sum_of_seq_terms_l657_657402

def sequence (n : ℕ) : ℤ :=
  if n ≤ 7 then n + 1 else n - 1

def is_not_arithmetic_sequence : Prop :=
  sequence 2 - sequence 1 ≠ sequence 8 - sequence 7

def sum_of_first_n_terms (n : ℕ) : ℤ :=
  if n ≤ 7 then (n^2 / 2) + (3 * n / 2)
  else ((n^2 - n) / 2) + 14

theorem sequence_not_arithmetic
  (h : ∀ n, sequence n = if n ≤ 7 then n + 1 else n - 1) :
  is_not_arithmetic_sequence :=
by
  unfold is_not_arithmetic_sequence
  simp [sequence]

theorem sum_of_seq_terms
  (h₁ : ∀ n, sequence n = if n ≤ 7 then n + 1 else n - 1)
  (n : ℕ) :
  sum_of_first_n_terms n =
    if n ≤ 7 then (n^2 / 2) + (3 * n / 2)
    else ((n^2 - n) / 2) + 14 :=
by
  unfold sum_of_first_n_terms
  simp [sequence]
  sorry -- the proof steps are omitted as per the instructions

end sequence_not_arithmetic_sum_of_seq_terms_l657_657402


namespace prove_minimum_bailing_rate_l657_657600

-- Define the conditions as hypotheses
def initial_distance : ℝ := 3 -- miles
def initial_leak_rate : ℝ := 8 -- gallons per minute
def additional_leak_rate : ℝ := 10 -- gallons per minute
def max_water_capacity : ℝ := 50 -- gallons
def rowing_speed : ℝ := 2 -- miles per hour
def halfway_distance : ℝ := initial_distance / 2
def rowing_speed_in_minutes : ℝ := rowing_speed / 60 -- miles per minute

-- Define the total time to shore in minutes
def total_time_to_shore : ℝ := initial_distance / rowing_speed * 60

-- Define the row time to halfway in minutes
def time_to_halfway : ℝ := halfway_distance / rowing_speed * 60

-- Define the total water intake up to halfway
def water_intake_initial : ℝ := initial_leak_rate * time_to_halfway

-- Define the total water intake from halfway to shore
def water_intake_additional : ℝ := (initial_leak_rate + additional_leak_rate) * time_to_halfway

-- Define the total water intake
def total_water_intake : ℝ := water_intake_initial + water_intake_additional

-- Define the minimum bailing rate required in gallons per minute
def minimum_bailing_rate : ℝ := (total_water_intake - max_water_capacity) / total_time_to_shore

-- Define the Lean theorem to prove the minimum bailing rate
theorem prove_minimum_bailing_rate : minimum_bailing_rate = 12 := by
  sorry

end prove_minimum_bailing_rate_l657_657600


namespace donation_calculation_l657_657878

/-- Patricia's initial hair length -/
def initial_length : ℕ := 14

/-- Patricia's hair growth -/
def growth_length : ℕ := 21

/-- Desired remaining hair length after donation -/
def remaining_length : ℕ := 12

/-- Calculate the donation length -/
def donation_length (L G R : ℕ) : ℕ := (L + G) - R

-- Theorem stating the donation length required for Patricia to achieve her goal.
theorem donation_calculation : donation_length initial_length growth_length remaining_length = 23 :=
by
  -- Proof omitted
  sorry

end donation_calculation_l657_657878


namespace function_range_domain_l657_657937

noncomputable def function_range (x : ℝ) : ℝ := (Real.sqrt (3 - x)) / (x + 2)

theorem function_range_domain
  (x : ℝ)
  (h1 : 3 - x ≥ 0)
  (h2 : x ≠ -2) :
  x ≤ 3 ∧ x ≠ -2 :=
by
  split;
  sorry

end function_range_domain_l657_657937


namespace negative_values_of_x_l657_657304

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l657_657304


namespace max_intersections_l657_657014

theorem max_intersections (d1 d2 : Line) 
  (black_points : Finset Point) (white_points : Finset Point) 
  (h_parallel : IsParallel d1 d2)
  (h_black : black_points.card = 11) 
  (h_white : white_points.card = 16) :
  ∃ max_intersection_points, max_intersection_points = 6600 :=
by
  sorry

end max_intersections_l657_657014


namespace intersection_point_y_difference_zero_l657_657544

def g (x : ℝ) : ℝ := 2 - x^2 + x^4
def h (x : ℝ) : ℝ := -1 + x^2 + x^4

theorem intersection_point_y_difference_zero :
  ∀ x, g x = h x → (g x - h x) = 0 :=
begin
  intros x hx,
  rw [g, h] at hx,
  sorry
end

end intersection_point_y_difference_zero_l657_657544


namespace simplify_expression1_simplify_expression2_l657_657530

-- Define variables as real numbers or appropriate domains
variables {a b x y: ℝ}

-- Problem 1
theorem simplify_expression1 : (2 * a - b) - (2 * b - 3 * a) - 2 * (a - 2 * b) = 3 * a + b :=
by sorry

-- Problem 2
theorem simplify_expression2 : (4 * x^2 - 5 * x * y) - (1 / 3 * y^2 + 2 * x^2) + 2 * (3 * x * y - 1 / 4 * y^2 - 1 / 12 * y^2) = 2 * x^2 + x * y - y^2 :=
by sorry

end simplify_expression1_simplify_expression2_l657_657530


namespace g_max_value_l657_657750

noncomputable def f : ℝ → ℝ := λ x, x^2 - 4*x + 3

def g (a : ℝ) (x : ℝ) : ℝ := f(x) + a * x

def h (a : ℝ) : ℝ :=
if a > 3 then 2*a - 1 else 8 - a

-- The theorem statement that needs to be proven.
theorem g_max_value (a : ℝ) : 
  ∀ x ∈ set.Icc (-1 : ℝ) 2, g a x ≤ h a :=
sorry

end g_max_value_l657_657750


namespace maximum_value_cosine_sine_combination_l657_657690

noncomputable def max_cosine_sine_combination : Real :=
  let g (θ : Real) := (Real.cos (θ / 2)) * (1 + Real.sin θ)
  have h₁ : ∃ θ : Real, -Real.pi / 2 < θ ∧ θ < Real.pi / 2 :=
    sorry -- Existence of such θ is trivial
  Real.sqrt 2

theorem maximum_value_cosine_sine_combination :
  ∀ θ : Real, -Real.pi / 2 < θ ∧ θ < Real.pi / 2 →
  (Real.cos (θ / 2)) * (1 + Real.sin θ) ≤ Real.sqrt 2 :=
by
  intros θ h
  let y := (Real.cos (θ / 2)) * (1 + Real.sin θ)
  have hy : y ≤ Real.sqrt 2 := sorry
  exact hy

end maximum_value_cosine_sine_combination_l657_657690


namespace proof_problem_l657_657728

-- Given definitions and assumptions
def has_property_A (A : Set ℝ) (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ t ∈ A, ∀ x ∈ D, f x ≤ f (x + t)

-- Question 1
def Q1 (A : Set ℝ) : Prop :=
  ∀ (f g : ℝ → ℝ) (D : Set ℝ),
    has_property_A A f D → has_property_A A g D → f = (λ x, -x) ∧ ¬(g = (λ x, 2 * x))

-- Question 2
def Q2 (A : Set ℝ) : Prop :=
  ∀ (f : ℝ → ℝ) (a : ℝ),
    (∀ x, x ∈ Set.Ici a → ∀ t ∈ A, f x ≤ f (x + t)) → a ∈ Set.Ici 1

-- Question 3
def Q3 (A : Set ℝ) : Prop :=
  ∀ (f : ℤ → ℤ) (m : ℤ),
    (∀ x, f x = f (x - 2) ∧ f x = f (x + m)) → ∃ k, m = 2 * k + 1

-- Statements to prove
theorem proof_problem : Q1 ({-1}) ∧ Q2 Set.Ioo 0 1 ∧ Q3 (Set.to_finset {-2}) := sorry

end proof_problem_l657_657728


namespace probability_jane_wins_total_possibilities_jane_wins_probability_l657_657555

def non_negative_difference (x y : ℕ) : ℕ := abs(x - y)

def winning_pairs (x y : ℕ) : Prop := non_negative_difference x y < 2

theorem probability_jane_wins :
  (finsets.univ.finset_product finsets.univ).filter (λ (pair : ℕ × ℕ), winning_pairs pair.1 pair.2).card = 16 :=
by
  sorry

theorem total_possibilities :
  (finsets.univ.finset_product finsets.univ).card = 36 :=
by
  sorry

theorem jane_wins_probability :
  (probability_jane_wins.to_nat / total_possibilities.to_nat) = 4 / 9 :=
by
  sorry

end probability_jane_wins_total_possibilities_jane_wins_probability_l657_657555


namespace triangle_relation_l657_657496

variables {A B C D P Q X Y : Type*}
variables [triangle ABC] [AB < AC]
variables [incircle ABC tangentside BC D]
variables [incircle_intersection_perpendicular_bisector BC X Y]
variables {line_1 : line AX intersects_side BC P} [line_1 : line AX intersects_side BC Q]

theorem triangle_relation 
(h1 : DP * DQ = (AC - AB) ^ 2) : 
AB + AC = 3 * BC := 
sorry

end triangle_relation_l657_657496


namespace card_placement_correct_l657_657798

open Nat

def num_ways_card_placement : Nat :=
  let num_cards := 6
  let card_labels := {1, 2, 3, 4, 5, 6}
  let num_envelopes := 3
  let cards_in_each_envelope := 2
  let card_grouping_condition := {1, 2} -- 1 and 2 must be in the same envelope
  9

theorem card_placement_correct :
  num_ways_card_placement = 9 :=
by
  sorry

end card_placement_correct_l657_657798


namespace probability_is_1_div_4_l657_657138

/-
We need to define a function to determine if two numbers are consecutive integers.
-/

def are_consecutive (a b : ℕ) : Prop := (a + 1 = b) ∨ (b + 1 = a)

/-
Define the lists representing the cards in each box.
-/

def box1 := [1, 2, 3, 4, 5]
def box2 := [2, 3, 6, 8]

/-
Calculate the number of ways to draw two cards such that they are consecutive.
-/

def num_consecutive_pairs : ℕ :=
  (box1.product box2).count (λ (p : ℕ × ℕ), are_consecutive p.fst p.snd)

/-
Calculate the total number of possible outcomes.
-/

def total_outcomes : ℕ := box1.length * box2.length

/-
Define the probability of drawing two consecutive cards.
-/

def probability_consecutive : ℚ := num_consecutive_pairs / total_outcomes

/-
The theorem that needs to be proved:
-/

theorem probability_is_1_div_4 :
  probability_consecutive = 1 / 4 :=
by
  /- Start your proof here -/
  sorry

end probability_is_1_div_4_l657_657138


namespace negation_of_universal_proposition_l657_657519

theorem negation_of_universal_proposition :
  (∀ x : ℝ, x^2 + 1 > 0) → ¬(∃ x : ℝ, x^2 + 1 ≤ 0) := sorry

end negation_of_universal_proposition_l657_657519


namespace train_length_l657_657984

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (speed_conv_factor : ℚ) :
  speed_kmh = 180 →
  time_s = 7 →
  speed_conv_factor = 5 / 18 →
  (speed_kmh * speed_conv_factor * time_s = 350) :=
by
  intros h_speed h_time h_conv_factor
  rw [h_speed, h_time, h_conv_factor]
  norm_num
  sorry

end train_length_l657_657984


namespace max_candy_remainder_l657_657776

theorem max_candy_remainder (x : ℕ) : x % 11 < 11 ∧ (∀ r : ℕ, r < 11 → x % 11 ≤ r) → x % 11 = 10 := 
sorry

end max_candy_remainder_l657_657776


namespace find_radius_l657_657799

-- Define the given values
def arc_length : ℝ := 4
def central_angle : ℝ := 2

-- We need to prove this statement
theorem find_radius (radius : ℝ) : arc_length = radius * central_angle → radius = 2 := 
by
  sorry

end find_radius_l657_657799


namespace inclination_angle_of_line_l657_657115

theorem inclination_angle_of_line (α : ℝ) :
  ( ∃ y, ∀ x:ℝ, y = -√3 * x + 3 )→
  ( -√3 = Real.tan α ) ∧ ( 0 ≤ α ∧ α < Real.pi ) →
  α = 2 * Real.pi / 3 :=
by
  sorry

end inclination_angle_of_line_l657_657115


namespace correct_calculation_l657_657597

theorem correct_calculation :
  (∀ (b x : ℝ), ¬(-sqrt 2 + 2 * sqrt 2 = -sqrt 2) ∧ ¬((-1)^0 = 0) ∧ ((b^2)^3 = b^6) ∧ ¬((x + 1)^2 = x^2 + 1)) :=
by {
  intros b x,
  split,
  { intro h, -- ¬(-sqrt 2 + 2 * sqrt 2 = -sqrt 2)
    linarith, 
  },
  split,
  { intro h, -- ¬((-1)^0 = 0)
    norm_num at h, 
  },
  split,
  { exact pow_mul b 2 3, -- ((b^2)^3 = b^6)
  },
  { intro h, -- ¬((x + 1)^2 = x^2 + 1)
    have h' := binom_expansion x 1 2,
    linarith, 
  }
}

end correct_calculation_l657_657597


namespace n_five_minus_n_divisible_by_30_l657_657891

theorem n_five_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end n_five_minus_n_divisible_by_30_l657_657891


namespace mutually_exclusive_not_complementary_l657_657807

def group : Finset (String × String) := {("boy1", "boy"), ("boy2", "boy"), ("boy3", "boy"), ("girl1", "girl"), ("girl2", "girl")}
def selection_size : ℕ := 2

def event_at_least_one_boy (s : Finset (String × String)) : Prop :=
  ∃ x ∈ s, x.2 = "boy"

def event_all_girls (s : Finset (String × String)) : Prop :=
  ∀ x ∈ s, x.2 = "girl"

theorem mutually_exclusive_not_complementary : 
  ∃ (s : Finset (String × String)), s.card = selection_size ∧ event_at_least_one_boy s ∧ event_all_girls s :=
sorry

end mutually_exclusive_not_complementary_l657_657807


namespace solve_math_problem_l657_657766

noncomputable def line_l (t : ℝ) : ℝ × ℝ := 
  ( (1/2) + (sqrt(2)/2) * t, 
    (1/2) - (sqrt(2)/2) * t )

def ellipse_C (α : ℝ) : ℝ × ℝ := 
  ( 2 * Real.cos α, 
    Real.sin α )

def point_A_polar : ℝ × ℝ := (2, Real.pi / 3)

def A_Cartesian : ℝ × ℝ := (1/2, Real.sqrt(3)/2)

def line_l_cartesian (x y : ℝ) : Prop := 
  x + y - 1 = 0

def ellipse_C_cartesian (x y : ℝ) : Prop := 
  (x^2 / 4) + y^2 = 1

def line_intersects_ellipse_P : Prop := 
  (0, 1) ∈ { p : ℝ × ℝ | line_l_cartesian p.1 p.2 ∧ ellipse_C_cartesian p.1 p.2 }

def line_intersects_ellipse_Q : Prop := 
  ((8/5), (-3/5)) ∈ { p : ℝ × ℝ | line_l_cartesian p.1 p.2 ∧ ellipse_C_cartesian p.1 p.2 }

def length_PQ : ℝ := Real.sqrt ((8/5) ^ 2 + (8/5) ^ 2)

def distance_A_line : ℝ := (|1 + Real.sqrt(3) - 1|) / Real.sqrt(2)

def area_triangle_APQ : ℝ := 
  1/2 * length_PQ * distance_A_line

theorem solve_math_problem :
  ellipse_C_cartesian (2 * Real.cos (Real.pi/3)) (Real.sin (Real.pi/3)) ∧
  line_intersects_ellipse_P ∧
  line_intersects_ellipse_Q ∧
  point_A_polar = (2, Real.pi / 3) ∧
  A_Cartesian = (1/2, Real.sqrt(3)/2) ∧
  area_triangle_APQ = 4 * Real.sqrt(3) / 5 :=
by
  sorry

end solve_math_problem_l657_657766


namespace lattice_points_independent_of_n_l657_657012

-- Define two-variable polynomials P and Q with integer coefficients
def P : ℤ × ℤ → ℤ := sorry
def Q : ℤ × ℤ → ℤ := sorry

-- Define the sequences {a_n} and {b_n}
noncomputable def a : ℕ → ℤ
| 0     := sorry
| (n+1) := P (a n, b n)

noncomputable def b : ℕ → ℤ
| 0     := sorry
| (n+1) := Q (a n, b n)

-- Define the given positive integer k
def k : ℕ := sorry

-- Define the periodic condition and non-repetition of the first two terms
axiom periodic : (a k, b k) = (a 0, b 0)
axiom non_trivial : (a 1, b 1) ≠ (a 0, b 0)

-- Statement to prove the number of lattice points on segment is independent of n
theorem lattice_points_independent_of_n :
  ∃ d : ℤ, ∀ n : ℕ, gcd (a (n+1) - a n) (b (n+1) - b n) = d :=
sorry

end lattice_points_independent_of_n_l657_657012


namespace rearrange_digits_2102_l657_657408

theorem rearrange_digits_2102 : 
  let digits : Multiset ℕ := {2, 1, 0, 2}
  in (multiset.count 2 digits = 2) →
  (∀ n, mul n.succ 1 = n.succ) →
  (∃ valid_rearrangements : ℕ, valid_rearrangements = 9) :=
by
  assume h1 : multiset.count 2 {2, 1, 0, 2} = 2,
  assume h2 : ∀ n,  mul n.succ 1 = n.succ,
  have h3 : ∃ valid_rearrangements : ℕ, valid_rearrangements = 9 := sorry,
  exact h3

end rearrange_digits_2102_l657_657408


namespace intersection_eq_l657_657027

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_eq : M ∩ N = {5, 7, 9} := sorry

end intersection_eq_l657_657027


namespace sum_of_products_mod_7_l657_657220

-- Define the numbers involved
def a := 1789
def b := 1861
def c := 1945
def d := 1533
def e := 1607
def f := 1688

-- Define the sum of products
def sum_of_products := a * b * c + d * e * f

-- The statement to prove:
theorem sum_of_products_mod_7 : sum_of_products % 7 = 3 := 
by sorry

end sum_of_products_mod_7_l657_657220


namespace geometric_ratio_AH_HG_one_to_one_l657_657879

variables (A B C D E F G H : Type)
variables [parallelogram A B C D]
variables [perpendicular B E D A] [perpendicular D F B C]
variables [equal_area_division A B C D E F]
variables [segment_ext N : Type] [congruent_segments D G B D A B D G]
variables [segment_intersection B E A G H]

theorem geometric_ratio_AH_HG_one_to_one :
  ratio AH HG = 1 :=
sorry

end geometric_ratio_AH_HG_one_to_one_l657_657879


namespace trajectory_of_M_l657_657852

theorem trajectory_of_M (x y : ℝ) (h : (real.sqrt ((x - 4)^2 + y^2) / abs (x - 3) = 2)) :
  3 * x^2 - y^2 - 16 * x + 20 = 0 :=
sorry

end trajectory_of_M_l657_657852


namespace smallest_positive_angle_l657_657225

theorem smallest_positive_angle (x : ℝ) (h: 8 * sin x * (cos x)^6 - 8 * (sin x)^6 * cos x = 2): x = 11.25 :=
sorry

end smallest_positive_angle_l657_657225


namespace group_division_l657_657443

theorem group_division (men women : ℕ) : 
  men = 4 → women = 5 →
  (∃ g1 g2 g3 : set (fin $ men + women), 
    g1.card = 3 ∧ g2.card = 3 ∧ g3.card = 3 ∧ 
    (∀ g, g ∈ [g1, g2, g3] → (∃ m w : ℕ, 1 ≤ m ∧ 1 ≤ w ∧ 
      finset.card (finset.filter (λ x, x < men) g) = m ∧ 
      finset.card (finset.filter (λ x, x ≥ men) g) = w)) 
    ∧ finset.disjoint g1 g2 ∧ finset.disjoint g2 g3 ∧ finset.disjoint g3 g1 
    ∧ g1 ∪ g2 ∪ g3 = finset.univ (fin $ men + women)) → 
  finset.card (finset.powerset' 3 (finset.univ (fin $ men + women))) / 2 = 180 :=
begin
  intros hmen hwomen,
  sorry
end

end group_division_l657_657443


namespace calculate_initial_money_l657_657166

noncomputable def initial_money (remaining_money: ℝ) (spent_percent: ℝ) : ℝ :=
  remaining_money / (1 - spent_percent)

theorem calculate_initial_money :
  initial_money 3500 0.30 = 5000 := 
by
  rw [initial_money]
  sorry

end calculate_initial_money_l657_657166


namespace ellipse_properties_l657_657381

-- Definitions
def isEllipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (a : ℝ)^2 ≠ 0 ∧ (b : ℝ)^2 ≠ 0

def eccentricity (a b : ℝ) : Prop :=
  (1 / 2) = (real.sqrt (a^2 - b^2) / a)

def focus (a : ℝ) : ℝ × ℝ :=
  (- (1 / 2) * a, 0)

def vertex (a b : ℝ) : ℝ × ℝ :=
  (0, (b / a) * real.sqrt(3) * a)

def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

def line_eq (A B : ℝ × ℝ) : ℝ → ℝ :=
  λ x, (B.2 - A.2)/(B.1 - A.1) * (x - A.1) + A.2

def on_x_axis (B : ℝ × ℝ) : Prop :=
  B.2 = 0

def tangency_condition (a x y b: ℝ): Prop :=
  x + real.sqrt 3 * y + 3 = 0 → (real.abs (a / 2 + 3) = a)

-- Problem
theorem ellipse_properties :
  ∃ (a b : ℝ), isEllipse a b ∧ eccentricity a b ∧
  ∃ (A F B : ℝ × ℝ), F = focus a ∧ A = vertex a b ∧ on_x_axis B ∧ perpendicular A F ∧
  tangency_condition a (B.1) (B.2) b ∧ (∃ (k : ℝ), k ≠ 0 → ¬ ∃ M N P Q : ℝ × ℝ,
  P = ((M.1 + N.1)/2, P.2) ∧ Q = ((2 * P.1, 2 * P.2)) ∧ (M.1 + N.1 = - (8 * k^2) / (3 + 4 * k^2)) ∧
  ∇y (\((M, N, Q), \frac{(3M.x_0x_P - \frac{4kx}{ \frac{2* Q}[]) = ((mP))])))): sorry

end ellipse_properties_l657_657381


namespace articles_produced_l657_657793

theorem articles_produced (x y z w : ℕ) :
  (x ≠ 0) → (y ≠ 0) → (z ≠ 0) → (w ≠ 0) →
  ((x * x * x * (1 / x^2) = x) →
  y * z * w * (1 / x^2) = y * z * w / x^2) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end articles_produced_l657_657793


namespace quadratic1_vertex_quadratic1_decreases_when_x_increases_quadratic2_vertex_quadratic2_decreases_when_x_increases_quadratic3_vertex_quadratic3_decreases_when_x_increases_l657_657257

noncomputable def quadratic1 := λ x : ℝ, x^2 + 2*x + 1
noncomputable def quadratic2 := λ x : ℝ, -1/2 * x^2 + 3
noncomputable def quadratic3 := λ x : ℝ, 2 * (x + 1) * (x - 3)

theorem quadratic1_vertex :
  ∃ (h k : ℝ), quadratic1 = λ x, (x + h)^2 + k ∧ (h, k) = (-1, 0) := sorry

theorem quadratic1_decreases_when_x_increases :
  ∀ x : ℝ, x < -1 → quadratic1 (x + 1) < quadratic1 x := sorry

theorem quadratic2_vertex :
  ∃ (h k : ℝ), quadratic2 = λ x, -1/2 * (x - h)^2 + k ∧ (h, k) = (0, 3) := sorry

theorem quadratic2_decreases_when_x_increases :
  ∀ x : ℝ, x > 0 → quadratic2 (x + 1) < quadratic2 x := sorry

theorem quadratic3_vertex :
  ∃ (h k : ℝ), quadratic3 = λ x, 2 * ((x - h) * (x - 1)) ∧ (h, k) = (1, -8) := sorry

theorem quadratic3_decreases_when_x_increases :
  ∀ x : ℝ, x < 1 → quadratic3 (x + 1) < quadratic3 x := sorry

end quadratic1_vertex_quadratic1_decreases_when_x_increases_quadratic2_vertex_quadratic2_decreases_when_x_increases_quadratic3_vertex_quadratic3_decreases_when_x_increases_l657_657257


namespace intersection_nonempty_a_value_l657_657404

theorem intersection_nonempty_a_value (a : ℕ) : 
  ({0, a} ∩ {1, 2} ≠ ∅) → (a = 1 ∨ a = 2) := 
by {
  sorry
}

end intersection_nonempty_a_value_l657_657404


namespace largest_n_for_factoring_l657_657268

theorem largest_n_for_factoring :
  ∃ (n : ℤ), (∀ (A B : ℤ), (3 * A + B = n) → (3 * A * B = 90) → n = 271) :=
by sorry

end largest_n_for_factoring_l657_657268


namespace anne_ben_charlie_difference_l657_657428

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def charlie_discount_rate : ℝ := 0.15

def anne_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)
def ben_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)
def charlie_total : ℝ := (original_price * (1 - charlie_discount_rate)) * (1 + sales_tax_rate)

def anne_minus_ben_minus_charlie : ℝ := anne_total - ben_total - charlie_total

theorem anne_ben_charlie_difference : anne_minus_ben_minus_charlie = -12.96 :=
by
  sorry

end anne_ben_charlie_difference_l657_657428


namespace exists_triangle_l657_657476

-- Definitions of points and properties must be assumed and declared correctly.
variables (A B C : Type) [HasScalar ℝ A] [HasAdd A] [HasSub A] [HasDiv A] [HasZero A]

-- Define the midpoints
variables {P Q R : A}
-- Midpoints of the sides
def is_midpoint (P : A) (X Y : A) : Prop := P = (X + Y) / 2

-- Defining the structure
structure Triangle (A B C P Q R U V W : A) :=
  (midP : is_midpoint P B C)
  (midQ : is_midpoint Q C A)
  (midR : is_midpoint R A B)
  (midU : is_midpoint U Q R)
  (midV : is_midpoint V R P)
  (midW : is_midpoint W P Q)

-- Definition of vectors AU, BV, CW in terms of sides
noncomputable def AU (A U : A) : A := U - A
noncomputable def BV (B V : A) : A := V - B
noncomputable def CW (C W : A) : A := W - C

-- Theorem stating the existence of a triangle
theorem exists_triangle (A B C P Q R U V W : A) [h : Triangle A B C P Q R U V W] :
  ∃ x y z : A, x = AU A U ∧ y = BV B V ∧ z = CW C W ∧ x + y + z = 0 :=
by {
  -- This is indicating the requirement but not providing the proof inline.

  sorry
}

end exists_triangle_l657_657476


namespace grouping_count_l657_657450

theorem grouping_count (men women : ℕ) 
  (h_men : men = 4) (h_women : women = 5)
  (at_least_one_man_woman : ∀ (g1 g2 g3 : Finset (Fin 9)), 
    g1.card = 3 → g2.card = 3 → g3.card = 3 → g1 ∩ g2 = ∅ → g2 ∩ g3 = ∅ → g3 ∩ g1 = ∅ → 
    (g1 ∩ univ.filter (· < 4)).nonempty ∧ (g1 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g2 ∩ univ.filter (· < 4)).nonempty ∧ (g2 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g3 ∩ univ.filter (· < 4)).nonempty ∧ (g3 ∩ univ.filter (· ≥ 4)).nonempty) :
  (choose 4 1 * choose 5 2 * choose 3 1 * choose 3 2) / 2! = 180 :=
sorry

end grouping_count_l657_657450


namespace inequality_abc_l657_657528

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := 
by 
  sorry

end inequality_abc_l657_657528


namespace one_third_greater_than_333_l657_657675

theorem one_third_greater_than_333 :
  (1 : ℝ) / 3 > (333 : ℝ) / 1000 - 1 / 3000 :=
sorry

end one_third_greater_than_333_l657_657675


namespace sequence_unique_l657_657701

theorem sequence_unique (a : ℕ → ℕ) (h1 : ∀ n, a n < a (n + 1))
  (h2 : ∀ (i j k : ℕ), a i + a j ≠ a k)
  (h3 : ∀ k, a k = 2 * k - 1 → infinite {n | a n = 2 * n - 1}) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_unique_l657_657701


namespace magnitude_of_w_is_one_l657_657475

def z : ℂ := ((-7 + 8 * complex.I) ^ 4 * (18 - 6 * complex.I) ^ 5) / (2 + 5 * complex.I)
def w : ℂ := (conj z) / z

theorem magnitude_of_w_is_one : |w| = 1 := 
by{
  -- proof will go here.
  sorry
}

end magnitude_of_w_is_one_l657_657475


namespace circumscribed_circles_intersect_l657_657743

noncomputable def circumcircle (a b c : Point) : Set Point := sorry

noncomputable def intersect_at_single_point (circles : List (Set Point)) : Option Point := sorry

variables {A1 A2 A3 B1 B2 B3 : Point}

theorem circumscribed_circles_intersect
  (h1 : ∃ P, ∀ circle ∈ [
    circumcircle A1 A2 B3, 
    circumcircle A1 B2 A3, 
    circumcircle B1 A2 A3
  ], P ∈ circle) :
  ∃ Q, ∀ circle ∈ [
    circumcircle B1 B2 A3, 
    circumcircle B1 A2 B3, 
    circumcircle A1 B2 B3
  ], Q ∈ circle :=
sorry

end circumscribed_circles_intersect_l657_657743


namespace collinear_M_N_P_l657_657983

-- Define the inscribed quadrilateral
variables {A B C D P O : Point}
variables {Γ Δ : Circle}
variables {M N : Point}

-- Conditions as given in the problem
def isInscribedQuadrilateral (A B C D : Point) : Prop := 
  ∃ O : Point, ∃ Γ : Circle, ∃ Δ : Circle,
    Γ.is_circumcircle_of A B O ∧
    Δ.is_circumcircle_of C D O ∧
    M.is_midpoint_of_arc A B Γ O ∧
    N.is_midpoint_of_arc C D Δ O ∧
    P = intersection_of_diagonals A B C D

-- Theorem statement
theorem collinear_M_N_P :
  isInscribedQuadrilateral A B C D →
  M N P are_collinear := 
sorry

end collinear_M_N_P_l657_657983


namespace line_circle_intersection_l657_657751

theorem line_circle_intersection {a : ℝ} :
  (∃ x1 y1 x2 y2 : ℝ, (x1 + y1 = a) ∧ (x2 + y2 = a) ∧ (x1^2 + y1^2 = 4) ∧ (x2^2 + y2^2 = 4) ∧
  (real.sqrt ((x1 + x2)^2 + (y1 + y2)^2) = real.sqrt ((x1 - x2)^2 + (y1 - y2)^2))) →
  (a = 2 ∨ a = -2) :=
by
  sorry

end line_circle_intersection_l657_657751


namespace minimum_value_of_u_l657_657706

noncomputable def u (x y : ℝ) : ℝ := x^2 + (81 / x^2) - 2 * x * y + (18 / x) * real.sqrt (2 - y^2)

theorem minimum_value_of_u :
  ∃ (x y : ℝ), (∀ (a b : ℝ), u a b ≥ u x y) ∧ u x y = 6 :=
sorry

end minimum_value_of_u_l657_657706


namespace numeral_in_150th_decimal_place_l657_657593

noncomputable def decimal_representation_13_17 : String :=
  "7647058823529411"

theorem numeral_in_150th_decimal_place :
  (decimal_representation_13_17.get (150 % 17)).iget = '1' :=
by
  sorry

end numeral_in_150th_decimal_place_l657_657593


namespace minimum_sum_x_maximum_sum_x_l657_657488

-- Definitions based on conditions
variables (n : ℕ) (x : ℕ → ℝ)
def non_negative (i : ℕ) : Prop := i ≥ 1 ∧ i ≤ n → 0 ≤ x i
def constraint : Prop := 
  ∑ i in Finset.range n, (x i)^2 + 2 * ∑ (k j : ℕ) in Finset.range n.off_diag, if k < j then (sqrt (k / j)) * (x k) * (x j) else 0 = 1

-- Minimum value theorem
theorem minimum_sum_x (n : ℕ) (x : ℕ → ℝ) (h1 : ∀ i, non_negative n x i) (h2 : constraint n x) :
  1 ≤ ∑ i in Finset.range n, x i :=
sorry

-- Maximum value theorem
theorem maximum_sum_x (n : ℕ) (x : ℕ → ℝ) (h1 : ∀ i, non_negative n x i) (h2 : constraint n x) :
  ∑ i in Finset.range n, x i ≤ (∑ k in Finset.range n, (sqrt k - sqrt (k - 1))^2)^0.5 :=
sorry

end minimum_sum_x_maximum_sum_x_l657_657488


namespace inequality_solution_set_l657_657753

theorem inequality_solution_set (a b : ℝ) (h : ∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - 5 * x + b > 0) :
  ∀ x : ℝ, x < -1/3 ∨ x > 1/2 ↔ b * x^2 - 5 * x + a > 0 :=
sorry

end inequality_solution_set_l657_657753


namespace negative_values_of_x_l657_657300

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l657_657300


namespace determinant_expression_l657_657485

open Matrix

variables {R : Type*} [CommRing R] (a b c p q : R)

-- Given conditions
def cubic_polynomial (x : R) := x^3 + p * x + q

def roots_condition : Prop := 
  ∀ (x : R), cubic_polynomial p q x = 0 ↔ (x = a ∨ x = b ∨ x = c)

-- Define matrix and its determinant
def det_matrix : Matrix (Fin 3) (Fin 3) R :=
  !![2 + a^2, 1, 1;
     1, 2 + b^2, 1;
     1, 1, 2 + c^2]

def det_val : R := det_matrix.det

theorem determinant_expression (h : roots_condition a b c p q) :
  det_val a b c = 2 * p^2 - 4 * q + q^2 :=
sorry

end determinant_expression_l657_657485


namespace triangle_inequality_l657_657357

theorem triangle_inequality
  (A B C : ℝ)
  (x y z : ℝ)
  (cos : ℝ → ℝ)
  (cos_A : cos A = cos B * cos C - ℕ.sin B * ℕ.sin C)
  (cos_B : cos B = cos C * cos A - ℕ.sin C * ℕ.sin A)
  (cos_C : cos C = cos A * cos B - ℕ.sin A * ℕ.sin B) :
  x^2 + y^2 + z^2 ≥ 2 * x * y * cos C + 2 * y * z * cos A + 2 * z * x * cos B := 
by
  sorry

end triangle_inequality_l657_657357


namespace smallest_N_divisors_of_8_l657_657386

theorem smallest_N_divisors_of_8 (N : ℕ) (h0 : N % 10 = 0) (h8 : ∃ (divisors : ℕ), divisors = 8 ∧ (∀ k, k ∣ N → k ≤ divisors)) : N = 30 := 
sorry

end smallest_N_divisors_of_8_l657_657386


namespace ratio_of_MBQ_ABQ_l657_657814

-- Definitions of angles and their properties

variables {x y : ℝ}
variable {angle_ABQ : ℝ}
variable {angle_MBQ : ℝ}
variable {angle_MBP : ℝ}
variable {angle_BPQ : ℝ}

-- Conditions given in the problem
def BP_bisects_ANGLE_ABQ (angle_ABQ : ℝ) (x : ℝ) : Prop :=
  angle_ABQ = 2 * x ∧ x > 0

def BM_bisects_ANGLE_PBQ (angle_PBQ : ℝ) (y : ℝ) : Prop :=
  y = angle_PBQ ∧ angle_MBP = angle_PBQ / 2 ∧ angle_MBQ = angle_PBQ / 2

-- Problem statement to prove
theorem ratio_of_MBQ_ABQ {angle_ABQ : ℝ} {angle_MBQ : ℝ} (h1 : BP_bisects_ANGLE_ABQ angle_ABQ x) (h2 : BM_bisects_ANGLE_PBQ angle_BPQ y) (h3 : angle_BPQ = y) :
  angle_MBQ / angle_ABQ = 1 / 4 :=
by
  -- proof steps to be filled in
  sorry

end ratio_of_MBQ_ABQ_l657_657814


namespace find_a_l657_657117

-- Define the prime number theorem, essential for the proof
theorem find_a (a b p : ℤ) (hp : p.prime)
  (h1 : p^2 + a * p + b = 0)
  (h2 : p^2 + b * p + 1100 = 0) :
  a = 274 ∨ a = 40 :=
by
  sorry

end find_a_l657_657117


namespace coloring_satisfies_conditions_l657_657168

def is_lattice_point (p : ℤ × ℤ) : Prop :=
  true -- by definition in the problem, every pair of integers is a lattice point

inductive Color
| white 
| red 
| black

def color (p : ℤ × ℤ) : Color :=
  if p.1 = 0 then 
    if p.2 % 2 = 0 then Color.black else Color.white 
  else 
    Color.red

theorem coloring_satisfies_conditions :
  (∀ y : ℤ, ∃ x : ℤ, color (x, y) = Color.red) ∧
  ∀ A B C : ℤ × ℤ, color A = Color.white → color B = Color.red → color C = Color.black →
    let D := (A.1 + C.1 - B.1, A.2 + C.2 - B.2) in
    is_lattice_point D ∧ color D = Color.red :=
by
  sorry

end coloring_satisfies_conditions_l657_657168


namespace largest_multiple_of_7_less_than_neg50_l657_657147

theorem largest_multiple_of_7_less_than_neg50 : ∃ x, (∃ k : ℤ, x = 7 * k) ∧ x < -50 ∧ ∀ y, (∃ m : ℤ, y = 7 * m) → y < -50 → y ≤ x :=
sorry

end largest_multiple_of_7_less_than_neg50_l657_657147


namespace minimum_rows_required_l657_657623

theorem minimum_rows_required (n : ℕ) (C : ℕ → ℕ) 
  (h1 : ∀ i, i < n → 0 ≤ C i ∧ C i ≤ 39) 
  (h2 : (Finset.range n).sum C = 1990) : 
  ∃ r : ℕ, r = 12 ∧ ∀ (assignment : ℕ → ℕ), 
  (∀ i, i < n → assignment i < r) ∧ 
  (∀ i j, i < n → j < n → (assignment i = assignment j → i = j)) ∧
  (∀ i, i < n → ∑ j, j < n ∧ assignment j = i → C j ≤ 199) := 
begin
  sorry
end

end minimum_rows_required_l657_657623


namespace additional_spheres_fitting_l657_657205

-- Definitions of the geometric entities
structure Sphere := (radius : ℝ)
structure Cone := (height : ℝ)

-- Properties of spheres and the cone
def sphere_O1 := Sphere.mk 2 -- Sphere O1 with radius 2
def sphere_O2 := Sphere.mk 3 -- Sphere O2 with radius 3

-- Truncated cone with a height of 8
def truncated_cone := Cone.mk 8 

-- Hypotheses from the problem conditions
axiom O1_center_on_axis (c : Cone) (O1 : Sphere) : O1.radius = 2 → c.height = 8 → true
axiom O1_tangent_to_upper_base_and_side (c : Cone) (O1 : Sphere) : O1.radius = 2 → c.height = 8 → true
axiom O2_tangent_to_O1_base_side (c : Cone) (O2 : Sphere) (O1 : Sphere) : O1.radius = 2 → O2.radius = 3 → c.height = 8 → true

-- The statement to prove
theorem additional_spheres_fitting (c : Cone) (O1 O2 : Sphere) 
  (h₁ : O1_center_on_axis c O1)
  (h₂ : O1_tangent_to_upper_base_and_side c O1)
  (h₃ : O2_tangent_to_O1_base_side c O2 O1) : 
  O1.radius = 2 → O2.radius = 3 → c.height = 8 →
  ∃ (n : ℕ), n = 2 :=
by  sorry

end additional_spheres_fitting_l657_657205


namespace net_profit_calc_l657_657167

theorem net_profit_calc:
  ∃ (x y : ℕ), x + y = 25 ∧ 1700 * x + 1800 * y = 44000 ∧ 2400 * x + 2600 * y = 63000 := by
  sorry

end net_profit_calc_l657_657167


namespace cone_base_circumference_l657_657622

-- Define the given conditions
def radius : ℝ := 6
def total_circumference : ℝ := 2 * Real.pi * radius
def sector_angle_degrees : ℕ := 180
def fraction_of_circle : ℝ := sector_angle_degrees / 360

-- Theorem stating the desired result
theorem cone_base_circumference :
  fraction_of_circle * total_circumference = 6 * Real.pi :=
sorry

end cone_base_circumference_l657_657622


namespace intersection_M_N_l657_657046

def M := {1, 3, 5, 7, 9}

def N := {x : ℤ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} := by
  sorry

end intersection_M_N_l657_657046


namespace faster_speed_additional_distance_l657_657419

noncomputable def additional_distance_walked_at_faster_speed
  (distance_actual : ℝ)
  (speed_slower : ℝ)
  (speed_faster : ℝ) : ℝ :=
  let time := distance_actual / speed_slower in
  (speed_faster * time) - distance_actual

theorem faster_speed_additional_distance
  (distance_actual : ℝ := 33.333333333333336)
  (speed_slower : ℝ := 10)
  (speed_faster : ℝ := 16) :
  additional_distance_walked_at_faster_speed distance_actual speed_slower speed_faster = 20 :=
  by
  sorry

end faster_speed_additional_distance_l657_657419


namespace min_fence_posts_needed_l657_657637

-- Definitions for the problem conditions
def area_length : ℕ := 72
def regular_side : ℕ := 30
def sloped_side : ℕ := 33
def interval : ℕ := 15

-- The property we want to prove
theorem min_fence_posts_needed : 3 * ((sloped_side + interval - 1) / interval) + 3 * ((regular_side + interval - 1) / interval) = 6 := 
by
  sorry

end min_fence_posts_needed_l657_657637


namespace find_polynomial_d_l657_657702

theorem find_polynomial_d (a : ℤ) (b : ℤ)
  (ha : (b = (a^2 - 1) / 4 ∨ ∃ k : ℤ, k ∣ 2 ∧ b = a^2 / 4 + k))
  (p q : ℤ[X]) (hq : q ≠ 0) :
    (p^2 - (X^2 + a * X + b) * q^2 = 1) :=
sorry

end find_polynomial_d_l657_657702


namespace smallest_number_increased_by_2_divisible_l657_657553

theorem smallest_number_increased_by_2_divisible :
  ∃ N, let LCM := Nat.lcm (Nat.lcm (Nat.lcm 12 30) 
                             (Nat.lcm (Nat.lcm 48 74) 
                                      (Nat.lcm (Nat.lcm 100 113) 
                                               (Nat.lcm 127 139)))) in 
  N + 2 = LCM :=
sorry

end smallest_number_increased_by_2_divisible_l657_657553


namespace triangle_side_length_l657_657104

theorem triangle_side_length (P Q R : Type) (cos_Q : ℝ) (PQ QR : ℝ) 
  (sin_Q : ℝ) (h_cos_Q : cos_Q = 0.6) (h_PQ : PQ = 10) (h_sin_Q : sin_Q = 0.8) : 
  QR = 50 / 3 :=
by
  sorry

end triangle_side_length_l657_657104


namespace correct_propositions_l657_657206

-- Proposition 1: Two planes parallel to the same plane are parallel
def prop1 (P Q R : Plane) (hPQ : parallel P Q) (hPR : parallel P R) : parallel Q R :=
sorry

-- Proposition 2: Two planes parallel to the same line are parallel
def prop2 (P Q : Plane) (l : Line) (hPl : parallel_plane_line P l) (hQl : parallel_plane_line Q l) : parallel P Q :=
sorry

-- Proposition 3: Two planes perpendicular to the same plane are parallel
def prop3 (P Q R : Plane) (hPR : perpendicular P R) (hQR : perpendicular Q R) : parallel P Q :=
sorry

-- Proposition 4: Two lines perpendicular to the same plane are parallel
def prop4 (l m : Line) (P : Plane) (hlP : perpendicular_line_plane l P) (hmP : perpendicular_line_plane m P) : parallel_line l m :=
sorry

-- The correct propositions are 1 and 4
theorem correct_propositions : (prop1 ∧ prop4) :=
sorry

end correct_propositions_l657_657206


namespace find_third_month_sale_l657_657181

theorem find_third_month_sale
  (sale_1 sale_2 sale_3 sale_4 sale_5 sale_6 : ℕ)
  (h1 : sale_1 = 800)
  (h2 : sale_2 = 900)
  (h4 : sale_4 = 700)
  (h5 : sale_5 = 800)
  (h6 : sale_6 = 900)
  (h_avg : (sale_1 + sale_2 + sale_3 + sale_4 + sale_5 + sale_6) / 6 = 850) : 
  sale_3 = 1000 :=
by
  sorry

end find_third_month_sale_l657_657181


namespace weight_of_b_l657_657164

theorem weight_of_b (a b c : ℝ) (h1 : a + b + c = 126) (h2 : a + b = 80) (h3 : b + c = 86) : b = 40 :=
sorry

end weight_of_b_l657_657164


namespace concurrency_of_inscribed_square_centroids_l657_657840

open EuclideanGeometry

-- Define the points A, B, C, and the centers of the squares A1, B1, C1
variables {A B C A1 B1 C1 : Point}
-- Assume A1, B1, C1 are centers of specific squares inscribed in the triangle ABC
variables (hA1 : IsInscribedSquare A B C A1 BC AB AC)
variables (hB1 : IsInscribedSquare B A C B1 AC AB BC)
variables (hC1 : IsInscribedSquare C A B C1 AB BC AC)

-- Define the lines AA1, BB1, and CC1
def line_AA1 := Line.through A A1
def line_BB1 := Line.through B B1
def line_CC1 := Line.through C C1

-- Prove that the lines AA1, BB1, and CC1 intersect at one point
theorem concurrency_of_inscribed_square_centroids :
  Concurrent line_AA1 line_BB1 line_CC1 :=
by
  -- proof to be provided
  sorry

end concurrency_of_inscribed_square_centroids_l657_657840


namespace possible_values_of_A_l657_657989

theorem possible_values_of_A 
  (A B C D E : ℤ) 
  (h_rel_prime : Int.gcd A (Int.gcd B (Int.gcd C (Int.gcd D E))) = 1) 
  (h_poly1 : ∃ p : Fin 4 → ℤ, 5 * A * x^4 + 4 * B * x^3 + 3 * C * x^2 + 2 * D * x + E = ∏ i, (x - p i))
  (h_poly2 : ∃ q : Fin 3 → ℤ, 10 * A * x^3 + 6 * B * x^2 + 3 * C * x + D = ∏ i, (x - q i)) :
  A = 1 ∨ A = -1 ∨ A = 3 ∨ A = -3 := 
sorry

end possible_values_of_A_l657_657989


namespace profit_per_unit_max_profit_l657_657612

-- Part 1: Proving profits per unit
theorem profit_per_unit (a b : ℝ) (h1 : a + b = 600) (h2 : 3 * a + 2 * b = 1400) :
  a = 200 ∧ b = 400 :=
by
  sorry

-- Part 2: Proving maximum profit calculation
theorem max_profit (a b x y : ℝ) (h1 : x + y = 20) (h2 : y ≤ (2 / 3) * x) (h3 : a = 200) (h4 : b = 400) :
  let w := -200 * x + 8000 in
  (y = 20 - x ∧ w = 5600) :=
by
  sorry

end profit_per_unit_max_profit_l657_657612


namespace smallest_value_w_i_l657_657846

theorem smallest_value_w_i (w : ℂ) (h : |w^2 + 1| = |w * (w + complex.I)|) :
  ∃ (z : ℂ), |w - complex.I| ≥ 1 ∧ (∀ y, |y - complex.I| < 1 → |y^2 + 1| ≠ |y * (y + complex.I)|) :=
sorry

end smallest_value_w_i_l657_657846


namespace count_negative_x_with_sqrt_pos_int_l657_657349

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l657_657349


namespace seating_arrangements_60_l657_657432

theorem seating_arrangements_60 : 
  ∀ (seats : ℕ) (people : ℕ) (arrangements : ℕ),
  seats = 9 → 
  people = 4 → 
  (∀ person : ℕ, person < people → seats ≥ person + 3) → 
  arrangements = 60 := 
begin 
  sorry 
end

end seating_arrangements_60_l657_657432


namespace canal_width_at_top_l657_657914

theorem canal_width_at_top (bottom_width : ℕ) (area : ℕ) (depth : ℕ) : 
  bottom_width = 8 →
  area = 840 →
  depth = 84 →
  let top_width := (2 * area) / (depth / (bottom_width + 8)) - bottom_width
  in top_width = 12 :=
by
  intros h1 h2 h3
  let top_width := (2 * area) / (depth / (bottom_width + 8)) - bottom_width
  sorry

end canal_width_at_top_l657_657914


namespace value_of_a_b_c_l657_657111

theorem value_of_a_b_c 
  (a b c : ℤ) 
  (h1 : x^2 + 12*x + 35 = (x + a)*(x + b)) 
  (h2 : x^2 - 15*x + 56 = (x - b)*(x - c)) : 
  a + b + c = 20 := 
sorry

end value_of_a_b_c_l657_657111


namespace non_congruent_triangles_with_perimeter_21_l657_657411

theorem non_congruent_triangles_with_perimeter_21 :
  {s : Set (ℤ × ℤ × ℤ) |
    ∃ (a b c : ℤ), a + b + c = 21 ∧
    a + b > c ∧ a + c > b ∧ b + c > a ∧
    a ≤ b ∧ b ≤ c ∧
    s = {⟨a, b, c⟩ }
  }.card = 6 := by
  sorry

end non_congruent_triangles_with_perimeter_21_l657_657411


namespace numeral_150th_decimal_place_l657_657583

theorem numeral_150th_decimal_place (k : ℕ) (h : k = 150) : 
  (decimal_place (13 / 17) k) = 5 :=
sorry

end numeral_150th_decimal_place_l657_657583


namespace slope_angle_of_line_l657_657691

theorem slope_angle_of_line (m : ℝ) (α : ℝ) (h1 : m = -1) (h2 : tan α = m) : α = 3 * Real.pi / 4 :=
sorry

end slope_angle_of_line_l657_657691


namespace find_triangle_side_lengths_l657_657262

theorem find_triangle_side_lengths (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) 
    (h4 : a < 6.25 * 2) (h5 : b < 6.25 * 2) (h6 : c < 6.25 * 2) 
    (h7 : (2 * 3.125)^2 = (a * b * c) / (3.125 * 3.125)): 
    (a = 5 ∧ b = 5 ∧ c = 6) ∨ (a = 5 ∧ b = 6 ∧ c = 5) ∨ (a = 6 ∧ b = 5 ∧ c = 5) := 
sorry

end find_triangle_side_lengths_l657_657262


namespace quadratic_condition_l657_657916

theorem quadratic_condition (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 + 4 * x - 3 = 0) → a ≠ 1 :=
by
  sorry

end quadratic_condition_l657_657916


namespace intersection_correct_l657_657022

-- Define the sets M and N
def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x | 2 * x > 7}

-- Define the expected intersection result
def expected_intersection : Set ℝ := {5, 7, 9}

-- State the theorem
theorem intersection_correct : ∀ x, x ∈ M ∩ N ↔ x ∈ expected_intersection :=
by
  sorry

end intersection_correct_l657_657022


namespace minimal_kn_inequality_l657_657015

noncomputable def kn (n : ℕ) : ℝ :=
  if n ≥ 3 then (2 * n - 1) / (n - 1) ^ 2 else 0

theorem minimal_kn_inequality (n : ℕ) (x : ℕ → ℝ) (h : n ≥ 3) (hx_prod : (∏ i in Finset.range n, x i) = 1) :
  (∑ i in Finset.range n, 1 / Real.sqrt (1 + kn n * x i)) ≤ n - 1 :=
sorry

end minimal_kn_inequality_l657_657015


namespace cost_of_child_ticket_l657_657652

-- Define the conditions
def adult_ticket_cost : ℕ := 60
def total_people : ℕ := 280
def total_collected_dollars : ℕ := 140
def total_collected_cents : ℕ := total_collected_dollars * 100
def children_attended : ℕ := 80
def adults_attended : ℕ := total_people - children_attended
def total_collected_from_adults : ℕ := adults_attended * adult_ticket_cost

-- State the theorem to prove the cost of a child ticket
theorem cost_of_child_ticket (x : ℕ) :
  total_collected_from_adults + children_attended * x = total_collected_cents →
  x = 25 :=
by
  sorry

end cost_of_child_ticket_l657_657652


namespace bullet_trains_crossing_time_l657_657165

theorem bullet_trains_crossing_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (speed1 speed2 : ℝ)
  (relative_speed : ℝ)
  (total_distance : ℝ)
  (cross_time : ℝ)
  (h_length : length = 120)
  (h_time1 : time1 = 10)
  (h_time2 : time2 = 20)
  (h_speed1 : speed1 = length / time1)
  (h_speed2 : speed2 = length / time2)
  (h_relative_speed : relative_speed = speed1 + speed2)
  (h_total_distance : total_distance = length + length)
  (h_cross_time : cross_time = total_distance / relative_speed) :
  cross_time = 240 / 18 := 
by
  sorry

end bullet_trains_crossing_time_l657_657165


namespace consecutive_integers_solution_l657_657132

theorem consecutive_integers_solution :
  ∃ (n : ℕ), n > 0 ∧ n * (n + 1) + 91 = n^2 + (n + 1)^2 ∧ n + 1 = 10 :=
by
  sorry

end consecutive_integers_solution_l657_657132


namespace range_of_m_l657_657764

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + m + 8 ≥ 0) ↔ (-8 / 9 ≤ m ∧ m ≤ 1) :=
sorry

end range_of_m_l657_657764


namespace probability_diff_colors_l657_657806

theorem probability_diff_colors (balls : Finset ℕ) :
  balls = {1, 1, 1, 0} → (1/2) := 
sorry

end probability_diff_colors_l657_657806


namespace intersection_M_N_l657_657035

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := { x | 2 * x > 7 }

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657035


namespace general_formula_arithmetic_sequence_sum_of_sequence_l657_657503

noncomputable def arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) :=
∀ n, a_n (n + 1) = a_n n + d

theorem general_formula_arithmetic_sequence
    {a_n : ℕ → ℝ} {d : ℝ}
    (h1 : arithmetic_sequence a_n d)
    (d_pos : 0 < d)
    (init_cond : a_n 1 = 2)
    (third_cond : a_n 3 = a_n 2 ^ 2 - 10) :
    a_n n = sqrt 10 * n + 2 - sqrt 10 := 
sorry

noncomputable def geometric_sequence (b_n : ℕ → ℝ) (r : ℝ) :=
∀ n, b_n (n + 1) = b_n n * r

noncomputable def sum_of_series (s_n : ℕ → ℝ) :=
∀ n, s_n n = ∑ i in finset.range n, s_n i

theorem sum_of_sequence
    {a_n b_n : ℕ → ℝ} {d r : ℝ}
    (h1 : arithmetic_sequence a_n d)
    (h2 : geometric_sequence b_n r)
    (d_pos : 0 < d)
    (r_ratio : r = 2)
    (init_cond : a_n 1 = 2)
    (third_cond : a_n 3 = a_n 2 ^ 2 - 10)
    (init_b : b_n 1 = 1) :
    sum_of_series (λ n, a_n n + b_n n) n =
    2 * n + sqrt 10 / 2 * n * (n - 1) + 2 ^ n - 1 := 
sorry

end general_formula_arithmetic_sequence_sum_of_sequence_l657_657503


namespace forgotten_angle_l657_657670

theorem forgotten_angle {n : ℕ} (h₁ : 2070 = (n - 2) * 180 - angle) : angle = 90 :=
by
  sorry

end forgotten_angle_l657_657670


namespace three_different_suits_probability_l657_657867

def probability_three_different_suits := (39 / 51) * (35 / 50) = 91 / 170

theorem three_different_suits_probability (deck : Finset (Fin 52)) (h : deck.card = 52) :
  probability_three_different_suits :=
sorry

end three_different_suits_probability_l657_657867


namespace cubic_polynomial_no_match_roots_coeffs_l657_657407

theorem cubic_polynomial_no_match_roots_coeffs :
  ∀ (a b c d : ℝ), 
    a ≠ 0 → 
    {a, b, c, d} = {r, s, t} →
    ∀ (r s t : ℝ), 
      r + s + t = -b/a ∧ 
      rs + rt + st = c/a ∧ 
      rst = -d/a → 
      false :=
by 
  intros a b c d ha hset r s t hroots
  sorry

end cubic_polynomial_no_match_roots_coeffs_l657_657407


namespace negative_values_of_x_l657_657302

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l657_657302


namespace intersection_correct_l657_657020

-- Define the sets M and N
def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x | 2 * x > 7}

-- Define the expected intersection result
def expected_intersection : Set ℝ := {5, 7, 9}

-- State the theorem
theorem intersection_correct : ∀ x, x ∈ M ∩ N ↔ x ∈ expected_intersection :=
by
  sorry

end intersection_correct_l657_657020


namespace P_Q_R_S_concyclic_l657_657372

variables {A B C P Q R S : Type*} [Inhabited A] [LinearOrder B] [LinearOrder C] [LinearOrder P]

-- Definitions and conditions
def is_triangle (A B C : Type*) : Prop := sorry

def on_segment (X Y : Type*) (Z : Type*) : Prop := sorry

def between (X Y Z : Type*) : Prop := sorry

def angle_eq (X Y Z X' Y' Z' : Type*) : Prop := sorry

def concyclic (P Q R S : Type*) : Prop := sorry

-- Given conditions
axiom triangle_ABC : is_triangle A B C
axiom P_on_AB : on_segment A B P
axiom Q_on_AC : on_segment A C Q
axiom AP_eq_AQ : AP = AQ
axiom S_R_on_BC : on_segment B C S ∧ on_segment B C R ∧ S ≠ R
axiom S_between_B_R : between B S R
axiom angle_BPS_eq_PRS : angle_eq B P S P R S
axiom angle_CQR_eq_QSR : angle_eq C Q R Q S R

-- Prove statement
theorem P_Q_R_S_concyclic : concyclic P Q R S := sorry

end P_Q_R_S_concyclic_l657_657372


namespace intersection_M_N_l657_657043

def M := {1, 3, 5, 7, 9}

def N := {x : ℤ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} := by
  sorry

end intersection_M_N_l657_657043


namespace number_of_idempotent_homomorphisms_l657_657678

def A : Set ℂ := {z | ∃ k : ℕ, k > 0 ∧ z ^ (2006^k) = 1}

theorem number_of_idempotent_homomorphisms :
  (∃ f : A → A, (∀ x ∈ A, f (f x) = f x) ∧ ∃ y : Finset (A → A), y.card = 8) :=
sorry

end number_of_idempotent_homomorphisms_l657_657678


namespace solve_sqrt_fraction_eq_l657_657895

theorem solve_sqrt_fraction_eq (x : ℝ) (h : √(5 * x - 4) + 15 / √(5 * x - 4) = 8) :
  x = 29 / 5 ∨ x = 13 / 5 :=
sorry

end solve_sqrt_fraction_eq_l657_657895


namespace train_length_correct_l657_657201

noncomputable def train_length (speed_kmph : ℕ) (time_sec : ℕ) (platform_length_m : ℝ): ℝ :=
  let speed_mps := (speed_kmph : ℝ) * 1000 / 3600
  let total_distance := speed_mps * (time_sec : ℝ)
  total_distance - platform_length_m

theorem train_length_correct :
  train_length 108 25 300.06 = 449.94 :=
by
  let speed_mps := (108 : ℝ) * 1000 / 3600
  let total_distance := speed_mps * 25
  have h1 : speed_mps = 30 := by norm_num
  have h2 : total_distance = 750 := by { rw h1, norm_num }
  show train_length 108 25 300.06 = 449.94
  simp [train_length, h1, h2]
  norm_num

end train_length_correct_l657_657201


namespace sum_theta_even_indices_l657_657944

theorem sum_theta_even_indices (n : ℕ) (h₀ : ∀ (z : ℂ), z^30 - z^10 - 1 = 0 → |z| = 1)
    (h₁ : ∀ m : ℕ, (0 ≤ θ_m ∧ θ_m < 360) ∧ (z_m = complex.cos θ_m + complex.sin θ_m * complex.I))
    (h₂ : ∀ i j, i < j → θ_i < θ_j) :
    ∑ i in (finset.range (2 * n)).filter (λ x, x % 2 = 1), θ (x + 1) = 2280 := 
sorry

end sum_theta_even_indices_l657_657944


namespace sampling_methods_l657_657808

def Population := { high_income: ℕ, middle_income: ℕ, low_income: ℕ }
def SurveyCondition1 := Population
def SurveyCondition2 := ℕ

-- Hypotheses based on conditions provided
variables (pop : Population) (students : SurveyCondition2)
(h1 : pop.high_income = 430)
(h2 : pop.middle_income = 980)
(h3 : pop.low_income = 290)
(h4 : students = 12)
(h5 : ∀ n : ℕ, n = 170)

def appropriate_sampling_method_1 : (SurveyCondition1 → Prop) :=
  λ pop, pop.high_income ≠ pop.middle_income ∧ 
          pop.middle_income ≠ pop.low_income ∧ 
          pop.high_income ≠ pop.low_income → stratified_sampling

def appropriate_sampling_method_2 : (SurveyCondition2 → Prop) :=
  λ students, students = 12 → simple_random_sampling

theorem sampling_methods :
  appropriate_sampling_method_1 pop ∧ appropriate_sampling_method_2 students :=
by
  sorry

end sampling_methods_l657_657808


namespace polynomial_inequality_l657_657502

noncomputable def f (x : ℝ) (a : ℕ → ℝ) (n : ℕ) :=
  ∑ i in finset.range (n + 1), a i * x^(n - i)

theorem polynomial_inequality {n : ℕ} (a : ℕ → ℝ) (b : ℕ → ℝ) (x : ℝ) (h : ∀ i, b i ∈ set.Icc (-1 : ℝ) (1 : ℝ)) :
  n ≥ 2 → (set.Icc (-1 : ℝ) (1 : ℝ)).nonempty → f x a n = 0 → x ≥ finset.sup (finset.range n) b → 
  f (x + 1) a n ≥ 2 * n^2 / (∑ i in finset.range n, 1 / (x - b i)) :=
begin
  sorry
end

end polynomial_inequality_l657_657502


namespace grouping_count_l657_657452

theorem grouping_count (men women : ℕ) 
  (h_men : men = 4) (h_women : women = 5)
  (at_least_one_man_woman : ∀ (g1 g2 g3 : Finset (Fin 9)), 
    g1.card = 3 → g2.card = 3 → g3.card = 3 → g1 ∩ g2 = ∅ → g2 ∩ g3 = ∅ → g3 ∩ g1 = ∅ → 
    (g1 ∩ univ.filter (· < 4)).nonempty ∧ (g1 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g2 ∩ univ.filter (· < 4)).nonempty ∧ (g2 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g3 ∩ univ.filter (· < 4)).nonempty ∧ (g3 ∩ univ.filter (· ≥ 4)).nonempty) :
  (choose 4 1 * choose 5 2 * choose 3 1 * choose 3 2) / 2! = 180 :=
sorry

end grouping_count_l657_657452


namespace solve_abs_eq_max_arith_seq_terms_l657_657995

-- Part 1: Number of solutions
theorem solve_abs_eq (a : ℝ) : 
  (a < 2 → ∀ x : ℝ, ¬ (|x+1| + |x+2| + |x+3| = a)) ∧
  (a = 2 → ∃! x : ℝ, |x+1| + |x+2| + |x+3| = a) ∧
  (a > 2 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |x₁+1| + |x₁+2| + |x₁+3| = a ∧ |x₂+1| + |x₂+2| + |x₂+3| = a) :=
sorry

-- Part 2: Maximum number of terms in arithmetic sequence
theorem max_arith_seq_terms (a_1 a_2 : ℤ → ℤ) (n : ℕ) 
  (h₀ : ∀ k, a_1 k = a_2 k - 1) 
  (h₁ : ∀ k, a_2 k = a_1 k + 2) 
  (h₂ : ∑ k in range n, |a_1 k| = 507) 
  (h₃ : ∑ k in range n, |a_1 k + 1| = 507) 
  (h₄ : ∑ k in range n, |a_1 k - 2| = 507) :
  n ≤ 26 :=
sorry


end solve_abs_eq_max_arith_seq_terms_l657_657995


namespace find_m_probability_divisors_l657_657842

theorem find_m_probability_divisors (S : Finset ℕ)
  (hS : S = {d | ∃ b c : ℕ, 0 ≤ b ∧ b ≤ 24 ∧ 0 ≤ c ∧ c ≤ 8 ∧ d = 2^b * 3^c})
  (hS_card : S.card = 225) :
  let prob := ((Nat.choose 27 3 * Nat.choose 11 3) : ℚ) / 225^3 in
  ∃ m n : ℕ, m = 17 ∧ Nat.coprime m n ∧ prob = (m : ℚ) / (n : ℚ) :=
by
  sorry

end find_m_probability_divisors_l657_657842


namespace largest_side_of_quadrilateral_l657_657922

theorem largest_side_of_quadrilateral :
  ∃ (x : ℕ), x ≤ 6020 ∧ Math.Inf {(a : ℕ) | a ≤ 6020} = 6020 := 
sorry

end largest_side_of_quadrilateral_l657_657922


namespace cuboid_can_form_square_projection_l657_657640

-- Definitions and conditions based directly on the problem
def length1 := 3
def length2 := 4
def length3 := 6

-- Statement to prove
theorem cuboid_can_form_square_projection (x y : ℝ) :
  (4 * x * x + y * y = 36) ∧ (x + y = 4) → True :=
by sorry

end cuboid_can_form_square_projection_l657_657640


namespace increase_in_average_l657_657618

variable (A : ℝ)
variable (new_avg : ℝ := 44)
variable (score_12th_inning : ℝ := 55)
variable (total_runs_after_11 : ℝ := 11 * A)

theorem increase_in_average :
  ((total_runs_after_11 + score_12th_inning) / 12 - A = 1) :=
by
  sorry

end increase_in_average_l657_657618


namespace product_of_solutions_l657_657710

theorem product_of_solutions :
  (∀ x : ℝ, |3 * x - 2| + 5 = 23 → x = 20 / 3 ∨ x = -16 / 3) →
  (20 / 3 * -16 / 3 = -320 / 9) :=
by
  intros h
  have h₁ : 20 / 3 * -16 / 3 = -320 / 9 := sorry
  exact h₁

end product_of_solutions_l657_657710


namespace f_even_of_g_odd_l657_657486

theorem f_even_of_g_odd (g : ℝ → ℝ) (f : ℝ → ℝ) (h1 : ∀ x, g (-x) = -g x) (h2 : ∀ x, f x = |g (x^5)|) : ∀ x, f (-x) = f x := 
by
  sorry

end f_even_of_g_odd_l657_657486


namespace cube_sum_l657_657497

theorem cube_sum (a b : ℝ) (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end cube_sum_l657_657497


namespace numeral_in_150th_decimal_place_l657_657591

noncomputable def decimal_representation_13_17 : String :=
  "7647058823529411"

theorem numeral_in_150th_decimal_place :
  (decimal_representation_13_17.get (150 % 17)).iget = '1' :=
by
  sorry

end numeral_in_150th_decimal_place_l657_657591


namespace intersection_equality_l657_657769

def setA := {x : ℝ | (x - 1) * (3 - x) < 0}
def setB := {x : ℝ | -3 ≤ x ∧ x ≤ 3}

theorem intersection_equality : setA ∩ setB = {x : ℝ | -3 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_equality_l657_657769


namespace minimum_bailing_rate_needed_l657_657875

noncomputable def distance := 3 -- miles
noncomputable def leaking_rate := 7 -- gallons per minute
noncomputable def boat_capacity := 20 -- gallons
noncomputable def rowing_speed := 6 -- miles per hour
noncomputable def time_to_shore := (distance : ℝ) / rowing_speed * 60 -- minutes

theorem minimum_bailing_rate_needed : 
  ∃ r, r = (leaking_rate * time_to_shore - boat_capacity) / time_to_shore ∧ r = 6.33 :=
by
  unfold distance leaking_rate boat_capacity rowing_speed time_to_shore
  norm_num
  use 6.33
  split
  norm_num
  sorry

end minimum_bailing_rate_needed_l657_657875


namespace stack_height_difference_l657_657951

theorem stack_height_difference :
  ∃ S : ℕ,
    (7 + S + (S - 6) + (S + 4) + 2 * S = 55) ∧ (S - 7 = 3) := 
by 
  sorry

end stack_height_difference_l657_657951


namespace inequality_solution_l657_657532

def f (x : ℝ) : ℝ := ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution :
  { x : ℝ | f x > 0 } = {x : ℝ | x < 1} ∪ {x : ℝ | 2 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry


end inequality_solution_l657_657532


namespace truck_travel_yards_l657_657651

variables (b t : ℝ)

theorem truck_travel_yards : 
  (2 * (2 * b / 7) / (2 * t)) * 240 / 3 = (80 * b) / (7 * t) :=
by 
  sorry

end truck_travel_yards_l657_657651


namespace number_of_negative_x_l657_657340

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l657_657340


namespace count_negative_values_of_x_l657_657315

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l657_657315


namespace calculate_expression_l657_657385

variable (x : ℝ)

def quadratic_condition : Prop := x^2 + x - 1 = 0

theorem calculate_expression (h : quadratic_condition x) : 2*x^3 + 3*x^2 - x = 1 := by
  sorry

end calculate_expression_l657_657385


namespace count_negative_values_of_x_l657_657314

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l657_657314


namespace find_g3_l657_657919

variable (g : ℝ → ℝ)

axiom condition_g :
  ∀ x : ℝ, x ≠ 1 / 2 → g x + g ((x + 2) / (2 - 4 * x)) = 2 * x

theorem find_g3 : g 3 = 9 / 2 :=
  by
    sorry

end find_g3_l657_657919


namespace solve_quartic_eqn_l657_657260

noncomputable def solutionSet : Set ℂ :=
  {x | x^2 = 6 ∨ x^2 = -6}

theorem solve_quartic_eqn (x : ℂ) : (x^4 - 36 = 0) ↔ (x ∈ solutionSet) := 
sorry

end solve_quartic_eqn_l657_657260


namespace company_A_profit_l657_657672

-- Define the conditions
def total_profit (x : ℝ) : ℝ := x
def company_B_share (x : ℝ) : Prop := 0.4 * x = 60000
def company_A_percentage : ℝ := 0.6

-- Define the statement to be proved
theorem company_A_profit (x : ℝ) (h : company_B_share x) : 0.6 * x = 90000 := sorry

end company_A_profit_l657_657672


namespace find_a_l657_657123

theorem find_a (a b : ℤ) (p : ℕ) (hp : prime p) (h1 : p^2 + a * p + b = 0) (h2 : p^2 + b * p + 1100 = 0) :
  a = 274 ∨ a = 40 :=
sorry

end find_a_l657_657123


namespace group_division_l657_657445

theorem group_division (men women : ℕ) : 
  men = 4 → women = 5 →
  (∃ g1 g2 g3 : set (fin $ men + women), 
    g1.card = 3 ∧ g2.card = 3 ∧ g3.card = 3 ∧ 
    (∀ g, g ∈ [g1, g2, g3] → (∃ m w : ℕ, 1 ≤ m ∧ 1 ≤ w ∧ 
      finset.card (finset.filter (λ x, x < men) g) = m ∧ 
      finset.card (finset.filter (λ x, x ≥ men) g) = w)) 
    ∧ finset.disjoint g1 g2 ∧ finset.disjoint g2 g3 ∧ finset.disjoint g3 g1 
    ∧ g1 ∪ g2 ∪ g3 = finset.univ (fin $ men + women)) → 
  finset.card (finset.powerset' 3 (finset.univ (fin $ men + women))) / 2 = 180 :=
begin
  intros hmen hwomen,
  sorry
end

end group_division_l657_657445


namespace trig_identity_simplify_l657_657894

def simplify_trig_expr (a : ℝ) : ℝ :=
  sqrt (1 + sin a) + sqrt (1 - sin a)

theorem trig_identity_simplify (a : ℝ) (h : a = 10) : simplify_trig_expr a = -2 * sin 5 :=
  sorry

end trig_identity_simplify_l657_657894


namespace temperature_on_Friday_l657_657108

-- Define the temperatures for each day
variables (M T W Th F : ℕ)

-- Declare the given conditions as assumptions
axiom cond1 : (M + T + W + Th) / 4 = 48
axiom cond2 : (T + W + Th + F) / 4 = 46
axiom cond3 : M = 40

-- State the theorem
theorem temperature_on_Friday : F = 32 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end temperature_on_Friday_l657_657108


namespace negation_of_implication_l657_657546

theorem negation_of_implication (a b : ℝ) : ¬ (a > b → 2^a > 2^b) ↔ (a ≤ b → 2^a ≤ 2^b) :=
sorry

end negation_of_implication_l657_657546


namespace find_x_of_parallel_vectors_l657_657406

theorem find_x_of_parallel_vectors
  (x : ℝ)
  (p : ℝ × ℝ := (2, -3))
  (q : ℝ × ℝ := (x, 6))
  (h : ∃ k : ℝ, q = k • p) :
  x = -4 :=
sorry

end find_x_of_parallel_vectors_l657_657406


namespace andrew_kept_stickers_l657_657656

theorem andrew_kept_stickers : 
  ∀ (a d f : ℕ), 
  a = 750 → 
  d = 250 → 
  f = d + 120 → 
  a - (d + f) = 130 := by
  intros a d f ha hd hf
  rw [ha, hd, hf]
  simp
  sorry

end andrew_kept_stickers_l657_657656


namespace bisecting_BQ_l657_657839

open EuclideanGeometry

variables {A B C H P E F Q : Point}
variables {Γ : Circle}
variables [Triangle ABC]
variables [Orthocenter H ABC]
variables [Circumcircle Γ ABC]

/-- Assuming defined geometric properties of the triangle and points -/
def setup : Prop :=
  acute_angle_triangle ABC ∧
  orthocenter H ABC ∧
  circumcircle Γ ABC ∧
  B ∈ Γ ∧
  C ∈ Γ ∧
  ∃ (BH_inter : Line) (CH_inter : Line) (AH_inter : Line) (PE_inter : Line),
    BH_inter ∩ AC = E ∧
    CH_inter ∩ AB = F ∧
    AH_inter ∩ Γ = {A, P} ∧ P ≠ A ∧
    PE_inter ∩ Γ = {P, Q} ∧ Q ≠ P

/-- Proving that BQ bisects segment EF -/
theorem bisecting_BQ (h : setup) : bisects B Q E F :=
  sorry

end bisecting_BQ_l657_657839


namespace value_of_x_plus_2y_l657_657906

theorem value_of_x_plus_2y :
  let x := 3
  let y := 1
  x + 2 * y = 5 :=
by
  sorry

end value_of_x_plus_2y_l657_657906


namespace num_funcs_equal_to_y_eq_x_l657_657779

-- Define the functions based on the conditions
def f1 := fun x : ℝ => (Real.sqrt x) ^ 2
def f2 := fun x : ℝ => 3 * x ^ 3
def f3 := fun x : ℝ => Real.sqrt (x ^ 2)
def f4 := fun x : ℝ => if x ≠ 0 then x ^ 2 / x else 0

-- Define the function y = x
def y := fun x : ℝ => x

-- Define the proposition to verify
theorem num_funcs_equal_to_y_eq_x : 
  (finset.filter (fun f => ∀ x : ℝ, f x = y x) [f1, f2, f3, f4]).card = 1 := 
by 
  sorry

end num_funcs_equal_to_y_eq_x_l657_657779


namespace slower_train_speed_l657_657964

noncomputable def speed_of_slower_train 
  (length1 length2 : ℝ) 
  (time_to_cross : ℝ) 
  (speed_faster : ℝ) : ℝ :=
let total_distance := length1 + length2 in
let relative_speed := total_distance / time_to_cross in
let relative_speed_kmh := relative_speed * 3.6 in
relative_speed_kmh - speed_faster

theorem slower_train_speed : 
  speed_of_slower_train 120 160 10.07919366450684 60 ≈ 39.972 := by
  sorry

end slower_train_speed_l657_657964


namespace problem_1_l657_657658

theorem problem_1 :
  (-7/4) - (19/3) - 9/4 + 10/3 = -7 := by
  sorry

end problem_1_l657_657658


namespace ratio_areas_l657_657465

-- Definitions based on problem conditions
variable (T P Q R S : Type) 
variable [OrderedField ℝ]
variable (length_PQ : ℝ := 10)
variable (length_RS : ℝ := 23)
variable (area_TPQ : ℝ)
variable (area_PQRS : ℝ)

-- Theorem statement as described in the problem
theorem ratio_areas 
  (h_ratio: area_TPQ / (area_TPQ + area_PQRS) = (length_PQ * length_PQ) / (length_RS * length_RS) ) : 
  (area_TPQ / area_PQRS) = 100 / 429 :=
by
  -- Skipping the proof
  sorry

end ratio_areas_l657_657465


namespace parallel_lines_l657_657744

def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + 4 * y - 6 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := x + m * y - 3 = 0

theorem parallel_lines (m : ℝ) (x y : ℝ) : (m = 2) → ((m * x + 4 * y - 6 = 0) ∧ (x + m * y - 3 = 0)) → False :=
begin
  sorry
end

end parallel_lines_l657_657744


namespace largest_n_for_factoring_l657_657269

theorem largest_n_for_factoring :
  ∃ (n : ℤ), (∀ (A B : ℤ), (3 * A + B = n) → (3 * A * B = 90) → n = 271) :=
by sorry

end largest_n_for_factoring_l657_657269


namespace solve_quadratic_1_solve_quadratic_2_l657_657096

open Real

theorem solve_quadratic_1 :
  (∃ x : ℝ, x^2 - 2 * x - 7 = 0) ∧
  (∀ x : ℝ, x^2 - 2 * x - 7 = 0 → x = 1 + 2 * sqrt 2 ∨ x = 1 - 2 * sqrt 2) :=
sorry

theorem solve_quadratic_2 :
  (∃ x : ℝ, 3 * (x - 2)^2 = x * (x - 2)) ∧
  (∀ x : ℝ, 3 * (x - 2)^2 = x * (x - 2) → x = 2 ∨ x = 3) :=
sorry

end solve_quadratic_1_solve_quadratic_2_l657_657096


namespace system_of_equations_l657_657980

theorem system_of_equations (x y : ℝ) (h1 : 3 * x + 210 = 5 * y) (h2 : 10 * y - 10 * x = 100) :
    (3 * x + 210 = 5 * y) ∧ (10 * y - 10 * x = 100) := by
  sorry

end system_of_equations_l657_657980


namespace problem_statement_l657_657359

def f (t : ℝ) (x : ℝ) : ℝ :=
  if h : x ≠ 0 then
    (1 - x^2) / x^2
  else
    0  -- Define f (t) explicitly for other x if needed

lemma f_evaluation (x : ℝ) (h : x ≠ 0) : f (2 * x - 1) x = (1 - x^2) / x^2 :=
by refl

theorem problem_statement : f (0) (1 / 2) = 3 :=
by {
  unfold f,
  rw if_pos (by norm_num : (1 / 2 : ℝ) ≠ 0),
  norm_num,
  sorry, -- Proof to be completed
}

end problem_statement_l657_657359


namespace integral_f_x_4_f_deriv_x_l657_657060

open Real

theorem integral_f_x_4_f_deriv_x (f : ℝ → ℝ) (h_cont : Continuous f) (h_integral_1 : (∫ x in 0..1, f x * f' x) = 0) (h_integral_2 : (∫ x in 0..1, f x ^ 2 * f' x) = 18) : 
  (∫ x in 0..1, f x ^ 4 * f' x) = 486 / 5 := 
begin
  -- the proof will be provided here
  sorry
end

end integral_f_x_4_f_deriv_x_l657_657060


namespace side_length_percentage_error_l657_657211

variable (s s' : Real)
-- Conditions
-- s' = s * 1.06 (measured side length is 6% more than actual side length)
-- (s'^2 - s^2) / s^2 * 100% = 12.36% (percentage error in area)

theorem side_length_percentage_error 
    (h1 : s' = s * 1.06)
    (h2 : (s'^2 - s^2) / s^2 * 100 = 12.36) :
    ((s' - s) / s) * 100 = 6 := 
sorry

end side_length_percentage_error_l657_657211


namespace probability_symmetric_interval_l657_657369

noncomputable def random_variable : Type := ℝ -- Define the type of random variable as real numbers since it's continuous.
def xi : random_variable := 0 -- given mean of 0 for standard normal distribution

axiom P {p : Prop} : ℝ -- Probabilities as real numbers, using ℝ

axiom normal_distribution : random_variable → Prop
axiom standard_normal : normal_distribution xi
axiom P_xi_le_1_98 : P (xi ≤ 1.98) = 0.9762 -- Given condition

theorem probability_symmetric_interval : P (-1.98 < xi ∧ xi ≤ 1.98) = 0.9524 :=
by
  -- sorry is a placeholder for the actual proof steps
  sorry

end probability_symmetric_interval_l657_657369


namespace area_of_shaded_region_l657_657815

theorem area_of_shaded_region 
  (r R : ℝ)
  (hR : R = 9)
  (h : 2 * r = R) :
  π * R^2 - 3 * (π * r^2) = 20.25 * π :=
by
  sorry

end area_of_shaded_region_l657_657815


namespace find_dividend_l657_657161

def quotient : ℝ := -427.86
def divisor : ℝ := 52.7
def remainder : ℝ := -14.5
def dividend : ℝ := (quotient * divisor) + remainder

theorem find_dividend : dividend = -22571.122 := by
  sorry

end find_dividend_l657_657161


namespace necessary_but_not_sufficient_condition_for_ellipse_l657_657913

theorem necessary_but_not_sufficient_condition_for_ellipse {a b : ℝ} : (a > 0 → b > 0 → (ax^2 + by^2 = 1 → (a ≠ b → true)) → false) :=
sorry

end necessary_but_not_sufficient_condition_for_ellipse_l657_657913


namespace vector_parallel_x_equals_neg9_l657_657400

-- Define the vectors
def a : ℝ × ℝ := (3, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

-- Define the collinearity condition
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, v1 = λ • v2

-- The theorem to prove
theorem vector_parallel_x_equals_neg9 (x : ℝ) (h : parallel a (b x)) : x = -9 :=
by sorry

end vector_parallel_x_equals_neg9_l657_657400


namespace coefficient_x3_correct_l657_657703

/-- The sum of the expansions of (1 + x)^k from k=3 to k=20 -/
def sum_of_expansions (x : ℕ) : ℕ :=
  (∑ k in finset.range 18, (1 + x)^(k + 3))

/-- The coefficient of x^3 in the expansion of (1 + x)^21 - (1 + x)^3 / x -/
def coefficient_x3 (x : ℕ) : ℕ :=
  (binomial 21 4)

theorem coefficient_x3_correct (x : ℕ) : coefficient_x3 x = 5985 :=
sorry

end coefficient_x3_correct_l657_657703


namespace imaginary_part_of_z_l657_657920

-- Define z as a complex number
def z : ℂ := -1 / 2 + (1 / 2) * complex.I

-- The statement to prove
theorem imaginary_part_of_z : z.im = 1 / 2 :=
by sorry

end imaginary_part_of_z_l657_657920


namespace polygon_intersection_points_l657_657526

theorem polygon_intersection_points :
  ∀ (circle : Type) (P6 P7 P8 P9 : set circle),
  regular_polygon P6 6 → regular_polygon P7 7 → regular_polygon P8 8 → regular_polygon P9 9 →
  (∃ V1 V2 : circle, V1 ∈ P6 ∧ V1 ∈ P9 ∧ V2 ∈ P6 ∧ V2 ∈ P9) ∧
  (∀ (P Q : set circle), (P = P6 ∨ P = P7 ∨ P = P8 ∨ P = P9) ∧ (Q = P6 ∨ Q = P7 ∨ Q = P8 ∨ Q = P9) → 
    (P ≠ Q → (∃! V : circle, V ∈ P ∧ V ∈ Q ∧ V ∉ V1 ∧ V ∉ V2)) ∧ 
    (∀ V : circle, V ∉ P ∨ V ∉ Q ∨ (V = V1 ∨ V = V2)))
  → (number_of_intersections P6 P7 P8 P9 = 76) :=
by
  sorry

end polygon_intersection_points_l657_657526


namespace sum_abs_values_l657_657795

theorem sum_abs_values (a b : ℝ) (h₁ : abs a = 4) (h₂ : abs b = 7) (h₃ : a < b) : a + b = 3 ∨ a + b = 11 :=
by
  sorry

end sum_abs_values_l657_657795


namespace x_intercept_line_3x_plus_6_symmetric_line_L2_triangle_area_l657_657653

-- Part (a): Prove that the x-intercept of the line y = 3x + 6 is -2.
theorem x_intercept_line_3x_plus_6 : ∃ x : ℝ, 3*x + 6 = 0 ∧ x = -2 :=
by {
  use -2,
  split,
  { sorry },  -- Here, you would show the detailed proof steps,
  { sorry }
}

-- Part (b): Prove that the equation of the symmetric line L_2 is y = -3x + 6.
theorem symmetric_line_L2 :
  ∀ x : ℝ, (3 * x + 6) = 0 → (∀ x2 : ℝ, (-3 * x2 + 6) = 0) :=
by {
  intros x hx x2,
  split,
  { sorry },  -- Here, you would show the line equations and their relationship
  { sorry }
}

-- Part (c): Prove that the area of the triangle formed by the lines y=3x+6, y=-3x+6, and the x-axis is 12.
theorem triangle_area (L1 L2 : ℝ → ℝ) (hL1 : L1 = λ x, 3*x + 6) (hL2 : L2 = λ x, -3*x + 6) :
  let A : ℝ := (1/2) * (4) * (6) in A = 12 :=
by {
  calc
  (1/2) * 4 * 6 = 12 : by sorry,
  sorry
}

end x_intercept_line_3x_plus_6_symmetric_line_L2_triangle_area_l657_657653


namespace count_negative_values_correct_l657_657282

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l657_657282


namespace intersection_eq_l657_657231

namespace Proof

universe u

-- Define the natural number set M
def M : Set ℕ := { x | x > 0 ∧ x < 6 }

-- Define the set N based on the condition |x-1| ≤ 2
def N : Set ℝ := { x | abs (x - 1) ≤ 2 }

-- Define the complement of N with respect to the real numbers
def ComplementN : Set ℝ := { x | x < -1 ∨ x > 3 }

-- Define the intersection of M and the complement of N
def IntersectMCompN : Set ℕ := { x | x ∈ M ∧ (x : ℝ) ∈ ComplementN }

-- Provide the theorem to be proved
theorem intersection_eq : IntersectMCompN = { 4, 5 } :=
by
  sorry

end Proof

end intersection_eq_l657_657231


namespace triangle_similarity_failure_l657_657813

-- Define the triangles and conditions
variable (A B C A' B' C' : Type)
variable [Triangle ABC : triangle A B C]
variable [Triangle A'B'C' : triangle A' B' C']

-- Assume the conditions given in the problem
hypothesis h1 : angle C = angle (C' : 90)
hypothesis hA : angle A = angle (A' : 90)
hypothesis hB : (AC / (A'C' )) = (BC / (B'C' ))
hypothesis hC : angle B = angle (B' : 90)
hypothesis hD : AC / (A'C' )≠ BC / (B'C' )

-- State the theorem to be proved
theorem triangle_similarity_failure (A B C A' B' C' : Type) [Triangle ABC : triangle A B C] [Triangle A'B'C' : triangle A' B' C'] (h1 : angle C = angle (C' : 90)) (h2 : angle A = angle (A' : 90)) (h3 : (AC / (A'C' )) = (BC / (B'C' ))) (h4 : angle B = angle (B' : 90)) (h5 : AC / (A'C' )≠ BC / (B'C' )) : 
∃ D, ¬ (similarity_criterion ABC A'B'C' D) :=
sorry

end triangle_similarity_failure_l657_657813


namespace solve_equation_l657_657942

theorem solve_equation : ∀ x : ℝ, (4 ^ x = 2 ^ (x + 1) - 1) → x = 0 := by
  intros x h
  sorry

end solve_equation_l657_657942


namespace problem1_l657_657661

theorem problem1 : (- (1 / 12) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = -21 :=
by
  sorry

end problem1_l657_657661


namespace smallest_k_iteration_to_zero_l657_657063

def f (a b n : ℤ) (M : ℤ) : ℤ :=
  if n < M then n + a else n - b

def iterate_f {a b n : ℤ} (M : ℤ) (f : ℤ → ℤ) (i : ℕ) : ℤ :=
  if i = 0 then n else iterate_f M f (i - 1) (f n)

theorem smallest_k_iteration_to_zero (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) :
  let d := Int.gcd a b
  let M := Int.floor ((a + b) / 2)
  let k := (a + b) / d in
  (∃ m ≥ 1, iterate_f M (f a b) m 0 = 0) → k = (a + b) / d :=
sorry

end smallest_k_iteration_to_zero_l657_657063


namespace sum_of_polynomials_l657_657834

open Polynomial

noncomputable def f : ℚ[X] := -4 * X^2 + 2 * X - 5
noncomputable def g : ℚ[X] := -6 * X^2 + 4 * X - 9
noncomputable def h : ℚ[X] := 6 * X^2 + 6 * X + 2

theorem sum_of_polynomials :
  f + g + h = -4 * X^2 + 12 * X - 12 :=
by sorry

end sum_of_polynomials_l657_657834


namespace A_plus_B_eq_one_fourth_l657_657064

noncomputable def A : ℚ := 1 / 3
noncomputable def B : ℚ := -1 / 12

theorem A_plus_B_eq_one_fourth :
  A + B = 1 / 4 := by
  sorry

end A_plus_B_eq_one_fourth_l657_657064


namespace general_formula_b_seq_l657_657504

open Real

noncomputable def a_seq : ℕ → ℝ
| 1     := 2
| (n+1) := 2 / (a_seq n + 1)

def b_seq (n : ℕ) : ℝ := abs ((a_seq n + 2) / (a_seq n - 1))

theorem general_formula_b_seq (n : ℕ) (hn : n > 0) : b_seq n = 2^(n + 1) :=
by
  sorry

end general_formula_b_seq_l657_657504


namespace max_possible_A_l657_657929

-- Definitions for the conditions
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def strictlyDecreasing (l : List ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → j < l.length → l.nth i > l.nth j

def consecutiveMultiplesOfThree (l : List ℕ) : Prop :=
  ∀ (i : ℕ), i < l.length - 1 → l.nth i + 3 = l.nth (i + 1)

def consecutiveOdds (l : List ℕ) : Prop :=
  ∀ (i : ℕ), i < l.length - 1 → l.nth i + 2 = l.nth (i + 1)

def sumNine (l : List ℕ) : Prop :=
  l.sum = 9

-- The problem statement
theorem max_possible_A : ∃ (A B C D E F G H I J : ℕ),
  A ∈ digits ∧ B ∈ digits ∧ C ∈ digits ∧
  D ∈ digits ∧ E ∈ digits ∧ F ∈ digits ∧
  G ∈ digits ∧ H ∈ digits ∧ I ∈ digits ∧ J ∈ digits ∧
  strictlyDecreasing [A, B, C] ∧ strictlyDecreasing [D, E, F] ∧ strictlyDecreasing [G, H, I, J] ∧
  consecutiveMultiplesOfThree [D, E, F] ∧ consecutiveOdds [G, H, I, J] ∧
  sumNine [A, B, C] ∧
  (A = 8) :=
sorry

end max_possible_A_l657_657929


namespace workers_in_first_group_l657_657100

-- Define the first condition: Some workers collect 48 kg of cotton in 4 days
def cotton_collected_by_W_workers_in_4_days (W : ℕ) : ℕ := 48

-- Define the second condition: 9 workers collect 72 kg of cotton in 2 days
def cotton_collected_by_9_workers_in_2_days : ℕ := 72

-- Define the rate of cotton collected per worker per day for both scenarios
def rate_per_worker_first_group (W : ℕ) : ℕ :=
cotton_collected_by_W_workers_in_4_days W / (W * 4)

def rate_per_worker_second_group : ℕ :=
cotton_collected_by_9_workers_in_2_days / (9 * 2)

-- Given the rates are the same for both groups, prove W = 3
theorem workers_in_first_group (W : ℕ) (h : rate_per_worker_first_group W = rate_per_worker_second_group) : W = 3 :=
sorry

end workers_in_first_group_l657_657100


namespace number_of_subsets_l657_657708

def num_subsets (n : ℕ) : ℕ := 2 ^ n

theorem number_of_subsets (A : Finset α) (n : ℕ) (h : A.card = n) : A.powerset.card = num_subsets n :=
by
  have : A.powerset.card = 2 ^ A.card := sorry -- Proof omitted
  rw [h] at this
  exact this

end number_of_subsets_l657_657708


namespace partition_unique_N2_partition_not_unique_N3_l657_657843

-- Define conditions
structure ArithmeticProgression (α : Type) :=
(start : α)
(diff : ℕ)

def is_partition {α : Type} [AddGroup α] [LinearOrder α] (X : Set α) (P : List (ArithmeticProgression α)) : Prop :=
  -- P is a list of arithmetic progressions covering X with no intersections
  (∀ p ∈ P, ∃ (a : α), ∃ (d : ℕ), p = (⟨a, d⟩ : ArithmeticProgression α)) ∧
  (⋃ p ∈ P, { x | ∃ (n : ℤ), x = p.start + n * p.diff }) = X ∧
  (∀ p1 p2 ∈ P, p1 ≠ p2 → disjoint (⋃ n : ℤ, { p1.start + n * p1.diff }) (⋃ n : ℤ, { p2.start + n * p2.diff }))

def unique_partition {α : Type} [AddGroup α] [LinearOrder α] (X : Set α) (P : List (ArithmeticProgression α)) : Prop :=
  ∀ Q : List (ArithmeticProgression α), is_partition X Q → Q = P

-- Theorem statement for N = 2
theorem partition_unique_N2 : ∀ (X : Set ℤ) (P : List (ArithmeticProgression ℤ)),
  is_partition X P → 
  (length P) = 2 → 
  unique_partition X P := 
sorry

-- Theorem statement for N = 3
theorem partition_not_unique_N3 : ∃ (X : Set ℤ), ∃ (P Q : List (ArithmeticProgression ℤ)),
  is_partition X P ∧ 
  is_partition X Q ∧ 
  length P = 3 ∧ 
  length Q = 3 ∧ 
  P ≠ Q := 
sorry

end partition_unique_N2_partition_not_unique_N3_l657_657843


namespace num_possible_radii_l657_657224

theorem num_possible_radii :
  let s_vals := {s : ℕ | s < 144 ∧ 144 % s = 0}
  ∃ m : ℕ, m = s_vals.card ∧ m = 14 :=
by
  sorry

end num_possible_radii_l657_657224


namespace count_quadratic_polynomials_l657_657483

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 4)

theorem count_quadratic_polynomials (Q : ℝ → ℝ) :
  (∃ (R : ℝ → ℝ), degree R = 3 ∧ ∀ x, P (Q x) = P x * R x) ↔ 
  (∃ qps : Finset (ℝ → ℝ), qps.card = 4 ∧ (∀ q ∈ qps, degree q = 2)) := sorry

end count_quadratic_polynomials_l657_657483


namespace radius_of_circle_B_l657_657669

-- Definitions of circles and their properties
noncomputable def circle_tangent_externally (r1 r2 : ℝ) := ∃ d : ℝ, d = r1 + r2
noncomputable def circle_tangent_internally (r1 r2 : ℝ) := ∃ d : ℝ, d = r2 - r1

-- Problem statement in Lean 4
theorem radius_of_circle_B
  (rA rB rC rD centerA centerB centerC centerD : ℝ)
  (h_rA : rA = 2)
  (h_congruent_B_C : rB = rC)
  (h_circle_A_tangent_to_B : circle_tangent_externally rA rB)
  (h_circle_A_tangent_to_C : circle_tangent_externally rA rC)
  (h_circle_B_C_tangent_e : circle_tangent_externally rB rC)
  (h_circle_B_D_tangent_i : circle_tangent_internally rB rD)
  (h_center_A_passes_D : centerA = centerD)
  (h_rD : rD = 4) : 
  rB = 1 := sorry

end radius_of_circle_B_l657_657669


namespace problem_1_problem_2_l657_657098

open Real

noncomputable def f (x : ℝ) : ℝ := log10 (abs (x + 3) - abs (x - 7))

theorem problem_1 : ∀ x : ℝ, log10 (abs (x + 3) - abs (x - 7)) < 1 ↔ (2 < x ∧ x < 7) :=
by
  sorry

theorem problem_2 : ∀ m : ℝ, (∀ x : ℝ, f x < m) ↔ m > 1 :=
by
  sorry

end problem_1_problem_2_l657_657098


namespace minimum_distance_between_points_l657_657748

noncomputable def minimum_distance : ℝ := 2 * real.sqrt 5 / 5

theorem minimum_distance_between_points :
  ∀ (P : ℝ × ℝ) (Q : ℝ × ℝ),
    (P.2 = 2 * P.1 + 1) →
    (Q.2 = Q.1 + real.log Q.1) →
    ∃ d, d = minimum_distance ∧
               ∀ (y : ℝ), 0 < y → y = d :=
begin
  -- sorry is used here to skip the proof
  sorry
end

end minimum_distance_between_points_l657_657748


namespace sum_two_angles_greater_third_l657_657885

-- Definitions of the angles and the largest angle condition
variables {P A B C} -- Points defining the trihedral angle
variables {α β γ : ℝ} -- Angles α, β, γ
variables (h1 : γ ≥ α) (h2 : γ ≥ β)

-- Statement of the theorem
theorem sum_two_angles_greater_third (P A B C : Type*) (α β γ : ℝ)
  (h1 : γ ≥ α) (h2 : γ ≥ β) : α + β > γ :=
sorry  -- Proof is omitted

end sum_two_angles_greater_third_l657_657885


namespace student_weight_loss_l657_657199

variables (S R L : ℕ)

theorem student_weight_loss :
  S = 75 ∧ S + R = 110 ∧ S - L = 2 * R → L = 5 :=
by
  sorry

end student_weight_loss_l657_657199


namespace kuziya_probability_l657_657540

-- Definitions for our problem
def is_at_distance (A : ℝ) (h : ℝ) (n : ℕ) : ℝ → Prop :=
  λ x, ∃ k m : ℕ, k + m = n ∧ k - m = 4 ∧ x = A + (k - m) * h

-- Probability calculation, restricting to range
def prob_at_distance (A : ℝ) (h : ℝ) : ℚ :=
  ∑ n in (finset.range 7).image (λ x, x + 3), ℙ (n % 2 = 0 ∧ is_at_distance A h n (A + 4 * h))

-- Main theorem statement
theorem kuziya_probability (A h : ℝ) :
  prob_at_distance A h = 47 / 224 :=
sorry

end kuziya_probability_l657_657540


namespace matrix_eigenvalue_problem_l657_657399

theorem matrix_eigenvalue_problem (a b : ℝ) (λ : ℝ) :
  let M : Matrix (Fin 2) (Fin 2) ℝ := ![![a, 1], ![b, 1]],
      v : Matrix (Fin 2) (Fin 1) ℝ := ![![1], ![0]],
      λv := λ • v in
  (M ⬝ v = λv) →
  a = 1 ∧ b = 0 ∧
  let Minv := M⁻¹ in
  Minv = ![![1, -1], ![0, 1]] ∧
  (∀ (x y : ℝ), x^2 + 2 * x * y + 2 * y^2 = 1 →
    let Mv := M ⬝ ![![x], ![y]],
        x' := Mv 0 0,
        y' := Mv 1 0 in
    x'^2 + y'^2 = 1) :=
sorry

end matrix_eigenvalue_problem_l657_657399


namespace lily_has_26_dollars_left_for_coffee_l657_657507

-- Define the initial amount of money Lily has
def initialMoney : ℕ := 60

-- Define the costs of items
def celeryCost : ℕ := 5
def cerealCost : ℕ := 12 / 2
def breadCost : ℕ := 8
def milkCost : ℕ := 10 * 9 / 10
def potatoCostEach : ℕ := 1
def numberOfPotatoes : ℕ := 6
def totalPotatoCost : ℕ := potatoCostEach * numberOfPotatoes

-- Define the total amount spent on the items
def totalSpent : ℕ := celeryCost + cerealCost + breadCost + milkCost + totalPotatoCost

-- Define the amount left for coffee
def amountLeftForCoffee : ℕ := initialMoney - totalSpent

-- The theorem to prove
theorem lily_has_26_dollars_left_for_coffee :
  amountLeftForCoffee = 26 := by
  sorry

end lily_has_26_dollars_left_for_coffee_l657_657507


namespace isosceles_triangle_l657_657072

theorem isosceles_triangle (A B C D E P G : Point) 
  (h1 : square A B D E)
  (h2 : square B C P G)
  (h3 : parallel DG AC) :
  is_isosceles ABC :=
sorry

end isosceles_triangle_l657_657072


namespace sum_of_squares_sum_of_cubes_l657_657659

def S2 (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), i^2

def S3 (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), i^3

theorem sum_of_squares (n : ℕ) : 
  S2 n = n * (n + 1) * (2 * n + 1) / 6 :=
by
  sorry

theorem sum_of_cubes (n : ℕ) : 
  S3 n = (n * (n + 1) / 2) ^ 2 :=
by
  sorry

end sum_of_squares_sum_of_cubes_l657_657659


namespace painting_price_difference_l657_657079

theorem painting_price_difference :
  let previous_painting := 9000
  let recent_painting := 44000
  let five_times_more := 5 * previous_painting + previous_painting
  five_times_more - recent_painting = 10000 :=
by
  intros
  sorry

end painting_price_difference_l657_657079


namespace area_difference_l657_657070

-- Define the unit square
def unit_square : Type := {
  side : ℝ // side = 1
}

-- Define an equilateral triangle with a given side length
def equilateral_triangle (s: ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

-- Define the regions R' and S'
def region_R (side_1: ℝ) (side_2: ℝ) (side_3: ℝ) (side_4: ℝ): ℝ :=
  1 + 4 * (equilateral_triangle side_2) + 8 * (equilateral_triangle side_1)

def region_S (side_length: ℝ) : ℝ :=
  (3 * sqrt 3 / 2) * side_length^2

theorem area_difference : 
  (region_S 3 - region_R 1 2 2 2) = 7.5 * sqrt 3 - 1 :=
by
  sorry

end area_difference_l657_657070


namespace cube_paint_probability_l657_657250

theorem cube_paint_probability:
  let colors := { "red", "blue", "green" }
  let faces := { 0, 1, 2, 3, 4, 5 }
  let painting : faces → colors := λ f, if f == 0 then "red" else if f == 1 then "blue" else "green" in
  let count_arrangements := 729 in
  let all_same := 3 in
  let five_same := 36 in
  let four_vertical_same := 36 in
  (all_same + five_same + four_vertical_same) / count_arrangements = 25 / 243 :=
by sorry

end cube_paint_probability_l657_657250


namespace smallest_value_w_i_l657_657847

theorem smallest_value_w_i (w : ℂ) (h : |w^2 + 1| = |w * (w + complex.I)|) :
  ∃ (z : ℂ), |w - complex.I| ≥ 1 ∧ (∀ y, |y - complex.I| < 1 → |y^2 + 1| ≠ |y * (y + complex.I)|) :=
sorry

end smallest_value_w_i_l657_657847


namespace third_in_decomposition_5_pow4_l657_657352

theorem third_in_decomposition_5_pow4 : ∃ x, (∃ l : List ℕ, l = [121, 123, 125, 127, 129] ∧ x = l.nthLe 2 (by norm_num) ∧ x = 125) :=
by {
  existsi 125,
  existsi ([121, 123, 125, 127, 129] : List ℕ),
  split,
  exact rfl,
  split,
  norm_num,
  refl,
}

end third_in_decomposition_5_pow4_l657_657352


namespace ivy_baked_55_cupcakes_l657_657470

-- Definitions based on conditions
def cupcakes_morning : ℕ := 20
def cupcakes_afternoon : ℕ := cupcakes_morning + 15
def total_cupcakes : ℕ := cupcakes_morning + cupcakes_afternoon

-- Theorem statement that needs to be proved
theorem ivy_baked_55_cupcakes : total_cupcakes = 55 := by
    sorry

end ivy_baked_55_cupcakes_l657_657470


namespace man_l657_657185

noncomputable def speed_of_current : ℝ := 3 -- in kmph
noncomputable def time_to_cover_100_meters_downstream : ℝ := 19.99840012798976 -- in seconds
noncomputable def distance_covered : ℝ := 0.1 -- in kilometers (100 meters)

noncomputable def speed_in_still_water : ℝ :=
  (distance_covered / (time_to_cover_100_meters_downstream / 3600)) - speed_of_current

theorem man's_speed_in_still_water :
  speed_in_still_water = 14.9997120913593 :=
  by
    sorry

end man_l657_657185


namespace coeff_x3y5_in_expansion_l657_657970

theorem coeff_x3y5_in_expansion : 
  (nat.choose 8 3) = 56 := 
by
  sorry

end coeff_x3y5_in_expansion_l657_657970


namespace slowest_time_l657_657860

open Real

def time_lola (stories : ℕ) (run_time : ℝ) : ℝ := stories * run_time

def time_sam (stories_run stories_elevator : ℕ) (run_time elevate_time stop_time : ℝ) (wait_time : ℝ) : ℝ :=
  let run_part  := stories_run * run_time
  let wait_part := wait_time
  let elevator_part := stories_elevator * elevate_time + (stories_elevator - 1) * stop_time
  run_part + wait_part + elevator_part

def time_tara (stories : ℕ) (elevate_time stop_time : ℝ) : ℝ :=
  stories * elevate_time + (stories - 1) * stop_time

theorem slowest_time 
  (build_stories : ℕ) (lola_run_time sam_run_time elevate_time stop_time wait_time : ℝ)
  (h_build : build_stories = 50)
  (h_lola_run : lola_run_time = 12) (h_sam_run : sam_run_time = 15)
  (h_elevate : elevate_time = 10) (h_stop : stop_time = 4) (h_wait : wait_time = 20) :
  max (time_lola build_stories lola_run_time) 
    (max (time_sam 25 25 sam_run_time elevate_time stop_time wait_time) 
         (time_tara build_stories elevate_time stop_time)) = 741 := by
  sorry

end slowest_time_l657_657860


namespace num_of_negative_x_l657_657291

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l657_657291


namespace opposite_of_number_l657_657547

-- Define the original number
def original_number : ℚ := -1 / 6

-- Statement to prove
theorem opposite_of_number : -original_number = 1 / 6 := by
  -- This is where the proof would go
  sorry

end opposite_of_number_l657_657547


namespace min_vertical_segment_len_l657_657114

theorem min_vertical_segment_len :
  ∃ (x : ℝ), (x < 0 ∧ (|x - (-x^2 - 5 * x - 3)| = 1)) ∨ 
           (x ≥ 0 ∧ (|-x - (-x^2 - 5 * x - 3)| = 1)) :=
by {
  use -2,
  left,
  split,
  { sorry },  -- confirming x < 0
  { sorry }   -- confirming the vertical segment length
}

end min_vertical_segment_len_l657_657114


namespace orthogonal_vectors_l657_657378

open Real

variables (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : (a + b)^2 = (a - b)^2)

theorem orthogonal_vectors (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (h : (a + b)^2 = (a - b)^2) : a * b = 0 :=
by 
  sorry

end orthogonal_vectors_l657_657378


namespace proof_problem_l657_657463

/-- A three-digit natural number n is called a "good number" if each digit is not 0,
and the sum of the hundreds digit and the tens digit is exactly divisible by the units digit. -/
def is_good_number (n : ℕ) : Prop :=
  let h := n / 100 in
  let t := (n % 100) / 10 in
  let u := n % 10 in
  h ≠ 0 ∧ t ≠ 0 ∧ u ≠ 0 ∧ (h + t) % u = 0

/-- The hundreds digit is 5 greater than the tens digit. -/
def is_good_with_hundreds_greater_by_5 (n : ℕ) : Prop :=
  let h := n / 100 in
  let t := (n % 100) / 10 in
  let u := n % 10 in
  h = t + 5 ∧ is_good_number n

/-- The proof problem is to show that:
1. 312 is a "good number" and 675 is not a "good number".
2. The number of "good numbers" where the hundreds digit is 5 greater than the tens digit is 7.
-/
theorem proof_problem :
  is_good_number 312 ∧ ¬is_good_number 675 ∧
  (finset.filter is_good_with_hundreds_greater_by_5 (finset.range 1000)).card = 7 :=
by
  sorry

end proof_problem_l657_657463


namespace negative_values_count_l657_657322

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l657_657322


namespace train_length_l657_657157

theorem train_length (L : ℝ) (V1 V2 : ℝ) 
  (h1 : V1 = L / 15) 
  (h2 : V2 = (L + 800) / 45) 
  (h3 : V1 = V2) : 
  L = 400 := 
sorry

end train_length_l657_657157


namespace systematic_sampling_fifth_group_l657_657139

theorem systematic_sampling_fifth_group (
  total_students : ℕ,
  sample_size : ℕ,
  first_group_number : ℕ
) (h_total : total_students = 2000)
  (h_sample : sample_size = 100)
  (h_first : first_group_number = 11) :
  let interval := total_students / sample_size
  let fifth_group_number := first_group_number + interval * 4
  in fifth_group_number = 91 :=
by
  -- Definitions and assumptions
  have h_interval : interval = 20 := by rw [h_total, h_sample]; norm_num
  have h_fifth_group : fifth_group_number = 11 + 20 * 4 := by rw [h_first, h_interval]; norm_num
  -- Conclusion
  exact h_fifth_group.symm

end systematic_sampling_fifth_group_l657_657139


namespace negative_values_of_x_l657_657301

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l657_657301


namespace sequence_integers_l657_657940

theorem sequence_integers (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 1) 
  (h3 : ∀ n ≥ 3, a n = (a (n - 1))^2 + 2 / a (n - 2)) : ∀ n, ∃ k : ℤ, a n = k :=
sorry

end sequence_integers_l657_657940


namespace num_elements_set_is_4_l657_657105

def a_n (n : ℕ) : ℕ := 2 * n
def term (n : ℕ) : ℚ := 1 / ((n + 1) * a_n n)
def elements_set := {term n | n in [1, 2, 3, 4, 5]}

theorem num_elements_set_is_4 : ∀ x > 1, x^2 < 3 → elements_set.card = 4 := by
  sorry

end num_elements_set_is_4_l657_657105


namespace sum_of_series_l657_657577

theorem sum_of_series : 
  (∑ n in Finset.range 2015, (2 / ((n+1)*(n+1 + 3)))) = 1.220 := by
  sorry

end sum_of_series_l657_657577


namespace intersection_proof_l657_657029

noncomputable def M : Set ℕ := {1, 3, 5, 7, 9}

noncomputable def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_proof : M ∩ (N ∩ Set.univ) = {5, 7, 9} :=
by sorry

end intersection_proof_l657_657029


namespace dot_product_sum_eq_fifteen_l657_657362

-- Define the vectors a, b, and c
def vec_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vec_b (y : ℝ) : ℝ × ℝ := (1, y)
def vec_c : ℝ × ℝ := (3, -6)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Conditions from the problem
def cond_perpendicular (x : ℝ) : Prop :=
  dot_product (vec_a x) vec_c = 0

def cond_parallel (y : ℝ) : Prop :=
  1 / 3 = y / -6

-- Lean statement for the problem
theorem dot_product_sum_eq_fifteen (x y : ℝ)
  (h1 : cond_perpendicular x) 
  (h2 : cond_parallel y) :
  dot_product (vec_a x + vec_b y) vec_c = 15 :=
sorry

end dot_product_sum_eq_fifteen_l657_657362


namespace intersection_proof_l657_657031

noncomputable def M : Set ℕ := {1, 3, 5, 7, 9}

noncomputable def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_proof : M ∩ (N ∩ Set.univ) = {5, 7, 9} :=
by sorry

end intersection_proof_l657_657031


namespace area_of_new_triangle_geq_twice_sum_of_areas_l657_657568

noncomputable def area_of_triangle (a b c : ℝ) (alpha : ℝ) : ℝ :=
  0.5 * a * b * (Real.sin alpha)

theorem area_of_new_triangle_geq_twice_sum_of_areas
  (a1 b1 c a2 b2 alpha : ℝ)
  (h1 : a1 <= b1) (h2 : b1 <= c) (h3 : a2 <= b2) (h4 : b2 <= c) :
  let α_1 := Real.arcsin ((a1 + a2) / (2 * c))
  let area1 := area_of_triangle a1 b1 c alpha
  let area2 := area_of_triangle a2 b2 c alpha
  let area_new := area_of_triangle (a1 + a2) (b1 + b2) (2 * c) α_1
  area_new >= 2 * (area1 + area2) :=
sorry

end area_of_new_triangle_geq_twice_sum_of_areas_l657_657568


namespace number_of_negative_x_l657_657341

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l657_657341


namespace cyclist_go_south_speed_l657_657961

noncomputable def speed_of_cyclist_go_south (v : ℝ) : Prop :=
  let north_speed := 10 -- speed of cyclist going north in kmph
  let time := 2 -- time in hours
  let distance := 50 -- distance apart in km
  (north_speed + v) * time = distance

theorem cyclist_go_south_speed (v : ℝ) : speed_of_cyclist_go_south v → v = 15 :=
by
  intro h
  -- Proof part is skipped
  sorry

end cyclist_go_south_speed_l657_657961


namespace steve_reads_book_in_weeks_l657_657899

noncomputable def total_pages_per_week (pages1 : ℕ) (days1 : ℕ) (pages2 : ℕ) (days2 : ℕ) : ℕ :=
  pages1 * days1 + pages2 * days2

def total_weeks (book_pages : ℕ) (pages_per_week : ℕ) : ℕ :=
  book_pages / pages_per_week + if book_pages % pages_per_week = 0 then 0 else 1

theorem steve_reads_book_in_weeks :
  total_weeks 2100 (total_pages_per_week 100 3 150 2) = 4 :=
by 
  unfold total_pages_per_week
  unfold total_weeks
  norm_num
  rw Nat.div_eq_of_lt
  · norm_num
  · norm_num
  sorry

end steve_reads_book_in_weeks_l657_657899


namespace express_set_l657_657698

open Set

/-- Define the set of natural numbers for which an expression is also a natural number. -/
theorem express_set : {x : ℕ | ∃ y : ℕ, 6 = y * (5 - x)} = {2, 3, 4} :=
by
  sorry

end express_set_l657_657698


namespace stretch_transformation_resulting_curve_l657_657731

-- Define the original circle equation
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the stretch transformation
def stretch_x (x : ℝ) : ℝ := 2 * x
def stretch_y (y : ℝ) : ℝ := 3 * y

-- Define the inverse stretch transformation
def inv_stretch_x (x' : ℝ) : ℝ := x' / 2
def inv_stretch_y (y' : ℝ) : ℝ := y' / 3

-- State the main problem
theorem stretch_transformation_resulting_curve (x' y' : ℝ) :
  ∃ x y : ℝ, circle_eq x y ∧ x' = stretch_x x ∧ y' = stretch_y y →
  (x'^2 / 4 + y'^2 / 9 = 1) :=
begin
  sorry
end

end stretch_transformation_resulting_curve_l657_657731


namespace mikaela_initial_paint_containers_l657_657870

theorem mikaela_initial_paint_containers
    (initial_paint_containers : ℕ)
    (ceil_paint_used leftover_paint_containers : ℕ)
    (plan_walls : ℕ := 4)
    (walls_painted : ℕ := 3)
    (ceil_paint_used = 1)
    (leftover_paint_containers = 3)
    (x = ceil_paint_used + leftover_paint_containers + walls_painted) :
    initial_paint_containers = 8 := 
begin
  sorry
end

end mikaela_initial_paint_containers_l657_657870


namespace bathroom_width_l657_657176

def length : ℝ := 4
def area : ℝ := 8
def width : ℝ := 2

theorem bathroom_width :
  area = length * width :=
by
  sorry

end bathroom_width_l657_657176


namespace largest_n_factorization_l657_657270

theorem largest_n_factorization :
  ∃ (n : ℤ), (∀ A B : ℤ, 3 * B + A = n -> A * B = 90 -> n ≤ 271) ∧ (∃ A B : ℤ, 3 * B + A = 271 ∧ A * B = 90) :=
by {
  apply Exists.intro 271,
  constructor,
  {
    intros A B eqn₁ eqn₂,
    have aux : 3 * B + A ≤ 271,
    sorry, -- Proof steps would go here
    exact aux,
  },
  {
    apply Exists.intro 1,
    apply Exists.intro 90,
    split,
    exact rfl,
    exact rfl,
  }
}

end largest_n_factorization_l657_657270


namespace possible_values_of_a_l657_657121

theorem possible_values_of_a 
  (p : ℕ) (hp : p.prime) 
  (a b k : ℤ) 
  (h1 : -p^2 * (1 + k) = 1100)
  (hb : b = k * p) 
  (ha : a = -((b + p^2) / p)) :
  (a = 274 ∨ a = 40) :=
sorry

end possible_values_of_a_l657_657121


namespace max_inscribed_circle_area_of_triangle_l657_657392

theorem max_inscribed_circle_area_of_triangle
  (a b : ℝ)
  (ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (f1 f2 : ℝ × ℝ)
  (F1_coords : f1 = (-1, 0))
  (F2_coords : f2 = (1, 0))
  (P Q : ℝ × ℝ)
  (line_through_F2 : ∀ y : ℝ, x = 1 → y^2 = 9 / 4)
  (P_coords : P = (1, 3/2))
  (Q_coords : Q = (1, -3/2))
  : (π * (3 / 4)^2 = 9 * π / 16) :=
  sorry

end max_inscribed_circle_area_of_triangle_l657_657392


namespace negative_values_count_l657_657330

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l657_657330


namespace odd_natural_numbers_last_4_digits_equal_count_l657_657207

def is_odd (n : ℕ) : Prop := n % 2 = 1

def last_4_digits (n : ℕ) : ℕ := n % 10000

theorem odd_natural_numbers_last_4_digits_equal_count :
  let odd_numbers := {n : ℕ | is_odd n ∧ 1 ≤ n ∧ n < 10000} in
  (∃ f : nat → nat, ∀ n ∈ odd_numbers, (last_4_digits (n^9) > n) ↔ (last_4_digits (n^9) < n) :=
sorry

end odd_natural_numbers_last_4_digits_equal_count_l657_657207


namespace exists_partition_with_equal_sum_l657_657413

/-- A domino piece is represented as a pair of integers. -/
def Domino := (ℕ × ℕ)

/-- The standard double-six set of dominoes. -/
def standardDoubleSixSet : set Domino :=
  { d | (d.1 <= 6 ∧ d.2 <= 6) ∧ (d.1, d.2) = (d.2, d.1) }

/-- Total sum of points on all dominoes in the standard double-six set. -/
def totalDominoPoints : ℕ :=
  ∑ d in standardDoubleSixSet, (d.1 + d.2)

/-- Define the partition -/
structure partition (dominoes : set Domino) :=
  (part1 : set Domino)
  (part2 : set Domino)
  (part3 : set Domino)
  (part4 : set Domino)
  (disjoint : part1 ∩ part2 = ∅ ∧ part1 ∩ part3 = ∅ ∧ part1 ∩ part4 = ∅ ∧ part2 ∩ part3 = ∅ ∧ part2 ∩ part4 = ∅ ∧ part3 ∩ part4 = ∅)
  (union  : part1 ∪ part2 ∪ part3 ∪ part4 = dominoes)
  (sum1 : ∑ d in part1, (d.1 + d.2) = 21)
  (sum2 : ∑ d in part2, (d.1 + d.2) = 21)
  (sum3 : ∑ d in part3, (d.1 + d.2) = 21)
  (sum4 : ∑ d in part4, (d.1 + d.2) = 21)

/-- The main theorem: there exists a partition of the standard double-six set where each group's sum is 21 points. -/
theorem exists_partition_with_equal_sum : ∃ p : partition standardDoubleSixSet, 
  ∑ d in p.part1, (d.1 + d.2) = 21 ∧
  ∑ d in p.part2, (d.1 + d.2) = 21 ∧
  ∑ d in p.part3, (d.1 + d.2) = 21 ∧
  ∑ d in p.part4, (d.1 + d.2) = 21 :=
sorry

end exists_partition_with_equal_sum_l657_657413


namespace find_two_smallest_naturals_l657_657713

def is_irreducible_fraction (k : ℕ) (n : ℕ) : Prop :=
  Nat.gcd k (n + 2) = 1

def fractions_irreducible_for_all (n : ℕ) : Prop :=
  ∀ k, 68 ≤ k ∧ k ≤ 133 → is_irreducible_fraction k n

theorem find_two_smallest_naturals (n1 n2 : ℕ) (h1 : n1 = 65) (h2 : n2 = 135) :
  fractions_irreducible_for_all n1 ∧ fractions_irreducible_for_all n2 :=
begin
  sorry
end

#eval find_two_smallest_naturals 65 135 rfl rfl  -- This should validate that the statement is correct

end find_two_smallest_naturals_l657_657713


namespace angle_at_apex_of_cone_on_table_with_spheres_l657_657608

theorem angle_at_apex_of_cone_on_table_with_spheres :
  let r₁ := 2
  let r₂ := 2
  let r₃ := 1
  let angle := 2 * Real.arctan ((Real.sqrt 5 - 2) / 3)
  in ∀ (C : ℝ × ℝ) 
       (O₁ O₂ O₃ : ℝ × ℝ) 
       (A₁ A₂ A₃ : ℝ × ℝ),
     (dist O₁ O₂) = (r₁ + r₂) ∧ (dist O₂ O₃) = (r₂ + r₃) ∧ (dist O₃ O₁) = (r₃ + r₁) ∧
     (dist C O₁) = (dist C O₂) ∧
     (dist (C, A₃) = r₃) ∧ 
     (angle = 2 * Real.arctan ((Real.sqrt 5 - 2) / 3)) :=
sorry

end angle_at_apex_of_cone_on_table_with_spheres_l657_657608


namespace arithmetic_sequence_iff_lambda_zero_l657_657730

theorem arithmetic_sequence_iff_lambda_zero (S : ℕ → ℤ) (a : ℕ → ℤ) (λ : ℤ) :
  (∀ n, S n = n^2 + 2*n + λ) → 
  (∀ n ≥ 2, a n = S n - S (n - 1)) →
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) ↔ λ = 0 :=
by
  sorry

end arithmetic_sequence_iff_lambda_zero_l657_657730


namespace closest_point_on_plane_l657_657709

noncomputable def closest_point_to_plane (A : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (d : ℝ) : ℝ × ℝ × ℝ :=
  let t := ((n.1 * A.1 + n.2 * A.2 + n.3 * A.3 - d) / (n.1 ^ 2 + n.2 ^ 2 + n.3 ^ 2)) in
  (A.1 - n.1 * t, A.2 - n.2 * t, A.3 - n.3 * t)

theorem closest_point_on_plane (P A : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (d : ℝ)
  (h1 : n = (5, -3, 4))
  (h2 : A = (2, 1, 4))
  (h3 : d = 40)
  (h4 : P = (37 / 10, 49 / 50, 134 / 25))
  : closest_point_to_plane A n d = P :=
sorry

end closest_point_on_plane_l657_657709


namespace complex_number_fourth_quadrant_l657_657595

theorem complex_number_fourth_quadrant (m : ℝ) (h1 : 2/3 < m) (h2 : m < 1) : 
  (3 * m - 2) > 0 ∧ (m - 1) < 0 := 
by 
  sorry

end complex_number_fourth_quadrant_l657_657595


namespace product_of_four_consecutive_integers_l657_657822

theorem product_of_four_consecutive_integers (n : ℤ) : ∃ k : ℤ, k^2 = (n-1) * n * (n+1) * (n+2) + 1 :=
by
  sorry

end product_of_four_consecutive_integers_l657_657822


namespace correct_statements_about_f_l657_657396

def f (x : ℝ) : ℝ :=
  5 * Real.sin (2 * x - Real.pi / 4)

theorem correct_statements_about_f :
  ¬ (∀ (x : ℝ), g (x) = f (x + Real.pi / 4)) ∧
  ¬ (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi) Real.pi → x ∈ Set.Icc (3 * Real.pi / 8) (7 * Real.pi / 8) → ∀ y ∈ Set.Icc (f (x - y)) (f (x + y)), (f y) > (f y)) ∧
  (∀ (x : ℝ), f x = f (3 * Real.pi / 8 - x)) ∧
  (∀ (x : ℝ), f (x + 5 * Real.pi / 8) = -f x) :=
sorry

end correct_statements_about_f_l657_657396


namespace monomial_2023_eq_l657_657068

def monomial (n : ℕ) : ℤ × ℕ :=
  ((-1)^(n+1) * (2*n - 1), n)

theorem monomial_2023_eq : monomial 2023 = (4045, 2023) :=
by
  sorry

end monomial_2023_eq_l657_657068


namespace intersection_proof_l657_657032

noncomputable def M : Set ℕ := {1, 3, 5, 7, 9}

noncomputable def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_proof : M ∩ (N ∩ Set.univ) = {5, 7, 9} :=
by sorry

end intersection_proof_l657_657032


namespace number_of_nonzero_terms_l657_657112

theorem number_of_nonzero_terms :
  let count_terms := 
    ∑ a in (finset.range 0 2009).filter (λ n, even n), 
    2009 - a
  count_terms = 1010025 :=
by sorry

end number_of_nonzero_terms_l657_657112


namespace negative_values_of_x_l657_657303

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l657_657303


namespace product_of_four_consecutive_integers_is_not_square_l657_657094

theorem product_of_four_consecutive_integers_is_not_square (n : ℤ) : 
  ¬ ∃ k : ℤ, k * k = (n-1)*n*(n+1)*(n+2) :=
sorry

end product_of_four_consecutive_integers_is_not_square_l657_657094


namespace tom_marbles_l657_657005

def jason_marbles := 44
def marbles_difference := 20

theorem tom_marbles : (jason_marbles - marbles_difference = 24) :=
by
  sorry

end tom_marbles_l657_657005


namespace negative_values_count_l657_657316

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l657_657316


namespace tangential_quadrilateral_perpendicular_diagonals_l657_657716

-- Define what it means for a quadrilateral to be tangential
def is_tangential_quadrilateral (a b c d : ℝ) : Prop :=
  a + c = b + d

-- Define what it means for a quadrilateral to be a kite
def is_kite (a b c d : ℝ) : Prop :=
  a = b ∧ c = d

-- Define what it means for the diagonals of a quadrilateral to be perpendicular
def diagonals_perpendicular (a b c d : ℝ) : Prop :=
  sorry -- Actual geometric definition needs to be elaborated

-- Main statement to prove
theorem tangential_quadrilateral_perpendicular_diagonals (a b c d : ℝ) :
  is_tangential_quadrilateral a b c d → 
  (diagonals_perpendicular a b c d ↔ is_kite a b c d) := 
sorry

end tangential_quadrilateral_perpendicular_diagonals_l657_657716


namespace count_negative_values_correct_l657_657285

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l657_657285


namespace rational_x0_if_eventually_zero_l657_657061

noncomputable def eventually_zero_sequence (x_0 : ℝ) (p : ℕ → ℕ) : ℕ → ℝ
| 0     := x_0
| (k+1) := if eventually_zero_sequence k = 0 then 0 else (p (k+1) / eventually_zero_sequence k) % 1

theorem rational_x0_if_eventually_zero (x_0 : ℝ) (p : ℕ → ℕ) :
  (∀ k, ∃ N, ∀ n ≥ N, eventually_zero_sequence x_0 p n = 0) ↔ ∃ m n : ℕ, n ≠ 0 ∧ x_0 = m / n :=
sorry

end rational_x0_if_eventually_zero_l657_657061


namespace total_trees_planted_l657_657536

theorem total_trees_planted (apple_trees orange_trees : ℕ) (h₁ : apple_trees = 47) (h₂ : orange_trees = 27) : apple_trees + orange_trees = 74 := 
by
  -- We skip the proof step
  sorry

end total_trees_planted_l657_657536


namespace exists_non_parallel_diagonal_l657_657882

theorem exists_non_parallel_diagonal {n : ℕ} (h : n > 0) :
  ∀ (P : list (ℝ × ℝ)), (convex_polygon P) ∧ (length P = 2 * n) →
    ∃ (d : diagonal P), ¬ parallel_to_any_side P d :=
by 
  sorry

end exists_non_parallel_diagonal_l657_657882


namespace omega_range_monotonically_decreasing_l657_657719

-- Definition of the function f(x)
def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 4)

-- The theorem to be proved
theorem omega_range_monotonically_decreasing (ω : ℝ) :
  ω > 0 →
  (∀ x, π / 2 < x ∧ x < π → f ω x ≤ f ω (x + ε))) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 :=
sorry

end omega_range_monotonically_decreasing_l657_657719


namespace arithmetic_expression_l657_657982

theorem arithmetic_expression :
  (((15 - 2) + (4 / (1 / 2)) - (6 * 8)) * (100 - 24)) / 38 = -54 := by
  sorry

end arithmetic_expression_l657_657982


namespace relationship_teachers_students_l657_657436

variables (m n k l : ℕ)

theorem relationship_teachers_students :
  ∀ (m n k l : ℕ) (h1: m > 0) (h2: n > 1) (h3: k > 0) (h4: l > 0),
  (∑ i in finset.range m, ∑ j in finset.range k, i ≠ j) * 1/2  =  
  (∑ i in finset.range n, ∑ j in finset.range k, i ≠ j) * 1/2 :=
  
  m * k * (k - 1) = n * (n - 1) * l := 
begin
  sorry
end

end relationship_teachers_students_l657_657436


namespace intersection_correct_l657_657021

-- Define the sets M and N
def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x | 2 * x > 7}

-- Define the expected intersection result
def expected_intersection : Set ℝ := {5, 7, 9}

-- State the theorem
theorem intersection_correct : ∀ x, x ∈ M ∩ N ↔ x ∈ expected_intersection :=
by
  sorry

end intersection_correct_l657_657021


namespace polygon_area_144_l657_657809

-- Given definitions
def polygon (n : ℕ) : Prop := -- definition to capture n squares arrangement
  n = 36

def is_perpendicular (sides : ℕ) : Prop := -- every pair of adjacent sides is perpendicular
  sides = 4

def all_sides_congruent (length : ℕ) : Prop := -- all sides have the same length
  true

def total_perimeter (perimeter : ℕ) : Prop := -- total perimeter of the polygon
  perimeter = 72

-- The side length s leading to polygon's perimeter
def side_length (s perimeter : ℕ) : Prop :=
  perimeter = 36 * s / 2 

-- Prove the area of polygon is 144
theorem polygon_area_144 (n sides length perimeter s: ℕ) 
    (h1 : polygon n) 
    (h2 : is_perpendicular sides) 
    (h3 : all_sides_congruent length) 
    (h4 : total_perimeter perimeter) 
    (h5 : side_length s perimeter) : 
    n * s * s = 144 := 
sorry

end polygon_area_144_l657_657809


namespace farthest_vertex_coordinates_l657_657101

-- Definitions based on the problem conditions
def center := (5 : ℝ, -3 : ℝ)
def area := (9 : ℝ)
def dilation_center := (0 : ℝ, 0 : ℝ)
def scale_factor := (3 : ℝ)

-- Problem statement in Lean 4
theorem farthest_vertex_coordinates :
  let side_length := Real.sqrt area
  let half_side := side_length / 2
  let original_vertices := [(center.1 - half_side, center.2 - half_side),
                            (center.1 - half_side, center.2 + half_side),
                            (center.1 + half_side, center.2 + half_side),
                            (center.1 + half_side, center.2 - half_side)]
  let dilated_vertices := original_vertices.map (λ v, (v.1 * scale_factor, v.2 * scale_factor))
  ∃ v ∈ dilated_vertices, v = (19.5, -13.5) ∧ ∀ u ∈ dilated_vertices, (Real.sqrt (v.1 ^ 2 + v.2 ^ 2)) ≥ (Real.sqrt (u.1 ^ 2 + u.2 ^ 2)) := by
  sorry

end farthest_vertex_coordinates_l657_657101


namespace find_x_for_y_equals_six_l657_657606

variable (x y k : ℚ)

-- Conditions
def varies_inversely_as_square := x = k / y^2
def initial_condition := (y = 3 ∧ x = 1)

-- Problem Statement
theorem find_x_for_y_equals_six (h₁ : varies_inversely_as_square x y k) (h₂ : initial_condition x y) :
  ∃ k, (k = 9 ∧ x = k / 6^2 ∧ x = 1 / 4) :=
sorry

end find_x_for_y_equals_six_l657_657606


namespace ratio_add_b_l657_657415

theorem ratio_add_b (a b : ℚ) (h : a / b = 3 / 5) : (a + b) / b = 8 / 5 :=
by
  sorry

end ratio_add_b_l657_657415


namespace rectangle_area_l657_657213

theorem rectangle_area (ABCD : Type) [rectangle ABCD] 
  (A B C D E F : ABCD)
  (isosceles_right_triangle : ∀ P Q R : ABCD, P ≠ Q → Q ≠ R → R ≠ P → bool)
  (h₁ : isosceles_right_triangle A E D = true)
  (h₂ : isosceles_right_triangle B F C = true)
  (h_eq : EF = AD)
  (h_AD : AD = 2) :
  area ABCD = 8 :=
sorry

end rectangle_area_l657_657213


namespace triangle_and_square_form_vertices_of_square_l657_657001

variables {A B C D E F G O1 O2 M N : Point}

-- Definitions representing geometric constructs
def is_triangle (A B C : Point) : Prop := 
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

def square_center (A B : Point) : Point :=
  midpoint A B

def midpoint (P Q : Point) : Point :=
  ⟨(P.1 + Q.1) / 2, (P.2 + Q.2) / 2⟩

def is_square (P Q R S : Point) : Prop := 
  distance P Q = distance Q R ∧ 
  distance Q R = distance R S ∧ 
  distance R S = distance S P ∧ 
  angle P Q R = 90 ∧ 
  angle Q R S = 90 ∧ 
  angle R S P = 90 ∧ 
  angle S P Q = 90

-- Abstract definitions for given problem
def conditions (A B C D E F G O1 O2 M N : Point) : Prop :=
  is_triangle A B C ∧
  square_center C A = O1 ∧
  square_center C B = O2 ∧
  M = midpoint A B ∧
  N = midpoint E G

-- Math proof problem definition in Lean 4
theorem triangle_and_square_form_vertices_of_square 
  (A B C D E F G O1 O2 M N : Point) :
  conditions A B C D E F G O1 O2 M N →
  is_square M N O1 O2 :=
by
  sorry

end triangle_and_square_form_vertices_of_square_l657_657001


namespace distinct_arrangements_B1B2A1N1A2N2A3_l657_657412

theorem distinct_arrangements_B1B2A1N1A2N2A3 :
  ∃ n : ℕ, n = 7! ∧ n = 5040 := 
by {
  use 7!, 
  simp
}

end distinct_arrangements_B1B2A1N1A2N2A3_l657_657412


namespace prices_are_correct_most_cost_effective_plan_l657_657552

-- Define the prices of rackets to prove they are correct
def price_of_rackets (x y : ℕ) : Prop :=
  20 * (x + y) = 1300 ∧ y = x - 15

-- Prove the prices of rackets
theorem prices_are_correct : ∃ (x y : ℕ), price_of_rackets x y ∧ x = 40 ∧ y = 25 :=
by
  -- Here we prove the correctness based on the conditions given
  sorry

-- Define the conditions for purchasing plans
def purchasing_plan (m n : ℕ) : Prop :=
  let x := 40 in let y := 25 in
  let discountedA := 0.8 * x in let discountedB := y - 4 in
  32 * m + 21 * n ≤ 1500 ∧ m + n = 50 ∧ m >= 38

-- Prove the most cost-effective purchasing plan
theorem most_cost_effective_plan : ∃ (m n : ℕ), purchasing_plan m n ∧ m = 38 ∧ n = 12 :=
by
  -- Here we prove the correctness based on the conditions given
  sorry

end prices_are_correct_most_cost_effective_plan_l657_657552


namespace negative_values_count_l657_657332

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l657_657332


namespace trig_identity_cosine_powers_l657_657171

theorem trig_identity_cosine_powers :
  12 * (Real.cos (Real.pi / 8)) ^ 4 + 
  (Real.cos (3 * Real.pi / 8)) ^ 4 + 
  (Real.cos (5 * Real.pi / 8)) ^ 4 + 
  (Real.cos (7 * Real.pi / 8)) ^ 4 = 
  3 / 2 := 
  sorry

end trig_identity_cosine_powers_l657_657171


namespace problem_conditions_imply_statements_l657_657900

theorem problem_conditions_imply_statements
  (a b c : ℝ)
  (h₀ : a < 0)
  (h₁ : b > 0)
  (h₂ : a < b)
  (h₃ : b < c) :
  ab < bc ∧ ac < bc ∧ ab < ac ∧ a + b < b + c ∧ c/a < c/b :=
begin
  sorry
end

end problem_conditions_imply_statements_l657_657900


namespace verify_drawn_numbers_when_x_is_24_possible_values_of_x_l657_657435

-- Population size and group division
def population_size := 1000
def number_of_groups := 10
def group_size := population_size / number_of_groups

-- Systematic sampling function
def systematic_sample (x : ℕ) (k : ℕ) : ℕ :=
  (x + 33 * k) % 1000

-- Prove the drawn 10 numbers when x = 24
theorem verify_drawn_numbers_when_x_is_24 :
  (∃ drawn_numbers, drawn_numbers = [24, 157, 290, 323, 456, 589, 622, 755, 888, 921]) :=
  sorry

-- Prove possible values of x given last two digits equal to 87
theorem possible_values_of_x (k : ℕ) (h : k < number_of_groups) :
  (∃ x_values, x_values = [87, 54, 21, 88, 55, 22, 89, 56, 23, 90]) :=
  sorry

end verify_drawn_numbers_when_x_is_24_possible_values_of_x_l657_657435


namespace slope_intercept_parallel_line_l657_657274

def is_parallel (m1 m2 : ℝ) : Prop :=
  m1 = m2

theorem slope_intercept_parallel_line (A : ℝ × ℝ) (hA₁ : A.1 = 3) (hA₂ : A.2 = 2) 
  (m : ℝ) (h_parallel : is_parallel m (-4)) : ∃ b : ℝ, ∀ x y : ℝ, y = -4 * x + b :=
by
  use 14
  intro x y
  sorry

end slope_intercept_parallel_line_l657_657274


namespace combo_sum_eq_l657_657614

open Function

variable {n : ℕ}

theorem combo_sum_eq :
  1 - ∑ m in Finset.range (n+1), (-1)^m * (1/(m+1)) * Nat.choose n m = 1/(n+1) :=
by
  sorry

end combo_sum_eq_l657_657614


namespace cofactor_value_l657_657247

variable (θ : ℝ)

def matrix3x3 : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2019, 4, 9], ![π, sin θ, cos θ], ![-5, sin (π / 2), cos (π / 3)]]

noncomputable def algebraic_cofactor (i j : Fin 3) :=
  (-1)^(i + j) * Matrix.det (Matrix.minor matrix3x3 (Fin.elim i) (Fin.elim j))

theorem cofactor_value : algebraic_cofactor θ 1 0 = 7 := by
  sorry

end cofactor_value_l657_657247


namespace salary_for_january_l657_657910

-- Define the months' salaries
variables (J F M A May : ℕ)

-- Define the conditions
def condition1 : Prop := J + F + M + A = 32000
def condition2 : Prop := F + M + A + May = 34000
def condition3 : Prop := May = 6500

-- Define the theorem to prove the salary for January under the given conditions
theorem salary_for_january : condition1 ∧ condition2 ∧ condition3 → J = 4500 :=
by
  intro h, cases h with hc1 h', cases h' with hc2 hc3,
  rw [hc3] at hc2,
  have hFMA : F + M + A = 27500,
  { linarith },
  have hJ : J = 32000 - 27500,
  { linarith },
  exact hJ

end salary_for_january_l657_657910


namespace cookies_left_l657_657695

theorem cookies_left (total_cookies : ℕ) (total_neighbors : ℕ) (cookies_per_neighbor : ℕ) (sarah_cookies : ℕ)
  (h1 : total_cookies = 150)
  (h2 : total_neighbors = 15)
  (h3 : cookies_per_neighbor = 10)
  (h4 : sarah_cookies = 12) :
  total_cookies - ((total_neighbors - 1) * cookies_per_neighbor + sarah_cookies) = 8 :=
by
  simp [h1, h2, h3, h4]
  sorry

end cookies_left_l657_657695


namespace sum_of_distances_l657_657644

section SquareDistances

variable (side : ℝ) (A B C D M N O P : EuclideanGeometry.Point)
variables (dist : EuclideanGeometry.Point → EuclideanGeometry.Point → ℝ)

def bottom_left_vertex (A : EuclideanGeometry.Point) : Prop :=
  A = (⟨0, 0⟩ : EuclideanGeometry.Point)

def midpoints_conditions (M N O P : EuclideanGeometry.Point) : Prop :=
  M = (⟨side / 2, 0⟩ : EuclideanGeometry.Point) ∧
  N = (⟨side, side / 2⟩ : EuclideanGeometry.Point) ∧
  O = (⟨side / 2, side⟩ : EuclideanGeometry.Point) ∧
  P = (⟨0, side / 2⟩ : EuclideanGeometry.Point)

theorem sum_of_distances 
  (h1 : side = 4)
  (h2 : bottom_left_vertex A)
  (h3 : midpoints_conditions M N O P)
  : 
  dist A M + dist A N + dist A O + dist A P = 4 + 4 * Real.sqrt 5 := 
sorry

end SquareDistances

end sum_of_distances_l657_657644


namespace garden_fencing_needed_l657_657193

/-- Given a rectangular garden where the length is 300 yards and the length is twice the width,
prove that the total amount of fencing needed to enclose the garden is 900 yards. -/
theorem garden_fencing_needed :
  ∃ (W L P : ℝ), L = 300 ∧ L = 2 * W ∧ P = 2 * (L + W) ∧ P = 900 :=
by
  sorry

end garden_fencing_needed_l657_657193


namespace unique_8_tuple_real_l657_657707

theorem unique_8_tuple_real (x : Fin 8 → ℝ) :
  (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + x 7^2 = 1 / 8 →
  ∃! (y : Fin 8 → ℝ), (1 - y 0)^2 + (y 0 - y 1)^2 + (y 1 - y 2)^2 + (y 2 - y 3)^2 + (y 3 - y 4)^2 + (y 4 - y 5)^2 + (y 5 - y 6)^2 + (y 6 - y 7)^2 + y 7^2 = 1 / 8 :=
by
  sorry

end unique_8_tuple_real_l657_657707


namespace no_correct_propositions_l657_657833

variables (a b : Type) [line a] [line b]
variables (α : Type) [plane α]
variables (a1 b1 : Type) [projection a α a1] [projection b α b1]

/-- Prove that none of the given propositions are correct -/
theorem no_correct_propositions 
  (h1 : ¬ (a ⊥ b ⟶ a1 ⊥ b1))
  (h2 : ¬ (a1 ⊥ b1 ⟶ a ⊥ b))
  (h3 : ¬ (a ∥ b ⟶ a1 ∥ b1))
  (h4 : ¬ (a1 ∥ b1 ⟶ a ∥ b)) :
  0 = 0 :=
by {
    sorry
}

end no_correct_propositions_l657_657833


namespace remainder_when_divided_by_r_minus_2_l657_657273

-- Define polynomial p(r)
def p (r : ℝ) : ℝ := r ^ 11 - 3

-- The theorem stating the problem
theorem remainder_when_divided_by_r_minus_2 : p 2 = 2045 := by
  sorry

end remainder_when_divided_by_r_minus_2_l657_657273


namespace intersection_M_N_l657_657047

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657047


namespace magnitude_of_a_l657_657859

noncomputable def e1 : Vector ℝ 3 := sorry
noncomputable def e2 : Vector ℝ 3 := sorry

axiom magnitude_e1 : ∥e1∥ = 1
axiom magnitude_e2 : ∥e2∥ = 1
axiom angle_120 : inner_product e1 e2 = -1/2

def a := 2 • e1 - e2

theorem magnitude_of_a : ∥a∥ = Real.sqrt 7 := by
  sorry

end magnitude_of_a_l657_657859


namespace bus_carrying_capacity_l657_657619

variables (C : ℝ)

theorem bus_carrying_capacity (h1 : ∀ x : ℝ, x = (3 / 5) * C) 
                              (h2 : ∀ y : ℝ, y = 50 - 18)
                              (h3 : ∀ z : ℝ, x + y = C) : C = 80 :=
by
  sorry

end bus_carrying_capacity_l657_657619


namespace total_amount_after_four_years_l657_657219

theorem total_amount_after_four_years : 
  let annual_earnings := 3^5 - 3^4 + 3^3 - 3^2 + 3 in
  let annual_investment_return := 2^5 - 2^4 + 2^3 - 2^2 + 2 in
  let years := 4 in
  (annual_earnings + annual_investment_return) * years = 820 :=
by
  sorry

end total_amount_after_four_years_l657_657219


namespace profit_percentage_now_l657_657234

-- Lean statement for the given problem
theorem profit_percentage_now :
  ∀ (P : ℝ),
  (0.35 * P) + 65 = P →
  let new_cost := 50 in
  let new_profit := P - new_cost in
  (new_profit / P) * 100 = 50 :=
by
  intros P h1
  let new_cost := 50
  let new_profit := P - new_cost
  have h2 : P = 100 :=
    calc
      P = 0.35 * P + 65 : by exact h1
      ... = 65 + 0.35 * P : by ring
      ... : 0.65 * P = 65
      P = 65 / 0.65 : by simp
      P = 100 : by norm_num
  have h3 : new_profit = 50 :=
    by calc
      new_profit = P - 50 : by rfl
      ... = 50 : by rw h2
  have h4 : (new_profit / P) * 100 = 50 :=
    by calc
      (new_profit / P) * 100 = (50 / P) * 100 : by rw h3
      ... = (50 / 100) * 100 : by rw h2
      ... = 0.5 * 100 : by norm_num
      ... = 50 : by norm_num
  exact h4

end profit_percentage_now_l657_657234


namespace arithmetic_sequence_middle_term_l657_657574

-- Definitions based on conditions
def first_term : ℕ := 9
def third_term : ℕ := 81
def arithmetic_mean (a b : ℕ) : ℕ := (a + b) / 2

-- Theorem statement
theorem arithmetic_sequence_middle_term : 
  let x := arithmetic_mean first_term third_term
  in x = 45 := by
  let x := arithmetic_mean first_term third_term
  -- Proof is omitted
  sorry

end arithmetic_sequence_middle_term_l657_657574


namespace annual_percentage_increase_of_doubling_volume_l657_657538

theorem annual_percentage_increase_of_doubling_volume (x y : ℝ) (h : x * (1 + y / 100) ^ 2 = 2 * x) :
  y ≈ 41.4 := by
  sorry

end annual_percentage_increase_of_doubling_volume_l657_657538


namespace max_phone_calls_without_triangles_l657_657515

theorem max_phone_calls_without_triangles (m : ℕ) (h_odd : Odd m) (h_m_ge1: 1 ≤ m) :
  ∃ (n : ℕ), n ≤ 101 :=
by
  -- Number of people
  let num_people : ℕ := 21
  -- No group of three mutually communicate
  let no_group_of_three := ∀ a b c : ℕ, a ≠ b → a ≠ c → b ≠ c 
    → a ≠ c 
    → b ≠ c 
    → c ≠ a 
    → ¬ (edge a b ∧ edge b c ∧ edge a c)
  -- There exists an odd number cycle of length m between chosen people
  let cyclic_communications : ∃ l : List ℕ, l.length = m ∧ ∀ i, edge l[i] l[(i+1)%m]
  sorry

end max_phone_calls_without_triangles_l657_657515


namespace perimeter_of_triangle_l657_657747

-- Definitions based on the conditions
def is_foci (F₁ F₂ : ℝ × ℝ) (a b : ℝ) : Prop :=
  F₁ = (sqrt (a^2 - b^2), 0) ∧ F₂ = (-sqrt (a^2 - b^2), 0)

def is_ellipse (x y a b : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

def is_chord_through_foci (A B F₁ : ℝ × ℝ) : Prop :=
  (A.1 - F₁.1) * (B.2 - F₁.2) = (B.1 - F₁.1) * (A.2 - F₁.2)

-- Theorem statement
theorem perimeter_of_triangle (a b : ℝ) (A B F₁ F₂ : ℝ × ℝ)
  (h_ellipse : ∀ x y, is_ellipse x y a b)  
  (h_foci : is_foci F₁ F₂ a b)  
  (h_chord : is_chord_through_foci A B F₁)  
  (h_ineq : a > b ∧ b > 0) : 
  A.2^2 + B.2^2 + F₂.2^2 = 4 * a :=
sorry

end perimeter_of_triangle_l657_657747


namespace existsThreeRedSquaresBlock_l657_657099

-- Definition of a chessboard with pieces colored
def Chessboard (n : Nat) := {i : Fin n // i.1 < n}

-- Predicate to check if a given \(2 \times 2\) block has at least three red squares
def hasThreeRedSquares (redSquares : List (Chessboard 9 × Chessboard 9)) 
  (block : Chessboard 9 × Chessboard 9 × Chessboard 9 × Chessboard 9) : Prop :=
  ((block.1 ∈ redSquares).toNat + 
   (block.2.1 ∈ redSquares).toNat + 
   (block.2.2.1 ∈ redSquares).toNat + 
   (block.2.2.2 ∈ redSquares).toNat) ≥ 3

-- Main theorem statement
theorem existsThreeRedSquaresBlock : 
  ∃ (redSquares : List (Chessboard 9 × Chessboard 9)) (h : redSquares.length = 46), 
    ∃ (block : (Chessboard 9 × Chessboard 9) × (Chessboard 9 × Chessboard 9)), hasThreeRedSquares redSquares block :=
by
  sorry

end existsThreeRedSquaresBlock_l657_657099


namespace tips_fraction_august_l657_657682

theorem tips_fraction_august (A : ℝ) :
  let total_tips_for_6_months := 6 * A,
      tips_for_august := 4 * A,
      total_tips := total_tips_for_6_months + tips_for_august
  in tips_for_august / total_tips = 2 / 5 :=
by
  let total_tips_for_6_months := 6 * A
  let tips_for_august := 4 * A
  let total_tips := total_tips_for_6_months + tips_for_august
  calc
    tips_for_august / total_tips
      = (4 * A) / (6 * A + 4 * A) : by rfl
    ... = (4 * A) / (10 * A) : by rfl
    ... = 4 / 10 : by { simp }
    ... = 2 / 5 : by { norm_num }

end tips_fraction_august_l657_657682


namespace intersection_M_N_l657_657039

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := { x | 2 * x > 7 }

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657039


namespace find_PB_correct_l657_657811

variables {A B C D P : Type} [inner_product_space ℝ A]

/-- Quadrilateral Setup -/
structure Quadrilateral (A B C D : A) :=
(cd_perpendicular_ab : ⟪C - D, A - B⟫ = 0)
(bc_perpendicular_ad : ⟪B - C, A - D⟫ = 0)
(cd_length : ∥C - D∥ = 39)
(bc_length : ∥B - C∥ = 84)
(perpendicular_through_P : ∃ P : A, affection P C (A - B) (B - D) ∧ ∥A - P∥ = 24)

noncomputable def find_PB (Q : Quadrilateral A B C D) : ℝ :=
  let CD := Q.cd_length,
      AP := 24,
      PB := 144
  in PB

theorem find_PB_correct : ∀ (A B C D : A) (Q : Quadrilateral A B C D), find_PB Q = 144 :=
by { sorry }

end find_PB_correct_l657_657811


namespace floss_per_student_l657_657694

theorem floss_per_student
  (students : ℕ)
  (yards_per_packet : ℕ)
  (floss_left_over : ℕ)
  (total_packets : ℕ)
  (total_floss : ℕ)
  (total_floss_bought : ℕ)
  (smallest_multiple_of_35 : ℕ)
  (each_student_needs : ℕ)
  (hs1 : students = 20)
  (hs2 : yards_per_packet = 35)
  (hs3 : floss_left_over = 5)
  (hs4 : total_floss = total_packets * yards_per_packet)
  (hs5 : total_floss_bought = total_floss + floss_left_over)
  (hs6 : total_floss_bought % 35 = 0)
  (hs7 : smallest_multiple_of_35 > total_packets * yards_per_packet - floss_left_over)
  (hs8 : 20 * each_student_needs + 5 = smallest_multiple_of_35)
  : each_student_needs = 5 :=
by
  sorry

end floss_per_student_l657_657694


namespace percentage_increase_each_job_l657_657006

-- Definitions of original and new amounts for each job as given conditions
def original_first_job : ℝ := 65
def new_first_job : ℝ := 70

def original_second_job : ℝ := 240
def new_second_job : ℝ := 315

def original_third_job : ℝ := 800
def new_third_job : ℝ := 880

-- Proof problem statement
theorem percentage_increase_each_job :
  (new_first_job - original_first_job) / original_first_job * 100 = 7.69 ∧
  (new_second_job - original_second_job) / original_second_job * 100 = 31.25 ∧
  (new_third_job - original_third_job) / original_third_job * 100 = 10 := by
  sorry

end percentage_increase_each_job_l657_657006


namespace x_pow_4_plus_inv_x_pow_4_l657_657493

theorem x_pow_4_plus_inv_x_pow_4 (x : ℝ) (h : x^2 - 15 * x + 1 = 0) : x^4 + (1 / x^4) = 49727 :=
by
  sorry

end x_pow_4_plus_inv_x_pow_4_l657_657493


namespace total_candies_l657_657089

axiom Adam_candies : ℕ := 6
axiom James_candies : ℕ := 3 * Adam_candies
axiom Rubert_candies : ℕ := 4 * James_candies
axiom Lisa_candies : ℕ := 2 * Rubert_candies - 5
axiom Chris_candies : ℕ := (Lisa_candies / 2) + 7
axiom Max_candies : ℕ := Rubert_candies + Chris_candies + 2
axiom Emily_candies : ℕ := 3 * Chris_candies - (Max_candies - Lisa_candies)

theorem total_candies : Adam_candies + James_candies + Rubert_candies + Lisa_candies +
    Chris_candies + Max_candies + Emily_candies = 678 :=
  sorry

end total_candies_l657_657089


namespace exist_complex_in_second_quadrant_with_magnitude_two_l657_657422

theorem exist_complex_in_second_quadrant_with_magnitude_two :
  ∃ z : ℂ, (∃ a b : ℝ, z = Complex.mk a b ∧ a < 0 ∧ b > 0 ∧ a^2 + b^2 = 4) ∧ z = -1 + Complex.I * Real.sqrt 3 :=
by
  sorry

end exist_complex_in_second_quadrant_with_magnitude_two_l657_657422


namespace ways_to_place_people_into_groups_l657_657439

theorem ways_to_place_people_into_groups :
  let men := 4
  let women := 5
  ∃ (groups : Nat), groups = 2 ∧
  ∀ (g : Nat → (Fin 3 → (Bool → Nat → Nat))),
    (∀ i, i < group_counts → ∃ m w, g i m w < people ∧ g i m (1 - w) < people ∧ m + 1 - w + (1 - m) + w = 3) →
    let groups : List (List (Fin 2)) := [
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)],
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)]
    ] in
    g.mk 1 dec_trivial * g.mk 2 dec_trivial = 360 :=
sorry

end ways_to_place_people_into_groups_l657_657439


namespace decimal_rep_150th_l657_657578

theorem decimal_rep_150th (num : ℚ := 13/17) (dec_repr : String := "7647058823529411") (length_seq : Nat := 16) :
  (dec_repr[(150 % length_seq) - 1] = '5') :=
by
  sorry

end decimal_rep_150th_l657_657578


namespace hitting_first_third_fifth_probability_hitting_exactly_three_out_of_five_probability_l657_657632

noncomputable def probability_hitting_first_third_fifth (P : ℚ) : ℚ :=
  P * (1 - P) * P * (1 - P) * P

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

noncomputable def probability_hitting_exactly_three_out_of_five (P : ℚ) : ℚ :=
  binomial_coefficient 5 3 * P^3 * (1 - P)^2

theorem hitting_first_third_fifth_probability :
  probability_hitting_first_third_fifth (3/5) = 108/3125 := by
  sorry

theorem hitting_exactly_three_out_of_five_probability :
  probability_hitting_exactly_three_out_of_five (3/5) = 216/625 := by
  sorry

end hitting_first_third_fifth_probability_hitting_exactly_three_out_of_five_probability_l657_657632


namespace find_a_l657_657124

theorem find_a (a b : ℤ) (p : ℕ) (hp : prime p) (h1 : p^2 + a * p + b = 0) (h2 : p^2 + b * p + 1100 = 0) :
  a = 274 ∨ a = 40 :=
sorry

end find_a_l657_657124


namespace probability_no_2x2_blue_square_l657_657253

theorem probability_no_2x2_blue_square : 
  let total_grids := 2^16
      total_2x2_blue_squares := 39400
      total_no_2x2_blue_squares := total_grids - total_2x2_blue_squares
      m := 409
      n := 1024
  in (total_no_2x2_blue_squares = 26136) ∧ 
     Rational.reduced_fraction 26136 65536 = (m, n) ∧ 
     Nat.gcd m n = 1 → m + n = 1433 :=
by
  sorry

end probability_no_2x2_blue_square_l657_657253


namespace negative_values_count_l657_657326

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l657_657326


namespace parallelogram_area_60_l657_657078

theorem parallelogram_area_60
  (α β : ℝ) (a b : ℝ)
  (h_angle : α = 150) 
  (h_adj_angle : β = 180 - α) 
  (h_len_1 : a = 10)
  (h_len_2 : b = 12) :
  ∃ (area : ℝ), area = 60 := 
by 
  use 60
  sorry

end parallelogram_area_60_l657_657078


namespace complex_transformation_result_l657_657967

theorem complex_transformation_result :
  let z := -1 - 2 * Complex.I 
  let rotation := (1 / 2 : ℂ) + (Complex.I * (Real.sqrt 3) / 2)
  let dilation := 2
  (z * (rotation * dilation)) = (2 * Real.sqrt 3 - 1 - (2 + Real.sqrt 3) * Complex.I) :=
by
  sorry

end complex_transformation_result_l657_657967


namespace smallest_m_plus_n_l657_657915

theorem smallest_m_plus_n (f : ℝ → ℝ) (m n : ℕ)
  (h_pos_m : 1 < m)
  (h_pos_n : 0 < n)
  (h_def_f : ∀ x, f x = Real.arcsin (Real.log nx / Real.log m))
  (h_dom_length : ∀ x, ⟦ Real.log (nx / m) ⟧ * ⟦ Real.log (m / nx) ⟧ = 1 / 2021)
  (h_min_m_n : m + n = (2022 * m * m - 2021) / m)
  : m + n = 86275 :=
by
  sorry

end smallest_m_plus_n_l657_657915


namespace profit_per_unit_max_profit_l657_657613

-- Part 1: Proving profits per unit
theorem profit_per_unit (a b : ℝ) (h1 : a + b = 600) (h2 : 3 * a + 2 * b = 1400) :
  a = 200 ∧ b = 400 :=
by
  sorry

-- Part 2: Proving maximum profit calculation
theorem max_profit (a b x y : ℝ) (h1 : x + y = 20) (h2 : y ≤ (2 / 3) * x) (h3 : a = 200) (h4 : b = 400) :
  let w := -200 * x + 8000 in
  (y = 20 - x ∧ w = 5600) :=
by
  sorry

end profit_per_unit_max_profit_l657_657613


namespace lily_has_26_dollars_left_for_coffee_l657_657508

-- Define the initial amount of money Lily has
def initialMoney : ℕ := 60

-- Define the costs of items
def celeryCost : ℕ := 5
def cerealCost : ℕ := 12 / 2
def breadCost : ℕ := 8
def milkCost : ℕ := 10 * 9 / 10
def potatoCostEach : ℕ := 1
def numberOfPotatoes : ℕ := 6
def totalPotatoCost : ℕ := potatoCostEach * numberOfPotatoes

-- Define the total amount spent on the items
def totalSpent : ℕ := celeryCost + cerealCost + breadCost + milkCost + totalPotatoCost

-- Define the amount left for coffee
def amountLeftForCoffee : ℕ := initialMoney - totalSpent

-- The theorem to prove
theorem lily_has_26_dollars_left_for_coffee :
  amountLeftForCoffee = 26 := by
  sorry

end lily_has_26_dollars_left_for_coffee_l657_657508


namespace area_of_triangle_FPQ_l657_657011

theorem area_of_triangle_FPQ : 
  (let L1 := (0, 0)  -- L1 as x = 0 (y-axis)
       L2 := (0, 0)  -- L2 as y = 0 (x-axis)
       F := (18, 25)  -- Point F at (18, 25)
       P := (13, 13)
       Q := (73, 73)
  in
  0.5 * abs (18 * 13 + 13 * 73 + 73 * 25 - (25 * 13 + 13 * 73 + 73 * 18)) = 210) := 
sorry

end area_of_triangle_FPQ_l657_657011


namespace correct_statements_l657_657714

def f (x : ℝ) : ℝ := 2 * cos x ^ 2 + 2 * sin x * cos x - 1

theorem correct_statements (x : ℝ) :
  (∀ a b, a < b → f a > f b) ∧ (∃ c, ∀ y, f (c - y) = f (c + y)) :=
sorry

end correct_statements_l657_657714


namespace lily_coffee_budget_l657_657505

variable (initial_amount celery_price cereal_original_price bread_price milk_original_price potato_price : ℕ)
variable (cereal_discount milk_discount number_of_potatoes : ℕ)

theorem lily_coffee_budget 
  (h_initial_amount : initial_amount = 60)
  (h_celery_price : celery_price = 5)
  (h_cereal_original_price : cereal_original_price = 12)
  (h_bread_price : bread_price = 8)
  (h_milk_original_price : milk_original_price = 10)
  (h_potato_price : potato_price = 1)
  (h_number_of_potatoes : number_of_potatoes = 6)
  (h_cereal_discount : cereal_discount = 50)
  (h_milk_discount : milk_discount = 10) :
  initial_amount - (celery_price + (cereal_original_price * cereal_discount / 100) + bread_price + (milk_original_price - (milk_original_price * milk_discount / 100)) + (potato_price * number_of_potatoes)) = 26 :=
by
  sorry

end lily_coffee_budget_l657_657505


namespace length_CD_l657_657773

-- Define the distances AB, AC and BC
def AB : ℝ := 13
def AC : ℝ := 5
def BC : ℝ := 12

-- Define the midpoint property of point D on AB
def D_midpoint : Prop := ∃ D : ℝ × ℝ, D = (6.5, 0)

-- Theorem statement that CD = 6.5
theorem length_CD (h_midpoint : D_midpoint) : (sqrt ((13 / 2) ^ 2 + AC ^ 2)) = 6.5 :=
by
  sorry

end length_CD_l657_657773


namespace negative_values_count_l657_657324

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l657_657324


namespace probability_of_picking_dumpling_with_egg_l657_657601

-- Definitions based on the conditions
def total_dumplings : ℕ := 10
def dumplings_with_eggs : ℕ := 3

-- The proof statement
theorem probability_of_picking_dumpling_with_egg :
  (dumplings_with_eggs : ℚ) / total_dumplings = 3 / 10 :=
by
  sorry

end probability_of_picking_dumpling_with_egg_l657_657601


namespace negative_values_count_l657_657329

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l657_657329


namespace quadratic_has_real_root_l657_657804

theorem quadratic_has_real_root (a : ℝ) : 
  ¬(∀ x : ℝ, x^2 + a * x + a - 1 ≠ 0) :=
sorry

end quadratic_has_real_root_l657_657804


namespace twenty_step_path_count_l657_657174

theorem twenty_step_path_count :
  let start := (-5, -5)
  let end := (5, 5)
  let condition := ∀ (x y : ℤ), (start.1 ≤ x ∧ x ≤ end.1 ∧ start.2 ≤ y ∧ y <= end.2) →
    ((x < -3 ∨ x > 3) ∨ (y < -3 ∨ y > 3))
  in
  let path_count : ℕ := 20
  in
  let paths : (ℤ × ℤ) → (ℤ × ℤ) → ℕ := sorry
  in
  paths start end = 4252 :=
sorry

end twenty_step_path_count_l657_657174


namespace lines_can_coincide_by_rotating_l657_657377

-- Define the two lines l1 and l2
def l1 (x : ℝ) (α : ℝ) : ℝ := x * Math.sin α
def l2 (x : ℝ) (c : ℝ) : ℝ := 2 * x + c

-- Prove that lines l1 and l2 can coincide by rotating around a certain point on l1
theorem lines_can_coincide_by_rotating (α c : ℝ) : 
  ∃ x y : ℝ, l1 x α = y ∧ ∃ θ : ℝ, y = l2 x (Math.sin θ) := 
sorry

end lines_can_coincide_by_rotating_l657_657377


namespace find_v_l657_657102

variables (a b c : ℝ)

def condition1 := (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -6
def condition2 := (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8

theorem find_v (h1 : condition1 a b c) (h2 : condition2 a b c) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = 17 / 2 :=
by
  sorry

end find_v_l657_657102


namespace find_positive_integer_m_l657_657382

-- Define the arithmetic sequence
def a_n (n : ℕ) : ℤ := 2 * n - 3

-- Define the sum of the first n terms of the sequence
def S_n (n : ℕ) : ℤ := n * n - 2 * n

-- Prove that m = 6 is the positive integer that satisfies S_m = 24
theorem find_positive_integer_m (m : ℕ) (hm : m = 6) : S_n m = 24 :=
by
  unfold S_n
  rw hm
  simp
  norm_num
  sorry

end find_positive_integer_m_l657_657382


namespace net_effect_on_sale_value_l657_657159

theorem net_effect_on_sale_value
(P Q : ℝ)
(h_new_price : ∃ P', P' = P - 0.22 * P)
(h_new_qty : ∃ Q', Q' = Q + 0.86 * Q) :
  let original_sale_value := P * Q
  let new_sale_value := (0.78 * P) * (1.86 * Q)
  let net_effect := ((new_sale_value / original_sale_value - 1) * 100 : ℝ)
  net_effect = 45.08 :=
by {
  sorry
}

end net_effect_on_sale_value_l657_657159


namespace train_length_is_50_meters_l657_657965

theorem train_length_is_50_meters
  (L : ℝ)
  (equal_length : ∀ (a b : ℝ), a = L ∧ b = L → a + b = 2 * L)
  (speed_faster_train : ℝ := 46) -- km/hr
  (speed_slower_train : ℝ := 36) -- km/hr
  (relative_speed : ℝ := speed_faster_train - speed_slower_train)
  (relative_speed_km_per_sec : ℝ := relative_speed / 3600) -- converting km/hr to km/sec
  (time : ℝ := 36) -- seconds
  (distance_covered : ℝ := 2 * L)
  (distance_eq : distance_covered = relative_speed_km_per_sec * time):
  L = 50 / 1000 :=
by 
  -- We will prove it as per the derived conditions
  sorry

end train_length_is_50_meters_l657_657965


namespace problem_1_problem_2_l657_657222

-- Problem 1
theorem problem_1 (a b : ℝ) :
  (a - b)^2 - (2a + b) * (b - 2a) = 5 * a^2 - 2 * a * b :=
sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ -2) :
  ((3 / (a + 1)) - a + 1) / ((a^2 + 4 * a + 4) / (a + 1)) = (2 - a) / (a + 2) :=
sorry

end problem_1_problem_2_l657_657222


namespace transform_expression_l657_657512

variable {a : ℝ}

theorem transform_expression (h : a - 1 < 0) : 
  (a - 1) * Real.sqrt (-1 / (a - 1)) = -Real.sqrt (1 - a) :=
by
  sorry

end transform_expression_l657_657512


namespace students_surveyed_l657_657156

-- Define the given conditions and question
theorem students_surveyed:
  (∃ S : ℕ, 
    (∃ F : ℕ,
      (0.25 * S = F ∧ F = 10 + 40) ∧ S = 4 * F
    ) ∧ S = 200
  ) :=
by
  sorry

end students_surveyed_l657_657156


namespace root_intervals_l657_657126

variable {a : ℝ}

def f (x : ℝ) : ℝ := x^2 + 2 * a * x - a

theorem root_intervals (h₀ : f 0 > 0) (h₁ : f 1 < 0) (h₂ : f 2 > 0) : a ∈ Set.Ioo (-4/3) (-1) := by
  sorry

end root_intervals_l657_657126


namespace decimal_rep_150th_l657_657579

theorem decimal_rep_150th (num : ℚ := 13/17) (dec_repr : String := "7647058823529411") (length_seq : Nat := 16) :
  (dec_repr[(150 % length_seq) - 1] = '5') :=
by
  sorry

end decimal_rep_150th_l657_657579


namespace number_of_solutions_eq_4_l657_657410

theorem number_of_solutions_eq_4 :
  let satisfies (n : ℤ) := (n^2 - n - 1)^(n + 2) = 1
  in (∃ n : ℤ, satisfies n ∧ n + 2 = 0 ∧ (n^2 - n - 1 ≠ 0)) ∧
     (∃ n : ℤ, satisfies n ∧ n^2 - n - 1 = 1) ∧
     (∃ n : ℤ, satisfies n ∧ n^2 - n - 1 = -1 ∧ (n + 2) % 2 = 0) →
     (finset.filter satisfies (finset.range 10)).card = 4 :=
by
  sorry

end number_of_solutions_eq_4_l657_657410


namespace Mindy_earns_more_l657_657871

theorem Mindy_earns_more (M k : ℝ) (h0 : 0 < M) 
  (h1 : Mork_tax_rate : 0.40) (h2 : Mindy_tax_rate : 0.25)
  (h3 : Combined_tax_rate : 0.28) 
  (h4 : Mindy_income : M * k = Mindy_income)
  (h5 : Combined_tax_eq : (0.40 * M + 0.25 * (M * k)) / (M + M * k) = 0.28) :
  k = 4 :=
by
  sorry

end Mindy_earns_more_l657_657871


namespace sum_of_angles_eq_l657_657401

variables (E L M I : Type) [Add E] [Sub E] [Add M] [Sub M] 
variables (angle : Type) [Add angle] [Eq angle] [Zero angle]

-- Assumptions as conditions for the problem
variable (EL [Add E] [Eq E] : E = EL)
variable (EI [Add E] [Eq E] : E = EI)
variable (LM [Add M] [Eq M] : M = LM)
variable (LME MEI LEM EMI MIE : angle)
variable (angle_sum_180 : LME + MEI = (180 : angle))
variable (segment_relation : EL = EI + LM)

-- Proof goal
theorem sum_of_angles_eq :
  LEM + EMI = MIE :=
sorry

end sum_of_angles_eq_l657_657401


namespace question1_question2_question3_l657_657726

namespace MathProofs

-- Statement for Question 1
theorem question1 (A : Set ℝ) (hA : A = {−1}) :
  (∀ x : ℝ, (−x) ≤ (−x+1) ) ∧ (∀ x : ℝ, ¬(2x ≤ 2x-2)) :=
sorry

-- Statement for Question 2
theorem question2 (A : Set ℝ) (f : ℝ → ℝ) (hA : A = Set.Ioo 0 1)
  (hf : ∀ x ∈ Set.Ici a, f(x) = x + 1/x) :
  a ∈ Set.Ici 1 :=
sorry

-- Statement for Question 3
theorem question3 (A : Set ℤ) (hA : A = {−2, m}) (hconstant : ∀ f : ℤ → ℤ , f is constant for property A) :
  m % 2 = 1 :=
sorry

end MathProofs

end question1_question2_question3_l657_657726


namespace total_lives_l657_657562

theorem total_lives (initial_friends : ℕ) (initial_lives_per_friend : ℕ) (additional_players : ℕ) (lives_per_new_player : ℕ) :
  initial_friends = 7 →
  initial_lives_per_friend = 7 →
  additional_players = 2 →
  lives_per_new_player = 7 →
  (initial_friends * initial_lives_per_friend + additional_players * lives_per_new_player) = 63 :=
by
  intros
  sorry

end total_lives_l657_657562


namespace polygon_diagonals_l657_657755

-- Definitions of the conditions
def sum_of_angles (n : ℕ) : ℝ := (n - 2) * 180 + 360

def num_diagonals (n : ℕ) : ℤ := n * (n - 3) / 2

-- Theorem statement
theorem polygon_diagonals (n : ℕ) (h : sum_of_angles n = 2160) : num_diagonals n = 54 :=
sorry

end polygon_diagonals_l657_657755


namespace concurrency_proof_l657_657062

variables {A B C D P : Type}
variables [convex_quadrilateral A B C D] [in_point A B C D P]
variables (anglePAD anglePBA angleDPB angleDPA angleCBP angleBAP angleBPC : ℝ)
variable (theta : ℝ)

-- Assuming the ratio equality given in the problem statement
axiom ratio_equality : 
  anglePAD / anglePBA = 1 ∧ 
  angleDPB / angleDPA = 2 ∧ 
  angleCBP / angleBAP = 3

theorem concurrency_proof :
  (concurrent (angle_bisector angleADP) (angle_bisector anglePCB) 
  (perpendicular_bisector (segment AB))) :=
sorry

end concurrency_proof_l657_657062


namespace find_angle_A_and_sin_C_l657_657810

theorem find_angle_A_and_sin_C (A B C a b c S : ℝ) (h₁ : sin (2 * A + π / 2) - cos (B + C) = -1 + 2 * cos A)
  (h₂ : A < π / 2) (h₃ : B < π / 2) (h₄ : C < π / 2) (h₅ : b = 3) (h₆ : S = 3 * sqrt 3)
  (h₇ : S = (1 / 2) * b * c * sin A) :
  A = π / 3 ∧ sin C = 2 * sqrt 39 / 13 :=
by 
  sorry

end find_angle_A_and_sin_C_l657_657810


namespace intersection_M_N_l657_657403

def M : Set ℝ := { x | -2 < x ∧ x < 3 }
def N : Set ℝ := { x | 2^(x + 1) ≥ 1 }

theorem intersection_M_N :
  M ∩ N = { x | -1 ≤ x ∧ x < 3 } := by
  sorry

end intersection_M_N_l657_657403


namespace negative_values_of_x_l657_657299

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l657_657299


namespace decimal_rep_150th_l657_657580

theorem decimal_rep_150th (num : ℚ := 13/17) (dec_repr : String := "7647058823529411") (length_seq : Nat := 16) :
  (dec_repr[(150 % length_seq) - 1] = '5') :=
by
  sorry

end decimal_rep_150th_l657_657580


namespace chessboard_tiling_condition_l657_657569

theorem chessboard_tiling_condition (m n : ℕ) (h1 : 8 ∣ m * n) (h2 : m ≠ 1) (h3 : n ≠ 1) : 
  ∃ mn : ℕ, 8 ∣ mn ∧ mn = m * n ∧ m ≠ 1 ∧ n ≠ 1 := 
begin
  sorry
end

end chessboard_tiling_condition_l657_657569


namespace decimal_representation_150th_digit_l657_657588

theorem decimal_representation_150th_digit : 
  ∃ s : ℕ → ℤ, 
    (∀ n, s n = (13/17 : ℚ)^n % 10) ∧ 
    let decSeq := [7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1].cycle in
    ∀ n, decSeq.get n = s n → decSeq.get 149 = 7 := 
begin
  sorry
end

end decimal_representation_150th_digit_l657_657588


namespace hot_dogs_leftover_l657_657796

theorem hot_dogs_leftover :
  36159782 % 6 = 2 :=
by
  sorry

end hot_dogs_leftover_l657_657796


namespace rectangle_dimensions_l657_657145

theorem rectangle_dimensions (l w : ℝ) : 
  (∃ x : ℝ, x = l - 3 ∧ x = w - 2 ∧ x^2 = (1 / 2) * l * w) → (l = 9 ∧ w = 8) :=
by
  sorry

end rectangle_dimensions_l657_657145


namespace money_raised_is_correct_l657_657629

noncomputable def total_money_raised : ℝ :=
  let ticket_sales := 120 * 2.50 + 80 * 4.50 + 40 * 8.00 + 15 * 14.00
  let donations := 3 * 20.00 + 2 * 55.00 + 75.00 + 95.00 + 150.00
  ticket_sales + donations

theorem money_raised_is_correct :
  total_money_raised = 1680 := by
  sorry

end money_raised_is_correct_l657_657629


namespace book_configurations_l657_657183

theorem book_configurations (n : ℕ) (h : n = 8) :
  ∃ (k : ℕ), k = 7 := by 
  have h1 : 1 ≤ n := by linarith
  have h2 : n ≤ 8 := by linarith
  use 7
  unfold has_dvd.dvd
  field_simp [*]
  sorry

end book_configurations_l657_657183


namespace triangle_area_from_line_and_axes_l657_657264

theorem triangle_area_from_line_and_axes :
  let line := λ (x y : ℝ), 2 * x + y - 16 = 0
  let x_intercept := (8 : ℝ, 0 : ℝ)
  let y_intercept := (0 : ℝ, 16 : ℝ)
  let base := 8
  let height := 16
  let area := (1 / 2) * base * height
  area = 64 :=
by
  sorry

end triangle_area_from_line_and_axes_l657_657264


namespace number_of_integer_solutions_l657_657409

theorem number_of_integer_solutions :
  ∃ (n : ℕ), n = 996 ∧ 
  (finset.card {v : vector ℤ 7 | 2 * v.nth 0 ^ 2 + v.nth 1 ^ 2 + v.nth 2 ^ 2 + 
                                    v.nth 3 ^ 2 + v.nth 4 ^ 2 + v.nth 5 ^ 2 + 
                                    v.nth 6 ^ 2 = 9}.to_set) = n :=
by {
  sorry
}

end number_of_integer_solutions_l657_657409


namespace ratio_problem_l657_657787

theorem ratio_problem (a b c d : ℝ) (h1 : a / b = 5) (h2 : b / c = 1 / 2) (h3 : c / d = 6) : 
  d / a = 1 / 15 :=
by sorry

end ratio_problem_l657_657787


namespace length_of_brick_l657_657712

-- Define the conditions
def width : ℕ := 4
def height : ℕ := 2
def surface_area : ℕ := 136

-- The proof problem translated to a Lean 4 statement
theorem length_of_brick : ∃ (length : ℕ), 2 * length * width + 2 * length * height + 2 * width * height = surface_area ∧ length = 10 := by
  sorry

end length_of_brick_l657_657712


namespace intersection_M_N_l657_657048

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657048


namespace incenter_of_triangle_CEF_l657_657489

open Real EuclideanGeometry

theorem incenter_of_triangle_CEF (Γ : Circle) (O : Point) (C D E F I A B : Point) : 
  (diameter BC) (center O Γ) (0 < angle AOB < 120) (midpoint_arc D AB Γ not_containing C) 
  (parallel DA OI Γ) (intersect I AC Γ) (perpendicular_bisector OA Γ) : 
  is_incenter I (triangle CEF) :=
  sorry

end incenter_of_triangle_CEF_l657_657489


namespace students_first_day_l657_657194

-- Definitions based on conditions
def total_books : ℕ := 120
def books_per_student : ℕ := 5
def students_second_day : ℕ := 5
def students_third_day : ℕ := 6
def students_fourth_day : ℕ := 9

-- Main goal
theorem students_first_day (total_books_eq : total_books = 120)
                           (books_per_student_eq : books_per_student = 5)
                           (students_second_day_eq : students_second_day = 5)
                           (students_third_day_eq : students_third_day = 6)
                           (students_fourth_day_eq : students_fourth_day = 9) :
  let books_given_second_day := students_second_day * books_per_student
  let books_given_third_day := students_third_day * books_per_student
  let books_given_fourth_day := students_fourth_day * books_per_student
  let total_books_given_after_first_day := books_given_second_day + books_given_third_day + books_given_fourth_day
  let books_first_day := total_books - total_books_given_after_first_day
  let students_first_day := books_first_day / books_per_student
  students_first_day = 4 :=
by sorry

end students_first_day_l657_657194


namespace poles_needed_l657_657639

theorem poles_needed (len longer_side shorter_side dist_long_side dist_short_side : ℕ) (h_len : len = 120) (h_wid : wid = 80) (h_dist_long : dist_long_side = 5) (h_dist_short : dist_short_side = 4) : 
  let n_longer_per_side := len / dist_long_side - 1
  let n_shorter_per_side := wid / dist_short_side - 1
  let n_longer_total := 2 * n_longer_per_side
  let n_shorter_total := 2 * n_shorter_per_side
  let n_total := n_longer_total + n_shorter_total in
  n_total = 84 :=
sorry

end poles_needed_l657_657639


namespace projectiles_initial_distance_l657_657567

theorem projectiles_initial_distance 
  (v₁ v₂ : ℝ) (t : ℝ) (d₁ d₂ d : ℝ) 
  (hv₁ : v₁ = 445 / 60) -- speed of first projectile in km/min
  (hv₂ : v₂ = 545 / 60) -- speed of second projectile in km/min
  (ht : t = 84) -- time to meet in minutes
  (hd₁ : d₁ = v₁ * t) -- distance traveled by the first projectile
  (hd₂ : d₂ = v₂ * t) -- distance traveled by the second projectile
  (hd : d = d₁ + d₂) -- total initial distance
  : d = 1385.6 :=
by 
  sorry

end projectiles_initial_distance_l657_657567


namespace right_angled_triangle_l657_657522

variable {α β γ a b c : ℝ}

theorem right_angled_triangle (h : Real.cot (α / 2) = (b + c) / a) : α + β + γ = π ∧ (α = π / 2 ∨ β = π / 2 ∨ γ = π / 2) :=
by
  sorry

end right_angled_triangle_l657_657522


namespace stone_105_is_3_l657_657699

def stone_numbered_at_105 (n : ℕ) := (15 + (n - 1) % 28)

theorem stone_105_is_3 :
  stone_numbered_at_105 105 = 3 := by
  sorry

end stone_105_is_3_l657_657699


namespace boys_meet_thirteen_times_l657_657566

-- Define the given conditions and setup
variables (circumference : ℝ)
def speed1 : ℝ := 5 -- speed of first boy in ft/s
def speed2 : ℝ := 9 -- speed of second boy in ft/s

-- Define the problem to prove the boys meet 13 times excluding start and finish
theorem boys_meet_thirteen_times (circumference_ne_zero : circumference ≠ 0) : 
  let relative_speed := speed1 + speed2 in
  let time_to_meet_A := circumference / relative_speed in
  let meetings_per_second := (5 / circumference * 9 / circumference) / (circumference / relative_speed) in
  let total_meetings := (meetings_per_second * time_to_meet_A) in
  (total_meetings : ℝ).floor = 13 :=
by {
  -- This is a placeholder for the actual proof which is not required for now
  sorry
}

end boys_meet_thirteen_times_l657_657566


namespace curve_symmetry_l657_657692

-- Define the curve equation
def curve_eq (x y : ℝ) : Prop := x * y^2 - x^2 * y = -2

-- Define the symmetry condition about the line y = -x
def symmetry_about_y_equals_neg_x (x y : ℝ) : Prop :=
  curve_eq (-y) (-x)

-- Define the original curve equation
def original_curve (x y : ℝ) : Prop := curve_eq x y

-- Proof statement: The curve xy^2 - x^2y = -2 is symmetric about the line y = -x.
theorem curve_symmetry : ∀ (x y : ℝ), original_curve x y ↔ symmetry_about_y_equals_neg_x x y :=
by
  sorry

end curve_symmetry_l657_657692


namespace used_compact_disk_cost_l657_657827

noncomputable def findCostOfUsedDisk : ℝ :=
  let N U : ℝ
  assume (h1 : 6 * N + 2 * U = 127.92) (h2 : 3 * N + 8 * U = 133.89),
  U

theorem used_compact_disk_cost (N U : ℝ) (h1 : 6 * N + 2 * U = 127.92) (h2 : 3 * N + 8 * U = 133.89) :
  U = 9.99 :=
by
  -- Proof omitted
  sorry

end used_compact_disk_cost_l657_657827


namespace negative_values_of_x_l657_657298

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l657_657298


namespace intersection_M_N_l657_657041

def M := {1, 3, 5, 7, 9}

def N := {x : ℤ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} := by
  sorry

end intersection_M_N_l657_657041


namespace x_mid_A_B_l657_657725

noncomputable theory
open Classical

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := (1 / 4) * x^2

-- Define the equation of the line with slope 1 through the focus (0, 1)
def line (x : ℝ) : ℝ := x + 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Given conditions about points A and B intersecting line and parabola
def points_condition (x : ℝ) (y : ℝ) : Prop :=
  parabola x = y ∧ line x = y

-- The coordinates of A and B satisfying points_condition
def A : ℝ × ℝ := Classical.some (Classical.indefinite_description _ (exists.intro 0 (points_condition 0 0)))
def B : ℝ × ℝ := Classical.some (Classical.indefinite_description _ (exists.intro 0 (points_condition 0 0)))

-- The x-coordinates of A and B
def x_A : ℝ := (Classical.some (Classical.indefinite_description _ (exists.intro 0 (points_condition 0 0)))).fst
def x_B : ℝ := (Classical.some (Classical.indefinite_description _ (exists.intro 0 (points_condition 0 0)))).fst

-- Theorem to prove the x-coordinate of the midpoint of A and B is 2
theorem x_mid_A_B : (x_A + x_B) / 2 = 2 :=
by
  sorry

end x_mid_A_B_l657_657725


namespace area_of_triangle_ABC_l657_657426

-- Define the side lengths a, b, and c
def a : ℝ := 3
def b : ℝ := 2
def c : ℝ := Real.sqrt 19

-- Define the function to calculate the area based on sides a, b, and angle C
def area_of_triangle (a b c : ℝ) : ℝ :=
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let C := Real.arccos cos_C
  let sin_C := Real.sin C
  (1/2) * a * b * sin_C

-- The statement to prove: the area of the triangle given sides a=3, b=2, c=sqrt(19) 
-- is equal to 3 * sqrt 3 / 2.
theorem area_of_triangle_ABC : area_of_triangle 3 2 (Real.sqrt 19) = 3 * Real.sqrt 3 / 2 := 
by sorry

end area_of_triangle_ABC_l657_657426


namespace correct_average_is_15_l657_657604

theorem correct_average_is_15 (n incorrect_avg correct_num wrong_num : ℕ) 
  (h1 : n = 10) (h2 : incorrect_avg = 14) (h3 : correct_num = 36) (h4 : wrong_num = 26) : 
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 15 := 
by 
  sorry

end correct_average_is_15_l657_657604


namespace problem1_problem2_problem3_l657_657395

-- Define the function f(x)
def f (a x : ℝ) : ℝ := -x^2 + 2 * a * x - 1

-- Problem 1: Maximum and minimum when a = 1
theorem problem1 : 
  let a := 1 
  in (∀ x ∈ set.Icc (-2:ℝ) 2, f a x ≤ 0) ∧ (∀ x ∈ set.Icc (-2:ℝ) 2, f a x ≥ -9) := 
by sorry

-- Problem 2: Determine the range of a for non-monotonicity
theorem problem2 : 
  (∀ a : ℝ, (∃ x1 x2 ∈ set.Icc (-2:ℝ) 2, x1 < x2 ∧ f a x1 < f a x2) ∧ (∃ x3 x4 ∈ set.Icc (-2:ℝ) 2, x3 < x4 ∧ f a x3 > f a x4)) ↔
  (a > -2 ∧ a < 2) := 
by sorry

-- Define g(a) as the maximum of f(x) on [-2, 2]
noncomputable def g (a : ℝ) : ℝ := 
  let values := (set.image (λ x, f a x) (set.Icc (-2:ℝ) 2)) in
  Sup values

-- Problem 3: Find the minimum value of g(a)
theorem problem3 :
  (∀ a : ℝ, g(a) ≥ -1) ∧ (∃ a : ℝ, g(a) = -1) := 
by sorry

end problem1_problem2_problem3_l657_657395


namespace intersection_M_N_l657_657038

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := { x | 2 * x > 7 }

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657038


namespace max_value_of_f_l657_657926

noncomputable def f (x: ℝ) := (Real.sqrt x) / (x + 1)

theorem max_value_of_f :
  (∀ x ≥ 0, f x ≤ 1 / 2) ∧ (f 1 = 1 / 2) := 
begin
  sorry
end

end max_value_of_f_l657_657926


namespace count_negative_x_with_sqrt_pos_int_l657_657347

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l657_657347


namespace fraction_of_original_price_l657_657152

theorem fraction_of_original_price
  (CP SP : ℝ)
  (h1 : SP = 1.275 * CP)
  (f: ℝ)
  (h2 : f * SP = 0.85 * CP)
  : f = 17 / 25 :=
by
  sorry

end fraction_of_original_price_l657_657152


namespace rectangle_area_circumscribed_circle_l657_657110

noncomputable def circle_eq (x y : ℝ) := x^2 + 4 * x + y^2 - 6 * y = 28

theorem rectangle_area_circumscribed_circle :
  (∃ x y : ℝ, circle_eq x y) →
  ∀ w h : ℝ, w = 2 * Real.sqrt 41 → h = 2 * Real.sqrt 41 → (w * h = 164) :=
by
  intros _ w h hw hh
  rw [hw, hh]
  ring_nf
  norm_num
  sorry

end rectangle_area_circumscribed_circle_l657_657110


namespace find_g_at_1_l657_657624

noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem find_g_at_1 : 
  (∀ x : ℝ, g (2*x + 3) = x^2 - 2*x + 4) → 
  g 1 = 7 := 
by
  intro h
  -- Proof goes here
  sorry

end find_g_at_1_l657_657624


namespace first_player_wins_l657_657543

noncomputable def has_winning_strategy (n : ℕ) : Prop :=
  if n = 1000 then true else
  ∃ m : ℕ, m < n ∧ has_winning_strategy (n + m) = false

theorem first_player_wins : has_winning_strategy 2 := sorry

end first_player_wins_l657_657543


namespace sum_of_ages_of_alex_and_allison_is_47_l657_657977

theorem sum_of_ages_of_alex_and_allison_is_47 (diane_age_now : ℕ)
  (diane_age_at_30_alex_relation : diane_age_now + 14 = 30 ∧ diane_age_now + 14 = 60 / 2)
  (diane_age_at_30_allison_relation : diane_age_now + 14 = 30 ∧ 30 = 2 * (diane_age_now + 14 - (30 - 15)))
  : (60 - (30 - 16)) + (15 - (30 - 16)) = 47 :=
by
  sorry

end sum_of_ages_of_alex_and_allison_is_47_l657_657977


namespace awards_distribution_l657_657095

-- Definition of our problem in Lean 4.
theorem awards_distribution (awards : Finset ℕ) (students : Finset ℕ) (h_awards_card : awards.card = 6) (h_students_card : students.card = 4) (h_each_student_gets_award : ∀ s : ℕ, s ∈ students → ∃ a : ℕ, a ∈ awards) :
  ∃ (distributions : Finset (ℕ → ℕ)), distributions.card = 1260 :=
by
  sorry

end awards_distribution_l657_657095


namespace inequality_proof_l657_657844

theorem inequality_proof (n : ℕ) (x : Fin (n+1) → ℝ) (h₀ : x 0 = 0) 
  (h_pos : ∀ i, 1 ≤ i → i ≤ n → x i > 0) 
  (h_sum1 : ∑ i in Finset.range n, x (i + 1) = 1) :
  1 ≤ ∑ i in Finset.range n, x (i + 1) / 
  (Real.sqrt (1 + (∑ j in Finset.range i, x (j + 1)))) * 
  (Real.sqrt (∑ k in (Finset.range (n - i)).map (Fin.castAdd 1), x k)) < 
  Real.pi / 2 := 
  sorry

end inequality_proof_l657_657844


namespace trisha_interest_l657_657565

noncomputable def total_amount (P : ℝ) (r : ℝ) (D : ℝ) (t : ℕ) : ℝ :=
  let rec compute (n : ℕ) (A : ℝ) :=
    if n = 0 then A
    else let A_next := A * (1 + r) + D
         compute (n - 1) A_next
  compute t P

noncomputable def total_deposits (D : ℝ) (t : ℕ) : ℝ :=
  D * t

noncomputable def total_interest (P : ℝ) (r : ℝ) (D : ℝ) (t : ℕ) : ℝ :=
  total_amount P r D t - P - total_deposits D t

theorem trisha_interest :
  total_interest 2000 0.05 300 5 = 710.25 :=
by
  sorry

end trisha_interest_l657_657565


namespace intersection_M_N_is_1_2_l657_657768

noncomputable def M : Set ℝ := {x : ℝ | x^2 - 2 * x - 3 < 0}
noncomputable def N : Set ℕ := {n : ℕ | True}

theorem intersection_M_N_is_1_2 : M ∩ N = {1, 2} :=
sorry

end intersection_M_N_is_1_2_l657_657768


namespace num_outfits_l657_657901

def num_shirts := 6
def num_ties := 4
def num_pants := 3
def outfits : ℕ := num_shirts * num_pants * (num_ties + 1)

theorem num_outfits: outfits = 90 :=
by 
  -- sorry will be removed when proof is provided
  sorry

end num_outfits_l657_657901


namespace no_integer_points_in_intersection_l657_657232

theorem no_integer_points_in_intersection :
  ∀ (x y z : ℤ), (x^2 + y^2 + (z - 10)^2 ≤ 9 ∧ x^2 + y^2 + (z - 2)^2 ≤ 16) → false :=
begin
  sorry
end

end no_integer_points_in_intersection_l657_657232


namespace intersection_A_B_l657_657739

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_A_B : A ∩ B = {1, 2} :=
  sorry

end intersection_A_B_l657_657739


namespace cyclic_quad_diameter_l657_657542

theorem cyclic_quad_diameter (A B C D E : Point) (h1 : same_circle {A, B, C, D}) 
  (h2 : ∃ (P : Point), collinear A B P ∧ collinear C D P ∧ ∃ (e : Point), e ∉ segment A B ∧ e ∉ segment C D ∧ P ∈ line_through e (segment A B) ∧  P ∈ line_through e (segment C D))
  (h3 : ∠ AED = 60 ∘) 
  (h4 : ∠ ABD = 3 * ∠ BAC) : is_diameter A D := sorry

end cyclic_quad_diameter_l657_657542


namespace find_f_zero_forall_x_f_pos_solve_inequality_l657_657487

variable {f : ℝ → ℝ}

-- Conditions
axiom condition_1 : ∀ x, x > 0 → f x > 1
axiom condition_2 : ∀ x y, f (x + y) = f x * f y
axiom condition_3 : f 2 = 3

-- Questions rewritten as Lean theorems

theorem find_f_zero : f 0 = 1 := sorry

theorem forall_x_f_pos : ∀ x, f x > 0 := sorry

theorem solve_inequality : ∀ x, f (7 + 2 * x) > 9 ↔ x > -3 / 2 := sorry

end find_f_zero_forall_x_f_pos_solve_inequality_l657_657487


namespace intersection_A_B_l657_657738

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_A_B : A ∩ B = {1, 2} :=
  sorry

end intersection_A_B_l657_657738


namespace general_term_sum_first_n_terms_l657_657817

-- Definitions based on conditions 
noncomputable def a2 : ℕ → ℝ := λ n, if n = 2 then 2 else 0
noncomputable def a8 : ℕ → ℝ := λ n, if n = 8 then 128 else 0

-- Theorem statement for part 1: General formula
theorem general_term (n : ℕ) : 
  (∃ a1 q : ℝ, a1 * q = 2 ∧ a1 * q^7 = 128 ∧ (a_n n = a1 * q^(n-1) ∨ a_n n = -a1 * (-q)^(n-1)))
  :=
sorry

-- Definitions for sum of sequence
noncomputable def sum_an_formula (n : ℕ) (a_n_formula : ℕ → ℝ) : ℝ :=
  if a_n_formula = (λ n, 2^(n - 1))
  then 2^n - 1
  else if a_n_formula = (λ n, -(-2)^(n - 1))
  then (1 / 3) * ((-2)^n - 1)
  else 0

-- Theorem statement for part 2: Sum of the first n terms
theorem sum_first_n_terms (n : ℕ) : 
  (∃ a1 q : ℝ, a1 * q = 2 ∧ a1 * q^7 = 128 ∧ 
  (S_n n = 2^n - 1 ∨ S_n n = (1 / 3) * ((-2)^n - 1)))
  :=
sorry

end general_term_sum_first_n_terms_l657_657817


namespace value_of_x_l657_657134

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := 
by
  sorry

end value_of_x_l657_657134


namespace group_division_l657_657444

theorem group_division (men women : ℕ) : 
  men = 4 → women = 5 →
  (∃ g1 g2 g3 : set (fin $ men + women), 
    g1.card = 3 ∧ g2.card = 3 ∧ g3.card = 3 ∧ 
    (∀ g, g ∈ [g1, g2, g3] → (∃ m w : ℕ, 1 ≤ m ∧ 1 ≤ w ∧ 
      finset.card (finset.filter (λ x, x < men) g) = m ∧ 
      finset.card (finset.filter (λ x, x ≥ men) g) = w)) 
    ∧ finset.disjoint g1 g2 ∧ finset.disjoint g2 g3 ∧ finset.disjoint g3 g1 
    ∧ g1 ∪ g2 ∪ g3 = finset.univ (fin $ men + women)) → 
  finset.card (finset.powerset' 3 (finset.univ (fin $ men + women))) / 2 = 180 :=
begin
  intros hmen hwomen,
  sorry
end

end group_division_l657_657444


namespace belinda_age_more_than_twice_tony_age_l657_657958

theorem belinda_age_more_than_twice_tony_age :
  ∀ (Tony_age Belinda_age : ℕ), 
  Tony_age = 16 → 
  Belinda_age = 40 → 
  Belinda_age - 2 * Tony_age = 8 
:= by
  intros Tony_age Belinda_age hTony hBelinda
  rw [hTony, hBelinda]
  sorry

end belinda_age_more_than_twice_tony_age_l657_657958


namespace part1_assoc_eq_part2_k_range_part3_m_range_l657_657186

-- Part 1
theorem part1_assoc_eq (x : ℝ) :
  (2 * (x + 1) - x = -3 ∧ (-4 < x ∧ x ≤ 4)) ∨ 
  ((x+1)/3 - 1 = x ∧ (-4 < x ∧ x ≤ 4)) ∨ 
  (2 * x - 7 = 0 ∧ (-4 < x ∧ x ≤ 4)) :=
sorry

-- Part 2
theorem part2_k_range (k : ℝ) :
  (∀ (x : ℝ), (x = (k + 6) / 2) → -5 < x ∧ x ≤ -3) ↔ (-16 < k) ∧ (k ≤ -12) :=
sorry 

-- Part 3
theorem part3_m_range (m : ℝ) :
  (∀ (x : ℝ), (x = 6 * m - 5) → (0 < x) ∧ (x ≤ 3 * m + 1) ∧ (1 ≤ x) ∧ (x ≤ 3)) ↔ (5/6 < m) ∧ (m < 1) :=
sorry

end part1_assoc_eq_part2_k_range_part3_m_range_l657_657186


namespace cotangent_sum_identity_l657_657832

theorem cotangent_sum_identity
  (A B C: ℝ) (a b c S: ℝ)
  (h1: a = 2 * S / (b * sin C))
  (h2: b = 2 * S / (c * sin A))
  (h3: c = 2 * S / (a * sin B))
  (h4: S = 1/2 * b * c * sin A) :
  (Real.cot (A/2) + Real.cot (B/2) + Real.cot (C/2) = (a + b + c)^2 / (4 * S)) := 
sorry

end cotangent_sum_identity_l657_657832


namespace complex_conjugate_polynomial_has_real_coefficients_l657_657836

theorem complex_conjugate_polynomial_has_real_coefficients
  (p q : ℝ) :
  (∀ z : ℂ, z^2 + (6 + p * complex.I) * z + (15 + q * complex.I + 2 * complex.I^2) = 0 → z.im ≠ 0 → z.conj ∈ ({z} :set ℂ)) →
  p = 0 ∧ q = 0 := 
by
  sorry

end complex_conjugate_polynomial_has_real_coefficients_l657_657836


namespace pascal_fourth_number_in_row_of_15_entries_l657_657573

theorem pascal_fourth_number_in_row_of_15_entries : (nat.choose 13 3 = 286) :=
by
  sorry

end pascal_fourth_number_in_row_of_15_entries_l657_657573


namespace cone_symmetry_cases_l657_657889

theorem cone_symmetry_cases (bounded_cone : Cone) (single_surface_cone : Cone) (double_surface_cone : Cone)
  (H_bounded : bounded_cone.is_bounded)
  (H_single_napped : single_surface_cone.is_single_napped)
  (H_double_napped : double_surface_cone.is_double_napped)
  (bounded_cone_symmetry : bounded_cone.symmetry = ⟨bounded_cone.axis, bounded_cone.bundle_of_planes, false⟩)
  (single_surface_cone_symmetry : single_surface_cone.symmetry = ⟨single_surface_cone.axis, single_surface_cone.bundle_of_planes, false⟩)
  (double_surface_cone_symmetry : double_surface_cone.symmetry = ⟨double_surface_cone.axis, double_surface_cone.bundle_of_planes, true, double_surface_cone.vertex_as_center⟩) :
  True :=
by
  sorry

end cone_symmetry_cases_l657_657889


namespace num_valid_Q_l657_657481

open Polynomial

namespace PolynomialProof

def P (x : ℚ) : ℚ := (x - 1) * (x - 2) * (x - 4)

theorem num_valid_Q : 
  ∃ (Q : Polynomial ℚ), (∃ (R : Polynomial ℚ) (hR : R.degree = 3), P(Q) = P * R) ∧ 
  Finset.card { t | ∃ i j k, t = (Q.eval i, Q.eval j, Q.eval k) ∧ i ∈ {1, 2, 4} ∧ j ∈ {1, 2, 4} ∧ k ∈ {1, 2, 4} } = 22 := 
sorry

end PolynomialProof

end num_valid_Q_l657_657481


namespace selena_book_pages_l657_657527

variable (S : ℕ)
variable (H : ℕ)

theorem selena_book_pages (cond1 : H = S / 2 - 20) (cond2 : H = 180) : S = 400 :=
by
  sorry

end selena_book_pages_l657_657527


namespace duration_period_l657_657182

-- Define the conditions and what we need to prove
theorem duration_period (t : ℝ) (h : 3200 * 0.025 * t = 400) : 
  t = 5 :=
sorry

end duration_period_l657_657182


namespace negative_values_count_l657_657328

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l657_657328


namespace paul_allowance_received_l657_657518

theorem paul_allowance_received 
  (saved_money : ℕ) 
  (toy_cost : ℕ) 
  (num_toys : ℕ) 
  (total_money_needed : ℕ) :
  saved_money = 3 →
  toy_cost = 5 →
  num_toys = 2 →
  total_money_needed = num_toys * toy_cost →
  (total_money_needed - saved_money) = 7 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  reflexivity

end paul_allowance_received_l657_657518


namespace sixth_candy_is_peter_l657_657249

-- Define who receives candies based on the position
def candyReceiver : ℕ → String
| 1 := "Peter"
| 2 := "Peter"
| 3 := "Peter"
| 4 := "Vasya"
| 5 := "Vasya"
| (n + 1) := candyReceiver ((n + 1) % 5 + 1)

-- The theorem stating that the sixth candy is received by Peter
theorem sixth_candy_is_peter : candyReceiver 6 = "Peter" := by
  sorry

end sixth_candy_is_peter_l657_657249


namespace joy_quadrilateral_rods_l657_657826

theorem joy_quadrilateral_rods :
  let rods := {n : ℕ | 1 ≤ n ∧ n ≤ 30}
  let used_rods := {4, 8, 16}
  let remaining_rods := rods \ used_rods
  let valid_rods := {n : ℕ | 5 ≤ n ∧ n < 28}
  ∃ s : Finset ℕ, s = remaining_rods ∩ valid_rods ∧ s.card = 20 :=
by
  let rods := {n : ℕ | 1 ≤ n ∧ n ≤ 30}
  let used_rods := {4, 8, 16}
  let remaining_rods := rods \ used_rods
  let valid_rods := {n : ℕ | 5 ≤ n ∧ n < 28}
  use remaining_rods ∩ valid_rods
  split
  { refl, }
  {
    sorry
  }

end joy_quadrilateral_rods_l657_657826


namespace base5_operations_l657_657535

noncomputable def base5_to_base10 (a b c : ℕ) : ℕ :=
  a * 5^2 + b * 5^1 + c * 5^0

noncomputable def base5_mult_sub (a b c d e : ℕ) : ℕ :=
  (base5_to_base10 a b c) * (base5_to_base10 0 d e) - (base5_to_base10 0 (d / 5) (d % 5))

noncomputable def base10_to_base5 (n : ℕ) : list ℕ := 
  let rec := λ
    | 0, _, acc => acc
    | n, b, acc => rec (n % b) (b / 5) ((n / b) :: acc)
  rec n 625 []

theorem base5_operations :
  base10_to_base5 (base5_mult_sub 2 3 1 2 4 1 2 0 7) = [1, 2, 1, 3, 2] :=
sorry

end base5_operations_l657_657535


namespace current_average_age_l657_657948

/-- Assuming there are 7 people in a group such that the youngest person's age is 4,
and the average age of the group when the youngest was born was 26 years,
prove that the current average age of the group is 184/7 years. -/
theorem current_average_age (n : ℕ) (youngest_age : ℕ) (past_avg_age : ℚ) (current_avg_age : ℚ) :
  n = 7 →
  youngest_age = 4 →
  past_avg_age = 26 →
  current_avg_age = (6 * past_avg_age + 6 * youngest_age + youngest_age) / n →
  current_avg_age = 184 / 7 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end current_average_age_l657_657948


namespace pond_fish_approximation_l657_657431

noncomputable def total_number_of_fish
  (tagged_first: ℕ) (total_caught_second: ℕ) (tagged_second: ℕ) : ℕ :=
  (tagged_first * total_caught_second) / tagged_second

theorem pond_fish_approximation :
  total_number_of_fish 60 50 2 = 1500 :=
by
  -- calculation of the total number of fish based on given conditions
  sorry

end pond_fish_approximation_l657_657431


namespace find_coordinates_of_P_l657_657763

noncomputable def f : ℝ → ℝ := λ x, x^3

theorem find_coordinates_of_P :
  ∃ (x y : ℝ), y = f x ∧ (3 * x^2 = 3) ∧ (x = 1 ∧ y = 1 ∨ x = -1 ∧ y = -1) :=
sorry

end find_coordinates_of_P_l657_657763


namespace group_count_4_men_5_women_l657_657455

theorem group_count_4_men_5_women : 
  let men := 4
  let women := 5
  let groups := List.replicate 3 (3, true)
  ∃ (m_w_combinations : List (ℕ × ℕ)),
    m_w_combinations = [(1, 2), (2, 1)] ∧
    ((men.choose m_w_combinations.head.fst * women.choose m_w_combinations.head.snd) * (men - m_w_combinations.head.fst).choose m_w_combinations.tail.head.fst * (women - m_w_combinations.head.snd).choose m_w_combinations.tail.head.snd) = 360 :=
by
  sorry

end group_count_4_men_5_women_l657_657455


namespace g_f_g_1_equals_82_l657_657054

def f (x : ℤ) : ℤ := 2 * x + 2
def g (x : ℤ) : ℤ := 5 * x + 2
def x : ℤ := 1

theorem g_f_g_1_equals_82 : g (f (g x)) = 82 := by
  sorry

end g_f_g_1_equals_82_l657_657054


namespace distance_squared_from_B_to_center_l657_657184

theorem distance_squared_from_B_to_center :
  ∃ a b : ℝ, (a^2 + b^2 = 5) ∧
             (a^2 + (b + 8)^2 = 50) ∧
             ((a + 3 * real.cos (real.pi / 4))^2 + (b + 3 * real.sin (real.pi / 4))^2 = 50) :=
begin
  -- From the two conditions, we need to find (a, b) such that:
  -- 1. a^2 + (b + 8)^2 = 50
  -- 2. (a + 3 * cos(45 degrees))^2 + (b + 3 * sin(45 degrees))^2 = 50
  -- And consequently, a^2 + b^2 = 5.
  sorry
end

end distance_squared_from_B_to_center_l657_657184


namespace largest_three_digit_multiple_of_4_and_5_l657_657972

theorem largest_three_digit_multiple_of_4_and_5 : 
  ∃ (n : ℕ), n < 1000 ∧ n ≥ 100 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n = 980 :=
by
  sorry

end largest_three_digit_multiple_of_4_and_5_l657_657972


namespace union_A_B_l657_657500

def A := {x : ℕ | -1 < x ∧ x ≤ 2}
def B := {2, 3}

theorem union_A_B : A ∪ B = {0, 1, 2, 3} :=
by {
  sorry
}

end union_A_B_l657_657500


namespace fourth_bus_people_difference_l657_657939

def bus1_people : Nat := 12
def bus2_people : Nat := 2 * bus1_people
def bus3_people : Nat := bus2_people - 6
def total_people : Nat := 75
def bus4_people : Nat := total_people - (bus1_people + bus2_people + bus3_people)
def difference_people : Nat := bus4_people - bus1_people

theorem fourth_bus_people_difference : difference_people = 9 := by
  -- Proof logic here
  sorry

end fourth_bus_people_difference_l657_657939


namespace possible_values_of_a_l657_657120

theorem possible_values_of_a 
  (p : ℕ) (hp : p.prime) 
  (a b k : ℤ) 
  (h1 : -p^2 * (1 + k) = 1100)
  (hb : b = k * p) 
  (ha : a = -((b + p^2) / p)) :
  (a = 274 ∨ a = 40) :=
sorry

end possible_values_of_a_l657_657120


namespace altitude_on_BC_eqn_midline_of_AC_eqn_circumcircle_eqn_l657_657758

noncomputable def A := (0 : ℝ, 1 : ℝ)
noncomputable def B := (0 : ℝ, -1 : ℝ)
noncomputable def C := (-2 : ℝ, 1 : ℝ)

theorem altitude_on_BC_eqn :
  ∃ (l : ℝ → ℝ → Prop), (∀ x y, l x y ↔ x - y + 1 = 0) :=
sorry

theorem midline_of_AC_eqn :
  ∃ (l : ℝ → ℝ → Prop), (∀ x y, l x y ↔ x = -1) :=
sorry

theorem circumcircle_eqn :
  ∃ (C : ℝ → ℝ → Prop), (∀ x y, C x y ↔ x^2 + y^2 + 2*x - 1 = 0) :=
sorry

end altitude_on_BC_eqn_midline_of_AC_eqn_circumcircle_eqn_l657_657758


namespace magnitude_of_complex_l657_657696

theorem magnitude_of_complex : 
  let z := Complex.mk (2/3) 3 in
  Complex.abs z = Real.sqrt 85 / 3 :=
  sorry

end magnitude_of_complex_l657_657696


namespace area_percentage_decrease_l657_657715

theorem area_percentage_decrease {a b : ℝ} 
  (h1 : 2 * b = 0.1 * 4 * a) :
  ((b^2) / (a^2) * 100 = 4) :=
by
  sorry

end area_percentage_decrease_l657_657715


namespace cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8_l657_657414

theorem cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8
  (α : ℝ) (h : Real.cos α = 2 * Real.cos (α + Real.pi / 4)) :
  Real.tan (α + Real.pi / 8) = 3 * (Real.sqrt 2 + 1) := 
sorry

end cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8_l657_657414


namespace tom_splitting_slices_l657_657957

theorem tom_splitting_slices :
  ∃ S : ℕ, (∃ t, t = 3/8 * S) → 
          (∃ u, u = 1/2 * (S - t)) → 
          (∃ v, v = u + t) → 
          (v = 5) → 
          (S / 2 = 8) :=
sorry

end tom_splitting_slices_l657_657957


namespace arcsin_arccos_arctan_order_l657_657212

theorem arcsin_arccos_arctan_order : 
  arccos (-3 / 4) > arcsin (-2 / 5) 
  ∧ arcsin (-2 / 5) > arctan (-5 / 4) := 
by 
  sorry

end arcsin_arccos_arctan_order_l657_657212


namespace angle_in_degrees_l657_657831

noncomputable def angle_between_vectors
  (a b c : ℝ) : ℝ :=
  let unit_a : Real := 1
  let unit_b : Real := 1
  let unit_c : Real := 1
  let linear_independence := True
  let triple_product_identity := (unit_b - unit_c) / sqrt 3

  if linear_independence 
      ∧ a * dot_product b c * b - a * dot_product a b * c == triple_product_identity then
    acos (unit_a * unit_c / (a * sqrt 3))
  else
    0

theorem angle_in_degrees (a b c : ℝ) : 
  a 
  ∧ unit_vector b 
  ∧ unit_vector c 
  ∧ linear_independent_set {a, b, c} 
  ∧ angle_between_vectors a b c = 54.74 :=
begin
  sorry
end

end angle_in_degrees_l657_657831


namespace parallelogram_area_60_l657_657077

theorem parallelogram_area_60
  (α β : ℝ) (a b : ℝ)
  (h_angle : α = 150) 
  (h_adj_angle : β = 180 - α) 
  (h_len_1 : a = 10)
  (h_len_2 : b = 12) :
  ∃ (area : ℝ), area = 60 := 
by 
  use 60
  sorry

end parallelogram_area_60_l657_657077


namespace area_ratio_invariance_l657_657495

open Euclid Geometry

variables {A B C D E F P : Point}
variables (AB CD AD BC : Segment)

-- Given conditions
def isCyclicQuad (ABCD : Quad) : Prop := isCyclic ABCD

def pointsOnSegments (AB CD : Segment) (E F : Point) : Prop :=
  E ∈ AB ∧ F ∈ CD

def equalRatios (E F : Point) (AB CD : Segment) : Prop :=
  (∃ AE EB CF FD : Segment, AE / EB = CF / FD ∧ AE + EB = AB ∧ CF + FD = CD)

def pointOnSegment (P : Point) (EF : Segment) : Prop :=
  P ∈ EF

def ratioPE_PF (P E F : Point) (AB CD : Segment) (EF : Segment) : Prop :=
  (∃ PE PF : Segment, PE / PF = AB / CD ∧ PE + PF = EF)

-- Triangle area ratio independence proof statement
theorem area_ratio_invariance
  (ABCD : Quad) (h_cyclic : isCyclicQuad ABCD)
  (h_points : pointsOnSegments AB CD E F)
  (h_ratios : equalRatios E F AB CD)
  (EF : Segment) (h_pos : pointOnSegment P EF) (h_ratio : ratioPE_PF P E F AB CD EF) :
  triangleAreaRatio (triangle A P D) (triangle B P C) = AD / BC := sorry

end area_ratio_invariance_l657_657495


namespace common_difference_arithmetic_progression_l657_657003

theorem common_difference_arithmetic_progression {n : ℕ} (x y : ℝ) (a : ℕ → ℝ) 
  (h : ∀ k : ℕ, k ≤ n → a (k+1) = a k + (y - x) / (n + 1)) 
  : (∃ d : ℝ, ∀ i : ℕ, i ≤ n + 1 → a (i+1) = x + i * d) ∧ d = (y - x) / (n + 1) := 
by
  sorry

end common_difference_arithmetic_progression_l657_657003


namespace grouping_count_l657_657449

theorem grouping_count (men women : ℕ) 
  (h_men : men = 4) (h_women : women = 5)
  (at_least_one_man_woman : ∀ (g1 g2 g3 : Finset (Fin 9)), 
    g1.card = 3 → g2.card = 3 → g3.card = 3 → g1 ∩ g2 = ∅ → g2 ∩ g3 = ∅ → g3 ∩ g1 = ∅ → 
    (g1 ∩ univ.filter (· < 4)).nonempty ∧ (g1 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g2 ∩ univ.filter (· < 4)).nonempty ∧ (g2 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g3 ∩ univ.filter (· < 4)).nonempty ∧ (g3 ∩ univ.filter (· ≥ 4)).nonempty) :
  (choose 4 1 * choose 5 2 * choose 3 1 * choose 3 2) / 2! = 180 :=
sorry

end grouping_count_l657_657449


namespace count_divisible_by_401_l657_657238

theorem count_divisible_by_401 : (∃ (n : ℕ), n = 1011 ∧ ∀ k, 1 ≤ k ∧ k ≤ 2023 →
  let term := 4 * 10^(k - 1) + 1 in term % 401 = 0 ↔ k % 2 = 0) :=
by
  have sequence_term (k : ℕ) := 4 * 10^(k - 1) + 1
  have correct_answer := 1011
  have condition (k : ℕ) : 1 ≤ k ∧ k ≤ 2023 → sequence_term k % 401 = 0 ↔ k % 2 = 0
  sorry
  exact ⟨correct_answer, ⟨rfl, condition⟩⟩

end count_divisible_by_401_l657_657238


namespace number_of_negative_x_l657_657337

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l657_657337


namespace even_hamiltonian_cycles_l657_657950

/-- 
Given a bipartite graph G with vertex sets A and B, each of size n (n ≥ 2), 
and each vertex of degree 3, prove that if N denotes the number of Hamiltonian 
cycles in G, then (N / 4n) is an even integer.
-/
theorem even_hamiltonian_cycles 
  (A B : Finset ℕ) (G : SimpleGraph (sum A B))
  (hA_size : A.card = n) (hB_size : B.card = n) (hn_ge_two : 2 ≤ n)
  (h_bipartite : G.IsBipartite (sum A B))
  (h_deg_3 : ∀ v, G.degree v = 3) :
  ∃ m : ℕ, (N / (4 * n)) = 2 * m :=
  sorry

end even_hamiltonian_cycles_l657_657950


namespace exponent_equality_l657_657786

theorem exponent_equality (n : ℕ) : (4^8 = 4^n) → (n = 8) := by
  intro h
  sorry

end exponent_equality_l657_657786


namespace box_volume_l657_657638

theorem box_volume (a b c : ℝ) (H1 : a * b = 15) (H2 : b * c = 10) (H3 : c * a = 6) : a * b * c = 30 := 
sorry

end box_volume_l657_657638


namespace increasing_function_shape_implies_number_l657_657571

variable {I : Set ℝ} {f : ℝ → ℝ}

theorem increasing_function_shape_implies_number (h : ∀ (x₁ x₂ : ℝ), x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ < x₂ → f x₁ < f x₂) 
: ∀ (x₁ x₂ : ℝ), x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ < x₂ → f x₁ < f x₂ :=
sorry

end increasing_function_shape_implies_number_l657_657571


namespace Daniel_had_more_than_200_marbles_at_day_6_l657_657235

noncomputable def marbles (k : ℕ) : ℕ :=
  5 * 2^k

theorem Daniel_had_more_than_200_marbles_at_day_6 :
  ∃ k : ℕ, marbles k > 200 ∧ ∀ m < k, marbles m ≤ 200 :=
by
  sorry

end Daniel_had_more_than_200_marbles_at_day_6_l657_657235


namespace product_of_fractions_l657_657662

-- Define the fractions
def one_fourth : ℚ := 1 / 4
def one_half : ℚ := 1 / 2
def one_eighth : ℚ := 1 / 8

-- State the theorem we are proving
theorem product_of_fractions :
  one_fourth * one_half = one_eighth :=
by
  sorry

end product_of_fractions_l657_657662


namespace F_multiplicative_l657_657081

theorem F_multiplicative (F : ℝ → ℝ) (z1 z2 : ℝ) (h1 : 0 < z1) (h2 : 0 < z2) : 
  F(z1 * z2) = F(z1) + F(z2) :=
by
  sorry

end F_multiplicative_l657_657081


namespace count_negative_values_of_x_l657_657308

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l657_657308


namespace range_f_l657_657243

def f (x : ℝ) : ℝ := (x^2 - (1/2) * x + 2) / (x^2 + 2 * x + 3)

theorem range_f : Set.Icc (-21/4) (3/4) = { y : ℝ | ∃ x : ℝ, f x = y } :=
sorry

end range_f_l657_657243


namespace common_root_unique_k_l657_657767

theorem common_root_unique_k (k : ℝ) (x : ℝ) 
  (h₁ : x^2 + k * x - 12 = 0) 
  (h₂ : 3 * x^2 - 8 * x - 3 * k = 0) 
  : k = 1 :=
sorry

end common_root_unique_k_l657_657767


namespace find_g5_l657_657918

-- Define the function g with the property given in the conditions
def g (x : ℝ) : ℝ := sorry

-- Express the conditions in Lean 4
axiom functional_equation : ∀ x y : ℝ, g(x * y) = g(x) * g(y)
axiom g0_not_zero : g(0) ≠ 0

-- State the theorem to prove
theorem find_g5 : g(5) = 1 :=
by
  -- Proof goes here, placeholder for now.
  sorry

end find_g5_l657_657918


namespace remainder_3249_div_82_eq_51_l657_657150

theorem remainder_3249_div_82_eq_51 : (3249 % 82) = 51 :=
by
  sorry

end remainder_3249_div_82_eq_51_l657_657150


namespace sqrt_meaningful_iff_l657_657803

theorem sqrt_meaningful_iff (x: ℝ) : (6 - 2 * x ≥ 0) ↔ (x ≤ 3) :=
by
  sorry

end sqrt_meaningful_iff_l657_657803


namespace negative_values_count_l657_657319

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l657_657319


namespace distance_from_0_to_pi_div_4_l657_657560

def velocity (t : ℝ) : ℝ := t * sin (2 * t)

def distance_traveled (a b : ℝ) (v : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, v x

theorem distance_from_0_to_pi_div_4 :
  distance_traveled 0 (π / 4) velocity = 1 / 4 :=
by
  sorry

end distance_from_0_to_pi_div_4_l657_657560


namespace intersection_correct_l657_657019

-- Define the sets M and N
def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x | 2 * x > 7}

-- Define the expected intersection result
def expected_intersection : Set ℝ := {5, 7, 9}

-- State the theorem
theorem intersection_correct : ∀ x, x ∈ M ∩ N ↔ x ∈ expected_intersection :=
by
  sorry

end intersection_correct_l657_657019


namespace coin_and_die_probability_l657_657783

theorem coin_and_die_probability :
  let coin_outcomes := 2
  let die_outcomes := 8
  let total_outcomes := coin_outcomes * die_outcomes
  let successful_outcomes := 1 in
  let P := (successful_outcomes : ℚ) / total_outcomes in
  P = 1 / 16 :=
by
  sorry

end coin_and_die_probability_l657_657783


namespace average_speed_x_to_y_l657_657605

theorem average_speed_x_to_y :
  ∃ V : ℝ, 
  (∀ D : ℝ, D > 0 → 
  let T1 := D / V in
  let T2 := D / 36 in
  let total_avg_speed := 2 * D / (T1 + T2) in
  total_avg_speed = 43.2 ) →
  V = 54 :=
begin
  -- Setup the assumption
  intro h,
  -- Provide the value of V that we want to prove
  use 54,
  -- Now provide the details of how the proof should conclude with V = 54
  intros D D_pos,
  -- Apply the provided condition to ensure the average speed matches
  specialize h D D_pos,
  -- Conclude the proof with how the average speed relates
  rw h,
  -- Simplify the expression to reach the desired V value
  sorry
end

end average_speed_x_to_y_l657_657605


namespace find_zebras_last_year_l657_657932

def zebras_last_year (current : ℕ) (born : ℕ) (died : ℕ) : ℕ :=
  current - born + died

theorem find_zebras_last_year :
  zebras_last_year 725 419 263 = 569 :=
by
  sorry

end find_zebras_last_year_l657_657932


namespace count_negative_values_of_x_l657_657312

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l657_657312


namespace two_layers_area_zero_l657_657954

theorem two_layers_area_zero (A X Y Z : ℕ)
  (h1 : A = 212)
  (h2 : X + Y + Z = 140)
  (h3 : Y + Z = 24)
  (h4 : Z = 24) : Y = 0 :=
by
  sorry

end two_layers_area_zero_l657_657954


namespace number_of_functions_satisfying_condition_l657_657173

theorem number_of_functions_satisfying_condition :
  ∀ (f : ℝ → ℝ), (∀ (x y z : ℝ), f(x * y) + f(x * z) - f(x) * f(y * z) > 1) → false :=
by
  sorry

end number_of_functions_satisfying_condition_l657_657173


namespace find_initial_bears_before_shipment_l657_657646

-- Definitions based on the conditions
variables (x : ℕ) (shipment : ℕ) (shelfBears shelves totalBears : ℕ)
axiom initial_condition : shipment = 10
axiom shelf_distribution : shelfBears = 9
axiom shelves_used : shelves = 3
axiom total_bears_shelves : totalBears = shelves * shelfBears

-- The formal statement to be proven
theorem find_initial_bears_before_shipment (h1 : initial_condition) 
                                           (h2 : shelf_distribution) 
                                           (h3 : shelves_used)
                                           (h4 : total_bears_shelves)
                                           (totalBears = 27) :
                                           x = totalBears - shipment := 
  by sorry

end find_initial_bears_before_shipment_l657_657646


namespace solveEquations_l657_657097

-- Define the floor function
def floor (x : ℝ) : ℤ := 
  Real.floor x

-- Define the fractional part function
def fractional_part (x : ℝ) : ℝ := 
  x - floor x

-- Define the given equations
def equation1 (x : ℝ) : Prop := 
  (floor x : ℝ) = (8/9) * (x - 4) + 3

def equation2 (x : ℝ) : Prop := 
  fractional_part x = (1/9) * (x + 5)

-- Define the set of solutions
def solutions : List ℝ := 
  [-5, -3.9, -2.8, -1.6, -0.5, 0.6, 1.8, 2.9]

-- Proof statement
theorem solveEquations : 
  ∀ x ∈ solutions, equation1 x ∧ equation2 x :=
by
  -- A proof for the theorem will be required here
  sorry

end solveEquations_l657_657097


namespace powderman_distance_l657_657636

theorem powderman_distance:
  let blast_time := 45 in
  let running_speed_yards_per_sec := 10 in
  let running_speed_feet_per_sec := running_speed_yards_per_sec * 3 in
  let sound_speed_feet_per_sec := 1200 in
  let reaction_time := 2 in
  let t := (sound_speed_feet_per_sec * (blast_time + reaction_time)) / (sound_speed_feet_per_sec - running_speed_feet_per_sec) in
  let distance_covered_feet := running_speed_feet_per_sec * t in
  let distance_covered_yards := distance_covered_feet / 3 in
  distance_covered_yards ≈ 461 := sorry

end powderman_distance_l657_657636


namespace horizontal_asymptote_f_l657_657245

-- Define the function f(x) = (7x^2 - 8) / (4x^2 + 3x + 1)
def f (x : ℝ) : ℝ := (7 * x^2 - 8) / (4 * x^2 + 3 * x + 1)

-- Statement asserting the horizontal asymptote of f(x) as x approaches infinity
theorem horizontal_asymptote_f :
  ∃ a : ℝ, (∀ ε > 0, ∃ M > 0, ∀ x > M, |f(x) - a| < ε) ∧ a = 7 / 4 :=
by
  have h : ∀ ε > 0, ∃ M > 0, ∀ x > M, |f(x) - 7 / 4| < ε := sorry,
  use 7 / 4,
  split,
  exact h,
  rfl

end horizontal_asymptote_f_l657_657245


namespace minimum_value_of_expression_l657_657746

noncomputable def min_value (a b : ℝ) : ℝ :=
  a^2 + (1 / (a * b)) + (1 / (a * (a - b)))

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) : min_value a b >= 4 := by
  sorry

end minimum_value_of_expression_l657_657746


namespace fraction_of_grid_covered_by_triangle_l657_657650

open Set

def P : ℝ × ℝ := (2, 2)
def Q : ℝ × ℝ := (7, 3)
def R : ℝ × ℝ := (6, 5)
def grid_width : ℝ := 8
def grid_height : ℝ := 6

def shoelace_area (A B C : ℝ × ℝ) : ℝ :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def grid_area (width height : ℝ) : ℝ :=
  width * height

theorem fraction_of_grid_covered_by_triangle :
  let triangle_area := shoelace_area P Q R
  let full_grid_area := grid_area grid_width grid_height
  triangle_area / full_grid_area = 11 / 96 :=
by
  let triangle_area := shoelace_area P Q R
  let full_grid_area := grid_area grid_width grid_height
  have h_triangle_area : triangle_area = 11 / 2 := sorry
  have h_full_grid_area : full_grid_area = 48 := sorry
  show triangle_area / full_grid_area = 11 / 96
  rw [h_triangle_area, h_full_grid_area]
  simp

end fraction_of_grid_covered_by_triangle_l657_657650


namespace AM_GM_inequality_l657_657491

theorem AM_GM_inequality (n : ℕ) (a : Fin n → ℝ) (h1 : 3 ≤ n)
  (h2 : ∀ i, (1 ≤ i) → (i < n) → (0 < a i))
  (h3 : ∏ i in finset.range (n - 1).map (nat_cast_embedding.to_embedding), a (i + 1) = 1) :
  (∏ i in finset.range (n - 1).map (nat_cast_embedding.to_embedding),
     (1 + a (i + 1))^(i + 2)) > n^n :=
sorry

end AM_GM_inequality_l657_657491


namespace prop_p_range_prop_p_xor_prop_q_range_l657_657361

variables {x m : ℝ}

-- Define proposition p: the equation x² - mx + m = 0 has two distinct real roots in (1, +∞)
def prop_p (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ 1 < a ∧ 1 < b ∧ a^2 - m*a + m = 0 ∧ b^2 - m*b + m = 0

-- Define proposition q: the function f(x) = 1 / (4x² + mx + m) is defined for all real numbers
def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + m*x + m ≠ 0

-- The range of m where prop_p is true
theorem prop_p_range (m : ℝ) : prop_p m → m > 4 :=
sorry

-- The range of m where exactly one of prop_p and prop_q is true
theorem prop_p_xor_prop_q_range (m : ℝ) : (prop_p m ∨ prop_q m) ∧ ¬(prop_p m ∧ prop_q m) → (m ∈ set.Icc 0 4) ∨ (m ∈ set.Ici 16) :=
sorry

end prop_p_range_prop_p_xor_prop_q_range_l657_657361


namespace no_point_M_exists_line_through_R_equation_l657_657393

-- Part (I): Prove M does not exist
theorem no_point_M_exists : ¬ ∃ M, 
  (λ C l, (∃ A B Q, 
    C.contain A ∧ C.contain B ∧
    l.contain Q ∧
    (|A M| / |M B| = |A Q| / |Q B|))) :=
by sorry

-- Part (II): Prove the line equation
theorem line_through_R_equation (R E F: Point) (C: Ellipse) (l: Line) :
  (|R E| = |E F|) ∧ C.contain E ∧ C.contain F ∧ l.through R (E, F) → 
  l = line.mk (2, 1, -6) ∨ l = line.mk (14, 1, -18) :=
by sorry

end no_point_M_exists_line_through_R_equation_l657_657393


namespace combined_average_l657_657853

theorem combined_average (n1 n2 A1 A2 : ℕ) (hn1 : n1 = 25) (hA1 : A1 = 40) (hn2 : n2 = 30) (hA2 : A2 = 60) : 
  let T1 := n1 * A1 in
  let T2 := n2 * A2 in
  let T := T1 + T2 in
  let n := n1 + n2 in
  A = T / n :=
by
  sorry

end combined_average_l657_657853


namespace problem1_problem2_l657_657371

noncomputable def b_seq (n : ℕ) : ℝ :=
if n = 1 then (3 / 10)
else 1 - (1 / (4 * b_seq (n - 1)))

noncomputable def a_seq (n : ℕ) : ℝ :=
2 / (1 - 2 * b_seq n)

def is_arithmetic (seq : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, seq (n + 1) = seq n + d

theorem problem1 :
is_arithmetic a_seq (-2) :=
sorry

noncomputable def c_seq (n : ℕ) : ℕ :=
(3^n + 7) / 2

theorem problem2 (k : ℝ) :
(k < -(1 / 81)) ↔ (∀ n : ℕ, k * (2 * c_seq n - 7) < a_seq n) :=
sorry

end problem1_problem2_l657_657371


namespace correct_relationships_count_l657_657931

def statement1 : Prop := 0 ∈ ({0} : Set ℕ)
def statement2 : Prop := (∅ : Set ℕ) ⊂ ({0} : Set ℕ)
def statement3 : Prop := ({0, 1} : Set (ℕ ⊕ (ℕ × ℕ))) ⊆ ({((0, 1), 0)} : Set (ℕ ⊕ (ℕ × ℕ)))
def statement4 : Prop := ({((1, 0), 0)} : Set (ℕ ⊕ (ℕ × ℕ))) = ({((0, 1), 0)} : Set (ℕ ⊕ (ℕ × ℕ)))

def correct_statement_count (s1 s2 s3 s4 : Prop) : ℕ :=
  ([s1, s2, s3, s4].count (fun s => s))  -- This counts the number of true statements in the list

theorem correct_relationships_count : correct_statement_count statement1 statement2 statement3 statement4 = 2 := 
by {
  sorry  -- proof goes here
}

end correct_relationships_count_l657_657931


namespace num_valid_Q_l657_657480

open Polynomial

namespace PolynomialProof

def P (x : ℚ) : ℚ := (x - 1) * (x - 2) * (x - 4)

theorem num_valid_Q : 
  ∃ (Q : Polynomial ℚ), (∃ (R : Polynomial ℚ) (hR : R.degree = 3), P(Q) = P * R) ∧ 
  Finset.card { t | ∃ i j k, t = (Q.eval i, Q.eval j, Q.eval k) ∧ i ∈ {1, 2, 4} ∧ j ∈ {1, 2, 4} ∧ k ∈ {1, 2, 4} } = 22 := 
sorry

end PolynomialProof

end num_valid_Q_l657_657480


namespace ratio_contribution_l657_657863

theorem ratio_contribution : 
  let Margo_contribution := 4300
  let Julie_contribution := 4700
  let Difference := Julie_contribution - Margo_contribution
  let Total_contribution := Julie_contribution + Margo_contribution 
  ∀ (Ratio : ℚ), 
    Ratio = Difference / Total_contribution → 
    Ratio = 2 / 45 := 
by 
  intros Margo_contribution Julie_contribution Difference Total_contribution Ratio h
  unfold Margo_contribution Julie_contribution Difference Total_contribution at *
  -- applying arithmetic operations and simplifying the ratio
  sorry

end ratio_contribution_l657_657863


namespace configure_friendship_groups_l657_657227

theorem configure_friendship_groups (num_individuals friends_options: ℕ) (h1: num_individuals = 8) (h2: friends_options = 6) : 
  let configurations := 385
  in configurations = 385 :=
by 
  sorry

end configure_friendship_groups_l657_657227


namespace sum_of_coefficients_eq_39_l657_657549

noncomputable def g (x : ℝ) : ℝ :=
  x^4 - 2 * x^3 + 14 * x^2 - 18 * x + 45

theorem sum_of_coefficients_eq_39 :
  let a := -2
  let b := 14
  let c := -18
  let d := 45
  a + b + c + d = 39 :=
by
  let a := -2
  let b := 14
  let c := -18
  let d := 45
  calc
    a + b + c + d = -2 + 14 + (-18) + 45 : by rfl
               ... = 39 : by ring

end sum_of_coefficients_eq_39_l657_657549


namespace birthday_stickers_l657_657869

def stickers_mika_had : Nat := 20
def stickers_bought : Nat := 26
def stickers_given_away : Nat := 6
def stickers_used_for_decoration : Nat := 58
def stickers_left : Nat := 2

theorem birthday_stickers : ∃ B : Nat, (stickers_mika_had + stickers_bought + B - stickers_given_away - stickers_used_for_decoration = stickers_left) ∧ (B = 20) :=
by
  use 20
  split
  sorry
  rfl

end birthday_stickers_l657_657869


namespace quadratic1_vertex_quadratic1_decreases_when_x_increases_quadratic2_vertex_quadratic2_decreases_when_x_increases_quadratic3_vertex_quadratic3_decreases_when_x_increases_l657_657256

noncomputable def quadratic1 := λ x : ℝ, x^2 + 2*x + 1
noncomputable def quadratic2 := λ x : ℝ, -1/2 * x^2 + 3
noncomputable def quadratic3 := λ x : ℝ, 2 * (x + 1) * (x - 3)

theorem quadratic1_vertex :
  ∃ (h k : ℝ), quadratic1 = λ x, (x + h)^2 + k ∧ (h, k) = (-1, 0) := sorry

theorem quadratic1_decreases_when_x_increases :
  ∀ x : ℝ, x < -1 → quadratic1 (x + 1) < quadratic1 x := sorry

theorem quadratic2_vertex :
  ∃ (h k : ℝ), quadratic2 = λ x, -1/2 * (x - h)^2 + k ∧ (h, k) = (0, 3) := sorry

theorem quadratic2_decreases_when_x_increases :
  ∀ x : ℝ, x > 0 → quadratic2 (x + 1) < quadratic2 x := sorry

theorem quadratic3_vertex :
  ∃ (h k : ℝ), quadratic3 = λ x, 2 * ((x - h) * (x - 1)) ∧ (h, k) = (1, -8) := sorry

theorem quadratic3_decreases_when_x_increases :
  ∀ x : ℝ, x < 1 → quadratic3 (x + 1) < quadratic3 x := sorry

end quadratic1_vertex_quadratic1_decreases_when_x_increases_quadratic2_vertex_quadratic2_decreases_when_x_increases_quadratic3_vertex_quadratic3_decreases_when_x_increases_l657_657256


namespace count_negative_values_of_x_l657_657310

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l657_657310


namespace walnut_trees_in_park_l657_657946

theorem walnut_trees_in_park : (current_trees : ℕ) → (new_trees : ℕ) → current_trees = 4 → new_trees = 6 → (current_trees + new_trees) = 10 :=
by
  intros current_trees new_trees hcurrent hnew
  rw [hcurrent, hnew]
  norm_num

end walnut_trees_in_park_l657_657946


namespace binomial_sum_identity_l657_657881

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_sum_identity (m t : ℕ) (hm : 0 < m) (ht : 0 < t) :
  (∑ k in Finset.range (m + 1), binomial m k * binomial (t + k) m) =
  (∑ k in Finset.range (m + 1), binomial m k * binomial t k * 2 ^ k) :=
by
  sorry

end binomial_sum_identity_l657_657881


namespace incenter_coordinates_l657_657467

noncomputable def incenter (d e f : ℝ) (D E F : ℝ × ℝ) := 
  (d * D.1 + e * E.1 + f * F.1) / (d + e + f),
  (d * D.2 + e * E.2 + f * F.2) / (d + e + f)

theorem incenter_coordinates 
  (D E F : ℝ × ℝ)
  (d e f : ℝ)
  (J : ℝ × ℝ)
  (hd : d = 5)
  (he : e = 8)
  (hf : f = 6)
  (hJ : J = incenter d e f D E F) :
  J = (5 / 19 * D.1 + 8 / 19 * E.1 + 6 / 19 * F.1, 
       5 / 19 * D.2 + 8 / 19 * E.2 + 6 / 19 * F.2) :=
by
  rw [hd, he, hf] at hJ
  simp [incenter, hd, he, hf] at hJ
  exact hJ

end incenter_coordinates_l657_657467


namespace computation_problem_points_l657_657645

/-- A teacher gives out a test of 30 problems. Each computation problem is worth some points, and
each word problem is worth 5 points. The total points you can receive on the test is 110 points,
and there are 20 computation problems. How many points is each computation problem worth? -/

theorem computation_problem_points (x : ℕ) (total_problems : ℕ := 30) (word_problem_points : ℕ := 5)
    (total_points : ℕ := 110) (computation_problems : ℕ := 20) :
    20 * x + (total_problems - computation_problems) * word_problem_points = total_points → x = 3 :=
by
  intro h
  sorry

end computation_problem_points_l657_657645


namespace quadratic_solutions_l657_657943

theorem quadratic_solutions (x : ℝ) : x * (x - 1) = 1 - x ↔ x = 1 ∨ x = -1 :=
by
  sorry

end quadratic_solutions_l657_657943


namespace num_of_negative_x_l657_657293

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l657_657293


namespace negative_values_count_l657_657320

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l657_657320


namespace count_negative_values_of_x_l657_657307

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l657_657307


namespace number_of_women_in_first_class_l657_657092

theorem number_of_women_in_first_class (total_passengers : ℕ) (percentage_women : ℕ) (percentage_first_class : ℕ) 
  (h_total : total_passengers = 300) 
  (h_percentage_women : percentage_women = 75) 
  (h_percentage_first_class : percentage_first_class = 15) :
  let num_women := total_passengers * percentage_women / 100 in
  let num_women_first_class := num_women * percentage_first_class / 100 in
  num_women_first_class = 34 :=
by
  sorry

end number_of_women_in_first_class_l657_657092


namespace solve_for_unknown_l657_657172

theorem solve_for_unknown :
  let prc1 := 0.375 * 1500
  let prc2 := 0.625 * 800
  let subtraction := prc1 - prc2
  let equation := (45 * ? * 4 / 3)
  subtraction = 62.5 → equation = 62.5 → ? = 1.0417 :=
begin
  -- let definitions and assumptions
  let prc1 := 0.375 * 1500,
  let prc2 := 0.625 * 800,
  let subtraction := prc1 - prc2,
  let equation := (45 * ? * 4 / 3),
  assume h1 : subtraction = 62.5,
  assume h2 : equation = 62.5,
  -- prove the final result
  sorry
end

end solve_for_unknown_l657_657172


namespace min_non_movable_tiles_l657_657912

-- Define the conditions of the problem
def grid := (8, 8)
def tile := (2, 1) ∨ (1, 2)

-- The required proof problem statement
theorem min_non_movable_tiles : ∃ n, n = 28 ∧
  ∀ (placement : list ((ℕ × ℕ) × (ℕ × ℕ))), placement.size = n ∧
    (∀ t ∈ placement, (fst t).fst < 8 ∧ (fst t).snd < 8 ∧
                      (snd t).fst < 8 ∧ (snd t).snd < 8 ∧
                      ((abs ((snd t).fst - (fst t).fst) = 1 ∧ ((snd t).snd - (fst t).snd) = 0) ∨
                       (abs ((snd t).snd - (fst t).snd) = 1 ∧ ((snd t).fst - (fst t).fst) = 0))) →
    (∀ t1 t2 ∈ placement, t1 ≠ t2 → (fst t1 ≠ fst t2 ∧ fst t1 ≠ snd t2 ∧ snd t1 ≠ fst t2 ∧ snd t1 ≠ snd t2)) : sorry

end min_non_movable_tiles_l657_657912


namespace count_negative_x_with_sqrt_pos_int_l657_657351

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l657_657351


namespace sum_of_numbers_with_lcm_and_ratio_l657_657904

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ) (h_lcm : Nat.lcm a b = 60) (h_ratio : a = 2 * b / 3) : a + b = 50 := 
by
  sorry

end sum_of_numbers_with_lcm_and_ratio_l657_657904


namespace train_length_correct_l657_657203

def initial_speed_train_kmph : ℝ := 100
def acceleration_train_mss : ℝ := 0.5
def initial_speed_bike_kmph : ℝ := 64
def acceleration_bike_mss : ℝ := 0.3
def time_seconds : ℝ := 85

-- Convert initial speeds from km/h to m/s
def initial_speed_train_mps : ℝ := (initial_speed_train_kmph * 1000) / 3600
def initial_speed_bike_mps : ℝ := (initial_speed_bike_kmph * 1000) / 3600

-- Calculate distances covered in 85 seconds
def distance_train : ℝ := initial_speed_train_mps * time_seconds + (1/2) * acceleration_train_mss * time_seconds^2
def distance_bike : ℝ := initial_speed_bike_mps * time_seconds + (1/2) * acceleration_bike_mss * time_seconds^2

-- Define the expected length of the train
def length_train : ℝ := distance_train - distance_bike

theorem train_length_correct : length_train = 1572.5 :=
by
  -- Skipping the proof, as it's not needed
  sorry

end train_length_correct_l657_657203


namespace complex_value_l657_657153

theorem complex_value (x y : ℝ) : (i + (x + y * complex.I)) * (i + (x - y * complex.I)) = 4 + 4 * complex.I :=
by
  sorry

end complex_value_l657_657153


namespace intersection_eq_l657_657023

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_eq : M ∩ N = {5, 7, 9} := sorry

end intersection_eq_l657_657023


namespace omega_range_for_monotonically_decreasing_l657_657718

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem omega_range_for_monotonically_decreasing
  (ω : ℝ)
  (hω : ω > 0)
  (h_decreasing : ∀ x ∈ Set.Ioo (Real.pi / 2) Real.pi, f ω x < f ω (x + 1e-6)) :
  1/2 ≤ ω ∧ ω ≤ 5/4 :=
by
  sorry

end omega_range_for_monotonically_decreasing_l657_657718


namespace parallelogram_area_l657_657074

theorem parallelogram_area :
  ∀ (A B C D : Type) [EuclideanGeometry A B C D],
    ∀ (AB AD : ℝ) (angle_BAD : ℝ) (area : ℝ),
      AB = 12 ∧ AD = 10 ∧ angle_BAD = 150 ∧ parallelogram A B C D →
      area = 60 :=
by sorry

end parallelogram_area_l657_657074


namespace negative_values_count_l657_657321

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l657_657321


namespace solve_x_l657_657380

theorem solve_x (x : ℝ) (h : (x - 1)^2 = 4) : x = 3 ∨ x = -1 :=
by
  -- proof goes here
  sorry

end solve_x_l657_657380


namespace product_all_possible_values_of_c_l657_657683

noncomputable def g (c : ℝ) (x : ℝ) : ℝ := c / (3 * x - 4)

theorem product_all_possible_values_of_c : 
  let c_values := {c : ℝ | g c 3 = g⁻¹ c (c+2)} in
  ∏ c in c_values, c = -8 / 3 :=
sorry

end product_all_possible_values_of_c_l657_657683


namespace intersection_eq_l657_657026

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_eq : M ∩ N = {5, 7, 9} := sorry

end intersection_eq_l657_657026


namespace angle_B_degrees_l657_657466

theorem angle_B_degrees (A B C : ℕ) (h1 : A < B) (h2 : B < C) (h3 : 4 * C = 7 * A) (h4 : A + B + C = 180) : B = 59 :=
sorry

end angle_B_degrees_l657_657466


namespace actual_area_from_map_scale_l657_657938

theorem actual_area_from_map_scale (scale : ℕ) (area_on_map : ℕ) (h_scale : scale = 10000) (h_area_on_map : area_on_map = 10) :
  let actual_area := (area_on_map * scale^2) / 10^4 in
  actual_area = 100000 := by
sorry

end actual_area_from_map_scale_l657_657938


namespace prove_sample_max_eq_thirteen_l657_657812

noncomputable def sample_max (x : Fin 5 → ℝ) := 
  ∃ x_max : ℝ, 
    (∀ i j : Fin 5, x i ≠ x j → x i ∈ x → x j ∈ x → x i ≠ x j) ∧ -- Distinct values
    (∑ i, x i / (5:ℝ)) = 7 ∧ -- Sample mean
    (∑ i, (x i - 7) ^ 2 / (4:ℝ)) = 4 ∧ -- Sample variance
    x_max = 13 ∧ -- Maximum value is 13
    ∃ i, x i = x_max

theorem prove_sample_max_eq_thirteen :
  ∀ x : Fin 5 → ℝ, sample_max x :=
begin
  sorry
=end

end prove_sample_max_eq_thirteen_l657_657812


namespace trapezoid_area_l657_657649

theorem trapezoid_area (x y : ℝ) (hx : y^2 + x^2 = 625) (hy : y^2 + (25 - x)^2 = 900) :
  1 / 2 * (11 + 36) * 24 = 564 :=
by
  sorry

end trapezoid_area_l657_657649


namespace product_of_all_possible_values_of_c_l657_657685

theorem product_of_all_possible_values_of_c
    (g : ℝ → ℝ := λ x, c / (3 * x - 4))
    (h_inv : ∀ y, g (g⁻¹ y) = y)
    (h : g 3 = g⁻¹ (c + 2)) :
  ∀ c ∈ {c | g 3 = g⁻¹ (c + 2)}, ∏ c = -40 / 3 := by
  sorry

end product_of_all_possible_values_of_c_l657_657685


namespace find_m_value_l657_657510

def magic_box_output (a b : ℝ) : ℝ := a^2 + b - 1

theorem find_m_value :
  ∃ m : ℝ, (magic_box_output m (-2 * m) = 2) ↔ (m = 3 ∨ m = -1) :=
by
  sorry

end find_m_value_l657_657510


namespace convex_polygon_in_triangle_l657_657365

theorem convex_polygon_in_triangle (P : set (ℝ × ℝ)) (hP_convex : convex P)
  (hP_condition : ∀ T : triangle, T ⊆ P → area T < 1) :
  ∃ T' : triangle, P ⊆ T' ∧ area T' ≤ 4 :=
sorry

end convex_polygon_in_triangle_l657_657365


namespace integer_sum_of_squares_power_l657_657082

theorem integer_sum_of_squares_power (a p q : ℤ) (k : ℕ) (h : a = p^2 + q^2) : 
  ∃ c d : ℤ, a^k = c^2 + d^2 := 
sorry

end integer_sum_of_squares_power_l657_657082


namespace count_negative_values_correct_l657_657280

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l657_657280


namespace distance_behind_l657_657158

-- Given conditions
variables {A B E : ℝ} -- Speed of Anusha, Banu, and Esha
variables {Da Db De : ℝ} -- distances covered by Anusha, Banu, and Esha

axiom const_speeds : Da = 100 ∧ Db = 90 ∧ Db / Da = De / Db ∧ De = 90 * (Db / 100)

-- The proof to be established
theorem distance_behind (h : Da = 100 ∧ Db = 90 ∧ Db / Da = De / Db ∧ De = 90 * (Db / 100)) :
  100 - De = 19 :=
by sorry

end distance_behind_l657_657158


namespace negative_values_count_l657_657325

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l657_657325


namespace count_quadratic_polynomials_l657_657482

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 4)

theorem count_quadratic_polynomials (Q : ℝ → ℝ) :
  (∃ (R : ℝ → ℝ), degree R = 3 ∧ ∀ x, P (Q x) = P x * R x) ↔ 
  (∃ qps : Finset (ℝ → ℝ), qps.card = 4 ∧ (∀ q ∈ qps, degree q = 2)) := sorry

end count_quadratic_polynomials_l657_657482


namespace randy_money_left_after_expenses_l657_657086

theorem randy_money_left_after_expenses : 
  ∀ (initial_money lunch_cost : ℕ) (ice_cream_fraction : ℚ), 
  initial_money = 30 → 
  lunch_cost = 10 → 
  ice_cream_fraction = 1 / 4 → 
  let post_lunch_money := initial_money - lunch_cost in
  let ice_cream_cost := ice_cream_fraction * post_lunch_money in
  let money_left := post_lunch_money - ice_cream_cost in
  money_left = 15 :=
by
  intros initial_money lunch_cost ice_cream_fraction
  assume h_initial h_lunch h_fraction
  let post_lunch_money := initial_money - lunch_cost
  let ice_cream_cost := ice_cream_fraction * post_lunch_money
  let money_left := post_lunch_money - ice_cream_cost
  sorry

end randy_money_left_after_expenses_l657_657086


namespace bob_distance_when_meet_l657_657877

variable (d : ℕ) (t : ℕ) (yolanda_rate : ℕ) (bob_rate : ℕ) (distance : ℕ)

theorem bob_distance_when_meet :
  yolanda_rate = 3 →
  bob_rate = 4 →
  distance = 31 →
  d = distance - yolanda_rate →
  7 * t = d →
  t = 4 →
  bob_rate * t = 16 := by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end bob_distance_when_meet_l657_657877


namespace net_population_increase_one_day_l657_657162

theorem net_population_increase_one_day:
  (birth_rate: ℕ → ℕ) (death_rate: ℕ → ℕ) (n_sec: ℕ)
  (h1: birth_rate 2 = 5) (h2: death_rate 2 = 3) 
  (h3: n_sec = 24 * 60 * 60):
  let net_increase_per_sec := (birth_rate 2 - death_rate 2) / 2
  in net_increase_per_sec * n_sec = 86400 := by
  sorry

end net_population_increase_one_day_l657_657162


namespace problem_rewrite_equation_l657_657676

theorem problem_rewrite_equation :
  ∃ a b c : ℤ, a > 0 ∧ (64*(x^2) + 96*x - 81 = 0) → ((a*x + b)^2 = c) ∧ (a + b + c = 131) :=
sorry

end problem_rewrite_equation_l657_657676


namespace gallons_per_hour_l657_657130

-- Define conditions
def total_runoff : ℕ := 240000
def days : ℕ := 10
def hours_per_day : ℕ := 24

-- Define the goal: proving the sewers handle 1000 gallons of run-off per hour
theorem gallons_per_hour : (total_runoff / (days * hours_per_day)) = 1000 :=
by
  -- Proof can be inserted here
  sorry

end gallons_per_hour_l657_657130


namespace OI_perp_MN_l657_657607

open EuclideanGeometry

noncomputable def perp_circumcenter_incenter_midpoints 
  (A B C O I D E F P Q M N : Point) : Prop :=
  circumcenter O (triangle A B C) ∧ 
  incenter I (triangle A B C) ∧ 
  incircle_touching (triangle A B C) D E F ∧ 
  intersect_line_at FD CA P ∧ 
  intersect_line_at DE AB Q ∧
  is_midpoint M (segment P E) ∧
  is_midpoint N (segment Q F) →
  perp (line O I) (line M N)

theorem OI_perp_MN 
  (A B C O I D E F P Q M N : Point)
  (h1 : circumcenter O (triangle A B C))
  (h2 : incenter I (triangle A B C))
  (h3 : incircle_touching (triangle A B C) D E F)
  (h4 : intersect_line_at FD CA P)
  (h5 : intersect_line_at DE AB Q)
  (h6 : is_midpoint M (segment P E))
  (h7 : is_midpoint N (segment Q F)) :
  perp (line O I) (line M N) :=
begin
  sorry
end

end OI_perp_MN_l657_657607


namespace intersection_M_N_l657_657042

def M := {1, 3, 5, 7, 9}

def N := {x : ℤ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} := by
  sorry

end intersection_M_N_l657_657042


namespace curve_intersection_R_range_l657_657548

theorem curve_intersection_R_range :
  (∃ (R : ℝ) (α : ℝ), R > 0 ∧
    (∃ (x y : ℝ), x = 2 + (sin α)^2 ∧ y = (sin α)^2 ∧
      x^2 + y^2 = R^2)) →
  (2 ≤ R ∧ R ≤ sqrt 10) :=
begin
  sorry
end

end curve_intersection_R_range_l657_657548


namespace math_problem_l657_657053

noncomputable def a_b_sum (x y z a b : ℝ) (T : set (ℝ × ℝ × ℝ)) : Prop :=
  (∀ (x y z : ℝ), (x, y, z) ∈ T → (log x + log y = z) ∧ (log (x^2 + y^2) = z + 2) ∧ 
    (x^3 + y^3 = a * 2^(3 * z) + b * 2^(2 * z))) →
  (a + b = 22)

theorem math_problem : ∃ (a b : ℝ), 
  let T := {p : ℝ × ℝ × ℝ | (∃ (x y z : ℝ), p = (x, y, z) ∧ 
    log (x + y) = z ∧ log (x^2 + y^2) = z + 2 ∧ 
    x^3 + y^3 = a * 2^(3 * z) + b * 2^(2 * z))} 
  in a_b_sum 0 0 0 a b T
:= sorry

end math_problem_l657_657053


namespace range_of_a_l657_657545

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -x^2 + 2 * x + 3 ≤ a^2 - 3 * a) ↔ (a ≤ -1 ∨ a ≥ 4) := by
  sorry

end range_of_a_l657_657545


namespace min_value_x_plus_inv_x_l657_657848

open Real

theorem min_value_x_plus_inv_x (x : ℝ) (hx : 0 < x) : x + 1/x ≥ 2 := by
  sorry

end min_value_x_plus_inv_x_l657_657848


namespace intersection_M_N_l657_657049

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657049


namespace tshirt_cost_l657_657667

theorem tshirt_cost (initial_amount sweater_cost shoes_cost amount_left spent_on_tshirt : ℕ) 
  (h_initial : initial_amount = 91) 
  (h_sweater : sweater_cost = 24) 
  (h_shoes : shoes_cost = 11) 
  (h_left : amount_left = 50)
  (h_spent : spent_on_tshirt = initial_amount - amount_left - sweater_cost - shoes_cost) :
  spent_on_tshirt = 6 :=
sorry

end tshirt_cost_l657_657667


namespace intersection_correct_l657_657018

-- Define the sets M and N
def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x | 2 * x > 7}

-- Define the expected intersection result
def expected_intersection : Set ℝ := {5, 7, 9}

-- State the theorem
theorem intersection_correct : ∀ x, x ∈ M ∩ N ↔ x ∈ expected_intersection :=
by
  sorry

end intersection_correct_l657_657018


namespace probability_of_die_showing_1_after_5_steps_l657_657673

def prob_showing_1 (steps : ℕ) : ℚ :=
  if steps = 5 then 37 / 192 else 0

theorem probability_of_die_showing_1_after_5_steps :
  prob_showing_1 5 = 37 / 192 :=
sorry

end probability_of_die_showing_1_after_5_steps_l657_657673


namespace bisect_line_through_center_l657_657093

theorem bisect_line_through_center (A B C D O E F : Point) (h1 : Rectangle A B C D)
  (h2 : Midpoint O A C) (h3 : Line_through O F E) (h4 : Intersect F A B) (h5 : Intersect E D C) :
  Seg_eq O E O F :=
by
  sorry

end bisect_line_through_center_l657_657093


namespace sum_of_coordinates_l657_657752

-- Given conditions
variables {g : ℝ → ℝ} {h : ℝ → ℝ}
axiom g_at_4 : g 4 = -5
noncomputable def h := λ x, (g x) ^ 3

-- Theorem to prove
theorem sum_of_coordinates : (4 + h 4 = -121) := by
  sorry

end sum_of_coordinates_l657_657752


namespace reduce_to_integral_l657_657525

noncomputable def boundary_value_problem (f : ℝ → ℝ → ℝ) (y : ℝ → ℝ) := 
  y'' = f x (y x) ∧ y 0 = 0 ∧ y 1 = 0

noncomputable def Greens_function (x ξ : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ ξ then (1 - ξ) * x 
  else if ξ ≤ x ∧ x ≤ 1 then (x - 1) * ξ 
  else 0

theorem reduce_to_integral (f : ℝ → ℝ → ℝ) (y : ℝ → ℝ) : 
  boundary_value_problem f y -> 
  (∀ x : ℝ, y x = ∫ ξ in 0..1, Greens_function x ξ * f ξ (y ξ)) :=
by sorry

end reduce_to_integral_l657_657525


namespace symbiotic_not_pair_neg2_1_symbiotic_pair_4_3_5_symbiotic_6_a_symbiotic_neg_pair_l657_657872

def symbiotic_rational (a b : ℚ) : Prop := a - b = a * b + 1

theorem symbiotic_not_pair_neg2_1 : ¬ symbiotic_rational (-2) (1) :=
by {
  unfold symbiotic_rational,
  simp,
  norm_num,
  intro h, -- Proof by contradiction
  exact h rfl
}

theorem symbiotic_pair_4_3_5 : symbiotic_rational (4) (3 / 5) :=
by {
  unfold symbiotic_rational,
  norm_num
}

theorem symbiotic_6_a (a : ℚ) (h : symbiotic_rational 6 a) : a = 5 / 7 :=
by {
  unfold symbiotic_rational at h,
  simp at h,
  linarith
}

theorem symbiotic_neg_pair (m n : ℚ) (h : symbiotic_rational m n) : symbiotic_rational (-n) (-m) :=
by {
  unfold symbiotic_rational at *,
  rw h,
  ring
}

end symbiotic_not_pair_neg2_1_symbiotic_pair_4_3_5_symbiotic_6_a_symbiotic_neg_pair_l657_657872


namespace max_leftover_candies_l657_657774

-- Given conditions as definitions
def pieces_of_candy := ℕ
def num_bags := 11

-- Statement of the problem
theorem max_leftover_candies (x : pieces_of_candy) (h : x % num_bags ≠ 0) :
  x % num_bags = 10 :=
sorry

end max_leftover_candies_l657_657774


namespace minimum_sum_reciprocal_l657_657358

theorem minimum_sum_reciprocal (a b x y : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_geom_mean : a * b = 2) 
  (h_log_ax : log a x = 3) (h_log_by : log b y = 3) : (1/x + 1/y) = sqrt 2 / 2 :=
by
  sorry

end minimum_sum_reciprocal_l657_657358


namespace tan_75_eq_2_plus_sqrt_3_l657_657133

theorem tan_75_eq_2_plus_sqrt_3 : Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := 
sorry

end tan_75_eq_2_plus_sqrt_3_l657_657133


namespace intersection_M_N_l657_657050

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657050


namespace at_least_two_relatively_prime_at_least_two_not_relatively_prime_l657_657782

-- Define the set of integers between 90 and 99
def set_of_integers : Set ℤ := {90, 91, 92, 93, 94, 95, 96, 97, 98, 99}

-- Define the main function that takes a list of six integers
def chosen_integers (lst : List ℤ) : Prop :=
  lst.length = 6 ∧ lst ⊆ set_of_integers 

-- Part (a): Prove at least two of the integers are relatively prime
theorem at_least_two_relatively_prime (lst : List ℤ) (h : chosen_integers lst) :
  ∃ a b, a ∈ lst ∧ b ∈ lst ∧ a ≠ b ∧ Int.gcd a b = 1 :=
by
  sorry

-- Part (b): Prove at least two of the integers are not relatively prime
theorem at_least_two_not_relatively_prime (lst : List ℤ) (h : chosen_integers lst) :
  ∃ a b, a ∈ lst ∧ b ∈ lst ∧ a ≠ b ∧ Int.gcd a b ≠ 1 :=
by
  sorry

end at_least_two_relatively_prime_at_least_two_not_relatively_prime_l657_657782


namespace price_of_large_pizza_l657_657934

variable {price_small_pizza : ℕ}
variable {total_revenue : ℕ}
variable {small_pizzas_sold : ℕ}
variable {large_pizzas_sold : ℕ}
variable {price_large_pizza : ℕ}

theorem price_of_large_pizza
  (h1 : price_small_pizza = 2)
  (h2 : total_revenue = 40)
  (h3 : small_pizzas_sold = 8)
  (h4 : large_pizzas_sold = 3) :
  price_large_pizza = 8 :=
by
  sorry

end price_of_large_pizza_l657_657934


namespace parallelogram_area_60_l657_657076

theorem parallelogram_area_60
  (α β : ℝ) (a b : ℝ)
  (h_angle : α = 150) 
  (h_adj_angle : β = 180 - α) 
  (h_len_1 : a = 10)
  (h_len_2 : b = 12) :
  ∃ (area : ℝ), area = 60 := 
by 
  use 60
  sorry

end parallelogram_area_60_l657_657076


namespace doubling_people_halves_time_l657_657163

theorem doubling_people_halves_time (P : ℕ) (d : ℕ) (h : d = 20) : (2 * P) * (d / 4) = P * (d / 2) :=
by 
  rw h
  sorry

end doubling_people_halves_time_l657_657163


namespace intersection_proof_l657_657030

noncomputable def M : Set ℕ := {1, 3, 5, 7, 9}

noncomputable def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_proof : M ∩ (N ∩ Set.univ) = {5, 7, 9} :=
by sorry

end intersection_proof_l657_657030


namespace intersection_M_N_l657_657037

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := { x | 2 * x > 7 }

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l657_657037


namespace value_of_bill_used_to_pay_l657_657824

-- Definitions of the conditions
def num_games : ℕ := 6
def cost_per_game : ℕ := 15
def num_change_bills : ℕ := 2
def change_per_bill : ℕ := 5
def total_cost : ℕ := num_games * cost_per_game
def total_change : ℕ := num_change_bills * change_per_bill

-- Proof statement: What was the value of the bill Jed used to pay
theorem value_of_bill_used_to_pay : 
  total_value = (total_cost + total_change) :=
by
  sorry

end value_of_bill_used_to_pay_l657_657824


namespace quadrilateral_is_square_l657_657693

noncomputable def is_square : Prop :=
  let A := (0, 3)
  let B := (0, -3)
  let C := (3, 0)
  let D := (-3, 0)
  (dist A B = 6) ∧ 
  (dist A D = 3*Real.sqrt 2) ∧ 
  (dist B C = 3*Real.sqrt 2) ∧ 
  (dist C D = 6) ∧
  (dist A C = 3*Real.sqrt 2) ∧
  (dist B D = 3*Real.sqrt 2)

theorem quadrilateral_is_square :
  is_square :=
begin
  -- proof will go here
  sorry
end

end quadrilateral_is_square_l657_657693


namespace customer_paid_l657_657128

theorem customer_paid : 
  ∀ (cost_price : ℕ) (markup_rate : ℕ), 
    markup_rate = 20 → cost_price = 7000 → 
    let extra_cost := (markup_rate * cost_price) / 100 in
    let total_cost := cost_price + extra_cost in
    total_cost = 8400 :=
by 
  intros cost_price markup_rate h1 h2
  let extra_cost := (markup_rate * cost_price) / 100 
  let total_cost := cost_price + extra_cost
  have h3 : cost_price = 7000 := h2
  have h4 : markup_rate = 20 := h1
  sorry

end customer_paid_l657_657128


namespace cycle_final_selling_price_l657_657630

-- Lean 4 statement capturing the problem definition and final selling price
theorem cycle_final_selling_price (original_price : ℝ) (initial_discount_rate : ℝ) 
  (loss_rate : ℝ) (exchange_discount_rate : ℝ) (final_price : ℝ) :
  original_price = 1400 →
  initial_discount_rate = 0.05 →
  loss_rate = 0.25 →
  exchange_discount_rate = 0.10 →
  final_price = 
    (original_price * (1 - initial_discount_rate) * (1 - loss_rate) * (1 - exchange_discount_rate)) →
  final_price = 897.75 :=
by
  sorry

end cycle_final_selling_price_l657_657630


namespace three_colored_vertices_exist_l657_657478

theorem three_colored_vertices_exist :
  ∀ (E : fin 5 → fin 5 → bool), ∃ (a b k : ℕ), 
  1 ≤ a ∧ a ≤ 5 ∧ 1 ≤ b ∧ b ≤ 5 ∧ 1 ≤ k ∧ a + k ≤ 5 ∧ b + k ≤ 5 ∧ 
  ((E a b = E (a+k) b ∧ E (a+k) b = E (a+k) (b+k) ∧ E (a+k) (b+k) = E a b) ∨
   (E a b = E a (b+k) ∧ E a (b+k) = E (a+k) (b+k) ∧ E (a+k) (b+k) = E a b)) :=
by sorry

end three_colored_vertices_exist_l657_657478


namespace odd_function_of_additive_l657_657367

theorem odd_function_of_additive (f : ℝ → ℝ) (h : ∀ x y : ℝ, f(x) + f(y) = f(x+y)) : 
  ∀ x : ℝ, f(-x) = -f(x) :=
by { sorry }

end odd_function_of_additive_l657_657367


namespace area_of_triangle_DEF_l657_657265

theorem area_of_triangle_DEF : 
  ∀ (DF EF DE : ℝ), 
  DF = 4 → 
  EF = 4 → 
  DE = DF * Real.sqrt 2 → 
  ∠ DEF = 45 ° → 
  (1/2) * DF * EF = 8 :=
by
  intros DF EF DE hDF hEF hDE h_angle 
  sorry

end area_of_triangle_DEF_l657_657265


namespace change_in_expression_correct_l657_657697

variable (x b a : ℝ)

def change_in_expression (x b a : ℝ) : ℝ :=
  let increase := (x + a)^2 - b * (x + a) - 3
  let decrease := (x - a)^2 - b * (x - a) - 3
  let original := x^2 - b * x - 3
  if a > 0 then
    change_when_increase := increase - original
    change_when_decrease := decrease - original
    change_in_expression := change_when_increase = (2 * a * x + a^2 - b * a) ∧
                                         change_when_decrease = (-2 * a * x + a^2 + b * a)
  else
    change_in_expression := false

theorem change_in_expression_correct (x b a : ℝ) (h : a > 0) :
  change_in_expression x b a := sorry

end change_in_expression_correct_l657_657697


namespace omega_range_for_monotonically_decreasing_l657_657717

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem omega_range_for_monotonically_decreasing
  (ω : ℝ)
  (hω : ω > 0)
  (h_decreasing : ∀ x ∈ Set.Ioo (Real.pi / 2) Real.pi, f ω x < f ω (x + 1e-6)) :
  1/2 ≤ ω ∧ ω ≤ 5/4 :=
by
  sorry

end omega_range_for_monotonically_decreasing_l657_657717


namespace count_negative_x_with_sqrt_pos_int_l657_657343

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l657_657343


namespace general_formula_minimum_value_n_l657_657857

variable (a : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℝ)

-- Define Sn
axiom Sn_def : ∀ n, S n = 2 * a n - a 1

-- Define that a1, a2+1, a3 forms an arithmetic sequence
axiom arithmetic_seq : a 1 + a 3 = 2 * (a 2 + 1)

-- Define general formula for a_n
theorem general_formula : ∀ n, a n = 2 ^ n := sorry

-- Define sum of the sequence of reciprocals
axiom Tn_def : ∀ n, T n = ∑ i in Finset.range n, 1 / (a i).toReal

-- Find the minimum value of n for which |Tn - 1| < 1/1000
theorem minimum_value_n : ∃ n, |T n - 1| < 1 / 1000 ∧ ∀ m, (|T m - 1| < 1 / 1000) → m ≥ 10 :=
  sorry

end general_formula_minimum_value_n_l657_657857


namespace factorial_mod_17_l657_657279

theorem factorial_mod_17 :
  (13! % 17) = 3 :=
by
  sorry

end factorial_mod_17_l657_657279


namespace geom_seq_general_term_b_seq_arithmetic_l657_657368

-- Given conditions translated to Lean definitions
def geometric_seq (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) :=
  a 1 = a1 ∧ ∃ (k : ℕ), ∀ n : ℕ, a (n + 1) = q * a n

def arithmetic_mean (x y z : ℝ) :=
  2 * x = y + z

def quadratic_eq (t : ℝ) (b : ℕ → ℝ) :=
  ∀ n : ℕ, 2 * (n : ℝ)^2 - (t + b n) * (n : ℝ) + (3/2) * b n = 0

-- Main statements to be proved 
theorem geom_seq_general_term (a : ℕ → ℝ) (q : ℝ) (h : geometric_seq a 2 q) (h_arith : arithmetic_mean (3 * a 3) (8 * a 1) (a 5)) : 
  a = (λ n, 2^n) :=
sorry

theorem b_seq_arithmetic (t : ℝ) (b : ℕ → ℝ) (h_quad : quadratic_eq t b) : t = 3 :=
sorry

end geom_seq_general_term_b_seq_arithmetic_l657_657368


namespace randy_money_left_l657_657085

theorem randy_money_left (initial_money lunch ice_cream_cone remaining : ℝ) 
  (h1 : initial_money = 30)
  (h2 : lunch = 10)
  (h3 : remaining = initial_money - lunch)
  (h4 : ice_cream_cone = remaining * (1/4)) :
  (remaining - ice_cream_cone) = 15 := by
  sorry

end randy_money_left_l657_657085


namespace extra_flowers_l657_657473

theorem extra_flowers (tulips roses used : ℕ) (h1 : tulips = 36) (h2 : roses = 37) (h3 : used = 70) : tulips + roses - used = 3 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end extra_flowers_l657_657473


namespace goods_train_length_is_280_l657_657631

noncomputable def length_of_goods_train (passenger_speed passenger_speed_kmh: ℝ) 
                                       (goods_speed goods_speed_kmh: ℝ) 
                                       (time_to_pass: ℝ) : ℝ :=
  let kmh_to_ms := (1000 : ℝ) / (3600 : ℝ)
  let passenger_speed_ms := passenger_speed * kmh_to_ms
  let goods_speed_ms     := goods_speed * kmh_to_ms
  let relative_speed     := passenger_speed_ms + goods_speed_ms
  relative_speed * time_to_pass

theorem goods_train_length_is_280 :
  length_of_goods_train 70 70 42 42 9 = 280 :=
by
  sorry

end goods_train_length_is_280_l657_657631


namespace cars_with_air_bags_l657_657069

/--
On a car lot with 65 cars:
- Some have air-bags.
- 30 have power windows.
- 12 have both air-bag and power windows.
- 2 have neither air-bag nor power windows.

Prove that the number of cars with air-bags is 45.
-/
theorem cars_with_air_bags 
    (total_cars : ℕ)
    (cars_with_power_windows : ℕ)
    (cars_with_both : ℕ)
    (cars_with_neither : ℕ)
    (total_cars_eq : total_cars = 65)
    (cars_with_power_windows_eq : cars_with_power_windows = 30)
    (cars_with_both_eq : cars_with_both = 12)
    (cars_with_neither_eq : cars_with_neither = 2) :
    ∃ (A : ℕ), A = 45 :=
by
  sorry

end cars_with_air_bags_l657_657069


namespace minimum_at_x_eq_3_l657_657398

-- Define the function f
def f (x : ℝ) : ℝ := x + 1 / (x - 2)

-- Define the condition x > 2
def domain (x : ℝ) : Prop := x > 2

-- State the theorem that the minimum value of f(x) is obtained at x = 3
theorem minimum_at_x_eq_3 : ∀ x, domain x → ∃ a : ℝ, (∀ y, domain y → f(y) ≥ f(a)) ∧ f(a) = f 3 :=
by
  intro x hx
  use 3
  split
  { intro y hy
    -- proof to show that f(y) ≥ f(3)
    sorry },
  -- proof to show f(3) = 4
  sorry

end minimum_at_x_eq_3_l657_657398


namespace intersection_eq_l657_657024

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_eq : M ∩ N = {5, 7, 9} := sorry

end intersection_eq_l657_657024


namespace brian_breath_holding_increase_l657_657217

theorem brian_breath_holding_increase :
  ∀ (initial_week1 week2 final : ℝ), 
  initial = 10 →
  week1 = 2 * initial →
  week2 = 2 * week1 →
  final = 60 →
  ((final - week2) / week2) * 100 = 50 :=
by
  intros initial week1 week2 final h_initial h_week1 h_week2 h_final
  -- Proof is omitted according to the instructions.
  sorry

end brian_breath_holding_increase_l657_657217


namespace Danny_found_11_wrappers_l657_657236

theorem Danny_found_11_wrappers :
  ∃ wrappers_at_park : ℕ,
  (wrappers_at_park = 11) ∧
  (∃ bottle_caps : ℕ, bottle_caps = 12) ∧
  (∃ found_bottle_caps : ℕ, found_bottle_caps = 58) ∧
  (wrappers_at_park + 1 = bottle_caps) :=
by
  sorry

end Danny_found_11_wrappers_l657_657236


namespace correct_statements_are_1_2_3_l657_657654

-- Definitions based on conditions
def synthetic_method := (is_cause_and_effect : Prop) × (is_forward_reasoning : Prop)
def analytical_method := (is_cause_seeking : Prop) × (is_indirect_proof : Prop)
def contradiction_method := (is_backward_reasoning : Prop)

-- Conditions as given in the problem
axiom synthetic_method_def : synthetic_method
axiom analytical_method_def : analytical_method
axiom contradiction_method_def : contradiction_method

-- Correct answer based on solution
theorem correct_statements_are_1_2_3 (c1 : synthetic_method_def.is_cause_and_effect)
  (c2 : synthetic_method_def.is_forward_reasoning)
  (c3 : analytical_method_def.is_cause_seeking)
  (c4 : ¬analytical_method_def.is_indirect_proof)
  (c5 : ¬contradiction_method_def) :
  {1, 2, 3} = { sequence_numbers | sequence_numbers ∈ {1, 2, 3, 4, 5} ∧ 
  ((sequence_numbers = 1 ∧ c1) ∨ 
   (sequence_numbers = 2 ∧ c2) ∨ 
   (sequence_numbers = 3 ∧ c3) ∨ 
   (sequence_numbers = 4 ∧ ¬c4) ∨ 
   (sequence_numbers = 5 ∧ ¬c5)) } := 
by
  sorry

end correct_statements_are_1_2_3_l657_657654


namespace sum_x_coordinates_eq_3_5_l657_657724

noncomputable def f : ℝ → ℝ 
| x <= -2  := -x -3
| -2 < x ∧ x <= -1  := -x - 3
| -1 < x ∧ x <= 1  := 2 * x
| 1 < x ∧ x <= 2  := 2 * x
| 2 < x ∧ x <= 4  := 2 * x - 3
| 4 < x ∧ x <= 5  := 2 * x - 3
| x > 5  := 2 * x - 3

theorem sum_x_coordinates_eq_3_5 : 
  let xs := {x : ℝ | f x = 2} in 
  ∑ x in xs, x = 3.5 := 
sorry

end sum_x_coordinates_eq_3_5_l657_657724


namespace lily_coffee_budget_l657_657506

variable (initial_amount celery_price cereal_original_price bread_price milk_original_price potato_price : ℕ)
variable (cereal_discount milk_discount number_of_potatoes : ℕ)

theorem lily_coffee_budget 
  (h_initial_amount : initial_amount = 60)
  (h_celery_price : celery_price = 5)
  (h_cereal_original_price : cereal_original_price = 12)
  (h_bread_price : bread_price = 8)
  (h_milk_original_price : milk_original_price = 10)
  (h_potato_price : potato_price = 1)
  (h_number_of_potatoes : number_of_potatoes = 6)
  (h_cereal_discount : cereal_discount = 50)
  (h_milk_discount : milk_discount = 10) :
  initial_amount - (celery_price + (cereal_original_price * cereal_discount / 100) + bread_price + (milk_original_price - (milk_original_price * milk_discount / 100)) + (potato_price * number_of_potatoes)) = 26 :=
by
  sorry

end lily_coffee_budget_l657_657506


namespace sequence_formula_sequence_formula_special_case_l657_657190

noncomputable def sequence (K : ℕ) : ℕ → ℕ
| 1       := 1
| (n + 1) := 1 + (K + 1) * (n * (n + 1) / 2)

theorem sequence_formula (K n : ℕ) : sequence K (n + 1) = 1 + (K + 1) * (n * (n + 1) / 2) := 
by
  sorry

theorem sequence_formula_special_case (n : ℕ) : sequence 2 (n + 1) = (3 * (n^2 - n) + 2) / 2 := 
by
  sorry

end sequence_formula_sequence_formula_special_case_l657_657190


namespace intersection_correct_l657_657735

def set_A : Set ℤ := {-1, 1, 2, 4}
def set_B : Set ℤ := {x | |x - 1| ≤ 1}

theorem intersection_correct :
  set_A ∩ set_B = {1, 2} :=
  sorry

end intersection_correct_l657_657735


namespace negative_values_count_l657_657333

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l657_657333


namespace largest_quadrilaterals_with_inscribed_circles_l657_657971

noncomputable def max_inscribed_quadrilaterals (n : ℕ) : ℕ :=
  n / 2

theorem largest_quadrilaterals_with_inscribed_circles (n : ℕ) : 
  (∀ (i : ℤ), 1 ≤ i ∧ i ≤ n - 3 → 
    let A := (A : ℤ → ℤ) in
    (A i + A (i+2) = A (i-1) + A (i+1)) → 
    max_inscribed_quadrilaterals n = n / 2) :=
  by 
    sorry

end largest_quadrilaterals_with_inscribed_circles_l657_657971


namespace correct_operation_is_B_l657_657598

-- Definitions for conditions
def condition_A := ∀ (x : ℝ), x^2 * x^3 = x^6 → false
def condition_B := ∀ (m : ℝ), m ≠ 0 → m^{2020} / m^{2019} = m
def condition_C := ∀ (a : ℝ), (-4 * a^2)^3 = -12 * a^6 → false
def condition_D := 2^(-3) = -8 → false

-- The proof statement
theorem correct_operation_is_B : 
  (condition_A ∧ condition_C ∧ condition_D) ∧ condition_B :=
by 
  -- Provide conditions assumptions
  unfold condition_A condition_B condition_C condition_D
  -- We mark the proof as sorry to indicate that it's omitted.
  sorry

end correct_operation_is_B_l657_657598


namespace count_negative_values_correct_l657_657287

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l657_657287


namespace number_of_seashells_given_l657_657511

theorem number_of_seashells_given (original_seashells : ℤ) (seashells_left : ℤ) 
  (h_original : original_seashells = 62) (h_left : seashells_left = 13) : 
  original_seashells - seashells_left = 49 :=
by 
  rw [h_original, h_left]
  simp
  norm_num
  sorry

end number_of_seashells_given_l657_657511


namespace percentage_second_liquid_l657_657516

namespace KeroseneProblem

def percentageFirstLiquid := 0.25 
def percentageNewMixture := 0.27
def partsFirstLiquid := 6
def partsSecondLiquid := 4
def totalPartsMixture := partsFirstLiquid + partsSecondLiquid

theorem percentage_second_liquid (x : ℝ) : 
  (partsFirstLiquid * percentageFirstLiquid + partsSecondLiquid * (x / 100)) / totalPartsMixture = percentageNewMixture -> 
  x = 30 := by
  sorry

end KeroseneProblem

end percentage_second_liquid_l657_657516


namespace solution_set_for_inequality_l657_657394

noncomputable def f (x : ℝ) : ℝ :=
  2017^x + Real.log (Real.sqrt (x^2 + 1) + x) - 2017^(-x) + 1

theorem solution_set_for_inequality :
  (∀ x : ℝ, f (-x) + f x = 2) →
  (strict_mono f) →
  {x : ℝ | f(2*x - 1) + f(x) > 2} = {x : ℝ | x > 1/3} :=
by
  sorry

end solution_set_for_inequality_l657_657394


namespace inequality_abc_l657_657529

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := 
by 
  sorry

end inequality_abc_l657_657529


namespace magnitude_of_z_l657_657800

theorem magnitude_of_z (z : ℂ) (hz : 3 * z^6 + 2 * complex.I * z^5 - 2 * z - 3 * complex.I = 0) : complex.abs z = 1 :=
sorry

end magnitude_of_z_l657_657800


namespace inequality_solution_l657_657533

theorem inequality_solution :
  {x : ℝ | ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0} = 
  {x : ℝ | (1 < x ∧ x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7)} :=
sorry

end inequality_solution_l657_657533


namespace inequality_solution_l657_657897

theorem inequality_solution (x : ℝ) 
  (hx1 : x ≠ 1) 
  (hx2 : x ≠ 2) 
  (hx3 : x ≠ 3) 
  (hx4 : x ≠ 4) :
  (1 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 1 / (x - 4) < 1 / 24) ↔ (x ∈ Set.Ioo (-7 : ℝ) 1 ∪ Set.Ioo 3 4) := 
sorry

end inequality_solution_l657_657897


namespace max_largest_of_five_distinct_nums_l657_657107

theorem max_largest_of_five_distinct_nums (a b c d e : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
    (h_avg : (a + b + c + d + e) = 75) (h_median : let S := {a, b, c, d, e}.to_list.sort (· ≤ ·) in S.nth_le 2 (by simp) = 18) :
    max (max (max (max a b) c) d) e = 37 := by
  sorry

end max_largest_of_five_distinct_nums_l657_657107


namespace solve_for_z_l657_657797

theorem solve_for_z :
  ∃ z : ℤ, (∀ x y : ℤ, x = 11 → y = 8 → 2 * x + 3 * z = 5 * y) → z = 6 :=
by
  sorry

end solve_for_z_l657_657797


namespace ways_to_place_people_into_groups_l657_657442

theorem ways_to_place_people_into_groups :
  let men := 4
  let women := 5
  ∃ (groups : Nat), groups = 2 ∧
  ∀ (g : Nat → (Fin 3 → (Bool → Nat → Nat))),
    (∀ i, i < group_counts → ∃ m w, g i m w < people ∧ g i m (1 - w) < people ∧ m + 1 - w + (1 - m) + w = 3) →
    let groups : List (List (Fin 2)) := [
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)],
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)]
    ] in
    g.mk 1 dec_trivial * g.mk 2 dec_trivial = 360 :=
sorry

end ways_to_place_people_into_groups_l657_657442


namespace fruit_basket_count_l657_657784

/-- We have seven identical apples and twelve identical oranges.
    A fruit basket must contain at least one piece of fruit.
    Prove that the number of different fruit baskets we can make
    is 103. -/
theorem fruit_basket_count :
  let apples := 7
  let oranges := 12
  let total_possible_baskets := (apples + 1) * (oranges + 1) - 1
  total_possible_baskets = 103 :=
by
  let apples := 7
  let oranges := 12
  let total_possible_baskets := (apples + 1) * (oranges + 1) - 1
  show total_possible_baskets = 103
  sorry

end fruit_basket_count_l657_657784


namespace num_of_negative_x_l657_657295

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l657_657295


namespace proof_problem_l657_657729

-- Given definitions and assumptions
def has_property_A (A : Set ℝ) (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ t ∈ A, ∀ x ∈ D, f x ≤ f (x + t)

-- Question 1
def Q1 (A : Set ℝ) : Prop :=
  ∀ (f g : ℝ → ℝ) (D : Set ℝ),
    has_property_A A f D → has_property_A A g D → f = (λ x, -x) ∧ ¬(g = (λ x, 2 * x))

-- Question 2
def Q2 (A : Set ℝ) : Prop :=
  ∀ (f : ℝ → ℝ) (a : ℝ),
    (∀ x, x ∈ Set.Ici a → ∀ t ∈ A, f x ≤ f (x + t)) → a ∈ Set.Ici 1

-- Question 3
def Q3 (A : Set ℝ) : Prop :=
  ∀ (f : ℤ → ℤ) (m : ℤ),
    (∀ x, f x = f (x - 2) ∧ f x = f (x + m)) → ∃ k, m = 2 * k + 1

-- Statements to prove
theorem proof_problem : Q1 ({-1}) ∧ Q2 Set.Ioo 0 1 ∧ Q3 (Set.to_finset {-2}) := sorry

end proof_problem_l657_657729


namespace find_a_plus_b_l657_657923

theorem find_a_plus_b (a b : ℝ) 
  (h1 : ∃ x y : ℝ, (y = a * x + 1) ∧ (x^2 + y^2 + b*x - y = 1))
  (h2 : ∀ x y : ℝ, (y = a * x + 1) ∧ (x^2 + y^2 + b*x - y = 1) → x + y = 0) : 
  a + b = 2 :=
sorry

end find_a_plus_b_l657_657923


namespace passing_percentage_correct_l657_657178

noncomputable theory -- Mark that the following definitions are noncomputations

-- Define the constants
def secured_marks : ℝ := 42
def shortfall_marks : ℝ := 22
def max_marks : ℝ := 152.38

-- Define the passing percentage calculation
def passing_percentage : ℝ := ((secured_marks + shortfall_marks) / max_marks) * 100

-- The statement we want to prove
theorem passing_percentage_correct : passing_percentage = 42 :=
by
  exact calc
    passing_percentage = ((secured_marks + shortfall_marks) / max_marks) * 100 : by rfl
                    ... = ((42 + 22) / 152.38) * 100 : by rfl
                    ... = (64 / 152.38) * 100 : by rfl
                    ... = 42.00 : by norm_num

end passing_percentage_correct_l657_657178


namespace cucumber_weight_l657_657251

theorem cucumber_weight (W : ℝ)
  (h1 : W * 0.99 + W * 0.01 = W)
  (h2 : (W * 0.01) / 20 = 1 / 95) :
  W = 100 :=
by
  sorry

end cucumber_weight_l657_657251


namespace negative_values_count_l657_657323

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l657_657323


namespace intersect_at_single_point_l657_657246

theorem intersect_at_single_point :
  (∃ (x y : ℝ), y = 3 * x + 5 ∧ y = -5 * x + 20 ∧ y = 4 * x + p) → p = 25 / 8 :=
by
  sorry

end intersect_at_single_point_l657_657246


namespace reyna_lamps_l657_657888

variables (L : ℕ)

-- Define the number of light bulbs per lamp
def bulbs_per_lamp := 7

-- Define the fraction of lamps with two burnt-out bulbs
def fraction_burnt_out := 1 / 4 : ℚ

-- Define the total number of working light bulbs
def total_working_bulbs := 130

-- Define the number of working bulbs per lamp with two burnt-out bulbs
def working_bulbs_with_burnt_out := 5

-- Define the number of working bulbs per lamp with no burnt-out bulbs
def working_bulbs_no_burnt_out := bulbs_per_lamp

theorem reyna_lamps : ((working_bulbs_with_burnt_out * fraction_burnt_out * L) + (working_bulbs_no_burnt_out * (1 - fraction_burnt_out) * L) = total_working_bulbs) → L = 20 :=
sorry

end reyna_lamps_l657_657888


namespace exists_divisible_by_n_with_digit_sum_l657_657479

theorem exists_divisible_by_n_with_digit_sum (n k : ℕ) (h1 : n > 0) (h2 : k ≥ n) (h3 : n % 3 ≠ 0) :
  ∃ m : ℕ, (m % n = 0) ∧ (sum_of_digits m = k) :=
sorry

end exists_divisible_by_n_with_digit_sum_l657_657479


namespace ways_to_place_people_into_groups_l657_657438

theorem ways_to_place_people_into_groups :
  let men := 4
  let women := 5
  ∃ (groups : Nat), groups = 2 ∧
  ∀ (g : Nat → (Fin 3 → (Bool → Nat → Nat))),
    (∀ i, i < group_counts → ∃ m w, g i m w < people ∧ g i m (1 - w) < people ∧ m + 1 - w + (1 - m) + w = 3) →
    let groups : List (List (Fin 2)) := [
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)],
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)]
    ] in
    g.mk 1 dec_trivial * g.mk 2 dec_trivial = 360 :=
sorry

end ways_to_place_people_into_groups_l657_657438


namespace rectangle_diagonal_intersections_l657_657524

-- Defining the problem as a Lean statement
theorem rectangle_diagonal_intersections : 
  ∀ (ℓ b : ℕ), ℓ = 6 → b = 4 → (ℓ + b - Nat.gcd ℓ b) = 8 :=
by
  intros ℓ b hℓ hb
  rw [hℓ, hb]
  -- Since gcd(6, 4) = 2
  have gcd_6_4 : Nat.gcd 6 4 = 2 := by sorry
  rw [gcd_6_4]
  -- Substituting the values: 6 + 4 - 2 = 8
  exact rfl

end rectangle_diagonal_intersections_l657_657524


namespace intersection_correct_l657_657736

def set_A : Set ℤ := {-1, 1, 2, 4}
def set_B : Set ℤ := {x | |x - 1| ≤ 1}

theorem intersection_correct :
  set_A ∩ set_B = {1, 2} :=
  sorry

end intersection_correct_l657_657736


namespace mary_vacuum_charges_l657_657109

theorem mary_vacuum_charges :
  let battery_life := 10
  let time_per_room := 8
  let bedrooms := 3
  let kitchen := 1
  let living_room := 1
  let dining_room := 1
  let office := 1
  let bathrooms := 2
  let total_rooms := bedrooms + kitchen + living_room + dining_room + office + bathrooms
  in total_rooms * time_per_room <= total_rooms * battery_life →
     total_rooms = 9 :=
by
  intros
  sorry

end mary_vacuum_charges_l657_657109


namespace different_log_values_count_l657_657355

theorem different_log_values_count : 
  ∃ (s : Finset ℕ), s = {1, 2, 3, 4, 7, 9} ∧ 
                     (∑ x in s, ∑ y in s, if x ≠ y then 1 else 0) - 4 + 1 = 17 :=
by
  let s := {1, 2, 3, 4, 7, 9}
  use s
  sorry

end different_log_values_count_l657_657355


namespace solution_set_of_f_mul_g_l657_657055

-- Define the types for odd and even functions
def is_odd_function {α β : Type*} [AddGroup α] [HasOne α] (f : α → β) : Prop :=
∀ x, f (-x) = -f x

def is_even_function {α β : Type*} [AddGroup α] [HasOne α] (g : α → β) : Prop :=
∀ x, g (-x) = g x

variable {f g : ℝ → ℝ}

-- Define the conditions given in the problem
axiom f_is_odd : is_odd_function f
axiom g_is_even : is_even_function g
axiom derivative_condition : ∀ x, x < 0 → f'(x) * g(x) + g'(x) * f(x) < 0
axiom g_at_3_zero : g 3 = 0

-- Define the proof statement
theorem solution_set_of_f_mul_g :
  {x : ℝ | f x * g x < 0} = (Set.Iio (-3) ∪ Set.Ioc 0 3) := sorry

end solution_set_of_f_mul_g_l657_657055


namespace part1_part2_l657_657373

-- Let's define the arithmetic sequence and conditions
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + d * (n - 1)
def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variables (a1 a4 a3 a5 : ℕ)
variable (d : ℕ)

-- Additional conditions for the problem  
axiom h1 : a1 = 2
axiom h2 : a4 = 8
axiom h3 : arithmetic_seq a1 d 3 + arithmetic_seq a1 d 5 = a4 + 8

-- Define S7
def S7 : ℕ := sum_arithmetic_seq a1 d 7

-- Part I: Prove S7 = 56
theorem part1 : S7 = 56 := 
by
  sorry

-- Part II: Prove k = 2 given additional conditions
variable (k : ℕ)

-- Given that a_3, a_{k+1}, S_k are a geometric sequence
def is_geom_seq (a b s : ℕ) : Prop := b*b = a * s

axiom h4 : a3 = arithmetic_seq a1 d 3
axiom h5 : ∃ k, 0 < k ∧ is_geom_seq a3 (arithmetic_seq a1 d (k + 1)) (sum_arithmetic_seq a1 d k)

theorem part2 : ∃ k, 0 < k ∧ k = 2 := 
by
  sorry

end part1_part2_l657_657373


namespace random_phenomenon_l657_657599

def is_certain_event (P : Prop) : Prop := ∀ h : P, true

def is_random_event (P : Prop) : Prop := ¬is_certain_event P

def scenario1 : Prop := ∀ pressure temperature : ℝ, (pressure = 101325) → (temperature = 100) → true
-- Under standard atmospheric pressure, water heated to 100°C will boil

def scenario2 : Prop := ∃ time : ℝ, true
-- Encountering a red light at a crossroads (which happens at random times)

def scenario3 (a b : ℝ) : Prop := true
-- For a rectangle with length and width a and b respectively, its area is a * b

def scenario4 : Prop := ∀ a b : ℝ, ∃ x : ℝ, a * x + b = 0
-- A linear equation with real coefficients always has one real root

theorem random_phenomenon : is_random_event scenario2 :=
by
  sorry

end random_phenomenon_l657_657599


namespace count_negative_values_of_x_l657_657313

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l657_657313


namespace chloe_dimes_l657_657668

variable (d : ℕ)

def jacket_cost : ℝ := 45.50
def ten_dollar_bills : ℝ := 4 * 10 -- Four $10 bills
def quarters : ℝ := 10 * 0.25 -- Ten quarters
def nickels : ℝ := 15 * 0.05 -- Fifteen nickels
def dimes_value (d : ℕ) : ℝ := d * 0.10 -- Unknown number of dimes

theorem chloe_dimes :
  10 * 4 + 0.25 * 10 + 0.05 * 15 + 0.10 * d ≥ 45.50 → d ≥ 23 :=
by
  intro h
  sorry

end chloe_dimes_l657_657668


namespace palabras_bookstore_workers_l657_657602

theorem palabras_bookstore_workers :
  let W := 40 in
  let S := W * 1 / 4 in
  let K := W * 5 / 8 in
  let B := 2 in
  ((S - B) - 1 = W - (S + K - B)) →
  (B = 2) :=
by
  intros
  sorry

end palabras_bookstore_workers_l657_657602


namespace num_planes_determined_by_basis_l657_657127

-- Definitions for the basis and space condition
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Define basis condition in Lean
def is_basis (a b c : V) : Prop :=
  linear_independent ℝ ![a, b, c] ∧ submodule.span ℝ ![a, b, c] = ⊤

-- The theorem stating that the number of planes determined by a basis {a, b, c} in space is 3
theorem num_planes_determined_by_basis (h : is_basis a b c) : 
    ∃ n, n = 3 :=
sorry

end num_planes_determined_by_basis_l657_657127


namespace eight_digit_descending_numbers_count_l657_657437

theorem eight_digit_descending_numbers_count : (Nat.choose 10 2) = 45 :=
by
  sorry

end eight_digit_descending_numbers_count_l657_657437


namespace log_eq_l657_657790

theorem log_eq (a b : ℝ) (ha : a = log 4 625) (hb : b = log 2 25) : a = b :=
by
  sorry

end log_eq_l657_657790


namespace num_of_negative_x_l657_657297

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l657_657297


namespace total_treats_l657_657563

theorem total_treats (children : ℕ) (hours : ℕ) (houses_per_hour : ℕ) (treats_per_house_per_kid : ℕ) :
  children = 3 → hours = 4 → houses_per_hour = 5 → treats_per_house_per_kid = 3 → 
  (children * hours * houses_per_hour * treats_per_house_per_kid) = 180 :=
by
  intros
  sorry

end total_treats_l657_657563


namespace angle_bisector_theorem_l657_657468

-- Definitions given as conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables {triangleABC : Triangle A B C}
variables (angleB : MeasureAngle triangleABC B = 40)
variables (angleC : MeasureAngle triangleABC C = 40)
variables (bisectorBD : IsAngleBisector triangleABC B D)

-- The theorem statement based on the question and conditions
theorem angle_bisector_theorem (h1 : MeasureAngle triangleABC B = 40)
                                (h2 : MeasureAngle triangleABC C = 40)
                                (h3 : IsAngleBisector triangleABC B D) 
                                : Length (Segment D B) + Length (Segment D A) = Length (Segment B C) := 
sorry

end angle_bisector_theorem_l657_657468


namespace max_candy_remainder_l657_657777

theorem max_candy_remainder (x : ℕ) : x % 11 < 11 ∧ (∀ r : ℕ, r < 11 → x % 11 ≤ r) → x % 11 = 10 := 
sorry

end max_candy_remainder_l657_657777


namespace least_points_l657_657492

/--
Let \( f(n) \) be the least number of distinct points in the plane such that 
for each \( k=1, 2, \ldots, n \), there exists a straight line containing exactly \( k \) of these points.
Show that \( f(n) = \left\lfloor \frac{n+1}{2} \right\rfloor \left\lfloor \frac{n+2}{2} \right\rfloor \),
where \( \lfloor x \rfloor \) denotes the greatest integer not exceeding \( x \).
-/
theorem least_points (n : ℕ) : 
  let f : ℕ → ℕ := λ n, 
    ⌊ (n + 1) / 2 ⌋ * ⌊ (n + 2) / 2 ⌋ in
  ∃ X : Set (ℝ × ℝ),
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ∃ l : set (ℝ × ℝ),
      l.Card = k ∧ l ⊆ X) ∧
    X.card = f n := by
  sorry

end least_points_l657_657492


namespace range_OP_OM_l657_657464

noncomputable def line_l_polar_eqn (ρ θ : ℝ) : Prop :=
  ρ * cos θ = 2

noncomputable def curve_C_polar_eqn (ρ θ : ℝ) : Prop :=
  ρ = 2 * sin θ

theorem range_OP_OM (β : ℝ) (h1 : 0 < β) (h2 : β < π / 2) :
  0 < (1 / 2) * sin (2 * β) ∧ (1 / 2) * sin (2 * β) ≤ 1 / 2 := 
  sorry

end range_OP_OM_l657_657464


namespace min_triangle_area_is_one_l657_657770

-- Define the points A and B
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (24, 10)

-- Define the area function using the Shoelace formula
def triangle_area (C : ℤ × ℤ) : ℝ :=
  let (p, q) := C
  in 1/2 * (abs (10 * p - 24 * q))

-- The main theorem stating the minimum area of triangle ABC is 1
theorem min_triangle_area_is_one : ∃ (C : ℤ × ℤ), triangle_area C = 1 :=
by
  sorry

end min_triangle_area_is_one_l657_657770


namespace number_of_negative_x_l657_657335

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l657_657335


namespace value_of_k_l657_657405

theorem value_of_k (x y k : ℝ) (h1 : 3 * x + 2 * y = k + 1) (h2 : 2 * x + 3 * y = k) (h3 : x + y = 2) :
  k = 9 / 2 :=
by
  sorry

end value_of_k_l657_657405


namespace stack_glasses_opacity_l657_657947

-- Define the main problem's parameters and conditions
def num_glass_pieces : Nat := 5
def rotations := [0, 90, 180, 270] -- Possible rotations

-- Define the main theorem to state the problem in Lean
theorem stack_glasses_opacity :
  (∃ count : Nat, count = 7200 ∧
   -- There are 5 glass pieces
   ∀ (g : Fin num_glass_pieces), 
     -- Each piece is divided into 4 triangles
     ∀ (parts : Fin 4),
     -- There exists a unique painting configuration for each piece, can one prove it is exactly 7200 ways
     True
  ) :=
  sorry

end stack_glasses_opacity_l657_657947


namespace length_increase_percentage_l657_657125

theorem length_increase_percentage 
  (L B : ℝ)
  (x : ℝ)
  (h1 : B' = B * 0.8)
  (h2 : L' = L * (1 + x / 100))
  (h3 : A = L * B)
  (h4 : A' = L' * B')
  (h5 : A' = A * 1.04) 
  : x = 30 :=
sorry

end length_increase_percentage_l657_657125


namespace absolute_diff_half_l657_657421

theorem absolute_diff_half (x y : ℝ) 
  (h : ((x + y = x - y ∧ x - y = x * y) ∨ 
       (x + y = x * y ∧ x * y = x / y) ∨ 
       (x - y = x * y ∧ x * y = x / y))
       ∧ x ≠ 0 ∧ y ≠ 0) : 
     |y| - |x| = 1 / 2 := 
sorry

end absolute_diff_half_l657_657421


namespace fenced_area_with_cutout_l657_657539

theorem fenced_area_with_cutout :
  let rectangle_length : ℕ := 20
  let rectangle_width : ℕ := 16
  let cutout_length : ℕ := 4
  let cutout_width : ℕ := 4
  rectangle_length * rectangle_width - cutout_length * cutout_width = 304 := by
  sorry

end fenced_area_with_cutout_l657_657539


namespace kitten_puppy_bites_ratio_l657_657626

-- Definitions based on conditions
def total_length (ℓ : ℝ) := ℓ > 0
def kitten_bite (ℓ : ℝ) := ℓ * (3 / 4)
def puppy_bite (ℓ : ℝ) := ℓ * (2 / 3)

theorem kitten_puppy_bites_ratio (ℓ : ℝ) (h : total_length ℓ) :
  let b₁ := kitten_bite ℓ / 2 in
  let b₂ := kitten_bite ℓ / 2 * (3 / 4) in
  b₁ = b₂ :=
by {
  sorry
}

end kitten_puppy_bites_ratio_l657_657626


namespace hexagon_largest_angle_l657_657905

variable (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ)
theorem hexagon_largest_angle (h : a₁ = 3)
                             (h₀ : a₂ = 3)
                             (h₁ : a₃ = 3)
                             (h₂ : a₄ = 4)
                             (h₃ : a₅ = 5)
                             (h₄ : a₆ = 6)
                             (sum_angles : 3*a₁ + 3*a₀ + 3*a₁ + 4*a₂ + 5*a₃ + 6*a₄ = 720) :
                             6 * 30 = 180 := by
    sorry

end hexagon_largest_angle_l657_657905


namespace determine_omega_l657_657785

noncomputable def f (x ω : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

theorem determine_omega (ω : ℝ) (h : ∃ k : ℤ, (ω * π / 8 + π / 4) = k * π) : ω = 6 :=
by
  have key_identity : ∀ (x : ℝ), f x ω = sqrt 2 * Real.sin (ω * x + π / 4) := sorry
  sorry

end determine_omega_l657_657785


namespace sum_of_roots_eq_zero_and_product_of_roots_eq_q_l657_657884

theorem sum_of_roots_eq_zero_and_product_of_roots_eq_q (p q : ℝ) :
  let P : Polynomial ℝ := polynomial.C q + polynomial.C p * polynomial.X^2 + polynomial.X^4 in
  (P.degree = 4) →
  (P.coeff 0 = q) ∧
  (P.coeff 1 = 0) ∧
  (P.coeff 2 = p) ∧
  (P.coeff 3 = 0) ∧
  (P.coeff 4 = 1) →
  (∑ i in P.rootSet ℝ, i = 0) ∧ 
  (∏ i in P.rootSet ℝ, i = q) :=
by {
  intro p q hs,
  simp only [Polynomial.coeff, Polynomial.degree, Polynomial.rootSet],
  sorry
}

end sum_of_roots_eq_zero_and_product_of_roots_eq_q_l657_657884


namespace association_confidence_level_l657_657805

theorem association_confidence_level :
  ∀ (χ2 : ℝ), χ2 = 6.825 → (∃ (df : ℕ), df = 1) → 0.99 :=
by
  intro χ2 hχ2
  intro hdf
  -- Proof is to be provided
  sorry

end association_confidence_level_l657_657805


namespace linda_speed_last_hour_l657_657509

theorem linda_speed_last_hour :
  ∀ (total_distance total_time speed_1 speed_2 : ℕ), 
  total_distance = 180 → 
  total_time = 3 → 
  speed_1 = 50 → 
  speed_2 = 70 → 
  (50 + 70 + (total_distance / total_time * total_time - (50 + 70))) / 3 = 60 :=
by
  intros total_distance total_time speed_1 speed_2 h_distance h_time h_speed1 h_speed2
  rw [h_distance, h_time, h_speed1, h_speed2]
  -- The proof steps will go here.
  sorry


end linda_speed_last_hour_l657_657509


namespace count_negative_values_correct_l657_657288

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l657_657288


namespace car_can_complete_trip_l657_657874

theorem car_can_complete_trip (n : ℕ) (fuel : Fin n → ℝ) (dist : Fin n → ℝ) :
  (∀ i, 0 ≤ fuel i) →
  (∀ i, 0 ≤ dist i) →
  (∑ i, fuel i = ∑ i, dist i) →
  ∃ (A : Fin n), 
  let start := A, remainder := fun (i : Fin n), ∑ j in Fin.range (i + 1), fuel (start + j) - dist (start + j)
  in ∀ (i : Fin n), 0 ≤ remainder (start + i) :=
by sorry

end car_can_complete_trip_l657_657874


namespace proof_ps_equals_qs_l657_657935

variable (o : Circle) (A B C D S P Q : Point)

-- Assume points A, B, C, D lie on circle o in given order
axiom points_on_circle_in_order : ∀ {A B C D}, A ∈ o ∧ B ∈ o ∧ C ∈ o ∧ D ∈ o

-- Assume S lies inside o such that ∠SAD = ∠SCB and ∠SDA = ∠SBC respectively
axiom angle_eq_1 : ∀ {A B C D S}, ∠ SAD = ∠ SCB
axiom angle_eq_2 : ∀ {A B C D S}, ∠ SDA = ∠ SBC

-- Assume the angle bisector of ∠ASB intersects the circle at P and Q
axiom angle_bisector_intersects : ∀ {A B S P Q}, P ∈ o ∧ Q ∈ o ∧ bisect (∠ ASB) (P, Q)

-- Objective is to prove that PS = QS
theorem proof_ps_equals_qs (h1 : points_on_circle_in_order A B C D)
                           (h2 : angle_eq_1 A B C D S)
                           (h3 : angle_eq_2 A B C D S)
                           (h4 : angle_bisector_intersects A B S P Q) :
  dist P S = dist Q S := sorry

end proof_ps_equals_qs_l657_657935


namespace sum_of_three_iterated_digits_of_A_is_7_l657_657663

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

def A : ℕ := 4444 ^ 4444

theorem sum_of_three_iterated_digits_of_A_is_7 :
  sum_of_digits (sum_of_digits (sum_of_digits A)) = 7 :=
by
  -- We'll skip the actual proof here
  sorry

end sum_of_three_iterated_digits_of_A_is_7_l657_657663


namespace total_new_students_l657_657474

-- Given conditions
def number_of_schools : ℝ := 25.0
def average_students_per_school : ℝ := 9.88

-- Problem statement
theorem total_new_students : number_of_schools * average_students_per_school = 247 :=
by sorry

end total_new_students_l657_657474


namespace decimal_representation_150th_digit_l657_657586

theorem decimal_representation_150th_digit : 
  ∃ s : ℕ → ℤ, 
    (∀ n, s n = (13/17 : ℚ)^n % 10) ∧ 
    let decSeq := [7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1].cycle in
    ∀ n, decSeq.get n = s n → decSeq.get 149 = 7 := 
begin
  sorry
end

end decimal_representation_150th_digit_l657_657586


namespace consecutive_odd_integers_l657_657557

theorem consecutive_odd_integers (n : ℤ) (h : (n - 2) + (n + 2) = 130) : n = 65 :=
sorry

end consecutive_odd_integers_l657_657557


namespace grouping_count_l657_657451

theorem grouping_count (men women : ℕ) 
  (h_men : men = 4) (h_women : women = 5)
  (at_least_one_man_woman : ∀ (g1 g2 g3 : Finset (Fin 9)), 
    g1.card = 3 → g2.card = 3 → g3.card = 3 → g1 ∩ g2 = ∅ → g2 ∩ g3 = ∅ → g3 ∩ g1 = ∅ → 
    (g1 ∩ univ.filter (· < 4)).nonempty ∧ (g1 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g2 ∩ univ.filter (· < 4)).nonempty ∧ (g2 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g3 ∩ univ.filter (· < 4)).nonempty ∧ (g3 ∩ univ.filter (· ≥ 4)).nonempty) :
  (choose 4 1 * choose 5 2 * choose 3 1 * choose 3 2) / 2! = 180 :=
sorry

end grouping_count_l657_657451


namespace count_negative_x_with_sqrt_pos_int_l657_657348

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l657_657348


namespace additional_flowers_needed_and_number_of_bouquets_l657_657177

def n : ℕ := 1273
def k : ℕ := 89

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m ∈ Finset.range p, m ≠ 0 → m ∣ p → m = 1

axiom n_is_prime : is_prime n
axiom k_is_prime : is_prime k

theorem additional_flowers_needed_and_number_of_bouquets :
  (∃ (m : ℕ), m > 0 ∧ (n + m) % k = 0 ∧ (n + m) / k = 15) :=
by
  use 62
  split
  · show 62 > 0, from Nat.zero_lt_succ 61
  split
  · show (1273 + 62) % 89 = 0, from rfl
  · show (1273 + 62) / 89 = 15, from rfl

end additional_flowers_needed_and_number_of_bouquets_l657_657177


namespace safe_paths_count_l657_657007

theorem safe_paths_count :
  let total_paths := Multinomial (4, 4, 4)
  let paths_through_mine := Multinomial (2, 2, 2) * Multinomial (2, 2, 2)
  let paths_near_mine := 6 * Multinomial (3, 2, 1) * Multinomial (3, 2, 1)
  let paths_two_units_away := 6 * Multinomial (3, 2, 0) * Multinomial (1, 2, 4)
  let safe_paths := total_paths - paths_through_mine - paths_near_mine + paths_two_units_away
  in
  total_paths = 10395 ∧ paths_through_mine = 225 ∧ paths_near_mine = 540 ∧ paths_two_units_away = 270 ∧ safe_paths = 9900 :=
begin
  sorry
end

end safe_paths_count_l657_657007


namespace circles_intersect_l657_657936

theorem circles_intersect :
  ∀ (x y : ℝ),
    ((x^2 + y^2 - 2 * x + 4 * y + 1 = 0) →
    (x^2 + y^2 - 6 * x + 2 * y + 9 = 0) →
    (∃ c1 c2 r1 r2 d : ℝ,
      (x - 1)^2 + (y + 2)^2 = r1 ∧ r1 = 4 ∧
      (x - 3)^2 + (y + 1)^2 = r2 ∧ r2 = 1 ∧
      d = Real.sqrt ((3 - 1)^2 + (-1 + 2)^2) ∧
      d > abs (r1 - r2) ∧ d < (r1 + r2))) :=
sorry

end circles_intersect_l657_657936


namespace find_fixed_point_c_l657_657572

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := 2 * x ^ 2 - c

theorem find_fixed_point_c (c : ℝ) : 
  (∃ a : ℝ, f a = a ∧ g a c = a) ↔ (c = 3 ∨ c = 6) := sorry

end find_fixed_point_c_l657_657572


namespace abs_of_difference_eq_neg_two_m_l657_657792

theorem abs_of_difference_eq_neg_two_m (m : ℝ) (h : m < 0) : |m - (-m)| = -2m := by
  sorry

end abs_of_difference_eq_neg_two_m_l657_657792


namespace infimum_and_no_minimum_l657_657841

noncomputable def P (X₁ X₂ : ℝ) : ℝ := X₁^2 + (1 - X₁ * X₂)^2

theorem infimum_and_no_minimum (P : ℝ → ℝ → ℝ) (hP : P = λ X₁ X₂, X₁^2 + (1 - X₁ * X₂)^2) : 
  (∀ X₁ X₂ : ℝ, 0 ≤ P X₁ X₂) ∧ ((∃ X : ℝ, ∃ Y : ℝ, ∀ ε > 0, ∃ X₀ Y₀ : ℝ, P X₀ Y₀ < ε) ∧ (¬ ∃ X₀ Y₀ : ℝ, ∀ (X Y : ℝ), P X Y ≥ P X₀ Y₀)) :=
by 
  split
  sorry -- Proof for non-negativity of P
  split
  sorry -- Proof that infimum is 0
  sorry -- Proof that P does not admit a minimum

end infimum_and_no_minimum_l657_657841


namespace decimal_representation_150th_digit_l657_657589

theorem decimal_representation_150th_digit : 
  ∃ s : ℕ → ℤ, 
    (∀ n, s n = (13/17 : ℚ)^n % 10) ∧ 
    let decSeq := [7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1].cycle in
    ∀ n, decSeq.get n = s n → decSeq.get 149 = 7 := 
begin
  sorry
end

end decimal_representation_150th_digit_l657_657589


namespace max_notebooks_l657_657825

-- Definitions based on the conditions
def joshMoney : ℕ := 1050
def notebookCost : ℕ := 75

-- Statement to prove
theorem max_notebooks (x : ℕ) : notebookCost * x ≤ joshMoney → x ≤ 14 := by
  -- Placeholder for the proof
  sorry

end max_notebooks_l657_657825


namespace total_cats_l657_657137

-- Define the conditions as constants
def asleep_cats : ℕ := 92
def awake_cats : ℕ := 6

-- State the theorem that proves the total number of cats
theorem total_cats : asleep_cats + awake_cats = 98 := 
by
  -- Proof omitted
  sorry

end total_cats_l657_657137


namespace color_points_odd_l657_657016

theorem color_points_odd (n : ℕ) (h₁ : 0 < n) (h₂ : 5 ≤ n) :
  (∃ (move : {k // 1 ≤ k ∧ k < n / 2} → Fin n → Fin n),
    (∀ (init_col : Fin n → Bool),
      ∃ (seq : List {k // 1 ≤ k ∧ k < n / 2}),
        ∀ (col : Fin n → Bool), (col = init_col ∨ col = fun _ => tt ∨ col = fun _ => ff))) ↔ Odd n :=
sorry

end color_points_odd_l657_657016


namespace not_divisible_l657_657664

theorem not_divisible (n k : ℕ) : ¬ (5 ^ n + 1) ∣ (5 ^ k - 1) :=
sorry

end not_divisible_l657_657664


namespace solve_floor_equation_l657_657609

noncomputable def x_solution_set : Set ℚ := 
  {x | x = 1 ∨ ∃ k : ℕ, 16 ≤ k ∧ k ≤ 22 ∧ x = (k : ℚ)/23 }

theorem solve_floor_equation (x : ℚ) (hx : x ∈ x_solution_set) : 
  (⌊20*x + 23⌋ : ℚ) = 20 + 23*x :=
sorry

end solve_floor_equation_l657_657609


namespace product_of_all_possible_values_of_c_l657_657686

theorem product_of_all_possible_values_of_c
    (g : ℝ → ℝ := λ x, c / (3 * x - 4))
    (h_inv : ∀ y, g (g⁻¹ y) = y)
    (h : g 3 = g⁻¹ (c + 2)) :
  ∀ c ∈ {c | g 3 = g⁻¹ (c + 2)}, ∏ c = -40 / 3 := by
  sorry

end product_of_all_possible_values_of_c_l657_657686


namespace coconut_tree_difference_l657_657523

-- Define the known quantities
def mango_trees : ℕ := 60
def total_trees : ℕ := 85
def half_mango_trees : ℕ := 30 -- half of 60
def coconut_trees : ℕ := 25 -- 85 - 60

-- Define the proof statement
theorem coconut_tree_difference : (half_mango_trees - coconut_trees) = 5 := by
  -- The proof steps are given
  sorry

end coconut_tree_difference_l657_657523


namespace circumcenter_tetrahedron_l657_657198

variables {A B C S A1 B1 C1 O : Type} [point : RO_point A] [point : RO_point B] [point : RO_point C] [point : RO_point S]
[point : RO_point A1] [point : RO_point B1] [point : RO_point C1] [point : RO_point O]

structure Tetrahedron (A B C S : Type) :=
(vertex1 : A)
(vertex2 : B)
(vertex3 : C)
(vertex4 : S)

structure Sphere (center : Type) :=
(radius : ℝ)

structure Plane (p1 p2 p3 : Type) :=
(point1 : p1)
(point2 : p2)
(point3 : p3)

noncomputable def circumcenter (t : Tetrahedron A1 B1 C1 S) : Type := sorry

theorem circumcenter_tetrahedron 
  (ABC_plane : Plane A B C) 
  (sphere_center : Sphere (Plane A B C))
  (intersections : (A1 = intersection_of_line_and_sphere SA sphere) ∧ 
                   (B1 = intersection_of_line_and_sphere SB sphere) ∧ 
                   (C1 = intersection_of_line_and_sphere SC sphere))
  (tangent_planes_meet_at_O : tangent_planes_meet_at A1 B1 C1 sphere = O):
  O = circumcenter (Tetrahedron A1 B1 C1 S) := 
sorry

end circumcenter_tetrahedron_l657_657198


namespace students_unable_to_partner_l657_657952

-- Define the conditions
def num_males_class1 : ℕ := 18
def num_females_class1 : ℕ := 12

def num_males_class2 : ℕ := 16
def num_females_class2 : ℕ := 20

def num_males_class3 : ℕ := 13
def num_females_class3 : ℕ := 19

def num_males_class7 : ℕ := 23
def num_females_class7 : ℕ := 21

-- Calculate total males
def total_males := num_males_class1 + num_males_class2 + num_males_class3 + num_males_class7
-- Calculate total females
def total_females := num_females_class1 + num_females_class2 + num_females_class3 + num_females_class7

-- Proof statement
theorem students_unable_to_partner : total_females - total_males = 2 := by
  calculate total_males  -- ensure correct calculation
  calculate total_females -- ensure correct calculation
  sorry -- skip the proof

end students_unable_to_partner_l657_657952


namespace largest_n_factorization_l657_657271

theorem largest_n_factorization :
  ∃ (n : ℤ), (∀ A B : ℤ, 3 * B + A = n -> A * B = 90 -> n ≤ 271) ∧ (∃ A B : ℤ, 3 * B + A = 271 ∧ A * B = 90) :=
by {
  apply Exists.intro 271,
  constructor,
  {
    intros A B eqn₁ eqn₂,
    have aux : 3 * B + A ≤ 271,
    sorry, -- Proof steps would go here
    exact aux,
  },
  {
    apply Exists.intro 1,
    apply Exists.intro 90,
    split,
    exact rfl,
    exact rfl,
  }
}

end largest_n_factorization_l657_657271


namespace omega_value_l657_657756

theorem omega_value (ω : ℝ) (h₁ : ω > 0) :
  (∃ (x₁ x₂ x₃ : ℝ), f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ f x₃ = g x₃ ∧ is_consecutive [x₁, x₂, x₃]) →
  ω = π / 2 := 
by
  sorry

noncomputable def f (x : ℝ) := sqrt 2 * sin (ω * x)
noncomputable def g (x : ℝ) := sqrt 2 * cos (ω * x)
noncomputable def is_consecutive (l : list ℝ) : Prop := 
sorry -- Helper definition to ensure that x₁, x₂, x₃ are consecutive points.

end omega_value_l657_657756


namespace numbers_between_1000_and_2013_with_property_l657_657928

def units_digit_sum_condition (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.length = 4 ∧
  digits.head! = digits.tail!.sum

def count_numbers_with_property : ℕ :=
  (List.range' 1001 (2013 - 1001)).count units_digit_sum_condition

theorem numbers_between_1000_and_2013_with_property :
  count_numbers_with_property = 46 :=
sorry

end numbers_between_1000_and_2013_with_property_l657_657928


namespace probability_different_suits_l657_657865

theorem probability_different_suits (h : ∀ (c1 c2 c3 : ℕ), c1 ≠ c2 ∧ c1 ≠ c3 ∧ c2 ≠ c3 ∧ 
                                    ∀ {x}, x ∈ {c1, c2, c3} → x ∈ finset.range 52) : 
  let prob := (13 / 17) * (13 / 25) in
  prob = (169 / 425) := 
by
  sorry

end probability_different_suits_l657_657865


namespace asymptotes_of_hyperbola_l657_657551

theorem asymptotes_of_hyperbola (b : ℝ) (h_focus : 2 * Real.sqrt 2 ≠ 0) :
  2 * Real.sqrt 2 = Real.sqrt ((2 * 2) + b^2) → 
  (∀ (x y : ℝ), ((x^2 / 4) - (y^2 / b^2) = 1 → x^2 - y^2 = 4)) → 
  (∀ (x y : ℝ), ((x^2 - y^2 = 4) → y = x ∨ y = -x)) := 
  sorry

end asymptotes_of_hyperbola_l657_657551


namespace set_intersection_example_l657_657742

theorem set_intersection_example (A : Set ℝ) (B : Set ℝ):
  A = { -1, 1, 2, 4 } → 
  B = { x | |x - 1| ≤ 1 } → 
  A ∩ B = {1, 2} :=
by
  intros hA hB
  sorry

end set_intersection_example_l657_657742


namespace sum_primes_1_to_50_mod_4_and_6_l657_657711

axiom problem_condition (p : ℕ) : 
  (prime p) ∧ p ∈ Icc 1 50 ∧ (p % 4 = 3) ∧ (p % 6 = 1)

theorem sum_primes_1_to_50_mod_4_and_6 : 
  (∑ p in (finset.filter (λ p, prime p ∧ p ∈ Icc 1 50 ∧ p % 4 = 3 ∧ p % 6 = 1) finset.range(51)), p) = 38 :=
by
  sorry

end sum_primes_1_to_50_mod_4_and_6_l657_657711


namespace product_inequality_hundred_c_value_l657_657013

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 4 / 5 else 2 * (sequence (n - 1)) ^ 2 - 1

theorem product_inequality (c : ℝ) : (∀ n : ℕ, (∏ i in Finset.range n, sequence i) ≤ c / 2 ^ n) → c = 5 / 3 :=
  sorry

theorem hundred_c_value : 100 * (5 / 3) ≈ 167 :=
  sorry

end product_inequality_hundred_c_value_l657_657013


namespace locus_of_internal_tangents_l657_657228

-- Define the setup
variables {A B C : Point} -- points of triangle
variable {Γ : Circle} -- circumcircle of triangle ABC
variable (d : Line) -- line parallel to BC intersecting triangle at D, E, and Γ at K, L

-- Define the problem
theorem locus_of_internal_tangents 
  (h_parallel : d.parallel_to (Line.mk B C)) -- d is parallel to line BC
  (h_intersect_AB : d.intersects (Line.mk A B) at_point D) -- intersects AB at D
  (h_intersect_AC : d.intersects (Line.mk A C) at_point E) -- intersects AC at E
  (h_intersect_Γ_K : d.intersects Γ at_point K) -- intersects Γ at K
  (h_intersect_Γ_L : d.intersects Γ at_point L) -- intersects Γ at L
  (γ1 : Circle) (γ2 : Circle) -- circles tangent to Γ and respective segments
  (h_tangent_γ1_Γ : γ1.tangent_to Γ)
  (h_tangent_γ2_Γ : γ2.tangent_to Γ)
  (h_tangent_γ1_KD : γ1.tangent_to_segment K D)
  (h_tangent_γ1_DB : γ1.tangent_to_segment D B)
  (h_tangent_γ2_LE : γ2.tangent_to_segment L E)
  (h_tangent_γ2_EC : γ2.tangent_to_segment E C) :
  locus_of_intersection_of_internal_tangents(γ1, γ2) = Line.bisector_internal_angle A B C :=
sorry

end locus_of_internal_tangents_l657_657228


namespace cars_left_parking_lot_l657_657424

/-- Given conditions:
    1. There are originally 12 cars in the parking lot.
    2. No additional cars have entered.
    3. There are currently 9 cars remaining.
    Prove that the number of cars that have left the parking lot is 3.
-/
theorem cars_left_parking_lot (initial_cars current_cars : ℕ) (h1 : initial_cars = 12) (h2 : current_cars = 9) :
  (initial_cars - current_cars) = 3 :=
by
  rw [h1, h2]
  rfl

end cars_left_parking_lot_l657_657424


namespace right_triangle_medians_l657_657004

theorem right_triangle_medians (m : ℝ) :
  (∃ (a b c d : ℝ), 
    let P := (a, b) 
    ∧ let Q := (a, b + 2 * c)
    ∧ let R := (a + 2 * d, b)
    ∧ let M1 := (a, b + c)
    ∧ let M2 := (a + d, b)
    ∧ (M1.2 - P.2) / (M1.1 - P.1) = 2
    ∧ (M2.2 - R.2) / (M2.1 - R.1) = m 
    ) → m = 2 :=
by
  -- proof goes here
  sorry

end right_triangle_medians_l657_657004


namespace value_of_a6_l657_657462

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem value_of_a6 {a : ℕ → ℝ} {r : ℝ}
  (h_geo : geometric_sequence a r)
  (h_root1 : a 4)
  (h_root2 : a 8)
  (h_quadratic_roots : ∀ x, x^2 + 11 * x + 9 = 0 → (x = a 4 ∨ x = a 8))
  (h_product : (a 4) * (a 8) = 9)
  (h_sum : (a 4) + (a 8) = -11) : 
  a 6 = -3 :=
by
  sorry

end value_of_a6_l657_657462


namespace ways_to_place_people_into_groups_l657_657440

theorem ways_to_place_people_into_groups :
  let men := 4
  let women := 5
  ∃ (groups : Nat), groups = 2 ∧
  ∀ (g : Nat → (Fin 3 → (Bool → Nat → Nat))),
    (∀ i, i < group_counts → ∃ m w, g i m w < people ∧ g i m (1 - w) < people ∧ m + 1 - w + (1 - m) + w = 3) →
    let groups : List (List (Fin 2)) := [
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)],
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)]
    ] in
    g.mk 1 dec_trivial * g.mk 2 dec_trivial = 360 :=
sorry

end ways_to_place_people_into_groups_l657_657440


namespace numeral_in_150th_decimal_place_l657_657590

noncomputable def decimal_representation_13_17 : String :=
  "7647058823529411"

theorem numeral_in_150th_decimal_place :
  (decimal_representation_13_17.get (150 % 17)).iget = '1' :=
by
  sorry

end numeral_in_150th_decimal_place_l657_657590


namespace company_A_profit_l657_657671

-- Define the conditions
def total_profit (x : ℝ) : ℝ := x
def company_B_share (x : ℝ) : Prop := 0.4 * x = 60000
def company_A_percentage : ℝ := 0.6

-- Define the statement to be proved
theorem company_A_profit (x : ℝ) (h : company_B_share x) : 0.6 * x = 90000 := sorry

end company_A_profit_l657_657671


namespace sally_balloon_count_l657_657090

theorem sally_balloon_count (n_initial : ℕ) (n_lost : ℕ) (n_final : ℕ) 
  (h_initial : n_initial = 9) 
  (h_lost : n_lost = 2) 
  (h_final : n_final = n_initial - n_lost) : 
  n_final = 7 :=
by
  sorry

end sally_balloon_count_l657_657090


namespace intersection_M_N_l657_657044

def M := {1, 3, 5, 7, 9}

def N := {x : ℤ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} := by
  sorry

end intersection_M_N_l657_657044


namespace measure_angle_RPT_l657_657461

-- Define the points and angles
variables {Q R S P T : Point}
variables (angle_PQT angle_PTQ angle_QPR : ℝ)

-- Assumptions
axiom QRS_straight_line : same_line Q R S
axiom angle_PQT : angle Q P T = 52
axiom angle_PTQ : angle P T Q = 40
axiom angle_QPR : angle Q P R = 72

-- Theorem statement
theorem measure_angle_RPT : angle R P T = 16 :=
by
  sorry

end measure_angle_RPT_l657_657461


namespace negative_values_count_l657_657318

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l657_657318


namespace rope_lengths_correct_l657_657517

variable (x y : ℝ) -- lengths of Rope A and Rope B

-- Conditions
def remaining_rope_A_after_Allan (x : ℝ) : ℝ := 0.60 * x
def remaining_rope_A_after_Jack (x : ℝ) : ℝ := remaining_rope_A_after_Allan x - (1 / 3) * (remaining_rope_A_after_Allan x)
def remaining_rope_B_after_Maria (y : ℝ) : ℝ := 0.75 * y
def remaining_rope_B_after_Mike (y : ℝ) : ℝ := remaining_rope_B_after_Maria y - (1 / 4) * (remaining_rope_B_after_Maria y)

-- Final length definitions
def p (x : ℝ) : ℝ := remaining_rope_A_after_Jack x
def q (y : ℝ) : ℝ := remaining_rope_B_after_Mike y

-- Theorem to prove the final lengths
theorem rope_lengths_correct (x y : ℝ) : 
  p x = 0.40 * x ∧ q y = 0.5625 * y :=
by
  -- We only state the theorem here
  sorry

end rope_lengths_correct_l657_657517


namespace n_five_minus_n_divisible_by_30_l657_657890

theorem n_five_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end n_five_minus_n_divisible_by_30_l657_657890


namespace profit_per_unit_maximize_profit_l657_657611

-- Define profits per unit for type A and B phones
variables (a b : ℝ)

-- Define the conditions as lean statements for the given problem
def cond1 : Prop := a + b = 600
def cond2 : Prop := 3 * a + 2 * b = 1400

-- Define the statement to find profit per unit for type A and B phones
theorem profit_per_unit (h1 : cond1 a b) (h2 : cond2 a b) :
  a = 200 ∧ b = 400 := by
sorry

-- Define the number of phones purchased
variables (x y : ℕ)

-- Define conditions for total number of phones and number of type B phones
def total_phones : Prop := x + y = 20
def cond_phones : Prop := y ≤ (2 * x) / 3

-- Define the profit function
def profit := 200 * x + 400 * y

-- Define the statement to maximize profit with purchasing plan
theorem maximize_profit (h3 : total_phones x y) (h4 : cond_phones x) :
  profit x y = 5600 ∧ x = 12 ∧ y = 8 := by
sorry

end profit_per_unit_maximize_profit_l657_657611


namespace volume_of_cuboid_l657_657366

variable (a b c : ℝ)

def is_cuboid_adjacent_faces (a b c : ℝ) := a * b = 3 ∧ a * c = 5 ∧ b * c = 15

theorem volume_of_cuboid (a b c : ℝ) (h : is_cuboid_adjacent_faces a b c) :
  a * b * c = 15 := by
  sorry

end volume_of_cuboid_l657_657366


namespace largest_multiple_of_7_smaller_than_neg_50_l657_657149

theorem largest_multiple_of_7_smaller_than_neg_50 : ∃ n, (∃ k : ℤ, n = 7 * k) ∧ n < -50 ∧ ∀ m, (∃ j : ℤ, m = 7 * j) ∧ m < -50 → m ≤ n :=
by
  sorry

end largest_multiple_of_7_smaller_than_neg_50_l657_657149


namespace find_k_l657_657187

-- Definitions based on the conditions
def number := 24
def bigPart := 13
  
theorem find_k (x y k : ℕ) 
  (original_number : x + y = 24)
  (big_part : x = 13 ∨ y = 13)
  (equation : k * x + 5 * y = 146) : k = 7 := 
  sorry

end find_k_l657_657187


namespace num_of_negative_x_l657_657296

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l657_657296


namespace intersection_eq_l657_657025

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_eq : M ∩ N = {5, 7, 9} := sorry

end intersection_eq_l657_657025


namespace find_y_payment_l657_657141

-- Defining the conditions
def total_payment : ℝ := 700
def x_payment (y_payment : ℝ) : ℝ := 1.2 * y_payment

-- The theorem we want to prove
theorem find_y_payment (y_payment : ℝ) (h1 : y_payment + x_payment y_payment = total_payment) :
  y_payment = 318.18 := 
sorry

end find_y_payment_l657_657141


namespace red_marbles_count_l657_657818

-- Mathematical definitions based on the problem conditions
def total_marbles (yellow_marbles : ℕ) (prob_yellow : ℚ) : ℕ :=
  yellow_marbles * (1/prob_yellow).toNat

def red_marbles (total_marbles blue_marbles yellow_marbles : ℕ) : ℕ :=
  total_marbles - blue_marbles - yellow_marbles

-- Given conditions
def blue_marbles : ℕ := 7
def yellow_marbles : ℕ := 6
def prob_yellow : ℚ := 1/4

-- Target statement to prove
theorem red_marbles_count : red_marbles (total_marbles yellow_marbles prob_yellow) blue_marbles yellow_marbles = 11 :=
by
  sorry

end red_marbles_count_l657_657818


namespace meal_cost_l657_657353

theorem meal_cost (x : ℝ) (h1 : ∀ (x : ℝ), (x / 4) - 6 = x / 9) : 
  x = 43.2 :=
by
  have h : (∀ (x : ℝ), (x / 4) - (x / 9) = 6) := sorry
  exact sorry

end meal_cost_l657_657353


namespace parabola_equation_l657_657389

noncomputable def parabola_vertex_focus_dot_product 
    (O : ℝ × ℝ) (A B F : ℝ × ℝ) (p : ℝ) : Prop :=
  O = (0, 0) ∧
  F = (0, p/2) ∧
  (A.1 ^ 2 = 2 * p * A.2 ∧ B.1 ^ 2 = 2 * p * B.2) ∧
  (A.2 = B.2 + p / 2) ∧
  (O.1 * A.1 + O.2 * A.2) * (O.1 * B.1 + O.2 * B.2) = -12 ∧
  (A.1 * B.1 + A.2 * B.2) = -p^2 + p^2/4

theorem parabola_equation (O A B F : ℝ × ℝ) (p : ℝ) 
    (h : parabola_vertex_focus_dot_product O A B F p) : 
    ∃ y : ℝ, ((A.1 ^ 2 = 8 * y) ∧ (B.1 ^ 2 = 8 * y)) :=
by
  rcases h with ⟨hO, hF, hAB, hLine, hDot1, hDot2⟩
  use (A.2 + B.2) / 2
  split
  · sorry
  · sorry

end parabola_equation_l657_657389


namespace slope_of_line_through_parallelogram_l657_657230

theorem slope_of_line_through_parallelogram (p1 p2 p3 p4 l : ℝ × ℝ)
  (h_p1 : p1 = (15, 55)) (h_p2 : p2 = (15, 124)) (h_p3 : p3 = (33, 163)) (h_p4 : p4 = (33, 94))
  (h_l : l = (5, 0)) :
  let m := 109 in let n := 19 in m + n = 128 :=
by
  sorry

end slope_of_line_through_parallelogram_l657_657230


namespace area_of_triangle_APQ_l657_657963

theorem area_of_triangle_APQ :
  ∃ (b1 b2 : ℝ) (P Q : EuclideanGeometry.Point),
    let A := EuclideanGeometry.mk 4 10 in
    let P := EuclideanGeometry.mk 0 b1 in
    let Q := EuclideanGeometry.mk 0 b2 in
    b1 * b2 = 100 ∧
    EuclideanGeometry.perpendicular (EuclideanGeometry.Line.mk A P) (EuclideanGeometry.Line.mk A Q) ∧
    EuclideanGeometry.area_of_triangle A P Q = 40 :=
sorry

end area_of_triangle_APQ_l657_657963


namespace exchange_ways_l657_657218

theorem exchange_ways :
  let n := (number of nickels : ℕ)
  let h := (number of half-dollars : ℕ)
  (∃ n h, 5 * n + 50 * h = 2000 ∧ n > 0 ∧ h > 0) →
  ∃ h_set, h_set = {h | 1 ≤ h ∧ h < 40} ∧ h_set.card = 39 :=
by
  sorry

end exchange_ways_l657_657218


namespace centroids_on_circle_l657_657828

theorem centroids_on_circle (a b c d : ℂ) (R : ℝ) 
  (ha : complex.abs a = R) (hb : complex.abs b = R) 
  (hc : complex.abs c = R) (hd : complex.abs d = R) :
  let g_a := (b + c + d) / 3
      g_b := (a + c + d) / 3
      g_c := (a + b + d) / 3
      g_d := (a + b + c) / 3
      p := (a + b + c + d) / 4 in
  complex.abs (p - g_a) = R / 4 ∧
  complex.abs (p - g_b) = R / 4 ∧
  complex.abs (p - g_c) = R / 4 ∧
  complex.abs (p - g_d) = R / 4 := 
by
  sorry

end centroids_on_circle_l657_657828


namespace gcd_xyz_square_of_diff_l657_657849

theorem gcd_xyz_square_of_diff {x y z : ℕ} 
    (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) : 
    ∃ n : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = n ^ 2 :=
by
  sorry

end gcd_xyz_square_of_diff_l657_657849


namespace QNR_angle_l657_657458

noncomputable def angle_QNR (P Q R S N : Point) : ℝ :=
  if PQ_len : ∀ x ∈ [P, Q], x.distance = 8 ∧
     QR_len : ∀ y ∈ [Q, R], y.distance = 4 ∧
     PN_len : ∀ z ∈ [P, N], z.distance = 6
  then
    let QN := 8 - 6 in
    let NR := 4 in
    real.arccos ((QN ^ 2 + QR ^ 2 - NR ^ 2) / (2 * QN * QR))
  else
    0 -- default value if conditions are not met

def proof_problem : Prop :=
  ∀ (P Q R S N : Point),
    (PQ.distance = 8) ∧ (QR.distance = 4) ∧ (PN.distance = 6) ∧
    (triangle QNR Q N R).is_isosceles (QN.distance) (NR.distance) →
    angle_QNR P Q R S N = 75.5

theorem QNR_angle : proof_problem :=
  by sorry

end QNR_angle_l657_657458


namespace geometric_problem_l657_657460

theorem geometric_problem 
  (P : ℝ × ℝ) 
  (hl : ∀ x y : ℝ, (x, y) ∈ P →
    real.real.sqrt (x^2 + (y + real.sqrt(3))^2) + real.real.sqrt (x^2 + (y - real.sqrt(3))^2) = 4) 
  (l : ℝ → ℝ) 
  (hline : l 0 = 1) 
  (C : ℝ → ℝ → Prop := λ x y, x^2 + y^2 / 4 = 1) 
  (A B : ℝ × ℝ) 
  (hac : C A.1 A.2) 
  (hbc : C B.1 B.2) 
  (dot_prod_zero : A.1 * (B.1 - A.1) + A.2 * (B.2 - A.2) = 0) : 
  (∀ x y, C x y ↔ x^2 + y^2 / 4 = 1) 
  ∧ (∀ k, l = (λ x, k * x + 1) ↔ k = 1/2 ∨ k = -1/2) 
  ∧ (real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4 * real.sqrt 65 / 17) := sorry

end geometric_problem_l657_657460


namespace count_negative_values_correct_l657_657286

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l657_657286


namespace cosine_identity_l657_657721

theorem cosine_identity
  (α : ℝ)
  (h : Real.sin (π / 6 + α) = (Real.sqrt 3) / 3) :
  Real.cos (π / 3 - α) = (Real.sqrt 3) / 2 :=
by
  sorry

end cosine_identity_l657_657721


namespace xiaoming_temperature_range_xiaoming_temperature_mean_xiaoming_temperature_median_xiaoming_temperature_75th_percentile_l657_657215

noncomputable def xiaoming_temperatures : List ℝ := [36.0, 36.2, 36.1, 36.4, 36.3, 36.1, 36.3]

theorem xiaoming_temperature_range :
  List.range xiaoming_temperatures = 0.4 :=
by
  sorry

theorem xiaoming_temperature_mean :
  List.mean xiaoming_temperatures = 36.2 :=
by
  sorry

theorem xiaoming_temperature_median :
  List.median xiaoming_temperatures = 36.2 :=
by
  sorry

theorem xiaoming_temperature_75th_percentile :
  List.percentile xiaoming_temperatures 75 = 36.3 :=
by
  sorry

end xiaoming_temperature_range_xiaoming_temperature_mean_xiaoming_temperature_median_xiaoming_temperature_75th_percentile_l657_657215


namespace quartic_roots_eq_l657_657065

noncomputable def quartic_eq (p q : ℚ) : polynomial ℚ :=
  (polynomial.X^2 + polynomial.C p * polynomial.X + polynomial.C q) *
  (polynomial.X^2 - polynomial.C p * polynomial.X + polynomial.C q)

theorem quartic_roots_eq {p q : ℚ} :
  quartic_eq p q = polynomial.X^4 - polynomial.C (p^2 - 2*q) * polynomial.X^2 + polynomial.C (q^2) :=
begin
  sorry
end

end quartic_roots_eq_l657_657065


namespace triangle_area_l657_657420

/-- Given that triangle ABC is a right-angled triangle with angles 45°, 45°, and 90°, 
and the altitude from point C to the hypotenuse AB is 2 cm,
prove that the area of triangle ABC is 4 square centimeters. -/
theorem triangle_area (a b c : ℝ) (h0 : ∀ x y z, a + b + c = x + y + z)
  (h1 : ∀ x y z, x = y → y = z → x = z)
  (h2 : ∀ (t: Triangle ℝ), Triangle.is_45_45_90 t ∧ t.altitude = 2):
    Triangle.area a b c = 4 := by 
  sorry

end triangle_area_l657_657420


namespace geometric_sequence_sum_2018_l657_657754

noncomputable def geometric_sum (n : ℕ) (a1 q : ℝ) : ℝ :=
  if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_2018 :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    (∀ n, S n = geometric_sum n (a 1) 2) →
    a 1 = 1 / 2 →
    (a 1 * 2^2)^2 = 8 * a 1 * 2^3 - 16 →
    S 2018 = 2^2017 - 1 / 2 :=
by sorry

end geometric_sequence_sum_2018_l657_657754


namespace count_negative_values_correct_l657_657281

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l657_657281


namespace question1_question2_question3_l657_657727

namespace MathProofs

-- Statement for Question 1
theorem question1 (A : Set ℝ) (hA : A = {−1}) :
  (∀ x : ℝ, (−x) ≤ (−x+1) ) ∧ (∀ x : ℝ, ¬(2x ≤ 2x-2)) :=
sorry

-- Statement for Question 2
theorem question2 (A : Set ℝ) (f : ℝ → ℝ) (hA : A = Set.Ioo 0 1)
  (hf : ∀ x ∈ Set.Ici a, f(x) = x + 1/x) :
  a ∈ Set.Ici 1 :=
sorry

-- Statement for Question 3
theorem question3 (A : Set ℤ) (hA : A = {−2, m}) (hconstant : ∀ f : ℤ → ℤ , f is constant for property A) :
  m % 2 = 1 :=
sorry

end MathProofs

end question1_question2_question3_l657_657727


namespace stratified_sampling_majors_l657_657430

theorem stratified_sampling_majors (total_students : ℕ) (students_A : ℕ) (students_B : ℕ) (sample_size : ℕ) :
  total_students = 1200 →
  students_A = 380 →
  students_B = 420 →
  sample_size = 120 →
  let students_C := total_students - students_A - students_B in
  let drawn_from_C := sample_size * students_C / total_students in
  drawn_from_C = 40 :=
by
  intros h_total h_A h_B h_sample
  let students_C := total_students - students_A - students_B
  let drawn_from_C := sample_size * students_C / total_students
  have : students_C = 400 := by
    rw [h_total, h_A, h_B]
    norm_num
  have : drawn_from_C = 40 := by
    rw [this, h_sample]
    norm_num
  exact this

end stratified_sampling_majors_l657_657430


namespace find_a4_l657_657388

noncomputable def S : ℕ → ℕ
| n := 2 * n ^ 2 - 3 * n

noncomputable def a : ℕ → ℕ
| 1 := S 1
| (n + 1) := S (n + 1) - S n

theorem find_a4 : a 4 = 11 :=
by
  have hS4 : S 4 = 23 := by
    unfold S
    norm_num
  have hS3 : S 3 = 12 := by
    unfold S
    norm_num
  have ha4 : a 4 = S 4 - S 3 := by
    unfold a
    norm_num
  rw [hS4, hS3] at ha4
  exact ha4

end find_a4_l657_657388


namespace combination_divisible_by_30_l657_657892

theorem combination_divisible_by_30 (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k :=
by
  sorry

end combination_divisible_by_30_l657_657892


namespace randy_money_left_l657_657084

theorem randy_money_left (initial_money lunch ice_cream_cone remaining : ℝ) 
  (h1 : initial_money = 30)
  (h2 : lunch = 10)
  (h3 : remaining = initial_money - lunch)
  (h4 : ice_cream_cone = remaining * (1/4)) :
  (remaining - ice_cream_cone) = 15 := by
  sorry

end randy_money_left_l657_657084


namespace intersection_correct_l657_657734

def set_A : Set ℤ := {-1, 1, 2, 4}
def set_B : Set ℤ := {x | |x - 1| ≤ 1}

theorem intersection_correct :
  set_A ∩ set_B = {1, 2} :=
  sorry

end intersection_correct_l657_657734


namespace cake_remaining_l657_657956

theorem cake_remaining (T J: ℝ) (h1: T = 0.60) (h2: J = 0.25) :
  (1 - ((1 - T) * J + T)) = 0.30 :=
by
  sorry

end cake_remaining_l657_657956


namespace calculate_ratio_l657_657057

variables (p q r : ℝ) (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Assumptions about the midpoints
def midpointBC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  dist A B / 2 = p
def midpointAC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  dist A C / 2 = q
def midpointAB (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  dist B C / 2 = r

-- Main theorem to prove
theorem calculate_ratio (hp : midpointBC A B C) (hq : midpointAC A B C) (hr : midpointAB A B C) :
  (dist A B ^ 2 + dist B C ^ 2 + dist C A ^ 2) / (p ^ 2 + q ^ 2 + r ^ 2) = 8 :=
by sorry

end calculate_ratio_l657_657057


namespace polynomial_binomial_square_l657_657791

theorem polynomial_binomial_square (b : ℝ) : 
  (∃ c : ℝ, (3*X + c)^2 = 9*X^2 - 24*X + b) → b = 16 :=
by
  sorry

end polynomial_binomial_square_l657_657791


namespace largest_multiple_of_7_less_than_neg50_l657_657146

theorem largest_multiple_of_7_less_than_neg50 : ∃ x, (∃ k : ℤ, x = 7 * k) ∧ x < -50 ∧ ∀ y, (∃ m : ℤ, y = 7 * m) → y < -50 → y ≤ x :=
sorry

end largest_multiple_of_7_less_than_neg50_l657_657146


namespace find_a_l657_657118

-- Define the prime number theorem, essential for the proof
theorem find_a (a b p : ℤ) (hp : p.prime)
  (h1 : p^2 + a * p + b = 0)
  (h2 : p^2 + b * p + 1100 = 0) :
  a = 274 ∨ a = 40 :=
by
  sorry

end find_a_l657_657118


namespace jordan_rectangle_length_l657_657665

variables (L : ℝ)

-- Condition: Carol's rectangle measures 12 inches by 15 inches.
def carol_area : ℝ := 12 * 15

-- Condition: Jordan's rectangle has the same area as Carol's rectangle.
def jordan_area : ℝ := carol_area

-- Condition: Jordan's rectangle is 20 inches wide.
def jordan_width : ℝ := 20

-- Proposition: Length of Jordan's rectangle == 9 inches.
theorem jordan_rectangle_length : L * jordan_width = jordan_area → L = 9 := 
by
  intros h
  sorry

end jordan_rectangle_length_l657_657665


namespace min_bottles_milk_l657_657223

theorem min_bottles_milk 
  (fl_oz_required : ℝ)
  (bottle_size_ml : ℝ)
  (liter_to_fl_oz : ℝ)
  (liter_to_ml : ℝ)
  (fl_oz_required = 60)
  (bottle_size_ml = 250)
  (liter_to_fl_oz = 33.8)
  (liter_to_ml = 1000) :
  ∃ n : ℕ, n = 8 ∧ (n * bottle_size_ml) / liter_to_ml * liter_to_fl_oz ≥ fl_oz_required :=
by
  sorry

end min_bottles_milk_l657_657223


namespace total_tickets_sold_l657_657955

theorem total_tickets_sold (A C : ℕ) (total_revenue : ℝ) (cost_adult cost_child : ℝ) :
  (cost_adult = 6.00) →
  (cost_child = 4.50) →
  (total_revenue = 2100.00) →
  (C = 200) →
  (cost_adult * ↑A + cost_child * ↑C = total_revenue) →
  A + C = 400 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof omitted
  sorry

end total_tickets_sold_l657_657955


namespace unique_solution_l657_657272

-- Given conditions in the problem:
def prime (p : ℕ) : Prop := Nat.Prime p
def is_solution (p n k : ℕ) : Prop :=
  3 ^ p + 4 ^ p = n ^ k ∧ k > 1 ∧ prime p

-- The only solution:
theorem unique_solution (p n k : ℕ) :
  is_solution p n k → (p, n, k) = (2, 5, 2) := 
by
  sorry

end unique_solution_l657_657272


namespace total_triangles_in_figure_l657_657781

theorem total_triangles_in_figure :
  let vertices := 4 -- Vertical sections
  let hori_sections := 2 -- Horizontal sections
  let diagonal_lines := true -- Diagonal lines from corners to opposite corners
  in 
  -- Number of triangles in the described figure is 42
  number_of_triangles vertices hori_sections diagonal_lines = 42 := 
  sorry

end total_triangles_in_figure_l657_657781


namespace range_of_a_l657_657802

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a + Real.cos (2 * x) < 5 - 4 * Real.sin x + Real.sqrt (5 * a - 4)) :
  a ∈ Set.Icc (4 / 5) 8 :=
sorry

end range_of_a_l657_657802


namespace sum_complementary_events_l657_657687

noncomputable theory
open Set

variables (Ω : Type*) (A : Set Ω) [MeasurableSpace Ω]

theorem sum_complementary_events (P : MeasureTheory.Measure Ω) (hA : MeasurableSet A) :
  P A + P (Ω \ A) = 1 :=
begin
  -- Conditions
  have h1 : MeasurableSet (Ω \ A), from MeasurableSet.compl hA,
  have h2 : P (univ : Set Ω) = 1, from MeasureTheory.Measure.measure_univ P,

  -- Proof goes here
  sorry
end

end sum_complementary_events_l657_657687


namespace sum_of_angles_l657_657275

theorem sum_of_angles (x u v : ℝ) (h1 : u = Real.sin x) (h2 : v = Real.cos x)
  (h3 : 0 ≤ x ∧ x ≤ 2 * Real.pi) 
  (h4 : Real.sin x ^ 4 - Real.cos x ^ 4 = (u - v) / (u * v)) 
  : x = Real.pi / 4 ∨ x = 5 * Real.pi / 4 → (Real.pi / 4 + 5 * Real.pi / 4) = 3 * Real.pi / 2 := 
by
  intro h
  sorry

end sum_of_angles_l657_657275


namespace maximum_value_of_x_minus_y_is_sqrt8_3_l657_657494

variable {x y z : ℝ}

noncomputable def maximum_value_of_x_minus_y (x y z : ℝ) : ℝ :=
  x - y

theorem maximum_value_of_x_minus_y_is_sqrt8_3 (h1 : x + y + z = 2) (h2 : x * y + y * z + z * x = 1) : 
  maximum_value_of_x_minus_y x y z = Real.sqrt (8 / 3) :=
sorry

end maximum_value_of_x_minus_y_is_sqrt8_3_l657_657494


namespace number_of_negative_x_l657_657334

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l657_657334


namespace hyperbola_eccentricity_l657_657765

theorem hyperbola_eccentricity (m : ℝ) (h : eccentricity (hyperbola_eq m) = sqrt 5) : m = 2 :=
by
  -- Definitions to set up the problem
  def a (m : ℝ) : ℝ := sqrt m
  def b (m : ℝ) : ℝ := sqrt (m^2 + 4)
  def c (m : ℝ) : ℝ := sqrt (a m ^ 2 + b m ^ 2)
  def hyperbola_eq (m : ℝ) (x y : ℝ) : Prop := (x^2 / m) - (y^2 / (m^2 + 4)) = 1
  def eccentricity (h : (ℝ → ℝ → Prop)) : ℝ := c m / a m

  -- Skip the proof
  sorry

end hyperbola_eccentricity_l657_657765


namespace helmet_costs_unique_max_purchase_profit_l657_657903

def helmet_cost_eqs (x y : ℕ) :=
  8 * x + 6 * y = 630 ∧ 6 * x + 8 * y = 700

def valid_purchase_plan (m : ℕ) :=
  30 * m + 65 * (200 - m) ≤ 10200 ∧ (28 * m + 33 * (200 - m) ≥ 6180) 

def plan_profit (m : ℕ) :=
  28 * m + 33 * (200 - m)

theorem helmet_costs_unique :
  ∃ (x y : ℕ), helmet_cost_eqs x y ∧ x = 30 ∧ y = 65 :=
sorry

theorem max_purchase_profit : 
  ∃ (count : ℕ), count = (list.range' 80 5).filter (λ m, valid_purchase_plan m) ∧
  ∃ (m : ℕ), m ∈ (list.range' 80 5).filter (λ m, valid_purchase_plan m) ∧
  ∀ n ∈ (list.range' 80 5).filter (λ m, valid_purchase_plan m), plan_profit m ≥ plan_profit n ∧
  plan_profit m = 6200 :=
sorry

end helmet_costs_unique_max_purchase_profit_l657_657903


namespace total_duration_is_2_9167_l657_657648

def part1_distance : ℝ := 75
def part1_speed : ℝ := 50
def part2_stop_time : ℝ := 0.25
def part2_distance : ℝ := 50
def part2_speed : ℝ := 75
def part3_time : ℝ := 0.5

def total_duration_of_journey (d1 d2 stop_t t3 s1 s2 : ℝ) : ℝ :=
  (d1 / s1) + stop_t + (d2 / s2) + t3

theorem total_duration_is_2_9167 :
  total_duration_of_journey part1_distance part2_distance part2_stop_time part3_time part1_speed part2_speed = 2.9167 := by
  sorry

end total_duration_is_2_9167_l657_657648


namespace tricycle_wheel_count_l657_657136

theorem tricycle_wheel_count (bicycles wheels_per_bicycle tricycles total_wheels : ℕ)
  (h1 : bicycles = 16)
  (h2 : wheels_per_bicycle = 2)
  (h3 : tricycles = 7)
  (h4 : total_wheels = 53)
  (h5 : total_wheels = (bicycles * wheels_per_bicycle) + (tricycles * (3 : ℕ))) : 
  (3 : ℕ) = 3 := by
  sorry

end tricycle_wheel_count_l657_657136


namespace find_a_l657_657122

theorem find_a (a b : ℤ) (p : ℕ) (hp : prime p) (h1 : p^2 + a * p + b = 0) (h2 : p^2 + b * p + 1100 = 0) :
  a = 274 ∨ a = 40 :=
sorry

end find_a_l657_657122


namespace negative_values_of_x_l657_657306

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l657_657306


namespace average_of_second_pair_l657_657106

theorem average_of_second_pair (S : ℝ) (S1 : ℝ) (S3 : ℝ) (S2 : ℝ) (avg : ℝ) :
  (S / 6 = 3.95) →
  (S1 / 2 = 3.8) →
  (S3 / 2 = 4.200000000000001) →
  (S = S1 + S2 + S3) →
  (avg = S2 / 2) →
  avg = 3.85 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end average_of_second_pair_l657_657106


namespace max_leftover_candies_l657_657775

-- Given conditions as definitions
def pieces_of_candy := ℕ
def num_bags := 11

-- Statement of the problem
theorem max_leftover_candies (x : pieces_of_candy) (h : x % num_bags ≠ 0) :
  x % num_bags = 10 :=
sorry

end max_leftover_candies_l657_657775


namespace find_point_coordinates_l657_657379

theorem find_point_coordinates :
  ∃ P : ℝ × ℝ × ℝ, 
    P = (0, P.2, 0) ∧
    (let A := (2, -1, 4) in let B := (-1, 2, 5) in 
    real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2 + (A.3 - P.3)^2) = 
    real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2 + (B.3 - P.3)^2)) ∧ 
    P = (0, 3/2, 0) :=
sorry

end find_point_coordinates_l657_657379


namespace decreasing_interval_ln_quadratic_l657_657927

theorem decreasing_interval_ln_quadratic :
  ∀ x : ℝ, (x < 1 ∨ x > 3) → (∀ a b : ℝ, (a ≤ b) → (a < 1 ∨ a > 3) → (b < 1 ∨ b > 3) → (a ≤ x ∧ x ≤ b → (x^2 - 4 * x + 3) ≥ (b^2 - 4 * b + 3))) :=
by
  sorry

end decreasing_interval_ln_quadratic_l657_657927


namespace area_of_rhombus_150_l657_657688

variable (EFG_radius EGH_radius : ℕ)
variable (EG_diagonal FH_diagonal : ℕ)

theorem area_of_rhombus_150 (EFG_radius = 15) (EGH_radius = 30) (EG_diagonal = 3 * FH_diagonal) :
  let d1 := 2 * FH_diagonal
  let d2 := 2 * EG_diagonal
  ((d1 * d2) / 2) = 150 :=
by
  sorry

end area_of_rhombus_150_l657_657688


namespace min_value_AP_AQ_l657_657375

noncomputable def min_distance (A P Q : ℝ × ℝ) : ℝ := dist A P + dist A Q

theorem min_value_AP_AQ :
  ∀ (A P Q : ℝ × ℝ),
    (∀ (x : ℝ), A = (x, 0)) →
    ((P.1 - 1) ^ 2 + (P.2 - 3) ^ 2 = 1) →
    ((Q.1 - 7) ^ 2 + (Q.2 - 5) ^ 2 = 4) →
    min_distance A P Q = 7 :=
by
  intros A P Q hA hP hQ
  -- Proof is to be provided here
  sorry

end min_value_AP_AQ_l657_657375


namespace find_probability_B_find_probability_nA_B_l657_657255

variables {Ω : Type*} [probability_space Ω]

variables {P : event Ω → ℝ}

variable {A B C : event Ω}

axioms (independent : ∀ {a b c : event Ω}, P(a) * P(b) * P(c) = P(a ∧ b) * P(a ∧ c) * P(b ∧ c))
       (P_AB   : P(A ∧ B) = 1/6)
       (P_nB_C : P(¬B ∧ C) = 1/8)
       (P_ABC  : P(A ∧ B ∧ ¬C) = 1/8)

theorem find_probability_B : P(B) = 1/2 :=
sorry

theorem find_probability_nA_B : P(¬A ∧ B) = 1/3 :=
sorry

end find_probability_B_find_probability_nA_B_l657_657255


namespace intersection_correct_l657_657017

-- Define the sets M and N
def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x | 2 * x > 7}

-- Define the expected intersection result
def expected_intersection : Set ℝ := {5, 7, 9}

-- State the theorem
theorem intersection_correct : ∀ x, x ∈ M ∩ N ↔ x ∈ expected_intersection :=
by
  sorry

end intersection_correct_l657_657017


namespace incorrect_expression_l657_657674

noncomputable def D (P Q : ℕ) (r s : ℕ) : ℝ := P * (10^(-r : ℝ)) + Q * (10^(-r : ℝ)) / (1 - 10^(-s : ℝ))

theorem incorrect_expression (P Q : ℕ) (r s : ℕ) (D : ℝ) 
  (h1 : D = (P : ℝ) * (10 : ℝ)^-r + (Q : ℝ) / ((1 : ℝ) - (10 : ℝ)^-s)) :
  10^r * (10^s - 1) * D ≠ Q * (P - 10) :=
sorry

end incorrect_expression_l657_657674


namespace pipe_a_filling_time_l657_657880

theorem pipe_a_filling_time :
  let rate_A := 1 / 10
    let rate_L := 1 / 30
    let net_rate := rate_A - rate_L
    let Time := 1 / net_rate
  in Time = 15 :=
by
  let rate_A := 1 / 10
  let rate_L := 1 / 30
  let net_rate := rate_A - rate_L
  let Time := 1 / net_rate
  have hA : rate_A = 1 / 10 := rfl
  have hL : rate_L = 1 / 30 := rfl
  have hNetRate : net_rate = rate_A - rate_L := rfl
  have hTime : Time = 1 / net_rate := rfl
  sorry

end pipe_a_filling_time_l657_657880


namespace rectangle_shaded_area_equal_l657_657214

theorem rectangle_shaded_area_equal {x : ℝ} :
  let total_area := 72
  let shaded_area := 24 + 6*x
  let non_shaded_area := total_area / 2
  shaded_area = non_shaded_area → x = 2 := 
by 
  intros h
  sorry

end rectangle_shaded_area_equal_l657_657214


namespace percentage_error_in_side_length_l657_657209

theorem percentage_error_in_side_length 
  (A A' s s' : ℝ) (h₁ : A = s^2)
  (h₂ : A' = s'^2)
  (h₃ : ((A' - A) / A * 100) = 12.36) :
  ∃ E : ℝ, (s' = s * (1 + E / 100)) ∧ (E = (real.sqrt 1.1236 - 1) * 100) := 
by
  sorry

end percentage_error_in_side_length_l657_657209


namespace tony_income_l657_657959

-- Definitions for the given conditions
def investment : ℝ := 3200
def purchase_price : ℝ := 85
def dividend : ℝ := 6.640625

-- Theorem stating Tony's income based on the conditions
theorem tony_income : (investment / purchase_price) * dividend = 250 :=
by
  sorry

end tony_income_l657_657959


namespace josephs_total_cards_l657_657472

def number_of_decks : ℕ := 4
def cards_per_deck : ℕ := 52
def total_cards : ℕ := number_of_decks * cards_per_deck

theorem josephs_total_cards : total_cards = 208 := by
  sorry

end josephs_total_cards_l657_657472


namespace ked_ben_eggs_ratio_l657_657066

theorem ked_ben_eggs_ratio 
  (saly_needs_ben_weekly_ratio : ℕ)
  (weeks_in_month : ℕ := 4) 
  (total_production_month : ℕ := 124)
  (saly_needs_weekly : ℕ := 10) 
  (ben_needs_weekly : ℕ := 14)
  (ben_needs_monthly : ℕ := ben_needs_weekly * weeks_in_month)
  (saly_needs_monthly : ℕ := saly_needs_weekly * weeks_in_month)
  (total_saly_ben_monthly : ℕ := saly_needs_monthly + ben_needs_monthly)
  (ked_needs_monthly : ℕ := total_production_month - total_saly_ben_monthly)
  (ked_needs_weekly : ℕ := ked_needs_monthly / weeks_in_month) :
  ked_needs_weekly / ben_needs_weekly = 1 / 2 :=
sorry

end ked_ben_eggs_ratio_l657_657066


namespace min_sticks_cover_200cm_l657_657981

def length_covered (n6 n7 : ℕ) : ℕ :=
  6 * n6 + 7 * n7

theorem min_sticks_cover_200cm :
  ∃ (n6 n7 : ℕ), length_covered n6 n7 = 200 ∧ (∀ (m6 m7 : ℕ), (length_covered m6 m7 = 200 → m6 + m7 ≥ n6 + n7)) ∧ (n6 + n7 = 29) :=
sorry

end min_sticks_cover_200cm_l657_657981


namespace count_negative_x_with_sqrt_pos_int_l657_657346

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l657_657346


namespace circles_intersect_l657_657550

theorem circles_intersect :
  ∀ (x y : ℝ),
  (x^2 + y^2 = 9) → 
  (x^2 + y^2 + 6*x - 8*y - 11 = 0) →
  ∃ (d : ℝ),
  let R := 6 in
  let r := 3 in
  let center1 := (0, 0) in
  let center2 := (-3, 4) in
  let dist := ( (center1.1 - center2.1) ^ 2 + (center1.2 - center2.2) ^ 2 ).sqrt in
  (R - r < dist) ∧ (dist < R + r) :=
  sorry

end circles_intersect_l657_657550


namespace problem_solution_l657_657363

-- Definition of the problem
theorem problem_solution :
  let f := (x^2 + 1) * (x - 1)^9
  let a := f.coeffs
in a 2 = -37 ∧
     max (list_map (λ (i : ℕ), a i) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) = 210 ∧
     ((a 1 + 3*a 3 + 5*a 5 + 7*a 7 + 9*a 9 + 11*a 11)^2 - 
      (2*a 2 + 4*a 4 + 6*a 6 + 8*a 8 + 10*a 10)^2 = 0) :=
by
  sorry

end problem_solution_l657_657363


namespace circle_arc_length_l657_657992

theorem circle_arc_length (n : ℕ) (hc : n > 0) : 
  ∃ (C : set (ℝ × ℝ)), 
    (∀ c ∈ C, metric.sphere c 1 ∈ C) ∧ 
    (∃ θ > (2 * Real.pi / n), 
      ∀ c1 c2 ∈ C, 
        c1 ≠ c2 → 
        metric.sphere c1 1 ∩ metric.sphere c2 1 = ∅) :=
by 
  sorry

end circle_arc_length_l657_657992


namespace maxValue_is_6084_over_17_l657_657058

open Real

noncomputable def maxValue (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4

theorem maxValue_is_6084_over_17 (x y : ℝ) (h : x + y = 5) :
  maxValue x y h ≤ 6084 / 17 := 
sorry

end maxValue_is_6084_over_17_l657_657058


namespace inequality_and_equality_condition_l657_657083

theorem inequality_and_equality_condition (a b : ℝ) :
  a^2 + 4 * b^2 + 4 * b - 4 * a + 5 ≥ 0 ∧ (a^2 + 4 * b^2 + 4 * b - 4 * a + 5 = 0 ↔ (a = 2 ∧ b = -1 / 2)) :=
by
  sorry

end inequality_and_equality_condition_l657_657083


namespace proper_subsets_of_A_inter_B_when_a_is_1_range_of_a_when_A_inter_B_has_4_subsets_l657_657855

-- Define sets and condition for problem (1)
def A := {x : ℤ | x^2 < 9}
def B (a : ℤ) := {x : ℤ | 2 * x > a}

-- Problem (1) : proper subsets of A ∩ B when a = 1
theorem proper_subsets_of_A_inter_B_when_a_is_1 : 
  (B 1 ∩ A = {1, 2} ∧ (∀ S ∈ ({∅, {1}, {2}} : set (set ℤ)), S ⊂ (B 1 ∩ A))) :=
by sorry

-- Problem (2) : range of values for a when A ∩ B has 4 subsets
theorem range_of_a_when_A_inter_B_has_4_subsets : 
  (∀ (a : ℤ), A ∩ B a = {1, 2} → 0 ≤ a ∧ a < 2) :=
by sorry

end proper_subsets_of_A_inter_B_when_a_is_1_range_of_a_when_A_inter_B_has_4_subsets_l657_657855


namespace smallest_k_divisible_by_30_l657_657837

/-- Let p be the largest prime with 2023 digits.
    Prove that the smallest positive integer k such that p^2 - k
    is divisible by 30 is 1. -/
theorem smallest_k_divisible_by_30 (p : ℕ) (hp : prime p) (digits : p.digits = 2023) :
  ∃ k : ℕ, k > 0 ∧ (p^2 - k) % 30 = 0 ∧ (∀ m : ℕ, m > 0 ∧ (p^2 - m) % 30 = 0 → m ≥ k) :=
by 
  use 1
  sorry  -- The proof goes here

end smallest_k_divisible_by_30_l657_657837


namespace speed_difference_is_45_mph_l657_657471

open Function

-- Assume the distance
def distance : ℝ := 15

-- Define Jim's travel time in hours
def jim_travel_time_hours : ℝ := 15 / 60

-- Define Sarah's travel time in hours
def sarah_travel_time_hours : ℝ := 1

-- Define Jim's average speed
def jim_average_speed : ℝ := distance / jim_travel_time_hours

-- Define Sarah's average speed
def sarah_average_speed : ℝ := distance / sarah_travel_time_hours

-- The statement to prove that the difference in their speeds is 45 mph
theorem speed_difference_is_45_mph (h1 : jim_average_speed = 60) (h2 : sarah_average_speed = 15) : jim_average_speed - sarah_average_speed = 45 :=
  by
  rw [h1, h2]
  norm_num
  sorry

end speed_difference_is_45_mph_l657_657471


namespace sum_of_cubes_is_zero_l657_657221

theorem sum_of_cubes_is_zero :
  ∑ i in (Finset.range 50).map (Function.Embedding.coe Nat.succ), i^3 + ∑ j in (Finset.range 50).map (Function.Embedding.coe (λ n => -(n:ℤ) - 1)), j^3 = 0 :=
by
  sorry

end sum_of_cubes_is_zero_l657_657221


namespace only_one_solution_l657_657009

def f (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1
  else 3 * n

theorem only_one_solution : ∀ n ∈ Icc 1 200, (∃ k, (f^[k] n) = 1) ↔ n = 1 :=
by
  intro n hn
  sorry

end only_one_solution_l657_657009


namespace set_intersection_example_l657_657741

theorem set_intersection_example (A : Set ℝ) (B : Set ℝ):
  A = { -1, 1, 2, 4 } → 
  B = { x | |x - 1| ≤ 1 } → 
  A ∩ B = {1, 2} :=
by
  intros hA hB
  sorry

end set_intersection_example_l657_657741


namespace Deepak_age_l657_657986

theorem Deepak_age (A D : ℕ) (h1 : A / D = 4 / 3) (h2 : A + 6 = 26) : D = 15 :=
by
  sorry

end Deepak_age_l657_657986


namespace grouping_count_l657_657448

theorem grouping_count (men women : ℕ) 
  (h_men : men = 4) (h_women : women = 5)
  (at_least_one_man_woman : ∀ (g1 g2 g3 : Finset (Fin 9)), 
    g1.card = 3 → g2.card = 3 → g3.card = 3 → g1 ∩ g2 = ∅ → g2 ∩ g3 = ∅ → g3 ∩ g1 = ∅ → 
    (g1 ∩ univ.filter (· < 4)).nonempty ∧ (g1 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g2 ∩ univ.filter (· < 4)).nonempty ∧ (g2 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g3 ∩ univ.filter (· < 4)).nonempty ∧ (g3 ∩ univ.filter (· ≥ 4)).nonempty) :
  (choose 4 1 * choose 5 2 * choose 3 1 * choose 3 2) / 2! = 180 :=
sorry

end grouping_count_l657_657448


namespace parallelogram_area_l657_657073

theorem parallelogram_area :
  ∀ (A B C D : Type) [EuclideanGeometry A B C D],
    ∀ (AB AD : ℝ) (angle_BAD : ℝ) (area : ℝ),
      AB = 12 ∧ AD = 10 ∧ angle_BAD = 150 ∧ parallelogram A B C D →
      area = 60 :=
by sorry

end parallelogram_area_l657_657073


namespace find_f_find_range_l657_657801

section
variables {A ω φ : ℝ} (x : ℝ)
def f := A * Real.sin (ω * x + φ)
def is_minimum (y : ℝ) := ∃ x, f A ω φ x = y
def passes_through_origin (y : ℝ) := f A ω φ 0 = y
def passes_through_point (pt : ℝ × ℝ) := f A ω φ pt.1 = pt.2
def is_monotonically_increasing (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f A ω φ x < f A ω φ y

theorem find_f :
  (A > 0) ∧ (ω > 0) ∧ (|φ| < π / 2) ∧
  is_minimum A (-2) ∧
  passes_through_origin (sqrt 3) ∧
  passes_through_point (5*π/6, 0) ∧
  is_monotonically_increasing 0 (π/6) → 
  ∃ ω : ℝ, ω = 4 / 5 ∧ φ = π / 3 ∧ ∀ x, f 2 ω (π / 3) x = 2 * Real.sin((4 / 5) * x + π / 3) :=
by sorry

theorem find_range (x : ℝ) :
  0 ≤ x ∧ x ≤ 5*π/8 →
  (∀ y : ℝ, 1 ≤ y ∧ y ≤ 2) →
  1 ≤ f 2 (4 / 5) (π / 3) x ∧ f 2 (4 / 5) (π / 3) x ≤ 2 :=
by sorry
end

end find_f_find_range_l657_657801


namespace combination_divisible_by_30_l657_657893

theorem combination_divisible_by_30 (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k :=
by
  sorry

end combination_divisible_by_30_l657_657893


namespace darius_scores_less_l657_657237

variable (D M Ma : ℕ)

-- Conditions
def condition1 := D = 10
def condition2 := Ma = D + 3
def condition3 := D + M + Ma = 38

-- Theorem to prove
theorem darius_scores_less (D M Ma : ℕ) (h1 : condition1 D) (h2 : condition2 D Ma) (h3 : condition3 D M Ma) : M - D = 5 :=
by
  sorry

end darius_scores_less_l657_657237


namespace sin_cos_angle_eq_l657_657819

theorem sin_cos_angle_eq {A B C : ℝ} (hABC : A + B + C = π) :
  sin C = cos A + cos B ↔ A = π / 2 ∨ B = π / 2 :=
by
  sorry

end sin_cos_angle_eq_l657_657819


namespace maximum_value_of_piecewise_function_l657_657925

noncomputable def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + 3 else 
  if 0 < x ∧ x ≤ 1 then x + 3 else 
  -x + 5

theorem maximum_value_of_piecewise_function : ∃ M, ∀ x, piecewise_function x ≤ M ∧ (∀ y, (∀ x, piecewise_function x ≤ y) → M ≤ y) := 
by
  use 4
  sorry

end maximum_value_of_piecewise_function_l657_657925


namespace evaluate_expression_l657_657745

theorem evaluate_expression (a b c : ℝ) (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 :=
by 
  sorry

end evaluate_expression_l657_657745


namespace kitten_puppy_bites_ratio_l657_657627

-- Definitions based on conditions
def total_length (ℓ : ℝ) := ℓ > 0
def kitten_bite (ℓ : ℝ) := ℓ * (3 / 4)
def puppy_bite (ℓ : ℝ) := ℓ * (2 / 3)

theorem kitten_puppy_bites_ratio (ℓ : ℝ) (h : total_length ℓ) :
  let b₁ := kitten_bite ℓ / 2 in
  let b₂ := kitten_bite ℓ / 2 * (3 / 4) in
  b₁ = b₂ :=
by {
  sorry
}

end kitten_puppy_bites_ratio_l657_657627


namespace problem1_problem2_problem3_l657_657142

-- Problem 1 Equivalent Proof Problem
theorem problem1 (x : ℝ) :
  let f (x : ℝ) := x * |x - 2|
  in f (sqrt 2 - x) ≤ f 1 → x ≥ -1 :=
sorry

-- Problem 2 Equivalent Proof Problem
theorem problem2 (t : ℝ) (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_func : ∀ x, x > 0 → f x = x^4)
  (ineq : ∀ x ∈ set.Icc 1 16, f (x + t) ≤ 4 * f x) :
  t ≤ sqrt 2 - 1 :=
sorry

-- Problem 3 Equivalent Proof Problem
theorem problem3 (x : ℝ) : 
  let f (x : ℝ) := if x > 1 then 2 else (x - 1)^2 + 2
  in f (1 - x^2) > f (2 * x) → x < -1 - sqrt 2 ∨ x > -1 + sqrt 2 :=
sorry

end problem1_problem2_problem3_l657_657142


namespace minimum_points_l657_657642

-- Define the problem of marking points on a regular polygon's perimeter
def canMarkPoints (n : ℕ) (k : ℕ) : Prop :=
  ∀ (P : ℕ → Prop), (P = λ x, (x = n)) →
  ∃ (points : list ℤ),
  (points.length = k) ∧
  (∀ (Q : ℕ → Prop), ¬ (Q = λ y, (y ≠ n) → (∀ p ∈ points, Q p)))

theorem minimum_points (n : ℕ) : canMarkPoints n 5 :=
sorry

end minimum_points_l657_657642


namespace f_eq_91_for_all_n_leq_100_l657_657845

noncomputable def f : ℤ → ℝ := sorry

theorem f_eq_91_for_all_n_leq_100 (n : ℤ) (h : n ≤ 100) : f n = 91 := sorry

end f_eq_91_for_all_n_leq_100_l657_657845


namespace num_of_negative_x_l657_657290

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l657_657290


namespace find_fraction_abs_l657_657418

-- Define the conditions and the main proof problem
theorem find_fraction_abs (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 5 * x * y) :
  abs ((x + y) / (x - y)) = Real.sqrt ((7 : ℝ) / 3) :=
by
  sorry

end find_fraction_abs_l657_657418


namespace Samantha_rectangle_longest_side_l657_657091

noncomputable def longest_side_length (P: ℝ) (A: ℝ) : ℝ :=
  let l := (120 - real.sqrt (14400 - 4 * A)) / 2 in
  let w := (120 + real.sqrt (14400 - 4 * A)) / 2 in
  max l w

theorem Samantha_rectangle_longest_side:
  let P := 240 / 2 in
  let A := 240 * 8 / 2 in
  longest_side_length P A = 80 := by
  sorry

end Samantha_rectangle_longest_side_l657_657091


namespace probability_of_triangle_in_14_gon_l657_657370

theorem probability_of_triangle_in_14_gon :
  let segments := {k | k ∈ (finset.range 7).image (λ k, 2 * real.sin (k * real.pi / 14))} in
  let total_segments := 91 in
  let possible_triangles := finset.filter (λ s, s.card = 3 ∧ ∀ {a b c : ℝ}, (a ∈ s → b ∈ s → c ∈ s → a + b > c ∧ a + c > b ∧ b + c > a)) 
                     (finset.powerset segments) in
  (possible_triangles.card : ℚ) / (finset.card segmentschoose 3) = 77 / 91 :=
sorry

end probability_of_triangle_in_14_gon_l657_657370


namespace find_z_when_w_15_l657_657417

-- Define a direct variation relationship
def varies_directly (z w : ℕ) (k : ℕ) : Prop :=
  z = k * w

-- Using the given conditions and to prove the statement
theorem find_z_when_w_15 :
  ∃ k, (varies_directly 10 5 k) → (varies_directly 30 15 k) :=
by
  sorry

end find_z_when_w_15_l657_657417


namespace exists_monomial_l657_657978

variables (x y : ℕ) -- Define x and y as natural numbers

theorem exists_monomial :
  ∃ (c : ℕ) (e_x e_y : ℕ), c = 3 ∧ e_x + e_y = 3 ∧ (c * x ^ e_x * y ^ e_y) = (3 * x ^ e_x * y ^ e_y) :=
by
  sorry

end exists_monomial_l657_657978


namespace side_length_of_S2_is_1001_l657_657887

-- Definitions and Conditions
variables (R1 R2 : Type) (S1 S2 S3 : Type)
variables (r s : ℤ)
variables (h_total_width : 2 * r + 3 * s = 4422)
variables (h_total_height : 2 * r + s = 2420)

theorem side_length_of_S2_is_1001 (R1 R2 S1 S2 S3 : Type) (r s : ℤ)
  (h_total_width : 2 * r + 3 * s = 4422)
  (h_total_height : 2 * r + s = 2420) : s = 1001 :=
by
  sorry -- proof to be provided

end side_length_of_S2_is_1001_l657_657887


namespace sufficient_but_not_necessary_condition_l657_657757

variable (x : ℝ)

def p : Prop := (x - 1) / (x + 2) ≥ 0
def q : Prop := (x - 1) * (x + 2) ≥ 0

theorem sufficient_but_not_necessary_condition : (p x) → (q x) ∧ ¬((q x) → (p x)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l657_657757
