import Mathlib

namespace carrots_total_l440_440178

theorem carrots_total (sandy_carrots : Nat) (sam_carrots : Nat) (h1 : sandy_carrots = 6) (h2 : sam_carrots = 3) :
  sandy_carrots + sam_carrots = 9 :=
by
  sorry

end carrots_total_l440_440178


namespace range_of_m_l440_440202

open Real

noncomputable def f : ℝ → ℝ := sorry
def I : Set ℝ := Set.Icc 1 3

theorem range_of_m 
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_decreasing : ∀ x y : ℝ, (0 < x ∧ x < y) → f y ≤ f x)
  (h_inequality : ∀ x : ℝ, x ∈ I → f(2 * m * x - log x - 3) ≥ 2 * f 3 - f(log x + 3 - 2 * m * x)):
  (1 / (2 * exp 1) ≤ m ∧ m ≤ (6 + log 3) / 6) := 
sorry

end range_of_m_l440_440202


namespace dice_sum_probability_l440_440772

theorem dice_sum_probability :
  let D := finset.range 1 7  -- outcomes of a fair six-sided die
  (∃! d1 d2 d3 d4 ∈ D, d1 + d2 + d3 + d4 = 24) ->
  (probability(space, {ω ∈ space | (ω 1 = 6) ∧ (ω 2 = 6) ∧ (ω 3 = 6) ∧ (ω 4 = 6)}) = 1/1296) :=
sorry

end dice_sum_probability_l440_440772


namespace gary_initial_money_l440_440774

theorem gary_initial_money (spent left : ℕ) (h_spent : spent = 55) (h_left : left = 18) :
  spent + left = 73 :=
by
  rw [h_spent, h_left]
  exact rfl

end gary_initial_money_l440_440774


namespace john_profit_proof_l440_440897

-- Define the conditions
variables 
  (parts_cost : ℝ := 800)
  (selling_price_multiplier : ℝ := 1.4)
  (monthly_build_quantity : ℝ := 60)
  (monthly_rent : ℝ := 5000)
  (monthly_extra_expenses : ℝ := 3000)

-- Define the computed variables based on conditions
def selling_price_per_computer := parts_cost * selling_price_multiplier
def total_revenue := monthly_build_quantity * selling_price_per_computer
def total_cost_of_components := monthly_build_quantity * parts_cost
def total_expenses := monthly_rent + monthly_extra_expenses
def profit_per_month := total_revenue - total_cost_of_components - total_expenses

-- The theorem statement of the proof
theorem john_profit_proof : profit_per_month = 11200 := 
by
  sorry

end john_profit_proof_l440_440897


namespace find_side_and_area_l440_440498

-- Conditions
variables {A B C a b c : ℝ} (S : ℝ)
axiom angle_sum : A + B + C = Real.pi
axiom side_a : a = 4
axiom side_b : b = 5
axiom angle_relation : C = 2 * A

-- Proven equalities
theorem find_side_and_area :
  c = 6 ∧ S = 5 * 6 * (Real.sqrt 7) / 4 / 2 := by
  sorry

end find_side_and_area_l440_440498


namespace int_solution_eqn_l440_440000

noncomputable def solvable_integer_pairs : set (ℤ × ℤ) :=
  { (x, y) | y ^ 5 + 2 * x * y = x ^ 2 + 2 * y ^ 4 }

theorem int_solution_eqn :
  solvable_integer_pairs = { (0, 0), (1, 1), (0, 2), (4, 2) } :=
  sorry

end int_solution_eqn_l440_440000


namespace find_x_intercept_of_rotated_line_l440_440162

def line_m := {x y : ℝ // 4 * x - 3 * y + 20 = 0}
def rotation_point := (10, 10 : ℝ)
def rotation_angle := 60 * (Real.pi / 180)

def x_intercept_of_rotated_line (x_int : ℝ) : Prop :=
  ∃ n : ℝ → ℝ, (∀ x y : ℝ, (4 * x - 3 * y + 20 = 0) → n = ((rotation_angle.mul x - rotation_angle.mul y) / (rotation_point.1*(Real.sin rotation_angle) + rotation_point.2*(Real.cos rotation_angle)))) ∧
    (∀ x y : ℝ, (rotation_point.1, rotation_point.2) → y = n x) ∧
    (∃ x' : ℝ, y = 0 ∧ x = x_int)

theorem find_x_intercept_of_rotated_line :
  x_intercept_of_rotated_line 15 :=
sorry

end find_x_intercept_of_rotated_line_l440_440162


namespace unique_right_triangle_solution_l440_440378

-- Definitions for the given problem
variables {a b c d : ℝ}

-- Conditions: one leg b and the difference d = a + b - c
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem unique_right_triangle_solution
  (h1 : b > 0)
  (h2 : is_right_triangle a b c)
  (h3 : a + b - c = d)
  (h4 : d < a) :
  ∃ (A B C : ℝ × ℝ),
  (is_right_triangle a b c) ∧ 
  (A ≠ B) ∧
  (B ≠ C) ∧
  (A ≠ C) ∧
  (∠ABC = 90 °) :=
sorry

end unique_right_triangle_solution_l440_440378


namespace probability_of_four_ones_approx_l440_440248

noncomputable def probability_of_four_ones_in_twelve_dice : ℚ :=
  (nat.choose 12 4 : ℚ) * (1 / 6 : ℚ) ^ 4 * (5 / 6 : ℚ) ^ 8

theorem probability_of_four_ones_approx :
  probability_of_four_ones_in_twelve_dice ≈ 0.089 :=
sorry

end probability_of_four_ones_approx_l440_440248


namespace solve_system_of_equations_in_nat_numbers_l440_440185

theorem solve_system_of_equations_in_nat_numbers :
  ∃ a b c d : ℕ, a * b = c + d ∧ c * d = a + b ∧ a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 :=
by
  sorry

end solve_system_of_equations_in_nat_numbers_l440_440185


namespace volume_circumscribed_sphere_regular_pyramid_l440_440891

noncomputable def volume_of_sphere (SA : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (3 ^ 3)

theorem volume_circumscribed_sphere_regular_pyramid :
  let S := 2 * Real.sqrt 3 in volume_of_sphere S = 36 * Real.pi :=
by
  sorry

end volume_circumscribed_sphere_regular_pyramid_l440_440891


namespace incorrect_equation_in_list_l440_440348

theorem incorrect_equation_in_list :
  ¬ (sqrt (121 / 225) = - (11 / 15)) ∧
  ¬ (sqrt (121 / 225) = 11 / 15) :=
by
  let sqrt_64 := 8
  -- Given conditions (equations to verify):
  let eq1 := (sqrt_64 = 8) ∨ (sqrt_64 = -8) -- A. $\pm \sqrt{64}= \pm 8$
  let eq2 := (sqrt (121 / 225) = 11 / 15) ∨ (sqrt (121 / 225) = - (11 / 15)) -- B. $\sqrt{\dfrac{121}{225}}= \pm \dfrac{11}{15}$
  let eq3 := (Real.cbrt (-216) = -6) -- C. $\sqrt[3]{-216}= -6$
  let eq4 := (Real.cbrt (0.001) = 0.1 ∧ - Real.cbrt (0.001) = -0.1)-- D. $-\sqrt[3]{0.001}= -0.1$
  
  -- The proposition that the second equation is incorrect
  cases eq2 with eq2_pos eq2_neg
  { intro h,
    contradiction,
    -- The second option is incorrect for positive root
    exact sorry, -- Details skipped
  }
  { intro h,
    contradiction,
    -- The second option is incorrect for negative root
    exact sorry, -- Details skipped
  }

end incorrect_equation_in_list_l440_440348


namespace total_snowfall_yardley_l440_440107

theorem total_snowfall_yardley (a b c d : ℝ) (ha : a = 0.12) (hb : b = 0.24) (hc : c = 0.5) (hd : d = 0.36) :
  a + b + c + d = 1.22 :=
by
  sorry

end total_snowfall_yardley_l440_440107


namespace right_triangle_medians_l440_440130

theorem right_triangle_medians
    (a b c d m : ℝ)
    (h1 : ∀(a b c d : ℝ), 2 * (c/d) = 3)
    (h2 : m = 4 * 3 ∨ m = (3/4)) :
    ∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧ (m₁ = 12 ∨ m₁ = 3/4) ∧ (m₂ = 12 ∨ m₂ = 3/4) :=
by 
  sorry

end right_triangle_medians_l440_440130


namespace range_of_expression_l440_440459

theorem range_of_expression:
  (∀ x y : ℝ, x^2 + y^2 - 6x = 0 → sqrt(2 * x^2 + y^2 - 4 * x + 5) ∈ set.Icc (sqrt 5) (sqrt 53)) :=
begin
  intros x y h,
  sorry
end

end range_of_expression_l440_440459


namespace cos_six_arccos_two_fifths_l440_440362

noncomputable def arccos (x : ℝ) : ℝ := Real.arccos x
noncomputable def cos (x : ℝ) : ℝ := Real.cos x

theorem cos_six_arccos_two_fifths : cos (6 * arccos (2 / 5)) = 12223 / 15625 := 
by
  sorry

end cos_six_arccos_two_fifths_l440_440362


namespace smallest_sum_of_factors_l440_440589

theorem smallest_sum_of_factors :
  ∃ (p q r s : ℕ), p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ p * q * r * s = nat.factorial 8 ∧ p + q + r + s = 138 :=
by
  -- Definitions are incorporated within the theorem statement.
  sorry

end smallest_sum_of_factors_l440_440589


namespace probability_exactly_four_ones_is_090_l440_440239
open Float (approxEq)

def dice_probability_exactly_four_ones : Float :=
  let n := 12
  let k := 4
  let p_one := (1 / 6 : Float)
  let p_not_one := (5 / 6 : Float)
  let combination := ((n.factorial) / (k.factorial * (n - k).factorial) : Float)
  let probability := combination * (p_one ^ k) * (p_not_one ^ (n - k))
  probability

theorem probability_exactly_four_ones_is_090 : dice_probability_exactly_four_ones ≈ 0.090 :=
  sorry

end probability_exactly_four_ones_is_090_l440_440239


namespace function_value_ordering_l440_440820

variable {α : Type*} [LinearOrder α] {f : α → α}

-- Definitions based on conditions:
def even_function (f : α → α) : Prop := ∀ x, f (-x) = f x
def monotone_decreasing_on_interval (f : α → α) (a b : α) : Prop :=
  ∀ x y ∈ set.Icc a b, x ≤ y → f (x) ≥ f (y)

-- Proof goal:
theorem function_value_ordering
  (h_even : even_function f)
  (h_monotone : monotone_decreasing_on_interval (λ x, f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 := by
sorry

end function_value_ordering_l440_440820


namespace tournament_triplet_exists_l440_440726

noncomputable theory

def tournament (players : Type) (matches : players → players → Prop) :=
  ∀ (P : players), ∃ (Q : players), matches P Q

theorem tournament_triplet_exists {players : Type} (matches : players → players → Prop) 
  (H : tournament players matches) :
  ∃ (A B C : players), matches A B ∧ matches B C ∧ matches C A :=
sorry

end tournament_triplet_exists_l440_440726


namespace all_elements_rational_l440_440983

variable (S : Set ℝ)
variable [Fintype S]
variable (h : ∀ x ∈ S, ∃ a b ∈ (S ∪ {0, 1} : Set ℝ), a ≠ x ∧ b ≠ x ∧ x = (a + b) / 2)

theorem all_elements_rational (h : ∀ x ∈ S, ∃ a b ∈ (S ∪ {0, 1} : Set ℝ), a ≠ x ∧ b ≠ x ∧ x = (a + b) / 2) :
  ∀ x ∈ S, ∃ q : ℚ, (x : ℝ) = q :=
by
  sorry

end all_elements_rational_l440_440983


namespace range_of_m_l440_440053

variable (m : ℝ)
def p : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h : p m ∧ q m) : -2 < m ∧ m < 0 := sorry

end range_of_m_l440_440053


namespace sale_in_first_month_l440_440674

theorem sale_in_first_month
  (s2 : ℕ)
  (s3 : ℕ)
  (s4 : ℕ)
  (s5 : ℕ)
  (s6 : ℕ)
  (required_total_sales : ℕ)
  (average_sales : ℕ)
  : (required_total_sales = 39000) → 
    (average_sales = 6500) → 
    (s2 = 6927) →
    (s3 = 6855) →
    (s4 = 7230) →
    (s5 = 6562) →
    (s6 = 4991) →
    s2 + s3 + s4 + s5 + s6 = 32565 →
    required_total_sales - (s2 + s3 + s4 + s5 + s6) = 6435 :=
by
  intros
  sorry

end sale_in_first_month_l440_440674


namespace distance_from_pole_eq_l440_440079

noncomputable def distance_from_pole_to_line (ρ θ : ℝ) (A B C x0 y0 : ℝ) : ℝ :=
  | A * x0 + B * y0 + C | / Real.sqrt (A^2 + B^2)

theorem distance_from_pole_eq (ρ θ : ℝ)
  (h : ρ * Real.cos (θ + Real.pi / 3) = Real.sqrt 3 / 2) :
  distance_from_pole_to_line ρ θ 1 (-Real.sqrt 3) (-Real.sqrt 3) 0 0 = Real.sqrt 3 / 2 :=
by
  sorry

end distance_from_pole_eq_l440_440079


namespace part_a_no_solution_part_b_existence_of_29_l440_440880

-- Part (a)
theorem part_a_no_solution (C : ℕ) (h1 : C % 6 = 1) (h2 : C % 8 = 2) : false :=
by sorry

-- Part (b)
theorem part_b_existence_of_29 : ∃ (C : ℕ), (C % 7 = 1) ∧ (C % 9 = 2) :=
by {
    use 29,
    split,
    { -- proof that 29 % 7 = 1
      sorry },
    { -- proof that 29 % 9 = 2
      sorry }
}

end part_a_no_solution_part_b_existence_of_29_l440_440880


namespace pow_ge_double_plus_one_l440_440173

theorem pow_ge_double_plus_one (n : ℕ) (h : n ≥ 3) : 2^n ≥ 2 * (n + 1) :=
sorry

end pow_ge_double_plus_one_l440_440173


namespace parallel_vectors_implies_x_value_l440_440836

variable (x : ℝ)

def vec_a : ℝ × ℝ := (1, 2)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 1)
def scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem parallel_vectors_implies_x_value :
  (∃ k : ℝ, vec_add vec_a (scalar_mul 2 (vec_b x)) = scalar_mul k (vec_sub (scalar_mul 2 vec_a) (scalar_mul 2 (vec_b x)))) →
  x = 1 / 2 :=
by
  sorry

end parallel_vectors_implies_x_value_l440_440836


namespace g_15_45_l440_440590

def g (x y : ℕ) : ℕ

axiom g_ax1 (x : ℕ) : g x x = x^2
axiom g_ax2 (x y : ℕ) : g x y = g y x
axiom g_ax3 (x y : ℕ) : (x + y) * g x y = y * g x (x + y)

theorem g_15_45 : g 15 45 = 1350 := by
  sorry

end g_15_45_l440_440590


namespace q1_q2_q3_l440_440562

-- Define the problem for the first question
theorem q1 : (1 : ℚ) / (99 * 100) = (1/99 : ℚ) - (1/100 : ℚ) :=
by
  sorry

-- Define the problem for the second question (general pattern)
theorem q2 (n : ℕ) : (1 : ℚ) / (n * (n + 1)) = (1/n : ℚ) - (1/(n + 1) : ℚ) :=
by
  sorry

-- Define the problem for the third question (sum calculation)
theorem q3 : ∑ k in finset.range 1011, (1 : ℚ) / ((2 * k + 2) * (2 * k + 4)) = 1011 / 4048 :=
by
  sorry

end q1_q2_q3_l440_440562


namespace finite_solutions_to_equation_l440_440182

theorem finite_solutions_to_equation :
  ∃ n : ℕ, ∀ (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ (1 / (a:ℝ) + 1 / (b:ℝ) + 1 / (c:ℝ) = 1 / 1983) →
    a ≤ n ∧ b ≤ n ∧ c ≤ n :=
sorry

end finite_solutions_to_equation_l440_440182


namespace find_m_range_l440_440059

-- Definitions for the conditions and the required proof
def condition_alpha (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m + 7
def condition_beta (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

-- Proof problem translated to Lean 4 statement
theorem find_m_range (m : ℝ) :
  (∀ x, condition_beta x → condition_alpha m x) → (-2 ≤ m ∧ m ≤ 0) :=
by sorry

end find_m_range_l440_440059


namespace exponential_plus_linear_increasing_l440_440638

theorem exponential_plus_linear_increasing (x : ℝ) : 
  (λ x, Real.exp x + x) (x + 1) > (λ x, Real.exp x + x) x + 1 :=
by
  sorry

end exponential_plus_linear_increasing_l440_440638


namespace determine_theta_l440_440465

theorem determine_theta (f : ℝ → ℝ) (θ : ℝ) (hθ : θ ∈ Icc (- (π / 2)) (π / 2)) 
  (hfeven : ∀ x : ℝ, f(-x) = f(x)) : θ = π / 6 :=
begin
  sorry
end

end determine_theta_l440_440465


namespace total_gray_area_trees_l440_440610

/-- 
Three aerial photos were taken by the drone, each capturing the same number of trees.
First rectangle has 100 trees in total and 82 trees in the white area.
Second rectangle has 90 trees in total and 82 trees in the white area.
Prove that the number of trees in gray areas in both rectangles is 26.
-/
theorem total_gray_area_trees : (100 - 82) + (90 - 82) = 26 := 
by sorry

end total_gray_area_trees_l440_440610


namespace repeating_decimal_to_fraction_l440_440430

theorem repeating_decimal_to_fraction :
  let r := (1 : ℝ) / 100 in
  let a_1 := 72 / 100 in 
  0.72 == a_1 / (1 - r) := by
    sorry

end repeating_decimal_to_fraction_l440_440430


namespace log3_sufficient_not_necessary_l440_440592

theorem log3_sufficient_not_necessary (x : ℝ) : (log x / log 3 > 1) → (2^x > 1) ∧ ¬((2^x > 1) → (log x / log 3 > 1)) :=
by {
  sorry,
}

end log3_sufficient_not_necessary_l440_440592


namespace jade_and_julia_total_money_l440_440523

theorem jade_and_julia_total_money (x : ℕ) : 
  let jade_initial := 38 
  let julia_initial := jade_initial / 2 
  let jade_after := jade_initial + x 
  let julia_after := julia_initial + x 
  jade_after + julia_after = 57 + 2 * x := by
  sorry

end jade_and_julia_total_money_l440_440523


namespace prove_county_growth_condition_l440_440510

variable (x : ℝ)
variable (investment2014 : ℝ) (investment2016 : ℝ)

def county_growth_condition
  (h1 : investment2014 = 2500)
  (h2 : investment2016 = 3500) : Prop :=
  investment2014 * (1 + x)^2 = investment2016

theorem prove_county_growth_condition
  (x : ℝ)
  (h1 : investment2014 = 2500)
  (h2 : investment2016 = 3500) : county_growth_condition x investment2014 investment2016 h1 h2 :=
by
  sorry

end prove_county_growth_condition_l440_440510


namespace jame_initial_gold_bars_l440_440896

theorem jame_initial_gold_bars (X : ℝ) (h1 : X * 0.1 + 0.5 * (X * 0.9) = 0.5 * (X * 0.9) - 27) :
  X = 60 :=
by
-- Placeholder for proof
sorry

end jame_initial_gold_bars_l440_440896


namespace greatest_mean_AC_l440_440648

variables {A B C : Type} 
variables (A_weights : A → ℝ) (B_weights : B → ℝ) (C_weights : C → ℝ)
variables (nA nB nC : ℝ)
variables (mean_A mean_B mean_AB mean_BC : ℝ)

def conditions : Prop :=
  mean_A = 35 ∧
  mean_B = 45 ∧
  mean_AB = 38 ∧
  mean_BC = 42

noncomputable def mean_AC : ℝ :=
  (nA * 35 + nC * 21 + nC * 42) / (nA + nC)

theorem greatest_mean_AC (h : conditions) :
  ∃ k, mean_AC 35 45 38 42 = 49 :=
sorry

end greatest_mean_AC_l440_440648


namespace garage_has_18_wheels_l440_440672

namespace Garage

def bike_wheels_per_bike : ℕ := 2
def bikes_assembled : ℕ := 9

theorem garage_has_18_wheels
  (b : ℕ := bikes_assembled) 
  (w : ℕ := bike_wheels_per_bike) :
  b * w = 18 :=
by
  sorry

end Garage

end garage_has_18_wheels_l440_440672


namespace triangle_geometric_prog_l440_440191

theorem triangle_geometric_prog (x p q r : ℝ) (θ : ℝ) 
(h1:  θ > 0) 
(h2:  6 > 0) 
(h3:  7 > 0) 
(h4:  x > 0) 
(h5: θ + θ^2 + 2 * θ = 180) 
(h6: x^2 = 6^2 + 7^2 - 2 * 6 * 7 * (- Math.cos 36))
(h7: p = 11) 
(h8: q = 140) 
(h9: r = 0) :
p + q + r = 151 :=
sorry

end triangle_geometric_prog_l440_440191


namespace sum_even_digits_1_to_200_l440_440364

def even_digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.filter (λ d, d % 2 = 0) |>.sum

theorem sum_even_digits_1_to_200 : (List.sum (List.map even_digits_sum (List.range' 1 200))) = 1600 :=
by
  sorry

end sum_even_digits_1_to_200_l440_440364


namespace range_of_k_l440_440554

variables {k : ℝ} (p : k > 0) (q : (2*k - 3)^2 - 4 > 0)

theorem range_of_k (h1 : ¬(p ∧ q)) (h2 : p ∨ q) :
  k ∈ Icc 0 ∞ ∨ k ∈ Icc (1/2) (5/2) :=
begin
  sorry
end

end range_of_k_l440_440554


namespace monotonically_increasing_interval_of_fx_l440_440071

theorem monotonically_increasing_interval_of_fx :
  (∀ (a: ℝ), ∃ A B: ℝ, (A ≠ B) ∧ (tan A = a) ∧ (tan B = a) ∧ (|A - B| = π)) →
    ∀ k: ℤ, f(x) = sqrt(3) * sin x - cos x →
      -π / 3 + 2 * k * π ≤ x ∧ x ≤ 2 * π / 3 + 2 * k * π :=
by
  sorry

end monotonically_increasing_interval_of_fx_l440_440071


namespace general_formula_a_n_general_formula_b_n_sum_c_n_T_n_l440_440558

open Classical

axiom S_n : ℕ → ℝ
axiom a_n : ℕ → ℝ
axiom b_n : ℕ → ℝ
axiom c_n : ℕ → ℝ
axiom T_n : ℕ → ℝ

noncomputable def general_a_n (n : ℕ) : ℝ :=
  sorry

axiom h1 : ∀ n, S_n n + a_n n = 2

theorem general_formula_a_n : ∀ n, a_n n = 1 / 2^(n-1) :=
  sorry

axiom h2 : b_n 1 = a_n 1
axiom h3 : ∀ n ≥ 2, b_n n = 3 * b_n (n-1) / (b_n (n-1) + 3)

theorem general_formula_b_n : ∀ n, b_n n = 3 / (n + 2) ∧
  (∀ n, 1 / b_n n = 1 + (n - 1) / 3) :=
  sorry

axiom h4 : ∀ n, c_n n = a_n n / b_n n

theorem sum_c_n_T_n : ∀ n, T_n n = 8 / 3 - (n + 4) / (3 * 2^(n-1)) :=
  sorry

end general_formula_a_n_general_formula_b_n_sum_c_n_T_n_l440_440558


namespace fractions_count_between_half_and_third_l440_440478

theorem fractions_count_between_half_and_third :
  (∃ S : Finset (ℚ), 
    (∀ x ∈ S, (1 : ℚ) / 6 ≤ x ∧ x ≤ 1 / 3 ∧ x.denom = 15) ∧
    S.card = 3) := by 
  sorry

end fractions_count_between_half_and_third_l440_440478


namespace dice_sum_24_l440_440759

noncomputable def probability_of_sum_24 : ℚ :=
  let die_probability := (1 : ℚ) / 6
  in die_probability ^ 4

theorem dice_sum_24 :
  ∑ x in {x | x ∈ {1, 2, 3, 4, 5, 6} ∧ x = 6} = 24 → probability_of_sum_24 = 1 / 1296 :=
sorry

end dice_sum_24_l440_440759


namespace sqrt_expression_result_l440_440731

theorem sqrt_expression_result :
  (Real.sqrt (16 - 8 * Real.sqrt 3) - Real.sqrt (16 + 8 * Real.sqrt 3)) ^ 2 = 48 := 
sorry

end sqrt_expression_result_l440_440731


namespace power_logarithm_identity_l440_440295

theorem power_logarithm_identity : 9^(2 - log 3 2) = 81 / 4 := by
  sorry

end power_logarithm_identity_l440_440295


namespace same_color_combination_probability_l440_440317

-- Defining the number of each color candy 
def num_red : Nat := 12
def num_blue : Nat := 12
def num_green : Nat := 6

-- Terry and Mary each pick 3 candies at random
def total_pick : Nat := 3

-- The total number of candies in the jar
def total_candies : Nat := num_red + num_blue + num_green

-- Probability of Terry and Mary picking the same color combination
def probability_same_combination : ℚ := 2783 / 847525

-- The theorem statement
theorem same_color_combination_probability :
  let terry_picks_red := (num_red * (num_red - 1) * (num_red - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_red := num_red - total_pick
  let mary_picks_red := (remaining_red * (remaining_red - 1) * (remaining_red - 2)) / (27 * 26 * 25)
  let combined_red := terry_picks_red * mary_picks_red

  let terry_picks_blue := (num_blue * (num_blue - 1) * (num_blue - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_blue := num_blue - total_pick
  let mary_picks_blue := (remaining_blue * (remaining_blue - 1) * (remaining_blue - 2)) / (27 * 26 * 25)
  let combined_blue := terry_picks_blue * mary_picks_blue

  let terry_picks_green := (num_green * (num_green - 1) * (num_green - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_green := num_green - total_pick
  let mary_picks_green := (remaining_green * (remaining_green - 1) * (remaining_green - 2)) / (27 * 26 * 25)
  let combined_green := terry_picks_green * mary_picks_green

  let total_probability := 2 * combined_red + 2 * combined_blue + combined_green
  total_probability = probability_same_combination := sorry

end same_color_combination_probability_l440_440317


namespace quadratic_eqn_coeffs_l440_440968

theorem quadratic_eqn_coeffs :
  ∀ (x : ℝ), (4 * x^2 - 6 * x + 1 = 0) → (4 = 4) ∧ (-6 = -6) ∧ (1 = 1) :=
by intros x h; exact ⟨rfl, rfl, rfl⟩ sorry

end quadratic_eqn_coeffs_l440_440968


namespace Trent_tears_l440_440616

def onions_per_pot := 4
def pots_of_soup := 6
def tears_per_3_onions := 2

theorem Trent_tears:
  (onions_per_pot * pots_of_soup) / 3 * tears_per_3_onions = 16 :=
by
  sorry

end Trent_tears_l440_440616


namespace shop_sold_price_l440_440310

noncomputable def clock_selling_price (C : ℝ) : ℝ :=
  let buy_back_price := 0.60 * C
  let maintenance_cost := 0.10 * buy_back_price
  let total_spent := buy_back_price + maintenance_cost
  let selling_price := 1.80 * total_spent
  selling_price

theorem shop_sold_price (C : ℝ) (h1 : C - 0.60 * C = 100) :
  clock_selling_price C = 297 := by
  sorry

end shop_sold_price_l440_440310


namespace lifting_equivalency_l440_440567

noncomputable def total_weight_lifted_initial (reps : ℕ) (weight : ℕ) : ℕ :=
  2 * weight * reps

noncomputable def total_weight_lifted_new (reps : ℕ) (weight : ℕ) : ℕ :=
  2 * weight * reps

theorem lifting_equivalency : 
  let initial_reps := 10 in
  let initial_weight := 25 in
  let new_weight := 20 in
  40 * (500 / 40) = total_weight_lifted_initial initial_reps initial_weight := sorry

end lifting_equivalency_l440_440567


namespace polynomial_degree_l440_440384

theorem polynomial_degree : 
  degree (-2 * X^5 + 3 * X^3 + 15 - 4 * real.pi * X^2 + (real.sqrt 3) * X^5 + 25 * X) = 5 :=
by
  sorry

end polynomial_degree_l440_440384


namespace cindy_finishes_first_l440_440701

variable (a r : ℝ)

-- Conditions
def area_ben := a / 3
def area_cindy := a / 4
def rate_allison := r
def rate_ben := 3 * r / 4
def rate_cindy := r / 2

-- Mowing times
def time_allison := a / r
def time_ben := area_ben / rate_ben
def time_cindy := area_cindy / rate_cindy

-- Proof statement
theorem cindy_finishes_first (h : a > 0) (hr : r > 0) : time_cindy < time_ben ∧ time_cindy < time_allison := by
  sorry

end cindy_finishes_first_l440_440701


namespace fifty_day_of_year_N_minus_one_is_monday_l440_440894

/-- Given that in year N, the 250th day is a Friday and in year N+1, the 150th day is a Friday,
    prove that the 50th day of year N-1 is a Monday. --/
theorem fifty_day_of_year_N_minus_one_is_monday 
  (N : ℕ)
  (h250 : (250 % 7 = 5))  -- 250th day is a Friday
  (h150 : ((150 % 7) - 1 = 5)%7)  -- 150th day in year N+1 is a Friday
  : (50 % 7 = 1) :=   -- 50th day in year N-1 is a Monday
sorry

end fifty_day_of_year_N_minus_one_is_monday_l440_440894


namespace inequality_proof_l440_440810

theorem inequality_proof 
  (a b c : ℝ) 
  (h_a : 0 ≤ a) 
  (h_b : 0 ≤ b) 
  (h_c : 0 ≤ c) 
  (h_sum : a + b + c = 1) : 
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧ 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1 + a)*(1 + b)*(1 + c) :=
begin
  sorry
end

end inequality_proof_l440_440810


namespace log_exp_example_l440_440439

theorem log_exp_example
  (a m n : ℝ)
  (h₁ : log a 3 = m)
  (h₂ : log a 5 = n) :
  a^(2 * m + n) = 45 := 
sorry

end log_exp_example_l440_440439


namespace polar_line_equation_l440_440012

theorem polar_line_equation (P : ℝ × ℝ) (hP : P = (2, π / 3)) (parallel : ∀ Q : ℝ × ℝ, Q.fst = ρ₂.sin Q.snd)
    : ∃ ρ θ, parallel (ρ, θ) ∧ ρ * sin θ = sqrt 3 := by
  cases hP
  use 2
  use π / 3
  split
  assumption
  sorry

end polar_line_equation_l440_440012


namespace not_correponding_analogy_l440_440114

-- Definitions based on conditions provided in the problem
def isAppropriateAnalogy (a b : Type) : Prop := sorry

def triangle : Type := sorry
def plane : Type := sorry
def parallelepiped : Type := sorry
def space : Type := sorry
def triangular_pyramid : Type := sorry

-- Condition: Comparing a triangle in a plane to a triangular pyramid in space is appropriate
axiom condition : isAppropriateAnalogy (triangle × plane) (triangular_pyramid × space)

-- Proof problem: Comparing a triangle in a plane to a parallelepiped in space is not appropriate
theorem not_correponding_analogy : ¬ isAppropriateAnalogy (triangle × plane) (parallelepiped × space) :=
sorry

end not_correponding_analogy_l440_440114


namespace question1_question2_l440_440368

theorem question1 :
  (1:ℝ) * (Real.sqrt 12 + Real.sqrt 20) + (Real.sqrt 3 - Real.sqrt 5) = 3 * Real.sqrt 3 + Real.sqrt 5 := 
by sorry

theorem question2 :
  (4 * Real.sqrt 2 - 3 * Real.sqrt 6) / (2 * Real.sqrt 2) - (Real.sqrt 8 + Real.pi)^0 = 1 - 3 * Real.sqrt 3 / 2 :=
by sorry

end question1_question2_l440_440368


namespace intersection_A_B_l440_440140

open Set

def A : Set ℝ := Icc 1 2

def B : Set ℤ := {x : ℤ | x^2 - 2 * x - 3 < 0}

theorem intersection_A_B :
  (A ∩ (coe '' B) : Set ℝ) = {1, 2} :=
sorry

end intersection_A_B_l440_440140


namespace fifth_iteration_perimeter_l440_440358

theorem fifth_iteration_perimeter :
  let A1_side_length := 1
  let P1 := 3 * A1_side_length
  let P2 := 3 * (A1_side_length * 4 / 3)
  ∀ n : ℕ, P_n = 3 * (4 / 3) ^ (n - 1) →
  P_5 = 3 * (4 / 3) ^ 4 :=
  by sorry

end fifth_iteration_perimeter_l440_440358


namespace number_of_3in_pipes_l440_440340

theorem number_of_3in_pipes (h : ℝ) : 
  let V_12 := π * (6^2) * h 
  let V_3 := π * (1.5^2) * h 
  (V_12 = 16 * V_3) :=
by
  let r_12 := 6
  let r_3 := 1.5
  let V_12 := π * (r_12^2) * h 
  let V_3 := π * (r_3^2) * h 
  have h_eq : 36 * π * h = 16 * (2.25 * π * h)
  {
    calc
    36 * π * h = 16 * (2.25 * π * h) : by
      rw [← mul_assoc, mul_comm 36, ← mul_assoc, one_div_inv_one, ← mul_assoc]
  }
  exact h_eq

end number_of_3in_pipes_l440_440340


namespace train_speed_l440_440812

theorem train_speed (length_bridge : ℕ) (time_total : ℕ) (time_on_bridge : ℕ) (speed_of_train : ℕ) 
  (h1 : length_bridge = 800)
  (h2 : time_total = 60)
  (h3 : time_on_bridge = 40)
  (h4 : length_bridge + (time_total - time_on_bridge) * speed_of_train = time_total * speed_of_train) :
  speed_of_train = 20 := sorry

end train_speed_l440_440812


namespace pages_to_read_l440_440931

variable (E P_Science P_Civics P_Chinese Total : ℕ)
variable (h_Science : P_Science = 16)
variable (h_Civics : P_Civics = 8)
variable (h_Chinese : P_Chinese = 12)
variable (h_Total : Total = 14)

theorem pages_to_read :
  (E / 4) + (P_Science / 4) + (P_Civics / 4) + (P_Chinese / 4) = Total → 
  E = 20 := by
  sorry

end pages_to_read_l440_440931


namespace length_FG_l440_440108

-- Define the given elements in the problem
constant O : Type
constant E F G : O
constant circle : O → Prop

-- Given conditions
axiom center_O : circle O
axiom chord_EF : E ≠ F
axiom chord_EFG : G ≠ F
axiom FO_eq_7 : dist F O = 7
axiom angle_EFO_eq_90 : ∠ E F O = 90
axiom arc_HG_eq_90 : arc HG = 90

-- Define the length FG
constant dist : O → O → ℝ

-- The length of FG is 7
theorem length_FG : dist F G = 7 :=
sorry

end length_FG_l440_440108


namespace part_a_l440_440294

theorem part_a (a b c : ℝ) (m : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (a + b)^m + (b + c)^m + (c + a)^m ≤ 2^m * (a^m + b^m + c^m) :=
by
  sorry

end part_a_l440_440294


namespace solution_set_of_inequality_l440_440985

theorem solution_set_of_inequality : 
    {x : ℝ | x + 2 / (x + 1) > 2} = set.union (set.Ioo (-1) 0) (set.Ioi 2) :=
sorry

end solution_set_of_inequality_l440_440985


namespace find_fourth_number_in_proportion_l440_440099

theorem find_fourth_number_in_proportion : 
  ∃ y : ℝ, (0.6 / 0.96 = 5 / y) ∧ y = 8 :=
by
  have h : 0.6 / 0.96 = 5 / 8, {
    calc 0.6 / 0.96 = 5 / 8 : sorry
  }
  use 8
  exact ⟨h, rfl⟩

end find_fourth_number_in_proportion_l440_440099


namespace min_solution_l440_440019

theorem min_solution :
  ∀ (x : ℝ), (min (1 / (1 - x)) (2 / (1 - x)) = 2 / (x - 1) - 3) → x = 7 / 3 := 
by
  sorry

end min_solution_l440_440019


namespace smallest_possible_integer_t_l440_440111

theorem smallest_possible_integer_t (t : ℤ) (h1 : 7.5 + t > 12) (h2 : 7.5 + 12 > t) (h3 : 12 + t > 7.5) : t = 5 := 
by {
  -- Placeholder for the proof
  sorry
}

end smallest_possible_integer_t_l440_440111


namespace monotone_intervals_max_floor_a_l440_440464

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + a

theorem monotone_intervals (a : ℝ) (h : a = 1) :
  (∀ x, 0 < x ∧ x < 1 → deriv (λ x => f x 1) x > 0) ∧
  (∀ x, 1 ≤ x → deriv (λ x => f x 1) x < 0) :=
by
  sorry

theorem max_floor_a (a : ℝ) (h : ∀ x > 0, f x a ≤ x) : ⌊a⌋ = 1 :=
by
  sorry

end monotone_intervals_max_floor_a_l440_440464


namespace solve_for_x_l440_440857

theorem solve_for_x (x : ℝ) (h : sqrt (3 / x + 3) = 5 / 3) : x = -27 / 2 :=
by
  sorry

end solve_for_x_l440_440857


namespace exists_divisible_pair_l440_440393

def is_valid_pair (three_digit : ℕ) (two_digit : ℕ) : Prop :=
  (three_digit / 100 ∈ {1, 2, 3, 4, 5}) ∧
  ((three_digit % 100) / 10 ∈ {1, 2, 3, 4, 5}) ∧
  (three_digit % 10 ∈ {1, 2, 3, 4, 5}) ∧
  (two_digit / 10 ∈ {1, 2, 3, 4, 5}) ∧
  (two_digit % 10 ∈ {1, 2, 3, 4, 5}) ∧
  (three_digit / 100 ≠ (three_digit % 100) / 10) ∧
  (three_digit / 100 ≠ three_digit % 10) ∧
  ((three_digit % 100) / 10 ≠ three_digit % 10) ∧
  (two_digit / 10 ≠ two_digit % 10) ∧
  (three_digit % 10 ≠ two_digit / 10) ∧
  (three_digit % 10 ≠ two_digit % 10) ∧
  (three_digit / 100 ≠ two_digit / 10) ∧
  (three_digit / 100 ≠ two_digit % 10) ∧
  ((three_digit % 100) / 10 ≠ two_digit / 10) ∧
  ((three_digit % 100) / 10 ≠ two_digit % 10)

theorem exists_divisible_pair :
  ∃ (three_digit two_digit : ℕ), 
    (three_digit ∈ {123, 124, 125, 132, 134, 135, 142, 143, 145, 153, 154, 234, 235, 245, 312, 314, 315, 324, 325, 345, 412, 413, 415, 423, 425, 435, 512, 513, 514, 523, 524, 534}) ∧
    (two_digit ∈ {12, 13, 14, 15, 21, 23, 24, 25, 31, 32, 34, 35, 41, 42, 43, 45, 51, 52, 53, 54}) ∧
    is_valid_pair three_digit two_digit ∧
    three_digit % two_digit = 0 := sorry

end exists_divisible_pair_l440_440393


namespace dice_sum_probability_l440_440771

theorem dice_sum_probability :
  let D := finset.range 1 7  -- outcomes of a fair six-sided die
  (∃! d1 d2 d3 d4 ∈ D, d1 + d2 + d3 + d4 = 24) ->
  (probability(space, {ω ∈ space | (ω 1 = 6) ∧ (ω 2 = 6) ∧ (ω 3 = 6) ∧ (ω 4 = 6)}) = 1/1296) :=
sorry

end dice_sum_probability_l440_440771


namespace angle_DBF_half_angle_FCD_l440_440388

-- Definitions of points, angles, and geometric properties
variables {A B C D E F : Type}
variable [is_isosceles_trapezoid A B C D AD BC AC BD]
variable [perpendicular A C B D]
variable [perpendicular_from D E A B]
variable [perpendicular_from C F D E]

-- Theorem statement
theorem angle_DBF_half_angle_FCD :
  ∀ (A B C D E F : Point) (AD BC AC BD DE CF : Line), 
    is_isosceles_trapezoid A B C D AD BC AC BD → 
    perpendicular AC BD → 
    perpendicular_from D E A B → 
    perpendicular_from C F D E →
    ∠ DBF = (1 / 2) * ∠ FCD :=
begin
  sorry
end

end angle_DBF_half_angle_FCD_l440_440388


namespace ratio_of_areas_of_concentric_circles_eq_9_over_4_l440_440999

theorem ratio_of_areas_of_concentric_circles_eq_9_over_4
  (C1 C2 : ℝ)
  (h1 : ∃ Q : ℝ, true) -- Existence of point Q
  (h2 : (30 / 360) * C1 = (45 / 360) * C2) -- Arcs formed by 30-degree and 45-degree angles are equal in length
  : (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 9 / 4 :=
by
  sorry

end ratio_of_areas_of_concentric_circles_eq_9_over_4_l440_440999


namespace cone_base_diameter_l440_440605

theorem cone_base_diameter 
  (r l : ℝ) 
  (h1 : 3 * Real.pi = Real.pi * r * (r + l)) 
  (h2 : Real.pi * l = 2 * Real.pi * r) : 
  2 * r = 2 :=
by
  -- Introduce necessary definitions and assertions.
  let r_sq := r * r
  -- Given h2 and h1, prove that 2 * r = 2
  have l_eq_2r : l = 2 * r := 
    by sorry -- use h2 here
  have h3 : 3 * r_sq = 3 :=
    by sorry -- from h1, substitute l = 2 * r and simplify
  have r_eq_1 : r = 1 :=
    by sorry -- from h3
  show 2 * r = 2 :=
    by sorry -- from r_eq_1

end cone_base_diameter_l440_440605


namespace count_special_integers_l440_440847

theorem count_special_integers : 
  let is_special (n : ℕ) := 
    100 ≤ n ∧ n ≤ 999 ∧ 
    (n % 10) % 2 = 1 ∧ 
    ((n / 10) % 10) % 2 = 0 ∧ 
    ((n / 100) % 10) % 2 = 0
  in (finset.filter is_special (finset.range 900)).card = 100 := by
  sorry

end count_special_integers_l440_440847


namespace probability_of_four_ones_l440_440247

noncomputable def probability_exactly_four_ones : ℚ :=
  (Nat.choose 12 4 * (1/6)^4 * (5/6)^8)

theorem probability_of_four_ones :
  abs (probability_exactly_four_ones.toReal - 0.114) < 0.001 :=
by
  sorry

end probability_of_four_ones_l440_440247


namespace number_of_friends_with_pears_l440_440943

-- Each friend either carries pears or oranges
def total_friends : Nat := 15
def friends_with_oranges : Nat := 6
def friends_with_pears : Nat := total_friends - friends_with_oranges

theorem number_of_friends_with_pears :
  friends_with_pears = 9 := by
  -- Proof steps would go here
  sorry

end number_of_friends_with_pears_l440_440943


namespace cookies_total_is_60_l440_440938

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Rachel_cookies : ℕ := Jasmine_cookies + 10
def Total_cookies : ℕ := Mona_cookies + Jasmine_cookies + Rachel_cookies

theorem cookies_total_is_60 : Total_cookies = 60 := by
  sorry

end cookies_total_is_60_l440_440938


namespace quadratic_roots_prime_distinct_l440_440105

theorem quadratic_roots_prime_distinct (a α β m : ℕ) (h1: α ≠ β) (h2: Nat.Prime α) (h3: Nat.Prime β) (h4: α + β = m / a) (h5: α * β = 1996 / a) :
    a = 2 := by
  sorry

end quadratic_roots_prime_distinct_l440_440105


namespace cookies_total_is_60_l440_440939

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Rachel_cookies : ℕ := Jasmine_cookies + 10
def Total_cookies : ℕ := Mona_cookies + Jasmine_cookies + Rachel_cookies

theorem cookies_total_is_60 : Total_cookies = 60 := by
  sorry

end cookies_total_is_60_l440_440939


namespace abs_is_even_and_increasing_l440_440351

def is_even (f: ℝ → ℝ) := ∀ x: ℝ, f x = f (-x)
def is_increasing_on (f: ℝ → ℝ) (I: set ℝ) := ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem abs_is_even_and_increasing :
  is_even (λ x: ℝ, |x|) ∧ is_increasing_on (λ x: ℝ, |x|) (set.Ioi 0) :=
sorry

end abs_is_even_and_increasing_l440_440351


namespace intersection_A_B_l440_440158

def A : Set ℝ := { x | abs x ≤ 1 }
def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_A_B : (A ∩ B) = { x | 0 ≤ x ∧ x ≤ 1 } :=
sorry

end intersection_A_B_l440_440158


namespace percentage_spent_on_other_items_l440_440166

variable (T : ℝ) (x : ℝ)
variables (hc : (0.04 * 0.50 * T = 0.02 * T)) (hf : (0.00 * 0.20 * T = 0.0 * T)) (ho : (0.08 * x * T))
variables (total_tax : 0.02 * T + 0.08 * x * T = 0.044 * T)

theorem percentage_spent_on_other_items
  (h1 : 0.50 * T)
  (h2 : 0.20 * T)
  (h4 : 4 / 100)
  (h6 : 8 / 100)
  : x = 0.3 :=
by
  sorry

end percentage_spent_on_other_items_l440_440166


namespace even_four_digit_increasing_order_count_l440_440090

theorem even_four_digit_increasing_order_count : 
  (∀ (a b c d : ℕ), (1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d % 2 = 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) →
    (∃ (set_data : set ℕ) (hl: list.nodup set_data.to_list) , 
      set_data = {a, b, c, d} ∧ 
      (∀ (x ∈ set_data), 1 ≤ x  ∧ x < 10 ) ∧ 
      d ∈ {2, 4, 6, 8}) ) → 
    (finset.filter
      (λ t : finset ℕ, t.card = 4 ∧ list.nodup t.val ∧ ∀ x ∈ t, (x ∈ {2, 4, 6, 8} ∧ x ≠ 0 ))
      (finset.powerset (finset.range 10))).card
      = 46 :=
begin
  sorry
end

end even_four_digit_increasing_order_count_l440_440090


namespace lateral_surface_area_of_triangular_pyramid_l440_440737

noncomputable def side_length (a : ℝ) : Prop :=
a = 24

noncomputable def triangular_pyramid_lateral_area (height slant_height : ℝ) : ℝ :=
let a := 24 in -- side length of the base
let face_area := (1 / 2) * a * slant_height in
3 * face_area

theorem lateral_surface_area_of_triangular_pyramid :
  let DM := 4 in
  let DK := 8 in
  triangular_pyramid_lateral_area DM DK = 288 :=
by
  let DM := 4
  let DK := 8
  have h1 : side_length 24 := by
    sorry
  have calc_area : triangular_pyramid_lateral_area DM DK = 3 * ((1 / 2) * 24 * 8) := by
    sorry
  show triangular_pyramid_lateral_area DM DK = 288 by
    rw calc_area
    norm_num

end lateral_surface_area_of_triangular_pyramid_l440_440737


namespace two_digit_numbers_condition_l440_440376

-- Define the problem
theorem two_digit_numbers_condition :
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧
        let a := n / 10 in
        let b := n % 10 in
        9 * a % 7 = 5}.to_finset.card = 10 :=
by
  sorry

end two_digit_numbers_condition_l440_440376


namespace find_k_l440_440042

theorem find_k (x₁ x₂ k : ℝ) (hx : x₁ + x₂ = 3) (h_prod : x₁ * x₂ = k) (h_cond : x₁ * x₂ + 2 * x₁ + 2 * x₂ = 1) : k = -5 :=
by
  sorry

end find_k_l440_440042


namespace product_of_integers_l440_440164

theorem product_of_integers (a b : ℚ) (h1 : a / b = 12) (h2 : a + b = 144) :
  a * b = 248832 / 169 := 
sorry

end product_of_integers_l440_440164


namespace sum_of_solutions_l440_440854

theorem sum_of_solutions (y : ℝ) (h : y^2 = 9) : 
  (∃ S : Set ℝ, S = { y | y^2 = 9 } ∧ (∑ y in S, y) = 0) :=
sorry

end sum_of_solutions_l440_440854


namespace translated_parabola_l440_440233

def f (x : ℝ) : ℝ := 2 * x^2

def translate_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x, f (x + a)

def translate_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ :=
  λ x, f x + b

theorem translated_parabola :
  translate_up (translate_left f 3) 4 = λ x, 2 * (x + 3)^2 + 4 :=
by
  -- proof would go here
  sorry

end translated_parabola_l440_440233


namespace percentage_decrease_is_25_l440_440200

-- Define the original number
def original_number := 80

-- Define the increased value
def increased_value := original_number + (0.125 * original_number)

-- Define the decreased value in terms of the percentage decrease
def decreased_value (percentage_decreased : ℝ) := original_number - (percentage_decreased / 100 * original_number)

-- Define the main theorem we want to prove
theorem percentage_decrease_is_25 (percentage_decreased : ℝ) :
  increased_value - decreased_value percentage_decreased = 30 → percentage_decreased = 25 :=
by
  -- Proof steps would go here
  sorry

end percentage_decrease_is_25_l440_440200


namespace max_S_R_squared_l440_440877

theorem max_S_R_squared (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) :
  (∃ a b c, DA = a ∧ DB = b ∧ DC = c ∧ S = 2 * (a * b + b * c + c * a) ∧
  R = (Real.sqrt (a^2 + b^2 + c^2)) / 2 ∧ (∃ max_val, max_val = (2 / 3) * (3 + Real.sqrt 3))) :=
sorry

end max_S_R_squared_l440_440877


namespace total_shaded_area_l440_440685

theorem total_shaded_area 
  (side_length_carpet : ℝ)
  (ratio_large_to_carpet : ℝ)
  (ratio_small_to_large : ℝ)
  (num_small_squares : ℕ)
  (large_square_area : ℝ)
  (small_square_area : ℝ)
  (total_shaded_area : ℝ) :
  side_length_carpet = 12 → ratio_large_to_carpet = 2 → ratio_small_to_large = 2 → num_small_squares = 12 → 
  large_square_area = (((side_length_carpet / ratio_large_to_carpet)^2)) → 
  small_square_area = ((side_length_carpet / ratio_large_to_carpet / ratio_small_to_large)^2) → 
  total_shaded_area = (num_small_squares * small_square_area + large_square_area) →
  total_shaded_area = 144 :=
by
  intros h_side_length_carpet h_ratio_large_to_carpet h_ratio_small_to_large h_num_small_squares h_large_square_area h_small_square_area h_total_shaded_area
  rw [h_side_length_carpet, h_ratio_large_to_carpet, h_ratio_small_to_large, h_num_small_squares] at *
  rw [h_large_square_area, h_small_square_area, h_total_shaded_area]
  -- Actual proof would go here
  sorry

end total_shaded_area_l440_440685


namespace slope_of_line_l440_440076

theorem slope_of_line (t : ℝ) :
  let x := 3 - t * real.sin (real.pi / 9)
  let y := 2 + t * real.cos (real.pi / 9 * 7)
  ∃ k : ℝ, k = (y - 2) / (x - 3) → k = -1 :=
sorry

end slope_of_line_l440_440076


namespace probability_divisor_of_12_l440_440665

open Probability

def divisors_of_12 := {1, 2, 3, 4, 6}

theorem probability_divisor_of_12 :
  ∃ (fair_die_roll : ProbabilityMeasure (Fin 6)), 
    P (fun x => x.val + 1 ∈ divisors_of_12) = 5 / 6 := 
by
  sorry

end probability_divisor_of_12_l440_440665


namespace triangle_area_parabola_hyperbola_l440_440455

theorem triangle_area_parabola_hyperbola :
  let p := 4
  let focus_F := (2, 0)
  let latus_rectum_length := p
  let A_coords := (2, 4)
  let K_coords := (6, 0) -- Given \( KF = 4 \), calculate and locate point K.
  ∃ (A F K : ℝ×ℝ), 
    F = focus_F ∧ 
    (A.1 = 2 ∧ A.2 = 4) ∧ 
    (K.1 = 6 ∧ K.2 = 0) ∧
    ∥(A.1 - K.1, A.2 - K.2)∥ = real.sqrt 2 * ∥(A.1 - F.1, A.2 - F.2)∥ →
    (1/2) * ∥(A.1 - K.1, A.2 - K.2)∥ * ∥(K.1 - F.1, K.2 - F.2)∥ * real.sin (real.pi / 4) = 8 :=
by
  intros
  sorry

end triangle_area_parabola_hyperbola_l440_440455


namespace nested_function_evaluation_l440_440541

def f (x : ℕ) : ℕ := x + 3
def g (x : ℕ) : ℕ := x / 2
def f_inv (x : ℕ) : ℕ := x - 3
def g_inv (x : ℕ) : ℕ := 2 * x

theorem nested_function_evaluation : 
  f (g_inv (f_inv (g_inv (f_inv (g (f 15)))))) = 21 := 
by 
  sorry

end nested_function_evaluation_l440_440541


namespace no_infinite_strictly_increasing_seq_l440_440746

-- Define the set Sk
def Sk (k : ℕ) (a : ℝ) : Prop := 
  ∃ (n : Fin k → ℕ), a = (Finset.univ.sum (λ i, (1 : ℝ) / n i))

-- Main theorem statement
theorem no_infinite_strictly_increasing_seq (k : ℕ) (h_k : k > 0) :
  ¬(∃ (f : ℕ → ℝ), (∀ i, Sk k (f i)) ∧ (∀ i, f i < f (i+1))) := sorry

end no_infinite_strictly_increasing_seq_l440_440746


namespace major_premise_wrong_l440_440748

theorem major_premise_wrong (a b : ℝ) :
  (a > b) → ¬ (a^2 > b^2) :=
by 
  assume h : a > b,
  -- Here we just use the given example to illustrate why the premise is wrong.
  have example : 2 > -2 := by norm_num,
  have not_example : ¬ (2^2 > (-2)^2) := by norm_num,
  exact sorry

end major_premise_wrong_l440_440748


namespace sum_of_digits_of_large_number_l440_440033

-- Define the conditions and the problem in Lean
theorem sum_of_digits_of_large_number (N : ℕ) (hN : N = 99999999999999999999999... (2015 times)) 
  (div9 : N % 9 = 0) :
  let a := Nat.digits 10 N in
  let b := a.sum in
  let c := (Nat.digits 10 b).sum in
  c = 9 := by
  sorry

end sum_of_digits_of_large_number_l440_440033


namespace find_line_eq_l440_440800

noncomputable def point := (real × real)

def line (a b c : real) : point → Prop := λ P, let (x, y) := P in a * x + b * y + c = 0

def line_eq (P Q : point) : real → real → real → Prop := 
  λ a b c, let (x1, y1) := P; let (x2, y2) := Q in 
  a * (x2 - x1) + b * (y2 - y1) = 0

def line_1 := line 2 1 (-3)
def line_2 := line 3 (-1) 6
def line_21x13y42 := line 21 13 (-42)
def line_xplusy2 := line 1 1 (-2)
def line_3x4y6 := line 3 4 (-6)

def M : point := (2, 0)

theorem find_line_eq
  (A B : point)
  (l_intersects : line_1 A ∧ line_2 A ∧ A = B ∧ A ≠ M)
  (sum_x1_x2_ne_0 : let (x1, _) := A; let (x2, _) := B in x1 + x2 ≠ 0) :
  ∃ (a b c : real), ((a = 21 ∧ b = 13 ∧ c = -42) ∨ (a = 1 ∧ b = 1 ∧ c = -2) ∨ (a = 3 ∧ b = 4 ∧ c = -6)) ∧ line_eq M A a b c := 
sorry

end find_line_eq_l440_440800


namespace equilateral_triangle_viviani_l440_440530

-- Definitions of the geometric objects and properties involved
variables {ABC P D E F : Type*} [metric_space D] [metric_space E] [metric_space F]
variables {s : ℝ} (triangle A B C : Point)
variables (PD PE PF : ℝ) (side AB BC CA : ℝ)

-- Proving the assertion
theorem equilateral_triangle_viviani (h_triangle: equilateral_triangle ABC s)
  (h_perp_PD: is_perpendicular P D (side BC))
  (h_perp_PE: is_perpendicular P E (side CA))
  (h_perp_PF: is_perpendicular P F (side AB)) :
  ((PD + PE + PF) / (AB + BC + CA)) = (1 / (2 * √3)) :=
sorry

end equilateral_triangle_viviani_l440_440530


namespace product_of_sines_one_product_of_sines_two_l440_440293

-- Problem: Prove the identities involving products of sines.
theorem product_of_sines_one (m : ℕ) (hm : 0 < m) :
  let prod_sin := ∏ k in Finset.range (m - 1) + 1, Real.sin (k * Real.pi / (2 * m))
  prod_sin = Real.sqrt m / (2 ^ (m - 1)) :=
by
  sorry

theorem product_of_sines_two (m : ℕ) (hm : 0 < m) :
  let prod_sin := ∏ k in Finset.range (2 * m), Real.sin ((2 * k + 1) * Real.pi / (4 * m))
  prod_sin = Real.sqrt 2 / (2 ^ m) :=
by
  sorry

end product_of_sines_one_product_of_sines_two_l440_440293


namespace g_two_l440_440486

theorem g_two (g : ℝ → ℝ) : (∀ x : ℝ, g (3 * x - 7) = 5 * x + 11) → g 2 = 26 :=
by
  intro h
  have x_val : 3 * 3 - 7 = 2 := by linarith
  have g_val : g 2 = 5 * 3 + 11 := by rw [←x_val, h 3]
  exact g_val

end g_two_l440_440486


namespace quadratic_eq_one_solution_has_ordered_pair_l440_440215

theorem quadratic_eq_one_solution_has_ordered_pair (a c : ℝ) 
  (h1 : a * c = 25) 
  (h2 : a + c = 17) 
  (h3 : a > c) : 
  (a, c) = (15.375, 1.625) :=
sorry

end quadratic_eq_one_solution_has_ordered_pair_l440_440215


namespace inequality_proof_l440_440540

noncomputable def a : ℝ := 0.3 ^ 2
noncomputable def b : ℝ := 2 ^ 0.3
noncomputable def c : ℝ := Real.logb 2 0.3

theorem inequality_proof : b > a ∧ a > c := by
  sorry

end inequality_proof_l440_440540


namespace total_film_length_in_meters_l440_440671

noncomputable def π : ℝ := 3.14
def core_diameter : ℝ := 60
def thickness : ℝ := 0.15
def num_turns : ℕ := 600

theorem total_film_length_in_meters:
  let length_of_film_mm := (core_diameter * num_turns + 0.3 * (num_turns - 1) * num_turns / 2) * π in
  length_of_film_mm / 1000 = 282.3 := by sorry

end total_film_length_in_meters_l440_440671


namespace tangent_line_eqn_at_origin_l440_440067

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / (Real.exp x)

def f' (x : ℝ) : ℝ := ( - (Real.sin x) - (Real.cos x)) / (Real.exp x)

theorem tangent_line_eqn_at_origin :
  let slope := f' 0 in let y_intercept := f 0 - slope * 0 in
  slope = -1 ∧ y_intercept = 1 → ∀ x y : ℝ, y = -x + 1 ↔ x + y - 1 = 0 :=
by
  intros slope y_intercept h
  sorry

end tangent_line_eqn_at_origin_l440_440067


namespace arithmetic_sequence_20th_term_l440_440627

open Real

theorem arithmetic_sequence_20th_term : 
  ∀ (a₁ d : ℤ) (n : ℕ), a₁ = 8 → d = -3 → n = 20 → a₁ + (n - 1) * d = -49 :=
by
  intros a₁ d n ha₁ hd hn
  rw [ha₁, hd, hn]
  rw [Int.ofNat_sub]
  simp
  sorry

end arithmetic_sequence_20th_term_l440_440627


namespace new_person_weight_l440_440192

noncomputable def weight_of_new_person (W : ℝ) : ℝ :=
  W + 61 - 25

theorem new_person_weight {W : ℝ} : 
  ((W + 61 - 25) / 12 = W / 12 + 3) → 
  weight_of_new_person W = 61 :=
by
  intro h
  sorry

end new_person_weight_l440_440192


namespace sum_of_three_consecutive_integers_divisible_by_3_l440_440209

theorem sum_of_three_consecutive_integers_divisible_by_3 (a : ℤ) :
  ∃ k : ℤ, k = 3 ∧ (a - 1 + a + (a + 1)) % k = 0 :=
by
  use 3
  sorry

end sum_of_three_consecutive_integers_divisible_by_3_l440_440209


namespace magic_8_ball_probability_l440_440526

theorem magic_8_ball_probability :
  let positive_prob := 3 / 7
  let negative_prob := 4 / 7
  let successes := 4
  let trials := 7
  let combinations (n k : ℕ) := nat.choose n k
  let prob (success_prob fail_prob : ℚ) (succs fails : ℕ) : ℚ :=
    success_prob ^ succs * fail_prob ^ fails
  let total_probability :=
    combinations trials successes *
    (prob positive_prob negative_prob successes (trials - successes))
  total_probability = 181440 / 823543 := 
by 
  sorry

end magic_8_ball_probability_l440_440526


namespace prob_divisor_of_12_l440_440668

theorem prob_divisor_of_12 :
  (∃ d : Finset ℕ, d = {1, 2, 3, 4, 6}) → (∃ s : Finset ℕ, s = {1, 2, 3, 4, 5, 6}) →
  let favorable := 5
  let total := 6
  favorable / total = (5 : ℚ / 6 ) := sorry

end prob_divisor_of_12_l440_440668


namespace percentage_in_first_subject_l440_440688

theorem percentage_in_first_subject (P : ℝ) (H1 : 80 = 80) (H2 : 75 = 75) (H3 : (P + 80 + 75) / 3 = 75) : P = 70 :=
by
  sorry

end percentage_in_first_subject_l440_440688


namespace find_k_l440_440041

theorem find_k (x₁ x₂ k : ℝ) (hx : x₁ + x₂ = 3) (h_prod : x₁ * x₂ = k) (h_cond : x₁ * x₂ + 2 * x₁ + 2 * x₂ = 1) : k = -5 :=
by
  sorry

end find_k_l440_440041


namespace greatest_possible_perimeter_l440_440505

theorem greatest_possible_perimeter (x : ℤ) (hx1 : 3 * x > 17) (hx2 : 17 > x) : 
  (3 * x + 17 ≤ 65) :=
by
  have Hx : x ≤ 16 := sorry -- Derived from inequalities hx1 and hx2
  have Hx_ge_6 : x ≥ 6 := sorry -- Derived from integer constraint and hx1, hx2
  sorry -- Show 3 * x + 17 has maximum value 65 when x = 16

end greatest_possible_perimeter_l440_440505


namespace ana_banana_probability_l440_440356

noncomputable def die_rolls : List ℕ := [1, 2, 3, 4, 5, 6]

def prob_multiple_of_6 (a1 a2 b1 b2 : ℕ) : Prop :=
  (a1 * b1 + a2 * b2) % 6 = 0

theorem ana_banana_probability :
  (1 / 6) = ∑ a1 in die_rolls, ∑ a2 in die_rolls, 
           ∑ b1 in die_rolls, ∑ b2 in die_rolls, 
           if prob_multiple_of_6 a1 a2 b1 b2 then 1 / (6 * 6) else 0 := 
begin
  sorry
end

end ana_banana_probability_l440_440356


namespace probability_of_four_ones_l440_440243

noncomputable def probability_exactly_four_ones : ℚ :=
  (Nat.choose 12 4 * (1/6)^4 * (5/6)^8)

theorem probability_of_four_ones :
  abs (probability_exactly_four_ones.toReal - 0.114) < 0.001 :=
by
  sorry

end probability_of_four_ones_l440_440243


namespace solve_system_of_equations_l440_440186

theorem solve_system_of_equations:
  (∀ x y : ℝ, 
    (7 * x^2 + 7 * y^2 - 3 * x^2 * y^2 = 7) ∧ 
    (x^4 + y^4 - x^2 * y^2 = 37) ↔ 
    (
      (x = sqrt 7 ∧ (y = sqrt 3 ∨ y = -sqrt 3)) ∨
      (x = -sqrt 7 ∧ (y = sqrt 3 ∨ y = -sqrt 3)) ∨
      (x = sqrt 3 ∧ (y = sqrt 7 ∨ y = -sqrt 7)) ∨
      (x = -sqrt 3 ∧ (y = sqrt 7 ∨ y = -sqrt 7))
    )) :=
by
  sorry

end solve_system_of_equations_l440_440186


namespace tangent_line_eq_l440_440207

noncomputable def A : Type := sorry
noncomputable def B : Type := sorry

theorem tangent_line_eq (α : ℝ) (x_B y_B : ℝ) 
  (h1 : x_B = 5 * real.cos (2 * α)) 
  (h2 : y_B = 5 * real.sin (2 * α)) 
  (h3 : x_B ^ 2 + y_B ^ 2 = 25) 
  (h4 : (7 * x_B + 24 * y_B + 125 = 0) ∨ (7 * x_B - 24 * y_B + 125 = 0)) :
  7 * x_B + 24 * y_B + 125 = 0 ∨ 7 * x_B - 24 * y_B + 125 = 0 :=
  by sorry

end tangent_line_eq_l440_440207


namespace compute_f3_l440_440157

def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 4*n + 3 else 2*n + 1

theorem compute_f3 : f (f (f 3)) = 99 :=
by
  sorry

end compute_f3_l440_440157


namespace non_officers_count_l440_440285

theorem non_officers_count (avg_salary_all : ℕ) (avg_salary_officers : ℕ) (avg_salary_non_officers : ℕ) (num_officers : ℕ) 
  (N : ℕ) 
  (h_avg_salary_all : avg_salary_all = 120) 
  (h_avg_salary_officers : avg_salary_officers = 430) 
  (h_avg_salary_non_officers : avg_salary_non_officers = 110) 
  (h_num_officers : num_officers = 15) 
  (h_eq : avg_salary_all * (num_officers + N) = avg_salary_officers * num_officers + avg_salary_non_officers * N) 
  : N = 465 :=
by
  -- Proof would be here
  sorry

end non_officers_count_l440_440285


namespace probability_of_four_ones_approx_l440_440251

noncomputable def probability_of_four_ones_in_twelve_dice : ℚ :=
  (nat.choose 12 4 : ℚ) * (1 / 6 : ℚ) ^ 4 * (5 / 6 : ℚ) ^ 8

theorem probability_of_four_ones_approx :
  probability_of_four_ones_in_twelve_dice ≈ 0.089 :=
sorry

end probability_of_four_ones_approx_l440_440251


namespace domino_placement_l440_440313
  
theorem domino_placement (n : ℕ) :
  let K := 2 * n choose n in
  (K * K) = (nat.choose (2 * n) n) ^ 2 := by
  sorry

end domino_placement_l440_440313


namespace amount_subtracted_is_30_l440_440306

-- Definitions based on conditions
def N : ℕ := 200
def subtracted_amount (A : ℕ) : Prop := 0.40 * (N : ℝ) - (A : ℝ) = 50

-- The theorem statement
theorem amount_subtracted_is_30 : subtracted_amount 30 :=
by 
  -- proof will be completed here
  sorry

end amount_subtracted_is_30_l440_440306


namespace least_odd_prime_factor_2027_l440_440739

-- Definitions for the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def order_divides (a n p : ℕ) : Prop := a ^ n % p = 1

-- Define lean function to denote the problem.
theorem least_odd_prime_factor_2027 :
  ∀ p : ℕ, 
  is_prime p → 
  order_divides 2027 12 p ∧ ¬ order_divides 2027 6 p → 
  p ≡ 1 [MOD 12] → 
  2027^6 + 1 % p = 0 → 
  p = 37 :=
by
  -- skipping proof steps
  sorry

end least_odd_prime_factor_2027_l440_440739


namespace triangle_angle_identity_l440_440795

theorem triangle_angle_identity (P Q R : ℝ) (hPQR : ∠PQR ∈ (set.Icc 0 π)) 
  (PQ PR QR : ℝ) (hPQ : PQ = 7) (hPR : PR = 8) (hQR : QR = 5) :
  (cos ((P - Q) / 2) / sin (R / 2)) - (sin ((P - Q) / 2) / cos (R / 2)) = 7 / 4 :=
by
  sorry

end triangle_angle_identity_l440_440795


namespace problem_statement_l440_440008

theorem problem_statement (a n : ℕ) (h1 : 1 ≤ a) (h2 : n = 1) : ∃ m : ℤ, ((a + 1)^n - a^n) = m * n := by
  sorry

end problem_statement_l440_440008


namespace sqrt_eq_pm_four_l440_440986

theorem sqrt_eq_pm_four (a : ℤ) : (a * a = 16) ↔ (a = 4 ∨ a = -4) :=
by sorry

end sqrt_eq_pm_four_l440_440986


namespace ramu_repairs_cost_l440_440959

theorem ramu_repairs_cost :
  ∃ R : ℝ, 64900 - (42000 + R) = (29.8 / 100) * (42000 + R) :=
by
  use 8006.16
  sorry

end ramu_repairs_cost_l440_440959


namespace geese_count_l440_440230

-- Define the number of ducks in the marsh
def number_of_ducks : ℕ := 37

-- Define the total number of birds in the marsh
def total_number_of_birds : ℕ := 95

-- Define the number of geese in the marsh
def number_of_geese : ℕ := total_number_of_birds - number_of_ducks

-- Theorem stating the number of geese in the marsh is 58
theorem geese_count : number_of_geese = 58 := by
  sorry

end geese_count_l440_440230


namespace number_of_solutions_l440_440720

def my_operation : ℝ → ℝ → ℝ := λ x y, 5 * x - 4 * y + 2 * x * y

theorem number_of_solutions : ∀ (f : ℝ → ℝ → ℝ), 
  (f = my_operation) → (∃! y : ℝ, f 4 y = 16) := by
sorry

end number_of_solutions_l440_440720


namespace odd_exponent_divisibility_l440_440623

theorem odd_exponent_divisibility (x y : ℤ) (k : ℕ) (h : (x^(2*k-1) + y^(2*k-1)) % (x + y) = 0) : 
  (x^(2*k+1) + y^(2*k+1)) % (x + y) = 0 :=
sorry

end odd_exponent_divisibility_l440_440623


namespace construction_paper_width_l440_440525

theorem construction_paper_width
    (width_of_second_piece : ℕ)
    (strip_width : ℕ)
    (width_first_piece_is_multiple : 0 < strip_width)
    : (∃ n : ℕ, width_of_second_piece = strip_width * n) →
    (∃ m : ℕ, ∃ n : ℕ, (strip_width = 11) → ∃ x : ℕ, x = 11 * m) :=
by {
  intros h,
  obtain ⟨n, hn⟩ := h,
  intros hw_strip_width,
  use n,
  rw hw_strip_width,
  sorry
}

end construction_paper_width_l440_440525


namespace discount_percentage_is_15_l440_440981

-- Definitions and conditions
def regular_price_per_can : ℝ := 0.30
def price_for_72_cans_discounted : ℝ := 18.36

-- The total regular price for 72 cans
def total_regular_price : ℝ := 72 * regular_price_per_can

-- The discount amount
def discount_amount : ℝ := total_regular_price - price_for_72_cans_discounted

-- The discount percentage calculation
def discount_percentage : ℝ := (discount_amount / total_regular_price) * 100

-- Prove that the discount percentage is 15%
theorem discount_percentage_is_15 :
  discount_percentage = 15 :=
by
  -- No proof required, just a statement.
  sorry

end discount_percentage_is_15_l440_440981


namespace solve_for_x_l440_440635

theorem solve_for_x (x : ℝ) (h : (2012 + x)^2 = x^2) : x = -1006 := 
sorry

end solve_for_x_l440_440635


namespace finite_writing_process_l440_440945

theorem finite_writing_process (a : ℕ → ℕ) :
  (∀ n, ¬∃ (k : ℕ) (k_1 k_2 ... k_{n-1} : {nn : ℕ // nn ≥ 0}),
    a n = a 1 * k_1 + a 2 * k_2 + ... + a (n-1) * k_(n-1)) →
  ∃ N, ∀ n, n ≥ N → a n = 0 :=
sorry

end finite_writing_process_l440_440945


namespace distance_points_l440_440709

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_points :
  distance 1 3 6 7 = real.sqrt 41 :=
by
  sorry

end distance_points_l440_440709


namespace tangent_parallel_to_line_l440_440101

def tangent_line_eq (x y : ℝ) (l : ℝ → ℝ) := l = λ x, x^4

theorem tangent_parallel_to_line (l : ℝ → ℝ) (x₀ y₀ : ℝ) 
  (h_deriv : ∀ x, deriv (λ x, x^4) x = 4 * x^3) 
  (h_point : y₀ = (-1/2)^4) 
  (h_parallel : 4 * (-1/2)^3 = -1/2) :
  tangent_line_eq 8 16 y₀ := 
sorry

end tangent_parallel_to_line_l440_440101


namespace determine_magical_diamond_l440_440694

theorem determine_magical_diamond (diamonds : Fin 11 → ℕ) (regular : Fin 10 → ℕ) (magical : ℕ)
  (h_diamonds : ∃ i : Fin 11, diamonds i = magical ∧ ∀ j : Fin 10, diamonds j ≠ diamonds (10 : Fin 11))
  : ∃ heavier_lighter : bool, -- return if it is heavier (true) or lighter (false)
      let weigh := λ (a b : List (Fin 11)), a.sum < b.sum in
      (weigh [0, 1, 2] [3, 4, 5] ∨ weigh [3, 4, 5] [0, 1, 2]) ∨
      ((∃ (s : Set (Fin 11)), diamonds ∈ (s ∧ weigh (s.toList) (finset.fin_range 6 \ s).to_list) ∨
      ¬weigh (finset.fin_range 6 \ s).to_list (s.toList)) ∧
       ∃ (s : Set (Fin 11)),|s| = 5 ∧ (weigh s.to_list (finset.fin_range 11 \ s).to_list) ∨
      ¬weigh (finset.fin_range 11 \ s).to_list s.to_list) :=
    sorry

end determine_magical_diamond_l440_440694


namespace imaginary_part_conjugate_l440_440036

theorem imaginary_part_conjugate (z : ℂ) (h : z * (complex.I - 1) = 1 + complex.I) : complex.im (conj z) = 1 := 
by
  sorry

end imaginary_part_conjugate_l440_440036


namespace abs_eq_solutions_2_l440_440089

theorem abs_eq_solutions_2 (x : ℝ) : 
  ∀ l : ℝ, l = 1 → 
  (∀ y : ℝ, (|x - 2| - |x - 6| = 1 → y = 4.5 ∨ y = 3.5) →
  (|x - 2| - |x - 6| = -1 → y = 4.5 ∨ y = 3.5)) →
  ∃! (x : ℝ), ||x - 2| - |x - 6|| = l :=
begin
  sorry
end

end abs_eq_solutions_2_l440_440089


namespace prime_between_40_50_largest_prime_lt_100_l440_440678

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def between (n m k : ℕ) : Prop := n < k ∧ k < m

theorem prime_between_40_50 :
  {x : ℕ | between 40 50 x ∧ isPrime x} = {41, 43, 47} :=
sorry

theorem largest_prime_lt_100 :
  ∃ p : ℕ, isPrime p ∧ p < 100 ∧ ∀ q : ℕ, isPrime q ∧ q < 100 → q ≤ p :=
sorry

end prime_between_40_50_largest_prime_lt_100_l440_440678


namespace probability_of_24_is_1_div_1296_l440_440750

def sum_of_dice_is_24 (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 + d2 + d3 + d4 = 24

def probability_of_six (d : ℕ) : Rat :=
  if d = 6 then 1 / 6 else 0

def probability_of_sum_24 (d1 d2 d3 d4 : ℕ) : Rat :=
  (probability_of_six d1) * (probability_of_six d2) * (probability_of_six d3) * (probability_of_six d4)

theorem probability_of_24_is_1_div_1296 :
  (probability_of_sum_24 6 6 6 6) = 1 / 1296 :=
by
  sorry

end probability_of_24_is_1_div_1296_l440_440750


namespace circumcenter_incenter_perpendicular_l440_440809

-- Define a triangle with its incenter and circumcenter
variables {A B C O I : Type}

-- Define CA, BC, AB, and required conditions
variables (CA BC AB : ℝ)
variables (h_arith_seq : 2 * BC = CA + AB) (circumcenter : O = circumcenter_of_triangle A B C)
variables (incenter : I = incenter_of_triangle A B C)

-- Main theorem statement
theorem circumcenter_incenter_perpendicular (CA BC AB : ℝ)
  (h_arith_seq : 2 * BC = CA + AB)
  (circumcenter : O = circumcenter_of_triangle A B C)
  (incenter : I = incenter_of_triangle A B C) :
  perp (line_through O I) (line_through I A) :=
sorry

end circumcenter_incenter_perpendicular_l440_440809


namespace thief_speed_l440_440339

theorem thief_speed 
    (d_initial : ℝ) -- Initial distance (in km) between thief and policeman
    (d_thief : ℝ) -- Distance (in km) the thief will run before being overtaken
    (v_policeman : ℝ) -- Speed of the policeman in km/hr
    (d_initial = 0.175)
    (d_thief = 0.7)
    (v_policeman = 10) :
    ∃ v_thief : ℝ, v_thief = d_thief / (d_initial + d_thief) / v_policeman  := 
begin
    sorry
end

end thief_speed_l440_440339


namespace probability_of_24_is_1_div_1296_l440_440752

def sum_of_dice_is_24 (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 + d2 + d3 + d4 = 24

def probability_of_six (d : ℕ) : Rat :=
  if d = 6 then 1 / 6 else 0

def probability_of_sum_24 (d1 d2 d3 d4 : ℕ) : Rat :=
  (probability_of_six d1) * (probability_of_six d2) * (probability_of_six d3) * (probability_of_six d4)

theorem probability_of_24_is_1_div_1296 :
  (probability_of_sum_24 6 6 6 6) = 1 / 1296 :=
by
  sorry

end probability_of_24_is_1_div_1296_l440_440752


namespace find_f_eq_l440_440544

-- Let's define f and the conditions provided in the problem.
variable {ℝ : Type*} [linear_ordered_field ℝ]

-- Definition of the function f with the conditions
noncomputable def f (x : ℝ) : ℝ := by 
  sorry -- Definition based solely on the conditions is not provided

-- Conditions
axiom f_condition1 : f 0 = 2
axiom f_condition2 : ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2 + 2 * x * y) / 2) + (x - y)^2

-- The main theorem to prove
theorem find_f_eq : ∀ x : ℝ, f x = 2 - 2 * x :=
by
sorry

end find_f_eq_l440_440544


namespace findPrincipalSum_l440_440286

variable (P : ℝ) -- declare the principal sum P
variable (R : ℝ := 10) -- declare the rate of interest as 10%
variable (T : ℝ := 2) -- declare the time period as 2 years
variable (difference : ℝ := 631) -- declare the difference between CI and SI as 631

def simpleInterest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T / 100

def compoundInterest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * ((1 + R / 100) ^ T - 1)

theorem findPrincipalSum : (compoundInterest P R T) - (simpleInterest P R T) = difference → P = 63100 :=
by
  sorry

end findPrincipalSum_l440_440286


namespace find_larger_integer_l440_440288

variable (x : ℤ) (smaller larger : ℤ)
variable (ratio_1_to_4 : smaller = 1 * x ∧ larger = 4 * x)
variable (condition : smaller + 12 = larger)

theorem find_larger_integer : larger = 16 :=
by
  sorry

end find_larger_integer_l440_440288


namespace translate_parabola_l440_440235

theorem translate_parabola (x : ℝ):
  (let f := λ x : ℝ, 2 * x^2 in
   let f1 := λ x : ℝ, f (x + 3) in
   let f2 := λ x : ℝ, f1 x + 4 in
   f2 x = 2 * (x + 3)^2 + 4) := 
sorry

end translate_parabola_l440_440235


namespace student_club_joining_ways_l440_440183

theorem student_club_joining_ways :
  let clubs := ["Chunhui Literature Society", "Dancers Roller Skating Club", "Basketball Home", "Go Garden"]
  let students := ["A", "B", "C", "D", "E"]
  let conditions := (∀ s, s ∈ students → ∃ c, c ∈ clubs ∧ (join s c)) ∧  -- each student joins one club
                    (∀ c, c ∈ clubs → ∃ s, s ∈ students ∧ (join s c)) ∧  -- each club has at least one student
                    (join "A" "Go Garden" = false) ∧  -- student A does not join "Go Garden"
                    (∀ s c1 c2, join s c1 → join s c2 → c1 = c2)  -- each student joins only one club
  in number_of_ways_to_join students clubs (join "A" "Go Garden" = false) = 216 :=
begin
  sorry
end

end student_club_joining_ways_l440_440183


namespace die_roll_divisor_of_12_prob_l440_440663

def fair_die_probability_divisor_of_12 : Prop :=
  let favorable_outcomes := {1, 2, 3, 4, 6}
  let total_outcomes := 6
  let probability := favorable_outcomes.size / total_outcomes
  probability = 5 / 6

theorem die_roll_divisor_of_12_prob:
  fair_die_probability_divisor_of_12 :=
by
  sorry

end die_roll_divisor_of_12_prob_l440_440663


namespace exists_line_through_P_l440_440431

variable {k : Type*} [field k] {V : Type*} [add_comm_group V] [module k V]

variables (e1 e2 e3 : affine_subspace k V)
variables (A1 A2 A3 P : V)
variables (B1 B2 B3 : V)

-- Conditions
axiom parallel_lines : e1 ∥ e2 ∧ e2 ∥ e3 ∧ e1 ∥ e3
axiom A1_on_e1 : A1 ∈ e1
axiom A2_on_e2 : A2 ∈ e2
axiom A3_on_e3 : A3 ∈ e3
axiom B1_on_e1 : B1 ∈ e1
axiom B2_on_e2 : B2 ∈ e2
axiom B3_on_e3 : B3 ∈ e3
axiom P_in_plane : ∃ plane, P ∈ plane ∧ A1 ∈ plane ∧ A2 ∈ plane ∧ A3 ∈ plane

-- Question
theorem exists_line_through_P (P A1 A2 A3 B1 B2 B3 : V) :
  (∃ (f : affine_subspace k V), P ∈ f ∧ (B1 ∈ f) ∧ (B2 ∈ f) ∧ (B3 ∈ f) ∧ 
  (B1 ∈ e1) ∧ (B2 ∈ e2) ∧ (B3 ∈ e3) ∧ 
  (A1 ∈ e1) ∧ (A2 ∈ e2) ∧ (A3 ∈ e3) ∧ 
  (parallel_lines e1 e2 e3) ∧ 
  (A1_on_e1 A1 e1) ∧ (A2_on_e2 A2 e2) ∧ (A3_on_e3 A3 e3) ∧ 
  (B1_on_e1 B1 e1) ∧ (B2_on_e2 B2 e2) ∧ (B3_on_e3 B3 e3)) → 
  \overrightarrow{A1 B1} + \overrightarrow{A2 B2} = \overrightarrow{A3 B3}) :=
sorry

end exists_line_through_P_l440_440431


namespace sum_arith_seq_eleven_l440_440095

variable {α : Type*} [LinearOrderedField α]

def arith_seq (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

def sum_arith_seq (a d : α) : ℕ → α
| 0       := 0
| (n + 1) := (n + 1) * (a + (n : ℕ) * d) / 2

theorem sum_arith_seq_eleven (a d : α) (h : sum_arith_seq a d 8 - sum_arith_seq a d 3 = 10) : 
  sum_arith_seq a d 11 = 16 := 
sorry

end sum_arith_seq_eleven_l440_440095


namespace probability_all_quitters_same_tribe_l440_440219

theorem probability_all_quitters_same_tribe :
  ∀ (people : Finset ℕ) (tribe1 tribe2 : Finset ℕ) (choose : ℕ → ℕ → ℕ) (prob : ℚ),
  people.card = 20 →
  tribe1.card = 10 →
  tribe2.card = 10 →
  tribe1 ∪ tribe2 = people →
  tribe1 ∩ tribe2 = ∅ →
  choose 20 3 = 1140 →
  choose 10 3 = 120 →
  prob = (2 * choose 10 3) / choose 20 3 →
  prob = 20 / 95 :=
by
  intro people tribe1 tribe2 choose prob
  intros hp20 ht1 ht2 hu hi hchoose20 hchoose10 hprob
  sorry

end probability_all_quitters_same_tribe_l440_440219


namespace parkway_girls_not_playing_soccer_l440_440123

theorem parkway_girls_not_playing_soccer:
  ∀ (total_students boys soccer_players: ℕ) 
    (pct_soccer_boys : ℚ)
    (h1 : total_students = 420)
    (h2 : boys = 312)
    (h3 : soccer_players = 250)
    (h4 : pct_soccer_boys = 0.86),
  let girls := total_students - boys,
      boys_playing_soccer := pct_soccer_boys * soccer_players,
      girls_playing_soccer := soccer_players - boys_playing_soccer,
      girls_not_playing_soccer := girls - girls_playing_soccer 
  in girls_not_playing_soccer = 73 := 
by
  intros total_students boys soccer_players pct_soccer_boys h1 h2 h3 h4
  let girls := total_students - boys
  let boys_playing_soccer := pct_soccer_boys * soccer_players
  let girls_playing_soccer := soccer_players - boys_playing_soccer
  let girls_not_playing_soccer := girls - girls_playing_soccer

  sorry

end parkway_girls_not_playing_soccer_l440_440123


namespace find_smaller_number_l440_440604

theorem find_smaller_number (a b : ℤ) (h1 : a + b = 18) (h2 : a - b = 24) : b = -3 :=
by
  sorry

end find_smaller_number_l440_440604


namespace total_books_l440_440641

theorem total_books (Zig_books : ℕ) (Flo_books : ℕ) (Tim_books : ℕ) 
  (hz : Zig_books = 60) (hf : Zig_books = 4 * Flo_books) (ht : Tim_books = Flo_books / 2) :
  Zig_books + Flo_books + Tim_books = 82 := by
  sorry

end total_books_l440_440641


namespace unique_four_digit_numbers_l440_440366

theorem unique_four_digit_numbers (digits : Finset ℕ) (odd_digits : Finset ℕ) :
  digits = {2, 3, 4, 5, 6} → 
  odd_digits = {3, 5} → 
  ∃ (n : ℕ), n = 14 :=
by
  sorry

end unique_four_digit_numbers_l440_440366


namespace nabla_property_l440_440422

-- Define the operation ∇
def nabla (a b : ℝ) := (a + b) / (1 + a * b)

-- The main theorem statement
theorem nabla_property : 
  0 < 1 ∧ 0 < 3 ∧ 0 < 2 → nabla (nabla 1 3) 2 = 1 :=
by
  intro h,
  sorry

end nabla_property_l440_440422


namespace janabel_widgets_total_l440_440565

theorem janabel_widgets_total (n : ℕ) (a_1 : ℕ) (d : ℕ) (h₁ : a_1 = 2) (h₂ : d = 2) (h₃ : n = 15) :
  let a_n := a_1 + (n - 1) * d,
      S_n := n * (a_1 + a_n) / 2
  in S_n = 240 := 
by 
  /- Assume and apply the given conditions -/
  rw [h₁, h₂, h₃]
  /- Simplify the arithmetic sequence and the sum formula -/
  have a_n_def : a_n = 2 + (15 - 1) * 2 := by sorry
  have S_n_def : S_n = 15 * (2 + a_n) / 2 := by sorry
  rw a_n_def at S_n_def
  have S_n_value : 7.5 * 32 = 240 := by sorry
  exact S_n_value

end janabel_widgets_total_l440_440565


namespace solution_set_for_f_l440_440039

theorem solution_set_for_f (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f(x + y) + 2 = f(x) + f(y))
(h2 : ∀ x : ℝ, 0 < x → f(x) > 2) (h3 : f(3) = 5) :
  {a : ℝ | f(a^2 - 2*a - 2) < 3} = {a : ℝ | -1 < a ∧ a < 3} :=
sorry

end solution_set_for_f_l440_440039


namespace p_implies_q_l440_440799

theorem p_implies_q (x : ℝ) (h : |5 * x - 1| > 4) : x^2 - (3/2) * x + (1/2) > 0 := sorry

end p_implies_q_l440_440799


namespace distance_calculation_l440_440404

noncomputable def distance_from_point_to_line (point : ℝ × ℝ × ℝ) (line_point : ℝ × ℝ × ℝ) (line_direction : ℝ × ℝ × ℝ) : ℝ :=
  let (px, py, pz) := point in
  let (lx, ly, lz) := line_point in
  let (dx, dy, dz) := line_direction in
  let t := (-(dx * (px - lx) + dy * (py - ly) + dz * (pz - lz))) / (dx^2 + dy^2 + dz^2) in
  let closest_point := (lx + t * dx, ly + t * dy, lz + t * dz) in
  let dist_vector := ((closest_point.1 - px), (closest_point.2 - py), (closest_point.3 - pz)) in
  Real.sqrt((dist_vector.1)^2 + (dist_vector.2)^2 + (dist_vector.3)^2)

theorem distance_calculation :
  distance_from_point_to_line (0, 1, 5) (4, 5, 6) (3, -1, 2) = Real.sqrt(1262) / 7 :=
by
  sorry

end distance_calculation_l440_440404


namespace prob_A_and_B_truth_l440_440855

-- Define the probabilities
def prob_A_truth := 0.70
def prob_B_truth := 0.60

-- State the theorem
theorem prob_A_and_B_truth : prob_A_truth * prob_B_truth = 0.42 :=
by
  sorry

end prob_A_and_B_truth_l440_440855


namespace smallest_positive_period_of_f_intervals_where_f_is_increasing_l440_440462

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * (Real.sin (x + π / 8))^2 + 2 * Real.sin (x + π / 8) * Real.cos (x + π / 8)

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ ε > 0, ε < T → ∃ y, y > 0 ∧ ∀ z, z < y → ¬ (f (z + ε) = f z)) :=
  sorry

theorem intervals_where_f_is_increasing :
  ∀ k ∈ ℤ, ∀ x, k * π - π / 2 ≤ x ∧ x ≤ k * π → ∀ y, x ≤ y → f y ≥ f x :=
  sorry

end smallest_positive_period_of_f_intervals_where_f_is_increasing_l440_440462


namespace dice_sum_24_l440_440763

noncomputable def probability_of_sum_24 : ℚ :=
  let die_probability := (1 : ℚ) / 6
  in die_probability ^ 4

theorem dice_sum_24 :
  ∑ x in {x | x ∈ {1, 2, 3, 4, 5, 6} ∧ x = 6} = 24 → probability_of_sum_24 = 1 / 1296 :=
sorry

end dice_sum_24_l440_440763


namespace closest_whole_number_ratio_l440_440725

theorem closest_whole_number_ratio :
  let r := (10^3000 + 10^3003) / (10^3001 + 10^3002)
  ∃ k : ℤ, abs (r - k) ≤ 0.5 ∧ k = 9 :=
by
  -- let r := (10^3000 + 10^3003) / (10^3001 + 10^3002)
  let r := (10^3000 + 10^3003) / (10^3001 + 10^3002)
  -- we will show that the closest integer is 9
  existsi (9 : ℤ)
  -- insert necessary calculations here
  sorry

end closest_whole_number_ratio_l440_440725


namespace power_function_evaluation_l440_440495

theorem power_function_evaluation :
  ∃ (a : ℝ), (∀ x : ℝ, f x = x^a) ∧ (4^a = 2) ∧ (f 16 = 4) :=
by
  sorry

end power_function_evaluation_l440_440495


namespace incorrect_equation_in_list_l440_440347

theorem incorrect_equation_in_list :
  ¬ (sqrt (121 / 225) = - (11 / 15)) ∧
  ¬ (sqrt (121 / 225) = 11 / 15) :=
by
  let sqrt_64 := 8
  -- Given conditions (equations to verify):
  let eq1 := (sqrt_64 = 8) ∨ (sqrt_64 = -8) -- A. $\pm \sqrt{64}= \pm 8$
  let eq2 := (sqrt (121 / 225) = 11 / 15) ∨ (sqrt (121 / 225) = - (11 / 15)) -- B. $\sqrt{\dfrac{121}{225}}= \pm \dfrac{11}{15}$
  let eq3 := (Real.cbrt (-216) = -6) -- C. $\sqrt[3]{-216}= -6$
  let eq4 := (Real.cbrt (0.001) = 0.1 ∧ - Real.cbrt (0.001) = -0.1)-- D. $-\sqrt[3]{0.001}= -0.1$
  
  -- The proposition that the second equation is incorrect
  cases eq2 with eq2_pos eq2_neg
  { intro h,
    contradiction,
    -- The second option is incorrect for positive root
    exact sorry, -- Details skipped
  }
  { intro h,
    contradiction,
    -- The second option is incorrect for negative root
    exact sorry, -- Details skipped
  }

end incorrect_equation_in_list_l440_440347


namespace theta_in_fourth_quadrant_l440_440778

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.tan (θ + Real.pi / 4) = 1 / 3) : 
  (θ > 3 * Real.pi / 2) ∧ (θ < 2 * Real.pi) :=
sorry

end theta_in_fourth_quadrant_l440_440778


namespace min_perimeter_triangle_ABC_l440_440106

noncomputable def minPerimeter (a b c : ℝ) := a + b + c

theorem min_perimeter_triangle_ABC (a b c : ℝ) (h₁ : a + b = 10) (h₂ : 2 * (real.cos (real.arccos (-1/2)))^2 - 3 * (real.cos (real.arccos (-1/2))) - 2 = 0) :
  minPerimeter a b c = 10 + 5 * real.sqrt 3 :=
sorry

end min_perimeter_triangle_ABC_l440_440106


namespace maximum_BD_cyclic_quad_l440_440533

theorem maximum_BD_cyclic_quad (AB BC CD : ℤ) (BD : ℝ)
  (h_side_bounds : AB < 15 ∧ BC < 15 ∧ CD < 15)
  (h_distinct_sides : AB ≠ BC ∧ BC ≠ CD ∧ CD ≠ AB)
  (h_AB_value : AB = 13)
  (h_BC_value : BC = 5)
  (h_CD_value : CD = 8)
  (h_sides_product : BC * CD = AB * (10 : ℤ)) :
  BD = Real.sqrt 179 := 
by 
  sorry

end maximum_BD_cyclic_quad_l440_440533


namespace sum_of_first_100_terms_is_5_l440_440793

-- Define the sequence {a_n}
def a (n : ℕ) : ℤ :=
  if n = 0 then 1
  else if n = 1 then 3
  else a (n - 1) - a (n - 2)

-- Define the sum of the first 100 terms of the sequence
def sum_first_100 : ℤ := (Finset.range 100).sum (λ n, a n)

-- The theorem: the sum of the first 100 terms of the sequence is 5
theorem sum_of_first_100_terms_is_5 : sum_first_100 = 5 := by sorry

end sum_of_first_100_terms_is_5_l440_440793


namespace wheat_distribution_l440_440291

theorem wheat_distribution (x y z : ℕ) (h1 : 3 * x + 2 * y + z / 2 = 100) (h2 : x + y + z = 100) :
  (x = 20 ∧ y = 0 ∧ z = 80) ∨
  (x = 17 ∧ y = 5 ∧ z = 78) ∨
  (x = 14 ∧ y = 10 ∧ z = 76) ∨
  (x = 11 ∧ y = 15 ∧ z = 74) ∨
  (x = 8 ∧ y = 20 ∧ z = 72) ∨
  (x = 5 ∧ y = 25 ∧ z = 70) ∨
  (x = 2 ∧ y = 30 ∧ z = 68) :=
begin
  sorry
end

end wheat_distribution_l440_440291


namespace sum_of_coords_of_reflected_midpoint_l440_440570

-- Define the original points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (13, 16)

-- Define the midpoint function
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Calculate the midpoint of AB
def N := midpoint A B

-- Define the reflection over the y-axis
def reflect_y (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, P.2)

-- Calculate the reflected points A' and B'
def A' := reflect_y A
def B' := reflect_y B

-- Calculate the midpoint of the reflected points
def N' := midpoint A' B'

-- Theorem: Prove that the sum of the coordinates of N' is 1
theorem sum_of_coords_of_reflected_midpoint :
  (N'.1 + N'.2) = 1 := by
  sorry

end sum_of_coords_of_reflected_midpoint_l440_440570


namespace probability_of_four_ones_approx_l440_440252

noncomputable def probability_of_four_ones_in_twelve_dice : ℚ :=
  (nat.choose 12 4 : ℚ) * (1 / 6 : ℚ) ^ 4 * (5 / 6 : ℚ) ^ 8

theorem probability_of_four_ones_approx :
  probability_of_four_ones_in_twelve_dice ≈ 0.089 :=
sorry

end probability_of_four_ones_approx_l440_440252


namespace Pascal_hexagon_collinear_l440_440195

-- Define the points B, D, K, M, N, L corresponding to the vertices and intersection points in hexagon BKMDNL
variables {B D K M N L X Y Z : Type}

-- Assume that these points form the structure of the hexagon around the circle passing through vertices B and D of the quadrilateral ABCD
variable (ABCD : Quadrilateral B D K M N L)

-- Define the intersections given in the conditions
variable (h1 : Segment KM ∩ Segment LN = X)
variable (h2 : Segment MD ∩ Segment BK = Y)
variable (h3 : Segment NL ∩ Segment DB = Z)

-- State Pascal's theorem for the hexagon
theorem Pascal_hexagon_collinear (ABCD : Quadrilateral B D K M N L)
  (h1 : Segment KM ∩ Segment LN = X)
  (h2 : Segment MD ∩ Segment BK = Y)
  (h3 : Segment NL ∩ Segment DB = Z) : 
  collinear {X, Y, Z} :=
sorry

end Pascal_hexagon_collinear_l440_440195


namespace fraction_not_whole_l440_440639

theorem fraction_not_whole : 
  ¬ (Real.is_int (60 / 8)) :=
by
  sorry

end fraction_not_whole_l440_440639


namespace total_cookies_l440_440935

def mona_cookies : ℕ := 20
def jasmine_cookies : ℕ := mona_cookies - 5
def rachel_cookies : ℕ := jasmine_cookies + 10

theorem total_cookies : mona_cookies + jasmine_cookies + rachel_cookies = 60 := 
by
  have h1 : jasmine_cookies = 15 := by sorry
  have h2 : rachel_cookies = 25 := by sorry
  have h3 : mona_cookies = 20 := by sorry
  sorry

end total_cookies_l440_440935


namespace tangents_of_convex_quad_l440_440110

theorem tangents_of_convex_quad (
  α β γ δ : ℝ
) (m : ℝ) (h₀ : α + β + γ + δ = 2 * Real.pi) (h₁ : 0 < α ∧ α < Real.pi) (h₂ : 0 < β ∧ β < Real.pi) 
  (h₃ : 0 < γ ∧ γ < Real.pi) (h₄ : 0 < δ ∧ δ < Real.pi) (t1 : Real.tan α = m) :
  ¬ (Real.tan β = m ∧ Real.tan γ = m ∧ Real.tan δ = m) :=
sorry

end tangents_of_convex_quad_l440_440110


namespace rectangle_area_l440_440979

-- Definitions based on conditions
variables {x d : ℝ}
def length := 3 * x
def width := 2 * x
def diagonal := d
def pythagorean_identity := d ^ 2 = (3 * x) ^ 2 + (2 * x) ^ 2
def area := length * width

-- Goal
theorem rectangle_area (h: pythagorean_identity) : area = (6 / 13) * (d ^ 2) :=
by sorry

end rectangle_area_l440_440979


namespace ellipse_standard_equation_l440_440409

theorem ellipse_standard_equation :
  ∃ (foci : ℝ) (minor_axis_length : ℝ), foci = 5 ∧ minor_axis_length = 2 ∧
  (∀ (b : ℝ), b = 1 →
  (∀ (a : ℝ), a^2 = b^2 + foci →
  (x^2 + (y^2 / a^2) = 1))) :=
begin
  sorry
end

end ellipse_standard_equation_l440_440409


namespace sin_inequality_of_triangle_l440_440521

theorem sin_inequality_of_triangle (B C : ℝ) (hB : 0 < B) (hB_lt_pi : B < π) 
(hC : 0 < C) (hC_lt_pi : C < π) :
  (B > C) ↔ (Real.sin B > Real.sin C) := 
  sorry

end sin_inequality_of_triangle_l440_440521


namespace find_b_l440_440805

-- Define the quadratic equation
def quadratic_eq (b : ℝ) (x : ℝ) : ℝ :=
  x^2 + b * x - 15

-- Prove that b = 49/8 given -8 is a solution to the quadratic equation
theorem find_b (b : ℝ) : quadratic_eq b (-8) = 0 -> b = 49 / 8 :=
by
  intro h
  sorry

end find_b_l440_440805


namespace parabola_equation_l440_440819

theorem parabola_equation
  (axis_of_symmetry : ∀ x y : ℝ, x = 1)
  (focus : ∀ x y : ℝ, x = -1 ∧ y = 0) :
  ∀ y x : ℝ, y^2 = -4*x := 
sorry

end parabola_equation_l440_440819


namespace rewrite_equation_l440_440573

theorem rewrite_equation (x y : ℝ) (h : 2 * x - y = 4) : y = 2 * x - 4 :=
by
  sorry

end rewrite_equation_l440_440573


namespace coplanar_direction_vectors_l440_440086

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

/- Define the condition: v1, v2, and v3 are the direction vectors of lines l1, l2, and l3 respectively, which are parallel to the same plane. -/
variables (v1 v2 v3 : V)

-- Theorem statement
theorem coplanar_direction_vectors
  (h_parallel_plane : ∃ (u v : V), ¬ (u = 0) ∧ ¬ (v = 0) ∧ ¬ (linear_independent ℝ ![u, v]) 
    ∧ (v1 = λu) ∧ (v2 = μu + νv) ∧ (v3 = γu + δv)) :
  ∃ (λ μ : ℝ), v1 = λ • v2 + μ • v3 :=
  sorry

end coplanar_direction_vectors_l440_440086


namespace num_segments_le_three_n_l440_440564

theorem num_segments_le_three_n (n : ℕ) 
  (points : Fin n → ℝ × ℝ) 
  (h : ∀ i j : Fin n, i ≠ j → dist (points i) (points j) ≥ 1) :
  (∑ i : Fin n, ∑ j : Fin n, if i ≠ j ∧ dist (points i) (points j) = 1 then 1 else 0) ≤ 3 * n :=
sorry

end num_segments_le_three_n_l440_440564


namespace sum_ages_in_five_years_l440_440560

theorem sum_ages_in_five_years (L J : ℕ) (hL : L = 13) (h_relation : L = 2 * J + 3) : 
  (L + 5) + (J + 5) = 28 := 
by 
  sorry

end sum_ages_in_five_years_l440_440560


namespace sale_price_percentage_l440_440318

-- Defining the initial conditions: original price and discount rates
def original_price := 400
def first_discount_rate := 0.25
def second_discount_rate := 0.10

-- Calculate the price after successive discounts and express it as a percentage of the original price
theorem sale_price_percentage :
  let P := original_price in
  let P1 := P * (1 - first_discount_rate) in
  let P2 := P1 * (1 - second_discount_rate) in
  P2 / P = 0.675 :=
by
  sorry

end sale_price_percentage_l440_440318


namespace problem_statement_l440_440005

theorem problem_statement (a n : ℕ) (h_a : a ≥ 1) (h_n : n ≥ 1) :
  (∃ k : ℕ, (a + 1)^n - a^n = k * n) ↔ n = 1 := by
  sorry

end problem_statement_l440_440005


namespace compute_b_l440_440104

theorem compute_b (x y b : ℚ) (h1 : 5 * x - 2 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (hy : y = 3) :
  b = 13 / 2 :=
sorry

end compute_b_l440_440104


namespace arith_seq_ratio_l440_440121

theorem arith_seq_ratio (a_2 a_3 S_4 S_5 : ℕ) 
  (arithmetic_seq : ∀ n : ℕ, ℕ)
  (sum_of_first_n_terms : ∀ n : ℕ, ℕ)
  (h1 : (a_2 : ℚ) / a_3 = 1 / 3) 
  (h2 : S_4 = 4 * (a_2 - (a_3 - a_2)) + ((4 * 3 * (a_3 - a_2)) / 2)) 
  (h3 : S_5 = 5 * (a_2 - (a_3 - a_2)) + ((5 * 4 * (a_3 - a_2)) / 2)) :
  (S_4 : ℚ) / S_5 = 8 / 15 :=
by sorry

end arith_seq_ratio_l440_440121


namespace triangle_AF_perpendicular_BC_l440_440127

noncomputable def point : Type := ℝ × ℝ

structure Triangle :=
  (A B C D E F : point)
  (angle_BAC : Real := 40)
  (angle_ABC : Real := 60)
  (angle_CBD : Real := 40)
  (angle_BCE : Real := 70)
  (line_AC : ℝ := (C.1 - A.1))
  (line_AB : ℝ := (B.1 - A.1))

def LineThrough (p1 p2 : point) : (x : ℝ × ℝ) → Prop :=
  λ x, (x.2 - p1.2) * (p2.1 - p1.1) = (p2.2 - p1.2) * (x.1 - p1.1)

def isIntersection (l1 l2 : (x : ℝ × ℝ) → Prop) (p : point) : Prop :=
  l1 p ∧ l2 p

def isPerpendicular (A B C F : point) : Prop :=
  let AF := (F.1 - A.1, F.2 - A.2) in
  let BC := (C.1 - B.1, C.2 - B.2) in
  AF.1 * BC.1 + AF.2 * BC.2 = 0

theorem triangle_AF_perpendicular_BC 
  (A B C D E F : point)
  (h1 : ∠ BAC = 40)
  (h2 : ∠ ABC = 60)
  (h3 : ∠ CBD = 40)
  (h4 : ∠ BCE = 70)
  (h5 : LineThrough B D F)
  (h6 : LineThrough C E F) :
  isPerpendicular A B C F :=
sorry

end triangle_AF_perpendicular_BC_l440_440127


namespace total_cookies_l440_440936

def mona_cookies : ℕ := 20
def jasmine_cookies : ℕ := mona_cookies - 5
def rachel_cookies : ℕ := jasmine_cookies + 10

theorem total_cookies : mona_cookies + jasmine_cookies + rachel_cookies = 60 := 
by
  have h1 : jasmine_cookies = 15 := by sorry
  have h2 : rachel_cookies = 25 := by sorry
  have h3 : mona_cookies = 20 := by sorry
  sorry

end total_cookies_l440_440936


namespace equation_of_line_AB_l440_440066

theorem equation_of_line_AB :
  ∀ (x y : ℝ), 
    (∃ (x1 y1 x2 y2: ℝ),
      (y1^2 / 9 + x1^2 = 1) ∧
      (y2^2 / 9 + x2^2 = 1) ∧
      (x1 + x2 = 1) ∧
      (y1 + y2 = 1) ∧
      (x = 1/2) ∧
      (y = 1/2) ∧
      (y - 1/2 = -9 * (x - 1/2))
    ) → 9 * x + y - 5 = 0 :=
by
  intros x y h
  cases h with x1 hx
  cases hx with y1 hx
  cases hx with x2 hx
  cases hx with y2 hx
  cases hx with h1 h1
  cases h1 with h2 h1
  cases h1 with h3 h1
  cases h1 with h4 h1
  cases h1 with hx hy
  cases h1 with h5 h6
  rw [←h6.symm] at hy
  -- proof steps go here if needed
  sorry

end equation_of_line_AB_l440_440066


namespace total_bill_l440_440965

theorem total_bill (total_people : ℕ) (children : ℕ) (adult_cost : ℕ) (child_cost : ℕ)
  (h : total_people = 201) (hc : children = 161) (ha : adult_cost = 8) (hc_cost : child_cost = 4) :
  (201 - 161) * 8 + 161 * 4 = 964 :=
by
  rw [←h, ←hc, ←ha, ←hc_cost]
  sorry

end total_bill_l440_440965


namespace arc_length_greater_than_diameter_l440_440957

theorem arc_length_greater_than_diameter {C : Circle} {A B : Point C} :
  (arc_bisects_area C A B) → (arc_length C A B > diameter C) :=
sorry

end arc_length_greater_than_diameter_l440_440957


namespace function_is_linear_l440_440813

noncomputable def f : ℕ → ℕ :=
  λ n => n + 1

axiom f_at_0 : f 0 = 1
axiom f_at_2016 : f 2016 = 2017
axiom f_equation : ∀ n : ℕ, f (f n) + f n = 2 * n + 3

theorem function_is_linear : ∀ n : ℕ, f n = n + 1 :=
by
  intro n
  sorry

end function_is_linear_l440_440813


namespace dice_probability_exactly_four_ones_l440_440259

noncomputable def dice_probability : ℚ := 
  (Nat.choose 12 4) * (1/6)^4 * (5/6)^8

theorem dice_probability_exactly_four_ones : (dice_probability : ℚ) ≈ 0.089 :=
  by sorry -- Skip the proof. 

#eval (dice_probability : ℚ)

end dice_probability_exactly_four_ones_l440_440259


namespace circumscribed_circle_center_location_l440_440113

structure Trapezoid where
  is_isosceles : Bool
  angle_base : ℝ
  angle_between_diagonals : ℝ

theorem circumscribed_circle_center_location (T : Trapezoid)
  (h1 : T.is_isosceles = true)
  (h2 : T.angle_base = 50)
  (h3 : T.angle_between_diagonals = 40) :
  ∃ loc : String, loc = "Outside" := by
  sorry

end circumscribed_circle_center_location_l440_440113


namespace find_b_exists_l440_440743

theorem find_b_exists (N : ℕ) (hN : N ≠ 1) : ∃ (a c d : ℕ), a > 1 ∧ c > 1 ∧ d > 1 ∧
  (N : ℝ) ^ (1/a + 1/(a*4) + 1/(a*4*c) + 1/(a*4*c*d)) = (N : ℝ) ^ (37/48) :=
by
  sorry

end find_b_exists_l440_440743


namespace fish_to_honey_l440_440878

variables (fish loaf honey : ℝ)

noncomputable def equivalent_honey_per_fish (f l h : ℝ) : ℝ :=
  f = 9 / 4 * h

theorem fish_to_honey : (4 * fish = 3 * loaf) → (loaf = 3 * honey) → (fish = 9 / 4 * honey) :=
by
  intros h1 h2
  subst h2
  simp at h1
  field_simp [h1]
  sorry

end fish_to_honey_l440_440878


namespace choir_singers_l440_440683
open Nat

theorem choir_singers (initial_robes : ℕ) (robe_cost : ℕ) (total_spent : ℕ) (h0 : initial_robes = 12) (h1 : robe_cost = 2) (h2 : total_spent = 36) : initial_robes + (total_spent / robe_cost) = 30 := 
by 
  rw [h0, h1, h2]
  norm_num
  exact rfl

end choir_singers_l440_440683


namespace solution_existence_l440_440490

noncomputable def solve_problem : Prop :=
  ∃ (a b : ℝ),
  (∀ x : ℝ, x = 1 → 2 = a + b / (x + 1)) ∧
  (∀ x : ℝ, x = 3 → 3 = a + b / (x + 1)) ∧
  (a + b = 0)

theorem solution_existence : solve_problem :=
begin
  -- The steps of the proof will go here
  sorry
end

end solution_existence_l440_440490


namespace relationship_among_a_b_c_l440_440058

variable (f : ℝ → ℝ)

-- Conditions
axiom even_function : ∀ x, f x = f (-x)
axiom increasing_on_neg_infinity_to_zero : ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

def a : ℝ := f (Real.log 7 / Real.log 4)
def b : ℝ := f (Real.log 3 / Real.log (1/2))
def c : ℝ := f (Real.exp (0.6 * Real.log 0.2))

-- Theorem to prove
theorem relationship_among_a_b_c : b < a ∧ a < c :=
sorry

end relationship_among_a_b_c_l440_440058


namespace miles_for_second_package_l440_440163

theorem miles_for_second_package (x : ℝ) (h1 : 10 + x + x / 2 = total_miles)
  (h2 : 2 * total_miles = 104) : x = 28 :=
by
  let total_miles := 10 + x + x / 2
  have h := eq.symm h1
  rw [←h] at h2
  exact_not_solved_proof

end miles_for_second_package_l440_440163


namespace num_paint_ways_l440_440093

-- Define the set of colors
inductive Color
| red : Color
| green : Color
| blue : Color
| yellow : Color

-- Function to get the proper divisors of a number
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).tail.filter (λ x => x > 1 ∧ n % x = 0)

-- Define our main theorem
theorem num_paint_ways :
  let primes := [2, 3, 5, 7]
  let has_no_red (c : ℕ → Color) := ∀ n, n ∈ primes → c n ≠ Color.red
  let unique_divisor_color (c : ℕ → Color) :=
    ∀ n, 2 ≤ n ∧ n ≤ 10 → ∀ d ∈ (proper_divisors n), c n ≠ c d
  ∃ (c : ℕ → Color), has_no_red c ∧ unique_divisor_color c ∧
    (Finset.univ.card {c : ℕ → Color // has_no_red c ∧ unique_divisor_color c} = 5832) :=
sorry

end num_paint_ways_l440_440093


namespace range_independent_variable_l440_440216

def domain_of_function (x : ℝ) : Prop :=
  x ≥ -1 ∧ x ≠ 0

theorem range_independent_variable (x : ℝ) :
  domain_of_function x ↔ x ≥ -1 ∧ x ≠ 0 :=
by
  sorry

end range_independent_variable_l440_440216


namespace max_S_over_R_squared_l440_440874

theorem max_S_over_R_squared (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let S := 2 * (a * b + b * c + c * a)
  let R := (sqrt (a^2 + b^2 + c^2)) / 2
  (S / R^2) ≤ (2 / 3) * (3 + sqrt 3) :=
by sorry

end max_S_over_R_squared_l440_440874


namespace construct_cyclic_quadrilateral_l440_440718

/-- Constructing a cyclic quadrilateral from given sides and angles -/
theorem construct_cyclic_quadrilateral
  (A B C D : Point)
  (AB CD : ℝ) 
  (α1 γ1 : ℝ) 
  (h1 : α1 + γ1 < 180) 
  (h2 : dist A B = AB) 
  (h3 : dist C D = CD) 
  (h4 : angle BAC = α1) 
  (h5 : angle DCA = γ1) : 
  ∃ (quadrilateral : CyclicQuadrilateral), 
    quadrilateral.A = A ∧ quadrilateral.B = B ∧ quadrilateral.C = C ∧ quadrilateral.D = D :=
begin
  sorry
end

end construct_cyclic_quadrilateral_l440_440718


namespace smallest_number_among_four_l440_440350

theorem smallest_number_among_four :
  let A := -2
  let B := abs (-2)
  let C := -(-1)
  let D := - (1/2)
  min {A, B, C, D} = A := by
  sorry

end smallest_number_among_four_l440_440350


namespace total_coins_l440_440170

theorem total_coins (x : ℕ) (hx : 2 * x^3 - 27 * x^2 + x = 0)
  (hx_int : x = 13) :
  let paul_coins := x^2,
      pete_coins := 5 * x^2,
      total_coins := paul_coins + pete_coins in
  total_coins = 1014 :=
by
  have h : x = 13 := hx_int
  sorry

end total_coins_l440_440170


namespace grasshoppers_jump_left_l440_440165

theorem grasshoppers_jump_left (n : ℕ) (h : n = 2019) :
  (∃ p : fin n → ℕ, ∀ i, ∃ j, |p j - p i| = 1) → 
  (∃ q : fin n → ℕ, ∀ i, ∃ j, |q j - q i| = 1) := 
by
  intro h1
  sorry

end grasshoppers_jump_left_l440_440165


namespace find_orange_juice_amount_l440_440396

variable (s y t oj : ℝ)

theorem find_orange_juice_amount (h1 : s = 0.2) (h2 : y = 0.1) (h3 : t = 0.5) (h4 : oj = t - (s + y)) : oj = 0.2 :=
by
  sorry

end find_orange_juice_amount_l440_440396


namespace back_wheel_revolutions_l440_440948

-- Defining the problem with given conditions
def front_wheel_radius : ℝ := 3 -- in feet
def back_wheel_radius : ℝ := 3 / 12 -- 3 inches converted to feet
def front_wheel_revolutions : ℕ := 150

-- Proof problem statement
theorem back_wheel_revolutions : 
  let front_circumference := 2 * Real.pi * front_wheel_radius,
      total_distance := front_circumference * front_wheel_revolutions,
      back_circumference := 2 * Real.pi * back_wheel_radius in
  total_distance / back_circumference = 1800 :=
by
  sorry

end back_wheel_revolutions_l440_440948


namespace dice_sum_24_probability_l440_440766

noncomputable def probability_sum_24 : ℚ :=
  let prob_single_six := (1 : ℚ) / 6 in
  prob_single_six ^ 4

theorem dice_sum_24_probability :
  probability_sum_24 = 1 / 1296 :=
by
  sorry

end dice_sum_24_probability_l440_440766


namespace solve_for_x_l440_440549

variables (a b c x : ℝ)

def f (x : ℝ) : ℝ := 1 / (a * x^2 + b * x + c)

noncomputable def f_inv (x : ℝ) : ℝ := sorry  -- Assuming some definition for f_inv

theorem solve_for_x (h₀ : a ≠ 0) (h₁ : b^2 - 4 * a * c > 0) : 
  f_inv x = 0 ↔ x = 1 / c :=
sorry

end solve_for_x_l440_440549


namespace lattice_point_exists_l440_440120

theorem lattice_point_exists 
  (P : C) (k : ℕ) (C : set (ℝ × ℝ))
  (unit_circle : ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + y^2 = 1)
  (ray_intersection : ∀ (Q : ℝ × ℝ), Q ≠ (0, 0) → ∃ Q' ∈ C, collinear (0, 0) Q Q') : 
  ∃ (Q : ℝ × ℝ), (|Q.1| = k ∨ |Q.2| = k) ∧ distance P Q' < 1 / (2 * k) :=
by
  sorry

end lattice_point_exists_l440_440120


namespace tom_free_lessons_l440_440614

theorem tom_free_lessons (total_lessons : ℕ) (cost_per_lesson : ℕ) (amount_paid : ℕ) :
    total_lessons = 10 →
    cost_per_lesson = 10 →
    amount_paid = 80 →
    total_lessons - amount_paid / cost_per_lesson = 2 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end tom_free_lessons_l440_440614


namespace find_person_speed_l440_440327

def convertMetersToKilometers (d_meters : ℕ) : ℝ := d_meters / 1000

def convertMinutesToHours (t_minutes : ℕ) : ℝ := t_minutes / 60

def calculateSpeed (distance_km : ℝ) (time_hours : ℝ) : ℝ := distance_km / time_hours

theorem find_person_speed :
  let d_meters := 1080
  let t_minutes := 12
  let d_kilometers := convertMetersToKilometers d_meters
  let t_hours := convertMinutesToHours t_minutes
  calculateSpeed d_kilometers t_hours = 5.4 := by
  sorry

end find_person_speed_l440_440327


namespace range_of_eccentricity_l440_440676

variables {a b c : ℝ} (h1 : a > 0)
def is_hyperbola (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (5 - a^2) = 1

theorem range_of_eccentricity 
  (h2 : 2 < b / a) (h3 : b / a < 3)
  (h4 : c = √(a^2 + b^2)) 
  (h5 : a > 0 ∧ ∀ x y : ℝ, is_hyperbola a x y → line_slope x y 2 ∧ line_slope x y 3) :
  (√5 < c / a) ∧ (c / a < √10) := 
sorry

end range_of_eccentricity_l440_440676


namespace number_of_siblings_l440_440656

-- Define the variables B and G
variables (B G : ℕ)

-- Condition that there is 1 boy
axiom B_def : B = 1

-- Condition that each sister has 1 brother and 1 sister
axiom G_sisters : G - 1 = 1

-- The theorem stating the total number of siblings is 3
theorem number_of_siblings : B + G = 3 := by
  rw [B_def, G_sisters]
  sorry

end number_of_siblings_l440_440656


namespace intersection_of_M_and_N_l440_440803

def M : Set ℤ := {0, 1}
def N : Set ℤ := {-1, 0}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end intersection_of_M_and_N_l440_440803


namespace angle_ABC_is_90_degrees_l440_440887

variable {α : Type}
variable [LinearOrderedField α]

/-- Given:
1. Triangle ADE is produced from triangle ABC by a rotation of 90 degrees about point A.
2. Angle D is 60 degrees.
3. Angle E is 40 degrees.
Prove: 
  Angle ABC is 90 degrees. -/
theorem angle_ABC_is_90_degrees 
  (triangle_ABC : Triangle α)
  (A B C D E : α)
  (is_rotation_90 : is_rotation_90_degrees_about_point triangle_ABC A D E)
  (angle_D : angle D = 60)
  (angle_E : angle E = 40) :
  angle ABC = 90 := sorry

end angle_ABC_is_90_degrees_l440_440887


namespace probability_exactly_four_ones_is_090_l440_440240
open Float (approxEq)

def dice_probability_exactly_four_ones : Float :=
  let n := 12
  let k := 4
  let p_one := (1 / 6 : Float)
  let p_not_one := (5 / 6 : Float)
  let combination := ((n.factorial) / (k.factorial * (n - k).factorial) : Float)
  let probability := combination * (p_one ^ k) * (p_not_one ^ (n - k))
  probability

theorem probability_exactly_four_ones_is_090 : dice_probability_exactly_four_ones ≈ 0.090 :=
  sorry

end probability_exactly_four_ones_is_090_l440_440240


namespace battleship_min_cells_l440_440352

theorem battleship_min_cells (n : ℕ) (H1 : n > 0) : 
  ∀ (board : fin (2 * n) → fin (2 * n) → Prop) 
  (row_indices : fin (2 * n) → Prop) 
  (col_indices : fin (2 * n) → Prop), 
  (∀ i j, row_indices i ∧ col_indices j → board i j) 
  → (∃ cells : fin (2 * n) → fin (2 * n) → Prop, 
       ∀ i j, cells i j → board i j) 
  → ∃ (min_cells : ℕ), min_cells = 3 * n + 1 := sorry

end battleship_min_cells_l440_440352


namespace find_x_l440_440496

def myOperation (x y : ℝ) : ℝ := 2 * x * y

theorem find_x (x : ℝ) (h : myOperation 9 (myOperation 4 x) = 720) : x = 5 :=
by
  sorry

end find_x_l440_440496


namespace number_of_permutations_of_three_balls_l440_440390

theorem number_of_permutations_of_three_balls : 
  (finset.univ : finset (finset {3, 6, 9})).card = 6 :=
by
  sorry

end number_of_permutations_of_three_balls_l440_440390


namespace necessary_condition_l440_440434

theorem necessary_condition (m : ℝ) : 
  (∀ x > 0, (x / 2) + (1 / (2 * x)) - (3 / 2) > m) → (m ≤ -1 / 2) :=
by
  -- Proof omitted
  sorry

end necessary_condition_l440_440434


namespace number_of_integers_satisfying_inequality_l440_440747

theorem number_of_integers_satisfying_inequality :
  ∃ (n_count : ℕ), 
    (∀ n : ℤ, (sqrt ((2 : ℤ) * n + 1) ≤ sqrt ((6 : ℤ) * n - 8) ∧ sqrt ((6 : ℤ) * n - 8) < sqrt ((3 : ℤ) * n + 7)) ↔ 
      n ≥ 3 ∧ n < 5) → 
    n_count = 2 :=
by
  sorry

end number_of_integers_satisfying_inequality_l440_440747


namespace compute_complex_expression_l440_440149

def x : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 9)

-- Conditions
lemma x_ninth_power_eq_one : x^9 = 1 := by
  sorry

lemma x_sum_eq_zero : x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 0 := by
  sorry

-- Main theorem
theorem compute_complex_expression : (3 * x + x^3) * (3 * x^3 + x^9) * (3 * x^6 + x^18) = 369 := by
  sorry

end compute_complex_expression_l440_440149


namespace solve_inequalities_l440_440721

theorem solve_inequalities (x : ℝ) :
    ((x / 2 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x))) ↔ (-6 ≤ x ∧ x < -3 / 2) :=
by
  sorry

end solve_inequalities_l440_440721


namespace good_pairs_count_l440_440377

def line1 : ℝ → ℝ := λ x => 4 * x + 2
def line2 : ℝ → ℝ := λ x => 2 * x + 1  -- simplified 3y = 6x + 3
def line3 : ℝ → ℝ := λ x => 4 * x - (1 / 2)  -- simplified 2y = 8x - 1
def line4 : ℝ → ℝ := λ x => (2 / 3) * x + 2  -- simplified 3y = 2x + 6
def line5 : ℝ → ℝ := λ x => 4 * x - 2  -- simplified 5y = 20x - 10

/-
We know that:
- Line1, Line3, and Line5 are parallel.
- No lines are perpendicular.

Task:
Prove that the number of pairs of lines that are either parallel or perpendicular is 3.
-/
theorem good_pairs_count : 
  let SlopeParallel (m1 m2 : ℝ) := m1 = m2
  let SlopePerpendicular (m1 m2 : ℝ) := m1 * m2 = -1
  -- Slopes from the line definitions above
  let slopes := [4 : ℝ, 2, 4, (2 / 3), 4]
  let slope_pairs := slopes.pairs
  let good_pairs := slope_pairs.filter (λ (m1, m2) => SlopeParallel m1 m2 ∨ SlopePerpendicular m1 m2)
  good_pairs.length = 3 :=
by {
  sorry
}

end good_pairs_count_l440_440377


namespace win_sector_area_l440_440309

/-- Given a circular spinner with a radius of 8 cm and the probability of winning being 3/8,
    prove that the area of the WIN sector is 24π square centimeters. -/
theorem win_sector_area (r : ℝ) (P_win : ℝ) (area_WIN : ℝ) :
  r = 8 → P_win = 3 / 8 → area_WIN = 24 * Real.pi := by
sorry

end win_sector_area_l440_440309


namespace lambda_plus_mu_range_l440_440151

variables {O A B C : Type*} [normed_add_comm_group O] [normed_space ℝ O]
variables (OA OB OC : O) (λ μ : ℝ)

-- Conditions
def distinct_points_on_circle (O A B C : O) : Prop :=
  ∥O∥ = 1 ∧ ∥A∥ = 1 ∧ ∥B∥ = 1 ∧ ∥C∥ = 1 ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A

def angle_AOB_eq_120 (O A B : O) : Prop :=
  ∃ v w : ℝ, v ≠ 0 ∧ w ≠ 0 ∧ 
  ∡ (A - O) (B - O) = 2*pi/3 -- 120 degrees in radians

def C_on_minor_arc_AB (O A B C : O) : Prop :=
  ∥O - C∥ = 1 ∧ ∃ θ : ℝ, 0 < θ ∧ θ < 2*pi/3 ∧ 
  (C - O) = ∥C - O∥ • (cos θ • (A - O) + sin θ • (B - O)) 

def OC_as_linear_combination (O A B C : O) (λ μ : ℝ) : Prop :=
  UC = λ • U A + μ • U B

-- Proof statement to be proven
theorem lambda_plus_mu_range (O A B C : O) (λ μ : ℝ) 
  (h_dist_pts : distinct_points_on_circle O A B C)
  (h_angle : angle_AOB_eq_120 O A B)
  (h_minor_arc : C_on_minor_arc_AB O A B C)
  (h_lin_comb : OC_as_linear_combination O A B C λ μ) :
  1 < λ + μ ∧ λ + μ ≤ 2 :=
sorry

end lambda_plus_mu_range_l440_440151


namespace iterative_average_difference_l440_440355

theorem iterative_average_difference :
  let seq := [1, 2, 3, 4, 5, 6]
  in
  let iterative_average (seq : List ℚ) :=
    seq.foldl (fun acc x => (acc + x) / 2) 0
  in
  let max_avg :=
    iterative_average [6, 5, 4, 3, 2, 1]
  in
  let min_avg :=
    iterative_average [1, 2, 3, 4, 5, 6]
  in
  max_avg - min_avg = 37 / 16 :=
sorry

end iterative_average_difference_l440_440355


namespace jeff_average_number_of_skips_l440_440960

noncomputable def sam_skips : ℕ := 16

noncomputable def jeff_first_round_skips : ℕ := sam_skips - 1
noncomputable def jeff_second_round_skips : ℕ := sam_skips - 3
noncomputable def jeff_third_round_skips : ℕ := sam_skips + 4
noncomputable def jeff_fourth_round_skips : ℕ := sam_skips / 2

noncomputable def jeff_average_four_rounds : ℝ := (jeff_first_round_skips + jeff_second_round_skips + jeff_third_round_skips + jeff_fourth_round_skips) / 4

noncomputable def jeff_fifth_round_skips : ℕ := 2 * Real.sqrt jeff_average_four_rounds

noncomputable def jeff_total_skips : ℕ := jeff_first_round_skips + jeff_second_round_skips + jeff_third_round_skips + jeff_fourth_round_skips + jeff_fifth_round_skips

noncomputable def jeff_average_skips_per_round : ℝ := jeff_total_skips / 5

theorem jeff_average_number_of_skips : jeff_average_skips_per_round = 12.6 := sorry

end jeff_average_number_of_skips_l440_440960


namespace proof_t_base_c_l440_440557

theorem proof_t_base_c :
  ∃ c : ℕ, (c + 2) * (c + 4) * (c + 8) = 5 * c ^ 3 + 3 * c ^ 2 + 2 * c ∧
           let t := (c + 2) + (c + 4) + (c + 8) + (c + 10) in
           t = 4 * c + 24 ∧ c = 4 ∧ t = 40 :=
by
  -- Explicitly stating the conditions and conclusion
  sorry

end proof_t_base_c_l440_440557


namespace math_problem_l440_440556

-- Conditions
variables {f g : ℝ → ℝ}
axiom f_zero : f 0 = 0
axiom inequality : ∀ x y : ℝ, g (x - y) ≥ f x * f y + g x * g y

-- Problem Statement
theorem math_problem : ∀ x : ℝ, f x ^ 2008 + g x ^ 2008 ≤ 1 :=
by
  sorry

end math_problem_l440_440556


namespace transformed_mean_and_variance_l440_440975

theorem transformed_mean_and_variance (mean_original var_original : ℝ)
  (h1 : mean_original = 2.8)
  (h2 : var_original = 3.6) :
  let mean_new := 2 * mean_original + 60,
      var_new := (2 : ℝ) ^ 2 * var_original
  in mean_new = 65.6 ∧ var_new = 14.4 := 
by
  -- We skip the proof as per instructions
  sorry

end transformed_mean_and_variance_l440_440975


namespace geometric_mean_of_4_and_9_l440_440203

theorem geometric_mean_of_4_and_9 :
  ∃ b : ℝ, (4 * 9 = b^2) ∧ (b = 6 ∨ b = -6) :=
by
  sorry

end geometric_mean_of_4_and_9_l440_440203


namespace garden_roller_diameter_l440_440315

theorem garden_roller_diameter
  (length_roller : ℝ)
  (total_area : ℝ)
  (num_revolutions : ℝ)
  (pi_approx : ℝ)
  (h_length_roller : length_roller = 2)
  (h_total_area : total_area = 35.2)
  (h_num_revolutions : num_revolutions = 4)
  (h_pi_approx : pi_approx = 22 / 7) :
  let area_per_revolution := total_area / num_revolutions,
      circumference := area_per_revolution / length_roller
  in circumference / pi_approx = 1.4 :=
by
  sorry

end garden_roller_diameter_l440_440315


namespace prob_sum_24_four_dice_l440_440755

section
open ProbabilityTheory

/-- Define the event E24 as the event where the sum of numbers on the top faces of four six-sided dice is 24 -/
def E24 : Event (StdGen) :=
eventOfFun {ω | ∑ i in range 4, (ω.gen_uniform int (6-1)) + 1 = 24}

/-- Probability that the sum of the numbers on top faces of four six-sided dice is 24 is 1/1296. -/
theorem prob_sum_24_four_dice : ⋆{ P(E24) = 1/1296 } := sorry

end

end prob_sum_24_four_dice_l440_440755


namespace sum_of_coefficients_l440_440419

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 : ℤ)
  (h : (1 - 2 * X)^5 = a + a1 * X + a2 * X^2 + a3 * X^3 + a4 * X^4 + a5 * X^5) :
  a1 + a2 + a3 + a4 + a5 = -2 :=
by {
  -- the proof steps would go here
  sorry
}

end sum_of_coefficients_l440_440419


namespace dani_pants_after_5_years_l440_440729

/--
Dani gets 8 pants per year. He initially had 50 pants. Calculate the number of pants he'll have in 5 years.
-/
theorem dani_pants_after_5_years (initial_pants : ℕ) (pants_per_year : ℕ) (years : ℕ) : 
    initial_pants = 50 → pants_per_year = 8 → years = 5 → initial_pants + (pants_per_year * years) = 90 :=
by
    intros h1 h2 h3
    rw [h1, h2, h3]
    exact rfl

end dani_pants_after_5_years_l440_440729


namespace positional_relationship_a_b_l440_440476

noncomputable def parallel_positional_relationship (α β a b : Type) [has_inter α β b] [parallel a α] [parallel a β] (h1 : α ∩ β = b) : Prop :=
parallel a b

-- The theorem assuming the conditions and proving the conclusion
theorem positional_relationship_a_b {α β a b : Type} [has_inter α β b] [parallel a α] [parallel a β] (h1 : α ∩ β = b) : parallel a b := 
sorry

end positional_relationship_a_b_l440_440476


namespace ratio_jake_to_clementine_l440_440369

-- Definitions based on conditions
def ClementineCookies : Nat := 72
def ToryCookies (J : Nat) : Nat := (J + ClementineCookies) / 2
def TotalCookies (J : Nat) : Nat := ClementineCookies + J + ToryCookies J
def TotalRevenue : Nat := 648
def CookiePrice : Nat := 2
def TotalCookiesSold : Nat := TotalRevenue / CookiePrice

-- The main proof statement
theorem ratio_jake_to_clementine : 
  ∃ J : Nat, TotalCookies J = TotalCookiesSold ∧ J / ClementineCookies = 2 :=
by
  sorry

end ratio_jake_to_clementine_l440_440369


namespace ratio_segments_equal_l440_440436

-- Definitions of given problem's conditions
variables {A B C O D E : Type*}
variable [EuclideanGeometry A B C O D E]
variables [Circumcenter O A B C] [Geometry.Distinct A B] [Geometry.Distinct B C] [Geometry.Distinct C A]
variable (angle_BAC : Geometry.Angle A B C = 60)

-- Prove the final solution
theorem ratio_segments_equal (circumcenter_O : Geometry.Circumcenter O A B C)
  (angle_BAC_is_60 : Geometry.Angle A B C = 60)
  (point_D : Geometry.PointOnLineExtension C O A B D)
  (point_E : Geometry.PointOnLineExtension B O A C E) :
  Geometry.SegmentRatio B D C E = 1 := 
sorry

end ratio_segments_equal_l440_440436


namespace alice_savings_l440_440700

theorem alice_savings :
  let num_notebooks := 8
  let original_price_per_notebook := 3.75
  let discount_rate := 0.25
  let original_total_price := num_notebooks * original_price_per_notebook
  let discount_per_notebook := original_price_per_notebook * discount_rate
  let discounted_price_per_notebook := original_price_per_notebook - discount_per_notebook
  let discounted_total_price := num_notebooks * discounted_price_per_notebook
  in original_total_price - discounted_total_price = 7.50 :=
by {
  let num_notebooks := 8
  let original_price_per_notebook := 3.75
  let discount_rate := 0.25
  let original_total_price := num_notebooks * original_price_per_notebook
  
  have h1 : discount_per_notebook = original_price_per_notebook * discount_rate, by {
    exact rfl,
  }
  let discount_per_notebook := original_price_per_notebook * discount_rate
  
  have h2 : discounted_price_per_notebook = original_price_per_notebook - discount_per_notebook, by {
    exact rfl,
  }
  let discounted_price_per_notebook := original_price_per_notebook - discount_per_notebook
  
  have h3 : discounted_total_price = num_notebooks * discounted_price_per_notebook, by {
    exact rfl,
  }
  let discounted_total_price := num_notebooks * discounted_price_per_notebook
  
  have h4 : original_total_price - discounted_total_price = 7.50, by {
    calc
      original_total_price - discounted_total_price
          = 8 * 3.75 - 8 * (3.75 - (3.75 * 0.25)) : by {
            exact rfl,
          }
      ... = 30.00 - 22.5 : by {
            exact rfl,
          }
      ... = 7.5 : by {
            exact rfl,
          } 

  }
  exact h4,
}

end alice_savings_l440_440700


namespace all_good_rational_are_integers_l440_440681

def good_rational (x : ℚ) (α : ℝ) (N : ℕ) : Prop :=
  x > 1 ∧ ∀ n : ℕ, n ≥ N → |(x^n - (x^n).floor) - α| ≤ 1 / (2 * (x.num.natAbs + x.denom.natAbs))

theorem all_good_rational_are_integers (x : ℚ) :
  (∃ α : ℝ, ∃ N : ℕ, good_rational x α N) → ∃ k : ℤ, x = k ∧ k > 1 :=
by
  sorry

end all_good_rational_are_integers_l440_440681


namespace wickets_before_last_match_l440_440282

-- Define the conditions
variable (W : ℕ)

-- Initial average
def initial_avg : ℝ := 12.4

-- Runs given in the last match
def runs_last_match : ℝ := 26

-- Wickets taken in the last match
def wickets_last_match : ℕ := 4

-- The new average after the last match
def new_avg : ℝ := initial_avg - 0.4

-- Prove the theorem
theorem wickets_before_last_match :
  (12.4 * W + runs_last_match) / (W + wickets_last_match) = new_avg → W = 55 :=
by
  sorry

end wickets_before_last_match_l440_440282


namespace max_profit_at_35_l440_440307

-- Define the conditions
def unit_purchase_price : ℝ := 20
def base_selling_price : ℝ := 30
def base_sales_volume : ℕ := 400
def price_increase_effect : ℝ := 1
def sales_volume_decrease_per_dollar : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - unit_purchase_price) * (base_sales_volume - sales_volume_decrease_per_dollar * (x - base_selling_price))

-- Lean statement to prove that the selling price which maximizes the profit is 35
theorem max_profit_at_35 : ∃ x : ℝ, x = 35 ∧ ∀ y : ℝ, profit y ≤ profit 35 := 
  sorry

end max_profit_at_35_l440_440307


namespace hourly_wage_decrease_l440_440325

variable (W H W' : ℝ)

-- Conditions
def original_income := W * H
def increased_hours := H' = 1.25 * H
def income_equation := W * H = W' * 1.25 * H

theorem hourly_wage_decrease :
  (increased_hours) → (income_equation) → W' = 0.8 * W :=
by
  intros increased_hours income_equation
  sorry

end hourly_wage_decrease_l440_440325


namespace max_ab_value_1_half_l440_440863

theorem max_ab_value_1_half 
  (a b : ℝ) 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_eq : a + 2 * b = 1) :
  a = 1 / 2 → ab = 1 / 8 :=
sorry

end max_ab_value_1_half_l440_440863


namespace arithmetic_sequence_a14_l440_440428

theorem arithmetic_sequence_a14 (a : ℕ → ℤ) (h1 : a 4 = 5) (h2 : a 9 = 17) (h3 : 2 * a 9 = a 14 + a 4) : a 14 = 29 := sorry

end arithmetic_sequence_a14_l440_440428


namespace find_price_before_tax_and_tip_l440_440643

-- Define the conditions as variables and proofs
variables 
(total_spent : ℚ) (tip_percent : ℚ) (tax_percent : ℚ)
(h1 : total_spent = 211.20) (h2 : tip_percent = 0.20) (h3 : tax_percent = 0.10)

-- Define the actual price of the food (P) and the known final amount
def price_with_tax (P : ℚ) : ℚ := P * (1 + tax_percent)
def total_price (P : ℚ) : ℚ := price_with_tax P * (1 + tip_percent)

-- The statement to be proven
theorem find_price_before_tax_and_tip : 
  ∃ P : ℚ, total_price P = total_spent ∧ P = 160 := by
  sorry

end find_price_before_tax_and_tip_l440_440643


namespace part1_part2_l440_440466

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^x + a^(-x)

-- Proof Problem for Part 1
theorem part1 (a : ℝ) (h1 : a > 1) (h2 : f 1 a = 10 / 3) : a = 3 := 
sorry

-- Definitions for Part 2
noncomputable def log_eq_x (m : ℝ) (x : ℝ) : ℝ := log 3 (m * 3^x - m) / (3^x + 3^(-x))

-- Proof Problem for Part 2
theorem part2 (m : ℝ) (a : ℝ) (h3 : ∀ x : ℝ, log_eq_x m x = 3^x) : m ∈ set.Ioo (-∞) (-1) ∪ {2 + 2 * real.sqrt 2} :=  
sorry

end part1_part2_l440_440466


namespace total_cost_of_repair_l440_440138

noncomputable def cost_of_repair (tire_cost: ℝ) (num_tires: ℕ) (tax: ℝ) (city_fee: ℝ) (discount: ℝ) : ℝ :=
  let total_cost := (tire_cost * num_tires : ℝ)
  let total_tax := (tax * num_tires : ℝ)
  let total_city_fee := (city_fee * num_tires : ℝ)
  (total_cost + total_tax + total_city_fee - discount)

def car_A_tire_cost : ℝ := 7
def car_A_num_tires : ℕ := 3
def car_A_tax : ℝ := 0.5
def car_A_city_fee : ℝ := 2.5
def car_A_discount : ℝ := (car_A_tire_cost * car_A_num_tires) * 0.05

def car_B_tire_cost : ℝ := 8.5
def car_B_num_tires : ℕ := 2
def car_B_tax : ℝ := 0 -- no sales tax
def car_B_city_fee : ℝ := 2.5
def car_B_discount : ℝ := 0 -- expired coupon

theorem total_cost_of_repair : 
  cost_of_repair car_A_tire_cost car_A_num_tires car_A_tax car_A_city_fee car_A_discount + 
  cost_of_repair car_B_tire_cost car_B_num_tires car_B_tax car_B_city_fee car_B_discount = 50.95 :=
by
  sorry

end total_cost_of_repair_l440_440138


namespace probability_X_greater_than_4_l440_440131

noncomputable def X : ℝ → ℝ := sorry

axiom normal_distribution : X ~ Normal 3 1
axiom probability_condition : P(2 ≤ X ≤ 4) = 0.6826

theorem probability_X_greater_than_4 : P(X > 4) = 0.1587 := by
  sorry

end probability_X_greater_than_4_l440_440131


namespace sum_15_l440_440122

variable (a : ℕ → ℚ)
variable (a1 d : ℚ)

-- Assume that the sequence is arithmetic: a_n = a1 + (n-1)d
def arith_seq (a : ℕ → ℚ) (a1 d : ℚ): Prop := 
  ∀ n, a n = a1 + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) (a : ℕ → ℚ) : ℚ :=
  ∑ i in finset.range(n + 1), a (i + 1)

-- Assuming given conditions
axiom cond1 (a : ℕ → ℚ) (a1 d : ℚ) : arith_seq a a1 d → a 2 + a 8 - a 12 = 0
axiom cond2 (a : ℕ → ℚ) (a1 d : ℚ) : arith_seq a a1 d → a 14 - a 4 = 2

-- The goal is to prove that under these conditions, s_15 = 30
theorem sum_15 (a : ℕ → ℚ) (a1 d : ℚ) :
  arith_seq a a1 d → 
  cond1 a a1 d → 
  cond2 a a1 d → 
  sum_seq 15 a = 30 :=
by
  sorry

end sum_15_l440_440122


namespace min_val_expression_l440_440951

theorem min_val_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 * b + b^2 * c + c^2 * a = 3) : 
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 :=
sorry

end min_val_expression_l440_440951


namespace average_age_after_person_leaves_l440_440581

theorem average_age_after_person_leaves
  (average_age_seven : ℕ := 28)
  (num_people_initial : ℕ := 7)
  (person_leaves : ℕ := 20) :
  (average_age_seven * num_people_initial - person_leaves) / (num_people_initial - 1) = 29 := by
  sorry

end average_age_after_person_leaves_l440_440581


namespace probability_exactly_four_1s_l440_440254

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_four_1s_in_12_dice : ℝ :=
  let n := 12
  let k := 4
  let p := 1 / 6
  let q := 5 / 6
  (binomial_coefficient n k : ℝ) * p^k * q^(n - k)

theorem probability_exactly_four_1s : probability_of_four_1s_in_12_dice ≈ 0.089 :=
  by
  sorry

end probability_exactly_four_1s_l440_440254


namespace area_of_common_region_is_one_third_l440_440686

noncomputable def common_area_of_squares (a : ℝ) : Prop :=
  let α : ℝ := real.arccos (3 / 5) in
  0 < α ∧ α < real.pi / 2 ∧ (5 / 3) = real.cos α ∧ a = 1 / 3

theorem area_of_common_region_is_one_third
  (ha : common_area_of_squares (1 / 3)) :
  ∃ a, common_area_of_squares a ∧ a = 1 / 3 :=
by sorry

end area_of_common_region_is_one_third_l440_440686


namespace correct_statement_D_l440_440280

def statement_A (P : ℝ → Prop) : Prop :=
∀ E, 0 < P E ∧ P E < 1

def statement_B (F : ℕ → ℝ) : Prop :=
∀ n m, F n = F m

def statement_C (P : ℝ → Prop) : Prop :=
∀ E, randomized (P E)

def statement_D (F : ℕ → ℝ) (P : ℝ → Prop) : Prop :=
∀ E, tendsto (λ n, F n) at_top (𝓝 (P E))

theorem correct_statement_D (F : ℕ → ℝ) (P : ℝ → Prop) :
  ∃! s, s = statement_D F P :=
by
  unfold statement_D
  sorry

end correct_statement_D_l440_440280


namespace problem_statement_l440_440535

-- Define necessary variables and conditions
variables (x₁ x₂ x₃ x₄ x₅ : ℕ)
variables (perm : list ℕ)
variables (condition : perm = [1, 2, 3, 4, 6].permutations)

-- Define the function to compute the sum for a given permutation
def sum_function (l : list ℕ) : ℕ :=
l.nth_le 0 sorry * l.nth_le 1 sorry +
l.nth_le 1 sorry * l.nth_le 2 sorry +
l.nth_le 2 sorry * l.nth_le 3 sorry +
l.nth_le 3 sorry * l.nth_le 4 sorry +
l.nth_le 4 sorry * l.nth_le 0 sorry 

-- Define M and N
def M : ℕ := max (perm.map sum_function)
def N : ℕ := length (filter (λ l, sum_function l = M) perm)

-- Statement of the theorem, without proof
theorem problem_statement : M + N = 78 := 
begin
  -- The proof here uses the conditions
  sorry
end

end problem_statement_l440_440535


namespace solve_for_x_l440_440859

theorem solve_for_x (x : ℝ) (h : sqrt (3 / x + 3) = 5 / 3) : x = -27 / 2 :=
by
  sorry

end solve_for_x_l440_440859


namespace num_solutions_abs_ineq_l440_440722

theorem num_solutions_abs_ineq (x : ℤ) : {x : ℤ | |7 * x + 2| ≤ 9}.to_finset.card = 3 := by
  sorry

end num_solutions_abs_ineq_l440_440722


namespace calculate_fraction_l440_440380

def op (x y : ℝ) : ℝ := x * y - 2 * y^2
def odot (x y : ℝ) : ℝ := Real.sqrt x + y - x * y^2

theorem calculate_fraction :
  let x := 9
  let y := 3
  (op x y) / (odot x y) = -3 / 25 :=
by
  -- Definitions and values: these are from the problem conditions and can appear here
  let x := 9
  let y := 3
  -- Definitions op and odot are derived directly from the problem statement above
  have h1 : x * y - 2 * y^2 = 9 := by sorry
  have h2 : Real.sqrt x + y - x * y^2 = -75 := by sorry
  -- This uses h1 and h2 to prove the final goal
  sorry

end calculate_fraction_l440_440380


namespace inequality_smallest_val_l440_440417

open Real

def cot (x : ℝ) := cos x / sin x
def tan (x : ℝ) := sin x / cos x

theorem inequality_smallest_val (a : ℝ) (h : a = -2.52) :
  (∀ x ∈ set.Ioo (-3 * π / 2) (-π), 
  (∛(cot x ^ 2) - ∛(tan x ^ 2)) / (∛(sin x ^ 2) - ∛(cos x ^ 2)) < a) :=
begin
  sorry
end

end inequality_smallest_val_l440_440417


namespace shaded_triangle_equilateral_l440_440263

theorem shaded_triangle_equilateral
  (A B C D O K L : Type)
  (ABO : Triangle A B O)
  (AB'_O : Triangle A B' O)
  (AKO DOL : Triangle A K O)
  (BDL : Triangle B D L)
  (cong_ABO_AB'O : ABO ≡ AB'_O)
  (parallel_AD_BC : A D ∥ B C) :
  Equilateral K O L := 
sorry

end shaded_triangle_equilateral_l440_440263


namespace angle_A_measure_max_triangle_area_l440_440870

variable (a b c : ℝ)
variable (A B C : ℝ)

noncomputable def measureAngle (b c a : ℝ) : ℝ := 
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))

def triangleArea (a b c : ℝ) (A : ℝ) : ℝ := 
  0.5 * b * c * Real.sin A

theorem angle_A_measure (h1 : b^2 + c^2 - a^2 + b * c = 0) : 
  measureAngle b c a = (2 * Real.pi) / 3 := by
  sorry

theorem max_triangle_area (h1 : b^2 + c^2 - a^2 + b * c = 0) (ha : a = Real.sqrt 3) : 
  ∃ b c, triangleArea a b c ((2 * Real.pi) / 3) = Real.sqrt 3 / 4 :=
  by
  sorry

end angle_A_measure_max_triangle_area_l440_440870


namespace cities_connectivity_l440_440990

noncomputable def cityGraphConstruction : Prop :=
  ∃ (G : SimpleGraph (Fin 2000)), 
  (∀ (k : ℕ), k ∈ Finset.range (1000 + 1) → 
    Finset.card {v : Fin 2000 | G.degree v = k} = 2)

-- Theorem statement in Lean 4
theorem cities_connectivity : cityGraphConstruction :=
sorry

end cities_connectivity_l440_440990


namespace divisibility_example_l440_440392

theorem divisibility_example : 
  ∃ (a b : ℕ), 
    (a = 532 ∧ b = 14 ∧ a % b = 0) ∨ 
    (a = 215 ∧ b = 43 ∧ a % b = 0) := 
by
  use 532, 14
  split
  { exact ⟨rfl, rfl, rfl⟩ }
  { use 215, 43
    exact ⟨rfl, rfl, rfl⟩ }

end divisibility_example_l440_440392


namespace part_I_part_II_part_III_l440_440467

def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + (2 * a - 1) * x

theorem part_I (x : ℝ) : 
  (∃a : ℝ, a = 1 ∧ (∀ x : ℝ, 0 < x ∧ x < 1 → derivative (f a x) > 0)
  ∧ (∀ x : ℝ, x > 1 → derivative (f a x) < 0)) := sorry

theorem part_II (a x : ℝ) : 
  a > 0 → (∃ x_a : ℝ, x_a = a ∧ (∀ x : ℝ, 0 < x ∧ x < a → derivative (f a x) > 0) 
  ∧ (∀ x : ℝ, x > a → derivative (f a x) < 0) ∧ f a a = a * (Real.log a + a - 1)) := sorry

theorem part_III (f : ℝ → ℝ) (a : ℝ) : 
  (∃ f : ℝ → ℝ, f = (λ x, a * Real.log x - x^2 + (2 * a - 1) * x) 
  ∧ (a > 1 → (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ f z1 = 0 ∧ f z2 = 0))) := sorry

end part_I_part_II_part_III_l440_440467


namespace p_pow_four_minus_one_divisible_by_ten_l440_440154

theorem p_pow_four_minus_one_divisible_by_ten
  (p : Nat) (prime_p : Nat.Prime p) (h₁ : p ≠ 2) (h₂ : p ≠ 5) : 
  10 ∣ (p^4 - 1) := 
by
  sorry

end p_pow_four_minus_one_divisible_by_ten_l440_440154


namespace calculate_weekly_charge_l440_440582

-- Defining conditions as constraints
def daily_charge : ℕ := 30
def total_days : ℕ := 11
def total_cost : ℕ := 310

-- Defining the weekly charge
def weekly_charge : ℕ := 190

-- Prove that the weekly charge for the first week of rental is $190
theorem calculate_weekly_charge (daily_charge total_days total_cost weekly_charge: ℕ) (daily_charge_eq : daily_charge = 30) (total_days_eq : total_days = 11) (total_cost_eq : total_cost = 310) : 
  weekly_charge = 190 :=
by
  sorry

end calculate_weekly_charge_l440_440582


namespace slope_of_line_l440_440075

theorem slope_of_line (t : ℝ) :
  let x := 3 - t * real.sin (real.pi / 9)
  let y := 2 + t * real.cos (real.pi / 9 * 7)
  ∃ k : ℝ, k = (y - 2) / (x - 3) → k = -1 :=
sorry

end slope_of_line_l440_440075


namespace OH_passes_through_centroid_l440_440569

open Real EuclideanGeometry

variable {A B C P Q O H G : Point} -- Define points as variables

-- Given conditions
axiom condition1 : ∀ (A B : Point), ∃ (C : Point), is_on_semicircle C A B
axiom condition2 : ∀ (A B C : Point), ∃ (P Q : Point), (P ∈ segment A B) ∧ (Q ∈ segment A B) ∧ dist A P = dist A C ∧ dist B Q = dist B C
axiom condition3 : ∀ (C P Q : Point), ∃ (O : Point), is_circumcenter O C P Q
axiom condition4 : ∀ (C P Q : Point), ∃ (H : Point), is_orthocenter H C P Q

-- To prove
theorem OH_passes_through_centroid (A B : Point) :
  ∀ (C P Q O H G : Point),
    (C ∈ semicircle A B) →
    (P ∈ segment A B) →
    (Q ∈ segment A B) →
    dist A P = dist A C →
    dist B Q = dist B C →
    is_circumcenter O C P Q →
    is_orthocenter H C P Q →
    G = centroid C P Q →
    line_passing_through O H G := 
by
  sorry

end OH_passes_through_centroid_l440_440569


namespace sequence_properties_l440_440919

-- Definitions of arithmetic mean, geometric mean, and harmonic mean
def arithmetic_mean (x y : ℝ) : ℝ := (x + y) / 2
def geometric_mean (x y : ℝ) : ℝ := sqrt (x * y)
def harmonic_mean (x y : ℝ) : ℝ := 2 / ((1 / x) + (1 / y))

-- Sequences definitions
noncomputable def A : ℕ → ℝ 
| 0 => arithmetic_mean x y
| n + 1 => arithmetic_mean (A n) (H n)

noncomputable def G : ℕ → ℝ 
| 0 => geometric_mean x y
| n + 1 => geometric_mean (A n) (H n)

noncomputable def H : ℕ → ℝ 
| 0 => harmonic_mean x y
| n + 1 => harmonic_mean (A n) (H n)

-- Theorem to prove truth of the statements
theorem sequence_properties (x y : ℝ) (hx : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (∀ n, A (n + 1) < A n) ∧
  (∀ n, G (n + 1) = G n) ∧
  (∀ n, H (n + 1) > H n) :=
sorry

end sequence_properties_l440_440919


namespace angle_OQP_is_right_angle_l440_440144

open EuclideanGeometry

noncomputable section

variables {A B C D O P Q : Point} {C : Circle}

def quadrilateral_intersect_at_P (A B C D P : Point) : Prop :=
  ∃ l : Line, l.Contains A ∧ l.Contains B ∧ ∃ m : Line, m.Contains C ∧ m.Contains D ∧ Line.Intersects l m P

def circumcircle_intersects_at_Q (A B P C D Q : Point) (C₁ C₂ : Circle) : Prop :=
  C₁ = Circle.circumcircle A B P ∧ C₁.Contains Q ∧ C₂ = Circle.circumcircle C D P ∧ C₂.Contains Q

theorem angle_OQP_is_right_angle
  (hC : Circle.with_center_radius O 1 = C)
  (hA : C.Contains A)
  (hB : C.Contains B)
  (hC : C.Contains C)
  (hD : C.Contains D)
  (hP : quadrilateral_intersect_at_P A B C D P)
  (hQ : circumcircle_intersects_at_Q A B P C D Q (Circle.circumcircle A B P) (Circle.circumcircle C D P)) :
  angle O Q P = 90 := 
sorry

end angle_OQP_is_right_angle_l440_440144


namespace sum_of_intercepts_l440_440321

theorem sum_of_intercepts (x y : ℝ) (h : y + 3 = -2 * (x + 5)) : 
  (- (13 / 2) : ℝ) + (- 13 : ℝ) = - (39 / 2) :=
by sorry

end sum_of_intercepts_l440_440321


namespace value_of_y_l440_440850

theorem value_of_y (x y z : ℕ) (h1 : 3 * x = 3 / 4 * y) (h2 : x + z = 24) (h3 : z = 8) : y = 64 :=
by
  -- Proof omitted
  sorry

end value_of_y_l440_440850


namespace trigonometric_identity_proof_l440_440011

noncomputable def trigonometric_expression : ℝ := 
  (Real.sin (15 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) 
  + Real.cos (165 * Real.pi / 180) * Real.cos (115 * Real.pi / 180)) /
  (Real.sin (35 * Real.pi / 180) * Real.cos (5 * Real.pi / 180) 
  + Real.cos (145 * Real.pi / 180) * Real.cos (85 * Real.pi / 180))

theorem trigonometric_identity_proof : trigonometric_expression = 1 :=
by
  sorry

end trigonometric_identity_proof_l440_440011


namespace chocolates_exceeding_200_l440_440418

-- Define the initial amount of chocolates
def initial_chocolates : ℕ := 3

-- Define the function that computes the amount of chocolates on the nth day
def chocolates_on_day (n : ℕ) : ℕ := initial_chocolates * 3 ^ (n - 1)

-- Define the proof problem
theorem chocolates_exceeding_200 : ∃ (n : ℕ), chocolates_on_day n > 200 :=
by
  -- Proof required here
  sorry

end chocolates_exceeding_200_l440_440418


namespace ball_travel_distance_l440_440689

noncomputable def total_distance_traveled (initial_height : ℕ) (rebound_ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial_height * (1 + 2 * (∑ n in finset.range bounces, rebound_ratio ^ (n + 1)))

theorem ball_travel_distance :
  total_distance_traveled 150 (2 / 3 : ℝ) 5 = 591.67 :=
by
  sorry

end ball_travel_distance_l440_440689


namespace vector_magnitude_l440_440150

noncomputable theory

variables (x y : ℝ)

def a : ℝ × ℝ × ℝ := (0, 1, x)
def b : ℝ × ℝ × ℝ := (2, y, 2)
def c : ℝ × ℝ × ℝ := (1, -2, 1)

def dot (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def parallel (u v : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2, k * v.3)

def subtract (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2 + u.3^2)

theorem vector_magnitude :
  dot a c = 0 →
  parallel b c →
  magnitude (subtract a b) = Real.sqrt 29 :=
by 
  sorry

end vector_magnitude_l440_440150


namespace tan_pi_four_plus_A_l440_440806

theorem tan_pi_four_plus_A (A : ℝ) (h1 : sin A + cos A = -7 / 13) :
  tan (π / 4 + A) = 7 / 17 := 
sorry

end tan_pi_four_plus_A_l440_440806


namespace reflect_point_l440_440198

def point_reflect_across_line (m : ℝ) :=
  (6 - m, m + 1)

theorem reflect_point (m : ℝ) :
  point_reflect_across_line m = (6 - m, m + 1) :=
  sorry

end reflect_point_l440_440198


namespace probability_of_four_ones_approx_l440_440250

noncomputable def probability_of_four_ones_in_twelve_dice : ℚ :=
  (nat.choose 12 4 : ℚ) * (1 / 6 : ℚ) ^ 4 * (5 / 6 : ℚ) ^ 8

theorem probability_of_four_ones_approx :
  probability_of_four_ones_in_twelve_dice ≈ 0.089 :=
sorry

end probability_of_four_ones_approx_l440_440250


namespace calculate_total_houses_built_l440_440706

theorem calculate_total_houses_built :
  let initial_houses := 1426
  let final_houses := 2000
  let rate_a := 25
  let time_a := 6
  let rate_b := 15
  let time_b := 9
  let rate_c := 30
  let time_c := 4
  let total_houses_built := (rate_a * time_a) + (rate_b * time_b) + (rate_c * time_c)
  total_houses_built = 405 :=
by
  sorry

end calculate_total_houses_built_l440_440706


namespace sum_8_terms_l440_440797

variable {a : ℕ → ℝ} [isArithmeticSeq : ∀ n, a (n+1) - a n = d]
variable {S : ℕ → ℝ} [sumSeq : ∀ n, S n = n * (a 1 + a n) / 2]

noncomputable def a_4_5_relation : Prop := a 4 = 18 - a 5

theorem sum_8_terms : a_4_5_relation → S 8 = 72 := by
  sorry

end sum_8_terms_l440_440797


namespace find_m_n_l440_440329

-- Given conditions
def point1 := (3, 3)
def point2 := (7, 1)
def point3 := (9, 4)
variables (m n : ℝ)

-- Fold conditions between points
def fold_line_equation (x y : ℝ) : Prop := y = 2 * x - 8
def matches_point (a b c d : ℝ) : Prop :=
    let midpoint := ((a + c) / 2, (b + d) / 2) in
    fold_line_equation midpoint.1 midpoint.2 ∧
    ((d - b) / (c - a) = -1 / 2)
def point4 := (m, n)

-- Proof statement
theorem find_m_n : matches_point 3 3 7 1 → matches_point 9 4 m n → m + n = 28 / 3 :=
by
sorry

end find_m_n_l440_440329


namespace twelfth_term_of_geometric_sequence_l440_440201

theorem twelfth_term_of_geometric_sequence (a : ℕ) (r : ℕ) (h1 : a * r ^ 4 = 8) (h2 : a * r ^ 8 = 128) : 
  a * r ^ 11 = 1024 :=
sorry

end twelfth_term_of_geometric_sequence_l440_440201


namespace necessary_and_sufficient_condition_for_spheres_l440_440037

theorem necessary_and_sufficient_condition_for_spheres
  (α : ℝ) (α_pos: 0 < α) (R r : ℝ) (R_gt_r: R > r)
  (conical_funnel: true) : sin(α) ≤ (R - r) / R :=
sorry

end necessary_and_sufficient_condition_for_spheres_l440_440037


namespace initial_numbers_unique_l440_440595

theorem initial_numbers_unique 
  (A B C A' B' C' : ℕ) 
  (h1: 1 ≤ A ∧ A ≤ 50) 
  (h2: 1 ≤ B ∧ B ≤ 50) 
  (h3: 1 ≤ C ∧ C ≤ 50) 
  (final_ana : 104 = 2 * A + B + C)
  (final_beto : 123 = A + 2 * B + C)
  (final_caio : 137 = A + B + 2 * C) : 
  A = 13 ∧ B = 32 ∧ C = 46 :=
sorry

end initial_numbers_unique_l440_440595


namespace tails_occurrences_l440_440112

theorem tails_occurrences
  (total_tosses : ℕ)
  (heads_frequency : ℝ)
  (total_tosses_eq : total_tosses = 100)
  (heads_frequency_eq : heads_frequency = 0.49) :
  (total_tosses - (heads_frequency * total_tosses).to_nat = 51) :=
by
  sorry

end tails_occurrences_l440_440112


namespace triangle_right_isosceles_l440_440515

theorem triangle_right_isosceles {A B C : Type*} [LinearOrderedField A] (a b c ha hb : A)
  (h_a_ge_a : ha ≥ b) (h_b_ge_b : hb ≥ c) :
  ∃ (angles : Finset (A)), angles = {90, 45, 45} :=
by sorry

end triangle_right_isosceles_l440_440515


namespace solve_for_x_l440_440184

theorem solve_for_x (x : ℤ) (h : 20 * 14 + x = 20 + 14 * x) : x = 20 := 
by 
  sorry

end solve_for_x_l440_440184


namespace line_passes_through_2nd_and_4th_quadrants_l440_440336

theorem line_passes_through_2nd_and_4th_quadrants (b : ℝ) :
  (∀ x : ℝ, x > 0 → -2 * x + b < 0) ∧ (∀ x : ℝ, x < 0 → -2 * x + b > 0) :=
by
  sorry

end line_passes_through_2nd_and_4th_quadrants_l440_440336


namespace probability_of_four_ones_l440_440246

noncomputable def probability_exactly_four_ones : ℚ :=
  (Nat.choose 12 4 * (1/6)^4 * (5/6)^8)

theorem probability_of_four_ones :
  abs (probability_exactly_four_ones.toReal - 0.114) < 0.001 :=
by
  sorry

end probability_of_four_ones_l440_440246


namespace order_of_values_l440_440146

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem order_of_values (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) : f (-π) > f 3 ∧ f 3 > f (-2) :=
by
  -- Proof would go here
  sorry

end order_of_values_l440_440146


namespace trig_identity_l440_440025

theorem trig_identity (x : ℝ) (h : sin (x + π / 6) = 1 / 3) :
  sin (x - 5 * π / 6) + sin (π / 3 - x) ^ 2 = 5 / 9 :=
by sorry

end trig_identity_l440_440025


namespace problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13_l440_440420

variable (x y a b : ℝ)

def A : ℝ := 2*x^2 + a*x - y + 6
def B : ℝ := b*x^2 - 3*x + 5*y - 1

theorem problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13 
  (h : A x y a - B x y b = -6*y + 7) : a^2 + b^2 = 13 := by
  sorry

end problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13_l440_440420


namespace crop_arrangement_l440_440179

theorem crop_arrangement 
  (varieties : Finset ℕ)
  (A B : ℕ)
  (h_varieties_size : varieties.card = 10)
  (h_AB_in_varieties : A ∈ varieties ∧ B ∈ varieties)
  (bottles : Finset ℕ)
  (h_bottles_size : bottles.card = 6) :
  ∃ (num_ways : ℕ), num_ways = (C(8, 1) * P(9, 5)) :=       
sorry

end crop_arrangement_l440_440179


namespace sum_S_15_l440_440048

noncomputable def sequence_a (n : ℕ) (a_5 a_11 : ℝ) (d : ℝ) : ℝ :=
  if n = 5 then a_5
  else if n = 11 then a_11
  else if n < 5 then a_5 - (5 - n) * d
  else a_5 + (n - 5) * d

def S_n (n : ℕ) (a_5 a_11 : ℝ) (d : ℝ) : ℝ :=
  (n / 2) * (a_5 + a_11)

theorem sum_S_15 (a_5 a_11 : ℝ) (d : ℝ) 
  (h1 : (a_5 - 1) ^ 2015 + 2016 * a_5 + (a_5 - 1) ^ 2017 = 2008)
  (h2 : (a_{11} - 1) ^ 2015 + 2016 * a_{11} + (a_{11} - 1) ^ 2017 = 2024)
  (h3 : d > 0) :
  S_n 15 a_5 a_11 d = 15 :=
sorry

end sum_S_15_l440_440048


namespace find_numbers_l440_440988

theorem find_numbers (a b : ℕ) (h_sum : a + b = 667) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 120) :
  (a = 552 ∧ b = 115) ∨ (a = 435 ∧ b = 232) ∨ (a = 115 ∧ b = 552) ∨ (a = 232 ∧ b = 435) :=
begin
  sorry
end

end find_numbers_l440_440988


namespace count_valid_x_l440_440893

def is_valid_x (x : Nat) : Prop :=
  x > 0 ∧ x < 10000 ∧ (2 ^ x - x ^ 2) % 7 = 0

theorem count_valid_x (s : Finset Nat) :
  s.filter is_valid_x.card = 2857 :=
  sorry

end count_valid_x_l440_440893


namespace prob_divisor_of_12_l440_440670

theorem prob_divisor_of_12 :
  (∃ d : Finset ℕ, d = {1, 2, 3, 4, 6}) → (∃ s : Finset ℕ, s = {1, 2, 3, 4, 5, 6}) →
  let favorable := 5
  let total := 6
  favorable / total = (5 : ℚ / 6 ) := sorry

end prob_divisor_of_12_l440_440670


namespace solve_for_x_l440_440858

theorem solve_for_x (x : ℝ) (h : sqrt (3 / x + 3) = 5 / 3) : x = -27 / 2 :=
by
  sorry

end solve_for_x_l440_440858


namespace race_distance_l440_440649

theorem race_distance (d v_A v_B v_C : ℝ) (h1 : d / v_A = (d - 20) / v_B)
  (h2 : d / v_B = (d - 10) / v_C) (h3 : d / v_A = (d - 28) / v_C) : d = 100 :=
by
  sorry

end race_distance_l440_440649


namespace lambda_range_l440_440801

variables (A B : ℝ × ℝ)
variables (H_line : ∃ p : ℝ × ℝ, 3 * p.1 - 4 * p.2 + 3 = 0)
variables (lambda : ℝ)

-- Define the points A and B
def A := (2, 3) : ℝ × ℝ
def B := (6, -3) : ℝ × ℝ

-- Define a point P on the line 3x - 4y + 3 = 0, given the form P = (x, (3(x+1))/4)
def P (x : ℝ) : ℝ × ℝ := (x, (3 * (x + 1)) / 4)

-- Define the vector from A to P
def AP (x : ℝ) : ℝ × ℝ := (P x).1 - A.1, (P x).2 - A.2

-- Define the vector from B to P
def BP (x : ℝ) : ℝ × ℝ := (P x).1 - B.1, (P x).2 - B.2

-- Define the dot product of the vectors AP and BP
def dot_product (x : ℝ) : ℝ := (AP x).1 * (BP x).1 + (AP x).2 * (BP x).2

-- Define the condition with lambda such that dot product plus 2 lambda equals zero
def equation (x : ℝ) : Prop := dot_product x + 2 * lambda = 0

-- The theorem to prove the range of lambda for which equation has two distinct roots
theorem lambda_range : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ equation x1 ∧ equation x2) ↔ lambda < 2 := 
by sorry

end lambda_range_l440_440801


namespace pages_left_l440_440528

theorem pages_left (total_pages read_fraction : ℕ) (h_total_pages : total_pages = 396) (h_read_fraction : read_fraction = 1/3) : total_pages * (1 - read_fraction) = 264 := 
by
  sorry

end pages_left_l440_440528


namespace isosceles_triangle_l440_440572

theorem isosceles_triangle 
  {a b : ℝ} {α β : ℝ} 
  (h : a / (Real.cos α) = b / (Real.cos β)) : 
  a = b :=
sorry

end isosceles_triangle_l440_440572


namespace complex_product_conjugate_eq_l440_440808

noncomputable def given_z : ℂ := (1 - 2 * complex.I) / (1 + complex.I)

theorem complex_product_conjugate_eq :
  given_z * (complex.conj given_z) = 5 / 2 := by
  sorry

end complex_product_conjugate_eq_l440_440808


namespace no_common_real_solution_l440_440723

theorem no_common_real_solution :
  ¬ ∃ (x y : ℝ), (x^2 - 6 * x + y + 9 = 0) ∧ (x^2 + 4 * y + 5 = 0) :=
by
  sorry

end no_common_real_solution_l440_440723


namespace modulus_eq_sqrt_five_of_pure_imaginary_l440_440030

theorem modulus_eq_sqrt_five_of_pure_imaginary (b : ℝ)
  (h : (1 + b * Complex.I) * (2 - Complex.I)).re = 0 :
  Complex.abs (1 + b * Complex.I) = Real.sqrt 5 :=
sorry

end modulus_eq_sqrt_five_of_pure_imaginary_l440_440030


namespace lennon_current_age_l440_440128

theorem lennon_current_age (L : ℕ) (ophelia_age_in_2_years : ℕ) (ophelia_current_age : ℕ) (h: ophelia_current_age + 2 = 4 * (L + 2)) : L = 8 :=
by
  -- ophelia_current_age = 38
  have ophelia_current_age := 38
  -- ophelia_age_in_2_years = ophelia_current_age + 2
  have ophelia_age_in_2_years := ophelia_current_age + 2
  -- L + 2 = lennon_age_in_2_years
  have lennon_age_in_2_years := L + 2
  -- 38 + 2 = 4 * (L + 2) -> 40 = 4 * (L + 2)
  have h := (ophelia_current_age + 2) * 4 = (L + 2) * 4 := h
  -- need to prove L = 8
  sorry

end lennon_current_age_l440_440128


namespace math_problem_l440_440484

noncomputable theory

variables {a c b d : ℝ} {x q y z : ℝ}

theorem math_problem (h1 : a^(2*x) = c^(3*q)) (h2 : c^(3*q) = b) (h3 : c^(4*y) = a^(5*z)) (h4 : a^(5*z) = d) :
  2*x*5*z = 3*q*4*y :=
by
  sorry

end math_problem_l440_440484


namespace diamond_op_example_l440_440489

def diamond_op (x y : ℕ) : ℕ := 3 * x + 5 * y

theorem diamond_op_example : diamond_op 2 7 = 41 :=
by {
    -- proof goes here
    sorry
}

end diamond_op_example_l440_440489


namespace range_of_t_l440_440827

def f (x: ℝ) (t: ℝ) : ℝ :=
if x < t then x + 6 else x^2 + 2 * x

theorem range_of_t : ∀ t: ℝ, (∀ y: ℝ, ∃ x: ℝ, f x t = y) ↔ (-7 ≤ t ∧ t ≤ 2) :=
begin
  sorry
end

end range_of_t_l440_440827


namespace max_tan_angle_BAD_l440_440617

theorem max_tan_angle_BAD (A B C D : Point) (h1 : angle C = 30°) (h2 : distance B C = 8)
(h3 : midpoint D B C) : 
  ∃ (x : ℝ), tan (angle A B D) = 2 * sqrt 3 / (4 * sqrt 2 - 6 * sqrt 3) :=
by
  sorry

end max_tan_angle_BAD_l440_440617


namespace exponential_to_log_l440_440481

theorem exponential_to_log (x : ℝ) (h : 2^x = 18) : x = Real.log 18 / Real.log 2 :=
by sorry

end exponential_to_log_l440_440481


namespace correct_scenarios_l440_440279

-- Conditions definitions
def scenarioA (h t d imp: Prop) : Prop :=
  ∃ (fall_speed knee_bending_time impact_reduction_time : ℝ),
  (fall_speed > 0) ∧ (knee_bending_time > 0) ∧
  (impact_reduction_time > knee_bending_time) ∧ (imp = (knee_bending_time < impact_reduction_time))

def scenarioB (glass_transport foam fill: Prop) : Prop :=
  ∃ (impact_force vibration_force : ℝ),
  (impact_force > 0) ∧ (vibration_force > 0) ∧
  (fill = (impact_force < vibration_force))

def scenarioC (cup paper pull_speed: Prop) : Prop :=
  ∃ (frictional_force impulse : ℝ),
  (frictional_force > 0) ∧ (impulse > 0) ∧
  (pull_speed = (impulse < frictional_force))

def scenarioD (egg floor_type momentum change_impact: Prop) : Prop :=
  ∃ (fall_height initial_momentum final_momentum : ℝ),
  (fall_height > 0) ∧ (initial_momentum > 0) ∧
  (final_momentum = 0) ∧
  (change_impact = (floor_type = "concrete" → change_impact > "sponge"))

-- Prove correctness of the scenarios B and D
theorem correct_scenarios (glass_transport foam fill : Prop) (egg floor_type momentum change_impact : Prop) :
  scenarioB glass_transport foam fill →
  scenarioD egg floor_type momentum change_impact → true :=
by
  unfold scenarioB scenarioD
  sorry

end correct_scenarios_l440_440279


namespace sum_S_10_l440_440437

-- Define the recursive sequence
def a : ℕ → ℚ
| 0 := 1/2
| (n+1) := 1 - 1 / a n

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ := (Finset.range n).sum a

theorem sum_S_10 : S 10 = 5 := by
  sorry

end sum_S_10_l440_440437


namespace cosine_angle_DE_BC_l440_440998

variables {A B C D E : Type}
variables [add_comm_group A] [module ℝ A]
variables {AB AD AC AE DE BC : A}
variable h_m := is_midpoint B D E
variable h_AB : ∥AB∥ = 1
variable h_DE : ∥DE∥ = 1
variable h_BC : ∥BC∥ = 8
variable h_CA : ∥AC∥ = sqrt 50
variable h_dotprod : (AB ∙ AD) + (AC ∙ AE) = 4

theorem cosine_angle_DE_BC :
  let θ : ℝ := angle DE BC in
  cos θ = 1 :=
sorry

end cosine_angle_DE_BC_l440_440998


namespace part1_part2_l440_440085

-- Define the vectors and conditions
def vector_a : ℝ × ℝ := (8, -4)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

-- Dot product of two vectors
noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Proof statement for part 1
theorem part1 (x : ℝ) (h : vector_a = (-4 : ℝ) • vector_b x) : 
  magnitude (vector_b x) = real.sqrt 5 := sorry

-- Proof statement for part 2
theorem part2 :
  let x := 2
  let b := vector_b 2
  let cos_theta := dot_product vector_a b / (magnitude vector_a * magnitude b)
  cos_theta = (3 / 5 : ℝ) := sorry

end part1_part2_l440_440085


namespace count_valid_pairs_l440_440777

def I : Set Nat := {1, 2, 3}

def is_nonempty (s : Set Nat) : Prop := s.nonempty

def sum_of_set (s : Set Nat) : Nat :=
  s.to_finset.sum (λ x, x)

def valid_pair (A B : Set Nat) : Prop :=
  is_nonempty A ∧ is_nonempty B ∧ sum_of_set A > sum_of_set B

theorem count_valid_pairs : 
  (set.prod (powerset I \ {∅}) (powerset I \ {∅})).count (λ p, valid_pair p.1 p.2) = 20 :=
sorry

end count_valid_pairs_l440_440777


namespace tom_total_spent_correct_l440_440994

-- Definitions for discount calculations
def original_price_skateboard : ℝ := 9.46
def discount_rate_skateboard : ℝ := 0.10
def discounted_price_skateboard : ℝ := original_price_skateboard * (1 - discount_rate_skateboard)

def original_price_marbles : ℝ := 9.56
def discount_rate_marbles : ℝ := 0.10
def discounted_price_marbles : ℝ := original_price_marbles * (1 - discount_rate_marbles)

def price_shorts : ℝ := 14.50

def original_price_action_figures : ℝ := 12.60
def discount_rate_action_figures : ℝ := 0.20
def discounted_price_action_figures : ℝ := original_price_action_figures * (1 - discount_rate_action_figures)

-- Total for all discounted items
def total_discounted_items : ℝ := 
  discounted_price_skateboard + discounted_price_marbles + price_shorts + discounted_price_action_figures

-- Currency conversion for video game
def price_video_game_eur : ℝ := 20.50
def exchange_rate_eur_to_usd : ℝ := 1.12
def price_video_game_usd : ℝ := price_video_game_eur * exchange_rate_eur_to_usd

-- Total amount spent including the video game
def total_spent : ℝ := total_discounted_items + price_video_game_usd

-- Lean proof statement
theorem tom_total_spent_correct :
  total_spent = 64.658 :=
by {
  -- This is a placeholder "by sorry" which means the proof is missing.
  sorry
}

end tom_total_spent_correct_l440_440994


namespace value_of_a_l440_440883

noncomputable def find_a : ℝ :=
  let C (x y : ℝ) (a : ℝ) : Prop := (x - real.sqrt a)^2 + (y - a)^2 = 1
  let l (x y : ℝ) : Prop := y = 2 * x - 6
  let distance_to_line (x y : ℝ) : ℝ := abs (2 * real.sqrt (x*x + y*y) - y - 6) / real.sqrt 5
  let d : ℝ := real.sqrt 5 - 1
  let center_x (a : ℝ) : ℝ := real.sqrt a
  let center_y (a : ℝ) : ℝ := a
  let center_distance (a : ℝ) : ℝ := distance_to_line (center_x a) (center_y a)
  if (a >= 0) ∧ (center_distance a = 1 + d) then a else sorry -- Proof or verification part is omitted

theorem value_of_a :
  ∃ a : ℝ, a = 1 ∧ 
    ∀ x y : ℝ, 
    (a ≥ 0) ∧ 
    ((x - (real.sqrt a))^2 + (y - a)^2 = 1) ∧ 
    (abs (2 * (real.sqrt (x*x + y*y)) - y - 6) / real.sqrt 5 = real.sqrt 5 - 1) := 
sorry

end value_of_a_l440_440883


namespace total_mangoes_calculation_l440_440989

-- Define conditions as constants
def boxes : ℕ := 36
def dozen_to_mangoes : ℕ := 12
def dozens_per_box : ℕ := 10

-- Define the expected correct answer for the total mangoes
def expected_total_mangoes : ℕ := 4320

-- Lean statement to prove
theorem total_mangoes_calculation :
  dozens_per_box * dozen_to_mangoes * boxes = expected_total_mangoes :=
by sorry

end total_mangoes_calculation_l440_440989


namespace tan_product_l440_440712

theorem tan_product : 
(1 + Real.tan (Real.pi / 60)) * (1 + Real.tan (Real.pi / 30)) * (1 + Real.tan (Real.pi / 20)) * (1 + Real.tan (Real.pi / 15)) * (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 10)) * (1 + Real.tan (Real.pi / 9)) * (1 + Real.tan (Real.pi / 6)) = 2^8 :=
by
  sorry 

end tan_product_l440_440712


namespace find_t_to_satisfy_conditions_l440_440440

theorem find_t_to_satisfy_conditions (t : ℝ) 
  (h1 : ∥(1 / t, 0)∥ = 1 / t)
  (h2 : ∥(0, t)∥ = t)
  (h3 : let AP := (1, 0) + (0, 4) in 
        let BC := (-1/t, t) in 
        (AP.1 * BC.1 + AP.2 * BC.2 = 0)) :
  t = 1 / 2 := 
sorry

end find_t_to_satisfy_conditions_l440_440440


namespace color_numbers_l440_440741

open Nat

def is_coprime_set (s1 s2 : Finset ℕ) : Prop :=
  s1.prod id.coprime s2.prod id

def colors (n : ℕ) : Finset ℕ × Finset ℕ := sorry

theorem color_numbers :
  (∃ blue red : Finset ℕ, (Finset.range 21).erase 0 = blue ∪ red ∧ 
  blue ∩ red = ∅ ∧ is_coprime_set blue red ∧ blue.nonempty ∧ red.nonempty ∧
  62 = Finset.powerset (Finset.range 21).erase 0)

end color_numbers_l440_440741


namespace imaginary_part_calculation_l440_440591

open Complex

theorem imaginary_part_calculation : 
  Im ((2 * I) / (1 - I) + 2) = 1 :=
by
  -- Definitions and conditions are implicitly addressed within Lean's complex arithmetic
  sorry

end imaginary_part_calculation_l440_440591


namespace intersection_of_A_and_B_l440_440471

open Set

variable {α : Type}

-- Definitions of the sets A and B
def A : Set ℤ := {-1, 0, 2, 3, 5}
def B : Set ℤ := {x | -1 < x ∧ x < 3}

-- Define the proof problem as a theorem
theorem intersection_of_A_and_B : A ∩ B = {0, 2} :=
by
  sorry

end intersection_of_A_and_B_l440_440471


namespace two_pipes_fill_tank_l440_440266

theorem two_pipes_fill_tank (C : ℝ) (hA : ∀ (t : ℝ), t = 10 → t = C / (C / 10)) (hB : ∀ (t : ℝ), t = 15 → t = C / (C / 15)) :
  ∀ (t : ℝ), t = C / (C / 6) → t = 6 :=
by
  sorry

end two_pipes_fill_tank_l440_440266


namespace expected_fixed_balls_after_swaps_l440_440992

/-- 
The expected number of balls occupying their original positions after Chris, Silva, and Alex 
each make one swap in a circular arrangement of six balls is 2.0. 
-/
theorem expected_fixed_balls_after_swaps : 
  let balls := {1, 2, 3, 4, 5, 6} 
  let swap := λ (b : Fin 6) (i j: Fin 6), if j == b then i else if i == b then j else b 
  ∀ chris_swap silva_swap alex_swap : Fin 6 → Fin 6,
  expected_value (λ b, if swap (swap (swap b chris_swap) silva_swap) alex_swap = b then 1 else 0) = 2 := 
by sorry

end expected_fixed_balls_after_swaps_l440_440992


namespace range_of_a_l440_440469

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) ↔ (a ≤ -2 ∨ a = 1) := 
sorry

end range_of_a_l440_440469


namespace compute_expr_l440_440371

-- Definitions
def a := 150 / 5
def b := 40 / 8
def c := 16 / 32
def d := 3

def expr := 20 * (a - b + c + d)

-- Theorem
theorem compute_expr : expr = 570 :=
by
  sorry

end compute_expr_l440_440371


namespace hcf_of_given_numbers_l440_440972

def hcf (x y : ℕ) : ℕ := Nat.gcd x y

theorem hcf_of_given_numbers :
  ∃ (A B : ℕ), A = 33 ∧ A * B = 363 ∧ hcf A B = 11 := 
by
  sorry

end hcf_of_given_numbers_l440_440972


namespace minimize_wood_frame_l440_440342

noncomputable def min_wood_frame (x y : ℝ) : Prop :=
  let area_eq : Prop := x * y + x^2 / 4 = 8
  let length := 2 * (x + y) + Real.sqrt 2 * x
  let y_expr := 8 / x - x / 4
  let length_expr := (3 / 2 + Real.sqrt 2) * x + 16 / x
  let min_x := Real.sqrt (16 / (3 / 2 + Real.sqrt 2))
  area_eq ∧ y = y_expr ∧ length = length_expr ∧ x = 2.343 ∧ y = 2.828

theorem minimize_wood_frame : ∃ x y : ℝ, min_wood_frame x y :=
by
  use 2.343
  use 2.828
  unfold min_wood_frame
  -- we leave the proof of the properties as sorry
  sorry

end minimize_wood_frame_l440_440342


namespace distance_from_midpoint_to_y_axis_correct_l440_440297

noncomputable def distance_from_midpoint_to_y_axis {F A B : Type*}
  (focus : F) (parabola : polynomial ℝ) (points_on_parabola : A × B)
  (distance_sum : ℝ) :
  real :=
  let F := (1 / 2 : ℝ, 0 : ℝ) in
  let parabola := y^2 = 2 * x in
  let (A, B) := points_on_parabola in
  let (x1, y1) := A in
  let (x2, y2) := B in
  have : x1 + x2 = 7, from distance_sum,
  (x1 + x2) / 2

theorem distance_from_midpoint_to_y_axis_correct :
  distance_from_midpoint_to_y_axis = 7 / 2 :=
by assumption

end distance_from_midpoint_to_y_axis_correct_l440_440297


namespace lattice_point_in_K_l440_440911

theorem lattice_point_in_K 
  (n : ℕ) (h_n : 2 ≤ n) (K : Set (ℝ × ℝ)) (h_closed : IsClosed K)
  (h_convex : Convex ℝ K) (h_area : MeasureTheory.Measure n K ≥ (n : ℝ))
  (h_subset : K ⊆ Set.Ioc (0 : ℝ) n ×ˢ Set.Ioc (0 : ℝ) n) :
  ∃ x : ℤ × ℤ, ↑x ∈ K :=
sorry

end lattice_point_in_K_l440_440911


namespace ab_cd_value_l440_440483

theorem ab_cd_value (a b c d: ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 5)
  (h3 : a + c + d = 14)
  (h4 : b + c + d = 9) :
  a * b + c * d = 338 / 9 := 
sorry

end ab_cd_value_l440_440483


namespace range_of_ab_plus_a_plus_b_l440_440460

noncomputable def f (x : ℝ) : ℝ := abs (x^2 + 2*x - 1)

theorem range_of_ab_plus_a_plus_b (a b : ℝ) (h₁ : a < b) (h₂ : b < -1) (h₃ : f a = f b) : 
  ∃ (S : Set ℝ), S = set.Ioo (-1) 1 ∧ ∀ x ∈ S, ∃ (a b : ℝ), a < b ∧ b < -1 ∧ f a = f b ∧ ab + a + b = x :=
sorry

end range_of_ab_plus_a_plus_b_l440_440460


namespace prove_p_or_q_l440_440802

-- Define propositions p and q
def p : Prop := ∃ n : ℕ, 0 = 2 * n
def q : Prop := ∃ m : ℕ, 3 = 2 * m

-- The Lean statement to prove
theorem prove_p_or_q : p ∨ q := by
  sorry

end prove_p_or_q_l440_440802


namespace complementary_events_A_B_l440_440716

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def A (n : ℕ) : Prop := is_odd n
def B (n : ℕ) : Prop := is_even n
def C (n : ℕ) : Prop := is_multiple_of_3 n

theorem complementary_events_A_B :
  (∀ n, A n → ¬ B n) ∧ (∀ n, B n → ¬ A n) ∧ (∀ n, A n ∨ B n) :=
  sorry

end complementary_events_A_B_l440_440716


namespace gcd_of_polynomials_l440_440443

theorem gcd_of_polynomials (b : ℤ) (h : b % 2 = 1 ∧ 8531 ∣ b) :
  Int.gcd (8 * b^2 + 33 * b + 125) (4 * b + 15) = 5 :=
by
  sorry

end gcd_of_polynomials_l440_440443


namespace ellipse_standard_eq_l440_440429

def eccentricity := Real
def point := ℝ × ℝ
def ellipse_eq (a b : Real) := ∀ (x y : Real), x^2 / a + y^2 / b = 1

axioms
  (center_origin : (0, 0) : point)
  (foci_on_x_axis : True)
  (e : eccentricity)
  (h_e : e = Real.sqrt 5 / 5)
  (P : point)
  (h_P : P = (-5, 4))

theorem ellipse_standard_eq :
  ∃ (a b : Real), ellipse_eq (45) (36) := by
  sorry

end ellipse_standard_eq_l440_440429


namespace peg_arrangement_l440_440606

theorem peg_arrangement :
  let Y := 5
  let R := 4
  let G := 3
  let B := 2
  let O := 1
  (Y! * R! * G! * B! * O!) = 34560 :=
by
  sorry

end peg_arrangement_l440_440606


namespace slope_divides_polygon_area_l440_440882

structure Point where
  x : ℝ
  y : ℝ

noncomputable def polygon_vertices : List Point :=
  [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩]

-- Define the area calculation and conditions needed 
noncomputable def area_of_polygon (vertices : List Point) : ℝ :=
  -- Assuming here that a function exists to calculate the area given the vertices
  sorry

def line_through_origin (slope : ℝ) (x : ℝ) : Point :=
  ⟨x, slope * x⟩

theorem slope_divides_polygon_area :
  let line := line_through_origin (2 / 7)
  ∀ x : ℝ, ∃ (G : Point), 
  polygon_vertices = [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩] →
  area_of_polygon polygon_vertices / 2 = 
  area_of_polygon [⟨0, 0⟩, line x, G] :=
sorry

end slope_divides_polygon_area_l440_440882


namespace cos_double_angle_of_triangle_angle_l440_440143

theorem cos_double_angle_of_triangle_angle (α : ℝ) (hα : α ∈ (0, π)) (h_tan : Real.tan α = -3 / 4) : 
  Real.cos (2 * α) = 7 / 25 :=
sorry

end cos_double_angle_of_triangle_angle_l440_440143


namespace difference_between_possible_x_values_l440_440098

theorem difference_between_possible_x_values :
  ∀ (x : ℝ), (x + 3) ^ 2 / (2 * x + 15) = 3 → (x = 6 ∨ x = -6) →
  (abs (6 - (-6)) = 12) :=
by
  intro x h1 h2
  sorry

end difference_between_possible_x_values_l440_440098


namespace xy_over_y_plus_x_l440_440096

theorem xy_over_y_plus_x {x y z : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : 1/x + 1/y = 1/z) : z = xy/(y+x) :=
sorry

end xy_over_y_plus_x_l440_440096


namespace find_line_passing_through_two_points_l440_440084

noncomputable def equation_of_line_passing_through_two_complex_points 
  (z1 z2 z3 z : ℂ) : Prop :=
  (z - z1) * (conj(z2) - conj(z1)) = (conj(z) - conj(z1)) * (z2 - z1)

theorem find_line_passing_through_two_points
  (z1 z2 z3 : ℂ) :
  ∃ z : ℂ, equation_of_line_passing_through_two_complex_points z1 z2 z3 z :=
sorry

end find_line_passing_through_two_points_l440_440084


namespace place_candle_safe_l440_440021

variable (A : Type) [LinearOrderedField A]
variable (A0 A1 A2 A3 A4 : A)

-- Definition of convex pentagon
def is_convex_pentagon (A0 A1 A2 A3 A4 : A) : Prop :=
  convex_hull (A0 :: A1 :: A2 :: A3 :: A4 :: [])

-- Definition of triangular cut from the pentagon
def is_triangulated_cut (A : Type) (vertices : List A) (p q : A) : Prop :=
  q ∉ vertices ∧ p ∉ vertices ∧
  ∃ (i : ℕ), vertices.nth i = some p ∧ vertices.nth (i + 1) = some q

-- Definition of a safe point for the candle
def candle_safe (A0 A1 A2 A3 A4 : A) (P : A) : Prop :=
  (∃ i : ℕ, i < 5 ∧
    ((P ∈ triangle (vertices.nth (i - 1) (i % 5)) (vertices.nth i (i % 5)) (vertices.nth (i + 1 % 5) (i % 5)))
      ∨ (P ∈ boundary (is_convex_pentagon A0 A1 A2 A3 A4))))

-- Theorem: Prove that placing a candle at any safe point ensures it isn't cut off.
theorem place_candle_safe (A : Type) [LinearOrderedField A]
  (A0 A1 A2 A3 A4 : A) (P : A) 
  (h_convex : is_convex_pentagon A0 A1 A2 A3 A4)
  (h_tri_cut : ∀ p q : A, is_triangulated_cut A [A0, A1, A2, A3, A4] p q) :
  candle_safe A0 A1 A2 A3 A4 P :=
by {
  sorry
}

end place_candle_safe_l440_440021


namespace triangle_right_isosceles_l440_440516

theorem triangle_right_isosceles {A B C : Type*} [LinearOrderedField A] (a b c ha hb : A)
  (h_a_ge_a : ha ≥ b) (h_b_ge_b : hb ≥ c) :
  ∃ (angles : Finset (A)), angles = {90, 45, 45} :=
by sorry

end triangle_right_isosceles_l440_440516


namespace max_value_f_l440_440038

def f (a x y : ℝ) : ℝ := a * x + y

theorem max_value_f (a : ℝ) (x y : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : |x| + |y| ≤ 1) :
    f a x y ≤ 1 :=
by
  sorry

end max_value_f_l440_440038


namespace simplest_radical_form_l440_440349

def is_simplest_radical_form (r : ℝ) : Prop :=
  ∀ x : ℝ, x * x = r → ∃ y : ℝ, y * y ≠ r

theorem simplest_radical_form :
   (is_simplest_radical_form 6) :=
by
  sorry

end simplest_radical_form_l440_440349


namespace total_transaction_loss_l440_440316

-- Define the cost and selling prices given the conditions
def cost_price_house (h : ℝ) := (7 / 10) * h = 15000
def cost_price_store (s : ℝ) := (5 / 4) * s = 15000

-- Define the loss calculation for the transaction
def transaction_loss : Prop :=
  ∃ (h s : ℝ),
    (7 / 10) * h = 15000 ∧
    (5 / 4) * s = 15000 ∧
    h + s - 2 * 15000 = 3428.57

-- The theorem stating the transaction resulted in a loss of $3428.57
theorem total_transaction_loss : transaction_loss :=
by
  sorry

end total_transaction_loss_l440_440316


namespace check_incorrect_equation_l440_440346

theorem check_incorrect_equation :
  ¬ (sqrt (121 / 225) = ± (11 / 15)) :=
sorry

end check_incorrect_equation_l440_440346


namespace least_gumballs_to_get_four_same_color_l440_440675

theorem least_gumballs_to_get_four_same_color
  (R W B : ℕ)
  (hR : R = 9)
  (hW : W = 7)
  (hB : B = 8) : 
  ∃ n, n = 10 ∧ (∀ m < n, ∀ r w b : ℕ, r + w + b = m → r < 4 ∧ w < 4 ∧ b < 4) ∧ 
  (∀ r w b : ℕ, r + w + b = n → r = 4 ∨ w = 4 ∨ b = 4) :=
sorry

end least_gumballs_to_get_four_same_color_l440_440675


namespace percent_geese_non_duck_is_53_33_l440_440901

  -- Define predicates for each condition of the problem
  def total_birds := 100       -- Assume we are proportional so use 100 for ease
  def geese_percent := 0.4
  def swans_percent := 0.2
  def herons_percent := 0.15
  def ducks_percent := 0.25
  
  -- Define the computed values based on percentages
  def geese := geese_percent * total_birds
  def swans := swans_percent * total_birds
  def herons := herons_percent * total_birds
  def ducks := ducks_percent * total_birds
  def non_duck_birds := total_birds - ducks

  -- The percentage of geese among non-duck birds
  def percent_geese_non_duck := (geese / non_duck_birds) * 100

  -- The theorem to prove
  theorem percent_geese_non_duck_is_53_33 :
    percent_geese_non_duck = 53.33 := by
    sorry
  
end percent_geese_non_duck_is_53_33_l440_440901


namespace probability_exactly_one_die_divisible_by_3_l440_440274

noncomputable def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def prob_exactly_one_divisible_by_3 : ℚ :=
  let outcomes := {1, 2, 3, 4, 5, 6}
  let total_outcomes := (outcomes × outcomes × outcomes).card
  let favorable_outcomes := (outcomes × outcomes × outcomes).filter (
    λ x : ℕ × ℕ × ℕ, 
    is_divisible_by_3 (x.1) ∧ ¬is_divisible_by_3 (x.2) ∧ ¬is_divisible_by_3 (x.3) ∨
    ¬is_divisible_by_3 (x.1) ∧ is_divisible_by_3 (x.2) ∧ ¬is_divisible_by_3 (x.3) ∨
    ¬is_divisible_by_3 (x.1) ∧ ¬is_divisible_by_3 (x.2) ∧ is_divisible_by_3 (x.3)
  ).card
  favorable_outcomes / total_outcomes

theorem probability_exactly_one_die_divisible_by_3:
  prob_exactly_one_divisible_by_3 = 4 / 9 := 
by
  sorry

end probability_exactly_one_die_divisible_by_3_l440_440274


namespace cosine_angle_between_planes_l440_440538

def plane1 (x y z : ℝ) : Prop := 2 * x - 3 * y + 4 * z + 1 = 0
def plane2 (x y z : ℝ) : Prop := 4 * x + 6 * y - 2 * z + 2 = 0

noncomputable def normal_vector_1 : ℝ × ℝ × ℝ := (2, -3, 4)
noncomputable def normal_vector_2 : ℝ × ℝ × ℝ := (4, 6, -2)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2)

theorem cosine_angle_between_planes :
  ∃ θ : ℝ, cos θ = -9 / Real.sqrt 406 :=
by
  have n1 : ℝ × ℝ × ℝ := normal_vector_1
  have n2 : ℝ × ℝ × ℝ := normal_vector_2
  have dot := dot_product n1 n2
  have mag1 := magnitude n1
  have mag2 := magnitude n2
  use Real.arccos (dot / (mag1 * mag2))
  sorry

end cosine_angle_between_planes_l440_440538


namespace exists_lambda_geometric_seq_sum_of_first_n_terms_of_a_l440_440427

-- Definition of the sequence a_n
def a : ℕ → ℤ
| 0     := 1
| 1     := 3
| (n+2) := a (n+1) + 2 * a n

-- Given definition for b_n involving lambda
def b (λ : ℝ) (n : ℕ) : ℝ := a (n + 1) + λ * a n

-- Proof problem for Question 1
theorem exists_lambda_geometric_seq :
  ∃ (λ : ℝ), (∀ n : ℕ, n ≥ 1 → b λ n * b λ (n - 2) = (b λ (n - 1))^2) ↔ (λ = 1 ∨ λ = -2) :=
sorry

-- Sum of the first n terms of the sequence a_n
def S (n : ℕ) : ℝ :=
  (List.range n).sum (λ k, a k)

-- Proof problem for Question 2
theorem sum_of_first_n_terms_of_a :
  ∀ n : ℕ, S n = if n % 2 = 0 then (1 / 3) * (2 ^ (n + 2) - 4)
                               else (1 / 3) * (2 ^ (n + 2) - 5) :=
sorry

end exists_lambda_geometric_seq_sum_of_first_n_terms_of_a_l440_440427


namespace probability_of_four_ones_l440_440244

noncomputable def probability_exactly_four_ones : ℚ :=
  (Nat.choose 12 4 * (1/6)^4 * (5/6)^8)

theorem probability_of_four_ones :
  abs (probability_exactly_four_ones.toReal - 0.114) < 0.001 :=
by
  sorry

end probability_of_four_ones_l440_440244


namespace distance_from_focus_l440_440065

noncomputable def point (x y : ℝ) := (x, y)
def parabola (t : ℝ) := (4 * t^2, 4 * t)
def focus : (ℝ × ℝ) := (1, 0)
def distance (A B : ℝ × ℝ) := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem distance_from_focus (m t : ℝ) (P : ℝ × ℝ) :
  P = point 3 m → P = parabola t → distance P focus = 4 :=
by
  intros hP hT
  sorry

end distance_from_focus_l440_440065


namespace regular_polygon_area_l440_440332

theorem regular_polygon_area (n : ℕ) (R : ℝ) (hR : R ≠ 0)
  (h_area : (1/2) * n * R^2 * real.sin (360 / n * (real.pi / 180)) = 2 * R^2) : n = 12 :=
sorry

end regular_polygon_area_l440_440332


namespace max_distance_C1_C2_area_triangle_C1_C3_l440_440118

-- Define the curve C1 in the Cartesian coordinate system
def C1 (α : ℝ) : ℝ × ℝ :=
  ⟨-2 + Real.cos α, -1 + Real.sin α⟩

-- Define the curve C2 in the polar coordinate system, translates to x = 3
def C2 (P : ℝ × ℝ) : Prop :=
  P.1 = 3

-- Define the curve C3 in the Cartesian coordinate system
def C3 (P : ℝ × ℝ) : Prop :=
  P.2 = P.1

-- (1) Prove the maximum distance to the line x = 3 is 6
theorem max_distance_C1_C2 : ∀ α, let P := C1 α in ∃ d, C2 ⟨d, P.2⟩ ∧ Real.dist (P.1, P.2) (d, P.2) = 6 := 
sorry

-- (2) Prove the area of the triangle formed by the intersections of C1 and C3, and the center of C1
theorem area_triangle_C1_C3 : 
  let A := ⟨-1, -1⟩, 
      B := ⟨-2, -2⟩, 
      C1_center := ⟨-2, -1⟩ in 
    C3 A ∧ C3 B ∧ (∃ S, S = 1 / 2) := 
sorry

end max_distance_C1_C2_area_triangle_C1_C3_l440_440118


namespace complex_number_quadrant_l440_440832

noncomputable def matrix_det (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_number_quadrant : 
  let z := matrix_det 1 2 complex.i (complex.i ^ 4) in
  ∃ (q : ℕ), (q = 4) ∧ 
  (z.re > 0) ∧ 
  (z.im < 0) :=
by
  sorry

end complex_number_quadrant_l440_440832


namespace minimum_RS_zero_l440_440912

def Rhombus (ABCD : Type) [AddGroup ABCD] [Module ℝ ABCD] := 
  ∃ (A B C D : ABCD),
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A) ∧ 
  (∥A - C∥ = 20) ∧ (∥B - D∥ = 24) ∧ 
  (∃ O, ∥A - O∥ = 10 ∧ ∥B - O∥ = 12 ∧ ∠AOB = π / 2)

def minimum_RS (ABCD : Type) [AddGroup ABCD] [Module ℝ ABCD]
               (M : ABCD) (BC : Subtype (ABCD × ABCD)) (AC BD RS : ℝ) : Prop :=
  let R := Foot M AC
  let S := Foot M BD
  is_rhombus (A B C D : ABCD)  ∧ M ∈ BC → RS = min (norm (R - S)) 0

theorem minimum_RS_zero (ABCD : Type) [AddGroup ABCD] [Module ℝ ABCD]
  (AC : ℝ) (BD : ℝ) (M : ABCD) (BC : Subtype (ABCD × ABCD)) :
  (Rhombus ABCD) → minimum_RS ABCD M BC AC BD 0 :=
by sorry

end minimum_RS_zero_l440_440912


namespace doris_took_out_erasers_l440_440607

-- Definitions for initial conditions
def initial_erasers : ℕ := 69
def remaining_erasers : ℕ := 15

-- The proposition we need to prove
theorem doris_took_out_erasers (initial_erasers = 69) (remaining_erasers = 15) : 
  doris_took_out = initial_erasers - remaining_erasers := 
  by sorry

end doris_took_out_erasers_l440_440607


namespace trig_identity_l440_440027

noncomputable def sin_eq := sorry -- Define necessary trigonometric properties

theorem trig_identity
  (x : ℝ)
  (h : sin (x + π / 6) = 1 / 3) :
  sin (x - 5 * π / 6) + sin (π / 3 - x) ^ 2 = 5 / 9 :=
by
  sorry

end trig_identity_l440_440027


namespace floor_alpha_six_eq_three_l440_440491

noncomputable def floor_of_alpha_six (α : ℝ) (h : α^5 - α^3 + α - 2 = 0) : ℤ :=
  Int.floor (α^6)

theorem floor_alpha_six_eq_three (α : ℝ) (h : α^5 - α^3 + α - 2 = 0) : floor_of_alpha_six α h = 3 :=
sorry

end floor_alpha_six_eq_three_l440_440491


namespace magnitude_AD_two_l440_440454

noncomputable theory
open_locale real

variables (m n : ℝ^3)
variables (A B C D : ℝ^3)

-- Given conditions
def angle_between_m_n : real := real.pi / 6
def magnitude_m : real := real.sqrt 3
def magnitude_n : real := 2
def vector_AB : ℝ^3 := 2 • m + 2 • n
def vector_AC : ℝ^3 := 2 • m - 6 • n

-- Midpoint definition
def D_midpoint : ℝ^3 := (B + C) / 2

-- AD vector
def vector_AD (A B C : ℝ^3) : ℝ^3 := (vector_AB m n + vector_AC m n) / 2

-- Theorem to prove
theorem magnitude_AD_two : 
  |vector_AD A B C| = 2 :=
sorry

end magnitude_AD_two_l440_440454


namespace distance_not_proportional_to_time_l440_440129

theorem distance_not_proportional_to_time
  (a v t : ℝ) : ¬ ∀ k : ℝ, s = k * t where
  s = a + v * t :=
by
  intro h
  let s' := a + 2 * v * t
  have h1 : s' = 2 * s := by sorry
  contradiction

end distance_not_proportional_to_time_l440_440129


namespace combined_room_size_l440_440226

theorem combined_room_size (M J S : ℝ) 
  (h1 : M + J + S = 800) 
  (h2 : J = M + 100) 
  (h3 : S = M - 50) : 
  J + S = 550 := 
by
  sorry

end combined_room_size_l440_440226


namespace first_floor_bedrooms_l440_440354

theorem first_floor_bedrooms (total_bedrooms : ℕ) (second_floor_bedrooms : ℕ) (h1 : total_bedrooms = 10) (h2 : second_floor_bedrooms = 2) : total_bedrooms - second_floor_bedrooms = 8 :=
by
  rw [h1, h2]
  sorry

end first_floor_bedrooms_l440_440354


namespace quadrilateral_side_difference_l440_440331

variable (a b c d : ℝ)

theorem quadrilateral_side_difference :
  a + b + c + d = 120 →
  a + c = 50 →
  (a^2 + c^2 = 1600) →
  (b + d = 70 ∧ b * d = 450) →
  |b - d| = 2 * Real.sqrt 775 :=
by
  intros ha hb hc hd
  sorry

end quadrilateral_side_difference_l440_440331


namespace least_positive_t_l440_440713

theorem least_positive_t (t : ℕ) (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : π / 10 < α ∧ α ≤ π / 6) 
  (h3 : (3 * α)^2 = α * (π - 5 * α)) :
  t = 27 :=
by
  have hα : α = π / 14 := 
    by
      sorry
  sorry

end least_positive_t_l440_440713


namespace radius_YZ_is_sqrt_136_l440_440503

noncomputable def radius_of_semicircle_on_YZ (XY_area : ℝ) (XZ_arc_length : ℝ) : ℝ :=
  if h : XY_area = 18 * Real.pi ∧ XZ_arc_length = 10 * Real.pi then
    let r_XY := Math.sqrt ((2 * XY_area) / Real.pi);
    let d_XY := 2 * r_XY;
    let r_XZ := XZ_arc_length / Real.pi;
    let d_XZ := 2 * r_XZ;
    let d_YZ := Math.sqrt (d_XY ^ 2 + d_XZ ^ 2);
    d_YZ / 2
  else
    0

theorem radius_YZ_is_sqrt_136 (XY_area : ℝ) (XZ_arc_length : ℝ) (h1 : XY_area = 18 * Real.pi) (h2 : XZ_arc_length = 10 * Real.pi) :
  radius_of_semicircle_on_YZ XY_area XZ_arc_length = Math.sqrt 136 :=
by
  have radius_YZ := radius_of_semicircle_on_YZ XY_area XZ_arc_length;
  rw [radius_of_semicircle_on_YZ, if_pos (and.intro h1 h2)];
  sorry

end radius_YZ_is_sqrt_136_l440_440503


namespace largest_possible_sum_l440_440922

theorem largest_possible_sum (a b : ℤ) (h : a^2 - b^2 = 144) : a + b ≤ 72 :=
sorry

end largest_possible_sum_l440_440922


namespace digit_2023_in_expansion_of_7_div_18_l440_440001

theorem digit_2023_in_expansion_of_7_div_18 :
  (decimal_digit (2023) (7 / 18) = 3) :=
by 
  have exp : (decimal_expansion (7 / 18) = "0.\overline{38}") := sorry,
  have repeat : (repeating_sequence_length "38" = 2) := sorry,
  sorry

end digit_2023_in_expansion_of_7_div_18_l440_440001


namespace count_elements_of_order_l440_440841

theorem count_elements_of_order 
  (p : ℕ) (hp : Nat.prime p) (d : ℕ) (hdiv : d ∣ (p - 1)) :
  ∃ x : ℕ, order_of (x % p) = (p - 1) / d ∧
    (order_of (x % p) = Nat.totient ((p - 1) / d)) := 
sorry

end count_elements_of_order_l440_440841


namespace equal_angles_iff_equidistant_l440_440950

open EuclideanGeometry

variables {P : Type} [EuclideanGeometry P]

/-- Points \( A_{1} \) and \( A_{2} \) belong to planes \( \Pi_{1} \) and \( \Pi_{2} \), which intersect along the line \( l \). 
Prove that the line \( A_{1}A_{2} \) forms equal angles with the planes \( \Pi_{1} \) and \( \Pi_{2} \) if and only if 
the points \( A_{1} \) and \( A_{2} \) are equidistant from the line \( l \). -/
theorem equal_angles_iff_equidistant (A₁ A₂ : P) (Π₁ Π₂ : Plane P)
  (hA₁ : A₁ ∈ Π₁) (hA₂ : A₂ ∈ Π₂) (l : Line P)
  (hΠ : Π₁ ∩ Π₂ = l) :
  (angle A₁ A₂ Π₁ = angle A₁ A₂ Π₂) ↔ (distance A₁ l = distance A₂ l) :=
sorry

end equal_angles_iff_equidistant_l440_440950


namespace nth_cosine_product_identity_l440_440055

theorem nth_cosine_product_identity 
  (h1 : cos (π / 3) = 1 / 2)
  (h2 : cos (π / 5) * cos (2 * π / 5) = 1 / 4)
  (h3 : cos (π / 7) * cos (2 * π / 7) * cos (3 * π / 7) = 1 / 8) :
  ∀ n : ℕ, cos (π / (2 * n + 1)) * cos (2 * π / (2 * n + 1)) * ... * cos (n * π / (2 * n + 1)) = 1 / (2^n) :=
sorry

end nth_cosine_product_identity_l440_440055


namespace total_pieces_of_clothing_l440_440568

def number_of_pieces_per_drawer : ℕ := 2
def number_of_drawers : ℕ := 4

theorem total_pieces_of_clothing : 
  (number_of_pieces_per_drawer * number_of_drawers = 8) :=
by sorry

end total_pieces_of_clothing_l440_440568


namespace ratio_abc_def_l440_440861

theorem ratio_abc_def (a b c d e f : ℝ) 
  (h1 : a / b = 5 / 2) 
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 1)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  abc / def = 7.5 := by
  sorry

end ratio_abc_def_l440_440861


namespace couples_sitting_arrangement_l440_440264

theorem couples_sitting_arrangement : 
  let positions := (1, 2, 3, 4)
  ∃ (P1 P2 C1 C2 : {p // p ∈ {positions}}) 
  (adjacent : ∀ (x y : {p // p ∈ {positions}}), (x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 4) ∨ (x = 2 ∧ y = 1) ∨ (x = 4 ∧ y = 3)) 
  (choices : ∀ (x y : {p // p ∈ {positions}}), (x = P1 ∧ y = P2) ∨ (x = C1 ∧ y = C2)), 
  nat.card {pair1 pair2 : ℕ // adjacent pair1 pair2 ∧ choices pair1 pair2} = 8 :=
sorry

end couples_sitting_arrangement_l440_440264


namespace minimum_cuts_for_10_pieces_l440_440719

theorem minimum_cuts_for_10_pieces :
  ∃ n : ℕ, (n * (n + 1)) / 2 ≥ 10 ∧ ∀ m < n, (m * (m + 1)) / 2 < 10 := sorry

end minimum_cuts_for_10_pieces_l440_440719


namespace correctness_of_statements_l440_440889

def p_value (k2 : ℝ) : ℝ := sorry -- This function would ideally define the P-value calculation.

noncomputable def is_correct_statement_1 (k2 : ℝ) : Prop :=
  p_value k2 >= 6.635 → 0.99

noncomputable def is_correct_statement_2 (k2 : ℝ) : Prop :=
  p_value k2 >= 6.635 → 99

noncomputable def is_correct_statement_3 (k2 : ℝ) : Prop :=
  0.99 → ∀ x, 0.99

noncomputable def is_correct_statement_4 (k2 : ℝ) : Prop :=
  0.99 → 0.01

theorem correctness_of_statements (k2 : ℝ) :
  (is_correct_statement_1 k2 ∧ is_correct_statement_4 k2) :=
by
  sorry

end correctness_of_statements_l440_440889


namespace y_coordinate_of_point_on_line_l440_440513

theorem y_coordinate_of_point_on_line (x y : ℝ) (h1 : -4 = x) (h2 : ∃ m b : ℝ, y = m * x + b ∧ y = 3 ∧ x = 10 ∧ m * 4 + b = 0) : y = -4 :=
sorry

end y_coordinate_of_point_on_line_l440_440513


namespace minimum_value_expression_l440_440449

theorem minimum_value_expression {a b c : ℤ} (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  4 * (a^2 + b^2 + c^2) - (a + b + c)^2 = 8 := 
sorry

end minimum_value_expression_l440_440449


namespace Penglai_sufficient_condition_l440_440312

variable (Penglai Immortal : Prop)

theorem Penglai_sufficient_condition (h : ¬Penglai → ¬Immortal) : Immortal → Penglai :=
by 
  intro hImmortal
  by_contradiction hNotPenglai
  exact h hNotPenglai hImmortal

end Penglai_sufficient_condition_l440_440312


namespace four_horses_start_together_l440_440608

def horse_run_times : List Nat := [2, 3, 5, 7, 11, 13, 17]

def smallest_time (n : Nat) (horses : List Nat) : Nat :=
  let cm := horses.sublistsLength n |>.map Finset.lcm
  cm.foldl min (horses.headD)

theorem four_horses_start_together :
  smallest_time 4 horse_run_times = 210 := by
  sorry

end four_horses_start_together_l440_440608


namespace dice_probability_exactly_four_ones_l440_440260

noncomputable def dice_probability : ℚ := 
  (Nat.choose 12 4) * (1/6)^4 * (5/6)^8

theorem dice_probability_exactly_four_ones : (dice_probability : ℚ) ≈ 0.089 :=
  by sorry -- Skip the proof. 

#eval (dice_probability : ℚ)

end dice_probability_exactly_four_ones_l440_440260


namespace cookies_total_is_60_l440_440937

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Rachel_cookies : ℕ := Jasmine_cookies + 10
def Total_cookies : ℕ := Mona_cookies + Jasmine_cookies + Rachel_cookies

theorem cookies_total_is_60 : Total_cookies = 60 := by
  sorry

end cookies_total_is_60_l440_440937


namespace card_M_l440_440745

-- Define the operation "※" as described in the problem
def my_op (m n : ℕ) : ℕ :=
  if m % 2 = 1 ∧ n % 2 = 1 then m + n else m * n

-- Define the set M
def M : set (ℕ × ℕ) :=
  { p | my_op p.1 p.2 = 16 ∧ p.1 > 0 ∧ p.2 > 0 }

-- Statement to prove
theorem card_M : (set M).finite.to_finset.card = 13 :=
sorry -- proof to be filled in

end card_M_l440_440745


namespace modulus_is_one_modulus_iff_l440_440811

theorem modulus_is_one (a b z : ℂ) (h : |a| ≠ |b|) (hz : |z| = 1) :
  let u := (a + b * z) / (conj b + conj a * z) in |u| = 1 :=
by 
  sorry

theorem modulus_iff (a b z : ℂ) (h : |a| ≠ |b|) :
  let u := (a + b * z) / (conj b + conj a * z) in |u| = 1 ↔ |z| = 1 :=
by 
  sorry

end modulus_is_one_modulus_iff_l440_440811


namespace polygon_sides_l440_440862

theorem polygon_sides (interior_angle : ℝ) (h : interior_angle = 120) : ∃ n : ℕ, n = 6 :=
by
  have exterior_angle := 180 - interior_angle
  have sum_exterior_angles := 360
  have number_of_sides := sum_exterior_angles / exterior_angle
  use 6
  sorry

end polygon_sides_l440_440862


namespace acute_angle_λ_range_l440_440838

variables (a b : ℝ × ℝ) (λ : ℝ)
def is_acute (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 * v2.1 + v1.2 * v2.2) > 0

def non_collinear (v2 : ℝ × ℝ) : Prop :=
  v2.2 ≠ 0

theorem acute_angle_λ_range (h_a : a = (4, 2)) (h_b : b = (2, -1)) :
  (is_acute (a.1 + 2 * b.1, a.2 + 2 * b.2) (a.1 + λ * b.1, a.2 + λ * b.2)) ∧
  (non_collinear (a.1 + λ * b.1, a.2 + λ * b.2)) ↔
  λ > -2 ∨ λ ≠ 2 :=
sorry

end acute_angle_λ_range_l440_440838


namespace find_m_b_l440_440276

-- Define the polynomial P
def P (x : ℚ) : ℚ := x^4 - 8 * x^3 + 20 * x^2 - 34 * x + 15

-- Define the divisor D with variable m
def D (x m : ℚ) : ℚ := x^2 - 3 * x + m

-- Define the remainder R
def R (x b : ℚ) : ℚ := 2 * x + b

theorem find_m_b :
  ∃ (m b : ℚ), 
    (∀ x, P x = (D x m) * (P x /. (D x m)) + (R x b)) →
    (m = 53 / 5 ∧ b = 1859 / 25) :=
by
  sorry

end find_m_b_l440_440276


namespace inequality_solution_l440_440563

theorem inequality_solution (x : ℝ) (h : 3 * x - 5 > 11 - 2 * x) : x > 16 / 5 := 
sorry

end inequality_solution_l440_440563


namespace weather_station_accuracy_l440_440188

def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k)

theorem weather_station_accuracy :
  binomial_probability 3 2 0.9 = 0.243 :=
by
  sorry

end weather_station_accuracy_l440_440188


namespace amount_of_flour_per_large_tart_l440_440561

-- Statement without proof
theorem amount_of_flour_per_large_tart 
  (num_small_tarts : ℕ) (flour_per_small_tart : ℚ) 
  (num_large_tarts : ℕ) (total_flour : ℚ) 
  (h1 : num_small_tarts = 50) 
  (h2 : flour_per_small_tart = 1/8) 
  (h3 : num_large_tarts = 25) 
  (h4 : total_flour = num_small_tarts * flour_per_small_tart) : 
  total_flour = num_large_tarts * (1/4) := 
sorry

end amount_of_flour_per_large_tart_l440_440561


namespace dice_sum_24_probability_l440_440768

noncomputable def probability_sum_24 : ℚ :=
  let prob_single_six := (1 : ℚ) / 6 in
  prob_single_six ^ 4

theorem dice_sum_24_probability :
  probability_sum_24 = 1 / 1296 :=
by
  sorry

end dice_sum_24_probability_l440_440768


namespace linda_total_amount_correct_l440_440930

-- Define the costs of individual items and their quantities
def cost_per_coloring_book : ℕ := 4
def num_coloring_books : ℕ := 2
def cost_per_pack_peanuts : ℕ := 1.50
def num_pack_peanuts : ℕ := 4
def cost_stuffed_animal : ℕ := 11

-- Define the total costs
def total_cost_coloring_books := cost_per_coloring_book * num_coloring_books
def total_cost_peanuts := cost_per_pack_peanuts * num_pack_peanuts

-- The total amount Linda gave the cashier
def total_amount_linda_gave := total_cost_coloring_books + total_cost_peanuts + cost_stuffed_animal

theorem linda_total_amount_correct : total_amount_linda_gave = 25 := by
  sorry

end linda_total_amount_correct_l440_440930


namespace ben_fewer_pints_than_kathryn_l440_440395

-- Define the conditions
def annie_picked := 8
def kathryn_picked := annie_picked + 2
def total_picked := 25

-- Add noncomputable because constants are involved
noncomputable def ben_picked : ℕ := total_picked - (annie_picked + kathryn_picked)

theorem ben_fewer_pints_than_kathryn : ben_picked = kathryn_picked - 3 := 
by 
  -- The problem statement does not require proof body
  sorry

end ben_fewer_pints_than_kathryn_l440_440395


namespace slope_of_line_l440_440077

  theorem slope_of_line : 
    ∀ t : ℝ, (x y : ℝ), (x = 3 - t * Real.sin (20 * Real.pi / 180)) → 
              (y = 2 + t * Real.cos (70 * Real.pi / 180)) → 
              (y - 2) / (x - 3) = -1 := 
  by
    intros t x y hx hy
    rw [hx, hy]
    sorry
  
end slope_of_line_l440_440077


namespace roots_sum_of_cubic_eqn_l440_440970

noncomputable def cubic_eqn_roots_value : ℝ :=
  let α := (13 : ℝ)^(1 / 3)
  let β := (53 : ℝ)^(1 / 3)
  let γ := (103 : ℝ)^(1 / 3)
  let roots := [α, β, γ]
  let p := 1 / 3 
  let r := roots.nth 0 |>.get!
  let s := roots.nth 1 |>.get!
  let t := roots.nth 2 |>.get!
  r^3 + s^3 + t^3

theorem roots_sum_of_cubic_eqn :
  cubic_eqn_roots_value = 170 := 
sorry

end roots_sum_of_cubic_eqn_l440_440970


namespace problem_l440_440597

theorem problem (a b : ℝ) (h₁ : a = -a) (h₂ : b = 1 / b) : a + b = 1 ∨ a + b = -1 :=
  sorry

end problem_l440_440597


namespace probability_of_four_ones_l440_440245

noncomputable def probability_exactly_four_ones : ℚ :=
  (Nat.choose 12 4 * (1/6)^4 * (5/6)^8)

theorem probability_of_four_ones :
  abs (probability_exactly_four_ones.toReal - 0.114) < 0.001 :=
by
  sorry

end probability_of_four_ones_l440_440245


namespace neg_number_appears_l440_440181

theorem neg_number_appears (numbers : List ℤ) :
  ∃ n : ℕ, ∃ X Y ∈ numbers, ∃ (steps : ℕ → List ℤ),
    (steps 0 = numbers) ∧ 
    (∀ k, steps (k + 1) = let (a, b) := (choose2 (steps k)) in (a - 2) :: (b + 1) ::) ∧ 
    (∃ k, steps k ∃ x ∈ (steps k), x < 0) :=
sorry

end neg_number_appears_l440_440181


namespace jill_age_l440_440603

theorem jill_age (H J : ℕ) (h1 : H + J = 41) (h2 : H - 7 = 2 * (J - 7)) : J = 16 :=
by
  sorry

end jill_age_l440_440603


namespace smallest_integer_with_24_factors_and_factors_8_and_18_l440_440407

def is_factor (a b : ℕ) : Prop := ∃ k, b = k * a

theorem smallest_integer_with_24_factors_and_factors_8_and_18 :
  ∃ x : ℕ, (∀ d : ℕ, d ∣ x ↔ is_factor d x) ∧ (x.factors.count = 24) ∧ (is_factor 8 x) ∧ (is_factor 18 x) ∧ (∀ y : ℕ, (∀ d : ℕ, d ∣ y ↔ is_factor d y) ∧ (y.factors.count = 24) ∧ (is_factor 8 y) ∧ (is_factor 18 y) → x ≤ y) :=
begin
  sorry
end

end smallest_integer_with_24_factors_and_factors_8_and_18_l440_440407


namespace correct_equation_l440_440637

theorem correct_equation : 
    (∀ a : ℝ, sqrt ((-5)^2) ≠ -5) → 
    (∀ a : ℝ, 4 * a^4 ≠ a) → 
    (sqrt (7^2) = 7) → 
    (3 * (-π)^3 ≠ π) → 
    (sqrt (7^2) = 7) :=
by 
  intros h1 h2 h3 h4
  exact h3

end correct_equation_l440_440637


namespace solution_k_l440_440334

noncomputable def matrix_problem (n : ℕ) (A : Matrix ℝ ℝ) (k : ℝ) : Prop :=
  ∀ i j, i < n ∧ j < n ∧ A i j ≠ 0 → (A i j * k = sum (λ m, if m ≠ j then A i m else 0) (finset.range n) +
                                      sum (λ m, if m ≠ i then A m j else 0) (finset.range n))

theorem solution_k (n : ℕ) (A : Matrix ℝ ℝ) (h : n > 2) :
  (∃ k : ℝ, matrix_problem n A k) ↔ (∃ k : ℝ, k = 1 / (- n^2 + 4*n - 3)) :=
begin
  sorry,
end

end solution_k_l440_440334


namespace pascal_tenth_number_in_hundred_row_l440_440634

def pascal_row (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_tenth_number_in_hundred_row :
  pascal_row 99 9 = Nat.choose 99 9 :=
by
  sorry

end pascal_tenth_number_in_hundred_row_l440_440634


namespace two_interns_same_row_l440_440109

theorem two_interns_same_row {R : Type} [Fintype R] [DecidableEq R]
  (rows : Finset R) (morning_seats afternoon_seats : R → Finset (Fin 10))
  (interns : Finset (Fin 50)) :
  rows.card = 7 →
  ∀ i : Fin 50, 
    ∃ r_m r_a : R, r_m ∈ rows ∧ r_a ∈ rows ∧ i ∈ morning_seats r_m ∧ i ∈ afternoon_seats r_a →
  ∃ r : R, ∃ i₁ i₂ : Fin 50, i₁ ≠ i₂ ∧ 
    (i₁ ∈ morning_seats r ∧ i₂ ∈ morning_seats r) ∧
    (β₁ ∈ afternoon_seats r ∧ β₂ ∈ afternoon_seats r) :=
by
  sorry

end two_interns_same_row_l440_440109


namespace abby_bridget_adjacent_probability_l440_440344

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def probability_adjacent : ℚ :=
  let total_seats := 9
  let ab_adj_same_row_pairs := 9
  let ab_adj_diagonal_pairs := 4
  let favorable_outcomes := (ab_adj_same_row_pairs + ab_adj_diagonal_pairs) * 2 * factorial 7
  let total_outcomes := factorial total_seats
  favorable_outcomes / total_outcomes

theorem abby_bridget_adjacent_probability :
  probability_adjacent = 13 / 36 :=
by
  sorry

end abby_bridget_adjacent_probability_l440_440344


namespace handshakes_exchanged_l440_440180

-- Let n be the number of couples
noncomputable def num_couples := 7

-- Total number of people at the gathering
noncomputable def total_people := num_couples * 2

-- Number of people each person shakes hands with
noncomputable def handshakes_per_person := total_people - 2

-- Total number of unique handshakes
noncomputable def total_handshakes := total_people * handshakes_per_person / 2

theorem handshakes_exchanged :
  total_handshakes = 77 :=
by
  sorry

end handshakes_exchanged_l440_440180


namespace find_m_range_l440_440456

variable {α : Type*} [LinearOrder α] {β : Type*} [Preorder β]

theorem find_m_range (f : α → β) (hf : ∀ x y, x < y → f x < f y) 
  (m : α) (hm : f (m + 1) > f (2 * m - 1)) : m < 2 :=
by
  sorry

end find_m_range_l440_440456


namespace incorrect_analogies_l440_440588

theorem incorrect_analogies (a b c : ℂ) (ha : a ≠ 0) :
  ¬(|a|^2 = a^2) ∧ ¬((b^2 - 4 * a * c) > 0 → (∃ z1 z2 : ℂ, z1 ≠ z2 ∧ a * z1^2 + b * z1 + c = 0 ∧ a * z2^2 + b * z2 + c = 0)) :=
by
  sorry

end incorrect_analogies_l440_440588


namespace circle_line_distance_difference_l440_440822

/-- We define the given circle and line and prove the difference between maximum and minimum distances
    from any point on the circle to the line is 5√2. -/
theorem circle_line_distance_difference :
  (∀ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 10 = 0) →
  (∀ (x y : ℝ), x + y - 8 = 0) →
  ∃ (d : ℝ), d = 5 * Real.sqrt 2 :=
by
  sorry

end circle_line_distance_difference_l440_440822


namespace probability_ZD_greater_than_7_correct_l440_440237

noncomputable def probability_ZD_greater_than_7 : Prop :=
  let XYZ : Type := {triangle // right_triangle ∧ angleXYZ = 90 ∧ angleXZY = 45 ∧ XY = 14}
  let P_Inside : Point XYZ := {p // inside_triangle p XYZ}
  let extension : D := extend_ZP_to_XY P_Inside
  let ZD : Real := distance Z D
  P_random (Z D extends ZP to XY)
      (prob = Pr(D > 7))
     Pr(D > 14) = ½  


theorem probability_ZD_greater_than_7_correct : probability_ZD_greater_than_7 := 
by sorry 

end probability_ZD_greater_than_7_correct_l440_440237


namespace general_formula_sum_inequality_sum_intersection_inequality_l440_440142

-- Definitions of the sequence and sum as per the problem
def U : Set ℕ := { n | 1 ≤ n ∧ n ≤ 100 }

def a (n : ℕ) : ℕ := if n > 0 then 3^(n-1) else 0

def S (T : Set ℕ) : ℕ :=
  T.toFinset.fold (λ s t => s + a t) 0

-- Conditions given in the problem
axiom h_T2_4 : S {2, 4} = 30

-- Problem (1): General formula for the sequence
theorem general_formula :
  ∀ n, n > 0 → a n = 3 ^ (n - 1) := by
  sorry

-- Problem (2): S_T < a_{k+1} if T ⊆ {1, 2, ..., k}
theorem sum_inequality (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 100) (T : Set ℕ) (hT : T ⊆ {n | n > 0 ∧ n ≤ k}) :
  S T < a (k + 1) := by
  sorry

-- Problem (3): S_C + S_{C ∩ D} ≥ 2 S_D if S_C ≥ S_D
theorem sum_intersection_inequality (C D : Set ℕ) (hC : C ⊆ U) (hD : D ⊆ U) (hSC_ge_SD : S C ≥ S D) :
  S C + S (C ∩ D) ≥ 2 * S D := by
  sorry

end general_formula_sum_inequality_sum_intersection_inequality_l440_440142


namespace bob_has_winning_strategy_l440_440699

noncomputable def alice_and_bob_game_winner : String :=
  let grid_size := 100
  let max_cell_value := grid_size ^ 2
  let alice_starts := true
  let valid_numbers := Set.range (max_cell_value)
  let cells := Set.range (grid_size^2)
  let alice_score_conditions (grid: List (List ℕ)) : ℕ :=
    max (List.sum <$> grid)
  let bob_score_conditions (grid: List (List ℕ)) : ℕ :=
    let columns := List.transpose grid
    max (List.sum <$> columns)
  if bob_score_conditions (List.repeat (List.range grid_size) grid_size) >
      alice_score_conditions (List.repeat (List.range grid_size) grid_size) then "Bob has the winning strategy"
    else "No one has a winning strategy or it's a tie"

theorem bob_has_winning_strategy : 
       alice_and_bob_game_winner = "Bob has the winning strategy" :=
begin
  sorry
end

end bob_has_winning_strategy_l440_440699


namespace dice_probability_exactly_four_ones_l440_440258

noncomputable def dice_probability : ℚ := 
  (Nat.choose 12 4) * (1/6)^4 * (5/6)^8

theorem dice_probability_exactly_four_ones : (dice_probability : ℚ) ≈ 0.089 :=
  by sorry -- Skip the proof. 

#eval (dice_probability : ℚ)

end dice_probability_exactly_four_ones_l440_440258


namespace inequality_px_qy_l440_440480

theorem inequality_px_qy 
  (p q x y : ℝ) 
  (hp : 0 < p) 
  (hq : 0 < q) 
  (hpq : p + q < 1) 
  : (p * x + q * y) ^ 2 ≤ p * x ^ 2 + q * y ^ 2 := 
sorry

end inequality_px_qy_l440_440480


namespace exists_super_number_B_l440_440300

-- Define a function is_super_number to identify super numbers.
def is_super_number (A : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 ≤ A n ∧ A n < 10

-- Define a function zero_super_number to represent the super number with all digits zero.
def zero_super_number (n : ℕ) := 0

-- Task: Prove the existence of B such that A + B = zero_super_number.
theorem exists_super_number_B (A : ℕ → ℕ) (hA : is_super_number A) :
  ∃ B : ℕ → ℕ, is_super_number B ∧ (∀ n : ℕ, (A n + B n) % 10 = zero_super_number n) :=
sorry

end exists_super_number_B_l440_440300


namespace locus_line_segment_l440_440210

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def is_locus (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (k : ℝ) : Prop :=
  dist P F1 + dist P F2 = k

theorem locus_line_segment :
  ∀ P : ℝ × ℝ,
  is_locus P (-3, 0) (3, 0) 6 → 
  ∃ x : ℝ, P = (x, 0) ∧ -3 ≤ x ∧ x ≤ 3 :=
by
  intros P h
  -- Proof would go here
  sorry

end locus_line_segment_l440_440210


namespace point_A_in_third_quadrant_l440_440171

theorem point_A_in_third_quadrant :
  let A : ℂ := complex.mk (real.cos (2023 * real.pi / 180)) (real.tan 8)
  in A.re < 0 ∧ A.im < 0 → 
     -- Statement describing that point A is in the third quadrant
     (A.re < 0 ∧ A.im < 0) := 
by {
  intro h,
  assumption
}

end point_A_in_third_quadrant_l440_440171


namespace product_of_square_roots_l440_440482

theorem product_of_square_roots (a b : ℝ) (h₁ : a^2 = 9) (h₂ : b^2 = 9) (h₃ : a ≠ b) : a * b = -9 :=
by
  -- Proof skipped
  sorry

end product_of_square_roots_l440_440482


namespace value_of_x_l440_440654

theorem value_of_x (x : ℝ) (h : 0.5 * x - (1 / 3) * x = 110) : x = 660 :=
sorry

end value_of_x_l440_440654


namespace find_income_4_l440_440304

noncomputable def income_4 (income_1 income_2 income_3 income_5 average_income num_days : ℕ) : ℕ :=
  average_income * num_days - (income_1 + income_2 + income_3 + income_5)

theorem find_income_4
  (income_1 : ℕ := 200)
  (income_2 : ℕ := 150)
  (income_3 : ℕ := 750)
  (income_5 : ℕ := 500)
  (average_income : ℕ := 400)
  (num_days : ℕ := 5) :
  income_4 income_1 income_2 income_3 income_5 average_income num_days = 400 :=
by
  unfold income_4
  sorry

end find_income_4_l440_440304


namespace repeating_decimal_product_l440_440710

theorem repeating_decimal_product :
  let x : ℚ := 456 / 999
  in (x * 11 = 1672 / 333) :=
by
  sorry

end repeating_decimal_product_l440_440710


namespace train_lengths_l440_440692

noncomputable def train_problem : Prop :=
  let speed_T1_mps := 54 * (5/18)
  let speed_T2_mps := 72 * (5/18)
  let L_T1 := speed_T1_mps * 20
  let L_p := (speed_T1_mps * 44) - L_T1
  let L_T2 := speed_T2_mps * 16
  (L_p = 360) ∧ (L_T1 = 300) ∧ (L_T2 = 320)

theorem train_lengths : train_problem := sorry

end train_lengths_l440_440692


namespace volume_eq_three_times_other_two_l440_440865

-- declare the given ratio of the radii
def r1 : ℝ := 1
def r2 : ℝ := 2
def r3 : ℝ := 3

-- calculate the volumes based on the given radii
noncomputable def V (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- defining the volumes of the three spheres
noncomputable def V1 : ℝ := V r1
noncomputable def V2 : ℝ := V r2
noncomputable def V3 : ℝ := V r3

theorem volume_eq_three_times_other_two : V3 = 3 * (V1 + V2) := 
by
  sorry

end volume_eq_three_times_other_two_l440_440865


namespace arrangement_count_l440_440682

-- Definitions based on the conditions in the problem
def classes := {1, 2, 3, 4, 5}
def factories := {a, b, c, d}

-- Each class should be assigned to one distinct factory
-- and each factory must have at least one class.

theorem arrangement_count :
  (∃ (f : classes → factories), 
    (∀ (c₁ c₂ : classes), c₁ ≠ c₂ → f c₁ ≠ f c₂) ∧  -- different classes go to different factories
    (∀ (fac : factories), ∃ (c : classes), f c = fac))  -- each factory has at least one class
  → 240 :=
sorry

end arrangement_count_l440_440682


namespace parallelogram_area_l440_440537

def vector1 := ⟨4, 7⟩ : ℝ × ℝ
def vector2 := ⟨-6, 3⟩ : ℝ × ℝ

theorem parallelogram_area : 
    abs (vector1.1 * vector2.2 - vector1.2 * vector2.1) = 54 := 
by
    sorry

end parallelogram_area_l440_440537


namespace cos_alpha_plus_7pi_over_6_l440_440804

variable {α : Real}

theorem cos_alpha_plus_7pi_over_6
  (h : cos (α - π / 6) - sin α = 2 * sqrt 3 / 5) :
  cos (α + 7 * π / 6) = -2 * sqrt 3 / 5 :=
sorry

end cos_alpha_plus_7pi_over_6_l440_440804


namespace find_a_l440_440531

theorem find_a (a : ℝ) :
  let A := {5}
  let B := { x : ℝ | a * x - 1 = 0 }
  A ∩ B = B ↔ (a = 0 ∨ a = 1 / 5) :=
by
  sorry

end find_a_l440_440531


namespace cesaro_lupu_real_analysis_l440_440904

noncomputable def proof_problem (a b c x y z : ℝ) : Prop :=
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) ∧
  (0 < x) ∧ (0 < y) ∧ (0 < z) ∧
  (a^x = b * c) ∧ (b^y = c * a) ∧ (c^z = a * b) →
  (1 / (2 + x) + 1 / (2 + y) + 1 / (2 + z) ≤ 3 / 4)

theorem cesaro_lupu_real_analysis (a b c x y z : ℝ) :
  proof_problem a b c x y z :=
by sorry

end cesaro_lupu_real_analysis_l440_440904


namespace greene_family_amusement_park_spending_l440_440964

def spent_on_admission : ℝ := 45
def original_ticket_cost : ℝ := 50
def spent_less_than_original_cost_on_food_and_beverages : ℝ := 13
def spent_on_souvenir_Mr_Greene : ℝ := 15
def spent_on_souvenir_Mrs_Greene : ℝ := 2 * spent_on_souvenir_Mr_Greene
def cost_per_game : ℝ := 9
def number_of_children : ℝ := 3
def spent_on_transportation : ℝ := 25
def tax_rate : ℝ := 0.08

def food_and_beverages_cost : ℝ := original_ticket_cost - spent_less_than_original_cost_on_food_and_beverages
def games_cost : ℝ := number_of_children * cost_per_game
def taxable_amount : ℝ := food_and_beverages_cost + spent_on_souvenir_Mr_Greene + spent_on_souvenir_Mrs_Greene + games_cost
def tax : ℝ := tax_rate * taxable_amount
def total_expenditure : ℝ := spent_on_admission + food_and_beverages_cost + spent_on_souvenir_Mr_Greene + spent_on_souvenir_Mrs_Greene + games_cost + spent_on_transportation + tax

theorem greene_family_amusement_park_spending : total_expenditure = 187.72 :=
by {
  sorry
}

end greene_family_amusement_park_spending_l440_440964


namespace probability_red_ball_bins_1_to_3_l440_440717

-- Define the necessary probability and the zeta function approximation 
def probability_in_bin (k : ℕ) : ℝ := k⁻² / (Real.pi ^ 2 / 6)
def probability_in_bins_1_to_3 : ℝ := (probability_in_bin 1) + (probability_in_bin 2) + (probability_in_bin 3)

-- The main theorem stating the final probability
theorem probability_red_ball_bins_1_to_3 : probability_in_bins_1_to_3 = 31 / Real.pi ^ 2 :=
by
  sorry

end probability_red_ball_bins_1_to_3_l440_440717


namespace domain_of_f_l440_440586

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 1)) / (x^2 + x)

theorem domain_of_f :
  {x : ℝ | (x + 1 ≥ 0) ∧ (x^2 + x ≠ 0)} = set.Union (set.Ioo (-1 : ℝ) 0) (set.Ioi 0) :=
by
  sorry

end domain_of_f_l440_440586


namespace area_of_abe_l440_440522

theorem area_of_abe (ABC_area : ℝ) (A B C D E F : Type*)
    [triangle ABC] [location D AB] [location E BC] [location F CA]
    (AD DB EF : ℝ) (H1 : ABC_area = 18) (H2 : AD = 1) (H3 : DB = 4) 
    (H4 : EF = AD) (H5 : area_of_triangle ABE = area_of_quad DBEF) :
    area_of_triangle ABE = 7.2 :=
sorry

end area_of_abe_l440_440522


namespace water_tank_empty_time_l440_440343

theorem water_tank_empty_time :
  let tank_full := 1
      tank_initial := 7 / 11
      pipe_A_fill_time := 15
      pipe_B_empty_time := 8
      pipe_C_fill_time := 20
      pipe_A_rate := 1 / pipe_A_fill_time
      pipe_B_rate := -1 / pipe_B_empty_time
      pipe_C_rate := 1 / pipe_C_fill_time
      net_rate := pipe_A_rate + pipe_B_rate + pipe_C_rate
      emptying_time := tank_initial / -net_rate
  in emptying_time ≈ 76.36 :=
by
  -- Leap statement checks
  let tank_full := 1
  let tank_initial := 7 / 11
  let pipe_A_fill_time := 15
  let pipe_B_empty_time := 8
  let pipe_C_fill_time := 20
  have pipe_A_rate : ℝ := 1 / pipe_A_fill_time
  have pipe_B_rate : ℝ := -1 / pipe_B_empty_time
  have pipe_C_rate : ℝ := 1 / pipe_C_fill_time
  have net_rate : ℝ := pipe_A_rate + pipe_B_rate + pipe_C_rate
  have emptying_time : ℝ := tank_initial / -net_rate
  have h : emptying_time ≈ 76.36 := by sorry
  exact h

end water_tank_empty_time_l440_440343


namespace total_items_at_bakery_l440_440609

theorem total_items_at_bakery (bread_rolls : ℕ) (croissants : ℕ) (bagels : ℕ) (h1 : bread_rolls = 49) (h2 : croissants = 19) (h3 : bagels = 22) : bread_rolls + croissants + bagels = 90 :=
by
  sorry

end total_items_at_bakery_l440_440609


namespace fixed_point_1_3_l440_440971

noncomputable def fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Prop :=
  (f (1) = 3) where f x := a^(x-1) + 2

theorem fixed_point_1_3 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : fixed_point a h1 h2 :=
by
  unfold fixed_point
  sorry

end fixed_point_1_3_l440_440971


namespace dice_sum_probability_l440_440769

theorem dice_sum_probability :
  let D := finset.range 1 7  -- outcomes of a fair six-sided die
  (∃! d1 d2 d3 d4 ∈ D, d1 + d2 + d3 + d4 = 24) ->
  (probability(space, {ω ∈ space | (ω 1 = 6) ∧ (ω 2 = 6) ∧ (ω 3 = 6) ∧ (ω 4 = 6)}) = 1/1296) :=
sorry

end dice_sum_probability_l440_440769


namespace remainder_xy_mod_n_l440_440148

variable (n : ℕ) (x y : ℤ)
variables [n > 0] [x ∈ Units (ZMod n)] [y ∈ Units (ZMod n)]

theorem remainder_xy_mod_n :
  (x ≡ 2 * y [ZMod n]) →
  (y ≡ 3 * x⁻¹ [ZMod n]) →
  (x * y ≡ 3 [ZMod n]) :=
sorry

end remainder_xy_mod_n_l440_440148


namespace percentage_of_exceedance_l440_440324

theorem percentage_of_exceedance (x p : ℝ) (h : x = (p / 100) * x + 52.8) (hx : x = 60) : p = 12 :=
by 
  sorry

end percentage_of_exceedance_l440_440324


namespace probability_exactly_four_ones_is_090_l440_440241
open Float (approxEq)

def dice_probability_exactly_four_ones : Float :=
  let n := 12
  let k := 4
  let p_one := (1 / 6 : Float)
  let p_not_one := (5 / 6 : Float)
  let combination := ((n.factorial) / (k.factorial * (n - k).factorial) : Float)
  let probability := combination * (p_one ^ k) * (p_not_one ^ (n - k))
  probability

theorem probability_exactly_four_ones_is_090 : dice_probability_exactly_four_ones ≈ 0.090 :=
  sorry

end probability_exactly_four_ones_is_090_l440_440241


namespace find_p_plus_r_l440_440205

-- We introduce variables and the conditions identified.

variables (p q r s : ℝ)

-- The conditions provided by the intersection points
def intersects_at (f g : ℝ -> ℝ) (a b : ℝ × ℝ) : Prop :=
  f a.1 = a.2 ∧ g a.1 = a.2 ∧ f b.1 = b.2 ∧ g b.1 = b.2

-- Define the two functions given in the problem
def f (x : ℝ) := -|x - p| + q
def g (x : ℝ) := |x - r| + s

-- Define the proof problem
theorem find_p_plus_r (h1 : intersects_at (f p q) (g r s) (3, 6))
                      (h2 : intersects_at (f p q) (g r s) (5, 2)) :
  p + r = 8 :=
sorry

end find_p_plus_r_l440_440205


namespace remainder_of_divisors_mod_2010_l440_440907

-- Define the number 2010 and its powers
def number := 2010

-- Define the power to which the number is raised
def power := 2010

-- Define what it means to be a divisor ending in 2
def is_divisor_ending_in_2 (d : ℕ) : Prop :=
  d ∣ number ^ power ∧ d % 10 = 2

-- Define the number of such divisors
def N : ℕ := (finset.range (number ^ power + 1)).filter is_divisor_ending_in_2 |>.card

-- Define the required properties for the proof
theorem remainder_of_divisors_mod_2010 : N % 2010 = 503 := by
  sorry

end remainder_of_divisors_mod_2010_l440_440907


namespace correct_statements_of_f_l440_440916

noncomputable def f : ℝ → ℝ :=
sorry -- This part should be the actual definition of f, provided it fulfills all conditions

theorem correct_statements_of_f :
  (∀ x : ℝ, f(x + 1) = f(x - 1)) ∧ 
  (∀ x ∈ [0, 1), f x = real.log 0.5 (1 - x)) ∧
  (∀ x, f x = f (-x)) → 
  (∃ k : ℝ, ∀ x, f (x + 2) = f x) ∧
  (∀ x ∈ (3, 4), f x = real.log 0.5 (x - 3)) :=
by
  sorry

end correct_statements_of_f_l440_440916


namespace intersection_product_eq_one_l440_440890

noncomputable def parametric_curve_C (β : ℝ) : ℝ × ℝ :=
  (2 * Real.cos β, 2 * Real.sin β + 1)

def polar_line_l (ρ α : ℝ) : Prop :=
  ρ * Real.cos (α - Real.pi / 4) = Real.sqrt 2

theorem intersection_product_eq_one :
  (∀ β : ℝ, ∃ (x y : ℝ), parametric_curve_C β = (x, y)) ∧
  (∃ ρ α : ℝ, polar_line_l ρ α) →
  ∃ P M N : ℝ × ℝ,
    let (mx, my) := M in
    let (nx, ny) := N in
    let (px, py) := P in
    x + y - 2 = 0 ∧  -- line l's rectangular equation
    (P = (2, 0)) ∧
    (x = mx ∧ y = my) ∧
    (x = nx ∧ y = ny) ∧
    Real.abs (P.1 - M.1) * Real.abs (N.2 - P.2) = 1 := sorry

end intersection_product_eq_one_l440_440890


namespace real_roots_of_quad_eq_l440_440571

theorem real_roots_of_quad_eq (p q a : ℝ) (h : p^2 - 4 * q > 0) : 
  (2 * a - p)^2 + 3 * (p^2 - 4 * q) > 0 := 
by
  sorry

end real_roots_of_quad_eq_l440_440571


namespace ruby_candies_distribution_l440_440574

theorem ruby_candies_distribution (candies : ℕ) (friends : ℕ) (h1 : candies = 36) (h2 : friends = 9) :
  candies / friends = 4 :=
by {
  rw [h1, h2],
  norm_num
}

end ruby_candies_distribution_l440_440574


namespace shaded_region_perimeter_l440_440873

theorem shaded_region_perimeter (r : ℝ) (θ : ℝ) (h₁ : r = 2) (h₂ : θ = 90) : 
  (2 * r + (2 * π * r * (1 - θ / 180))) = π + 4 := 
by sorry

end shaded_region_perimeter_l440_440873


namespace EveIs14_l440_440696

noncomputable def EveAge : ℕ :=
  let AdamAge := 9
  let AgeDifference := 5
  (AdamAge + AgeDifference)

theorem EveIs14 : EveAge = 14 := by
  have AdamAge : ℕ := 9
  have AgeDifference : ℕ := 5
  have EveAge : ℕ := AdamAge + AgeDifference
  calc EveAge = AdamAge + AgeDifference : by rfl
         ... = 9 + 5 : by rw [AdamAge, AgeDifference]
         ... = 14 : by norm_num
  done

end EveIs14_l440_440696


namespace geometric_series_sum_l440_440365

theorem geometric_series_sum :
  let a := (1 : ℚ) / 3
  let r := (-1 : ℚ) / 4
  let n := 6
  let S6 := a * (1 - r^n) / (1 - r)
  S6 = 4095 / 30720 :=
by
  sorry

end geometric_series_sum_l440_440365


namespace probability_three_aligned_l440_440425

theorem probability_three_aligned (total_arrangements favorable_arrangements : ℕ) 
  (h1 : total_arrangements = 126)
  (h2 : favorable_arrangements = 48) :
  (favorable_arrangements : ℚ) / total_arrangements = 8 / 21 :=
by sorry

end probability_three_aligned_l440_440425


namespace A_should_play_against_B_first_l440_440695

-- Define the probabilities and assumptions
variables (p_A_B p_A_C p_B_C : ℝ)
variables (h1 : p_A_B < p_A_C) (h2 : p_A_B < p_B_C)

-- Theorem that A should choose to play against B first
theorem A_should_play_against_B_first : 
  A_should_choose_play_first p_A_B p_A_C p_B_C :=
sorry

end A_should_play_against_B_first_l440_440695


namespace c_100_value_l440_440892

-- Define the sequence \{a_n\}
def a : ℕ → ℤ
| 1 := 2
| (n + 2) := a n - 3

-- Define the quadratic equation and its coefficients
noncomputable def c (n : ℕ) : ℤ := a n * a (n + 1)

-- Calculate the value of c_100 and state the theorem
theorem c_100_value : c 100 = 22496 := by
  sorry

end c_100_value_l440_440892


namespace conjugate_of_z_l440_440816

open Complex

theorem conjugate_of_z (z : ℂ) (h : (1 + z) * I = 3 - I) : conj z = -2 + 3 * I := sorry

end conjugate_of_z_l440_440816


namespace problem_1_l440_440788

variable {t : ℝ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {T : ℕ → ℝ}

theorem problem_1 (h1: a 1 = 1) 
                  (h2: t = 5) 
                  (h3: ∀ n, 6 * S n = (a n)^2 + 3 * a n + 2) 
                  (h4: b 1 = 1) 
                  (h5: ∀ n, b (n + 1) - b n = a (n + 1)) :
  (∀ n: ℕ, n ≥ 1 → a n = 3 * n - 2) ∧ 
  (∀ n: ℕ, n ≥ 1 → T n = ∑ k in Finset.range n, (1 / (2 * b (k + 1) + 7 * (k + 1))) = 
                  (3 * n^2 + 5 * n) / (12 * (n + 1) * (n + 2))) :=
sorry

end problem_1_l440_440788


namespace binomial_expansion_m3n6_l440_440629

theorem binomial_expansion_m3n6 :
  (∑ k in Finset.range (10), Nat.choose 9 k * m^k * n^(9-k)) = (∑ k in Finset.range (10), if k = 3 then 84 else 0) :=
by
  sorry

end binomial_expansion_m3n6_l440_440629


namespace magnitude_proof_l440_440835

-- Define vectors a and b
def a : ℝ × ℝ × ℝ := (3, 5, 1)
def b : ℝ × ℝ × ℝ := (2, 2, 3)

-- Define scalar operations on vectors
def scalar_mul (s : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (s * v.1, s * v.2, s * v.3)

-- Define vector subtraction
def vector_sub (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2, v.3 - w.3)

-- Define vector magnitude calculation
def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- To prove: |2a - 3b| = sqrt 65
theorem magnitude_proof : vector_magnitude (vector_sub (scalar_mul 2 a) (scalar_mul 3 b)) = Real.sqrt 65 :=
by
  sorry  -- proof to be completed

end magnitude_proof_l440_440835


namespace derivative_product_l440_440779

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x else x - x^2

def f' (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x + 1 else 1 - 2 * x

theorem derivative_product:
  f' 1 * f' (-1) = 9 :=
by
  -- assume the required proofs are done here
  sorry

end derivative_product_l440_440779


namespace coeff_x_squared_l440_440628

theorem coeff_x_squared :
  let p := (2 - x + x^2) * (1 + 2 * x) ^ 6 in
  (coeff (p : ℚ[x]) 2) = 69 := by sorry

end coeff_x_squared_l440_440628


namespace tan_double_angle_l440_440457

theorem tan_double_angle (θ : ℝ) (P : ℝ × ℝ) 
  (h_vertex : θ = 0) 
  (h_initial_side : ∀ x, θ = x)
  (h_terminal_side : P = (-1, 2)) : 
  Real.tan (2 * θ) = 4 / 3 := 
by 
  sorry

end tan_double_angle_l440_440457


namespace total_cookies_l440_440941

def MonaCookies : ℕ := 20
def JasmineCookies : ℕ := MonaCookies - 5
def RachelCookies : ℕ := JasmineCookies + 10

theorem total_cookies : MonaCookies + JasmineCookies + RachelCookies = 60 := by
  -- Since we don't need to provide the solution steps, we simply use sorry.
  sorry

end total_cookies_l440_440941


namespace find_common_ratio_l440_440051

noncomputable def geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 5 - a 1 = 15) ∧ (a 4 - a 2 = 6) → (q = 1/2 ∨ q = 2)

-- We declare this as a theorem statement
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) : geometric_sequence_common_ratio a q :=
sorry

end find_common_ratio_l440_440051


namespace find_m_condition_l440_440839

theorem find_m_condition : 
  ∃ m : ℝ, real.sqrt ((m - 1)^2 + (-3)^2) = real.sqrt ((m + 1)^2 + (-1)^2) ∧ m = 2 :=
begin
  sorry
end

end find_m_condition_l440_440839


namespace girl_scouts_short_amount_l440_440204

-- Definitions based on conditions
def amount_earned : ℝ := 30
def pool_entry_cost_per_person : ℝ := 2.50
def num_people : ℕ := 10
def transportation_fee_per_person : ℝ := 1.25
def snack_cost_per_person : ℝ := 3.00

-- Calculate individual costs
def total_pool_entry_cost : ℝ := pool_entry_cost_per_person * num_people
def total_transportation_fee : ℝ := transportation_fee_per_person * num_people
def total_snack_cost : ℝ := snack_cost_per_person * num_people

-- Calculate total expenses
def total_expenses : ℝ := total_pool_entry_cost + total_transportation_fee + total_snack_cost

-- The amount left after expenses
def amount_left : ℝ := amount_earned - total_expenses

-- Proof problem statement
theorem girl_scouts_short_amount : amount_left = -37.50 := by
  sorry

end girl_scouts_short_amount_l440_440204


namespace asymptote_tangent_circle_l440_440124

-- Definitions of the hyperbola and circle
def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos: 0 < b) : Prop :=
  ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1

def circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 1

-- Tangency condition between the asymptote and the circle
def tangency_condition (a b : ℝ) (a_pos : 0 < a) (b_pos: 0 < b) : Prop :=
  abs(2 * a - b) / real.sqrt (a^2 + b^2) = 1

-- Main proposition: proving the ratio b/a = 3/4 under given conditions
theorem asymptote_tangent_circle
  (a b : ℝ) (a_pos : 0 < a) (b_pos: 0 < b)
  (hyp_hyperbola : hyperbola a b a_pos b_pos)
  (hyp_circle : circle 2 1)
  (hyp_tangency_condition : tangency_condition a b a_pos b_pos) :
  b / a = 3 / 4 :=
by {
  sorry
}

end asymptote_tangent_circle_l440_440124


namespace two_lines_perpendicular_to_same_plane_are_parallel_l440_440447

def Line : Type := sorry
def Plane : Type := sorry

variables (m n : Line) (α β γ : Plane)

-- Given conditions
axiom different_lines : m ≠ n
axiom different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ
axiom m_perp_α : m ⊥ α
axiom n_perp_α : n ⊥ α

-- Proof problem
theorem two_lines_perpendicular_to_same_plane_are_parallel :
  m ⊥ α → n ⊥ α → m ∥ n :=
sorry

end two_lines_perpendicular_to_same_plane_are_parallel_l440_440447


namespace trig_identity_proof_l440_440387

theorem trig_identity_proof :
  sin (50 * Real.pi / 180) * cos (20 * Real.pi / 180) - 
  sin (40 * Real.pi / 180) * cos (70 * Real.pi / 180) = 1 / 2 :=
by sorry

end trig_identity_proof_l440_440387


namespace future_age_ratio_l440_440612

theorem future_age_ratio (j e x : ℕ) 
  (h1 : j - 3 = 5 * (e - 3)) 
  (h2 : j - 7 = 6 * (e - 7)) 
  (h3 : x = 17) : (j + x) / (e + x) = 3 := 
by
  sorry

end future_age_ratio_l440_440612


namespace extremum_point_exists_contradiction_assumption_l440_440636

theorem extremum_point_exists (a b : ℝ) : 
  ¬ (∀ x : ℝ, f' x ≠ 0) → ∃ x : ℝ, f' x = 0 :=
by
  sorry

noncomputable def f (x : ℝ) : ℝ := x^3 + a * x + b

noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + a

theorem contradiction_assumption (a b : ℝ) : 
  (∀ x : ℝ, f' x ≠ 0) → false :=
by
  sorry

end extremum_point_exists_contradiction_assumption_l440_440636


namespace proof_problem_l440_440837

noncomputable def problem_statement (a b c : ℝ × ℝ) (α β : ℝ) 
  (θ₁ θ₂ : ℝ) (A B C : ℝ) (circumradius : ℝ) : Prop :=
  (a = (1 + (Real.cos α), (Real.sin α))) ∧ 
  (b = (1 - (Real.cos β), (Real.sin β))) ∧ 
  (c = (1, 0)) ∧ 
  (α ∈ Set.Ioo 0 Real.pi) ∧ 
  (β ∈ Set.Ioo Real.pi (2 * Real.pi)) ∧ 
  (A = β - α) ∧ 
  (Real.angle_between a c = θ₁) ∧ 
  (Real.angle_between b c = θ₂) ∧ 
  (θ₁ - θ₂ = Real.pi / 6) ∧ 
  (circumradius = 4 * Real.sqrt 3) →
  (A = 2 * Real.pi / 3) ∧ 
  (∀ (b c : ℝ), b + c ∈ (Set.Ioc 12 (8 * Real.sqrt 3)))

theorem proof_problem (a b c : ℝ × ℝ) (α β : ℝ) (θ₁ θ₂ : ℝ) (A B C : ℝ) (circumradius : ℝ) :
  problem_statement a b c α β θ₁ θ₂ A B C circumradius :=
by sorry

end proof_problem_l440_440837


namespace tank_width_is_12_l440_440338

-- Define the conditions
def tank_length : ℝ := 25
def tank_depth : ℝ := 6
def plastering_cost : ℝ := 223.2
def cost_per_square_meter : ℝ := 0.30
def total_area_plastered : ℝ := 744

-- Define the bottom area in terms of width
def bottom_area (width : ℝ) : ℝ := tank_length * width

-- Define the walls area in terms of width
def walls_area (width : ℝ) : ℝ :=
  2 * (tank_length * tank_depth) + 2 * (width * tank_depth)

-- Define the total area function in terms of width
def total_area (width : ℝ) : ℝ :=
  bottom_area(width) + walls_area(width)

-- Lean theorem stating the width of the tank is 12 meters given the conditions
theorem tank_width_is_12 :
  (∀ width : ℝ, plastering_cost = total_area_plastered * cost_per_square_meter → total_area_plastered = total_area width) →
  ∃ width : ℝ, width = 12 := 
by
  sorry

end tank_width_is_12_l440_440338


namespace part1_period_and_dec_intervals_part2_max_min_l440_440829

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (2 * x + Real.pi / 6) + 2 * (Real.cos x)^2

theorem part1_period_and_dec_intervals :
  ∀ k : ℤ, 
  ∃ T : ℝ,
  ∀ x, f(x) = f(x + T) ∧ T = Real.pi ∧ 
  (T > 0 → k * T + Real.pi / 12 ≤ x ∧ x ≤ k * T + 7 * Real.pi / 12) :=
sorry

theorem part2_max_min (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  ∃ max min: ℝ,
  (∀ y : ℝ, y = f(x) → y ≤ max ∧ y ≥ min) ∧ 
  max = sqrt(3) + 1 ∧ min = -1 / 2 :=
sorry

end part1_period_and_dec_intervals_part2_max_min_l440_440829


namespace coeff_x4_expansion_l440_440493

def binom_expansion (a : ℚ) : ℚ :=
  let term1 : ℚ := a * 28
  let term2 : ℚ := -56
  term1 + term2

theorem coeff_x4_expansion (a : ℚ) : (binom_expansion a = -42) → a = 1/2 := 
by 
  intro h
  -- continuation of proof will go here.
  sorry

end coeff_x4_expansion_l440_440493


namespace find_probability_l440_440330

noncomputable def prob_roots_condition : ℝ :=
  let k_segment := set.Icc 6 11 in
  let f (k : ℝ) := ((k^2 - 2*k - 15) * (x^2) + (3*k - 7) * x + 2 = 0) in
  let condition (x1 x2 : ℝ) := x1 ≤ 2*x2 in
  let probability := (interval_integral (λ k, if ∃ x1 x2, f k ∧ condition x1 x2 then 1 else 0) 6 (23/3) / interval_integral (λ k, 1) 6 11) in
  probability

theorem find_probability : prob_roots_condition = 1 / 3 :=
  sorry

end find_probability_l440_440330


namespace tan_sum_pi_over_four_l440_440056

theorem tan_sum_pi_over_four (α : ℝ) (h1 : Real.sin α = 3 / 5) (h2 : α ∈ Ioo (π / 2) π) :
    Real.tan (α + π / 4) = 1 / 7 := sorry

end tan_sum_pi_over_four_l440_440056


namespace time_spent_on_road_l440_440640

theorem time_spent_on_road (Total_time_hours Stop1_minutes Stop2_minutes Stop3_minutes : ℕ) 
  (h1: Total_time_hours = 13) 
  (h2: Stop1_minutes = 25) 
  (h3: Stop2_minutes = 10) 
  (h4: Stop3_minutes = 25) : 
  Total_time_hours - (Stop1_minutes + Stop2_minutes + Stop3_minutes) / 60 = 12 :=
by
  sorry

end time_spent_on_road_l440_440640


namespace tara_total_gas_cost_l440_440189

def tara_first_day_cost (distance : ℕ) (efficiency : ℕ) (tank_capacity : ℕ) (price_per_gallon : ℕ) : ℕ :=
  let used_gallons := distance / efficiency
  tank_capacity * price_per_gallon

def tara_second_day_cost (distance : ℕ) (efficiency : ℕ) (tank_capacity : ℕ) (price_per_gallon : ℕ) : ℕ :=
  let used_gallons := distance / efficiency
  tank_capacity * price_per_gallon

theorem tara_total_gas_cost :
  tara_first_day_cost 300 30 12 3 + tara_second_day_cost 375 25 15 3.5.to_nat = 88.5.to_nat := 
sorry

end tara_total_gas_cost_l440_440189


namespace find_x_when_y_is_10_l440_440451

-- Definitions of inverse proportionality and initial conditions
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k

-- Given constants
def k : ℝ := 160
def x_initial : ℝ := 40
def y_initial : ℝ := 4

-- Theorem statement to prove the value of x when y = 10
theorem find_x_when_y_is_10 (h : inversely_proportional x_initial y_initial k) : 
  ∃ (x : ℝ), inversely_proportional x 10 k :=
sorry

end find_x_when_y_is_10_l440_440451


namespace pegs_arrangement_count_l440_440963

def num_ways_to_arrange_pegs : ℕ :=
  (nat.factorial 6) * (nat.factorial 5) * (nat.factorial 4) * (nat.factorial 3) * (nat.factorial 2)

theorem pegs_arrangement_count :
  num_ways_to_arrange_pegs = 12441600 :=
by
  unfold num_ways_to_arrange_pegs
  rw [nat.factorial, nat.factorial, nat.factorial, nat.factorial, nat.factorial]
  norm_num
  sorry

end pegs_arrangement_count_l440_440963


namespace problem_l440_440852

def f (x : ℤ) : ℤ := 7 * x - 3

theorem problem : f (f (f 3)) = 858 := by
  sorry

end problem_l440_440852


namespace find_m_l440_440080

theorem find_m (m : ℝ) (A : Set ℝ) (B : Set ℝ) (hA : A = { -1, 2, 2 * m - 1 }) (hB : B = { 2, m^2 }) (hSubset : B ⊆ A) : m = 1 := by
  sorry
 
end find_m_l440_440080


namespace trajectory_of_B_l440_440452

-- Conditions
def parallelogram (A B C D : Point) : Prop := -- Define parallelogram property (place holder)
sorry

noncomputable def Point := (ℝ × ℝ)

def A : Point := (3, -1)
def C : Point := (2, -3)
def line_D (D : Point) : Prop := 3 * D.1 - D.2 + 1 = 0

-- Statement
theorem trajectory_of_B :
  ∀ (B D : Point), 
    parallelogram A B C D → 
    line_D D → 
    3 * B.1 - B.2 - 20 = 0 :=
by
  intros,
  sorry

end trajectory_of_B_l440_440452


namespace consecutive_primes_sum_l440_440013

theorem consecutive_primes_sum 
    (p1 p2 p3 p4 : ℕ)
    (prime p1) (prime p2) (prime p3) (prime p4) 
    (13 < p1) (p1 < p2) (p2 < p3) (p3 < p4) 
    (h : ∑ i in [p1, p2, p3, p4], i % 4 = 0)
    (h_consec : p2 = next_prime p1 ∧
                p3 = next_prime p2 ∧
                p4 = next_prime p3) :
    ∑ i in [p1, p2, p3, p4], i = 88 :=
sorry

end consecutive_primes_sum_l440_440013


namespace probability_of_score_is_5_l440_440871

-- Definitions and conditions
noncomputable def total_outcomes := 2 ^ 3
noncomputable def favorable_outcomes := 3

-- Hypothesis: the number of outcomes yielding a total score of 5 points
theorem probability_of_score_is_5 : 
  let p := (favorable_outcomes : ℚ) / total_outcomes in
  p = 3 / 8 :=
by
  sorry

end probability_of_score_is_5_l440_440871


namespace walking_distance_l440_440625

theorem walking_distance (west east : ℤ) (h_west : west = 5) (h_east : east = -5) : west + east = 10 := 
by 
  rw [h_west, h_east] 
  sorry

end walking_distance_l440_440625


namespace O_N_C_l440_440359

-- Definitions based on conditions
variables
  (circle : Type)
  (O A B C M N : circle)
  (C' : circle)

-- Conditions
def tangent_at (pt : circle) := sorry
def perpendicular (a b c : circle) := sorry
def symmetric (a b : circle) := sorry
def divides_in_ratio (n a b : circle) (ratio : ℚ) := sorry
def collinear (a b c : circle) := sorry

-- Given definitions
axiom tangent_tangent_at_A : tangent_at A
axiom B_M_perpendicular : perpendicular B M A
axiom B_intersects_circle_at_C : intersects B (circle) C
axiom N_divides_AB_ratio : divides_in_ratio N A B (1/2)
axiom C_symmetric_to_M : symmetric C C'

-- The proof statement
theorem O_N_C'_collinear :
  collinear O N C' :=
sorry

end O_N_C_l440_440359


namespace number_of_valid_integers_between_1_and_300_l440_440845

theorem number_of_valid_integers_between_1_and_300 :
  let is_valid (n : ℕ) : Prop :=
    n % 10 = 0 ∧ n % 3 ≠ 0 ∧ n % 7 ≠ 0 ∧ (1 ≤ n) ∧ (n ≤ 300) in
  (finset.filter is_valid (finset.range 301)).card = 17 := sorry

end number_of_valid_integers_between_1_and_300_l440_440845


namespace solve_for_x_l440_440386

theorem solve_for_x :
  ∃ (x : ℝ), (x = Real.sqrt 2) ∧ (1 / 998 * (∑ k in Finset.range 1995, Real.sqrt (2 * Real.sqrt 2 * x - x^2 + k^2 - 2))) = 1995 := sorry

end solve_for_x_l440_440386


namespace problem_1_problem_2_problem_3_l440_440069

noncomputable def f (x : ℝ) : ℝ := exp(x) - log(x) 
noncomputable def g (x : ℝ) : ℝ := exp(x) - (1/x)
def m : ℝ := Int 0 

theorem problem_1 (x : ℝ) (h : x > 0) : (exp (x) + (1 / (x ^ 2))) > 0 :=
by sorry

theorem problem_2 (h : 2 > 0) : m > 2 :=
by sorry

noncomputable def h (x : ℝ) : ℝ := exp(x) - exp(m) * log(x)

theorem problem_3 (x : ℝ) (h1 : x > 0) (h2: m > 2) : ∃ x ∈ (1, m), h x = 0 :=
by sorry

end problem_1_problem_2_problem_3_l440_440069


namespace problem_solution_l440_440539

-- Define the roots a, b, c of the polynomial
variables {a b c : ℝ}

-- Assume a, b, c are the roots of the given polynomial
axiom roots_of_polynomial :
  a^3 - 7 * a^2 + 8 * a - 1 = 0 ∧
  b^3 - 7 * b^2 + 8 * b - 1 = 0 ∧
  c^3 - 7 * c^2 + 8 * c - 1 = 0

-- Define t as given in the problem
noncomputable def t : ℝ := Real.sqrt a + Real.sqrt b + Real.sqrt c

-- Prove the final expression
theorem problem_solution : t^6 - 21 * t^3 - 9 * t = 24 * t - 41 :=
sorry

end problem_solution_l440_440539


namespace problem_I51_problem_I52_problem_I53_l440_440551

-- Problem I5.1
theorem problem_I51 (x : Real) (a : Nat) (h : x = 19 + (0.87...).repeat) : x = a / 99 ↔ a = 1968 :=
by sorry

-- Problem I5.2
theorem problem_I52 (b c : Real) (b : Nat)
  (h₁ : ∀ y : Real, f y = 4 * sin (y * Real.pi / 180))
  (h₂ : f (1950 - 18) = b)
  (h₃ : sin (1950 * Real.pi / 180) = 0.5) : 
  b = 2 :=
by sorry

-- Problem I5.3
theorem problem_I53 (b c : Real)
  (h : ∀ x : Real, (sqrt 3) / (b * sqrt 7 - sqrt 3) = (2 * sqrt 21 + 3) / c) :
  c = 25 :=
by sorry

end problem_I51_problem_I52_problem_I53_l440_440551


namespace magnitude_of_expr_l440_440823

-- Define the given complex number z
def z : Complex := 3 + 4*Complex.i

-- Define the conjugate of z
def z_conj : Complex := Complex.conj z

-- Define the expression we are interested in
def expr : Complex := z_conj / Complex.i

-- State the theorem
theorem magnitude_of_expr : ‖expr‖ = 5 := by
  sorry

end magnitude_of_expr_l440_440823


namespace find_coordinates_of_C_l440_440886

structure Point where
  x : ℝ
  y : ℝ

def parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = C.x - D.x ∧ B.y - A.y = C.y - D.y) ∧
  (D.x - A.x = C.x - B.x ∧ D.y - A.y = C.y - B.y)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨7, 3⟩
def D : Point := ⟨3, 7⟩
def C : Point := ⟨8, 7⟩

theorem find_coordinates_of_C :
  parallelogram A B C D → C = ⟨8, 7⟩ :=
by
  intro h
  have h₁ := h.1.1
  have h₂ := h.1.2
  have h₃ := h.2.1
  have h₄ := h.2.2
  sorry

end find_coordinates_of_C_l440_440886


namespace determinant_of_given_matrix_l440_440399

-- Define the given matrix
def given_matrix (z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![z + 2, z, z], ![z, z + 3, z], ![z, z, z + 4]]

-- Define the proof statement
theorem determinant_of_given_matrix (z : ℂ) : Matrix.det (given_matrix z) = 22 * z + 24 :=
by
  sorry

end determinant_of_given_matrix_l440_440399


namespace price_increase_correct_l440_440977

variables (a b : ℝ)

def price_increase_over_two_years (a b : ℝ) : ℝ :=
  a + b + (a * b) / 100

theorem price_increase_correct :
  ∀ (a b : ℝ), price_increase_over_two_years a b = a + b + (a * b) / 100 :=
by
  intros a b
  unfold price_increase_over_two_years
  simp
  sorry

end price_increase_correct_l440_440977


namespace max_handshakes_60_men_l440_440301

theorem max_handshakes_60_men : 
  ∀ (n : ℕ), n = 60 → (n * (n - 1)) / 2 = 1770 :=
by
  intro n
  intro hn
  rw [hn]
  have : 60 * 59 = 3540 := by norm_num
  rw this
  have : 3540 / 2 = 1770 := by norm_num
  rw this
  rfl

end max_handshakes_60_men_l440_440301


namespace triangle_is_isosceles_l440_440499

theorem triangle_is_isosceles
  {A B C : ℝ} {a b c : ℝ} (h : a * cos B = b * cos A) :
  A = B ∨ B = C ∨ C = A :=
by
  sorry

end triangle_is_isosceles_l440_440499


namespace center_of_symmetry_tan_function_l440_440194

noncomputable def center_of_symmetry (k: ℤ) : ℝ × ℝ :=
  (π * (2 * k + 1) / 4, 1)

theorem center_of_symmetry_tan_function (k: ℤ) :
  let f (x: ℝ) := (1/2 : ℝ) * Real.tan (2 * x + π / 3) + 1 in
  -- Prove the center of symmetry for the graph of the function is (π * (2k + 1) / 4, 1)
  ∃ c: ℝ × ℝ, c = center_of_symmetry k ∧
    ∀ x: ℝ, f (c.fst - x) = f (c.fst + x) :=
begin
  sorry
end

end center_of_symmetry_tan_function_l440_440194


namespace die_roll_divisor_of_12_prob_l440_440662

def fair_die_probability_divisor_of_12 : Prop :=
  let favorable_outcomes := {1, 2, 3, 4, 6}
  let total_outcomes := 6
  let probability := favorable_outcomes.size / total_outcomes
  probability = 5 / 6

theorem die_roll_divisor_of_12_prob:
  fair_die_probability_divisor_of_12 :=
by
  sorry

end die_roll_divisor_of_12_prob_l440_440662


namespace sqrt_ceil_eq_one_range_of_x_l440_440268

/-- Given $[m]$ represents the largest integer not greater than $m$, prove $[\sqrt{2}] = 1$. -/
theorem sqrt_ceil_eq_one (floor : ℝ → ℤ) 
  (h_floor : ∀ m : ℝ, (floor m : ℝ) ≤ m ∧ ∀ z : ℤ, (z : ℝ) ≤ m → z ≤ floor m) :
  floor (Real.sqrt 2) = 1 :=
sorry

/-- Given $[m]$ represents the largest integer not greater than $m$ and $[3 + \sqrt{x}] = 6$, 
  prove $9 \leq x < 16$. -/
theorem range_of_x (floor : ℝ → ℤ) 
  (h_floor : ∀ m : ℝ, (floor m : ℝ) ≤ m ∧ ∀ z : ℤ, (z : ℝ) ≤ m → z ≤ floor m) 
  (x : ℝ) (h : floor (3 + Real.sqrt x) = 6) :
  9 ≤ x ∧ x < 16 :=
sorry

end sqrt_ceil_eq_one_range_of_x_l440_440268


namespace six_pointed_star_has_perimeter_l440_440534

variables (ABCDE : Type) [linear_ordered_field ABCDE] 

def regular_hexagon_perimeter (perimeter_sides: ℝ) : Type := sorry

def six_pointed_star_perimeter (hexagon_perimeter: ℝ) : ℝ := 
  let side_length := hexagon_perimeter / 6 in
  let segment_length := side_length * 2 / real.sqrt 3 in
  12 * segment_length

theorem six_pointed_star_has_perimeter (hexagon_perimeter: ℝ) :
  regular_hexagon_perimeter hexagon_perimeter →
  six_pointed_star_perimeter hexagon_perimeter = 4 * real.sqrt 3 :=
by
  intro h
  rw [six_pointed_star_perimeter, h]
  sorry

end six_pointed_star_has_perimeter_l440_440534


namespace xy_sum_correct_l440_440974

theorem xy_sum_correct (x y : ℝ) 
  (h : (4 + 10 + 16 + 24) / 4 = (14 + x + y) / 3) : 
  x + y = 26.5 :=
by
  sorry

end xy_sum_correct_l440_440974


namespace valid_range_m_l440_440927

def set_A (x : ℝ) : Prop := (1 / 32 ≤ 2^(-x)) ∧ (2^(-x) ≤ 4)

def set_B (x m : ℝ) : Prop :=
  let p := x^2 - 3 * m * x + 2 * m^2 - m - 1
  p < 0

theorem valid_range_m (m : ℝ) :
  (set_of (λ x, set_A x) ⊇ set_of (λ x, set_B x m)) ↔ (m = -2 ∨ (-1 ≤ m ∧ m ≤ 2)) :=
by
  sorry

end valid_range_m_l440_440927


namespace scalar_d_exists_l440_440382

open RealEuclideanSpace

theorem scalar_d_exists (v : ℝ^3) (i j k : ℝ^3)
  (hi : i = ![1, 0, 0]) 
  (hj : j = ![0, 1, 0]) 
  (hk : k = ![0, 0, 1]) :
  ∃ d : ℝ, 
    (i × (2 • (v × i)) + j × (2 • (v × j)) + k × (2 • (v × k)) = d • v) := 
by 
  use 0
  sorry

end scalar_d_exists_l440_440382


namespace rectangle_side_lengths_l440_440598

def rectangle_sides (x y : ℕ) : Prop :=
  (2 * (x + y) = 124) ∧ 
  (4 * (Int.sqrt (Int.ofNat ((x / 2)^2 + ((62 - x) / 2)^2))) = 100)

theorem rectangle_side_lengths : ∃ x y : ℕ, rectangle_sides x y ∧ (x = 48 ∧ y = 14) ∨ (x = 14 ∧ y = 48) :=
by
  use 48, 14
  split
  exact sorry
  right
  split
  rfl
  rfl

end rectangle_side_lengths_l440_440598


namespace equation_of_line_l_l440_440815

-- Define the conditions
def passes_through_origin (l : ℝ → ℝ × ℝ) := l 0 = (0, 0)
def parallel (l m : ℝ → ℝ × ℝ) := ∃ k, ∀ t, l t = m (k * t)
def equal_intercepts (m : ℝ → ℝ × ℝ) := 
  ∃ a, a ≠ 0 ∧ (∃ t, m t = (a, 0)) ∧ (∃ t, m t = (0, a))

-- Define line l and line m
def line_l (t : ℝ) : ℝ × ℝ := (t, -t)
noncomputable def line_m (t : ℝ) : ℝ × ℝ := (t, -t + 1) -- example, can be adjusted

-- The statement to prove
theorem equation_of_line_l :
  passes_through_origin line_l ∧ 
  (∃ t, equal_intercepts line_m ∧ parallel line_l line_m) → 
  ∀ t, line_l t = (t, -t) := 
by {
  sorry
}

end equation_of_line_l_l440_440815


namespace find_pq_l440_440546

noncomputable def find_k_squared (x y : ℝ) : ℝ :=
  let u1 := x^2 + y^2 - 12 * x + 16 * y - 160
  let u2 := x^2 + y^2 + 12 * x + 16 * y - 36
  let k_sq := 741 / 324
  k_sq

theorem find_pq : (741 + 324) = 1065 := by
  sorry

end find_pq_l440_440546


namespace graph_contains_K4_minus_one_edge_l440_440550

open SimpleGraph

theorem graph_contains_K4_minus_one_edge {n : ℕ} (h : n ≥ 2) (G : SimpleGraph (Fin (2 * n))) (h_edges : G.edgeFinset.card = n^2 + 1) : 
  ∃ (H : Subgraph G), (H = completeGraph (Fin 4)).deleteEdge ⟨(0 : Fin 4), 1⟩ :=
begin
  sorry
end

end graph_contains_K4_minus_one_edge_l440_440550


namespace greatest_of_3_consecutive_integers_l440_440631

theorem greatest_of_3_consecutive_integers (x : ℤ) (h : x + (x + 1) + (x + 2) = 24) : (x + 2) = 9 :=
by
-- Proof would go here.
sorry

end greatest_of_3_consecutive_integers_l440_440631


namespace equilateral_triangle_side_length_l440_440879

theorem equilateral_triangle_side_length
  (A B C D P : Point) 
  (hAB : dist A B = 11)
  (hCD : dist C D = 13)
  (hADP : EquilateralTriangle A D P)
  (hBCP : EquilateralTriangle B C P)
  (hCongruent : CongruentTriangles (EquilateralTriangle A D P) (EquilateralTriangle B C P)) :
  (side_length (EquilateralTriangle A D P) = 7) :=
sorry

end equilateral_triangle_side_length_l440_440879


namespace dice_sum_24_probability_l440_440764

noncomputable def probability_sum_24 : ℚ :=
  let prob_single_six := (1 : ℚ) / 6 in
  prob_single_six ^ 4

theorem dice_sum_24_probability :
  probability_sum_24 = 1 / 1296 :=
by
  sorry

end dice_sum_24_probability_l440_440764


namespace C1_equation_l440_440511

def circle_C2 (x y : ℝ) : Prop := (x - 5) ^ 2 + y ^ 2 = 9
def on_C1 (M : ℝ × ℝ) : Prop := 
  let (x, y) := M in
  x^2 + y^2 > 9 ∧ ∀ (M : ℝ × ℝ), 
    let (x, y) := M in
    abs (x + 2) = min (sqrt ((x - 5) ^ 2 + y ^ 2) - 3)

theorem C1_equation : ∀ (x y : ℝ), on_C1 (x, y) → y^2 = 20 * x :=
sorry

end C1_equation_l440_440511


namespace find_room_width_l440_440585

noncomputable def unknown_dimension (total_cost per_sqft_cost area_excluding_openings room_height room_length width_wall1 : ℝ) : ℝ :=
  (total_cost - area_excluding_openings * per_sqft_cost) / (2 * room_height * per_sqft_cost) - width_wall1

theorem find_room_width (x : ℝ) :
  let door_area : ℝ := 6 * 3
      window_area : ℝ := 3 * (4 * 3)
      room_height : ℝ := 12
      room_length : ℝ := 25
      total_area : ℝ := 2 * (room_length * room_height) + 2 * (x * room_height)
      area_excluding_openings : ℝ := total_area - door_area - window_area
      per_sqft_cost : ℝ := 3
      total_cost : ℝ := 2718
  in unknown_dimension total_cost per_sqft_cost area_excluding_openings room_height room_length 0 = 15 :=
by
  -- Here we would have the proof steps, but we use 'sorry' as required.
  sorry

end find_room_width_l440_440585


namespace proof_all_statements_l440_440853

-- Given conditions
variables {p q : ℕ}
axiom hp : nat.prime p
axiom hq : nat.prime q
axiom h_roots : ∃ x y : ℕ, x ≠ y ∧ (x^2 - p * x + q = 0) ∧ (y^2 - p * y + q = 0)

-- Goals to prove
theorem proof_all_statements (hp : nat.prime p) (hq : nat.prime q)
  (h_roots : ∃ x y : ℕ, x ≠ y ∧ ((x^2 - p * x + q = 0) ∧ (y^2 - p * y + q = 0))):
  (∃ x y : ℕ, x ≠ y ∧ (x - y).odd) ∧
  (∃ x y : ℕ, x ≠ y ∧ ((x = 1 ∨ nat.prime x) ∧ (y = 1 ∨ nat.prime y))) ∧
  nat.prime (p^2 - q) ∧
  nat.prime (p + q) :=
  sorry

end proof_all_statements_l440_440853


namespace quadratic_eq_unique_k_l440_440046

theorem quadratic_eq_unique_k (k : ℝ) (x1 x2 : ℝ) 
  (h_quad : x1^2 - 3*x1 + k = 0 ∧ x2^2 - 3*x2 + k = 0)
  (h_cond : x1 * x2 + 2 * x1 + 2 * x2 = 1) : k = -5 :=
by 
  sorry

end quadratic_eq_unique_k_l440_440046


namespace sqrt_infinite_l440_440398

theorem sqrt_infinite (x : ℝ) (h₁: x = sqrt (3 + x)) : x = (1 + sqrt 13) / 2 := by
  sorry

end sqrt_infinite_l440_440398


namespace license_plates_count_l440_440477

def numConsonantsExcludingY : Nat := 19
def numVowelsIncludingY : Nat := 6
def numConsonantsIncludingY : Nat := 21
def numEvenDigits : Nat := 5

theorem license_plates_count : 
  numConsonantsExcludingY * numVowelsIncludingY * numConsonantsIncludingY * numEvenDigits = 11970 := by
  sorry

end license_plates_count_l440_440477


namespace dice_sum_24_l440_440761

noncomputable def probability_of_sum_24 : ℚ :=
  let die_probability := (1 : ℚ) / 6
  in die_probability ^ 4

theorem dice_sum_24 :
  ∑ x in {x | x ∈ {1, 2, 3, 4, 5, 6} ∧ x = 6} = 24 → probability_of_sum_24 = 1 / 1296 :=
sorry

end dice_sum_24_l440_440761


namespace negation_divisible_by_5_is_odd_l440_440596

theorem negation_divisible_by_5_is_odd : 
  ¬∀ n : ℤ, (n % 5 = 0) → (n % 2 ≠ 0) ↔ ∃ n : ℤ, (n % 5 = 0) ∧ (n % 2 = 0) := 
by 
  sorry

end negation_divisible_by_5_is_odd_l440_440596


namespace avg_students_difference_l440_440684

open Real

theorem avg_students_difference :
  let students := 120
  let teachers := 6
  let enrollments := [60, 30, 10, 10, 5, 5]
  let t := (∑ e in enrollments, e) / (teachers : ℝ)
  let s := (60 * (60 / students) + 30 * (30 / students) +
            10 * (10 / students) + 10 * (10 / students) +
            5 * (5 / students) + 5 * (5 / students))
  (t - s) = -19.583 := by
  sorry

end avg_students_difference_l440_440684


namespace line_equation_l440_440010

theorem line_equation
  (x y : ℝ)
  (h1 : 2 * x + y + 2 = 0)
  (h2 : 2 * x - y + 2 = 0)
  (h3 : ∀ x y, x + y = 0 → x - 1 = y): 
  x - y + 1 = 0 :=
sorry

end line_equation_l440_440010


namespace find_m_n_l440_440400

open Nat

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem find_m_n (m n : ℕ) (h1 : binom (n+1) (m+1) / binom (n+1) m = 5 / 3) 
  (h2 : binom (n+1) m / binom (n+1) (m-1) = 5 / 3) : m = 3 ∧ n = 6 :=
  sorry

end find_m_n_l440_440400


namespace sum_proof_p_q_r_sum_l440_440913

noncomputable def sum_expr (n : ℕ) := 1 / (Real.sqrt (n + Real.sqrt (n^2 - 4)))

noncomputable def telescoping_sum : ℝ :=
  (1 / Real.sqrt 2) * (Real.sqrt 4902 + Real.sqrt 4900 - 2 - Real.sqrt 2)

theorem sum_proof :
  ∑ n in Finset.range 4899, sum_expr (n + 2) = 35 + 49 * Real.sqrt 2 :=
begin
  sorry
end

theorem p_q_r_sum :
  (35 + 49 + 2) = 86 :=
begin
  norm_num,
end

end sum_proof_p_q_r_sum_l440_440913


namespace conjugate_in_first_quadrant_l440_440445

def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem conjugate_in_first_quadrant :
  ∀ (i z : ℂ), 
  (i = complex.I) →
  ((i - 1) * z = complex.abs (1 + real.sqrt 3 * complex.I) + 3 * complex.I) →
  is_in_first_quadrant (complex.conj z) :=
by
  intros i z h_i h_eq
  sorry

end conjugate_in_first_quadrant_l440_440445


namespace probability_exactly_four_1s_l440_440253

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_four_1s_in_12_dice : ℝ :=
  let n := 12
  let k := 4
  let p := 1 / 6
  let q := 5 / 6
  (binomial_coefficient n k : ℝ) * p^k * q^(n - k)

theorem probability_exactly_four_1s : probability_of_four_1s_in_12_dice ≈ 0.089 :=
  by
  sorry

end probability_exactly_four_1s_l440_440253


namespace circle_passes_through_fixed_point_l440_440049

-- Define the ellipse C_1 with the equation x^2 / 4 + y^2 / 3 = 1
structure Ellipse (a b : ℝ) : Prop where
  a_pos : a > 0
  b_pos : b > 0
  b_lt_a : b < a
  equation : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x, y) ∈ C₁

-- Define the parabola C_2 with the equation y^2 = 4 * x
structure Parabola : Prop where
  equation : ∀ (x y : ℝ), y^2 = 4 * x → (x, y) ∈ C₂

-- Define |PF2| = 5/3 and the point on ellipse and parabola intersection
structure IntersectionCondition (x y PF2 : ℝ) : Prop where
  PF2_eq : |PF2| = 5 / 3
  on_parabola : y^2 = 4 * x
  on_ellipse : x^2 / 4 + y^2 / 3 = 1

-- Define the circle C_3 centered at a point T on parabola C_2
structure Circle (x₀ y₀ r : ℝ) : Prop where
  on_parabola : y₀^2 = 4 * x₀
  radius_eq : r = sqrt(4 + x₀^2)
  intersect_y_axis : ∀ M N : ℝ, sqrt(r^2 - x₀^2) = 2

-- Proof that circle C_3 always passes through point (2, 0) on ellipse C_1
theorem circle_passes_through_fixed_point (x₀ y₀ r a b : ℝ) (h₁ : Ellipse a b) (h₂ : Parabola) 
  (h₃ : Circle x₀ y₀ r) (h₄ : IntersectionCondition 2 0 (sqrt(2^2 + 0^2))) : 
  (2, 0) ∈ C₁ := 
sorry

end circle_passes_through_fixed_point_l440_440049


namespace partition_cities_l440_440311

structure City := 
  (name : String)

structure Flight :=
  (from to : City)
  (airline : Nat)

def k : Nat := sorry -- Assume k is given

def valid_flight (f : Flight) : Prop :=
  f.from ≠ f.to -- Ensure the flight connects different cities

def street_valid (flights : List Flight) (cities : List City) :=
  ∀ (f1 f2 : Flight), 
    (f1 ∈ flights) → 
    (f2 ∈ flights) →
    (f1.airline = f2.airline) → 
    (f1.from = f2.from ∨ f1.from = f2.to ∨ f1.to = f2.from ∨ f1.to = f2.to)

theorem partition_cities (flights : List Flight) (cities : List City) (h_valid : street_valid flights cities) : 
  ∃ (groups : List (List City)), (groups.length = k + 2) ∧ (∀ g ∈ groups, ∀ c1 c2 ∈ g, ¬ ∃ f ∈ flights, (f.from = c1 ∧ f.to = c2) ∨ (f.from = c2 ∧ f.to = c1)) :=
sorry

end partition_cities_l440_440311


namespace min_distance_midpoint_to_origin_l440_440492

theorem min_distance_midpoint_to_origin 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : x1 - y1 = 5) 
  (h2 : x2 - y2 = 15) 
  (mx my : ℝ)
  (hm : mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2) :
  (abs (0 - 0 - 10) / real.sqrt 2 = 5 * real.sqrt 2) :=
by
  sorry

end min_distance_midpoint_to_origin_l440_440492


namespace check_incorrect_equation_l440_440345

theorem check_incorrect_equation :
  ¬ (sqrt (121 / 225) = ± (11 / 15)) :=
sorry

end check_incorrect_equation_l440_440345


namespace tilings_of_3_by_5_rectangle_l440_440372

def num_tilings_of_3_by_5_rectangle : ℕ := 96

theorem tilings_of_3_by_5_rectangle (h : ℕ := 96) :
  (∃ (tiles : List (ℕ × ℕ)),
    tiles = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)] ∧
    -- Whether we are counting tiles in the context of a 3x5 rectangle
    -- with all distinct rotations and reflections allowed.
    True
  ) → num_tilings_of_3_by_5_rectangle = h :=
by {
  sorry -- Proof goes here
}

end tilings_of_3_by_5_rectangle_l440_440372


namespace center_on_circumcircle_of_triangle_l440_440584

variables {A B C D P O : Type*} [Geometry A] [Geometry B] [Geometry C] [Geometry D] [Geometry P] [Geometry O]

-- Definitions
def is_isosceles_trapezoid (ABCD : ConvexPolygon) (AB : Line) (CD : Line) : Prop :=
  is_congruent AB CD

def diagonals_intersect_at (AC BD : Line) (P : Point) : Prop :=
  intersect_at AC BD P

def circumcircle_center (ABCD : ConvexPolygon) (O : Point) : Prop :=
  is_center_of_circumcircle O ABCD

def lies_on_circumcircle (O : Point) (tri: Triangle) : Prop :=
  is_on_circumcircle O tri

-- Theorem statement
theorem center_on_circumcircle_of_triangle
  (ABCD : ConvexPolygon) (AB CD : Line) (AC BD : Line) (P : Point) (O : Point)
  (H1 : is_isosceles_trapezoid ABCD AB CD) 
  (H2 : diagonals_intersect_at AC BD P)
  (H3 : circumcircle_center ABCD O) :
  lies_on_circumcircle O (triangle A P B) :=
sorry

end center_on_circumcircle_of_triangle_l440_440584


namespace time_to_cross_bridge_l440_440840

/-- Define the constants for the problem. -/
def length_of_train : ℕ := 165
def speed_of_train_kmph : ℕ := 54
def length_of_bridge : ℕ := 850

/-- Convert speed from kmph to m/s. -/
def speed_of_train_m_s : ℝ := (speed_of_train_kmph * 1000) / 3600

/-- Define the total distance the train travels. -/
def total_distance : ℕ := length_of_train + length_of_bridge

/-- Define the time taken for the train to cross the bridge. -/
def time_to_cross : ℝ := total_distance / speed_of_train_m_s

/-- The main theorem stating the time it takes for the train to cross the bridge. -/
theorem time_to_cross_bridge : time_to_cross = 67.67 := by
  sorry

end time_to_cross_bridge_l440_440840


namespace probability_exactly_four_ones_is_090_l440_440242
open Float (approxEq)

def dice_probability_exactly_four_ones : Float :=
  let n := 12
  let k := 4
  let p_one := (1 / 6 : Float)
  let p_not_one := (5 / 6 : Float)
  let combination := ((n.factorial) / (k.factorial * (n - k).factorial) : Float)
  let probability := combination * (p_one ^ k) * (p_not_one ^ (n - k))
  probability

theorem probability_exactly_four_ones_is_090 : dice_probability_exactly_four_ones ≈ 0.090 :=
  sorry

end probability_exactly_four_ones_is_090_l440_440242


namespace multi_digit_perfect_square_has_two_distinct_digits_l440_440956

theorem multi_digit_perfect_square_has_two_distinct_digits :
  ∀ (n : ℕ), (∃ k : ℕ, n = k^2 ∧ n > 9) → ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ d1 ∈ digits 10 n ∧ d2 ∈ digits 10 n :=
by
  sorry

end multi_digit_perfect_square_has_two_distinct_digits_l440_440956


namespace certain_number_eq_40_l440_440580

theorem certain_number_eq_40 (x : ℝ) 
    (h : (20 + x + 60) / 3 = (20 + 60 + 25) / 3 + 5) : x = 40 := 
by
  sorry

end certain_number_eq_40_l440_440580


namespace total_cookies_l440_440934

def mona_cookies : ℕ := 20
def jasmine_cookies : ℕ := mona_cookies - 5
def rachel_cookies : ℕ := jasmine_cookies + 10

theorem total_cookies : mona_cookies + jasmine_cookies + rachel_cookies = 60 := 
by
  have h1 : jasmine_cookies = 15 := by sorry
  have h2 : rachel_cookies = 25 := by sorry
  have h3 : mona_cookies = 20 := by sorry
  sorry

end total_cookies_l440_440934


namespace range_of_a_I_minimum_value_of_a_II_l440_440424

open Real

def f (x a : ℝ) : ℝ := abs (x - a)

theorem range_of_a_I (a : ℝ) :
  (∀ x, -1 ≤ x → x ≤ 3 → f x a ≤ 3) ↔ 0 ≤ a ∧ a ≤ 2 := sorry

theorem minimum_value_of_a_II :
  ∀ a : ℝ, (∀ x : ℝ, f (x - a) a + f (x + a) a ≥ 1 - 2 * a) ↔ a ≥ (1 / 4) :=
sorry

end range_of_a_I_minimum_value_of_a_II_l440_440424


namespace wine_cost_today_l440_440303

theorem wine_cost_today (C : ℝ) (h1 : ∀ (new_tariff : ℝ), new_tariff = 0.25) (h2 : ∀ (total_increase : ℝ), total_increase = 25) (h3 : C = 20) : 5 * (1.25 * C - C) = 25 :=
by
  sorry

end wine_cost_today_l440_440303


namespace johnson_family_seating_l440_440190

-- Defining the total number of children:
def total_children := 8

-- Defining the number of sons and daughters:
def sons := 5
def daughters := 3

-- Factoring in the total number of unrestricted seating arrangements:
def total_seating_arrangements : ℕ := Nat.factorial total_children

-- Factoring in the number of non-adjacent seating arrangements for sons:
def non_adjacent_arrangements : ℕ :=
  (Nat.factorial daughters) * (Nat.factorial sons)

-- The lean proof statement to prove:
theorem johnson_family_seating :
  total_seating_arrangements - non_adjacent_arrangements = 39600 :=
by
  sorry

end johnson_family_seating_l440_440190


namespace area_transformation_l440_440141

noncomputable theory

def initial_area (T : Set (ℝ × ℝ)) : ℝ := 9

def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![3, 5], ![-2, 4]]

def transformed_area (T T' : Set (ℝ × ℝ)) : Prop :=
  let det := abs (3 * 4 - 5 * (-2))
  T' = (det : ℝ) * initial_area T

theorem area_transformation (T T' : Set (ℝ × ℝ)) :
  transformed_area T T' → T' = 198 := 
sorry

end area_transformation_l440_440141


namespace sum_of_digits_of_large_number_l440_440032

-- Define the conditions and the problem in Lean
theorem sum_of_digits_of_large_number (N : ℕ) (hN : N = 99999999999999999999999... (2015 times)) 
  (div9 : N % 9 = 0) :
  let a := Nat.digits 10 N in
  let b := a.sum in
  let c := (Nat.digits 10 b).sum in
  c = 9 := by
  sorry

end sum_of_digits_of_large_number_l440_440032


namespace min_value_of_M_l440_440925

theorem min_value_of_M 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) :
  let M := max (xy + 2 / z) (max (z + 2 / y) (y + z + 1 / x))
  in 3 ≤ M :=
by
  have : M ≥ 3 :=
    sorry
  exact this

end min_value_of_M_l440_440925


namespace total_cookies_l440_440940

def MonaCookies : ℕ := 20
def JasmineCookies : ℕ := MonaCookies - 5
def RachelCookies : ℕ := JasmineCookies + 10

theorem total_cookies : MonaCookies + JasmineCookies + RachelCookies = 60 := by
  -- Since we don't need to provide the solution steps, we simply use sorry.
  sorry

end total_cookies_l440_440940


namespace volume_of_folded_polyhedron_l440_440512

open Classical
open Real

noncomputable def volume_of_polyhedron (A E F : triangle ℝ) (B C D : rectangle ℝ) (G : triangle ℝ) : ℝ :=
  -- Volume of the polyhedron assuming proper folding
  -- Note: This is a simplified definition, the actual implementation would require specifying fold rules and solid geometry
  4

structure Problem where
  A E F : triangle ℝ
  B C D : rectangle ℝ
  G : triangle ℝ
  A_properties : is_scalene_right_triangle A 1 2
  E_properties : is_scalene_right_triangle E 1 2
  F_properties : is_scalene_right_triangle F 1 2
  B_properties : is_rectangle B 1 2
  C_properties : is_rectangle C 1 2
  D_properties : is_rectangle D 1 2
  G_properties : is_equilateral_triangle G (sqrt (1^2 + 2^2))
  folded_polyhedron : can_be_folded_into_polyhedron A B C D E F G

open Problem

theorem volume_of_folded_polyhedron (p : Problem) : volume_of_polyhedron p.A p.E p.F p.B p.C p.D p.G = 4 :=
  by sorry

end volume_of_folded_polyhedron_l440_440512


namespace count_four_digit_increasing_odd_l440_440842

open Nat

-- Define the problem constraints
def increasing_order_digits {a b c d : ℕ} (h : 0 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 9) : Prop := true

def odd_last_digit {d : ℕ} (h : d % 2 = 1) : Prop := true

-- State the problem
theorem count_four_digit_increasing_odd :
  ∃ n : ℕ, (n = 130) ∧ 
  (∀ (a b c d : ℕ), (increasing_order_digits (and.intro (le_refl 0) (and.intro (lt_add_one_iff.mpr (lt_add_one_iff.mpr (lt_add_one_iff.mpr (lt_add_one_iff.mpr (le_refl 0))))) (and.intro (nat.le_succ _) (and.intro (nat.le_succ _) (nat.le_succ _))))))) →
  (odd_last_digit (and.intro (lt_add_one_iff.mpr (lt_add_one_iff.mpr (lt_add_one_iff.mpr (lt_add_one_iff.mpr (le_refl 0))))) (nat.le_succ _))) → n = 130) :=
begin
  use 130,
  split,
  { refl },
  { intros a b c d ho hi, sorry }
end


end count_four_digit_increasing_odd_l440_440842


namespace radius_of_inscribed_circle_l440_440308

variable (height : ℝ) (alpha : ℝ)

theorem radius_of_inscribed_circle (h : ℝ) (α : ℝ) : 
∃ r : ℝ, r = (h / 2) * (Real.tan (Real.pi / 4 - α / 4)) ^ 2 := 
sorry

end radius_of_inscribed_circle_l440_440308


namespace min_sum_at_five_l440_440514

-- Define the sequence a_n as an arithmetic sequence with initial term -56 and common difference 12
def a (n : ℕ) : ℤ :=
  if n = 0 then -56 else a (n - 1) + 12

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) : ℤ :=
  (List.range n).map (λ k, a (k + 1)).sum

-- State the main theorem to be proven
theorem min_sum_at_five : ∀ n : ℕ, S n ≥ S 5 :=
sorry

end min_sum_at_five_l440_440514


namespace tina_mother_took_out_coins_l440_440613

theorem tina_mother_took_out_coins :
  let first_hour := 20
  let next_two_hours := 30 * 2
  let fourth_hour := 40
  let total_coins := first_hour + next_two_hours + fourth_hour
  let coins_left_after_fifth_hour := 100
  let coins_taken_out := total_coins - coins_left_after_fifth_hour
  coins_taken_out = 20 :=
by
  sorry

end tina_mother_took_out_coins_l440_440613


namespace behavior_of_g_as_x_approaches_infinity_l440_440381

noncomputable def g (x : ℝ) : ℝ := -3 * x^3 + 50 * x^2 - 4 * x + 10

theorem behavior_of_g_as_x_approaches_infinity :
  (tendsto g at_top (at_bot)) ∧ (tendsto g at_bot (at_top)) :=
  by
    sorry

end behavior_of_g_as_x_approaches_infinity_l440_440381


namespace initial_balance_l440_440996

theorem initial_balance (B : ℝ) (payment : ℝ) (new_balance : ℝ)
  (h1 : payment = 50) (h2 : new_balance = 120) (h3 : B - payment = new_balance) :
  B = 170 :=
by
  rw [h1, h2] at h3
  linarith

end initial_balance_l440_440996


namespace negative_to_zero_power_l440_440851

theorem negative_to_zero_power (a : ℝ) (h : a ≠ 0) : (-a) ^ 0 = 1 :=
by
  sorry

end negative_to_zero_power_l440_440851


namespace matrix_det_eq_l440_440926

theorem matrix_det_eq (A : Matrix (Fin 2) (Fin 2) ℝ)
  (h : ∀ d ∈ ({2014, 2016} : Set ℕ), abs (A^d - 1 : Matrix (Fin 2) (Fin 2) ℝ) = abs (A^d + 1))
  (n : ℕ) :
  abs (A^n - 1 : Matrix (Fin 2) (Fin 2) ℝ) = abs (A^n + 1) :=
sorry

end matrix_det_eq_l440_440926


namespace quadratic_eq_unique_k_l440_440044

theorem quadratic_eq_unique_k (k : ℝ) (x1 x2 : ℝ) 
  (h_quad : x1^2 - 3*x1 + k = 0 ∧ x2^2 - 3*x2 + k = 0)
  (h_cond : x1 * x2 + 2 * x1 + 2 * x2 = 1) : k = -5 :=
by 
  sorry

end quadratic_eq_unique_k_l440_440044


namespace trains_clear_time_l440_440618

theorem trains_clear_time
  (T1 T2 : ℕ) (S1_kmph S2_kmph : ℕ)
  (S1 : ℚ := S1_kmph * (5 / 18))
  (S2 : ℚ := S2_kmph * (5 / 18))
  (S : ℚ := S1 + S2)
  (D : ℕ := T1 + T2) :
  T1 = 120 → T2 = 280 →
  S1_kmph = 42 → S2_kmph = 30 →
  S1 = 11.67 → S2 = 8.33 → -- alternatively, replace with e.g. conversion definition
  S = 20 →
  D = 400 →
  D / S = 20 := by
  intro h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6, h7]
  sorry

end trains_clear_time_l440_440618


namespace find_x_l440_440868

theorem find_x (x : ℤ) :
  3 < x ∧ x < 10 →
  5 < x ∧ x < 18 →
  -2 < x ∧ x < 9 →
  0 < x ∧ x < 8 →
  x + 1 < 9 →
  x = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_x_l440_440868


namespace parabola_and_ellipse_proof_l440_440074

noncomputable def parabola_passing_point (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

noncomputable def parabola_proof (x y : ℝ) : Prop :=
  y^2 = 4 * x

noncomputable def lines_through_foci (C: Prop) (F: ℝ × ℝ) (l1 l2: set (ℝ × ℝ)) (P1 P2 Q1 Q2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧  (l1 = {p | p.snd = k * (p.fst - F.fst)}) ∧ (l2 = {p | p.snd = - (1/k) * (p.fst - F.fst)}) ∧
  C ∧ parabola_passing_point 2 2 (-2 * real.sqrt 2) ∧ -- Condition (C that is true for the points)
  (|P1.fst + P2.fst + 2 = (4*k^2 + 4)/k^2) ∧ |Q1.fst + Q2.fst + 2 = 4 + 4*k^2 ∧ 
  (1/|P1.fst + P2.fst + 2| + 1/|Q1.fst + Q2.fst + 2| = 1/4)

noncomputable def ellipse_condition (P1 P2 Q1 Q2 : ℝ × ℝ) : Prop :=
  ∃l1 l2 : ℝ×ℝ -> Prop,
    ellipse_intersects_points P1 P2 l1 ∧ ellipse_intersects_points Q1 Q2 l2 ∧
      (1/|P1.fst + P2.fst + 2| + 1/|Q1.fst + Q2.fst + 2| = 7/12)

theorem parabola_and_ellipse_proof :
  parabola_passing_point 2 2 (-2 * real.sqrt 2) →
  (lines_through_foci parabola_proof (1, 0) l1 l2 P1 P2 Q1 Q2) →
  ellipse_condition P1 P2 Q1 Q2 :=
sorry -- Proof omitted as specified.

end parabola_and_ellipse_proof_l440_440074


namespace circle_q_radius_expression_correct_l440_440520

/-- 
In triangle ABC with AB = AC = 60 and BC = 40, circle P has radius 12 and 
is tangent to AC and BC. Circle Q is externally tangent to P and tangent to 
AB and BC, with no point of circle Q lying outside of triangle ABC.
The radius of circle Q can be written as m - n√k, where m, n, and k 
are positive integers and k is the product of distinct primes. 
Prove that m + nk = 92.
-/
noncomputable def radius_q_expression : ℕ := 36 - 4 * Real.sqrt 14

theorem circle_q_radius_expression_correct :
  ∃ (m n k : ℕ), radius_q_expression = m - n * Real.sqrt k ∧
                 m + n * k = 92 ∧ 
                 is_product_of_distinct_primes k :=
sorry

end circle_q_radius_expression_correct_l440_440520


namespace stephen_total_distance_l440_440578

def mountain_height : ℝ := 40000
def round_trips : ℕ := 10
def ascent_fraction : ℝ := 3 / 4
def descent_fraction : ℝ := 2 / 3

noncomputable def ascent_distance : ℝ := ascent_fraction * mountain_height
noncomputable def descent_distance : ℝ := descent_fraction * ascent_distance
noncomputable def round_trip_distance : ℝ := ascent_distance + descent_distance
noncomputable def total_distance_covered : ℝ := round_trip_distance * round_trips

theorem stephen_total_distance :
  total_distance_covered = 500000 :=
by
  sorry

end stephen_total_distance_l440_440578


namespace sum_b_formula_l440_440821

noncomputable def a (n : ℕ) : ℝ := 2^(2*n - 1)

def b (n : ℕ) : ℝ := a (n + 1) * Real.logb 2 (a n)

noncomputable def sum_b (n : ℕ) : ℝ := ∑ i in Finset.range n, b i

theorem sum_b_formula (n : ℕ) : sum_b n = (40 / 9) + ((6 * ↑n - 5) / 9) * 2^(2 * ↑n + 3) :=
sorry

end sum_b_formula_l440_440821


namespace exact_time_between_3_and_4_l440_440895

theorem exact_time_between_3_and_4 :
  let t : ℝ := 3 + 21.81818181818182 / 60 in
  3 < t ∧ t < 4 ∧
  let minutes_past_3 := 21.81818181818182 in
  let minute_hand_pos := 6 * (minutes_past_3 + 8) in
  let hour_hand_pos_four_minutes_ago := 90 + 0.5 * (minutes_past_3 - 4) in
  (minute_hand_pos - hour_hand_pos_four_minutes_ago) % 360 = 180 ∨ 
  (minute_hand_pos - hour_hand_pos_four_minutes_ago) % 360 = -180
  ∧ t = 3 + 22.2 / 60 := -- equivalent to 3:22(2/11)
by
  sorry

end exact_time_between_3_and_4_l440_440895


namespace min_value_l440_440468

theorem min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∃ x y : ℝ, 2 * a * x - b * y + 2 = 0 ∧ x^2 + y^2 + 2 * x - 4 * y + 1 = 0)
    (h4 : ∃ x1 y1 x2 y2 : ℝ, (2 * a * x1 - b * y1 + 2 = 0) ∧ (2 * a * x2 - b * y2 + 2 = 0) ∧ 
          (x1^2 + y1^2 + 2 * x1 - 4 * y1 + 1 = 0) ∧ (x2^2 + y2^2 + 2 * x2 - 4 * y2 + 1 = 0) ∧ 
          (sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 4)) : 
    (4 / a + 1 / b) = 9 :=
sorry

end min_value_l440_440468


namespace geometric_sequence_fourth_term_l440_440673

theorem geometric_sequence_fourth_term (r : ℕ) (a₁ a₅ : ℕ) (h₁ : a₁ = 5) (h₄ : a₅ = 3125) (h₅ : a₅ = a₁ * r^4) : 
  a₁ * r^3 = 625 :=
by
  -- Using the given conditions
  rw [h₁] at h₅
  have hr : r = 5 := by
    -- Prove r = 5
    linarith_only
  rw [hr]  -- Substitute r = 5
  linarith_only

end geometric_sequence_fourth_term_l440_440673


namespace quadrilateral_ABCD_proof_l440_440176

theorem quadrilateral_ABCD_proof (AB BC CD BD CE BE AED CEB: ℝ) (right_angle_B: ∠B = 90°) (right_angle_C: ∠C = 90°)
(similar_ABC_BCD: ∼(△ABC ≈ △BCD)) (AB_gt_BC: AB > BC) (similar_ABC_CEB: ∼(△ABC ≈ △CEB)) 
(area_AED: area(△AED) = 13 * area(△CEB)) 
: ∃ x, AB = x^2 ∧ BC = x ∧ x = √(6 + 4√2) :=
by
  sorry

end quadrilateral_ABCD_proof_l440_440176


namespace exists_schoolchild_who_participated_in_all_competitions_l440_440507

theorem exists_schoolchild_who_participated_in_all_competitions
    (competitions : Fin 50 → Finset ℕ)
    (h_card : ∀ i, (competitions i).card = 30)
    (h_unique : ∀ i j, i ≠ j → competitions i ≠ competitions j)
    (h_intersect : ∀ S : Finset (Fin 50), S.card = 30 → 
      ∃ x, ∀ i ∈ S, x ∈ competitions i) :
    ∃ x, ∀ i, x ∈ competitions i :=
by
  sorry

end exists_schoolchild_who_participated_in_all_competitions_l440_440507


namespace g_inv_undefined_at_one_l440_440487

noncomputable def g (x : ℝ) : ℝ := (x + 2) / (x + 5)

def g_inv (y : ℝ) : ℝ := (5 * y - 2) / (1 - y)

theorem g_inv_undefined_at_one : ∀ x : ℝ,  g_inv x = 0 → x = 1 :=
by {
  sorry
}

end g_inv_undefined_at_one_l440_440487


namespace CDXY_concyclic_l440_440908

open_locale big_operators

noncomputable theory

-- Definitions based on given conditions
variables (Γ1 Γ2 : Type) [circle Γ1] [circle Γ2]
variables (A B C D P Q X Y : Type)
variables (line_A : line) (intersect_line_Γ1 : intersects line_A Γ1 C) (intersect_line_Γ2 : intersects line_A Γ2 D)
variables (tangent_AC : tangent Γ1 A) (tangent_CC : tangent Γ1 C) (tangent_AD : tangent Γ2 A) (tangent_DD : tangent Γ2 D)
variables (P : point) (Q : point)
variables (circ_BCP : circumcircle B C P) (circ_BDQ : circumcircle B D Q)
variables (line_AB : line) (line_PQ : line)

-- Problem statement translations
def centroids_collinear : Prop :=
  are_concyclic C D X Y

-- Main theorem
theorem CDXY_concyclic
  (h1 : intersects Γ1 Γ2 → A)
  (h2 : intersects Π_line_Γ1 → C)
  (h3 : intersects Π_line_Γ2 → D)
  (h4 : tend_to_A_predicate Γ1 A tangent_AC) 
  (h5 : tend_to_C_predicate Γ1 C tangent_CC) 
  (h6 : tend_to_A_predicate Γ2 A tangent_AD)
  (h7 : tend_to_D_predicate Γ2 D tangent_DD)
  (h8 : intersects line_AB PQ → Y)
  (h9 : intersects PQ Γ1)
  (h10 : intersects PQ Γ2)
  :
  centroids_collinear Γ1 Γ2 A B C D P Q X Y := sorry

end CDXY_concyclic_l440_440908


namespace calculation_error_l440_440687

theorem calculation_error (x y : ℕ) : (25 * x + 5 * y) = 25 * x + 5 * y :=
by
  sorry

end calculation_error_l440_440687


namespace distance_between_A_and_B_l440_440735

def distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2 + (b.3 - a.3)^2)

theorem distance_between_A_and_B :
  let A : ℝ × ℝ × ℝ := (3, -4, -1)
  let B : ℝ × ℝ × ℝ := (-1, 2, -3)
  distance A B = real.sqrt 56 :=
by
  sorry

end distance_between_A_and_B_l440_440735


namespace Jamie_cookie_selection_l440_440135

def num_ways_to_choose_cookies : ℕ := 120

theorem Jamie_cookie_selection :
  let o ch s p : ℕ in
  o + ch + s + p = 7 →
  ∃ (selection : ℕ), selection = num_ways_to_choose_cookies :=
by
  sorry

end Jamie_cookie_selection_l440_440135


namespace find_s_l440_440147

noncomputable def polynomial : Polynomial ℝ :=
  polynomial.monomial 4 1 + polynomial.monomial 3 p + polynomial.monomial 2 q + polynomial.monomial 1 r + polynomial.monomial 0 s

theorem find_s (p q r s : ℝ) (h_roots : ∃ m1 m2 m3 m4 : ℝ, polynomial = (Polynomial.C 1) * (X + m1) * (X + m2) * (X + m3) * (X + m4)) :
  p + q + r + s = 8091 → s = 8064 :=
by
  sorry

end find_s_l440_440147


namespace nick_coin_collection_l440_440946

theorem nick_coin_collection
  (total_coins : ℕ)
  (quarters_coins : ℕ)
  (dimes_coins : ℕ)
  (nickels_coins : ℕ)
  (state_quarters : ℕ)
  (pa_state_quarters : ℕ)
  (roosevelt_dimes : ℕ)
  (h_total : total_coins = 50)
  (h_quarters : quarters_coins = total_coins * 3 / 10)
  (h_dimes : dimes_coins = total_coins * 40 / 100)
  (h_nickels : nickels_coins = total_coins - (quarters_coins + dimes_coins))
  (h_state_quarters : state_quarters = quarters_coins * 2 / 5)
  (h_pa_state_quarters : pa_state_quarters = state_quarters * 3 / 8)
  (h_roosevelt_dimes : roosevelt_dimes = dimes_coins * 75 / 100) :
  pa_state_quarters = 2 ∧ roosevelt_dimes = 15 ∧ nickels_coins = 15 :=
by
  sorry

end nick_coin_collection_l440_440946


namespace Tom_distance_before_Karen_wins_l440_440284

-- Defining the conditions
def Karen_start_delay := 4 / 60 -- in hours (4 minutes)
def Karen_speed := 60 -- mph
def Tom_speed := 45 -- mph
def Winning_margin := 4 -- miles

-- Distance Tom covers in Karen's delay
def Tom_headstart := Tom_speed * Karen_start_delay

-- Total additional distance Karen needs to cover to win the bet
def Distance_to_win := Tom_headstart + Winning_margin

-- Time for Karen to cover the additional distance
def Time_to_win := Distance_to_win / Karen_speed

-- Distance Tom covers in that time
def Tom_distance := Tom_speed * Time_to_win

-- Proof statement
theorem Tom_distance_before_Karen_wins : Tom_distance = 5.25 := by
  -- Proof skipped
  sorry

end Tom_distance_before_Karen_wins_l440_440284


namespace monotonically_decreasing_interval_exists_sum_of_real_roots_l440_440072

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * cos x ^ 2 - 2 * sqrt 3 * sin x * cos x

theorem monotonically_decreasing_interval_exists (k : ℤ) :
  monotone_decreasing_on f (Icc (k * π - π / 6) (k * π + π / 3)) :=
sorry

theorem sum_of_real_roots :
  ∑ (x : ℝ) in {x | x ∈ Icc 0 (π / 2) ∧ f x = -1 / 3}, x = 2 * π / 3 :=
sorry

end monotonically_decreasing_interval_exists_sum_of_real_roots_l440_440072


namespace equivalent_single_discount_l440_440187

theorem equivalent_single_discount (x : ℝ) :
  let discount1 := 0.15
  let discount2 := 0.10
  let discount3 := 0.05
  let final_price := (1 - discount3) * (1 - discount2) * (1 - discount1) * x
  let single_discount := (1 - final_price / x)
  single_discount = 0.27325 := 
by {
  have h1 : final_price = 0.72675 * x,
  { rw [mul_assoc, mul_comm (1-0.1), ← mul_assoc, mul_assoc, ← mul_assoc, mul_assoc, ← mul_assoc, (1-0.05) * (1-0.1) * (1-0.15), mul_comm x],
    sorry }, 
  have h2 : single_discount = 1 - 0.72675,
  { rw [single_discount, h1, mul_comm, div_mul_cancel, ← sub_self],
    sorry },
  exact h2,
  sorry
}

end equivalent_single_discount_l440_440187


namespace sequence_is_odd_l440_440593

theorem sequence_is_odd (a : ℕ → ℤ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 7) 
  (h3 : ∀ n ≥ 2, -1/2 < (a (n + 1)) - (a n) * (a n) / a (n-1) ∧
                (a (n + 1)) - (a n) * (a n) / a (n-1) ≤ 1/2) :
  ∀ n > 1, (a n) % 2 = 1 :=
by
  sorry

end sequence_is_odd_l440_440593


namespace exists_n_gt_60n_l440_440794

def a : ℕ → ℕ
| 0 := 1
| n + 1 := 3 * a n

def b (n : ℕ) := 2 * n + 1

def T (n : ℕ) : ℕ :=
  (List.range (n + 1)).sum (λ i, a i * b i)

theorem exists_n_gt_60n : ∃ n : ℕ, T n > 60 * n :=
by {
  sorry
}

end exists_n_gt_60n_l440_440794


namespace sum_geometric_series_l440_440715

-- Define the geometric series parameters
def first_term (a : ℤ) := a = 2
def common_ratio (r : ℤ) := r = -2
def num_terms (n : ℤ) := n = 10
def last_term (l : ℤ) := l = -1024

-- Define the sum formula for the geometric series
def geometric_sum (a r n : ℤ) : ℤ := a * (r^n - 1) / (r - 1)

-- Problem: the sum of the geometric series with the given parameters is -682
theorem sum_geometric_series : first_term 2 → common_ratio (-2) → num_terms 10 → last_term (-1024) →
  geometric_sum 2 (-2) 10 = -682 := by
  sorry

end sum_geometric_series_l440_440715


namespace complement_U_M_l440_440473

def U : Set ℕ := {x | x > 0 ∧ ∃ y : ℝ, y = Real.sqrt (5 - x)}
def M : Set ℕ := {x ∈ U | 4^x ≤ 16}

theorem complement_U_M : U \ M = {3, 4, 5} := by
  sorry

end complement_U_M_l440_440473


namespace equiv_proof_problem_l440_440909

theorem equiv_proof_problem (b c : ℝ) (h1 : b ≠ 1 ∨ c ≠ 1) (h2 : ∃ n : ℝ, b = 1 + n ∧ c = 1 + 2 * n) (h3 : b * 1 = c * c) : 
  100 * (b - c) = 75 := 
by sorry

end equiv_proof_problem_l440_440909


namespace cell_phone_plan_cost_l440_440305

theorem cell_phone_plan_cost:
  let base_cost : ℕ := 25
  let text_cost : ℕ := 8
  let extra_min_cost : ℕ := 12
  let texts_sent : ℕ := 150
  let hours_talked : ℕ := 27
  let extra_minutes := (hours_talked - 25) * 60
  let total_cost := (base_cost * 100) + (texts_sent * text_cost) + (extra_minutes * extra_min_cost)
  (total_cost = 5140) :=
by
  sorry

end cell_phone_plan_cost_l440_440305


namespace six_digit_palindrome_probability_l440_440373

def is_palindrome (n : ℕ) : Prop :=
  let s := to_digits 10 n in
  s == s.reverse

theorem six_digit_palindrome_probability :
  let total_palindromes := 900
  let valid_palindromes := 800
  valid_palindromes / total_palindromes = 8 / 9 :=
  sorry

end six_digit_palindrome_probability_l440_440373


namespace quadratic_eq_unique_k_l440_440045

theorem quadratic_eq_unique_k (k : ℝ) (x1 x2 : ℝ) 
  (h_quad : x1^2 - 3*x1 + k = 0 ∧ x2^2 - 3*x2 + k = 0)
  (h_cond : x1 * x2 + 2 * x1 + 2 * x2 = 1) : k = -5 :=
by 
  sorry

end quadratic_eq_unique_k_l440_440045


namespace exists_finite_arith_seq_with_increasing_S_not_exists_infinite_arith_seq_with_increasing_S_l440_440410

def S (k : ℕ) : ℕ :=
  (Nat.digits 10 k).sum

theorem exists_finite_arith_seq_with_increasing_S (n : ℕ) (hn : 0 < n) :
  ∃ (a : Fin n → ℕ), (∀ i : Fin n, 0 < a i) ∧ StrictMono (λ i : Fin n, S (a i)) :=
sorry

theorem not_exists_infinite_arith_seq_with_increasing_S :
  ¬∃ (a : ℕ → ℕ), (∀ i : ℕ, 0 < a i ∧ S (a i) < S (a (i + 1))) :=
sorry

end exists_finite_arith_seq_with_increasing_S_not_exists_infinite_arith_seq_with_increasing_S_l440_440410


namespace time_for_B_alone_l440_440647

theorem time_for_B_alone (r_A r_B r_C : ℚ)
  (h1 : r_A + r_B = 1/3)
  (h2 : r_B + r_C = 2/7)
  (h3 : r_A + r_C = 1/4) :
  1/r_B = 168/31 :=
by
  sorry

end time_for_B_alone_l440_440647


namespace product_decrease_by_9_numbers_exist_l440_440611

-- Define the variables and assumptions
variables (a b c : ℕ)

-- Define the conditions from the problem
def condition1 : Prop := (a - 1) * (b - 1) * (c - 1) = a * b * c - 1
def condition2 : Prop := (a - 2) * (b - 2) * (c - 2) = a * b * c - 2

-- Define the property to be proven
def to_prove : Prop := (a - 3) * (b - 3) * (c - 3) = a * b * c - 9

-- Formalize the proof problem
theorem product_decrease_by_9 (h1 : condition1 a b c) (h2 : condition2 a b c) : to_prove a b c := sorry

-- Confirm the three numbers fulfilling the conditions exist
theorem numbers_exist : ∃ a b c : ℕ, condition1 a b c ∧ condition2 a b c :=
begin
  use [1, 1, 1],
  split;
  { unfold condition1 condition2,
    simp,
    norm_num }
end

end product_decrease_by_9_numbers_exist_l440_440611


namespace problem_statement_l440_440007

theorem problem_statement (a n : ℕ) (h1 : 1 ≤ a) (h2 : n = 1) : ∃ m : ℤ, ((a + 1)^n - a^n) = m * n := by
  sorry

end problem_statement_l440_440007


namespace triangle_angles_l440_440518

-- Definitions for altitude and angles in a triangle
variables {A B C : Type} [linear_ordered_field A]

-- Lean statement for the problem
theorem triangle_angles (a b c : A)
    (hA : a > 0)
    (hB : b > 0)
    (hC : c > 0)
    (h_alt_A : a ≤ b)
    (h_alt_B : b ≤ a) :
  (a = 90 ∧ b = 45 ∧ c = 45) :=
begin
  sorry
end

end triangle_angles_l440_440518


namespace max_S_R_squared_l440_440876

theorem max_S_R_squared (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) :
  (∃ a b c, DA = a ∧ DB = b ∧ DC = c ∧ S = 2 * (a * b + b * c + c * a) ∧
  R = (Real.sqrt (a^2 + b^2 + c^2)) / 2 ∧ (∃ max_val, max_val = (2 / 3) * (3 + Real.sqrt 3))) :=
sorry

end max_S_R_squared_l440_440876


namespace normal_distribution_prob_eq_l440_440864

open ProbabilityTheory

noncomputable def xi := Normal (-2) 4

theorem normal_distribution_prob_eq :
  ℙ (xi ∈ Ioc (-4) (-2)) = ℙ (xi ∈ Ioc (-2) 0) :=
sorry

end normal_distribution_prob_eq_l440_440864


namespace old_barbell_cost_l440_440134

theorem old_barbell_cost (x : ℝ) (new_barbell_cost : ℝ) (h1 : new_barbell_cost = 1.30 * x) (h2 : new_barbell_cost = 325) : x = 250 :=
by
  sorry

end old_barbell_cost_l440_440134


namespace imaginary_part_of_z_l440_440446

namespace ComplexNumberProof

-- Define the imaginary unit
def i : ℂ := ⟨0, 1⟩

-- Define the complex number
def z : ℂ := i^2 * (1 + i)

-- Prove the imaginary part of z is -1
theorem imaginary_part_of_z : z.im = -1 := by
    -- Proof goes here
    sorry

end ComplexNumberProof

end imaginary_part_of_z_l440_440446


namespace translate_right_coincide_l440_440232

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) - sin (2 * x)

theorem translate_right_coincide (φ : ℝ) (hφ : φ > 0) :
  (∀ x, f (x + φ) = f x) → φ = Real.pi :=
by
  sorry

end translate_right_coincide_l440_440232


namespace cadence_worked_longer_by_5_months_l440_440707

-- Definitions
def months_old_company : ℕ := 36

def salary_old_company : ℕ := 5000

def salary_new_company : ℕ := 6000

def total_earnings : ℕ := 426000

-- Prove that Cadence worked 5 months longer at her new company
theorem cadence_worked_longer_by_5_months :
  ∃ x : ℕ, 
  total_earnings = salary_old_company * months_old_company + 
                  salary_new_company * (months_old_company + x)
  ∧ x = 5 :=
by {
  sorry
}

end cadence_worked_longer_by_5_months_l440_440707


namespace collinear_vectors_no_perpendicular_vectors_l440_440087

variables {α : ℝ}
def m := (Real.sin α, Real.cos α)
def n := (Real.cos α, Real.cos α)

-- If \overrightarrow{m} and \overrightarrow{n} are collinear, find α.
theorem collinear_vectors (hα : α ∈ Set.Icc 0 Real.pi) (h_collinear : (Real.sin α) * (Real.cos α) - (Real.cos α) * (Real.cos α) = 0) :
  α = Real.pi / 2 ∨ α = Real.pi / 4 :=
sorry

-- Is there an α such that \overrightarrow{m} is perpendicular to (\overrightarrow{m} + \overrightarrow{n})?
theorem no_perpendicular_vectors (hα : α ∈ Set.Icc 0 Real.pi) :
  ¬ ((m α).1 * ((m α).1 + (n α).1) + (m α).2 * ((m α).2 + (n α).2) = 0) :=
sorry

end collinear_vectors_no_perpendicular_vectors_l440_440087


namespace cos_AMB_is_one_third_l440_440296

-- Define the regular tetrahedron and its properties
structure RegularTetrahedron :=
(vertices : fin 4 → ℝ × ℝ × ℝ)
(is_regular : ∀ i j k l, i ≠ j → j ≠ k → k ≠ l → l ≠ i → 
                        dist (vertices i) (vertices j) = 
                        dist (vertices k) (vertices l))

-- Define the midpoint function
def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- Define the distance function
def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

-- Define cosine of the angle function using the law of cosines
def cosine_of_angle (a b c : ℝ) : ℝ :=
(b^2 + c^2 - a^2) / (2 * b * c)

-- Assume tetrahedron ABCD and midpoint M of segment CD
noncomputable def cos_angle_AMB (tet : RegularTetrahedron) : ℝ :=
let A := tet.vertices 0 in
let B := tet.vertices 1 in
let C := tet.vertices 2 in
let D := tet.vertices 3 in
let M := midpoint C D in
cosine_of_angle 
  (dist A B)
  (dist A M)
  (dist B M)

-- The theorem we want to prove
theorem cos_AMB_is_one_third (tet : RegularTetrahedron) (h : tet.is_regular) :
  cos_angle_AMB tet = 1 / 3 :=
sorry

end cos_AMB_is_one_third_l440_440296


namespace solution_set_l440_440494

def f (x : ℝ) : ℝ := (x + 1) / (abs x + 1)

theorem solution_set (x : ℝ) : 1 < x ∧ x < 2 ↔ f (x ^ 2 - 2 * x) < f (3 * x - 4) := 
sorry

end solution_set_l440_440494


namespace intersection_A_B_l440_440160

def A : Set ℝ := {y | ∃ x : ℝ, y = x ^ (1 / 3)}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_B :
  A ∩ B = {x | x > 1} :=
sorry

end intersection_A_B_l440_440160


namespace find_tan_Z_l440_440126

variable {X Y Z : Real}

-- Conditions
def cot_X_cot_Z := cot X * cot Z = 1 / 4
def cot_Y_cot_Z := cot Y * cot Z = 1 / 8
def sum_angles := X + Y + Z = π  -- as 180 degrees expressed in radians (π radians)

-- Conjecture to be proved
theorem find_tan_Z (h1 : cot_X_cot_Z) (h2 : cot_Y_cot_Z) (h3 : sum_angles) : 
  tan Z = 2 * Real.sqrt 5 := 
sorry

end find_tan_Z_l440_440126


namespace sum_of_coefficients_l440_440022

noncomputable def polynomial_eq (x : ℝ) : ℝ := 1 + x^5
noncomputable def linear_combination (a0 a1 a2 a3 a4 a5 x : ℝ) : ℝ :=
  a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 + a5 * (x - 1) ^ 5

theorem sum_of_coefficients (a0 a1 a2 a3 a4 a5 : ℝ) :
  polynomial_eq 1 = linear_combination a0 a1 a2 a3 a4 a5 1 →
  polynomial_eq 2 = linear_combination a0 a1 a2 a3 a4 a5 2 →
  a0 = 2 →
  a1 + a2 + a3 + a4 + a5 = 31 :=
by
  intros h1 h2 h3
  sorry

end sum_of_coefficients_l440_440022


namespace maclaurin_arcsin_l440_440292

noncomputable def arcsin_series (x : ℝ) : ℝ :=
  x + ∑ n in Nat.range (n + 1), ((2 * n - 1).doubleFactorial / (2 * n).doubleFactorial) * x^(2 * n + 1) / (2 * n + 1)

theorem maclaurin_arcsin (x : ℝ) (hx : |x| ≤ 1) : 
    (∫ t in 0..x, (1 - t^2:ℝ)^(-1/2)) = arcsin_series x := 
by
  sorry

end maclaurin_arcsin_l440_440292


namespace triangle_side_lengths_expression_neg_l440_440450

theorem triangle_side_lengths_expression_neg {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2 * a^2 * b^2 - 2 * b^2 * c^2 - 2 * c^2 * a^2 < 0 := 
by 
  sorry

end triangle_side_lengths_expression_neg_l440_440450


namespace sqrt_21000_calculation_l440_440029

theorem sqrt_21000_calculation :
  (sqrt 2.1 = 1.449) →
  (sqrt 21 = 4.573) →
  sqrt 21000 = 144.9 :=
by
  -- Introduce the conditions as given
  intros h₁ h₂ 
  -- skip the proof
  sorry

end sqrt_21000_calculation_l440_440029


namespace line_through_PQ_l440_440814

theorem line_through_PQ (x y : ℝ) (P Q : ℝ × ℝ)
  (hP : P = (3, 2)) (hQ : Q = (1, 4))
  (h_line : ∀ t, (x, y) = (1 - t) • P + t • Q):
  y = x - 2 :=
by
  have h1 : P = ((3 : ℝ), (2 : ℝ)) := hP
  have h2 : Q = ((1 : ℝ), (4 : ℝ)) := hQ
  sorry

end line_through_PQ_l440_440814


namespace career_preference_circle_degrees_l440_440978

theorem career_preference_circle_degrees (males females : ℕ) (h_ratio : males / females = 2 / 3)
  (m_preference : ℕ) (f_preference : ℕ)
  (h_m_pref : m_preference = males / 4)
  (h_f_pref : f_preference = 3 * females / 4) :
  let total_students := males + females in
  let pref_students := m_preference + f_preference in
  360 * (pref_students / total_students) = 198 :=
by sorry

end career_preference_circle_degrees_l440_440978


namespace min_value_l440_440787

noncomputable theory

-- Define the conditions
def geometric_sequence (a_n : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

def condition_1 (a_n : ℕ → ℝ) : Prop :=
  a_n 7 = a_n 6 + 2 * a_n 5

def condition_2 (a_n : ℕ → ℝ) (m n : ℕ) : Prop :=
  sqrt (a_n m * a_n n) = 4 * a_n 1

-- Goal: Prove the minimum value of (1 / m) + (4 / n) = 3 / 2
theorem min_value (a_n : ℕ → ℝ) (a1 q : ℝ) :
  geometric_sequence a_n a1 q →
  condition_1 a_n →
  ∃ m n, condition_2 a_n m n →
  ((m + n = 6) → (1 / m + 4 / n = 3 / 2)) := sorry

end min_value_l440_440787


namespace arithmetic_sequence_formula_and_max_sum_l440_440442

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (a_arith_seq : ∀ n m : ℕ, a n - a m = (n - m) * (a 1 - a 0))
variable (h1 : a 1 + a 3 = 16)
variable (h2 : S 4 = 28)
variable (S_sum : ∀ n : ℕ, S n = ∑ i in range n, a (i + 1))

theorem arithmetic_sequence_formula_and_max_sum :
  (∀ n : ℕ, a n = 12 - 2 * n) ∧ (∀ n : ℕ, 0 < n → n ≤ 5 ∧ S n = 45) :=
sorry

end arithmetic_sequence_formula_and_max_sum_l440_440442


namespace probability_all_quit_from_same_tribe_l440_440218

def num_ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

def num_ways_to_choose_3_from_20 : ℕ := num_ways_to_choose 20 3
def num_ways_to_choose_3_from_10 : ℕ := num_ways_to_choose 10 3

theorem probability_all_quit_from_same_tribe :
  (num_ways_to_choose_3_from_10 * 2).toRat / num_ways_to_choose_3_from_20.toRat = 4 / 19 := 
  by
  sorry

end probability_all_quit_from_same_tribe_l440_440218


namespace find_b_l440_440702

theorem find_b
  (b : ℝ)
  (h1 : ∃ r : ℝ, 2 * r^2 + b * r - 65 = 0 ∧ r = 5)
  (h2 : 2 * 5^2 + b * 5 - 65 = 0) :
  b = 3 := by
  sorry

end find_b_l440_440702


namespace find_fifth_digit_l440_440335

theorem find_fifth_digit :
  ∃ (d : ℤ), 
    d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    5678 * 6789 = 385400000 + d * 1000 + 942 ∧
    ((3 + 8 + 5 + 4 + d + 9 + 4 + 2) % 9) = ((5 + 6 + 7 + 8) * (6 + 7 + 8 + 9) % 9) ∧ d = 7 :=
begin
  sorry
end

end find_fifth_digit_l440_440335


namespace trajectory_of_M_eq_circle_tangent_lines_of_circle_l440_440786

-- Define the given conditions
def pointA : ℝ × ℝ := (2, 0)
def pointB : ℝ × ℝ := (8, 0)

def M (x y : ℝ) : Prop := 
  real.sqrt ((x - 2) ^ 2 + y ^ 2) = (1 / 2) * real.sqrt ((x - 8) ^ 2 + y ^ 2)

-- Proof for question 1
theorem trajectory_of_M_eq_circle (x y : ℝ) : M x y → x ^ 2 + y ^ 2 = 16 :=
sorry

-- Define tangent line with equal intercepts
def tangent_line (a : ℝ) (x y : ℝ) : Prop :=
  x + y = a

-- Proof for question 2
theorem tangent_lines_of_circle (a : ℝ) (x y : ℝ) : (x ^ 2 + y ^ 2 = 16) ∧ tangent_line a x y → (a = 4 * real.sqrt 2 ∨ a = -4 * real.sqrt 2) :=
sorry

end trajectory_of_M_eq_circle_tangent_lines_of_circle_l440_440786


namespace percentage_students_B_is_26_and_2_by_3_l440_440705

def scores : List ℕ := [91, 82, 68, 99, 79, 86, 88, 76, 71, 58, 80, 89, 65, 85, 93]

def B_range (score: ℕ) : Prop := 87 ≤ score ∧ score ≤ 94

def num_students_with_B : ℕ := scores.countP B_range

def total_students : ℕ := scores.length

def percentage_of_B : ℚ := (num_students_with_B : ℚ) / (total_students : ℚ) * 100

theorem percentage_students_B_is_26_and_2_by_3 : percentage_of_B = 26 + 2 / 3 := by
  sorry

end percentage_students_B_is_26_and_2_by_3_l440_440705


namespace probability_of_24_is_1_div_1296_l440_440751

def sum_of_dice_is_24 (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 + d2 + d3 + d4 = 24

def probability_of_six (d : ℕ) : Rat :=
  if d = 6 then 1 / 6 else 0

def probability_of_sum_24 (d1 d2 d3 d4 : ℕ) : Rat :=
  (probability_of_six d1) * (probability_of_six d2) * (probability_of_six d3) * (probability_of_six d4)

theorem probability_of_24_is_1_div_1296 :
  (probability_of_sum_24 6 6 6 6) = 1 / 1296 :=
by
  sorry

end probability_of_24_is_1_div_1296_l440_440751


namespace simplify_expression_l440_440826

theorem simplify_expression (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^3 + 2) / x * (y^3 + 2) / y) - ((x^3 - 2) / y * (y^3 - 2) / x) = 4 * (x^2 / y + y^2 / x) :=
by sorry

end simplify_expression_l440_440826


namespace cot_ratio_l440_440884

variable {a b c : ℝ}

theorem cot_ratio (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : (a^2 + b^2) / c^2 = 2011) :
  (Real.cot (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) / 
  (Real.cot (Real.arccos ((c^2 + b^2 - a^2) / (2 * c * b))) + 
  Real.cot (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))))) = 1005 :=
sorry

end cot_ratio_l440_440884


namespace marble_probability_l440_440872

open BigOperators

theorem marble_probability:
  let total_marbles := 2500
  let red_marbles := 1500
  let blue_marbles := 1000
  
  let total_combinations := (total_marbles * (total_marbles - 1)) / 2
  let red_combinations := (red_marbles * (red_marbles - 1)) / 2
  let blue_combinations := (blue_marbles * (blue_marbles - 1)) / 2
  let same_color_combinations := red_combinations + blue_combinations
  
  let p_s := same_color_combinations.toRational / total_combinations.toRational
  let p_d := (red_marbles * blue_marbles).toRational / total_combinations.toRational
  
  abs (p_s - p_d) = (1 : ℚ) / 25
:= by
  sorry

end marble_probability_l440_440872


namespace distance_AB_l440_440117

-- Define the parametric equations of curve C1
def C1_parametric (t : ℝ) := (x = (real.sqrt 3 / 2) * t) ∧ (y = (1 / 2) * t)

-- Define the polar equations of curve C2 and C3
def C2_polar (θ : ℝ) := ρ = 2 * real.sin θ
def C3_polar (θ : ℝ) := ρ = 2 * real.sqrt 3 * real.cos θ

noncomputable def Cartesian_to_polar (x y : ℝ) := 
    if y ≠ 0 then real.atan2 y x else 
    if x > 0 then (0) else
    if x < 0 then (real.pi) else (0)

-- Given θ = π/6 is the polar form of curve C1
def C1_polar := θ = real.pi / 6

-- Prove |AB| = 2
theorem distance_AB : ∃ A B : ℝ, C2_polar (real.pi / 6) ∧ C3_polar (real.pi / 6) ∧ (|C3_polar (real.pi / 6) - C2_polar (real.pi / 6)|) = 2 := by
    sorry

end distance_AB_l440_440117


namespace sum_and_product_of_integers_abs_lt_6_l440_440222

theorem sum_and_product_of_integers_abs_lt_6 :
  let S := {n : ℤ | abs n < 6}
  (finset.sum (finset.filter (λ n, abs n < 6) finset.univ) id = 0) ∧
  (finset.prod (finset.filter (λ n, abs n < 6) finset.univ) id = 0) := by
  sorry

end sum_and_product_of_integers_abs_lt_6_l440_440222


namespace num_valid_orderings_6_students_l440_440991

def num_orderings (n : Nat) : Nat := Nat.factorial n
def undesired_a (n : Nat) : Nat := Nat.factorial (n - 1)
def undesired_b (n : Nat) : Nat := Nat.factorial (n - 1)
def undesired_a_and_b (n : Nat) : Nat := Nat.factorial (n - 2)

theorem num_valid_orderings_6_students : 
  let n := 6 in
  let total := num_orderings n in
  let undesired := undesired_a n + undesired_b n - undesired_a_and_b n in
  total - undesired = 504 := by
  sorry

end num_valid_orderings_6_students_l440_440991


namespace quadratic_solution_interval_length_l440_440050

theorem quadratic_solution_interval_length {m : ℝ}
  (ineq : ∀ x : ℝ, (m-3) * x^2 - 2 * m * x - 8 > 0 → (interval : set ℝ).nonempty)
  (interval_length : ∀ (a b : ℝ), interval = set.Ioo a b → 1 ≤ b - a ∧ b - a ≤ 2) :
  m ∈ set.Iic (-15) ∪ set.Icc (7/3) (33/14) :=
sorry

end quadratic_solution_interval_length_l440_440050


namespace actual_area_of_region_l440_440947

-- Problem Definitions
def map_scale : ℕ := 300000
def map_area_cm_squared : ℕ := 24

-- The actual area calculation should be 216 km²
theorem actual_area_of_region :
  let scale_factor_distance := map_scale
  let scale_factor_area := scale_factor_distance ^ 2
  let actual_area_cm_squared := map_area_cm_squared * scale_factor_area
  let actual_area_km_squared := actual_area_cm_squared / 10^10
  actual_area_km_squared = 216 := 
by
  sorry

end actual_area_of_region_l440_440947


namespace count_multiples_10_but_not_3_or_7_l440_440844

def multiple_of (m n : ℕ) : Prop := n % m = 0

def multiples_between (m start stop : ℕ) : List ℕ :=
  List.filter (multiple_of m) (List.range' start (stop - start + 1))

def exclude_multiples (l : List ℕ) (m : ℕ) : List ℕ :=
  List.filter (fun n => ¬ multiple_of m n) l

theorem count_multiples_10_but_not_3_or_7 : 
  (multiples_between 10 1 300)
  |> exclude_multiples 3
  |> exclude_multiples 7
  |>.length = 17 := sorry

end count_multiples_10_but_not_3_or_7_l440_440844


namespace range_of_a_given_quadratic_condition_l440_440083

theorem range_of_a_given_quadratic_condition:
  (∀ (a : ℝ), (∀ (x : ℝ), x^2 - 3 * a * x + 9 ≥ 0) → (-2 ≤ a ∧ a ≤ 2)) :=
by
  sorry

end range_of_a_given_quadratic_condition_l440_440083


namespace line_equation_slope_intercept_l440_440320

theorem line_equation_slope_intercept (A B C : Point ℝ) (m b : ℝ)
  (hA : A.y = b) (hC : C.x = m) (hAC_length : (A.x - C.x)^2 + (A.y - C.y)^2 = 32)
  (hB_midpoint : B.x = (A.x + C.x) / 2 ∧ B.y = (A.y + C.y) / 2)
  (hB_on_line : B.y = 2 * B.x + 2) : 
  ∃ m b, ∀ x y, y = m * x + b ↔ (m = -7 ∧ b = 28 / 5) := 
sorry

end line_equation_slope_intercept_l440_440320


namespace sum_of_digits_l440_440035

theorem sum_of_digits (N a b c : ℕ) (h1 : Nat.digits_count 10 N = 2015)
  (h2 : 9 ∣ N) (h3 : a = (Nat.digits 10 N).sum) (h4 : b = (Nat.digits 10 a).sum)
  (h5 : c = (Nat.digits 10 b).sum) : c = 9 :=
by sorry

end sum_of_digits_l440_440035


namespace odd_function_condition_l440_440651

open Real

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^3 + log (1 + x)
else x^3 - log (1 - x)

theorem odd_function_condition {x : ℝ} (hx : x < 0) :
  f(x) = x^3 - log (1 - x) :=
by
  have h_neg_x_pos : -x > 0 := by linarith
  have f_neg_x : f(-x) = (-x)^3 + log (1 - x) := by
    simp only [f, if_pos h_neg_x_pos]
    ring
  have h_odd : f(-x) = -f(x) := by
    sorry -- This would be shown using the odd function property.
  rw [h_odd, f_neg_x]
  simp only [neg_cube, neg_neg]
  sorry

end odd_function_condition_l440_440651


namespace smallest_blocks_count_l440_440302

theorem smallest_blocks_count
  (wall_length : ℕ) (wall_height : ℕ)
  (block_height : ℕ) (block_length1 block_length2 : ℕ)
  (staggered : Prop) (even_ends : Prop) :
  wall_length = 120 →
  wall_height = 8 →
  block_height = 1 →
  block_length1 = 2 →
  block_length2 = 1 →
  staggered →
  even_ends →
  ∃ n, n = 484 := 
by
  intros wall_length_eq wall_height_eq block_height_eq block_length1_eq block_length2_eq staggered even_ends
  use 484
  sorry

end smallest_blocks_count_l440_440302


namespace students_receptivity_strongest_receptivity_comparison_cannot_maintain_receptivity_l440_440175

def f : ℝ → ℝ
| x if 0 < x ∧ x ≤ 10  := -0.1 * x^2 + 2.6 * x + 44
| x if 10 < x ∧ x ≤ 15 := 60
| x if 15 < x ∧ x ≤ 25 := -3 * x + 105
| x if 25 < x ∧ x ≤ 40 := 30
| x                     := 0

theorem students_receptivity_strongest :
  ∀ x: ℝ, (0 < x ∧ x ≤ 10 ∨ 10 < x ∧ x ≤ 15 ∨ 15 < x ∧ x ≤ 25 ∨ 25 < x ∧ x ≤ 40) →
  (f 10 = 60 ∧ (∀ t: ℝ, 10 < t ∧ t ≤ 15 → f t = 60)) := by
  sorry

theorem receptivity_comparison :
  f 5 = 54.5 ∧ f 20 = 45 ∧ f 35 = 30 := by
  sorry

theorem cannot_maintain_receptivity :
  ¬ (∀ x: ℝ, (0 < x ∧ x ≤ 10 ∧ f x ≥ 56) ∨ (10 < x ∧ x ≤ 15 ∧ f x ≥ 56) ∨ (15 < x ∧ x ≤ 16.333333 ∧ f x ≥ 56) →
  16.333333 - 6 ≥ 12) := by
  sorry

end students_receptivity_strongest_receptivity_comparison_cannot_maintain_receptivity_l440_440175


namespace ratio_of_segments_tangents_perpendicular_l440_440547

-- Define the geometric setup
variables {A B C D : Type} 
variables [inst : EuclideanSpace ℝ A] [inst : EuclideanSpace ℝ B] [inst : EuclideanSpace ℝ C] [inst : EuclideanSpace ℝ D]
include inst

-- The conditions given in the problem
def point_in_ac_triangle (A B C D : A) : Prop :=
  acute_triangle A B C ∧
  ∠ A D B = ∠ A C B + 90° ∧
  (distance A C) * (distance B D) = (distance A D) * (distance B C)

-- The first question: Calculate the ratio AB * CD / AC * BD
theorem ratio_of_segments (A B C D : A) (h : point_in_ac_triangle A B C D) :
  (distance A B) * (distance C D) / ((distance A C) * (distance B D)) = sqrt 2 :=
sorry

-- The second question: Prove that the tangents at C are perpendicular
theorem tangents_perpendicular (A B C D : A) (h : point_in_ac_triangle A B C D) :
  tangents_perpendicular_at_C (circumcircle A C D) (circumcircle B C D) :=
sorry

end ratio_of_segments_tangents_perpendicular_l440_440547


namespace problem_l440_440023

noncomputable def a : ℝ := sorry  -- Add appropriate definition
noncomputable def b : ℝ := sorry  -- Add appropriate definition

theorem problem : 
  (2^a = 6 ∧ 3^b = 6) → 
  (ab = a + b) ∧ 
  (a + b > 4) ∧ 
  (4^a ≥ 8^b) ∧ 
  (log 2 a + log 2 b > 2) :=
by
  intros h,
  sorry

end problem_l440_440023


namespace probability_of_color_change_l440_440341

theorem probability_of_color_change :
  let cycle_duration := 100
  let green_duration := 45
  let yellow_duration := 5
  let red_duration := 50
  let green_to_yellow_interval := 5
  let yellow_to_red_interval := 5
  let red_to_green_interval := 5
  let total_color_change_duration := green_to_yellow_interval + yellow_to_red_interval + red_to_green_interval
  let observation_probability := total_color_change_duration / cycle_duration
  observation_probability = 3 / 20 := by sorry

end probability_of_color_change_l440_440341


namespace ac_eq_bc_l440_440206

-- Definitions based on the conditions above
def is_incircle_touched (ABC : Triangle) (D E : Point) : Prop :=
  touches_incircle BC D ∧ touches_incircle AC E

def sides_have_integral_lengths (ABC : Triangle) : Prop :=
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ side_length BC = a ∧ side_length AC = b ∧ side_length AB = c

def ad_be_squared_difference_condition (D E : Point) : Prop :=
  ∃ (AD BE : ℕ), AD = distance A D ∧ BE = distance B E ∧ |AD^2 - BE^2| ≤ 2

-- Main theorem to be proven
theorem ac_eq_bc (ABC : Triangle) (D E : Point)
  (hc1 : is_incircle_touched ABC D E)
  (hc2 : sides_have_integral_lengths ABC)
  (hc3 : ad_be_squared_difference_condition D E) :
  side_length AC = side_length BC := 
sorry

end ac_eq_bc_l440_440206


namespace total_cookies_l440_440942

def MonaCookies : ℕ := 20
def JasmineCookies : ℕ := MonaCookies - 5
def RachelCookies : ℕ := JasmineCookies + 10

theorem total_cookies : MonaCookies + JasmineCookies + RachelCookies = 60 := by
  -- Since we don't need to provide the solution steps, we simply use sorry.
  sorry

end total_cookies_l440_440942


namespace snail_distance_l440_440962

def speed_A : ℝ := 10
def speed_B : ℝ := 15
def time_difference : ℝ := 0.5

theorem snail_distance : 
  ∃ (D : ℝ) (t_A t_B : ℝ), 
    D = speed_A * t_A ∧ 
    D = speed_B * t_B ∧
    t_A = t_B + time_difference ∧ 
    D = 15 := 
by
  sorry

end snail_distance_l440_440962


namespace range_of_even_sin_function_l440_440103

theorem range_of_even_sin_function (f : ℝ → ℝ) (ϕ : ℝ) (h1: f = (λ x, Real.sin (2 * x + ϕ)))
  (h2: -π < ϕ ∧ ϕ < 0) (h3 : ∀ x, f x = f (-x)) :
  set.range (λ x, f x) ∩ set.Icc 0 (π / 4) = set.Icc (-1) 0 :=
by sorry

end range_of_even_sin_function_l440_440103


namespace complex_magnitude_problem_l440_440921

open Complex

theorem complex_magnitude_problem
  (z w : ℂ)
  (hz : abs z = 1)
  (hw : abs w = 2)
  (hzw : abs (z + w) = 3) :
  abs ((1 / z) + (1 / w)) = 3 / 2 :=
by {
  sorry
}

end complex_magnitude_problem_l440_440921


namespace geometric_series_sum_l440_440714

theorem geometric_series_sum : 
  let a := -2
  let r := 3
  let n := 8
  S_n = a * (r^n - 1) / (r - 1) := -6560 :=
by
  -- Definitions from the conditions
  let a := -2
  let r := 3
  let n := 8
  -- Expression for the sum of the geometric series
  let S_n := a * (r^n - 1) / (r - 1)
  -- Verifying that it equals -6560
  have : S_n = -6560
  -- Requesting help to finish this proof
  sorry

end geometric_series_sum_l440_440714


namespace find_constant_s_l440_440009

-- Statement of the problem in Lean
variable (a : ℝ) (A B C : ℝ × ℝ)
variable (x1 x2 m : ℝ)

-- Conditions that + x1 + x2 = m and x1 * x2 = -a * m
axiom parabola_conditions : x1 + x2 = m ∧ x1 * x2 = -a * m

-- Points A, B, and the midpoint C calculated as given
def point_A := (x1, x1^2)
def point_B := (x2, x2^2)
def midpoint_C := (m / 2, (m^2 + 2 * m * a) / 2)

-- Distance calculations
def AC2 := (x1 - (m / 2)) ^ 2 + (x1^2 - (m^2 + 2 * m * a) / 2) ^ 2
def BC2 := (x2 - (m / 2)) ^ 2 + (x2^2 - (m^2 + 2 * m * a) / 2) ^ 2

-- The statement to be proved
theorem find_constant_s (ha : a ≠ 0) : 
  let s := 1 / AC2 + 1 / BC2
  in ∀ A B, point_A = A ∧ point_B = B → s = s :=
by 
  sorry

end find_constant_s_l440_440009


namespace integer_pairs_divisibility_l440_440733

theorem integer_pairs_divisibility:
  ∀ (a b : ℤ), (2 ^ a + 1) % (2 ^ b - 1) = 0 ↔ (b = 1) ∨ (b = 2 ∧ ∃ k : ℤ, a = 2 * k + 1) :=
by 
sorrey

end integer_pairs_divisibility_l440_440733


namespace modulus_of_z_pure_imaginary_l440_440825

-- Define the required constants and variables
noncomputable def z (m : ℝ) : ℂ := m - complex.I 

-- Define the condition that (1 + i)z is a pure imaginary number
def isPureImaginary (z : ℂ) : Prop :=
  ∃ y : ℝ, z = y * complex.I

-- Define the main theorem statement
theorem modulus_of_z_pure_imaginary (m : ℝ) (h : isPureImaginary ((1 + complex.I) * (z m))) :
  complex.abs (z (-1)) = real.sqrt 2 :=
sorry

end modulus_of_z_pure_imaginary_l440_440825


namespace intersection_complement_eq_l440_440559

universe u
variable {U A B : Set}

theorem intersection_complement_eq :
  let U : Set ℤ := {-2, -1, 0, 1, 2}
  let A : Set ℤ := {1, 2}
  let B : Set ℤ := {-2, -1, 2}
  A ∩ (U \ B) = {1} :=
sorry

end intersection_complement_eq_l440_440559


namespace coronavirus_diameter_in_meters_l440_440775

theorem coronavirus_diameter_in_meters (n : ℕ) (h₁ : 1 = (10 : ℤ) ^ 9) (h₂ : n = 125) :
  (n * 10 ^ (-9 : ℤ) : ℝ) = 1.25 * 10 ^ (-7 : ℤ) :=
by
  sorry

end coronavirus_diameter_in_meters_l440_440775


namespace min_val_expression_l440_440952

theorem min_val_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 * b + b^2 * c + c^2 * a = 3) : 
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 :=
sorry

end min_val_expression_l440_440952


namespace number_of_two_element_subsets_mean_eight_l440_440092

def original_set : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

def mean_remaining_is_eight (removed_set : set ℕ) : Prop :=
  let remaining_set := original_set \ removed_set
  let remaining_sum := remaining_set.sum id
  let remaining_count := remaining_set.size
  remaining_count = 13 ∧ remaining_sum / remaining_count = 8

theorem number_of_two_element_subsets_mean_eight :
  ∃ (removed_sets : finset (set ℕ)), 
  removed_sets.card = 7 ∧ 
  ∀ removed_set ∈ removed_sets, 
  mean_remaining_is_eight removed_set :=
sorry

end number_of_two_element_subsets_mean_eight_l440_440092


namespace dice_probability_exactly_four_ones_l440_440262

noncomputable def dice_probability : ℚ := 
  (Nat.choose 12 4) * (1/6)^4 * (5/6)^8

theorem dice_probability_exactly_four_ones : (dice_probability : ℚ) ≈ 0.089 :=
  by sorry -- Skip the proof. 

#eval (dice_probability : ℚ)

end dice_probability_exactly_four_ones_l440_440262


namespace polynomial_degree_l440_440275

-- Definitions based on conditions
def f (x : ℝ) : ℝ := 3*x^4 + 3*x^3 + x - 14
def g (x : ℝ) : ℝ := 3*x^10 - 9*x^7 + 9*x^4 + 30
def h (x : ℝ) : ℝ := x^2 + 5

-- Polynomial degree function
def poly_degree (p : ℝ → ℝ) : ℕ :=
  sorry -- placeholder for the actual degree calculation function

-- Main proof statement
theorem polynomial_degree :
  poly_degree (λ x, f(x) * g(x) - (h(x))^7) = 14 :=
by
  sorry

end polynomial_degree_l440_440275


namespace smallest_difference_l440_440993

theorem smallest_difference (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 362880) (h_order : a < b ∧ b < c) : c - a = 92 := 
sorry

end smallest_difference_l440_440993


namespace smallest_a_inequality_l440_440414

theorem smallest_a_inequality 
  (x : ℝ)
  (h1 : x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi)) : 
  (∃ a : ℝ, a = -2.52 ∧ (∀ x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi), 
    ( ((Real.sqrt (Real.cos x / Real.sin x)^2) - (Real.sqrt (Real.sin x / Real.cos x)^2))
    / ((Real.sqrt (Real.sin x)^2) - (Real.sqrt (Real.cos x)^2)) ) < a )) :=
  sorry

end smallest_a_inequality_l440_440414


namespace triangle_ABC_angle_ABC_42_l440_440906

theorem triangle_ABC_angle_ABC_42 (A B C D : Type*)
  [Point A] [Point B] [Point C] [Point D]
  (BAC : Angle A B C) (ABC : Angle A B C)
  (angle_bisector : Line A B D C)
  (ABD : Triangle A B D) (ACB : Triangle A C B)
  (similar: Triangle.Similar ABD ACB)
  (BAC_eq : BAC = 117) :
  ABC = 42 := 
by
  sorry

end triangle_ABC_angle_ABC_42_l440_440906


namespace solve_cubic_eq_l440_440408

theorem solve_cubic_eq (z : ℂ) :
  z^3 = 1 ↔ (z = 1) ∨ (z = -1 / 2 + complex.I * (real.sqrt 3) / 2) ∨ (z = -1 / 2 - complex.I * (real.sqrt 3) / 2) :=
by sorry

end solve_cubic_eq_l440_440408


namespace line_segment_parametrization_l440_440594

theorem line_segment_parametrization :
  ∃ a b c d : ℝ, 
    (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) * (1 : ℝ) + t * (4 : ℝ) = at + b) ∧
    ((b = 1) ∧ (d = -3)) ∧
    ((a + b = 4) ∧ (c + d = 9)) ∧
    (a^2 + b^2 + c^2 + d^2 = 163) :=
sorry

end line_segment_parametrization_l440_440594


namespace find_x_squared_plus_inv_squared_l440_440448

noncomputable def x : ℝ := sorry

theorem find_x_squared_plus_inv_squared (h : x^4 + 1 / x^4 = 240) : x^2 + 1 / x^2 = Real.sqrt 242 := by
  sorry

end find_x_squared_plus_inv_squared_l440_440448


namespace P_0_lt_X_lt_3_l440_440789

variable (X : ℝ → Prop)
variable (delta : ℝ)

-- Define the normal distribution
def normal_distribution (μ σ : ℝ) (X : ℝ → Prop) : Prop :=
  ∀ x, X x ↔ (1 / (σ * √(2 * π))) * exp (-(x - μ)^2 / (2 * σ^2)) ∈ Set.Ioo 0 1

-- Conditions
axiom X_is_normal : normal_distribution 3 delta X
axiom P_X_le_6 : P (λ x, X x ∧ x ≤ 6) = 0.9

-- Theorem to prove
theorem P_0_lt_X_lt_3 : P (λ x, X x ∧ 0 < x ∧ x < 3) = 0.4 := by
  sorry

end P_0_lt_X_lt_3_l440_440789


namespace integer_count_valid_n_l440_440016

theorem integer_count_valid_n :
  {n : ℕ | 1 ≤ n ∧ n ≤ 75 ∧ ((n^3 - 1)! / ((n!) ^ n^2)) ∈ ℤ}.finite.card = 3 :=
by
  sorry

end integer_count_valid_n_l440_440016


namespace prob_sum_24_four_dice_l440_440754

section
open ProbabilityTheory

/-- Define the event E24 as the event where the sum of numbers on the top faces of four six-sided dice is 24 -/
def E24 : Event (StdGen) :=
eventOfFun {ω | ∑ i in range 4, (ω.gen_uniform int (6-1)) + 1 = 24}

/-- Probability that the sum of the numbers on top faces of four six-sided dice is 24 is 1/1296. -/
theorem prob_sum_24_four_dice : ⋆{ P(E24) = 1/1296 } := sorry

end

end prob_sum_24_four_dice_l440_440754


namespace min_distance_midpoint_to_C3_l440_440458

open Real

-- Definitions of the parameterized curves
def curve1 (t : ℝ) : ℝ × ℝ := (-4 + cos t, 3 + sin t)
def curve2 (θ : ℝ) : ℝ × ℝ := (8 * cos θ, 3 * sin θ)
def lineC3 (t : ℝ) : ℝ × ℝ := (3 + 2 * t, -2 + t)

-- The point P on curve1 when t = π/2
def pointP : ℝ × ℝ := curve1 (π / 2)

-- The point Q on curve2 is a moving point parameterized by θ
def pointQ (θ : ℝ) : ℝ × ℝ := curve2 θ

-- The midpoint M of P and Q
def midpointM (θ : ℝ) : ℝ × ℝ :=
  let P := pointP
  let Q := pointQ θ
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- The line equation of C3 in standard form
def lineC3_equation (x y : ℝ) : ℝ := x - 2 * y - 7

-- The distance from a point (x, y) to the line C3
def distance_to_lineC3 (x y : ℝ) : ℝ :=
  abs (lineC3_equation x y) / sqrt (1^2 + (-2)^2)

-- Minimum distance from the midpoint to the line C3 as stated in the problem
theorem min_distance_midpoint_to_C3 : 
  ∃ θ : ℝ, distance_to_lineC3 (midpointM θ).1 (midpointM θ).2 = 8 * sqrt 5 / 5 := 
sorry

end min_distance_midpoint_to_C3_l440_440458


namespace fraction_under_21_half_l440_440658

variable (P : ℕ) -- Total number of people in the room
variable (A : ℕ) -- Number of people under the age of 21
variable (B : ℕ) -- Number of people over the age of 65

-- Conditions
axiom total_people_bound : 50 < P ∧ P < 100
axiom people_under_21 : A = 30
axiom people_over_65_fraction : B = P / 2

-- Question in Lean: Prove that the fraction of people under 21 is 1/2
theorem fraction_under_21_half (h1 : total_people_bound) 
                              (h2 : people_under_21) 
                              (h3 : people_over_65_fraction) : 
                               A / P = 1 / 2 := sorry

end fraction_under_21_half_l440_440658


namespace angle_between_vectors_is_ninety_degrees_l440_440024

variables (α : Real)
def vector_a : Vector3 ℝ := (Real.cos α, 1, Real.sin α)
def vector_b : Vector3 ℝ := (Real.sin α, 1, Real.cos α)
def vector_sum := vector_a α + vector_b α
def vector_diff := vector_a α - vector_b α

theorem angle_between_vectors_is_ninety_degrees
  (h1 : vector_sum α) 
  (h2 : vector_diff α) : 
  Vector3.dot h1 h2 = 0 :=
sorry

end angle_between_vectors_is_ninety_degrees_l440_440024


namespace min_lateral_surface_area_achieved_at_inscribed_center_l440_440955

noncomputable def min_lateral_surface_area (a b c : ℝ) (h : ℝ) : ℝ :=
  let p := (a + b + c) / 2
  let r := sqrt ((p - a) * (p - b) * (p - c) / p)
  p * sqrt (h ^ 2 + r ^ 2)

theorem min_lateral_surface_area_achieved_at_inscribed_center 
  (a b c h x y z : ℝ) :
  let p := (a + b + c) / 2 in
  let r := sqrt ((p - a) * (p - b) * (p - c) / p) in
  let S := (1 / 2 * a * sqrt (h ^ 2 + x ^ 2) + 1 / 2 * b * sqrt (h ^ 2 + y ^ 2) + 1 / 2 * c * sqrt (h ^ 2 + z ^ 2)) in
  p * sqrt (h ^ 2 + r ^ 2) ≤ S := 
sorry

end min_lateral_surface_area_achieved_at_inscribed_center_l440_440955


namespace fraction_subtraction_equivalence_l440_440014

theorem fraction_subtraction_equivalence : 
  (16 / 24 - (1 + 2 / 9)) = -(5 / 9) := by
  -- Simplification of the fractions
  have h1 : 16 / 24 = 2 / 3, by sorry
  have h2 : 1 + 2 / 9 = 11 / 9, by sorry
  -- Conversion to a common denominator
  have h3 : (2 / 3) = 6 / 9, by sorry
  -- Subtraction
  show (6 / 9 - 11 / 9) = -(5 / 9), by sorry

end fraction_subtraction_equivalence_l440_440014


namespace sum_smallest_numbers_eq_six_l440_440987

theorem sum_smallest_numbers_eq_six :
  let smallest_natural := 0
  let smallest_prime := 2
  let smallest_composite := 4
  smallest_natural + smallest_prime + smallest_composite = 6 := by
  sorry

end sum_smallest_numbers_eq_six_l440_440987


namespace problem_1_problem_2_problem_3_l440_440828

/-- Problem 1: Given the function f(x) = (x+a)/e^x and the slope of the tangent line at x=0 is -1, prove that a = 2. -/
theorem problem_1 (a : ℝ) : 
  let f (x : ℝ) := (x + a) / Real.exp x in
  (differentiable ℝ f 0) ∧ (f.derivative 0 = -1) → a = 2 := 
by
  sorry

/-- Problem 2: Given the function f(x) = (x+a)/e^x, find the maximum value on the interval [-1,1].
  Prove for g(a):
  if a < 0, g(a) = (1+a)/e
  if 0 ≤ a ≤ 2, g(a) = 1/e^(1-a)
  if a > 2, g(a) = (a-1)e
-/
theorem problem_2 (a : ℝ) : 
  let f (x : ℝ) := (x + a) / Real.exp x in
  (∀ a < 0, max (f 1)) = (1 + a) / Real.exp 1) ∧
  (∀ 0 ≤ a ∧ a ≤ 2, max (f (1 - a)) = 1 / Real.exp (1 - a)) ∧
  (∀ a > 2, max (f (-1)) = (a - 1) * Real.exp 1 :=
by
  sorry

/-- Problem 3: Given the function f(x) = (x+a)/e^x with a=0, prove the minimum m > 0 such that 
    f(x) > f(m/x) for x ∈ (0,1) is m = 1
-/
theorem problem_3 : 
  let f (x : ℝ) := x / Real.exp x in
  (∀ x ∈ Ioo 0 1, f x > f (1 / x)) → m = 1 :=
by
  sorry

end problem_1_problem_2_problem_3_l440_440828


namespace minimum_value_A_l440_440954

theorem minimum_value_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + c^2 * a = 3) : 
  (a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3) ≥ 6 :=
by
  sorry

end minimum_value_A_l440_440954


namespace cost_of_fencing_each_side_43_l440_440102

def cost_of_fencing_each_side (total_cost : ℕ) (number_of_sides : ℕ) : ℕ :=
  total_cost / number_of_sides

theorem cost_of_fencing_each_side_43 :
  ∀ (total_cost number_of_sides : ℕ), total_cost = 172 → number_of_sides = 4 → cost_of_fencing_each_side total_cost number_of_sides = 43 :=
by
  intros total_cost number_of_sides h_total h_sides
  rw [cost_of_fencing_each_side, h_total, h_sides]
  apply Nat.div_eq_of_eq_mul_left _
  norm_num
  sorry

end cost_of_fencing_each_side_43_l440_440102


namespace isosceles_triangle_vertex_angle_l440_440600

theorem isosceles_triangle_vertex_angle (x : ℝ) (h₁ : 2 * x + 2 * x + x = 180) (h₂ : (2 * x) / x = 2) : x = 36 := 
by 
  have h₃ : 5 * x = 180 := by linarith,
  have h₄ : x = 36 := by linarith,
  exact h₄

end isosceles_triangle_vertex_angle_l440_440600


namespace number_of_valid_pairs_correct_l440_440545

noncomputable def number_of_valid_pairs : ℕ :=
  let odd_integers_greater_than_1 := {n : ℕ | n > 1 ∧ n % 2 = 1}
  let m_values := {m : ℕ | m ∈ odd_integers_greater_than_1 ∧ 2000 % m = 0}
  let n_values := {n : ℕ | n ∈ odd_integers_greater_than_1}
  let valid_pairs := {(m, n) | m ∈ m_values ∧ n ∈ n_values ∧ m * n ≥ 2000 ∧ intersects_line m n 2000 200 1099}
  valid_pairs.size

-- Hypothetical function that checks the intersection of the line through two squares.
def intersects_line (m n : ℕ) (k1 k2 target : ℕ) : Prop :=
  -- Implementation would go here.
  sorry

theorem number_of_valid_pairs_correct : number_of_valid_pairs = 248 :=
  by sorry

end number_of_valid_pairs_correct_l440_440545


namespace pairs_with_green_shirts_l440_440502

-- Conditions
variables (R G : ℕ)
variables (total_students total_pairs red_pairs : ℕ)
variables (students_in_red_shirts students_in_green_shirts : ℕ)

-- Assigning given values to variables
def given := students_in_red_shirts = 65 ∧ students_in_green_shirts = 83 ∧
              total_students = 148 ∧ total_pairs = 74 ∧ red_pairs = 27

-- Goal
def goal := (G, total_pairs - red_pairs, students_in_green_shirts - 11, total_pairs - red_pairs) = (72, 36, 83 - 11, 74 - 27)

-- Theorem to prove
theorem pairs_with_green_shirts (h : given) : ∃ green_pairs, green_pairs = 36 := 
sorry

end pairs_with_green_shirts_l440_440502


namespace nisos_population_reaches_capacity_140_years_after_1998_l440_440508

noncomputable def nisos_population_years_to_capacity
  (initial_year : ℕ) (initial_population : ℕ) (acres_available : ℕ) (acres_per_person : ℕ) (growth_period : ℤ) : ℤ :=
by
  let max_population := acres_available / acres_per_person
  let years := (log 2) (max_population / initial_population) * growth_period
  exact years

theorem nisos_population_reaches_capacity_140_years_after_1998 :
  nisos_population_years_to_capacity 1998 200 32000 2 20 = 140 :=
by
  sorry

end nisos_population_reaches_capacity_140_years_after_1998_l440_440508


namespace find_initial_shells_l440_440529

theorem find_initial_shells (x : ℕ) (h : x + 23 = 28) : x = 5 :=
by
  sorry

end find_initial_shells_l440_440529


namespace quadrilateral_area_l440_440961

noncomputable def AB : ℝ := 5
noncomputable def BC : ℝ := 12
noncomputable def CD : ℝ := 30
noncomputable def DA : ℝ := 34
def angle_CBA_is_right : Prop := ∠ C B A = 90

theorem quadrilateral_area : ∀ (Q : ConvexQuadrilateral), 
  Q.sides = (AB, BC, CD, DA) ∧ angle_CBA_is_right → Q.area = 30 := 
sorry

end quadrilateral_area_l440_440961


namespace largest_sum_225_l440_440227

theorem largest_sum_225 (a b c d : ℕ) (h1 : {a, b, c, d} = {6, 7, 8, 9}) :
  (∃ ab_bc_cd_ad : ℕ, ab_bc_cd_ad = ab + bc + cd + ad ∧ ab_bc_cd_ad = 225) :=
by sorry

end largest_sum_225_l440_440227


namespace intersection_complement_eq_singleton_l440_440474

def U : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) }
def M : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ (y - 3) / (x - 2) = 1 }
def N : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ y = x + 1 }
def complement_U (M : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := { p | p ∈ U ∧ p ∉ M }

theorem intersection_complement_eq_singleton :
  N ∩ complement_U M = {(2,3)} :=
by
  sorry

end intersection_complement_eq_singleton_l440_440474


namespace calculate_total_amount_l440_440196

theorem calculate_total_amount (CI: ℝ) (r: ℝ) (P: ℝ) (Total amount: ℝ) :
  CI = 2828.80 →
  r = 0.08 →
  P = 17000 →
  Total amount = P + CI →
  Total amount = 19828.80 :=
by
  sorry

end calculate_total_amount_l440_440196


namespace product_ab_l440_440587

-- Define complex numbers u and v
def u : Complex := 3 - 4 * Complex.i
def v : Complex := -2 + 2 * Complex.i

-- Define constants a and b based on conditions
def a : Complex := -6 - 5 * Complex.i
def b : Complex := 6 - 5 * Complex.i

-- Define the equation of the line
def line_eq (z : Complex) : Prop := 
  a * z + b * Complex.conj z = 47

-- Prove that the product of a and b is 61
theorem product_ab : a * b = 61 := by
  sorry

end product_ab_l440_440587


namespace dice_sum_probability_l440_440770

theorem dice_sum_probability :
  let D := finset.range 1 7  -- outcomes of a fair six-sided die
  (∃! d1 d2 d3 d4 ∈ D, d1 + d2 + d3 + d4 = 24) ->
  (probability(space, {ω ∈ space | (ω 1 = 6) ∧ (ω 2 = 6) ∧ (ω 3 = 6) ∧ (ω 4 = 6)}) = 1/1296) :=
sorry

end dice_sum_probability_l440_440770


namespace cos_value_proof_l440_440441

variable (α : Real)
variable (h1 : -Real.pi / 2 < α ∧ α < 0)
variable (h2 : Real.sin (α + Real.pi / 3) + Real.sin α = -(4 * Real.sqrt 3) / 5)

theorem cos_value_proof : Real.cos (α + 2 * Real.pi / 3) = 4 / 5 :=
by
  sorry

end cos_value_proof_l440_440441


namespace two_pow_log_mul_l440_440807

open Real

theorem two_pow_log_mul (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  2^(log x * log y) = 2^(log x) * 2^(log y) := 
sorry

end two_pow_log_mul_l440_440807


namespace original_number_l440_440679

theorem original_number (x : ℕ) (h1 : 4 * x = 108) (h2 : ∃ k : ℕ, 3 * x = 9 * k) : x = 27 :=
begin
  sorry
end

end original_number_l440_440679


namespace lines_angle_less_than_26_deg_l440_440031

theorem lines_angle_less_than_26_deg (lines : set (line ℝ)) (h_card : lines.card = 7) (h_parallel : ∀ l1 l2 ∈ lines, l1 ≠ l2 → ¬ parallel l1 l2) :
  ∃ l1 l2 ∈ lines, angle l1 l2 < 26 * real.pi / 180 :=
by
  sorry

end lines_angle_less_than_26_deg_l440_440031


namespace dice_sum_probability_l440_440773

theorem dice_sum_probability :
  let D := finset.range 1 7  -- outcomes of a fair six-sided die
  (∃! d1 d2 d3 d4 ∈ D, d1 + d2 + d3 + d4 = 24) ->
  (probability(space, {ω ∈ space | (ω 1 = 6) ∧ (ω 2 = 6) ∧ (ω 3 = 6) ∧ (ω 4 = 6)}) = 1/1296) :=
sorry

end dice_sum_probability_l440_440773


namespace distance_from_center_to_A_l440_440509

noncomputable def A_polar_coords : ℝ × ℝ := (2 * Real.sqrt 2, Real.pi / 4)
noncomputable def A_cartesian_coords : ℝ × ℝ := (2, 2)
noncomputable def circle_polar_eq (θ : ℝ) : ℝ := 4 * Real.sin θ
noncomputable def circle_cartesian_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0
noncomputable def circle_center : ℝ × ℝ := (0, 2)
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_from_center_to_A :
  let d := distance (2, 2) (0, 2) in d = 2 :=
by
  -- Directly skip the proof for the sake of this statement generation
  sorry

end distance_from_center_to_A_l440_440509


namespace quadratic_real_roots_l440_440426

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x - 1 = 0) ↔ m ≥ -3 ∧ m ≠ 1 := 
by 
  sorry

end quadratic_real_roots_l440_440426


namespace real_part_of_complex_expression_l440_440270

theorem real_part_of_complex_expression : 
  let i := Complex.I in
  Complex.re (i^2 * (1 + i : ℝ)) = -1 := 
by 
  sorry

end real_part_of_complex_expression_l440_440270


namespace unique_determination_of_T_l440_440411

theorem unique_determination_of_T'_n (b c : ℝ) (S₁₀₀₀ : ℝ) :
  let S' := λ n : ℕ, n * (2 * b + (n - 1) * c) / 2 in
  let T' := λ n : ℕ, ∑ k in Finset.range (n + 1), S' k in
  S' 1000 = 1000 * (b + 499.5 * c) →
  ∃! n : ℕ, T' n = T' 1500 :=
begin
  sorry
end

end unique_determination_of_T_l440_440411


namespace frank_reads_books_l440_440553

variables {a b c n : ℕ}

def p := 2 * a
def d := 3 * b
def t := 2 * c * 3 * b

theorem frank_reads_books (h1 : n * p = t) (h2 : n * d = t) : n = 2 * c :=
by {
  -- The proof would go here, but we'll use sorry for now
  sorry
}

end frank_reads_books_l440_440553


namespace Q1_no_such_a_b_Q2_no_such_a_b_c_l440_440298

theorem Q1_no_such_a_b :
  ∀ (a b : ℕ), (0 < a) ∧ (0 < b) → ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, k^2 = 2^n * a + 5^n * b) := sorry

theorem Q2_no_such_a_b_c :
  ∀ (a b c : ℕ), (0 < a) ∧ (0 < b) ∧ (0 < c) → ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, k^2 = 2^n * a + 5^n * b + c) := sorry

end Q1_no_such_a_b_Q2_no_such_a_b_c_l440_440298


namespace difference_of_cubes_not_divisible_by_19_l440_440172

theorem difference_of_cubes_not_divisible_by_19 (a b : ℤ) : 
  ¬ (19 ∣ ((3 * a + 2) ^ 3 - (3 * b + 2) ^ 3)) := by
  sorry

end difference_of_cubes_not_divisible_by_19_l440_440172


namespace consecutive_digits_sum_190_to_199_l440_440401

-- Define the digits sum function
def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define ten consecutive numbers starting from m
def ten_consecutive_sum (m : ℕ) : ℕ :=
  (List.range 10).map (λ i => digits_sum (m + i)) |>.sum

theorem consecutive_digits_sum_190_to_199:
  ten_consecutive_sum 190 = 145 :=
by
  sorry

end consecutive_digits_sum_190_to_199_l440_440401


namespace a5_equals_5_l440_440792

def fibonacci_sequence : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci_sequence n + fibonacci_sequence (n + 1)

theorem a5_equals_5 : fibonacci_sequence 5 = 5 :=
by
  sorry

end a5_equals_5_l440_440792


namespace right_triangle_acute_angle_l440_440115

theorem right_triangle_acute_angle (A B : ℝ) (h₁ : A + B = 90) (h₂ : A = 40) : B = 50 :=
by
  sorry

end right_triangle_acute_angle_l440_440115


namespace total_pages_l440_440413

theorem total_pages (x : ℕ) (h : 9 + 180 + 3 * (x - 99) = 1392) : x = 500 :=
by
  sorry

end total_pages_l440_440413


namespace Rose_more_correct_than_Liza_l440_440501

theorem Rose_more_correct_than_Liza :
  let total_items := 60
  let liza_correct_items := 0.9 * total_items
  let rose_incorrect_items := 4
  let rose_correct_items := total_items - rose_incorrect_items
  rose_correct_items - liza_correct_items = 2
  :=
by
  let total_items := 60
  let liza_correct_items := 0.9 * total_items
  let rose_incorrect_items := 4
  let rose_correct_items := total_items - rose_incorrect_items
  have h1: liza_correct_items = 54 := by norm_num
  have h2: rose_correct_items = 56 := by norm_num
  have h3: 56 - 54 = 2 := by norm_num
  show 56 - 54 = 2 from h3
  sorry

end Rose_more_correct_than_Liza_l440_440501


namespace James_weight_after_gain_l440_440524

theorem James_weight_after_gain 
    (initial_weight : ℕ)
    (muscle_gain_perc : ℕ)
    (fat_gain_fraction : ℚ)
    (weight_after_gain : ℕ) :
    initial_weight = 120 →
    muscle_gain_perc = 20 →
    fat_gain_fraction = 1/4 →
    weight_after_gain = 150 :=
by
  intros
  sorry

end James_weight_after_gain_l440_440524


namespace lines_concurrent_l440_440139

open_locale euclidean_geometry

noncomputable theory

variable {A B C D E F E' F' H X Y : Type}

-- Assume ABC is a triangle with given points and segments
variables [EuclideanGeometry A B C D E F E' F' H X Y]

-- Define the orthocenter condition
def is_orthocenter (A B C H : Type) : Prop :=
  ∃ (AD BE CF : Type), 
    are_altitudes A B C AD BE CF ∧
    AD = perpendicular A B ∧
    BE = perpendicular B C ∧
    CF = perpendicular C A ∧
    H = intersection AD BE CF

-- Define reflections
def is_reflection (E E' : Type) (AD : Type) : Prop := 
  E' = reflection_over AD E

def is_reflection' (F F' : Type) (AD : Type) : Prop := 
  F' = reflection_over AD F

-- Define intersections
def line_intersections (B F' C E' X : Type) : Prop := 
  X = intersection (line_through B F') (line_through C E')

def line_intersections' (B E' C F' Y : Type) : Prop := 
  Y = intersection (line_through B E') (line_through C F')

-- Combine conditions and prove concurrency
theorem lines_concurrent (H : Type) 
    (orthocenter_cond : is_orthocenter A B C H) 
    (reflection_E : is_reflection E E' AD)
    (reflection_F : is_reflection' F F' AD)
    (intersect_X : line_intersections B F' C E' X)
    (intersect_Y : line_intersections' B E' C F' Y) :
  loci_concurrent (line_through A X) (line_through Y H) (line_through B C) :=
sorry

end lines_concurrent_l440_440139


namespace projection_problem_l440_440213

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

theorem projection_problem :
  let v : ℝ × ℝ := (1, -1/2)
  let sum_v := (v.1 + 1, v.2 + 1)
  projection (3, 5) sum_v = (104/17, 26/17) :=
by
  sorry

end projection_problem_l440_440213


namespace divisibility_example_l440_440391

theorem divisibility_example : 
  ∃ (a b : ℕ), 
    (a = 532 ∧ b = 14 ∧ a % b = 0) ∨ 
    (a = 215 ∧ b = 43 ∧ a % b = 0) := 
by
  use 532, 14
  split
  { exact ⟨rfl, rfl, rfl⟩ }
  { use 215, 43
    exact ⟨rfl, rfl, rfl⟩ }

end divisibility_example_l440_440391


namespace find_numbers_l440_440267

theorem find_numbers (x y : ℚ) :
  (2 / 3) * x + 2 * y = 20 ∧ (1 / 4) * x - y = 2 →
  x = 144 / 7 ∧ y = 22 / 7 :=
by
  intro h
  cases h with h1 h2
  -- Proof steps would go here.
  sorry

end find_numbers_l440_440267


namespace find_angle_C_find_sin_B_plus_pi_over_3_l440_440500

namespace TriangleProblem

noncomputable def cos_C (a b c : ℝ) := (a^2 + b^2 - c^2) / (2 * a * b)

noncomputable def sin_B (b c C : ℝ) := b / (c * real.sin C)

noncomputable def cos_B (sin_B : ℝ) := real.sqrt (1 - sin_B ^ 2)

theorem find_angle_C (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 7) : 
  cos_C a b c = -1 / 2 → C = 2 * real.pi / 3 := 
  by sorry

theorem find_sin_B_plus_pi_over_3 (B C : ℝ) (h₁ : sin_B 5 7 (2 * real.pi / 3) = 5 * real.sqrt 3 / 14) 
  (h₂ : cos_B (5 * real.sqrt 3 / 14) = 11 / 14) : 
  real.sin (B + real.pi / 3) = 4 * real.sqrt 3 / 7 := 
  by sorry

end TriangleProblem

end find_angle_C_find_sin_B_plus_pi_over_3_l440_440500


namespace dice_sum_24_probability_l440_440765

noncomputable def probability_sum_24 : ℚ :=
  let prob_single_six := (1 : ℚ) / 6 in
  prob_single_six ^ 4

theorem dice_sum_24_probability :
  probability_sum_24 = 1 / 1296 :=
by
  sorry

end dice_sum_24_probability_l440_440765


namespace find_n_l440_440744

theorem find_n (n : ℕ) (h : (∏ k in finset.range (n - 1) + 2, real.log (k + 1) / real.log k) = 10) : 
  n = 1023 := 
sorry

end find_n_l440_440744


namespace train_still_there_when_susan_arrives_l440_440691

-- Define the conditions and primary question
def time_between_1_and_2 (t : ℝ) : Prop := 0 ≤ t ∧ t ≤ 60

def train_arrival := {t : ℝ // time_between_1_and_2 t}
def susan_arrival := {t : ℝ // time_between_1_and_2 t}

def train_present (train : train_arrival) (susan : susan_arrival) : Prop :=
  susan.val ≥ train.val ∧ susan.val ≤ (train.val + 30)

-- Define the probability calculation
noncomputable def probability_train_present : ℝ :=
  (30 * 30 + (30 * (60 - 30) * 2) / 2) / (60 * 60)

theorem train_still_there_when_susan_arrives :
  probability_train_present = 1 / 2 :=
sorry

end train_still_there_when_susan_arrives_l440_440691


namespace youseff_blocks_l440_440290

theorem youseff_blocks (x : ℕ) (h1 : x = 1 * x) (h2 : (20 / 60 : ℚ) * x = x / 3) (h3 : x = x / 3 + 8) : x = 12 := by
  have : x = x := rfl  -- trivial step to include the equality
  sorry

end youseff_blocks_l440_440290


namespace inverse_composition_l440_440928

variables {α β γ δ : Type}
variables (p : α → β) (q : β → γ) (r : γ → δ) (s : δ → α)
variables [HP : Function.Bijective p] [HQ : Function.Bijective q] [HR : Function.Bijective r] [HS : Function.Bijective s]

noncomputable def f := s ∘ q ∘ p ∘ r

theorem inverse_composition :
  Function.inverse f = (r⁻¹ ∘ p⁻¹ ∘ q⁻¹ ∘ s⁻¹) :=
sorry

end inverse_composition_l440_440928


namespace total_volume_of_ice_cream_l440_440208

noncomputable def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def volume_hemisphere (r : ℝ) : ℝ := (2 / 3) * π * r^3

theorem total_volume_of_ice_cream :
  let r := 3 in
  let h := 10 in
  volume_cone r h + volume_hemisphere r = 48 * π :=
by
  -- Proof omitted, results provided
  sorry

end total_volume_of_ice_cream_l440_440208


namespace cubic_meter_to_cubic_centimeters_l440_440088

theorem cubic_meter_to_cubic_centimeters : 
  (1 : ℝ)^3 = (100 : ℝ)^3 * (1 : ℝ)^0 := 
by 
  sorry

end cubic_meter_to_cubic_centimeters_l440_440088


namespace polar_conversion_example_l440_440881

noncomputable def polar_equiv (r θ : ℝ) : Prop :=
  ∃ (r' θ' : ℝ), r' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * Real.pi ∧ 
  (r', θ') = if r < 0 then (-r, θ + Real.pi) else (r, θ)

theorem polar_conversion_example :
  polar_equiv (-3) (5 * Real.pi / 6) = (3, 11 * Real.pi / 6) :=
by
  unfold polar_equiv
  sorry

end polar_conversion_example_l440_440881


namespace divides_seven_l440_440174

theorem divides_seven (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : Nat.gcd x y = 1) (h5 : x^2 + y^2 = z^4) : 7 ∣ x * y :=
by
  sorry

end divides_seven_l440_440174


namespace influenza_probability_l440_440703

theorem influenza_probability :
  let flu_rate_A := 0.06
  let flu_rate_B := 0.05
  let flu_rate_C := 0.04
  let population_ratio_A := 6
  let population_ratio_B := 5
  let population_ratio_C := 4
  (population_ratio_A * flu_rate_A + population_ratio_B * flu_rate_B + population_ratio_C * flu_rate_C) / 
  (population_ratio_A + population_ratio_B + population_ratio_C) = 77 / 1500 :=
by
  sorry

end influenza_probability_l440_440703


namespace prime_odd_distinct_digits_count_l440_440091

theorem prime_odd_distinct_digits_count : 
    {n : ℕ // 1000 ≤ n ∧ n ≤ 9999 ∧ nat.prime n ∧ (n % 2 = 1) ∧ all_distinct (digits n)}.to_finset.card = 224 := 
sorry

-- Helper function to check if all digits are distinct
def digits (n : ℕ) : list ℕ := sorry -- You should expand this to convert a number to a list of its digits

def all_distinct (l : list ℕ) : Prop := sorry -- You should expand this to check if all elements in the list are distinct

end prime_odd_distinct_digits_count_l440_440091


namespace prob_sum_24_four_dice_l440_440756

section
open ProbabilityTheory

/-- Define the event E24 as the event where the sum of numbers on the top faces of four six-sided dice is 24 -/
def E24 : Event (StdGen) :=
eventOfFun {ω | ∑ i in range 4, (ω.gen_uniform int (6-1)) + 1 = 24}

/-- Probability that the sum of the numbers on top faces of four six-sided dice is 24 is 1/1296. -/
theorem prob_sum_24_four_dice : ⋆{ P(E24) = 1/1296 } := sorry

end

end prob_sum_24_four_dice_l440_440756


namespace solve_for_x_l440_440856

theorem solve_for_x (x : ℝ) (h : sqrt (3 / x + 3) = 5 / 3) : x = -27 / 2 :=
by
  sorry

end solve_for_x_l440_440856


namespace maximize_volume_cone_l440_440660

-- Condition: Funnel is a cone with a slant height of 20 cm
noncomputable def r (h : ℝ) : ℝ := Real.sqrt(400 - h^2)

-- Volume of the cone given the height
noncomputable def V_cone (h : ℝ) : ℝ := (π * (r h)^2 * h) / 3

-- The height that maximizes the volume
theorem maximize_volume_cone :
  ∃ h : ℝ, h = 20 * Real.sqrt 3 / 3 ∧
  (∀ h' ∈ Set.Ioo 0 (20 * Real.sqrt 3 / 3), V_cone h' < V_cone h) ∧
  (∀ h' ∈ Set.Ioo (20 * Real.sqrt 3 / 3) 20, V_cone h' < V_cone h) := sorry

end maximize_volume_cone_l440_440660


namespace desired_digit_set_l440_440323

noncomputable def prob_digit (d : ℕ) : ℝ := if d > 0 then Real.log (d + 1) - Real.log d else 0

theorem desired_digit_set : 
  (prob_digit 5 = (1 / 2) * (prob_digit 5 + prob_digit 6 + prob_digit 7 + prob_digit 8)) ↔
  {d | d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8} = {5, 6, 7, 8} :=
by
  sorry

end desired_digit_set_l440_440323


namespace faulty_balance_inequality_l440_440229

variable (m n a b G : ℝ)

theorem faulty_balance_inequality
  (h1 : m * a = n * G)
  (h2 : n * b = m * G) :
  (a + b) / 2 > G :=
sorry

end faulty_balance_inequality_l440_440229


namespace abs_eq_solution_diff_l440_440385

theorem abs_eq_solution_diff : 
  ∀ x₁ x₂ : ℝ, 
  (2 * x₁ - 3 = 18 ∨ 2 * x₁ - 3 = -18) → 
  (2 * x₂ - 3 = 18 ∨ 2 * x₂ - 3 = -18) → 
  |x₁ - x₂| = 18 :=
by
  sorry

end abs_eq_solution_diff_l440_440385


namespace max_expression_value_l440_440212

theorem max_expression_value : 
  let expr (a b c d : ℝ) := a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a in 
  ∃ (a b c d : ℝ), 
  (∀ x ∈ set.Icc (-6 : ℝ) 6, ∀ y ∈ set.Icc (-6 : ℝ) 6, ∀ z ∈ set.Icc (-6 : ℝ) 6, ∀ w ∈ set.Icc (-6 : ℝ) 6, 
   expr x y z w ≤ 156) ∧ 
  expr a b c d = 156 := 
sorry

end max_expression_value_l440_440212


namespace prob_divisor_of_12_l440_440669

theorem prob_divisor_of_12 :
  (∃ d : Finset ℕ, d = {1, 2, 3, 4, 6}) → (∃ s : Finset ℕ, s = {1, 2, 3, 4, 5, 6}) →
  let favorable := 5
  let total := 6
  favorable / total = (5 : ℚ / 6 ) := sorry

end prob_divisor_of_12_l440_440669


namespace solution_of_inequality_system_l440_440602

variable (x : ℝ)

-- Define the conditions
def condition1 : Prop := 3 * x + 5 ≥ -1
def condition2 : Prop := 3 - x > (1 / 2) * x

-- Define the goal
def solution_set : set ℝ := { x | -2 ≤ x ∧ x < 2 }

-- The proof problem statement
theorem solution_of_inequality_system (h1 : condition1 x) (h2 : condition2 x) : x ∈ solution_set :=
sorry

end solution_of_inequality_system_l440_440602


namespace find_value_of_y_l440_440231

variable (p y : ℝ)
variable (h1 : p > 45)
variable (h2 : p * p / 100 = (2 * p / 300) * (p + y))

theorem find_value_of_y (h1 : p > 45) (h2 : p * p / 100 = (2 * p / 300) * (p + y)) : y = p / 2 :=
sorry

end find_value_of_y_l440_440231


namespace net_effect_on_sale_l440_440289

variable (P S : ℝ) (orig_revenue : ℝ := P * S) (new_revenue : ℝ := 0.7 * P * 1.8 * S)

theorem net_effect_on_sale : new_revenue = orig_revenue * 1.26 := by
  sorry

end net_effect_on_sale_l440_440289


namespace Alexandra_magazines_l440_440698

def magazines_on_friday : ℕ := 15
def magazines_on_saturday : ℕ := 20
def magazines_multiplier_on_sunday : ℕ := 4
def magazines_chewed : ℕ := 8

theorem Alexandra_magazines : 
  let magazines_sunday := magazines_on_friday * magazines_multiplier_on_sunday in
  let total_magazines := magazines_on_friday + magazines_on_saturday + magazines_sunday in
  total_magazines - magazines_chewed = 87 :=
by
  let magazines_sunday := magazines_on_friday * magazines_multiplier_on_sunday
  let total_magazines := magazines_on_friday + magazines_on_saturday + magazines_sunday
  sorry

end Alexandra_magazines_l440_440698


namespace hexagon_inequality_l440_440152

theorem hexagon_inequality
  (A B C D E F G H : Point)
  (h_hex : ConvexHexagon A B C D E F)
  (h_eq1 : Distance A B = Distance B C)
  (h_eq2 : Distance B C = Distance C D)
  (h_eq3 : Distance D E = Distance E F)
  (h_eq4 : Distance E F = Distance F A)
  (h_angle1 : Angle B C D = pi / 3)
  (h_angle2 : Angle E F A = pi / 3)
  (h_angle3 : Angle A G B = 2 * pi / 3)
  (h_angle4 : Angle D H E = 2 * pi / 3) :
  Distance A G + Distance G B + Distance G H + Distance H D + Distance H E ≥ Distance C F :=
sorry

end hexagon_inequality_l440_440152


namespace total_space_occupied_l440_440657

noncomputable def box_volume (l w h : ℝ) : ℝ :=
  l * w * h

def boxes_stored (total_cost cost_per_box : ℝ) : ℝ :=
  total_cost / cost_per_box

theorem total_space_occupied :
  let l := 15
  let w := 12
  let h := 10
  let volume := box_volume l w h
  let cost_per_box := 0.4
  let total_cost := 240
  let number_of_boxes := boxes_stored total_cost cost_per_box
  volume * number_of_boxes = 1,080,000 :=
by
  let l := 15
  let w := 12
  let h := 10
  let volume := box_volume l w h
  let cost_per_box := 0.4
  let total_cost := 240
  let number_of_boxes := boxes_stored total_cost cost_per_box
  sorry

end total_space_occupied_l440_440657


namespace probability_pyramid_l440_440796

-- Define the tetrahedral pyramid S-ABC.
constant S : Point
constant A B C : Point
constant volume_SABC : ℝ

-- Define a point P randomly selected within S-ABC.
constant P : Point
constant within_tetrahedron : Point → Prop
axiom Tetrahedron : within_tetrahedron P

-- Define heights h1 and h2.
constant h1 h2 : ℝ
axiom height_relation : h1 < (1 / 2) * h2

-- Volume relation.
constant volume_PABC : ℝ
axiom volume_relation : volume_PABC = (1 / 3) * (area_triangle A B C) * h1

theorem probability_pyramid (h1 h2 : ℝ) (H : h1 < (1 / 2) * h2)
  (vol_SABC : ℝ) (vol_PABC := (1 / 3) * (area_triangle A B C) * h1) : 
  (P_in_smaller_tetrahedron : 
    (1 - ((1 / 2) ^ 3))) = (7 / 8) :=
sorry

end probability_pyramid_l440_440796


namespace existence_of_indices_l440_440548

theorem existence_of_indices 
  (a1 a2 a3 a4 a5 : ℝ) 
  (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) (h4 : 0 < a4) (h5 : 0 < a5) : 
  ∃ (i j k l : Fin 5), 
    (i ≠ j) ∧ (i ≠ k) ∧ (i ≠ l) ∧ (j ≠ k) ∧ (j ≠ l) ∧ (k ≠ l) ∧ 
    |(a1 / a2) - (a3 / a4)| < 1/2 :=
by 
  sorry

end existence_of_indices_l440_440548


namespace card_trick_face_down_proof_card_trick_unlaid_proof_l440_440644

-- Part (a)
theorem card_trick_face_down_proof (deck : Finset (Fin 52)) (h_shuffled : deck.cardinal = 52)
  (draw : Finset (Fin 52)) (h_draw : draw.cardinal = 5)
  (face_up : Finset (Fin 52)) (h_face_up : face_up.cardinal = 4) :
  ∃ method : (Finset (Fin 52) → Finset (Fin 52) → Fin 52), ∀ f : Finset (Fin 52), method draw face_up = f :=
by
  sorry

-- Part (b)
theorem card_trick_unlaid_proof (deck : Finset (Fin 52)) (h_shuffled : deck.cardinal = 52)
  (draw : Finset (Fin 52)) (h_draw : draw.cardinal = 5)
  (face_up : Finset (Fin 52)) (h_face_up : face_up.cardinal = 4) :
  ∃ method : (Finset (Fin 52) → Finset (Fin 52) → Fin 52), ∀ f : Finset (Fin 52), method draw face_up = f :=
by
  sorry

end card_trick_face_down_proof_card_trick_unlaid_proof_l440_440644


namespace proof_of_triangle_is_right_angled_l440_440497

noncomputable def triangle_is_right_angled (A B C a b c : ℝ) (h1 : ∀ (A B C : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π)
  (h2 : ∀ (A : ℝ), cos (A / 2) ^ 2 = (c + b) / (2 * c)) : Prop :=
  C = π / 2

theorem proof_of_triangle_is_right_angled (A B C a b c : ℝ) (h1 : ∀ (A B C : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π)
  (h2 : ∀ (A : ℝ), cos (A / 2) ^ 2 = (c + b) / (2 * c)) : triangle_is_right_angled A B C a b c h1 h2 :=
by {
  sorry
}

end proof_of_triangle_is_right_angled_l440_440497


namespace shaded_to_white_area_ratio_l440_440633

theorem shaded_to_white_area_ratio (largest_square_area : ℝ)
  (midpoints_condition : ∀ (s : SymmetricSquare largest_square_area), vertices_at_midpoints s):
  ∃ r : ℝ, r = 5 / 3 :=
by
  sorry

end shaded_to_white_area_ratio_l440_440633


namespace domain_of_function_l440_440630

noncomputable def function_domain : Set ℝ := 
  {x : ℝ | x + 64 ≠ 0}

theorem domain_of_function :
  function_domain = {x : ℝ | x ∈ Set.Ioo (-∞) (-64) ∪ Set.Ioo (-64) ∞} :=
by
  sorry

end domain_of_function_l440_440630


namespace min_value_xy_l440_440488

theorem min_value_xy {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : (2 / x) + (8 / y) = 1) : x * y ≥ 64 :=
sorry

end min_value_xy_l440_440488


namespace dac_base4_to_base10_l440_440661

theorem dac_base4_to_base10 :
  let encode : Char -> Nat
      encode 'A' := 3
      encode 'B' := 1
      encode 'C' := 0
      encode 'D' := 2
  in (encode 'D') * 4^2 + (encode 'A') * 4^1 + (encode 'C') * 4^0 = 44 :=
by
  let encode : Char -> Nat
      encode 'A' := 3
      encode 'B' := 1
      encode 'C' := 0
      encode 'D' := 2
  sorry

end dac_base4_to_base10_l440_440661


namespace ellipse_standard_equation_hyperbola_standard_equation_l440_440742

-- Ellipse Definitions and Proof Goal
def ellipse_major_axis_length := 12
def ellipse_eccentricity := 2 / 3

theorem ellipse_standard_equation :
  ∃ a b : ℝ, 2 * a = ellipse_major_axis_length ∧ (4 / 3) / a = ellipse_eccentricity ∧
  a^2 - (4^2) = b^2 ∧ by ellipse_major_axis
    (x^2 : ℝ) / a^2 + (y^2 : ℝ) / b^2 = 1 :=
sorry

-- Hyperbola Definitions and Proof Goal
def hyperbola_shared_foci := ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1 →   ∃ (a b : ℝ) (P : ℝ × ℝ),
  P = (-(sqrt 5 / 2), -sqrt 6) ∧  x^2 / a^2 - y^2 / b^2 = 1 ∧ a^2 + b^2 = 25 :=
sorry

theorem hyperbola_standard_equation :
  ∃ a b : ℝ, exists₀) :=
sorry

end ellipse_standard_equation_hyperbola_standard_equation_l440_440742


namespace haircuts_away_from_next_free_l440_440579

def free_haircut (total_paid : ℕ) : ℕ := total_paid / 14

theorem haircuts_away_from_next_free (total_haircuts : ℕ) (free_haircuts : ℕ) (haircuts_per_free : ℕ) :
  total_haircuts = 79 → free_haircuts = 5 → haircuts_per_free = 14 → 
  (haircuts_per_free - (total_haircuts - free_haircuts)) % haircuts_per_free = 10 :=
by
  intros h1 h2 h3
  sorry

end haircuts_away_from_next_free_l440_440579


namespace question1_question2_l440_440783

def f (x : ℝ) : ℝ := 2^x - 1 / 2^|x|

-- Question 1
theorem question1 : ∃ x : ℝ, f x = 3 / 2 ↔ x = 1 := by
  sorry

-- Question 2
theorem question2 (m : ℝ) : (∀ t ∈ Icc (1:ℝ) 2, 2^t * f (2 * t) + m * f t ≥ 0) ↔ -5 ≤ m := by
  sorry

end question1_question2_l440_440783


namespace ben_gross_monthly_income_l440_440360

variables {G : ℝ}

def car_cost : ℝ := 400 + 150

def after_tax_income (gross_income : ℝ) : ℝ := (2 / 3) * gross_income

def car_expenditure (after_tax_inc : ℝ) : ℝ := 0.20 * after_tax_inc

theorem ben_gross_monthly_income 
    (h0 : car_expenditure (after_tax_income G) = car_cost) : 
    G = 4125 :=
by
  unfold car_cost at h0
  unfold after_tax_income at h0
  unfold car_expenditure at h0
  sorry

end ben_gross_monthly_income_l440_440360


namespace area_of_triangle_l440_440064

theorem area_of_triangle (a b c A B C : ℝ) (h_b : b = 2) (h_B : B = π / 6) (h_C : C = π / 4) :
  let s := (b * (Real.sin C) / (Real.sin B)) * (Real.sin (π - B - C))
  let area := 1 / 2 * b * (b * (Real.sin C) / (Real.sin B)) * s
  area = sqrt 3 + 1 := by sorry

end area_of_triangle_l440_440064


namespace angle_bisector_divides_DE_in_ratio_l440_440566

noncomputable def ratio_of_angle_bisector_divides_DE (AC BC : ℝ) (angleC_right : ∠ C = 90) (ratio_correct : AC = 2 ∧ BC = 5) : ℝ :=
  if h : (∠C = 90) ∧ (AC = 2) ∧ (BC = 5) then 2 / 5 else 0

theorem angle_bisector_divides_DE_in_ratio (AC BC : ℝ) (H : ∠ C = 90 ∧ AC = 2 ∧ BC = 5) :
  ratio_of_angle_bisector_divides_DE AC BC H.left H.right = 2 / 5 := 
sorry

end angle_bisector_divides_DE_in_ratio_l440_440566


namespace johnson_potatoes_left_l440_440136

theorem johnson_potatoes_left :
  ∀ (initial gina tom anne remaining : Nat),
  initial = 300 →
  gina = 69 →
  tom = 2 * gina →
  anne = tom / 3 →
  remaining = initial - (gina + tom + anne) →
  remaining = 47 := by
sorry

end johnson_potatoes_left_l440_440136


namespace problem_solution_l440_440094

theorem problem_solution (x : ℝ) (h : 1 - 9 / x + 20 / x^2 = 0) : (2 / x = 1 / 2 ∨ 2 / x = 2 / 5) := 
  sorry

end problem_solution_l440_440094


namespace sum_divisible_by_4003_l440_440958

theorem sum_divisible_by_4003 :
  let S := (Finset.range 2001).prod (λ i => (i + 1) : ℕ) + 
           (Finset.range 2001).prod (λ i => (2002 + i) : ℕ)
  in S % 4003 = 0 := 
by sorry

end sum_divisible_by_4003_l440_440958


namespace planes_parallel_l440_440060

variables {m n : Type} {α β : Type}

-- Assuming m and n are lines, and α and β are planes.
axiom line (x : Type) : Prop
axiom plane (x : Type) : Prop

-- Conditions
axiom mn_parallel : line m → line n → m ∥ n
axiom m_perp_alpha : line m → plane α → m ⊥ α
axiom n_perp_beta : line n → plane β → n ⊥ β

-- Statement to prove
theorem planes_parallel (hm : line m) (hn : line n) (ha : plane α) (hb : plane β) (hpm : m ∥ n) (hma : m ⊥ α) (hnb : n ⊥ β) : α ∥ β :=
by sorry

end planes_parallel_l440_440060


namespace floor_T_squared_l440_440156

def T : ℝ :=
  ∑ i in Finset.range 2007 \end Finset.range 1,
   real.sqrt (1 + 1/(i+2 : ℝ)^2 + 1/((i+3 : ℝ)^2))

theorem floor_T_squared : ∀ T, T = ∑ i in Finset.range 2007 \end Finset.range 1,
      real.sqrt (1 + 1/(i+2 : ℝ)^2 + 1/((i+3 : ℝ)^2)) → ⌊T^2⌋ = 4034064 := by
  sorry

end floor_T_squared_l440_440156


namespace count_integers_log_inequality_l440_440017

open Real

theorem count_integers_log_inequality : 
  ∃ (n : ℕ), n = 28 ∧ ∀ (x : ℤ), 50 < x ∧ x < 80 → log 10 ((x - 50) * (80 - x)) < 1.5 :=
by {
  sorry
}

end count_integers_log_inequality_l440_440017


namespace einstein_birth_weekday_l440_440967

-- Defining the reference day of the week for 31 May 2006
def reference_date := 31
def reference_month := 5
def reference_year := 2006
def reference_weekday := 3  -- Wednesday

-- Defining Albert Einstein's birth date
def einstein_birth_day := 14
def einstein_birth_month := 3
def einstein_birth_year := 1879

-- Defining the calculation of weekday
def weekday_from_reference(reference_day reference_weekday einstein_birth_day einstein_birth_month einstein_birth_year : Nat) : Nat :=
  let days_from_reference_to_birth := 46464  -- Total days calculated in solution
  (reference_weekday - (days_from_reference_to_birth % 7) + 7) % 7

-- Stating the theorem
theorem einstein_birth_weekday : weekday_from_reference reference_day reference_weekday einstein_birth_day einstein_birth_month einstein_birth_year = 5 :=
by
  -- Proof omitted
  sorry

end einstein_birth_weekday_l440_440967


namespace num_elements_in_A_l440_440438

-- Define the greatest integer function
def greatest_int (x : ℝ) : ℤ := Int.floor x

-- Define the function f(x) = [x[x]]
def f (n : ℕ) (x : ℝ) : ℤ := greatest_int (x * (greatest_int x))

-- Define the main statement we want to prove
theorem num_elements_in_A (n : ℕ) (h : 0 < n): 
  ∃ A : Set ℤ, (A = { y | ∃ x ∈ Ico 0 n, f n x = y }) ∧ A.to_finset.card = (n^2 - n + 2) / 2 :=
by
  sorry

end num_elements_in_A_l440_440438


namespace exists_coprime_subseq_l440_440905

-- Definitions and conditions from step a)
def is_finite {α : Type*} (s : set α) : Prop := ∃ n, s ∈ finset.univ.powerset n
def divides (a b : ℕ) : Prop := ∃ k, a * k = b

theorem exists_coprime_subseq (a : ℕ → ℕ) 
  (h : ∀ p : ℕ, nat.prime p → is_finite { i : ℕ | divides p (a i) }) :
  ∃ (i : ℕ → ℕ), (∀ m n : ℕ, m ≠ n → (nat.gcd (a (i m)) (a (i n)) = 1)) :=
sorry

end exists_coprime_subseq_l440_440905


namespace number_of_dogs_is_112_l440_440645

-- Definitions based on the given conditions.
def ratio_dogs_to_cats_to_bunnies (D C B : ℕ) : Prop := 4 * C = 7 * D ∧ 9 * C = 7 * B
def total_dogs_and_bunnies (D B : ℕ) (total : ℕ) : Prop := D + B = total

-- The hypothesis and conclusion of the problem.
theorem number_of_dogs_is_112 (D C B : ℕ) (x : ℕ) (h1: ratio_dogs_to_cats_to_bunnies D C B) (h2: total_dogs_and_bunnies D B 364) : D = 112 :=
by 
  sorry

end number_of_dogs_is_112_l440_440645


namespace pie_shop_earnings_l440_440680

-- Define the conditions
def price_per_slice : ℕ := 3
def slices_per_pie : ℕ := 10
def number_of_pies : ℕ := 6

-- Calculate the total slices
def total_slices : ℕ := number_of_pies * slices_per_pie

-- Calculate the total earnings
def total_earnings : ℕ := total_slices * price_per_slice

-- State the theorem
theorem pie_shop_earnings : total_earnings = 180 :=
by
  -- Proof can be skipped with a sorry
  sorry

end pie_shop_earnings_l440_440680


namespace angle_A_measure_l440_440125

theorem angle_A_measure (a b c: ℝ) (C : ℝ)
  (h1: 2 * a * real.cos C + c = 2 * b) : 
  A = π / 3 :=
sorry

end angle_A_measure_l440_440125


namespace translated_parabola_l440_440234

def f (x : ℝ) : ℝ := 2 * x^2

def translate_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x, f (x + a)

def translate_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ :=
  λ x, f x + b

theorem translated_parabola :
  translate_up (translate_left f 3) 4 = λ x, 2 * (x + 3)^2 + 4 :=
by
  -- proof would go here
  sorry

end translated_parabola_l440_440234


namespace range_t_of_hope_function_l440_440969

-- Define the conditions and the function
def is_monotonic_within (f : ℝ → ℝ) (C : set ℝ) : Prop :=
  ∀ x y ∈ C, x < y → f x ≤ f y

def is_hope_function (f : ℝ → ℝ) (C D : set ℝ) : Prop :=
  is_monotonic_within f C ∧ ∃ (m n : ℝ), m ∈ D ∧ n ∈ D ∧ m < n ∧ ∀ x ∈ set.Icc m n, f x ∈ set.Icc (m / 2) (n / 2)

noncomputable def log_hope_function (a t : ℝ) (x : ℝ) : ℝ :=
  log a (a^x + t)

-- Define the necessary properties and the theorem to prove the range of t
theorem range_t_of_hope_function (a : ℝ) (h_a : 0 < a ∧ a ≠ 1) (C D : set ℝ) :
  is_hope_function (log_hope_function a) C D → 
  ∃ (t : ℝ), (0 < t ∧ t < 1 / 4) :=
by
  sorry

end range_t_of_hope_function_l440_440969


namespace train_A_total_distance_l440_440621

variables (Speed_A : ℝ) (Time_meet : ℝ) (Total_Distance : ℝ)

def Distance_A_to_C (Speed_A Time_meet : ℝ) : ℝ := Speed_A * Time_meet
def Distance_B_to_C (Total_Distance Distance_A_to_C : ℝ) : ℝ := Total_Distance - Distance_A_to_C
def Additional_Distance_A (Speed_A Time_meet : ℝ) : ℝ := Speed_A * Time_meet
def Total_Distance_A (Distance_A_to_C Additional_Distance_A : ℝ) : ℝ :=
  Distance_A_to_C + Additional_Distance_A

theorem train_A_total_distance
  (h1 : Speed_A = 50)
  (h2 : Time_meet = 0.5)
  (h3 : Total_Distance = 120) :
  Total_Distance_A (Distance_A_to_C Speed_A Time_meet)
                   (Additional_Distance_A Speed_A Time_meet) = 50 :=
by 
  rw [Distance_A_to_C, Additional_Distance_A, Total_Distance_A]
  rw [h1, h2]
  norm_num

end train_A_total_distance_l440_440621


namespace revenue_december_multiple_average_l440_440100

variable (D : ℝ) -- Revenue in December
variable (N : ℝ) -- Revenue in November
variable (J : ℝ) -- Revenue in January

def revenue_in_november : N = (2/5) * D := sorry
def revenue_in_january : J = (2/25) * D := sorry
def average_revenue : ℝ := (N + J) / 2

theorem revenue_december_multiple_average : D = (25 / 6) * average_revenue := sorry

end revenue_december_multiple_average_l440_440100


namespace range_of_a_l440_440470

theorem range_of_a (a : ℝ) : (¬ (∃ x0 : ℝ, a * x0^2 + x0 + 1/2 ≤ 0)) → a > 1/2 :=
by
  sorry

end range_of_a_l440_440470


namespace exists_row_or_column_with_at_least_sqrt_n_diff_numbers_l440_440923

theorem exists_row_or_column_with_at_least_sqrt_n_diff_numbers 
  (n : ℕ)
  (grid : Π (i j : ℕ), {x // x ∈ finset.range (n + 1)}) 
  (h_grid_property : ∀ k : ℕ, k ≤ n → ∑ i in finset.range (n + 1), ∑ j in finset.range (n + 1), if grid i j = k then 1 else 0 = n) :
  ∃ (i : ℕ), (∃ S, finset.card S ≥ nat_ceil (real.sqrt n) ∧ ∀ x ∈ S, ∃ j : ℕ, j < n + 1 ∧ grid i j = x) ∨
  (∃ S, finset.card S ≥ nat_ceil (real.sqrt n) ∧ ∀ x ∈ S, ∃ i : ℕ, i < n + 1 ∧ grid i x = x) :=
sorry

end exists_row_or_column_with_at_least_sqrt_n_diff_numbers_l440_440923


namespace running_speed_l440_440328

theorem running_speed (R : ℝ) (walking_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) (half_distance : ℝ) (walking_time : ℝ) (running_time : ℝ)
  (h1 : walking_speed = 4)
  (h2 : total_distance = 16)
  (h3 : total_time = 3)
  (h4 : half_distance = total_distance / 2)
  (h5 : walking_time = half_distance / walking_speed)
  (h6 : running_time = half_distance / R)
  (h7 : walking_time + running_time = total_time) :
  R = 8 := 
sorry

end running_speed_l440_440328


namespace find_p_plus_q_l440_440374

noncomputable def radius_of_J := 12  -- Since \( \sqrt{144} - 12 = 12 \)
noncomputable def radius_of_K := 6
noncomputable def radius_of_L := 2

theorem find_p_plus_q :
  ∃ p q : ℕ, (sqrt p - q = radius_of_J) ∧ (p + q = 156) :=
by
  use 144
  use 12
  split
  {
    -- Proof that sqrt(144) - 12 = 12
    sorry
  }
  {
    -- Proof that 144 + 12 = 156
    sorry
  }

end find_p_plus_q_l440_440374


namespace average_of_possible_values_l440_440383

theorem average_of_possible_values :
  (average (x : ℝ) (h : ∃ (x : ℝ), sqrt (3 * x ^ 2 + 5) = sqrt 32) = 0) :=
sorry

noncomputable def average (x : ℝ) (h : ∃ (x : ℝ), sqrt (3 * x ^ 2 + 5) = sqrt 32 ) : ℝ :=
if hx : ∃ (x1 x2 : ℝ), sqrt (3 * x1 ^ 2 + 5) = sqrt 32 ∧ sqrt (3 * x2 ^ 2 + 5) = sqrt 32 then 
(x1 + x2) / 2 
else 
0

end average_of_possible_values_l440_440383


namespace compute_expression_l440_440281

theorem compute_expression : 1005^2 - 995^2 - 1003^2 + 997^2 = 8000 :=
by
  sorry

end compute_expression_l440_440281


namespace find_x_from_exponential_eq_l440_440435

theorem find_x_from_exponential_eq (x : ℕ) (h : 3^x + 3^x + 3^x + 3^x = 6561) : x = 6 := 
sorry

end find_x_from_exponential_eq_l440_440435


namespace digit_2023_in_expansion_of_7_div_18_l440_440002

theorem digit_2023_in_expansion_of_7_div_18 :
  (decimal_digit (2023) (7 / 18) = 3) :=
by 
  have exp : (decimal_expansion (7 / 18) = "0.\overline{38}") := sorry,
  have repeat : (repeating_sequence_length "38" = 2) := sorry,
  sorry

end digit_2023_in_expansion_of_7_div_18_l440_440002


namespace overall_gain_percent_l440_440177

theorem overall_gain_percent {initial_cost first_repair second_repair third_repair sell_price : ℝ} 
  (h1 : initial_cost = 800) 
  (h2 : first_repair = 150) 
  (h3 : second_repair = 75) 
  (h4 : third_repair = 225) 
  (h5 : sell_price = 1600) :
  (sell_price - (initial_cost + first_repair + second_repair + third_repair)) / 
  (initial_cost + first_repair + second_repair + third_repair) * 100 = 28 := 
by 
  sorry

end overall_gain_percent_l440_440177


namespace range_of_a_l440_440081

def A (x : ℝ) : Prop := x^2 - 6*x + 5 ≤ 0
def B (x a : ℝ) : Prop := x < a + 1

theorem range_of_a (a : ℝ) : (∃ x : ℝ, A x ∧ B x a) ↔ a > 0 := by
  sorry

end range_of_a_l440_440081


namespace unique_value_of_a_l440_440403

theorem unique_value_of_a 
  (x y a : ℝ) 
  (h1 : 1 + (4*x^2 - 12*x + 9)^2 + 2^(y+2) = a)
  (h2 : log 3 (x^2 - 3*x + 117/4) + 32 = a + log 3 (2*y + 3)) :
  a = 33 := 
sorry

end unique_value_of_a_l440_440403


namespace smallest_a_inequality_l440_440415

theorem smallest_a_inequality 
  (x : ℝ)
  (h1 : x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi)) : 
  (∃ a : ℝ, a = -2.52 ∧ (∀ x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi), 
    ( ((Real.sqrt (Real.cos x / Real.sin x)^2) - (Real.sqrt (Real.sin x / Real.cos x)^2))
    / ((Real.sqrt (Real.sin x)^2) - (Real.sqrt (Real.cos x)^2)) ) < a )) :=
  sorry

end smallest_a_inequality_l440_440415


namespace find_marks_in_physics_l440_440899

-- Define the marks for each subject and the average marks
def marks_english : ℕ := 76
def marks_math : ℕ := 60
def marks_chemistry : ℕ := 65
def marks_biology : ℕ := 82
def average_marks : ℕ := 71
def number_of_subjects : ℕ := 5

-- Define the problem statement to find the marks in Physics
theorem find_marks_in_physics (P : ℕ) 
  (total_marks : ℕ := average_marks * number_of_subjects)
  (marks_known_subjects : ℕ := marks_english + marks_math + marks_chemistry + marks_biology) 
  (P = total_marks - marks_known_subjects) : P = 72 :=
sorry

end find_marks_in_physics_l440_440899


namespace lcm_of_18_and_24_l440_440738

noncomputable def lcm_18_24 : ℕ :=
  Nat.lcm 18 24

theorem lcm_of_18_and_24 : lcm_18_24 = 72 :=
by
  sorry

end lcm_of_18_and_24_l440_440738


namespace problem_conditions_l440_440068
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x * Real.sin x + Real.cos x

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := Real.log (m * x + 1) + (1 - x) / (1 + x)

theorem problem_conditions (a k m : ℝ) (x1 : ℝ) (x2 : ℝ) :
  -- Given conditions
  (f (π/4) a = (√2 * π) / 8) →
  (∀ (x1 : ℝ), 0 ≤ x1 → ∃ (x2 : ℝ), 0 ≤ x2 ∧ x2 ≤ π/2 ∧ g x1 m ≥ f x2 a) →
  -- Proof of the corresponding mathematical problem
  a = 1 ∧ (∀ m, m ≥ 2) :=
by
sorry

end problem_conditions_l440_440068


namespace number_of_lawns_mowed_l440_440902

noncomputable def ChargePerLawn : ℕ := 33
noncomputable def TotalTips : ℕ := 30
noncomputable def TotalEarnings : ℕ := 558

theorem number_of_lawns_mowed (L : ℕ) 
  (h1 : ChargePerLawn * L + TotalTips = TotalEarnings) : L = 16 := 
by
  sorry

end number_of_lawns_mowed_l440_440902


namespace problem_statement_l440_440006

theorem problem_statement (a n : ℕ) (h_a : a ≥ 1) (h_n : n ≥ 1) :
  (∃ k : ℕ, (a + 1)^n - a^n = k * n) ↔ n = 1 := by
  sorry

end problem_statement_l440_440006


namespace S_15_value_l440_440047

/-
Given:
1. a sequence {a_n} with a_1 = 1 and a_2 = 2.
2. For any integer n > 1, the relationship S_(n+1) + S_(n-1) = 2(S_n + S_1),
   where S_k denotes the sum of the first k terms of the sequence.

Prove:
S_15 = 211
-/

noncomputable def a : ℕ → ℤ
| 0     := 0  -- Generally 0th term is considered 0.
| 1     := 1
| 2     := 2
| (n+3) := a (n+2) + 2

noncomputable def S : ℕ → ℤ
| 0     := 0
| (n+1) := S n + a (n+1)

/- Theorem to prove -/
theorem S_15_value : S 15 = 211 :=
by {
  sorry
}

end S_15_value_l440_440047


namespace max_vector_sum_l440_440432

/-- Given plane vectors a, b, and c, with angles and magnitudes specified, 
    we prove the maximum magnitude of their sum is 3 + sqrt(7). -/
theorem max_vector_sum (a b c : EuclideanSpace ℝ (Fin 2))
  (ha : ‖a‖ = 1 ∨ ‖a‖ = 2 ∨ ‖a‖ = 3)
  (hb : ‖b‖ = 1 ∨ ‖b‖ = 2 ∨ ‖b‖ = 3)
  (hc : ‖c‖ = 1 ∨ ‖c‖ = 2 ∨ ‖c‖ = 3)
  (hab : ∠ a b = real.pi / 3) :
  ‖a + b + c‖ ≤ 3 + real.sqrt 7 :=
sorry

end max_vector_sum_l440_440432


namespace colleen_pencils_l440_440137

theorem colleen_pencils (joy_pencils : ℕ) (pencil_cost : ℕ) (extra_cost : ℕ) (colleen_paid : ℕ)
  (H1 : joy_pencils = 30)
  (H2 : pencil_cost = 4)
  (H3 : extra_cost = 80)
  (H4 : colleen_paid = (joy_pencils * pencil_cost) + extra_cost) :
  colleen_paid / pencil_cost = 50 := 
by 
  -- Hints, if necessary
sorry

end colleen_pencils_l440_440137


namespace find_a_l440_440423

noncomputable def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem find_a : ∀ (a : ℝ), pure_imaginary ((a + complex.I) / (1 - complex.I)) → a = 1 :=
by
  intros a h
  have h_simp : (a + complex.I) / (1 - complex.I) = ((a - 1) / 2) + ((a + 1) / 2) * complex.I :=
    by field_simp [complex.I_re, complex.I_im, complex.I_mul_I, complex.of_real_1, mul_one]
  simp [pure_imaginary, complex.I_re, complex.I_im, complex.I_mul_I, complex.of_real_1, mul_one, h_simp] at h
  cases h with h1 h2
  exact sorry

end find_a_l440_440423


namespace distance_from_point_to_line_example_l440_440736

noncomputable def distance_point_to_line (P : ℝ × ℝ × ℝ) (A : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ :=
let B := (P.1 - A.1, P.2 - A.2, P.3 - A.3) in
let projection_length := ((B.1 * v.1 + B.2 * v.2 + B.3 * v.3)^2) / (v.1^2 + v.2^2 + v.3^2) in
real.sqrt ((B.1^2 + B.2^2 + B.3^2) - projection_length)

theorem distance_from_point_to_line_example :
  distance_point_to_line (3, 5, -1) (2, 4, 6) (4, 3, -1) = real.sqrt 51 := 
sorry

end distance_from_point_to_line_example_l440_440736


namespace eccentricity_range_constant_lambda_exists_l440_440982

variable {a b c : ℝ}
noncomputable def is_ellipse := c = sqrt(a^2 - b^2)

theorem eccentricity_range (a b c e : ℝ) 
  (h_ellipse : is_ellipse)
  (c_sq : c^2 = a^2 - b^2) 
  (range_condition : c^2 ≤ b^2 ≤ 3 * c^2) 
  (e_def : e = c / a):
  1 / 2 ≤ e ∧ e ≤ sqrt(2) / 2 := 
sorry

theorem constant_lambda_exists (a b c e λ : ℝ) 
  (h_ellipse : is_ellipse)
  (h_e : e = 1 / 2)
  (lambda_pos : λ > 0)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (F1 : ℝ × ℝ)
  (F2 : ℝ × ℝ)
  (lambda_condition : angle B A F1 = λ * angle B F1 A) :
  λ = 2 :=
sorry

end eccentricity_range_constant_lambda_exists_l440_440982


namespace alien_abduction_l440_440353

theorem alien_abduction :
  let total_abducted := 500
  let percentage_returned := 0.675
  let taken_to_zog := 55
  ∃ home_planet_count, home_planet_count = total_abducted - (nat.floor (percentage_returned * total_abducted)) - taken_to_zog ∧ home_planet_count = 108
:=
by
  let total_abducted := 500
  let percentage_returned := 0.675
  let taken_to_zog := 55

  have returned := nat.floor (percentage_returned * total_abducted)
  have not_returned := total_abducted - returned
  have home_planet_count := not_returned - taken_to_zog

  exists home_planet_count
  split
  {
    refl
  }
  {
    sorry -- proof that home_planet_count = 108
  }

end alien_abduction_l440_440353


namespace right_triangle_area_ratio_l440_440620

theorem right_triangle_area_ratio (S S' : ℝ) :
  (∀ (b c : ℝ), S = (1/2) * b * c ∧
               S' = (2 * ((b * c) / (b + c + sqrt (b^2 + c^2)))^2)) →
  S / S' ≥ 3 + 2 * sqrt 2 :=
by
  sorry

end right_triangle_area_ratio_l440_440620


namespace complex_number_a_eq_1_l440_440057

theorem complex_number_a_eq_1 
  (a : ℝ) 
  (h : ∃ b : ℝ, (a - b * I) / (1 + I) = 0 + b * I) : 
  a = 1 := 
sorry

end complex_number_a_eq_1_l440_440057


namespace collinear_points_min_value_l440_440145

open Real

/-- Let \(\overrightarrow{e_{1}}\) and \(\overrightarrow{e_{2}}\) be two non-collinear vectors in a plane,
    \(\overrightarrow{AB} = (a-1) \overrightarrow{e_{1}} + \overrightarrow{e_{2}}\),
    \(\overrightarrow{AC} = b \overrightarrow{e_{1}} - 2 \overrightarrow{e_{2}}\),
    with \(a > 0\) and \(b > 0\). 
    If points \(A\), \(B\), and \(C\) are collinear, then the minimum value of \(\frac{1}{a} + \frac{2}{b}\) is \(4\). -/
theorem collinear_points_min_value 
  (e1 e2 : ℝ) 
  (H_non_collinear : (e1 ≠ 0 ∨ e2 ≠ 0))
  (a b : ℝ) 
  (H_a_pos : a > 0) 
  (H_b_pos : b > 0)
  (H_collinear : ∃ x : ℝ, (a - 1) * e1 + e2 = x * (b * e1 - 2 * e2)) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + (1/2) * b = 1 ∧ (∀ a b : ℝ, (1/a) + (2/b) ≥ 4) :=
sorry

end collinear_points_min_value_l440_440145


namespace calc_expression_l440_440370

theorem calc_expression :
  (12^4 + 375) * (24^4 + 375) * (36^4 + 375) * (48^4 + 375) * (60^4 + 375) /
  ((6^4 + 375) * (18^4 + 375) * (30^4 + 375) * (42^4 + 375) * (54^4 + 375)) = 159 :=
by
  sorry

end calc_expression_l440_440370


namespace curved_surface_area_of_cone_l440_440984

noncomputable def curvedSurfaceAreaCone (h l : ℝ) (r : ℝ) :=
  π * r * l

theorem curved_surface_area_of_cone :
  ∀ (h l : ℝ), h = 8 ∧ l = 10 → ∃ r, r = Real.sqrt (l^2 - h^2) ∧ curvedSurfaceAreaCone h l r = 60 * π :=
by
  intros h l
  assume hl
  sorry

end curved_surface_area_of_cone_l440_440984


namespace inequality_holds_l440_440155

variable {a : ℕ → ℝ}

noncomputable def satisfies_condition (a : ℕ → ℝ) :=
  ∀ i j : ℕ, a (i + j) ≤ a i + a j

theorem inequality_holds (a : ℕ → ℝ) (h : satisfies_condition a) : 
  ∀ n : ℕ, (∑ i in finset.range n, a (i + 1) / (i + 1)) ≥ a n := 
by
  sorry

end inequality_holds_l440_440155


namespace exists_divisible_pair_l440_440394

def is_valid_pair (three_digit : ℕ) (two_digit : ℕ) : Prop :=
  (three_digit / 100 ∈ {1, 2, 3, 4, 5}) ∧
  ((three_digit % 100) / 10 ∈ {1, 2, 3, 4, 5}) ∧
  (three_digit % 10 ∈ {1, 2, 3, 4, 5}) ∧
  (two_digit / 10 ∈ {1, 2, 3, 4, 5}) ∧
  (two_digit % 10 ∈ {1, 2, 3, 4, 5}) ∧
  (three_digit / 100 ≠ (three_digit % 100) / 10) ∧
  (three_digit / 100 ≠ three_digit % 10) ∧
  ((three_digit % 100) / 10 ≠ three_digit % 10) ∧
  (two_digit / 10 ≠ two_digit % 10) ∧
  (three_digit % 10 ≠ two_digit / 10) ∧
  (three_digit % 10 ≠ two_digit % 10) ∧
  (three_digit / 100 ≠ two_digit / 10) ∧
  (three_digit / 100 ≠ two_digit % 10) ∧
  ((three_digit % 100) / 10 ≠ two_digit / 10) ∧
  ((three_digit % 100) / 10 ≠ two_digit % 10)

theorem exists_divisible_pair :
  ∃ (three_digit two_digit : ℕ), 
    (three_digit ∈ {123, 124, 125, 132, 134, 135, 142, 143, 145, 153, 154, 234, 235, 245, 312, 314, 315, 324, 325, 345, 412, 413, 415, 423, 425, 435, 512, 513, 514, 523, 524, 534}) ∧
    (two_digit ∈ {12, 13, 14, 15, 21, 23, 24, 25, 31, 32, 34, 35, 41, 42, 43, 45, 51, 52, 53, 54}) ∧
    is_valid_pair three_digit two_digit ∧
    three_digit % two_digit = 0 := sorry

end exists_divisible_pair_l440_440394


namespace curve_intersection_distance_l440_440116

theorem curve_intersection_distance :
  let l := (line.slope 1) (point.mk (-2) (-4))
  let C := curve.mk_polar (λ θ ρ, ρ * (sin θ)^2 - 4 * cos θ = 0)
  let P := point.mk (-2) (-4)
  let M, N := intersection_points l C
  distance P M + distance P N = 12 * sqrt 2 := sorry

end curve_intersection_distance_l440_440116


namespace decreasing_condition_log_sum_condition_l440_440830

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x * Real.log x - (1/2) * m * x^2 - x

def f_prime (x : ℝ) (m : ℝ) : ℝ := Real.log x - m * x

theorem decreasing_condition (m : ℝ) : (∀ x : ℝ, 0 < x → f_prime x m ≤ 0) ↔ m ≥ 1 / Real.exp 1 :=
by
  sorry

theorem log_sum_condition (x1 x2 m : ℝ) (hx : 0 < x1 ∧ x1 < x2 ∧ 0 < x2) (extreme_pts: f_prime x1 m = 0 ∧ f_prime x2 m = 0) :
  Real.log x1 + Real.log x2 > 2 :=
by
  sorry

end decreasing_condition_log_sum_condition_l440_440830


namespace particle_speed_l440_440326

-- Define the position functions x(t) and y(t)
def x (t : ℝ) : ℝ := t^2 + 2 * t + 3
def y (t : ℝ) : ℝ := 3 * t^2 - t + 2

-- Define the velocity components as derivatives of the position functions
def vx (t : ℝ) : ℝ := deriv (λ t, x t) t
def vy (t : ℝ) : ℝ := deriv (λ t, y t) t

-- Define the speed function v(t) as the magnitude of the velocity vector
def speed (t : ℝ) : ℝ := Real.sqrt ((vx t)^2 + (vy t)^2)

-- State the theorem
theorem particle_speed (t : ℝ) : speed t = Real.sqrt (40 * t^2 - 4 * t + 5) :=
sorry

end particle_speed_l440_440326


namespace mary_balloon_count_l440_440944

theorem mary_balloon_count (n m : ℕ) (hn : n = 7) (hm : m = 4 * n) : m = 28 :=
by
  sorry

end mary_balloon_count_l440_440944


namespace quadratic_root_conditions_l440_440389

theorem quadratic_root_conditions : ∃ p q : ℝ, (p - 1)^2 - 4 * q > 0 ∧ (p + 1)^2 - 4 * q > 0 ∧ p^2 - 4 * q < 0 := 
sorry

end quadratic_root_conditions_l440_440389


namespace find_k_l440_440043

theorem find_k (x₁ x₂ k : ℝ) (hx : x₁ + x₂ = 3) (h_prod : x₁ * x₂ = k) (h_cond : x₁ * x₂ + 2 * x₁ + 2 * x₂ = 1) : k = -5 :=
by
  sorry

end find_k_l440_440043


namespace find_2023rd_digit_past_decimal_of_7_over_18_l440_440003

-- Definitions that come from conditions in step a)
def decimal_expansion (x : ℚ) : ℕ → ℕ := sorry
def repeating_block (x : ℚ) : string := "38"

-- Conditions from step a)
def expansion_of_7_over_18 : decimal_expansion (7 / 18) = sorry := sorry
def repeating_block_of_7_over_18 : repeating_block (7 / 18) = "38" := sorry
def length_of_repeating_block_of_7_over_18 : (repeating_block (7 / 18)).length = 2 := by
  simp [repeating_block]

-- Proof problem statement
theorem find_2023rd_digit_past_decimal_of_7_over_18 :
  decimal_expansion (7 / 18) 2023 = 3 :=
sorry

end find_2023rd_digit_past_decimal_of_7_over_18_l440_440003


namespace LM_lt_LK_l440_440168

variable (A B C K L M : Point) -- Points in the plane.

-- Given conditions as hypotheses
-- 1. K, L, and M lie on AB, AC, and BC respectively
hypothesis (hK : K ∈ segment A B)
hypothesis (hL : L ∈ segment A C)
hypothesis (hM : M ∈ segment B C)

-- 2. Angles
hypothesis (h_angle_eq1 : ∠ A = ∠ K L M)
hypothesis (h_angle_eq2 : ∠ A = ∠ C)

-- 3. Length condition
hypothesis (h_length : dist A L + dist L M + dist M B > dist C L + dist L K + dist K B)

-- To Prove: LM < LK
theorem LM_lt_LK : dist L M < dist L K := 
by
  sorry

end LM_lt_LK_l440_440168


namespace business_card_exchanges_l440_440653

theorem business_card_exchanges :
  let n := 10,
  let exchanges := (n * (n - 1)) / 2
  in exchanges = 45 :=
by
  let n := 10
  let exchanges := (n * (n - 1)) / 2
  have calculation : (n * (n - 1)) / 2 = 45 := by sorry
  exact calculation

end business_card_exchanges_l440_440653


namespace slope_of_line_l440_440078

  theorem slope_of_line : 
    ∀ t : ℝ, (x y : ℝ), (x = 3 - t * Real.sin (20 * Real.pi / 180)) → 
              (y = 2 + t * Real.cos (70 * Real.pi / 180)) → 
              (y - 2) / (x - 3) = -1 := 
  by
    intros t x y hx hy
    rw [hx, hy]
    sorry
  
end slope_of_line_l440_440078


namespace count_multiples_10_but_not_3_or_7_l440_440843

def multiple_of (m n : ℕ) : Prop := n % m = 0

def multiples_between (m start stop : ℕ) : List ℕ :=
  List.filter (multiple_of m) (List.range' start (stop - start + 1))

def exclude_multiples (l : List ℕ) (m : ℕ) : List ℕ :=
  List.filter (fun n => ¬ multiple_of m n) l

theorem count_multiples_10_but_not_3_or_7 : 
  (multiples_between 10 1 300)
  |> exclude_multiples 3
  |> exclude_multiples 7
  |>.length = 17 := sorry

end count_multiples_10_but_not_3_or_7_l440_440843


namespace parker_bed_time_l440_440169

def sleep_duration : Time := Time.mk 7 0 0

def wake_up_time : Time := Time.mk 9 0 0

def bed_time (wake_up : Time) (sleep_duration : Time) : Time :=
  wake_up - sleep_duration

theorem parker_bed_time : bed_time wake_up_time sleep_duration = Time.mk 2 0 0 :=
sorry

end parker_bed_time_l440_440169


namespace blueberries_to_bonnies_ratio_l440_440228

-- Defining the variables and conditions **(as direct definitions from the problem)**
variables (B : ℕ) (F : ℚ) (total_fruits : ℕ) (bonnies : ℕ)

-- Given conditions
def condition1 : Prop := bonnies = 60
def condition2 : Prop := 3 * B + B + bonnies = total_fruits
def condition3 : Prop := total_fruits = 240
def condition4 : Prop := F = 3 / 4
def condition5 : Prop := B = F * bonnies

-- Theorem to prove the ratio B/bonnies = 3/4
theorem blueberries_to_bonnies_ratio (B F : ℕ) (bonnies : ℕ) :
  condition1 bonnies ∧ condition2 B bonnies ∧ condition3 total_fruits ∧ condition4 F ∧ condition5 B F bonnies →
  B / bonnies = 3 / 4 :=
by
  intros h,
  sorry 

end blueberries_to_bonnies_ratio_l440_440228


namespace probability_of_four_ones_approx_l440_440249

noncomputable def probability_of_four_ones_in_twelve_dice : ℚ :=
  (nat.choose 12 4 : ℚ) * (1 / 6 : ℚ) ^ 4 * (5 / 6 : ℚ) ^ 8

theorem probability_of_four_ones_approx :
  probability_of_four_ones_in_twelve_dice ≈ 0.089 :=
sorry

end probability_of_four_ones_approx_l440_440249


namespace jims_taxi_total_charge_l440_440646

theorem jims_taxi_total_charge :
  let initial_fee := 2.0
  let additional_fee_per_increment := 0.35
  let distance := 3.6
  let increment := 2 / 5
  let num_increments := distance / increment
  let additional_charge := num_increments * additional_fee_per_increment
  let total_charge := initial_fee + additional_charge
  in total_charge = 5.15 := by
  sorry

end jims_taxi_total_charge_l440_440646


namespace similar_quadrilaterals_cyclic_quadrilaterals_l440_440781

variables {A1 B1 C1 D1 P : Type}
variables [ConvexQuadrilateral A1 B1 C1 D1]

-- Reflect function for point across a line segment
def reflect (P : Type) (line : Type) : Type := sorry

-- Recursive reflection definitions
noncomputable def A (k : ℕ) : Type :=
  if k = 1 then A1 else reflect P (B (k-1), A (k-1))
noncomputable def B (k : ℕ) : Type :=
  if k = 1 then B1 else reflect P (C (k-1), B (k-1))
noncomputable def C (k : ℕ) : Type :=
  if k = 1 then C1 else reflect P (D (k-1), C (k-1))
noncomputable def D (k : ℕ) : Type :=
  if k = 1 then D1 else reflect P (A (k-1), D (k-1))

-- Definition of similarity
def similar (quad1 quad2 : Type) : Prop := sorry

-- Definition of cyclicity
def cyclic (quad : Type) : Prop := sorry

-- Statement of the first proof problem
theorem similar_quadrilaterals (k : ℕ) (hk : k ∈ {1, 5, 9}) :
  similar (A k, B k, C k, D k) (A 1997, B 1997, C 1997, D 1997) := sorry

-- Statement of the second proof problem
theorem cyclic_quadrilaterals (h : cyclic (A 1997, B 1997, C 1997, D 1997)) (k : ℕ) 
  (hk : k ∈ {1, 3, 5, 7, 9, 11}) :
  cyclic (A k, B k, C k, D k) := sorry

end similar_quadrilaterals_cyclic_quadrilaterals_l440_440781


namespace hyperbola_eccentricity_l440_440966

theorem hyperbola_eccentricity (a b c: ℝ) (h_asymptotes : ∀ x y, x = 2 * y ∨ x = -2 * y) :
  c^2 = a^2 + b^2 → 
  (c / a = sqrt (1 + (b^2 / a^2)) → 
  (c / a = sqrt 5 ∨ c / a = sqrt 5 / 2)) :=
sorry

end hyperbola_eccentricity_l440_440966


namespace general_formula_for_seq_a_sum_of_first_n_terms_l440_440791

noncomputable def sequence_a (p : ℝ) : ℕ → ℝ
| 1       := 1 / 2
| (n + 1) := p * (sequence_a p n)

theorem general_formula_for_seq_a (p : ℝ) (h_pos : p > 0) (h_arith_mean : 1 / 2 = 6 * (sequence_a p 3) + (sequence_a p 2) / 2) :
  ∀ n : ℕ, sequence_a p n = 1 / (2 ^ n) := sorry

noncomputable def sequence_b (n : ℕ) (seq_a : ℕ → ℝ) : ℕ → ℝ
| n       := (2 * n + 1) / (seq_a n)

noncomputable def sum_T (n : ℕ) (sequence_b : ℕ → ℝ) : ℝ :=
(∑ i in finset.range n, sequence_b i)

theorem sum_of_first_n_terms (p : ℝ) (h_pos : p > 0) (h_arith_mean : 1 / 2 = 6 * (sequence_a p 3) + (sequence_a p 2) / 2) :
  ∀ n : ℕ, (∀ k, sequence_a p k * sequence_b k = 2 * k + 1) →
  sum_T n (sequence_b n (sequence_a p)) = 2 + (2 * n - 1) * 2^(n + 1) := sorry

end general_formula_for_seq_a_sum_of_first_n_terms_l440_440791


namespace find_2023rd_digit_past_decimal_of_7_over_18_l440_440004

-- Definitions that come from conditions in step a)
def decimal_expansion (x : ℚ) : ℕ → ℕ := sorry
def repeating_block (x : ℚ) : string := "38"

-- Conditions from step a)
def expansion_of_7_over_18 : decimal_expansion (7 / 18) = sorry := sorry
def repeating_block_of_7_over_18 : repeating_block (7 / 18) = "38" := sorry
def length_of_repeating_block_of_7_over_18 : (repeating_block (7 / 18)).length = 2 := by
  simp [repeating_block]

-- Proof problem statement
theorem find_2023rd_digit_past_decimal_of_7_over_18 :
  decimal_expansion (7 / 18) 2023 = 3 :=
sorry

end find_2023rd_digit_past_decimal_of_7_over_18_l440_440004


namespace task_completion_days_l440_440690

theorem task_completion_days (a b c d : ℝ) 
    (h1 : 1/a + 1/b = 1/8)
    (h2 : 1/b + 1/c = 1/6)
    (h3 : 1/c + 1/d = 1/12) :
    1/a + 1/d = 1/24 :=
by
  sorry

end task_completion_days_l440_440690


namespace smallest_positive_period_of_f_max_value_of_f_on_interval_l440_440463

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + (Real.sqrt 3) * (Real.sin x) * (Real.cos x)

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ ε > 0, ε < T → ∃ x : ℝ, f (x + ε) ≠ f x) :=
sorry

theorem max_value_of_f_on_interval :
  ∃ x ∈ Icc (0 : ℝ) (Real.pi / 2), ∀ y ∈ Icc (0 : ℝ) (Real.pi / 2), f y ≤ f x ∧ f x = (3 / 2) :=
sorry

end smallest_positive_period_of_f_max_value_of_f_on_interval_l440_440463


namespace sum_reciprocal_g_l440_440917

-- Definition of g based on provided condition
def g (n : ℕ) : ℕ := 
  let m := Nat.root 3 n  -- cube root of n truncated to nearest integer
  if (m ^ 3 + 3 * m ^ 2 / 2 + 3 * m / 4 - 1 / 8 < n) && (n < (m + 1/2) ^ 3)
  then m
  else m + 1

-- The main theorem we aim to prove based on the translated problem
theorem sum_reciprocal_g : (∑ k in Finset.range 4095.succ, (1 : ℚ) / (g k)) = 120 := 
by
  -- Insert the actual proof or 'sorry' to skip it.
  sorry

end sum_reciprocal_g_l440_440917


namespace original_matchsticks_l440_440932

-- Define the conditions
def matchstick_house := 10
def matchstick_tower := 15
def matchstick_bridge := 25
def houses_created := 30
def towers_created := 20
def bridges_created := 10
def matchsticks_used := houses_created * matchstick_house + towers_created * matchstick_tower + bridges_created * matchstick_bridge
def matchsticks_half := matchsticks_used

-- The theorem stating the problem
theorem original_matchsticks (houses_created: ℕ) (towers_created: ℕ) (bridges_created: ℕ)
                            (matchstick_house: ℕ) (matchstick_tower: ℕ) (matchstick_bridge: ℕ) : 
    (houses_created = 30) → 
    (towers_created = 20) → 
    (bridges_created = 10) → 
    (matchstick_house = 10) → 
    (matchstick_tower = 15) → 
    (matchstick_bridge = 25) → 
    2 * matchsticks_used = 1700 := 
by
  intro H1 H2 H3 H4 H5 H6
  rw [H1, H2, H3, H4, H5, H6]
  show matchsticks_used = 850
  sorry

end original_matchsticks_l440_440932


namespace coefficient_of_x9_l440_440583

theorem coefficient_of_x9 :
  (let coeff := (-1)^9 * (Nat.choose 12 9)
   in coeff = -220) :=
by
  let coeff := (-1)^9 * (Nat.choose 12 9)
  sorry

end coefficient_of_x9_l440_440583


namespace min_value_f_when_a_eq_1_value_of_a_for_f_geq_1_l440_440555

noncomputable def f (x a : ℝ) : ℝ := exp x - a * x

theorem min_value_f_when_a_eq_1 :
  (∀ x : ℝ, f x 1 ≥ 1) :=
  by
  sorry

theorem value_of_a_for_f_geq_1 :
  (∀ x : ℝ, f x a ≥ 1) → a = 1 :=
  by 
  sorry

end min_value_f_when_a_eq_1_value_of_a_for_f_geq_1_l440_440555


namespace probability_exactly_four_1s_l440_440255

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_four_1s_in_12_dice : ℝ :=
  let n := 12
  let k := 4
  let p := 1 / 6
  let q := 5 / 6
  (binomial_coefficient n k : ℝ) * p^k * q^(n - k)

theorem probability_exactly_four_1s : probability_of_four_1s_in_12_dice ≈ 0.089 :=
  by
  sorry

end probability_exactly_four_1s_l440_440255


namespace max_sectional_area_l440_440224

theorem max_sectional_area (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) :
  ∃ (S : ℝ), S = max (a * (sqrt (b^2 + c^2))) (max (b * (sqrt (a^2 + c^2))) (c * (sqrt (a^2 + b^2)))) ∧
    S = c * (sqrt (a^2 + b^2)) :=
by sorry

end max_sectional_area_l440_440224


namespace stock_reaches_N_fourth_time_l440_440601

noncomputable def stock_at_k (c0 a b : ℝ) (k : ℕ) : ℝ :=
  if k % 2 = 0 then c0 + (k / 2) * (a - b)
  else c0 + (k / 2 + 1) * a - (k / 2) * b

theorem stock_reaches_N_fourth_time (c0 a b N : ℝ) (hN3 : ∃ k1 k2 k3 : ℕ, k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≠ k3 ∧ stock_at_k c0 a b k1 = N ∧ stock_at_k c0 a b k2 = N ∧ stock_at_k c0 a b k3 = N) :
  ∃ k4 : ℕ, k4 ≠ k1 ∧ k4 ≠ k2 ∧ k4 ≠ k3 ∧ stock_at_k c0 a b k4 = N := 
sorry

end stock_reaches_N_fourth_time_l440_440601


namespace probability_all_quit_from_same_tribe_l440_440217

def num_ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

def num_ways_to_choose_3_from_20 : ℕ := num_ways_to_choose 20 3
def num_ways_to_choose_3_from_10 : ℕ := num_ways_to_choose 10 3

theorem probability_all_quit_from_same_tribe :
  (num_ways_to_choose_3_from_10 * 2).toRat / num_ways_to_choose_3_from_20.toRat = 4 / 19 := 
  by
  sorry

end probability_all_quit_from_same_tribe_l440_440217


namespace count_valid_permutations_l440_440406

-- Define the permutation predicate
def valid_permutation (a : Fin 10 → ℕ) : Prop :=
  (∀ i : Fin 9, a i.succ ≥ a i - 1) ∧
  Multiset.ofFn a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Statement to be proved
theorem count_valid_permutations : 
  (Multiset.card (Multiset.filter valid_permutation (Finset.univ : Finset (Fin 10 → ℕ)).val)) = 512 :=
sorry

end count_valid_permutations_l440_440406


namespace fifth_selected_individual_is_443_l440_440976

-- Define the population size
def population_size : ℕ := 600

-- Example random number table (excerpt from rows 7 to 9)
def random_table : list (list ℕ) :=
  [ [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76, 63, 01, 63],
    [78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79, 33, 21, 12, 34, 29, 78],
    [64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54] ]

-- Selection starts from 8th row and 8th column, reading to the right, excluding numbers > population_size
def select_valid_numbers (table: list (list ℕ)) (start_row start_col : ℕ) (pop_size : ℕ) : list ℕ :=
  table.nth start_row |>.get_or_else [] |>.drop start_col |>.filter (λ n => n ≤ pop_size)

-- Define the selection process to get the nth valid individual
def nth_selected_individual (table : list (list ℕ)) (start_row start_col : ℕ) (pop_size nth : ℕ) : ℕ :=
  (select_valid_numbers table start_row start_col pop_size).nth (nth - 1) |>.get_or_else 0

-- Theorem: Prove that the number of the 5th individual selected is 443
theorem fifth_selected_individual_is_443 : 
  nth_selected_individual random_table 0 7 population_size 5 = 443 := 
  sorry

end fifth_selected_individual_is_443_l440_440976


namespace angela_age_in_5_years_l440_440357

-- Define the variables representing Angela and Beth's ages.
variable (A B : ℕ)

-- State the conditions as hypotheses.
def condition_1 : Prop := A = 4 * B
def condition_2 : Prop := (A - 5) + (B - 5) = 45

-- State the final proposition that Angela will be 49 years old in five years.
theorem angela_age_in_5_years (h1 : condition_1 A B) (h2 : condition_2 A B) : A + 5 = 49 := by
  sorry

end angela_age_in_5_years_l440_440357


namespace ratio_of_areas_l440_440728

theorem ratio_of_areas (side_length : ℝ) (h_side_length : side_length > 0) : 
  (let quarter := side_length / 4
       inscribed_side := Real.sqrt ((3 * quarter)^2 + (3 * quarter)^2)) in
  (inscribed_side ^ 2) / (side_length ^ 2) = 9 / 8 :=
by
  let quarter := side_length / 4
  let inscribed_side := Real.sqrt ((3 * quarter)^2 + (3 * quarter)^2)
  suffices : inscribed_side = 3 * Real.sqrt 2 * quarter / 4
  rw [this, real.pow_two]
  have : (3 * Real.sqrt 2 * quarter / 4) ^ 2 = (9 / 8) * (side_length ^ 2),
  sorry,
  exact (this).symm

end ratio_of_areas_l440_440728


namespace probability_point_within_circle_l440_440867

def fair_dice_outcomes : Finset (ℕ × ℕ) :=
  Finset.pi (Finset.range 6) (Finset.range 6)

def circle (x y : ℕ) : Prop :=
  x^2 + y^2 < 17

theorem probability_point_within_circle :
  (∑ p in fair_dice_outcomes, if circle p.1 p.2 then 1 else 0 : ℝ) / (Finset.card fair_dice_outcomes : ℝ) = 2 / 9 :=
by
  -- proof omitted
  sorry

end probability_point_within_circle_l440_440867


namespace max_value_expression_l440_440910

theorem max_value_expression (k : ℕ) (a b c : ℝ) (h : k > 0) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (habc : a + b + c = 3 * k) :
  a^(3 * k - 1) * b + b^(3 * k - 1) * c + c^(3 * k - 1) * a + k^2 * a^k * b^k * c^k ≤ (3 * k - 1)^(3 * k - 1) :=
sorry

end max_value_expression_l440_440910


namespace minimize_travel_time_l440_440693

theorem minimize_travel_time
  (a b c d : ℝ)
  (v₁ v₂ v₃ v₄ : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > d)
  (h4 : v₁ > v₂)
  (h5 : v₂ > v₃)
  (h6 : v₃ > v₄) : 
  (a / v₁ + b / v₂ + c / v₃ + d / v₄) ≤ (a / v₁ + b / v₄ + c / v₃ + d / v₂) :=
sorry

end minimize_travel_time_l440_440693


namespace min_value_of_f_on_interval_l440_440405

noncomputable def f (x : ℝ) : ℝ := 6 / (2^x + 3^x)

theorem min_value_of_f_on_interval : ∀ x ∈ Icc (-1 : ℝ) 1, f 1 ≤ f x :=
by
  intros x hx
  sorry

end min_value_of_f_on_interval_l440_440405


namespace reciprocal_opposite_abs_val_l440_440980

theorem reciprocal_opposite_abs_val (a : ℚ) (h : a = -1 - 2/7) :
    (1 / a = -7/9) ∧ (-a = 1 + 2/7) ∧ (|a| = 1 + 2/7) := 
sorry

end reciprocal_opposite_abs_val_l440_440980


namespace probability_of_24_is_1_div_1296_l440_440749

def sum_of_dice_is_24 (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 + d2 + d3 + d4 = 24

def probability_of_six (d : ℕ) : Rat :=
  if d = 6 then 1 / 6 else 0

def probability_of_sum_24 (d1 d2 d3 d4 : ℕ) : Rat :=
  (probability_of_six d1) * (probability_of_six d2) * (probability_of_six d3) * (probability_of_six d4)

theorem probability_of_24_is_1_div_1296 :
  (probability_of_sum_24 6 6 6 6) = 1 / 1296 :=
by
  sorry

end probability_of_24_is_1_div_1296_l440_440749


namespace geometric_sequence_reciprocals_sum_l440_440784

theorem geometric_sequence_reciprocals_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), 
    (a 1 = 2) ∧ 
    (a 1 + a 3 + a 5 = 14) ∧ 
    (∀ n : ℕ, a (n + 1) = a n * q) → 
      (1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8) :=
sorry

end geometric_sequence_reciprocals_sum_l440_440784


namespace value_of_V3_l440_440624

-- Define the polynomial function using Horner's rule
def f (x : ℤ) := (((((2 * x + 0) * x - 3) * x + 2) * x + 1) * x - 3)

-- Define the value of x
def x : ℤ := 2

-- Prove the value of V_3 when x = 2
theorem value_of_V3 : f x = 12 := by
  sorry

end value_of_V3_l440_440624


namespace probability_divisor_of_12_l440_440667

open Probability

def divisors_of_12 := {1, 2, 3, 4, 6}

theorem probability_divisor_of_12 :
  ∃ (fair_die_roll : ProbabilityMeasure (Fin 6)), 
    P (fun x => x.val + 1 ∈ divisors_of_12) = 5 / 6 := 
by
  sorry

end probability_divisor_of_12_l440_440667


namespace correct_statements_l440_440225

-- Define the scores
def StudentA_scores : list ℝ := [138, 127, 131, 132, 128, 135]
def StudentB_scores : list ℝ := [130, 116, 128, 115, 126, 120]
def StudentC_scores : list ℝ := [108, 105, 113, 112, 115, 123]
def ClassAvg_scores : list ℝ := [128.2, 118.3, 125.4, 120.3, 115.7, 122.1]

-- Define predicates for the statements
def StudentA_above_avg := ∀ (n : ℕ), n < StudentA_scores.length → StudentA_scores.nth_le n _ > ClassAvg_scores.nth_le n _
def StudentB_fluctuating := ∃ i j : ℕ, i < StudentB_scores.length ∧ j < StudentB_scores.length ∧ 
  ((StudentB_scores.nth_le i _ > ClassAvg_scores.nth_le i _) ∧ (StudentB_scores.nth_le j _ < ClassAvg_scores.nth_le j _))
def StudentC_below_avg := ∀ (n : ℕ), n < StudentC_scores.length - 1 → StudentC_scores.nth_le n _ < ClassAvg_scores.nth_le n _
def StudentC_improving := ∀ n : ℕ, n < StudentC_scores.length - 1 → 
  StudentC_scores.nth_le n _ < ClassAvg_scores.nth_le n _ ∧ StudentC_scores.nth_le (n + 1) _ ≥ ClassAvg_scores.nth_le (n + 1) _

-- The theorem to prove, based on the conditions
theorem correct_statements : StudentA_above_avg ∧ StudentB_fluctuating ∧ ∀ n : ℕ, (n < StudentC_scores.length → (StudentC_below_avg n ∨ StudentC_improving n)) := 
by {
  sorry
}

end correct_statements_l440_440225


namespace total_pay_is_186_l440_440322

-- Define the conditions
def regular_rate : ℕ := 3 -- dollars per hour
def regular_hours : ℕ := 40 -- hours
def overtime_rate_multiplier : ℕ := 2
def overtime_hours : ℕ := 11

-- Calculate the regular pay
def regular_pay : ℕ := regular_hours * regular_rate

-- Calculate the overtime pay
def overtime_rate : ℕ := regular_rate * overtime_rate_multiplier
def overtime_pay : ℕ := overtime_hours * overtime_rate

-- Calculate the total pay
def total_pay : ℕ := regular_pay + overtime_pay

-- The statement to be proved
theorem total_pay_is_186 : total_pay = 186 :=
by 
  sorry

end total_pay_is_186_l440_440322


namespace max_elements_l440_440333

variables (S : Set α) (R : α → α → Prop)
-- Condition 1
axiom cond1 : ∀ (a b : α), a ≠ b → (R a b ↔ ¬R b a)
-- Condition 2
axiom cond2 : ∀ (a b c : α), a ≠ b ∧ b ≠ c ∧ a ≠ c → (R a b ∧ R b c → R c a)

theorem max_elements (S : Set α) : set.finite S → ∃ n : ℕ, n ≤ 3 ∧ Fintype.card (Fintype.ofFinite S) = n :=
sorry

end max_elements_l440_440333


namespace remainder_problem_l440_440277

theorem remainder_problem (x y z : ℤ) 
  (hx : x % 15 = 11) (hy : y % 15 = 13) (hz : z % 15 = 14) : 
  (y + z - x) % 15 = 1 := 
by 
  sorry

end remainder_problem_l440_440277


namespace floor_div_add_floor_div_succ_eq_l440_440924

theorem floor_div_add_floor_div_succ_eq (n : ℤ) : 
  (⌊(n : ℝ)/2⌋ + ⌊(n + 1 : ℝ)/2⌋ : ℤ) = n := 
sorry

end floor_div_add_floor_div_succ_eq_l440_440924


namespace express_f_area_enclosed_value_of_t_l440_440920

-- Define the quadratic function f(x)
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- 1. Expression for f(x)
theorem express_f :
  ∃ a b c : ℝ, (∀ x : ℝ, f(x) = a*x^2 + b*x + c) ∧
                (f'(x) = 2*x + 2) ∧
                (∃ x : ℝ, f(x) = 0) ∧
                a = 1 ∧
                b = 2 ∧
                c = 1 :=
by
  -- proof goes here
  sorry

-- 2. Area enclosed by the graph of y = f(x) and the coordinate axes
theorem area_enclosed :
  ∫ x in (-1:ℝ)..0, f(x) = 1/3 :=
by
  -- proof goes here
  sorry

-- 3. Value of t dividing the area into two equal parts
theorem value_of_t (t : ℝ) :
  0 < t ∧ t < 1 ∧
  ∫ x in (-1:ℝ)..-t, f(x) = ∫ x in -t..0, f(x) ∧
  t = 1 - 1/(32) :=
by
  -- proof goes here
  sorry

end express_f_area_enclosed_value_of_t_l440_440920


namespace find_n_l440_440287

theorem find_n (n : ℕ) (h_odd : n % 2 = 1) (h_sum : ∑ k in range (n // 2), (2 * k) = 95 * 96) : n = 191 :=
sorry

end find_n_l440_440287


namespace part_I_part_II_l440_440070

def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem part_I (x : ℝ) : (f x > 4) ↔ (x < -1.5 ∨ x > 2.5) :=
by
  sorry

theorem part_II (x : ℝ) : ∀ x : ℝ, f x ≥ 3 :=
by
  sorry

end part_I_part_II_l440_440070


namespace sum_algebra_values_l440_440973

def alphabet_value (n : ℕ) : ℤ :=
  match n % 8 with
  | 1 => 3
  | 2 => 1
  | 3 => 0
  | 4 => -1
  | 5 => -3
  | 6 => -1
  | 7 => 0
  | _ => 1

theorem sum_algebra_values : 
  alphabet_value 1 + 
  alphabet_value 12 + 
  alphabet_value 7 +
  alphabet_value 5 +
  alphabet_value 2 +
  alphabet_value 18 +
  alphabet_value 1 
  = 5 := by
  sorry

end sum_algebra_values_l440_440973


namespace Clea_ride_time_l440_440133

theorem Clea_ride_time
  (c s d t : ℝ)
  (h1 : d = 80 * c)
  (h2 : d = 30 * (c + s))
  (h3 : s = 5 / 3 * c)
  (h4 : t = d / s) :
  t = 48 := by sorry

end Clea_ride_time_l440_440133


namespace is_quadratic_C_l440_440278

-- Define the functions given in the problem
def fA (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def fB (x : ℝ) : ℝ := 1/x^2 + x
def fC (x : ℝ) : ℝ := x * (2 * x - 1)
def fD (x : ℝ) : ℝ := (x + 4)^2 - x^2

-- Prove that function C is quadratic
theorem is_quadratic_C : ∀ x : ℝ, ∃ a b c : ℝ, fC x = a * x^2 + b * x + c :=
by
  intro x
  use [2, -1, 0]
  simp [fC]
  sorry

end is_quadratic_C_l440_440278


namespace dice_sum_24_l440_440760

noncomputable def probability_of_sum_24 : ℚ :=
  let die_probability := (1 : ℚ) / 6
  in die_probability ^ 4

theorem dice_sum_24 :
  ∑ x in {x | x ∈ {1, 2, 3, 4, 5, 6} ∧ x = 6} = 24 → probability_of_sum_24 = 1 / 1296 :=
sorry

end dice_sum_24_l440_440760


namespace triangle_angles_l440_440517

-- Definitions for altitude and angles in a triangle
variables {A B C : Type} [linear_ordered_field A]

-- Lean statement for the problem
theorem triangle_angles (a b c : A)
    (hA : a > 0)
    (hB : b > 0)
    (hC : c > 0)
    (h_alt_A : a ≤ b)
    (h_alt_B : b ≤ a) :
  (a = 90 ∧ b = 45 ∧ c = 45) :=
begin
  sorry
end

end triangle_angles_l440_440517


namespace sum_a_i_is_integer_l440_440015

theorem sum_a_i_is_integer (k : ℕ) (a b : Fin k → ℝ) 
  (X : ℕ → ℤ)
  (H_Xn: ∀ n : ℕ, X n = ∑ i in Finset.range k, ⌊a i * n + b i⌋)
  (H_arith: ∃ d: ℤ, ∀ n : ℕ, X (n + 1) = X n + d) :
  ∃ m : ℤ, (∑ i in Finset.range k, a i) = m :=
by
  sorry

end sum_a_i_is_integer_l440_440015


namespace value_of_f_at_2_l440_440542

def f (x : ℝ) := x^2 + 2 * x - 3

theorem value_of_f_at_2 : f 2 = 5 :=
by
  sorry

end value_of_f_at_2_l440_440542


namespace inequality_solution_interval_l440_440818

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
∀ x, f(x) = f(2 * a - x)

def function_monotonic_in (f : ℝ → ℝ) (φ : ℝ → ℝ) : Prop :=
(∀ x < 1, ∃ (φ := λ x, (x - 1) ^ 2 * f x), φ'(x) > 0)

theorem inequality_solution_interval 
  (f : ℝ → ℝ)
  (h_dom : ∀ x : ℝ, f x ∈ ℝ)
  (h_symmetric : symmetric_about f 1)
  (h_deriv : ∀ x, ∃ f' x)
  (h_cond : ∀ x < 1, 2 * f x + (x - 1) * (deriv f) x < 0) :
  ∀ x, (x + 1) ^ 2 * f (x + 2) > f 2 ↔ x ∈ Ioo (-2) 0 :=
by 
  sorry

end inequality_solution_interval_l440_440818


namespace dice_probability_exactly_four_ones_l440_440261

noncomputable def dice_probability : ℚ := 
  (Nat.choose 12 4) * (1/6)^4 * (5/6)^8

theorem dice_probability_exactly_four_ones : (dice_probability : ℚ) ≈ 0.089 :=
  by sorry -- Skip the proof. 

#eval (dice_probability : ℚ)

end dice_probability_exactly_four_ones_l440_440261


namespace slope_of_line_l440_440833

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line l intersecting the parabola C at points A and B
def line (k x : ℝ) : ℝ := k * (x - 1)

-- Condition based on the intersection and the given relationship 2 * (BF) = FA
def intersection_condition (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ x1 x2 y1 y2,
    A = (x1, y1) ∧ B = (x2, y2) ∧
    parabola x1 y1 ∧ parabola x2 y2 ∧
    (y1 = line k x1) ∧ (y2 = line k x2) ∧
    2 * (dist (x2, y2) focus) = dist focus (x1, y1)

-- The main theorem to be proven
theorem slope_of_line (k : ℝ) (A B : ℝ × ℝ) :
  intersection_condition k A B → k = 2 * Real.sqrt 2 :=
sorry

end slope_of_line_l440_440833


namespace exponential_identity_l440_440776

theorem exponential_identity
  (a b : ℝ)
  (h1 : 2^a = 5)
  (h2 : log 8 3 = b) :
  4^(a - 3 * b) = 25 / 9 :=
sorry

end exponential_identity_l440_440776


namespace sin_half_C_proof_l440_440869

variables {a b c : ℝ}
variables {A B C : ℝ} -- angles A, B, C in triangle

-- Conditions
def condition1 : Prop := a + sqrt 2 * c = 2 * b
def condition2 : Prop := Real.sin B = sqrt 2 * Real.sin C

-- Question
def sin_half_C := Real.sin (C / 2)

-- Theorem statement
theorem sin_half_C_proof (h1 : condition1) (h2 : condition2) : 
  sin_half_C = sqrt 2 / 4 :=
sorry

end sin_half_C_proof_l440_440869


namespace preimage_of_f_is_correct_l440_440073

theorem preimage_of_f_is_correct :
  ∃ (x y : ℝ), (x + y = 3) ∧ (x - y = 1) ∧ (x = 2) ∧ (y = 1) := by
sory

end preimage_of_f_is_correct_l440_440073


namespace solve_linear_function_l440_440785

theorem solve_linear_function :
  (∀ (x y : ℤ), (x = -3 ∧ y = -4) ∨ (x = -2 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ 
                      (x = 0 ∧ y = 2) ∨ (x = 1 ∧ y = 4) ∨ (x = 2 ∧ y = 6) →
   ∃ (a b : ℤ), y = a * x + b ∧ a * 1 + b = 4) :=
sorry

end solve_linear_function_l440_440785


namespace range_of_a_range_of_m_l440_440831

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f x < |1 - 2 * a|) ↔ a ∈ (Set.Iic (-3/2) ∪ Set.Ici (5/2)) := by sorry

theorem range_of_m (m : ℝ) : 
  (∃ t : ℝ, t^2 - 2 * Real.sqrt 6 * t + f m = 0) ↔ m ∈ (Set.Icc (-1) 2) := by sorry

end range_of_a_range_of_m_l440_440831


namespace proposition_1_proposition_2_proposition_3_proposition_4_all_propositions_correct_l440_440461

def f (x : ℝ) : ℝ :=
  if x ∈ ℚ then 1 else 0

theorem proposition_1 : ∀ x : ℝ, f (f x) = 1 := sorry

theorem proposition_2 : ∀ x : ℝ, f x = f (-x) := sorry

theorem proposition_3 (T : ℚ) (hT : T ≠ 0) : ∀ x : ℝ, f (x + T) = f x := sorry

theorem proposition_4 : 
  ∃ A B C : ℝ × ℝ, 
    A = (sqrt 3 / 3, 0) ∧ 
    B = (0, 1) ∧ 
    C = (-sqrt 3 / 3, 0) ∧ 
    (f A.1 = A.2 ∧ f B.1 = B.2 ∧ f C.1 = C.2) ∧ 
    (dist A B = dist B C ∧ dist B C = dist C A) := sorry

theorem all_propositions_correct : 
  (∀ x : ℝ, f (f x) = 1) ∧ 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ T : ℚ, T ≠ 0 → ∀ x : ℝ, f (x + T) = f x) ∧ 
  ∃ A B C : ℝ × ℝ, 
    A = (sqrt 3 / 3, 0) ∧ 
    B = (0, 1) ∧ 
    C = (-sqrt 3 / 3, 0) ∧ 
    (f A.1 = A.2 ∧ f B.1 = B.2 ∧ f C.1 = C.2) ∧ 
    (dist A B = dist B C ∧ dist B C = dist C A) := 
  ⟨proposition_1, proposition_2, proposition_3, proposition_4⟩

end proposition_1_proposition_2_proposition_3_proposition_4_all_propositions_correct_l440_440461


namespace bamboo_break_height_l440_440506

-- Conditions provided in the problem
def original_height : ℝ := 20  -- 20 chi
def distance_tip_to_root : ℝ := 6  -- 6 chi

-- Function to check if the height of the break satisfies the equation
def equationHolds (x : ℝ) : Prop :=
  (original_height - x) ^ 2 - x ^ 2 = distance_tip_to_root ^ 2

-- Main statement to prove the height of the break is 9.1 chi
theorem bamboo_break_height : equationHolds 9.1 :=
by
  sorry

end bamboo_break_height_l440_440506


namespace integer_pairs_sum_product_l440_440402

theorem integer_pairs_sum_product (x y : ℤ) (h : x + y = x * y) : (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) :=
by
  sorry

end integer_pairs_sum_product_l440_440402


namespace minimum_value_A_l440_440953

theorem minimum_value_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + c^2 * a = 3) : 
  (a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3) ≥ 6 :=
by
  sorry

end minimum_value_A_l440_440953


namespace p_is_sufficient_but_not_necessary_for_q_l440_440052

-- Definitions and conditions
def p (x : ℝ) : Prop := (x = 1)
def q (x : ℝ) : Prop := (x^2 - 3 * x + 2 = 0)

-- Theorem statement
theorem p_is_sufficient_but_not_necessary_for_q : ∀ x : ℝ, (p x → q x) ∧ (¬ (q x → p x)) :=
by
  sorry

end p_is_sufficient_but_not_necessary_for_q_l440_440052


namespace find_constants_l440_440536

noncomputable def t' : ℝ := 1 / 5
noncomputable def u' : ℝ := 4 / 5

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem find_constants (C D Q : V) (ratio_condition : ∃ k : ℝ, k = 4 ∧ Q = (4 : ℝ) / (5 : ℝ) • D + (1 : ℝ) / (5 : ℝ) • C) :
  Q = t' • C + u' • D :=
by
  rcases ratio_condition with ⟨k, hk1, hk2⟩
  have ht' : t' = 1 / 5 := rfl
  have hu' : u' = 4 / 5 := rfl
  rw [ht', hu']
  exact hk2

end find_constants_l440_440536


namespace trajectory_and_slope_l440_440817

theorem trajectory_and_slope :
  (∀ x y : ℝ, abs (x - 4) = 2 * sqrt ((x - 1)^2 + y^2)) → 
  (∃ k : ℝ, (∀ (x y : ℝ), (x, y) ∈ ellipse_eqn x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧ 
             (line_through_pt_satisfies_slope (0, 3) A B k → (k = 3/2 ∨ k = -3/2))) :=
begin
  intros h,
  simp only [ellipse_eqn, line_through_pt_satisfies_slope],
  sorry
end

end trajectory_and_slope_l440_440817


namespace problem_solution_l440_440421

variables {a b c : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem problem_solution (h1 : f 0 = f 4) (h2 : f 0 > f 1) : a > 0 ∧ 4 * a + b = 0 :=
by
  unfold f at h1 h2
  sorry

end problem_solution_l440_440421


namespace largest_of_nine_consecutive_integers_l440_440223

theorem largest_of_nine_consecutive_integers (sum_eq_99: ∃ (n : ℕ), 99 = (n - 4) + (n - 3) + (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) : 
  ∃ n : ℕ, n = 15 :=
by
  sorry

end largest_of_nine_consecutive_integers_l440_440223


namespace complement_union_l440_440472

variable (U : Set ℤ) (A : Set ℤ) (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := 
by 
  -- Proof is omitted
  sorry

end complement_union_l440_440472


namespace inequality_smallest_val_l440_440416

open Real

def cot (x : ℝ) := cos x / sin x
def tan (x : ℝ) := sin x / cos x

theorem inequality_smallest_val (a : ℝ) (h : a = -2.52) :
  (∀ x ∈ set.Ioo (-3 * π / 2) (-π), 
  (∛(cot x ^ 2) - ∛(tan x ^ 2)) / (∛(sin x ^ 2) - ∛(cos x ^ 2)) < a) :=
begin
  sorry
end

end inequality_smallest_val_l440_440416


namespace kim_shoes_total_l440_440900

theorem kim_shoes_total :
  ∃ (N : ℕ), N = 16 ∧
  (8 * 2 = 16) ∧
  ((∃ (p : ℝ), p = 0.06666666666666667 ∧
  (∃ M, M = 16 * (N - 1) ∧ p = ((M : ℝ)⁻¹) * (N⁻¹ * 16 / M)))) ∧
  (N ≤ 16) ∧ (N > 0) ∧ (N % 2 = 0) :=
sorry

end kim_shoes_total_l440_440900


namespace resulting_graph_is_correct_l440_440615

noncomputable def initial_function (x : ℝ) : ℝ := sin (x - π / 6)

noncomputable def translated_function (x : ℝ) : ℝ := sin (x - 5 * π / 12)

noncomputable def expanded_function (x : ℝ) : ℝ := sin (x / 2 - 5 * π / 12)

theorem resulting_graph_is_correct :
  (∀ x : ℝ, translated_function x = sin (x - 5 * π / 12)) →
  (∀ x : ℝ, expanded_function x = sin (x / 2 - 5 * π / 12)) → 
  (∀ x : ℝ, expanded_function x = sin (x / 2 - 5 * π / 12)) :=
by
  intros h1 h2 x
  exact h2 x

end resulting_graph_is_correct_l440_440615


namespace non_attacking_rooks_l440_440479

-- Define the number of rows and columns on the board
def n : ℕ := 4

-- Define the set of blocked cells, using (row, column) notation
def blocked_cells : set (ℕ × ℕ) := { ... }
  -- specific blocked cells should be enumerated here

-- Define the condition for non-attacking rook placements
def non_attacking (placement : set (ℕ × ℕ)) : Prop :=
  ∀ (p₁ p₂ : ℕ × ℕ), p₁ ≠ p₂ → p₁ ∈ placement → p₂ ∈ placement →
  p₁.1 ≠ p₂.1 ∧ p₁.2 ≠ p₂.2

-- Define the main theorem: number of valid ways to place 3 rooks
theorem non_attacking_rooks : 
  (∃ placements : set (set (ℕ × ℕ)), 
    ∀ placement ∈ placements, non_attacking placement ∧ 
    placement.card = 3 ∧ 
    ∀ cell ∈ placement, cell ∉ blocked_cells ∧ 
    placements.card = 13) := sorry

end non_attacking_rooks_l440_440479


namespace power_of_i_l440_440711

-- Define imaginary unit i such that i^2 = -1
def imaginary_unit : ℂ := complex.I

theorem power_of_i (i : ℂ) (h : i^2 = -1) : i^2013 = i :=
by
  sorry

end power_of_i_l440_440711


namespace transformed_cos_to_neg_sin_l440_440997

theorem transformed_cos_to_neg_sin :
  ∀ x, cos ((2 * x) + (π / 4)) = -sin (2 * x) :=
by
  intro x
  sorry

end transformed_cos_to_neg_sin_l440_440997


namespace coin_arrangement_possible_l440_440949

def cell := {i // 0 ≤ i ∧ i < 4} × {j // 0 ≤ j ∧ j < 4}

def is_gold (bd : cell → ℕ) (c : cell) : Prop := bd c = 1
def is_silver (bd : cell → ℕ) (c : cell) : Prop := bd c > 1

def count_gold (bd : cell → ℕ) : ℕ := ∑ i j, if is_gold bd ⟨i, j⟩ then 1 else 0
def count_silver (bd : cell → ℕ) : ℕ := ∑ i j, if is_silver bd ⟨i, j⟩ then 1 else 0

def three_by_three_subsquares : list (cell → Prop) :=
  [λ ⟨i, j⟩, (0 ≤ i + j ∧ i + j < 3), λ ⟨i, j⟩, (0 ≤ i + j ∧ i + j < 3)]

def valid (bd : cell → ℕ) : Prop :=
  ∀ sq ∈ three_by_three_subsquares, 
    count_silver (λ c, if sq c then bd c else 0) > count_gold (λ c, if sq c then bd c else 0)

def problem_valid : Prop :=
  ∃ bd : cell → ℕ, 
    valid bd ∧ count_gold bd > count_silver bd

theorem coin_arrangement_possible : problem_valid :=
sorry

end coin_arrangement_possible_l440_440949


namespace solve_for_R_l440_440577

theorem solve_for_R (R : ℝ) (h : sqrt (R^3) = 18 * real.root 27 (9 : ℝ)) : R = 6 * real.root 3 (3 : ℝ) :=
sorry

end solve_for_R_l440_440577


namespace transformed_roots_l440_440054

theorem transformed_roots 
  (a b c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a * (-1)^2 + b * (-1) + c = 0)
  (h₃ : a * 2^2 + b * 2 + c = 0) :
  (a * 0^2 + b * 0 + c = 0) ∧ (a * 3^2 + b * 3 + c = 0) :=
by 
  sorry

end transformed_roots_l440_440054


namespace maximum_value_l440_440453

section

variables {α : Type*} [inner_product_space ℝ α]
variables (a b : α) (x y : ℝ)

-- Given conditions
def conditions : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ real.angle a b = real.pi / 3 ∧ ∥x • a + y • b∥ = real.sqrt 3

-- Statement for maximum value of |x a - y b|
theorem maximum_value (h : conditions a b x y) :
  ∥x • a - y • b∥ ≤ 3 :=
sorry

end

end maximum_value_l440_440453


namespace difference_mean_median_l440_440504

theorem difference_mean_median :
  let percentage_scored_60 : ℚ := 0.20
  let percentage_scored_70 : ℚ := 0.30
  let percentage_scored_85 : ℚ := 0.25
  let percentage_scored_95 : ℚ := 1 - (percentage_scored_60 + percentage_scored_70 + percentage_scored_85)
  let score_60 : ℚ := 60
  let score_70 : ℚ := 70
  let score_85 : ℚ := 85
  let score_95 : ℚ := 95
  let mean : ℚ := percentage_scored_60 * score_60 + percentage_scored_70 * score_70 + percentage_scored_85 * score_85 + percentage_scored_95 * score_95
  let median : ℚ := 85
  (median - mean) = 7 := 
by 
  sorry

end difference_mean_median_l440_440504


namespace sequence_nature_l440_440552

-- Define the conditions for the problem
variable (a : ℝ) (h : 0 < a ∧ a < 1)

-- Define the sequence x_n
def x : ℕ → ℝ
| 0       := a
| (n + 1) := a ^ x n

-- State the theorem to be proved
theorem sequence_nature (h : 0 < a ∧ a < 1) :
  (∀ n : ℕ, (n.bodd = true → x (n + 2) < x n) ∧ (n.bodd = false → x (n + 2) > x n)) := sorry

end sequence_nature_l440_440552


namespace probability_exactly_four_ones_is_090_l440_440238
open Float (approxEq)

def dice_probability_exactly_four_ones : Float :=
  let n := 12
  let k := 4
  let p_one := (1 / 6 : Float)
  let p_not_one := (5 / 6 : Float)
  let combination := ((n.factorial) / (k.factorial * (n - k).factorial) : Float)
  let probability := combination * (p_one ^ k) * (p_not_one ^ (n - k))
  probability

theorem probability_exactly_four_ones_is_090 : dice_probability_exactly_four_ones ≈ 0.090 :=
  sorry

end probability_exactly_four_ones_is_090_l440_440238


namespace largest_adjacent_to_1_number_of_good_cells_l440_440167

def table_width := 51
def table_height := 3
def total_cells := 153

-- Conditions
def condition_1_present (n : ℕ) : Prop := n ∈ Finset.range (total_cells + 1)
def condition_2_bottom_left : Prop := (1 = 1)
def condition_3_adjacent (a b : ℕ) : Prop := 
  (a = b + 1) ∨ 
  (a + 1 = b) ∧ 
  (condition_1_present a) ∧ 
  (condition_1_present b)

-- Part (a): Largest number adjacent to cell containing 1 is 152.
theorem largest_adjacent_to_1 : ∃ b, b = 152 ∧ condition_3_adjacent 1 b :=
by sorry

-- Part (b): Number of good cells that can contain the number 153 is 76.
theorem number_of_good_cells : ∃ count, count = 76 ∧ 
  ∀ (i : ℕ) (j: ℕ), (i, j) ∈ (Finset.range table_height).product (Finset.range table_width) →
  condition_1_present 153 ∧
  (i = table_height - 1 ∨ j = 0 ∨ j = table_width - 1 ∨ j ∈ (Finset.range (table_width - 2)).erase 1) →
  (condition_3_adjacent (i*table_width + j) 153) :=
by sorry

end largest_adjacent_to_1_number_of_good_cells_l440_440167


namespace sum_of_solutions_l440_440903

def f (x : ℝ) : ℝ :=
  if x < -3 then 3*x + 9
  else if x < 1 then -x^2 + 2*x + 3
  else 4*x - 5

theorem sum_of_solutions : 
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = -1 ∧ x₂ = 5/4) → -1 + (5/4) = 0.25 :=
by
  intro h
  sorry

end sum_of_solutions_l440_440903


namespace journey_duration_correct_l440_440575

noncomputable def journey_duration : ℕ :=
  let start_minutes := 1 * 60 + 5 + 27 / 60
  let end_minutes := 6 * 60
  in end_minutes - start_minutes

theorem journey_duration_correct :
  journey_duration = 4 * 60 + 54 :=
by sorry

end journey_duration_correct_l440_440575


namespace proof_problem_l440_440433

variable {x y : ℝ}

def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y = 1

theorem proof_problem (h : conditions x y) :
  x + y - 4 * x * y ≥ 0 ∧ (1 / x) + 4 / (1 + y) ≥ 9 / 2 :=
by
  sorry

end proof_problem_l440_440433


namespace trig_identity_l440_440028

noncomputable def sin_eq := sorry -- Define necessary trigonometric properties

theorem trig_identity
  (x : ℝ)
  (h : sin (x + π / 6) = 1 / 3) :
  sin (x - 5 * π / 6) + sin (π / 3 - x) ^ 2 = 5 / 9 :=
by
  sorry

end trig_identity_l440_440028


namespace ab_finish_job_in_15_days_l440_440642

theorem ab_finish_job_in_15_days (A B C : ℝ) (h1 : A + B + C = 1/12) (h2 : C = 1/60) : 1 / (A + B) = 15 := 
by
  sorry

end ab_finish_job_in_15_days_l440_440642


namespace intersection_of_A_and_B_l440_440159

def setA : Set ℕ := {1, 2, 3}
def setB : Set ℕ := {2, 4, 6}

theorem intersection_of_A_and_B : setA ∩ setB = {2} :=
by
  sorry

end intersection_of_A_and_B_l440_440159


namespace minimum_value_inequality_l440_440485

theorem minimum_value_inequality {a b : ℝ} (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * Real.log b / Real.log a + 2 * Real.log a / Real.log b = 7) :
  a^2 + 3 / (b - 1) ≥ 2 * Real.sqrt 3 + 1 :=
sorry

end minimum_value_inequality_l440_440485


namespace geometric_sequence_properties_l440_440888

noncomputable def geometric_sequence_sum (r a1 : ℝ) : Prop :=
  a1 * (r^3 + r^4) = 27 ∨ a1 * (r^3 + r^4) = -27

theorem geometric_sequence_properties (a1 r : ℝ) (h1 : a1 + a1 * r = 1) (h2 : a1 * r^2 + a1 * r^3 = 9) :
  geometric_sequence_sum r a1 :=
sorry

end geometric_sequence_properties_l440_440888


namespace find_f1_solve_inequality_l440_440543

variable {f : ℝ → ℝ}
variable (f_decreasing : ∀ x y, 0 < x → 0 < y → x < y → f(x) > f(y))
variable (f_add : ∀ x y, 0 < x → 0 < y → f(x + y) = f(x) + f(y) - 1)
variable (f_at_4 : f(4) = 5)

theorem find_f1 : f(1) = 2 :=
by
  sorry

theorem solve_inequality (m : ℝ) (h : f(m - 2) ≥ 2) : 2 < m ∧ m ≤ 3 :=
by
  sorry

end find_f1_solve_inequality_l440_440543


namespace old_clock_slow_by_12_minutes_l440_440211

theorem old_clock_slow_by_12_minutes (overlap_interval: ℕ) (standard_day_minutes: ℕ)
  (h1: overlap_interval = 66) (h2: standard_day_minutes = 24 * 60):
  standard_day_minutes - 24 * 60 / 66 * 66 = 12 :=
by
  sorry

end old_clock_slow_by_12_minutes_l440_440211


namespace average_age_union_l440_440650

open Real

variables {a b c d A B C D : ℝ}

theorem average_age_union (h1 : A / a = 40)
                         (h2 : B / b = 30)
                         (h3 : C / c = 45)
                         (h4 : D / d = 35)
                         (h5 : (A + B) / (a + b) = 37)
                         (h6 : (A + C) / (a + c) = 42)
                         (h7 : (A + D) / (a + d) = 39)
                         (h8 : (B + C) / (b + c) = 40)
                         (h9 : (B + D) / (b + d) = 37)
                         (h10 : (C + D) / (c + d) = 43) : 
  (A + B + C + D) / (a + b + c + d) = 44.5 := 
sorry

end average_age_union_l440_440650


namespace gcd_exponent_min_speed_for_meeting_game_probability_difference_l440_440652

-- Problem p4
theorem gcd_exponent (a b : ℕ) (h1 : a = 6) (h2 : b = 9) (h3 : gcd a b = 3) : gcd (2^a - 1) (2^b - 1) = 7 := by
  sorry

-- Problem p5
theorem min_speed_for_meeting (v_S s : ℚ) (h : v_S = 1/2) : ∀ (s : ℚ), (s - v_S) ≥ 1 → s = 3/2 := by
  sorry

-- Problem p6
theorem game_probability_difference (N : ℕ) (p : ℚ) (h1 : N = 1) (h2 : p = 5/16) : N + p = 21/16 := by
  sorry

end gcd_exponent_min_speed_for_meeting_game_probability_difference_l440_440652


namespace find_z_l440_440599

open Real

theorem find_z (z : ℝ) 
  (proj_condition : (inner ⟨1, 4, z⟩ ⟨1, -3, 2⟩ / inner ⟨1, -3, 2⟩ ⟨1, -3, 2⟩) • ⟨1, -3, 2⟩ = (5/14) • ⟨1, -3, 2⟩) : 
  z = 8 := 
by 
  sorry

end find_z_l440_440599


namespace dice_sum_24_l440_440762

noncomputable def probability_of_sum_24 : ℚ :=
  let die_probability := (1 : ℚ) / 6
  in die_probability ^ 4

theorem dice_sum_24 :
  ∑ x in {x | x ∈ {1, 2, 3, 4, 5, 6} ∧ x = 6} = 24 → probability_of_sum_24 = 1 / 1296 :=
sorry

end dice_sum_24_l440_440762


namespace value_of_fff1_l440_440860

def f(x : ℝ) : ℝ := 4 * x^3 + 2 * x^2 - 5 * x + 1

theorem value_of_fff1 : f(f(1)) = 31 :=
by sorry

end value_of_fff1_l440_440860


namespace prob_sum_24_four_dice_l440_440758

section
open ProbabilityTheory

/-- Define the event E24 as the event where the sum of numbers on the top faces of four six-sided dice is 24 -/
def E24 : Event (StdGen) :=
eventOfFun {ω | ∑ i in range 4, (ω.gen_uniform int (6-1)) + 1 = 24}

/-- Probability that the sum of the numbers on top faces of four six-sided dice is 24 is 1/1296. -/
theorem prob_sum_24_four_dice : ⋆{ P(E24) = 1/1296 } := sorry

end

end prob_sum_24_four_dice_l440_440758


namespace average_one_half_one_fourth_one_eighth_l440_440724

theorem average_one_half_one_fourth_one_eighth : 
  ((1 / 2.0 + 1 / 4.0 + 1 / 8.0) / 3.0) = 7 / 24 := 
by sorry

end average_one_half_one_fourth_one_eighth_l440_440724


namespace converse_proposition_l440_440197

-- Define the condition: The equation x^2 + x - m = 0 has real roots
def has_real_roots (a b c : ℝ) : Prop :=
  let Δ := b * b - 4 * a * c
  Δ ≥ 0

theorem converse_proposition (m : ℝ) :
  has_real_roots 1 1 (-m) → m > 0 :=
by
  sorry

end converse_proposition_l440_440197


namespace multiplicity_greater_than_one_iff_p_eq_zero_l440_440834

variable {R : Type*} [CommRing R]
variable (P : R[X]) (a : R)

theorem multiplicity_greater_than_one_iff_p_eq_zero :
  (∀ Q : R[X], ∃ p : R, P = (X - C a) ^ 2 * Q + p * (X - C a) ∧ p = 0) ↔ (P.derivative.eval a = 0) :=
  sorry

end multiplicity_greater_than_one_iff_p_eq_zero_l440_440834


namespace farmer_initial_plan_days_l440_440314

def initialDaysPlan
    (daily_hectares : ℕ)
    (increased_productivity : ℕ)
    (hectares_ploughed_first_two_days : ℕ)
    (hectares_remaining : ℕ)
    (days_ahead_schedule : ℕ)
    (total_hectares : ℕ)
    (days_actual : ℕ) : ℕ :=
  days_actual + days_ahead_schedule

theorem farmer_initial_plan_days : 
  ∀ (x days_ahead_schedule : ℕ) 
    (daily_hectares hectares_ploughed_first_two_days increased_productivity hectares_remaining total_hectares days_actual : ℕ),
  daily_hectares = 120 →
  increased_productivity = daily_hectares + daily_hectares / 4 →
  hectares_ploughed_first_two_days = 2 * daily_hectares →
  total_hectares = 1440 →
  days_ahead_schedule = 2 →
  days_actual = 10 →
  hectares_remaining = total_hectares - hectares_ploughed_first_two_days →
  hectares_remaining = increased_productivity * (days_actual - 2) →
  x = 12 :=
by
  intros x days_ahead_schedule daily_hectares hectares_ploughed_first_two_days increased_productivity hectares_remaining total_hectares days_actual
  intros h_daily_hectares h_increased_productivity h_hectares_ploughed_first_two_days h_total_hectares h_days_ahead_schedule h_days_actual h_hectares_remaining h_hectares_ploughed
  sorry

end farmer_initial_plan_days_l440_440314


namespace sum_of_digits_l440_440034

theorem sum_of_digits (N a b c : ℕ) (h1 : Nat.digits_count 10 N = 2015)
  (h2 : 9 ∣ N) (h3 : a = (Nat.digits 10 N).sum) (h4 : b = (Nat.digits 10 a).sum)
  (h5 : c = (Nat.digits 10 b).sum) : c = 9 :=
by sorry

end sum_of_digits_l440_440034


namespace distance_between_foci_l440_440734

theorem distance_between_foci :
  let x := Hyperbola
  (9 : ℤ) * x ^ 2 - (18 : ℤ) * x - (16 : ℤ) * y ^ 2 - (32 : ℤ) * y = -(144 : ℤ)
  in
  let a_sq := 16
  let b_sq := 9
  let c_sq := a_sq + b_sq
  let c := Int.sqrt c_sq
  2 * c = 10 :=
by
  let x := Hyperbola
    sorry

end distance_between_foci_l440_440734


namespace prime_gt_3_divides_exp_l440_440918

theorem prime_gt_3_divides_exp (p : ℕ) (hprime : Nat.Prime p) (hgt3 : p > 3) :
  42 * p ∣ 3^p - 2^p - 1 :=
sorry

end prime_gt_3_divides_exp_l440_440918


namespace distance_is_correct_l440_440866

noncomputable def radius (d : ℝ) := d / 2.0

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def distance_covered (d : ℝ) (revolutions : ℝ) : ℝ :=
  circumference (radius d) * revolutions

-- Given conditions
def diameter : ℝ := 12.0
def number_of_revolutions : ℝ := 14.012738853503185
def expected_distance : ℝ := 528.002

-- Proof problem
theorem distance_is_correct : 
  distance_covered diameter number_of_revolutions ≈ expected_distance :=
by
  -- Proof details would go here
  sorry

end distance_is_correct_l440_440866


namespace translate_parabola_l440_440236

theorem translate_parabola (x : ℝ):
  (let f := λ x : ℝ, 2 * x^2 in
   let f1 := λ x : ℝ, f (x + 3) in
   let f2 := λ x : ℝ, f1 x + 4 in
   f2 x = 2 * (x + 3)^2 + 4) := 
sorry

end translate_parabola_l440_440236


namespace quadratic_j_value_l440_440214

theorem quadratic_j_value (a b c : ℝ) (h : a * (0 : ℝ)^2 + b * (0 : ℝ) + c = 5 * ((0 : ℝ) - 3)^2 + 15) :
  ∃ m j n, 4 * a * (0 : ℝ)^2 + 4 * b * (0 : ℝ) + 4 * c = m * ((0 : ℝ) - j)^2 + n ∧ j = 3 :=
by
  sorry

end quadratic_j_value_l440_440214


namespace find_x_l440_440475

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vectors_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, (u.1 * k = v.1) ∧ (u.2 * k = v.2)

theorem find_x :
  let a := (1, -2)
  let b := (3, -1)
  let c := (x, 4)
  vectors_parallel (vector_add a c) (vector_add b c) → x = 3 :=
by intros; sorry

end find_x_l440_440475


namespace lewis_total_earnings_l440_440929

def Weekly_earnings : ℕ := 92
def Number_of_weeks : ℕ := 5

theorem lewis_total_earnings : Weekly_earnings * Number_of_weeks = 460 := by
  sorry

end lewis_total_earnings_l440_440929


namespace dart_distribution_count_l440_440379

-- Definitions of the conditions
def num_boards : Nat := 5
def num_darts : Nat := 6
def valid_distribution (dist : List Nat) : Prop :=
  dist.length = num_boards ∧
  dist.sum = num_darts ∧
  (∃ x, ∀ y, y ∈ dist → y ≤ x ∧ x - y ∈ {0, 1}) ∧
  (∃ a, ∀ b, b ∈ dist → b ≤ a ∧ a ≠ 0)

-- Main theorem to be proved
theorem dart_distribution_count :
  (List.filter valid_distribution (List.permutations [6, 0, 0, 0, 0] ++ 
    List.permutations [5, 1, 0, 0, 0] ++ 
    List.permutations [4, 2, 0, 0, 0] ++ 
    List.permutations [4, 1, 1, 0, 0] ++ 
    List.permutations [3, 3, 0, 0, 0])).length = 5 := 
sorry

end dart_distribution_count_l440_440379


namespace remainder_x14_minus_1_div_x_plus_1_l440_440273

-- Define the polynomial f(x) = x^14 - 1
def f (x : ℝ) := x^14 - 1

-- Statement to prove that the remainder when f(x) is divided by x + 1 is 0
theorem remainder_x14_minus_1_div_x_plus_1 : f (-1) = 0 :=
by
  -- This is where the proof would go, but for now, we will just use sorry
  sorry

end remainder_x14_minus_1_div_x_plus_1_l440_440273


namespace cylinder_ratio_l440_440782

theorem cylinder_ratio
  (V : ℝ) (R H : ℝ)
  (hV : V = 1000)
  (hVolume : π * R^2 * H = V) :
  H / R = 1 :=
by
  sorry

end cylinder_ratio_l440_440782


namespace dice_sum_24_probability_l440_440767

noncomputable def probability_sum_24 : ℚ :=
  let prob_single_six := (1 : ℚ) / 6 in
  prob_single_six ^ 4

theorem dice_sum_24_probability :
  probability_sum_24 = 1 / 1296 :=
by
  sorry

end dice_sum_24_probability_l440_440767


namespace worker_assignment_l440_440397

theorem worker_assignment :
  ∃ (x y : ℕ), x + y = 85 ∧
  (16 * x) / 2 = (10 * y) / 3 ∧
  x = 25 ∧ y = 60 :=
by
  sorry

end worker_assignment_l440_440397


namespace angle_between_AB_and_B1C_is_90_l440_440299

noncomputable def angle_between_lines_in_cube (A B C D A1 B1 C1 D1 : Point)
  (length : ℝ := 1) : ℝ :=
  let AB := line_through A B
  let B1C := line_through B1 C
  -- Assume necessary points and cube edge-length relationships are defined
  -- This function should calculate the angle between the lines AB and B1C
  sorry

-- Define the theorem to prove the required angle.
theorem angle_between_AB_and_B1C_is_90 {A B C D A1 B1 C1 D1 : Point}
  (h_cube : cube ABCD A1B1C1D1) (h_length : edge_length ABCD A1B1C1D1 = 1) :
  angle_between_lines_in_cube A B C D A1 B1 C1 D1 = 90 :=
by sorry

end angle_between_AB_and_B1C_is_90_l440_440299


namespace complex_problem_l440_440824

open Complex

theorem complex_problem
  (z : ℂ)
  (h : z * (1 + I) = 1 - I) :
  abs (z - 1) = √2 :=
sorry

end complex_problem_l440_440824


namespace number_of_ordered_pairs_l440_440849

theorem number_of_ordered_pairs (f : ℝ → ℝ → ℝ)
  (log_b : ℝ → ℝ → ℝ) (a : ℝ) (b : ℤ) :
  (∀a : ℝ, a > 0) → (b ∈ set_of (λ b, 20 ≤ b ∧ b ≤ 220)) →
  (f (log_b b (real.sqrt a)) = log_b b (a ^ 2017)) →
  ∃ n : ℕ, n = 603 := 
sorry

end number_of_ordered_pairs_l440_440849


namespace value_of_a_l440_440040

theorem value_of_a (a : ℝ) :
  ∃ P : ℝ × ℝ, P = (4, a) ∧ (∃ θ : ℝ, θ = real.pi / 3 ∧ real.tan θ = a / 4) → a = 4 * real.sqrt 3 :=
by
  -- Proof goes here
  sorry

end value_of_a_l440_440040


namespace exact_time_is_3_07_27_l440_440132

theorem exact_time_is_3_07_27 (t : ℝ) (H1 : t > 0) (H2 : t < 60) 
(H3 : 6 * (t + 8) = 89 + 0.5 * t) : t = 7 + 27/60 :=
by
  sorry

end exact_time_is_3_07_27_l440_440132


namespace greatest_positive_integer_difference_l440_440283

-- Define the conditions
def condition_x (x : ℝ) : Prop := 4 < x ∧ x < 6
def condition_y (y : ℝ) : Prop := 6 < y ∧ y < 10

-- Define the problem statement
theorem greatest_positive_integer_difference (x y : ℕ) (hx : condition_x x) (hy : condition_y y) : y - x = 4 :=
sorry

end greatest_positive_integer_difference_l440_440283


namespace arithmetic_sequence_property_determine_a_and_k_sum_of_first_n_inversed_l440_440798

noncomputable def arithmetic_sequence (a : ℕ) : ℕ → ℕ
| 1 => a
| 2 => 4
| 3 => 3 * a
| n => (arithmetic_sequence 2) + (n - 2) * ((arithmetic_sequence 2) - (arithmetic_sequence 1))

noncomputable def S (a : ℕ) (n : ℕ) : ℕ :=
n * a + (n * (n - 1)) / 2 * ((arithmetic_sequence a 2) - a)

noncomputable def b (n : ℕ) : ℝ :=
1 / S 2 n

theorem arithmetic_sequence_property (a : ℕ) :
  arithmetic_sequence a 1 + arithmetic_sequence a 3 = 2 * arithmetic_sequence a 2 :=
by
  rw [arithmetic_sequence, arithmetic_sequence, arithmetic_sequence]
  sorry

theorem determine_a_and_k (a k : ℕ) (h : S a k = 90) : a = 2 ∧ k = 9 :=
by
  sorry

theorem sum_of_first_n_inversed (n : ℕ) :
  (finset.range n).sum (λ i, b (i+1)) = n / (n + 1) :=
by
  sorry

end arithmetic_sequence_property_determine_a_and_k_sum_of_first_n_inversed_l440_440798


namespace Michaela_needs_20_oranges_l440_440933

variable (M : ℕ)
variable (C : ℕ)

theorem Michaela_needs_20_oranges 
  (h1 : C = 2 * M)
  (h2 : M + C = 60):
  M = 20 :=
by 
  sorry

end Michaela_needs_20_oranges_l440_440933


namespace nonzero_fraction_power_zero_l440_440626

theorem nonzero_fraction_power_zero (a b : ℤ) (h : a ≠ 0 ∧ b ≠ 0) : ((a : ℚ) / b)^0 = 1 := 
by
  -- proof goes here
  sorry

end nonzero_fraction_power_zero_l440_440626


namespace length_of_AC_l440_440885

/-- 
Given points A, B, C, D, such that AB = 12 cm, DC = 15 cm, and AD = 9 cm, 
prove the length of AC to the nearest tenth of a centimeter.
--/

theorem length_of_AC 
  (A B C D : Type) 
  (hAB : dist A B = 12)
  (hDC : dist D C = 15)
  (hAD : dist A D = 9) :
  dist A C = 24.2 :=
by sorry

end length_of_AC_l440_440885


namespace tom_paid_1145_l440_440995

-- Define the quantities
def quantity_apples : ℕ := 8
def rate_apples : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 65

-- Calculate costs
def cost_apples : ℕ := quantity_apples * rate_apples
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Calculate the total amount paid
def total_amount_paid : ℕ := cost_apples + cost_mangoes

-- The theorem to prove
theorem tom_paid_1145 :
  total_amount_paid = 1145 :=
by sorry

end tom_paid_1145_l440_440995


namespace circle_equation_l440_440062

noncomputable def circle1 : ℝ × ℝ × ℝ := (1, 0, 1) -- Center at (1, 0) with radius 1
noncomputable def point_Q : ℝ × ℝ := (3, -√3)
noncomputable def line1 (x y : ℝ) : Prop := x + √3 * y = 0

def is_tangent (c1 : ℝ × ℝ × ℝ) (c2 : ℝ × ℝ × ℝ) : Prop :=
  let (x1, y1, r1) := c1 in
  let (x2, y2, r2) := c2 in
  (x1 - x2)^2 + (y1 - y2)^2 = (r1 + r2)^2

def tangent_at_point (c : ℝ × ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  let (x, y, r) := c in
  line1 (fst p) (snd p) ∧ (fst p - x) * (fst p - x) + (snd p - y) * (snd p - y) = r * r

def is_circle_equation (c : ℝ × ℝ × ℝ) (x y : ℝ) : Prop :=
  let (a, b, r) := c in
  (x - a)^2 + (y - b)^2 = r * r

theorem circle_equation 
  (circleC : ℝ × ℝ × ℝ)
  (h1 : is_tangent circle1 circleC)
  (h2 : tangent_at_point circleC point_Q)
  :
  is_circle_equation circleC 4 1 ∨ is_circle_equation circleC 6 (-4*√3) := sorry

end circle_equation_l440_440062


namespace rowing_around_lake_l440_440898

noncomputable def rowing_time
  (side_length : ℕ)
  (row_time_divisor : ℕ) -- Since he can row at twice the speed he can swim, we have a divisor
  (swim_time_per_mile : ℚ): ℚ := 
  let total_distance := 4 * side_length in
  let swim_speed := swim_time_per_mile in
  let row_speed := swim_speed / row_time_divisor in
  total_distance * row_speed

theorem rowing_around_lake
  (side_length : ℕ)
  (row_time_divisor : ℕ)
  (swim_time_per_mile : ℚ)
  (h_side_length : side_length = 15)
  (h_row_time_divisor : row_time_divisor = 2)
  (h_swim_time_per_mile : swim_time_per_mile = 20 / 60) :
  rowing_time side_length row_time_divisor swim_time_per_mile = 10 :=
by
  sorry

end rowing_around_lake_l440_440898


namespace f_2048_gt_13_div_2_l440_440915

noncomputable def f (n : ℕ) : ℝ := (finset.range n).sum (λ k, 1 / (k + 1))

example : 2048 = 2^11 :=
by norm_num1

theorem f_2048_gt_13_div_2 :
  f 2048 > 13 / 2 :=
begin
  -- Assume the pattern f(2^n) > (n+2) / 2
  have pattern : ∀ n : ℕ, f (2^n) > (n + 2) / 2,
  {
    -- This is where one would need to prove the pattern, omitted.
    sorry
  },
  specialize pattern 11,
  -- Since 2048 = 2^11, we apply the pattern to get the result for f(2048)
  linarith,
end

end f_2048_gt_13_div_2_l440_440915


namespace probability_all_quitters_same_tribe_l440_440220

theorem probability_all_quitters_same_tribe :
  ∀ (people : Finset ℕ) (tribe1 tribe2 : Finset ℕ) (choose : ℕ → ℕ → ℕ) (prob : ℚ),
  people.card = 20 →
  tribe1.card = 10 →
  tribe2.card = 10 →
  tribe1 ∪ tribe2 = people →
  tribe1 ∩ tribe2 = ∅ →
  choose 20 3 = 1140 →
  choose 10 3 = 120 →
  prob = (2 * choose 10 3) / choose 20 3 →
  prob = 20 / 95 :=
by
  intro people tribe1 tribe2 choose prob
  intros hp20 ht1 ht2 hu hi hchoose20 hchoose10 hprob
  sorry

end probability_all_quitters_same_tribe_l440_440220


namespace no_non_congruent_right_triangles_l440_440848

noncomputable def non_congruent_right_triangles : ℕ :=
  let is_right_triangle := λ (a b : ℕ) (c : ℝ), (a^2 + b^2 = c^2)
  ∃ a b : ℕ, ∃ c : ℝ, is_right_triangle a b c ∧ (
    let P := a + b + c
    let A := (1 / 2 : ℝ) * a * b
    P^2 = 4 * A
  )

theorem no_non_congruent_right_triangles : non_congruent_right_triangles = 0 := by
  sorry

end no_non_congruent_right_triangles_l440_440848


namespace smallest_n_for_sum_of_very_special_numbers_l440_440375

-- Definition of very special number
def very_special (x : ℝ) : Prop :=
  ∃ d : ℕ, (d ≤ 3) ∧ ( ∀ i < d, ∃ k : ℕ, (k < 10 ∧ (x * 10^i - k) % 10 = 0) ∧ (k = 0 ∨ k = 5) )

-- The main theorem to be proven
theorem smallest_n_for_sum_of_very_special_numbers : ∃ (n : ℕ), n = 2 ∧ 
  ( ∃ (a : ℕ → ℝ), (∀ i < n, very_special (a i) ∧ (a i < 10 ∧ (a i).denom ≤ 1000)) ∧ 
    ∑ i in finset.range n, a i = 1 ) :=
by
  sorry

end smallest_n_for_sum_of_very_special_numbers_l440_440375


namespace trajectory_proof_l440_440119

noncomputable def trajectory_eqn (x y : ℝ) : Prop :=
  (y + Real.sqrt 2) * (y - Real.sqrt 2) / (x * x) = -2

theorem trajectory_proof :
  ∀ (x y : ℝ), x ≠ 0 → trajectory_eqn x y → (y*y / 2 + x*x = 1) :=
by
  intros x y hx htrajectory
  sorry

end trajectory_proof_l440_440119


namespace projection_is_same_l440_440161

-- Define the vectors a, b, and p
def a : Matrix (Fin 2) (Fin 1) ℚ := ![![3], ![-2]]
def b : Matrix (Fin 2) (Fin 1) ℚ := ![![6], ![-1]]
def p : Matrix (Fin 2) (Fin 1) ℚ := ![![9/10], ![-27/10]]

-- The statement that needs to be proved
theorem projection_is_same (v : Matrix (Fin 2) (Fin 1) ℚ) : 
  (a • v = p ∧ b • v = p) → p = ![![9/10], ![-27/10]] :=
by
  sorry

end projection_is_same_l440_440161


namespace calculate_expr_equals_zero_l440_440367

theorem calculate_expr_equals_zero:
  ( (2^(1/2)) * (2^(2/3)) * (2^(5/6)) + log10 (1/100) - 3^(Real.log 2 / Real.log 3) = 0) :=
  by sorry

end calculate_expr_equals_zero_l440_440367


namespace john_pays_correct_tax_l440_440527

-- Define John's earnings and deductions
def earnings := 100000
def deductions := 30000

-- Define tax rates and income brackets
def lower_rate := 0.1
def upper_rate := 0.2
def lower_bracket := 20000

-- Define the taxable income calculation
def taxable_income := earnings - deductions

-- Define the tax calculations
def tax_at_lower_rate := lower_bracket * lower_rate
def remaining_income := taxable_income - lower_bracket
def tax_at_upper_rate := remaining_income * upper_rate

-- Define the total tax bill calculation
def total_tax_bill := tax_at_lower_rate + tax_at_upper_rate

-- Theorem: John pays $12,000 in taxes
theorem john_pays_correct_tax : total_tax_bill = 12000 := 
  by
  sorry

end john_pays_correct_tax_l440_440527


namespace color_theorem_l440_440780

theorem color_theorem (points : Finset (ℝ × ℝ)) (n : ℕ) (collinear : ∀ p1 p2 p3 ∈ points, ¬collinear_points p1 p2 p3) 
  (color : (ℝ × ℝ) → (ℝ × ℝ) → ℕ) 
  (triangle_color_condition : ∀ p1 p2 p3 ∈ points, 
    set_of_colors (color p1 p2) (color p2 p3) (color p3 p1)) : 
  n = (1 : ℕ) ∨ n ≥ 11 :=
sorry

end color_theorem_l440_440780


namespace weight_of_replaced_person_l440_440193

def average_weight_increases 
  (num_people : ℕ) 
  (weight_increase_per_person : ℝ) 
  (new_person_weight : ℝ) 
  (old_person_weight : ℝ) : Prop :=
  num_people * weight_increase_per_person = new_person_weight - old_person_weight

theorem weight_of_replaced_person 
  (num_people : ℕ)
  (weight_increase_per_person : ℝ)
  (new_person_weight : ℝ)
  (replaced_person_weight : ℝ) 
  (H : average_weight_increases num_people weight_increase_per_person new_person_weight replaced_person_weight) :
  replaced_person_weight = new_person_weight - num_people * weight_increase_per_person
  :=
  sorry

-- Specific instantiation for the given problem
example : weight_of_replaced_person 8 1.5 77 65 
  (by simp [average_weight_increases, mul_comm]) :=
  by rfl

end weight_of_replaced_person_l440_440193


namespace most_attendees_day_is_wednesday_l440_440727

def attendance_table : list (string × list (option bool)) :=
[ ("Anna", [some true, none, some true, none, none]),
  ("Bill", [none, some true, none, some true, some true]),
  ("Carl", [some true, some true, none, some true, some true]) ]

def attendees_per_day (table : list (string × list (option bool))) : list (nat) :=
let days := list.transpose (list.map snd table) in
list.map (λ day, list.length (list.filter (option.is_none) day)) days

noncomputable def day_with_max_attendees (table : list (string × list (option bool))) : nat :=
list.index_of (list.maximum (attendees_per_day table)) (attendees_per_day table)

theorem most_attendees_day_is_wednesday :
  day_with_max_attendees attendance_table = 2 := -- index 2 corresponds to Wednesday
by
  sorry

end most_attendees_day_is_wednesday_l440_440727


namespace domain_v_l440_440269

-- Define the function
def v (x : ℝ) : ℝ := 1 / (Real.cbrt(2 * x - 1))

-- State the domain condition
def domain_condition (x : ℝ) : Prop := 2 * x - 1 ≠ 0

-- The theorem asserting the domain of the function
theorem domain_v : {x : ℝ // domain_condition x} = {x : ℝ // x ≠ 1 / 2} := 
by
  sorry

end domain_v_l440_440269


namespace simplify_expression_l440_440576

theorem simplify_expression
  (h0 : (Real.pi / 2) < 2 ∧ 2 < Real.pi)  -- Given conditions on 2 related to π.
  (h1 : Real.sin 2 > 0)  -- Given condition that sin 2 is positive.
  (h2 : Real.cos 2 < 0)  -- Given condition that cos 2 is negative.
  : 2 * Real.sqrt (1 + Real.sin 4) + Real.sqrt (2 + 2 * Real.cos 4) = 2 * Real.sin 2 :=
sorry

end simplify_expression_l440_440576


namespace domain_all_real_iff_l440_440732

theorem domain_all_real_iff (k : ℝ) :
  (∀ x : ℝ, -3 * x ^ 2 - x + k ≠ 0 ) ↔ k < -1 / 12 :=
by
  sorry

end domain_all_real_iff_l440_440732


namespace TF_AB_orthogonal_circumcenter_locus_l440_440655

noncomputable theory
open_locale classical

-- Define the necessary structures and assumptions
variables {O A B S J H F T : Type} [Coord O A B S J H F T]
variables (circle : Circle O) (AB_perpendicular : Perpendicular O A B) (SJ_perpendicular : Perpendicular O S J)

-- Assume initial point constraints
variables (arc_H : OnArc H B S)

-- Intersection points F and T
variables (F_defined : Intersection F (LineAH O A H) (LineSB O S B))
variables (T_defined : Intersection T (LineSA O S A) (LineBH O B H))

-- Define orthogonality of TF and AB
theorem TF_AB_orthogonal : IsOrthogonal (LineTF O T F) (LineAB O A B) :=
sorry

-- Define and find the locus of the circumcenter of triangle FSH
theorem circumcenter_locus : ∃ Locus, ∀ (M : Type), IsCircumcenter (TriangleFSH O F S H) M → OnLocus M Locus :=
sorry

end TF_AB_orthogonal_circumcenter_locus_l440_440655


namespace quadrilateral_is_square_and_point_is_intersection_l440_440337

-- Define the quadrilateral properties and given conditions
variables {a b c d : ℝ} (S : ℝ)

-- Define MP^2 + NP^2 + KP^2 + LP^2 = 2S condition
axiom distance_condition : a^2 + b^2 + c^2 + d^2 = 2 * S

-- Lean statement to formalize the problem:
theorem quadrilateral_is_square_and_point_is_intersection (h : a^2 + b^2 + c^2 + d^2 = 2 * S) :
  -- first condition: The quadrilateral is a square
  let is_square : Prop := sorry in
  -- second condition: Point P is the intersection of the diagonals
  let is_intersection : Prop := sorry in
  is_square ∧ is_intersection :=
by 
  sorry

end quadrilateral_is_square_and_point_is_intersection_l440_440337


namespace probability_exactly_four_1s_l440_440257

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_four_1s_in_12_dice : ℝ :=
  let n := 12
  let k := 4
  let p := 1 / 6
  let q := 5 / 6
  (binomial_coefficient n k : ℝ) * p^k * q^(n - k)

theorem probability_exactly_four_1s : probability_of_four_1s_in_12_dice ≈ 0.089 :=
  by
  sorry

end probability_exactly_four_1s_l440_440257


namespace asymptotes_of_hyperbola_l440_440061

noncomputable def hyperbola (x y m : ℝ) : Prop := x^2 - y^2 / m = 1
def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def focus_of_parabola (p : ℝ) : ℝ := p / 2

theorem asymptotes_of_hyperbola :
  ∃ (x y m : ℝ), hyperbola x y m ∧ parabola x y ∧ (∃ (F : ℝ), F = focus_of_parabola 8 ∧ dist (x, y) (F, 0) = 5) →
  (∀ (a b : ℝ), (a = √3 ∧ b = 1) → y = a * x ∨ y = -a * x ) :=
sorry

end asymptotes_of_hyperbola_l440_440061


namespace sum_of_primes_between_10_and_20_l440_440363

def is_prime (n : Nat) : Prop := 
  ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

theorem sum_of_primes_between_10_and_20 : 
  (∑ i in { x : Nat | 10 < x ∧ x < 20 ∧ is_prime x }, i) = 60 := 
by
  sorry

end sum_of_primes_between_10_and_20_l440_440363


namespace M_inter_N_eq_l440_440082

def M : Set ℕ := {1, 2, 3}
def N : Set ℤ := {x | 1 < x ∧ x < 4}

theorem M_inter_N_eq : M ∩ N = {2, 3} :=
sorry

end M_inter_N_eq_l440_440082


namespace regular_pentagon_divided_into_five_congruent_smaller_l440_440199

theorem regular_pentagon_divided_into_five_congruent_smaller :
  (∀ (ABC : Triangle), (ABC.angle_at_base = 36 ∨ ABC.angle_at_base = 72) → ABC.ratio_of_sides = (Real.sqrt 5 + 1) / 2) →
  ∃ (small_pentagons : Finset Pentagon), small_pentagons.card = 5 ∧
  (∀ p ∈ small_pentagons, p.is_regular) ∧
  ∀ p₁ p₂ ∈ small_pentagons, p₁ ≠ p₂ → p₁ ≃ p₂ :=
by
  sorry

end regular_pentagon_divided_into_five_congruent_smaller_l440_440199


namespace value_of_expression_l440_440271

theorem value_of_expression (a : ℚ) (h : a = 1/3) : (3 * a⁻¹ + a⁻¹ / 3) / (2 * a) = 15 := by
  sorry

end value_of_expression_l440_440271


namespace min_diagonal_length_of_inscribed_rectangle_l440_440790

theorem min_diagonal_length_of_inscribed_rectangle
  (A B C : Type)
  [ordered_triangle A B C]
  (AC BC AB : ℝ)
  (h_right : ∠ C = π / 2)
  (h_hypotenuse : AB = sqrt (AC ^ 2 + BC ^ 2)):
  (min_diagonal AC BC AB = (AC * BC) / AB) :=
sorry

end min_diagonal_length_of_inscribed_rectangle_l440_440790


namespace die_roll_divisor_of_12_prob_l440_440664

def fair_die_probability_divisor_of_12 : Prop :=
  let favorable_outcomes := {1, 2, 3, 4, 6}
  let total_outcomes := 6
  let probability := favorable_outcomes.size / total_outcomes
  probability = 5 / 6

theorem die_roll_divisor_of_12_prob:
  fair_die_probability_divisor_of_12 :=
by
  sorry

end die_roll_divisor_of_12_prob_l440_440664


namespace trig_identity_l440_440026

theorem trig_identity (x : ℝ) (h : sin (x + π / 6) = 1 / 3) :
  sin (x - 5 * π / 6) + sin (π / 3 - x) ^ 2 = 5 / 9 :=
by sorry

end trig_identity_l440_440026


namespace range_of_a_l440_440063

noncomputable def f (x : ℝ) : ℝ := (2 * real.log x) / x

theorem range_of_a (a : ℝ) (h : ∀ x, (1/real.exp 1) ≤ x ∧ x ≤ real.exp 2 → real.exp (x/a) > x*x) :
  0 < a ∧ a < real.exp 1 / 2 := 
sorry

end range_of_a_l440_440063


namespace shortest_path_light_ray_l440_440319

noncomputable def point_reflection (x₀ y₀ : ℝ) : ℝ × ℝ :=
  (-x₀, y₀)

noncomputable def distance (x₀ y₀ x₁ y₁ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₀) ^ 2 + (y₁ - y₀) ^ 2)

theorem shortest_path_light_ray :
  let A := (1 : ℝ, 0 : ℝ)
  let M := point_reflection 1 0
  let C := (3 : ℝ, 3 : ℝ)
  let radius := (1 : ℝ)
  let dist_MC := distance (-1) 0 3 3
  dist_MC - radius = 4 :=
by
  sorry

end shortest_path_light_ray_l440_440319


namespace benny_missed_games_l440_440361

noncomputable def baseball_games_played := 39
noncomputable def cancellation_rate := 0.15
noncomputable def attendance_rate := 0.45

theorem benny_missed_games :
  let games_cancelled := (cancellation_rate * baseball_games_played).toInt;
  let games_played := baseball_games_played - games_cancelled;
  let games_attended := (attendance_rate * games_played).toInt;
  games_played - games_attended = 19 :=
by
  sorry

end benny_missed_games_l440_440361


namespace distance_between_parallel_lines_l440_440619

theorem distance_between_parallel_lines (a d : ℝ) (d_pos : 0 ≤ d) (a_pos : 0 ≤ a) :
  {d_ | d_ = d + a ∨ d_ = |d - a|} = {d + a, abs (d - a)} :=
by
  sorry

end distance_between_parallel_lines_l440_440619


namespace length_AB_proof_l440_440622

noncomputable def length_AB (AB BC CA : ℝ) (DEF DE EF DF : ℝ) (angle_BAC angle_DEF : ℝ) : ℝ :=
  if h : (angle_BAC = 120 ∧ angle_DEF = 120 ∧ AB = 5 ∧ BC = 17 ∧ CA = 12 ∧ DE = 9 ∧ EF = 15 ∧ DF = 12) then
    (5 * 15) / 17
  else
    0

theorem length_AB_proof : length_AB 5 17 12 9 15 12 120 120 = 75 / 17 := by
  sorry

end length_AB_proof_l440_440622


namespace probability_of_24_is_1_div_1296_l440_440753

def sum_of_dice_is_24 (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 + d2 + d3 + d4 = 24

def probability_of_six (d : ℕ) : Rat :=
  if d = 6 then 1 / 6 else 0

def probability_of_sum_24 (d1 d2 d3 d4 : ℕ) : Rat :=
  (probability_of_six d1) * (probability_of_six d2) * (probability_of_six d3) * (probability_of_six d4)

theorem probability_of_24_is_1_div_1296 :
  (probability_of_sum_24 6 6 6 6) = 1 / 1296 :=
by
  sorry

end probability_of_24_is_1_div_1296_l440_440753


namespace prob_sum_24_four_dice_l440_440757

section
open ProbabilityTheory

/-- Define the event E24 as the event where the sum of numbers on the top faces of four six-sided dice is 24 -/
def E24 : Event (StdGen) :=
eventOfFun {ω | ∑ i in range 4, (ω.gen_uniform int (6-1)) + 1 = 24}

/-- Probability that the sum of the numbers on top faces of four six-sided dice is 24 is 1/1296. -/
theorem prob_sum_24_four_dice : ⋆{ P(E24) = 1/1296 } := sorry

end

end prob_sum_24_four_dice_l440_440757


namespace max_S_over_R_squared_l440_440875

theorem max_S_over_R_squared (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let S := 2 * (a * b + b * c + c * a)
  let R := (sqrt (a^2 + b^2 + c^2)) / 2
  (S / R^2) ≤ (2 / 3) * (3 + sqrt 3) :=
by sorry

end max_S_over_R_squared_l440_440875


namespace solution_set_of_inequality_l440_440444

noncomputable def f : ℝ → ℝ := sorry
axiom even_f : ∀ x, f (-x) = f x
axiom deriv_f_lt_f : ∀ x, f' x < f x
axiom f_1_3_minus_x : ∀ x, f (x + 1) = f (3 - x)
axiom f_2015 : f 2015 = 2

theorem solution_set_of_inequality : { x : ℝ | f x < 2 * real.exp (x - 1) } = set.Ioi 1 :=
sorry

end solution_set_of_inequality_l440_440444


namespace probability_exactly_four_1s_l440_440256

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_four_1s_in_12_dice : ℝ :=
  let n := 12
  let k := 4
  let p := 1 / 6
  let q := 5 / 6
  (binomial_coefficient n k : ℝ) * p^k * q^(n - k)

theorem probability_exactly_four_1s : probability_of_four_1s_in_12_dice ≈ 0.089 :=
  by
  sorry

end probability_exactly_four_1s_l440_440256


namespace general_term_sequence_l440_440221

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = (2 * a n) / (a n + 2)

theorem general_term_sequence (a : ℕ → ℝ) (h : sequence a) :
  ∀ n, a n = 2 / (n + 1) := 
sorry

end general_term_sequence_l440_440221


namespace probability_divisor_of_12_l440_440666

open Probability

def divisors_of_12 := {1, 2, 3, 4, 6}

theorem probability_divisor_of_12 :
  ∃ (fair_die_roll : ProbabilityMeasure (Fin 6)), 
    P (fun x => x.val + 1 ∈ divisors_of_12) = 5 / 6 := 
by
  sorry

end probability_divisor_of_12_l440_440666


namespace number_of_solutions_l440_440740

theorem number_of_solutions :
  ∃ (s : ℕ), s = 10626 ∧ 
  ∀ (x y z u : ℕ), x + y + z + u ≤ 20 → ∃ (v : ℕ), v = 20 - (x + y + z + u) :=
begin
  sorry
end

end number_of_solutions_l440_440740


namespace max_items_per_cycle_l440_440697

theorem max_items_per_cycle (shirts : Nat) (pants : Nat) (sweaters : Nat) (jeans : Nat)
  (cycle_time : Nat) (total_time : Nat) 
  (h_shirts : shirts = 18)
  (h_pants : pants = 12)
  (h_sweaters : sweaters = 17)
  (h_jeans : jeans = 13)
  (h_cycle_time : cycle_time = 45)
  (h_total_time : total_time = 3 * 60) :
  (shirts + pants + sweaters + jeans) / (total_time / cycle_time) = 15 :=
by
  -- We will provide the proof here
  sorry

end max_items_per_cycle_l440_440697


namespace emma_balance_proof_l440_440730

def monday_withdrawal (balance : ℝ) : ℝ :=
  0.08 * balance

def tuesday_deposit (withdrawal : ℝ) : ℝ :=
  0.25 * withdrawal

def wednesday_deposit (withdrawal : ℝ) : ℝ :=
  1.5 * withdrawal

def one_week_later_withdrawal (balance : ℝ) : ℝ :=
  0.05 * balance

theorem emma_balance_proof :
  let initial_balance := 1200 in
  let withdrawal_monday := monday_withdrawal initial_balance in
  let balance_tuesday := initial_balance - withdrawal_monday in
  let deposit_tuesday := tuesday_deposit withdrawal_monday in
  let balance_wednesday := balance_tuesday + deposit_tuesday in
  let deposit_wednesday := wednesday_deposit withdrawal_monday in
  let balance_one_week := balance_wednesday + deposit_wednesday in
  let withdrawal_one_week := one_week_later_withdrawal balance_one_week in
  let final_balance := balance_one_week - withdrawal_one_week in
  final_balance = 1208.40 := by
  sorry

end emma_balance_proof_l440_440730


namespace cos_inequality_l440_440153

theorem cos_inequality (n : ℕ) (x : ℝ) (h : n > 0) : 
  (finset.range n).sum (λ k, |real.cos (2 ^ k * x)|) ≥ n / 2 :=
sorry

end cos_inequality_l440_440153


namespace black_female_pigeons_more_than_males_l440_440677

theorem black_female_pigeons_more_than_males:
  let total_pigeons := 70
  let black_pigeons := total_pigeons / 2
  let black_male_percentage := 20 / 100
  let black_male_pigeons := black_pigeons * black_male_percentage
  let black_female_pigeons := black_pigeons - black_male_pigeons
  black_female_pigeons - black_male_pigeons = 21 := by
{
  let total_pigeons := 70
  let black_pigeons := total_pigeons / 2
  let black_male_percentage := 20 / 100
  let black_male_pigeons := black_pigeons * black_male_percentage
  let black_female_pigeons := black_pigeons - black_male_pigeons
  show black_female_pigeons - black_male_pigeons = 21
  sorry
}

end black_female_pigeons_more_than_males_l440_440677


namespace max_area_ABCD_l440_440532

theorem max_area_ABCD
    (A B C D : Point ℝ)
    (h_convex : ConvexQuadrilateral A B C D)
    (BC : dist B C = 3)
    (CD : dist C D = 5)
    (equilateral_centroids : Equilateral (centroid A B C) (centroid B C D) (centroid A C D)) :
    area A B C D ≤ 8.5 * (Real.sqrt 3) + 7.5 := 
sorry

end max_area_ABCD_l440_440532


namespace largest_not_sum_of_37_and_composite_l440_440632

theorem largest_not_sum_of_37_and_composite:
  ∃ (n : ℕ), (n = 66) ∧ (∀ (a b : ℕ), (a > 0) → (b > 0) → (¬(b = 1 ∨ b = 37) → b.mod 37 < 37) → (¬n = 37 * a + b ∧ ¬(b > 1 ∧ ∀ (d : ℕ), d < b → d.divides b → d = 1 ∨ d = b)))
:= by
  sorry

end largest_not_sum_of_37_and_composite_l440_440632


namespace fresh_grapes_water_percent_l440_440020

theorem fresh_grapes_water_percent
  (P : ℕ) -- Percentage of water in fresh grapes
  (H1 : ∃ P : ℕ, P > 0 ∧ P < 100) -- P is a valid percentage
  (H2 : ∀ fresh_weight dried_weight, fresh_weight = 20 → dried_weight = 2.5 → P / 100 * fresh_weight = 80 / 100 * dried_weight) :
  P = 90 :=
by
  sorry

end fresh_grapes_water_percent_l440_440020


namespace real_values_of_c_l440_440412

theorem real_values_of_c (c : ℝ) : ∃ c₁ c₂ : ℝ, (abs ((3/4 : ℝ) - c * complex.i) = 5/4) ∧ 
                                                (c₁ ≠ c₂) ∧ (abs c₁ = 1) ∧ 
                                                (abs c₂ = 1) := by
  sorry

end real_values_of_c_l440_440412


namespace number_of_valid_integers_between_1_and_300_l440_440846

theorem number_of_valid_integers_between_1_and_300 :
  let is_valid (n : ℕ) : Prop :=
    n % 10 = 0 ∧ n % 3 ≠ 0 ∧ n % 7 ≠ 0 ∧ (1 ≤ n) ∧ (n ≤ 300) in
  (finset.filter is_valid (finset.range 301)).card = 17 := sorry

end number_of_valid_integers_between_1_and_300_l440_440846


namespace convex_value_sequence_count_l440_440018

theorem convex_value_sequence_count :
  ∃ (a : ℕ+ → ℕ+), (∀ k : ℕ+, 
    (k = 1 → a 1 = 1) ∧ 
    (k = 2 → a 2 ≤ 3) ∧ 
    (k = 3 → a 3 ≤ 3) ∧ 
    (k = 4 → a 4 ≤ 9) ∧ 
    (k = 5 → a 5 ≤ 9) ∧ 
    (k = 2 → max (a 1) (a 2) = 3) ∧ 
    (k = 3 → max (a 1) (a 2) (a 3) = 3) ∧ 
    (k = 4 → max (a 1) (a 2) (a 3) (a 4) = 9) ∧ 
    (k = 5 → max (a 1) (a 2) (a 3) (a 4) (a 5) = 9)) → 
    card {a : ℕ+ → ℕ+ // ∀ k : ℕ+, 
      (k = 1 → a 1 = 1) ∧ 
      (k = 2 → a 2 ≤ 3) ∧ 
      (k = 3 → a 3 ≤ 3) ∧ 
      (k = 4 → a 4 ≤ 9) ∧ 
      (k = 5 → a 5 ≤ 9) ∧ 
      (k = 2 → max (a 1) (a 2) = 3) ∧ 
      (k = 3 → max (a 1) (a 2) (a 3) = 3) ∧ 
      (k = 4 → max (a 1) (a 2) (a 3) (a 4) = 9) ∧ 
      (k = 5 → max (a 1) (a 2) (a 3) (a 4) (a 5) = 9) } = 27 :=
sorry

end convex_value_sequence_count_l440_440018


namespace polynomial_div_rem_l440_440272

/-- Prove the remainder when 4*z^4 + 2*z^3 - 5*z^2 - 20*z + 7 is divided by 4*z + 7 is 4795/64. -/
theorem polynomial_div_rem (z : ℚ) :
  (eval (-7 / 4) (4 * z ^ 4 + 2 * z ^ 3 - 5 * z ^ 2 - 20 * z + 7)) = 4795 / 64 := sorry

end polynomial_div_rem_l440_440272


namespace find_c_plus_inv_b_l440_440914

theorem find_c_plus_inv_b (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h1 : a * b * c = 1) (h2 : a + 1 / c = 7) (h3 : b + 1 / a = 35) :
  c + 1 / b = 11 / 61 :=
by
  sorry

end find_c_plus_inv_b_l440_440914


namespace area_quadrilateral_EFCD_l440_440519

-- Define the conditions as given in the problem
variables (AB CD : ℝ) (hAB : AB = 10) (hCD : CD = 30)
variable (altitude_ABCD : ℝ) (h_altitude_ABCD : altitude_ABCD = 15)
variables (E F : ℝ) 
variables (hE : E = (1/2) * (AB + CD)) 
variables (hF : F = (1/2) * (AB + CD))

-- Translate the conditions and the question to a proof problem
theorem area_quadrilateral_EFCD : 
  let EF := (AB + CD) / 2 in
  let altitude_EFCD := altitude_ABCD / 2 in
  let area_EFCD := altitude_EFCD * (EF + CD) / 2 in
  area_EFCD = 187.5 :=
by
  sorry

end area_quadrilateral_EFCD_l440_440519


namespace triangle_area_l440_440265

def line1 (x : ℝ) : ℝ := (1 / 3) * x + (2 / 3)
def line2 (x : ℝ) : ℝ := 3 * x - 2
def line3 (x y : ℝ) : Prop := x + y = 8

def point_intersection1 : ℝ × ℝ := (5.5, 2.5)
def point_intersection2 : ℝ × ℝ := (2.5, 5.5)
def point_intersection_given : ℝ × ℝ := (1, 1)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 * (B.2 - C.2))
            + (B.1 * (C.2 - A.2))
            + (C.1 * (A.2 - B.2)))

theorem triangle_area : 
  let A := point_intersection_given 
  let B := point_intersection2 
  let C := point_intersection1 in
  area_of_triangle A B C = 8.625 :=
by
  -- Sorry to skip the proof
  sorry

end triangle_area_l440_440265


namespace f1_not_differentiable_f2_not_differentiable_f3_differentiable_f4_differentiable_l440_440704

-- Define the conditions and the associated proofs in Lean 4

def f1 (z : ℂ) : ℂ := (z.im ^ 2 + z.re ^ 2) + 2 * complex.I * (z.re * z.im)
def f2 (z : ℂ) : ℂ := (real.exp (3 * z.re) * real.cos (2 * z.im)) - complex.I * (real.exp (3 * z.re) * real.sin (2 * z.im))
def f3 (n : ℕ) (z : ℂ) : ℂ := n^2 * z^n
def f4 (z : ℂ) : ℂ := 2 * complex.sinh (real.cos z) + 2 * complex.cosh (real.sin z)

theorem f1_not_differentiable : ¬ complex.differentiable f1 := 
sorry

theorem f2_not_differentiable : ¬ complex.differentiable f2 := 
sorry

theorem f3_differentiable (n : ℕ) : complex.differentiable (f3 n) := 
sorry

theorem f4_differentiable : complex.differentiable f4 := 
sorry

end f1_not_differentiable_f2_not_differentiable_f3_differentiable_f4_differentiable_l440_440704


namespace fifteenth_term_is_correct_l440_440708

-- Define the initial conditions of the arithmetic sequence
def firstTerm : ℕ := 4
def secondTerm : ℕ := 9

-- Calculate the common difference
def commonDifference : ℕ := secondTerm - firstTerm

-- Define the nth term formula of the arithmetic sequence
def nthTerm (a d n : ℕ) : ℕ := a + (n - 1) * d

-- The main statement: proving that the 15th term of the given sequence is 74
theorem fifteenth_term_is_correct : nthTerm firstTerm commonDifference 15 = 74 :=
by
  sorry

end fifteenth_term_is_correct_l440_440708


namespace cell_division_50_closest_to_10_15_l440_440659

theorem cell_division_50_closest_to_10_15 :
  10^14 < 2^50 ∧ 2^50 < 10^16 :=
sorry

end cell_division_50_closest_to_10_15_l440_440659


namespace rationalize_denominator_l440_440097

theorem rationalize_denominator (t : ℝ) (h : t = 1 / (1 - Real.sqrt (Real.sqrt 2))) : 
  t = -(1 + Real.sqrt (Real.sqrt 2)) * (1 + Real.sqrt 2) :=
by
  sorry

end rationalize_denominator_l440_440097
