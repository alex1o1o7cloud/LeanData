import Mathlib

namespace john_total_distance_l952_95224

def speed : ℕ := 45
def time1 : ℕ := 2
def time2 : ℕ := 3

theorem john_total_distance:
  speed * (time1 + time2) = 225 := by
  sorry

end john_total_distance_l952_95224


namespace find_n_values_l952_95230

theorem find_n_values (n : ℤ) (hn : ∃ x y : ℤ, x ≠ y ∧ x^2 - 6*x - 4*n^2 - 32*n = 0 ∧ y^2 - 6*y - 4*n^2 - 32*n = 0):
  n = 10 ∨ n = 0 ∨ n = -8 ∨ n = -18 := 
sorry

end find_n_values_l952_95230


namespace intersection_single_point_l952_95241

def A (x y : ℝ) := x^2 + y^2 = 4
def B (x y : ℝ) (r : ℝ) := (x - 3)^2 + (y - 4)^2 = r^2

theorem intersection_single_point (r : ℝ) (h : r > 0) :
  (∃! p : ℝ × ℝ, A p.1 p.2 ∧ B p.1 p.2 r) → r = 3 :=
by
  apply sorry -- Proof goes here

end intersection_single_point_l952_95241


namespace sum_of_fifth_powers_l952_95261

theorem sum_of_fifth_powers (a b c d : ℝ) (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := sorry

end sum_of_fifth_powers_l952_95261


namespace salary_spending_l952_95291

theorem salary_spending (S_A S_B : ℝ) (P_A P_B : ℝ) 
  (h1 : S_A = 4500) 
  (h2 : S_A + S_B = 6000)
  (h3 : P_B = 0.85) 
  (h4 : S_A * (1 - P_A) = S_B * (1 - P_B)) : 
  P_A = 0.95 :=
by
  -- Start proofs here
  sorry

end salary_spending_l952_95291


namespace continuous_at_4_l952_95246

noncomputable def f (x : ℝ) := 3 * x^2 - 3

theorem continuous_at_4 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x - f 4| < ε :=
by
  sorry

end continuous_at_4_l952_95246


namespace equation_of_line_parallel_to_x_axis_l952_95216

theorem equation_of_line_parallel_to_x_axis (x: ℝ) :
  ∃ (y: ℝ), (y-2=0) ∧ ∀ (P: ℝ × ℝ), (P = (1, 2)) → P.2 = 2 := 
by
  sorry

end equation_of_line_parallel_to_x_axis_l952_95216


namespace fraction_of_girls_is_one_half_l952_95248

def fraction_of_girls (total_students_jasper : ℕ) (ratio_jasper : ℕ × ℕ) (total_students_brookstone : ℕ) (ratio_brookstone : ℕ × ℕ) : ℚ :=
  let (boys_ratio_jasper, girls_ratio_jasper) := ratio_jasper
  let (boys_ratio_brookstone, girls_ratio_brookstone) := ratio_brookstone
  let girls_jasper := (total_students_jasper * girls_ratio_jasper) / (boys_ratio_jasper + girls_ratio_jasper)
  let girls_brookstone := (total_students_brookstone * girls_ratio_brookstone) / (boys_ratio_brookstone + girls_ratio_brookstone)
  let total_girls := girls_jasper + girls_brookstone
  let total_students := total_students_jasper + total_students_brookstone
  total_girls / total_students

theorem fraction_of_girls_is_one_half :
  fraction_of_girls 360 (7, 5) 240 (3, 5) = 1 / 2 :=
  sorry

end fraction_of_girls_is_one_half_l952_95248


namespace negation_if_then_l952_95284

theorem negation_if_then (x : ℝ) : ¬ (x > 2 → x > 1) ↔ (x ≤ 2 → x ≤ 1) :=
by 
  sorry

end negation_if_then_l952_95284


namespace part1_part2_l952_95218

theorem part1 (m : ℝ) (P : ℝ × ℝ) : (P = (3*m - 6, m + 1)) → (P.1 = 0) → (P = (0, 3)) :=
by
  sorry

theorem part2 (m : ℝ) (A P : ℝ × ℝ) : A = (1, -2) → (P = (3*m - 6, m + 1)) → (P.2 = A.2) → (P = (-15, -2)) :=
by
  sorry

end part1_part2_l952_95218


namespace max_minute_hands_l952_95214

theorem max_minute_hands (m n : ℕ) (h1 : m * n = 27) : m + n ≤ 28 :=
by sorry

end max_minute_hands_l952_95214


namespace divisors_log_sum_eq_l952_95260

open BigOperators

/-- Given the sum of the base-10 logarithms of the divisors of \( 10^{2n} = 4752 \), prove that \( n = 12 \). -/
theorem divisors_log_sum_eq (n : ℕ) (h : ∑ a in Finset.range (2*n + 1), ∑ b in Finset.range (2*n + 1), 
  (a * Real.log (2) / Real.log (10) + b * Real.log (5) / Real.log (10)) = 4752) : n = 12 :=
by {
  sorry
}

end divisors_log_sum_eq_l952_95260


namespace quadratic_function_difference_zero_l952_95274

theorem quadratic_function_difference_zero
  (a b c x1 x2 x3 x4 x5 p q : ℝ)
  (h1 : a ≠ 0)
  (h2 : a * x1^2 + b * x1 + c = 5)
  (h3 : a * (x2 + x3 + x4 + x5)^2 + b * (x2 + x3 + x4 + x5) + c = 5)
  (h4 : x1 ≠ x2 + x3 + x4 + x5)
  (h5 : a * (x1 + x2)^2 + b * (x1 + x2) + c = p)
  (h6 : a * (x3 + x4 + x5)^2 + b * (x3 + x4 + x5) + c = q) :
  p - q = 0 := 
sorry

end quadratic_function_difference_zero_l952_95274


namespace math_problem_l952_95290

noncomputable def exponential_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n = 2 * 3^(n - 1)

noncomputable def geometric_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(2 * 3^n - 2) / 2

theorem math_problem 
  (a : ℕ → ℝ) (b : ℕ → ℕ) (c : ℕ → ℝ) (S T P : ℕ → ℝ)
  (h1 : exponential_sequence a)
  (h2 : a 1 * a 3 = 36)
  (h3 : a 3 + a 4 = 9 * (a 1 + a 2))
  (h4 : ∀ n, S n + 1 = 3^(b n))
  (h5 : ∀ n, T n = (2 * n - 1) * 3^n / 2 + 1 / 2)
  (h6 : ∀ n, c n = a n / ((a n + 1) * (a (n + 1) + 1)))
  (h7 : ∀ n, P (2 * n) = 1 / 6 - 1 / (4 * 3^(2 * n) + 2)) :
  (∀ n, a n = 2 * 3^(n - 1)) ∧
  ∀ n, b n = n ∧
  ∀ n, a n * b n = 2 * n * 3^(n - 1) ∧
  ∃ n, T n = (2 * n - 1) * 3^n / 2 + 1 / 2 ∧
  P (2 * n) = 1 / 6 - 1 / (4 * 3^(2 * n) + 2) :=
by sorry

end math_problem_l952_95290


namespace midpoint_P_AB_l952_95215

structure Point := (x : ℝ) (y : ℝ)

def segment_midpoint (P A B : Point) : Prop := P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

variables {A D C E P B : Point}
variables (h1 : A.x = D.x ∧ A.y = D.y)
variables (h2 : D.x = C.x ∧ D.y = C.y)
variables (h3 : D.x = P.x ∧ D.y = P.y ∧ P.x = E.x ∧ P.y = E.y)
variables (h4 : B.x = E.x ∧ B.y = E.y)
variables (h5 : A.x = C.x ∧ A.y = C.y)
variables (angle_ADC : ∀ x y : ℝ, (x - A.x)^2 + (y - A.y)^2 = (x - D.x)^2 + (y - D.y)^2 → (x - C.x)^2 + (y - C.y)^2 = (x - D.x)^2 + (y - D.y)^2)
variables (angle_DPE : ∀ x y : ℝ, (x - D.x)^2 + (y - P.y)^2 = (x - P.x)^2 + (y - E.y)^2 → (x - E.x)^2 + (y - E.y)^2 = (x - P.x)^2 + (y - E.y)^2)
variables (angle_BEC : ∀ x y : ℝ, (x - B.x)^2 + (y - E.y)^2 = (x - E.x)^2 + (y - C.y)^2 → (x - B.x)^2 + (y - C.y)^2 = (x - E.x)^2 + (y - C.y)^2)

theorem midpoint_P_AB : segment_midpoint P A B := 
sorry

end midpoint_P_AB_l952_95215


namespace sin_eq_sin_sinx_l952_95280

noncomputable def S (x : ℝ) := Real.sin x - x

theorem sin_eq_sin_sinx (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.arcsin 742) :
  ∃! x, Real.sin x = Real.sin (Real.sin x) :=
by
  sorry

end sin_eq_sin_sinx_l952_95280


namespace seating_arrangement_l952_95254

def num_ways_to_seat (A B C D E F : Type) (chairs : List (Option Type)) : Nat := sorry

theorem seating_arrangement {A B C D E F : Type} :
  ∀ (chairs : List (Option Type)),
    (A ≠ B ∧ A ≠ C ∧ F ≠ B) → num_ways_to_seat A B C D E F chairs = 28 :=
by
  sorry

end seating_arrangement_l952_95254


namespace find_b_l952_95275

-- Define the conditions as constants
def x := 36 -- angle a in degrees
def y := 44 -- given
def z := 52 -- given
def w := 48 -- angle b we need to find

-- Define the problem as a theorem
theorem find_b : x + w + y + z = 180 :=
by
  -- Substitute the given values and show the sum
  have h : 36 + 48 + 44 + 52 = 180 := by norm_num
  exact h

end find_b_l952_95275


namespace power_equivalence_l952_95277

theorem power_equivalence (p : ℕ) (hp : 81^10 = 3^p) : p = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end power_equivalence_l952_95277


namespace geometric_series_sum_l952_95238

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l952_95238


namespace rectangle_area_constant_l952_95223

noncomputable def k (d : ℝ) : ℝ :=
  let x := d / Real.sqrt 29
  10 / 29

theorem rectangle_area_constant (d : ℝ) : 
  let k := 10 / 29
  let length := 5 * (d / Real.sqrt 29)
  let width := 2 * (d / Real.sqrt 29)
  let diagonal := d
  let area := length * width
  area = k * d^2 :=
by
  sorry

end rectangle_area_constant_l952_95223


namespace original_number_is_neg2_l952_95283

theorem original_number_is_neg2 (x : ℚ) (h : 2 - 1/x = 5/2) : x = -2 :=
sorry

end original_number_is_neg2_l952_95283


namespace sphere_radius_eq_three_l952_95295

theorem sphere_radius_eq_three (r : ℝ) (h : 4 / 3 * π * r ^ 3 = 4 * π * r ^ 2) : r = 3 :=
by
  sorry

end sphere_radius_eq_three_l952_95295


namespace find_A_in_terms_of_B_and_C_l952_95220

theorem find_A_in_terms_of_B_and_C 
  (A B C : ℝ) (hB : B ≠ 0) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = A * x - 2 * B^2)
  (hg : ∀ x, g x = B * x + C * x^2)
  (hfg : f (g 1) = 4 * B^2)
  : A = 6 * B * B / (B + C) :=
by
  sorry

end find_A_in_terms_of_B_and_C_l952_95220


namespace largest_integer_n_l952_95270

-- Define the condition for existence of positive integers x, y, z that satisfy the given equation
def condition (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10

-- State that the largest such integer n is 4
theorem largest_integer_n : ∀ (n : ℕ), condition n → n ≤ 4 :=
by {
  sorry
}

end largest_integer_n_l952_95270


namespace sum_of_interiors_l952_95297

theorem sum_of_interiors (n : ℕ) (h : 180 * (n - 2) = 1620) : 180 * ((n + 3) - 2) = 2160 :=
by sorry

end sum_of_interiors_l952_95297


namespace initial_price_of_article_l952_95298

theorem initial_price_of_article (P : ℝ) (h : 0.4025 * P = 620) : P = 620 / 0.4025 :=
by
  sorry

end initial_price_of_article_l952_95298


namespace systematic_sampling_interval_l952_95281

theorem systematic_sampling_interval 
  (N : ℕ) (n : ℕ) (hN : N = 630) (hn : n = 45) :
  N / n = 14 :=
by {
  sorry
}

end systematic_sampling_interval_l952_95281


namespace standard_eq_circle_C_equation_line_AB_l952_95229

-- Define the center of circle C and the line l
def center_C : ℝ × ℝ := (2, 1)
def line_l (x y : ℝ) : Prop := x = 3

-- Define the standard equation of circle C
def eq_circle_C (x y : ℝ) : Prop :=
  (x - center_C.1)^2 + (y - center_C.2)^2 = 1

-- Equation of circle O
def eq_circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the condition that circle C intersects with circle O at points A and B
def intersects (x y : ℝ) : Prop :=
  eq_circle_C x y ∧ eq_circle_O x y

-- Define the equation of line AB in general form
def eq_line_AB (x y : ℝ) : Prop :=
  2 * x + y - 4 = 0

-- Prove the standard equation of circle C is (x-2)^2 + (y-1)^2 = 1
theorem standard_eq_circle_C:
  eq_circle_C x y ↔ (x - 2)^2 + (y - 1)^2 = 1 :=
sorry

-- Prove that the equation of line AB is 2x + y - 4 = 0, given the intersection points A and B
theorem equation_line_AB (x y : ℝ) (h : intersects x y) :
  eq_line_AB x y :=
sorry

end standard_eq_circle_C_equation_line_AB_l952_95229


namespace fractional_equation_no_solution_l952_95208

theorem fractional_equation_no_solution (a : ℝ) :
  (¬ ∃ x, x ≠ 1 ∧ x ≠ 0 ∧ ((x - a) / (x - 1) - 3 / x = 1)) → (a = 1 ∨ a = -2) :=
by
  sorry

end fractional_equation_no_solution_l952_95208


namespace max_ab_condition_l952_95233

-- Define the circles and the tangency condition
def circle1 (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + (p.2 + 2)^2 = 4}
def circle2 (b : ℝ) : Set (ℝ × ℝ) := {p | (p.1 + b)^2 + (p.2 + 2)^2 = 1}
def internally_tangent (a b : ℝ) : Prop := (a + b) ^ 2 = 1

-- Define the maximum value condition
def max_ab (a b : ℝ) : ℝ := a * b

-- Main theorem
theorem max_ab_condition {a b : ℝ} (h_tangent : internally_tangent a b) : max_ab a b ≤ 1 / 4 :=
by
  -- Proof steps are not necessary, so we use sorry to end the proof.
  sorry

end max_ab_condition_l952_95233


namespace tetrahedron_volume_l952_95289

noncomputable def volume_of_tetrahedron (S1 S2 a α : ℝ) : ℝ :=
  (2 * S1 * S2 * Real.sin α) / (3 * a)

theorem tetrahedron_volume (S1 S2 a α : ℝ) :
  a > 0 → S1 > 0 → S2 > 0 → α ≥ 0 → α ≤ Real.pi → volume_of_tetrahedron S1 S2 a α =
  (2 * S1 * S2 * Real.sin α) / (3 * a) := 
by
  intros
  -- The proof is omitted here.
  sorry

end tetrahedron_volume_l952_95289


namespace roller_coaster_ticket_cost_l952_95269

def ferrisWheelCost : ℕ := 6
def logRideCost : ℕ := 7
def initialTickets : ℕ := 2
def ticketsToBuy : ℕ := 16

def totalTicketsNeeded : ℕ := initialTickets + ticketsToBuy
def ridesCost : ℕ := ferrisWheelCost + logRideCost
def rollerCoasterCost : ℕ := totalTicketsNeeded - ridesCost

theorem roller_coaster_ticket_cost :
  rollerCoasterCost = 5 :=
by
  sorry

end roller_coaster_ticket_cost_l952_95269


namespace trig_expression_value_l952_95235

open Real

theorem trig_expression_value (x : ℝ) (h : tan (π - x) = -2) : 
  4 * sin x ^ 2 - 3 * sin x * cos x - 5 * cos x ^ 2 = 1 := 
sorry

end trig_expression_value_l952_95235


namespace solve_for_x_l952_95256

def f (x : ℝ) : ℝ := 2 * x - 3

theorem solve_for_x : ∃ (x : ℝ), 2 * (f x) - 11 = f (x - 2) :=
by
  use 5
  have h1 : f 5 = 2 * 5 - 3 := rfl
  have h2 : f (5 - 2) = 2 * (5 - 2) - 3 := rfl
  simp [f] at *
  exact sorry

end solve_for_x_l952_95256


namespace C_completion_time_l952_95267

noncomputable def racer_time (v_C : ℝ) : ℝ := 100 / v_C

theorem C_completion_time
  (v_A v_B v_C : ℝ)
  (h1 : 100 / v_A = 10)
  (h2 : 85 / v_B = 10)
  (h3 : 90 / v_C = 100 / v_B) :
  racer_time v_C = 13.07 :=
by
  sorry

end C_completion_time_l952_95267


namespace initial_amount_of_money_l952_95234

-- Define the costs and purchased quantities
def cost_tshirt : ℕ := 8
def cost_keychain_set : ℕ := 2
def cost_bag : ℕ := 10
def tshirts_bought : ℕ := 2
def bags_bought : ℕ := 2
def keychains_bought : ℕ := 21

-- Define derived quantities
def sets_of_keychains_bought : ℕ := keychains_bought / 3

-- Define the total costs
def total_cost_tshirts : ℕ := tshirts_bought * cost_tshirt
def total_cost_bags : ℕ := bags_bought * cost_bag
def total_cost_keychains : ℕ := sets_of_keychains_bought * cost_keychain_set

-- Define the initial amount of money
def total_initial_amount : ℕ := total_cost_tshirts + total_cost_bags + total_cost_keychains

-- The theorem proving the initial amount Timothy had
theorem initial_amount_of_money : total_initial_amount = 50 := by
  -- The proof is not required, so we use sorry to skip it
  sorry

end initial_amount_of_money_l952_95234


namespace required_butter_l952_95272

-- Define the given conditions
variables (butter sugar : ℕ)
def recipe_butter : ℕ := 25
def recipe_sugar : ℕ := 125
def used_sugar : ℕ := 1000

-- State the theorem
theorem required_butter (h1 : butter = recipe_butter) (h2 : sugar = recipe_sugar) :
  (used_sugar * recipe_butter) / recipe_sugar = 200 := 
by 
  sorry

end required_butter_l952_95272


namespace incorrect_sum_Sn_l952_95276

-- Define the geometric sequence sum formula
def Sn (a r : ℕ) (n : ℕ) : ℕ := a * (1 - r^n) / (1 - r)

-- Define the given values
def S1 : ℕ := 8
def S2 : ℕ := 20
def S3 : ℕ := 36
def S4 : ℕ := 65

-- The main proof statement
theorem incorrect_sum_Sn : 
  ∃ (a r : ℕ), 
  a = 8 ∧ 
  Sn a r 1 = S1 ∧ 
  Sn a r 2 = S2 ∧ 
  Sn a r 3 ≠ S3 ∧ 
  Sn a r 4 = S4 :=
by sorry

end incorrect_sum_Sn_l952_95276


namespace fraction_multiplication_l952_95237

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l952_95237


namespace rob_final_value_in_euros_l952_95212

noncomputable def initial_value_in_usd : ℝ := 
  (7 * 0.25) + (3 * 0.10) + (5 * 0.05) + (12 * 0.01) + (3 * 0.50) + (2 * 1.00)

noncomputable def value_after_losing_coins : ℝ := 
  (6 * 0.25) + (2 * 0.10) + (4 * 0.05) + (11 * 0.01) + (2 * 0.50) + (1 * 1.00)

noncomputable def value_after_first_exchange : ℝ :=
  (6 * 0.25) + (4 * 0.10) + (1 * 0.05) + (11 * 0.01) + (2 * 0.50) + (1 * 1.00)

noncomputable def value_after_second_exchange : ℝ :=
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (11 * 0.01) + (1 * 0.50) + (1 * 1.00)

noncomputable def value_after_third_exchange : ℝ :=
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (61 * 0.01) + (1 * 0.50)

noncomputable def final_value_in_usd : ℝ := 
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (61 * 0.01) + (1 * 0.50)

noncomputable def exchange_rate_usd_to_eur : ℝ := 0.85

noncomputable def final_value_in_eur : ℝ :=
  final_value_in_usd * exchange_rate_usd_to_eur

theorem rob_final_value_in_euros : final_value_in_eur = 2.9835 := by
  sorry

end rob_final_value_in_euros_l952_95212


namespace possible_slopes_l952_95253

theorem possible_slopes (k : ℝ) (H_pos : k > 0) :
  (∃ x1 x2 : ℤ, (x1 + x2 : ℝ) = k ∧ (x1 * x2 : ℝ) = -2020) ↔ 
  k = 81 ∨ k = 192 ∨ k = 399 ∨ k = 501 ∨ k = 1008 ∨ k = 2019 := 
by
  sorry

end possible_slopes_l952_95253


namespace man_speed_is_correct_l952_95264

noncomputable def speed_of_man (train_speed_kmh : ℝ) (train_length_m : ℝ) (time_to_pass_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed_ms := train_length_m / time_to_pass_s
  let man_speed_ms := relative_speed_ms - train_speed_ms
  man_speed_ms * 3600 / 1000

theorem man_speed_is_correct : 
  speed_of_man 60 110 5.999520038396929 = 6.0024 := 
by
  sorry

end man_speed_is_correct_l952_95264


namespace total_yards_run_l952_95258

theorem total_yards_run (Malik_yards_per_game : ℕ) (Josiah_yards_per_game : ℕ) (Darnell_yards_per_game : ℕ) (games : ℕ) 
  (hM : Malik_yards_per_game = 18) (hJ : Josiah_yards_per_game = 22) (hD : Darnell_yards_per_game = 11) (hG : games = 4) : 
  Malik_yards_per_game * games + Josiah_yards_per_game * games + Darnell_yards_per_game * games = 204 := by
  sorry

end total_yards_run_l952_95258


namespace area_fraction_above_line_l952_95273

-- Define the points of the rectangle
def A := (2,0)
def B := (7,0)
def C := (7,4)
def D := (2,4)

-- Define the points used for the line
def P := (2,1)
def Q := (7,3)

-- The area of the rectangle
def rect_area := (7 - 2) * 4

-- The fraction of the area of the rectangle above the line
theorem area_fraction_above_line : 
  ∀ A B C D P Q, 
    A = (2,0) → B = (7,0) → C = (7,4) → D = (2,4) →
    P = (2,1) → Q = (7,3) →
    (rect_area = 20) → 1 - ((1/2) * 5 * 2 / 20) = 3 / 4 :=
by
  intros A B C D P Q
  intros hA hB hC hD hP hQ h_area
  sorry

end area_fraction_above_line_l952_95273


namespace harrison_croissant_expenditure_l952_95231

-- Define the conditions
def cost_regular_croissant : ℝ := 3.50
def cost_almond_croissant : ℝ := 5.50
def weeks_in_year : ℕ := 52

-- Define the total cost of croissants in a year
def total_cost (cost_regular cost_almond : ℝ) (weeks : ℕ) : ℝ :=
  (weeks * cost_regular) + (weeks * cost_almond)

-- State the proof problem
theorem harrison_croissant_expenditure :
  total_cost cost_regular_croissant cost_almond_croissant weeks_in_year = 468.00 :=
by
  sorry

end harrison_croissant_expenditure_l952_95231


namespace variance_of_planted_trees_l952_95263

def number_of_groups := 10

def planted_trees : List ℕ := [5, 5, 5, 6, 6, 6, 6, 7, 7, 7]

noncomputable def mean (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / (xs.length : ℚ)

noncomputable def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m) ^ 2)).sum / (xs.length : ℚ)

theorem variance_of_planted_trees :
  variance planted_trees = 0.6 := sorry

end variance_of_planted_trees_l952_95263


namespace range_of_x_in_function_l952_95222

theorem range_of_x_in_function (x : ℝ) (h : x ≠ 8) : true := sorry

end range_of_x_in_function_l952_95222


namespace third_year_award_count_l952_95251

-- Define the variables and conditions
variables (x x1 x2 x3 x4 x5 : ℕ)

-- The conditions and definition for the problem
def conditions : Prop :=
  (x1 = x) ∧
  (x5 = 3 * x) ∧
  (x1 < x2) ∧
  (x2 < x3) ∧
  (x3 < x4) ∧
  (x4 < x5) ∧
  (x1 + x2 + x3 + x4 + x5 = 27)

-- The theorem statement
theorem third_year_award_count (h : conditions x x1 x2 x3 x4 x5) : x3 = 5 :=
sorry

end third_year_award_count_l952_95251


namespace equations_of_motion_l952_95205

-- Initial conditions and setup
def omega : ℝ := 10
def OA : ℝ := 90
def AB : ℝ := 90
def AM : ℝ := 45

-- Questions:
-- 1. Equations of motion for point M
-- 2. Equation of the trajectory of point M
-- 3. Velocity of point M

theorem equations_of_motion (t : ℝ) :
  let xM := 45 * (1 + Real.cos (omega * t))
  let yM := 45 * Real.sin (omega * t)
  xM = 45 * (1 + Real.cos (omega * t)) ∧
  yM = 45 * Real.sin (omega * t) ∧
  ((yM / 45) ^ 2 + ((xM - 45) / 45) ^ 2 = 1) ∧
  let vMx := -450 * Real.sin (omega * t)
  let vMy := 450 * Real.cos (omega * t)
  (vMx = -450 * Real.sin (omega * t)) ∧
  (vMy = 450 * Real.cos (omega * t)) :=
by
  sorry

end equations_of_motion_l952_95205


namespace difference_in_interest_rates_l952_95242

-- Definitions
def Principal : ℝ := 2300
def Time : ℝ := 3
def ExtraInterest : ℝ := 69

-- The difference in rates
theorem difference_in_interest_rates (R dR : ℝ) (h : (Principal * (R + dR) * Time) / 100 =
    (Principal * R * Time) / 100 + ExtraInterest) : dR = 1 :=
  sorry

end difference_in_interest_rates_l952_95242


namespace frac_x_y_value_l952_95288

theorem frac_x_y_value (x y : ℝ) (h1 : 3 < (2 * x - y) / (x + 2 * y))
(h2 : (2 * x - y) / (x + 2 * y) < 7) (h3 : ∃ (t : ℤ), x = t * y) : x / y = -4 := by
  sorry

end frac_x_y_value_l952_95288


namespace munchausen_forest_l952_95293

theorem munchausen_forest (E B : ℕ) (h : B = 10 * E) : B > E := by sorry

end munchausen_forest_l952_95293


namespace intersection_M_N_l952_95257

-- Definitions for sets M and N
def set_M : Set ℝ := {x | abs x < 1}
def set_N : Set ℝ := {x | x^2 <= x}

-- The theorem stating the intersection of M and N
theorem intersection_M_N : {x : ℝ | x ∈ set_M ∧ x ∈ set_N} = {x : ℝ | 0 <= x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l952_95257


namespace boy_reaches_early_l952_95247

-- Given conditions
def usual_time : ℚ := 42
def rate_multiplier : ℚ := 7 / 6

-- Derived variables
def new_time : ℚ := (6 / 7) * usual_time
def early_time : ℚ := usual_time - new_time

-- The statement to prove
theorem boy_reaches_early : early_time = 6 := by
  sorry

end boy_reaches_early_l952_95247


namespace range_of_a_l952_95268

theorem range_of_a (hP : ¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) : 0 < a ∧ a < 1 :=
sorry

end range_of_a_l952_95268


namespace range_of_a_opposite_sides_l952_95286

theorem range_of_a_opposite_sides (a : ℝ) :
  (3 * (-2) - 2 * 1 - a) * (3 * 1 - 2 * 1 - a) < 0 ↔ -8 < a ∧ a < 1 := by
  sorry

end range_of_a_opposite_sides_l952_95286


namespace seventh_diagram_shaded_triangles_l952_95204

-- Define the factorial function
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- The main theorem stating the relationship between the number of shaded sub-triangles and the factorial/Fibonacci sequence
theorem seventh_diagram_shaded_triangles :
  ∃ k : ℕ, (k : ℚ) = (fib 7 : ℚ) / (fact 7 : ℚ) ∧ k = 13 := sorry

end seventh_diagram_shaded_triangles_l952_95204


namespace secretary_longest_time_l952_95221

theorem secretary_longest_time (h_ratio : ∃ x : ℕ, ∃ y : ℕ, ∃ z : ℕ, y = 2 * x ∧ z = 3 * x ∧ (5 * x = 40)) :
  5 * x = 40 := sorry

end secretary_longest_time_l952_95221


namespace eggs_left_over_l952_95262

def david_eggs : ℕ := 44
def elizabeth_eggs : ℕ := 52
def fatima_eggs : ℕ := 23
def carton_size : ℕ := 12

theorem eggs_left_over : 
  (david_eggs + elizabeth_eggs + fatima_eggs) % carton_size = 11 :=
by sorry

end eggs_left_over_l952_95262


namespace non_negative_integers_abs_less_than_3_l952_95200

theorem non_negative_integers_abs_less_than_3 :
  { x : ℕ | x < 3 } = {0, 1, 2} :=
by
  sorry

end non_negative_integers_abs_less_than_3_l952_95200


namespace find_c_l952_95201

theorem find_c (c d : ℝ) (h : ∀ x : ℝ, 9 * x^2 - 24 * x + c = (3 * x + d)^2) : c = 16 :=
sorry

end find_c_l952_95201


namespace Mr_Tom_invested_in_fund_X_l952_95255

theorem Mr_Tom_invested_in_fund_X (a b : ℝ) (h1 : a + b = 100000) (h2 : 0.17 * b = 0.23 * a + 200) : a = 42000 := 
by
  sorry

end Mr_Tom_invested_in_fund_X_l952_95255


namespace sum_of_squares_multiple_of_five_sum_of_consecutive_squares_multiple_of_five_l952_95244

theorem sum_of_squares_multiple_of_five :
  ( (-1)^2 + 0^2 + 1^2 + 2^2 + 3^2 ) % 5 = 0 :=
by
  sorry

theorem sum_of_consecutive_squares_multiple_of_five 
  (n : ℤ) :
  ((n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2) % 5 = 0 :=
by
  sorry

end sum_of_squares_multiple_of_five_sum_of_consecutive_squares_multiple_of_five_l952_95244


namespace valid_vector_parameterizations_of_line_l952_95206

theorem valid_vector_parameterizations_of_line (t : ℝ) :
  (∃ t : ℝ, (∃ x y : ℝ, (x = 1 + t ∧ y = t ∧ y = x - 1)) ∨
            (∃ x y : ℝ, (x = -t ∧ y = -1 - t ∧ y = x - 1)) ∨
            (∃ x y : ℝ, (x = 2 + 0.5 * t ∧ y = 1 + 0.5 * t ∧ y = x - 1))) :=
by sorry

end valid_vector_parameterizations_of_line_l952_95206


namespace polynomial_has_one_positive_real_solution_l952_95226

-- Define the polynomial
def f (x : ℝ) : ℝ := x ^ 10 + 4 * x ^ 9 + 7 * x ^ 8 + 2023 * x ^ 7 - 2024 * x ^ 6

-- The proof problem statement
theorem polynomial_has_one_positive_real_solution :
  ∃! x : ℝ, 0 < x ∧ f x = 0 := by
  sorry

end polynomial_has_one_positive_real_solution_l952_95226


namespace union_A_B_l952_95225

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {y | ∃ x : ℝ, y = sin x}

theorem union_A_B : A ∪ B = Ico (-1 : ℝ) 2 := by
  sorry

end union_A_B_l952_95225


namespace no_three_digit_number_l952_95278

theorem no_three_digit_number :
  ¬ ∃ (a b c : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (100 * a + 10 * b + c = 3 * (100 * b + 10 * c + a)) :=
by
  sorry

end no_three_digit_number_l952_95278


namespace abs_neg_2023_l952_95252

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 :=
by
  sorry

end abs_neg_2023_l952_95252


namespace lcm_gcd_product_eq_product_12_15_l952_95202

theorem lcm_gcd_product_eq_product_12_15 :
  lcm 12 15 * gcd 12 15 = 12 * 15 :=
sorry

end lcm_gcd_product_eq_product_12_15_l952_95202


namespace average_visitors_per_day_l952_95227

theorem average_visitors_per_day (avg_sunday : ℕ) (avg_other_day : ℕ) (days_in_month : ℕ) (starts_on_sunday : Bool) :
  avg_sunday = 570 →
  avg_other_day = 240 →
  days_in_month = 30 →
  starts_on_sunday = true →
  (5 * avg_sunday + 25 * avg_other_day) / days_in_month = 295 :=
by
  intros
  sorry

end average_visitors_per_day_l952_95227


namespace original_selling_price_l952_95219

theorem original_selling_price (P : ℝ) (h : 0.7 * P = 560) : P = 800 :=
by
  sorry

end original_selling_price_l952_95219


namespace yeast_population_at_130pm_l952_95213

noncomputable def yeast_population (initial_population : ℕ) (time_increments : ℕ) (growth_factor : ℕ) : ℕ :=
  initial_population * growth_factor ^ time_increments

theorem yeast_population_at_130pm : yeast_population 30 3 3 = 810 :=
by
  sorry

end yeast_population_at_130pm_l952_95213


namespace parabola_addition_l952_95243

def f (a b c x : ℝ) : ℝ := a * x^2 - b * (x + 3) + c
def g (a b c x : ℝ) : ℝ := a * x^2 + b * (x - 4) + c

theorem parabola_addition (a b c x : ℝ) : 
  (f a b c x + g a b c x) = (2 * a * x^2 + 2 * c - 7 * b) :=
by
  sorry

end parabola_addition_l952_95243


namespace fewer_servings_per_day_l952_95285

theorem fewer_servings_per_day :
  ∀ (daily_consumption servings_old servings_new: ℕ),
    daily_consumption = 64 →
    servings_old = 8 →
    servings_new = 16 →
    (daily_consumption / servings_old) - (daily_consumption / servings_new) = 4 :=
by
  intros daily_consumption servings_old servings_new h1 h2 h3
  sorry

end fewer_servings_per_day_l952_95285


namespace find_angle_BCD_l952_95249

-- Defining the given conditions in the problem
def angleA : ℝ := 100
def angleD : ℝ := 120
def angleE : ℝ := 80
def angleABC : ℝ := 140
def pentagonInteriorAngleSum : ℝ := 540

-- Statement: Prove that the measure of ∠ BCD is 100 degrees given the conditions
theorem find_angle_BCD (h1 : angleA = 100) (h2 : angleD = 120) (h3 : angleE = 80) 
                       (h4 : angleABC = 140) (h5 : pentagonInteriorAngleSum = 540) :
    (angleBCD : ℝ) = 100 :=
sorry

end find_angle_BCD_l952_95249


namespace internet_bill_is_100_l952_95294

theorem internet_bill_is_100 (initial_amount rent paycheck electricity_bill phone_bill final_amount internet_bill : ℝ)
  (h1 : initial_amount = 800)
  (h2 : rent = 450)
  (h3 : paycheck = 1500)
  (h4 : electricity_bill = 117)
  (h5 : phone_bill = 70)
  (h6 : final_amount = 1563)
  (h7 : initial_amount - rent + paycheck - electricity_bill - internet_bill - phone_bill = final_amount) :
  internet_bill = 100 :=
by
  sorry

end internet_bill_is_100_l952_95294


namespace ratio_area_of_circle_to_triangle_l952_95228

theorem ratio_area_of_circle_to_triangle
  (h r b : ℝ)
  (h_triangle : ∃ a, a = b + r ∧ a^2 + b^2 = h^2) :
  (∃ A s : ℝ, s = b + (r + h) / 2 ∧ A = r * s ∧ (∃ circle_area triangle_area : ℝ, circle_area = π * r^2 ∧ triangle_area = 2 * A ∧ circle_area / triangle_area = 2 * π * r / (2 * b + r + h))) :=
by
  sorry

end ratio_area_of_circle_to_triangle_l952_95228


namespace sum_of_ages_twins_l952_95292

-- Define that Evan has two older twin sisters and their ages are such that the product of all three ages is 162
def twin_sisters_ages (a : ℕ) (b : ℕ) (c : ℕ) : Prop :=
  a * b * c = 162

-- Given the above definition, we need to prove the sum of these ages is 20
theorem sum_of_ages_twins (a b c : ℕ) (h : twin_sisters_ages a b c) (ha : b = c) : a + b + c = 20 :=
by 
  sorry

end sum_of_ages_twins_l952_95292


namespace base7_to_base10_321_is_162_l952_95203

-- Define the conversion process from a base-7 number to base-10
def convert_base7_to_base10 (n: ℕ) : ℕ :=
  3 * 7^2 + 2 * 7^1 + 1 * 7^0

theorem base7_to_base10_321_is_162 :
  convert_base7_to_base10 321 = 162 :=
by
  sorry

end base7_to_base10_321_is_162_l952_95203


namespace floor_div_add_floor_div_succ_eq_l952_95240

theorem floor_div_add_floor_div_succ_eq (n : ℤ) : 
  (⌊(n : ℝ)/2⌋ + ⌊(n + 1 : ℝ)/2⌋ : ℤ) = n := 
sorry

end floor_div_add_floor_div_succ_eq_l952_95240


namespace field_trip_fraction_l952_95287

theorem field_trip_fraction (b g : ℕ) (hb : g = b)
  (girls_trip_fraction : ℚ := 4/5)
  (boys_trip_fraction : ℚ := 3/4) :
  girls_trip_fraction * g / (girls_trip_fraction * g + boys_trip_fraction * b) = 16 / 31 :=
by {
  sorry
}

end field_trip_fraction_l952_95287


namespace smallest_xy_l952_95245

theorem smallest_xy :
  ∃ (x y : ℕ), (0 < x) ∧ (0 < y) ∧ (1 / x + 1 / (3 * y) = 1 / 6) ∧ (∀ (x' y' : ℕ), (0 < x') ∧ (0 < y') ∧ (1 / x' + 1 / (3 * y') = 1 / 6) → x' * y' ≥ x * y) ∧ x * y = 48 :=
sorry

end smallest_xy_l952_95245


namespace smallest_integer_k_distinct_real_roots_l952_95209

theorem smallest_integer_k_distinct_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, x^2 - x + 2 - k = 0 → x ≠ 0) ∧ k = 2 :=
by
  sorry

end smallest_integer_k_distinct_real_roots_l952_95209


namespace factor_of_increase_l952_95217

noncomputable def sum_arithmetic_progression (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem factor_of_increase (a1 d n : ℕ) (h1 : a1 > 0) (h2 : (sum_arithmetic_progression a1 (3 * d) n = 2 * sum_arithmetic_progression a1 d n)) :
  sum_arithmetic_progression a1 (4 * d) n = (5 / 2) * sum_arithmetic_progression a1 d n :=
sorry

end factor_of_increase_l952_95217


namespace original_price_of_dish_l952_95265

variable (P : ℝ)

def john_paid (P : ℝ) : ℝ := 0.9 * P + 0.15 * P
def jane_paid (P : ℝ) : ℝ := 0.9 * P + 0.135 * P

theorem original_price_of_dish (h : john_paid P = jane_paid P + 1.26) : P = 84 := by
  sorry

end original_price_of_dish_l952_95265


namespace youngest_age_is_20_l952_95207

-- Definitions of the ages
def siblings_ages (y : ℕ) : List ℕ := [y, y+2, y+7, y+11]

-- Condition of the problem: average age is 25
def average_age_25 (y : ℕ) : Prop := (siblings_ages y).sum = 100

-- The statement to be proven
theorem youngest_age_is_20 (y : ℕ) (h : average_age_25 y) : y = 20 :=
  sorry

end youngest_age_is_20_l952_95207


namespace find_y_value_l952_95279

theorem find_y_value (k : ℝ) (h1 : ∀ (x : ℝ), y = k * x) 
(h2 : y = 4 ∧ x = 2) : 
(∀ (x : ℝ), x = -2 → y = -4) := 
by 
  sorry

end find_y_value_l952_95279


namespace meat_needed_l952_95250

theorem meat_needed (meat_per_hamburger : ℚ) (h_meat : meat_per_hamburger = (3 : ℚ) / 8) : 
  (24 * meat_per_hamburger) = 9 :=
by
  sorry

end meat_needed_l952_95250


namespace solve_system_of_equations_solve_system_of_inequalities_l952_95296

-- For the system of equations
theorem solve_system_of_equations (x y : ℝ) (h1 : 3 * x + 4 * y = 2) (h2 : 2 * x - y = 5) : 
    x = 2 ∧ y = -1 :=
sorry

-- For the system of inequalities
theorem solve_system_of_inequalities (x : ℝ) 
    (h1 : x - 3 * (x - 1) < 7) 
    (h2 : x - 2 ≤ (2 * x - 3) / 3) :
    -2 < x ∧ x ≤ 3 :=
sorry

end solve_system_of_equations_solve_system_of_inequalities_l952_95296


namespace smallest_x_for_M_squared_l952_95210

theorem smallest_x_for_M_squared (M x : ℤ) (h1 : 540 = 2^2 * 3^3 * 5) (h2 : 540 * x = M^2) (h3 : x > 0) : x = 15 :=
sorry

end smallest_x_for_M_squared_l952_95210


namespace find_num_carbon_atoms_l952_95211

def num_carbon_atoms (nH nO mH mC mO mol_weight : ℕ) : ℕ :=
  (mol_weight - (nH * mH + nO * mO)) / mC

theorem find_num_carbon_atoms :
  num_carbon_atoms 2 3 1 12 16 62 = 1 :=
by
  -- The proof is skipped
  sorry

end find_num_carbon_atoms_l952_95211


namespace sum_of_reciprocals_l952_95259

noncomputable def reciprocal_sum (x y : ℝ) : ℝ :=
  (1 / x) + (1 / y)

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) :
  reciprocal_sum x y = 8 / 75 :=
by
  unfold reciprocal_sum
  -- Intermediate steps would go here, but we'll use sorry to denote the proof is omitted.
  sorry

end sum_of_reciprocals_l952_95259


namespace coconut_tree_difference_l952_95266

-- Define the known quantities
def mango_trees : ℕ := 60
def total_trees : ℕ := 85
def half_mango_trees : ℕ := 30 -- half of 60
def coconut_trees : ℕ := 25 -- 85 - 60

-- Define the proof statement
theorem coconut_tree_difference : (half_mango_trees - coconut_trees) = 5 := by
  -- The proof steps are given
  sorry

end coconut_tree_difference_l952_95266


namespace necessary_but_not_sufficient_condition_is_purely_imaginary_l952_95236

noncomputable def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ (b : ℝ), z = ⟨0, b⟩

theorem necessary_but_not_sufficient_condition_is_purely_imaginary (a b : ℝ) (h_imaginary : is_purely_imaginary (⟨a, b⟩)) : 
  (a = 0) ∧ (b ≠ 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_is_purely_imaginary_l952_95236


namespace solution_set_of_tan_eq_two_l952_95271

open Real

theorem solution_set_of_tan_eq_two :
  {x | ∃ k : ℤ, x = k * π + (-1 : ℤ) ^ k * arctan 2} = {x | tan x = 2} :=
by
  sorry

end solution_set_of_tan_eq_two_l952_95271


namespace max_take_home_pay_at_5000_dollars_l952_95239

noncomputable def income_tax (x : ℕ) : ℕ :=
  if x ≤ 5000 then x * 5 / 100
  else 250 + 10 * ((x - 5000 / 1000) - 5) ^ 2

noncomputable def take_home_pay (y : ℕ) : ℕ :=
  y - income_tax y

theorem max_take_home_pay_at_5000_dollars : ∀ y : ℕ, take_home_pay y ≤ take_home_pay 5000 := by
  sorry

end max_take_home_pay_at_5000_dollars_l952_95239


namespace time_to_cross_approx_l952_95299

-- Define train length, tunnel length, speed in km/hr, conversion factors, and the final equation
def length_of_train : ℕ := 415
def length_of_tunnel : ℕ := 285
def speed_in_kmph : ℕ := 63
def km_to_m : ℕ := 1000
def hr_to_sec : ℕ := 3600

-- Convert speed to m/s
def speed_in_mps : ℚ := (speed_in_kmph * km_to_m) / hr_to_sec

-- Calculate total distance
def total_distance : ℕ := length_of_train + length_of_tunnel

-- Calculate the time to cross the tunnel in seconds
def time_to_cross : ℚ := total_distance / speed_in_mps

theorem time_to_cross_approx : abs (time_to_cross - 40) < 0.1 :=
sorry

end time_to_cross_approx_l952_95299


namespace find_discount_l952_95282

noncomputable def children_ticket_cost : ℝ := 4.25
noncomputable def adult_ticket_cost : ℝ := children_ticket_cost + 3.25
noncomputable def total_cost_without_discount : ℝ := 2 * adult_ticket_cost + 4 * children_ticket_cost
noncomputable def total_spent : ℝ := 30
noncomputable def discount_received : ℝ := total_cost_without_discount - total_spent

theorem find_discount :
  discount_received = 2 := by
  sorry

end find_discount_l952_95282


namespace no_odd_integers_satisfy_equation_l952_95232

theorem no_odd_integers_satisfy_equation :
  ¬ ∃ (x y z : ℤ), (x % 2 ≠ 0) ∧ (y % 2 ≠ 0) ∧ (z % 2 ≠ 0) ∧ 
  (x + y)^2 + (x + z)^2 = (y + z)^2 :=
by
  sorry

end no_odd_integers_satisfy_equation_l952_95232
