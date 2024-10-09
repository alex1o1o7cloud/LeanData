import Mathlib

namespace simple_interest_rate_l1221_122155

theorem simple_interest_rate (P : ℝ) (T : ℝ) (r : ℝ) (h1 : T = 10) (h2 : (3 / 5) * P = (P * r * T) / 100) : r = 6 := by
  sorry

end simple_interest_rate_l1221_122155


namespace ball_falls_total_distance_l1221_122153

noncomputable def total_distance : ℕ → ℤ → ℤ → ℤ
| 0, a, _ => 0
| (n+1), a, d => a + total_distance n (a + d) d

theorem ball_falls_total_distance :
  total_distance 5 30 (-6) = 90 :=
by
  sorry

end ball_falls_total_distance_l1221_122153


namespace find_three_digit_perfect_square_l1221_122138

noncomputable def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n % 100) / 10) * (n % 10)

theorem find_three_digit_perfect_square :
  ∃ (n H : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (n = H * H) ∧ (digit_product n = H - 1) :=
by {
  sorry
}

end find_three_digit_perfect_square_l1221_122138


namespace profit_percentage_is_25_percent_l1221_122142

noncomputable def costPrice : ℝ := 47.50
noncomputable def markedPrice : ℝ := 64.54
noncomputable def discountRate : ℝ := 0.08

noncomputable def discountAmount : ℝ := discountRate * markedPrice
noncomputable def sellingPrice : ℝ := markedPrice - discountAmount
noncomputable def profit : ℝ := sellingPrice - costPrice
noncomputable def profitPercentage : ℝ := (profit / costPrice) * 100

theorem profit_percentage_is_25_percent :
  profitPercentage = 25 := by
  sorry

end profit_percentage_is_25_percent_l1221_122142


namespace negation_of_existence_l1221_122193

variable (x : ℝ)

theorem negation_of_existence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2 * x₀ - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by sorry

end negation_of_existence_l1221_122193


namespace series_result_l1221_122161

noncomputable def series_sum (u : ℕ → ℚ) (s : ℚ) : Prop :=
  ∑' n, u n = s

def nth_term (n : ℕ) : ℚ := (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem series_result : series_sum nth_term (1 / 200) := by
  sorry

end series_result_l1221_122161


namespace clare_remaining_money_l1221_122154

-- Definitions based on conditions
def clare_initial_money : ℕ := 47
def bread_quantity : ℕ := 4
def milk_quantity : ℕ := 2
def bread_cost : ℕ := 2
def milk_cost : ℕ := 2

-- The goal is to prove that Clare has $35 left after her purchases.
theorem clare_remaining_money : 
  clare_initial_money - (bread_quantity * bread_cost + milk_quantity * milk_cost) = 35 := 
sorry

end clare_remaining_money_l1221_122154


namespace daily_savings_amount_l1221_122191

theorem daily_savings_amount (total_savings : ℕ) (days : ℕ) (daily_savings : ℕ)
  (h1 : total_savings = 12410)
  (h2 : days = 365)
  (h3 : total_savings = daily_savings * days) :
  daily_savings = 34 :=
sorry

end daily_savings_amount_l1221_122191


namespace missed_bus_time_l1221_122192

theorem missed_bus_time (T: ℕ) (speed_ratio: ℚ) (T_slow: ℕ) (missed_time: ℕ) : 
  T = 16 → speed_ratio = 4/5 → T_slow = (5/4) * T → missed_time = T_slow - T → missed_time = 4 :=
by
  sorry

end missed_bus_time_l1221_122192


namespace quadratic_inequality_solution_set_l1221_122162

theorem quadratic_inequality_solution_set :
  {x : ℝ | - x ^ 2 + 4 * x + 12 > 0} = {x : ℝ | -2 < x ∧ x < 6} :=
sorry

end quadratic_inequality_solution_set_l1221_122162


namespace same_different_color_ways_equal_l1221_122136

-- Definitions based on conditions in the problem
def num_black : ℕ := 15
def num_white : ℕ := 10

def same_color_ways : ℕ :=
  Nat.choose num_black 2 + Nat.choose num_white 2

def different_color_ways : ℕ :=
  num_black * num_white

-- The proof statement
theorem same_different_color_ways_equal : same_color_ways = different_color_ways :=
by
  sorry

end same_different_color_ways_equal_l1221_122136


namespace min_a_plus_b_l1221_122113

open Real

theorem min_a_plus_b (a b : ℕ) (h_a_pos : a > 1) (h_ab : ∃ a b, (a^2 * b - 1) / (a * b^2) = 1 / 2024) :
  a + b = 228 :=
sorry

end min_a_plus_b_l1221_122113


namespace burpees_percentage_contribution_l1221_122134

theorem burpees_percentage_contribution :
  let total_time : ℝ := 20
  let jumping_jacks : ℝ := 30
  let pushups : ℝ := 22
  let situps : ℝ := 45
  let burpees : ℝ := 15
  let lunges : ℝ := 25

  let jumping_jacks_rate := jumping_jacks / total_time
  let pushups_rate := pushups / total_time
  let situps_rate := situps / total_time
  let burpees_rate := burpees / total_time
  let lunges_rate := lunges / total_time

  let total_rate := jumping_jacks_rate + pushups_rate + situps_rate + burpees_rate + lunges_rate

  (burpees_rate / total_rate) * 100 = 10.95 :=
by
  sorry

end burpees_percentage_contribution_l1221_122134


namespace fraction_equality_l1221_122119

theorem fraction_equality : (18 / (5 * 107 + 3) = 18 / 538) := 
by
  -- Proof skipped
  sorry

end fraction_equality_l1221_122119


namespace find_m_if_divisible_by_11_l1221_122108

theorem find_m_if_divisible_by_11 : ∃ m : ℕ, m < 10 ∧ (734000000 + m*100000 + 8527) % 11 = 0 ↔ m = 6 :=
by {
    sorry
}

end find_m_if_divisible_by_11_l1221_122108


namespace solve_for_a_l1221_122163

theorem solve_for_a (a : ℝ) (h : 4 * a + 9 + (3 * a + 5) = 0) : a = -2 :=
by
  sorry

end solve_for_a_l1221_122163


namespace expand_product_l1221_122140

theorem expand_product (x : ℝ) (hx : x ≠ 0) : (3 / 7) * (7 / x - 5 * x ^ 3) = 3 / x - (15 / 7) * x ^ 3 :=
by
  sorry

end expand_product_l1221_122140


namespace find_ck_l1221_122128

theorem find_ck 
  (d r : ℕ)                -- d : common difference, r : common ratio
  (k : ℕ)                  -- k : integer such that certain conditions hold
  (hn2 : (k-2) > 0)        -- ensure (k-2) > 0
  (hk1 : (k+1) > 0)        -- ensure (k+1) > 0
  (h1 : 1 + (k-3) * d + r^(k-3) = 120) -- c_{k-1} = 120
  (h2 : 1 + k * d + r^k = 1200) -- c_{k+1} = 1200
  : (1 + (k-1) * d + r^(k-1)) = 263 := -- c_k = 263
sorry

end find_ck_l1221_122128


namespace bucket_initial_amount_l1221_122190

theorem bucket_initial_amount (A B : ℝ) 
  (h1 : A - 6 = (1 / 3) * (B + 6)) 
  (h2 : B - 6 = (1 / 2) * (A + 6)) : 
  A = 13.2 := 
sorry

end bucket_initial_amount_l1221_122190


namespace find_b_in_triangle_l1221_122151

theorem find_b_in_triangle (c : ℝ) (B C : ℝ) (h1 : c = Real.sqrt 3)
  (h2 : B = Real.pi / 4) (h3 : C = Real.pi / 3) : ∃ b : ℝ, b = Real.sqrt 2 :=
by
  sorry

end find_b_in_triangle_l1221_122151


namespace central_angle_correct_l1221_122111

-- Define arc length, radius, and central angle
variables (l r α : ℝ)

-- Given conditions
def arc_length := 3
def radius := 2

-- Theorem to prove
theorem central_angle_correct : (l = arc_length) → (r = radius) → (l = r * α) → α = 3 / 2 :=
by
  intros h1 h2 h3
  sorry

end central_angle_correct_l1221_122111


namespace unique_pos_neg_roots_of_poly_l1221_122175

noncomputable def poly : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^4 + Polynomial.C 5 * Polynomial.X^3 + Polynomial.C 15 * Polynomial.X - Polynomial.C 9

theorem unique_pos_neg_roots_of_poly : 
  (∃! x : ℝ, (0 < x) ∧ poly.eval x = 0) ∧ (∃! x : ℝ, (x < 0) ∧ poly.eval x = 0) :=
  sorry

end unique_pos_neg_roots_of_poly_l1221_122175


namespace quadratic_equation_in_x_l1221_122137

theorem quadratic_equation_in_x (m : ℤ) (h1 : abs m = 2) (h2 : m + 2 ≠ 0) : m = 2 :=
sorry

end quadratic_equation_in_x_l1221_122137


namespace tan_beta_value_l1221_122170

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = 1 / 3) (h2 : Real.tan (α + β) = 1 / 2) : Real.tan β = 1 / 7 :=
by
  sorry

end tan_beta_value_l1221_122170


namespace find_x_l1221_122164

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vectors_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, (u.1 * k = v.1) ∧ (u.2 * k = v.2)

theorem find_x :
  let a := (1, -2)
  let b := (3, -1)
  let c := (x, 4)
  vectors_parallel (vector_add a c) (vector_add b c) → x = 3 :=
by intros; sorry

end find_x_l1221_122164


namespace alpha_parallel_to_beta_l1221_122112

variables (a b : ℝ → ℝ → ℝ) (α β : ℝ → ℝ)

-- Definitions based on conditions
def are_distinct_lines : a ≠ b := sorry
def are_distinct_planes : α ≠ β := sorry

def line_parallel_to_plane (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ) : Prop := sorry -- Define parallel relation
def line_perpendicular_to_plane (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ) : Prop := sorry -- Define perpendicular relation
def planes_parallel (p1 p2 : ℝ → ℝ) : Prop := sorry -- Define planes being parallel

-- Given as conditions
axiom a_perpendicular_to_alpha : line_perpendicular_to_plane a α
axiom b_perpendicular_to_beta : line_perpendicular_to_plane b β
axiom a_parallel_to_b : a = b

-- The proposition to prove
theorem alpha_parallel_to_beta : planes_parallel α β :=
by {
  -- Placeholder for the logic provided through the previous solution steps.
  sorry
}

end alpha_parallel_to_beta_l1221_122112


namespace problem_l1221_122103

theorem problem (p q : ℝ) (h : 5 * p^2 - 20 * p + 15 = 0 ∧ 5 * q^2 - 20 * q + 15 = 0) : (p * q - 3)^2 = 0 := 
sorry

end problem_l1221_122103


namespace xiao_pang_xiao_ya_books_l1221_122105

theorem xiao_pang_xiao_ya_books : 
  ∀ (x y : ℕ), 
    (x + 2 * x = 66) → 
    (y + y / 3 = 92) → 
    (2 * x = 2 * x) → 
    (y = 3 * (y / 3)) → 
    ((22 + 69) - (2 * 22 + 69 / 3) = 24) :=
by
  intros x y h1 h2 h3 h4
  sorry

end xiao_pang_xiao_ya_books_l1221_122105


namespace correct_operation_l1221_122106

theorem correct_operation (a : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧ 
  ((-4 * a^3)^2 = 16 * a^6) ∧ 
  (a^6 / a^6 ≠ 0) ∧ 
  ((a - 1)^2 ≠ a^2 - 1) := by
  sorry

end correct_operation_l1221_122106


namespace problem_l1221_122141

def is_acute_angle (θ: ℝ) : Prop := θ > 0 ∧ θ < 90
def in_first_quadrant (θ: ℝ) : Prop := θ > 0 ∧ θ < 90
def in_second_quadrant (θ: ℝ) : Prop := θ > 90 ∧ θ < 180

def cond1 (θ: ℝ) : Prop := θ < 90 → is_acute_angle θ
def cond2 (θ: ℝ) : Prop := in_first_quadrant θ → θ ≥ 0
def cond3 (θ: ℝ) : Prop := is_acute_angle θ → in_first_quadrant θ
def cond4 (θ θ': ℝ) : Prop := in_second_quadrant θ → in_first_quadrant θ' → θ > θ'

theorem problem :
  (¬ ∃ θ, cond1 θ) ∧ (¬ ∃ θ, cond2 θ) ∧ (∃ θ, cond3 θ) ∧ (¬ ∃ θ θ', cond4 θ θ') →
  (number_of_correct_propositions = 1) :=
  by
    sorry

end problem_l1221_122141


namespace holiday_not_on_22nd_l1221_122194

def isThirdWednesday (d : ℕ) : Prop :=
  d = 15 ∨ d = 16 ∨ d = 17 ∨ d = 18 ∨ d = 19 ∨ d = 20 ∨ d = 21

theorem holiday_not_on_22nd :
  ¬ isThirdWednesday 22 :=
by
  intro h
  cases h
  repeat { contradiction }

end holiday_not_on_22nd_l1221_122194


namespace min_area_of_triangle_l1221_122186

noncomputable def area_of_triangle (p q : ℤ) : ℚ :=
  (1 / 2 : ℚ) * abs (3 * p - 5 * q)

theorem min_area_of_triangle :
  (∀ p q : ℤ, p ≠ 0 ∨ q ≠ 0 → area_of_triangle p q ≥ (1 / 2 : ℚ)) ∧
  (∃ p q : ℤ, p ≠ 0 ∨ q ≠ 0 ∧ area_of_triangle p q = (1 / 2 : ℚ)) := 
by { 
  sorry 
}

end min_area_of_triangle_l1221_122186


namespace bd_squared_l1221_122133

theorem bd_squared (a b c d : ℤ) (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 9) : 
  (b - d) ^ 2 = 4 := 
sorry

end bd_squared_l1221_122133


namespace A_days_l1221_122152

theorem A_days (B_days : ℕ) (total_wage A_wage : ℕ) (h_B_days : B_days = 15) (h_total_wage : total_wage = 3000) (h_A_wage : A_wage = 1800) :
  ∃ A_days : ℕ, A_days = 10 := by
  sorry

end A_days_l1221_122152


namespace measure_of_MNP_l1221_122183

-- Define the conditions of the pentagon
variables {M N P Q S : Type} -- Define the vertices of the pentagon
variables {MN NP PQ QS SM : ℝ} -- Define the lengths of the sides
variables (MNP QNS : ℝ) -- Define the measures of the involved angles

-- State the conditions
-- Pentagon sides are equal
axiom equal_sides : MN = NP ∧ NP = PQ ∧ PQ = QS ∧ QS = SM ∧ SM = MN 
-- Angle relation
axiom angle_relation : MNP = 2 * QNS

-- The goal is to prove that measure of angle MNP is 60 degrees
theorem measure_of_MNP : MNP = 60 :=
by {
  sorry -- The proof goes here
}

end measure_of_MNP_l1221_122183


namespace arun_age_l1221_122145

theorem arun_age (A G M : ℕ) (h1 : (A - 6) / 18 = G) (h2 : G = M - 2) (h3 : M = 5) : A = 60 :=
by
  sorry

end arun_age_l1221_122145


namespace polynomial_sum_l1221_122129

noncomputable def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem polynomial_sum : ∃ a b c d : ℝ, 
  (g a b c d (-3 * Complex.I) = 0) ∧
  (g a b c d (1 + Complex.I) = 0) ∧
  (g a b c d (3 * Complex.I) = 0) ∧
  (g a b c d (1 - Complex.I) = 0) ∧ 
  (a + b + c + d = 9) := by
  sorry

end polynomial_sum_l1221_122129


namespace uncovered_area_is_52_l1221_122179

-- Define the dimensions of the rectangles
def smaller_rectangle_length : ℕ := 4
def smaller_rectangle_width : ℕ := 2
def larger_rectangle_length : ℕ := 10
def larger_rectangle_width : ℕ := 6

-- Define the areas of both rectangles
def area_larger_rectangle : ℕ := larger_rectangle_length * larger_rectangle_width
def area_smaller_rectangle : ℕ := smaller_rectangle_length * smaller_rectangle_width

-- Define the area of the uncovered region
def area_uncovered_region : ℕ := area_larger_rectangle - area_smaller_rectangle

-- State the theorem
theorem uncovered_area_is_52 : area_uncovered_region = 52 := by sorry

end uncovered_area_is_52_l1221_122179


namespace pencils_ratio_l1221_122102

theorem pencils_ratio 
  (cindi_pencils : ℕ := 60)
  (marcia_mul_cindi : ℕ := 2)
  (total_pencils : ℕ := 480)
  (marcia_pencils : ℕ := marcia_mul_cindi * cindi_pencils) 
  (donna_pencils : ℕ := total_pencils - marcia_pencils) :
  donna_pencils / marcia_pencils = 3 := by
  sorry

end pencils_ratio_l1221_122102


namespace simplify_expression_solve_inequality_system_l1221_122139

-- Problem 1
theorem simplify_expression (m n : ℝ) (h1 : 3 * m - 2 * n ≠ 0) (h2 : 3 * m + 2 * n ≠ 0) (h3 : 9 * m ^ 2 - 4 * n ^ 2 ≠ 0) :
  ((1 / (3 * m - 2 * n) - 1 / (3 * m + 2 * n)) / (m * n / (9 * m ^ 2 - 4 * n ^ 2))) = (4 / m) :=
sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) (h1 : 3 * x + 10 > 5 * x - 2 * (5 - x)) (h2 : (x + 3) / 5 > 1 - x) :
  1 / 3 < x ∧ x < 5 :=
sorry

end simplify_expression_solve_inequality_system_l1221_122139


namespace orthogonal_trajectories_angle_at_origin_l1221_122146

theorem orthogonal_trajectories_angle_at_origin (x y : ℝ) (a : ℝ) :
  ((x + 2 * y) ^ 2 = a * (x + y)) →
  (∃ φ : ℝ, φ = π / 4) :=
by
  sorry

end orthogonal_trajectories_angle_at_origin_l1221_122146


namespace max_projection_area_of_tetrahedron_l1221_122177

theorem max_projection_area_of_tetrahedron (a : ℝ) (h1 : a > 0) :
  ∃ (A : ℝ), (A = a^2 / 2) :=
by
  sorry

end max_projection_area_of_tetrahedron_l1221_122177


namespace Eddie_number_divisibility_l1221_122144

theorem Eddie_number_divisibility (n: ℕ) (h₁: n = 40) (h₂: n % 5 = 0): n % 2 = 0 := 
by
  sorry

end Eddie_number_divisibility_l1221_122144


namespace side_length_S2_l1221_122199

-- Define the variables
variables (r s : ℕ)

-- Given conditions
def condition1 : Prop := 2 * r + s = 2300
def condition2 : Prop := 2 * r + 3 * s = 4000

-- The main statement to be proven
theorem side_length_S2 (h1 : condition1 r s) (h2 : condition2 r s) : s = 850 := sorry

end side_length_S2_l1221_122199


namespace solution_to_system_l1221_122109

def system_of_equations (x y : ℝ) : Prop := (x^2 - 9 * y^2 = 36) ∧ (3 * x + y = 6)

theorem solution_to_system : 
  {p : ℝ × ℝ | system_of_equations p.1 p.2} = { (12 / 5, -6 / 5), (3, -3) } := 
by sorry

end solution_to_system_l1221_122109


namespace unique_k_largest_n_l1221_122116

theorem unique_k_largest_n :
  ∃! k : ℤ, ∃ n : ℕ, (n > 0) ∧ (5 / 18 < n / (n + k) ∧ n / (n + k) < 9 / 17) ∧ (n = 1) :=
by
  sorry

end unique_k_largest_n_l1221_122116


namespace square_area_l1221_122185

theorem square_area (x : ℝ) (side_length : ℝ) 
  (h1_side_length : side_length = 5 * x - 10)
  (h2_side_length : side_length = 3 * (x + 4)) :
  side_length ^ 2 = 2025 :=
by
  sorry

end square_area_l1221_122185


namespace smallest_non_factor_product_l1221_122178

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 18 :=
by
  -- proof intentionally omitted
  sorry

end smallest_non_factor_product_l1221_122178


namespace range_of_a_l1221_122165

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 0 < x → x + (4 / x) - 1 - a^2 + 2 * a > 0) : -1 < a ∧ a < 3 :=
sorry

end range_of_a_l1221_122165


namespace full_tank_capacity_l1221_122107

theorem full_tank_capacity (speed : ℝ) (gas_usage_per_mile : ℝ) (time : ℝ) (gas_used_fraction : ℝ) (distance_per_tank : ℝ) (gallons_used : ℝ)
  (h1 : speed = 50)
  (h2 : gas_usage_per_mile = 1 / 30)
  (h3 : time = 5)
  (h4 : gas_used_fraction = 0.8333333333333334)
  (h5 : distance_per_tank = speed * time)
  (h6 : gallons_used = distance_per_tank * gas_usage_per_mile)
  (h7 : gallons_used = 0.8333333333333334 * 10) :
  distance_per_tank / 30 / 0.8333333333333334 = 10 :=
by sorry

end full_tank_capacity_l1221_122107


namespace solve_for_x_l1221_122131

theorem solve_for_x (x y : ℤ) (h1 : x + 3 * y = 33) (h2 : y = 10) : x = 3 :=
by
  sorry

end solve_for_x_l1221_122131


namespace length_of_rectangle_l1221_122143

theorem length_of_rectangle (l : ℝ) (s : ℝ) 
  (perimeter_square : 4 * s = 160) 
  (area_relation : s^2 = 5 * (l * 10)) : 
  l = 32 :=
by
  sorry

end length_of_rectangle_l1221_122143


namespace total_selling_price_l1221_122173

theorem total_selling_price (profit_per_meter cost_price_per_meter meters : ℕ)
  (h_profit : profit_per_meter = 20)
  (h_cost : cost_price_per_meter = 85)
  (h_meters : meters = 85) :
  (cost_price_per_meter + profit_per_meter) * meters = 8925 :=
by
  sorry

end total_selling_price_l1221_122173


namespace maximize_prob_l1221_122197

-- Define the probability of correctly answering each question
def prob_A : ℝ := 0.6
def prob_B : ℝ := 0.8
def prob_C : ℝ := 0.5

-- Define the probability of getting two questions correct in a row for each order
def prob_A_first : ℝ := (prob_A * prob_B * (1 - prob_C) + (1 - prob_A) * prob_B * prob_C) +
                        (prob_A * prob_C * (1 - prob_B) + (1 - prob_A) * prob_C * prob_B)
def prob_B_first : ℝ := (prob_B * prob_A * (1 - prob_C) + (1 - prob_B) * prob_A * prob_C) +
                        (prob_B * prob_C * (1 - prob_A) + (1 - prob_B) * prob_C * prob_A)
def prob_C_first : ℝ := (prob_C * prob_A * (1 - prob_B) + (1 - prob_C) * prob_A * prob_B) +
                        (prob_C * prob_B * (1 - prob_A) + (1 - prob_C) * prob_B * prob_A)

-- Prove that the maximum probability is obtained when question C is answered first
theorem maximize_prob : prob_C_first > prob_A_first ∧ prob_C_first > prob_B_first :=
by
  -- Add the proof details here
  sorry

end maximize_prob_l1221_122197


namespace arithmetic_sequence_sum_eight_l1221_122171

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sum (a₁ a₈ : α) (n : α) : α := (n * (a₁ + a₈)) / 2

theorem arithmetic_sequence_sum_eight {a₄ a₅ : α} (h₄₅ : a₄ + a₅ = 10) :
  let a₁ := a₄ - 3 * ((a₅ - a₄) / 1) -- a₁ in terms of a₄ and a₅
  let a₈ := a₄ + 4 * ((a₅ - a₄) / 1) -- a₈ in terms of a₄ and a₅
  arithmetic_sum a₁ a₈ 8 = 40 :=
by
  sorry

end arithmetic_sequence_sum_eight_l1221_122171


namespace min_value_of_quadratic_l1221_122167

-- Define the given quadratic function
def quadratic (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

-- Define the assertion that the minimum value of the quadratic function is 29/3
theorem min_value_of_quadratic : ∃ x : ℝ, quadratic x = 29/3 ∧ ∀ y : ℝ, quadratic y ≥ 29/3 :=
by
  sorry

end min_value_of_quadratic_l1221_122167


namespace balance_balls_l1221_122160

theorem balance_balls (G Y B W : ℝ) (h₁ : 4 * G = 10 * B) (h₂ : 3 * Y = 8 * B) (h₃ : 8 * B = 6 * W) :
  5 * G + 5 * Y + 4 * W = 31.1 * B :=
by
  sorry

end balance_balls_l1221_122160


namespace committee_vote_change_l1221_122121

-- Let x be the number of votes for the resolution initially.
-- Let y be the number of votes against the resolution initially.
-- The total number of voters is 500: x + y = 500.
-- The initial margin by which the resolution was defeated: y - x = m.
-- In the re-vote, the resolution passed with a margin three times the initial margin: x' - y' = 3m.
-- The number of votes for the re-vote was 13/12 of the votes against initially: x' = 13/12 * y.
-- The total number of voters remains 500 in the re-vote: x' + y' = 500.

theorem committee_vote_change (x y x' y' m : ℕ)
  (h1 : x + y = 500)
  (h2 : y - x = m)
  (h3 : x' - y' = 3 * m)
  (h4 : x' = 13 * y / 12)
  (h5 : x' + y' = 500) : x' - x = 40 := 
  by
  sorry

end committee_vote_change_l1221_122121


namespace replaced_person_age_is_40_l1221_122158

def average_age_decrease_replacement (T age_of_replaced: ℕ) : Prop :=
  let original_average := T / 10
  let new_total_age := T - age_of_replaced + 10
  let new_average := new_total_age / 10
  original_average - 3 = new_average

theorem replaced_person_age_is_40 (T : ℕ) (h : average_age_decrease_replacement T 40) : Prop :=
  ∀ age_of_replaced, age_of_replaced = 40 → average_age_decrease_replacement T age_of_replaced

-- To actually formalize the proof, you can use the following structure:
-- proof by calculation omitted
lemma replaced_person_age_is_40_proof (T : ℕ) (h : average_age_decrease_replacement T 40) : 
  replaced_person_age_is_40 T h :=
by
  sorry

end replaced_person_age_is_40_l1221_122158


namespace total_trees_after_planting_l1221_122117

def initial_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20

theorem total_trees_after_planting :
  initial_trees + trees_planted_today + trees_planted_tomorrow = 100 := 
by sorry

end total_trees_after_planting_l1221_122117


namespace option_D_not_right_angled_l1221_122195

def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def option_A (a b c : ℝ) : Prop :=
  b^2 = a^2 - c^2

def option_B (a b c : ℝ) : Prop :=
  a = 3 * c / 5 ∧ b = 4 * c / 5

def option_C (A B C : ℝ) : Prop :=
  C = A - B ∧ A + B + C = 180

def option_D (A B C : ℝ) : Prop :=
  A / 3 = B / 4 ∧ B / 4 = C / 5

theorem option_D_not_right_angled (a b c A B C : ℝ) :
  ¬ is_right_angled_triangle a b c ↔ option_D A B C :=
  sorry

end option_D_not_right_angled_l1221_122195


namespace quadratic_inequality_solution_l1221_122118

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 3 * x - 10 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 5} :=
by
  sorry

end quadratic_inequality_solution_l1221_122118


namespace beth_sold_l1221_122148

theorem beth_sold {initial_coins additional_coins total_coins sold_coins : ℕ} 
  (h_init : initial_coins = 125)
  (h_add : additional_coins = 35)
  (h_total : total_coins = initial_coins + additional_coins)
  (h_sold : sold_coins = total_coins / 2) :
  sold_coins = 80 := 
sorry

end beth_sold_l1221_122148


namespace determine_cards_per_friend_l1221_122147

theorem determine_cards_per_friend (n_cards : ℕ) (n_friends : ℕ) (h : n_cards = 12) : n_friends > 0 → (n_cards / n_friends) = (12 / n_friends) :=
by
  sorry

end determine_cards_per_friend_l1221_122147


namespace point_B_coordinates_l1221_122100

-- Defining the vector a
def vec_a : ℝ × ℝ := (1, 0)

-- Defining the point A
def A : ℝ × ℝ := (4, 4)

-- Definition of the line y = 2x
def on_line (P : ℝ × ℝ) : Prop := P.2 = 2 * P.1

-- Defining a vector as being parallel to another vector
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

-- Lean statement for the proof
theorem point_B_coordinates (B : ℝ × ℝ) (h1 : on_line B) (h2 : parallel (B.1 - 4, B.2 - 4) vec_a) :
  B = (2, 4) :=
sorry

end point_B_coordinates_l1221_122100


namespace find_a_for_square_of_binomial_l1221_122176

theorem find_a_for_square_of_binomial (a : ℝ) :
  (∃ r s : ℝ, (r * x + s)^2 = a * x^2 + 18 * x + 9) ↔ a = 9 := 
sorry

end find_a_for_square_of_binomial_l1221_122176


namespace rate_per_kg_of_grapes_l1221_122181

-- Define the conditions 
namespace Problem

-- Given conditions
variables (G : ℝ) (rate_mangoes : ℝ := 55) (cost_paid : ℝ := 1055)
variables (kg_grapes : ℝ := 8) (kg_mangoes : ℝ := 9)

-- Statement to prove
theorem rate_per_kg_of_grapes : 8 * G + 9 * rate_mangoes = cost_paid → G = 70 := 
by
  intro h
  sorry -- proof goes here

end Problem

end rate_per_kg_of_grapes_l1221_122181


namespace sin_pi_over_6_plus_α_cos_pi_over_3_plus_2α_l1221_122184

variable (α : ℝ)

-- Given conditions
def α_condition (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) : Prop := 
  true

-- Prove the first part: sin(π / 6 + α) = (3 + 4 * real.sqrt 3) / 10
theorem sin_pi_over_6_plus_α (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) :
  Real.sin (π / 6 + α) = (3 + 4 * Real.sqrt 3) / 10 :=
by
  sorry

-- Prove the second part: cos(π / 3 + 2 * α) = -(7 + 24 * real.sqrt 3) / 50
theorem cos_pi_over_3_plus_2α (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) :
  Real.cos (π / 3 + 2 * α) = -(7 + 24 * Real.sqrt 3) / 50 :=
by
  sorry

end sin_pi_over_6_plus_α_cos_pi_over_3_plus_2α_l1221_122184


namespace area_of_10th_square_l1221_122172

noncomputable def area_of_square (n: ℕ) : ℚ :=
  if n = 1 then 4
  else 2 * (1 / 2)^(n - 1)

theorem area_of_10th_square : area_of_square 10 = 1 / 256 := 
  sorry

end area_of_10th_square_l1221_122172


namespace circle_center_radius_sum_correct_l1221_122127

noncomputable def circle_center_radius_sum (eq : String) : ℝ :=
  if h : eq = "x^2 + 8x - 2y^2 - 6y = -6" then
    let c : ℝ := -4
    let d : ℝ := -3 / 2
    let s : ℝ := Real.sqrt (47 / 4)
    c + d + s
  else 0

theorem circle_center_radius_sum_correct :
  circle_center_radius_sum "x^2 + 8x - 2y^2 - 6y = -6" = (-11 + Real.sqrt 47) / 2 :=
by
  -- proof omitted
  sorry

end circle_center_radius_sum_correct_l1221_122127


namespace ratio_pentagon_rectangle_l1221_122126

theorem ratio_pentagon_rectangle (s_p w : ℝ) (H_pentagon : 5 * s_p = 60) (H_rectangle : 6 * w = 80) : s_p / w = 9 / 10 :=
by
  sorry

end ratio_pentagon_rectangle_l1221_122126


namespace ratio_of_ages_l1221_122166

theorem ratio_of_ages (M : ℕ) (S : ℕ) (h1 : M = 24) (h2 : S + 6 = 38) : 
  (S / M : ℚ) = 4 / 3 :=
by
  sorry

end ratio_of_ages_l1221_122166


namespace find_current_listens_l1221_122188

theorem find_current_listens (x : ℕ) (h : 15 * x = 900000) : x = 60000 :=
by
  sorry

end find_current_listens_l1221_122188


namespace max_marks_l1221_122196

theorem max_marks (M : ℕ) (h_pass : 55 / 100 * M = 510) : M = 928 :=
sorry

end max_marks_l1221_122196


namespace students_in_classes_saved_money_strategy_class7_1_l1221_122110

-- Part (1): Prove the number of students in each class
theorem students_in_classes (x : ℕ) (h1 : 40 < x) (h2 : x < 50) 
  (h3 : 105 - x > 50) (h4 : 15 * x + 12 * (105 - x) = 1401) : x = 47 ∧ (105 - x) = 58 := by
  sorry

-- Part (2): Prove the amount saved by purchasing tickets together
theorem saved_money(amt_per_ticket : ℕ → ℕ) 
  (h1 : amt_per_ticket 105 = 1401) 
  (h2 : ∀n, n > 100 → amt_per_ticket n = 1050) : amt_per_ticket 105 - 1050 = 351 := by
  sorry

-- Part (3): Strategy to save money for class 7 (1)
theorem strategy_class7_1 (students_1 : ℕ) (h1 : students_1 = 47) 
  (cost_15 : students_1 * 15 = 705) 
  (cost_51 : 51 * 12 = 612) : 705 - 612 = 93 := by
  sorry

end students_in_classes_saved_money_strategy_class7_1_l1221_122110


namespace simplify_frac_l1221_122189

theorem simplify_frac : (5^4 + 5^2) / (5^3 - 5) = 65 / 12 :=
by 
  sorry

end simplify_frac_l1221_122189


namespace sum_at_simple_interest_l1221_122187

theorem sum_at_simple_interest 
  (P R : ℕ)
  (h : ((P * (R + 1) * 3) / 100) - ((P * R * 3) / 100) = 69) : 
  P = 2300 :=
by sorry

end sum_at_simple_interest_l1221_122187


namespace perfect_squares_between_50_and_1000_l1221_122182

theorem perfect_squares_between_50_and_1000 :
  ∃ (count : ℕ), count = 24 ∧ ∀ (n : ℕ), 50 < n * n ∧ n * n < 1000 ↔ 8 ≤ n ∧ n ≤ 31 :=
by {
  -- proof goes here
  sorry
}

end perfect_squares_between_50_and_1000_l1221_122182


namespace cars_sold_on_second_day_l1221_122124

theorem cars_sold_on_second_day (x : ℕ) 
  (h1 : 14 + x + 27 = 57) : x = 16 :=
by 
  sorry

end cars_sold_on_second_day_l1221_122124


namespace rectangular_prism_length_l1221_122169

theorem rectangular_prism_length (w l h : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : h = 3 * w) 
  (h3 : 4 * l + 4 * w + 4 * h = 256) : 
  l = 32 :=
by
  sorry

end rectangular_prism_length_l1221_122169


namespace range_of_y_l1221_122168

theorem range_of_y (y: ℝ) (hy: y > 0) (h_eq: ⌈y⌉ * ⌊y⌋ = 72) : 8 < y ∧ y < 9 :=
by
  sorry

end range_of_y_l1221_122168


namespace probability_of_type_A_probability_of_different_type_l1221_122198

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

end probability_of_type_A_probability_of_different_type_l1221_122198


namespace friends_meeting_probability_l1221_122125

noncomputable def n_value (d e f : ℝ) (h1 : d = 60) (h2 : e = 30) (h3 : f = 2) : ℝ :=
  d - e * Real.sqrt f

theorem friends_meeting_probability (n : ℝ) (d e f : ℝ) (h1 : d = 60) (h2 : e = 30) (h3 : f = 2)
  (H : n = n_value d e f h1 h2 h3) : d + e + f = 92 :=
  by
  sorry

end friends_meeting_probability_l1221_122125


namespace problem_solution_l1221_122122

noncomputable def a (n : ℕ) : ℕ := 2 * n - 3

noncomputable def b (n : ℕ) : ℕ := 2 ^ n

noncomputable def c (n : ℕ) : ℕ := a n * b n

noncomputable def sum_c (n : ℕ) : ℕ :=
  (2 * n - 5) * 2 ^ (n + 1) + 10

theorem problem_solution :
  ∀ n : ℕ, n > 0 →
  (S_n = 2 * (b n - 1)) ∧
  (a 2 = b 1 - 1) ∧
  (a 5 = b 3 - 1)
  →
  (∀ n, a n = 2 * n - 3) ∧
  (∀ n, b n = 2 ^ n) ∧
  (sum_c n = (2 * n - 5) * 2 ^ (n + 1) + 10) :=
by
  intros n hn h
  sorry


end problem_solution_l1221_122122


namespace evaluate_f_at_points_l1221_122115

def f (x : ℝ) : ℝ :=
  3 * x ^ 2 - 6 * x + 10

theorem evaluate_f_at_points : 3 * f 2 + 2 * f (-2) = 98 :=
by
  sorry

end evaluate_f_at_points_l1221_122115


namespace max_value_abs_diff_PQ_PR_l1221_122149

-- Definitions for the points on the given curves
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1
def circle1 (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x + 5)^2 + y^2 = 1

-- Statement of the problem as a theorem
theorem max_value_abs_diff_PQ_PR (P Q R : ℝ × ℝ)
(hyp_P : hyperbola P.1 P.2)
(hyp_Q : circle1 Q.1 Q.2)
(hyp_R : circle2 R.1 R.2) :
  max (abs (dist P Q - dist P R)) = 10 :=
sorry

end max_value_abs_diff_PQ_PR_l1221_122149


namespace equilateral_triangle_M_properties_l1221_122180

-- Define the points involved
variables (A B C M P Q R : ℝ)
-- Define distances from M to the sides as given by perpendiculars
variables (d_AP d_BQ d_CR d_PB d_QC d_RA : ℝ)

-- Equilateral triangle assumption and perpendiculars from M to sides
def equilateral_triangle (A B C : ℝ) : Prop := sorry
def perpendicular_from_point (M P R : ℝ) (line : ℝ) : Prop := sorry

-- Problem statement encapsulating the given conditions and what needs to be proved:
theorem equilateral_triangle_M_properties
  (h_triangle: equilateral_triangle A B C)
  (h_perp_AP: perpendicular_from_point M P A B)
  (h_perp_BQ: perpendicular_from_point M Q B C)
  (h_perp_CR: perpendicular_from_point M R C A) :
  (d_AP^2 + d_BQ^2 + d_CR^2 = d_PB^2 + d_QC^2 + d_RA^2) ∧ 
  (d_AP + d_BQ + d_CR = d_PB + d_QC + d_RA) := sorry

end equilateral_triangle_M_properties_l1221_122180


namespace problem1_problem2_l1221_122104

def A (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 7
def S (x : ℝ) (k : ℝ) : Prop := k + 1 ≤ x ∧ x ≤ 2 * k - 1

theorem problem1 (k : ℝ) : (∀ x, S x k → A x) → k ≤ 4 :=
by
  sorry

theorem problem2 (k : ℝ) : (∀ x, ¬(A x ∧ S x k)) → k < 2 ∨ k > 6 :=
by
  sorry

end problem1_problem2_l1221_122104


namespace greatest_possible_selling_price_l1221_122132

variable (products : ℕ)
variable (average_price : ℝ)
variable (min_price : ℝ)
variable (less_than_1000_products : ℕ)

theorem greatest_possible_selling_price
  (h1 : products = 20)
  (h2 : average_price = 1200)
  (h3 : min_price = 400)
  (h4 : less_than_1000_products = 10) :
  ∃ max_price, max_price = 11000 := 
by
  sorry

end greatest_possible_selling_price_l1221_122132


namespace xiaoxiao_age_in_2015_l1221_122150

-- Definitions for conditions
variables (x : ℕ) (T : ℕ)

-- The total age of the family in 2015 was 7 times Xiaoxiao's age
axiom h1 : T = 7 * x

-- The total age of the family in 2020 after the sibling is 6 times Xiaoxiao's age in 2020
axiom h2 : T + 19 = 6 * (x + 5)

-- Proof goal: Xiaoxiao’s age in 2015 is 11
theorem xiaoxiao_age_in_2015 : x = 11 :=
by
  sorry

end xiaoxiao_age_in_2015_l1221_122150


namespace no_such_two_digit_number_exists_l1221_122114

theorem no_such_two_digit_number_exists :
  ¬ ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
                 (10 * x + y = 2 * (x^2 + y^2) + 6) ∧
                 (10 * x + y = 4 * (x * y) + 6) := by
  -- We need to prove that no two-digit number satisfies
  -- both conditions.
  sorry

end no_such_two_digit_number_exists_l1221_122114


namespace alfred_saving_goal_l1221_122120

theorem alfred_saving_goal (leftover : ℝ) (monthly_saving : ℝ) (months : ℕ) :
  leftover = 100 → monthly_saving = 75 → months = 12 → leftover + monthly_saving * months = 1000 :=
by
  sorry

end alfred_saving_goal_l1221_122120


namespace weight_of_person_being_replaced_l1221_122157

variable (W_old : ℝ)

theorem weight_of_person_being_replaced :
  (W_old : ℝ) = 35 :=
by
  -- Given: The average weight of 8 persons increases by 5 kg.
  -- The weight of the new person is 75 kg.
  -- The total weight increase is 40 kg.
  -- Prove that W_old = 35 kg.
  sorry

end weight_of_person_being_replaced_l1221_122157


namespace gumball_draw_probability_l1221_122130

def prob_blue := 2 / 3
def prob_two_blue := (16 / 36)
def prob_pink := 1 - prob_blue

theorem gumball_draw_probability
    (h1 : prob_two_blue = prob_blue * prob_blue)
    (h2 : prob_blue + prob_pink = 1) :
    prob_pink = 1 / 3 := 
by
  sorry

end gumball_draw_probability_l1221_122130


namespace correct_option_is_D_l1221_122156

noncomputable def expression1 (a b : ℝ) : Prop := a + b > 2 * b^2
noncomputable def expression2 (a b : ℝ) : Prop := a^5 + b^5 > a^3 * b^2 + a^2 * b^3
noncomputable def expression3 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * (a - b - 1)
noncomputable def expression4 (a b : ℝ) : Prop := (b / a) + (a / b) > 2

theorem correct_option_is_D (a b : ℝ) (h : a ≠ b) : 
  (expression3 a b ∧ ¬expression1 a b ∧ ¬expression2 a b ∧ ¬expression4 a b) :=
by
  sorry

end correct_option_is_D_l1221_122156


namespace hexagon_perimeter_eq_4_sqrt_3_over_3_l1221_122135

noncomputable def hexagon_perimeter (s : ℝ) : ℝ :=
  6 * s

theorem hexagon_perimeter_eq_4_sqrt_3_over_3 :
  ∀ (s : ℝ), (∃ s, (3 * Real.sqrt 3 / 2) * s^2 = s) → hexagon_perimeter s = 4 * Real.sqrt 3 / 3 :=
by
  simp
  sorry

end hexagon_perimeter_eq_4_sqrt_3_over_3_l1221_122135


namespace units_digit_product_even_composite_l1221_122174

/-- The units digit of the product of the first three even composite numbers greater than 10 is 8. -/
theorem units_digit_product_even_composite :
  let a := 12
  let b := 14
  let c := 16
  (a * b * c) % 10 = 8 :=
by
  let a := 12
  let b := 14
  let c := 16
  have h : (a * b * c) % 10 = 8
  { sorry }
  exact h

end units_digit_product_even_composite_l1221_122174


namespace aluminum_percentage_in_new_alloy_l1221_122159

theorem aluminum_percentage_in_new_alloy :
  ∀ (x1 x2 x3 : ℝ),
  0 ≤ x1 ∧ x1 ≤ 1 ∧
  0 ≤ x2 ∧ x2 ≤ 1 ∧
  0 ≤ x3 ∧ x3 ≤ 1 ∧
  x1 + x2 + x3 = 1 ∧
  0.15 * x1 + 0.3 * x2 = 0.2 →
  0.15 ≤ 0.6 * x1 + 0.45 * x3 ∧ 0.6 * x1 + 0.45 * x3 ≤ 0.40 :=
by
  -- The proof will be inserted here
  sorry

end aluminum_percentage_in_new_alloy_l1221_122159


namespace probability_of_25_cents_heads_l1221_122123

/-- 
Considering the flipping of five specific coins: a penny, a nickel, a dime,
a quarter, and a half dollar, prove that the probability of getting at least
25 cents worth of heads is 3 / 4.
-/
theorem probability_of_25_cents_heads :
  let total_outcomes := 2^5
  let successful_outcomes_1 := 2^4
  let successful_outcomes_2 := 2^3
  let successful_outcomes := successful_outcomes_1 + successful_outcomes_2
  (successful_outcomes / total_outcomes : ℚ) = 3 / 4 :=
by
  sorry

end probability_of_25_cents_heads_l1221_122123


namespace min_f_l1221_122101

noncomputable def f (x y z : ℝ) : ℝ :=
  (3 * x^2 - x) / (1 + x^2) + 
  (3 * y^2 - y) / (1 + y^2) + 
  (3 * z^2 - z) / (1 + z^2)

theorem min_f (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  f x y z = 0 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
sorry

end min_f_l1221_122101
