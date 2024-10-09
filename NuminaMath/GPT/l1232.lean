import Mathlib

namespace problem1_l1232_123253

theorem problem1 : (- (1 / 12) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = -21 :=
by
  sorry

end problem1_l1232_123253


namespace solution_to_fractional_equation_l1232_123278

theorem solution_to_fractional_equation (x : ℝ) (h : x ≠ 3 ∧ x ≠ 1) :
  (x / (x - 3) = (x + 1) / (x - 1)) ↔ (x = -3) :=
by
  sorry

end solution_to_fractional_equation_l1232_123278


namespace solve_for_x_l1232_123242

theorem solve_for_x :
  ∃ x : ℝ, (24 / 36) = Real.sqrt (x / 36) ∧ x = 16 :=
by
  use 16
  sorry

end solve_for_x_l1232_123242


namespace triangle_area_l1232_123267

def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem triangle_area :
  area_of_triangle 0 0 0 6 8 0 = 24 :=
by
  sorry

end triangle_area_l1232_123267


namespace factorization_proof_l1232_123273

def factorization_problem (x : ℝ) : Prop := (x^2 - 1)^2 - 6 * (x^2 - 1) + 9 = (x - 2)^2 * (x + 2)^2

theorem factorization_proof (x : ℝ) : factorization_problem x :=
by
  -- The proof is omitted.
  sorry

end factorization_proof_l1232_123273


namespace number_of_albums_l1232_123276

-- Definitions for the given conditions
def pictures_from_phone : ℕ := 7
def pictures_from_camera : ℕ := 13
def pictures_per_album : ℕ := 4

-- We compute the total number of pictures
def total_pictures : ℕ := pictures_from_phone + pictures_from_camera

-- Statement: Prove the number of albums is 5
theorem number_of_albums :
  total_pictures / pictures_per_album = 5 := by
  sorry

end number_of_albums_l1232_123276


namespace parametric_to_cartesian_l1232_123240

variable (R t : ℝ)

theorem parametric_to_cartesian (x y : ℝ) (h1 : x = R * Real.cos t) (h2 : y = R * Real.sin t) : 
  x^2 + y^2 = R^2 := 
by
  sorry

end parametric_to_cartesian_l1232_123240


namespace find_x_satisfying_condition_l1232_123214

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem find_x_satisfying_condition : ∀ x : ℝ, (A x ∪ B x = A x) ↔ (x = 2 ∨ x = -2 ∨ x = 0) := by
  sorry

end find_x_satisfying_condition_l1232_123214


namespace soccer_league_teams_l1232_123203

theorem soccer_league_teams (n : ℕ) (h : n * (n - 1) / 2 = 55) : n = 11 := 
sorry

end soccer_league_teams_l1232_123203


namespace distance_to_x_axis_l1232_123208

theorem distance_to_x_axis (x y : ℤ) (h : (x, y) = (-3, 5)) : |y| = 5 := by
  -- coordinates of point A are (-3, 5)
  sorry

end distance_to_x_axis_l1232_123208


namespace acute_triangle_iff_sum_of_squares_l1232_123258

theorem acute_triangle_iff_sum_of_squares (a b c R : ℝ) 
  (hRpos : R > 0) 
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  (∀ α β γ, (a = 2 * R * Real.sin α) ∧ (b = 2 * R * Real.sin β) ∧ (c = 2 * R * Real.sin γ) → 
   (α < Real.pi / 2 ∧ β < Real.pi / 2 ∧ γ < Real.pi / 2)) ↔ 
  (a^2 + b^2 + c^2 > 8 * R^2) :=
sorry

end acute_triangle_iff_sum_of_squares_l1232_123258


namespace total_weight_full_bucket_l1232_123279

theorem total_weight_full_bucket (x y p q : ℝ)
  (h1 : x + (3 / 4) * y = p)
  (h2 : x + (1 / 3) * y = q) :
  x + y = (8 * p - 11 * q) / 5 :=
by
  sorry

end total_weight_full_bucket_l1232_123279


namespace minimum_frosting_time_l1232_123218

def ann_time_per_cake := 8 -- Ann's time per cake in minutes
def bob_time_per_cake := 6 -- Bob's time per cake in minutes
def carol_time_per_cake := 10 -- Carol's time per cake in minutes
def passing_time := 1 -- time to pass a cake from one person to another in minutes
def total_cakes := 10 -- total number of cakes to be frosted

theorem minimum_frosting_time : 
  (ann_time_per_cake + passing_time + bob_time_per_cake + passing_time + carol_time_per_cake) + (total_cakes - 1) * carol_time_per_cake = 116 := 
by 
  sorry

end minimum_frosting_time_l1232_123218


namespace number_of_pigs_l1232_123248

theorem number_of_pigs (daily_feed_per_pig : ℕ) (weekly_feed_total : ℕ) (days_per_week : ℕ)
  (h1 : daily_feed_per_pig = 10) (h2 : weekly_feed_total = 140) (h3 : days_per_week = 7) : 
  (weekly_feed_total / days_per_week) / daily_feed_per_pig = 2 := by
  sorry

end number_of_pigs_l1232_123248


namespace lorie_total_bills_l1232_123292

-- Definitions for the conditions
def initial_hundred_bills := 2
def hundred_to_fifty (bills : Nat) : Nat := bills * 2 / 100
def hundred_to_ten (bills : Nat) : Nat := (bills / 2) / 10
def hundred_to_five (bills : Nat) : Nat := (bills / 2) / 5

-- Statement of the problem
theorem lorie_total_bills : 
  let fifty_bills := hundred_to_fifty 100
  let ten_bills := hundred_to_ten 100
  let five_bills := hundred_to_five 100
  fifty_bills + ten_bills + five_bills = 2 + 5 + 10 :=
sorry

end lorie_total_bills_l1232_123292


namespace minimize_quadratic_function_l1232_123233

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l1232_123233


namespace ellipse_hyperbola_same_foci_l1232_123226

theorem ellipse_hyperbola_same_foci (k : ℝ) (h1 : k > 0) :
  (∀ (x y : ℝ), (x^2 / 9 + y^2 / k^2 = 1) ↔ (x^2 / k - y^2 / 3 = 1)) → k = 2 :=
by
  sorry

end ellipse_hyperbola_same_foci_l1232_123226


namespace final_result_after_subtracting_15_l1232_123237

theorem final_result_after_subtracting_15 :
  ∀ (n : ℕ) (r : ℕ) (f : ℕ),
  n = 120 → 
  r = n / 6 → 
  f = r - 15 → 
  f = 5 :=
by
  intros n r f hn hr hf
  have h1 : n = 120 := hn
  have h2 : r = n / 6 := hr
  have h3 : f = r - 15 := hf
  sorry

end final_result_after_subtracting_15_l1232_123237


namespace arithmetic_progression_probability_l1232_123232

def is_arithmetic_progression (a b c : ℕ) (d : ℕ) : Prop :=
  b - a = d ∧ c - b = d

noncomputable def probability_arithmetic_progression_diff_two : ℚ :=
  have total_outcomes : ℚ := 6 * 6 * 6
  have favorable_outcomes : ℚ := 12
  favorable_outcomes / total_outcomes

theorem arithmetic_progression_probability (d : ℕ) (h : d = 2) :
  probability_arithmetic_progression_diff_two = 1 / 18 :=
by 
  sorry

end arithmetic_progression_probability_l1232_123232


namespace part_a_part_b_part_c_l1232_123225

theorem part_a (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (ineq : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a ≤ b + c ∧ b ≤ a + c ∧ c ≤ a + b := 
sorry

theorem part_b (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (ineq : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) := 
sorry

theorem part_c (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) :
  ¬ (a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) → 
     a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :=
sorry

end part_a_part_b_part_c_l1232_123225


namespace remainder_product_l1232_123211

theorem remainder_product (x y : ℤ) 
  (hx : x % 792 = 62) 
  (hy : y % 528 = 82) : 
  (x * y) % 66 = 24 := 
by 
  sorry

end remainder_product_l1232_123211


namespace calculate_expression_l1232_123200

theorem calculate_expression :
  (10^4 - 9^4 + 8^4 - 7^4 + 6^4 - 5^4 + 4^4 - 3^4 + 2^4 - 1^4) +
  (10^2 + 9^2 + 5 * 8^2 + 5 * 7^2 + 9 * 6^2 + 9 * 5^2 + 13 * 4^2 + 13 * 3^2) = 7615 := by
  sorry

end calculate_expression_l1232_123200


namespace tickets_spent_on_beanie_l1232_123262

-- Define the initial number of tickets Jerry had.
def initial_tickets : ℕ := 4

-- Define the number of tickets Jerry won later.
def won_tickets : ℕ := 47

-- Define the current number of tickets Jerry has.
def current_tickets : ℕ := 49

-- The statement of the problem to prove the tickets spent on the beanie.
theorem tickets_spent_on_beanie :
  initial_tickets + won_tickets - 2 = current_tickets := by
  sorry

end tickets_spent_on_beanie_l1232_123262


namespace find_period_l1232_123277

variable (x : ℕ)
variable (theo_daily : ℕ := 8)
variable (mason_daily : ℕ := 7)
variable (roxy_daily : ℕ := 9)
variable (total_water : ℕ := 168)

theorem find_period (h : (theo_daily + mason_daily + roxy_daily) * x = total_water) : x = 7 :=
by
  sorry

end find_period_l1232_123277


namespace Ganesh_avg_speed_l1232_123287

theorem Ganesh_avg_speed (D : ℝ) : 
  (∃ (V : ℝ), (39.6 = (2 * D) / ((D / 44) + (D / V))) ∧ V = 36) :=
by
  sorry

end Ganesh_avg_speed_l1232_123287


namespace contradiction_method_assumption_l1232_123251

-- Definitions for three consecutive positive integers
variables {a b c : ℕ}

-- Definitions for the proposition and its negation
def consecutive_integers (a b c : ℕ) : Prop := b = a + 1 ∧ c = b + 1
def at_least_one_divisible_by_2 (a b c : ℕ) : Prop := a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0
def all_not_divisible_by_2 (a b c : ℕ) : Prop := a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0

theorem contradiction_method_assumption (a b c : ℕ) (h : consecutive_integers a b c) :
  (¬ at_least_one_divisible_by_2 a b c) ↔ all_not_divisible_by_2 a b c :=
by sorry

end contradiction_method_assumption_l1232_123251


namespace product_of_odd_primes_mod_sixteen_l1232_123205

-- Define the set of odd primes less than 16
def odd_primes_less_than_sixteen : List ℕ := [3, 5, 7, 11, 13]

-- Define the product of all odd primes less than 16
def N : ℕ := odd_primes_less_than_sixteen.foldl (· * ·) 1

-- Proposition to prove: N ≡ 7 (mod 16)
theorem product_of_odd_primes_mod_sixteen :
  (N % 16) = 7 :=
  sorry

end product_of_odd_primes_mod_sixteen_l1232_123205


namespace solve_log_eq_l1232_123230

theorem solve_log_eq : ∀ x : ℝ, (2 : ℝ) ^ (Real.log x / Real.log 3) = (1 / 4 : ℝ) → x = 1 / 9 :=
by
  intro x
  sorry

end solve_log_eq_l1232_123230


namespace obtain_x_squared_obtain_xy_l1232_123294

theorem obtain_x_squared (x y : ℝ) (hx : x ≠ 1) (hx0 : 0 < x) (hy0 : 0 < y) :
  ∃ (k : ℝ), k = x^2 :=
by
  sorry

theorem obtain_xy (x y : ℝ) (hx0 : 0 < x) (hy0 : 0 < y) :
  ∃ (k : ℝ), k = x * y :=
by
  sorry

end obtain_x_squared_obtain_xy_l1232_123294


namespace eval_expression_eq_one_l1232_123254

theorem eval_expression_eq_one (x : ℝ) (hx1 : x^3 + 1 = (x+1)*(x^2 - x + 1)) (hx2 : x^3 - 1 = (x-1)*(x^2 + x + 1)) :
  ( ((x+1)^3 * (x^2 - x + 1)^3 / (x^3 + 1)^3)^2 * ((x-1)^3 * (x^2 + x + 1)^3 / (x^3 - 1)^3)^2 ) = 1 :=
by
  sorry

end eval_expression_eq_one_l1232_123254


namespace sum_of_coordinates_l1232_123239

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 2 = 3) : 
  let x := 2 / 3
  let y := 2 * f (3 * x) + 4
  x + y = 32 / 3 :=
by
  sorry

end sum_of_coordinates_l1232_123239


namespace ratio_of_areas_l1232_123221

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l1232_123221


namespace arithmetic_sequence_sum_10_l1232_123261

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

noncomputable def a_n (a1 d : α) (n : ℕ) : α :=
a1 + (n - 1) • d

def sequence_sum (a1 d : α) (n : ℕ) : α :=
n • a1 + (n • (n - 1) / 2) • d

theorem arithmetic_sequence_sum_10 
  (a1 d : ℤ)
  (h1 : a_n a1 d 2 + a_n a1 d 4 = 4)
  (h2 : a_n a1 d 3 + a_n a1 d 5 = 10) :
  sequence_sum a1 d 10 = 95 :=
by
  sorry

end arithmetic_sequence_sum_10_l1232_123261


namespace problem_part_1_problem_part_2_l1232_123295

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x

noncomputable def g (x : ℝ) : ℝ := Real.log ((x + 2) / (x - 2))

theorem problem_part_1 :
  ∀ (x₁ x₂ : ℝ), 0 < x₂ ∧ x₂ < x₁ → Real.log x₁ + 2 * x₁ > Real.log x₂ + 2 * x₂ :=
sorry

theorem problem_part_2 :
  ∃ k : ℕ, ∀ (x₁ : ℝ), 0 < x₁ ∧ x₁ < 1 → (∃ (x₂ : ℝ), x₂ ∈ Set.Ioo (k : ℝ) (k + 1) ∧ Real.log x₁ + 2 * x₁ < Real.log ((x₂ + 2) / (x₂ - 2))) → k = 2 :=
sorry

end problem_part_1_problem_part_2_l1232_123295


namespace initial_cards_eq_4_l1232_123275

theorem initial_cards_eq_4 (x : ℕ) (h : x + 3 = 7) : x = 4 :=
by
  sorry

end initial_cards_eq_4_l1232_123275


namespace part1_expression_for_f_part2_three_solutions_l1232_123234

noncomputable def f1 (x : ℝ) := x^2

noncomputable def f2 (x : ℝ) := 8 / x

noncomputable def f (x : ℝ) := f1 x + f2 x

theorem part1_expression_for_f : ∀ x:ℝ, f x = x^2 + 8 / x := by
  sorry  -- This is where the proof would go

theorem part2_three_solutions (a : ℝ) (h : a > 3) : 
  ∃ x1 x2 x3 : ℝ, f x1 = f a ∧ f x2 = f a ∧ f x3 = f a ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 := by
  sorry  -- This is where the proof would go

end part1_expression_for_f_part2_three_solutions_l1232_123234


namespace point_P_trajectory_circle_l1232_123280

noncomputable def trajectory_of_point_P (d h1 h2 : ℝ) (x y : ℝ) : Prop :=
  (x - d/2)^2 + y^2 = (h1^2 + h2^2) / (2 * (h2/h1)^(2/3))

theorem point_P_trajectory_circle :
  ∀ (d h1 h2 x y : ℝ),
  d = 20 →
  h1 = 15 →
  h2 = 10 →
  (∃ x y, trajectory_of_point_P d h1 h2 x y) →
  (∃ x y, (x - 16)^2 + y^2 = 24^2) :=
by
  intros d h1 h2 x y hd hh1 hh2 hxy
  sorry

end point_P_trajectory_circle_l1232_123280


namespace largest_expr_is_expr1_l1232_123219

def U : ℝ := 3 * 2005 ^ 2006
def V : ℝ := 2005 ^ 2006
def W : ℝ := 2004 * 2005 ^ 2005
def X : ℝ := 3 * 2005 ^ 2005
def Y : ℝ := 2005 ^ 2005
def Z : ℝ := 2005 ^ 2004

def expr1 : ℝ := U - V
def expr2 : ℝ := V - W
def expr3 : ℝ := W - X
def expr4 : ℝ := X - Y
def expr5 : ℝ := Y - Z

theorem largest_expr_is_expr1 : 
  max (max (max expr1 expr2) (max expr3 expr4)) expr5 = expr1 := 
sorry

end largest_expr_is_expr1_l1232_123219


namespace total_amount_invested_l1232_123229

theorem total_amount_invested (x y : ℝ) (hx : 0.06 * x = 0.05 * y + 160) (hy : 0.05 * y = 6000) :
  x + y = 222666.67 :=
by
  sorry

end total_amount_invested_l1232_123229


namespace find_y_l1232_123299

theorem find_y (AB BC : ℕ) (y x : ℕ) 
  (h1 : AB = 3 * y)
  (h2 : BC = 2 * x)
  (h3 : AB * BC = 2400) 
  (h4 : AB * BC = 6 * x * y) :
  y = 20 := by
  sorry

end find_y_l1232_123299


namespace geometric_sum_n_eq_3_l1232_123290

theorem geometric_sum_n_eq_3 :
  (∃ n : ℕ, (1 / 2) * (1 - (1 / 3) ^ n) = 728 / 2187) ↔ n = 3 :=
by
  sorry

end geometric_sum_n_eq_3_l1232_123290


namespace ellipse_perimeter_l1232_123271

noncomputable def perimeter_of_triangle (a b : ℝ) (e : ℝ) : ℝ :=
  if (b = 4 ∧ e = 3 / 5 ∧ a = b / (1 - e^2) ^ (1 / 2))
  then 4 * a
  else 0

theorem ellipse_perimeter :
  let a : ℝ := 5
  let b : ℝ := 4
  let e : ℝ := 3 / 5
  4 * a = 20 :=
by
  sorry

end ellipse_perimeter_l1232_123271


namespace brianne_yard_length_l1232_123222

theorem brianne_yard_length 
  (derrick_yard_length : ℝ)
  (h₁ : derrick_yard_length = 10)
  (alex_yard_length : ℝ)
  (h₂ : alex_yard_length = derrick_yard_length / 2)
  (brianne_yard_length : ℝ)
  (h₃ : brianne_yard_length = 6 * alex_yard_length) :
  brianne_yard_length = 30 :=
by sorry

end brianne_yard_length_l1232_123222


namespace repeated_root_value_l1232_123285

theorem repeated_root_value (m : ℝ) :
  (∃ x : ℝ, x ≠ 1 ∧ (2 / (x - 1) + 3 = m / (x - 1)) ∧ 
            ∀ y : ℝ, y ≠ 1 ∧ (2 / (y - 1) + 3 = m / (y - 1)) → y = x) →
  m = 2 :=
by
  sorry

end repeated_root_value_l1232_123285


namespace lowest_possible_students_l1232_123243

theorem lowest_possible_students :
  ∃ n : ℕ, (n % 10 = 0 ∧ n % 24 = 0) ∧ n = 120 :=
by
  sorry

end lowest_possible_students_l1232_123243


namespace line_passes_through_fixed_point_minimal_triangle_area_eq_line_l1232_123224

-- Part (1)
theorem line_passes_through_fixed_point (m : ℝ) :
  ∃ M : ℝ × ℝ, M = (-1, -2) ∧
    (∀ m : ℝ, (2 + m) * (-1) + (1 - 2 * m) * (-2) + (4 - 3 * m) = 0) := by
  sorry

-- Part (2)
theorem minimal_triangle_area_eq_line :
  ∃ k : ℝ, k = -2 ∧ 
    (∀ x y : ℝ, y = k * (x + 1) - 2 ↔ y = 2 * x + 4) := by
  sorry

end line_passes_through_fixed_point_minimal_triangle_area_eq_line_l1232_123224


namespace sum_f_positive_l1232_123215

variable (a b c : ℝ)

def f (x : ℝ) := x^3 + x

theorem sum_f_positive (h1 : a + b > 0) (h2 : a + c > 0) (h3 : b + c > 0) :
  f a + f b + f c > 0 :=
sorry

end sum_f_positive_l1232_123215


namespace trig_identity_product_l1232_123284

theorem trig_identity_product :
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) * 
  (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12)) = 1 / 16 :=
by
  sorry

end trig_identity_product_l1232_123284


namespace find_number_l1232_123270

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l1232_123270


namespace correctness_of_statements_l1232_123212

theorem correctness_of_statements (p q : Prop) (x y : ℝ) : 
  (¬ (p ∧ q) → (p ∨ q)) ∧
  ((xy = 0) → ¬(x^2 + y^2 = 0)) ∧
  ¬(∀ (L P : ℝ → ℝ), (∃ x, L x = P x) ↔ (∃ x, L x = P x ∧ ∀ x₁ x₂, x₁ ≠ x₂ → L x₁ ≠ P x₂)) →
  (0 + 1 + 0 = 1) :=
by
  sorry

end correctness_of_statements_l1232_123212


namespace smallest_k_for_g_l1232_123231

theorem smallest_k_for_g (k : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x + k = -3) ↔ k ≤ -3/4 := sorry

end smallest_k_for_g_l1232_123231


namespace total_students_l1232_123286

-- Define the condition that the sum of boys (75) and girls (G) is the total number of students (T)
def sum_boys_girls (G T : ℕ) := 75 + G = T

-- Define the condition that the number of girls (G) equals 75% of the total number of students (T)
def girls_percentage (G T : ℕ) := G = Nat.div (3 * T) 4

-- State the theorem that given the above conditions, the total number of students (T) is 300
theorem total_students (G T : ℕ) (h1 : sum_boys_girls G T) (h2 : girls_percentage G T) : T = 300 := 
sorry

end total_students_l1232_123286


namespace probability_A_more_than_B_sum_m_n_l1232_123209

noncomputable def prob_A_more_than_B : ℚ :=
  0.6 + 0.4 * (1 / 2) * (1 - (63 / 512))

theorem probability_A_more_than_B : prob_A_more_than_B = 779 / 1024 := sorry

theorem sum_m_n : 779 + 1024 = 1803 := sorry

end probability_A_more_than_B_sum_m_n_l1232_123209


namespace least_possible_c_l1232_123201

theorem least_possible_c 
  (a b c : ℕ) 
  (h_avg : (a + b + c) / 3 = 20)
  (h_median : b = a + 13)
  (h_ord : a ≤ b ∧ b ≤ c)
  : c = 45 :=
sorry

end least_possible_c_l1232_123201


namespace sum_first_4_terms_l1232_123217

-- Define the sequence and its properties
def a (n : ℕ) : ℝ := sorry   -- The actual definition will be derived based on n, a_1, and q
def S (n : ℕ) : ℝ := sorry   -- The sum of the first n terms, also will be derived

-- Define the initial sequence properties based on the given conditions
axiom h1 : 0 < a 1  -- The sequence is positive
axiom h2 : a 4 * a 6 = 1 / 4
axiom h3 : a 7 = 1 / 8

-- The goal is to prove the sum of the first 4 terms equals 15
theorem sum_first_4_terms : S 4 = 15 := by
  sorry

end sum_first_4_terms_l1232_123217


namespace move_point_right_l1232_123259

theorem move_point_right (A B : ℤ) (hA : A = -3) (hAB : B = A + 4) : B = 1 :=
by {
  sorry
}

end move_point_right_l1232_123259


namespace percentage_of_original_price_l1232_123216
-- Define the original price and current price in terms of real numbers
def original_price : ℝ := 25
def current_price : ℝ := 20

-- Lean statement to verify the correctness of the percentage calculation
theorem percentage_of_original_price :
  (current_price / original_price) * 100 = 80 := 
by
  sorry

end percentage_of_original_price_l1232_123216


namespace total_cards_given_away_l1232_123297

-- Define the conditions in Lean
def Jim_initial_cards : ℕ := 365
def sets_given_to_brother : ℕ := 8
def sets_given_to_sister : ℕ := 5
def sets_given_to_friend : ℕ := 2
def cards_per_set : ℕ := 13

-- Define a theorem to prove the total number of cards given away
theorem total_cards_given_away : 
  sets_given_to_brother + sets_given_to_sister + sets_given_to_friend = 15 ∧
  15 * cards_per_set = 195 := 
by
  sorry

end total_cards_given_away_l1232_123297


namespace negation_exists_ge_zero_l1232_123260

theorem negation_exists_ge_zero (h : ∀ x > 0, x^2 - 3 * x + 2 < 0) :
  ∃ x > 0, x^2 - 3 * x + 2 ≥ 0 :=
sorry

end negation_exists_ge_zero_l1232_123260


namespace amount_with_r_l1232_123210

theorem amount_with_r (p q r : ℝ) (h₁ : p + q + r = 7000) (h₂ : r = (2 / 3) * (p + q)) : r = 2800 :=
  sorry

end amount_with_r_l1232_123210


namespace sin_210_eq_neg_half_l1232_123283

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 :=
by 
  sorry

end sin_210_eq_neg_half_l1232_123283


namespace park_area_calculation_l1232_123246

def scale := 300 -- miles per inch
def short_diagonal := 10 -- inches
def real_length := short_diagonal * scale -- miles
def park_area := (1/2) * real_length * real_length -- square miles

theorem park_area_calculation : park_area = 4500000 := by
  sorry

end park_area_calculation_l1232_123246


namespace part1_part2_part3_l1232_123296

-- Given conditions and definitions
def A : ℝ := 1
def B : ℝ := 3
def y1 : ℝ := sorry  -- simply a placeholder value as y1 == y2
def y2 : ℝ := y1
def y (x m n : ℝ) : ℝ := x^2 + m * x + n

-- (1) Proof of m = -4
theorem part1 (n : ℝ) (h1 : y A m n = y1) (h2 : y B m n = y2) : m = -4 := sorry

-- (2) Proof of n = 4 when the parabola intersects the x-axis at one point
theorem part2 (h : ∃ n, ∀ x : ℝ, y x (-4) n = 0 → x = (x - 2)^2) : n = 4 := sorry

-- (3) Proof of the range of real number values for a
theorem part3 (a : ℝ) (b1 b2 : ℝ) (n : ℝ) (h1 : y a (-4) n = b1) 
  (h2 : y B (-4) n = b2) (h3 : b1 > b2) : a < 1 ∨ a > 3 := sorry

end part1_part2_part3_l1232_123296


namespace ratio_condition_equivalence_l1232_123255

variable (a b c d : ℝ)

theorem ratio_condition_equivalence
  (h : (2 * a + 3 * b) / (b + 2 * c) = (3 * c + 2 * d) / (d + 2 * a)) :
  2 * a = 3 * c ∨ 2 * a + 3 * b + d + 2 * c = 0 :=
by
  sorry

end ratio_condition_equivalence_l1232_123255


namespace player1_winning_strategy_l1232_123264

/--
Player 1 has a winning strategy if and only if N is not an odd power of 2,
under the game rules where players alternately subtract proper divisors
and a player loses when given a prime number or 1.
-/
theorem player1_winning_strategy (N: ℕ) : 
  ¬ (∃ k: ℕ, k % 2 = 1 ∧ N = 2^k) ↔ (∃ strategy: ℕ → ℕ, ∀ n ≠ 1, n ≠ prime → n - strategy n = m) :=
sorry

end player1_winning_strategy_l1232_123264


namespace find_eighth_number_l1232_123206

def average_of_numbers (a b c d e f g h x : ℕ) : ℕ :=
  (a + b + c + d + e + f + g + h + x) / 9

theorem find_eighth_number (a b c d e f g h x : ℕ) (avg : ℕ) 
    (h_avg : average_of_numbers a b c d e f g h x = avg)
    (h_total_sum : a + b + c + d + e + f + g + h + x = 540)
    (h_x_val : x = 65) : a = 53 :=
by
  sorry

end find_eighth_number_l1232_123206


namespace algebraic_expression_value_l1232_123235

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2 * x - 2 = 0) :
  x * (x + 2) + (x + 1)^2 = 5 :=
by
  sorry

end algebraic_expression_value_l1232_123235


namespace eq_b_minus_a_l1232_123266

   -- Definition for rotating a point counterclockwise by 180° around another point
   def rotate_180 (h k x y : ℝ) : ℝ × ℝ :=
     (2 * h - x, 2 * k - y)

   -- Definition for reflecting a point about the line y = -x
   def reflect_y_eq_neg_x (x y : ℝ) : ℝ × ℝ :=
     (-y, -x)

   -- Given point Q(a, b)
   variables (a b : ℝ)

   -- Image of Q after the transformations
   def Q_transformed :=
     (5, -1)

   -- Image of Q after reflection about y = -x
   def Q_reflected :=
     reflect_y_eq_neg_x (5) (-1)

   -- Image of Q after 180° rotation around (2,3)
   def Q_original :=
     rotate_180 (2) (3) a b

   -- Statement we want to prove:
   theorem eq_b_minus_a : b - a = 6 :=
   by
     -- Calculation steps
     sorry
   
end eq_b_minus_a_l1232_123266


namespace relationship_between_m_and_n_l1232_123228

theorem relationship_between_m_and_n
  (b m n : ℝ)
  (h₁ : m = 2 * (-1 / 2) + b)
  (h₂ : n = 2 * 2 + b) :
  m < n :=
by
  sorry

end relationship_between_m_and_n_l1232_123228


namespace cost_of_first_house_l1232_123249

theorem cost_of_first_house (C : ℝ) (h₀ : 2 * C + C = 600000) : C = 200000 := by
  -- proof placeholder
  sorry

end cost_of_first_house_l1232_123249


namespace solve_problem_l1232_123250

noncomputable def problem_statement : Prop :=
  ∀ (tons_to_pounds : ℕ) 
    (packet_weight_pounds : ℕ) 
    (packet_weight_ounces : ℕ)
    (num_packets : ℕ)
    (bag_capacity_tons : ℕ)
    (X : ℕ),
    tons_to_pounds = 2300 →
    packet_weight_pounds = 16 →
    packet_weight_ounces = 4 →
    num_packets = 1840 →
    bag_capacity_tons = 13 →
    X = (packet_weight_ounces * bag_capacity_tons * tons_to_pounds) / 
        ((bag_capacity_tons * tons_to_pounds) - (num_packets * packet_weight_pounds)) →
    X = 16

theorem solve_problem : problem_statement :=
by sorry

end solve_problem_l1232_123250


namespace find_x_l1232_123265

theorem find_x (x : ℤ) (h : (2 * x + 7) / 5 = 22) : x = 103 / 2 :=
by
  sorry

end find_x_l1232_123265


namespace sin_neg_045_unique_solution_l1232_123263

theorem sin_neg_045_unique_solution (x : ℝ) (hx : 0 ≤ x ∧ x < 180) (h: ℝ) :
  (h = Real.sin x → h = -0.45) → 
  ∃! x, 0 ≤ x ∧ x < 180 ∧ Real.sin x = -0.45 :=
by sorry

end sin_neg_045_unique_solution_l1232_123263


namespace students_bought_pencils_l1232_123252

theorem students_bought_pencils (h1 : 2 * 2 + 6 * 3 + 2 * 1 = 24) : 
  2 + 6 + 2 = 10 := by
  sorry

end students_bought_pencils_l1232_123252


namespace probability_prime_sum_l1232_123272

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def possible_outcomes : ℕ := 48

def prime_sums : Finset ℕ := {2, 3, 5, 7, 11, 13}

def prime_count : ℕ := 19

theorem probability_prime_sum :
  ((prime_count : ℚ) / possible_outcomes) = 19 / 48 := 
by
  sorry

end probability_prime_sum_l1232_123272


namespace function_machine_output_l1232_123247

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 25 then step1 - 7 else step1 + 10
  step2

theorem function_machine_output : function_machine 15 = 38 :=
by
  sorry

end function_machine_output_l1232_123247


namespace coefficients_sum_eq_four_l1232_123238

noncomputable def simplified_coefficients_sum (y : ℚ → ℚ) : ℚ :=
  let A := 1
  let B := 3
  let C := 2
  let D := -2
  A + B + C + D

theorem coefficients_sum_eq_four : simplified_coefficients_sum (λ x => 
  (x^3 + 5*x^2 + 8*x + 4) / (x + 2)) = 4 := by
  sorry

end coefficients_sum_eq_four_l1232_123238


namespace not_divisible_by_15_l1232_123241

theorem not_divisible_by_15 (a : ℤ) : ¬ (15 ∣ (a^2 + a + 2)) :=
by
  sorry

end not_divisible_by_15_l1232_123241


namespace count_less_than_threshold_is_zero_l1232_123245

def numbers := [0.8, 0.5, 0.9]
def threshold := 0.4

theorem count_less_than_threshold_is_zero :
  (numbers.filter (λ x => x < threshold)).length = 0 :=
by
  sorry

end count_less_than_threshold_is_zero_l1232_123245


namespace flute_player_count_l1232_123236

-- Define the total number of people in the orchestra
def total_people : Nat := 21

-- Define the number of people in each section
def sebastian : Nat := 1
def brass : Nat := 4 + 2 + 1
def strings : Nat := 3 + 1 + 1
def woodwinds_excluding_flutes : Nat := 3
def maestro : Nat := 1

-- Calculate the number of accounted people
def accounted_people : Nat := sebastian + brass + strings + woodwinds_excluding_flutes + maestro

-- State the number of flute players
def flute_players : Nat := total_people - accounted_people

-- The theorem stating the number of flute players
theorem flute_player_count : flute_players = 4 := by
  unfold flute_players accounted_people total_people sebastian brass strings woodwinds_excluding_flutes maestro
  -- Need to evaluate the expressions step by step to reach the final number 4.
  -- (Or simply "sorry" since we are skipping the proof steps)
  sorry

end flute_player_count_l1232_123236


namespace finite_solutions_to_equation_l1232_123293

theorem finite_solutions_to_equation :
  ∃ (n : ℕ), ∀ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) ∧ (1 / (a:ℚ) + 1 / (b:ℚ) + 1 / (c:ℚ) = 1 / 1983) → 
  (a ≤ n ∧ b ≤ n ∧ c ≤ n) :=
sorry

end finite_solutions_to_equation_l1232_123293


namespace triangle_area_less_than_sqrt3_div_3_l1232_123291

-- Definitions for a triangle and its properties
structure Triangle :=
  (a b c : ℝ)
  (ha hb hc : ℝ)
  (area : ℝ)

def valid_triangle (Δ : Triangle) : Prop :=
  0 < Δ.a ∧ 0 < Δ.b ∧ 0 < Δ.c ∧ Δ.ha < 1 ∧ Δ.hb < 1 ∧ Δ.hc < 1

theorem triangle_area_less_than_sqrt3_div_3 (Δ : Triangle) (h : valid_triangle Δ) : Δ.area < (Real.sqrt 3) / 3 :=
sorry

end triangle_area_less_than_sqrt3_div_3_l1232_123291


namespace true_compound_proposition_l1232_123289

-- Define conditions and propositions in Lean
def proposition_p : Prop := ∃ (x : ℝ), x^2 + x + 1 < 0
def proposition_q : Prop := ∀ (x : ℝ), 1 ≤ x → x ≤ 2 → x^2 - 1 ≥ 0

-- Define the compound proposition
def correct_proposition : Prop := ¬ proposition_p ∧ proposition_q

-- Prove the correct compound proposition
theorem true_compound_proposition : correct_proposition :=
by
  sorry

end true_compound_proposition_l1232_123289


namespace intersection_M_complement_N_l1232_123268

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def N : Set ℝ := {x | ∃ y : ℝ, y = 3*x^2 + 1 }

def complement_N : Set ℝ := {x | ¬ ∃ y : ℝ, y = 3*x^2 + 1}

theorem intersection_M_complement_N :
  (M ∩ complement_N) = {x | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_complement_N_l1232_123268


namespace counterexample_conjecture_l1232_123269

theorem counterexample_conjecture 
    (odd_gt_5 : ℕ → Prop) 
    (is_prime : ℕ → Prop) 
    (conjecture : ∀ n, odd_gt_5 n → ∃ p1 p2 p3, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ n = p1 + p2 + p3) : 
    ∃ n, odd_gt_5 n ∧ ¬ (∃ p1 p2 p3, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ n = p1 + p2 + p3) :=
sorry

end counterexample_conjecture_l1232_123269


namespace paving_stones_needed_l1232_123282

variables (length_courtyard width_courtyard num_paving_stones length_paving_stone area_courtyard area_paving_stone : ℝ)
noncomputable def width_paving_stone := 2

theorem paving_stones_needed : 
  length_courtyard = 60 → 
  width_courtyard = 14 → 
  num_paving_stones = 140 →
  length_paving_stone = 3 →
  area_courtyard = length_courtyard * width_courtyard →
  area_paving_stone = length_paving_stone * width_paving_stone →
  num_paving_stones = area_courtyard / area_paving_stone :=
by
  intros h_length_courtyard h_width_courtyard h_num_paving_stones h_length_paving_stone h_area_courtyard h_area_paving_stone
  rw [h_length_courtyard, h_width_courtyard, h_length_paving_stone] at *
  simp at *
  sorry

end paving_stones_needed_l1232_123282


namespace graveling_cost_is_969_l1232_123220

-- Definitions for lawn dimensions
def lawn_length : ℝ := 75
def lawn_breadth : ℝ := 45

-- Definitions for road widths and costs
def road1_width : ℝ := 6
def road1_cost_per_sq_meter : ℝ := 0.90

def road2_width : ℝ := 5
def road2_cost_per_sq_meter : ℝ := 0.85

def road3_width : ℝ := 4
def road3_cost_per_sq_meter : ℝ := 0.80

def road4_width : ℝ := 3
def road4_cost_per_sq_meter : ℝ := 0.75

-- Calculate the area of each road
def road1_area : ℝ := road1_width * lawn_length
def road2_area : ℝ := road2_width * lawn_length
def road3_area : ℝ := road3_width * lawn_breadth
def road4_area : ℝ := road4_width * lawn_breadth

-- Calculate the cost of graveling each road
def road1_graveling_cost : ℝ := road1_area * road1_cost_per_sq_meter
def road2_graveling_cost : ℝ := road2_area * road2_cost_per_sq_meter
def road3_graveling_cost : ℝ := road3_area * road3_cost_per_sq_meter
def road4_graveling_cost : ℝ := road4_area * road4_cost_per_sq_meter

-- Calculate the total cost
def total_graveling_cost : ℝ := 
  road1_graveling_cost + road2_graveling_cost + road3_graveling_cost + road4_graveling_cost

-- Statement to be proved
theorem graveling_cost_is_969 : total_graveling_cost = 969 := by
  sorry

end graveling_cost_is_969_l1232_123220


namespace ceil_sqrt_225_l1232_123244

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end ceil_sqrt_225_l1232_123244


namespace max_angle_position_l1232_123204

-- Definitions for points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

-- Definitions for points A and B on the X-axis
def A (a : ℝ) : Point := { x := -a, y := 0 }
def B (a : ℝ) : Point := { x := a, y := 0 }

-- Definition for point C moving along the line y = 10 - x
def moves_along_line (C : Point) : Prop :=
  C.y = 10 - C.x

-- Definition for calculating the angle ACB (gamma)
def angle_ACB (A B C : Point) : ℝ := sorry -- The detailed function to calculate angle is omitted for brevity

-- Main statement to prove
theorem max_angle_position (a : ℝ) (C : Point) (ha : 0 ≤ a ∧ a ≤ 10) (hC : moves_along_line C) :
  (C = { x := 4, y := 6 } ∨ C = { x := 16, y := -6 }) ↔ (∀ C', moves_along_line C' → (angle_ACB (A a) (B a) C') ≤ angle_ACB (A a) (B a) C) :=
sorry

end max_angle_position_l1232_123204


namespace evaluate_at_two_l1232_123227

def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 1

theorem evaluate_at_two : f 2 = 15 :=
by
  sorry

end evaluate_at_two_l1232_123227


namespace fraction_addition_l1232_123256

theorem fraction_addition (a b : ℚ) (h : a / b = 1 / 3) : (a + b) / b = 4 / 3 := by
  sorry

end fraction_addition_l1232_123256


namespace reversed_digits_sum_l1232_123213

theorem reversed_digits_sum (a b n : ℕ) (x y : ℕ) (ha : a < 10) (hb : b < 10) 
(hx : x = 10 * a + b) (hy : y = 10 * b + a) (hsq : x^2 + y^2 = n^2) : 
  x + y + n = 264 :=
sorry

end reversed_digits_sum_l1232_123213


namespace quadratic_roots_l1232_123202

theorem quadratic_roots {α p q : ℝ} (hα : 0 < α ∧ α ≤ 1) (hroots : ∃ x : ℝ, x^2 + p * x + q = 0) :
  ∃ x : ℝ, α * x^2 + p * x + q = 0 :=
by sorry

end quadratic_roots_l1232_123202


namespace union_sets_l1232_123298

open Set

def setM : Set ℝ := {x : ℝ | x^2 < x}
def setN : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}

theorem union_sets : setM ∪ setN = {x : ℝ | -3 < x ∧ x < 1} :=
by
  sorry

end union_sets_l1232_123298


namespace polynomial_value_l1232_123223

def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_value (a b c d : ℝ) 
  (h1 : P 1 a b c d = 1993) 
  (h2 : P 2 a b c d = 3986) 
  (h3 : P 3 a b c d = 5979) :
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 := 
by 
  sorry

end polynomial_value_l1232_123223


namespace set_C_is_pythagorean_triple_l1232_123281

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem set_C_is_pythagorean_triple : is_pythagorean_triple 5 12 13 :=
sorry

end set_C_is_pythagorean_triple_l1232_123281


namespace seokjin_paper_count_l1232_123274

theorem seokjin_paper_count :
  ∀ (jimin_paper seokjin_paper : ℕ),
  jimin_paper = 41 →
  jimin_paper = seokjin_paper + 1 →
  seokjin_paper = 40 :=
by
  intros jimin_paper seokjin_paper h_jimin h_relation
  sorry

end seokjin_paper_count_l1232_123274


namespace number_of_triangles_l1232_123207

theorem number_of_triangles (x : ℕ) (h₁ : 2 + x > 6) (h₂ : 8 > x) : ∃! t, t = 3 :=
by {
  sorry
}

end number_of_triangles_l1232_123207


namespace find_line_equation_l1232_123257

-- Definition of a line passing through a point
def passes_through (l : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := l p.1 p.2

-- Definition of intercepts being opposite
def opposite_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ l a 0 ∧ l 0 (-a)

-- The line passing through the point (7, 1)
def line_exists (l : ℝ → ℝ → Prop) : Prop :=
  passes_through l (7, 1) ∧ opposite_intercepts l

-- Main theorem to prove the equation of the line
theorem find_line_equation (l : ℝ → ℝ → Prop) :
  line_exists l ↔ (∀ x y, l x y ↔ x - 7 * y = 0) ∨ (∀ x y, l x y ↔ x - y - 6 = 0) :=
sorry

end find_line_equation_l1232_123257


namespace probability_Z_l1232_123288

theorem probability_Z (p_X p_Y p_Z : ℚ)
  (hX : p_X = 2 / 5)
  (hY : p_Y = 1 / 4)
  (hTotal : p_X + p_Y + p_Z = 1) :
  p_Z = 7 / 20 := by sorry

end probability_Z_l1232_123288
