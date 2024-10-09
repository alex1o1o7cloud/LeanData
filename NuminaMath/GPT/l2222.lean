import Mathlib

namespace max_k_value_l2222_222279

theorem max_k_value (x y : ℝ) (k : ℝ) (hx : 0 < x) (hy : 0 < y)
(h : 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + 2 * k * (x / y + y / x)) :
  k ≤ Real.sqrt (5 / 6) := sorry

end max_k_value_l2222_222279


namespace nonnegative_integer_solutions_l2222_222269

theorem nonnegative_integer_solutions (x : ℕ) (h : 1 + x ≥ 2 * x - 1) : x = 0 ∨ x = 1 ∨ x = 2 :=
by
  sorry

end nonnegative_integer_solutions_l2222_222269


namespace sum_three_positive_numbers_ge_three_l2222_222205

theorem sum_three_positive_numbers_ge_three 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_eq : (a + 1) * (b + 1) * (c + 1) = 8) : 
  a + b + c ≥ 3 :=
sorry

end sum_three_positive_numbers_ge_three_l2222_222205


namespace value_of_a_l2222_222237

noncomputable def a : ℕ := 4

def A : Set ℕ := {0, 2, a}
def B : Set ℕ := {1, a*a}
def C : Set ℕ := {0, 1, 2, 4, 16}

theorem value_of_a : A ∪ B = C → a = 4 := by
  intro h
  sorry

end value_of_a_l2222_222237


namespace range_of_xy_l2222_222241

-- Given conditions
variables {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : 2 / x + 8 / y = 1)

-- To Prove
theorem range_of_xy (hx : x > 0) (hy : y > 0) (hxy : 2 / x + 8 / y = 1) : 64 ≤ x * y :=
sorry

end range_of_xy_l2222_222241


namespace solve_inequality_l2222_222249

theorem solve_inequality (x : ℝ) (hx : x ≥ 0) :
  2021 * (x ^ (2020/202)) - 1 ≥ 2020 * x ↔ x = 1 :=
by sorry

end solve_inequality_l2222_222249


namespace sum_of_first_six_terms_l2222_222258

def geometric_seq_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem sum_of_first_six_terms (a : ℕ) (r : ℕ) (h1 : r = 2) (h2 : a * (1 + r + r^2) = 3) :
  geometric_seq_sum a r 6 = 27 :=
by
  sorry

end sum_of_first_six_terms_l2222_222258


namespace a_is_multiple_of_2_l2222_222257

theorem a_is_multiple_of_2 (a : ℕ) (h1 : 0 < a) (h2 : (4 ^ a) % 10 = 6) : a % 2 = 0 :=
sorry

end a_is_multiple_of_2_l2222_222257


namespace find_a_plus_b_l2222_222246

theorem find_a_plus_b (a b : ℝ)
  (h1 : ab^2 = 0)
  (h2 : 2 * a^2 * b = 0)
  (h3 : a^3 + b^2 = 0)
  (h4 : ab = 1) : a + b = -2 :=
sorry

end find_a_plus_b_l2222_222246


namespace determine_a_l2222_222266

theorem determine_a (a : ℝ) (f : ℝ → ℝ) (h : f = fun x => a * x^3 - 2 * x) (pt : f (-1) = 4) : a = -2 := by
  sorry

end determine_a_l2222_222266


namespace find_k_exact_one_real_solution_l2222_222286

theorem find_k_exact_one_real_solution (k : ℝ) :
  (∀ x : ℝ, (3*x + 6)*(x - 4) = -33 + k*x) ↔ (k = -6 + 6*Real.sqrt 3 ∨ k = -6 - 6*Real.sqrt 3) := 
by
  sorry

end find_k_exact_one_real_solution_l2222_222286


namespace arithmetic_sequence_second_term_l2222_222250

theorem arithmetic_sequence_second_term (a d : ℤ)
  (h1 : a + 11 * d = 11)
  (h2 : a + 12 * d = 14) :
  a + d = -19 :=
sorry

end arithmetic_sequence_second_term_l2222_222250


namespace cos_B_eq_zero_l2222_222268

variable {a b c A B C : ℝ}
variable (h1 : ∀ A B C, 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
variable (h2 : b * Real.cos A = c)

theorem cos_B_eq_zero (h1 : a = b) (h2 : b * Real.cos A = c) : Real.cos B = 0 :=
sorry

end cos_B_eq_zero_l2222_222268


namespace solution_set_of_inequality_l2222_222217

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3 * x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l2222_222217


namespace fundraiser_contribution_l2222_222203

theorem fundraiser_contribution :
  let sasha_muffins := 30
  let melissa_muffins := 4 * sasha_muffins
  let tiffany_muffins := (sasha_muffins + melissa_muffins) / 2
  let total_muffins := sasha_muffins + melissa_muffins + tiffany_muffins
  let price_per_muffin := 4
  total_muffins * price_per_muffin = 900 :=
by
  let sasha_muffins := 30
  let melissa_muffins := 4 * sasha_muffins
  let tiffany_muffins := (sasha_muffins + melissa_muffins) / 2
  let total_muffins := sasha_muffins + melissa_muffins + tiffany_muffins
  let price_per_muffin := 4
  sorry

end fundraiser_contribution_l2222_222203


namespace average_eq_5_times_non_zero_l2222_222204

theorem average_eq_5_times_non_zero (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5 * x) : x = 9 := 
by sorry

end average_eq_5_times_non_zero_l2222_222204


namespace find_k_plus_m_l2222_222216

def initial_sum := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
def initial_count := 9

def new_list_sum (m k : ℕ) := initial_sum + 8 * m + 9 * k
def new_list_count (m k : ℕ) := initial_count + m + k

def average_eq_73 (m k : ℕ) := (new_list_sum m k : ℝ) / (new_list_count m k : ℝ) = 7.3

theorem find_k_plus_m : ∃ (m k : ℕ), average_eq_73 m k ∧ (k + m = 21) :=
by
  sorry

end find_k_plus_m_l2222_222216


namespace range_of_a_l2222_222290

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x - 12| < 6 → False) → (a ≤ 6 ∨ a ≥ 18) :=
by 
  intro h
  sorry

end range_of_a_l2222_222290


namespace min_sum_arth_seq_l2222_222295

theorem min_sum_arth_seq (a : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ k, a k = a 1 + (k - 1) * (a 2 - a 1))
  (h2 : a 1 = -3)
  (h3 : 11 * a 5 = 5 * a 8) : n = 4 := by
  sorry

end min_sum_arth_seq_l2222_222295


namespace unique_handshakes_l2222_222281

theorem unique_handshakes :
  let twins_sets := 12
  let triplets_sets := 3
  let twins := twins_sets * 2
  let triplets := triplets_sets * 3
  let twin_shakes_twins := twins * (twins - 2)
  let triplet_shakes_triplets := triplets * (triplets - 3)
  let twin_shakes_triplets := twins * (triplets / 3)
  (twin_shakes_twins + triplet_shakes_triplets + twin_shakes_triplets) / 2 = 327 := by
  sorry

end unique_handshakes_l2222_222281


namespace rhind_papyrus_prob_l2222_222201

theorem rhind_papyrus_prob (a₁ a₂ a₃ a₄ a₅ : ℝ) (q : ℝ) 
  (h_geom_seq : a₂ = a₁ * q ∧ a₃ = a₁ * q^2 ∧ a₄ = a₁ * q^3 ∧ a₅ = a₁ * q^4)
  (h_loaves_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 93)
  (h_condition : a₁ + a₂ = (3/4) * a₃) 
  (q_gt_one : q > 1) :
  a₃ = 12 :=
sorry

end rhind_papyrus_prob_l2222_222201


namespace unique_function_satisfying_conditions_l2222_222208

open Nat

def satisfies_conditions (f : ℕ → ℕ) : Prop :=
  (f 1 = 1) ∧ (∀ n, f n * f (n + 2) = (f (n + 1))^2 + 1997)

theorem unique_function_satisfying_conditions :
  (∃! f : ℕ → ℕ, satisfies_conditions f) :=
sorry

end unique_function_satisfying_conditions_l2222_222208


namespace initial_students_l2222_222223

variable (n : ℝ) (W : ℝ)

theorem initial_students 
  (h1 : W = n * 15)
  (h2 : W + 11 = (n + 1) * 14.8)
  (h3 : 15 * n + 11 = 14.8 * n + 14.8)
  (h4 : 0.2 * n = 3.8) :
  n = 19 :=
sorry

end initial_students_l2222_222223


namespace product_of_numbers_is_86_l2222_222234

-- Definitions of the two conditions
def sum_eq_24 (x y : ℝ) : Prop := x + y = 24
def sum_of_squares_eq_404 (x y : ℝ) : Prop := x^2 + y^2 = 404

-- The theorem to prove the product of the two numbers
theorem product_of_numbers_is_86 (x y : ℝ) (h1 : sum_eq_24 x y) (h2 : sum_of_squares_eq_404 x y) : x * y = 86 :=
  sorry

end product_of_numbers_is_86_l2222_222234


namespace subcommittee_count_l2222_222299

theorem subcommittee_count :
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose republicans subcommittee_republicans * choose democrats subcommittee_democrats = 11760 :=
by
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  sorry

end subcommittee_count_l2222_222299


namespace cost_of_sneakers_l2222_222226

theorem cost_of_sneakers (saved money per_action_figure final_money cost : ℤ) 
  (h1 : saved = 15) 
  (h2 : money = 10) 
  (h3 : per_action_figure = 10) 
  (h4 : final_money = 25) 
  (h5 : money * per_action_figure + saved - cost = final_money) 
  : cost = 90 := 
sorry

end cost_of_sneakers_l2222_222226


namespace full_price_tickets_revenue_l2222_222255

theorem full_price_tickets_revenue (f h p : ℕ) (h1 : f + h + 12 = 160) (h2 : f * p + h * (p / 2) + 12 * (2 * p) = 2514) :  f * p = 770 := 
sorry

end full_price_tickets_revenue_l2222_222255


namespace cube_volume_surface_area_l2222_222213

theorem cube_volume_surface_area (x : ℝ) (s : ℝ)
  (h1 : s^3 = 3 * x)
  (h2 : 6 * s^2 = 6 * x) :
  x = 3 :=
by sorry

end cube_volume_surface_area_l2222_222213


namespace max_alpha_value_l2222_222200

variable (a b x y α : ℝ)

theorem max_alpha_value (h1 : a = 2 * b)
    (h2 : a^2 + y^2 = b^2 + x^2)
    (h3 : b^2 + x^2 = (a - x)^2 + (b - y)^2)
    (h4 : 0 ≤ x) (h5 : x < a) (h6 : 0 ≤ y) (h7 : y < b) :
    α = a / b → α^2 = 4 := 
by
  sorry

end max_alpha_value_l2222_222200


namespace remainder_x_squared_l2222_222287

theorem remainder_x_squared (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 7 * x ≡ 14 [ZMOD 20]) : 
  (x^2 ≡ 4 [ZMOD 20]) :=
sorry

end remainder_x_squared_l2222_222287


namespace short_trees_after_planting_l2222_222273

-- Define the current number of short trees
def current_short_trees : ℕ := 41

-- Define the number of short trees to be planted today
def new_short_trees : ℕ := 57

-- Define the expected total number of short trees after planting
def total_short_trees_after_planting : ℕ := 98

-- The theorem to prove that the total number of short trees after planting is as expected
theorem short_trees_after_planting :
  current_short_trees + new_short_trees = total_short_trees_after_planting :=
by
  -- Proof skipped using sorry
  sorry

end short_trees_after_planting_l2222_222273


namespace domain_of_f_l2222_222253

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 1)) / (x - 2)

theorem domain_of_f : {x : ℝ | x > -1 ∧ x ≠ 2} = {x : ℝ | x ∈ Set.Ioo (-1) 2 ∪ Set.Ioi 2} :=
by {
  sorry
}

end domain_of_f_l2222_222253


namespace nehas_mother_age_l2222_222264

variables (N M : ℕ)

axiom age_condition1 : M - 12 = 4 * (N - 12)
axiom age_condition2 : M + 12 = 2 * (N + 12)

theorem nehas_mother_age : M = 60 :=
by
  -- Sorry added to skip the proof
  sorry

end nehas_mother_age_l2222_222264


namespace xiaoming_climb_stairs_five_steps_l2222_222298

def count_ways_to_climb (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else count_ways_to_climb (n - 1) + count_ways_to_climb (n - 2)

theorem xiaoming_climb_stairs_five_steps :
  count_ways_to_climb 5 = 5 :=
by
  sorry

end xiaoming_climb_stairs_five_steps_l2222_222298


namespace income_max_takehome_pay_l2222_222265

theorem income_max_takehome_pay :
  ∃ x : ℝ, (∀ y : ℝ, 1000 * y - 5 * y^2 ≤ 1000 * x - 5 * x^2) ∧ x = 100 :=
by
  sorry

end income_max_takehome_pay_l2222_222265


namespace prob_yellow_and_straight_l2222_222238

-- Definitions of probabilities given in the problem
def prob_green : ℚ := 2 / 3
def prob_straight : ℚ := 1 / 2

-- Derived probability of picking a yellow flower
def prob_yellow : ℚ := 1 - prob_green

-- Statement to prove
theorem prob_yellow_and_straight : prob_yellow * prob_straight = 1 / 6 :=
by
  -- sorry is used here to skip the proof.
  sorry

end prob_yellow_and_straight_l2222_222238


namespace group_B_equal_l2222_222242

noncomputable def neg_two_pow_three := (-2)^3
noncomputable def minus_two_pow_three := -(2^3)

theorem group_B_equal : neg_two_pow_three = minus_two_pow_three :=
by sorry

end group_B_equal_l2222_222242


namespace math_problem_l2222_222274

theorem math_problem
  (a b c d : ℕ)
  (h1 : a = 234)
  (h2 : b = 205)
  (h3 : c = 86400)
  (h4 : d = 300) :
  (a * b = 47970) ∧ (c / d = 288) :=
by
  sorry

end math_problem_l2222_222274


namespace perimeter_T2_l2222_222256

def Triangle (a b c : ℝ) :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem perimeter_T2 (a b c : ℝ) (h : Triangle a b c) (ha : a = 10) (hb : b = 15) (hc : c = 20) : 
  let AM := a / 2
  let BN := b / 2
  let CP := c / 2
  0 < AM ∧ 0 < BN ∧ 0 < CP →
  AM + BN + CP = 22.5 :=
by
  sorry

end perimeter_T2_l2222_222256


namespace triangle_obtuse_l2222_222247

theorem triangle_obtuse 
  (A B : ℝ)
  (hA : 0 < A ∧ A < π/2)
  (hB : 0 < B ∧ B < π/2)
  (h_cosA_gt_sinB : Real.cos A > Real.sin B) :
  π - (A + B) > π/2 ∧ π - (A + B) < π :=
by
  sorry

end triangle_obtuse_l2222_222247


namespace problems_finished_equals_45_l2222_222245

/-- Mathematical constants and conditions -/
def ratio_finished_left (F L : ℕ) : Prop := F = 9 * (L / 4)
def total_problems (F L : ℕ) : Prop := F + L = 65

/-- Lean theorem to prove the problem statement -/
theorem problems_finished_equals_45 :
  ∃ F L : ℕ, ratio_finished_left F L ∧ total_problems F L ∧ F = 45 :=
by
  sorry

end problems_finished_equals_45_l2222_222245


namespace angle_RPS_is_27_l2222_222207

theorem angle_RPS_is_27 (PQ BP PR QS QS PSQ QPRS : ℝ) :
  PQ + PSQ + QS = 180 ∧ 
  QS = 48 ∧ 
  PSQ = 38 ∧ 
  QPRS = 67
  → (QS - QPRS = 27) := 
by {
  sorry
}

end angle_RPS_is_27_l2222_222207


namespace farm_horses_cows_ratio_l2222_222239

variable (x y : ℕ)  -- x is the base variable related to the initial counts, y is the number of horses sold (and cows bought)

theorem farm_horses_cows_ratio (h1 : 4 * x / x = 4)
    (h2 : 13 * (x + y) = 7 * (4 * x - y))
    (h3 : 4 * x - y = (x + y) + 30) :
    y = 15 := sorry

end farm_horses_cows_ratio_l2222_222239


namespace equal_chessboard_numbers_l2222_222211

theorem equal_chessboard_numbers (n : ℕ) (board : ℕ → ℕ → ℕ) 
  (mean_property : ∀ (x y : ℕ), board x y = (board (x-1) y + board (x+1) y + board x (y-1) + board x (y+1)) / 4) : 
  ∀ (x y : ℕ), board x y = board 0 0 :=
by
  -- Proof not required
  sorry

end equal_chessboard_numbers_l2222_222211


namespace enrollment_difference_l2222_222235

theorem enrollment_difference :
  let M := 1500
  let S := 2100
  let L := 2700
  let R := 1800
  let B := 900
  max M (max S (max L (max R B))) - min M (min S (min L (min R B))) = 1800 := 
by 
  sorry

end enrollment_difference_l2222_222235


namespace range_of_a_l2222_222243

variable {a : ℝ}

def A (a : ℝ) : Set ℝ := { x | (x - 2) * (x - (a + 1)) < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - 2 * a) / (x - (a^2 + 1)) < 0 }

theorem range_of_a (a : ℝ) : B a ⊆ A a ↔ (a = -1 / 2) ∨ (2 ≤ a ∧ a ≤ 3) := by
  sorry

end range_of_a_l2222_222243


namespace sugar_percentage_of_second_solution_l2222_222225

theorem sugar_percentage_of_second_solution :
  ∀ (W : ℝ) (P : ℝ),
  (0.10 * W * (3 / 4) + P / 100 * (1 / 4) * W = 0.18 * W) → 
  (P = 42) :=
by
  intros W P h
  sorry

end sugar_percentage_of_second_solution_l2222_222225


namespace exists_real_a_l2222_222230

noncomputable def A (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def C : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

theorem exists_real_a : ∃ a : ℝ, a = -2 ∧ A a ∩ C = ∅ ∧ ∅ ⊂ A a ∩ B := 
by {
  sorry
}

end exists_real_a_l2222_222230


namespace tomatoes_left_l2222_222227

theorem tomatoes_left (initial_tomatoes picked_yesterday picked_today : ℕ)
    (h_initial : initial_tomatoes = 171)
    (h_picked_yesterday : picked_yesterday = 134)
    (h_picked_today : picked_today = 30) :
    initial_tomatoes - picked_yesterday - picked_today = 7 :=
by
    sorry

end tomatoes_left_l2222_222227


namespace suitable_sampling_method_l2222_222292

theorem suitable_sampling_method 
  (seniorTeachers : ℕ)
  (intermediateTeachers : ℕ)
  (juniorTeachers : ℕ)
  (totalSample : ℕ)
  (totalTeachers : ℕ)
  (prob : ℚ)
  (seniorSample : ℕ)
  (intermediateSample : ℕ)
  (juniorSample : ℕ)
  (excludeOneSenior : ℕ) :
  seniorTeachers = 28 →
  intermediateTeachers = 54 →
  juniorTeachers = 81 →
  totalSample = 36 →
  excludeOneSenior = 27 →
  totalTeachers = excludeOneSenior + intermediateTeachers + juniorTeachers →
  prob = totalSample / totalTeachers →
  seniorSample = excludeOneSenior * prob →
  intermediateSample = intermediateTeachers * prob →
  juniorSample = juniorTeachers * prob →
  seniorSample + intermediateSample + juniorSample = totalSample :=
by
  intros hsenior hins hjunior htotal hexclude htotalTeachers hprob hseniorSample hintermediateSample hjuniorSample
  sorry

end suitable_sampling_method_l2222_222292


namespace range_of_k_l2222_222252

noncomputable def f (k : ℝ) (x : ℝ) := 1 - k * x^2
noncomputable def g (x : ℝ) := Real.cos x

theorem range_of_k (k : ℝ) : (∀ x : ℝ, f k x < g x) ↔ k ≥ (1 / 2) :=
by
  sorry

end range_of_k_l2222_222252


namespace time_morning_is_one_l2222_222254

variable (D : ℝ)  -- Define D as the distance between the two points.

def morning_speed := 20 -- Morning speed (km/h)
def afternoon_speed := 10 -- Afternoon speed (km/h)
def time_difference := 1 -- Time difference (hour)

-- Proving that the morning time t_m is equal to 1 hour
theorem time_morning_is_one (t_m t_a : ℝ) 
  (h1 : t_m - t_a = time_difference) 
  (h2 : D = morning_speed * t_m) 
  (h3 : D = afternoon_speed * t_a) : 
  t_m = 1 := 
by
  sorry

end time_morning_is_one_l2222_222254


namespace notebook_price_l2222_222220

theorem notebook_price (students_buying_notebooks n c : ℕ) (total_students : ℕ := 36) (total_cost : ℕ := 990) :
  students_buying_notebooks > 18 ∧ c > n ∧ students_buying_notebooks * n * c = total_cost → c = 15 :=
by
  sorry

end notebook_price_l2222_222220


namespace subtract_30_divisible_l2222_222221

theorem subtract_30_divisible (n : ℕ) (d : ℕ) (r : ℕ) 
  (h1 : n = 13602) (h2 : d = 87) (h3 : r = 30) 
  (h4 : n % d = r) : (n - r) % d = 0 :=
by
  -- Skipping the proof as it's not required
  sorry

end subtract_30_divisible_l2222_222221


namespace smallest_positive_debt_resolved_l2222_222285

theorem smallest_positive_debt_resolved : ∃ (D : ℤ), D > 0 ∧ (∃ (p g : ℤ), D = 400 * p + 250 * g) ∧ D = 50 :=
by
  sorry

end smallest_positive_debt_resolved_l2222_222285


namespace fitted_ball_volume_l2222_222228

noncomputable def volume_of_fitted_ball (d_ball d_h1 r_h1 d_h2 r_h2 : ℝ) : ℝ :=
  let r_ball := d_ball / 2
  let v_ball := (4 / 3) * Real.pi * r_ball^3
  let r_hole1 := r_h1
  let r_hole2 := r_h2
  let v_hole1 := Real.pi * r_hole1^2 * d_h1
  let v_hole2 := Real.pi * r_hole2^2 * d_h2
  v_ball - 2 * v_hole1 - v_hole2

theorem fitted_ball_volume :
  volume_of_fitted_ball 24 10 (3 / 2) 10 2 = 2219 * Real.pi :=
by
  sorry

end fitted_ball_volume_l2222_222228


namespace neg_exists_equiv_forall_l2222_222280

theorem neg_exists_equiv_forall (p : Prop) :
  (¬ (∃ n : ℕ, n^2 > 2^n)) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := sorry

end neg_exists_equiv_forall_l2222_222280


namespace sin_cos_identity_second_quadrant_l2222_222259

open Real

theorem sin_cos_identity_second_quadrant (α : ℝ) (hcos : cos α < 0) (hsin : sin α > 0) :
  (sin α / cos α) * sqrt ((1 / (sin α)^2) - 1) = -1 :=
sorry

end sin_cos_identity_second_quadrant_l2222_222259


namespace coordinates_on_y_axis_l2222_222262

theorem coordinates_on_y_axis (a : ℝ) 
  (h : (a - 3) = 0) : 
  P = (0, -1) :=
by 
  have ha : a = 3 := by sorry
  subst ha
  sorry

end coordinates_on_y_axis_l2222_222262


namespace scrooge_no_equal_coins_l2222_222275

theorem scrooge_no_equal_coins (n : ℕ → ℕ)
  (initial_state : n 1 = 1 ∧ n 2 = 0 ∧ n 3 = 0 ∧ n 4 = 0 ∧ n 5 = 0 ∧ n 6 = 0)
  (operation : ∀ x i, 1 ≤ i ∧ i ≤ 6 → (n (i + 1) = n i - x ∧ n ((i % 6) + 2) = n ((i % 6) + 2) + 6 * x) 
                      ∨ (n (i + 1) = n i + 6 * x ∧ n ((i % 6) + 2) = n ((i % 6) + 2) - x)) :
  ¬ ∃ k, n 1 = k ∧ n 2 = k ∧ n 3 = k ∧ n 4 = k ∧ n 5 = k ∧ n 6 = k :=
by {
  sorry
}

end scrooge_no_equal_coins_l2222_222275


namespace find_sin_beta_l2222_222202

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π/2) -- α is acute
variable (hβ : 0 < β ∧ β < π/2) -- β is acute

variable (hcosα : Real.cos α = 4/5)
variable (hcosαβ : Real.cos (α + β) = 5/13)

theorem find_sin_beta (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
    (hcosα : Real.cos α = 4/5) (hcosαβ : Real.cos (α + β) = 5/13) : 
    Real.sin β = 33/65 := 
sorry

end find_sin_beta_l2222_222202


namespace calories_burned_l2222_222272

theorem calories_burned {running_minutes walking_minutes total_minutes calories_per_minute_running calories_per_minute_walking calories_total : ℕ}
    (h_run : running_minutes = 35)
    (h_total : total_minutes = 60)
    (h_calories_run : calories_per_minute_running = 10)
    (h_calories_walk : calories_per_minute_walking = 4)
    (h_walk : walking_minutes = total_minutes - running_minutes)
    (h_calories_total : calories_total = running_minutes * calories_per_minute_running + walking_minutes * calories_per_minute_walking) : 
    calories_total = 450 := by
  sorry

end calories_burned_l2222_222272


namespace carrots_per_bundle_l2222_222271

theorem carrots_per_bundle (potatoes_total: ℕ) (potatoes_in_bundle: ℕ) (price_per_potato_bundle: ℝ) 
(carrot_total: ℕ) (price_per_carrot_bundle: ℝ) (total_revenue: ℝ) (carrots_per_bundle : ℕ) :
potatoes_total = 250 → potatoes_in_bundle = 25 → price_per_potato_bundle = 1.90 → 
carrot_total = 320 → price_per_carrot_bundle = 2 → total_revenue = 51 →
((carrots_per_bundle = carrot_total / ((total_revenue - (potatoes_total / potatoes_in_bundle) 
    * price_per_potato_bundle) / price_per_carrot_bundle))  ↔ carrots_per_bundle = 20) := by
  sorry

end carrots_per_bundle_l2222_222271


namespace find_lamp_cost_l2222_222297

def lamp_and_bulb_costs (L B : ℝ) : Prop :=
  B = L - 4 ∧ 2 * L + 6 * B = 32

theorem find_lamp_cost : ∃ L : ℝ, ∃ B : ℝ, lamp_and_bulb_costs L B ∧ L = 7 :=
by
  sorry

end find_lamp_cost_l2222_222297


namespace perpendicular_dot_product_zero_l2222_222277

variables (a : ℝ)
def m := (a, 2)
def n := (1, 1 - a)

theorem perpendicular_dot_product_zero : (m a).1 * (n a).1 + (m a).2 * (n a).2 = 0 → a = 2 :=
by sorry

end perpendicular_dot_product_zero_l2222_222277


namespace kiki_total_money_l2222_222270

theorem kiki_total_money 
  (S : ℕ) (H : ℕ) (M : ℝ)
  (h1: S = 18)
  (h2: H = 2 * S)
  (h3: 0.40 * M = 36) : 
  M = 90 :=
by
  sorry

end kiki_total_money_l2222_222270


namespace sarah_total_weeds_l2222_222244

theorem sarah_total_weeds :
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  tuesday_weeds + wednesday_weeds + thursday_weeds + friday_weeds = 120 :=
by
  intros
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  sorry

end sarah_total_weeds_l2222_222244


namespace greatest_perimeter_of_strips_l2222_222283

theorem greatest_perimeter_of_strips :
  let base := 10
  let height := 12
  let half_base := base / 2
  let right_triangle_area := (base / 2 * height) / 2
  let number_of_pieces := 10
  let sub_area := right_triangle_area / (number_of_pieces / 2)
  let h1 := (2 * sub_area) / half_base
  let hypotenuse := Real.sqrt (h1^2 + (half_base / 2)^2)
  let perimeter := half_base + 2 * hypotenuse
  perimeter = 11.934 :=
by
  sorry

end greatest_perimeter_of_strips_l2222_222283


namespace cups_filled_l2222_222248

def total_tea : ℕ := 1050
def tea_per_cup : ℕ := 65

theorem cups_filled : Nat.floor (total_tea / (tea_per_cup : ℚ)) = 16 :=
by
  sorry

end cups_filled_l2222_222248


namespace triangle_inequality_l2222_222231

variable {a b c : ℝ}

theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (habc1 : a + b > c) (habc2 : a + c > b) (habc3 : b + c > a) :
  (a / (b + c) + b / (c + a) + c / (a + b) < 2) :=
sorry

end triangle_inequality_l2222_222231


namespace at_least_six_stones_empty_l2222_222212

def frogs_on_stones (a : Fin 23 → Fin 23) (k : Nat) : Fin 22 → Fin 23 :=
  fun i => (a i + i.1 * k) % 23

theorem at_least_six_stones_empty 
  (a : Fin 22 → Fin 23) :
  ∃ k : Nat, ∀ (s : Fin 23), ∃ (j : Fin 22), frogs_on_stones (fun i => a i) k j ≠ s ↔ ∃! t : Fin 23, ∃! j, (frogs_on_stones (fun i => a i) k j) = t := 
  sorry

end at_least_six_stones_empty_l2222_222212


namespace cos_alpha_solution_l2222_222276

theorem cos_alpha_solution (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 1 / 2) : 
  Real.cos α = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_alpha_solution_l2222_222276


namespace train_speed_l2222_222288

theorem train_speed (length : ℝ) (time : ℝ) (length_eq : length = 375.03) (time_eq : time = 5) :
  let speed_kmph := (length / 1000) / (time / 3600)
  speed_kmph = 270.02 :=
by
  sorry

end train_speed_l2222_222288


namespace average_minutes_run_per_day_l2222_222278

variables (f : ℕ)
def third_graders := 6 * f
def fourth_graders := 2 * f
def fifth_graders := f

def total_minutes_run := 14 * third_graders f + 18 * fourth_graders f + 8 * fifth_graders f
def total_students := third_graders f + fourth_graders f + fifth_graders f

theorem average_minutes_run_per_day : 
  (total_minutes_run f) / (total_students f) = 128 / 9 :=
by
  sorry

end average_minutes_run_per_day_l2222_222278


namespace boy_completes_work_in_nine_days_l2222_222261

theorem boy_completes_work_in_nine_days :
  let M := (1 : ℝ) / 6
  let W := (1 : ℝ) / 18
  let B := (1 / 3 : ℝ) - M - W
  B = (1 : ℝ) / 9 := by
    sorry

end boy_completes_work_in_nine_days_l2222_222261


namespace sphere_surface_area_increase_l2222_222233

theorem sphere_surface_area_increase (V A : ℝ) (r : ℝ)
  (hV : V = (4/3) * π * r^3)
  (hA : A = 4 * π * r^2)
  : (∃ r', (V = 8 * ((4/3) * π * r'^3)) ∧ (∃ A', A' = 4 * A)) :=
by
  sorry

end sphere_surface_area_increase_l2222_222233


namespace analysis_method_proves_sufficient_condition_l2222_222215

-- Definitions and conditions from part (a)
def analysis_method_traces_cause_from_effect : Prop := true
def analysis_method_seeks_sufficient_conditions : Prop := true
def analysis_method_finds_conditions_for_inequality : Prop := true

-- The statement to be proven
theorem analysis_method_proves_sufficient_condition :
  analysis_method_finds_conditions_for_inequality →
  analysis_method_traces_cause_from_effect →
  analysis_method_seeks_sufficient_conditions →
  (B = "Sufficient condition") :=
by 
  sorry

end analysis_method_proves_sufficient_condition_l2222_222215


namespace store_discount_problem_l2222_222240

theorem store_discount_problem (original_price : ℝ) :
  let price_after_first_discount := original_price * 0.75
  let price_after_second_discount := price_after_first_discount * 0.90
  let true_discount := 1 - price_after_second_discount / original_price
  let claimed_discount := 0.40
  let difference := claimed_discount - true_discount
  true_discount = 0.325 ∧ difference = 0.075 :=
by
  sorry

end store_discount_problem_l2222_222240


namespace quadratic_roots_relationship_l2222_222284

theorem quadratic_roots_relationship 
  (a b c α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0) 
  (h_eq' : a * β^2 + b * β + c = 0)
  (h_roots : β = 3 * α) :
  3 * b^2 = 16 * a * c := 
sorry

end quadratic_roots_relationship_l2222_222284


namespace range_of_a_l2222_222209

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then 2^x + 1 else -x^2 + a * x

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a < 3) ↔ (2 ≤ a ∧ a < 2 * Real.sqrt 3) := by
  sorry

end range_of_a_l2222_222209


namespace distance_between_Q_and_R_l2222_222219

noncomputable def distance_QR : ℝ :=
  let DE : ℝ := 9
  let EF : ℝ := 12
  let DF : ℝ := 15
  let N : ℝ := 7.5
  let QF : ℝ := (N * DF) / EF
  let QD : ℝ := DF - QF
  let QR : ℝ := (QD * DF) / EF
  QR

theorem distance_between_Q_and_R 
  (DE EF DF N QF QD QR : ℝ )
  (h1 : DE = 9)
  (h2 : EF = 12)
  (h3 : DF = 15)
  (h4 : N = DF / 2)
  (h5 : QF = N * DF / EF)
  (h6 : QD = DF - QF)
  (h7 : QR = QD * DF / EF) :
  QR = 7.03125 :=
by
  sorry

end distance_between_Q_and_R_l2222_222219


namespace g_eq_one_l2222_222260

theorem g_eq_one (g : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), g (x - y) = g x * g y) 
  (h2 : ∀ (x : ℝ), g x ≠ 0) : 
  g 5 = 1 :=
by
  sorry

end g_eq_one_l2222_222260


namespace problem_solution_l2222_222206

variable (a b c : ℝ)

theorem problem_solution (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) :
  a + b ≤ 3 * c := 
sorry

end problem_solution_l2222_222206


namespace distance_between_intersections_l2222_222222

open Function

def cube_vertices : List (ℝ × ℝ × ℝ) :=
  [(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0), (5, 0, 5), (5, 5, 0), (5, 5, 5)]

def intersecting_points : List (ℝ × ℝ × ℝ) :=
  [(0, 3, 0), (2, 0, 0), (2, 5, 5)]

noncomputable def plane_distance_between_points : ℝ :=
  let S := (11 / 3, 0, 5)
  let T := (0, 5, 4)
  Real.sqrt ((11 / 3 - 0)^2 + (0 - 5)^2 + (5 - 4)^2)

theorem distance_between_intersections : plane_distance_between_points = Real.sqrt (355 / 9) :=
  sorry

end distance_between_intersections_l2222_222222


namespace calc_1_calc_2_calc_3_calc_4_l2222_222282

section
variables {m n x y z : ℕ} -- assuming all variables are natural numbers for simplicity.
-- Problem 1
theorem calc_1 : (2 * m * n) / (3 * m ^ 2) * (6 * m * n) / (5 * n) = (4 * n) / 5 :=
sorry

-- Problem 2
theorem calc_2 : (5 * x - 5 * y) / (3 * x ^ 2 * y) * (9 * x * y ^ 2) / (x ^ 2 - y ^ 2) = 
  15 * y / (x * (x + y)) :=
sorry

-- Problem 3
theorem calc_3 : ((x ^ 3 * y ^ 2) / z) ^ 2 * ((y * z) / x ^ 2) ^ 3 = y ^ 7 * z :=
sorry

-- Problem 4
theorem calc_4 : (4 * x ^ 2 * y ^ 2) / (2 * x + y) * (4 * x ^ 2 + 4 * x * y + y ^ 2) / (2 * x + y) / 
  ((2 * x * y) * (2 * x - y) / (4 * x ^ 2 - y ^ 2)) = 4 * x ^ 2 * y + 2 * x * y ^ 2 :=
sorry
end

end calc_1_calc_2_calc_3_calc_4_l2222_222282


namespace percentage_of_number_l2222_222214

theorem percentage_of_number (P : ℝ) (h : 0.10 * 3200 - 190 = P * 650) :
  P = 0.2 :=
sorry

end percentage_of_number_l2222_222214


namespace total_boys_school_l2222_222294

variable (B : ℕ)
variables (percMuslim percHindu percSikh boysOther : ℕ)

-- Defining the conditions
def condition1 : percMuslim = 44 := by sorry
def condition2 : percHindu = 28 := by sorry
def condition3 : percSikh = 10 := by sorry
def condition4 : boysOther = 54 := by sorry

-- Main theorem statement
theorem total_boys_school (h1 : percMuslim = 44) (h2 : percHindu = 28) (h3 : percSikh = 10) (h4 : boysOther = 54) : 
  B = 300 := by sorry

end total_boys_school_l2222_222294


namespace correct_calculation_l2222_222251

variable (a b : ℕ)

theorem correct_calculation : 3 * a * b - 2 * a * b = a * b := 
by sorry

end correct_calculation_l2222_222251


namespace simplify_expression_l2222_222218

theorem simplify_expression (x : ℤ) : 
  (2 * x ^ 13 + 3 * x ^ 12 - 4 * x ^ 9 + 5 * x ^ 7) + 
  (8 * x ^ 11 - 2 * x ^ 9 + 3 * x ^ 7 + 6 * x ^ 4 - 7 * x + 9) + 
  (x ^ 13 + 4 * x ^ 12 + x ^ 11 + 9 * x ^ 9) = 
  3 * x ^ 13 + 7 * x ^ 12 + 9 * x ^ 11 + 3 * x ^ 9 + 8 * x ^ 7 + 6 * x ^ 4 - 7 * x + 9 :=
sorry

end simplify_expression_l2222_222218


namespace two_pow_n_add_two_gt_n_sq_l2222_222210

open Nat

theorem two_pow_n_add_two_gt_n_sq (n : ℕ) (h : n > 0) : 2^n + 2 > n^2 :=
by
  sorry

end two_pow_n_add_two_gt_n_sq_l2222_222210


namespace problem1_problem2_l2222_222296

variable (a : ℝ) -- Declaring a as a real number

-- Proof statement for Problem 1
theorem problem1 : (a + 2) * (a - 2) = a^2 - 4 :=
sorry

-- Proof statement for Problem 2
theorem problem2 (h : a ≠ -2) : (a^2 - 4) / (a + 2) + 2 = a :=
sorry

end problem1_problem2_l2222_222296


namespace smallest_B_for_divisibility_by_4_l2222_222289

theorem smallest_B_for_divisibility_by_4 : 
  ∃ (B : ℕ), B < 10 ∧ (4 * 1000000 + B * 100000 + 80000 + 3961) % 4 = 0 ∧ ∀ (B' : ℕ), (B' < B ∧ B' < 10) → ¬ ((4 * 1000000 + B' * 100000 + 80000 + 3961) % 4 = 0) := 
sorry

end smallest_B_for_divisibility_by_4_l2222_222289


namespace proof_not_sufficient_nor_necessary_l2222_222224

noncomputable def not_sufficient_nor_necessary (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : Prop :=
  ¬ ((a > b) → (Real.log b / Real.log a < 1)) ∧ ¬ ((Real.log b / Real.log a < 1) → (a > b))

theorem proof_not_sufficient_nor_necessary (a b: ℝ) (h₁: 0 < a) (h₂: 0 < b) :
  not_sufficient_nor_necessary a b h₁ h₂ :=
  sorry

end proof_not_sufficient_nor_necessary_l2222_222224


namespace nat_number_solution_odd_l2222_222229

theorem nat_number_solution_odd (x y z : ℕ) (h : x + y + z = 100) : 
  ∃ P : ℕ, P = 49 ∧ P % 2 = 1 := 
sorry

end nat_number_solution_odd_l2222_222229


namespace find_p_for_natural_roots_l2222_222267

-- The polynomial is given.
def cubic_polynomial (p x : ℝ) : ℝ := 5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1

-- Problem statement to prove that p = 76 is the only real number such that
-- the cubic polynomial cubic_polynomial equals 66 * p has at least two natural number roots.
theorem find_p_for_natural_roots (p : ℝ) :
  (∃ (u v : ℕ), u ≠ v ∧ cubic_polynomial p u = 66 * p ∧ cubic_polynomial p v = 66 * p) ↔ p = 76 :=
by
  sorry

end find_p_for_natural_roots_l2222_222267


namespace donuts_distribution_l2222_222291

theorem donuts_distribution (kinds total min_each : ℕ) (h_kinds : kinds = 4) (h_total : total = 7) (h_min_each : min_each = 1) :
  ∃ n : ℕ, n = 20 := by
  sorry

end donuts_distribution_l2222_222291


namespace inequality_proof_l2222_222232

theorem inequality_proof
  (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (6841 * x - 1) / 9973 + (9973 * y - 1) / 6841 = z) :
  x / 9973 + y / 6841 > 1 :=
sorry

end inequality_proof_l2222_222232


namespace value_of_x_l2222_222236

theorem value_of_x (x y z w : ℕ) (h1 : x = y + 7) (h2 : y = z + 12) (h3 : z = w + 25) (h4 : w = 90) : x = 134 :=
by
  sorry

end value_of_x_l2222_222236


namespace fraction_problem_l2222_222263

theorem fraction_problem :
  ((3 / 4 - 5 / 8) / 2) = 1 / 16 :=
by
  sorry

end fraction_problem_l2222_222263


namespace min_expression_value_l2222_222293

variable {a : ℕ → ℝ}
variable (m n : ℕ)
variable (q : ℝ)

axiom pos_seq (n : ℕ) : a n > 0
axiom geom_seq (n : ℕ) : a (n + 1) = q * a n
axiom seq_condition : a 7 = a 6 + 2 * a 5
axiom exists_terms :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (Real.sqrt (a m * a n) = 4 * a 1)

theorem min_expression_value : 
  (∃m n : ℕ, m > 0 ∧ n > 0 ∧ (Real.sqrt (a m * a n) = 4 * a 1) ∧ 
  a 7 = a 6 + 2 * a 5 ∧ 
  (∀ n, a n > 0 ∧ a (n + 1) = q * a n)) → 
  (1 / m + 4 / n) ≥ 3 / 2 :=
sorry

end min_expression_value_l2222_222293
