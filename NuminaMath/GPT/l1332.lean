import Mathlib

namespace intersection_A_B_eq_C_l1332_133267

def A : Set ℝ := { x | 4 - x^2 ≥ 0 }
def B : Set ℝ := { x | x > -1 }
def C : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_A_B_eq_C : A ∩ B = C := 
by {
  sorry
}

end intersection_A_B_eq_C_l1332_133267


namespace exists_coprime_linear_combination_l1332_133252

theorem exists_coprime_linear_combination (a b p : ℤ) :
  ∃ k l : ℤ, Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
  sorry

end exists_coprime_linear_combination_l1332_133252


namespace max_value_of_f_symmetric_about_point_concave_inequality_l1332_133233

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 - x)

theorem max_value_of_f : ∃ x, f x = -4 :=
by
  sorry

theorem symmetric_about_point : ∀ x, f (1 - x) + f (1 + x) = -4 :=
by
  sorry

theorem concave_inequality (x1 x2 : ℝ) (h1 : x1 > 1) (h2 : x2 > 1) : 
  f ((x1 + x2) / 2) ≥ (f x1 + f x2) / 2 :=
by
  sorry

end max_value_of_f_symmetric_about_point_concave_inequality_l1332_133233


namespace at_least_one_not_less_than_one_l1332_133275

open Real

theorem at_least_one_not_less_than_one (x : ℝ) :
  let a := x^2 + 1/2
  let b := 2 - x
  let c := x^2 - x + 1
  a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1 :=
by
  -- Definitions of a, b, and c
  let a := x^2 + 1/2
  let b := 2 - x
  let c := x^2 - x + 1
  -- Proof is omitted
  sorry

end at_least_one_not_less_than_one_l1332_133275


namespace evaluate_expression_at_3_l1332_133247

theorem evaluate_expression_at_3 :
  (∀ x ≠ 2, (x = 3) → (x^2 - 5 * x + 6) / (x - 2) = 0) :=
by
  sorry

end evaluate_expression_at_3_l1332_133247


namespace find_a7_over_b7_l1332_133295

-- Definitions of the sequences and the arithmetic properties
variable {a b: ℕ → ℕ}  -- sequences a_n and b_n
variable {S T: ℕ → ℕ}  -- sums of the first n terms

-- Problem conditions
def is_arithmetic_sequence (seq: ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) - seq n = d

def sum_of_first_n_terms (seq: ℕ → ℕ) (sum_fn: ℕ → ℕ) : Prop :=
  ∀ n, sum_fn n = n * (seq 1 + seq n) / 2

-- Given conditions
axiom h1: is_arithmetic_sequence a
axiom h2: is_arithmetic_sequence b
axiom h3: sum_of_first_n_terms a S
axiom h4: sum_of_first_n_terms b T
axiom h5: ∀ n, S n / T n = (3 * n + 2) / (2 * n)

-- Main theorem to prove
theorem find_a7_over_b7 : (a 7) / (b 7) = (41 / 26) :=
sorry

end find_a7_over_b7_l1332_133295


namespace perpendicular_line_through_point_l1332_133258

theorem perpendicular_line_through_point (x y : ℝ) (c : ℝ) (P : ℝ × ℝ) :
  P = (-1, 2) →
  (∀ x y c : ℝ, (2*x - y + c = 0) ↔ (x+2*y-1=0) → (x+2*y-1=0)) →
  ∃ c : ℝ, 2*(-1) - 2 + c = 0 ∧ (2*x - y + c = 0) :=
by
  sorry

end perpendicular_line_through_point_l1332_133258


namespace trisha_dogs_food_expense_l1332_133230

theorem trisha_dogs_food_expense :
  ∀ (meat chicken veggies eggs initial remaining final: ℤ),
    meat = 17 → 
    chicken = 22 → 
    veggies = 43 → 
    eggs = 5 → 
    remaining = 35 → 
    initial = 167 →
    final = initial - (meat + chicken + veggies + eggs) - remaining →
    final = 45 := 
by
  intros meat chicken veggies eggs initial remaining final h_meat h_chicken h_veggies h_eggs h_remaining h_initial h_final
  sorry

end trisha_dogs_food_expense_l1332_133230


namespace negation_of_proposition_l1332_133222

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 0 < x → (x^2 + x > 0)) ↔ ∃ x : ℝ, 0 < x ∧ (x^2 + x ≤ 0) :=
sorry

end negation_of_proposition_l1332_133222


namespace union_A_B_eq_intersection_A_B_complement_eq_l1332_133231

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x ≤ 0}
def B_complement : Set ℝ := {x | x < 0 ∨ x > 4}

theorem union_A_B_eq : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} := by
  sorry

theorem intersection_A_B_complement_eq : A ∩ B_complement = {x | -1 ≤ x ∧ x < 0} := by
  sorry

end union_A_B_eq_intersection_A_B_complement_eq_l1332_133231


namespace true_statement_D_l1332_133219

-- Definitions related to the problem conditions
def supplementary_angles (a b : ℝ) : Prop := a + b = 180

def exterior_angle_sum_of_polygon (n : ℕ) : ℝ := 360

def acute_angle (a : ℝ) : Prop := a < 90

def triangle_inequality (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem to be proven based on the correct evaluation
theorem true_statement_D (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0):
  triangle_inequality a b c :=
by 
  sorry

end true_statement_D_l1332_133219


namespace line_bisects_circle_l1332_133274

theorem line_bisects_circle (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, l x y ↔ x - y = 0) → 
  (∀ x y : ℝ, C x y ↔ x^2 + y^2 = 1) → 
  ∀ x y : ℝ, (x - y = 0) ∨ (x + y = 0) → l x y ∧ C x y → l x y = (x - y = 0) := by
  sorry

end line_bisects_circle_l1332_133274


namespace johns_drive_distance_l1332_133217

/-- John's driving problem -/
theorem johns_drive_distance
  (d t : ℝ)
  (h1 : d = 25 * (t + 1.5))
  (h2 : d = 25 + 45 * (t - 1.25)) :
  d = 123.4375 := 
sorry

end johns_drive_distance_l1332_133217


namespace michelle_has_total_crayons_l1332_133287

noncomputable def michelle_crayons : ℕ :=
  let type1_crayons_per_box := 5
  let type2_crayons_per_box := 12
  let type1_boxes := 4
  let type2_boxes := 3
  let missing_crayons := 2
  (type1_boxes * type1_crayons_per_box - missing_crayons) + (type2_boxes * type2_crayons_per_box)

theorem michelle_has_total_crayons : michelle_crayons = 54 :=
by
  -- The proof step would go here, but it is omitted according to instructions.
  sorry

end michelle_has_total_crayons_l1332_133287


namespace smallest_possible_a_l1332_133281

noncomputable def f (a b c : ℕ) (x : ℝ) : ℝ := a * x^2 + b * x + ↑c

theorem smallest_possible_a
  (a b c : ℕ)
  (r s : ℝ)
  (h_arith_seq : b - a = c - b)
  (h_order_pos : 0 < a ∧ a < b ∧ b < c)
  (h_distinct : r ≠ s)
  (h_rs_2017 : r * s = 2017)
  (h_fr_eq_s : f a b c r = s)
  (h_fs_eq_r : f a b c s = r) :
  a = 1 := sorry

end smallest_possible_a_l1332_133281


namespace solve_equation_l1332_133271

noncomputable def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, x = Real.arcsin (3/4) + 2 * k * Real.pi ∨ x = Real.pi - Real.arcsin (3/4) + 2 * k * Real.pi

theorem solve_equation (x : ℝ) :
  (5 * Real.sin x = 4 + 2 * Real.cos (2 * x)) ↔ solution_set x := 
sorry

end solve_equation_l1332_133271


namespace maximum_profit_and_price_range_l1332_133211

-- Definitions
def cost_per_item : ℝ := 60
def max_profit_percentage : ℝ := 0.45
def sales_volume (x : ℝ) : ℝ := -x + 120
def profit (x : ℝ) : ℝ := sales_volume x * (x - cost_per_item)

-- The main theorem
theorem maximum_profit_and_price_range :
  (∃ x : ℝ, x = 87 ∧ profit x = 891) ∧
  (∀ x : ℝ, profit x ≥ 500 ↔ (70 ≤ x ∧ x ≤ 110)) :=
by
  sorry

end maximum_profit_and_price_range_l1332_133211


namespace equilateral_triangle_not_centrally_symmetric_l1332_133273

-- Definitions for the shapes
def is_centrally_symmetric (shape : Type) : Prop := sorry
def Parallelogram : Type := sorry
def LineSegment : Type := sorry
def EquilateralTriangle : Type := sorry
def Rhombus : Type := sorry

-- Main theorem statement
theorem equilateral_triangle_not_centrally_symmetric :
  ¬ is_centrally_symmetric EquilateralTriangle ∧
  is_centrally_symmetric Parallelogram ∧
  is_centrally_symmetric LineSegment ∧
  is_centrally_symmetric Rhombus :=
sorry

end equilateral_triangle_not_centrally_symmetric_l1332_133273


namespace axis_of_symmetry_range_of_m_l1332_133244

/-- The conditions given in the original mathematical problem -/
noncomputable def f (x : ℝ) : ℝ :=
  let OA := (2 * Real.cos x, Real.sqrt 3)
  let OB := (Real.sin x + Real.sqrt 3 * Real.cos x, -1)
  (OA.1 * OB.1 + OA.2 * OB.2) + 2

/-- Question 1: The axis of symmetry for the function f(x) -/
theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, (2 * x + Real.pi / 3 = Real.pi / 2 + k * Real.pi) ↔ (x = k * Real.pi / 2 + Real.pi / 12) :=
sorry

/-- Question 2: The range of m such that g(x) = f(x) + m has zero points for x in (0, π/2) -/
theorem range_of_m (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∃ c : ℝ, (f x + c = 0)) ↔ ( -4 ≤ c ∧ c < Real.sqrt 3 - 2) :=
sorry

end axis_of_symmetry_range_of_m_l1332_133244


namespace abs_sum_zero_eq_neg_one_l1332_133204

theorem abs_sum_zero_eq_neg_one (a b : ℝ) (h : |3 + a| + |b - 2| = 0) : a + b = -1 :=
sorry

end abs_sum_zero_eq_neg_one_l1332_133204


namespace marginal_cost_per_product_calculation_l1332_133205

def fixed_cost : ℝ := 12000
def total_cost : ℝ := 16000
def num_products : ℕ := 20

theorem marginal_cost_per_product_calculation :
  (total_cost - fixed_cost) / num_products = 200 := by
  sorry

end marginal_cost_per_product_calculation_l1332_133205


namespace man_swim_downstream_distance_l1332_133251

-- Define the given conditions
def t_d : ℝ := 6
def t_u : ℝ := 6
def d_u : ℝ := 18
def V_m : ℝ := 4.5

-- The distance the man swam downstream
def distance_downstream : ℝ := 36

-- Prove that given the conditions, the man swam 36 km downstream
theorem man_swim_downstream_distance (V_c : ℝ) :
  (d_u / (V_m - V_c) = t_u) →
  (distance_downstream / (V_m + V_c) = t_d) →
  distance_downstream = 36 :=
by
  sorry

end man_swim_downstream_distance_l1332_133251


namespace simplify_expression_l1332_133266

-- Defining the original expression
def original_expr (y : ℝ) : ℝ := 3 * y^3 - 7 * y^2 + 12 * y + 5 - (2 * y^3 - 4 + 3 * y^2 - 9 * y)

-- Defining the simplified expression
def simplified_expr (y : ℝ) : ℝ := y^3 - 10 * y^2 + 21 * y + 9

-- The statement to prove
theorem simplify_expression (y : ℝ) : original_expr y = simplified_expr y :=
by sorry

end simplify_expression_l1332_133266


namespace profit_percent_l1332_133223

variable {P C : ℝ}

theorem profit_percent (h1: 2 / 3 * P = 0.82 * C) : ((P - C) / C) * 100 = 23 := by
  have h2 : C = (2 / 3 * P) / 0.82 := by sorry
  have h3 : (P - C) / C = (P - (2 / 3 * P) / 0.82) / ((2 / 3 * P) / 0.82) := by sorry
  have h4 : (P - (2 / 3 * P) / 0.82) / ((2 / 3 * P) / 0.82) = (0.82 * P - 2 / 3 * P) / (2 / 3 * P) := by sorry
  have h5 : (0.82 * P - 2 / 3 * P) / (2 / 3 * P) = 0.1533 := by sorry
  have h6 : 0.1533 * 100 = 23 := by sorry
  sorry

end profit_percent_l1332_133223


namespace last_two_digits_of_sum_of_factorials_l1332_133288

-- Problem statement: Sum of factorials from 1 to 15
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => Nat.factorial k)

-- Define the main problem
theorem last_two_digits_of_sum_of_factorials : 
  (sum_factorials 15) % 100 = 13 :=
by 
  sorry

end last_two_digits_of_sum_of_factorials_l1332_133288


namespace binomial_coeff_x5y3_in_expansion_eq_56_l1332_133297

theorem binomial_coeff_x5y3_in_expansion_eq_56:
  let n := 8
  let k := 3
  let binom_coeff := Nat.choose n k
  binom_coeff = 56 := 
by sorry

end binomial_coeff_x5y3_in_expansion_eq_56_l1332_133297


namespace solve_for_k_l1332_133279

theorem solve_for_k (x : ℝ) (k : ℝ) (h₁ : 2 * x - 1 = 3) (h₂ : 3 * x + k = 0) : k = -6 :=
by
  sorry

end solve_for_k_l1332_133279


namespace stratified_sampling_numbers_l1332_133264

-- Definitions of the conditions
def total_teachers : ℕ := 300
def senior_teachers : ℕ := 90
def intermediate_teachers : ℕ := 150
def junior_teachers : ℕ := 60
def sample_size : ℕ := 40

-- Hypothesis of proportions
def proportion_senior := senior_teachers / total_teachers
def proportion_intermediate := intermediate_teachers / total_teachers
def proportion_junior := junior_teachers / total_teachers

-- Expected sample counts using stratified sampling method
def expected_senior_drawn := proportion_senior * sample_size
def expected_intermediate_drawn := proportion_intermediate * sample_size
def expected_junior_drawn := proportion_junior * sample_size

-- Proof goal
theorem stratified_sampling_numbers :
  (expected_senior_drawn = 12) ∧ 
  (expected_intermediate_drawn = 20) ∧ 
  (expected_junior_drawn = 8) :=
by
  sorry

end stratified_sampling_numbers_l1332_133264


namespace probability_of_interval_l1332_133236

-- Define the random variable ξ and its probability distribution P(ξ = k)
variables (ξ : ℕ → ℝ) (P : ℕ → ℝ)

-- Define a constant a
noncomputable def a : ℝ := 5/4

-- Given conditions
axiom condition1 : ∀ k, k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 → P k = a / (k * (k + 1))
axiom condition2 : P 1 + P 2 + P 3 + P 4 = 1

-- Statement to prove
theorem probability_of_interval : P 1 + P 2 = 5/6 :=
by sorry

end probability_of_interval_l1332_133236


namespace Q_investment_l1332_133246

-- Given conditions
variables (P Q : Nat) (P_investment : P = 30000) (profit_ratio : 2 / 3 = P / Q)

-- Target statement
theorem Q_investment : Q = 45000 :=
by 
  sorry

end Q_investment_l1332_133246


namespace problem_statement_l1332_133253

noncomputable def f (x : ℝ) (b c : ℝ) := x^2 + b * x + c

theorem problem_statement (b c : ℝ) (h : ∀ x : ℝ, f (x - 1) b c = f (3 - x) b c) : f 0 b c < f (-2) b c ∧ f (-2) b c < f 5 b c := 
by sorry

end problem_statement_l1332_133253


namespace yoongi_more_points_l1332_133243

def yoongiPoints : ℕ := 4
def jungkookPoints : ℕ := 6 - 3

theorem yoongi_more_points : yoongiPoints > jungkookPoints := by
  sorry

end yoongi_more_points_l1332_133243


namespace value_of_n_l1332_133265

theorem value_of_n {k n : ℕ} (h1 : k = 71 * n + 11) (h2 : (k : ℝ) / (n : ℝ) = 71.2) : n = 55 :=
sorry

end value_of_n_l1332_133265


namespace solve_proof_problem_l1332_133248

noncomputable def proof_problem : Prop :=
  let short_videos_per_day := 2
  let short_video_time := 2
  let longer_videos_per_day := 1
  let week_days := 7
  let total_weekly_video_time := 112
  let total_short_video_time_per_week := short_videos_per_day * short_video_time * week_days
  let total_longer_video_time_per_week := total_weekly_video_time - total_short_video_time_per_week
  let longer_video_multiple := total_longer_video_time_per_week / short_video_time
  longer_video_multiple = 42

theorem solve_proof_problem : proof_problem :=
by
  /- Proof goes here -/
  sorry

end solve_proof_problem_l1332_133248


namespace geometric_sequence_sum_l1332_133215

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 0 + a 1 = 1) (h2 : a 1 + a 2 = 2) : a 5 + a 6 = 32 :=
sorry

end geometric_sequence_sum_l1332_133215


namespace find_plane_speed_l1332_133278

-- Defining the values in the problem
def distance_with_wind : ℝ := 420
def distance_against_wind : ℝ := 350
def wind_speed : ℝ := 23

-- The speed of the plane in still air
def plane_speed_in_still_air : ℝ := 253

-- Proof goal: Given the conditions, the speed of the plane in still air is 253 mph
theorem find_plane_speed :
  ∃ p : ℝ, (distance_with_wind / (p + wind_speed) = distance_against_wind / (p - wind_speed)) ∧ p = plane_speed_in_still_air :=
by
  use plane_speed_in_still_air
  have h : plane_speed_in_still_air = 253 := rfl
  sorry

end find_plane_speed_l1332_133278


namespace cost_of_goods_l1332_133242

theorem cost_of_goods
  (x y z : ℝ)
  (h1 : 3 * x + 7 * y + z = 315)
  (h2 : 4 * x + 10 * y + z = 420) :
  x + y + z = 105 :=
by
  sorry

end cost_of_goods_l1332_133242


namespace mod_z_range_l1332_133206

noncomputable def z (t : ℝ) : ℂ := Complex.ofReal (1/t) + Complex.I * t

noncomputable def mod_z (t : ℝ) : ℝ := Complex.abs (z t)

theorem mod_z_range : 
  ∀ (t : ℝ), t ≠ 0 → ∃ (r : ℝ), r = mod_z t ∧ r ≥ Real.sqrt 2 :=
  by sorry

end mod_z_range_l1332_133206


namespace number_of_children_l1332_133218

theorem number_of_children (A V S : ℕ) (x : ℕ → ℕ) (n : ℕ) 
  (h1 : (A / 2) + V = (A + V + S + (Finset.range (n - 3)).sum x) / n)
  (h2 : S + A = V + (Finset.range (n - 3)).sum x) : 
  n = 6 :=
sorry

end number_of_children_l1332_133218


namespace soul_inequality_phi_inequality_iff_t_one_l1332_133210

noncomputable def e : ℝ := Real.exp 1

theorem soul_inequality (x : ℝ) : e^x ≥ x + 1 ↔ x = 0 :=
by sorry

theorem phi_inequality_iff_t_one (x t : ℝ) : (∀ x, e^x - t*x - 1 ≥ 0) ↔ t = 1 :=
by sorry

end soul_inequality_phi_inequality_iff_t_one_l1332_133210


namespace least_number_subtracted_l1332_133234

theorem least_number_subtracted (n : ℕ) (h : n = 427398) : ∃ x, x = 8 ∧ (n - x) % 10 = 0 :=
by
  sorry

end least_number_subtracted_l1332_133234


namespace distance_of_intersection_points_l1332_133292

def C1 (x y : ℝ) : Prop := x - y + 4 = 0
def C2 (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 1

theorem distance_of_intersection_points {A B : ℝ × ℝ} (hA1 : C1 A.fst A.snd) (hA2 : C2 A.fst A.snd)
  (hB1 : C1 B.fst B.snd) (hB2 : C2 B.fst B.snd) : dist A B = Real.sqrt 2 := by
  sorry

end distance_of_intersection_points_l1332_133292


namespace third_number_in_list_l1332_133229

theorem third_number_in_list :
  let nums : List ℕ := [201, 202, 205, 206, 209, 209, 210, 212, 212]
  nums.nthLe 2 (by simp [List.length]) = 205 :=
sorry

end third_number_in_list_l1332_133229


namespace inequality_subtraction_l1332_133291

variable (a b : ℝ)

theorem inequality_subtraction (h : a > b) : a - 5 > b - 5 :=
sorry

end inequality_subtraction_l1332_133291


namespace exist_line_l1_exist_line_l2_l1332_133208

noncomputable def P : ℝ × ℝ := ⟨3, 2⟩

def line1_eq (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def line2_eq (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def perpend_line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 15 = 0

def line_l1 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0
def line_l2_case1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def line_l2_case2 (x y : ℝ) : Prop := x + y - 5 = 0

theorem exist_line_l1 : ∃ (x y : ℝ), line1_eq x y ∧ line2_eq x y ∧ perpend_line_eq x y → line_l1 x y :=
by
  sorry

theorem exist_line_l2 : ∃ (x y : ℝ), line1_eq x y ∧ line2_eq x y ∧ ((line_l2_case1 x y) ∨ (line_l2_case2 x y)) :=
by
  sorry

end exist_line_l1_exist_line_l2_l1332_133208


namespace inequality_proof_l1332_133289

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^4 * b^b * c^c ≥ a⁻¹ * b⁻¹ * c⁻¹ :=
sorry

end inequality_proof_l1332_133289


namespace isosceles_triangle_angles_l1332_133225

theorem isosceles_triangle_angles (A B C : ℝ) (h_iso: (A = B) ∨ (B = C) ∨ (A = C)) (angle_A : A = 50) :
  (B = 50) ∨ (B = 65) ∨ (B = 80) :=
by
  sorry

end isosceles_triangle_angles_l1332_133225


namespace small_boxes_count_correct_l1332_133298

-- Definitions of constants
def feet_per_large_box_seal : ℕ := 4
def feet_per_medium_box_seal : ℕ := 2
def feet_per_small_box_seal : ℕ := 1
def feet_per_box_label : ℕ := 1

def large_boxes_packed : ℕ := 2
def medium_boxes_packed : ℕ := 8
def total_tape_used : ℕ := 44

-- Definition for the total tape used for large and medium boxes
def tape_used_large_boxes : ℕ := (large_boxes_packed * feet_per_large_box_seal) + (large_boxes_packed * feet_per_box_label)
def tape_used_medium_boxes : ℕ := (medium_boxes_packed * feet_per_medium_box_seal) + (medium_boxes_packed * feet_per_box_label)
def tape_used_large_and_medium_boxes : ℕ := tape_used_large_boxes + tape_used_medium_boxes
def tape_used_small_boxes : ℕ := total_tape_used - tape_used_large_and_medium_boxes

-- The number of small boxes packed
def small_boxes_packed : ℕ := tape_used_small_boxes / (feet_per_small_box_seal + feet_per_box_label)

-- Proof problem statement
theorem small_boxes_count_correct (n : ℕ) (h : small_boxes_packed = n) : n = 5 :=
by
  sorry

end small_boxes_count_correct_l1332_133298


namespace sally_and_mary_picked_16_lemons_l1332_133237

theorem sally_and_mary_picked_16_lemons (sally_lemons mary_lemons : ℕ) (sally_picked : sally_lemons = 7) (mary_picked : mary_lemons = 9) :
  sally_lemons + mary_lemons = 16 :=
by {
  sorry
}

end sally_and_mary_picked_16_lemons_l1332_133237


namespace decreasing_cubic_function_l1332_133203

theorem decreasing_cubic_function (a : ℝ) :
  (∀ x : ℝ, 3 * a * x^2 - 1 ≤ 0) → a ≤ 0 :=
sorry

end decreasing_cubic_function_l1332_133203


namespace carrie_hours_per_day_l1332_133216

theorem carrie_hours_per_day (h : ℕ) 
  (worked_4_days : ∀ n, n = 4 * h) 
  (paid_per_hour : ℕ := 22)
  (cost_of_supplies : ℕ := 54)
  (profit : ℕ := 122) :
  88 * h - cost_of_supplies = profit → h = 2 := 
by 
  -- Assume problem conditions and solve
  sorry

end carrie_hours_per_day_l1332_133216


namespace baseball_team_wins_more_than_three_times_losses_l1332_133249

theorem baseball_team_wins_more_than_three_times_losses
    (total_games : ℕ)
    (wins : ℕ)
    (losses : ℕ)
    (h1 : total_games = 130)
    (h2 : wins = 101)
    (h3 : wins + losses = total_games) :
    wins - 3 * losses = 14 :=
by
    -- Proof goes here
    sorry

end baseball_team_wins_more_than_three_times_losses_l1332_133249


namespace boats_distance_one_minute_before_collision_l1332_133213

theorem boats_distance_one_minute_before_collision :
  let speedA := 5  -- miles/hr
  let speedB := 21 -- miles/hr
  let initial_distance := 20 -- miles
  let combined_speed := speedA + speedB -- combined speed in miles/hr
  let speed_per_minute := combined_speed / 60 -- convert to miles/minute
  let time_to_collision := initial_distance / speed_per_minute -- time in minutes until collision
  initial_distance - (time_to_collision - 1) * speed_per_minute = 0.4333 :=
by
  sorry

end boats_distance_one_minute_before_collision_l1332_133213


namespace cos_half_angle_inequality_1_cos_half_angle_inequality_2_l1332_133207

open Real

variable {A B C : ℝ} (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hA_sum : A + B + C = π)

theorem cos_half_angle_inequality_1 :
  cos (A / 2) < cos (B / 2) + cos (C / 2) :=
by sorry

theorem cos_half_angle_inequality_2 :
  cos (A / 2) < sin (B / 2) + sin (C / 2) :=
by sorry

end cos_half_angle_inequality_1_cos_half_angle_inequality_2_l1332_133207


namespace maximum_area_of_triangle_l1332_133282

theorem maximum_area_of_triangle :
  ∃ (b c : ℝ), (a = 2) ∧ (A = 60 * Real.pi / 180) ∧
  (∀ S : ℝ, S = (1/2) * b * c * Real.sin A → S ≤ Real.sqrt 3) :=
by sorry

end maximum_area_of_triangle_l1332_133282


namespace highest_score_not_necessarily_at_least_12_l1332_133260

section

-- Define the number of teams
def teams : ℕ := 12

-- Define the number of games each team plays
def games_per_team : ℕ := teams - 1

-- Define the total number of games
def total_games : ℕ := (teams * games_per_team) / 2

-- Define the points system
def points_for_win : ℕ := 2
def points_for_draw : ℕ := 1

-- Define the total points in the tournament
def total_points : ℕ := total_games * points_for_win

-- The highest score possible statement
def highest_score_must_be_at_least_12_statement : Prop :=
  ∀ (scores : Fin teams → ℕ), (∃ i, scores i ≥ 12)

-- Theorem stating that the statement "The highest score must be at least 12" is false
theorem highest_score_not_necessarily_at_least_12 (h : ∀ (scores : Fin teams → ℕ), (∃ i, scores i ≥ 12)) : False :=
  sorry

end

end highest_score_not_necessarily_at_least_12_l1332_133260


namespace diagonal_splits_odd_vertices_l1332_133201

theorem diagonal_splits_odd_vertices (n : ℕ) (H : n^2 ≤ (2 * n + 2) * (2 * n + 1) / 2) :
  ∃ (x y : ℕ), x < y ∧ x ≤ 2 * n + 1 ∧ y ≤ 2 * n + 2 ∧ (y - x) % 2 = 0 :=
sorry

end diagonal_splits_odd_vertices_l1332_133201


namespace geometric_sequences_identical_l1332_133214

theorem geometric_sequences_identical
  (a_0 q r : ℝ)
  (a_n b_n c_n : ℕ → ℝ)
  (H₁ : ∀ n, a_n n = a_0 * q ^ n)
  (H₂ : ∀ n, b_n n = a_0 * r ^ n)
  (H₃ : ∀ n, c_n n = a_n n + b_n n)
  (H₄ : ∃ s : ℝ, ∀ n, c_n n = c_n 0 * s ^ n):
  ∀ n, a_n n = b_n n := sorry

end geometric_sequences_identical_l1332_133214


namespace acute_triangle_integers_count_l1332_133283

theorem acute_triangle_integers_count :
  ∃ (x_vals : List ℕ), (∀ x ∈ x_vals, 7 < x ∧ x < 33 ∧ (if x > 20 then x^2 < 569 else x > Int.sqrt 231)) ∧ x_vals.length = 8 :=
by
  sorry

end acute_triangle_integers_count_l1332_133283


namespace M_ends_in_two_zeros_iff_l1332_133202

theorem M_ends_in_two_zeros_iff (n : ℕ) (h : n > 0) : 
  (1^n + 2^n + 3^n + 4^n) % 100 = 0 ↔ n % 4 = 3 :=
by sorry

end M_ends_in_two_zeros_iff_l1332_133202


namespace factorization_problem_l1332_133232

theorem factorization_problem (a b c : ℝ) :
  let E := a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3)
  let P := -(a^2 + ab + b^2 + bc + c^2 + ac)
  E = (a - b) * (b - c) * (c - a) * P :=
by
  sorry

end factorization_problem_l1332_133232


namespace sum_of_digits_least_N_l1332_133256

-- Define the function P(N)
def P (N : ℕ) : ℚ := (Nat.ceil (3 * N / 5 + 1) : ℕ) / (N + 1)

-- Define the predicate that checks if P(N) is less than 321/400
def P_lt_321_over_400 (N : ℕ) : Prop := P N < (321 / 400 : ℚ)

-- Define a function that sums the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- The main statement: we claim the least multiple of 5 satisfying the condition
-- That the sum of its digits is 12
theorem sum_of_digits_least_N :
  ∃ N : ℕ, 
    (N % 5 = 0) ∧ 
    P_lt_321_over_400 N ∧ 
    (∀ N' : ℕ, (N' % 5 = 0) → P_lt_321_over_400 N' → N' ≥ N) ∧ 
    sum_of_digits N = 12 := 
sorry

end sum_of_digits_least_N_l1332_133256


namespace nell_more_ace_cards_than_baseball_l1332_133286

-- Definitions based on conditions
def original_baseball_cards : ℕ := 239
def original_ace_cards : ℕ := 38
def current_ace_cards : ℕ := 376
def current_baseball_cards : ℕ := 111

-- The statement we need to prove
theorem nell_more_ace_cards_than_baseball :
  current_ace_cards - current_baseball_cards = 265 :=
by
  -- Add the proof here
  sorry

end nell_more_ace_cards_than_baseball_l1332_133286


namespace matrix_det_eq_l1332_133226

open Matrix

def matrix3x3 (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![x + 1, x, x],
    ![x, x + 2, x],
    ![x, x, x + 3]
  ]

theorem matrix_det_eq (x : ℝ) : det (matrix3x3 x) = 2 * x^2 + 11 * x + 6 :=
  sorry

end matrix_det_eq_l1332_133226


namespace suzhou_metro_scientific_notation_l1332_133257

theorem suzhou_metro_scientific_notation : 
  (∃(a : ℝ) (n : ℤ), 
    1 ≤ abs a ∧ abs a < 10 ∧ 15.6 * 10^9 = a * 10^n) → 
    (a = 1.56 ∧ n = 9) := 
by
  sorry

end suzhou_metro_scientific_notation_l1332_133257


namespace total_spent_l1332_133290

theorem total_spent (B D : ℝ) (h1 : D = 0.7 * B) (h2 : B = D + 15) : B + D = 85 :=
sorry

end total_spent_l1332_133290


namespace temperature_conversion_l1332_133245

theorem temperature_conversion :
  ∀ (k t : ℝ),
    (t = (5 / 9) * (k - 32) ∧ k = 95) →
    t = 35 := by
  sorry

end temperature_conversion_l1332_133245


namespace land_to_water_time_ratio_l1332_133239

-- Define the conditions
def distance_water : ℕ := 50
def distance_land : ℕ := 300
def speed_ratio : ℕ := 3

-- Define the Lean theorem statement
theorem land_to_water_time_ratio (x : ℝ) (hx : x > 0) : 
  (distance_land / (speed_ratio * x)) / (distance_water / x) = 2 := by
  sorry

end land_to_water_time_ratio_l1332_133239


namespace rainfall_march_correct_l1332_133254

def rainfall_march : ℝ :=
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let average := 4
  let total_expected := 5 * average
  let total_april_to_july := april + may + june + july
  total_expected - total_april_to_july

theorem rainfall_march_correct (march_rainfall : ℝ) :
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let average := 4
  let total_expected := 5 * average
  let total_april_to_july := april + may + june + july
  march_rainfall = total_expected - total_april_to_july :=
by
  sorry

end rainfall_march_correct_l1332_133254


namespace polygon_max_sides_l1332_133272

theorem polygon_max_sides (n : ℕ) (h : (n - 2) * 180 < 2005) : n ≤ 13 :=
by {
  sorry
}

end polygon_max_sides_l1332_133272


namespace b_days_solve_l1332_133285

-- Definitions from the conditions
variable (b_days : ℝ)
variable (a_rate : ℝ) -- work rate of a
variable (b_rate : ℝ) -- work rate of b

-- Condition 1: a is twice as fast as b
def twice_as_fast_as_b : Prop :=
  a_rate = 2 * b_rate

-- Condition 2: a and b together can complete the work in 3.333333333333333 days
def combined_completion_time : Prop :=
  1 / (a_rate + b_rate) = 10 / 3

-- The number of days b alone can complete the work should satisfy this equation
def b_alone_can_complete_in_b_days : Prop :=
  b_rate = 1 / b_days

-- The actual theorem we want to prove:
theorem b_days_solve (b_rate a_rate : ℝ) (h1 : twice_as_fast_as_b a_rate b_rate) (h2 : combined_completion_time a_rate b_rate) : b_days = 10 :=
by
  sorry

end b_days_solve_l1332_133285


namespace find_integer_pairs_l1332_133235

theorem find_integer_pairs : 
  ∀ (x y : Int), x^3 = y^3 + 2 * y^2 + 1 ↔ (x, y) = (1, 0) ∨ (x, y) = (1, -2) ∨ (x, y) = (-2, -3) :=
by
  intros x y
  sorry

end find_integer_pairs_l1332_133235


namespace initial_students_l1332_133221

def students_got_off : ℕ := 3
def students_left : ℕ := 7

theorem initial_students (h1 : students_got_off = 3) (h2 : students_left = 7) :
    students_got_off + students_left = 10 :=
by
  sorry

end initial_students_l1332_133221


namespace hh_two_eq_902_l1332_133259

def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem hh_two_eq_902 : h (h 2) = 902 := 
by
  sorry

end hh_two_eq_902_l1332_133259


namespace fencing_cost_is_correct_l1332_133241

def length : ℕ := 60
def cost_per_meter : ℕ := 27 -- using the closest integer value to 26.50
def breadth (l : ℕ) : ℕ := l - 20
def perimeter (l b : ℕ) : ℕ := 2 * l + 2 * b
def total_cost (P : ℕ) (c : ℕ) : ℕ := P * c

theorem fencing_cost_is_correct :
  total_cost (perimeter length (breadth length)) cost_per_meter = 5300 :=
  sorry

end fencing_cost_is_correct_l1332_133241


namespace smallest_odd_m_satisfying_inequality_l1332_133209

theorem smallest_odd_m_satisfying_inequality : ∃ m : ℤ, m^2 - 11 * m + 24 ≥ 0 ∧ (m % 2 = 1) ∧ ∀ n : ℤ, n^2 - 11 * n + 24 ≥ 0 ∧ (n % 2 = 1) → m ≤ n → m = 3 :=
by
  sorry

end smallest_odd_m_satisfying_inequality_l1332_133209


namespace smallest_number_to_add_for_divisibility_l1332_133238

theorem smallest_number_to_add_for_divisibility :
  ∃ x : ℕ, 1275890 + x ≡ 0 [MOD 2375] ∧ x = 1360 :=
by sorry

end smallest_number_to_add_for_divisibility_l1332_133238


namespace Nina_now_l1332_133268

def Lisa_age (l m n : ℝ) := l + m + n = 36
def Nina_age (l n : ℝ) := n - 5 = 2 * l
def Mike_age (l m : ℝ) := m + 2 = (l + 2) / 2

theorem Nina_now (l m n : ℝ) (h1 : Lisa_age l m n) (h2 : Nina_age l n) (h3 : Mike_age l m) : n = 34.6 := by
  sorry

end Nina_now_l1332_133268


namespace price_of_refrigerator_l1332_133293

variable (R W : ℝ)

theorem price_of_refrigerator 
  (h1 : W = R - 1490) 
  (h2 : R + W = 7060) 
  : R = 4275 :=
sorry

end price_of_refrigerator_l1332_133293


namespace first_number_is_105_percent_of_second_kilograms_reduced_by_10_percent_l1332_133224

-- Proof problem 1: Given a number is 5% more than another number
theorem first_number_is_105_percent_of_second (x y : ℚ) (h : x = y * 1.05) : x = y * (1 + 0.05) :=
by {
  -- proof here
  sorry
}

-- Proof problem 2: 10 kilograms reduced by 10%
theorem kilograms_reduced_by_10_percent (kg : ℚ) (h : kg = 10) : kg * (1 - 0.1) = 9 :=
by {
  -- proof here
  sorry
}

end first_number_is_105_percent_of_second_kilograms_reduced_by_10_percent_l1332_133224


namespace largest_divisor_of_n_pow4_minus_n_for_composites_l1332_133220

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(∀ k : ℕ, 1 < k ∧ k < n → ¬(k ∣ n))

theorem largest_divisor_of_n_pow4_minus_n_for_composites : ∀ n : ℕ, is_composite n → 6 ∣ (n^4 - n) :=
by
  intro n
  intro hn
  -- we would add proof steps here if necessary
  sorry

end largest_divisor_of_n_pow4_minus_n_for_composites_l1332_133220


namespace hayley_initial_meatballs_l1332_133269

theorem hayley_initial_meatballs (x : ℕ) (stolen : ℕ) (left : ℕ) (h1 : stolen = 14) (h2 : left = 11) (h3 : x - stolen = left) : x = 25 := 
by 
  sorry

end hayley_initial_meatballs_l1332_133269


namespace fraction_to_zero_power_l1332_133250

theorem fraction_to_zero_power :
  756321948 ≠ 0 ∧ -3958672103 ≠ 0 →
  (756321948 / -3958672103 : ℝ) ^ 0 = 1 :=
by
  intro h
  have numerator_nonzero : 756321948 ≠ 0 := h.left
  have denominator_nonzero : -3958672103 ≠ 0 := h.right
  -- Skipping the rest of the proof.
  sorry

end fraction_to_zero_power_l1332_133250


namespace marble_color_197_l1332_133227

-- Define the types and properties of the marbles
inductive Color where
  | red | blue | green

-- Define a function to find the color of the nth marble in the cycle pattern
def colorOfMarble (n : Nat) : Color :=
  let cycleLength := 15
  let positionInCycle := n % cycleLength
  if positionInCycle < 6 then Color.red  -- first 6 marbles are red
  else if positionInCycle < 11 then Color.blue  -- next 5 marbles are blue
  else Color.green  -- last 4 marbles are green

-- The theorem asserting the color of the 197th marble
theorem marble_color_197 : colorOfMarble 197 = Color.red :=
sorry

end marble_color_197_l1332_133227


namespace symmetric_point_origin_l1332_133228

-- Define the notion of symmetry with respect to the origin
def symmetric_with_origin (p : ℤ × ℤ) : ℤ × ℤ :=
  (-p.1, -p.2)

-- Define the given point
def given_point : ℤ × ℤ :=
  (-2, 5)

-- State the theorem to be proven
theorem symmetric_point_origin : 
  symmetric_with_origin given_point = (2, -5) :=
by 
  -- The proof will go here, use sorry for now
  sorry

end symmetric_point_origin_l1332_133228


namespace roots_polynomial_sum_cubes_l1332_133200

theorem roots_polynomial_sum_cubes (u v w : ℂ) (h : (∀ x, (x = u ∨ x = v ∨ x = w) → 5 * x ^ 3 + 500 * x + 1005 = 0)) :
  (u + v) ^ 3 + (v + w) ^ 3 + (w + u) ^ 3 = 603 := sorry

end roots_polynomial_sum_cubes_l1332_133200


namespace min_tiles_needed_l1332_133284

-- Definitions for the problem
def tile_width : ℕ := 3
def tile_height : ℕ := 4

def region_width_ft : ℕ := 2
def region_height_ft : ℕ := 5

def inches_in_foot : ℕ := 12

-- Conversion
def region_width_in := region_width_ft * inches_in_foot
def region_height_in := region_height_ft * inches_in_foot

-- Calculations
def region_area := region_width_in * region_height_in
def tile_area := tile_width * tile_height

-- Theorem statement
theorem min_tiles_needed : region_area / tile_area = 120 := 
  sorry

end min_tiles_needed_l1332_133284


namespace angle_ABC_40_degrees_l1332_133280

theorem angle_ABC_40_degrees (ABC ABD CBD : ℝ) 
    (h1 : CBD = 90) 
    (h2 : ABD = 60)
    (h3 : ABC + ABD + CBD = 190) : 
    ABC = 40 := 
by {
  sorry
}

end angle_ABC_40_degrees_l1332_133280


namespace remainder_103_107_div_11_l1332_133240

theorem remainder_103_107_div_11 :
  (103 * 107) % 11 = 10 :=
by
  sorry

end remainder_103_107_div_11_l1332_133240


namespace Mary_avg_speed_l1332_133276

def Mary_uphill_distance := 1.5 -- km
def Mary_uphill_time := 45.0 / 60.0 -- hours
def Mary_downhill_distance := 1.5 -- km
def Mary_downhill_time := 15.0 / 60.0 -- hours

def total_distance := Mary_uphill_distance + Mary_downhill_distance
def total_time := Mary_uphill_time + Mary_downhill_time

theorem Mary_avg_speed : 
  (total_distance / total_time) = 3.0 := by
  sorry

end Mary_avg_speed_l1332_133276


namespace area_of_polygon_DEFG_l1332_133263

-- Given conditions
def isosceles_triangle (A B C : Type) (AB AC BC : ℝ) : Prop :=
  AB = AC ∧ AB = 2 ∧ AC = 2 ∧ BC = 1

def square (side : ℝ) : ℝ :=
  side * side

def constructed_square_areas_equal (AB AC : ℝ) (D E F G : Type) : Prop :=
  square AB = square AC ∧ square AB = 4 ∧ square AC = 4

-- Question to prove
theorem area_of_polygon_DEFG (A B C D E F G : Type) (AB AC BC : ℝ) 
  (h1 : isosceles_triangle A B C AB AC BC) 
  (h2 : constructed_square_areas_equal AB AC D E F G) : 
  square AB + square AC = 8 :=
by
  sorry

end area_of_polygon_DEFG_l1332_133263


namespace nonWhiteHomesWithoutFireplace_l1332_133261

-- Definitions based on the conditions
def totalHomes : ℕ := 400
def whiteHomes (h : ℕ) : ℕ := h / 4
def nonWhiteHomes (h w : ℕ) : ℕ := h - w
def nonWhiteHomesWithFireplace (nh : ℕ) : ℕ := nh / 5

-- Theorem statement to prove the required result
theorem nonWhiteHomesWithoutFireplace : 
  let h := totalHomes
  let w := whiteHomes h
  let nh := nonWhiteHomes h w
  let nf := nonWhiteHomesWithFireplace nh
  nh - nf = 240 :=
by
  let h := totalHomes
  let w := whiteHomes h
  let nh := nonWhiteHomes h w
  let nf := nonWhiteHomesWithFireplace nh
  show nh - nf = 240
  sorry

end nonWhiteHomesWithoutFireplace_l1332_133261


namespace complement_of_P_in_U_l1332_133270

/-- Definitions of sets U and P -/
def U := { y : ℝ | ∃ x : ℝ, x > 1 ∧ y = Real.log x / Real.log 2 }
def P := { y : ℝ | ∃ x : ℝ, x > 2 ∧ y = 1 / x }

/-- The complement of P in U -/
def complement_U_P := { y : ℝ | y = 0 ∨ y ≥ 1 / 2 }

/-- Proving the complement of P in U is as expected -/
theorem complement_of_P_in_U : { y : ℝ | y ∈ U ∧ y ∉ P } = complement_U_P := by
  sorry

end complement_of_P_in_U_l1332_133270


namespace highest_geometric_frequency_count_l1332_133255

-- Define the problem conditions and the statement to be proved
theorem highest_geometric_frequency_count :
  ∀ (vol : ℕ) (num_groups : ℕ) (cum_freq_first_seven : ℝ)
  (remaining_freqs : List ℕ) (total_freq_remaining : ℕ)
  (r : ℕ) (a : ℕ),
  vol = 100 → 
  num_groups = 10 → 
  cum_freq_first_seven = 0.79 → 
  total_freq_remaining = 21 → 
  r > 1 →
  remaining_freqs = [a, a * r, a * r ^ 2] → 
  a * (1 + r + r ^ 2) = total_freq_remaining → 
  ∃ max_freq, max_freq ∈ remaining_freqs ∧ max_freq = 12 :=
by
  intro vol num_groups cum_freq_first_seven remaining_freqs total_freq_remaining r a
  intros h_vol h_num_groups h_cum_freq_first h_total_freq_remaining h_r_pos h_geom_seq h_freq_sum
  use 12
  sorry

end highest_geometric_frequency_count_l1332_133255


namespace count_logical_propositions_l1332_133212

def proposition_1 : Prop := ∃ d : ℕ, d = 1
def proposition_2 : Prop := ∀ n : ℕ, n % 10 = 0 → n % 5 = 0
def proposition_3 : Prop := ∀ t : Prop, t → ¬t

theorem count_logical_propositions :
  (proposition_1 ∧ proposition_3) →
  (proposition_1 ∧ proposition_2 ∧ proposition_3) →
  (∃ (n : ℕ), n = 10 ∧ n % 5 = 0) ∧ n = 2 :=
sorry

end count_logical_propositions_l1332_133212


namespace sin2θ_value_l1332_133296

theorem sin2θ_value (θ : Real) (h1 : Real.sin θ = 4/5) (h2 : Real.sin θ - Real.cos θ > 1) : Real.sin (2*θ) = -24/25 := 
by 
  sorry

end sin2θ_value_l1332_133296


namespace find_c_l1332_133262

theorem find_c (c : ℕ) (h : 111111222222 = c * (c + 1)) : c = 333333 :=
by
  -- proof goes here
  sorry

end find_c_l1332_133262


namespace smallest_perimeter_consecutive_integers_triangle_l1332_133294

theorem smallest_perimeter_consecutive_integers_triangle :
  ∃ (a b c : ℕ), 
    1 < a ∧ a + 1 = b ∧ b + 1 = c ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    a + b + c = 12 :=
by
  -- proof placeholder
  sorry

end smallest_perimeter_consecutive_integers_triangle_l1332_133294


namespace solution_set_line_l1332_133299

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l1332_133299


namespace ruby_candies_l1332_133277

theorem ruby_candies (number_of_friends : ℕ) (candies_per_friend : ℕ) (total_candies : ℕ)
  (h1 : number_of_friends = 9)
  (h2 : candies_per_friend = 4)
  (h3 : total_candies = number_of_friends * candies_per_friend) :
  total_candies = 36 :=
by {
  sorry
}

end ruby_candies_l1332_133277
