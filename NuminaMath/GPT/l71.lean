import Mathlib

namespace correct_factorization_option_A_l71_71466

variable (x y : ℝ)

theorem correct_factorization_option_A :
  (2 * x^2 + 3 * x + 1 = (2 * x + 1) * (x + 1)) :=
by {
  sorry
}

end correct_factorization_option_A_l71_71466


namespace dot_product_nD_vectors_l71_71280

noncomputable def dot_product (a b : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) a b)

theorem dot_product_nD_vectors (a b : List ℝ) (n : ℕ) (h_length : a.length = n ∧ b.length = n) :
  dot_product a b = ∑ i in Finset.range n, (a.nth_le i h_length.left) * (b.nth_le i h_length.right) :=
by
  sorry

end dot_product_nD_vectors_l71_71280


namespace daria_multiple_pizzas_l71_71192

variable (m : ℝ)
variable (don_pizzas : ℝ) (total_pizzas : ℝ)

axiom don_pizzas_def : don_pizzas = 80
axiom total_pizzas_def : total_pizzas = 280

theorem daria_multiple_pizzas (m : ℝ) (don_pizzas : ℝ) (total_pizzas : ℝ) 
    (h1 : don_pizzas = 80) (h2 : total_pizzas = 280) 
    (h3 : total_pizzas = don_pizzas + m * don_pizzas) : 
    m = 2.5 :=
by sorry

end daria_multiple_pizzas_l71_71192


namespace tangent_line_equation_at_e_l71_71261

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_equation_at_e :
  let e := Real.exp 1 in
  let tangent_line := 2 * x - y - e = 0 in
  let slope := 2 in
  let point := (e, f e) in
  ∃ k b, (k = slope ∧ b = -e ∧ tangent_line) :=
by {
  sorry
}

end tangent_line_equation_at_e_l71_71261


namespace f_e_greater_f3_greater_f2_l71_71983

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

lemma f_prime (x : ℝ) : deriv f x = (1 - Real.log x) / (x^2) :=
by {
  sorry -- Skipping the proof for the derivative calculation
}

theorem f_e_greater_f3_greater_f2 : f Real.e > f 3 ∧ f 3 > f 2 :=
by {
  apply And.intro,
  -- proof for f(e) > f(3)
  sorry,
  -- proof for f(3) > f(2)
  sorry
}

end f_e_greater_f3_greater_f2_l71_71983


namespace sum_of_roots_of_quadratic_l71_71029

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l71_71029


namespace percent_50_to_59_l71_71876

theorem percent_50_to_59 (f_90_100 f_80_89 f_70_79 f_60_69 f_50_59 f_below_50 : ℕ) 
  (h_total : f_90_100 = 6 ∧ f_80_89 = 7 ∧ f_70_79 = 10 ∧ f_60_69 = 8 ∧ f_50_59 = 4 ∧ f_below_50 = 3) :
  (f_50_59 : ℕ) / (f_90_100 + f_80_89 + f_70_79 + f_60_69 + f_50_59 + f_below_50 : ℕ) * 100 ≈ 10.81 := 
by
  sorry

end percent_50_to_59_l71_71876


namespace tetrahedron_edge_length_of_tangent_spheres_l71_71973

theorem tetrahedron_edge_length_of_tangent_spheres (r : ℝ) (h₁ : r = 2) :
  ∃ s : ℝ, s = 4 :=
by
  sorry

end tetrahedron_edge_length_of_tangent_spheres_l71_71973


namespace diag_NQ_len_l71_71683

-- Definitions of the problem
variables (M N L Q : Type) [Add M] [NormedSpace ℝ M] [NormedSpace ℝ N] [NormedSpace ℝ L] [NormedSpace ℝ Q]

def LQ := (x : ℝ)
def LN := (x - 2 : ℝ)
def MK := (x : ℝ)
def QK := (x - 2 : ℝ)

-- Given conditions
variable (tan_angle_QMN : ℝ := 2 / 3)
variable (LQ_eq : LQ = MK)
variable (LQ_rel : LQ = 2 + LN)

-- The lengths found in the solution
variable (x_value : ℝ := 6)

-- Lengths from the relationships
variable (LQ_val : ℝ := x_value)
variable (LN_val : ℝ := x_value - 2)
variable (MK_val : ℝ := x_value)
variable (QK_val : ℝ := x_value - 2)

-- Set up the Pythagorean theorem relationship
noncomputable def NQ_sq := (LN_val ^ 2) + (QK_val ^ 2)

-- The goal is to prove that NQ = 2√13
theorem diag_NQ_len : ∀ (M N L Q : Type) [Add M] [NormedSpace ℝ M] [NormedSpace ℝ N] 
  [NormedSpace ℝ L] [NormedSpace ℝ Q], NQ_sq = 52 := by
  sorry

end diag_NQ_len_l71_71683


namespace solve_x_for_equation_l71_71835

theorem solve_x_for_equation :
  ∃ (x : ℚ), 3 * x - 5 = abs (-20 + 6) ∧ x = 19 / 3 :=
by
  sorry

end solve_x_for_equation_l71_71835


namespace sum_c_d_is_eight_l71_71705

-- Define the points P, Q, R, S
def P : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (3, 4)
def R : ℝ × ℝ := (6, 1)
def S : ℝ × ℝ := (8, -1)

-- Define distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Calculate distances between points
def PQ : ℝ := distance P Q
def QR : ℝ := distance Q R
def RS : ℝ := distance R S
def SP : ℝ := distance S P

-- Calculate perimeter
def perimeter : ℝ := PQ + QR + RS + SP

theorem sum_c_d_is_eight :
  let c := 7 in
  let d := 1 in
  c + d = 8 := by
    sorry

end sum_c_d_is_eight_l71_71705


namespace sum_of_roots_eq_14_l71_71074

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l71_71074


namespace range_of_x_l71_71967

theorem range_of_x (a : ℝ) (x : ℝ) (h : a ∈ set.Icc (-1:ℝ) 1) :
  x^2 + (a-4) * x + 4 - 2 * a > 0 ↔ x < 1 ∨ x > 3 :=
by
  sorry

end range_of_x_l71_71967


namespace sum_of_roots_eq_14_l71_71088

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l71_71088


namespace volume_of_tetrahedron_l71_71762

/-- Given the side lengths of a tetrahedron PQRS, prove its volume -/
theorem volume_of_tetrahedron (PQ PR PS QR QS : ℝ) (RS : ℝ) (hPQ : PQ = 6) (hPR : PR = 4) 
  (hPS : PS = 5) (hQR : QR = 5) (hQS : QS = 4) (hRS : RS = (7 / 5) * sqrt 11) : 
  ∃ V : ℝ, V = (7 / 2) * sqrt 22 := 
  sorry

end volume_of_tetrahedron_l71_71762


namespace sum_real_imag_part_l71_71767

theorem sum_real_imag_part (z : ℂ) (h : z * (2 + complex.I) = 2 * complex.I - 1) : 
  z.re + z.im = 1 := 
sorry

end sum_real_imag_part_l71_71767


namespace fraction_of_shoppers_avoiding_checkout_l71_71864

theorem fraction_of_shoppers_avoiding_checkout 
  (total_shoppers : ℕ) 
  (shoppers_at_checkout : ℕ) 
  (h1 : total_shoppers = 480) 
  (h2 : shoppers_at_checkout = 180) : 
  (total_shoppers - shoppers_at_checkout) / total_shoppers = 5 / 8 :=
by
  sorry

end fraction_of_shoppers_avoiding_checkout_l71_71864


namespace max_segment_length_l71_71996

noncomputable def max_length_segment_MN (a b c Mx My : ℝ) : ℝ :=
  let A := (0 : ℝ, -1 : ℝ)
  let N := (3 : ℝ, 3 : ℝ)
  dist (0, -1) (3, 3) + sqrt 2

theorem max_segment_length (a b c Mx My : ℝ) 
  (h1 : 2 * b = a + c) 
  (h2 : forall M : ℝ × ℝ, (M.fst = Mx ∧ M.snd = My) → 
    dist (Mx, My) (-1, 0) = dist (0, -1) (-1, 0)) :
  max_length_segment_MN a b c Mx My = 5 + sqrt 2 :=
  sorry

end max_segment_length_l71_71996


namespace sum_of_roots_eq_14_l71_71093

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l71_71093


namespace tan_phi_l71_71326

-- Definition of the problem conditions
variable (β φ : ℝ)

-- Given condition: the tangent of half of β
def given_condition : Prop := tan (β / 2) = 1 / (2 ^ (1 / 4))

-- The statement to prove
theorem tan_phi (h : given_condition β φ) : tan φ = 1 / 2 := 
sorry

end tan_phi_l71_71326


namespace count_multiples_of_7_with_units_digit_7_l71_71291

theorem count_multiples_of_7_with_units_digit_7 :
  (finset.filter (λ n, (n % 10 = 7)) (finset.range 200).filter (λ n, n % 7 = 0)).card = 3 :=
by
  sorry

end count_multiples_of_7_with_units_digit_7_l71_71291


namespace circumscribed_area_right_triangle_l71_71212

open Real

theorem circumscribed_area_right_triangle (a b c : ℝ) (u v : ℝ)
  (hroot1 : a * u^2 + b * u + c = 0)
  (hroot2 : a * v^2 + b * v + c = 0) :
  let w := sqrt (u * u + v * v),
      R := w / 2 in
  π * R^2 = π * (b^2 - 2 * a * c) / (4 * a^2) :=
begin
  -- Proof will be inserted here
  sorry
end

end circumscribed_area_right_triangle_l71_71212


namespace pairs_of_values_l71_71960

theorem pairs_of_values (x y : ℂ) :
  (y = (x + 2)^3 ∧ x * y + 2 * y = 2) →
  (∃ (r1 r2 i1 i2 : ℂ), (r1.im = 0 ∧ r2.im = 0) ∧ (i1.im ≠ 0 ∧ i2.im ≠ 0) ∧ 
    ((r1, (r1 + 2)^3) = (x, y) ∨ (r2, (r2 + 2)^3) = (x, y) ∨
     (i1, (i1 + 2)^3) = (x, y) ∨ (i2, (i2 + 2)^3) = (x, y))) :=
sorry

end pairs_of_values_l71_71960


namespace div_neg_forty_five_l71_71915

theorem div_neg_forty_five : (-40 / 5) = -8 :=
by
  sorry

end div_neg_forty_five_l71_71915


namespace violin_enjoyment_claims_l71_71553

theorem violin_enjoyment_claims :
  ∀ (total students j e : ℕ)
  (p_j : j = 40)
  (p_e : e = 60)
  (p_accurate_j : 70% of j accurately claim they enjoy it)
  (p_false_j : 30% of j falsely claim they dislike it)
  (p_accurate_e : 80% of e accurately claim they dislike it)
  (p_false_e : 20% of e falsely claim they enjoy it),
  (fraction of students who claim dislike but enjoy playing
  is equal to 20%) :=
by sorry


end violin_enjoyment_claims_l71_71553


namespace unique_three_digit_multiple_of_66_ending_in_4_l71_71292

theorem unique_three_digit_multiple_of_66_ending_in_4 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 66 = 0 ∧ n % 10 = 4 := sorry

end unique_three_digit_multiple_of_66_ending_in_4_l71_71292


namespace decreasing_interval_l71_71574

noncomputable def f : ℝ → ℝ := λ x, x^3 + 3 * x^2 - 9 * x

theorem decreasing_interval : ∀ x, -3 < x ∧ x < 1 → f' x < 0 :=
begin
  intros x hx,
  sorry
end

end decreasing_interval_l71_71574


namespace inequality_solution_minimum_value_inequality_l71_71266

section Part1

def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 2 * x + 1) + Real.sqrt (x^2 - 10 * x + 25)

theorem inequality_solution :
  {x : ℝ | f x > 6} = {x : ℝ | x < 0 ∨ x > 6} :=
by
  sorry

end Part1

section Part2

variable (a b c : ℝ) (m : ℝ := 4)

theorem minimum_value_inequality
  (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (condition : 1/a + 1/(2*b) + 1/(3*c) = 1) :
  a + 2 * b + 3 * c ≥ 9 :=
by
  sorry

end Part2

end inequality_solution_minimum_value_inequality_l71_71266


namespace fractions_product_eq_one_l71_71694

theorem fractions_product_eq_one : ∃ (a b c : ℚ), a * b * c = 1 ∧ 
  (a ∈ ({k / (2018 - k) | k : ℕ ∧ k > 0 ∧ k ≤ 2017} : set ℚ)) ∧
  (b ∈ ({k / (2018 - k) | k : ℕ ∧ k > 0 ∧ k ≤ 2017} : set ℚ)) ∧
  (c ∈ ({k / (2018 - k) | k : ℕ ∧ k > 0 ∧ k ≤ 2017} : set ℚ)) :=
by {
  let fractions := {k / (2018 - k) | k : ℕ ∧ k > 0 ∧ k ≤ 2017},
  existsi (1 / 2017 : ℚ),
  existsi (1009 / 1009 : ℚ),
  existsi (2017 / 1 : ℚ),
  split,
  { calc (1 / 2017) * (1009 / 1009) * (2017 / 1)
        = (1 * 1009 * 2017) / (2017 * 1009 * 1) : by ring
    ... = 1 : by norm_num },
  split,
  { exact set.mem_image_of_mem _ (by norm_num) },
  split,
  { exact set.mem_image_of_mem _ (by norm_num) },
  { exact set.mem_image_of_mem _ (by norm_num) }
}

end fractions_product_eq_one_l71_71694


namespace unique_function_satisfying_condition_l71_71952

open Function

theorem unique_function_satisfying_condition :
  ∀ f : ℤ → ℤ, (∀ n : ℤ, n^2 + 4 * f(n) = f(f(n))^2) → (∀ n : ℤ, f(n) = n + 1) :=
by
  sorry

end unique_function_satisfying_condition_l71_71952


namespace possible_rankings_l71_71586

-- Define the teams
inductive Team
| A | B | C | D | E | F

open Team

-- Define the conditions of the matches
def saturday_matches : List (Team × Team) :=
[(A, B), (C, D), (E, F)]

-- The main theorem to prove the number of possible ranking sequences
theorem possible_rankings : 
  let winners := 3!
  let losers := 3!
  winners * losers = 36 :=
by
  let winners := Nat.factorial 3
  let losers := Nat.factorial 3
  have h₁ : winners = 6 := by sorry
  have h₂ : losers = 6 := by sorry
  have h₃ : winners * losers = 36 := by sorry
  exact h₃

end possible_rankings_l71_71586


namespace sum_of_roots_eq_l71_71059

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l71_71059


namespace exist_uv_solution_l71_71936

theorem exist_uv_solution :
  ∃ (u v : ℝ), (λ (u v : ℝ), (3 + 8 * u = 2 - 3 * v ∧ 1 - 6 * u = -2 + 4 * v)) (-13/14) (15/7) := 
by
  exists -13/14, 15/7
  split
  { linarith }
  { linarith }

end exist_uv_solution_l71_71936


namespace max_value_of_f_l71_71219

noncomputable def f (x : ℝ) : ℝ := x^6 / (x^10 + 3 * x^8 - 6 * x^6 + 12 * x^4 + 32)

theorem max_value_of_f : ∃ x : ℝ, ∀ y : ℝ, f(y) ≤ 1/18 :=
by
  sorry

end max_value_of_f_l71_71219


namespace campers_in_afternoon_l71_71129

theorem campers_in_afternoon (total_campers morning_campers : ℕ) (h1 : total_campers = 32) (h2 : morning_campers = 15) :
  total_campers - morning_campers = 17 :=
by
  rw [h1, h2]
  norm_num
  sorry

end campers_in_afternoon_l71_71129


namespace total_marbles_l71_71668

-- Definitions to state the problem
variables {r b g : ℕ}
axiom ratio_condition : r / b = 2 / 4 ∧ r / g = 2 / 6
axiom blue_marbles : b = 30

-- Theorem statement
theorem total_marbles : r + b + g = 90 :=
by sorry

end total_marbles_l71_71668


namespace common_root_exists_l71_71591

noncomputable def cubic_polynomial (a b c d : ℝ) : polynomial ℝ :=
  polynomial.C d + polynomial.X * (polynomial.C c + polynomial.X * (polynomial.C b + polynomial.C a * polynomial.X))

theorem common_root_exists : ∃ c d : ℝ, 
  (cubic_polynomial 1 c 15 10).is_root 1 ∧ (cubic_polynomial 1 d 17 12).is_root 1 ∧ c = -3 ∧ d = -4 :=
by {
  let r := 1,
  use [-3, -4],
  have hc : (cubic_polynomial 1 (-3) 15 10).is_root r,
  { simp [cubic_polynomial, polynomial.is_root, polynomial.eval] },
  have hd : (cubic_polynomial 1 (-4) 17 12).is_root r,
  { simp [cubic_polynomial, polynomial.is_root, polynomial.eval] },
  exact ⟨hc, hd, rfl, rfl⟩,
}

end common_root_exists_l71_71591


namespace intersection_A_B_l71_71228

def f (x : ℝ) : ℝ := x^2 - 2*x

def A : set ℝ := {x | f x < 0}

def B : set ℝ := {x | (deriv f x) > 0}

theorem intersection_A_B :
  A ∩ B = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l71_71228


namespace sum_of_roots_eq_l71_71054

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l71_71054


namespace part1_part2_l71_71371

-- Given Definitions
variable (p : ℕ) [hp : Fact (p > 3)] [prime : Fact (Nat.Prime p)]
variable (A_l : ℕ → ℕ)

-- Assertions to Prove
theorem part1 (l : ℕ) (hl : 1 ≤ l ∧ l ≤ p - 2) : A_l l % p = 0 :=
sorry

theorem part2 (l : ℕ) (hl : 1 < l ∧ l < p ∧ l % 2 = 1) : A_l l % (p * p) = 0 :=
sorry

end part1_part2_l71_71371


namespace bill_apples_left_l71_71911

-- Definitions based on the conditions
def total_apples : Nat := 50
def apples_per_child : Nat := 3
def number_of_children : Nat := 2
def apples_per_pie : Nat := 10
def number_of_pies : Nat := 2

-- The main statement to prove
theorem bill_apples_left : total_apples - ((apples_per_child * number_of_children) + (apples_per_pie * number_of_pies)) = 24 := by
sorry

end bill_apples_left_l71_71911


namespace planks_needed_for_table_l71_71699

theorem planks_needed_for_table
  (trees : ℕ)
  (planks_per_tree : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ)
  (profit : ℕ)
  (total_planks : ℕ := trees * planks_per_tree)
  (total_revenue : ℕ := profit + labor_cost)
  (tables_made : ℕ := total_revenue / price_per_table)
  (planks_per_table : ℕ := total_planks / tables_made) :
  trees = 30 ∧
  planks_per_tree = 25 ∧
  price_per_table = 300 ∧
  labor_cost = 3000 ∧
  profit = 12000 →
  planks_per_table = 15 :=
begin
  intros,
  -- proof to be done
  sorry
end

end planks_needed_for_table_l71_71699


namespace pairwise_sums_l71_71724

theorem pairwise_sums (
  a b c d e : ℕ
) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  (a + b = 21) ∧ (a + c = 26) ∧ (a + d = 35) ∧ (a + e = 40) ∧
  (b + c = 49) ∧ (b + d = 51) ∧ (b + e = 54) ∧ (c + d = 60) ∧
  (c + e = 65) ∧ (d + e = 79)
  ↔ 
  (a = 6) ∧ (b = 15) ∧ (c = 20) ∧ (d = 34) ∧ (e = 45) := 
by 
  sorry

end pairwise_sums_l71_71724


namespace joan_socks_remaining_l71_71354

-- Definitions based on conditions
def total_socks : ℕ := 1200
def white_socks : ℕ := total_socks / 4
def blue_socks : ℕ := total_socks * 3 / 8
def red_socks : ℕ := total_socks / 6
def green_socks : ℕ := total_socks / 12
def white_socks_lost : ℕ := white_socks / 3
def blue_socks_sold : ℕ := blue_socks / 2
def remaining_white_socks : ℕ := white_socks - white_socks_lost
def remaining_blue_socks : ℕ := blue_socks - blue_socks_sold

-- Theorem to prove the total number of remaining socks
theorem joan_socks_remaining :
  remaining_white_socks + remaining_blue_socks + red_socks + green_socks = 725 := by
  sorry

end joan_socks_remaining_l71_71354


namespace number_of_sets_B_l71_71272

open Set

theorem number_of_sets_B (A : Set ℕ) (B : Set ℕ) : A = {1, 2} ∧ A ∪ B = {1, 2, 3} → {B | A ∪ B = {1, 2, 3}}.to_finset.card = 4 := by
  intro h
  simp [Set.ext_iff] at h
  sorry

end number_of_sets_B_l71_71272


namespace find_x_for_h_x_eq_x_l71_71368

variable (h : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
axiom h_def : ∀ x : ℝ, h(4 * x - 1) = 2 * x + 7

-- Goal
theorem find_x_for_h_x_eq_x : h 15 = 15 :=
by
  sorry

end find_x_for_h_x_eq_x_l71_71368


namespace gcf_4370_13824_l71_71005

/-- Define the two numbers 4370 and 13824 -/
def num1 := 4370
def num2 := 13824

/-- The statement that the GCF of num1 and num2 is 1 -/
theorem gcf_4370_13824 : Nat.gcd num1 num2 = 1 := by
  sorry

end gcf_4370_13824_l71_71005


namespace machine_working_time_l71_71810

theorem machine_working_time (y : ℝ) :
  (1 / (y + 4) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) → y = 2 :=
by
  sorry

end machine_working_time_l71_71810


namespace sam_pennies_l71_71740

theorem sam_pennies (initial : ℕ) (spent : ℕ) (remaining : ℕ) 
  (h_initial : initial = 989) (h_spent : spent = 728) (h_remaining : remaining = initial - spent) 
  : remaining = 261 :=
by
  rw [h_initial, h_spent]
  exact h_remaining

end sam_pennies_l71_71740


namespace remainders_of_65_powers_l71_71940

theorem remainders_of_65_powers (n : ℕ) :
  (65 ^ (6 * n)) % 9 = 1 ∧
  (65 ^ (6 * n + 1)) % 9 = 2 ∧
  (65 ^ (6 * n + 2)) % 9 = 4 ∧
  (65 ^ (6 * n + 3)) % 9 = 8 :=
by
  sorry

end remainders_of_65_powers_l71_71940


namespace general_term_an_sum_of_bn_min_positive_integer_m_l71_71620

namespace SequenceProofs

open Nat

-- (1) Prove the general term a_n of the sequence {a_n}
theorem general_term_an (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : a 1 = 1) 
                         (h2 : ∀ n : ℕ, n > 0 → S n + 1 = a (n + 1)) :
    ∀ n : ℕ, n > 0 → a n = 2^(n - 1) :=
sorry

-- (2) Prove the sum of the first n terms T_n of the sequence {b_n}
theorem sum_of_bn (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℚ)
                  (h1 : ∀ n : ℕ, n > 0 → a n = 2^(n - 1))
                  (h2 : ∀ n : ℕ, n > 0 → b n = n / (4 * a n)) :
    ∀ n : ℕ, n > 0 → T n = 1 - (n + 2) / 2^(n + 1) :=
sorry

-- (3) Prove the minimum positive integer m such that A_n < m holds
theorem min_positive_integer_m (S : ℕ → ℕ) (T : ℕ → ℚ) (c : ℕ → ℚ) (A : ℕ → ℚ)
                               (h1 : ∀ n : ℕ, n > 0 → S n = (1 + a n))
                               (h2 : ∀ n : ℕ, n > 0 → a n = 2^(n - 1))
                               (h3 : ∀ n : ℕ, n > 0 → T n = 1 - (n + 2) / 2^(n + 1))
                               (h4 : ∀ k : ℕ, k > 0 → c k = (k + 2) / (S k * (T k + k + 1)))
                               (h5 : ∀ n : ℕ, n > 0 → A n = (∑ k in range n, c k)) :
    ∃ m : ℕ, m > 0 ∧ (∀ n : ℕ, n > 0 → A n < m) ∧ m = 2 :=
sorry

end SequenceProofs

end general_term_an_sum_of_bn_min_positive_integer_m_l71_71620


namespace sum_of_roots_eq_14_l71_71045

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l71_71045


namespace valid_statements_l71_71930

theorem valid_statements (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (sqrt (a^2 + b^2) = a - b) ∨ (a^2 + b^2 = a^3 + b^3) :=
by
  sorry

end valid_statements_l71_71930


namespace seq_2005th_term_l71_71425

-- Defining the sequence 
def seq : ℕ → ℕ
| 0       := 2005
| (n + 1) := let digits := ((seq n).digits 10) in
             ((digits.map (λ d => d^3)).sum)

theorem seq_2005th_term : seq 2004 = 250 :=
by
  sorry

end seq_2005th_term_l71_71425


namespace combined_weight_l71_71384

theorem combined_weight (x y z : ℕ) (h1 : x + y = 110) (h2 : y + z = 130) (h3 : z + x = 150) : x + y + z = 195 :=
by
  sorry

end combined_weight_l71_71384


namespace sum_of_roots_eq_14_l71_71051

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l71_71051


namespace parallelogram_area_l71_71396

theorem parallelogram_area (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (angle_A : ℝ) (side_AB : ℝ) (side_BC : ℝ) (h_angle_A : angle_A = π * 3 / 4)
  (h_side_AB : side_AB = 17) (h_side_BC : side_BC = 10) :
  let height_BD := 5 * real.sqrt 2 in
  side_AB * height_BD = 85 * real.sqrt 2 := 
begin
  sorry
end

end parallelogram_area_l71_71396


namespace carnival_game_ratio_l71_71176

theorem carnival_game_ratio (L W : ℕ) (h_ratio : 4 * L = W) (h_lost : L = 7) : W = 28 :=
by {
  sorry
}

end carnival_game_ratio_l71_71176


namespace magnitude_sum_l71_71281

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 2)

-- To prove: the magnitude of a + b is 5
theorem magnitude_sum (a b : ℝ × ℝ) : real.sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) = 5 := by
  -- Using the given vectors a and b
  have ha : a = (1, 2) := rfl
  have hb : b = (2, 2) := rfl
  sorry

end magnitude_sum_l71_71281


namespace number_of_solutions_l71_71939

theorem number_of_solutions (x : ℝ) :
  (|x - |2 * x - 3|| = 5) → 
  (x = 8) ∨ (x = -2/3) :=
begin
  sorry
end

end number_of_solutions_l71_71939


namespace longest_side_of_acute_triangle_gt_25_l71_71783

theorem longest_side_of_acute_triangle_gt_25 :
  ∀ (x : ℝ), 
  (x > 5 ∧ 0 < ((x - 15) / (2 * x)) < 1) 
    → (x + 10 > 25) :=
by
  sorry

end longest_side_of_acute_triangle_gt_25_l71_71783


namespace negation_of_exists_l71_71787

theorem negation_of_exists (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end negation_of_exists_l71_71787


namespace part1_part2_l71_71265

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x^2 - 2*x + 1) + real.sqrt (x^2 - 10*x + 25)

theorem part1 (x : ℝ) : (f x > 6) ↔ (x < 0 ∨ x > 6) :=
sorry

theorem part2 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : 1/a + 1/(2*b) + 1/(3*c) = 1) :
  a + 2*b + 3*c ≥ 9 :=
sorry

end part1_part2_l71_71265


namespace closed_convex_curve_with_equal_tangents_is_circle_l71_71867

theorem closed_convex_curve_with_equal_tangents_is_circle 
  (γ : Set Point) 
  (h₁ : IsClosedConvexCurve γ) 
  (h₂ : ∀P ∉ γ, ∀p₁ p₂ ∈ PointsOfTangency(γ, P), Distance(P, p₁) = Distance(P, p₂)) 
  : IsCircle(γ) := 
sorry

end closed_convex_curve_with_equal_tangents_is_circle_l71_71867


namespace product_is_168_l71_71393

-- Define the coordinates of the points
structure Point where
  x : ℕ
  y : ℕ

-- Define the points E, F, G, H
def E : Point := {x := 1, y := 5}
def F : Point := {x := 5, y := 5}
def G : Point := {x := 5, y := 2}
def H : Point := {x := 1, y := 2}

-- Define the function to calculate the distance between points
def distance (a b : Point) : ℕ :=
  if a.x = b.x then abs (a.y - b.y)
  else if a.y = b.y then abs (a.x - b.x)
  else 0 -- Should not happen in the context of this problem

-- Define the lengths of the sides of the rectangle
def length_EF : ℕ := distance E F
def length_EH : ℕ := distance E H

-- Define the area and perimeter of the rectangle
def area : ℕ := length_EF * length_EH
def perimeter : ℕ := 2 * (length_EF + length_EH)

-- Define the product of the area and the perimeter
def product_of_area_and_perimeter : ℕ := area * perimeter

-- Lean statement to prove the final product
theorem product_is_168 :
  product_of_area_and_perimeter = 168 :=
by
  -- Definitions based on given coordinates
  have length_EF_def : length_EF = 4 := by simp [length_EF, distance]
  have length_EH_def : length_EH = 3 := by simp [length_EH, distance]
  have area_def : area = 12 := by rw [length_EF_def, length_EH_def]; simp [area]
  have perimeter_def : perimeter = 14 := by rw [length_EF_def, length_EH_def]; simp [perimeter]
  -- Resulting product
  rw [length_EF_def, length_EH_def, area_def, perimeter_def, product_of_area_and_perimeter]
  simp [product_of_area_and_perimeter]; sorry

end product_is_168_l71_71393


namespace right_triangle_construction_l71_71568

variable (O C : Point) (L : ℝ)

-- Assume O and C are distinct points
axiom distinct_points : O ≠ C

-- Assume A and B are points lying on perpendicular lines OX, OY
variable (A B : Point)
axiom A_on_OX : LiesOnLine A OX
axiom B_on_OY : LiesOnLine B OY

-- Assume AB is the hypotenuse with a given length L
axiom hypotenuse_length : dist A B = L

-- Assume ∠ACB = 90°
axiom right_angle_triangle : angle A C B = 90

-- Assume OC ≤ L
axiom OC_le_L : dist O C ≤ L

-- Goal: Prove that two distinct triangles if OC < AB and one triangle if OC = AB
theorem right_triangle_construction :
  (dist O C < L → ∃ A1 B1 A2 B2, A1 ≠ A2 ∧ right_angle_triangle A1 C B1 ∧ right_angle_triangle A2 C B2) ∧
  (dist O C = L → ∃! (A B : Point), right_angle_triangle A C B) := by
  sorry

end right_triangle_construction_l71_71568


namespace power_of_two_representation_l71_71401

/-- Prove that any number 2^n, where n = 3,4,5,..., can be represented 
as 7x^2 + y^2 where x and y are odd numbers. -/
theorem power_of_two_representation (n : ℕ) (hn : n ≥ 3) : 
  ∃ x y : ℤ, (2*x ≠ 0 ∧ 2*y ≠ 0) ∧ 2^n = 7 * x^2 + y^2 :=
by
  sorry

end power_of_two_representation_l71_71401


namespace shortest_path_count_is_45_l71_71552

-- Define the points A, B, C in terms of their coordinates
def A : (ℕ × ℕ) := (0, 0)
def B : (ℕ × ℕ) := (x_B, y_B)
def C : (ℕ × ℕ) := (x_C, y_C)

-- Define a function to compute the number of shortest paths in a grid
noncomputable def number_of_shortest_paths (A B C : (ℕ × ℕ)) : ℕ := sorry

-- Statement of the problem
theorem shortest_path_count_is_45 (x_B y_B x_C y_C : ℕ) :
  number_of_shortest_paths A B C = 45 := 
sorry

end shortest_path_count_is_45_l71_71552


namespace line_angle_through_origin_l71_71432

noncomputable def angle_of_inclination (p1 p2 : Point) : Real :=
  let m := (p2.y - p1.y) / (p2.x - p1.x)  -- Calculating the slope
  Real.atan m * (180 / Real.pi)   -- Convert radians to degrees

theorem line_angle_through_origin :
  let p1 := ⟨0, 0⟩ in
  let p2 := ⟨-1, -1⟩ in
  angle_of_inclination p1 p2 = 45 :=
by
  sorry

end line_angle_through_origin_l71_71432


namespace overall_gain_percentage_correct_l71_71146

-- Define the cost prices and selling prices
def CPcycle : ℝ := 900
def SPcycle : ℝ := 1170
def CPscooter : ℝ := 15000
def SPscooter : ℝ := 18000
def CPskateboard : ℝ := 2000
def SPskateboard : ℝ := 2400

-- Define the overall gain percentage calculation
def overall_gain_percentage (CPcycle SPcycle CPscooter SPscooter CPskateboard SPskateboard : ℝ) : ℝ :=
  let total_CP := CPcycle + CPscooter + CPskateboard
  let total_SP := SPcycle + SPscooter + SPskateboard
  let total_gain := total_SP - total_CP
  (total_gain / total_CP) * 100

-- Given problem statement
theorem overall_gain_percentage_correct :
  overall_gain_percentage CPcycle SPcycle CPscooter SPscooter CPskateboard SPskateboard = 20.50 :=
by
  sorry

end overall_gain_percentage_correct_l71_71146


namespace solve_quadratic_problem_l71_71745

theorem solve_quadratic_problem :
  ∀ x : ℝ, (x^2 + 6 * x + 8 = -(x + 4) * (x + 7)) ↔ (x = -4 ∨ x = -4.5) := by
  sorry

end solve_quadratic_problem_l71_71745


namespace perpendicular_line_slope_l71_71459

-- Definitions of given conditions
def point1 := (2, 3) : ℤ × ℤ
def point2 := (7, 8) : ℤ × ℤ

-- Define the slope function
def slope (A B : ℤ × ℤ) : ℚ := (B.2 - A.2) / (B.1 - A.1 : ℚ)

-- Define the function to find the perpendicular slope
def perpendicular_slope (s : ℚ) : ℚ := -1 / s

-- The slope of the line passing through the given points
def original_slope := slope point1 point2

-- Define the expected slope of the perpendicular line
def expected_perpendicular_slope := -1

-- The statement that we need to prove
theorem perpendicular_line_slope : perpendicular_slope original_slope = expected_perpendicular_slope := by
  sorry

end perpendicular_line_slope_l71_71459


namespace solve_system_l71_71749

open Real

theorem solve_system (x y : ℝ) : 
  (1/2 * log 2 x - log 2 y = 0) ∧ (x^2 - 2 * y^2 = 8) → ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = -2)) :=
by
  sorry

end solve_system_l71_71749


namespace andrea_rhinestones_ratio_l71_71548

theorem andrea_rhinestones_ratio :
  (∃ (B : ℕ), B = 45 - (1 / 5 * 45) - 21) →
  (1/5 * 45 : ℕ) + B + 21 = 45 →
  (B : ℕ) / 45 = 1 / 3 := 
sorry

end andrea_rhinestones_ratio_l71_71548


namespace Sadie_l71_71408

theorem Sadie's_homework_problems (T : ℝ) 
  (h1 : 0.40 * T = A) 
  (h2 : 0.5 * A = 28) 
  : T = 140 := 
by
  sorry

end Sadie_l71_71408


namespace find_n_from_equation_l71_71461

theorem find_n_from_equation : ∃ n : ℤ, n + (n + 1) + (n + 2) + (n + 3) = 22 ∧ n = 4 :=
by
  sorry

end find_n_from_equation_l71_71461


namespace sum_of_roots_l71_71106

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l71_71106


namespace blue_zone_points_l71_71523

noncomputable def target : Type := unit

-- Conditions as definitions
def bullseye_radius : ℝ := 1
def bullseye_points : ℝ := 315
def blue_ring_outer_radius : ℝ := 4
def blue_ring_inner_radius : ℝ := 3

-- The target consists of five zones with the specified properties
def target_zones := (bullseye_radius, bullseye_points, blue_ring_outer_radius, blue_ring_inner_radius)

-- Required to prove the points for hitting the blue ring
theorem blue_zone_points (r : ℝ) (bp : ℝ) (or br : ℝ) : ℝ :=
by
  let blue_ring_area := real.pi * (or ^ 2 - br ^ 2)
  let bullseye_area := real.pi * (r ^ 2)
  let blue_to_bullseye_ratio := blue_ring_area / bullseye_area
  let blue_points := bp / blue_to_bullseye_ratio
  have blue_points_eq_45 : blue_points = 45 := 
    by sorry
  exact blue_points

#eval blue_zone_points bullseye_radius bullseye_points blue_ring_outer_radius blue_ring_inner_radius

end blue_zone_points_l71_71523


namespace min_cells_marked_l71_71828

def is_adjacent (a b : (ℕ × ℕ)) : Bool :=
  let (ax, ay) := a
  let (bx, by) := b
  (ax = bx && (ay = by + 1 || ay = by - 1)) ||
  (ay = by && (ax = bx + 1 || ax = bx - 1)) ||
  ((ax = bx + 1 || ax = bx - 1) && (ay = by + 1 || ay = by - 1))

def valid_marking (markings : List (ℕ × ℕ)) : Bool :=
  markings.pairwise (λ a b, ¬ is_adjacent a b)

theorem min_cells_marked : ∃ (markings : List (ℕ × ℕ)), valid_marking markings ∧ 
    markings.length = 4 ∧ 
    ∀ additional_cell, ¬ valid_marking (additional_cell :: markings) :=
sorry

end min_cells_marked_l71_71828


namespace solve_system_l71_71752

open Real

theorem solve_system :
  (∃ x y : ℝ, (1 / 2) * log 2 x - log 2 y = 0 ∧ x^2 - 2 * y^2 = 8 ∧
    ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = -2))) :=
by 
  sorry

end solve_system_l71_71752


namespace simplify_expr_1_simplify_expr_2_l71_71412

-- Problem 1: Simplify the algebraic expression
theorem simplify_expr_1 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  6 * (8 * a^3 / (125 * b^3))^4 * (8 * a^(-3) / (27 * b^6))^(-1/3) = 6 / 25 * a^3 := 
by sorry

-- Problem 2: Simplify the logarithmic expression
theorem simplify_expr_2 : 
  (log 10 2) * ((log (nat.sqrt base := 2) 5) + (1 / log base 2 e)) = 2 := 
by sorry

end simplify_expr_1_simplify_expr_2_l71_71412


namespace total_amount_paid_l71_71282

section
variables (qty_grapes rate_grapes qty_mangoes rate_mangoes : ℕ)
def cost_grapes := qty_grapes * rate_grapes
def cost_mangoes := qty_mangoes * rate_mangoes
def total_cost := cost_grapes + cost_mangoes

theorem total_amount_paid : 
  qty_grapes = 8 → rate_grapes = 70 →
  qty_mangoes = 9 → rate_mangoes = 60 →
  total_cost qty_grapes rate_grapes qty_mangoes rate_mangoes = 1100 :=
by intros; sorry
end

end total_amount_paid_l71_71282


namespace rational_function_sum_l71_71428

noncomputable def p (x : ℝ) : ℝ := (2 / 3) * (x - 2)
noncomputable def q (x : ℝ) : ℝ := (-4 / 3) * (x - 2)

theorem rational_function_sum :
  (p (-1) = 2) ∧ (q (-1) = 4) ∧ (degree q = 2) ∧ 
  (∃ l, filter.tendsto (λ (x: ℝ), p x / q x) filter.at_top (nhds 1)) ∧ 
  (∃ l, filter.tendsto (λ (x: ℝ), p x / q x) (nhds 2) filter.at_top) →
    p x + q x = (-2 / 3) * x + (4 / 3) :=
by
  sorry

end rational_function_sum_l71_71428


namespace mitchell_chews_54_pieces_l71_71379

theorem mitchell_chews_54_pieces : 
  ∀ (packets pieces_not_chewed pieces_per_packet total_pieces pieces_chewed : ℕ),
    packets = 8 →
    pieces_per_packet = 7 →
    pieces_not_chewed = 2 →
    total_pieces = packets * pieces_per_packet →
    pieces_chewed = total_pieces - pieces_not_chewed →
    pieces_chewed = 54 :=
by
  intros packets pieces_not_chewed pieces_per_packet total_pieces pieces_chewed h_packet h_per_packet h_not_chewed h_total h_chewed
  rw [h_packet, h_per_packet] at h_total
  rw [h_total] at h_chewed
  simp at h_chewed
  assumption

end mitchell_chews_54_pieces_l71_71379


namespace three_digit_numbers_middle_twice_average_l71_71293
theorem three_digit_numbers_middle_twice_average :
  let digit (n : ℕ) := n < 10
  in ∃ (a b c : ℕ), digit a ∧ digit b ∧ digit c ∧
  a ≠ 0 ∧ b = a + c ∧ (b < 10) ∧
  (a ∈ range 10 \{ 0 }) ∧ 
  (∑ a in finset.range(10).filter(λa, a > 0), ∑ c in finset.range(10).filter(λc, a + c < 10), 1) = 45

end three_digit_numbers_middle_twice_average_l71_71293


namespace height_of_parallelogram_l71_71454

noncomputable def parallelogram_height (base area : ℝ) : ℝ :=
  area / base

theorem height_of_parallelogram :
  parallelogram_height 8 78.88 = 9.86 :=
by
  -- This is where the proof would go, but it's being omitted as per instructions.
  sorry

end height_of_parallelogram_l71_71454


namespace box_height_equation_l71_71519

theorem box_height_equation (x : ℝ) : 
  let original_length := 8
  let original_width := 6
  let original_area := original_length * original_width
  let desired_base_area := (2 / 3) * original_area
  let new_length := original_length - 2 * x
  let new_width := original_width - 2 * x
  new_length * new_width = 32 := 
begin
  assume (x : ℝ),
  let original_length := 8,
  let original_width := 6,
  let original_area := original_length * original_width,
  let desired_base_area := (2 / 3) * original_area,
  let new_length := original_length - 2 * x,
  let new_width := original_width - 2 * x,
  show new_length * new_width = 32,
  sorry
end

end box_height_equation_l71_71519


namespace tangent_lines_to_minimized_circle_l71_71632

noncomputable def circle_eq (m : ℝ) : ℝ × ℝ → ℝ :=
  λ ⟨x, y⟩, x^2 + y^2 - 4 * m * x - 2 * y + 8 * m - 7

noncomputable def is_tangent_line (L : ℝ × ℝ → ℝ) (circle : ℝ × ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  L p = 0 ∧ ∃ slope k, (L = λ ⟨x, y⟩, k * x - y + k * p.1 - p.2) ∧
    ∀ x y, circle (x, y) = 0 → L (x, y) = 0

theorem tangent_lines_to_minimized_circle :
  ∃ m : ℝ, (∀ m', m' ∈ set_of (λ m, (circle_eq m (2*m, 1) = 4*(m-1)^2 + 4)) → m' = m) ∧
  m = 1 ∧
  (∀ L : ℝ × ℝ → ℝ, is_tangent_line L (circle_eq 1) (4, -3) → 
    (∀ x y, L (x, y) = 0 ↔ (3 * x + 4 * y = 0 ∨ x = 4))) :=
by
  sorry

end tangent_lines_to_minimized_circle_l71_71632


namespace half_of_number_l71_71858

theorem half_of_number (N : ℕ) (h : (4 / 15 * 5 / 7 * N) - (4 / 9 * 2 / 5 * N) = 24) : N / 2 = 945 :=
by
  sorry

end half_of_number_l71_71858


namespace part1_part2_l71_71364

variable {a b : ℝ} (h1 : a > 0) (h2 : b > 0) 
variable (h3 : ∀ x, |x + a| + |x - b| ≥ 2)

theorem part1 : (1 / a) + (1 / b) + (1 / (a * b)) ≥ 3 :=
sorry

theorem part2 (t : ℝ) : (sin t)^4 / a + (cos t)^4 / b ≥ 1 / 2 :=
sorry

end part1_part2_l71_71364


namespace presidency_meeting_arrangements_l71_71507

/-- The total number of possible ways to arrange a presidency meeting is 2160 under the given conditions:
1. The club must choose one of the 3 schools at which to host the meeting.
2. The host school sends 3 representatives to the meeting.
3. Each of the other two schools sends 1 representative.
-/
theorem presidency_meeting_arrangements : 
  let total_members := 18
  let num_schools := 3
  ∀ (members_per_school : ℕ) (chosen_host_school : ℕ),
  members_per_school = 6 →
  chosen_host_school ∈ {0, 1, 2} →
  ∃(arrangements : ℕ), arrangements = 3 * (nat.choose 6 3) * 6 * 6 := 
begin
  sorry
end

end presidency_meeting_arrangements_l71_71507


namespace sum_first_15_terms_l71_71689

def sequence (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  a 1 = 1 ∧ 
  (∀ n, a n > 0) ∧ 
  (∀ n, n ≥ 2 → a n = Real.sqrt (S n) + Real.sqrt (S (n - 1)))

def sum_sequence (a : ℕ → ℝ) : ℝ :=
  (Finset.range 16).sum (λ n, if n = 0 then 0 else 1 / (a n * a (n + 1)))

theorem sum_first_15_terms :
  ∃ a S : ℕ → ℝ, sequence a S ∧ sum_sequence a = 15 / 31 :=
sorry

end sum_first_15_terms_l71_71689


namespace error_estimate_alternating_series_l71_71201

theorem error_estimate_alternating_series :
  let S := (1:ℝ) - (1 / 2) + (1 / 3) - (1 / 4) + (-(1 / 5)) 
  let S₄ := (1:ℝ) - (1 / 2) + (1 / 3) - (1 / 4)
  ∃ ΔS : ℝ, ΔS = |-(1 / 5)| ∧ ΔS < 0.2 := by
  sorry

end error_estimate_alternating_series_l71_71201


namespace clock_hands_form_angle_120_degrees_exactly_at_716_l71_71577

-- Define the conditions and relevant functions
def minute_angle (m : ℕ) : ℝ := m * 6
def hour_angle (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def clock_angle (h m : ℕ) : ℝ := let θ := abs (hour_angle h m - minute_angle m) in min θ (360 - θ)

-- Define the time we need to prove
def time_when_120_degree : ℕ × ℕ := (7, 16)

-- Proof statement
theorem clock_hands_form_angle_120_degrees_exactly_at_716 :
  ∃ (h m : ℕ), h = 7 ∧ 7 ≤ h ∧ h < 8 ∧ m = 16 ∧ clock_angle h m = 120 :=
by {
  use (7, 16),
  split, refl,
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, refl,
  sorry
}

end clock_hands_form_angle_120_degrees_exactly_at_716_l71_71577


namespace binom_identity_sum_evaluation_l71_71650

-- Given the definition of binomial coefficient and the Pascal identity
def binom (n k : ℕ) : ℕ := if h : k ≤ n then Nat.choose n k else 0

theorem binom_identity (n p : ℕ) (h : p ≥ 1): 
  binom n p = ∑ k in Finset.range (n - p + 1), binom (k + p - 1) (p - 1) :=
by sorry

-- Evaluating the sum
theorem sum_evaluation : 
  (∑ k in Finset.range 97, k * (k + 1) * (k + 2)) = 23527350 :=
by sorry

end binom_identity_sum_evaluation_l71_71650


namespace point_side_opposite_l71_71903

def equation_lhs (x y : ℝ) : ℝ := 2 * y - 6 * x + 1

theorem point_side_opposite : 
  (equation_lhs 0 0 * equation_lhs 2 1 < 0) := 
by 
   sorry

end point_side_opposite_l71_71903


namespace problem_l71_71628

-- Definitions from the problem conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(-x)

def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f(x + p) = f(x)

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 2 then Real.log (x + 1) / Real.log 2 else 0 -- A dummy definition; we will only use it for 0 ≤ x < 2

-- The proof problem
theorem problem 
  (h_even : is_even_function f)
  (h_periodic : is_periodic_function f 2)
  (h_interval : ∀ x, 0 ≤ x ∧ x < 2 → f(x) = Real.log (x + 1) / Real.log 2)
  : f(-2008) + f(2009) = 1 := by
  sorry

end problem_l71_71628


namespace tim_shells_left_l71_71814

theorem tim_shells_left
  (initial_clam_shells : ℕ)
  (initial_conch_shells : ℕ)
  (initial_oyster_shells : ℕ)
  (starfish : ℕ)
  (clam_percentage_given : ℚ)
  (conch_percentage_given : ℚ)
  (oyster_fraction_given : ℚ) :
  initial_clam_shells = 325 →
  initial_conch_shells = 210 →
  initial_oyster_shells = 144 →
  starfish = 110 →
  clam_percentage_given = 0.40 →
  conch_percentage_given = 0.25 →
  oyster_fraction_given = 1/3 →
  let clam_shells_given := clam_percentage_given * initial_clam_shells,
      clam_shells_left := initial_clam_shells - clam_shells_given.to_nat,
      conch_shells_given := conch_percentage_given * initial_conch_shells,
      conch_shells_left := initial_conch_shells - conch_shells_given.to_nat,
      oyster_shells_given := oyster_fraction_given * initial_oyster_shells,
      oyster_shells_left := initial_oyster_shells - oyster_shells_given.to_nat,
      total_shells_left := clam_shells_left + conch_shells_left + oyster_shells_left + starfish
  in total_shells_left = 558 :=
by
  intros
  sorry

end tim_shells_left_l71_71814


namespace red_light_after_two_red_light_expectation_and_variance_l71_71158

noncomputable def prob_red_light_after_two : ℚ := (2/3) * (2/3) * (1/3)
theorem red_light_after_two :
  prob_red_light_after_two = 4/27 :=
by
  -- We have defined the probability calculation directly
  sorry

noncomputable def expected_red_lights (n : ℕ) (p : ℚ) : ℚ := n * p
noncomputable def variance_red_lights (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem red_light_expectation_and_variance :
  expected_red_lights 6 (1/3) = 2 ∧ variance_red_lights 6 (1/3) = 4/3 :=
by
  -- We have defined expectation and variance calculations directly
  sorry

end red_light_after_two_red_light_expectation_and_variance_l71_71158


namespace solve_inequality_l71_71445

theorem solve_inequality (x : ℝ) :
  (x * (x + 2) / (x - 3) < 0) ↔ (x < -2 ∨ (0 < x ∧ x < 3)) :=
sorry

end solve_inequality_l71_71445


namespace sum_of_coefficients_eq_10_l71_71714

theorem sum_of_coefficients_eq_10 
  (s : ℕ → ℝ) 
  (a b c : ℝ) 
  (h0 : s 0 = 3) 
  (h1 : s 1 = 5) 
  (h2 : s 2 = 9)
  (h : ∀ k ≥ 2, s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)) : 
  a + b + c = 10 :=
sorry

end sum_of_coefficients_eq_10_l71_71714


namespace percentage_absent_students_l71_71153

def total_students : ℕ := 150
def boys : ℕ := 90
def girls : ℕ := 60
def absent_boys_fraction : ℚ := 1 / 5
def absent_girls_fraction : ℚ := 1 / 4

theorem percentage_absent_students :
  (absent_boys_fraction * boys + absent_girls_fraction * girls).natAbs * 100 / total_students = 22 :=
by
  sorry

end percentage_absent_students_l71_71153


namespace cos_theta_value_l71_71883

-- Define the vectors
def vector1 : ℝ × ℝ := (4, 5)
def vector2 : ℝ × ℝ := (2, 7)

-- Calculate the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Calculate the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Calculate cos theta
def cos_theta : ℝ :=
  dot_product vector1 vector2 / (magnitude vector1 * magnitude vector2)

-- The statement to be proved
theorem cos_theta_value :
  cos_theta = 43 / Real.sqrt 2173 :=
by
  -- Proof goes here
  sorry

end cos_theta_value_l71_71883


namespace solution_set_of_inequality_l71_71446

theorem solution_set_of_inequality (x : ℝ) : 
  (1 / x ≤ 2 ↔ x ∈ (-∞ : ℝ, 0) ∪ set.Ici (1 / 2)) :=
begin
  sorry
end

end solution_set_of_inequality_l71_71446


namespace sum_of_roots_eq_14_l71_71089

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l71_71089


namespace container_volume_ratio_l71_71538

theorem container_volume_ratio
  (A B : ℝ)
  (h : (5 / 6) * A = (3 / 4) * B) :
  (A / B = 9 / 10) :=
sorry

end container_volume_ratio_l71_71538


namespace paul_lost_crayons_l71_71731

theorem paul_lost_crayons :
  ∀ (initial_crayons given_crayons left_crayons lost_crayons : ℕ),
    initial_crayons = 1453 →
    given_crayons = 563 →
    left_crayons = 332 →
    lost_crayons = (initial_crayons - given_crayons) - left_crayons →
    lost_crayons = 558 :=
by
  intros initial_crayons given_crayons left_crayons lost_crayons
  intros h_initial h_given h_left h_lost
  sorry

end paul_lost_crayons_l71_71731


namespace sum_of_roots_of_equation_l71_71085

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l71_71085


namespace prod_inequality_l71_71719

open Real

theorem prod_inequality
  {n : ℕ} (x : fin n → ℝ) (a s : ℝ)
  (hx : ∀ i, x i > 0)
  (ha_pos : a > 0)
  (hs : ∑ i, x i = s)
  (hsa : s ≤ a) :
  (∏ i, (a + x i) / (a - x i)) ≥ ((n * a + s) / (n * a - s)) ^ n :=
    sorry

end prod_inequality_l71_71719


namespace sum_of_roots_eq_l71_71060

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l71_71060


namespace vasya_misha_expected_coincidences_l71_71486

noncomputable def expected_coincidences (n : ℕ) (pA pB : ℝ) : ℝ :=
  n * ((pA * pB) + ((1 - pA) * (1 - pB)))

theorem vasya_misha_expected_coincidences :
  expected_coincidences 20 (6 / 20) (8 / 20) = 10.8 :=
by
  -- Test definition and expected output
  let n := 20
  let pA := 6 / 20
  let pB := 8 / 20
  have h : expected_coincidences n pA pB =  20 * ((pA * pB) + ((1 - pA) * (1 - pB))) := rfl
  rw h
  sorry

end vasya_misha_expected_coincidences_l71_71486


namespace pure_alcohol_addition_l71_71859

theorem pure_alcohol_addition (x : ℝ) (h1 : 3 / 10 * 10 = 3)
    (h2 : 60 / 100 * (10 + x) = (3 + x) ) : x = 7.5 :=
sorry

end pure_alcohol_addition_l71_71859


namespace find_integer_pairs_l71_71210

theorem find_integer_pairs (m n : ℤ) (k : ℕ) (h_n_even : n % 2 = 0) (h_m : m = k * (n + 1) - 2)
  (h_coprime : Int.gcd m (n + 1) = 1) :
  (∀ k : ℕ, (m, n + 1) = 1 ∧ ∑ i in finset.range (n + 1), (m^(i+1) / (i+1) * (nat.choose n i)) ∈ ℤ) := 
by 
  sorry

end find_integer_pairs_l71_71210


namespace exists_graph_with_properties_l71_71742

theorem exists_graph_with_properties (n : ℕ) (N : ℕ)
  (h_n : n > 2) (h_N : N > 0) :
  ∃ (G : SimpleGraph ℕ), G.chromaticNumber = n ∧ G.vertexCount ≥ N ∧
  ∀ v, (G.removeVertex v).chromaticNumber = n - 1 :=
sorry

end exists_graph_with_properties_l71_71742


namespace meet_again_at_X_l71_71336

noncomputable def meet_at_X (P_distance : ℕ) (Q_distance : ℕ) (P_to_Q : ℕ) (P_speed : ℕ) (Q_speed : ℕ) (circumference : ℕ) : ℕ :=
  let next_meet_time := 96
  in next_meet_time

theorem meet_again_at_X (P_distance : ℕ) (Q_distance : ℕ) (P_to_Q : ℕ) (P_speed : ℕ) (Q_speed : ℕ) (circumference : ℕ)
  (h1 : P_distance = 8) (h2 : Q_distance = 16) (h3 : P_to_Q = 16) (h4 : P_speed = 3) (h5 : Q_speed = 3.5) (h6 : circumference = 40) :
  meet_at_X P_distance Q_distance P_to_Q P_speed Q_speed circumference = 96 := by
  sorry

end meet_again_at_X_l71_71336


namespace geometric_seq_max_product_l71_71240

theorem geometric_seq_max_product (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 2 + a 4 = 5) 
  (geo_seq : ∀ m, a (m + 1) = a m * (a 2 / a 1)) :
  ∏ i in finset.range (n + 1), a i ≤ 64 := 
sorry

end geometric_seq_max_product_l71_71240


namespace angle_between_lateral_face_and_base_l71_71782

theorem angle_between_lateral_face_and_base (S A B C: Point) 
  (h_eq_as_sb_sc: (distance S A = distance S B) ∧ (distance S B = distance S C))
  (h_right_angles: (angle A S C = 90) ∧ (angle C S B = 90) ∧ (angle A S B = 90))
  (h_base_equilateral: is_equilateral_triangle A B C) :
  angle_between_face_and_plane (triangular_face S A B) (plane_through ABC) = arccos (sqrt 3 / 3) := 
sorry

end angle_between_lateral_face_and_base_l71_71782


namespace no_real_roots_of_quad_eq_l71_71304

theorem no_real_roots_of_quad_eq (k : ℝ) :
  ∀ x : ℝ, ¬ (x^2 - 2*x - k = 0) ↔ k < -1 :=
by sorry

end no_real_roots_of_quad_eq_l71_71304


namespace max_abs_value_l71_71609

-- Define complex number and absolute value properties
noncomputable def max_value_condition (z : ℂ) : Prop :=
  complex.abs z = 1

theorem max_abs_value (z : ℂ) (h : max_value_condition z) : complex.abs (z + (3 - 4 * complex.I)) = 6 :=
sorry

end max_abs_value_l71_71609


namespace B4_eq_B1_l71_71986

-- Define the main structures: Points and Circles
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Condition 1: Three circles passing through a common point O
variables {O : Point} {O1 O2 O3 : Circle}
  (hO1: O ∈ O1) (hO2: O ∈ O2) (hO3: O ∈ O3)

-- Condition 2: Other intersections of the circles
variables {A1 A2 A3 : Point}
  (hA1 : A1 ≠ O ∧ A1 ∈ O1 ∧ A1 ∈ O2)
  (hA2 : A2 ≠ O ∧ A2 ∈ O2 ∧ A2 ∈ O3)
  (hA3 : A3 ≠ O ∧ A3 ∈ O3 ∧ A3 ∈ O1)

-- Condition 3: Arbitrary point B1 on O1
variables {B1 : Point}
  (hB1 : B1 ∈ O1 ∧ B1 ≠ A1)

-- Condition 4: Line through B1 and A1 intersects O2 again at B2
variables {B2 : Point}
  (hB2 : B2 ∈ O2 ∧ B2 ≠ A2 ∧ lies_on_line B1 A1 B2)

-- Condition 5: Line through B2 and A2 intersects O3 again at B3
variables {B3 : Point}
  (hB3 : B3 ∈ O3 ∧ B3 ≠ A3 ∧ lies_on_line B2 A2 B3)

-- Condition 6: Line through B3 and A3 intersects O1 again at B4
variables {B4 : Point}
  (hB4 : B4 ∈ O1 ∧ lies_on_line B3 A3 B4)

-- Prove that B4 coincides with B1
theorem B4_eq_B1 :
  B4 = B1 :=
sorry

end B4_eq_B1_l71_71986


namespace triangle_ABC_area_l71_71275

-- Define the vertices of the triangle
def A := (-4, 0)
def B := (24, 0)
def C := (0, 2)

-- Function to calculate the determinant, used for the area calculation
def det (x1 y1 x2 y2 x3 y3 : ℝ) :=
  x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)

-- Area calculation for triangle given vertices using determinant method
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * |det x1 y1 x2 y2 x3 y3|

-- The goal is to prove that the area of triangle ABC is 14
theorem triangle_ABC_area :
  triangle_area (-4) 0 24 0 0 2 = 14 := sorry

end triangle_ABC_area_l71_71275


namespace integral_eval_l71_71582

theorem integral_eval : 3 * ∫ x in -1..1, (sin x + x^2) = 2 := 
by sorry

end integral_eval_l71_71582


namespace sum_of_roots_eq_14_l71_71092

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l71_71092


namespace count_multiples_of_7_with_units_digit_7_l71_71290

theorem count_multiples_of_7_with_units_digit_7 :
  (finset.filter (λ n, (n % 10 = 7)) (finset.range 200).filter (λ n, n % 7 = 0)).card = 3 :=
by
  sorry

end count_multiples_of_7_with_units_digit_7_l71_71290


namespace lambda_range_geometric_sequence_l71_71339

theorem lambda_range_geometric_sequence (a_n : ℕ → ℝ) (λ : ℝ) (n : ℕ) (h : 0 < n) : 
  (a_n 4 = 8) ∧ (∀ n, a_n (n + 1) = 2 * a_n n) →
  (∀ n, 1 / (a_n n) ^ 2 < 4 / 3) ∧ 
  (∀ n, (λ ^ 2) / 3 - λ ≥ ∑ i in finset.range n, (1 / (a_n i) ^ 2)) →
  λ ≤ -1 ∨ λ ≥ 4 :=
by
  sorry

end lambda_range_geometric_sequence_l71_71339


namespace sum_of_roots_of_equation_l71_71081

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l71_71081


namespace chessboard_tiling_exists_l71_71450

def Chessboard := Fin 8 × Fin 8

def isDomino (d : set Chessboard) : Prop :=
  ∃ (a b : Chessboard), d = {a, b} ∧ (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2) ∨ a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

def tiling (T : set (set Chessboard)) : Prop :=
  (∀ t ∈ T, isDomino t) ∧ 
  (⋃₀ T = set.univ : set Chessboard) ∧ 
  (∀ t₁ ∈ T, ∀ t₂ ∈ T, t₁ ≠ t₂ → t₁ ∩ t₂ = ∅)

theorem chessboard_tiling_exists : ∃ T : set (set Chessboard), tiling T := 
by 
  sorry

end chessboard_tiling_exists_l71_71450


namespace find_angle_BAO_l71_71681

-- Given conditions
variables {O A B C D E : Type}
variables [add_comm_group O] [module ℝ O]
variables [add_comm_group A] [module ℝ A]
variables [add_comm_group B] [module ℝ B]
variables [add_comm_group C] [module ℝ C]
variables [add_comm_group D] [module ℝ D]
variables [add_comm_group E] [module ℝ E]

-- Definitions and assumptions for the problem
variables (CD := diameter_semicircle O)
variables (A_exists_on_ext := A_on_extension_DC_past_C A C D)
variables (E_on_semicircle := E_on_semicircle O E)
variables (B_intersect_AE_semicircle := B_intersection_distinct_from_E AE_s O E)
variables (length_AB_eq_OD := length_AB_eq_OD AB OD)
variables (angle_EOD := angle_EOD_measures O E D (60 : ℝ))

-- Goal
theorem find_angle_BAO
  (CD : diameter_semicircle O)
  (A_exists_on_ext : A_on_extension_DC_past_C A C D)
  (E_on_semicircle : E_on_semicircle O E)
  (B_intersect_AE_semicircle : B_intersection_distinct_from_E AE_s O E)
  (length_AB_eq_OD : length_AB_eq_OD AB OD)
  (angle_EOD : angle_EOD_measures O E D (60 : ℝ)) :
  measure_angle_BAO O A B = 15 :=
sorry

end find_angle_BAO_l71_71681


namespace probability_top_card_five_or_joker_l71_71137

theorem probability_top_card_five_or_joker (total_cards jokers fives : ℕ) (total_cards_eq : total_cards = 54) 
(jokers_eq : jokers = 2) (fives_eq : fives = 4) :
  ((fives + jokers) : ℚ) / total_cards = 1 / 9 :=
by {
  rw [total_cards_eq, jokers_eq, fives_eq],
  norm_num,
  sorry
}

end probability_top_card_five_or_joker_l71_71137


namespace log_exp_comparison_l71_71621

noncomputable def a : ℝ := Real.logb 0.4 0.3
noncomputable def b : ℝ := 0.4^0.4
noncomputable def c : ℝ := 0.4^0.3

theorem log_exp_comparison : b < c ∧ c < a :=
by {
  have h1 : a = Real.logb 0.4 0.3,
  have h2 : b = 0.4^0.4,
  have h3 : c = 0.4^0.3,
  have h4 : Real.logb 0.4 0.3 > 1 := sorry,
  have h5 : 0.4^0.4 < 0.4^0.3 := sorry,
  have h6 : 0.4^0.3 < 1 := sorry,
  exact ⟨h5, lt_trans h6 h4⟩,
}

end log_exp_comparison_l71_71621


namespace right_angled_if_scalene_and_median_through_circumcenter_l71_71512

-- Definitions from conditions
variable {α : Type} [LinearOrder α] [AddGroup α] [Sub α] [Mul α] [Div α]
variable {A B C O : α}
variable {R : α}
variable (midpoint : α -> α -> α) (is_median : α -> α -> α -> Prop)
variable (is_circumcenter : α -> α -> α -> α -> Prop) (is_right_angled : α -> α -> α -> Prop)
variable (scalene : α -> α -> α -> Prop)

-- Given the conditions
def median_through_circumcenter (A B C O : α) (midpoint : α -> α -> α) (is_median : α -> α -> α -> Prop) :=
  is_median B (midpoint A C) O

def scalene_triangle (A B C : α) := scalene A B C

-- Claim statement
theorem right_angled_if_scalene_and_median_through_circumcenter
  (A B C O : α) (midpoint : α -> α -> α) (is_median : α -> α -> α -> Prop) (is_circumcenter : α -> α -> α -> α -> Prop) 
  (is_right_angled : α -> α -> α -> Prop) (scalene : α -> α -> α -> Prop) :
  (scalene_triangle A B C) → (median_through_circumcenter A B C O midpoint is_median) → (is_circumcenter A B C O) → is_right_angled A B C :=
by
  sorry
 
end right_angled_if_scalene_and_median_through_circumcenter_l71_71512


namespace problem_l71_71547

noncomputable def octagon_chord_length : ℚ := sorry

theorem problem : ∃ (p q : ℕ), p + q = 5 ∧ (nat.gcd p q = 1) ∧ (octagon_chord_length = p / q) :=
begin
  sorry
end

end problem_l71_71547


namespace ratio_female_to_male_l71_71174

-- Definitions of the conditions
variables (f m : ℕ)
axiom avg_age_female : ℕ := 45
axiom avg_age_male : ℕ := 20
axiom avg_age_total : ℕ := 28

-- The ratio to prove
theorem ratio_female_to_male (h : (45 * f + 20 * m) / (f + m) = 28) : f / m = 8 / 17 := by
  sorry

end ratio_female_to_male_l71_71174


namespace eval_expression_l71_71948

noncomputable def i : ℂ := complex.I

theorem eval_expression : (i^14560 + i^14561 + i^14562 + i^14563) = 0 :=
by sorry

end eval_expression_l71_71948


namespace max_value_y_l71_71605

def f (x : ℝ) : ℝ := 2 + real.log x / real.log 3

def y (x : ℝ) : ℝ := (f x) ^ 2 + f (x^2)

theorem max_value_y : (1 ≤ x ∧ x ≤ 9) → ∃ b, ∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → y x ≤ b ∧ b = 13 :=
begin
  sorry  -- Proof is omitted.
end

end max_value_y_l71_71605


namespace zero_intervals_l71_71599

theorem zero_intervals (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  ∃ x₁ x₂ : ℝ, (a < x₁ ∧ x₁ < b) ∧ (b < x₂ ∧ x₂ < c) ∧ 
  ((λ x, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)) x₁ = 0) ∧
  ((λ x, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)) x₂ = 0) :=
by
  sorry

end zero_intervals_l71_71599


namespace ratio_of_amounts_l71_71353

noncomputable theory

variable (R P : ℝ)
variable (h1 : R + 0.2 * P = 15)
variable (h2 : 9.6 < 15)  -- This ensures she spent 9.6 rupees.

theorem ratio_of_amounts : (15 - 9.6) / 15 = 9 / 25 :=
by
  sorry

end ratio_of_amounts_l71_71353


namespace rhombus_side_length_l71_71889

theorem rhombus_side_length (R S : ℝ) (hR : R > 0) (hS : S > 0) :
  let AB := (4 * R^3) / S in
  ∃ AB : ℝ, AB = (4 * R^3) / S :=
by
  sorry

end rhombus_side_length_l71_71889


namespace stephan_more_than_clara_l71_71759

-- Let M be the amount of chocolates Mary has
variables (M : ℝ) (S : ℝ) (C : ℝ)

-- Condition: Stephan has 60% more chocolates than Mary
def stephan_chocolates := (S = 1.60 * M)

-- Condition: Clara has 40% more chocolates than Mary
def clara_chocolates := (C = 1.40 * M)

-- Definition: The percentage difference between Stephan's and Clara's chocolates
def percentage_difference := ((S - C) / C) * 100 = 14.29

-- The theorem stating the problem to prove
theorem stephan_more_than_clara :
  stephan_chocolates M S →
  clara_chocolates M C →
  percentage_difference M S C :=
by
  sorry

end stephan_more_than_clara_l71_71759


namespace disk_rearrangement_moves_l71_71173

/-- 
Given an initial arrangement of disks B W B W B W B W, prove that it takes exactly 5 moves to rearrange them to B B B B W W W W under the constraint of moving and swapping two adjacent disks.
-/
theorem disk_rearrangement_moves 
  (initial_arrangement : List (Bool)) 
  (goal_arrangement : List (Bool)) 
  (condition : ∀ (a b : Nat), initial_arrangement[n] == initial_arrangement[a] ∧ initial_arrangement[m] == initial_arrangement[b] -> initial_arrangement' == initial_arrangement with (swap))
  : initial_arrangement = [ff, tt, ff, tt, ff, tt, ff, tt] → 
    goal_arrangement = [ff, ff, ff, ff, tt, tt, tt, tt] → 
    (∃ (moves : Nat), moves = 5) :=
begin
  sorry
end

end disk_rearrangement_moves_l71_71173


namespace no_real_roots_of_quad_eq_l71_71303

theorem no_real_roots_of_quad_eq (k : ℝ) :
  ∀ x : ℝ, ¬ (x^2 - 2*x - k = 0) ↔ k < -1 :=
by sorry

end no_real_roots_of_quad_eq_l71_71303


namespace diff_yellow_red_tiles_l71_71350

theorem diff_yellow_red_tiles 
  (initial_red_tiles : ℕ)
  (initial_yellow_tiles : ℕ)
  (additional_yellow_per_side : ℕ)
  (hexagon_sides : ℕ)
  (initial_red_tiles = 15)
  (initial_yellow_tiles = 9)
  (additional_yellow_per_side = 4)
  (hexagon_sides = 6) :
  (initial_yellow_tiles + additional_yellow_per_side * hexagon_sides) - initial_red_tiles = 18 :=
by 
  sorry

end diff_yellow_red_tiles_l71_71350


namespace maximize_segment_outside_inscribed_circle_l71_71231

def segment_area (R α : ℝ) : ℝ :=
  Real.pi - (α - Real.sin α * Real.cos α)

def inscribed_circle_area (R α : ℝ) : ℝ :=
  Real.pi * (1 + Real.cos α)^2 / 4

def area_function (R α : ℝ) : ℝ :=
  segment_area R α - inscribed_circle_area R α

theorem maximize_segment_outside_inscribed_circle (R α : ℝ) (h : R = 1) :
  α ∈ Icc 0 Real.pi →
  ( ∀ beta ∈ Icc 0 Real.pi, area_function R α ≥ area_function R beta ) →
  2 * Real.sin α = 2 * 16 * Real.pi / (16 + (Real.pi)^2) :=
by
  sorry

end maximize_segment_outside_inscribed_circle_l71_71231


namespace minimum_value_l71_71242

noncomputable def min_max_prod (n : ℕ) (h : n ≥ 3) : ℝ :=
  Inf (set_of (λ (A : set ℂ), (∀ z ∈ A, ∃ i : ℕ, z = complex.exp (2 * real.pi * complex.I * i / n) ∧ complex.abs z = 1)
    ∧ (|A| = n) ∧ (∀ u : ℂ, complex.abs u = 1 →
          (∏ z in A, complex.abs (u - z)) ≤ 2)))

theorem minimum_value (n : ℕ) (h : n ≥ 3) :
  min_max_prod n h = 2 :=
sorry

end minimum_value_l71_71242


namespace find_natural_number_l71_71953

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_natural_number (n : ℕ) : sum_of_digits (2 ^ n) = 5 ↔ n = 5 := by
  sorry

end find_natural_number_l71_71953


namespace last_digit_of_a_power_b_l71_71117

-- Define the constants from the problem
def a : ℕ := 954950230952380948328708
def b : ℕ := 470128749397540235934750230

-- Define a helper function to calculate the last digit of a natural number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the main statement to be proven
theorem last_digit_of_a_power_b : last_digit ((last_digit a) ^ (b % 4)) = 4 :=
by
  -- Here go the proof steps if we were to provide them
  sorry

end last_digit_of_a_power_b_l71_71117


namespace total_number_of_legs_is_40_l71_71186

-- Define the number of octopuses Carson saw.
def number_of_octopuses := 5

-- Define the number of legs per octopus.
def legs_per_octopus := 8

-- Define the total number of octopus legs Carson saw.
def total_octopus_legs : Nat := number_of_octopuses * legs_per_octopus

-- Prove that the total number of octopus legs Carson saw is 40.
theorem total_number_of_legs_is_40 : total_octopus_legs = 40 := by
  sorry

end total_number_of_legs_is_40_l71_71186


namespace item_at_16_and_29_l71_71341

def sequence : Nat → Char
| n := let cycle := "••⭘⭘" in cycle.getOp (n % 4)

theorem item_at_16_and_29 :
  sequence 15 = '⭘' ∧ sequence 28 = '•' :=
by {
  -- The 16th item (0-indexed as 15) in the sequence is '⭘' and the 29th item (0-indexed as 28) in the sequence is '•'.
  sorry
}

end item_at_16_and_29_l71_71341


namespace shopkeeper_discount_problem_l71_71154

theorem shopkeeper_discount_problem (CP SP_with_discount SP_without_discount Discount : ℝ)
  (h1 : SP_with_discount = CP + 0.273 * CP)
  (h2 : SP_without_discount = CP + 0.34 * CP) :
  Discount = SP_without_discount - SP_with_discount →
  (Discount / SP_without_discount) * 100 = 5 := 
sorry

end shopkeeper_discount_problem_l71_71154


namespace diameter_of_circular_field_l71_71213

theorem diameter_of_circular_field :
  ∀ (π : ℝ) (cost_per_meter total_cost circumference diameter : ℝ),
    π = Real.pi → 
    cost_per_meter = 1.50 → 
    total_cost = 94.24777960769379 → 
    circumference = total_cost / cost_per_meter →
    circumference = π * diameter →
    diameter = 20 := 
by
  intros π cost_per_meter total_cost circumference diameter hπ hcp ht cutoff_circ hcirc
  sorry

end diameter_of_circular_field_l71_71213


namespace steve_bank_account_amount_after_two_years_l71_71760

-- Conditions
def initial_deposit : ℝ := 100
def yearly_deposit : ℝ := 10
def interest_rate : ℝ := 0.10

-- Problem statement
theorem steve_bank_account_amount_after_two_years :
  let first_year_amount := initial_deposit * (1 + interest_rate) + yearly_deposit in
  let second_year_amount := first_year_amount * (1 + interest_rate) in
  second_year_amount = 132 := 
by
  sorry

end steve_bank_account_amount_after_two_years_l71_71760


namespace a_2_value_l71_71238

-- Define the conditions
variable {a : ℕ → ℤ} (d : ℤ := 2)
hypothesis h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d
hypothesis h_geom_seq : (a 0 + 4)^2 = a 0 * (a 0 + 6)

-- Prove that a_2 = -6
theorem a_2_value : a 2 = -6 :=
by
  -- Using the hypothesis and properties, we can prove the required value.
  sorry

end a_2_value_l71_71238


namespace composite_probability_l71_71475

noncomputable def probability_composite_or_nonzero_imaginary_part : ℚ :=
  let n := 100
  let composites := (Finset.range n).filter (λ k, ¬ Nat.Prime k ∧ k ≠ 1)
  let count_composites := composites.card
  count_composites / n

theorem composite_probability : 
  probability_composite_or_nonzero_imaginary_part = 74 / 100 := 
by 
  let composite_numbers := 74
  let total_numbers := 100
  have h : probability_composite_or_nonzero_imaginary_part = composite_numbers / total_numbers, by sorry
  exact h

end composite_probability_l71_71475


namespace relationship_and_range_dimensions_when_area_18_impossibility_of_area_21_l71_71150

variable (x y : ℝ)
variable (h1 : 2 * (x + y) = 18)
variable (h2 : x * y = 18)
variable (h3 : x > 0) (h4 : y > 0) (h5 : x > y)
variable (h6 : x * y = 21)

theorem relationship_and_range : (y = 9 - x ∧ 0 < x ∧ x < 9) :=
by sorry

theorem dimensions_when_area_18 :
  (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 6) :=
by sorry

theorem impossibility_of_area_21 :
  ¬(∃ x y, x * y = 21 ∧ 2 * (x + y) = 18 ∧ x > y) :=
by sorry

end relationship_and_range_dimensions_when_area_18_impossibility_of_area_21_l71_71150


namespace necessary_and_sufficient_condition_for_equal_shaded_areas_l71_71684

-- Define the conditions
def theta : ℝ := sorry -- Given θ is within the range
def r : ℝ := sorry -- Radius of the circle r

-- Additional necessary axioms/definitions might be needed, we use sorry for now.
axiom theta_range : (0 < theta) ∧ (theta < real.pi / 2)
axiom is_tangent : ∀ (A B C : Type), true -- Placeholder for tangent definition 
axiom is_center : ∀ (C : Type), true -- Placeholder for center definition
axiom straight_line_segments : ∀ (B C D E : Type), true -- Placeholder for straight line segments

-- Define the equality of areas condition
def equal_shaded_areas_condition (θ r: ℝ) : Prop :=
  1/2 * r^2 * tan θ - 1/2 * r^2 * θ = 1/2 * r^2 * θ ∧ tan θ = 2 * θ

-- Required theorem statement
theorem necessary_and_sufficient_condition_for_equal_shaded_areas
  (θ : ℝ) (r : ℝ) (theta_range : (0 < θ) ∧ (θ < real.pi / 2)) :
  equal_shaded_areas_condition θ r :=
sorry

end necessary_and_sufficient_condition_for_equal_shaded_areas_l71_71684


namespace option_D_is_correct_system_l71_71467

def is_linear (eq : LinearEquation) : Prop :=
  eq.degree = 1

def set_of_equations_D : LinearSystem := {
  eq1 := 4 * x - y = 1
  eq2 := y = 2 * x + 3
}

theorem option_D_is_correct_system : is_linear (set_of_equations_D.eq1) ∧ is_linear (set_of_equations_D.eq2) := by
  sorry

end option_D_is_correct_system_l71_71467


namespace hexagon_perimeter_proof_l71_71685

noncomputable def hexagon_perimeter (s : ℝ) : ℝ :=
  6 * s

theorem hexagon_perimeter_proof :
  ∀ (hexagon : { h : ℕ → ℝ × ℝ // ∀ i, i < 6 → dist (h i) (h ((i+1) % 6)) = dist (h i) (h 0) ∧
                                      ∠ (h i, h ((i+2) % 6), h 0) = 30 })
    (area : ℝ),
  (area = 6 * real.sqrt 3) →
  (∃ s : ℝ, area = 0.5 * s * s * real.sin (real.pi / 6) * 3 + 0.5 * (s^2 - (0.5 * s^2 - 0.5 * s^2 * real.sqrt 3 / 2) / 2) * real.sqrt 3) →
  hexagon_perimeter (real.sqrt 12) = 12 * real.sqrt 3 :=
begin
  sorry
end

end hexagon_perimeter_proof_l71_71685


namespace sum_n_binom_30_15_eq_31_16_l71_71831

open Nat

-- Given n = 30 and k = 15, we are given the components to test Pascal's identity
def PascalIdentity (n k : Nat) : Prop :=
  Nat.choose (n-1) (k-1) + Nat.choose (n-1) k = Nat.choose n k

theorem sum_n_binom_30_15_eq_31_16 : 
  (∑ n in { n : ℕ | Nat.choose 30 15 + Nat.choose 30 n = Nat.choose 31 16 }, n) = 30 := 
sorry

end sum_n_binom_30_15_eq_31_16_l71_71831


namespace find_numbers_l71_71722

theorem find_numbers (a b c d e : ℕ) 
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (sums : multiset ℕ) 
  (h_sums : sums = {21, 26, 35, 40, 49, 51, 54, 60, 65, 79}) :
  a = 6 ∧ b = 15 ∧ c = 20 ∧ d = 34 ∧ e = 45 :=
begin
  sorry
end

end find_numbers_l71_71722


namespace area_of_triangle_AEF_l71_71406

-- Definitions and conditions
def is_parallelogram (A B C D : Type) [Add AB AC AD AND TR ADD] := sorry -- Condition indicating ABCD is a parallelogram

def division_ratio_2_3 (A B E : Type) [Add AE AB], AB := sorry -- E divides AB in ratio 2:3
def division_ratio_3_2 (C D F : Type) [Add CF CD], CD := sorry -- F divides CD in ratio 3:2
def mid_point (G : Type) (B C: Add), Add BC BG GC := sorry -- G divides BC into equal halves


-- Main proof statement
theorem area_of_triangle_AEF (A B C D E F G: Type) (area_ABCD: ℝ) (H_ABCD: is_parallelogram A B C D) (H_area: area_ABCD = 50) (H_DE: division_ratio_2_3 A B E) (H_F: division_ratio_3_2 C D F) (H_G_mid: mid_point G B C) : 
  area_of_triangle A E F = 12 :=
sorry

end area_of_triangle_AEF_l71_71406


namespace all_students_meet_second_time_l71_71758

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem all_students_meet_second_time :
  ∀ (period1 period2 period3 period4 : ℕ),
  period1 = 4 ∧ period2 = 5 ∧ period3 = 6 ∧ period4 = 9 →
  (let lcm1 := lcm period1 period2 in
   let lcm2 := lcm lcm1 period3 in
   let total_lcm := lcm lcm2 period4 in
   total_lcm = 180 ∧ 2 * total_lcm = 360) := 
by
  intros period1 period2 period3 period4 h
  unfold lcm
  sorry

end all_students_meet_second_time_l71_71758


namespace max_ab_min_one_over_a_plus_one_over_b_range_of_m_l71_71600

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : ab ≤ 1 := by
  sorry

theorem min_one_over_a_plus_one_over_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : (1/a) + (1/b) ≥ 2 := by
  sorry

theorem range_of_m (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) (h4 : ∀ x : ℝ, | x + m | - | x + 1 | ≤ (1/a) + (1/b)) : -1 ≤ m ∧ m ≤ 3 := by
  sorry

end max_ab_min_one_over_a_plus_one_over_b_range_of_m_l71_71600


namespace negation_of_at_most_one_obtuse_is_at_least_two_obtuse_l71_71786

-- Definition of a triangle with a property on the angles.
def triangle (a b c : ℝ) : Prop := a + b + c = 180 ∧ 0 < a ∧ 0 < b ∧ 0 < c

-- Definition of an obtuse angle.
def obtuse (x : ℝ) : Prop := x > 90

-- Proposition: In a triangle, at most one angle is obtuse.
def at_most_one_obtuse (a b c : ℝ) : Prop := 
  triangle a b c ∧ (obtuse a → ¬ obtuse b ∧ ¬ obtuse c) ∧ (obtuse b → ¬ obtuse a ∧ ¬ obtuse c) ∧ (obtuse c → ¬ obtuse a ∧ ¬ obtuse b)

-- Negation: In a triangle, there are at least two obtuse angles.
def at_least_two_obtuse (a b c : ℝ) : Prop := 
  triangle a b c ∧ (obtuse a ∧ obtuse b) ∨ (obtuse a ∧ obtuse c) ∨ (obtuse b ∧ obtuse c)

-- Prove the negation equivalence
theorem negation_of_at_most_one_obtuse_is_at_least_two_obtuse (a b c : ℝ) :
  (¬ at_most_one_obtuse a b c) ↔ at_least_two_obtuse a b c :=
sorry

end negation_of_at_most_one_obtuse_is_at_least_two_obtuse_l71_71786


namespace sum_of_roots_eq_l71_71055

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l71_71055


namespace number_of_true_propositions_eq_2_l71_71191

theorem number_of_true_propositions_eq_2 :
  (¬(∀ (a b : ℝ), a < 0 → b > 0 → a + b < 0)) ∧
  (∀ (α β : ℝ), α = 90 → β = 90 → α = β) ∧
  (∀ (α β : ℝ), α + β = 90 → (∀ (γ : ℝ), γ + α = 90 → β = γ)) ∧
  (¬(∀ (ℓ m n : ℕ), (ℓ ≠ m ∧ ℓ ≠ n ∧ m ≠ n) → (∀ (α β : ℝ), α = β))) →
  2 = 2 :=
by
  sorry

end number_of_true_propositions_eq_2_l71_71191


namespace shayan_produces_nice_polynomials_l71_71820

theorem shayan_produces_nice_polynomials (d : ℕ) 
  (r : Fin d → ℝ) 
  (h_distinct : Function.Injective r) 
  (h_neq_pm_one : ∀ i, r i ≠ 1 ∧ r i ≠ -1)
  (N : ℕ) (hN : N > 1) :
  ∃ n : ℕ, ∀ m ≥ n, let q := λ (k : Fin d), (r k) ^ (N ^ m)
  in let P := Polynomial.monic (Polynomial.of_roots (Finset.image q Finset.univ))
  in (2021 * (Polynomial.eval 0 P + Polynomial.eval 1 P + ... + Polynomial.eval d P)) / 2022 < max (Polynomial.eval 0 P) ... (Polynomial.eval d P) :=
begin
  sorry
end

end shayan_produces_nice_polynomials_l71_71820


namespace sum_of_abs_b_i_l71_71595

def P (x : ℝ) : ℝ := 1 - (1 / 2) * x + (1 / 4) * x ^ 2

def Q (x : ℝ) : ℝ := 
  P x * P (x ^ 4) * P (x ^ 6) * P (x ^ 8) * P (x ^ 10) * P (x ^ 12)

-- Prove that the sum of absolute values of the coefficients in Q(x) is 117649/4096
theorem sum_of_abs_b_i : Σ (i : ℕ) in Finset.range 101, |(Q (x)).coeff i| = 117649 / 4096 := 
  sorry

end sum_of_abs_b_i_l71_71595


namespace students_playing_both_l71_71391

theorem students_playing_both (total_students : ℕ) (play_tennis_fraction : ℚ) (play_hockey_fraction : ℚ) :
  total_students = 600 →
  play_tennis_fraction = 3/4 →
  play_hockey_fraction = 0.60 →
  (total_students * play_tennis_fraction * play_hockey_fraction).to_nat = 270 :=
by
  intros h_total h_tennis h_hockey
  sorry

end students_playing_both_l71_71391


namespace octahedron_has_eulerian_circuit_l71_71140

-- Define the vertices of the octahedron
inductive OctahedronVertex
| A | B | C | A_1 | B_1 | C_1

open OctahedronVertex

-- Define the edge connections
def octahedronEdges : List (OctahedronVertex × OctahedronVertex) := [
  (A, B), (A, C), (A, A_1),
  (B, C), (B, B_1), (B, A_1),
  (C, A), (C, B), (C, C_1),
  (A_1, A), (A_1, B_1), (A_1, C_1),
  (B_1, B), (B_1, A_1), (B_1, C_1),
  (C_1, C), (C_1, B_1), (C_1, A_1)
]

-- Prove an Eulerian circuit exists
theorem octahedron_has_eulerian_circuit :
  ∃ cycle : List OctahedronVertex, (cycle.head = cycle.last) ∧
    (∀ edge ∈ octahedronEdges, ∃ i, (cycle.nth i, cycle.nth (i + 1) % cycle.length) = edge) :=
sorry

end octahedron_has_eulerian_circuit_l71_71140


namespace perimeter_is_120_l71_71765

-- Define the constants and the necessary conditions
def area : ℝ := 800
def width : ℝ := 20
def length : ℝ := area / width

-- Define the target property (perimeter calculation)
def perimeter : ℝ := 2 * (length + width)

-- The theorem we want to prove
theorem perimeter_is_120 : perimeter = 120 := by
  -- Use this placeholder instead of the complete proof
  sorry

end perimeter_is_120_l71_71765


namespace daria_needs_additional_two_weeks_l71_71571

def vacuum_cleaner_initial_cost : ℝ := 420
def piggy_bank_savings : ℝ := 65
def weekly_allowance : ℝ := 25

def dog_walking_income_week1 : ℝ := 40
def dog_walking_income_week2 : ℝ := 50
def dog_walking_income_week3 : ℝ := 30

def discount_percentage : ℝ := 0.15
def new_cost_after_discount : ℝ := vacuum_cleaner_initial_cost * (1 - discount_percentage)
def earnings_over_first_3_weeks : ℝ :=
  dog_walking_income_week1 + dog_walking_income_week2 + dog_walking_income_week3

def total_savings_after_3_weeks : ℝ :=
  piggy_bank_savings + 3 * weekly_allowance + earnings_over_first_3_weeks

def amount_still_needed : ℝ := new_cost_after_discount - total_savings_after_3_weeks
def weekly_savings : ℝ := weekly_allowance + dog_walking_income_week3

def number_of_additional_weeks_required : ℕ := (amount_still_needed / weekly_savings).ceil.to_nat

theorem daria_needs_additional_two_weeks :
  number_of_additional_weeks_required = 2 :=
by
  sorry

end daria_needs_additional_two_weeks_l71_71571


namespace solve_system_l71_71753

open Real

theorem solve_system :
  (∃ x y : ℝ, (1 / 2) * log 2 x - log 2 y = 0 ∧ x^2 - 2 * y^2 = 8 ∧
    ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = -2))) :=
by 
  sorry

end solve_system_l71_71753


namespace sum_of_roots_eq_14_l71_71090

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l71_71090


namespace tangent_line_at_point_l71_71956

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
x + 4 * y - 3 = 0

theorem tangent_line_at_point (x y : ℝ) (h₁ : y = 1 / x^2) (h₂ : x = 2) (h₃ : y = 1/4) :
  tangent_line_equation x y :=
by 
  sorry

end tangent_line_at_point_l71_71956


namespace green_peppers_weight_l71_71570

theorem green_peppers_weight (total_weight : ℝ) (w : ℝ) (h1 : total_weight = 5.666666667)
  (h2 : 2 * w = total_weight) : w = 2.8333333335 :=
by
  sorry

end green_peppers_weight_l71_71570


namespace identity_n1_n2_product_l71_71707

theorem identity_n1_n2_product :
  (∃ (N1 N2 : ℤ),
    (∀ x : ℚ, (35 * x - 29) / (x^2 - 3 * x + 2) = N1 / (x - 1) + N2 / (x - 2)) ∧
    N1 * N2 = -246) :=
sorry

end identity_n1_n2_product_l71_71707


namespace largest_base_digits_sum_not_12_l71_71581

-- Define the mathematical conditions
def nine_cubed := 9^3
def base_b_digits_sum (b : ℕ) (n : ℕ) : ℕ :=
  if n < b then n
  else base_b_digits_sum b (n / b) + (n % b)

-- Define the theorem to prove the solution
theorem largest_base_digits_sum_not_12 
  (b : ℕ) (h_b_gt_1 : b > 1) 
  (h_base_10_sum : base_b_digits_sum 10 nine_cubed ≠ 12) : 
  ∀ (b' : ℕ), (b' > b) → base_b_digits_sum b' nine_cubed = 12 → b' ≤ 10 :=
by
  sorry

end largest_base_digits_sum_not_12_l71_71581


namespace q_10_value_l71_71690

theorem q_10_value :
    ∃ (a : ℕ → ℝ) (q : ℕ → ℝ),
    (a 1 = 0) ∧ 
    (a 2 = 2) ∧
    (∀ k : ℕ, k > 0 → (2 * a (2*k) = a (2*k-1) + a (2*k+1))) ∧
    (∀ k : ℕ, k > 0 → (a (2*k+1) = a (2*k) * q k) ∧ (a (2*k+2) = a (2*k) * (q k)^2)) ∧
    (q 10 = 11 / 10) :=
begin
  sorry
end

end q_10_value_l71_71690


namespace a_less_than_2_l71_71302

-- Define the quadratic function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x + 2

-- Define the condition that the inequality f(x) - a > 0 has solutions in the interval [0,5]
def inequality_holds (a : ℝ) : Prop := ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 5 → f x - a > 0

-- Theorem stating that a must be less than 2 to satisfy the above condition
theorem a_less_than_2 : ∀ (a : ℝ), (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 5 ∧ f x - a > 0) → a < 2 := 
sorry

end a_less_than_2_l71_71302


namespace find_largest_integer_l71_71500

theorem find_largest_integer (n : ℕ) (h1 : n < 2012) 
(h2 : ∀ p : ℕ, p.prime → p ∣ n → (p^2 - 1) ∣ n) : n = 1944 :=
sorry

end find_largest_integer_l71_71500


namespace line_through_M_intersects_a_and_b_l71_71813

-- Definitions
variables (Point Line Plane : Type)
variables (M : Point) (a b : Line) (π1 : Plane)

-- Conditions
axiom M_not_on_a_or_b (h1 : ¬ (M ∈ a)) (h2: ¬ (M ∈ b)) : true
axiom plane_contains_line_and_point (π1_contains : ∀ (P : Point), (P ∈ a) → (P ∈ π1)) : true
axiom plane_contains_given_point (π1_contains_M : M ∈ π1) : true

-- Proof Problem
theorem line_through_M_intersects_a_and_b :
  (∃ (L : Line), (M ∈ L) ∧ (L ∩ a ≠ ∅) ∧ (L ∩ b ≠ ∅)) ↔
  ((∃ (A : Point), (A ∈ b) ∧ (A ∈ π1)) ∧ (∃ (L : Line), (L = line_through M A) ∧ (L ∩ a ≠ ∅))) := sorry

end line_through_M_intersects_a_and_b_l71_71813


namespace sum_of_roots_of_equation_l71_71078

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l71_71078


namespace sum_of_roots_of_quadratic_l71_71031

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l71_71031


namespace expected_coincidences_l71_71483

/-- Given conditions for the test -/
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8
def vasya_prob_correct : ℝ := 6 / 20
def misha_prob_correct : ℝ := 8 / 20
def coincidence_prob : ℝ :=
  (vasya_prob_correct * misha_prob_correct) + (1 - vasya_prob_correct) * (1 - misha_prob_correct)

/-- Expected number of coincidences -/
theorem expected_coincidences :
  20 * coincidence_prob = 10.8 :=
by {
  -- vasya_prob_correct = 0.3
  -- misha_prob_correct = 0.4
  -- probability of coincidence = 0.3 * 0.4 + 0.7 * 0.6 = 0.54
  -- expected number of coincidences = 20 * 0.54 = 10.8
  sorry
}

end expected_coincidences_l71_71483


namespace fraction_of_sum_l71_71120

theorem fraction_of_sum (numbers : List ℝ) (h_len : numbers.length = 21)
  (n : ℝ) (h_n : n ∈ numbers)
  (h_avg : n = 5 * ((numbers.sum - n) / 20)) :
  n / numbers.sum = 1 / 5 :=
by
  sorry

end fraction_of_sum_l71_71120


namespace valid_k_range_l71_71640

theorem valid_k_range (k : ℕ) (h : k ≥ 3) :
  ∀ n : ℕ, n ≥ 1 → let a : ℕ → ℝ := λ n, if n = 0 then (1 / k : ℝ) else 
    have recursively_defined : ∀ m : ℕ → ℝ, m n = m (n-1) + (1 / (n:ℝ)^2) * (m (n-1))^2 by sorry in
    recursively_defined n
  in ∀ m : ℕ → ℝ, m n < 1 := sorry

end valid_k_range_l71_71640


namespace problem_l71_71546

noncomputable def octagon_chord_length : ℚ := sorry

theorem problem : ∃ (p q : ℕ), p + q = 5 ∧ (nat.gcd p q = 1) ∧ (octagon_chord_length = p / q) :=
begin
  sorry
end

end problem_l71_71546


namespace tomatoes_picked_l71_71138

theorem tomatoes_picked (initial_tomatoes picked_tomatoes : ℕ)
  (h₀ : initial_tomatoes = 17)
  (h₁ : initial_tomatoes - picked_tomatoes = 8) :
  picked_tomatoes = 9 :=
by
  sorry

end tomatoes_picked_l71_71138


namespace remainder_and_division_l71_71836

theorem remainder_and_division (x y : ℕ) (h1 : x % y = 8) (h2 : (x / y : ℝ) = 76.4) : y = 20 :=
sorry

end remainder_and_division_l71_71836


namespace exists_subset_divisible_sum_l71_71733

theorem exists_subset_divisible_sum {n : ℕ} (h : 0 < n) (a : Fin (2 * n + 1) → ℤ) : 
  ∃ (s : Finset (Fin (2 * n + 1))), s.card = n ∧ (∑ i in s, a i) % n = 0 :=
sorry

end exists_subset_divisible_sum_l71_71733


namespace sum_of_roots_eq_14_l71_71036

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l71_71036


namespace equal_roots_polynomial_l71_71593

open ComplexConjugate

theorem equal_roots_polynomial (k : ℚ) :
  (3 : ℚ) * x^2 - k * x + 2 * x + (12 : ℚ) = 0 → 
  (b : ℚ) ^ 2 - 4 * (3 : ℚ) * (12 : ℚ) = 0 ↔ k = -10 ∨ k = 14 :=
by
  sorry

end equal_roots_polynomial_l71_71593


namespace percentage_students_below_8_years_l71_71669

theorem percentage_students_below_8_years :
  ∀ (n8 : ℕ) (n_gt8 : ℕ) (n_total : ℕ),
  n8 = 24 →
  n_gt8 = 2 * n8 / 3 →
  n_total = 50 →
  (n_total - (n8 + n_gt8)) * 100 / n_total = 20 :=
by
  intros n8 n_gt8 n_total h1 h2 h3
  sorry

end percentage_students_below_8_years_l71_71669


namespace least_positive_integer_condition_original_integer_is_725_l71_71959

theorem least_positive_integer_condition 
  (d n p : ℤ) 
  (h1 : 1 ≤ d ∧ d ≤ 9 ∧ d ∣ 28) 
  (h2 : 10^p * d + n = 29 * n) 
  (h3 : p > 0) 
  : n = 25 * 10^(p - 2) := 
sorry

theorem original_integer_is_725 
  (d n p : ℤ) 
  (h1 : 1 ≤ d ∧ d ≤ 9 ∧ d ∣ 28 = 7)
  (h2 : 10^p * d + n = 29 * n)
  (h3 : p = 2)
  (h4 : 25 * 10^(p - 2) = n)
  : (7 * 10^p + 25 * 10^(p - 2)) = 725 :=
sorry

end least_positive_integer_condition_original_integer_is_725_l71_71959


namespace small_load_clothing_count_l71_71816

def initial_clothes : ℕ := 36
def first_load_clothes : ℕ := 18
def remaining_clothes := initial_clothes - first_load_clothes
def small_load_clothes := remaining_clothes / 2

theorem small_load_clothing_count : 
  small_load_clothes = 9 :=
by
  sorry

end small_load_clothing_count_l71_71816


namespace sum_of_roots_eq_14_l71_71069

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l71_71069


namespace loss_percentage_is_20_l71_71162

-- Define necessary conditions
def CP : ℕ := 2000
def gain_percent : ℕ := 6
def SP_new : ℕ := CP + ((gain_percent * CP) / 100)
def increase : ℕ := 520

-- Define the selling price condition
def SP : ℕ := SP_new - increase

-- Define the loss percentage condition
def loss_percent : ℕ := ((CP - SP) * 100) / CP

-- Prove the loss percentage is 20%
theorem loss_percentage_is_20 : loss_percent = 20 :=
by sorry

end loss_percentage_is_20_l71_71162


namespace john_needs_to_contribute_zero_euros_l71_71672

/-- Given the conditions:
- The pastry costs 10 euros.
- Richard has a 20 CAD bill.
- The exchange rate is 1 euro = 1.50 CAD.
Prove that John needs to contribute 0 euros. -/
theorem john_needs_to_contribute_zero_euros (pastry_cost_in_euros : ℝ) (richard_money_in_CAD : ℝ) (exchange_rate : ℝ) (h1 : pastry_cost_in_euros = 10) (h2 : richard_money_in_CAD = 20) (h3 : exchange_rate = 1.50) : 
  ((richard_money_in_CAD / exchange_rate) < pastry_cost_in_euros) → 0 = 0 := 
by 
  assume h4 : (richard_money_in_CAD / exchange_rate) < pastry_cost_in_euros
  -- sorry is used to skip the proof.
  sorry

end john_needs_to_contribute_zero_euros_l71_71672


namespace side_length_of_square_field_l71_71503

theorem side_length_of_square_field 
  (time : ℕ) 
  (speed_kmph : ℕ) 
  (time = 60) 
  (speed_kmph = 12) : 
  side_length = 50 := 
sorry

end side_length_of_square_field_l71_71503


namespace no_remove_all_tokens_l71_71155

theorem no_remove_all_tokens
    (board : Type)
    [fintype board]
    (neighbors : board → set board)
    (initial_state : board → bool)
    (vertex : board)
    (triangle : ∀ b, card (neighbors b) = 2 ∨ card (neighbors b) = 3)
    (initial_black : initial_state vertex = tt)
    (initial_white : ∀ b, b ≠ vertex → initial_state b = ff)
    (flip : bool → bool)
    (flip_black : flip tt = ff)
    (flip_white : flip ff = tt) :
  ¬(∃ moves : list board,
      ∀ b ∈ moves, initial_state b = tt ∧
      (∀ b, 
       initial_state b = tt → 
       (∀ n ∈ neighbors b, initial_state n = flip (initial_state n))) ∧
      ∀ b, initial_state b = ff) :=
by
  sorry

end no_remove_all_tokens_l71_71155


namespace parity_of_F_solve_inequality_l71_71634

noncomputable def f (x : ℝ) : ℝ := Real.logBase (1/2) x

noncomputable def F (x : ℝ) : ℝ := f (x + 1) + f (1 - x)

theorem parity_of_F :
  (∀ x : ℝ, F (-x) = F x) :=
sorry

theorem solve_inequality :
  (∀ x : ℝ, abs (F x) ≤ 1 ↔ -Real.sqrt 2 / 2 ≤ x ∧ x ≤ Real.sqrt 2 / 2) :=
sorry

end parity_of_F_solve_inequality_l71_71634


namespace find_a_l71_71710

theorem find_a (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) 
    (h3 : a ^ 2 - b ^ 2 - c ^ 2 + a * b = 2011) 
    (h4 : a ^ 2 + 3 * b ^ 2 + 3 * c ^ 2 - 3 * a * b - 2 * a * c - 2 * b * c = -1997) : 
    a = 253 :=
by 
  sorry

end find_a_l71_71710


namespace solve_problem_l71_71603

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.logb 2 x else 3^x

theorem solve_problem : f (f (1 / 2)) = 1 / 3 := by
  sorry

end solve_problem_l71_71603


namespace side_length_of_square_l71_71768

theorem side_length_of_square (d : ℝ) (h : d = 4) : ∃ (s : ℝ), s = 2 * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l71_71768


namespace vasya_misha_expected_coincidences_l71_71484

noncomputable def expected_coincidences (n : ℕ) (pA pB : ℝ) : ℝ :=
  n * ((pA * pB) + ((1 - pA) * (1 - pB)))

theorem vasya_misha_expected_coincidences :
  expected_coincidences 20 (6 / 20) (8 / 20) = 10.8 :=
by
  -- Test definition and expected output
  let n := 20
  let pA := 6 / 20
  let pB := 8 / 20
  have h : expected_coincidences n pA pB =  20 * ((pA * pB) + ((1 - pA) * (1 - pB))) := rfl
  rw h
  sorry

end vasya_misha_expected_coincidences_l71_71484


namespace banana_equivalence_l71_71761

theorem banana_equivalence :
  (3 / 4 : ℚ) * 12 = 9 → (1 / 3 : ℚ) * 6 = 2 :=
by
  intro h1
  linarith

end banana_equivalence_l71_71761


namespace value_two_std_dev_less_l71_71420

noncomputable def mean : ℝ := 15.5
noncomputable def std_dev : ℝ := 1.5

theorem value_two_std_dev_less : mean - 2 * std_dev = 12.5 := by
  sorry

end value_two_std_dev_less_l71_71420


namespace find_leftmost_vertex_l71_71211

theorem find_leftmost_vertex
    (f : ℝ → ℝ)
    (vert : ℕ → (ℝ × ℝ))
    (A : ℝ)
    (n : ℕ)
    (ln : ℝ → ℝ)
    (h_vert : ∀ n : ℕ, vert n = (n, ln n))
    (h_f : ∀ x : ℝ, f x = ln x )
    (h_A : A = Real.log (24 / 23)) :
    ∃ n : ℕ, A = Real.log ((n + 1) ^ 2 / (n * (n + 2))) ∧ n = 12 := 
by
  sorry

end find_leftmost_vertex_l71_71211


namespace sum_of_roots_eq_14_l71_71071

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l71_71071


namespace age_difference_l71_71126

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 10) : c = a - 10 :=
by
  sorry

end age_difference_l71_71126


namespace sector_area_perimeter_one_angle_one_l71_71419

-- Define the conditions
variables {r l : ℝ} (h1 : l + 2 * r = 1) (h2 : l = r)

-- Define the theorem to prove the area
theorem sector_area_perimeter_one_angle_one : 
  let S := (1 / 2) * l * r in S = 1 / 18 :=
by
  sorry

end sector_area_perimeter_one_angle_one_l71_71419


namespace vasya_misha_expected_coincidences_l71_71487

noncomputable def expected_coincidences (n : ℕ) (pA pB : ℝ) : ℝ :=
  n * ((pA * pB) + ((1 - pA) * (1 - pB)))

theorem vasya_misha_expected_coincidences :
  expected_coincidences 20 (6 / 20) (8 / 20) = 10.8 :=
by
  -- Test definition and expected output
  let n := 20
  let pA := 6 / 20
  let pB := 8 / 20
  have h : expected_coincidences n pA pB =  20 * ((pA * pB) + ((1 - pA) * (1 - pB))) := rfl
  rw h
  sorry

end vasya_misha_expected_coincidences_l71_71487


namespace goods_train_speed_l71_71885

theorem goods_train_speed
  (man_train_speed : ℝ := 50)  -- condition 1
  (goods_train_length_m : ℝ := 400)  -- condition 4
  (time_seconds : ℝ := 6)  -- condition 3
  (km_per_hour_conversion : ℝ := 3600)  -- conversion factor from seconds to hours
  (goods_train_length_km : ℝ := goods_train_length_m / 1000)  -- conversion from meters to kilometers
  :
  let relative_speed := goods_train_length_km / (time_seconds / km_per_hour_conversion) in
  relative_speed = man_train_speed + 190 :=  -- correct answer with condition 2
sorry

end goods_train_speed_l71_71885


namespace mean_combined_l71_71433

-- Definitions for the two sets and their properties
def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

variables (set₁ set₂ : List ℕ)
-- Conditions based on the problem
axiom h₁ : set₁.length = 7
axiom h₂ : mean set₁ = 15
axiom h₃ : set₂.length = 8
axiom h₄ : mean set₂ = 30

-- Prove that the mean of the combined set is 23
theorem mean_combined (h₁ : set₁.length = 7) (h₂ : mean set₁ = 15)
  (h₃ : set₂.length = 8) (h₄ : mean set₂ = 30) : mean (set₁ ++ set₂) = 23 := 
sorry

end mean_combined_l71_71433


namespace expression_value_l71_71478

theorem expression_value (x y z : ℤ) (hx : x = -2) (hy : y = 1) (hz : z = 1) : 
  x^2 * y * z - x * y * z^2 = 6 :=
by
  rw [hx, hy, hz]
  rfl

end expression_value_l71_71478


namespace probability_first_spade_second_ace_l71_71001

theorem probability_first_spade_second_ace :
  let deck_size := 52
  let non_ace_spades := 12
  let num_aces := 4
  let ace_of_spades := 1
  let remaining_after_first_non_ace_spade := (deck_size - 1)
  let remaining_after_first_ace_of_spade := (deck_size - 1)
  let probability :=
    (non_ace_spades / deck_size: ℚ) * (num_aces / remaining_after_first_non_ace_spade) + 
    (ace_of_spades / deck_size: ℚ) * ((num_aces - 1) / remaining_after_first_ace_of_spade)
  in
  probability = (1 / deck_size: ℚ) :=
by
  sorry

end probability_first_spade_second_ace_l71_71001


namespace moses_gets_more_l71_71451

noncomputable def moses_share : ℝ := 0.4 * 50
noncomputable def remainder : ℝ := 50 - moses_share
noncomputable def esther_share : ℝ := remainder / 2

theorem moses_gets_more (moses_share esther_share : ℝ) : moses_share - esther_share = 5 :=
by
  rw [moses_share, esther_share, remainder]
  calc
    0.4 * 50 - ((50 - (0.4 * 50)) / 2) 
      = 20 - ((50 - 20) / 2) : by norm_num
  ... = 20 - 15 : by norm_num
  ... = 5 : by norm_num

end moses_gets_more_l71_71451


namespace count_false_propositions_l71_71790

def prop (a : ℝ) := a > 1 → a > 2
def converse (a : ℝ) := a > 2 → a > 1
def inverse (a : ℝ) := a ≤ 1 → a ≤ 2
def contrapositive (a : ℝ) := a ≤ 2 → a ≤ 1

theorem count_false_propositions (a : ℝ) (h : ¬(prop a)) : 
  (¬(prop a) ∧ ¬(contrapositive a)) ∧ (converse a ∧ inverse a) ↔ 2 = 2 := 
  by
    sorry

end count_false_propositions_l71_71790


namespace least_number_divisible_by_13_l71_71827

theorem least_number_divisible_by_13 (n : ℕ) :
  (∀ m : ℕ, 2 ≤ m ∧ m ≤ 7 → n % m = 2) ∧ (n % 13 = 0) → n = 1262 :=
by sorry

end least_number_divisible_by_13_l71_71827


namespace cos_angle_a_c_l71_71623

open BigOperators

variables (a : ℝ × ℝ × ℝ) (c : ℝ × ℝ × ℝ)

noncomputable def vector_length (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def cos_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (vector_length v1 * vector_length v2)

theorem cos_angle_a_c : 
  vector_length a = 3 →
  c = (1, 2, 0) →
  dot_product (a - c) a = 4 →
  cos_angle a c = Real.sqrt 5 / 3 :=
by
  sorry -- proof goes here


end cos_angle_a_c_l71_71623


namespace minimum_cost_l71_71872

noncomputable def volume : ℝ := 4800
noncomputable def depth : ℝ := 3
noncomputable def base_cost_per_sqm : ℝ := 150
noncomputable def wall_cost_per_sqm : ℝ := 120
noncomputable def base_area (volume depth : ℝ) : ℝ := volume / depth
noncomputable def wall_surface_area (x : ℝ) : ℝ :=
  6 * x + (2 * (volume * depth / x))

noncomputable def construction_cost (x : ℝ) : ℝ :=
  wall_surface_area x * wall_cost_per_sqm + base_area volume depth * base_cost_per_sqm

theorem minimum_cost :
  ∃(x : ℝ), x = 40 ∧ construction_cost x = 297600 := by
  sorry

end minimum_cost_l71_71872


namespace total_interest_received_l71_71880

def principal_B : ℝ := 5000
def time_B : ℝ := 2
def principal_C : ℝ := 3000
def time_C : ℝ := 4
def rate : ℝ := 10
def SI (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem total_interest_received: 
  SI principal_B rate time_B + SI principal_C rate time_C = 2200 :=
by
  sorry

end total_interest_received_l71_71880


namespace large_pizza_increase_in_area_l71_71845

def percent_increase_in_area (r : ℝ) : ℝ :=
  let A_medium := π * r^2 in
  let A_large := π * (1.40 * r)^2 in
  ((A_large - A_medium) / A_medium) * 100

theorem large_pizza_increase_in_area (r : ℝ) :
  percent_increase_in_area r = 96 := by
  sorry

end large_pizza_increase_in_area_l71_71845


namespace factorize_x4_minus_4x2_l71_71950

theorem factorize_x4_minus_4x2 (x : ℝ) : 
  x^4 - 4 * x^2 = x^2 * (x - 2) * (x + 2) :=
by
  sorry

end factorize_x4_minus_4x2_l71_71950


namespace day_of_week_after_2_power_50_days_l71_71316

-- Conditions:
def today_is_monday : ℕ := 1  -- Monday corresponds to 1

def days_later (n : ℕ) : ℕ := (today_is_monday + n) % 7

theorem day_of_week_after_2_power_50_days :
  days_later (2^50) = 6 :=  -- Saturday corresponds to 6 (0 is Sunday)
by {
  -- Proof steps are skipped
  sorry
}

end day_of_week_after_2_power_50_days_l71_71316


namespace handshake_count_l71_71803

theorem handshake_count (n m : ℕ) (h1 : n = 15) (h2 : m = 5) (h3 : ∀ i : ℕ, i ∈ {1,2,3} → #(finsupp.single i (finsupp.support 0)) = m) :
  (n * (n - m)) / 2 = 75 :=
by
  -- Given that there are 15 people (n = 15) and each group has 5 people (m = 5)
  -- and each person will not shake hands with 4 others from their own group and themselves
  have handshakes_per_person : (n - m) := 10
  -- Total handshakes counted without removig duplicates: 15*10 = 150
  have total_handshakes : (n * (n - m)) := 150
  -- Removing duplicate counts, the final unique handshakes should be 75
  have unique_handshakes : (total_handshakes / 2) := 75
  -- Conclude the proof
  exact unique_handshakes

end handshake_count_l71_71803


namespace sum_of_roots_eq_14_l71_71037

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l71_71037


namespace combined_total_score_is_correct_l71_71670

-- Definitions of point values
def touchdown_points := 6
def extra_point_points := 1
def field_goal_points := 3

-- Hawks' Scores
def hawks_touchdowns := 4
def hawks_successful_extra_points := 2
def hawks_field_goals := 2

-- Eagles' Scores
def eagles_touchdowns := 3
def eagles_successful_extra_points := 3
def eagles_field_goals := 3

-- Calculations
def hawks_total_points := hawks_touchdowns * touchdown_points +
                          hawks_successful_extra_points * extra_point_points +
                          hawks_field_goals * field_goal_points

def eagles_total_points := eagles_touchdowns * touchdown_points +
                           eagles_successful_extra_points * extra_point_points +
                           eagles_field_goals * field_goal_points

def combined_total_score := hawks_total_points + eagles_total_points

-- The theorem that needs to be proved
theorem combined_total_score_is_correct : combined_total_score = 62 :=
by
  -- proof would go here
  sorry

end combined_total_score_is_correct_l71_71670


namespace inequality_solution_l71_71935

theorem inequality_solution (x : ℝ) : (x ≠ -2) ↔ (0 ≤ x^2 / (x + 2)^2) := by
  sorry

end inequality_solution_l71_71935


namespace sum_of_roots_l71_71098

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l71_71098


namespace companion_sets_count_l71_71656

def companion_set (A : Set ℝ) : Prop :=
  ∀ x ∈ A, (x ≠ 0) → (1 / x) ∈ A

def M : Set ℝ := { -1, 0, 1/2, 2, 3 }

theorem companion_sets_count : 
  ∃ S : Finset (Set ℝ), (∀ A ∈ S, companion_set A) ∧ (∀ A ∈ S, A ⊆ M) ∧ S.card = 3 := 
by
  sorry

end companion_sets_count_l71_71656


namespace sum_of_roots_eq_l71_71056

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l71_71056


namespace percentage_error_in_side_l71_71545

theorem percentage_error_in_side
  (s s' : ℝ) -- the actual and measured side lengths
  (h : (s' * s' - s * s) / (s * s) * 100 = 41.61) : 
  ((s' - s) / s) * 100 = 19 :=
sorry

end percentage_error_in_side_l71_71545


namespace largest_n_implies_condition_l71_71688

def Box (B : Type) := {
  x_min : ℝ,
  x_max : ℝ,
  y_min : ℝ,
  y_max : ℝ,
  non_empty : x_min < x_max ∧ y_min < y_max
}

def intersects (B1 B2 : Box) : Prop :=
  B1.x_min < B2.x_max ∧ B1.x_max > B2.x_min ∧
  B1.y_min < B2.y_max ∧ B1.y_max > B2.y_min

def boxes_intersect_cond (n : ℕ) (boxes : Fin n → Box) : Prop :=
  ∀ i j, i ≠ j → (intersects (boxes i) (boxes j)) ↔ (i ≠ j - 1 ∧ i ≠ (j + 1) % n)

theorem largest_n_implies_condition (n : ℕ) :
  (∀ (boxes : Fin n → Box), boxes_intersect_cond n boxes → n ≤ 6) ∧
  (∃ (boxes : Fin 6 → Box), boxes_intersect_cond 6 boxes) :=
by
  sorry

end largest_n_implies_condition_l71_71688


namespace total_right_handed_players_l71_71397

theorem total_right_handed_players
  (total_players throwers mp_players non_throwers L R : ℕ)
  (ratio_L_R : 2 * R = 3 * L)
  (total_eq : total_players = 120)
  (throwers_eq : throwers = 60)
  (mp_eq : mp_players = 20)
  (non_throwers_eq : non_throwers = total_players - throwers - mp_players)
  (non_thrower_sum_eq : L + R = non_throwers) :
  (throwers + mp_players + R = 104) :=
by
  sorry

end total_right_handed_players_l71_71397


namespace units_digit_S7890_l71_71977

noncomputable def c : ℝ := 4 + 3 * Real.sqrt 2
noncomputable def d : ℝ := 4 - 3 * Real.sqrt 2
noncomputable def S (n : ℕ) : ℝ := (1/2:ℝ) * (c^n + d^n)

theorem units_digit_S7890 : (S 7890) % 10 = 8 :=
sorry

end units_digit_S7890_l71_71977


namespace smaller_of_two_digit_prod_4500_l71_71439

theorem smaller_of_two_digit_prod_4500 (n1 n2 : ℕ) (hn1 : 10 ≤ n1 ∧ n1 < 100) (hn2 : 10 ≤ n2 ∧ n2 < 100) (hprod : n1 * n2 = 4500) :
  min n1 n2 = 50 :=
begin
  sorry
end

end smaller_of_two_digit_prod_4500_l71_71439


namespace normalized_sum_convergence_l71_71708

noncomputable theory
open ProbabilityMeasure

-- Define the sequence of independent random variables, each with a normal distribution N(0,1)
def xi (i : ℕ) : ℝ →ᵣ ℝ := sorry

-- Assumption that xi is distributed as N(0,1)
axiom xi_dist (i : ℕ) : xi i ~ normalMeasure 0 1

-- Statement: Convergence in distribution of the normalized sum
theorem normalized_sum_convergence :
  ∀ (ω : ℝ →ᵣ probability_space),
  tendsto (λ n, (∑ i in finset.range n, xi i ω) / real.sqrt n) at_top (𝓝 (normalMeasure 0 1)) :=
sorry

end normalized_sum_convergence_l71_71708


namespace dormouse_is_thief_l71_71320

-- Definitions of the suspects
inductive Suspect
| MarchHare
| Hatter
| Dormouse

open Suspect

-- Definitions of the statement conditions
def statement (s : Suspect) : Suspect :=
match s with
| MarchHare => Hatter
| Hatter => sorry -- Sonya and Hatter's testimonies are not recorded
| Dormouse => sorry -- Sonya and Hatter's testimonies are not recorded

-- Condition that only the thief tells the truth
def tells_truth (thief : Suspect) (s : Suspect) : Prop :=
s = thief

-- Conditions of the problem
axiom condition1 : statement MarchHare = Hatter
axiom condition2 : ∃ t, tells_truth t MarchHare ∧ ¬ tells_truth t Hatter ∧ ¬ tells_truth t Dormouse

-- Proposition that Dormouse (Sonya) is the thief
theorem dormouse_is_thief : (∃ t, tells_truth t MarchHare ∧ ¬ tells_truth t Hatter ∧ ¬ tells_truth t Dormouse) → t = Dormouse :=
sorry

end dormouse_is_thief_l71_71320


namespace binomial_sum_eq_sum_valid_n_values_l71_71830

theorem binomial_sum_eq (n : ℕ) (h₁ : nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16) :
  n = 14 ∨ n = 16 :=
sorry

theorem sum_valid_n_values :
  let n1 := 16
  let n2 := 14
  n1 + n2 = 30 :=
by
  -- proof to be provided; this is to check if the theorem holds
  sorry

end binomial_sum_eq_sum_valid_n_values_l71_71830


namespace sum_of_first_2014_terms_is_1007_l71_71615

-- Definitions and Conditions
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range n, a i

def collinear (A B C : ℝ × ℕ) : Prop :=
  ∃ m b : ℝ, ∀ P ∈ [A, B, C], P.2 = m * P.1 + b

-- Given conditions
variable {a : ℕ → ℤ}
variable {d : ℤ}
variable {a1 a2014 S2014 : ℤ}
variable {A B C : ℝ × ℕ}

axiom arithmetic_sequence : arithmetic_seq a d
axiom S_n_formula : sum_of_first_n a 2014 = a 0 + a 2014
axiom collinear_points : collinear (A.1, 0) (B.1, 0) (C.1, 2014)
axiom zero_not_origin_line : ¬ (collinear (0, 0) (A.1, 0) (C.1, 2014))

-- Proof goal
theorem sum_of_first_2014_terms_is_1007 :
  S2014 = 1007 :=
sorry

end sum_of_first_2014_terms_is_1007_l71_71615


namespace twice_not_square_l71_71798

theorem twice_not_square (m : ℝ) : 2 * m ≠ m * m := by
  sorry

end twice_not_square_l71_71798


namespace sum_of_valid_two_digit_integers_l71_71008

theorem sum_of_valid_two_digit_integers:
  (∑ n in ((range 90).filter 
  (λ n, 
    let a := n / 10,
    let b := n % 10 in
    ((1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) 
    ∧ ((a + b - 1) ∣ n) 
    ∧ ((a * b + 1) ∣ n)))
  , n) = 21 :=
by sorry

end sum_of_valid_two_digit_integers_l71_71008


namespace binom_10_5_eq_252_l71_71918

theorem binom_10_5_eq_252 : binomial 10 5 = 252 :=
by {
  sorry
}

end binom_10_5_eq_252_l71_71918


namespace pool_length_calc_l71_71943

variable (total_water : ℕ) (drinking_cooking_water : ℕ) (shower_water : ℕ) (shower_count : ℕ)
variable (pool_width : ℕ) (pool_height : ℕ) (pool_volume : ℕ)

theorem pool_length_calc (h1 : total_water = 1000)
    (h2 : drinking_cooking_water = 100)
    (h3 : shower_water = 20)
    (h4 : shower_count = 15)
    (h5 : pool_width = 10)
    (h6 : pool_height = 6)
    (h7 : pool_volume = total_water - (drinking_cooking_water + shower_water * shower_count)) :
    pool_volume = 600 →
    pool_volume = 60 * length →
    length = 10 :=
by
  sorry

end pool_length_calc_l71_71943


namespace no_rectangular_prism_equal_measures_l71_71695

theorem no_rectangular_prism_equal_measures (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0): 
  ¬ (4 * (a + b + c) = 2 * (a * b + b * c + c * a) ∧ 2 * (a * b + b * c + c * a) = a * b * c) :=
by
  sorry

end no_rectangular_prism_equal_measures_l71_71695


namespace distance_between_first_and_last_is_140_l71_71580

-- Given conditions
def eightFlowers : ℕ := 8
def distanceFirstToFifth : ℕ := 80
def intervalsBetweenFirstAndFifth : ℕ := 4 -- 1 to 5 means 4 intervals
def intervalsBetweenFirstAndLast : ℕ := 7 -- 1 to 8 means 7 intervals
def distanceBetweenConsecutiveFlowers : ℕ := distanceFirstToFifth / intervalsBetweenFirstAndFifth
def totalDistanceFirstToLast : ℕ := distanceBetweenConsecutiveFlowers * intervalsBetweenFirstAndLast

-- Theorem to prove the question equals the correct answer
theorem distance_between_first_and_last_is_140 :
  totalDistanceFirstToLast = 140 := by
  sorry

end distance_between_first_and_last_is_140_l71_71580


namespace solve_n_p_l71_71954

theorem solve_n_p (n : ℕ) (p : ℕ) [Nat.prime p] (hn : 0 < n) :
  17^n * 2^(n^2) - p = (2^(n^2 + 3) + 2^(n^2) - 1) * n^2 → 
  (p, n) = (17, 1) :=
by
  sorry

end solve_n_p_l71_71954


namespace simplify_and_evaluate_l71_71411

theorem simplify_and_evaluate (x : ℚ) (h1 : x = -1/3) :
    (3 * x + 2) * (3 * x - 2) - 5 * x * (x - 1) - (2 * x - 1)^2 = 9 * x - 5 ∧
    (9 * x - 5) = -8 := 
by sorry

end simplify_and_evaluate_l71_71411


namespace left_handed_women_percentage_l71_71319

theorem left_handed_women_percentage
  (x y : ℕ)
  (h1 : 4 * x = 5 * y)
  (h2 : 3 * x ≥ 3 * y) :
  (x / (4 * x) : ℚ) * 100 = 25 :=
by
  sorry

end left_handed_women_percentage_l71_71319


namespace sum_of_roots_eq_l71_71061

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l71_71061


namespace last_digit_of_a_power_b_l71_71116

-- Define the constants from the problem
def a : ℕ := 954950230952380948328708
def b : ℕ := 470128749397540235934750230

-- Define a helper function to calculate the last digit of a natural number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the main statement to be proven
theorem last_digit_of_a_power_b : last_digit ((last_digit a) ^ (b % 4)) = 4 :=
by
  -- Here go the proof steps if we were to provide them
  sorry

end last_digit_of_a_power_b_l71_71116


namespace problem1_l71_71561

theorem problem1 : 2 * real.cos (real.pi / 3) + abs (1 - 2 * real.sin (real.pi / 4)) + (1 / 2)^0 = real.sqrt 2 + 1 := 
by 
  have h1 : real.cos (real.pi / 3) = 1 / 2 := by sorry
  have h2 : real.sin (real.pi / 4) = real.sqrt 2 / 2 := by sorry
  have h3 : (1 / 2)^0 = 1 := by sorry
  sorry

end problem1_l71_71561


namespace first_reduction_is_12_percent_l71_71894

theorem first_reduction_is_12_percent (P : ℝ) (x : ℝ) (h1 : (1 - x / 100) * 0.9 * P = 0.792 * P) : x = 12 :=
by
  sorry

end first_reduction_is_12_percent_l71_71894


namespace cone_height_is_six_volume_at_two_thirds_height_l71_71868

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

def cone_height (V r : ℝ) : ℝ := (3 * V) / (Real.pi * r^2)

theorem cone_height_is_six 
  (r : ℝ := 5)
  (V : ℝ := 150)
  (h := cone_height V r) :
  h ≈ 6 :=
by 
  sorry

theorem volume_at_two_thirds_height 
  (r : ℝ := 5)
  (h : ℝ := 6)
  (new_height := (2 / 3) * h) :
  cone_volume r new_height ≈ 419 :=
by 
  sorry

end cone_height_is_six_volume_at_two_thirds_height_l71_71868


namespace trigonometric_identity_l71_71631

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = -3 / 2 :=
by
  sorry

end trigonometric_identity_l71_71631


namespace find_s_l71_71513

noncomputable def area_parallelogram (s : ℝ) : ℝ :=
  2 * s^2 * real.sqrt 3

theorem find_s (s : ℝ) (h : area_parallelogram s = 12 * real.sqrt 3) : s = real.sqrt 6 := by
  sorry

end find_s_l71_71513


namespace inflow_level_correct_l71_71871

-- Define conditions based on the problem
def water_tower_capacity : ℝ := 300
def domestic_water_consumption_rate : ℝ := 10
def industrial_water_consumption (t : ℝ) : ℝ := 100 * real.sqrt t
def inflow_rate (n : ℝ) : ℝ := 10 * n
def initial_water_volume : ℝ := 100

-- Define the water volume function
def water_volume (n t : ℝ) : ℝ := initial_water_volume + inflow_rate n * t 
  - domestic_water_consumption_rate * t - industrial_water_consumption t

-- Define the proof goal
theorem inflow_level_correct (t n : ℝ) (h : 0 < t ∧ t ≤ 16) : 
  0 < water_volume n t ∧ water_volume n t ≤ water_tower_capacity 
  → n = 4 :=
sorry

end inflow_level_correct_l71_71871


namespace smallest_n_f_gt_15_l71_71297

-- Define the function f
def f (n : ℕ) : ℕ :=
  let decimal_digits := (1 / (3^n : ℝ)).to_digits.right_of_decimal
  decimal_digits.sum_digits

-- The main theorem
theorem smallest_n_f_gt_15 : ∃ n : ℕ, n > 0 ∧ f(n) > 15 ∧ ∀ m : ℕ, m > 0 → f(m) > 15 → n ≤ m :=
by sorry

end smallest_n_f_gt_15_l71_71297


namespace calc_result_l71_71560

noncomputable def sqrt_expr : ℝ :=
  real.sqrt 48 / real.sqrt 3 + real.sqrt (1 / 2) * real.sqrt 12 - real.sqrt 24

theorem calc_result :
  sqrt_expr = 4 - real.sqrt 6 :=
sorry

end calc_result_l71_71560


namespace scientific_notation_of_34_million_l71_71531

theorem scientific_notation_of_34_million :
  let n := 34000000 in n = 3.4 * 10 ^ 7 :=
by
  sorry

end scientific_notation_of_34_million_l71_71531


namespace sum_first_9_terms_arithmetic_sequence_l71_71334

theorem sum_first_9_terms_arithmetic_sequence :
  ∀ {a : ℕ → ℝ} (h1 : a 3 + a 7 = 4), 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →
  ((∑ i in (finset.range 9), a i) = 18) :=
by
  intros a h1 h2
  sorry

end sum_first_9_terms_arithmetic_sequence_l71_71334


namespace non_overlapping_cubes_non_overlapping_tetrahedrons_non_overlapping_octahedrons_non_overlapping_icosahedrons_non_overlapping_dodecahedrons_l71_71498

open Function

-- Definitions related to convex polyhedra and Platonic solids
structure ConvexPolyhedron (P : Type) :=
(is_convex : ∀ x y z: P, (x≠y) ∧ (y≠z) ∧ (x≠z))
(contains : Set P → Prop)  -- contains the vertices

structure PlatonicSolid :=
(vertices : ∀ (n : ℕ), Set ℝ^3)
(faces : ∀ (n : ℕ), Set (Set(ℝ^3)))
(edges : ∀ (n : ℕ), Set (ℝ×ℝ))
(is_regular : ∀ {n m : ℕ}, n ≠ m → vertices n ≠ vertices m)

-- Lean 4 Statement for Problem 109
theorem non_overlapping_cubes (P : Type) [ConvexPolyhedron P] (cubes : Set (PlatonicSolid)) : 
  ∀ (cube1 cube2 : PlatonicSolid), cube1 ∈ cubes → cube2 ∈ cubes → (cube1 ≠ cube2) → ¬ is_convex P :=
sorry

theorem non_overlapping_tetrahedrons (P : Type) [ConvexPolyhedron P] (tetrahedra : Set (PlatonicSolid)) : 
  ∀ (t1 t2 : PlatonicSolid), t1 ∈ tetrahedra → t2 ∈ tetrahedra → (t1 ≠ t2) → ¬ is_convex P :=
sorry

theorem non_overlapping_octahedrons (P : Type) [ConvexPolyhedron P] (octahedra : Set (PlatonicSolid)) : 
  ∀ (o1 o2 : PlatonicSolid), o1 ∈ octahedra → o2 ∈ octahedra → (o1 ≠ o2) → ¬ is_convex P :=
sorry

theorem non_overlapping_icosahedrons (P : Type) [ConvexPolyhedron P] (icosahedra : Set (PlatonicSolid)) : 
  ∀ (i1 i2 : PlatonicSolid), i1 ∈ icosahedra → i2 ∈ icosahedra → (i1 ≠ i2) → ¬ is_convex P :=
sorry

theorem non_overlapping_dodecahedrons (P : Type) [ConvexPolyhedron P] (dodecahedra : Set (PlatonicSolid)) : 
  ∀ (d1 d2 : PlatonicSolid), d1 ∈ dodecahedra → d2 ∈ dodecahedra → (d1 ≠ d2) → ¬ is_convex P :=
sorry

end non_overlapping_cubes_non_overlapping_tetrahedrons_non_overlapping_octahedrons_non_overlapping_icosahedrons_non_overlapping_dodecahedrons_l71_71498


namespace smallest_m_value_l71_71198

theorem smallest_m_value :
  ∃ m, (∀ (a b c d : ℕ), gcd (gcd a b) (gcd c d) = 125 ∧ lcm (lcm a b) (lcm c d) = m → ∃! (a, b, c, d), (a, b, c, d) ∈ set.univ) ∧ -- Condition to count 125000 quadruplets
  m = 9450000 :=
begin
  sorry,
end

end smallest_m_value_l71_71198


namespace reverse_digits_multiplication_l71_71003

theorem reverse_digits_multiplication (a b : ℕ) (h₁ : a < 10) (h₂ : b < 10) : 
  (10 * a + b) * (10 * b + a) = 101 * a * b + 10 * (a^2 + b^2) :=
by 
  sorry

end reverse_digits_multiplication_l71_71003


namespace cylinder_volume_transformation_l71_71875

-- Define the original volume of the cylinder
def original_volume (V: ℝ) := V = 5

-- Define the transformation of quadrupling the dimensions of the cylinder
def new_volume (V V': ℝ) := V' = 64 * V

-- The goal is to show that under these conditions, the new volume is 320 gallons
theorem cylinder_volume_transformation (V V': ℝ) (h: original_volume V) (h': new_volume V V'):
  V' = 320 :=
by
  -- Proof is left as an exercise
  sorry

end cylinder_volume_transformation_l71_71875


namespace quadratic_non_negative_iff_a_in_range_l71_71642

theorem quadratic_non_negative_iff_a_in_range :
  (∀ x : ℝ, x^2 + (a - 2) * x + 1/4 ≥ 0) ↔ 1 ≤ a ∧ a ≤ 3 :=
sorry

end quadratic_non_negative_iff_a_in_range_l71_71642


namespace total_profit_at_year_end_l71_71529

noncomputable def calculate_total_profit (b_investment : ℝ) (c_share : ℝ) : ℝ :=
  let a_investment := 3 * b_investment
  let c_investment := (3 * b_investment) * (3/2)
  let total_parts := 6 * b_investment + 2 * b_investment + 9 * (b_investment / 2)
  let total_profit := (17 * c_share) / 9
  total_profit

theorem total_profit_at_year_end (b_investment : ℝ) (c_share : ℝ) (total_profit : ℝ) :
  A_invests_three_times_B : 3 * b_investment = a_investment,
  A_invests_two_thirds_C : a_investment = (2/3) * c_investment,
  C_share_is_6000 : c_share = 6000.000000000001,
  calculate_total_profit b_investment c_share = total_profit := by
  sorry

end total_profit_at_year_end_l71_71529


namespace sum_of_roots_of_quadratic_l71_71025

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l71_71025


namespace complex_number_in_first_quadrant_l71_71869

noncomputable def z : ℂ := (4 + 3 * complex.I) / (3 - 2 * complex.I)

theorem complex_number_in_first_quadrant :
  (3 - 2 * complex.I) * z = 4 + 3 * complex.I → (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_number_in_first_quadrant_l71_71869


namespace sum_of_remainders_is_six_l71_71812

theorem sum_of_remainders_is_six (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
  (a + b + c) % 15 = 6 :=
by
  sorry

end sum_of_remainders_is_six_l71_71812


namespace bus_on_time_probabilities_and_k2_statistic_l71_71850

theorem bus_on_time_probabilities_and_k2_statistic :
  let n := 500
  let a := 240
  let b := 20
  let c := 210
  let d := 30
  let p_A := a / (a + b)
  let p_B := c / (c + d)
  let k2 := n * ((a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  in p_A = 12 / 13 ∧ p_B = 7 / 8 ∧ k2 > 2.706 :=
by sorry

end bus_on_time_probabilities_and_k2_statistic_l71_71850


namespace train_length_correct_l71_71160

variables (time_to_cross : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ)

def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

noncomputable def length_of_train (time_to_cross bridge_length train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := kmph_to_mps train_speed_kmph
  let total_distance := train_speed_mps * time_to_cross
  total_distance - bridge_length

theorem train_length_correct :
  (time_to_cross = 27.997760179185665) →
  (bridge_length = 180) →
  (train_speed_kmph = 36) →
  length_of_train time_to_cross bridge_length train_speed_kmph = 99.97760179185665 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end train_length_correct_l71_71160


namespace polynomial_coefficient_sum_l71_71594

theorem polynomial_coefficient_sum
  (a b c d : ℤ)
  (h1 : (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + 2 * x^3 - 5 * x^2 + 8 * x - 12) :
  a + b + c + d = 6 := 
sorry

end polynomial_coefficient_sum_l71_71594


namespace sum_of_valid_x_l71_71833

noncomputable def median_is_mean (lst : List ℝ) : Prop :=
let med := (lst.sorted.get (2 : Fin lst.length)) in
let mean := (lst.sum / lst.length) in
med = mean

theorem sum_of_valid_x : 
  let lst := [3, 5, 7, 10] in 
  ∑ x in (finset.filter (λ x, median_is_mean (lst ++ [x])) (finset.Icc (-100) 100)), id x = 10 := 
sorry

end sum_of_valid_x_l71_71833


namespace bus_on_time_probabilities_and_k2_statistic_l71_71849

theorem bus_on_time_probabilities_and_k2_statistic :
  let n := 500
  let a := 240
  let b := 20
  let c := 210
  let d := 30
  let p_A := a / (a + b)
  let p_B := c / (c + d)
  let k2 := n * ((a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  in p_A = 12 / 13 ∧ p_B = 7 / 8 ∧ k2 > 2.706 :=
by sorry

end bus_on_time_probabilities_and_k2_statistic_l71_71849


namespace locus_of_points_where_tangents_are_adjoint_lines_l71_71643

theorem locus_of_points_where_tangents_are_adjoint_lines 
  (p : ℝ) (y x : ℝ)
  (h_parabola : y^2 = 2 * p * x) :
  y^2 = - (p / 2) * x :=
sorry

end locus_of_points_where_tangents_are_adjoint_lines_l71_71643


namespace sum_of_roots_eq_14_l71_71034

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l71_71034


namespace magnitude_relationship_l71_71610

variable (f : ℝ → ℝ)

axiom condition1 : ∀ x : ℝ, f(x + 2) = -f(x)
axiom condition2 : ∀ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2) → f(x1) < f(x2)
axiom condition3 : ∀ x : ℝ, f(x + 4) = f(-x + 4)

theorem magnitude_relationship : f(7) < f(4.5) ∧ f(4.5) < f(6.5) :=
by
  sorry

end magnitude_relationship_l71_71610


namespace sum_of_roots_l71_71101

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l71_71101


namespace normal_line_equation_at_x0_l71_71118

def curve (x : ℝ) : ℝ := x - x^3
noncomputable def x0 : ℝ := -1
noncomputable def y0 : ℝ := curve x0

theorem normal_line_equation_at_x0 :
  ∀ (y : ℝ), y = (1/2 : ℝ) * x + 1/2 ↔ (∃ (x : ℝ), y = curve x ∧ x = x0) :=
by
  sorry

end normal_line_equation_at_x0_l71_71118


namespace seating_arrangements_correct_l71_71677

def countSeatingArrangements : Nat :=
  let teams : List (List Nat) := [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  let perm (l : List α) : List (List α) :=
    if l = [] then [[]]
    else l.bind (λ hd => perm (l.erase hd)).map (λ tl => hd::tl)
  let teamPerms := teams.map perm
  teamPerms.length * teamPerms.head!.length * teamPerms.get! 1!.length * teamPerms.get! 2!.length

theorem seating_arrangements_correct :
  countSeatingArrangements = 1296 := by
  sorry

end seating_arrangements_correct_l71_71677


namespace parallelogram_triangle_area_l71_71394

theorem parallelogram_triangle_area (A B C D F G E : Point)
  (parallelogram : Parallelogram A B C D)
  (F_on_BC_extension : IsOnLine B C F)
  (AF_intersects_BD_at_E : IntersectsAt A F B D E)
  (AF_intersects_CD_at_G : IntersectsAt A F C D G)
  (GF_eq_3 : Distance G F = 3)
  (AE_eq_EG_plus_1 : Distance A E = Distance G E + 1) :
  Area (Triangle A D E) = (1/6) * Area (Parallelogram A B C D) :=
sorry

end parallelogram_triangle_area_l71_71394


namespace vertices_distance_l71_71766

noncomputable def distance_between_vertices_of_equilateral_triangles
  (a : ℝ) (distance_between_bases : ℝ) (h1 : distance_between_bases = 2 * a) : ℝ := sorry

theorem vertices_distance 
(a : ℝ) 
(h_nonneg : a ≥ 0) 
(distance_between_bases : ℝ) 
(h_db : distance_between_bases = 2 * a) : 
distance_between_vertices_of_equilateral_triangles a distance_between_bases h_db = 2 * a * real.sqrt 7 := 
sorry

end vertices_distance_l71_71766


namespace probability_calculation_l71_71579

def probability_each_delegate_sits_next_to_other_country (delegates : Fin 8 → Fin 4) : Prop := 
  ∀ (i : Fin 8), 
  delegates i ≠ delegates ((i + 1) % 8)

theorem probability_calculation :
  let total_arrangements := (8.fact / ((2.fact) ^ 4))
  let unwanted_arrangements := 360 - 36 + 8 - 8
  let wanted_arrangements := total_arrangements - unwanted_arrangements
  let probability := wanted_arrangements / total_arrangements
  probability = 131 / 140 :=
by 
  sorry

end probability_calculation_l71_71579


namespace percentage_increase_in_cellphone_pay_rate_l71_71163

theorem percentage_increase_in_cellphone_pay_rate
    (regular_rate : ℝ)
    (total_surveys : ℕ)
    (cellphone_surveys : ℕ)
    (total_earnings : ℝ)
    (regular_surveys : ℕ := total_surveys - cellphone_surveys)
    (higher_rate : ℝ := (total_earnings - (regular_surveys * regular_rate)) / cellphone_surveys)
    : regular_rate = 30 ∧ total_surveys = 100 ∧ cellphone_surveys = 50 ∧ total_earnings = 3300
    → ((higher_rate - regular_rate) / regular_rate) * 100 = 20 := by
  sorry

end percentage_increase_in_cellphone_pay_rate_l71_71163


namespace sum_of_roots_l71_71018

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l71_71018


namespace impossible_configuration_l71_71456

-- Define the initial state of stones in boxes
def stones_in_box (n : ℕ) : ℕ :=
  if n ≥ 1 ∧ n ≤ 100 then n else 0

-- Define the condition for moving stones between boxes
def can_move_stones (box1 box2 : ℕ) : Prop :=
  stones_in_box box1 + stones_in_box box2 = 101

-- The proposition: it is impossible to achieve the desired configuration
theorem impossible_configuration :
  ¬ ∃ boxes : ℕ → ℕ, 
    (boxes 70 = 69) ∧ 
    (boxes 50 = 51) ∧ 
    (∀ n, n ≠ 70 → n ≠ 50 → boxes n = stones_in_box n) ∧
    (∀ n1 n2, can_move_stones n1 n2 → (boxes n1 + boxes n2 = 101)) :=
sorry

end impossible_configuration_l71_71456


namespace probability_pink_second_marble_l71_71178

def bagA := (5, 5)  -- (red, green)
def bagB := (8, 2)  -- (pink, purple)
def bagC := (3, 7)  -- (pink, purple)

def P (success total : ℕ) := success / total

def probability_red := P 5 10
def probability_green := P 5 10

def probability_pink_given_red := P 8 10
def probability_pink_given_green := P 3 10

theorem probability_pink_second_marble :
  probability_red * probability_pink_given_red +
  probability_green * probability_pink_given_green = 11 / 20 :=
sorry

end probability_pink_second_marble_l71_71178


namespace ellipse_properties_l71_71256

-- Define the curve C: mx^2 + (1-m)y^2 = 1
def curve (m : ℝ) (x y : ℝ) := m * x^2 + (1 - m) * y^2 = 1

-- Given condition: C is an ellipse with its focus on the y-axis
def is_ellipse_with_focus_on_y_axis (m : ℝ) : Prop :=
  1 / (1 - m) > 1 / m

-- Define the range for m
def m_range (m : ℝ) : Prop :=
  1 / 2 < m ∧ m < 1

-- Proof statement
theorem ellipse_properties (m : ℝ) :
  (curve m x y) ∧ is_ellipse_with_focus_on_y_axis m →
  m_range m ∧
  (2 < 2 * sqrt (1 / m) ∧ 2 * sqrt (1 / m) < 2 * sqrt 2) :=
sorry

end ellipse_properties_l71_71256


namespace first_player_wins_with_optimal_play_l71_71141

theorem first_player_wins_with_optimal_play :
  ∃ strategy : ℕ → ℕ, ∀ match_count, (1 ≤ match_count ∧ match_count ≤ 5) →
  (match_count = 0 → first_player_wins_with_strat_n (strategy match_count)) → True :=
by sorry

end first_player_wins_with_optimal_play_l71_71141


namespace sum_of_roots_l71_71017

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l71_71017


namespace frank_eats_each_day_l71_71223

theorem frank_eats_each_day :
  ∀ (cookies_per_tray cookies_per_day days ted_eats remaining_cookies : ℕ),
  cookies_per_tray = 12 →
  cookies_per_day = 2 →
  days = 6 →
  ted_eats = 4 →
  remaining_cookies = 134 →
  (2 * cookies_per_tray * days) - (ted_eats + remaining_cookies) / days = 1 :=
  by
    intros cookies_per_tray cookies_per_day days ted_eats remaining_cookies ht hc hd hted hr
    sorry

end frank_eats_each_day_l71_71223


namespace parabola_problems_l71_71333

noncomputable def parabola_intersection_points (a : ℝ) (h₁ : a ≠ 0) : Prop :=
  let coords := (-1, 0) ∧ (5, 0)
  (∃ x y : ℝ, y = a * x^2 - 4 * a * x - 5 * a ∧ y = 0 ∧ (x, y) = coords)

noncomputable def analytical_expression_of_parabola (a : ℝ) (h₁ : a > 0) (m : ℝ) (h₂ : m ≥ 0) (n : ℝ) (h₃ : n ≥ -9) : Prop :=
  let expr := x^2 - 4 * x - 5
  (∃ y : ℝ, y = a * (x^2 - 4 * x - 5) ∧ y = x^2 - 4 * x - 5)

noncomputable def new_parabola_range (a m t : ℝ) (h₁ : a = 1) (h₂ : m > 0) (h₃ : -1/2 < t ∧ t < 5/2) : Prop :=
  let new_expr := x^2 - 4 * x - 5 + m
  (∃ m : ℝ, new_expr = 0 ∧ 11/4 < m ∧ m ≤ 9)

-- The main statement encompassing all the sub-problems
theorem parabola_problems {a m t : ℝ} (h₁ : a ≠ 0) (h₂ : a > 0) (h₃ : m ≥ 0) (h₄ : n ≥ -9) (h₅ : a = 1) (h₆ : m > 0) (h₇ : -1/2 < t ∧ t < 5/2) :
  parabola_intersection_points a h₁ ∧
  analytical_expression_of_parabola a h₂ m h₃ n h₄ ∧
  new_parabola_range a m t h₅ h₆ h₇ :=
sorry

end parabola_problems_l71_71333


namespace student_contribution_is_4_l71_71332

-- Definitions based on the conditions in the problem statement
def total_contribution := 90
def available_class_funds := 14
def number_of_students := 19

-- The theorem statement to be proven
theorem student_contribution_is_4 : 
  (total_contribution - available_class_funds) / number_of_students = 4 :=
by
  sorry  -- Proof is not required as per the instructions

end student_contribution_is_4_l71_71332


namespace pieces_eaten_first_l71_71180

variable (initial_candy : ℕ) (remaining_candy : ℕ) (candy_eaten_second : ℕ)

theorem pieces_eaten_first 
    (initial_candy := 21) 
    (remaining_candy := 7)
    (candy_eaten_second := 9) :
    (initial_candy - remaining_candy - candy_eaten_second = 5) :=
sorry

end pieces_eaten_first_l71_71180


namespace max_value_x_2y_2z_l71_71984

theorem max_value_x_2y_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : x + 2*y + 2*z ≤ 15 :=
sorry

end max_value_x_2y_2z_l71_71984


namespace positive_solution_iff_abs_a_b_lt_one_l71_71735

theorem positive_solution_iff_abs_a_b_lt_one
  (a b : ℝ)
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 - x2 = a)
  (h2 : x3 - x4 = b)
  (h3 : x1 + x2 + x3 + x4 = 1)
  (h4 : x1 > 0)
  (h5 : x2 > 0)
  (h6 : x3 > 0)
  (h7 : x4 > 0) :
  |a| + |b| < 1 :=
sorry

end positive_solution_iff_abs_a_b_lt_one_l71_71735


namespace sum_of_roots_eq_14_l71_71091

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l71_71091


namespace scientific_notation_of_34_million_l71_71532

theorem scientific_notation_of_34_million :
  let n := 34000000 in n = 3.4 * 10 ^ 7 :=
by
  sorry

end scientific_notation_of_34_million_l71_71532


namespace calculate_value_l71_71462

-- Given conditions
def n : ℝ := 2.25

-- Lean statement to express the proof problem
theorem calculate_value : (n / 3) * 12 = 9 := by
  -- Proof will be supplied here
  sorry

end calculate_value_l71_71462


namespace complex_addition_l71_71981

def imag_unit_squared (i : ℂ) : Prop := i * i = -1

theorem complex_addition (a b : ℝ) (i : ℂ)
  (h1 : a + b * i = i * i)
  (h2 : imag_unit_squared i) : a + b = -1 := 
sorry

end complex_addition_l71_71981


namespace fraction_of_number_l71_71505

theorem fraction_of_number (x : ℚ) (h₁ : x = 1/3) (h₂ : ∀ f : ℚ, f * x = (16/216) * (1/x) → f = 2/3) :
  ∃ f : ℚ, f * x = (16/216) * (1/x) ∧ f = 2/3 :=
by {
  use 2/3,
  split,
  { 
    sorry
  },
  { 
    sorry
  }
}

end fraction_of_number_l71_71505


namespace calculate_expression_l71_71184

theorem calculate_expression (x : ℝ) : 2 * x^3 * (-3 * x)^2 = 18 * x^5 :=
by
  sorry

end calculate_expression_l71_71184


namespace robert_salary_loss_l71_71847

theorem robert_salary_loss (S : ℝ) :
  let decreased_salary := S - 0.6 * S in 
  let increased_salary := decreased_salary + 0.6 * decreased_salary in
  (S - increased_salary) / S * 100 = 36 :=
by
  sorry

end robert_salary_loss_l71_71847


namespace simplify_and_evaluate_l71_71410

variables (a b : ℝ) (h_a : a = real.sqrt 3 - 3) (h_b : b = 3)

theorem simplify_and_evaluate :
  1 - (a - b) / (a + 2 * b) / ((a^2 - b^2) / (a^2 + 4 * a * b + 4 * b^2)) = - real.sqrt 3 :=
by 
  -- proof will be provided here
  sorry

end simplify_and_evaluate_l71_71410


namespace sum_of_roots_eq_14_l71_71087

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l71_71087


namespace super_cool_number_count_l71_71898

/-- A digit is an element from 1 to 9, inclusive. -/
def Digit : Type := Fin 9

/-- A three-digit number is called "cool" if one digit is half the product of the other two. -/
def is_cool (a b c : Digit) : Prop :=
    (2 * a = b * c) ∨ (2 * b = a * c) ∨ (2 * c = a * b)

/-- A three-digit number is called "super cool" if it satisfies the cool condition for two or more places. -/
def is_super_cool (a b c : Digit) : Prop := 
    (2 * a = b * c ∧ 2 * b = a * c) ∨ 
    (2 * a = b * c ∧ 2 * c = a * b) ∨ 
    (2 * b = a * c ∧ 2 * c = a * b) ∨
    (2 * a = b * c ∧ 2 * b = a * c ∧ 2 * c = a * b)

/-- There are exactly 25 different super cool numbers. -/
theorem super_cool_number_count : ∃ count : Nat, count = 25 ∧  
  count = (Finset.univ.filter (λ (a : Digit) (b : Digit) (c : Digit),
    is_super_cool a b c)).card := 
begin
  -- proof omitted
  sorry
end

end super_cool_number_count_l71_71898


namespace find_third_divisor_l71_71592

theorem find_third_divisor (n : ℕ) (d : ℕ) 
  (h1 : (n - 4) % 12 = 0)
  (h2 : (n - 4) % 16 = 0)
  (h3 : (n - 4) % d = 0)
  (h4 : (n - 4) % 21 = 0)
  (h5 : (n - 4) % 28 = 0)
  (h6 : n = 1012) :
  d = 3 :=
by
  sorry

end find_third_divisor_l71_71592


namespace identify_twin_points_l71_71258

open Real

def twin_points (F : ℝ × ℝ → Prop) (P1 P2 : ℝ × ℝ) : Prop :=
  F P1 ∧ F P2 ∧ P1.1 ≤ P2.1 ∧ P1.2 ≥ P2.2

theorem identify_twin_points :
  (∃ P1 P2, twin_points (λ ⟨x, y⟩, x^2 / 20 + y^2 / 16 = 1 ∧ x * y > 0) P1 P2) ∧
  (∃ P1 P2, twin_points (λ ⟨x, y⟩, y^2 = 4 * x) P1 P2) ∧
  (∃ P1 P2, twin_points (λ ⟨x, y⟩, |x| + |y| = 1) P1 P2) ∧
  ¬ (∃ P1 P2, twin_points (λ ⟨x, y⟩, x^2 / 20 - y^2 / 16 = 1 ∧ x * y > 0) P1 P2) := sorry

end identify_twin_points_l71_71258


namespace emily_eggs_collected_l71_71947

theorem emily_eggs_collected :
  let number_of_baskets := 1525
  let eggs_per_basket := 37.5
  let total_eggs := number_of_baskets * eggs_per_basket
  total_eggs = 57187.5 :=
by
  sorry

end emily_eggs_collected_l71_71947


namespace endpoint_distance_of_ellipse_l71_71926

noncomputable def distance_A'B'_ellipse : ℝ :=
  let ellipse_eq := 16 * (x + 2)^2 + 4 * y^2 = 64
  let a' := (0, 4)  -- Endpoint of the major axis (vertical radius)
  let b' := (2, 0)  -- Endpoint of the minor axis (horizontal radius)
  let dist := Real.dist a' b'
  dist

theorem endpoint_distance_of_ellipse :
  let A' := (0, 4)
  let B' := (2, 0)
  Real.dist A' B' = 2 * Real.sqrt 5 :=
by
  sorry

end endpoint_distance_of_ellipse_l71_71926


namespace no_point_IG_parallel_F1F2_equation_of_line_l71_71687

def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

def foci1 : ℝ × ℝ := (-3, 0)
def foci2 : ℝ × ℝ := (3, 0)
def vertex_left : ℝ × ℝ := (-2, 0)

def point_on_hyperbola (x y : ℝ) : Prop :=
  hyperbola x y

def centroid (xP yP : ℝ) : ℝ × ℝ :=
  ((xP - 3 + 3) / 3, yP / 3)

def parallel {a b c d : ℝ} : Prop :=
  (b - d) * (a - c) = 0

theorem no_point_IG_parallel_F1F2 (xP yP : ℝ) (hP : point_on_hyperbola xP yP) :
  ¬ parallel (centroid xP yP).1 (centroid xP yP).2 foci1.1 foci1.2 :=
sorry

theorem equation_of_line (k1 k2 : ℝ) (h : k1 + k2 = -1/2) :
  ∃ l : ℝ → ℝ → Prop, (∃ x1 x2 : ℝ, l x1 x2) ∧ (∀ x y, (l x y) ↔ x = 6) :=
sorry

end no_point_IG_parallel_F1F2_equation_of_line_l71_71687


namespace polynomial_symmetric_equiv_l71_71404

variable {R : Type*} [CommRing R]

def symmetric_about (P : R → R) (a b : R) : Prop :=
  ∀ x, P (2 * a - x) = 2 * b - P x

def polynomial_form (P : R → R) (a b : R) (Q : R → R) : Prop :=
  ∀ x, P x = b + (x - a) * Q ((x - a) * (x - a))

theorem polynomial_symmetric_equiv (P Q : R → R) (a b : R) :
  (symmetric_about P a b ↔ polynomial_form P a b Q) :=
sorry

end polynomial_symmetric_equiv_l71_71404


namespace distances_are_valid_triangle_inequality_l71_71122

noncomputable def distance_between_circles (O₁ O₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : ℝ :=
  let d := (O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2
  max 0 (sqrt d - abs (r₁ - r₂))

theorem distances_are_valid (O₁ O₂ : ℝ × ℝ) (r₁ r₂ : ℝ) :
  let d := (O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2 in
  let ρ := max 0 (sqrt d - abs (r₁ - r₂)) in
  let P := ρ in
  P = max 0 (sqrt d - abs (r₁ - r₂)) :=
by
  sorry

theorem triangle_inequality (Φ₁ Φ₂ Φ₃ : Type) [metric_space Φ₁] [metric_space Φ₂] [metric_space Φ₃] (P : Φ₁ → Φ₂ → ℝ) :
  ∀ x y z : Φ₁, P x y + P y z ≥ P x z :=
by
  sorry

end distances_are_valid_triangle_inequality_l71_71122


namespace castiel_sausages_l71_71383

theorem castiel_sausages : 
  let initial_sausages := 1200 in
  let after_monday := initial_sausages - (2 / 5 * initial_sausages) in
  let after_tuesday := after_monday - (1 / 2 * after_monday) in
  let after_wednesday := after_tuesday - (1 / 4 * after_tuesday) in
  let after_thursday := after_wednesday - (1 / 3 * after_wednesday) in
  let after_friday := after_thursday - (3 / 5 * after_thursday) in
  after_friday = 72 :=
by
  sorry

end castiel_sausages_l71_71383


namespace sum_of_roots_l71_71104

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l71_71104


namespace solve_system_l71_71750

open Real

theorem solve_system (x y : ℝ) : 
  (1/2 * log 2 x - log 2 y = 0) ∧ (x^2 - 2 * y^2 = 8) → ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = -2)) :=
by
  sorry

end solve_system_l71_71750


namespace team_selection_count_l71_71804

theorem team_selection_count :
  (nat.choose 6 2) * (nat.choose 5 1) = 75 := 
by sorry

end team_selection_count_l71_71804


namespace find_numbers_l71_71721

theorem find_numbers (a b c d e : ℕ) 
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (sums : multiset ℕ) 
  (h_sums : sums = {21, 26, 35, 40, 49, 51, 54, 60, 65, 79}) :
  a = 6 ∧ b = 15 ∧ c = 20 ∧ d = 34 ∧ e = 45 :=
begin
  sorry
end

end find_numbers_l71_71721


namespace A_share_of_annual_gain_l71_71530

-- Definitions based on the conditions
def investment_A (x : ℝ) : ℝ := 12 * x
def investment_B (x : ℝ) : ℝ := 12 * x
def investment_C (x : ℝ) : ℝ := 12 * x
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def annual_gain : ℝ := 15000

-- Theorem based on the question and correct answer
theorem A_share_of_annual_gain (x : ℝ) : (investment_A x / total_investment x) * annual_gain = 5000 :=
by
  sorry

end A_share_of_annual_gain_l71_71530


namespace library_pupils_count_l71_71511

-- Definitions for the conditions provided in the problem
def num_rectangular_tables : Nat := 7
def num_pupils_per_rectangular_table : Nat := 10
def num_square_tables : Nat := 5
def num_pupils_per_square_table : Nat := 4

-- Theorem stating the problem's question and the required proof
theorem library_pupils_count :
  num_rectangular_tables * num_pupils_per_rectangular_table + 
  num_square_tables * num_pupils_per_square_table = 90 :=
sorry

end library_pupils_count_l71_71511


namespace pairwise_sums_l71_71723

theorem pairwise_sums (
  a b c d e : ℕ
) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  (a + b = 21) ∧ (a + c = 26) ∧ (a + d = 35) ∧ (a + e = 40) ∧
  (b + c = 49) ∧ (b + d = 51) ∧ (b + e = 54) ∧ (c + d = 60) ∧
  (c + e = 65) ∧ (d + e = 79)
  ↔ 
  (a = 6) ∧ (b = 15) ∧ (c = 20) ∧ (d = 34) ∧ (e = 45) := 
by 
  sorry

end pairwise_sums_l71_71723


namespace sum_of_roots_of_quadratic_l71_71021

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l71_71021


namespace number_of_such_polynomials_l71_71286

noncomputable def polynomial_with_root_property : Prop :=
  ∃ (a b c d e : ℝ),
    (∀ r : ℂ, (r ≠ 0 → r^6 + (a : ℂ) * r^5 + (b : ℂ) * r^4 + (c : ℂ) * r^3 + (d : ℂ) * r^2 + (e : ℂ) * r + (2024 : ℂ) = 0 →
                (i * r)^6 + (a : ℂ) * (i * r)^5 + (b : ℂ) * (i * r)^4 + (c : ℂ) * (i * r)^3 + (d : ℂ) * (i * r)^2 + (e : ℂ) * (i * r) + (2024 : ℂ) = 0)) ∧ 
    (∀ s : ℂ, (s^6 + (a : ℂ) * s^5 + (b : ℂ) * s^4 + (c : ℂ) * s^3 + (d : ℂ) * s^2 + (e : ℂ) * s + (2024 : ℂ) = 0 → s = 0 ∨ s = i * s ∨ s = -s ∨ s = -i * s))

theorem number_of_such_polynomials : polynomial_with_root_property ∧ ∃! (f : ℂ → ℂ),
  ∃ (a b c d e : ℝ),
    (∀ r : ℂ, (r ≠ 0 → f r = r^6 + (a : ℂ) * r^5 + (b : ℂ) * r^4 + (c : ℂ) * r^3 + (d : ℂ) * r^2 + (e : ℂ) * r + (2024 : ℂ) ∧
                (f (i * r) = f r))) :=
sorry

end number_of_such_polynomials_l71_71286


namespace hyperbola_eccentricity_l71_71638

theorem hyperbola_eccentricity 
  (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) :
  let h_hyperbola := ∀ (x y : ℝ), x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1,
      h_function := ∀ (x : ℝ), y = sqrt x,
      intersect_pt := ∃ x0 y0, h_hyperbola x0 y0 ∧ h_function x0 y0,
      tangent_through_focus := ∀ x1 y1, 
        let slope := (1 / (2 * sqrt x1)),
        ∀ F : ℝ × ℝ, F = (-1, 0) → slope = (y1 - 0) / (x1 + 1) 
  in ∃ e : ℝ, e = (sqrt 5 + 1) / 2 :=
sorry

end hyperbola_eccentricity_l71_71638


namespace triangle_number_placement_l71_71934

theorem triangle_number_placement
  (A B C D E F : ℕ)
  (h1 : A + B + C = 6)
  (h2 : D = 5)
  (h3 : E = 6)
  (h4 : D + E + F = 14)
  (h5 : B = 3) : 
  (A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4) :=
by {
  sorry
}

end triangle_number_placement_l71_71934


namespace binom_10_5_eq_252_l71_71919

theorem binom_10_5_eq_252 : binomial 10 5 = 252 :=
by {
  sorry
}

end binom_10_5_eq_252_l71_71919


namespace general_term_formula_sum_of_first_n_terms_l71_71995

noncomputable def a_n (n : ℕ) : ℝ := n + 1

theorem general_term_formula :
  ∀ (d a_1 : ℝ) (n : ℕ), 
    d ≠ 0 → 
    (a_1 + 4 * d) ^ 2 = (a_1 + d) * (a_1 + 10 * d) → 
    7 * a_1 + (7 * 6 / 2) * d = 35 → 
    a_n 1 = a_1 → 
    a_n n = n + 1 :=
  sorry

theorem sum_of_first_n_terms (n : ℕ) :
  (∀ n, a_n n = n + 1) → (∑ i in range n, 1 / (a_n i * a_n (i + 1))) = (1 / 2) - (1 / (n + 2)) :=
  sorry

end general_term_formula_sum_of_first_n_terms_l71_71995


namespace sum_of_roots_eq_14_l71_71039

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l71_71039


namespace max_area_of_sector_l71_71992

theorem max_area_of_sector (α R C : Real) (hC : C > 0) (h : C = 2 * R + α * R) : 
  ∃ S_max : Real, S_max = (C^2) / 16 :=
by
  sorry

end max_area_of_sector_l71_71992


namespace magnitude_b_eq_sqrt_17_l71_71979

-- Definitions and conditions
def a : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (0, 2)
def b : ℝ × ℝ := (c.1 - a.1, c.2 - a.2)

-- Magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Statement to be proved
theorem magnitude_b_eq_sqrt_17 : magnitude b = real.sqrt 17 := by
  sorry

end magnitude_b_eq_sqrt_17_l71_71979


namespace triangle_side_range_l71_71692

theorem triangle_side_range (AB AC x : ℝ) (hAB : AB = 16) (hAC : AC = 7) (hBC : BC = x) :
  9 < x ∧ x < 23 :=
by
  sorry

end triangle_side_range_l71_71692


namespace probability_part1_probability_part2_l71_71991

-- Define the quadratic equation condition
def has_real_roots (a b : ℝ) : Prop :=
  a^2 + b^2 ≥ 4

-- Define finite sets for part (1)
def a_set : Finset ℝ := { -1, 0, 1 }
def b_set : Finset ℝ := { -3, -2, -1, 0, 1 }

-- Number of elements in sets for part (1)
def num_a := a_set.card
def num_b := b_set.card

-- Number of favorable cases for part (1)
def favorable_cases_1 : Finset (ℝ × ℝ) := 
  (a_set.product b_set).filter (λ ab, has_real_roots ab.1 ab.2)

-- Probability for part (1)
theorem probability_part1 : 
  (favorable_cases_1.card : ℝ) / (num_a * num_b : ℝ) = 2 / 5 :=
sorry

-- Define intervals for part (2)
def interval (l r : ℝ) : Set ℝ := { x | l ≤ x ∧ x ≤ r }
def interval_ab : Set (ℝ × ℝ) := 
  { p |  interval (-2) 2 p.1 ∧ interval (-2) 2 p.2 }

-- Area calculations for part (2)
def square_area := (4 : ℝ) * 4
def circle_area := real.pi * 2^2

-- Probability for part (2)
theorem probability_part2 : 
  (square_area - circle_area) / square_area = 1 - real.pi / 4 :=
sorry

end probability_part1_probability_part2_l71_71991


namespace initial_amount_P_l71_71349

-- Define the principal amount P, interest rate r, and number of years n
def P : ℝ
def r : ℝ := 0.04
def n : ℝ := 2
def loss : ℝ := 2.0000000000002274

-- Define compound interest calculation for 2 years
def compound_interest (P : ℝ) : ℝ := P * (1 + r)^n

-- Define simple interest calculation for 2 years
def simple_interest (P : ℝ) : ℝ := P * (1 + r * n)

-- The proof goal
theorem initial_amount_P :
  P * (1.0816 - 1.08) = loss -> P = 1250 :=
sorry

end initial_amount_P_l71_71349


namespace triangle_properties_l71_71693

theorem triangle_properties
  (A B C M : Point)
  (hA : A = (3, -1))
  (hCM : ∀ P : Point, is_on_line P (6x + 10y - 59 = 0) → P ∈ is_median C M)
  (hBT : ∀ P : Point, is_on_line P (x - 4y + 10 = 0) → P ∈ is_angle_bisector B) :
  (B = (10, 5)) ∧ (∃ (BC : Line), is_eq_line BC (2x + 9y - 65 = 0)) :=
by
  sorry

end triangle_properties_l71_71693


namespace find_a_l71_71376

noncomputable def A (a : ℝ) : Set ℝ := {1, 2, a}
noncomputable def B (a : ℝ) : Set ℝ := {1, a^2 - a}

theorem find_a (a : ℝ) : A a ⊇ B a → a = -1 ∨ a = 0 :=
by
  sorry

end find_a_l71_71376


namespace problem1_problem2_l71_71976

noncomputable def part1 (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4} ∩ {x | x ≤ 2 * a - 5}
noncomputable def part2 (a : ℝ) : Prop := ∀ x : ℝ, (-2 ≤ x ∧ x ≤ 4) → (x ≤ 2 * a - 5)

theorem problem1 : part1 3 = {x | -2 ≤ x ∧ x ≤ 1} :=
by { sorry }

theorem problem2 : ∀ a : ℝ, (part2 a) ↔ (a ≥ 9/2) :=
by { sorry }

end problem1_problem2_l71_71976


namespace hyperbola_eccentricity_l71_71255

theorem hyperbola_eccentricity (a b c : ℝ) (e : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
    (asymptote : ℝ → ℝ → Prop) 
    (h_asymptote_x : asymptote = (λ x y, y = sqrt 2 * x)) : 
    (e = sqrt 3 ∨ e = sqrt 6 / 2) → 
    (asymptote → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ asymptote = (λ x y, y = (b / a) * x ∨ asymptote = (λ x y, y = (a / b) * x) ∧ e = sqrt (1 + (b / a)^2) ∨ e = sqrt (1 + (a / b)^2))) →
      sorry

end hyperbola_eccentricity_l71_71255


namespace final_price_l71_71788

theorem final_price (S P : ℝ) : 
  let new_shirt_price := 1.2 * S,
      new_pants_price := 0.9 * P,
      combined_price := new_shirt_price + new_pants_price,
      increased_combined_price := combined_price * 1.15,
      final_combined_price := increased_combined_price * 0.95 in
  final_combined_price = 1.311 * S + 0.98325 * P :=
by
  sorry

end final_price_l71_71788


namespace sum_of_roots_eq_14_l71_71042

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l71_71042


namespace circle_radius_l71_71866

/-- Consider a square ABCD with a side length of 4 cm. A circle touches the extensions 
of sides AB and AD. From point C, two tangents are drawn to this circle, 
and the angle between the tangents is 60 degrees. -/
theorem circle_radius (side_length : ℝ) (angle_between_tangents : ℝ) : 
  side_length = 4 ∧ angle_between_tangents = 60 → 
  ∃ (radius : ℝ), radius = 4 * (Real.sqrt 2 + 1) :=
by
  sorry

end circle_radius_l71_71866


namespace jillian_max_apartment_size_l71_71171

variable (rent_per_sqft : ℝ) (max_budget : ℝ) (s : ℝ)
variable (h_rent_per_sqft : rent_per_sqft = 1.15)
variable (h_max_budget : max_budget = 690)

theorem jillian_max_apartment_size : rent_per_sqft * s = max_budget → s = 600 := by
  assume h : rent_per_sqft * s = max_budget
  -- Real calculation here
  sorry

end jillian_max_apartment_size_l71_71171


namespace pentagon_area_l71_71931

open Real

-- Define the pentagon and its properties
structure Pentagon :=
  (A B C D E P Q R S T : Point)
  (AB BC CD DE EA : ℝ)
  (ha : AB = 5)
  (hb : BC = 6)
  (hc : CD = 6)
  (hd : DE = 6)
  (he : EA = 7)
  (inscribed_circle : Prop)

-- Define the convex pentagon with the given properties
def convex_pentagon (ABCDE : Pentagon) : Prop :=
  is_convex ABCDE.A ABCDE.B ABCDE.C ABCDE.D ABCDE.E ∧
  ABCDE.inscribed_circle

-- The goal is to find the area of the convex pentagon
theorem pentagon_area (ABCDE : Pentagon) (h : convex_pentagon ABCDE) : 
  ∑ (triangle_area_by_sides ABCDE.ABC ABCDE.BCD ABCDE.CDE ABCDE.DEA ABCDE.EAB) = 60 :=
sorry

end pentagon_area_l71_71931


namespace point_not_on_graph_l71_71842

theorem point_not_on_graph : ¬ (∃ y, y = (λ x : ℝ, (x - 1) / (x + 2)) (-2) ∧ y = -3) :=
by
  intro h,
  rcases h with ⟨y, h_eq, h_y⟩,
  have h_undef : (λ x : ℝ, (x - 1) / (x + 2)) (-2) = (λ x : ℝ, (x - 1) / (x + 2)) (-2),
  exact rfl,
  rw h_eq at h_y,
  have : (-2 + 2) = 0, by norm_num,
  rw this at h_undef,
  linarith,

end point_not_on_graph_l71_71842


namespace equilateral_O1O2O3_l71_71551

variables {A B C D E F O O1 O2 O3 : Type}
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space D] [metric_space E] [metric_space F]
variables (dist : A → A → ℝ) (midpoint : A → A → A)
variables (eq_triangle : ∀ {X Y Z : A}, dist X Y = dist Y Z ∧ dist Y Z = dist Z X ∧ dist Z X = dist X Y)

-- Condition: Triangles ∆AOB, ∆COD, and ∆EOF are all equilateral.
def equilateral_AOB : Prop := eq_triangle A B O
def equilateral_COD : Prop := eq_triangle C O D
def equilateral_EOF : Prop := eq_triangle E O F

-- Condition: O1, O2, O3 are the midpoints of BC, DE, and FA respectively.
def midpoint_O1 : Prop := O1 = midpoint B C
def midpoint_O2 : Prop := O2 = midpoint D E
def midpoint_O3 : Prop := O3 = midpoint F A

-- Goal: Prove that ∆O1O2O3 is an equilateral triangle.
theorem equilateral_O1O2O3 
  (h1 : equilateral_AOB) 
  (h2 : equilateral_COD) 
  (h3 : equilateral_EOF) 
  (h4 : midpoint_O1) 
  (h5 : midpoint_O2) 
  (h6 : midpoint_O3) : 
  eq_triangle O1 O2 O3 :=
sorry

end equilateral_O1O2O3_l71_71551


namespace car_dealership_cars_l71_71175

theorem car_dealership_cars (total_cars : ℝ)
  (h1 : 15% cars cost less than $15000)
  (h2 : 40% cars cost more than $20000)
  (h3 : 1350 cars cost between $15000 and $20000) : total_cars = 3000 :=
sorry

end car_dealership_cars_l71_71175


namespace sum_of_roots_eq_14_l71_71096

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l71_71096


namespace eccentricity_of_hyperbola_l71_71249

-- Definitions based on the conditions
def is_hyperbola (E : Type) : Prop :=
  ∃ a b : ℝ, (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) ∧ a > 0 ∧ b > 0

def are_foci (F1 F2 : ℝ × ℝ) (a b c : ℝ) : Prop :=
  c = real.sqrt (a^2 + b^2) ∧ F1 = (-c, 0) ∧ F2 = (c, 0)

def on_hyperbola (M : ℝ × ℝ) (E : Type) : Prop :=
  ∃ x y a b : ℝ, is_hyperbola E ∧ M = (x, y) ∧ x^2 / a^2 - y^2 / b^2 = 1

def perpendicular_to_x_axis (M F1 : ℝ × ℝ) : Prop :=
  let ⟨xm, ym⟩ := M in let ⟨xf1, yf1⟩ := F1 in xm = xf1 ∧ ym ≠ yf1

def sine_condition (M F1 F2 : ℝ × ℝ) (sin_value : ℝ) : Prop :=
  let ⟨xm, ym⟩ := M in let ⟨xf1, yf1⟩ := F1 in let ⟨xf2, yf2⟩ := F2 in
  sin (real.arcsin (abs (ym - yf2) / real.sqrt ((xm - xf2)^2 + (ym - yf2)^2))) = sin_value

-- Lean statement for the proof problem
theorem eccentricity_of_hyperbola :
  ∀ (a b c : ℝ) (E : Type) (F1 F2 M : ℝ × ℝ),
  is_hyperbola E ∧ are_foci F1 F2 a b c ∧ on_hyperbola M E ∧ perpendicular_to_x_axis M F1 ∧ sine_condition M F1 F2 (1/3) →
  real.sqrt (a^2 + b^2) = real.sqrt 2 :=
sorry

end eccentricity_of_hyperbola_l71_71249


namespace work_completion_time_l71_71863

theorem work_completion_time :
  let work_rate_A := 1 / 8
  let work_rate_B := 1 / 6
  let work_rate_C := 1 / 4.8
  (work_rate_A + work_rate_B + work_rate_C) = 1 / 2 :=
by
  sorry

end work_completion_time_l71_71863


namespace binomial_10_5_is_252_l71_71922

theorem binomial_10_5_is_252 :
  Nat.binom 10 5 = 252 := by
  sorry

end binomial_10_5_is_252_l71_71922


namespace find_XY_l71_71951

-- Define the 30-60-90 triangle properties
def is_30_60_90 (X Y Z : Type) : Prop :=
  ∃ (l : ℝ), l > 0 ∧ XZ = l ∧ XY = l * √3 ∧ YZ = 2 * l

def XZ := 12

theorem find_XY (X Y Z : Type) (h: is_30_60_90 X Y Z) :
  XY = 12 * √3 :=
by
  sorry

end find_XY_l71_71951


namespace power_function_passes_through_fixed_point_l71_71779

theorem power_function_passes_through_fixed_point 
  (a : ℝ) (ha : 0 < a ∧ a ≠ 1)
  (P : ℝ × ℝ) (hP : P = (4, 2))
  (f : ℝ → ℝ) (hf : f x = x ^ a) : ∀ x, f x = x ^ (1 / 2) :=
by
  sorry

end power_function_passes_through_fixed_point_l71_71779


namespace total_trip_duration_l71_71499

theorem total_trip_duration (x : ℝ) (hours_minutes : ℝ × ℝ) :
  let rate := 35
  let standby_time := 6 - x
  let total_duration := (2 * (5 + 50/60))
  35 * standby_time = x ∧ total_duration = 11 + 40/60 :=
by
  unfold rate standby_time total_duration
  apply and.intro
  sorry

end total_trip_duration_l71_71499


namespace condition_1_correct_condition_2_correct_condition_3_correct_condition_4_correct_condition_5_correct_l71_71776

-- Formalize conditions as separate statements

def num_ways_condition_1 : ℕ :=
  4 * 5!

def num_ways_condition_2 : ℕ :=
  2 * 5!

def num_ways_condition_3 : ℕ :=
  6 * 5! - num_ways_condition_2

def num_ways_condition_4 : ℕ :=
  6 * 5! - num_ways_condition_2

def num_ways_condition_5 : ℕ :=
  2 * 4!

-- Lean statements for proof problems

theorem condition_1_correct : num_ways_condition_1 = 480 :=
by sorry

theorem condition_2_correct : num_ways_condition_2 = 240 :=
by sorry

theorem condition_3_correct : num_ways_condition_3 = 480 :=
by sorry

theorem condition_4_correct : num_ways_condition_4 = 360 :=
by sorry

theorem condition_5_correct : num_ways_condition_5 = 48 :=
by sorry

end condition_1_correct_condition_2_correct_condition_3_correct_condition_4_correct_condition_5_correct_l71_71776


namespace sum_of_roots_l71_71100

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l71_71100


namespace definite_integral_eval_l71_71202

-- Definitions based on the problem conditions
def first_integral := ∫ x in -2..2, real.sqrt (4 - x^2)
def second_integral := ∫ x in -2..2, -x^2017

-- Main theorem statement
theorem definite_integral_eval :
  (first_integral + second_integral) = 2 * real.pi :=
by sorry

end definite_integral_eval_l71_71202


namespace symmetry_center_monotonic_interval_min_value_of_a_l71_71982

noncomputable def f (x : ℝ) : ℝ :=
  cos x ^ 2 - (sqrt 3 / 2) * sin (2 * x) - 1 / 2

theorem symmetry_center (k : ℤ) : 
  ∃ x : ℝ, f(x) = f(x + k * π) :=
sorry

theorem monotonic_interval (k : ℤ) :
  ∀ x y : ℝ, -2 * π / 3 + k * π ≤ x ∧ x ≤ -π / 6 + k * π ∧ -2 * π / 3 + k * π ≤ y ∧ y ≤ -π / 6 + k * π → (x < y → f x < f y) :=
sorry

variables {a b c A : ℝ}

theorem min_value_of_a :
  ∃ A : ℝ, 99 <= 45 * cos (2 * A + π / 3) ∧ A = π / 3 → 
  ∃ a : ℝ, a ^ 2 = 1 :=
sorry

end symmetry_center_monotonic_interval_min_value_of_a_l71_71982


namespace tattoo_ratio_l71_71166

theorem tattoo_ratio (a j k : ℕ) (ha : a = 23) (hj : j = 10) (rel : a = k * j + 3) : a / j = 23 / 10 :=
by {
  -- Insert proof here
  sorry
}

end tattoo_ratio_l71_71166


namespace max_value_of_f_tan_expression_value_l71_71980

noncomputable def f (x : ℝ) : ℝ :=
  let a : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 / 2)
  let b : ℝ × ℝ := (Real.sin (x - Real.pi / 3), 1)
  a.1 * b.1 + a.2 * b.2

theorem max_value_of_f (x : ℝ) :
  f x = 1 ↔ ∃ (k : ℤ), x = (5 * Real.pi / 12) + k * Real.pi :=
by
  sorry

theorem tan_expression_value
  (x0 : ℝ)
  (hx0 : x0 ∈ (5 * Real.pi / 12)..(2 * Real.pi / 3))
  (hfx0 : f x0 = 4 / 5) :
  Real.tan (2 * x0 - Real.pi / 12) = -1 / 7 :=
by
  sorry

end max_value_of_f_tan_expression_value_l71_71980


namespace reciprocal_squares_sum_l71_71928

theorem reciprocal_squares_sum :
  (∃ a b c : ℝ, (a + b + c = 15) ∧ (a * b + b * c + c * a = 26) ∧ (a * b * c = 8) ∧
  (x - a) * (x - b) * (x - c) = polynomial.map polynomial.C polynomial.C (x^3 - 15 * x^2 + 26 * x - 8)) →
  1 / a^2 + 1 / b^2 + 1 / c^2 = 109 / 16 :=
begin
  sorry
end

end reciprocal_squares_sum_l71_71928


namespace pure_imaginary_a_value_l71_71299

theorem pure_imaginary_a_value (a : ℝ) : 
  let z := (1 + a * complex.I) / (2 - complex.I) 
  in (z.re = 0) → a = 2 := 
by
  let z := (1 + a * complex.I) / (2 - complex.I)
  assume h : z.re = 0
  sorry

end pure_imaginary_a_value_l71_71299


namespace current_speed_l71_71143

-- The main statement of our problem
theorem current_speed (v_with_current v_against_current c man_speed : ℝ) 
  (h1 : v_with_current = man_speed + c) 
  (h2 : v_against_current = man_speed - c) 
  (h_with : v_with_current = 15) 
  (h_against : v_against_current = 9.4) : 
  c = 2.8 :=
by
  sorry

end current_speed_l71_71143


namespace largest_k_for_matrices_l71_71360

theorem largest_k_for_matrices (n : ℕ) (k : ℕ) 
    (M : fin k → matrix (fin n) (fin n) ℝ) 
    (N : fin k → matrix (fin n) (fin n) ℝ) :
  (∀ i j : fin k, (i ≠ j → ∃ (l : fin n), (M i ⬝ N j) l l = 0) /\ 
  (i = j → ∃ l, (M i ⬝ N j) l l ≠ 0)) → 
  k ≤ n^n := 
begin
  sorry
end

end largest_k_for_matrices_l71_71360


namespace water_added_10_liters_l71_71860

variable 
  (V_initial : ℝ) (perc_water : ℝ) (perc_kola : ℝ) (perc_sugar : ℝ)
  (sugar_added : ℝ) (water_added : ℝ) (kola_added : ℝ)
  (perc_sugar_final : ℝ) (V_final : ℝ) (amount_sugar_initial : ℝ) 
  (amount_sugar_final : ℝ) (total_added : ℝ) (sugar_ratio_final : ℝ)
(initialize_conditions : V_initial = 340 ∧ perc_water = 0.8 ∧ perc_kola = 0.06 ∧ perc_sugar = 0.14 ∧ sugar_added = 3.2 ∧ kola_added = 6.8 ∧ perc_sugar_final = 14.111111111111112 / 100)
(volume_computation : V_final = V_initial + sugar_added + water_added + kola_added)
(sugar_initial_computation : amount_sugar_initial = perc_sugar * V_initial)
(sugar_final_computation : amount_sugar_final = amount_sugar_initial + sugar_added)
(sugar_ratio_final_computation : perc_sugar_final * V_final = sugar_final_computation)
(final_volume_computation : V_final = sugar_final_computation / perc_sugar_final)
(total_added_computation : total_added = V_final - V_initial)
(water_computation : water_added = total_added - sugar_added - kola_added )

theorem water_added_10_liters (h : initialize_conditions): water_added = 10 := by
  sorry

end water_added_10_liters_l71_71860


namespace proof_problem_l71_71271

section Problem
  variable (a : ℕ → ℝ) (b : ℕ → ℝ)
  
  -- Define the sequence {a_n}
  def seq_a (n : ℕ) : ℝ := a n

  -- Condition: Given equation for sequence {a_n}
  def seq_a_cond (n : ℕ) : Prop :=
    (finset.range n).sum (λ k => (k + 1: ℕ) * a (k + 1)) = 4 - (n + 2) / 2^(n - 1)

  -- General formula for sequence {a_n}
  def general_a_formula : Prop :=
    ∀ n, a n = 1 / 2^(n - 1)

  -- Define the sequence {b_n}
  def seq_b (n : ℕ) : ℝ := (3 * n - 2) * a n

  -- Define the sum of the first n terms of {b_n}
  def sum_b (n : ℕ) : ℝ :=
    (finset.range n).sum (λ k => b (k + 1))

  -- Given condition for sequence {b_n}
  def seq_b_cond : Prop :=
    ∀ n, b n = (3 * n - 2) * a n

  -- Prove that the sum of the first n terms of {b_n} equals 8 - (3n + 4) / 2^(n - 1)
  def sum_b_formula : Prop :=
    ∀ n, sum_b n = 8 - (3 * n + 4) / 2^(n - 1)

  -- Final theorem combining all
  theorem proof_problem :
    (∀ n, seq_a_cond a n) →
    general_a_formula a →
    seq_b_cond b →
    sum_b_formula a b :=
    sorry
end Problem

end proof_problem_l71_71271


namespace num_ways_coloring_l71_71417

def f : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 3
| n + 3 := f (n + 2) + f (n + 1)

theorem num_ways_coloring (n : ℕ) (h : n = 9) : f n = 89 :=
by
  sorry

end num_ways_coloring_l71_71417


namespace total_area_covered_by_strips_l71_71188

theorem total_area_covered_by_strips (L W : ℝ) (n : ℕ) (overlap_area : ℝ) (end_to_end_area : ℝ) :
  L = 15 → W = 1 → n = 4 → overlap_area = 15 → end_to_end_area = 30 → 
  (L * W * n - overlap_area + end_to_end_area) = 45 :=
by
  intros hL hW hn hoverlap hend_to_end
  sorry

end total_area_covered_by_strips_l71_71188


namespace checkerboard_probability_l71_71392

theorem checkerboard_probability :
  let total_squares := 10 * 10,
      perimeter_squares := 36,
      inner_squares := total_squares - perimeter_squares,
      probability := inner_squares / total_squares in
  probability = 16 / 25 :=
by
  sorry

end checkerboard_probability_l71_71392


namespace power_identity_l71_71657

theorem power_identity (x : ℝ) (m n : ℕ) (h1 : x^m = 5) (h2 : x^n = -2) : x^(m + 2 * n) = 20 :=
by
  sorry

end power_identity_l71_71657


namespace pow2_gt_square_for_all_n_ge_5_l71_71002

theorem pow2_gt_square_for_all_n_ge_5 (n : ℕ) (h : n ≥ 5) : 2^n > n^2 :=
by
  sorry

end pow2_gt_square_for_all_n_ge_5_l71_71002


namespace triangle_angles_are_correct_l71_71794

noncomputable def angle_of_triangle (a b c : ℝ) (cos : ℝ) : ℝ :=
  real.arccos cos

theorem triangle_angles_are_correct :
  let a := 3
  let b := 3
  let c := real.sqrt 15 - real.sqrt 3
  let cos_angle := (a^2 + b^2 - c^2) / (2 * a * b)
  let theta := angle_of_triangle a b c cos_angle
  let phi := (180 - theta) / 2
  theta = 26.57 ∧ phi = 76.715 :=
by
  sorry

end triangle_angles_are_correct_l71_71794


namespace sum_of_roots_l71_71015

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l71_71015


namespace membership_total_discount_l71_71893

theorem membership_total_discount 
  (original_price : ℝ)
  (h1 : original_price > 0) :
  let sale_price := 0.4 * original_price in
  let member_discount_price := 0.9 * sale_price in
  let final_price := 0.95 * member_discount_price in
  let total_discount := 1 - (final_price / original_price) in
  total_discount = 0.658 :=
by
  sorry

end membership_total_discount_l71_71893


namespace correct_option_l71_71611

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom periodicity : ∀ x : ℝ, f(x + 4) = f(x)
axiom decreasing : ∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 → f(x1) > f(x2)
axiom symmetry : ∀ x : ℝ, f(x - 2) = f(-x - 2)

-- Proof goal
theorem correct_option :
  f(-1.5) < f(7) ∧ f(7) < f(-4.5) :=
sorry

end correct_option_l71_71611


namespace sum_of_roots_of_quadratic_l71_71030

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l71_71030


namespace total_pieces_of_art_l71_71696

variable (asianArt egyptianArt europeanArt : ℕ)

theorem total_pieces_of_art (h1 : asianArt = 465) (h2 : egyptianArt = 527) (h3 : europeanArt = 320) :
  asianArt + egyptianArt + europeanArt = 1312 :=
by
  rw [h1, h2, h3]
  norm_num

/-! Sorry, we're skipping the detailed proof steps -/

end total_pieces_of_art_l71_71696


namespace identify_quadratic_function_l71_71540

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

theorem identify_quadratic_function :
  let fA := (λ x : ℝ, -3 * x^2)
      fB := (λ x : ℝ, 2 * x)
      fC := (λ x : ℝ, x + 1)
      fD := (λ x : ℝ, x^3)
  in is_quadratic fA ∧ ¬ is_quadratic fB ∧ ¬ is_quadratic fC ∧ ¬ is_quadratic fD :=
by
  sorry

end identify_quadratic_function_l71_71540


namespace am_gm_inequality_l71_71374

theorem am_gm_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c) ^ 2 :=
by
  sorry

end am_gm_inequality_l71_71374


namespace shampoo_bottles_needed_l71_71556

variable (daily_use : ℚ) (days_in_leap_year : ℕ) (bottle_size : ℚ)

def bottles_needed (daily_use : ℚ) (days_in_leap_year : ℕ) (bottle_size : ℚ) : ℕ :=
  let washes_per_bottle := bottle_size / daily_use
  let total_washes := days_in_leap_year / 2
  let bottles_required := (total_washes : ℚ) / washes_per_bottle
  ⌈bottles_required⌉ -- Ceiling function to ensure we get an integer number of bottles

theorem shampoo_bottles_needed : bottles_needed 1/4 366 14 = 4 := by
  sorry

end shampoo_bottles_needed_l71_71556


namespace find_n_l71_71652

variable (Q r j m n : ℝ)

theorem find_n (h : Q = r / (1 + j + m) ^ n) : 
  n = log (r / Q) / log (1 + j + m) :=
sorry

end find_n_l71_71652


namespace sum_of_roots_of_equation_l71_71083

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l71_71083


namespace factor_polynomial_l71_71587

theorem factor_polynomial (x y z : ℝ) : 
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (-(x * y + x * z + y * z)) :=
by
  sorry

end factor_polynomial_l71_71587


namespace eval_expression_l71_71183

theorem eval_expression :
  -1^(2022) + (Real.sqrt 16) * (-3)^2 + (-6) / Real.cbrt (-8) = 38 :=
by
  sorry

end eval_expression_l71_71183


namespace triangle_construction_valid_l71_71569

variable (a : ℝ) (α : ℝ)
variable (h1 : α < 90)

theorem triangle_construction_valid : (α < 90) → 
  let angle_bisector_45 : Prop := α < 90 ∧ ∃ β, β = 45 ∧ 2 * β + (90 - α) = 180
  in angle_bisector_45 :=
by
  sorry

end triangle_construction_valid_l71_71569


namespace value_of_a_l71_71270

theorem value_of_a (a : ℝ) (h₁ : ∀ x : ℝ, (2 * x - (1/3) * a ≤ 0) → (x ≤ 2)) : a = 12 :=
sorry

end value_of_a_l71_71270


namespace recurring_decimal_division_l71_71822

theorem recurring_decimal_division : (0.\overline{81} : ℚ) / (0.\overline{54} : ℚ) = 3 / 2 :=
by sorry

end recurring_decimal_division_l71_71822


namespace sum_of_roots_of_equation_l71_71080

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l71_71080


namespace problem_statement_l71_71775

theorem problem_statement
  (c d : ℕ)
  (h_factorization : ∀ x, x^2 - 18 * x + 72 = (x - c) * (x - d))
  (h_c_nonnegative : c ≥ 0)
  (h_d_nonnegative : d ≥ 0)
  (h_c_greater_d : c > d) :
  4 * d - c = 12 :=
sorry

end problem_statement_l71_71775


namespace perfect_squares_multiples_of_24_l71_71649

noncomputable def number_of_perfect_squares_less_than (n : ℕ) :=
  finset.card (finset.filter (λ x, is_square x) (finset.range n))

def is_multiple_of_24 (k : ℕ) : Prop :=
  k % 24 = 0

theorem perfect_squares_multiples_of_24 (m : ℕ) (h_m : m < 1000000) :
  number_of_perfect_squares_less_than 1000000 = 83 :=
by
  sorry

end perfect_squares_multiples_of_24_l71_71649


namespace simplify_f_find_cos_tan_of_alpha_l71_71604

noncomputable def f (x : ℝ) : ℝ := (cos (x - π / 2)) / (sin (7 * π / 2 + x)) * cos (π - x)

theorem simplify_f (x : ℝ) : f x = sin x :=
by
  sorry

theorem find_cos_tan_of_alpha (α : ℝ) (h : f α = -5 / 13) :
  cos α = 12 / 13 ∨ cos α = -12 / 13 ∧ (tan α = 5 / 12 ∨ tan α = -5 / 12) :=
by
  sorry

end simplify_f_find_cos_tan_of_alpha_l71_71604


namespace inequality_solution_l71_71797

theorem inequality_solution (x : ℝ) : 4 * x - 1 < 0 ↔ x < 1 / 4 := 
sorry

end inequality_solution_l71_71797


namespace distance_from_point_to_line_is_correct_l71_71214

open Real EuclideanSpace

noncomputable def point_distance_to_line (pt : ℝ × ℝ × ℝ) 
    (l_base l_dir : ℝ × ℝ × ℝ) : ℝ :=
  let s : ℝ := -(2 * l_dir.1 + 4 * l_dir.2 + 5 * l_dir.3 - (4 * l_base.1 + 3 * l_base.2 - l_base.3)) / (4 * l_dir.1 + 3 * l_dir.2 - l_dir.3) in
  let x := l_base.1 + s * l_dir.1 in
  let y := l_base.2 + s * l_dir.2 in
  let z := l_base.3 + s * l_dir.3 in
  (⟨x - pt.1, y - pt.2, z - pt.3⟩ : ℝ × ℝ × ℝ) ≫ (⟨1, 3, 1⟩: ℝ × ℝ × ℝ)

theorem distance_from_point_to_line_is_correct :
  point_distance_to_line (2, 4, 5) (4, 5, 6) (4, 3, -1) = sqrt 649 / 13 := sorry

end distance_from_point_to_line_is_correct_l71_71214


namespace sum_of_roots_eq_14_l71_71043

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l71_71043


namespace find_valid_pair_l71_71576

noncomputable def valid_angle (x : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 3 ∧ x = 180 * (n - 2) / n

noncomputable def valid_pair (x k : ℕ) : Prop :=
  valid_angle x ∧ valid_angle (k * x) ∧ 1 < k ∧ k < 5

theorem find_valid_pair : valid_pair 60 2 :=
by
  sorry

end find_valid_pair_l71_71576


namespace angle_quadrant_l71_71662

theorem angle_quadrant (α : ℝ) (h₁ : α = 2) (h₂ : real.pi / 2 < α) (h₃ : α < real.pi) : 
  second_quadrant α :=
by
  sorry

def second_quadrant (α : ℝ) : Prop :=
  real.pi / 2 < α ∧ α < real.pi

end angle_quadrant_l71_71662


namespace cube_volume_in_pyramid_is_correct_l71_71515

-- Conditions
def pyramid_base_side_length := 1
def lateral_faces_are_equilateral_triangles := true

-- Helper functions for volume calculations
def height_of_equilateral_triangle (a : ℝ) : ℝ := (real.sqrt 3 / 2) * a
def diagonal_of_square (s : ℝ) : ℝ := s * real.sqrt 2
def volume_of_cube (s : ℝ) : ℝ := s^3

-- Definition of the pyramid and the inner cube
def pyramid_height := height_of_equilateral_triangle (diagonal_of_square pyramid_base_side_length)

def cube_side_length := (real.sqrt 6 / 6)
def cube_volume := volume_of_cube cube_side_length

-- Theorem stating the required volume
theorem cube_volume_in_pyramid_is_correct :
  cube_volume = real.sqrt 6 / 36 :=
by sorry

end cube_volume_in_pyramid_is_correct_l71_71515


namespace no_real_roots_iff_l71_71309

theorem no_real_roots_iff (k : ℝ) : (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_iff_l71_71309


namespace shaded_area_l71_71916

noncomputable def area_shaded_region_rsght (r p q g h s t : Point) (radius : ℝ) : ℝ :=
  let pq := dist p q
  let gh := 2 * radius
  let rectangle_area := pq * gh
  let hypotenuse := dist r p
  let triangle_height := sqrt (hypotenuse^2 - radius^2)
  let triangle_area := (1/2) * radius * triangle_height
  let sector_angle := π / 4
  let sector_area := (sector_angle / (2 * π)) * π * radius^2
  let shaded_area := rectangle_area - 2 * (triangle_area + sector_area)
  shaded_area

theorem shaded_area :=
  let p, q, g, h, r, s, t : Point
  let radius := 3
  let pr := 3 * sqrt 3
  pq = 2 * pr
  pr = 3 * sqrt 3
  area_shaded_region_rsght r p q g h s t 3 = 36 * sqrt 3 - 9 * sqrt 2 - 9 * π / 4

end shaded_area_l71_71916


namespace greatest_among_three_l71_71573

theorem greatest_among_three : 
  let a := (2:ℝ)⁻³
  let b := (3:ℝ)^(1 / 2)
  let c := Real.log 5 / Real.log 2
  a < b → b < c → c = Real.log 5 / Real.log 2 := 
by
  intros a b c ha hb
  exact Real.log 5 / Real.log 2 = c

end greatest_among_three_l71_71573


namespace range_of_a_l71_71262

noncomputable def f (a x : ℝ) : ℝ := (Real.log x)^2 - (a / 2) * x * (Real.log x) + (a / Real.exp 1) * x^2

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (x1 x2 x3 : ℝ) (H1 : f = λ x, (Real.log x)^2 - (a / 2) * x * (Real.log x) + (a / Real.exp 1) * x^2)
  (H2 : f x1 = 0) (H3 : f x2 = 0) (H4 : f x3 = 0) (H5 : x1 < x2) (H6 : x2 < x3) :
  -2 / Real.exp 1 < a ∧ a < 0 :=
sorry

end range_of_a_l71_71262


namespace factor_polynomial_l71_71218

theorem factor_polynomial (m : ℤ) : 
  (∃ A B C D E F : ℤ, (λ x y, (A * x + B * y + C) * (D * x + E * y + F) = x^2 + 4 * x * y + x + m * y - 2 * m)) ↔ (m = 0) :=
by sorry

end factor_polynomial_l71_71218


namespace area_of_triangle_ABC_l71_71225

noncomputable def vectorAB : ℝ × ℝ := (Real.cos (23 * Real.pi / 180), Real.cos (67 * Real.pi / 180))
noncomputable def vectorBC : ℝ × ℝ := (2 * Real.cos (68 * Real.pi / 180), 2 * Real.cos (22 * Real.pi / 180))

def magnitude (v : ℝ × ℝ) := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

noncomputable def cos_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

noncomputable def sin_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  Real.sqrt (1 - (cos_angle v1 v2) ^ 2)

def triangle_area (v1 v2 : ℝ × ℝ) : ℝ :=
  0.5 * magnitude v1 * magnitude v2 * sin_angle v1 v2

theorem area_of_triangle_ABC : 
  triangle_area (-vectorAB) vectorBC = Real.sqrt 2 / 2 := by
  sorry

end area_of_triangle_ABC_l71_71225


namespace parabola_coefficients_l71_71414

theorem parabola_coefficients :
  ∃ a b c : ℝ, 
    (∀ x : ℝ, (a * (x - 3)^2 + 2 = 0 → (x = 1) ∧ (a * (1 - 3)^2 + 2 = 0))
    ∧ (a = -1/2 ∧ b = 3 ∧ c = -5/2)) 
    ∧ (∀ x : ℝ, a * x^2 + b * x + c = - 1 / 2 * x^2 + 3 * x - 5 / 2) :=
sorry

end parabola_coefficients_l71_71414


namespace number_of_zebras_l71_71329

theorem number_of_zebras : 
  ∃ Z : ℕ, 
  let heads := 30 + Z + 8 + 12 in
  let feet := 2 * 30 + 4 * Z + 4 * 8 + 2 * 12 in
  heads = feet - 132 ∧ Z = 22 :=
by
  let heads := 30 + Z + 8 + 12
  let feet := 2 * 30 + 4 * Z + 4 * 8 + 2 * 12
  sorry

end number_of_zebras_l71_71329


namespace sum_of_roots_eq_l71_71058

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l71_71058


namespace trigonometric_identity_l71_71987

variable {α : Type*} [real_field α]

theorem trigonometric_identity (α : α) (m : α) (h : tan α = m) :
  (3 * sin α + sin (3 * α)) / (3 * cos α + cos (3 * α)) = m / 2 * (m ^ 2 + 3) :=
by
  sorry

end trigonometric_identity_l71_71987


namespace bao_interest_l71_71908

noncomputable def initial_amount : ℝ := 1000
noncomputable def interest_rate : ℝ := 0.05
noncomputable def periods : ℕ := 6
noncomputable def final_amount : ℝ := initial_amount * (1 + interest_rate) ^ periods
noncomputable def interest_earned : ℝ := final_amount - initial_amount

theorem bao_interest :
  interest_earned = 340.095 := by
  sorry

end bao_interest_l71_71908


namespace sum_of_roots_l71_71103

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l71_71103


namespace sum_of_roots_eq_14_l71_71065

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l71_71065


namespace sum_of_roots_l71_71014

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l71_71014


namespace scientific_notation_of_34_million_l71_71533

theorem scientific_notation_of_34_million :
  34_000_000 = 3.4 * 10^7 := 
by
  sorry

end scientific_notation_of_34_million_l71_71533


namespace tip_percentage_is_20_l71_71870

theorem tip_percentage_is_20 (total_spent price_before_tax_and_tip : ℝ) (sales_tax_rate : ℝ) (h1 : total_spent = 158.40) (h2 : price_before_tax_and_tip = 120) (h3 : sales_tax_rate = 0.10) :
  ((total_spent - (price_before_tax_and_tip * (1 + sales_tax_rate))) / (price_before_tax_and_tip * (1 + sales_tax_rate))) * 100 = 20 :=
by
  sorry

end tip_percentage_is_20_l71_71870


namespace incorrect_conclusions_l71_71200

theorem incorrect_conclusions (p q : ℝ) :
  (¬ ∀ x, x|x| + p*x + q = 0 → ¬ (num_real_roots x = 3)) ∧
  (¬ (∃ x, x|x| + p*x + q = 0) → false) ∧
  (¬ (x : ℝ) ((p^2 - 4*q < 0) → (¬∃ x, x|x| + p*x + q = 0))) ∧
  (p < 0 ∧ q > 0 → ¬ (num_real_roots (λ x, x|x| + p*x + q) = 3)) :=
begin
  sorry,
end

end incorrect_conclusions_l71_71200


namespace percentage_reduction_is_58_perc_l71_71131

-- Define the conditions
def initial_price (P : ℝ) : ℝ := P
def discount_price (P : ℝ) : ℝ := 0.7 * P
def increased_price (P : ℝ) : ℝ := 1.2 * (discount_price P)
def clearance_price (P : ℝ) : ℝ := 0.5 * (increased_price P)

-- The statement of the proof problem
theorem percentage_reduction_is_58_perc (P : ℝ) (h : P > 0) :
  (1 - (clearance_price P / initial_price P)) * 100 = 58 :=
by
  -- Proof omitted
  sorry

end percentage_reduction_is_58_perc_l71_71131


namespace max_n_l71_71639

-- Definition of the sequence a_n
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * ∑ i in Finset.range (n + 1), a i

lemma max_n_lemma : ∀ n, a n ≤ 2018 → n ≤ 7 :=
by
  assume n,
  have h₁: a(0) = 1, by rfl, -- a₀ = 1 is given
  have h₂: ∀ n, a(n + 1) = 2 * ∑ i in Finset.range (n + 1), a i, by rfl, -- recurrence relation
  
  -- Using mathematical induction to form the sequence and find the largest n
  -- ... (skipped proof steps)

  sorry
  
-- The final theorem stating that the maximum n such that a_n ≤ 2018 is 7
theorem max_n : ∃ n, a n ≤ 2018 ∧ ∀ m > n, a m > 2018 :=
  ⟨7, (by norm_num), max_n_lemma⟩

end max_n_l71_71639


namespace no_real_roots_of_quad_eq_l71_71305

theorem no_real_roots_of_quad_eq (k : ℝ) :
  ∀ x : ℝ, ¬ (x^2 - 2*x - k = 0) ↔ k < -1 :=
by sorry

end no_real_roots_of_quad_eq_l71_71305


namespace new_average_after_doubling_l71_71848

theorem new_average_after_doubling (n : ℕ) (avg : ℝ) (h_n : n = 12) (h_avg : avg = 50) :
  2 * avg = 100 :=
by
  sorry

end new_average_after_doubling_l71_71848


namespace Alice_average_speed_l71_71536

-- Define constants for the problem
def distance1 : ℝ := 24
def time1 : ℝ := 3
def distance2 : ℝ := 36
def time2 : ℝ := 3

-- Define the total distance and total time
def total_distance : ℝ := distance1 + distance2
def total_time : ℝ := time1 + time2

-- Define the average speed calculation
def average_speed : ℝ := total_distance / total_time

-- The theorem stating that the average speed of the entire journey is 10 miles per hour
theorem Alice_average_speed : average_speed = 10 := by
  sorry

end Alice_average_speed_l71_71536


namespace part_a_part_b_l71_71942

theorem part_a (a : ℕ) : ¬ (∃ k : ℕ, k^2 = ( ((a^2 - 3)^3 + 1)^a - 1)) :=
sorry

theorem part_b (a : ℕ) : ¬ (∃ k : ℕ, k^2 = ( ((a^2 - 3)^3 + 1)^(a + 1) - 1)) :=
sorry

end part_a_part_b_l71_71942


namespace find_side_BC_l71_71279

noncomputable def find_BC : ℝ :=
    let AB := 1
    let AC := 2
    let cosB_plus_sinC := 1
    sqrt (109 + 6 * real.sqrt 21) / 5

theorem find_side_BC :
  ∀ (α β γ : ℝ),
    α = 1 →
    β = 2 →
    γ = 1 →
    find_BC = (2 * real.sqrt 21 + 3) / 5 :=
by
  intros α β γ h_AB h_AC h_trig
  simp [find_BC, h_AB, h_AC, h_trig]
  sorry

end find_side_BC_l71_71279


namespace find_a_if_max_f_l71_71269

def f (a x : ℝ) : ℝ :=
  1 - (1/2) * Real.cos (2 * x) + a * Real.sin (x / 2) * Real.cos (x / 2)

theorem find_a_if_max_f (a : ℝ) : (∃ x, f a x = 3) → (a = 3 ∨ a = -3) :=
  sorry

end find_a_if_max_f_l71_71269


namespace solve_for_x_l71_71584

theorem solve_for_x :
    ∀ x : ℝ, 13 + real.sqrt (-4 + x - 3 * 3) = 14 → x = 14 := by
  intro x
  intro h
  sorry

end solve_for_x_l71_71584


namespace geometric_seq_product_l71_71686

theorem geometric_seq_product
  (a : ℕ → ℝ)
  (h_geom : ∀ n, a n = a 1 * (r ^ n))
  (h1 : a 1 * a 99 = 16)
  (eq_root1 : a 1 = 2 + √4)
  (eq_root2 : a 99 = 2 - √4) :
  a 20 * a 50 * a 80 = 64 :=
sorry

end geometric_seq_product_l71_71686


namespace isosceles_triangle_angles_l71_71764

theorem isosceles_triangle_angles 
  (α β γ : ℝ) 
  (h_isosceles : α = β ∨ α = γ) 
  (h_sine_relation : sin α ^ 2 + sin β ^ 2 = sin γ) : 
  (α = 45 ∧ β = 45 ∧ γ = 90) ∨ (α ≈ 11.95 ∧ γ ≈ 11.95 ∧ β ≈ 156.1) :=
by
  sorry

end isosceles_triangle_angles_l71_71764


namespace circumcircles_coincide_l71_71348

-- Definition of a triangle
structure Triangle where
  A B C : Point

-- Definition of a point
structure Point where
  x y : Real

open Point

-- Hypotheses for the problem
def is_angle_bisector (A B C L : Point) : Prop := 
  -- This expresses that CL is the angle bisector of ∠ACB.
  sorry

def is_perpendicular_bisector (A C K : Point) : Prop :=
  -- This expresses that the line through K is the perpendicular bisector of AC.
  sorry

def is_circumcircle (A B C : Point) : Prop :=
  -- This expresses that A, B, and C lie on a circumcircle.
  sorry

-- Main theorem statement
theorem circumcircles_coincide :
  ∀ (A B C L K : Point),
    Triangle A B C →
    is_angle_bisector A B C L →
    is_perpendicular_bisector A C K →
    (circumcircle A B C ∩ circumcircle K) ≠ ∅ :=
by
  intros A B C L K
  sorry

end circumcircles_coincide_l71_71348


namespace C2_parametric_equation_AB_distance_l71_71680

-- Definition of the parametric equation of curve C1
def C1 (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 + 2 * Real.sin α)

-- Definition of the parametric equation of curve C2
def C2 (α : ℝ) : ℝ × ℝ := (4 * Real.cos α, 4 + 4 * Real.sin α)

-- Condition that P satisfies 2OP = OM
def point_P_on_C2 (P : ℝ × ℝ) : Prop :=
  ∃ M : ℝ × ℝ, M = (P.1 / 2, P.2 / 2) ∧ (M = C1 (Real.arctan (P.2 - 2) / 2))

-- Lean statement for the proof problem
theorem C2_parametric_equation :
  ∃ α : ℝ, point_P_on_C2 (C2 α) :=
sorry

-- Lean statement for the distance AB
def intersection_distance (θ : ℝ) : ℝ :=
  let ρ1 := 4 * Real.sin θ
  let ρ2 := 8 * Real.sin θ
  (ρ2 - ρ1).abs

theorem AB_distance :
  intersection_distance (π / 3) = 2 * Real.sqrt 3 :=
sorry

end C2_parametric_equation_AB_distance_l71_71680


namespace correct_statement_of_abs_l71_71112

theorem correct_statement_of_abs (r : ℚ) :
  ¬ (∀ r : ℚ, abs r > 0) ∧
  ¬ (∀ a b : ℚ, a ≠ b → abs a ≠ abs b) ∧
  (∀ r : ℚ, abs r ≥ 0) ∧
  ¬ (∀ r : ℚ, r < 0 → abs r = -r ∧ abs r < 0 → abs r ≠ -r) :=
by
  sorry

end correct_statement_of_abs_l71_71112


namespace problem_l71_71359

theorem problem (a b c : ℕ) (h1 : 3 * a = b ^ 3) (h2 : 5 * a = c ^ 2) (d : ℕ) (hd : d = 1) (h3 : a % (d^6) = 0) :
  (a % 3 = 0) ∧ (a % 5 = 0) ∧ (∀ p : ℕ, (nat.prime p → p ∣ a → p = 3 ∨ p = 5)) ∧ a = 1125 :=
by
  sorry

end problem_l71_71359


namespace expected_coincidences_l71_71489

-- Definitions for the given conditions
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8

def prob_correct (correct : ℕ) : ℚ := correct / num_questions
def prob_incorrect (correct : ℕ) : ℚ := 1 - prob_correct correct 

def prob_vasya_correct := prob_correct vasya_correct
def prob_vasya_incorrect := prob_incorrect vasya_correct
def prob_misha_correct := prob_correct misha_correct
def prob_misha_incorrect := prob_incorrect misha_correct

-- Probability that both guessed correctly or incorrectly
def prob_both_correct_or_incorrect : ℚ := 
  (prob_vasya_correct * prob_misha_correct) + (prob_vasya_incorrect * prob_misha_incorrect)

-- Expected value for one question being a coincidence
def expected_I_k : ℚ := prob_both_correct_or_incorrect

-- Definition of the total number of coincidences
def total_coincidences : ℚ := num_questions * expected_I_k

-- Proof statement
theorem expected_coincidences : 
  total_coincidences = 10.8 := by
  -- calculation for the expected number
  sorry

end expected_coincidences_l71_71489


namespace parity_of_F_solve_inequality_l71_71633

noncomputable def f (x : ℝ) : ℝ := Real.logBase (1/2) x

noncomputable def F (x : ℝ) : ℝ := f (x + 1) + f (1 - x)

theorem parity_of_F :
  (∀ x : ℝ, F (-x) = F x) :=
sorry

theorem solve_inequality :
  (∀ x : ℝ, abs (F x) ≤ 1 ↔ -Real.sqrt 2 / 2 ≤ x ∧ x ≤ Real.sqrt 2 / 2) :=
sorry

end parity_of_F_solve_inequality_l71_71633


namespace sum_of_roots_eq_l71_71057

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l71_71057


namespace general_term_correct_sum_reciprocal_less_than_two_l71_71617

def sequence (n : ℕ) : ℝ :=
  if n = 0 then 1 
  else let a : ℕ → ℝ := fun n => if n = 0 then 1 else (5 * a (n-1) + Real.sqrt (9 * (a (n-1))^2 + 16)) / 4
  in a n

noncomputable def a (n : ℕ) : ℝ :=
  (2 / 3) * (2^n - 1 / 2^n)

theorem general_term_correct (n : ℕ) (h : n > 0) : sequence n = a n :=
sorry

theorem sum_reciprocal_less_than_two (n : ℕ) (h : n > 0) : (∑ i in Finset.range n, 1 / sequence i.succ) < 2 :=
sorry

end general_term_correct_sum_reciprocal_less_than_two_l71_71617


namespace solve_system_l71_71754

theorem solve_system :
  ∃ x y : ℝ, ((0.5 * log 2 x - log 2 y = 0) ∧ (x^2 - 2 * y^2 = 8)) ∧ 
             ((x = 4) ∧ (y = 2) ∨ (x = 4) ∧ (y = -2)) :=
by
  sorry

end solve_system_l71_71754


namespace max_volume_of_pyramid_PABC_l71_71237

noncomputable def max_pyramid_volume (PA PB AB BC CA : ℝ) (hPA : PA = 3) (hPB : PB = 3) 
(hAB : AB = 2) (hBC : BC = 2) (hCA : CA = 2) : ℝ :=
  let D := 1 -- Midpoint of segment AB
  let PD : ℝ := Real.sqrt (PA ^ 2 - D ^ 2) -- Distance PD using Pythagorean theorem
  let S_ABC : ℝ := (Real.sqrt 3 / 4) * (AB ^ 2) -- Area of triangle ABC
  let V_PABC : ℝ := (1 / 3) * S_ABC * PD -- Volume of the pyramid
  V_PABC -- Return the volume

theorem max_volume_of_pyramid_PABC : 
  max_pyramid_volume 3 3 2 2 2  (rfl) (rfl) (rfl) (rfl) (rfl) = (2 * Real.sqrt 6) / 3 :=
by
  sorry

end max_volume_of_pyramid_PABC_l71_71237


namespace difference_of_squares_not_2018_l71_71185

theorem difference_of_squares_not_2018 (a b : ℕ) : a^2 - b^2 ≠ 2018 :=
by
  sorry

end difference_of_squares_not_2018_l71_71185


namespace polynomial_minimal_degree_rational_coeffs_l71_71961

theorem polynomial_minimal_degree_rational_coeffs :
  ∃ p : polynomial ℚ, p.leadingCoeff = 1 ∧
  (p.eval (2 + real.sqrt 5) = 0) ∧
  (p.eval (2 - real.sqrt 5) = 0) ∧
  (p.eval (3 + real.sqrt 7) = 0) ∧
  (p.eval (3 - real.sqrt 7) = 0) ∧
  (p = polynomial.C (2 : ℚ) + polynomial.Monic.compose 
       [polynomial.X^4, -10*polynomial.X^3, 21*polynomial.X^2, 16*polynomial.X, polynomial.C 2]) :=
begin
  sorry
end

end polynomial_minimal_degree_rational_coeffs_l71_71961


namespace select_3_people_in_5x5_matrix_l71_71666

theorem select_3_people_in_5x5_matrix :
  let n := 5
  let k := 3
  choose n k * k! == 600 :=
by
  let n := 5
  let k := 3
  have h1 : choose n k = Nat.choose n k := rfl
  have h2 : k! = Nat.factorial k := rfl
  have h3 : Nat.choose 5 3 = 10 := rfl
  have h4 : Nat.factorial 3 = 6 := rfl
  calc
    Nat.choose 5 3 * 3! = 10 * 6 : by rw [h3, h4]
                 ... = 60 := by rfl
                 ... = 600 := by sorry

end select_3_people_in_5x5_matrix_l71_71666


namespace max_value_of_expression_l71_71375

theorem max_value_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h_sum : x + y + z = 1) :
  x^4 * y^3 * z^2 ≤ 1024 / 14348907 :=
sorry

end max_value_of_expression_l71_71375


namespace total_interest_received_l71_71879

def principal_B : ℝ := 5000
def time_B : ℝ := 2
def principal_C : ℝ := 3000
def time_C : ℝ := 4
def rate : ℝ := 10
def SI (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem total_interest_received: 
  SI principal_B rate time_B + SI principal_C rate time_C = 2200 :=
by
  sorry

end total_interest_received_l71_71879


namespace circle_equation_l71_71136

/-- Define the point P and center C --/
def P := (2, 5)
def C := (8, -3)

/-- Define the radius as the distance between P and C --/
def radius : ℝ := Real.sqrt ((2 - 8)^2 + (5 + 3)^2)

/-- Prove the equation of the circle --/
theorem circle_equation :
  radius = 10 ∧
  (∀ x y : ℝ, (x - 8) ^ 2 + (y + 3) ^ 2 = 100 ↔ (x, y) ∈ set_of (λ (p : ℝ × ℝ), (p.1 - 8) ^ 2 + (p.2 + 3) ^ 2 = 100)) :=
by sorry

end circle_equation_l71_71136


namespace expected_coincidence_proof_l71_71494

noncomputable def expected_coincidences (total_questions : ℕ) (vasya_correct : ℕ) (misha_correct : ℕ) : ℝ :=
  let vasya_probability := vasya_correct / total_questions
  let misha_probability := misha_correct / total_questions
  let both_correct_probability := vasya_probability * misha_probability
  let both_incorrect_probability := (1 - vasya_probability) * (1 - misha_probability)
  let coincidence_probability := both_correct_probability + both_incorrect_probability
  total_questions * coincidence_probability

theorem expected_coincidence_proof : 
  expected_coincidences 20 6 8 = 10.8 :=
by {
  let total_questions := 20
  let vasya_correct := 6
  let misha_correct := 8
  
  let vasya_probability := 0.3
  let misha_probability := 0.4
  
  let both_correct_probability := vasya_probability * misha_probability
  let both_incorrect_probability := (1 - vasya_probability) * (1 - misha_probability)
  let coincidence_probability := both_correct_probability + both_incorrect_probability
  let expected := total_questions * coincidence_probability
  
  have h1 : vasya_probability = 6 / 20 := by sorry
  have h2 : misha_probability = 8 / 20 := by sorry
  have h3 : both_correct_probability = 0.3 * 0.4 := by sorry
  have h4 : both_incorrect_probability = 0.7 * 0.6 := by sorry
  have h5 : coincidence_probability = 0.54 := by sorry
  have h6 : total_questions * coincidence_probability = 20 * 0.54 := by sorry
  have h7 : 20 * 0.54 = 10.8 := by sorry

  sorry
}

end expected_coincidence_proof_l71_71494


namespace graph_intersects_x_axis_l71_71564

def transformed_logarithmic_function (x : ℝ) : ℝ := 2 * log (x - 1) / log 2

theorem graph_intersects_x_axis :
  ∃ x : ℝ, x > 1 ∧ transformed_logarithmic_function x = 0 :=
by
  use 2
  split
  · norm_num
  · have h : log (2 - 1) / log 2 = 0,
    { norm_num },
    rw [transformed_logarithmic_function, h],
    ring


end graph_intersects_x_axis_l71_71564


namespace concyclic_O1_O2_A_N_l71_71235

open EuclideanGeometry

-- Noncomputable to handle constructs involving geometric objects and proofs
noncomputable def midpoint_arc (A B C : Point) : Point := sorry -- Placeholder.

theorem concyclic_O1_O2_A_N
  (A B C: Point)
  (h1: ¬(is_isosceles (Triangle A B C)))
  (circumcircle: Circle)
  (h2: passesThrough circumcircle A ∧ passesThrough circumcircle B ∧ passesThrough circumcircle C)
  (N := midpoint_arc A B C)
  (h3: N ∈ circumcircle)
  (M : Point)
  (h4: is_on_angle_bisector M (angle A B C))
  (O1 : Point)
  (h5: is_circumcenter O1 (Triangle A B M))
  (O2: Point)
  (h6: is_circumcenter O2 (Triangle A C M)) :
  is_concyclic O1 O2 A N := 
sorry -- Proof would be handled here.

end concyclic_O1_O2_A_N_l71_71235


namespace downstream_speed_l71_71884

theorem downstream_speed (v : ℝ) (stream_speed : ℝ) (upstream_speed : ℝ) 
  (h1 : upstream_speed = 8) (h2 : stream_speed = 2) (h3 : upstream_speed = v - stream_speed) : 
  v + stream_speed = 12 :=
by
  have h4 : v = 10 := by linarith [h1, h2, h3]
  have h5 : 10 + 2 = 12 := by norm_num
  rw [h4, h2]
  exact h5

end downstream_speed_l71_71884


namespace number_ways_to_pave_1x10_block_l71_71216

-- Define the sequence a_n based on the given recurrence relation and initial conditions.
def a : ℕ → ℕ
| 0     := 0  -- Adding base case for n = 0 which is not needed but useful for consistency
| 1     := 1
| 2     := 2
| 3     := 3
| 4     := 6
| (n+1) := a n + a (n-1) + if h : n ≥ 3 then a (n-3) else 0

theorem number_ways_to_pave_1x10_block : a 10 = 169 :=
sorry

end number_ways_to_pave_1x10_block_l71_71216


namespace sum_of_roots_eq_14_l71_71094

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l71_71094


namespace minimum_cubes_needed_l71_71874

noncomputable def minimum_cubes : ℕ :=
  4

theorem minimum_cubes_needed (unit_cubes : set (ℕ × ℕ × ℕ))
  (h1 : ∀ (c ∈ unit_cubes), ∃ (d ∈ unit_cubes), adjacent_faces c d)
  (front_view : finset (ℕ × ℕ))
  (side_view : finset (ℕ × ℕ))
  (h2 : front_view = {(0,0), (0,1), (1,0), (1,1)})
  (h3 : side_view = {(0,0), (1,0), (2,0), (1,1)}) :
  ∃ (n : ℕ), n = minimum_cubes :=
sorry

end minimum_cubes_needed_l71_71874


namespace max_real_part_z_w_l71_71413

noncomputable def real_part_sum (z w : ℂ) : ℝ :=
  (z + w).re

theorem max_real_part_z_w (z w : ℂ) (hz : |z| = 1) (hw : |w| = 1) 
  (hzw: z * conj w + conj z * w = 2) : real_part_sum z w = 2 :=
by {
  sorry
}

end max_real_part_z_w_l71_71413


namespace james_received_stickers_l71_71697

theorem james_received_stickers (initial_stickers : ℕ) (total_stickers : ℕ) (received_stickers : ℕ) : 
  initial_stickers = 269 → 
  total_stickers = 423 → 
  received_stickers = total_stickers - initial_stickers → 
  received_stickers = 154 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  exact nat.sub_refl 154

#check james_received_stickers

end james_received_stickers_l71_71697


namespace fries_sold_l71_71873

theorem fries_sold (small_fries large_fries : ℕ) (h1 : small_fries = 4) (h2 : large_fries = 5 * small_fries) :
  small_fries + large_fries = 24 :=
  by
    sorry

end fries_sold_l71_71873


namespace expr_value_zero_l71_71460

def calc_expr : ℝ :=
  sqrt 5 * 5^(1 / 2) + 18 / 3 * 2 - 9^(3 / 2) + 10

theorem expr_value_zero : calc_expr = 0 := by
  sorry

end expr_value_zero_l71_71460


namespace cos_R_in_triangle_PQR_l71_71346

theorem cos_R_in_triangle_PQR
  (P Q R : ℝ) (hP : P = 90) (hQ : Real.sin Q = 3/5)
  (h_sum : P + Q + R = 180) (h_PQ_comp : P + Q = 90) :
  Real.cos R = 3 / 5 := 
sorry

end cos_R_in_triangle_PQR_l71_71346


namespace students_playing_both_l71_71388

theorem students_playing_both (total_students tennis_fraction hockey_fraction : ℕ) 
  (h1 : total_students = 600) 
  (h2 : tennis_fraction = (3/4))
  (h3 : hockey_fraction = (60/100)) :
  ∃ (students_play_tennis students_play_both : ℕ),
    students_play_tennis = tennis_fraction * total_students ∧
    students_play_both = hockey_fraction * students_play_tennis ∧
    students_play_both = 270 :=
  by 
    use 450
    use 270
    split
    · sorry
    split
    · sorry
    · sorry

end students_playing_both_l71_71388


namespace count_rectangles_no_perimeter_edge_l71_71807

def grid : ℕ × ℕ := (11, 11)
def k : ℕ := 5
def perimeter_condition (rects : list (ℕ × ℕ)) : Prop := 
  ∃ r ∈ rects, ¬(∀ edge ∈ perimeter r, edge ∈ grid_periphery grid)

theorem count_rectangles_no_perimeter_edge : 
  ∃ (n : ℕ), n = 81 ∧ ∃ (rects : list (ℕ × ℕ)), grid_divided_into_k_rectangles rects k ∧ perimeter_condition rects := 
sorry

end count_rectangles_no_perimeter_edge_l71_71807


namespace billiard_ball_weight_l71_71861

theorem billiard_ball_weight 
  (weight_with_balls : ℝ)
  (number_of_balls : ℕ)
  (weight_empty_box : ℝ)
  (one_ball_weight : ℝ) 
  (h1 : weight_with_balls = 1.82) 
  (h2 : number_of_balls = 6)
  (h3 : weight_empty_box = 0.5)
  (h4 : one_ball_weight = (weight_with_balls - weight_empty_box) / number_of_balls) :
  one_ball_weight = 0.22 :=
begin
  sorry
end

end billiard_ball_weight_l71_71861


namespace valid_paths_from_P_to_Q_l71_71877

-- Define the grid dimensions and alternate coloring conditions
def grid_width := 10
def grid_height := 8
def is_white_square (r c : ℕ) : Bool :=
  (r + c) % 2 = 1

-- Define the starting and ending squares P and Q
def P : ℕ × ℕ := (0, grid_width / 2)
def Q : ℕ × ℕ := (grid_height - 1, grid_width / 2)

-- Define a function to count valid 9-step paths from P to Q
noncomputable def count_valid_paths : ℕ :=
  -- Here the function to compute valid paths would be defined
  -- This is broad outline due to lean's framework missing specific combinatorial functions
  245

-- The theorem to state the proof problem
theorem valid_paths_from_P_to_Q : count_valid_paths = 245 :=
sorry

end valid_paths_from_P_to_Q_l71_71877


namespace can_measure_8_minutes_l71_71555

/-- Baba Yaga needs to measure exactly 8 minutes using a 5-minute hourglass and a 2-minute hourglass. 
Initially, all the sand in the 5-minute hourglass is at the bottom,
and an unknown amount of sand is in the top half of the 2-minute hourglass. -/
theorem can_measure_8_minutes :
  ∀ (G5 G2 : nat) (initial_G2_top : nat),
  (G5 = 5) ∧ (G2 = 2) ∧ (initial_G2_top ≤ G2) →
  ∃ (time_elapsed : nat), time_elapsed = 8 := 
by
  intros,
  sorry

end can_measure_8_minutes_l71_71555


namespace solve_for_N_l71_71199

theorem solve_for_N : ∃ N : ℕ, 32^4 * 4^5 = 2^N ∧ N = 30 := by
  sorry

end solve_for_N_l71_71199


namespace profit_percentage_l71_71472

variable {SP : ℝ} (h : SP > 0)

def CP : ℝ := 0.92 * SP

def profit : ℝ := SP - CP

theorem profit_percentage (h : SP > 0) : (profit / CP) * 100 ≈ 8.70 := by
  sorry

end profit_percentage_l71_71472


namespace evaluate_expression_l71_71203

theorem evaluate_expression : 3 - (-3)^(-3 : ℤ) = 82 / 27 := by
sorry

end evaluate_expression_l71_71203


namespace angle_between_a_b_is_90_degrees_l71_71363

variable {V : Type*} [InnerProductSpace ℝ V]
variable (a b : V)
variable (h_nonzero_a : a ≠ 0)
variable (h_nonzero_b : b ≠ 0)
variable (h_eq_norm : ∥a∥ = ∥b∥)
variable (h_eq_sum_norm : ∥a + b∥ = Real.sqrt 2 * ∥a∥)

theorem angle_between_a_b_is_90_degrees : real.angle a b = Real.pi / 2 :=
by
  sorry

end angle_between_a_b_is_90_degrees_l71_71363


namespace smallest_three_digit_number_with_property_l71_71963

theorem smallest_three_digit_number_with_property :
  ∃ (a : ℕ), 100 ≤ a ∧ a ≤ 999 ∧ (∃ (n : ℕ), 317 ≤ n ∧ n ≤ 999 ∧ 1001 * a + 1 = n^2) ∧ a = 183 :=
by
  sorry

end smallest_three_digit_number_with_property_l71_71963


namespace valid_integer_lattice_points_count_l71_71862

def point := (ℤ × ℤ)
def A : point := (-4, 3)
def B : point := (4, -3)

def manhattan_distance (p1 p2 : point) : ℤ :=
  abs (p2.1 - p1.1) + abs (p2.2 - p1.2)

def valid_path_length (p1 p2 : point) : Prop :=
  manhattan_distance p1 p2 ≤ 18

def does_not_cross_y_eq_x (p1 p2 : point) : Prop :=
  ∀ x y, (x, y) ∈ [(p1, p2)] → y ≠ x

def integer_lattice_points_on_path (p1 p2 : point) : ℕ := sorry

theorem valid_integer_lattice_points_count :
  integer_lattice_points_on_path A B = 112 :=
sorry

end valid_integer_lattice_points_count_l71_71862


namespace monotone_f_iff_a_range_l71_71260

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logBase 2 (3 * x + a / x - 2)

theorem monotone_f_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → y → (f x a ≤ f y a)) ↔ (-1 < a ∧ a ≤ 3) := 
by
  sorry

end monotone_f_iff_a_range_l71_71260


namespace find_range_of_x_l71_71259

noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

theorem find_range_of_x (x : ℝ) :
  (f (2 * x) > f (x - 3)) ↔ (x < -3 ∨ x > 1) :=
sorry

end find_range_of_x_l71_71259


namespace find_λ_l71_71644

def vector := (ℝ × ℝ)

noncomputable def λ_solution (a b c : vector) (λ : ℝ) : Prop :=
  ((a.1 + λ * b.1) / c.1 = a.2 / c.2)

theorem find_λ (λ : ℝ) :
  let a : vector := (1, 2)
  let b : vector := (1, 0)
  let c : vector := (3, 4)
  ((a.1 + λ * b.1) / c.1 = a.2 / c.2) → λ = 1 / 2 :=
by 
  intro h
  sorry

end find_λ_l71_71644


namespace range_of_m_l71_71248

variables (m l : ℝ)

def p : Prop := ∀ x ∈ set.Icc (-1 : ℝ) 2, m ≤ x ^ 2

def q : Prop := ∀ x : ℝ, x ^ 2 + m * x + l > 0

theorem range_of_m (h : p m ∧ q m l) : -2 < m ∧ m ≤ 1 :=
sorry

end range_of_m_l71_71248


namespace sum_of_interior_angles_of_convex_polyhedron_l71_71508

theorem sum_of_interior_angles_of_convex_polyhedron (c e l : ℕ) (h : c - e + l = 2) :
  let sum_of_angles := (c - 2) * 360 in
  sum_of_angles = 360 * (c - 2) := by
  sorry

end sum_of_interior_angles_of_convex_polyhedron_l71_71508


namespace EF_perp_O1O2_l71_71691

variables {A B C D F E O1 O2 : Type}
variables [trapezoid ABCD] [parallel AB CD] [is_point_F_on_AB F] [CF_eq_DF CF DF]
          [intersection_AC_BD E] 
          [circumcenter_ADF O1] [circumcenter_BCF O2]

theorem EF_perp_O1O2 : EF ⟂ O1O2 :=
sorry -- proof to be done

end EF_perp_O1O2_l71_71691


namespace find_side_y_l71_71938

noncomputable def side_length_y : ℝ :=
  let AB := 10 / Real.sqrt 2
  let AD := 10
  let CD := AD / 2
  CD * Real.sqrt 3

theorem find_side_y : side_length_y = 5 * Real.sqrt 3 := by
  let AB : ℝ := 10 / Real.sqrt 2
  let AD : ℝ := 10
  let CD : ℝ := AD / 2
  have h1 : CD * Real.sqrt 3 = 5 * Real.sqrt 3 := by sorry
  exact h1

end find_side_y_l71_71938


namespace remainder_2024_3047_mod_800_l71_71457

theorem remainder_2024_3047_mod_800 :
  let a := 424
      b := 647
      m := 800 in
  (a * b) % m = 728 := by
  sorry

end remainder_2024_3047_mod_800_l71_71457


namespace person_B_work_days_l71_71818

theorem person_B_work_days :
  ∃ x : ℝ, (1 / 30 + 1 / x) * 2 = 1 / 9 ∧ x = 45 := by
  -- Declare the conditions
  let workRateA := (1:ℝ) / 30
  let portionTogether := (1:ℝ) / 9
  -- Assertion
  use 45
  have workRateB := (1:ℝ) / 45
  -- Show the combined rate completes 1/9 of the work in 2 days
  have combined_rate := workRateA + workRateB
  have two_day_work := combined_rate * 2
  -- Conclude
  exact ⟨two_day_work = portionTogether, workRateB = (1:ℝ) / 45⟩

end person_B_work_days_l71_71818


namespace quadratic_no_real_roots_l71_71308

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x - k = 0)) ↔ k < -1 :=
by sorry

end quadratic_no_real_roots_l71_71308


namespace find_side_BC_l71_71278

noncomputable def find_BC : ℝ :=
    let AB := 1
    let AC := 2
    let cosB_plus_sinC := 1
    sqrt (109 + 6 * real.sqrt 21) / 5

theorem find_side_BC :
  ∀ (α β γ : ℝ),
    α = 1 →
    β = 2 →
    γ = 1 →
    find_BC = (2 * real.sqrt 21 + 3) / 5 :=
by
  intros α β γ h_AB h_AC h_trig
  simp [find_BC, h_AB, h_AC, h_trig]
  sorry

end find_side_BC_l71_71278


namespace area_of_square_EFGH_l71_71550

open Real

theorem area_of_square_EFGH (a : ℝ) (r : ℝ) (h : a = 6) (hr : r = a / 2) :
  (a + 2 * r) ^ 2 = 144 := by
  -- Given a = 6 and r = 3
  rw [h, hr]

  -- Simplify the expression
  have : a + 2 * r = 12 := by
    rw [h, hr, ←two_mul, ←mul_add, ←mul_two]
    norm_num
  rw this
  norm_num
  sorry

end area_of_square_EFGH_l71_71550


namespace mean_age_euler_family_l71_71763

theorem mean_age_euler_family :
  let girls := [8, 8, 8, 8]
  let boys := [10, 12, 12]
  let updated_ages := (girls.map (λ age => age + 1)) ++ (boys.map (λ age => age + 1))
  let mean_age := (updated_ages.sum.toReal) / updated_ages.length
  mean_age = 10.43 := 
by {
  sorry
}

end mean_age_euler_family_l71_71763


namespace obtuse_angle_range_l71_71227

noncomputable def vector_a : ℝ × ℝ := (-2, -1)
noncomputable def vector_b (λ : ℝ) : ℝ × ℝ := (λ, 1)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem obtuse_angle_range (λ : ℝ) :
  let a := vector_a
  let b := vector_b λ
  (dot_product a b < 0) → (λ > -1/2 ∧ λ ≠ 2) :=
begin
  intros h,
  sorry
end

end obtuse_angle_range_l71_71227


namespace find_z_l71_71627

theorem find_z (z : ℂ) (h : (sqrt 3 - complex.i) * z = 4 * complex.i) :
  z = -1 + sqrt 3 * complex.i :=
sorry

end find_z_l71_71627


namespace smallest_three_digit_contains_4_l71_71004

theorem smallest_three_digit_contains_4 (digits : List ℕ) (h : digits ~ [3, 4, 7]) :
  ∃ n : ℕ, n = 347 ∧ n.digits 10 = [3, 4, 7] :=
by
  use 347
  split
  · reflexivity
  · sorry

end smallest_three_digit_contains_4_l71_71004


namespace domain_of_f_l71_71826

def f (t : ℝ) : ℝ := 1 / ((t - 1)^2 - (t + 1)^2)

theorem domain_of_f : {t : ℝ | t ≠ 0} = set_of (λ t, -∞ < t ∧ t < 0 ∨ 0 < t ∧ t < ∞) :=
by
sorry

end domain_of_f_l71_71826


namespace find_range_a_l71_71251

noncomputable def even_function_monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, x ∈ set.Ioo (-1 : ℝ) (1 : ℝ) → f x = f (-x))
  ∧ (∀ x y : ℝ, x ∈ set.Ioo (0 : ℝ) (1 : ℝ) → y ∈ set.Ioo (0 : ℝ) (1 : ℝ) → x < y → f x < f y)

theorem find_range_a (f : ℝ → ℝ) (a : ℝ) (hf : even_function_monotonic f) 
  (h_condition : f (a - 2) - f (4 - a^2) < 0) : (1 < a ∧ a < 2) ∨ (2 < a ∧ a < real.sqrt 5) :=
sorry

end find_range_a_l71_71251


namespace prisoners_freedom_guaranteed_l71_71497

-- Definition of the problem strategy
def prisoners_strategy (n : ℕ) : Prop :=
  ∃ counter regular : ℕ → ℕ,
    (∀ i, i < n - 1 → regular i < 2) ∧ -- Each regular prisoner turns on the light only once
    (∃ count : ℕ, 
      counter count = 99 ∧  -- The counter counts to 99 based on the strategy
      (∀ k, k < 99 → (counter (k + 1) = counter k + 1))) -- Each turn off increases the count by one

-- The main proof statement that there is a strategy ensuring the prisoners' release
theorem prisoners_freedom_guaranteed : ∀ (n : ℕ), n = 100 →
  prisoners_strategy n :=
by {
  sorry -- The actual proof is omitted
}

end prisoners_freedom_guaranteed_l71_71497


namespace garden_perimeter_l71_71148

/-- Definition: Rectangular garden with given diagonal and area --/
variables (x y : ℝ)

def is_rectangle (x y : ℝ) := (x^2 + y^2 = 900) ∧ (x * y = 200)

/-- Theorem: The perimeter of the rectangular garden --/
theorem garden_perimeter (x y : ℝ) (h : is_rectangle x y) :
  2 * (x + y) = 20 * Real.sqrt 13 :=
by {
  sorry,
}

end garden_perimeter_l71_71148


namespace sum_of_roots_eq_14_l71_71047

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l71_71047


namespace triangle_has_two_acute_angles_l71_71899

-- Let's define the concept of an angle in a triangle and its properties.
def triangle := {a b c : ℝ // 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 180}

-- Define acute angle
def is_acute (x : ℝ) : Prop := 0 < x ∧ x < 90

-- Define right angle
def is_right (x : ℝ) : Prop := x = 90

-- Define obtuse angle
def is_obtuse (x : ℝ) : Prop := 90 < x ∧ x < 180

theorem triangle_has_two_acute_angles (T : triangle) : 
  ∃ a b c : ℝ, (T = ⟨a, b, c⟩) ∧ (¬(is_alute a ∨ is_alute b) → is_acute c) :=
by 
  sorry

end triangle_has_two_acute_angles_l71_71899


namespace junior_girl_defensemen_count_l71_71910

-- Definitions: the conditions from a)
def total_players := 50
def perc_boys := 0.60
def perc_boys_defensemen := 0.50
def perc_girls := 1 - perc_boys
def num_girls := total_players * perc_girls

def perc_junior_girls := 0.40
def num_junior_girls := num_girls * perc_junior_girls

def perc_junior_girl_defensemen := 0.50
def num_junior_girl_defensemen := num_junior_girls * perc_junior_girl_defensemen

-- Theorem to prove: the translation from c)
theorem junior_girl_defensemen_count :
  num_junior_girl_defensemen = 4 :=
by sorry

end junior_girl_defensemen_count_l71_71910


namespace slope_angle_of_line_x_eq_2_l71_71301

theorem slope_angle_of_line_x_eq_2 : ∀ (l : ℝ → Prop), (∀ x y : ℝ, l x ↔ x = 2) → (∃ θ : ℝ, θ = 90) :=
by
  intro l h
  use 90
  sorry

end slope_angle_of_line_x_eq_2_l71_71301


namespace product_increase_l71_71549

variable (x : ℤ)

theorem product_increase (h : 53 * x = 1585) : 1585 - (35 * x) = 535 :=
by sorry

end product_increase_l71_71549


namespace sum_of_roots_l71_71102

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l71_71102


namespace sum_of_roots_eq_14_l71_71073

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l71_71073


namespace sum_inequality_l71_71263

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log x - k * x + 1

theorem sum_inequality (n : ℕ) (h : n > 1) :
    (∑ i in Finset.range n \ {0, 1}, Real.log i / (i + 1)) < (n * (n - 1) / 4) :=
sorry

end sum_inequality_l71_71263


namespace integers_satisfy_equation_l71_71111

theorem integers_satisfy_equation (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  sorry

end integers_satisfy_equation_l71_71111


namespace sum_of_roots_of_quadratic_l71_71026

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l71_71026


namespace other_candidate_votes_correct_l71_71674

-- Definitions based on conditions
def total_votes : ℕ := 9000
def invalid_votes_percentage : ℝ := 0.30
def valid_votes_percentage : ℝ := 1 - invalid_votes_percentage  -- 70%
def one_candidate_votes_percentage : ℝ := 0.60

-- Total valid votes
def total_valid_votes : ℕ := (valid_votes_percentage * total_votes).toInt -- 0.70 * 9000 = 6300

-- Votes received by one candidate
def one_candidate_votes : ℕ := (one_candidate_votes_percentage * total_valid_votes).toInt -- 0.60 * 6300 = 3780

-- Votes received by the other candidate
def other_candidate_votes : ℕ := total_valid_votes - one_candidate_votes -- 6300 - 3780 = 2520

-- Proof statement
theorem other_candidate_votes_correct : other_candidate_votes = 2520 := by
  sorry

end other_candidate_votes_correct_l71_71674


namespace sufficient_condition_for_intersection_l71_71276

variables {α β γ : Type} -- placeholder types representing the planes
noncomputable def angle_between_planes (p1 p2 : Type) : ℝ := sorry -- function to represent angle between two planes
def intersection (p1 p2 : Type) : Type := sorry -- function to represent the intersection of two planes
def lines_intersect_at_single_point (a b c : Type) : Prop := sorry -- function representing lines intersecting at a single point

-- Given conditions
variables (a b c : Type)
variables (θ : ℝ)
variables (Proposition_A : θ > π / 3)
variables (Proposition_B : lines_intersect_at_single_point a b c)

-- Theorem statement
theorem sufficient_condition_for_intersection 
  (H1 : Proposition_A)
  (H2 : intersection α β = a)
  (H3 : intersection β γ = b)
  (H4 : intersection γ α = c)
  (H5 : θ = angle_between_planes α β) 
  (H6 : θ = angle_between_planes β γ)
  (H7 : θ = angle_between_planes γ α) 
  : Proposition_B := 
  sorry

end sufficient_condition_for_intersection_l71_71276


namespace fourth_number_eighth_number_l71_71455

def initial_number : ℕ := 1
def medians : list ℝ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 5]

theorem fourth_number (board : list ℕ) (H : board = [1, 3, 3, 2, 3, 2, 2, 2]) :
  board.nth 3 = some 2 :=
  by
  rw H
  simp

theorem eighth_number (board : list ℕ) (H : board = [1, 3, 3, 2, 3, 2, 2, 2]) :
  board.nth 7 = some 2 :=
  by
  rw H
  simp

end fourth_number_eighth_number_l71_71455


namespace sum_of_roots_l71_71010

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l71_71010


namespace set_inter_complement_l71_71273

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {1, 3}

-- Define set B
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem set_inter_complement :
  B ∩ (U \ A) = {2} :=
by
  sorry

end set_inter_complement_l71_71273


namespace simplify_fraction_l71_71744

theorem simplify_fraction (n : ℤ) : 
  (2 ^ (n + 4) - 3 * 2 ^ n) / (3 * 2 ^ (n + 3)) = 13 / 24 :=
by
  sorry

end simplify_fraction_l71_71744


namespace train_crossing_time_l71_71526

-- Define the length of the train, the speed of the train, and the length of the bridge
def train_length : ℕ := 140
def train_speed_kmh : ℕ := 45
def bridge_length : ℕ := 235

-- Define constants for unit conversions
def km_to_m : ℕ := 1000
def hr_to_s : ℕ := 3600

-- Calculate the speed in m/s
def train_speed_ms : ℝ :=
  (train_speed_kmh : ℝ) * (km_to_m : ℝ) / (hr_to_s : ℝ)

-- Calculate the total distance to cover (length of train + length of bridge)
def total_distance : ℕ := train_length + bridge_length

-- Calculate the time in seconds required for the train to cross the bridge
def crossing_time : ℝ :=
  (total_distance : ℝ) / train_speed_ms

-- Prove that the crossing time is 30 seconds
theorem train_crossing_time : crossing_time = 30 := by
  sorry

end train_crossing_time_l71_71526


namespace find_d_l71_71415

theorem find_d :
  ∃ d : ℝ, (∀ x y : ℝ, x^2 + 3 * y^2 + 6 * x - 18 * y + d = 0 → x = -3 ∧ y = 3) ↔ d = -27 :=
by {
  sorry
}

end find_d_l71_71415


namespace mean_of_pens_median_of_pens_l71_71535

def pen_values : List ℕ := [22, 25, 30, 40]

def mean (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

def median (lst : List ℕ) : ℚ :=
  let sorted := lst.qsort (· < ·)
  if sorted.length % 2 = 0 then
    (sorted.nth! (sorted.length / 2 - 1) + sorted.nth! (sorted.length / 2)) / 2
  else
    sorted.nth! (sorted.length / 2)

theorem mean_of_pens :
  mean pen_values = 29.25 := by
  sorry

theorem median_of_pens :
  median pen_values = 27.5 := by
  sorry

end mean_of_pens_median_of_pens_l71_71535


namespace find_a_l71_71711

theorem find_a (a b c : ℕ) (h1 : a ≥ b ∧ b ≥ c)  
  (h2 : (a:ℤ) ^ 2 - b ^ 2 - c ^ 2 + a * b = 2011) 
  (h3 : (a:ℤ) ^ 2 + 3 * b ^ 2 + 3 * c ^ 2 - 3 * a * b - 2 * a * c - 2 * b * c = -1997) : 
a = 253 := 
sorry

end find_a_l71_71711


namespace standard_equation_of_ellipse_l71_71277

noncomputable def P := (5/2, -3/2)
def A := (-2, 0)
def B := (2, 0)
def a : ℝ := Real.sqrt 10
def c : ℝ := 2
def b := Real.sqrt (a ^ 2 - c ^ 2)

theorem standard_equation_of_ellipse :
  ∃ (H : ∀ x y: ℝ, (x, y) ≠ P → (x, y) ≠ A → (x, y) ≠ B →
    foci_at_AB : (A = (-2, 0)) ∧ (B = (2, 0)),
    passes_through_P : (x = 5/2) ∧ (y = -3/2)),
    (x^2 / 10 + y^2 / 6 = 1) := 
sorry

end standard_equation_of_ellipse_l71_71277


namespace fill_bucket_time_l71_71123

-- Problem statement:
-- Prove that the time taken to fill the bucket completely is 150 seconds
-- given that two-thirds of the bucket is filled in 100 seconds.

theorem fill_bucket_time (t : ℕ) (h : (2 / 3) * t = 100) : t = 150 :=
by
  -- Proof should be here
  sorry

end fill_bucket_time_l71_71123


namespace part1_part2_l71_71239

def equilateralTriangle (side_length : ℕ) : Type :=
  sorry

def subTriangles (large_triangle : equilateralTriangle 10) : Type :=
  sorry

def numberOfSmallTriangles (triangle : Type) : ℕ :=
  sorry

def numberOfParallelograms (large_triangle : equilateralTriangle 10) (m : ℕ) : ℕ :=
  25 - m

def validShapes (large_triangle : equilateralTriangle 10) (m : ℕ) : Prop :=
  let num_small_triangles := numberOfSmallTriangles large_triangle
  let num_larger_triangles := m
  let num_parallelograms := numberOfParallelograms large_triangle m
  num_larger_triangles * 4 + num_parallelograms * 4 = num_small_triangles

theorem part1 (m : ℕ) (h_m : m = 10) :
  ¬ validShapes (equilateralTriangle.mk 10) m :=
  by sorry

theorem part2 (m : ℕ) :
  m ∈ {5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25} ↔ validShapes (equilateralTriangle.mk 10) m :=
  by sorry

end part1_part2_l71_71239


namespace sum_of_roots_of_quadratic_l71_71023

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l71_71023


namespace hyperbola_properties_l71_71233

noncomputable def hyperbola_eq (x y : ℝ) : ℝ := x^2 - y^2

theorem hyperbola_properties :
  (∃ (a : ℝ) (c : ℝ) (λ : ℝ) (m : ℝ),
  -- Given conditions:
  a ≠ 0 ∧ c = sqrt 2 * a ∧ λ = 6 ∧ hyperbola_eq 4 (- sqrt 10) = λ ∧
  (x^2 - y^2 = λ) ∧ (3^2 - m^2 = λ) ∧
  let F1 : point (ℝ × ℝ) := ⟨2 * sqrt 3, 0⟩ in
  let F2 : point (ℝ × ℝ) := ⟨-2 * sqrt 3, 0⟩ in
  ∃ (M : point (ℝ × ℝ)),
  M.1 = 3 ∧ M.2 = m ∧
  -- Prove that:
  ((F1 - M) • (F2 - M) = 0) ∧   -- Perpendicular condition
  (M lies on the circle with diameter (F1F2)) ∧
  let area : ℝ := abs ((F1.1 * M.2 - F2.1 * M.2) / 2) in
  -- Area of triangle F1MF2
  area = 6
  sorry

end hyperbola_properties_l71_71233


namespace factorize_expression_l71_71207

theorem factorize_expression (m : ℝ) : 
  4 * m^2 - 64 = 4 * (m + 4) * (m - 4) :=
sorry

end factorize_expression_l71_71207


namespace linda_lines_through_O_l71_71544

theorem linda_lines_through_O (O : ℕ × ℕ) (n : ℕ) (grid_Size : ℕ)
  (h_O : O = (0, 0)) (h_n : n = 5) (h_grid : grid_Size = n * n) : 
  ∃ (count : ℕ), count = 8 :=
by
  let valid_points := {(x, y) | 1 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 4 ∧ Nat.gcd x y = 1}
  let lines_count := valid_points.card
  exact Exists.intro lines_count sorry -- The proof should eventually show lines_count = 8

end linda_lines_through_O_l71_71544


namespace snowman_height_l71_71522

theorem snowman_height (r1 r2 r3 : ℝ) (d1 d2 d3: ℝ)
  (h_r1: r1 = 10)
  (h_r2: r2 = 20)
  (h_r3: r3 = 30)
  (h_d1: d1 = 2 * r1)
  (h_d2: d2 = 2 * r2)
  (h_d3: d3 = 2 * r3) : 
  d1 + d2 + d3 = 120 :=
by
  simp [h_r1, h_r2, h_r3, h_d1, h_d2, h_d3]
  calc 
    2 * 10 + 2 * 20 + 2 * 30 = 20 + 40 + 60 := rfl
    ... = 120 := rfl

end snowman_height_l71_71522


namespace twelve_possible_outcomes_prob_B_gt_3_given_A_eq_3_game_not_fair_l71_71399

-- Specify that the reasoning is not computationally bound.
noncomputable theory

-- Define the set of cards.
def cards := { 2, 3, 4, 4 }

-- Define the probability that a card drawn by Player B is greater than 3 given that Player A draws 3.
def prob_card_B_greater_than_3_given_A_draws_3 : ℚ := 2 / 3

-- Define the probability of Player A winning the game.
def prob_player_A_wins : ℚ := 5 / 12

-- Define the probability of Player B winning the game.
def prob_player_B_wins : ℚ := 7 / 12

-- State that there are 12 possible outcomes of cards drawn by Player A and Player B.
theorem twelve_possible_outcomes : ∃ (outcomes : finset (ℕ × ℕ)), outcomes.card = 12 := sorry

-- State that the probability that Player B's card is greater than 3 given Player A draws 3 is 2/3.
theorem prob_B_gt_3_given_A_eq_3 : (∃ (draws : finset ℕ), 
draws = {2, 4, 4} ∧ 
((draws.card : ℚ) = 2 / 3)) := sorry

-- State that the game is not fair because Player A's winning probability is less than Player B's.
theorem game_not_fair : prob_player_A_wins < prob_player_B_wins := sorry

end twelve_possible_outcomes_prob_B_gt_3_given_A_eq_3_game_not_fair_l71_71399


namespace cannot_simplify_to_AD_l71_71465

-- Define vector expressions
variables (AB DC CB AD CD MC DA BM MB : Vector3)

-- Define the expressions as predicates
def exprA := (AB - DC) - CB = AD
def exprB := AD - (CD + DC) = AD
def exprC := -(CB + MC) - (DA + BM) = AD
def exprD := -BM - DA + MB ≠ AD

theorem cannot_simplify_to_AD :
  exprA ∧ exprB ∧ exprC → exprD :=
by
  -- This statement asserts that given the conditions, exprD must hold.
  sorry

end cannot_simplify_to_AD_l71_71465


namespace min_unit_cubes_l71_71566

-- Define the conditions from step a)
def front_view : List Nat := [3, 2, 1]
def side_view : List Nat := [3]

-- Define the property that a cube shares at least one face with another
def shares_face (cubes : List Nat) : Prop :=
  ∀ i, cubes.nth i ≠ none → cubes.nth (i + 1) ≠ none ∨ cubes.nth (i - 1) ≠ none

-- Define the problem statement
theorem min_unit_cubes (cubes : List Nat) (h1 : front_view = [3, 2, 1]) (h2 : side_view = [3]) :
  cubes.length ≥ 6 ∧ shares_face cubes ∧ cubes = [3, 2, 1, 1, 1, 1] := sorry

end min_unit_cubes_l71_71566


namespace sqrt3_decimal_part_is_correct_problem_2_result_problem_3_result_l71_71378

noncomputable def sqrt3_decimal_part : ℝ := real.sqrt 3 - 1

theorem sqrt3_decimal_part_is_correct :
  sqrt3_decimal_part = real.sqrt 3 - 1 :=
by
  sorry

noncomputable def a : ℝ := real.sqrt 3 - 1
noncomputable def b : ℕ := 2  -- Integer part of sqrt 5

theorem problem_2_result :
  a + ↑b - real.sqrt 3 = 1 :=
by
  sorry

noncomputable def x : ℕ := 9
noncomputable def y : ℝ := real.sqrt 3 - 1

theorem problem_3_result :
  2 * x + (y - real.sqrt 3) ^ 2023 = 17 :=
by
  sorry

end sqrt3_decimal_part_is_correct_problem_2_result_problem_3_result_l71_71378


namespace f_nonneg_f_positive_f_zero_condition_l71_71622

noncomputable def f (A B C a b c : ℝ) : ℝ :=
  A * (a^3 + b^3 + c^3) +
  B * (a^2 * b + b^2 * c + c^2 * a + a * b^2 + b * c^2 + c * a^2) +
  C * a * b * c

theorem f_nonneg (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 ≥ 0) 
  (h2 : f A B C 1 1 0 ≥ 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c ≥ 0 :=
by sorry

theorem f_positive (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 > 0) 
  (h2 : f A B C 1 1 0 ≥ 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c > 0 :=
by sorry

theorem f_zero_condition (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 = 0) 
  (h2 : f A B C 1 1 0 > 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c ≥ 0 :=
by sorry

end f_nonneg_f_positive_f_zero_condition_l71_71622


namespace equivalence_sup_limsup_l71_71400

-- Definitions and conditions
variables {ι : Type*} [encodable ι]
variables {ξ : ι → ℝ}
variables (E : ℝ → ℝ)
variables (I : ℝ → Prop → ℝ)

-- Expectation operator, assumed finite for each ξ_n
def expectation_finite (n : ι) : Prop :=
  E (|ξ n|) < ⊤

-- Supremum condition
def sup_condition (c : ℝ) : ℝ :=
  ⨆ (n : ι), E (| ξ n |) * (if I (| ξ n | > c) then 1 else 0)

-- Limsup condition
def limsup_condition (c : ℝ) : ℝ :=
  ⨆ (n : ι), E (| ξ n |) * (if I (| ξ n | > c) then 1 else 0)

-- Theorem statement (proof not included)
theorem equivalence_sup_limsup (h : ∀ (n : ι), expectation_finite n) :
  (∀ c ∈ (set.Ioi 0), sup_condition c → 0) ↔ 
  (∀ c ∈ (set.Ioi 0), limsup_condition c → 0) :=
begin
  sorry
end

end equivalence_sup_limsup_l71_71400


namespace tangent_lines_condition_l71_71241

def f (a x : ℝ) : ℝ := 2 + a * log x
def g (a x : ℝ) : ℝ := a * x^2 + 1

theorem tangent_lines_condition (a : ℝ) :
    (∃ l1 l2 : ℝ → ℝ, l1 ≠ l2 ∧
    (∀ x : ℝ, x > 0 → l1 x = f a x → l2 x = f a x → 
    (∃ y : ℝ, y > 0 ∧ (g a) y = l1 y ∧ (g a) y = l2 y)))
    ↔ (a ∈ (Set.Ioo (-∞ : ℝ) 0) ∪ Set.Ioo (2 / (1 + log 2)) +∞) := 
sorry

end tangent_lines_condition_l71_71241


namespace sum_ratio_l71_71706

variable {α : Type _} [LinearOrderedField α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0       => a₁
| (n + 1) => (geometric_sequence a₁ q n) * q

noncomputable def sum_geometric (a₁ q : α) (n : ℕ) : α :=
  if q = 1 then a₁ * (n + 1)
  else a₁ * (1 - q^(n + 1)) / (1 - q)

theorem sum_ratio {a₁ q : α} (h : 8 * (geometric_sequence a₁ q 1) + (geometric_sequence a₁ q 4) = 0) :
  (sum_geometric a₁ q 4) / (sum_geometric a₁ q 1) = -11 :=
sorry

end sum_ratio_l71_71706


namespace sum_of_roots_of_quadratic_l71_71024

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l71_71024


namespace decimal_equivalent_of_one_quarter_l71_71477

theorem decimal_equivalent_of_one_quarter:
  ( (1:ℚ) / (4:ℚ) )^1 = 0.25 := 
sorry

end decimal_equivalent_of_one_quarter_l71_71477


namespace indefinite_integral_l71_71957

noncomputable def integral : ℝ → ℝ := λ x,
  ∫ (x : ℝ) in -∞..∞, (4 * x^2 + 3 * x + 4) / ((x^2 + 1) * (x^2 + x + 1)) 

theorem indefinite_integral :
  ∃ C : ℝ, 
  (λ x : ℝ, ∫ (t : ℝ) in 0..x, (4 * t^2 + 3 * t + 4) / ((t^2 + 1) * (t^2 + t + 1)) ) = 
  (λ x : ℝ, 3 * arctan x + (2 / real.sqrt 3) * arctan ((2 * x + 1) / real.sqrt 3) + C) :=
sorry

end indefinite_integral_l71_71957


namespace smallest_n_for_sum_of_modified_special_l71_71572

-- Define the condition for a modified special number in Lean
def is_modified_special (x : ℝ) : Prop :=
  ∃ (s : String), (∀ c ∈ s, c = '0' ∨ c = '3') ∧ x = s.to_real

theorem smallest_n_for_sum_of_modified_special :
  ∃ (n : ℕ), (∀ (a : ℕ → ℝ), (∀ i, is_modified_special (a i)) → 1 = ∑ i in range n, a i) ∧ n = 3 :=
sorry -- Proof is omitted

end smallest_n_for_sum_of_modified_special_l71_71572


namespace find_a_l71_71606

variable (f : ℤ → ℤ)
variable (a : ℤ)

axiom h1 : ∀ x : ℤ, f(x+1) = 2 * x + 4
axiom h2 : f(a) = 8

theorem find_a : a = 3 :=
by
  sorry

end find_a_l71_71606


namespace company_A_on_time_probability_bus_company_relation_l71_71852

theorem company_A_on_time_probability (a_on_time b_not_on_time : ℕ) (c_on_time d_not_on_time : ℕ) (total : ℕ) :
  a_on_time = 240 → b_not_on_time = 20 → 
  c_on_time = 210 → d_not_on_time = 30 → 
  total = 500 → 
  (a_on_time / (a_on_time + b_not_on_time)) = 12 / 13 → 
  (c_on_time / (c_on_time + d_not_on_time)) = 7 / 8 :=
by sorry

theorem bus_company_relation (a_on_time b_not_on_time : ℕ) (c_on_time d_not_on_time : ℕ) (total : ℕ) :
  a_on_time = 240 → b_not_on_time = 20 → 
  c_on_time = 210 → d_not_on_time = 30 → 
  total = 500 → 
  let k_square := (total * (a_on_time * d_not_on_time - c_on_time * b_not_on_time) ^ 2) /
                  ((a_on_time + b_not_on_time) * (c_on_time + d_not_on_time) * (a_on_time + c_on_time) * (b_on_time + d_not_on_time)) in
  k_square = 3.205 → 
  3.205 > 2.706 :=
by sorry

end company_A_on_time_probability_bus_company_relation_l71_71852


namespace P_necessary_for_Q_P_not_sufficient_for_Q_l71_71253

variable (f : ℝ → ℝ)

noncomputable def P : Prop := ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → abs ((f x₁ - f x₂) / (x₁ - x₂)) < 2017
noncomputable def Q : Prop := ∀ x : ℝ, abs (derivative f x) < 2017

theorem P_necessary_for_Q : Q f → P f := 
by
  sorry

theorem P_not_sufficient_for_Q : ¬ (P f → Q f) := 
by
  sorry

end P_necessary_for_Q_P_not_sufficient_for_Q_l71_71253


namespace find_a_l71_71709

theorem find_a (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) 
    (h3 : a ^ 2 - b ^ 2 - c ^ 2 + a * b = 2011) 
    (h4 : a ^ 2 + 3 * b ^ 2 + 3 * c ^ 2 - 3 * a * b - 2 * a * c - 2 * b * c = -1997) : 
    a = 253 :=
by 
  sorry

end find_a_l71_71709


namespace rectangle_area_l71_71823

theorem rectangle_area (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 4) :
  l * w = 8 / 9 :=
by
  sorry

end rectangle_area_l71_71823


namespace cone_inscribed_sphere_radius_l71_71520

/-- A right cone with a base radius of 15 cm and a height of 20 cm has a sphere inscribed within it.
    The radius of the sphere can be expressed as b * sqrt d - b cm where b and d are integers. 
    Prove that the value of b + d is 17. -/
theorem cone_inscribed_sphere_radius (r b d : ℝ) (hb hd : ℤ) (r_eq : r = b * (Real.sqrt d) - b) 
  (cone_base_radius : ℝ) (cone_height : ℝ) (h_base : cone_base_radius = 15) (h_height : cone_height = 20)
  (tri_ineq : 25 * r = (20 - r) * 15) :
  (b + d : ℤ) = 17 :=
sorry

end cone_inscribed_sphere_radius_l71_71520


namespace supplement_angle_greater_complement_angle_equal_unique_perpendicular_shortest_perpendicular_distance_incorrect_statement_l71_71839

theorem supplement_angle_greater (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ 180) : ¬(180 - θ > θ) → (θ ≥ 90) :=
begin
  intro h,
  sorry,
end

theorem complement_angle_equal (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ 90) : (90 - θ = 90 - θ) :=
begin
  refl,
end

theorem unique_perpendicular {P : Type} [EuclideanSpace ℝ P]
  (p : P) (l : line P) (hp : p ∉ l) : ∃! m : line P, p ∈ m ∧ m ⊥ l :=
begin
  sorry,
end

theorem shortest_perpendicular_distance {P : Type} [EuclideanSpace ℝ P]
  (p : P) (l : line P) (hp : p ∉ l) : ∀ q ∈ l, dist p l = dist p (orth_proj l p) :=
begin
  intros q hq,
  sorry,
end

theorem incorrect_statement (θ : ℝ) : (¬ θ ≥ 90) :=
begin
  sorry,
end

end supplement_angle_greater_complement_angle_equal_unique_perpendicular_shortest_perpendicular_distance_incorrect_statement_l71_71839


namespace TriangleAreaInRegularOctagon_l71_71149

-- Definitions for the regular octagon and its properties
def is_regular_octagon (O : Type) [metric_space O] (A B C D E F G H : O) (s : ℝ) : Prop :=
  dist A B = s ∧
  dist B C = s ∧
  dist C D = s ∧
  dist D E = s ∧
  dist E F = s ∧
  dist F G = s ∧
  dist G H = s ∧
  dist H A = s ∧
  ∀ θ ∈ ({A, B, C, D, E, F, G, H} : set O),
    ∃ ϕ : ℝ, (ϕ = 45) ∧ dist (circumcenter {A, B, C, D, E, F, G, H}) θ = ϕ

-- Main theorem statement
theorem TriangleAreaInRegularOctagon
  (O : Type) [metric_space O]
  (A B C D E F G H : O) (s : ℝ)
  (h_oct : is_regular_octagon O A B C D E F G H s) :
  area_ACE = 24 * (1 + real.sqrt 2) := sorry

end TriangleAreaInRegularOctagon_l71_71149


namespace slope_of_line_angle_l71_71795

def slope_angle (x : ℝ) : ℝ := -real.tan (72 * real.pi / 180)

theorem slope_of_line_angle :
  ∃ θ : ℝ, slope_angle θ = real.tan (108 * real.pi / 180) :=
begin
  use 72 * real.pi / 180,
  sorry
end

end slope_of_line_angle_l71_71795


namespace expected_coincidences_l71_71481

/-- Given conditions for the test -/
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8
def vasya_prob_correct : ℝ := 6 / 20
def misha_prob_correct : ℝ := 8 / 20
def coincidence_prob : ℝ :=
  (vasya_prob_correct * misha_prob_correct) + (1 - vasya_prob_correct) * (1 - misha_prob_correct)

/-- Expected number of coincidences -/
theorem expected_coincidences :
  20 * coincidence_prob = 10.8 :=
by {
  -- vasya_prob_correct = 0.3
  -- misha_prob_correct = 0.4
  -- probability of coincidence = 0.3 * 0.4 + 0.7 * 0.6 = 0.54
  -- expected number of coincidences = 20 * 0.54 = 10.8
  sorry
}

end expected_coincidences_l71_71481


namespace solution_l71_71972

variable {P Q R S : ℤ}

def problem_conditions : Prop :=
  P + Q + R + S = 100 ∧
  P + 5 = Q - 5 ∧
  P + 5 = 2 * R ∧
  P + 5 = S / 2

theorem solution :
  problem_conditions →
  P * Q * R * S = 1509400000 / 6561 :=
by
  sorry

end solution_l71_71972


namespace equilateral_triangle_division_exists_l71_71351

theorem equilateral_triangle_division_exists :
  ∃ (P : Finset (Finset Point)), (∀ T : Finset Point, valid_triangle T) ∧ 
  size P = 1000000 ∧ (∀ l : Line, ∃ q : ℕ, q ≤ 40 ∧ 
  ∀ p ∈ P, p ∩ l ≠ ∅ → p ∈ q) :=
sorry

end equilateral_triangle_division_exists_l71_71351


namespace vasya_misha_expected_coincidences_l71_71485

noncomputable def expected_coincidences (n : ℕ) (pA pB : ℝ) : ℝ :=
  n * ((pA * pB) + ((1 - pA) * (1 - pB)))

theorem vasya_misha_expected_coincidences :
  expected_coincidences 20 (6 / 20) (8 / 20) = 10.8 :=
by
  -- Test definition and expected output
  let n := 20
  let pA := 6 / 20
  let pB := 8 / 20
  have h : expected_coincidences n pA pB =  20 * ((pA * pB) + ((1 - pA) * (1 - pB))) := rfl
  rw h
  sorry

end vasya_misha_expected_coincidences_l71_71485


namespace sum_of_roots_eq_l71_71064

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l71_71064


namespace greatest_possible_x_l71_71429

theorem greatest_possible_x (x : ℕ) : nat.lcm (nat.lcm x 10) 14 = 70 → x = 70 :=
by
  sorry

end greatest_possible_x_l71_71429


namespace length_B_l71_71597

-- Define points on the coordinate plane and their respective circle radii
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (4, 0)
def D : ℝ × ℝ := (5, 0)
def rA := 2
def rB := 2
def rC := 3
def rD := 3

-- Define the conditions
def AB := 3  -- Distance between A and B
def AC := 4  -- Distance between A and C
def AD := 5  -- Distance between A and D

-- The given intersection points outside the respective circles
def B' := (∀ x y, x^2 + y^2 = rA^2 ∧ (x - 4)^2 + y^2 = rC^2 ∧ x > 2 ∧ x < 4)
def D' := (∀ x y, x^2 + y^2 = rA^2 ∧ (x - 5)^2 + y^2 = rD^2 ∧ x > 3 ∧ x < 5)

-- Proving the length B'D' approximately equals 0.8
theorem length_B'D'_is_0.8 : 
  (∃ (x1 y1 x2 y2 : ℝ), x1^2 + y1^2 = rA^2 ∧ (x1 - 4)^2 + y1^2 = rC^2 ∧ 
    x1 > 2 ∧ x1 < 4 ∧ 
    x2^2 + y2^2 = rA^2 ∧ (x2 - 5)^2 + y2^2 = rD^2 ∧ 
    x2 > 3 ∧ x2 < 5 ∧ 
    real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 0.8) :=
  sorry

end length_B_l71_71597


namespace union_of_sets_l71_71997

def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B : Set ℝ := {x | x > 0}
def C : Set ℝ := {x | x > -2}

theorem union_of_sets (A B : Set ℝ) : (A ∪ B) = C :=
  sorry

end union_of_sets_l71_71997


namespace find_three_digit_number_l71_71208

-- Define the function that calculates the total number of digits required
def total_digits (x : ℕ) : ℕ :=
  (if x >= 1 then 9 else 0) +
  (if x >= 10 then 90 * 2 else 0) +
  (if x >= 100 then 3 * (x - 99) else 0)

theorem find_three_digit_number : ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧ 2 * x = total_digits x := by
  sorry

end find_three_digit_number_l71_71208


namespace set_intersection_l71_71998

theorem set_intersection (A B : set ℝ) :
  (A = {x | ∃ y : ℝ, y = 1 / x}) →
  (B = {y | ∃ x : ℝ, y = real.log x}) →
  (A ∩ B = {x | x > 0}) :=
by
  intros hA hB
  rw [hA, hB]
  sorry

end set_intersection_l71_71998


namespace combined_girls_avg_l71_71539

noncomputable def centralHS_boys_avg := 68
noncomputable def deltaHS_boys_avg := 78
noncomputable def combined_boys_avg := 74
noncomputable def centralHS_girls_avg := 72
noncomputable def deltaHS_girls_avg := 85
noncomputable def centralHS_combined_avg := 70
noncomputable def deltaHS_combined_avg := 80

theorem combined_girls_avg (C c D d : ℝ) 
  (h1 : (68 * C + 72 * c) / (C + c) = 70)
  (h2 : (78 * D + 85 * d) / (D + d) = 80)
  (h3 : (68 * C + 78 * D) / (C + D) = 74) :
  (3/7 * 72 + 4/7 * 85) = 79 := 
by 
  sorry

end combined_girls_avg_l71_71539


namespace company_A_on_time_probability_bus_company_relation_l71_71851

theorem company_A_on_time_probability (a_on_time b_not_on_time : ℕ) (c_on_time d_not_on_time : ℕ) (total : ℕ) :
  a_on_time = 240 → b_not_on_time = 20 → 
  c_on_time = 210 → d_not_on_time = 30 → 
  total = 500 → 
  (a_on_time / (a_on_time + b_not_on_time)) = 12 / 13 → 
  (c_on_time / (c_on_time + d_not_on_time)) = 7 / 8 :=
by sorry

theorem bus_company_relation (a_on_time b_not_on_time : ℕ) (c_on_time d_not_on_time : ℕ) (total : ℕ) :
  a_on_time = 240 → b_not_on_time = 20 → 
  c_on_time = 210 → d_not_on_time = 30 → 
  total = 500 → 
  let k_square := (total * (a_on_time * d_not_on_time - c_on_time * b_not_on_time) ^ 2) /
                  ((a_on_time + b_not_on_time) * (c_on_time + d_not_on_time) * (a_on_time + c_on_time) * (b_on_time + d_not_on_time)) in
  k_square = 3.205 → 
  3.205 > 2.706 :=
by sorry

end company_A_on_time_probability_bus_company_relation_l71_71851


namespace find_matrix_l71_71215

noncomputable def M : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![7 / 29, 5 / 29, 0], ![3 / 29, 2 / 29, 0], ![0, 0, 1]]

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![2, -5, 0], ![-3, 7, 0], ![0, 0, 1]]

def I : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 0, 0], ![0, 1, 0], ![0, 0, 1]]

theorem find_matrix : M * A = I :=
by
  sorry

end find_matrix_l71_71215


namespace Cindy_crayons_l71_71701

variable (K : ℕ) -- Karen's crayons
variable (C : ℕ) -- Cindy's crayons

-- Given conditions
def Karen_has_639_crayons : Prop := K = 639
def Karen_has_135_more_crayons_than_Cindy : Prop := K = C + 135

-- The proof problem: showing Cindy's crayons
theorem Cindy_crayons (h1 : Karen_has_639_crayons K) (h2 : Karen_has_135_more_crayons_than_Cindy K C) : C = 504 :=
by
  sorry

end Cindy_crayons_l71_71701


namespace number_of_zeros_of_f_in_0_to_10_is_11_l71_71716

noncomputable def count_zeros_in_interval (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  if b < a then 0 else
  (b - a).to_nat.succ

theorem number_of_zeros_of_f_in_0_to_10_is_11
  (f : ℝ → ℝ)
  (hf_odd : ∀ x : ℝ, f (-x) = -f x)
  (hf_periodic : ∀ x : ℝ, f (x + 2 * Real.pi) = f x)
  (h_f3_zero : f 3 = 0)
  (h_f4_zero : f 4 = 0) :
  count_zeros_in_interval f 0 10 = 11 :=
sorry

end number_of_zeros_of_f_in_0_to_10_is_11_l71_71716


namespace remainder_7_pow_2010_l71_71458

theorem remainder_7_pow_2010 :
  (7 ^ 2010) % 100 = 49 := 
by 
  sorry

end remainder_7_pow_2010_l71_71458


namespace school_total_students_l71_71799

theorem school_total_students (T G : ℕ) (h1 : 80 + G = T) (h2 : G = (80 * T) / 100) : T = 400 :=
by
  sorry

end school_total_students_l71_71799


namespace smallest_integer_no_inverse_mod_72_and_mod_45_l71_71007

theorem smallest_integer_no_inverse_mod_72_and_mod_45 : ∃ a : ℕ, a > 0 ∧ ¬ is_unit (↑a : zmod 72) ∧ ¬ is_unit (↑a : zmod 45) ∧ (∀ b : ℕ, b > 0 ∧ ¬ is_unit (↑b : zmod 72) ∧ ¬ is_unit (↑b : zmod 45) → a ≤ b) :=
sorry

end smallest_integer_no_inverse_mod_72_and_mod_45_l71_71007


namespace reflected_line_equation_l71_71424

def line_reflection_about_x_axis (x y : ℝ) : Prop :=
  x - y + 1 = 0 → y = -x - 1

theorem reflected_line_equation :
  ∀ (x y : ℝ), x - y + 1 = 0 → x + y + 1 = 0 :=
by
  intros x y h
  suffices y = -x - 1 by
    linarith
  sorry

end reflected_line_equation_l71_71424


namespace exponential_fixed_point_l71_71427

theorem exponential_fixed_point (a : ℝ) (hx₁ : a > 0) (hx₂ : a ≠ 1) : (0, 1) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, a ^ x) } := by
  sorry 

end exponential_fixed_point_l71_71427


namespace complex_in_first_quadrant_l71_71463

-- Define the conditions for m
variable (m : ℝ)
variable (h : m > 1)

-- Define the complex number expression
def complex_expression := m * (3 + complex.I) - (2 + complex.I)

-- Prove that the complex number is in the first quadrant
theorem complex_in_first_quadrant (h : m > 1) : (complex_expression m).re > 0 ∧ (complex_expression m).im > 0 :=
sorry

end complex_in_first_quadrant_l71_71463


namespace problem_statement_l71_71607

def Point : Type := ℝ × ℝ

noncomputable def H : Point := (-3, 0)
noncomputable def on_y_axis (P : Point) : Prop := P.1 = 0
noncomputable def on_positive_x_axis (Q : Point) : Prop := Q.1 > 0 
noncomputable def on_line (P Q M : Point) : Prop := ∃ t : ℝ, M = (P.1 + t * (Q.1 - P.1), P.2 + t * (Q.2 - P.2))
noncomputable def dot_product (v1 v2 : Point) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
noncomputable def vector_sub (P Q : Point) : Point := (Q.1 - P.1, Q.2 - P.2)
noncomputable def trajectory_C (x y : ℝ) : Prop := y^2 = 4 * x ∧ x > 0

theorem problem_statement (P Q M T E : Point) (x0 : ℝ) :
  on_y_axis P →
  on_positive_x_axis Q →
  on_line P Q M →
  dot_product (vector_sub H P) (vector_sub P M) = 0 →
  vector_sub P M = (-3/2 : ℝ) • vector_sub M Q →
  (trajectory_C M.1 M.2) →
  T = (-1, 0) →
  (exists (k : ℝ) (A B : Point), on_line T A ∧ on_line T B ∧ ((A.2 = k * (A.1 + 1)) ∧ (B.2 = k * (B.1 + 1))) ∧ (dot_product (vector_sub E A) (vector_sub E B) = 0) ∧ distance A B > 0 ∧ E = (x0, 0) ∧ (triangle A B E is equilateral)) →
  x0 = 11 / 3 :=
begin
  sorry
end

end problem_statement_l71_71607


namespace income_of_A_l71_71476

theorem income_of_A (x y : ℝ) 
    (ratio_income : 5 * x = y * 4)
    (ratio_expenditure : 3 * x = y * 2)
    (savings_A : 5 * x - 3 * y = 1600)
    (savings_B : 4 * x - 2 * y = 1600) : 
    5 * x = 4000 := 
by
  sorry

end income_of_A_l71_71476


namespace no_real_roots_iff_l71_71310

theorem no_real_roots_iff (k : ℝ) : (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_iff_l71_71310


namespace baron_munchausen_incorrect_l71_71909

theorem baron_munchausen_incorrect : 
  ∀ (n : ℕ) (ab : ℕ), 10 ≤ n → n ≤ 99 → 0 ≤ ab → ab ≤ 99 
  → ¬ (∃ (m : ℕ), n * 100 + ab = m * m) := 
by
  intros n ab n_lower_bound n_upper_bound ab_lower_bound ab_upper_bound
  sorry

end baron_munchausen_incorrect_l71_71909


namespace totalShortBushes_l71_71801

namespace ProofProblem

def initialShortBushes : Nat := 37
def additionalShortBushes : Nat := 20

theorem totalShortBushes :
  initialShortBushes + additionalShortBushes = 57 := by
  sorry

end ProofProblem

end totalShortBushes_l71_71801


namespace angle_bisector_CFD_l71_71854

open Real EuclideanGeometry

-- noncomputable because Lean cannot compute points in geometry without explicit constructions
noncomputable def midpoint (A B : Point) : Point :=
  EuclideanGeometry.euclidean_geometry.midpoint ℝ A B

-- Define the semicircle with center O and points C and D on it
variables {O A B C D E F : Point}

-- Conditions
def conditions :=
  SemicircleContainsPoint O A B C ∧
  SemicircleContainsPoint O A B D ∧
  TangentAtPointMeetsExtendedDiameter O C A B B ∧
  TangentAtPointMeetsExtendedDiameter O D A B A ∧
  OppositeSides O A B ∧
  IntersectionPoint (LineThrough A C) (LineThrough B D) E ∧
  PerpendicularFootFromPointToPoint (LineThrough A B) E F

-- Theorem to prove
theorem angle_bisector_CFD (h : conditions) : AngleBisector E F C F D :=
sorry

end angle_bisector_CFD_l71_71854


namespace D_cannot_have_1000_elements_l71_71373

noncomputable def D : Set ℝ := { d : ℝ | d ≠ 0 ∧ d ≠ 1 ∧ 
                                (∀ d ∈ D, 1 - (1 / d) ∈ D) ∧ 
                                (∀ d ∈ D, 1 - d ∈ D) }

theorem D_cannot_have_1000_elements : 
  ¬(cardinality D = 1000) :=
by
  sorry

end D_cannot_have_1000_elements_l71_71373


namespace count_positive_integers_l71_71196

theorem count_positive_integers (x : ℤ) : 
  (150 ≤ x^2 + 25 ∧ x^2 + 25 ≤ 300) → (0 < x) → 5 :=
by sorry

end count_positive_integers_l71_71196


namespace train_crossing_time_l71_71130

theorem train_crossing_time :
  ∀ (length : ℝ) (speed_kmh : ℝ) (speed_ms : ℝ) (time : ℝ),
  length = 160 →
  speed_kmh = 72 →
  speed_ms = speed_kmh * (1000 / 3600) →
  time = length / speed_ms →
  time = 8 :=
by
  intros length speed_kmh speed_ms time
  intros h_length h_speed h_conversion h_time
  rw [h_length, h_speed, h_conversion, h_time]
  sorry

end train_crossing_time_l71_71130


namespace sum_of_roots_eq_14_l71_71044

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l71_71044


namespace daniel_waist_size_correct_l71_71965

noncomputable def Daniel_waist_size_cm (inches_to_feet : ℝ) (feet_to_cm : ℝ) (waist_size_in_inches : ℝ) : ℝ := 
  (waist_size_in_inches * feet_to_cm) / inches_to_feet

theorem daniel_waist_size_correct :
  Daniel_waist_size_cm 12 30.5 34 = 86.4 :=
by
  -- This skips the proof for now
  sorry

end daniel_waist_size_correct_l71_71965


namespace circumcenter_condition_l71_71506

noncomputable def circumcenter_axiom (A B C O E F D M N : Point)
  (circle_inscribed_in_ABC : Circle O inscribed_in Δ ABC)
  (touches_AB_at_E : touch AB E)
  (touches_BC_at_F : touch BC F)
  (touches_AC_at_D : touch AC D)
  (AO_inter_EF_at_M : intersect AO EF M)
  (CO_inter_EF_at_N : intersect CO EF N) : Prop :=
∃ K : Point,
  orthocenter K A C = O ∧
  perpendicular (line O D) (line A C) ∧
  circumcenter Δ OMN = point_on line KD O

theorem circumcenter_condition (A B C O E F D M N : Point)
  (circle_inscribed_in_ABC : Circle O inscribed_in Δ ABC)
  (touches_AB_at_E : touch AB E)
  (touches_BC_at_F : touch BC F)
  (touches_AC_at_D : touch AC D)
  (AO_inter_EF_at_M : intersect AO EF M)
  (CO_inter_EF_at_N : intersect CO EF N) :
  circumcenter_axiom A B C O E F D M N :=
begin
  sorry
end

end circumcenter_condition_l71_71506


namespace exists_segment_with_midpoint_l71_71614

-- Define the geometrical setup and conditions
variables {A B C D A' C' : Type} [AffineSpace ℝ (affine ℝ)] (A B C D A' C' : affine ℝ)

-- Given an angle \(\angle ABC\) and a point \(D\) inside this angle
variables (h_angle : angle A B C) (h_inside : D ∈ interior (angle A B C))

-- Define the parallel lines through D parallel to AB and BC
variables (h_parallel_AB : ∃ l, line(l) ∧ parallel l (line (A -ᵥ B)))
variables (h_parallel_BC : ∃ m, line(m) ∧ parallel m (line (B -ᵥ C)))

-- Define the intersection points A' and C'
variables (A'_on_BC : A' ∈ h_parallel_AB.intersect(line (B -ᵥ C)))
variables (C'_on_AB : C' ∈ h_parallel_BC.intersect(line (A -ᵥ B)))

-- Prove the existence of the segment A'C' with midpoint D
theorem exists_segment_with_midpoint (h1 : D ∈ midpoint A' C') : 
  ∃ A' C', midpoint A' C' = D ∧ A' ∈ line (B -ᵥ C) ∧ C' ∈ line (A -ᵥ B) :=
  sorry

end exists_segment_with_midpoint_l71_71614


namespace man_rate_in_still_water_l71_71121

theorem man_rate_in_still_water (V_m V_s: ℝ) 
(h1 : V_m + V_s = 19) 
(h2 : V_m - V_s = 11) : 
V_m = 15 := 
by
  sorry

end man_rate_in_still_water_l71_71121


namespace arc_cos_difference_l71_71182

theorem arc_cos_difference :
  let α := real.arccos ((real.sqrt 6 + 1) / (2 * real.sqrt 3))
  let β := real.arccos (real.sqrt (2 / 3))
  let A := α - β
  A = -real.pi / 6 -> 
  let a := -1
  let b := 6
  abs (a - b) = 7 :=
by
  intros
  let α := real.arccos ((real.sqrt 6 + 1) / (2 * real.sqrt 3))
  let β := real.arccos (real.sqrt (2 / 3))
  let A := α - β
  have h : A = -real.pi / 6, sorry
  let a := -1
  let b := 6
  show abs (a - b) = 7, from abs_sub a b
  rw h
  exact abs_neg 1

end arc_cos_difference_l71_71182


namespace trigonometric_expression_value_l71_71624

theorem trigonometric_expression_value (α : Real) (h1 : π < α ∧ α < 3 * π / 2) (h2 : Real.sin(α) = -3 / 5) :
  (Real.tan(2 * π - α) * Real.cos(3 * π / 2 - α) * Real.cos(6 * π - α)) /
  (Real.sin(α + 3 * π / 2) * Real.cos(α + 3 * π / 2)) = -3 / 4 := 
sorry

end trigonometric_expression_value_l71_71624


namespace lateral_surface_area_of_cone_l71_71152

theorem lateral_surface_area_of_cone :
  let r := 1
  let h := 1
  let l := Real.sqrt (r^2 + h^2)
  S = π * r * l
  in S = Real.sqrt 2 * π :=
by
  sorry

end lateral_surface_area_of_cone_l71_71152


namespace ratio_of_x_to_y_l71_71312

theorem ratio_of_x_to_y (x y : ℚ) (h : (2 * x - 3 * y) / (x + 2 * y) = 5 / 4) : x / y = 22 / 3 := by
  sorry

end ratio_of_x_to_y_l71_71312


namespace sum_of_roots_of_quadratic_l71_71027

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l71_71027


namespace age_ratio_l71_71974
open Nat

theorem age_ratio (B A x : ℕ) (h1 : B - 4 = 2 * (A - 4)) 
                                (h2 : B - 8 = 3 * (A - 8)) 
                                (h3 : (B + x) / (A + x) = 3 / 2) : 
                                x = 4 :=
by
  sorry

end age_ratio_l71_71974


namespace find_a_b_range_of_a_l71_71635

-- Define the function f(x)
def f (x : ℝ) (a b : ℝ) : ℝ := x - a * Real.log x + b

-- Define the derivative of f(x)
def f' (x : ℝ) (a : ℝ) : ℝ := 1 - a / x

-- 1. Prove the values of a and b given tangent line condition
theorem find_a_b (a b : ℝ) (h1 : f' 1 a = 2) (h2 : f 1 a b = 5) : a = -1 ∧ b = 4 :=
by
  sorry

-- 2. Prove the range of values of a given the condition on f'(x)
theorem range_of_a (a : ℝ) (h : ∀ x ∈ Set.Icc 2 3, |f' x a| < 3 / x^2) : 2 ≤ a ∧ a ≤ 7 / 2 :=
by
  sorry

end find_a_b_range_of_a_l71_71635


namespace denise_spent_l71_71358

theorem denise_spent (price_simple : ℕ) (price_meat : ℕ) (price_fish : ℕ)
  (price_milk_smoothie : ℕ) (price_fruit_smoothie : ℕ) (price_special_smoothie : ℕ)
  (julio_spent_more : ℕ) :
  price_simple = 7 →
  price_meat = 11 →
  price_fish = 14 →
  price_milk_smoothie = 6 →
  price_fruit_smoothie = 7 →
  price_special_smoothie = 9 →
  julio_spent_more = 6 →
  ∃ (d_price : ℕ), (d_price = 14 ∨ d_price = 17) :=
by
  sorry

end denise_spent_l71_71358


namespace ratio_of_segments_l71_71345

variables (A B C M E F D G : Type)
variables [ordered_comm_ring ℝ] [add_comm_group G]

-- Definitions and conditions
def is_triangle (A B C : G) := true
def midpoint (A B : G) (M : G) := M = (A + B) / 2
def on_line (A B : G) (P : G) := ∃(t : ℝ), P = t • A + (1 - t) • B
def bisects_angle (A O B : G) (D : G) := true
def length_eq (A B : G) (d : ℝ) := true

-- Given conditions
def conditions :=
  is_triangle A B C ∧  -- Triangle ABC
  midpoint B C M ∧  -- M is midpoint of BC
  length_eq A B 15 ∧  -- AB = 15
  length_eq A C 18 ∧  -- AC = 18
  on_line A C E ∧  -- E is on AC
  on_line A B F ∧  -- F is on AB 
  on_line B C D ∧  -- D is on BC
  midpoint B C D ∧  -- D is the midpoint, hence BD = DC
  bisects_angle B A C D ∧  -- AD bisects ∠BAC
  ∃ x : ℝ, E = 3 • F  -- AE = 3AF

-- Goal statement to prove
theorem ratio_of_segments (EG GF : G) (h : conditions) :
  length_eq E G (7/4 * length_eq G F) := 
sorry 

end ratio_of_segments_l71_71345


namespace coefficient_x3_is_39_l71_71913

noncomputable def coefficient_x3 : ℤ :=
  let expr1 := 2 * (λ x : ℕ => x^2 - 2 * x^3 + 2 * x)
  let expr2 := 4 * (λ x : ℕ => x + 3 * x^3 - 2 * x^2 + 2 * x^5 - x^3)
  let expr3 := -7 * (λ x : ℕ => 2 + 2 * x - 5 * x^3 - x^2)
  expr1 3 + expr2 3 + expr3 3

theorem coefficient_x3_is_39 : coefficient_x3 = 39 := by
  sorry

end coefficient_x3_is_39_l71_71913


namespace minimum_value_of_F2A_F2B_over_area_triangle_l71_71619

variables {a b c : ℝ} (F1 F2 : ℝ × ℝ) (D E : ℝ × ℝ)
variables (x1 y1 x2 y2 : ℝ) (t : ℝ)

def is_focus_of_ellipse (F : ℝ × ℝ) (a b : ℝ) : Prop :=
  F.1^2 = a^2 - b^2 ∧ F.2 = 0

def vertices_of_ellipse (D E : ℝ × ℝ) (a b : ℝ) : Prop :=
  D = (0, b) ∧ E = (a, 0)

def area_triangle (S : ℝ) : Prop :=
  S = √3 / 2

def eccentricity (e : ℝ) : Prop := 
  e = 1 / 2

noncomputable def find_standard_equation : Prop :=
  a^2 = 4 ∧ b^2 = 3

theorem minimum_value_of_F2A_F2B_over_area_triangle :
  is_focus_of_ellipse F2 a b →
  vertices_of_ellipse D E a b →
  area_triangle (sqrt 3 / 2) →
  eccentricity (1 / 2) →
  (a^2 = 4 ∧ b^2 = 3 ∧
   ∃ (A B : ℝ × ℝ),
     line_through F2 A ∧ 
     line_through F2 B ∧ 
     ellipse_intersect A B x1 y1 x2 y2 →
     min_value (2 * (1 + t^2) * (9 / (3 * t^2 + 4)) / (sqrt (36 * t^2 / ((3 * t^2 + 4)^2) + 36 / (3 * t^2 + 4))) = (3/2)) :=
sorry

end minimum_value_of_F2A_F2B_over_area_triangle_l71_71619


namespace sum_of_roots_eq_14_l71_71035

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l71_71035


namespace maximum_Sn_Sm_l71_71426

noncomputable def a_n (n : ℕ) : ℤ :=
  -n^2 + 12 * n - 32

noncomputable def S_n (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), a_n i

theorem maximum_Sn_Sm (m n : ℕ) (hmn : m < n) :
  S_n n - S_n m ≤ 10 :=
sorry

end maximum_Sn_Sm_l71_71426


namespace number_of_divisors_gcd_48_180_l71_71287

theorem number_of_divisors_gcd_48_180 : 
  let gcd_48_180 := gcd 48 180
  in (gcd_48_180 = 12) → (finset.card (finset.filter (λ d, 12 % d = 0) (finset.range (12 + 1))) = 6) :=
by
  let gcd_48_180 := gcd 48 180
  have h_gcd : gcd_48_180 = 12 := sorry
  rw h_gcd
  sorry

end number_of_divisors_gcd_48_180_l71_71287


namespace mitchell_chews_54_pieces_l71_71381

theorem mitchell_chews_54_pieces (packets : ℕ) (pieces_per_packet : ℕ) (except_pieces : ℕ) (total_pieces : ℕ) :
  packets = 8 → pieces_per_packet = 7 → except_pieces = 2 → total_pieces = packets * pieces_per_packet → total_pieces - except_pieces = 54 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  have h5 : total_pieces = 56 := by linarith
  rw [←h5, h3]
  linarith

end mitchell_chews_54_pieces_l71_71381


namespace residue_calculation_l71_71557

theorem residue_calculation :
  (196 * 18 - 21 * 9 + 5) % 18 = 14 := 
by 
  sorry

end residue_calculation_l71_71557


namespace mitchell_chews_54_pieces_l71_71380

theorem mitchell_chews_54_pieces : 
  ∀ (packets pieces_not_chewed pieces_per_packet total_pieces pieces_chewed : ℕ),
    packets = 8 →
    pieces_per_packet = 7 →
    pieces_not_chewed = 2 →
    total_pieces = packets * pieces_per_packet →
    pieces_chewed = total_pieces - pieces_not_chewed →
    pieces_chewed = 54 :=
by
  intros packets pieces_not_chewed pieces_per_packet total_pieces pieces_chewed h_packet h_per_packet h_not_chewed h_total h_chewed
  rw [h_packet, h_per_packet] at h_total
  rw [h_total] at h_chewed
  simp at h_chewed
  assumption

end mitchell_chews_54_pieces_l71_71380


namespace incorrect_solution_among_four_l71_71747

theorem incorrect_solution_among_four 
  (x y : ℤ) 
  (h1 : 2 * x - 3 * y = 5) 
  (h2 : 3 * x - 2 * y = 7) : 
  ¬ ((2 * (2 * x - 3 * y) - ((-3) * (3 * x - 2 * y))) = (2 * 5 - (-3) * 7)) :=
sorry

end incorrect_solution_among_four_l71_71747


namespace min_b8_b92_l71_71891

noncomputable def is_geometric_sequence (b : ℕ → ℝ) :=
  ∃ (q : ℝ) (h : q ≠ 0), ∀ n : ℕ, b (n + 1) = q * (b n)

theorem min_b8_b92 (b : ℕ → ℝ) (h1 : is_geometric_sequence b)
  (h2 : ∏ i in finset.range 99, b (i + 1) = 2 ^ 99) :
  b 8 + b 92 = 4 :=
sorry

end min_b8_b92_l71_71891


namespace num_isosceles_triangles_with_perimeter_31_l71_71285

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ (a + b > c ∧ b + c > a ∧ a + c > b)

def is_valid_triangle_perimeter (a b c : ℕ) : Prop :=
  a + b + c = 31

def odd (n : ℕ) : Prop := n % 2 = 1

theorem num_isosceles_triangles_with_perimeter_31 : 
  ∃ n, n = 6 ∧ ∀ (a b c : ℕ), is_isosceles_triangle a b c ∧ is_valid_triangle_perimeter a b c ∧ odd c → (finset.card (finset.filter (λ ⟨a, b, c⟩, is_isosceles_triangle a b c ∧ is_valid_triangle_perimeter a b c ∧ odd c) (finset.univ : finset (ℕ × ℕ × ℕ))) = n) :=
by sorry

end num_isosceles_triangles_with_perimeter_31_l71_71285


namespace multiples_of_7_units_digit_7_less_than_200_l71_71288

theorem multiples_of_7_units_digit_7_less_than_200 : 
  { n : ℕ // n > 0 ∧ n < 200 ∧ n % 7 = 0 } → { n : ℕ // n < 200 ∧ n % 7 = 0 ∧ n % 10 = 7 }.set.card = 3 :=
by
  sorry -- Proof left as an exercise

end multiples_of_7_units_digit_7_less_than_200_l71_71288


namespace container_volume_ratio_l71_71537

theorem container_volume_ratio
  (A B : ℝ)
  (h : (5 / 6) * A = (3 / 4) * B) :
  (A / B = 9 / 10) :=
sorry

end container_volume_ratio_l71_71537


namespace third_number_is_60_l71_71421

theorem third_number_is_60 (x : ℤ) :
  (20 + 40 + x) / 3 = (10 + 80 + 15) / 3 + 5 → x = 60 :=
by
  intro h
  sorry

end third_number_is_60_l71_71421


namespace exists_vertices_no_three_form_triangle_l71_71941

theorem exists_vertices_no_three_form_triangle (n : ℕ) (h : n > 3^nat.log (3 * 1993)) :
  ∃ (vertices : set (ℕ × ℕ)) (htri : vertices.card = 1993 * n), 
  ∀ (v1 v2 v3 : ℕ × ℕ), v1 ∈ vertices → v2 ∈ vertices → v3 ∈ vertices → 
    ¬ (is_equilateral_triangle v1 v2 v3) :=
sorry

-- Assume is_equilateral_triangle is some predicate that checks if three points v1, v2, v3 form an equilateral triangle.
@[simp]
def is_equilateral_triangle (v1 v2 v3 : ℕ × ℕ) : Prop :=
  -- some implementation of the predicate based on geometric properties
  true := sorry

end exists_vertices_no_three_form_triangle_l71_71941


namespace tangent_line_m_value_l71_71431

theorem tangent_line_m_value : 
  (∀ m : ℝ, ∃ (x y : ℝ), (x = my + 2) ∧ (x + one)^2 + (y + one)^2 = 2) → 
  (m = 1 ∨ m = -7) :=
  sorry

end tangent_line_m_value_l71_71431


namespace max_product_decomposition_l71_71193

theorem max_product_decomposition : ∃ x y : ℝ, x + y = 100 ∧ x * y = 50 * 50 := by
  sorry

end max_product_decomposition_l71_71193


namespace cylinder_volume_l71_71517

theorem cylinder_volume (length width : ℝ) (h₁ h₂ : ℝ) (radius1 radius2 : ℝ) (V1 V2 : ℝ) (π : ℝ)
  (h_length : length = 12) (h_width : width = 8) 
  (circumference1 : circumference1 = length)
  (circumference2 : circumference2 = width)
  (h_radius1 : radius1 = 6 / π) (h_radius2 : radius2 = 4 / π)
  (h_height1 : h₁ = width) (h_height2 : h₂ = length)
  (h_V1 : V1 = π * radius1^2 * h₁) (h_V2 : V2 = π * radius2^2 * h₂) :
  V1 = 288 / π ∨ V2 = 192 / π :=
sorry


end cylinder_volume_l71_71517


namespace length_of_GH_l71_71006

-- Definitions and conditions
def parallel_lines (a b c d e f : ℝ) : Prop := 
  a / b = c / d ∧ a / b = e / f

def segment_length_JK := 180 -- cm
def segment_length_LM := 120 -- cm

-- Theorem statement
theorem length_of_GH
  (JK LM GH : ℝ)
  (h1 : parallel_lines JK LM GH segment_length_LM JK)
  : GH = 72 := 
sorry

end length_of_GH_l71_71006


namespace problem_solution_l71_71436

variable {R : Type*} [CommRing R]
variable (u v w : R)
variable (h1 : u + v + w = 0)
variable (h2 : u * v + v * w + w * u = -3)
variable (h3 : u * v * w = 1)

noncomputable def S : ℕ → R
| 0     := 3
| 1     := 0
| 2     := 6
| 3     := 3
| (n+4) := 3 * S n + S (n + 1)

theorem problem_solution : S u v w 9 = 246 :=
by
  sorry

end problem_solution_l71_71436


namespace sum_of_roots_l71_71107

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l71_71107


namespace coefficient_x6_in_expansion_l71_71337

open Nat

-- Define factorial function to calculate binomial coefficients
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := fact n / (fact k * fact (n - k))

-- The coefficient of x^6 in the expansion of (x - 2)^10 should be 16 * binom(10, 4)
theorem coefficient_x6_in_expansion : (∀ x y n : ℕ, (x - y) ^ 10) → (binom 10 4 * 16) = (16 * binom 10 4) :=
by
  intro x y n
  sorry

end coefficient_x6_in_expansion_l71_71337


namespace total_balloons_l71_71741

theorem total_balloons:
  ∀ (R1 R2 G1 G2 B1 B2 Y1 Y2 O1 O2: ℕ),
    R1 = 31 →
    R2 = 24 →
    G1 = 15 →
    G2 = 7 →
    B1 = 12 →
    B2 = 14 →
    Y1 = 18 →
    Y2 = 20 →
    O1 = 10 →
    O2 = 16 →
    (R1 + R2 = 55) ∧
    (G1 + G2 = 22) ∧
    (B1 + B2 = 26) ∧
    (Y1 + Y2 = 38) ∧
    (O1 + O2 = 26) :=
by
  intros
  sorry

end total_balloons_l71_71741


namespace sum_of_coeffs_l71_71720

theorem sum_of_coeffs (a : ℕ → ℤ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = ∑ i in Finset.range (Nat.succ a.length), a i * (x + 2)^i) →
  ∑ i in Finset.range (Nat.succ a.length), a i = -2 :=
by
  intros h
  -- proof skipped
  sorry

end sum_of_coeffs_l71_71720


namespace basketball_cards_price_l71_71386

theorem basketball_cards_price :
  let toys_cost := 3 * 10
  let shirts_cost := 5 * 6
  let total_cost := 70
  let basketball_cards_cost := total_cost - (toys_cost + shirts_cost)
  let packs_of_cards := 2
  (basketball_cards_cost / packs_of_cards) = 5 :=
by
  sorry

end basketball_cards_price_l71_71386


namespace sum_of_roots_eq_14_l71_71072

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l71_71072


namespace triangle_distinct_lines_l71_71325

theorem triangle_distinct_lines :
  ∀ (T : triangle),
    T.is_right ∧ T.base_angles = (45, 45) → T.distinct_lines = 7 :=
by {
  -- T represents the triangle
  -- T.is_right asserts the triangle is right-angled
  -- T.base_angles = (45, 45) asserts angles at the base are each 45 degrees
  -- T.distinct_lines counts the total number of distinct lines (altitudes, medians, and angle bisectors)
  
  sorry
}

end triangle_distinct_lines_l71_71325


namespace slope_angle_AB_is_135_degrees_l71_71243

-- Defining points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-1, 3⟩
def B : Point := ⟨3, -1⟩

-- Defining slope
def slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

-- Defining slope angle
def slope_angle (P Q : Point) : ℝ :=
  let m := slope P Q
  if m < 0 then Real.arctan (-m) + 180 else Real.arctan m

-- The theorem to prove
theorem slope_angle_AB_is_135_degrees : slope_angle A B = 135 :=
by
  sorry

end slope_angle_AB_is_135_degrees_l71_71243


namespace ratio_satisfies_cubic_l71_71660

-- Define the vertex as condition
def vertex_is_origin (O : Point) : Prop := O = ⟨0, 0⟩

-- Define the parabola equation condition
def on_parabola (P : Point) (a : ℝ) : Prop := P.y = a * P.x^2

-- Define the first quadrant condition
def in_first_quadrant (P : Point) : Prop := P.x > 0 ∧ P.y > 0

-- Define the coordinates ratio condition
def coord_ratio (x_C y_C : ℝ) (t : ℝ) : Prop := t = y_C / x_C

-- Main theorem stating that the ratio t satisfies a cubic equation
theorem ratio_satisfies_cubic (a : ℝ) (h_a : a > 0) (x_C y_C : ℝ) (C_in_first_q : in_first_quadrant ⟨x_C, y_C⟩) (on_parabola_C : on_parabola ⟨x_C, y_C⟩ a) (t : ℝ) (t_def : coord_ratio x_C y_C t):
  t^3 - 2 * t^2 - 1 = 0 :=
sorry

end ratio_satisfies_cubic_l71_71660


namespace sum_of_roots_eq_14_l71_71048

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l71_71048


namespace fourth_boy_payment_l71_71596

theorem fourth_boy_payment (a b c d : ℝ) 
  (h₁ : a = (1 / 2) * (b + c + d)) 
  (h₂ : b = (1 / 3) * (a + c + d)) 
  (h₃ : c = (1 / 4) * (a + b + d)) 
  (h₄ : a + b + c + d = 60) : 
  d = 13 := 
sorry

end fourth_boy_payment_l71_71596


namespace sum_of_roots_l71_71013

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l71_71013


namespace sufficient_condition_l71_71655

theorem sufficient_condition (a b : ℝ) (h : |a + b| > 1) : |a| + |b| > 1 := 
by sorry

end sufficient_condition_l71_71655


namespace sin_cos_sum_l71_71990

noncomputable def r (x y : ℝ) := real.sqrt (x^2 + y^2)

theorem sin_cos_sum (x y : ℝ) (h₁ : x = 3) (h₂ : y = 4) :
  real.sin (real.arctan (y / x)) + real.cos (real.arctan (y / x)) = 8 / 5 :=
by
  have hr : r x y = 5 := by
    sorry
  have hsin : real.sin (real.arctan (y / x)) = y / r x y := by
    sorry
  have hcos : real.cos (real.arctan (y / x)) = x / r x y := by
    sorry
  rw [hsin, hcos, hr]
  simp
  norm_num
  done
  sorry

end sin_cos_sum_l71_71990


namespace simplify_expression_l71_71204

/--
Given \(8 = 2^3\) and \(16 = 2^4\),
prove that the expression \(\dfrac{\sqrt[6]{2^3}}{\sqrt[4]{2^4}}\) 
simplifies to \(2^{-\frac{1}{2}}\).
-/
theorem simplify_expression : 
  (∛(2^3) : ℝ) / (∜(2^4) : ℝ) = 2^(-1/2) :=
by
  sorry

end simplify_expression_l71_71204


namespace clockwise_rotation_angle_240_l71_71937

/-
  Given a counterclockwise rotation is positive,
  and a clockwise rotation is negative,
  the angle formed by rotating a ray 240° clockwise is -240°.
-/

theorem clockwise_rotation_angle_240 :
  ∀ θ : ℝ, θ = 240 → -θ = -240 := 
begin
  intros θ h,
  rw h,
  norm_num,
end

end clockwise_rotation_angle_240_l71_71937


namespace number_of_solutions_in_interval_l71_71370

def f (x : ℝ) : ℝ := -3 * Real.sin (Real.pi * x)

theorem number_of_solutions_in_interval :
  (set.count (λ x, f (f (f x)) = f x) {x | -3 ≤ x ∧ x ≤ 3}) = 79 :=
sorry

end number_of_solutions_in_interval_l71_71370


namespace plane_Q_equation_l71_71955

theorem plane_Q_equation (a b : ℝ) (x y z : ℝ) (p : ℝ × ℝ × ℝ) :
  let Q := (2 * a + 3 * b) * x + (-3 * a + b) * y + (4 * a - 2 * b) * z - (5 * a + b),
      distance := |(2 * a + 3 * b) * 1 + (-3 * a + b) * 2 + (4 * a - 2 * b) * 3 - (5 * a + b)| /
            real.sqrt ((2 * a + 3 * b)^2 + (-3 * a + b)^2 + (4 * a - 2 * b)^2) in
  2 * x - 3 * y + 4 * z = 5 →
  3 * x + y - 2 * z = 1 →
  distance = 3 / real.sqrt 5 →
  (a, b) ≠ (0, 0) →
  Q = 6 * x - y + 10 * z - 11 :=
by
  sorry

end plane_Q_equation_l71_71955


namespace sum_of_roots_eq_14_l71_71050

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l71_71050


namespace distinct_fractional_linear_functions_count_l71_71932

theorem distinct_fractional_linear_functions_count (p : ℕ) : 
  ∃ (p' : ℕ), p' = p + 1 ∧ 
  ∀ y1 y2 y3 : ℕ, 
  y1 ≠ y2 ∧ y2 ≠ y3 ∧ y3 ≠ y1 ∧ y1 ∈ (Finset.range p ∪ {p}) ∧ y2 ∈ (Finset.range p ∪ {p}) ∧ y3 ∈ (Finset.range p ∪ {p}) → 
  (p' * p * (p - 1)) :=
sorry

end distinct_fractional_linear_functions_count_l71_71932


namespace reduced_fraction_correct_l71_71737

noncomputable def P : Polynomial ℝ := 
  2 * X^6 + 5 * X^4 - 3 * X^3 + 2 * X^2 - 12 * X - 14

noncomputable def Q : Polynomial ℝ := 
  4 * X^6 - 4 * X^4 - 6 * X^3 - 3 * X^2 + 25 * X - 28

noncomputable def gcd_PQ : Polynomial ℝ := 
  4 * X^3 + 2 * X - 14

noncomputable def P_div_gcd : Polynomial ℝ := 
  X^3 + 2 * X + 2

noncomputable def Q_div_gcd : Polynomial ℝ := 
  2 * X^3 - 3 * X + 4

theorem reduced_fraction_correct : 
  (P / Q : Fraction ℚ).num = P_div_gcd ∧ (P / Q : Fraction ℚ).denom = Q_div_gcd := by
    sorry

end reduced_fraction_correct_l71_71737


namespace investment_worth_after_two_years_l71_71676

noncomputable def investment_value : ℝ → ℝ
| initial_value := initial_value * (1 + 1/9) * (1 + 1/6)

theorem investment_worth_after_two_years :
  investment_value 64000 = 82962.96 :=
by
  have first_year_value := 64000 * (1 + (1 / 9 : ℝ))
  have second_year_value := first_year_value * (1 + (1 / 6 : ℝ))
  rw [investment_value]
  simp [first_year_value, second_year_value]
  sorry

end investment_worth_after_two_years_l71_71676


namespace OZ_length_l71_71924

-- Define the convex quadrilateral WXYZ and all the given conditions
variable {W X Y Z O : Type} [ConvexQuadrilateral W X Y Z]
variable (WY XZ OW: ℝ) (WO OZ: ℝ)
variable (area_WOX area_YOZ: ℝ) (WZ XY: ℝ)

-- Given values
axiom WY_val : WY = 10
axiom XZ_val : XZ = 15
axiom WO_val : WO = 8
axiom equal_areas : area_WOX = area_YOZ
axiom ratio_WZ_XY : WZ / XY = 5 / 3

-- Proven statement
theorem OZ_length : OZ = 24 / 5 := by
  sorry

end OZ_length_l71_71924


namespace rhombus_perimeter_l71_71151

-- Definitions based on given conditions.
def diagonal1 : ℝ := 10
def diagonal2 : ℝ := 24

-- The question translated to Lean: Prove that the perimeter is 52 inches.
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = diagonal1) (h2 : d2 = diagonal2) : 
  let leg1 := d1 / 2
  let leg2 := d2 / 2
  let hypotenuse := real.sqrt (leg1 ^ 2 + leg2 ^ 2)
  4 * hypotenuse = 52 :=
by
  sorry

end rhombus_perimeter_l71_71151


namespace final_amount_l71_71356

theorem final_amount (orig_a orig_b orig_c : ℕ) (discount_a discount_b discount_c tax_rate : ℚ)
                     (h_a : orig_a = 200) (h_b : orig_b = 300) (h_c : orig_c = 400)
                     (d_a : discount_a = 0.50) (d_b : discount_b = 0.30) (d_c : discount_c = 0.40)
                     (tax : tax_rate = 0.05) :
  let discounted_a := orig_a * (1 - discount_a),
      discounted_b := orig_b * (1 - discount_b),
      discounted_c := orig_c * (1 - discount_c),
      total_discounted := discounted_a + discounted_b + discounted_c,
      total_amount := total_discounted * (1 + tax_rate)
  in total_amount = 577.50 := 
by 
  sorry

end final_amount_l71_71356


namespace inequality_solution_minimum_value_inequality_l71_71267

section Part1

def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 2 * x + 1) + Real.sqrt (x^2 - 10 * x + 25)

theorem inequality_solution :
  {x : ℝ | f x > 6} = {x : ℝ | x < 0 ∨ x > 6} :=
by
  sorry

end Part1

section Part2

variable (a b c : ℝ) (m : ℝ := 4)

theorem minimum_value_inequality
  (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (condition : 1/a + 1/(2*b) + 1/(3*c) = 1) :
  a + 2 * b + 3 * c ≥ 9 :=
by
  sorry

end Part2

end inequality_solution_minimum_value_inequality_l71_71267


namespace proof_problem_l71_71367

theorem proof_problem
  (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (f g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * Real.log (x^2 + 1) + b * x)
  (h_g : ∀ x, g x = b * x^2 + 2 * a * x + b)
  (x1 x2 : ℝ)
  (h_roots : g x1 = 0 ∧ g x2 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ 0 ∧ x2 ≠ 0) :
  (x1 + x2 < -2) ∧ 
  (∀ λ : ℝ, (f x1 + f x2 + 3 * a - λ * b = 0) → λ > 2 * Real.log 2 + 1) :=
by
  sorry

end proof_problem_l71_71367


namespace sum_of_roots_of_equation_l71_71076

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l71_71076


namespace students_playing_both_l71_71390

theorem students_playing_both (total_students : ℕ) (play_tennis_fraction : ℚ) (play_hockey_fraction : ℚ) :
  total_students = 600 →
  play_tennis_fraction = 3/4 →
  play_hockey_fraction = 0.60 →
  (total_students * play_tennis_fraction * play_hockey_fraction).to_nat = 270 :=
by
  intros h_total h_tennis h_hockey
  sorry

end students_playing_both_l71_71390


namespace evaluate_expression_l71_71583

theorem evaluate_expression :
  (18 : ℝ) / (14 * 5.3) = (1.8 : ℝ) / 7.42 :=
by
  sorry

end evaluate_expression_l71_71583


namespace senior_ticket_cost_l71_71819

theorem senior_ticket_cost (total_tickets : ℕ) (adult_ticket_cost : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) (senior_ticket_cost : ℕ) :
  total_tickets = 510 →
  adult_ticket_cost = 21 →
  total_receipts = 8748 →
  senior_tickets = 327 →
  senior_ticket_cost = 15 :=
by
  sorry

end senior_ticket_cost_l71_71819


namespace swimming_meet_second_time_l71_71224

theorem swimming_meet_second_time
    (pool_length : ℕ)
    (george_speed : ℕ)
    (henry_speed : ℕ)
    (first_meet_time : ℕ := pool_length / (george_speed + henry_speed))
    (george_turnaround_time : ℕ := (pool_length / 2) / george_speed)
    (henry_turnaround_time : ℕ := (pool_length / 2) / henry_speed)
    :
    second_meet_time : ℕ :=
    pool_length + george_turnaround_time + ((pool_length - (henry_turnaround_time - first_meet_time) * henry_speed) / (george_speed + henry_speed)) = 6 :=
begin
    sorry
end

end swimming_meet_second_time_l71_71224


namespace range_of_g_l71_71217

noncomputable def g (x : ℝ) : ℝ := (3 * x - 4) / (x + 2)

theorem range_of_g : set.range g = {y : ℝ | y ≠ 3} :=
by
  sorry

end range_of_g_l71_71217


namespace correct_graph_l71_71637

noncomputable def g (x : ℝ) : ℝ :=
  if h₁ : -4 ≤ x ∧ x ≤ -1 then -2 * x - 3
  else if h₂ : -1 ≤ x ∧ x ≤ 3 then -x
  else if h₃ : 3 ≤ x ∧ x ≤ 4 then 2 * (x - 3) - 2
  else 0

noncomputable def g_transformed (x : ℝ) : ℝ := g(x - 3) + 1

-- Graph A function description
noncomputable def graph_A (x : ℝ) : ℝ := -- Assume the same intervals shifted
  if h₁ : -1 ≤ x ∧ x ≤ 2 then -2 * (x - 3) - 3 + 1
  else if h₂ : 2 ≤ x ∧ x ≤ 6 then -(x - 3) + 1
  else if h₃ : 6 ≤ x ∧ x ≤ 7 then 2 * (x - 6) - 2 + 1
  else 0

theorem correct_graph : ∀ x, g_transformed x = graph_A x := 
by 
  intros x
  unfold g_transformed
  unfold g
  unfold graph_A
  split_ifs
  case : h₁ h₂ h₃ h₄ => sorry -- Skipping proof details

end correct_graph_l71_71637


namespace num_isosceles_triangles_with_perimeter_31_l71_71284

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ (a + b > c ∧ b + c > a ∧ a + c > b)

def is_valid_triangle_perimeter (a b c : ℕ) : Prop :=
  a + b + c = 31

def odd (n : ℕ) : Prop := n % 2 = 1

theorem num_isosceles_triangles_with_perimeter_31 : 
  ∃ n, n = 6 ∧ ∀ (a b c : ℕ), is_isosceles_triangle a b c ∧ is_valid_triangle_perimeter a b c ∧ odd c → (finset.card (finset.filter (λ ⟨a, b, c⟩, is_isosceles_triangle a b c ∧ is_valid_triangle_perimeter a b c ∧ odd c) (finset.univ : finset (ℕ × ℕ × ℕ))) = n) :=
by sorry

end num_isosceles_triangles_with_perimeter_31_l71_71284


namespace line_through_M_with_equal_intercepts_l71_71588

theorem line_through_M_with_equal_intercepts (M : Point) (line : Line) :
  (M = ⟨1, 1⟩ ∧ (∃ a, ∀ x y, line x y ↔ x + y = a) ∧ (∃ k, ∀ x y, line x y ↔ y = k * x)) →
  (∀ x y, line x y ↔ (x + y = 2) ∨ (y = x)) :=
by
  sorry

end line_through_M_with_equal_intercepts_l71_71588


namespace sum_of_roots_eq_l71_71062

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l71_71062


namespace sum_of_roots_of_quadratic_l71_71028

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l71_71028


namespace exists_smallest_n_l71_71962

theorem exists_smallest_n :
  ∃ n : ℕ, (n^2 + 20 * n + 19) % 2019 = 0 ∧ n = 2000 :=
sorry

end exists_smallest_n_l71_71962


namespace focus_of_parabola_l71_71589

noncomputable def parabola_equation : ℝ → ℝ := λ x, 4 * x^2 + 8 * x - 5

theorem focus_of_parabola :
  (∃ h k : ℝ, h = -1 ∧ k = -8.9375 ∧ ∀ y x : ℝ, parabola_equation x = y → y = 4 * (x + 1)^2 - 9) :=
sorry

end focus_of_parabola_l71_71589


namespace common_difference_is_4_l71_71362

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Defining the arithmetic sequence {a_n}
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
variable (d : ℤ) (a4_a5_sum : a 4 + a 5 = 24) (S6_val : S 6 = 48)

-- Statement to prove: given the conditions, d = 4
theorem common_difference_is_4 (h_seq : is_arithmetic_sequence a d) :
  d = 4 := sorry

end common_difference_is_4_l71_71362


namespace area_of_scalene_triangle_l71_71704

variables {A B C : ℝ} -- Angles in the triangle
variables {r r_A r_B r_C : ℝ} -- Radii
variables {s Δ : ℝ} -- Semiperimeter and Area

noncomputable def scalene_triangle_inradius_one (r_A r_B r_C : ℝ) (h1 : 20*(r_B^2 * r_C^2 + r_C^2 * r_A^2 + r_A^2 * r_B^2) = 19*(r_A * r_B * r_C)^2)
  (h2 : real.tan (A / 2) + real.tan (B / 2) + real.tan (C / 2) = 2.019)
  (r : ℝ) (hr : r = 1) : ℝ :=
let Δ := s in Δ

theorem area_of_scalene_triangle : scalene_triangle_inradius_one r_A r_B r_C (by norm_num) (by norm_num) 1 (by norm_num 1) = 2019 / 25 :=
begin
  sorry
end

#eval 100 * 2019 + 25 -- 201925

end area_of_scalene_triangle_l71_71704


namespace mango_distribution_l71_71283

theorem mango_distribution (friends : ℕ) (initial_mangos : ℕ) 
    (share_left : ℕ) (share_right : ℕ) 
    (eat_mango : ℕ) (pass_mango_right : ℕ)
    (H1 : friends = 100) 
    (H2 : initial_mangos = 2019)
    (H3 : share_left = 2) 
    (H4 : share_right = 1) 
    (H5 : eat_mango = 1) 
    (H6 : pass_mango_right = 1) :
    ∃ final_count, final_count = 8 :=
by
  -- Proof is omitted.
  sorry

end mango_distribution_l71_71283


namespace f_zero_derivative_not_extremum_l71_71778

noncomputable def f (x : ℝ) : ℝ := x ^ 3

theorem f_zero_derivative_not_extremum (x : ℝ) : 
  deriv f 0 = 0 ∧ ∀ (y : ℝ), y ≠ 0 → (∃ δ > 0, ∀ z, abs (z - 0) < δ → (f z / z : ℝ) ≠ 0) :=
by
  sorry

end f_zero_derivative_not_extremum_l71_71778


namespace sum_of_roots_eq_14_l71_71067

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l71_71067


namespace min_value_4x2_plus_y2_l71_71985

theorem min_value_4x2_plus_y2 {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 6) : 
  4 * x^2 + y^2 ≥ 18 := by
  sorry

end min_value_4x2_plus_y2_l71_71985


namespace shaded_region_area_l71_71770

theorem shaded_region_area {w h : ℕ} (hw : h = 12) (ww : w = 20) (shaded_ratio : ℚ) (hr : shaded_ratio = 5/8) : 
  let right_triangle_area := 0.5 * w/2 * h in 
  let isosceles_triangle_area := 0.5 * w * h in
  let shaded_right_triangle_area := shaded_ratio * right_triangle_area in
  let shaded_isosceles_triangle_area := shaded_ratio * isosceles_triangle_area in
  2 * shaded_right_triangle_area + shaded_isosceles_triangle_area = 150 := 
  sorry

end shaded_region_area_l71_71770


namespace pick_two_black_cards_l71_71157

-- Definition: conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13
def black_suits : ℕ := 2
def red_suits : ℕ := 2
def total_black_cards : ℕ := black_suits * cards_per_suit

-- Theorem: number of ways to pick two different black cards
theorem pick_two_black_cards :
  (total_black_cards * (total_black_cards - 1)) = 650 :=
by
  -- proof here
  sorry

end pick_two_black_cards_l71_71157


namespace combination_of_10_choose_3_l71_71331

theorem combination_of_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end combination_of_10_choose_3_l71_71331


namespace problem_1_problem_2_l71_71268

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a+1)*x + a

theorem problem_1 :
  let a := 2
  f(x, a) > 0 ↔ x ∈ (-∞, 1) ∪ (2, +∞) :=
sorry

theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, x > 1 → f(x, a) + 2*x ≥ 0) ↔ a ≤ 2 * sqrt 2 + 3 :=
sorry

end problem_1_problem_2_l71_71268


namespace problem_statement_l71_71541

def A : Prop := (∀ (x : ℝ), x^2 - 3*x + 2 = 0 → x = 2)
def B : Prop := (∃ (x : ℝ), x^2 - x + 1 < 0)
def C : Prop := (¬(∀ (x : ℝ), x > 2 → x^2 - 3*x + 2 > 0))

theorem problem_statement :
  ¬ (A ∧ ∀ (x : ℝ), (B → (x^2 - x + 1) ≥ 0) ∧ (¬(A) ∧ C)) :=
sorry

end problem_statement_l71_71541


namespace magnitude_difference_l71_71645

variables {x y : ℝ}
def a : ℝ × ℝ := (x, y)
def b : ℝ × ℝ := (-1, 2)

theorem magnitude_difference (h : a + b = (1, 3)) : 
  |(a.1 - 2 * b.1, a.2 - 2 * b.2)| = 5 :=
  sorry

end magnitude_difference_l71_71645


namespace sqrt_four_p_plus_one_over_five_eq_l71_71125

theorem sqrt_four_p_plus_one_over_five_eq (p : ℕ) (h1 : nat.prime p) (h2 : ∃ n : ℤ, p = (n+1)^5 - n^5) :
  ∃ k : ℤ, k % 2 = 1 ∧ (√(4 * p + 1) / 5) = (k^2 + 1) / 2 := by
  sorry

end sqrt_four_p_plus_one_over_five_eq_l71_71125


namespace hyperbola_intersection_l71_71578

theorem hyperbola_intersection (b : ℝ) (h₁ : b > 0) :
  (b > 1) → (∀ x y : ℝ, ((x + 3 * y - 1 = 0) → ( ∃ x y : ℝ, (x^2 / 4 - y^2 / b^2 = 1) ∧ (x + 3 * y - 1 = 0))))
  :=
  sorry

end hyperbola_intersection_l71_71578


namespace range_of_t_l71_71725

-- Definitions from conditions
def A := { x : ℝ | -1 ≤ x }
def B (t : ℝ) := { y : ℝ | t ≤ y }
def f (x : ℝ) : ℝ := x^2

-- Statement of the proof problem
theorem range_of_t (t : ℝ) : (∀ x ∈ A, f x ∈ B t) → t ∈ Iic 0 := by
  sorry

end range_of_t_l71_71725


namespace correct_statements_l71_71838

-- Definitions of the statements as Lean propositions
def StatementA : Prop := ∀ (T : Type) [RightTriangle T], Rotating T.results_cone_or_double_cone
def StatementB : Prop := ∀ (T : Type) [IsoscelesTriangle T], RotatingAroundMidline T.results_cone
def StatementC : Prop := ∀ (C : Cone), ∀ (g1 g2 : Generatrix C), g1 ∩ g2 = IsoscelesTriangle
def StatementD : Prop := ∀ (C : Cone), ∃ (g : Generatrix C), length g > diameter (base C)

-- Theorem stating that B, C, and D are correct
theorem correct_statements : StatementB ∧ StatementC ∧ StatementD := by
  sorry

end correct_statements_l71_71838


namespace problem_1_minimum_value_problem_2_range_of_a_l71_71989

noncomputable def e : ℝ := Real.exp 1  -- Definition of e as exp(1)

-- Question I:
-- Prove that the minimum value of the function f(x) = e^x - e*x - e is -e.
theorem problem_1_minimum_value :
  ∃ x : ℝ, (∀ y : ℝ, (Real.exp x - e * x - e) ≤ (Real.exp y - e * y - e))
  ∧ (Real.exp x - e * x - e) = -e := 
sorry

-- Question II:
-- Prove that the range of values for a such that f(x) = e^x - a*x - a >= 0 for all x is [0, 1].
theorem problem_2_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, (Real.exp x - a * x - a) ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 :=
sorry

end problem_1_minimum_value_problem_2_range_of_a_l71_71989


namespace tan_alpha_minus_beta_l71_71654

theorem tan_alpha_minus_beta (α β : ℝ) (hα : Real.tan α = 8) (hβ : Real.tan β = 7) :
  Real.tan (α - β) = 1 / 57 := 
sorry

end tan_alpha_minus_beta_l71_71654


namespace unit_prices_cost_relationship_purchasing_plans_l71_71142

-- Definitions of the unit prices and conditions
def unit_price_A (p_A : ℝ) := p_A
def unit_price_B (p_B : ℝ) := p_B

-- Condition: unit price of A is 10 yuan higher than B
def price_difference (p_A p_B : ℝ) : Prop := p_A = p_B + 10

-- Condition: 2 brand A and 3 brand B cost 220 yuan
def total_cost_condition (p_A p_B : ℝ) : Prop := 2 * p_A + 3 * p_B = 220

-- Theorem: Find unit prices of A and B given the conditions
theorem unit_prices (p_A p_B : ℝ) (h_diff : price_difference p_A p_B) 
  (h_cost : total_cost_condition p_A p_B) :
  p_A = 50 ∧ p_B = 40 :=
sorry

-- Define the total cost relationship
def total_cost (m : ℕ) : ℕ := 10 * m + 2400

-- Theorem: Relationship between total cost W and number of brand A balls m
theorem cost_relationship (m : ℕ) : total_cost m = 10 * m + 2400 :=
rfl

-- Define the constraints for purchasing plans
def valid_purchasing_plan (m : ℕ) : Prop :=
  43 ≤ m ∧ m ≤ 45 ∧ total_cost m ≤ 2850

-- Theorem: Valid purchasing plans and minimum cost
theorem purchasing_plans : 
  {m | valid_purchasing_plan m}.finite ∧
  (∃ min_m, min_m ∈ {m | valid_purchasing_plan m} 
    ∧ total_cost min_m = 2830) :=
sorry

end unit_prices_cost_relationship_purchasing_plans_l71_71142


namespace expected_coincidences_l71_71480

/-- Given conditions for the test -/
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8
def vasya_prob_correct : ℝ := 6 / 20
def misha_prob_correct : ℝ := 8 / 20
def coincidence_prob : ℝ :=
  (vasya_prob_correct * misha_prob_correct) + (1 - vasya_prob_correct) * (1 - misha_prob_correct)

/-- Expected number of coincidences -/
theorem expected_coincidences :
  20 * coincidence_prob = 10.8 :=
by {
  -- vasya_prob_correct = 0.3
  -- misha_prob_correct = 0.4
  -- probability of coincidence = 0.3 * 0.4 + 0.7 * 0.6 = 0.54
  -- expected number of coincidences = 20 * 0.54 = 10.8
  sorry
}

end expected_coincidences_l71_71480


namespace find_slope_l71_71315

theorem find_slope (k : ℝ) : 
  (∃ P₁ P₂ P₃ : ℝ × ℝ, 
    (P₁.1 ^ 2 + P₁.2 ^ 2 - 2 * P₁.1 - 6 * P₁.2 + 1 = 0)
    ∧ (P₂.1 ^ 2 + P₂.2 ^ 2 - 2 * P₂.1 - 6 * P₂.2 + 1 = 0)
    ∧ (P₃.1 ^ 2 + P₃.2 ^ 2 - 2 * P₃.1 - 6 * P₃.2 + 1 = 0)
    ∧ ((abs (-k * 1 + 3) / real.sqrt (k ^ 2 + 1)) = 1))
  → k = 4/3 := 
by sorry

end find_slope_l71_71315


namespace sum_of_roots_l71_71020

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l71_71020


namespace find_r_in_geometric_sum_l71_71314

theorem find_r_in_geometric_sum (S_n : ℕ → ℕ) (r : ℤ)
  (hSn : ∀ n : ℕ, S_n n = 2 * 3^n + r)
  (hgeo : ∀ n : ℕ, n ≥ 2 → S_n n - S_n (n - 1) = 4 * 3^(n - 1))
  (hn1 : S_n 1 = 6 + r) :
  r = -2 :=
by
  sorry

end find_r_in_geometric_sum_l71_71314


namespace probability_sum_odd_l71_71806

theorem probability_sum_odd (x y : ℕ) 
  (hx : x > 0) (hy : y > 0) 
  (h_even : ∃ z : ℕ, z % 2 = 0 ∧ z > 0) 
  (h_odd : ∃ z : ℕ, z % 2 = 1 ∧ z > 0) : 
  (∃ p : ℝ, 0 < p ∧ p < 1 ∧ p = 0.5) :=
sorry

end probability_sum_odd_l71_71806


namespace num_classes_received_basketballs_l71_71792

theorem num_classes_received_basketballs (total_basketballs left_basketballs : ℕ) 
  (h : total_basketballs = 54) (h_left : left_basketballs = 5) : 
  (total_basketballs - left_basketballs) / 7 = 7 :=
by
  sorry

end num_classes_received_basketballs_l71_71792


namespace questions_answered_second_half_l71_71843

theorem questions_answered_second_half :
  ∀ (q1 q2 p s : ℕ), q1 = 3 → p = 3 → s = 15 → s = (q1 + q2) * p → q2 = 2 :=
by
  intros q1 q2 p s hq1 hp hs h_final_score
  -- proofs go here, but we skip them
  sorry

end questions_answered_second_half_l71_71843


namespace sum_first_six_terms_reciprocal_l71_71613

theorem sum_first_six_terms_reciprocal 
  (S_n : ℕ → ℝ) (a_n : ℕ → ℝ)
  (h₁ : ∀ n, S_n n = 2 * a_n n - 1)
  (h₂ : S_n 1 = a_n 1) :
  (∑ k in Finset.range 6, 1 / a_n (k + 1)) = 63 / 32 := by
  sorry

end sum_first_six_terms_reciprocal_l71_71613


namespace sum_of_roots_l71_71105

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l71_71105


namespace ratio_sum_eq_l71_71718

variable {x y z : ℝ}

theorem ratio_sum_eq (h1 : (4 * y) ^ 2 = 15 * x * z)
(h2 : 2 / y = 1 / x + 1 / z) :
  (x / z) + (z / x) = 34 / 15 :=
sorry

end ratio_sum_eq_l71_71718


namespace min_people_liking_both_l71_71452

theorem min_people_liking_both (A B C V : ℕ) (hA : A = 200) (hB : B = 150) (hC : C = 120) (hV : V = 80) :
  ∃ D, D = 80 ∧ D ≤ min B (A - C + V) :=
by {
  sorry
}

end min_people_liking_both_l71_71452


namespace chandler_weeks_to_buy_bike_l71_71562

-- Define the given problem conditions as variables/constants
def bike_cost : ℕ := 650
def grandparents_gift : ℕ := 60
def aunt_gift : ℕ := 45
def cousin_gift : ℕ := 25
def weekly_earnings : ℕ := 20
def total_birthday_money : ℕ := grandparents_gift + aunt_gift + cousin_gift

-- Define the total money Chandler will have after x weeks
def total_money_after_weeks (x : ℕ) : ℕ := total_birthday_money + weekly_earnings * x

-- The main theorem states that Chandler needs 26 weeks to save enough money to buy the bike
theorem chandler_weeks_to_buy_bike : ∃ x : ℕ, total_money_after_weeks x = bike_cost :=
by
  -- Since we know x = 26 from the solution:
  use 26
  sorry

end chandler_weeks_to_buy_bike_l71_71562


namespace quadratic_function_min_value_at_1_l71_71440

-- Define the quadratic function y = (x - 1)^2 - 3
def quadratic_function (x : ℝ) : ℝ :=
  (x - 1) ^ 2 - 3

-- The theorem to prove is that this quadratic function reaches its minimum value when x = 1.
theorem quadratic_function_min_value_at_1 : ∃ x : ℝ, quadratic_function x = quadratic_function 1 :=
by
  sorry

end quadratic_function_min_value_at_1_l71_71440


namespace solve_system_l71_71755

theorem solve_system :
  ∃ x y : ℝ, ((0.5 * log 2 x - log 2 y = 0) ∧ (x^2 - 2 * y^2 = 8)) ∧ 
             ((x = 4) ∧ (y = 2) ∨ (x = 4) ∧ (y = -2)) :=
by
  sorry

end solve_system_l71_71755


namespace shorter_piece_length_l71_71501

theorem shorter_piece_length 
  (total_length : ℕ)
  (difference : ℕ)
  (h : total_length = 120)
  (d : difference = 22) :
  ∃ short_piece_len : ℕ, short_piece_len + (short_piece_len + difference) = total_length ∧ short_piece_len = 49 :=
by {
  use 49,
  split,
  { 
    rw [h, d],
    norm_num,
  },
  {
    norm_num,
  }
}

end shorter_piece_length_l71_71501


namespace sum_of_roots_eq_14_l71_71052

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l71_71052


namespace solve_for_a_and_b_l71_71313
-- Import the necessary library

open Classical

variable (a b x : ℝ)

theorem solve_for_a_and_b (h1 : 0 ≤ x) (h2 : x < 1) (h3 : x + 2 * a ≥ 4) (h4 : (2 * x - b) / 3 < 1) : a + b = 1 := 
by
  sorry

end solve_for_a_and_b_l71_71313


namespace sum_of_roots_l71_71099

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l71_71099


namespace lcm_not_consecutive_numbers_l71_71800

open Nat

theorem lcm_not_consecutive_numbers (n : ℕ) (h : n = 10^1000) :
  ∀ (a : Fin n → ℕ), ¬ ∃ b : Fin n → ℕ, (∀ i : Fin n, b i = lcm (a i.val) (a ((i + 1) % n).val)) ∧ (∃ k : ℕ, ∀ i : Fin n, b i = k + i) :=
sorry

end lcm_not_consecutive_numbers_l71_71800


namespace problem_I_problem_II_l71_71988

open Set

variable (a x : ℝ)

def p : Prop := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem problem_I (hp : p a) : a ≤ 1 :=
  sorry

theorem problem_II (hpq : ¬ (p a ∧ q a)) : a ∈ Ioo (-2 : ℝ) (1 : ℝ) ∪ Ioi 1 :=
  sorry

end problem_I_problem_II_l71_71988


namespace value_of_a_l71_71338

theorem value_of_a (a : ℝ) : 
  (∃ (T : ℝ), T = (∑ r in range(7), (binom 6 r) * a^(6 - r) * (x^(6 - 2 * r)))) → T = -160 → a = -2 :=
by sorry

end value_of_a_l71_71338


namespace three_digit_even_two_odd_no_repetition_l71_71975

-- Define sets of digits
def digits : List ℕ := [0, 1, 3, 4, 5, 6]
def evens : List ℕ := [0, 4, 6]
def odds : List ℕ := [1, 3, 5]

noncomputable def total_valid_numbers : ℕ :=
  let choose_0 := 12 -- Given by A_{2}^{1} A_{3}^{2} = 12
  let without_0 := 36 -- Given by C_{2}^{1} * C_{3}^{2} * A_{3}^{3} = 36
  choose_0 + without_0

theorem three_digit_even_two_odd_no_repetition : total_valid_numbers = 48 :=
by
  -- Proof would be provided here
  sorry

end three_digit_even_two_odd_no_repetition_l71_71975


namespace promotional_price_difference_l71_71904

theorem promotional_price_difference
  (normal_price : ℝ)
  (months : ℕ)
  (issues_per_month : ℕ)
  (discount_per_issue : ℝ)
  (h1 : normal_price = 34)
  (h2 : months = 18)
  (h3 : issues_per_month = 2)
  (h4 : discount_per_issue = 0.25) : 
  normal_price - (months * issues_per_month * discount_per_issue) = 9 := 
by 
  sorry

end promotional_price_difference_l71_71904


namespace expression1_expression2_l71_71855

theorem expression1 : 6 - (-12) / (-3) = 2 := by
  sorry

theorem expression2 : (-10)^4 + [(-4)^2 - (3 + 3^2) * 2] = 9992 := by
  sorry

end expression1_expression2_l71_71855


namespace tetrahedral_pyramid_area_l71_71524

noncomputable def midpoint (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2, (P.3 + Q.3) / 2)

noncomputable def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

theorem tetrahedral_pyramid_area :
  let A := (0, 0, 0)
  let B := (6, 0, 0)
  let C := (5.8, 4, 0)
  let D := (2, 1.2, 2.4)
  let M := midpoint A D
  let N := midpoint B C
  distance A B = 6 ∧ distance B C = 4 ∧ distance C D = 5 ∧
  distance A D = 4 ∧ distance B D = 5 ∧ distance A C = 7 ∧
  distance M N = 4.4221 ∧ distance B N = 4.0212 → 
  let MB := (5, -0.6, -1.2)
  let BN := (-0.1, 2, 0)
  let cross_MB_BN := ( -2.4, -0.12, 10.2 )
  (real.sqrt (cross_MB_BN.1^2 + cross_MB_BN.2^2 + cross_MB_BN.3^2)) / 2 = 5.13 :=
sorry

end tetrahedral_pyramid_area_l71_71524


namespace correct_multiplication_l71_71294

theorem correct_multiplication (n : ℕ) (h₁ : 15 * n = 45) : 5 * n = 15 :=
by
  -- skipping the proof
  sorry

end correct_multiplication_l71_71294


namespace geometric_log_sequence_sum_sequence_formula_limit_value_l71_71993

-- Given sequence \{a_n\} where a_1 = 2 and recursion relation a_{n+1} = a_n^2 + 2a_n
def a_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  a 0 = 2 ∧ ∀ n : ℕ, a (n + 1) = a n^2 + 2 * a n

-- Prove that \{\lg(1 + a_n)\} is a geometric sequence with ratio 2
theorem geometric_log_sequence (a : ℕ → ℝ) (n : ℕ) (h : a_sequence a n) :
  ∃ r : ℝ, r = 2 ∧ ∀ k : ℕ, (Real.log (1 + a k)) = (Real.geom_ratio (Real.log (1 + a k)) r ^ k) :=
sorry

-- Given condition for sum sequence S_n and b_1 = 1
def sum_sequence (b S : ℕ → ℝ) (n : ℕ) : Prop :=
  b 0 = 1 ∧ ∀ n : ℕ, S n ^ 2 = b n * (S n - 0.5)

-- Prove S_n = \frac{1}{2n-1}
noncomputable
def S_n_formula (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  1 / (2 * n - 1)

theorem sum_sequence_formula (b S : ℕ → ℝ) (n : ℕ) (h : sum_sequence b S n) :
  S n = S_n_formula S n :=
sorry

-- Given definitions for T_n and c_n 
def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (1 + a i)

def c_n (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  (2 * S n) / (2 * n + 1)

-- Prove limit \lim_{n \to \infty} \left[ \frac{T_n}{3^{2^n}+1} \cdot \sum_{k=1}^{n}c_k \right] = \frac{1}{3}
theorem limit_value (a S : ℕ → ℝ) (n : ℕ) (h_a : a_sequence a n) (h_S : sum_sequence b S n) :
  Filter.Tendsto (λ n:ℕ, (T_n a n / (3 ^ (2 ^ n) + 1) * (∑ k in finset.range n, c_n S k))) Filter.atTop (Filter.principal ({y : ℝ | y = 1/3})) :=
sorry

end geometric_log_sequence_sum_sequence_formula_limit_value_l71_71993


namespace proposition_A_proposition_B_proposition_C_proposition_D_l71_71837

theorem proposition_A (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a ≠ 1) :=
by {
  sorry
}

theorem proposition_B : (¬ ∀ x : ℝ, x^2 + x + 1 < 0) → (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≥ 0) :=
by {
  sorry
}

theorem proposition_C : ¬ ∀ x ≠ 0, x + 1 / x ≥ 2 :=
by {
  sorry
}

theorem proposition_D (m : ℝ) : (∃ x : ℝ, (1 < x ∧ x < 2) ∧ x^2 + m * x + 4 < 0) → m < -4 :=
by {
  sorry
}

end proposition_A_proposition_B_proposition_C_proposition_D_l71_71837


namespace determinant_B_power_four_l71_71653

-- Given conditions
variable (B : Matrix ℕ ℕ ℝ) (h : det B = -3)

-- Statement to prove
theorem determinant_B_power_four : det (B ^ 4) = 81 := by
  sorry

end determinant_B_power_four_l71_71653


namespace projectile_first_reaches_28_l71_71772

theorem projectile_first_reaches_28 (t : ℝ) (h_eq : ∀ t, -4.9 * t^2 + 23.8 * t = 28) : 
    t = 2 :=
sorry

end projectile_first_reaches_28_l71_71772


namespace pine_trees_multiple_of_27_l71_71164

noncomputable def numberOfPineTrees (n : ℕ) : ℕ := 27 * n

theorem pine_trees_multiple_of_27 (oak_trees : ℕ) (max_trees_per_row : ℕ) (rows_of_oak : ℕ) :
  oak_trees = 54 → max_trees_per_row = 27 → rows_of_oak = oak_trees / max_trees_per_row →
  ∃ n : ℕ, numberOfPineTrees n = 27 * n :=
by
  intros
  use (oak_trees - rows_of_oak * max_trees_per_row) / 27
  sorry

end pine_trees_multiple_of_27_l71_71164


namespace solve_system_l71_71751

open Real

theorem solve_system :
  (∃ x y : ℝ, (1 / 2) * log 2 x - log 2 y = 0 ∧ x^2 - 2 * y^2 = 8 ∧
    ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = -2))) :=
by 
  sorry

end solve_system_l71_71751


namespace triangle_properties_l71_71330

-- Definitions and conditions
variable {A B C M D E F L P N : Type}
variables (h₁ : isAcuteAngledTriangle A B C) 
          (h₂ : isOrthocenter M A B C) 
          (h₃ : isMidpoint D M A) 
          (h₄ : isMidpoint E M B) 
          (h₅ : isMidpoint F M C) 
          (h₆ : parallelThroughMidpoints D E F A B C)

-- The proof problem
theorem triangle_properties :
  similarTriangles A B C D E F ∧
  congruentTriangles L P N A B C ∧
  isCircumcenter M L P N ∧
  centersOfCirclesPassingThrough L P N A B C M := 
by
  sorry

end triangle_properties_l71_71330


namespace bug_visits_correct_number_of_tiles_l71_71518

def rectangle : Type := {width length : ℕ}

def number_of_tiles_visited (rect : rectangle) : ℕ :=
  rect.width + rect.length - Nat.gcd rect.width rect.length

theorem bug_visits_correct_number_of_tiles :
  ∀ (rect : rectangle), rect.width = 15 → rect.length = 35 →
  number_of_tiles_visited rect = 45 := by
  sorry

end bug_visits_correct_number_of_tiles_l71_71518


namespace probability_of_both_contracts_l71_71789

noncomputable def P (X : Type) := ℝ

variable {Ω : Type}
variable (A B : Ω → Prop)

axiom P_A : P Ω A = 4 / 5
axiom P_B : P Ω B = 3 / 5
axiom P_A_union_B : P Ω (λ ω, A ω ∨ B ω) = 9 / 10

theorem probability_of_both_contracts :
  P Ω (λ ω, A ω ∧ ¬ B ω) = 7 / 10 :=
by {
  -- This is where the proof would go
  sorry
}

end probability_of_both_contracts_l71_71789


namespace missionaries_and_cannibals_successful_crossing_l71_71811

structure State :=
  (left_missionaries : ℕ)
  (left_cannibals : ℕ)
  (right_missionaries : ℕ)
  (right_cannibals : ℕ)
  (is_boat_on_left : Bool)

def initial_state : State :=
  { left_missionaries := 3
  , left_cannibals := 3
  , right_missionaries := 0
  , right_cannibals := 0
  , is_boat_on_left := true  
  }

def final_state : State :=
  { left_missionaries := 0
  , left_cannibals := 0
  , right_missionaries := 3
  , right_cannibals := 3
  , is_boat_on_left := false 
  }

noncomputable def valid_move (s1 s2 : State) : Bool :=
  let valid := 
    (s1.left_missionaries >= s2.left_missionaries) && 
    (s1.left_cannibals >= s2.left_cannibals) && 
    (s2.left_missionaries >= s2.left_cannibals ∨ s2.left_missionaries = 0) && 
    (s2.right_missionaries >= s2.right_cannibals ∨ s2.right_missionaries = 0) 
  valid && 
  (if s1.is_boat_on_left then (s1.left_missionaries - s2.left_missionaries + s1.left_cannibals - s2.left_cannibals <= 2)
  else (s2.left_missionaries - s1.left_missionaries + s2.left_cannibals - s1.left_cannibals <= 2)) &&
  (s1.is_boat_on_left != s2.is_boat_on_left)

noncomputable def sequence_of_moves (states : List State) : Prop :=
  match states with
  | [] => False
  | [s] => s = final_state
  | s1 :: s2 :: ss => valid_move s1 s2 && sequence_of_moves (s2 :: ss)

theorem missionaries_and_cannibals_successful_crossing : 
  ∃ states : List State, 
    sequence_of_moves (initial_state :: states) := 
sorry

end missionaries_and_cannibals_successful_crossing_l71_71811


namespace find_angle_B_range_cosA_plus_sinC_l71_71994

-- Given definitions and conditions
def acute_triangle (A B C : ℝ) : Prop := 
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ 
  A + B + C = π

def sides_opposite_angles (A B C a b c : ℝ) : Prop :=
  a / sin A = b / sin B ∧ b / sin B = c / sin C

def a_eq_2b_sinA (A B C a b c : ℝ) : Prop := 
  a = 2 * b * sin A

-- (I) Prove that angle B = π / 6
theorem find_angle_B
  (A B C a b c : ℝ)
  (h1 : acute_triangle A B C)
  (h2 : sides_opposite_angles A B C a b c)
  (h3 : a_eq_2b_sinA A B C a b c) :
  B = π / 6 :=
sorry

-- (II) Prove that the range of cos A + sin C is (√3 / 2, 3 / 2)
theorem range_cosA_plus_sinC
  (A B C a b c : ℝ)
  (h1 : acute_triangle A B C)
  (h2 : sides_opposite_angles A B C a b c)
  (h3 : a_eq_2b_sinA A B C a b c) :
  ∀ x, x = cos A + sin C → (√3 / 2 < x ∧ x < 3 / 2) :=
sorry

end find_angle_B_range_cosA_plus_sinC_l71_71994


namespace largest_k_for_coprime_set_l71_71958

theorem largest_k_for_coprime_set : 
  ∃ k, (∀ n : ℕ, n > 0 → ∃ s : finset ℕ, (∀ x ∈ s, x ∈ finset.Icc (n + 1) (n + 16) ∧ nat.coprime x (n * (n + 17))) ∧ s.card ≥ k) ∧ k = 1 :=
begin
  sorry,
end

end largest_k_for_coprime_set_l71_71958


namespace bill_apples_left_l71_71912

-- Definitions based on the conditions
def total_apples : Nat := 50
def apples_per_child : Nat := 3
def number_of_children : Nat := 2
def apples_per_pie : Nat := 10
def number_of_pies : Nat := 2

-- The main statement to prove
theorem bill_apples_left : total_apples - ((apples_per_child * number_of_children) + (apples_per_pie * number_of_pies)) = 24 := by
sorry

end bill_apples_left_l71_71912


namespace plotted_points_lie_on_line_l71_71220

theorem plotted_points_lie_on_line :
  ∀ (t : ℝ), let x := Real.cos t ^ 2 in let y := 1 - Real.cos t ^ 2 in x + y = 1 :=
by sorry

end plotted_points_lie_on_line_l71_71220


namespace gray_region_area_proof_l71_71343

def radius_inner_circle : ℝ := 2 -- Given r = 2 feet from the solution

def radius_outer_circle : ℝ := 6 -- Given 3r = 6 feet from the solution

def area_inner_circle : ℝ := π * radius_inner_circle^2 -- Area of the inner circle

def area_outer_circle : ℝ := π * radius_outer_circle^2 -- Area of the outer circle

def area_gray_region : ℝ := area_outer_circle - area_inner_circle -- Area of the gray region

theorem gray_region_area_proof :
  radius_outer_circle = 3 * radius_inner_circle →
  radius_outer_circle - radius_inner_circle = 4 →
  area_gray_region = 32 * π :=
by
  intros h1 h2
  rw [radius_inner_circle, radius_outer_circle, area_inner_circle, area_outer_circle, area_gray_region]
  -- Further proofs would go here
  sorry -- Proof is omitted as per instructions

end gray_region_area_proof_l71_71343


namespace probability_sandwich_l71_71352

noncomputable def prob_sandwich_letters : ℚ :=
  let beach := ['B', 'E', 'A', 'C', 'H']
  let gardens := ['G', 'A', 'R', 'D', 'E', 'N', 'S']
  let shine := ['S', 'H', 'I', 'N', 'E']
  let prob_beach := (Finset.card {s | s ⊆ Finset.mk (beach) ∧ Finset.card s = 2} / Finset.card {s | s ⊆ Finset.mk (beach) ∧ Finset.card s = 3} : ℚ) -- Probability of getting A and C from BEACH
  let prob_gardens := (Finset.card {s | s ⊆ Finset.mk (gardens) ∧ S ∈ s ∧ N ∈ s ∧ D ∈ s ∧ Finset.card s = 5} / Finset.card {s | s ⊆ Finset.mk (gardens) ∧ Finset.card s = 5} : ℚ) -- Probability of getting S, N, D from GARDENS
  let prob_shine := (Finset.card {s | s ⊆ Finset.mk (shine) ∧ H ∈ s ∧ I ∈ s ∧ Finset.card s = 2} / Finset.card {s | s ⊆ Finset.mk (shine) ∧ Finset.card s = 2} : ℚ) -- Probability of getting H and I from SHINE
  prob_beach * prob_gardens * prob_shine

theorem probability_sandwich : prob_sandwich_letters = 1/350 := by
  sorry

end probability_sandwich_l71_71352


namespace min_number_of_benches_l71_71435

theorem min_number_of_benches (male_students : ℕ) (female_students : ℕ) (benches : ℕ) 
  (h1 : female_students = 4 * male_students)
  (h2 : male_students = 29)
  (h3 : ∀ s, s ≥ 5) :
  benches ≥ (male_students + female_students) / 5 :=
begin
  sorry
end

end min_number_of_benches_l71_71435


namespace sum_of_roots_eq_14_l71_71038

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l71_71038


namespace cistern_filling_time_l71_71453

theorem cistern_filling_time
  (R1 R2 R3 : ℝ)
  (hR1 : R1 = 1 / 10)
  (hR2 : R2 = 1 / 12)
  (hR3 : R3 = - 1 / 25) :
  let R_total := R1 + R2 + R3 in
  let T := 1 / R_total in
  T = 300 / 43 := by
begin
  sorry
end

end cistern_filling_time_l71_71453


namespace max_average_of_multiples_l71_71468

def multiples_of (k n : ℕ) : List ℕ :=
  List.filter (λ x, x % k = 0) (List.range (n + 1))

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem max_average_of_multiples :
  let A := multiples_of 3 201
  let B := multiples_of 4 201
  let C := multiples_of 6 201
  let D := multiples_of 7 201
  let E := multiples_of 9 201
  average E > average A ∧ average E > average B ∧ average E > average C ∧ average E > average D :=
by
  sorry

end max_average_of_multiples_l71_71468


namespace girls_ratio_correct_l71_71808

-- Define the number of total attendees
def total_attendees : ℕ := 100

-- Define the percentage of faculty and staff
def faculty_staff_percentage : ℕ := 10

-- Define the number of boys among the students
def number_of_boys : ℕ := 30

-- Define the function to calculate the number of faculty and staff
def faculty_staff (total_attendees faculty_staff_percentage: ℕ) : ℕ :=
  (faculty_staff_percentage * total_attendees) / 100

-- Define the function to calculate the number of students
def number_of_students (total_attendees faculty_staff_percentage: ℕ) : ℕ :=
  total_attendees - faculty_staff total_attendees faculty_staff_percentage

-- Define the function to calculate the number of girls
def number_of_girls (total_attendees faculty_staff_percentage number_of_boys: ℕ) : ℕ :=
  number_of_students total_attendees faculty_staff_percentage - number_of_boys

-- Define the function to calculate the ratio of girls to the remaining attendees
def ratio_girls_to_attendees (total_attendees faculty_staff_percentage number_of_boys: ℕ) : ℚ :=
  (number_of_girls total_attendees faculty_staff_percentage number_of_boys) / 
  (number_of_students total_attendees faculty_staff_percentage)

-- The theorem statement that needs to be proven (no proof required)
theorem girls_ratio_correct : ratio_girls_to_attendees total_attendees faculty_staff_percentage number_of_boys = 2 / 3 := 
by 
  -- The proof is skipped.
  sorry

end girls_ratio_correct_l71_71808


namespace remainder_when_M_divided_by_45_l71_71361

open Nat

def M : ℕ := -- define M as the 95-digit number formed by writing the integers from 1 to 50 in order
  Nat.ofDigits 10 ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50].flatMap digits)

theorem remainder_when_M_divided_by_45 : M % 45 = 15 :=
  by
    have h5 : M % 5 = 0 := 
      sorry  -- M is divisible by 5 because its last digit is 0

    have h9 : M % 9 = 6 := 
      sorry  -- M mod 9 as calculated from the sum of digits

    -- Use Chinese Remainder Theorem for final result:
    sorry
    
-- sorrys are used to skip proofs for helpers

end remainder_when_M_divided_by_45_l71_71361


namespace area_trapezoid_EFBA_l71_71856

open Set

noncomputable section

-- Define the variables and conditions provided in the problem.
variables {A B C D E F: Point}
variables {rectangle_ABCD: is_rectangle A B C D}
variables {area_ABCD: area A B C D = 20}
variables {AE_length: distance A E = 4}
variables {BF_length: distance B F = 3}
variables {E_aligns_horizontally_with_F: aligned_horizontally E F}

-- The theorem to prove that the area of trapezoid EFBA is 3 square units.
theorem area_trapezoid_EFBA : 
  area_trapezoid E F B A = 3 :=
sorry

end area_trapezoid_EFBA_l71_71856


namespace probability_distribution_correct_l71_71895

noncomputable def X_possible_scores : Set ℤ := {-90, -30, 30, 90}

def prob_correct : ℚ := 0.8
def prob_incorrect : ℚ := 1 - prob_correct

def P_X_neg90 : ℚ := prob_incorrect ^ 3
def P_X_neg30 : ℚ := 3 * prob_correct * prob_incorrect ^ 2
def P_X_30 : ℚ := 3 * prob_correct ^ 2 * prob_incorrect
def P_X_90 : ℚ := prob_correct ^ 3

def P_advance : ℚ := P_X_30 + P_X_90

theorem probability_distribution_correct :
  (P_X_neg90 = (1/125) ∧ P_X_neg30 = (12/125) ∧ P_X_30 = (48/125) ∧ P_X_90 = (64/125)) ∧ 
  P_advance = (112/125) := 
by
  sorry

end probability_distribution_correct_l71_71895


namespace sum_of_g_36_l71_71713

noncomputable def g : ℕ → ℕ := sorry

axiom g_increasing : ∀ n : ℕ, n > 0 → g(n + 1) > g(n)
axiom g_multiplicative : ∀ m n : ℕ, m > 0 → n > 0 → g(m * n) = g(m) * g(n)
axiom g_power_equiv : ∀ m n : ℕ, m ≠ n → m > 0 → n > 0 → m^n = n^m → (g(m) = n ∨ g(n) = m)

theorem sum_of_g_36 : g(36) = 1296 :=
by 
  sorry

end sum_of_g_36_l71_71713


namespace supplement_angle_greater_complement_angle_equal_unique_perpendicular_shortest_perpendicular_distance_incorrect_statement_l71_71840

theorem supplement_angle_greater (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ 180) : ¬(180 - θ > θ) → (θ ≥ 90) :=
begin
  intro h,
  sorry,
end

theorem complement_angle_equal (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ 90) : (90 - θ = 90 - θ) :=
begin
  refl,
end

theorem unique_perpendicular {P : Type} [EuclideanSpace ℝ P]
  (p : P) (l : line P) (hp : p ∉ l) : ∃! m : line P, p ∈ m ∧ m ⊥ l :=
begin
  sorry,
end

theorem shortest_perpendicular_distance {P : Type} [EuclideanSpace ℝ P]
  (p : P) (l : line P) (hp : p ∉ l) : ∀ q ∈ l, dist p l = dist p (orth_proj l p) :=
begin
  intros q hq,
  sorry,
end

theorem incorrect_statement (θ : ℝ) : (¬ θ ≥ 90) :=
begin
  sorry,
end

end supplement_angle_greater_complement_angle_equal_unique_perpendicular_shortest_perpendicular_distance_incorrect_statement_l71_71840


namespace inequality_proof_l71_71245

theorem inequality_proof
  (a b c d e f : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (he : 0 < e)
  (hf : 0 < f)
  (h_condition : abs (Real.sqrt (a * b) - Real.sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := 
sorry

end inequality_proof_l71_71245


namespace sum_of_roots_of_equation_l71_71079

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l71_71079


namespace finite_elements_sum_cond_l71_71469

-- Define the conditions required for the proof problem
variables (A : Set ℕ) (n : ℕ) [Infinite A]
variable (hn : n > 1)
variable (hp : ∀ p, Prime p → p ∣ n → False → Set.Infinite {a ∈ A | ¬ p ∣ a})

-- Define the main theorem statement
theorem finite_elements_sum_cond (m : ℕ) (hm : m > 1) (hngcd : Nat.gcd m n = 1) :
  ∃ B ⊆ A, Finite B ∧
    (let S := Finset.sum B id in
     S ≡ 1 [MOD m] ∧ S ≡ 0 [MOD n]) :=
sorry

end finite_elements_sum_cond_l71_71469


namespace no_solutions_for_divisibility_by_3_l71_71209

theorem no_solutions_for_divisibility_by_3 (x y : ℤ) : ¬ (x^2 + y^2 + x + y ∣ 3) :=
sorry

end no_solutions_for_divisibility_by_3_l71_71209


namespace markers_blue_count_l71_71187

theorem markers_blue_count (total_markers : ℝ) (red_markers : ℝ) (blue_markers : ℝ) :
  total_markers = 64.0 → red_markers = 41.0 → blue_markers = total_markers - red_markers → blue_markers = 23.0 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3
  sorry

end markers_blue_count_l71_71187


namespace find_d_l71_71430

noncomputable def problem_condition :=
  ∃ (v d : ℝ × ℝ) (t : ℝ) (x y : ℝ),
  (y = (5 * x - 7) / 6) ∧ 
  ((x, y) = (v.1 + t * d.1, v.2 + t * d.2)) ∧ 
  (x ≥ 4) ∧ 
  (dist (x, y) (4, 2) = t)

noncomputable def correct_answer : ℝ × ℝ := ⟨6 / 7, 5 / 7⟩

theorem find_d 
  (h : problem_condition) : 
  ∃ (d : ℝ × ℝ), d = correct_answer :=
sorry

end find_d_l71_71430


namespace cosine_identity_l71_71978

theorem cosine_identity (alpha : ℝ) (h1 : -180 < alpha ∧ alpha < -90)
  (cos_75_alpha : Real.cos (75 * Real.pi / 180 + alpha) = 1 / 3) :
  Real.cos (15 * Real.pi / 180 - alpha) = -2 * Real.sqrt 2 / 3 := by
sorry

end cosine_identity_l71_71978


namespace expand_polynomial_l71_71205

theorem expand_polynomial (x : ℝ) :
    (5*x^2 + 3*x - 7) * (4*x^3) = 20*x^5 + 12*x^4 - 28*x^3 :=
by
  sorry

end expand_polynomial_l71_71205


namespace cosine_law_l71_71317

theorem cosine_law (a b c x : ℝ) (hx : x > 0) 
  (ha : a = 2 * x) (hb : b = 3 * x) (hc : c = 4 * x) :
  Real.cos (Real.acos ((a^2 + b^2 - c^2) / (2 * a * b))) = -1 / 4 := by
  sorry

end cosine_law_l71_71317


namespace geometric_sequence_property_l71_71661

theorem geometric_sequence_property (n : ℕ) (S P M : ℝ) 
  (h_pos : ∀ i, 0 < i ∧ i ≤ n → 0 < (r^i))
  (h_sum : S = ∑ i in finset.range n, r^i)
  (h_prod : P = ∏ i in finset.range n, r^i)
  (h_recip_sum : M = ∑ i in finset.range n, (r^i)⁻¹) :
  P^2 = (S / M)^n := 
sorry

end geometric_sequence_property_l71_71661


namespace line_through_point_parallel_l71_71773

theorem line_through_point_parallel
  (p : ℝ × ℝ)
  (h1 : p = (1, 1))
  (h2 : ∃ a b c : ℝ, a * 1 + b * 1 + c = 0 ∧ b ≠ 0 ∧ (a, b, c) = (1, -2, -2)) :
  ∃ a b c : ℝ, a * p.1 + b * p.2 + c = 0 ∧ (a, b, c) = (1, -2, 1) :=
begin
  sorry
end

end line_through_point_parallel_l71_71773


namespace sum_of_roots_eq_14_l71_71040

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l71_71040


namespace inclination_angle_range_correct_l71_71441

noncomputable def inclination_angle_range (α b : ℝ) : set ℝ :=
  let k := - (sin α) / (sqrt 3)
  let θ := arctan k
  if θ ≥ 0 ∧ θ ≤ π
  then if θ ≤ π / 6 ∨ (θ ≥ 5 * π / 6 ∧ θ < π)
       then {θ}
       else ∅
  else ∅

theorem inclination_angle_range_correct (α b : ℝ) :
  inclination_angle_range α b = {θ | θ ∈ [0, π / 6] ∪ [5 * π / 6, π)} :=
sorry

end inclination_angle_range_correct_l71_71441


namespace complex_power_sixty_l71_71181

noncomputable def complex_base : ℂ := (1 - Complex.i) / Real.sqrt 2

theorem complex_power_sixty : complex_base ^ 60 = -1 := 
  by
  -- skipping the proof
  sorry

end complex_power_sixty_l71_71181


namespace luke_piles_of_quarters_l71_71728

theorem luke_piles_of_quarters (Q D : ℕ) 
  (h1 : Q = D) -- number of piles of quarters equals number of piles of dimes
  (h2 : 3 * Q + 3 * D = 30) -- total number of coins is 30
  : Q = 5 :=
by
  sorry

end luke_piles_of_quarters_l71_71728


namespace sum_of_roots_of_equation_l71_71077

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l71_71077


namespace joan_spent_on_shorts_l71_71355

theorem joan_spent_on_shorts :
  ∀ (j s t : ℝ), j = 14.82 → s = 12.51 → t = 42.33 → t - j - s = 15.00 :=
by
  intros j s t hj hs ht
  rw [hj, hs, ht]
  norm_num
  sorry

end joan_spent_on_shorts_l71_71355


namespace quadratic_two_equal_real_roots_c_l71_71969

theorem quadratic_two_equal_real_roots_c (c : ℝ) : 
  (∃ x : ℝ, (2*x^2 - x + c = 0) ∧ (∃ y : ℝ, y ≠ x ∧ 2*y^2 - y + c = 0)) →
  c = 1/8 :=
sorry

end quadratic_two_equal_real_roots_c_l71_71969


namespace max_blue_drummers_l71_71907

theorem max_blue_drummers (n : ℕ) (hn : n = 50) :
  ∃ (blue_positions : fin n.succ → fin n.succ → bool),
  (∀ i j, blue_positions i j = tt → 
    ∀ k l, (i ≠ k ∨ j ≠ l) ∧ (i = k ∨ j = l) → blue_positions k l = ff) ∧
  (∑ i in fin.succ_range n, ∑ j in fin.succ_range n,
      if blue_positions i j = tt then 1 else 0 = 625) := 
by
  sorry

end max_blue_drummers_l71_71907


namespace time_to_cross_is_30_seconds_l71_71528

variable (length_train : ℕ) (speed_km_per_hr : ℕ) (length_bridge : ℕ)

def total_distance := length_train + length_bridge

def speed_m_per_s := (speed_km_per_hr * 1000 : ℕ) / 3600

def time_to_cross_bridge := total_distance length_train length_bridge / speed_m_per_s speed_km_per_hr

theorem time_to_cross_is_30_seconds 
  (h_train_length : length_train = 140)
  (h_train_speed : speed_km_per_hr = 45)
  (h_bridge_length : length_bridge = 235) :
  time_to_cross_bridge length_train speed_km_per_hr length_bridge = 30 :=
by
  sorry

end time_to_cross_is_30_seconds_l71_71528


namespace no_fraction_increases_by_20_percent_l71_71929

theorem no_fraction_increases_by_20_percent (x y : ℕ) (hx : Nat.coprime x y) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (↑(x + 1) / ↑(y + 1) = 1.2 * ↑x / ↑y) → false :=
by
  have h : 5 * y * (x + 1) = 6 * x * (y + 1) := sorry
  have h' : 5 * y * x + 5 * y = 6 * x * y + 6 * x := sorry
  have h'' : 5 * y - 6 * x = 6 * x - 5 * y := sorry
  have h''' : 11 * x - 5 * y = 0 := sorry
  -- Further demonstration that no such x and y exist.
  sorry

end no_fraction_increases_by_20_percent_l71_71929


namespace expected_coincidence_proof_l71_71492

noncomputable def expected_coincidences (total_questions : ℕ) (vasya_correct : ℕ) (misha_correct : ℕ) : ℝ :=
  let vasya_probability := vasya_correct / total_questions
  let misha_probability := misha_correct / total_questions
  let both_correct_probability := vasya_probability * misha_probability
  let both_incorrect_probability := (1 - vasya_probability) * (1 - misha_probability)
  let coincidence_probability := both_correct_probability + both_incorrect_probability
  total_questions * coincidence_probability

theorem expected_coincidence_proof : 
  expected_coincidences 20 6 8 = 10.8 :=
by {
  let total_questions := 20
  let vasya_correct := 6
  let misha_correct := 8
  
  let vasya_probability := 0.3
  let misha_probability := 0.4
  
  let both_correct_probability := vasya_probability * misha_probability
  let both_incorrect_probability := (1 - vasya_probability) * (1 - misha_probability)
  let coincidence_probability := both_correct_probability + both_incorrect_probability
  let expected := total_questions * coincidence_probability
  
  have h1 : vasya_probability = 6 / 20 := by sorry
  have h2 : misha_probability = 8 / 20 := by sorry
  have h3 : both_correct_probability = 0.3 * 0.4 := by sorry
  have h4 : both_incorrect_probability = 0.7 * 0.6 := by sorry
  have h5 : coincidence_probability = 0.54 := by sorry
  have h6 : total_questions * coincidence_probability = 20 * 0.54 := by sorry
  have h7 : 20 * 0.54 = 10.8 := by sorry

  sorry
}

end expected_coincidence_proof_l71_71492


namespace path_length_travelled_l71_71172

noncomputable def path_length (EF : ℝ) : ℝ := 
  let r := EF in
  let circumference := 2 * Real.pi * r in
  let half_circumference := circumference / 2 in
  half_circumference + r

theorem path_length_travelled : path_length (3 / Real.pi) = 3 + 3 / Real.pi := by
  let r := 3 / Real.pi
  let circumference := 2 * Real.pi * r
  let half_circumference := circumference / 2
  have h1 : circumference = 6 := by
    calc
      circumference = 2 * Real.pi * r : by rfl
      _ = 2 * Real.pi * (3 / Real.pi) : by rfl
      _ = 6 : by simp [Real.pi, div_mul_cancel]
  have h2 : half_circumference = 3 := by
    calc
      half_circumference = circumference / 2 : by rfl
      _ = 6 / 2 : by rw h1
      _ = 3 : by norm_num
  have h3 : path_length (3 / Real.pi) = 3 + 3 / Real.pi := by
    calc
      path_length (3 / Real.pi) = half_circumference + r : by rfl
      _ = 3 + 3 / Real.pi : by rw [h2, rfl]
  exact h3

end path_length_travelled_l71_71172


namespace expected_coincidences_l71_71488

-- Definitions for the given conditions
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8

def prob_correct (correct : ℕ) : ℚ := correct / num_questions
def prob_incorrect (correct : ℕ) : ℚ := 1 - prob_correct correct 

def prob_vasya_correct := prob_correct vasya_correct
def prob_vasya_incorrect := prob_incorrect vasya_correct
def prob_misha_correct := prob_correct misha_correct
def prob_misha_incorrect := prob_incorrect misha_correct

-- Probability that both guessed correctly or incorrectly
def prob_both_correct_or_incorrect : ℚ := 
  (prob_vasya_correct * prob_misha_correct) + (prob_vasya_incorrect * prob_misha_incorrect)

-- Expected value for one question being a coincidence
def expected_I_k : ℚ := prob_both_correct_or_incorrect

-- Definition of the total number of coincidences
def total_coincidences : ℚ := num_questions * expected_I_k

-- Proof statement
theorem expected_coincidences : 
  total_coincidences = 10.8 := by
  -- calculation for the expected number
  sorry

end expected_coincidences_l71_71488


namespace book_cost_is_2_l71_71738

-- Define initial amount of money
def initial_amount : ℕ := 48

-- Define the number of books purchased
def num_books : ℕ := 5

-- Define the amount of money left after purchasing the books
def amount_left : ℕ := 38

-- Define the cost per book
def cost_per_book (initial amount_left : ℕ) (num_books : ℕ) : ℕ := (initial - amount_left) / num_books

-- The theorem to prove
theorem book_cost_is_2
    (initial_amount : ℕ := 48) 
    (amount_left : ℕ := 38) 
    (num_books : ℕ := 5) :
    cost_per_book initial_amount amount_left num_books = 2 :=
by
  sorry

end book_cost_is_2_l71_71738


namespace question_1_part_1_question_1_part_2_question_2_l71_71274

universe u

variables (U : Type u) [PartialOrder U]
noncomputable def A : Set ℝ := {x | (x - 2) * (x - 9) < 0}
noncomputable def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
noncomputable def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2 - a }

theorem question_1_part_1 : A ∩ B = {x | 2 < x ∧ x ≤ 5} :=
sorry

theorem question_1_part_2 : B ∪ (Set.compl A) = {x | x ≤ 5 ∨ x ≥ 9} :=
sorry

theorem question_2 (a : ℝ) (h : C a ∪ (Set.compl B) = Set.univ) : a ≤ -3 :=
sorry

end question_1_part_1_question_1_part_2_question_2_l71_71274


namespace meeting_success_probability_l71_71886

noncomputable def meeting_probability : ℝ :=
  let totalVolume := 1.5 ^ 3
  let z_gt_x_y := (1.5 * 1.5 * 1.5) / 3
  let assistants_leave := 2 * ((1.5 * 0.5 / 2) / 3 * 0.5)
  let effectiveVolume := z_gt_x_y - assistants_leave
  let probability := effectiveVolume / totalVolume
  probability

theorem meeting_success_probability :
  meeting_probability = 8 / 27 := by
  sorry

end meeting_success_probability_l71_71886


namespace symmetry_condition_l71_71667

-- Define grid and initial conditions
def grid : Type := ℕ × ℕ
def is_colored (pos : grid) : Prop := 
  pos = (1,4) ∨ pos = (2,1) ∨ pos = (4,2)

-- Conditions for symmetry: horizontal and vertical line symmetry and 180-degree rotational symmetry
def is_symmetric_line (grid_size : grid) (pos : grid) : Prop :=
  (pos.1 <= grid_size.1 / 2 ∧ pos.2 <= grid_size.2 / 2) ∨ 
  (pos.1 > grid_size.1 / 2 ∧ pos.2 <= grid_size.2 / 2) ∨
  (pos.1 <= grid_size.1 / 2 ∧ pos.2 > grid_size.2 / 2) ∨
  (pos.1 > grid_size.1 / 2 ∧ pos.2 > grid_size.2 / 2)

def grid_size : grid := (4, 5)
def add_squares_needed (num : ℕ) : Prop :=
  ∀ (pos : grid), is_symmetric_line grid_size pos → is_colored pos

theorem symmetry_condition : 
  ∃ n, add_squares_needed n ∧ n = 9
  := sorry

end symmetry_condition_l71_71667


namespace circle_radius_tangent_lines_l71_71865

noncomputable def circle_tangent_radius : ℝ := 10 * real.sqrt 34 * 
  (real.sqrt 2 / (real.sqrt 2 - 1 / real.sqrt 17)) - 10 * real.sqrt 2

theorem circle_radius_tangent_lines {k r : ℝ} (h : k > 10) 
    (tangent_yx : abs (k / real.sqrt 2) = r)
    (tangent_ynx : abs (k / real.sqrt 2) = r)
    (tangent_y10 : abs (k - 10) = r)
    (tangent_y_4x : abs (k / real.sqrt 17) = r) :
  r = circle_tangent_radius :=
sorry

end circle_radius_tangent_lines_l71_71865


namespace sum_of_104th_bracket_l71_71906

noncomputable def sequence (n : ℕ) : ℕ := 2 * n + 1

def bracket_size (n : ℕ) : ℕ := (n - 1) % 4 + 1

def bracket_sum (k : ℕ) : ℕ :=
  let start := (4 * (k - 1) * ((4 * (k - 1)) + 1)) / 2
  let size := bracket_size k
  finset.sum (finset.range size) (λ i, sequence (start + i + 1))

theorem sum_of_104th_bracket : bracket_sum 104 = 2072 :=
sorry

end sum_of_104th_bracket_l71_71906


namespace find_radius_of_circle_l71_71449

-- Definitions and conditions
def MN := 1
def MP := 6
def MQ := 2
def α := ∠NMP
def cos_α := 1/4
def sin_α := Real.sqrt 15 / 4
def NP := Real.sqrt (37 - 12 * cos_α)

-- Theorem statement
theorem find_radius_of_circle : ∃ R, R = 2 * Real.sqrt (34 / 15) :=
by
  -- Proving the radius
  sorry

end find_radius_of_circle_l71_71449


namespace major_axis_length_is_4_eccentricity_is_half_l71_71598

-- Define the ellipse E by its equation
def ellipse : Type := { p : ℝ × ℝ // p.2^2 / 4 + p.1^2 / 3 = 1 }

-- Property 1: Length of the major axis is 4
theorem major_axis_length_is_4 : ∀ p ∈ ellipse, 
  let b := 2 in 
  2 * b = 4 := 
by
  -- Proof omitted
  sorry

-- Property 2: The eccentricity of the ellipse is 1/2
theorem eccentricity_is_half : 
  let a := real.sqrt 3 in
  let b := real.sqrt 4 in
  let c := real.sqrt (b^2 - a^2) in
  c / a = 1 / 2 :=
by
  -- Proof omitted
  sorry

end major_axis_length_is_4_eccentricity_is_half_l71_71598


namespace partition_even_friends_even_count_exists_l71_71236

-- Definition of friendship relation and group partition
variables {People : Type} (friends : People → People → Prop)

-- Mutual friendship
axiom mutual_friendship (a b : People) : friends a b ↔ friends b a

-- Number of friends function
def num_friends (G : set People) (p : People) : ℕ :=
finset.filter (λ q, friends p q ∧ q ∈ G) (finset.univ : finset People).card

-- Main theorem statement
theorem partition_even_friends_even_count_exists (n : ℕ) (h_pos : 0 < n) :
  ∃ (k : ℕ), (finset.univ : finset People).card = n ∧ ∀ (G1 G2 : finset People),
    (G1 ∪ G2 = finset.univ ∧ G1 ∩ G2 = ∅ ∧ (∀ p ∈ G1, even (num_friends G1 p)) ∧ (∀ p ∈ G2, even (num_friends G2 p))) ↔ (G1.card + G2.card = 2^k) :=
sorry

end partition_even_friends_even_count_exists_l71_71236


namespace hexagon_diagonals_l71_71665

theorem hexagon_diagonals (n : ℕ) (h : n = 6) : (n * (n - 3)) / 2 = 9 := by
  sorry

end hexagon_diagonals_l71_71665


namespace alcohol_percentage_in_new_solution_l71_71119

theorem alcohol_percentage_in_new_solution :
  let original_volume := 40 -- liters
  let original_percentage_alcohol := 0.05
  let added_alcohol := 5.5 -- liters
  let added_water := 4.5 -- liters
  let original_alcohol := original_percentage_alcohol * original_volume
  let new_alcohol := original_alcohol + added_alcohol
  let new_volume := original_volume + added_alcohol + added_water
  (new_alcohol / new_volume) * 100 = 15 := by
  sorry

end alcohol_percentage_in_new_solution_l71_71119


namespace last_digit_of_large_exponentiation_l71_71115

theorem last_digit_of_large_exponentiation
  (a : ℕ) (b : ℕ)
  (h1 : a = 954950230952380948328708) 
  (h2 : b = 470128749397540235934750230) :
  (a ^ b) % 10 = 4 :=
sorry

end last_digit_of_large_exponentiation_l71_71115


namespace exponent_properties_l71_71601

variables (a : ℝ) (m n : ℕ)
-- Conditions
axiom h1 : a^m = 3
axiom h2 : a^n = 2

-- Goal
theorem exponent_properties :
  a^(m + n) = 6 :=
by
  sorry

end exponent_properties_l71_71601


namespace sum_of_roots_eq_14_l71_71075

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l71_71075


namespace multiples_of_7_units_digit_7_less_than_200_l71_71289

theorem multiples_of_7_units_digit_7_less_than_200 : 
  { n : ℕ // n > 0 ∧ n < 200 ∧ n % 7 = 0 } → { n : ℕ // n < 200 ∧ n % 7 = 0 ∧ n % 10 = 7 }.set.card = 3 :=
by
  sorry -- Proof left as an exercise

end multiples_of_7_units_digit_7_less_than_200_l71_71289


namespace round_to_nearest_whole_l71_71407

theorem round_to_nearest_whole (x : ℝ) (hx : x = 5672.399201) : Int.nearest x = 5672 :=
by
  sorry

end round_to_nearest_whole_l71_71407


namespace exists_natural_numbers_with_digit_sum_condition_l71_71194

def digit_sum (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

theorem exists_natural_numbers_with_digit_sum_condition :
  ∃ (a b c : ℕ), digit_sum (a + b) < 5 ∧ digit_sum (a + c) < 5 ∧ digit_sum (b + c) < 5 ∧ digit_sum (a + b + c) > 50 :=
by
  sorry

end exists_natural_numbers_with_digit_sum_condition_l71_71194


namespace jose_birds_left_l71_71357

-- Define initial conditions
def chickens_initial : Nat := 28
def ducks : Nat := 18
def turkeys : Nat := 15
def chickens_sold : Nat := 12

-- Calculate remaining chickens
def chickens_left : Nat := chickens_initial - chickens_sold

-- Calculate total birds left
def total_birds_left : Nat := chickens_left + ducks + turkeys

-- Theorem statement to prove the number of birds left
theorem jose_birds_left : total_birds_left = 49 :=
by
  -- This is where the proof would typically go
  sorry

end jose_birds_left_l71_71357


namespace find_n_l71_71900

-- Definitions based on conditions
def cube (n : ℕ) :=
  struct {
    side_length : ℕ := n
    painted_sides : ℕ := 2 -- top and front faces painted
  }

-- Theorem statement
theorem find_n (n : ℕ) (c : cube n) (unit_cubes : ℕ := n^3) (painted_faces : ℕ := 2 * n^2) (total_faces : ℕ := 6 * n^3)
  (h : painted_faces = total_faces / 6) : n = 2 :=
begin
  sorry
end

end find_n_l71_71900


namespace mitchell_chews_54_pieces_l71_71382

theorem mitchell_chews_54_pieces (packets : ℕ) (pieces_per_packet : ℕ) (except_pieces : ℕ) (total_pieces : ℕ) :
  packets = 8 → pieces_per_packet = 7 → except_pieces = 2 → total_pieces = packets * pieces_per_packet → total_pieces - except_pieces = 54 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  have h5 : total_pieces = 56 := by linarith
  rw [←h5, h3]
  linarith

end mitchell_chews_54_pieces_l71_71382


namespace total_interest_received_l71_71882

-- Definitions according to the conditions
def principal_b : ℝ := 5000
def principal_c : ℝ := 3000
def time_b : ℝ := 2
def time_c : ℝ := 4
def rate : ℝ := 10 / 100  -- 10% converted to decimal

-- Simple interest formula
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Interest received from B
def interest_b : ℝ := simple_interest principal_b rate time_b

-- Interest received from C
def interest_c : ℝ := simple_interest principal_c rate time_c

-- Total interest
def total_interest : ℝ := interest_b + interest_c

-- Theorem to prove total interest
theorem total_interest_received : total_interest = 2200 := by
  sorry

end total_interest_received_l71_71882


namespace time_to_cross_is_30_seconds_l71_71527

variable (length_train : ℕ) (speed_km_per_hr : ℕ) (length_bridge : ℕ)

def total_distance := length_train + length_bridge

def speed_m_per_s := (speed_km_per_hr * 1000 : ℕ) / 3600

def time_to_cross_bridge := total_distance length_train length_bridge / speed_m_per_s speed_km_per_hr

theorem time_to_cross_is_30_seconds 
  (h_train_length : length_train = 140)
  (h_train_speed : speed_km_per_hr = 45)
  (h_bridge_length : length_bridge = 235) :
  time_to_cross_bridge length_train speed_km_per_hr length_bridge = 30 :=
by
  sorry

end time_to_cross_is_30_seconds_l71_71527


namespace profit_percentage_is_20_percent_l71_71470

-- Definitions based on conditions
def cost_price_per_article : ℝ := 1
def num_articles_sold : ℕ := 50
def num_articles_cost_price : ℕ := 60

-- Calculations derived from the conditions
def total_cost_price := num_articles_sold * cost_price_per_article
def total_selling_price := num_articles_cost_price * cost_price_per_article

-- Profit calculation
def profit := total_selling_price - total_cost_price
def profit_percentage := (profit / total_cost_price) * 100

-- Theorem to prove
theorem profit_percentage_is_20_percent : profit_percentage = 20 := by
  -- This skips the proof
  sorry

end profit_percentage_is_20_percent_l71_71470


namespace weight_of_3_moles_AuCl4_3H2O_is_1178_49_l71_71834

-- Given atomic masses
def atomic_mass_Au : ℝ := 196.97
def atomic_mass_Cl : ℝ := 35.45
def atomic_mass_H : ℝ := 1.01
def atomic_mass_O : ℝ := 16.00

-- Molar mass of AuCl₄·3H₂O
def molar_mass_AuCl4_3H2O : ℝ :=
  atomic_mass_Au + (4 * atomic_mass_Cl) + (6 * atomic_mass_H) + (3 * atomic_mass_O)

-- Weight of 3 moles of AuCl₄·3H₂O
def weight_3_moles_AuCl4_3H2O : ℝ :=
  3 * molar_mass_AuCl4_3H2O

-- Theorem to prove the weight of 3 moles is 1178.49 grams
theorem weight_of_3_moles_AuCl4_3H2O_is_1178_49 :
  weight_3_moles_AuCl4_3H2O = 1178.49 :=
by
  have h_molar_mass : molar_mass_AuCl4_3H2O = 392.83, sorry
  rw h_molar_mass
  norm_num

end weight_of_3_moles_AuCl4_3H2O_is_1178_49_l71_71834


namespace complex_simplify_l71_71743

theorem complex_simplify :
  10.25 * Real.sqrt 6 * Complex.exp (Complex.I * 160 * Real.pi / 180)
  / (Real.sqrt 3 * Complex.exp (Complex.I * 40 * Real.pi / 180))
  = (-Real.sqrt 2 / 2) + Complex.I * (Real.sqrt 6 / 2) := by
  sorry

end complex_simplify_l71_71743


namespace no_real_roots_iff_l71_71311

theorem no_real_roots_iff (k : ℝ) : (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_iff_l71_71311


namespace not_square_of_expression_l71_71464

theorem not_square_of_expression (n : ℕ) (h : n > 0) : ∀ k : ℕ, (4 * n^2 + 4 * n + 4 ≠ k^2) :=
by
  sorry

end not_square_of_expression_l71_71464


namespace green_pill_cost_21_l71_71167

noncomputable theory
open_locale classical

variables (price_total : ℝ) (days : ℕ) (price_green price_pink : ℝ)

-- Given Conditions
def three_weeks := 3 * 7 = days
def green_more_than_pink := price_green = price_pink + 3
def total_cost := price_total = 819
def days_total := days = 21

-- Theorem to prove
theorem green_pill_cost_21 (H : three_weeks) (H1 : green_more_than_pink)
    (H2 : total_cost) (H3 : days_total): price_green = 21 :=
by sorry

end green_pill_cost_21_l71_71167


namespace second_team_mushroom_soup_l71_71385

theorem second_team_mushroom_soup (r a c : ℕ) (h_r : r = 280) (h_a : a = 90) (h_c : c = 70) : 
  r - (a + c) = 120 := by 
  rw [h_r, h_a, h_c]
  norm_num
  sorry

end second_team_mushroom_soup_l71_71385


namespace election_candidates_at_least_two_l71_71809

def number_of_candidates (w_votes : ℕ) (percent_votes : ℚ) (winning_margin : ℕ) : ℕ := 
  if percent_votes > 0 ∧ percent_votes < 1 then 
      let total_votes := w_votes / percent_votes in 
      let runner_up_votes := w_votes - winning_margin in
      if runner_up_votes > 0 then 2 else 1 -- considering at least the winner and runner-up
  else 0

theorem election_candidates_at_least_two (w_votes : ℕ) (percent_votes : ℚ) (winning_margin : ℕ) :
    w_votes = 992 →
    percent_votes = 0.62 →
    winning_margin = 384 →
    number_of_candidates w_votes percent_votes winning_margin ≥ 2 :=
sorry

end election_candidates_at_least_two_l71_71809


namespace sum_of_roots_eq_14_l71_71068

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l71_71068


namespace deductive_vs_inductive_l71_71442

def is_inductive_reasoning (stmt : String) : Prop :=
  match stmt with
  | "C" => True
  | _ => False

theorem deductive_vs_inductive (A B C D : String) 
  (hA : A = "All trigonometric functions are periodic functions, sin(x) is a trigonometric function, therefore sin(x) is a periodic function.")
  (hB : B = "All odd numbers cannot be divided by 2, 525 is an odd number, therefore 525 cannot be divided by 2.")
  (hC : C = "From 1=1^2, 1+3=2^2, 1+3+5=3^2, it follows that 1+3+…+(2n-1)=n^2 (n ∈ ℕ*)")
  (hD : D = "If two lines are parallel, the corresponding angles are equal. If ∠A and ∠B are corresponding angles of two parallel lines, then ∠A = ∠B.") :
  is_inductive_reasoning C :=
by
  sorry

end deductive_vs_inductive_l71_71442


namespace find_angle_measure_find_triangle_area_l71_71254

variable (A B C : ℝ)
variable (a b c : ℝ) (AD : ℝ)
variable (sin : ℝ → ℝ)
variable (cos : ℝ → ℝ)
variable (sqrt : ℝ → ℝ)

hypothesis (h1 : ∃ A B C a b c, 
  a = sqrt 7 ∧
  AD = 2 / 3 ∧
  (sin B + sin C) / (sin A - sin C) = (sin A + sin C) / sin B
)

theorem find_angle_measure
  (sinB : sin B)
  (sinC : sin C)
  (sinA : sin A)
  (cosA : cos A)
  (h2 : A = arccos 0.5)
  :
  A = 2 * arccos 0.5 / 3 :=
sorry

theorem find_triangle_area
  (area : ℝ)
  (b_area : ℝ)
  (c_area : ℝ)
  (sinA : sin A)
  (sqrt3 : sqrt 3)
  :
  area = (b_area * c_area * (sinA / 2)) :=
sorry

end find_angle_measure_find_triangle_area_l71_71254


namespace general_formula_for_an_l71_71444

-- Definitions and conditions given in the problem statement
def is_positive_sequence (a : ℕ → ℝ) : Prop := ∀ n, 0 < a n
def sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = ∑ i in Finset.range (n + 1), a i
def forms_arithmetic_sequence (x y z : ℝ) : Prop := 2 * y = x + z

-- Formal statement of the proof problem
theorem general_formula_for_an (a S : ℕ → ℝ) (h_pos : is_positive_sequence a)
    (h_sum : sequence_sum a S) 
    (h_arith : ∀ n, forms_arithmetic_sequence (2 * a n) (2 * S n) (a n ^ 2)) : 
    ∀ n, a n = 2 * n :=
by
  sorry

end general_formula_for_an_l71_71444


namespace max_pivot_points_l71_71514

-- Definition of a pivot point in a polyhedron
def is_pivot_point (P : Point ℝ^3) (polyhedron : Polyhedron ℝ^3) : Prop :=
  ∀ line : Line ℝ^3, (contains_line P line) → (∃!v: Vertex, (vertex_in_line v line ∧ vertex_in_polyhedron v polyhedron))

-- Theorem stating the maximum number of pivot points in a convex polyhedron
theorem max_pivot_points (polyhedron : Polyhedron ℝ^3) : 
  is_convex polyhedron → ∃! P : Point ℝ^3, is_pivot_point P polyhedron :=
by
  sorry

end max_pivot_points_l71_71514


namespace vector_equation_solution_l71_71565

theorem vector_equation_solution :
  ∃ (u v : ℚ), 
    u = -3 / 11 ∧ v = -16 / 11 ∧
    (⟨3 + u * 5, -1 + u * -3⟩ : ℚ × ℚ) = ⟨0 + v * -3, 2 + v * 4⟩ :=
by
  use (-3 / 11)
  use (-16 / 11)
  split
  { reflexivity }
  split
  { reflexivity }
  congr
  sorry

end vector_equation_solution_l71_71565


namespace binom_p_adic_expansion_mod_p_l71_71372

open Nat

theorem binom_p_adic_expansion_mod_p (a b p : ℕ) (a_coeffs b_coeffs : ℕ → ℕ) (m : ℕ) 
  [Fact (Nat.Prime p)]
  (ha : a = ∑ i in finset.range (m + 1), a_coeffs i * p^i)
  (hb : b = ∑ i in finset.range (m + 1), b_coeffs i * p^i)
  : binom b a % p = (∏ i in finset.range (m + 1), binom (b_coeffs i) (a_coeffs i)) % p :=
sorry

end binom_p_adic_expansion_mod_p_l71_71372


namespace rhombus_diagonal_l71_71769

theorem rhombus_diagonal (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 10) (h2 : area = 60) : 
  d1 = 12 :=
by 
  have : (d1 * d2) / 2 = area := sorry
  sorry

end rhombus_diagonal_l71_71769


namespace distance_from_P_to_focus_l71_71612

-- Define the parabola equation and the definition of the point P
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the given condition that P's distance to the x-axis is 12
def point_P (x y : ℝ) : Prop := parabola x y ∧ |y| = 12

-- The Lean proof problem statement
theorem distance_from_P_to_focus :
  ∃ (x y : ℝ), point_P x y → dist (x, y) (4, 0) = 13 :=
by {
  sorry   -- proof to be completed
}

end distance_from_P_to_focus_l71_71612


namespace sum_of_roots_l71_71016

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l71_71016


namespace car_rental_cost_per_mile_l71_71133

theorem car_rental_cost_per_mile
    (daily_rental_cost : ℕ)
    (daily_budget : ℕ)
    (miles_limit : ℕ)
    (cost_per_mile : ℕ) :
    daily_rental_cost = 30 →
    daily_budget = 76 →
    miles_limit = 200 →
    cost_per_mile = (daily_budget - daily_rental_cost) * 100 / miles_limit →
    cost_per_mile = 23 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end car_rental_cost_per_mile_l71_71133


namespace right_triangle_acute_angle_l71_71679

-- Defining the conditions:
def is_right_triangle (ABC : Triangle) : Prop := ABC.is_right_triangle
def angle_A_is_50_degrees (A B C : Point) : Prop := ∠A = 50

-- Defining the right angle and acute angles in the right triangle
def right_triangle_sum (A B C : Point) (h: is_right_triangle ABC) : Prop :=
  ∠A + ∠B = 90

-- The proof problem: Given the conditions, prove the answer.
theorem right_triangle_acute_angle (A B C : Point)
  (h1 : is_right_triangle ABC)
  (h2 : angle_A_is_50_degrees A B C) :
  ∠B = 40 :=
by
  -- Proof here
  sorry

end right_triangle_acute_angle_l71_71679


namespace sufficient_condition_quadratic_l71_71229

theorem sufficient_condition_quadratic :
  ∃ x : ℝ, (x = 1) ∧ (x^2 - x - 6 < 0) :=
begin
  use 1,
  split,
  { refl }, -- This confirms x = 1
  { calc
    1^2 - 1 - 6 = 1 - 1 - 6 : by norm_num
            ... = -6         : by norm_num
            ... < 0          : by norm_num, },
end

end sufficient_condition_quadratic_l71_71229


namespace negation_of_proposition_l71_71434

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, 0 ≤ x ∧ (x^2 - 2*x - 3 = 0)) ↔ (∀ x : ℝ, 0 ≤ x → (x^2 - 2*x - 3 ≠ 0)) := 
by 
  sorry

end negation_of_proposition_l71_71434


namespace binomial_coeff_x_squared_l71_71682

theorem binomial_coeff_x_squared (x : ℝ) :
  (∑ r in Finset.range 7, (Nat.choose 6 r) * ((sqrt x / 2) ^ (6 - r)) * ((- 2 / sqrt x) ^ r)) =
  ((-2) * ((1 / 2)^5) * (Nat.choose 6 1)) * x ^ 2 + 
  ∑ r in (Finset.range 7).filter (λ r, r ≠ 1), (Nat.choose 6 r) * ((sqrt x / 2) ^ (6 - r)) * ((- 2 / sqrt x) ^ r) :=
begin
  sorry
end

end binomial_coeff_x_squared_l71_71682


namespace average_of_scores_l71_71473

theorem average_of_scores :
  let scores := [50, 60, 70, 80, 80]
  let total := 340
  let num_subjects := 5
  let average := total / num_subjects
  average = 68 :=
by
  sorry

end average_of_scores_l71_71473


namespace quadratic_polynomial_exists_l71_71567

theorem quadratic_polynomial_exists (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ p : ℝ → ℝ, (∀ x, p x = (a^2 + ab + b^2 + ac + bc + c^2) * x^2 
                   - (a + b) * (b + c) * (a + c) * x 
                   + abc * (a + b + c))
              ∧ p a = a^4 
              ∧ p b = b^4 
              ∧ p c = c^4 := 
sorry

end quadratic_polynomial_exists_l71_71567


namespace max_area_of_quadrilateral_l71_71575

theorem max_area_of_quadrilateral (a b c d : ℝ) (A B C D : Quadrilateral) (area : ℝ)
  (hA : A.side1 = 1) (hB : A.side2 = 3) (hC : A.side3 = 6) (hD : A.side4 = 8) :
  area = 12 :=
begin
  sorry
end

end max_area_of_quadrilateral_l71_71575


namespace balloon_ways_is_seven_times_pot_ways_l71_71496

open Set

def numberOfBalloonWays : ℕ :=
  7 * 6 ^ 24 - ∑ m in (finset.range 7), (-1 : ℤ) ^ m * finset.card (finset.image (λ s : finset ℕ, m) (finset.powerset (finset.range 7))) * (m * (m - 1) ^ 24)

def numberOfPotWays : ℕ :=
  6 ^ 24 - ∑ m in (finset.range 6), (-1 : ℤ) ^ m * finset.card (finset.image (λ s : finset ℕ, m) (finset.powerset (finset.range 6))) * m ^ 24

theorem balloon_ways_is_seven_times_pot_ways :
  numberOfBalloonWays = 7 * numberOfPotWays :=
by
  -- Proof steps are omitted


-- Placeholder to declare the statement without proof
sorry

end balloon_ways_is_seven_times_pot_ways_l71_71496


namespace correct_operation_l71_71887

variable (N : ℚ) -- Original number (assumed rational for simplicity)
variable (x : ℚ) -- Unknown multiplier

theorem correct_operation (h : (N / 10) = (5 / 100) * (N * x)) : x = 2 :=
by
  sorry

end correct_operation_l71_71887


namespace sum_of_roots_eq_14_l71_71066

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l71_71066


namespace sum_of_roots_eq_14_l71_71097

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l71_71097


namespace bookstore_branch_difference_l71_71132

theorem bookstore_branch_difference (x y : ℕ) 
  (h1 : x + y = 5000)
  (h2 : y + 400 = (x - 400) / 2 - 400) :
  |x - y| = 3000 :=
by {
  sorry
}

end bookstore_branch_difference_l71_71132


namespace relationship_between_x_and_y_l71_71295

theorem relationship_between_x_and_y (m x y : ℝ) (h1 : x = 3 - m) (h2 : y = 2 * m + 1) : 2 * x + y = 7 :=
sorry

end relationship_between_x_and_y_l71_71295


namespace true_discount_correct_l71_71422

noncomputable def find_true_discount (BD SD : ℕ) (hBD : BD = 288) (hSD : SD = 1440) : ℕ :=
let D := 246 in
D

theorem true_discount_correct (BD SD : ℕ) (hBD : BD = 288) (hSD : SD = 1440) : 
  let TD := find_true_discount BD SD hBD hSD in
  BD = TD + TD^2 / SD → TD = 246 :=
by
  intro h1
  simp [find_true_discount, h1]
  sorry

end true_discount_correct_l71_71422


namespace tail_flip_probability_after_six_heads_l71_71700

-- Define the problem conditions
def john_flips_six_heads : Prop := true -- This just stands for the fact that John flipped six heads, which does not affect future flips.

-- Define the probability of flipping a tail on a fair coin
def prob_tail : ℚ := 1 / 2

-- State the theorem to be proven
theorem tail_flip_probability_after_six_heads (h : john_flips_six_heads) : 
  Prob (fair_coin = tail) = prob_tail := 
sorry

end tail_flip_probability_after_six_heads_l71_71700


namespace incorrect_statement_B_l71_71841

theorem incorrect_statement_B :
  (∀ (x y : ℝ), (∃ ε : ℝ, y = f x + ε) → correlation x y) ∧
  (∀ r : ℝ, abs r ≤ 1 → (if abs r = 1 then strong_linear_correlation else weak_linear_correlation)) ∧
  (∀ residuals : list ℝ, (narrow_band residuals) → higher_accuracy_model residuals) ∧
  (∀ (R1 R2 : ℝ), R1 > R2 → better_fit (model R1) (model R2)) →
  ¬(∀ r : ℝ, r > 0 → strong_linear_correlation r) :=
by
  sorry

end incorrect_statement_B_l71_71841


namespace time_to_pass_jogger_l71_71878

noncomputable def jogger_speed_kmh : ℕ := 9
noncomputable def jogger_speed_ms : ℝ := jogger_speed_kmh * 1000 / 3600
noncomputable def train_length : ℕ := 130
noncomputable def jogger_ahead_distance : ℕ := 240
noncomputable def train_speed_kmh : ℕ := 45
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
noncomputable def relative_speed : ℝ := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_to_cover : ℕ := jogger_ahead_distance + train_length
noncomputable def time_taken_to_pass : ℝ := total_distance_to_cover / relative_speed

theorem time_to_pass_jogger : time_taken_to_pass = 37 := sorry

end time_to_pass_jogger_l71_71878


namespace max_sum_S_at_16_l71_71793

variable (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

-- Define the conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def b_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = a n * a (n + 1) * a (n + 2)

def sum_b_sequence (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in range n, b i

def given_condition (a : ℕ → ℝ) : Prop :=
  a 12 = (3 / 8) * a 5 ∧ a 5 > 0

-- Prove that the sum S_n reaches its maximum at n = 16
theorem max_sum_S_at_16 (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_sequence a →
  b_sequence a b →
  sum_b_sequence b S →
  given_condition a →
  ∀ n : ℕ, S n ≤ S 16 := 
by
  intros arith_seq b_seq sum_seq cond
  sorry

end max_sum_S_at_16_l71_71793


namespace sum_of_digits_repeating_fraction_l71_71443

theorem sum_of_digits_repeating_fraction (c d : ℕ) (h : (5 : ℚ) / 13 = 0.cdcdcdc...) : c + d = 11 :=
by
  sorry

end sum_of_digits_repeating_fraction_l71_71443


namespace money_last_weeks_l71_71398

theorem money_last_weeks (a b c : ℕ) (h_a : a = 44) (h_b : b = 28) (h_c : c = 9) : 
  (a + b) / c = 8 :=
by
  rw [h_a, h_b, h_c]
  exact eq.refl 8

end money_last_weeks_l71_71398


namespace ratio_pen_to_pencil_l71_71145

-- Define the costs
def cost_of_pencil (P : ℝ) : ℝ := P
def cost_of_pen (P : ℝ) : ℝ := 4 * P
def total_cost (P : ℝ) : ℝ := cost_of_pencil P + cost_of_pen P

-- The proof that the total cost of the pen and pencil is $6 given the provided ratio
theorem ratio_pen_to_pencil (P : ℝ) (h_total_cost : total_cost P = 6) (h_pen_cost : cost_of_pen P = 4) :
  cost_of_pen P / cost_of_pencil P = 4 :=
by
  -- Proof skipped
  sorry

end ratio_pen_to_pencil_l71_71145


namespace det_min_matrix_l71_71703

open Matrix

def min (a b : ℕ) : ℕ := if a ≤ b then a else b

noncomputable def compute_det_matrix (n : ℕ) : Matrix (Fin n) (Fin n) ℝ :=
  λ i j, 1 / (min i j).val

theorem det_min_matrix (n : ℕ) : 
  (compute_det_matrix n).det = (-1)^(n - 1) / (n - 1)! / n! :=
sorry

end det_min_matrix_l71_71703


namespace boy_age_is_10_l71_71109

-- Define the boy's current age as a variable
def boy_current_age := 10

-- Define a condition based on the boy's statement
def boy_statement_condition (x : ℕ) : Prop :=
  x = 2 * (x - 5)

-- The main theorem stating equivalence of the boy's current age to 10 given the condition
theorem boy_age_is_10 (x : ℕ) (h : boy_statement_condition x) : x = boy_current_age := by
  sorry

end boy_age_is_10_l71_71109


namespace inequality_sum_sqrt_l71_71717

theorem inequality_sum_sqrt
    (n k : ℕ) (hn : 0 < n) (hk : 0 < k) 
    (a : Fin n → ℕ) (h_bound : ∀ i, 1 ≤ a i ∧ a i ≤ 2^k) : 
    ∑ i in Finset.range n, a i / Real.sqrt (∑ j in Finset.range (i + 1), (a j)^2) ≤ 4 * Real.sqrt (k * n) := by 
  sorry

end inequality_sum_sqrt_l71_71717


namespace number_of_four_digit_numbers_l71_71805

-- We use a broader import for better compatibility

def cards : List ℕ := [2, 0, 0, 9]

theorem number_of_four_digit_numbers (h : 9 ≃ 6) : ℕ :=
  by sorry

#eval number_of_four_digit_numbers ⟨⟨6, by decide⟩⟩  -- should return 6

end number_of_four_digit_numbers_l71_71805


namespace sum_of_roots_l71_71108

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l71_71108


namespace original_price_of_apples_l71_71647

-- Define the conditions and problem
theorem original_price_of_apples 
  (discounted_price : ℝ := 0.60 * original_price)
  (total_cost : ℝ := 30)
  (weight : ℝ := 10) :
  original_price = 5 :=
by
  -- This is the point where the proof steps would go.
  sorry

end original_price_of_apples_l71_71647


namespace f_2017_eq_neg_two_l71_71366

noncomputable def f (x : ℝ) : ℝ :=
if (-1 ≤ x ∧ x < 0) then log (2, -3 * x + 1)
else if (∃ k : ℤ, x = 4 * k + 1) then -log(2, 4)
else 0 -- for simplicity, defined for other values, these aren't needed for our proof

lemma odd_f : ∀ x : ℝ, f (-x) = -f x :=
sorry -- given that f is an odd function

lemma f_two_sub_x : ∀ x : ℝ, f (2 - x) = f x :=
sorry -- given that f(2 - x) = f(x)

lemma periodic_f : ∀ x : ℝ, f (4 + x) = f x :=
sorry -- derived property that f has period of 4

lemma f_neg_one : f (-1) = log (2, 4) :=
by
  have : f(-1) = log(2, 4), 
  from if_pos (and.intro (by norm_num) (by norm_num)),
  exact this

theorem f_2017_eq_neg_two : f 2017 = -2 :=
by
  have period : ∀ n : ℕ, f (n * 4 + 1) = f 1,
  from λ n, by rw [← periodic_f (n * 4), ← periodic_f, ← periodic_f, ← periodic_f],
  
  have f_1 : f 1 = -f (-1),
  from eq.symm (f_two_sub_x (-1)),
  
  have f_1_value : f 1 = -log(2, 4),
  from eq.symm f_neg_one,
  
  rw period 504 at f_1_value,
  exact f_1_value

end f_2017_eq_neg_two_l71_71366


namespace constant_term_of_expansion_l71_71774

theorem constant_term_of_expansion (x : ℝ) : 
  ∃ (n : ℕ), 
    (n = 10) ∧
    let T := (λ r : ℕ, Nat.choose n r * 2^r * x^(5 - 5 * r / 2)) in 
    T 2 = 180 := 
begin
  use 10,
  split,
  { refl, },
  { sorry, }
end

end constant_term_of_expansion_l71_71774


namespace area_increase_l71_71416

theorem area_increase (original_side : ℝ) (increase : ℝ) : 
  original_side = 6 ∧ increase = 1 → 
  let new_side := original_side + increase in
  let original_area := original_side * original_side in
  let new_area := new_side * new_side in
  new_area - original_area = 13 :=
by
  intros h
  let new_side := original_side + increase
  let original_area := original_side * original_side
  let new_area := new_side * new_side
  have original_side_eq : original_side = 6 := h.1
  have increase_eq : increase = 1 := h.2
  rw [original_side_eq, increase_eq] at *
  dsimp at *
  rw [show new_side = 7, by ring]
  rw [show original_area = 36, by ring]
  rw [show new_area = 49, by ring]
  sorry

end area_increase_l71_71416


namespace coin_toss_sequence_count_l71_71327

def HH_subsequence (seq : list char) : ℕ := sorry -- counts HH subsequences
def HT_subsequence (seq : list char) : ℕ := sorry -- counts HT subsequences
def TH_subsequence (seq : list char) : ℕ := sorry -- counts TH subsequences
def TT_subsequence (seq : list char) : ℕ := sorry -- counts TT subsequences

theorem coin_toss_sequence_count : 
  ∀ (seq : list char),
    seq.length = 17 →
    HH_subsequence seq = 3 →
    HT_subsequence seq = 4 →
    TH_subsequence seq = 3 →
    TT_subsequence seq = 6 →
    ∃ (n : ℕ), n = 4200 := 
by
  sorry

end coin_toss_sequence_count_l71_71327


namespace possible_distance_AG_l71_71905

theorem possible_distance_AG : 
  (∀ (AB VG : ℝ), AB = 600 → VG = 600 → 
  ∃ (d : ℝ), AG = 3 * d ∧ (AG = 900 ∨ AG = 1800)) :=
begin
  -- Conditions
  intros AB VG hAB hVG,
  -- Ensuring distances are correct based on given conditions
  have h1 : AB = 600 := hAB,
  have h2 : VG = 600 := hVG,
  -- Introducing d as the distance between B and V
  have h3 : ∃ (d : ℝ), AG = 3 * d,
  -- Possible values for AG given constraints
  have h4 : AG = 900 ∨ AG = 1800,
  sorry,
end

end possible_distance_AG_l71_71905


namespace polynomial_evaluation_l71_71702

theorem polynomial_evaluation (p q : Polynomial ℤ) (h : Polynomial.monic p ∧ Polynomial.monic q ∧ ∀ x, x^8 - 50*x^4 + 49 = p.eval x * q.eval x ∧ p.natDegree > 0 ∧ q.natDegree > 0) :
  p.eval 1 + q.eval 1 = 100 := 
sorry

end polynomial_evaluation_l71_71702


namespace solve_system_l71_71756

theorem solve_system :
  ∃ x y : ℝ, ((0.5 * log 2 x - log 2 y = 0) ∧ (x^2 - 2 * y^2 = 8)) ∧ 
             ((x = 4) ∧ (y = 2) ∨ (x = 4) ∧ (y = -2)) :=
by
  sorry

end solve_system_l71_71756


namespace marble_draw_probability_l71_71179

def marble_probabilities : ℚ :=
  let prob_white_a := 5 / 10
  let prob_black_a := 5 / 10
  let prob_yellow_b := 8 / 15
  let prob_yellow_c := 3 / 10
  let prob_green_d := 6 / 10
  let prob_white_then_yellow_then_green := prob_white_a * prob_yellow_b * prob_green_d
  let prob_black_then_yellow_then_green := prob_black_a * prob_yellow_c * prob_green_d
  prob_white_then_yellow_then_green + prob_black_then_yellow_then_green

theorem marble_draw_probability :
  marble_probabilities = 17 / 50 := by
  sorry

end marble_draw_probability_l71_71179


namespace tan_22_5_sum_l71_71791

noncomputable def a_value : ℕ := 2
noncomputable def b_value : ℕ := 1
noncomputable def c_value : ℕ := 0
noncomputable def d_value : ℕ := 0

theorem tan_22_5_sum :
  let a := a_value,
      b := b_value,
      c := c_value,
      d := d_value in
  a + b + c + d = 3 := by
  sorry

end tan_22_5_sum_l71_71791


namespace sam_coupons_l71_71409

/-- Sam buys 9 cans, each costing 175 cents. He uses some coupons, each giving a 25 cent discount.
He pays $20 and receives $5.50 in change. Show that he has 5 coupons. -/
theorem sam_coupons 
  (num_cans : ℕ) 
  (cost_per_can : ℕ) 
  (coupon_discount : ℕ)
  (amount_paid : ℚ) 
  (change_received : ℚ) 
  (num_coupons : ℕ) 
  (total_spent : ℚ)
  (total_cost_without_coupons : ℚ)
  (savings : ℚ)
  (total_cost_after_coupons : ℚ)
  (num_coupons_calculated : ℕ) :
  num_cans = 9 ∧
  cost_per_can = 175 ∧
  coupon_discount = 25 ∧
  amount_paid = 20 ∧
  change_received = 5.5 ∧
  total_spent = 14.5 ∧
  total_cost_without_coupons = 1575 ∧
  savings = 125 ∧
  total_cost_after_coupons = 1450 ∧
  num_coupons_calculated = (savings / coupon_discount).nat_abs →
  num_coupons_calculated = 5 :=
by {
  -- Proof would go here, but we'll just assume it's correct for now.
  sorry
}

end sam_coupons_l71_71409


namespace total_students_playing_sports_students_who_like_basketball_or_cricket_or_both_l71_71846

-- Define the number of students as natural numbers (ℕ)
def B : ℕ := 7
def C : ℕ := 5
def B_inter_C : ℕ := 3

-- The proof problem statement: Prove the inclusion-exclusion principle in this context
theorem total_students_playing_sports : B ∪ C = B + C - B_inter_C := by sorry

-- Assertion for the current problem
theorem students_who_like_basketball_or_cricket_or_both (B C B_inter_C : ℕ) :
  B + C - B_inter_C = 9 :=
by {
  have h1 : B = 7 := rfl,
  have h2 : C = 5 := rfl,
  have h3 : B_inter_C = 3 := rfl,
  rw [h1, h2, h3],
  exact rfl
}

end total_students_playing_sports_students_who_like_basketball_or_cricket_or_both_l71_71846


namespace expected_coincidences_l71_71490

-- Definitions for the given conditions
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8

def prob_correct (correct : ℕ) : ℚ := correct / num_questions
def prob_incorrect (correct : ℕ) : ℚ := 1 - prob_correct correct 

def prob_vasya_correct := prob_correct vasya_correct
def prob_vasya_incorrect := prob_incorrect vasya_correct
def prob_misha_correct := prob_correct misha_correct
def prob_misha_incorrect := prob_incorrect misha_correct

-- Probability that both guessed correctly or incorrectly
def prob_both_correct_or_incorrect : ℚ := 
  (prob_vasya_correct * prob_misha_correct) + (prob_vasya_incorrect * prob_misha_incorrect)

-- Expected value for one question being a coincidence
def expected_I_k : ℚ := prob_both_correct_or_incorrect

-- Definition of the total number of coincidences
def total_coincidences : ℚ := num_questions * expected_I_k

-- Proof statement
theorem expected_coincidences : 
  total_coincidences = 10.8 := by
  -- calculation for the expected number
  sorry

end expected_coincidences_l71_71490


namespace number_of_correct_propositions_is_zero_l71_71169

theorem number_of_correct_propositions_is_zero 
  (h1 : ∀α β : ℂ, α = α.re ∧ β = β.re → (α ≤ β ∨ β ≤ α))
  (h2 : ((complex.i - 1).re, (complex.i - 1).im) = (-1, 1))
  (h3 : ∀ x : ℝ, (x^2 - 1 = 0) ∧ (x^2 + 3 * x + 2 ≠ 0) → x = 1)
  (h4 : ∀ z1 z2 z3 : ℂ, (z1 - z2 = complex.i) ∧ (z2 - z3 = 1) → (z1 - z2)^2 + (z2 - z3)^2 = 0 → False) : 
  0 = 0 := 
by 
  sorry

end number_of_correct_propositions_is_zero_l71_71169


namespace sum_of_roots_eq_14_l71_71095

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l71_71095


namespace correct_proposition_l71_71168

-- Definitions based on given conditions
def P : Prop := sqrt 49 = 7
def Q : Prop := ∀ (a b c d: Type) (quad: a × b × c × d), (diagonals_perpendicular_bisectors quad) → rhombus quad
def R : Prop := ∀ (chord: Type) (diameter: Type), (bisects (diameter, chord)) → perpendicular_to_and_bisects_arcs (diameter, chord)
def S : Prop := ∀ (p : ℝ), ∃ (x y: ℝ), p = (x, y)

-- Problem statement in Lean
theorem correct_proposition : Q :=
by
  sorry

end correct_proposition_l71_71168


namespace sum_n_binom_30_15_eq_31_16_l71_71832

open Nat

-- Given n = 30 and k = 15, we are given the components to test Pascal's identity
def PascalIdentity (n k : Nat) : Prop :=
  Nat.choose (n-1) (k-1) + Nat.choose (n-1) k = Nat.choose n k

theorem sum_n_binom_30_15_eq_31_16 : 
  (∑ n in { n : ℕ | Nat.choose 30 15 + Nat.choose 30 n = Nat.choose 31 16 }, n) = 30 := 
sorry

end sum_n_binom_30_15_eq_31_16_l71_71832


namespace simplify_expression_l71_71113

theorem simplify_expression : 
  (-5) - (-3) + (+1) - (-6) = -5 + 3 + 1 + 6 := 
  by 
    sorry

end simplify_expression_l71_71113


namespace conjugate_of_z_is_1_sub_4i_l71_71232

noncomputable def z : ℂ := (5 + 3 * Complex.i) / (1 - Complex.i)

theorem conjugate_of_z_is_1_sub_4i : Complex.conj z = 1 - 4 * Complex.i :=
sorry

end conjugate_of_z_is_1_sub_4i_l71_71232


namespace bird_legs_l71_71802

theorem bird_legs (n_birds : ℕ) (legs_per_bird : ℕ) (h1 : n_birds = 5) (h2 : legs_per_bird = 2) :
  n_birds * legs_per_bird = 10 :=
by
  rw [h1, h2]
  norm_num
  sorry

end bird_legs_l71_71802


namespace middle_term_is_correct_sum_of_coefficients_excluding_constant_is_correct_term_with_largest_coefficient_is_correct_l71_71221

-- Define the binomial coefficients
noncomputable def binom_coeff (n k : ℕ) : ℤ :=
  if h : k ≤ n then
    (nat.factorial n / (nat.factorial k * nat.factorial (n - k))) 
  else 0

-- Expansion term definition
noncomputable def binom_term (n k : ℕ) (x : ℤ) : ℤ :=
  binom_coeff n k * (-x)^k

-- Define the problem statement conditions
def binomial := (1 - x)^10

-- 1. Definition for middle term
def middle_term (x : ℤ) : ℤ := binom_term 10 5 x

-- 2. Definition for the sum of coefficients excluding the constant term
noncomputable def sum_of_coefficients_excluding_constant (x : ℤ) : ℤ :=
  let sum := (list.range 11).sum (λ k, binom_term 10 k x) in
  sum - 1

-- 3. Definitions for terms with the largest coefficient
def term_with_largest_coefficient (x : ℤ) : ℤ :=
  max (binom_term 10 4 x) (binom_term 10 6 x)

-- Theorem statements as equivalent proof problems
theorem middle_term_is_correct :
  middle_term 1 = -252 * 1^5 := 
by sorry

theorem sum_of_coefficients_excluding_constant_is_correct :
  sum_of_coefficients_excluding_constant 1 = -1 := 
by sorry

theorem term_with_largest_coefficient_is_correct :
  term_with_largest_coefficient 1 = 210 * 1^4 := 
by sorry

end middle_term_is_correct_sum_of_coefficients_excluding_constant_is_correct_term_with_largest_coefficient_is_correct_l71_71221


namespace cosine_inclination_angle_l71_71234

variable (t : ℝ)

def parametric_equation_x : ℝ := -2 + 3 * t
def parametric_equation_y : ℝ := 3 - 4 * t

theorem cosine_inclination_angle :
  let angle : ℝ := real.atan (-4 / 3)
  in real.cos angle = -3 / 5 :=
by
  sorry

end cosine_inclination_angle_l71_71234


namespace moles_of_H2O_formed_l71_71590

def NH4NO3 (n : ℕ) : Prop := n = 1
def NaOH (n : ℕ) : Prop := ∃ m : ℕ, m = n
def H2O (n : ℕ) : Prop := n = 1

theorem moles_of_H2O_formed :
  ∀ (n : ℕ), NH4NO3 1 → NaOH n → H2O 1 := 
by
  intros n hNH4NO3 hNaOH
  exact sorry

end moles_of_H2O_formed_l71_71590


namespace area_of_triangle_inscribed_in_circle_arcs_l71_71161

noncomputable def triangle_area_inscribed_circle (r : ℝ) (arc1 arc2 arc3 : ℝ) : ℝ :=
  let angle1 := 5 * θ in
  let angle2 := 7 * θ in
  let angle3 := 8 * θ in
  0.5 * r^2 * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem area_of_triangle_inscribed_in_circle_arcs 
  (r : ℝ) (arc1 arc2 arc3 : ℝ) (h1 : arc1 = 5) (h2 : arc2 = 7) (h3 : arc3 = 8) :
  triangle_area_inscribed_circle (10 / Real.pi) 5 7 8 = 119.85 / (Real.pi ^ 2) :=
sorry

end area_of_triangle_inscribed_in_circle_arcs_l71_71161


namespace area_of_isosceles_triangle_l71_71817

open Real

-- Define the isosceles triangle with given side lengths
structure TriangleXYZ :=
  (XY : ℝ := 13)
  (YZ : ℝ := 13)
  (XZ : ℝ := 24)
  (isosceles : XY = YZ)

-- The theorem to prove the area of the given triangle
theorem area_of_isosceles_triangle (t : TriangleXYZ) :
  t.XY = 13 ∧ t.YZ = 13 ∧ t.XZ = 24 → 
  (1/2 * t.XZ * (Real.sqrt(t.XY^2 - (t.XZ/2)^2))) = 60 :=
by
  intro h
  cases h with hXY hYZ hXZ
  have : t.XY = 13 := hXY
  have : t.YZ = 13 := hYZ
  have : t.XZ = 24 := hXZ
  sorry

end area_of_isosceles_triangle_l71_71817


namespace petes_original_number_l71_71732

theorem petes_original_number (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = y - 5) (h3 : 3 * z = 96) :
  x = 12.33 :=
by
  -- Proof goes here
  sorry

end petes_original_number_l71_71732


namespace geometric_sequence_sum_l71_71328

variable {a : ℕ → ℕ}

def is_geometric_sequence_with_common_product (k : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem geometric_sequence_sum :
  is_geometric_sequence_with_common_product 27 a →
  a 1 = 1 →
  a 2 = 3 →
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 +
   a 11 + a 12 + a 13 + a 14 + a 15 + a 16 + a 17 + a 18) = 78 :=
by
  intros h_geom h_a1 h_a2
  sorry

end geometric_sequence_sum_l71_71328


namespace probability_heart_or_king_l71_71504

theorem probability_heart_or_king :
  let deck_size := 52
  let hearts := 13
  let kings := 4
  let king_of_hearts := 1
  let total_heart_or_king := hearts + kings - king_of_hearts
  let neither_heart_nor_king := deck_size - total_heart_or_king
  let p_not_heart_nor_king : ℚ := neither_heart_nor_king / deck_size
  let p_neither : ℚ := p_not_heart_nor_king * p_not_heart_nor_king
  let p_at_least_one : ℚ := 1 - p_neither
  in p_at_least_one = 88 / 169 :=
by {
  let deck_size := 52
  let hearts := 13
  let kings := 4
  let king_of_hearts := 1
  let total_heart_or_king := hearts + kings - king_of_hearts
  let neither_heart_nor_king := deck_size - total_heart_or_king
  let p_not_heart_nor_king : ℚ := neither_heart_nor_king / deck_size
  let p_neither : ℚ := p_not_heart_nor_king * p_not_heart_nor_king
  let p_at_least_one : ℚ := 1 - p_neither
  show (1 - ((36 / 52) * (36 / 52))) = 88 / 169, from rfl
}

end probability_heart_or_king_l71_71504


namespace geometric_sequence_properties_sum_of_geometric_sequence_l71_71727

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : Prop :=
  S n = (a 1 * (q ^ n - 1)) / (q - 1)
  
theorem geometric_sequence_properties (a_3 : a 3 = 4) (prod_a456 : a 4 * a 5 * a 6 = 2^12) :
  ∃ a_1 q, a 1 = a_1 ∧ (geometric_sequence a q) ∧ (a_1 = 1) ∧ (q = 2) :=
by sorry

theorem sum_of_geometric_sequence (S_n : S 10 = 2^10 - 1) :
  ∃ n, (sum_of_first_n_terms a S n) ∧ (n = 10) :=
by sorry

end geometric_sequence_properties_sum_of_geometric_sequence_l71_71727


namespace part1_part2_l71_71264

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x^2 - 2*x + 1) + real.sqrt (x^2 - 10*x + 25)

theorem part1 (x : ℝ) : (f x > 6) ↔ (x < 0 ∨ x > 6) :=
sorry

theorem part2 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : 1/a + 1/(2*b) + 1/(3*c) = 1) :
  a + 2*b + 3*c ≥ 9 :=
sorry

end part1_part2_l71_71264


namespace dot_product_isosceles_triangle_l71_71340

theorem dot_product_isosceles_triangle 
  (ABC : Type) [planar_geometry ABC] 
  (A B C : ABC) 
  (h_iso : is_isosceles_triangle A B C)
  (h_angleA : angle B A C = 2 * π / 3)
  (h_lengthBC : distance B C = 2 * √3) :
  (vector B A) • (vector C A) = 2 := 
sorry

end dot_product_isosceles_triangle_l71_71340


namespace polynomial_no_2n_positive_roots_l71_71734

noncomputable def P (x : ℝ) (n : ℕ) : ℝ :=
  x^(2 * n) - 2 * n * x^(2 * n - 1) + 2 * n * x^(2 * n - 2) - 2 * n * x^(2 * n - 3) + -- ... more terms ...
    2 * n * x^2 - 2 * n * x + 1

theorem polynomial_no_2n_positive_roots {n : ℕ} (hn : n ≥ 2) :
  ¬ ∃ roots : fin (2 * n) → ℝ, (∀ i, 0 < roots i) ∧ (∀ i, P (roots i) n = 0) := by sorry

end polynomial_no_2n_positive_roots_l71_71734


namespace problem_solution_l71_71377

-- Definitions of sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Complement within U
def complement_U (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- The proof goal
theorem problem_solution : (complement_U A) ∪ B = {2, 3, 4, 5} := by
  sorry

end problem_solution_l71_71377


namespace range_of_values_l71_71298

theorem range_of_values (x : ℝ) (h1 : x - 1 ≥ 0) (h2 : x ≠ 0) : x ≥ 1 := 
sorry

end range_of_values_l71_71298


namespace f_sum_even_odd_l71_71365

variable (f : ℝ → ℝ)

def g (x : ℝ) : ℝ := 1 / 2 * (f x + f (-x))
def h (x : ℝ) : ℝ := 1 / 2 * (f x - f (-x))

theorem f_sum_even_odd : ∀ x : ℝ, f x = g f x + h f x := by
  sorry

end f_sum_even_odd_l71_71365


namespace binomial_sum_eq_sum_valid_n_values_l71_71829

theorem binomial_sum_eq (n : ℕ) (h₁ : nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16) :
  n = 14 ∨ n = 16 :=
sorry

theorem sum_valid_n_values :
  let n1 := 16
  let n2 := 14
  n1 + n2 = 30 :=
by
  -- proof to be provided; this is to check if the theorem holds
  sorry

end binomial_sum_eq_sum_valid_n_values_l71_71829


namespace allocation_count_l71_71815

def allocate_volunteers (num_service_points : Nat) (num_volunteers : Nat) : Nat :=
  -- Definition that captures the counting logic as per the problem statement
  if num_service_points = 4 ∧ num_volunteers = 6 then 660 else 0

theorem allocation_count :
  allocate_volunteers 4 6 = 660 :=
sorry

end allocation_count_l71_71815


namespace range_of_a_l71_71968

theorem range_of_a (a : ℝ) : (∀ x > 0, a - x - |Real.log x| ≤ 0) → a ≤ 1 := by
  sorry

end range_of_a_l71_71968


namespace pyramid_ratio_l71_71342

structure Pyramid (Point : Type) :=
  (S A B C M A1 B1 C1 : Point)
  (on_base : Point)
  (on_faces : Point)
  (parallels : Point)

axiom similar_triangles (pyramid : Pyramid ℝ) :
  let SABC := \triangle pyramid.S pyramid.A pyramid.B pyramid.C,
      \triangle MAP S := \triangle pyramid.M (edge intersection of lines ++ to make up for axes. Similarity)
  covers all faces including spins on all intersecting points upto faces

theorem pyramid_ratio (pyramid : Pyramid ℝ) (cond1 : M ∈ triangle pyramid.A pyramid.B pyramid.C) 
  (cond2 : ∀ SA SB SC ∈ Lateral_faces pyramid.S pyramid.A pyramid.B pyramid.C):
  \(\frac{MA_{1}}{SA}+\frac{MB_{1}}{SB}+\frac{MC_{1}}{SC}==1\)
by sorry

end pyramid_ratio_l71_71342


namespace introduce_people_no_three_same_acquaintances_l71_71128

theorem introduce_people_no_three_same_acquaintances (n : ℕ) :
  ∃ f : ℕ → ℕ, (∀ i, i < n → f i ≤ n - 1) ∧ (∀ i j k, i < n → j < n → k < n → i ≠ j → j ≠ k → i ≠ k → ¬(f i = f j ∧ f j = f k)) := 
sorry

end introduce_people_no_three_same_acquaintances_l71_71128


namespace parabola_area_l71_71626

theorem parabola_area (m p : ℝ) (h1 : p > 0) (h2 : (1:ℝ)^2 = 2 * p * m)
    (h3 : (1/2) * (m + p / 2) = 1/2) : p = 1 :=
  by
    sorry

end parabola_area_l71_71626


namespace arithmetic_sequence_difference_l71_71543

theorem arithmetic_sequence_difference :
  ∀ (a d : ℝ),
  a ≥ 15 ∧ a ≤ 120 ∧ 15 ≤ a + 299 * d ∧ a + 299 * d ≤ 120 ∧ 
  300 * (a + 149 * d) = 12000 →
  (G - L = 31320 / 299) :=
by
  intros a d h,
  let L := a + 148 * (-105 / 299),
  let G := a + 148 * (105 / 299),
  have h_1 : G = 120 + 15660 / 299,
  have h_2 : L = 15 - 15660 / 299,
  sorry

end arithmetic_sequence_difference_l71_71543


namespace series_sum_equals_4290_l71_71659

theorem series_sum_equals_4290 :
  ∑ n in Finset.range 10, n.succ * (n + 2) * (n + 3) = 4290 := 
by
  sorry

end series_sum_equals_4290_l71_71659


namespace sequence_inequality_l71_71369

theorem sequence_inequality (a : ℕ → ℝ) (h1 : ∀ n m : ℕ, a (n + m) ≤ a n + a m)
  (h2 : ∀ n : ℕ, 0 ≤ a n) (n m : ℕ) (hnm : n ≥ m) : 
  a n ≤ m * a 1 + (n / m - 1) * a m :=
sorry

end sequence_inequality_l71_71369


namespace total_tickets_used_l71_71177

-- Definitions based on the conditions
def ferris_wheel_rides : ℕ := 7
def bumper_car_rides : ℕ := 3
def tickets_per_ride : ℕ := 3

-- Hypothesis
theorem total_tickets_used (ferris_wheel_rides bumper_car_rides tickets_per_ride : ℕ) :
  ferris_wheel_rides = 7 → 
  bumper_car_rides = 3 → 
  tickets_per_ride = 3 → 
  (ferris_wheel_rides + bumper_car_rides) * tickets_per_ride = 30 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  apply eq.refl
  -- This serves as a placeholder for the actual calculation and steps
  sorry

end total_tickets_used_l71_71177


namespace inequality_proof_l71_71244

theorem inequality_proof
  (a b c d e f : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (he : 0 < e)
  (hf : 0 < f)
  (h_condition : abs (Real.sqrt (a * b) - Real.sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := 
sorry

end inequality_proof_l71_71244


namespace parallelogram_area_l71_71471

def base : ℝ := 36
def height : ℝ := 18
def area (b h : ℝ) : ℝ := b * h

theorem parallelogram_area : area base height = 648 := by
  sorry

end parallelogram_area_l71_71471


namespace probability_event_A_probability_event_B_probability_event_C_l71_71970

section

variables (Ω : Type) [Fintype Ω] (cards : Set {n : ℕ | n ∈ {1, 2, 3, 4}}) (draws : Π (i : Fin 3), Ω)

def event_A : Set (Π (i : Fin 3), Ω) := {draws | draws 0 = draws 1 ∧ draws 1 = draws 2}
def event_B : Set (Π (i : Fin 3), Ω) := {draws | ¬ (draws 0 = draws 1 ∧ draws 1 = draws 2)}
def event_C : Set (Π (i : Fin 3), Ω) := {draws | draws 0 * draws 1 = draws 2}

theorem probability_event_A : (Finset.card (event_A draws).to_finset : ℚ) / (Finset.card (@Set.univ (Π (i : Fin 3), Ω)).to_finset : ℚ) = 1 / 16 := 
sorry

theorem probability_event_B : (Finset.card (event_B draws).to_finset : ℚ) / (Finset.card (@Set.univ (Π (i : Fin 3), Ω)).to_finset : ℚ) = 15 / 16 := 
sorry

theorem probability_event_C : (Finset.card (event_C draws).to_finset : ℚ) / (Finset.card (@Set.univ (Π (i : Fin 3), Ω)).to_finset : ℚ) = 1 / 8 := 
sorry

end

end probability_event_A_probability_event_B_probability_event_C_l71_71970


namespace youngest_dwarf_age_l71_71901

-- Definitions based on the conditions of the problem
variables (B S A Y : ℕ)

-- Initial conditions
axiom condition1 : B - 1 = S
axiom condition2 : B + 2 = A
axiom condition3 : A = S - Y

theorem youngest_dwarf_age :
  Y = 1 :=
by
  -- Outline the steps here
  have h1 : B - 1 = S := condition1
  have h2 : B + 2 = A := condition2
  have h3 : A = S - Y := condition3
  rw [h3, h1] at h2
  have : B + 2 = B - 1 - Y := h2
  linarith

end youngest_dwarf_age_l71_71901


namespace hyperbola_eccentricity_l71_71641

theorem hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (asymptote : ∀ x, (y : ℝ) = 3 * x → b / a = 3) :
  let e := sqrt (1 + (b / a) ^ 2) in e = sqrt 10 :=
by
  sorry

end hyperbola_eccentricity_l71_71641


namespace set_union_example_l71_71999

theorem set_union_example :
  let A := {1, 2, 3, 4}
  let B := {1, 4, 7, 8}
  (A ∩ B = {1, 4}) → (A ∪ B = {1, 2, 3, 4, 7, 8}) :=
by
  intros A B h_inter
  sorry

end set_union_example_l71_71999


namespace scientific_notation_of_34_million_l71_71534

theorem scientific_notation_of_34_million :
  34_000_000 = 3.4 * 10^7 := 
by
  sorry

end scientific_notation_of_34_million_l71_71534


namespace solve_problem_l71_71946

def parametric_equation_line_l (t : ℝ) : ℝ × ℝ :=
  (sqrt 2 / 2 * t, 3 + sqrt 2 / 2 * t)

def polar_equation_curve_C (ρ θ : ℝ) : Prop :=
  ρ * cos θ ^ 2 = 2 * sin θ

def standard_equation_line_l (x y : ℝ) : Prop :=
  x - y + 3 = 0

def cartesian_equation_curve_C (x y : ℝ) : Prop :=
  x ^ 2 = 2 * y

def distance (P M : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - M.1) ^ 2 + (P.2 - M.2) ^ 2)

theorem solve_problem :
  (∀ t, parametric_equation_line_l t) →
  (∀ ρ θ, polar_equation_curve_C ρ θ) →
  (∃ M P : ℝ × ℝ, 
    let M := (1, 4) in
    let P := (1, 1) in
    distance P M = 3) → 
  (∀ x y, standard_equation_line_l x y) ∧ 
  (∀ x y, cartesian_equation_curve_C x y) :=
by sorry

end solve_problem_l71_71946


namespace quadratic_no_real_roots_l71_71306

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x - k = 0)) ↔ k < -1 :=
by sorry

end quadratic_no_real_roots_l71_71306


namespace sum_of_roots_of_quadratic_l71_71022

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l71_71022


namespace equations_no_root_zero_l71_71945

theorem equations_no_root_zero :
  ∀ {x : ℝ}, ¬ (5 * x^2 - 3 = 47 ∧ (3 * x + 2)^2 = (x + 2)^2 ∧ sqrt(2 * x^2 - 6) = sqrt(2 * x - 2)) → 
  ¬ (x = 0) :=
by
  intros x h
  -- Here, h represents the conjunction of our three equations.
  sorry

end equations_no_root_zero_l71_71945


namespace population_increase_period_l71_71438

noncomputable def population_growth (n : ℕ) : ℝ := 240 * (1.10 ^ n)

theorem population_increase_period :
  ∃ n : ℕ, population_growth n = 264 :=
  by
    use 1
    unfold population_growth
    norm_num
    sorry  -- Proof steps not required

end population_increase_period_l71_71438


namespace sum_of_powers_eq_l71_71853

variable (ε : ℂ) (n k : ℕ)

-- Definition of a primitive n-th root of unity.
def is_primitive_nth_root (ε : ℂ) (n : ℕ) : Prop :=
  ε ^ n = 1 ∧ ∀ m : ℕ, 1 ≤ m < n → ε ^ m ≠ 1

-- Main statement of the problem
theorem sum_of_powers_eq (h : is_primitive_nth_root ε n) :
  (1 ≤ k ∧ k < n → (∑ i in Finset.range n, ε ^ (k * i) = 0)) ∧ (k = n → (∑ i in Finset.range n, ε ^ (k * i) = n)) :=
  by sorry

end sum_of_powers_eq_l71_71853


namespace sum_of_roots_eq_14_l71_71046

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l71_71046


namespace slope_of_line_l_l71_71625

def slope_of_line (d : ℝ × ℝ) : ℝ :=
  - (d.2 / d.1)

theorem slope_of_line_l (d : ℝ × ℝ) (h : d = (3, -4)) : slope_of_line d = -4 / 3 :=
by
  rw [h]
  simp [slope_of_line]
  norm_num

end slope_of_line_l_l71_71625


namespace number_of_pants_l71_71844

theorem number_of_pants (tshirts: ℕ) (ways: ℕ) (pants: ℕ) (h_tshirts: tshirts = 8) (h_ways: ways = 72) (h_eq: ways = tshirts * pants) : pants = 9 := 
by
  rw [h_ways, h_tshirts] at h_eq
  exact Nat.eq_of_mul_cancel_right (by decide) h_eq

end number_of_pants_l71_71844


namespace max_can_achieve_goal_a_max_can_achieve_goal_b_max_can_achieve_goal_c_l71_71479

-- Define the number of cans
def num_cans : ℕ := 2015

-- Define the initial configurations as functions from can number to coin count
def initial_config_a (n : ℕ) : ℕ := 0
def initial_config_b (n : ℕ) : ℕ := n
def initial_config_c (n : ℕ) : ℕ := num_cans + 1 - n

-- Define the step operation Max can perform
def step (coins : ℕ → ℕ) (n : ℕ) : ℕ → ℕ :=
  λ i, if i = n then coins i else coins i + n

-- Define the final condition where all cans have the same number of coins
def all_equal (coins : ℕ → ℕ) : Prop :=
  ∃ k, ∀ i, 1 ≤ i ∧ i ≤ num_cans → coins i = k

-- Prove Max can achieve the goal for configuration (a)
theorem max_can_achieve_goal_a :
  ∃ (f : ℕ → ℕ), (∀ n, 1 ≤ n ∧ n ≤ num_cans → f n = 0) →
    (∃ t, (∀ i, 1 ≤ i ∧ i ≤ num_cans → all_equal (nat.iterate (step f) t i))) :=
  sorry

-- Prove Max can achieve the goal for configuration (b)
theorem max_can_achieve_goal_b :
  ∃ (f : ℕ → ℕ), (∀ n, 1 ≤ n ∧ n ≤ num_cans → f n = n) →
    (∃ t, (∀ i, 1 ≤ i ∧ i ≤ num_cans → all_equal (nat.iterate (step f) t i))) :=
  sorry

-- Prove Max can achieve the goal for configuration (c)
theorem max_can_achieve_goal_c :
  ∃ (f : ℕ → ℕ), (∀ n, 1 ≤ n ∧ n ≤ num_cans → f n = num_cans + 1 - n) →
    (∃ t, (∀ i, 1 ≤ i ∧ i ≤ num_cans → all_equal (nat.iterate (step f) t i))) :=
  sorry

end max_can_achieve_goal_a_max_can_achieve_goal_b_max_can_achieve_goal_c_l71_71479


namespace icosahedron_probability_div_by_three_at_least_one_fourth_l71_71821
open ProbabilityTheory

theorem icosahedron_probability_div_by_three_at_least_one_fourth (a b c : ℕ) (h : a + b + c = 20) :
  (a^3 + b^3 + c^3 + 6 * a * b * c : ℚ) / (a + b + c)^3 ≥ 1 / 4 :=
sorry

end icosahedron_probability_div_by_three_at_least_one_fourth_l71_71821


namespace find_a6_over_a5_range_l71_71616

universe u

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 0 + n * d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * a 0 + (∙/2) * n * (n + 1)

theorem find_a6_over_a5_range
  (a : ℕ → ℝ)
  (h1 : is_arithmetic_sequence a)
  (h2 : ∀ n : ℕ, sum_of_first_n_terms a n ≥ sum_of_first_n_terms a 2) :
  3/2 ≤ a 5 / a 4 ∧ a 5 / a 4 ≤ 2 := by
  sorry

end find_a6_over_a5_range_l71_71616


namespace caesar_sum_new_seq_991_l71_71966

-- Define the Caesar sum for a given sequence
def caesar_sum (seq : List ℕ) : ℕ :=
  let partial_sums := List.scanl Nat.add 0 seq
  (partial_sums.tail.sum) / seq.length

-- The initial conditions of the problem
variables (P : List ℕ)
variables (hP_len : P.length = 99)
variables (hP_sum : caesar_sum P = 1000)

-- Define the new sequence with an additional leading element
def new_seq := 1 :: P

-- The main theorem to prove
theorem caesar_sum_new_seq_991 : caesar_sum new_seq = 991 :=
by
  sorry

end caesar_sum_new_seq_991_l71_71966


namespace find_new_pyramid_volume_l71_71147

noncomputable def original_volume : ℝ := 81
noncomputable def smaller_volume : ℝ := 3

theorem find_new_pyramid_volume (V : ℝ) (v : ℝ) (V = original_volume) (v = smaller_volume) :
  let k := (V / v)^(1 / 3)
  k = 3 →
  let V_new := v * (k - 1)
  V_new = 6 :=
by
  sorry

end find_new_pyramid_volume_l71_71147


namespace find_a_b_and_compare_y_values_l71_71257

-- Conditions
def quadratic (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 1
def linear (a : ℝ) (x : ℝ) : ℝ := a * x

-- Problem statement
theorem find_a_b_and_compare_y_values (a b y1 y2 y3 : ℝ) (h₀ : quadratic a b (-2) = 1) (h₁ : linear a (-2) = 1)
    (h2 : y1 = quadratic a b 2) (h3 : y2 = quadratic a b b) (h4 : y3 = quadratic a b (a - b)) :
  (a = -1/2) ∧ (b = -2) ∧ y1 < y3 ∧ y3 < y2 :=
by
  -- Placeholder for the proof
  sorry

end find_a_b_and_compare_y_values_l71_71257


namespace calc_expression_eq_l71_71127

theorem calc_expression_eq : (|Real.sqrt 2 - Real.sqrt 5| + 2 * Real.sqrt 2) = (Real.sqrt 5 + Real.sqrt 2) :=
by simp [abs_of_nonneg, Real.sqrt_pos.2, sub_nonneg_of_le_eq]; sorry

end calc_expression_eq_l71_71127


namespace derivative_at_pi_over_4_l71_71296

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_over_4 : deriv f (Real.pi / 4) = 0 :=
by
  sorry

end derivative_at_pi_over_4_l71_71296


namespace math_problem_l71_71246

open Real

variables {a b c d e f : ℝ}

theorem math_problem 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (hcond : abs (sqrt (a * b) - sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) :=
sorry

end math_problem_l71_71246


namespace charles_winning_strategy_l71_71165

-- Define the function that determines Charles's winning strategy based on the initial number n
noncomputable def charles_wins (n : ℕ) : Prop :=
  (n > 1) ∧ (¬ (nat.prime n) ∨ n = 2 ∨ n = 7 ∨ n = 13 ∨ n = 8 ∨ n = 14 ∨ n = 26 ∨ n = 49 ∨ n = 91 ∨ n = 169)

theorem charles_winning_strategy (n : ℕ) (h : n > 1) : 
  charles_wins n ↔ 
  (¬ nat.prime n ∨ n = 2 ∨ n = 7 ∨ n = 13 ∨ n = 8 ∨ n = 14 ∨ n = 26 ∨ n = 49 ∨ n = 91 ∨ n = 169) := by
  sorry

end charles_winning_strategy_l71_71165


namespace smaller_mold_radius_l71_71510

theorem smaller_mold_radius (R : ℝ) (third_volume_sharing : ℝ) (molds_count : ℝ) (r : ℝ) 
  (hR : R = 3) 
  (h_third_volume_sharing : third_volume_sharing = 1/3) 
  (h_molds_count : molds_count = 9) 
  (h_r : (2/3) * Real.pi * r^3 = (2/3) * Real.pi / molds_count) : 
  r = 1 := 
by
  sorry

end smaller_mold_radius_l71_71510


namespace symmedian_AD_in_DBC_l71_71402

open EuclideanGeometry

-- The conditions required in the problem
variables {A B C D X : Point}

-- Assume we have a triangle ABC and points A, D, and X satisfying the given conditions
axiom h1 : Triangle A B C
axiom h2 : Collinear A D X
axiom h3 : IsSymmedian A X A B C
axiom h4 : TangentIntersection X (Circumcircle B C)

theorem symmedian_AD_in_DBC :
  IsSymmedian D X D B C :=
sorry

end symmedian_AD_in_DBC_l71_71402


namespace rectangle_perimeter_l71_71892

-- Define the conditions
def is_square {α : Type} [OrderedField α] (s : α) := (perim : α) (perim = 4 * s)
def is_divided_into_rectangles {α : Type} [OrderedField α] (s : α) := (w l : α) (w = s / 2) ∧ (l = s)

-- Theorem statement
theorem rectangle_perimeter (s : ℝ) (w l : ℝ) (h_square : is_square s) (h_div : is_divided_into_rectangles s w l) :
  4 * s = 200 → 2 * (l + w) = 150 :=
by sorry

end rectangle_perimeter_l71_71892


namespace product_of_distinct_nonzero_real_satisfying_eq_l71_71252

theorem product_of_distinct_nonzero_real_satisfying_eq (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
    (h : x + 3/x = y + 3/y) : x * y = 3 :=
by sorry

end product_of_distinct_nonzero_real_satisfying_eq_l71_71252


namespace quadratic_no_real_roots_l71_71307

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x - k = 0)) ↔ k < -1 :=
by sorry

end quadratic_no_real_roots_l71_71307


namespace sum_of_roots_eq_14_l71_71049

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l71_71049


namespace speed_of_train_l71_71159

-- Define conditions as variables/constants in Lean
def train_length_km : ℝ := 0.24
def man_speed_kmph : ℝ := 6
def passing_time_hrs : ℝ := 13.090909090909092 / 3600

-- The given statement to prove
theorem speed_of_train :
  let relative_speed := train_length_km / passing_time_hrs in
  let train_speed := relative_speed - man_speed_kmph in
  train_speed = 60 := 
begin
  -- Placeholder for proof
  sorry
end

end speed_of_train_l71_71159


namespace average_multiples_of_5_eq_55_l71_71825

theorem average_multiples_of_5_eq_55 (n : ℕ) (h : (n : ℚ) ≠ 0)
  (h1 : ∑ i in finset.range n, (5 * (i + 1) : ℚ) = (n/2) * (5 + 5 * n))
  (h2 : ((n/2) * (5 + 5 * n)) / n = 55) : 
  n = 21 :=
sorry

end average_multiples_of_5_eq_55_l71_71825


namespace lines_parallel_or_perpendicular_count_l71_71927

noncomputable def slope (a b : ℝ) : ℝ := a / b

def line1_slope : ℝ := 4
def line2_slope : ℝ := slope 6 2
def line3_slope : ℝ := slope 12 3
def line4_slope : ℝ := slope (-3) 2
def line5_slope : ℝ := slope (-8) 4

def count_parallel_perpendicular_pairs (slopes : list ℝ) : ℕ :=
(let rec_helper : list ℝ → ℕ → ℕ
  | [], cnt := cnt
  | (h :: t), cnt :=
    rec_helper t (cnt + (t.count (λ s, s = h ∨ s * h = -1)))
 in rec_helper slopes 0) / 2

theorem lines_parallel_or_perpendicular_count :
  count_parallel_perpendicular_pairs [line1_slope, line2_slope, line3_slope, line4_slope, line5_slope] = 1 :=
sorry

end lines_parallel_or_perpendicular_count_l71_71927


namespace area_triangle_APB_l71_71156

theorem area_triangle_APB
  (A B C P E : ℝ×ℝ)
  (PA PB PC CE : ℝ)
  (side : ℝ)
  (h1 : PA = 6)
  (h2 : PB = 8)
  (h3 : PC = 10)
  (h4 : CE = 5)
  (h5 : side = 10)
  (h6 : (P.fst^2 + P.snd^2 = PC^2))
  (h7 : (A.fst A.snd) = (0, 0))
  (h8 : (B.fst B.snd) = (side, 0))
  (h9 : (E.fst E.snd) = (5, 5√3)) :
  1/2 * side * CE = 25 * Real.sqrt 3 :=
by
  sorry

end area_triangle_APB_l71_71156


namespace transform_curve_l71_71000

theorem transform_curve :
  ∀ x y x' y', (x' = 2 * x) → (y' = 3 * y) → (y = sin (2 * x)) → (y' = 3 * sin x') :=
by
  intros x y x' y' hx hy hcurve
  sorry

end transform_curve_l71_71000


namespace min_cost_seedlings_l71_71139

-- Define the conditions
def seedlings_base := 4
def profit_base := 5
def profit_decrease_per_seedling := 0.5
def target_profits := 24

-- Define the function for profit per seedling
def profit_per_seedling (x : ℕ) := profit_base - profit_decrease_per_seedling * x

-- Define the function for total profit based on additional seedlings
def total_profit (x : ℕ) := (seedlings_base + x) * profit_per_seedling x

-- The theorem to prove
theorem min_cost_seedlings : (total_profit 2) = target_profits → (seedlings_base + 2) = 6 :=
by
  sorry

end min_cost_seedlings_l71_71139


namespace volume_of_solid_l71_71796

-- Definitions based on given conditions
def s : ℝ := 4 * Real.sqrt 2
def base_edge : ℝ := s
def vertical_edge : ℝ := s
def top_distance_opposite_edges1 : ℝ := 3 * s
def top_distance_opposite_edges2 : ℝ := s

-- The base area of the square
def base_area : ℝ := base_edge * base_edge

-- The volume of the solid
def volume : ℝ := base_area * vertical_edge

-- The goal is to prove the volume of the solid is 128 * sqrt(2)
theorem volume_of_solid : volume = 128 * Real.sqrt 2 := by
  sorry

end volume_of_solid_l71_71796


namespace integers_satisfy_equation_l71_71110

theorem integers_satisfy_equation (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  sorry

end integers_satisfy_equation_l71_71110


namespace cookies_per_batch_l71_71222

theorem cookies_per_batch :
  let area_rectangle (base height : ℝ) := base * height
  let area_circle (radius : ℝ) := real.pi * radius^2
  let area_square (side : ℝ) := side^2
  let area_hexagon (side : ℝ) := (3 * real.sqrt 3 / 2) * side^2
  let total_dough_olivia := 10 * area_rectangle 4 3
  let total_dough_maya := total_dough_olivia
  total_dough_maya / area_hexagon 2 = 12 :=
by
  -- placeholder for proof
  sorry

end cookies_per_batch_l71_71222


namespace find_a_from_limit_l71_71226

theorem find_a_from_limit (a : ℝ) (h : (Filter.Tendsto (fun n : ℕ => (a * n - 2) / (n + 1)) Filter.atTop (Filter.principal {1}))) :
    a = 1 := 
sorry

end find_a_from_limit_l71_71226


namespace negation_of_prop_l71_71785

theorem negation_of_prop : ¬(∀ x : ℝ, x > 0 → 2^x > 0) ↔ (∃ x : ℝ, x > 0 ∧ 2^x ≤ 0) :=
by
  sorry

end negation_of_prop_l71_71785


namespace symmetric_about_one_symmetric_about_two_l71_71736

-- Part 1
theorem symmetric_about_one (rational_num_x : ℚ) (rational_num_r : ℚ) 
(h1 : 3 - 1 = 1 - rational_num_x) (hr1 : r = 3 - 1): 
  rational_num_x = -1 ∧ rational_num_r = 2 := 
by
  sorry

-- Part 2
theorem symmetric_about_two (a b : ℚ) (symmetric_radius : ℚ) 
(h2 : (a + b) / 2 = 2) (condition : |a| = 2 * |b|) : 
  symmetric_radius = 2 / 3 ∨ symmetric_radius = 6 := 
by
  sorry

end symmetric_about_one_symmetric_about_two_l71_71736


namespace sum_of_roots_eq_14_l71_71032

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l71_71032


namespace volume_of_revolution_l71_71964

theorem volume_of_revolution (a : ℝ) (h : 0 < a) :
  let x (θ : ℝ) := a * (1 + Real.cos θ) * Real.cos θ
  let y (θ : ℝ) := a * (1 + Real.cos θ) * Real.sin θ
  V = (8 / 3) * π * a^3 :=
sorry

end volume_of_revolution_l71_71964


namespace Ana_will_win_if_and_only_if_n_is_odd_l71_71925

theorem Ana_will_win_if_and_only_if_n_is_odd (n : ℕ) (h : 2 < n) : 
  (∃ f : ℕ → ℝ, ∀ k, f k ∈ set.Icc (0 : ℝ) (2 * n) ∧ (∀ (i j : ℕ), i ≠ j → abs (f i - f j) > 2)) 
  ↔ nat.odd n := 
sorry

end Ana_will_win_if_and_only_if_n_is_odd_l71_71925


namespace solve_system_l71_71748

open Real

theorem solve_system (x y : ℝ) : 
  (1/2 * log 2 x - log 2 y = 0) ∧ (x^2 - 2 * y^2 = 8) → ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = -2)) :=
by
  sorry

end solve_system_l71_71748


namespace students_playing_both_l71_71389

theorem students_playing_both (total_students tennis_fraction hockey_fraction : ℕ) 
  (h1 : total_students = 600) 
  (h2 : tennis_fraction = (3/4))
  (h3 : hockey_fraction = (60/100)) :
  ∃ (students_play_tennis students_play_both : ℕ),
    students_play_tennis = tennis_fraction * total_students ∧
    students_play_both = hockey_fraction * students_play_tennis ∧
    students_play_both = 270 :=
  by 
    use 450
    use 270
    split
    · sorry
    split
    · sorry
    · sorry

end students_playing_both_l71_71389


namespace expected_coincidences_l71_71482

/-- Given conditions for the test -/
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8
def vasya_prob_correct : ℝ := 6 / 20
def misha_prob_correct : ℝ := 8 / 20
def coincidence_prob : ℝ :=
  (vasya_prob_correct * misha_prob_correct) + (1 - vasya_prob_correct) * (1 - misha_prob_correct)

/-- Expected number of coincidences -/
theorem expected_coincidences :
  20 * coincidence_prob = 10.8 :=
by {
  -- vasya_prob_correct = 0.3
  -- misha_prob_correct = 0.4
  -- probability of coincidence = 0.3 * 0.4 + 0.7 * 0.6 = 0.54
  -- expected number of coincidences = 20 * 0.54 = 10.8
  sorry
}

end expected_coincidences_l71_71482


namespace total_scarves_l71_71729

theorem total_scarves :
  (let redScarvesPerYarn := 3 in
   let blueScarvesPerYarn := 2 in
   let yellowScarvesPerYarn := 4 in
   let greenScarvesPerYarn := 5 in
   let greenYarns := 3 in
   let purpleScarvesPerYarn := 6 in
   let purpleYarns := 2 in
   redScarvesPerYarn * 1 + blueScarvesPerYarn * 1 + yellowScarvesPerYarn * 1 + greenScarvesPerYarn * greenYarns + purpleScarvesPerYarn * purpleYarns = 36) :=
sorry

end total_scarves_l71_71729


namespace eccentricity_range_proof_l71_71780

noncomputable def hyperbola_eccentricity_range (F l : ℝ) (C : ℝ → ℝ) (A B : ℝ) : Prop :=
  -- Conditions
  let H := (3 * x ^ 2 - y ^ 2 + 24 * x + 36 = 0) in
  let ell_eq := C = F ∧ l = -3 in
  let intersect_line := A + B = 2 * F in
  let center_in_circle := (A - B) ^ 2 + (2 * (F - (A + B) / 2) ^ 2 = (A + B) ^ 2) in
  -- Conclusion
  0 < e ∧ e < sqrt (2 - sqrt 2)
  
theorem eccentricity_range_proof :
  ∀ F l C A B,
  hyperbola_eccentricity_range F l C A B → (0 < e ∧ e < sqrt (2 - sqrt 2)) :=
  by 
    intros,
    sorry

end eccentricity_range_proof_l71_71780


namespace last_digit_of_large_exponentiation_l71_71114

theorem last_digit_of_large_exponentiation
  (a : ℕ) (b : ℕ)
  (h1 : a = 954950230952380948328708) 
  (h2 : b = 470128749397540235934750230) :
  (a ^ b) % 10 = 4 :=
sorry

end last_digit_of_large_exponentiation_l71_71114


namespace domain_of_function_l71_71771

theorem domain_of_function :
  {x : ℝ | ∀ k : ℤ, 2 * x + (π / 4) ≠ k * π + (π / 2)}
  = {x : ℝ | ∀ k : ℤ, x ≠ (k * π / 2) + (π / 8)} :=
sorry

end domain_of_function_l71_71771


namespace polynomial_evaluation_l71_71230

noncomputable def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem polynomial_evaluation : f 2 = 123 := by
  sorry

end polynomial_evaluation_l71_71230


namespace reflect_A_across_x_axis_l71_71423

-- Define the point A
def A : ℝ × ℝ := (-3, 2)

-- Define the reflection function across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Theorem statement: The reflection of point A across the x-axis should be (-3, -2)
theorem reflect_A_across_x_axis : reflect_x A = (-3, -2) := by
  sorry

end reflect_A_across_x_axis_l71_71423


namespace arithmetic_series_sum_l71_71335

variable (a : ℕ → ℝ) (d a1 : ℝ)

def an (n : ℕ) : ℝ := a1 + (n - 1) * d

def S (n : ℕ) : ℝ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_series_sum :
  (an 5 + an 10 + an 15 + an 20 = 20) →
  (S 24 = 120) :=
by
  sorry

end arithmetic_series_sum_l71_71335


namespace problem_statement_l71_71559

noncomputable def given_expression :=
  2 * sin (Real.toRadians 45) + |Real.sqrt 2| - (Real.pi - 2023)^0 - Real.sqrt 2

theorem problem_statement : 
  given_expression = Real.sqrt 2 - 1 := by
  sorry

end problem_statement_l71_71559


namespace sample_frequency_in_range_l71_71521

theorem sample_frequency_in_range :
  let total_capacity := 100
  let freq_0_10 := 12
  let freq_10_20 := 13
  let freq_20_30 := 24
  let freq_30_40 := 15
  (freq_0_10 + freq_10_20 + freq_20_30 + freq_30_40) / total_capacity = 0.64 :=
by
  sorry

end sample_frequency_in_range_l71_71521


namespace train_crossing_time_l71_71525

-- Define the length of the train, the speed of the train, and the length of the bridge
def train_length : ℕ := 140
def train_speed_kmh : ℕ := 45
def bridge_length : ℕ := 235

-- Define constants for unit conversions
def km_to_m : ℕ := 1000
def hr_to_s : ℕ := 3600

-- Calculate the speed in m/s
def train_speed_ms : ℝ :=
  (train_speed_kmh : ℝ) * (km_to_m : ℝ) / (hr_to_s : ℝ)

-- Calculate the total distance to cover (length of train + length of bridge)
def total_distance : ℕ := train_length + bridge_length

-- Calculate the time in seconds required for the train to cross the bridge
def crossing_time : ℝ :=
  (total_distance : ℝ) / train_speed_ms

-- Prove that the crossing time is 30 seconds
theorem train_crossing_time : crossing_time = 30 := by
  sorry

end train_crossing_time_l71_71525


namespace pizza_order_correct_l71_71971

def eva_portion : ℚ := 1 / 4
def gwen_portion : ℚ := 1 / 6
def noah_portion : ℚ := 1 / 5
def mia_portion (total_pizza : ℚ) (eva_portion gwen_portion noah_portion : ℚ) : ℚ :=
  total_pizza - (eva_portion + gwen_portion + noah_portion)

def total_pizza : ℚ := 1

def eva_slices (slices : ℕ) : ℚ := eva_portion * slices
def gwen_slices (slices : ℕ) : ℚ := gwen_portion * slices
def noah_slices (slices : ℕ) : ℚ := noah_portion * slices
def mia_slices (slices : ℕ) : ℚ := mia_portion 1 eva_portion gwen_portion noah_portion * slices

def slices_order (eva_slices gwen_slices noah_slices mia_slices : ℕ) : List (String × ℚ) :=
  [("Eva", eva_slices),
   ("Gwen", gwen_slices),
   ("Noah", noah_slices),
   ("Mia", mia_slices)].sort_by (λ x y, x.2 > y.2)

theorem pizza_order_correct (slices : ℕ) :
  slices_order (eva_slices slices) (gwen_slices slices) (noah_slices slices) (mia_slices slices) =
    [("Eva", eva_slices slices), ("Mia", mia_slices slices), ("Noah", noah_slices slices), ("Gwen", gwen_slices slices)] :=
  sorry

end pizza_order_correct_l71_71971


namespace sum_of_reciprocals_of_roots_l71_71189

noncomputable def cubicEquation := (x : ℝ) -> 40 * x ^ 3 - 60 * x ^ 2 + 25 * x - 1

theorem sum_of_reciprocals_of_roots :
  ∀ (a b c : ℝ), cubicEquation a = 0 ∧ cubicEquation b = 0 ∧ cubicEquation c = 0 ∧ 
  0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2) :=
by
  intro a b c h
  sorry

end sum_of_reciprocals_of_roots_l71_71189


namespace expected_coincidence_proof_l71_71495

noncomputable def expected_coincidences (total_questions : ℕ) (vasya_correct : ℕ) (misha_correct : ℕ) : ℝ :=
  let vasya_probability := vasya_correct / total_questions
  let misha_probability := misha_correct / total_questions
  let both_correct_probability := vasya_probability * misha_probability
  let both_incorrect_probability := (1 - vasya_probability) * (1 - misha_probability)
  let coincidence_probability := both_correct_probability + both_incorrect_probability
  total_questions * coincidence_probability

theorem expected_coincidence_proof : 
  expected_coincidences 20 6 8 = 10.8 :=
by {
  let total_questions := 20
  let vasya_correct := 6
  let misha_correct := 8
  
  let vasya_probability := 0.3
  let misha_probability := 0.4
  
  let both_correct_probability := vasya_probability * misha_probability
  let both_incorrect_probability := (1 - vasya_probability) * (1 - misha_probability)
  let coincidence_probability := both_correct_probability + both_incorrect_probability
  let expected := total_questions * coincidence_probability
  
  have h1 : vasya_probability = 6 / 20 := by sorry
  have h2 : misha_probability = 8 / 20 := by sorry
  have h3 : both_correct_probability = 0.3 * 0.4 := by sorry
  have h4 : both_incorrect_probability = 0.7 * 0.6 := by sorry
  have h5 : coincidence_probability = 0.54 := by sorry
  have h6 : total_questions * coincidence_probability = 20 * 0.54 := by sorry
  have h7 : 20 * 0.54 = 10.8 := by sorry

  sorry
}

end expected_coincidence_proof_l71_71495


namespace bisect_segment_through_incenter_l71_71405

open Function

variables {A B C M C1 I : Type}
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] 
variables [AffineSpace ℝ M] [AffineSpace ℝ C1] [AffineSpace ℝ I]

-- Definitions of the elements in the problem
variable [Triangle A B C]
variable (M : Midpoint ℝ A B)
variable (I : Incenter ℝ A B C)
variable (C1 : TangencyPoint ℝ I A B)
variable (N : Midpoint ℝ C C1)

-- The theorem stating the desired property
theorem bisect_segment_through_incenter (hM: Midpoint ℝ A B M) (hI: Incenter ℝ A B C I) (hC1 : TangencyPoint ℝ I A B C1): 
  LineThrough ℝ M I → Bisects ℝ M I C C1 N :=
sorry

end bisect_segment_through_incenter_l71_71405


namespace units_digit_Fermat_5_l71_71558

def Fermat_number (n: ℕ) : ℕ :=
  2 ^ (2 ^ n) + 1

theorem units_digit_Fermat_5 : (Fermat_number 5) % 10 = 7 := by
  sorry

end units_digit_Fermat_5_l71_71558


namespace symmedians_of_apollonian_and_circumcircle_l71_71403

-- Let A, B, C be points of a triangle, and D be the defined intersection point.
variables {A B C D : Point}

-- Assume we have a triangle formed by points A, B, C
definition triangle (A B C : Point) : Prop := euclidean_geometry.collinear_set {A, B, C}

-- Given the conditions regarding the Apollonian circle and circumcircle
def apollonian_locus_B (A B C M : Point) : Prop :=
  triangle A B C ∧ (∀ M : Point, dist A M / dist M C = dist A B / dist B C)

def intersection_D (A B C D : Point) : Prop :=
  circle_circum A B C ∧ apollonian_locus_B A B C D ∧ D ∈ circumcircle A B C

-- Statement: Proof that the common chords of the circumcircle and Apollonian circles are symmedians.
theorem symmedians_of_apollonian_and_circumcircle (A B C D : Point) :
  triangle A B C → 
  apollonian_locus_B A B C D → 
  intersection_D A B C D → 
  symmedian A B C D := sorry

end symmedians_of_apollonian_and_circumcircle_l71_71403


namespace at_least_two_contestants_solved_5_problems_l71_71321

theorem at_least_two_contestants_solved_5_problems (n : ℕ) (contestants : Fin n → Fin 6 → Bool)
  (h1 : ∀ j : Fin 6, ∃ k : ℕ, 2*k + 1 ≤ 5 * n ∧ (∃ i : Fin n, contestants i j = true))
  (h2 : ¬ ∃ i : Fin n, ∀ j : Fin 6, contestants i j = true) : 
  ∃ i1 i2 : Fin n, i1 ≠ i2 ∧ (∑ j, if contestants i1 j then 1 else 0 = 5) ∧ (∑ j, if contestants i2 j then 1 else 0 = 5) := 
sorry

end at_least_two_contestants_solved_5_problems_l71_71321


namespace probability_divisible_by_18_l71_71448

theorem probability_divisible_by_18 :
  let cards := {2, 3, 4, 5, 6, 7}
  let total_outcomes := 36
  let favorable_outcomes := { (2, 4), (4, 2), (2, 7), (7, 2), (3, 3), (3, 6), (6, 3), (4, 5), (5, 4) }
  finset.card favorable_outcomes = 9 →
  (9:ℚ) / (36:ℚ) = 1 / 4 :=
by
  sorry

end probability_divisible_by_18_l71_71448


namespace no_coffee_buyers_l71_71730

theorem no_coffee_buyers (total_people served: ℕ) 
  (fraction_coffee: ℚ)
  (hc: total_people = 25)
  (h_fraction: fraction_coffee = 3 / 5) :
  total_people - (fraction_coffee * total_people).toNat = 10 :=
by
  sorry

end no_coffee_buyers_l71_71730


namespace distance_from_origin_to_point1_distance_from_point1_to_point2_l71_71324

-- Define points in the coordinate system
def origin : ℝ × ℝ := (0, 0)
def point1 : ℝ × ℝ := (-12, 16)
def point2 : ℝ × ℝ := (3, -8)

-- Define the Euclidean distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Prove the distance from the origin to the point (-12, 16) is 20 units
theorem distance_from_origin_to_point1 : distance origin point1 = 20 := by
  sorry

-- Prove the distance from the point (-12, 16) to the point (3, -8) is sqrt(801) units
theorem distance_from_point1_to_point2 : distance point1 point2 = Real.sqrt 801 := by
  sorry

end distance_from_origin_to_point1_distance_from_point1_to_point2_l71_71324


namespace squares_difference_l71_71387

theorem squares_difference (n : ℕ) : (n + 1)^2 - n^2 = 2 * n + 1 := by
  sorry

end squares_difference_l71_71387


namespace volume_of_pyramid_l71_71673

theorem volume_of_pyramid 
  (A B C D S : Point) 
  (area_SAB : ℝ)
  (area_SBC : ℝ)
  (area_SCD : ℝ) 
  (area_SDA : ℝ) 
  (dihedral_angle_eq : DihedralAngle S A B = DihedralAngle S B C ∧ DihedralAngle S B C = DihedralAngle S C D ∧ DihedralAngle S C D = DihedralAngle S D A)
  (ABCD_cyclic : Cyclic ABCD)
  (area_ABCD : ℝ) 
  (area_ABCD_eq : area_ABCD = 36) 
  (area_SAB_eq : area_SAB = 9)
  (area_SBC_eq : area_SBC = 9)
  (area_SCD_eq : area_SCD = 27)
  (area_SDA_eq : area_SDA = 27)
  : volume S A B C D = 54 :=
sorry

end volume_of_pyramid_l71_71673


namespace value_of_a_l71_71250

theorem value_of_a (a : ℝ) : (|a| - 1 = 1) ∧ (a - 2 ≠ 0) → a = -2 :=
by
  sorry

end value_of_a_l71_71250


namespace square_side_length_and_area_l71_71197

theorem square_side_length_and_area (P : ℕ) (h : P = 48) :
  ∃ (s A : ℕ), 4 * s = P ∧ A = s * s ∧ s = 12 ∧ A = 144 :=
by
  -- To construct the statement, we take the conditions identified
  let s := 12
  let A := 144
  use [s, A]
  -- Adding conditions as definitions
  split
  · exact h ▸ rfl
  split
  · exact rfl
  split
  · exact rfl
  exact rfl

end square_side_length_and_area_l71_71197


namespace zebra_chasing_tiger_time_l71_71902

theorem zebra_chasing_tiger_time : 
  ∃ t : ℝ, (t + 6) * 30 = 330 ∧ 55 * 6 = 330 ∧ 30 * 6 = 180 ∧ t = 5 :=
by
  use 5
  split
  · calc 5 + 6 = 11 : by rfl
         11 * 30 = 330 : by norm_num
  split
  · calc 55 * 6 = 330 : by norm_num
  split
  · calc 30 * 6 = 180 : by norm_num
  · rfl

end zebra_chasing_tiger_time_l71_71902


namespace decagon_diagonals_l71_71914

-- The condition for the number of diagonals in a polygon
def number_of_diagonals (n : Nat) : Nat :=
  n * (n - 3) / 2

-- The specific proof statement for a decagon
theorem decagon_diagonals : number_of_diagonals 10 = 35 := by
  -- The proof would go here
  sorry

end decagon_diagonals_l71_71914


namespace sum_of_roots_eq_l71_71063

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l71_71063


namespace sum_of_roots_l71_71019

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l71_71019


namespace first_candidate_percentage_l71_71675

theorem first_candidate_percentage (total_votes : ℕ) (invalid_percentage : ℕ) (second_candidate_votes : ℕ) 
  (h_total_votes : total_votes = 7500) 
  (h_invalid_percentage : invalid_percentage = 20) 
  (h_second_candidate_votes : second_candidate_votes = 2700) : 
  (100 * (total_votes * (1 - (invalid_percentage / 100)) - second_candidate_votes) / (total_votes * (1 - (invalid_percentage / 100)))) = 55 :=
by
  sorry

end first_candidate_percentage_l71_71675


namespace pairs_of_positive_integers_l71_71195

theorem pairs_of_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
    (∃ (m : ℕ), m ≥ 2 ∧ (x = m^3 + 2*m^2 - m - 1 ∧ y = m^3 + m^2 - 2*m - 1 ∨ 
                        x = m^3 + m^2 - 2*m - 1 ∧ y = m^3 + 2*m^2 - m - 1)) ∨
    (x = 1 ∧ y = 1) ↔ 
    (∃ n : ℝ, n^3 = 7*x^2 - 13*x*y + 7*y^2) ∧ (Int.natAbs (x - y) - 1 = n) :=
by
  sorry

end pairs_of_positive_integers_l71_71195


namespace sum_of_roots_of_equation_l71_71084

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l71_71084


namespace find_m_l71_71563

def f (x : ℝ) : ℝ := 3 * x^3 - 1 / x + 5
def g (x : ℝ) (m : ℝ) : ℝ := 3 * x^2 - m

theorem find_m : ∃ m : ℝ, (f (-1) - g (-1) m = 1) ∧ m = 1 :=
by
  have h₁ : f (-1) = 3 := by sorry
  have h₂ : g (-1) m = 3 - m := by sorry
  have h_cond : f (-1) - g (-1) m = 1 := by sorry
  use 1
  split
  case left =>
    exact h_cond
  case right =>
    exact rfl

end find_m_l71_71563


namespace probability_of_valid_sum_l71_71502

-- Define the set of numbers in the box
def slips := [1, 3, 5, 8, 13, 21, 34, 55]

-- Define the condition for drawing two slips without replacement
def draw_two (s: List ℕ) : List (ℕ × ℕ) := 
  List.bind s (λ n, List.map (λ m, (n, m)) (s.erase n))

-- Define the possible pairs and their sums
def sums (pairs: List (ℕ × ℕ)) : List ℕ := 
  List.map (λ (p: ℕ × ℕ), p.1 + p.2) pairs

-- Check which sums are in the original list
def valid_sums (s: List ℕ) (pairs: List (ℕ × ℕ)) : List (ℕ × ℕ) := 
  List.filter (λ p, List.contains s (p.1 + p.2)) pairs

-- Calculate the probability
def probability_of_sum_in_slips : ℚ := 
  let pairs := draw_two slips
  let valid_pairs := valid_sums slips pairs
  valid_pairs.length / pairs.length

-- Proof statement
theorem probability_of_valid_sum : 
  probability_of_sum_in_slips = 5 / 28 := 
by sorry

end probability_of_valid_sum_l71_71502


namespace angle_ACS_eq_angle_BCP_l71_71395

variables {A B C K L M N X Y P S : Point}

-- Define the conditions 
variables [hABC: AcuteTriangle A B C]
variables [hCAKL: SquareOutsideTriangle C A K L]
variables [hCBMN: SquareOutsideTriangle C B M N]
variables [hCNX: Intersects CN AK X]
variables [hCLY: Intersects CL BM Y]
variables [hP: CircumcenterIntersects KXN LYM P]
variables [hS: Midpoint S A B]

theorem angle_ACS_eq_angle_BCP :
  ∠ A C S = ∠ B C P :=
sorry

end angle_ACS_eq_angle_BCP_l71_71395


namespace distance_between_parabola_vertices_l71_71190

theorem distance_between_parabola_vertices : 
  ∀ x y : ℝ, (sqrt (x^2 + y^2) + abs (y - 2) = 4) →
    |(6 - max y 2) - (-(2 - min y 2))| = 4 :=
by
  intros x y h_eqn
  sorry

end distance_between_parabola_vertices_l71_71190


namespace sum_last_two_digits_7_13_23_l71_71009

theorem sum_last_two_digits_7_13_23 :
  (7 ^ 23 + 13 ^ 23) % 100 = 40 :=
by 
-- Proof goes here
sorry

end sum_last_two_digits_7_13_23_l71_71009


namespace sum_of_roots_l71_71011

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l71_71011


namespace binomial_10_5_is_252_l71_71921

theorem binomial_10_5_is_252 :
  Nat.binom 10 5 = 252 := by
  sorry

end binomial_10_5_is_252_l71_71921


namespace sum_of_roots_eq_14_l71_71053

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l71_71053


namespace point_below_line_l71_71663

theorem point_below_line (a : ℝ) (h : 2 * a - 3 > 3) : a > 3 :=
sorry

end point_below_line_l71_71663


namespace problem_statement_l71_71726

/-
Let the universal set U = ℝ, and define the sets M = {x | x > 1}, P = {x | x^2 > 1}.
We need to prove that the correct relation is M ⊆ P.
-/

open Set

variable (U : Set ℝ)
variable (M P : Set ℝ)

def collection_of_sets := ∃ (M P : Set ℝ), 
  (∀ x, x ∈ M ↔ x > 1) ∧
  (∀ x, x ∈ P ↔ x^2 > 1)

theorem problem_statement (h : collection_of_sets U M P) :
  M \not\subseteq P :=
by
  sorry

end problem_statement_l71_71726


namespace binom_10_5_eq_252_l71_71917

theorem binom_10_5_eq_252 : binomial 10 5 = 252 :=
by {
  sorry
}

end binom_10_5_eq_252_l71_71917


namespace special_burger_cost_l71_71554

/-
  Prices of individual items and meals:
  - Burger: $5
  - French Fries: $3
  - Soft Drink: $3
  - Kid’s Burger: $3
  - Kid’s French Fries: $2
  - Kid’s Juice Box: $2
  - Kids Meal: $5

  Mr. Parker purchases:
  - 2 special burger meals for adults
  - 2 special burger meals and 2 kids' meals for 4 children
  - Saving $10 by buying 6 meals instead of the individual items

  Goal: 
  - Prove that the cost of one special burger meal is $8.
-/

def price_burger : Nat := 5
def price_fries : Nat := 3
def price_drink : Nat := 3
def price_kid_burger : Nat := 3
def price_kid_fries : Nat := 2
def price_kid_juice : Nat := 2
def price_kids_meal : Nat := 5

def total_adults_cost : Nat :=
  2 * price_burger + 2 * price_fries + 2 * price_drink

def total_kids_cost : Nat :=
  2 * price_kid_burger + 2 * price_kid_fries + 2 * price_kid_juice

def total_individual_cost : Nat :=
  total_adults_cost + total_kids_cost

def total_meals_cost : Nat :=
  total_individual_cost - 10

def cost_kids_meals : Nat :=
  2 * price_kids_meal

def total_cost_4_meals : Nat :=
  total_meals_cost

def cost_special_burger_meal : Nat :=
  (total_cost_4_meals - cost_kids_meals) / 2

theorem special_burger_cost : cost_special_burger_meal = 8 := by
  sorry

end special_burger_cost_l71_71554


namespace domain_of_composite_function_l71_71300

theorem domain_of_composite_function (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 4 → ∃ y, f y = x) :
  ∃ y, f (y^2) = 0 ∧ ∃ z, f (z^2) = 4 :=
begin
  sorry
end

end domain_of_composite_function_l71_71300


namespace train_speed_excluding_stoppages_l71_71585

noncomputable def speed_including_stoppages : ℝ := 36
noncomputable def stoppage_time_per_hour_in_hours : ℝ := 15 / 60
noncomputable def running_time_per_hour : ℝ := 1 - stoppage_time_per_hour_in_hours

theorem train_speed_excluding_stoppages :
  ∃ S : ℝ, (S * running_time_per_hour = speed_including_stoppages) ∧ S = 48 :=
by
  use (48 : ℝ)
  split
  {
    have r1 : running_time_per_hour = 3 / 4 := by sorry
    rw [r1]
    norm_num
  }
  norm_num

end train_speed_excluding_stoppages_l71_71585


namespace circle_tangent_to_x_axis_at_origin_l71_71664

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h1 : ∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 → x = 0 ∧ y = 0 ∨ y = -D/E ∧ x = 0 ∧ F = 0):
  D = 0 ∧ E ≠ 0 ∧ F = 0 :=
sorry

end circle_tangent_to_x_axis_at_origin_l71_71664


namespace length_AE_is_correct_l71_71678

-- Define the conditions of the pentagon
variables {A B C D E : Type} [point A] [point B] [point C] [point D] [point E]

-- Distances between points
axiom length_BC : dist B C = 2
axiom length_CD : dist C D = 2
axiom length_DE : dist D E = 2

-- Angles of the pentagon
axiom angle_E : ∠E = 90
axiom angle_B : ∠B = 135
axiom angle_C : ∠C = 135
axiom angle_D : ∠D = 135

-- Definition of length AE in simplest radical form
noncomputable def length_AE_in_radical_form : ℝ := 4 + 2 * sqrt 2

-- Proof statement to show the correct length of AE
theorem length_AE_is_correct : 
  ∃ a b : ℝ, length_AE_in_radical_form = a + 2 * sqrt b ∧ a + b = 6 :=
begin
  use [4, 2],
  split,
  { dsimp [length_AE_in_radical_form], },
  { norm_num, },
  sorry
end

end length_AE_is_correct_l71_71678


namespace solve_equation_l71_71746

theorem solve_equation (x : ℚ) :
  (x + 10) / (x - 4) = (x - 3) / (x + 6) ↔ x = -48 / 23 :=
by sorry

end solve_equation_l71_71746


namespace binomial_10_5_is_252_l71_71920

theorem binomial_10_5_is_252 :
  Nat.binom 10 5 = 252 := by
  sorry

end binomial_10_5_is_252_l71_71920


namespace expected_coincidences_l71_71491

-- Definitions for the given conditions
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8

def prob_correct (correct : ℕ) : ℚ := correct / num_questions
def prob_incorrect (correct : ℕ) : ℚ := 1 - prob_correct correct 

def prob_vasya_correct := prob_correct vasya_correct
def prob_vasya_incorrect := prob_incorrect vasya_correct
def prob_misha_correct := prob_correct misha_correct
def prob_misha_incorrect := prob_incorrect misha_correct

-- Probability that both guessed correctly or incorrectly
def prob_both_correct_or_incorrect : ℚ := 
  (prob_vasya_correct * prob_misha_correct) + (prob_vasya_incorrect * prob_misha_incorrect)

-- Expected value for one question being a coincidence
def expected_I_k : ℚ := prob_both_correct_or_incorrect

-- Definition of the total number of coincidences
def total_coincidences : ℚ := num_questions * expected_I_k

-- Proof statement
theorem expected_coincidences : 
  total_coincidences = 10.8 := by
  -- calculation for the expected number
  sorry

end expected_coincidences_l71_71491


namespace sum_of_roots_eq_14_l71_71041

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l71_71041


namespace remainder_at_5_l71_71890

noncomputable def polynomial_remainder (p : ℚ[X]) : ℚ[X] :=
  ((X - 3) * (X + 1) * (X - 4)).mod_by_monic p

noncomputable def r_5 (p : ℚ[X]) (h₁ : p.eval 3 = 2) (h₂ : p.eval (-1) = -2) (h₃ : p.eval 4 = 5) : ℚ :=
  (polynomial_remainder p).eval 5

theorem remainder_at_5 (p : ℚ[X]) (h₁ : p.eval 3 = 2) (h₂ : p.eval (-1) = -2) (h₃ : p.eval 4 = 5) :
  r_5 p h₁ h₂ h₃ = 9 :=
sorry

end remainder_at_5_l71_71890


namespace sum_of_squares_of_roots_l71_71923

def polynomial : Polynomial ℝ := 3 * X ^ 3 - 2 * X ^ 2 + 5 * X - 7

theorem sum_of_squares_of_roots :
  (∃ p q r : ℝ, (polynomial.eval p polynomial = 0) ∧ (polynomial.eval q polynomial = 0) ∧ (polynomial.eval r polynomial = 0)) →
  ∃ p q r : ℝ, p^2 + q^2 + r^2 = -26/9 :=
by
  sorry

end sum_of_squares_of_roots_l71_71923


namespace area_of_triangle_ABC_l71_71824

-- Define the given points, lengths, and conditions
def A : ℝ × ℝ := (15, 0)
def D : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (0, 13)

-- Define the lengths given in the problem
def AC := 15
def AB := 17
def DC := 8

-- Define that the points are coplanar, and angle D is a right angle
def coplanar (A B C D : ℝ × ℝ) : Prop := 
  ∃ a b : ℝ, A.2 = 0 ∧ D.2 = 0 ∧ B.1 = 0 ∧ C.1 = 0
def right_angle (D : ℝ × ℝ) (C : ℝ × ℝ) := D.1 = C.1 ∧ D.2 = 0

-- Definition of distance between two points
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Prove that the area of triangle ABC is 4(√161 + √322)
theorem area_of_triangle_ABC : 
  coplanar A B C D → right_angle D C → AC = dist A C → AB = dist A B → DC = dist D C → 
  let AD := dist A D in
  let BD := dist B D in
  let area_ACD := (1/2 * AD * DC : ℝ) in
  let area_ABD := (1/2 * AD * BD : ℝ) in
  (area_ACD + area_ABD = 4 * (real.sqrt 161 + real.sqrt 322)) := 
by {
  -- Add apologies since we are not providing the proof
  sorry,
}

end area_of_triangle_ABC_l71_71824


namespace complex_division_by_conjugate_l71_71781

noncomputable def calc_complex_div (a b : ℂ) : ℂ := a / b

theorem complex_division_by_conjugate : calc_complex_div complex.I (2 + complex.I) = (2 * complex.I + 1) / 5 := 
by
  sorry

end complex_division_by_conjugate_l71_71781


namespace triangle_area_l71_71318

-- Define the conditions of the problem
variables (a b c : ℝ) (C : ℝ)
axiom cond1 : c^2 = a^2 + b^2 - 2 * a * b + 6
axiom cond2 : C = Real.pi / 3

-- Define the goal
theorem triangle_area : 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end triangle_area_l71_71318


namespace actual_average_height_l71_71474

theorem actual_average_height
  (num_boys : ℕ)
  (incorrect_avg_height total_num_boys : ℝ)
  (wrong_height correct_height correct_total_height : ℝ)
  (orign_correct_total orign_wrong_total : ℝ)
  (num_boys = 35)
  (incorrect_avg_height = 182)
  (total_num_boys = 35)
  (wrong_height = 166)
  (correct_height = 106)
  (orign_correct_total = 6370)
  (orign_wrong_total = 6310)
  (correct_total_height = orign_correct_total - (wrong_height - correct_height))
  (correct_avg_height : ℝ) 
  (correct_avg_height = correct_total_height / total_num_boys):
  correct_avg_height = 180.29 := 
by 
  simp only at *,
  sorry

end actual_average_height_l71_71474


namespace range_of_t_l71_71636

-- Definitions for f, g and F
def f (x : ℝ) : ℝ := x^3 + 1
def g (x : ℝ) (t : ℝ) : ℝ := 2 * (Real.log2 x)^2 - 2 * (Real.log2 x) + t - 4
def F (x : ℝ) (t : ℝ) : ℝ := f (g x t) - 1

-- Condition for F(x) having exactly two distinct zeros in the given interval
def has_two_distinct_zeros (F : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x1 x2 ∈ set.Icc a b, F x1 = 0 ∧ F x2 = 0 ∧ x1 ≠ x2

-- Lean statement to prove the range of t
theorem range_of_t (t : ℝ) : has_two_distinct_zeros (λ x, F x t) 1 (2 * Real.sqrt 2) ↔ 4 ≤ t ∧ t < 9 / 2 :=
by
  sorry

end range_of_t_l71_71636


namespace sum_of_roots_eq_14_l71_71033

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l71_71033


namespace sum_of_roots_l71_71012

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l71_71012


namespace problem_rational_sum_of_powers_l71_71651

theorem problem_rational_sum_of_powers :
  ∃ (a b : ℚ), (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 ∧ a + b = 70 :=
by
  sorry

end problem_rational_sum_of_powers_l71_71651


namespace expected_coincidence_proof_l71_71493

noncomputable def expected_coincidences (total_questions : ℕ) (vasya_correct : ℕ) (misha_correct : ℕ) : ℝ :=
  let vasya_probability := vasya_correct / total_questions
  let misha_probability := misha_correct / total_questions
  let both_correct_probability := vasya_probability * misha_probability
  let both_incorrect_probability := (1 - vasya_probability) * (1 - misha_probability)
  let coincidence_probability := both_correct_probability + both_incorrect_probability
  total_questions * coincidence_probability

theorem expected_coincidence_proof : 
  expected_coincidences 20 6 8 = 10.8 :=
by {
  let total_questions := 20
  let vasya_correct := 6
  let misha_correct := 8
  
  let vasya_probability := 0.3
  let misha_probability := 0.4
  
  let both_correct_probability := vasya_probability * misha_probability
  let both_incorrect_probability := (1 - vasya_probability) * (1 - misha_probability)
  let coincidence_probability := both_correct_probability + both_incorrect_probability
  let expected := total_questions * coincidence_probability
  
  have h1 : vasya_probability = 6 / 20 := by sorry
  have h2 : misha_probability = 8 / 20 := by sorry
  have h3 : both_correct_probability = 0.3 * 0.4 := by sorry
  have h4 : both_incorrect_probability = 0.7 * 0.6 := by sorry
  have h5 : coincidence_probability = 0.54 := by sorry
  have h6 : total_questions * coincidence_probability = 20 * 0.54 := by sorry
  have h7 : 20 * 0.54 = 10.8 := by sorry

  sorry
}

end expected_coincidence_proof_l71_71493


namespace car_total_distance_l71_71134

theorem car_total_distance (h1 h2 h3 : ℕ) :
  h1 = 180 → h2 = 160 → h3 = 220 → h1 + h2 + h3 = 560 :=
by
  intros h1_eq h2_eq h3_eq
  sorry

end car_total_distance_l71_71134


namespace sum_of_roots_of_equation_l71_71082

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l71_71082


namespace fractional_part_inequality_condition_iff_l71_71857

open Int

noncomputable def fractional_part (x : ℚ) : ℚ := x - ⌊x⌋

theorem fractional_part_inequality_condition_iff
  (p : ℕ) (hp : p.prime)
  (s : ℕ) (hs : 0 < s ∧ s < p) :
  (∃ m n : ℕ, 0 < m ∧ m < n ∧ n < p ∧ fractional_part (s * m / p) < fractional_part (s * n / p) ∧ fractional_part (s * n / p) < s / p) ↔ ¬ (s ∣ (p - 1)) :=
sorry

end fractional_part_inequality_condition_iff_l71_71857


namespace evaluate_compound_function_l71_71658

def g (x: ℝ) : ℝ := 3 * x ^ 2 + 2
def h (x: ℝ) : ℝ := 2 * x ^ 3 - 5

theorem evaluate_compound_function :
  g(h(2)) = 365 := 
by
  sorry

end evaluate_compound_function_l71_71658


namespace pirates_walk_per_day_l71_71437

theorem pirates_walk_per_day :
  let x := 20 in
  let islands := 4 in
  let days_per_island := 1.5 in
  let total_distance := 135 in
  let distance_per_day_other_islands := 25 in
  let days_total := islands * days_per_island in
  let distance_other_islands := distance_per_day_other_islands * days_per_island in
  let total_distance_other_islands := 2 * distance_other_islands in
  let distance_first_two_islands := total_distance - total_distance_other_islands in
  let days_first_two_islands := 2 * days_per_island in
  let miles_per_day := distance_first_two_islands / days_first_two_islands in
  miles_per_day = x :=
by
  sorry

end pirates_walk_per_day_l71_71437


namespace parallel_sufficient_condition_l71_71618

def line1 (m : ℝ) := (m - 4) * x - (2 * m + 4) * y + 2 * m - 4 = 0
def line2 (m : ℝ) := (m - 1) * x + (m + 2) * y + 1 = 0

/-- Prove that m = -2 is a sufficient but not necessary condition for lines l1 and l2 to be parallel -/
theorem parallel_sufficient_condition (m : ℝ) (x y : ℝ) : 
  (parallel (line1 (-2)) (line2 (-2))) → (∃ m, m = -2 ∨ m = 2 ∧ parallel (line1 m) (line2 m)) :=
sorry

end parallel_sufficient_condition_l71_71618


namespace Andy_earnings_per_hour_l71_71170

def hourly_earnings (wage_per_hour: ℕ) : ℕ := wage_per_hour * 8

def service_earnings (racquets: ℕ) (grommets: ℕ) (stencils: ℕ) : ℕ :=
  (racquets * 15) + (grommets * 10) + (stencils * 1)

theorem Andy_earnings_per_hour : ∀ (wage_per_hour total_earnings : ℕ),
  let total_service_earnings := service_earnings 7 2 5 in
  total_earnings = (hourly_earnings wage_per_hour) + total_service_earnings →
  total_earnings = 202 →
  wage_per_hour = 9 :=
begin
  intros wage_per_hour total_earnings,
  let total_service_earnings := service_earnings 7 2 5,
  intro h1,
  intro h2,
  sorry  -- Proof goes here
end

end Andy_earnings_per_hour_l71_71170


namespace largest_class_is_28_l71_71124

-- definition and conditions
def largest_class_students (x : ℕ) : Prop :=
  let total_students := x + (x - 2) + (x - 4) + (x - 6) + (x - 8)
  total_students = 120

-- statement to prove
theorem largest_class_is_28 : ∃ x : ℕ, largest_class_students x ∧ x = 28 :=
by
  sorry

end largest_class_is_28_l71_71124


namespace find_a_l71_71712

theorem find_a (a b c : ℕ) (h1 : a ≥ b ∧ b ≥ c)  
  (h2 : (a:ℤ) ^ 2 - b ^ 2 - c ^ 2 + a * b = 2011) 
  (h3 : (a:ℤ) ^ 2 + 3 * b ^ 2 + 3 * c ^ 2 - 3 * a * b - 2 * a * c - 2 * b * c = -1997) : 
a = 253 := 
sorry

end find_a_l71_71712


namespace car_avg_mpg_l71_71135

variable (x : ℝ) -- x is the distance from town B to town C

-- Define the conditions
def distance_A_to_B := 2 * x -- Distance from town A to town B
def efficiency_A_to_B := 25 -- Fuel efficiency from town A to town B
def distance_B_to_C := x -- Distance from town B to town C
def efficiency_B_to_C := 30 -- Fuel efficiency from town B to town C

-- Define the total distance and total fuel used derived from the conditions
def total_distance := distance_A_to_B x + distance_B_to_C x
def total_fuel_used := (distance_A_to_B x / efficiency_A_to_B) + (distance_B_to_C x / efficiency_B_to_C)

-- Calculate the average miles per gallon
def avg_miles_per_gallon := total_distance x / total_fuel_used x

-- The main statement
theorem car_avg_mpg : avg_miles_per_gallon x = 450 / 11 :=
by
  sorry

end car_avg_mpg_l71_71135


namespace additional_days_when_selling_5_goats_l71_71509

variables (G D F X : ℕ)

def total_feed (num_goats days : ℕ) := G * num_goats * days

theorem additional_days_when_selling_5_goats
  (h1 : total_feed G 20 D = F)
  (h2 : total_feed G 15 (D + X) = F)
  (h3 : total_feed G 30 (D - 3) = F):
  X = 9 :=
by
  -- the exact proof is omitted and presented as 'sorry'
  sorry

end additional_days_when_selling_5_goats_l71_71509


namespace rodney_correct_guess_probability_l71_71739

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  (n / 10) % 2 = 1 ∧
  n % 2 = 0 ∧
  n > 75

def valid_numbers : List ℕ := 
  [76, 78, 90, 92, 94, 96, 98]

theorem rodney_correct_guess_probability :
  (1 : ℚ) / 7 =
  (1 : ℚ) / (valid_numbers.length : ℚ) :=
by
  have h_valid_len : valid_numbers.length = 7 := by simp
  rw h_valid_len
  simp

#print rodney_correct_guess_probability

end rodney_correct_guess_probability_l71_71739


namespace fraction_picked_l71_71542

/--
An apple tree has three times as many apples as the number of plums on a plum tree.
Damien picks a certain fraction of the fruits from the trees, and there are 96 plums
and apples remaining on the tree. There were 180 apples on the apple tree before 
Damien picked any of the fruits. Prove that Damien picked 3/5 of the fruits from the trees.
-/
theorem fraction_picked (P F : ℝ) (h1 : 3 * P = 180) (h2 : (1 - F) * (180 + P) = 96) :
  F = 3 / 5 :=
by
  sorry

end fraction_picked_l71_71542


namespace math_problem_l71_71247

open Real

variables {a b c d e f : ℝ}

theorem math_problem 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (hcond : abs (sqrt (a * b) - sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) :=
sorry

end math_problem_l71_71247


namespace total_colors_needed_l71_71671

def num_planets : Nat := 8
def moons : List Nat := [0, 0, 1, 2, 79, 82, 27, 14]
def num_friends : Nat := 3

theorem total_colors_needed : 
  let total_moons := moons.sum in
  let total_celestial_bodies := num_planets + total_moons in
  total_friends * total_celestial_bodies = 639 :=
by 
  let total_moons := moons.sum
  let total_celestial_bodies := num_planets + total_moons
  show num_friends * total_celestial_bodies = 639
  sorry

end total_colors_needed_l71_71671


namespace janet_hiking_distance_l71_71698

theorem janet_hiking_distance
  (A B C : EuclideanSpace ℝ)
  (hAB : dist A B = 3)
  (hAC : dist B C = 8)
  (angle_ABC : angle A B C = π / 6) : 
  dist A C = Real.sqrt 57 :=
by
  sorry

end janet_hiking_distance_l71_71698


namespace perpendicular_vector_l71_71646

-- Vectors a and b are given
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 3)

-- Defining the vector addition and scalar multiplication for our context
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (m : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (m * v.1, m * v.2)

-- The vector a + m * b
def a_plus_m_b (m : ℝ) : ℝ × ℝ := vector_add a (scalar_mul m b)

-- The dot product of vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The statement that a is perpendicular to (a + m * b) when m = 5
theorem perpendicular_vector : dot_product a (a_plus_m_b 5) = 0 :=
sorry

end perpendicular_vector_l71_71646


namespace problem_solution_l71_71608

theorem problem_solution (n : ℕ) (a : Fin n → ℝ) (hpos : ∀ i, 0 < a i) (hprod : (∏ i in Finset.univ, a i) = 1) :
  (∏ i in Finset.univ, 2 + a i) ≥ 3^n :=
by
  sorry

end problem_solution_l71_71608


namespace sum_of_roots_eq_14_l71_71070

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l71_71070


namespace eval_expression_l71_71949

-- Define the floor and ceiling functions for the given values
def floor_1999 := Int.floor 1.999
def ceil_3001 := Int.ceil 3.001
def ceil_0001 := Int.ceil 0.001

-- The theorem statement
theorem eval_expression : floor_1999 + ceil_3001 + ceil_0001 = 6 := by
  -- Adding the results of floor and ceiling functions
  have h1 : floor_1999 = 1 := by sorry
  have h2 : ceil_3001 = 4 := by sorry
  have h3 : ceil_0001 = 1 := by sorry
  rw [h1, h2, h3]
  norm_num

end eval_expression_l71_71949


namespace one_plus_f_prime_at_one_eq_neg_three_l71_71602

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 3 * x * (f' 2)

-- Define the derivative of f
def f' (x : ℝ) : ℝ :=
  if x = 2 then 4 + 3 * f' 2
  else 2 * x + 3 * f' 2

-- State the proof problem: 1 + f'(1) = -3
theorem one_plus_f_prime_at_one_eq_neg_three : (1 + f' 1) = -3 :=
  sorry

end one_plus_f_prime_at_one_eq_neg_three_l71_71602


namespace sum_of_roots_of_equation_l71_71086

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l71_71086


namespace monotonic_decreasing_interval_l71_71629

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem monotonic_decreasing_interval (φ k : ℝ) (k : ℤ) 
  (hφ1 : 0 < φ) (hφ2 : φ < Real.pi / 2) 
  (hsym : f (3 * Real.pi / 8) φ = 0) :
  ∃ a b : ℝ, (a = k * Real.pi + Real.pi / 8) ∧ (b = k * Real.pi + 5 * Real.pi / 8) ∧ 
             ∀ x : ℝ, x ∈ Set.Icc a b → StrictAntiOn (f x φ) :=
by sorry

end monotonic_decreasing_interval_l71_71629


namespace three_letter_list_product_l71_71944

def letter_value (c : Char) : ℕ :=
  if 'A' ≤ c ∧ c ≤ 'Z' then c.to_nat - 'A'.to_nat + 1 else 0

/--
The only other three-letter list with a product equal to the product of the list BDF is BCH.
-/
theorem three_letter_list_product :
  ∃ (list : List Char), list = ['B', 'C', 'H'] ∧
    list = list.sorted ∧   -- ensuring alphabetical order
    (∀ c ∈ list, letter_value c > 0 ∧ letter_value c ≤ 26) ∧
    (list.foldl (λ acc c => acc * (letter_value c)) 1 = 48) ∧
    list ≠ ['B', 'D', 'F'] :=
sorry

end three_letter_list_product_l71_71944


namespace max_lambda_l71_71630

noncomputable theory
open_locale classical

variables {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Define the arithmetic sequence and its sum
axiom arithmetic_seq (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  : ∀ n, S n = n * (a 1 + a n) / 2

-- The given conditions
axiom condition_1 (a : ℕ → ℝ) (h_nonzero : ∀ n, a n ≠ 0)
axiom condition_2 (a : ℕ → ℝ) (S : ℕ → ℝ) (h_sum : ∀ n : ℕ, a (n + 1) ^ 2 = S (2 * (n + 1) - 1))
axiom condition_3 (a : ℕ → ℝ) (h_ineq : ∀ n : ℕ, ∃ (λ : ℝ), λ ≤ (n + 8 * (-1) ^ n) / n * a (n + 1))

-- Proving the maximum value of λ
theorem max_lambda (a : ℕ → ℝ) (S : ℕ → ℝ) (h : ∀ n : ℕ, ∃ (λ : ℝ), λ ≤ (n + 8 * (-1) ^ n) / (n * a (n + 1))) :
  -21 = λ :=
begin
  sorry -- Proof omitted
end

end max_lambda_l71_71630


namespace max_area_rectangle_l71_71516

theorem max_area_rectangle :
  ∃ (l w : ℕ), (2 * (l + w) = 40) ∧ (l ≥ w + 3) ∧ (l * w = 91) :=
by
  sorry

end max_area_rectangle_l71_71516


namespace total_friends_l71_71757

theorem total_friends (total_bill share additional_share total_paid : ℤ) (n : ℕ) 
  (h1 : total_bill = 650)
  (h2 : share = 65)
  (h3 : additional_share = 50)
  (h4 : total_paid = 115)
  (h5 : total_paid - additional_share = share)
  (h6 : total_bill = n * share) :
  n + 1 = 11 := 
begin
  sorry
end

end total_friends_l71_71757


namespace meal_cost_l71_71447

variable (s c p : ℝ)

axiom cond1 : 5 * s + 8 * c + p = 5.00
axiom cond2 : 7 * s + 12 * c + p = 7.20
axiom cond3 : 4 * s + 6 * c + 2 * p = 6.00

theorem meal_cost : s + c + p = 1.90 :=
by
  sorry

end meal_cost_l71_71447


namespace total_interest_received_l71_71881

-- Definitions according to the conditions
def principal_b : ℝ := 5000
def principal_c : ℝ := 3000
def time_b : ℝ := 2
def time_c : ℝ := 4
def rate : ℝ := 10 / 100  -- 10% converted to decimal

-- Simple interest formula
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Interest received from B
def interest_b : ℝ := simple_interest principal_b rate time_b

-- Interest received from C
def interest_c : ℝ := simple_interest principal_c rate time_c

-- Total interest
def total_interest : ℝ := interest_b + interest_c

-- Theorem to prove total interest
theorem total_interest_received : total_interest = 2200 := by
  sorry

end total_interest_received_l71_71881


namespace lucas_150_mod_5_l71_71784

def mod_lucas_seq : ℕ → ℕ
| 1       := 2
| 2       := 1
| (n + 1) := (mod_lucas_seq n + mod_lucas_seq (n - 1)) % 5

theorem lucas_150_mod_5 :
  (mod_lucas_seq 150) % 5 = 1 :=
sorry

end lucas_150_mod_5_l71_71784


namespace non_science_majors_percentage_l71_71322

-- Definitions of conditions
def women_percentage (class_size : ℝ) : ℝ := 0.6 * class_size
def men_percentage (class_size : ℝ) : ℝ := 0.4 * class_size

def women_science_majors (class_size : ℝ) : ℝ := 0.2 * women_percentage class_size
def men_science_majors (class_size : ℝ) : ℝ := 0.7 * men_percentage class_size

def total_science_majors (class_size : ℝ) : ℝ := women_science_majors class_size + men_science_majors class_size

-- Theorem to prove the percentage of the class that are non-science majors is 60%
theorem non_science_majors_percentage (class_size : ℝ) : total_science_majors class_size / class_size = 0.4 → (class_size - total_science_majors class_size) / class_size = 0.6 := 
by
  sorry

end non_science_majors_percentage_l71_71322


namespace remainder_when_divided_by_29_l71_71888

theorem remainder_when_divided_by_29 (N : ℤ) (k : ℤ) (h : N = 751 * k + 53) : 
  N % 29 = 24 := 
by 
  sorry

end remainder_when_divided_by_29_l71_71888


namespace smallest_positive_period_of_f_axis_of_symmetry_one_of_f_l71_71777

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos x)^2 + Real.cos (2 * x + Real.pi / 3) - 1

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x :=
  sorry

theorem axis_of_symmetry_one_of_f :
  let a1 := 5 * Real.pi / 12 in
  let a2 := 11 * Real.pi / 12 in
  (∀ x ∈ Set.Icc 0 Real.pi, f x = f (2 * a1 - x) ∨ f x = f (2 * a2 - x)) :=
  sorry

end smallest_positive_period_of_f_axis_of_symmetry_one_of_f_l71_71777


namespace count_four_digit_int_with_5_or_6_l71_71648

-- Defining the set of four-digit positive integers
def four_digit_integers : Finset ℕ := Finset.Icc 1000 9999

-- Defining a function to check whether a number has at least one digit 5 or 6
def has_digit_5_or_6 (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.any (λ d, d = 5 ∨ d = 6)

-- Counting all four-digit integers with the condition
theorem count_four_digit_int_with_5_or_6 : ∑ n in four_digit_integers.filter has_digit_5_or_6, (1 : ℕ) = 5416 :=
by sorry

end count_four_digit_int_with_5_or_6_l71_71648


namespace raj_is_older_than_ravi_l71_71715

theorem raj_is_older_than_ravi
  (R V H L x : ℕ)
  (h1 : R = V + x)
  (h2 : H = V - 2)
  (h3 : R = 3 * L)
  (h4 : H * 2 = 3 * L)
  (h5 : 20 = (4 * H) / 3) :
  x = 13 :=
by
  sorry

end raj_is_older_than_ravi_l71_71715


namespace taxi_position_at_6pm_taxi_fuel_consumption_l71_71896

/-- Conditions and declarations --/
def distances : List Int := [10, 8, -7, 12, -15, -9, 16]
def fuel_consumption_per_km : Float := 0.2

theorem taxi_position_at_6pm :
  List.sum distances = 15 :=
sorry

theorem taxi_fuel_consumption :
  fuel_consumption_per_km * (List.sum (List.map Int.natAbs distances)) = 15.4 :=
sorry

end taxi_position_at_6pm_taxi_fuel_consumption_l71_71896


namespace scientific_notation_103M_l71_71206

theorem scientific_notation_103M : 103000000 = 1.03 * 10^8 := sorry

end scientific_notation_103M_l71_71206


namespace Debby_bought_bottles_l71_71933

theorem Debby_bought_bottles :
  (5 : ℕ) * (71 : ℕ) = 355 :=
by
  -- Math proof goes here
  sorry

end Debby_bought_bottles_l71_71933


namespace first_day_exceeds_500_l71_71323

noncomputable def bacterial_population (n : ℕ) : ℕ :=
  4 * 3^n

theorem first_day_exceeds_500 : ∃ n, bacterial_population n > 500 ∧ ∀ m < n, bacterial_population m <= 500 :=
by
  use 6
  split
  · sorry -- here we would show bacterial_population 6 > 500
  · intro m
    intro hm
    sorry -- here we would show bacterial_population m <= 500 for m < 6

end first_day_exceeds_500_l71_71323


namespace triangle_proof_unique_condition_area_under_condition_2_l71_71347

variables {a b c : ℝ} 

-- Given the condition 
def given_condition : Prop :=
  b^2 + c^2 = a^2 + sqrt 3 * b * c

-- Measure of angle A
def angle_A := Real.arccos (sqrt 3 / 2)

-- Conditions
def cond_1 := (Real.sin (Real.arccos (sqrt 2 / 2)) = sqrt 2 / 2) ∧ (b = sqrt 2)
def cond_2 := (Real.cos (Real.arccos (2 * sqrt 2 / 3)) = 2 * sqrt 2 / 3) ∧ (a = sqrt 2)
def cond_3 := (a = 1) ∧ (b = sqrt 2)

-- Function to check if a triangle exists under given conditions
def triangle_exist (cond : Prop) : Prop :=
  ∃ A B C, 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π ∧ cond

-- Calculate area of triangle given a, b, and c
noncomputable def area_triangle_ABC (a b c : ℝ) : ℝ :=
  (1 / 2) * a * b * (sqrt 3 / 2)

-- The Lean statement to prove
theorem triangle_proof :
  given_condition →
  angle_A = π / 6 :=
sorry

theorem unique_condition :
  (triangle_exist cond_2) ∧ ¬(triangle_exist cond_1) ∧ ¬(triangle_exist cond_3) :=
sorry

theorem area_under_condition_2 :
  cond_2 → area_triangle_ABC (sqrt 2) (2 * sqrt 2 / 3) c = (2 * sqrt 2 + sqrt 3) / 9 :=
sorry

end triangle_proof_unique_condition_area_under_condition_2_l71_71347


namespace allocation_schemes_correct_l71_71418

noncomputable def allocation_schemes : Nat :=
  let C (n k : Nat) : Nat := Nat.choose n k
  -- Calculate category 1: one school gets 1 professor, two get 2 professors each
  let category1 := C 3 1 * C 5 1 * C 4 2 * C 2 2 / 2
  -- Calculate category 2: one school gets 3 professors, two get 1 professor each
  let category2 := C 3 1 * C 5 3 * C 2 1 * C 1 1 / 2
  -- Total allocation ways
  let totalWays := 6 * (category1 + category2)
  totalWays

theorem allocation_schemes_correct : allocation_schemes = 900 := by
  sorry

end allocation_schemes_correct_l71_71418


namespace optimal_square_length_l71_71144

def volume (x : ℝ) : ℝ := (48 - 2 * x) * (36 - 2 * x) * x

theorem optimal_square_length :
  ∃ x : ℝ, volume x = 3456 ∧ x = 12 :=
begin
  sorry
end

end optimal_square_length_l71_71144


namespace three_digit_decimal_bounds_l71_71897

-- Define the decimal number
def isThreeDigitDecimal (x : ℝ) : Prop := 
  (x * 1000).floor / 1000 = x

-- Define the rounding condition
def roundsTo  (x : ℝ) (t : ℝ) : Prop := 
  (x * 100).round / 100 = t

-- State the problem in Lean
theorem three_digit_decimal_bounds:
  ∀ (x : ℝ), isThreeDigitDecimal x → roundsTo x 4.10 → 4.095 ≤ x ∧ x ≤ 4.104 :=
by {
  intros,
  sorry
}

end three_digit_decimal_bounds_l71_71897


namespace problem_solution_l71_71344

variables {A B C D M N P : Type}
variables (a b c : ℝ^3)

-- Conditions encoded as lean definitions
-- Coordinates for A, B, C, and D
def coord_B : ℝ^3 := ⟨1, 0, 0⟩
def coord_C : ℝ^3 := ⟨1/2, (sqrt 3) / 2, 0⟩
def coord_A : ℝ^3 := ⟨1/2, (sqrt 3) / 6, (sqrt 6) / 3⟩
def coord_D : ℝ^3 := ⟨1/2, (sqrt 3) / 6, (-sqrt 6) / 3⟩

-- Centroid calculation for ⟨triangle⟩ ADC
def centroid_M : ℝ^3 := (coord_A + coord_D + coord_C) / 3

-- Centroid calculation for ⟨triangle⟩ BDC
def centroid_N : ℝ^3 := (coord_B + coord_D + coord_C) / 3

-- Let P satisfy given condition
def coord_P (x y z : ℝ) (a b c : ℝ^3) : ℝ^3 := x • a + y • b + z • c

-- Main theorem to prove
theorem problem_solution 
  (x y z : ℝ)
  (h1 : coord_P x y z a b c = ⟨1/2, (sqrt 3) / 3, - (sqrt 6) / 9⟩)
  (h2 : (centroid_M - coord_P x y z a b c) = 2 • (coord_P x y z a b c - centroid_N)) :
  9 * x + 81 * y + 729 * z = 439 := by
  sorry

end problem_solution_l71_71344
