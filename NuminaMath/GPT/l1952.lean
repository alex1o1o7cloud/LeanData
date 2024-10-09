import Mathlib

namespace p_and_q_and_not_not_p_or_q_l1952_195246

theorem p_and_q_and_not_not_p_or_q (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬(¬p ∨ q)) : p ∧ ¬q :=
by
  sorry

end p_and_q_and_not_not_p_or_q_l1952_195246


namespace arithmetic_mean_of_fractions_l1952_195254
-- Import the Mathlib library to use fractional arithmetic

-- Define the problem in Lean
theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 :=
by
  let a : ℚ := 3 / 8
  let b : ℚ := 5 / 9
  have := (a + b) / 2 = 67 / 144
  sorry

end arithmetic_mean_of_fractions_l1952_195254


namespace part1_part2_l1952_195266

-- Conditions
def U := ℝ
def A : Set ℝ := {x | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 2}
def B (m : ℝ) : Set ℝ := {x | x ≤ 3 * m - 4 ∨ x ≥ 8 + m}
def complement_U (B : Set ℝ) : Set ℝ := {x | ¬(x ∈ B)}
def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

-- Assertions
theorem part1 (m : ℝ) (h1 : m = 2) : intersection A (complement_U (B m)) = {x | 2 < x ∧ x < 4} :=
  sorry

theorem part2 (h : intersection A (complement_U (B m)) = ∅) : -4 ≤ m ∧ m ≤ 5 / 3 :=
  sorry

end part1_part2_l1952_195266


namespace range_of_f_l1952_195213

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.cos x) + Real.arccos (Real.sin x)

theorem range_of_f : Set.range f = Set.Icc 0 Real.pi :=
sorry

end range_of_f_l1952_195213


namespace kenny_total_liquid_l1952_195268

def total_liquid (oil_per_recipe water_per_recipe : ℚ) (times : ℕ) : ℚ :=
  (oil_per_recipe + water_per_recipe) * times

theorem kenny_total_liquid :
  total_liquid 0.17 1.17 12 = 16.08 := by
  sorry

end kenny_total_liquid_l1952_195268


namespace union_inter_example_l1952_195230

noncomputable def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
noncomputable def B : Set ℕ := {4, 7, 8, 9}

theorem union_inter_example :
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ (A ∩ B = {4, 7, 8}) :=
by
  sorry

end union_inter_example_l1952_195230


namespace find_b_l1952_195233

variable (a b c : ℝ)
variable (sin cos : ℝ → ℝ)

-- Assumptions or Conditions
variables (h1 : a^2 - c^2 = 2 * b) 
variables (h2 : sin (b) = 4 * cos (a) * sin (c))

theorem find_b (h1 : a^2 - c^2 = 2 * b) (h2 : sin (b) = 4 * cos (a) * sin (c)) : b = 4 := 
by
  sorry

end find_b_l1952_195233


namespace sandwiches_count_l1952_195211

def total_sandwiches : ℕ :=
  let meats := 12
  let cheeses := 8
  let condiments := 5
  meats * (Nat.choose cheeses 2) * condiments

theorem sandwiches_count : total_sandwiches = 1680 := by
  sorry

end sandwiches_count_l1952_195211


namespace prop1_prop2_prop3_prop4_final_l1952_195255

variables (a b c : ℝ) (h_a : a ≠ 0)

-- Proposition ①
theorem prop1 (h1 : a + b + c = 0) : b^2 - 4 * a * c ≥ 0 := 
sorry

-- Proposition ②
theorem prop2 (h2 : ∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) : 2 * a + c = 0 := 
sorry

-- Proposition ③
theorem prop3 (h3 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + c = 0 ∧ a * x2^2 + c = 0) : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
sorry

-- Proposition ④
theorem prop4 (h4 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ ∃! x : ℝ, a * x^2 + b * x + c = 0) : ¬ (∃ x : ℝ, a * x^2 + b * x + c = 1 ∧ a * x^2 + b * x + 1 = 0) :=
sorry

-- Collectively checking that ①, ②, and ③ are true, and ④ is false
theorem final (h1 : a + b + c = 0)
              (h2 : ∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0)
              (h3 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + c = 0 ∧ a * x2^2 + c = 0)
              (h4 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ ∃! x : ℝ, a * x^2 + b * x + c = 0) : 
  (b^2 - 4 * a * c ≥ 0 ∧ 2 * a + c = 0 ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) ∧ 
  ¬ (∃ x : ℝ, a * x^2 + b * x + c = 1 ∧ a * x^2 + b * x + 1 = 0)) :=
sorry

end prop1_prop2_prop3_prop4_final_l1952_195255


namespace updated_mean_corrected_l1952_195224

theorem updated_mean_corrected (mean observations decrement : ℕ) 
  (h1 : mean = 350) (h2 : observations = 100) (h3 : decrement = 63) :
  (mean * observations + decrement * observations) / observations = 413 :=
by
  sorry

end updated_mean_corrected_l1952_195224


namespace total_cost_of_supplies_l1952_195229

variable (E P M : ℝ)

open Real

theorem total_cost_of_supplies (h1 : E + 3 * P + 2 * M = 240)
                                (h2 : 2 * E + 4 * M + 5 * P = 440)
                                : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_of_supplies_l1952_195229


namespace fencing_required_l1952_195283

theorem fencing_required (L W A F : ℝ) (hL : L = 20) (hA : A = 390) (hArea : A = L * W) (hF : F = 2 * W + L) : F = 59 :=
by
  sorry

end fencing_required_l1952_195283


namespace binomial_probability_4_l1952_195232

noncomputable def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ := 
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem binomial_probability_4 (n : ℕ) (p : ℝ) (ξ : ℕ → ℝ)
  (H1 : (ξ 0) = (n*p))
  (H2 : (ξ 1) = (n*p*(1-p))) :
  binomial_pmf n 4 p = 10 / 243 :=
by {
  sorry 
}

end binomial_probability_4_l1952_195232


namespace votes_cast_l1952_195204

theorem votes_cast (V : ℝ) (h1 : ∃ V, (0.65 * V) = (0.35 * V + 2340)) : V = 7800 :=
by
  sorry

end votes_cast_l1952_195204


namespace infinite_geometric_series_sum_l1952_195244

theorem infinite_geometric_series_sum (a r S : ℚ) (ha : a = 1 / 4) (hr : r = 1 / 3) :
  (S = a / (1 - r)) → (S = 3 / 8) :=
by
  sorry

end infinite_geometric_series_sum_l1952_195244


namespace sin_double_angle_neg_l1952_195249

variable (α : Real)
variable (h1 : Real.tan α < 0)
variable (h2 : Real.sin α = -Real.sqrt 3 / 3)

theorem sin_double_angle_neg (h1 : Real.tan α < 0) (h2 : Real.sin α = -Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = -2 * Real.sqrt 2 / 3 := 
by 
  sorry

end sin_double_angle_neg_l1952_195249


namespace ratio_sums_is_five_sixths_l1952_195209

theorem ratio_sums_is_five_sixths
  (a b c x y z : ℝ)
  (h_positive_a : a > 0) (h_positive_b : b > 0) (h_positive_c : c > 0)
  (h_positive_x : x > 0) (h_positive_y : y > 0) (h_positive_z : z > 0)
  (h₁ : a^2 + b^2 + c^2 = 25)
  (h₂ : x^2 + y^2 + z^2 = 36)
  (h₃ : a * x + b * y + c * z = 30) :
  (a + b + c) / (x + y + z) = (5 / 6) :=
sorry

end ratio_sums_is_five_sixths_l1952_195209


namespace mother_returns_to_freezer_l1952_195263

noncomputable def probability_return_to_freezer : ℝ :=
  1 - ((5 / 17) * (4 / 16) * (3 / 15) * (2 / 14) * (1 / 13))

theorem mother_returns_to_freezer :
  abs (probability_return_to_freezer - 0.99979) < 0.00001 :=
by
    sorry

end mother_returns_to_freezer_l1952_195263


namespace johns_quadratic_l1952_195292

theorem johns_quadratic (d e : ℤ) (h1 : d^2 = 16) (h2 : 2 * d * e = -40) : d * e = -20 :=
sorry

end johns_quadratic_l1952_195292


namespace sufficient_not_necessary_l1952_195239

theorem sufficient_not_necessary (x : ℝ) : (x < 1 → x < 2) ∧ (¬(x < 2 → x < 1)) :=
by
  sorry

end sufficient_not_necessary_l1952_195239


namespace marks_difference_l1952_195295

variable (P C M : ℕ)

-- Conditions
def total_marks_more_than_physics := P + C + M > P
def average_chemistry_mathematics := (C + M) / 2 = 65

-- Proof Statement
theorem marks_difference (h1 : total_marks_more_than_physics P C M) (h2 : average_chemistry_mathematics C M) : 
  P + C + M = P + 130 := by
  sorry

end marks_difference_l1952_195295


namespace blocks_tower_l1952_195228

theorem blocks_tower (T H Total : ℕ) (h1 : H = 53) (h2 : Total = 80) (h3 : T + H = Total) : T = 27 :=
by
  -- proof goes here
  sorry

end blocks_tower_l1952_195228


namespace opposite_of_neg_one_third_l1952_195218

noncomputable def a : ℚ := -1 / 3

theorem opposite_of_neg_one_third : -a = 1 / 3 := 
by 
sorry

end opposite_of_neg_one_third_l1952_195218


namespace points_where_star_is_commutative_are_on_line_l1952_195202

def star (a b : ℝ) : ℝ := a * b * (a - b)

theorem points_where_star_is_commutative_are_on_line :
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1} = {p : ℝ × ℝ | p.1 = p.2} :=
by
  sorry

end points_where_star_is_commutative_are_on_line_l1952_195202


namespace second_intersection_of_parabola_l1952_195242

theorem second_intersection_of_parabola (x_vertex_Pi1 x_vertex_Pi2 : ℝ) : 
  (∀ x : ℝ, x = (10 + 13) / 2 → x_vertex_Pi1 = x) →
  (∀ y : ℝ, y = (x_vertex_Pi2 / 2) → x_vertex_Pi1 = y) →
  (x_vertex_Pi2 = 2 * x_vertex_Pi1) →
  (13 + 33) / 2 = x_vertex_Pi2 :=
by
  sorry

end second_intersection_of_parabola_l1952_195242


namespace xiao_yang_correct_answers_l1952_195226

noncomputable def problems_group_a : ℕ := 5
noncomputable def points_per_problem_group_a : ℕ := 8
noncomputable def problems_group_b : ℕ := 12
noncomputable def points_per_problem_group_b_correct : ℕ := 5
noncomputable def points_per_problem_group_b_incorrect : ℤ := -2
noncomputable def total_score : ℕ := 71
noncomputable def correct_answers_group_a : ℕ := 2 -- minimum required
noncomputable def correct_answers_total : ℕ := 13 -- provided correct result by the problem

theorem xiao_yang_correct_answers : correct_answers_total = 13 := by
  sorry

end xiao_yang_correct_answers_l1952_195226


namespace james_total_earnings_l1952_195275

-- Assume the necessary info for January, February, and March earnings
-- Definitions given as conditions in a)
def January_earnings : ℝ := 4000

def February_earnings : ℝ := January_earnings * 1.5 * 1.2

def March_earnings : ℝ := February_earnings * 0.8

-- The total earnings to be calculated
def Total_earnings : ℝ := January_earnings + February_earnings + March_earnings

-- Prove the total earnings is $16960
theorem james_total_earnings : Total_earnings = 16960 := by
  sorry

end james_total_earnings_l1952_195275


namespace friend_spent_more_than_you_l1952_195240

-- Define the total amount spent by both
def total_spent : ℤ := 19

-- Define the amount spent by your friend
def friend_spent : ℤ := 11

-- Define the amount spent by you
def you_spent : ℤ := total_spent - friend_spent

-- Define the difference in spending
def difference_in_spending : ℤ := friend_spent - you_spent

-- Prove that the difference in spending is $3
theorem friend_spent_more_than_you : difference_in_spending = 3 :=
by
  sorry

end friend_spent_more_than_you_l1952_195240


namespace arcsin_sqrt2_div2_l1952_195285

theorem arcsin_sqrt2_div2 :
  Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
sorry

end arcsin_sqrt2_div2_l1952_195285


namespace minimum_banks_needed_l1952_195210

-- Condition definitions
def total_amount : ℕ := 10000000
def max_insurance_payout_per_bank : ℕ := 1400000

-- Theorem statement
theorem minimum_banks_needed :
  ∃ n : ℕ, n * max_insurance_payout_per_bank ≥ total_amount ∧ n = 8 :=
sorry

end minimum_banks_needed_l1952_195210


namespace problem_a_problem_b_problem_c_problem_d_l1952_195247

def rotate (n : Nat) : Nat := 
  sorry -- Function definition for rotating the last digit to the start
def add_1001 (n : Nat) : Nat := 
  sorry -- Function definition for adding 1001
def subtract_1001 (n : Nat) : Nat := 
  sorry -- Function definition for subtracting 1001

theorem problem_a :
  ∃ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) ∧ (List.foldl (λacc step => step acc) 202122 steps = 313233) :=
sorry

theorem problem_b :
  ∃ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) ∧ (steps.length = 8) ∧ (List.foldl (λacc step => step acc) 999999 steps = 000000) :=
sorry

theorem problem_c (n : Nat) (hn : n % 11 = 0) : 
  ∀ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) → (List.foldl (λacc step => step acc) n steps) % 11 = 0 :=
sorry

theorem problem_d : 
  ∀ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) → ¬(List.foldl (λacc step => step acc) 112233 steps = 000000) :=
sorry

end problem_a_problem_b_problem_c_problem_d_l1952_195247


namespace number_of_parallel_lines_l1952_195265

theorem number_of_parallel_lines (n : ℕ) (h : (n * (n - 1) / 2) * (8 * 7 / 2) = 784) : n = 8 :=
sorry

end number_of_parallel_lines_l1952_195265


namespace discount_price_l1952_195223

theorem discount_price (P : ℝ) (h : P > 0) (discount : ℝ) (h_discount : discount = 0.80) : 
  (P - P * discount) = P * 0.20 :=
by
  sorry

end discount_price_l1952_195223


namespace book_cost_l1952_195259

theorem book_cost (initial_money : ℕ) (remaining_money : ℕ) (num_books : ℕ) 
  (h1 : initial_money = 79) (h2 : remaining_money = 16) (h3 : num_books = 9) :
  (initial_money - remaining_money) / num_books = 7 :=
by
  sorry

end book_cost_l1952_195259


namespace sin_over_sin_l1952_195203

theorem sin_over_sin (a : Real) (h_cos : Real.cos (Real.pi / 4 - a) = 12 / 13)
  (h_quadrant : 0 < Real.pi / 4 - a ∧ Real.pi / 4 - a < Real.pi / 2) :
  Real.sin (Real.pi / 2 - 2 * a) / Real.sin (Real.pi / 4 + a) = 119 / 144 := by
sorry

end sin_over_sin_l1952_195203


namespace chewbacca_gum_l1952_195262

variable {y : ℝ}

theorem chewbacca_gum (h1 : 25 - 2 * y ≠ 0) (h2 : 40 + 4 * y ≠ 0) :
    25 - 2 * y/40 = 25/(40 + 4 * y) → y = 2.5 :=
by
  intros h
  sorry

end chewbacca_gum_l1952_195262


namespace smallest_m_for_integral_solutions_l1952_195252

theorem smallest_m_for_integral_solutions :
  ∃ m : ℕ, m > 0 ∧ (∃ p q : ℤ, 10 * p * q = 660 ∧ p + q = m/10) ∧ m = 170 :=
by
  sorry

end smallest_m_for_integral_solutions_l1952_195252


namespace equivalent_angle_l1952_195274

theorem equivalent_angle (theta : ℤ) (k : ℤ) : 
  (∃ k : ℤ, (-525 + k * 360 = 195)) :=
by
  sorry

end equivalent_angle_l1952_195274


namespace distance_between_trees_l1952_195281

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (yard_length_eq : yard_length = 325) (num_trees_eq : num_trees = 26) :
  (yard_length / (num_trees - 1)) = 13 := by
  sorry

end distance_between_trees_l1952_195281


namespace correct_ranking_l1952_195250

-- Definitions for the colleagues
structure Colleague :=
  (name : String)
  (seniority : ℕ)

-- Colleagues: Julia, Kevin, Lana
def Julia := Colleague.mk "Julia" 1
def Kevin := Colleague.mk "Kevin" 0
def Lana := Colleague.mk "Lana" 2

-- Statements definitions
def Statement_I (c1 c2 c3 : Colleague) := c2.seniority < c1.seniority ∧ c1.seniority < c3.seniority 
def Statement_II (c1 c2 c3 : Colleague) := c1.seniority > c3.seniority
def Statement_III (c1 c2 c3 : Colleague) := c1.seniority ≠ c1.seniority

-- Exactly one of the statements is true
def Exactly_One_True (s1 s2 s3 : Prop) := (s1 ∨ s2 ∨ s3) ∧ ¬(s1 ∧ s2 ∨ s1 ∧ s3 ∨ s2 ∧ s3) ∧ ¬(s1 ∧ s2 ∧ s3)

-- The theorem to be proved
theorem correct_ranking :
  Exactly_One_True (Statement_I Kevin Lana Julia) (Statement_II Kevin Lana Julia) (Statement_III Kevin Lana Julia) →
  (Kevin.seniority < Lana.seniority ∧ Lana.seniority < Julia.seniority) := 
  by  sorry

end correct_ranking_l1952_195250


namespace number_of_tea_bags_l1952_195225

theorem number_of_tea_bags (n : ℕ) 
  (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n)
  (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) :
  n = 20 :=
by
  sorry

end number_of_tea_bags_l1952_195225


namespace triangle_inequality_condition_l1952_195208

theorem triangle_inequality_condition (a b : ℝ) (h : a + b = 1) (ha : a ≥ 0) (hb : b ≥ 0) :
    a + b > 1 → a + 1 > b ∧ b + 1 > a := by
  sorry

end triangle_inequality_condition_l1952_195208


namespace sequence_is_arithmetic_sum_of_sequence_l1952_195217

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = 3 * a n + 2 * 3 ^ (n + 1)

def arithmetic_seq (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n, (a (n + 1) / 3 ^ (n + 1)) - (a n / 3 ^ n) = c

def sum_S (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = (n - 1) * 3 ^ (n + 1) + 3

theorem sequence_is_arithmetic (a : ℕ → ℕ)
  (h : sequence_a a) : 
  arithmetic_seq a 2 :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : sequence_a a) :
  sum_S a S :=
sorry

end sequence_is_arithmetic_sum_of_sequence_l1952_195217


namespace find_y_l1952_195279

theorem find_y (x y : ℚ) (h1 : x = 151) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 342200) : 
  y = 342200 / 3354151 :=
by
  sorry

end find_y_l1952_195279


namespace completing_the_square_l1952_195238

theorem completing_the_square (x : ℝ) : 
  x^2 - 2 * x = 9 → (x - 1)^2 = 10 :=
by
  intro h
  sorry

end completing_the_square_l1952_195238


namespace system_of_equations_solution_l1952_195282

theorem system_of_equations_solution (x y : ℝ) (h1 : 2 * x ^ 2 - 5 * x + 3 = 0) (h2 : y = 3 * x + 1) : 
  (x = 1.5 ∧ y = 5.5) ∨ (x = 1 ∧ y = 4) :=
sorry

end system_of_equations_solution_l1952_195282


namespace middle_number_is_12_l1952_195235

theorem middle_number_is_12 (x y z : ℕ) (h1 : x + y = 20) (h2 : x + z = 25) (h3 : y + z = 29) (h4 : x < y) (h5 : y < z) : y = 12 :=
by
  sorry

end middle_number_is_12_l1952_195235


namespace stamps_in_last_page_l1952_195287

-- Define the total number of books, pages per book, and stamps per original page.
def total_books : ℕ := 6
def pages_per_book : ℕ := 30
def original_stamps_per_page : ℕ := 7

-- Define the new stamps per page after reorganization.
def new_stamps_per_page : ℕ := 9

-- Define the number of fully filled books and pages in the fourth book.
def filled_books : ℕ := 3
def pages_in_fourth_book : ℕ := 26

-- Define the total number of stamps originally.
def total_original_stamps : ℕ := total_books * pages_per_book * original_stamps_per_page

-- Prove that the last page in the fourth book contains 9 stamps under the given conditions.
theorem stamps_in_last_page : 
  total_original_stamps / new_stamps_per_page - (filled_books * pages_per_book + pages_in_fourth_book) * new_stamps_per_page = 9 :=
by
  sorry

end stamps_in_last_page_l1952_195287


namespace range_of_independent_variable_l1952_195206

theorem range_of_independent_variable (x : ℝ) : (x - 4) ≠ 0 ↔ x ≠ 4 :=
by
  sorry

end range_of_independent_variable_l1952_195206


namespace oxygen_mass_percentage_is_58_3_l1952_195200

noncomputable def C_molar_mass := 12.01
noncomputable def H_molar_mass := 1.01
noncomputable def O_molar_mass := 16.0

noncomputable def molar_mass_C6H8O7 :=
  6 * C_molar_mass + 8 * H_molar_mass + 7 * O_molar_mass

noncomputable def O_mass := 7 * O_molar_mass

noncomputable def oxygen_mass_percentage_C6H8O7 :=
  (O_mass / molar_mass_C6H8O7) * 100

theorem oxygen_mass_percentage_is_58_3 :
  oxygen_mass_percentage_C6H8O7 = 58.3 := by
  sorry

end oxygen_mass_percentage_is_58_3_l1952_195200


namespace mr_johnson_fencing_l1952_195280

variable (Length Width : ℕ)

def perimeter_of_rectangle (Length Width : ℕ) : ℕ :=
  2 * (Length + Width)

theorem mr_johnson_fencing
  (hLength : Length = 25)
  (hWidth : Width = 15) :
  perimeter_of_rectangle Length Width = 80 := by
  sorry

end mr_johnson_fencing_l1952_195280


namespace find_a_plus_b_l1952_195214

theorem find_a_plus_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a^2 - b^4 = 2009) : a + b = 47 := 
by 
  sorry

end find_a_plus_b_l1952_195214


namespace square_side_length_l1952_195243

variable (s d k : ℝ)

theorem square_side_length {s d k : ℝ} (h1 : s + d = k) (h2 : d = s * Real.sqrt 2) : 
  s = k / (1 + Real.sqrt 2) :=
sorry

end square_side_length_l1952_195243


namespace square_side_length_l1952_195298

theorem square_side_length (s : ℝ) (h : 8 * s^2 = 3200) : s = 20 :=
by
  sorry

end square_side_length_l1952_195298


namespace Benny_spent_95_dollars_l1952_195256

theorem Benny_spent_95_dollars
    (amount_initial : ℕ)
    (amount_left : ℕ)
    (amount_spent : ℕ) :
    amount_initial = 120 →
    amount_left = 25 →
    amount_spent = amount_initial - amount_left →
    amount_spent = 95 :=
by
  intros h_initial h_left h_spent
  rw [h_initial, h_left] at h_spent
  exact h_spent

end Benny_spent_95_dollars_l1952_195256


namespace minimum_value_condition_l1952_195216

-- Define the function y = x^3 - 2ax + a
noncomputable def f (a x : ℝ) : ℝ := x^3 - 2 * a * x + a

-- Define its derivative
noncomputable def f' (a x : ℝ) : ℝ := 3 * x^2 - 2 * a

-- Define the lean theorem statement
theorem minimum_value_condition (a : ℝ) : 
  (∃ x y : ℝ, 0 < x ∧ x < 1 ∧ y = f a x ∧ (∀ z : ℝ, 0 < z ∧ z < 1 → f a z ≥ y)) ∧
  ¬(∃ x y : ℝ, 0 < x ∧ x < 1 ∧ y = f a x ∧ (∀ z : ℝ, 0 < z ∧ z < 1 → f a z < y)) 
  ↔ 0 < a ∧ a < 3 / 2 :=
sorry

end minimum_value_condition_l1952_195216


namespace find_other_root_l1952_195264

theorem find_other_root (z : ℂ) (z_squared : z^2 = -91 + 104 * I) (root1 : z = 7 + 10 * I) : z = -7 - 10 * I :=
by
  sorry

end find_other_root_l1952_195264


namespace pat_stickers_at_end_of_week_l1952_195299

def initial_stickers : ℕ := 39
def monday_transaction : ℕ := 15
def tuesday_transaction : ℕ := 22
def wednesday_transaction : ℕ := 10
def thursday_trade_net_loss : ℕ := 4
def friday_find : ℕ := 5

def final_stickers (initial : ℕ) (mon : ℕ) (tue : ℕ) (wed : ℕ) (thu : ℕ) (fri : ℕ) : ℕ :=
  initial + mon - tue + wed - thu + fri

theorem pat_stickers_at_end_of_week :
  final_stickers initial_stickers 
                 monday_transaction 
                 tuesday_transaction 
                 wednesday_transaction 
                 thursday_trade_net_loss 
                 friday_find = 43 :=
by
  sorry

end pat_stickers_at_end_of_week_l1952_195299


namespace store_owner_uniforms_l1952_195293

theorem store_owner_uniforms (U E : ℕ) (h1 : U + 1 = 2 * E) (h2 : U % 2 = 1) : U = 3 := 
sorry

end store_owner_uniforms_l1952_195293


namespace find_angle_A_l1952_195269

-- Conditions
def is_triangle (A B C : ℝ) : Prop := A + B + C = 180
def B_is_two_C (B C : ℝ) : Prop := B = 2 * C
def B_is_80 (B : ℝ) : Prop := B = 80

-- Theorem statement
theorem find_angle_A (A B C : ℝ) (h₁ : is_triangle A B C) (h₂ : B_is_two_C B C) (h₃ : B_is_80 B) : A = 60 := by
  sorry

end find_angle_A_l1952_195269


namespace betty_books_l1952_195231

variable (B : ℝ)
variable (h : B + (5/4) * B = 45)

theorem betty_books : B = 20 := by
  sorry

end betty_books_l1952_195231


namespace geometric_sequence_product_l1952_195236

theorem geometric_sequence_product 
    (a : ℕ → ℝ)
    (h_geom : ∀ n m, a (n + m) = a n * a m)
    (h_roots : ∀ x, x^2 - 3*x + 2 = 0 → (x = a 7 ∨ x = a 13)) :
  a 2 * a 18 = 2 := 
sorry

end geometric_sequence_product_l1952_195236


namespace Jim_time_to_fill_pool_l1952_195253

-- Definitions for the work rates of Sue, Tony, and their combined work rate.
def Sue_work_rate : ℚ := 1 / 45
def Tony_work_rate : ℚ := 1 / 90
def Combined_work_rate : ℚ := 1 / 15

-- Proving the time it takes for Jim to fill the pool alone.
theorem Jim_time_to_fill_pool : ∃ J : ℚ, 1 / J + Sue_work_rate + Tony_work_rate = Combined_work_rate ∧ J = 30 :=
by {
  sorry
}

end Jim_time_to_fill_pool_l1952_195253


namespace quadratic_function_min_value_at_1_l1952_195257

-- Define the quadratic function y = (x - 1)^2 - 3
def quadratic_function (x : ℝ) : ℝ :=
  (x - 1) ^ 2 - 3

-- The theorem to prove is that this quadratic function reaches its minimum value when x = 1.
theorem quadratic_function_min_value_at_1 : ∃ x : ℝ, quadratic_function x = quadratic_function 1 :=
by
  sorry

end quadratic_function_min_value_at_1_l1952_195257


namespace original_profit_percentage_l1952_195205

-- Our definitions based on conditions.
variables (P S : ℝ)
-- Selling at double the price results in 260% profit
axiom h : (2 * S - P) / P * 100 = 260

-- Prove that the original profit percentage is 80%
theorem original_profit_percentage : (S - P) / P * 100 = 80 := 
sorry

end original_profit_percentage_l1952_195205


namespace max_length_small_stick_l1952_195234

theorem max_length_small_stick (a b c : ℕ) 
  (ha : a = 24) (hb : b = 32) (hc : c = 44) :
  Nat.gcd (Nat.gcd a b) c = 4 :=
by
  rw [ha, hb, hc]
  -- At this point, the gcd calculus will be omitted, filing it with sorry
  sorry

end max_length_small_stick_l1952_195234


namespace compute_abs_ab_eq_2_sqrt_111_l1952_195273

theorem compute_abs_ab_eq_2_sqrt_111 (a b : ℝ) 
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = 2 * Real.sqrt 111 := 
sorry

end compute_abs_ab_eq_2_sqrt_111_l1952_195273


namespace proof_problem_l1952_195258

variables {R : Type*} [Field R] (p q r u v w : R)

theorem proof_problem (h₁ : 15*u + q*v + r*w = 0)
                      (h₂ : p*u + 25*v + r*w = 0)
                      (h₃ : p*u + q*v + 50*w = 0)
                      (hp : p ≠ 15)
                      (hu : u ≠ 0) : 
                      (p / (p - 15) + q / (q - 25) + r / (r - 50)) = 1 := 
by sorry

end proof_problem_l1952_195258


namespace smallest_number_of_students_l1952_195297

theorem smallest_number_of_students 
    (ratio_9th_10th : Nat := 3 / 2)
    (ratio_9th_11th : Nat := 5 / 4)
    (ratio_9th_12th : Nat := 7 / 6) :
  ∃ N9 N10 N11 N12 : Nat, 
  N9 / N10 = 3 / 2 ∧ N9 / N11 = 5 / 4 ∧ N9 / N12 = 7 / 6 ∧ N9 + N10 + N11 + N12 = 349 :=
by {
  sorry
}

#print axioms smallest_number_of_students

end smallest_number_of_students_l1952_195297


namespace jovana_shells_l1952_195278

variable (initial_shells : Nat) (additional_shells : Nat)

theorem jovana_shells (h1 : initial_shells = 5) (h2 : additional_shells = 12) : initial_shells + additional_shells = 17 := 
by 
  sorry

end jovana_shells_l1952_195278


namespace first_discount_percentage_l1952_195222

/-- A theorem to determine the first discount percentage on sarees -/
theorem first_discount_percentage (x : ℝ) (h : 
((400 - (x / 100) * 400) - (8 / 100) * (400 - (x / 100) * 400) = 331.2)) : x = 10 := by
  sorry

end first_discount_percentage_l1952_195222


namespace domain_of_function_l1952_195296

-- Definitions of the conditions
def condition1 (x : ℝ) : Prop := x - 5 ≠ 0
def condition2 (x : ℝ) : Prop := x - 2 > 0

-- The theorem stating the domain of the function
theorem domain_of_function (x : ℝ) : condition1 x ∧ condition2 x ↔ 2 < x ∧ x ≠ 5 :=
by
  sorry

end domain_of_function_l1952_195296


namespace determine_real_numbers_l1952_195251

theorem determine_real_numbers (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
    (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) :=
sorry

end determine_real_numbers_l1952_195251


namespace problem_1_problem_2_l1952_195276

section proof_problem

variables (a b c d : ℤ)
variables (op : ℤ → ℤ → ℤ)
variables (add : ℤ → ℤ → ℤ)

-- Define the given conditions
axiom op_idem : ∀ (a : ℤ), op a a = a
axiom op_zero : ∀ (a : ℤ), op a 0 = 2 * a
axiom op_add : ∀ (a b c d : ℤ), add (op a b) (op c d) = op (a + c) (b + d)

-- Define the problems to prove
theorem problem_1 : add (op 2 3) (op 0 3) = -2 := sorry
theorem problem_2 : op 1024 48 = 2000 := sorry

end proof_problem

end problem_1_problem_2_l1952_195276


namespace basketball_game_points_half_l1952_195294

theorem basketball_game_points_half (a d b r : ℕ) (h_arith_seq : a + (a + d) + (a + 2 * d) + (a + 3 * d) ≤ 100)
    (h_geo_seq : b + b * r + b * r^2 + b * r^3 ≤ 100)
    (h_win_by_two : 4 * a + 6 * d = b * (1 + r + r^2 + r^3) + 2) :
    (a + (a + d)) + (b + b * r) = 14 :=
sorry

end basketball_game_points_half_l1952_195294


namespace necessary_condition_lg_l1952_195201

theorem necessary_condition_lg (x : ℝ) : ¬(x > -1) → ¬(10^1 > x + 1) := by {
    sorry
}

end necessary_condition_lg_l1952_195201


namespace LilyUsed14Dimes_l1952_195220

variable (p n d : ℕ)

theorem LilyUsed14Dimes
  (h1 : p + n + d = 50)
  (h2 : p + 5 * n + 10 * d = 200) :
  d = 14 := by
  sorry

end LilyUsed14Dimes_l1952_195220


namespace f_2011_l1952_195227

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 4) = f x + f 2
axiom f_1 : f 1 = 2

theorem f_2011 : f 2011 = -2 :=
by sorry

end f_2011_l1952_195227


namespace roots_of_quadratic_eq_l1952_195207

theorem roots_of_quadratic_eq {x1 x2 : ℝ} (h1 : x1 * x1 - 3 * x1 - 5 = 0) (h2 : x2 * x2 - 3 * x2 - 5 = 0) 
                              (h3 : x1 + x2 = 3) (h4 : x1 * x2 = -5) : x1^2 + x2^2 = 19 := 
sorry

end roots_of_quadratic_eq_l1952_195207


namespace investment_rate_l1952_195248

theorem investment_rate (r : ℝ) (A : ℝ) (income_diff : ℝ) (total_invested : ℝ) (eight_percent_invested : ℝ) :
  total_invested = 2000 → 
  eight_percent_invested = 750 → 
  income_diff = 65 → 
  A = total_invested - eight_percent_invested → 
  (A * r) - (eight_percent_invested * 0.08) = income_diff → 
  r = 0.1 :=
by
  intros h_total h_eight h_income_diff h_A h_income_eq
  sorry

end investment_rate_l1952_195248


namespace divisible_by_7_of_sum_of_squares_l1952_195219

theorem divisible_by_7_of_sum_of_squares (a b : ℤ) (h : 7 ∣ (a^2 + b^2)) : 
    (7 ∣ a) ∧ (7 ∣ b) :=
sorry

end divisible_by_7_of_sum_of_squares_l1952_195219


namespace find_xy_yz_xz_l1952_195272

-- Define the conditions given in the problem
variables (x y z : ℝ)
variable (hxyz_pos : x > 0 ∧ y > 0 ∧ z > 0)
variable (h1 : x^2 + x * y + y^2 = 12)
variable (h2 : y^2 + y * z + z^2 = 16)
variable (h3 : z^2 + z * x + x^2 = 28)

-- State the theorem to be proved
theorem find_xy_yz_xz : x * y + y * z + x * z = 16 :=
by {
    -- Proof will be done here
    sorry
}

end find_xy_yz_xz_l1952_195272


namespace calc_triple_hash_30_l1952_195290

def hash_fn (N : ℝ) : ℝ := 0.6 * N + 2

theorem calc_triple_hash_30 :
  hash_fn (hash_fn (hash_fn 30)) = 10.4 :=
by 
  -- Proof goes here
  sorry

end calc_triple_hash_30_l1952_195290


namespace find_a_l1952_195241

theorem find_a (a : ℝ) (h : (∃ x : ℝ, (a - 3) * x ^ |a - 2| + 4 = 0) ∧ |a-2| = 1) : a = 1 :=
sorry

end find_a_l1952_195241


namespace line_equation_l1952_195289

-- Given conditions
def param_x (t : ℝ) : ℝ := 3 * t + 6
def param_y (t : ℝ) : ℝ := 5 * t - 7

-- Proof problem: for any real t, the parameterized line can be described by the equation y = 5x/3 - 17.
theorem line_equation (t : ℝ) : ∃ (m b : ℝ), (∃ t : ℝ, param_y t = m * (param_x t) + b) ∧ m = 5 / 3 ∧ b = -17 :=
by
  exists 5 / 3
  exists -17
  sorry

end line_equation_l1952_195289


namespace total_spent_at_music_store_l1952_195284

-- Defining the costs
def clarinet_cost : ℝ := 130.30
def song_book_cost : ℝ := 11.24

-- The main theorem to prove
theorem total_spent_at_music_store : clarinet_cost + song_book_cost = 141.54 :=
by
  sorry

end total_spent_at_music_store_l1952_195284


namespace pages_remaining_l1952_195221

def total_pages : ℕ := 120
def science_project_pages : ℕ := (25 * total_pages) / 100
def math_homework_pages : ℕ := 10
def total_used_pages : ℕ := science_project_pages + math_homework_pages
def remaining_pages : ℕ := total_pages - total_used_pages

theorem pages_remaining : remaining_pages = 80 := by
  sorry

end pages_remaining_l1952_195221


namespace num_int_values_n_terminated_l1952_195270

theorem num_int_values_n_terminated (N : ℕ) (hN1 : 1 ≤ N) (hN2 : N ≤ 500) :
  ∃ n : ℕ, n = 10 ∧ ∀ k, 0 ≤ k → k < n → ∃ (m : ℕ), N = m * 49 :=
sorry

end num_int_values_n_terminated_l1952_195270


namespace sum_c_d_eq_24_l1952_195215

theorem sum_c_d_eq_24 (c d : ℕ) (h_pos_c : c > 0) (h_pos_d : d > 1) (h_max_power : c^d < 500 ∧ ∀ ⦃x y : ℕ⦄, x^y < 500 → x^y ≤ c^d) : c + d = 24 :=
sorry

end sum_c_d_eq_24_l1952_195215


namespace solve_part_one_solve_part_two_l1952_195237

-- Define function f
def f (a x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- Prove for part (1)
theorem solve_part_one : 
  {x : ℝ | -1 / 3 ≤ x ∧ x ≤ 5} = {x : ℝ | f 2 x ≤ 1} :=
by
  -- Replace the proof with sorry
  sorry

-- Prove for part (2)
theorem solve_part_two :
  {a : ℝ | a = 1 ∨ a = -1} = {a : ℝ | ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4} :=
by
  -- Replace the proof with sorry
  sorry

end solve_part_one_solve_part_two_l1952_195237


namespace sequence_general_term_l1952_195271

-- Define the sequence based on the given conditions
def seq (n : ℕ) : ℚ := if n = 0 then 1 else (n : ℚ) / (2 * n - 1)

theorem sequence_general_term (n : ℕ) :
  seq (n + 1) = (n + 1) / (2 * (n + 1) - 1) :=
by
  sorry

end sequence_general_term_l1952_195271


namespace greatest_three_digit_number_l1952_195260

theorem greatest_three_digit_number :
  ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ N % 8 = 2 ∧ N % 7 = 4 ∧ N = 978 :=
by
  sorry

end greatest_three_digit_number_l1952_195260


namespace geometric_sequence_properties_l1952_195288

-- Given conditions as definitions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 * a 3 = a 4 ∧ a 3 = 8

-- Prove the common ratio and the sum of the first n terms
theorem geometric_sequence_properties (a : ℕ → ℝ)
  (h : seq a) :
  (∃ q, ∀ n, a n = a 1 * q ^ (n - 1) ∧ q = 2) ∧
  (∀ S_n, S_n = (1 - (2 : ℝ) ^ S_n) / (1 - 2) ∧ S_n = 2 ^ S_n - 1) :=
by
  sorry

end geometric_sequence_properties_l1952_195288


namespace no_positive_integer_n_eqn_l1952_195291

theorem no_positive_integer_n_eqn (n : ℕ) : (120^5 + 97^5 + 79^5 + 44^5 ≠ n^5) ∨ n = 144 :=
by
  -- Proof omitted for brevity
  sorry

end no_positive_integer_n_eqn_l1952_195291


namespace range_of_c_l1952_195261

noncomputable def is_monotonically_decreasing (c: ℝ) : Prop := ∀ x1 x2: ℝ, x1 < x2 → c^x2 ≤ c^x1

def inequality_holds (c: ℝ) : Prop := ∀ x: ℝ, x^2 + x + (1/2)*c > 0

theorem range_of_c (c: ℝ) (h1: c > 0) :
  ((is_monotonically_decreasing c ∨ inequality_holds c) ∧ ¬(is_monotonically_decreasing c ∧ inequality_holds c)) 
  → (0 < c ∧ c ≤ 1/2 ∨ c ≥ 1) := 
sorry

end range_of_c_l1952_195261


namespace polynomial_product_evaluation_l1952_195277

theorem polynomial_product_evaluation :
  let p1 := (2*x^3 - 3*x^2 + 5*x - 1)
  let p2 := (8 - 3*x)
  let product := p1 * p2
  let a := -6
  let b := 25
  let c := -39
  let d := 43
  let e := -8
  (16 * a + 8 * b + 4 * c + 2 * d + e) = 26 :=
by
  sorry

end polynomial_product_evaluation_l1952_195277


namespace evaluate_expression_l1952_195286

theorem evaluate_expression :
  ((-2: ℤ)^2) ^ (1 ^ (0 ^ 2)) + 3 ^ (0 ^(1 ^ 2)) = 5 :=
by
  -- sorry allows us to skip the proof
  sorry

end evaluate_expression_l1952_195286


namespace mean_of_points_scored_l1952_195267

def mean (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem mean_of_points_scored (lst : List ℕ)
  (h1 : lst = [81, 73, 83, 86, 73]) : 
  mean lst = 79.2 :=
by
  rw [h1, mean]
  sorry

end mean_of_points_scored_l1952_195267


namespace area_of_rectangle_l1952_195212

theorem area_of_rectangle (a b : ℝ) (h1 : 2 * (a + b) = 16) (h2 : 2 * a^2 + 2 * b^2 = 68) :
  a * b = 15 :=
by
  have h3 : a + b = 8 := by sorry
  have h4 : a^2 + b^2 = 34 := by sorry
  have h5 : (a + b) ^ 2 = a^2 + b^2 + 2 * a * b := by sorry
  have h6 : 64 = 34 + 2 * a * b := by sorry
  have h7 : 2 * a * b = 30 := by sorry
  exact sorry

end area_of_rectangle_l1952_195212


namespace simplify_abs_sum_l1952_195245

theorem simplify_abs_sum (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  |c - a - b| + |c + b - a| = 2 * b :=
sorry

end simplify_abs_sum_l1952_195245
