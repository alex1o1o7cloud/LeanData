import Mathlib

namespace trig_identity_l716_71607

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) : 
  ∃ (res : ℝ), res = 10 / 7 ∧ res = Real.sin α / (Real.sin α ^ 3 - Real.cos α ^ 3) := by
  sorry

end trig_identity_l716_71607


namespace Shell_Ratio_l716_71678

-- Definitions of the number of shells collected by Alan, Ben, and Laurie.
variable (A B L : ℕ)

-- Hypotheses based on the given conditions:
-- 1. Alan collected four times as many shells as Ben did.
-- 2. Laurie collected 36 shells.
-- 3. Alan collected 48 shells.
theorem Shell_Ratio (h1 : A = 4 * B) (h2 : L = 36) (h3 : A = 48) : B / Nat.gcd B L = 1 ∧ L / Nat.gcd B L = 3 :=
by
  sorry

end Shell_Ratio_l716_71678


namespace roots_product_eq_l716_71604

theorem roots_product_eq
  (a b m p r : ℚ)
  (h₀ : a * b = 3)
  (h₁ : ∀ x, x^2 - m * x + 3 = 0 → (x = a ∨ x = b))
  (h₂ : ∀ x, x^2 - p * x + r = 0 → (x = a + 1 / b ∨ x = b + 1 / a)) : 
  r = 16 / 3 :=
by
  sorry

end roots_product_eq_l716_71604


namespace prove_tan_sum_is_neg_sqrt3_l716_71615

open Real

-- Given conditions as definitions
def condition1 (α β : ℝ) : Prop := 0 < α ∧ α < π ∧ 0 < β ∧ β < π
def condition2 (α β : ℝ) : Prop := sin α + sin β = sqrt 3 * (cos α + cos β)

-- The statement of the proof
theorem prove_tan_sum_is_neg_sqrt3 (α β : ℝ) (h1 : condition1 α β) (h2 : condition2 α β) :
  tan (α + β) = -sqrt 3 :=
sorry

end prove_tan_sum_is_neg_sqrt3_l716_71615


namespace find_e_l716_71696

theorem find_e (x y e : ℝ) (h1 : x / (2 * y) = 5 / e) (h2 : (7 * x + 4 * y) / (x - 2 * y) = 13) : e = 2 := 
by
  sorry

end find_e_l716_71696


namespace arithmetic_sqrt_of_9_l716_71639

def arithmetic_sqrt (n : ℕ) : ℕ :=
  Nat.sqrt n

theorem arithmetic_sqrt_of_9 : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_9_l716_71639


namespace inequality_of_fractions_l716_71638

theorem inequality_of_fractions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x / (x + y)) + (y / (y + z)) + (z / (z + x)) ≤ 2 := 
by 
  sorry

end inequality_of_fractions_l716_71638


namespace total_ducks_l716_71603

-- Definitions based on the given conditions
def Muscovy : ℕ := 39
def Cayuga : ℕ := Muscovy - 4
def KhakiCampbell : ℕ := (Cayuga - 3) / 2

-- Proof statement
theorem total_ducks : Muscovy + Cayuga + KhakiCampbell = 90 := by
  sorry

end total_ducks_l716_71603


namespace flags_count_l716_71600

-- Define the colors available
inductive Color
| purple | gold | silver

-- Define the number of stripes on the flag
def number_of_stripes : Nat := 3

-- Define a function to calculate the total number of combinations
def total_flags (colors : Nat) (stripes : Nat) : Nat :=
  colors ^ stripes

-- The main theorem we want to prove
theorem flags_count : total_flags 3 number_of_stripes = 27 :=
by
  -- This is the statement only, and the proof is omitted
  sorry

end flags_count_l716_71600


namespace number_of_students_like_photography_l716_71640

variable (n_dislike n_like n_neutral : ℕ)

theorem number_of_students_like_photography :
  (3 * n_dislike = n_dislike + 12) →
  (5 * n_dislike = n_like) →
  n_like = 30 :=
by
  sorry

end number_of_students_like_photography_l716_71640


namespace sin_2alpha_plus_sin_squared_l716_71628

theorem sin_2alpha_plus_sin_squared (α : ℝ) (h : Real.tan α = 1 / 2) : Real.sin (2 * α) + Real.sin α ^ 2 = 1 :=
sorry

end sin_2alpha_plus_sin_squared_l716_71628


namespace count_values_of_b_l716_71616

theorem count_values_of_b : 
  ∃! n : ℕ, (n = 4) ∧ (∀ b : ℕ, (b > 0) → (b ≤ 100) → (∃ k : ℤ, 5 * b^2 + 12 * b + 4 = k^2) → 
    (b = 4 ∨ b = 20 ∨ b = 44 ∨ b = 76)) :=
by
  sorry

end count_values_of_b_l716_71616


namespace third_candidate_votes_l716_71688

theorem third_candidate_votes
  (total_votes : ℝ)
  (votes_for_two_candidates : ℝ)
  (winning_percentage : ℝ)
  (H1 : votes_for_two_candidates = 4636 + 11628)
  (H2 : winning_percentage = 67.21387283236994 / 100)
  (H3 : total_votes = votes_for_two_candidates / (1 - winning_percentage)) :
  (total_votes - votes_for_two_candidates) = 33336 :=
by
  sorry

end third_candidate_votes_l716_71688


namespace percentage_of_singles_l716_71659

/-- In a baseball season, Lisa had 50 hits. Among her hits were 2 home runs, 
2 triples, 8 doubles, and 1 quadruple. The rest of her hits were singles. 
What percent of her hits were singles? --/
theorem percentage_of_singles
  (total_hits : ℕ := 50)
  (home_runs : ℕ := 2)
  (triples : ℕ := 2)
  (doubles : ℕ := 8)
  (quadruples : ℕ := 1)
  (non_singles := home_runs + triples + doubles + quadruples)
  (singles := total_hits - non_singles) :
  (singles : ℚ) / (total_hits : ℚ) * 100 = 74 := by
  sorry

end percentage_of_singles_l716_71659


namespace percent_of_y_l716_71664

theorem percent_of_y (y : ℝ) (hy : y > 0) : (8 * y) / 20 + (3 * y) / 10 = 0.7 * y :=
by
  sorry

end percent_of_y_l716_71664


namespace remaining_watermelons_l716_71680

-- Define the given conditions
def initial_watermelons : ℕ := 35
def watermelons_eaten : ℕ := 27

-- Define the question as a theorem
theorem remaining_watermelons : 
  initial_watermelons - watermelons_eaten = 8 :=
by
  sorry

end remaining_watermelons_l716_71680


namespace bisection_method_root_interval_l716_71651

def f (x : ℝ) : ℝ := x^3 + x - 8

theorem bisection_method_root_interval :
  f 1 < 0 → f 1.5 < 0 → f 1.75 < 0 → f 2 > 0 → ∃ x, (1.75 < x ∧ x < 2 ∧ f x = 0) :=
by
  intros h1 h15 h175 h2
  sorry

end bisection_method_root_interval_l716_71651


namespace calculate_discount_percentage_l716_71655

theorem calculate_discount_percentage :
  ∃ (x : ℝ), (∀ (P S : ℝ),
    (S = 439.99999999999966) →
    (S = 1.10 * P) →
    (1.30 * (1 - x / 100) * P = S + 28) →
    x = 10) :=
sorry

end calculate_discount_percentage_l716_71655


namespace linear_equation_solution_l716_71632

theorem linear_equation_solution (x y b : ℝ) (h1 : x - 2*y + b = 0) (h2 : y = (1/2)*x + b - 1) :
  b = 2 :=
by
  sorry

end linear_equation_solution_l716_71632


namespace students_count_l716_71608

theorem students_count (n : ℕ) (avg_age_n_students : ℕ) (sum_age_7_students1 : ℕ) (sum_age_7_students2 : ℕ) (last_student_age : ℕ) :
  avg_age_n_students = 15 →
  sum_age_7_students1 = 7 * 14 →
  sum_age_7_students2 = 7 * 16 →
  last_student_age = 15 →
  (sum_age_7_students1 + sum_age_7_students2 + last_student_age = avg_age_n_students * n) →
  n = 15 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end students_count_l716_71608


namespace people_left_line_l716_71665

theorem people_left_line (initial new final L : ℕ) 
  (h1 : initial = 30) 
  (h2 : new = 5) 
  (h3 : final = 25) 
  (h4 : initial - L + new = final) : L = 10 := by
  sorry

end people_left_line_l716_71665


namespace tv_sales_value_increase_l716_71689

theorem tv_sales_value_increase (P V : ℝ) :
    let P1 := 0.82 * P
    let V1 := 1.72 * V
    let P2 := 0.75 * P1
    let V2 := 1.90 * V1
    let initial_sales := P * V
    let final_sales := P2 * V2
    final_sales = 2.00967 * initial_sales :=
by
  sorry

end tv_sales_value_increase_l716_71689


namespace functional_equation_solution_l716_71652

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x) + f (f y) = 2 * y + f (x - y)) ↔ (∀ x : ℝ, f x = x) := by
  sorry

end functional_equation_solution_l716_71652


namespace coordinates_of_points_l716_71672

theorem coordinates_of_points
  (R : ℝ) (a b : ℝ)
  (hR : R = 10)
  (h_area : 1/2 * a * b = 600)
  (h_a_gt_b : a > b) :
  (a, 0) = (40, 0) ∧ (0, b) = (0, 30) ∧ (16, 18) = (16, 18) :=
  sorry

end coordinates_of_points_l716_71672


namespace monotonically_increasing_a_range_l716_71676

noncomputable def f (a x : ℝ) : ℝ := (a * x - 1) * Real.exp x

theorem monotonically_increasing_a_range :
  ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≥ 0) ↔ 1 ≤ a  :=
by
  sorry

end monotonically_increasing_a_range_l716_71676


namespace total_number_of_orders_l716_71635

-- Define the conditions
def num_original_programs : Nat := 6
def num_added_programs : Nat := 3

-- State the theorem
theorem total_number_of_orders : ∃ n : ℕ, n = 210 :=
by
  -- This is where the proof would go
  sorry

end total_number_of_orders_l716_71635


namespace problem_2011_Mentougou_l716_71662

theorem problem_2011_Mentougou 
  (f : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (H2 : ∀ x : ℝ, 0 < x → 0 < f x) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :=
sorry

end problem_2011_Mentougou_l716_71662


namespace horizontal_distance_l716_71691

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - x^2 - x - 6

-- Condition: y-coordinate of point P is 8
def P_y : ℝ := 8

-- Condition: y-coordinate of point Q is -8
def Q_y : ℝ := -8

-- x-coordinates of points P and Q solve these equations respectively
def P_satisfies (x : ℝ) : Prop := curve x = P_y
def Q_satisfies (x : ℝ) : Prop := curve x = Q_y

-- The horizontal distance between P and Q is 1
theorem horizontal_distance : ∃ (Px Qx : ℝ), P_satisfies Px ∧ Q_satisfies Qx ∧ |Px - Qx| = 1 :=
by
  sorry

end horizontal_distance_l716_71691


namespace range_of_abscissa_l716_71630

/--
Given three points A, F1, F2 in the Cartesian plane and a point P satisfying the given conditions,
prove that the range of the abscissa of point P is [0, 3].

Conditions:
- A = (1, 0)
- F1 = (-2, 0)
- F2 = (2, 0)
- \| overrightarrow{PF1} \| + \| overrightarrow{PF2} \| = 6
- \| overrightarrow{PA} \| ≤ sqrt(6)
-/
theorem range_of_abscissa :
  ∀ (P : ℝ × ℝ),
    (|P.1 + 2| + |P.1 - 2| = 6) →
    ((P.1 - 1)^2 + P.2^2 ≤ 6) →
    (0 ≤ P.1 ∧ P.1 ≤ 3) :=
by
  intros P H1 H2
  sorry

end range_of_abscissa_l716_71630


namespace triangle_right_angle_l716_71620

variable {A B C a b c : ℝ}

theorem triangle_right_angle (h1 : Real.sin (A / 2) ^ 2 = (c - b) / (2 * c)) 
                             (h2 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) : 
                             a^2 + b^2 = c^2 :=
by
  sorry

end triangle_right_angle_l716_71620


namespace simplify_expression_l716_71609

theorem simplify_expression : (Real.sqrt 12 - |1 - Real.sqrt 3| + (7 + Real.pi)^0) = (Real.sqrt 3 + 2) :=
by
  sorry

end simplify_expression_l716_71609


namespace italian_dressing_mixture_l716_71606

/-- A chef is using a mixture of two brands of Italian dressing. 
  The first brand contains 8% vinegar, and the second brand contains 13% vinegar.
  The chef wants to make 320 milliliters of a dressing that is 11% vinegar.
  This statement proves the amounts required for each brand of dressing. -/

theorem italian_dressing_mixture
  (x y : ℝ)
  (hx : x + y = 320)
  (hv : 0.08 * x + 0.13 * y = 0.11 * 320) :
  x = 128 ∧ y = 192 :=
sorry

end italian_dressing_mixture_l716_71606


namespace fraction_sum_product_roots_of_quadratic_l716_71644

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l716_71644


namespace kevin_prizes_l716_71631

theorem kevin_prizes (total_prizes stuffed_animals yo_yos frisbees : ℕ)
  (h1 : total_prizes = 50) (h2 : stuffed_animals = 14) (h3 : yo_yos = 18) :
  frisbees = total_prizes - (stuffed_animals + yo_yos) → frisbees = 18 :=
by
  intro h4
  sorry

end kevin_prizes_l716_71631


namespace geometric_series_first_term_l716_71684

noncomputable def first_term_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) : Prop :=
  S = a / (1 - r)

theorem geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (hr : r = 1/6)
  (hS : S = 54) :
  first_term_geometric_series r S a →
  a = 45 :=
by
  intros h
  -- The proof goes here
  sorry

end geometric_series_first_term_l716_71684


namespace Ethan_uses_8_ounces_each_l716_71660

def Ethan (b: ℕ): Prop :=
  let number_of_candles := 10 - 3
  let total_coconut_oil := number_of_candles * 1
  let total_beeswax := 63 - total_coconut_oil
  let beeswax_per_candle := total_beeswax / number_of_candles
  beeswax_per_candle = b

theorem Ethan_uses_8_ounces_each (b: ℕ) (hb: Ethan b): b = 8 :=
  sorry

end Ethan_uses_8_ounces_each_l716_71660


namespace slope_of_line_passes_through_points_l716_71697

theorem slope_of_line_passes_through_points :
  let k := (2 + Real.sqrt 3 - 2) / (4 - 1)
  k = Real.sqrt 3 / 3 :=
by
  sorry

end slope_of_line_passes_through_points_l716_71697


namespace decimal_to_base7_conversion_l716_71675

theorem decimal_to_base7_conversion :
  (2023 : ℕ) = 5 * (7^3) + 6 * (7^2) + 2 * (7^1) + 0 * (7^0) :=
by
  sorry

end decimal_to_base7_conversion_l716_71675


namespace factorize1_factorize2_factorize3_factorize4_l716_71645

-- Statement for the first equation
theorem factorize1 (a x : ℝ) : 
  a * x^2 - 7 * a * x + 6 * a = a * (x - 6) * (x - 1) :=
sorry

-- Statement for the second equation
theorem factorize2 (x y : ℝ) : 
  x * y^2 - 9 * x = x * (y + 3) * (y - 3) :=
sorry

-- Statement for the third equation
theorem factorize3 (x y : ℝ) : 
  1 - x^2 + 2 * x * y - y^2 = (1 + x - y) * (1 - x + y) :=
sorry

-- Statement for the fourth equation
theorem factorize4 (x y : ℝ) : 
  8 * (x^2 - 2 * y^2) - x * (7 * x + y) + x * y = (x + 4 * y) * (x - 4 * y) :=
sorry

end factorize1_factorize2_factorize3_factorize4_l716_71645


namespace mary_friends_count_l716_71637

-- Definitions based on conditions
def total_stickers := 50
def stickers_left := 8
def total_students := 17
def classmates := total_students - 1 -- excluding Mary

-- Defining the proof problem
theorem mary_friends_count (F : ℕ) (h1 : 4 * F + 2 * (classmates - F) = total_stickers - stickers_left) :
  F = 5 :=
by sorry

end mary_friends_count_l716_71637


namespace find_multiple_l716_71634

variables (total_questions correct_answers score : ℕ)
variable (m : ℕ)
variable (incorrect_answers : ℕ := total_questions - correct_answers)

-- Given conditions
axiom total_questions_eq : total_questions = 100
axiom correct_answers_eq : correct_answers = 92
axiom score_eq : score = 76

-- Define the scoring method
def score_formula : ℕ := correct_answers - m * incorrect_answers

-- Statement to prove
theorem find_multiple : score = 76 → correct_answers = 92 → total_questions = 100 → score_formula total_questions correct_answers m = score → m = 2 := by
  intros h1 h2 h3 h4
  sorry

end find_multiple_l716_71634


namespace symmetric_point_yaxis_correct_l716_71613

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetric_yaxis (P : Point3D) : Point3D :=
  { x := -P.x, y := P.y, z := P.z }

theorem symmetric_point_yaxis_correct (P : Point3D) (P' : Point3D) :
  P = {x := 1, y := 2, z := -1} → 
  P' = symmetric_yaxis P → 
  P' = {x := -1, y := 2, z := -1} :=
by
  intros hP hP'
  rw [hP] at hP'
  simp [symmetric_yaxis] at hP'
  exact hP'

end symmetric_point_yaxis_correct_l716_71613


namespace initial_amount_correct_l716_71671

-- Definitions
def spent_on_fruits : ℝ := 15.00
def left_to_spend : ℝ := 85.00
def initial_amount_given (spent: ℝ) (left: ℝ) : ℝ := spent + left

-- Theorem stating the problem
theorem initial_amount_correct :
  initial_amount_given spent_on_fruits left_to_spend = 100.00 :=
by
  sorry

end initial_amount_correct_l716_71671


namespace sasha_remainder_20_l716_71663

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l716_71663


namespace math_problem_example_l716_71692

theorem math_problem_example (m n : ℤ) (h0 : m > 0) (h1 : n > 0)
    (h2 : 3 * m + 2 * n = 225) (h3 : Int.gcd m n = 15) : m + n = 105 :=
sorry

end math_problem_example_l716_71692


namespace necessary_but_not_sufficient_for_inequality_l716_71650

theorem necessary_but_not_sufficient_for_inequality : 
  ∀ x : ℝ, (-2 < x ∧ x < 4) → (x < 5) ∧ (¬(x < 5) → (-2 < x ∧ x < 4) ) :=
by 
  sorry

end necessary_but_not_sufficient_for_inequality_l716_71650


namespace sequences_identity_l716_71601

variables {α β γ : ℤ}
variables {a b : ℕ → ℤ}

-- Define the recurrence relations conditions
def conditions (a b : ℕ → ℤ) (α β γ : ℤ) : Prop :=
  a 0 = 1 ∧ b 0 = 1 ∧
  (∀ n, a (n + 1) = α * a n + β * b n) ∧
  (∀ n, b (n + 1) = β * a n + γ * b n) ∧
  α < γ ∧ α * γ = β^2 + 1

-- Define the main statement
theorem sequences_identity (a b : ℕ → ℤ) 
  (h : conditions a b α β γ) (m n : ℕ) :
  a (m + n) + b (m + n) = a m * a n + b m * b n :=
sorry

end sequences_identity_l716_71601


namespace find_fourth_student_in_sample_l716_71657

theorem find_fourth_student_in_sample :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 48 ∧ 
           (∀ (k : ℕ), k = 29 → 1 ≤ k ∧ k ≤ 48 ∧ ((k = 5 + 2 * 12) ∨ (k = 41 - 12)) ∧ n = 17) :=
sorry

end find_fourth_student_in_sample_l716_71657


namespace reciprocal_of_neg_three_l716_71643

theorem reciprocal_of_neg_three : ∃ x : ℚ, (-3) * x = 1 ∧ x = (-1) / 3 := sorry

end reciprocal_of_neg_three_l716_71643


namespace value_of_a_l716_71673

theorem value_of_a (a : ℝ) (h : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 → (a * x + 6 ≤ 10)) :
  a = 2 ∨ a = -4 ∨ a = 0 :=
sorry

end value_of_a_l716_71673


namespace sum_between_100_and_500_ending_in_3_l716_71617

-- Definition for the sum of all integers between 100 and 500 that end in 3
def sumOfIntegersBetween100And500EndingIn3 : ℕ :=
  let a := 103
  let d := 10
  let n := (493 - a) / d + 1
  (n * (a + 493)) / 2

-- Statement to prove that the sum is 11920
theorem sum_between_100_and_500_ending_in_3 : sumOfIntegersBetween100And500EndingIn3 = 11920 := by
  sorry

end sum_between_100_and_500_ending_in_3_l716_71617


namespace bisection_method_termination_condition_l716_71690

theorem bisection_method_termination_condition (x1 x2 e : ℝ) (h : e > 0) :
  |x1 - x2| < e → true :=
sorry

end bisection_method_termination_condition_l716_71690


namespace parallel_case_perpendicular_case_l716_71669

variables (m : ℝ)
def a := (2, -1)
def b := (-1, m)
def c := (-1, 2)
def sum_ab := (1, m - 1)

-- Parallel case (dot product is zero)
theorem parallel_case : (sum_ab m).fst * c.fst + (sum_ab m).snd * c.snd = 0 ↔ m = -1 :=
by
  sorry

-- Perpendicular case (dot product is zero)
theorem perpendicular_case : (sum_ab m).fst * c.fst + (sum_ab m).snd * c.snd = 0 ↔ m = 3 / 2 :=
by
  sorry

end parallel_case_perpendicular_case_l716_71669


namespace determine_k_and_solution_l716_71621

theorem determine_k_and_solution :
  ∃ (k : ℚ), (5 * k * x^2 + 30 * x + 10 = 0 → k = 9/2) ∧
    (∃ (x : ℚ), (5 * (9/2) * x^2 + 30 * x + 10 = 0) ∧ x = -2/3) := by
  sorry

end determine_k_and_solution_l716_71621


namespace range_of_a_l716_71636

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x
noncomputable def g (x a : ℝ) : ℝ := x + 1 / (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x1 : ℝ, x1 ∈ Set.Icc 0 2 → ∃ x2 : ℝ, x2 ∈ Set.Ioi a ∧ f x1 ≥ g x2 a) →
  a ≤ -1 :=
by
  intro h
  sorry

end range_of_a_l716_71636


namespace q_range_l716_71695

def q (x : ℝ) : ℝ := (x^2 - 2)^2

theorem q_range : 
  ∀ y : ℝ, y ∈ Set.range q ↔ 0 ≤ y :=
by sorry

end q_range_l716_71695


namespace isosceles_triangle_perimeter_l716_71647

theorem isosceles_triangle_perimeter 
  (a b c : ℝ)  (h_iso : a = b ∨ b = c ∨ c = a)
  (h_len1 : a = 4 ∨ b = 4 ∨ c = 4)
  (h_len2 : a = 9 ∨ b = 9 ∨ c = 9)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c = 22 :=
sorry

end isosceles_triangle_perimeter_l716_71647


namespace manfred_average_paycheck_l716_71679

def average_paycheck : ℕ → ℕ → ℕ → ℕ := fun total_paychecks first_paychecks_value num_first_paychecks =>
  let remaining_paychecks_value := first_paychecks_value + 20
  let total_payment := (num_first_paychecks * first_paychecks_value) + ((total_paychecks - num_first_paychecks) * remaining_paychecks_value)
  let average_payment := total_payment / total_paychecks
  average_payment

theorem manfred_average_paycheck :
  average_paycheck 26 750 6 = 765 := by
  sorry

end manfred_average_paycheck_l716_71679


namespace points_on_intersecting_lines_l716_71605

def clubsuit (a b : ℝ) := a^3 * b - a * b^3

theorem points_on_intersecting_lines (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = y ∨ x = -y) := 
by
  sorry

end points_on_intersecting_lines_l716_71605


namespace sum_minimum_values_l716_71623

def P (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
def Q (x : ℝ) (d e f : ℝ) : ℝ := x^3 + d * x^2 + e * x + f

theorem sum_minimum_values (a b c d e f : ℝ)
  (hPQ : ∀ x, P (Q x d e f) a b c = 0 → x = -4 ∨ x = -2 ∨ x = 0 ∨ x = 2 ∨ x = 4)
  (hQP : ∀ x, Q (P x a b c) d e f = 0 → x = -3 ∨ x = -1 ∨ x = 1 ∨ x = 3) :
  P 0 a b c + Q 0 d e f = -20 := sorry

end sum_minimum_values_l716_71623


namespace expression_evaluation_l716_71646

theorem expression_evaluation : 
  2000 * 1995 * 0.1995 - 10 = 0.2 * 1995^2 - 10 := 
by 
  sorry

end expression_evaluation_l716_71646


namespace max_sum_of_digits_l716_71698

theorem max_sum_of_digits (a b c : ℕ) (x : ℕ) (N : ℕ) :
  N = 100 * a + 10 * b + c →
  100 <= N →
  N < 1000 →
  a ≠ 0 →
  (100 * a + 10 * b + c) + (100 * a + 10 * c + b) = 1730 + x →
  a + b + c = 20 :=
by
  intros hN hN_ge_100 hN_lt_1000 ha_ne_0 hsum
  sorry

end max_sum_of_digits_l716_71698


namespace original_selling_price_is_1100_l716_71626

-- Let P be the original purchase price.
variable (P : ℝ)

-- Condition 1: Bill made a profit of 10% on the original purchase price.
def original_selling_price := 1.10 * P

-- Condition 2: If he had purchased that product for 10% less 
-- and sold it at a profit of 30%, he would have received $70 more.
def new_purchase_price := 0.90 * P
def new_selling_price := 1.17 * P
def price_difference := new_selling_price - original_selling_price

-- Theorem: The original selling price was $1100.
theorem original_selling_price_is_1100 (h : price_difference P = 70) : 
  original_selling_price P = 1100 :=
sorry

end original_selling_price_is_1100_l716_71626


namespace simplify_expression_l716_71653

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (yz + xz + xy) / (xyz * (x + y + z)) :=
by
  sorry

end simplify_expression_l716_71653


namespace A_scores_2_points_B_scores_at_least_2_points_l716_71674

-- Define the probabilities of outcomes.
def prob_A_win := 0.5
def prob_A_lose := 0.3
def prob_A_draw := 0.2

-- Calculate the probability of A scoring 2 points.
theorem A_scores_2_points : 
    (prob_A_win * prob_A_lose + prob_A_lose * prob_A_win + prob_A_draw * prob_A_draw) = 0.34 :=
by
  sorry

-- Calculate the probability of B scoring at least 2 points.
theorem B_scores_at_least_2_points : 
    (1 - (prob_A_win * prob_A_win + (prob_A_win * prob_A_draw + prob_A_draw * prob_A_win))) = 0.55 :=
by
  sorry

end A_scores_2_points_B_scores_at_least_2_points_l716_71674


namespace range_of_a_l716_71633

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x + a| + |x - 2| + a < 2010) ↔ a < 1006 :=
sorry

end range_of_a_l716_71633


namespace range_of_a_l716_71682

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + 3 < 0 ∧ 2^(1 - x) + a ≤ 0 ∧ x^2 - 2 * (a + 7) * x + 5 ≤ 0 ) ↔ (-4 ≤ a ∧ a ≤ -1) :=
by
  sorry

end range_of_a_l716_71682


namespace solve_system_l716_71649

noncomputable def sqrt_cond (x y : ℝ) : Prop :=
  Real.sqrt ((3 * x - 2 * y) / (2 * x)) + Real.sqrt ((2 * x) / (3 * x - 2 * y)) = 2

noncomputable def quad_cond (x y : ℝ) : Prop :=
  x^2 - 18 = 2 * y * (4 * y - 9)

theorem solve_system (x y : ℝ) : sqrt_cond x y ∧ quad_cond x y ↔ (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 1.5) :=
by
  sorry

end solve_system_l716_71649


namespace joan_kittens_remaining_l716_71614

def original_kittens : ℕ := 8
def kittens_given_away : ℕ := 2

theorem joan_kittens_remaining : original_kittens - kittens_given_away = 6 := by
  sorry

end joan_kittens_remaining_l716_71614


namespace middle_aged_selection_l716_71618

def total_teachers := 80 + 160 + 240
def sample_size := 60
def middle_aged_proportion := 160 / total_teachers
def middle_aged_sample := middle_aged_proportion * sample_size

theorem middle_aged_selection : middle_aged_sample = 20 :=
  sorry

end middle_aged_selection_l716_71618


namespace treehouse_total_planks_l716_71624

theorem treehouse_total_planks (T : ℕ) 
    (h1 : T / 4 + T / 2 + 20 + 30 = T) : T = 200 :=
sorry

end treehouse_total_planks_l716_71624


namespace flat_terrain_length_l716_71666

noncomputable def terrain_distance_equation (x y z : ℝ) : Prop :=
  (x + y + z = 11.5) ∧
  (x / 3 + y / 4 + z / 5 = 2.9) ∧
  (z / 3 + y / 4 + x / 5 = 3.1)

theorem flat_terrain_length (x y z : ℝ) 
  (h : terrain_distance_equation x y z) :
  y = 4 :=
sorry

end flat_terrain_length_l716_71666


namespace total_sum_of_rupees_l716_71668

theorem total_sum_of_rupees :
  ∃ (total_coins : ℕ) (paise20_coins : ℕ) (paise25_coins : ℕ),
    total_coins = 344 ∧ paise20_coins = 300 ∧ paise25_coins = total_coins - paise20_coins ∧
    (60 + (44 * 0.25)) = 71 :=
by
  sorry

end total_sum_of_rupees_l716_71668


namespace sum_div_by_24_l716_71681

theorem sum_div_by_24 (m n : ℕ) (h : ∃ k : ℤ, mn + 1 = 24 * k): (m + n) % 24 = 0 := 
by
  sorry

end sum_div_by_24_l716_71681


namespace sequence_monotonically_decreasing_l716_71654

theorem sequence_monotonically_decreasing (t : ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, a n = -↑n^2 + t * ↑n) →
  (∀ n : ℕ, a (n + 1) < a n) →
  t < 3 :=
by
  intros h1 h2
  sorry

end sequence_monotonically_decreasing_l716_71654


namespace total_legs_correct_l716_71686

def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goats : ℕ := 1
def legs_per_animal : ℕ := 4

theorem total_legs_correct :
  num_horses * legs_per_animal +
  num_dogs * legs_per_animal +
  num_cats * legs_per_animal +
  num_turtles * legs_per_animal +
  num_goats * legs_per_animal = 72 :=
by
  sorry

end total_legs_correct_l716_71686


namespace jen_ducks_l716_71625

theorem jen_ducks (c d : ℕ) (h1 : d = 4 * c + 10) (h2 : c + d = 185) : d = 150 := by
  sorry

end jen_ducks_l716_71625


namespace relationship_of_y_l716_71612

theorem relationship_of_y {k y1 y2 y3 : ℝ} (hk : k > 0) :
  (y1 = k / -1) → (y2 = k / 2) → (y3 = k / 3) → y1 < y3 ∧ y3 < y2 :=
by
  intros h1 h2 h3
  sorry

end relationship_of_y_l716_71612


namespace amount_allocated_to_food_l716_71667

theorem amount_allocated_to_food (total_amount : ℝ) (household_ratio food_ratio misc_ratio : ℝ) 
  (h₁ : total_amount = 1800) (h₂ : household_ratio = 5) (h₃ : food_ratio = 4) (h₄ : misc_ratio = 1) :
  food_ratio / (household_ratio + food_ratio + misc_ratio) * total_amount = 720 :=
by
  sorry

end amount_allocated_to_food_l716_71667


namespace polynomial_remainder_l716_71627

theorem polynomial_remainder :
  let f := X^2023 + 1
  let g := X^6 - X^4 + X^2 - 1
  ∃ (r : Polynomial ℤ), (r = -X^3 + 1) ∧ (∃ q : Polynomial ℤ, f = q * g + r) :=
by
  sorry

end polynomial_remainder_l716_71627


namespace hyperbola_eccentricity_l716_71602

-- Definitions of conditions
variables {a b c : ℝ}
variables (h : a > 0) (h' : b > 0)
variables (hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
variables (parab : ∀ y : ℝ, y^2 = 4 * b * y)
variables (ratio_cond : (b + c) / (c - b) = 5 / 3)

-- Proof statement
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 4 * Real.sqrt 15 / 15 :=
by
  have hyp_foci_distance : ∃ c : ℝ, c^2 = a^2 + b^2 := sorry
  have e := (4 * Real.sqrt 15) / 15
  use e
  sorry

end hyperbola_eccentricity_l716_71602


namespace exists_m_n_for_d_l716_71694

theorem exists_m_n_for_d (d : ℤ) : ∃ m n : ℤ, d = (n - 2 * m + 1) / (m^2 - n) := 
sorry

end exists_m_n_for_d_l716_71694


namespace find_value_divide_subtract_l716_71661

theorem find_value_divide_subtract :
  (Number = 8 * 156 + 2) → 
  (CorrectQuotient = Number / 5) → 
  (Value = CorrectQuotient - 3) → 
  Value = 247 :=
by
  intros h1 h2 h3
  sorry

end find_value_divide_subtract_l716_71661


namespace arithmetic_sequence_sum_l716_71656

variable {a_n : ℕ → ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∀ (m n k : ℕ), m < n → (n - m) = k → a_n n = a_n m + k * (a_n 1 - a_n 0)

theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a_n →
  a_n 2 = 5 →
  a_n 6 = 33 →
  a_n 3 + a_n 5 = 38 :=
by
  intros h_seq h_a2 h_a6
  sorry

end arithmetic_sequence_sum_l716_71656


namespace sqrt_S_n_arithmetic_seq_seq_sqrt_S_n_condition_l716_71629

-- (1)
theorem sqrt_S_n_arithmetic_seq (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d) (h3 : S n = (n * (2 * a 1 + (n - 1) * (2 : ℝ))) / 2) :
  ∃ d, ∀ n, Real.sqrt (S (n + 1)) = Real.sqrt (S n) + d :=
by sorry

-- (2)
theorem seq_sqrt_S_n_condition (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ) :
  (∃ d, ∀ n, S n / 2 = n * (a1 + (n - 1) * d)) ↔ (∀ n, S n = a1 * n^2) :=
by sorry

end sqrt_S_n_arithmetic_seq_seq_sqrt_S_n_condition_l716_71629


namespace value_of_b_plus_a_l716_71642

theorem value_of_b_plus_a (a b : ℝ) (h1 : |a| = 8) (h2 : |b| = 2) (h3 : |a - b| = |b - a|) : b + a = -6 ∨ b + a = -10 :=
by
  sorry

end value_of_b_plus_a_l716_71642


namespace find_multiplier_l716_71648

theorem find_multiplier (n m : ℕ) (h1 : 2 * n = (26 - n) + 19) (h2 : n = 15) : m = 2 :=
by
  sorry

end find_multiplier_l716_71648


namespace total_time_to_row_l716_71670

theorem total_time_to_row (boat_speed_in_still_water : ℝ) (stream_speed : ℝ) (distance : ℝ) :
  boat_speed_in_still_water = 9 → stream_speed = 1.5 → distance = 105 → 
  (distance / (boat_speed_in_still_water + stream_speed)) + (distance / (boat_speed_in_still_water - stream_speed)) = 24 :=
by
  intro h_boat_speed h_stream_speed h_distance
  rw [h_boat_speed, h_stream_speed, h_distance]
  sorry

end total_time_to_row_l716_71670


namespace average_of_roots_l716_71658

theorem average_of_roots (c : ℝ) (h : ∃ x1 x2 : ℝ, 2 * x1^2 - 6 * x1 + c = 0 ∧ 2 * x2^2 - 6 * x2 + c = 0 ∧ x1 ≠ x2) :
    (∃ p q : ℝ, (2 : ℝ) * (p : ℝ)^2 + (-6 : ℝ) * p + c = 0 ∧ (2 : ℝ) * (q : ℝ)^2 + (-6 : ℝ) * q + c = 0 ∧ p ≠ q) →
    (p + q) / 2 = 3 / 2 := 
sorry

end average_of_roots_l716_71658


namespace hyperbola_focal_coordinates_l716_71622

theorem hyperbola_focal_coordinates:
  ∀ (x y : ℝ), x^2 / 16 - y^2 / 9 = 1 → ∃ c : ℝ, c = 5 ∧ (x = -c ∨ x = c) ∧ y = 0 :=
by
  intro x y
  sorry

end hyperbola_focal_coordinates_l716_71622


namespace sqrt_three_pow_divisible_l716_71687

/-- For any non-negative integer n, (1 + sqrt 3)^(2*n + 1) is divisible by 2^(n + 1) -/
theorem sqrt_three_pow_divisible (n : ℕ) :
  ∃ k : ℕ, (⌊(1 + Real.sqrt 3)^(2 * n + 1)⌋ : ℝ) = k * 2^(n + 1) :=
sorry

end sqrt_three_pow_divisible_l716_71687


namespace identical_graphs_l716_71610

theorem identical_graphs :
  (∃ (b c : ℝ), (∀ (x y : ℝ), 3 * x + b * y + c = 0 ↔ c * x - 2 * y + 12 = 0) ∧
                 ((b, c) = (1, 6) ∨ (b, c) = (-1, -6))) → ∃ n : ℕ, n = 2 :=
by
  sorry

end identical_graphs_l716_71610


namespace smallest_divisor_of_7614_l716_71699

theorem smallest_divisor_of_7614 (h : Nat) (H_h_eq : h = 1) (n : Nat) (H_n_eq : n = (7600 + 10 * h + 4)) :
  ∃ d, d > 1 ∧ d ∣ n ∧ ∀ x, x > 1 ∧ x ∣ n → d ≤ x :=
by
  sorry

end smallest_divisor_of_7614_l716_71699


namespace find_ending_number_l716_71619

theorem find_ending_number (n : ℕ) 
  (h1 : n ≥ 7) 
  (h2 : ∀ m, 7 ≤ m ∧ m ≤ n → m % 7 = 0)
  (h3 : (7 + n) / 2 = 15) : n = 21 := 
sorry

end find_ending_number_l716_71619


namespace tetrahedron_labeling_count_l716_71683

def is_valid_tetrahedron_labeling (labeling : Fin 4 → ℕ) : Prop :=
  let f1 := labeling 0 + labeling 1 + labeling 2
  let f2 := labeling 0 + labeling 1 + labeling 3
  let f3 := labeling 0 + labeling 2 + labeling 3
  let f4 := labeling 1 + labeling 2 + labeling 3
  labeling 0 + labeling 1 + labeling 2 + labeling 3 = 10 ∧ 
  f1 = f2 ∧ f2 = f3 ∧ f3 = f4

theorem tetrahedron_labeling_count : 
  ∃ (n : ℕ), n = 3 ∧ (∃ (labelings: Finset (Fin 4 → ℕ)), 
  ∀ labeling ∈ labelings, is_valid_tetrahedron_labeling labeling) :=
sorry

end tetrahedron_labeling_count_l716_71683


namespace ratio_of_mixture_l716_71611

theorem ratio_of_mixture (x y : ℚ)
  (h1 : 0.6 = (4 * x + 7 * y) / (9 * x + 9 * y))
  (h2 : 50 = 9 * x + 9 * y) : x / y = 8 / 7 := 
sorry

end ratio_of_mixture_l716_71611


namespace num_boys_in_class_l716_71693

-- Definitions based on conditions
def num_positions (p1 p2 : Nat) (total : Nat) : Nat :=
  if h : p1 < p2 then p2 - p1
  else total - (p1 - p2)

theorem num_boys_in_class (p1 p2 : Nat) (total : Nat) :
  p1 = 6 ∧ p2 = 16 ∧ num_positions p1 p2 total = 10 → total = 22 :=
by
  intros h
  sorry

end num_boys_in_class_l716_71693


namespace Nancy_needs_5_loads_l716_71685

/-- Definition of the given problem conditions. -/
def pieces_of_clothing (shirts sweaters socks jeans : ℕ) : ℕ :=
  shirts + sweaters + socks + jeans

def washing_machine_capacity : ℕ := 12

def loads_required (total_clothing capacity : ℕ) : ℕ :=
  (total_clothing + capacity - 1) / capacity -- integer division with rounding up

/-- Theorem statement. -/
theorem Nancy_needs_5_loads :
  loads_required (pieces_of_clothing 19 8 15 10) washing_machine_capacity = 5 :=
by
  -- Insert proof here when needed.
  sorry

end Nancy_needs_5_loads_l716_71685


namespace third_root_of_polynomial_l716_71641

theorem third_root_of_polynomial (a b : ℚ) 
  (h₁ : a*(-1)^3 + (a + 3*b)*(-1)^2 + (2*b - 4*a)*(-1) + (10 - a) = 0)
  (h₂ : a*(4)^3 + (a + 3*b)*(4)^2 + (2*b - 4*a)*(4) + (10 - a) = 0) :
  ∃ (r : ℚ), r = -24 / 19 :=
by
  sorry

end third_root_of_polynomial_l716_71641


namespace reinforcement_size_l716_71677

theorem reinforcement_size (initial_men : ℕ) (initial_days : ℕ) (days_before_reinforcement : ℕ) (days_remaining : ℕ) (reinforcement : ℕ) : 
  initial_men = 150 → initial_days = 31 → days_before_reinforcement = 16 → days_remaining = 5 → (150 * 15) = (150 + reinforcement) * 5 → reinforcement = 300 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end reinforcement_size_l716_71677
