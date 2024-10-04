import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Automation
import Mathlib.Algebra.Order.Function
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Slope
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Probability.ProbSpace
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Function
import Mathlib.LinearAlgebra.AffineSpace.Basic
import Mathlib.Tactic
import data.real.basic

namespace correct_conclusions_l166_166004

variable (a b c m : ℝ)
variable (y1 y2 : ℝ)

-- Conditions: 
-- Parabola y = ax^2 + bx + c, intersects x-axis at (-3,0) and (1,0)
-- a < 0
-- Points P(m-2, y1) and Q(m, y2) are on the parabola, y1 < y2

def parabola_intersects_x_axis_at_A_B : Prop :=
  ∀ x : ℝ, x = -3 ∨ x = 1 → a * x^2 + b * x + c = 0

def concavity_and_roots : Prop :=
  a < 0 ∧ b = 2 * a ∧ c = -3 * a

def conclusion_1 : Prop :=
  a * b * c < 0

def conclusion_2 : Prop :=
  b^2 - 4 * a * c > 0

def conclusion_3 : Prop :=
  3 * b + 2 * c = 0

def conclusion_4 : Prop :=
  y1 < y2 → m ≤ -1

-- Correct conclusions given the parabola properties
theorem correct_conclusions :
  concavity_and_roots a b c →
  parabola_intersects_x_axis_at_A_B a b c →
  conclusion_1 a b c ∨ conclusion_2 a b c ∨ conclusion_3 a b c ∨ conclusion_4 a b c :=
sorry

end correct_conclusions_l166_166004


namespace no_integer_soln_x_y_l166_166336

theorem no_integer_soln_x_y (x y : ℤ) : x^2 + 5 ≠ y^3 := 
sorry

end no_integer_soln_x_y_l166_166336


namespace sum_of_first_15_odd_positives_l166_166801

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l166_166801


namespace percentage_men_speak_french_l166_166651

-- Definitions
variables (E : ℕ) -- Number of employees
def percentage_men : ℝ := 0.65
def percentage_speak_french : ℝ := 0.40
def percentage_women_donot_speak_french : ℝ := 0.9714285714285714

-- Theorem to prove
theorem percentage_men_speak_french (h_men : (percentage_men * E : ℝ) = 0.65 * E)
    (h_speak_french : (percentage_speak_french * E : ℝ) = 0.40 * E)
    (h_women_donot_speak_french: (percentage_women_donot_speak_french * (1 - percentage_men) * E : ℝ) = 0.9714285714285714 * 0.35 * E) :
    ((0.40 * E - (0.9714285714285714 * 0.35 * E)) / (0.65 * E) * 100) = 60 :=
by sorry

end percentage_men_speak_french_l166_166651


namespace initial_candies_in_bag_l166_166391

theorem initial_candies_in_bag (A_candies : ℕ) (total_candies_took : ℕ) : A_candies = 90 → total_candies_took = 260 :=
begin
  sorry
end

end initial_candies_in_bag_l166_166391


namespace product_of_numbers_l166_166775

open Real

theorem product_of_numbers (numbers : Finset ℝ) (h_card : numbers.card = 2015)
  (h_distinct : ∀ (x y : ℝ), x ∈ numbers → y ∈ numbers → x ≠ y)
  (h_parity : ∀ (a : ℝ), 0 < a → (numbers.filter (λ x, x < 2014 / a)).card % 2 = (numbers.filter (λ x, x > a)).card % 2)
  : ∏ x in numbers, x = 2014^1007 * sqrt 2014 := 
sorry

end product_of_numbers_l166_166775


namespace student_arrangement_l166_166026

theorem student_arrangement :
  let M1 := "M1"
  let M2 := "M2"
  let F1 := "F1"
  let F2 := "F2"
  let F3 := "F3"
  let students := [M1, M2, F1, F2, F3]
  ∃ arrangements, arrangements.count = 48 ∧
    (∀ (arrangement : List String), 
      arrangement ∈ arrangements →
      arrangement.length = students.length ∧
      ¬arrangement.head = M1 ∧
      ¬arrangement.last = M1 ∧
      (∃ i, i < arrangement.length - 1 ∧
      arrangement.get? i ∈ [F1, F2, F3] ∧
      arrangement.get? (i+1) ∈ [F1, F2, F3])) :=
begin
  sorry
end

end student_arrangement_l166_166026


namespace correct_transformation_l166_166047

theorem correct_transformation (x : ℝ) (h : 3 * x - 7 = 2 * x) : 3 * x - 2 * x = 7 :=
sorry

end correct_transformation_l166_166047


namespace angle_equality_l166_166889

-- Given: triangle ABC with points D, E, F as tangency points on sides or extensions.
-- DP is perpendicular to EF at point P.
-- To prove: ∠BPD = ∠CPD

variables (A B C D E F P : Point)
variables (O : Circle)
variables (hIncircle : IsIncircle O (Triangle ABC) (Tripoint D E F) ∨ IsExcircle O (Triangle ABC) (Tripoint D E F))
variables (hPerp : Perpendicular (Line EF) (Line DP) (Point P))

theorem angle_equality :
  ∠BPD = ∠CPD := sorry

end angle_equality_l166_166889


namespace num_math_students_l166_166544

noncomputable def centralParkCricketTeam : Nat := 15
noncomputable def physicsStudents : Nat := 10
noncomputable def bothSubjects : Nat := 4

theorem num_math_students (team_total : Nat) (phy_students : Nat) (both_subj : Nat) (H_total : team_total = 15) (H_phy : phy_students = 10) (H_both : both_subj = 4) :
  team_total - (phy_students - both_subj) = 9 := by
  /- definitions -/
  let only_physics := phy_students - both_subj
  have H_only_physics : only_physics = 6, by linarith [H_phy, H_both]
  let math_students := team_total - only_physics
  have H_math_students : math_students = 9, by linarith [H_total, H_only_physics]
  exact H_math_students

end num_math_students_l166_166544


namespace min_abs_sum_is_two_l166_166425

theorem min_abs_sum_is_two : ∃ x ∈ set.Ioo (- ∞) (∞), ∀ y ∈ set.Ioo (- ∞) (∞), (|y + 1| + |y + 3| + |y + 6| ≥ 2) ∧ (|x + 1| + |x + 3| + |x + 6| = 2) := sorry

end min_abs_sum_is_two_l166_166425


namespace ramsey_6_3_3_l166_166263

-- Define a type for people in the group
inductive Person : Type
| p1 | p2 | p3 | p4 | p5 | p6

-- Define the acquaintance relationship as a function from pairs of People to bool
variable (knows : Person → Person → Bool)

-- Mutual relationship condition
axiom mutual : ∀ (a b : Person), knows a b = knows b a

-- The proof that there exists a subset of 3 people who either all know each other or all do not know each other
theorem ramsey_6_3_3 (knows mutual) : 
  ∃ (a b c : Person), (knows a b ∧ knows b c ∧ knows c a) ∨ (¬ knows a b ∧ ¬ knows b c ∧ ¬ knows c a) :=
sorry

end ramsey_6_3_3_l166_166263


namespace triangle_area_ratio_l166_166027

theorem triangle_area_ratio (a b c a' b' c' r : ℝ)
    (h1 : a^2 + b^2 = c^2)
    (h2 : a'^2 + b'^2 = c'^2)
    (h3 : r = c' / 2)
    (S : ℝ := (1/2) * a * b)
    (S' : ℝ := (1/2) * a' * b') :
    S / S' ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end triangle_area_ratio_l166_166027


namespace sum_digits_18_to_21_l166_166768

-- Define the range of integers
def digits_sum_in_range (a b : ℕ) : ℕ :=
  (list.range' a (b - a + 1)).map (λ n, n.digits 10).sum

-- The sum of the digits of all integers from 18 to 21 inclusive
theorem sum_digits_18_to_21 : digits_sum_in_range 18 21 = 24 := by
  sorry

end sum_digits_18_to_21_l166_166768


namespace sandy_age_when_record_l166_166374

noncomputable def calc_age (record_length current_length monthly_growth_rate age : ℕ) : ℕ :=
  let yearly_growth_rate := monthly_growth_rate * 12
  let needed_length := record_length - current_length
  let years_needed := needed_length / yearly_growth_rate
  age + years_needed

theorem sandy_age_when_record (record_length current_length monthly_growth_rate age : ℕ) :
  record_length = 26 →
  current_length = 2 →
  monthly_growth_rate = 1 →
  age = 12 →
  calc_age record_length current_length monthly_growth_rate age = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  unfold calc_age
  simp
  sorry

end sandy_age_when_record_l166_166374


namespace sum_first_15_odd_integers_l166_166818

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l166_166818


namespace tiffany_lives_next_level_l166_166782

theorem tiffany_lives_next_level (L1 L2 L3 : ℝ)
    (h1 : L1 = 43.0)
    (h2 : L2 = 14.0)
    (h3 : L3 = 84.0) :
    L3 - (L1 + L2) = 27 :=
by
  rw [h1, h2, h3]
  -- The proof is skipped with "sorry"
  sorry

end tiffany_lives_next_level_l166_166782


namespace billy_final_lap_time_l166_166931

/-- Given that Billy and Margaret are competing to swim 10 laps,
    and under the provided conditions of lap times for Billy and
    the total time for Margaret, prove the time it took for Billy to
    swim his final lap. --/
theorem billy_final_lap_time :
  let billy_first_5_laps := 2 * 60,
      billy_next_3_laps := 4 * 60,
      billy_ninth_lap := 1 * 60,
      margaret_total_time := 10 * 60,
      billy_ahead_margin := 30,
      billy_total_time := margaret_total_time - billy_ahead_margin,
      billy_first_9_laps_time := billy_first_5_laps + billy_next_3_laps + billy_ninth_lap in
  billy_total_time - billy_first_9_laps_time = 150 :=
by
  sorry

end billy_final_lap_time_l166_166931


namespace problem_1_problem_2_l166_166943

-- Problem 1
theorem problem_1 (x : ℝ) : (2*x - 1) * (2*x - 3) - (1 - 2*x) * (2 - x) = 2*x^2 - 3*x + 1 :=
by {
  -- Proof omitted
  sorry,
}

-- Problem 2
theorem problem_2 (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) : (a^2 - 1) / a * (1 - (2*a + 1) / (a^2 + 2*a + 1)) / (a - 1) = a / (a + 1) :=
by {
  -- Proof omitted
  sorry,
}

end problem_1_problem_2_l166_166943


namespace compute_fraction_when_x_is_3_l166_166557

theorem compute_fraction_when_x_is_3 :
  (let x := 3 in (x^8 + 16 * x^4 + 64 + 4 * x^2) / (x^4 + 8) = 89 + 36 / 89) := 
by
  let x := 3
  have h1 : x^4 + 8 ≠ 0 := by sorry  -- we should check the denominator is indeed non-zero
  exact sorry

end compute_fraction_when_x_is_3_l166_166557


namespace min_abs_sum_l166_166417

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem min_abs_sum :
  ∃ (x : ℝ), (abs (x + 1) + abs (x + 3) + abs (x + 6)) = 7 :=
by {
  sorry
}

end min_abs_sum_l166_166417


namespace original_number_solution_l166_166445

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end original_number_solution_l166_166445


namespace A_lies_on_segment_BC_l166_166718

variables {Point : Type} [MetricSpace Point]

-- Given points A, B, and C
variables (A B C : Point)

-- Condition: For any point M, either MA ≤ MB or MA ≤ MC
axiom distance_condition (M : Point) : dist M A ≤ dist M B ∨ dist M A ≤ dist M C

-- Prove that point A lies on the segment BC
theorem A_lies_on_segment_BC : LiesOnSegment A B C :=
sorry

end A_lies_on_segment_BC_l166_166718


namespace probability_of_f_above_xaxis_l166_166618

-- Define the function f(x)
def f (x a b : ℝ) := x^2 + a*x + b^2

-- Conditions for a and b
def isValid_a (a : ℝ) := 0 ≤ a ∧ a ≤ 3
def isValid_b (b : ℝ) := 0 ≤ b ∧ b ≤ 2

-- Define the event area condition
def event_area_condition (a b : ℝ) := |a| < 2 * |b|

-- Define the sample space area
def sampleSpaceArea := 6

-- Define the event area
def eventArea := 6 - (1/2) * 3 * 1.5

-- Define the probability calculation
def probability := eventArea / sampleSpaceArea

theorem probability_of_f_above_xaxis : probability = 5 / 8 :=
by
  unfold sampleSpaceArea eventArea probability
  simp -- Simplification to show the calculation
  sorry

end probability_of_f_above_xaxis_l166_166618


namespace smallest_real_number_among_minus3_minus2_0_2_is_minus3_l166_166921

theorem smallest_real_number_among_minus3_minus2_0_2_is_minus3 :
  min (min (-3:ℝ) (-2)) (min 0 2) = -3 :=
by {
    sorry
}

end smallest_real_number_among_minus3_minus2_0_2_is_minus3_l166_166921


namespace crayons_left_l166_166385

theorem crayons_left (initial_crayons : ℕ) (crayons_taken : ℕ) : initial_crayons = 7 → crayons_taken = 3 → initial_crayons - crayons_taken = 4 :=
by
  sorry

end crayons_left_l166_166385


namespace div_relation_l166_166639

variables (a b c : ℚ)

theorem div_relation (h1 : a / b = 3) (h2 : b / c = 1 / 2) : c / a = 2 / 3 :=
by
  -- proof to be filled in
  sorry

end div_relation_l166_166639


namespace range_of_a_l166_166208

noncomputable def f (a x : ℝ) : ℝ := 2 * a * x - a + 3

theorem range_of_a (a : ℝ) (h : ∃ (x₀ : ℝ), -1 < x₀ ∧ x₀ < 1 ∧ f a x₀ = 0) : a ∈ set.Iic (-3) ∪ set.Ioi 1 :=
by
  cases h with x₀ hx₀
  sorry

end range_of_a_l166_166208


namespace sum_of_first_15_odd_integers_l166_166831

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l166_166831


namespace min_abs_sum_l166_166413

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem min_abs_sum :
  ∃ (x : ℝ), (abs (x + 1) + abs (x + 3) + abs (x + 6)) = 7 :=
by {
  sorry
}

end min_abs_sum_l166_166413


namespace number_of_correct_statements_l166_166532

section

-- Define even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Define statements
def statement1 (f : ℝ → ℝ) : Prop := ∃ x:ℝ, f x = f (-x) → f 0 = 0
def statement2 (f : ℝ → ℝ) : Prop := ∃ x:ℝ, f (-x) = -f x → f 0 = 0
def statement3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def statement4 (f : ℝ → ℝ) : Prop := ∀ (f : ℝ → ℝ), (is_even f ∧ is_odd f) → f = λ x, 0

-- Number of correct statements
theorem number_of_correct_statements : ∀ (f : ℝ → ℝ),
  (¬ statement1 f) ∧ (¬ statement2 f) ∧
  (statement3 f) ∧ (¬ statement4 f) → 
  1 :=
by
  intros f h
  cases h with s1 s2_s4
  cases s2_s4 with s2 s3_s4
  cases s3_s4 with s3 s4
  exact 1

end

end number_of_correct_statements_l166_166532


namespace largest_possible_median_l166_166790

theorem largest_possible_median (x : ℕ) (h : 0 < x) : 
  median {x, 2 * x, 3 * x, 4, 3, 2, 6} = 4 := 
sorry

end largest_possible_median_l166_166790


namespace min_value_expression_l166_166691

theorem min_value_expression 
  (a d : ℝ) (b c : ℝ)
  (h0 : 0 ≤ a)
  (h1 : 0 ≤ d)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : b + c ≥ a + d) :
  ∃ x : ℝ, x = sqrt 2 - 1 / 2 ∧ (b / (c + d) + c / (a + b) ≥ x) :=
sorry

end min_value_expression_l166_166691


namespace segment_HX_length_l166_166337
open Real

noncomputable def HX_length (AB : ℝ) (AX : ℝ) : ℝ := sqrt (81 + 36 * sqrt 2)

theorem segment_HX_length :
  ∀ (AB : ℝ) (AX : ℝ),
    AB = 3 →
    AX = 4 * AB →
    HX_length AB AX = 9 + 6 * sqrt 2 :=
begin
  intros AB AX hAB hAX,
  unfold HX_length,
  rw [hAB, hAX],
  sorry,
end

end segment_HX_length_l166_166337


namespace wendy_created_albums_l166_166787

theorem wendy_created_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) (total_pics : ℕ) (albums : ℕ) :
  phone_pics = 22 → camera_pics = 2 → pics_per_album = 6 → total_pics = phone_pics + camera_pics → albums = total_pics / pics_per_album → albums = 4 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end wendy_created_albums_l166_166787


namespace proof_equivalent_l166_166177

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

def x := log_base 0.1 7
def y := Real.log (Real.sqrt 7)

def p := x + y < x * y
def q := x + y > 0

theorem proof_equivalent : p ∧ ¬ q := by
  sorry

end proof_equivalent_l166_166177


namespace charlie_coins_worth_44_cents_l166_166947

-- Definitions based on the given conditions
def total_coins := 17
def p_eq_n_plus_2 (p n : ℕ) := p = n + 2

-- The main theorem stating the problem and the expected answer
theorem charlie_coins_worth_44_cents (p n : ℕ) (h1 : p + n = total_coins) (h2 : p_eq_n_plus_2 p n) :
  (7 * 5 + p * 1 = 44) :=
sorry

end charlie_coins_worth_44_cents_l166_166947


namespace parabola_conditions_l166_166002

theorem parabola_conditions 
  (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b = 2 * a) 
  (hc : c = -3 * a) 
  (hA : a * (-3)^2 + b * (-3) + c = 0) 
  (hB : a * (1)^2 + b * (1) + c = 0) : 
  (b^2 - 4 * a * c > 0) ∧ (3 * b + 2 * c = 0) :=
sorry

end parabola_conditions_l166_166002


namespace calc_expr_l166_166109

noncomputable def expr : ℝ :=
  (Real.sqrt 4) + (Real.cbrt (-125)) - (Real.sqrt ((-3)^2)) + (Real.abs (Real.sqrt 3 - 1))

theorem calc_expr : expr = Real.sqrt 3 - 7 :=
by
  sorry

end calc_expr_l166_166109


namespace sum_value_l166_166998

theorem sum_value : 
    (∑ a in set.Ici 1, ∑ b in set.Ici 1, ∑ c in set.Ici 1, (a * b * (3 * a + c))/(4 ^ (a + b + c) * (a + b) * (b + c) * (c + a)))
    = 1 / 54 := 
    by 
    sorry

end sum_value_l166_166998


namespace tangent_line_through_origin_l166_166371

noncomputable def curve (x : ℝ) : ℝ := Real.exp (x - 1) + x

theorem tangent_line_through_origin :
  ∃ k : ℝ, k = 2 ∧ ∀ x y : ℝ, (y = k * x) ↔ (∃ m : ℝ, curve m = m + Real.exp (m - 1) ∧ (curve m) = (m + Real.exp (m - 1)) ∧ k = (Real.exp (m - 1) + 1) ∧ y = k * x ∧ y = 2*x) :=
by 
  sorry

end tangent_line_through_origin_l166_166371


namespace cricket_students_l166_166260

theorem cricket_students (B C BC U : ℕ) (h1 : B = 9) (h2 : BC = 6) (h3 : U = 11) :
  C = U - B + BC :=
by
  -- Assigning the values as per the conditions
  have hB : B = 9 := h1
  have hBC : BC = 6 := h2
  have hU : U = 11 := h3

  -- Simplifying using given values
  calc
    C = U - B + BC : by sorry
      ... = 11 - 9 + 6 : by rw [hB, hBC, hU]
      ... = 8 : by sorry

end cricket_students_l166_166260


namespace circles_intersect_concurrence_l166_166630

open EuclideanGeometry

noncomputable def midpoint (A B : Point) : Point := sorry

theorem circles_intersect_concurrence {O1 O2 A B O C D P E F Q : Point} 
  (h1 : Circle O1)
  (h2 : Circle O2)
  (h3 : intersects O1 O2 A)
  (h4 : intersects O1 O2 B)
  (h5 : midpoint A B = O)
  (h6 : chord O1 C D)
  (h7 : chord O2 E F)
  (h8 : C ≠ D)
  (h9 : E ≠ F)
  (h10 : Segment C D = Segment P Q)
  (h11 : P ∈ Circle O2)
  (h12 : Q ∈ Circle O1)
  : concurrent (Line A B) (Line C Q) (Line E P) := sorry

end circles_intersect_concurrence_l166_166630


namespace terminating_decimals_nat_l166_166146

theorem terminating_decimals_nat (n : ℕ) (h1 : ∃ a b : ℕ, n = 2^a * 5^b)
  (h2 : ∃ c d : ℕ, n + 1 = 2^c * 5^d) : n = 1 ∨ n = 4 :=
by
  sorry

end terminating_decimals_nat_l166_166146


namespace problem_solution_l166_166900

noncomputable def ellipse_radius_interval_sum : ℝ :=
  let a := 4 in
  let b := 1 in
  let c := Real.sqrt (a^2 - b^2) in
  c + 8

theorem problem_solution : ellipse_radius_interval_sum = Real.sqrt 15 + 8 :=
  sorry

end problem_solution_l166_166900


namespace min_abs_sum_is_5_l166_166436

noncomputable def min_abs_sum (x : ℝ) : ℝ :=
  |x + 1| + |x + 3| + |x + 6|

theorem min_abs_sum_is_5 : ∃ x : ℝ, (∀ y : ℝ, min_abs_sum y ≥ min_abs_sum x) ∧ min_abs_sum x = 5 :=
by
  use -3
  sorry

end min_abs_sum_is_5_l166_166436


namespace original_number_value_l166_166454

noncomputable def orig_number_condition (x : ℝ) : Prop :=
  1000 * x = 3 * (1 / x)

theorem original_number_value : ∃ x : ℝ, 0 < x ∧ orig_number_condition x ∧ x = √(30) / 100 :=
begin
  -- the proof
  sorry
end

end original_number_value_l166_166454


namespace population_percentage_l166_166654

theorem population_percentage (p1 p2 : ℕ) (h1 : p1 = 20) (h2 : p2 = 30) : p1 + p2 = 50 := by
  rw [h1, h2]
  exact rfl

end population_percentage_l166_166654


namespace jasmine_iteration_reaches_one_l166_166737

def iterative_div_floor (n : ℕ) : ℕ :=
  n / 3

theorem jasmine_iteration_reaches_one : ∃ n : ℕ, n = 150 ∧
  let n1 := iterative_div_floor n in
  let n2 := iterative_div_floor n1 in
  let n3 := iterative_div_floor n2 in
  let n4 := iterative_div_floor n3 in
  n4 = 1 :=
by
  sorry

end jasmine_iteration_reaches_one_l166_166737


namespace coefficient_term_x3_l166_166275

theorem coefficient_term_x3 :
  ∀ (x : ℝ),
  (term_coeff : ∃ c : ℝ, (5 * x^2 + 8 / x)^9 = c * x^3 + ...) →
  c = (nat.choose 9 5) * 5^4 * 8^5 :=
by
  sorry

end coefficient_term_x3_l166_166275


namespace coat_price_reduction_l166_166011

theorem coat_price_reduction 
  (original_price : ℝ) 
  (reduction_percentage : ℝ)
  (h_original_price : original_price = 500)
  (h_reduction_percentage : reduction_percentage = 60) :
  original_price * (reduction_percentage / 100) = 300 :=
by 
  sorry

end coat_price_reduction_l166_166011


namespace matrix_inverse_l166_166988

-- Define the given matrix
def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5, 4], ![-2, 8]]

-- Define the expected inverse matrix
def A_inv_expected : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1/6, -1/12], ![1/24, 5/48]]

-- The main statement: Prove that the inverse of A is equal to the expected inverse
theorem matrix_inverse :
  A⁻¹ = A_inv_expected := sorry

end matrix_inverse_l166_166988


namespace initial_geese_ratio_l166_166101

theorem initial_geese_ratio (G : ℕ) : 
  (∃ G, G - 10 = 29 + 1 ∧ G % 25 = 0 ∧ (40 / 8 = 25 / 5) := by
{
  sorry
}

end initial_geese_ratio_l166_166101


namespace correct_propositions_l166_166595

-- Definitions of perpendicular and parallel lines and planes
variables {α β : Type*} [plane α] [plane β] [line l] [line m]
variables (l_perp_α : l ⊥ α) (m_in_β : m ∈ β)

-- Theorem statement
theorem correct_propositions : 
  (l ⊥ α ∧ m ∈ β) →
  (if α ∥ β then l ⊥ m else true ∧ -- Proposition ①
  if α ⊥ β then l ∥ m else true ∧ -- Proposition ②
  if l ∥ m then α ⊥ β else true ∧ -- Proposition ③
  if l ⊥ m then α ∥ β else true) = (l ⊥ α ∧ m ∈ β) → (α ∥ β → l ⊥ m) ∧ (l ∥ m → α ⊥ β) := 
sorry

end correct_propositions_l166_166595


namespace sum_of_first_three_terms_is_zero_l166_166751

variable (a d : ℤ) 

-- Definitions from the conditions
def a₄ := a + 3 * d
def a₅ := a + 4 * d
def a₆ := a + 5 * d

-- Theorem statement
theorem sum_of_first_three_terms_is_zero 
  (h₁ : a₄ = 8) 
  (h₂ : a₅ = 12) 
  (h₃ : a₆ = 16) : 
  a + (a + d) + (a + 2 * d) = 0 := 
by 
  sorry

end sum_of_first_three_terms_is_zero_l166_166751


namespace school_purchase_cost_l166_166078

def pencil_cost : ℝ := 2.50
def pen_cost : ℝ := 3.50
def pencil_discount_threshold : ℝ := 30.0
def pencil_discount_rate : ℝ := 0.10
def pen_discount_threshold : ℝ := 50.0
def pen_discount_rate : ℝ := 0.15
def additional_discount_threshold : ℝ := 250.0
def additional_discount_rate : ℝ := 0.05

def total_cost (pencils : ℝ) (pens : ℝ) : ℝ := 
  let initial_pencil_cost := pencils * pencil_cost
  let initial_pen_cost := pens * pen_cost
  let total_initial_cost := initial_pencil_cost + initial_pen_cost
  let pencil_discount := if pencils > pencil_discount_threshold then initial_pencil_cost * pencil_discount_rate else 0
  let pen_discount := if pens > pen_discount_threshold then initial_pen_cost * pen_discount_rate else 0
  let discounted_total := total_initial_cost - pencil_discount - pen_discount
  let additional_discount := if discounted_total > additional_discount_threshold then discounted_total * additional_discount_rate else 0
  discounted_total - additional_discount

theorem school_purchase_cost
  (pencils : ℝ)
  (pens : ℝ)
  (h1 : pencils = 38)
  (h2 : pens = 56) :
  total_cost pencils pens = 239.50 := by
  sorry

end school_purchase_cost_l166_166078


namespace find_point_B_l166_166223

theorem find_point_B 
  (A : ℝ × ℝ)
  (C : ℝ × ℝ → Prop)
  (intersects : ∃ B : ℝ × ℝ, C B)
  (H1 : A = (-7, 9))
  (H2 : ∀ x y, C (x, y) → y = 1 - x)
  (H3 : ∃ B, C B ∧ (B = (-8, 8))) : 
  ∃ B, B = (-8, 8) :=
by 
  obtain ⟨B, CB⟩ := intersects
  use B
  rw H3
  exact CB

end find_point_B_l166_166223


namespace radius_of_circle_C1_l166_166120

-- Definitions of the points and distances given in the conditions
variables {O X Y Z : Type*} [metric_space O] [metric_space X] [metric_space Y] [metric_space Z]

-- Distance from X to Z
def dist_XZ : ℝ := 17

-- Distance from O to Z
def dist_OZ : ℝ := 15

-- Distance from Y to Z
def dist_YZ : ℝ := 9

-- Assertion of the required result
theorem radius_of_circle_C1 (r : ℝ) :
  r = 8 * real.sqrt 13 :=
sorry

end radius_of_circle_C1_l166_166120


namespace sequence_count_length_21_l166_166235

def g : ℕ → ℕ
| 3     := 1
| 4     := 1
| 5     := 1
| (n+6) := g (n+2) + g (n+1) + g n
| _     := 0  -- default base case for non-positive inputs

theorem sequence_count_length_21 : g 21 = 86 := 
by {
  sorry
}

end sequence_count_length_21_l166_166235


namespace integral_solution_l166_166939

noncomputable def integral_value (n : ℕ) : ℝ :=
  ∫ x in 0..(Real.pi / 2^(n+1)), 
    (sin x) * (cos x) * (cos (2 * x)) * (cos (2^2 * x)) * ... * (cos (2^(n-1) * x))

theorem integral_solution (n : ℕ) : 
  integral_value n = 1 / (2^(2 * n)) :=
by
  sorry

end integral_solution_l166_166939


namespace sum_first_15_odd_integers_l166_166845

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l166_166845


namespace simple_interest_calculation_l166_166082

def simple_interest (P R T : ℝ) : ℝ := P * (R / 100) * T

theorem simple_interest_calculation :
  simple_interest 6693.75 12 5 = 4016.25 := by
  sorry

end simple_interest_calculation_l166_166082


namespace sister_age_is_one_l166_166345

variable (B S : ℕ)

theorem sister_age_is_one (h : B = B * S) : S = 1 :=
by {
  sorry
}

end sister_age_is_one_l166_166345


namespace peter_flight_distance_l166_166329

theorem peter_flight_distance :
  ∀ (distance_spain_to_russia distance_spain_to_germany : ℕ),
  distance_spain_to_russia = 7019 →
  distance_spain_to_germany = 1615 →
  (distance_spain_to_russia - distance_spain_to_germany) + 2 * distance_spain_to_germany = 8634 :=
by
  intros distance_spain_to_russia distance_spain_to_germany h1 h2
  rw [h1, h2]
  sorry

end peter_flight_distance_l166_166329


namespace ratio_BC_CD_l166_166540

theorem ratio_BC_CD (A B C D E : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (angle_bad : Real.Acos ((BA • DA) / (|BA| * |DA|)) = real.pi / 3)
  (length_ab : BA.norm = 4)
  (length_ad : DA.norm = 5)
  (angle_b : ∠ B = real.pi / 2)
  (angle_d : ∠ D = real.pi / 2) :
  BC.length / CD.length = 2 := 
  sorry

end ratio_BC_CD_l166_166540


namespace sin_angle_HAC_is_sqrt_143_over_13_l166_166886

def prism_coords (a b c : ℝ) : Prop :=
  A = (0, 0, 0)
  ∧ B = (a, 0, 0)
  ∧ C = (a, b, 0)
  ∧ D = (0, b, 0)
  ∧ E = (0, 0, c)
  ∧ F = (a, 0, c)
  ∧ G = (a, b, c)
  ∧ H = (0, b, c)

def sin_HAC (a b c : ℝ) : ℝ :=
  let HA := (0, -b, -c)
      AC := (a, b, 0) in
  let dot_product := (0 * a) + (-b) * b + (-c) * 0
      magnitude_HA := real.sqrt (0^2 + (-b)^2 + (-c)^2)
      magnitude_AC := real.sqrt (a^2 + b^2 + 0^2) in
  let cos_theta := dot_product / (magnitude_HA * magnitude_AC) in
  real.sqrt (1 - cos_theta^2)

theorem sin_angle_HAC_is_sqrt_143_over_13 : prism_coords 2 2 3 → sin_HAC 2 2 3 = sqrt 143 / 13 :=
by
  assume h
  sorry

end sin_angle_HAC_is_sqrt_143_over_13_l166_166886


namespace sequence_int_values_for_infinite_indices_l166_166016

theorem sequence_int_values_for_infinite_indices :
  ∃ (N : ℕ) (n : ℕ → ℕ), ∀ k, a (n k) ∈ ℤ ∧ a (n k) = 1 + (2 ^ (n k + 1) - 1) / 9 :=
begin
  let a : ℕ → ℚ :=
  λ n, if n = 0 then 1
        else if n = 1 then 10 / 9
        else a (n - 2) * 3 - a (n - 1) * 2,
  let n : ℕ → ℕ := λ k, 6 * k - 1,
  use [N, n],
  sorry
end

end sequence_int_values_for_infinite_indices_l166_166016


namespace sum_of_first_15_odd_integers_l166_166830

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l166_166830


namespace exists_composite_value_l166_166689

-- Definition of polynomial P(x) with the given conditions
def polynomial (n : Nat) (coeffs : Fin (n + 1) → Int) (x : Nat) : Int :=
  ∑ i in Finset.range (n + 1), coeffs ⟨i, Nat.lt_succ_self i⟩ * (x : Int) ^ i

def is_composite (n : Nat) : Prop := ¬Prime n ∧ n > 1

theorem exists_composite_value (n : Nat) (coeffs : Fin (n + 1) → Int)
    (h1 : coeffs ⟨n, Nat.lt_succ_self n⟩ > 0) (h2 : n ≥ 2) :
    ∃ m : Nat, m > 0 ∧ is_composite (polynomial n coeffs (Nat.factorial m)) := by
  sorry

end exists_composite_value_l166_166689


namespace determine_triangle_ratio_l166_166971

theorem determine_triangle_ratio (a d : ℝ) (h : (a + d) ^ 2 = (a - d) ^ 2 + a ^ 2) : a / d = 2 + Real.sqrt 3 :=
sorry

end determine_triangle_ratio_l166_166971


namespace trajectory_parabola_minimum_distance_slope_l166_166601

-- Definitions based on conditions
def circle_F (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def parabola_W (x y : ℝ) : Prop := y^2 = 4 * x

theorem trajectory_parabola (x y : ℝ) (hx : 0 ≤ x) :
  (x, y) = Q → (moving_point : ℝ × ℝ) → (y² = 4 * x) :=
by sorry

theorem minimum_distance_slope (y₁ y₂ : ℝ) :
  let A := (y₁^2 / 4, y₁)
  let D := (y₂^2 / 4, y₂)
  let FA := (y₁^2 / 4 - 1, y₁)
  ∃ line_l_slope : ℝ, -- the slope of the line l
  |AB| + 4 * |CD| = 4 ∧ (line_l_slope = 2 * sqrt 2 ∨ line_l_slope = -2 * sqrt 2) :=
by sorry

end trajectory_parabola_minimum_distance_slope_l166_166601


namespace find_a_10_l166_166272

/-- 
a_n is an arithmetic sequence
-/
def a (n : ℕ) : ℝ := sorry

/-- 
Given conditions:
- Condition 1: a_2 + a_5 = 19
- Condition 2: S_5 = 40, where S_5 is the sum of the first five terms
-/
axiom condition1 : a 2 + a 5 = 19
axiom condition2 : (a 1 + a 2 + a 3 + a 4 + a 5) = 40

noncomputable def a_10 : ℝ := a 10

theorem find_a_10 : a_10 = 29 :=
by
  sorry

end find_a_10_l166_166272


namespace value_of_x_in_sequence_l166_166281

theorem value_of_x_in_sequence :
  ∃ x: ℕ, (∀ (n: ℕ), (seq n = seq (n-1) + n) ∧ (seq 0 = 1) ∧ (seq 1 = 3) ∧ (seq 2 = 6) ∧ (seq 3 = 10) ∧ (seq 5 = 21) ∧ (seq 6 = 28) → seq 4 = 15) :=
by
  sorry

end value_of_x_in_sequence_l166_166281


namespace point_A_on_segment_BC_l166_166715

theorem point_A_on_segment_BC (A B C : Point) : 
  (∀ M : Point, dist M A ≤ dist M B ∨ dist M A ≤ dist M C) → on_segment A B C :=
begin
  intro h,
  sorry
end

end point_A_on_segment_BC_l166_166715


namespace number_of_integers_with_D3_l166_166125

def D (n : ℕ) : ℕ :=
  -- Definition of D would need to be provided here
  sorry

theorem number_of_integers_with_D3 :
  ∃ (count : ℕ), count = 11 ∧
    count = (nat.count (λ n, D n = 3) (finset.filter (λ n, n ≤ 50) (finset.range 51))) :=
sorry

end number_of_integers_with_D3_l166_166125


namespace sum_of_first_15_odd_integers_l166_166854

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l166_166854


namespace molecular_weight_of_Y_l166_166535

def molecular_weight_X : ℝ := 136
def molecular_weight_C6H8O7 : ℝ := 192
def moles_C6H8O7 : ℝ := 5

def total_mass_reactants := molecular_weight_X + moles_C6H8O7 * molecular_weight_C6H8O7

theorem molecular_weight_of_Y :
  total_mass_reactants = 1096 := by
  sorry

end molecular_weight_of_Y_l166_166535


namespace pipe_fill_rate_l166_166084

theorem pipe_fill_rate 
  (C : ℝ) (t : ℝ) (capacity : C = 4000) (time_to_fill : t = 300) :
  (3/4 * C / t) = 10 := 
by 
  sorry

end pipe_fill_rate_l166_166084


namespace rectangle_side_length_l166_166956

theorem rectangle_side_length (x : ℝ) (h1 : 0 < x) (h2 : 2 * (x + 6) = 40) : x = 14 :=
by
  sorry

end rectangle_side_length_l166_166956


namespace min_value_of_f_l166_166410

def f (x : ℝ) := abs (x + 1) + abs (x + 3) + abs (x + 6)

theorem min_value_of_f : ∃ (x : ℝ), f x = 5 :=
by
  use -3
  simp [f]
  sorry

end min_value_of_f_l166_166410


namespace range_of_m_l166_166626

-- Definitions for the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B (m : ℝ) : Set ℝ := {x | x ≥ m}

-- Prove that m ≥ 2 given the condition A ∪ B = A 
theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : m ≥ 2 :=
by
  sorry

end range_of_m_l166_166626


namespace sandy_age_when_record_l166_166375

noncomputable def calc_age (record_length current_length monthly_growth_rate age : ℕ) : ℕ :=
  let yearly_growth_rate := monthly_growth_rate * 12
  let needed_length := record_length - current_length
  let years_needed := needed_length / yearly_growth_rate
  age + years_needed

theorem sandy_age_when_record (record_length current_length monthly_growth_rate age : ℕ) :
  record_length = 26 →
  current_length = 2 →
  monthly_growth_rate = 1 →
  age = 12 →
  calc_age record_length current_length monthly_growth_rate age = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  unfold calc_age
  simp
  sorry

end sandy_age_when_record_l166_166375


namespace probability_of_nine_or_more_stayed_l166_166342

def probability_at_least_nine_stayed_whole_time :
  (5 : ℕ) → (5 : ℕ) → (1 / 3 : ℝ) → ((10 : ℕ) → ℝ) := by
  sorry


theorem probability_of_nine_or_more_stayed :
  (hCertain : 5)
  (hUncertain : 5)
  (pUncertainStay : 1 / 3) :
  probability_at_least_nine_stayed_whole_time hCertain hUncertain pUncertainStay 10 = 11 / 243 := by
  sorry

end probability_of_nine_or_more_stayed_l166_166342


namespace area_of_triangle_ABC_l166_166258

theorem area_of_triangle_ABC :
  ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C], 
  (60°:ℝ = π/3) ∧ (30°:ℝ = π/6) ∧ (90°:ℝ = π/2) ∧
  ∀ {A B C : Type} (hABC : 30-60-90_triangle A B C) (hC: ∠C = 90°) (h_AB: dist A B = 10),
  (∃ (X Y Z W : Type), are_on_circle {X, Y, Z, W} ∧ 
  (is_square A B X Y ∧ is_square A C W Z))
→ 
  ∃ (Area : ℝ), Area = (25 * sqrt 3 / 2) :=
begin
  sorry -- The actual proof will be filled in here.
end

end area_of_triangle_ABC_l166_166258


namespace factorial_zeros_30_l166_166547

theorem factorial_zeros_30 : 
  let n := 30! 
  in (count_factors_of_10 n = 7) :=
by
  -- Define a function to count the factors of 10 in a number
  def count_factors_of_10 (n : ℕ) : ℕ :=
    let count_factors (p : ℕ) : ℕ := (n.digits p).count 0
    min (count_factors 2) (count_factors 5)
  
  have h30 : ∀ n : ℕ, 30 <= n := by sorry
  
  -- Applying the appropriate prime factor counting
  show count_factors_of_10 30! = 7, sorry

end factorial_zeros_30_l166_166547


namespace find_lambda_l166_166188

variable {Point : Type*}
variable (A B C D O : Point)
variable [AddCommGroup Point] [Module ℝ Point]

-- Hypotheses corresponding to the conditions
variable (coplanar : ∃ (u v : ℝ), ∀ (P : Point), P = A + u • (B - A) + v • (C - A))
variable (non_collinear : ¬ (∃ (u v : ℝ), ∀ (P : Point), P = A + u • (B - A) ∨ P = B + v • (C - B)))
variable (outside_plane : ¬ ∃ (a b c : ℝ), O = a • A + b • B + c • C)

-- Definitions of vectors
variable [VectorSpace ℝ Point]
variable (OA OB OC OD : Point) (λ : ℝ)

-- The vector equation
def relation := OD = 3 • OA + 2 • OB + λ • OC

-- The proof problem statement
theorem find_lambda : relation OA OB OC OD λ → λ = -4 :=
by
  intros h
  sorry

end find_lambda_l166_166188


namespace brenda_distance_when_first_met_l166_166546

theorem brenda_distance_when_first_met
  (opposite_points : ∀ (d : ℕ), d = 150) -- Starting at diametrically opposite points on a 300m track means distance is 150m
  (constant_speeds : ∀ (B S x : ℕ), B * x = S * x) -- Brenda/ Sally run at constant speed
  (meet_again : ∀ (d₁ d₂ : ℕ), d₁ + d₂ = 300 + 100) -- Together they run 400 meters when they meet again, additional 100m by Sally
  : ∃ (x : ℕ), x = 150 :=
  by
    sorry

end brenda_distance_when_first_met_l166_166546


namespace volume_of_sphere_l166_166201

theorem volume_of_sphere (α : Type _) (O : Type _) [MetricSpace O] [MetricSpace α]
    (intersection_area : ∀ (sph : MetricSphere O) (plane : α), CrossSectionArea intersect_plane sph π8)
    (dist_center_plane : (dist (center sph) α) = 1) :
    ∃ (sph : MetricSphere O), volume sph = 36 * π := 
sorry

end volume_of_sphere_l166_166201


namespace mirror_area_correct_l166_166320

-- Given conditions
def outer_length : ℕ := 80
def outer_width : ℕ := 60
def frame_width : ℕ := 10

-- Deriving the dimensions of the mirror
def mirror_length : ℕ := outer_length - 2 * frame_width
def mirror_width : ℕ := outer_width - 2 * frame_width

-- Statement: Prove that the area of the mirror is 2400 cm^2
theorem mirror_area_correct : mirror_length * mirror_width = 2400 := by
  -- Proof should go here
  sorry

end mirror_area_correct_l166_166320


namespace nature_of_f_l166_166010

noncomputable def f (x : ℝ) : ℝ := x ^ (real.log 2 / real.log 4)

theorem nature_of_f : 
  f (4) = 2 ∧ (
  ¬ ∀ x, f (x) = f (-x)) ∧ (
  ¬ ∀ x, f (x) = -f (-x)) ∧ (
  ∀ x, 0 < x → f (x) ≤ f (x+1)) :=
by
  sorry

end nature_of_f_l166_166010


namespace ray_climbing_stairs_l166_166483

theorem ray_climbing_stairs (n : ℕ) (h1 : n % 4 = 3) (h2 : n % 5 = 2) (h3 : 10 < n) : n = 27 :=
sorry

end ray_climbing_stairs_l166_166483


namespace product_of_two_numbers_l166_166370

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 + y^2 = 120) :
  x * y = -20 :=
sorry

end product_of_two_numbers_l166_166370


namespace tan_150_degree_is_correct_l166_166122

noncomputable def tan_150_degree_is_negative_sqrt_3_div_3 : Prop :=
  let theta := Real.pi * 150 / 180
  let ref_angle := Real.pi * 30 / 180
  let cos_150 := -Real.cos ref_angle
  let sin_150 := Real.sin ref_angle
  Real.tan theta = -Real.sqrt 3 / 3

theorem tan_150_degree_is_correct :
  tan_150_degree_is_negative_sqrt_3_div_3 :=
by
  sorry

end tan_150_degree_is_correct_l166_166122


namespace tan_diff_eq_sqrt_three_l166_166172

open Real

theorem tan_diff_eq_sqrt_three (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : cos α * cos β = 1 / 6) (h5 : sin α * sin β = 1 / 3) : 
  tan (β - α) = sqrt 3 := by
  sorry

end tan_diff_eq_sqrt_three_l166_166172


namespace semi_minor_axis_zero_l166_166259

-- Define the center of the ellipse
def center : ℝ × ℝ := (2, -4)

-- Define one focus of the ellipse
def focus : ℝ × ℝ := (2, -7)

-- Define one endpoint of the semi-major axis
def endpoint : ℝ × ℝ := (2, -1)

-- Define semi-major axis length and focal distance
def a : ℝ := dist center endpoint
def c : ℝ := dist center focus

-- Define the semi-minor axis length using the ellipse relationship
def b := Real.sqrt (a^2 - c^2)

-- Prove that the semi-minor axis is zero
theorem semi_minor_axis_zero : b = 0 :=
by sorry

end semi_minor_axis_zero_l166_166259


namespace smallest_M_is_10_l166_166307

def S := {f : ℕ → ℝ // f 1 = 2 ∧ ∀ n, f (n + 1) ≥ f n ∧ f n ≥ (n / (n + 1 : ℝ)) * f (2 * n)}

noncomputable def smallest_M : ℕ :=
  Classical.some (∃ M : ℕ, ∀ f ∈ S, ∀ n, f n < M ∧ ∀ m < M, ∃ f ∈ S, ∃ n, f n ≥ m)

theorem smallest_M_is_10 : smallest_M = 10 :=
  sorry

end smallest_M_is_10_l166_166307


namespace winning_margin_was_300_l166_166023

noncomputable def total_votes : ℕ := 1000

theorem winning_margin_was_300 
  (votes_received_by_winner : ℕ)
  (percentage_received_by_winner : ℕ)
  (total_votes : ℕ)
  (votes_received_by_winner = 650) 
  (percentage_received_by_winner = 65)
  (total_votes = votes_received_by_winner * 100 / percentage_received_by_winner) :
  votes_received_by_winner - (total_votes - votes_received_by_winner) = 300 :=
  sorry

end winning_margin_was_300_l166_166023


namespace dot_product_ellipse_line_l166_166514

theorem dot_product_ellipse_line (O A B : EuclideanSpace ℝ 2)
    (a b c : ℝ) (h1 : a^2 = 2) (h2 : b^2 = 1) (h3 : c^2 = 1)
    (h4 : ∃ (F : EuclideanSpace ℝ 2), F = (1 : ℝ, 0 : ℝ))
    (l : AffineSubspace ℝ (EuclideanSpace ℝ 2)) 
    (h5 : ∃ (θ : ℝ), θ = π / 4 ∧ l.direction = Vector.fromAngle θ)
    (h6 : F ∈ l ∧ A ∈ l ∧ B ∈ l)
    (h7 : A ∈ ellipse (a,b))
    (h8 : B ∈ ellipse (a,b)) :
    (O - A) • (O - B) = -1 / 3 := sorry

end dot_product_ellipse_line_l166_166514


namespace Keikos_speed_l166_166295

-- Define the conditions for the inner and outer track lengths
def inner_track_length (r : ℝ) :=
  200 + 2 * real.pi * r

def outer_track_length (r : ℝ) :=
  200 + 2 * real.pi * (r + 8)

-- Given the time difference and Keiko's speed
def time_difference (s : ℝ) (r : ℝ) :=
  (outer_track_length r) / s - (inner_track_length r) / s = 48

-- Calculate Keiko's speed in meters per second
theorem Keikos_speed (s : ℝ) (r : ℝ) (h : time_difference s r) : 
  s = real.pi / 3 :=
by 
  sorry

end Keikos_speed_l166_166295


namespace michael_fish_count_l166_166700

theorem michael_fish_count (initial_fish : ℝ) (ben_fish : ℝ) (maria_fish : ℝ) :
  initial_fish = 49.5 → ben_fish = 18.25 → maria_fish = 23.75 → (initial_fish + ben_fish + maria_fish = 91.5) :=
by
  intros h_initial h_ben h_maria
  rw [h_initial, h_ben, h_maria]
  norm_num

-- As an optional step, define the inputs and check the result:
#eval michael_fish_count 49.5 18.25 23.75 (rfl) (rfl) (rfl) -- Expected eval result: true

end michael_fish_count_l166_166700


namespace number_of_al_atoms_is_one_l166_166066

noncomputable def num_al_atoms (molecular_weight : ℝ) (num_i_atoms : ℕ) (weight_i : ℝ) (weight_al : ℝ) : ℕ :=
  let weight_i_total := num_i_atoms * weight_i
  let weight_al_total := molecular_weight - weight_i_total
  let num_al := weight_al_total / weight_al
  Int.toNat (Real.floor (num_al + 0.5))

theorem number_of_al_atoms_is_one 
  (molecular_weight : ℝ) 
  (num_i_atoms : ℕ) 
  (weight_i : ℝ) 
  (weight_al : ℝ) 
  (h_mw : molecular_weight = 408) 
  (h_ni : num_i_atoms = 3) 
  (h_wi : weight_i = 126.90) 
  (h_wa : weight_al = 26.98) : 
  num_al_atoms molecular_weight num_i_atoms weight_i weight_al = 1 := 
by 
  sorry

end number_of_al_atoms_is_one_l166_166066


namespace find_original_number_l166_166474

theorem find_original_number (x : ℝ) (hx : 0 < x) 
  (h : 1000 * x = 3 * (1 / x)) : x ≈ 0.01732 :=
by 
  sorry

end find_original_number_l166_166474


namespace octagon_area_of_parallelogram_l166_166333

noncomputable def midpoint {α : Type*} [field α] {V : Type*} [add_comm_group V] [module α V]
  (p q : V) : V := (1 / 2) • (p + q)

structure Parallelogram (α : Type*) [field α] (V : Type*) [add_comm_group V] [module α V] :=
(A B C D : V)
(property : (B - A) + (D - A) = (C - A) + (D - B))

theorem octagon_area_of_parallelogram (α : Type*) [field α] 
  (V : Type*) [add_comm_group V] [module α V] 
  (P : Parallelogram α V) 
  (A B C D : V) 
  (h : (B - A) + (D - A) = (C - A) + (D - B)) : 
  let Q := midpoint α V A D,
      N := midpoint α V B C,
      M := midpoint α V D C,
      -- Let K, P, R be points of intersection (details skipped for brevity)
      octagon := sorry  -- definition of octagon based on given points
  in Area(octagon) = (1/6) * Area(P) :=
sorry

end octagon_area_of_parallelogram_l166_166333


namespace james_meditation_time_is_30_l166_166668

noncomputable def james_meditation_time_per_session 
  (sessions_per_day : ℕ) 
  (days_per_week : ℕ) 
  (hours_per_week : ℕ) 
  (minutes_per_hour : ℕ) : ℕ :=
  (hours_per_week * minutes_per_hour) / (sessions_per_day * days_per_week)

theorem james_meditation_time_is_30
  (sessions_per_day : ℕ) 
  (days_per_week : ℕ) 
  (hours_per_week : ℕ) 
  (minutes_per_hour : ℕ) 
  (h_sessions : sessions_per_day = 2) 
  (h_days : days_per_week = 7) 
  (h_hours : hours_per_week = 7) 
  (h_minutes : minutes_per_hour = 60) : 
  james_meditation_time_per_session sessions_per_day days_per_week hours_per_week minutes_per_hour = 30 := by
  sorry

end james_meditation_time_is_30_l166_166668


namespace garden1_tomato_percentage_l166_166711

noncomputable theory

open Locale.Real

def garden1_plants := 20
def garden2_plants := 15
def garden2_tomato_plants := garden2_plants / 3
def total_plants := garden1_plants + garden2_plants
def total_tomato_plants := total_plants * 0.20
def garden1_tomato_plants := total_tomato_plants - garden2_tomato_plants

theorem garden1_tomato_percentage :
  (garden1_tomato_plants / garden1_plants) * 100 = 10 := 
by
  sorry

end garden1_tomato_percentage_l166_166711


namespace divisibility_of_binomial_l166_166682

theorem divisibility_of_binomial (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 1) :
    (∀ x : ℕ, 1 ≤ x ∧ x ≤ n-1 → p ∣ Nat.choose n x) ↔ ∃ m : ℕ, n = p^m := sorry

end divisibility_of_binomial_l166_166682


namespace Craig_total_commission_l166_166132

structure ApplianceCommissionParams where
  base : ℝ
  percentage : ℝ

structure ApplianceSales where
  units_sold : ℕ
  total_price : ℝ

def commission (params : ApplianceCommissionParams) (sales : ApplianceSales) : ℝ :=
  sales.units_sold * (params.base + params.percentage * sales.total_price / 100)

def total_commission : ℝ :=
  let brandA_refrigerator_commission := commission ⟨75, 8⟩ ⟨3, 5280⟩
  let brandA_washing_machine_commission := commission ⟨50, 10⟩ ⟨4, 2140⟩
  let brandA_oven_commission := commission ⟨60, 12⟩ ⟨5, 4620⟩
  
  let brandB_refrigerator_commission := commission ⟨90, 6⟩ ⟨2, 3780⟩
  let brandB_washing_machine_commission := commission ⟨40, 14⟩ ⟨3, 2490⟩
  let brandB_oven_commission := commission ⟨70, 10⟩ ⟨4, 3880⟩
  
  (brandA_refrigerator_commission + brandA_washing_machine_commission 
    + brandA_oven_commission + brandB_refrigerator_commission 
    + brandB_washing_machine_commission + brandB_oven_commission)

theorem Craig_total_commission : total_commission = 9252.60 :=
by
  sorry

end Craig_total_commission_l166_166132


namespace problem_solution_l166_166561

def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | 0 < x ∧ x < 1 }
def complement_N : Set ℝ := { x | x ≤ 0 ∨ x ≥ 1 }

theorem problem_solution : M ∪ complement_N = Set.univ := 
sorry

end problem_solution_l166_166561


namespace solution_to_sequence_problem_l166_166051

def sequence_definition (x : Real) (n : Nat) : Real :=
  if h : n = 0 then
    1 + sqrt (1 + x)
  else
    let rec seq : Nat → Real
        | 0     => 1 + sqrt (1 + x)
        | k + 1 => 2 + x / seq k
    seq n

theorem solution_to_sequence_problem :
  ∀ x : Real, (∀ n : Nat, sequence_definition x (n + 1) = 2 + x / sequence_definition x n) →
  sequence_definition x 1985 = x ↔ x = 3 :=
by
  intro x h
  sorry

end solution_to_sequence_problem_l166_166051


namespace sequence_sum_proof_l166_166587

theorem sequence_sum_proof :
  (∀ n : ℕ, ∃ a : ℕ → ℕ, (∀ k : ℕ, a k = S k - S (k - 1)) ∧ S n = n^2 - 2 * n) → 
  let a (k : ℕ) := 2 * k - 3 in
  a 3 + a 17 = 34 :=
by
  sorry

end sequence_sum_proof_l166_166587


namespace sufficient_but_not_necessary_l166_166742

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1) → (2^x > 1) ∧ ¬((2^x > 1) → (x > 1)) :=
by
  sorry

end sufficient_but_not_necessary_l166_166742


namespace sequence_general_term_l166_166660

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 3 * a n - 2 * n ^ 2 + 4 * n + 4) :
  ∀ n, a n = 3^n + n^2 - n - 2 :=
sorry

end sequence_general_term_l166_166660


namespace min_abs_sum_l166_166418

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem min_abs_sum :
  ∃ (x : ℝ), (abs (x + 1) + abs (x + 3) + abs (x + 6)) = 7 :=
by {
  sorry
}

end min_abs_sum_l166_166418


namespace matrix_vector_problem_l166_166301

variables (M : Matrix (Fin 2) (Fin 2) ℝ) (v w u : Vector ℝ 2)

theorem matrix_vector_problem 
  (h1 : M.mul_vec v = ![3, 4])
  (h2 : M.mul_vec w = ![-1, -2])
  (h3 : M.mul_vec u = ![5, 1]) :
  M.mul_vec (2 • v - w + u) = ![12, 11] :=
by
  sorry

end matrix_vector_problem_l166_166301


namespace min_abs_sum_is_5_l166_166431

noncomputable def min_abs_sum (x : ℝ) : ℝ :=
  |x + 1| + |x + 3| + |x + 6|

theorem min_abs_sum_is_5 : ∃ x : ℝ, (∀ y : ℝ, min_abs_sum y ≥ min_abs_sum x) ∧ min_abs_sum x = 5 :=
by
  use -3
  sorry

end min_abs_sum_is_5_l166_166431


namespace part_I_part_II_part_III_l166_166686

variables {α : Type*} [linear_order α] [has_sub α] [has_add α] [has_le α]

-- Definition of a unimodal function
def unimodal (f : α → ℝ) (x* : α) (a b : α) : Prop :=
  (a ≤ x* ∧ x* ≤ b) ∧
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ x* → f x ≤ f y) ∧
  (∀ x y, x* ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y)

-- Statement of (I)
theorem part_I (f : α → ℝ) (x* x1 x2 : α) (h : unimodal f x* 0 1) 
  (h12 : x1 < x2) : (f x1 ≥ f x2 → x* ≤ x2) ∧ (f x1 ≤ f x2 → x* ≥ x1) :=
sorry

-- Statement of (II)
theorem part_II (f : α → ℝ) (x* x1 x2 : α) (r : ℝ) (h : unimodal f x* 0 1) 
  (x1x2 : x1 < x2) (hx1x2 : x2 - x1 ≥ 2 * r) (hr : 0 < r ∧ r < 0.5) : 
  ∃ x1 x2, x2 - x1 = 2 * r ∧ max x2 (1 - x1) ≤ 0.5 + r :=
sorry

-- Statement of (III)
theorem part_III (f : α → ℝ) (x1 x2 x3 : α) (h : unimodal f x* 0 1)
  (hx1x2 : x1 < x2) (hx3 : x3 + x1 = x2) 
  (h_diff : |x1 - x3| ≥ 0.02 ∧ |x2 - x3| ≥ 0.02 ∧ |x2 - x1| ≥ 0.02) :
  ∃ (x1 x2 x3 : α), peak_interval_length (unimodal f x* x1 x2 x3) = 0.34 :=
sorry

end part_I_part_II_part_III_l166_166686


namespace max_size_T_l166_166217

def S : set (ℕ × ℕ) := {(x, y) | x ∈ finset.range 1993 ∧ y ∈ finset.range 4}

def T (T : set (ℕ × ℕ)) : Prop :=
  T ⊆ S ∧ (∀ p1 p2 p3 p4 ∈ T, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p1 ≠ p3 ∧ 
    p1 ≠ p4 ∧ p2 ≠ p4 → ¬is_square p1 p2 p3 p4)

def is_square (p1 p2 p3 p4 : (ℕ × ℕ)) : Prop :=
  p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.2 = p3.2 ∧ p2.2 = p4.2 ∧ 
  abs (p1.1 - p3.1) = abs (p1.2 - p2.2)

theorem max_size_T : ∃ T : set (ℕ × ℕ), T T ∧ finset.card T = 4983 := sorry

end max_size_T_l166_166217


namespace normal_distribution_probability_l166_166213

noncomputable def probability (X : ℝ → ℝ) [measure_theory.measure_space ℝ] (s : set ℝ) : ℝ :=
measure_theory.measure_theory.probability P, X, s -- assume probability function

theorem normal_distribution_probability :
  ∀ (X : ℝ → ℝ) (μ σ : ℝ) (P : measure_theory.measure_space ℝ), 
  -- X is normally distributed with mean μ and variance σ^2
  (X ~ measure_theory.measure_theory.normal(2, σ^2)) ∧ 
  -- Given condition
  (probability(X, {x | x ≥ 4}) = 0.2) → 
  -- Prove that P(0 < X < 4) = 0.6
  probability(X, {x | 0 < x ∧ x < 4}) = 0.6 := by
  sorry

end normal_distribution_probability_l166_166213


namespace num_integer_b_for_quadratic_inequality_l166_166564

theorem num_integer_b_for_quadratic_inequality :
  {b : ℤ // ∃ (S : Finset ℤ), S.card = 3 ∧ ∀ x : ℤ, x ∈ S → x^2 + b*x + 4 ≤ 0}.to_finset.card = 4 :=
sorry

end num_integer_b_for_quadratic_inequality_l166_166564


namespace find_integer_divisible_by_18_and_sqrt_between_26_and_26_2_l166_166574

theorem find_integer_divisible_by_18_and_sqrt_between_26_and_26_2 : 
    ∃ n : ℕ, (n % 18 = 0) ∧ (676 ≤ n ∧ n ≤ 686) ∧ n = 684 :=
by
  use 684
  split
  {
    norm_num
  }
  split
  {
    norm_num
  }
  {
    refl
  }

end find_integer_divisible_by_18_and_sqrt_between_26_and_26_2_l166_166574


namespace frequency_approx_probability_l166_166870

-- Definitions directly from the conditions in the problem
def frequency (n : ℕ) (trials : ℕ → bool) : ℚ := (finset.filter id (finset.range n).val).card / n

def probability (event : Type) (sample_space : set event) : ℚ :=
finset.card (event ∩ sample_space) / finset.card sample_space

-- Statement to be proved: As the number of trials increases, the frequency gets closer to the probability
theorem frequency_approx_probability 
    (n : ℕ) (trials : ℕ → bool) (event : Type) (sample_space : set event) :
    as n → ∞, frequency n trials → probability event sample_space :=
sorry

end frequency_approx_probability_l166_166870


namespace compute_xy_l166_166088

theorem compute_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h1 : x ^ (Real.sqrt y) = 27) (h2 : (Real.sqrt x) ^ y = 9) :
  x * y = 12 * Real.sqrt 3 :=
sorry

end compute_xy_l166_166088


namespace total_value_of_remaining_books_l166_166105

-- initial definitions
def total_books : ℕ := 55
def hardback_books : ℕ := 10
def hardback_price : ℕ := 20
def paperback_price : ℕ := 10
def books_sold : ℕ := 14

-- calculate remaining books
def remaining_books : ℕ := total_books - books_sold

-- calculate remaining hardback and paperback books
def remaining_hardback_books : ℕ := hardback_books
def remaining_paperback_books : ℕ := remaining_books - remaining_hardback_books

-- calculate total values
def remaining_hardback_value : ℕ := remaining_hardback_books * hardback_price
def remaining_paperback_value : ℕ := remaining_paperback_books * paperback_price

-- total value of remaining books
def total_remaining_value : ℕ := remaining_hardback_value + remaining_paperback_value

theorem total_value_of_remaining_books : total_remaining_value = 510 := by
  -- calculation steps are skipped as instructed
  sorry

end total_value_of_remaining_books_l166_166105


namespace problem_1_problem_2_l166_166942

-- Problem 1
theorem problem_1 (x : ℝ) : (2*x - 1) * (2*x - 3) - (1 - 2*x) * (2 - x) = 2*x^2 - 3*x + 1 :=
by {
  -- Proof omitted
  sorry,
}

-- Problem 2
theorem problem_2 (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) : (a^2 - 1) / a * (1 - (2*a + 1) / (a^2 + 2*a + 1)) / (a - 1) = a / (a + 1) :=
by {
  -- Proof omitted
  sorry,
}

end problem_1_problem_2_l166_166942


namespace axis_of_symmetry_translated_function_l166_166387

theorem axis_of_symmetry_translated_function :
  ∀ (k : ℤ), ∀x (h : x = -π / 6 + k * π / 2), (∃ y, y = 2 * cos (2 * (x + π / 6))) → x = -π / 6 + k * π / 2 :=
by
  sorry

end axis_of_symmetry_translated_function_l166_166387


namespace find_y_values_l166_166694

open Real

-- Problem statement as a Lean statement.
theorem find_y_values (x : ℝ) (hx : x^2 + 2 * (x / (x - 1)) ^ 2 = 20) :
  ∃ y : ℝ, (y = ((x - 1) ^ 3 * (x + 2)) / (2 * x - 1)) ∧ (y = 14 ∨ y = -56 / 3) := 
sorry

end find_y_values_l166_166694


namespace divisibility_by_2_criterion_l166_166893

theorem divisibility_by_2_criterion (d : ℕ) (a : ℕ → ℕ) (n : ℕ) 
  (h_odd : d % 2 = 1)
  (x_def : ∑ i in finset.range n, a i * d^(n - 1 - i) = x) :
  (x % 2 = 0 ↔ ∑ i in finset.range n, a i % 2 = 0) :=
sorry

end divisibility_by_2_criterion_l166_166893


namespace probability_x_plus_2y_leq_6_l166_166909

noncomputable def probability_condition (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5 ∧ x + 2*y ≤ 6

theorem probability_x_plus_2y_leq_6 :
  let probability := (8 / 20 : ℝ)
  in probability = 2 / 5 :=
  sorry

end probability_x_plus_2y_leq_6_l166_166909


namespace larger_package_size_l166_166962

noncomputable def larger_package_cupcakes (x : ℕ) : Prop :=
  ∃ (packs_of_larger_package : ℕ), packs_of_larger_package > 0 ∧
  let packs_of_10 := 4 in
  let total_cupcakes_needed := 100 in
  let cupcakes_from_10_packs := packs_of_10 * 10 in
  total_cupcakes_needed = cupcakes_from_10_packs + packs_of_larger_package * x

theorem larger_package_size : larger_package_cupcakes 60 :=
sorry

end larger_package_size_l166_166962


namespace slope_angle_tangent_l166_166764

-- Define the curve function
def curve (x : ℝ) : ℝ := x^3 - 4 * x

-- Define the slope function which is the derivative of the curve
def slope (x : ℝ) : ℝ := 3 * x^2 - 4

-- Define the given point
def point : ℝ × ℝ := (1, curve 1)

-- Main theorem statement: The slope angle of the tangent to the curve at the point (1, -3) is 3π/4
theorem slope_angle_tangent :
  ∀ (α : ℝ), slope 1 = -1 → α = Real.arctan (-1) → α = 3 * Real.pi / 4 :=
by
  intro α h_slope h_arctan
  -- Skipping proof details
  sorry

end slope_angle_tangent_l166_166764


namespace point_A_on_segment_BC_l166_166716

theorem point_A_on_segment_BC (A B C : Point) : 
  (∀ M : Point, dist M A ≤ dist M B ∨ dist M A ≤ dist M C) → on_segment A B C :=
begin
  intro h,
  sorry
end

end point_A_on_segment_BC_l166_166716


namespace rhombus_area_l166_166745

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 25) (h2 : d2 = 50) : 
  (d1 * d2) / 2 = 625 :=
by
  rw [h1, h2]
  norm_num
  sorry

end rhombus_area_l166_166745


namespace probability_of_centrally_symmetric_card_l166_166872

def is_centrally_symmetric (shape : String) : Bool :=
  shape = "parallelogram" ∨ shape = "circle"

theorem probability_of_centrally_symmetric_card :
  let shapes := ["parallelogram", "isosceles_right_triangle", "regular_pentagon", "circle"]
  let total_cards := shapes.length
  let centrally_symmetric_cards := shapes.filter is_centrally_symmetric
  let num_centrally_symmetric := centrally_symmetric_cards.length
  (num_centrally_symmetric : ℚ) / (total_cards : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_centrally_symmetric_card_l166_166872


namespace sum_possible_values_bella_card_l166_166671

theorem sum_possible_values_bella_card (x y : ℝ) (hx : 0 < x ∧ x < 90) (hy : 0 < y ∧ y < 90)
  (dist : ∀ f g : ℝ → ℝ, f x ≠ g x ∨ f y ≠ g y)
  (bella_identify : ∃ sec_y, sec_y = 2 ∧ 
    (sec y = sec_y → ∀ f, f x ≠ sec_y ∧ f y ≠ sec_y)) :
  (∑ v in {sec y}, v) = 2 := 
sorry

end sum_possible_values_bella_card_l166_166671


namespace comprehensive_survey_suitability_l166_166871

-- Define the survey options as an enumeration
inductive SurveyOption 
| EnvironmentalAwareness
| MooncakeQuality
| StudentWeight
| FireworkSafety

-- Define the conditions for a comprehensive survey suitability
def is_comprehensive_survey_suitable (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.StudentWeight => true
  | _ => false

-- The statement of the proof problem
theorem comprehensive_survey_suitability : 
  ∃ (option : SurveyOption), is_comprehensive_survey_suitable option ∧ option = SurveyOption.StudentWeight :=
by
  exists SurveyOption.StudentWeight
  split
  · rfl
  · rfl

end comprehensive_survey_suitability_l166_166871


namespace largest_sum_l166_166115

theorem largest_sum :
  let s1 := (1 : ℚ) / 3 + (1 : ℚ) / 4
  let s2 := (1 : ℚ) / 3 + (1 : ℚ) / 5
  let s3 := (1 : ℚ) / 3 + (1 : ℚ) / 2
  let s4 := (1 : ℚ) / 3 + (1 : ℚ) / 9
  let s5 := (1 : ℚ) / 3 + (1 : ℚ) / 6
  in max s1 (max s2 (max s3 (max s4 s5))) = (5 : ℚ) / 6 := by
  sorry

end largest_sum_l166_166115


namespace intersection_A_B_l166_166215

def A (x : ℝ) : Prop := x > 3
def B (x : ℝ) : Prop := x ≤ 4

theorem intersection_A_B : {x | A x} ∩ {x | B x} = {x | 3 < x ∧ x ≤ 4} :=
by
  sorry

end intersection_A_B_l166_166215


namespace eleventh_number_in_list_is_145_l166_166090

theorem eleventh_number_in_list_is_145 :
  ∃ (n : ℕ), (∀ m < n, n > 0 ∧ (∑ d in m.digits, d) = 13) ∧ (∀ (k : ℕ), k < n → (∑ d in k.digits, d) = 13 → k < 145) ∧ n = 145 :=
by sorry

end eleventh_number_in_list_is_145_l166_166090


namespace summer_camp_sampling_proof_l166_166343

noncomputable def summer_camp_sampling_problem : Prop :=
  let total_students := 500
  let camp_1_students := 200
  let camp_2_students := 150
  let camp_3_students := 150
  let sample_size := 50
  let sampling_ratio := sample_size / total_students in
  let drawn_camp_1 := sampling_ratio * camp_1_students
  let drawn_camp_2 := sampling_ratio * camp_2_students
  let drawn_camp_3 := sampling_ratio * camp_3_students in
  drawn_camp_1 = 20 ∧ drawn_camp_2 = 15 ∧ drawn_camp_3 = 15

theorem summer_camp_sampling_proof : summer_camp_sampling_problem :=
  by
    -- Proof will be provided here
    sorry

end summer_camp_sampling_proof_l166_166343


namespace probability_at_least_one_boy_one_girl_l166_166923

theorem probability_at_least_one_boy_one_girl :
  (∀ (P : SampleSpace → Prop), (P = (fun outcomes => nat.size outcomes = 4
                            ∧ (∃ outcome : outcomes, outcome = "boy")
                            ∧ ∃ outcome : outcomes, outcome = "girl"))
  -> (probability P = 7/8)) :=
by
  sorry

end probability_at_least_one_boy_one_girl_l166_166923


namespace part_1_part_2_1_part_2_2_l166_166286

variable (A B C a b c : ℝ)

-- Given conditions
def triangle_angles := 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π
def triangle_sides := a > 0 ∧ b > 0 ∧ c > 0

def equation :=
  4 * (Real.sin ((A - B) / 2))^2 + 4 * Real.sin A * Real.sin B = 2 + Real.sqrt 2

def area_eq := 6
def side_b := 4

-- Proof goal for each part

-- (1) Prove C = π / 4
theorem part_1 (h1 : triangle_angles A B C) (h2 : triangle_sides a b c) (h3 : equation A B) : C = π / 4 := 
sorry

-- (2.1) Given additional conditions to prove c = √10
theorem part_2_1 (h1 : triangle_angles A B C) (h2 : triangle_sides a b c) 
  (h3 : equation A B) (h4 : C = π / 4) (h5 : b = 4) (h6 : area_eq = 6) : c = Real.sqrt 10 := 
sorry

-- (2.2) Given additional conditions to prove tan(2B - C) = 7
theorem part_2_2 (h1 : triangle_angles A B C) (h2 : triangle_sides a b c) 
  (h3 : equation A B) (h4 : C = π / 4) (h5 : b = 4) (h6 : area_eq = 6) : Real.tan (2 * B - C) = 7 :=
sorry

end part_1_part_2_1_part_2_2_l166_166286


namespace original_number_value_l166_166451

noncomputable def orig_number_condition (x : ℝ) : Prop :=
  1000 * x = 3 * (1 / x)

theorem original_number_value : ∃ x : ℝ, 0 < x ∧ orig_number_condition x ∧ x = √(30) / 100 :=
begin
  -- the proof
  sorry
end

end original_number_value_l166_166451


namespace number_not_divisible_by_6_or_11_l166_166361

theorem number_not_divisible_by_6_or_11 (n : ℕ) (h₁ : n = 1500) :
  (∑ i in Finset.range n, if ¬((i % 6 = 0) ∨ (i % 11 = 0)) then 1 else 0) = 1136 := 
sorry

end number_not_divisible_by_6_or_11_l166_166361


namespace Kyle_older_than_Julian_l166_166678

variable (Tyson_age : ℕ)
variable (Frederick_age Julian_age Kyle_age : ℕ)

-- Conditions
def condition1 := Tyson_age = 20
def condition2 := Frederick_age = 2 * Tyson_age
def condition3 := Julian_age = Frederick_age - 20
def condition4 := Kyle_age = 25

-- The proof problem (statement only)
theorem Kyle_older_than_Julian :
  Tyson_age = 20 ∧
  Frederick_age = 2 * Tyson_age ∧
  Julian_age = Frederick_age - 20 ∧
  Kyle_age = 25 →
  Kyle_age - Julian_age = 5 := by
  intro h
  sorry

end Kyle_older_than_Julian_l166_166678


namespace sum_x1_x2_l166_166739

open ProbabilityTheory

variable {Ω : Type*} {X : Ω → ℝ}
variable (p1 p2 : ℝ) (x1 x2 : ℝ)
variable (h1 : 2/3 * x1 + 1/3 * x2 = 4/9)
variable (h2 : 2/3 * (x1 - 4/9)^2 + 1/3 * (x2 - 4/9)^2 = 2)
variable (h3 : x1 < x2)

theorem sum_x1_x2 : x1 + x2 = 17/9 :=
by
  sorry

end sum_x1_x2_l166_166739


namespace lowest_possible_price_l166_166877

theorem lowest_possible_price (msrp : ℝ) (regular_discount : ℝ) (additional_sale_discount : ℝ) 
  (h_msrp : msrp = 30) (h_regular_discount_range : 0.1 ≤ regular_discount ∧ regular_discount ≤ 0.3) 
  (h_max_regular_discount : regular_discount = 0.3) (h_additional_sale_discount : additional_sale_discount = 0.2) : 
  let regular_discounted_price := msrp * (1 - regular_discount) in
  let final_price := regular_discounted_price * (1 - additional_sale_discount) in
  final_price = 16.8 :=
by
  sorry

end lowest_possible_price_l166_166877


namespace min_value_of_f_max_value_of_f_l166_166606

open Real

noncomputable def f (x : ℝ) := (1/(cos x)^2) + 2 * tan x + 1

theorem min_value_of_f :
  ∃ x ∈ Icc (-π/3) (π/4), f x = 1 ∧ ∀ y ∈ Icc (-π/3) (π/4), f y ≥ 1 :=
begin
  sorry
end

theorem max_value_of_f :
  ∃ x ∈ Icc (-π/3) (π/4), f x = 5 ∧ ∀ y ∈ Icc (-π/3) (π/4), f y ≤ 5 :=
begin
  sorry
end

end min_value_of_f_max_value_of_f_l166_166606


namespace symmetry_line_intersection_l166_166496

theorem symmetry_line_intersection 
  (k : ℝ) (k_pos : k > 0) (k_ne_one : k ≠ 1)
  (k1 : ℝ) (h_sym : ∀ (P : ℝ × ℝ), (P.2 = k1 * P.1 + 1) ↔ P.2 - 1 = k * (P.1 + 1) + 1)
  (H : ∀ M : ℝ × ℝ, (M.2 = k * M.1 + 1) → (M.1^2 / 4 + M.2^2 = 1)) :
  (k * k1 = 1) ∧ (∀ k : ℝ, ∃ P : ℝ × ℝ, (P.fst = 0) ∧ (P.snd = -5 / 3)) :=
sorry

end symmetry_line_intersection_l166_166496


namespace num_x_intercepts_in_interval_l166_166969

-- Definitions and conditions
def f (x : ℝ) : ℝ := sin (1 / x) + cos (1 / x)
def interval : set ℝ := { x | 0.00001 < x ∧ x < 0.0001 }

-- Statement of the theorem to prove
theorem num_x_intercepts_in_interval : 
  (∃ (n : ℕ), n = 28648 ∧ (∀ x ∈ interval, f x = 0) ↔ n = 28648) := 
sorry

end num_x_intercepts_in_interval_l166_166969


namespace area_of_section_ABD_l166_166076
-- Import everything from the Mathlib library

-- Define the conditions
def is_equilateral_triangle (a b c : ℝ) (ABC_angle : ℝ) : Prop := 
  a = b ∧ b = c ∧ ABC_angle = 60

def plane_angle (angle : ℝ) : Prop := 
  angle = 35 + 18/60

def volume_of_truncated_pyramid (volume : ℝ) : Prop := 
  volume = 15

-- The main theorem based on the above conditions
theorem area_of_section_ABD
  (a b c ABC_angle : ℝ)
  (S : ℝ)
  (V : ℝ)
  (h1 : is_equilateral_triangle a b c ABC_angle)
  (h2 : plane_angle S)
  (h3 : volume_of_truncated_pyramid V) :
  ∃ (area : ℝ), area = 16.25 :=
by
  -- skipping the proof
  sorry

end area_of_section_ABD_l166_166076


namespace min_abs_sum_is_two_l166_166426

theorem min_abs_sum_is_two : ∃ x ∈ set.Ioo (- ∞) (∞), ∀ y ∈ set.Ioo (- ∞) (∞), (|y + 1| + |y + 3| + |y + 6| ≥ 2) ∧ (|x + 1| + |x + 3| + |x + 6| = 2) := sorry

end min_abs_sum_is_two_l166_166426


namespace find_N_divisible_by_18_and_within_bounds_l166_166576

theorem find_N_divisible_by_18_and_within_bounds :
  ∃ (N : ℕ), (N > 0) ∧ (N % 18 = 0) ∧ (676 ≤ N ∧ N ≤ 686) ∧ N = 684 :=
by
  use 684
  split
  sorry

end find_N_divisible_by_18_and_within_bounds_l166_166576


namespace sequence_general_formula_l166_166198

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else 3 * (seq (n - 1))^2 + 12 * (seq (n - 1)) + 10

theorem sequence_general_formula (n : ℕ) : seq n = 3^(2^(n-1)) - 2 := 
  sorry

end sequence_general_formula_l166_166198


namespace find_m_l166_166638

theorem find_m (m : ℤ) (x y : ℤ) (h1 : x = 1) (h2 : y = m) (h3 : 3 * x - 4 * y = 7) : m = -1 :=
by
  sorry

end find_m_l166_166638


namespace f_integer_f_non_integer_l166_166297

noncomputable def f (x : ℝ) : ℝ :=
  ⌊x⌋ + ⌊1 - x⌋

theorem f_integer (x : ℝ) (hx : x.floor : ℤ = x) : f x = 1 :=
by
  sorry

theorem f_non_integer (x : ℝ) (hx : x.floor : ℤ ≠ x) : f x = 0 :=
by
  sorry

end f_integer_f_non_integer_l166_166297


namespace sum_first_15_odd_integers_l166_166810

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l166_166810


namespace internal_bisector_projections_l166_166356

open Real

variables {A B C M D E : Point}
variables {circumcircle : Circle}
variables (AB AC BD CE AD AE : ℝ)

-- Definitions indicating geometric constructions and properties
def is_bisector (A B C M: Point) : Prop :=
  is_intersection_of_bisector_and_circle A B C M

def is_projection (M D AB: Point) : Prop :=
  is_perpendicular (line_through M D) (line_through A B)

def is_projection (M E AC: Point) : Prop :=
  is_perpendicular (line_through M E) (line_through A C)

-- Main theorem statement
theorem internal_bisector_projections (h1 : is_bisector A B C M)
    (h2 : circumcircle.contains M)
    (h3 : is_projection M D (line_through A B))
    (h4 : is_projection M E (line_through A C)) :
  BD = CE ∧ AD = AE ∧ AD = AE = (1/2) * (AB + AC) :=
by sorry

end internal_bisector_projections_l166_166356


namespace temperature_difference_l166_166709

theorem temperature_difference
  (lowest_temp : ℤ) (highest_temp : ℤ)
  (h1 : lowest_temp = -2)
  (h2 : highest_temp = 9) :
  highest_temp - lowest_temp = 11 :=
by
  rw [h2, h1]
  norm_num
  sorry

end temperature_difference_l166_166709


namespace sum_of_first_n_odd_numbers_l166_166836

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l166_166836


namespace rectangle_diagonal_length_l166_166008

theorem rectangle_diagonal_length (l : ℝ) (L W d : ℝ) 
  (h_ratio : L = 5 * l ∧ W = 2 * l)
  (h_perimeter : 2 * (L + W) = 100) :
  d = (5 * Real.sqrt 290) / 7 :=
by
  sorry

end rectangle_diagonal_length_l166_166008


namespace negation_proposition_l166_166755

theorem negation_proposition (a b : ℝ) :
  ¬ (∀ c : ℝ, a < b → ac^2 < bc^2) ↔ (a < b ∧ ∃ c : ℝ, ac^2 ≥ bc^2) :=
by
  sorry

end negation_proposition_l166_166755


namespace math_problem_equivalence_l166_166273

-- Define the Cartesian equation of curve (C_1)
def curveC1_cartesian (x y : ℝ) : Prop :=
  (x^2) / 3 + y^2 = 1

-- Define the polar equation of curve (C_2)
def curveC2_polar (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.sin (θ + Real.pi / 3)

-- Define ray OM in polar coordinates
def ray_OM_polar (ρ θ : ℝ) : Prop :=
  θ = Real.pi / 6 ∧ ρ ≥ 0

-- Prove the equivalence between Cartesian and polar equations and the distance
theorem math_problem_equivalence
  (x y ρ θ : ℝ)
  (hC1_cart : curveC1_cartesian x y)
  (hC2_pol : curveC2_polar ρ θ)
  (hOM_pol : ray_OM_polar ρ θ) :
  (∃ θ, ρ^2 = 3 / (1 + 2 * Real.sin θ ^ 2)) ∧
  (∃ (x y), x = Real.sqrt 3 ∧ y = 1 ∧ (x - Real.sqrt 3)^2 + (y - 1)^2 = 4) ∧
  (∃ (ρA ρB : ℝ), (ρA^2 = 2) ∧ (ρB^2 = 6 / 5) ∧ (Real.sqrt (ρA^2 + ρB^2) = 4 * Real.sqrt 5 / 5)) :=
sorry

end math_problem_equivalence_l166_166273


namespace sum_first_15_odd_integers_l166_166851

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l166_166851


namespace ab_can_do_work_in_8_days_l166_166873

noncomputable theory

variables (A B C : ℝ) (x : ℝ)

def condition_1 := (A + B) * x = 1
def condition_2 := (B + C) * 12 = 1
def condition_3 := (A + B + C) * 6 = 1
def condition_4 := (A + C) * 8 = 1

theorem ab_can_do_work_in_8_days
  (h1 : condition_1 A B x)
  (h2 : condition_2 B C)
  (h3 : condition_3 A B C)
  (h4 : condition_4 A C) :
  x = 8 := 
sorry

end ab_can_do_work_in_8_days_l166_166873


namespace min_abs_sum_is_two_l166_166429

theorem min_abs_sum_is_two : ∃ x ∈ set.Ioo (- ∞) (∞), ∀ y ∈ set.Ioo (- ∞) (∞), (|y + 1| + |y + 3| + |y + 6| ≥ 2) ∧ (|x + 1| + |x + 3| + |x + 6| = 2) := sorry

end min_abs_sum_is_two_l166_166429


namespace hyperbola_right_focus_l166_166348

theorem hyperbola_right_focus (x y : ℝ) (h : x^2 - 2 * y^2 = 1) : x = sqrt 6 / 2 ∧ y = 0 := sorry

end hyperbola_right_focus_l166_166348


namespace tom_total_spent_on_video_games_l166_166024

-- Conditions
def batman_game_cost : ℝ := 13.6
def superman_game_cost : ℝ := 5.06

-- Statement to be proved
theorem tom_total_spent_on_video_games : batman_game_cost + superman_game_cost = 18.66 := by
  sorry

end tom_total_spent_on_video_games_l166_166024


namespace expand_polynomial_l166_166572

theorem expand_polynomial (x : ℝ) : (x + 4) * (5 * x - 10) = 5 * x ^ 2 + 10 * x - 40 := by
  sorry

end expand_polynomial_l166_166572


namespace water_tank_capacity_l166_166045

theorem water_tank_capacity
  (tank_capacity : ℝ)
  (h : 0.30 * tank_capacity = 0.90 * tank_capacity - 54) :
  tank_capacity = 90 :=
by
  -- proof goes here
  sorry

end water_tank_capacity_l166_166045


namespace sum_first_15_odd_integers_l166_166850

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l166_166850


namespace sum_of_legs_of_similar_larger_triangle_l166_166392

-- Define the conditions for the problem
def smaller_triangle_area : ℝ := 10
def larger_triangle_area : ℝ := 400
def smaller_triangle_hypotenuse : ℝ := 10

-- Define the correct answer (sum of the lengths of the legs of the larger triangle)
def sum_of_legs_of_larger_triangle : ℝ := 88.55

-- State the Lean theorem
theorem sum_of_legs_of_similar_larger_triangle :
  (∀ (A B C a b c : ℝ), 
    a * b / 2 = smaller_triangle_area ∧ 
    c = smaller_triangle_hypotenuse ∧
    C * C / 4 = larger_triangle_area / smaller_triangle_area ∧
    A / a = B / b ∧ 
    A^2 + B^2 = C^2 → 
    A + B = sum_of_legs_of_larger_triangle) :=
  by sorry

end sum_of_legs_of_similar_larger_triangle_l166_166392


namespace range_of_function_l166_166196

-- Define the function y = √3 sin x - cos x
def func (x : ℝ) : ℝ := (sqrt 3) * sin x - cos x

-- Define the range of x for the given problem
def x_interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ π

-- Define the expected range of the function
def expected_range (y : ℝ) : Prop := -1 ≤ y ∧ y ≤ 2

-- State the problem in Lean: Given x in [0, π], show that the range of y is [-1, 2]
theorem range_of_function : 
  ∀ (x : ℝ), x_interval x → expected_range (func x) :=
sorry

end range_of_function_l166_166196


namespace exists_competitor_with_at_least_five_friends_l166_166666

theorem exists_competitor_with_at_least_five_friends
    (competitors : Finset ℕ)
    (h1 : competitors.card = 12)
    (h2 : ∀ (x y : ℕ), x ≠ y → x ∈ competitors → y ∈ competitors 
          → ∃ z : ℕ, z ∈ competitors ∧ z ≠ x ∧ z ≠ y) :
    ∃ (x : ℕ), x ∈ competitors ∧ (competitors.filter (λ y, y ≠ x ∧ ∃ z, z ∈ competitors ∧ z = x ∧ z = y)).card ≥ 5 := 
sorry

end exists_competitor_with_at_least_five_friends_l166_166666


namespace find_b_plus_c_range_of_b_plus_c_l166_166199

variable {A B C : Real}
variable (a b c : ℝ)
variable (m n : Vector ℝ 2)

-- Conditions
def internalAngles : Prop := A + B + C = π
def sideOpposite : Prop := a = 2 * sqrt 3
def mVector : Prop := m = vector [-cos (A/2), sin (A/2)]
def nVector : Prop := n = vector [cos (A/2), sin (A/2)]
def dotProduct : Prop := m.dotProduct n = 1/2

-- Area condition
def triangleArea (S : ℝ) : Prop := S = sqrt 3

-- Problem 1
theorem find_b_plus_c (S : ℝ) (hS : triangleArea S) (hAngles : internalAngles) (hSide : sideOpposite) (hmVect : mVector) (hnVect : nVector) (hdot : dotProduct) :
  b + c = 4 := sorry

-- Problem 2
theorem range_of_b_plus_c (S : ℝ) (hS : triangleArea S) (hAngles : internalAngles) (hSide : sideOpposite) (hmVect : mVector) (hnVect : nVector) (hdot : dotProduct) :
  2 * sqrt 3 < b + c ∧ b + c ≤ 4 := sorry

end find_b_plus_c_range_of_b_plus_c_l166_166199


namespace cost_of_fencing_l166_166344

def π := 3.14159
def hectares_to_sqm (h : ℝ) : ℝ := h * 10000
def calculate_radius (A : ℝ) : ℝ := real.sqrt (A / π)
def calculate_circumference (r : ℝ) : ℝ := 2 * π * r
def calculate_cost (C : ℝ) (rate : ℝ) : ℝ := C * rate

theorem cost_of_fencing (A_hectares : ℝ) (rate_per_meter : ℝ) (expected_cost : ℝ) :
  hectares_to_sqm A_hectares = 175600 →
  calculate_radius (hectares_to_sqm A_hectares) ≈ 236.43 →
  calculate_circumference (calculate_radius (hectares_to_sqm A_hectares)) ≈ 1484.22 →
  calculate_cost (calculate_circumference (calculate_radius (hectares_to_sqm A_hectares))) rate_per_meter ≈ expected_cost :=
begin
  -- Given values
  have A_sqm : ℝ := 175600,
  have radius : ℝ := 236.43,
  have circumference : ℝ := 1484.22,
  have fencing_cost : ℝ := 4452.66,

  -- Conditions from the problem
  assume h_eq : hectares_to_sqm A_hectares = A_sqm,
  assume r_eq : calculate_radius A_sqm ≈ radius,
  assume c_eq : calculate_circumference radius ≈ circumference,

  -- Proof (skipped, just using sorry)
  sorry
end

end cost_of_fencing_l166_166344


namespace min_abs_sum_is_5_l166_166433

noncomputable def min_abs_sum (x : ℝ) : ℝ :=
  |x + 1| + |x + 3| + |x + 6|

theorem min_abs_sum_is_5 : ∃ x : ℝ, (∀ y : ℝ, min_abs_sum y ≥ min_abs_sum x) ∧ min_abs_sum x = 5 :=
by
  use -3
  sorry

end min_abs_sum_is_5_l166_166433


namespace length_of_chord_l166_166358

-- Define the first circle
def circle1 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 4 = 0

-- Define the second circle
def circle2 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 4*x + 4*y - 12 = 0

-- Define the length of the common chord
def length_of_common_chord : ℝ := 2 * real.sqrt 2

-- The theorem stating the length of the common chord
theorem length_of_chord : 
  (∃ x y, circle1 x y ∧ circle2 x y) → 2 * real.sqrt 2 = length_of_common_chord := 
by
  sorry

end length_of_chord_l166_166358


namespace prop_logic_example_l166_166251

theorem prop_logic_example (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end prop_logic_example_l166_166251


namespace find_original_number_l166_166467

noncomputable def original_number (x : ℝ) : Prop :=
  (0 < x) ∧ (1000 * x = 3 / x)

theorem find_original_number : ∃ x : ℝ, original_number x ∧ x = real.sqrt 30 / 100 :=
by
  sorry

end find_original_number_l166_166467


namespace peter_flight_distance_l166_166328

theorem peter_flight_distance :
  ∀ (distance_spain_to_russia distance_spain_to_germany : ℕ),
  distance_spain_to_russia = 7019 →
  distance_spain_to_germany = 1615 →
  (distance_spain_to_russia - distance_spain_to_germany) + 2 * distance_spain_to_germany = 8634 :=
by
  intros distance_spain_to_russia distance_spain_to_germany h1 h2
  rw [h1, h2]
  sorry

end peter_flight_distance_l166_166328


namespace min_value_of_f_l166_166412

def f (x : ℝ) := abs (x + 1) + abs (x + 3) + abs (x + 6)

theorem min_value_of_f : ∃ (x : ℝ), f x = 5 :=
by
  use -3
  simp [f]
  sorry

end min_value_of_f_l166_166412


namespace hyperbola_focus_l166_166349

theorem hyperbola_focus :
  (∃ (c : ℝ), (∀ x y : ℝ, x^2 - 2 * y^2 = 1 
  → (x = (c) ∨ x = (-c)) ∧ y = 0) := by
sorry

end hyperbola_focus_l166_166349


namespace determine_winner_l166_166536

-- Definitions for the lengths of the strips and the piece
variables (a b d : ℕ)

-- Define the number of pieces
def num_pieces (len d : ℕ) : ℕ := len / d

theorem determine_winner (a b d : ℕ) : 
  let x := num_pieces a d in
  let y := num_pieces b d in
  (x + y) % 2 = 1 ∨ (x + y) % 2 = 0 :=
by
  sorry

end determine_winner_l166_166536


namespace distance_between_points_PQ_l166_166522

noncomputable def tetrahedron_vertices : Type := ℝ × ℝ × ℝ

def tetrahedron (A B C D : tetrahedron_vertices) : Prop :=
  ∀ (A B C D : tetrahedron_vertices),
  dist A B = 1 ∧ dist A C = 1 ∧ dist A D = 1 ∧ dist B C = 1 ∧ dist B D = 1 ∧ dist C D = 1

def pointP(A B : tetrahedron_vertices) : tetrahedron_vertices :=
  (3 * A + B) / 4

def pointQ(C D : tetrahedron_vertices) : tetrahedron_vertices :=
  (3 * C + D) / 4

def distancePQ(P Q : tetrahedron_vertices) : ℝ :=
  dist P Q

theorem distance_between_points_PQ (A B C D : tetrahedron_vertices) (tetrahedron A B C D) :
  distancePQ (pointP A B) (pointQ C D) = sqrt (3 / 2) :=
sorry

end distance_between_points_PQ_l166_166522


namespace choose_roots_sum_eq_l166_166129

-- Define the quadratic equation as per the conditions
def quadratic_equations (i : ℕ) : ℝ → ℝ :=
  λ x, x^2 + (b i) * x + (c i)

-- Define the discriminants as per the conditions
def discriminants (i : ℕ) : ℝ :=
  (b i)^2 - 4 * (c i)

theorem choose_roots_sum_eq
  (b : ℕ → ℝ)
  (c : ℕ → ℝ)
  (h₁ : discriminants 1 = 1)
  (h₂ : discriminants 2 = 4)
  (h₃ : discriminants 3 = 9) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁ = (-b 1 + real.sqrt 1) / 2 ∧ y₁ = (-b 1 - real.sqrt 1) / 2) ∧
    (x₂ = (-b 2 + real.sqrt 4) / 2 ∧ y₂ = (-b 2 - real.sqrt 4) / 2) ∧
    (x₃ = (-b 3 + real.sqrt 9) / 2 ∧ y₃ = (-b 3 - real.sqrt 9) / 2) ∧
    (x₁ + x₂ + y₃ = y₁ + y₂ + x₃) :=
sorry

end choose_roots_sum_eq_l166_166129


namespace rosa_can_see_leo_time_l166_166539

theorem rosa_can_see_leo_time :
  ∀ (rosa_speed leo_speed initial_distance : ℝ), 
  rosa_speed = 15 →
  leo_speed = 5 →
  initial_distance = 3/4 →
  let relative_speed := rosa_speed - leo_speed in
  let time_to_overtake := initial_distance / relative_speed in
  let relative_distance := initial_distance in
  let total_time := time_to_overtake + relative_distance / relative_speed in
  total_time * 60 = 9 :=
begin
  intros,
  sorry,
end

end rosa_can_see_leo_time_l166_166539


namespace probability_x_plus_2y_leq_6_l166_166910

noncomputable def probability_condition (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5 ∧ x + 2*y ≤ 6

theorem probability_x_plus_2y_leq_6 :
  let probability := (8 / 20 : ℝ)
  in probability = 2 / 5 :=
  sorry

end probability_x_plus_2y_leq_6_l166_166910


namespace inequality_solution_l166_166733

theorem inequality_solution (x : ℝ) : 
    (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
    sorry

end inequality_solution_l166_166733


namespace twentieth_monomial_in_pattern_l166_166703

theorem twentieth_monomial_in_pattern : 
  (let monomial_formula (n : ℕ) : ℤ × ℕ := ((-1) ^ n * (2 * n - 1), 2 * n))
  (let twentieth_monomial := monomial_formula 20)
  (twentieth_monomial.1 * x ^ twentieth_monomial.2 = 39 * x ^ 40) :=
by
  sorry

end twentieth_monomial_in_pattern_l166_166703


namespace moles_of_HCl_needed_l166_166990

-- Define the reaction and corresponding stoichiometry
def reaction_relates (NaHSO3 HCl NaCl H2O SO2 : ℕ) : Prop :=
  NaHSO3 = HCl ∧ HCl = NaCl ∧ NaCl = H2O ∧ H2O = SO2

-- Given condition: one mole of each reactant produces one mole of each product
axiom reaction_stoichiometry : reaction_relates 1 1 1 1 1

-- Prove that 2 moles of NaHSO3 reacting with 2 moles of HCl forms 2 moles of NaCl
theorem moles_of_HCl_needed :
  ∀ (NaHSO3 HCl NaCl : ℕ), reaction_relates NaHSO3 HCl NaCl NaCl NaCl → NaCl = 2 → HCl = 2 :=
by
  intros NaHSO3 HCl NaCl h_eq h_NaCl
  sorry

end moles_of_HCl_needed_l166_166990


namespace min_abs_sum_is_5_l166_166434

noncomputable def min_abs_sum (x : ℝ) : ℝ :=
  |x + 1| + |x + 3| + |x + 6|

theorem min_abs_sum_is_5 : ∃ x : ℝ, (∀ y : ℝ, min_abs_sum y ≥ min_abs_sum x) ∧ min_abs_sum x = 5 :=
by
  use -3
  sorry

end min_abs_sum_is_5_l166_166434


namespace original_cost_price_of_car_l166_166876

theorem original_cost_price_of_car (x : ℝ) (y : ℝ) (h1 : y = 0.87 * x) (h2 : 1.20 * y = 54000) :
  x = 54000 / 1.044 :=
by
  sorry

end original_cost_price_of_car_l166_166876


namespace sum_first_15_odd_integers_l166_166812

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l166_166812


namespace tan_half_angle_second_quadrant_l166_166302

variables (θ : ℝ) (k : ℤ)
open Real

theorem tan_half_angle_second_quadrant (h : (π / 2) + 2 * k * π < θ ∧ θ < π + 2 * k * π) : 
  tan (θ / 2) > 1 := 
sorry

end tan_half_angle_second_quadrant_l166_166302


namespace three_digit_integers_count_l166_166231

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_perfect_square_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 4

def is_allowed_digit (d : ℕ) : Prop :=
  is_prime_digit d ∨ is_perfect_square_digit d

theorem three_digit_integers_count : ∃ n : ℕ, n = 216 ∧ 
  (∀ x y z : ℕ, x ≠ 0 → is_allowed_digit x → is_allowed_digit y → is_allowed_digit z → 
   (x * 100 + y * 10 + z) > 99 ∧ (x * 100 + y * 10 + z) < 1000 → 
   (nat.card {n : ℕ | n = x * 100 + y * 10 + z ∧ is_allowed_digit x ∧ is_allowed_digit y ∧ is_allowed_digit z}) = 216) :=
begin
  sorry
end

end three_digit_integers_count_l166_166231


namespace not_divisible_by_5_square_plus_or_minus_1_divisible_by_5_l166_166331

theorem not_divisible_by_5_square_plus_or_minus_1_divisible_by_5 (a : ℤ) (h : a % 5 ≠ 0) :
  (a^2 + 1) % 5 = 0 ∨ (a^2 - 1) % 5 = 0 :=
by
  sorry

end not_divisible_by_5_square_plus_or_minus_1_divisible_by_5_l166_166331


namespace check_H_functions_l166_166698

def H_function (f : ℝ → ℝ) : Prop :=
∀ x₁ x₂ : ℝ, x₁ < x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem check_H_functions :
  H_function (fun x => Real.exp x + x) ∧
  ¬ H_function (fun x => x^2) ∧
  H_function (fun x => 3 * x - Real.sin x) ∧
  ¬ H_function (fun x => if x = 0 then 0 else Real.log (Real.abs x)) :=
by {
  sorry
}

end check_H_functions_l166_166698


namespace probability_at_least_5_in_6_rolls_l166_166070

-- We define the conditions from the problem statement.
def event_A (n : ℕ) : Prop := n = 5 ∨ n = 6

def probability_event_A : ℚ := 1 / 3 
def probability_not_event_A : ℚ := 2 / 3

def binom (n k : ℕ) : ℚ := (nat.choose n k : ℚ)

-- The main theorem to prove the desired probability.
theorem probability_at_least_5_in_6_rolls : 
  (binom 6 5 * probability_event_A^5 * probability_not_event_A^1) + (probability_event_A^6) = 13 / 729 :=
  sorry

end probability_at_least_5_in_6_rolls_l166_166070


namespace fraction_division_l166_166034

theorem fraction_division : (3/4) / (5/8) = (6/5) := by
  sorry

end fraction_division_l166_166034


namespace original_number_l166_166476

theorem original_number :
  ∃ x : ℝ, 0 < x ∧ (move_decimal_point x 3 = 3 / x) ∧ x = sqrt 30 / 100 :=
sorry

noncomputable def move_decimal_point (x : ℝ) (places : ℕ) : ℝ := x * 10^places

end original_number_l166_166476


namespace find_length_PA_find_sine_dihedral_angle_l166_166279

-- Define the quadrilateral pyramid P-ABCD with PA perpendicular to base ABCD
structure Pyramid where
  P A B C D : Point
  PA_perp_base : Perpendicular PA (Plane ABCD)
  BC_eq_CD : BC = 2
  AC_eq_4 : AC = 4
  ACB_eq_ACD : Angle ACB = Angle ACD = π / 3

-- Define the midpoint F of PC and condition AF perpendicular to PB
structure MidpointF (pyramid : Pyramid) where
  F : Point
  F_midpoint_PC : F = Midpoint P C
  AF_perp_PB : Perpendicular AF PB

-- Define the theorem statements
theorem find_length_PA (pyramid : Pyramid) : Length PA = 2 * sqrt 3 := by
  sorry

theorem find_sine_dihedral_angle (
  pyramid : Pyramid,
  midpointF : MidpointF pyramid
) : sin (DihedralAngle B AF D) = 3 * sqrt 7 / 8 := by
  sorry

end find_length_PA_find_sine_dihedral_angle_l166_166279


namespace intersection_exists_l166_166714

open Set

variable {α : Type*} [LinearOrderedField α]

-- Definitions for points and collinearity
structure Point (α : Type*) :=
(x y : α)

variables (A B C P : Point α)

def collinear (A B C : Point α) : Prop :=
∃ l : {l // is_line l},
  A ∈ l ∧ B ∈ l ∧ C ∈ l

def on_line (P Q R : Point α) : Prop :=
∃ l : {l // is_line l},
  P ∈ l ∧ Q ∈ l

-- The main theorem
theorem intersection_exists 
  (h1 : ¬ collinear A B C) 
  (h2 : ¬ on_line P B C) 
  (h3 : ¬ on_line P C A) 
  (h4 : ¬ on_line P A B) :
  (exists p, between p B C ∧ collinear P A p) ∨ 
  (exists q, between q C A ∧ collinear P B q) ∨
  (exists r, between r A B ∧ collinear P C r) :=
sorry

end intersection_exists_l166_166714


namespace two_p_plus_q_l166_166245

variable {p q : ℚ}  -- Variables are rationals

theorem two_p_plus_q (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 :=
by sorry

end two_p_plus_q_l166_166245


namespace smallest_value_of_abs_sum_l166_166422

theorem smallest_value_of_abs_sum : ∃ x : ℝ, (x = -3) ∧ ( ∀ y : ℝ, |y + 1| + |y + 3| + |y + 6| ≥ 5 ) :=
by
  use -3
  split
  . exact rfl
  . intro y
    have h1 : |y + 1| + |y + 3| + |y + 6| = sorry,
    sorry

end smallest_value_of_abs_sum_l166_166422


namespace measure_50kg_sugar_l166_166050

theorem measure_50kg_sugar (bag_of_sugar : Type) [finite_bag : fintype bag_of_sugar]
    (balance_scale : bag_of_sugar → ℝ) (one_kg_weight : ℝ) (paper_bag : bag_of_sugar → ℝ) :
    (∃ s1 s2 s4 s8 s16 s32 : bag_of_sugar,
        balance_scale s1 = 1 ∧
        balance_scale s2 = 2 ∧
        balance_scale s4 = 4 ∧
        balance_scale s8 = 8 ∧
        balance_scale s16 = 16 ∧
        balance_scale s32 = 32 ∧
        balance_scale (s1 + s2 + s4 + s8 + s16 + s32) = 50)
        ∧ (∃ (weighings : ℕ), weighings ≤ 6) :=
begin
  sorry
end

end measure_50kg_sugar_l166_166050


namespace plane_divided_into_four_regions_by_lines_l166_166958

theorem plane_divided_into_four_regions_by_lines :
  (∀ x y : ℝ, (y = 3 * x ∨ x = 3 * y)) →
  ∃ n : ℕ, n = 4 :=
by
  intros x y h
  let lines := [
    {p : ℝ × ℝ // p.2 = 3 * p.1},
    {p : ℝ × ℝ // p.2 = p.1 / 3}
  ]
  have h_region_division : length lines = 2 → ∃ n, n = 4 := sorry
  exact h_region_division (length lines)

end plane_divided_into_four_regions_by_lines_l166_166958


namespace number_of_employees_l166_166267

-- Definitions based on conditions
variables (E : ℕ) -- Total number of employees
variables (H1 : 0.65 * E = 0.65 * E) -- This is somewhat redundant, kept for completeness
variables (H2 : 0.25 * (0.65 * E) = 0.25 * (0.65 * E)) -- Males at least 50 years old
variables (H3 : 0.75 * (0.65 * E) = 3120) -- Males below 50 years old = 3120

-- The theorem statement proving E = 6400
theorem number_of_employees (E : ℕ) (H1 : 0.65 * E = 0.65 * E) (H2 : 0.25 * (0.65 * E) = 0.25 * (0.65 * E)) (H3 : 0.75 * (0.65 * E) = 3120) :
  E = 6400 :=
by
  sorry

end number_of_employees_l166_166267


namespace age_of_b_l166_166487

variable (a b c d : ℕ)
variable (h1 : a = b + 2)
variable (h2 : b = 2 * c)
variable (h3 : d = b / 2)
variable (h4 : a + b + c + d = 44)

theorem age_of_b : b = 14 :=
by 
  sorry

end age_of_b_l166_166487


namespace original_number_l166_166456

theorem original_number (x : ℝ) (h : 1000 * x = 3 / x) : x = (real.sqrt 30) / 100 :=
by sorry

end original_number_l166_166456


namespace smallest_four_digit_palindrome_divisible_by_8_l166_166795

def is_palindrome (n : Nat) : Prop :=
  let s := n.repr
  s = s.reverse

theorem smallest_four_digit_palindrome_divisible_by_8 :
  ∃ (n : Nat), n = 2112 ∧ is_palindrome n ∧ n % 8 = 0 ∧ 1000 ≤ n ∧ n < 10000 ∧
  ∀ m : Nat, is_palindrome m ∧ m % 8 = 0 ∧ 1000 ≤ m ∧ m < 10000 → 2112 ≤ m :=
sorry

end smallest_four_digit_palindrome_divisible_by_8_l166_166795


namespace ED_calculation_n_calculation_d_x_relation_l166_166274

-- All the necessary declarations for given conditions
variables {A B C D E : Type} [AB_rectangle : Rectangle ABCD]
variables (n x d : ℝ)
variables (H1 : Distance E DC = n)
variables (H2 : Distance E BC = 1)
variables (H3 : Distance E AB = x)
variables (H4 : Diagonal_length AB E = d)

-- The main statements to prove
theorem ED_calculation (H1 : Distance E DC = n) (H2 : Distance E BC = 1) (H3 : Distance E AB = x) (H4 : Diagonal_length AB E = d) :
  Distance DE = x^2 * sqrt(1 + x^2) := sorry

theorem n_calculation (H1 : Distance E DC = n) (H2 : Distance E BC = 1) (H3 : Distance E AB = x) (H4 : Diagonal_length AB E = d) :
  n = x^3 := sorry

theorem d_x_relation (H1 : Distance E DC = n) (H2 : Distance E BC = 1) (H3 : Distance E AB = x) (H4 : Diagonal_length AB E = d) :
  d^(2/3) - x^(2/3) = 1 := sorry

end ED_calculation_n_calculation_d_x_relation_l166_166274


namespace fraction_division_l166_166033

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := 
by
  sorry

end fraction_division_l166_166033


namespace probability_heart_spade_queen_l166_166779

theorem probability_heart_spade_queen (h_cards : ℕ) (s_cards : ℕ) (q_cards : ℕ) (total_cards : ℕ) 
    (h_not_q : ℕ) (remaining_cards_after_2 : ℕ) (remaining_spades : ℕ) 
    (queen_remaining_after_2 : ℕ) (remaining_cards_after_1 : ℕ) :
    h_cards = 13 ∧ s_cards = 13 ∧ q_cards = 4 ∧ total_cards = 52 ∧ h_not_q = 12 ∧ remaining_cards_after_2 = 50 ∧
    remaining_spades = 13 ∧ queen_remaining_after_2 = 3 ∧ remaining_cards_after_1 = 51 →
    (h_cards / total_cards) * (remaining_spades / remaining_cards_after_1) * (q_cards / remaining_cards_after_2) + 
    (q_cards / total_cards) * (remaining_spades / remaining_cards_after_1) * (queen_remaining_after_2 / remaining_cards_after_2) = 
    221 / 44200 := by 
  sorry

end probability_heart_spade_queen_l166_166779


namespace tan_alpha_value_l166_166238

theorem tan_alpha_value (α : ℝ) (h1 : α ∈ Ioo (π / 2) π) (h2 : cos α ^ 2 - sin α = 1 / 4) :
  tan α = -real.sqrt 3 / 3 :=
sorry

end tan_alpha_value_l166_166238


namespace sum_of_first_15_odd_integers_l166_166832

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l166_166832


namespace smallest_positive_period_axis_of_symmetry_intervals_monotonically_increasing_l166_166615

def f (x : ℝ) : ℝ := sin x ^ 2 + sqrt 3 * sin x * cos x
def period (T : ℝ) : Prop := ∀ x, f(x + T) = f(x)

theorem smallest_positive_period : period π :=
sorry

theorem axis_of_symmetry (k : ℤ) : f (π / 3 + k * π / 2) = f (π / 3) :=
sorry

theorem intervals_monotonically_increasing (k : ℤ) :
  ∀ x, -π/6 + k * π ≤ x ∧ x ≤ π/3 + k * π → ∀ y, f y ≤ f (y + (π / 3 - π/6)) :=
sorry

end smallest_positive_period_axis_of_symmetry_intervals_monotonically_increasing_l166_166615


namespace asymptotic_function_example1_not_asymptotic_function_example2_l166_166161

noncomputable def f1 (x : ℝ) : ℝ := (x * x + 2 * x + 3) / (x + 1)
noncomputable def g1 (x : ℝ) : ℝ := x + 1

theorem asymptotic_function_example1 :
  (∀ x ≥ 0, f1 x - g1 x > 0 ∧ f1 x - g1 x ≤ 2) ∧
  (∀ x ≥ 0, ∀ y ≥ x + 1, f1 y - g1 y < f1 x - g1 x) ∧
  ∃ p, (∀ x ≥ 0, f1 x - g1 x > 0 ∧ f1 x - g1 x ≤ p) ∧ p = 2 :=
sorry

noncomputable def f2 (x : ℝ) := Real.sqrt (x * x + 1)
noncomputable def g2 (x : ℝ) := a * x

theorem not_asymptotic_function_example2 (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ¬ (∀ x ≥ 0, f2 x - g2 x > 0 ∧ f2 x - g2 x ≤ 2) :=
sorry

end asymptotic_function_example1_not_asymptotic_function_example2_l166_166161


namespace min_value_of_f_l166_166409

def f (x : ℝ) := abs (x + 1) + abs (x + 3) + abs (x + 6)

theorem min_value_of_f : ∃ (x : ℝ), f x = 5 :=
by
  use -3
  simp [f]
  sorry

end min_value_of_f_l166_166409


namespace p_sq_plus_q_sq_l166_166687

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 := by
  sorry

end p_sq_plus_q_sq_l166_166687


namespace hexagon_chord_length_l166_166071

theorem hexagon_chord_length (AB CD EF BC DE FA : ℝ) (p q : ℕ) (hpq_coprime : Nat.coprime p q) :
  AB = 4 → CD = 4 → EF = 4 → BC = 6 → DE = 6 → FA = 6 →
  (∀ p' q', BD = (p' : ℝ) / (q' : ℝ) → Nat.coprime p' q' → p' = 47 ∧ q' = 25) → 
  p + q = 72 :=
by
  sorry

end hexagon_chord_length_l166_166071


namespace area_of_curve_l166_166137

theorem area_of_curve : 
  let curve_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 4 * x + 6 * y - 3 = 0 in
  ∀ (x y : ℝ), curve_eq x y → ∃ (r : ℝ), r = 4 ∧ ∃ (area : ℝ), area = π * r^2 ∧ area = 16 * π :=
by
  sorry

end area_of_curve_l166_166137


namespace maximum_expression_l166_166683

theorem maximum_expression (n : ℕ) (x : Fin n → ℝ) (hx_pos : ∀ i, 0 < x i) (hx_mul : (∀ (i : Fin n → ℝ), set.prod (λ (i : Fin n), x) = 1) (S : ℝ) (hS : S = ∑ i, (x i) ^ 3) :
  ∑ i, x i / (S - (x i) ^ 3 + (x i)²) ≤ 1 :=
sorry

end maximum_expression_l166_166683


namespace williams_tips_fraction_l166_166059

theorem williams_tips_fraction
  (A : ℝ) -- average tips for months other than August
  (h : ∀ A, A > 0) -- assuming some positivity constraint for non-degenerate mean
  (h_august : A ≠ 0) -- assuming average can’t be zero
  (august_tips : ℝ := 10 * A)
  (other_months_tips : ℝ := 6 * A)
  (total_tips : ℝ := 16 * A) :
  (august_tips / total_tips) = (5 / 8) := 
sorry

end williams_tips_fraction_l166_166059


namespace probability_at_least_two_same_die_l166_166327

open Nat

-- The conditions
def eight_sided_die_outcomes := 8
def num_dice := 6

-- Summarized as a statement to be proved:
theorem probability_at_least_two_same_die :
  let total_possible_outcomes := eight_sided_die_outcomes ^ num_dice,
      favorable_outcomes_all_different := eight_sided_die_outcomes * 
                                           (eight_sided_die_outcomes - 1) * 
                                           (eight_sided_die_outcomes - 2) * 
                                           (eight_sided_die_outcomes - 3) * 
                                           (eight_sided_die_outcomes - 4) * 
                                           (eight_sided_die_outcomes - 5),
      probability_all_different := (favorable_outcomes_all_different : ℚ) / total_possible_outcomes,
      probability_at_least_two_same := 1 - probability_all_different
  in probability_at_least_two_same = 3781 / 4096 := by
  sorry

end probability_at_least_two_same_die_l166_166327


namespace complement_union_l166_166220

open Set

noncomputable def U := set.univ
noncomputable def A := {x : ℝ | x ≤ 1}
noncomputable def B := {x : ℝ | x ≥ 2}

theorem complement_union (U : set ℝ) (A B : set ℝ) (hU : U = set.univ)
  (hA : A = {x | x ≤ 1}) (hB : B = {x | x ≥ 2}) :
  compl (A ∪ B) = {x | 1 < x ∧ x < 2} :=
by {
  sorry
}

end complement_union_l166_166220


namespace find_original_number_l166_166475

theorem find_original_number (x : ℝ) (hx : 0 < x) 
  (h : 1000 * x = 3 * (1 / x)) : x ≈ 0.01732 :=
by 
  sorry

end find_original_number_l166_166475


namespace interest_after_5_years_l166_166316

noncomputable def initial_amount : ℝ := 2000
noncomputable def interest_rate : ℝ := 0.08
noncomputable def duration : ℕ := 5
noncomputable def final_amount : ℝ := initial_amount * (1 + interest_rate) ^ duration
noncomputable def interest_earned : ℝ := final_amount - initial_amount

theorem interest_after_5_years : interest_earned = 938.66 := by
  sorry

end interest_after_5_years_l166_166316


namespace board_tiling_condition_l166_166169

-- Define the problem in Lean

theorem board_tiling_condition (n : ℕ) : 
  (∃ m : ℕ, n * n = m + 4 * m) ↔ (∃ k : ℕ, n = 5 * k ∧ n > 5) := by 
sorry

end board_tiling_condition_l166_166169


namespace number_of_even_factors_l166_166644

theorem number_of_even_factors {n : ℕ} (h : n = 2^4 * 3^3 * 7) : 
  ∃ (count : ℕ), count = 32 ∧ ∀ k, (k ∣ n) → k % 2 = 0 → count = 32 :=
by
  sorry

end number_of_even_factors_l166_166644


namespace problem1_solution_problem2_solution_l166_166941

noncomputable def problem1 : ℝ :=
    2 * Real.sqrt 6 * 5 * Real.sqrt (1 / 3) / Real.sqrt 2 - (Real.abs (1 - Real.sqrt 3))

theorem problem1_solution : problem1 = 11 - Real.sqrt 3 :=
by
  sorry

noncomputable def problem2 : ℝ :=
    (Real.sqrt 7 + 2 * Real.sqrt 2) * (Real.sqrt 7 - 2 * Real.sqrt 2) - Real.sqrt 20 + Real.sqrt 5 * (Real.sqrt 5 - 2)

theorem problem2_solution : problem2 = 4 - 4 * Real.sqrt 5 :=
by
  sorry

end problem1_solution_problem2_solution_l166_166941


namespace range_area_OACB_l166_166605

theorem range_area_OACB
  (O : ℝ × ℝ)
  (A B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (hO : O = (0, 0))
  (hA : A.1^2 / 4 + A.2^2 / 3 = 1)
  (hB : B.1^2 / 4 + B.2^2 / 3 = 1)
  (hAB : real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 / 2)
  (hC : ∃ tA tB : ℝ, (4 * A.1 * tA + 3 * A.2 = 4) ∧ (4 * B.1 * tB + 3 * B.2 = 4)) :
  ∃ S : ℝ, (S = area_quadrilateral O A C B) ∧ 
  (S ∈ set.Icc (6 * real.sqrt 165 / 55) (6 * real.sqrt 13 / 13)) :=
sorry

end range_area_OACB_l166_166605


namespace problem_statement_l166_166770

-- Define the arithmetic sequence {a_n} with a_1 = 3 and common difference d
def an_arith (n : ℕ) : ℕ := 2 * n + 1

-- Define the geometric sequence {b_n} with b_1 = 1 and common ratio q = 8
def bn_geom (n : ℕ) : ℕ := 8 ^ (n - 1)

-- Define the sum of the first n terms of the arithmetic sequence S_n
def Sn (n : ℕ) : ℕ := n * (n + 2)

-- The sum we want to compute
def Sum_of_reciprocals (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, 1 / (Sn (k + 1) : ℚ))

theorem problem_statement (n : ℕ) : 
  Sum_of_reciprocals n = (3 / 4) - ((2 * n + 3 : ℚ) / (2 * (n + 1) * (n + 2))) := sorry

end problem_statement_l166_166770


namespace sum_first_15_odd_integers_l166_166843

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l166_166843


namespace elmer_will_save_48_percent_l166_166975

def elmer_saving (efficiency_old : ℝ) (cost_old : ℝ) (distance_old : ℝ)
  (efficiency_new : ℝ := 1.75 * efficiency_old)
  (cost_new_per_liter : ℝ := 1.3 * cost_old)
  (distance_new : ℝ := 1.5 * distance_old) : Prop :=
  let cost_old_trip := distance_new / efficiency_old * cost_old in
  let cost_new_trip := distance_new / efficiency_new * cost_new_per_liter in
  cost_new_trip = cost_old_trip * 0.52

theorem elmer_will_save_48_percent :
  ∀ (x c : ℝ), x > 0 → c > 0 →
  elmer_saving x c x :=
by
  intros x c hx hc
  sorry

end elmer_will_save_48_percent_l166_166975


namespace hyperbola_focus_l166_166350

theorem hyperbola_focus :
  (∃ (c : ℝ), (∀ x y : ℝ, x^2 - 2 * y^2 = 1 
  → (x = (c) ∨ x = (-c)) ∧ y = 0) := by
sorry

end hyperbola_focus_l166_166350


namespace shuffle_problem_l166_166157

-- Define the essential elements
variable (c n : ℕ) -- total number of cards and distinct names
variable (S : Fin c → Set (Fin n)) -- card names for each card
variable (v : Fin c → Int) -- stack positions: +1 for left, -1 for right

-- Condition: more cards in the left stack initially
axiom initial_condition : ∑ i, v i > 0

-- Define the difference function D(E) after shuffling by names in E
def D (E : Set (Fin n)) : Int :=
  ∑ i, (-1)^(E ∩ S i).card * v i

-- Formal statement of the problem
theorem shuffle_problem : ∃ E : Set (Fin n), D S E < 0 :=
by
  sorry

end shuffle_problem_l166_166157


namespace min_ab_eq_4_l166_166252

theorem min_ab_eq_4 (a b : ℝ) (h : 4 / a + 1 / b = Real.sqrt (a * b)) : a * b ≥ 4 :=
sorry

end min_ab_eq_4_l166_166252


namespace sum_first_15_odd_integers_l166_166814

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l166_166814


namespace original_number_solution_l166_166444

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end original_number_solution_l166_166444


namespace correct_propositions_l166_166224

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Assume basic predicates for lines and planes
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (planar_parallel : Plane → Plane → Prop)

-- Stating the theorem to be proved
theorem correct_propositions :
  (parallel m n ∧ perp m α → perp n α) ∧ 
  (planar_parallel α β ∧ parallel m n ∧ perp m α → perp n β) :=
by
  sorry

end correct_propositions_l166_166224


namespace find_n_l166_166437

theorem find_n
  (n : ℤ)
  (h : n + (n + 1) + (n + 2) + (n + 3) = 30) :
  n = 6 :=
by
  sorry

end find_n_l166_166437


namespace Kato_finishes_first_l166_166325

-- Define constants and variables from the problem conditions
def Kato_total_pages : ℕ := 10
def Kato_lines_per_page : ℕ := 20
def Gizi_lines_per_page : ℕ := 30
def conversion_ratio : ℚ := 3 / 4
def initial_pages_written_by_Kato : ℕ := 4
def initial_additional_lines_by_Kato : ℚ := 2.5
def Kato_to_Gizi_writing_ratio : ℚ := 3 / 4

-- Calculate total lines in Kato's manuscript
def Kato_total_lines : ℕ := Kato_total_pages * Kato_lines_per_page

-- Convert Kato's lines to Gizi's format
def Kato_lines_in_Gizi_format : ℚ := Kato_total_lines * conversion_ratio

-- Calculate total pages Gizi needs to type
def Gizi_total_pages : ℚ := Kato_lines_in_Gizi_format / Gizi_lines_per_page

-- Calculate initial lines by Kato before Gizi starts typing
def initial_lines_by_Kato : ℚ := initial_pages_written_by_Kato * Kato_lines_per_page + initial_additional_lines_by_Kato

-- Lines Kato writes for every page Gizi types including setup time consideration
def additional_lines_by_Kato_per_Gizi_page : ℚ := Gizi_lines_per_page * Kato_to_Gizi_writing_ratio + initial_additional_lines_by_Kato / Gizi_total_pages

-- Calculate total lines Kato writes while Gizi finishes 5 pages
def final_lines_by_Kato : ℚ := additional_lines_by_Kato_per_Gizi_page * Gizi_total_pages

-- Remaining lines after initial setup for Kato
def remaining_lines_by_Kato_after_initial : ℚ := Kato_total_lines - initial_lines_by_Kato

-- Final proof statement
theorem Kato_finishes_first : final_lines_by_Kato ≥ remaining_lines_by_Kato_after_initial :=
by sorry

end Kato_finishes_first_l166_166325


namespace problem1_l166_166944

theorem problem1 (x : ℝ) : (2 * x - 1) * (2 * x - 3) - (1 - 2 * x) * (2 - x) = 2 * x^2 - 3 * x + 1 :=
by
  sorry

end problem1_l166_166944


namespace sum_of_first_15_odd_positives_l166_166799

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l166_166799


namespace janice_work_days_l166_166669

variable (dailyEarnings : Nat)
variable (overtimeEarnings : Nat)
variable (numOvertimeShifts : Nat)
variable (totalEarnings : Nat)

theorem janice_work_days
    (h1 : dailyEarnings = 30)
    (h2 : overtimeEarnings = 15)
    (h3 : numOvertimeShifts = 3)
    (h4 : totalEarnings = 195)
    : let overtimeTotal := numOvertimeShifts * overtimeEarnings
      let regularEarnings := totalEarnings - overtimeTotal
      let workDays := regularEarnings / dailyEarnings
      workDays = 5 :=
by
  sorry

end janice_work_days_l166_166669


namespace lcm_eq_792_l166_166582

-- Define the integers
def a : ℕ := 8
def b : ℕ := 9
def c : ℕ := 11

-- Define their prime factorizations (included for clarity, though not directly necessary)
def a_factorization : a = 2^3 := rfl
def b_factorization : b = 3^2 := rfl
def c_factorization : c = 11 := rfl

-- Define the LCM function
def lcm_abc := Nat.lcm (Nat.lcm a b) c

-- Prove that lcm of a, b, c is 792
theorem lcm_eq_792 : lcm_abc = 792 := 
by
  -- Include the necessary properties of LCM and prime factorizations if necessary
  sorry

end lcm_eq_792_l166_166582


namespace probability_of_triple_intersection_l166_166780

noncomputable def probability_intersection (A_X B_X C_X : ℝ) : ℝ :=
if (0 <= A_X ∧ A_X <= 4) ∧ (0 <= B_X ∧ B_X <= 4) ∧ (0 <= C_X ∧ C_X <= 4) ∧
   (A_X - B_X)^2 ≤ 12 ∧ (A_X - C_X)^2 ≤ 12 then 0.375 else 0

theorem probability_of_triple_intersection :
  (∫ (A_X : ℝ) in 0..4, ∫ (B_X : ℝ) in 0..4, ∫ (C_X : ℝ) in 0..4, probability_intersection A_X B_X C_X) = 0.375 := sorry

end probability_of_triple_intersection_l166_166780


namespace proj_w_v_eq_v_l166_166160

-- Define the vectors v and w in Lean
def v : ℝ × ℝ := (4, -2)
def w : ℝ × ℝ := (6, -3)

-- Define a function for projection of v onto w
def proj (v w : ℝ × ℝ) : ℝ × ℝ := 
  let dot_vw := v.1 * w.1 + v.2 * w.2 -- Dot product v ⬝ w
  let dot_ww := w.1 * w.1 + w.2 * w.2 -- Dot product w ⬝ w
  let scalar := dot_vw / dot_ww
  (scalar * w.1, scalar * w.2)

-- The theorem to prove that proj_w(v) = v
theorem proj_w_v_eq_v : proj v w = v :=
by
  sorry -- Proof goes here

end proj_w_v_eq_v_l166_166160


namespace product_of_numbers_l166_166774

open Real

theorem product_of_numbers (numbers : Finset ℝ) (h_card : numbers.card = 2015)
  (h_distinct : ∀ (x y : ℝ), x ∈ numbers → y ∈ numbers → x ≠ y)
  (h_parity : ∀ (a : ℝ), 0 < a → (numbers.filter (λ x, x < 2014 / a)).card % 2 = (numbers.filter (λ x, x > a)).card % 2)
  : ∏ x in numbers, x = 2014^1007 * sqrt 2014 := 
sorry

end product_of_numbers_l166_166774


namespace gallons_needed_to_grandmas_house_l166_166702

def car_fuel_efficiency : ℝ := 20
def distance_to_grandmas_house : ℝ := 100

theorem gallons_needed_to_grandmas_house : (distance_to_grandmas_house / car_fuel_efficiency) = 5 :=
by
  sorry

end gallons_needed_to_grandmas_house_l166_166702


namespace sin_X_eq_8_over_9_l166_166287

theorem sin_X_eq_8_over_9
  (XYZ : Triangle)
  (area_XYZ : XYZ.area = 100)
  (geo_mean_XY_XZ : real.sqrt (XYZ.s1 * (XYZ.s1 / 2)) = 15)
  (XY_twice_XZ : XYZ.s1 = 2 * (XYZ.s1 / 2)) :
  real.sin XYZ.angle_X = 8/9 := 
sorry

end sin_X_eq_8_over_9_l166_166287


namespace max_fraction_eq_one_l166_166690

theorem max_fraction_eq_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ M, M = 1 ∧ ∀ x y : ℝ, (x = a) ∧ (y = b) → (a + b)^2 / (a^2 + 2*a*b + b^2) ≤ M :=
by {
  let expr := (a + b)^2 / (a^2 + 2*a*b + b^2),
  use 1,
  split,
  { refl },
  {
    intros x y xy,
    cases xy,
    rw [←xy_left, ←xy_right],
    have h : (a + b)^2 = a^2 + 2*a*b + b^2 := by ring,
    rw h,
    exact le_of_eq (by ring),
  }
}

end max_fraction_eq_one_l166_166690


namespace wood_stove_afternoon_burn_rate_l166_166528

-- Conditions extracted as definitions
def morning_burn_rate : ℝ := 2
def morning_duration : ℝ := 4
def initial_wood : ℝ := 30
def final_wood : ℝ := 3
def afternoon_duration : ℝ := 4

-- Theorem statement matching the conditions and correct answer
theorem wood_stove_afternoon_burn_rate :
  let morning_burned := morning_burn_rate * morning_duration
  let total_burned := initial_wood - final_wood
  let afternoon_burned := total_burned - morning_burned
  ∃ R : ℝ, (afternoon_burned = R * afternoon_duration) ∧ (R = 4.75) :=
by
  sorry

end wood_stove_afternoon_burn_rate_l166_166528


namespace problem_part_1_problem_part_2_l166_166227

noncomputable def a : ℝ := 1
noncomputable def m : ℝ := (sqrt 3 / 2) + 1

def vec_a (ax : ℝ) : ℝ × ℝ := (Real.cos ax, Real.sin ax)
def vec_b (ax : ℝ) : ℝ × ℝ := (sqrt 3 * Real.cos ax, -Real.cos ax)

def f (ax : ℝ) (x : ℝ) : ℝ := 
  let (ax₁, ax₂) := vec_a (ax * x)
  let (bx₁, bx₂) := vec_b (ax * x)
  ax₁ * bx₁ + ax₂ * bx₂

theorem problem_part_1 (ax : ℝ) (hx : ax > 0) (hm : f ax 1 = m) : 
  ax = a ∧ hm = m :=
sorry

def S (b c : ℝ) : ℝ := (sqrt 3 / 4) * b * c

theorem problem_part_2 (b c : ℝ) (hb : b = 4 ∧ c = 4) : 
  S b c = 4 * sqrt 3 :=
sorry

end problem_part_1_problem_part_2_l166_166227


namespace positive_value_of_n_l166_166588

theorem positive_value_of_n (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 25 = 0 ∧ ∃! x : ℝ, 4 * x^2 + n * x + 25 = 0) →
  n = 20 :=
by
  sorry

end positive_value_of_n_l166_166588


namespace problem_statement_l166_166726

noncomputable def original_expression (x : ℕ) : ℚ :=
(1 - 1 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1))

theorem problem_statement (x : ℕ) (hx1 : 3 - x ≥ 0) (hx2 : x ≠ 2) (hx3 : x ≠ 1) :
  original_expression 3 = 1 :=
by
  sorry

end problem_statement_l166_166726


namespace sum_first_15_odd_integers_l166_166813

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l166_166813


namespace sum_first_n_terms_eq_l166_166661

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {c : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
axiom h1 : a 1 = 1 / 4
axiom h2 : ∀ (n : ℕ), a (n + 1) = (1 / 4) * a n
axiom h3 : ∀ (n : ℕ), n > 0 → b n + 2 = 3 * (Real.log (a n) / Real.log (1 / 4))

-- Definitions
def gen_a (n : ℕ) : ℝ := (1 / 4) ^ n
def gen_b (n : ℕ) : ℝ := 3 * n - 2
def c_n (n : ℕ) : ℝ := gen_a n * gen_b n
def S_n (n : ℕ) : ℝ := (finset.range n).sum (λ i, c_n (i + 1))

-- Proof statement
theorem sum_first_n_terms_eq : ∀ n, S_n n = (2 / 3) - ((3 * n + 2) / 3) * (1 / 4) ^ n :=
by
  sorry

end sum_first_n_terms_eq_l166_166661


namespace three_pairs_exist_l166_166916

theorem three_pairs_exist :
  ∃! S P : ℕ, 5 * S + 7 * P = 90 :=
by
  sorry

end three_pairs_exist_l166_166916


namespace shortest_distance_correct_l166_166884

noncomputable def shortest_distance_a_to_c1 (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2 + 2 * b * c)

theorem shortest_distance_correct (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) :
  shortest_distance_a_to_c1 a b c h₁ h₂ = Real.sqrt (a^2 + b^2 + c^2 + 2 * b * c) :=
by
  -- This is where the proof would go.
  sorry

end shortest_distance_correct_l166_166884


namespace base_number_is_two_l166_166242

theorem base_number_is_two (a : ℝ) (x : ℕ) (h1 : x = 14) (h2 : a^x - a^(x - 2) = 3 * a^12) : a = 2 := by
  sorry

end base_number_is_two_l166_166242


namespace max_parallelogram_area_compare_parallelogram_areas_cauchy_schwarz_inequality_l166_166495

-- Part (a)
theorem max_parallelogram_area (b h : ℝ) (parallelogram : Bool) :
  parallelogram = true → b > 0 → h > 0 → 
  ∃ (max_area : ℝ), max_area = b * h := 
sorry

-- Part (b)
theorem compare_parallelogram_areas (b h : ℝ) (general_parallelogram rectangle : Bool) :
  general_parallelogram = true → rectangle = true → b > 0 → h > 0 → 
  ∃ (area_gen area_rec : ℝ), area_rec ≥ area_gen := 
sorry

-- Part (c)
theorem cauchy_schwarz_inequality (a b x y : ℝ) :
  a > 0 → b > 0 → x > 0 → y > 0 → 
  ax + by ≤ sqrt (a^2 + b^2) * sqrt (x^2 + y^2) := 
sorry

end max_parallelogram_area_compare_parallelogram_areas_cauchy_schwarz_inequality_l166_166495


namespace find_A_from_equation_and_conditions_l166_166643

theorem find_A_from_equation_and_conditions 
  (A B C D : ℕ)
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
  (h4 : B ≠ C) (h5 : B ≠ D)
  (h6 : C ≠ D)
  (h7 : 10 * A + B ≠ 0)
  (h8 : 10 * 10 * 10 * A + 10 * 10 * B + 8 * 10 + 2 - (900 + C * 10 + 9) = 490 + 3 * 10 + D) :
  A = 5 :=
by
  sorry

end find_A_from_equation_and_conditions_l166_166643


namespace second_carpenter_days_l166_166875

theorem second_carpenter_days (x : ℚ) (h1 : 1 / 5 + 1 / x = 1 / 2) : x = 10 / 3 :=
by
  sorry

end second_carpenter_days_l166_166875


namespace sequence_contains_integer_term_l166_166135

theorem sequence_contains_integer_term (M : ℕ) (hM : M > 1) :
  ∃ n, ∃ (a : ℚ), (a_0 = M + (1 / 2) ∧ (∀ k ≥ 0, a_{k+1} = a_k * floor(a_k))) ∧ 
  (a_n ∈ ℤ) := sorry

end sequence_contains_integer_term_l166_166135


namespace bobby_weekly_salary_l166_166104

variable (S : ℝ)
variables (federal_tax : ℝ) (state_tax : ℝ) (health_insurance : ℝ) (life_insurance : ℝ) (city_fee : ℝ) (net_paycheck : ℝ)

def bobby_salary_equation := 
  S - (federal_tax * S) - (state_tax * S) - health_insurance - life_insurance - city_fee = net_paycheck

theorem bobby_weekly_salary 
  (S : ℝ) 
  (federal_tax : ℝ := 1/3) 
  (state_tax : ℝ := 0.08) 
  (health_insurance : ℝ := 50) 
  (life_insurance : ℝ := 20) 
  (city_fee : ℝ := 10) 
  (net_paycheck : ℝ := 184) 
  (valid_solution : bobby_salary_equation S (1/3) 0.08 50 20 10 184) : 
  S = 450.03 := 
  sorry

end bobby_weekly_salary_l166_166104


namespace sum_first_15_odd_integers_l166_166849

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l166_166849


namespace num_distinct_points_common_to_graphs_l166_166565

def System1 (x y : ℝ) : Prop := (x + y = 7) ∧ (x - y = 3)
def System2 (x y : ℝ) : Prop := (x + y = 7) ∧ (3x + 2y = 15)
def System3 (x y : ℝ) : Prop := (2x - 3y = -7) ∧ (x - y = 3)
def System4 (x y : ℝ) : Prop := (2x - 3y = -7) ∧ (3x + 2y = 15)

def common_points (x y : ℝ) : Prop :=
  ((x + y - 7) * (2x - 3y + 7) = 0) ∧ ((x - y + 3) * (3x + 2y - 15) = 0)

def num_distinct_points (n : ℕ) : Prop :=
  ∃ points : (set (ℝ × ℝ)), (∀ p ∈ points, common_points p.1 p.2) ∧ points.finite ∧ points.card = n

theorem num_distinct_points_common_to_graphs : num_distinct_points 3 :=
  sorry

end num_distinct_points_common_to_graphs_l166_166565


namespace billy_final_lap_equals_150_l166_166929

variables {first_5_laps_time : ℕ} {next_3_laps_time : ℕ} {next_1_lap_time : ℕ}
variables {margaret_total_time : ℕ} {billy_margin : ℕ}

def billy_first_5_laps : ℕ := first_5_laps_time * 60
def billy_next_3_laps : ℕ := next_3_laps_time * 60
def billy_next_lap : ℕ := next_1_lap_time * 60
def billy_total_time : ℕ := margaret_total_time - billy_margin
def billy_first_9_laps : ℕ := billy_first_5_laps + billy_next_3_laps + billy_next_lap
def billy_final_lap_time : ℕ := billy_total_time - billy_first_9_laps

theorem billy_final_lap_equals_150 :
  first_5_laps_time = 2 ∧
  next_3_laps_time = 4 ∧
  next_1_lap_time = 1 ∧
  margaret_total_time = 10 * 60 ∧
  billy_margin = 30
  → billy_final_lap_time = 150 := 
by {
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h_rest,
  cases h_rest with h4 h5,
  simp [billy_first_5_laps, billy_next_3_laps, billy_next_lap, billy_total_time, billy_first_9_laps, h1, h2, h3, h4, h5],
  sorry
}

end billy_final_lap_equals_150_l166_166929


namespace min_area_circle_equation_l166_166607

theorem min_area_circle_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 3 / (2 + x) + 3 / (2 + y) = 1) : (x - 4)^2 + (y - 4)^2 = 256 :=
sorry

end min_area_circle_equation_l166_166607


namespace tan_sub_pi_div_four_eq_neg_seven_f_range_l166_166633

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 4)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

-- Proof for the first part
theorem tan_sub_pi_div_four_eq_neg_seven (x : ℝ) (h : 3 / 4 * Real.cos x + Real.sin x = 0) :
  Real.tan (x - Real.pi / 4) = -7 := sorry

noncomputable def f (x : ℝ) : ℝ := 
  2 * ((a x).fst + (b x).fst) * (b x).fst + 2 * ((a x).snd + (b x).snd) * (b x).snd

-- Proof for the second part
theorem f_range (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  1 / 2 < f x ∧ f x < 3 / 2 + Real.sqrt 2 := sorry

end tan_sub_pi_div_four_eq_neg_seven_f_range_l166_166633


namespace solve_for_x_l166_166439

-- Let x be a non-zero real number
variable {x : ℝ} (hx : x ≠ 0)

-- Define the equation given in the problem
def equation := (6 * x)^5 = (12 * x)^4

-- Prove that x = 8 / 3 satisfies the equation
theorem solve_for_x (hx : x ≠ 0) (eq : equation x) : x = 8 / 3 :=
sorry

end solve_for_x_l166_166439


namespace volume_of_pyramid_l166_166974

-- Definition of the base triangle sides
def a : ℝ := 13
def b : ℝ := 14
def c : ℝ := 15

-- Calculation of the semi-perimeter
def s : ℝ := (a + b + c) / 2

-- Calculation of the area of the base triangle using Heron's formula
def area_base_triangle : ℝ := real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Calculation of the circumradius of the base triangle
def circumradius : ℝ := (a * b * c) / (4 * area_base_triangle)

-- Length of the lateral edge
def lateral_edge : ℝ := 269 / 32

-- Height from apex to the base (using Pythagorean theorem)
def height := real.sqrt (lateral_edge^2 - circumradius^2)

-- Formula for the volume of the pyramid
def volume_pyramid : ℝ := (1 / 3) * area_base_triangle * height

-- The theorem stating that the volume of the pyramid is as required
theorem volume_of_pyramid : volume_pyramid = 483 / 8 :=
by sorry

end volume_of_pyramid_l166_166974


namespace original_number_solution_l166_166443

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end original_number_solution_l166_166443


namespace sum_of_first_15_odd_positives_l166_166806

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l166_166806


namespace AxB_empty_l166_166298

noncomputable theory

open Set

-- Define sets A and B
def A : Set ℝ := { x | ∃ y, y = 2^x ∧ x > 0 }
def B : Set ℝ := { y | ∃ x, y = 2^x ∧ x > 0 }

-- Define the custom operation A × B
def AxB (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∪ B ∧ x ∉ A ∩ B }

-- Theorem statement asserting A × B = ∅
theorem AxB_empty : AxB A B = ∅ :=
by
  -- Proof goes here
  sorry

end AxB_empty_l166_166298


namespace correct_propositions_l166_166222

variables (m n l : Line) (α β : Plane)

def proposition_1 (h1 : m.parallel n) (h2 : n.is_in α) : Prop := 
  m.parallel α

def proposition_2 (h1 : l.perpendicular α) (h2 : m.perpendicular β) (h3 : l.parallel m) : Prop := 
  α.parallel β

def proposition_3 (h1 : m.is_in α) (h2 : n.is_in α) (h3 : m.parallel β) (h4 : n.parallel β) : Prop := 
  α.parallel β

def proposition_4 (h1 : α.perpendicular β) (h2 : α.intersection β = m) (h3 : n.is_in β) (h4 : n.perpendicular m) : Prop := 
  n.perpendicular α

theorem correct_propositions : proposition_2 m n l α β ∧ proposition_4 m n l α β :=
by sorry

end correct_propositions_l166_166222


namespace checkered_board_cut_l166_166167

def can_cut_equal_squares (n : ℕ) : Prop :=
  n % 5 = 0 ∧ n > 5

theorem checkered_board_cut (n : ℕ) (h : n % 5 = 0 ∧ n > 5) :
  ∃ m, n^2 = 5 * m :=
by
  sorry

end checkered_board_cut_l166_166167


namespace shaded_region_area_l166_166152

def area_of_square (side : ℕ) : ℕ := side * side

def area_of_triangle (base height : ℕ) : ℕ := (base * height) / 2

def combined_area_of_triangles (base height : ℕ) : ℕ := 2 * area_of_triangle base height

def shaded_area (square_side : ℕ) (triangle_base triangle_height : ℕ) : ℕ :=
  area_of_square square_side - combined_area_of_triangles triangle_base triangle_height

theorem shaded_region_area (h₁ : area_of_square 40 = 1600)
                          (h₂ : area_of_triangle 30 30 = 450)
                          (h₃ : combined_area_of_triangles 30 30 = 900) :
  shaded_area 40 30 30 = 700 :=
by
  sorry

end shaded_region_area_l166_166152


namespace sum_of_first_15_odd_integers_l166_166852

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l166_166852


namespace infinite_sum_eq_one_sixth_l166_166935

theorem infinite_sum_eq_one_sixth :
  ∑' n : ℕ, (n ^ 3 - n) / (n + 3)! = 1 / 6 :=
sorry

end infinite_sum_eq_one_sixth_l166_166935


namespace length_PQ_l166_166280

def line_intersects_plane (a p : ℝ × ℝ × ℝ) (plane_point_1 plane_point_2 plane_point_3 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
-- Placeholder function for determining intersection point
sorry

noncomputable def length (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

def P : ℝ × ℝ × ℝ := (3 / 2, 1, 1)
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (3, 0, 0)
def D : ℝ × ℝ × ℝ := (0, 2, 0)
def E : ℝ × ℝ × ℝ := (0, 0, 1)

def plane_BDE := (B, D, E)

def Q : ℝ × ℝ × ℝ := line_intersects_plane A P B D E

theorem length_PQ : length P Q = real.sqrt 17 / 4 :=
sorry

end length_PQ_l166_166280


namespace coins_arrangement_l166_166723

-- Define the initial conditions and the problem statement
def stacks_valid (coins : List ℕ) : Prop :=
  coins.length = 10 ∧
  coins.count (λ c, c = 0) = 5 ∧  -- 5 gold coins represented by 0
  coins.count (λ c, c = 1) = 5 ∧  -- 5 silver coins represented by 1
  ∀ i, i < coins.length - 2 →
    (coins.nth_le i (by linarith) ≠ coins.nth_le (i+1) (by linarith) ∨
     coins.nth_le (i+1) (by linarith) ≠ coins.nth_le (i+2) (by linarith))

def count_stacks (n : ℕ) : ℕ :=
  if n = 10 then 69048 else 0  -- Only compute for n = 10

theorem coins_arrangement :
  ∃ (l : List ℕ), stacks_valid l ∧ count_stacks 10 = 69048 := 
sorry

end coins_arrangement_l166_166723


namespace round_to_nearest_hundredth_l166_166724

theorem round_to_nearest_hundredth (x : ℝ) (hx : x = 48.26459) : Real.round (x * 100) / 100 = 48.26 :=
by
  have h1 : (48.26459 * 100).round = 4826 := by sorry
  have h2 : 4826 / 100 = 48.26 := by norm_num
  rw [hx, h1, h2]

end round_to_nearest_hundredth_l166_166724


namespace distance_between_planes_correct_l166_166548

noncomputable def distance_between_planes
  {Point : Type*} [metric_space Point]
  (plane1 plane2 : Point → ℝ)
  [is_plane1 : linear_map ℝ Point ℝ]
  [is_plane2 : linear_map ℝ Point ℝ]
  : ℝ :=
sorry

/- Define the first plane -/
def plane1 (x y z : ℝ) : ℝ := 3 * x + 4 * y - z - 12

/- Define the second plane -/
def plane2 (x y z : ℝ) : ℝ := 6 * x + 8 * y - 2 * z - 18

/- Prove that the distance between the planes is the given value -/
theorem distance_between_planes_correct :
  distance_between_planes plane1 plane2 = (3 * real.sqrt 26) / 26 :=
sorry

end distance_between_planes_correct_l166_166548


namespace multiples_of_6_or_9_but_not_both_l166_166229

theorem multiples_of_6_or_9_but_not_both (n : ℕ) (h : n < 150) : 
  (∃ m : ℕ, m ∣ 6 ∧ m ∣ n ∧ ¬(m ∣ n ∧ m ∣ 9) ⟩ ∨ (m ∣ 9 ∧ m ∣ n ∧ ¬(m ∣ n ∧ m ∣ 6))) := sorry

end multiples_of_6_or_9_but_not_both_l166_166229


namespace min_value_of_f_l166_166411

def f (x : ℝ) := abs (x + 1) + abs (x + 3) + abs (x + 6)

theorem min_value_of_f : ∃ (x : ℝ), f x = 5 :=
by
  use -3
  simp [f]
  sorry

end min_value_of_f_l166_166411


namespace min_abs_sum_l166_166415

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem min_abs_sum :
  ∃ (x : ℝ), (abs (x + 1) + abs (x + 3) + abs (x + 6)) = 7 :=
by {
  sorry
}

end min_abs_sum_l166_166415


namespace hexagon_to_equilateral_l166_166499

theorem hexagon_to_equilateral (O A B C D E F M N L : Point)
  (hexagon_cond : inside_hexagon O [A, B, C, D, E, F])
  (triangleAOB_cond : is_equilateral_triangle O A B)
  (triangleCOD_cond : is_equilateral_triangle O C D)
  (triangleEOF_cond : is_equilateral_triangle O E F)
  (midpointM_cond : is_midpoint M B C)
  (midpointN_cond : is_midpoint N D E)
  (midpointL_cond : is_midpoint L F A) :
  is_equilateral_triangle M N L := 
sorry

end hexagon_to_equilateral_l166_166499


namespace sin_cos_ratio_l166_166590

theorem sin_cos_ratio (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2)
  (h2 : Real.tan (α - β) = 3) : 
  Real.sin (2 * α) / Real.cos (2 * β) = (Real.sqrt 5 + 3 * Real.sqrt 2) / 20 := 
by
  sorry

end sin_cos_ratio_l166_166590


namespace sum_of_first_15_odd_integers_l166_166857

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l166_166857


namespace count_convex_numbers_eq_240_l166_166185

-- Definition of a convex number
def is_convex_number (a1 a2 a3 : ℕ) : Prop := a1 < a2 ∧ a2 > a3

-- Statement of the problem
theorem count_convex_numbers_eq_240 : 
  ∑ a2 in (finset.range 10).filter (λ a2, 2 ≤ a2 ∧ a2 ≤ 9), 
    ∑ a1 in (finset.range a2), 
      ∑ a3 in (finset.Ico 0 a2), 
        if is_convex_number a1 a2 a3 then 1 else 0 = 240 := 
sorry

end count_convex_numbers_eq_240_l166_166185


namespace number_of_ways_to_blacken_l166_166542

-- Define the grid as a 5x5 structure
def Grid := Fin 5 × Fin 5

-- Conditions: three blackened squares
variable (B : Finset Grid)
variable (hB : B.card = 3)

-- Predicate defining valid placement of 1x3 rectangle within the grid
def valid_placement (rect : Finset Grid) : Prop :=
  rect.card = 3 ∧
  ∀ (b ∈ B) (r ∈ rect), ∀ (adjacent : Grid), (adjacent.1 = b.1 ∨ adjacent.2 = b.2 ∨ (adjacent.1 - b.1).natAbs = (adjacent.2 - b.2).natAbs) → adjacent ≠ r

-- Question: Number of valid placements of such rectangles
def count_valid_placements : Nat :=
  Finset.filter (valid_placement B) (Finset.powerSetOfSize (Finset.univ : Finset Grid) 3).card

-- Theorem to proof the final result
theorem number_of_ways_to_blacken :
  count_valid_placements B = 8 :=
sorry

end number_of_ways_to_blacken_l166_166542


namespace inequality_solution_l166_166735

theorem inequality_solution (x : ℝ) : (2 * x - 1) / 3 ≥ 1 → x ≥ 2 := by
  sorry

end inequality_solution_l166_166735


namespace probability_green_ball_l166_166131

theorem probability_green_ball :
  let prob_container_A := 1 / 3,
      prob_container_B := 1 / 3,
      prob_container_C := 1 / 3,
      prob_green_A := 5 / 10,
      prob_green_B := 3 / 6,
      prob_green_C := 3 / 6 in
  (prob_container_A * prob_green_A) + 
  (prob_container_B * prob_green_B) + 
  (prob_container_C * prob_green_C) = 1 / 2 :=
by
  let prob_container_A := 1 / 3,
      prob_container_B := 1 / 3,
      prob_container_C := 1 / 3,
      prob_green_A := 5 / 10,
      prob_green_B := 3 / 6,
      prob_green_C := 3 / 6
  sorry

end probability_green_ball_l166_166131


namespace max_min_xy_l166_166580

/-- Given the condition \(x + y = \sqrt{2x - 1} + \sqrt{4y + 3}\), 
    the maximum and minimum values of \(x + y\) are \(3 + \sqrt{\frac{21}{2}}\)
    and \(1 + \sqrt{\frac{3}{2}}\) respectively. -/
theorem max_min_xy (x y : ℝ) (h : x + y = real.sqrt (2 * x - 1) + real.sqrt (4 * y + 3)) :
  1 + real.sqrt (3 / 2) ≤ x + y ∧ x + y ≤ 3 + real.sqrt (21 / 2) :=
by {
  sorry
}

end max_min_xy_l166_166580


namespace solve_inequality_l166_166340

open Real

noncomputable def expression (x : ℝ) : ℝ :=
  (sqrt (x^2 - 4*x + 3) + 1) * log x / (log 2 * 5) + (1 / x) * (sqrt (8 * x - 2 * x^2 - 6) + 1)

theorem solve_inequality :
  ∃ x : ℝ, x = 1 ∧
    (x > 0) ∧
    (x^2 - 4 * x + 3 ≥ 0) ∧
    (8 * x - 2 * x^2 - 6 ≥ 0) ∧
    expression x ≤ 0 :=
by
  sorry

end solve_inequality_l166_166340


namespace sum_first_15_odd_integers_l166_166817

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l166_166817


namespace range_of_a_l166_166619

theorem range_of_a :
  ∀ (a : ℝ), (∀ (x1 : ℝ) (x2 : ℝ), 0 < x1 → f x1 a ≤ g x2) ↔ a ∈ Set.Icc (-1 : ℝ) (0 : ℝ) :=
by
  let f (x : ℝ) (a : ℝ) := a * x ^ 2 - (2 * a + 1) * x + Real.log x
  let g (x : ℝ) := Real.exp x - x - 1
  sorry

end range_of_a_l166_166619


namespace min_bottles_needed_l166_166119

theorem min_bottles_needed (fluid_ounces_needed : ℝ) (bottle_size_ml : ℝ) (conversion_factor : ℝ) :
  fluid_ounces_needed = 60 ∧ bottle_size_ml = 250 ∧ conversion_factor = 33.8 →
  ∃ (n : ℕ), n = 8 ∧ (fluid_ounces_needed / conversion_factor * 1000 / bottle_size_ml) <= ↑n :=
by
  sorry

end min_bottles_needed_l166_166119


namespace r_amount_l166_166880

-- Let p, q, and r be the amounts of money p, q, and r have, respectively
variables (p q r : ℝ)

-- Given conditions: p + q + r = 5000 and r = (2 / 3) * (p + q)
theorem r_amount (h1 : p + q + r = 5000) (h2 : r = (2 / 3) * (p + q)) :
  r = 2000 :=
sorry

end r_amount_l166_166880


namespace original_number_l166_166479

theorem original_number :
  ∃ x : ℝ, 0 < x ∧ (move_decimal_point x 3 = 3 / x) ∧ x = sqrt 30 / 100 :=
sorry

noncomputable def move_decimal_point (x : ℝ) (places : ℕ) : ℝ := x * 10^places

end original_number_l166_166479


namespace max_value_of_z_l166_166256

theorem max_value_of_z (x y : ℝ) (h1 : x + 2 * y - 5 ≥ 0) (h2 : x - 2 * y + 3 ≥ 0) (h3 : x - 5 ≤ 0) :
  ∃ x y, x + y = 9 :=
by {
  sorry
}

end max_value_of_z_l166_166256


namespace max_inner_a_b_l166_166623

variables {ℝ : Type} [inner_product_space ℝ]

variables (a b e : ℝ)
axiom norm_e : ∥e∥ = 1
axiom inner_a_e : ⟪a, e⟫ = 2
axiom inner_b_e : ⟪b, e⟫ = -1
axiom norm_a_b : ∥a + b∥ = 2

theorem max_inner_a_b : ∃ λ, ∀ a b e : ℝ, (∥e∥ = 1) ∧ (⟪a, e⟫ = 2) ∧ (⟪b, e⟫ = -1) 
                     ∧ (∥a + b∥ = 2) → ⟪a, b⟫ ≤ -5/4 := sorry

end max_inner_a_b_l166_166623


namespace percent_motorists_exceeding_speed_limit_l166_166321

-- Definitions based on conditions:
def total_motorists := 100
def percent_receiving_tickets := 10
def percent_exceeding_no_ticket := 50

-- The Lean 4 statement to prove the question
theorem percent_motorists_exceeding_speed_limit :
  (percent_receiving_tickets + (percent_receiving_tickets * percent_exceeding_no_ticket / 100)) = 20 :=
by
  sorry

end percent_motorists_exceeding_speed_limit_l166_166321


namespace max_value_z_in_D_l166_166624

-- Define the conditions for the region D
def in_region_D (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ real.sqrt 2 ∧ y ≤ 2 ∧ x ≤ real.sqrt 2 * y

-- Define point A
def point_A : ℝ × ℝ := (real.sqrt 2, 1)

-- Define the dot product z
def dot_product (OM OA : ℝ × ℝ) : ℝ := OM.1 * OA.1 + OM.2 * OA.2

-- Define the point OM as (x, y)
def OM (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define the value of z in terms of x and y
def z (x y : ℝ) : ℝ := dot_product (OM x y) point_A

-- Prove that the maximum value of z in region D is 4
theorem max_value_z_in_D : ∃ x y, in_region_D x y ∧ z x y = 4 :=
by {
  sorry
}

end max_value_z_in_D_l166_166624


namespace sum_first_15_odd_integers_l166_166848

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l166_166848


namespace isosceles_triangle_area_l166_166986

-- Definitions outlining the conditions
def isosceles_triangle (a b c : ℝ) : Prop :=
  a = b

-- Hypotheses specifying the conditions
variables (BK : ℝ) (AC : ℝ) (AB : ℝ) (CM : ℝ)
variables (h_base_height : BK = 10)
variables (h_lateral_height : CM = 12)
variables (h_isosceles : isosceles_triangle AB AB AC)

-- Theorem to prove the area of the triangle is 75
theorem isosceles_triangle_area :
  (∃ (x : ℝ), x > 0 ∧ AC = 2 * x ∧ AB = Math.sqrt (x^2 + BK^2) ∧ (1 / 2) * AC * BK = 75) :=
  sorry

end isosceles_triangle_area_l166_166986


namespace race_course_length_l166_166052

variable (v_A v_B d : ℝ)

theorem race_course_length (h1 : v_A = 4 * v_B) (h2 : (d - 60) / v_B = d / v_A) : d = 80 := by
  sorry

end race_course_length_l166_166052


namespace sum_of_integers_between_neg20_5_and_10_5_l166_166797

noncomputable def sum_arithmetic_series (a l n : ℤ) : ℤ :=
  n * (a + l) / 2

theorem sum_of_integers_between_neg20_5_and_10_5 :
  (sum_arithmetic_series (-20) 10 31) = -155 := by
  sorry

end sum_of_integers_between_neg20_5_and_10_5_l166_166797


namespace usual_time_to_bus_stop_l166_166058

theorem usual_time_to_bus_stop
  (T : ℕ) (S : ℕ)
  (h : S * T = (4/5 * S) * (T + 9)) :
  T = 36 :=
by
  sorry

end usual_time_to_bus_stop_l166_166058


namespace vector_sum_zero_l166_166728

-- Definitions of the vectors involved
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (CE AC DE AD AE : V)

-- Conditions for the problem
axiom h1 : CE + AC = AE
axiom h2 : DE + AD = AE

-- The goal statement
theorem vector_sum_zero : CE + AC - DE - AD = 0 :=
by {
  calc
  CE + AC - DE - AD = (CE + AC) - (DE + AD) : by rw [sub_add_sub_cancel]
                ... = AE - AE : by rw [h1, h2]
                ... = 0 : sub_self AE
}

end vector_sum_zero_l166_166728


namespace luka_age_difference_l166_166294

theorem luka_age_difference (a l : ℕ) (h1 : a = 8) (h2 : ∀ m : ℕ, m = 6 → l = m + 4) : l - a = 2 :=
by
  -- Assume Aubrey's age is 8
  have ha : a = 8 := h1
  -- Assume Max's age at Aubrey's 8th birthday is 6
  have hl : l = 10 := h2 6 rfl
  -- Hence, Luka is 2 years older than Aubrey
  sorry

end luka_age_difference_l166_166294


namespace alpha_beta_sum_l166_166136

variable (α β : ℝ)

theorem alpha_beta_sum (h : ∀ x, (x - α) / (x + β) = (x^2 - 64 * x + 992) / (x^2 + 56 * x - 3168)) :
  α + β = 82 :=
sorry

end alpha_beta_sum_l166_166136


namespace calculate_value_l166_166108

theorem calculate_value : (24 + 12) / ((5 - 3) * 2) = 9 := by 
  sorry

end calculate_value_l166_166108


namespace find_original_number_l166_166462

noncomputable def original_number (x : ℝ) : Prop :=
  (0 < x) ∧ (1000 * x = 3 / x)

theorem find_original_number : ∃ x : ℝ, original_number x ∧ x = real.sqrt 30 / 100 :=
by
  sorry

end find_original_number_l166_166462


namespace PA_PA_l166_166186

open Classical

variable {A B C : Point}
variable {P : Point}
variable {PA' PB' PC' PA'' PB'' : Line}

-- Define the conditions given in the problem
axiom triangle_ABC : Triangle
axiom point_P_inside_triangle : P ∈ triangle_ABC
axiom perpendicular_PA'_to_BC : IsPerpendicular PA' (LineSegment B C)
axiom perpendicular_PB'_to_CA : IsPerpendicular PB' (LineSegment C A)
axiom perpendicular_PC'_to_AB : IsPerpendicular PC' (LineSegment A B)
axiom perpendicular_PA''_to_BC' : IsPerpendicular PA'' (LineSegment B' C')
axiom perpendicular_PB''_to_CA' : IsPerpendicular PB'' (LineSegment C' A')

-- The theorem statement
theorem PA_PA'_PA''_eq_PB_PB'_PB'' :
  PA * PA' * PA'' = PB * PB' * PB'' :=
sorry

end PA_PA_l166_166186


namespace trains_crossing_time_l166_166786

-- Definitions based on conditions
def train_length : ℕ := 120
def time_train1_cross_pole : ℕ := 10
def time_train2_cross_pole : ℕ := 15

-- Question reformulated as a proof goal
theorem trains_crossing_time :
  let v1 := train_length / time_train1_cross_pole  -- Speed of train 1
  let v2 := train_length / time_train2_cross_pole  -- Speed of train 2
  let relative_speed := v1 + v2                    -- Relative speed in opposite directions
  let total_distance := train_length + train_length -- Sum of both trains' lengths
  let time_to_cross := total_distance / relative_speed -- Time to cross each other
  time_to_cross = 12 := 
by
  -- The proof here is stated, but not needed in this task
  -- All necessary computation steps
  sorry

end trains_crossing_time_l166_166786


namespace area_of_circle_l166_166697

-- Define the points P and Q
def P : ℝ × ℝ := (4, 11)
def Q : ℝ × ℝ := (10, 9)

-- Define that P and Q are on circle τ
def on_circle (P Q : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), dist P center = r ∧ dist Q center = r

-- Define the condition that the tangent lines at P and Q intersect on the x-axis
def tangent_intersection (P Q : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ (LP LQ : ℝ × ℝ), LP.2 = 0 ∧ LQ.2 = 0 ∧ tangent_line P LP ∧ tangent_line Q LQ ∧ LP.1 = LQ.1

-- (We would need to define what a tangent line means if not available in Lean)

-- Given conditions
def conditions : Prop :=
  on_circle P Q r ∧ tangent_intersection P Q r

-- Proof statement
theorem area_of_circle (r : ℝ) :
  conditions → (π * r^2 = 109 * π / 10) :=
sorry

end area_of_circle_l166_166697


namespace triangle_inequality_l166_166705

variables {α : Type} [EuclideanGeometry α]

-- Definitions of the points and isosceles triangle.
variables A B C D E : α

-- Conditions
variable h1 : IsoscelesTriangle A B C
variable h2 : OnSegment D A C
variable h3 : OnExtension E C A
variable h4 : Distance A D = Distance C

-- Required to prove BD + BE > AB + BC.
theorem triangle_inequality (h1 : IsoscelesTriangle A B C)
                           (h2 : OnSegment D A C)
                           (h3 : OnExtension E C A)
                           (h4 : Distance A D = Distance C) :
  SegmentLength (mkSegment B D) + SegmentLength (mkSegment B E) >
  SegmentLength (mkSegment A B) + SegmentLength (mkSegment B C) :=
sorry

end triangle_inequality_l166_166705


namespace problem1_problem2_problem3_l166_166184

-- Definition of the sequence
def a (n : ℕ) (k : ℚ) : ℚ := (k * n - 3) / (n - 3 / 2)

-- The first condition proof problem
theorem problem1 (k : ℚ) : (∀ n : ℕ, a n k = (a (n + 1) k + a (n - 1) k) / 2) → k = 2 :=
sorry

-- The second condition proof problem
theorem problem2 (k : ℚ) : 
  k ≠ 2 → 
  (if k > 2 then (a 1 k < k ∧ a 2 k = max (a 1 k) (a 2 k))
   else if k < 2 then (a 2 k < k ∧ a 1 k = max (a 1 k) (a 2 k))
   else False) :=
sorry

-- The third condition proof problem
theorem problem3 (k : ℚ) : 
  (∀ n : ℕ, n > 0 → a n k > (k * 2^n + (-1)^n) / 2^n) → 
  101 / 48 < k ∧ k < 13 / 6 :=
sorry

end problem1_problem2_problem3_l166_166184


namespace fraction_division_l166_166031

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := 
by
  sorry

end fraction_division_l166_166031


namespace arithmetic_sequence_binomial_expansion_l166_166351

open Nat

theorem arithmetic_sequence_binomial_expansion :
  ∃ (n r : ℕ), 
    n = 8 ∧ 
    (∀ x : ℝ, 
      let a := (binom n 0 : ℝ),
      let b := (binom n 1) * (1 / 2 : ℝ),
      let c := (binom n 2) * (1 / 2)^2,
      b - a = c - b ∧
        ((a = 1) ∧ 
         (b = (n : ℝ) / 2) ∧ 
         (c = (n : ℝ) * (n - 1) / 8) ∧
         r = 2 ∧
         (x ^ (n - 4 * r) / 6 = 1) ∧
         (binom n r) * (1 / 2)^r = 14)) :=
sorry

end arithmetic_sequence_binomial_expansion_l166_166351


namespace area_of_isosceles_triangle_PQR_l166_166268

noncomputable def area_of_triangle (P Q R : ℝ) (PQ PR QR PS QS SR : ℝ) : Prop :=
PQ = 17 ∧ PR = 17 ∧ QR = 16 ∧ PS = 15 ∧ QS = 8 ∧ SR = 8 →
(1 / 2) * QR * PS = 120

theorem area_of_isosceles_triangle_PQR :
  ∀ (P Q R : ℝ), 
  ∀ (PQ PR QR PS QS SR : ℝ), 
  PQ = 17 → PR = 17 → QR = 16 → PS = 15 → QS = 8 → SR = 8 →
  area_of_triangle P Q R PQ PR QR PS QS SR := 
by
  intros P Q R PQ PR QR PS QS SR hPQ hPR hQR hPS hQS hSR
  unfold area_of_triangle
  simp [hPQ, hPR, hQR, hPS, hQS, hSR]
  sorry

end area_of_isosceles_triangle_PQR_l166_166268


namespace equiv_pairs_l166_166533

def f1 (x : ℝ) := Real.ln x
def g1 (x : ℝ) := (1/2) * Real.ln (x^2)

def f2 (x : ℝ) := x
def g2 (x : ℝ) := Real.sqrt (x^2)

def f3 (x : ℝ) := Real.ln (Real.exp x)
def g3 (x : ℝ) := Real.exp (Real.ln x)

def f4 (x : ℝ) := Real.log (1/2) x
def g4 (x : ℝ) := Real.log 2 (1/x)

theorem equiv_pairs (x : ℝ) : 
  (f1 x = g1 x) ∧
  (f4 x = g4 x) := by
  sorry

end equiv_pairs_l166_166533


namespace div_result_l166_166558

theorem div_result : 2.4 / 0.06 = 40 := 
sorry

end div_result_l166_166558


namespace product_sequence_l166_166559

theorem product_sequence : (∏ (n : ℕ) in Finset.range (12 - 2 + 1) \map (function.add 2) 
  (λ n, (1 - 1/(n^2 : ℝ)))) = (13 / 24 : ℝ) := 
sorry

end product_sequence_l166_166559


namespace problemI_problemII_l166_166216

-- Define set B
def setB (a : ℝ) : set ℝ := {x | x^2 - (a + 2) * x + 2 * a = 0}

-- Define set A
def setA (a : ℝ) : set ℝ := {x | a - 2 < x ∧ x < a + 2}

-- Define the complement of set A
def complementA (a : ℝ) : set ℝ := {x | x ≤ a - 2 ∨ x ≥ a + 2}

-- Question I
theorem problemI (a : ℝ) (h : a = 0) : setA 0 ∪ setB 0 = {x | -2 < x ∧ x ≤ 2} :=
by sorry

-- Question II
theorem problemII (a : ℝ) (h : ¬ (setB a ∩ complementA a).empty) : a ≤ 0 ∨ a ≥ 4 :=
by sorry

end problemI_problemII_l166_166216


namespace find_original_number_l166_166463

noncomputable def original_number (x : ℝ) : Prop :=
  (0 < x) ∧ (1000 * x = 3 / x)

theorem find_original_number : ∃ x : ℝ, original_number x ∧ x = real.sqrt 30 / 100 :=
by
  sorry

end find_original_number_l166_166463


namespace solve_for_y_l166_166729

theorem solve_for_y : ∀ y : ℝ, 3^(y + 2) = 27^y → y = 1 :=
by
  intro y
  intro h
  sorry

end solve_for_y_l166_166729


namespace bug_meeting_time_and_distance_l166_166388

noncomputable def radius_large_circle := 6
noncomputable def radius_small_circle := 3
noncomputable def speed_large_circle := 4 * Real.pi
noncomputable def speed_small_circle := 3 * Real.pi

def circumference (radius : ℝ) : ℝ := 2 * radius * Real.pi

theorem bug_meeting_time_and_distance :
  let time_bug_meet := Nat.lcm (circumference radius_large_circle / speed_large_circle) (circumference radius_small_circle / speed_small_circle)
  time_bug_meet = 6 ∧
  (6 * speed_large_circle = 24 * Real.pi) ∧
  (6 * speed_small_circle = 18 * Real.pi) :=
by
  let circ_large := circumference radius_large_circle
  let circ_small := circumference radius_small_circle
  let time_large := circ_large / speed_large_circle -- time for bug on large circle to complete one round
  let time_small := circ_small / speed_small_circle -- time for bug on small circle to complete one round
  let lcm_time := Nat.lcm time_large time_small -- least common multiple of round completion times
  have time_eq : lcm_time = 6 := sorry
  have dist_large := 6 * speed_large_circle
  have dist_large_eq : dist_large = 24 * Real.pi := sorry
  have dist_small := 6 * speed_small_circle
  have dist_small_eq : dist_small = 18 * Real.pi := sorry
  exact ⟨time_eq, dist_large_eq, dist_small_eq⟩

end bug_meeting_time_and_distance_l166_166388


namespace circle_cartesian_eq_chord_length_range_f_inequality_solution_a_range_l166_166500

-- Coordinate System and Parametric Equations
theorem circle_cartesian_eq (r : ℝ) (θ : ℝ) (h : θ = π / 4) (h_r : r = sqrt 3) : 
  ∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 3 := 
sorry

theorem chord_length_range (α : ℝ) (h_α : α ∈ Ioo 0 (π / 4))
  (t x y : ℝ) (h_line : x = 2 + t * cos α ∧ y = 2 + t * sin α)
  (h_chord : ∀ (t₁ t₂ : ℝ), t₁ + t₂ = -2 * (cos α + sin α) ∧ t₁ * t₂ = -1) : 
  2 * sqrt 2 ≤ dist (2 + t₁ * cos α, 2 + t₁ * sin α) (2 + t₂ * cos α, 2 + t₂ * sin α) ∧ 
  dist (2 + t₁ * cos α, 2 + t₁ * sin α) (2 + t₂ * cos α, 2 + t₂ * sin α) < 2 * sqrt 3 :=
sorry

-- Inequalities
def f (x : ℝ) := |x + 1| + |x - 1|

theorem f_inequality_solution : {x : ℝ | f x < 4} = Ioo (-2) 2 :=
sorry

theorem a_range (a : ℝ) (h : ∃ x : ℝ, f x - |a + 1| < 0) : 
  a ∈ Iio (-3) ∪ Ioi 1 :=
sorry

end circle_cartesian_eq_chord_length_range_f_inequality_solution_a_range_l166_166500


namespace sum_first_15_odd_integers_l166_166844

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l166_166844


namespace triangle_sides_l166_166754

theorem triangle_sides (m : ℝ) (A B C M : ℝ × ℝ) :
  ∃ (BC AC AB : ℝ), 
  ∠BAC = 90 ∧
  BC = m ∧ AC = m * sqrt 3 ∧ AB = 2 * m :=
sorry

end triangle_sides_l166_166754


namespace set_complement_union_l166_166218

open Set

def M : Set ℝ := { x : ℝ | -3 < x ∧ x < 1 }
def N : Set ℝ := { x : ℝ | x ≤ -3 }
def desired_set : Set ℝ := { x : ℝ | x ≥ 1 }

theorem set_complement_union (M N : Set ℝ) :
  desired_set = (univ \ (M ∪ N)) :=
by
  unfold M
  unfold N
  unfold desired_set
  sorry

end set_complement_union_l166_166218


namespace solve_for_a_l166_166253

-- Define the sets M and N as given in the problem
def M : Set ℝ := {x : ℝ | x^2 + 6 * x - 16 = 0}
def N (a : ℝ) : Set ℝ := {x : ℝ | x * a - 3 = 0}

-- Define the proof statement
theorem solve_for_a (a : ℝ) : (N a ⊆ M) ↔ (a = 0 ∨ a = 3/2 ∨ a = -3/8) :=
by
  -- The proof would go here
  sorry

end solve_for_a_l166_166253


namespace sum_of_first_15_odd_positives_l166_166805

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l166_166805


namespace polynomial_degree_is_14_l166_166038

noncomputable def polynomial_degree (p q r s t u v : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0)
  (h4 : s ≠ 0) (h5 : t ≠ 0) (h6 : u ≠ 0) (h7 : v ≠ 0) : ℕ :=
  (polynomial.degree ((X^5 + C p * X^8 + C q * X + C r) *
  (X^4 + C s * X^3 + C t) *
  (X^2 + C u * X + C v))).natDegree

theorem polynomial_degree_is_14 (p q r s t u v : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0)
  (h4 : s ≠ 0) (h5 : t ≠ 0) (h6 : u ≠ 0) (h7 : v ≠ 0) :
  polynomial_degree p q r s t u v h1 h2 h3 h4 h5 h6 h7 = 14 :=
sorry

end polynomial_degree_is_14_l166_166038


namespace sum_of_first_n_odd_numbers_l166_166840

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l166_166840


namespace triangle_inequality_x_values_l166_166083

theorem triangle_inequality_x_values :
  {x : ℕ | 1 ≤ x ∧ x < 14} = {x | x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10 ∨ x = 11 ∨ x = 12 ∨ x = 13} :=
  by
    sorry

end triangle_inequality_x_values_l166_166083


namespace prasolov_inequality_l166_166497

variable (a b c γ : ℝ)

-- Conditions: a, b, c are sides of a triangle, and γ is the angle opposite side c.
-- Assuming some ancillary conditions for the sides of a triangle
axiom sides_of_triangle (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4: a + b > c) (h5 : b + c > a) (h6 : c + a > b)
axiom angle_opposite_c (h7 : 0 < γ) (h8 : γ < π)

theorem prasolov_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4: a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : 0 < γ) (h8 : γ < π) :
  c ≥ (a + b) * Real.sin (γ / 2) :=
sorry

end prasolov_inequality_l166_166497


namespace sum_of_first_three_terms_is_zero_l166_166750

variable (a d : ℤ) 

-- Definitions from the conditions
def a₄ := a + 3 * d
def a₅ := a + 4 * d
def a₆ := a + 5 * d

-- Theorem statement
theorem sum_of_first_three_terms_is_zero 
  (h₁ : a₄ = 8) 
  (h₂ : a₅ = 12) 
  (h₃ : a₆ = 16) : 
  a + (a + d) + (a + 2 * d) = 0 := 
by 
  sorry

end sum_of_first_three_terms_is_zero_l166_166750


namespace perpendicular_and_parallel_relationships_l166_166646

variables {l m : Line} {α : Plane}

def is_perpendicular (l : Line) (α : Plane) : Prop := sorry
def is_parallel (l1 l2 : Line) : Prop := sorry
def is_parallel_to_plane (m : Line) (α : Plane) : Prop := sorry

-- Proof problem statement
theorem perpendicular_and_parallel_relationships :
  (is_perpendicular l α) →
  ((is_parallel m l → is_perpendicular m α) ∧
   (is_perpendicular m α → is_parallel m l) ∧
   (is_parallel_to_plane m α → is_perpendicular m l)) :=
by {
  intro h₁,
  split; {
    intro h₂; sorry
  }
}

end perpendicular_and_parallel_relationships_l166_166646


namespace sum_of_first_15_odd_integers_l166_166825

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l166_166825


namespace checkered_board_cut_l166_166166

def can_cut_equal_squares (n : ℕ) : Prop :=
  n % 5 = 0 ∧ n > 5

theorem checkered_board_cut (n : ℕ) (h : n % 5 = 0 ∧ n > 5) :
  ∃ m, n^2 = 5 * m :=
by
  sorry

end checkered_board_cut_l166_166166


namespace molecular_weight_of_one_mole_l166_166792

theorem molecular_weight_of_one_mole (total_molecular_weight : ℝ) (number_of_moles : ℕ) (h1 : total_molecular_weight = 304) (h2 : number_of_moles = 4) : 
  total_molecular_weight / number_of_moles = 76 := 
by
  sorry

end molecular_weight_of_one_mole_l166_166792


namespace calculate_expression_l166_166938

theorem calculate_expression (x : ℝ) (h : x = 3) : (x^2 - 5 * x + 4) / (x - 4) = 2 :=
by
  rw [h]
  sorry

end calculate_expression_l166_166938


namespace find_S_when_R_75_T_5_l166_166299

def R_varies (R S T : ℝ) : ℝ :=
  (S^2) / (T^2)

theorem find_S_when_R_75_T_5 (c S : ℝ) (h: c = 12) :
  (∀ S : ℝ, (3 : ℝ) = c * R_varies 3 1 2) →
  (75 : ℝ) = 12 * R_varies 75 S 5 → 
  S = 12.5 :=
by 
  intro h1 h2
  sorry

end find_S_when_R_75_T_5_l166_166299


namespace omega_area_divided_by_pi_correct_l166_166963

-- Define the sides of the triangle
def AB : ℝ := 5
def BC : ℝ := 12
def CA : ℝ := 13

-- Define the incenter and the circle omega centered at the incenter
noncomputable def incenter := sorry
noncomputable def omega_radius := 2 * (4 / (Real.sqrt 3))

-- Define the area of circle omega divided by pi
noncomputable def omega_area_divided_by_pi := (omega_radius ^ 2) / pi

-- Theorem to prove the given problem statement
theorem omega_area_divided_by_pi_correct :
  omega_area_divided_by_pi = 16 / 3 := sorry

end omega_area_divided_by_pi_correct_l166_166963


namespace cos_beta_half_l166_166174

open Real Trigonometric

-- Defining the main conditions
variables (α β : ℝ)
hypotheses (h1 : cos α = 1 / 7)
           (h2 : cos (α + β) = -11 / 14)
           (h3 : 0 < α ∧ α < π / 2)
           (h4 : π / 2 < α + β ∧ α + β < π)

-- The goal is to prove that cos β = 1 / 2
theorem cos_beta_half : cos β = 1 / 2 :=
by
  sorry

end cos_beta_half_l166_166174


namespace original_number_l166_166461

theorem original_number (x : ℝ) (h : 1000 * x = 3 / x) : x = (real.sqrt 30) / 100 :=
by sorry

end original_number_l166_166461


namespace max_non_overshadowed_rooms_l166_166095

def is_overshadowed (g : ℕ → ℕ → bool) (x y : ℕ) : Prop :=
  g x y ∧ g (x - 1) y ∧ g (x + 1) y ∧ g x (y - 1)

def is_valid_position (x y : ℕ) : Prop :=
  x < 8 ∧ y < 8 ∧ (x > 0 ∧ y > 0) ∧ (y < 7)

theorem max_non_overshadowed_rooms : ∀ (g : ℕ → ℕ → bool),
  (∀ x y, is_valid_position x y → ¬ is_overshadowed g x y) → ∃ n, n ≤ 50 :=
by
  sorry

end max_non_overshadowed_rooms_l166_166095


namespace circle_tangents_l166_166685

theorem circle_tangents {
  Γ₁ Γ₂ Γ₃ Γ₄ : Type
  (tangent1 : ∃ P : Point, tangent_at Γ₁ Γ₃ P)
  (tangent2 : ∃ P : Point, tangent_at Γ₂ Γ₄ P)
  (meetA : ∃ A : Point, on_meet Γ₁ Γ₂ A)
  (meetB : ∃ B : Point, on_meet Γ₂ Γ₃ B)
  (meetC : ∃ C : Point, on_meet Γ₃ Γ₄ C)
  (meetD : ∃ D : Point, on_meet Γ₄ Γ₁ D)
  (distinctA : ∀ P : Point, ¬ P = A)
  (distinctB : ∀ P : Point, ¬ P = B)
  (distinctC : ∀ P : Point, ¬ P = C)
  (distinctD : ∀ P : Point, ¬ P = D)
} :
  ∀ (AB BC AD DC PB PD : ℝ),
  AB * BC = (PB^2) * (AD * DC / PD^2) :=
by
  sorry

end circle_tangents_l166_166685


namespace cab_driver_income_l166_166898

theorem cab_driver_income (x : ℕ) 
  (h₁ : (45 + x + 60 + 65 + 70) / 5 = 58) : x = 50 := 
by
  -- Insert the proof here
  sorry

end cab_driver_income_l166_166898


namespace homework_duration_equation_l166_166030

-- Define the initial and final durations and the rate of decrease
def initial_duration : ℝ := 100
def final_duration : ℝ := 70
def rate_of_decrease (x : ℝ) : ℝ := x

-- Statement of the proof problem
theorem homework_duration_equation (x : ℝ) :
  initial_duration * (1 - rate_of_decrease x) ^ 2 = final_duration :=
sorry

end homework_duration_equation_l166_166030


namespace billy_final_lap_time_l166_166932

/-- Given that Billy and Margaret are competing to swim 10 laps,
    and under the provided conditions of lap times for Billy and
    the total time for Margaret, prove the time it took for Billy to
    swim his final lap. --/
theorem billy_final_lap_time :
  let billy_first_5_laps := 2 * 60,
      billy_next_3_laps := 4 * 60,
      billy_ninth_lap := 1 * 60,
      margaret_total_time := 10 * 60,
      billy_ahead_margin := 30,
      billy_total_time := margaret_total_time - billy_ahead_margin,
      billy_first_9_laps_time := billy_first_5_laps + billy_next_3_laps + billy_ninth_lap in
  billy_total_time - billy_first_9_laps_time = 150 :=
by
  sorry

end billy_final_lap_time_l166_166932


namespace pq_passes_through_midpoint_of_rs_l166_166741

variable {α : Type*} [MetricSpace α] [ProperSpace α]

noncomputable def midpoint (A B : α) : α := sorry

theorem pq_passes_through_midpoint_of_rs
  (ω₁ ω₂ : Circle α)
  (center_ω₂_on_ω₁ : ω₂.center ∈ ω₁)
  (X : α)
  (X_on_ω₁ : X ∈ ω₁)
  (P Q : α)
  (Xp_tangent_ω₂ : TangentLine X P ω₂)
  (Xq_tangent_ω₂ : TangentLine X Q ω₂)
  (R S : α)
  (Xp_intersect_R : Xp_tangent_ω₂.line ∩ ω₁ = {R})
  (Xq_intersect_S : Xq_tangent_ω₂.line ∩ ω₁ = {S}) :
  Line PQ ∋ midpoint R S := sorry

end pq_passes_through_midpoint_of_rs_l166_166741


namespace sum_first_15_odd_integers_l166_166808

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l166_166808


namespace original_number_l166_166478

theorem original_number :
  ∃ x : ℝ, 0 < x ∧ (move_decimal_point x 3 = 3 / x) ∧ x = sqrt 30 / 100 :=
sorry

noncomputable def move_decimal_point (x : ℝ) (places : ℕ) : ℝ := x * 10^places

end original_number_l166_166478


namespace midpoint_B_l166_166362

-- Define the original points
def B : ℝ × ℝ := (1, 1)
def G : ℝ × ℝ := (5, 1)

-- Define the translation function
def translate (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - 5, p.2 + 2)

-- Define the new points after translation
def B' := translate B
def G' := translate G

-- Define the midpoint function
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Statement of the theorem
theorem midpoint_B'G' : midpoint B' G' = (-2, 3) := by
  sorry

end midpoint_B_l166_166362


namespace min_value_expression_l166_166179

noncomputable def f (t : ℝ) : ℝ :=
  (1 / (t + 1)) + (2 * t / (2 * t + 1))

theorem min_value_expression (x y : ℝ) (h : x * y > 0) :
  ∃ t, (x / y = t) ∧ t > 0 ∧ f t = 4 - 2 * Real.sqrt 2 := 
  sorry

end min_value_expression_l166_166179


namespace cars_closest_distance_l166_166389

noncomputable def min_distance {s1 s2 v1 v2 : ℝ} : ℝ :=
  let t := (s1 * v1 + s2 * v2) / (v1 * v1 + v2 * v2) in
  let d_squared := (s1^2 + s2^2 - (s1 * v1 + s2 * v2)^2 / (v1^2 + v2^2)) in
  real.sqrt d_squared

theorem cars_closest_distance :
  let s1 : ℝ := 200    -- initial distance of Car 1 from intersection in meters
  let s2 : ℝ := 500    -- initial distance of Car 2 from intersection in meters
  let v1 : ℝ := 60 * 1000 / 3600   -- velocity of Car 1 in meters/second
  let v2 : ℝ := 40 * 1000 / 3600   -- velocity of Car 2 in meters/second
  min_distance s1 s2 v1 v2 = 305
:= by
  -- The detailed proof goes here
  sorry

end cars_closest_distance_l166_166389


namespace possible_volumes_of_polyhedron_l166_166901

noncomputable def calculate_polyhedron_volume : ℝ :=
  sorry

theorem possible_volumes_of_polyhedron : 
  let volumes := {real.sqrt 3, 19 * real.sqrt 2 / 6, 4 + 7 * real.sqrt 2 / 3, 35 / 12 + 31 * real.sqrt 5 / 12} 
  in volumes = calculate_polyhedron_volume :=
sorry

end possible_volumes_of_polyhedron_l166_166901


namespace expected_rolls_leap_year_l166_166089

def dice_roll_prob : Nat → ℚ
| 8     := 1 / 8
| _     := 7 / 8

def leap_year_days : Nat := 366

noncomputable def expected_rolls (days : ℕ) : ℚ :=
(let E := (7/8 : ℚ) * 1 + (1/8 : ℚ) * (1 + E) in E) * days

theorem expected_rolls_leap_year : expected_rolls leap_year_days = 421.71 :=
by
  sorry

end expected_rolls_leap_year_l166_166089


namespace sum_first_15_odd_integers_l166_166847

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l166_166847


namespace maintain_income_with_new_demand_l166_166918

variable (P D : ℝ) -- Original Price and Demand
def new_price := 1.20 * P -- New Price after 20% increase
def new_demand := 1.12 * D -- New Demand after 12% increase due to advertisement
def original_income := P * D -- Original income
def new_income := new_price * new_demand -- New income after changes

theorem maintain_income_with_new_demand :
  ∀ P D : ℝ, P * D = 1.20 * P * 1.12 * (D_new : ℝ) → (D_new = 14/15 * D) :=
by
  intro P D h
  sorry

end maintain_income_with_new_demand_l166_166918


namespace cannot_make_all_numbers_zero_l166_166706

theorem cannot_make_all_numbers_zero :
  let initial_sum := (1989 * 1990) / 2 in
  let is_even (n : ℕ) := n % 2 = 0 in
  let operation (x y : ℕ) := |x - y| in
  ∀ n : ℕ, n ≠ 0 ->
  ¬(∀ (numbers : list ℕ), 
    (numbers = list.range 1989.succ ∧ 
     ∀ i ∈ numbers, i ∈ list.range 1989.succ ∧ 
     ∀ x y ∈ numbers, x ≠ y → 
     ∃ z, z = operation x y → 
     is_even (initial_sum - 2 * min x y)) →
    numbers.sum = 0)
:= sorry

end cannot_make_all_numbers_zero_l166_166706


namespace trig_identity_cosine_difference_l166_166107

theorem trig_identity_cosine_difference :
  cos (96 * π / 180) * cos (24 * π / 180) - sin (96 * π / 180) * sin (66 * π / 180) = -1 / 2 :=
by
  sorry

end trig_identity_cosine_difference_l166_166107


namespace police_force_female_officers_l166_166324

theorem police_force_female_officers :
  ∃ (female_officers_A female_officers_B female_officers_C total_female_officers : ℕ),
  (0.18 * female_officers_A = 90) ∧
  (0.25 * female_officers_B = 60) ∧
  (0.30 * female_officers_C = 40) ∧
  (female_officers_A = 500) ∧
  (female_officers_B = 240) ∧
  (female_officers_C = 133) ∧
  (total_female_officers = female_officers_A + female_officers_B + female_officers_C) ∧
  (total_female_officers = 873) :=
by
  -- Proof is not required
  use 500, 240, 133, 873
  sorry

end police_force_female_officers_l166_166324


namespace intersection_A_B_l166_166684

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem intersection_A_B : A ∩ (B ∩ ℕ) = {2, 3, 4} := by
  sorry

end intersection_A_B_l166_166684


namespace estimate_total_observations_in_interval_l166_166262

def total_observations : ℕ := 1000
def sample_size : ℕ := 50
def frequency_in_sample : ℝ := 0.12

theorem estimate_total_observations_in_interval : 
  frequency_in_sample * (total_observations : ℝ) = 120 :=
by
  -- conditions defined above
  -- use given frequency to estimate the total observations in the interval
  -- actual proof omitted
  sorry

end estimate_total_observations_in_interval_l166_166262


namespace sum_of_first_15_odd_integers_l166_166860

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l166_166860


namespace horizontal_chord_length_l166_166903

def has_horizontal_chord (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x : ℝ, f(a + x) = f(x)

def function_f (x : ℝ) : ℝ :=
  x^3 - x

theorem horizontal_chord_length (a : ℝ) : 0 < a ∧ a ≤ 2 ↔ has_horizontal_chord function_f a := 
sorry

end horizontal_chord_length_l166_166903


namespace log_eq_15_given_log_base3_x_eq_5_l166_166191

variable (x : ℝ)
variable (log_base3_x : ℝ)
variable (h : log_base3_x = 5)

theorem log_eq_15_given_log_base3_x_eq_5 (h : log_base3_x = 5) : log_base3_x * 3 = 15 :=
by
  sorry

end log_eq_15_given_log_base3_x_eq_5_l166_166191


namespace increase_by_30_percent_l166_166065

theorem increase_by_30_percent (initial : ℝ) (percentage : ℝ) (final : ℝ) (h1 : initial = 500) (h2 : percentage = 0.30) : final = 650 := by
  have h3 : final = initial * percentage + initial := sorry
  rw [h1, h2] at h3
  exact h3

end increase_by_30_percent_l166_166065


namespace ernesto_conditional_figure_l166_166977

theorem ernesto_conditional_figure (semicircles: ℕ) (segments: ℕ) (l: ℝ) 
  (h_semicircles: semicircles = 3) 
  (h_segments: segments = 4) 
  (h_length: ∀ s, s = l):
  (figure_options.determine_figure semicircles segments) = d := 
sorry

end ernesto_conditional_figure_l166_166977


namespace min_abs_sum_is_two_l166_166430

theorem min_abs_sum_is_two : ∃ x ∈ set.Ioo (- ∞) (∞), ∀ y ∈ set.Ioo (- ∞) (∞), (|y + 1| + |y + 3| + |y + 6| ≥ 2) ∧ (|x + 1| + |x + 3| + |x + 6| = 2) := sorry

end min_abs_sum_is_two_l166_166430


namespace miae_closer_than_hyori_l166_166897

def bowl_volume : ℝ := 1000
def miae_estimate : ℝ := 1100
def hyori_estimate : ℝ := 850

def miae_difference : ℝ := abs (miae_estimate - bowl_volume)
def hyori_difference : ℝ := abs (bowl_volume - hyori_estimate)

theorem miae_closer_than_hyori : miae_difference < hyori_difference :=
by
  sorry

end miae_closer_than_hyori_l166_166897


namespace pow_mod_1110_l166_166680

theorem pow_mod_1110 (n : ℕ) (h₀ : 0 ≤ n ∧ n < 1111)
    (h₁ : 2^1110 % 11 = 1) (h₂ : 2^1110 % 101 = 14) : 
    n = 1024 := 
sorry

end pow_mod_1110_l166_166680


namespace largest_number_of_three_l166_166781

theorem largest_number_of_three
  (x y z : ℝ)
  (h1 : x + y + z = 1)
  (h2 : x * y + x * z + y * z = -7)
  (h3 : x * y * z = -21) : max x (max y z) = sqrt 7 :=
sorry

end largest_number_of_three_l166_166781


namespace collinear_AKP_l166_166681

noncomputable def rhombus (A B C D : Point) :=
  quadrilateral A B C D ∧
  ∀ (X Y Z W : Point), (X, Y, Z, W) == (A, B, C, D) →
    (dist X Y = dist Y Z ∧ dist Y Z = dist Z W ∧ dist Z W = dist W X) ∧
    (angle A C = π/2 ∧ angle B D = π/2)

theorem collinear_AKP
  (A B C D K P : Point)
  (h_rhombus : rhombus A B C D)
  (h_K_on_CD : K ∈ line (C, D))
  (h_AD_eq_BK : dist A D = dist B K)
  (h_P_inter_BD_perp_bisect_BC : P ∈ line (B, D) ∧ P ∈ perp_bisector (B, C)) :
  collinear {A, K, P} :=
sorry

end collinear_AKP_l166_166681


namespace probability_intersection_diagonals_nonagon_l166_166264

theorem probability_intersection_diagonals_nonagon :
  let nonagon_diagonals := SetOfDiagonalsOfNonagon
  let pairs_diagonals := PairsNonagonDiagonals
  let intersect_diagonals := IntersectingDiagonalsNonagon
  ∃ (p : ℚ), p = 14 / 39 ∧ probabilityPairsIntersect nonagon_diagonals pairs_diagonals intersect_diagonals = p := 
sorry -- proof not required

end probability_intersection_diagonals_nonagon_l166_166264


namespace three_digit_integers_count_l166_166232

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_perfect_square_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 4

def is_allowed_digit (d : ℕ) : Prop :=
  is_prime_digit d ∨ is_perfect_square_digit d

theorem three_digit_integers_count : ∃ n : ℕ, n = 216 ∧ 
  (∀ x y z : ℕ, x ≠ 0 → is_allowed_digit x → is_allowed_digit y → is_allowed_digit z → 
   (x * 100 + y * 10 + z) > 99 ∧ (x * 100 + y * 10 + z) < 1000 → 
   (nat.card {n : ℕ | n = x * 100 + y * 10 + z ∧ is_allowed_digit x ∧ is_allowed_digit y ∧ is_allowed_digit z}) = 216) :=
begin
  sorry
end

end three_digit_integers_count_l166_166232


namespace min_abs_sum_l166_166402

theorem min_abs_sum (x : ℝ) : 
  (∃ x, (∀ y, ∥ y + 1 ∥ + ∥ y + 3 ∥ + ∥ y + 6 ∥ ≥ ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥) ∧ 
        ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥ = 5) := 
sorry

end min_abs_sum_l166_166402


namespace sum_of_digits_unit_digit_square_l166_166162

noncomputable def S (n : ℕ) : ℕ := n.digits.sum

noncomputable def U (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_unit_digit_square (n : ℕ) (h : n = S(n) + U(n)^2) :
  n = 13 ∨ n = 46 ∨ n = 99 :=
by 
  sorry

end sum_of_digits_unit_digit_square_l166_166162


namespace arithmetic_sequence_sum_reciprocals_lt_two_l166_166599

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < a n ∧
    2 * (n * (n+1) / 2) = (a n)^2 + a n

theorem arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  sequence a ∧ 
  (∀ n, S n = n * (n + 1) / 2) → 
  (∀ n : ℕ, a n = n) :=
begin
  sorry
end

theorem sum_reciprocals_lt_two (a : ℕ → ℝ) (S : ℕ → ℝ) :
  sequence a ∧ 
  (∀ n, S n = n * (n + 1) / 2) → 
  (∀ n : ℕ, (∑ k in finset.range (n + 1), 1 / S k) < 2) :=
begin
  sorry
end

end arithmetic_sequence_sum_reciprocals_lt_two_l166_166599


namespace min_subsets_of_A_l166_166609

theorem min_subsets_of_A (A : Set ℝ) (h1 : ∀ p ∈ A, p ≠ 0 ∧ p ≠ -1 → -1 / (p + 1) ∈ A) (h2 : 2 ∈ A) :
  ∃ n : ℕ, n = 3 ∧ ∀ S : Set (Set ℝ), S = Set.powerset A → S.card = 8 :=
by
  sorry

end min_subsets_of_A_l166_166609


namespace perimeter_quad_l166_166911

/--
Given a quadrilateral with vertices at (0, 0), (2, 6), (6, 5), and (3, 2),
prove that its perimeter, when expressed in the form p√10 + q√17 + r√2 + s√13,
has the coefficient sum equal to 7.
-/
theorem perimeter_quad (p q r s : ℤ) :
  let A := (0 : ℝ, 0 : ℝ),
      B := (2 : ℝ, 6 : ℝ),
      C := (6 : ℝ, 5 : ℝ),
      D := (3 : ℝ, 2 : ℝ)
  in
  let dist (P Q : ℝ × ℝ) := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  in
  let AB := dist A B,
      BC := dist B C,
      CD := dist C D,
      DA := dist D A
  in
  AB = 2 * Real.sqrt 10 ∧
  BC = Real.sqrt 17 ∧
  CD = 3 * Real.sqrt 2 ∧
  DA = Real.sqrt 13 →
  AB + BC + CD + DA = (p * Real.sqrt 10) + (q * Real.sqrt 17) + (r * Real.sqrt 2) + (s * Real.sqrt 13) →
  p + q + r + s = 7 :=
sorry

end perimeter_quad_l166_166911


namespace largest_fraction_sum_l166_166116

theorem largest_fraction_sum : 
  (max (max (max (max 
  ((1 : ℚ) / 3 + (1 : ℚ) / 4) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 5)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 2)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 9)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 6)) = (5 : ℚ) / 6 
:= 
by
  sorry

end largest_fraction_sum_l166_166116


namespace compute_expression_l166_166950

def floor_sixth_root (x : ℤ) : ℤ := int.floor (real.rpow x (1 / 6))

theorem compute_expression :
  (∏ x in finset.range (204 + 1), if x % 2 = 1 then floor_sixth_root x else 1) /
  (∏ x in finset.range (204 + 1), if x % 2 = 0 then floor_sixth_root x else 1) = 5 / 12 := 
sorry

end compute_expression_l166_166950


namespace min_marked_squares_l166_166187

theorem min_marked_squares (n : ℕ) (h_even : n % 2 = 0) :
  ∃ (m : ℕ), (∀ (i j : ℕ), i < n → j < n → (∃ (adj : (i = 0 ∨ i = n-1 ∨ j = 0 ∨ j = n-1) ∧ (∃ k l : ℕ, k < n ∧ l < n ∧ m ≠ n ∧ ((k = i ∨ k = i+1 ∨ k = i-1) ∧ (l = j ∨ l = j+1 ∨ l = j-1)) ∧ m = l))), 
  m = (n*(n+2))/4 :=
sorry

end min_marked_squares_l166_166187


namespace sum_series_identity_l166_166123

theorem sum_series_identity :
  (∑ n in Finset.range 500, 1 / ((n + 1) ^ 2)) = (∑ m in Finset.range 501, 1 / (m ^ 2)) - 1 := 
sorry

end sum_series_identity_l166_166123


namespace casey_number_of_ducks_l166_166946

-- Definitions based on the problem conditions
def water_per_minute := 3
def corn_rows := 4
def plants_per_row := 15
def water_per_corn_plant := 0.5
def pigs := 10
def water_per_pig := 4
def ducks_needed_water := 0.25
def pumping_minutes := 25

-- Calculate the variables required to prove the final result
def total_corn_plants := corn_rows * plants_per_row
def total_water_corn := total_corn_plants * water_per_corn_plant
def total_water_pigs := pigs * water_per_pig
def total_pumped_water := water_per_minute * pumping_minutes
def water_for_ducks := total_pumped_water - (total_water_corn + total_water_pigs)
def number_of_ducks := water_for_ducks / ducks_needed_water

-- Lean 4 statement to prove the number of ducks
theorem casey_number_of_ducks : number_of_ducks = 20 := by
  sorry

end casey_number_of_ducks_l166_166946


namespace find_m_l166_166707

theorem find_m (m : ℝ) (h : |m| = |m + 2|) : m = -1 :=
sorry

end find_m_l166_166707


namespace sequence_bound_l166_166312

-- Define the sequence as an infinite sequence of positive real numbers
variable {x : ℕ → ℝ} 

-- Conditions
axiom decreasing_sequence (h1 : ∀ i, x i > 0) (h2 : ∀ i j, i ≤ j → x i ≥ x j)
axiom inequality_condition : ∀ n, x 1 + ∑ i in range n, x ((i + 1) ^ 2) / (i + 1) ≤ 1

-- The theorem to be proved
theorem sequence_bound (n : ℕ) : x 1 + ∑ i in range n, x (i + 1) / (i + 1) < 3 := 
by
  sorry

end sequence_bound_l166_166312


namespace original_number_value_l166_166448

noncomputable def orig_number_condition (x : ℝ) : Prop :=
  1000 * x = 3 * (1 / x)

theorem original_number_value : ∃ x : ℝ, 0 < x ∧ orig_number_condition x ∧ x = √(30) / 100 :=
begin
  -- the proof
  sorry
end

end original_number_value_l166_166448


namespace number_of_valid_arrangements_is_24_l166_166173

variable {P : Type} (teacher : P) (male1 male2 female1 female2 female3 : P)
noncomputable def validArrangements (l : List P) : Bool :=
  l.head = male1 ∧ l.last = male2 ∧
  (sameAdjacentTwo female1 female2 l ∨ sameAdjacentTwo female1 female3 l ∨ sameAdjacentTwo female2 female3 l)

noncomputable def sameAdjacentTwo {P : Type} (a b : P) (l : List P) : Bool :=
  ∃ (xs ys : List P), l = xs ++ [a, b] ++ ys ∨ l = xs ++ [b, a] ++ ys

theorem number_of_valid_arrangements_is_24 :
  ∃ l : List P, l.length = 6 ∧ validArrangements teacher male1 male2 female1 female2 female3 l = 24 :=
sorry

end number_of_valid_arrangements_is_24_l166_166173


namespace domain_tan_neg_x_correct_l166_166039

noncomputable def domain_of_tan_neg_x : Set ℝ := { x : ℝ | ∀ n : ℤ, x ≠ -π/2 + n * π }

theorem domain_tan_neg_x_correct : domain_of_tan_neg_x = { x : ℝ | ∀ n : ℤ, x ≠ -π/2 + n * π } :=
by
  sorry

end domain_tan_neg_x_correct_l166_166039


namespace smallest_n_satisfying_conditions_l166_166395

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, 
    (∃ m₁ : ℕ, n * 2 = m₁^2) ∧ -- n * 2 is a perfect square
    (∃ m₂ : ℕ, n * 3 = m₂^3) ∧ -- n * 3 is a perfect cube
    is_least { k | (∃ m₁ : ℕ, k * 2 = m₁^2) ∧ (∃ m₂ : ℕ, k * 3 = m₂^3)} 72 := 
sorry

end smallest_n_satisfying_conditions_l166_166395


namespace cost_of_each_soda_l166_166382

theorem cost_of_each_soda
  (num_people : ℕ)
  (price_sandwich : ℕ)
  (price_fruit_salad : ℕ)
  (price_snack_bag : ℕ)
  (total_expenditure : ℕ)
  (sodas_per_person : ℕ)
  (num_sandwiches : num_people)
  (num_fruit_salads : num_people)
  (num_snack_bags : ℕ) 
  (total_sandwiches_cost : price_sandwich * num_people)
  (total_fruit_salads_cost : price_fruit_salad * num_people)
  (total_snack_bags_cost : price_snack_bag * num_snack_bags)
  (remaining_amount_for_sodas : total_expenditure - total_sandwiches_cost - total_fruit_salads_cost - total_snack_bags_cost)
  (total_sodas : sodas_per_person * num_people)
  (soda_cost : remaining_amount_for_sodas / total_sodas)
  (sandwich_price_is_correct : price_sandwich = 5)
  (fruit_salad_price_is_correct : price_fruit_salad = 3)
  (snack_bag_price_correct : price_snack_bag = 4)
  (num_people_correct : num_people = 4)
  (total_expenditure_correct : total_expenditure = 60)
  (sodas_per_person_correct : sodas_per_person = 2)
  (num_snack_bags_correct : num_snack_bags = 3) :
  soda_cost = 2 :=
by
  sorry

end cost_of_each_soda_l166_166382


namespace quadratic_function_l166_166868

-- Define the four functions mentioned in the problem
def funcA (x : ℝ) : ℝ := 2 * x + 1
def funcB (x : ℝ) : ℝ := x^2 + 1
def funcC (x : ℝ) : ℝ := (x - 1)^2 - x^2
def funcD (x : ℝ) : ℝ := 1 / x^2

-- Define what it means for a function to be quadratic
def isQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

-- Theorem stating that funcB is the only quadratic function among the options
theorem quadratic_function : isQuadratic funcB ∧ ¬isQuadratic funcA ∧ ¬isQuadratic funcC ∧ ¬isQuadratic funcD :=
by
  sorry

end quadratic_function_l166_166868


namespace probability_of_first_three_red_cards_l166_166072

def total_cards : ℕ := 104
def suits : ℕ := 4
def cards_per_suit : ℕ := 26
def red_suits : ℕ := 2
def black_suits : ℕ := 2
def total_red_cards : ℕ := 52
def total_black_cards : ℕ := 52

noncomputable def probability_first_three_red : ℚ :=
  (total_red_cards / total_cards) * ((total_red_cards - 1) / (total_cards - 1)) * ((total_red_cards - 2) / (total_cards - 2))

theorem probability_of_first_three_red_cards :
  probability_first_three_red = 425 / 3502 :=
sorry

end probability_of_first_three_red_cards_l166_166072


namespace inequality_solution_l166_166734

theorem inequality_solution (x : ℝ) : 
    (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
    sorry

end inequality_solution_l166_166734


namespace power_function_decreasing_intervals_l166_166212

theorem power_function_decreasing_intervals (α : ℝ) (f : ℝ → ℝ) 
  (h₁ : f = λ x, x ^ α)
  (h₂ : f 2 = 1/2) :
  ∃ (α : ℝ), α = -1 ∧ (∀ x ∈ (-∞, 0) ∪ (0, +∞), f x = 1 / x) :=
by
  sorry

end power_function_decreasing_intervals_l166_166212


namespace molecular_weight_K2Cr2O7_l166_166791

/--
K2Cr2O7 consists of:
- 2 K atoms
- 2 Cr atoms
- 7 O atoms

Atomic weights:
- K: 39.10 g/mol
- Cr: 52.00 g/mol
- O: 16.00 g/mol

We need to prove that the molecular weight of 4 moles of K2Cr2O7 is 1176.80 g/mol.
-/
theorem molecular_weight_K2Cr2O7 :
  let weight_K := 39.10
  let weight_Cr := 52.00
  let weight_O := 16.00
  let mol_weight_K2Cr2O7 := (2 * weight_K) + (2 * weight_Cr) + (7 * weight_O)
  (4 * mol_weight_K2Cr2O7) = 1176.80 :=
by
  sorry

end molecular_weight_K2Cr2O7_l166_166791


namespace constant_term_in_expansion_l166_166138

theorem constant_term_in_expansion : 
  let expr := (2 * Rat.mkInt 1) * x + Rat.mk 1 1 / x - 1
  let expansion := expr ^ 5
  ∃ const_term : ℤ, constant_term expansion = -161 := 
by
  sorry

end constant_term_in_expansion_l166_166138


namespace units_digit_sum_l166_166997

def base8_to_base10 (n : Nat) : Nat :=
  let units := n % 10
  let tens := (n / 10) % 10
  tens * 8 + units

theorem units_digit_sum (n1 n2 : Nat) (h1 : n1 = 45) (h2 : n2 = 67) : ((base8_to_base10 n1) + (base8_to_base10 n2)) % 8 = 4 := by
  sorry

end units_digit_sum_l166_166997


namespace digit_sum_of_nines_l166_166756

theorem digit_sum_of_nines (k : ℕ) (n : ℕ) (h : n = 9 * (10^k - 1) / 9):
  (8 + 9 * (k - 1) + 1 = 500) → k = 55 := 
by 
  sorry

end digit_sum_of_nines_l166_166756


namespace area_of_circular_lid_l166_166976

theorem area_of_circular_lid (d : ℝ) (h : d = 2.75) : 
  let r := d / 2 in
  real.pi * r^2 ≈ 5.9375 :=
by
  sorry

end area_of_circular_lid_l166_166976


namespace correct_statements_l166_166890

variable {m n : Line}
variable {α β : Plane}

-- Assuming the definitions of parallelism and perpendicularity for lines and planes
axiom parallel (l : Line) (p : Plane) : Prop
axiom perpendicular (l : Line) (p : Plane) : Prop
axiom parallel_planes (p q : Plane) : Prop
axiom perpendicular_planes (p q : Plane) : Prop

-- Define the four statements from the problem
def statement_1 := parallel m α ∧ parallel n β ∧ parallel_planes α β → parallel m n
def statement_2 := parallel m α ∧ perpendicular n β ∧ perpendicular_planes α β → parallel m n
def statement_3 := perpendicular m α ∧ parallel n β ∧ parallel_planes α β → perpendicular m n
def statement_4 := perpendicular m α ∧ perpendicular n β ∧ perpendicular_planes α β → perpendicular m n

-- The correct number of statements among the four
theorem correct_statements : (statement_3 ∧ statement_4) ∧ ¬(statement_1) ∧ ¬(statement_2) := by
  sorry

end correct_statements_l166_166890


namespace range_of_x_l166_166244

variable (x : ℝ) (m : ℝ)
variable ht: |m| ≤ 2

theorem range_of_x 
  (h: ∀ m, |m| ≤ 2 → 2 * x - 1 > m * (x ^ 2 - 1)) : 
  (sqrt 7 - 1) / 2 < x ∧ x < (sqrt 3 + 1) / 2 :=
by
  sorry

end range_of_x_l166_166244


namespace billy_final_lap_equals_150_l166_166930

variables {first_5_laps_time : ℕ} {next_3_laps_time : ℕ} {next_1_lap_time : ℕ}
variables {margaret_total_time : ℕ} {billy_margin : ℕ}

def billy_first_5_laps : ℕ := first_5_laps_time * 60
def billy_next_3_laps : ℕ := next_3_laps_time * 60
def billy_next_lap : ℕ := next_1_lap_time * 60
def billy_total_time : ℕ := margaret_total_time - billy_margin
def billy_first_9_laps : ℕ := billy_first_5_laps + billy_next_3_laps + billy_next_lap
def billy_final_lap_time : ℕ := billy_total_time - billy_first_9_laps

theorem billy_final_lap_equals_150 :
  first_5_laps_time = 2 ∧
  next_3_laps_time = 4 ∧
  next_1_lap_time = 1 ∧
  margaret_total_time = 10 * 60 ∧
  billy_margin = 30
  → billy_final_lap_time = 150 := 
by {
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h_rest,
  cases h_rest with h4 h5,
  simp [billy_first_5_laps, billy_next_3_laps, billy_next_lap, billy_total_time, billy_first_9_laps, h1, h2, h3, h4, h5],
  sorry
}

end billy_final_lap_equals_150_l166_166930


namespace classical_prob_exp_is_exp1_l166_166092

-- Define the conditions under which an experiment is a classical probability model
def classical_probability_model (experiment : String) : Prop :=
  match experiment with
  | "exp1" => true  -- experiment ①: finite outcomes and equal likelihood
  | "exp2" => false -- experiment ②: infinite outcomes
  | "exp3" => false -- experiment ③: unequal likelihood
  | "exp4" => false -- experiment ④: infinite outcomes
  | _ => false

theorem classical_prob_exp_is_exp1 : classical_probability_model "exp1" = true ∧
                                      classical_probability_model "exp2" = false ∧
                                      classical_probability_model "exp3" = false ∧
                                      classical_probability_model "exp4" = false :=
by
  sorry

end classical_prob_exp_is_exp1_l166_166092


namespace total_valid_votes_l166_166266

theorem total_valid_votes (V : ℝ) (H_majority : 0.70 * V - 0.30 * V = 188) : V = 470 :=
by
  sorry

end total_valid_votes_l166_166266


namespace probability_C_l166_166512

variable (pA pB pD pC : ℚ)
variable (hA : pA = 1 / 4)
variable (hB : pB = 1 / 3)
variable (hD : pD = 1 / 6)
variable (total_prob : pA + pB + pD + pC = 1)

theorem probability_C (hA : pA = 1 / 4) (hB : pB = 1 / 3) (hD : pD = 1 / 6) (total_prob : pA + pB + pD + pC = 1) : pC = 1 / 4 :=
sorry

end probability_C_l166_166512


namespace find_B_l166_166912

noncomputable def point_A : ℝ × ℝ × ℝ := (-2, 8, 10)
noncomputable def plane : ℝ × ℝ × ℝ → Prop := λ p, p.1 + p.2 + p.3 = 15
noncomputable def point_C : ℝ × ℝ × ℝ := (4, 4, 7)

theorem find_B :
  ∃ B : ℝ × ℝ × ℝ,
    (∃ B1 : ℝ × ℝ × ℝ, plane B1 ∧ B1 = B) ∧
    ∃ D : ℝ × ℝ × ℝ,
      ( ∃ P : ℝ × ℝ × ℝ, plane P ∧ P = ((-11/3, 19/3, 25/3)) ∧
        D = (2 * (-11/3) + 2, 2 * (19/3) - 8, 2 * (25/3) - 10) ) ∧
      D = B ∧
      ∃ t : ℝ, B = (4 + 20 * t, 4 + 2 * t, 7 + 11 * t) ∧
               plane B :=
  B = (14/3, 11/3, 32/3) sorry

end find_B_l166_166912


namespace sum_and_product_formulas_l166_166600

/-- 
Given an arithmetic sequence {a_n} with the sum of the first n terms S_n = 2n^2, 
and in the sequence {b_n}, b_1 = 1 and b_{n+1} = 3b_n (n ∈ ℕ*),
prove that:
(Ⅰ) The general formula for sequences {a_n} is a_n = 4n - 2,
(Ⅱ) The general formula for sequences {b_n} is b_n = 3^{n-1},
(Ⅲ) Let c_n = a_n * b_n, prove that the sum of the first n terms of the sequence {c_n}, denoted as T_n, is T_n = (2n - 2) * 3^n + 2.
-/
theorem sum_and_product_formulas (S_n : ℕ → ℕ) (b : ℕ → ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (T_n : ℕ → ℕ) :
  (∀ n, S_n n = 2 * n^2) →
  (b 1 = 1) →
  (∀ n, b (n + 1) = 3 * (b n)) →
  (∀ n, a n = S_n n - S_n (n - 1)) →
  ∀ n, (T_n n = (2*n - 2) * 3^n + 2) := sorry

end sum_and_product_formulas_l166_166600


namespace product_of_nonreal_roots_l166_166993

noncomputable def polynomial_equation := (x : ℂ) → x^4 - 6 * x^3 + 15 * x^2 - 20 * x = 4020

theorem product_of_nonreal_roots :
  (∀ (x : ℂ), polynomial_equation x → x.im ≠ 0 → (2 + (complex.I * (4036)^(1/4))) * (2 - (complex.I * (4036)^(1/4))) = 4 + (4036)^(1/2)) :=
begin
  sorry
end

end product_of_nonreal_roots_l166_166993


namespace candies_left_after_eating_l166_166019

theorem candies_left_after_eating :
  let initial_candies := 1000,
      vasya_interval := 7,
      petya_interval := 9,
      vasya_start := 9,
      petya_start := 7,
      candies_after_vasya := initial_candies - (initial_candies - vasya_start + vasya_interval - 1) / vasya_interval,
      candies_left := candies_after_vasya - (candies_after_vasya - petya_start + petya_interval - 1) / petya_interval
  in candies_left = 761 := 
by
  let initial_candies := 1000
  let vasya_interval := 7
  let petya_interval := 9
  let vasya_start := 9
  let petya_start := 7
  let candies_after_vasya := initial_candies - (initial_candies - vasya_start + vasya_interval - 1) / vasya_interval
  let candies_left := candies_after_vasya - (candies_after_vasya - petya_start + petya_interval - 1) / petya_interval
  show candies_left = 761
  sorry

end candies_left_after_eating_l166_166019


namespace smallest_prime_after_six_nonprimes_l166_166282

theorem smallest_prime_after_six_nonprimes :
  ∃ (n : ℕ), prime n ∧ (∀ m < n, m > 0 ∧ (¬ prime m ∧ (∃ i : ℕ, m = 89+i))) :=
begin
  sorry
end

end smallest_prime_after_six_nonprimes_l166_166282


namespace part_I_monotonic_intervals_part_II_min_value_of_a_l166_166617

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := ln (x + 1) - (a * x) / (x + 1) - x

theorem part_I_monotonic_intervals (a : ℝ) (h1 : 0 < a) :
  (a ≥ 1/4 → ∀ x : ℝ, -1 < x → f x a ≤ f (x + 1) a) ∧
  (0 < a ∧ a < 1/4 →
    ∃ x1 x2 : ℝ, x1 < x2 ∧
    (∀ x : ℝ, -1 < x ∧ x < x1 → f x a ≤ f (x + 1) a) ∧
    (∀ x : ℝ, x1 < x ∧ x < x2 → f x a ≥ f (x + 1) a) ∧
    (∀ x : ℝ, x > x2 → f x a ≤ f (x + 1) a)) :=
sorry

theorem part_II_min_value_of_a :
  ∃ a : ℤ, a > 2 ∧ (∀ x > 0, f x (a : ℝ) + x + 1 < - x / (x + 1)) :=
  ∃ (a : ℤ) (a_min : ℤ), a = 5 ∧ (∀ x > 0, f x (5 : ℝ) + x + 1 < - x / (x + 1)) :=
  sorry

end part_I_monotonic_intervals_part_II_min_value_of_a_l166_166617


namespace team_total_points_l166_166396

-- Definition of Wade's average points per game
def wade_avg_points_per_game := 20

-- Definition of teammates' average points per game
def teammates_avg_points_per_game := 40

-- Definition of the number of games
def number_of_games := 5

-- The total points calculation problem
theorem team_total_points 
  (Wade_avg : wade_avg_points_per_game = 20)
  (Teammates_avg : teammates_avg_points_per_game = 40)
  (Games : number_of_games = 5) :
  5 * wade_avg_points_per_game + 5 * teammates_avg_points_per_game = 300 := 
by 
  -- The proof is omitted and marked as sorry
  sorry

end team_total_points_l166_166396


namespace EquivalentGraphs_l166_166867

noncomputable def GraphI : ℝ → ℝ := λ x, x^2 - 1

noncomputable def GraphII : ℝ → ℝ := λ x, if x ≠ 1 then x * (x + 1) else 0

noncomputable def GraphIII : ℝ → ℝ := λ x, if x ≠ 1 then x * (x + 1) else 0

theorem EquivalentGraphs :
  (∀ x, GraphII x = GraphIII x) ∧ ¬ (∀ x, GraphI x = GraphII x) :=
by
  sorry

end EquivalentGraphs_l166_166867


namespace sum_of_first_15_odd_integers_l166_166853

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l166_166853


namespace candidate_total_score_l166_166507

theorem candidate_total_score (written_score : ℝ) (interview_score : ℝ) (written_weight : ℝ) (interview_weight : ℝ) :
    written_score = 90 → interview_score = 80 → written_weight = 0.70 → interview_weight = 0.30 →
    written_score * written_weight + interview_score * interview_weight = 87 :=
by
  intros
  sorry

end candidate_total_score_l166_166507


namespace can_be_divided_into_four_parts_summing_to_20_l166_166743

def clock_numerals : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

theorem can_be_divided_into_four_parts_summing_to_20 :
  ∃ (A B C D : List ℕ),
    A.sum = 20 ∧ B.sum = 20 ∧ C.sum = 20 ∧ D.sum = 20 ∧
    A.all (λ n, n ∈ clock_numerals) ∧
    B.all (λ n, n ∈ clock_numerals) ∧
    C.all (λ n, n ∈ clock_numerals) ∧
    D.all (λ n, n ∈ clock_numerals) ∧
    Multiset.card (A.to_multiset + B.to_multiset + C.to_multiset + D.to_multiset) = 12 ∧
    (A.to_multiset + B.to_multiset + C.to_multiset + D.to_multiset) = clock_numerals.to_multiset := by
  sorry

end can_be_divided_into_four_parts_summing_to_20_l166_166743


namespace area_VRS_l166_166736

open real

-- Definitions based on conditions
def square_side_length : ℝ := sqrt 225
def triangle_area (b h : ℝ) : ℝ := 1 / 2 * b * h

def area_PTS : ℝ := triangle_area square_side_length square_side_length
def area_QUTV : ℝ := 45
def area_UTV : ℝ := area_PTS / 4
def area_remaining_in_QUTV : ℝ := area_QUTV - area_UTV
def area_triangle_QTV : ℝ := area_remaining_in_QUTV / 2

-- Prove the area of triangle VRS
theorem area_VRS : area_PTS - area_UTV - 2 * area_triangle_QTV = 67.5 := by
  sorry

end area_VRS_l166_166736


namespace segment_ratio_l166_166721

variables {A B C D M N : Type} 
variables {a c : ℝ}
variables [trapezoid A B C D]

-- Assuming AB and CD are the bases of the trapezoid ABCD with lengths a and c respectively
axiom base_lengths : AB.length = a ∧ CD.length = c

-- Assuming MN is a mid-segment parallel to the bases and equal to the geometric mean of the bases
axiom mid_segment : MN.length = Real.sqrt (a * c) ∧ parallel MN AB ∧ parallel MN CD

-- Definition stating the ratio property to be proven
theorem segment_ratio : ∀ {AM MD : ℝ}, MN.length = Real.sqrt (a * c) → parallel MN AB → parallel MN CD → 
  (MN.length = Real.sqrt (a * c)) → (AM / MD = Real.sqrt (a / c)) :=
by sorry

end segment_ratio_l166_166721


namespace min_points_to_win_l166_166652

theorem min_points_to_win : ∀ (points : ℕ), (∀ (race_results : ℕ → ℕ), 
  (points = race_results 1 * 4 + race_results 2 * 2 + race_results 3 * 1) 
  ∧ (∀ i, 1 ≤ race_results i ∧ race_results i ≤ 4) 
  ∧ (∀ i j, i ≠ j → race_results i ≠ race_results j) 
  ∧ (race_results 1 + race_results 2 + race_results 3 = 4)) → (15 ≤ points) :=
by
  sorry

end min_points_to_win_l166_166652


namespace min_distance_PA_l166_166696

theorem min_distance_PA :
  let A : ℝ × ℝ := (0, 1)
  ∀ (P : ℝ × ℝ), (∃ x : ℝ, x > 0 ∧ P = (x, (x + 2) / x)) →
  ∃ d : ℝ, d = 2 ∧ ∀ Q : ℝ × ℝ, (∃ x : ℝ, x > 0 ∧ Q = (x, (x + 2) / x)) → dist A Q ≥ d :=
by
  sorry

end min_distance_PA_l166_166696


namespace butterfat_mixture_l166_166491

/-
  Given:
  - 8 gallons of milk with 40% butterfat
  - x gallons of milk with 10% butterfat
  - Resulting mixture with 20% butterfat

  Prove:
  - x = 16 gallons
-/

theorem butterfat_mixture (x : ℝ) : 
  (0.40 * 8 + 0.10 * x) / (8 + x) = 0.20 → x = 16 := 
by
  sorry

end butterfat_mixture_l166_166491


namespace dog_food_bags_needed_l166_166068

theorem dog_food_bags_needed
  (cup_weight: ℝ)
  (dogs: ℕ)
  (cups_per_day: ℕ)
  (days_in_month: ℕ)
  (bag_weight: ℝ)
  (hcw: cup_weight = 1/4)
  (hd: dogs = 2)
  (hcd: cups_per_day = 6 * 2)
  (hdm: days_in_month = 30)
  (hbw: bag_weight = 20) :
  (dogs * cups_per_day * days_in_month * cup_weight) / bag_weight = 9 :=
by
  sorry

end dog_food_bags_needed_l166_166068


namespace perimeter_of_ABFCDE_l166_166524

-- Define the problem parameters
def square_perimeter : ℤ := 60
def side_length (p : ℤ) : ℤ := p / 4
def equilateral_triangle_side (l : ℤ) : ℤ := l
def new_shape_sides : ℕ := 6
def new_perimeter (s : ℤ) : ℤ := new_shape_sides * s

-- Define the theorem to be proved
theorem perimeter_of_ABFCDE (p : ℤ) (s : ℕ) (len : ℤ) : len = side_length p → len = equilateral_triangle_side len →
  new_perimeter len = 90 :=
by
  intros h1 h2
  sorry

end perimeter_of_ABFCDE_l166_166524


namespace correct_conclusions_l166_166005

variable (a b c m : ℝ)
variable (y1 y2 : ℝ)

-- Conditions: 
-- Parabola y = ax^2 + bx + c, intersects x-axis at (-3,0) and (1,0)
-- a < 0
-- Points P(m-2, y1) and Q(m, y2) are on the parabola, y1 < y2

def parabola_intersects_x_axis_at_A_B : Prop :=
  ∀ x : ℝ, x = -3 ∨ x = 1 → a * x^2 + b * x + c = 0

def concavity_and_roots : Prop :=
  a < 0 ∧ b = 2 * a ∧ c = -3 * a

def conclusion_1 : Prop :=
  a * b * c < 0

def conclusion_2 : Prop :=
  b^2 - 4 * a * c > 0

def conclusion_3 : Prop :=
  3 * b + 2 * c = 0

def conclusion_4 : Prop :=
  y1 < y2 → m ≤ -1

-- Correct conclusions given the parabola properties
theorem correct_conclusions :
  concavity_and_roots a b c →
  parabola_intersects_x_axis_at_A_B a b c →
  conclusion_1 a b c ∨ conclusion_2 a b c ∨ conclusion_3 a b c ∨ conclusion_4 a b c :=
sorry

end correct_conclusions_l166_166005


namespace probability_tan_gt_one_correct_l166_166278

noncomputable def probability_tan_gt_one : Prop :=
  let interval := Icc (- (Real.pi / 2)) (Real.pi / 2)
  let sub_interval := Ioc (Real.pi / 4) (Real.pi / 2)
  let prob := (MeasureTheory.Measure.measureOf {x | x ∈ sub_interval}) / 
              (MeasureTheory.Measure.measureOf {x | x ∈ interval})
  prob = 1 / 4

theorem probability_tan_gt_one_correct : probability_tan_gt_one :=
  sorry

end probability_tan_gt_one_correct_l166_166278


namespace scatter_plot_variable_placement_l166_166046

theorem scatter_plot_variable_placement
  (forecast explanatory : Type)
  (scatter_plot : explanatory → forecast → Prop) : 
  ∀ (x : explanatory) (y : forecast), scatter_plot x y → (True -> True) := 
by
  intros x y h
  sorry

end scatter_plot_variable_placement_l166_166046


namespace min_M_value_partition_l166_166581

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1

theorem min_M_value_partition (partition : (ℕ → ℝ)) (n : ℕ)
  (h : 0 = partition 0 ∧ partition n = 4 ∧ ∀ i, 0 < i → i < n → partition i > partition (i - 1)) :
  ∑ i in (finset.range n).succ) (abs (f (partition (i + 1)) - f (partition i))) ≤ 10 :=
by
  sorry

end min_M_value_partition_l166_166581


namespace solution_to_integral_equation_l166_166394

noncomputable def phi (x : ℝ) : ℝ := 1 + ∫ t in 0..x, phi t

theorem solution_to_integral_equation : ∀ x : ℝ, phi x = Real.exp x :=
by
  sorry

end solution_to_integral_equation_l166_166394


namespace print_time_rounded_l166_166519

noncomputable def pages_per_minute : ℚ := 20
noncomputable def total_pages : ℚ := 250

theorem print_time_rounded :
  Real.toNearestInt (total_pages / pages_per_minute) = 13 :=
by
  sorry

end print_time_rounded_l166_166519


namespace average_minutes_heard_l166_166081

theorem average_minutes_heard 
  (total_people : ℕ)
  (full_talk : ℕ)
  (heard_entire : ℕ)
  (heard_none : ℕ)
  (percentage_half : ℕ)
  (half_talk : ℕ)
  (two_thirds_talk : ℕ)
  (total_minutes : ℕ → ℕ → ℕ → ℕ → ℕ)
  (average_minutes : ℕ → ℕ → ℚ) :
  total_people = 100 →
  full_talk = 90 →
  heard_entire = 30 →
  heard_none = 15 →
  percentage_half = 40 →
  half_talk = 45 →
  two_thirds_talk = 60 →
  total_minutes heard_entire full_talk (heard_none + heard_entire) half_talk = 5670 →
  average_minutes 5670 total_people = 56.7 := 
by {
  assume h1 h2 h3 h4 h5 h6 h7 h8,
  sorry,
}

end average_minutes_heard_l166_166081


namespace smallest_value_of_abs_sum_l166_166420

theorem smallest_value_of_abs_sum : ∃ x : ℝ, (x = -3) ∧ ( ∀ y : ℝ, |y + 1| + |y + 3| + |y + 6| ≥ 5 ) :=
by
  use -3
  split
  . exact rfl
  . intro y
    have h1 : |y + 1| + |y + 3| + |y + 6| = sorry,
    sorry

end smallest_value_of_abs_sum_l166_166420


namespace count_eligible_three_digit_numbers_l166_166233

def is_eligible_digit (d : Nat) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem count_eligible_three_digit_numbers : 
  (∃ n : Nat, 100 ≤ n ∧ n < 1000 ∧
  (∀ d : Nat, d ∈ [n / 100, (n / 10) % 10, n % 10] → is_eligible_digit d)) →
  ∃ count : Nat, count = 343 :=
by
  sorry

end count_eligible_three_digit_numbers_l166_166233


namespace sqrt_div_correct_l166_166866

theorem sqrt_div_correct : real.sqrt 27 / real.sqrt 3 = 3 := 
by
  sorry

end sqrt_div_correct_l166_166866


namespace smallest_value_of_abs_sum_l166_166421

theorem smallest_value_of_abs_sum : ∃ x : ℝ, (x = -3) ∧ ( ∀ y : ℝ, |y + 1| + |y + 3| + |y + 6| ≥ 5 ) :=
by
  use -3
  split
  . exact rfl
  . intro y
    have h1 : |y + 1| + |y + 3| + |y + 6| = sorry,
    sorry

end smallest_value_of_abs_sum_l166_166421


namespace coefficient_x4_l166_166368

theorem coefficient_x4 (n : ℕ) (H : (2^n = 256)) :
  let T_r := λ r, (binomial n r) * 2^(n-r) * (-1)^r * (x^(r/2)) in 
  (coeff (x^4) (∑ r in range (n+1), T_r r)) = 1 :=
by
  sorry

end coefficient_x4_l166_166368


namespace ab_root_inequality_l166_166747

theorem ab_root_inequality (a b : ℝ) (h1: ∀ x : ℝ, (x + a) * (x + b) = -9) (h2: a < 0) (h3: b < 0) :
  a + b < -6 :=
sorry

end ab_root_inequality_l166_166747


namespace least_c_l166_166692

def f (x : ℤ) : ℤ :=
  if x % 2 = 1 then x + 5 else x / 2

theorem least_c (b : ℤ) (c : ℤ) (h1 : c % 2 = 1) (h2 : f(f(f(c))) = b) : c = 21 :=
sorry

end least_c_l166_166692


namespace find_n_l166_166885

def smallest_a (n : ℕ) : ℕ := 
  Inf { k : ℕ | n ∣ k! }

theorem find_n (n : ℕ) (h : 0 < n) : 
  (smallest_a n * 3 = 2 * n) ↔ n = 9 := 
by 
  sorry

end find_n_l166_166885


namespace simplify_percent_l166_166788

theorem simplify_percent (rational_form : ℚ) :
  let decimal_form : ℚ := 0.166 in
  let simplified_form : ℚ := 83 / 50 in
  (rational_form = decimal_form) →
  (rational_form = simplified_form) :=
by
  sorry

end simplify_percent_l166_166788


namespace conjugate_of_z_find_a_and_b_l166_166207

noncomputable def z : ℂ := ((1 - complex.i) ^ 2 + 3 * (1 + complex.i)) / (2 - complex.i)

theorem conjugate_of_z : conj z = 1 - complex.i := by
  sorry

theorem find_a_and_b (a b : ℝ) (h : a * z + b = 1 - complex.i) : a = -1 ∧ b = 2 := by
  sorry

end conjugate_of_z_find_a_and_b_l166_166207


namespace maximum_magnitude_diff_l166_166226

-- Define the vectors
def a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def b : ℝ × ℝ := (Real.sqrt 3, 1)

-- Define the magnitude of the difference
def mag_diff (θ : ℝ) : ℝ := Real.sqrt ((a θ).fst - b.fst)^2 + ((a θ).snd - b.snd)^2)

-- Define the theorem to be proved
theorem maximum_magnitude_diff : ∃ θ : ℝ, mag_diff(θ) = 3 :=
sorry

end maximum_magnitude_diff_l166_166226


namespace total_distance_flash_runs_l166_166530

-- Define the problem with given conditions
theorem total_distance_flash_runs (v k d a : ℝ) (hk : k > 1) : 
  let t := d / (v * (k - 1))
  let distance_to_catch_ace := k * v * t
  let total_distance := distance_to_catch_ace + a
  total_distance = (k * d) / (k - 1) + a := 
by
  sorry

end total_distance_flash_runs_l166_166530


namespace emma_age_proof_l166_166738

def is_age_of_emma (age : Nat) : Prop := 
  let guesses := [26, 29, 31, 33, 35, 39, 42, 44, 47, 50]
  let at_least_60_percent_low := (guesses.filter (· < age)).length * 10 ≥ 6 * guesses.length
  let exactly_two_off_by_one := (guesses.filter (λ x => x = age - 1 ∨ x = age + 1)).length = 2
  let is_prime := Nat.Prime age
  at_least_60_percent_low ∧ exactly_two_off_by_one ∧ is_prime

theorem emma_age_proof : is_age_of_emma 43 := 
  by sorry

end emma_age_proof_l166_166738


namespace nests_count_l166_166384

theorem nests_count :
  ∃ (N : ℕ), (6 = N + 3) ∧ (N = 3) :=
by
  sorry

end nests_count_l166_166384


namespace find_ellipse_equation_sum_of_slopes_constant_l166_166143

noncomputable def ellipse_pass_through (A : ℝ × ℝ) (a b : ℝ) (e : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 
  (A.fst ^ 2 / a ^ 2 + A.snd ^ 2 / b ^ 2 = 1) ∧ 
  (e = Real.sqrt 2 / 2)

theorem find_ellipse_equation {A : ℝ × ℝ} {a b : ℝ} (h : ellipse_pass_through A a b (Real.sqrt 2 / 2)) :
  a = Real.sqrt 2 ∧ b = 1 :=
sorry

def line_passing_through (P : ℝ × ℝ) (k : ℝ) : Prop := 
  ∃ Q : ℝ × ℝ, P ≠ Q ∧ 
  ∃ m n : ℝ, m ≠ 0 ∧ P = (m, k * (m - 1) + 1) ∧ Q = (n, k * (n - 1) + 1)

theorem sum_of_slopes_constant {a b : ℝ} (h₁ : a = Real.sqrt 2) (h₂ : b = 1) (k : ℝ) :
  let E : ℝ × ℝ → Prop := λ (P : ℝ × ℝ), 
    (P.fst ^ 2 / h₁ ^ 2 + P.snd ^ 2 / h₂ ^ 2 = 1)
  ∀ (P Q : ℝ × ℝ), 
    E P → E Q → P ≠ (0, -1) → Q ≠ (0, -1) →
    line_passing_through (1, 1) k →
    (P = (x1, y1)) → (Q = (x2, y2)) →
    ((y1 + 1) / x1 + (y2 + 1) / x2 = 2) :=
sorry

end find_ellipse_equation_sum_of_slopes_constant_l166_166143


namespace calculate_distance_and_midpoint_l166_166398

-- Define the points
def x1 := 2
def y1 := -2
def x2 := 8
def y2 := 8

-- State the properties to be proven
theorem calculate_distance_and_midpoint :
  let d := Real.sqrt((x2 - x1)^2 + (y2 - y1)^2) in
  let m := ((x1 + x2) / 2, (y1 + y2) / 2) in
  d = Real.sqrt(136) ∧ m = (5, 3) :=
by
  sorry

end calculate_distance_and_midpoint_l166_166398


namespace longest_side_is_2_sqrt_5_l166_166139

-- Define conditions as separate terms and predicates
def condition1 (x y : ℝ) : Prop := x + y ≤ 4
def condition2 (x y : ℝ) : Prop := x + 2y ≥ 4
def condition3 (x : ℝ) : Prop := x ≥ 0
def condition4 (y : ℝ) : Prop := y ≥ 0

-- Define the vertices of the polygon formed by the constraints
def vertices : Set (ℝ × ℝ) := 
  {p | ∃ x y, p = (x, y) ∧ condition1 x y ∧ condition2 x y ∧ condition3 x ∧ condition4 y}

-- Function to find distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define a set of sides of the polygon
def sides : Set ℝ := 
  { distance (0, 2) (4, 0), distance (0, 2) (0, 0), distance (4, 0) (0, 0)}

-- The longest side length should be the maximum in the set of sides
theorem longest_side_is_2_sqrt_5 : 
  ∃ max L, L ∈ sides ∧ max = 2 * real.sqrt 5 ∧ ∀ l, l ∈ sides → l ≤ max := 
begin
  sorry
end

end longest_side_is_2_sqrt_5_l166_166139


namespace find_constants_l166_166679

open Matrix 

noncomputable def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 2, 1; 2, 0, 2; 1, 2, 0]

theorem find_constants :
  let s := (-10 : ℤ)
  let t := (-8 : ℤ)
  let u := (-36 : ℤ)
  B^3 + s • (B^2) + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := sorry

end find_constants_l166_166679


namespace managers_salary_l166_166740

-- Definitions based on conditions
def avg_salary_50_employees : ℝ := 2000
def num_employees : ℕ := 50
def new_avg_salary : ℝ := 2150
def num_employees_with_manager : ℕ := 51

-- Condition statement: The manager's salary such that when added, average salary increases as given.
theorem managers_salary (M : ℝ) :
  (num_employees * avg_salary_50_employees + M) / num_employees_with_manager = new_avg_salary →
  M = 9650 := sorry

end managers_salary_l166_166740


namespace casper_initial_candies_l166_166552

theorem casper_initial_candies : 
  ∃ x : ℕ, 
    (∃ y1 : ℕ, y1 = x / 2 - 3) ∧
    (∃ y2 : ℕ, y2 = y1 / 2 - 5) ∧
    (∃ y3 : ℕ, y3 = y2 / 2 - 2) ∧
    (y3 = 10) ∧
    x = 122 := 
sorry

end casper_initial_candies_l166_166552


namespace sum_of_first_15_odd_positives_l166_166800

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l166_166800


namespace largest_sum_l166_166114

theorem largest_sum :
  let s1 := (1 : ℚ) / 3 + (1 : ℚ) / 4
  let s2 := (1 : ℚ) / 3 + (1 : ℚ) / 5
  let s3 := (1 : ℚ) / 3 + (1 : ℚ) / 2
  let s4 := (1 : ℚ) / 3 + (1 : ℚ) / 9
  let s5 := (1 : ℚ) / 3 + (1 : ℚ) / 6
  in max s1 (max s2 (max s3 (max s4 s5))) = (5 : ℚ) / 6 := by
  sorry

end largest_sum_l166_166114


namespace sandy_will_be_32_l166_166379

  -- Define the conditions
  def world_record_length : ℝ := 26
  def sandy_current_length : ℝ := 2
  def monthly_growth_rate : ℝ := 0.1
  def sandy_current_age : ℝ := 12

  -- Define the annual growth rate calculation
  def annual_growth_rate : ℝ := monthly_growth_rate * 12

  -- Define total growth needed
  def total_growth_needed : ℝ := world_record_length - sandy_current_length

  -- Define the years needed to grow the fingernails to match the world record
  def years_needed : ℝ := total_growth_needed / annual_growth_rate

  -- Define Sandy's age when she achieves the world record length
  def sandy_age_when_record_achieved : ℝ := sandy_current_age + years_needed

  -- The statement we want to prove
  theorem sandy_will_be_32 :
    sandy_age_when_record_achieved = 32 :=
  by
    -- Placeholder proof
    sorry
  
end sandy_will_be_32_l166_166379


namespace M_properties_l166_166560

def div_by_11 (n : ℤ) : Prop := (n ^ 3 + 2 * n ^ 2 - 5 * n - 6) % 11 = 0

def set_M : set ℤ := { n | -100 ≤ n ∧ n ≤ 500 ∧ div_by_11 n }

theorem M_properties : 
  (∀ n, n ∈ set_M → -100 ≤ n ∧ n ≤ 500) ∧
  (card (set_M) = 136) ∧
  (∀ n, n ∈ set_M → -100 ≤ n) ∧
  (∀ n, n ∈ set_M → n ≤ 497) :=
by
  sorry

end M_properties_l166_166560


namespace find_Monday_sales_l166_166134

variables (initial_inventory Tuesday_sales WedToSun_sales_per_day Saturday_delivery end_inventory : ℕ)
variable Monday_sales : ℕ

-- Conditions
def initial_inventory := 4500
def Tuesday_sales := 900
def WedToSun_sales_per_day := 50
def Saturday_delivery := 650
def end_inventory := 1555

-- Calculation
def total_sales_excluding_Monday := Tuesday_sales + 5 * WedToSun_sales_per_day
def Monday_sales := initial_inventory - total_sales_excluding_Monday + Saturday_delivery - end_inventory

-- Theorem statement
theorem find_Monday_sales : Monday_sales = 2445 := sorry

end find_Monday_sales_l166_166134


namespace find_positive_integer_n_l166_166153

theorem find_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧ 
  (sin (Real.pi / (3 * n)) + cos (Real.pi / (3 * n))) = (Real.sqrt (n + 1) / 2) ∧
  n = 7 :=
by
  sorry

end find_positive_integer_n_l166_166153


namespace probability_satisfies_inequality_l166_166908

/-- Define the conditions for the points (x, y) -/
def within_rectangle (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5

def satisfies_inequality (x y : ℝ) : Prop :=
  x + 2 * y ≤ 6

/-- Compute the probability that a randomly selected point within the rectangle
also satisfies the inequality -/
theorem probability_satisfies_inequality : (∃ p : ℚ, p = 3 / 10) :=
sorry

end probability_satisfies_inequality_l166_166908


namespace positive_integer_solution_of_inequality_l166_166363

theorem positive_integer_solution_of_inequality :
  {x : ℕ // 0 < x ∧ x < 2} → x = 1 :=
by
  sorry

end positive_integer_solution_of_inequality_l166_166363


namespace slope_of_tangent_at_2_l166_166511

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then
  x / (x - 1)
else
  (λ y : ℝ, ite (y < 0) (y / (y - 1)) (y / (y + 1))) x

theorem slope_of_tangent_at_2 :
  (f 2) = 2 / 3 ∧ (deriv f 2) = 1 / 9 :=
by
  sorry

end slope_of_tangent_at_2_l166_166511


namespace calculate_value_l166_166937

theorem calculate_value : 12 * ((1/3 : ℝ) + (1/4) - (1/12))⁻¹ = 24 :=
by
  sorry

end calculate_value_l166_166937


namespace compare_quadratic_expression_l166_166554

theorem compare_quadratic_expression (x : ℝ) : (x^2 - x) > (x - 2) := 
by {
  have h : (x^2 - x) - (x - 2) = (x - 1)^2 + 1,
  { ring },
  have h' : (x - 1)^2 + 1 > 0,
  { exact add_pos_of_nonneg_of_pos (pow_two_nonneg (x - 1)) zero_lt_one },
  linarith
}

end compare_quadratic_expression_l166_166554


namespace A_lies_on_segment_BC_l166_166717

variables {Point : Type} [MetricSpace Point]

-- Given points A, B, and C
variables (A B C : Point)

-- Condition: For any point M, either MA ≤ MB or MA ≤ MC
axiom distance_condition (M : Point) : dist M A ≤ dist M B ∨ dist M A ≤ dist M C

-- Prove that point A lies on the segment BC
theorem A_lies_on_segment_BC : LiesOnSegment A B C :=
sorry

end A_lies_on_segment_BC_l166_166717


namespace tangent_line_l166_166987

noncomputable def f : ℝ → ℝ := λ x, x^2 + x - 1

theorem tangent_line (x y : ℝ) (hpt : (x, y) = (1, 1)) :
  (∃ k, y - 1 = k * (x - 1) ∧ k = 3) → 3 * x - y - 2 = 0 :=
by
  sorry

end tangent_line_l166_166987


namespace sequence_product_l166_166144

theorem sequence_product :
  (∏ n in finset.range 5, (1 / (3 ^ (2 * n + 1))) * (3 ^ (2 * n + 2))) = 243 :=
by
  sorry

end sequence_product_l166_166144


namespace arithmetic_geometric_mean_negatives_l166_166192

variables {a b : ℝ}
def A := (a + b) / 2
noncomputable def G := real.sqrt (a * b)

theorem arithmetic_geometric_mean_negatives (h1 : a < 0) (h2 : b < 0) (h3 : a ≠ b) : A < G :=
by
  sorry

end arithmetic_geometric_mean_negatives_l166_166192


namespace parallelogram_not_symmetrical_l166_166048

-- Define the shapes
inductive Shape
| circle
| rectangle
| isosceles_trapezoid
| parallelogram

-- Define what it means for a shape to be symmetrical
def is_symmetrical (s: Shape) : Prop :=
  match s with
  | Shape.circle => True
  | Shape.rectangle => True
  | Shape.isosceles_trapezoid => True
  | Shape.parallelogram => False -- The condition we're interested in proving

-- The main theorem stating the problem
theorem parallelogram_not_symmetrical : is_symmetrical Shape.parallelogram = False :=
by
  sorry

end parallelogram_not_symmetrical_l166_166048


namespace trajectory_and_ellipse_l166_166596

def point := (ℝ × ℝ)
def line := ℝ → ℝ

variables (F : point) (l : line) (Q : point → Prop) (N : point → Prop) (A : point)
variable (m : line)
variables (B C D E : point)
def distance (p1 p2 : point) : ℝ := sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
def area (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) : ℝ := abs ((x₁ * y₂ + x₂ * y₃ + x₃ * y₁ - y₁ * x₂ - y₂ * x₃ - y₃ * x₁) / 2)

noncomputable def S1 (A F B C : point) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := F
  let (x₃, y₃) := B
  let (x₄, y₄) := C
  area x₁ x₂ x₃ y₁ y₂ y₃

noncomputable def S2 (O F D E : point) : ℝ :=
  let (x₀, y₀) := O
  let (x₂, y₂) := F
  let (x₃, y₃) := D
  let (x₄, y₄) := E
  area x₀ x₃ x₄ y₀ y₃ y₄

-- Given conditions
axiom h1 : F = (0, -1)
axiom h2 : (∀ Q, Q F ∧ Q (0, 1))
axiom h3 : ∀ A, N A → A = (0, 2)
axiom h4 : ∀ m, ∃ B C D E, m F ∧ N B ∧ N C ∧ Q D ∧ Q E

-- The translated problem as a Lean theorem
theorem trajectory_and_ellipse (F O A : point) (k : ℝ) : 
  (∀ Q, Q F ∧ Q (0, 1) → ∃ x y, Q (x, y) → x^2 = -4 * y) ∧
  (∀ A, N A → A = (0, 2) →
  (∃ (b a : ℝ), a = 2 ∧ b = sqrt (4 - 1) ∧
    ∀ x y, N (x, y) ↔ x^2 / 3 + y^2 / 4 = 1)) ∧
  (∀ (k : ℝ), let Z := S1 (A F B C) * S2 (O F D E) in
    ∀ B C D E, m F ∧ N B ∧ N C ∧ Q D ∧ Q E → 
    9 ≤ Z ∧ Z < 12) :=
sorry

end trajectory_and_ellipse_l166_166596


namespace sum_of_first_15_odd_positives_l166_166802

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l166_166802


namespace sum_first_15_odd_integers_l166_166846

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l166_166846


namespace ladder_wood_sufficiency_l166_166290

theorem ladder_wood_sufficiency
  (total_wood : ℝ)
  (rung_length_in: ℝ)
  (rung_distance_in: ℝ)
  (ladder_height_ft: ℝ)
  (total_wood_ft : total_wood = 300)
  (rung_length_ft : rung_length_in = 18 / 12)
  (rung_distance_ft : rung_distance_in = 6 / 12)
  (ladder_height_ft : ladder_height_ft = 50) :
  (∃ wood_needed : ℝ, wood_needed ≤ total_wood ∧ total_wood - wood_needed = 162.5) :=
sorry

end ladder_wood_sufficiency_l166_166290


namespace club_officer_selection_l166_166326

theorem club_officer_selection:
  ∃ (f : Fin 4 → Fin 15), Function.Injective f ∧ (15 * 14 * 13 * 12 = 32760) :=
by
  -- Define our function that assigns each member to a unique office
  let f : Fin 4 → Fin 15 := sorry
  -- Show this function is injective (no member holds more than one office)
  have h1 : Function.Injective f := sorry
  -- Verify the number of ways to choose the officers
  have h2 : 15 * 14 * 13 * 12 = 32760 := by
    calc
      15 * 14 * 13 * 12 = 32760 : by norm_num
  -- Conclude that such an injective function exists and we have the necessary product
  exact ⟨f, h1, h2⟩

end club_officer_selection_l166_166326


namespace problem1_l166_166945

theorem problem1 (x : ℝ) : (2 * x - 1) * (2 * x - 3) - (1 - 2 * x) * (2 - x) = 2 * x^2 - 3 * x + 1 :=
by
  sorry

end problem1_l166_166945


namespace function_decreasing_l166_166189

variables {R : Type*} [linear_ordered_field R]

def p (f : R → R) : Prop := ∃ x1 x2 : R, (f x1 - f x2) * (x1 - x2) ≥ 0

def q : Prop := ∀ x y : R, x + y > 2 → (x > 1 ∨ y > 1)

theorem function_decreasing
  (f : R → R)
  (h_q : q)
  (h_not_p : ¬ p f):
  ∀ x1 x2 : R, x1 > x2 → f x1 < f x2 :=
by
  sorry

end function_decreasing_l166_166189


namespace positive_multiples_11_end_5_l166_166230

-- Define the conditions as a Lean statement
theorem positive_multiples_11_end_5 (h : ℕ → ℕ) (H : ∀ n : ℕ, h n = 11 * n) : 
  (∃ n, 0 < n ∧ h n < 2000 ∧ h n % 10 = 5) ↔ (finset.filter (λ x, h x < 2000) (finset.filter (λ x, h x % 10 = 5) (finset.range 200))).card = 18 :=
by {
  sorry
}

end positive_multiples_11_end_5_l166_166230


namespace man_l166_166515

-- Define the variables used in the conditions
variables (V_down V_s V_m V_up : ℝ)

-- Define the hypothesis as the given conditions
axiom h1 : V_down = 14   -- The downstream speed is 14 kmph
axiom h2 : V_s = 3       -- The speed of the stream is 3 kmph

-- Noncomputable definition is used since we are not giving an explicit computation
noncomputable def V_m := V_down - V_s

-- Final theorem to prove the upstream speed
theorem man's_speed_upstream : V_up = V_m - V_s → V_up = 8 :=
by
  sorry

end man_l166_166515


namespace a6_value_l166_166598

def sequence (n : ℕ) : ℤ :=
  if h : n > 0 then
    let seq_rec : ℕ → ℤ := λ n, if n = 1 then -1 else if h' : n > 1 then 
                                           2 * (sequence (n - 1)) - 1 
                                        else 0
    seq_rec n 
  else 0

theorem a6_value : sequence 6 = -63 :=
sorry

end a6_value_l166_166598


namespace fly_distance_from_ceiling_l166_166510

theorem fly_distance_from_ceiling : 
  ∀ (x y z : ℝ), 
    (x = 2) → (y = 7) → (sqrt (x*x + y*y + z*z) = 10) → 
    z = sqrt 47 :=
by
intros x y z h1 h2 h3
sorry

end fly_distance_from_ceiling_l166_166510


namespace B_N_C_collinear_l166_166128

-- Define the geometric entities and properties
def point := Type
def circle := point → Prop
def line := point → Prop

variables (O : point)
variables (C1 C2 C3 : circle)
variables (M N P A B C : point)

-- Hypotheses based on the conditions
axiom C1_C2_C3_intersect_at_O : C1 O ∧ C2 O ∧ C3 O
axiom M_is_intersection_C1_C2 : C1 M ∧ C2 M ∧ M ≠ O
axiom N_is_intersection_C2_C3 : C2 N ∧ C3 N ∧ N ≠ O
axiom P_is_intersection_C3_C1 : C3 P ∧ C1 P ∧ P ≠ O
axiom A_is_on_C1 : C1 A
axiom C_is_intersection_AP_C3 : ∃ (AP : line), AP A ∧ AP P ∧ C3 C ∧ AP C ∧ C ≠ P
axiom B_is_intersection_AM_C3 : ∃ (AM : line), AM A ∧ AM M ∧ C3 B ∧ AM B ∧ B ≠ M

-- The theorem to prove
theorem B_N_C_collinear : ∃ (line_BNC : line), line_BNC B ∧ line_BNC N ∧ line_BNC C :=
sorry

end B_N_C_collinear_l166_166128


namespace circle_center_and_radius_l166_166968

-- Define a circle in the plane according to the given equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x = 0

-- Define the center of the circle
def center (x : ℝ) (y : ℝ) : Prop := x = -2 ∧ y = 0

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 2

-- The theorem statement
theorem circle_center_and_radius :
  (∀ x y, circle_eq x y → center x y) ∧ radius 2 :=
sorry

end circle_center_and_radius_l166_166968


namespace problem_proof_l166_166183

noncomputable theory

open Nat

def a_seq (n : ℕ) (h : 0 < n) : ℚ :=
  match n with
  | 1 => 1
  | k + 1 => 1 - (1 / (4 * a_seq k (Nat.succ_pos k)))

def b_seq (n : ℕ) (h : 0 < n) : ℚ := 2 / (2 * a_seq n h - 1)

def C_seq (n : ℕ) (h : 0 < n) : ℚ := (4 * a_seq n h) / (n + 1)

def T_n (n : ℕ) (h : 0 < n) : ℚ :=
  ∑ i in finset.range n, (C_seq i.succ (pos_of_gt (succ_pos i)) * C_seq (i + 2) (lt_of_le_of_lt (succ_le_succ (le_of_lt (succ_pos i))) (succ_pos (i + 1))))

theorem problem_proof :
  ((∀ n : ℕ, 0 < n → b_seq (n+1) (succ_pos n) - b_seq n (Nat.pos_of_ne_zero (ne_of_gt (pos_of_gt n'.succ_pos))) = 2) ∧
  (∀ n : ℕ, 0 < n → a_seq n (Nat.pos_of_ne_zero (ne_of_gt (pos_of_gt n'.succ_pos))) = (n + 1)/(2 * n)) ∧
  (∃ m : ℕ, 0 < m ∧ ∀ n : ℕ, 0 < n → T_n n (succ_pos n) < 1 / (C_seq m (pos_of_gt (pos_of_gt n.succ_pos)) * C_seq (m + 1) (succ_pos (succ_pos m))) ∧
  (m = 3))) := sorry

end problem_proof_l166_166183


namespace boys_girls_dance_l166_166100

theorem boys_girls_dance (b g : ℕ) 
  (h : ∀ n, (n <= b) → (n + 7) ≤ g) 
  (hb_lasts : b + 7 = g) :
  b = g - 7 := by
  sorry

end boys_girls_dance_l166_166100


namespace range_of_a_l166_166209

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (x + 1)^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (hx1x2 : x1 ≠ x2)
  (h : f a x1 - f a x2 ≥ 4 * (x1 - x2)) : a ≥ 1 / 2 :=
begin
  sorry
end

end range_of_a_l166_166209


namespace math_problem_l166_166284

noncomputable def triangle_conditions (a b c A B C : ℝ) := 
  (2 * b - c) / a = (Real.cos C) / (Real.cos A) ∧ 
  a = Real.sqrt 5 ∧
  1 / 2 * b * c * (Real.sin A) = Real.sqrt 3 / 2

theorem math_problem (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) :
  A = π / 3 ∧ a + b + c = Real.sqrt 5 + Real.sqrt 11 :=
by
  sorry

end math_problem_l166_166284


namespace solution_l166_166965

noncomputable def greatestIntegerNotExceeding (x : ℝ) : ℝ := floor x
noncomputable def fractionalPart (x : ℝ) : ℝ := x - greatestIntegerNotExceeding x
noncomputable def f (x : ℝ) : ℝ := greatestIntegerNotExceeding x * fractionalPart x
noncomputable def g (x : ℝ) : ℝ := x - 1

noncomputable def length_of_interval (a b : ℝ) : ℝ := b - a

-- Defining the lengths for the inequality, equality, and inequality solution sets.
def d₁ := length_of_interval 0 1 -- Length of [0, 1)
def d₂ := length_of_interval 1 2 -- Length of [1, 2)
def d₃ := length_of_interval 2 2012 -- Length of [2, 2012]

theorem solution :
  d₁ = 1 ∧ d₂ = 1 ∧ d₃ = 2010 := 
by
  sorry

end solution_l166_166965


namespace area_of_square_EFGH_l166_166341

theorem area_of_square_EFGH {ABCD : Square ℝ} {EFGH : Square ℝ} (h₁ : side_length ABCD = 10) 
  (E F G H : Point ℝ) (h₂ : dist B E = 2) (h₃ : tilt_at_45_degrees EFGH) : 
  area EFGH = 100 - 4 * Real.sqrt 96 := 
sorry

end area_of_square_EFGH_l166_166341


namespace sum_of_first_15_odd_positives_l166_166804

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l166_166804


namespace length_of_OM_l166_166662

theorem length_of_OM {A B C D O M : Type*}
  (triangle_ABC : Triangle A B C)
  (angle_A_eq_60 : angle A B C = 60)
  (AD_is_angle_bisector : angle_bisector AD)
  (circum_radius_ADC_eq_sqrt3 : circumcircle_radius ADC = sqrt 3)
  (AB_length : distance A B = 1.5)
  (M_is_intersection_AD_BO : intersection M AD BO) :
  distance O M = sqrt 21 / 3 :=
begin
  sorry
end

end length_of_OM_l166_166662


namespace find_n_l166_166883

theorem find_n (n : ℕ) (h1 : Nat.lcm n 14 = 56) (h2 : Nat.gcd n 14 = 10) : n = 40 :=
by
  sorry

end find_n_l166_166883


namespace lcm_gcd_product_l166_166793

def a : ℕ := 20 -- Defining the first number as 20
def b : ℕ := 90 -- Defining the second number as 90

theorem lcm_gcd_product : Nat.lcm a b * Nat.gcd a b = 1800 := 
by 
  -- Computation and proof steps would go here
  sorry -- Replace with actual proof

end lcm_gcd_product_l166_166793


namespace cube_painted_faces_l166_166086

theorem cube_painted_faces (n : ℕ) (h1 : 2 < n) : (n-2)^3 = n-2 → n = 3 :=
by
  intro h
  have h_eq : (n-2)^2 = 1 := by
    sorry
  have h_solve : n-2 = 1 := by
    sorry
  exact Eq.symm (Nat.eq_of_sub_eq_add_left h_solve (show 1 + 2 = n), by sorry) 

end cube_painted_faces_l166_166086


namespace area_of_shaded_part_l166_166541

-- Define the given condition: area of the square
def area_of_square : ℝ := 100

-- Define the proof goal: area of the shaded part
theorem area_of_shaded_part : area_of_square / 2 = 50 := by
  sorry

end area_of_shaded_part_l166_166541


namespace Sandy_record_age_l166_166376

theorem Sandy_record_age :
  ∀ (current_length age_years : ℕ) (goal_length : ℕ) (growth_rate tenths_per_month : ℕ),
  goal_length = 26 ∧
  current_length = 2 ∧
  age_years = 12 ∧
  growth_rate = 10 *
  tenths_per_month / 10 →
  age_years + (goal_length - current_length) * 10 / growth_rate = 32 :=
by
  intros,
  sorry

end Sandy_record_age_l166_166376


namespace train_crossing_time_l166_166878

-- Given conditions
def length_of_train : ℝ := 50
def speed_kmh : ℝ := 360
def speed_ms : ℝ := speed_kmh * 1000 / 3600  -- converting km/hr to m/s

-- Goal: To prove the train crosses the electric pole in 0.5 seconds
theorem train_crossing_time : (length_of_train / speed_ms) = 0.5 :=
by
  sorry

end train_crossing_time_l166_166878


namespace even_number_of_black_squares_l166_166570

-- Definition: chessboard configuration
def is_chessboard (m n : ℕ) (color : ℕ × ℕ → bool) : Prop :=
  ∀ i j, i < m → j < n → color (i, j) = tt ∨ color (i, j) = ff

-- Definition: adjacency condition
def odd_black_neighbors (m n : ℕ) (color : ℕ × ℕ → bool) : Prop :=
  ∀ i j, color (i, j) = tt → odd (finset.card ((finset.univ.filter (λ ⟨di, dj⟩, 
    di = i ∧ (dj = j + 1 ∨ dj = j - 1) ∨ 
    dj = j ∧ (di = i + 1 ∨ di = i - 1)) ∧ 
    i < m ∧ j < n ∧ color (di, dj) = tt)))

-- Theorem: Number of black squares is even
theorem even_number_of_black_squares (m n : ℕ) (color : ℕ × ℕ → bool) 
  (h1 : is_chessboard m n color) (h2 : odd_black_neighbors m n color) : 
  even (finset.card (finset.univ.filter (λ ⟨i, j⟩, 
    i < m ∧ j < n ∧ color (i, j) = tt))) :=
by
  sorry

end even_number_of_black_squares_l166_166570


namespace shaded_area_proof_l166_166701

-- Define the side length of the square
def side_length_square : ℝ := 24

-- Define the number of circles
def num_circles : ℕ := 8

-- Define the radius of each circle
def radius_circle : ℝ := side_length_square / 4 / 2

-- Define the area of the square
def area_square : ℝ := side_length_square ^ 2

-- Define the area of one circle
def area_one_circle : ℝ := π * radius_circle ^ 2

-- Define the total area of the eight circles
def total_area_circles : ℝ := num_circles * area_one_circle

-- Define the shaded area
def shaded_area : ℝ := area_square - total_area_circles

-- Lean statement to prove the required shaded area
theorem shaded_area_proof : shaded_area = 576 - 72 * π := by
  sorry

end shaded_area_proof_l166_166701


namespace correct_conclusions_l166_166006

variable (a b c m : ℝ)
variable (y1 y2 : ℝ)

-- Conditions: 
-- Parabola y = ax^2 + bx + c, intersects x-axis at (-3,0) and (1,0)
-- a < 0
-- Points P(m-2, y1) and Q(m, y2) are on the parabola, y1 < y2

def parabola_intersects_x_axis_at_A_B : Prop :=
  ∀ x : ℝ, x = -3 ∨ x = 1 → a * x^2 + b * x + c = 0

def concavity_and_roots : Prop :=
  a < 0 ∧ b = 2 * a ∧ c = -3 * a

def conclusion_1 : Prop :=
  a * b * c < 0

def conclusion_2 : Prop :=
  b^2 - 4 * a * c > 0

def conclusion_3 : Prop :=
  3 * b + 2 * c = 0

def conclusion_4 : Prop :=
  y1 < y2 → m ≤ -1

-- Correct conclusions given the parabola properties
theorem correct_conclusions :
  concavity_and_roots a b c →
  parabola_intersects_x_axis_at_A_B a b c →
  conclusion_1 a b c ∨ conclusion_2 a b c ∨ conclusion_3 a b c ∨ conclusion_4 a b c :=
sorry

end correct_conclusions_l166_166006


namespace probability_satisfies_inequality_l166_166907

/-- Define the conditions for the points (x, y) -/
def within_rectangle (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5

def satisfies_inequality (x y : ℝ) : Prop :=
  x + 2 * y ≤ 6

/-- Compute the probability that a randomly selected point within the rectangle
also satisfies the inequality -/
theorem probability_satisfies_inequality : (∃ p : ℚ, p = 3 / 10) :=
sorry

end probability_satisfies_inequality_l166_166907


namespace sum_first_n_terms_l166_166620

def a (n : ℕ) : ℕ := n * 2^n

def S (n : ℕ) : ℕ := ∑ i in finset.range n, a (i + 1)

theorem sum_first_n_terms (n : ℕ) : S n = (n - 1) * 2^(n + 1) + 2 :=
by
  sorry

end sum_first_n_terms_l166_166620


namespace at_most_one_solution_iff_l166_166578

noncomputable def system_eq (a x y z : ℝ) : Prop :=
  (x^4 = yz - x^2 + a) ∧ (y^4 = zx - y^2 + a) ∧ (z^4 = xy - z^2 + a)

theorem at_most_one_solution_iff (a : ℝ) :
  (∀ x y z : ℝ, system_eq a x y z → (x, y, z) = (0, 0, 0)) ↔ a ≤ 0 :=
sorry

end at_most_one_solution_iff_l166_166578


namespace polynomial_condition_polynomial_value_when_exponential_is_one_l166_166182

-- Definition of the polynomial A given the condition
def polynomial_A (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 4

-- Prove the polynomial A satisfies the given condition
theorem polynomial_condition (x : ℝ) : polynomial_A x - (x-2)^2 = x * (x + 7) :=
by {
  calc polynomial_A x - (x-2)^2 = (2*x^2 + 3*x + 4) - (x^2 - 4*x + 4) : by rewrite polynomial_A x
  ... = x^2 + 7*x : by ring
  ... = x * (x + 7) : by ring
}

-- Prove that the value of the polynomial A when 3^(x+1) = 1 is 3
theorem polynomial_value_when_exponential_is_one {x : ℝ} (h : 3^(x + 1) = 1) : polynomial_A (-1) = 3 :=
by {
  have hx : x + 1 = 0 :=
    by {
      have h1 : 3 ^ (x + 1) = 3 ^ 0 := by rw [h]
      exact (by apply nat.pow_inj); apply_instance; norm_num; assumption
    },

  have h2 : x = -1 := by linarith,

  show polynomial_A (-1) = 3,
  calc polynomial_A (-1) = 2 * (-1)^2 + 3 * (-1) + 4 : by rw [polynomial_A]
  ... = 2 * 1 + (-3) + 4 : by norm_num
  ... = 3 : by norm_num
}

end polynomial_condition_polynomial_value_when_exponential_is_one_l166_166182


namespace maximum_number_of_intersections_l166_166776

theorem maximum_number_of_intersections
  (A B C D E : Point)
  (h_no_parallel : ∀ l1 l2 : Line, l1 ≠ l2 → ¬ (Parallel l1 l2))
  (h_no_perpendicular : ∀ l1 l2 : Line, l1 ≠ l2 → ¬ (Perpendicular l1 l2))
  (h_no_concurrent : ∀ P : Point, ∀ l1 l2 l3 : Line, (Concurrent l1 l2 l3 P) → False) :
  (number_of_intersections A B C D E = 315) :=
sorry

end maximum_number_of_intersections_l166_166776


namespace solve_xyz_l166_166730

def is_solution (x y z : ℕ) : Prop :=
  x * y + y * z + z * x = 2 * (x + y + z)

theorem solve_xyz (x y z : ℕ) :
  is_solution x y z ↔ (x = 1 ∧ y = 2 ∧ z = 4) ∨
                     (x = 1 ∧ y = 4 ∧ z = 2) ∨
                     (x = 2 ∧ y = 1 ∧ z = 4) ∨
                     (x = 2 ∧ y = 4 ∧ z = 1) ∨
                     (x = 2 ∧ y = 2 ∧ z = 2) ∨
                     (x = 4 ∧ y = 1 ∧ z = 2) ∨
                     (x = 4 ∧ y = 2 ∧ z = 1) := sorry

end solve_xyz_l166_166730


namespace num_senior_students_drawn_l166_166905

theorem num_senior_students_drawn
  (total_students : ℕ) (num_freshmen : ℕ) (num_sophomores : ℕ) (sample_size : ℕ)
  (h_total : total_students = 900) (h_freshmen : num_freshmen = 240)
  (h_sophomores : num_sophomores = 260) (h_sample : sample_size = 45) :
  let num_seniors := total_students - num_freshmen - num_sophomores
  (sampling_fraction : ℚ) := (sample_size : ℚ) / (total_students : ℚ)
  in num_seniors * sampling_fraction = 20 := 
by
  sorry

end num_senior_students_drawn_l166_166905


namespace cube_volume_and_diagonal_l166_166438

theorem cube_volume_and_diagonal (A : ℝ) (s : ℝ) (V : ℝ) (d : ℝ) 
  (h1 : A = 864)
  (h2 : 6 * s^2 = A)
  (h3 : V = s^3)
  (h4 : d = s * Real.sqrt 3) :
  V = 1728 ∧ d = 12 * Real.sqrt 3 :=
by 
  sorry

end cube_volume_and_diagonal_l166_166438


namespace coprime_pow_div_iff_l166_166310

theorem coprime_pow_div_iff (a b m n : ℕ) (h_coprime: Nat.coprime a b) (h_anb_ge_2 : a >= 2) (h_div: a^m + b^m ∣ a^n + b^n) : m ∣ n :=
by
  sorry

end coprime_pow_div_iff_l166_166310


namespace find_a_from_point_on_line_l166_166608

variable (a : ℝ)
variable (t : ℝ)

-- Given point and parametric line equations
def point_P : ℝ × ℝ := (2, 4)
def line_l_x (t : ℝ) : ℝ := 1 + t
def line_l_y (t : ℝ) : ℝ := 3 - a * t

-- Proposition stating the problem
theorem find_a_from_point_on_line (P_on_line : point_P = (line_l_x t, line_l_y t)) : a = -1 := 
by 
  sorry

end find_a_from_point_on_line_l166_166608


namespace log_expression_evaluation_l166_166571

theorem log_expression_evaluation :
  logb 0.5 0.125 + logb 2 (logb 3 (logb 4 64)) = 3 :=
sorry

end log_expression_evaluation_l166_166571


namespace not_algebraic_integer_of_sqrt2_div_2_l166_166562

-- Definitions based on conditions
def is_root (P : ℤ[X]) (x : ℂ) : Prop := P.isRoot x

def is_algebraic_integer (x : ℂ) : Prop :=
  ∃ (P : ℤ[X]), (P.degree > 0) ∧ (P.leading_coeff = 1) ∧ is_root P x

-- Statement of the problem
theorem not_algebraic_integer_of_sqrt2_div_2 :
  ¬ is_algebraic_integer (Complex.sqrt 2 / 2) :=
sorry

end not_algebraic_integer_of_sqrt2_div_2_l166_166562


namespace original_number_l166_166455

theorem original_number (x : ℝ) (h : 1000 * x = 3 / x) : x = (real.sqrt 30) / 100 :=
by sorry

end original_number_l166_166455


namespace g_of_f_at_3_eq_1902_l166_166693

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3 * x^2 + x + 2

theorem g_of_f_at_3_eq_1902 : g (f 3) = 1902 := by
  sorry

end g_of_f_at_3_eq_1902_l166_166693


namespace coin_heads_probability_l166_166012

theorem coin_heads_probability
    (prob_tails : ℚ := 1/2)
    (prob_specific_sequence : ℚ := 0.0625)
    (flips : ℕ := 4)
    (ht : prob_tails = 1 / 2)
    (hs : prob_specific_sequence = (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)) 
    : ∀ (p_heads : ℚ), p_heads = 1 - prob_tails := by
  sorry

end coin_heads_probability_l166_166012


namespace center_of_mass_is_incenter_of_medial_l166_166719

-- Definitions
variables {A B C A1 B1 C1 : point}
variables {a b c : ℝ}

-- Assuming A1, B1, C1 are midpoints of sides BC, CA, and AB respectively
def is_midpoint (P Q R : point) : Prop := dist P Q = dist P R

-- Center of mass definition for weighted points
def center_of_mass (M : Type) [metric_space M] {n : ℕ}
  (s : finset M) (m : M → ℝ) : M :=
  classical.some (finset.exists_mem_of_forall_sum_ne_zero m s)

-- Incenter of a triangle
def incenter (A B C : point) : point := sorry -- assuming we have some definition of incenter

-- Problem statement
theorem center_of_mass_is_incenter_of_medial (h_mid_A1 : is_midpoint A1 B C)
  (h_mid_B1 : is_midpoint B1 C A) (h_mid_C1 : is_midpoint C1 A B)
  (mass_A1 : a = dist B C) (mass_B1 : b = dist C A) (mass_C1 : c = dist A B):
  center_of_mass {A1, B1, C1} (λ p, if p = A1 then a else if p = B1 then b else c) = incenter A1 B1 C1 :=
sorry

end center_of_mass_is_incenter_of_medial_l166_166719


namespace problem1_problem2_l166_166113

theorem problem1 : sqrt 8 - 2 * sqrt 18 + sqrt 24 = -4 * sqrt 2 + 2 * sqrt 6 :=
by
  sorry

theorem problem2 : (sqrt (4 / 3) + sqrt 3) * sqrt 12 - sqrt 48 + sqrt 6 = 10 - 4 * sqrt 3 + sqrt 6 :=
by
  sorry

end problem1_problem2_l166_166113


namespace board_tiling_condition_l166_166168

-- Define the problem in Lean

theorem board_tiling_condition (n : ℕ) : 
  (∃ m : ℕ, n * n = m + 4 * m) ↔ (∃ k : ℕ, n = 5 * k ∧ n > 5) := by 
sorry

end board_tiling_condition_l166_166168


namespace original_number_l166_166459

theorem original_number (x : ℝ) (h : 1000 * x = 3 / x) : x = (real.sqrt 30) / 100 :=
by sorry

end original_number_l166_166459


namespace proof_problem_l166_166640

variable (a b c : ℝ)
variable (cond₁ : a > 0)
variable (cond₂ : b < 0)
variable (cond₃ : b > c)

theorem proof_problem :
  (a > 0) → (b < 0) → (b > c) →
  (  (a / c > a / b) ∧ 
     ((a - b) / (a - c) > b / c) ∧ 
     (a - c ≥ 2 * Real.sqrt ((a - b) * (b - c)))
  ) := 
by
  intros hc1 hc2 hc3
  split
  sorry
  split
  sorry
  sorry

end proof_problem_l166_166640


namespace number_of_possible_values_for_m_is_895_l166_166359

noncomputable def number_of_possible_values_for_m : ℕ :=
  let log_x := Real.log10 15
  let log_y := Real.log10 60
  fun (m : ℕ) =>
    log_x + log_y > Real.log10 m ∧
    log_x + Real.log10 m > log_y ∧
    log_y + Real.log10 m > log_x

theorem number_of_possible_values_for_m_is_895 :
  (∃ m : ℕ, number_of_possible_values_for_m m) → 
  (∃ n : ℕ, card {m : ℕ | number_of_possible_values_for_m m} = 895) :=
sorry

end number_of_possible_values_for_m_is_895_l166_166359


namespace profit_percentage_l166_166874

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 60) (h_selling : selling_price = 78) :
  ((selling_price - cost_price) / cost_price) * 100 = 30 :=
by
  sorry

end profit_percentage_l166_166874


namespace min_abs_sum_l166_166403

theorem min_abs_sum (x : ℝ) : 
  (∃ x, (∀ y, ∥ y + 1 ∥ + ∥ y + 3 ∥ + ∥ y + 6 ∥ ≥ ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥) ∧ 
        ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥ = 5) := 
sorry

end min_abs_sum_l166_166403


namespace find_original_number_l166_166464

noncomputable def original_number (x : ℝ) : Prop :=
  (0 < x) ∧ (1000 * x = 3 / x)

theorem find_original_number : ∃ x : ℝ, original_number x ∧ x = real.sqrt 30 / 100 :=
by
  sorry

end find_original_number_l166_166464


namespace hyperbola_eccentricity_proof_l166_166621

noncomputable def hyperbola_eccentricity (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) : ℝ :=
sqrt 3

theorem hyperbola_eccentricity_proof (a b c : ℝ)
  (ha : a > 0) (hb : b > 0) (h1 : a = c / sqrt 3) (h2 : b = sqrt (a^2 - c^2)):
  hyperbola_eccentricity a b c ha hb (by
  simp only [true_and, eq_self_iff_true]; sorry) = sqrt 3:=
by sorry

end hyperbola_eccentricity_proof_l166_166621


namespace alpha_beta_squared_l166_166239

section
variables (α β : ℝ)
-- Given conditions
def is_root (a b : ℝ) : Prop :=
  a + b = 2 ∧ a * b = -1 ∧ (∀ x : ℝ, x^2 - 2 * x - 1 = 0 → x = a ∨ x = b)

-- The theorem to prove
theorem alpha_beta_squared (h: is_root α β) : α^2 + β^2 = 6 :=
sorry
end

end alpha_beta_squared_l166_166239


namespace min_filtrations_to_meet_market_requirements_l166_166508

noncomputable def minimum_filtrations_required (p0 : ℝ) (f : ℝ) (pm : ℝ) : ℕ :=
let n := - (Real.log pm / Real.log f) in
nat_ceil n

theorem min_filtrations_to_meet_market_requirements :
 (minimum_filtrations_required 0.01 (2 / 3) 0.001) = 6 :=
by
  sorry

end min_filtrations_to_meet_market_requirements_l166_166508


namespace even_function_monotonic_decrease_range_of_function_l166_166869

def f (x : ℝ) : ℝ := (2 - |x|) / (1 + |x|)

theorem even_function : ∀ x : ℝ, f x = f (-x) :=
by sorry

theorem monotonic_decrease : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x :=
by sorry

theorem range_of_function : ∀ y : ℝ, y ∈ set.Icc (-1 : ℝ) 2 ↔ ∃ x : ℝ, f x = y :=
by sorry

end even_function_monotonic_decrease_range_of_function_l166_166869


namespace sum_of_first_15_odd_integers_l166_166828

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l166_166828


namespace sandy_age_when_record_l166_166373

noncomputable def calc_age (record_length current_length monthly_growth_rate age : ℕ) : ℕ :=
  let yearly_growth_rate := monthly_growth_rate * 12
  let needed_length := record_length - current_length
  let years_needed := needed_length / yearly_growth_rate
  age + years_needed

theorem sandy_age_when_record (record_length current_length monthly_growth_rate age : ℕ) :
  record_length = 26 →
  current_length = 2 →
  monthly_growth_rate = 1 →
  age = 12 →
  calc_age record_length current_length monthly_growth_rate age = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  unfold calc_age
  simp
  sorry

end sandy_age_when_record_l166_166373


namespace sum_first_15_odd_integers_l166_166820

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l166_166820


namespace total_toothpicks_480_l166_166784

/- Define the number of toothpicks per side -/
def toothpicks_per_side : ℕ := 15

/- Define the number of horizontal lines in the grid -/
def horizontal_lines (sides : ℕ) : ℕ := sides + 1

/- Define the number of vertical lines in the grid -/
def vertical_lines (sides : ℕ) : ℕ := sides + 1

/- Define the total number of toothpicks used -/
def total_toothpicks (sides : ℕ) : ℕ :=
  (horizontal_lines sides * toothpicks_per_side) + (vertical_lines sides * toothpicks_per_side)

/- Theorem statement: Prove that for a grid with 15 toothpicks per side, the total number of toothpicks is 480 -/
theorem total_toothpicks_480 : total_toothpicks 15 = 480 :=
  sorry

end total_toothpicks_480_l166_166784


namespace sequence_periodicity_l166_166625

noncomputable def a : ℕ → ℚ
| 0       => 0
| (n + 1) => (a n - 2) / ((5/4) * a n - 2)

theorem sequence_periodicity : a 2017 = 0 := by
  sorry

end sequence_periodicity_l166_166625


namespace sequence_periodic_sum_of_sequence_l166_166364

def sequence (a : ℕ → ℕ) : ℕ → ℕ
| 0       := a(0)
| (n + 1) := if a(n) > 1 then a(n) - 1 else 2 * a(n)

def S (n : ℕ) : ℕ := 
if n % 2 = 0 then (3 * n + 4) / 2 
else (3 * (n + 1)) / 2

theorem sequence_periodic :
  sequence (λ n, match n with
    | 0       := 3
    | 1       := 2
    | _ := if n % 2 = 0 then 2 else 1) 2016 = 2 :=
sorry

theorem sum_of_sequence (n : ℕ) :
  S n = if n % 2 = 0 then (3 * n + 4) / 2 else (3 * (n + 1)) / 2 :=
sorry

end sequence_periodic_sum_of_sequence_l166_166364


namespace greatest_integer_b_l166_166151

theorem greatest_integer_b (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ 0) ↔ b ≤ 6 := 
by
  sorry

end greatest_integer_b_l166_166151


namespace midpoint_product_eq_three_l166_166400

theorem midpoint_product_eq_three : 
  let p1 := (4, -2)
  let p2 := (-2, 8)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 * midpoint.2) = 3 :=
by
  let p1 := (4, -2)
  let p2 := (-2, 8)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  have : midpoint = (1, 3) := sorry
  rw this
  exact sorry

end midpoint_product_eq_three_l166_166400


namespace sum_of_first_15_odd_integers_l166_166826

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l166_166826


namespace subtract_complex_numbers_l166_166397

theorem subtract_complex_numbers 
  (a b : ℂ) 
  (h₁ : a = 5 - 3 * complex.I) 
  (h₂ : b = 4 + complex.I) : 
  a - 3 * b = -7 - 6 * complex.I := 
by 
  sorry

end subtract_complex_numbers_l166_166397


namespace minimal_difference_big_small_sum_l166_166656

theorem minimal_difference_big_small_sum :
  ∀ (N : ℕ), N > 0 → ∃ (S : ℕ), 
  S = (N * (N - 1) * (2 * N + 5)) / 6 :=
  by 
    sorry

end minimal_difference_big_small_sum_l166_166656


namespace exists_cubic_polynomial_l166_166989

theorem exists_cubic_polynomial (Q : Polynomial ℚ) (h_nonconstant : 0 ≠ degree Q) :
  degree Q = 3 ∧ Q.comp(Q) = ((X^3 + X^2 + X + 1) * Q) :=
sorry

end exists_cubic_polynomial_l166_166989


namespace sequence_formulas_l166_166314

theorem sequence_formulas (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ) :
  (∀ n, S n = n^2 + 1) →
  (b 2 = 2) →
  (b 5 = 16) →
  (∀ n, S n = a n + S (n - 1) ∨ n = 1) →
  (∀ n, a n = (if n = 1 then 2 else 2n - 1)) →
  (∀ n, b n = 2^(n-1)) →
  T 1 = 2 ∧ (∀ n, T n = (if n = 1 then 2 else (2n - 3) * 2^n + 4)) := 
sorry

end sequence_formulas_l166_166314


namespace problem1_problem2_l166_166111

-- Problem 1 Statement
theorem problem1 :
  sqrt 8 - 2 * sqrt 18 + sqrt 24 = -4 * sqrt 2 + 2 * sqrt 6 :=
by
  sorry

-- Problem 2 Statement
theorem problem2 :
  (sqrt (4 / 3) + sqrt 3) * sqrt 12 - sqrt 48 + sqrt 6 = 10 - 4 * sqrt 3 + sqrt 6 :=
by
  sorry

end problem1_problem2_l166_166111


namespace min_value_of_f_l166_166408

def f (x : ℝ) := abs (x + 1) + abs (x + 3) + abs (x + 6)

theorem min_value_of_f : ∃ (x : ℝ), f x = 5 :=
by
  use -3
  simp [f]
  sorry

end min_value_of_f_l166_166408


namespace total_painting_area_is_590_l166_166895

def width : ℝ := 10
def length : ℝ := 13
def height : ℝ := 5

def area_of_wall (w l h : ℝ) : ℝ := 2 * (w * h) + 2 * (l * h)
def area_of_ceiling  (w l : ℝ) : ℝ := w * l

-- Total painting area calculation
def total_area_to_paint (w l h : ℝ) : ℝ :=
  2 * area_of_wall w l h + area_of_ceiling w l

theorem total_painting_area_is_590:
  total_area_to_paint width length height = 590 := by
  sorry

end total_painting_area_is_590_l166_166895


namespace phase_shift_of_cosine_l166_166991

theorem phase_shift_of_cosine (x : ℝ) : 
  ∀ x, 5 * cos (x - π/3 + π/6) = 5 * cos (x - π/6) :=
by sorry

end phase_shift_of_cosine_l166_166991


namespace andrei_club_visits_l166_166061

theorem andrei_club_visits (d c : ℕ) (h : 15 * d + 11 * c = 115) : d + c = 9 :=
by
  sorry

end andrei_club_visits_l166_166061


namespace probability_of_at_least_one_boy_and_one_girl_is_correct_l166_166926

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  (1 - ((1/2)^4 + (1/2)^4))

theorem probability_of_at_least_one_boy_and_one_girl_is_correct : 
  probability_at_least_one_boy_and_one_girl = 7/8 :=
by
  sorry

end probability_of_at_least_one_boy_and_one_girl_is_correct_l166_166926


namespace lcm_of_two_numbers_l166_166492
-- Importing the math library

-- Define constants and variables
variables (A B LCM HCF : ℕ)

-- Given conditions
def product_condition : Prop := A * B = 17820
def hcf_condition : Prop := HCF = 12
def lcm_condition : Prop := LCM = Nat.lcm A B

-- Theorem to prove
theorem lcm_of_two_numbers : product_condition A B ∧ hcf_condition HCF →
                              lcm_condition A B LCM →
                              LCM = 1485 := 
by
  sorry

end lcm_of_two_numbers_l166_166492


namespace sum_of_first_n_odd_numbers_l166_166841

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l166_166841


namespace distance_ratio_of_chords_l166_166025

open Real EuclideanGeometry

variables {r r1 : ℝ} {T1 T2 K O O1 : Point}

-- Assuming circles are defined with centers O and O1 and radii r and r1 respectively, and touching each other externally at K
variable (c1 : Circle O r)
variable (c2 : Circle O1 r1)

-- Also assuming existence of points T1 and T2 on these circles where external tangents touch
axiom touch_T1 : T1 ∈ tangent_points c1 K
axiom touch_T2 : T2 ∈ tangent_points c2 K

-- Declaring distances from the chords to the centers
noncomputable def dist_O_T1_path : ℝ := distance_to_line O (line_through K T1)
noncomputable def dist_O1_T2_path : ℝ := distance_to_line O1 (line_through K T2)

theorem distance_ratio_of_chords :
  dist_O_T1_path c1 = dist_O1_T2_path c2 →
  dist_O_T1_path / dist_O1_T2_path = (distance K T1) ^ 3 / (distance K T2) ^ 3 :=
by
  intros h
  sorry

end distance_ratio_of_chords_l166_166025


namespace min_abs_sum_l166_166405

theorem min_abs_sum (x : ℝ) : 
  (∃ x, (∀ y, ∥ y + 1 ∥ + ∥ y + 3 ∥ + ∥ y + 6 ∥ ≥ ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥) ∧ 
        ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥ = 5) := 
sorry

end min_abs_sum_l166_166405


namespace exists_lucky_integer_nearby_l166_166075

-- Definition of a lucky integer
def is_lucky (x : ℕ) : Prop :=
  x % 7 = 0 ∧ (x.digits 10).sum % 7 = 0

-- Main proof statement
theorem exists_lucky_integer_nearby (n : ℕ) : ∃ l : ℕ, is_lucky(l) ∧ |n - l| ≤ 70 :=
sorry

end exists_lucky_integer_nearby_l166_166075


namespace sandy_will_be_32_l166_166380

  -- Define the conditions
  def world_record_length : ℝ := 26
  def sandy_current_length : ℝ := 2
  def monthly_growth_rate : ℝ := 0.1
  def sandy_current_age : ℝ := 12

  -- Define the annual growth rate calculation
  def annual_growth_rate : ℝ := monthly_growth_rate * 12

  -- Define total growth needed
  def total_growth_needed : ℝ := world_record_length - sandy_current_length

  -- Define the years needed to grow the fingernails to match the world record
  def years_needed : ℝ := total_growth_needed / annual_growth_rate

  -- Define Sandy's age when she achieves the world record length
  def sandy_age_when_record_achieved : ℝ := sandy_current_age + years_needed

  -- The statement we want to prove
  theorem sandy_will_be_32 :
    sandy_age_when_record_achieved = 32 :=
  by
    -- Placeholder proof
    sorry
  
end sandy_will_be_32_l166_166380


namespace prod_exp_difference_l166_166555

noncomputable def complex_roots (n : ℕ) : Finset ℂ :=
  Finset.univ.image (λ k, Complex.exp (2 * k * Complex.pi * Complex.I / n))

theorem prod_exp_difference :
  (∏ k in Finset.range 10, ∏ j in Finset.range 8, (Complex.exp (2 * j * Complex.pi * Complex.I / 9) - Complex.exp (2 * k * Complex.pi * Complex.I / 11))) = 1 :=
by sorry

end prod_exp_difference_l166_166555


namespace sum_of_first_n_odd_numbers_l166_166837

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l166_166837


namespace find_N_divisible_by_18_and_within_bounds_l166_166577

theorem find_N_divisible_by_18_and_within_bounds :
  ∃ (N : ℕ), (N > 0) ∧ (N % 18 = 0) ∧ (676 ≤ N ∧ N ≤ 686) ∧ N = 684 :=
by
  use 684
  split
  sorry

end find_N_divisible_by_18_and_within_bounds_l166_166577


namespace max_non_attacking_rooks_l166_166171

-- Definitions based on given conditions
def grid : Type := fin 12 × fin 12  -- 12x12 grid

def is_cutout_square (pos : grid) : Prop :=
  (2 ≤ pos.1 ∧ pos.1 ≤ 5) ∧ (2 ≤ pos.2 ∧ pos.2 ≤ 5)

def can_place_rook (pos1 pos2 : grid) : Prop :=
  ¬is_cutout_square pos1 ∧ ¬is_cutout_square pos2 ∧
  (pos1.1 ≠ pos2.1 ∧ pos1.2 ≠ pos2.2)

-- Main theorem to prove the number of non-attacking rooks
theorem max_non_attacking_rooks : ∃ (positions : list grid), 
  ∀ pos1 pos2 ∈ positions, pos1 ≠ pos2 → can_place_rook pos1 pos2 ∧ 
  list.length positions = 14 :=
sorry

end max_non_attacking_rooks_l166_166171


namespace platform_length_l166_166053

noncomputable def length_of_platform 
  (L_t : ℕ) -- length of train in meters
  (s : ℕ)  -- speed of train in kmph
  (t : ℕ)  -- time to cross the platform in seconds
  : ℕ := 
  let speed_m_per_s := s * 1000 / 3600 in
  let total_distance := speed_m_per_s * t in
  total_distance - L_t

theorem platform_length : 
  ∀ (L_t s t : ℕ), 
  L_t = 175 ∧ s = 36 ∧ t = 40 → 
  length_of_platform L_t s t = 225 :=
by
  intros L_t s t h
  obtain ⟨h1, h2, h3⟩ := h
  simp [length_of_platform, h1, h2, h3]
  norm_num
  sorry

end platform_length_l166_166053


namespace jane_oldest_babysat_age_l166_166055

theorem jane_oldest_babysat_age
  (start_age : ℕ) -- Jane started baby-sitting at 16 years old
  (current_age : ℕ) -- Jane is currently 32 years old
  (stop_years_ago : ℕ) -- Jane stopped baby-sitting 10 years ago
  (half_age_condition : ∀ (jane_age : ℕ), jane_age / 2 ≤ jane_age) -- Children no more than half her age

  (jane_start_babysitting : start_age = 16)
  (jane_current_age : current_age = 32)
  (jane_stop_years_ago : stop_years_ago = 10)
  (jane_stopped_age : current_age - stop_years_ago = 22)
  (oldest_child_age_when_stopped : 22 / 2 = 11) :
  11 + 10 = 21 :=
begin
  sorry,
end

end jane_oldest_babysat_age_l166_166055


namespace exists_consecutive_numbers_divisible_by_3_5_7_l166_166021

theorem exists_consecutive_numbers_divisible_by_3_5_7 :
  ∃ (a : ℕ), 100 ≤ a ∧ a ≤ 200 ∧
    a % 3 = 0 ∧ (a + 1) % 5 = 0 ∧ (a + 2) % 7 = 0 :=
by
  sorry

end exists_consecutive_numbers_divisible_by_3_5_7_l166_166021


namespace common_difference_is_4_l166_166300

variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}
variable {a_4 a_5 S_6 : ℝ}
variable {d : ℝ}

-- Definitions of conditions given in the problem
def a4_cond : a_4 = a_n 4 := sorry
def a5_cond : a_5 = a_n 5 := sorry
def sum_six : S_6 = (6/2) * (2 * a_n 1 + 5 * d) := sorry
def term_sum : a_4 + a_5 = 24 := sorry

-- Proof statement
theorem common_difference_is_4 : d = 4 :=
by
  sorry

end common_difference_is_4_l166_166300


namespace remainder_of_4672_div_34_is_14_l166_166794

theorem remainder_of_4672_div_34_is_14 :
  ∃ r q : ℕ, r = 4672 % 34 ∧ r = 14 :=
by
  use (4672 % 34)
  use (4672 / 34)
  simp
  sorry

end remainder_of_4672_div_34_is_14_l166_166794


namespace angle_EGD_of_acute_triangle_altitudes_intersect_at_orthocenter_l166_166091

theorem angle_EGD_of_acute_triangle_altitudes_intersect_at_orthocenter
  {A B C G : Type*} 
  [triangle : triangle A B C] 
  (h1 : altitude A B C G) 
  (h2 : altitude B A C G) 
  (h3 : angle B A C = 54) 
  (h4 : angle C A B = 62) :
  angle (altitude_meeting_point _ h1 h2) A G G C = 26 :=
begin
  -- sorry to skip the proof.
  sorry,
end

end angle_EGD_of_acute_triangle_altitudes_intersect_at_orthocenter_l166_166091


namespace problem1_problem2_l166_166110

-- Problem 1 Statement
theorem problem1 :
  sqrt 8 - 2 * sqrt 18 + sqrt 24 = -4 * sqrt 2 + 2 * sqrt 6 :=
by
  sorry

-- Problem 2 Statement
theorem problem2 :
  (sqrt (4 / 3) + sqrt 3) * sqrt 12 - sqrt 48 + sqrt 6 = 10 - 4 * sqrt 3 + sqrt 6 :=
by
  sorry

end problem1_problem2_l166_166110


namespace log_a_b_iff_a_minus_1_b_minus_1_pos_l166_166501

theorem log_a_b_iff_a_minus_1_b_minus_1_pos (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ 1) :
  (log a b > 0) ↔ ((a - 1) * (b - 1) > 0) := 
sorry

end log_a_b_iff_a_minus_1_b_minus_1_pos_l166_166501


namespace pair_not_equal_neg_48_l166_166566

theorem pair_not_equal_neg_48 (a b c d e : ℝ) :
    (a = 4 ∧ b = -12 ∧ a * b = -48) ∧ 
    (c = -2 ∧ d = 24 ∧ c * d = -48) ∧ 
    (e = 6 ∧ d = -8 ∧ e * d = -48) ∧ 
    (e = -1/2 ∧ d = 96 ∧ e * d = -48) ∧ 
    (¬ (3 * 16 = -48))
:=
begin
    sorry
end

end pair_not_equal_neg_48_l166_166566


namespace intersection_of_sets_l166_166604

theorem intersection_of_sets :
  let A := { x : ℝ | x^2 - 2 * x < 0 },
      B := { x : ℝ | 1 < x ∧ x < 3 } in
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_of_sets_l166_166604


namespace find_tangent_value_l166_166622

noncomputable def tangent_value (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧
  (1 / (x₀ + a) = 1)

theorem find_tangent_value : tangent_value 2 :=
  sorry

end find_tangent_value_l166_166622


namespace geometric_series_sum_l166_166155

theorem geometric_series_sum :
  (1 + ∑ i in Finset.range 2022, 5^(i+1)) = (5^2023 - 1) / 4 := 
sorry

end geometric_series_sum_l166_166155


namespace part1_lP_part1_lQ_part2_min_lA_l166_166627

open Set

-- Define the set A and the function l(A)
def A (n : ℕ) := {a : ℝ | ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ a = f i}  -- assuming f(i) gives the elements of {a_1, a_2, ..., a_n}
def l (A : Set ℝ) := (finite {x : ℝ | ∃ a b ∈ A, a < b ∧ x = a + b}).toFinset.card

-- Specific sets P and Q
def P := {2, 4, 6, 8}
def Q := {2, 4, 8, 16}

-- Statements to prove
theorem part1_lP : l P = 5 := 
  by
  sorry

theorem part1_lQ : l Q = 6 := 
  by
  sorry

-- General set A for minimum l(A)
def A (n : ℕ) := {a : ℝ | ∃i : ℕ, 1 ≤ i ∧ i ≤ n ∧ a = f(i)} -- f as placeholder for actual sequence function

theorem part2_min_lA (n : ℕ) (h : n > 2) : 
  ∀ (a : ℕ → ℝ), (∀ i, a i ∈ A n) → l (A n) ≥ 2 * n - 3 :=
  by
  sorry

end part1_lP_part1_lQ_part2_min_lA_l166_166627


namespace no_such_six_tuples_exist_l166_166603

theorem no_such_six_tuples_exist :
  ∀ (a b c x y z : ℕ),
    1 ≤ c → c ≤ b → b ≤ a →
    1 ≤ z → z ≤ y → y ≤ x →
    2 * a + b + 4 * c = 4 * x * y * z →
    2 * x + y + 4 * z = 4 * a * b * c →
    False :=
by
  intros a b c x y z h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end no_such_six_tuples_exist_l166_166603


namespace sum_of_ages_3_years_hence_l166_166369

theorem sum_of_ages_3_years_hence (A B C D S : ℝ) (h1 : A = 2 * B) (h2 : C = A / 2) (h3 : D = A - C) (h_sum : A + B + C + D = S) : 
  (A + 3) + (B + 3) + (C + 3) + (D + 3) = S + 12 :=
by sorry

end sum_of_ages_3_years_hence_l166_166369


namespace value_of_x25_l166_166757

theorem value_of_x25 (x : Fin 100 → ℝ)
  (h : ∀ k : Fin 100, x k < (∑ i, if i ≠ k then x i else 0) + k.val + 1) :
  x ⟨24, by decide⟩ = 650 / 49 := 
sorry

end value_of_x25_l166_166757


namespace olaf_total_cars_l166_166319

noncomputable def olaf_initial_cars : ℕ := 150
noncomputable def uncle_cars : ℕ := 5
noncomputable def grandpa_cars : ℕ := 2 * uncle_cars
noncomputable def dad_cars : ℕ := 10
noncomputable def mum_cars : ℕ := dad_cars + 5
noncomputable def auntie_cars : ℕ := 6
noncomputable def liam_cars : ℕ := dad_cars / 2
noncomputable def emma_cars : ℕ := uncle_cars / 3
noncomputable def grandma_cars : ℕ := 3 * auntie_cars

noncomputable def total_gifts : ℕ := 
  grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars + liam_cars + emma_cars + grandma_cars

noncomputable def total_cars_after_gifts : ℕ := olaf_initial_cars + total_gifts

theorem olaf_total_cars : total_cars_after_gifts = 220 := by
  sorry

end olaf_total_cars_l166_166319


namespace q_evaluation_at_14_l166_166688

-- Define polynomial q(x) and requisite conditions
def q (x : ℝ) : ℝ := -6/77 * x^2 + x + 24/77

theorem q_evaluation_at_14 : q(14) = -74/77 :=
by
  -- Placeholder for proof
  sorry

end q_evaluation_at_14_l166_166688


namespace original_number_l166_166482

theorem original_number :
  ∃ x : ℝ, 0 < x ∧ (move_decimal_point x 3 = 3 / x) ∧ x = sqrt 30 / 100 :=
sorry

noncomputable def move_decimal_point (x : ℝ) (places : ℕ) : ℝ := x * 10^places

end original_number_l166_166482


namespace original_number_l166_166460

theorem original_number (x : ℝ) (h : 1000 * x = 3 / x) : x = (real.sqrt 30) / 100 :=
by sorry

end original_number_l166_166460


namespace solve_for_x_l166_166440

theorem solve_for_x (x : ℕ) : (1 : ℚ) / 2 = x / 8 → x = 4 := by
  sorry

end solve_for_x_l166_166440


namespace difference_of_numbers_l166_166769

variable (x y : ℝ)

theorem difference_of_numbers (h1 : x + y = 10) (h2 : x - y = 19) (h3 : x^2 - y^2 = 190) : x - y = 19 :=
by sorry

end difference_of_numbers_l166_166769


namespace smallest_value_of_abs_sum_l166_166419

theorem smallest_value_of_abs_sum : ∃ x : ℝ, (x = -3) ∧ ( ∀ y : ℝ, |y + 1| + |y + 3| + |y + 6| ≥ 5 ) :=
by
  use -3
  split
  . exact rfl
  . intro y
    have h1 : |y + 1| + |y + 3| + |y + 6| = sorry,
    sorry

end smallest_value_of_abs_sum_l166_166419


namespace find_original_number_l166_166466

noncomputable def original_number (x : ℝ) : Prop :=
  (0 < x) ∧ (1000 * x = 3 / x)

theorem find_original_number : ∃ x : ℝ, original_number x ∧ x = real.sqrt 30 / 100 :=
by
  sorry

end find_original_number_l166_166466


namespace geometric_series_sum_l166_166863

theorem geometric_series_sum :
  ∀ (a r : ℚ) (n : ℕ), 
  a = 1 / 5 → 
  r = -1 / 5 → 
  n = 6 →
  (a - a * r^n) / (1 - r) = 1562 / 9375 :=
by 
  intro a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end geometric_series_sum_l166_166863


namespace tourist_times_l166_166919

theorem tourist_times 
    (dist_walk : ℕ)
    (dist_row : ℕ)
    (t_walk : ℕ)
    (h1 : dist_walk = 10)
    (h2 : dist_row = 90)
    (h3 : t_walk + 4 = t_row)
    (h4 : (10 * (t_walk + 4)) = (90 * t_walk / (t_walk + 4))
    : t_walk = 2 ∧ t_row = 6 :=
by {
  sorry
}

end tourist_times_l166_166919


namespace shaina_chocolate_amount_l166_166293

variable (total_chocolate : ℚ) (num_piles : ℕ) (fraction_kept : ℚ)
variable (eq_total_chocolate : total_chocolate = 72 / 7)
variable (eq_num_piles : num_piles = 6)
variable (eq_fraction_kept : fraction_kept = 1 / 3)

theorem shaina_chocolate_amount :
  (total_chocolate / num_piles) * (1 - fraction_kept) = 8 / 7 :=
by
  sorry

end shaina_chocolate_amount_l166_166293


namespace cos_B_value_l166_166649

noncomputable def cosB_in_triangle (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) (h_sum : A + B + C = π) (h_ratio : sin A / sin B = 3 / 4) (h_ratio' : sin A / sin C = 3 / 6) : ℝ :=
  real.cos B

theorem cos_B_value (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) (h_sum : A + B + C = π) (h_ratio : sin A / sin B = 3 / 4) (h_ratio' : sin A / sin C = 3 / 6) :
  cosB_in_triangle A B C hA hB hC h_sum h_ratio h_ratio' = 29 / 36 :=
by
  sorry

end cos_B_value_l166_166649


namespace relationship_l166_166194

variable a b c : ℝ

-- Conditions from the problem
def a_def := a = 0.8 ^ 0.8
def b_def := b = 0.8 ^ 1.2
def c_def := c = 1.2 ^ 0.8

-- Theorem that needs to be proved
theorem relationship : 
  a = 0.8 ^ 0.8 → 
  b = 0.8 ^ 1.2 → 
  c = 1.2 ^ 0.8 → 
  c > a ∧ a > b :=
by
  intros ha hb hc
  sorry

end relationship_l166_166194


namespace alyssa_picked_42_l166_166531

variable (totalPears nancyPears : ℕ)
variable (total_picked : totalPears = 59)
variable (nancy_picked : nancyPears = 17)

theorem alyssa_picked_42 (h1 : totalPears = 59) (h2 : nancyPears = 17) :
  totalPears - nancyPears = 42 :=
by
  sorry

end alyssa_picked_42_l166_166531


namespace billy_initial_crayons_l166_166103

theorem billy_initial_crayons (left_crayons eaten_crayons : ℕ) (h1 : left_crayons = 10) (h2 : eaten_crayons = 52) :
  left_crayons + eaten_crayons = 62 :=
by
  rw [h1, h2]
  exact rfl

end billy_initial_crayons_l166_166103


namespace sum_of_first_15_odd_integers_l166_166856

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l166_166856


namespace solution_set_of_inequality_l166_166124

variable (f : ℝ → ℝ)

-- Conditions
def is_odd_function := ∀ x : ℝ, f(-x) = -f(x)
def is_increasing_on_pos := ∀ x y : ℝ, 0 < x → x < y → f(x) < f(y)
def f_minus_3_eq_zero := f (-3) = 0

-- Goal
theorem solution_set_of_inequality :
  is_odd_function f →
  is_increasing_on_pos f →
  f_minus_3_eq_zero f →
  {x : ℝ | x * f(x) < 0} = {x : ℝ | x ∈ (-∞ : set ℝ) (−3) ∪ (3, +∞)} :=
by
  sorry

end solution_set_of_inequality_l166_166124


namespace find_a_exactly_two_solutions_l166_166985

theorem find_a_exactly_two_solutions :
  (∀ x y : ℝ, |x - 6 - y| + |x - 6 + y| = 12 ∧ (|x| - 6)^2 + (|y| - 8)^2 = a) ↔ (a = 4 ∨ a = 100) :=
sorry

end find_a_exactly_two_solutions_l166_166985


namespace ellipse_standard_equation_l166_166748

theorem ellipse_standard_equation (focal_distance sum_distances : ℝ)
  (h_focal_distance : focal_distance = 8)
  (h_sum_distances : sum_distances = 10) :
  (∃ a b : ℝ, a = 5 ∧ b = 3 ∧
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1 ∨ y^2 / 25 + x^2 / 9 = 1))) :=
by
  use [5, 3]
  simp [h_focal_distance, h_sum_distances]
  sorry

end ellipse_standard_equation_l166_166748


namespace max_value_trig_l166_166304

noncomputable def find_max_value (a b c : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2 + 2 * c^2)

theorem max_value_trig (a b c : ℝ) :  
  ∃ θ : ℝ, a * real.cos θ + b * real.sin θ + c * real.cos (2 * θ) ≤ find_max_value a b c :=
by
  sorry

end max_value_trig_l166_166304


namespace probability_of_at_least_one_boy_and_one_girl_is_correct_l166_166925

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  (1 - ((1/2)^4 + (1/2)^4))

theorem probability_of_at_least_one_boy_and_one_girl_is_correct : 
  probability_at_least_one_boy_and_one_girl = 7/8 :=
by
  sorry

end probability_of_at_least_one_boy_and_one_girl_is_correct_l166_166925


namespace faster_overtake_30_seconds_l166_166028

noncomputable def faster_overtake_time
  (length : ℝ) (V_s : ℝ) (V_f : ℝ) (time_half_speed : ℝ)
  (half_speed : ℝ) (total_distance : ℝ)
  (relative_speed_half : ℝ) : ℝ :=
  total_distance / relative_speed_half

theorem faster_overtake_30_seconds :
  ∀ (length : ℝ) (V_s V_f : ℝ),
  V_s = 32 ∧ length = 120 ∧ V_f = 40 ∧
  faster_overtake_time length V_s V_f 10
    (V_s / 2) (2 * length)
    (V_f - V_s / 2) = 30 :=
by
  intro length V_s V_f
  intro h
  sorry

end faster_overtake_30_seconds_l166_166028


namespace find_number_l166_166074

theorem find_number : ∃ n : ℚ, n = (n - 5) * 4 ∧ n = 20 / 3 :=
by {
  use 20 / 3,
  split,
  { -- verify the condition
    rw (show 20 / 3 = (20 / 3 - 5) * 4, by simp [sub_eq_add_neg, mul_assoc]),
    ring,
  },
  { -- verify the answer
    refl
  }
  sorry
}

end find_number_l166_166074


namespace inequality_proof_l166_166311

theorem inequality_proof {n : ℕ} {x : Fin n → ℝ} 
  (h1 : ∀ i, 0 < x i) 
  (h2 : (Finset.univ : Finset (Fin n)).prod x = 1) : 
  (Finset.univ : Finset (Fin n)).sum (λ i, (x i)^(n-1)) ≥ 
  (Finset.univ : Finset (Fin n)).sum (λ i, 1 / (x i)) := 
by
  sorry

end inequality_proof_l166_166311


namespace general_form_of_function_l166_166778

noncomputable def f : ℝ → ℝ := λ x, 3 - (1/2) * x

theorem general_form_of_function (f : ℝ → ℝ) 
  (h₁ : ∀ x, f(x) + 3*f(8 - x) = x)
  (h₂ : f(2) = 2) :
  f = λ x, 3 - (1/2) * x :=
by
  sorry

end general_form_of_function_l166_166778


namespace distance_yolkino_palkino_l166_166323

theorem distance_yolkino_palkino (d_1 d_2 : ℕ) (h : ∀ k : ℕ, d_1 + d_2 = 13) : 
  ∀ k : ℕ, d_1 + d_2 = 13 → (d_1 + d_2 = 13) :=
by
  sorry

end distance_yolkino_palkino_l166_166323


namespace purple_coincide_pairs_l166_166569

theorem purple_coincide_pairs
    (yellow_triangles_upper : ℕ)
    (yellow_triangles_lower : ℕ)
    (green_triangles_upper : ℕ)
    (green_triangles_lower : ℕ)
    (purple_triangles_upper : ℕ)
    (purple_triangles_lower : ℕ)
    (yellow_coincide_pairs : ℕ)
    (green_coincide_pairs : ℕ)
    (yellow_purple_pairs : ℕ) :
    yellow_triangles_upper = 4 →
    yellow_triangles_lower = 4 →
    green_triangles_upper = 6 →
    green_triangles_lower = 6 →
    purple_triangles_upper = 10 →
    purple_triangles_lower = 10 →
    yellow_coincide_pairs = 3 →
    green_coincide_pairs = 4 →
    yellow_purple_pairs = 3 →
    (∃ purple_coincide_pairs : ℕ, purple_coincide_pairs = 5) :=
by sorry

end purple_coincide_pairs_l166_166569


namespace acute_triangle_cos_inequality_l166_166657

theorem acute_triangle_cos_inequality
  {A B C : ℝ}
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (h_sum : A + B + C = π) :
  cos (B - C) * cos (C - A) * cos (A - B) ≥ 8 * cos A * cos B * cos C := by
  sorry

end acute_triangle_cos_inequality_l166_166657


namespace distance_from_P_to_A_l166_166653

theorem distance_from_P_to_A (ABC : Type) [right_triangle ABC]
  (A B C P : ABC) (H₁ : right_angle A) (H₂ : ∀ b c, angle_bisector_inter_point b c = P)  
  (H₃ : ∀ h, distance P h = real.sqrt 80000) : 
  distance P A = 400 :=
by sorry

end distance_from_P_to_A_l166_166653


namespace parabola_equation_l166_166543

theorem parabola_equation (a m f : Point) (p : ℝ) 
  (vertex_origin : a = (0, 0))
  (opens_upward : True) -- This is implicit in our assumption
  (point_on_parabola : Point) 
  (AM_sqrt17 : |a.distance m| = Real.sqrt 17)
  (AF_3 : |a.distance f| = 3) :
  ∃ p > 0, parabola_equation' := 
  x ^ 2 = 12 * y := 
begin
  sorry,
end

end parabola_equation_l166_166543


namespace average_rate_first_half_80_l166_166318

theorem average_rate_first_half_80
    (total_distance : ℝ)
    (average_rate_trip : ℝ)
    (distance_first_half : ℝ)
    (time_first_half : ℝ)
    (time_second_half : ℝ)
    (time_total : ℝ)
    (R : ℝ)
    (H1 : total_distance = 640)
    (H2 : average_rate_trip = 40)
    (H3 : distance_first_half = total_distance / 2)
    (H4 : time_first_half = distance_first_half / R)
    (H5 : time_second_half = 3 * time_first_half)
    (H6 : time_total = time_first_half + time_second_half)
    (H7 : average_rate_trip = total_distance / time_total) :
    R = 80 := 
by 
  -- Given conditions
  sorry

end average_rate_first_half_80_l166_166318


namespace car_average_speed_l166_166057

-- Definitions based on conditions
def distance_first_hour : ℤ := 100
def distance_second_hour : ℤ := 60
def time_first_hour : ℤ := 1
def time_second_hour : ℤ := 1

-- Total distance and time calculations
def total_distance : ℤ := distance_first_hour + distance_second_hour
def total_time : ℤ := time_first_hour + time_second_hour

-- The average speed of the car
def average_speed : ℤ := total_distance / total_time

-- Proof statement
theorem car_average_speed : average_speed = 80 := by
  sorry

end car_average_speed_l166_166057


namespace line_forms_equiv_l166_166149

noncomputable def line_eq_two_pt_form (A B : ℝ × ℝ) : (ℝ × ℝ → Prop) :=
  λ (x y), (y + 1) / (1 - (-1)) = (x - 2) / (5 - 2)

noncomputable def line_eq_pt_slope_form (A B : ℝ × ℝ) : (ℝ × ℝ → Prop) :=
  λ (x y), y + 1 = (2 / 3) * (x - 2)

noncomputable def line_eq_slope_intercept_form (A B : ℝ × ℝ) : (ℝ × ℝ → Prop) :=
  λ (x y), y = (2 / 3) * x - (7 / 3)

noncomputable def line_eq_intercept_form (A B : ℝ × ℝ) : (ℝ × ℝ → Prop) :=
  λ (x y), (x / (7 / 2)) + (y / -(7 / 3)) = 1

theorem line_forms_equiv (A B : ℝ × ℝ) (x y : ℝ) (hA : A = (2, -1)) (hB : B = (5, 1)) :
  line_eq_two_pt_form A B (x, y) ↔
  line_eq_pt_slope_form A B (x, y) ↔
  line_eq_slope_intercept_form A B (x, y) ↔
  line_eq_intercept_form A B (x, y) :=
sorry

end line_forms_equiv_l166_166149


namespace sum_first_15_odd_integers_l166_166807

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l166_166807


namespace original_number_value_l166_166452

noncomputable def orig_number_condition (x : ℝ) : Prop :=
  1000 * x = 3 * (1 / x)

theorem original_number_value : ∃ x : ℝ, 0 < x ∧ orig_number_condition x ∧ x = √(30) / 100 :=
begin
  -- the proof
  sorry
end

end original_number_value_l166_166452


namespace sum_of_first_15_odd_integers_l166_166859

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l166_166859


namespace wheel_revolutions_l166_166085

theorem wheel_revolutions (r_course r_wheel : ℝ) (laps : ℕ) (C_course C_wheel : ℝ) (d_total : ℝ) :
  r_course = 7 →
  r_wheel = 5 →
  laps = 15 →
  C_course = 2 * Real.pi * r_course →
  d_total = laps * C_course →
  C_wheel = 2 * Real.pi * r_wheel →
  ((d_total) / (C_wheel)) = 21 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end wheel_revolutions_l166_166085


namespace total_study_time_l166_166674

/-- Joey's SAT study schedule conditions. -/
variables
  (weekday_hours_per_night : ℕ := 2)
  (weekday_nights_per_week : ℕ := 5)
  (weekend_hours_per_day : ℕ := 3)
  (weekend_days_per_week : ℕ := 2)
  (weeks_until_exam : ℕ := 6)

/-- Total time Joey will spend studying for his SAT exam. -/
theorem total_study_time :
  (weekday_hours_per_night * weekday_nights_per_week + weekend_hours_per_day * weekend_days_per_week) * weeks_until_exam = 96 :=
by
  sorry

end total_study_time_l166_166674


namespace point_in_fourth_quadrant_l166_166713

def point : ℝ × ℝ := (3, -2)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l166_166713


namespace exists_231_four_digit_integers_l166_166563

theorem exists_231_four_digit_integers (n : ℕ) : 
  (∃ A B C D : ℕ, 
     A ≠ 0 ∧ 
     1 ≤ A ∧ A ≤ 9 ∧ 
     0 ≤ B ∧ B ≤ 9 ∧ 
     0 ≤ C ∧ C ≤ 9 ∧ 
     0 ≤ D ∧ D ≤ 9 ∧ 
     999 * (A - D) + 90 * (B - C) = n^3) ↔ n = 231 :=
by sorry

end exists_231_four_digit_integers_l166_166563


namespace find_circle_equation_l166_166579

def circle_eq (h : ℝ) (k : ℝ) (r : ℝ) : ℝ → ℝ → Prop :=
  λ (x y : ℝ), (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

theorem find_circle_equation :
  ∃ h : ℝ, (circle_eq h 0 5) 2 (-3) ∧ ((h = 6) ∨ (h = -2)) :=
by 
  use 6,
  split,
  { simp [circle_eq], ring, norm_num },
  { left, refl }
sorry

end find_circle_equation_l166_166579


namespace area_of_right_triangle_l166_166785

theorem area_of_right_triangle (PQ PR : ℝ) (hPQ : PQ = 8) (hPR : PR = 10) :
  0.5 * PQ * PR = 40 :=
by
  rw [hPQ, hPR]
  norm_num
  sorry

end area_of_right_triangle_l166_166785


namespace final_num_open_doors_l166_166771

-- Define the main problem and its conditions
def num_open_doors (n : ℕ) : ℕ :=
  ∑ i in finset.range (n+1), if (nat.sqrt i) * (nat.sqrt i) = i then 1 else 0

-- Lean statement to prove the main problem
theorem final_num_open_doors : num_open_doors 1000 = 31 :=
by
  sorry

end final_num_open_doors_l166_166771


namespace animal_costs_l166_166098

theorem animal_costs (S K L : ℕ) (h1 : K = 4 * S) (h2 : L = 4 * K) (h3 : S + 2 * K + L = 200) :
  S = 8 ∧ K = 32 ∧ L = 128 :=
by
  sorry

end animal_costs_l166_166098


namespace solution_set_of_inequality_l166_166765

theorem solution_set_of_inequality (x : ℝ) : 
  (x * |x - 1| > 0) ↔ ((0 < x ∧ x < 1) ∨ (x > 1)) := 
by
  sorry

end solution_set_of_inequality_l166_166765


namespace sum_of_first_15_odd_integers_l166_166833

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l166_166833


namespace find_a_l166_166156

theorem find_a (a : ℝ) :
  (∀ x : ℝ, |a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3) →
  a = -3 :=
sorry

end find_a_l166_166156


namespace sum_of_first_15_odd_positives_l166_166803

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l166_166803


namespace sequence_a_n_eq_1_l166_166225

open Nat

theorem sequence_a_n_eq_1 (a b : ℕ → ℝ)
  (h₁ : a 0 = 1 ∧ a 0 ≥ a 1)
  (h₂ : ∀ n ≥ 1, a n * (b (n-1) + b (n+1)) = a (n-1) * b (n-1) + a (n+1) * b (n+1))
  (h₃ : ∀ n ≥ 1, (∑ i in range (n+1), b i) ≤ (n : ℝ)^(3/2)) :
  ∀ n, a n = 1 :=
by
  sorry -- Proof omitted

end sequence_a_n_eq_1_l166_166225


namespace original_number_l166_166457

theorem original_number (x : ℝ) (h : 1000 * x = 3 / x) : x = (real.sqrt 30) / 100 :=
by sorry

end original_number_l166_166457


namespace probability_snow_at_least_once_first_week_l166_166322

noncomputable def snow_probability_first_three_days : ℚ := 1 / 2
noncomputable def snow_probability_next_four_days : ℚ := 1 / 3

theorem probability_snow_at_least_once_first_week :
  let no_snow_first_three_days := (1 - snow_probability_first_three_days) ^ 3
  let no_snow_next_four_days := (1 - snow_probability_next_four_days) ^ 4
  let no_snow_all_week := no_snow_first_three_days * no_snow_next_four_days
  let snow_at_least_once := 1 - no_snow_all_week
  in snow_at_least_once = 79 / 81 :=
by
  sorry

end probability_snow_at_least_once_first_week_l166_166322


namespace Michael_rides_six_miles_l166_166354

theorem Michael_rides_six_miles
  (rate : ℝ)
  (time : ℝ)
  (interval_time : ℝ)
  (interval_distance : ℝ)
  (intervals : ℝ)
  (total_distance : ℝ) :
  rate = 1.5 ∧ time = 40 ∧ interval_time = 10 ∧ interval_distance = 1.5 ∧ intervals = time / interval_time ∧ total_distance = intervals * interval_distance →
  total_distance = 6 :=
by
  intros h
  -- Placeholder for the proof
  sorry

end Michael_rides_six_miles_l166_166354


namespace segment_measure_l166_166631

theorem segment_measure (a b : ℝ) (m : ℝ) (h : a = m * b) : (1 / m) * a = b :=
by sorry

end segment_measure_l166_166631


namespace proof_of_propositions_correct_l166_166485

noncomputable def vec_eq (AB MB BC OM CO: ℝ × ℝ): Prop :=
AB + MB + BC + OM + CO = AB

noncomputable def obtuse_angle (a b: ℝ × ℝ) (k: ℝ): Prop :=
(k < 9)

noncomputable def lin_indep (e1 e2: ℝ × ℝ): Prop :=
¬(∃ (c: ℝ), e1 = c • e2)

noncomputable def proj_parallel (a b: ℝ × ℝ): Prop :=
(a = b ∨ a = -b) → (a = b)

noncomputable def propositions_correct (A B C D: Prop): Prop :=
(∀ (AB MB BC OM CO: ℝ × ℝ), vec_eq AB MB BC OM CO = A) ∧
(∀ (a: ℝ × ℝ) (b: ℝ) (k: ℝ), obtuse_angle a b k = B) ∧
(∀ (e1 e2: ℝ × ℝ), lin_indep e1 e2 = C) ∧
(∀ (a b: ℝ × ℝ), proj_parallel a b = D) →
(A ∧ D)

theorem proof_of_propositions_correct:
  propositions_correct (vec_eq (1, 0) (0, 0) (0, 0) (0, 0) (0, 0))
    (obtuse_angle (6, 2) (-3, 1) 8.999)
    (lin_indep (2, -3) (0.5, -0.75))
    (proj_parallel (1, 1) (1, 1)) :=
sorry

end proof_of_propositions_correct_l166_166485


namespace probability_not_between_zero_and_one_l166_166022

theorem probability_not_between_zero_and_one (A : Prop) (P : Prop -> ℝ) (h : P(A) = 1.5) : ¬ (0 ≤ P(A) ∧ P(A) ≤ 1) :=
by 
  sorry

end probability_not_between_zero_and_one_l166_166022


namespace anna_interest_l166_166537

noncomputable def interest_earned (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n - P

theorem anna_interest : interest_earned 2000 0.08 5 = 938.66 := by
  sorry

end anna_interest_l166_166537


namespace height_of_hypotenuse_l166_166040

theorem height_of_hypotenuse (p q : ℝ) : 
  ∃ m : ℝ, 
  let c := p + q in
  let c1 := (c * p^2) / (p^2 + q^2) in
  let c2 := (c * q^2) / (p^2 + q^2) in
  m = sqrt(c1 * c2) := 
by
  let c := p + q
  let c1 := (c * p^2) / (p^2 + q^2)
  let c2 := (c * q^2) / (p^2 + q^2)
  existsi (sqrt(c1 * c2))
  sorry

end height_of_hypotenuse_l166_166040


namespace shaded_region_area_l166_166018

noncomputable def square_side_length : ℝ := 12
noncomputable def circle_radius : ℝ := 2

def square_area : ℝ := square_side_length ^ 2

def sector_area : ℝ := (circle_radius ^ 2 * Real.pi) / 2

def triangle_area : ℝ := (circle_radius ^ 2) / 2

def non_shaded_area : ℝ := 4 * sector_area + 4 * triangle_area

def shaded_area : ℝ := square_area - non_shaded_area

theorem shaded_region_area : shaded_area = 136 - 2 * Real.pi :=
by
  sorry

end shaded_region_area_l166_166018


namespace equilateral_triangle_area_l166_166096

theorem equilateral_triangle_area (p : ℕ) (h : 2 * p ≠ 0) :
  let x := (2 * p) / 3,
      A := (sqrt 3 / 4) * x^2
  in A = (sqrt 3 * p^2) / 9 :=
by
  sorry

end equilateral_triangle_area_l166_166096


namespace sum_of_first_15_odd_integers_l166_166855

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l166_166855


namespace probability_white_given_popped_l166_166894

noncomputable def white_kernel_probability_popped (P_white P_white_popped P_yellow P_yellow_popped P_blue P_blue_popped : ℝ) : ℝ :=
    let P_popped := P_white * P_white_popped + P_yellow * P_yellow_popped + P_blue * P_blue_popped in
    (P_white * P_white_popped) / P_popped

theorem probability_white_given_popped :
    white_kernel_probability_popped (2/5) (1/4) (1/5) (3/4) (2/5) (1/2) = 2 / 9 :=
by
    sorry

end probability_white_given_popped_l166_166894


namespace solve_inequality_l166_166366

theorem solve_inequality (x : ℝ) : x^2 ≥ 2 * x ↔ x ∈ Icc 0 2 ∪ (Iio 0) ∪ (Ioi 2) :=
sorry

end solve_inequality_l166_166366


namespace meteorological_forecasts_inaccuracy_l166_166317

theorem meteorological_forecasts_inaccuracy :
  let pA_accurate := 0.8
  let pB_accurate := 0.7
  let pA_inaccurate := 1 - pA_accurate
  let pB_inaccurate := 1 - pB_accurate
  pA_inaccurate * pB_inaccurate = 0.06 :=
by
  sorry

end meteorological_forecasts_inaccuracy_l166_166317


namespace computation_correct_l166_166951

theorem computation_correct : 12 * ((216 / 3) + (36 / 6) + (16 / 8) + 2) = 984 := 
by 
  sorry

end computation_correct_l166_166951


namespace no_triangle_with_cos_sum_eq_one_l166_166567

theorem no_triangle_with_cos_sum_eq_one (A B C : ℝ) (hT : A + B + C = π) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  cos A + cos B + cos C ≠ 1 :=
sorry

end no_triangle_with_cos_sum_eq_one_l166_166567


namespace find_original_number_l166_166470

theorem find_original_number (x : ℝ) (hx : 0 < x) 
  (h : 1000 * x = 3 * (1 / x)) : x ≈ 0.01732 :=
by 
  sorry

end find_original_number_l166_166470


namespace original_number_l166_166458

theorem original_number (x : ℝ) (h : 1000 * x = 3 / x) : x = (real.sqrt 30) / 100 :=
by sorry

end original_number_l166_166458


namespace yogurt_combinations_l166_166529

/-- Given 4 flavors of yogurt and 8 different toppings, if a customer can either choose no topping or exactly two different toppings, then the total number of combinations is 116. -/
theorem yogurt_combinations :
  let flavors := 4
  let toppings := 8
  ∃ (n : ℕ), n = (flavors * (1 + (Nat.choose toppings 2))) ∧ n = 116 :=
by
  sorry

end yogurt_combinations_l166_166529


namespace sum_first_15_odd_integers_l166_166822

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l166_166822


namespace exists_best_restaurant_l166_166228

theorem exists_best_restaurant (n : ℕ) (R : ℕ → Type) 
  (better_tastiness better_service : R → R → Prop)
  (h_trans_tastiness : ∀ A B C, better_tastiness A B → better_tastiness B C → better_tastiness A C)
  (h_trans_service : ∀ A B C, better_service A B → better_service B C → better_service A C)
  (h_comparable : ∀ A B, (better_tastiness A B ∨ better_tastiness B A) → (better_service A B ∨ better_service B A) → true) :
  ∃ R_t, ∀ R_other, R_other ≠ R_t → (better_tastiness R_t R_other ∨ better_service R_t R_other) :=
sorry

end exists_best_restaurant_l166_166228


namespace problem1_problem2_l166_166339

theorem problem1 (x : ℝ) : x * (x + 2) = (x + 2) → x = -2 ∨ x = 1 := 
by
  sorry

theorem problem2 (a b c : ℝ) (Δ : ℝ) : 
  a = 2 → 
  b = -6 → 
  c = 1 → 
  Δ = b^2 - 4 * a * c → 
  2*a*x^2 + b*x + c = 0 → 
  x = (3 - real.sqrt 7) / 2 ∨ x = (3 + real.sqrt 7) / 2 :=
by
  sorry

end problem1_problem2_l166_166339


namespace distinct_sums_exist_iff_even_l166_166170

theorem distinct_sums_exist_iff_even (n : ℕ) (hn : 0 < n) :
  (∃ (A : Matrix (Fin n) (Fin n) ℤ), 
    (∀ i : Fin n, ∀ j : Fin n, A i j ∈ {-1, 0, 1}) ∧ 
    (∀ i j k l : Fin n, (∑ m, A i m ≠ ∑ m, A j m ∨ i = j) ∧ (∑ m, A m k ≠ ∑ m, A m l ∨ k = l))) ↔ n % 2 = 0 :=
by
  sorry

end distinct_sums_exist_iff_even_l166_166170


namespace sum_of_first_15_odd_integers_l166_166858

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l166_166858


namespace longest_segment_inside_cylinder_eq_total_surface_area_cylinder_eq_l166_166069

-- Define the constants for the cylinder's dimensions
def radius := 5 -- in cm
def height := 10 -- in cm

-- Prove that the longest segment that fits inside the cylinder is 10 * sqrt(2) cm
theorem longest_segment_inside_cylinder_eq : 
  let diameter := 2 * radius in
  Real.sqrt (height^2 + diameter^2) = 10 * Real.sqrt 2 :=
by simp [radius, height, Real.sqrt_eq_rpow, Real.pow_two]; sorry

-- Prove that the total surface area of the cylinder is 150π square cm
theorem total_surface_area_cylinder_eq : 
  let surface_area := 2 * Real.pi * radius * (height + radius) in 
  surface_area = 150 * Real.pi :=
by simp [radius, height, Real.pi]; sorry

end longest_segment_inside_cylinder_eq_total_surface_area_cylinder_eq_l166_166069


namespace sum_of_first_15_odd_positives_l166_166798

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l166_166798


namespace max_add_sub_of_set_l166_166041

theorem max_add_sub_of_set : 
  let S := {-20, -5, 1, 3, 7, 15}
  ∃ a b ∈ S, a ≠ b ∧ (a - b = 35) := 
by 
  let S := {-20, -5, 1, 3, 7, 15}
  use 15
  use (-20)
  simp
  sorry

end max_add_sub_of_set_l166_166041


namespace sum_of_first_three_terms_l166_166752

theorem sum_of_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 8) 
  (h5 : a 5 = 12) 
  (h6 : a 6 = 16) : 
  a 1 + a 2 + a 3 = 0 :=
by
  sorry

end sum_of_first_three_terms_l166_166752


namespace Sandy_record_age_l166_166377

theorem Sandy_record_age :
  ∀ (current_length age_years : ℕ) (goal_length : ℕ) (growth_rate tenths_per_month : ℕ),
  goal_length = 26 ∧
  current_length = 2 ∧
  age_years = 12 ∧
  growth_rate = 10 *
  tenths_per_month / 10 →
  age_years + (goal_length - current_length) * 10 / growth_rate = 32 :=
by
  intros,
  sorry

end Sandy_record_age_l166_166377


namespace sequence_product_l166_166145

theorem sequence_product :
  (∏ n in finset.range 5, (1 / (3 ^ (2 * n + 1))) * (3 ^ (2 * n + 2))) = 243 :=
by
  sorry

end sequence_product_l166_166145


namespace motorist_total_distance_l166_166517

-- Define the individual time durations in hours
def time1 := 2
def time2 := 3
def time3 := 75 / 60
def time4 := 95 / 60

-- Define the speeds in kmph
def speed1 := 60
def speed2 := 100
def speed3 := 72
def speed4 := 50

-- Define the distances for each segment
def distance1 := speed1 * time1
def distance2 := speed2 * time2
def distance3 := speed3 * time3
def distance4 := speed4 * time4

-- Define the total distance
def total_distance := distance1 + distance2 + distance3 + distance4

-- The statement that we need to prove
theorem motorist_total_distance : total_distance = 589.17 := 
by sorry

end motorist_total_distance_l166_166517


namespace original_number_solution_l166_166442

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end original_number_solution_l166_166442


namespace compute_g_200_l166_166964

def g (x : ℕ) : ℕ :=
  if ∃ n : ℕ, x = 2^n then 
    nat.log2 x 
  else 
    1 + g (x + 1)

theorem compute_g_200 : g 200 = 64 :=
  sorry

end compute_g_200_l166_166964


namespace digit_of_fraction_415th_l166_166037

theorem digit_of_fraction_415th (rational : ℚ) (cycle : ℕ → ℕ) (h_cycle : ∀ n, cycle n = (⟨7, 29⟩ : ℚ).decimal_digit (n + 1)) :
  (cycle 415) = 2 :=
by
  sorry

end digit_of_fraction_415th_l166_166037


namespace circles_intersect_l166_166767

-- Define the first circle as a predicate
def circle1 (x y : ℝ) : Prop := (x + 1) ^ 2 + (y - 2) ^ 2 = 1

-- Define the second circle as a predicate
def circle2 (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 9

-- Define centers and radii of the circles
def center1 : (ℝ × ℝ) := (-1, 2)
def radius1 : ℝ := 1

def center2 : (ℝ × ℝ) := (0, 0)
def radius2 : ℝ := 3

-- Function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Proof that the circles intersect
theorem circles_intersect : 
  3 - 1 < distance center1 center2 ∧ distance center1 center2 < 3 + 1 :=
by 
  sorry

end circles_intersect_l166_166767


namespace sum_of_arithmetic_sequence_l166_166042

theorem sum_of_arithmetic_sequence :
  ∀ (a d n : ℤ), a = -5 → d = 7 → n = 10 →
  let Tₙ := a + (n - 1) * d in 
  let S := (a + Tₙ) / 2 * n in 
  S = 265 :=
begin
  intros a d n ha hd hn,
  rw [ha, hd, hn],
  let Tₙ := -5 + (10 - 1) * 7,
  let S := (-5 + Tₙ) / 2 * 10,
  rw [Tₙ, S],
  sorry
end

end sum_of_arithmetic_sequence_l166_166042


namespace fixed_point_difference_l166_166211

noncomputable def func (a x : ℝ) : ℝ := a^x + Real.log a

theorem fixed_point_difference (a m n : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  (func a 0 = n) ∧ (y = func a x → (x = m) ∧ (y = n)) → (m - n = -2) :=
by 
  intro h
  sorry

end fixed_point_difference_l166_166211


namespace smallest_value_of_abs_sum_l166_166424

theorem smallest_value_of_abs_sum : ∃ x : ℝ, (x = -3) ∧ ( ∀ y : ℝ, |y + 1| + |y + 3| + |y + 6| ≥ 5 ) :=
by
  use -3
  split
  . exact rfl
  . intro y
    have h1 : |y + 1| + |y + 3| + |y + 6| = sorry,
    sorry

end smallest_value_of_abs_sum_l166_166424


namespace total_students_l166_166503

-- Definition of the problem conditions
def buses : ℕ := 18
def seats_per_bus : ℕ := 15
def empty_seats_per_bus : ℕ := 3

-- Formulating the mathematically equivalent proof problem
theorem total_students :
  (buses * (seats_per_bus - empty_seats_per_bus) = 216) :=
by
  sorry

end total_students_l166_166503


namespace seating_arrangements_l166_166915

theorem seating_arrangements (seats : Fin 7 → Option ℕ) :
  let positions := (0:Fin 7).val upto (6:Fin 7).val
  ∃ (f : Fin 7 → Option ℕ), 
  (f ≠ (λ i, none)) ∧
  f (0:Fin 7) = some 1 ∧
  f (1:Fin 7) = some 2 ∧
  f (2:Fin 7) = none  ∧
  f (3:Fin 7) = none ∧
  f (4:Fin 7) = some 1 ∧
  f (5:Fin 7) = none ∧
  f (6:Fin 7) = some 2 ∧
  ∃ a b : Fin 7, a ≠ b ∧
  f a = some A ∧
  f b = some B ∧
  ((a: ℕ) ≠ (b: ℕ) + 1  ∧ (a: ℕ) ≠ (b: ℕ) - 1) :=
by
  sorry

end seating_arrangements_l166_166915


namespace ratio_of_heights_is_3_to_5_l166_166390

-- Definitions of the angles and their relationships
structure IsoscelesTriangle :=
(base_angle : ℝ)
(vertical_angle : ℝ)
(area : ℝ)
(height : ℝ)

-- The first triangle has base angles of 40 degrees each
def triangleA : IsoscelesTriangle :=
{ base_angle := 40,
  vertical_angle := 180 - 2 * 40,
  area := sorry, -- the actual area value is not required for the statement
  height := sorry -- to be used in the proof
}

-- The second triangle has base angles of 50 degrees each
def triangleB : IsoscelesTriangle :=
{ base_angle := 50,
  vertical_angle := 180 - 2 * 50,
  area := sorry, -- the actual area value is not required for the statement
  height := sorry -- to be used in the proof
}

-- Assuming both triangles are similar and the given proportions
axiom sides_ratio : 3 / 5 -- ratio of the side lengths is 3:5
axiom areas_ratio : 9 / 25 -- ratio of the areas is 9:25

-- Prove the ratio of the corresponding heights
theorem ratio_of_heights_is_3_to_5 (hA hB : ℝ) :
  (triangleA.vertical_angle = triangleB.vertical_angle) →
  (3 / 5 = sides_ratio) →
  (9 / 25 = areas_ratio) →
  (hA / hB = 3 / 5) :=
begin
  sorry
end

end ratio_of_heights_is_3_to_5_l166_166390


namespace lamps_initial_odd_off_after_configuration_one_off_l166_166891

/-- 
  There are 12 lamps arranged in a circle. Each lamp can be either on or off. 
  At each step, you can choose an off lamp and toggle the state of its two neighbors. 
  Prove that the initial configurations that can reach a state where only one lamp is off 
  are those configurations where the initial number of off lamps is odd.
--/
theorem lamps_initial_odd_off_after_configuration_one_off (initial_state: Fin 12 → Bool) :
    (∃ k, k % 2 = 1 ∧ k = (Finset.filter (λ i, initial_state i = false) Finset.univ).card) ↔ 
    (∃ n, n = 1 ∧ reachable_one_off initial_state) := 
sorry

/-- 
  Predicate stating that a given initial configuration can reach a state with exactly one lamp off.
--/
def reachable_one_off (initial_state: Fin 12 → Bool) : Prop := 
sorry

end lamps_initial_odd_off_after_configuration_one_off_l166_166891


namespace probability_diff_numbers_l166_166505

-- Define our ball bag and draw process.
def bag : finset ℕ := {1, 2, 3, 4, 5, 6}

def draw (n : ℕ) : finset (ℕ × ℕ) := (bag ×ᶠ bag)

-- The probability that the numbers on the balls drawn by A and B are different.
theorem probability_diff_numbers :
  let total_outcomes := 36
  let favorable_outcomes := 30
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 6 :=
by
  -- Proof will be provided here.
  sorry

end probability_diff_numbers_l166_166505


namespace car_mileage_approx_30_l166_166899

def car_mileage_per_gallon (distance_traveled : ℝ) (gallons_used : ℝ) : ℝ :=
  distance_traveled / gallons_used

theorem car_mileage_approx_30 :
  car_mileage_per_gallon 200 6.666666666666667 ≈ 30 :=
by 
  sorry

end car_mileage_approx_30_l166_166899


namespace negation_of_universal_proposition_l166_166360

theorem negation_of_universal_proposition :
  (¬ ∀ x > 1, (1 / 2)^x < 1 / 2) ↔ (∃ x > 1, (1 / 2)^x ≥ 1 / 2) :=
sorry

end negation_of_universal_proposition_l166_166360


namespace find_f_2017_l166_166202

noncomputable def f : ℝ → ℝ :=
  λ x, if 1 ≤ x ∧ x ≤ 2 then 2^x - 1 else 0 -- We're considering the definition is given piecewise under the range [1, 2].

axiom f_condition1 : ∀ x, f (1 + x) + f (1 - x) = 0
axiom f_condition2 : ∀ x, f (-x) = f (x)
axiom f_condition3 : ∀ x, 1 ≤ x ∧ x ≤ 2 → f (x) = 2^x - 1

theorem find_f_2017 : f 2017 = 1 := sorry

end find_f_2017_l166_166202


namespace rate_of_paving_l166_166357

-- Define the given conditions and the corresponding values
def length : ℝ := 5.5
def width : ℝ := 3.75
def total_cost : ℝ := 20625

-- Calculate expected area and rate for later use
def area : ℝ := length * width
def expected_rate : ℝ := total_cost / area

-- State the proposition (no proof included)
theorem rate_of_paving : expected_rate = 1000 :=
by
  -- We use sorry to skip the proof.
  sorry

end rate_of_paving_l166_166357


namespace problem1_problem2_l166_166940

-- Problem 1: Calculation
theorem problem1 :
  (1:Real) - 1^2 + Real.sqrt 12 + Real.sqrt (4 / 3) = -1 + (8 * Real.sqrt 3) / 3 :=
by
  sorry
  
-- Problem 2: Solve the equation 2x^2 - x - 1 = 0
theorem problem2 (x : Real) :
  (2 * x^2 - x - 1 = 0) → (x = -1/2 ∨ x = 1) :=
by
  sorry

end problem1_problem2_l166_166940


namespace preservation_interval_l166_166246

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) * (x - 1)^2 + 1

theorem preservation_interval {a b : ℝ} (h_dom_range : (∀ y, y ∈ set.Icc a b → (∃ x, x ∈ set.Icc a b ∧ f x = y))) :
  set.Icc a b = set.Icc 1 3 :=
begin
  -- We assume the domain and range conditions
  -- We need to prove that the preservation interval is [1, 3]
  sorry
end

end preservation_interval_l166_166246


namespace sum_of_first_n_odd_numbers_l166_166842

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l166_166842


namespace largest_fraction_sum_l166_166117

theorem largest_fraction_sum : 
  (max (max (max (max 
  ((1 : ℚ) / 3 + (1 : ℚ) / 4) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 5)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 2)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 9)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 6)) = (5 : ℚ) / 6 
:= 
by
  sorry

end largest_fraction_sum_l166_166117


namespace ivan_prefers_at_least_k_factorial_l166_166158

variables {n k : ℕ} {M : Finset ℕ}
variables {prefers : Finset (List ℕ)} (h0 : 2 ≤ k) (h1 : k ≤ n) (h2 : M.card = n)
           (h3 : ∀ x ∈ prefers, ∃ i : Fin k → Fin n, ∀ j < k - 1, List.swap x (i j) (i (j + 1)) ∈ prefers)

theorem ivan_prefers_at_least_k_factorial : prefers.card ≥ k.fact :=
by sorry

end ivan_prefers_at_least_k_factorial_l166_166158


namespace isosceles_triangle_perimeter_l166_166255

variable (a b c : ℝ) (h_iso : a = b ∨ a = c ∨ b = c) (h_a : a = 6) (h_b : b = 6) (h_c : c = 3)
variable (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)

theorem isosceles_triangle_perimeter : a + b + c = 15 :=
by 
  -- Given definitions and triangle inequality
  have h_valid : a = 6 ∧ b = 6 ∧ c = 3 := ⟨h_a, h_b, h_c⟩
  sorry

end isosceles_triangle_perimeter_l166_166255


namespace min_disks_required_l166_166067

/-- A structure to hold information about the file storage problem -/
structure FileStorageConditions where
  total_files : ℕ
  disk_capacity : ℝ
  num_files_1_6MB : ℕ
  num_files_1MB : ℕ
  num_files_0_5MB : ℕ

/-- Define specific conditions given in the problem -/
def storage_conditions : FileStorageConditions := {
  total_files := 42,
  disk_capacity := 2.88,
  num_files_1_6MB := 8,
  num_files_1MB := 16,
  num_files_0_5MB := 18 -- Derived from total_files - num_files_1_6MB - num_files_1MB
}

/-- Theorem stating the minimum number of disks required to store all files is 16 -/
theorem min_disks_required (c : FileStorageConditions)
  (h1 : c.total_files = 42)
  (h2 : c.disk_capacity = 2.88)
  (h3 : c.num_files_1_6MB = 8)
  (h4 : c.num_files_1MB = 16)
  (h5 : c.num_files_0_5MB = 18) :
  ∃ n : ℕ, n = 16 := by
  sorry

end min_disks_required_l166_166067


namespace area_of_triangle_formed_by_intersection_l166_166130

-- Define the points determining the lines
def point1_line1 : ℝ × ℝ := (0, 3)
def point2_line1 : ℝ × ℝ := (6, 0)
def point1_line2 : ℝ × ℝ := (1, 6)
def point2_line2 : ℝ × ℝ := (7, 1)

-- Define the desired area of the triangular region
def area_triangle : ℝ := 9

-- Statement to prove
theorem area_of_triangle_formed_by_intersection (p1l1 p2l1 p1l2 p2l2 : ℝ × ℝ) :
  p1l1 = point1_line1 → p2l1 = point2_line1 → p1l2 = point1_line2 → p2l2 = point2_line2 →
  calculate_area_of_triangle p1l1 p2l1 p1l2 p2l2 = area_triangle :=
by {
  intros, -- assuming conditions
  sorry   -- proof to be filled in
}

end area_of_triangle_formed_by_intersection_l166_166130


namespace hyperbola_focal_length_l166_166749

-- Define the constants a^2 and b^2 based on the given hyperbola equation.
def a_squared : ℝ := 16
def b_squared : ℝ := 25

-- Define the constants a and b as the square roots of a^2 and b^2.
noncomputable def a : ℝ := Real.sqrt a_squared
noncomputable def b : ℝ := Real.sqrt b_squared

-- Define the constant c based on the relation c^2 = a^2 + b^2.
noncomputable def c : ℝ := Real.sqrt (a_squared + b_squared)

-- The focal length of the hyperbola is 2c.
noncomputable def focal_length : ℝ := 2 * c

-- The theorem that captures the statement of the problem.
theorem hyperbola_focal_length : focal_length = 2 * Real.sqrt 41 := by
  -- Proof omitted.
  sorry

end hyperbola_focal_length_l166_166749


namespace coefficient_of_x2_in_expression_is_7_l166_166967

def polynomial := 3 * (λ x : ℝ, x - 2 * x^3) 
                - 4 * (λ x : ℝ, 2 * x^2 - x^3 + x^4) 
                + 5 * (λ x : ℝ, 3 * x^2 - x^5)

-- Theorem: Prove that the coefficient of x^2 in the simplified polynomial is 7
theorem coefficient_of_x2_in_expression_is_7 : 
  ∀ (x : ℝ), (3 * (x - 2 * x^3) - 4 * (2 * x^2 - x^3 + x^4) + 5 * (3 * x^2 - x^5)).coeff (2 : ℕ) = 7 :=
by
  sorry

end coefficient_of_x2_in_expression_is_7_l166_166967


namespace stewart_farm_l166_166760

theorem stewart_farm (S H : ℕ)
  (h1 : S / H = 2 / 7)
  (h2 : H * 230 = 12,880) : S = 16 :=
by
  sorry

end stewart_farm_l166_166760


namespace peach_difference_l166_166892

theorem peach_difference (red_peaches green_peaches : ℕ) (h_red : red_peaches = 5) (h_green : green_peaches = 11) :
  green_peaches - red_peaches = 6 :=
by
  rw [h_red, h_green]
  rfl

end peach_difference_l166_166892


namespace exists_perpendicular_area_bisector_l166_166960

-- Conditions
variables {A B C M N : Point}

-- Definitions: Triangle ABC, line MN perpendicular to AB, line bisecting the area
def triangle_ABC (A B C : Point) : Triangle := ⟨A, B, C⟩
def M_on_AB (A B M : Point) : Prop := M ∈ Line(A, B)
def perpendicular_to_AB (A B M N : Point) : Prop := Perpendicular(Line(A, B), Line(M, N))
def area_bisector (A B C M N : Point) : Prop := 
  let T := Triangle(⟨A, M, N⟩) in 
  (area A B C) = 2 * (area A M N)

-- Main theorem
theorem exists_perpendicular_area_bisector (A B C : Point) :
  ∃ M N : Point, M_on_AB A B M ∧ perpendicular_to_AB A B M N ∧ area_bisector A B C M N :=
  by sorry

end exists_perpendicular_area_bisector_l166_166960


namespace sum_of_first_15_odd_integers_l166_166827

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l166_166827


namespace parabola_conditions_l166_166001

theorem parabola_conditions 
  (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b = 2 * a) 
  (hc : c = -3 * a) 
  (hA : a * (-3)^2 + b * (-3) + c = 0) 
  (hB : a * (1)^2 + b * (1) + c = 0) : 
  (b^2 - 4 * a * c > 0) ∧ (3 * b + 2 * c = 0) :=
sorry

end parabola_conditions_l166_166001


namespace solution_set_for_quadratic_inequality_l166_166165

theorem solution_set_for_quadratic_inequality {a : ℝ} :
  let S := { x : ℤ | (x : ℝ)^2 - (a + 1) * (x : ℝ) + a < 0 } in
  (S.card = 2) ↔ (3 < a ∧ a ≤ 4) ∨ (-2 ≤ a ∧ a < -1) :=
by
  sorry

end solution_set_for_quadratic_inequality_l166_166165


namespace fraction_product_l166_166043

theorem fraction_product :
  (2 / 3) * (3 / 4) * (5 / 6) * (6 / 7) * (8 / 9) = 80 / 63 :=
by sorry

end fraction_product_l166_166043


namespace Sandy_record_age_l166_166378

theorem Sandy_record_age :
  ∀ (current_length age_years : ℕ) (goal_length : ℕ) (growth_rate tenths_per_month : ℕ),
  goal_length = 26 ∧
  current_length = 2 ∧
  age_years = 12 ∧
  growth_rate = 10 *
  tenths_per_month / 10 →
  age_years + (goal_length - current_length) * 10 / growth_rate = 32 :=
by
  intros,
  sorry

end Sandy_record_age_l166_166378


namespace max_roses_purchase_l166_166159

theorem max_roses_purchase
  (roses_picked : ℕ := 6)
  (flowers_needed : ℕ := 18)
  (cost_constraint : ℕ := 30)
  (cost_rose : ℕ := 3)
  (cost_tulip : ℕ := 2)
  (cost_daisy : ℕ := 1):
  ∃ R T D, 
  3 * R + 2 * T + 1 * D ≤ cost_constraint ∧ 
  R + T + D = flowers_needed ∧
  ∀ R' T' D', 
  3 * R' + 2 * T' + 1 * D' ≤ cost_constraint ∧ 
  R' + T' + D' = flowers_needed → 
  R' ≤ R :=
begin
  -- We provide a proof here to check existence and optimality
  -- The specific values fulfilling our constraints are R = 9, T = 1, D = 8
  use [9, 1, 8],
  split,
  {
    simp,
    norm_num,
  },
  split,
  {
    simp,
    norm_num,
  },
  {
    intros R' T' D' hR'T'D hsumR'T'D,
    have htotal := add_assoc R' T' D',
    simp at htotal hsumR'T'D,
    rw hsumR'T'D at htotal,
    linarith,
  },
end

end max_roses_purchase_l166_166159


namespace larger_volume_of_cut_cube_l166_166961

theorem larger_volume_of_cut_cube :
  let A := (0, 0, 0)
  let P := (1, 2, 1)
  let Q := (2, 1, 2)
  let plane := {x : ℝ × ℝ × ℝ | x.1 + x.2 + x.3 = 3}
  let cube_volume := 8
  let pyramid_volume := (1 : ℝ) / 3
  let larger_solid_volume := cube_volume - pyramid_volume
  in larger_solid_volume = 23 / 3 :=
by
  sorry

end larger_volume_of_cut_cube_l166_166961


namespace initial_average_l166_166676

theorem initial_average (A : ℝ) (h : (15 * A + 14 * 15) / 15 = 54) : A = 40 :=
by
  sorry

end initial_average_l166_166676


namespace total_legs_l166_166551

-- Definitions related to the problem conditions.
def justin_dogs : ℕ := 14
def rico_dogs : ℕ := justin_dogs + 10
def camden_dogs : ℕ := 3 * rico_dogs / 4 -- fractions are not directly handled as int multiplication and division in Lean
def camden_3_legs_dogs : ℕ := 5
def camden_4_legs_dogs : ℕ := 7
def camden_2_legs_dogs : ℕ := 2
def samantha_cats : ℕ := 8
def samantha_4_legs_cats : ℕ := 6
def samantha_3_legs_cats : ℕ := 2

-- Calculation of the number of legs.
def camden_dogs_legs : ℕ := camden_3_legs_dogs * 3 + camden_4_legs_dogs * 4 + camden_2_legs_dogs * 2
def rico_dogs_legs : ℕ := rico_dogs * 4
def samantha_cats_legs : ℕ := samantha_4_legs_cats * 4 + samantha_3_legs_cats * 3

-- Theorem stating the problem's assertion.
theorem total_legs :
  camden_dogs_legs + rico_dogs_legs + samantha_cats_legs = 173 :=
by
  unfold camden_dogs_legs rico_dogs_legs samantha_cats_legs
  unfold camden_dogs rico_dogs justin_dogs
  sorry

end total_legs_l166_166551


namespace shortest_segment_is_AD_l166_166352

-- Definitions for the points and segments
variables (A B C D : Type)

-- Angle conditions
def angle_ABD : ℝ := 30
def angle_ADB : ℝ := 60
def angle_CBD : ℝ := 90
def angle_BDC : ℝ := 60

-- Question: prove that the shortest segment is AD
theorem shortest_segment_is_AD (h1 : angle_ABD = 30)
                                (h2 : angle_ADB = 60)
                                (h3 : angle_CBD = 90)
                                (h4 : angle_BDC = 60) :
  shortest_segment A B C D = AD := 
sorry

end shortest_segment_is_AD_l166_166352


namespace sum_of_first_n_odd_numbers_l166_166835

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l166_166835


namespace combination_identity_l166_166887

open BigOperators

theorem combination_identity : nat.choose 12 5 + nat.choose 12 6 = nat.choose 13 6 := 
sorry

end combination_identity_l166_166887


namespace sally_pokemon_cards_count_l166_166725

-- Defining the initial conditions
def initial_cards : ℕ := 27
def cards_given_by_dan : ℕ := 41
def cards_bought_by_sally : ℕ := 20

-- Statement of the problem to be proved
theorem sally_pokemon_cards_count :
  initial_cards + cards_given_by_dan + cards_bought_by_sally = 88 := by
  sorry

end sally_pokemon_cards_count_l166_166725


namespace focus_of_parabola_shift_l166_166150

noncomputable def parabola_focus (a b : ℝ) (x : ℝ) : ℝ × ℝ :=
(let vertex : ℝ × ℝ := (0, b) in
  let y_focus := b + 1 / (4 * a) in
  (0, y_focus))

theorem focus_of_parabola_shift (x : ℝ) :
  parabola_focus 9 5 x = (0, 181 / 36) :=
by
  sorry

end focus_of_parabola_shift_l166_166150


namespace problem_statement_l166_166957

noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * Real.sqrt x

theorem problem_statement : 3 * g 3 - g 9 = -48 - 6 * Real.sqrt 3 := by
  sorry

end problem_statement_l166_166957


namespace sum_of_valid_x_l166_166936

theorem sum_of_valid_x : 
  let S := Finset.range (50) \ Finset.range (34) -- The range [34, 50)
  let x_values := S.filter (λ x, x * 2 < 100 ∧ x * 3 ≥ 100)
  (Finset.sum x_values id) = 664 :=
by
  sorry

end sum_of_valid_x_l166_166936


namespace product_of_numbers_is_2014_pow_1007_sqrt_2014_l166_166772

-- Given condition definitions
variable (numbers : List ℝ)
variable (h_distinct : numbers.Nodup)
variable (h_positive : ∀ x ∈ numbers, x > 0)
variable (h_count : numbers.length = 2015)
variable (h_parity : ∀ a > 0, 
  Parity (numbers.filter (λ x, x < 2014 / a)).length = Parity (numbers.filter (λ x, x > a)).length)

-- Lean theorem statement
theorem product_of_numbers_is_2014_pow_1007_sqrt_2014 : 
  numbers.Prod = 2014 ^ 1007 * Real.sqrt 2014 := by
sorry

end product_of_numbers_is_2014_pow_1007_sqrt_2014_l166_166772


namespace selection_competition_l166_166079

variables (p q r : Prop)

theorem selection_competition 
  (h1 : p ∨ q) 
  (h2 : ¬ (p ∧ q)) 
  (h3 : ¬ q ∧ r) : p ∧ ¬ q ∧ r :=
by
  sorry

end selection_competition_l166_166079


namespace apartment_complex_equation_l166_166922

variable (x y z w v : ℝ)

def buildings := 8
def fullOccupancy := 3000
def currentOccupancy := 0.90 * fullOccupancy
def perBuildingOccupancy := currentOccupancy / buildings

-- Calculate the number of people currently living in each type of apartment
def studioOccupancy := 0.95 * x
def twoPersonOccupancy := 1.70 * y
def threePersonOccupancy := 2.40 * z
def fourPersonOccupancy := 3.00 * w
def fivePersonOccupancy := 3.25 * v

-- The sum of occupancies per building should equal perBuildingOccupancy
theorem apartment_complex_equation :
  0.11875 * x + 0.2125 * y + 0.3 * z + 0.375 * w + 0.40625 * v = perBuildingOccupancy := 
  by
    have h : currentOccupancy = 2700 := rfl
    have h_per_building : perBuildingOccupancy = 337.5 := by
      rw [perBuildingOccupancy, currentOccupancy]
      simp [h]
    exact h_per_building

end apartment_complex_equation_l166_166922


namespace find_integer_divisible_by_18_and_sqrt_between_26_and_26_2_l166_166575

theorem find_integer_divisible_by_18_and_sqrt_between_26_and_26_2 : 
    ∃ n : ℕ, (n % 18 = 0) ∧ (676 ≤ n ∧ n ≤ 686) ∧ n = 684 :=
by
  use 684
  split
  {
    norm_num
  }
  split
  {
    norm_num
  }
  {
    refl
  }

end find_integer_divisible_by_18_and_sqrt_between_26_and_26_2_l166_166575


namespace solve_inequality_l166_166731

theorem solve_inequality (x : ℝ) : (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
  sorry

end solve_inequality_l166_166731


namespace original_number_l166_166480

theorem original_number :
  ∃ x : ℝ, 0 < x ∧ (move_decimal_point x 3 = 3 / x) ∧ x = sqrt 30 / 100 :=
sorry

noncomputable def move_decimal_point (x : ℝ) (places : ℕ) : ℝ := x * 10^places

end original_number_l166_166480


namespace beef_weight_loss_percentage_l166_166917

theorem beef_weight_loss_percentage (weight_before weight_after weight_lost_percentage : ℝ) 
  (before_process : weight_before = 861.54)
  (after_process : weight_after = 560) 
  (weight_lost : (weight_before - weight_after) = 301.54)
  : weight_lost_percentage = 34.99 :=
by
  sorry

end beef_weight_loss_percentage_l166_166917


namespace complex_multiplication_result_l166_166934

-- Define the complex numbers used in the problem
def a : ℂ := 4 - 3 * Complex.I
def b : ℂ := 4 + 3 * Complex.I

-- State the theorem we want to prove
theorem complex_multiplication_result : a * b = 25 := 
by
  -- Proof is omitted
  sorry

end complex_multiplication_result_l166_166934


namespace proof_problem_l166_166163

-- Define conditions
def poly (n : ℕ) (x : ℝ) : ℝ := n * x^3 + 2 * x - n
def x_n (n : ℕ) (hx: n ≥ 2) : ℝ := sorry
def a_n (n : ℕ) (hx : n ≥ 2) : ℕ := Int.floor ((n + 1 : ℕ) * x_n n hx)

-- Statement of the problem
theorem proof_problem : (1 / 1005 : ℝ) * ∑ n in Finset.range 2010 \ {0, 1}, (a_n (2 + n) (by linarith)) = 2013 :=
by
  sorry

end proof_problem_l166_166163


namespace ratio_students_preference_l166_166265

theorem ratio_students_preference
  (total_students : ℕ)
  (mac_preference : ℕ)
  (windows_preference : ℕ)
  (no_preference : ℕ)
  (students_equally_preferred_both : ℕ)
  (h_total : total_students = 210)
  (h_mac : mac_preference = 60)
  (h_windows : windows_preference = 40)
  (h_no_pref : no_preference = 90)
  (h_students_equally : students_equally_preferred_both = total_students - (mac_preference + windows_preference + no_preference)) :
  (students_equally_preferred_both : ℚ) / mac_preference = 1 / 3 := 
by
  sorry

end ratio_students_preference_l166_166265


namespace sum_of_variables_l166_166178

theorem sum_of_variables (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2 * x + 4 * y - 6 * z + 14 = 0) : x + y + z = 2 :=
sorry

end sum_of_variables_l166_166178


namespace ratio_of_investments_l166_166489

variable (A B C : ℝ) (k : ℝ)

-- Conditions
def investments_ratio := (6 * k + 5 * k + 4 * k = 7250) ∧ (5 * k - 6 * k = 250)

-- Theorem we need to prove
theorem ratio_of_investments (h : investments_ratio k) : (A / B = 6 / 5) ∧ (B / C = 5 / 4) := 
  sorry

end ratio_of_investments_l166_166489


namespace sum_of_roots_2002_to_2010_l166_166097

noncomputable def f : ℝ → ℝ := sorry -- Definition of the function f satisfying the given conditions

axiom even_f : ∀ x, f (x) = f (-x)
axiom periodic_f : ∀ x, f (x + 2) = -f (x)
axiom monotonic_f : monotone (λ x, f x) -- Monotonic in [0, 2]
axiom exists_zero_f : ∃ x ∈ Ioo 0 2, f x = 0 -- Existence of x in (0, 2) such that f(x) = 0

theorem sum_of_roots_2002_to_2010 : 
  (∑ x in (finset.Icc 2002 2010).filter (λ x, f x = 0), x) = 8024 :=
  sorry

end sum_of_roots_2002_to_2010_l166_166097


namespace minimum_area_of_folded_triangle_l166_166920

theorem minimum_area_of_folded_triangle :
  let area := 1
  in ∃ (c : ℝ), 0 ≤ c ∧ c ≤ 1 ∧
  ((c ≤ (1:ℝ)/2 ∧ 1 - c^2 ≥ (3:ℝ)/4) ∨
  (c > (1:ℝ)/2 ∧ 3*c^2 - 4*c + 2 ≥ (2:ℝ)/3)) ∧
  ∀ (d : ℝ), 0 ≤ d ∧ d ≤ 1 → 
  ((d ≤ (1:ℝ)/2 → 1 - d^2 ≥ (3:ℝ)/4) ∧
  (d > (1:ℝ)/2 → 3*d^2 - 4*d + 2 ≥ (2:ℝ)/3)) :=
sorry

end minimum_area_of_folded_triangle_l166_166920


namespace extremum_only_at_2_l166_166616

noncomputable def f (x k : ℝ) : ℝ :=
  (exp x / x^2) - k * (2 / x + log x)

open Set

theorem extremum_only_at_2 (k : ℝ) :
  (∀ x > 0, deriv (λ x, f x k) x = 0 ↔ x = 2) → k ≤ real.exp 1 :=
by
  sorry

end extremum_only_at_2_l166_166616


namespace complex_magnitude_sum_l166_166346

-- Define the function that calculates the midpoint between two complex numbers
def midpoint (z1 z2 : ℂ) : ℂ :=
  (z1 + z2) / 2

-- The theorem proving |z1|^2 + |z2|^2 given the conditions
theorem complex_magnitude_sum
  (z1 z2 : ℂ)
  (h1 : |z1 + z2| = |z1 - z2|)
  (h2 : midpoint z1 z2 = 4 + 3 * complex.i) :
  |z1|^2 + |z2|^2 = 200 := 
sorry

end complex_magnitude_sum_l166_166346


namespace light_path_properties_l166_166520

noncomputable def point := ℝ × ℝ

def A : point := (2, 3)
def B : point := (1, 1)
def mirror (p : point) : Prop := p.1 + p.2 + 1 = 0

def incident_ray_eq : ℝ × ℝ := (5, -4)
def reflected_ray_eq : ℝ × ℝ := (4, -5)
def incident_constant : ℝ := -2
def reflected_constant : ℝ := -1
def path_length : ℝ := Real.sqrt 41

theorem light_path_properties :
  ∃ (eq1 eq2 : ℝ × ℝ) (c1 c2 : ℝ) (len : ℝ),
    eq1 = incident_ray_eq ∧
    eq2 = reflected_ray_eq ∧
    c1 = incident_constant ∧
    c2 = reflected_constant ∧
    len = path_length ∧
    (∀ p : point, p ∈ {p : point | eq1.1 * p.1 + eq1.2 * p.2 + c1 = 0} → mirror p) ∧
    (∀ p : point, p ∈ {p : point | eq2.1 * p.1 + eq2.2 * p.2 + c2 = 0} → mirror p) := 
by {
  use incident_ray_eq, reflected_ray_eq, incident_constant, reflected_constant, path_length,
  split; try {  simp },
  { split,
    { split; try { simp },
      { intro P,
        change P ∈ {P : point | incident_ray_eq.1 * P.1 + incident_ray_eq.2 * P.2 + incident_constant = 0} → mirror P,
        sorry },
      { intro P,
        change P ∈ {P : point | reflected_ray_eq.1 * P.1 + reflected_ray_eq.2 * P.2 + reflected_constant = 0} → mirror P,
        sorry }
    }
  } 
}

end light_path_properties_l166_166520


namespace shape_at_22_l166_166007

-- Define the pattern
def pattern : List String := ["triangle", "square", "diamond", "diamond", "circle"]

-- Function to get the nth shape in the repeated pattern sequence
def getShape (n : Nat) : String :=
  pattern.get! (n % pattern.length)

-- Statement to prove
theorem shape_at_22 : getShape 21 = "square" :=
by
  sorry

end shape_at_22_l166_166007


namespace complement_intersection_l166_166315

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection : 
  U = {1, 2, 3, 4, 5} → 
  A = {1, 2, 3} → 
  B = {2, 3, 4} → 
  (U \ (A ∩ B) = {1, 4, 5}) := 
by
  sorry

end complement_intersection_l166_166315


namespace trajectory_is_ellipse_l166_166206

open Set

noncomputable def circle (x y : ℝ) := (x + 2)^2 + y^2 = 36

def center (M : ℝ × ℝ) := M = (-2, 0)

def on_circle (A : ℝ × ℝ) : Prop := circle A.1 A.2

def point (N : ℝ × ℝ) := N = (2, 0)

def is_perpendicular_bisector (P A N : ℝ × ℝ) : Prop := dist P A = dist P N

def is_trajectory (P : ℝ × ℝ → ℝ) : Prop := ∃ M A N, center M ∧ point N ∧ on_circle A ∧ is_perpendicular_bisector P A N ∧ dist P M + dist P N = 6

theorem trajectory_is_ellipse : ∀ (P : ℝ × ℝ), is_trajectory P → ellipse P :=
sorry

end trajectory_is_ellipse_l166_166206


namespace probability_at_least_one_boy_one_girl_l166_166924

theorem probability_at_least_one_boy_one_girl :
  (∀ (P : SampleSpace → Prop), (P = (fun outcomes => nat.size outcomes = 4
                            ∧ (∃ outcome : outcomes, outcome = "boy")
                            ∧ ∃ outcome : outcomes, outcome = "girl"))
  -> (probability P = 7/8)) :=
by
  sorry

end probability_at_least_one_boy_one_girl_l166_166924


namespace total_birds_on_fence_l166_166498

-- Definitions based on conditions.
def initial_birds : ℕ := 12
def additional_birds : ℕ := 8

-- Theorem corresponding to the problem statement.
theorem total_birds_on_fence : initial_birds + additional_birds = 20 := by 
  sorry

end total_birds_on_fence_l166_166498


namespace original_number_l166_166477

theorem original_number :
  ∃ x : ℝ, 0 < x ∧ (move_decimal_point x 3 = 3 / x) ∧ x = sqrt 30 / 100 :=
sorry

noncomputable def move_decimal_point (x : ℝ) (places : ℕ) : ℝ := x * 10^places

end original_number_l166_166477


namespace coefficient_of_x6_in_expansion_l166_166789

theorem coefficient_of_x6_in_expansion : 
  let p := 2 + 3 * x^3 in
  (expand_nat_binomial 2 3 x 3 4).coeff 6 = 216 :=
by
  sorry

end coefficient_of_x6_in_expansion_l166_166789


namespace unique_subsequences_of_length_17_l166_166695

def is_balanced (n : ℕ) := 
  let digits := (List.range (n.toString.length)).map (λ i, (n / (10 ^ i)) % 10)
  (digits.length = 9) ∧ (∀ d ∈ digits, (1 ≤ d) ∧ (d ≤ 9)) ∧ (digits.nodup)

def S : List ℕ :=
  (List.range 1000000000).filter is_balanced

noncomputable def k := 17

theorem unique_subsequences_of_length_17 :
  ∀ subseq1 subseq2 : List ℕ,
    (subseq1.length = k) →
    (subseq2.length = k) →
    (subseq1 ≠ subseq2) :=
by
  sorry

end unique_subsequences_of_length_17_l166_166695


namespace initial_scooter_value_l166_166372

theorem initial_scooter_value (V : ℝ) 
    (h : (9 / 16) * V = 22500) : V = 40000 :=
sorry

end initial_scooter_value_l166_166372


namespace calculate_expression_l166_166550

noncomputable def sqrt4 : ℝ := real.sqrt 4
noncomputable def sin30 : ℝ := real.sin (real.pi / 6)
noncomputable def pi_minus_one_pow_zero : ℝ := (real.pi - 1)^0
noncomputable def two_neg_one : ℝ := 2^(-1)

theorem calculate_expression : sqrt4 - sin30 - pi_minus_one_pow_zero + two_neg_one = 1 := by
  sorry

end calculate_expression_l166_166550


namespace inequality_lambda_range_l166_166250

theorem inequality_lambda_range 
  (λ : ℝ)
  (h : ∀ x : ℝ, 0 < x ∧ x < 2 → sqrt (x * (x^2 + 8) * (8 - x)) < λ * (x + 1)) : 
  4 ≤ λ :=
by 
  sorry

end inequality_lambda_range_l166_166250


namespace pizza_area_percent_increase_l166_166243

noncomputable def radius_medium_pizza : ℝ := sorry
noncomputable def radius_large_pizza : ℝ := 1.20 * radius_medium_pizza
noncomputable def area_medium_pizza : ℝ := real.pi * radius_medium_pizza^2
noncomputable def area_large_pizza : ℝ := real.pi * (1.20 * radius_medium_pizza)^2

theorem pizza_area_percent_increase :
  (area_large_pizza - area_medium_pizza) / area_medium_pizza * 100 = 44 :=
by sorry

end pizza_area_percent_increase_l166_166243


namespace range_of_lambda_l166_166249

theorem range_of_lambda
  (h : ∀ (n : ℕ), 0 < n → ∀ (x : ℝ), x ≤ λ → (x^2 + (1/2) * x - (1/2)^n ≥ 0)) :
  λ ≤ -1 :=
sorry

end range_of_lambda_l166_166249


namespace fraction_division_l166_166032

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := 
by
  sorry

end fraction_division_l166_166032


namespace cuboid_count_l166_166635

def length_small (m : ℕ) : ℕ := 6
def width_small (m : ℕ) : ℕ := 4
def height_small (m : ℕ) : ℕ := 3

def length_large (m : ℕ): ℕ := 18
def width_large (m : ℕ) : ℕ := 15
def height_large (m : ℕ) : ℕ := 2

def volume (l : ℕ) (w : ℕ) (h : ℕ) : ℕ := l * w * h

def n_small_cuboids (v_large v_small : ℕ) : ℕ := v_large / v_small

theorem cuboid_count : 
  n_small_cuboids (volume (length_large 1) (width_large 1) (height_large 1)) (volume (length_small 1) (width_small 1) (height_small 1)) = 7 :=
by
  sorry

end cuboid_count_l166_166635


namespace number_of_zeros_f_l166_166000

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + 2 * x + 5

theorem number_of_zeros_f : 
  (∃ a b : ℝ, f a = 0 ∧ f b = 0 ∧ 0 < a ∧ 0 < b ∧ a ≠ b) ∧ ∀ c, f c = 0 → c = a ∨ c = b :=
by
  sorry

end number_of_zeros_f_l166_166000


namespace inequality_proof_l166_166197

variable {m n : ℕ}
variable {x : Fin m → ℝ}
variable {y : Fin n → ℝ}

-- Assume all x_i and y_i are positive
variable (hx : ∀ i, 0 < x i)
variable (hy : ∀ j, 0 < y j)

noncomputable def X : ℝ := Finset.univ.sum x
noncomputable def Y : ℝ := Finset.univ.sum y

theorem inequality_proof (hx : ∀ i, 0 < x i) (hy : ∀ j, 0 < y j) :
  2 * X x * Y y * (Finset.univ.sum $ λ i, Finset.univ.sum $ λ j, abs (x i - y j)) ≥
    X x ^ 2 * (Finset.univ.sum $ λ j, Finset.univ.sum $ λ l, abs (y j - y l)) +
    Y y ^ 2 * (Finset.univ.sum $ λ i, Finset.univ.sum $ λ k, abs (x i - x k)) :=
sorry

end inequality_proof_l166_166197


namespace compare_two_and_neg_three_l166_166121

theorem compare_two_and_neg_three (h1 : 2 > 0) (h2 : -3 < 0) : 2 > -3 :=
by
  sorry

end compare_two_and_neg_three_l166_166121


namespace sum_inverse_seq_l166_166205

variable (n : ℕ) (k : ℝ)
variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)

noncomputable def sum_seq_S (n : ℕ) := k * (n:ℝ)^2 - 1

axiom a_2_value : a_n 2 = 12
axiom S_sum_seq : ∀ (n : ℕ), S_n n = k * (n:ℝ)^2 - 1
axiom n_positive_nat : n > 0

theorem sum_inverse_seq (n : ℕ) (h1 : ∀ (n : ℕ), S_n n = sum_seq_S n) (h2 : a_n 2 = 12) : 
  (∑ i in finset.range (n + 1), (1 / S_n i)) = (n : ℚ) / ((2 * n) + 1 : ℚ) := 
  sorry

end sum_inverse_seq_l166_166205


namespace problem_I_problem_II_l166_166190

theorem problem_I (k : ℝ) (m : ℝ) (λ : ℝ) (hk : k ≠ 0) (eigenvector_condition : (∀ k, A * (Vector.mk k 0) = λ * (Vector.mk k 0))) :
  m = 0 ∧ λ = 1 :=
begin
  sorry
end

theorem problem_II (B A: Matrix (Fin 2) (Fin 2) ℝ)
  (hB : B = !![3, 2; 2, 1])
  (hA : A = !![1, 0; 0, 2]) :
  let B_inv := !![-1, 2; 2, -3],
  B_inv * A = !![-1, 4; 2, -6] :=
begin
  sorry
end

end problem_I_problem_II_l166_166190


namespace simplify_expr_l166_166727

variable (a b : ℝ)

theorem simplify_expr (h : a + b ≠ 0) : 
  a - b + 2 * b^2 / (a + b) = (a^2 + b^2) / (a + b) :=
sorry

end simplify_expr_l166_166727


namespace area_of_circle_d10_l166_166399

-- Definitions based on the conditions
def diameter (d : ℝ) := d
def radius (d : ℝ) := d / 2
def area_of_circle (r : ℝ) := Real.pi * r * r

-- The statement to prove
theorem area_of_circle_d10 : area_of_circle (radius (diameter 10)) = 25 * Real.pi :=
by
  sorry

end area_of_circle_d10_l166_166399


namespace determine_value_of_f_l166_166610

def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem determine_value_of_f :
  (∃ (α : ℝ), power_function α (1/2) = (sqrt 2) / 2) →
  power_function (1/2) 4 = 2 :=
by intros h; sorry

end determine_value_of_f_l166_166610


namespace altitude_geometric_mean_of_projections_legs_geometric_mean_of_hypotenuse_and_projections_l166_166722

variables {Point : Type}
variables [HasDistance Point]
variables {A B C H : Point}
variables {CH : Point → Point → ℝ}
variables {projections : Point → Point → Point → ℝ}

-- Conditions:
-- Right triangle ABC with right angle at C and H as the foot of the altitude CH from C to hypotenuse AB
def is_right_triangle (A B C : Point) : Prop := sorry
def is_altitude (A B C H : Point) (CH : Point → Point → ℝ) : Prop := sorry
def is_projection (A B C H : Point) (projections : Point → Point → Point → ℝ) : Prop := sorry
def is_similar (ABC AHC CHB : Point → Point → Point → Point → Prop) : Prop := sorry -- Similarity of triangles

-- Questions to be proved:
-- 1. Altitude is the geometric mean of the projections of the legs onto the hypotenuse
-- 2. Each leg is the geometric mean of the hypotenuse and its projection onto it

theorem altitude_geometric_mean_of_projections (A B C H : Point) (CH : Point → Point → ℝ)
  (projections : Point → Point → Point → ℝ)
  (h1 : is_right_triangle A B C) (h2 : is_altitude A B C H CH)
  (h3 : is_projection A B C H projections) (h4 : is_similar ABC AHC CHB) : 
  CH = sqrt (projections A H * projections B H) := by 
    sorry

theorem legs_geometric_mean_of_hypotenuse_and_projections (A B C H : Point) (CH : Point → Point → ℝ)
  (projections : Point → Point → Point → ℝ)
  (h1 : is_right_triangle A B C) (h2 : is_altitude A B C H CH)
  (h3 : is_projection A B C H projections) (h4 : is_similar ABC AHC CHB) : 
  (distance A C = sqrt (distance A B * projections A H)) 
  ∧ (distance B C = sqrt (distance A B * projections B H)) := by 
    sorry

end altitude_geometric_mean_of_projections_legs_geometric_mean_of_hypotenuse_and_projections_l166_166722


namespace unit_price_B_discount_relation_buy_for_270_l166_166049

open Nat

-- Define the conditions
def price_A (a : ℕ) := a - 2
def condition_1 (a : ℕ) := 60 / (a - 2) = 100 / a

def discount_plan_1 := (x : ℕ) → (1 ≤ x ∧ x ≤ 20) → 4.5 * x
def discount_plan_2 := (x : ℕ) → (x > 20) → 4 * x + 10

-- Statements:
-- Prove the unit price of type B paintbrushes equals 5 yuan.
theorem unit_price_B : ∃ a : ℕ, condition_1 a ∧ a = 5 :=
by
  exists 5
  sorry

-- Prove the function relationship y(x).
theorem discount_relation : (∀ x, (1 ≤ x ∧ x ≤ 20) → y = 4.5 * x) ∧ (∀ x, (x > 20) → y = 4 * x + 10)
  :=
by
  sorry

-- Prove that Xiaogang can buy 65 type B paintbrushes for 270 yuan.
theorem buy_for_270 : ∃ (x : ℕ), (discount_plan_1 x ∨ discount_plan_2 x ) ∧ y = 270 → x = 65 :=
by
  exists 65
  sorry

end unit_price_B_discount_relation_buy_for_270_l166_166049


namespace smallest_value_of_abs_sum_l166_166423

theorem smallest_value_of_abs_sum : ∃ x : ℝ, (x = -3) ∧ ( ∀ y : ℝ, |y + 1| + |y + 3| + |y + 6| ≥ 5 ) :=
by
  use -3
  split
  . exact rfl
  . intro y
    have h1 : |y + 1| + |y + 3| + |y + 6| = sorry,
    sorry

end smallest_value_of_abs_sum_l166_166423


namespace problem_ordering_l166_166221

noncomputable def f (x : ℝ) : ℝ := 2^x + x
noncomputable def g (x : ℝ) : ℝ := x - 2
noncomputable def h (x : ℝ) : ℝ := Real.log x / Real.log 2 + x

lemma zero_of_f : ∃ a : ℝ, f a = 0 ∧ a < 0 :=
begin
  use -1/2,
  sorry
end

lemma zero_of_g : ∃ b : ℝ, g b = 0 ∧ b = 2 :=
begin
  use 2,
  sorry
end

lemma zero_of_h : ∃ c : ℝ, h c = 0 ∧ 0.5 < c ∧ c < 1 :=
begin
  use 0.75,
  sorry
end

theorem problem_ordering : ∃ a b c : ℝ,
  (f a = 0 ∧ a < 0) ∧ 
  (g b = 0 ∧ b = 2) ∧ 
  (h c = 0 ∧ 0.5 < c ∧ c < 1) 
  ∧ a < c ∧ c < b :=
by {
  obtain ⟨a, hfa, ha⟩ := zero_of_f,
  obtain ⟨b, hgb, hb⟩ := zero_of_g,
  obtain ⟨c, hhc, hc1, hc2⟩ := zero_of_h,
  exact ⟨a, b, c, ⟨hfa, ha⟩, ⟨hgb, hb⟩, ⟨hhc, hc1, hc2⟩, ha.trans hc1, hc2.trans hb.ge⟩
}

end problem_ordering_l166_166221


namespace find_a_c_and_cos_B_minus_C_l166_166650

theorem find_a_c_and_cos_B_minus_C 
  {a b c : ℝ}
  {A B C : ℝ} 
  (h_sin_A_gt_sin_C : sin A > sin C)
  (h_dot_product : vector.dot_product (vector.ofFn [A, B]) (vector.ofFn [B, C]) = -2)
  (h_cos_B : cos B = 1 / 3)
  (h_b_eq_3 : b = 3)
  (h_a_side_def : a = B)
  (h_c_side_def : c = C):
  (a = 3 ∧ c = 2) ∧ cos (B - C) = 23 / 27 := sorry

end find_a_c_and_cos_B_minus_C_l166_166650


namespace ellipse_eccentricity_theorem_l166_166746

def ellipse_eccentricity (x y : ℝ) : ℝ :=
    √((2^2 - 1^2)) / 2

theorem ellipse_eccentricity_theorem : 
    (ellipse_eccentricity 4 1) = √3/2 := 
sorry

end ellipse_eccentricity_theorem_l166_166746


namespace quadratic_ineq_solution_l166_166766

theorem quadratic_ineq_solution (a b : ℝ) 
  (h_solution_set : ∀ x, (ax^2 + bx - 1 > 0) ↔ (1 / 3 < x ∧ x < 1))
  (h_roots : (a / 3 + b = -1 / a) ∧ (a / 3 = -1 / a)) 
  (h_a_neg : a < 0) : a + b = 1 := 
sorry 

end quadratic_ineq_solution_l166_166766


namespace exponent_subtraction_l166_166237

theorem exponent_subtraction (m n : ℝ) (h1 : 2^m = 3) (h2 : 2^n = 4) : 2^(m - n) = 3 / 4 :=
by
  sorry

end exponent_subtraction_l166_166237


namespace product_of_numbers_is_2014_pow_1007_sqrt_2014_l166_166773

-- Given condition definitions
variable (numbers : List ℝ)
variable (h_distinct : numbers.Nodup)
variable (h_positive : ∀ x ∈ numbers, x > 0)
variable (h_count : numbers.length = 2015)
variable (h_parity : ∀ a > 0, 
  Parity (numbers.filter (λ x, x < 2014 / a)).length = Parity (numbers.filter (λ x, x > a)).length)

-- Lean theorem statement
theorem product_of_numbers_is_2014_pow_1007_sqrt_2014 : 
  numbers.Prod = 2014 ^ 1007 * Real.sqrt 2014 := by
sorry

end product_of_numbers_is_2014_pow_1007_sqrt_2014_l166_166773


namespace range_of_a_l166_166181

-- Assuming all necessary imports and definitions are included

variable {R : Type} [LinearOrderedField R]

def satisfies_conditions (f : R → R) (a : R) : Prop :=
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∀ x y, 1 ≤ x → x < y → f x < f y) ∧
  (∀ x, (1/2 : R) ≤ x ∧ x ≤ 1 → f (a * x) < f (x - 1))

theorem range_of_a (f : R → R) (a : R) :
  satisfies_conditions f a → 0 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l166_166181


namespace relationship_ab_l166_166240

noncomputable def a : ℝ := Real.log 243 / Real.log 5
noncomputable def b : ℝ := Real.log 27 / Real.log 3

theorem relationship_ab : a = (5 / 3) * b := sorry

end relationship_ab_l166_166240


namespace charlie_delta_total_products_l166_166902

-- Define the conditions:
def oreo_flavors : ℕ := 8
def milk_types : ℕ := 4
def charlie_constraint : ℕ := oreo_flavors + milk_types
def delta_constraint : ℕ := oreo_flavors
def total_products : ℕ := 5

-- Define the proof problem:
theorem charlie_delta_total_products : 
  (∑ k in Finset.range (total_products + 1),
     (Nat.choose charlie_constraint k) * 
     match total_products - k with 
     | 0 => 1
     | 1 => delta_constraint
     | 2 => (Nat.choose delta_constraint 2) + delta_constraint
     | 3 => (Nat.choose delta_constraint 3) + 
             (delta_constraint * (delta_constraint - 1)) + 
             delta_constraint
     | 4 => (Nat.choose delta_constraint 4) + 
             ((Nat.choose delta_constraint 2) * 
             (Nat.choose (delta_constraint - 2) 2)) / 2 + 
             (delta_constraint * (delta_constraint - 1)) + 
             delta_constraint
     | 5 => (Nat.choose delta_constraint 5) + 
             ((Nat.choose delta_constraint 2) * 
             (Nat.choose (delta_constraint - 3) 3)) + 
             (delta_constraint * (delta_constraint - 1)) + 
             delta_constraint
     | _ => 0) = 25512 := by sorry

end charlie_delta_total_products_l166_166902


namespace fraction_division_l166_166035

theorem fraction_division : (3/4) / (5/8) = (6/5) := by
  sorry

end fraction_division_l166_166035


namespace sum_of_r_b_l166_166586

-- Definitions provided by conditions
def r_x (x y : ℕ) : ℕ :=
  Nat.find (λ r, r > 0 ∧ r % x = y % x)

-- The main theorem to prove
theorem sum_of_r_b (a b n : ℕ) (h_a : 0 < a) (h_b : 0 < b) (h_n : 0 < n) :
  ∑ i in Finset.range n, r_x b (a * (i + 1)) ≤ (n * (a + b)) / 2 :=
sorry

end sum_of_r_b_l166_166586


namespace walking_speed_increase_factor_l166_166073

theorem walking_speed_increase_factor :
  ∀ (t1 t2 v1 v2 : ℚ),
    t1 > 0 ∧ t2 > 0 ∧ 
    v1 > 0 ∧ v2 > 0 ∧ 
    125 = 45 + 80 ∧
    125 = 55 + 70 ∧ 
    (80 / t1) = (70 / t2) ∧ 
    v1 = 45 / t1 ∧ 
    v2 = 55 / t2 → 
    v2 / v1 = 1.397 := 
by
  intros t1 t2 v1 v2 ht1t2 hv1v2 Heq1 Heq2 Hspeed Heqv1 Heqv2
  -- The proof should go here
  sorry

end walking_speed_increase_factor_l166_166073


namespace shared_vertex_probability_correct_m_n_sum_l166_166126

-- Define the grid and the properties
def side_length : ℕ := 5
def total_triangles : ℕ := 55 -- Sum of squares of first 5 natural numbers

-- Number of ways to choose 2 triangles out of total_triangles
def total_ways_to_choose_two : ℕ := (total_triangles * (total_triangles - 1)) / 2

-- Number of pairs that share exactly one vertex
def num_pairs_sharing_one_vertex : ℕ := 450 -- Calculated from cases in solution

-- Probability in simplified form as a rational number
def shared_vertex_probability : ℚ := num_pairs_sharing_one_vertex /. total_ways_to_choose_two

-- Fractions are relatively prime
def m : ℕ := 10
def n : ℕ := 33

theorem shared_vertex_probability_correct :
  shared_vertex_probability = 10 /. 33 := sorry

theorem m_n_sum :
  m + n = 43 := by
  exact rfl

end shared_vertex_probability_correct_m_n_sum_l166_166126


namespace part1_part2_part3_l166_166219

noncomputable theory
open Set

variable {R : Type} [linear_order R] [conditionally_complete_linear_order R]

def A : Set R := {x | 2 ≤ x ∧ x < 4}
def B : Set R := {x | 3 * x - 7 ≥ 8 - 2 * x}
def C (a : R) : Set R := {x | x < a}

theorem part1 : A ∩ B = {x : R | 3 ≤ x ∧ x < 4} :=
sorry

theorem part2 : A ∪ (R \ B) = {x : R | x < 4} :=
sorry

theorem part3 (a : R) : A ⊆ C a → a ≥ 4 :=
sorry

end part1_part2_part3_l166_166219


namespace lcm_gcd_product_eq_product_12_15_l166_166992

theorem lcm_gcd_product_eq_product_12_15 :
  lcm 12 15 * gcd 12 15 = 12 * 15 :=
sorry

end lcm_gcd_product_eq_product_12_15_l166_166992


namespace polygon_incircle_triangle_inequality_l166_166332

theorem polygon_incircle_triangle_inequality
  (polygon : Type)
  (sides : polygon → polygon → ℝ)
  (tangent_circle : polygon → polygon → ℝ)
  (longest_side DE : polygon)
  (adjacent_sides CD EF : polygon)
  (x y : ℝ)
  (Tangent_DKDI : tangent_circle CD = x)
  (Tangent_DI : tangent_circle DE = x)
  (Tangnet_IE : tangent_circle DE = y)
  (Tangent_IE : tangent_circle EF = y)
  (Longest_side_eq : sides longest_side DE = x + y) :
  sides longest_side DE < sides longest_side CD + sides longest_side EF :=
  sorry

end polygon_incircle_triangle_inequality_l166_166332


namespace initial_money_given_l166_166513

def bracelet_cost : ℕ := 15
def necklace_cost : ℕ := 10
def mug_cost : ℕ := 20
def num_bracelets : ℕ := 3
def num_necklaces : ℕ := 2
def num_mugs : ℕ := 1
def change_received : ℕ := 15

theorem initial_money_given : num_bracelets * bracelet_cost + num_necklaces * necklace_cost + num_mugs * mug_cost + change_received = 100 := 
sorry

end initial_money_given_l166_166513


namespace find_original_number_l166_166471

theorem find_original_number (x : ℝ) (hx : 0 < x) 
  (h : 1000 * x = 3 * (1 / x)) : x ≈ 0.01732 :=
by 
  sorry

end find_original_number_l166_166471


namespace relationship_among_abc_l166_166904

noncomputable theory

def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ π / 2 then x^3 * Real.sin x else sorry

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def a : ℝ := f (Real.sin (π / 3))
def b : ℝ := f (Real.sin 2)
def c : ℝ := f (Real.sin 3)

theorem relationship_among_abc (hf_even : is_even f)
  (hf_defined : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x = x^3 * Real.sin x) :
  b > a ∧ a > c := 
sorry

end relationship_among_abc_l166_166904


namespace max_min_PF_PA_l166_166614

open Real

-- Definitions based on conditions in a)

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 8) + (y^2 / 4) = 1

def point_A := (1, 1 : ℝ × ℝ)
def foci_1 := (-2, 0 : ℝ × ℝ)
def foci_2 := (2, 0 : ℝ × ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def P_on_ellipse (P : ℝ × ℝ) := ellipse P.1 P.2

-- Proof goal based on c)
theorem max_min_PF_PA :
  ∀ (P : ℝ × ℝ), P_on_ellipse P →
  min (distance P foci_1 + distance P point_A) (distance P foci_1 + distance P foci_2) = 3 * sqrt 2 ∧
  max (distance P foci_1 + distance P point_A) (distance P foci_1 + distance P foci_2) = 5 * sqrt 2 :=
sorry

end max_min_PF_PA_l166_166614


namespace maximize_sum_is_24_l166_166699

-- Here we assume A, B, C, and D are different digits and prove that A + B + C = 24
noncomputable def maximize_sum_digits (A B C D : ℕ) : Prop :=
  0 ≤ A ∧ A ≤ 9 ∧
  0 ≤ B ∧ B ≤ 9 ∧
  0 ≤ C ∧ C ≤ 9 ∧
  0 ≤ D ∧ D ≤ 9 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A + B + C = 24 ∧
  ∃ k : ℕ, A + B + C = D * k

-- Proof that the maximum value of A + B + C is 24
theorem maximize_sum_is_24 :
  ∃ A B C D : ℕ, maximize_sum_digits A B C D :=
by
  use 9, 8, 7, 1
  have h : maximize_sum_digits 9 8 7 1 := by
    simp [maximize_sum_digits]
    split
    repeat {norm_num}
    intros
    finish
  exact h

end maximize_sum_is_24_l166_166699


namespace f_spec_l166_166594

-- Define the function f
def f (x : ℝ) : ℝ := if x ∈ set.Ico 0 1 then 2^x - real.sqrt 2 else _

-- Lean doesn't support piecewise native, workaround with setting placeholder
lemma f_periodic (x : ℝ) : f (x + 2) = f x := sorry
lemma f_odd (x : ℝ) : f (-x) = -f x := sorry

-- The main proof statement
theorem f_spec :
  f (real.logb (1/2) (4 * real.sqrt 2)) = 0 :=
by  
  have H0 : real.logb (1/2) (4 * real.sqrt 2) = -5 / 2 := sorry,
  have H1 : f(-5 / 2) = f(-5 / 2 + 2) := f_periodic (-5 / 2),
  have H2 : f(-5 / 2 + 2) = f(-1 / 2) := by rw [add_assoc, sub_self 2, add_zero],
  have H3 : f(-1 / 2) = -f(1 / 2) := f_odd (1 / 2),
  have H4 : f(1 / 2) = 2 ** (1 / 2) - real.sqrt 2 := sorry, -- derived from initial definition on the interval [0, 1)
  have H5 : 2 ** (1 / 2) = real.sqrt 2 := sorry, -- property of exponentiation
  have H6 : f(1 / 2) = real.sqrt 2 - real.sqrt 2 := by rw [H4, H5],
  rw [H1, H2, H3, H6],
  exact zero_sub (sqrt 2 - sqrt 2)

end f_spec_l166_166594


namespace min_side_length_l166_166094

def table_diagonal (w h : ℕ) : ℕ :=
  Nat.sqrt (w * w + h * h)

theorem min_side_length (w h : ℕ) (S : ℕ) (dw : w = 9) (dh : h = 12) (dS : S = 15) :
  S >= table_diagonal w h :=
by
  sorry

end min_side_length_l166_166094


namespace midpoint_distance_l166_166602

variable (A B O M : (ℝ × ℝ))

variable H1 : A = (-1, 5)
variable H2 : B = (3, -7)
variable H3 : O = (0, 0)

def midpoint (P Q : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def distance (P Q : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem midpoint_distance : distance O (midpoint A B) = Real.sqrt 2 :=
by
  simp [H1, H2, H3, midpoint, distance]
  sorry

end midpoint_distance_l166_166602


namespace intersection_A1_B1_complement_A1_B1_union_A2_B2_l166_166063

-- Problem 1: Intersection and Complement
def setA1 : Set ℕ := {x : ℕ | x > 0 ∧ x < 9}
def setB1 : Set ℕ := {1, 2, 3}

theorem intersection_A1_B1 : (setA1 ∩ setB1) = {1, 2, 3} := by
  sorry

theorem complement_A1_B1 : {x : ℕ | x ∈ setA1 ∧ x ∉ setB1} = {4, 5, 6, 7, 8} := by
  sorry

-- Problem 2: Union
def setA2 : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def setB2 : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem union_A2_B2 : (setA2 ∪ setB2) = {x : ℝ | (-3 < x ∧ x < 1) ∨ (2 < x ∧ x < 10)} := by
  sorry

end intersection_A1_B1_complement_A1_B1_union_A2_B2_l166_166063


namespace find_original_number_l166_166469

theorem find_original_number (x : ℝ) (hx : 0 < x) 
  (h : 1000 * x = 3 * (1 / x)) : x ≈ 0.01732 :=
by 
  sorry

end find_original_number_l166_166469


namespace problem1_problem2_l166_166112

theorem problem1 : sqrt 8 - 2 * sqrt 18 + sqrt 24 = -4 * sqrt 2 + 2 * sqrt 6 :=
by
  sorry

theorem problem2 : (sqrt (4 / 3) + sqrt 3) * sqrt 12 - sqrt 48 + sqrt 6 = 10 - 4 * sqrt 3 + sqrt 6 :=
by
  sorry

end problem1_problem2_l166_166112


namespace distance_between_planes_l166_166148

def plane1 (x y z : ℝ) := 3 * x - y + z - 3 = 0
def plane2 (x y z : ℝ) := 6 * x - 2 * y + 2 * z + 4 = 0

theorem distance_between_planes :
  ∃ d : ℝ, d = (5 * Real.sqrt 11) / 11 ∧ 
            ∀ x y z : ℝ, plane1 x y z → plane2 x y z → d = (5 * Real.sqrt 11) / 11 :=
sorry

end distance_between_planes_l166_166148


namespace maximize_probability_nonneg_l166_166313

open ProbabilityTheory

variables {Ω : Type*} [MeasurableSpace Ω] (ξ : Ω → ℝ) (a m : ℝ) 

theorem maximize_probability_nonneg (h1 : ∀ ω, 0 ≤ ξ ω)
    (h2 : ∫ ω, ξ ω ∂ measure_theory.measure_space.volume = m) :
  P (λ ω, ξ ω ≥ a) = 
  if a ≤ m 
  then 1 
  else m / a :=
by
  sorry
  
end maximize_probability_nonneg_l166_166313


namespace percent_of_total_l166_166502

theorem percent_of_total (p n : ℝ) (h1 : p = 35 / 100) (h2 : n = 360) : p * n = 126 := by
  sorry

end percent_of_total_l166_166502


namespace smallest_r_l166_166195

variables (p q r s : ℤ)

-- Define the conditions
def condition1 : Prop := p + 3 = q - 1
def condition2 : Prop := p + 3 = r + 5
def condition3 : Prop := p + 3 = s - 2

-- Prove that r is the smallest
theorem smallest_r (h1 : condition1 p q) (h2 : condition2 p r) (h3 : condition3 p s) : r < p ∧ r < q ∧ r < s :=
sorry

end smallest_r_l166_166195


namespace retailer_markup_percentage_l166_166077

-- Definitions of initial conditions
def CP : ℝ := 100
def intended_profit_percentage : ℝ := 0.25
def discount_percentage : ℝ := 0.25
def actual_profit_percentage : ℝ := 0.2375

-- Proving the retailer marked his goods at 65% above the cost price
theorem retailer_markup_percentage : ∃ (MP : ℝ), ((0.75 * MP - CP) / CP) * 100 = actual_profit_percentage * 100 ∧ ((MP - CP) / CP) * 100 = 65 := 
by
  -- The mathematical proof steps mean to be filled here  
  sorry

end retailer_markup_percentage_l166_166077


namespace probability_not_order_dessert_l166_166056

def P (A : Prop) : ℝ

variables (D : Prop) (C : Prop)

-- Given conditions
axiom P_D : P D = 0.6
axiom P_not_C_given_D : P (C ∧ D) = 0.8 * P D

-- Define the complement probability we need to prove
theorem probability_not_order_dessert : P (¬D) = 0.4 :=
by
  -- Proof omitted
  sorry

end probability_not_order_dessert_l166_166056


namespace sum_of_first_three_terms_l166_166753

theorem sum_of_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 8) 
  (h5 : a 5 = 12) 
  (h6 : a 6 = 16) : 
  a 1 + a 2 + a 3 = 0 :=
by
  sorry

end sum_of_first_three_terms_l166_166753


namespace boat_speed_problem_l166_166896

def speed_of_boat_in_still_water (b : ℝ) : Prop :=
  (36 / (b - 2) - 36 / (b + 2) = 1.5) → b = 10

theorem boat_speed_problem : ∃ b : ℝ, speed_of_boat_in_still_water b :=
begin
  use 10,
  sorry
end

end boat_speed_problem_l166_166896


namespace solve_inequality_l166_166732

theorem solve_inequality (x : ℝ) : (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
  sorry

end solve_inequality_l166_166732


namespace secondTrain_speed_l166_166029

/-
Conditions:
1. Two trains start from A and B and travel towards each other.
2. The distance between them is 1100 km.
3. At the time of their meeting, one train has traveled 100 km more than the other.
4. The first train's speed is 50 kmph.
-/

-- Let v be the speed of the second train
def secondTrainSpeed (v : ℝ) : Prop :=
  ∃ d : ℝ, 
    d > 0 ∧
    v > 0 ∧
    (d + (d - 100) = 1100) ∧
    ((d / 50) = ((d - 100) / v))

-- Here is the main theorem translating the problem statement:
theorem secondTrain_speed :
  secondTrainSpeed (250 / 6) :=
by
  sorry

end secondTrain_speed_l166_166029


namespace travel_agency_comparison_l166_166762

variable (x : ℕ)

def cost_A (x : ℕ) : ℕ := 150 * x
def cost_B (x : ℕ) : ℕ := 160 * x - 160

theorem travel_agency_comparison (x : ℕ) : 150 * x < 160 * x - 160 → x > 16 :=
by
  intro h
  linarith

end travel_agency_comparison_l166_166762


namespace regression_eq_change_in_y_l166_166597

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 2 - 1.5 * x

-- Define the statement to be proved
theorem regression_eq_change_in_y (x : ℝ) :
  regression_eq (x + 1) = regression_eq x - 1.5 :=
by sorry

end regression_eq_change_in_y_l166_166597


namespace min_abs_sum_is_5_l166_166435

noncomputable def min_abs_sum (x : ℝ) : ℝ :=
  |x + 1| + |x + 3| + |x + 6|

theorem min_abs_sum_is_5 : ∃ x : ℝ, (∀ y : ℝ, min_abs_sum y ≥ min_abs_sum x) ∧ min_abs_sum x = 5 :=
by
  use -3
  sorry

end min_abs_sum_is_5_l166_166435


namespace blue_pill_cost_l166_166087

theorem blue_pill_cost
  (days : ℕ)
  (total_cost : ℤ)
  (cost_diff : ℤ)
  (daily_cost : ℤ)
  (y : ℤ) : 
  days = 21 →
  total_cost = 966 →
  cost_diff = 2 →
  daily_cost = total_cost / days →
  daily_cost = 46 →
  2 * y - cost_diff = daily_cost →
  y = 24 := 
by
  intros days_eq total_cost_eq cost_diff_eq daily_cost_eq d_cost_eq daily_eq_46;
  sorry

end blue_pill_cost_l166_166087


namespace min_measurements_3x3_grid_min_measurements_5x5_grid_l166_166054

-- Proof problem for the 3x3 grid
theorem min_measurements_3x3_grid : 
  (∃ n (is_min : n ≥ 0), ∀ (nodes : Fin 16 → Fin 16), (∀ (i j : Fin 16), i ≠ j → (∃ k : Fin n, is_path_connected nodes i j k)) ↔ n = 8) := 
sorry

-- Proof problem for the 5x5 grid
theorem min_measurements_5x5_grid : 
  (∃ n (is_min : n ≥ 0), ∀ (nodes : Fin 36 → Fin 36), (∀ (i j : Fin 36), i ≠ j → (∃ k : Fin n, is_path_connected nodes i j k)) ↔ n = 18) := 
sorry

end min_measurements_3x3_grid_min_measurements_5x5_grid_l166_166054


namespace original_number_l166_166481

theorem original_number :
  ∃ x : ℝ, 0 < x ∧ (move_decimal_point x 3 = 3 / x) ∧ x = sqrt 30 / 100 :=
sorry

noncomputable def move_decimal_point (x : ℝ) (places : ℕ) : ℝ := x * 10^places

end original_number_l166_166481


namespace no_partition_1_to_33_into_11_groups_of_3_elements_with_sum_property_l166_166665

theorem no_partition_1_to_33_into_11_groups_of_3_elements_with_sum_property :
  ¬(∃ G : Finset (Finset ℕ), G.card = 11 ∧ (∀ g ∈ G, g.card = 3 ∧ ∃ x y z ∈ g, z = x + y) ∧ 
   (∀ g ∈ G, ∃ x y z ∈ g, x + y + z = x + y + (x + y)) ∧ (Finset.sum Finset.univ id = 561)) :=
sorry

end no_partition_1_to_33_into_11_groups_of_3_elements_with_sum_property_l166_166665


namespace tens_digit_of_9_pow_1010_l166_166862

theorem tens_digit_of_9_pow_1010 : (9 ^ 1010) % 100 = 1 :=
by sorry

end tens_digit_of_9_pow_1010_l166_166862


namespace original_number_value_l166_166450

noncomputable def orig_number_condition (x : ℝ) : Prop :=
  1000 * x = 3 * (1 / x)

theorem original_number_value : ∃ x : ℝ, 0 < x ∧ orig_number_condition x ∧ x = √(30) / 100 :=
begin
  -- the proof
  sorry
end

end original_number_value_l166_166450


namespace number_of_good_subsets_of_S_l166_166308

open Finset

def S : Finset ℕ := (finset.range 50).image (λ n, n + 1)

def is_good (s : Finset ℕ) : Prop := s.card = 3 ∧ (s.sum id) % 3 = 0

def count_good_subsets : ℕ := (S.subsets_of_card 3).filter is_good |>.card

theorem number_of_good_subsets_of_S :
  count_good_subsets = 6544 := by
  sorry

end number_of_good_subsets_of_S_l166_166308


namespace largest_of_sums_l166_166118

noncomputable def a1 := (1 / 4 : ℚ) + (1 / 5 : ℚ)
noncomputable def a2 := (1 / 4 : ℚ) + (1 / 6 : ℚ)
noncomputable def a3 := (1 / 4 : ℚ) + (1 / 3 : ℚ)
noncomputable def a4 := (1 / 4 : ℚ) + (1 / 8 : ℚ)
noncomputable def a5 := (1 / 4 : ℚ) + (1 / 7 : ℚ)

theorem largest_of_sums :
  max a1 (max a2 (max a3 (max a4 a5))) = 7 / 12 :=
by sorry

end largest_of_sums_l166_166118


namespace speed_of_stream_l166_166486

def boatSpeedDownstream (V_b V_s : ℝ) : ℝ :=
  V_b + V_s

def boatSpeedUpstream (V_b V_s : ℝ) : ℝ :=
  V_b - V_s

theorem speed_of_stream (V_b V_s : ℝ) (h1 : V_b + V_s = 25) (h2 : V_b - V_s = 5) : V_s = 10 :=
by {
  sorry
}

end speed_of_stream_l166_166486


namespace rhombus_area_l166_166881

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 21) : 
  (d1 * d2) / 2 = 157.5 :=
by
  sorry

end rhombus_area_l166_166881


namespace total_study_time_l166_166675

/-- Joey's SAT study schedule conditions. -/
variables
  (weekday_hours_per_night : ℕ := 2)
  (weekday_nights_per_week : ℕ := 5)
  (weekend_hours_per_day : ℕ := 3)
  (weekend_days_per_week : ℕ := 2)
  (weeks_until_exam : ℕ := 6)

/-- Total time Joey will spend studying for his SAT exam. -/
theorem total_study_time :
  (weekday_hours_per_night * weekday_nights_per_week + weekend_hours_per_day * weekend_days_per_week) * weeks_until_exam = 96 :=
by
  sorry

end total_study_time_l166_166675


namespace problem_lean_l166_166099

open EuclideanGeometry

def conditions (A B C D G E F : Point) (Γ : Circle) : Prop :=
  ∃(a b c d ac bd ad bc : ℝ), 
    tri_ang A B C ∧ 
    tri_point_in_triangle D A B C ∧
    angle_eq (angle A D B) (angle A C B + 90) ∧
    mul_eq ac bd ad bc

theorem problem_lean (A B C D G E F : Point) (Γ : Circle) (h₁ : tri_ang A B C) 
  (h₂ : tri_point_in_triangle D A B C) 
  (h₃ : angle_eq (angle A D B) (angle A C B + 90)) 
  (h₄ : mul_eq (AC.mul BD) (AD.mul BC)) : 
  (segment_eq (segment_EQ E F) (segment_EQ F G)) ∧ 
  (area_eq (triangle_EFG) (circle_area π)) :=
sorry

end problem_lean_l166_166099


namespace volume_of_regular_triangular_pyramid_l166_166972

theorem volume_of_regular_triangular_pyramid (h m : ℝ) (h_pos : 0 < h) (m_pos : 0 < m) :
  volume_of_pyramid h m = (Real.sqrt 3 / 27) * h^2 * Real.sqrt (9 * m^2 - h^2) :=
by
  -- definitions
  let AB := 2 * h / Real.sqrt 3
  let S := (1 / 2) * AB * h
  let MO := h / 3
  let H := Real.sqrt (m^2 - MO^2)
  let V := (1 / 3) * S * H
  
  -- final expression for volume
  have volume_calculated : V = (Real.sqrt 3 / 27) * h^2 * Real.sqrt (9 * m^2 - h^2) := sorry
  
  -- conclude
  exact volume_calculated

noncomputable def volume_of_pyramid (h m : ℝ) : ℝ :=
  sorry -- placeholder for the actual volume computation

end volume_of_regular_triangular_pyramid_l166_166972


namespace calf_additional_grazing_area_l166_166914

noncomputable def pi : ℝ := 3.14159

/-- 
Initial radius of grazing area -/
def initial_radius : ℝ := 9

/-- 
Extended radius of grazing area -/
def extended_radius : ℝ := 35

/-- 
Side length of each square obstacle -/
def obstacle_side : ℝ := 3

/-- 
Number of obstacles -/
def number_of_obstacles : ℝ := 2

/-- 
Calculate area of a circle given its radius -/
def circle_area (radius : ℝ) : ℝ := pi * radius * radius

/-- 
Area of the initial grazing circle -/
def initial_area : ℝ := circle_area initial_radius

/-- 
Area of the extended grazing circle -/
def extended_area : ℝ := circle_area extended_radius

/-- 
Total area of the obstacles -/
def obstacles_area : ℝ := number_of_obstacles * obstacle_side * obstacle_side

/-- 
Additional grazing area after rope extension without considering obstacles -/
def additional_area_without_obstacles : ℝ := extended_area - initial_area

/-- 
Additional grazing area considering obstacles -/
def additional_area_with_obstacles : ℝ := additional_area_without_obstacles - obstacles_area

theorem calf_additional_grazing_area : 
  abs (additional_area_with_obstacles - 3575.76) < 0.1 := 
by
  sorry

end calf_additional_grazing_area_l166_166914


namespace min_abs_sum_l166_166414

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem min_abs_sum :
  ∃ (x : ℝ), (abs (x + 1) + abs (x + 3) + abs (x + 6)) = 7 :=
by {
  sorry
}

end min_abs_sum_l166_166414


namespace range_of_f_l166_166994

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + 2 * sin x - 2

theorem range_of_f : ∀ x : ℝ, -4 ≤ f x ∧ f x ≤ 0 :=
by
  intros
  sorry

end range_of_f_l166_166994


namespace Joey_study_time_l166_166672

theorem Joey_study_time :
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96 := by
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  show (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96
  -- define study times
  let weekday_hours_per_week := weekday_hours_per_night * nights_per_week
  let weekend_hours_per_week := weekend_hours_per_day * days_per_weekend
  -- sum times per week
  let total_hours_per_week := weekday_hours_per_week + weekend_hours_per_week
  -- multiply by weeks until exam
  let total_study_time := total_hours_per_week * weeks_until_exam
  have h : total_study_time = 96 := by sorry
  exact h

end Joey_study_time_l166_166672


namespace prob_all_green_is_1_div_30_l166_166289

-- Definitions as per the conditions
def total_apples : ℕ := 10
def red_apples : ℕ := 6
def green_apples  : ℕ := 4

def choose (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.descFactorial n k / (Nat.factorial k) else 0

-- Question part: probability calculation
def prob_all_green (n total green k : ℕ) (h_total : total = 10) (h_green : green = 4) (h_k : k = 3) : ℚ :=
  (choose green k) / (choose total k)

-- Statement to be proved
theorem prob_all_green_is_1_div_30 : (prob_all_green 3 total_apples green_apples 3 rfl rfl rfl) = 1/30 :=
  by sorry

end prob_all_green_is_1_div_30_l166_166289


namespace compare_powers_l166_166629

def n1 := 22^44
def n2 := 33^33
def n3 := 44^22

theorem compare_powers : n1 > n2 ∧ n2 > n3 := by
  sorry

end compare_powers_l166_166629


namespace solution_set_suff_not_necessary_l166_166365

theorem solution_set_suff_not_necessary (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * a * x + a > 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end solution_set_suff_not_necessary_l166_166365


namespace min_value_of_function_l166_166592

open Real

theorem min_value_of_function (x : ℝ) (h : x > 2) : (∃ a : ℝ, (∀ y : ℝ, y = (4 / (x - 2) + x) → y ≥ a) ∧ a = 6) :=
sorry

end min_value_of_function_l166_166592


namespace sandy_will_be_32_l166_166381

  -- Define the conditions
  def world_record_length : ℝ := 26
  def sandy_current_length : ℝ := 2
  def monthly_growth_rate : ℝ := 0.1
  def sandy_current_age : ℝ := 12

  -- Define the annual growth rate calculation
  def annual_growth_rate : ℝ := monthly_growth_rate * 12

  -- Define total growth needed
  def total_growth_needed : ℝ := world_record_length - sandy_current_length

  -- Define the years needed to grow the fingernails to match the world record
  def years_needed : ℝ := total_growth_needed / annual_growth_rate

  -- Define Sandy's age when she achieves the world record length
  def sandy_age_when_record_achieved : ℝ := sandy_current_age + years_needed

  -- The statement we want to prove
  theorem sandy_will_be_32 :
    sandy_age_when_record_achieved = 32 :=
  by
    -- Placeholder proof
    sorry
  
end sandy_will_be_32_l166_166381


namespace value_of_expression_l166_166642

theorem value_of_expression (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / |x| + |y| / y = 2) ∨ (x / |x| + |y| / y = 0) ∨ (x / |x| + |y| / y = -2) :=
by
  sorry

end value_of_expression_l166_166642


namespace min_abs_sum_is_two_l166_166427

theorem min_abs_sum_is_two : ∃ x ∈ set.Ioo (- ∞) (∞), ∀ y ∈ set.Ioo (- ∞) (∞), (|y + 1| + |y + 3| + |y + 6| ≥ 2) ∧ (|x + 1| + |x + 3| + |x + 6| = 2) := sorry

end min_abs_sum_is_two_l166_166427


namespace triangle_side_length_l166_166204

theorem triangle_side_length (a b c : ℝ) (h1 : a + b + c = 20)
  (h2 : (1 / 2) * b * c * (Real.sin (Real.pi / 3)) = 10 * Real.sqrt 3) : a = 7 :=
sorry

end triangle_side_length_l166_166204


namespace hyperbola_right_focus_l166_166347

theorem hyperbola_right_focus (x y : ℝ) (h : x^2 - 2 * y^2 = 1) : x = sqrt 6 / 2 ∧ y = 0 := sorry

end hyperbola_right_focus_l166_166347


namespace quadratic_expression_value_l166_166254

theorem quadratic_expression_value :
  (x1 x2 : ℝ) (h1 : x1^2 - 4*x1 - 5 = 0) (h2 : x2^2 - 4*x2 - 5 = 0) (h3 : x1 ≠ x2) :
  (x1 - 1) * (x2 - 1) = -8 :=
by
  sorry

end quadratic_expression_value_l166_166254


namespace line_through_origin_line_perpendicular_solve_lines_l166_166203

def intersection_point (l1 l2 : ℝ × ℝ × ℝ) : ℝ × ℝ :=
  let (a1, b1, c1) := l1
  let (a2, b2, c2) := l2
  let det := a1 * b2 - a2 * b1
  if det ≠ 0 then
    ((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det)
  else
    (0, 0)  -- Just a placeholder for non-intersecting case

theorem line_through_origin (M : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := M
  if x₀ ≠ 0 then y₀ / x₀ = -2 else y₀ = 0

theorem line_perpendicular (M : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := M
  x₀ - 2*y₀ + 5 = 0

theorem solve_lines :
  let l1 := (3, 4, -5)
  let l2 := (2, -3, 8)
  let M := intersection_point l1 l2
  M = (-1, 2) →
  line_through_origin M ∧ line_perpendicular M :=
by
  intro h1
  unfold line_through_origin
  unfold line_perpendicular
  sorry

end line_through_origin_line_perpendicular_solve_lines_l166_166203


namespace sum_of_first_15_odd_integers_l166_166829

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l166_166829


namespace cos_75_degree_identity_l166_166556

theorem cos_75_degree_identity :
  Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by
  sorry

end cos_75_degree_identity_l166_166556


namespace keychain_arrangement_l166_166658

open Function

theorem keychain_arrangement (keys : Finset ℕ) (h : keys.card = 7)
  (house_key car_key office_key : ℕ) (hmem : house_key ∈ keys)
  (cmem : car_key ∈ keys) (omem : office_key ∈ keys) : 
  ∃ n : ℕ, n = 72 :=
by
  sorry

end keychain_arrangement_l166_166658


namespace quadrilateral_congruent_l166_166013

-- Definitions of vertices A, B, C, D inscribed in a circle with perpendicular diagonals
variables (A B C D : ℂ) (circle : set ℂ) (inscribed_in_circle : ∀ X ∈ {A, B, C, D}, X ∈ circle)
         (perpendicular_diagonals : ∀ P Q R S, P = A ∧ Q = C ∧ R = B ∧ S = D → (P - Q) * complex.conj (R - S) + (R - Q) * complex.conj (P - S) = 0) 

-- Definitions for the orthocenters of the triangles
def orthocenter_ABD := A + B + D
def orthocenter_ACD := A + C + D
def orthocenter_BCD := B + C + D
def orthocenter_ABC := A + B + C

-- Variables for the orthocenters of triangles ABD, ACD, BCD, and ABC
variables (K = orthocenter_ABD A B D) (L = orthocenter_ACD A C D) 
          (M = orthocenter_BCD B C D) (Q = orthocenter_ABC A B C)

-- Theorem: Quadrilateral KLMQ is congruent to quadrilateral ABCD
theorem quadrilateral_congruent : (K - L = C - B) ∧ (M - L = B - A) ∧ (Q - M = A - D) ∧ (K - Q = D - C) → congruent_quadrilaterals K L M Q A B C D :=
by sorry

end quadrilateral_congruent_l166_166013


namespace total_distance_east_l166_166538

-- Conditions
def hike_rate := 10 -- in minutes per kilometer
def initial_distance_east := 2.5 -- in kilometers
def remaining_time := 35 -- in minutes
def time_spent_initial := initial_distance_east * hike_rate -- in minutes
def time_left := remaining_time - time_spent_initial -- in minutes
def additional_distance_east := time_left / hike_rate -- in kilometers

-- Proof that total distance hiked east is 3.5 kilometers
theorem total_distance_east :
  initial_distance_east + additional_distance_east = 3.5 := by
  sorry

end total_distance_east_l166_166538


namespace find_k_l166_166140

-- Define the points A, B, X, Y with their coordinates
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := -4, y := 0}
def B : Point := {x := 0, y := -2}
def X : Point := {x := 0, y := 8}
def Y (k : ℝ) : Point := {x := 18, y := k}

-- Define the slope of a line segment given two points
def slope (P1 P2 : Point) : ℝ :=
  (P2.y - P1.y) / (P2.x - P1.x)

-- Define the condition that the line segments AB and XY are parallel
def segments_parallel (P1 P2 P3 P4 : Point) : Prop :=
  slope P1 P2 = slope P3 P4

-- Define the proof statement that we need to show
theorem find_k : ∃ k : ℝ, segments_parallel A B X (Y k) ∧ k = -1 := by
  sorry

end find_k_l166_166140


namespace triangle_ineq_l166_166288

variable (A B C O : ℝ)
variable (α β γ : ℝ)
variable (p : ℝ)

def is_perimeter (p : ℝ) (A B C : ℝ) : Prop := A + B + C = p

def ineq (A B C O α β γ p : ℝ) : Prop := 
  (OA * Real.sin γ + OB * Real.sin α + OC * Real.sin β <= p)

theorem triangle_ineq (A B C O α β γ p : ℝ)
  (H1 : is_perimeter p A B C) :
  ineq A B C O α β γ p :=
by
  sorry

end triangle_ineq_l166_166288


namespace no_solutions_l166_166966

theorem no_solutions (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hne : a + b ≠ 0) :
  ¬ (1 / a + 2 / b = 3 / (a + b)) :=
by { sorry }

end no_solutions_l166_166966


namespace bracelets_count_l166_166106

-- Define the conditions
def stones_total : Nat := 36
def stones_per_bracelet : Nat := 12

-- Define the theorem statement
theorem bracelets_count : stones_total / stones_per_bracelet = 3 := by
  sorry

end bracelets_count_l166_166106


namespace sum_first_eight_terms_l166_166303

variable {α : Type*} [LinearOrderedField α]

-- Definitions of a_n being an arithmetic sequence, a_1 = 1, and a_1, a_3, a_6 forming a geometric sequence.
def arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → α) : Prop :=
∀ i j k, 2 * j = i + k → (a j) ^ 2 = (a i) * (a k)

theorem sum_first_eight_terms {a : ℕ → α} {d : α}
  (h_arith : arithmetic_sequence a d)
  (h_nonzero : d ≠ 0)
  (h_a1 : a 1 = 1)
  (h_geom : geometric_sequence a)
  (h_geom_spec : h_geom 1 3 6) :
  (8 / 2) * (2 * (a 1) + (8 - 1) * d) = 15 := by
  sorry

end sum_first_eight_terms_l166_166303


namespace jaymee_older_than_twice_shara_l166_166670

-- Given conditions
def shara_age : ℕ := 10
def jaymee_age : ℕ := 22

-- Theorem to prove how many years older Jaymee is than twice Shara's age
theorem jaymee_older_than_twice_shara : jaymee_age - 2 * shara_age = 2 := by
  sorry

end jaymee_older_than_twice_shara_l166_166670


namespace coeff_x2_in_polynomial_l166_166306

theorem coeff_x2_in_polynomial
  (m n : ℕ)
  (h_pos_m : m > 0)
  (h_pos_n : n > 0)
  (h_coeff_x : 2 * m + 5 * n = 16) :
  (nat.choose m 2 * (-2)^2 + nat.choose n 2 * (-5)^2) = 37 :=
sorry

end coeff_x2_in_polynomial_l166_166306


namespace distance_is_correct_l166_166952

def vector2 := (Real × Real)

def point1 : vector2 := (3, -2)
def point2 : vector2 := (4, -1)
def direction : vector2 := (2, -5)

def distance_between_parallel_lines : Real := 
  let a := point1
  let b := point2
  let d := direction
  let v := (a.1 - b.1, a.2 - b.2)
  let dot_prod (u v : vector2) := u.1 * v.1 + u.2 * v.2
  let projection (u v : vector2) := 
    let factor := (dot_prod u v) / (dot_prod v v)
    (factor * v.1, factor * v.2)
  let c := (b.1 + (projection v d).1, b.2 + (projection v d).2)
  let diff := (a.1 - c.1, a.2 - c.2)
  let magnitude (u : vector2) := Real.sqrt ((u.1)^2 + (u.2)^2)
  magnitude diff

theorem distance_is_correct : distance_between_parallel_lines = 5 * Real.sqrt 29 / 29 := 
by 
  sorry

end distance_is_correct_l166_166952


namespace sum_of_solutions_sum_of_solutions_is_16_l166_166996

theorem sum_of_solutions (x : ℝ) (hx : (x - 8) ^ 2 = 36) : x = 14 ∨ x = 2 :=
by
  admit

theorem sum_of_solutions_is_16 : (∃ x1 x2 : ℝ, (x1 - 8) ^ 2 = 36 ∧ (x2 - 8) ^ 2 = 36 ∧ x1 + x2 = 16) :=
by
  use [14, 2]
  split
  { exact sum_of_solutions 14 (by norm_num) } -- proof for x1 = 14
  split
  { exact sum_of_solutions 2 (by norm_num) } -- proof for x2 = 2
  { norm_num } -- proof that 14 + 2 = 16
  sorry

end sum_of_solutions_sum_of_solutions_is_16_l166_166996


namespace sum_computation_l166_166954

noncomputable def sum_expr (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1), k * (⌈Real.log k / Real.log (sqrt 3)⌉ - ⌊Real.log k / Real.log (sqrt 3)⌋ : ℚ)

theorem sum_computation :
  sum_expr 500 = 125129 :=
sorry

end sum_computation_l166_166954


namespace cost_of_paint_per_quart_l166_166783

/-- Tommy has a flag that is 5 feet wide and 4 feet tall. 
He needs to paint both sides of the flag. 
A quart of paint covers 4 square feet. 
He spends $20 on paint. 
Prove that the cost of paint per quart is $2. --/
theorem cost_of_paint_per_quart
  (width height : ℕ) (paint_area_per_quart : ℕ) (total_cost : ℕ) (total_area : ℕ) (quarts_needed : ℕ) :
  width = 5 →
  height = 4 →
  paint_area_per_quart = 4 →
  total_cost = 20 →
  total_area = 2 * (width * height) →
  quarts_needed = total_area / paint_area_per_quart →
  total_cost / quarts_needed = 2 := 
by
  intros h_w h_h h_papq h_tc h_ta h_qn
  sorry

end cost_of_paint_per_quart_l166_166783


namespace money_inequality_l166_166338

-- Definitions and conditions
variables (a b : ℝ)
axiom cond1 : 6 * a + b > 78
axiom cond2 : 4 * a - b = 42

-- Theorem that encapsulates the problem and required proof
theorem money_inequality (a b : ℝ) (h1: 6 * a + b > 78) (h2: 4 * a - b = 42) : a > 12 ∧ b > 6 :=
  sorry

end money_inequality_l166_166338


namespace measure_of_B_range_of_sinA_plus_sinC_l166_166257

-- Define the angles and sides of triangle
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
axiom triangle_condition : 2 * b * cos B = a * cos C + c * cos A
axiom angle_bounds : B ∈ (0, π) ∧ A ∈ (0, (2 * π) / 3)

-- Target statements
theorem measure_of_B : B = π / 3 := by sorry

theorem range_of_sinA_plus_sinC :
    sqrt 3 * sin (A + π / 6) ∈ (sqrt 3 / 2, sqrt 3] := by sorry

end measure_of_B_range_of_sinA_plus_sinC_l166_166257


namespace least_3_digit_7_shifty_l166_166527

def is_7_shifty (n : ℕ) : Prop :=
  n % 7 > 2

theorem least_3_digit_7_shifty : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ is_7_shifty n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ is_7_shifty m → n ≤ m :=
begin
  sorry
end

end least_3_digit_7_shifty_l166_166527


namespace unique_parallel_line_l166_166200
open Set

-- Definitions for conditions in a)
variable (Plane : Type) [Nonempty Plane]
variable (Point : Type) [Nonempty Point]
variable (Line : Type) [Nonempty Line]

-- Definition for parallel planes
def parallel_planes (α β : Plane) : Prop := ∀ l : Line, l ⊆ α → ∃ m : Line, m ⊆ β ∧ l ∥ m

-- Definition for a line contained in a plane
def line_in_plane (l : Line) (α : Plane) : Prop := l ⊆ α

-- Definition for point lying in a plane
def point_in_plane (B : Point) (β : Plane) : Prop := B ∈ β

-- Theorem statement
theorem unique_parallel_line 
  (α β : Plane) 
  (a : Line) 
  (B : Point)
  (hp1 : parallel_planes α β) 
  (hp2 : line_in_plane a α) 
  (hp3 : point_in_plane B β) : 
  ∃! b : Line, b ⊆ β ∧ B ∈ b ∧ a ∥ b :=
sorry

end unique_parallel_line_l166_166200


namespace find_original_number_l166_166468

noncomputable def original_number (x : ℝ) : Prop :=
  (0 < x) ∧ (1000 * x = 3 / x)

theorem find_original_number : ∃ x : ℝ, original_number x ∧ x = real.sqrt 30 / 100 :=
by
  sorry

end find_original_number_l166_166468


namespace arithmetic_geometric_sequence_l166_166628

theorem arithmetic_geometric_sequence {a b c x y : ℝ} (h₁: a ≠ b) (h₂: b ≠ c) (h₃: a ≠ c)
  (h₄ : 2 * b = a + c) (h₅ : x^2 = a * b) (h₆ : y^2 = b * c) :
  (x^2 + y^2 = 2 * b^2) ∧ (x^2 * y^2 ≠ b^4) :=
by {
  sorry
}

end arithmetic_geometric_sequence_l166_166628


namespace sum_gcd_lcm_l166_166861

theorem sum_gcd_lcm (a b : ℕ) (h₀ : a = 50) (h₁ : b = 5005) :
  Nat.gcd a b + Nat.lcm a b = 50055 :=
by
  -- Variables a and b are instantiated to 50 and 5005 respectively.
  rw [h₀, h₁]
  -- Perform the gcd and lcm computation directly.
  have gcd_result : Nat.gcd 50 5005 = 5 := by sorry  -- gcd computation step
  have lcm_result : Nat.lcm 50 5005 = 50050 := by sorry  -- lcm computation step
  rw [gcd_result, lcm_result]
  -- Summing the gcd and lcm results.
  exact congrArg2 (· + ·) rfl rfl

end sum_gcd_lcm_l166_166861


namespace sum_first_15_odd_integers_l166_166824

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l166_166824


namespace total_students_approximation_l166_166241

theorem total_students_approximation (h : 0.725 * T = 638) : T ≈ 880 :=
by
  -- placeholder for the actual proof
  sorry

end total_students_approximation_l166_166241


namespace volume_of_given_cube_l166_166882

variable (p : ℝ) -- Given the perimeter of one face of the cube

def length_of_one_side (p : ℝ) : ℝ :=
  p / 4

def volume_of_cube (s : ℝ) : ℝ :=
  s ^ 3

theorem volume_of_given_cube (h : p = 24) : volume_of_cube (length_of_one_side p) = 216 :=
by
  rw [h, length_of_one_side, volume_of_cube]
  norm_num
  sorry

end volume_of_given_cube_l166_166882


namespace find_original_number_l166_166465

noncomputable def original_number (x : ℝ) : Prop :=
  (0 < x) ∧ (1000 * x = 3 / x)

theorem find_original_number : ∃ x : ℝ, original_number x ∧ x = real.sqrt 30 / 100 :=
by
  sorry

end find_original_number_l166_166465


namespace remainder_n_squared_plus_3n_plus_4_l166_166641

theorem remainder_n_squared_plus_3n_plus_4 (n : ℤ) (h : n % 100 = 99) : (n^2 + 3*n + 4) % 100 = 2 := 
by sorry

end remainder_n_squared_plus_3n_plus_4_l166_166641


namespace problem_statement_l166_166309

def T (m : ℕ) : ℕ := sorry
def H (m : ℕ) : ℕ := sorry

def p (m k : ℕ) : ℝ := 
  if k % 2 = 1 then 0 else sorry

theorem problem_statement (m : ℕ) : p m 0 ≥ p (m + 1) 0 := sorry

end problem_statement_l166_166309


namespace min_visible_pairs_l166_166180

-- Define the problem conditions
def bird_circle_flock (P : ℕ) : Prop :=
  P = 155

def mutual_visibility_condition (θ : ℝ) : Prop :=
  θ ≤ 10

-- Define the minimum number of mutually visible pairs
def min_mutual_visible_pairs (P_pairs : ℕ) : Prop :=
  P_pairs = 270

-- The main theorem statement
theorem min_visible_pairs (n : ℕ) (θ : ℝ) (P_pairs : ℕ)
  (H1 : bird_circle_flock n)
  (H2 : mutual_visibility_condition θ) :
  min_mutual_visible_pairs P_pairs :=
by
  sorry

end min_visible_pairs_l166_166180


namespace how_much_milk_did_joey_drink_l166_166568

theorem how_much_milk_did_joey_drink (initial_milk : ℚ) (rachel_fraction : ℚ) (joey_fraction : ℚ) :
  initial_milk = 3/4 →
  rachel_fraction = 1/3 →
  joey_fraction = 1/2 →
  joey_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/4 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end how_much_milk_did_joey_drink_l166_166568


namespace min_abs_sum_is_two_l166_166428

theorem min_abs_sum_is_two : ∃ x ∈ set.Ioo (- ∞) (∞), ∀ y ∈ set.Ioo (- ∞) (∞), (|y + 1| + |y + 3| + |y + 6| ≥ 2) ∧ (|x + 1| + |x + 3| + |x + 6| = 2) := sorry

end min_abs_sum_is_two_l166_166428


namespace min_abs_sum_l166_166404

theorem min_abs_sum (x : ℝ) : 
  (∃ x, (∀ y, ∥ y + 1 ∥ + ∥ y + 3 ∥ + ∥ y + 6 ∥ ≥ ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥) ∧ 
        ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥ = 5) := 
sorry

end min_abs_sum_l166_166404


namespace eight_segments_configurable_l166_166142

noncomputable def exists_eight_segments : Prop :=
∃ (segments : Fin 8 → Set (ℝ × ℝ)),
  (∀ i j : Fin 8, i ≠ j → ∃ x, x ∈ segments i ∩ segments j) ∧
  (∀ i : Fin 8, Set.card {j : Fin 8 | ∃ x, x ∈ segments i ∩ segments j} = 3) ∧
  (∀ x : ℝ × ℝ, ∃! (i : Fin 8), x ∈ segments i) ∧
  (∀ x : ℝ × ℝ, ∃ at_most_two_segments : Finset Fin 8, x ∈ ⋂₀ {segments i | i ∈ at_most_two_segments} ∧ at_most_two_segments.card ≤ 2)

theorem eight_segments_configurable : exists_eight_segments :=
sorry

end eight_segments_configurable_l166_166142


namespace molar_mass_calculation_l166_166393

theorem molar_mass_calculation
  (a b : ℝ) (m n x : ℝ) (c : ℝ) :
  let mixed_solution_molarity := c in
  ∀ (A B C : Prop),
  (A → (false)) ∧
  (B → (false)) ∧
  (C → (false)) → 
  ¬(A ∨ B ∨ C) :=
by 
  intros mixed_solution_molarity A B C h;
  cases h with hA hBC;
  cases hBC with hB hC;
  tauto;
  sorry

end molar_mass_calculation_l166_166393


namespace original_number_value_l166_166449

noncomputable def orig_number_condition (x : ℝ) : Prop :=
  1000 * x = 3 * (1 / x)

theorem original_number_value : ∃ x : ℝ, 0 < x ∧ orig_number_condition x ∧ x = √(30) / 100 :=
begin
  -- the proof
  sorry
end

end original_number_value_l166_166449


namespace original_number_solution_l166_166446

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end original_number_solution_l166_166446


namespace monochromatic_triangle_probability_l166_166913

noncomputable def probability_of_monochromatic_triangle_in_hexagon : ℝ := 0.968324

theorem monochromatic_triangle_probability :
  ∃ (H : Hexagon), probability_of_monochromatic_triangle_in_hexagon = 0.968324 :=
sorry

end monochromatic_triangle_probability_l166_166913


namespace seeds_in_bucket_A_l166_166777

theorem seeds_in_bucket_A (A B C : ℕ) (h_total : A + B + C = 100) (h_B : B = 30) (h_C : C = 30) : A = 40 :=
by
  sorry

end seeds_in_bucket_A_l166_166777


namespace total_wheels_at_park_l166_166928

-- Conditions as definitions
def number_of_adults := 6
def number_of_children := 15
def wheels_per_bicycle := 2
def wheels_per_tricycle := 3

-- To prove: total number of wheels = 57
theorem total_wheels_at_park : 
  (number_of_adults * wheels_per_bicycle) + (number_of_children * wheels_per_tricycle) = 57 :=
by
  sorry

end total_wheels_at_park_l166_166928


namespace min_value_of_f_l166_166407

def f (x : ℝ) := abs (x + 1) + abs (x + 3) + abs (x + 6)

theorem min_value_of_f : ∃ (x : ℝ), f x = 5 :=
by
  use -3
  simp [f]
  sorry

end min_value_of_f_l166_166407


namespace complex_number_is_3i_quadratic_equation_roots_l166_166593

open Complex

-- Given complex number z satisfies 2z + |z| = 3 + 6i
-- We need to prove that z = 3i
theorem complex_number_is_3i (z : ℂ) (h : 2 * z + abs z = 3 + 6 * I) : z = 3 * I :=
sorry

-- Given that z = 3i is a root of the quadratic equation with real coefficients
-- Prove that b - c = -9
theorem quadratic_equation_roots (b c : ℝ) (h1 : 3 * I + -3 * I = -b)
  (h2 : 3 * I * -3 * I = c) : b - c = -9 :=
sorry

end complex_number_is_3i_quadratic_equation_roots_l166_166593


namespace g_difference_l166_166305

def g (n : ℕ) : ℚ := (1 / 4) * n * (n + 3) * (n + 5) + 2

theorem g_difference (s : ℕ) : g s - g (s - 1) = (3 * s^2 + 9 * s + 8) / 4 :=
by
  -- skip the proof
  sorry

end g_difference_l166_166305


namespace big_bottles_in_storage_l166_166523

theorem big_bottles_in_storage (B : ℕ) :
  let small_bottles := 5000 in
  let small_remaining := 0.85 * small_bottles in
  let big_remaining := 0.82 * B in
  small_remaining + big_remaining = 14090 → B = 12000 :=
by
  intros small_bottles small_remaining big_remaining h
  sorry

end big_bottles_in_storage_l166_166523


namespace exists_adjacent_cells_with_large_diff_l166_166712

noncomputable def exists_adjacent_cells_with_diff_at_least_n (n : ℕ) (grid : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j : ℕ, 1 ≤ grid i j ∧ grid i j ≤ n * n) ∧
  (∃ i j : ℕ, i < n ∧ j < n ∧
  ((j + 1 < n ∧ (grid i j - grid i (j + 1)).nat_abs ≥ n) ∨
   (i + 1 < n ∧ (grid i j - grid (i + 1) j).nat_abs ≥ n)))

theorem exists_adjacent_cells_with_large_diff (n : ℕ) (grid : ℕ → ℕ → ℕ)
  (h : ∀ i j, grid i j ∈ finset.range (n * n + 1)) :
  exists_adjacent_cells_with_diff_at_least_n n grid :=
sorry

end exists_adjacent_cells_with_large_diff_l166_166712


namespace min_abs_sum_l166_166406

theorem min_abs_sum (x : ℝ) : 
  (∃ x, (∀ y, ∥ y + 1 ∥ + ∥ y + 3 ∥ + ∥ y + 6 ∥ ≥ ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥) ∧ 
        ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥ = 5) := 
sorry

end min_abs_sum_l166_166406


namespace sum_first_15_odd_integers_l166_166809

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l166_166809


namespace g_is_even_l166_166664

def g (x : ℝ) : ℝ := 2^(x^2 - 4) - 2 * |x|

theorem g_is_even : ∀ x : ℝ, g (-x) = g x :=
by
  intro x
  unfold g
  have h1: (-x)^2 = x^2 := by norm_num
  have h2: | -x | = | x | := by exact abs_neg x
  rw [h1, h2]
  sorry

end g_is_even_l166_166664


namespace decagon_diagonal_intersection_probability_l166_166955

def probability_intersect_within_decagon : ℚ :=
  let total_vertices := 10
  let total_pairs_points := Nat.choose total_vertices 2
  let total_diagonals := total_pairs_points - total_vertices
  let ways_to_pick_2_diagonals := Nat.choose total_diagonals 2
  let combinations_4_vertices := Nat.choose total_vertices 4
  (combinations_4_vertices : ℚ) / (ways_to_pick_2_diagonals : ℚ)

theorem decagon_diagonal_intersection_probability :
  probability_intersect_within_decagon = 42 / 119 :=
sorry

end decagon_diagonal_intersection_probability_l166_166955


namespace probability_of_same_color_shoes_l166_166677

open Finset

-- Conditions
def num_pairs : ℕ := 9
def num_shoes_per_pair : ℕ := 2
def total_shoes : ℕ := num_pairs * num_shoes_per_pair

-- Number of ways Kim can select 2 shoes out of 18
def total_combinations : ℕ := choose total_shoes 2

-- Number of favorable outcomes (selecting 2 shoes of the same color)
def favorable_outcomes : ℕ := num_pairs

-- Probability of selecting 2 shoes of the same color
def probability : ℚ := favorable_outcomes / total_combinations

theorem probability_of_same_color_shoes : 
  probability = 1 / 17 := 
by 
  sorry

end probability_of_same_color_shoes_l166_166677


namespace train_passes_platform_in_40_8_seconds_l166_166488

def length_of_train : ℝ := 360
def speed_of_train_kmh : ℝ := 45
def length_of_platform : ℝ := 150
def speed_of_train_ms : ℝ := speed_of_train_kmh * 1000 / 3600
def total_distance_to_cover : ℝ := length_of_train + length_of_platform
def time_to_pass_platform := total_distance_to_cover / speed_of_train_ms

theorem train_passes_platform_in_40_8_seconds :
  time_to_pass_platform = 40.8 := by
    -- Assuming the conditions provided, we would derive the conclusion here.
    -- Proof steps would be filled in here.
    sorry

end train_passes_platform_in_40_8_seconds_l166_166488


namespace initial_volume_mixture_l166_166516

theorem initial_volume_mixture (x : ℝ) :
  (4 * x) / (3 * x + 13) = 5 / 7 →
  13 * x = 65 →
  7 * x = 35 := 
by
  intro h1 h2
  sorry

end initial_volume_mixture_l166_166516


namespace incenter_and_excenter_l166_166283

variable (A B C D E F : Type*) [geometry (A)]
variables [in_triangle (A B C : A_points)] [touches_circumcircle (l A : line)]

noncomputable def passes_through_incenter (D E A B C : A_points) : Prop := sorry
noncomputable def passes_through_excenter (D F A B C : A_points) : Prop := sorry

theorem incenter_and_excenter (A B C D E F : A_points)
  (h1 : AB > AC) 
  (h2 : tangent_line l touches_circumcircle_of_triangle ABC at A)
  (h3 : centered_circle_of_radius_AC intersects_segment_AB at D)
  (h4 : centered_circle_of_radius_AC intersects_line_l at (E F)) :
  passes_through_incenter D E A B C ∧ passes_through_excenter D F A B C :=
sorry

end incenter_and_excenter_l166_166283


namespace ordered_pair_count_l166_166613

noncomputable def count_pairs (n : ℕ) (c : ℕ) : ℕ := 
  (if n < c then 0 else n - c + 1)

theorem ordered_pair_count :
  (count_pairs 39 5 = 35) :=
sorry

end ordered_pair_count_l166_166613


namespace exceeding_fraction_l166_166933

def repeatingDecimal (n : ℚ) (d : ℕ) : ℚ := n / (10^d - 1) -- Function for repeating decimal form

def decimal (n : ℚ) (d : ℕ) : ℚ := n / (10^d) -- Function for non-repeating decimal form

theorem exceeding_fraction : 
  let x := repeatingDecimal 6 2 in
  let y := decimal 6 2 in
  x - y = 2 / 3300 :=
by {
  have hx : x = repeatingDecimal 6 2 := rfl,
  have hy : y = decimal 6 2 := rfl,
  conversion,
  rw [hx, hy],
  rw [repeatingDecimal, decimal],
  have h_repeating : repeatingDecimal 6 2 = (6 / 99),  -- Known value for repeating decimal
  { rw ← div_div, norm_num },
  have h_decimal : decimal 6 2 = (6 / 100),  -- Known value for decimal
  { rw ← div_div, norm_num },
  calc
    (6 / 99) - (6 / 100)
    = ((6 * 100) - (6 * 99)) / (99 * 100) : by {
        rw [sub_div, mul_comm (10^2 - 1), mul_comm 2 5, pow_add, pow_one, mul_comm 10 99],
        norm_num
    }
    ... = ((600 - 594) / 9900)         : by norm_num
    ... = (6 / 9900)                  : by norm_num
    ... = (2 / 3300)                  : by norm_num
}

end exceeding_fraction_l166_166933


namespace sum_of_common_divisors_36_and_24_l166_166995

open Nat

theorem sum_of_common_divisors_36_and_24 : 
  (∑ d in ({d | d ∣ 36} ∩ {d | d ∣ 24}), d) = 28 := 
by
  sorry

end sum_of_common_divisors_36_and_24_l166_166995


namespace LiZhi_is_SJTU_and_city_volunteer_l166_166973

def LiZhiUniversity : Prop := "Li Zhi is from Shanghai Jiao Tong University"
def LiZhiRole : Prop := "Li Zhi is a city volunteer"
def Cond1 : Prop := "Li Zhi is not from Tongji University"
def Cond2 : Prop := "Wen Wen is not from Shanghai Jiao Tong University"
def Cond3 : Prop := "The student from Tongji University is not a translator volunteer"
def Cond4 : Prop := "The student from Shanghai Jiao Tong University is a city volunteer"
def Cond5 : Prop := "Wen Wen is not a social volunteer"

theorem LiZhi_is_SJTU_and_city_volunteer (h1 : Cond1) (h2 : Cond2) (h3 : Cond3) (h4 : Cond4) (h5 : Cond5) : LiZhiUniversity ∧ LiZhiRole := by
  sorry

end LiZhi_is_SJTU_and_city_volunteer_l166_166973


namespace smallest_n_for_divisibility_condition_l166_166970

theorem smallest_n_for_divisibility_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) →
    (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (xyz ∣ (x + y + z)^n)) ∧
    n = 13 :=
by
  use 13
  sorry

end smallest_n_for_divisibility_condition_l166_166970


namespace journey_total_time_l166_166236

theorem journey_total_time (speed1 time1 speed2 total_distance : ℕ) 
  (h1 : speed1 = 40) 
  (h2 : time1 = 3) 
  (h3 : speed2 = 60) 
  (h4 : total_distance = 240) : 
  time1 + (total_distance - speed1 * time1) / speed2 = 5 := 
by 
  sorry

end journey_total_time_l166_166236


namespace trajectory_equation_y_range_l166_166270

section
open Real

def point_A : ℝ × ℝ := (-sqrt 2, 0)
def point_B : ℝ × ℝ := (sqrt 2, 0)
def point_F : ℝ × ℝ := (1, 0)
def trajectory_C (E : ℝ × ℝ) (x : ℝ) (y : ℝ) : Prop :=
    E = (x, y) ∧ 
    (E ≠ point_A ∧ E ≠ point_B) ∧
    (y / (x + sqrt 2) * y / (x - sqrt 2) = - 1 / 2)

-- Statement for the first part
theorem trajectory_equation (x y : ℝ) :
    (E : ℝ × ℝ), trajectory_C E x y →
    (x^2 / 2) + y^2 = 1 ∧ x ≠ sqrt 2 ∧ x ≠ -sqrt 2 :=
sorry

-- Additional conditions for the second part of the problem
def intersects_curve (l : ℝ → ℝ) (curve_C : ℝ × ℝ → Prop) (x1 x2 y1 y2 : ℝ) (F : ℝ × ℝ) : Prop :=
    l 1 = 0 ∧
    curve_C (x1, y1) ∧ curve_C (x2, y2) ∧
    (λ x y, (y = l (x - F.1))) (x1, y1) ∧ (λ x y, (y = l (x - F.1))) (x2, y2) ∧ x1 ≠ x2

def y_coordinate_range (P : ℝ × ℝ) (curve_C : ℝ × ℝ → Prop) (F : ℝ × ℝ) (l : ℝ → ℝ) (x1 x2 y1 y2 : ℝ) : Prop :=
    intersects_curve l curve_C x1 x2 y1 y2 F ∧
    P.1 = 0 ∧
    (complex.abs (P.2 - y1) = complex.abs (P.2 - y2))

-- Statement for the second part
theorem y_range (P : ℝ × ℝ) (curve_C : ℝ × ℝ → Prop) (F : ℝ × ℝ) (l : ℝ → ℝ) (x1 x2 y1 y2 : ℝ) 
    (h_intersect : intersects_curve l curve_C x1 x2 y1 y2 F)
    (h_point_on_axis : P.1 = 0)
    (h_abs_eq : complex.abs (P.2 - y1) = complex.abs (P.2 - y2)):
    P.2 ∈ Icc (- sqrt 2 / 4) (sqrt 2 / 4) :=
sorry

end

end trajectory_equation_y_range_l166_166270


namespace graveling_cost_is_969_l166_166521

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

end graveling_cost_is_969_l166_166521


namespace repeating_decimals_count_l166_166164

theorem repeating_decimals_count :
  (Finset.filter (λ n => ¬ (∃ k: ℕ, n = 3 * k)) (Finset.range 12)).card = 8 :=
by
  sorry

end repeating_decimals_count_l166_166164


namespace focus_coordinates_l166_166102

-- Define the points and the parameters used in the problem.
def point1 := (0, -2)
def point2 := (7, -2)
def point3 := (3.5, 1)
def point4 := (3.5, -5)
def center := ((point1.1 + point2.1) / 2, (point1.2 + point3.2) / 2)

-- The major axis length
def major_axis := 7
-- The minor axis length
def minor_axis := 6
-- The distance between the center and one of the foci
def focal_distance := Real.sqrt (major_axis^2 - minor_axis^2) / 2

-- The coordinates of the focus with the greater x-coordinate.
def focus_with_greater_x := (center.1 + focal_distance, center.2)

theorem focus_coordinates : 
  focus_with_greater_x = (3.5 + Real.sqrt 13 /2, -2) := by
  -- proof goes here
  sorry

end focus_coordinates_l166_166102


namespace pool_capacity_l166_166060

theorem pool_capacity:
  (∃ (V1 V2 : ℝ) (t : ℝ), 
    (V1 = t / 120) ∧ 
    (V2 = V1 + 50) ∧ 
    (V1 + V2 = t / 48) ∧ 
    t = 12000) := 
by 
  sorry

end pool_capacity_l166_166060


namespace minor_axis_length_is_2sqrt3_l166_166525

-- Define the points given in the problem
def points : List (ℝ × ℝ) := [(1, 1), (0, 0), (0, 3), (4, 0), (4, 3)]

-- Define a function that checks if an ellipse with axes parallel to the coordinate axes
-- passes through given points, and returns the length of its minor axis if it does.
noncomputable def minor_axis_length (pts : List (ℝ × ℝ)) : ℝ :=
  if h : (0,0) ∈ pts ∧ (0,3) ∈ pts ∧ (4,0) ∈ pts ∧ (4,3) ∈ pts ∧ (1,1) ∈ pts then
    let a := (4 - 0) / 2 -- half the width of the rectangle
    let b_sq := 3 -- derived from solving the ellipse equation
    2 * Real.sqrt b_sq
  else 0

-- The theorem statement:
theorem minor_axis_length_is_2sqrt3 : minor_axis_length points = 2 * Real.sqrt 3 := by
  sorry

end minor_axis_length_is_2sqrt3_l166_166525


namespace proof_problem_l166_166193

theorem proof_problem
  (a b : ℝ)
  (h1 : a = -(-3))
  (h2 : b = - (- (1 / 2))⁻¹)
  (m n : ℝ) :
  (|m - a| + |n + b| = 0) → (a = 3 ∧ b = -2 ∧ m = 3 ∧ n = -2) :=
by {
  sorry
}

end proof_problem_l166_166193


namespace minimum_keys_required_l166_166133

theorem minimum_keys_required (drivers cars : ℕ) (h_drivers : drivers = 50) (h_cars : cars = 40) :
  ∃ keys : ℕ, keys = 440 :=
by
  use 440
  sorry

end minimum_keys_required_l166_166133


namespace sum_of_square_areas_l166_166648

theorem sum_of_square_areas (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 :=
sorry

end sum_of_square_areas_l166_166648


namespace average_billboards_per_hour_correct_l166_166708

/-- Define the parameters for each counting period and the number of billboards seen. -/
def first_hour_counting_period : ℝ := 25 / 60
def first_hour_billboards_seen : ℕ := 15
def second_hour_counting_period : ℝ := 45 / 60
def second_hour_billboards_seen : ℕ := 31
def third_hour_lunch_period : ℝ := 35 / 60
def third_hour_lunch_billboards_seen : ℕ := 10
def third_hour_post_lunch_period : ℝ := 20 / 60
def third_hour_post_lunch_billboards_seen : ℕ := 12

/-- Calculate the total number of billboards seen. -/
def total_billboards_seen : ℕ := first_hour_billboards_seen + second_hour_billboards_seen + third_hour_lunch_billboards_seen + third_hour_post_lunch_billboards_seen

/-- Calculate the total counting time in hours. -/
def total_counting_time : ℝ := first_hour_counting_period + second_hour_counting_period + third_hour_lunch_period + third_hour_post_lunch_period

/-- The average number of electronic billboards seen per hour. -/
def average_billboards_per_hour : ℝ := total_billboards_seen / total_counting_time

theorem average_billboards_per_hour_correct :
  average_billboards_per_hour ≈ 32.64 :=
begin
  -- We skip the detailed proof steps here and leave a sorry statement instead.
  sorry
end

end average_billboards_per_hour_correct_l166_166708


namespace arithmetic_sequence_properties_l166_166271

open Real

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℝ) :=
  ∀ (n m : ℕ), b n * b (n + m) = b (n + 1) * b (n + m - 1)

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  (a 1)^2 + a 3 = 4 →
  a 5 + a 6 + a 7 = 18 →
  (∀ n : ℕ, a n = n) ∧
  (∀ m : ℕ, (∑ i in finset.range m, 1 / ((2 * (i + 1) + 2) * a (i + 1))) = m / (2 * m + 2)) :=
by
  intro h_arith_sequence h_condition1 h_condition2
  sorry

end arithmetic_sequence_properties_l166_166271


namespace units_digit_is_six_l166_166888

-- Define the recursive function for the original sequence
def sequence_simplified : ℕ := 
  let expr := List.foldr (*) 1 ([0,1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32].map (λ n, 2^n + 1)) + 1
  expr % 10

theorem units_digit_is_six : sequence_simplified = 6 :=
by 
  sorry

end units_digit_is_six_l166_166888


namespace Joey_study_time_l166_166673

theorem Joey_study_time :
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96 := by
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  show (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96
  -- define study times
  let weekday_hours_per_week := weekday_hours_per_night * nights_per_week
  let weekend_hours_per_week := weekend_hours_per_day * days_per_weekend
  -- sum times per week
  let total_hours_per_week := weekday_hours_per_week + weekend_hours_per_week
  -- multiply by weeks until exam
  let total_study_time := total_hours_per_week * weeks_until_exam
  have h : total_study_time = 96 := by sorry
  exact h

end Joey_study_time_l166_166673


namespace sick_day_probability_l166_166655

theorem sick_day_probability :
  (let sick_prob := 1 / 40
       not_sick_prob := 1 - sick_prob
       prob_one_sick :=
         sick_prob * not_sick_prob * not_sick_prob +
         not_sick_prob * sick_prob * not_sick_prob +
         not_sick_prob * not_sick_prob * sick_prob in
  prob_one_sick * 100 ≈ 7.1) :=
by
  sorry

end sick_day_probability_l166_166655


namespace concentration_after_removing_water_l166_166044

theorem concentration_after_removing_water :
  ∀ (initial_volume : ℝ) (initial_percentage : ℝ) (water_removed : ℝ),
  initial_volume = 18 →
  initial_percentage = 0.4 →
  water_removed = 6 →
  (initial_percentage * initial_volume) / (initial_volume - water_removed) * 100 = 60 :=
by
  intros initial_volume initial_percentage water_removed h1 h2 h3
  rw [h1, h2, h3]
  sorry

end concentration_after_removing_water_l166_166044


namespace simplified_sum_of_series_l166_166276

noncomputable def sum_of_series (x : ℝ) : ℝ :=
  (∑ k in Finset.range(2020) + 1, (x^k + x^(-k))^2)

theorem simplified_sum_of_series (x : ℝ) :
  sum_of_series x = 4040 + (x^8082 - 1) / (x^4040 * (x^2 - 1)) :=
by
  sorry

end simplified_sum_of_series_l166_166276


namespace book_price_l166_166667

theorem book_price (P : ℝ) : 
  (3 * 12 * P - 500 = 220) → 
  P = 20 :=
by
  intro h
  sorry

end book_price_l166_166667


namespace sum_of_first_n_odd_numbers_l166_166839

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l166_166839


namespace volume_of_region_l166_166864

theorem volume_of_region :
  let region := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + 2 * abs (p.3) ≤ 2 ∧ abs p.1 + abs p.2 + 2 * abs (p.3 - 1) ≤ 2}
  volume_of_region region = 4 / 3 := sorry

end volume_of_region_l166_166864


namespace b_n_expression_l166_166611

-- Define sequence a_n as an arithmetic sequence with given conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + d * (n - 1)

-- Define the conditions for the sequence a_n
def a_conditions (a : ℕ → ℤ) : Prop :=
  a 2 = 8 ∧ a 8 = 26

-- Define the new sequence b_n based on the terms of a_n
def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  a (3^n)

theorem b_n_expression (a : ℕ → ℤ) (n : ℕ)
  (h_arith : is_arithmetic_sequence a)
  (h_conditions : a_conditions a) :
  b a n = 3^(n + 1) + 2 := 
sorry

end b_n_expression_l166_166611


namespace quad_ABCD_BD_one_l166_166269

theorem quad_ABCD_BD_one
  (A B C D : Point)
  (AB BC : ℝ)
  (angle_B angle_D : ℝ)
  (h_AB : AB = 1)
  (h_BC : BC = 1)
  (h_angle_B : angle_B = 100)
  (h_angle_D : angle_D = 130) :
  distance B D = 1 :=
sorry

end quad_ABCD_BD_one_l166_166269


namespace count_pairs_l166_166585

theorem count_pairs : {
  let count := (finset.range 6).sum (λ b, (finset.range 6).count (λ c, b > 0 ∧ c > 0 ∧ (b^2 ≥ 4 * c ∨ c^2≥ 4 * b))),
  count = 13
} :=
begin
  sorry
end

end count_pairs_l166_166585


namespace probability_of_five_dice_all_same_l166_166490

theorem probability_of_five_dice_all_same : 
  (6 / (6 ^ 5) = 1 / 1296) :=
by
  sorry

end probability_of_five_dice_all_same_l166_166490


namespace triangle_BN_l166_166959

noncomputable def BN_length : ℝ :=
  let AB := 4
  let BC := 7
  let AC := 8
  let M := (AB / 2 : ℝ)
  let theta_cos := (4^2 + 8^2 - 7^2) / (2 * 4 * 8) 
  let AN := (M * AB) / AC
  let NC := AC - AN
  let BN_squared := AB^2 + AN^2 - 2 * AB * AN * theta_cos^2
  sqrt (BN_squared / 8)

theorem triangle_BN :
  BN_length = ( √(210) / 4) :=
by
  sorry

end triangle_BN_l166_166959


namespace unique_four_topping_pizzas_count_l166_166518

-- Defining the problem conditions and the proof goal
theorem unique_four_topping_pizzas_count : 
  (∃ n k : ℕ, n = 7 ∧ k = 4 ∧ nat.choose n k = 35) :=
by 
  sorry

end unique_four_topping_pizzas_count_l166_166518


namespace collinear_points_sum_l166_166141

theorem collinear_points_sum (x y : ℝ) : 
  (∃ a b : ℝ, a * x + b * 3 + (1 - a - b) * 2 = a * x + b * y + (1 - a - b) * y ∧ 
               a * y + b * 4 + (1 - a - b) * y = a * x + b * y + (1 - a - b) * x) → 
  x = 2 → y = 4 → x + y = 6 :=
by sorry

end collinear_points_sum_l166_166141


namespace find_k_slope_eq_l166_166906

theorem find_k_slope_eq :
  ∃ k: ℝ, (∃ k: ℝ, ((k - 4) / 7 = (-2 - k) / 14) → k = 2) :=
by
  sorry

end find_k_slope_eq_l166_166906


namespace min_abs_sum_l166_166416

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem min_abs_sum :
  ∃ (x : ℝ), (abs (x + 1) + abs (x + 3) + abs (x + 6)) = 7 :=
by {
  sorry
}

end min_abs_sum_l166_166416


namespace min_abs_sum_is_5_l166_166432

noncomputable def min_abs_sum (x : ℝ) : ℝ :=
  |x + 1| + |x + 3| + |x + 6|

theorem min_abs_sum_is_5 : ∃ x : ℝ, (∀ y : ℝ, min_abs_sum y ≥ min_abs_sum x) ∧ min_abs_sum x = 5 :=
by
  use -3
  sorry

end min_abs_sum_is_5_l166_166432


namespace sum_a_b_c_l166_166645

theorem sum_a_b_c (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 390) (h2: a * b + b * c + c * a = 5) : a + b + c = 20 ∨ a + b + c = -20 := 
by 
  sorry

end sum_a_b_c_l166_166645


namespace sum_of_fractions_irreducible_l166_166367

noncomputable def is_irreducible (num denom : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ num ∧ d ∣ denom → d = 1

theorem sum_of_fractions_irreducible (a b : ℕ) (h_coprime : Nat.gcd a b = 1) :
  is_irreducible (2 * a + b) (a * (a + b)) :=
by
  sorry

end sum_of_fractions_irreducible_l166_166367


namespace two_fours_express_64_l166_166062

theorem two_fours_express_64 : (real.sqrt ((real.sqrt (real.sqrt 4)) ^ (nat.factorial 4)) = 64) :=
sorry

end two_fours_express_64_l166_166062


namespace derivative_of_f_l166_166744

noncomputable def f (x : ℝ) : ℝ := (sin x) / x

theorem derivative_of_f : (deriv f x) = (x * cos x - sin x) / (x^2) :=
by 
  sorry

end derivative_of_f_l166_166744


namespace surface_area_of_circumscribed_sphere_l166_166017

noncomputable def surfaceAreaCircumscribedSphere 
  (OA OB OC : ℝ) 
  (h1 : OA = 2) 
  (h2 : OB = 2) 
  (h3 : OC = 1) 
  (h_perpendicular : OA * OB = 0 ∧ OA * OC = 0 ∧ OB * OC = 0) 
  : ℝ :=
  4 * Real.pi * ((Real.sqrt (OA ^ 2 + OB ^ 2 + OC ^ 2) / 2) ^ 2)

theorem surface_area_of_circumscribed_sphere : 
  surfaceAreaCircumscribedSphere 2 2 1 
  (by rfl) 
  (by rfl) 
  (by rfl) 
  (by { split, repeat {exact zero_mul _}}) 
  = 9 * Real.pi :=
sorry

end surface_area_of_circumscribed_sphere_l166_166017


namespace sum_equals_1584_l166_166549

-- Let's define the function that computes the sum, according to the pattern
def sumPattern : ℕ → ℝ
  | 0 => 0
  | k + 1 => if (k + 1) % 3 = 0 then - (k + 1) + sumPattern k
             else (k + 1) + sumPattern k

-- This function defines the problem setting and the final expected result
theorem sum_equals_1584 : sumPattern 99 = 1584 := by
  sorry

end sum_equals_1584_l166_166549


namespace compound_interest_time_period_l166_166147

variable (P : ℝ := 4000)
variable (r : ℝ := 0.15)
variable (CI : ℝ := 1554.5)
variable (n : ℝ := 1)
variable (A : ℝ := P + CI)

noncomputable def time_period : ℝ :=
  log (A / P) / log (1 + r / n)

theorem compound_interest_time_period : time_period P r CI n ≈ 2 := by
  sorry

end compound_interest_time_period_l166_166147


namespace find_some_number_l166_166584

noncomputable def floor (x : ℝ) : ℤ := int.floor x

theorem find_some_number : 
  let z := floor 6.5 * floor (2 / 3) + floor (2 : ℝ) * 7.2 + floor 8.4 - 9.8 in
  z = 12.599999999999998 → floor (2 : ℝ) = 2 :=
by
  intro h
  have h1 : floor 6.5 = 6 := by sorry
  have h2 : floor (2 / 3) = 0 := by sorry
  have h3 : floor 8.4 = 8 := by sorry
  have h4 : z = 6 * 0 + floor (2 : ℝ) * 7.2 + 8 - 9.8 := by sorry
  have h5 : z = floor (2 : ℝ) * 7.2 - 1.8 := by sorry
  have h6 : floor (2 : ℝ) * 7.2 = 14.4 := by sorry
  have h7 : 14.4 / 7.2 = 2 := by sorry
  show floor (2 : ℝ) = 2 from h7

end find_some_number_l166_166584


namespace min_abs_sum_l166_166401

theorem min_abs_sum (x : ℝ) : 
  (∃ x, (∀ y, ∥ y + 1 ∥ + ∥ y + 3 ∥ + ∥ y + 6 ∥ ≥ ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥) ∧ 
        ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥ = 5) := 
sorry

end min_abs_sum_l166_166401


namespace find_other_root_l166_166710

theorem find_other_root (k r : ℝ) (h1 : ∀ x : ℝ, 3 * x^2 + k * x + 6 = 0) (h2 : ∃ x : ℝ, 3 * x^2 + k * x + 6 = 0 ∧ x = 3) :
  r = 2 / 3 :=
sorry

end find_other_root_l166_166710


namespace original_number_solution_l166_166441

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end original_number_solution_l166_166441


namespace company_asset_percent_conditions_l166_166553

variables (A B C P : ℝ)

-- Conditions
def condition1 := P = A + 0.2 * A
def condition2 := P = B + B
def condition3 := C = 1.5 * P

-- Goal
def goal := (P / (A + B + C)) * 100 = 600 / 17

-- Theorem stating the goal given the conditions
theorem company_asset_percent_conditions : 
  condition1 → condition2 → condition3 → goal :=
by
  sorry

end company_asset_percent_conditions_l166_166553


namespace evaluate_expression_l166_166999

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := by 
  sorry

end evaluate_expression_l166_166999


namespace not_coexistent_pair_1_2_coexistent_pair_neg_value_of_negative_five_l166_166704

def coexistent_rational_pair (a b : ℚ) : Prop := a - b = a * b + 1

theorem not_coexistent_pair_1_2 : ¬ coexistent_rational_pair 1 2 := 
by 
  sorry 

theorem coexistent_pair_neg (m n : ℚ) (h : coexistent_rational_pair m n): 
  coexistent_rational_pair (-n) (-m) := 
by
  sorry

theorem value_of_negative_five (m n : ℚ) (h : coexistent_rational_pair m n) (h' : m - n = 4): 
  (-5)^(m * n) = -125 := 
by
  sorry

end not_coexistent_pair_1_2_coexistent_pair_neg_value_of_negative_five_l166_166704


namespace find_f_10_l166_166353

variable {f : ℝ → ℝ}

-- Define conditions
def is_odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x

-- Given conditions
axiom odd_f_minus_one : is_odd (λ x, f (x - 1))
axiom even_f_plus_one : is_even (λ x, f (x + 1))
axiom f_for_small_x : ∀ x, 0 ≤ x ∧ x < 1 → f x = 2 * x

-- Proof problem statement
theorem find_f_10 : f 10 = 1 :=
by sorry

end find_f_10_l166_166353


namespace time_difference_l166_166494

theorem time_difference (linda_speed tom_speed : ℝ) (h_linda_speed : linda_speed = 2) (h_tom_speed : tom_speed = 7) :
  let time_to_cover_half := (1 / 7 : ℝ) * 60,
      time_to_cover_twice := (4 / 7 : ℝ) * 60
  in abs (time_to_cover_twice - time_to_cover_half) ≈ 25.72 :=
by
  sorry

end time_difference_l166_166494


namespace find_number_l166_166865

theorem find_number (x : ℝ) (h : 4 * (x - 220) = 320) : (5 * x) / 3 = 500 :=
by
  sorry

end find_number_l166_166865


namespace find_original_number_l166_166472

theorem find_original_number (x : ℝ) (hx : 0 < x) 
  (h : 1000 * x = 3 * (1 / x)) : x ≈ 0.01732 :=
by 
  sorry

end find_original_number_l166_166472


namespace smallest_m_l166_166796

theorem smallest_m (m : ℕ) (p q : ℤ) (h_eq : 10*(p:ℤ)^2 - m*(p:ℤ) + 360 = 0) (h_cond : q = 2 * p) :
  p * q = 36 → 3 * p + 3 * q = m → m = 90 :=
by sorry

end smallest_m_l166_166796


namespace nests_count_l166_166383

theorem nests_count :
  ∃ (N : ℕ), (6 = N + 3) ∧ (N = 3) :=
by
  sorry

end nests_count_l166_166383


namespace sum_first_15_odd_integers_l166_166815

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l166_166815


namespace no_solution_geometric_sequence_cos_l166_166982

-- Define the problem statement
def no_geometric_sequence_cos (a : ℝ) : Prop := 
  ∃ r : ℝ, cos a = r * (cos 2 * a) ∧ cos 2 * a = r * (cos 3 * a)

theorem no_solution_geometric_sequence_cos : 
  ¬ (∃ a : ℝ, 0 < a ∧ a < 360 ∧ no_geometric_sequence_cos a) :=
by sorry

end no_solution_geometric_sequence_cos_l166_166982


namespace sum_of_first_n_odd_numbers_l166_166838

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l166_166838


namespace non_congruent_triangles_with_perimeter_10_l166_166637

theorem non_congruent_triangles_with_perimeter_10 :
  let integer_sides := {a, b, c : ℕ | a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a}
  let num_non_congruent := (integer_sides.to_finset.card / 6)
  num_non_congruent = 3 :=
by
  sorry

end non_congruent_triangles_with_perimeter_10_l166_166637


namespace calories_per_person_l166_166291

theorem calories_per_person (oranges : ℕ) (pieces_per_orange : ℕ) (people : ℕ) (calories_per_orange : ℕ) :
  oranges = 5 →
  pieces_per_orange = 8 →
  people = 4 →
  calories_per_orange = 80 →
  (oranges * pieces_per_orange) / people * ((oranges * calories_per_orange) / (oranges * pieces_per_orange)) = 100 :=
by
  intros h_oranges h_pieces_per_orange h_people h_calories_per_orange
  sorry

end calories_per_person_l166_166291


namespace range_of_possible_values_for_a_l166_166759

theorem range_of_possible_values_for_a {a : ℝ} : 
  (∀ x : ℝ, log a (x^2 + 3*x + a) ∈ ℝ) ↔ (a ∈ set.Ioo 0 1 ∪ set.Ioc 1 (9/4)) :=
by sorry

end range_of_possible_values_for_a_l166_166759


namespace bisections_to_approximation_l166_166248

theorem bisections_to_approximation :
  ∃ (n : ℕ), (2 < n ∧ n < 3) ∧ (1 / (2^n : ℝ) < 0.001) ∧ (n = 9) :=
by 
  use 9
  split
  · sorry
  · split
    · sorry
    · sorry

end bisections_to_approximation_l166_166248


namespace part_a_part_b_l166_166330

variable {A : Type*} [Ring A]

def B (A : Type*) [Ring A] : Set A :=
  {a | a^2 = 1}

variable (a : A) (b : B A)

theorem part_a (a : A) (b : A) (h : b ∈ B A) : a * b - b * a = b * a * b - a := by
  sorry

theorem part_b (A : Type*) [Ring A] (h : ∀ x : A, x^2 = 0 -> x = 0) : Group (B A) := by
  sorry

end part_a_part_b_l166_166330


namespace area_inside_C_outside_A_B_l166_166948

-- Define the given circles with corresponding radii and positions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the circles A, B, and C with the specific properties given
def CircleA : Circle := { center := (0, 0), radius := 1 }
def CircleB : Circle := { center := (2, 0), radius := 1 }
def CircleC : Circle := { center := (1, 2), radius := 2 }

-- Given that Circle C is tangent to the midpoint M of the line segment AB
-- Prove the area inside Circle C but outside Circle A and B
theorem area_inside_C_outside_A_B : 
  let area_inside_C := π * CircleC.radius ^ 2
  let overlap_area := (π - 2)
  area_inside_C - overlap_area = 3 * π + 2 := by
  sorry

end area_inside_C_outside_A_B_l166_166948


namespace sum_reciprocal_seq_a_l166_166214

-- Definitions
def seq_a : ℕ → ℕ
| 0     := 1
| (n+1) := seq_a n + n + 1

-- Theorem statement
theorem sum_reciprocal_seq_a : 
  (∑ i in finset.range 1001, (1 : ℚ) / seq_a i.succ) = 1001 / 501 :=
by
  -- Add proof here
  sorry

end sum_reciprocal_seq_a_l166_166214


namespace sphere_radius_given_surface_area_l166_166758

noncomputable def surface_area_of_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem sphere_radius_given_surface_area :
  ∃ r : ℝ, surface_area_of_sphere r = 2463.0086404143976 ∧ r = 14 :=
by
  use 14
  unfold surface_area_of_sphere
  linarith

end sphere_radius_given_surface_area_l166_166758


namespace unit_digit_2_pow_2024_l166_166589

theorem unit_digit_2_pow_2024 : (2 ^ 2024) % 10 = 6 := by
  -- We observe the repeating pattern in the unit digits of powers of 2:
  -- 2^1 = 2 -> unit digit is 2
  -- 2^2 = 4 -> unit digit is 4
  -- 2^3 = 8 -> unit digit is 8
  -- 2^4 = 16 -> unit digit is 6
  -- The cycle repeats every 4 powers: 2, 4, 8, 6
  -- 2024 ≡ 0 (mod 4), so it corresponds to the unit digit of 2^4, which is 6
  sorry

end unit_digit_2_pow_2024_l166_166589


namespace distinct_flags_count_l166_166509

theorem distinct_flags_count : 
  ∃ n, n = 36 ∧ (∀ c1 c2 c3 : Fin 4, c1 ≠ c2 ∧ c2 ≠ c3 → n = 4 * 3 * 3) := 
sorry

end distinct_flags_count_l166_166509


namespace percentage_decrease_of_b_l166_166761

variables (a b x m : ℝ) (p : ℝ)

-- Given conditions
def ratio_ab : Prop := a / b = 4 / 5
def expression_x : Prop := x = 1.25 * a
def expression_m : Prop := m = b * (1 - p / 100)
def ratio_mx : Prop := m / x = 0.6

-- The theorem to be proved
theorem percentage_decrease_of_b 
  (h1 : ratio_ab a b)
  (h2 : expression_x a x)
  (h3 : expression_m b m p)
  (h4 : ratio_mx m x) 
  : p = 40 :=
sorry

end percentage_decrease_of_b_l166_166761


namespace numbers_written_by_Kiara_and_Yndira_kiara_positive_numbers_l166_166296

theorem numbers_written_by_Kiara_and_Yndira
  (kiara_numbers : Finset ℤ) 
  (hk20 : kiara_numbers.card = 20) 
  (hnz : ∀ x ∈ kiara_numbers, x ≠ 0) 
  (total_numbers : 210 := 210) :
  kiara_numbers.card + (kiara_numbers.card * (kiara_numbers.card - 1)) = total_numbers :=
by
  sorry

theorem kiara_positive_numbers
  (kiara_numbers : Finset ℤ)
  (hk20 : kiara_numbers.card = 20)
  (hnz : ∀ x ∈ kiara_numbers, x ≠ 0)
  (total_numbers : ℕ)
  (total_positive : ℕ := 120)
  (kiara_numbers_positive : ℕ)
  (kiara_numbers_negative : ℕ := 20 - kiara_numbers_positive)
  (more_positive_than_negative : kiara_numbers_positive > kiara_numbers_negative) :
  ∃ p : ℕ, p = 14 ∧ p = kiara_numbers.filter (λ x, 0 < x).card :=
by
  sorry

end numbers_written_by_Kiara_and_Yndira_kiara_positive_numbers_l166_166296


namespace silverware_cost_20_l166_166292

-- Define the conditions
def cost_of_silverware (S : ℝ) : Prop :=
  let P := 0.5 * S in
  S + P = 30

-- Prove the main statement
theorem silverware_cost_20 : ∃ (S : ℝ), cost_of_silverware S ∧ S = 20 :=
by
  use 20
  sorry

end silverware_cost_20_l166_166292


namespace problem_proof_l166_166386

def scores : List ℕ := [88, 81, 96, 86, 97, 95, 90, 100, 87, 80, 
                        85, 86, 82, 90, 90, 100, 100, 94, 93, 100]

def data_organization (x : ℕ) : Prop :=
  (80 ≤ x ∧ x < 85) ∨ (85 ≤ x ∧ x < 90) ∨ (90 ≤ x ∧ x < 95) ∨ (95 ≤ x ∧ x < 100)

def frequency := 
  {80 ≤ x ∧ x < 85 | x ∈ scores}.card = 3 ∧
  {85 ≤ x ∧ x < 90 | x ∈ scores}.card = 5 ∧
  {90 ≤ x ∧ x < 95 | x ∈ scores}.card = 5 ∧
  {95 ≤ x ∧ x < 100 | x ∈ scores}.card = 7

noncomputable def average := 91
noncomputable def median := 90
noncomputable def mode := 100

theorem problem_proof :
  frequency ∧ average = 91 ∧ median = 90 ∧ mode = 100 ∧ 
  let sector_angle := 360 * 5 / 20 in 
  sector_angle = 90 ∧
  let estimated_people := 1400 * 12 / 20 in
  estimated_people = 840 :=
  sorry

end problem_proof_l166_166386


namespace fraction_of_previous_time_is_1_l166_166506

-- Define the known conditions
def distance : ℕ := 540 -- the distance in km
def original_time : ℕ := 6 -- the original time in hours
def new_speed : ℕ := 60 -- the new speed in km/h

-- Define the calculations
def original_speed := distance / original_time -- original speed in km/h
def new_time := distance / new_speed -- new time in hours
def time_fraction := new_time / original_time -- fraction of the original time

-- The main statement to be proved
theorem fraction_of_previous_time_is_1.5 : time_fraction = 3 / 2 := sorry

end fraction_of_previous_time_is_1_l166_166506


namespace find_original_number_l166_166473

theorem find_original_number (x : ℝ) (hx : 0 < x) 
  (h : 1000 * x = 3 * (1 / x)) : x ≈ 0.01732 :=
by 
  sorry

end find_original_number_l166_166473


namespace n_to_power_eight_plus_n_to_power_seven_plus_one_prime_l166_166984

theorem n_to_power_eight_plus_n_to_power_seven_plus_one_prime (n : ℕ) (hn_pos : n > 0) :
  (Nat.Prime (n^8 + n^7 + 1)) → (n = 1) :=
by
  sorry

end n_to_power_eight_plus_n_to_power_seven_plus_one_prime_l166_166984


namespace percent_absent_is_26_l166_166927

-- We bring in the necessary context and data
def total_students : ℕ := 150
def boys : ℕ := 90
def girls : ℕ := 60
def boys_absent_fraction : ℝ := 1 / 6
def girls_absent_fraction : ℝ := 2 / 5

-- Define the function to calculate absentees and assert the result in percent
theorem percent_absent_is_26 :
  100 * ((boys_absent_fraction * (boys : ℝ) + girls_absent_fraction * (girls : ℝ)) / (total_students : ℝ)) = 26 := 
by 
  sorry

end percent_absent_is_26_l166_166927


namespace sum_first_15_odd_integers_l166_166821

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l166_166821


namespace counterfeit_coin_two_weighings_100_counterfeit_coin_two_weighings_99_counterfeit_coin_two_weighings_98_l166_166634

theorem counterfeit_coin_two_weighings_100 :
  ∃ counterfeit : nat → Prop, ∀ (coins : finset ℕ), (coins.card = 100) → 
  ∃ l r : finset ℕ, l.card = 50 ∧ r.card = 50 ∧ 
  ( 
    (weigh(l) = weigh(r) → ∃ remaining : finset ℕ, remaining.card = 50 ∧ counterfeit ∈ remaining) ∨
    (weigh(l) ≠ weigh(r) → ∃ heavy_light, (heavy_light = heavy l r ∨ heavy_light = light l r) ∧ counterfeit ∈ heavy_light)
  ) ∧
  ∃ l2 r2 : finset ℕ, l2.card = 25 ∧ r2.card = 25 ∧ 
  (
    (weigh(l2) = weigh(r2) → ∃ remaining2 : finset ℕ, remaining2.card = 25 ∧ counterfeit ∈ remaining2) ∨
    (weigh(l2) ≠ weigh(r2) → ∃ heavy_light2, (heavy_light2 = heavy l2 r2 ∨ heavy_light2 = light l2 r2) ∧ counterfeit ∈ heavy_light2)
  ) := sorry

theorem counterfeit_coin_two_weighings_99 :
  ∃ counterfeit : nat → Prop, ∀ (coins : finset ℕ), (coins.card = 99) → 
  ∃ l r : finset ℕ, l.card = 33 ∧ r.card = 33 ∧ 
  ( 
    (weigh(l) = weigh(r) → ∃ remaining : finset ℕ, remaining.card = 33 ∧ counterfeit ∈ remaining) ∨
    (weigh(l) ≠ weigh(r) → ∃ heavy_light, (heavy_light = heavy l r ∨ heavy_light = light l r) ∧ counterfeit ∈ heavy_light)
  ) ∧
  ∃ l2 r2 : finset ℕ, l2.card = 11 ∧ r2.card = 11 ∧ 
  (
    (weigh(l2) = weigh(r2) → ∃ remaining2 : finset ℕ, remaining2.card = 11 ∧ counterfeit ∈ remaining2) ∨
    (weigh(l2) ≠ weigh(r2) → ∃ heavy_light2, (heavy_light2 = heavy l2 r2 ∨ heavy_light2 = light l2 r2) ∧ counterfeit ∈ heavy_light2)
  ) := sorry

theorem counterfeit_coin_two_weighings_98 :
  ∃ counterfeit : nat → Prop, ∀ (coins : finset ℕ), (coins.card = 98) → 
  ∃ l r : finset ℕ, l.card = 48 ∧ r.card = 48 ∧ 
  ( 
    (weigh(l) = weigh(r) → ∃ remaining : finset ℕ, remaining.card = 2 ∧ counterfeit ∈ remaining) ∨
    (weigh(l) ≠ weigh(r) → ∃ heavy_light, (heavy_light = heavy l r ∨ heavy_light = light l r) ∧ counterfeit ∈ heavy_light)
  ) ∧
  ∃ l2 r2 : finset ℕ, l2.card = 24 ∧ r2.card = 24 ∧ 
  (
    (weigh(l2) = weigh(r2) → ∃ remaining2 : finset ℕ, remaining2.card = 24 ∧ counterfeit ∈ remaining2) ∨
    (weigh(l2) ≠ weigh(r2) → ∃ heavy_light2, (heavy_light2 = heavy l2 r2 ∨ heavy_light2 = light l2 r2) ∧ counterfeit ∈ heavy_light2)
  ) := sorry

end counterfeit_coin_two_weighings_100_counterfeit_coin_two_weighings_99_counterfeit_coin_two_weighings_98_l166_166634


namespace apples_shared_among_friends_l166_166504

theorem apples_shared_among_friends : 
  ∃ X : ℕ,
    (let A := (X / 2 - 1 / 2),
         B := (A / 2 - 1 / 2),
         C := (B / 2 - 1 / 2),
         D := (C / 2 - 1 / 2),
         E := (D / 2 - 1 / 2)
     in E = 0) ∧ X = 15 :=
begin
  -- Proof goes here
  sorry
end

end apples_shared_among_friends_l166_166504


namespace point_Q_in_first_quadrant_l166_166647

theorem point_Q_in_first_quadrant (a b : ℝ) (h : a < 0 ∧ b < 0) : (0 < -a) ∧ (0 < -b) :=
by
  have ha : -a > 0 := by linarith
  have hb : -b > 0 := by linarith
  exact ⟨ha, hb⟩

end point_Q_in_first_quadrant_l166_166647


namespace slope_angle_of_line_l166_166763

theorem slope_angle_of_line (θ : ℝ) (t : ℝ) : 
  let x := λ t : ℝ, sin θ + t * sin (real.pi / 12)
  let y := λ t : ℝ, cos θ - t * sin (5 * real.pi / 12)
  -- show that the slope angle of this line is 105 degrees
  ∃ α : ℝ, (tan α = -cot (real.pi / 12)) ∧ 
          (α = 7 * real.pi / 12) :=
by
  sorry

end slope_angle_of_line_l166_166763


namespace clea_ride_escalator_time_l166_166949

theorem clea_ride_escalator_time
  (s v d : ℝ)
  (h1 : 75 * s = d)
  (h2 : 30 * (s + v) = d) :
  t = 50 :=
by
  sorry

end clea_ride_escalator_time_l166_166949


namespace smallest_period_of_3sin2x_l166_166154

noncomputable def smallest_positive_period (f : ℝ → ℝ) : ℝ := sorry

theorem smallest_period_of_3sin2x : 
  smallest_positive_period (λ x : ℝ, 3 * sin (2 * x)) = π := 
sorry

end smallest_period_of_3sin2x_l166_166154


namespace sum_first_15_odd_integers_l166_166811

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l166_166811


namespace length_of_QR_l166_166285

theorem length_of_QR {P Q R N : Type} 
  (PQ PR QR : ℝ) (QN NR PN : ℝ)
  (h1 : PQ = 5)
  (h2 : PR = 10)
  (h3 : QN = 3 * NR)
  (h4 : PN = 6)
  (h5 : QR = QN + NR) :
  QR = 724 / 3 :=
by sorry

end length_of_QR_l166_166285


namespace sum_first_15_odd_integers_l166_166823

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l166_166823


namespace equal_areas_RPK_RQL_l166_166663

variable {A B C R P Q K L : Type} [T : Triangle A B C]

-- Define the relevant points as living within the same ambient space
variables [is_angle_bisector (angle B C A) R C]
variables [is_circumcircle A B C R]
variables [is_perpendicular_bisector B C P]
variables [is_perpendicular_bisector A C Q]
variables [is_midpoint B C K]
variables [is_midpoint A C L]

theorem equal_areas_RPK_RQL 
  (h_bisector : is_angle_bisector (angle B C A) R C)
  (h_circumcircle : is_circumcircle A B C R)
  (h_perp_bisector_BC : is_perpendicular_bisector B C P)
  (h_perp_bisector_AC : is_perpendicular_bisector A C Q)
  (h_midpoint_BC : is_midpoint B C K)
  (h_midpoint_AC : is_midpoint A C L) : 
  area (triangle R P K) = area (triangle R Q L) := 
sorry

end equal_areas_RPK_RQL_l166_166663


namespace line_polar_equation_intersection_points_l166_166659

-- Definitions of the parametric equations of line l
def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 + (1 / 2) * t, (sqrt 3 / 2) * t)

-- Definition of the polar equation of curve C
def curve_C (theta : ℝ) : ℝ :=
  4 * cos theta

-- Theorem statement to prove the conversion of the parametric line to polar form
theorem line_polar_equation (t θ ρ : ℝ) (h1 : ρ = 4 * cos θ) : 
  (∃ t : ℝ, (2 + (1 / 2) * t, (sqrt 3 / 2) * t) = (ρ * cos θ, ρ * sin θ)) →
  (sqrt 3 * ρ * cos θ - ρ * sin θ - 2 * sqrt 3 = 0) :=
sorry

-- Theorem statement to prove the intersection points
theorem intersection_points (h1 : ρ = 4 * cos θ) : 
  (sqrt 3 * ρ * cos θ - ρ * sin θ - 2 * sqrt 3 = 0) →
  ((ρ = 2 ∧ θ = 5 * π / 3) ∨ (ρ = 2 * sqrt 3 ∧ θ = π / 6)) :=
sorry

end line_polar_equation_intersection_points_l166_166659


namespace intersection_points_l166_166953

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - x.floor

def circle_eq (x y : ℝ) : Prop :=
  (fractional_part x)^2 + y^2 = fractional_part x

def line_eq (x y : ℝ) : Prop :=
  y = (1/3) * x + (1/3)

theorem intersection_points :
  ∃ n : ℕ, ∀ x y : ℝ, (circle_eq x y ∧ line_eq x y) → n = sorry :=
sorry

end intersection_points_l166_166953


namespace recurring_decimal_to_fraction_l166_166979

theorem recurring_decimal_to_fraction : (∃ (x : ℚ), x = 3 + 56 / 99) :=
by
  have x : ℚ := 3 + 56 / 99
  exists x
  sorry

end recurring_decimal_to_fraction_l166_166979


namespace fraction_to_decimal_l166_166978

theorem fraction_to_decimal (n d : ℕ) (h : d = 160) : (n : ℚ) / d = 0.35625 := 
by
  -- Check that the denominator can be expressed as a product of powers of 2 and 5
  have factorization : d = 2^5 * 5 := by
    rw [h, show 160 = 2^5 * 5, by norm_num]
  -- Prove the fraction is equivalent to the terminating decimal 0.35625
  have equality : (n : ℚ) / d = 57 / 160 := by sorry
  have decimal : 57 / 160 = 0.35625 := by
    norm_num
  exact decimal

end fraction_to_decimal_l166_166978


namespace basketball_three_point_shots_l166_166009

theorem basketball_three_point_shots (t h f : ℕ) 
  (h1 : 2 * t = 6 * h)
  (h2 : f = h - 4)
  (h3: 2 * t + 3 * h + f = 76)
  (h4: t + h + f = 40) : h = 8 :=
sorry

end basketball_three_point_shots_l166_166009


namespace number_of_incorrect_statements_is_three_l166_166093

def A : Set ℕ := {0, 1}
def m (x : ℝ) := (x - 1) * (x + 2)
def p (a b : ℝ) := ∀ m : ℝ, a < b → a * m ^ 2 < b * m ^ 2

theorem number_of_incorrect_statements_is_three :
  ¬∃ s, s ∈ ({A} : Set (Set ℕ)) ∧ s.card = 3 ∨
  (∀ a b : ℝ, p a b → ¬(∃ m : ℝ, a = b)) ∨
  (∀ p q : Prop, (p ∨ q) → (p ∧ q)) ∨
  ¬(¬∃ x : ℝ, m x < 0 ∧ ¬(∀ x : ℝ, m x ≥ 0)) :=
sorry

end number_of_incorrect_statements_is_three_l166_166093


namespace isosceles_right_triangle_leg_length_l166_166534

theorem isosceles_right_triangle_leg_length :
  ∀ (hypotenuse : ℝ),
  hypotenuse = 2 →
  ∃ (leg_length : ℝ),
  (∃ (k : ℝ), k = 3 ∧ 3 * (1/2 * leg_length^2) = 1/2 * ((hypotenuse / Math.sqrt 2)^2) )  →
  leg_length = Math.sqrt (2 / 3) :=
by
  intro hypotenuse
  intro hypotenuse_eq
  use Math.sqrt (2 / 3)
  intros _
  sorry

end isosceles_right_triangle_leg_length_l166_166534


namespace induction_inequality_difference_expression_l166_166176

def f (n : ℕ) : ℚ := (List.range n).sum (λ i, 1 / (i + 1 : ℚ))

theorem induction_inequality (n : ℕ) : f (2^n) > n / 2 :=
by
  sorry

theorem difference_expression (k : ℕ) :
  f (2^(k+1)) - f (2^k) = (List.range ((2^(k+1) + 1) - (2^k + 1))).sum (λ i, 1 / (2^k + 1 + i : ℚ)) :=
by
  sorry

end induction_inequality_difference_expression_l166_166176


namespace vector_parallel_solution_l166_166632

theorem vector_parallel_solution :
  ∀ (x : ℝ), let a : ℝ × ℝ := (1, 3)
                 b : ℝ × ℝ := (2, x + 2)
             in (a.1 * b.2 = a.2 * b.1) → x = 4 :=
by
  intros x a b h
  let a := (1, 3)
  let b := (2, x + 2)
  cases a with a1 a2
  cases b with b1 b2
  sorry

end vector_parallel_solution_l166_166632


namespace four_digit_numbers_with_average_digit_l166_166636

theorem four_digit_numbers_with_average_digit :
  let digits := {d : Finset ℕ // ∀ x ∈ d, x > 0 ∧ x < 10},
      arithmetic_sequences := { seq : List ℕ // seq.length = 4 ∧ ∃ a b c, b = (a + c) / 2},
      num_sequences := 21,
      permutations := 24 in
  num_sequences * permutations = 504 :=
by
  sorry

end four_digit_numbers_with_average_digit_l166_166636


namespace original_number_value_l166_166453

noncomputable def orig_number_condition (x : ℝ) : Prop :=
  1000 * x = 3 * (1 / x)

theorem original_number_value : ∃ x : ℝ, 0 < x ∧ orig_number_condition x ∧ x = √(30) / 100 :=
begin
  -- the proof
  sorry
end

end original_number_value_l166_166453


namespace evaluate_ff_neg1_l166_166210

def f : ℝ → ℝ :=
  λ x, if x > 0 then log x / log 2 - 1 else |2 * x - 6|

theorem evaluate_ff_neg1 :
  f (f (-1) - 1) = log 7 / log 2 - 1 :=
by
  sorry

end evaluate_ff_neg1_l166_166210


namespace sum_of_first_n_odd_numbers_l166_166834

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l166_166834


namespace min_cubes_needed_l166_166080

def minimum_cubes_for_views (front_view side_view : ℕ) : ℕ :=
  4

theorem min_cubes_needed (front_view_cond side_view_cond : ℕ) :
  front_view_cond = 2 ∧ side_view_cond = 3 → minimum_cubes_for_views front_view_cond side_view_cond = 4 :=
by
  intro h
  cases h
  -- Proving the condition based on provided views
  sorry

end min_cubes_needed_l166_166080


namespace find_asterisk_value_l166_166484

theorem find_asterisk_value :
  ∃ x : ℤ, (x / 21) * (42 / 84) = 1 ↔ x = 21 :=
by
  sorry

end find_asterisk_value_l166_166484


namespace farmer_randy_total_acres_l166_166573

-- Define the conditions
def acres_per_tractor_per_day : ℕ := 68
def tractors_first_2_days : ℕ := 2
def days_first_period : ℕ := 2
def tractors_next_3_days : ℕ := 7
def days_second_period : ℕ := 3

-- Prove the total acres Farmer Randy needs to plant
theorem farmer_randy_total_acres :
  (tractors_first_2_days * acres_per_tractor_per_day * days_first_period) +
  (tractors_next_3_days * acres_per_tractor_per_day * days_second_period) = 1700 :=
by
  -- Here, we would provide the proof, but in this example, we will use sorry.
  sorry

end farmer_randy_total_acres_l166_166573


namespace parabola_intersects_x_axis_l166_166064

theorem parabola_intersects_x_axis 
  (a c : ℝ) 
  (h : ∃ x : ℝ, x = 1 ∧ (a * x^2 + x + c = 0)) : 
  a + c = -1 :=
sorry

end parabola_intersects_x_axis_l166_166064


namespace inequality_proof_l166_166720

theorem inequality_proof (a b : ℝ) : 
  (a^4 + a^2 * b^2 + b^4) / 3 ≥ (a^3 * b + b^3 * a) / 2 :=
by
  sorry

end inequality_proof_l166_166720


namespace fraction_division_l166_166036

theorem fraction_division : (3/4) / (5/8) = (6/5) := by
  sorry

end fraction_division_l166_166036


namespace lincoln_high_school_students_l166_166545

theorem lincoln_high_school_students (total students_in_either_or_both_clubs students_in_photography students_in_science : ℕ)
  (h1 : total = 300)
  (h2 : students_in_photography = 120)
  (h3 : students_in_science = 140)
  (h4 : students_in_either_or_both_clubs = 220):
  ∃ x, x = 40 ∧ (students_in_photography + students_in_science - students_in_either_or_both_clubs = x) := 
by
  use 40
  sorry

end lincoln_high_school_students_l166_166545


namespace sum_of_extremes_in_second_row_l166_166980

def spiral_grid_15x15 : list (list ℕ) := sorry

-- Define the function to locate the second row (row 1 in 0-based index)
def second_row (grid : list (list ℕ)) : list ℕ :=
  grid.nth 1

-- Define the maximal function
def max_in_list (l : list ℕ) : ℕ :=
  list.maximum l

-- Define the minimal function
def min_in_list (l : list ℕ) : ℕ :=
  list.minimum l

theorem sum_of_extremes_in_second_row :
  let grid := spiral_grid_15x15 in
  let row := second_row grid in
  max_in_list row + min_in_list row = 367 :=
sorry

end sum_of_extremes_in_second_row_l166_166980


namespace functional_equation_solution_l166_166983

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2) + f (2 * y^2) = (f (x + y) + f y) * (f (x - y) + f y)) →
  (f = (λ x, x^2) ∨ f = (λ x, 0) ∨ f = (λ x, 1 / 2)) :=
by sorry

end functional_equation_solution_l166_166983


namespace infinite_terms_ending_with_2024_l166_166127

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 2024
  else sequence (n - 1) + sequence (n - 2)

theorem infinite_terms_ending_with_2024 :
  ∃ (S : ℕ → ℕ) (P : ℕ), 
    (∀ n, S n = sequence n % 10000) ∧ 
    (∀ k : ℕ, ∃ m : ℕ, m > k ∧ S m = 2024) :=
    sorry

end infinite_terms_ending_with_2024_l166_166127


namespace bridge_length_is_correct_l166_166526

noncomputable def train_length : ℝ := 135
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def bridge_crossing_time : ℝ := 30

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
noncomputable def total_distance_crossed : ℝ := train_speed_ms * bridge_crossing_time
noncomputable def bridge_length : ℝ := total_distance_crossed - train_length

theorem bridge_length_is_correct : bridge_length = 240 := by
  sorry

end bridge_length_is_correct_l166_166526


namespace new_girl_weight_l166_166493

theorem new_girl_weight (W : ℝ) (h : (W + 24) / 8 = W / 8 + 3) :
  (W + 24) - (W - 70) = 94 :=
by
  sorry

end new_girl_weight_l166_166493


namespace original_number_solution_l166_166447

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end original_number_solution_l166_166447


namespace vector_parallel_l166_166175

theorem vector_parallel (x : ℝ) : (1, 2) ∥ (2 * x, -3) → x = -3 / 4 :=
by
  sorry

end vector_parallel_l166_166175


namespace reciprocal_of_neg_three_arcseconds_to_degrees_inequality_neg_fractions_l166_166015

-- Question 1: Prove the reciprocal of -3 is -1/3
theorem reciprocal_of_neg_three : (-3)⁻¹ = -1 / 3 :=
sorry

-- Question 2: Prove that 7200 arcseconds is equivalent to 2 degrees
theorem arcseconds_to_degrees : (7200 / 3600 : ℝ) = 2 :=
sorry

-- Question 3: Prove that -3/4 is greater than -4/5
theorem inequality_neg_fractions : (-3 / 4 : ℝ) > -4 / 5 :=
sorry

end reciprocal_of_neg_three_arcseconds_to_degrees_inequality_neg_fractions_l166_166015


namespace range_of_function_l166_166014

theorem range_of_function : 
  ∀ (x : ℝ), 2^x + 1 > 1 → 0 < 1 / (2^x + 1) ∧ 1 / (2^x + 1) < 1 :=
by
  intro x h
  split
  sorry
  sorry

end range_of_function_l166_166014


namespace original_price_l166_166879

-- Definitions based on the problem conditions
variables (P : ℝ)

def john_payment (P : ℝ) : ℝ :=
  0.9 * P + 0.15 * P

def jane_payment (P : ℝ) : ℝ :=
  0.9 * P + 0.15 * (0.9 * P)

def price_difference (P : ℝ) : ℝ :=
  john_payment P - jane_payment P

theorem original_price (h : price_difference P = 0.51) : P = 34 := 
by
  sorry

end original_price_l166_166879


namespace regular_decagon_product_l166_166334

theorem regular_decagon_product :
  let P := [ (1, 0), (?, ?), (?, ?), (?, ?), (?, ?), (3, 0), (?, ?), (?, ?), (?, ?), (?, ?) ] in
  ∀ (P₁ P₂ P₃ P₄ P₅ P₆ P₇ P₈ P₉ P₁₀ : ℂ),
  P₁ = (1 + 0 * I) ∧ P₆ = (3 + 0 * I) ∧
  P₂ = P[1] ∧ P₃ = P[2] ∧ 
  P₄ = P[3] ∧ P₅ = P[4] ∧
  P₇ = P[6] ∧ P₈ = P[7] ∧
  P₉ = P[8] ∧ P₁₀ = P[9] ∧
  (∀ k, 1 ≤ k → k ≤ 10 → P[k] * P[k] = (2 + 0 * I)) →
  ((P₁ * P₂ * P₃ * P₄ * P₅ * P₆ * P₇ * P₈ * P₉ * P₁₀) = 1023) :=
  sorry

end regular_decagon_product_l166_166334


namespace chess_tournament_l166_166020

theorem chess_tournament (n k : ℕ) (S : ℕ) (m : ℕ) 
  (h1 : S ≤ k * n) 
  (h2 : S ≥ m * n) 
  : m ≤ k := 
by 
  sorry

end chess_tournament_l166_166020


namespace simplified_sum_of_series_l166_166277

noncomputable def sum_of_series (x : ℝ) : ℝ :=
  (∑ k in Finset.range(2020) + 1, (x^k + x^(-k))^2)

theorem simplified_sum_of_series (x : ℝ) :
  sum_of_series x = 4040 + (x^8082 - 1) / (x^4040 * (x^2 - 1)) :=
by
  sorry

end simplified_sum_of_series_l166_166277


namespace parabola_conditions_l166_166003

theorem parabola_conditions 
  (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b = 2 * a) 
  (hc : c = -3 * a) 
  (hA : a * (-3)^2 + b * (-3) + c = 0) 
  (hB : a * (1)^2 + b * (1) + c = 0) : 
  (b^2 - 4 * a * c > 0) ∧ (3 * b + 2 * c = 0) :=
sorry

end parabola_conditions_l166_166003


namespace sum_first_15_odd_integers_l166_166816

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l166_166816


namespace exists_powers_mod_eq_l166_166335

theorem exists_powers_mod_eq (N : ℕ) (A : ℤ) : ∃ r s : ℕ, r ≠ s ∧ (A ^ r - A ^ s) % N = 0 :=
sorry

end exists_powers_mod_eq_l166_166335


namespace prove_length_of_PB_l166_166261

variables {A B C D P : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup P]

def point := (ℝ × ℝ)

variables A B C D P : point

variables (conv_quad : ∀ {A B C D : point}, Prop)
variables (perp : (point × point) → (point × point) → Prop)

axiom condition1 : conv_quad A B C D
axiom condition2 : perp (C, D) (A, B)
axiom condition3 : perp (B, C) (A, D)
axiom condition4 : dist C D = 39
axiom condition5 : dist B C = 45
axiom condition6 : ∃ (P : point), perp (C, P) (B, D) ∧ ∃ (s : ℝ), s > 0 ∧ (A, P) = (s, 0)
axiom condition7 : dist A P = 5

theorem prove_length_of_PB :
  dist P B = 400 :=
sorry

end prove_length_of_PB_l166_166261


namespace slopes_of_altitudes_l166_166612

def point := (ℝ × ℝ)
def A : point := (-1, 0)
def B : point := (1, 1)
def C : point := (0, 2)

-- Slope calculation function
def slope (p1 p2 : point) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Prove the slopes of the altitudes on the sides of ΔABC
theorem slopes_of_altitudes (k1 k2 k3 : ℝ) :
  slope A B = 1/2 → slope A C = 2 → slope B C = -1 →
  k1 = -2 ∧ k2 = -1/2 ∧ k3 = 1 :=
by
  intros hAB hAC hBC
  have h1 := (1/2) * k1 = -1
  have h2 := 2 * k2 = -1
  have h3 := (-1) * k3 = -1
  -- Fill in necessary manipulation to use hAB, hAC, and hBC ensuring the proof
  sorry

end slopes_of_altitudes_l166_166612


namespace find_f_prime_half_l166_166247

def f (x : ℝ) (f'1 : ℝ) : ℝ := ln(x - f'1 * x^2) + 5 * x - 4
def f_prime (x : ℝ) (f'1 : ℝ) : ℝ := 1 / (x - f'1 * x^2) * (1 - 2 * f'1 * x) + 5

theorem find_f_prime_half : f_prime (1 / 2) 2 = 5 := by
  sorry

end find_f_prime_half_l166_166247


namespace div_by_240_l166_166591

theorem div_by_240 (a b c d : ℕ) : 240 ∣ (a ^ (4 * b + d) - a ^ (4 * c + d)) :=
sorry

end div_by_240_l166_166591


namespace count_eligible_three_digit_numbers_l166_166234

def is_eligible_digit (d : Nat) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem count_eligible_three_digit_numbers : 
  (∃ n : Nat, 100 ≤ n ∧ n < 1000 ∧
  (∀ d : Nat, d ∈ [n / 100, (n / 10) % 10, n % 10] → is_eligible_digit d)) →
  ∃ count : Nat, count = 343 :=
by
  sorry

end count_eligible_three_digit_numbers_l166_166234


namespace binomial_theorem_problem_statement_l166_166583

-- Binomial Coefficient definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Binomial Theorem
theorem binomial_theorem (a b : ℝ) (n : ℕ) : (a + b) ^ n = ∑ k in Finset.range (n + 1), binom n k • (a ^ (n - k) * b ^ k) := sorry

-- Problem Statement
theorem problem_statement (n : ℕ) : (∑ k in Finset.filter (λ x => x % 2 = 0) (Finset.range (2 * n + 1)), binom (2 * n) k * 9 ^ (k / 2)) = 2^(2*n-1) + 8^(2*n-1) := sorry

end binomial_theorem_problem_statement_l166_166583


namespace find_pqr_l166_166981

variable (p q r : ℚ)

theorem find_pqr (h1 : ∃ a : ℚ, ∀ x : ℚ, (p = a) ∧ (q = -2 * a * 3) ∧ (r = a * 3 * 3 + 7) ∧ (r = 10 + 7)) :
  p + q + r = 8 + 1/3 := by
  sorry

end find_pqr_l166_166981


namespace sum_first_15_odd_integers_l166_166819

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l166_166819


namespace gcd_of_three_numbers_l166_166355

theorem gcd_of_three_numbers :
  Nat.gcd (Nat.gcd 72 120) 168 = 24 :=
sorry

end gcd_of_three_numbers_l166_166355
