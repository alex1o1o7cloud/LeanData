import Mathlib

namespace houses_in_block_l2401_240174

theorem houses_in_block (junk_mail_per_house : ℕ) (total_junk_mail : ℕ) (h1 : junk_mail_per_house = 2) (h2 : total_junk_mail = 14) :
  total_junk_mail / junk_mail_per_house = 7 := by
  sorry

end houses_in_block_l2401_240174


namespace cube_problem_l2401_240161

theorem cube_problem (n : ℕ) (H1 : 6 * n^2 = 1 / 3 * 6 * n^3) : n = 3 :=
sorry

end cube_problem_l2401_240161


namespace boys_girls_difference_l2401_240110

/--
If there are 550 students in a class and the ratio of boys to girls is 7:4, 
prove that the number of boys exceeds the number of girls by 150.
-/
theorem boys_girls_difference : 
  ∀ (students boys_ratio girls_ratio : ℕ),
  students = 550 →
  boys_ratio = 7 →
  girls_ratio = 4 →
  (students * boys_ratio) % (boys_ratio + girls_ratio) = 0 ∧
  (students * girls_ratio) % (boys_ratio + girls_ratio) = 0 →
  (students * boys_ratio - students * girls_ratio) / (boys_ratio + girls_ratio) = 150 :=
by
  intros students boys_ratio girls_ratio h_students h_boys_ratio h_girls_ratio h_divisibility
  -- The detailed proof would follow here, but we add 'sorry' to bypass it.
  sorry

end boys_girls_difference_l2401_240110


namespace sin_square_range_l2401_240114

def range_sin_square_values (α β : ℝ) : Prop :=
  3 * (Real.sin α) ^ 2 - 2 * Real.sin α + 2 * (Real.sin β) ^ 2 = 0

theorem sin_square_range (α β : ℝ) (h : range_sin_square_values α β) :
  0 ≤ (Real.sin α) ^ 2 + (Real.sin β) ^ 2 ∧ 
  (Real.sin α) ^ 2 + (Real.sin β) ^ 2 ≤ 4 / 9 :=
sorry

end sin_square_range_l2401_240114


namespace problem_system_of_equations_l2401_240180

-- Define the problem as a theorem in Lean 4
theorem problem_system_of_equations (x y c d : ℝ) (h1 : 4 * x + 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 :=
by
  -- The proof is omitted
  sorry

end problem_system_of_equations_l2401_240180


namespace julia_shortfall_l2401_240138

-- Definitions based on the problem conditions
def rock_and_roll_price : ℕ := 5
def pop_price : ℕ := 10
def dance_price : ℕ := 3
def country_price : ℕ := 7
def quantity : ℕ := 4
def julia_money : ℕ := 75

-- Proof problem: Prove that Julia is short $25
theorem julia_shortfall : (quantity * rock_and_roll_price + quantity * pop_price + quantity * dance_price + quantity * country_price) - julia_money = 25 := by
  sorry

end julia_shortfall_l2401_240138


namespace sam_weight_l2401_240134

theorem sam_weight (Tyler Sam Peter : ℕ) : 
  (Peter = 65) →
  (Peter = Tyler / 2) →
  (Tyler = Sam + 25) →
  Sam = 105 :=
  by
  intros hPeter1 hPeter2 hTyler
  sorry

end sam_weight_l2401_240134


namespace perpendicular_vectors_x_eq_5_l2401_240130

def vector_a (x : ℝ) : ℝ × ℝ := (2, x + 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x - 2, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors_x_eq_5 (x : ℝ)
  (h : dot_product (vector_a x) (vector_b x) = 0) :
  x = 5 :=
sorry

end perpendicular_vectors_x_eq_5_l2401_240130


namespace simplify_and_evaluate_at_x_eq_4_l2401_240164

noncomputable def simplify_and_evaluate (x : ℚ) : ℚ :=
  (x - 1 - (3 / (x + 1))) / ((x^2 - 2*x) / (x + 1))

theorem simplify_and_evaluate_at_x_eq_4 : simplify_and_evaluate 4 = 3 / 2 := by
  sorry

end simplify_and_evaluate_at_x_eq_4_l2401_240164


namespace percentage_is_26_53_l2401_240193

noncomputable def percentage_employees_with_six_years_or_more (y: ℝ) : ℝ :=
  let total_employees := 10*y + 4*y + 6*y + 5*y + 8*y + 3*y + 5*y + 4*y + 2*y + 2*y
  let employees_with_six_years_or_more := 5*y + 4*y + 2*y + 2*y
  (employees_with_six_years_or_more / total_employees) * 100

theorem percentage_is_26_53 (y: ℝ) (hy: y ≠ 0): percentage_employees_with_six_years_or_more y = 26.53 :=
by
  sorry

end percentage_is_26_53_l2401_240193


namespace polynomial_distinct_positive_roots_l2401_240173

theorem polynomial_distinct_positive_roots (a b : ℝ) (P : ℝ → ℝ) (hP : ∀ x, P x = x^3 + a * x^2 + b * x - 1) 
(hroots : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0) : 
  P (-1) < -8 := 
by
  sorry

end polynomial_distinct_positive_roots_l2401_240173


namespace pair_exists_l2401_240167

def exists_pair (a b : ℕ → ℕ) : Prop :=
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q

theorem pair_exists (a b : ℕ → ℕ) : exists_pair a b :=
sorry

end pair_exists_l2401_240167


namespace find_original_number_l2401_240190

theorem find_original_number (x : ℕ) (h1 : 10 * x + 9 + 2 * x = 633) : x = 52 :=
by
  sorry

end find_original_number_l2401_240190


namespace complement_union_l2401_240123

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_union :
  (U \ M) ∪ N = {2, 3, 4} :=
sorry

end complement_union_l2401_240123


namespace abs_sum_condition_l2401_240140

theorem abs_sum_condition (a b : ℝ) (h₁ : |a| = 2) (h₂ : b = -1) : |a + b| = 1 ∨ |a + b| = 3 :=
by
  sorry

end abs_sum_condition_l2401_240140


namespace triangle_area_specific_l2401_240122

noncomputable def vector2_area_formula (u v : ℝ × ℝ) : ℝ :=
|u.1 * v.2 - u.2 * v.1|

noncomputable def triangle_area (u v : ℝ × ℝ) : ℝ :=
(vector2_area_formula u v) / 2

theorem triangle_area_specific :
  let A := (1, 3)
  let B := (5, -1)
  let C := (9, 4)
  let u := (1 - 9, 3 - 4)
  let v := (5 - 9, -1 - 4)
  triangle_area u v = 18 := 
by sorry

end triangle_area_specific_l2401_240122


namespace eldest_age_l2401_240145

theorem eldest_age (A B C : ℕ) (x : ℕ) 
  (h1 : A = 5 * x)
  (h2 : B = 7 * x)
  (h3 : C = 8 * x)
  (h4 : (5 * x - 7) + (7 * x - 7) + (8 * x - 7) = 59) :
  C = 32 := 
by 
  sorry

end eldest_age_l2401_240145


namespace gymnast_score_difference_l2401_240184

theorem gymnast_score_difference 
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x2 + x3 + x4 + x5 = 36)
  (h2 : x1 + x2 + x3 + x4 = 36.8) :
  x1 - x5 = 0.8 :=
by sorry

end gymnast_score_difference_l2401_240184


namespace tank_fill_time_l2401_240182

theorem tank_fill_time (R1 R2 t_required : ℝ) (hR1: R1 = 1 / 8) (hR2: R2 = 1 / 12) (hT : t_required = 4.8) :
  t_required = 1 / (R1 + R2) :=
by 
  -- Proof goes here
  sorry

end tank_fill_time_l2401_240182


namespace triangle_right_angled_l2401_240124

theorem triangle_right_angled
  (a b c : ℝ) (A B C : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : A + B + C = π)
  (h₅ : b * Real.cos C + c * Real.cos B = a * Real.sin A) :
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 :=
sorry

end triangle_right_angled_l2401_240124


namespace no_real_solutions_eqn_l2401_240159

theorem no_real_solutions_eqn : ∀ x : ℝ, (2 * x - 4 * x + 7)^2 + 1 ≠ -|x^2 - 1| :=
by
  intro x
  sorry

end no_real_solutions_eqn_l2401_240159


namespace unique_function_l2401_240142

theorem unique_function (f : ℝ → ℝ) 
  (H : ∀ (x y : ℝ), f (f x + 9 * y) = f y + 9 * x + 24 * y) : 
  ∀ x : ℝ, f x = 3 * x :=
by 
  sorry

end unique_function_l2401_240142


namespace probability_three_dice_less_than_seven_l2401_240170

open Nat

def probability_of_exactly_three_less_than_seven (dice_count : ℕ) (sides : ℕ) (target_faces : ℕ) : ℚ :=
  let p : ℚ := target_faces / sides
  let q : ℚ := 1 - p
  (Nat.choose dice_count (dice_count / 2)) * (p^(dice_count / 2)) * (q^(dice_count / 2))

theorem probability_three_dice_less_than_seven :
  probability_of_exactly_three_less_than_seven 6 12 6 = 5 / 16 := by
  sorry

end probability_three_dice_less_than_seven_l2401_240170


namespace y_percent_of_x_l2401_240149

theorem y_percent_of_x (x y : ℝ) (h : 0.60 * (x - y) = 0.20 * (x + y)) : y / x = 0.5 :=
sorry

end y_percent_of_x_l2401_240149


namespace radius_of_tangent_intersection_l2401_240141

variable (x y : ℝ)

def circle_eq : Prop := x^2 + y^2 = 25

def tangent_condition : Prop := y = 5 ∧ x = 0

theorem radius_of_tangent_intersection (h1 : circle_eq x y) (h2 : tangent_condition x y) : ∃r : ℝ, r = 5 :=
by sorry

end radius_of_tangent_intersection_l2401_240141


namespace min_value_f_l2401_240162

open Real

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (15 - 12 * cos x) + 
  sqrt (4 - 2 * sqrt 3 * sin x) +
  sqrt (7 - 4 * sqrt 3 * sin x) +
  sqrt (10 - 4 * sqrt 3 * sin x - 6 * cos x)

theorem min_value_f : ∃ x : ℝ, f x = 6 := 
sorry

end min_value_f_l2401_240162


namespace trigonometric_identity_l2401_240109

variable {α β γ n : Real}

-- Condition:
axiom h : Real.sin (2 * (α + γ)) = n * Real.sin (2 * β)

-- Statement to be proved:
theorem trigonometric_identity : 
  Real.tan (α + β + γ) / Real.tan (α - β + γ) = (n + 1) / (n - 1) :=
by
  sorry

end trigonometric_identity_l2401_240109


namespace atomic_weight_S_is_correct_l2401_240178

-- Conditions
def molecular_weight_BaSO4 : Real := 233
def atomic_weight_Ba : Real := 137.33
def atomic_weight_O : Real := 16
def num_O_in_BaSO4 : Nat := 4

-- Definition of total weight of Ba and O
def total_weight_Ba_O := atomic_weight_Ba + num_O_in_BaSO4 * atomic_weight_O

-- Expected atomic weight of S
def atomic_weight_S : Real := molecular_weight_BaSO4 - total_weight_Ba_O

-- Theorem to prove that the atomic weight of S is 31.67
theorem atomic_weight_S_is_correct : atomic_weight_S = 31.67 := by
  -- placeholder for the proof
  sorry

end atomic_weight_S_is_correct_l2401_240178


namespace new_students_count_l2401_240176

theorem new_students_count (O N : ℕ) (avg_class_age avg_new_students_age avg_decrease original_strength : ℕ)
  (h1 : avg_class_age = 40)
  (h2 : avg_new_students_age = 32)
  (h3 : avg_decrease = 4)
  (h4 : original_strength = 8)
  (total_age_class : ℕ := avg_class_age * original_strength)
  (new_avg_age : ℕ := avg_class_age - avg_decrease)
  (total_age_new_students : ℕ := avg_new_students_age * N)
  (total_students : ℕ := original_strength + N)
  (new_total_age : ℕ := total_age_class + total_age_new_students)
  (new_avg_class_age : ℕ := new_total_age / total_students)
  (h5 : new_avg_class_age = new_avg_age) : N = 8 :=
by
  sorry

end new_students_count_l2401_240176


namespace rectangle_perimeter_l2401_240175

variable (a b : ℕ)

theorem rectangle_perimeter (h1 : a ≠ b) (h2 : ab = 8 * (a + b)) : 
  2 * (a + b) = 66 := 
sorry

end rectangle_perimeter_l2401_240175


namespace atomic_weight_Br_correct_l2401_240131

def atomic_weight_Ba : ℝ := 137.33
def molecular_weight_compound : ℝ := 297
def atomic_weight_Br : ℝ := 79.835

theorem atomic_weight_Br_correct :
  molecular_weight_compound = atomic_weight_Ba + 2 * atomic_weight_Br :=
by
  sorry

end atomic_weight_Br_correct_l2401_240131


namespace tangent_line_equation_l2401_240111

noncomputable def f (x : ℝ) : ℝ := (2 + Real.sin x) / Real.cos x

theorem tangent_line_equation :
  let x0 : ℝ := 0
  let y0 : ℝ := f x0
  let m : ℝ := (2 * x0 + 1) / (Real.cos x0 ^ 2)
  ∃ (a b c : ℝ), a * x0 + b * y0 + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
by
  sorry

end tangent_line_equation_l2401_240111


namespace ellipse_iff_constant_sum_l2401_240192

-- Let F_1 and F_2 be two fixed points in the plane.
variables (F1 F2 : Point)
-- Let d be a constant.
variable (d : ℝ)

-- A point M in a plane
variable (M : Point)

-- Define the distance function between two points.
def dist (P Q : Point) : ℝ := sorry

-- Definition: M is on an ellipse with foci F1 and F2
def on_ellipse (M F1 F2 : Point) (d : ℝ) : Prop :=
  dist M F1 + dist M F2 = d

-- Proof that shows the two parts of the statement
theorem ellipse_iff_constant_sum :
  (∀ M, on_ellipse M F1 F2 d) ↔ (∀ M, dist M F1 + dist M F2 = d) ∧ d > dist F1 F2 :=
sorry

end ellipse_iff_constant_sum_l2401_240192


namespace total_tennis_balls_used_l2401_240115

theorem total_tennis_balls_used :
  let rounds := [1028, 514, 257, 128, 64, 32, 16, 8, 4]
  let cans_per_game_A := 6
  let cans_per_game_B := 8
  let balls_per_can_A := 3
  let balls_per_can_B := 4
  let games_A_to_B := rounds.splitAt 4
  let total_A := games_A_to_B.1.sum * cans_per_game_A * balls_per_can_A
  let total_B := games_A_to_B.2.sum * cans_per_game_B * balls_per_can_B
  total_A + total_B = 37573 := 
by
  sorry

end total_tennis_balls_used_l2401_240115


namespace num_unpainted_cubes_l2401_240119

theorem num_unpainted_cubes (n : ℕ) (h1 : n ^ 3 = 125) : (n - 2) ^ 3 = 27 :=
by
  sorry

end num_unpainted_cubes_l2401_240119


namespace salary_increase_difference_l2401_240139

structure Person where
  name : String
  salary : ℕ
  raise_percent : ℕ
  investment_return : ℕ

def hansel := Person.mk "Hansel" 30000 10 5
def gretel := Person.mk "Gretel" 30000 15 4
def rapunzel := Person.mk "Rapunzel" 40000 8 6
def rumpelstiltskin := Person.mk "Rumpelstiltskin" 35000 12 7
def cinderella := Person.mk "Cinderella" 45000 7 8
def jack := Person.mk "Jack" 50000 6 10

def salary_increase (p : Person) : ℕ := p.salary * p.raise_percent / 100
def investment_return (p : Person) : ℕ := salary_increase p * p.investment_return / 100
def total_increase  (p : Person) : ℕ := salary_increase p + investment_return p

def problem_statement : Prop :=
  let hansel_increase := total_increase hansel
  let gretel_increase := total_increase gretel
  let rapunzel_increase := total_increase rapunzel
  let rumpelstiltskin_increase := total_increase rumpelstiltskin
  let cinderella_increase := total_increase cinderella
  let jack_increase := total_increase jack

  let highest_increase := max gretel_increase (max rumpelstiltskin_increase (max cinderella_increase (max rapunzel_increase (max jack_increase hansel_increase))))
  let lowest_increase := min gretel_increase (min rumpelstiltskin_increase (min cinderella_increase (min rapunzel_increase (min jack_increase hansel_increase))))

  highest_increase - lowest_increase = 1530

theorem salary_increase_difference : problem_statement := by
  sorry

end salary_increase_difference_l2401_240139


namespace multiplier_for_second_part_l2401_240160

theorem multiplier_for_second_part {x y k : ℝ} (h1 : x + y = 52) (h2 : 10 * x + k * y = 780) (hy : y = 30.333333333333332) (hx : x = 21.666666666666668) :
  k = 18.571428571428573 :=
by
  sorry

end multiplier_for_second_part_l2401_240160


namespace part1_part2_l2401_240113

noncomputable def f (x m : ℝ) : ℝ := abs (x - m) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 ≤ x + 4 ↔ 0 ≤ x ∧ x ≤ 2 := sorry

theorem part2 (m n t : ℝ) (hm : m > 0) (hn : n > 0) (ht : t > 0) 
  (hmin : ∀ x, f x m ≥ 5 - n - t) :
  1 / (m + n) + 1 / t ≥ 2 := sorry

end part1_part2_l2401_240113


namespace benny_has_24_books_l2401_240171

def books_sandy : ℕ := 10
def books_tim : ℕ := 33
def total_books : ℕ := 67

def books_benny : ℕ := total_books - (books_sandy + books_tim)

theorem benny_has_24_books : books_benny = 24 := by
  unfold books_benny
  unfold total_books
  unfold books_sandy
  unfold books_tim
  sorry

end benny_has_24_books_l2401_240171


namespace equal_profit_for_Robi_and_Rudy_l2401_240135

theorem equal_profit_for_Robi_and_Rudy
  (robi_contrib : ℕ)
  (rudy_extra_contrib : ℕ)
  (profit_percent : ℚ)
  (share_profit_equally : Prop)
  (total_profit: ℚ)
  (each_share: ℕ) :
  robi_contrib = 4000 →
  rudy_extra_contrib = (1/4) * robi_contrib →
  profit_percent = 0.20 →
  share_profit_equally →
  total_profit = profit_percent * (robi_contrib + robi_contrib + rudy_extra_contrib) →
  each_share = (total_profit / 2) →
  each_share = 900 :=
by {
  sorry
}

end equal_profit_for_Robi_and_Rudy_l2401_240135


namespace polynomial_sum_at_points_l2401_240196

def P (x : ℝ) : ℝ := x^5 - 1.7 * x^3 + 2.5

theorem polynomial_sum_at_points :
  P 19.1 + P (-19.1) = 5 := by
  sorry

end polynomial_sum_at_points_l2401_240196


namespace sectors_containing_all_numbers_l2401_240191

theorem sectors_containing_all_numbers (n : ℕ) (h : 0 < n) :
  ∃ (s : Finset (Fin (2 * n))), (s.card = n) ∧ (∀ i : Fin n, ∃ j : Fin (2 * n), j ∈ s ∧ (j.val % n) + 1 = i.val) :=
  sorry

end sectors_containing_all_numbers_l2401_240191


namespace integer_triples_soln_l2401_240165

theorem integer_triples_soln (x y z : ℤ) :
  (x^3 + y^3 + z^3 - 3*x*y*z = 2003) ↔ ( (x = 668 ∧ y = 668 ∧ z = 667) ∨ (x = 668 ∧ y = 667 ∧ z = 668) ∨ (x = 667 ∧ y = 668 ∧ z = 668) ) := 
by
  sorry

end integer_triples_soln_l2401_240165


namespace degree_measure_supplement_complement_l2401_240126

theorem degree_measure_supplement_complement : 
  let alpha := 63 -- angle value
  let theta := 90 - alpha -- complement of the angle
  let phi := 180 - theta -- supplement of the complement
  phi = 153 := -- prove the final step
by
  sorry

end degree_measure_supplement_complement_l2401_240126


namespace correct_calculated_value_l2401_240150

theorem correct_calculated_value (x : ℤ) (h : x - 749 = 280) : x + 479 = 1508 :=
by 
  sorry

end correct_calculated_value_l2401_240150


namespace chain_of_tangent_circles_exists_iff_integer_angle_multiple_l2401_240148

noncomputable def angle_between_tangent_circles (R₁ R₂ : Circle) (line : Line) : ℝ :=
-- the definition should specify how we get the angle between the tangent circles
sorry

def n_tangent_circles_exist (R₁ R₂ : Circle) (n : ℕ) : Prop :=
-- the definition should specify the existence of a chain of n tangent circles
sorry

theorem chain_of_tangent_circles_exists_iff_integer_angle_multiple 
  (R₁ R₂ : Circle) (n : ℕ) (line : Line) : 
  n_tangent_circles_exist R₁ R₂ n ↔ ∃ k : ℤ, angle_between_tangent_circles R₁ R₂ line = k * (360 / n) :=
sorry

end chain_of_tangent_circles_exists_iff_integer_angle_multiple_l2401_240148


namespace evaluate_expression_l2401_240129

theorem evaluate_expression : 
  ( (5 ^ 2014) ^ 2 - (5 ^ 2012) ^ 2 ) / ( (5 ^ 2013) ^ 2 - (5 ^ 2011) ^ 2 ) = 25 := 
by sorry

end evaluate_expression_l2401_240129


namespace rank_from_start_l2401_240137

theorem rank_from_start (n r_l : ℕ) (h_n : n = 31) (h_r_l : r_l = 15) : n - (r_l - 1) = 17 := by
  sorry

end rank_from_start_l2401_240137


namespace area_of_isosceles_right_triangle_l2401_240136

def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  (a = b) ∧ (a^2 + b^2 = c^2)

theorem area_of_isosceles_right_triangle (a : ℝ) (hypotenuse : ℝ) (h_isosceles : is_isosceles_right_triangle a a hypotenuse) (h_hypotenuse : hypotenuse = 6) :
  (1 / 2) * a * a = 9 :=
by
  sorry

end area_of_isosceles_right_triangle_l2401_240136


namespace valentines_given_l2401_240197

theorem valentines_given (x y : ℕ) (h : x * y = x + y + 40) : x * y = 84 :=
by
  -- solving for x, y based on the factors of 41
  sorry

end valentines_given_l2401_240197


namespace lemon_ratio_l2401_240154

variable (Levi Jayden Eli Ian : ℕ)

theorem lemon_ratio (h1: Levi = 5)
    (h2: Jayden = Levi + 6)
    (h3: Jayden = Eli / 3)
    (h4: Levi + Jayden + Eli + Ian = 115) :
    Eli = Ian / 2 :=
by
  sorry

end lemon_ratio_l2401_240154


namespace twin_brothers_age_l2401_240108

theorem twin_brothers_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 17) : x = 8 := 
  sorry

end twin_brothers_age_l2401_240108


namespace opposite_of_2021_l2401_240189

theorem opposite_of_2021 : -(2021) = -2021 := 
sorry

end opposite_of_2021_l2401_240189


namespace winning_candidate_votes_percentage_l2401_240104

theorem winning_candidate_votes_percentage (majority : ℕ) (total_votes : ℕ) (winning_percentage : ℚ) :
  majority = 174 ∧ total_votes = 435 ∧ winning_percentage = 70 → 
  ∃ P : ℚ, (P / 100) * total_votes - ((100 - P) / 100) * total_votes = majority ∧ P = 70 :=
by
  sorry

end winning_candidate_votes_percentage_l2401_240104


namespace cost_of_toast_l2401_240168

theorem cost_of_toast (egg_cost : ℕ) (toast_cost : ℕ)
  (dale_toasts : ℕ) (dale_eggs : ℕ)
  (andrew_toasts : ℕ) (andrew_eggs : ℕ)
  (total_cost : ℕ)
  (h1 : egg_cost = 3)
  (h2 : dale_toasts = 2)
  (h3 : dale_eggs = 2)
  (h4 : andrew_toasts = 1)
  (h5 : andrew_eggs = 2)
  (h6 : 2 * toast_cost + dale_eggs * egg_cost 
        + andrew_toasts * toast_cost + andrew_eggs * egg_cost = total_cost) :
  total_cost = 15 → toast_cost = 1 :=
by
  -- Proof not needed
  sorry

end cost_of_toast_l2401_240168


namespace convert_deg_to_min_compare_negatives_l2401_240188

theorem convert_deg_to_min : (0.3 : ℝ) * 60 = 18 :=
by sorry

theorem compare_negatives : -2 > -3 :=
by sorry

end convert_deg_to_min_compare_negatives_l2401_240188


namespace pedro_more_squares_l2401_240186

theorem pedro_more_squares
  (jesus_squares : ℕ)
  (linden_squares : ℕ)
  (pedro_squares : ℕ)
  (jesus_linden_combined : jesus_squares + linden_squares = 135)
  (pedro_total : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 :=
by
  sorry

end pedro_more_squares_l2401_240186


namespace ai_eq_i_l2401_240199

namespace Problem

def gcd (m n : ℕ) : ℕ := Nat.gcd m n

def sequence_satisfies (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → gcd (a i) (a j) = gcd i j

theorem ai_eq_i (a : ℕ → ℕ) (h : sequence_satisfies a) : ∀ i : ℕ, a i = i :=
by
  sorry

end Problem

end ai_eq_i_l2401_240199


namespace no_solution_system_l2401_240100

theorem no_solution_system (v : ℝ) :
  (∀ x y z : ℝ, ¬(x + y + z = v ∧ x + v * y + z = v ∧ x + y + v^2 * z = v^2)) ↔ (v = -1) :=
  sorry

end no_solution_system_l2401_240100


namespace smallest_possible_value_abs_sum_l2401_240152

theorem smallest_possible_value_abs_sum :
  ∃ x : ℝ, (∀ y : ℝ, abs (y + 3) + abs (y + 5) + abs (y + 7) ≥ abs (x + 3) + abs (x + 5) + abs (x + 7))
  ∧ (abs (x + 3) + abs (x + 5) + abs (x + 7) = 4) := by
  sorry

end smallest_possible_value_abs_sum_l2401_240152


namespace equivalent_single_increase_l2401_240106

-- Defining the initial price of the mobile
variable (P : ℝ)
-- Condition stating the price after a 40% increase
def increased_price := 1.40 * P
-- Condition stating the new price after a further 15% decrease
def final_price := 0.85 * increased_price P

-- The mathematically equivalent statement to prove
theorem equivalent_single_increase:
  final_price P = 1.19 * P :=
sorry

end equivalent_single_increase_l2401_240106


namespace sequence_infinite_divisibility_l2401_240158

theorem sequence_infinite_divisibility :
  ∃ (u : ℕ → ℤ), (∀ n, u (n + 2) = u (n + 1) ^ 2 - u n) ∧ u 1 = 39 ∧ u 2 = 45 ∧ (∀ N, ∃ k ≥ N, 1986 ∣ u k) := 
by
  sorry

end sequence_infinite_divisibility_l2401_240158


namespace crackers_eaten_l2401_240143

-- Define the number of packs and their respective number of crackers
def num_packs_8 : ℕ := 5
def num_packs_10 : ℕ := 10
def num_packs_12 : ℕ := 7
def num_packs_15 : ℕ := 3

def crackers_per_pack_8 : ℕ := 8
def crackers_per_pack_10 : ℕ := 10
def crackers_per_pack_12 : ℕ := 12
def crackers_per_pack_15 : ℕ := 15

-- Calculate the total number of animal crackers
def total_crackers : ℕ :=
  (num_packs_8 * crackers_per_pack_8) +
  (num_packs_10 * crackers_per_pack_10) +
  (num_packs_12 * crackers_per_pack_12) +
  (num_packs_15 * crackers_per_pack_15)

-- Define the number of students who didn't eat their crackers and the respective number of crackers per pack
def num_students_not_eaten : ℕ := 4
def different_crackers_not_eaten : List ℕ := [8, 10, 12, 15]

-- Calculate the total number of crackers not eaten by adding those packs.
def total_crackers_not_eaten : ℕ := different_crackers_not_eaten.sum

-- Theorem to prove the total number of crackers eaten.
theorem crackers_eaten : total_crackers - total_crackers_not_eaten = 224 :=
by
  -- Total crackers: 269
  -- Subtract crackers not eaten: 8 + 10 + 12 + 15 = 45
  -- Therefore: 269 - 45 = 224
  sorry

end crackers_eaten_l2401_240143


namespace product_of_repeating_decimal_l2401_240132

noncomputable def repeating_decimal := 1357 / 9999
def product_with_7 (x : ℚ) := 7 * x

theorem product_of_repeating_decimal :
  product_with_7 repeating_decimal = 9499 / 9999 :=
by sorry

end product_of_repeating_decimal_l2401_240132


namespace smallest_angle_terminal_side_l2401_240194

theorem smallest_angle_terminal_side (θ : ℝ) (H : θ = 2011) :
  ∃ φ : ℝ, 0 ≤ φ ∧ φ < 360 ∧ (∃ k : ℤ, φ = θ - 360 * k) ∧ φ = 211 :=
by
  sorry

end smallest_angle_terminal_side_l2401_240194


namespace squirrels_acorns_l2401_240151

theorem squirrels_acorns (x : ℕ) : 
    (5 * (x - 15) = 575) → 
    x = 130 := 
by 
  intros h
  sorry

end squirrels_acorns_l2401_240151


namespace compute_expression_l2401_240120

theorem compute_expression : (3 + 7)^3 + 2 * (3^2 + 7^2) = 1116 := by
  sorry

end compute_expression_l2401_240120


namespace jogging_distance_apart_l2401_240157

theorem jogging_distance_apart
  (alice_speed : ℝ)
  (bob_speed : ℝ)
  (time_in_minutes : ℝ)
  (distance_apart : ℝ)
  (h1 : alice_speed = 1 / 12)
  (h2 : bob_speed = 3 / 40)
  (h3 : time_in_minutes = 120)
  (h4 : distance_apart = alice_speed * time_in_minutes + bob_speed * time_in_minutes) :
  distance_apart = 19 := by
  sorry

end jogging_distance_apart_l2401_240157


namespace difference_between_numbers_l2401_240147

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 24365) (h2 : a % 5 = 0) (h3 : (a / 10) = 2 * b) : a - b = 19931 :=
by sorry

end difference_between_numbers_l2401_240147


namespace sum_q_p_evaluations_l2401_240125

def p (x : ℝ) : ℝ := |x^2 - 4|
def q (x : ℝ) : ℝ := -|x|

theorem sum_q_p_evaluations : 
  q (p (-3)) + q (p (-2)) + q (p (-1)) + q (p (0)) + q (p (1)) + q (p (2)) + q (p (3)) = -20 := 
by 
  sorry

end sum_q_p_evaluations_l2401_240125


namespace average_page_count_l2401_240153

theorem average_page_count 
  (n1 n2 n3 n4 : ℕ)
  (p1 p2 p3 p4 total_students : ℕ)
  (h1 : n1 = 8)
  (h2 : p1 = 3)
  (h3 : n2 = 10)
  (h4 : p2 = 5)
  (h5 : n3 = 7)
  (h6 : p3 = 2)
  (h7 : n4 = 5)
  (h8 : p4 = 4)
  (h9 : total_students = 30) :
  (n1 * p1 + n2 * p2 + n3 * p3 + n4 * p4) / total_students = 36 / 10 := 
sorry

end average_page_count_l2401_240153


namespace probability_A_B_C_adjacent_l2401_240179

theorem probability_A_B_C_adjacent (students : Fin 5 → Prop) (A B C : Fin 5) :
  (students A ∧ students B ∧ students C) →
  (∃ n m : ℕ, n = 48 ∧ m = 12 ∧ m / n = (1 : ℚ) / 4) :=
by
  sorry

end probability_A_B_C_adjacent_l2401_240179


namespace total_amount_spent_by_jim_is_50_l2401_240163

-- Definitions for conditions
def cost_per_gallon_nc : ℝ := 2.00  -- Cost per gallon in North Carolina
def gallons_nc : ℕ := 10  -- Gallons bought in North Carolina
def additional_cost_per_gallon_va : ℝ := 1.00  -- Additional cost per gallon in Virginia
def gallons_va : ℕ := 10  -- Gallons bought in Virginia

-- Definition for total cost in North Carolina
def total_cost_nc : ℝ := gallons_nc * cost_per_gallon_nc

-- Definition for cost per gallon in Virginia
def cost_per_gallon_va : ℝ := cost_per_gallon_nc + additional_cost_per_gallon_va

-- Definition for total cost in Virginia
def total_cost_va : ℝ := gallons_va * cost_per_gallon_va

-- Definition for total amount spent
def total_spent : ℝ := total_cost_nc + total_cost_va

-- Theorem to prove
theorem total_amount_spent_by_jim_is_50 : total_spent = 50.00 :=
by
  -- Place proof here
  sorry

end total_amount_spent_by_jim_is_50_l2401_240163


namespace jo_bob_pulled_chain_first_time_l2401_240112

/-- Given the conditions of the balloon ride, prove that Jo-Bob pulled the chain
    for the first time for 15 minutes. --/
theorem jo_bob_pulled_chain_first_time (x : ℕ) : 
  (50 * x - 100 + 750 = 1400) → (x = 15) :=
by
  intro h
  sorry

end jo_bob_pulled_chain_first_time_l2401_240112


namespace albert_snakes_count_l2401_240127

noncomputable def garden_snake_length : ℝ := 10.0
noncomputable def boa_ratio : ℝ := 1 / 7.0
noncomputable def boa_length : ℝ := 1.428571429

theorem albert_snakes_count : 
  garden_snake_length = 10.0 ∧ 
  boa_ratio = 1 / 7.0 ∧ 
  boa_length = 1.428571429 → 
  2 = 2 :=
by
  intro h
  sorry   -- Proof will go here

end albert_snakes_count_l2401_240127


namespace math_problem_l2401_240172

open Real

variables {a b c d e f : ℝ}

theorem math_problem 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (hcond : abs (sqrt (a * b) - sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) :=
sorry

end math_problem_l2401_240172


namespace remainder_x14_minus_1_div_x_plus_1_l2401_240102

-- Define the polynomial f(x) = x^14 - 1
def f (x : ℝ) := x^14 - 1

-- Statement to prove that the remainder when f(x) is divided by x + 1 is 0
theorem remainder_x14_minus_1_div_x_plus_1 : f (-1) = 0 :=
by
  -- This is where the proof would go, but for now, we will just use sorry
  sorry

end remainder_x14_minus_1_div_x_plus_1_l2401_240102


namespace average_condition_l2401_240155

theorem average_condition (x : ℝ) :
  (1275 + x) / 51 = 80 * x → x = 1275 / 4079 :=
by
  sorry

end average_condition_l2401_240155


namespace friends_recycled_pounds_l2401_240128

theorem friends_recycled_pounds (total_points chloe_points each_points pounds_per_point : ℕ)
  (h1 : each_points = pounds_per_point / 6)
  (h2 : total_points = 5)
  (h3 : chloe_points = pounds_per_point / 6)
  (h4 : pounds_per_point = 28) 
  (h5 : total_points - chloe_points = 1) :
  pounds_per_point = 6 :=
by
  sorry

end friends_recycled_pounds_l2401_240128


namespace temperature_difference_is_correct_l2401_240107

def highest_temperature : ℤ := -9
def lowest_temperature : ℤ := -22
def temperature_difference : ℤ := highest_temperature - lowest_temperature

theorem temperature_difference_is_correct :
  temperature_difference = 13 := by
  -- We need to prove this statement is correct
  sorry

end temperature_difference_is_correct_l2401_240107


namespace committee_count_l2401_240144

theorem committee_count :
  let total_owners := 30
  let not_willing := 3
  let eligible_owners := total_owners - not_willing
  let committee_size := 5
  eligible_owners.choose committee_size = 65780 := by
  let total_owners := 30
  let not_willing := 3
  let eligible_owners := total_owners - not_willing
  let committee_size := 5
  have lean_theorem : eligible_owners.choose committee_size = 65780 := sorry
  exact lean_theorem

end committee_count_l2401_240144


namespace probability_each_mailbox_has_at_least_one_letter_l2401_240166

noncomputable def probability_mailbox (total_letters : ℕ) (mailboxes : ℕ) : ℚ := 
  let total_ways := mailboxes ^ total_letters
  let favorable_ways := Nat.choose total_letters (mailboxes - 1) * (mailboxes - 1).factorial
  favorable_ways / total_ways

theorem probability_each_mailbox_has_at_least_one_letter :
  probability_mailbox 3 2 = 3 / 4 := by
  sorry

end probability_each_mailbox_has_at_least_one_letter_l2401_240166


namespace gcd_a_b_eq_1023_l2401_240103

def a : ℕ := 2^1010 - 1
def b : ℕ := 2^1000 - 1

theorem gcd_a_b_eq_1023 : Nat.gcd a b = 1023 := 
by
  sorry

end gcd_a_b_eq_1023_l2401_240103


namespace triangle_ABC_perimeter_l2401_240146

noncomputable def triangle_perimeter (A B C D : Type) (AD BC AC AB : ℝ) : ℝ :=
  AD + BC + AC + AB

theorem triangle_ABC_perimeter (A B C D : Type) (AD BC : ℝ) (cos_BDC : ℝ) (angle_sum : ℝ) (AC : ℝ) (AB : ℝ) :
  AD = 3 → BC = 2 → cos_BDC = 13 / 20 → angle_sum = 180 → 
  (triangle_perimeter A B C D AD BC AC AB = 11) :=
by
  sorry

end triangle_ABC_perimeter_l2401_240146


namespace cos_beta_of_acute_angles_l2401_240101

theorem cos_beta_of_acute_angles (α β : ℝ) (hαβ : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2)
  (hcosα : Real.cos α = Real.sqrt 5 / 5)
  (hsin_alpha_minus_beta : Real.sin (α - β) = 3 * Real.sqrt 10 / 10) :
  Real.cos β = 7 * Real.sqrt 2 / 10 :=
sorry

end cos_beta_of_acute_angles_l2401_240101


namespace students_chose_greek_food_l2401_240118
  
theorem students_chose_greek_food (total_students : ℕ) (percentage_greek : ℝ) (h1 : total_students = 200) (h2 : percentage_greek = 0.5) :
  (percentage_greek * total_students : ℝ) = 100 :=
by
  rw [h1, h2]
  norm_num
  sorry

end students_chose_greek_food_l2401_240118


namespace rhombus_diagonal_length_l2401_240133

theorem rhombus_diagonal_length (d1 d2 : ℝ) (A : ℝ) (h1 : d2 = 17) (h2 : A = 127.5) 
  (h3 : A = (d1 * d2) / 2) : d1 = 15 := 
by 
  -- Definitions
  sorry

end rhombus_diagonal_length_l2401_240133


namespace mark_more_than_kate_by_100_l2401_240121

variable (Pat Kate Mark : ℕ)
axiom total_hours : Pat + Kate + Mark = 180
axiom pat_twice_as_kate : Pat = 2 * Kate
axiom pat_third_of_mark : Pat = Mark / 3

theorem mark_more_than_kate_by_100 : Mark - Kate = 100 :=
by
  sorry

end mark_more_than_kate_by_100_l2401_240121


namespace finite_decimal_fractions_l2401_240116

theorem finite_decimal_fractions (a b c d : ℕ) (n : ℕ) 
  (h1 : n = 2^a * 5^b)
  (h2 : n + 1 = 2^c * 5^d) :
  n = 1 ∨ n = 4 :=
by
  sorry

end finite_decimal_fractions_l2401_240116


namespace ab_bc_cd_da_leq_1_over_4_l2401_240169

theorem ab_bc_cd_da_leq_1_over_4 (a b c d : ℝ) (h : a + b + c + d = 1) : 
  a * b + b * c + c * d + d * a ≤ 1 / 4 := 
sorry

end ab_bc_cd_da_leq_1_over_4_l2401_240169


namespace sin_cos_sum_l2401_240177

theorem sin_cos_sum (α : ℝ) (h : ∃ (c : ℝ), Real.sin α = -1 / c ∧ Real.cos α = 2 / c ∧ c = Real.sqrt 5) :
  Real.sin α + Real.cos α = Real.sqrt 5 / 5 :=
by sorry

end sin_cos_sum_l2401_240177


namespace average_speed_home_l2401_240117

theorem average_speed_home
  (s_to_retreat : ℝ)
  (d_to_retreat : ℝ)
  (total_round_trip_time : ℝ)
  (t_retreat : d_to_retreat / s_to_retreat = 6)
  (t_total : d_to_retreat / s_to_retreat + 4 = total_round_trip_time) :
  (d_to_retreat / 4 = 75) :=
by
  sorry

end average_speed_home_l2401_240117


namespace only_one_of_A_B_qualifies_at_least_one_qualifies_l2401_240195

-- Define the probabilities
def P_A_written : ℚ := 2/3
def P_B_written : ℚ := 1/2
def P_C_written : ℚ := 3/4

def P_A_interview : ℚ := 1/2
def P_B_interview : ℚ := 2/3
def P_C_interview : ℚ := 1/3

-- Calculate the overall probabilities for each student qualifying
def P_A_qualifies : ℚ := P_A_written * P_A_interview
def P_B_qualifies : ℚ := P_B_written * P_B_interview
def P_C_qualifies : ℚ := P_C_written * P_C_interview

-- Part 1: Probability that only one of A or B qualifies
theorem only_one_of_A_B_qualifies :
  P_A_qualifies * (1 - P_B_qualifies) + (1 - P_A_qualifies) * P_B_qualifies = 4/9 :=
by sorry

-- Part 2: Probability that at least one of A, B, or C qualifies
theorem at_least_one_qualifies :
  1 - (1 - P_A_qualifies) * (1 - P_B_qualifies) * (1 - P_C_qualifies) = 2/3 :=
by sorry

end only_one_of_A_B_qualifies_at_least_one_qualifies_l2401_240195


namespace negation_of_universal_l2401_240198

open Classical

theorem negation_of_universal (P : ∀ x : ℤ, x^3 < 1) : ∃ x : ℤ, x^3 ≥ 1 :=
by sorry

end negation_of_universal_l2401_240198


namespace complement_A_in_U_l2401_240156

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {1, 5, 7}

theorem complement_A_in_U : (U \ A) = {3, 9} := 
by sorry

end complement_A_in_U_l2401_240156


namespace inequality_does_not_hold_l2401_240187

theorem inequality_does_not_hold (x y : ℝ) (h : x > y) : ¬ (-3 * x > -3 * y) :=
by {
  sorry
}

end inequality_does_not_hold_l2401_240187


namespace rectangle_area_inscribed_circle_l2401_240181

theorem rectangle_area_inscribed_circle (r : ℝ) (h : r = 7) (ratio : ℝ) (hratio : ratio = 3) : 
  (2 * r) * (ratio * (2 * r)) = 588 :=
by
  rw [h, hratio]
  sorry

end rectangle_area_inscribed_circle_l2401_240181


namespace no_solution_ineq_l2401_240183

theorem no_solution_ineq (m : ℝ) : 
  (∀ x : ℝ, x - m ≥ 0 → ¬(0.5 * x + 0.5 < 2)) → m ≥ 3 :=
by
  sorry

end no_solution_ineq_l2401_240183


namespace fraction_transformation_correct_l2401_240185

theorem fraction_transformation_correct
  {a b : ℝ} (hb : b ≠ 0) : 
  (2 * a) / (2 * b) = a / b := by
  sorry

end fraction_transformation_correct_l2401_240185


namespace intersection_point_of_lines_l2401_240105

theorem intersection_point_of_lines : 
  ∃ x y : ℝ, (3 * x + 4 * y - 2 = 0) ∧ (2 * x + y + 2 = 0) ∧ (x = -2) ∧ (y = 2) := 
by 
  sorry

end intersection_point_of_lines_l2401_240105
