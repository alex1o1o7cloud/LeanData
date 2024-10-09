import Mathlib

namespace four_digit_number_condition_l990_99019

theorem four_digit_number_condition (x n : ℕ) (h1 : n = 2000 + x) (h2 : 10 * x + 2 = 2 * n + 66) : n = 2508 :=
sorry

end four_digit_number_condition_l990_99019


namespace sum_of_squares_l990_99057

theorem sum_of_squares (x y z w a b c d : ℝ) (h1: x * y = a) (h2: x * z = b) (h3: y * z = c) (h4: x * w = d) :
  x^2 + y^2 + z^2 + w^2 = (ab + bd + da)^2 / abd := 
by
  sorry

end sum_of_squares_l990_99057


namespace sum_of_fractions_equals_16_l990_99075

def list_of_fractions : List (ℚ) := [
  2 / 10,
  4 / 10,
  6 / 10,
  8 / 10,
  10 / 10,
  15 / 10,
  20 / 10,
  25 / 10,
  30 / 10,
  40 / 10
]

theorem sum_of_fractions_equals_16 : list_of_fractions.sum = 16 := by
  sorry

end sum_of_fractions_equals_16_l990_99075


namespace percentage_of_Luccas_balls_are_basketballs_l990_99093

-- Defining the variables and their conditions 
variables (P : ℝ) (Lucca_Balls : ℕ := 100) (Lucien_Balls : ℕ := 200)
variable (Total_Basketballs : ℕ := 50)

-- Condition that Lucien has 20% basketballs
def Lucien_Basketballs := (20 / 100) * Lucien_Balls

-- We need to prove that percentage of Lucca's balls that are basketballs is 10%
theorem percentage_of_Luccas_balls_are_basketballs :
  (P / 100) * Lucca_Balls + Lucien_Basketballs = Total_Basketballs → P = 10 :=
by
  sorry

end percentage_of_Luccas_balls_are_basketballs_l990_99093


namespace evaluate_expression_l990_99090

theorem evaluate_expression :
  (2 / 10 + 3 / 100 + 5 / 1000 + 7 / 10000)^2 = 0.05555649 :=
by
  sorry

end evaluate_expression_l990_99090


namespace ball_bounces_to_C_l990_99029

/--
On a rectangular table with dimensions 9 cm in length and 7 cm in width, a small ball is shot from point A at a 45-degree angle. Upon reaching point E, it bounces off at a 45-degree angle and continues to roll forward. Throughout its motion, the ball bounces off the table edges at a 45-degree angle each time. Prove that, starting from point A, the ball first reaches point C after exactly 14 bounces.
-/
theorem ball_bounces_to_C (length width : ℝ) (angle : ℝ) (bounce_angle : ℝ) :
  length = 9 ∧ width = 7 ∧ angle = 45 ∧ bounce_angle = 45 → bounces_to_C = 14 :=
by
  intros
  sorry

end ball_bounces_to_C_l990_99029


namespace isosceles_obtuse_triangle_angle_correct_l990_99014

noncomputable def isosceles_obtuse_triangle_smallest_angle (A B C : ℝ) (h1 : A = 1.3 * 90) (h2 : B = C) (h3 : A + B + C = 180) : ℝ :=
  (180 - A) / 2

theorem isosceles_obtuse_triangle_angle_correct 
  (A B C : ℝ)
  (h1 : A = 1.3 * 90)
  (h2 : B = C)
  (h3 : A + B + C = 180) :
  isosceles_obtuse_triangle_smallest_angle A B C h1 h2 h3 = 31.5 :=
sorry

end isosceles_obtuse_triangle_angle_correct_l990_99014


namespace average_rainfall_l990_99020

theorem average_rainfall (r d h : ℕ) (rainfall_eq : r = 450) (days_eq : d = 30) (hours_eq : h = 24) :
  r / (d * h) = 25 / 16 := 
  by 
    -- Insert appropriate proof here
    sorry

end average_rainfall_l990_99020


namespace perimeter_of_original_rectangle_l990_99068

theorem perimeter_of_original_rectangle
  (s : ℕ)
  (h1 : 4 * s = 24)
  (l w : ℕ)
  (h2 : l = 3 * s)
  (h3 : w = s) :
  2 * (l + w) = 48 :=
by
  sorry

end perimeter_of_original_rectangle_l990_99068


namespace assignment_plan_count_l990_99035

noncomputable def number_of_assignment_plans : ℕ :=
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
  let tasks := ["translation", "tour guide", "etiquette", "driver"]
  let v1 := ["Xiao Zhang", "Xiao Zhao"]
  let v2 := ["Xiao Li", "Xiao Luo", "Xiao Wang"]
  -- Condition: Xiao Zhang and Xiao Zhao can only take positions for translation and tour guide
  -- Calculate the number of ways to assign based on the given conditions
  -- 36 is the total number of assignment plans
  36

theorem assignment_plan_count :
  number_of_assignment_plans = 36 :=
  sorry

end assignment_plan_count_l990_99035


namespace sum_a4_a5_a6_l990_99052

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions
axiom h1 : a 1 = 2
axiom h2 : a 3 = -10

-- Definition of arithmetic sequence
axiom h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d

-- Proof problem statement
theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = -66 :=
by
  sorry

end sum_a4_a5_a6_l990_99052


namespace incorrect_statement_B_l990_99031

noncomputable def y (x : ℝ) : ℝ := 2 / x 

theorem incorrect_statement_B :
  ¬ ∀ x > 0, ∀ y1 y2 : ℝ, x < y1 → y1 < y2 → y x < y y2 := sorry

end incorrect_statement_B_l990_99031


namespace product_of_two_numbers_l990_99008

theorem product_of_two_numbers (a b : ℝ)
  (h1 : a + b = 8 * (a - b))
  (h2 : a * b = 30 * (a - b)) :
  a * b = 400 / 7 :=
by
  sorry

end product_of_two_numbers_l990_99008


namespace functional_eq_app_only_solutions_l990_99080

noncomputable def f : Real → Real := sorry

theorem functional_eq_app (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) :
  f (Finset.univ.sum fun i => (x i)^2) = Finset.univ.sum fun i => (f (x i))^2 :=
sorry

theorem only_solutions (f : ℝ → ℝ) (hf : ∀ n : ℕ, ∀ x : Fin n → ℝ, (∀ i, 0 ≤ x i) → f (Finset.univ.sum fun i => (x i)^2) = Finset.univ.sum fun i => (f (x i))^2) :
  f = (fun x => 0) ∨ f = (fun x => x) :=
sorry

end functional_eq_app_only_solutions_l990_99080


namespace min_value_of_quadratic_l990_99054

theorem min_value_of_quadratic (m : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2 * x + m) 
  (min_val : ∀ x ≥ 2, f x ≥ -2) : m = -2 := 
by
  sorry

end min_value_of_quadratic_l990_99054


namespace perfect_square_iff_n_eq_one_l990_99039

theorem perfect_square_iff_n_eq_one (n : ℕ) : ∃ m : ℕ, n^2 + 3 * n = m^2 ↔ n = 1 := by
  sorry

end perfect_square_iff_n_eq_one_l990_99039


namespace exists_nat_lt_100_two_different_squares_l990_99071

theorem exists_nat_lt_100_two_different_squares :
  ∃ n : ℕ, n < 100 ∧ 
    ∃ a b c d : ℕ, a^2 + b^2 = n ∧ c^2 + d^2 = n ∧ (a ≠ c ∨ b ≠ d) ∧ a ≠ b ∧ c ≠ d :=
by
  sorry

end exists_nat_lt_100_two_different_squares_l990_99071


namespace three_pow_zero_eq_one_l990_99022

theorem three_pow_zero_eq_one : 3^0 = 1 :=
by {
  -- Proof would go here
  sorry
}

end three_pow_zero_eq_one_l990_99022


namespace inequality_correct_l990_99098

variable {a b : ℝ}

theorem inequality_correct (h₁ : a < 1) (h₂ : b > 1) : ab < a + b :=
sorry

end inequality_correct_l990_99098


namespace total_distinct_symbols_l990_99016

def numSequences (n : ℕ) : ℕ := 3^n

theorem total_distinct_symbols :
  numSequences 1 + numSequences 2 + numSequences 3 + numSequences 4 = 120 :=
by
  sorry

end total_distinct_symbols_l990_99016


namespace sum_of_digits_base2_310_l990_99077

-- We define what it means to convert a number to binary and sum its digits.
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (Nat.digits 2 n).sum

-- The main statement of the problem.
theorem sum_of_digits_base2_310 :
  sum_of_binary_digits 310 = 5 :=
by
  sorry

end sum_of_digits_base2_310_l990_99077


namespace village_population_l990_99030

theorem village_population (P : ℝ) (h : 0.8 * P = 64000) : P = 80000 :=
sorry

end village_population_l990_99030


namespace range_of_m_l990_99092

def p (m : ℝ) : Prop := m > 3
def q (m : ℝ) : Prop := m > (1 / 4)

theorem range_of_m (m : ℝ) (h1 : ¬p m) (h2 : p m ∨ q m) : (1 / 4) < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l990_99092


namespace initial_bushes_count_l990_99086

theorem initial_bushes_count (n : ℕ) (h : 2 * (27 * n - 26) + 26 = 190 + 26) : n = 8 :=
by
  sorry

end initial_bushes_count_l990_99086


namespace ellipse_foci_coordinates_l990_99005

/-- Define the parameters for the ellipse. -/
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 169 = 1

/-- Prove the coordinates of the foci of the given ellipse. -/
theorem ellipse_foci_coordinates :
  (∀ (x y : ℝ), ellipse_eq x y → False) →
  ∃ (c : ℝ), c = 12 ∧ 
  ((0, c) = (0, 12) ∧ (0, -c) = (0, -12)) := 
by
  sorry

end ellipse_foci_coordinates_l990_99005


namespace trigonometric_identity_l990_99017

theorem trigonometric_identity (α : ℝ) (h : Real.tan (α + π / 4) = -3) :
  2 * Real.cos (2 * α) + 3 * Real.sin (2 * α) - Real.sin α ^ 2 = 2 / 5 :=
by
  sorry

end trigonometric_identity_l990_99017


namespace train_takes_longer_l990_99088

-- Definitions for the conditions
def train_speed : ℝ := 48
def ship_speed : ℝ := 60
def distance : ℝ := 480

-- Theorem statement for the proof
theorem train_takes_longer : (distance / train_speed) - (distance / ship_speed) = 2 := by
  sorry

end train_takes_longer_l990_99088


namespace obtuse_right_triangle_cannot_exist_l990_99069

-- Definitions of various types of triangles

def is_acute (θ : ℕ) : Prop := θ < 90
def is_right (θ : ℕ) : Prop := θ = 90
def is_obtuse (θ : ℕ) : Prop := θ > 90

def is_isosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ a = c
def is_scalene (a b c : ℕ) : Prop := ¬ (a = b) ∧ ¬ (b = c) ∧ ¬ (a = c)
def is_triangle (a b c : ℕ) : Prop := a + b + c = 180

-- Propositions for the types of triangles given in the problem

def acute_isosceles_triangle (a b : ℕ) : Prop :=
  is_triangle a a (180 - 2 * a) ∧ is_acute a ∧ is_isosceles a a (180 - 2 * a)

def isosceles_right_triangle (a : ℕ) : Prop :=
  is_triangle a a 90 ∧ is_right 90 ∧ is_isosceles a a 90

def obtuse_right_triangle (a b : ℕ) : Prop :=
  is_triangle a 90 (180 - 90 - a) ∧ is_right 90 ∧ is_obtuse (180 - 90 - a)

def scalene_right_triangle (a b : ℕ) : Prop :=
  is_triangle a b 90 ∧ is_right 90 ∧ is_scalene a b 90

def scalene_obtuse_triangle (a b : ℕ) : Prop :=
  is_triangle a b (180 - a - b) ∧ is_obtuse (180 - a - b) ∧ is_scalene a b (180 - a - b)

-- The final theorem stating that obtuse right triangle cannot exist

theorem obtuse_right_triangle_cannot_exist (a b : ℕ) :
  ¬ exists (a b : ℕ), obtuse_right_triangle a b :=
by
  sorry

end obtuse_right_triangle_cannot_exist_l990_99069


namespace minimum_value_of_f_l990_99095

noncomputable def f (x y z : ℝ) : ℝ := x^2 + 2 * y^2 + 3 * z^2 + 2 * x * y + 4 * y * z + 2 * z * x - 6 * x - 10 * y - 12 * z

theorem minimum_value_of_f : ∃ x y z : ℝ, f x y z = -14 :=
by
  sorry

end minimum_value_of_f_l990_99095


namespace gambler_received_max_2240_l990_99063

def largest_amount_received_back (x y l : ℕ) : ℕ :=
  if 2 * l + 2 = 14 ∨ 2 * l - 2 = 14 then 
    let lost_value_1 := (6 * 100 + 8 * 20)
    let lost_value_2 := (8 * 100 + 6 * 20)
    max (3000 - lost_value_1) (3000 - lost_value_2)
  else 0

theorem gambler_received_max_2240 {x y : ℕ} (hx : 20 * x + 100 * y = 3000)
  (hl : ∃ l : ℕ, (l + (l + 2) = 14 ∨ l + (l - 2) = 14)) :
  largest_amount_received_back x y 6 = 2240 ∧ largest_amount_received_back x y 8 = 2080 := by
  sorry

end gambler_received_max_2240_l990_99063


namespace convex_polygon_diagonals_l990_99006

theorem convex_polygon_diagonals (n : ℕ) (h_n : n = 25) : 
  (n * (n - 3)) / 2 = 275 :=
by
  sorry

end convex_polygon_diagonals_l990_99006


namespace remainder_of_power_division_l990_99050

theorem remainder_of_power_division :
  (2^222 + 222) % (2^111 + 2^56 + 1) = 218 :=
by sorry

end remainder_of_power_division_l990_99050


namespace shekar_biology_marks_l990_99078

variable (M S SS E A : ℕ)

theorem shekar_biology_marks (hM : M = 76) (hS : S = 65) (hSS : SS = 82) (hE : E = 67) (hA : A = 77) :
  let total_marks := M + S + SS + E
  let total_average_marks := A * 5
  let biology_marks := total_average_marks - total_marks
  biology_marks = 95 :=
by
  sorry

end shekar_biology_marks_l990_99078


namespace total_money_l990_99041

-- Define the problem statement
theorem total_money (n : ℕ) (hn : 3 * n = 75) : (n * 1 + n * 5 + n * 10) = 400 :=
by sorry

end total_money_l990_99041


namespace arithmetic_sequence_l990_99010

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) (d a1 : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (n : ℕ) (d a1 : ℤ) : ℤ := n * a1 + (n * (n - 1)) / 2 * d

-- Given conditions
theorem arithmetic_sequence (n : ℕ) (d a1 : ℤ) (S3 : ℤ) (h1 : a1 = 10) (h2 : S_n 3 d a1 = 24) :
  (a_n n d a1 = 12 - 2 * n) ∧ (S_n n (-2) 12 = -n^2 + 11 * n) ∧ (∀ k, S_n k (-2) 12 ≤ 30) :=
by
  sorry

end arithmetic_sequence_l990_99010


namespace roots_quadratic_l990_99064

theorem roots_quadratic (a b : ℝ) (h : ∀ x : ℝ, x^2 - 7 * x + 7 = 0 → (x = a) ∨ (x = b)) :
  a^2 + b^2 = 35 :=
sorry

end roots_quadratic_l990_99064


namespace rhombus_area_l990_99012

-- Definition of a rhombus with given conditions
structure Rhombus where
  side : ℝ
  d1 : ℝ
  d2 : ℝ

noncomputable def Rhombus.area (r : Rhombus) : ℝ :=
  (r.d1 * r.d2) / 2

noncomputable example : Rhombus :=
{ side := 20,
  d1 := 16,
  d2 := 8 * Real.sqrt 21 }

theorem rhombus_area : 
  let r : Rhombus := { side := 20, d1 := 16, d2 := 8 * Real.sqrt 21 }
  Rhombus.area r = 64 * Real.sqrt 21 :=
by
  let r : Rhombus := { side := 20, d1 := 16, d2 := 8 * Real.sqrt 21 }
  sorry

end rhombus_area_l990_99012


namespace fourth_grade_students_l990_99047

theorem fourth_grade_students (initial_students left_students new_students final_students : ℕ) 
    (h1 : initial_students = 33) 
    (h2 : left_students = 18) 
    (h3 : new_students = 14) 
    (h4 : final_students = initial_students - left_students + new_students) :
    final_students = 29 := 
by 
    sorry

end fourth_grade_students_l990_99047


namespace arithmetic_sum_first_11_terms_l990_99056

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

variable (a : ℕ → ℝ)

theorem arithmetic_sum_first_11_terms (h_arith_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_sum_condition : a 2 + a 6 + a 10 = 6) :
  sum_first_n_terms a 11 = 22 :=
sorry

end arithmetic_sum_first_11_terms_l990_99056


namespace four_p_plus_one_composite_l990_99003

theorem four_p_plus_one_composite (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_five : p ≥ 5) (h2p_plus1_prime : Nat.Prime (2 * p + 1)) : ¬ Nat.Prime (4 * p + 1) :=
sorry

end four_p_plus_one_composite_l990_99003


namespace discriminant_formula_l990_99028

def discriminant_cubic_eq (x1 x2 x3 p q : ℝ) : ℝ :=
  (x1 - x2)^2 * (x2 - x3)^2 * (x3 - x1)^2

theorem discriminant_formula (x1 x2 x3 p q : ℝ)
  (h1 : x1 + x2 + x3 = 0)
  (h2 : x1 * x2 + x1 * x3 + x2 * x3 = p)
  (h3 : x1 * x2 * x3 = -q) :
  discriminant_cubic_eq x1 x2 x3 p q = -4 * p^3 - 27 * q^2 :=
by sorry

end discriminant_formula_l990_99028


namespace ratio_of_areas_l990_99042

theorem ratio_of_areas 
  (a b c : ℕ) (d e f : ℕ)
  (hABC : a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2)
  (hDEF : d = 8 ∧ e = 15 ∧ f = 17 ∧ d^2 + e^2 = f^2) :
  (1/2 * a * b) / (1/2 * d * e) = 2 / 5 :=
by
  sorry

end ratio_of_areas_l990_99042


namespace lemons_count_l990_99074

def total_fruits (num_baskets : ℕ) (total : ℕ) : Prop := num_baskets = 5 ∧ total = 58
def basket_contents (basket : ℕ → ℕ) : Prop := 
  basket 1 = 18 ∧ -- mangoes
  basket 2 = 10 ∧ -- pears
  basket 3 = 12 ∧ -- pawpaws
  (∀ i, (i = 4 ∨ i = 5) → basket i = (basket 4 + basket 5) / 2)

theorem lemons_count (num_baskets : ℕ) (total : ℕ) (basket : ℕ → ℕ) : 
  total_fruits num_baskets total ∧ basket_contents basket → basket 5 = 9 :=
by
  sorry

end lemons_count_l990_99074


namespace pen_count_l990_99048

-- Define the conditions
def total_pens := 140
def difference := 20

-- Define the quantities to be proven
def ballpoint_pens := (total_pens - difference) / 2
def fountain_pens := total_pens - ballpoint_pens

-- The theorem to be proved
theorem pen_count :
  ballpoint_pens = 60 ∧ fountain_pens = 80 :=
by
  -- Proof omitted
  sorry

end pen_count_l990_99048


namespace Inequality_Solution_Set_Range_of_c_l990_99037

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x

noncomputable def g (x : ℝ) : ℝ := -(((-x)^2) + 2 * (-x))

theorem Inequality_Solution_Set (x : ℝ) :
  (g x ≥ f x - |x - 1|) ↔ (-1 ≤ x ∧ x ≤ 1/2) :=
by
  sorry

theorem Range_of_c (c : ℝ) :
  (∀ x : ℝ, g x + c ≤ f x - |x - 1|) ↔ (c ≤ -9/8) :=
by
  sorry

end Inequality_Solution_Set_Range_of_c_l990_99037


namespace well_depth_is_correct_l990_99033

noncomputable def depth_of_well : ℝ :=
  122500

theorem well_depth_is_correct (d t1 : ℝ) : 
  t1 = Real.sqrt (d / 20) ∧ 
  (d / 1100) + t1 = 10 →
  d = depth_of_well := 
by
  sorry

end well_depth_is_correct_l990_99033


namespace part1_part2_l990_99081

noncomputable def A (a : ℝ) := { x : ℝ | x^2 - a * x + a^2 - 19 = 0 }
def B := { x : ℝ | x^2 - 5 * x + 6 = 0 }
def C := { x : ℝ | x^2 + 2 * x - 8 = 0 }

-- Proof Problem 1: Prove that if A ∩ B ≠ ∅ and A ∩ C = ∅, then a = -2
theorem part1 (a : ℝ) (h1 : (A a ∩ B) ≠ ∅) (h2 : (A a ∩ C) = ∅) : a = -2 :=
sorry

-- Proof Problem 2: Prove that if A ∩ B = A ∩ C ≠ ∅, then a = -3
theorem part2 (a : ℝ) (h1 : (A a ∩ B = A a ∩ C) ∧ (A a ∩ B) ≠ ∅) : a = -3 :=
sorry

end part1_part2_l990_99081


namespace value_of_expression_l990_99096

theorem value_of_expression (a b c d m : ℝ) (h1 : a = -b) (h2 : a ≠ 0) (h3 : c * d = 1) (h4 : |m| = 3) :
  m^2 - (-1) + |a + b| - c * d * m = 7 ∨ m^2 - (-1) + |a + b| - c * d * m = 13 :=
by
  sorry

end value_of_expression_l990_99096


namespace centroid_coordinates_of_tetrahedron_l990_99082

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (O A B C G G1 : V) (OG1_subdivides : G -ᵥ O = 3 • (G1 -ᵥ G))
variable (A_centroid : G1 -ᵥ O = (1/3 : ℝ) • (A -ᵥ O + B -ᵥ O + C -ᵥ O))

-- The main proof problem
theorem centroid_coordinates_of_tetrahedron :
  G -ᵥ O = (1/4 : ℝ) • (A -ᵥ O + B -ᵥ O + C -ᵥ O) :=
sorry

end centroid_coordinates_of_tetrahedron_l990_99082


namespace find_interest_rate_l990_99046

theorem find_interest_rate (initial_investment : ℚ) (duration_months : ℚ) 
  (first_rate : ℚ) (final_value : ℚ) (s : ℚ) :
  initial_investment = 15000 →
  duration_months = 9 →
  first_rate = 0.09 →
  final_value = 17218.50 →
  (∃ s : ℚ, 16012.50 * (1 + (s * 0.75) / 100) = final_value) →
  s = 10 := 
by
  sorry

end find_interest_rate_l990_99046


namespace oil_bill_for_january_l990_99024

-- Definitions and conditions
def ratio_F_J (F J : ℝ) : Prop := F / J = 3 / 2
def ratio_F_M (F M : ℝ) : Prop := F / M = 4 / 5
def ratio_F_J_modified (F J : ℝ) : Prop := (F + 20) / J = 5 / 3
def ratio_F_M_modified (F M : ℝ) : Prop := (F + 20) / M = 2 / 3

-- The main statement to prove
theorem oil_bill_for_january (J F M : ℝ) 
  (h1 : ratio_F_J F J)
  (h2 : ratio_F_M F M)
  (h3 : ratio_F_J_modified F J)
  (h4 : ratio_F_M_modified F M) :
  J = 120 :=
sorry

end oil_bill_for_january_l990_99024


namespace cookie_baking_time_l990_99059

theorem cookie_baking_time 
  (total_time : ℕ) 
  (white_icing_time: ℕ)
  (chocolate_icing_time: ℕ) 
  (total_icing_time : white_icing_time + chocolate_icing_time = 60)
  (total_cooking_time : total_time = 120):

  (total_time - (white_icing_time + chocolate_icing_time) = 60) :=
by
  sorry

end cookie_baking_time_l990_99059


namespace find_diameter_C_l990_99004

noncomputable def diameter_of_circle_C (diameter_of_D : ℝ) (ratio_shaded_to_C : ℝ) : ℝ :=
  let radius_D := diameter_of_D / 2
  let radius_C := radius_D / (2 * Real.sqrt ratio_shaded_to_C)
  2 * radius_C

theorem find_diameter_C :
  let diameter_D := 20
  let ratio_shaded_area_to_C := 7
  diameter_of_circle_C diameter_D ratio_shaded_area_to_C = 5 * Real.sqrt 2 :=
by
  -- The proof is omitted.
  sorry

end find_diameter_C_l990_99004


namespace no_real_solutions_cubic_eq_l990_99023

theorem no_real_solutions_cubic_eq : ∀ x : ℝ, ¬ (∃ (y : ℝ), y = x^(1/3) ∧ y = 15 / (6 - y)) :=
by
  intro x
  intro hexist
  obtain ⟨y, hy1, hy2⟩ := hexist
  have h_cubic : y * (6 - y) = 15 := by sorry -- from y = 15 / (6 - y)
  have h_quad : y^2 - 6 * y + 15 = 0 := by sorry -- after expanding y(6 - y) = 15
  sorry -- remainder to show no real solution due to negative discriminant

end no_real_solutions_cubic_eq_l990_99023


namespace part1_solution_part2_solution_l990_99002

-- Definition for part 1
noncomputable def f_part1 (x : ℝ) := abs (x - 3) + 2 * x

-- Proof statement for part 1
theorem part1_solution (x : ℝ) : (f_part1 x ≥ 3) ↔ (x ≥ 0) :=
by sorry

-- Definition for part 2
noncomputable def f_part2 (x a : ℝ) := abs (x - a) + 2 * x

-- Proof statement for part 2
theorem part2_solution (a : ℝ) : 
  (∀ x, f_part2 x a ≤ 0 ↔ x ≤ -2) → (a = 2 ∨ a = -6) :=
by sorry

end part1_solution_part2_solution_l990_99002


namespace range_of_x_if_p_and_q_range_of_a_if_not_p_sufficient_for_not_q_l990_99015

variable (x a : ℝ)

-- Condition p
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

-- Condition q
def q (x : ℝ) : Prop :=
  (x^2 - x - 6 ≤ 0) ∧ (x^2 + 3*x - 10 > 0)

-- Proof problem for question (1)
theorem range_of_x_if_p_and_q (h1 : p x 1) (h2 : q x) : 2 < x ∧ x < 3 :=
  sorry

-- Proof problem for question (2)
theorem range_of_a_if_not_p_sufficient_for_not_q (h : (¬p x a) → (¬q x)) : 1 < a ∧ a ≤ 2 :=
  sorry

end range_of_x_if_p_and_q_range_of_a_if_not_p_sufficient_for_not_q_l990_99015


namespace number_of_stacks_l990_99013

theorem number_of_stacks (total_coins stacks coins_per_stack : ℕ) (h1 : coins_per_stack = 3) (h2 : total_coins = 15) (h3 : total_coins = stacks * coins_per_stack) : stacks = 5 :=
by
  sorry

end number_of_stacks_l990_99013


namespace calc_value_l990_99072

theorem calc_value : (3000 * (3000 ^ 2999) * 2 = 2 * 3000 ^ 3000) := 
by
  sorry

end calc_value_l990_99072


namespace quadratic_non_real_roots_l990_99001

variable (b : ℝ)

theorem quadratic_non_real_roots : (b^2 - 64 < 0) → (-8 < b ∧ b < 8) :=
by
  sorry

end quadratic_non_real_roots_l990_99001


namespace geometric_sequence_ratio_l990_99007

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def condition_1 (n : ℕ) : Prop := S n = 2 * a n - 2

theorem geometric_sequence_ratio (h : ∀ n, condition_1 a S n) : (a 8 / a 6 = 4) :=
sorry

end geometric_sequence_ratio_l990_99007


namespace inequality_proof_l990_99038

theorem inequality_proof (n : ℕ) (a : ℝ) (h₀ : n > 1) (h₁ : 0 < a) (h₂ : a < 1) : 
  1 + a < (1 + a / n) ^ n ∧ (1 + a / n) ^ n < (1 + a / (n + 1)) ^ (n + 1) := 
sorry

end inequality_proof_l990_99038


namespace zoey_holidays_l990_99000

def visits_per_year (visits_per_month : ℕ) (months_per_year : ℕ) : ℕ :=
  visits_per_month * months_per_year

def visits_every_two_months (months_per_year : ℕ) : ℕ :=
  months_per_year / 2

def visits_every_four_months (visits_per_period : ℕ) (periods_per_year : ℕ) : ℕ :=
  visits_per_period * periods_per_year

theorem zoey_holidays (visits_per_month_first : ℕ) 
                      (months_per_year : ℕ) 
                      (visits_per_period_third : ℕ) 
                      (periods_per_year : ℕ) : 
  visits_per_year visits_per_month_first months_per_year 
  + visits_every_two_months months_per_year 
  + visits_every_four_months visits_per_period_third periods_per_year = 39 := 
  by 
  sorry

end zoey_holidays_l990_99000


namespace two_workers_two_hours_holes_l990_99021

theorem two_workers_two_hours_holes
    (workers1: ℝ) (holes1: ℝ) (hours1: ℝ)
    (workers2: ℝ) (hours2: ℝ)
    (h1: workers1 = 1.5)
    (h2: holes1 = 1.5)
    (h3: hours1 = 1.5)
    (h4: workers2 = 2)
    (h5: hours2 = 2)
    : (workers2 * (holes1 / (workers1 * hours1)) * hours2 = 8 / 3) := 
by {
   -- To be filled with proof, currently a placeholder.
  sorry
}

end two_workers_two_hours_holes_l990_99021


namespace combined_cost_l990_99084

theorem combined_cost (wallet_cost : ℕ) (purse_cost : ℕ)
    (h_wallet_cost : wallet_cost = 22)
    (h_purse_cost : purse_cost = 4 * wallet_cost - 3) :
    wallet_cost + purse_cost = 107 :=
by
  rw [h_wallet_cost, h_purse_cost]
  norm_num
  sorry

end combined_cost_l990_99084


namespace initial_sum_of_money_l990_99060

theorem initial_sum_of_money (A2 A7 : ℝ) (H1 : A2 = 520) (H2 : A7 = 820) :
  ∃ P : ℝ, P = 400 :=
by
  -- Proof starts here
  sorry

end initial_sum_of_money_l990_99060


namespace positive_difference_even_odd_sums_l990_99087

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l990_99087


namespace remainder_pow_2023_l990_99032

theorem remainder_pow_2023 (a b : ℕ) (h : b = 2023) : (3 ^ b) % 11 = 5 :=
by
  sorry

end remainder_pow_2023_l990_99032


namespace series_converges_to_three_fourths_l990_99073

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l990_99073


namespace distinct_students_l990_99018

theorem distinct_students 
  (students_euler : ℕ) (students_gauss : ℕ) (students_fibonacci : ℕ) (overlap_euler_gauss : ℕ)
  (h_euler : students_euler = 15) 
  (h_gauss : students_gauss = 10) 
  (h_fibonacci : students_fibonacci = 12) 
  (h_overlap : overlap_euler_gauss = 3) 
  : students_euler + students_gauss + students_fibonacci - overlap_euler_gauss = 34 :=
by
  sorry

end distinct_students_l990_99018


namespace min_value_problem1_min_value_problem2_l990_99053

-- Problem 1: Prove that the minimum value of the function y = x + 4/(x + 1) + 6 is 9 given x > -1
theorem min_value_problem1 (x : ℝ) (h : x > -1) : (x + 4 / (x + 1) + 6) ≥ 9 := 
sorry

-- Problem 2: Prove that the minimum value of the function y = (x^2 + 8) / (x - 1) is 8 given x > 1
theorem min_value_problem2 (x : ℝ) (h : x > 1) : ((x^2 + 8) / (x - 1)) ≥ 8 :=
sorry

end min_value_problem1_min_value_problem2_l990_99053


namespace min_n_for_circuit_l990_99089

theorem min_n_for_circuit
  (n : ℕ) 
  (p_success_component : ℝ)
  (p_work_circuit : ℝ) 
  (h1 : p_success_component = 0.5)
  (h2 : p_work_circuit = 1 - p_success_component ^ n) 
  (h3 : p_work_circuit ≥ 0.95) :
  n ≥ 5 := 
sorry

end min_n_for_circuit_l990_99089


namespace unique_number_not_in_range_l990_99049

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : g a b c d 13 = 13)
  (h2 : g a b c d 31 = 31)
  (h3 : ∀ x, x ≠ -d / c → g a b c d (g a b c d x) = x) :
  ∀ y, ∃! x, g a b c d x = y :=
by {
  sorry
}

end unique_number_not_in_range_l990_99049


namespace cheddar_cheese_slices_l990_99091

-- Define the conditions
def cheddar_slices (C : ℕ) := ∃ (packages : ℕ), packages * C = 84
def swiss_slices := 28
def randy_bought_same_slices (C : ℕ) := swiss_slices = 28 ∧ 84 = 84

-- Lean theorem statement to prove the number of slices per package of cheddar cheese equals 28.
theorem cheddar_cheese_slices {C : ℕ} (h1 : cheddar_slices C) (h2 : randy_bought_same_slices C) : C = 28 :=
sorry

end cheddar_cheese_slices_l990_99091


namespace infinite_series_value_l990_99079

noncomputable def infinite_series : ℝ :=
  ∑' n, if n ≥ 2 then (n^4 + 5 * n^2 + 8 * n + 8) / (2^(n + 1) * (n^4 + 4)) else 0

theorem infinite_series_value :
  infinite_series = 3 / 10 :=
by
  sorry

end infinite_series_value_l990_99079


namespace correct_operation_l990_99043

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l990_99043


namespace find_intersection_complement_B_find_A_minus_B_find_A_minus_A_minus_B_l990_99009

def U := Set ℝ
def A : Set ℝ := {x | x > 4}
def B : Set ℝ := {x | -6 < x ∧ x < 6}

theorem find_intersection (x : ℝ) : x ∈ A ∧ x ∈ B ↔ 4 < x ∧ x < 6 :=
by
  sorry

theorem complement_B (x : ℝ) : x ∉ B ↔ x ≥ 6 ∨ x ≤ -6 :=
by
  sorry

def A_minus_B : Set ℝ := {x | x ∈ A ∧ x ∉ B}

theorem find_A_minus_B (x : ℝ) : x ∈ A_minus_B ↔ x ≥ 6 :=
by
  sorry

theorem find_A_minus_A_minus_B (x : ℝ) : x ∈ (A \ A_minus_B) ↔ 4 < x ∧ x < 6 :=
by
  sorry

end find_intersection_complement_B_find_A_minus_B_find_A_minus_A_minus_B_l990_99009


namespace probability_at_least_one_white_ball_l990_99070

/-
  We define the conditions:
  - num_white: the number of white balls,
  - num_red: the number of red balls,
  - total_balls: the total number of balls,
  - num_drawn: the number of balls drawn.
-/
def num_white : ℕ := 5
def num_red : ℕ := 4
def total_balls : ℕ := num_white + num_red
def num_drawn : ℕ := 3

/-
  Given the conditions, we need to prove that the probability of drawing at least one white ball is 20/21.
-/
theorem probability_at_least_one_white_ball :
  (1 : ℚ) - (4 / 84) = 20 / 21 :=
by
  sorry

end probability_at_least_one_white_ball_l990_99070


namespace value_of_expression_l990_99055

theorem value_of_expression : (7^2 - 6^2)^4 = 28561 :=
by sorry

end value_of_expression_l990_99055


namespace sum_of_five_consecutive_even_integers_l990_99083

theorem sum_of_five_consecutive_even_integers (a : ℤ) 
  (h : a + (a + 4) = 144) : a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 370 := by
  sorry

end sum_of_five_consecutive_even_integers_l990_99083


namespace monotonic_intervals_range_of_a_min_value_of_c_l990_99062

noncomputable def f (a c x : ℝ) : ℝ :=
  a * Real.log x + (x - c) * abs (x - c)

-- 1. Monotonic intervals
theorem monotonic_intervals (a c : ℝ) (ha : a = -3 / 4) (hc : c = 1 / 4) :
  ((∀ x, 0 < x ∧ x < 3 / 4 → f a c x > f a c (x - 1)) ∧ (∀ x, 3 / 4 < x → f a c x > f a c (x - 1))) :=
sorry

-- 2. Range of values for a
theorem range_of_a (a c : ℝ) (hc : c = a / 2 + 1) (h : ∀ x > c, f a c x ≥ 1 / 4) :
  -2 < a ∧ a ≤ -1 :=
sorry

-- 3. Minimum value of c
theorem min_value_of_c (a c x1 x2 : ℝ) (hx1 : x1 = Real.sqrt (-a / 2)) (hx2 : x2 = c)
  (h_tangents_perpendicular : f a c x1 * f a c x2 = -1) :
  c = 3 * Real.sqrt 3 / 2 :=
sorry

end monotonic_intervals_range_of_a_min_value_of_c_l990_99062


namespace square_table_production_l990_99085

theorem square_table_production (x y : ℝ) :
  x + y = 5 ∧ 50 * x * 4 = 300 * y → 
  x = 3 ∧ y = 2 ∧ 50 * x = 150 :=
by
  sorry

end square_table_production_l990_99085


namespace central_angle_of_sector_l990_99061

theorem central_angle_of_sector (r l : ℝ) (h1 : l + 2 * r = 4) (h2 : (1 / 2) * l * r = 1) : l / r = 2 :=
by
  -- The proof should be provided here
  sorry

end central_angle_of_sector_l990_99061


namespace initial_men_l990_99051

variable (x : ℕ)

-- Conditions
def condition1 (x : ℕ) : Prop :=
  -- The hostel had provisions for x men for 28 days.
  true

def condition2 (x : ℕ) : Prop :=
  -- If 50 men left, the food would last for 35 days for the remaining x - 50 men.
  (x - 50) * 35 = x * 28

-- Theorem to prove
theorem initial_men (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 250 :=
by
  sorry

end initial_men_l990_99051


namespace jamie_total_balls_after_buying_l990_99065

theorem jamie_total_balls_after_buying (red_balls : ℕ) (blue_balls : ℕ) (yellow_balls : ℕ) (lost_red_balls : ℕ) (final_red_balls : ℕ) (total_balls : ℕ)
  (h1 : red_balls = 16)
  (h2 : blue_balls = 2 * red_balls)
  (h3 : lost_red_balls = 6)
  (h4 : final_red_balls = red_balls - lost_red_balls)
  (h5 : yellow_balls = 32)
  (h6 : total_balls = final_red_balls + blue_balls + yellow_balls) :
  total_balls = 74 := by
    sorry

end jamie_total_balls_after_buying_l990_99065


namespace total_pennies_donated_l990_99026

theorem total_pennies_donated:
  let cassandra_pennies := 5000
  let james_pennies := cassandra_pennies - 276
  let stephanie_pennies := 2 * james_pennies
  cassandra_pennies + james_pennies + stephanie_pennies = 19172 :=
by
  sorry

end total_pennies_donated_l990_99026


namespace infinite_geometric_series_correct_l990_99011

noncomputable def infinite_geometric_series_sum : ℚ :=
  let a : ℚ := 5 / 3
  let r : ℚ := -9 / 20
  a / (1 - r)

theorem infinite_geometric_series_correct : infinite_geometric_series_sum = 100 / 87 := 
by
  sorry

end infinite_geometric_series_correct_l990_99011


namespace sin_690_eq_neg_half_l990_99027

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end sin_690_eq_neg_half_l990_99027


namespace four_digit_number_properties_l990_99066

theorem four_digit_number_properties :
  ∃ (a b c d : ℕ), 
    a + b + c + d = 8 ∧ 
    a = 3 * b ∧ 
    d = 4 * c ∧ 
    1000 * a + 100 * b + 10 * c + d = 6200 :=
by
  sorry

end four_digit_number_properties_l990_99066


namespace inequality_solution_l990_99058

theorem inequality_solution (x : ℝ) : (3 * x - 1) / (x - 2) > 0 ↔ x < 1 / 3 ∨ x > 2 :=
sorry

end inequality_solution_l990_99058


namespace fraction_correct_l990_99044

theorem fraction_correct (x : ℚ) (h : (5 / 6) * 576 = x * 576 + 300) : x = 5 / 16 := 
sorry

end fraction_correct_l990_99044


namespace incorrect_inequality_given_conditions_l990_99045

variable {a b x y : ℝ}

theorem incorrect_inequality_given_conditions 
  (h1 : a > b) (h2 : x > y) : ¬ (|a| * x > |a| * y) :=
sorry

end incorrect_inequality_given_conditions_l990_99045


namespace number_of_subsets_of_P_l990_99025

noncomputable def P : Set ℝ := {x | x^2 - 2*x + 1 = 0}

theorem number_of_subsets_of_P : ∃ (n : ℕ), n = 2 ∧ ∀ S : Set ℝ, S ⊆ P → S = ∅ ∨ S = {1} := by
  sorry

end number_of_subsets_of_P_l990_99025


namespace min_ratio_ax_l990_99067

theorem min_ratio_ax (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) 
: y^2 - 1 = a^2 * (x^2 - 1) → ∃ (k : ℕ), k = 2 ∧ (a = k * x) := 
sorry

end min_ratio_ax_l990_99067


namespace ratio_of_numbers_l990_99094

theorem ratio_of_numbers (A B : ℕ) (HCF_AB : Nat.gcd A B = 3) (LCM_AB : Nat.lcm A B = 36) : 
  A / B = 3 / 4 :=
sorry

end ratio_of_numbers_l990_99094


namespace Bhupathi_amount_l990_99034

variable (A B : ℝ)

theorem Bhupathi_amount
  (h1 : A + B = 1210)
  (h2 : (4 / 15) * A = (2 / 5) * B) :
  B = 484 := by
  sorry

end Bhupathi_amount_l990_99034


namespace perimeter_shaded_region_l990_99076

noncomputable def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def arc_length_per_circle (C : ℝ) : ℝ := C / 4

theorem perimeter_shaded_region (C : ℝ) (hC : C = 48) : 
  3 * arc_length_per_circle C = 36 := by
  sorry

end perimeter_shaded_region_l990_99076


namespace gcd_2015_15_l990_99040

theorem gcd_2015_15 : Nat.gcd 2015 15 = 5 :=
by
  have h1 : 2015 = 15 * 134 + 5 := by rfl
  have h2 : 15 = 5 * 3 := by rfl
  sorry

end gcd_2015_15_l990_99040


namespace non_talking_birds_count_l990_99097

def total_birds : ℕ := 77
def talking_birds : ℕ := 64

theorem non_talking_birds_count : total_birds - talking_birds = 13 := by
  sorry

end non_talking_birds_count_l990_99097


namespace Adam_total_cost_l990_99036

theorem Adam_total_cost :
  let laptop1_cost := 500
  let laptop2_base_cost := 3 * laptop1_cost
  let discount := 0.15 * laptop2_base_cost
  let laptop2_cost := laptop2_base_cost - discount
  let external_hard_drive := 80
  let mouse := 20
  let software1 := 120
  let software2 := 2 * 120
  let insurance1 := 0.10 * laptop1_cost
  let insurance2 := 0.10 * laptop2_cost
  let total_cost1 := laptop1_cost + external_hard_drive + mouse + software1 + insurance1
  let total_cost2 := laptop2_cost + external_hard_drive + mouse + software2 + insurance2
  total_cost1 + total_cost2 = 2512.5 :=
by
  sorry

end Adam_total_cost_l990_99036


namespace magician_earning_correct_l990_99099

def magician_earning (initial_decks : ℕ) (remaining_decks : ℕ) (price_per_deck : ℕ) : ℕ :=
  (initial_decks - remaining_decks) * price_per_deck

theorem magician_earning_correct :
  magician_earning 5 3 2 = 4 :=
by
  sorry

end magician_earning_correct_l990_99099
