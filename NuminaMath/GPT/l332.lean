import Mathlib

namespace NUMINAMATH_GPT_Nina_has_16dollars65_l332_33287

-- Definitions based on given conditions
variables (W M : ℝ)

-- Condition 1: Nina has exactly enough money to purchase 5 widgets
def condition1 : Prop := 5 * W = M

-- Condition 2: If the cost of each widget were reduced by $1.25, Nina would have exactly enough money to purchase 8 widgets
def condition2 : Prop := 8 * (W - 1.25) = M

-- Statement: Proving the amount of money Nina has is $16.65
theorem Nina_has_16dollars65 (h1 : condition1 W M) (h2 : condition2 W M) : M = 16.65 :=
sorry

end NUMINAMATH_GPT_Nina_has_16dollars65_l332_33287


namespace NUMINAMATH_GPT_expand_expression_l332_33216

theorem expand_expression (x y : ℝ) : 24 * (3 * x - 4 * y + 6) = 72 * x - 96 * y + 144 := 
by
  sorry

end NUMINAMATH_GPT_expand_expression_l332_33216


namespace NUMINAMATH_GPT_valid_range_and_difference_l332_33222

/- Assume side lengths as given expressions -/
def BC (x : ℝ) : ℝ := x + 11
def AC (x : ℝ) : ℝ := x + 6
def AB (x : ℝ) : ℝ := 3 * x + 2

/- Define the inequalities representing the triangle inequalities and largest angle condition -/
def triangle_inequality1 (x : ℝ) : Prop := AB x + AC x > BC x
def triangle_inequality2 (x : ℝ) : Prop := AB x + BC x > AC x
def triangle_inequality3 (x : ℝ) : Prop := AC x + BC x > AB x
def largest_angle_condition (x : ℝ) : Prop := BC x > AB x

/- Define the combined condition for x, ensuring all relevant conditions are met -/
def valid_x_range (x : ℝ) : Prop :=
  1 < x ∧ x < 4.5 ∧ triangle_inequality1 x ∧ triangle_inequality2 x ∧ triangle_inequality3 x ∧ largest_angle_condition x

/- Compute n - m for the interval (m, n) where x lies -/
def n_minus_m : ℝ :=
  4.5 - 1

/- Main theorem stating the final result -/
theorem valid_range_and_difference :
  (∃ x : ℝ, valid_x_range x) ∧ (n_minus_m = 7 / 2) :=
by
  sorry

end NUMINAMATH_GPT_valid_range_and_difference_l332_33222


namespace NUMINAMATH_GPT_solve_equation_l332_33273

theorem solve_equation (x : ℝ) :
  3 * x + 6 = abs (-20 + x^2) →
  x = (3 + Real.sqrt 113) / 2 ∨ x = (3 - Real.sqrt 113) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l332_33273


namespace NUMINAMATH_GPT_wooden_block_even_blue_faces_l332_33226

theorem wooden_block_even_blue_faces :
  let length := 6
  let width := 6
  let height := 2
  let total_cubes := length * width * height
  let corners := 8
  let edges_not_corners := 24
  let faces_not_edges := 24
  let interior := 16
  let even_blue_faces := edges_not_corners + interior
  total_cubes = 72 →
  even_blue_faces = 40 :=
by
  sorry

end NUMINAMATH_GPT_wooden_block_even_blue_faces_l332_33226


namespace NUMINAMATH_GPT_expand_expression_l332_33286

theorem expand_expression (x : ℝ) : 24 * (3 * x + 4 - 2) = 72 * x + 48 :=
by 
  sorry

end NUMINAMATH_GPT_expand_expression_l332_33286


namespace NUMINAMATH_GPT_geometric_sequence_value_of_b_l332_33201

theorem geometric_sequence_value_of_b : 
  ∃ b : ℝ, 180 * (b / 180) = b ∧ (b / 180) * b = 64 / 25 ∧ b > 0 ∧ b = 21.6 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_value_of_b_l332_33201


namespace NUMINAMATH_GPT_Randy_drew_pictures_l332_33202

variable (P Q R: ℕ)

def Peter_drew_pictures (P : ℕ) : Prop := P = 8
def Quincy_drew_pictures (Q P : ℕ) : Prop := Q = P + 20
def Total_drawing (R P Q : ℕ) : Prop := R + P + Q = 41

theorem Randy_drew_pictures
  (P_eq : Peter_drew_pictures P)
  (Q_eq : Quincy_drew_pictures Q P)
  (Total_eq : Total_drawing R P Q) :
  R = 5 :=
by 
  sorry

end NUMINAMATH_GPT_Randy_drew_pictures_l332_33202


namespace NUMINAMATH_GPT_solve_system_solve_equation_l332_33266

-- 1. System of Equations
theorem solve_system :
  ∀ (x y : ℝ), (x + 2 * y = 9) ∧ (3 * x - 2 * y = 3) → (x = 3) ∧ (y = 3) :=
by sorry

-- 2. Single Equation
theorem solve_equation :
  ∀ (x : ℝ), (2 - x) / (x - 3) + 3 = 2 / (3 - x) → x = 5 / 2 :=
by sorry

end NUMINAMATH_GPT_solve_system_solve_equation_l332_33266


namespace NUMINAMATH_GPT_lambs_goats_solution_l332_33227

theorem lambs_goats_solution : ∃ l g : ℕ, l > 0 ∧ g > 0 ∧ 30 * l + 32 * g = 1200 ∧ l = 24 ∧ g = 15 :=
by
  existsi 24
  existsi 15
  repeat { split }
  sorry

end NUMINAMATH_GPT_lambs_goats_solution_l332_33227


namespace NUMINAMATH_GPT_g_property_l332_33265

theorem g_property (g : ℝ → ℝ) (h : ∀ x y : ℝ, g x * g y - g (x * y) = 2 * x + 2 * y) :
  let n := 2
  let s := 14 / 3
  n = 2 ∧ s = 14 / 3 ∧ n * s = 28 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_g_property_l332_33265


namespace NUMINAMATH_GPT_mrs_sheridan_gave_away_14_cats_l332_33275

def num_initial_cats : ℝ := 17.0
def num_left_cats : ℝ := 3.0
def num_given_away (x : ℝ) : Prop := num_initial_cats - x = num_left_cats

theorem mrs_sheridan_gave_away_14_cats : num_given_away 14.0 :=
by
  sorry

end NUMINAMATH_GPT_mrs_sheridan_gave_away_14_cats_l332_33275


namespace NUMINAMATH_GPT_find_union_of_sets_l332_33247

-- Define the sets A and B in terms of a
def A (a : ℤ) : Set ℤ := { n | n = |a + 1| ∨ n = 3 ∨ n = 5 }
def B (a : ℤ) : Set ℤ := { n | n = 2 * a + 1 ∨ n = a^2 + 2 * a ∨ n = a^2 + 2 * a - 1 }

-- Given condition: A ∩ B = {2, 3}
def condition (a : ℤ) : Prop := A a ∩ B a = {2, 3}

-- The correct answer: A ∪ B = {-5, 2, 3, 5}
theorem find_union_of_sets (a : ℤ) (h : condition a) : A a ∪ B a = {-5, 2, 3, 5} :=
sorry

end NUMINAMATH_GPT_find_union_of_sets_l332_33247


namespace NUMINAMATH_GPT_find_p_l332_33249

/-- Given the points Q(0, 15), A(3, 15), B(15, 0), O(0, 0), and C(0, p).
The area of triangle ABC is given as 45.
We need to prove that p = 11.25. -/
theorem find_p (ABC_area : ℝ) (p : ℝ) (h : ABC_area = 45) :
  p = 11.25 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l332_33249


namespace NUMINAMATH_GPT_fraction_value_l332_33208

theorem fraction_value : (20 * 21) / (2 + 0 + 2 + 1) = 84 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l332_33208


namespace NUMINAMATH_GPT_number_of_students_l332_33228

theorem number_of_students 
  (P S : ℝ)
  (total_cost : ℝ) 
  (percent_free : ℝ) 
  (lunch_cost : ℝ)
  (h1 : percent_free = 0.40)
  (h2 : total_cost = 210)
  (h3 : lunch_cost = 7)
  (h4 : P = 0.60 * S)
  (h5 : P * lunch_cost = total_cost) :
  S = 50 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l332_33228


namespace NUMINAMATH_GPT_coordinates_of_point_P_l332_33244

open Real

def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distance_to_y_axis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem coordinates_of_point_P (P : ℝ × ℝ) 
  (h1 : in_fourth_quadrant P) 
  (h2 : distance_to_x_axis P = 1) 
  (h3 : distance_to_y_axis P = 2) : 
  P = (2, -1) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_point_P_l332_33244


namespace NUMINAMATH_GPT_find_A_plus_B_l332_33229

theorem find_A_plus_B {A B : ℚ} (h : ∀ x : ℚ, 
                     (Bx - 17) / (x^2 - 9 * x + 20) = A / (x - 4) + 5 / (x - 5)) : 
                     A + B = 9 / 5 := sorry

end NUMINAMATH_GPT_find_A_plus_B_l332_33229


namespace NUMINAMATH_GPT_polynomial_g_correct_l332_33297

noncomputable def polynomial_g : Polynomial ℚ := 
  Polynomial.C (-41 / 2) + Polynomial.X * 41 / 2 + Polynomial.X ^ 2

theorem polynomial_g_correct
  (f g : Polynomial ℚ)
  (h1 : f ≠ 0)
  (h2 : g ≠ 0)
  (hx : ∀ x, f.eval (g.eval x) = (Polynomial.eval x f) * (Polynomial.eval x g))
  (h3 : Polynomial.eval 3 g = 50) :
  g = polynomial_g :=
sorry

end NUMINAMATH_GPT_polynomial_g_correct_l332_33297


namespace NUMINAMATH_GPT_total_groups_l332_33237

-- Define the problem conditions
def boys : ℕ := 9
def girls : ℕ := 12

-- Calculate the required combinations
def C (n k: ℕ) : ℕ := n.choose k
def groups_with_two_boys_one_girl : ℕ := C boys 2 * C girls 1
def groups_with_two_girls_one_boy : ℕ := C girls 2 * C boys 1

-- Statement of the theorem to prove
theorem total_groups : groups_with_two_boys_one_girl + groups_with_two_girls_one_boy = 1026 := 
by sorry

end NUMINAMATH_GPT_total_groups_l332_33237


namespace NUMINAMATH_GPT_no_positive_integer_solution_l332_33251

theorem no_positive_integer_solution (p x y : ℕ) (hp : Nat.Prime p) (hp_gt3 : p > 3) 
  (h_p_div_x : p ∣ x) (hx_pos : 0 < x) (hy_pos : 0 < y) : x^2 - 1 ≠ y^p :=
sorry

end NUMINAMATH_GPT_no_positive_integer_solution_l332_33251


namespace NUMINAMATH_GPT_range_of_a_l332_33203

theorem range_of_a (a x y : ℝ) (h1 : x - y = a + 3) (h2 : 2 * x + y = 5 * a) (h3 : x < y) : a < -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l332_33203


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l332_33276

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) :=
  ∃ a_1 d, ∀ n, a (n + 1) = a_1 + n * d

theorem arithmetic_sequence_a5 (a : ℕ → α) (h_seq : is_arithmetic_sequence a) (h_cond : a 1 + a 7 = 12) :
  a 4 = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l332_33276


namespace NUMINAMATH_GPT_problem_proof_l332_33225

open Set

noncomputable def A : Set ℝ := {x | abs (4 * x - 1) < 9}
noncomputable def B : Set ℝ := {x | x / (x + 3) ≥ 0}
noncomputable def complement_A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 5 / 2}
noncomputable def correct_answer : Set ℝ := Iio (-3) ∪ Ici (5 / 2)

theorem problem_proof : (compl A) ∩ B = correct_answer := 
  by
    sorry

end NUMINAMATH_GPT_problem_proof_l332_33225


namespace NUMINAMATH_GPT_unique_function_satisfying_conditions_l332_33236

theorem unique_function_satisfying_conditions (f : ℤ → ℤ) :
  (∀ n : ℤ, f (f n) + f n = 2 * n + 3) → 
  (f 0 = 1) → 
  (∀ n : ℤ, f n = n + 1) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_unique_function_satisfying_conditions_l332_33236


namespace NUMINAMATH_GPT_square_pieces_placement_l332_33211

theorem square_pieces_placement (n : ℕ) (H : n = 8) :
  {m : ℕ // m = 17} :=
sorry

end NUMINAMATH_GPT_square_pieces_placement_l332_33211


namespace NUMINAMATH_GPT_Eric_white_marbles_l332_33263

theorem Eric_white_marbles (total_marbles blue_marbles green_marbles : ℕ) (h1 : total_marbles = 20) (h2 : blue_marbles = 6) (h3 : green_marbles = 2) : 
  total_marbles - (blue_marbles + green_marbles) = 12 := by
  sorry

end NUMINAMATH_GPT_Eric_white_marbles_l332_33263


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l332_33260

-- Definitions of propositions p and q
def p (a b m : ℝ) : Prop := a * m^2 < b * m^2
def q (a b : ℝ) : Prop := a < b

-- Problem statement as a Lean theorem
theorem sufficient_but_not_necessary (a b m : ℝ) : 
  (p a b m → q a b) ∧ (¬ (q a b → p a b m)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l332_33260


namespace NUMINAMATH_GPT_candy_given_away_l332_33224

-- Define the conditions
def pieces_per_student := 2
def number_of_students := 9

-- Define the problem statement as a theorem
theorem candy_given_away : pieces_per_student * number_of_students = 18 := by
  -- This is where the proof would go, but we omit it with sorry.
  sorry

end NUMINAMATH_GPT_candy_given_away_l332_33224


namespace NUMINAMATH_GPT_max_surface_area_of_rectangular_solid_l332_33255

theorem max_surface_area_of_rectangular_solid {r a b c : ℝ} (h_sphere : 4 * π * r^2 = 4 * π)
  (h_diagonal : a^2 + b^2 + c^2 = (2 * r)^2) :
  2 * (a * b + a * c + b * c) ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_max_surface_area_of_rectangular_solid_l332_33255


namespace NUMINAMATH_GPT_inequality_hold_l332_33292

theorem inequality_hold (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - |c| > b - |c| :=
sorry

end NUMINAMATH_GPT_inequality_hold_l332_33292


namespace NUMINAMATH_GPT_percentage_proof_l332_33268

theorem percentage_proof (n : ℝ) (h : 0.3 * 0.4 * n = 24) : 0.4 * 0.3 * n = 24 :=
sorry

end NUMINAMATH_GPT_percentage_proof_l332_33268


namespace NUMINAMATH_GPT_intersection_complement_B_l332_33250

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - 3 * x < 0 }
def B : Set ℝ := { x | abs x > 2 }

-- Complement of B
def complement_B : Set ℝ := { x | x ≥ -2 ∧ x ≤ 2 }

-- Final statement to prove the intersection equals the given set
theorem intersection_complement_B :
  A ∩ complement_B = { x : ℝ | 0 < x ∧ x ≤ 2 } := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_intersection_complement_B_l332_33250


namespace NUMINAMATH_GPT_least_common_multiple_first_ten_l332_33240

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end NUMINAMATH_GPT_least_common_multiple_first_ten_l332_33240


namespace NUMINAMATH_GPT_boat_trip_l332_33258

variable {v v_T : ℝ}

theorem boat_trip (d_total t_total : ℝ) (h1 : d_total = 10) (h2 : t_total = 5) (h3 : 2 / (v - v_T) = 3 / (v + v_T)) :
  v_T = 5 / 12 ∧ (5 / (v - v_T)) = 3 ∧ (5 / (v + v_T)) = 2 :=
by
  have h4 : 1 / (d_total / t_total) = v - v_T := sorry
  have h5 : 1 / (d_total / t_total) = v + v_T := sorry
  have h6 : v = 5 * v_T := sorry
  have h7 : v_T = 5 / 12 := sorry
  have t_upstream : 5 / (v - v_T) = 3 := sorry
  have t_downstream : 5 / (v + v_T) = 2 := sorry
  exact ⟨h7, t_upstream, t_downstream⟩

end NUMINAMATH_GPT_boat_trip_l332_33258


namespace NUMINAMATH_GPT_turtles_remaining_l332_33217

/-- 
In one nest, there are x baby sea turtles, while in the other nest, there are 2x baby sea turtles.
One-fourth of the turtles in the first nest and three-sevenths of the turtles in the second nest
got swept to the sea. Prove the total number of turtles still on the sand is (53/28)x.
-/
theorem turtles_remaining (x : ℕ) (h1 : ℕ := x) (h2 : ℕ := 2 * x) : ((3/4) * x + (8/7) * (2 * x)) = (53/28) * x :=
by
  sorry

end NUMINAMATH_GPT_turtles_remaining_l332_33217


namespace NUMINAMATH_GPT_division_remainder_l332_33213

theorem division_remainder :
  ∃ (R D Q : ℕ), D = 3 * Q ∧ D = 3 * R + 3 ∧ 251 = D * Q + R ∧ R = 8 := by
  sorry

end NUMINAMATH_GPT_division_remainder_l332_33213


namespace NUMINAMATH_GPT_A_intersection_B_eq_intersection_set_l332_33212

def A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
def B : Set ℝ := {x : ℝ | x > 1}
def intersection_set := {x : ℝ | 1 < x ∧ x < 2}

theorem A_intersection_B_eq_intersection_set : A ∩ B = intersection_set := by
  sorry

end NUMINAMATH_GPT_A_intersection_B_eq_intersection_set_l332_33212


namespace NUMINAMATH_GPT_algebraic_expression_value_l332_33243

theorem algebraic_expression_value (a b : ℝ) (h1 : a = 1 + Real.sqrt 2) (h2 : b = Real.sqrt 3) : 
  a^2 + b^2 - 2 * a + 1 = 5 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l332_33243


namespace NUMINAMATH_GPT_problem_1_problem_2_l332_33241

-- Condition for Question 1
def f (x : ℝ) (a : ℝ) := |x - a|

-- Proof Problem for Question 1
theorem problem_1 (a : ℝ) (h : a = 1) : {x : ℝ | f x a > 1/2 * (x + 1)} = {x | x > 3 ∨ x < 1/3} :=
sorry

-- Condition for Question 2
def g (x : ℝ) (a : ℝ) := |x - a| + |x - 2|

-- Proof Problem for Question 2
theorem problem_2 (a : ℝ) : (∃ x : ℝ, g x a ≤ 3) → (-1 ≤ a ∧ a ≤ 5) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l332_33241


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l332_33223

theorem arithmetic_sequence_problem
  (a : ℕ → ℚ)
  (h : a 2 + a 4 + a 9 + a 11 = 32) :
  a 6 + a 7 = 16 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l332_33223


namespace NUMINAMATH_GPT_todd_has_40_left_after_paying_back_l332_33205

def todd_snowcone_problem : Prop :=
  let borrowed := 100
  let repay := 110
  let cost_ingredients := 75
  let snowcones_sold := 200
  let price_per_snowcone := 0.75
  let total_earnings := snowcones_sold * price_per_snowcone
  let remaining_money := total_earnings - repay
  remaining_money = 40

theorem todd_has_40_left_after_paying_back : todd_snowcone_problem :=
by
  -- Add proof here if needed
  sorry

end NUMINAMATH_GPT_todd_has_40_left_after_paying_back_l332_33205


namespace NUMINAMATH_GPT_parabola_directrix_l332_33238

theorem parabola_directrix (x : ℝ) :
  (y = (x^2 - 8 * x + 12) / 16) →
  (∃ y, y = -17/4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parabola_directrix_l332_33238


namespace NUMINAMATH_GPT_bear_weight_gain_l332_33280

theorem bear_weight_gain :
  let total_weight := 1000
  let weight_from_berries := total_weight / 5
  let weight_from_acorns := 2 * weight_from_berries
  let weight_from_salmon := (total_weight - weight_from_berries - weight_from_acorns) / 2
  let weight_from_small_animals := total_weight - (weight_from_berries + weight_from_acorns + weight_from_salmon)
  weight_from_small_animals = 200 :=
by sorry

end NUMINAMATH_GPT_bear_weight_gain_l332_33280


namespace NUMINAMATH_GPT_correct_remainder_l332_33214

-- Define the problem
def count_valid_tilings (n k : Nat) : Nat :=
  Nat.factorial (n + k) / (Nat.factorial n * Nat.factorial k) * (3 ^ (n + k) - 3 * 2 ^ (n + k) + 3)

noncomputable def tiles_mod_1000 : Nat :=
  let pairs := [(8, 0), (6, 1), (4, 2), (2, 3), (0, 4)]
  let M := pairs.foldl (λ acc (nk : Nat × Nat) => acc + count_valid_tilings nk.1 nk.2) 0
  M % 1000

theorem correct_remainder : tiles_mod_1000 = 328 :=
  by sorry

end NUMINAMATH_GPT_correct_remainder_l332_33214


namespace NUMINAMATH_GPT_simplify_parentheses_l332_33274

theorem simplify_parentheses (a b c x y : ℝ) : (3 * a - (2 * a - c) = 3 * a - 2 * a + c) := 
by 
  sorry

end NUMINAMATH_GPT_simplify_parentheses_l332_33274


namespace NUMINAMATH_GPT_find_a_l332_33299

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then 1 - x else a * x

theorem find_a (a : ℝ) : f (-1) a = f 1 a → a = 2 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l332_33299


namespace NUMINAMATH_GPT_total_emails_received_l332_33285

theorem total_emails_received (emails_morning emails_afternoon : ℕ) 
  (h1 : emails_morning = 3) 
  (h2 : emails_afternoon = 5) : 
  emails_morning + emails_afternoon = 8 := 
by 
  sorry

end NUMINAMATH_GPT_total_emails_received_l332_33285


namespace NUMINAMATH_GPT_phil_cards_left_l332_33257

-- Conditions
def cards_per_week : ℕ := 20
def weeks_per_year : ℕ := 52

-- Total number of cards in a year
def total_cards (cards_per_week weeks_per_year : ℕ) : ℕ := cards_per_week * weeks_per_year

-- Number of cards left after losing half in fire
def cards_left (total_cards : ℕ) : ℕ := total_cards / 2

-- Theorem to prove
theorem phil_cards_left (cards_per_week weeks_per_year : ℕ) :
  cards_left (total_cards cards_per_week weeks_per_year) = 520 :=
by
  sorry

end NUMINAMATH_GPT_phil_cards_left_l332_33257


namespace NUMINAMATH_GPT_intended_profit_l332_33269

variables (C P : ℝ)

theorem intended_profit (L S : ℝ) (h1 : L = C * (1 + P)) (h2 : S = 0.90 * L) (h3 : S = 1.17 * C) :
  P = 0.3 + 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_intended_profit_l332_33269


namespace NUMINAMATH_GPT_sum_of_irreducible_fractions_is_integer_iff_same_denominator_l332_33296

theorem sum_of_irreducible_fractions_is_integer_iff_same_denominator
  (a b c d A : ℤ) (h_irred1 : Int.gcd a b = 1) (h_irred2 : Int.gcd c d = 1) (h_sum : (a : ℚ) / b + (c : ℚ) / d = A) :
  b = d := 
by
  sorry

end NUMINAMATH_GPT_sum_of_irreducible_fractions_is_integer_iff_same_denominator_l332_33296


namespace NUMINAMATH_GPT_ratio_female_to_male_l332_33220

theorem ratio_female_to_male
  (a b c : ℕ)
  (ha : a = 60)
  (hb : b = 80)
  (hc : c = 65) :
  f / m = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_female_to_male_l332_33220


namespace NUMINAMATH_GPT_weekly_income_l332_33281

-- Defining the daily catches
def blue_crabs_per_bucket (day : String) : ℕ :=
  match day with
  | "Monday"    => 10
  | "Tuesday"   => 8
  | "Wednesday" => 12
  | "Thursday"  => 6
  | "Friday"    => 14
  | "Saturday"  => 10
  | "Sunday"    => 8
  | _           => 0

def red_crabs_per_bucket (day : String) : ℕ :=
  match day with
  | "Monday"    => 14
  | "Tuesday"   => 16
  | "Wednesday" => 10
  | "Thursday"  => 18
  | "Friday"    => 12
  | "Saturday"  => 10
  | "Sunday"    => 8
  | _           => 0

-- Prices per crab
def price_per_blue_crab : ℕ := 6
def price_per_red_crab : ℕ := 4
def buckets : ℕ := 8

-- Daily income calculation
def daily_income (day : String) : ℕ :=
  let blue_income := (blue_crabs_per_bucket day) * buckets * price_per_blue_crab
  let red_income := (red_crabs_per_bucket day) * buckets * price_per_red_crab
  blue_income + red_income

-- Proving the weekly income is $6080
theorem weekly_income : 
  (daily_income "Monday" +
  daily_income "Tuesday" +
  daily_income "Wednesday" +
  daily_income "Thursday" +
  daily_income "Friday" +
  daily_income "Saturday" +
  daily_income "Sunday") = 6080 :=
by sorry

end NUMINAMATH_GPT_weekly_income_l332_33281


namespace NUMINAMATH_GPT_cleaner_flow_rate_after_second_unclogging_l332_33204

theorem cleaner_flow_rate_after_second_unclogging
  (rate1 rate2 : ℕ) (time1 time2 total_time total_cleaner : ℕ)
  (used_cleaner1 used_cleaner2 : ℕ)
  (final_rate : ℕ)
  (H1 : rate1 = 2)
  (H2 : rate2 = 3)
  (H3 : time1 = 15)
  (H4 : time2 = 10)
  (H5 : total_time = 30)
  (H6 : total_cleaner = 80)
  (H7 : used_cleaner1 = rate1 * time1)
  (H8 : used_cleaner2 = rate2 * time2)
  (H9 : used_cleaner1 + used_cleaner2 ≤ total_cleaner)
  (H10 : final_rate = (total_cleaner - (used_cleaner1 + used_cleaner2)) / (total_time - (time1 + time2))) :
  final_rate = 4 := by
  sorry

end NUMINAMATH_GPT_cleaner_flow_rate_after_second_unclogging_l332_33204


namespace NUMINAMATH_GPT_original_radius_of_cylinder_l332_33242

theorem original_radius_of_cylinder (r z : ℝ) (h : ℝ := 3) :
  z = 3 * π * ((r + 8)^2 - r^2) → z = 8 * π * r^2 → r = 8 :=
by
  intros hz1 hz2
  -- Translate given conditions into their equivalent expressions and equations
  sorry

end NUMINAMATH_GPT_original_radius_of_cylinder_l332_33242


namespace NUMINAMATH_GPT_geometric_sum_eight_terms_l332_33253

noncomputable def geometric_series_sum_8 (a r : ℝ) : ℝ :=
  a * (1 - r^8) / (1 - r)

theorem geometric_sum_eight_terms
  (a r : ℝ) (h_geom_pos : r > 0)
  (h_sum_two : a + a * r = 2)
  (h_sum_eight : a * r^2 + a * r^3 = 8) :
  geometric_series_sum_8 a r = 170 := 
sorry

end NUMINAMATH_GPT_geometric_sum_eight_terms_l332_33253


namespace NUMINAMATH_GPT_points_lie_on_hyperbola_l332_33271

noncomputable
def point_on_hyperbola (t : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ 
    (2 * t * x - 3 * y - 4 * t = 0 ∧ x - 3 * t * y + 4 = 0) }

theorem points_lie_on_hyperbola : 
  ∀ t : ℝ, ∀ x y : ℝ, (2 * t * x - 3 * y - 4 * t = 0 ∧ x - 3 * t * y + 4 = 0) → (x^2 / 16) - (y^2 / 1) = 1 :=
by 
  intro t x y h
  obtain ⟨hx, hy⟩ := h
  sorry

end NUMINAMATH_GPT_points_lie_on_hyperbola_l332_33271


namespace NUMINAMATH_GPT_circle_radius_l332_33298

theorem circle_radius (x y : ℝ) :
  y = (x - 2)^2 ∧ x - 3 = (y + 1)^2 →
  (∃ c d r : ℝ, (c, d) = (3/2, -1/2) ∧ r^2 = 25/4) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l332_33298


namespace NUMINAMATH_GPT_greatest_possible_median_l332_33262

theorem greatest_possible_median : 
  ∀ (k m r s t : ℕ),
    k < m → m < r → r < s → s < t →
    (k + m + r + s + t = 90) →
    (t = 40) →
    (r = 23) :=
by
  intros k m r s t h1 h2 h3 h4 h_sum h_t
  sorry

end NUMINAMATH_GPT_greatest_possible_median_l332_33262


namespace NUMINAMATH_GPT_minimize_material_l332_33219

theorem minimize_material (π V R h : ℝ) (hV : V > 0) (h_cond : π * R^2 * h = V) :
  R = h / 2 :=
sorry

end NUMINAMATH_GPT_minimize_material_l332_33219


namespace NUMINAMATH_GPT_max_min_x_sub_2y_l332_33248

theorem max_min_x_sub_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) : 0 ≤ x - 2*y ∧ x - 2*y ≤ 10 :=
sorry

end NUMINAMATH_GPT_max_min_x_sub_2y_l332_33248


namespace NUMINAMATH_GPT_roots_of_polynomial_l332_33290

-- Define the polynomial
def poly := fun (x : ℝ) => x^3 - 7 * x^2 + 14 * x - 8

-- Define the statement
theorem roots_of_polynomial : (poly 1 = 0) ∧ (poly 2 = 0) ∧ (poly 4 = 0) :=
  by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l332_33290


namespace NUMINAMATH_GPT_find_first_term_and_common_difference_l332_33252

variable (n : ℕ)
variable (a_1 d : ℚ)

-- Definition of the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (n : ℕ) (a_1 d : ℚ) : ℚ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom sum_condition : ∀ (n : ℕ), sum_arithmetic_seq n a_1 d = n^2 / 2

-- Theorem to prove
theorem find_first_term_and_common_difference 
  (a_1 d : ℚ) 
  (sum_condition : ∀ (n : ℕ), sum_arithmetic_seq n a_1 d = n^2 / 2) 
: a_1 = 1/2 ∧ d = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_first_term_and_common_difference_l332_33252


namespace NUMINAMATH_GPT_union_sets_l332_33231

namespace Proof

def setA : Set ℝ := { x : ℝ | x * (x + 1) ≤ 0 }
def setB : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }

theorem union_sets : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
sorry

end Proof

end NUMINAMATH_GPT_union_sets_l332_33231


namespace NUMINAMATH_GPT_eval_expression_l332_33230

theorem eval_expression : 1999^2 - 1998 * 2002 = -3991 := 
by
  sorry

end NUMINAMATH_GPT_eval_expression_l332_33230


namespace NUMINAMATH_GPT_fraction_is_one_fourth_l332_33246

-- Defining the numbers
def num1 : ℕ := 16
def num2 : ℕ := 8

-- Conditions
def difference_correct : Prop := num1 - num2 = 8
def sum_of_numbers : ℕ := num1 + num2
def fraction_of_sum (f : ℚ) : Prop := f * sum_of_numbers = 6

-- Theorem stating the fraction
theorem fraction_is_one_fourth (f : ℚ) (h1 : difference_correct) (h2 : fraction_of_sum f) : f = 1 / 4 :=
by {
  -- This will use the conditions and show that f = 1/4
  sorry
}

end NUMINAMATH_GPT_fraction_is_one_fourth_l332_33246


namespace NUMINAMATH_GPT_point_in_third_quadrant_l332_33294

theorem point_in_third_quadrant (x y : ℤ) (hx : x = -8) (hy : y = -3) : (x < 0) ∧ (y < 0) :=
by
  have hx_neg : x < 0 := by rw [hx]; norm_num
  have hy_neg : y < 0 := by rw [hy]; norm_num
  exact ⟨hx_neg, hy_neg⟩

end NUMINAMATH_GPT_point_in_third_quadrant_l332_33294


namespace NUMINAMATH_GPT_bert_total_stamp_cost_l332_33245

theorem bert_total_stamp_cost :
    let numA := 150
    let numB := 90
    let numC := 60
    let priceA := 2
    let priceB := 3
    let priceC := 5
    let costA := numA * priceA
    let costB := numB * priceB
    let costC := numC * priceC
    let total_cost := costA + costB + costC
    total_cost = 870 := 
by
    sorry

end NUMINAMATH_GPT_bert_total_stamp_cost_l332_33245


namespace NUMINAMATH_GPT_investment_amount_first_rate_l332_33259

theorem investment_amount_first_rate : ∀ (x y : ℝ) (r : ℝ),
  x + y = 15000 → -- Condition 1 (Total investments)
  8200 * r + 6800 * 0.075 = 1023 → -- Condition 2 (Interest yield)
  x = 8200 → -- Condition 3 (Amount invested at first rate)
  x = 8200 := -- Question (How much was invested)
by
  intros x y r h₁ h₂ h₃
  exact h₃

end NUMINAMATH_GPT_investment_amount_first_rate_l332_33259


namespace NUMINAMATH_GPT_find_perpendicular_vector_l332_33232

def vector_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def vector_magnitude_equal (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 ^ 2 + v1.2 ^ 2) = (v2.1 ^ 2 + v2.2 ^ 2)

theorem find_perpendicular_vector (a b : ℝ) :
  ∃ n : ℝ × ℝ, vector_perpendicular (a, b) n ∧ vector_magnitude_equal (a, b) n ∧ n = (b, -a) :=
by
  sorry

end NUMINAMATH_GPT_find_perpendicular_vector_l332_33232


namespace NUMINAMATH_GPT_tom_first_part_speed_l332_33282

theorem tom_first_part_speed 
  (total_distance : ℕ)
  (distance_first_part : ℕ)
  (speed_second_part : ℕ)
  (average_speed : ℕ)
  (total_time : ℕ)
  (distance_remaining : ℕ)
  (T2 : ℕ)
  (v : ℕ) :
  total_distance = 80 →
  distance_first_part = 30 →
  speed_second_part = 50 →
  average_speed = 40 →
  total_time = 2 →
  distance_remaining = 50 →
  T2 = 1 →
  total_time = distance_first_part / v + T2 →
  v = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Here, we need to prove that v = 30 given the above conditions.
  sorry

end NUMINAMATH_GPT_tom_first_part_speed_l332_33282


namespace NUMINAMATH_GPT_box_combination_is_correct_l332_33270

variables (C A S T t u : ℕ)

theorem box_combination_is_correct
    (h1 : 3 * S % t = C)
    (h2 : 2 * A + C = T)
    (h3 : 2 * C + A + u = T) :
  (1000 * C + 100 * A + 10 * S + T = 7252) :=
sorry

end NUMINAMATH_GPT_box_combination_is_correct_l332_33270


namespace NUMINAMATH_GPT_circle_equation_standard_form_l332_33215

theorem circle_equation_standard_form (x y : ℝ) :
  (∃ (center : ℝ × ℝ), center.1 = -1 ∧ center.2 = 2 * center.1 ∧ (center.2 = -2) ∧ (center.1 + 1)^2 + center.2^2 = 4 ∧ (center.1 = -1) ∧ (center.2 = -2)) ->
  (x + 1)^2 + (y + 2)^2 = 4 :=
sorry

end NUMINAMATH_GPT_circle_equation_standard_form_l332_33215


namespace NUMINAMATH_GPT_mean_temperature_l332_33200

def temperatures : List Int := [-8, -3, -3, -6, 2, 4, 1]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : Int) = -2 := by
  sorry

end NUMINAMATH_GPT_mean_temperature_l332_33200


namespace NUMINAMATH_GPT_fifty_third_card_is_A_s_l332_33278

def sequence_position (n : ℕ) : String :=
  let cycle_length := 26
  let pos_in_cycle := (n - 1) % cycle_length + 1
  if pos_in_cycle <= 13 then
    "A_s"
  else
    "A_h"

theorem fifty_third_card_is_A_s : sequence_position 53 = "A_s" := by
  sorry  -- proof placeholder

end NUMINAMATH_GPT_fifty_third_card_is_A_s_l332_33278


namespace NUMINAMATH_GPT_prob_statement_l332_33209

open Set

-- Definitions from the conditions
def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x^2 + 2 * x < 0}

-- Proposition to be proved
theorem prob_statement : A ∩ (Bᶜ) = {-2, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_prob_statement_l332_33209


namespace NUMINAMATH_GPT_probability_both_selected_l332_33234

/- 
Problem statement: Given that the probability of selection of Ram is 5/7 and that of Ravi is 1/5,
prove that the probability that both Ram and Ravi are selected is 1/7.
-/

theorem probability_both_selected (pRam : ℚ) (pRavi : ℚ) (hRam : pRam = 5 / 7) (hRavi : pRavi = 1 / 5) :
  (pRam * pRavi) = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_both_selected_l332_33234


namespace NUMINAMATH_GPT_quadratic_equation_solutions_l332_33284

theorem quadratic_equation_solutions (x : ℝ) : x * (x - 7) = 0 ↔ x = 0 ∨ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_solutions_l332_33284


namespace NUMINAMATH_GPT_moles_HCl_formed_l332_33289

-- Define the initial moles of CH4 and Cl2
def CH4_initial : ℕ := 2
def Cl2_initial : ℕ := 4

-- Define the balanced chemical equation in terms of the number of moles
def balanced_equation (CH4 : ℕ) (Cl2 : ℕ) : Prop :=
  CH4 + 4 * Cl2 = 1 * CH4 + 4 * Cl2

-- Theorem statement: Given the conditions, prove the number of moles of HCl formed is 4
theorem moles_HCl_formed (CH4_initial Cl2_initial : ℕ) (h_CH4 : CH4_initial = 2) (h_Cl2 : Cl2_initial = 4) :
  ∃ (HCl : ℕ), HCl = 4 :=
  sorry

end NUMINAMATH_GPT_moles_HCl_formed_l332_33289


namespace NUMINAMATH_GPT_monotonic_range_of_b_l332_33295

noncomputable def f (b x : ℝ) : ℝ := x^3 - b * x^2 + 3 * x - 5

theorem monotonic_range_of_b (b : ℝ) : (∀ x y: ℝ, (f b x) ≤ (f b y) → x ≤ y) ↔ -3 ≤ b ∧ b ≤ 3 :=
sorry

end NUMINAMATH_GPT_monotonic_range_of_b_l332_33295


namespace NUMINAMATH_GPT_odd_positive_integer_minus_twenty_l332_33267

theorem odd_positive_integer_minus_twenty (x : ℕ) (h : x = 53) : (2 * x - 1) - 20 = 85 := by
  subst h
  rfl

end NUMINAMATH_GPT_odd_positive_integer_minus_twenty_l332_33267


namespace NUMINAMATH_GPT_ordered_pairs_of_positive_integers_l332_33256

theorem ordered_pairs_of_positive_integers (x y : ℕ) (h : x * y = 2800) :
  2^4 * 5^2 * 7 = 2800 → ∃ (n : ℕ), n = 30 ∧ (∃ x y : ℕ, x * y = 2800 ∧ n = 30) :=
by
  sorry

end NUMINAMATH_GPT_ordered_pairs_of_positive_integers_l332_33256


namespace NUMINAMATH_GPT_number_with_20_multiples_l332_33210

theorem number_with_20_multiples : ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k) → (k ≤ 100) → (n ∣ k) → (k / n ≤ 20) ) ∧ n = 5 := 
  sorry

end NUMINAMATH_GPT_number_with_20_multiples_l332_33210


namespace NUMINAMATH_GPT_final_position_west_of_bus_stop_distance_from_bus_stop_total_calories_consumed_l332_33283

-- Define the movements as a list of integers
def movements : List ℤ := [1000, -900, 700, -1200, 1200, 100, -1100, -200]

-- Define the function to calculate the final position
def final_position (movements : List ℤ) : ℤ :=
  movements.foldl (· + ·) 0

-- Define the function to find the total distance walked (absolute sum)
def total_distance (movements : List ℤ) : ℕ :=
  movements.foldl (fun acc x => acc + x.natAbs) 0

-- Calorie consumption rate per kilometer (1000 meters)
def calories_per_kilometer : ℕ := 7000

-- Calculate the calories consumed
def calories_consumed (total_meters : ℕ) : ℕ :=
  (total_meters / 1000) * calories_per_kilometer

-- Lean 4 theorem statements

theorem final_position_west_of_bus_stop : final_position movements = -400 := by
  sorry

theorem distance_from_bus_stop : |final_position movements| = 400 := by
  sorry

theorem total_calories_consumed : calories_consumed (total_distance movements) = 44800 := by
  sorry

end NUMINAMATH_GPT_final_position_west_of_bus_stop_distance_from_bus_stop_total_calories_consumed_l332_33283


namespace NUMINAMATH_GPT_sqrt_five_gt_two_l332_33277

theorem sqrt_five_gt_two : Real.sqrt 5 > 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sqrt_five_gt_two_l332_33277


namespace NUMINAMATH_GPT_distinct_x_intercepts_l332_33293

theorem distinct_x_intercepts : 
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, (x + 5) * (x^2 + 5 * x - 6) = 0 ↔ x ∈ s :=
by { 
  sorry 
}

end NUMINAMATH_GPT_distinct_x_intercepts_l332_33293


namespace NUMINAMATH_GPT_b_negative_l332_33206

variable {R : Type*} [LinearOrderedField R]

theorem b_negative (a b : R) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : R, 0 ≤ x → (x - a) * (x - b) * (x - (2*a + b)) ≥ 0) : b < 0 := 
sorry

end NUMINAMATH_GPT_b_negative_l332_33206


namespace NUMINAMATH_GPT_cannot_tile_remaining_with_dominoes_l332_33288

def can_tile_remaining_board (pieces : List (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), ∃ (piece : ℕ × ℕ), piece ∈ pieces ∧ piece.1 = i ∧ piece.2 = j

theorem cannot_tile_remaining_with_dominoes : 
  ∃ (pieces : List (ℕ × ℕ)), (∀ (i j : ℕ), (1 ≤ i ∧ i ≤ 10) ∧ (1 ≤ j ∧ j ≤ 10) → ∃ (piece : ℕ × ℕ), piece ∈ pieces ∧ piece.1 = i ∧ piece.2 = j) ∧ ¬ can_tile_remaining_board pieces :=
sorry

end NUMINAMATH_GPT_cannot_tile_remaining_with_dominoes_l332_33288


namespace NUMINAMATH_GPT_sum_of_all_possible_values_of_abs_b_l332_33235

theorem sum_of_all_possible_values_of_abs_b {a b : ℝ}
  {r s : ℝ} (hr : r^3 + a * r + b = 0) (hs : s^3 + a * s + b = 0)
  (hr4 : (r + 4)^3 + a * (r + 4) + b + 240 = 0) (hs3 : (s - 3)^3 + a * (s - 3) + b + 240 = 0) :
  |b| = 20 ∨ |b| = 42 →
  20 + 42 = 62 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_all_possible_values_of_abs_b_l332_33235


namespace NUMINAMATH_GPT_line_points_sum_slope_and_intercept_l332_33254

-- Definition of the problem
theorem line_points_sum_slope_and_intercept (a b : ℝ) :
  (∀ x y : ℝ, (x = 2 ∧ y = 3) ∨ (x = 10 ∧ y = 19) → y = a * x + b) →
  a + b = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_line_points_sum_slope_and_intercept_l332_33254


namespace NUMINAMATH_GPT_monkey_slip_distance_l332_33272

theorem monkey_slip_distance
  (height : ℕ)
  (climb_per_hour : ℕ)
  (hours : ℕ)
  (s : ℕ)
  (total_hours : ℕ)
  (final_climb : ℕ)
  (reach_top : height = hours * (climb_per_hour - s) + final_climb)
  (total_hours_constraint : total_hours = 17)
  (climb_per_hour_constraint : climb_per_hour = 3)
  (height_constraint : height = 19)
  (final_climb_constraint : final_climb = 3)
  (hours_constraint : hours = 16) :
  s = 2 := sorry

end NUMINAMATH_GPT_monkey_slip_distance_l332_33272


namespace NUMINAMATH_GPT_inequality_solution_set_l332_33261

noncomputable def solution_set : Set ℝ := { x : ℝ | x > 5 ∨ x < -2 }

theorem inequality_solution_set (x : ℝ) :
  x^2 - 3 * x - 10 > 0 ↔ x > 5 ∨ x < -2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l332_33261


namespace NUMINAMATH_GPT_num_4digit_special_integers_l332_33291

noncomputable def count_valid_4digit_integers : ℕ :=
  let first_two_options := 3 * 3 -- options for the first two digits
  let valid_last_two_pairs := 4 -- (6,9), (7,8), (8,7), (9,6)
  first_two_options * valid_last_two_pairs

theorem num_4digit_special_integers : count_valid_4digit_integers = 36 :=
by
  sorry

end NUMINAMATH_GPT_num_4digit_special_integers_l332_33291


namespace NUMINAMATH_GPT_part_I_part_II_l332_33218

noncomputable def f (a x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem part_I (a : ℝ) : ∀ x : ℝ, (0 < (2^x * Real.log 2) / (2^x + 1)^2) :=
by
  sorry

theorem part_II (h : ∀ x : ℝ, f a x = -f a (-x)) : 
  a = (1:ℝ)/2 ∧ ∀ x : ℝ, -((1:ℝ)/2) < f (1/2) x ∧ f (1/2) x < (1:ℝ)/2 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l332_33218


namespace NUMINAMATH_GPT_new_ratio_of_boarders_to_day_scholars_l332_33221

theorem new_ratio_of_boarders_to_day_scholars
  (B_initial D_initial : ℕ)
  (B_initial_eq : B_initial = 560)
  (ratio_initial : B_initial / D_initial = 7 / 16)
  (new_boarders : ℕ)
  (new_boarders_eq : new_boarders = 80)
  (B_new : ℕ)
  (B_new_eq : B_new = B_initial + new_boarders)
  (D_new : ℕ)
  (D_new_eq : D_new = D_initial) :
  B_new / D_new = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_new_ratio_of_boarders_to_day_scholars_l332_33221


namespace NUMINAMATH_GPT_plane_determination_l332_33264

inductive Propositions : Type where
  | p1 : Propositions
  | p2 : Propositions
  | p3 : Propositions
  | p4 : Propositions

open Propositions

def correct_proposition := p4

theorem plane_determination (H: correct_proposition = p4): correct_proposition = p4 := 
by 
  exact H

end NUMINAMATH_GPT_plane_determination_l332_33264


namespace NUMINAMATH_GPT_prove_y_l332_33233

theorem prove_y (x y : ℝ) (h1 : 3 * x^2 - 4 * x + 7 * y + 3 = 0) (h2 : 3 * x - 5 * y + 6 = 0) :
  25 * y^2 - 39 * y + 69 = 0 := sorry

end NUMINAMATH_GPT_prove_y_l332_33233


namespace NUMINAMATH_GPT_boss_spends_7600_per_month_l332_33279

def hoursPerWeekFiona : ℕ := 40
def hoursPerWeekJohn : ℕ := 30
def hoursPerWeekJeremy : ℕ := 25
def hourlyRate : ℕ := 20
def weeksPerMonth : ℕ := 4

def weeklyEarnings (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def monthlyEarnings (weekly : ℕ) (weeks : ℕ) : ℕ := weekly * weeks

def totalMonthlyExpenditure : ℕ :=
  monthlyEarnings (weeklyEarnings hoursPerWeekFiona hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJohn hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJeremy hourlyRate) weeksPerMonth

theorem boss_spends_7600_per_month :
  totalMonthlyExpenditure = 7600 :=
by
  sorry

end NUMINAMATH_GPT_boss_spends_7600_per_month_l332_33279


namespace NUMINAMATH_GPT_Bernoulli_inequality_l332_33239

theorem Bernoulli_inequality (p : ℝ) (k : ℚ) (hp : 0 < p) (hk : 1 < k) : 
  (1 + p) ^ (k : ℝ) > 1 + p * (k : ℝ) := by
sorry

end NUMINAMATH_GPT_Bernoulli_inequality_l332_33239


namespace NUMINAMATH_GPT_angle_triple_complement_l332_33207

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end NUMINAMATH_GPT_angle_triple_complement_l332_33207
