import Mathlib

namespace complex_number_problem_l1168_116817

theorem complex_number_problem (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) * (1 - Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) →
  (a = -1 ∧ Complex.abs (z + Complex.I) = 3) := by
sorry

end complex_number_problem_l1168_116817


namespace hyperbola_eccentricity_l1168_116831

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 where a > 0, b > 0,
    and one of its asymptotes is y = √2 x, prove that its eccentricity is √3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x : ℝ, y = Real.sqrt 2 * x) →
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 :=
by sorry

end hyperbola_eccentricity_l1168_116831


namespace perfect_square_factors_count_l1168_116836

/-- Given a natural number with prime factorization 2^6 × 3^3, 
    this function returns the count of its positive integer factors that are perfect squares -/
def count_perfect_square_factors (N : ℕ) : ℕ :=
  8

/-- The theorem stating that for a number with prime factorization 2^6 × 3^3,
    the count of its positive integer factors that are perfect squares is 8 -/
theorem perfect_square_factors_count (N : ℕ) 
  (h : N = 2^6 * 3^3) : 
  count_perfect_square_factors N = 8 := by
  sorry

end perfect_square_factors_count_l1168_116836


namespace vkontakte_problem_l1168_116882

-- Define predicates for each person being on VKontakte
variable (M I A P : Prop)

-- State the theorem
theorem vkontakte_problem :
  (M → (I ∧ A)) →  -- If M is on VKontakte, then both I and A are on VKontakte
  (A ↔ ¬P) →       -- Only one of A or P is on VKontakte
  (I ∨ M) →        -- At least one of I or M is on VKontakte
  (P ↔ I) →        -- P and I are either both on or both not on VKontakte
  (I ∧ P ∧ ¬M ∧ ¬A) -- Conclusion: I and P are on VKontakte, M and A are not
  := by sorry

end vkontakte_problem_l1168_116882


namespace sum_f_two_and_neg_two_l1168_116883

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -1 / x else x^2

theorem sum_f_two_and_neg_two : f 2 + f (-2) = 7/2 := by
  sorry

end sum_f_two_and_neg_two_l1168_116883


namespace one_correct_statement_l1168_116897

theorem one_correct_statement : 
  (∃! n : ℕ, n = 1 ∧ 
    (∀ x : ℤ, x < 0 → x ≤ -1) ∧ 
    (∃ y : ℝ, -(y) ≤ 0) ∧
    (∃ z : ℚ, z = 0) ∧
    (∃ a : ℝ, -a > 0) ∧
    (∃ b₁ b₂ : ℚ, b₁ < 0 ∧ b₂ < 0 ∧ b₁ * b₂ > 0)) :=
by sorry

end one_correct_statement_l1168_116897


namespace smallest_x_for_equation_l1168_116821

theorem smallest_x_for_equation : 
  ∃ (x : ℕ+), x = 9 ∧ 
  (∃ (y : ℕ+), (9 : ℚ) / 10 = (y : ℚ) / (151 + x)) ∧ 
  (∀ (x' : ℕ+), x' < x → 
    ¬∃ (y : ℕ+), (9 : ℚ) / 10 = (y : ℚ) / (151 + x')) := by
  sorry

end smallest_x_for_equation_l1168_116821


namespace total_purchase_cost_l1168_116826

def snake_toy_cost : ℚ := 11.76
def cage_cost : ℚ := 14.54

theorem total_purchase_cost : snake_toy_cost + cage_cost = 26.30 := by
  sorry

end total_purchase_cost_l1168_116826


namespace red_subsequence_2009_l1168_116894

/-- Represents the coloring rule for the red subsequence -/
def red_subsequence : ℕ → ℕ → ℕ → ℕ
| 0, _, _ => 1
| (n+1), count, last =>
  if n % 2 = 0 then
    if count < n + 1 then red_subsequence n (count + 1) (last + 2)
    else red_subsequence n 0 (last + 1)
  else
    if count < n + 2 then red_subsequence n (count + 1) (last + 2)
    else red_subsequence (n + 1) 0 last

/-- The 2009th number in the red subsequence is 3953 -/
theorem red_subsequence_2009 :
  (red_subsequence 1000 0 1) = 3953 := by sorry

end red_subsequence_2009_l1168_116894


namespace total_shaded_area_is_one_third_l1168_116823

/-- Represents the fractional area shaded in each step of the square division pattern. -/
def shadedAreaSequence : ℕ → ℚ
  | 0 => 1/4
  | n + 1 => (1/4) * shadedAreaSequence n

/-- The sum of the infinite geometric series representing the total shaded area. -/
noncomputable def totalShadedArea : ℚ := ∑' n, shadedAreaSequence n

/-- Theorem stating that the total shaded area is equal to 1/3. -/
theorem total_shaded_area_is_one_third :
  totalShadedArea = 1/3 := by sorry

end total_shaded_area_is_one_third_l1168_116823


namespace correct_answer_points_l1168_116864

/-- Represents the scoring system for a math competition --/
structure ScoringSystem where
  total_problems : ℕ
  wang_score : ℤ
  zhang_score : ℤ
  correct_points : ℕ
  incorrect_points : ℕ

/-- Theorem stating that the given scoring system results in 25 points for correct answers --/
theorem correct_answer_points (s : ScoringSystem) : 
  s.total_problems = 20 ∧ 
  s.wang_score = 328 ∧ 
  s.zhang_score = 27 ∧ 
  s.correct_points ≥ 10 ∧ s.correct_points ≤ 99 ∧
  s.incorrect_points ≥ 10 ∧ s.incorrect_points ≤ 99 →
  s.correct_points = 25 := by
  sorry


end correct_answer_points_l1168_116864


namespace margaux_lending_problem_l1168_116872

/-- Margaux's money lending problem -/
theorem margaux_lending_problem (brother_payment cousin_payment total_days total_collection : ℕ) 
  (h1 : brother_payment = 8)
  (h2 : cousin_payment = 4)
  (h3 : total_days = 7)
  (h4 : total_collection = 119) :
  ∃ (friend_payment : ℕ), 
    friend_payment * total_days + brother_payment * total_days + cousin_payment * total_days = total_collection ∧ 
    friend_payment = 5 := by
  sorry

end margaux_lending_problem_l1168_116872


namespace rat_value_l1168_116846

/-- Represents the alphabet with corresponding numeric values. --/
def alphabet : List (Char × Nat) := [
  ('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7), ('h', 8), ('i', 9), ('j', 10),
  ('k', 11), ('l', 12), ('m', 13), ('n', 14), ('o', 15), ('p', 16), ('q', 17), ('r', 18), ('s', 19),
  ('t', 20), ('u', 21), ('v', 22), ('w', 23), ('x', 24), ('y', 25), ('z', 26)
]

/-- Gets the numeric value of a character based on its position in the alphabet. --/
def letterValue (c : Char) : Nat :=
  (alphabet.find? (fun p => p.1 == c.toLower)).map Prod.snd |>.getD 0

/-- Calculates the number value of a word based on the given rules. --/
def wordValue (word : String) : Nat :=
  let letterSum := word.toList.map letterValue |>.sum
  letterSum * word.length

/-- Theorem stating that the number value of "rat" is 117. --/
theorem rat_value : wordValue "rat" = 117 := by
  sorry

end rat_value_l1168_116846


namespace smallest_undefined_inverse_l1168_116842

theorem smallest_undefined_inverse (a : ℕ) : 
  (a > 0) → 
  (¬ ∃ (x : ℕ), x * a ≡ 1 [MOD 77]) → 
  (¬ ∃ (y : ℕ), y * a ≡ 1 [MOD 91]) → 
  (∀ (b : ℕ), b > 0 ∧ b < a → 
    (∃ (x : ℕ), x * b ≡ 1 [MOD 77]) ∨ 
    (∃ (y : ℕ), y * b ≡ 1 [MOD 91])) → 
  a = 7 :=
by sorry

end smallest_undefined_inverse_l1168_116842


namespace parking_theorem_l1168_116874

/-- The number of parking spaces in a row -/
def total_spaces : ℕ := 7

/-- The number of cars to be parked -/
def num_cars : ℕ := 4

/-- The number of consecutive empty spaces required -/
def consecutive_empty : ℕ := 3

/-- The number of different parking arrangements -/
def parking_arrangements : ℕ := 120

/-- Theorem stating that the number of ways to arrange 4 cars and 3 consecutive
    empty spaces in a row of 7 parking spaces is equal to 120 -/
theorem parking_theorem :
  (total_spaces = 7) →
  (num_cars = 4) →
  (consecutive_empty = 3) →
  (parking_arrangements = 120) :=
by sorry

end parking_theorem_l1168_116874


namespace cube_volume_from_circumscribed_sphere_l1168_116802

theorem cube_volume_from_circumscribed_sphere (V_sphere : ℝ) :
  V_sphere = (32 / 3) * Real.pi →
  ∃ (V_cube : ℝ), V_cube = (64 * Real.sqrt 3) / 9 ∧ 
  (∃ (a : ℝ), V_cube = a^3 ∧ V_sphere = (4 / 3) * Real.pi * ((a * Real.sqrt 3) / 2)^3) :=
by sorry

end cube_volume_from_circumscribed_sphere_l1168_116802


namespace linear_equation_m_value_l1168_116855

theorem linear_equation_m_value (m : ℤ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (m - 1) * x^(abs m) - 2 = a * x + b) → m = -1 :=
by sorry

end linear_equation_m_value_l1168_116855


namespace perpendicular_lines_from_parallel_planes_l1168_116880

-- Define the necessary types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (belongs_to : Point → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (α β : Plane) (l n : Line) :
  parallel α β →
  perpendicular l α →
  contained_in n β →
  perpendicular_lines l n :=
sorry

end perpendicular_lines_from_parallel_planes_l1168_116880


namespace union_M_N_equals_N_l1168_116839

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x > 0}
def N : Set ℝ := {x | |x| > 2}

-- State the theorem
theorem union_M_N_equals_N : M ∪ N = N := by sorry

end union_M_N_equals_N_l1168_116839


namespace dot_product_specific_value_l1168_116889

/-- Dot product of two 3D vectors -/
def dot_product (a b c p q r : ℝ) : ℝ := a * p + b * q + c * r

theorem dot_product_specific_value :
  let y : ℝ := 12.5
  let n : ℝ := dot_product 3 4 5 y (-2) 1
  n = 34.5 := by
  sorry

end dot_product_specific_value_l1168_116889


namespace trumpet_cost_l1168_116825

/-- The cost of the trumpet given the total spent and the costs of other items. -/
theorem trumpet_cost (total_spent music_tool_cost song_book_cost : ℚ) 
  (h1 : total_spent = 163.28)
  (h2 : music_tool_cost = 9.98)
  (h3 : song_book_cost = 4.14) :
  total_spent - (music_tool_cost + song_book_cost) = 149.16 := by
  sorry

end trumpet_cost_l1168_116825


namespace simplify_square_roots_l1168_116830

theorem simplify_square_roots : 
  2 * Real.sqrt 12 - Real.sqrt 27 - Real.sqrt 3 * Real.sqrt (1/9) = (2 * Real.sqrt 3) / 3 := by
  sorry

end simplify_square_roots_l1168_116830


namespace m_range_l1168_116853

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 ≥ 0

def q (m : ℝ) : Prop := ∀ x, (8*x + 4*(m - 1)) ≠ 0

-- Define the theorem
theorem m_range (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → ((-2 ≤ m ∧ m < 1) ∨ m > 2) :=
sorry

end m_range_l1168_116853


namespace equation_solutions_l1168_116888

def solution_set : Set (ℤ × ℤ) :=
  {(-15, -3), (-1, -1), (2, 14), (3, -21), (5, -7), (6, -6), (20, -4)}

def satisfies_equation (pair : ℤ × ℤ) : Prop :=
  let (x, y) := pair
  x ≠ 0 ∧ y ≠ 0 ∧ (5 : ℚ) / x - (7 : ℚ) / y = 2

theorem equation_solutions :
  ∀ (x y : ℤ), satisfies_equation (x, y) ↔ (x, y) ∈ solution_set := by
  sorry

end equation_solutions_l1168_116888


namespace smallest_side_of_triangle_l1168_116843

theorem smallest_side_of_triangle : ∃ (s : ℕ),
  (s : ℝ) > 0 ∧ 
  7.5 + (s : ℝ) > 11 ∧ 
  7.5 + 11 > (s : ℝ) ∧ 
  11 + (s : ℝ) > 7.5 ∧
  ∀ (t : ℕ), t > 0 → 
    (7.5 + (t : ℝ) > 11 ∧ 
     7.5 + 11 > (t : ℝ) ∧ 
     11 + (t : ℝ) > 7.5) → 
    s ≤ t ∧
  s = 4 :=
sorry

end smallest_side_of_triangle_l1168_116843


namespace girls_in_class_l1168_116850

/-- 
Given a class with a total of 60 people and a ratio of girls to boys to teachers of 3:2:1,
prove that the number of girls in the class is 30.
-/
theorem girls_in_class (total : ℕ) (girls boys teachers : ℕ) : 
  total = 60 →
  girls + boys + teachers = total →
  girls = 3 * teachers →
  boys = 2 * teachers →
  girls = 30 := by
sorry

end girls_in_class_l1168_116850


namespace janes_shadow_length_l1168_116814

/-- Given a tree and a person (Jane) casting shadows, this theorem proves
    the length of Jane's shadow based on the heights of the tree and Jane,
    and the length of the tree's shadow. -/
theorem janes_shadow_length
  (tree_height : ℝ)
  (tree_shadow : ℝ)
  (jane_height : ℝ)
  (h_tree_height : tree_height = 30)
  (h_tree_shadow : tree_shadow = 10)
  (h_jane_height : jane_height = 1.5) :
  jane_height * tree_shadow / tree_height = 0.5 := by
  sorry


end janes_shadow_length_l1168_116814


namespace lara_future_age_l1168_116893

def lara_age_7_years_ago : ℕ := 9

def lara_current_age : ℕ := lara_age_7_years_ago + 7

def lara_age_10_years_from_now : ℕ := lara_current_age + 10

theorem lara_future_age : lara_age_10_years_from_now = 26 := by
  sorry

end lara_future_age_l1168_116893


namespace pythagorean_triple_check_l1168_116885

def isPythagoreanTriple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem pythagorean_triple_check :
  ¬(isPythagoreanTriple 2 3 4) ∧
  (isPythagoreanTriple 3 4 5) ∧
  (isPythagoreanTriple 6 8 10) ∧
  (isPythagoreanTriple 5 12 13) :=
by sorry

end pythagorean_triple_check_l1168_116885


namespace complement_intersection_theorem_l1168_116832

def U : Set ℕ := {1,2,3,4,5,6,7,8,9}
def A : Set ℕ := {2,4,5,7}
def B : Set ℕ := {3,4,5,6,8}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {3,6,8} := by sorry

end complement_intersection_theorem_l1168_116832


namespace first_coordinate_on_line_l1168_116805

theorem first_coordinate_on_line (n : ℝ) (a : ℝ) :
  (a = 4 * n + 5 ∧ a + 2 = 4 * (n + 0.5) + 5) → a = 4 * n + 5 :=
by sorry

end first_coordinate_on_line_l1168_116805


namespace length_OP_specific_case_l1168_116881

/-- Given a circle with center O and radius r, and two intersecting chords AB and CD,
    this function calculates the length of OP, where P is the intersection point of the chords. -/
def length_OP (r : ℝ) (chord_AB : ℝ) (chord_CD : ℝ) (midpoint_distance : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for a circle with radius 20 and two intersecting chords of lengths 24 and 18,
    if the distance between their midpoints is 10, then the length of OP is approximately 14.8. -/
theorem length_OP_specific_case :
  let r := 20
  let chord_AB := 24
  let chord_CD := 18
  let midpoint_distance := 10
  ∃ ε > 0, |length_OP r chord_AB chord_CD midpoint_distance - 14.8| < ε :=
by sorry

end length_OP_specific_case_l1168_116881


namespace unique_modular_solution_l1168_116852

theorem unique_modular_solution : ∃! n : ℕ, n ≤ 9 ∧ n ≡ -1345 [ZMOD 10] := by
  sorry

end unique_modular_solution_l1168_116852


namespace corn_harvest_difference_l1168_116833

theorem corn_harvest_difference (greg_harvest sharon_harvest : ℝ) 
  (h1 : greg_harvest = 0.4)
  (h2 : sharon_harvest = 0.1) :
  greg_harvest - sharon_harvest = 0.3 := by
  sorry

end corn_harvest_difference_l1168_116833


namespace average_difference_is_negative_six_point_fifteen_l1168_116816

/- Define the parameters of the problem -/
def total_students : ℕ := 120
def total_teachers : ℕ := 6
def dual_enrolled_students : ℕ := 10
def class_enrollments : List ℕ := [40, 30, 25, 15, 5, 5]

/- Define the average number of students per teacher -/
def t : ℚ := (total_students : ℚ) / total_teachers

/- Define the average number of students per student, including dual enrollments -/
def s : ℚ :=
  let total_enrollments := total_students + dual_enrolled_students
  (class_enrollments.map (λ x => (x : ℚ) * x / total_enrollments)).sum

/- The theorem to be proved -/
theorem average_difference_is_negative_six_point_fifteen :
  t - s = -315 / 100 := by sorry

end average_difference_is_negative_six_point_fifteen_l1168_116816


namespace coffee_table_price_is_330_l1168_116873

/-- Represents the living room set purchase -/
structure LivingRoomSet where
  sofa_price : ℕ
  armchair_price : ℕ
  num_armchairs : ℕ
  total_invoice : ℕ

/-- Calculates the price of the coffee table -/
def coffee_table_price (set : LivingRoomSet) : ℕ :=
  set.total_invoice - (set.sofa_price + set.armchair_price * set.num_armchairs)

/-- Theorem stating that the coffee table price is 330 -/
theorem coffee_table_price_is_330 (set : LivingRoomSet) 
  (h1 : set.sofa_price = 1250)
  (h2 : set.armchair_price = 425)
  (h3 : set.num_armchairs = 2)
  (h4 : set.total_invoice = 2430) :
  coffee_table_price set = 330 := by
  sorry

#check coffee_table_price_is_330

end coffee_table_price_is_330_l1168_116873


namespace smallest_number_l1168_116861

theorem smallest_number : 
  let numbers := [-0.991, -0.981, -0.989, -0.9801, -0.9901]
  ∀ x ∈ numbers, -0.991 ≤ x :=
by sorry

end smallest_number_l1168_116861


namespace addition_multiplication_equality_l1168_116803

theorem addition_multiplication_equality : 300 + 5 * 8 = 340 := by
  sorry

end addition_multiplication_equality_l1168_116803


namespace ellipse_eccentricity_l1168_116844

/-- Given an ellipse C with semi-major axis a and semi-minor axis b,
    and a circle with diameter 2a tangent to a line,
    prove that the eccentricity of C is √(6)/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let L := {(x, y) : ℝ × ℝ | b * x - a * y + 2 * a * b = 0}
  let circle_diameter := 2 * a
  (∃ (p : ℝ × ℝ), p ∈ C ∧ p ∈ L) →  -- The circle is tangent to the line
  let e := Real.sqrt (1 - b^2 / a^2)  -- Eccentricity definition
  e = Real.sqrt 6 / 3 := by
sorry

end ellipse_eccentricity_l1168_116844


namespace inequality_proof_l1168_116818

theorem inequality_proof (p q r : ℝ) (n : ℕ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0) 
  (h_product : p * q * r = 1) : 
  1 / (p^n + q^n + 1) + 1 / (q^n + r^n + 1) + 1 / (r^n + p^n + 1) ≤ 1 := by
  sorry

end inequality_proof_l1168_116818


namespace geometric_sequence_problem_l1168_116867

/-- A positive geometric sequence -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_problem (a : ℕ → ℝ)
    (h_geom : IsPositiveGeometricSequence a)
    (h_sum : a 1 + 2/3 * a 2 = 3)
    (h_prod : a 4 ^ 2 = 1/9 * a 3 * a 7) :
    a 4 = 27 := by
  sorry

end geometric_sequence_problem_l1168_116867


namespace sum_of_solutions_l1168_116895

theorem sum_of_solutions (x : ℝ) : 
  (9 * x / 45 = 6 / x) → (x = 0 ∨ x = 6 / 5) ∧ (0 + 6 / 5 = 6 / 5) := by
  sorry

end sum_of_solutions_l1168_116895


namespace triangle_side_length_l1168_116887

/-- Given a triangle ABC with the following properties:
  * The product of sides a and b is 60√3
  * The sine of angle B equals the sine of angle C
  * The area of the triangle is 15√3
  This theorem states that the length of side b is 2√15 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  a * b = 60 * Real.sqrt 3 →
  Real.sin B = Real.sin C →
  (1/2) * a * b * Real.sin C = 15 * Real.sqrt 3 →
  b = 2 * Real.sqrt 15 := by
  sorry

end triangle_side_length_l1168_116887


namespace transformed_circle_equation_l1168_116866

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the scaling transformation
def scaling_transform (x y x' y' : ℝ) : Prop := x' = 5*x ∧ y' = 3*y

-- State the theorem
theorem transformed_circle_equation (x y x' y' : ℝ) :
  original_circle x y ∧ scaling_transform x y x' y' →
  x'^2 / 25 + y'^2 / 9 = 1 :=
by sorry

end transformed_circle_equation_l1168_116866


namespace circle_properties_l1168_116827

-- Define the circle equation type
def CircleEquation := ℝ → ℝ → ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the properties of the circles and points
def CircleProperties (f : CircleEquation) (P₁ P₂ : Point) :=
  f P₁.x P₁.y = 0 ∧ f P₂.x P₂.y ≠ 0

-- Define the new circle equation
def NewCircleEquation (f : CircleEquation) (P₁ P₂ : Point) : CircleEquation :=
  fun x y => f x y - f P₁.x P₁.y - f P₂.x P₂.y

-- Theorem statement
theorem circle_properties
  (f : CircleEquation)
  (P₁ P₂ : Point)
  (h : CircleProperties f P₁ P₂) :
  let g := NewCircleEquation f P₁ P₂
  (g P₂.x P₂.y = 0) ∧
  (∀ x y, g x y = 0 → f x y = f P₂.x P₂.y) := by
  sorry

end circle_properties_l1168_116827


namespace library_books_end_of_month_l1168_116879

theorem library_books_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) : 
  initial_books = 75 →
  loaned_books = 20 →
  return_rate = 65 / 100 →
  initial_books - loaned_books + (↑loaned_books * return_rate).floor = 68 :=
by sorry

end library_books_end_of_month_l1168_116879


namespace gcd_105_45_l1168_116854

theorem gcd_105_45 : Nat.gcd 105 45 = 15 := by
  sorry

end gcd_105_45_l1168_116854


namespace parallel_vectors_x_value_l1168_116800

def a : Fin 2 → ℝ := ![3, -2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 4]

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), k ≠ 0 ∧ b x = k • a) → x = -6 := by
  sorry

end parallel_vectors_x_value_l1168_116800


namespace existence_of_x_l1168_116807

theorem existence_of_x (a : Fin 1997 → ℕ)
  (h1 : ∀ i j : Fin 1997, i + j ≤ 1997 → a i + a j ≤ a (i + j))
  (h2 : ∀ i j : Fin 1997, i + j ≤ 1997 → a (i + j) ≤ a i + a j + 1) :
  ∃ x : ℝ, ∀ n : Fin 1997, a n = ⌊n * x⌋ := by
sorry

end existence_of_x_l1168_116807


namespace difference_of_squares_l1168_116884

theorem difference_of_squares : (502 : ℤ) * 502 - (501 : ℤ) * 503 = 1 := by sorry

end difference_of_squares_l1168_116884


namespace min_value_sqrt_a_plus_four_over_sqrt_a_plus_one_l1168_116838

theorem min_value_sqrt_a_plus_four_over_sqrt_a_plus_one (a : ℝ) (ha : a > 0) :
  Real.sqrt a + 4 / (Real.sqrt a + 1) ≥ 3 := by
  sorry

end min_value_sqrt_a_plus_four_over_sqrt_a_plus_one_l1168_116838


namespace species_assignment_theorem_l1168_116860

/-- Represents the compatibility between species -/
def Compatibility := Fin 8 → Finset (Fin 8)

/-- Theorem stating that it's possible to assign 8 species to 4 cages
    given the compatibility constraints -/
theorem species_assignment_theorem (c : Compatibility)
  (h : ∀ s : Fin 8, (c s).card ≤ 4) :
  ∃ (assignment : Fin 8 → Fin 4),
    ∀ s₁ s₂ : Fin 8, assignment s₁ = assignment s₂ → s₂ ∈ c s₁ := by
  sorry

end species_assignment_theorem_l1168_116860


namespace different_color_chips_probability_l1168_116891

theorem different_color_chips_probability :
  let total_chips : ℕ := 7 + 6 + 5
  let purple_chips : ℕ := 7
  let green_chips : ℕ := 6
  let orange_chips : ℕ := 5
  let prob_purple : ℚ := purple_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  let prob_orange : ℚ := orange_chips / total_chips
  let prob_not_purple : ℚ := (green_chips + orange_chips) / total_chips
  let prob_not_green : ℚ := (purple_chips + orange_chips) / total_chips
  let prob_not_orange : ℚ := (purple_chips + green_chips) / total_chips
  (prob_purple * prob_not_purple + prob_green * prob_not_green + prob_orange * prob_not_orange) = 107 / 162 :=
by sorry

end different_color_chips_probability_l1168_116891


namespace marbles_in_first_jar_l1168_116869

theorem marbles_in_first_jar (jar1 jar2 jar3 : ℕ) : 
  jar2 = 2 * jar1 →
  jar3 = jar1 / 4 →
  jar1 + jar2 + jar3 = 260 →
  jar1 = 80 := by
sorry

end marbles_in_first_jar_l1168_116869


namespace x_equation_implies_polynomial_value_l1168_116892

theorem x_equation_implies_polynomial_value (x : ℝ) (h : x + 1/x = Real.sqrt 3) :
  x^7 - 5*x^5 + x^2 = -1 := by
  sorry

end x_equation_implies_polynomial_value_l1168_116892


namespace rectangle_not_stable_l1168_116863

-- Define the shape type
inductive Shape
| AcuteTriangle
| Rectangle
| RightTriangle
| IsoscelesTriangle

-- Define stability property
def IsStable (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => False
  | _ => True

-- State the theorem
theorem rectangle_not_stable :
  ∀ (s : Shape), ¬(IsStable s) ↔ s = Shape.Rectangle :=
by sorry

end rectangle_not_stable_l1168_116863


namespace father_sons_ages_l1168_116804

theorem father_sons_ages (father_age : ℕ) (youngest_son_age : ℕ) (years_until_equal : ℕ) :
  father_age = 33 →
  youngest_son_age = 2 →
  years_until_equal = 12 →
  ∃ (middle_son_age oldest_son_age : ℕ),
    (father_age + years_until_equal = youngest_son_age + years_until_equal + 
                                      middle_son_age + years_until_equal + 
                                      oldest_son_age + years_until_equal) ∧
    (middle_son_age = 3 ∧ oldest_son_age = 4) :=
by sorry

end father_sons_ages_l1168_116804


namespace function_passes_through_point_l1168_116820

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x - 1)

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = 5 := by
  sorry

end function_passes_through_point_l1168_116820


namespace half_inequality_l1168_116815

theorem half_inequality (a b : ℝ) (h : a > b) : (1/2) * a > (1/2) * b := by
  sorry

end half_inequality_l1168_116815


namespace triangle_max_side_length_range_l1168_116813

theorem triangle_max_side_length_range (P : ℝ) (a b c : ℝ) (h_triangle : a + b + c = P) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_max : c = max a (max b c)) : P / 3 ≤ c ∧ c < P / 2 := by
  sorry

end triangle_max_side_length_range_l1168_116813


namespace section_b_average_weight_l1168_116848

/-- Proves that the average weight of section B is 30 kg given the conditions of the problem -/
theorem section_b_average_weight 
  (students_a : ℕ) 
  (students_b : ℕ) 
  (total_students : ℕ) 
  (avg_weight_a : ℝ) 
  (avg_weight_total : ℝ) 
  (h1 : students_a = 36)
  (h2 : students_b = 24)
  (h3 : total_students = students_a + students_b)
  (h4 : avg_weight_a = 30)
  (h5 : avg_weight_total = 30) :
  (total_students * avg_weight_total - students_a * avg_weight_a) / students_b = 30 :=
by sorry

end section_b_average_weight_l1168_116848


namespace triangle_area_sum_l1168_116862

theorem triangle_area_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + y^2 = 3^2)
  (h2 : y^2 + y*z + z^2 = 4^2)
  (h3 : x^2 + Real.sqrt 3 * x*z + z^2 = 5^2) :
  2*x*y + x*z + Real.sqrt 3 * y*z = 24 := by
sorry

end triangle_area_sum_l1168_116862


namespace exists_tangent_circle_l1168_116809

/-- Two parallel lines in a plane -/
structure ParallelLines where
  distance : ℝ
  distance_pos : distance > 0

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- The configuration of our geometry problem -/
structure Configuration where
  lines : ParallelLines
  given_circle : Circle
  circle_between_lines : given_circle.center.2 > 0 ∧ given_circle.center.2 < lines.distance

/-- The theorem stating the existence of the sought circle -/
theorem exists_tangent_circle (config : Configuration) :
  ∃ (tangent_circle : Circle),
    tangent_circle.radius = config.lines.distance / 2 ∧
    (tangent_circle.center.2 = config.lines.distance / 2 ∨
     tangent_circle.center.2 = config.lines.distance / 2) ∧
    ((tangent_circle.center.1 - config.given_circle.center.1) ^ 2 +
     (tangent_circle.center.2 - config.given_circle.center.2) ^ 2 =
     (tangent_circle.radius + config.given_circle.radius) ^ 2) :=
by sorry

end exists_tangent_circle_l1168_116809


namespace at_hash_sum_l1168_116834

def at_operation (a b : ℕ+) : ℚ := (a.val * b.val : ℚ) / (a.val + b.val)

def hash_operation (a b : ℕ+) : ℚ := (a.val + 3 * b.val : ℚ) / (b.val + 3 * a.val)

theorem at_hash_sum :
  (at_operation 3 9) + (hash_operation 3 9) = 47 / 12 := by sorry

end at_hash_sum_l1168_116834


namespace external_internal_triangles_form_parallelogram_l1168_116801

-- Define the basic structures
structure Point :=
  (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Quadrilateral :=
  (A B C D : Point)

-- Define the properties
def isEquilateral (t : Triangle) : Prop :=
  sorry

def areSimilar (t1 t2 : Triangle) : Prop :=
  sorry

def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

def constructedExternally (base outer : Triangle) : Prop :=
  sorry

def constructedInternally (base inner : Triangle) : Prop :=
  sorry

-- State the theorem
theorem external_internal_triangles_form_parallelogram
  (ABC : Triangle)
  (AB₁C AC₁B BA₁C : Triangle)
  (ABB₁AC₁ : Quadrilateral) :
  isEquilateral AB₁C ∧
  isEquilateral AC₁B ∧
  areSimilar AB₁C ABC ∧
  areSimilar AC₁B ABC ∧
  constructedExternally ABC AB₁C ∧
  constructedExternally ABC AC₁B ∧
  constructedInternally ABC BA₁C ∧
  ABB₁AC₁.A = ABC.A ∧
  ABB₁AC₁.B = AB₁C.B ∧
  ABB₁AC₁.C = AC₁B.C ∧
  ABB₁AC₁.D = BA₁C.A →
  isParallelogram ABB₁AC₁ :=
sorry

end external_internal_triangles_form_parallelogram_l1168_116801


namespace eugene_pencils_l1168_116837

def distribute_pencils (initial : ℕ) (received : ℕ) (per_friend : ℕ) : ℕ :=
  (initial + received) % per_friend

theorem eugene_pencils : distribute_pencils 127 14 7 = 1 := by
  sorry

end eugene_pencils_l1168_116837


namespace annika_hikes_four_km_l1168_116822

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  flatRate : ℝ  -- Rate on flat terrain in minutes per kilometer
  initialDistance : ℝ  -- Initial distance hiked east in kilometers
  totalTime : ℝ  -- Total time available for the round trip in minutes
  uphillDistance : ℝ  -- Distance of uphill section in kilometers
  uphillRate : ℝ  -- Rate on uphill section in minutes per kilometer
  downhillDistance : ℝ  -- Distance of downhill section in kilometers
  downhillRate : ℝ  -- Rate on downhill section in minutes per kilometer

/-- Calculates the total distance hiked east given the hiking scenario -/
def totalDistanceEast (scenario : HikingScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, Annika will hike 4 km east -/
theorem annika_hikes_four_km : 
  let scenario : HikingScenario := {
    flatRate := 10,
    initialDistance := 2.75,
    totalTime := 45,
    uphillDistance := 0.5,
    uphillRate := 15,
    downhillDistance := 0.5,
    downhillRate := 5
  }
  totalDistanceEast scenario = 4 := by
  sorry

end annika_hikes_four_km_l1168_116822


namespace prob_two_sixes_one_four_l1168_116868

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of rolling a specific number on a single die -/
def single_prob : ℚ := 1 / num_sides

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The number of ways to arrange two 6's and one 4 in three dice rolls -/
def num_arrangements : ℕ := 3

/-- The probability of rolling exactly two 6's and one 4 when rolling three six-sided dice simultaneously -/
theorem prob_two_sixes_one_four : 
  (single_prob ^ num_dice * num_arrangements : ℚ) = 1 / 72 := by sorry

end prob_two_sixes_one_four_l1168_116868


namespace uncle_lou_peanuts_l1168_116878

/-- Calculates the number of peanuts in each bag given the conditions of Uncle Lou's flight. -/
theorem uncle_lou_peanuts (bags : ℕ) (flight_duration : ℕ) (eating_rate : ℕ) : bags = 4 → flight_duration = 120 → eating_rate = 1 → (flight_duration / bags : ℕ) = 30 := by
  sorry

end uncle_lou_peanuts_l1168_116878


namespace unpainted_cubes_6x6x6_l1168_116841

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  painted_area : Nat
  total_cubes : Nat

/-- Calculate the number of unpainted cubes in a PaintedCube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_cubes - (6 * cube.painted_area - 4 * cube.painted_area + 8)

/-- Theorem: In a 6x6x6 cube with central 4x4 areas painted, there are 160 unpainted cubes -/
theorem unpainted_cubes_6x6x6 :
  let cube : PaintedCube := { size := 6, painted_area := 16, total_cubes := 216 }
  unpainted_cubes cube = 160 := by
  sorry

end unpainted_cubes_6x6x6_l1168_116841


namespace fraction_evaluation_l1168_116845

theorem fraction_evaluation : 
  (1 - 1/4) / (1 - 2/3) + 1/6 = 29/12 := by
  sorry

end fraction_evaluation_l1168_116845


namespace negation_of_universal_statement_l1168_116808

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ≥ 0 → x^2 + x - 1 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 + x - 1 ≤ 0) :=
by sorry

end negation_of_universal_statement_l1168_116808


namespace initial_experiment_range_is_appropriate_l1168_116857

-- Define the types of microorganisms
inductive Microorganism
| Bacteria
| Actinomycetes
| Fungi
| Unknown

-- Define a function to represent the typical dilution range for each microorganism
def typicalDilutionRange (m : Microorganism) : Set ℕ :=
  match m with
  | Microorganism.Bacteria => {4, 5, 6}
  | Microorganism.Actinomycetes => {3, 4, 5}
  | Microorganism.Fungi => {2, 3, 4}
  | Microorganism.Unknown => {}

-- Define the general dilution range for initial experiments
def initialExperimentRange : Set ℕ := {n | 1 ≤ n ∧ n ≤ 7}

-- Theorem statement
theorem initial_experiment_range_is_appropriate :
  ∀ m : Microorganism, (typicalDilutionRange m).Subset initialExperimentRange :=
sorry

end initial_experiment_range_is_appropriate_l1168_116857


namespace motion_equation_l1168_116810

/-- Given a point's rectilinear motion with velocity v(t) = t^2 - 8t + 3,
    prove that its displacement function s(t) satisfies
    s(t) = t^3/3 - 4t^2 + 3t + C for some constant C. -/
theorem motion_equation (v : ℝ → ℝ) (s : ℝ → ℝ) :
  (∀ t, v t = t^2 - 8*t + 3) →
  (∀ t, (deriv s) t = v t) →
  ∃ C, ∀ t, s t = t^3/3 - 4*t^2 + 3*t + C :=
sorry

end motion_equation_l1168_116810


namespace select_and_arrange_five_three_unique_descending_arrangement_select_three_from_five_descending_l1168_116877

/-- The number of ways to select and arrange 3 people from 5 in descending height order -/
def select_and_arrange (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem select_and_arrange_five_three :
  select_and_arrange 5 3 = Nat.choose 5 3 := by
  sorry

/-- The number of ways to arrange 3 people in descending height order -/
def arrange_descending (k : ℕ) : ℕ := 1

theorem unique_descending_arrangement (k : ℕ) :
  arrange_descending k = 1 := by
  sorry

/-- The main theorem: selecting and arranging 3 from 5 equals C(5,3) -/
theorem select_three_from_five_descending :
  select_and_arrange 5 3 = Nat.choose 5 3 := by
  sorry

end select_and_arrange_five_three_unique_descending_arrangement_select_three_from_five_descending_l1168_116877


namespace sum_of_digits_2017_power_l1168_116806

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that S(S(S(S(2017^2017)))) = 1 -/
theorem sum_of_digits_2017_power : S (S (S (S (2017^2017)))) = 1 := by sorry

end sum_of_digits_2017_power_l1168_116806


namespace three_planes_max_parts_l1168_116849

/-- The maximum number of parts into which three planes can divide three-dimensional space -/
def max_parts_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of parts into which three planes can divide three-dimensional space is 8 -/
theorem three_planes_max_parts :
  max_parts_three_planes = 8 := by sorry

end three_planes_max_parts_l1168_116849


namespace neighborhood_cable_cost_l1168_116870

/-- Calculates the total cost of power cable for a neighborhood with the given specifications. -/
theorem neighborhood_cable_cost
  (ew_streets : ℕ)
  (ew_length : ℝ)
  (ns_streets : ℕ)
  (ns_length : ℝ)
  (cable_per_mile : ℝ)
  (cable_cost : ℝ)
  (h1 : ew_streets = 18)
  (h2 : ew_length = 2)
  (h3 : ns_streets = 10)
  (h4 : ns_length = 4)
  (h5 : cable_per_mile = 5)
  (h6 : cable_cost = 2000) :
  (ew_streets * ew_length + ns_streets * ns_length) * cable_per_mile * cable_cost = 760000 := by
  sorry

end neighborhood_cable_cost_l1168_116870


namespace problem_solution_inequality_proof_l1168_116899

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := |x - m|
def g (m : ℝ) (x : ℝ) : ℝ := 2 * f m x - f m (x + m)

-- Theorem statement
theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∃ (x : ℝ), g m x = -1 ∧ ∀ (y : ℝ), g m y ≥ -1) ↔ m = 1 :=
sorry

theorem inequality_proof (m : ℝ) (h_m : m > 0) 
  (a b : ℝ) (h_a : |a| < m) (h_b : |b| < m) (h_a_neq_0 : a ≠ 0) :
  |a * b - m| > |a| * |b / a - m| :=
sorry

end problem_solution_inequality_proof_l1168_116899


namespace no_strictly_monotonic_pair_l1168_116875

theorem no_strictly_monotonic_pair :
  ¬∃ (f g : ℕ → ℕ),
    (∀ x y, x < y → f x < f y) ∧
    (∀ x y, x < y → g x < g y) ∧
    (∀ n, f (g (g n)) < g (f n)) :=
by sorry

end no_strictly_monotonic_pair_l1168_116875


namespace cube_root_equation_solution_l1168_116824

theorem cube_root_equation_solution :
  ∀ x : ℝ, (((5 - x / 3) ^ (1/3 : ℝ) = -2) ↔ (x = 39)) :=
by sorry

end cube_root_equation_solution_l1168_116824


namespace trig_simplification_l1168_116847

theorem trig_simplification (x y : ℝ) :
  (Real.cos (x + π/4))^2 + (Real.cos (x + y + π/2))^2 - 
  2 * Real.cos (x + π/4) * Real.cos (y + π/4) * Real.cos (x + y + π/2) = 1 := by
  sorry

end trig_simplification_l1168_116847


namespace inequality_system_integer_solutions_l1168_116829

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (5 * x + 1 > 3 * (x - 1)) ∧ ((x - 1) / 2 ≥ 2 * x - 4)}
  S = {-1, 0, 1, 2} := by
  sorry

end inequality_system_integer_solutions_l1168_116829


namespace abs_neg_one_third_l1168_116865

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end abs_neg_one_third_l1168_116865


namespace gas_bill_calculation_l1168_116886

/-- Represents the household bills and payments -/
structure HouseholdBills where
  electricity : ℕ
  water : ℕ
  internet : ℕ
  gas : ℕ
  gasPaidFraction : ℚ
  gasAdditionalPayment : ℕ
  remainingPayment : ℕ

/-- Theorem stating that given the household bill conditions, the gas bill is $120 -/
theorem gas_bill_calculation (bills : HouseholdBills) 
  (h1 : bills.electricity = 60)
  (h2 : bills.water = 40)
  (h3 : bills.internet = 25)
  (h4 : bills.gasPaidFraction = 3/4)
  (h5 : bills.gasAdditionalPayment = 5)
  (h6 : bills.remainingPayment = 30)
  (h7 : bills.water / 2 + (bills.internet - 4 * 5) + (bills.gas * (1 - bills.gasPaidFraction) - bills.gasAdditionalPayment) = bills.remainingPayment) :
  bills.gas = 120 := by
  sorry

#check gas_bill_calculation

end gas_bill_calculation_l1168_116886


namespace systematic_sampling_interval_l1168_116859

theorem systematic_sampling_interval 
  (total_items : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_items = 2005)
  (h2 : sample_size = 20) :
  ∃ (removed : ℕ) (interval : ℕ),
    removed < sample_size ∧
    interval * sample_size = total_items - removed ∧
    interval = 100 := by
  sorry

end systematic_sampling_interval_l1168_116859


namespace job_completion_time_l1168_116819

/-- Given two workers A and B, where A completes a job in 10 days and B completes it in 6 days,
    prove that they can complete the job together in 3.75 days. -/
theorem job_completion_time (a_time b_time combined_time : ℝ) 
  (ha : a_time = 10) 
  (hb : b_time = 6) 
  (hc : combined_time = (a_time * b_time) / (a_time + b_time)) : 
  combined_time = 3.75 := by
  sorry

end job_completion_time_l1168_116819


namespace bakers_new_cakes_l1168_116856

/-- Baker's cake problem -/
theorem bakers_new_cakes 
  (initial_cakes : ℕ) 
  (sold_initial : ℕ) 
  (sold_difference : ℕ) 
  (h1 : initial_cakes = 170)
  (h2 : sold_initial = 78)
  (h3 : sold_difference = 47)
  : ∃ (new_cakes : ℕ), 
    sold_initial + sold_difference = new_cakes + sold_difference ∧ 
    new_cakes = 78 :=
by sorry

end bakers_new_cakes_l1168_116856


namespace sum_of_distances_bound_l1168_116828

/-- A rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ
  length_ge_width : length ≥ width

/-- A point inside a rectangle -/
structure PointInRectangle (rect : Rectangle) where
  x : ℝ
  y : ℝ
  x_bounds : 0 ≤ x ∧ x ≤ rect.length
  y_bounds : 0 ≤ y ∧ y ≤ rect.width

/-- The sum of distances from a point to the extensions of all sides of a rectangle -/
def sum_of_distances (rect : Rectangle) (p : PointInRectangle rect) : ℝ :=
  p.x + (rect.length - p.x) + p.y + (rect.width - p.y)

/-- The theorem stating that the sum of distances is at most 2l + 2w -/
theorem sum_of_distances_bound (rect : Rectangle) (p : PointInRectangle rect) :
  sum_of_distances rect p ≤ 2 * rect.length + 2 * rect.width := by
  sorry


end sum_of_distances_bound_l1168_116828


namespace perpendicular_line_equation_l1168_116896

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation y = -2x + 1 in slope-intercept form. -/
theorem perpendicular_line_equation (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 6 * y = 9
  let P : ℝ × ℝ := (2, -3)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = -2 * x + 1
  (∀ x y, L1 x y ↔ y = (1/2) * x - (3/2)) →
  (L2 P.1 P.2) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (y₂ - y₁) * (x₂ - x₁) = -1 / ((y₂ - y₁) / (x₂ - x₁))) →
  ∀ x y, L2 x y ↔ y = -2 * x + 1 :=
by sorry


end perpendicular_line_equation_l1168_116896


namespace quadratic_two_distinct_roots_l1168_116871

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 + (4*m + 1)*x₁ + m = 0) ∧
  (x₂^2 + (4*m + 1)*x₂ + m = 0) :=
sorry

end quadratic_two_distinct_roots_l1168_116871


namespace max_candies_is_18_l1168_116851

/-- Represents the candy store's pricing structure and discount policy. -/
structure CandyStore where
  individual_price : ℕ
  pack4_price : ℕ
  pack7_price : ℕ
  double_pack7_discount : ℕ

/-- Calculates the maximum number of candies that can be bought with a given amount of money. -/
def max_candies (store : CandyStore) (budget : ℕ) : ℕ :=
  sorry

/-- The theorem states that with $25 and the given pricing structure, 
    the maximum number of candies that can be bought is 18. -/
theorem max_candies_is_18 : 
  let store : CandyStore := {
    individual_price := 2,
    pack4_price := 6,
    pack7_price := 10,
    double_pack7_discount := 3
  }
  max_candies store 25 = 18 := by
  sorry

end max_candies_is_18_l1168_116851


namespace alice_above_quota_l1168_116811

def alice_sales (adidas_price nike_price reebok_price : ℕ)
                (adidas_qty nike_qty reebok_qty : ℕ)
                (quota : ℕ) : ℤ :=
  (adidas_price * adidas_qty + nike_price * nike_qty + reebok_price * reebok_qty) - quota

theorem alice_above_quota :
  alice_sales 45 60 35 6 8 9 1000 = 65 := by
  sorry

end alice_above_quota_l1168_116811


namespace ratio_transitive_l1168_116812

theorem ratio_transitive (a b c : ℝ) 
  (h1 : a / b = 7 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 7 / 5 := by
  sorry

end ratio_transitive_l1168_116812


namespace cookie_circle_radius_l1168_116876

theorem cookie_circle_radius (x y : ℝ) :
  (∃ (h k r : ℝ), ∀ x y : ℝ, x^2 + y^2 - 12*x + 16*y + 64 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) →
  (∃ (r : ℝ), r = 6 ∧ ∀ x y : ℝ, x^2 + y^2 - 12*x + 16*y + 64 = 0 ↔ (x - 6)^2 + (y + 8)^2 = r^2) :=
by sorry

end cookie_circle_radius_l1168_116876


namespace reciprocal_of_negative_two_thirds_l1168_116835

theorem reciprocal_of_negative_two_thirds :
  let x : ℚ := -2/3
  let reciprocal (y : ℚ) : ℚ := if y ≠ 0 then 1 / y else 0
  reciprocal x = -3/2 := by
sorry

end reciprocal_of_negative_two_thirds_l1168_116835


namespace square_sum_value_l1168_116890

theorem square_sum_value (x y : ℝ) (h : 5 * x^2 + y^2 - 4*x*y + 24 ≤ 10*x - 1) : x^2 + y^2 = 125 := by
  sorry

end square_sum_value_l1168_116890


namespace set_intersection_range_l1168_116898

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}
def B : Set ℝ := {x | x^2 + 4*x = 0}

-- Define the theorem
theorem set_intersection_range (a : ℝ) :
  A a ∩ B = A a → (a ≤ -1 ∨ a = 1) :=
by sorry

end set_intersection_range_l1168_116898


namespace circle_center_correct_l1168_116840

/-- The equation of a circle in the form x^2 - 2ax + y^2 - 2by + c = 0 -/
def CircleEquation (a b c : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 - 2*a*x + y^2 - 2*b*y + c = 0

/-- The center of a circle given by its equation -/
def CircleCenter (a b c : ℝ) : ℝ × ℝ := (a, b)

theorem circle_center_correct (x y : ℝ) :
  CircleEquation 1 2 (-28) x y → CircleCenter 1 2 (-28) = (1, 2) := by
  sorry

end circle_center_correct_l1168_116840


namespace total_green_is_seven_l1168_116858

/-- The number of green marbles Sara has -/
def sara_green : ℕ := 3

/-- The number of green marbles Tom has -/
def tom_green : ℕ := 4

/-- The total number of green marbles Sara and Tom have together -/
def total_green : ℕ := sara_green + tom_green

/-- Theorem stating that the total number of green marbles is 7 -/
theorem total_green_is_seven : total_green = 7 := by
  sorry

end total_green_is_seven_l1168_116858
