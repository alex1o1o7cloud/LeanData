import Mathlib

namespace NUMINAMATH_GPT_exists_line_with_two_colors_l642_64210

open Classical

/-- Given a grid with 1x1 squares where each vertex is painted one of four colors such that each 1x1 square's vertices are all different colors, 
    there exists a line in the grid with nodes of exactly two different colors. -/
theorem exists_line_with_two_colors 
  (A : Type)
  [Inhabited A]
  [DecidableEq A]
  (colors : Finset A) 
  (h_col : colors.card = 4) 
  (grid : ℤ × ℤ → A) 
  (h_diff_colors : ∀ (i j : ℤ), i ≠ j → ∀ (k l : ℤ), grid (i, k) ≠ grid (j, k) ∧ grid (i, l) ≠ grid (i, k)) :
  ∃ line : ℤ → ℤ × ℤ, ∃ a b : A, a ≠ b ∧ ∀ n : ℤ, grid (line n) = a ∨ grid (line n) = b :=
sorry

end NUMINAMATH_GPT_exists_line_with_two_colors_l642_64210


namespace NUMINAMATH_GPT_max_integer_value_of_f_l642_64293

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 8 * x + 21) / (4 * x^2 + 8 * x + 5)

theorem max_integer_value_of_f :
  ∃ n : ℤ, n = 17 ∧ ∀ x : ℝ, f x ≤ (n : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_max_integer_value_of_f_l642_64293


namespace NUMINAMATH_GPT_hens_ratio_l642_64228

theorem hens_ratio
  (total_chickens : ℕ)
  (fraction_roosters : ℚ)
  (chickens_not_laying : ℕ)
  (h : total_chickens = 80)
  (fr : fraction_roosters = 1/4)
  (cnl : chickens_not_laying = 35) :
  (total_chickens * (1 - fraction_roosters) - chickens_not_laying) / (total_chickens * (1 - fraction_roosters)) = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_hens_ratio_l642_64228


namespace NUMINAMATH_GPT_number_of_sets_A_l642_64297

/-- Given conditions about intersections and unions of set A, we want to find the number of 
  possible sets A that satisfy the given conditions. Specifically, prove the following:
  - A ∩ {-1, 0, 1} = {0, 1}
  - A ∪ {-2, 0, 2} = {-2, 0, 1, 2}
  Total number of such sets A is 4.
-/
theorem number_of_sets_A : ∃ (As : Finset (Finset ℤ)), 
  (∀ A ∈ As, A ∩ {-1, 0, 1} = {0, 1} ∧ A ∪ {-2, 0, 2} = {-2, 0, 1, 2}) ∧
  As.card = 4 := 
sorry

end NUMINAMATH_GPT_number_of_sets_A_l642_64297


namespace NUMINAMATH_GPT_marching_band_members_l642_64227

theorem marching_band_members (B W P : ℕ) (h1 : P = 4 * W) (h2 : W = 2 * B) (h3 : B = 10) : B + W + P = 110 :=
by
  sorry

end NUMINAMATH_GPT_marching_band_members_l642_64227


namespace NUMINAMATH_GPT_bc_fraction_ad_l642_64231

theorem bc_fraction_ad
  (B C E A D : Type)
  (on_AD : ∀ P : Type, P = B ∨ P = C ∨ P = E)
  (AB BD AC CD DE EA: ℝ)
  (h1 : AB = 3 * BD)
  (h2 : AC = 5 * CD)
  (h3 : DE = 2 * EA)

  : ∃ BC AD: ℝ, BC = 1 / 12 * AD := 
sorry -- Proof is omitted

end NUMINAMATH_GPT_bc_fraction_ad_l642_64231


namespace NUMINAMATH_GPT_bobs_password_probability_l642_64246

theorem bobs_password_probability :
  (5 / 10) * (5 / 10) * 1 * (9 / 10) = 9 / 40 :=
by
  sorry

end NUMINAMATH_GPT_bobs_password_probability_l642_64246


namespace NUMINAMATH_GPT_digit_makes_divisible_by_nine_l642_64229

theorem digit_makes_divisible_by_nine (A : ℕ) : (7 + A + 4 + 6) % 9 = 0 ↔ A = 1 :=
by
  sorry

end NUMINAMATH_GPT_digit_makes_divisible_by_nine_l642_64229


namespace NUMINAMATH_GPT_points_on_line_any_real_n_l642_64260

theorem points_on_line_any_real_n (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 1 = 2 * (n + 0.5) + 5) : 
  True :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_any_real_n_l642_64260


namespace NUMINAMATH_GPT_find_positive_integers_l642_64203

theorem find_positive_integers (a b c : ℕ) (ha : a ≥ b) (hb : b ≥ c) :
  (∃ n₁ : ℕ, a^2 + 3 * b = n₁^2) ∧ 
  (∃ n₂ : ℕ, b^2 + 3 * c = n₂^2) ∧ 
  (∃ n₃ : ℕ, c^2 + 3 * a = n₃^2) →
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 37 ∧ b = 25 ∧ c = 17) :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integers_l642_64203


namespace NUMINAMATH_GPT_problemStatement_l642_64207

-- Define the set of values as a type
structure SetOfValues where
  k : ℤ
  b : ℤ

-- The given sets of values
def A : SetOfValues := ⟨2, 2⟩
def B : SetOfValues := ⟨2, -2⟩
def C : SetOfValues := ⟨-2, -2⟩
def D : SetOfValues := ⟨-2, 2⟩

-- Define the conditions for the function
def isValidSet (s : SetOfValues) : Prop :=
  s.k < 0 ∧ s.b > 0

-- The problem statement: Prove that D is a valid set
theorem problemStatement : isValidSet D := by
  sorry

end NUMINAMATH_GPT_problemStatement_l642_64207


namespace NUMINAMATH_GPT_increase_speed_to_pass_correctly_l642_64230

theorem increase_speed_to_pass_correctly
  (x a : ℝ)
  (ha1 : 50 < a)
  (hx1 : (a - 40) * x = 30)
  (hx2 : (a + 50) * x = 210) :
  a - 50 = 5 :=
by
  sorry

end NUMINAMATH_GPT_increase_speed_to_pass_correctly_l642_64230


namespace NUMINAMATH_GPT_range_of_m_l642_64282

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (m x : ℝ) : ℝ := m * x + 1
noncomputable def h (x : ℝ) : ℝ := (1 / x) - (2 * Real.log x / x)

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 2)) ∧ (g m x = 2 - 2 * f x)) ↔
  (-2 * Real.exp (-3/2) ≤ m ∧ m ≤ 3 * Real.exp 1) :=
sorry

end NUMINAMATH_GPT_range_of_m_l642_64282


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_condition_l642_64215

variable (a : ℕ → ℤ)

theorem arithmetic_sequence_sum_condition (h1 : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) : 
  a 2 + a 10 = 120 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_condition_l642_64215


namespace NUMINAMATH_GPT_valid_third_side_l642_64274

theorem valid_third_side (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 8) (h₃ : 5 < c) (h₄ : c < 11) : c = 8 := 
by 
  sorry

end NUMINAMATH_GPT_valid_third_side_l642_64274


namespace NUMINAMATH_GPT_ratio_M_N_l642_64277

variable (M Q P N : ℝ)

-- Conditions
axiom h1 : M = 0.40 * Q
axiom h2 : Q = 0.25 * P
axiom h3 : N = 0.60 * P

theorem ratio_M_N : M / N = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_M_N_l642_64277


namespace NUMINAMATH_GPT_sum_of_four_numbers_in_ratio_is_correct_l642_64209

variable (A B C D : ℝ)
variable (h_ratio : A / B = 2 / 3 ∧ B / C = 3 / 4 ∧ C / D = 4 / 5)
variable (h_biggest : D = 672)

theorem sum_of_four_numbers_in_ratio_is_correct :
  A + B + C + D = 1881.6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_four_numbers_in_ratio_is_correct_l642_64209


namespace NUMINAMATH_GPT_rubies_in_chest_l642_64280

theorem rubies_in_chest (R : ℕ) (h₁ : 421 = R + 44) : R = 377 :=
by 
  sorry

end NUMINAMATH_GPT_rubies_in_chest_l642_64280


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l642_64256

-- Given conditions
variable {T : Type} -- Type representing teachers
variable {S : Type} -- Type representing students

def arrangements_ends (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_next_to_each_other (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_not_next_to_each_other (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_two_between (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

-- Statements to prove

-- 1. Prove that if teachers A and B must stand at the two ends, there are 48 different arrangements
theorem problem_1 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_ends teachers students = 48 :=
  sorry

-- 2. Prove that if teachers A and B must stand next to each other, there are 240 different arrangements
theorem problem_2 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_next_to_each_other teachers students = 240 :=
  sorry 

-- 3. Prove that if teachers A and B cannot stand next to each other, there are 480 different arrangements
theorem problem_3 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_not_next_to_each_other teachers students = 480 :=
  sorry 

-- 4. Prove that if there must be two students standing between teachers A and B, there are 144 different arrangements
theorem problem_4 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_two_between teachers students = 144 :=
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l642_64256


namespace NUMINAMATH_GPT_intersection_point_at_neg4_l642_64242

def f (x : Int) (b : Int) : Int := 4 * x + b
def f_inv (y : Int) (b : Int) : Int := (y - b) / 4

theorem intersection_point_at_neg4 (a b : Int) (h1 : f (-4) b = a) (h2 : f_inv (-4) b = a) : a = -4 := 
by 
  sorry

end NUMINAMATH_GPT_intersection_point_at_neg4_l642_64242


namespace NUMINAMATH_GPT_find_P_coordinates_l642_64290

-- Given points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 2)

-- The area of triangle PAB is 5
def areaPAB (P : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (A.2 - B.2) + A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2))

-- Point P lies on the x-axis
def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

theorem find_P_coordinates (P : ℝ × ℝ) :
  on_x_axis P → areaPAB P = 5 → (P = (-4, 0) ∨ P = (6, 0)) :=
by
  sorry

end NUMINAMATH_GPT_find_P_coordinates_l642_64290


namespace NUMINAMATH_GPT_limit_r_l642_64211

noncomputable def L (m : ℝ) : ℝ := (m - Real.sqrt (m^2 + 24)) / 2

noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

theorem limit_r (h : ∀ m : ℝ, m ≠ 0) : Filter.Tendsto r (nhds 0) (nhds (-1)) :=
sorry

end NUMINAMATH_GPT_limit_r_l642_64211


namespace NUMINAMATH_GPT_daily_production_n_l642_64219

theorem daily_production_n (n : ℕ) 
  (h1 : (60 * n) / n = 60)
  (h2 : (60 * n + 90) / (n + 1) = 65) : 
  n = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_daily_production_n_l642_64219


namespace NUMINAMATH_GPT_angle_value_l642_64222

theorem angle_value (x : ℝ) (h₁ : (90 : ℝ) = 44 + x) : x = 46 :=
by
  sorry

end NUMINAMATH_GPT_angle_value_l642_64222


namespace NUMINAMATH_GPT_square_diagonal_l642_64243

theorem square_diagonal (P : ℝ) (d : ℝ) (hP : P = 200 * Real.sqrt 2) :
  d = 100 :=
by
  sorry

end NUMINAMATH_GPT_square_diagonal_l642_64243


namespace NUMINAMATH_GPT_carbon_neutrality_l642_64234

theorem carbon_neutrality (a b : ℝ) (t : ℕ) (ha : a > 0)
  (h1 : S = a * b ^ t)
  (h2 : a * b ^ 7 = 4 * a / 5)
  (h3 : a / 4 = S) :
  t = 42 := 
sorry

end NUMINAMATH_GPT_carbon_neutrality_l642_64234


namespace NUMINAMATH_GPT_factorize_a_squared_plus_2a_l642_64296

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2 * a = a * (a + 2) :=
  sorry

end NUMINAMATH_GPT_factorize_a_squared_plus_2a_l642_64296


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l642_64233

-- Statement for the first equation: x^2 - 16 = 0
theorem solve_equation1 (x : ℝ) : x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 :=
by sorry

-- Statement for the second equation: (x + 10)^3 + 27 = 0
theorem solve_equation2 (x : ℝ) : (x + 10)^3 + 27 = 0 ↔ x = -13 :=
by sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l642_64233


namespace NUMINAMATH_GPT_part1_part2_l642_64206
noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

theorem part1 : {x : ℝ | f x ≥ 3} = {x | x ≤ 0} ∪ {x | x ≥ 3} :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x ≤ -a^2 + a + 7) ↔ -2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l642_64206


namespace NUMINAMATH_GPT_partI_inequality_partII_inequality_l642_64259

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Part (Ⅰ): Prove f(x) ≤ x + 1 for 1 ≤ x ≤ 5
theorem partI_inequality (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 5) : f x ≤ x + 1 := by
  sorry

-- Part (Ⅱ): Prove (a^2)/(a+1) + (b^2)/(b+1) ≥ 1 when a + b = 2 and a > 0, b > 0
theorem partII_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) : 
    (a^2) / (a + 1) + (b^2) / (b + 1) ≥ 1 := by
  sorry

end NUMINAMATH_GPT_partI_inequality_partII_inequality_l642_64259


namespace NUMINAMATH_GPT_tan_half_prod_eq_sqrt3_l642_64205

theorem tan_half_prod_eq_sqrt3 (a b : ℝ) (h : 7 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ (xy : ℝ), xy = Real.tan (a / 2) * Real.tan (b / 2) ∧ (xy = Real.sqrt 3 ∨ xy = -Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_tan_half_prod_eq_sqrt3_l642_64205


namespace NUMINAMATH_GPT_length_ab_is_constant_l642_64285

noncomputable def length_AB_constant (p : ℝ) (hp : p > 0) : Prop :=
  let parabola := { P : ℝ × ℝ | P.1 ^ 2 = 2 * p * P.2 }
  let line := { P : ℝ × ℝ | P.2 = P.1 + p / 2 }
  (∃ A B : ℝ × ℝ, A ∈ parabola ∧ B ∈ parabola ∧ A ∈ line ∧ B ∈ line ∧ 
    dist A B = 4 * p)

theorem length_ab_is_constant (p : ℝ) (hp : p > 0) : length_AB_constant p hp :=
by {
  sorry
}

end NUMINAMATH_GPT_length_ab_is_constant_l642_64285


namespace NUMINAMATH_GPT_find_a2_a3_sequence_constant_general_formula_l642_64251

-- Definition of the sequence and its sum Sn
variables (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
axiom a1_eq : a 1 = 2
axiom S_eq : ∀ n, S (n + 1) = 4 * a n - 2

-- Prove that a_2 = 4 and a_3 = 8
theorem find_a2_a3 : a 2 = 4 ∧ a 3 = 8 :=
sorry

-- Prove that the sequence {a_n - 2a_{n-1}} is constant
theorem sequence_constant {n : ℕ} (hn : n ≥ 2) :
  ∃ c, ∀ k ≥ 2, a k - 2 * a (k - 1) = c :=
sorry

-- Find the general formula for the sequence
theorem general_formula :
  ∀ n, a n = 2^n :=
sorry

end NUMINAMATH_GPT_find_a2_a3_sequence_constant_general_formula_l642_64251


namespace NUMINAMATH_GPT_find_sum_l642_64272

variable {x y z w : ℤ}

-- Conditions: Consecutive integers and their sum condition
def consecutive_integers (x y z : ℤ) : Prop := y = x + 1 ∧ z = x + 2
def sum_is_150 (x y z : ℤ) : Prop := x + y + z = 150
def w_definition (w z x : ℤ) : Prop := w = 2 * z - x

-- Theorem statement
theorem find_sum (h1 : consecutive_integers x y z) (h2 : sum_is_150 x y z) (h3 : w_definition w z x) :
  x + y + z + w = 203 :=
sorry

end NUMINAMATH_GPT_find_sum_l642_64272


namespace NUMINAMATH_GPT_age_product_difference_l642_64245

theorem age_product_difference 
  (age_today : ℕ) 
  (Arnold_age : age_today = 6) 
  (Danny_age : age_today = 6) : 
  (7 * 7) - (6 * 6) = 13 := 
by
  sorry

end NUMINAMATH_GPT_age_product_difference_l642_64245


namespace NUMINAMATH_GPT_longest_side_length_l642_64248

-- Define the sides of the triangle
def side_a : ℕ := 9
def side_b (x : ℕ) : ℕ := 2 * x + 3
def side_c (x : ℕ) : ℕ := 3 * x - 2

-- Define the perimeter condition
def perimeter_condition (x : ℕ) : Prop := side_a + side_b x + side_c x = 45

-- Main theorem statement: Length of the longest side is 19
theorem longest_side_length (x : ℕ) (h : perimeter_condition x) : side_b x = 19 ∨ side_c x = 19 :=
sorry

end NUMINAMATH_GPT_longest_side_length_l642_64248


namespace NUMINAMATH_GPT_range_of_a_l642_64283

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_a (h_odd : ∀ x, f (-x) = -f x) 
  (h_period : ∀ x, f (x + 3) = f x)
  (h1 : f 1 > 1) 
  (h2018 : f 2018 = (a : ℝ) ^ 2 - 5) : 
  -2 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l642_64283


namespace NUMINAMATH_GPT_find_triangle_sides_l642_64276

variable (a b c : ℕ)
variable (P : ℕ)
variable (R : ℚ := 65 / 8)
variable (r : ℕ := 4)

theorem find_triangle_sides (h1 : R = 65 / 8) (h2 : r = 4) (h3 : P = a + b + c) : 
  a = 13 ∧ b = 14 ∧ c = 15 :=
  sorry

end NUMINAMATH_GPT_find_triangle_sides_l642_64276


namespace NUMINAMATH_GPT_real_solutions_count_is_two_l642_64266

def equation_has_two_real_solutions (a b c : ℝ) : Prop :=
  (3*a^2 - 8*b + 2 = c) → (∀ x : ℝ, 3*x^2 - 8*x + 2 = 0) → ∃! x₁ x₂ : ℝ, (3*x₁^2 - 8*x₁ + 2 = 0) ∧ (3*x₂^2 - 8*x₂ + 2 = 0)

theorem real_solutions_count_is_two : equation_has_two_real_solutions (3 : ℝ) (-8 : ℝ) (2 : ℝ) := by
  sorry

end NUMINAMATH_GPT_real_solutions_count_is_two_l642_64266


namespace NUMINAMATH_GPT_nuts_per_student_l642_64292

theorem nuts_per_student (bags : ℕ) (nuts_per_bag : ℕ) (students : ℕ) (total_nuts : ℕ) (nuts_per_student : ℕ)
    (h1 : bags = 65)
    (h2 : nuts_per_bag = 15)
    (h3 : students = 13)
    (h4 : total_nuts = bags * nuts_per_bag)
    (h5 : nuts_per_student = total_nuts / students)
    : nuts_per_student = 75 :=
by
  sorry

end NUMINAMATH_GPT_nuts_per_student_l642_64292


namespace NUMINAMATH_GPT_cheapest_lamp_cost_l642_64275

/--
Frank wants to buy a new lamp for his bedroom. The cost of the cheapest lamp is some amount, and the most expensive in the store is 3 times more expensive. Frank has $90, and if he buys the most expensive lamp available, he would have $30 remaining. Prove that the cost of the cheapest lamp is $20.
-/
theorem cheapest_lamp_cost (c most_expensive : ℝ) (h_cheapest_lamp : most_expensive = 3 * c) 
(h_frank_money : 90 - most_expensive = 30) : c = 20 := 
sorry

end NUMINAMATH_GPT_cheapest_lamp_cost_l642_64275


namespace NUMINAMATH_GPT_area_of_shaded_trapezoid_l642_64250

-- Definitions of conditions:
def side_lengths : List ℕ := [1, 3, 5, 7]
def total_base : ℕ := side_lengths.sum
def height_largest_square : ℕ := 7
def ratio : ℚ := height_largest_square / total_base

def height_at_end (n : ℕ) : ℚ := ratio * n
def lower_base_height : ℚ := height_at_end 4
def upper_base_height : ℚ := height_at_end 9
def trapezoid_height : ℕ := 2

-- Main theorem:
theorem area_of_shaded_trapezoid :
  (1 / 2) * (lower_base_height + upper_base_height) * trapezoid_height = 91 / 8 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_trapezoid_l642_64250


namespace NUMINAMATH_GPT_quadrilateral_centroid_perimeter_l642_64217

-- Definition for the side length of the square and distances for points Q
def side_length : ℝ := 40
def EQ_dist : ℝ := 18
def FQ_dist : ℝ := 34

-- Theorem statement: Perimeter of the quadrilateral formed by centroids
theorem quadrilateral_centroid_perimeter :
  let centroid_perimeter := (4 * ((2 / 3) * side_length))
  centroid_perimeter = (320 / 3) := by
  sorry

end NUMINAMATH_GPT_quadrilateral_centroid_perimeter_l642_64217


namespace NUMINAMATH_GPT_dan_has_13_limes_l642_64202

theorem dan_has_13_limes (picked_limes : ℕ) (given_limes : ℕ) (h1 : picked_limes = 9) (h2 : given_limes = 4) : 
  picked_limes + given_limes = 13 := 
by
  sorry

end NUMINAMATH_GPT_dan_has_13_limes_l642_64202


namespace NUMINAMATH_GPT_cost_of_first_10_kgs_of_apples_l642_64287

theorem cost_of_first_10_kgs_of_apples 
  (l q : ℝ) 
  (h1 : 30 * l + 3 * q = 663) 
  (h2 : 30 * l + 6 * q = 726) : 
  10 * l = 200 :=
by
  -- Proof would follow here
  sorry

end NUMINAMATH_GPT_cost_of_first_10_kgs_of_apples_l642_64287


namespace NUMINAMATH_GPT_values_of_t_l642_64299

theorem values_of_t (x y z t : ℝ) 
  (h1 : 3 * x^2 + 3 * x * z + z^2 = 1)
  (h2 : 3 * y^2 + 3 * y * z + z^2 = 4)
  (h3 : x^2 - x * y + y^2 = t) : 
  t ≤ 10 :=
sorry

end NUMINAMATH_GPT_values_of_t_l642_64299


namespace NUMINAMATH_GPT_quadratic_coeff_sum_l642_64254

theorem quadratic_coeff_sum {a b c : ℝ} (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 1) * (x - 5))
    (h2 : a * 3^2 + b * 3 + c = 36) : a + b + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_coeff_sum_l642_64254


namespace NUMINAMATH_GPT_meaningful_expression_l642_64214

theorem meaningful_expression (x : ℝ) : (∃ y, y = 5 / (Real.sqrt (x + 1))) ↔ x > -1 :=
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_l642_64214


namespace NUMINAMATH_GPT_matchstick_problem_l642_64263

theorem matchstick_problem (n : ℕ) (T : ℕ → ℕ) :
  (∀ n, T n = 4 + 9 * (n - 1)) ∧ n = 15 → T n = 151 :=
by
  sorry

end NUMINAMATH_GPT_matchstick_problem_l642_64263


namespace NUMINAMATH_GPT_problem_solution_l642_64218

-- Define the problem
noncomputable def a_b_sum : ℝ := 
  let a := 5
  let b := 3
  a + b

-- Theorem statement
theorem problem_solution (a b i : ℝ) (h1 : a + b * i = (11 - 7 * i) / (1 - 2 * i)) (hi : i * i = -1) :
  a + b = 8 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l642_64218


namespace NUMINAMATH_GPT_quadratic_func_inequality_l642_64298

theorem quadratic_func_inequality (c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 + 4 * x + c)
  (h_increasing : ∀ x y, x ≤ y → -2 ≤ x → f x ≤ f y) :
  f 1 > f 0 ∧ f 0 > f (-2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_func_inequality_l642_64298


namespace NUMINAMATH_GPT_contradiction_proof_l642_64236

theorem contradiction_proof (a b : ℝ) : a + b = 12 → ¬ (a < 6 ∧ b < 6) :=
by
  intro h
  intro h_contra
  sorry

end NUMINAMATH_GPT_contradiction_proof_l642_64236


namespace NUMINAMATH_GPT_order_of_logs_l642_64288

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 16 / Real.log 8

theorem order_of_logs : a > b ∧ b > c := 
by
  sorry

end NUMINAMATH_GPT_order_of_logs_l642_64288


namespace NUMINAMATH_GPT_true_and_false_propositions_l642_64232

theorem true_and_false_propositions (p q : Prop) 
  (hp : p = true) (hq : q = false) : (¬q) = true :=
by
  sorry

end NUMINAMATH_GPT_true_and_false_propositions_l642_64232


namespace NUMINAMATH_GPT_value_of_7x_minus_3y_l642_64278

theorem value_of_7x_minus_3y (x y : ℚ) (h1 : 4 * x + y = 8) (h2 : 3 * x - 4 * y = 5) : 
  7 * x - 3 * y = 247 / 19 := 
sorry

end NUMINAMATH_GPT_value_of_7x_minus_3y_l642_64278


namespace NUMINAMATH_GPT_common_chord_length_common_chord_diameter_eq_circle_l642_64291

/-
Given two circles C1: x^2 + y^2 - 2x + 10y - 24 = 0 and C2: x^2 + y^2 + 2x + 2y - 8 = 0,
prove that 
1. The length of the common chord is 2 * sqrt(5).
2. The equation of the circle that has the common chord as its diameter is (x + 8/5)^2 + (y - 6/5)^2 = 36/5.
-/

-- Define the first circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 10 * y - 24 = 0

-- Define the second circle C2
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Prove the length of the common chord
theorem common_chord_length : ∃ d : ℝ, d = 2 * Real.sqrt 5 :=
sorry

-- Prove the equation of the circle that has the common chord as its diameter
theorem common_chord_diameter_eq_circle : ∃ (x y : ℝ → ℝ), (x + 8/5)^2 + (y - 6/5)^2 = 36/5 :=
sorry

end NUMINAMATH_GPT_common_chord_length_common_chord_diameter_eq_circle_l642_64291


namespace NUMINAMATH_GPT_Ben_ate_25_percent_of_cake_l642_64216

theorem Ben_ate_25_percent_of_cake (R B : ℕ) (h_ratio : R / B = 3 / 1) : B / (R + B) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_Ben_ate_25_percent_of_cake_l642_64216


namespace NUMINAMATH_GPT_initial_percentage_alcohol_l642_64253

variables (P : ℝ) (initial_volume : ℝ) (added_volume : ℝ) (total_volume : ℝ) (final_percentage : ℝ) (init_percentage : ℝ)

theorem initial_percentage_alcohol (h1 : initial_volume = 6)
                                  (h2 : added_volume = 3)
                                  (h3 : total_volume = initial_volume + added_volume)
                                  (h4 : final_percentage = 50)
                                  (h5 : init_percentage = 100 * (initial_volume * P / 100 + added_volume) / total_volume)
                                  : P = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_percentage_alcohol_l642_64253


namespace NUMINAMATH_GPT_compounding_frequency_l642_64201

variable (i : ℝ) (EAR : ℝ)

/-- Given the nominal annual rate (i = 6%) and the effective annual rate (EAR = 6.09%), 
    prove that the frequency of payment (n) is 4. -/
theorem compounding_frequency (h1 : i = 0.06) (h2 : EAR = 0.0609) : 
  ∃ n : ℕ, (1 + i / n)^n - 1 = EAR ∧ n = 4 := sorry

end NUMINAMATH_GPT_compounding_frequency_l642_64201


namespace NUMINAMATH_GPT_solve_for_n_l642_64223

theorem solve_for_n (n : ℕ) (h : 9^n * 9^n * 9^n * 9^n = 81^n) : n = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l642_64223


namespace NUMINAMATH_GPT_simultaneous_equations_in_quadrant_I_l642_64284

theorem simultaneous_equations_in_quadrant_I (c : ℝ) :
  (∃ x y : ℝ, x - y = 3 ∧ c * x + y = 4 ∧ x > 0 ∧ y > 0) ↔ (-1 < c ∧ c < 4 / 3) :=
  sorry

end NUMINAMATH_GPT_simultaneous_equations_in_quadrant_I_l642_64284


namespace NUMINAMATH_GPT_time_after_9876_seconds_l642_64237

-- Define the initial time in seconds
def initial_seconds : ℕ := 6 * 3600

-- Define the elapsed time in seconds
def elapsed_seconds : ℕ := 9876

-- Convert given time in seconds to hours, minutes, and seconds
def time_in_hms (total_seconds : ℕ) : (ℕ × ℕ × ℕ) :=
  let hours := total_seconds / 3600
  let minutes := (total_seconds % 3600) / 60
  let seconds := total_seconds % 60
  (hours, minutes, seconds)

-- Define the final time in 24-hour format (08:44:36)
def final_time : (ℕ × ℕ × ℕ) := (8, 44, 36)

-- The question's proof statement
theorem time_after_9876_seconds : 
  time_in_hms (initial_seconds + elapsed_seconds) = final_time :=
sorry

end NUMINAMATH_GPT_time_after_9876_seconds_l642_64237


namespace NUMINAMATH_GPT_dickens_birth_day_l642_64212

def is_leap_year (year : ℕ) : Prop :=
  (year % 400 = 0) ∨ (year % 4 = 0 ∧ year % 100 ≠ 0)

theorem dickens_birth_day :
  let day_of_week_2012 := 2 -- 0: Sunday, 1: Monday, ..., 2: Tuesday
  let years := 200
  let regular_years := 151
  let leap_years := 49
  let days_shift := regular_years + 2 * leap_years
  let day_of_week_birth := (day_of_week_2012 + days_shift) % 7
  day_of_week_birth = 5 -- 5: Friday
:= 
sorry -- proof not supplied

end NUMINAMATH_GPT_dickens_birth_day_l642_64212


namespace NUMINAMATH_GPT_n_to_power_eight_plus_n_to_power_seven_plus_one_prime_l642_64264

theorem n_to_power_eight_plus_n_to_power_seven_plus_one_prime (n : ℕ) (hn_pos : n > 0) :
  (Nat.Prime (n^8 + n^7 + 1)) → (n = 1) :=
by
  sorry

end NUMINAMATH_GPT_n_to_power_eight_plus_n_to_power_seven_plus_one_prime_l642_64264


namespace NUMINAMATH_GPT_range_of_a_l642_64249

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l642_64249


namespace NUMINAMATH_GPT_interest_after_4_years_l642_64238
-- Importing the necessary library

-- Definitions based on the conditions
def initial_amount : ℝ := 1500
def annual_interest_rate : ℝ := 0.12
def number_of_years : ℕ := 4

-- Calculating the total amount after 4 years using compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- Calculating the interest earned
def interest_earned : ℝ :=
  compound_interest initial_amount annual_interest_rate number_of_years - initial_amount

-- The Lean statement to prove the interest earned is $859.25
theorem interest_after_4_years : interest_earned = 859.25 :=
by
  sorry

end NUMINAMATH_GPT_interest_after_4_years_l642_64238


namespace NUMINAMATH_GPT_vasya_max_consecutive_liked_numbers_l642_64286

def is_liked_by_vasya (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d ≠ 0 → n % d = 0

theorem vasya_max_consecutive_liked_numbers : 
  ∃ (seq : ℕ → ℕ), 
    (∀ n, seq n = n ∧ is_liked_by_vasya (seq n)) ∧
    (∀ m, seq m + 1 < seq (m + 1)) ∧ seq 12 - seq 0 + 1 = 13 :=
sorry

end NUMINAMATH_GPT_vasya_max_consecutive_liked_numbers_l642_64286


namespace NUMINAMATH_GPT_max_marks_l642_64247

theorem max_marks (M : ℝ) : 0.33 * M = 59 + 40 → M = 300 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l642_64247


namespace NUMINAMATH_GPT_tangent_line_parabola_l642_64269

theorem tangent_line_parabola (a : ℝ) :
  (∀ x y : ℝ, y^2 = 32 * x → 4 * x + 3 * y + a = 0) → a = 18 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_parabola_l642_64269


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l642_64244

theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h1 : (a = 2 ∧ b = 4 ∧ c = 4) ∨ (a = 4 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) 
  (h2 : a + b > c ∧ a + c > b ∧ b + c > a) : a + b + c = 10 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l642_64244


namespace NUMINAMATH_GPT_evaluate_expression_l642_64235

variable {x y : ℝ}

theorem evaluate_expression (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 1 / y ^ 2) :
  (x - 1 / x ^ 2) * (y + 2 / y) = 2 * x ^ (5 / 2) - 1 / x := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l642_64235


namespace NUMINAMATH_GPT_greatest_possible_perimeter_l642_64271

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem greatest_possible_perimeter :
  ∃ x : ℕ, 6 ≤ x ∧ x < 17 ∧ is_triangle x (2 * x) 17 ∧ (x + 2 * x + 17 = 65) := by
  sorry

end NUMINAMATH_GPT_greatest_possible_perimeter_l642_64271


namespace NUMINAMATH_GPT_find_positive_integer_solutions_l642_64265

theorem find_positive_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  2^x + 3^y = z^2 ↔ (x = 0 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 0 ∧ z = 3) ∨ (x = 4 ∧ y = 2 ∧ z = 5) := 
sorry

end NUMINAMATH_GPT_find_positive_integer_solutions_l642_64265


namespace NUMINAMATH_GPT_sequence_b_n_l642_64224

theorem sequence_b_n (b : ℕ → ℝ) 
  (h1 : b 1 = 3)
  (h2 : ∀ n ≥ 1, (b (n + 1))^3 = 27 * (b n)^3) :
  b 50 = 3^50 :=
sorry

end NUMINAMATH_GPT_sequence_b_n_l642_64224


namespace NUMINAMATH_GPT_max_value_of_a_plus_b_l642_64289

theorem max_value_of_a_plus_b (a b : ℝ) (h₁ : a^2 + b^2 = 25) (h₂ : a ≤ 3) (h₃ : b ≥ 3) :
  a + b ≤ 7 :=
sorry

end NUMINAMATH_GPT_max_value_of_a_plus_b_l642_64289


namespace NUMINAMATH_GPT_right_triangle_area_l642_64204

theorem right_triangle_area (a : ℝ) (h : a > 2)
  (h_arith_seq : a - 2 > 0)
  (pythagorean : (a - 2)^2 + a^2 = (a + 2)^2) :
  (1 / 2) * (a - 2) * a = 24 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l642_64204


namespace NUMINAMATH_GPT_common_difference_is_4_l642_64220

variable (a : ℕ → ℤ) (d : ℤ)

-- Conditions of the problem
def arithmetic_sequence := ∀ n m : ℕ, a n = a m + (n - m) * d

axiom a7_eq_25 : a 7 = 25
axiom a4_eq_13 : a 4 = 13

-- The theorem to prove
theorem common_difference_is_4 : d = 4 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_is_4_l642_64220


namespace NUMINAMATH_GPT_CALI_area_is_180_l642_64241

-- all the conditions used in Lean definitions
def is_square (s : ℕ) : Prop := (s > 0)

def are_midpoints (T O W N B E R K : ℕ) : Prop := 
  (T = (B + E) / 2) ∧ (O = (E + R) / 2) ∧ (W = (R + K) / 2) ∧ (N = (K + B) / 2)

def is_parallel (CA BO : ℕ) : Prop :=
  CA = BO 

-- the condition indicates the length of each side of the square BERK is 10
def side_length_of_BERK : ℕ := 10

-- definition of lengths and condition
def BERK_lengths (BERK_side_length : ℕ) (BERK_diag_length : ℕ): Prop :=
  BERK_side_length = side_length_of_BERK ∧ BERK_diag_length = BERK_side_length * (2^(1/2))

def CALI_area_of_length (length: ℕ): ℕ := length^2

theorem CALI_area_is_180 
(BERK_side_length BERK_diag_length : ℕ)
(CALI_length : ℕ)
(T O W N B E R K CA BO : ℕ)
(h1 : is_square BERK_side_length)
(h2 : are_midpoints T O W N B E R K)
(h3 : is_parallel CA BO)
(h4 : BERK_lengths BERK_side_length BERK_diag_length)
(h5 : CA = CA)
: CALI_area_of_length 15 = 180 :=
sorry

end NUMINAMATH_GPT_CALI_area_is_180_l642_64241


namespace NUMINAMATH_GPT_cara_optimal_reroll_two_dice_probability_l642_64225

def probability_reroll_two_dice : ℚ :=
  -- Probability derived from Cara's optimal reroll decisions
  5 / 27

theorem cara_optimal_reroll_two_dice_probability :
  cara_probability_optimal_reroll_two_dice = 5 / 27 := by sorry

end NUMINAMATH_GPT_cara_optimal_reroll_two_dice_probability_l642_64225


namespace NUMINAMATH_GPT_intersection_A_B_l642_64208

def A : Set ℝ := {x | x ≤ 2*x + 1 ∧ 2*x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : 
  A ∩ B = {x | 0 < x ∧ x ≤ 2} :=
  sorry

end NUMINAMATH_GPT_intersection_A_B_l642_64208


namespace NUMINAMATH_GPT_cylinder_radius_in_cone_l642_64258

-- Define the conditions
def cone_diameter := 18
def cone_height := 20
def cylinder_height_eq_diameter {r : ℝ} := 2 * r

-- Define the theorem to prove
theorem cylinder_radius_in_cone : ∃ r : ℝ, r = 90 / 19 ∧ (20 - 2 * r) / r = 20 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_radius_in_cone_l642_64258


namespace NUMINAMATH_GPT_correct_mark_l642_64252

theorem correct_mark (x : ℝ) (n : ℝ) (avg_increase : ℝ) :
  n = 40 → avg_increase = 1 / 2 → (83 - x) / n = avg_increase → x = 63 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_correct_mark_l642_64252


namespace NUMINAMATH_GPT_middle_number_l642_64267

theorem middle_number (x y z : ℤ) 
  (h1 : x + y = 21)
  (h2 : x + z = 25)
  (h3 : y + z = 28)
  (h4 : x < y)
  (h5 : y < z) : 
  y = 12 :=
sorry

end NUMINAMATH_GPT_middle_number_l642_64267


namespace NUMINAMATH_GPT_smallest_positive_and_largest_negative_l642_64239

theorem smallest_positive_and_largest_negative:
  (∃ (a : ℤ), a > 0 ∧ ∀ (b : ℤ), b > 0 → b ≥ a ∧ a = 1) ∧
  (∃ (c : ℤ), c < 0 ∧ ∀ (d : ℤ), d < 0 → d ≤ c ∧ c = -1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_and_largest_negative_l642_64239


namespace NUMINAMATH_GPT_flower_bed_l642_64200

def planting_schemes (A B C D E F : Prop) : Prop :=
  A ≠ B ∧ B ≠ C ∧ D ≠ E ∧ E ≠ F ∧ A ≠ D ∧ B ≠ D ∧ B ≠ E ∧ C ≠ E ∧ C ≠ F ∧ D ≠ F

theorem flower_bed (A B C D E F : Prop) (plant_choices : Finset (Fin 6))
  (h_choice : plant_choices.card = 6)
  (h_different : ∀ x ∈ plant_choices, ∀ y ∈ plant_choices, x ≠ y → x ≠ y)
  (h_adj : planting_schemes A B C D E F) :
  ∃! planting_schemes, planting_schemes ∧ plant_choices.card = 13230 :=
by sorry

end NUMINAMATH_GPT_flower_bed_l642_64200


namespace NUMINAMATH_GPT_xy_value_l642_64273

noncomputable def x (y : ℝ) : ℝ := 36 * y

theorem xy_value (y : ℝ) (h1 : y = 0.16666666666666666) : x y * y = 1 :=
by
  rw [h1, x]
  sorry

end NUMINAMATH_GPT_xy_value_l642_64273


namespace NUMINAMATH_GPT_gcd_90_250_l642_64281

theorem gcd_90_250 : Nat.gcd 90 250 = 10 := by
  sorry

end NUMINAMATH_GPT_gcd_90_250_l642_64281


namespace NUMINAMATH_GPT_leak_empties_cistern_in_24_hours_l642_64279

theorem leak_empties_cistern_in_24_hours (F L : ℝ) (h1: F = 1 / 8) (h2: F - L = 1 / 12) :
  1 / L = 24 := 
by {
  sorry
}

end NUMINAMATH_GPT_leak_empties_cistern_in_24_hours_l642_64279


namespace NUMINAMATH_GPT_man_arrived_earlier_l642_64295

-- Definitions of conditions as Lean variables
variables
  (usual_arrival_time_home : ℕ)  -- The usual arrival time at home
  (usual_drive_time : ℕ) -- The usual drive time for the wife to reach the station
  (early_arrival_difference : ℕ := 16) -- They arrived home 16 minutes earlier
  (man_walk_time : ℕ := 52) -- The man walked for 52 minutes

-- The proof statement
theorem man_arrived_earlier
  (usual_arrival_time_home : ℕ)
  (usual_drive_time : ℕ)
  (H : usual_arrival_time_home - man_walk_time <= usual_drive_time - early_arrival_difference)
  : man_walk_time = 52 :=
sorry

end NUMINAMATH_GPT_man_arrived_earlier_l642_64295


namespace NUMINAMATH_GPT_scientific_notation_of_2270000_l642_64294

theorem scientific_notation_of_2270000 : 
  (2270000 : ℝ) = 2.27 * 10^6 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_2270000_l642_64294


namespace NUMINAMATH_GPT_maximal_difference_of_areas_l642_64255

-- Given:
-- A circle of radius R
-- A chord of length 2x is drawn perpendicular to the diameter of the circle
-- The endpoints of this chord are connected to the endpoints of the diameter
-- We need to prove that under these conditions, the length of the chord 2x that maximizes the difference in areas of the triangles is R √ 2

theorem maximal_difference_of_areas (R x : ℝ) (h : 2 * x = R * Real.sqrt 2) :
  2 * x = R * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_maximal_difference_of_areas_l642_64255


namespace NUMINAMATH_GPT_large_buckets_needed_l642_64261

def capacity_large_bucket (S: ℚ) : ℚ := 2 * S + 3

theorem large_buckets_needed (n : ℕ) (L S : ℚ) (h1 : L = capacity_large_bucket S) (h2 : L = 4) (h3 : 2 * S + n * L = 63)
: n = 16 := sorry

end NUMINAMATH_GPT_large_buckets_needed_l642_64261


namespace NUMINAMATH_GPT_isolate_y_l642_64213

theorem isolate_y (x y : ℝ) (h : 3 * x - 2 * y = 6) : y = 3 * x / 2 - 3 :=
sorry

end NUMINAMATH_GPT_isolate_y_l642_64213


namespace NUMINAMATH_GPT_total_tax_in_cents_l642_64268

-- Declare the main variables and constants
def wage_per_hour_cents : ℕ := 2500
def local_tax_rate : ℝ := 0.02
def state_tax_rate : ℝ := 0.005

-- Define the total tax calculation as a proof statement
theorem total_tax_in_cents :
  local_tax_rate * wage_per_hour_cents + state_tax_rate * wage_per_hour_cents = 62.5 :=
by sorry

end NUMINAMATH_GPT_total_tax_in_cents_l642_64268


namespace NUMINAMATH_GPT_find_x_l642_64226

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (8, 1/2 * x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : vector_a x = (8, 1/2 * x)) 
(h3 : vector_b x = (x, 1)) 
(h4 : ∀ k : ℝ, (vector_a x).1 = k * (vector_b x).1 ∧ 
                       (vector_a x).2 = k * (vector_b x).2) : 
                       x = 4 := sorry

end NUMINAMATH_GPT_find_x_l642_64226


namespace NUMINAMATH_GPT_smallest_value_l642_64270

theorem smallest_value (y : ℝ) (hy : 0 < y ∧ y < 1) :
  y^3 < y^2 ∧ y^3 < 3*y ∧ y^3 < (y)^(1/3:ℝ) ∧ y^3 < (1/y) :=
sorry

end NUMINAMATH_GPT_smallest_value_l642_64270


namespace NUMINAMATH_GPT_scientific_notation_of_213_million_l642_64240

theorem scientific_notation_of_213_million : ∃ (n : ℝ), (213000000 : ℝ) = 2.13 * 10^8 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_213_million_l642_64240


namespace NUMINAMATH_GPT_find_k_inv_h_of_10_l642_64262

-- Assuming h and k are functions with appropriate properties
variables (h k : ℝ → ℝ)
variables (h_inv : ℝ → ℝ) (k_inv : ℝ → ℝ)

-- Given condition: h_inv (k(x)) = 4 * x - 5
axiom h_inv_k_eq : ∀ x, h_inv (k x) = 4 * x - 5

-- Statement to prove
theorem find_k_inv_h_of_10 :
  k_inv (h 10) = 15 / 4 := 
sorry

end NUMINAMATH_GPT_find_k_inv_h_of_10_l642_64262


namespace NUMINAMATH_GPT_union_A_B_inter_A_B_C_U_union_A_B_C_U_inter_A_B_C_U_A_C_U_B_union_C_U_A_C_U_B_inter_C_U_A_C_U_B_l642_64221

def U : Set ℕ := { x | 1 ≤ x ∧ x < 9 }
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}
def C (S : Set ℕ) : Set ℕ := U \ S

theorem union_A_B : A ∪ B = {1, 2, 3, 4, 5, 6} := 
by {
  -- proof here
  sorry
}

theorem inter_A_B : A ∩ B = {3} := 
by {
  -- proof here
  sorry
}

theorem C_U_union_A_B : C (A ∪ B) = {7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_inter_A_B : C (A ∩ B) = {1, 2, 4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_A : C A = {4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_B : C B = {1, 2, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem union_C_U_A_C_U_B : C A ∪ C B = {1, 2, 4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem inter_C_U_A_C_U_B : C A ∩ C B = {7, 8} := 
by {
  -- proof here
  sorry
}

end NUMINAMATH_GPT_union_A_B_inter_A_B_C_U_union_A_B_C_U_inter_A_B_C_U_A_C_U_B_union_C_U_A_C_U_B_inter_C_U_A_C_U_B_l642_64221


namespace NUMINAMATH_GPT_committee_count_is_correct_l642_64257

-- Definitions of the problem conditions
def total_people : ℕ := 10
def committee_size : ℕ := 5
def remaining_people := total_people - 1
def members_to_choose := committee_size - 1

-- The combinatorial function for selecting committee members
def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def number_of_ways_to_form_committee : ℕ :=
  binomial remaining_people members_to_choose

-- Statement of the problem to prove the number of ways is 126
theorem committee_count_is_correct :
  number_of_ways_to_form_committee = 126 :=
by
  sorry

end NUMINAMATH_GPT_committee_count_is_correct_l642_64257
