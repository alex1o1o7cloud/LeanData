import Mathlib

namespace NUMINAMATH_GPT_math_problem_l1700_170003

theorem math_problem :
    3 * 3^4 - (27 ^ 63 / 27 ^ 61) = -486 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1700_170003


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1700_170046

theorem perfect_square_trinomial (x : ℝ) : (x + 9)^2 = x^2 + 18 * x + 81 := by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1700_170046


namespace NUMINAMATH_GPT_integer_solution_pairs_l1700_170031

theorem integer_solution_pairs (a b : ℕ) (h_pos : a > 0 ∧ b > 0):
  (∃ k : ℕ, k > 0 ∧ a^2 = k * (2 * a * b^2 - b^3 + 1)) ↔ 
  (∃ l : ℕ, l > 0 ∧ ((a = 2 * l ∧ b = 1) ∨ (a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l))) :=
sorry

end NUMINAMATH_GPT_integer_solution_pairs_l1700_170031


namespace NUMINAMATH_GPT_value_of_a_l1700_170056

theorem value_of_a (a : ℝ) :
  ((abs ((1) - (2) + a)) = 1) ↔ (a = 0 ∨ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1700_170056


namespace NUMINAMATH_GPT_trajectory_equation_no_such_point_l1700_170039

-- Conditions for (I): The ratio of the distances is given
def ratio_condition (P : ℝ × ℝ) : Prop :=
  let M := (1, 0)
  let N := (4, 0)
  2 * Real.sqrt ((P.1 - M.1)^2 + P.2^2) = Real.sqrt ((P.1 - N.1)^2 + P.2^2)

-- Proof of (I): Find the trajectory equation of point P
theorem trajectory_equation : 
  ∀ P : ℝ × ℝ, ratio_condition P → P.1^2 + P.2^2 = 4 :=
by
  sorry

-- Conditions for (II): Given points A, B, C
def points_condition (P : ℝ × ℝ) : Prop :=
  let A := (-2, -2)
  let B := (-2, 6)
  let C := (-4, 2)
  (P.1 + 2)^2 + (P.2 + 2)^2 + 
  (P.1 + 2)^2 + (P.2 - 6)^2 + 
  (P.1 + 4)^2 + (P.2 - 2)^2 = 36

-- Proof of (II): Determine the non-existence of point P
theorem no_such_point (P : ℝ × ℝ) : 
  P.1^2 + P.2^2 = 4 → ¬ points_condition P :=
by
  sorry

end NUMINAMATH_GPT_trajectory_equation_no_such_point_l1700_170039


namespace NUMINAMATH_GPT_initial_total_perimeter_l1700_170082

theorem initial_total_perimeter (n : ℕ) (a : ℕ) (m : ℕ)
  (h1 : n = 2 * m)
  (h2 : 40 = 2 * a * m)
  (h3 : 4 * n - 6 * m = 4 * n - 40) :
  4 * n = 280 :=
by sorry

end NUMINAMATH_GPT_initial_total_perimeter_l1700_170082


namespace NUMINAMATH_GPT_circumradius_eq_exradius_opposite_BC_l1700_170073

-- Definitions of points and triangles
variable {A B C : Point}
variable (O I D : Point)
variable {α β γ : Angle}

-- Definitions of circumcenter, incenter, altitude, and collinearity
def is_circumcenter (O : Point) (A B C : Point) : Prop := sorry
def is_incenter (I : Point) (A B C : Point) : Prop := sorry
def is_altitude (A D B C : Point) : Prop := sorry
def collinear (O D I : Point) : Prop := sorry

-- Definitions of circumradius and exradius
def circumradius (A B C : Point) : ℝ := sorry
def exradius_opposite_BC (A B C : Point) : ℝ := sorry

-- Main theorem statement
theorem circumradius_eq_exradius_opposite_BC
  (h_circ : is_circumcenter O A B C)
  (h_incenter : is_incenter I A B C)
  (h_altitude : is_altitude A D B C)
  (h_collinear : collinear O D I) : 
  circumradius A B C = exradius_opposite_BC A B C :=
sorry

end NUMINAMATH_GPT_circumradius_eq_exradius_opposite_BC_l1700_170073


namespace NUMINAMATH_GPT_sum_of_first_six_terms_l1700_170092

def geometric_sequence (a : ℕ → ℤ) :=
  a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = -2 * a (n - 1)

def sum_first_six_terms (a : ℕ → ℤ) : ℤ := 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_of_first_six_terms (a : ℕ → ℤ) 
  (h : geometric_sequence a) :
  sum_first_six_terms a = -21 :=
sorry

end NUMINAMATH_GPT_sum_of_first_six_terms_l1700_170092


namespace NUMINAMATH_GPT_evaluate_seventy_five_squared_minus_twenty_five_squared_l1700_170024

theorem evaluate_seventy_five_squared_minus_twenty_five_squared :
  75^2 - 25^2 = 5000 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_seventy_five_squared_minus_twenty_five_squared_l1700_170024


namespace NUMINAMATH_GPT_bottles_remaining_l1700_170075

-- Define the initial number of bottles.
def initial_bottles : ℝ := 45.0

-- Define the number of bottles Maria drank.
def maria_drinks : ℝ := 14.0

-- Define the number of bottles Maria's sister drank.
def sister_drinks : ℝ := 8.0

-- The value that needs to be proved.
def bottles_left : ℝ := initial_bottles - maria_drinks - sister_drinks

-- The theorem statement.
theorem bottles_remaining :
  bottles_left = 23.0 :=
by
  sorry

end NUMINAMATH_GPT_bottles_remaining_l1700_170075


namespace NUMINAMATH_GPT_find_a_and_b_l1700_170037

theorem find_a_and_b (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {2, 3}) 
  (hB : B = {x | x^2 + a * x + b = 0}) 
  (h_intersection : A ∩ B = {2}) 
  (h_union : A ∪ B = A) : 
  (a + b = 0) ∨ (a + b = 1) := 
sorry

end NUMINAMATH_GPT_find_a_and_b_l1700_170037


namespace NUMINAMATH_GPT_inequality_transformation_l1700_170025

theorem inequality_transformation (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, f x = 2 * x + 3) (h2 : a > 0) (h3 : b > 0) :
  (∀ x, |f x + 5| < a → |x + 3| < b) ↔ b ≤ a / 2 :=
sorry

end NUMINAMATH_GPT_inequality_transformation_l1700_170025


namespace NUMINAMATH_GPT_one_greater_l1700_170067

theorem one_greater (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) 
  (h5 : a + b + c > 1 / a + 1 / b + 1 / c) : 
  (1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (1 < b ∧ a ≤ 1 ∧ c ≤ 1) ∨ (1 < c ∧ a ≤ 1 ∧ b ≤ 1) :=
sorry

end NUMINAMATH_GPT_one_greater_l1700_170067


namespace NUMINAMATH_GPT_limit_of_sequence_z_l1700_170006

open Nat Real

noncomputable def sequence_z (n : ℕ) : ℝ :=
  -3 + (-1)^n / (n^2 : ℝ)

theorem limit_of_sequence_z :
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, abs (sequence_z n + 3) < ε :=
by
  sorry

end NUMINAMATH_GPT_limit_of_sequence_z_l1700_170006


namespace NUMINAMATH_GPT_scientific_notation_140000000_l1700_170018

theorem scientific_notation_140000000 :
  140000000 = 1.4 * 10^8 := 
sorry

end NUMINAMATH_GPT_scientific_notation_140000000_l1700_170018


namespace NUMINAMATH_GPT_part1_part2_l1700_170032

def pointA : (ℝ × ℝ) := (1, 2)
def pointB : (ℝ × ℝ) := (-2, 3)
def pointC : (ℝ × ℝ) := (8, -5)

-- Definitions of the vectors
def OA : (ℝ × ℝ) := pointA
def OB : (ℝ × ℝ) := pointB
def OC : (ℝ × ℝ) := pointC
def AB : (ℝ × ℝ) := (pointB.1 - pointA.1, pointB.2 - pointA.2)

-- Part 1: Proving the values of x and y
theorem part1 : ∃ (x y : ℝ), OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2) ∧ x = 2 ∧ y = -3 :=
by
  sorry

-- Part 2: Proving the value of m when vectors are parallel
theorem part2 : ∃ (m : ℝ), ∃ k : ℝ, AB = (k * (m + 8), k * (2 * m - 5)) ∧ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1700_170032


namespace NUMINAMATH_GPT_average_output_l1700_170077

theorem average_output (t1 t2 t_total : ℝ) (c1 c2 c_total : ℕ) 
                        (h1 : c1 = 60) (h2 : c2 = 60) 
                        (rate1 : ℝ := 15) (rate2 : ℝ := 60) :
  t1 = c1 / rate1 ∧ t2 = c2 / rate2 ∧ t_total = t1 + t2 ∧ c_total = c1 + c2 → 
  (c_total / t_total = 24) := 
by 
  sorry

end NUMINAMATH_GPT_average_output_l1700_170077


namespace NUMINAMATH_GPT_eleven_pow_four_l1700_170064

theorem eleven_pow_four : 11 ^ 4 = 14641 := 
by sorry

end NUMINAMATH_GPT_eleven_pow_four_l1700_170064


namespace NUMINAMATH_GPT_number_of_people_got_on_train_l1700_170008

theorem number_of_people_got_on_train (initial_people : ℕ) (people_got_off : ℕ) (final_people : ℕ) (x : ℕ) 
  (h_initial : initial_people = 78) 
  (h_got_off : people_got_off = 27) 
  (h_final : final_people = 63) 
  (h_eq : final_people = initial_people - people_got_off + x) : x = 12 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_people_got_on_train_l1700_170008


namespace NUMINAMATH_GPT_unknown_number_lcm_hcf_l1700_170012

theorem unknown_number_lcm_hcf (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 192) 
  (hcf_ab : Nat.gcd a b = 16) 
  (known_number : a = 64) :
  b = 48 :=
by
  sorry -- Proof is omitted as per instruction

end NUMINAMATH_GPT_unknown_number_lcm_hcf_l1700_170012


namespace NUMINAMATH_GPT_Matilda_fathers_chocolate_bars_l1700_170048

theorem Matilda_fathers_chocolate_bars
  (total_chocolates : ℕ) (sisters : ℕ) (chocolates_each : ℕ) (given_to_father_each : ℕ) 
  (chocolates_given : ℕ) (given_to_mother : ℕ) (eaten_by_father : ℕ) : 
  total_chocolates = 20 → 
  sisters = 4 → 
  chocolates_each = total_chocolates / (sisters + 1) → 
  given_to_father_each = chocolates_each / 2 → 
  chocolates_given = (sisters + 1) * given_to_father_each → 
  given_to_mother = 3 → 
  eaten_by_father = 2 → 
  chocolates_given - given_to_mother - eaten_by_father = 5 :=
by
  intros h_total h_sisters h_chocolates_each h_given_to_father_each h_chocolates_given h_given_to_mother h_eaten_by_father
  sorry

end NUMINAMATH_GPT_Matilda_fathers_chocolate_bars_l1700_170048


namespace NUMINAMATH_GPT_find_f_prime_at_1_l1700_170001

variable (f : ℝ → ℝ)

-- Initial condition
variable (h : ∀ x, f x = x^2 + deriv f 2 * (Real.log x - x))

-- The goal is to prove that f'(1) = 2
theorem find_f_prime_at_1 : deriv f 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_prime_at_1_l1700_170001


namespace NUMINAMATH_GPT_total_initial_seashells_l1700_170036

-- Definitions for the conditions
def Henry_seashells := 11
def Paul_seashells := 24

noncomputable def Leo_initial_seashells (total_seashells : ℕ) :=
  (total_seashells - (Henry_seashells + Paul_seashells)) * 4 / 3

theorem total_initial_seashells 
  (total_seashells_now : ℕ)
  (leo_shared_fraction : ℕ → ℕ)
  (h : total_seashells_now = 53) : 
  Henry_seashells + Paul_seashells + leo_shared_fraction 53 = 59 :=
by
  let L := Leo_initial_seashells 53
  have L_initial : L = 24 := by sorry
  exact sorry

end NUMINAMATH_GPT_total_initial_seashells_l1700_170036


namespace NUMINAMATH_GPT_unique_k_for_triangle_inequality_l1700_170050

theorem unique_k_for_triangle_inequality (k : ℕ) (h : 0 < k) :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → k * (a * b + b * c + c * a) > 5 * (a * b + b * b + c * c) → a + b > c ∧ b + c > a ∧ c + a > b) ↔ (k = 6) :=
by
  sorry

end NUMINAMATH_GPT_unique_k_for_triangle_inequality_l1700_170050


namespace NUMINAMATH_GPT_largest_divisor_of_odd_product_l1700_170019

theorem largest_divisor_of_odd_product (n : ℕ) (h : Even n ∧ n > 0) :
  ∃ m, m > 0 ∧ (∀ k, (n+1)*(n+3)*(n+7)*(n+9)*(n+11) % k = 0 ↔ k ≤ 15) := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_largest_divisor_of_odd_product_l1700_170019


namespace NUMINAMATH_GPT_factorize_quadratic_l1700_170034

theorem factorize_quadratic (x : ℝ) : x^2 - 6 * x + 9 = (x - 3)^2 :=
by {
  sorry  -- Proof goes here
}

end NUMINAMATH_GPT_factorize_quadratic_l1700_170034


namespace NUMINAMATH_GPT_smallest_x_l1700_170090

theorem smallest_x (x : ℕ) (h : 67 * 89 * x % 35 = 0) : x = 35 := 
by sorry

end NUMINAMATH_GPT_smallest_x_l1700_170090


namespace NUMINAMATH_GPT_quadratic_roots_solution_l1700_170044

noncomputable def quadratic_roots_differ_by_2 (p q : ℝ) (hq_pos : 0 < q) (hp_pos : 0 < p) : Prop :=
  let root1 := (-p + Real.sqrt (p^2 - 4*q)) / 2
  let root2 := (-p - Real.sqrt (p^2 - 4*q)) / 2
  abs (root1 - root2) = 2

theorem quadratic_roots_solution (p q : ℝ) (hq_pos : 0 < q) (hp_pos : 0 < p) :
  quadratic_roots_differ_by_2 p q hq_pos hp_pos →
  p = 2 * Real.sqrt (q + 1) :=
sorry

end NUMINAMATH_GPT_quadratic_roots_solution_l1700_170044


namespace NUMINAMATH_GPT_new_students_weights_correct_l1700_170011

-- Definitions of the initial conditions
def initial_student_count : ℕ := 29
def initial_avg_weight : ℚ := 28
def total_initial_weight := initial_student_count * initial_avg_weight
def new_student_counts : List ℕ := [30, 31, 32, 33]
def new_avg_weights : List ℚ := [27.2, 27.8, 27.6, 28]

-- Weights of the four new students
def W1 : ℚ := 4
def W2 : ℚ := 45.8
def W3 : ℚ := 21.4
def W4 : ℚ := 40.8

-- The proof statement
theorem new_students_weights_correct :
  total_initial_weight = 812 ∧
  W1 = 4 ∧
  W2 = 45.8 ∧
  W3 = 21.4 ∧
  W4 = 40.8 ∧
  (total_initial_weight + W1) = 816 ∧
  (total_initial_weight + W1) / new_student_counts.head! = new_avg_weights.head! ∧
  (total_initial_weight + W1 + W2) = 861.8 ∧
  (total_initial_weight + W1 + W2) / new_student_counts.tail.head! = new_avg_weights.tail.head! ∧
  (total_initial_weight + W1 + W2 + W3) = 883.2 ∧
  (total_initial_weight + W1 + W2 + W3) / new_student_counts.tail.tail.head! = new_avg_weights.tail.tail.head! ∧
  (total_initial_weight + W1 + W2 + W3 + W4) = 924 ∧
  (total_initial_weight + W1 + W2 + W3 + W4) / new_student_counts.tail.tail.tail.head! = new_avg_weights.tail.tail.tail.head! :=
by
  sorry

end NUMINAMATH_GPT_new_students_weights_correct_l1700_170011


namespace NUMINAMATH_GPT_part_one_part_two_l1700_170047

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - x + 1

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1 / x

theorem part_one (x a : ℝ) (hx : x > 0) (ineq : x * f' x ≤ x^2 + a * x + 1) : a ∈ Set.Ici (-1) :=
by sorry

theorem part_two (x : ℝ) (hx : x > 0) : (x - 1) * f x ≥ 0 :=
by sorry

end NUMINAMATH_GPT_part_one_part_two_l1700_170047


namespace NUMINAMATH_GPT_range_of_m_l1700_170054

theorem range_of_m
  (m : ℝ)
  (h1 : (m - 1) * (3 - m) ≠ 0) 
  (h2 : 3 - m > 0) 
  (h3 : m - 1 > 0) 
  (h4 : 3 - m ≠ m - 1) :
  1 < m ∧ m < 3 ∧ m ≠ 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1700_170054


namespace NUMINAMATH_GPT_mod_remainder_l1700_170041

theorem mod_remainder :
  ((85^70 + 19^32)^16) % 21 = 16 := by
  -- Given conditions
  have h1 : 85^70 % 21 = 1 := sorry
  have h2 : 19^32 % 21 = 4 := sorry
  -- Conclusion
  sorry

end NUMINAMATH_GPT_mod_remainder_l1700_170041


namespace NUMINAMATH_GPT_right_triangle_bc_is_3_l1700_170010

-- Define the setup: a right triangle with given side lengths
structure RightTriangle :=
  (AB AC BC : ℝ)
  (right_angle : AB^2 = AC^2 + BC^2)
  (AB_val : AB = 5)
  (AC_val : AC = 4)

-- The goal is to prove that BC = 3 given the conditions
theorem right_triangle_bc_is_3 (T : RightTriangle) : T.BC = 3 :=
  sorry

end NUMINAMATH_GPT_right_triangle_bc_is_3_l1700_170010


namespace NUMINAMATH_GPT_ratio_after_girls_leave_l1700_170091

-- Define the initial conditions
def initial_conditions (B G : ℕ) : Prop :=
  B = G ∧ B + G = 32

-- Define the event of girls leaving
def girls_leave (G : ℕ) : ℕ :=
  G - 8

-- Define the final ratio of boys to girls
def final_ratio (B G : ℕ) : ℕ :=
  B / (girls_leave G)

-- Prove the final ratio is 2:1
theorem ratio_after_girls_leave (B G : ℕ) (h : initial_conditions B G) :
  final_ratio B G = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_after_girls_leave_l1700_170091


namespace NUMINAMATH_GPT_combined_work_rate_l1700_170059

theorem combined_work_rate (W : ℝ) 
  (A_rate : ℝ := W / 10) 
  (B_rate : ℝ := W / 5) : 
  A_rate + B_rate = 3 * W / 10 := 
by
  sorry

end NUMINAMATH_GPT_combined_work_rate_l1700_170059


namespace NUMINAMATH_GPT_locus_of_points_l1700_170078

def point := (ℝ × ℝ)

variables (F_1 F_2 : point) (r k : ℝ)

def distance (P Q : point) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2)

def on_circle (P : point) (center : point) (radius : ℝ) : Prop :=
  distance P center = radius

theorem locus_of_points
  (P : point)
  (r1 r2 PF1 PF2 : ℝ)
  (h_pF1 : r1 = distance P F_1)
  (h_pF2 : PF2 = distance P F_2)
  (h_outside_circle : PF2 = r2 + r)
  (h_inside_circle : PF2 = r - r2)
  (h_k : r1 + PF2 = k) :
  (∀ P, distance P F_1 + distance P F_2 = k →
  ( ∃ e_ellipse : Prop, on_circle P F_2 r → e_ellipse) ∨ 
  ( ∃ h_hyperbola : Prop, on_circle P F_2 r → h_hyperbola)) :=
by
  sorry

end NUMINAMATH_GPT_locus_of_points_l1700_170078


namespace NUMINAMATH_GPT_faster_speed_l1700_170094

theorem faster_speed (S : ℝ) (actual_speed : ℝ := 10) (extra_distance : ℝ := 20) (actual_distance : ℝ := 20) :
  actual_distance / actual_speed = (actual_distance + extra_distance) / S → S = 20 :=
by
  sorry

end NUMINAMATH_GPT_faster_speed_l1700_170094


namespace NUMINAMATH_GPT_product_identity_l1700_170009

theorem product_identity :
  (1 + 1 / Nat.factorial 1) * (1 + 1 / Nat.factorial 2) * (1 + 1 / Nat.factorial 3) *
  (1 + 1 / Nat.factorial 4) * (1 + 1 / Nat.factorial 5) * (1 + 1 / Nat.factorial 6) *
  (1 + 1 / Nat.factorial 7) = 5041 / 5040 := sorry

end NUMINAMATH_GPT_product_identity_l1700_170009


namespace NUMINAMATH_GPT_paula_bought_fewer_cookies_l1700_170020
-- Import the necessary libraries

-- Definitions
def paul_cookies : ℕ := 45
def total_cookies : ℕ := 87

-- Theorem statement
theorem paula_bought_fewer_cookies : ∃ (paula_cookies : ℕ), paul_cookies + paula_cookies = total_cookies ∧ paul_cookies - paula_cookies = 3 := by
  sorry

end NUMINAMATH_GPT_paula_bought_fewer_cookies_l1700_170020


namespace NUMINAMATH_GPT_rectangle_width_l1700_170016

-- The Lean statement only with given conditions and the final proof goal
theorem rectangle_width (w l : ℕ) (P : ℕ) (h1 : l = w - 3) (h2 : P = 2 * w + 2 * l) (h3 : P = 54) :
  w = 15 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l1700_170016


namespace NUMINAMATH_GPT_Solomon_collected_66_l1700_170052

-- Definitions
variables (J S L : ℕ) -- J for Juwan, S for Solomon, L for Levi

-- Conditions
axiom C1 : S = 3 * J
axiom C2 : L = J / 2
axiom C3 : J + S + L = 99

-- Theorem to prove
theorem Solomon_collected_66 : S = 66 :=
by
  sorry

end NUMINAMATH_GPT_Solomon_collected_66_l1700_170052


namespace NUMINAMATH_GPT_color_points_l1700_170080

def is_white (p : ℤ × ℤ) : Prop := (p.1 % 2 = 1) ∧ (p.2 % 2 = 1)
def is_black (p : ℤ × ℤ) : Prop := (p.1 % 2 = 0) ∧ (p.2 % 2 = 0)
def is_red (p : ℤ × ℤ) : Prop := (p.1 % 2 = 1 ∧ p.2 % 2 = 0) ∨ (p.1 % 2 = 0 ∧ p.2 % 2 = 1)

theorem color_points :
  (∀ n : ℤ, ∃ (p : ℤ × ℤ), (p.2 = n) ∧ is_white p ∧
                             is_black ⟨p.1, n * 2⟩ ∧
                             is_red ⟨p.1, n * 2 + 1⟩) ∧ 
  (∀ (A B C : ℤ × ℤ), 
    is_white A → is_red B → is_black C → 
    ∃ D : ℤ × ℤ, is_red D ∧ 
    (A.1 + C.1 - B.1 = D.1 ∧
     A.2 + C.2 - B.2 = D.2)) := sorry

end NUMINAMATH_GPT_color_points_l1700_170080


namespace NUMINAMATH_GPT_value_of_sum_l1700_170099

theorem value_of_sum (a x y : ℝ) (h1 : 17 * x + 19 * y = 6 - a) (h2 : 13 * x - 7 * y = 10 * a + 1) : 
  x + y = 1 / 3 := 
sorry

end NUMINAMATH_GPT_value_of_sum_l1700_170099


namespace NUMINAMATH_GPT_simplify_expression_l1700_170017

theorem simplify_expression (b : ℝ) (hb : b = -1) : 
  (3 * b⁻¹ + (2 * b⁻¹) / 3) / b = 11 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1700_170017


namespace NUMINAMATH_GPT_width_of_sheet_of_paper_l1700_170097

theorem width_of_sheet_of_paper (W : ℝ) (h1 : ∀ (W : ℝ), W > 0) (length_paper : ℝ) (margin : ℝ)
  (width_picture_area : ∀ (W : ℝ), W - 2 * margin = (W - 3)) 
  (area_picture : ℝ) (length_picture_area : ℝ) :
  length_paper = 10 ∧ margin = 1.5 ∧ area_picture = 38.5 ∧ length_picture_area = 7 →
  W = 8.5 :=
by
  sorry

end NUMINAMATH_GPT_width_of_sheet_of_paper_l1700_170097


namespace NUMINAMATH_GPT_find_interest_rate_of_initial_investment_l1700_170042

def initial_investment : ℝ := 1400
def additional_investment : ℝ := 700
def total_investment : ℝ := 2100
def additional_interest_rate : ℝ := 0.08
def target_total_income_rate : ℝ := 0.06
def target_total_income : ℝ := target_total_income_rate * total_investment

theorem find_interest_rate_of_initial_investment (r : ℝ) :
  (initial_investment * r + additional_investment * additional_interest_rate = target_total_income) → 
  (r = 0.05) :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_of_initial_investment_l1700_170042


namespace NUMINAMATH_GPT_grazing_area_proof_l1700_170095

noncomputable def grazing_area (s r : ℝ) : ℝ :=
  let A_circle := 3.14 * r^2
  let A_sector := (300 / 360) * A_circle
  let A_triangle := (1.732 / 4) * s^2
  let A_triangle_part := A_triangle / 3
  let A_grazing := A_sector - A_triangle_part
  3 * A_grazing

theorem grazing_area_proof : grazing_area 5 7 = 136.59 :=
  by
  sorry

end NUMINAMATH_GPT_grazing_area_proof_l1700_170095


namespace NUMINAMATH_GPT_river_width_l1700_170072

def boat_width : ℕ := 3
def num_boats : ℕ := 8
def space_between_boats : ℕ := 2
def riverbank_space : ℕ := 2

theorem river_width : 
  let boat_space := num_boats * boat_width
  let between_boat_space := (num_boats - 1) * space_between_boats
  let riverbank_space_total := 2 * riverbank_space
  boat_space + between_boat_space + riverbank_space_total = 42 :=
by
  sorry

end NUMINAMATH_GPT_river_width_l1700_170072


namespace NUMINAMATH_GPT_perpendicular_os_bc_l1700_170045

variable {A B C O S : Type}

noncomputable def acute_triangle (A B C : Type) := true -- Placeholder definition for acute triangle.

noncomputable def circumcenter (O : Type) (A B C : Type) := true -- Placeholder definition for circumcenter.

noncomputable def line_intersects_circumcircle_second_time (AC : Type) (circ : Type) (S : Type) := true -- Placeholder def.

-- Define the problem in Lean
theorem perpendicular_os_bc
  (ABC_is_acute : acute_triangle A B C)
  (O_is_circumcenter : circumcenter O A B C)
  (AC_intersects_AOB_circumcircle_at_S : line_intersects_circumcircle_second_time (A → C) (A → B → O) S) :
  true := -- Place for the proof that OS ⊥ BC
sorry

end NUMINAMATH_GPT_perpendicular_os_bc_l1700_170045


namespace NUMINAMATH_GPT_find_A_minus_C_l1700_170057

/-- There are three different natural numbers A, B, and C. 
    When A + B = 84, B + C = 60, and A = 6B, find the value of A - C. -/
theorem find_A_minus_C (A B C : ℕ) 
  (h1 : A + B = 84) 
  (h2 : B + C = 60) 
  (h3 : A = 6 * B) 
  (h4 : A ≠ B) 
  (h5 : A ≠ C) 
  (h6 : B ≠ C) :
  A - C = 24 :=
sorry

end NUMINAMATH_GPT_find_A_minus_C_l1700_170057


namespace NUMINAMATH_GPT_distance_travelled_l1700_170029

variables (D : ℝ) (h_eq : D / 10 = (D + 20) / 14)

theorem distance_travelled : D = 50 :=
by sorry

end NUMINAMATH_GPT_distance_travelled_l1700_170029


namespace NUMINAMATH_GPT_most_representative_sample_l1700_170098

/-- Options for the student sampling methods -/
inductive SamplingMethod
| NinthGradeStudents : SamplingMethod
| FemaleStudents : SamplingMethod
| BasketballStudents : SamplingMethod
| StudentsWithIDEnding5 : SamplingMethod

/-- Definition of representativeness for each SamplingMethod -/
def isMostRepresentative (method : SamplingMethod) : Prop :=
  method = SamplingMethod.StudentsWithIDEnding5

/-- Prove that the students with ID ending in 5 is the most representative sampling method -/
theorem most_representative_sample : isMostRepresentative SamplingMethod.StudentsWithIDEnding5 :=
  by
  sorry

end NUMINAMATH_GPT_most_representative_sample_l1700_170098


namespace NUMINAMATH_GPT_simplify_expression_l1700_170074

-- Definitions of intermediate calculations
def a : ℤ := 3 + 5 + 6 - 2
def b : ℚ := a * 2 / 4
def c : ℤ := 3 * 4 + 6 - 4
def d : ℚ := c / 3

-- The statement to be proved
theorem simplify_expression : b + d = 32 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1700_170074


namespace NUMINAMATH_GPT_amount_received_by_A_is_4_over_3_l1700_170038

theorem amount_received_by_A_is_4_over_3
  (a d : ℚ)
  (h1 : a - 2 * d + a - d = a + (a + d) + (a + 2 * d))
  (h2 : 5 * a = 5) :
  a - 2 * d = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_amount_received_by_A_is_4_over_3_l1700_170038


namespace NUMINAMATH_GPT_moles_of_naoh_combined_number_of_moles_of_naoh_combined_l1700_170053

-- Define the reaction equation and given conditions
def reaction_equation := "2 NaOH + Cl₂ → NaClO + NaCl + H₂O"

-- Given conditions
def moles_chlorine : ℕ := 2
def moles_water_produced : ℕ := 2
def moles_naoh_needed_for_one_mole_water : ℕ := 2

-- Stoichiometric relationship from the reaction equation
def moles_naoh_per_mole_water : ℕ := 2

-- Theorem to prove the number of moles of NaOH combined
theorem moles_of_naoh_combined (moles_water_produced : ℕ)
  (moles_naoh_per_mole_water : ℕ) : ℕ :=
  moles_water_produced * moles_naoh_per_mole_water

-- Statement of the theorem
theorem number_of_moles_of_naoh_combined : moles_of_naoh_combined 2 2 = 4 :=
by sorry

end NUMINAMATH_GPT_moles_of_naoh_combined_number_of_moles_of_naoh_combined_l1700_170053


namespace NUMINAMATH_GPT_phillip_spent_on_oranges_l1700_170069

theorem phillip_spent_on_oranges 
  (M : ℕ) (A : ℕ) (C : ℕ) (L : ℕ) (O : ℕ)
  (hM : M = 95) (hA : A = 25) (hC : C = 6) (hL : L = 50)
  (h_total_spending : O + A + C = M - L) : 
  O = 14 := 
sorry

end NUMINAMATH_GPT_phillip_spent_on_oranges_l1700_170069


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l1700_170071

variable (v_m v_s : ℝ)

-- Conditions
def downstream_distance : ℝ := 51
def upstream_distance : ℝ := 18
def time : ℝ := 3

-- Equations based on the conditions
def downstream_speed_eq : Prop := downstream_distance = (v_m + v_s) * time
def upstream_speed_eq : Prop := upstream_distance = (v_m - v_s) * time

-- The theorem to prove
theorem speed_of_man_in_still_water : downstream_speed_eq v_m v_s ∧ upstream_speed_eq v_m v_s → v_m = 11.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l1700_170071


namespace NUMINAMATH_GPT_max_min_of_f_find_a_and_theta_l1700_170055

noncomputable def f (x θ a : ℝ) : ℝ :=
  Real.sin (x + θ) + a * Real.cos (x + 2 * θ)

theorem max_min_of_f (a θ : ℝ) (h1 : a = Real.sqrt 2) (h2 : θ = π / 4) :
  (∀ x ∈ Set.Icc 0 π, -1 ≤ f x θ a ∧ f x θ a ≤ (Real.sqrt 2) / 2) := sorry

theorem find_a_and_theta (a θ : ℝ) (h1 : f (π / 2) θ a = 0) (h2 : f π θ a = 1) :
  a = -1 ∧ θ = -π / 6 := sorry

end NUMINAMATH_GPT_max_min_of_f_find_a_and_theta_l1700_170055


namespace NUMINAMATH_GPT_min_sum_x_y_l1700_170066

theorem min_sum_x_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0 ∧ y > 0) (h3 : (1 : ℚ)/x + (1 : ℚ)/y = 1/12) : x + y = 49 :=
sorry

end NUMINAMATH_GPT_min_sum_x_y_l1700_170066


namespace NUMINAMATH_GPT_min_value_fraction_geq_3_div_2_l1700_170058

theorem min_value_fraction_geq_3_div_2 (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) (h1 : q > 0) 
  (h2 : ∀ k, a (k + 2) = q * a (k + 1)) (h3 : a 2016 = a 2015 + 2 * a 2014) 
  (h4 : a m * a n = 16 * (a 1) ^ 2) :
  (∃ q, q = 2 ∧ m + n = 6) → 4 / m + 1 / n ≥ 3 / 2 :=
by sorry

end NUMINAMATH_GPT_min_value_fraction_geq_3_div_2_l1700_170058


namespace NUMINAMATH_GPT_problem_proof_l1700_170086

-- Define positive integers and the conditions given in the problem
variables {p q r s : ℕ}

-- The product of the four integers is 7!
axiom product_of_integers : p * q * r * s = 5040  -- 7! = 5040

-- The equations defining the relationships
axiom equation1 : p * q + p + q = 715
axiom equation2 : q * r + q + r = 209
axiom equation3 : r * s + r + s = 143

-- The goal is to prove p - s = 10
theorem problem_proof : p - s = 10 :=
sorry

end NUMINAMATH_GPT_problem_proof_l1700_170086


namespace NUMINAMATH_GPT_ones_digit_of_73_pow_355_l1700_170081

theorem ones_digit_of_73_pow_355 : (73 ^ 355) % 10 = 7 := 
  sorry

end NUMINAMATH_GPT_ones_digit_of_73_pow_355_l1700_170081


namespace NUMINAMATH_GPT_cone_generatrix_length_theorem_l1700_170023

noncomputable def cone_generatrix_length 
  (diameter : ℝ) 
  (unfolded_side_area : ℝ) 
  (h_diameter : diameter = 6)
  (h_area : unfolded_side_area = 18 * Real.pi) : 
  ℝ :=
6

theorem cone_generatrix_length_theorem 
  (diameter : ℝ) 
  (unfolded_side_area : ℝ) 
  (h_diameter : diameter = 6)
  (h_area : unfolded_side_area = 18 * Real.pi) :
  cone_generatrix_length diameter unfolded_side_area h_diameter h_area = 6 :=
sorry

end NUMINAMATH_GPT_cone_generatrix_length_theorem_l1700_170023


namespace NUMINAMATH_GPT_lcm_of_numbers_l1700_170022

theorem lcm_of_numbers (a b c d : ℕ) (h1 : a = 8) (h2 : b = 24) (h3 : c = 36) (h4 : d = 54) :
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 216 := 
by 
  sorry

end NUMINAMATH_GPT_lcm_of_numbers_l1700_170022


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1700_170014

theorem eccentricity_of_ellipse 
  (a b : ℝ) (e : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), x = 0 ∧ y > 0 ∧ (9 * b^2 = 16/7 * a^2)) :
  e = Real.sqrt (10) / 6 :=
sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1700_170014


namespace NUMINAMATH_GPT_remainder_of_S_mod_1000_l1700_170015

def digit_contribution (d pos : ℕ) : ℕ := (d * d) * pos

def sum_of_digits_with_no_repeats : ℕ :=
  let thousands := (16 + 25 + 36 + 49 + 64 + 81) * (9 * 8 * 7) * 1000
  let hundreds := (16 + 25 + 36 + 49 + 64 + 81) * (8 * 7 * 6) * 100
  let tens := (0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 - (16 + 25 + 36 + 49 + 64 + 81)) * 6 * 5 * 10
  let units := (0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 - (16 + 25 + 36 + 49 + 64 + 81)) * 6 * 5 * 1
  thousands + hundreds + tens + units

theorem remainder_of_S_mod_1000 : (sum_of_digits_with_no_repeats % 1000) = 220 :=
  by
  sorry

end NUMINAMATH_GPT_remainder_of_S_mod_1000_l1700_170015


namespace NUMINAMATH_GPT_correct_average_marks_l1700_170087

theorem correct_average_marks (n : ℕ) (average initial_wrong current_correct : ℕ) 
  (h_n : n = 10) 
  (h_avg : average = 100) 
  (h_wrong : initial_wrong = 60)
  (h_correct : current_correct = 10) : 
  (average * n - initial_wrong + current_correct) / n = 95 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_correct_average_marks_l1700_170087


namespace NUMINAMATH_GPT_dice_product_sum_impossible_l1700_170084

theorem dice_product_sum_impossible (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6) (h2 : 1 ≤ d2 ∧ d2 ≤ 6) (h3 : 1 ≤ d3 ∧ d3 ≤ 6) (h4 : 1 ≤ d4 ∧ d4 ≤ 6) (hprod : d1 * d2 * d3 * d4 = 180) :
  (d1 + d2 + d3 + d4 ≠ 14) ∧ (d1 + d2 + d3 + d4 ≠ 17) :=
by
  sorry

end NUMINAMATH_GPT_dice_product_sum_impossible_l1700_170084


namespace NUMINAMATH_GPT_average_weight_of_section_B_l1700_170093

theorem average_weight_of_section_B
  (num_students_A : ℕ) (num_students_B : ℕ)
  (avg_weight_A : ℝ) (avg_weight_class : ℝ)
  (total_students : ℕ := num_students_A + num_students_B)
  (total_weight_class : ℝ := total_students * avg_weight_class)
  (total_weight_A : ℝ := num_students_A * avg_weight_A)
  (total_weight_B : ℝ := total_weight_class - total_weight_A)
  (avg_weight_B : ℝ := total_weight_B / num_students_B) :
  num_students_A = 50 →
  num_students_B = 40 →
  avg_weight_A = 50 →
  avg_weight_class = 58.89 →
  avg_weight_B = 70.0025 :=
by intros; sorry

end NUMINAMATH_GPT_average_weight_of_section_B_l1700_170093


namespace NUMINAMATH_GPT_abs_value_difference_l1700_170028

theorem abs_value_difference (x y : ℤ) (h1 : |x| = 7) (h2 : |y| = 9) (h3 : |x + y| = -(x + y)) :
  x - y = 16 ∨ x - y = -16 :=
sorry

end NUMINAMATH_GPT_abs_value_difference_l1700_170028


namespace NUMINAMATH_GPT_remainder_when_7n_div_by_3_l1700_170013

theorem remainder_when_7n_div_by_3 (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := 
sorry

end NUMINAMATH_GPT_remainder_when_7n_div_by_3_l1700_170013


namespace NUMINAMATH_GPT_solve_for_x_l1700_170083

theorem solve_for_x (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (h_neq : m ≠ n) :
  ∃ x : ℝ, (x + 2 * m)^2 - (x - 3 * n)^2 = 9 * (m + n)^2 ↔
  x = (5 * m^2 + 18 * m * n + 18 * n^2) / (10 * m + 6 * n) := sorry

end NUMINAMATH_GPT_solve_for_x_l1700_170083


namespace NUMINAMATH_GPT_eq_30_apples_n_7_babies_min_3_max_6_l1700_170088

theorem eq_30_apples_n_7_babies_min_3_max_6 (x : ℕ) 
    (h1 : 30 = x + 7 * 4)
    (h2 : 21 ≤ 30) 
    (h3 : 30 ≤ 42) 
    (h4 : x = 2) :
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_eq_30_apples_n_7_babies_min_3_max_6_l1700_170088


namespace NUMINAMATH_GPT_profit_amount_l1700_170005

-- Conditions: Selling Price and Profit Percentage
def SP : ℝ := 850
def P_percent : ℝ := 37.096774193548384

-- Theorem: The profit amount is $230
theorem profit_amount : (SP / (1 + P_percent / 100)) * P_percent / 100 = 230 := by
  -- sorry will be replaced with the proof
  sorry

end NUMINAMATH_GPT_profit_amount_l1700_170005


namespace NUMINAMATH_GPT_jane_needs_9_more_days_l1700_170076

def jane_rate : ℕ := 16
def mark_rate : ℕ := 20
def mark_days : ℕ := 3
def total_vases : ℕ := 248

def vases_by_mark_in_3_days : ℕ := mark_rate * mark_days
def vases_by_jane_and_mark_in_3_days : ℕ := (jane_rate + mark_rate) * mark_days
def remaining_vases_after_3_days : ℕ := total_vases - vases_by_jane_and_mark_in_3_days
def days_jane_needs_alone : ℕ := (remaining_vases_after_3_days + jane_rate - 1) / jane_rate

theorem jane_needs_9_more_days :
  days_jane_needs_alone = 9 :=
by
  sorry

end NUMINAMATH_GPT_jane_needs_9_more_days_l1700_170076


namespace NUMINAMATH_GPT_unique_g_zero_l1700_170063

theorem unique_g_zero (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x + y) = g (x) + g (y) - 1) : g 0 = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_g_zero_l1700_170063


namespace NUMINAMATH_GPT_right_triangle_x_value_l1700_170033

theorem right_triangle_x_value (BM MA BC CA: ℝ) (M_is_altitude: BM + MA = BC + CA)
  (x: ℝ) (h: ℝ) (d: ℝ) (M: BM = x) (CB: BC = h) (CA: CA = d) :
  x = (2 * h * d - d ^ 2 / 4) / (2 * d + 2 * h) := by
  sorry

end NUMINAMATH_GPT_right_triangle_x_value_l1700_170033


namespace NUMINAMATH_GPT_Mrs_Heine_treats_l1700_170040

theorem Mrs_Heine_treats :
  let dogs := 2
  let cats := 1
  let parrots := 3
  let biscuits_per_dog := 3
  let treats_per_cat := 2
  let sticks_per_parrot := 1
  let total_treats := dogs * biscuits_per_dog + cats * treats_per_cat + parrots * sticks_per_parrot
  total_treats = 11 :=
by
  let dogs := 2
  let cats := 1
  let parrots := 3
  let biscuits_per_dog := 3
  let treats_per_cat := 2
  let sticks_per_parrot := 1
  let total_treats := dogs * biscuits_per_dog + cats * treats_per_cat + parrots * sticks_per_parrot
  show total_treats = 11
  sorry

end NUMINAMATH_GPT_Mrs_Heine_treats_l1700_170040


namespace NUMINAMATH_GPT_factor_expression_l1700_170035

theorem factor_expression (y : ℝ) : 16 * y^3 + 8 * y^2 = 8 * y^2 * (2 * y + 1) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1700_170035


namespace NUMINAMATH_GPT_period_start_time_l1700_170002

theorem period_start_time (end_time : ℕ) (rained_hours : ℕ) (not_rained_hours : ℕ) (total_hours : ℕ) (start_time : ℕ) 
  (h1 : end_time = 17) -- 5 pm as 17 in 24-hour format 
  (h2 : rained_hours = 2)
  (h3 : not_rained_hours = 6)
  (h4 : total_hours = rained_hours + not_rained_hours)
  (h5 : total_hours = 8)
  (h6 : start_time = end_time - total_hours)
  : start_time = 9 :=
sorry

end NUMINAMATH_GPT_period_start_time_l1700_170002


namespace NUMINAMATH_GPT_sum_of_consecutive_negatives_l1700_170043

theorem sum_of_consecutive_negatives (n : ℤ) (h1 : n * (n + 1) = 2720) (h2 : n < 0) : 
  n + (n + 1) = -103 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_negatives_l1700_170043


namespace NUMINAMATH_GPT_freds_average_book_cost_l1700_170021

theorem freds_average_book_cost :
  ∀ (initial_amount spent_amount num_books remaining_amount avg_cost : ℕ),
    initial_amount = 236 →
    remaining_amount = 14 →
    num_books = 6 →
    spent_amount = initial_amount - remaining_amount →
    avg_cost = spent_amount / num_books →
    avg_cost = 37 :=
by
  intros initial_amount spent_amount num_books remaining_amount avg_cost h_init h_rem h_books h_spent h_avg
  sorry

end NUMINAMATH_GPT_freds_average_book_cost_l1700_170021


namespace NUMINAMATH_GPT_complement_A_U_l1700_170070

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A : Set ℕ := {1, 3}

-- Define the complement of A with respect to U
def C_U_A : Set ℕ := U \ A

-- Theorem: The complement of A with respect to U is {2, 4}
theorem complement_A_U : C_U_A = {2, 4} := by
  sorry

end NUMINAMATH_GPT_complement_A_U_l1700_170070


namespace NUMINAMATH_GPT_vincent_correct_answer_l1700_170061

theorem vincent_correct_answer (y : ℕ) (h : (y - 7) / 5 = 23) : (y - 5) / 7 = 17 :=
by
  sorry

end NUMINAMATH_GPT_vincent_correct_answer_l1700_170061


namespace NUMINAMATH_GPT_point_on_line_has_correct_y_l1700_170007

theorem point_on_line_has_correct_y (a : ℝ) : (2 * 3 + a - 7 = 0) → a = 1 :=
by 
  sorry

end NUMINAMATH_GPT_point_on_line_has_correct_y_l1700_170007


namespace NUMINAMATH_GPT_sum_of_roots_eq_14_l1700_170096

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_roots_eq_14_l1700_170096


namespace NUMINAMATH_GPT_solve_congruence_l1700_170089

open Nat

theorem solve_congruence (x : ℕ) (h : x^2 + x - 6 ≡ 0 [MOD 143]) : 
  x = 2 ∨ x = 41 ∨ x = 101 ∨ x = 140 :=
by
  sorry

end NUMINAMATH_GPT_solve_congruence_l1700_170089


namespace NUMINAMATH_GPT_find_older_friend_age_l1700_170051

theorem find_older_friend_age (A B C : ℕ) 
  (h1 : A - B = 2) 
  (h2 : A - C = 5) 
  (h3 : A + B + C = 110) : 
  A = 39 := 
by 
  sorry

end NUMINAMATH_GPT_find_older_friend_age_l1700_170051


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1700_170068

-- Conditions: definitions of sets A and B
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def set_B : Set ℝ := {x | x < 1}

-- The proof goal: A ∩ B = {x | -1 ≤ x ∧ x < 1}
theorem intersection_of_A_and_B : set_A ∩ set_B = {x | -1 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1700_170068


namespace NUMINAMATH_GPT_seating_arrangements_l1700_170060

def valid_seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  if total_seats = 8 ∧ people = 3 then 12 else 0

theorem seating_arrangements (total_seats people : ℕ) (h1 : total_seats = 8) (h2 : people = 3) :
  valid_seating_arrangements total_seats people = 12 :=
by
  rw [valid_seating_arrangements, h1, h2]
  simp
  done

end NUMINAMATH_GPT_seating_arrangements_l1700_170060


namespace NUMINAMATH_GPT_max_possible_percentage_l1700_170079

theorem max_possible_percentage (p_wi : ℝ) (p_fs : ℝ) (h_wi : p_wi = 0.4) (h_fs : p_fs = 0.7) :
  ∃ p_both : ℝ, p_both = min p_wi p_fs ∧ p_both = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_max_possible_percentage_l1700_170079


namespace NUMINAMATH_GPT_sum_of_coordinates_of_other_endpoint_l1700_170000

theorem sum_of_coordinates_of_other_endpoint
  (x y : ℝ)
  (midpoint_cond : (x + 1) / 2 = 3)
  (midpoint_cond2 : (y - 3) / 2 = 5) :
  x + y = 18 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_other_endpoint_l1700_170000


namespace NUMINAMATH_GPT_bridge_length_l1700_170065

theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross_bridge : ℝ) 
  (train_speed_m_s : train_speed_kmh * (1000 / 3600) = 15) : 
  train_length = 110 → train_speed_kmh = 54 → time_to_cross_bridge = 16.13204276991174 → 
  ((train_speed_kmh * (1000 / 3600)) * time_to_cross_bridge - train_length = 131.9806415486761) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_bridge_length_l1700_170065


namespace NUMINAMATH_GPT_sum_first_40_terms_l1700_170004

-- Given: The sum of the first 10 terms of a geometric sequence is 9
axiom S_10 : ℕ → ℕ
axiom sum_S_10 : S_10 10 = 9 

-- Given: The sum of the terms from the 11th to the 20th is 36
axiom S_20 : ℕ → ℕ
axiom sum_S_20 : S_20 20 - S_10 10 = 36

-- Let Sn be the sum of the first n terms in the geometric sequence
def Sn (n : ℕ) : ℕ := sorry

-- Prove: The sum of the first 40 terms is 144
theorem sum_first_40_terms : Sn 40 = 144 := sorry

end NUMINAMATH_GPT_sum_first_40_terms_l1700_170004


namespace NUMINAMATH_GPT_difference_nickels_is_8q_minus_20_l1700_170026

variable (q : ℤ)

-- Define the number of quarters for Alice and Bob
def alice_quarters : ℤ := 7 * q - 3
def bob_quarters : ℤ := 3 * q + 7

-- Define the worth of a quarter in nickels
def quarter_to_nickels (quarters : ℤ) : ℤ := 2 * quarters

-- Define the difference in quarters
def difference_quarters : ℤ := alice_quarters q - bob_quarters q

-- Define the difference in their amount of money in nickels
def difference_nickels (q : ℤ) : ℤ := quarter_to_nickels (difference_quarters q)

theorem difference_nickels_is_8q_minus_20 : difference_nickels q = 8 * q - 20 := by
  sorry

end NUMINAMATH_GPT_difference_nickels_is_8q_minus_20_l1700_170026


namespace NUMINAMATH_GPT_evaluate_expression_l1700_170085

theorem evaluate_expression : 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 221 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1700_170085


namespace NUMINAMATH_GPT_percent_students_both_correct_l1700_170027

def percent_answered_both_questions (total_students first_correct second_correct neither_correct : ℕ) : ℕ :=
  let at_least_one_correct := total_students - neither_correct
  let total_individual_correct := first_correct + second_correct
  total_individual_correct - at_least_one_correct

theorem percent_students_both_correct
  (total_students : ℕ)
  (first_question_correct : ℕ)
  (second_question_correct : ℕ)
  (neither_question_correct : ℕ) 
  (h_total_students : total_students = 100)
  (h_first_correct : first_question_correct = 80)
  (h_second_correct : second_question_correct = 55)
  (h_neither_correct : neither_question_correct = 20) :
  percent_answered_both_questions total_students first_question_correct second_question_correct neither_question_correct = 55 :=
by
  rw [h_total_students, h_first_correct, h_second_correct, h_neither_correct]
  sorry


end NUMINAMATH_GPT_percent_students_both_correct_l1700_170027


namespace NUMINAMATH_GPT_average_speed_round_trip_l1700_170030

theorem average_speed_round_trip (d : ℝ) (h_d_pos : d > 0) : 
  let t1 := d / 80
  let t2 := d / 120
  let d_total := 2 * d
  let t_total := t1 + t2
  let v_avg := d_total / t_total
  v_avg = 96 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_round_trip_l1700_170030


namespace NUMINAMATH_GPT_smallest_s_for_347_l1700_170062

open Nat

theorem smallest_s_for_347 (r s : ℕ) (hr_pos : 0 < r) (hs_pos : 0 < s) 
  (h_rel_prime : Nat.gcd r s = 1) (h_r_lt_s : r < s) 
  (h_contains_347 : ∃ k : ℕ, ∃ y : ℕ, 10 ^ k * r - s * y = 347): 
  s = 653 := 
by sorry

end NUMINAMATH_GPT_smallest_s_for_347_l1700_170062


namespace NUMINAMATH_GPT_smallest_n_l1700_170049

theorem smallest_n (n : ℕ) : 17 * n ≡ 136 [MOD 5] → n = 3 := 
by sorry

end NUMINAMATH_GPT_smallest_n_l1700_170049
