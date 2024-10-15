import Mathlib

namespace NUMINAMATH_GPT_money_sum_l1590_159057

theorem money_sum (A B : ℕ) (h₁ : (1 / 3 : ℝ) * A = (1 / 4 : ℝ) * B) (h₂ : B = 484) : A + B = 847 := by
  sorry

end NUMINAMATH_GPT_money_sum_l1590_159057


namespace NUMINAMATH_GPT_cuboid_face_areas_l1590_159012

-- Conditions
variables (a b c S : ℝ)
-- Surface area of the sphere condition
theorem cuboid_face_areas 
  (h1 : a * b = 6) 
  (h2 : b * c = 10) 
  (h3 : a^2 + b^2 + c^2 = 76) 
  (h4 : 4 * π * 38 = 152 * π) :
  a * c = 15 :=
by 
  -- Prove that the solution matches the conclusion
  sorry

end NUMINAMATH_GPT_cuboid_face_areas_l1590_159012


namespace NUMINAMATH_GPT_dot_product_is_ten_l1590_159045

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the condition that the vectors are parallel
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 / v2.1 = v1.2 / v2.2

-- The main theorem statement
theorem dot_product_is_ten (m : ℝ) (h : parallel a (b m)) : 
  a.1 * (b m).1 + a.2 * (b m).2 = 10 := by
  sorry

end NUMINAMATH_GPT_dot_product_is_ten_l1590_159045


namespace NUMINAMATH_GPT_evaluate_fraction_l1590_159019

theorem evaluate_fraction (a b c : ℝ) (h : a^3 - b^3 + c^3 ≠ 0) :
  (a^6 - b^6 + c^6) / (a^3 - b^3 + c^3) = a^3 + b^3 + c^3 :=
sorry

end NUMINAMATH_GPT_evaluate_fraction_l1590_159019


namespace NUMINAMATH_GPT_sum_of_smallest_natural_numbers_l1590_159032

-- Define the problem statement
def satisfies_eq (A B : ℕ) := 360 / (A^3 / B) = 5

-- Prove that there exist natural numbers A and B such that 
-- satisfies_eq A B is true, and their sum is 9
theorem sum_of_smallest_natural_numbers :
  ∃ (A B : ℕ), satisfies_eq A B ∧ A + B = 9 :=
by
  -- Sorry is used here to indicate the proof is not given
  sorry

end NUMINAMATH_GPT_sum_of_smallest_natural_numbers_l1590_159032


namespace NUMINAMATH_GPT_distinct_pen_distribution_l1590_159089

theorem distinct_pen_distribution :
  ∃! (a b c d : ℕ), a + b + c + d = 10 ∧
                    1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧
                    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d :=
sorry

end NUMINAMATH_GPT_distinct_pen_distribution_l1590_159089


namespace NUMINAMATH_GPT_triangle_parallel_vectors_l1590_159016

noncomputable def collinear {V : Type*} [AddCommGroup V] [Module ℝ V]
  (P₁ P₂ P₃ : V) : Prop :=
∃ t : ℝ, P₃ = P₁ + t • (P₂ - P₁)

theorem triangle_parallel_vectors
  (A B C C₁ A₁ B₁ C₂ A₂ B₂ : ℝ × ℝ)
  (h1 : collinear A B C₁) (h2 : collinear B C A₁) (h3 : collinear C A B₁)
  (ratio1 : ∀ (AC1 CB : ℝ), AC1 / CB = 1) (ratio2 : ∀ (BA1 AC : ℝ), BA1 / AC = 1) (ratio3 : ∀ (CB B1A : ℝ), CB / B1A = 1)
  (h4 : collinear A₁ B₁ C₂) (h5 : collinear B₁ C₁ A₂) (h6 : collinear C₁ A₁ B₂)
  (n : ℝ)
  (ratio4 : ∀ (A1C2 C2B1 : ℝ), A1C2 / C2B1 = n) (ratio5 : ∀ (B1A2 A2C1 : ℝ), B1A2 / A2C1 = n) (ratio6 : ∀ (C1B2 B2A1 : ℝ), C1B2 / B2A1 = n) :
  collinear A C A₂ ∧ collinear C B C₂ ∧ collinear B A B₂ :=
sorry

end NUMINAMATH_GPT_triangle_parallel_vectors_l1590_159016


namespace NUMINAMATH_GPT_prime_p_square_condition_l1590_159041

theorem prime_p_square_condition (p : ℕ) (h_prime : Prime p) (h_square : ∃ n : ℤ, 5^p + 4 * p^4 = n^2) :
  p = 31 :=
sorry

end NUMINAMATH_GPT_prime_p_square_condition_l1590_159041


namespace NUMINAMATH_GPT_bottom_row_bricks_l1590_159088

theorem bottom_row_bricks {x : ℕ} 
  (c1 : ∀ i, i < 5 → (x - i) > 0)
  (c2 : x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 100) : 
  x = 22 := 
by
  sorry

end NUMINAMATH_GPT_bottom_row_bricks_l1590_159088


namespace NUMINAMATH_GPT_sum_of_powers_l1590_159075

theorem sum_of_powers : 5^5 + 5^5 + 5^5 + 5^5 = 4 * 5^5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_powers_l1590_159075


namespace NUMINAMATH_GPT_sum_of_constants_l1590_159006

theorem sum_of_constants (c d : ℝ) (h₁ : 16 = 2 * 4 + c) (h₂ : 16 = 4 * 4 + d) : c + d = 8 := by
  sorry

end NUMINAMATH_GPT_sum_of_constants_l1590_159006


namespace NUMINAMATH_GPT_total_chairs_taken_l1590_159034

def num_students : ℕ := 5
def chairs_per_trip : ℕ := 5
def num_trips : ℕ := 10

theorem total_chairs_taken :
  (num_students * chairs_per_trip * num_trips) = 250 :=
by
  sorry

end NUMINAMATH_GPT_total_chairs_taken_l1590_159034


namespace NUMINAMATH_GPT_kaleb_non_working_games_l1590_159033

theorem kaleb_non_working_games (total_games working_game_price earning : ℕ) (h1 : total_games = 10) (h2 : working_game_price = 6) (h3 : earning = 12) :
  total_games - (earning / working_game_price) = 8 :=
by
  sorry

end NUMINAMATH_GPT_kaleb_non_working_games_l1590_159033


namespace NUMINAMATH_GPT_final_color_all_blue_l1590_159080

-- Definitions based on the problem's initial conditions
def initial_blue_sheep : ℕ := 22
def initial_red_sheep : ℕ := 18
def initial_green_sheep : ℕ := 15

-- The final problem statement: prove that all sheep end up being blue
theorem final_color_all_blue (B R G : ℕ) 
  (hB : B = initial_blue_sheep) 
  (hR : R = initial_red_sheep) 
  (hG : G = initial_green_sheep) 
  (interaction : ∀ (B R G : ℕ), (B > 0 ∨ R > 0 ∨ G > 0) → (R ≡ G [MOD 3])) :
  ∃ b, b = B + R + G ∧ R = 0 ∧ G = 0 ∧ b % 3 = 1 ∧ B = b :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_final_color_all_blue_l1590_159080


namespace NUMINAMATH_GPT_range_of_m_l1590_159048

noncomputable def f (m x : ℝ) : ℝ := m * (x - 2 * m) * (x + m + 3)
noncomputable def g (x : ℝ) : ℝ := 2^x - 2

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) ∧ (∃ x : ℝ, x < -4 ∧ f m x * g x < 0) → (-4 < m ∧ m < -2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1590_159048


namespace NUMINAMATH_GPT_cube_volume_l1590_159037

theorem cube_volume (d : ℝ) (s : ℝ) (h : d = 3 * Real.sqrt 3) (h_s : s * Real.sqrt 3 = d) : s ^ 3 = 27 := by
  -- Assuming h: the formula for the given space diagonal
  -- Assuming h_s: the formula connecting side length and the space diagonal
  sorry

end NUMINAMATH_GPT_cube_volume_l1590_159037


namespace NUMINAMATH_GPT_correct_statements_l1590_159042

variable (a b : ℝ)

theorem correct_statements (hab : a * b > 0) :
  (|a + b| > |a| ∧ |a + b| > |a - b|) ∧ (¬ (|a + b| < |b|)) ∧ (¬ (|a + b| < |a - b|)) :=
by
  -- The proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_correct_statements_l1590_159042


namespace NUMINAMATH_GPT_find_f_3_l1590_159025

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f_3 (a b : ℝ) (h : f (-3) a b = 10) : f 3 a b = -26 :=
by sorry

end NUMINAMATH_GPT_find_f_3_l1590_159025


namespace NUMINAMATH_GPT_range_of_a_l1590_159081

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - a| < 4) ↔ (-5 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1590_159081


namespace NUMINAMATH_GPT_cost_of_five_plastic_chairs_l1590_159092

theorem cost_of_five_plastic_chairs (C T : ℕ) (h1 : 3 * C = T) (h2 : T + 2 * C = 55) : 5 * C = 55 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_five_plastic_chairs_l1590_159092


namespace NUMINAMATH_GPT_minimum_surface_area_of_combined_cuboids_l1590_159096

noncomputable def cuboid_combinations (l w h : ℕ) (n : ℕ) : ℕ :=
sorry

theorem minimum_surface_area_of_combined_cuboids :
  ∃ n, cuboid_combinations 2 1 3 3 = 4 ∧ n = 42 :=
sorry

end NUMINAMATH_GPT_minimum_surface_area_of_combined_cuboids_l1590_159096


namespace NUMINAMATH_GPT_find_ab_pairs_l1590_159015

theorem find_ab_pairs (a b s : ℕ) (a_pos : a > 0) (b_pos : b > 0) (s_gt_one : s > 1) :
  (a = 2^s ∧ b = 2^(2*s) - 1) ↔
  (∃ p k : ℕ, Prime p ∧ (a^2 + b + 1 = p^k) ∧
   (a^2 + b + 1 ∣ b^2 - a^3 - 1) ∧
   ¬ (a^2 + b + 1 ∣ (a + b - 1)^2)) :=
sorry

end NUMINAMATH_GPT_find_ab_pairs_l1590_159015


namespace NUMINAMATH_GPT_employee_B_paid_l1590_159046

variable (A B : ℝ)

/-- Two employees A and B are paid a total of Rs. 550 per week by their employer. 
A is paid 120 percent of the sum paid to B. -/
theorem employee_B_paid (h₁ : A + B = 550) (h₂ : A = 1.2 * B) : B = 250 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_employee_B_paid_l1590_159046


namespace NUMINAMATH_GPT_linear_if_abs_k_eq_1_l1590_159049

theorem linear_if_abs_k_eq_1 (k : ℤ) : |k| = 1 ↔ (k = 1 ∨ k = -1) := by
  sorry

end NUMINAMATH_GPT_linear_if_abs_k_eq_1_l1590_159049


namespace NUMINAMATH_GPT_parallelogram_height_l1590_159090

theorem parallelogram_height (A : ℝ) (b : ℝ) (h : ℝ) (h1 : A = 320) (h2 : b = 20) :
  h = A / b → h = 16 := by
  sorry

end NUMINAMATH_GPT_parallelogram_height_l1590_159090


namespace NUMINAMATH_GPT_largest_n_for_divisibility_l1590_159029

theorem largest_n_for_divisibility :
  ∃ (n : ℕ), n = 5 ∧ 3^n ∣ (4^27000 - 82) ∧ ¬ 3^(n + 1) ∣ (4^27000 - 82) :=
by
  sorry

end NUMINAMATH_GPT_largest_n_for_divisibility_l1590_159029


namespace NUMINAMATH_GPT_sum_in_base4_eq_in_base5_l1590_159011

def base4_to_base5 (n : ℕ) : ℕ := sorry -- Placeholder for the conversion function

theorem sum_in_base4_eq_in_base5 :
  base4_to_base5 (203 + 112 + 321) = 2222 := 
sorry

end NUMINAMATH_GPT_sum_in_base4_eq_in_base5_l1590_159011


namespace NUMINAMATH_GPT_fractional_equation_solution_l1590_159000

theorem fractional_equation_solution (x : ℝ) (h : x = 7) : (3 / (x - 3)) - 1 = 1 / (3 - x) := by
  sorry

end NUMINAMATH_GPT_fractional_equation_solution_l1590_159000


namespace NUMINAMATH_GPT_number_of_pupils_l1590_159078

theorem number_of_pupils
  (pupil_mark_wrong : ℕ)
  (pupil_mark_correct : ℕ)
  (average_increase : ℚ)
  (n : ℕ)
  (h1 : pupil_mark_wrong = 73)
  (h2 : pupil_mark_correct = 45)
  (h3 : average_increase = 1/2)
  (h4 : 28 / n = average_increase) : n = 56 := 
sorry

end NUMINAMATH_GPT_number_of_pupils_l1590_159078


namespace NUMINAMATH_GPT_plane_through_point_and_line_l1590_159035

noncomputable def plane_equation (x y z : ℝ) : Prop :=
  12 * x + 67 * y + 23 * z - 26 = 0

theorem plane_through_point_and_line :
  ∃ (A B C D : ℤ), 
  (A > 0) ∧ (Int.gcd (abs A) (Int.gcd (abs B) (Int.gcd (abs C) (abs D))) = 1) ∧
  (plane_equation 1 4 (-6)) ∧  
  ∀ t : ℝ, (plane_equation (4 * t + 2)  (-t - 1) (5 * t + 3)) :=
sorry

end NUMINAMATH_GPT_plane_through_point_and_line_l1590_159035


namespace NUMINAMATH_GPT_compute_expression_l1590_159068

theorem compute_expression : ((-5) * 3) - (7 * (-2)) + ((-4) * (-6)) = 23 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1590_159068


namespace NUMINAMATH_GPT_new_volume_is_correct_l1590_159023

variable (l w h : ℝ)

-- Conditions given in the problem
axiom volume : l * w * h = 4320
axiom surface_area : 2 * (l * w + w * h + h * l) = 1704
axiom edge_sum : 4 * (l + w + h) = 208

-- The proposition we need to prove:
theorem new_volume_is_correct : (l + 2) * (w + 2) * (h + 2) = 6240 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_new_volume_is_correct_l1590_159023


namespace NUMINAMATH_GPT_find_y_given_x_zero_l1590_159005

theorem find_y_given_x_zero (t : ℝ) (y : ℝ) : 
  (3 - 2 * t = 0) → (y = 3 * t + 6) → y = 21 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_given_x_zero_l1590_159005


namespace NUMINAMATH_GPT_find_m_range_l1590_159076

noncomputable def range_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : Prop :=
  m ≥ 4

-- Here is the theorem statement
theorem find_m_range (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : range_m a b c m h1 h2 h3 :=
sorry

end NUMINAMATH_GPT_find_m_range_l1590_159076


namespace NUMINAMATH_GPT_first_number_positive_l1590_159028

-- Define the initial condition
def initial_pair : ℕ × ℕ := (1, 1)

-- Define the allowable transformations
def transform1 (x y : ℕ) : Prop :=
(x, y - 1) = initial_pair ∨ (x + y, y + 1) = initial_pair

def transform2 (x y : ℕ) : Prop :=
(x, x * y) = initial_pair ∨ (1 / x, y) = initial_pair

-- Define discriminant function
def discriminant (a b : ℕ) : ℤ := b ^ 2 - 4 * a

-- Define the invariants maintained by the transformations
def invariant (a b : ℕ) : Prop :=
discriminant a b < 0

-- Statement to be proven
theorem first_number_positive :
(∀ (a b : ℕ), invariant a b → a > 0) :=
by
  sorry

end NUMINAMATH_GPT_first_number_positive_l1590_159028


namespace NUMINAMATH_GPT_mike_peaches_l1590_159052

theorem mike_peaches (initial_peaches picked_peaches : ℝ) (h1 : initial_peaches = 34.0) (h2 : picked_peaches = 86.0) : initial_peaches + picked_peaches = 120.0 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_mike_peaches_l1590_159052


namespace NUMINAMATH_GPT_division_problem_l1590_159018

theorem division_problem : 75 / 0.05 = 1500 := 
  sorry

end NUMINAMATH_GPT_division_problem_l1590_159018


namespace NUMINAMATH_GPT_find_a_of_extreme_value_at_one_l1590_159039

-- Define the function f(x) = x^3 - a * x
def f (x a : ℝ) : ℝ := x^3 - a * x
  
-- Define the derivative of f with respect to x
def f' (x a : ℝ) : ℝ := 3 * x^2 - a

-- The theorem statement: for f(x) having an extreme value at x = 1, the corresponding a must be 3
theorem find_a_of_extreme_value_at_one (a : ℝ) : 
  (f' 1 a = 0) ↔ (a = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_a_of_extreme_value_at_one_l1590_159039


namespace NUMINAMATH_GPT_prime_cubic_condition_l1590_159058

theorem prime_cubic_condition (p : ℕ) (hp : Nat.Prime p) (hp_prime : Nat.Prime (p^4 - 3 * p^2 + 9)) : p = 2 :=
sorry

end NUMINAMATH_GPT_prime_cubic_condition_l1590_159058


namespace NUMINAMATH_GPT_distance_AO_min_distance_BM_l1590_159008

open Real

-- Definition of rectangular distance
def rectangular_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

-- Point A and O
def A : ℝ × ℝ := (-1, 3)
def O : ℝ × ℝ := (0, 0)

-- Point B
def B : ℝ × ℝ := (1, 0)

-- Line "x - y + 2 = 0"
def on_line (M : ℝ × ℝ) : Prop :=
  M.1 - M.2 + 2 = 0

-- Proof statement 1: distance from A to O is 4
theorem distance_AO : rectangular_distance A O = 4 := 
sorry

-- Proof statement 2: minimum distance from B to any point on the line is 3
theorem min_distance_BM (M : ℝ × ℝ) (h : on_line M) : rectangular_distance B M = 3 := 
sorry

end NUMINAMATH_GPT_distance_AO_min_distance_BM_l1590_159008


namespace NUMINAMATH_GPT_circle_problem_l1590_159038

theorem circle_problem (P : ℝ × ℝ) (QR : ℝ) (S : ℝ × ℝ) (k : ℝ)
  (h1 : P = (5, 12))
  (h2 : QR = 5)
  (h3 : S = (0, k))
  (h4 : dist (0,0) P = 13) -- OP = 13 from the origin to point P
  (h5 : dist (0,0) S = 8) -- OQ = 8 from the origin to point S
: k = 8 ∨ k = -8 :=
by sorry

end NUMINAMATH_GPT_circle_problem_l1590_159038


namespace NUMINAMATH_GPT_prove_n_eq_1_l1590_159094

-- Definitions of the given conditions
def is_prime (x : ℕ) : Prop := Nat.Prime x

variable {p q r n : ℕ}
variable (hp : is_prime p) (hq : is_prime q) (hr : is_prime r)
variable (hn_pos : n > 0)
variable (h_eq : p^n + q^n = r^2)

-- Statement to prove
theorem prove_n_eq_1 : n = 1 :=
  sorry

end NUMINAMATH_GPT_prove_n_eq_1_l1590_159094


namespace NUMINAMATH_GPT_age_difference_in_decades_l1590_159027

-- Declare the ages of x, y, and z as real numbers
variables (x y z : ℝ)

-- Define the condition
def age_condition (x y z : ℝ) : Prop := x + y = y + z + 18

-- The proof problem statement
theorem age_difference_in_decades (h : age_condition x y z) : (x - z) / 10 = 1.8 :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end NUMINAMATH_GPT_age_difference_in_decades_l1590_159027


namespace NUMINAMATH_GPT_preston_high_school_teachers_l1590_159056

theorem preston_high_school_teachers 
  (num_students : ℕ)
  (classes_per_student : ℕ)
  (classes_per_teacher : ℕ)
  (students_per_class : ℕ)
  (teachers_per_class : ℕ)
  (H : num_students = 1500)
  (C : classes_per_student = 6)
  (T : classes_per_teacher = 5)
  (S : students_per_class = 30)
  (P : teachers_per_class = 1) : 
  (num_students * classes_per_student / students_per_class / classes_per_teacher = 60) :=
by sorry

end NUMINAMATH_GPT_preston_high_school_teachers_l1590_159056


namespace NUMINAMATH_GPT_problem_solution_l1590_159069

def eq_A (x : ℝ) : Prop := 2 * x = 7
def eq_B (x y : ℝ) : Prop := x^2 + y = 5
def eq_C (x : ℝ) : Prop := x = 1 / x + 1
def eq_D (x : ℝ) : Prop := x^2 + x = 4

def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ a * x^2 + b * x + c = 0

theorem problem_solution : is_quadratic eq_D := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1590_159069


namespace NUMINAMATH_GPT_base_number_is_five_l1590_159001

theorem base_number_is_five (x k : ℝ) (h1 : x^k = 5) (h2 : x^(2 * k + 2) = 400) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_base_number_is_five_l1590_159001


namespace NUMINAMATH_GPT_sin_315_eq_neg_sqrt_2_div_2_l1590_159085

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_315_eq_neg_sqrt_2_div_2_l1590_159085


namespace NUMINAMATH_GPT_handshakes_at_gathering_l1590_159036

def total_handshakes (num_couples : ℕ) (exceptions : ℕ) : ℕ :=
  let num_people := 2 * num_couples
  let handshakes_per_person := num_people - exceptions - 1
  num_people * handshakes_per_person / 2

theorem handshakes_at_gathering : total_handshakes 6 2 = 54 := by
  sorry

end NUMINAMATH_GPT_handshakes_at_gathering_l1590_159036


namespace NUMINAMATH_GPT_prime_geq_7_div_240_l1590_159044

theorem prime_geq_7_div_240 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 240 ∣ p^4 - 1 :=
sorry

end NUMINAMATH_GPT_prime_geq_7_div_240_l1590_159044


namespace NUMINAMATH_GPT_inequality_integral_ln_bounds_l1590_159004

-- Define the conditions
variables (x a : ℝ)
variables (hx : 0 < x) (ha : x < a)

-- First part: inequality involving integral
theorem inequality_integral (hx : 0 < x) (ha : x < a) :
  (2 * x / a) < (∫ t in a - x..a + x, 1 / t) ∧ (∫ t in a - x..a + x, 1 / t) < x * (1 / (a + x) + 1 / (a - x)) :=
sorry

-- Second part: to prove 0.68 < ln(2) < 0.71 using the result of the first part
theorem ln_bounds :
  0.68 < Real.log 2 ∧ Real.log 2 < 0.71 :=
sorry

end NUMINAMATH_GPT_inequality_integral_ln_bounds_l1590_159004


namespace NUMINAMATH_GPT_older_brother_pocket_money_l1590_159031

-- Definitions of the conditions
axiom sum_of_pocket_money (O Y : ℕ) : O + Y = 12000
axiom older_brother_more (O Y : ℕ) : O = Y + 1000

-- The statement to prove
theorem older_brother_pocket_money (O Y : ℕ) (h1 : O + Y = 12000) (h2 : O = Y + 1000) : O = 6500 :=
by
  exact sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_older_brother_pocket_money_l1590_159031


namespace NUMINAMATH_GPT_find_angle_B_l1590_159065

theorem find_angle_B 
  (a b : ℝ) (A B : ℝ) 
  (ha : a = 2 * Real.sqrt 2) 
  (hb : b = 2)
  (hA : A = Real.pi / 4) -- 45 degrees in radians
  (h_triangle : ∃ c, a^2 + b^2 - 2*a*b*Real.cos A = c^2 ∧ a^2 * Real.sin 45 = b^2 * Real.sin B) :
  B = Real.pi / 6 := -- 30 degrees in radians
sorry

end NUMINAMATH_GPT_find_angle_B_l1590_159065


namespace NUMINAMATH_GPT_bridge_length_is_correct_l1590_159097

noncomputable def train_length : ℝ := 135
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def bridge_crossing_time : ℝ := 30

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
noncomputable def total_distance_crossed : ℝ := train_speed_ms * bridge_crossing_time
noncomputable def bridge_length : ℝ := total_distance_crossed - train_length

theorem bridge_length_is_correct : bridge_length = 240 := by
  sorry

end NUMINAMATH_GPT_bridge_length_is_correct_l1590_159097


namespace NUMINAMATH_GPT_speed_of_stream_l1590_159043

theorem speed_of_stream (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 14) (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 3 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_speed_of_stream_l1590_159043


namespace NUMINAMATH_GPT_find_missing_number_l1590_159040

theorem find_missing_number (x : ℕ) : (4 + 3) + (8 - 3 - x) = 11 → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_number_l1590_159040


namespace NUMINAMATH_GPT_Sarah_brother_apples_l1590_159083

theorem Sarah_brother_apples (n : Nat) (h1 : 45 = 5 * n) : n = 9 := 
  sorry

end NUMINAMATH_GPT_Sarah_brother_apples_l1590_159083


namespace NUMINAMATH_GPT_problem_six_circles_l1590_159060

noncomputable def six_circles_centers : List (ℝ × ℝ) := [(1,1), (1,3), (3,1), (3,3), (5,1), (5,3)]

noncomputable def slope_of_line_dividing_circles := (2 : ℝ)

def gcd_is_1 (p q r : ℕ) : Prop := Nat.gcd (Nat.gcd p q) r = 1

theorem problem_six_circles (p q r : ℕ) (h_gcd : gcd_is_1 p q r)
  (h_line_eq : ∀ x y, y = slope_of_line_dividing_circles * x - 3 → px = qy + r) :
  p^2 + q^2 + r^2 = 14 :=
sorry

end NUMINAMATH_GPT_problem_six_circles_l1590_159060


namespace NUMINAMATH_GPT_min_k_squared_floor_l1590_159014

open Nat

theorem min_k_squared_floor (n : ℕ) :
  (∀ k : ℕ, k >= 1 → k^2 + (n / k^2) ≥ 1991) ∧
  (∃ k : ℕ, k >= 1 ∧ k^2 + (n / k^2) < 1992) ↔
  1024 * 967 ≤ n ∧ n ≤ 1024 * 967 + 1023 := 
by
  sorry

end NUMINAMATH_GPT_min_k_squared_floor_l1590_159014


namespace NUMINAMATH_GPT_evaluate_f_a_plus_1_l1590_159091

variable (a : ℝ)  -- The variable a is a real number.

def f (x : ℝ) : ℝ := x^2 + 1  -- The function f is defined as x^2 + 1.

theorem evaluate_f_a_plus_1 : f (a + 1) = a^2 + 2 * a + 2 := by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_evaluate_f_a_plus_1_l1590_159091


namespace NUMINAMATH_GPT_quadratic_roots_l1590_159047

theorem quadratic_roots (A B C : ℝ) (r s p : ℝ) (h1 : 2 * A * r^2 + 3 * B * r + 4 * C = 0)
  (h2 : 2 * A * s^2 + 3 * B * s + 4 * C = 0) (h3 : r + s = -3 * B / (2 * A)) (h4 : r * s = 2 * C / A) :
  p = (16 * A * C - 9 * B^2) / (4 * A^2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1590_159047


namespace NUMINAMATH_GPT_cornelia_age_l1590_159095

theorem cornelia_age :
  ∃ C : ℕ, 
  (∃ K : ℕ, K = 30 ∧ (C + 20 = 2 * (K + 20))) ∧
  ((K - 5)^2 = 3 * (C - 5)) := by
  sorry

end NUMINAMATH_GPT_cornelia_age_l1590_159095


namespace NUMINAMATH_GPT_analytical_expression_l1590_159055

theorem analytical_expression (k : ℝ) (h : k ≠ 0) (x y : ℝ) (hx : x = 4) (hy : y = 6) 
  (eqn : y = k * x) : y = (3 / 2) * x :=
by {
  sorry
}

end NUMINAMATH_GPT_analytical_expression_l1590_159055


namespace NUMINAMATH_GPT_weaving_problem_l1590_159087

theorem weaving_problem
  (a : ℕ → ℝ) -- the sequence
  (a_arith_seq : ∀ n, a n = a 0 + n * (a 1 - a 0)) -- arithmetic sequence condition
  (sum_seven_days : 7 * a 0 + 21 * (a 1 - a 0) = 21) -- sum in seven days
  (sum_days_2_5_8 : 3 * a 1 + 12 * (a 1 - a 0) = 15) -- sum on 2nd, 5th, and 8th days
  : a 10 = 15 := sorry

end NUMINAMATH_GPT_weaving_problem_l1590_159087


namespace NUMINAMATH_GPT_linear_term_coefficient_l1590_159030

theorem linear_term_coefficient : (x - 1) * (1 / x + x) ^ 6 = a + b * x + c * x^2 + d * x^3 + e * x^4 + f * x^5 + g * x^6 →
  b = 20 :=
by
  sorry

end NUMINAMATH_GPT_linear_term_coefficient_l1590_159030


namespace NUMINAMATH_GPT_time_with_walkway_l1590_159051

-- Definitions
def length_walkway : ℝ := 60
def time_against_walkway : ℝ := 120
def time_stationary_walkway : ℝ := 48

-- Theorem statement
theorem time_with_walkway (v w : ℝ)
  (h1 : 60 = 120 * (v - w))
  (h2 : 60 = 48 * v)
  (h3 : v = 1.25)
  (h4 : w = 0.75) :
  60 = 30 * (v + w) :=
by
  sorry

end NUMINAMATH_GPT_time_with_walkway_l1590_159051


namespace NUMINAMATH_GPT_a_b_total_money_l1590_159073

variable (A B : ℝ)

theorem a_b_total_money (h1 : (4 / 15) * A = (2 / 5) * 484) (h2 : B = 484) : A + B = 1210 := by
  sorry

end NUMINAMATH_GPT_a_b_total_money_l1590_159073


namespace NUMINAMATH_GPT_find_ab_l1590_159010

theorem find_ab (a b : ℝ) (h₁ : a - b = 3) (h₂ : a^2 + b^2 = 29) : a * b = 10 :=
sorry

end NUMINAMATH_GPT_find_ab_l1590_159010


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l1590_159066

-- Given conditions translated into Lean definitions
variables (AB AD AC1 AA1 : ℕ)

def rectangular_prism_properties : Prop :=
  AB = 2 ∧ AD = 2 ∧ AC1 = 3 ∧ AA1 = 1

-- The mathematical volume of the rectangular prism
def volume (AB AD AA1 : ℕ) := AB * AD * AA1

-- Prove that given the conditions, the volume of the rectangular prism is 4
theorem volume_of_rectangular_prism (h : rectangular_prism_properties AB AD AC1 AA1) : volume AB AD AA1 = 4 :=
by
  sorry

#check volume_of_rectangular_prism

end NUMINAMATH_GPT_volume_of_rectangular_prism_l1590_159066


namespace NUMINAMATH_GPT_rhombus_area_l1590_159017

def diagonal1 : ℝ := 24
def diagonal2 : ℝ := 16

theorem rhombus_area : 0.5 * diagonal1 * diagonal2 = 192 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l1590_159017


namespace NUMINAMATH_GPT_smallest_integer_inequality_l1590_159070

theorem smallest_integer_inequality:
  ∃ x : ℤ, (2 * x < 3 * x - 10) ∧ ∀ y : ℤ, (2 * y < 3 * y - 10) → y ≥ 11 := by
  sorry

end NUMINAMATH_GPT_smallest_integer_inequality_l1590_159070


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1590_159021

noncomputable def hyperbola_eccentricity : ℝ → ℝ → ℝ → ℝ
| p, a, b => 
  let c := p / 2
  let e := c / a
  have h₁ : 9 * e^2 - 12 * e^2 / (e^2 - 1) = 1 := sorry
  e

theorem eccentricity_of_hyperbola (p a b : ℝ) (hp : p > 0) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity p a b = (Real.sqrt 7 + 2) / 3 :=
sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1590_159021


namespace NUMINAMATH_GPT_inequality_solution_l1590_159063

theorem inequality_solution (a b : ℝ)
  (h : ∀ x : ℝ, 0 ≤ x → (x^2 + 1 ≥ a * x + b ∧ a * x + b ≥ (3 / 2) * x^(2 / 3) )) :
  (2 - Real.sqrt 2) / 4 ≤ b ∧ b ≤ (2 + Real.sqrt 2) / 4 ∧
  (1 / Real.sqrt (2 * b)) ≤ a ∧ a ≤ 2 * Real.sqrt (1 - b) :=
  sorry

end NUMINAMATH_GPT_inequality_solution_l1590_159063


namespace NUMINAMATH_GPT_valid_square_numbers_l1590_159024

noncomputable def is_valid_number (N P Q : ℕ) (q : ℕ) : Prop :=
  N = P * 10^q + Q ∧ N = 2 * P * Q

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem valid_square_numbers : 
  ∀ (N : ℕ), (∃ (P Q : ℕ) (q : ℕ), is_valid_number N P Q q) → is_perfect_square N :=
sorry

end NUMINAMATH_GPT_valid_square_numbers_l1590_159024


namespace NUMINAMATH_GPT_coprime_condition_exists_l1590_159002

theorem coprime_condition_exists : ∃ (A B C : ℕ), (A > 0 ∧ B > 0 ∧ C > 0) ∧ (Nat.gcd (Nat.gcd A B) C = 1) ∧ 
  (A * Real.log 5 / Real.log 50 + B * Real.log 2 / Real.log 50 = C) ∧ (A + B + C = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_coprime_condition_exists_l1590_159002


namespace NUMINAMATH_GPT_translated_parabola_expression_correct_l1590_159007

-- Definitions based on the conditions
def original_parabola (x : ℝ) : ℝ := x^2 - 1
def translated_parabola (x : ℝ) : ℝ := (x + 2)^2

-- The theorem to prove
theorem translated_parabola_expression_correct :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) + 1 :=
by
  sorry

end NUMINAMATH_GPT_translated_parabola_expression_correct_l1590_159007


namespace NUMINAMATH_GPT_find_circle_center_l1590_159020

noncomputable def circle_center : (ℝ × ℝ) :=
  let x_center := 5
  let y_center := 4
  (x_center, y_center)

theorem find_circle_center (x y : ℝ) (h : x^2 - 10 * x + y^2 - 8 * y = 16) :
  circle_center = (5, 4) := by
  sorry

end NUMINAMATH_GPT_find_circle_center_l1590_159020


namespace NUMINAMATH_GPT_age_problem_l1590_159009

theorem age_problem (x y : ℕ) 
  (h1 : 3 * x = 4 * y) 
  (h2 : 3 * y - x = 140) : x = 112 ∧ y = 84 := 
by 
  sorry

end NUMINAMATH_GPT_age_problem_l1590_159009


namespace NUMINAMATH_GPT_longer_side_is_40_l1590_159071

-- Given the conditions
variable (small_rect_width : ℝ) (small_rect_length : ℝ)
variable (num_rects : ℕ)

-- Conditions 
axiom rect_width_is_10 : small_rect_width = 10
axiom length_is_twice_width : small_rect_length = 2 * small_rect_width
axiom four_rectangles : num_rects = 4

-- Prove length of the longer side of the large rectangle
theorem longer_side_is_40 :
  small_rect_width = 10 → small_rect_length = 2 * small_rect_width → num_rects = 4 →
  (2 * small_rect_length) = 40 := sorry

end NUMINAMATH_GPT_longer_side_is_40_l1590_159071


namespace NUMINAMATH_GPT_roots_of_quadratic_l1590_159026

theorem roots_of_quadratic (a b : ℝ) (h₁ : a + b = 2) (h₂ : a * b = -3) : a^2 + b^2 = 10 := 
by
  -- proof steps go here, but not required as per the instruction
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1590_159026


namespace NUMINAMATH_GPT_units_digit_of_7_pow_6_pow_5_l1590_159099

theorem units_digit_of_7_pow_6_pow_5 : ((7 : ℕ)^ (6^5) % 10) = 1 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_7_pow_6_pow_5_l1590_159099


namespace NUMINAMATH_GPT_smallest_denominator_between_l1590_159077

theorem smallest_denominator_between :
  ∃ (a b : ℕ), b > 0 ∧ a < b ∧ 6 / 17 < (a : ℚ) / b ∧ (a : ℚ) / b < 9 / 25 ∧ (∀ (c d : ℕ), d > 0 → c < d → 6 / 17 < (c : ℚ) / d → (c : ℚ) / d < 9 / 25 → b ≤ d) ∧ a = 5 ∧ b = 14 :=
by
  existsi 5
  existsi 14
  sorry

end NUMINAMATH_GPT_smallest_denominator_between_l1590_159077


namespace NUMINAMATH_GPT_sum_digits_single_digit_l1590_159067

theorem sum_digits_single_digit (n : ℕ) (h : n = 2^100) : (n % 9) = 7 := 
sorry

end NUMINAMATH_GPT_sum_digits_single_digit_l1590_159067


namespace NUMINAMATH_GPT_intersection_empty_l1590_159050

def A : Set ℝ := {x | x^2 + 2 * x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end NUMINAMATH_GPT_intersection_empty_l1590_159050


namespace NUMINAMATH_GPT_jean_total_cost_l1590_159093

theorem jean_total_cost 
  (num_pants : ℕ)
  (original_price_per_pant : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (num_pants_eq : num_pants = 10)
  (original_price_per_pant_eq : original_price_per_pant = 45)
  (discount_rate_eq : discount_rate = 0.2)
  (tax_rate_eq : tax_rate = 0.1) : 
  ∃ total_cost : ℝ, total_cost = 396 :=
by
  sorry

end NUMINAMATH_GPT_jean_total_cost_l1590_159093


namespace NUMINAMATH_GPT_derivative_at_pi_over_4_l1590_159074

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_at_pi_over_4 :
  deriv f (π / 4) = (Real.sqrt 2 / 2) + (Real.sqrt 2 * π / 8) :=
by
  -- Since the focus is only on the statement, the proof is not required.
  sorry

end NUMINAMATH_GPT_derivative_at_pi_over_4_l1590_159074


namespace NUMINAMATH_GPT_f_one_value_l1590_159003

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h_f_defined : ∀ x, x > 0 → ∃ y, f x = y
axiom h_f_strict_increasing : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom h_f_eq : ∀ x, x > 0 → f x * f (f x + 1/x) = 1

theorem f_one_value : f 1 = (1 + Real.sqrt 5) / 2 := 
by
  sorry

end NUMINAMATH_GPT_f_one_value_l1590_159003


namespace NUMINAMATH_GPT_reflect_P_across_x_axis_l1590_159084

def point_reflection_over_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem reflect_P_across_x_axis : 
  point_reflection_over_x_axis (-3, 1) = (-3, -1) :=
  by
    sorry

end NUMINAMATH_GPT_reflect_P_across_x_axis_l1590_159084


namespace NUMINAMATH_GPT_max_S_n_of_arithmetic_seq_l1590_159082

theorem max_S_n_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a 1 + n * d)
  (h2 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h3 : a 1 + a 3 + a 5 = 15)
  (h4 : a 2 + a 4 + a 6 = 0) : 
  ∃ n : ℕ, S n = 40 ∧ (∀ m : ℕ, S m ≤ 40) :=
sorry

end NUMINAMATH_GPT_max_S_n_of_arithmetic_seq_l1590_159082


namespace NUMINAMATH_GPT_minimum_value_expression_l1590_159053

theorem minimum_value_expression (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  4 ≤ (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (2 * q) / (p + r) :=
by sorry

end NUMINAMATH_GPT_minimum_value_expression_l1590_159053


namespace NUMINAMATH_GPT_coefficient_a7_l1590_159072

theorem coefficient_a7 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) (x : ℝ) 
  (h : x^9 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 
          + a_4 * (x - 1)^4 + a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7 
          + a_8 * (x - 1)^8 + a_9 * (x - 1)^9) : 
  a_7 = 36 := 
by
  sorry

end NUMINAMATH_GPT_coefficient_a7_l1590_159072


namespace NUMINAMATH_GPT_solve_inequality_l1590_159059

open Real

theorem solve_inequality (f : ℝ → ℝ)
  (h_cos : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f (cos x) ≥ 0) :
  ∀ k : ℤ, ∀ x, (2 * ↑k * π ≤ x ∧ x ≤ 2 * ↑k * π + π) → f (sin x) ≥ 0 :=
by
  intros k x hx
  sorry

end NUMINAMATH_GPT_solve_inequality_l1590_159059


namespace NUMINAMATH_GPT_wall_length_l1590_159013

theorem wall_length (s : ℕ) (w : ℕ) (a_ratio : ℕ) (A_mirror : ℕ) (A_wall : ℕ) (L : ℕ) 
  (hs : s = 24) (hw : w = 42) (h_ratio : a_ratio = 2) 
  (hA_mirror : A_mirror = s * s) 
  (hA_wall : A_wall = A_mirror * a_ratio) 
  (h_area : A_wall = w * L) : L = 27 :=
  sorry

end NUMINAMATH_GPT_wall_length_l1590_159013


namespace NUMINAMATH_GPT_sqrt_nested_expr_l1590_159054

theorem sqrt_nested_expr (x : ℝ) (hx : 0 ≤ x) : 
  (x * (x * (x * x)^(1 / 2))^(1 / 2))^(1 / 2) = (x^7)^(1 / 4) :=
sorry

end NUMINAMATH_GPT_sqrt_nested_expr_l1590_159054


namespace NUMINAMATH_GPT_tan_of_fourth_quadrant_l1590_159062

theorem tan_of_fourth_quadrant (α : ℝ) (h₁ : Real.sin α = -5 / 13) (h₂ : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) : Real.tan α = -5 / 12 :=
sorry

end NUMINAMATH_GPT_tan_of_fourth_quadrant_l1590_159062


namespace NUMINAMATH_GPT_max_a_for_f_l1590_159086

theorem max_a_for_f :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |a * x^2 - a * x + 1| ≤ 1) → a ≤ 8 :=
sorry

end NUMINAMATH_GPT_max_a_for_f_l1590_159086


namespace NUMINAMATH_GPT_handshake_count_l1590_159064

theorem handshake_count (num_companies : ℕ) (num_representatives : ℕ) 
  (total_handshakes : ℕ) (h1 : num_companies = 5) (h2 : num_representatives = 5)
  (h3 : total_handshakes = (num_companies * num_representatives * 
   (num_companies * num_representatives - 1 - (num_representatives - 1)) / 2)) :
  total_handshakes = 250 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_handshake_count_l1590_159064


namespace NUMINAMATH_GPT_average_score_for_entire_class_l1590_159022

def total_students : ℕ := 100
def assigned_day_percentage : ℝ := 0.70
def make_up_day_percentage : ℝ := 0.30
def assigned_day_avg_score : ℝ := 65
def make_up_day_avg_score : ℝ := 95

theorem average_score_for_entire_class :
  (assigned_day_percentage * total_students * assigned_day_avg_score + make_up_day_percentage * total_students * make_up_day_avg_score) / total_students = 74 := by
  sorry

end NUMINAMATH_GPT_average_score_for_entire_class_l1590_159022


namespace NUMINAMATH_GPT_meaningful_fraction_range_l1590_159079

theorem meaningful_fraction_range (x : ℝ) : (3 - x) ≠ 0 ↔ x ≠ 3 :=
by sorry

end NUMINAMATH_GPT_meaningful_fraction_range_l1590_159079


namespace NUMINAMATH_GPT_solution_set_eq_two_l1590_159098

theorem solution_set_eq_two (m : ℝ) (h : ∀ x : ℝ, mx + 2 > 0 ↔ x < 2) :
  m = -1 :=
sorry

end NUMINAMATH_GPT_solution_set_eq_two_l1590_159098


namespace NUMINAMATH_GPT_sequence_product_mod_five_l1590_159061

theorem sequence_product_mod_five : 
  let seq := List.range 20 |>.map (λ k => 10 * k + 3)
  seq.prod % 5 = 1 := 
by
  sorry

end NUMINAMATH_GPT_sequence_product_mod_five_l1590_159061
