import Mathlib

namespace NUMINAMATH_GPT_find_pairs_of_positive_integers_l2299_229996

theorem find_pairs_of_positive_integers (n m : ℕ) (hn : 0 < n) (hm : 0 < m) : 
  3 * 2^m + 1 = n^2 ↔ (n = 7 ∧ m = 4) ∨ (n = 5 ∧ m = 3) :=
sorry

end NUMINAMATH_GPT_find_pairs_of_positive_integers_l2299_229996


namespace NUMINAMATH_GPT_remainder_when_13_plus_y_divided_by_31_l2299_229934

theorem remainder_when_13_plus_y_divided_by_31
  (y : ℕ)
  (hy : 7 * y % 31 = 1) :
  (13 + y) % 31 = 22 :=
sorry

end NUMINAMATH_GPT_remainder_when_13_plus_y_divided_by_31_l2299_229934


namespace NUMINAMATH_GPT_range_of_a1_l2299_229974

noncomputable def sequence_a (n : ℕ) : ℤ := sorry
noncomputable def sum_S (n : ℕ) : ℤ := sorry

theorem range_of_a1 :
  (∀ n : ℕ, n > 0 → sum_S n + sum_S (n+1) = 2 * n^2 + n) ∧
  (∀ n : ℕ, n > 0 → sequence_a n < sequence_a (n+1)) →
  -1/4 < sequence_a 1 ∧ sequence_a 1 < 3/4 := sorry

end NUMINAMATH_GPT_range_of_a1_l2299_229974


namespace NUMINAMATH_GPT_construct_one_degree_l2299_229993

theorem construct_one_degree (theta : ℝ) (h : theta = 19) : 1 = 19 * theta - 360 :=
by
  -- Proof here will be filled
  sorry

end NUMINAMATH_GPT_construct_one_degree_l2299_229993


namespace NUMINAMATH_GPT_angle_is_120_degrees_l2299_229992

-- Define the magnitudes of vectors a and b and their dot product
def magnitude_a : ℝ := 10
def magnitude_b : ℝ := 12
def dot_product_ab : ℝ := -60

-- Define the angle between vectors a and b
def angle_between_vectors (θ : ℝ) : Prop :=
  magnitude_a * magnitude_b * Real.cos θ = dot_product_ab

-- Prove that the angle θ is 120 degrees
theorem angle_is_120_degrees : angle_between_vectors (2 * Real.pi / 3) :=
by 
  unfold angle_between_vectors
  sorry

end NUMINAMATH_GPT_angle_is_120_degrees_l2299_229992


namespace NUMINAMATH_GPT_point_on_line_y_coordinate_l2299_229979

variables (m b x : ℝ)

def line_equation := m * x + b

theorem point_on_line_y_coordinate : m = 4 → b = 4 → x = 199 → line_equation m b x = 800 :=
by 
  intros h_m h_b h_x
  unfold line_equation
  rw [h_m, h_b, h_x]
  norm_num
  done

end NUMINAMATH_GPT_point_on_line_y_coordinate_l2299_229979


namespace NUMINAMATH_GPT_gcd_problem_l2299_229950

variable (A B : ℕ)
variable (hA : A = 2 * 3 * 5)
variable (hB : B = 2 * 2 * 5 * 7)

theorem gcd_problem : Nat.gcd A B = 10 :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_gcd_problem_l2299_229950


namespace NUMINAMATH_GPT_perpendicular_vectors_solution_l2299_229944

theorem perpendicular_vectors_solution (m : ℝ) (a : ℝ × ℝ := (m-1, 2)) (b : ℝ × ℝ := (m, -3)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : m = 3 ∨ m = -2 :=
by sorry

end NUMINAMATH_GPT_perpendicular_vectors_solution_l2299_229944


namespace NUMINAMATH_GPT_coaching_fee_correct_l2299_229911

noncomputable def total_coaching_fee : ℝ :=
  let daily_fee : ℝ := 39
  let discount_threshold : ℝ := 50
  let discount_rate : ℝ := 0.10
  let total_days : ℝ := 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 3 -- non-leap year days count up to Nov 3
  let discount_days : ℝ := total_days - discount_threshold
  let discounted_fee : ℝ := daily_fee * (1 - discount_rate)
  let fee_before_discount : ℝ := discount_threshold * daily_fee
  let fee_after_discount : ℝ := discount_days * discounted_fee
  fee_before_discount + fee_after_discount

theorem coaching_fee_correct :
  total_coaching_fee = 10967.7 := by
  sorry

end NUMINAMATH_GPT_coaching_fee_correct_l2299_229911


namespace NUMINAMATH_GPT_expression_evaluation_l2299_229973

theorem expression_evaluation :
  (-2: ℤ)^3 + ((36: ℚ) / (3: ℚ)^2 * (-1 / 2: ℚ)) + abs (-5: ℤ) = -5 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2299_229973


namespace NUMINAMATH_GPT_smallest_d_for_inverse_l2299_229937

noncomputable def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse : ∃ d : ℝ, (∀ x1 x2, x1 ≥ d → x2 ≥ d → g x1 = g x2 → x1 = x2) ∧ d = 3 := 
sorry

end NUMINAMATH_GPT_smallest_d_for_inverse_l2299_229937


namespace NUMINAMATH_GPT_sum_of_areas_of_squares_l2299_229947

def is_right_angle (a b c : ℝ) : Prop := (a^2 + b^2 = c^2)

def isSquare (side : ℝ) : Prop := (side > 0)

def area_of_square (side : ℝ) : ℝ := side^2

theorem sum_of_areas_of_squares 
  (P Q R S X Y : ℝ) 
  (h1 : is_right_angle P Q R)
  (h2 : PR = 15)
  (h3 : isSquare PR)
  (h4 : isSquare PQ) :
  area_of_square PR + area_of_square PQ = 450 := 
sorry


end NUMINAMATH_GPT_sum_of_areas_of_squares_l2299_229947


namespace NUMINAMATH_GPT_fixed_point_of_f_l2299_229976

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.logb a (|x + 1|)

theorem fixed_point_of_f (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 0 = 1 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_f_l2299_229976


namespace NUMINAMATH_GPT_nth_term_correct_l2299_229961

noncomputable def term_in_sequence (n : ℕ) : ℚ :=
  2^n / (2^n + 3)

theorem nth_term_correct (n : ℕ) : term_in_sequence n = 2^n / (2^n + 3) :=
by
  sorry

end NUMINAMATH_GPT_nth_term_correct_l2299_229961


namespace NUMINAMATH_GPT_evaluate_product_l2299_229945

theorem evaluate_product (n : ℤ) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = n^5 - n^4 - 5 * n^3 + 4 * n^2 + 4 * n := 
by
  -- Omitted proof steps
  sorry

end NUMINAMATH_GPT_evaluate_product_l2299_229945


namespace NUMINAMATH_GPT_sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq_l2299_229956

-- Define the real interval [0, π/2]
def interval_0_pi_over_2 (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ Real.pi / 2

-- Define the proposition to be proven
theorem sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq (a b : ℝ) 
  (ha : interval_0_pi_over_2 a) (hb : interval_0_pi_over_2 b) :
  (Real.sin a)^6 + 3 * (Real.sin a)^2 * (Real.cos b)^2 + (Real.cos b)^6 = 1 ↔ a = b :=
by
  sorry

end NUMINAMATH_GPT_sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq_l2299_229956


namespace NUMINAMATH_GPT_simplify_sqrt_l2299_229941

theorem simplify_sqrt (x : ℝ) (h : x < 2) : Real.sqrt (x^2 - 4*x + 4) = 2 - x :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_l2299_229941


namespace NUMINAMATH_GPT_graph_not_passing_through_origin_l2299_229940

theorem graph_not_passing_through_origin (m : ℝ) (h : 3 * m^2 - 2 * m ≠ 0) : m = -(1 / 3) :=
sorry

end NUMINAMATH_GPT_graph_not_passing_through_origin_l2299_229940


namespace NUMINAMATH_GPT_find_functions_l2299_229960

variable (f : ℝ → ℝ)

theorem find_functions (h : ∀ x y : ℝ, f (x + f y) = f x + f y ^ 2 + 2 * x * f y) :
  ∃ c : ℝ, (∀ x, f x = x ^ 2 + c) ∨ (∀ x, f x = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_functions_l2299_229960


namespace NUMINAMATH_GPT_basketball_team_initial_games_l2299_229901

theorem basketball_team_initial_games (G W : ℝ) 
  (h1 : W = 0.70 * G) 
  (h2 : W + 2 = 0.60 * (G + 10)) : 
  G = 40 :=
by
  sorry

end NUMINAMATH_GPT_basketball_team_initial_games_l2299_229901


namespace NUMINAMATH_GPT_domain_h_l2299_229942

def domain_f : Set ℝ := Set.Icc (-12) 6
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-3*x)

theorem domain_h {f : ℝ → ℝ} (hf : ∀ x, x ∈ domain_f → f x ∈ Set.univ) {x : ℝ} :
  h f x ∈ Set.univ ↔ x ∈ Set.Icc (-2) 4 :=
by
  sorry

end NUMINAMATH_GPT_domain_h_l2299_229942


namespace NUMINAMATH_GPT_no_tiling_triminos_l2299_229949

theorem no_tiling_triminos (board_size : ℕ) (trimino_size : ℕ) (remaining_squares : ℕ) 
  (H_board : board_size = 8) (H_trimino : trimino_size = 3) (H_remaining : remaining_squares = 63) : 
  ¬ ∃ (triminos : ℕ), triminos * trimino_size = remaining_squares :=
by {
  sorry
}

end NUMINAMATH_GPT_no_tiling_triminos_l2299_229949


namespace NUMINAMATH_GPT_find_number_l2299_229915

theorem find_number (n : ℕ) (some_number : ℕ) 
  (h : (1/5 : ℝ)^n * (1/4 : ℝ)^(18 : ℕ) = 1 / (2 * (some_number : ℝ)^n))
  (hn : n = 35) : some_number = 10 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l2299_229915


namespace NUMINAMATH_GPT_course_selection_plans_l2299_229912

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem course_selection_plans :
  let A_courses := C 4 2
  let B_courses := C 4 3
  let C_courses := C 4 3
  A_courses * B_courses * C_courses = 96 :=
by
  sorry

end NUMINAMATH_GPT_course_selection_plans_l2299_229912


namespace NUMINAMATH_GPT_parallel_heater_time_l2299_229919

theorem parallel_heater_time (t1 t2 : ℕ) (R1 R2 : ℝ) (t : ℕ) (I : ℝ) (Q : ℝ) (h₁ : t1 = 3) 
  (h₂ : t2 = 6) (hq1 : Q = I^2 * R1 * t1) (hq2 : Q = I^2 * R2 * t2) :
  t = (t1 * t2) / (t1 + t2) := by
  sorry

end NUMINAMATH_GPT_parallel_heater_time_l2299_229919


namespace NUMINAMATH_GPT_man_older_than_son_l2299_229930

variables (S M : ℕ)

theorem man_older_than_son (h1 : S = 32) (h2 : M + 2 = 2 * (S + 2)) : M - S = 34 :=
by
  sorry

end NUMINAMATH_GPT_man_older_than_son_l2299_229930


namespace NUMINAMATH_GPT_negation_of_existence_l2299_229975

theorem negation_of_existence :
  ¬ (∃ x : ℝ, x^2 > 2) ↔ ∀ x : ℝ, x^2 ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l2299_229975


namespace NUMINAMATH_GPT_hare_overtakes_tortoise_l2299_229989

noncomputable def hare_distance (t: ℕ) : ℕ := 
  if t ≤ 5 then 10 * t
  else if t ≤ 20 then 50
  else 50 + 20 * (t - 20)

noncomputable def tortoise_distance (t: ℕ) : ℕ :=
  2 * t

theorem hare_overtakes_tortoise : 
  ∃ t : ℕ, t ≤ 60 ∧ hare_distance t = tortoise_distance t ∧ 60 - t = 22 :=
sorry

end NUMINAMATH_GPT_hare_overtakes_tortoise_l2299_229989


namespace NUMINAMATH_GPT_evaluate_expression_l2299_229910

theorem evaluate_expression : 68 + (126 / 18) + (35 * 13) - 300 - (420 / 7) = 170 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2299_229910


namespace NUMINAMATH_GPT_number_of_Cl_atoms_l2299_229953

def atomic_weight_H : ℝ := 1
def atomic_weight_Cl : ℝ := 35.5
def atomic_weight_O : ℝ := 16

def H_atoms : ℕ := 1
def O_atoms : ℕ := 2
def total_molecular_weight : ℝ := 68

theorem number_of_Cl_atoms :
  (total_molecular_weight - (H_atoms * atomic_weight_H + O_atoms * atomic_weight_O)) / atomic_weight_Cl = 1 :=
by
  -- proof to show this holds
  sorry

end NUMINAMATH_GPT_number_of_Cl_atoms_l2299_229953


namespace NUMINAMATH_GPT_smallest_number_remainder_l2299_229946

open Nat

theorem smallest_number_remainder
  (b : ℕ)
  (h1 : b % 4 = 2)
  (h2 : b % 3 = 2)
  (h3 : b % 5 = 3) :
  b = 38 :=
sorry

end NUMINAMATH_GPT_smallest_number_remainder_l2299_229946


namespace NUMINAMATH_GPT_fourth_number_is_8_l2299_229938

theorem fourth_number_is_8 (a b c : ℕ) (mean : ℕ) (h_mean : mean = 20) (h_a : a = 12) (h_b : b = 24) (h_c : c = 36) :
  ∃ d : ℕ, mean * 4 = a + b + c + d ∧ (∃ x : ℕ, d = x^2) ∧ d = 8 := by
sorry

end NUMINAMATH_GPT_fourth_number_is_8_l2299_229938


namespace NUMINAMATH_GPT_point_comparison_on_inverse_proportion_l2299_229958

theorem point_comparison_on_inverse_proportion :
  (∃ y1 y2, (y1 = 2 / 1) ∧ (y2 = 2 / 2) ∧ y1 > y2) :=
by
  use 2
  use 1
  sorry

end NUMINAMATH_GPT_point_comparison_on_inverse_proportion_l2299_229958


namespace NUMINAMATH_GPT_geometric_sequence_q_and_an_l2299_229935

theorem geometric_sequence_q_and_an
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_pos : q > 0)
  (h2_eq : a 2 = 1)
  (h2_h6_eq_9h4 : a 2 * a 6 = 9 * a 4) :
  q = 3 ∧ ∀ n, a n = 3^(n - 2) := by
sorry

end NUMINAMATH_GPT_geometric_sequence_q_and_an_l2299_229935


namespace NUMINAMATH_GPT_odd_expression_divisible_by_48_l2299_229966

theorem odd_expression_divisible_by_48 (x : ℤ) (h : Odd x) : 48 ∣ (x^3 + 3*x^2 - x - 3) :=
  sorry

end NUMINAMATH_GPT_odd_expression_divisible_by_48_l2299_229966


namespace NUMINAMATH_GPT_locus_of_point_P_l2299_229923

/-- Given three points in the coordinate plane A(0,3), B(-√3, 0), and C(√3, 0), 
    and a point P on the coordinate plane such that PA = PB + PC, 
    determine the equation of the locus of point P. -/
noncomputable def locus_equation : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | (P.1^2 + (P.2 - 1)^2 = 4) ∧ (P.2 ≤ 0)}

theorem locus_of_point_P :
  ∀ (P : ℝ × ℝ),
  (∃ A B C : ℝ × ℝ, A = (0, 3) ∧ B = (-Real.sqrt 3, 0) ∧ C = (Real.sqrt 3, 0) ∧ 
     dist P A = dist P B + dist P C) →
  P ∈ locus_equation :=
by
  intros P hp
  sorry

end NUMINAMATH_GPT_locus_of_point_P_l2299_229923


namespace NUMINAMATH_GPT_exists_m_n_for_any_d_l2299_229933

theorem exists_m_n_for_any_d (d : ℤ) : ∃ m n : ℤ, d = (n - 2 * m + 1) / (m^2 - n) :=
by
  sorry

end NUMINAMATH_GPT_exists_m_n_for_any_d_l2299_229933


namespace NUMINAMATH_GPT_y_intercept_line_l2299_229977

theorem y_intercept_line : 
  ∃ m b : ℝ, 
  (2 * m + b = -3) ∧ 
  (6 * m + b = 5) ∧ 
  b = -7 :=
by 
  sorry

end NUMINAMATH_GPT_y_intercept_line_l2299_229977


namespace NUMINAMATH_GPT_watermelon_vendor_profit_l2299_229997

theorem watermelon_vendor_profit 
  (purchase_price : ℝ) (selling_price_initial : ℝ) (initial_quantity_sold : ℝ) 
  (decrease_factor : ℝ) (additional_quantity_per_decrease : ℝ) (fixed_cost : ℝ) 
  (desired_profit : ℝ) 
  (x : ℝ)
  (h_purchase : purchase_price = 2)
  (h_selling_initial : selling_price_initial = 3)
  (h_initial_quantity : initial_quantity_sold = 200)
  (h_decrease_factor : decrease_factor = 0.1)
  (h_additional_quantity : additional_quantity_per_decrease = 40)
  (h_fixed_cost : fixed_cost = 24)
  (h_desired_profit : desired_profit = 200) :
  (x = 2.8 ∨ x = 2.7) ↔ 
  ((x - purchase_price) * (initial_quantity_sold + additional_quantity_per_decrease / decrease_factor * (selling_price_initial - x)) - fixed_cost = desired_profit) :=
by sorry

end NUMINAMATH_GPT_watermelon_vendor_profit_l2299_229997


namespace NUMINAMATH_GPT_ellipse_equation_and_line_intersection_unique_l2299_229918

-- Definitions from conditions
def ellipse (x y : ℝ) : Prop := (x^2)/4 + (y^2)/3 = 1
def line (x0 y0 x y : ℝ) : Prop := 3*x0*x + 4*y0*y - 12 = 0
def on_ellipse (x0 y0 : ℝ) : Prop := ellipse x0 y0

theorem ellipse_equation_and_line_intersection_unique :
  ∀ (x0 y0 : ℝ), on_ellipse x0 y0 → ∀ (x y : ℝ), line x0 y0 x y → ellipse x y → x = x0 ∧ y = y0 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_and_line_intersection_unique_l2299_229918


namespace NUMINAMATH_GPT_integer_triangle_600_integer_triangle_144_l2299_229904

-- Problem Part I
theorem integer_triangle_600 :
  ∃ (a b c : ℕ), a * b * c = 600 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 26 :=
by {
  sorry
}

-- Problem Part II
theorem integer_triangle_144 :
  ∃ (a b c : ℕ), a * b * c = 144 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_integer_triangle_600_integer_triangle_144_l2299_229904


namespace NUMINAMATH_GPT_line_tangent_to_parabola_l2299_229988

theorem line_tangent_to_parabola (k : ℝ) (x₀ y₀ : ℝ) 
  (h₁ : y₀ = k * x₀ - 2) 
  (h₂ : x₀^2 = 4 * y₀) 
  (h₃ : ∀ x y, (x = x₀ ∧ y = y₀) → (k = (1/2) * x₀)) :
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_line_tangent_to_parabola_l2299_229988


namespace NUMINAMATH_GPT_solve_k_n_l2299_229981
-- Import the entire Mathlib

-- Define the theorem statement
theorem solve_k_n (k n : ℕ) (hk : k > 0) (hn : n > 0) : k^2 - 2016 = 3^n ↔ k = 45 ∧ n = 2 :=
  by sorry

end NUMINAMATH_GPT_solve_k_n_l2299_229981


namespace NUMINAMATH_GPT_probability_forming_more_from_remont_probability_forming_papa_from_papaha_l2299_229917

-- Definition for part (a)
theorem probability_forming_more_from_remont : 
  (6 * 5 * 4 * 3 = 360) ∧ (1 / 360 = 0.00278) :=
by
  sorry

-- Definition for part (b)
theorem probability_forming_papa_from_papaha : 
  (6 * 5 * 4 * 3 = 360) ∧ (12 / 360 = 0.03333) :=
by
  sorry

end NUMINAMATH_GPT_probability_forming_more_from_remont_probability_forming_papa_from_papaha_l2299_229917


namespace NUMINAMATH_GPT_triangle_inequality_x_values_l2299_229970

theorem triangle_inequality_x_values :
  {x : ℕ | 1 ≤ x ∧ x < 14} = {x | x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10 ∨ x = 11 ∨ x = 12 ∨ x = 13} :=
  by
    sorry

end NUMINAMATH_GPT_triangle_inequality_x_values_l2299_229970


namespace NUMINAMATH_GPT_proof_of_k_values_l2299_229987

noncomputable def problem_statement : Prop :=
  ∀ k : ℝ,
    (∃ a b : ℝ, (6 * a^2 + 5 * a + k = 0 ∧ 6 * b^2 + 5 * b + k = 0 ∧ a ≠ b ∧
    |a - b| = 3 * (a^2 + b^2))) ↔ (k = 1 ∨ k = -20.717)

theorem proof_of_k_values : problem_statement :=
by sorry

end NUMINAMATH_GPT_proof_of_k_values_l2299_229987


namespace NUMINAMATH_GPT_reflect_center_of_circle_l2299_229954

def reflect_point (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

theorem reflect_center_of_circle :
  reflect_point (3, -7) = (7, -3) :=
by
  sorry

end NUMINAMATH_GPT_reflect_center_of_circle_l2299_229954


namespace NUMINAMATH_GPT_log_expression_equals_l2299_229994

noncomputable def expression (x y : ℝ) : ℝ :=
  (Real.log x^2) / (Real.log y^10) *
  (Real.log y^3) / (Real.log x^7) *
  (Real.log x^4) / (Real.log y^8) *
  (Real.log y^6) / (Real.log x^9) *
  (Real.log x^11) / (Real.log y^5)

theorem log_expression_equals (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  expression x y = (1 / 15) * Real.log y / Real.log x :=
sorry

end NUMINAMATH_GPT_log_expression_equals_l2299_229994


namespace NUMINAMATH_GPT_most_stable_performance_l2299_229962

theorem most_stable_performance 
    (S_A S_B S_C S_D : ℝ)
    (h_A : S_A = 0.54) 
    (h_B : S_B = 0.61) 
    (h_C : S_C = 0.7) 
    (h_D : S_D = 0.63) :
    S_A <= S_B ∧ S_A <= S_C ∧ S_A <= S_D :=
by {
  sorry
}

end NUMINAMATH_GPT_most_stable_performance_l2299_229962


namespace NUMINAMATH_GPT_extra_mangoes_l2299_229983

-- Definitions of the conditions
def original_price_per_mango := 433.33 / 130
def new_price_per_mango := original_price_per_mango - 0.10 * original_price_per_mango
def mangoes_at_original_price := 360 / original_price_per_mango
def mangoes_at_new_price := 360 / new_price_per_mango

-- Statement to be proved
theorem extra_mangoes : mangoes_at_new_price - mangoes_at_original_price = 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_extra_mangoes_l2299_229983


namespace NUMINAMATH_GPT_seats_in_row_l2299_229908

theorem seats_in_row (y : ℕ → ℕ) (k b : ℕ) :
  (∀ x, y x = k * x + b) →
  y 1 = 20 →
  y 19 = 56 →
  y 26 = 70 :=
by
  intro h1 h2 h3
  -- Additional constraints to prove the given requirements
  sorry

end NUMINAMATH_GPT_seats_in_row_l2299_229908


namespace NUMINAMATH_GPT_complement_intersection_eq_l2299_229951

def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem complement_intersection_eq :
  (U \ A) ∩ B = {3} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_eq_l2299_229951


namespace NUMINAMATH_GPT_Amanda_lost_notebooks_l2299_229980

theorem Amanda_lost_notebooks (initial_notebooks ordered additional_notebooks remaining_notebooks : ℕ)
  (h1 : initial_notebooks = 10)
  (h2 : ordered = 6)
  (h3 : remaining_notebooks = 14) :
  initial_notebooks + ordered - remaining_notebooks = 2 := by
sorry

end NUMINAMATH_GPT_Amanda_lost_notebooks_l2299_229980


namespace NUMINAMATH_GPT_eggs_in_second_tree_l2299_229920

theorem eggs_in_second_tree
  (nests_in_first_tree : ℕ)
  (eggs_per_nest : ℕ)
  (eggs_in_front_yard : ℕ)
  (total_eggs : ℕ)
  (eggs_in_second_tree : ℕ)
  (h1 : nests_in_first_tree = 2)
  (h2 : eggs_per_nest = 5)
  (h3 : eggs_in_front_yard = 4)
  (h4 : total_eggs = 17)
  (h5 : nests_in_first_tree * eggs_per_nest + eggs_in_front_yard + eggs_in_second_tree = total_eggs) :
  eggs_in_second_tree = 3 :=
sorry

end NUMINAMATH_GPT_eggs_in_second_tree_l2299_229920


namespace NUMINAMATH_GPT_noah_garden_larger_by_75_l2299_229943

-- Define the dimensions of Liam's garden
def length_liam : ℕ := 30
def width_liam : ℕ := 50

-- Define the dimensions of Noah's garden
def length_noah : ℕ := 35
def width_noah : ℕ := 45

-- Define the areas of the gardens
def area_liam : ℕ := length_liam * width_liam
def area_noah : ℕ := length_noah * width_noah

theorem noah_garden_larger_by_75 :
  area_noah - area_liam = 75 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_noah_garden_larger_by_75_l2299_229943


namespace NUMINAMATH_GPT_ellen_dinner_calories_proof_l2299_229986

def ellen_daily_calories := 2200
def ellen_breakfast_calories := 353
def ellen_lunch_calories := 885
def ellen_snack_calories := 130
def ellen_remaining_calories : ℕ :=
  ellen_daily_calories - (ellen_breakfast_calories + ellen_lunch_calories + ellen_snack_calories)

theorem ellen_dinner_calories_proof : ellen_remaining_calories = 832 := by
  sorry

end NUMINAMATH_GPT_ellen_dinner_calories_proof_l2299_229986


namespace NUMINAMATH_GPT_knight_reachability_l2299_229952

theorem knight_reachability (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) :
  (p + q) % 2 = 1 ∧ Nat.gcd p q = 1 ↔
  ∀ x y x' y', ∃ k h n m, x' = x + k * p + h * q ∧ y' = y + n * p + m * q :=
by
  sorry

end NUMINAMATH_GPT_knight_reachability_l2299_229952


namespace NUMINAMATH_GPT_remainder_when_n_plus_5040_divided_by_7_l2299_229955

theorem remainder_when_n_plus_5040_divided_by_7 (n : ℤ) (h: n % 7 = 2) : (n + 5040) % 7 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_n_plus_5040_divided_by_7_l2299_229955


namespace NUMINAMATH_GPT_correct_exponentiation_calculation_l2299_229991

theorem correct_exponentiation_calculation (a : ℝ) : a^2 * a^6 = a^8 :=
by sorry

end NUMINAMATH_GPT_correct_exponentiation_calculation_l2299_229991


namespace NUMINAMATH_GPT_zach_needs_more_money_l2299_229932

noncomputable def cost_of_bike : ℕ := 100
noncomputable def weekly_allowance : ℕ := 5
noncomputable def mowing_income : ℕ := 10
noncomputable def babysitting_rate_per_hour : ℕ := 7
noncomputable def initial_savings : ℕ := 65
noncomputable def hours_babysitting : ℕ := 2

theorem zach_needs_more_money : 
  cost_of_bike - (initial_savings + weekly_allowance + mowing_income + (babysitting_rate_per_hour * hours_babysitting)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_zach_needs_more_money_l2299_229932


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2299_229967

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : q = 2) (h3 : a 0 + a 1 + a 2 = 21) : 
  a 2 + a 3 + a 4 = 84 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2299_229967


namespace NUMINAMATH_GPT_original_cost_of_article_l2299_229995

theorem original_cost_of_article : ∃ C : ℝ, 
  (∀ S : ℝ, S = 1.35 * C) ∧
  (∀ C_new : ℝ, C_new = 0.75 * C) ∧
  (∀ S_new : ℝ, (S_new = 1.35 * C - 25) ∧ (S_new = 1.0875 * C)) ∧
  (C = 95.24) :=
sorry

end NUMINAMATH_GPT_original_cost_of_article_l2299_229995


namespace NUMINAMATH_GPT_remainder_of_polynomial_division_l2299_229985

theorem remainder_of_polynomial_division :
  Polynomial.eval 2 (8 * X^3 - 22 * X^2 + 30 * X - 45) = -9 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_of_polynomial_division_l2299_229985


namespace NUMINAMATH_GPT_sum_of_distinct_elements_not_square_l2299_229936

open Set

noncomputable def setS : Set ℕ := { n | ∃ k : ℕ, n = 2^(2*k+1) }

theorem sum_of_distinct_elements_not_square (s : Finset ℕ) (hs: ∀ x ∈ s, x ∈ setS) :
  ¬∃ k : ℕ, s.sum id = k^2 :=
sorry

end NUMINAMATH_GPT_sum_of_distinct_elements_not_square_l2299_229936


namespace NUMINAMATH_GPT_incorrect_statement_d_l2299_229982

theorem incorrect_statement_d :
  (¬(abs 2 = -2)) :=
by sorry

end NUMINAMATH_GPT_incorrect_statement_d_l2299_229982


namespace NUMINAMATH_GPT_f_m_plus_1_positive_l2299_229939

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + x + a

theorem f_m_plus_1_positive {m a : ℝ} (h_a_pos : a > 0) (h_f_m_neg : f m a < 0) : f (m + 1) a > 0 := by
  sorry

end NUMINAMATH_GPT_f_m_plus_1_positive_l2299_229939


namespace NUMINAMATH_GPT_trigonometric_identity_l2299_229907

theorem trigonometric_identity 
  (α : ℝ)
  (h : Real.tan (α + Real.pi / 4) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2299_229907


namespace NUMINAMATH_GPT_range_of_x_when_a_is_1_range_of_a_for_necessity_l2299_229921

-- Define the statements p and q based on the conditions
def p (x a : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := (x - 3) / (x - 2) ≤ 0

-- (1) Prove the range of x when a = 1 and p ∧ q is true
theorem range_of_x_when_a_is_1 {x : ℝ} (h1 : ∀ x, p x 1) (h2 : q x) : 2 < x ∧ x < 3 :=
  sorry

-- (2) Prove the range of a for p to be necessary but not sufficient for q
theorem range_of_a_for_necessity : ∀ a, (∀ x, p x a → q x) → (1 ≤ a ∧ a ≤ 2) :=
  sorry

end NUMINAMATH_GPT_range_of_x_when_a_is_1_range_of_a_for_necessity_l2299_229921


namespace NUMINAMATH_GPT_work_duration_l2299_229968

theorem work_duration (X_full_days : ℕ) (Y_full_days : ℕ) (Y_worked_days : ℕ) (R : ℚ) :
  X_full_days = 18 ∧ Y_full_days = 15 ∧ Y_worked_days = 5 ∧ R = (2 / 3) →
  (R / (1 / X_full_days)) = 12 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_work_duration_l2299_229968


namespace NUMINAMATH_GPT_tan_triple_angle_l2299_229929

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 1/3) : Real.tan (3 * θ) = 13/9 :=
by
  sorry

end NUMINAMATH_GPT_tan_triple_angle_l2299_229929


namespace NUMINAMATH_GPT_problem_statement_l2299_229990

/-- Definition of the function f that relates the input n with floor functions -/
def f (n : ℕ) : ℤ :=
  n + ⌊(n : ℤ) / 6⌋ - ⌊(n : ℤ) / 2⌋ - ⌊2 * (n : ℤ) / 3⌋

/-- Prove the main statement -/
theorem problem_statement (n : ℕ) (hpos : 0 < n) :
  f n = 0 ↔ ∃ k : ℕ, n = 6 * k + 1 :=
sorry -- Proof goes here.

end NUMINAMATH_GPT_problem_statement_l2299_229990


namespace NUMINAMATH_GPT_number_of_cherries_l2299_229927

-- Definitions for the problem conditions
def total_fruits : ℕ := 580
def raspberries (b : ℕ) : ℕ := 2 * b
def grapes (c : ℕ) : ℕ := 3 * c
def cherries (r : ℕ) : ℕ := 3 * r

-- Theorem to prove the number of cherries
theorem number_of_cherries (b r g c : ℕ) 
  (H1 : b + r + g + c = total_fruits)
  (H2 : r = raspberries b)
  (H3 : g = grapes c)
  (H4 : c = cherries r) :
  c = 129 :=
by sorry

end NUMINAMATH_GPT_number_of_cherries_l2299_229927


namespace NUMINAMATH_GPT_rice_in_first_5_days_l2299_229969

-- Define the arithmetic sequence for number of workers dispatched each day
def num_workers (n : ℕ) : ℕ := 64 + (n - 1) * 7

-- Function to compute the sum of the first n terms of the arithmetic sequence
def sum_workers (n : ℕ) : ℕ := n * 64 + (n * (n - 1)) / 2 * 7

-- Given the rice distribution conditions
def rice_per_worker : ℕ := 3

-- Given the problem specific conditions
def total_rice_distributed_first_5_days : ℕ := 
  rice_per_worker * (sum_workers 1 + sum_workers 2 + sum_workers 3 + sum_workers 4 + sum_workers 5)
  
-- Proof goal
theorem rice_in_first_5_days : total_rice_distributed_first_5_days = 3300 :=
  by
  sorry

end NUMINAMATH_GPT_rice_in_first_5_days_l2299_229969


namespace NUMINAMATH_GPT_number_of_rectangles_l2299_229928

-- Definition of the problem: We have 12 equally spaced points on a circle.
def points_on_circle : ℕ := 12

-- The number of diameters is half the number of points, as each diameter involves two points.
def diameters (n : ℕ) : ℕ := n / 2

-- The number of ways to choose 2 diameters out of n/2 is given by the binomial coefficient.
noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Prove the number of rectangles that can be formed is 15.
theorem number_of_rectangles :
  binomial_coefficient (diameters points_on_circle) 2 = 15 := by
  sorry

end NUMINAMATH_GPT_number_of_rectangles_l2299_229928


namespace NUMINAMATH_GPT_hannahs_peppers_total_weight_l2299_229924

theorem hannahs_peppers_total_weight:
  let green := 0.3333333333333333
  let red := 0.3333333333333333
  let yellow := 0.25
  let orange := 0.5
  green + red + yellow + orange = 1.4166666666666665 :=
by
  repeat { sorry } -- Placeholder for the actual proof

end NUMINAMATH_GPT_hannahs_peppers_total_weight_l2299_229924


namespace NUMINAMATH_GPT_min_bottles_required_l2299_229902

theorem min_bottles_required (bottle_ounces : ℕ) (total_ounces : ℕ) (h : bottle_ounces = 15) (ht : total_ounces = 150) :
  ∃ (n : ℕ), n * bottle_ounces >= total_ounces ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_min_bottles_required_l2299_229902


namespace NUMINAMATH_GPT_cannot_tile_regular_pentagon_l2299_229965

theorem cannot_tile_regular_pentagon :
  ¬ (∃ n : ℕ, 360 % (180 - (360 / 5 : ℕ)) = 0) :=
by sorry

end NUMINAMATH_GPT_cannot_tile_regular_pentagon_l2299_229965


namespace NUMINAMATH_GPT_log_base_2_of_7_l2299_229914

variable (m n : ℝ)

theorem log_base_2_of_7 (h1 : Real.log 5 = m) (h2 : Real.log 7 = n) : Real.logb 2 7 = n / (1 - m) :=
by
  sorry

end NUMINAMATH_GPT_log_base_2_of_7_l2299_229914


namespace NUMINAMATH_GPT_actual_time_when_watch_reads_11_pm_is_correct_l2299_229957

-- Define the conditions
def noon := 0 -- Time when Cassandra sets her watch to the correct time
def actual_time_2_pm := 120 -- 2:00 PM in minutes
def watch_time_2_pm := 113.2 -- 1:53 PM and 12 seconds in minutes (113 minutes + 0.2 minutes)

-- Define the goal
def actual_time_watch_reads_11_pm := 731.25 -- 12:22 PM and 15 seconds in minutes from noon

-- Provide the theorem statement without proof
theorem actual_time_when_watch_reads_11_pm_is_correct :
  actual_time_watch_reads_11_pm = 731.25 :=
sorry

end NUMINAMATH_GPT_actual_time_when_watch_reads_11_pm_is_correct_l2299_229957


namespace NUMINAMATH_GPT_triangle_inequality_sqrt_sides_l2299_229964

theorem triangle_inequality_sqrt_sides {a b c : ℝ} (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b):
  (Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c) 
  ∧ (Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) = Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_sqrt_sides_l2299_229964


namespace NUMINAMATH_GPT_water_volume_in_B_when_A_is_0_point_4_l2299_229948

noncomputable def pool_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

noncomputable def valve_rate (volume time : ℝ) : ℝ :=
  volume / time

theorem water_volume_in_B_when_A_is_0_point_4 :
  ∀ (length width depth : ℝ)
    (time_A_fill time_A_to_B : ℝ)
    (depth_A_target : ℝ),
    length = 3 → width = 2 → depth = 1.2 →
    time_A_fill = 18 → time_A_to_B = 24 →
    depth_A_target = 0.4 →
    pool_volume length width depth = 7.2 →
    valve_rate 7.2 time_A_fill = 0.4 →
    valve_rate 7.2 time_A_to_B = 0.3 →
    ∃ (time_required : ℝ),
    time_required = 24 →
    (valve_rate 7.2 time_A_to_B * time_required = 7.2) :=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_water_volume_in_B_when_A_is_0_point_4_l2299_229948


namespace NUMINAMATH_GPT_problem_solution_l2299_229922

theorem problem_solution (a b : ℤ) (h1 : 6 * b + 4 * a = -50) (h2 : a * b = -84) : a + 2 * b = -17 := 
  sorry

end NUMINAMATH_GPT_problem_solution_l2299_229922


namespace NUMINAMATH_GPT_painted_cube_faces_l2299_229959

theorem painted_cube_faces (a : ℕ) (h : 2 < a) :
  ∃ (one_face two_faces three_faces : ℕ),
  (one_face = 6 * (a - 2) ^ 2) ∧
  (two_faces = 12 * (a - 2)) ∧
  (three_faces = 8) := by
  sorry

end NUMINAMATH_GPT_painted_cube_faces_l2299_229959


namespace NUMINAMATH_GPT_selection_count_l2299_229916

def choose (n k : ℕ) : ℕ := -- Binomial coefficient definition
  if h : 0 ≤ k ∧ k ≤ n then
    Nat.choose n k
  else
    0

theorem selection_count : choose 9 5 - choose 6 5 = 120 := by
  sorry

end NUMINAMATH_GPT_selection_count_l2299_229916


namespace NUMINAMATH_GPT_profit_relationship_max_profit_l2299_229906

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 2 then 5 * (x^2 + 3)
else if h : 2 < x ∧ x ≤ 5 then 50 * x / (1 + x)
else 0

noncomputable def f (x : ℝ) : ℝ :=
15 * W x - 10 * x - 20 * x

theorem profit_relationship:
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x = 75 * x^2 - 30 * x + 225) ∧
  (∀ x, 2 < x ∧ x ≤ 5 → f x = (750 * x)/(1 + x) - 30 * x) :=
by
  -- to be proven
  sorry

theorem max_profit:
  ∃ x, 0 ≤ x ∧ x ≤ 5 ∧ f x = 480 ∧ 10 * x = 40 :=
by
  -- to be proven
  sorry

end NUMINAMATH_GPT_profit_relationship_max_profit_l2299_229906


namespace NUMINAMATH_GPT_possible_values_of_x_and_factors_l2299_229913

theorem possible_values_of_x_and_factors (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (x : ℕ), x = p^5 ∧ (∀ (d : ℕ), d ∣ x → d = p^0 ∨ d = p^1 ∨ d = p^2 ∨ d = p^3 ∨ d = p^4 ∨ d = p^5) ∧ Nat.divisors x ≠ ∅ ∧ (Nat.divisors x).card = 6 := 
  by 
    sorry

end NUMINAMATH_GPT_possible_values_of_x_and_factors_l2299_229913


namespace NUMINAMATH_GPT_pyramid_surface_area_l2299_229978

theorem pyramid_surface_area (base_edge volume : ℝ)
  (h_base_edge : base_edge = 1)
  (h_volume : volume = 1) :
  let height := 3
  let slant_height := Real.sqrt (9.25)
  let base_area := base_edge * base_edge
  let lateral_area := 4 * (1 / 2 * base_edge * slant_height)
  let total_surface_area := base_area + lateral_area
  total_surface_area = 7.082 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_surface_area_l2299_229978


namespace NUMINAMATH_GPT_cristina_catches_nicky_l2299_229926

-- Definitions from the conditions
def cristina_speed : ℝ := 4 -- meters per second
def nicky_speed : ℝ := 3 -- meters per second
def nicky_head_start : ℝ := 36 -- meters

-- The proof to find the time 't'
theorem cristina_catches_nicky (t : ℝ) : cristina_speed * t = nicky_head_start + nicky_speed * t -> t = 36 := by
  intros h
  sorry

end NUMINAMATH_GPT_cristina_catches_nicky_l2299_229926


namespace NUMINAMATH_GPT_even_sine_function_phi_eq_pi_div_2_l2299_229999
open Real

theorem even_sine_function_phi_eq_pi_div_2 (φ : ℝ) (h : 0 ≤ φ ∧ φ ≤ π)
    (even_f : ∀ x : ℝ, sin (x + φ) = sin (-x + φ)) : φ = π / 2 :=
sorry

end NUMINAMATH_GPT_even_sine_function_phi_eq_pi_div_2_l2299_229999


namespace NUMINAMATH_GPT_barbi_monthly_loss_l2299_229972

variable (x : Real)

theorem barbi_monthly_loss : 
  (∃ x : Real, 12 * x = 99 - 81) → x = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_barbi_monthly_loss_l2299_229972


namespace NUMINAMATH_GPT_simplify_fraction_l2299_229909

variable (c : ℝ)

theorem simplify_fraction :
  (6 + 2 * c) / 7 + 3 = (27 + 2 * c) / 7 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2299_229909


namespace NUMINAMATH_GPT_integral_cosine_l2299_229971

noncomputable def a : ℝ := 2 * Real.pi / 3

theorem integral_cosine (ha : a = 2 * Real.pi / 3) :
  ∫ x in -a..a, Real.cos x = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_integral_cosine_l2299_229971


namespace NUMINAMATH_GPT_least_positive_integer_not_representable_as_fraction_l2299_229984

theorem least_positive_integer_not_representable_as_fraction : 
  ¬ ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ (2^a - 2^b) / (2^c - 2^d) = 11 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_not_representable_as_fraction_l2299_229984


namespace NUMINAMATH_GPT_second_polygon_sides_l2299_229963

theorem second_polygon_sides (a b n m : ℕ) (s : ℝ) 
  (h1 : a = 45) 
  (h2 : b = 3 * s)
  (h3 : n * b = m * s)
  (h4 : n = 45) : m = 135 := 
by
  sorry

end NUMINAMATH_GPT_second_polygon_sides_l2299_229963


namespace NUMINAMATH_GPT_population_initial_count_l2299_229925

theorem population_initial_count
  (P : ℕ)
  (birth_rate : ℕ := 52)
  (death_rate : ℕ := 16)
  (net_growth_rate : ℝ := 1.2) :
  36 = (net_growth_rate / 100) * P ↔ P = 3000 :=
by sorry

end NUMINAMATH_GPT_population_initial_count_l2299_229925


namespace NUMINAMATH_GPT_Janet_horses_l2299_229998

theorem Janet_horses (acres : ℕ) (gallons_per_acre : ℕ) (spread_acres_per_day : ℕ) (total_days : ℕ)
  (gallons_per_day_per_horse : ℕ) (total_gallons_needed : ℕ) (total_gallons_spread : ℕ) (horses : ℕ) :
  acres = 20 ->
  gallons_per_acre = 400 ->
  spread_acres_per_day = 4 ->
  total_days = 25 ->
  gallons_per_day_per_horse = 5 ->
  total_gallons_needed = acres * gallons_per_acre ->
  total_gallons_spread = spread_acres_per_day * gallons_per_acre * total_days ->
  horses = total_gallons_needed / (gallons_per_day_per_horse * total_days) ->
  horses = 64 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_Janet_horses_l2299_229998


namespace NUMINAMATH_GPT_lassis_from_mangoes_l2299_229905

theorem lassis_from_mangoes (mangoes lassis mangoes' lassis' : ℕ) 
  (h1 : lassis = (8 * mangoes) / 3)
  (h2 : mangoes = 15) :
  lassis = 40 :=
by
  sorry

end NUMINAMATH_GPT_lassis_from_mangoes_l2299_229905


namespace NUMINAMATH_GPT_cone_height_ratio_l2299_229931

theorem cone_height_ratio (C : ℝ) (h₁ : ℝ) (V₂ : ℝ) (r : ℝ) (h₂ : ℝ) :
  C = 20 * Real.pi → 
  h₁ = 40 →
  V₂ = 400 * Real.pi →
  2 * Real.pi * r = 20 * Real.pi →
  V₂ = (1 / 3) * Real.pi * r^2 * h₂ →
  h₂ / h₁ = (3 / 10) := by
sorry

end NUMINAMATH_GPT_cone_height_ratio_l2299_229931


namespace NUMINAMATH_GPT_incorrect_statement_C_l2299_229900

theorem incorrect_statement_C 
  (x y : ℝ)
  (n : ℕ)
  (data : Fin n → (ℝ × ℝ))
  (h : ∀ (i : Fin n), (x, y) = data i)
  (reg_eq : ∀ (x : ℝ), 0.85 * x - 85.71 = y) :
  ¬ (forall (x : ℝ), x = 160 → ∀ (y : ℝ), y = 50.29) := 
sorry

end NUMINAMATH_GPT_incorrect_statement_C_l2299_229900


namespace NUMINAMATH_GPT_eq_correct_l2299_229903

variable (x : ℝ)

def width (x : ℝ) : ℝ := x - 6

def area_eq (x : ℝ) : Prop := x * width x = 720

theorem eq_correct (h : area_eq x) : x * (x - 6) = 720 :=
by exact h

end NUMINAMATH_GPT_eq_correct_l2299_229903
