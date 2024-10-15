import Mathlib

namespace NUMINAMATH_GPT_polygon_area_leq_17_point_5_l1246_124638

theorem polygon_area_leq_17_point_5 (proj_OX proj_bisector_13 proj_OY proj_bisector_24 : ℝ)
  (h1: proj_OX = 4)
  (h2: proj_bisector_13 = 3 * Real.sqrt 2)
  (h3: proj_OY = 5)
  (h4: proj_bisector_24 = 4 * Real.sqrt 2)
  (S : ℝ) :
  S ≤ 17.5 := sorry

end NUMINAMATH_GPT_polygon_area_leq_17_point_5_l1246_124638


namespace NUMINAMATH_GPT_M_eq_N_l1246_124693

def M (u : ℤ) : Prop := ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l
def N (u : ℤ) : Prop := ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r

theorem M_eq_N : ∀ u : ℤ, M u ↔ N u := by
  sorry

end NUMINAMATH_GPT_M_eq_N_l1246_124693


namespace NUMINAMATH_GPT_number_of_terms_in_sequence_l1246_124673

def arithmetic_sequence_terms (a d l : ℕ) : ℕ :=
  (l - a) / d + 1

theorem number_of_terms_in_sequence : arithmetic_sequence_terms 1 4 57 = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_terms_in_sequence_l1246_124673


namespace NUMINAMATH_GPT_extreme_value_f_at_a_eq_1_monotonic_intervals_f_exists_no_common_points_l1246_124607

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a * Real.log (x - 1)

-- Problem 1: Prove that the extreme value of f(x) when a = 1 is \frac{3}{4} + \ln 2
theorem extreme_value_f_at_a_eq_1 : 
  f (3/2) 1 = 3/4 + Real.log 2 :=
sorry

-- Problem 2: Prove the monotonic intervals of f(x) based on the value of a
theorem monotonic_intervals_f :
  ∀ a : ℝ, 
    (if a ≤ 0 then 
      ∀ x, 1 < x → f x' a > 0
     else
      ∀ x, 1 < x ∧ x ≤ (a + 2) / 2 → f x a ≤ 0 ∧ ∀ x, x ≥ (a + 2) / 2 → f x a > 0) :=
sorry

-- Problem 3: Prove that for a ≥ 1, there exists an a such that f(x) has no common points with y = \frac{5}{8} + \ln 2
theorem exists_no_common_points (h : 1 ≤ a) :
  ∃ x : ℝ, f x a ≠ 5/8 + Real.log 2 :=
sorry

end NUMINAMATH_GPT_extreme_value_f_at_a_eq_1_monotonic_intervals_f_exists_no_common_points_l1246_124607


namespace NUMINAMATH_GPT_equation_of_curve_t_circle_through_fixed_point_l1246_124678

noncomputable def problem (x y : ℝ) : Prop :=
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (0, -1)
  let O : ℝ × ℝ := (0, 0)
  let M : ℝ × ℝ := (x, y)
  let N : ℝ × ℝ := (0, y)
  (x + 1) * (x - 1) + y * y = y * (y + 1)

noncomputable def curve_t_equation (x : ℝ) : ℝ :=
  x^2 - 1

theorem equation_of_curve_t (x y : ℝ) 
  (h : problem x y) :
  y = curve_t_equation x := 
sorry

noncomputable def passing_through_fixed_point (x y : ℝ) : Prop :=
  let y := x^2 - 1
  let y' := 2 * x
  let P : ℝ × ℝ := (x, y)
  let Q_x := (4 * x^2 - 1) / (8 * x)
  let Q : ℝ × ℝ := (Q_x, -5 / 4)
  let H : ℝ × ℝ := (0, -3 / 4)
  (x * Q_x + (-3 / 4 - y) * ( -3 / 4 + 5 / 4)) = 0

theorem circle_through_fixed_point (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y = curve_t_equation x)
  (h : passing_through_fixed_point x y) :
  ∃ t : ℝ, passing_through_fixed_point x t ∧ t = -3 / 4 :=
sorry

end NUMINAMATH_GPT_equation_of_curve_t_circle_through_fixed_point_l1246_124678


namespace NUMINAMATH_GPT_chives_planted_l1246_124652

theorem chives_planted (total_rows : ℕ) (plants_per_row : ℕ)
  (parsley_rows : ℕ) (rosemary_rows : ℕ) :
  total_rows = 20 →
  plants_per_row = 10 →
  parsley_rows = 3 →
  rosemary_rows = 2 →
  (plants_per_row * (total_rows - (parsley_rows + rosemary_rows))) = 150 :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_chives_planted_l1246_124652


namespace NUMINAMATH_GPT_midpoint_of_line_segment_l1246_124630

theorem midpoint_of_line_segment :
  let z1 := Complex.mk (-7) 5
  let z2 := Complex.mk 5 (-3)
  (z1 + z2) / 2 = Complex.mk (-1) 1 := by sorry

end NUMINAMATH_GPT_midpoint_of_line_segment_l1246_124630


namespace NUMINAMATH_GPT_find_B_value_l1246_124651

-- Define the polynomial and conditions
def polynomial (A B : ℤ) (z : ℤ) : ℤ := z^4 - 12 * z^3 + A * z^2 + B * z + 36

-- Define roots and their properties according to the conditions
def roots_sum_to_twelve (r1 r2 r3 r4 : ℕ) : Prop := r1 + r2 + r3 + r4 = 12

-- The final statement to prove
theorem find_B_value (r1 r2 r3 r4 : ℕ) (A B : ℤ) (h_sum : roots_sum_to_twelve r1 r2 r3 r4)
    (h_pos : r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0) 
    (h_poly : polynomial A B = (z^4 - 12*z^3 + Az^2 + Bz + 36)) :
    B = -96 :=
    sorry

end NUMINAMATH_GPT_find_B_value_l1246_124651


namespace NUMINAMATH_GPT_prob_A_wins_match_is_correct_l1246_124615

/-- Definitions -/

def prob_A_wins_game : ℝ := 0.6

def prob_B_wins_game : ℝ := 1 - prob_A_wins_game

def prob_A_wins_match (p: ℝ) : ℝ :=
  p * p * (1 - p) + p * (1 - p) * p + p * p

/-- Theorem -/

theorem prob_A_wins_match_is_correct : 
  prob_A_wins_match prob_A_wins_game = 0.648 :=
by
  sorry

end NUMINAMATH_GPT_prob_A_wins_match_is_correct_l1246_124615


namespace NUMINAMATH_GPT_cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7_l1246_124657

theorem cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7 (p q r : ℝ)
  (h_roots : 3*p^3 - 4*p^2 + 220*p - 7 = 0 ∧ 3*q^3 - 4*q^2 + 220*q - 7 = 0 ∧ 3*r^3 - 4*r^2 + 220*r - 7 = 0)
  (h_vieta : p + q + r = 4 / 3) :
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 64.556 :=
sorry

end NUMINAMATH_GPT_cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7_l1246_124657


namespace NUMINAMATH_GPT_variance_of_X_is_correct_l1246_124600

/-!
  There is a batch of products, among which there are 12 genuine items and 4 defective items.
  If 3 items are drawn with replacement, and X represents the number of defective items drawn,
  prove that the variance of X is 9 / 16 given that X follows a binomial distribution B(3, 1 / 4).
-/

noncomputable def variance_of_binomial : Prop :=
  let n := 3
  let p := 1 / 4
  let variance := n * p * (1 - p)
  variance = 9 / 16

theorem variance_of_X_is_correct : variance_of_binomial := by
  sorry

end NUMINAMATH_GPT_variance_of_X_is_correct_l1246_124600


namespace NUMINAMATH_GPT_output_increase_percentage_l1246_124661

theorem output_increase_percentage (O : ℝ) (P : ℝ) (h : (O * (1 + P / 100) * 1.60) * 0.5682 = O) : P = 10.09 :=
by 
  sorry

end NUMINAMATH_GPT_output_increase_percentage_l1246_124661


namespace NUMINAMATH_GPT_AM_minus_GM_lower_bound_l1246_124699

theorem AM_minus_GM_lower_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) : 
  (x + y) / 2 - Real.sqrt (x * y) ≥ (x - y)^2 / (8 * x) := 
by {
  sorry -- Proof to be filled in
}

end NUMINAMATH_GPT_AM_minus_GM_lower_bound_l1246_124699


namespace NUMINAMATH_GPT_red_balls_count_l1246_124647

-- Lean 4 statement for proving the number of red balls in the bag is 336
theorem red_balls_count (x : ℕ) (total_balls red_balls : ℕ) 
  (h1 : total_balls = 60 + 18 * x) 
  (h2 : red_balls = 56 + 14 * x) 
  (h3 : (56 + 14 * x : ℚ) / (60 + 18 * x) = 4 / 5) : red_balls = 336 := 
by
  sorry

end NUMINAMATH_GPT_red_balls_count_l1246_124647


namespace NUMINAMATH_GPT_triangle_angle_measure_l1246_124655

theorem triangle_angle_measure
  (D E F : ℝ)
  (hD : D = 70)
  (hE : E = 2 * F + 18)
  (h_sum : D + E + F = 180) :
  F = 92 / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_measure_l1246_124655


namespace NUMINAMATH_GPT_matrix_product_is_zero_l1246_124648

def vec3 := (ℝ × ℝ × ℝ)

def M1 (a b c : ℝ) : vec3 × vec3 × vec3 :=
  ((0, 2 * c, -2 * b),
   (-2 * c, 0, 2 * a),
   (2 * b, -2 * a, 0))

def M2 (a b c : ℝ) : vec3 × vec3 × vec3 :=
  ((2 * a^2, a^2 + b^2, a^2 + c^2),
   (a^2 + b^2, 2 * b^2, b^2 + c^2),
   (a^2 + c^2, b^2 + c^2, 2 * c^2))

def matrix_mul (m1 m2 : vec3 × vec3 × vec3) : vec3 × vec3 × vec3 := sorry

theorem matrix_product_is_zero (a b c : ℝ) :
  matrix_mul (M1 a b c) (M2 a b c) = ((0, 0, 0), (0, 0, 0), (0, 0, 0)) := by
  sorry

end NUMINAMATH_GPT_matrix_product_is_zero_l1246_124648


namespace NUMINAMATH_GPT_largest_rectangle_area_l1246_124676

theorem largest_rectangle_area (x y : ℝ) (h : 2*x + 2*y = 60) : x * y ≤ 225 :=
sorry

end NUMINAMATH_GPT_largest_rectangle_area_l1246_124676


namespace NUMINAMATH_GPT_cyclist_speed_l1246_124637

theorem cyclist_speed (v : ℝ) (h : 0.7142857142857143 * (30 + v) = 50) : v = 40 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_speed_l1246_124637


namespace NUMINAMATH_GPT_total_birds_count_l1246_124629

def blackbirds_per_tree : ℕ := 3
def tree_count : ℕ := 7
def magpies : ℕ := 13

theorem total_birds_count : (blackbirds_per_tree * tree_count) + magpies = 34 := by
  sorry

end NUMINAMATH_GPT_total_birds_count_l1246_124629


namespace NUMINAMATH_GPT_greg_pages_per_day_l1246_124623

variable (greg_pages : ℕ)
variable (brad_pages : ℕ)

theorem greg_pages_per_day :
  brad_pages = 26 → brad_pages = greg_pages + 8 → greg_pages = 18 :=
by
  intros h1 h2
  rw [h1, add_comm] at h2
  linarith

end NUMINAMATH_GPT_greg_pages_per_day_l1246_124623


namespace NUMINAMATH_GPT_sin_theta_correct_l1246_124679

noncomputable def sin_theta : ℝ :=
  let d := (4, 5, 7)
  let n := (3, -4, 5)
  let d_dot_n := 4 * 3 + 5 * (-4) + 7 * 5
  let norm_d := Real.sqrt (4^2 + 5^2 + 7^2)
  let norm_n := Real.sqrt (3^2 + (-4)^2 + 5^2)
  let cos_theta := d_dot_n / (norm_d * norm_n)
  cos_theta

theorem sin_theta_correct :
  sin_theta = 27 / Real.sqrt 4500 :=
by
  sorry

end NUMINAMATH_GPT_sin_theta_correct_l1246_124679


namespace NUMINAMATH_GPT_apple_price_33_kgs_l1246_124640

theorem apple_price_33_kgs (l q : ℕ) (h1 : 30 * l + 6 * q = 366) (h2 : 15 * l = 150) : 
  30 * l + 3 * q = 333 :=
by
  sorry

end NUMINAMATH_GPT_apple_price_33_kgs_l1246_124640


namespace NUMINAMATH_GPT_blithe_toy_count_l1246_124625

-- Define the initial number of toys, the number lost, and the number found.
def initial_toys := 40
def toys_lost := 6
def toys_found := 9

-- Define the total number of toys after the changes.
def total_toys_after_changes := initial_toys - toys_lost + toys_found

-- The proof statement.
theorem blithe_toy_count : total_toys_after_changes = 43 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_blithe_toy_count_l1246_124625


namespace NUMINAMATH_GPT_minimum_value_y_l1246_124633

theorem minimum_value_y (x : ℝ) (h : x ≥ 1) : 5*x^2 - 8*x + 20 ≥ 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_y_l1246_124633


namespace NUMINAMATH_GPT_at_least_one_less_than_zero_l1246_124684

theorem at_least_one_less_than_zero {a b : ℝ} (h: a + b < 0) : a < 0 ∨ b < 0 := 
by 
  sorry

end NUMINAMATH_GPT_at_least_one_less_than_zero_l1246_124684


namespace NUMINAMATH_GPT_cycling_race_difference_l1246_124639

-- Define the speeds and time
def s_Chloe : ℝ := 18
def s_David : ℝ := 15
def t : ℝ := 5

-- Define the distances based on the speeds and time
def d_Chloe : ℝ := s_Chloe * t
def d_David : ℝ := s_David * t
def distance_difference : ℝ := d_Chloe - d_David

-- The theorem to prove
theorem cycling_race_difference :
  distance_difference = 15 := by
  sorry

end NUMINAMATH_GPT_cycling_race_difference_l1246_124639


namespace NUMINAMATH_GPT_problem_l1246_124680

noncomputable def roots1 : Set ℝ := { α | α^2 - 2*α + 1 = 0 }
noncomputable def roots2 : Set ℝ := { γ | γ^2 - 3*γ + 1 = 0 }

theorem problem 
  (α β γ δ : ℝ) 
  (hαβ : α ∈ roots1 ∧ β ∈ roots1)
  (hγδ : γ ∈ roots2 ∧ δ ∈ roots2) : 
  (α - γ)^2 * (β - δ)^2 = 1 := 
sorry

end NUMINAMATH_GPT_problem_l1246_124680


namespace NUMINAMATH_GPT_cover_points_with_two_disks_l1246_124611

theorem cover_points_with_two_disks :
  ∀ (points : Fin 2014 → ℝ × ℝ),
    (∀ (i j k : Fin 2014), i ≠ j → j ≠ k → i ≠ k → 
      dist (points i) (points j) ≤ 1 ∨ dist (points j) (points k) ≤ 1 ∨ dist (points i) (points k) ≤ 1) →
    ∃ (A B : ℝ × ℝ), ∀ (p : Fin 2014),
      dist (points p) A ≤ 1 ∨ dist (points p) B ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_cover_points_with_two_disks_l1246_124611


namespace NUMINAMATH_GPT_increasing_arithmetic_sequence_l1246_124631

theorem increasing_arithmetic_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) = a n + 2) : ∀ n : ℕ, a (n + 1) > a n :=
by
  sorry

end NUMINAMATH_GPT_increasing_arithmetic_sequence_l1246_124631


namespace NUMINAMATH_GPT_find_speeds_of_A_and_B_l1246_124632

noncomputable def speed_A_and_B (x y : ℕ) : Prop :=
  30 * x - 30 * y = 300 ∧ 2 * x + 2 * y = 300

theorem find_speeds_of_A_and_B : ∃ (x y : ℕ), speed_A_and_B x y ∧ x = 80 ∧ y = 70 :=
by
  sorry

end NUMINAMATH_GPT_find_speeds_of_A_and_B_l1246_124632


namespace NUMINAMATH_GPT_find_multiple_l1246_124677

theorem find_multiple (x m : ℝ) (hx : x = 3) (h : x + 17 = m * (1 / x)) : m = 60 := 
by
  sorry

end NUMINAMATH_GPT_find_multiple_l1246_124677


namespace NUMINAMATH_GPT_simplify_expression_l1246_124685

variable (y : ℝ)

theorem simplify_expression : 
  3 * y - 5 * y^2 + 2 + (8 - 5 * y + 2 * y^2) = -3 * y^2 - 2 * y + 10 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1246_124685


namespace NUMINAMATH_GPT_min_x2_y2_z2_given_condition_l1246_124606

theorem min_x2_y2_z2_given_condition (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  ∃ (c : ℝ), c = 3 ∧ (∀ x y z : ℝ, x^3 + y^3 + z^3 - 3 * x * y * z = 8 → x^2 + y^2 + z^2 ≥ c) := 
sorry

end NUMINAMATH_GPT_min_x2_y2_z2_given_condition_l1246_124606


namespace NUMINAMATH_GPT_equivalence_statements_l1246_124662

variables (P Q : Prop)

theorem equivalence_statements :
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by sorry

end NUMINAMATH_GPT_equivalence_statements_l1246_124662


namespace NUMINAMATH_GPT_Q_no_negative_roots_and_at_least_one_positive_root_l1246_124622

def Q (x : ℝ) : ℝ := x^7 - 2 * x^6 - 6 * x^4 - 4 * x + 16

theorem Q_no_negative_roots_and_at_least_one_positive_root :
  (∀ x, x < 0 → Q x > 0) ∧ (∃ x, x > 0 ∧ Q x = 0) := 
sorry

end NUMINAMATH_GPT_Q_no_negative_roots_and_at_least_one_positive_root_l1246_124622


namespace NUMINAMATH_GPT_range_of_a_l1246_124644

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ (-2 < a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1246_124644


namespace NUMINAMATH_GPT_positive_solution_range_l1246_124603

theorem positive_solution_range (a : ℝ) (h : a > 0) (x : ℝ) : (∃ x, (a / (x + 3) = 1 / 2) ∧ x > 0) ↔ a > 3 / 2 := by
  sorry

end NUMINAMATH_GPT_positive_solution_range_l1246_124603


namespace NUMINAMATH_GPT_general_term_formula_l1246_124635

/-- Define that the point (n, S_n) lies on the function y = 2x^2 + x, hence S_n = 2 * n^2 + n --/
def S_n (n : ℕ) : ℕ := 2 * n^2 + n

/-- Define the nth term of the sequence a_n --/
def a_n (n : ℕ) : ℕ := if n = 0 then 0 else 4 * n - 1

theorem general_term_formula (n : ℕ) (hn : 0 < n) :
  a_n n = S_n n - S_n (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_general_term_formula_l1246_124635


namespace NUMINAMATH_GPT_range_of_a_for_inequality_l1246_124675

theorem range_of_a_for_inequality (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_for_inequality_l1246_124675


namespace NUMINAMATH_GPT_smallest_integer_y_l1246_124665

theorem smallest_integer_y (y : ℤ) (h : 3 - 5 * y < 23) : -3 ≥ y :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_integer_y_l1246_124665


namespace NUMINAMATH_GPT_largest_N_with_square_in_base_nine_l1246_124694

theorem largest_N_with_square_in_base_nine:
  ∃ N: ℕ, (9^2 ≤ N^2 ∧ N^2 < 9^3) ∧ ∀ M: ℕ, (9^2 ≤ M^2 ∧ M^2 < 9^3) → M ≤ N ∧ N = 26 := 
sorry

end NUMINAMATH_GPT_largest_N_with_square_in_base_nine_l1246_124694


namespace NUMINAMATH_GPT_sum_five_smallest_primes_l1246_124650

theorem sum_five_smallest_primes : (2 + 3 + 5 + 7 + 11) = 28 := by
  -- We state the sum of the known five smallest prime numbers.
  sorry

end NUMINAMATH_GPT_sum_five_smallest_primes_l1246_124650


namespace NUMINAMATH_GPT_income_final_amount_l1246_124612

noncomputable def final_amount (income : ℕ) : ℕ :=
  let children_distribution := (income * 45) / 100
  let wife_deposit := (income * 30) / 100
  let remaining_after_distribution := income - children_distribution - wife_deposit
  let donation := (remaining_after_distribution * 5) / 100
  remaining_after_distribution - donation

theorem income_final_amount : final_amount 200000 = 47500 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_income_final_amount_l1246_124612


namespace NUMINAMATH_GPT_geometric_series_solution_l1246_124697

-- Let a, r : ℝ be real numbers representing the parameters from the problem's conditions.
variables (a r : ℝ)

-- Define the conditions as hypotheses.
def condition1 : Prop := a / (1 - r) = 20
def condition2 : Prop := a / (1 - r^2) = 8

-- The theorem states that under these conditions, r equals 3/2.
theorem geometric_series_solution (hc1 : condition1 a r) (hc2 : condition2 a r) : r = 3 / 2 :=
sorry

end NUMINAMATH_GPT_geometric_series_solution_l1246_124697


namespace NUMINAMATH_GPT_find_missing_number_l1246_124643

theorem find_missing_number
  (x : ℝ)
  (h1 : (12 + x + y + 78 + 104) / 5 = 62)
  (h2 : (128 + 255 + 511 + 1023 + x) / 5 = 398.2) : 
  y = 42 :=
  sorry

end NUMINAMATH_GPT_find_missing_number_l1246_124643


namespace NUMINAMATH_GPT_percentage_to_decimal_l1246_124682

theorem percentage_to_decimal : (5 / 100 : ℚ) = 0.05 := by
  sorry

end NUMINAMATH_GPT_percentage_to_decimal_l1246_124682


namespace NUMINAMATH_GPT_expression_evaluation_l1246_124668

theorem expression_evaluation (x : ℝ) (h : 2 * x - 7 = 8 * x - 1) : 5 * (x - 3) = -20 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1246_124668


namespace NUMINAMATH_GPT_smallest_base_l1246_124663

-- Definitions of the conditions
def condition1 (b : ℕ) : Prop := b > 3
def condition2 (b : ℕ) : Prop := b > 7
def condition3 (b : ℕ) : Prop := b > 6
def condition4 (b : ℕ) : Prop := b > 8

-- Main theorem statement
theorem smallest_base : ∀ b : ℕ, condition1 b ∧ condition2 b ∧ condition3 b ∧ condition4 b → b = 9 := by
  sorry

end NUMINAMATH_GPT_smallest_base_l1246_124663


namespace NUMINAMATH_GPT_custom_op_1_neg3_l1246_124671

-- Define the custom operation as per the condition
def custom_op (a b : ℤ) : ℤ := a^2 + 2 * a * b - b^2

-- The theorem to prove that 1 * (-3) = -14 using the defined operation
theorem custom_op_1_neg3 : custom_op 1 (-3) = -14 := sorry

end NUMINAMATH_GPT_custom_op_1_neg3_l1246_124671


namespace NUMINAMATH_GPT_bottle_caps_proof_l1246_124695

def bottle_caps_difference (found thrown : ℕ) := found - thrown

theorem bottle_caps_proof : bottle_caps_difference 50 6 = 44 := by
  sorry

end NUMINAMATH_GPT_bottle_caps_proof_l1246_124695


namespace NUMINAMATH_GPT_quadratic_real_roots_l1246_124609

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k + 1) * x^2 + 4 * x - 1 = 0) ↔ k ≥ -5 ∧ k ≠ -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l1246_124609


namespace NUMINAMATH_GPT_digit_B_l1246_124692

def is_valid_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 7

def unique_digits (A B C D E F G : ℕ) : Prop :=
  is_valid_digit A ∧ is_valid_digit B ∧ is_valid_digit C ∧ is_valid_digit D ∧ 
  is_valid_digit E ∧ is_valid_digit F ∧ is_valid_digit G ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ 
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ 
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ 
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ 
  E ≠ F ∧ E ≠ G ∧ 
  F ≠ G

def total_sum (A B C D E F G : ℕ) : ℕ :=
  (A + B + C) + (A + E + F) + (C + D + E) + (B + D + G) + (B + F) + (G + E)

theorem digit_B (A B C D E F G : ℕ) 
  (h1 : unique_digits A B C D E F G)
  (h2 : total_sum A B C D E F G = 65) : B = 7 := 
sorry

end NUMINAMATH_GPT_digit_B_l1246_124692


namespace NUMINAMATH_GPT_example_problem_l1246_124654

-- Define the numbers of students in each grade
def freshmen : ℕ := 240
def sophomores : ℕ := 260
def juniors : ℕ := 300

-- Define the total number of spots for the trip
def total_spots : ℕ := 40

-- Define the total number of students
def total_students : ℕ := freshmen + sophomores + juniors

-- Define the fraction of sophomores relative to the total number of students
def fraction_sophomores : ℚ := sophomores / total_students

-- Define the number of spots allocated to sophomores
def spots_sophomores : ℚ := fraction_sophomores * total_spots

-- The theorem we need to prove
theorem example_problem : spots_sophomores = 13 :=
by 
  sorry

end NUMINAMATH_GPT_example_problem_l1246_124654


namespace NUMINAMATH_GPT_rectangle_longer_side_length_l1246_124634

theorem rectangle_longer_side_length (r : ℝ) (h1 : r = 4) 
  (h2 : ∃ w l, w * l = 2 * (π * r^2) ∧ w = 2 * r) : 
  ∃ l, l = 4 * π :=
by 
  obtain ⟨w, l, h_area, h_shorter_side⟩ := h2
  sorry

end NUMINAMATH_GPT_rectangle_longer_side_length_l1246_124634


namespace NUMINAMATH_GPT_find_x_plus_y_l1246_124602

-- Define the points A, B, and C with given conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 1, y := 1}
def C : Point := {x := 2, y := 4}

-- Define what it means for C to divide AB in the ratio 2:1
open Point

def divides_in_ratio (A B C : Point) (r₁ r₂ : ℝ) :=
  (C.x = (r₁ * A.x + r₂ * B.x) / (r₁ + r₂))
  ∧ (C.y = (r₁ * A.y + r₂ * B.y) / (r₁ + r₂))

-- Prove that x + y = 8 given the conditions
theorem find_x_plus_y {x y : ℝ} (B : Point) (H_B : B = {x := x, y := y}) :
  divides_in_ratio A B C 2 1 →
  x + y = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l1246_124602


namespace NUMINAMATH_GPT_potato_yield_l1246_124618

/-- Mr. Green's gardening problem -/
theorem potato_yield
  (steps_length : ℝ)
  (steps_width : ℝ)
  (step_size : ℝ)
  (yield_rate : ℝ)
  (feet_length := steps_length * step_size)
  (feet_width := steps_width * step_size)
  (area := feet_length * feet_width)
  (yield := area * yield_rate) :
  steps_length = 18 →
  steps_width = 25 →
  step_size = 2.5 →
  yield_rate = 0.75 →
  yield = 2109.375 :=
by
  sorry

end NUMINAMATH_GPT_potato_yield_l1246_124618


namespace NUMINAMATH_GPT_carrots_total_l1246_124688

-- Define the initial number of carrots Maria picked
def initial_carrots : ℕ := 685

-- Define the number of carrots Maria threw out
def thrown_out : ℕ := 156

-- Define the number of carrots Maria picked the next day
def picked_next_day : ℕ := 278

-- Define the total number of carrots Maria has after these actions
def total_carrots : ℕ :=
  initial_carrots - thrown_out + picked_next_day

-- The proof statement
theorem carrots_total : total_carrots = 807 := by
  sorry

end NUMINAMATH_GPT_carrots_total_l1246_124688


namespace NUMINAMATH_GPT_train_cross_tunnel_time_l1246_124626

noncomputable def train_length : ℝ := 800 -- in meters
noncomputable def train_speed : ℝ := 78 * 1000 / 3600 -- converted to meters per second
noncomputable def tunnel_length : ℝ := 500 -- in meters
noncomputable def total_distance : ℝ := train_length + tunnel_length -- total distance to travel

theorem train_cross_tunnel_time : total_distance / train_speed / 60 = 1 := by
  sorry

end NUMINAMATH_GPT_train_cross_tunnel_time_l1246_124626


namespace NUMINAMATH_GPT_area_of_flowerbed_l1246_124604

theorem area_of_flowerbed :
  ∀ (a b : ℕ), 2 * (a + b) = 24 → b + 1 = 3 * (a + 1) → 
  let shorter_side := 3 * a
  let longer_side := 3 * b
  shorter_side * longer_side = 144 :=
by
  sorry

end NUMINAMATH_GPT_area_of_flowerbed_l1246_124604


namespace NUMINAMATH_GPT_water_fall_amount_l1246_124683

theorem water_fall_amount (M_before J_before M_after J_after n : ℕ) 
  (h1 : M_before = 48) 
  (h2 : M_before = J_before + 32)
  (h3 : M_after = M_before + n) 
  (h4 : J_after = J_before + n)
  (h5 : M_after = 2 * J_after) : 
  n = 16 :=
by 
  -- proof omitted
  sorry

end NUMINAMATH_GPT_water_fall_amount_l1246_124683


namespace NUMINAMATH_GPT_roberto_outfits_l1246_124608

-- Roberto's wardrobe constraints
def num_trousers : ℕ := 5
def num_shirts : ℕ := 6
def num_jackets : ℕ := 4
def num_shoes : ℕ := 3
def restricted_jacket_shoes : ℕ := 2

-- The total number of valid outfits
def total_outfits_with_constraint : ℕ := 330

-- Proving the equivalent of the problem statement
theorem roberto_outfits :
  (num_trousers * num_shirts * (num_jackets - 1) * num_shoes) + (num_trousers * num_shirts * 1 * restricted_jacket_shoes) = total_outfits_with_constraint :=
by
  sorry

end NUMINAMATH_GPT_roberto_outfits_l1246_124608


namespace NUMINAMATH_GPT_gcd_72_and_120_l1246_124667

theorem gcd_72_and_120 : Nat.gcd 72 120 = 24 := 
by
  sorry

end NUMINAMATH_GPT_gcd_72_and_120_l1246_124667


namespace NUMINAMATH_GPT_cannot_form_square_with_sticks_l1246_124620

theorem cannot_form_square_with_sticks
    (num_1cm_sticks : ℕ)
    (num_2cm_sticks : ℕ)
    (num_3cm_sticks : ℕ)
    (num_4cm_sticks : ℕ)
    (len_1cm_stick : ℕ)
    (len_2cm_stick : ℕ)
    (len_3cm_stick : ℕ)
    (len_4cm_stick : ℕ)
    (sum_lengths : ℕ) :
    num_1cm_sticks = 6 →
    num_2cm_sticks = 3 →
    num_3cm_sticks = 6 →
    num_4cm_sticks = 5 →
    len_1cm_stick = 1 →
    len_2cm_stick = 2 →
    len_3cm_stick = 3 →
    len_4cm_stick = 4 →
    sum_lengths = num_1cm_sticks * len_1cm_stick + 
                  num_2cm_sticks * len_2cm_stick + 
                  num_3cm_sticks * len_3cm_stick + 
                  num_4cm_sticks * len_4cm_stick →
    ∃ (s : ℕ), sum_lengths = 4 * s → False := 
by
  intros num_1cm_sticks_eq num_2cm_sticks_eq num_3cm_sticks_eq num_4cm_sticks_eq
         len_1cm_stick_eq len_2cm_stick_eq len_3cm_stick_eq len_4cm_stick_eq
         sum_lengths_def

  sorry

end NUMINAMATH_GPT_cannot_form_square_with_sticks_l1246_124620


namespace NUMINAMATH_GPT_find_t_of_decreasing_function_l1246_124605

theorem find_t_of_decreasing_function 
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_A : f 0 = 4)
  (h_B : f 3 = -2)
  (h_solution_set : ∀ x, |f (x + 1) - 1| < 3 ↔ -1 < x ∧ x < 2) :
  (1 : ℝ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_t_of_decreasing_function_l1246_124605


namespace NUMINAMATH_GPT_find_r_l1246_124641

theorem find_r (r s : ℝ) (h_quadratic : ∀ y, y^2 - r * y - s = 0) (h_r_pos : r > 0) 
    (h_root_diff : ∀ (y₁ y₂ : ℝ), (y₁ = (r + Real.sqrt (r^2 + 4 * s)) / 2 
        ∧ y₂ = (r - Real.sqrt (r^2 + 4 * s)) / 2) → |y₁ - y₂| = 2) : r = 2 :=
sorry

end NUMINAMATH_GPT_find_r_l1246_124641


namespace NUMINAMATH_GPT_anika_more_than_twice_reeta_l1246_124649

theorem anika_more_than_twice_reeta (R A M : ℕ) (h1 : R = 20) (h2 : A + R = 64) (h3 : A = 2 * R + M) : M = 4 :=
by
  sorry

end NUMINAMATH_GPT_anika_more_than_twice_reeta_l1246_124649


namespace NUMINAMATH_GPT_find_number_of_moles_of_CaCO3_formed_l1246_124660

-- Define the molar ratios and the given condition in structures.
structure Reaction :=
  (moles_CaOH2 : ℕ)
  (moles_CO2 : ℕ)
  (moles_CaCO3 : ℕ)

-- Define a balanced reaction for Ca(OH)2 + CO2 -> CaCO3 + H2O with 1:1 molar ratio.
def balanced_reaction (r : Reaction) : Prop :=
  r.moles_CaOH2 = r.moles_CO2 ∧ r.moles_CaCO3 = r.moles_CO2

-- Define the given condition, which is we have 3 moles of CO2 and formed 3 moles of CaCO3.
def given_condition : Reaction :=
  { moles_CaOH2 := 3, moles_CO2 := 3, moles_CaCO3 := 3 }

-- Theorem: Given 3 moles of CO2, we need to prove 3 moles of CaCO3 are formed based on the balanced reaction.
theorem find_number_of_moles_of_CaCO3_formed :
  balanced_reaction given_condition :=
by {
  -- This part will contain the proof when implemented.
  sorry
}

end NUMINAMATH_GPT_find_number_of_moles_of_CaCO3_formed_l1246_124660


namespace NUMINAMATH_GPT_luisa_mpg_l1246_124610

theorem luisa_mpg
  (d_grocery d_mall d_pet d_home : ℕ)
  (cost_per_gal total_cost : ℚ)
  (total_miles : ℕ )
  (total_gallons : ℚ)
  (mpg : ℚ):
  d_grocery = 10 →
  d_mall = 6 →
  d_pet = 5 →
  d_home = 9 →
  cost_per_gal = 3.5 →
  total_cost = 7 →
  total_miles = d_grocery + d_mall + d_pet + d_home →
  total_gallons = total_cost / cost_per_gal →
  mpg = total_miles / total_gallons →
  mpg = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_luisa_mpg_l1246_124610


namespace NUMINAMATH_GPT_problem_l1246_124614

def f (x : ℝ) : ℝ := sorry  -- f is a function from ℝ to ℝ

theorem problem (h : ∀ x : ℝ, 3 * f x + f (2 - x) = 4 * x^2 + 1) : f 5 = 133 / 4 := 
by 
  sorry -- the proof is omitted

end NUMINAMATH_GPT_problem_l1246_124614


namespace NUMINAMATH_GPT_find_constants_l1246_124621

noncomputable def f (a b x : ℝ) : ℝ :=
(a * x + b) / (x + 1)

theorem find_constants (a b : ℝ) (x : ℝ) (h : x ≠ -1) : 
  (f a b (f a b x) = x) → (a = -1 ∧ ∀ b, ∃ c : ℝ, b = c) :=
by 
  sorry

end NUMINAMATH_GPT_find_constants_l1246_124621


namespace NUMINAMATH_GPT_x_pow_10_eq_correct_answer_l1246_124686

noncomputable def x : ℝ := sorry

theorem x_pow_10_eq_correct_answer (h : x + (1 / x) = Real.sqrt 5) : 
  x^10 = (50 + 25 * Real.sqrt 5) / 2 := 
sorry

end NUMINAMATH_GPT_x_pow_10_eq_correct_answer_l1246_124686


namespace NUMINAMATH_GPT_max_ball_height_l1246_124601

/-- 
The height (in feet) of a ball traveling on a parabolic path is given by -20t^2 + 80t + 36,
where t is the time after launch. This theorem shows that the maximum height of the ball is 116 feet.
-/
theorem max_ball_height : ∃ t : ℝ, ∀ t', -20 * t^2 + 80 * t + 36 ≤ -20 * t'^2 + 80 * t' + 36 → -20 * t^2 + 80 * t + 36 = 116 :=
sorry

end NUMINAMATH_GPT_max_ball_height_l1246_124601


namespace NUMINAMATH_GPT_ned_price_per_game_l1246_124645

def number_of_games : Nat := 15
def non_working_games : Nat := 6
def total_earnings : Nat := 63
def number_of_working_games : Nat := number_of_games - non_working_games
def price_per_working_game : Nat := total_earnings / number_of_working_games

theorem ned_price_per_game : price_per_working_game = 7 :=
by
  sorry

end NUMINAMATH_GPT_ned_price_per_game_l1246_124645


namespace NUMINAMATH_GPT_platform_length_l1246_124636

theorem platform_length (speed_km_hr : ℝ) (time_man : ℝ) (time_platform : ℝ) (L : ℝ) (P : ℝ) :
  speed_km_hr = 54 → time_man = 20 → time_platform = 22 → 
  L = (speed_km_hr * (1000 / 3600)) * time_man →
  L + P = (speed_km_hr * (1000 / 3600)) * time_platform → 
  P = 30 := 
by
  intros hs ht1 ht2 hL hLP
  sorry

end NUMINAMATH_GPT_platform_length_l1246_124636


namespace NUMINAMATH_GPT_enterprise_b_pays_more_in_2015_l1246_124691

variable (a b x y : ℝ)
variable (ha2x : a + 2 * x = b)
variable (ha1y : a * (1+y)^2 = b)

theorem enterprise_b_pays_more_in_2015 : b * (1 + y) > b + x := by
  sorry

end NUMINAMATH_GPT_enterprise_b_pays_more_in_2015_l1246_124691


namespace NUMINAMATH_GPT_lim_sup_eq_Union_lim_inf_l1246_124616

open Set

theorem lim_sup_eq_Union_lim_inf
  (Ω : Type*)
  (A : ℕ → Set Ω) :
  (⋂ n, ⋃ k ≥ n, A k) = ⋃ (n_infty : ℕ → ℕ) (hn : StrictMono n_infty), ⋃ n, ⋂ k ≥ n, A (n_infty k) :=
by
  sorry

end NUMINAMATH_GPT_lim_sup_eq_Union_lim_inf_l1246_124616


namespace NUMINAMATH_GPT_zoo_peacocks_l1246_124689

theorem zoo_peacocks (R P : ℕ) (h1 : R + P = 60) (h2 : 4 * R + 2 * P = 192) : P = 24 :=
by
  sorry

end NUMINAMATH_GPT_zoo_peacocks_l1246_124689


namespace NUMINAMATH_GPT_cake_has_more_calories_l1246_124669

-- Define the conditions
def cake_slices : Nat := 8
def cake_calories_per_slice : Nat := 347
def brownie_count : Nat := 6
def brownie_calories_per_brownie : Nat := 375

-- Define the total calories for the cake and the brownies
def total_cake_calories : Nat := cake_slices * cake_calories_per_slice
def total_brownie_calories : Nat := brownie_count * brownie_calories_per_brownie

-- Prove the difference in calories
theorem cake_has_more_calories : 
  total_cake_calories - total_brownie_calories = 526 :=
by
  sorry

end NUMINAMATH_GPT_cake_has_more_calories_l1246_124669


namespace NUMINAMATH_GPT_john_wages_decrease_percentage_l1246_124690

theorem john_wages_decrease_percentage (W : ℝ) (P : ℝ) :
  (0.20 * (W - P/100 * W)) = 0.50 * (0.30 * W) → P = 25 :=
by 
  intro h
  -- Simplification and other steps omitted; focus on structure
  sorry

end NUMINAMATH_GPT_john_wages_decrease_percentage_l1246_124690


namespace NUMINAMATH_GPT_expression_value_as_fraction_l1246_124613

theorem expression_value_as_fraction (x y : ℕ) (hx : x = 3) (hy : y = 5) : 
  ( ( (1 / (y : ℚ)) / (1 / (x : ℚ)) ) ^ 2 ) = 9 / 25 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_as_fraction_l1246_124613


namespace NUMINAMATH_GPT_science_homework_is_50_minutes_l1246_124619

-- Define the times for each homework and project in minutes
def total_time : ℕ := 3 * 60  -- 3 hours converted to minutes
def math_homework : ℕ := 45
def english_homework : ℕ := 30
def history_homework : ℕ := 25
def special_project : ℕ := 30

-- Define a function to compute the time for science homework
def science_homework_time 
  (total_time : ℕ) 
  (math_time : ℕ) 
  (english_time : ℕ) 
  (history_time : ℕ) 
  (project_time : ℕ) : ℕ :=
  total_time - (math_time + english_time + history_time + project_time)

-- The theorem to prove the time Porche's science homework takes
theorem science_homework_is_50_minutes : 
  science_homework_time total_time math_homework english_homework history_homework special_project = 50 := 
sorry

end NUMINAMATH_GPT_science_homework_is_50_minutes_l1246_124619


namespace NUMINAMATH_GPT_decipher_proof_l1246_124624

noncomputable def decipher_message (n : ℕ) (hidden_message : String) :=
  if n = 2211169691162 then hidden_message = "Kiss me, dearest" else false

theorem decipher_proof :
  decipher_message 2211169691162 "Kiss me, dearest" = true :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_decipher_proof_l1246_124624


namespace NUMINAMATH_GPT_find_parabola_vertex_l1246_124696

-- Define the parabola with specific roots.
def parabola (x : ℝ) : ℝ := -x^2 + 2 * x + 24

-- Define the vertex of the parabola.
def vertex : ℝ × ℝ := (1, 25)

-- Prove that the vertex of the parabola is indeed at (1, 25).
theorem find_parabola_vertex : vertex = (1, 25) :=
  sorry

end NUMINAMATH_GPT_find_parabola_vertex_l1246_124696


namespace NUMINAMATH_GPT_kylie_coins_l1246_124674

open Nat

theorem kylie_coins :
  ∀ (coins_from_piggy_bank coins_from_brother coins_from_father coins_given_to_friend total_coins_left : ℕ),
  coins_from_piggy_bank = 15 →
  coins_from_brother = 13 →
  coins_from_father = 8 →
  coins_given_to_friend = 21 →
  total_coins_left = coins_from_piggy_bank + coins_from_brother + coins_from_father - coins_given_to_friend →
  total_coins_left = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_kylie_coins_l1246_124674


namespace NUMINAMATH_GPT_num_real_solutions_l1246_124670

theorem num_real_solutions (x : ℝ) (A B : Set ℝ) (hx : x ∈ A) (hx2 : x^2 ∈ A) :
  A = {0, 1, 2, x} → B = {1, x^2} → A ∪ B = A → 
  ∃! y : ℝ, y = -Real.sqrt 2 ∨ y = Real.sqrt 2 :=
by
  intro hA hB hA_union_B
  sorry

end NUMINAMATH_GPT_num_real_solutions_l1246_124670


namespace NUMINAMATH_GPT_sam_received_87_l1246_124687

def sam_total_money : Nat :=
  sorry

theorem sam_received_87 (spent left_over : Nat) (h1 : spent = 64) (h2 : left_over = 23) :
  sam_total_money = spent + left_over :=
by
  rw [h1, h2]
  sorry

example : sam_total_money = 64 + 23 :=
  sam_received_87 64 23 rfl rfl

end NUMINAMATH_GPT_sam_received_87_l1246_124687


namespace NUMINAMATH_GPT_exists_x_nat_l1246_124627

theorem exists_x_nat (a c : ℕ) (b : ℤ) : ∃ x : ℕ, (a^x + x) % c = b % c :=
by
  sorry

end NUMINAMATH_GPT_exists_x_nat_l1246_124627


namespace NUMINAMATH_GPT_simplify_expression_l1246_124628

variable (x y : ℝ)

-- Define the proposition
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1246_124628


namespace NUMINAMATH_GPT_minimum_value_inequality_l1246_124681

variable {x y z : ℝ}

theorem minimum_value_inequality (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x + y + z = 4) :
  (1 / x + 4 / y + 9 / z) ≥ 9 :=
sorry

end NUMINAMATH_GPT_minimum_value_inequality_l1246_124681


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1246_124659

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ^ 2 = a n * a (n + 2)

theorem necessary_but_not_sufficient_condition
  (a : ℕ → ℝ) :
  condition a → ¬ is_geometric_sequence a :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1246_124659


namespace NUMINAMATH_GPT_intersection_A_complement_B_l1246_124653

-- Definitions of sets A and B and their complement in the universal set R, which is the real numbers.
def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2 * x > 0}
def complement_R_B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- The proof statement verifying the intersection of set A with the complement of set B.
theorem intersection_A_complement_B : A ∩ complement_R_B = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_A_complement_B_l1246_124653


namespace NUMINAMATH_GPT_adult_ticket_cost_l1246_124646

-- Definitions based on the conditions
def num_adults : ℕ := 10
def num_children : ℕ := 11
def total_bill : ℝ := 124
def child_ticket_cost : ℝ := 4

-- The proof which determines the cost of one adult ticket
theorem adult_ticket_cost : ∃ (A : ℝ), A * num_adults = total_bill - (num_children * child_ticket_cost) ∧ A = 8 := 
by
  sorry

end NUMINAMATH_GPT_adult_ticket_cost_l1246_124646


namespace NUMINAMATH_GPT_billboard_shorter_side_length_l1246_124617

theorem billboard_shorter_side_length
  (L W : ℝ)
  (h1 : L * W = 120)
  (h2 : 2 * L + 2 * W = 46) :
  min L W = 8 :=
by
  sorry

end NUMINAMATH_GPT_billboard_shorter_side_length_l1246_124617


namespace NUMINAMATH_GPT_diet_soda_bottles_l1246_124672

/-- Define variables for the number of bottles. -/
def total_bottles : ℕ := 38
def regular_soda : ℕ := 30

/-- Define the problem of finding the number of diet soda bottles -/
def diet_soda := total_bottles - regular_soda

/-- Claim that the number of diet soda bottles is 8 -/
theorem diet_soda_bottles : diet_soda = 8 :=
by
  sorry

end NUMINAMATH_GPT_diet_soda_bottles_l1246_124672


namespace NUMINAMATH_GPT_lines_intersect_at_point_l1246_124666

def ParametricLine1 (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 4 - 3 * t)

def ParametricLine2 (u : ℝ) : ℝ × ℝ :=
  (-2 + 3 * u, 5 - u)

theorem lines_intersect_at_point :
  ∃ t u : ℝ, ParametricLine1 t = ParametricLine2 u ∧ ParametricLine1 t = (-5, 13) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_at_point_l1246_124666


namespace NUMINAMATH_GPT_vector_parallel_sum_l1246_124664

theorem vector_parallel_sum (m n : ℝ) (a b : ℝ × ℝ × ℝ)
  (h_a : a = (2, -1, 3))
  (h_b : b = (4, m, n))
  (h_parallel : ∃ k : ℝ, a = k • b) :
  m + n = 4 :=
sorry

end NUMINAMATH_GPT_vector_parallel_sum_l1246_124664


namespace NUMINAMATH_GPT_simplify_expression_l1246_124658

noncomputable def givenExpression : ℝ := 
  abs (-0.01) ^ 2 - (-5 / 8) ^ 0 - 3 ^ (Real.log 2 / Real.log 3) + 
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 5) + Real.log 5

theorem simplify_expression : givenExpression = -1.9999 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1246_124658


namespace NUMINAMATH_GPT_evaluate_at_points_l1246_124642

noncomputable def f (x : ℝ) : ℝ :=
if x > 3 then x^2 - 3*x + 2
else if -2 ≤ x ∧ x ≤ 3 then -3*x + 5
else 9

theorem evaluate_at_points : f (-3) + f (0) + f (4) = 20 := by
  sorry

end NUMINAMATH_GPT_evaluate_at_points_l1246_124642


namespace NUMINAMATH_GPT_floor_sqrt_20_squared_eq_16_l1246_124656

theorem floor_sqrt_20_squared_eq_16 : (Int.floor (Real.sqrt 20))^2 = 16 := by
  sorry

end NUMINAMATH_GPT_floor_sqrt_20_squared_eq_16_l1246_124656


namespace NUMINAMATH_GPT_total_spent_l1246_124698

-- Define the conditions
def cost_fix_automobile := 350
def cost_fix_formula (S : ℕ) := 3 * S + 50

-- Prove the total amount spent is $450
theorem total_spent (S : ℕ) (h : cost_fix_automobile = cost_fix_formula S) :
  S + cost_fix_automobile = 450 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_l1246_124698
