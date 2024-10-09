import Mathlib

namespace eliza_height_l273_27376

theorem eliza_height
  (n : ℕ) (H_total : ℕ) 
  (sib1_height : ℕ) (sib2_height : ℕ) (sib3_height : ℕ)
  (eliza_height : ℕ) (last_sib_height : ℕ) :
  n = 5 →
  H_total = 330 →
  sib1_height = 66 →
  sib2_height = 66 →
  sib3_height = 60 →
  eliza_height = last_sib_height - 2 →
  H_total = sib1_height + sib2_height + sib3_height + eliza_height + last_sib_height →
  eliza_height = 68 :=
by
  intros n_eq H_total_eq sib1_eq sib2_eq sib3_eq eliza_eq H_sum_eq
  sorry

end eliza_height_l273_27376


namespace quarter_circle_area_ratio_l273_27383

theorem quarter_circle_area_ratio (R : ℝ) (hR : 0 < R) :
  let O := pi * R^2
  let AXC := pi * (R/2)^2 / 4
  let BYD := pi * (R/2)^2 / 4
  (2 * (AXC + BYD) / O = 1 / 8) := 
by
  let O := pi * R^2
  let AXC := pi * (R/2)^2 / 4
  let BYD := pi * (R/2)^2 / 4
  sorry

end quarter_circle_area_ratio_l273_27383


namespace herd_compuation_l273_27365

theorem herd_compuation (a b c : ℕ) (total_animals total_payment : ℕ) 
  (H1 : total_animals = a + b + 10 * c) 
  (H2 : total_payment = 20 * a + 10 * b + 10 * c) 
  (H3 : total_animals = 100) 
  (H4 : total_payment = 200) :
  a = 1 ∧ b = 9 ∧ 10 * c = 90 :=
by
  sorry

end herd_compuation_l273_27365


namespace inequality_solution_l273_27345

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)

theorem inequality_solution :
  -3 < x ∧ x < -1 ↔ f (x^2 - 3) < f (x - 1) :=
sorry

end inequality_solution_l273_27345


namespace sample_size_is_40_l273_27395

theorem sample_size_is_40 (total_students : ℕ) (sample_students : ℕ) (h1 : total_students = 240) (h2 : sample_students = 40) : sample_students = 40 :=
by
  sorry

end sample_size_is_40_l273_27395


namespace certain_number_is_sixteen_l273_27309

theorem certain_number_is_sixteen (x : ℝ) (h : x ^ 5 = 4 ^ 10) : x = 16 :=
by
  sorry

end certain_number_is_sixteen_l273_27309


namespace average_age_increase_l273_27306

variable (A B C : ℕ)

theorem average_age_increase (A : ℕ) (B : ℕ) (C : ℕ) (h1 : 21 < B) (h2 : 23 < C) (h3 : A + B + C > A + 21 + 23) :
  (B + C) / 2 > 22 := by
  sorry

end average_age_increase_l273_27306


namespace f_at_neg_8_5_pi_eq_pi_div_2_l273_27353

def f (x : Real) : Real := sorry

axiom functional_eqn (x : Real) : f (x + (3 * Real.pi / 2)) = -1 / f x
axiom f_interval (x : Real) (h : x ∈ Set.Icc (-Real.pi) Real.pi) : f x = x * Real.sin x

theorem f_at_neg_8_5_pi_eq_pi_div_2 : f (-8.5 * Real.pi) = Real.pi / 2 := 
  sorry

end f_at_neg_8_5_pi_eq_pi_div_2_l273_27353


namespace tangents_intersection_perpendicular_parabola_l273_27330

theorem tangents_intersection_perpendicular_parabola :
  ∀ (C D : ℝ × ℝ), C.2 = 4 * C.1 ^ 2 → D.2 = 4 * D.1 ^ 2 → 
  (8 * C.1) * (8 * D.1) = -1 → 
  ∃ Q : ℝ × ℝ, Q.2 = -1 / 16 :=
by
  sorry

end tangents_intersection_perpendicular_parabola_l273_27330


namespace sum_prime_factors_of_77_l273_27308

theorem sum_prime_factors_of_77 : ∃ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 ∧ p1 + p2 = 18 := by
  sorry

end sum_prime_factors_of_77_l273_27308


namespace black_squares_in_35th_row_l273_27337

-- Define the condition for the starting color based on the row
def starts_with_black (n : ℕ) : Prop := n % 2 = 1
def ends_with_white (n : ℕ) : Prop := true  -- This is trivially true by the problem condition
def total_squares (n : ℕ) : ℕ := 2 * n 
-- Black squares are half of the total squares for rows starting with a black square
def black_squares (n : ℕ) : ℕ := total_squares n / 2

theorem black_squares_in_35th_row : black_squares 35 = 35 :=
sorry

end black_squares_in_35th_row_l273_27337


namespace chemistry_more_than_physics_l273_27385

theorem chemistry_more_than_physics
  (M P C : ℕ)
  (h1 : M + P = 60)
  (h2 : (M + C) / 2 = 35) :
  ∃ x : ℕ, C = P + x ∧ x = 10 := 
by
  sorry

end chemistry_more_than_physics_l273_27385


namespace find_BP_l273_27356

theorem find_BP
  (A B C D P : Type) 
  (AP PC BP DP : ℝ)
  (hAP : AP = 8) 
  (hPC : PC = 1)
  (hBD : BD = 6)
  (hBP_less_DP : BP < DP) 
  (hPower_of_Point : AP * PC = BP * DP)
  : BP = 2 := 
by {
  sorry
}

end find_BP_l273_27356


namespace b_geometric_l273_27335

def a (n : ℕ) : ℝ := sorry
def b (n : ℕ) : ℝ := sorry

axiom a1 : a 1 = 1
axiom a_n_recurrence (n : ℕ) : a n + a (n + 1) = 1 / (3^n)
axiom b_def (n : ℕ) : b n = 3^(n - 1) * a n - 1/4

theorem b_geometric (n : ℕ) : b (n + 1) = -3 * b n := sorry

end b_geometric_l273_27335


namespace find_m_n_sum_product_l273_27396

noncomputable def sum_product_of_roots (m n : ℝ) : Prop :=
  (m^2 - 4*m - 12 = 0) ∧ (n^2 - 4*n - 12 = 0) 

theorem find_m_n_sum_product (m n : ℝ) (h : sum_product_of_roots m n) :
  m + n + m * n = -8 :=
by 
  sorry

end find_m_n_sum_product_l273_27396


namespace metres_sold_is_200_l273_27343

-- Define the conditions
def loss_per_metre : ℕ := 6
def cost_price_per_metre : ℕ := 66
def total_selling_price : ℕ := 12000

-- Define the selling price per metre based on the conditions
def selling_price_per_metre := cost_price_per_metre - loss_per_metre

-- Define the number of metres sold
def metres_sold : ℕ := total_selling_price / selling_price_per_metre

-- Proof statement: Check if the number of metres sold equals 200
theorem metres_sold_is_200 : metres_sold = 200 :=
  by
  sorry

end metres_sold_is_200_l273_27343


namespace find_a_from_quadratic_inequality_l273_27394

theorem find_a_from_quadratic_inequality :
  ∀ (a : ℝ), (∀ x : ℝ, (x > - (1 / 2)) ∧ (x < 1 / 3) → a * x^2 - 2 * x + 2 > 0) → a = -12 :=
by
  intros a h
  have h1 := h (-1 / 2)
  have h2 := h (1 / 3)
  sorry

end find_a_from_quadratic_inequality_l273_27394


namespace complement_union_l273_27366

namespace SetTheory

def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

theorem complement_union (U A B: Set ℕ) (hU : U = {1, 3, 5, 9}) (hA : A = {1, 3, 9}) (hB : B = {1, 9}) :
  U \ (A ∪ B) = {5} :=
by
  sorry

end SetTheory

end complement_union_l273_27366


namespace total_bricks_used_l273_27380

def numCoursesPerWall : Nat := 6
def bricksPerCourse : Nat := 10
def numWalls : Nat := 4
def unfinishedCoursesLastWall : Nat := 2

theorem total_bricks_used : 
  let totalCourses := numWalls * numCoursesPerWall
  let bricksRequired := totalCourses * bricksPerCourse
  let bricksMissing := unfinishedCoursesLastWall * bricksPerCourse
  let bricksUsed := bricksRequired - bricksMissing
  bricksUsed = 220 := 
by
  sorry

end total_bricks_used_l273_27380


namespace find_x_parallel_find_x_perpendicular_l273_27318

def a (x : ℝ) : ℝ × ℝ := (x, x + 2)
def b : ℝ × ℝ := (1, 2)

-- Given that a vector is proportional to another
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Given that the dot product is zero
def are_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_x_parallel (x : ℝ) (h : are_parallel (a x) b) : x = 2 :=
by sorry

theorem find_x_perpendicular (x : ℝ) (h : are_perpendicular (a x - b) b) : x = (1 / 3 : ℝ) :=
by sorry

end find_x_parallel_find_x_perpendicular_l273_27318


namespace triangle_perimeter_is_26_l273_27390

-- Define the lengths of the medians as given conditions
def median1 := 3
def median2 := 4
def median3 := 6

-- Define the perimeter of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- The theorem to prove that the perimeter is 26 cm
theorem triangle_perimeter_is_26 :
  perimeter (2 * median1) (2 * median2) (2 * median3) = 26 :=
by
  -- Calculation follows directly from the definition
  sorry

end triangle_perimeter_is_26_l273_27390


namespace arithmetic_seq_third_term_l273_27386

theorem arithmetic_seq_third_term
  (a d : ℝ)
  (h : a + (a + 2 * d) = 10) :
  a + d = 5 := by
  sorry

end arithmetic_seq_third_term_l273_27386


namespace binary_modulo_eight_l273_27397

theorem binary_modulo_eight : (0b1110101101101 : ℕ) % 8 = 5 := 
by {
  -- This is where the proof would go.
  sorry
}

end binary_modulo_eight_l273_27397


namespace inequality_problem_l273_27373

theorem inequality_problem
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_sum : a + b + c ≤ 3) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) ≥ 3 / 2 :=
by sorry

end inequality_problem_l273_27373


namespace minimize_sum_first_n_terms_l273_27313

noncomputable def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

noncomputable def sum_first_n_terms (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n-1) / 2) * d

theorem minimize_sum_first_n_terms (a₁ : ℤ) (a₃_plus_a₅ : ℤ) (n_min : ℕ) :
  a₁ = -9 → a₃_plus_a₅ = -6 → n_min = 5 := by
  sorry

end minimize_sum_first_n_terms_l273_27313


namespace problem1_problem2_l273_27327

-- Define the first problem
theorem problem1 : (Real.cos (25 / 3 * Real.pi) + Real.tan (-15 / 4 * Real.pi)) = 3 / 2 :=
by
  sorry

-- Define vector operations and the problem
variables (a b : ℝ)

theorem problem2 : 2 * (a - b) - (2 * a + b) + 3 * b = 0 :=
by
  sorry

end problem1_problem2_l273_27327


namespace hypotenuse_of_triangle_PQR_l273_27301

theorem hypotenuse_of_triangle_PQR (PA PB PC QR : ℝ) (h1: PA = 2) (h2: PB = 3) (h3: PC = 2)
  (h4: PA + PB + PC = QR) (h5: QR = PA + 3 + 2 * PA): QR = 5 * Real.sqrt 2 := 
sorry

end hypotenuse_of_triangle_PQR_l273_27301


namespace positive_operation_l273_27391

def operation_a := 1 + (-2)
def operation_b := 1 - (-2)
def operation_c := 1 * (-2)
def operation_d := 1 / (-2)

theorem positive_operation : operation_b > 0 ∧ 
  (operation_a <= 0) ∧ (operation_c <= 0) ∧ (operation_d <= 0) := by
  sorry

end positive_operation_l273_27391


namespace infinite_solutions_if_one_exists_l273_27341

namespace RationalSolutions

def has_rational_solution (a b : ℚ) : Prop :=
  ∃ (x y : ℚ), a * x^2 + b * y^2 = 1

def infinite_rational_solutions (a b : ℚ) : Prop :=
  ∀ (x₀ y₀ : ℚ), (a * x₀^2 + b * y₀^2 = 1) → ∃ (f : ℕ → ℚ × ℚ), ∀ n : ℕ, a * (f n).1^2 + b * (f n).2^2 = 1 ∧ (f 0 = (x₀, y₀)) ∧ ∀ m n : ℕ, m ≠ n → (f m) ≠ (f n)

theorem infinite_solutions_if_one_exists (a b : ℚ) (h : has_rational_solution a b) : infinite_rational_solutions a b :=
  sorry

end RationalSolutions

end infinite_solutions_if_one_exists_l273_27341


namespace aluminum_weight_proportional_l273_27379

noncomputable def area_equilateral_triangle (side_length : ℝ) : ℝ :=
  (side_length * side_length * Real.sqrt 3) / 4

theorem aluminum_weight_proportional (weight1 weight2 : ℝ) 
  (side_length1 side_length2 : ℝ)
  (h_density_thickness : ∀ s t, area_equilateral_triangle s * weight1 = area_equilateral_triangle t * weight2)
  (h_weight1 : weight1 = 20)
  (h_side_length1 : side_length1 = 2)
  (h_side_length2 : side_length2 = 4) : 
  weight2 = 80 :=
by
  sorry

end aluminum_weight_proportional_l273_27379


namespace non_coincident_angles_l273_27357

theorem non_coincident_angles : ¬ ∃ k : ℤ, 1050 - (-300) = k * 360 := by
  sorry

end non_coincident_angles_l273_27357


namespace swap_tens_units_digits_l273_27346

theorem swap_tens_units_digits (x a b : ℕ) (h1 : 9 < x) (h2 : x < 100) (h3 : a = x / 10) (h4 : b = x % 10) :
  10 * b + a = (x % 10) * 10 + (x / 10) :=
by
  sorry

end swap_tens_units_digits_l273_27346


namespace xyz_squared_sum_l273_27348

theorem xyz_squared_sum (x y z : ℤ) 
  (h1 : |x + y| + |y + z| + |z + x| = 4)
  (h2 : |x - y| + |y - z| + |z - x| = 2) :
  x^2 + y^2 + z^2 = 2 := 
by 
  sorry

end xyz_squared_sum_l273_27348


namespace sophia_fraction_of_pie_l273_27389

theorem sophia_fraction_of_pie
  (weight_fridge : ℕ) (weight_eaten : ℕ)
  (h1 : weight_fridge = 1200)
  (h2 : weight_eaten = 240) :
  (weight_eaten : ℚ) / ((weight_fridge + weight_eaten : ℚ)) = (1 / 6) :=
by
  sorry

end sophia_fraction_of_pie_l273_27389


namespace sequence_term_2010_l273_27321

theorem sequence_term_2010 :
  ∀ (a : ℕ → ℤ), a 1 = 1 → a 2 = 2 → 
    (∀ n : ℕ, n ≥ 3 → a n = a (n - 1) - a (n - 2)) → 
    a 2010 = -1 :=
by
  sorry

end sequence_term_2010_l273_27321


namespace smallest_positive_sum_l273_27393

structure ArithmeticSequence :=
  (a_n : ℕ → ℤ)  -- The sequence is an integer sequence
  (d : ℤ)        -- The common difference of the sequence

def sum_of_first_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n * (seq.a_n 1 + seq.a_n n)) / 2  -- Sum of first n terms

def condition (seq : ArithmeticSequence) : Prop :=
  (seq.a_n 11 < -1 * seq.a_n 10)

theorem smallest_positive_sum (seq : ArithmeticSequence) (H : condition seq) :
  ∃ n, sum_of_first_n seq n > 0 ∧ ∀ m < n, sum_of_first_n seq m ≤ 0 → n = 19 :=
sorry

end smallest_positive_sum_l273_27393


namespace necessary_but_not_sufficient_l273_27364

variable {a b c : ℝ}

theorem necessary_but_not_sufficient (h1 : b^2 - 4 * a * c ≥ 0) (h2 : a * c > 0) (h3 : a * b < 0) : 
  ¬∀ r1 r2 : ℝ, (r1 = (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) ∧ (r2 = (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) → r1 > 0 ∧ r2 > 0 :=
sorry

end necessary_but_not_sufficient_l273_27364


namespace largest_radius_cone_l273_27315

structure Crate :=
  (width : ℝ)
  (depth : ℝ)
  (height : ℝ)

structure Cone :=
  (radius : ℝ)
  (height : ℝ)

noncomputable def larger_fit_within_crate (c : Crate) (cone : Cone) : Prop :=
  cone.radius = min c.width c.depth / 2 ∧ cone.height = max (max c.width c.depth) c.height

theorem largest_radius_cone (c : Crate) (cone : Cone) : 
  c.width = 5 → c.depth = 8 → c.height = 12 → larger_fit_within_crate c cone → cone.radius = 2.5 :=
by
  sorry

end largest_radius_cone_l273_27315


namespace arrange_descending_order_l273_27388

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.log (1 / 3) / Real.log 2
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem arrange_descending_order : c > a ∧ a > b := by
  sorry

end arrange_descending_order_l273_27388


namespace find_p_q_r_l273_27305

theorem find_p_q_r : 
  ∃ (p q r : ℕ), 
  p > 0 ∧ q > 0 ∧ r > 0 ∧ 
  4 * (Real.sqrt (Real.sqrt 7) - Real.sqrt (Real.sqrt 6)) 
  = Real.sqrt (Real.sqrt p) + Real.sqrt (Real.sqrt q) - Real.sqrt (Real.sqrt r) 
  ∧ p + q + r = 99 := 
sorry

end find_p_q_r_l273_27305


namespace count_integers_satisfying_conditions_l273_27399

theorem count_integers_satisfying_conditions :
  (∃ (s : Finset ℤ), s.card = 3 ∧
  ∀ x : ℤ, x ∈ s ↔ (-5 ≤ x ∧ x ≤ -3)) :=
by {
  sorry
}

end count_integers_satisfying_conditions_l273_27399


namespace cuboid_first_dimension_l273_27342

theorem cuboid_first_dimension (x : ℕ)
  (h₁ : ∃ n : ℕ, n = 24) 
  (h₂ : ∃ a b c d e f g : ℕ, x = a ∧ 9 = b ∧ 12 = c ∧ a * b * c = d * e * f ∧ g = Nat.gcd b c ∧ f = (g^3) ∧ e = (n * f) ∧ d = 648) : 
  x = 6 :=
by
  sorry

end cuboid_first_dimension_l273_27342


namespace compute_quotient_of_q_and_r_l273_27310

theorem compute_quotient_of_q_and_r (p q r s t : ℤ) (h_eq_4 : 256 * p + 64 * q + 16 * r + 4 * s + t = 0)
                                     (h_eq_neg3 : -27 * p + 9 * q - 3 * r + s + t = 0)
                                     (h_eq_0 : t = 0)
                                     (h_p_nonzero : p ≠ 0) :
                                     (q + r) / p = -13 :=
by
  have eq1 := h_eq_4
  have eq2 := h_eq_neg3
  rw [h_eq_0] at eq1 eq2
  sorry

end compute_quotient_of_q_and_r_l273_27310


namespace complex_expression_evaluation_l273_27350

theorem complex_expression_evaluation : (i : ℂ) * (1 + i : ℂ)^2 = -2 := 
by
  sorry

end complex_expression_evaluation_l273_27350


namespace shorter_stick_length_l273_27312

variable (L S : ℝ)

theorem shorter_stick_length
  (h1 : L - S = 12)
  (h2 : (2 / 3) * L = S) :
  S = 24 := by
  sorry

end shorter_stick_length_l273_27312


namespace eval_fraction_l273_27304

theorem eval_fraction (a b : ℕ) : (40 : ℝ) = 2^3 * 5 → (10 : ℝ) = 2 * 5 → (40^56 / 10^28) = 160^28 :=
by 
  sorry

end eval_fraction_l273_27304


namespace sum_of_roots_l273_27323
-- Import Mathlib to cover all necessary functionality.

-- Define the function representing the given equation.
def equation (x : ℝ) : ℝ :=
  (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- State the theorem to be proved.
theorem sum_of_roots : (6 : ℝ) + (-4 / 3) = 14 / 3 :=
by
  sorry

end sum_of_roots_l273_27323


namespace select_more_stable_athlete_l273_27334

-- Define the problem conditions
def athlete_average_score : ℝ := 9
def athlete_A_variance : ℝ := 1.2
def athlete_B_variance : ℝ := 2.4

-- Define what it means to have more stable performance
def more_stable (variance_A variance_B : ℝ) : Prop := variance_A < variance_B

-- The theorem to prove
theorem select_more_stable_athlete :
  more_stable athlete_A_variance athlete_B_variance →
  "A" = "A" :=
by
  sorry

end select_more_stable_athlete_l273_27334


namespace tangent_line_to_curve_at_Mpi_l273_27384

noncomputable def tangent_line_eq_at_point (x : ℝ) (y : ℝ) : Prop :=
  y = (Real.sin x) / x

theorem tangent_line_to_curve_at_Mpi :
  (∀ x y, tangent_line_eq_at_point x y →
    (∃ (m : ℝ), m = -1 / π) →
    (∀ x1 y1 (hx : x1 = π) (hy : y1 = 0), x + π * y - π = 0)) :=
by
  sorry

end tangent_line_to_curve_at_Mpi_l273_27384


namespace a_plus_b_eq_2_l273_27307

theorem a_plus_b_eq_2 (a b : ℝ) 
  (h₁ : 2 = a + b) 
  (h₂ : 4 = a + b / 4) : a + b = 2 :=
by
  sorry

end a_plus_b_eq_2_l273_27307


namespace even_product_probability_l273_27354

def number_on_first_spinner := [3, 6, 5, 10, 15]
def number_on_second_spinner := [7, 6, 11, 12, 13, 14]

noncomputable def probability_even_product : ℚ :=
  1 - (3 / 5) * (3 / 6)

theorem even_product_probability :
  probability_even_product = 7 / 10 :=
by
  sorry

end even_product_probability_l273_27354


namespace total_length_of_board_l273_27358

theorem total_length_of_board (x y : ℝ) (h1 : y = 2 * x) (h2 : y = 46) : x + y = 69 :=
by
  sorry

end total_length_of_board_l273_27358


namespace problem_inequality_l273_27329

variables {a b c x1 x2 x3 x4 x5 : ℝ} 

theorem problem_inequality
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_x1: 0 < x1) (h_pos_x2: 0 < x2) (h_pos_x3: 0 < x3) (h_pos_x4: 0 < x4) (h_pos_x5: 0 < x5)
  (h_sum_abc : a + b + c = 1) (h_prod_x : x1 * x2 * x3 * x4 * x5 = 1) :
  (a * x1^2 + b * x1 + c) * (a * x2^2 + b * x2 + c) * (a * x3^2 + b * x3 + c) * 
  (a * x4^2 + b * x4 + c) * (a * x5^2 + b * x5 + c) ≥ 1 :=
sorry

end problem_inequality_l273_27329


namespace triangle_area_l273_27300

noncomputable def area_of_triangle := 
  let a := 4
  let b := 5
  let c := 6
  let cosA := 3 / 4
  let sinA := Real.sqrt (1 - cosA ^ 2)
  (1 / 2) * b * c * sinA

theorem triangle_area :
  ∃ (a b c : ℝ), a = 4 ∧ b = 5 ∧ c = 6 ∧ 
  a < b ∧ b < c ∧ 
  -- Additional conditions
  (∃ A B C : ℝ, C = 2 * A ∧ 
   Real.cos A = 3 / 4 ∧ 
   Real.sin A * Real.cos A = sinA * cosA ∧ 
   0 < A ∧ A < Real.pi ∧ 
   (1 / 2) * b * c * sinA = (15 * Real.sqrt 7) / 4) :=
by
  sorry

end triangle_area_l273_27300


namespace repetend_five_seventeen_l273_27352

noncomputable def repetend_of_fraction (n : ℕ) (d : ℕ) : ℕ := sorry

theorem repetend_five_seventeen : repetend_of_fraction 5 17 = 294117647058823529 := sorry

end repetend_five_seventeen_l273_27352


namespace math_problem_l273_27316

theorem math_problem
  (a b c x1 x2 : ℝ)
  (h1 : a > 0)
  (h2 : a^2 = 4 * b)
  (h3 : |x1 - x2| = 4)
  (h4 : x1 < x2) :
  (a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (c = 4) :=
by
  sorry

end math_problem_l273_27316


namespace problem1_problem2_l273_27374

variable (t : ℝ)

-- Problem 1
theorem problem1 (h : (4:ℝ) - 8 * t + 16 < 0) : t > 5 / 2 :=
sorry

-- Problem 2
theorem problem2 (hp: 4 - t > t - 2) (hq : t - 2 > 0) (hdisjoint : (∃ (p : Prop) (q : Prop), (p ∨ q) ∧ ¬(p ∧ q))):
  (2 < t ∧ t ≤ 5 / 2) ∨ (t ≥ 3) :=
sorry


end problem1_problem2_l273_27374


namespace prob_single_trial_l273_27320

theorem prob_single_trial (P : ℝ) : 
  (1 - (1 - P)^4) = 65 / 81 → P = 1 / 3 :=
by
  intro h
  sorry

end prob_single_trial_l273_27320


namespace total_presents_l273_27314

-- Definitions based on the problem conditions
def numChristmasPresents : ℕ := 60
def numBirthdayPresents : ℕ := numChristmasPresents / 2

-- Theorem statement
theorem total_presents : numChristmasPresents + numBirthdayPresents = 90 :=
by
  -- Proof is omitted
  sorry

end total_presents_l273_27314


namespace max_value_of_quadratic_l273_27336

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := -2 * x^2 + 8

-- State the problem formally in Lean
theorem max_value_of_quadratic : ∀ x : ℝ, quadratic x ≤ quadratic 0 :=
by
  -- Skipping the proof
  sorry

end max_value_of_quadratic_l273_27336


namespace proof_n_value_l273_27347

theorem proof_n_value (n : ℕ) (h : (9^n) * (9^n) * (9^n) * (9^n) * (9^n) = 81^5) : n = 2 :=
by
  sorry

end proof_n_value_l273_27347


namespace total_amount_Rs20_l273_27332

theorem total_amount_Rs20 (x y z : ℕ) 
(h1 : x + y + z = 130) 
(h2 : 95 * x + 45 * y + 20 * z = 7000) : 
∃ z : ℕ, (20 * z) = (7000 - 95 * x - 45 * y) / 20 := sorry

end total_amount_Rs20_l273_27332


namespace greatest_integer_e_minus_5_l273_27398

theorem greatest_integer_e_minus_5 (e : ℝ) (h : 2 < e ∧ e < 3) : ⌊e - 5⌋ = -3 :=
by
  sorry

end greatest_integer_e_minus_5_l273_27398


namespace cubics_product_equals_1_over_1003_l273_27382

theorem cubics_product_equals_1_over_1003
  (x_1 y_1 x_2 y_2 x_3 y_3 : ℝ)
  (h1 : x_1^3 - 3 * x_1 * y_1^2 = 2007)
  (h2 : y_1^3 - 3 * x_1^2 * y_1 = 2006)
  (h3 : x_2^3 - 3 * x_2 * y_2^2 = 2007)
  (h4 : y_2^3 - 3 * x_2^2 * y_2 = 2006)
  (h5 : x_3^3 - 3 * x_3 * y_3^2 = 2007)
  (h6 : y_3^3 - 3 * x_3^2 * y_3 = 2006) :
  (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = 1 / 1003 :=
by
  sorry

end cubics_product_equals_1_over_1003_l273_27382


namespace parallel_lines_condition_l273_27375

theorem parallel_lines_condition (m n : ℝ) :
  (∃x y, (m * x + y - n = 0) ∧ (x + m * y + 1 = 0)) →
  (m = 1 ∧ n ≠ -1) ∨ (m = -1 ∧ n ≠ 1) :=
by
  sorry

end parallel_lines_condition_l273_27375


namespace canal_depth_l273_27325

theorem canal_depth (A : ℝ) (w_top w_bottom : ℝ) (h : ℝ) 
    (hA : A = 10290) 
    (htop : w_top = 6) 
    (hbottom : w_bottom = 4) 
    (harea : A = 1 / 2 * (w_top + w_bottom) * h) : 
    h = 2058 :=
by
  -- here goes the proof steps
  sorry

end canal_depth_l273_27325


namespace tree_shadow_length_l273_27324

theorem tree_shadow_length (jane_shadow : ℝ) (jane_height : ℝ) (tree_height : ℝ) (tree_shadow : ℝ)
  (h₁ : jane_shadow = 0.5)
  (h₂ : jane_height = 1.5)
  (h₃ : tree_height = 30)
  (h₄ : jane_height / jane_shadow = tree_height / tree_shadow)
  : tree_shadow = 10 :=
by
  -- skipping the proof steps
  sorry

end tree_shadow_length_l273_27324


namespace a_perp_a_minus_b_l273_27302

noncomputable def a : ℝ × ℝ := (-2, 1)
noncomputable def b : ℝ × ℝ := (-1, 3)
noncomputable def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem a_perp_a_minus_b : (a.1 * a_minus_b.1 + a.2 * a_minus_b.2) = 0 := by
  sorry

end a_perp_a_minus_b_l273_27302


namespace students_diff_l273_27377

-- Define the conditions
def M : ℕ := 457
def B : ℕ := 394

-- Prove the final answer
theorem students_diff : M - B = 63 := by
  -- The proof is omitted here with a sorry placeholder
  sorry

end students_diff_l273_27377


namespace negation_statement_l273_27319

theorem negation_statement (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) : ¬ (x = 0 ∧ y = 0) :=
sorry

end negation_statement_l273_27319


namespace company_bought_oil_l273_27349

-- Define the conditions
def tank_capacity : ℕ := 32
def oil_in_tank : ℕ := 24

-- Formulate the proof problem
theorem company_bought_oil : oil_in_tank = 24 := by
  sorry

end company_bought_oil_l273_27349


namespace third_side_of_triangle_l273_27381

theorem third_side_of_triangle (a b : ℝ) (γ : ℝ) (x : ℝ) 
  (ha : a = 6) (hb : b = 2 * Real.sqrt 7) (hγ : γ = Real.pi / 3) :
  x = 2 ∨ x = 4 :=
by 
  sorry

end third_side_of_triangle_l273_27381


namespace count_two_digit_remainders_l273_27371

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l273_27371


namespace seating_arrangements_l273_27322

theorem seating_arrangements (n m k : Nat) (couples : Fin n -> Fin m -> Prop):
  let pairs : Nat := k
  let adjusted_pairs : Nat := pairs / 24
  adjusted_pairs = 5760 := by
  sorry

end seating_arrangements_l273_27322


namespace complement_U_A_inter_B_eq_l273_27344

open Set

-- Definitions
def U : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def A : Set ℤ := {-2, 0, 2, 4}
def B : Set ℤ := {-2, 0, 4, 6, 8}

-- Complement of A in U
def complement_U_A : Set ℤ := U \ A

-- Proof Problem
theorem complement_U_A_inter_B_eq : complement_U_A ∩ B = {6, 8} := by
  sorry

end complement_U_A_inter_B_eq_l273_27344


namespace engineers_crimson_meet_in_tournament_l273_27369

noncomputable def probability_engineers_crimson_meet : ℝ := 
  1 - Real.exp (-1)

theorem engineers_crimson_meet_in_tournament :
  (∃ (n : ℕ), n = 128) → 
  (∀ (i : ℕ), i < 128 → (∀ (j : ℕ), j < 128 → i ≠ j → ∃ (p : ℝ), p = probability_engineers_crimson_meet)) :=
sorry

end engineers_crimson_meet_in_tournament_l273_27369


namespace find_x_l273_27355

def binop (a b : ℤ) : ℤ := a * b + a + b + 2

theorem find_x :
  ∃ x : ℤ, binop x 3 = 1 ∧ x = -1 :=
by
  sorry

end find_x_l273_27355


namespace flower_beds_fraction_correct_l273_27317

noncomputable def flower_beds_fraction (yard_length : ℝ) (yard_width : ℝ) (trapezoid_parallel_side1 : ℝ) (trapezoid_parallel_side2 : ℝ) : ℝ :=
  let leg_length := (trapezoid_parallel_side2 - trapezoid_parallel_side1) / 2
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 2 * triangle_area
  let yard_area := yard_length * yard_width
  total_flower_bed_area / yard_area

theorem flower_beds_fraction_correct :
  flower_beds_fraction 30 5 20 30 = 1 / 6 :=
by
  sorry

end flower_beds_fraction_correct_l273_27317


namespace chord_length_l273_27328

noncomputable def circle_eq (θ : ℝ) : ℝ × ℝ :=
  (2 + 5 * Real.cos θ, 1 + 5 * Real.sin θ)

noncomputable def line_eq (t : ℝ) : ℝ × ℝ :=
  (-2 + 4 * t, -1 - 3 * t)

theorem chord_length :
  let center := (2, 1)
  let radius := 5
  let line_dist := |3 * center.1 + 4 * center.2 + 10| / Real.sqrt (3^2 + 4^2)
  let chord_len := 2 * Real.sqrt (radius^2 - line_dist^2)
  chord_len = 6 := 
by
  sorry

end chord_length_l273_27328


namespace range_of_a_l273_27303

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (1 + a) * x^2 + a * x + a < x^2 + 1) : a ≤ 0 := 
sorry

end range_of_a_l273_27303


namespace original_cost_of_remaining_shirt_l273_27363

theorem original_cost_of_remaining_shirt 
  (total_original_cost : ℝ) 
  (shirts_on_discount : ℕ) 
  (original_cost_per_discounted_shirt : ℝ) 
  (discount : ℝ) 
  (current_total_cost : ℝ) : 
  total_original_cost = 100 → 
  shirts_on_discount = 3 → 
  original_cost_per_discounted_shirt = 25 → 
  discount = 0.4 → 
  current_total_cost = 85 → 
  ∃ (remaining_shirts : ℕ) (original_cost_per_remaining_shirt : ℝ), 
    remaining_shirts = 2 ∧ 
    original_cost_per_remaining_shirt = 12.5 :=
by 
  sorry

end original_cost_of_remaining_shirt_l273_27363


namespace a5_a6_val_l273_27311

variable (a : ℕ → ℝ)
variable (r : ℝ)

axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom pos_seq (n : ℕ) : a n > 0

axiom a1_a2 : a 1 + a 2 = 1
axiom a3_a4 : a 3 + a 4 = 9

theorem a5_a6_val :
  a 5 + a 6 = 81 :=
by
  sorry

end a5_a6_val_l273_27311


namespace smallest_x_for_cubic_1890_l273_27333

theorem smallest_x_for_cubic_1890 (x : ℕ) (N : ℕ) (hx : 1890 * x = N ^ 3) : x = 4900 :=
sorry

end smallest_x_for_cubic_1890_l273_27333


namespace min_accommodation_cost_l273_27361

theorem min_accommodation_cost :
  ∃ (x y z : ℕ), x + y + z = 20 ∧ 3 * x + 2 * y + z = 50 ∧ 100 * 3 * x + 150 * 2 * y + 200 * z = 5500 :=
by
  sorry

end min_accommodation_cost_l273_27361


namespace dot_product_conditioned_l273_27340

variables (a b : ℝ×ℝ)

def condition1 : Prop := 2 • a + b = (1, 6)
def condition2 : Prop := a + 2 • b = (-4, 9)
def dot_product (u v : ℝ×ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product_conditioned :
  condition1 a b ∧ condition2 a b → dot_product a b = -2 :=
by
  sorry

end dot_product_conditioned_l273_27340


namespace minimum_final_percentage_to_pass_l273_27372

-- Conditions
def problem_sets : ℝ := 100
def midterm_worth : ℝ := 100
def final_worth : ℝ := 300
def perfect_problem_sets_score : ℝ := 100
def midterm1_score : ℝ := 0.60 * midterm_worth
def midterm2_score : ℝ := 0.70 * midterm_worth
def midterm3_score : ℝ := 0.80 * midterm_worth
def passing_percentage : ℝ := 0.70

-- Derived Values
def total_points_available : ℝ := problem_sets + 3 * midterm_worth + final_worth
def required_points_to_pass : ℝ := passing_percentage * total_points_available
def total_points_before_final : ℝ := perfect_problem_sets_score + midterm1_score + midterm2_score + midterm3_score
def points_needed_from_final : ℝ := required_points_to_pass - total_points_before_final

-- Proof Statement
theorem minimum_final_percentage_to_pass : 
  ∃ (final_score : ℝ), (final_score / final_worth * 100) ≥ 60 :=
by
  -- Calculations for proof
  let required_final_percentage := (points_needed_from_final / final_worth) * 100
  -- We need to show that the required percentage is at least 60%
  have : required_final_percentage = 60 := sorry
  exact Exists.intro 180 sorry

end minimum_final_percentage_to_pass_l273_27372


namespace jane_doe_investment_l273_27360

theorem jane_doe_investment (total_investment mutual_funds real_estate : ℝ)
  (h1 : total_investment = 250000)
  (h2 : real_estate = 3 * mutual_funds)
  (h3 : total_investment = mutual_funds + real_estate) :
  real_estate = 187500 :=
by
  sorry

end jane_doe_investment_l273_27360


namespace iron_balls_count_l273_27368

-- Conditions
def length_bar := 12  -- in cm
def width_bar := 8    -- in cm
def height_bar := 6   -- in cm
def num_bars := 10
def volume_iron_ball := 8  -- in cubic cm

-- Calculate the volume of one iron bar
def volume_one_bar := length_bar * width_bar * height_bar

-- Calculate the total volume of the ten iron bars
def total_volume := volume_one_bar * num_bars

-- Calculate the number of iron balls
def num_iron_balls := total_volume / volume_iron_ball

-- The proof statement
theorem iron_balls_count : num_iron_balls = 720 := by
  sorry

end iron_balls_count_l273_27368


namespace binom_25_7_l273_27387

theorem binom_25_7 :
  (Nat.choose 23 5 = 33649) →
  (Nat.choose 23 6 = 42504) →
  (Nat.choose 23 7 = 33649) →
  Nat.choose 25 7 = 152306 :=
by
  intros h1 h2 h3
  sorry

end binom_25_7_l273_27387


namespace math_equivalence_example_l273_27378

theorem math_equivalence_example :
  ((3.242^2 * (16 + 8)) / (100 - (3 * 25))) + (32 - 10)^2 = 494.09014144 := 
by
  sorry

end math_equivalence_example_l273_27378


namespace proof_problem_l273_27351

-- Define the conditions based on Classmate A and Classmate B's statements
def classmateA_statement (x y : ℝ) : Prop := 6 * x = 5 * y
def classmateB_statement (x y : ℝ) : Prop := x = 2 * y - 40

-- Define the system of equations derived from the statements
def system_of_equations (x y : ℝ) : Prop := (6 * x = 5 * y) ∧ (x = 2 * y - 40)

-- Proof goal: Prove the system of equations if classmate statements hold
theorem proof_problem (x y : ℝ) :
  classmateA_statement x y ∧ classmateB_statement x y → system_of_equations x y :=
by
  sorry

end proof_problem_l273_27351


namespace votes_candidate_X_l273_27338

theorem votes_candidate_X (X Y Z : ℕ) (h1 : X = (3 / 2 : ℚ) * Y) (h2 : Y = (3 / 5 : ℚ) * Z) (h3 : Z = 25000) : X = 22500 :=
by
  sorry

end votes_candidate_X_l273_27338


namespace candy_total_l273_27359

theorem candy_total (chocolate_boxes caramel_boxes mint_boxes berry_boxes : ℕ)
  (chocolate_pieces caramel_pieces mint_pieces berry_pieces : ℕ)
  (h_chocolate : chocolate_boxes = 7)
  (h_caramel : caramel_boxes = 3)
  (h_mint : mint_boxes = 5)
  (h_berry : berry_boxes = 4)
  (p_chocolate : chocolate_pieces = 8)
  (p_caramel : caramel_pieces = 8)
  (p_mint : mint_pieces = 10)
  (p_berry : berry_pieces = 12) :
  (chocolate_boxes * chocolate_pieces + caramel_boxes * caramel_pieces + mint_boxes * mint_pieces + berry_boxes * berry_pieces) = 178 := by
  sorry

end candy_total_l273_27359


namespace min_value_hyperbola_l273_27326

open Real 

theorem min_value_hyperbola :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1) → (3 * x^2 - 2 * y ≥ 143 / 12) ∧ 
                                          (∃ (y' : ℝ), y = y' ∧  3 * (2 + 2*y'^2)^2 - 2 * y' = 143 / 12) := 
by
  sorry

end min_value_hyperbola_l273_27326


namespace arithmetic_sequence_problem_l273_27362

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_problem 
  (h : is_arithmetic_sequence a)
  (h_cond : a 2 + 2 * a 6 + a 10 = 120) :
  a 3 + a 9 = 60 :=
sorry

end arithmetic_sequence_problem_l273_27362


namespace train_vs_airplane_passenger_capacity_l273_27370

theorem train_vs_airplane_passenger_capacity :
  (60 * 16) - (366 * 2) = 228 := by
sorry

end train_vs_airplane_passenger_capacity_l273_27370


namespace ratio_sum_pqr_uvw_l273_27339

theorem ratio_sum_pqr_uvw (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end ratio_sum_pqr_uvw_l273_27339


namespace ratio_A_B_l273_27367

noncomputable def A : ℝ := ∑' n : ℕ, if n % 4 = 0 then 0 else 1 / (n:ℝ) ^ 2
noncomputable def B : ℝ := ∑' k : ℕ, (-1)^(k+1) / (4 * (k:ℝ)) ^ 2

theorem ratio_A_B : A / B = 32 := by
  -- proof here
  sorry

end ratio_A_B_l273_27367


namespace sum_between_52_and_53_l273_27331

theorem sum_between_52_and_53 (x y : ℝ) (h1 : y = 4 * (⌊x⌋ : ℝ) + 2) (h2 : y = 5 * (⌊x - 3⌋ : ℝ) + 7) (h3 : ∀ n : ℤ, x ≠ n) :
  52 < x + y ∧ x + y < 53 := 
sorry

end sum_between_52_and_53_l273_27331


namespace identical_remainders_l273_27392

theorem identical_remainders (a : Fin 11 → Fin 11) (h_perm : ∀ n, ∃ m, a m = n) :
  ∃ (i j : Fin 11), i ≠ j ∧ (i * a i) % 11 = (j * a j) % 11 :=
by 
  sorry

end identical_remainders_l273_27392
