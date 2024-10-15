import Mathlib

namespace NUMINAMATH_GPT_gain_percent_l2208_220857

theorem gain_percent (cp sp : ℝ) (h_cp : cp = 900) (h_sp : sp = 1080) :
    ((sp - cp) / cp) * 100 = 20 :=
by
    sorry

end NUMINAMATH_GPT_gain_percent_l2208_220857


namespace NUMINAMATH_GPT_average_payment_l2208_220882

theorem average_payment (n m : ℕ) (p1 p2 : ℕ) (h1 : n = 20) (h2 : m = 45) (h3 : p1 = 410) (h4 : p2 = 475) :
  (20 * p1 + 45 * p2) / 65 = 455 :=
by
  sorry

end NUMINAMATH_GPT_average_payment_l2208_220882


namespace NUMINAMATH_GPT_transformed_roots_polynomial_l2208_220894

-- Given conditions
variables {a b c : ℝ}
variables (h : ∀ x, (x - a) * (x - b) * (x - c) = x^3 - 4 * x + 6)

-- Prove the equivalent polynomial with the transformed roots
theorem transformed_roots_polynomial :
  (∀ x, (x - (a - 3)) * (x - (b - 3)) * (x - (c - 3)) = x^3 + 9 * x^2 + 23 * x + 21) :=
sorry

end NUMINAMATH_GPT_transformed_roots_polynomial_l2208_220894


namespace NUMINAMATH_GPT_problem_l2208_220803

def f (x : ℝ) := 5 * x^3

theorem problem : f 2012 + f (-2012) = 0 := 
by
  sorry

end NUMINAMATH_GPT_problem_l2208_220803


namespace NUMINAMATH_GPT_yearly_return_of_1500_investment_is_27_percent_l2208_220892

-- Definitions based on conditions
def combined_yearly_return (x : ℝ) : Prop :=
  let investment1 := 500
  let investment2 := 1500
  let total_investment := investment1 + investment2
  let combined_return := 0.22 * total_investment
  let return_from_500 := 0.07 * investment1
  let return_from_1500 := combined_return - return_from_500
  x / 100 * investment2 = return_from_1500

-- Theorem statement to be proven
theorem yearly_return_of_1500_investment_is_27_percent : combined_yearly_return 27 :=
by sorry

end NUMINAMATH_GPT_yearly_return_of_1500_investment_is_27_percent_l2208_220892


namespace NUMINAMATH_GPT_log_eq_l2208_220811

theorem log_eq {a b : ℝ} (h₁ : a = Real.log 256 / Real.log 4) (h₂ : b = Real.log 27 / Real.log 3) : 
  a = (4 / 3) * b :=
by
  sorry

end NUMINAMATH_GPT_log_eq_l2208_220811


namespace NUMINAMATH_GPT_problem_statement_l2208_220843

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := x^2 + 2 * x + 5
def g (x : ℝ) : ℝ := 2 * x + 3

-- Statement to prove: f(g(3)) - g(f(3)) = 61
theorem problem_statement : f (g 3) - g (f 3) = 61 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2208_220843


namespace NUMINAMATH_GPT_chord_angle_measure_l2208_220875

theorem chord_angle_measure (AB_ratio : ℕ) (circ : ℝ) (h : AB_ratio = 1 + 5) : 
  ∃ θ : ℝ, θ = (1 / 6) * circ ∧ θ = 60 :=
by
  sorry

end NUMINAMATH_GPT_chord_angle_measure_l2208_220875


namespace NUMINAMATH_GPT_boat_upstream_time_l2208_220840

theorem boat_upstream_time (v t : ℝ) (d c : ℝ) 
  (h1 : d = 24) (h2 : c = 1) (h3 : 4 * (v + c) = d) 
  (h4 : d / (v - c) = t) : t = 6 :=
by
  sorry

end NUMINAMATH_GPT_boat_upstream_time_l2208_220840


namespace NUMINAMATH_GPT_total_cows_in_herd_l2208_220865

theorem total_cows_in_herd {n : ℚ} (h1 : 1/3 + 1/6 + 1/9 = 11/18) 
                           (h2 : (1 - 11/18) = 7/18) 
                           (h3 : 8 = (7/18) * n) : 
                           n = 144/7 :=
by sorry

end NUMINAMATH_GPT_total_cows_in_herd_l2208_220865


namespace NUMINAMATH_GPT_pascal_row_10_sum_l2208_220871

-- Define the function that represents the sum of Row n in Pascal's Triangle
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- State the theorem to be proven
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_pascal_row_10_sum_l2208_220871


namespace NUMINAMATH_GPT_nuts_eaten_condition_not_all_nuts_eaten_l2208_220897

/-- proof problem with conditions and questions --/

-- Let's define the initial setup and the conditions:

def anya_has_all_nuts (nuts : Nat) := nuts > 3

def distribution (a b c : ℕ → ℕ) (n : ℕ) := 
  ((a (n + 1) = b n + c n + (a n % 2)) ∧ 
   (b (n + 1) = a n / 2) ∧ 
   (c (n + 1) = a n / 2))

def nuts_eaten (a b c : ℕ → ℕ) (n : ℕ) := 
  (a n % 2 > 0 ∨ b n % 2 > 0 ∨ c n % 2 > 0)

-- Prove at least one nut will be eaten
theorem nuts_eaten_condition (a b c : ℕ → ℕ) (n : ℕ) :
  anya_has_all_nuts (a 0) → distribution a b c n → nuts_eaten a b c n :=
sorry

-- Prove not all nuts will be eaten
theorem not_all_nuts_eaten (a b c : ℕ → ℕ):
  anya_has_all_nuts (a 0) → distribution a b c n → 
  ¬∀ (n: ℕ), (a n = 0 ∧ b n = 0 ∧ c n = 0) :=
sorry

end NUMINAMATH_GPT_nuts_eaten_condition_not_all_nuts_eaten_l2208_220897


namespace NUMINAMATH_GPT_total_selling_price_l2208_220828

theorem total_selling_price (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ)
    (h1 : original_price = 80) (h2 : discount_rate = 0.25) (h3 : tax_rate = 0.10) :
  let discount_amt := original_price * discount_rate
  let sale_price := original_price - discount_amt
  let tax_amt := sale_price * tax_rate
  let total_price := sale_price + tax_amt
  total_price = 66 := by
  sorry

end NUMINAMATH_GPT_total_selling_price_l2208_220828


namespace NUMINAMATH_GPT_ages_of_children_l2208_220863

theorem ages_of_children : ∃ (a1 a2 a3 a4 : ℕ),
  a1 + a2 + a3 + a4 = 33 ∧
  (a1 - 3) + (a2 - 3) + (a3 - 3) + (a4 - 3) = 22 ∧
  (a1 - 7) + (a2 - 7) + (a3 - 7) + (a4 - 7) = 11 ∧
  (a1 - 13) + (a2 - 13) + (a3 - 13) + (a4 - 13) = 1 ∧
  a1 = 14 ∧ a2 = 11 ∧ a3 = 6 ∧ a4 = 2 :=
by
  sorry

end NUMINAMATH_GPT_ages_of_children_l2208_220863


namespace NUMINAMATH_GPT_no_conf_of_7_points_and_7_lines_l2208_220824

theorem no_conf_of_7_points_and_7_lines (points : Fin 7 → Prop) (lines : Fin 7 → (Fin 7 → Prop)) :
  (∀ p : Fin 7, ∃ l₁ l₂ l₃ : Fin 7, lines l₁ p ∧ lines l₂ p ∧ lines l₃ p ∧ l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
  (∀ l : Fin 7, ∃ p₁ p₂ p₃ : Fin 7, lines l p₁ ∧ lines l p₂ ∧ lines l p₃ ∧ p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃) 
  → false :=
by
  sorry

end NUMINAMATH_GPT_no_conf_of_7_points_and_7_lines_l2208_220824


namespace NUMINAMATH_GPT_solve_for_x_l2208_220861

-- Definitions of conditions
def sqrt_81_as_3sq : ℝ := (81 : ℝ)^(1/2)  -- sqrt(81)
def sqrt_81_as_3sq_simplified : ℝ := (3^4 : ℝ)^(1/2)  -- equivalent to (3^2) since 81 = 3^4

-- Theorem and goal statement
theorem solve_for_x :
  sqrt_81_as_3sq = sqrt_81_as_3sq_simplified →
  (3 : ℝ)^(3 * (2/3)) = sqrt_81_as_3sq :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_solve_for_x_l2208_220861


namespace NUMINAMATH_GPT_bacteria_colony_exceeds_500_l2208_220846

theorem bacteria_colony_exceeds_500 :
  ∃ (n : ℕ), (∀ m : ℕ, m < n → 4 * 3^m ≤ 500) ∧ 4 * 3^n > 500 :=
sorry

end NUMINAMATH_GPT_bacteria_colony_exceeds_500_l2208_220846


namespace NUMINAMATH_GPT_find_a_b_l2208_220835

noncomputable def f (a b x: ℝ) : ℝ := x / (a * x + b)

theorem find_a_b (a b : ℝ) (h₁ : a ≠ 0) (h₂ : f a b (-4) = 4) (h₃ : ∀ x, f a b x = f b a x) :
  a + b = 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_a_b_l2208_220835


namespace NUMINAMATH_GPT_question1_question2_l2208_220821

section problem1

variable (a b : ℝ)

theorem question1 (h1 : a = 1) (h2 : b = 2) : 
  ∀ x : ℝ, abs (2 * x + 1) + abs (3 * x - 2) ≤ 5 ↔ 
  (-4 / 5 ≤ x ∧ x ≤ 6 / 5) :=
sorry

end problem1

section problem2

theorem question2 :
  (∀ x : ℝ, abs (x - 1) + abs (x + 2) ≥ m^2 - 3 * m + 5) → 
  ∃ (m : ℝ), m ≤ 2 :=
sorry

end problem2

end NUMINAMATH_GPT_question1_question2_l2208_220821


namespace NUMINAMATH_GPT_polygon_sides_l2208_220895

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360)
  (h2 : n ≥ 3) : 
  n = 8 := 
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l2208_220895


namespace NUMINAMATH_GPT_two_pow_gt_n_square_plus_one_l2208_220852

theorem two_pow_gt_n_square_plus_one (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_two_pow_gt_n_square_plus_one_l2208_220852


namespace NUMINAMATH_GPT_value_of_a_value_of_sin_A_plus_pi_over_4_l2208_220896

section TriangleABC

variables {a b c A B : ℝ}
variables (h_b : b = 3) (h_c : c = 1) (h_A_eq_2B : A = 2 * B)

theorem value_of_a : a = 2 * Real.sqrt 3 :=
sorry

theorem value_of_sin_A_plus_pi_over_4 : Real.sin (A + π / 4) = (4 - Real.sqrt 2) / 6 :=
sorry

end TriangleABC

end NUMINAMATH_GPT_value_of_a_value_of_sin_A_plus_pi_over_4_l2208_220896


namespace NUMINAMATH_GPT_distance_circumcenter_centroid_inequality_l2208_220853

variable {R r d : ℝ}

theorem distance_circumcenter_centroid_inequality 
  (h1 : d = distance_circumcenter_to_centroid)
  (h2 : R = circumradius)
  (h3 : r = inradius) : d^2 ≤ R * (R - 2 * r) := 
sorry

end NUMINAMATH_GPT_distance_circumcenter_centroid_inequality_l2208_220853


namespace NUMINAMATH_GPT_polygon_diagonals_15_sides_l2208_220885

/-- Given a convex polygon with 15 sides, the number of diagonals is 90. -/
theorem polygon_diagonals_15_sides (n : ℕ) (h : n = 15) (convex : Prop) : 
  ∃ d : ℕ, d = 90 :=
by
    sorry

end NUMINAMATH_GPT_polygon_diagonals_15_sides_l2208_220885


namespace NUMINAMATH_GPT_batsman_avg_after_17th_inning_l2208_220854

def batsman_average : Prop :=
  ∃ (A : ℕ), 
    (A + 3 = (16 * A + 92) / 17) → 
    (A + 3 = 44)

theorem batsman_avg_after_17th_inning : batsman_average :=
by
  sorry

end NUMINAMATH_GPT_batsman_avg_after_17th_inning_l2208_220854


namespace NUMINAMATH_GPT_solve_triangle_l2208_220833

noncomputable def angle_A := 45
noncomputable def angle_B := 60
noncomputable def side_a := Real.sqrt 2

theorem solve_triangle {A B : ℕ} {a b : Real}
    (hA : A = angle_A)
    (hB : B = angle_B)
    (ha : a = side_a) :
    b = Real.sqrt 3 := 
by sorry

end NUMINAMATH_GPT_solve_triangle_l2208_220833


namespace NUMINAMATH_GPT_part_a_l2208_220855

theorem part_a (n : ℕ) (h_n : n ≥ 3) (x : Fin n → ℝ) (hx : ∀ i j : Fin n, i ≠ j → x i ≠ x j) (hx_pos : ∀ i : Fin n, 0 < x i) :
  ∃ (i j : Fin n), i ≠ j ∧ 0 < (x i - x j) / (1 + (x i) * (x j)) ∧ (x i - x j) / (1 + (x i) * (x j)) < Real.tan (π / (2 * (n - 1))) :=
by
  sorry

end NUMINAMATH_GPT_part_a_l2208_220855


namespace NUMINAMATH_GPT_range_of_a_l2208_220880

-- Given definition of the function
def f (x a : ℝ) := abs (x - a)

-- Statement of the problem
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < -1 → x₂ < -1 → f x₁ a ≤ f x₂ a) → a ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2208_220880


namespace NUMINAMATH_GPT_nth_equation_l2208_220829

theorem nth_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = (n - 1) * 10 + 1 :=
sorry

end NUMINAMATH_GPT_nth_equation_l2208_220829


namespace NUMINAMATH_GPT_notebooks_difference_l2208_220866

theorem notebooks_difference 
  (cost_mika : ℝ) (cost_leo : ℝ) (notebook_price : ℝ)
  (h_cost_mika : cost_mika = 2.40)
  (h_cost_leo : cost_leo = 3.20)
  (h_notebook_price : notebook_price > 0.10)
  (h_mika : ∃ m : ℕ, cost_mika = m * notebook_price)
  (h_leo : ∃ l : ℕ, cost_leo = l * notebook_price)
  : ∃ n : ℕ, (l - m = 4) :=
by
  sorry

end NUMINAMATH_GPT_notebooks_difference_l2208_220866


namespace NUMINAMATH_GPT_root_of_quadratic_l2208_220881

theorem root_of_quadratic (a : ℝ) (h : ∃ (x : ℝ), x = 0 ∧ x^2 + x + 2 * a - 1 = 0) : a = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_root_of_quadratic_l2208_220881


namespace NUMINAMATH_GPT_tshirt_cost_correct_l2208_220842

   -- Definitions of the conditions
   def initial_amount : ℕ := 91
   def cost_of_sweater : ℕ := 24
   def cost_of_shoes : ℕ := 11
   def remaining_amount : ℕ := 50

   -- Define the total cost of the T-shirt purchase
   noncomputable def cost_of_tshirt := 
     initial_amount - remaining_amount - cost_of_sweater - cost_of_shoes

   -- Proof statement that cost_of_tshirt = 6
   theorem tshirt_cost_correct : cost_of_tshirt = 6 := 
   by
     sorry
   
end NUMINAMATH_GPT_tshirt_cost_correct_l2208_220842


namespace NUMINAMATH_GPT_half_radius_of_circle_y_l2208_220845

theorem half_radius_of_circle_y (Cx Cy : ℝ) (r_x r_y : ℝ) 
  (h1 : Cx = 10 * π) 
  (h2 : Cx = 2 * π * r_x) 
  (h3 : π * r_x ^ 2 = π * r_y ^ 2) :
  (1 / 2) * r_y = 2.5 := 
by
-- sorry skips the proof
sorry

end NUMINAMATH_GPT_half_radius_of_circle_y_l2208_220845


namespace NUMINAMATH_GPT_sequence_solution_l2208_220802

theorem sequence_solution :
  ∃ (a : ℕ → ℕ), a 1 = 2 ∧ (∀ n : ℕ, n ∈ (Set.Icc 1 9) → 
    (n * a (n + 1) = (n + 1) * a n + 2)) ∧ a 10 = 38 :=
by
  sorry

end NUMINAMATH_GPT_sequence_solution_l2208_220802


namespace NUMINAMATH_GPT_train_passing_time_l2208_220834

theorem train_passing_time :
  ∀ (length : ℕ) (speed_kmph : ℕ),
    length = 120 →
    speed_kmph = 72 →
    ∃ (time : ℕ), time = 6 :=
by
  intro length speed_kmph hlength hspeed_kmph
  sorry

end NUMINAMATH_GPT_train_passing_time_l2208_220834


namespace NUMINAMATH_GPT_a_cubed_divisible_l2208_220860

theorem a_cubed_divisible {a : ℤ} (h1 : 60 ≤ a) (h2 : a^3 ∣ 216000) : a = 60 :=
by {
   sorry
}

end NUMINAMATH_GPT_a_cubed_divisible_l2208_220860


namespace NUMINAMATH_GPT_homework_time_decrease_l2208_220800

theorem homework_time_decrease (x : ℝ) :
  let T_initial := 100
  let T_final := 70
  T_initial * (1 - x) * (1 - x) = T_final :=
by
  sorry

end NUMINAMATH_GPT_homework_time_decrease_l2208_220800


namespace NUMINAMATH_GPT_simplify_vector_expression_l2208_220859

-- Definitions for vectors
variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Defining the vectors
variables (AB CA BD CD : A)

-- A definition using the head-to-tail addition of vectors.
def vector_add (v1 v2 : A) : A := v1 + v2

-- Statement to prove
theorem simplify_vector_expression :
  vector_add (vector_add AB CA) BD = CD :=
sorry

end NUMINAMATH_GPT_simplify_vector_expression_l2208_220859


namespace NUMINAMATH_GPT_area_of_triangle_l2208_220887

theorem area_of_triangle (A B C : ℝ) (a c : ℝ) (d B_value: ℝ) (h1 : A + B + C = 180) 
                         (h2 : A = B - d) (h3 : C = B + d) (h4 : a = 4) (h5 : c = 3)
                         (h6 : B = 60) :
  (1 / 2) * a * c * Real.sin (B * Real.pi / 180) = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l2208_220887


namespace NUMINAMATH_GPT_free_space_on_new_drive_l2208_220878

theorem free_space_on_new_drive
  (initial_free : ℝ) (initial_used : ℝ) (delete_size : ℝ) (new_files_size : ℝ) (new_drive_size : ℝ) :
  initial_free = 2.4 → initial_used = 12.6 → delete_size = 4.6 → new_files_size = 2 → new_drive_size = 20 →
  (new_drive_size - ((initial_used - delete_size) + new_files_size)) = 10 :=
by simp; sorry

end NUMINAMATH_GPT_free_space_on_new_drive_l2208_220878


namespace NUMINAMATH_GPT_find_finite_sets_l2208_220806

open Set

theorem find_finite_sets (X : Set ℝ) (h1 : X.Nonempty) (h2 : X.Finite)
  (h3 : ∀ x ∈ X, (x + |x|) ∈ X) :
  ∃ (F : Set ℝ), F.Finite ∧ (∀ x ∈ F, x < 0) ∧ X = insert 0 F :=
sorry

end NUMINAMATH_GPT_find_finite_sets_l2208_220806


namespace NUMINAMATH_GPT_red_ball_probability_l2208_220815

theorem red_ball_probability : 
  let red_A := 2
  let white_A := 3
  let red_B := 4
  let white_B := 1
  let total_A := red_A + white_A
  let total_B := red_B + white_B
  let prob_red_A := red_A / total_A
  let prob_white_A := white_A / total_A
  let prob_red_B_after_red_A := (red_B + 1) / (total_B + 1)
  let prob_red_B_after_white_A := red_B / (total_B + 1)
  (prob_red_A * prob_red_B_after_red_A + prob_white_A * prob_red_B_after_white_A) = 11 / 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_red_ball_probability_l2208_220815


namespace NUMINAMATH_GPT_no_real_value_x_l2208_220877

theorem no_real_value_x (R H : ℝ) (π : ℝ := Real.pi) :
  R = 10 → H = 5 →
  ¬∃ x : ℝ,  π * (R + x)^2 * H = π * R^2 * (H + x) ∧ x ≠ 0 :=
by
  intros hR hH; sorry

end NUMINAMATH_GPT_no_real_value_x_l2208_220877


namespace NUMINAMATH_GPT_intersection_with_y_axis_l2208_220807

theorem intersection_with_y_axis (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + x - 2) : f 0 = -2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_with_y_axis_l2208_220807


namespace NUMINAMATH_GPT_initial_boys_l2208_220847

-- Define the initial conditions
def initial_girls := 4
def final_children := 8
def boys_left := 3
def girls_entered := 2

-- Define the statement to be proved
theorem initial_boys : 
  ∃ (B : ℕ), (B - boys_left) + (initial_girls + girls_entered) = final_children ∧ B = 5 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_initial_boys_l2208_220847


namespace NUMINAMATH_GPT_problem_1_problem_2_l2208_220827

-- Problem I
theorem problem_1 (x : ℝ) (h : |x - 2| + |x - 1| < 4) : (-1/2 : ℝ) < x ∧ x < 7/2 :=
sorry

-- Problem II
theorem problem_2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + |x - 1| ≥ 2) : a ≤ -1 ∨ a ≥ 3 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2208_220827


namespace NUMINAMATH_GPT_quadratic_eq_one_solution_has_ordered_pair_l2208_220874

theorem quadratic_eq_one_solution_has_ordered_pair (a c : ℝ) 
  (h1 : a * c = 25) 
  (h2 : a + c = 17) 
  (h3 : a > c) : 
  (a, c) = (15.375, 1.625) :=
sorry

end NUMINAMATH_GPT_quadratic_eq_one_solution_has_ordered_pair_l2208_220874


namespace NUMINAMATH_GPT_total_pages_allowed_l2208_220891

noncomputable def words_total := 48000
noncomputable def words_per_page_large := 1800
noncomputable def words_per_page_small := 2400
noncomputable def pages_large := 4
noncomputable def total_pages : ℕ := 21

theorem total_pages_allowed :
  pages_large * words_per_page_large + (total_pages - pages_large) * words_per_page_small = words_total :=
  by sorry

end NUMINAMATH_GPT_total_pages_allowed_l2208_220891


namespace NUMINAMATH_GPT_income_on_first_day_l2208_220831

theorem income_on_first_day (income : ℕ → ℚ) (h1 : income 10 = 18)
  (h2 : ∀ n, income (n + 1) = 2 * income n) :
  income 1 = 0.03515625 :=
by
  sorry

end NUMINAMATH_GPT_income_on_first_day_l2208_220831


namespace NUMINAMATH_GPT_maximum_value_ratio_l2208_220851

theorem maximum_value_ratio (a b : ℝ) (h1 : a + b - 2 ≥ 0) (h2 : b - a - 1 ≤ 0) (h3 : a ≤ 1) :
  ∃ x, x = (a + 2 * b) / (2 * a + b) ∧ x ≤ 7/5 := sorry

end NUMINAMATH_GPT_maximum_value_ratio_l2208_220851


namespace NUMINAMATH_GPT_probability_white_balls_le_1_l2208_220830

-- Definitions and conditions
def total_balls : ℕ := 6
def red_balls : ℕ := 4
def white_balls : ℕ := 2
def selected_balls : ℕ := 3

-- Combinatorial computations
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculations based on the conditions
def total_combinations : ℕ := C total_balls selected_balls
def red_combinations : ℕ := C red_balls selected_balls
def white_combinations : ℕ := C white_balls 1 * C red_balls 2

-- Probability calculations
def P_xi_le_1 : ℚ :=
  (red_combinations / total_combinations : ℚ) +
  (white_combinations / total_combinations : ℚ)

-- Problem statement: Prove that the calculated probability is 4/5
theorem probability_white_balls_le_1 : P_xi_le_1 = 4 / 5 := 
  sorry

end NUMINAMATH_GPT_probability_white_balls_le_1_l2208_220830


namespace NUMINAMATH_GPT_find_a10_l2208_220868

variable {a : ℕ → ℝ}
variable (h1 : ∀ n m, a (n + 1) = a n + a m)
variable (h2 : a 6 + a 8 = 16)
variable (h3 : a 4 = 1)

theorem find_a10 : a 10 = 15 := by
  sorry

end NUMINAMATH_GPT_find_a10_l2208_220868


namespace NUMINAMATH_GPT_probability_intersection_interval_l2208_220883

theorem probability_intersection_interval (PA PB p : ℝ) (hPA : PA = 5 / 6) (hPB : PB = 3 / 4) :
  0 ≤ p ∧ p ≤ 3 / 4 :=
sorry

end NUMINAMATH_GPT_probability_intersection_interval_l2208_220883


namespace NUMINAMATH_GPT_value_of_each_gift_card_l2208_220873

theorem value_of_each_gift_card (students total_thank_you_cards with_gift_cards total_value : ℕ) 
  (h1 : students = 50)
  (h2 : total_thank_you_cards = 30 * students / 100)
  (h3 : with_gift_cards = total_thank_you_cards / 3)
  (h4 : total_value = 50) :
  total_value / with_gift_cards = 10 := by
  sorry

end NUMINAMATH_GPT_value_of_each_gift_card_l2208_220873


namespace NUMINAMATH_GPT_james_remaining_balance_l2208_220886

theorem james_remaining_balance 
  (initial_balance : ℕ := 500) 
  (ticket_1_2_cost : ℕ := 150)
  (ticket_3_cost : ℕ := ticket_1_2_cost / 3)
  (total_cost : ℕ := 2 * ticket_1_2_cost + ticket_3_cost)
  (roommate_share : ℕ := total_cost / 2) :
  initial_balance - roommate_share = 325 := 
by 
  -- By not considering the solution steps, we skip to the proof.
  sorry

end NUMINAMATH_GPT_james_remaining_balance_l2208_220886


namespace NUMINAMATH_GPT_total_fuel_proof_l2208_220801

def highway_consumption_60 : ℝ := 3 -- gallons per mile at 60 mph
def highway_consumption_70 : ℝ := 3.5 -- gallons per mile at 70 mph
def city_consumption_30 : ℝ := 5 -- gallons per mile at 30 mph
def city_consumption_15 : ℝ := 4.5 -- gallons per mile at 15 mph

def day1_highway_60_hours : ℝ := 2 -- hours driven at 60 mph on the highway
def day1_highway_70_hours : ℝ := 1 -- hours driven at 70 mph on the highway
def day1_city_30_hours : ℝ := 4 -- hours driven at 30 mph in the city

def day2_highway_70_hours : ℝ := 3 -- hours driven at 70 mph on the highway
def day2_city_15_hours : ℝ := 3 -- hours driven at 15 mph in the city
def day2_city_30_hours : ℝ := 1 -- hours driven at 30 mph in the city

def day3_highway_60_hours : ℝ := 1.5 -- hours driven at 60 mph on the highway
def day3_city_30_hours : ℝ := 3 -- hours driven at 30 mph in the city
def day3_city_15_hours : ℝ := 1 -- hours driven at 15 mph in the city

def total_fuel_consumption (c1 c2 c3 c4 : ℝ) (h1 h2 h3 h4 h5 h6 h7 h8 h9 : ℝ) :=
  (h1 * 60 * c1) + (h2 * 70 * c2) + (h3 * 30 * c3) + 
  (h4 * 70 * c2) + (h5 * 15 * c4) + (h6 * 30 * c3) +
  (h7 * 60 * c1) + (h8 * 30 * c3) + (h9 * 15 * c4)

theorem total_fuel_proof :
  total_fuel_consumption highway_consumption_60 highway_consumption_70 city_consumption_30 city_consumption_15
  day1_highway_60_hours day1_highway_70_hours day1_city_30_hours day2_highway_70_hours
  day2_city_15_hours day2_city_30_hours day3_highway_60_hours day3_city_30_hours day3_city_15_hours
  = 3080 := by
  sorry

end NUMINAMATH_GPT_total_fuel_proof_l2208_220801


namespace NUMINAMATH_GPT_units_digit_4659_pow_157_l2208_220869

theorem units_digit_4659_pow_157 : 
  (4659^157) % 10 = 9 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_4659_pow_157_l2208_220869


namespace NUMINAMATH_GPT_remainder_when_divided_by_84_l2208_220812

/-- 
  Given conditions:
  x ≡ 11 [MOD 14]
  Find the remainder when x is divided by 84, which equivalently means proving: 
  x ≡ 81 [MOD 84]
-/

theorem remainder_when_divided_by_84 (x : ℤ) (h1 : x % 14 = 11) : x % 84 = 81 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_84_l2208_220812


namespace NUMINAMATH_GPT_remainder_of_2n_div4_l2208_220872

theorem remainder_of_2n_div4 (n : ℕ) (h : ∃ k : ℕ, n = 4 * k + 3) : (2 * n) % 4 = 2 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_2n_div4_l2208_220872


namespace NUMINAMATH_GPT_union_complement_A_B_eq_l2208_220884

-- Define the universal set U, set A, and set B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The statement to be proved
theorem union_complement_A_B_eq {U A B : Set ℕ} (hU : U = {0, 1, 2, 3, 4}) 
  (hA : A = {0, 1, 2, 3}) (hB : B = {2, 3, 4}) :
  (complement_U_A) ∪ B = {2, 3, 4} := 
by
  sorry

end NUMINAMATH_GPT_union_complement_A_B_eq_l2208_220884


namespace NUMINAMATH_GPT_min_S_in_grid_l2208_220876

def valid_grid (grid : Fin 10 × Fin 10 → Fin 100) (S : ℕ) : Prop :=
  ∀ i j, 
    (i < 9 → grid (i, j) + grid (i + 1, j) ≤ S) ∧
    (j < 9 → grid (i, j) + grid (i, j + 1) ≤ S)

theorem min_S_in_grid : ∃ grid : Fin 10 × Fin 10 → Fin 100, ∃ S : ℕ, valid_grid grid S ∧ 
  (∀ (other_S : ℕ), valid_grid grid other_S → S ≤ other_S) ∧ S = 106 :=
sorry

end NUMINAMATH_GPT_min_S_in_grid_l2208_220876


namespace NUMINAMATH_GPT_symmetric_point_of_M_neg2_3_l2208_220805

-- Conditions
def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, -M.2)

-- Main statement
theorem symmetric_point_of_M_neg2_3 :
  symmetric_point (-2, 3) = (2, -3) := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_symmetric_point_of_M_neg2_3_l2208_220805


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l2208_220888

theorem parabola_focus_coordinates : 
  ∀ (x y : ℝ), y = 4 * x^2 → (0, y / 16) = (0, 1 / 16) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l2208_220888


namespace NUMINAMATH_GPT_value_of_a_l2208_220819

theorem value_of_a (a : ℤ) (h0 : 0 ≤ a) (h1 : a < 13) (h2 : 13 ∣ 12^20 + a) : a = 12 :=
by sorry

end NUMINAMATH_GPT_value_of_a_l2208_220819


namespace NUMINAMATH_GPT_constant_term_binomial_expansion_n_6_middle_term_coefficient_l2208_220867

open Nat

-- Define the binomial expansion term
def binomial_term (n : ℕ) (r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * (2 ^ r) * x^(2 * (n-r) - r)

-- (I) Prove the constant term of the binomial expansion when n = 6
theorem constant_term_binomial_expansion_n_6 :
  binomial_term 6 4 (1 : ℝ) = 240 := 
sorry

-- (II) Prove the coefficient of the middle term under given conditions
theorem middle_term_coefficient (n : ℕ) :
  (Nat.choose 8 2 = Nat.choose 8 6) →
  binomial_term 8 4 (1 : ℝ) = 1120 := 
sorry

end NUMINAMATH_GPT_constant_term_binomial_expansion_n_6_middle_term_coefficient_l2208_220867


namespace NUMINAMATH_GPT_line_contains_point_l2208_220817

theorem line_contains_point (k : ℝ) (x : ℝ) (y : ℝ) (H : 2 - 2 * k * x = -4 * y) : k = -1 ↔ (x = 3 ∧ y = -2) :=
by
  sorry

end NUMINAMATH_GPT_line_contains_point_l2208_220817


namespace NUMINAMATH_GPT_percentage_calculation_l2208_220820

theorem percentage_calculation (P : ℕ) (h1 : 0.25 * 16 = 4) 
    (h2 : P / 100 * 40 = 6) : P = 15 :=
by 
    sorry

end NUMINAMATH_GPT_percentage_calculation_l2208_220820


namespace NUMINAMATH_GPT_max_area_of_rectangular_playground_l2208_220813

theorem max_area_of_rectangular_playground (P : ℕ) (hP : P = 160) :
  (∃ (x y : ℕ), 2 * (x + y) = P ∧ x * y = 1600) :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_rectangular_playground_l2208_220813


namespace NUMINAMATH_GPT_Jim_paycheck_after_deductions_l2208_220809

def calculateRemainingPay (grossPay : ℕ) (retirementPercentage : ℕ) 
                          (taxDeduction : ℕ) : ℕ :=
  let retirementAmount := (grossPay * retirementPercentage) / 100
  let afterRetirement := grossPay - retirementAmount
  let afterTax := afterRetirement - taxDeduction
  afterTax

theorem Jim_paycheck_after_deductions :
  calculateRemainingPay 1120 25 100 = 740 := 
by
  sorry

end NUMINAMATH_GPT_Jim_paycheck_after_deductions_l2208_220809


namespace NUMINAMATH_GPT_carson_gold_stars_l2208_220816

theorem carson_gold_stars (yesterday_stars today_total_stars earned_today : ℕ) 
  (h1 : yesterday_stars = 6) 
  (h2 : today_total_stars = 15) 
  (h3 : earned_today = today_total_stars - yesterday_stars) 
  : earned_today = 9 :=
sorry

end NUMINAMATH_GPT_carson_gold_stars_l2208_220816


namespace NUMINAMATH_GPT_problem_21_divisor_l2208_220850

theorem problem_21_divisor 
    (k : ℕ) 
    (h1 : ∃ k, 21^k ∣ 435961) 
    (h2 : 21^k ∣ 435961) 
    : 7^k - k^7 = 1 := 
sorry

end NUMINAMATH_GPT_problem_21_divisor_l2208_220850


namespace NUMINAMATH_GPT_sphere_surface_area_from_box_l2208_220804

/--
Given a rectangular box with length = 2, width = 2, and height = 1,
prove that if all vertices of the rectangular box lie on the surface of a sphere,
then the surface area of the sphere is 9π.
--/
theorem sphere_surface_area_from_box :
  let length := 2
  let width := 2
  let height := 1
  ∃ (r : ℝ), ∀ (d := Real.sqrt (length^2 + width^2 + height^2)),
  r = d / 2 → 4 * Real.pi * r^2 = 9 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_from_box_l2208_220804


namespace NUMINAMATH_GPT_max_largest_integer_of_five_l2208_220838

theorem max_largest_integer_of_five (a b c d e : ℕ) (h1 : (a + b + c + d + e) = 500)
    (h2 : e > c ∧ c > d ∧ d > b ∧ b > a)
    (h3 : (a + b + d + e) / 4 = 105)
    (h4 : b + e = 150) : d ≤ 269 := 
sorry

end NUMINAMATH_GPT_max_largest_integer_of_five_l2208_220838


namespace NUMINAMATH_GPT_turtle_ran_while_rabbit_sleeping_l2208_220810

-- Define the constants and variables used in the problem
def total_distance : ℕ := 1000
def rabbit_speed_multiple : ℕ := 5
def rabbit_behind_distance : ℕ := 10

-- Define a function that represents the turtle's distance run while the rabbit is sleeping
def turtle_distance_while_rabbit_sleeping (total_distance : ℕ) (rabbit_speed_multiple : ℕ) (rabbit_behind_distance : ℕ) : ℕ :=
  total_distance - total_distance / (rabbit_speed_multiple + 1)

-- Prove that the turtle ran 802 meters while the rabbit was sleeping
theorem turtle_ran_while_rabbit_sleeping :
  turtle_distance_while_rabbit_sleeping total_distance rabbit_speed_multiple rabbit_behind_distance = 802 :=
by
  -- We reserve the proof and focus only on the statement
  sorry

end NUMINAMATH_GPT_turtle_ran_while_rabbit_sleeping_l2208_220810


namespace NUMINAMATH_GPT_sphere_surface_area_ratio_l2208_220890

axiom prism_has_circumscribed_sphere : Prop
axiom prism_has_inscribed_sphere : Prop

theorem sphere_surface_area_ratio 
  (h1 : prism_has_circumscribed_sphere)
  (h2 : prism_has_inscribed_sphere) : 
  ratio_surface_area_of_circumscribed_to_inscribed_sphere = 5 :=
sorry

end NUMINAMATH_GPT_sphere_surface_area_ratio_l2208_220890


namespace NUMINAMATH_GPT_sum_of_4_corners_is_200_l2208_220836

-- Define the conditions: 9x9 grid, numbers start from 10, and filled sequentially from left to right and top to bottom.
def topLeftCorner : ℕ := 10
def topRightCorner : ℕ := 18
def bottomLeftCorner : ℕ := 82
def bottomRightCorner : ℕ := 90

-- The main theorem stating that the sum of the numbers in the four corners is 200.
theorem sum_of_4_corners_is_200 :
  topLeftCorner + topRightCorner + bottomLeftCorner + bottomRightCorner = 200 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_sum_of_4_corners_is_200_l2208_220836


namespace NUMINAMATH_GPT_num_square_tiles_l2208_220839

theorem num_square_tiles (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 3 * a + 4 * b + 5 * c = 100) : b = 10 :=
  sorry

end NUMINAMATH_GPT_num_square_tiles_l2208_220839


namespace NUMINAMATH_GPT_erik_ate_more_pie_l2208_220879

theorem erik_ate_more_pie :
  let erik_pies := 0.67
  let frank_pies := 0.33
  erik_pies - frank_pies = 0.34 :=
by
  sorry

end NUMINAMATH_GPT_erik_ate_more_pie_l2208_220879


namespace NUMINAMATH_GPT_ratio_of_areas_l2208_220898

theorem ratio_of_areas 
  (t : ℝ) (q : ℝ)
  (h1 : t = 1 / 4)
  (h2 : q = 1 / 2) :
  q / t = 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_areas_l2208_220898


namespace NUMINAMATH_GPT_three_digit_number_count_correct_l2208_220844

def number_of_three_digit_numbers_with_repetition (digit_count : ℕ) (positions : ℕ) : ℕ :=
  let choices_for_repeated_digit := 5  -- 5 choices for repeated digit
  let ways_to_place_repeated_digit := 3 -- 3 ways to choose positions
  let choices_for_remaining_digit := 4 -- 4 choices for the remaining digit
  choices_for_repeated_digit * ways_to_place_repeated_digit * choices_for_remaining_digit

theorem three_digit_number_count_correct :
  number_of_three_digit_numbers_with_repetition 5 3 = 60 := 
sorry

end NUMINAMATH_GPT_three_digit_number_count_correct_l2208_220844


namespace NUMINAMATH_GPT_time_for_train_to_pass_platform_is_190_seconds_l2208_220899

def trainLength : ℕ := 1200
def timeToCrossTree : ℕ := 120
def platformLength : ℕ := 700
def speed (distance time : ℕ) := distance / time
def distanceToCrossPlatform (trainLength platformLength : ℕ) := trainLength + platformLength
def timeToCrossPlatform (distance speed : ℕ) := distance / speed

theorem time_for_train_to_pass_platform_is_190_seconds
  (trainLength timeToCrossTree platformLength : ℕ) (h1 : trainLength = 1200) (h2 : timeToCrossTree = 120) (h3 : platformLength = 700) :
  timeToCrossPlatform (distanceToCrossPlatform trainLength platformLength) (speed trainLength timeToCrossTree) = 190 := by
  sorry

end NUMINAMATH_GPT_time_for_train_to_pass_platform_is_190_seconds_l2208_220899


namespace NUMINAMATH_GPT_flowers_bloom_l2208_220848

theorem flowers_bloom (num_unicorns : ℕ) (flowers_per_step : ℕ) (distance_km : ℕ) (step_length_m : ℕ) 
  (h1 : num_unicorns = 6) (h2 : flowers_per_step = 4) (h3 : distance_km = 9) (h4 : step_length_m = 3) : 
  num_unicorns * (distance_km * 1000 / step_length_m) * flowers_per_step = 72000 :=
by
  sorry

end NUMINAMATH_GPT_flowers_bloom_l2208_220848


namespace NUMINAMATH_GPT_completion_days_l2208_220858

theorem completion_days (D : ℝ) :
  (1 / D + 1 / 9 = 1 / 3.2142857142857144) → D = 5 := by
  sorry

end NUMINAMATH_GPT_completion_days_l2208_220858


namespace NUMINAMATH_GPT_point_P_coordinates_l2208_220822

theorem point_P_coordinates :
  ∃ (P : ℝ × ℝ), P.1 > 0 ∧ P.2 < 0 ∧ abs P.2 = 3 ∧ abs P.1 = 8 ∧ P = (8, -3) :=
sorry

end NUMINAMATH_GPT_point_P_coordinates_l2208_220822


namespace NUMINAMATH_GPT_taylor_family_reunion_l2208_220862

theorem taylor_family_reunion :
  let number_of_kids := 45
  let number_of_adults := 123
  let number_of_tables := 14
  (number_of_kids + number_of_adults) / number_of_tables = 12 := by sorry

end NUMINAMATH_GPT_taylor_family_reunion_l2208_220862


namespace NUMINAMATH_GPT_sam_and_david_licks_l2208_220849

theorem sam_and_david_licks :
  let Dan_licks := 58
  let Michael_licks := 63
  let Lance_licks := 39
  let avg_licks := 60
  let total_people := 5
  let total_licks := avg_licks * total_people
  let total_licks_Dan_Michael_Lance := Dan_licks + Michael_licks + Lance_licks
  total_licks - total_licks_Dan_Michael_Lance = 140 := by
  sorry

end NUMINAMATH_GPT_sam_and_david_licks_l2208_220849


namespace NUMINAMATH_GPT_find_coefficients_l2208_220825

noncomputable def P (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^4 - 8 * a * x^3 + b * x^2 - 32 * c * x + 16 * c

theorem find_coefficients (a b c : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 x3 x4 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ P a b c x1 = 0 ∧ P a b c x2 = 0 ∧ P a b c x3 = 0 ∧ P a b c x4 = 0) →
  (b = 16 * a ∧ c = a) :=
by
  sorry

end NUMINAMATH_GPT_find_coefficients_l2208_220825


namespace NUMINAMATH_GPT_motorcycles_count_l2208_220870

/-- In a parking lot, there are cars and motorcycles. 
    Each car has 5 wheels (including one spare) and each motorcycle has 2 wheels. 
    There are 19 cars in the parking lot. 
    Altogether all vehicles have 117 wheels. 
    Prove that there are 11 motorcycles in the parking lot. -/
theorem motorcycles_count 
  (C M : ℕ)
  (hc : C = 19)
  (total_wheels : ℕ)
  (total_wheels_eq : total_wheels = 117)
  (car_wheels : ℕ)
  (car_wheels_eq : car_wheels = 5 * C)
  (bike_wheels : ℕ)
  (bike_wheels_eq : bike_wheels = total_wheels - car_wheels)
  (wheels_per_bike : ℕ)
  (wheels_per_bike_eq : wheels_per_bike = 2):
  M = bike_wheels / wheels_per_bike :=
by
  sorry

end NUMINAMATH_GPT_motorcycles_count_l2208_220870


namespace NUMINAMATH_GPT_intersection_point_of_lines_l2208_220832

noncomputable def line1 (x : ℝ) : ℝ := 3 * x - 4

noncomputable def line2 (x : ℝ) : ℝ := -1 / 3 * x + 10 / 3

def point : ℝ × ℝ := (4, 2)

theorem intersection_point_of_lines :
  ∃ (x y : ℝ), line1 x = y ∧ line2 x = y ∧ (x, y) = (2.2, 2.6) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l2208_220832


namespace NUMINAMATH_GPT_green_eyes_count_l2208_220893

noncomputable def people_count := 100
noncomputable def blue_eyes := 19
noncomputable def brown_eyes := people_count / 2
noncomputable def black_eyes := people_count / 4
noncomputable def green_eyes := people_count - (blue_eyes + brown_eyes + black_eyes)

theorem green_eyes_count : green_eyes = 6 := by
  sorry

end NUMINAMATH_GPT_green_eyes_count_l2208_220893


namespace NUMINAMATH_GPT_length_of_PS_l2208_220818

theorem length_of_PS
  (PT TR QT TS PQ : ℝ)
  (h1 : PT = 5)
  (h2 : TR = 7)
  (h3 : QT = 9)
  (h4 : TS = 4)
  (h5 : PQ = 7) :
  PS = Real.sqrt 66.33 := 
  sorry

end NUMINAMATH_GPT_length_of_PS_l2208_220818


namespace NUMINAMATH_GPT_find_m_l2208_220856

/-- Given vectors \(\overrightarrow{OA} = (1, m)\) and \(\overrightarrow{OB} = (m-1, 2)\), if 
\(\overrightarrow{OA} \perp \overrightarrow{AB}\), then \(m = \frac{1}{3}\). -/
theorem find_m (m : ℝ) (h : (1, m).1 * (m - 1 - 1, 2 - m).1 + (1, m).2 * (m - 1 - 1, 2 - m).2 = 0) :
  m = 1 / 3 :=
sorry

end NUMINAMATH_GPT_find_m_l2208_220856


namespace NUMINAMATH_GPT_M_inter_N_l2208_220814

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
noncomputable def N : Set ℝ := { x | ∃ y, y = Real.sqrt x + Real.log (1 - x) }

theorem M_inter_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_GPT_M_inter_N_l2208_220814


namespace NUMINAMATH_GPT_max_disjoint_regions_l2208_220889

theorem max_disjoint_regions {p : ℕ} (hp : Nat.Prime p) (hp_ge3 : 3 ≤ p) : ∃ R, R = 3 * p^2 - 3 * p + 1 :=
by
  sorry

end NUMINAMATH_GPT_max_disjoint_regions_l2208_220889


namespace NUMINAMATH_GPT_no_blonde_girls_added_l2208_220808

-- Initial number of girls
def total_girls : Nat := 80
def initial_blonde_girls : Nat := 30
def black_haired_girls : Nat := 50

-- Number of blonde girls added
def blonde_girls_added : Nat := total_girls - black_haired_girls - initial_blonde_girls

theorem no_blonde_girls_added : blonde_girls_added = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_blonde_girls_added_l2208_220808


namespace NUMINAMATH_GPT_marathon_end_time_l2208_220837

open Nat

def marathonStart := 15 * 60  -- 3:00 p.m. in minutes (15 hours * 60 minutes)
def marathonDuration := 780    -- Duration in minutes

theorem marathon_end_time : marathonStart + marathonDuration = 28 * 60 := -- 4:00 a.m. in minutes (28 hours * 60 minutes)
  sorry

end NUMINAMATH_GPT_marathon_end_time_l2208_220837


namespace NUMINAMATH_GPT_find_original_wage_l2208_220841

-- Defining the conditions
variables (W : ℝ) (W_new : ℝ) (h : W_new = 35) (h2 : W_new = 1.40 * W)

-- Statement that needs to be proved
theorem find_original_wage : W = 25 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_original_wage_l2208_220841


namespace NUMINAMATH_GPT_gcd_9009_14014_l2208_220826

-- Given conditions
def decompose_9009 : 9009 = 9 * 1001 := by sorry
def decompose_14014 : 14014 = 14 * 1001 := by sorry
def coprime_9_14 : Nat.gcd 9 14 = 1 := by sorry

-- Proof problem statement
theorem gcd_9009_14014 : Nat.gcd 9009 14014 = 1001 := by
  have h1 : 9009 = 9 * 1001 := decompose_9009
  have h2 : 14014 = 14 * 1001 := decompose_14014
  have h3 : Nat.gcd 9 14 = 1 := coprime_9_14
  sorry

end NUMINAMATH_GPT_gcd_9009_14014_l2208_220826


namespace NUMINAMATH_GPT_rectangle_width_l2208_220864

theorem rectangle_width (r l w : ℝ) (h_r : r = Real.sqrt 12) (h_l : l = 3 * Real.sqrt 2)
  (h_area_eq: Real.pi * r^2 = l * w) : w = 2 * Real.sqrt 2 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l2208_220864


namespace NUMINAMATH_GPT_correct_options_A_and_D_l2208_220823

noncomputable def problem_statement :=
  ∃ A B C D : Prop,
  (A = (∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0)) ∧ 
  (B = ∀ (a b c d : ℝ), a > b → c > d → ¬(a * c > b * d)) ∧
  (C = ∀ m : ℝ, ¬((∀ x : ℝ, x > 0 → (m^2 - m - 1) * x^(m^2 - 2 * m - 3) < 0) → (-1 < m ∧ m < 2))) ∧
  (D = ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ + x₂ = 3 - a ∧ x₁ * x₂ = a) → a < 0)

-- We need to prove that only A and D are true
theorem correct_options_A_and_D : problem_statement :=
  sorry

end NUMINAMATH_GPT_correct_options_A_and_D_l2208_220823
