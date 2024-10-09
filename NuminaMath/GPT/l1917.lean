import Mathlib

namespace solve_linear_system_l1917_191700

theorem solve_linear_system :
  ∃ x y z : ℝ, 
    (2 * x + y + z = -1) ∧ 
    (3 * y - z = -1) ∧ 
    (3 * x + 2 * y + 3 * z = -5) ∧ 
    (x = 1) ∧ 
    (y = -1) ∧ 
    (z = -2) :=
by
  sorry

end solve_linear_system_l1917_191700


namespace expression_I_evaluation_expression_II_evaluation_l1917_191705

theorem expression_I_evaluation :
  ( (3 / 2) ^ (-2: ℤ) - (49 / 81) ^ (0.5: ℝ) + (0.008: ℝ) ^ (-2 / 3: ℝ) * (2 / 25) ) = (5 / 3) := 
by
  sorry

theorem expression_II_evaluation :
  ( (Real.logb 2 2) ^ 2 + (Real.logb 10 20) * (Real.logb 10 5) ) = (17 / 9) := 
by
  sorry

end expression_I_evaluation_expression_II_evaluation_l1917_191705


namespace valid_tree_arrangements_l1917_191719

-- Define the types of trees
inductive TreeType
| Birch
| Oak

-- Define the condition that each tree must be adjacent to a tree of the other type
def isValidArrangement (trees : List TreeType) : Prop :=
  ∀ (i : ℕ), i < trees.length - 1 → trees.nthLe i sorry ≠ trees.nthLe (i + 1) sorry

-- Define the main problem
theorem valid_tree_arrangements : ∃ (ways : Nat), ways = 16 ∧
  ∃ (arrangements : List (List TreeType)), arrangements.length = ways ∧
    ∀ arrangement ∈ arrangements, arrangement.length = 7 ∧ isValidArrangement arrangement :=
sorry

end valid_tree_arrangements_l1917_191719


namespace window_width_correct_l1917_191786

def total_width_window (x : ℝ) : ℝ :=
  let pane_width := 4 * x
  let num_panes_per_row := 4
  let num_borders := 5
  num_panes_per_row * pane_width + num_borders * 3

theorem window_width_correct (x : ℝ) :
  total_width_window x = 16 * x + 15 := sorry

end window_width_correct_l1917_191786


namespace cost_price_of_article_l1917_191766

variable (C : ℝ)
variable (h1 : (0.18 * C - 0.09 * C = 72))

theorem cost_price_of_article : C = 800 :=
by
  sorry

end cost_price_of_article_l1917_191766


namespace remainder_when_divided_by_x_minus_2_l1917_191748

def p (x : ℕ) : ℕ := x^5 - 2 * x^3 + 4 * x + 5

theorem remainder_when_divided_by_x_minus_2 : p 2 = 29 := 
by {
  sorry
}

end remainder_when_divided_by_x_minus_2_l1917_191748


namespace unique_prime_with_conditions_l1917_191750

theorem unique_prime_with_conditions (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (p + 2)) (hp4 : Nat.Prime (p + 4)) : p = 3 :=
by
  sorry

end unique_prime_with_conditions_l1917_191750


namespace arithmetic_sequence_general_term_and_sum_max_l1917_191796

-- Definitions and conditions
def a1 : ℤ := 4
def d : ℤ := -2
def a (n : ℕ) : ℤ := a1 + (n - 1) * d
def Sn (n : ℕ) : ℤ := n * (a1 + (a n)) / 2

-- Prove the general term formula and maximum value
theorem arithmetic_sequence_general_term_and_sum_max :
  (∀ n, a n = -2 * n + 6) ∧ (∃ n, Sn n = 6) :=
by
  sorry

end arithmetic_sequence_general_term_and_sum_max_l1917_191796


namespace max_value_g_f_less_than_e_x_div_x_sq_l1917_191785

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem max_value_g : ∃ x, x = 3 ∧ g x = 2 * Real.log 2 - 7 / 4 := by
  sorry

theorem f_less_than_e_x_div_x_sq (x : ℝ) (hx : x > 0) : f x < (Real.exp x - 1) / x ^ 2 := by
  sorry

end max_value_g_f_less_than_e_x_div_x_sq_l1917_191785


namespace remainder_of_h_x10_div_h_x_l1917_191798

noncomputable def h (x : ℤ) : ℤ := x^5 - x^4 + x^3 - x^2 + x - 1

theorem remainder_of_h_x10_div_h_x (x : ℤ) : (h (x ^ 10)) % (h x) = -6 :=
by
  sorry

end remainder_of_h_x10_div_h_x_l1917_191798


namespace total_weight_is_correct_l1917_191736

-- Define the weight of apples
def weight_of_apples : ℕ := 240

-- Define the multiplier for pears
def pears_multiplier : ℕ := 3

-- Define the weight of pears
def weight_of_pears := pears_multiplier * weight_of_apples

-- Define the total weight of apples and pears
def total_weight : ℕ := weight_of_apples + weight_of_pears

-- The theorem that states the total weight calculation
theorem total_weight_is_correct : total_weight = 960 := by
  sorry

end total_weight_is_correct_l1917_191736


namespace no_solution_15x_29y_43z_t2_l1917_191716

theorem no_solution_15x_29y_43z_t2 (x y z t : ℕ) : ¬ (15 ^ x + 29 ^ y + 43 ^ z = t ^ 2) :=
by {
  -- We'll insert the necessary conditions for the proof here
  sorry -- proof goes here
}

end no_solution_15x_29y_43z_t2_l1917_191716


namespace triple_apply_l1917_191781

def f (x : ℝ) : ℝ := 5 * x - 4

theorem triple_apply : f (f (f 2)) = 126 :=
by
  rw [f, f, f]
  sorry

end triple_apply_l1917_191781


namespace ratio_of_hypothetical_to_actual_children_l1917_191757

theorem ratio_of_hypothetical_to_actual_children (C H : ℕ) 
  (h1 : H = 16 * 8)
  (h2 : ∀ N : ℕ, N = C / 8 → C * N = 512) 
  (h3 : C^2 = 512 * 8) : H / C = 2 := 
by 
  sorry

end ratio_of_hypothetical_to_actual_children_l1917_191757


namespace solve_problem_l1917_191741

-- Define the variables and conditions
def problem_statement : Prop :=
  ∃ x : ℕ, 865 * 48 = 240 * x ∧ x = 173

-- Statement to prove
theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l1917_191741


namespace books_sold_l1917_191758

theorem books_sold (initial_books sold_books remaining_books : ℕ) 
  (h_initial : initial_books = 242) 
  (h_remaining : remaining_books = 105)
  (h_relation : sold_books = initial_books - remaining_books) :
  sold_books = 137 := 
by
  sorry

end books_sold_l1917_191758


namespace expression_value_l1917_191731

theorem expression_value : (2^2003 + 5^2004)^2 - (2^2003 - 5^2004)^2 = 40 * 10^2003 := 
by
  sorry

end expression_value_l1917_191731


namespace solve_absolute_inequality_l1917_191724

theorem solve_absolute_inequality (x : ℝ) : |x - 1| - |x - 2| > 1 / 2 ↔ x > 7 / 4 :=
by sorry

end solve_absolute_inequality_l1917_191724


namespace locus_of_circle_center_l1917_191754

theorem locus_of_circle_center (x y : ℝ) : 
    (exists C : ℝ × ℝ, (C.1, C.2) = (x,y)) ∧ 
    ((x - 0)^2 + (y - 3)^2 = r^2) ∧ 
    (y + 3 = 0) → x^2 = 12 * y :=
sorry

end locus_of_circle_center_l1917_191754


namespace B_is_subset_of_A_l1917_191764
open Set

def A := {x : ℤ | ∃ n : ℤ, x = 2 * n}
def B := {y : ℤ | ∃ k : ℤ, y = 4 * k}

theorem B_is_subset_of_A : B ⊆ A :=
by sorry

end B_is_subset_of_A_l1917_191764


namespace train_platform_length_l1917_191707

theorem train_platform_length (time_platform : ℝ) (time_man : ℝ) (speed_km_per_hr : ℝ) :
  time_platform = 34 ∧ time_man = 20 ∧ speed_km_per_hr = 54 →
  let speed_m_per_s := speed_km_per_hr * (5/18)
  let length_train := speed_m_per_s * time_man
  let time_to_cover_platform := time_platform - time_man
  let length_platform := speed_m_per_s * time_to_cover_platform
  length_platform = 210 := 
by {
  sorry
}

end train_platform_length_l1917_191707


namespace no_consecutive_integer_sum_to_36_l1917_191761

theorem no_consecutive_integer_sum_to_36 :
  ∀ (a n : ℕ), n ≥ 2 → (n * a + n * (n - 1) / 2) = 36 → false :=
by
  sorry

end no_consecutive_integer_sum_to_36_l1917_191761


namespace composite_for_positive_integers_l1917_191738

def is_composite (n : ℤ) : Prop :=
  ∃ a b : ℤ, 1 < a ∧ 1 < b ∧ n = a * b

theorem composite_for_positive_integers (n : ℕ) (h_pos : 1 < n) :
  is_composite (3^(2*n+1) - 2^(2*n+1) - 6*n) := 
sorry

end composite_for_positive_integers_l1917_191738


namespace solve_for_x_l1917_191710

theorem solve_for_x (i x : ℂ) (h : i^2 = -1) (eq : 3 - 2 * i * x = 5 + 4 * i * x) : x = i / 3 := 
by
  sorry

end solve_for_x_l1917_191710


namespace angleC_equals_40_of_angleA_40_l1917_191701

-- Define an arbitrary quadrilateral type and its angle A and angle C
structure Quadrilateral :=
  (angleA : ℝ)  -- angleA is in degrees
  (angleC : ℝ)  -- angleC is in degrees

-- Given condition in the problem
def quadrilateral_with_A_40 : Quadrilateral :=
  { angleA := 40, angleC := 0 } -- Initialize angleC as a placeholder

-- Theorem stating the problem's claim
theorem angleC_equals_40_of_angleA_40 :
  quadrilateral_with_A_40.angleA = 40 → quadrilateral_with_A_40.angleC = 40 :=
by
  sorry  -- Proof is omitted for brevity

end angleC_equals_40_of_angleA_40_l1917_191701


namespace triangle_base_length_l1917_191745

/-
Theorem: Given a triangle with height 5.8 meters and area 24.36 square meters,
the length of the base is 8.4 meters.
-/

theorem triangle_base_length (h : ℝ) (A : ℝ) (b : ℝ) :
  h = 5.8 ∧ A = 24.36 ∧ A = (b * h) / 2 → b = 8.4 :=
by
  sorry

end triangle_base_length_l1917_191745


namespace weight_loss_comparison_l1917_191747

-- Define the conditions
def weight_loss_Barbi : ℝ := 1.5 * 24
def weight_loss_Luca : ℝ := 9 * 15
def weight_loss_Kim : ℝ := (2 * 12) + (3 * 60)

-- Define the combined weight loss of Luca and Kim
def combined_weight_loss_Luca_Kim : ℝ := weight_loss_Luca + weight_loss_Kim

-- Define the difference in weight loss between Luca and Kim combined and Barbi
def weight_loss_difference : ℝ := combined_weight_loss_Luca_Kim - weight_loss_Barbi

-- State the theorem to be proved
theorem weight_loss_comparison : weight_loss_difference = 303 := by
  sorry

end weight_loss_comparison_l1917_191747


namespace polynomial_factorization_l1917_191727

theorem polynomial_factorization : ∃ q : Polynomial ℝ, (Polynomial.X ^ 4 - 6 * Polynomial.X ^ 2 + 25) = (Polynomial.X ^ 2 + 5) * q :=
by
  sorry

end polynomial_factorization_l1917_191727


namespace find_fourth_mark_l1917_191714

-- Definitions of conditions
def average_of_four (a b c d : ℕ) : Prop :=
  (a + b + c + d) / 4 = 60

def known_marks (a b c : ℕ) : Prop :=
  a = 30 ∧ b = 55 ∧ c = 65

-- Theorem statement
theorem find_fourth_mark {d : ℕ} (h_avg : average_of_four 30 55 65 d) (h_known : known_marks 30 55 65) : d = 90 := 
by 
  sorry

end find_fourth_mark_l1917_191714


namespace xy_value_l1917_191760

theorem xy_value (x y : ℝ) (h : |x - 1| + (x + y)^2 = 0) : x * y = -1 := 
by
  sorry

end xy_value_l1917_191760


namespace roots_quartic_ab_plus_a_plus_b_l1917_191708

theorem roots_quartic_ab_plus_a_plus_b (a b : ℝ) 
  (h1 : Polynomial.eval a (Polynomial.C 1 + Polynomial.C (-4) * Polynomial.X + Polynomial.C (-6) * (Polynomial.X ^ 2) + Polynomial.C 1 * (Polynomial.X ^ 4)) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1 + Polynomial.C (-4) * Polynomial.X + Polynomial.C (-6) * (Polynomial.X ^ 2) + Polynomial.C 1 * (Polynomial.X ^ 4)) = 0) :
  a * b + a + b = -1 := 
sorry

end roots_quartic_ab_plus_a_plus_b_l1917_191708


namespace marty_combinations_l1917_191779

theorem marty_combinations : 
  let C := 5
  let P := 4
  C * P = 20 :=
by
  sorry

end marty_combinations_l1917_191779


namespace gcd_3pow600_minus_1_3pow612_minus_1_l1917_191777

theorem gcd_3pow600_minus_1_3pow612_minus_1 :
  Nat.gcd (3^600 - 1) (3^612 - 1) = 531440 :=
by
  sorry

end gcd_3pow600_minus_1_3pow612_minus_1_l1917_191777


namespace leak_time_to_empty_l1917_191774

def pump_rate : ℝ := 0.1 -- P = 0.1 tanks/hour
def effective_rate : ℝ := 0.05 -- P - L = 0.05 tanks/hour

theorem leak_time_to_empty (P L : ℝ) (hp : P = pump_rate) (he : P - L = effective_rate) :
  1 / L = 20 := by
  sorry

end leak_time_to_empty_l1917_191774


namespace factorize_expr1_factorize_expr2_l1917_191791

variable (x y a b : ℝ)

theorem factorize_expr1 : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := sorry

theorem factorize_expr2 : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := sorry

end factorize_expr1_factorize_expr2_l1917_191791


namespace trajectory_ellipse_l1917_191744

/--
Given two fixed points A(-2,0) and B(2,0) in the Cartesian coordinate system, 
if a moving point P satisfies |PA| + |PB| = 6, 
then prove that the equation of the trajectory for point P is (x^2) / 9 + (y^2) / 5 = 1.
-/
theorem trajectory_ellipse (P : ℝ × ℝ)
  (A B : ℝ × ℝ)
  (hA : A = (-2, 0))
  (hB : B = (2, 0))
  (hPA_PB : dist P A + dist P B = 6) :
  (P.1 ^ 2) / 9 + (P.2 ^ 2) / 5 = 1 :=
sorry

end trajectory_ellipse_l1917_191744


namespace problem_solution_l1917_191723

theorem problem_solution
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d :=
sorry

end problem_solution_l1917_191723


namespace max_A_plus_B_l1917_191769

theorem max_A_plus_B:
  ∃ A B C D : ℕ,
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  A + B + C + D = 17 ∧ ∃ k : ℕ, C + D ≠ 0 ∧ A + B = k * (C + D) ∧
  A + B = 16 :=
by sorry

end max_A_plus_B_l1917_191769


namespace fourth_vertex_of_parallelogram_l1917_191703

structure Point where
  x : ℝ
  y : ℝ

def Q := Point.mk 1 (-1)
def R := Point.mk (-1) 0
def S := Point.mk 0 1
def V := Point.mk (-2) 2

theorem fourth_vertex_of_parallelogram (Q R S V : Point) :
  Q = ⟨1, -1⟩ ∧ R = ⟨-1, 0⟩ ∧ S = ⟨0, 1⟩ → V = ⟨-2, 2⟩ := by 
  sorry

end fourth_vertex_of_parallelogram_l1917_191703


namespace find_initial_books_each_l1917_191715

variable (x : ℝ)
variable (sandy_books : ℝ := x)
variable (tim_books : ℝ := 2 * x + 33)
variable (benny_books : ℝ := 3 * x - 24)
variable (total_books : ℝ := 100)

theorem find_initial_books_each :
  sandy_books + tim_books + benny_books = total_books → x = 91 / 6 := by
  sorry

end find_initial_books_each_l1917_191715


namespace problem_1_problem_2_l1917_191772

noncomputable def f (x : ℝ) : ℝ := x^3 + 1 / (x + 1)

theorem problem_1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x ≥ 1 - x + x^2 := 
sorry

theorem problem_2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (h1 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≥ 1 - x + x^2) : f x > 3 / 4 := 
sorry

end problem_1_problem_2_l1917_191772


namespace find_q_l1917_191732

theorem find_q (q : ℕ) (h1 : 32 = 2^5) (h2 : 32^5 = 2^q) : q = 25 := by
  sorry

end find_q_l1917_191732


namespace eq_solution_set_l1917_191728

theorem eq_solution_set (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^(a^a)) :
  (a, b) = (1, 1) ∨ (a, b) = (2, 16) ∨ (a, b) = (3, 27) :=
by
  sorry

end eq_solution_set_l1917_191728


namespace greatest_median_l1917_191735

theorem greatest_median (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t) (h5 : (k + m + r + s + t) = 80) (h6 : t = 42) : r = 17 :=
by
  sorry

end greatest_median_l1917_191735


namespace numWaysToPaintDoors_l1917_191702

-- Define the number of doors and choices per door
def numDoors : ℕ := 3
def numChoicesPerDoor : ℕ := 2

-- Theorem statement that we want to prove
theorem numWaysToPaintDoors : numChoicesPerDoor ^ numDoors = 8 := by
  sorry

end numWaysToPaintDoors_l1917_191702


namespace solve_eq_l1917_191713

theorem solve_eq (x a b : ℝ) (h₁ : x^2 + 10 * x = 34) (h₂ : a = 59) (h₃ : b = 5) :
  a + b = 64 :=
by {
  -- insert proof here, eventually leading to a + b = 64
  sorry
}

end solve_eq_l1917_191713


namespace sum_of_8th_and_10th_terms_arithmetic_sequence_l1917_191721

theorem sum_of_8th_and_10th_terms_arithmetic_sequence (a d : ℤ)
  (h1 : a + 3 * d = 25) (h2 : a + 5 * d = 61) :
  (a + 7 * d) + (a + 9 * d) = 230 := 
sorry

end sum_of_8th_and_10th_terms_arithmetic_sequence_l1917_191721


namespace additional_pencils_l1917_191712

theorem additional_pencils (original_pencils new_pencils per_container distributed_pencils : ℕ)
  (h1 : original_pencils = 150)
  (h2 : per_container = 5)
  (h3 : distributed_pencils = 36)
  (h4 : new_pencils = distributed_pencils * per_container) :
  (new_pencils - original_pencils) = 30 :=
by
  -- Proof will go here
  sorry

end additional_pencils_l1917_191712


namespace minimum_number_of_rooks_l1917_191773

theorem minimum_number_of_rooks (n : ℕ) : 
  ∃ (num_rooks : ℕ), (∀ (cells_colored : ℕ), cells_colored = n^2 → num_rooks = n) :=
by sorry

end minimum_number_of_rooks_l1917_191773


namespace stmt_A_stmt_B_stmt_C_stmt_D_l1917_191706
open Real

def x_and_y_conditions := ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 3

theorem stmt_A : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (2 * (x * x + y * y) = 4) :=
by sorry

theorem stmt_B : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (x * y = 9 / 8) :=
by sorry

theorem stmt_C : x_and_y_conditions → ¬ (∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (sqrt (x) + sqrt (2 * y) = sqrt 6)) :=
by sorry

theorem stmt_D : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (x^2 + 4 * y^2 = 9 / 2) :=
by sorry

end stmt_A_stmt_B_stmt_C_stmt_D_l1917_191706


namespace sum_due_l1917_191752

theorem sum_due (BD TD S : ℝ) (hBD : BD = 18) (hTD : TD = 15) (hRel : BD = TD + (TD^2 / S)) : S = 75 :=
by
  sorry

end sum_due_l1917_191752


namespace special_numbers_count_l1917_191759

-- Define conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_zero (n : ℕ) : Prop := n % 10 = 0
def divisible_by_30 (n : ℕ) : Prop := n % 30 = 0

-- Define the count of numbers with the specified conditions
noncomputable def count_special_numbers : ℕ :=
  (9990 - 1020) / 30 + 1

-- The proof problem
theorem special_numbers_count : count_special_numbers = 300 := sorry

end special_numbers_count_l1917_191759


namespace remainder_1234_mul_5678_mod_1000_l1917_191733

theorem remainder_1234_mul_5678_mod_1000 :
  (1234 * 5678) % 1000 = 652 := by
  sorry

end remainder_1234_mul_5678_mod_1000_l1917_191733


namespace gain_percent_l1917_191767

theorem gain_percent (C S : ℝ) (h : 50 * C = 15 * S) :
  (S > C) →
  ((S - C) / C * 100) = 233.33 := 
sorry

end gain_percent_l1917_191767


namespace pyramid_circumscribed_sphere_volume_l1917_191775

theorem pyramid_circumscribed_sphere_volume 
  (PA ABCD : ℝ) 
  (square_base : Prop)
  (perpendicular_PA_base : Prop)
  (AB : ℝ)
  (PA_val : PA = 1)
  (AB_val : AB = 2) 
  : (∃ (volume : ℝ), volume = (4/3) * π * (3/2)^3 ∧ volume = 9 * π / 2) := 
by
  -- Provided the conditions, we need to prove that the volume of the circumscribed sphere is 9π/2
  sorry

end pyramid_circumscribed_sphere_volume_l1917_191775


namespace proof_by_contradiction_conditions_l1917_191737

theorem proof_by_contradiction_conditions :
  ∀ (P Q : Prop),
    (∃ R : Prop, (R = ¬Q) ∧ (P → R) ∧ (R → P) ∧ (∀ T : Prop, (T = Q) → false)) →
    (∃ S : Prop, (S = ¬Q) ∧ P ∧ (∃ U : Prop, U) ∧ ¬Q) :=
by
  sorry

end proof_by_contradiction_conditions_l1917_191737


namespace complex_div_imag_unit_l1917_191730

theorem complex_div_imag_unit (i : ℂ) (h : i^2 = -1) : (1 + i) / (1 - i) = i :=
sorry

end complex_div_imag_unit_l1917_191730


namespace max_a_value_l1917_191725

theorem max_a_value (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : b + d = 200) : a ≤ 449 :=
by sorry

end max_a_value_l1917_191725


namespace smaller_number_l1917_191790

theorem smaller_number (x y : ℝ) (h1 : y - x = (1 / 3) * y) (h2 : y = 71.99999999999999) : x = 48 :=
by
  sorry

end smaller_number_l1917_191790


namespace factorize_a_cubed_minus_a_l1917_191788

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l1917_191788


namespace pool_length_l1917_191729

def volume_of_pool (width length depth : ℕ) : ℕ :=
  width * length * depth

def volume_of_water (volume : ℕ) (capacity : ℝ) : ℝ :=
  volume * capacity

theorem pool_length (L : ℕ) (width depth : ℕ) (capacity : ℝ) (drain_rate drain_time : ℕ) (h_capacity : capacity = 0.80)
  (h_width : width = 50) (h_depth : depth = 10)
  (h_drain_rate : drain_rate = 60) (h_drain_time : drain_time = 1000)
  (h_drain_volume : volume_of_water (volume_of_pool width L depth) capacity = drain_rate * drain_time) :
  L = 150 :=
by
  sorry

end pool_length_l1917_191729


namespace cars_meet_after_40_minutes_l1917_191784

noncomputable def time_to_meet 
  (BC CD : ℝ) (speed : ℝ) 
  (constant_speed : ∀ t, t > 0 → speed = (BC + CD) / t) : ℝ :=
  (BC + CD) / speed * 40 / 60

-- Define the condition that must hold: cars meet at 40 minutes
theorem cars_meet_after_40_minutes
  (BC CD : ℝ) (speed : ℝ)
  (constant_speed : ∀ t, t > 0 → speed = (BC + CD) / t) :
  time_to_meet BC CD speed constant_speed = 40 := sorry

end cars_meet_after_40_minutes_l1917_191784


namespace factor_polynomial_l1917_191743

theorem factor_polynomial (a b : ℕ) : 
  2 * a^3 - 3 * a^2 * b - 3 * a * b^2 + 2 * b^3 = (a + b) * (a - 2 * b) * (2 * a - b) :=
by sorry

end factor_polynomial_l1917_191743


namespace rate_downstream_l1917_191751

-- Define the man's rate in still water
def rate_still_water : ℝ := 24.5

-- Define the rate of the current
def rate_current : ℝ := 7.5

-- Define the man's rate upstream (unused in the proof but given in the problem)
def rate_upstream : ℝ := 17.0

-- Prove that the man's rate when rowing downstream is as stated given the conditions
theorem rate_downstream : rate_still_water + rate_current = 32 := by
  simp [rate_still_water, rate_current]
  norm_num

end rate_downstream_l1917_191751


namespace tetrahedron_surface_area_l1917_191776

theorem tetrahedron_surface_area (a : ℝ) (h : a = Real.sqrt 2) :
  let R := (a * Real.sqrt 6) / 4
  let S := 4 * Real.pi * R^2
  S = 3 * Real.pi := by
  /- Proof here -/
  sorry

end tetrahedron_surface_area_l1917_191776


namespace lakshmi_share_annual_gain_l1917_191793

theorem lakshmi_share_annual_gain (x : ℝ) (annual_gain : ℝ) (Raman_inv_months : ℝ) (Lakshmi_inv_months : ℝ) (Muthu_inv_months : ℝ) (Gowtham_inv_months : ℝ) (Pradeep_inv_months : ℝ)
  (total_inv_months : ℝ) (lakshmi_share : ℝ) :
  Raman_inv_months = x * 12 →
  Lakshmi_inv_months = 2 * x * 6 →
  Muthu_inv_months = 3 * x * 4 →
  Gowtham_inv_months = 4 * x * 9 →
  Pradeep_inv_months = 5 * x * 1 →
  total_inv_months = Raman_inv_months + Lakshmi_inv_months + Muthu_inv_months + Gowtham_inv_months + Pradeep_inv_months →
  annual_gain = 58000 →
  lakshmi_share = (Lakshmi_inv_months / total_inv_months) * annual_gain →
  lakshmi_share = 9350.65 :=
by
  sorry

end lakshmi_share_annual_gain_l1917_191793


namespace find_a_l1917_191717

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.sqrt 2

theorem find_a (a : ℝ) (h : f a (f a (Real.sqrt 2)) = -Real.sqrt 2) : 
  a = Real.sqrt 2 / 2 :=
by
  sorry

end find_a_l1917_191717


namespace intersect_x_axis_iff_k_le_4_l1917_191739

theorem intersect_x_axis_iff_k_le_4 (k : ℝ) :
  (∃ x : ℝ, (k-3) * x^2 + 2 * x + 1 = 0) ↔ k ≤ 4 :=
sorry

end intersect_x_axis_iff_k_le_4_l1917_191739


namespace time_b_is_54_l1917_191720

-- Define the time A takes to complete the work
def time_a := 27

-- Define the time B takes to complete the work as twice the time A takes
def time_b := 2 * time_a

-- Prove that B takes 54 days to complete the work
theorem time_b_is_54 : time_b = 54 :=
by
  sorry

end time_b_is_54_l1917_191720


namespace sin_750_eq_one_half_l1917_191762

theorem sin_750_eq_one_half :
  ∀ (θ: ℝ), (∀ n: ℤ, Real.sin (θ + n * 360) = Real.sin θ) → Real.sin 30 = 1 / 2 → Real.sin 750 = 1 / 2 :=
by 
  intros θ periodic_sine sin_30
  -- insert proof here
  sorry

end sin_750_eq_one_half_l1917_191762


namespace solve_part_a_solve_part_b_solve_part_c_l1917_191795

-- Part (a)
theorem solve_part_a (x : ℝ) : 
  (2 * x^2 + 3 * x - 1)^2 - 5 * (2 * x^2 + 3 * x + 3) + 24 = 0 ↔ 
  x = 1 ∨ x = -2 ∨ x = 0.5 ∨ x = -2.5 := sorry

-- Part (b)
theorem solve_part_b (x : ℝ) : 
  (x - 1) * (x + 3) * (x + 4) * (x + 8) = -96 ↔ 
  x = 0 ∨ x = -7 ∨ x = (-7 + Real.sqrt 33) / 2 ∨ x = (-7 - Real.sqrt 33) / 2 := sorry

-- Part (c)
theorem solve_part_c (x : ℝ) (hx : x ≠ 0) : 
  (x - 1) * (x - 2) * (x - 4) * (x - 8) = 4 * x^2 ↔ 
  x = 4 + 2 * Real.sqrt 2 ∨ x = 4 - 2 * Real.sqrt 2 := sorry

end solve_part_a_solve_part_b_solve_part_c_l1917_191795


namespace quadratic_has_two_distinct_real_roots_determine_k_from_roots_relation_l1917_191770

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4*a*c

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  let a := 1
  let b := 2*k - 1
  let c := -k - 1
  discriminant a b c > 0 := by
  sorry

theorem determine_k_from_roots_relation (x1 x2 k : ℝ) 
  (h1 : x1 + x2 = -(2*k - 1))
  (h2 : x1 * x2 = -k - 1)
  (h3 : x1 + x2 - 4*(x1 * x2) = 2) :
  k = -3/2 := by
  sorry

end quadratic_has_two_distinct_real_roots_determine_k_from_roots_relation_l1917_191770


namespace find_b_of_expression_l1917_191799

theorem find_b_of_expression (y : ℝ) (b : ℝ) (hy : y > 0)
  (h : (7 / 10) * y = (8 * y) / b + (3 * y) / 10) : b = 20 :=
sorry

end find_b_of_expression_l1917_191799


namespace Canada_moose_population_l1917_191749

theorem Canada_moose_population (moose beavers humans : ℕ) (h1 : beavers = 2 * moose) 
                              (h2 : humans = 19 * beavers) (h3 : humans = 38 * 10^6) : 
                              moose = 1 * 10^6 :=
by
  sorry

end Canada_moose_population_l1917_191749


namespace jimmy_exams_l1917_191756

theorem jimmy_exams (p l a : ℕ) (h_p : p = 50) (h_l : l = 5) (h_a : a = 5) (x : ℕ) :
  (20 * x - (l + a) ≥ p) ↔ (x ≥ 3) :=
by
  sorry

end jimmy_exams_l1917_191756


namespace find_integer_modulo_l1917_191711

theorem find_integer_modulo : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ 123456 [MOD 11] := by
  use 3
  sorry

end find_integer_modulo_l1917_191711


namespace single_colony_reaches_limit_in_24_days_l1917_191740

/-- A bacteria colony doubles in size every day. -/
def double (n : ℕ) : ℕ := 2 ^ n

/-- Two bacteria colonies growing simultaneously will take 24 days to reach the habitat's limit. -/
axiom two_colonies_24_days : ∀ k : ℕ, double k + double k = double 24

/-- Prove that it takes 24 days for a single bacteria colony to reach the habitat's limit. -/
theorem single_colony_reaches_limit_in_24_days : ∃ x : ℕ, double x = double 24 :=
sorry

end single_colony_reaches_limit_in_24_days_l1917_191740


namespace log_bound_sum_l1917_191768

theorem log_bound_sum (c d : ℕ) (h_c : c = 10) (h_d : d = 11) (h_bound : 10 < Real.log 1350 / Real.log 2 ∧ Real.log 1350 / Real.log 2 < 11) : c + d = 21 :=
by
  -- omitted proof
  sorry

end log_bound_sum_l1917_191768


namespace find_c_l1917_191789

theorem find_c (c : ℝ) (h : ∃ a : ℝ, x^2 - 50 * x + c = (x - a)^2) : c = 625 :=
  by
  sorry

end find_c_l1917_191789


namespace combined_weight_l1917_191783

-- Given constants
def JakeWeight : ℕ := 198
def WeightLost : ℕ := 8
def KendraWeight := (JakeWeight - WeightLost) / 2

-- Prove the combined weight of Jake and Kendra
theorem combined_weight : JakeWeight + KendraWeight = 293 := by
  sorry

end combined_weight_l1917_191783


namespace contradiction_example_l1917_191753

theorem contradiction_example (x : ℝ) (h : x^2 - 1 = 0) : x = -1 ∨ x = 1 :=
by
  sorry

end contradiction_example_l1917_191753


namespace gcd_420_135_l1917_191771

theorem gcd_420_135 : Nat.gcd 420 135 = 15 := by
  sorry

end gcd_420_135_l1917_191771


namespace new_total_lines_l1917_191787

-- Definitions and conditions
variable (L : ℕ)
def increased_lines : ℕ := L + 60
def percentage_increase := (60 : ℚ) / L = 1 / 3

-- Theorem statement
theorem new_total_lines : percentage_increase L → increased_lines L = 240 :=
by
  sorry

end new_total_lines_l1917_191787


namespace eyes_per_ant_proof_l1917_191755

noncomputable def eyes_per_ant (s a e_s E : ℕ) : ℕ :=
  let e_spiders := s * e_s
  let e_ants := E - e_spiders
  e_ants / a

theorem eyes_per_ant_proof : eyes_per_ant 3 50 8 124 = 2 :=
by
  sorry

end eyes_per_ant_proof_l1917_191755


namespace cat_food_customers_l1917_191792

/-
Problem: There was a big sale on cat food at the pet store. Some people bought cat food that day. The first 8 customers bought 3 cases each. The next four customers bought 2 cases each. The last 8 customers of the day only bought 1 case each. In total, 40 cases of cat food were sold. How many people bought cat food that day?
-/

theorem cat_food_customers:
  (8 * 3) + (4 * 2) + (8 * 1) = 40 →
  8 + 4 + 8 = 20 :=
by
  intro h
  linarith

end cat_food_customers_l1917_191792


namespace sufficient_but_not_necessary_condition_l1917_191718

theorem sufficient_but_not_necessary_condition (a b : ℝ) (hb : b < -1) : |a| + |b| > 1 := 
by
  sorry

end sufficient_but_not_necessary_condition_l1917_191718


namespace nba_conference_division_impossible_l1917_191734

theorem nba_conference_division_impossible :
  let teams := 30
  let games_per_team := 82
  let total_games := teams * games_per_team
  let unique_games := total_games / 2
  let inter_conference_games := unique_games / 2
  ¬∃ (A B : ℕ), A + B = teams ∧ A * B = inter_conference_games := 
by
  let teams := 30
  let games_per_team := 82
  let total_games := teams * games_per_team
  let unique_games := total_games / 2
  let inter_conference_games := unique_games / 2
  sorry

end nba_conference_division_impossible_l1917_191734


namespace sqrt_product_simplify_l1917_191742

theorem sqrt_product_simplify (x : ℝ) (hx : 0 ≤ x):
  Real.sqrt (48*x) * Real.sqrt (3*x) * Real.sqrt (50*x) = 60 * x * Real.sqrt x := 
by
  sorry

end sqrt_product_simplify_l1917_191742


namespace find_y_in_terms_of_x_l1917_191746

theorem find_y_in_terms_of_x (p : ℝ) (x y : ℝ) (h1 : x = 1 + 3^p) (h2 : y = 1 + 3^(-p)) : y = x / (x - 1) :=
by
  sorry

end find_y_in_terms_of_x_l1917_191746


namespace spaceship_distance_traveled_l1917_191726

theorem spaceship_distance_traveled (d_ex : ℝ) (d_xy : ℝ) (d_total : ℝ) :
  d_ex = 0.5 → d_xy = 0.1 → d_total = 0.7 → (d_total - (d_ex + d_xy)) = 0.1 :=
by
  intros h1 h2 h3
  sorry

end spaceship_distance_traveled_l1917_191726


namespace linear_eq_must_be_one_l1917_191765

theorem linear_eq_must_be_one (m : ℝ) : (∀ x y : ℝ, (m + 1) * x + 3 * y ^ m = 5 → (m = 1)) :=
by
  intros x y h
  sorry

end linear_eq_must_be_one_l1917_191765


namespace factorize_expression_l1917_191780

theorem factorize_expression (x y : ℝ) : 
  x^3 - x*y^2 = x * (x + y) * (x - y) :=
sorry

end factorize_expression_l1917_191780


namespace abs_h_eq_one_l1917_191709

theorem abs_h_eq_one (h : ℝ) (roots_square_sum_eq : ∀ x : ℝ, x^2 + 6 * h * x + 8 = 0 → x^2 + (x + 6 * h)^2 = 20) : |h| = 1 :=
by
  sorry

end abs_h_eq_one_l1917_191709


namespace bob_daily_earnings_l1917_191794

-- Define Sally's daily earnings
def Sally_daily_earnings : ℝ := 6

-- Define the total savings after a year for both Sally and Bob
def total_savings : ℝ := 1825

-- Define the number of days in a year
def days_in_year : ℝ := 365

-- Define Bob's daily earnings
variable (B : ℝ)

-- Define the proof statement
theorem bob_daily_earnings : (3 + B / 2) * days_in_year = total_savings → B = 4 :=
by
  sorry

end bob_daily_earnings_l1917_191794


namespace probability_one_from_each_l1917_191797

-- Define the total number of cards
def total_cards : ℕ := 10

-- Define the number of cards from Amelia's name
def amelia_cards : ℕ := 6

-- Define the number of cards from Lucas's name
def lucas_cards : ℕ := 4

-- Define the probability that one letter is from each person's name
theorem probability_one_from_each : (amelia_cards / total_cards) * (lucas_cards / (total_cards - 1)) +
                                    (lucas_cards / total_cards) * (amelia_cards / (total_cards - 1)) = 8 / 15 :=
by
  sorry

end probability_one_from_each_l1917_191797


namespace daniel_earnings_l1917_191763

theorem daniel_earnings :
  let monday_fabric := 20
  let monday_yarn := 15
  let tuesday_fabric := 2 * monday_fabric
  let tuesday_yarn := monday_yarn + 10
  let wednesday_fabric := (1 / 4) * tuesday_fabric
  let wednesday_yarn := (1 / 2) * tuesday_yarn
  let total_fabric := monday_fabric + tuesday_fabric + wednesday_fabric
  let total_yarn := monday_yarn + tuesday_yarn + wednesday_yarn
  let fabric_cost := 2
  let yarn_cost := 3
  let fabric_earnings_before_discount := total_fabric * fabric_cost
  let yarn_earnings_before_discount := total_yarn * yarn_cost
  let fabric_discount := if total_fabric > 30 then 0.10 * fabric_earnings_before_discount else 0
  let yarn_discount := if total_yarn > 20 then 0.05 * yarn_earnings_before_discount else 0
  let fabric_earnings_after_discount := fabric_earnings_before_discount - fabric_discount
  let yarn_earnings_after_discount := yarn_earnings_before_discount - yarn_discount
  let total_earnings := fabric_earnings_after_discount + yarn_earnings_after_discount
  total_earnings = 275.625 := by
  {
    sorry
  }

end daniel_earnings_l1917_191763


namespace age_multiple_l1917_191704

variables {R J K : ℕ}

theorem age_multiple (h1 : R = J + 6) (h2 : R = K + 3) (h3 : (R + 4) * (K + 4) = 108) :
  ∃ M : ℕ, R + 4 = M * (J + 4) ∧ M = 2 :=
sorry

end age_multiple_l1917_191704


namespace arithmetic_sequence_value_l1917_191782

variable (a : ℕ → ℤ) (d : ℤ)
variable (h1 : a 1 + a 4 + a 7 = 39)
variable (h2 : a 2 + a 5 + a 8 = 33)
variable (h_arith : ∀ n, a (n + 1) = a n + d)

theorem arithmetic_sequence_value : a 5 + a 8 + a 11 = 15 := by
  sorry

end arithmetic_sequence_value_l1917_191782


namespace solution_set_of_inequality_af_neg2x_pos_l1917_191778

-- Given conditions:
-- f(x) = x^2 + ax + b has roots -1 and 2
-- We need to prove that the solution set for af(-2x) > 0 is -1 < x < 1/2
theorem solution_set_of_inequality_af_neg2x_pos (a b : ℝ) (x : ℝ) 
  (h1 : -1 + 2 = -a) 
  (h2 : -1 * 2 = b) : 
  (a * ((-2 * x)^2 + a * (-2 * x) + b) > 0) = (-1 < x ∧ x < 1/2) :=
by
  sorry

end solution_set_of_inequality_af_neg2x_pos_l1917_191778


namespace cubic_sum_l1917_191722

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cubic_sum_l1917_191722
