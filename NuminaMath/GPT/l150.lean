import Mathlib

namespace vector2d_propositions_l150_150879

-- Define the vector structure in ℝ²
structure Vector2D where
  x : ℝ
  y : ℝ

-- Define the relation > on Vector2D
def Vector2D.gt (a1 a2 : Vector2D) : Prop :=
  a1.x > a2.x ∨ (a1.x = a2.x ∧ a1.y > a2.y)

-- Define vectors e1, e2, and 0
def e1 : Vector2D := ⟨ 1, 0 ⟩
def e2 : Vector2D := ⟨ 0, 1 ⟩
def zero : Vector2D := ⟨ 0, 0 ⟩

-- Define propositions
def prop1 : Prop := Vector2D.gt e1 e2 ∧ Vector2D.gt e2 zero
def prop2 (a1 a2 a3 : Vector2D) : Prop := Vector2D.gt a1 a2 → Vector2D.gt a2 a3 → Vector2D.gt a1 a3
def prop3 (a1 a2 a : Vector2D) : Prop := Vector2D.gt a1 a2 → Vector2D.gt (Vector2D.mk (a1.x + a.x) (a1.y + a.y)) (Vector2D.mk (a2.x + a.x) (a2.y + a.y))
def prop4 (a a1 a2 : Vector2D) : Prop := Vector2D.gt a zero → Vector2D.gt a1 a2 → Vector2D.gt (Vector2D.mk (a.x * a1.x + a.y * a1.y) (0)) (Vector2D.mk (a.x * a2.x + a.y * a2.y) 0)

-- Main theorem to prove
theorem vector2d_propositions : prop1 ∧ (∀ a1 a2 a3, prop2 a1 a2 a3) ∧ (∀ a1 a2 a, prop3 a1 a2 a) := 
by
  sorry

end vector2d_propositions_l150_150879


namespace simplify_expression_l150_150000

-- Define the conditions as parameters
variable (x y : ℕ)

-- State the theorem with the required conditions and proof goal
theorem simplify_expression (hx : x = 2) (hy : y = 3) :
  (8 * x * y^2) / (6 * x^2 * y) = 2 := by
  -- We'll provide the outline and leave the proof as sorry
  sorry

end simplify_expression_l150_150000


namespace rupert_candles_l150_150169

theorem rupert_candles (peter_candles : ℕ) (rupert_times_older : ℝ) (h1 : peter_candles = 10) (h2 : rupert_times_older = 3.5) :
    ∃ rupert_candles : ℕ, rupert_candles = peter_candles * rupert_times_older := 
by
  sorry

end rupert_candles_l150_150169


namespace sets_are_equal_l150_150022

def setA : Set ℤ := {x | ∃ a b : ℤ, x = 12 * a + 8 * b}
def setB : Set ℤ := {y | ∃ c d : ℤ, y = 20 * c + 16 * d}

theorem sets_are_equal : setA = setB := 
by
  sorry

end sets_are_equal_l150_150022


namespace range_of_a_l150_150456

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) ↔ -8 ≤ a ∧ a ≤ 0 := sorry

end range_of_a_l150_150456


namespace car_b_speed_l150_150008

theorem car_b_speed
  (v_A v_B : ℝ) (d_A d_B d : ℝ)
  (h1 : v_A = 5 / 3 * v_B)
  (h2 : d_A = v_A * 5)
  (h3 : d_B = v_B * 5)
  (h4 : d = d_A + d_B)
  (h5 : d_A = d / 2 + 25) :
  v_B = 15 := 
sorry

end car_b_speed_l150_150008


namespace probability_of_three_given_sum_seven_l150_150748

theorem probability_of_three_given_sum_seven : 
  (∃ (dice1 dice2 : ℕ), (1 ≤ dice1 ∧ dice1 ≤ 6 ∧ 1 ≤ dice2 ∧ dice2 ≤ 6) ∧ (dice1 + dice2 = 7) 
    ∧ (dice1 = 3 ∨ dice2 = 3)) →
  (∃ (dice1 dice2 : ℕ), (1 ≤ dice1 ∧ dice1 ≤ 6 ∧ 1 ≤ dice2 ∧ dice2 ≤ 6) ∧ (dice1 + dice2 = 7)) →
  ∃ (p : ℚ), p = 1/3 :=
by 
  sorry

end probability_of_three_given_sum_seven_l150_150748


namespace range_of_a_l150_150661

noncomputable def func (x a : ℝ) : ℝ := -x^2 - 2 * a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → func x a ≤ a^2) →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ func x a = a^2) →
  -1 ≤ a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l150_150661


namespace table_capacity_l150_150458

theorem table_capacity :
  ∀ (n_invited no_show tables : ℕ), n_invited = 47 → no_show = 7 → tables = 8 → 
  (n_invited - no_show) / tables = 5 := by
  intros n_invited no_show tables h_invited h_no_show h_tables
  sorry

end table_capacity_l150_150458


namespace toothpicks_total_l150_150349

-- Definitions based on the conditions
def grid_length : ℕ := 50
def grid_width : ℕ := 40

-- Mathematical statement to prove
theorem toothpicks_total : (grid_length + 1) * grid_width + (grid_width + 1) * grid_length = 4090 := by
  sorry

end toothpicks_total_l150_150349


namespace total_distance_traveled_is_7_75_l150_150051

open Real

def walking_time_minutes : ℝ := 30
def walking_rate : ℝ := 3.5

def running_time_minutes : ℝ := 45
def running_rate : ℝ := 8

theorem total_distance_traveled_is_7_75 :
  let walking_hours := walking_time_minutes / 60
  let distance_walked := walking_rate * walking_hours
  let running_hours := running_time_minutes / 60
  let distance_run := running_rate * running_hours
  let total_distance := distance_walked + distance_run
  total_distance = 7.75 :=
by
  sorry

end total_distance_traveled_is_7_75_l150_150051


namespace quadratic_inequality_l150_150734

theorem quadratic_inequality (a : ℝ) 
  (x₁ x₂ : ℝ) (h_roots : ∀ x, x^2 + (3 * a - 1) * x + a + 8 = 0) 
  (h_distinct : x₁ ≠ x₂)
  (h_x1_lt_1 : x₁ < 1) (h_x2_gt_1 : x₂ > 1) : 
  a < -2 := 
by
  sorry

end quadratic_inequality_l150_150734


namespace evaluate_expression_l150_150096

theorem evaluate_expression (a b x y c : ℝ) (h1 : a = -b) (h2 : x * y = 1) (h3 : |c| = 2) :
  (c = 2 → (a + b) / 2 + x * y - (1 / 4) * c = 1 / 2) ∧
  (c = -2 → (a + b) / 2 + x * y - (1 / 4) * c = 3 / 2) := by
  sorry

end evaluate_expression_l150_150096


namespace value_subtracted_3_times_number_eq_1_l150_150065

variable (n : ℝ) (v : ℝ)

theorem value_subtracted_3_times_number_eq_1 (h1 : n = 1.0) (h2 : 3 * n - v = 2 * n) : v = 1 :=
by
  sorry

end value_subtracted_3_times_number_eq_1_l150_150065


namespace arithmetic_sequence_sum_minimum_l150_150126

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ := 
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem arithmetic_sequence_sum_minimum (a1 : ℝ) (d : ℝ) :
  a1 = -20 ∧ (∀ n : ℕ, (S_n a1 d n) > (S_n a1 d 6)) → 
  (10 / 3 < d ∧ d < 4) := 
sorry

end arithmetic_sequence_sum_minimum_l150_150126


namespace fg_of_3_eq_29_l150_150839

def g (x : ℝ) : ℝ := x^2
def f (x : ℝ) : ℝ := 3 * x + 2

theorem fg_of_3_eq_29 : f (g 3) = 29 := by
  sorry

end fg_of_3_eq_29_l150_150839


namespace fourth_group_students_l150_150593

theorem fourth_group_students (total_students group1 group2 group3 group4 : ℕ)
  (h_total : total_students = 24)
  (h_group1 : group1 = 5)
  (h_group2 : group2 = 8)
  (h_group3 : group3 = 7)
  (h_groups_sum : group1 + group2 + group3 + group4 = total_students) :
  group4 = 4 :=
by
  -- Proof will go here
  sorry

end fourth_group_students_l150_150593


namespace youngest_child_age_l150_150655

theorem youngest_child_age :
  ∃ x : ℕ, x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 65 ∧ x = 7 :=
by
  sorry

end youngest_child_age_l150_150655


namespace arithmetic_sequence_n_2005_l150_150937

/-- Define an arithmetic sequence with first term a₁ = 1 and common difference d = 3. -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + (n - 1) * 3

/-- Statement of the proof problem. -/
theorem arithmetic_sequence_n_2005 : 
  ∃ n : ℕ, arithmetic_sequence n = 2005 ∧ n = 669 := 
sorry

end arithmetic_sequence_n_2005_l150_150937


namespace no_four_points_with_equal_tangents_l150_150751

theorem no_four_points_with_equal_tangents :
  ∀ (A B C D : ℝ × ℝ),
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
    A ≠ C ∧ B ≠ D →
    ¬ (∀ (P Q : ℝ × ℝ), (P = A ∧ Q = B) ∨ (P = C ∧ Q = D) →
      ∃ (M : ℝ × ℝ) (r : ℝ), M ≠ P ∧ M ≠ Q ∧
      (dist A M = dist C M ∧ dist B M = dist D M ∧
       dist P M > r ∧ dist Q M > r)) :=
by sorry

end no_four_points_with_equal_tangents_l150_150751


namespace find_y_l150_150168

variable {a b y : ℝ}
variable (ha : a ≠ 0) (hb : b ≠ 0)

theorem find_y (h1 : (3 * a) ^ (4 * b) = a ^ b * y ^ b) : y = 81 * a ^ 3 := by
  sorry

end find_y_l150_150168


namespace present_population_l150_150664

variable (P : ℝ)
variable (H1 : P * 1.20 = 2400)

theorem present_population (H1 : P * 1.20 = 2400) : P = 2000 :=
by {
  sorry
}

end present_population_l150_150664


namespace ratio_rect_prism_l150_150782

namespace ProofProblem

variables (w l h : ℕ)
def rect_prism (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_rect_prism (h1 : rect_prism w l h) :
  (w : ℕ) ≠ 0 ∧ (l : ℕ) ≠ 0 ∧ (h : ℕ) ≠ 0 ∧ 
  (∃ k, w = k ∧ l = k ∧ h = 2 * k) :=
sorry

end ProofProblem

end ratio_rect_prism_l150_150782


namespace four_integers_product_sum_l150_150279

theorem four_integers_product_sum (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_prod : a * b * c * d = 2002) (h_sum : a + b + c + d < 40) :
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) :=
sorry

end four_integers_product_sum_l150_150279


namespace find_N_l150_150568

theorem find_N : 
  (1993 + 1994 + 1995 + 1996 + 1997) / N = (3 + 4 + 5 + 6 + 7) / 5 → 
  N = 1995 :=
by
  sorry

end find_N_l150_150568


namespace bob_age_l150_150964

variable {b j : ℝ}

theorem bob_age (h1 : b = 3 * j - 20) (h2 : b + j = 75) : b = 51 := by
  sorry

end bob_age_l150_150964


namespace division_value_l150_150792

theorem division_value (x : ℚ) (h : (5 / 2) / x = 5 / 14) : x = 7 :=
sorry

end division_value_l150_150792


namespace robbery_participants_l150_150054

variables (A B V G : Prop)

-- Conditions
axiom cond1 : ¬G → (B ∧ ¬A)
axiom cond2 : V → ¬A ∧ ¬B
axiom cond3 : G → B
axiom cond4 : B → (A ∨ V)

-- Theorem to be proved
theorem robbery_participants : A ∧ B ∧ G :=
by 
  sorry

end robbery_participants_l150_150054


namespace area_DEFG_l150_150358

-- Define points and the properties of the rectangle ABCD
variable (A B C D E G F : Type)
variables (area_ABCD : ℝ) (Eg_parallel_AB_CD Df_parallel_AD_BC : Prop)
variable (E_position_AD : ℝ) (G_position_CD : ℝ) (F_midpoint_BC : Prop)
variables (length_abcd width_abcd : ℝ)

-- Assumptions based on given conditions
axiom h1 : area_ABCD = 150
axiom h2 : E_position_AD = 1 / 3
axiom h3 : G_position_CD = 1 / 3
axiom h4 : Eg_parallel_AB_CD
axiom h5 : Df_parallel_AD_BC
axiom h6 : F_midpoint_BC

-- Theorem to prove the area of DEFG
theorem area_DEFG : length_abcd * width_abcd / 3 = 50 :=
    sorry

end area_DEFG_l150_150358


namespace cosine_angle_between_vectors_l150_150142

noncomputable def vector_cosine (a b : ℝ × ℝ) : ℝ :=
  let dot_product := (a.1 * b.1 + a.2 * b.2)
  let magnitude_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / (magnitude_a * magnitude_b)

theorem cosine_angle_between_vectors : ∀ (k : ℝ), 
  let a := (3, 1)
  let b := (1, 3)
  let c := (k, -2)
  (3 - k) / 3 = 1 →
  vector_cosine a c = Real.sqrt 5 / 5 := by
  intros
  sorry

end cosine_angle_between_vectors_l150_150142


namespace magic_square_exists_l150_150870

theorem magic_square_exists : 
  ∃ (a b c d e f g h : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ 
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ 
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ 
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    a + b + c = 12 ∧ d + e + f = 12 ∧ g + h + 0 = 12 ∧
    a + d + g = 12 ∧ b + 0 + h = 12 ∧ c + f + 0 = 12 :=
sorry

end magic_square_exists_l150_150870


namespace units_digit_sum_l150_150560

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum :
  units_digit (24^3 + 17^3) = 7 :=
by
  sorry

end units_digit_sum_l150_150560


namespace quadratic_one_root_iff_discriminant_zero_l150_150812

theorem quadratic_one_root_iff_discriminant_zero (m : ℝ) : 
  (∃ x : ℝ, ∀ y : ℝ, y^2 - m*y + 1 ≤ 0 ↔ y = x) ↔ (m = 2 ∨ m = -2) :=
by 
  -- We assume the discriminant condition which implies the result
  sorry

end quadratic_one_root_iff_discriminant_zero_l150_150812


namespace correct_proposition_l150_150503

theorem correct_proposition (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
sorry

end correct_proposition_l150_150503


namespace hyperbola_equation_l150_150536

-- Define the conditions of the problem
def asymptotic_eq (C : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C x y → (y = 2 * x ∨ y = -2 * x)

def passes_through_point (C : ℝ → ℝ → Prop) : Prop :=
  C 2 2

-- State the equation of the hyperbola
def is_equation_of_hyperbola (C : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C x y ↔ x^2 / 3 - y^2 / 12 = 1

-- The theorem statement combining all conditions to prove the final equation
theorem hyperbola_equation {C : ℝ → ℝ → Prop} :
  asymptotic_eq C →
  passes_through_point C →
  is_equation_of_hyperbola C :=
by
  sorry

end hyperbola_equation_l150_150536


namespace distribution_of_tickets_l150_150893

-- Define the number of total people and the number of tickets
def n : ℕ := 10
def k : ℕ := 3

-- Define the permutation function P(n, k)
def P (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Main theorem statement
theorem distribution_of_tickets : P n k = 720 := by
  unfold P
  sorry

end distribution_of_tickets_l150_150893


namespace experienced_sailors_monthly_earnings_l150_150214

theorem experienced_sailors_monthly_earnings :
  let total_sailors : Nat := 17
  let inexperienced_sailors : Nat := 5
  let hourly_wage_inexperienced : Nat := 10
  let workweek_hours : Nat := 60
  let weeks_in_month : Nat := 4
  let experienced_sailors : Nat := total_sailors - inexperienced_sailors
  let hourly_wage_experienced := hourly_wage_inexperienced + (hourly_wage_inexperienced / 5)
  let weekly_earnings_experienced := hourly_wage_experienced * workweek_hours
  let total_weekly_earnings_experienced := weekly_earnings_experienced * experienced_sailors
  let monthly_earnings_experienced := total_weekly_earnings_experienced * weeks_in_month
  monthly_earnings_experienced = 34560 := by
  sorry

end experienced_sailors_monthly_earnings_l150_150214


namespace polynomial_sum_squares_l150_150635

theorem polynomial_sum_squares (a0 a1 a2 a3 a4 a5 a6 a7 : ℤ)
  (h₁ : (1 - 2) ^ 7 = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7)
  (h₂ : (1 + -2) ^ 7 = a0 - a1 + a2 - a3 + a4 - a5 + a6 - a7) :
  (a0 + a2 + a4 + a6) ^ 2 - (a1 + a3 + a5 + a7) ^ 2 = -2187 := 
  sorry

end polynomial_sum_squares_l150_150635


namespace count_squares_containing_A_l150_150555

-- Given conditions
def figure_with_squares : Prop := ∃ n : ℕ, n = 20

-- The goal is to prove that the number of squares containing A is 13
theorem count_squares_containing_A (h : figure_with_squares) : ∃ k : ℕ, k = 13 :=
by 
  sorry

end count_squares_containing_A_l150_150555


namespace simplify_expression_evaluate_expression_with_values_l150_150947

-- Problem 1: Simplify the expression to -xy
theorem simplify_expression (x y : ℤ) : 
  3 * x^2 + 2 * x * y - 4 * y^2 - 3 * x * y + 4 * y^2 - 3 * x^2 = - x * y :=
  sorry

-- Problem 2: Evaluate the expression with given values
theorem evaluate_expression_with_values (a b : ℤ) (ha : a = 2) (hb : b = -3) :
  a + (5 * a - 3 * b) - 2 * (a - 2 * b) = 5 :=
  sorry

end simplify_expression_evaluate_expression_with_values_l150_150947


namespace simplify_expression_l150_150613

theorem simplify_expression (x : ℝ) : (5 * x + 2 * x + 7 * x) = 14 * x :=
by
  sorry

end simplify_expression_l150_150613


namespace problem_l150_150253

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem 
  (h_odd : ∀ x : ℝ, f (-x) = -f x) 
  (h_periodic : ∀ x : ℝ, f (x + 1) = f (1 - x)) 
  (h_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2 ^ x - 1) 
  : f 2019 = -1 := 
sorry

end problem_l150_150253


namespace melanie_more_turnips_l150_150468

theorem melanie_more_turnips (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) :
  melanie_turnips - benny_turnips = 26 := by
  sorry

end melanie_more_turnips_l150_150468


namespace locus_of_Q_l150_150196

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2/a^2 + y^2/b^2 = 1

def A_vertice (a b : ℝ) (x y : ℝ) : Prop :=
  (x = a ∧ y = 0) ∨ (x = -a ∧ y = 0)

def chord_parallel_y_axis (x : ℝ) : Prop :=
  -- Assuming chord's x coordinate is given
  True

def lines_intersect_at_Q (a b Qx Qy : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse a b x y ∧
  A_vertice a b x y ∧
  chord_parallel_y_axis x ∧
  (
    ( (Qy - y) / (Qx - (-a)) = (Qy - 0) / (Qx - a) ) ∨ -- A'P slope-comp
    ( (Qy - (-y)) / (Qx - a) = (Qy - 0) / (Qx - (-a)) ) -- AP' slope-comp
  )

theorem locus_of_Q (a b Qx Qy : ℝ) :
  (lines_intersect_at_Q a b Qx Qy) →
  (Qx^2 / a^2 - Qy^2 / b^2 = 1) := by
  sorry

end locus_of_Q_l150_150196


namespace intersection_A_B_l150_150171

noncomputable def set_A : Set ℝ := { x | 2 ≤ x ∧ x < 4 }
noncomputable def set_B : Set ℝ := { x | 3 ≤ x }

theorem intersection_A_B :
  set_A ∩ set_B = { x | 3 ≤ x ∧ x < 4 } := 
sorry

end intersection_A_B_l150_150171


namespace largest_power_of_three_dividing_A_l150_150266

theorem largest_power_of_three_dividing_A (A : ℕ)
  (h1 : ∃ (factors : List ℕ), (∀ b ∈ factors, b > 0) ∧ factors.sum = 2011 ∧ factors.prod = A)
  : ∃ k : ℕ, 3^k ∣ A ∧ ∀ m : ℕ, 3^m ∣ A → m ≤ 669 :=
by
  sorry

end largest_power_of_three_dividing_A_l150_150266


namespace percentage_x_of_yz_l150_150478

theorem percentage_x_of_yz (x y z w : ℝ) (h1 : x = 0.07 * y) (h2 : y = 0.35 * z) (h3 : z = 0.60 * w) :
  (x / (y + z) * 100) = 1.8148 :=
by
  sorry

end percentage_x_of_yz_l150_150478


namespace scientific_notation_of_284000000_l150_150866

/--
Given the number 284000000, prove that it can be expressed in scientific notation as 2.84 * 10^8.
-/
theorem scientific_notation_of_284000000 :
  284000000 = 2.84 * 10^8 :=
sorry

end scientific_notation_of_284000000_l150_150866


namespace tangent_line_circle_midpoint_locus_l150_150132

/-- 
Let O be the circle x^2 + y^2 = 1,
M be the point (-1, -4), and
N be the point (2, 0).
-/
structure CircleTangentMidpointProblem where
  (x y : ℝ)
  (O_eq : x^2 + y^2 = 1)
  (M_eq : x = -1 ∧ y = -4)
  (N_eq : x = 2 ∧ y = 0)

/- Part (1) -/
theorem tangent_line_circle (x y : ℝ) (O_eq : x^2 + y^2 = 1) 
                            (Mx My : ℝ) : ((Mx = -1 ∧ My = -4) → 
                          
                            (x = -1 ∨ 15 * x - 8 * y - 17 = 0)) := by
  sorry

/- Part (2) -/
theorem midpoint_locus (x y : ℝ) (O_eq : x^2 + y^2 = 1) 
                       (Nx Ny : ℝ) : ((Nx = 2 ∧ Ny = 0) → 
                       
                       ((x-1)^2 + y^2 = 1 ∧ (0 ≤ x ∧ x < 1 / 2))) := by
  sorry

end tangent_line_circle_midpoint_locus_l150_150132


namespace store_profit_l150_150948

variables (m n : ℝ)

def total_profit (m n : ℝ) : ℝ :=
  110 * m - 50 * n

theorem store_profit (m n : ℝ) : total_profit m n = 110 * m - 50 * n :=
  by
  -- sorry indicates that the proof is skipped
  sorry

end store_profit_l150_150948


namespace find_x_l150_150205

theorem find_x (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 :=
by
  sorry

end find_x_l150_150205


namespace unique_positive_solution_eq_15_l150_150801

theorem unique_positive_solution_eq_15 
  (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_solution_eq_15_l150_150801


namespace value_of_s_l150_150843

theorem value_of_s (s : ℝ) : (3 * (-1)^5 + 2 * (-1)^4 - (-1)^3 + (-1)^2 - 4 * (-1) + s = 0) → (s = -5) :=
by
  intro h
  sorry

end value_of_s_l150_150843


namespace ab_divisible_by_6_l150_150368

theorem ab_divisible_by_6
  (n : ℕ) (a b : ℕ)
  (h1 : 2^n = 10 * a + b)
  (h2 : n > 3)
  (h3 : b < 10) :
  (a * b) % 6 = 0 :=
sorry

end ab_divisible_by_6_l150_150368


namespace find_expression_value_l150_150544

theorem find_expression_value (x : ℝ) (h : 4 * x^2 - 2 * x + 5 = 7) :
  2 * (x^2 - x) - (x - 1) + (2 * x + 3) = 5 := by
  sorry

end find_expression_value_l150_150544


namespace problem1_part1_problem1_part2_problem2_part1_problem2_part2_l150_150922

open Set

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | -2 < x ∧ x < 5 }
def B : Set ℝ := { x | -1 ≤ x - 1 ∧ x - 1 ≤ 2 }

theorem problem1_part1 : A ∪ B = { x | -2 < x ∧ x < 5 } := sorry
theorem problem1_part2 : A ∩ B = { x | 0 ≤ x ∧ x ≤ 3 } := sorry

def B_c : Set ℝ := { x | x < 0 ∨ 3 < x }

theorem problem2_part1 : A ∪ B_c = U := sorry
theorem problem2_part2 : A ∩ B_c = { x | (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 5) } := sorry

end problem1_part1_problem1_part2_problem2_part1_problem2_part2_l150_150922


namespace hexagon_midpoints_equilateral_l150_150763

noncomputable def inscribed_hexagon_midpoints_equilateral (r : ℝ) (h : ℝ) 
  (hex : ∀ (A B C D E F : ℝ) (O : ℝ), 
    true) : Prop :=
  ∀ (M N P : ℝ), 
    true

theorem hexagon_midpoints_equilateral (r : ℝ) (h : ℝ) 
  (hex : ∀ (A B C D E F : ℝ) (O : ℝ), 
    true) : 
  inscribed_hexagon_midpoints_equilateral r h hex :=
sorry

end hexagon_midpoints_equilateral_l150_150763


namespace fraction_product_eq_l150_150343

theorem fraction_product_eq :
  (1 / 3) * (3 / 5) * (5 / 7) * (7 / 9) = 1 / 9 := by
  sorry

end fraction_product_eq_l150_150343


namespace points_in_groups_l150_150181

theorem points_in_groups (n1 n2 : ℕ) (h_total : n1 + n2 = 28) 
  (h_lines_diff : (n1*(n1 - 1) / 2) - (n2*(n2 - 1) / 2) = 81) : 
  (n1 = 17 ∧ n2 = 11) ∨ (n1 = 11 ∧ n2 = 17) :=
by
  sorry

end points_in_groups_l150_150181


namespace complex_fraction_sum_zero_l150_150040

section complex_proof
open Complex

theorem complex_fraction_sum_zero (z1 z2 : ℂ) (hz1 : z1 = 1 + I) (hz2 : z2 = 1 - I) :
  (z1 / z2) + (z2 / z1) = 0 := by
  sorry
end complex_proof

end complex_fraction_sum_zero_l150_150040


namespace complement_of_angleA_is_54_l150_150333

variable (A : ℝ)

-- Condition: \(\angle A = 36^\circ\)
def angleA := 36

-- Definition of complement
def complement (angle : ℝ) : ℝ := 90 - angle

-- Proof statement
theorem complement_of_angleA_is_54 (h : angleA = 36) : complement angleA = 54 :=
sorry

end complement_of_angleA_is_54_l150_150333


namespace find_largest_number_l150_150012

theorem find_largest_number (w x y z : ℕ) 
  (h1 : w + x + y = 190) 
  (h2 : w + x + z = 210) 
  (h3 : w + y + z = 220) 
  (h4 : x + y + z = 235) : 
  max (max w x) (max y z) = 95 := 
sorry

end find_largest_number_l150_150012


namespace largest_of_four_consecutive_primes_l150_150412

noncomputable def sum_of_primes_is_prime (p1 p2 p3 p4 : ℕ) : Prop :=
  Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ Prime (p1 + p2 + p3 + p4)

theorem largest_of_four_consecutive_primes :
  ∃ p1 p2 p3 p4, 
  sum_of_primes_is_prime p1 p2 p3 p4 ∧ 
  p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ 
  (p1, p2, p3, p4) = (2, 3, 5, 7) ∧ 
  max p1 (max p2 (max p3 p4)) = 7 :=
by {
  sorry                                 -- solve this in Lean
}

end largest_of_four_consecutive_primes_l150_150412


namespace compare_a_b_l150_150121

theorem compare_a_b (a b : ℝ) (h₁ : a = 1.9 * 10^5) (h₂ : b = 9.1 * 10^4) : a > b := by
  sorry

end compare_a_b_l150_150121


namespace no_valid_solutions_l150_150914

theorem no_valid_solutions (x : ℝ) (h : x ≠ 1) : 
  ¬(3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) :=
sorry

end no_valid_solutions_l150_150914


namespace pretty_number_characterization_l150_150725

def is_pretty (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ k ℓ : ℕ, k < n → ℓ < n → k > 0 → ℓ > 0 → 
    (n ∣ 2*k - ℓ ∨ n ∣ 2*ℓ - k)

theorem pretty_number_characterization :
  ∀ n : ℕ, is_pretty n ↔ (Prime n ∨ n = 6 ∨ n = 9 ∨ n = 15) :=
by
  sorry

end pretty_number_characterization_l150_150725


namespace zongzi_cost_per_bag_first_batch_l150_150071

theorem zongzi_cost_per_bag_first_batch (x : ℝ)
  (h1 : 7500 / (x - 4) = 3 * (3000 / x))
  (h2 : 3000 > 0)
  (h3 : 7500 > 0)
  (h4 : x > 4) :
  x = 24 :=
by sorry

end zongzi_cost_per_bag_first_batch_l150_150071


namespace probability_of_sequence_HTHT_l150_150280

noncomputable def prob_sequence_HTHT : ℚ :=
  let p := 1 / 2
  (p * p * p * p)

theorem probability_of_sequence_HTHT :
  prob_sequence_HTHT = 1 / 16 := 
by
  sorry

end probability_of_sequence_HTHT_l150_150280


namespace simplify_expression_l150_150359

variable (a b : ℝ)

theorem simplify_expression :
  3 * a * (3 * a^3 + 2 * a^2) - 2 * a^2 * (b^2 + 1) = 9 * a^4 + 6 * a^3 - 2 * a^2 * b^2 - 2 * a^2 :=
by
  sorry

end simplify_expression_l150_150359


namespace booth_visibility_correct_l150_150396

noncomputable def booth_visibility (L : ℝ) : ℝ × ℝ :=
  let ρ_min := L
  let ρ_max := (1 + Real.sqrt 2) / 2 * L
  (ρ_min, ρ_max)

theorem booth_visibility_correct (L : ℝ) (hL : L > 0) :
  booth_visibility L = (L, (1 + Real.sqrt 2) / 2 * L) :=
by
  sorry

end booth_visibility_correct_l150_150396


namespace intersection_is_empty_l150_150541

-- Define the domain and range sets
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | 0 < x}

-- The Lean theorem to prove that the intersection of A and B is the empty set
theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end intersection_is_empty_l150_150541


namespace verify_squaring_method_l150_150378

theorem verify_squaring_method (x : ℝ) :
  ((x + 1)^3 - (x - 1)^3 - 2) / 6 = x^2 :=
by
  sorry

end verify_squaring_method_l150_150378


namespace fractions_arithmetic_lemma_l150_150293

theorem fractions_arithmetic_lemma : (8 / 15 : ℚ) - (7 / 9) + (3 / 4) = 1 / 2 := 
by
  sorry

end fractions_arithmetic_lemma_l150_150293


namespace carousel_seats_count_l150_150895

theorem carousel_seats_count :
  ∃ (yellow blue red : ℕ), 
  (yellow + blue + red = 100) ∧ 
  (yellow = 34) ∧ 
  (blue = 20) ∧ 
  (red = 46) ∧ 
  (∀ i : ℕ, i < yellow → ∃ j : ℕ, j = yellow.succ * j ∧ (j < 100 ∧ j ≠ yellow.succ * j)) ∧ 
  (∀ k : ℕ, k < blue → ∃ m : ℕ, m = blue.succ * m ∧ (m < 100 ∧ m ≠ blue.succ * m)) ∧ 
  (∀ n : ℕ, n < red → ∃ p : ℕ, p = red.succ * p ∧ (p < 100 ∧ p ≠ red.succ * p)) :=
sorry

end carousel_seats_count_l150_150895


namespace minimum_value_f_l150_150537

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem minimum_value_f (x : ℝ) (h : x > 1) : (∃ y, (f y = 3) ∧ ∀ z, z > 1 → f z ≥ 3) :=
by sorry

end minimum_value_f_l150_150537


namespace time_to_pass_tree_l150_150176

noncomputable def length_of_train : ℝ := 275
noncomputable def speed_in_kmh : ℝ := 90
noncomputable def speed_in_m_per_s : ℝ := speed_in_kmh * (5 / 18)

theorem time_to_pass_tree : (length_of_train / speed_in_m_per_s) = 11 :=
by {
  sorry
}

end time_to_pass_tree_l150_150176


namespace sum_of_exponents_l150_150617

-- Define the expression inside the radical
def radicand (a b c : ℝ) : ℝ := 40 * a^6 * b^3 * c^14

-- Define the simplified expression outside the radical
def simplified_expr (a b c : ℝ) : ℝ := (2 * a^2 * b * c^4)

-- State the theorem to prove the sum of the exponents of the variables outside the radical
theorem sum_of_exponents (a b c : ℝ) : 
  let exponents_sum := 2 + 1 + 4
  exponents_sum = 7 :=
by
  sorry

end sum_of_exponents_l150_150617


namespace find_unknown_number_l150_150092

theorem find_unknown_number (x : ℕ) (hx1 : 100 % x = 16) (hx2 : 200 % x = 4) : x = 28 :=
by 
  sorry

end find_unknown_number_l150_150092


namespace scientific_notation_of_0_0000012_l150_150084

theorem scientific_notation_of_0_0000012 :
  0.0000012 = 1.2 * 10^(-6) :=
sorry

end scientific_notation_of_0_0000012_l150_150084


namespace problem1_problem2_problem3_l150_150648

noncomputable 
def f (x : ℝ) : ℝ := Real.exp x

theorem problem1 
  (a b : ℝ)
  (h1 : f 1 = a) 
  (h2 : b = 0) : f x = Real.exp x :=
sorry

theorem problem2 
  (k : ℝ) 
  (h : ∀ x : ℝ, f x ≥ k * x) : 0 ≤ k ∧ k ≤ Real.exp 1 :=
sorry

theorem problem3 
  (t : ℝ)
  (h : t ≤ 2) : ∀ x : ℝ, f x > t + Real.log x :=
sorry

end problem1_problem2_problem3_l150_150648


namespace out_of_pocket_expense_l150_150229

theorem out_of_pocket_expense :
  let initial_purchase := 3000
  let tv_return := 700
  let bike_return := 500
  let sold_bike_cost := bike_return + (0.20 * bike_return)
  let sold_bike_sell_price := 0.80 * sold_bike_cost
  let toaster_purchase := 100
  (initial_purchase - tv_return - bike_return - sold_bike_sell_price + toaster_purchase) = 1420 :=
by
  sorry

end out_of_pocket_expense_l150_150229


namespace geometric_progression_fraction_l150_150535

theorem geometric_progression_fraction (a₁ a₂ a₃ a₄ : ℝ) (h1 : a₂ = 2 * a₁) (h2 : a₃ = 2 * a₂) (h3 : a₄ = 2 * a₃) : 
  (2 * a₁ + a₂) / (2 * a₃ + a₄) = 1 / 4 := 
by 
  sorry

end geometric_progression_fraction_l150_150535


namespace project_estimated_hours_l150_150058

theorem project_estimated_hours (extra_hours_per_day : ℕ) (normal_work_hours : ℕ) (days_to_finish : ℕ)
  (total_hours_estimation : ℕ)
  (h1 : extra_hours_per_day = 5)
  (h2 : normal_work_hours = 10)
  (h3 : days_to_finish = 100)
  (h4 : total_hours_estimation = days_to_finish * (normal_work_hours + extra_hours_per_day))
  : total_hours_estimation = 1500 :=
  by
  -- Proof to be provided 
  sorry

end project_estimated_hours_l150_150058


namespace part_I_part_II_l150_150971

theorem part_I (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_adbc: a * d = b * c) (h_ineq1: a + d > b + c): |a - d| > |b - c| :=
sorry

theorem part_II (a b c d t: ℝ) 
(h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(h_eq: t * (Real.sqrt (a^2 + b^2) * Real.sqrt (c^2 + d^2)) = Real.sqrt (a^4 + c^4) + Real.sqrt (b^4 + d^4)):
t >= Real.sqrt 2 :=
sorry

end part_I_part_II_l150_150971


namespace ratio_of_areas_GHI_to_JKL_l150_150120

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end ratio_of_areas_GHI_to_JKL_l150_150120


namespace solve_for_x_l150_150437

theorem solve_for_x (x : ℕ) (h : x + 1 = 2) : x = 1 :=
sorry

end solve_for_x_l150_150437


namespace prob_match_ends_two_games_A_wins_prob_match_ends_four_games_prob_A_wins_overall_l150_150073

noncomputable def prob_A_wins_game := 2 / 3
noncomputable def prob_B_wins_game := 1 / 3

/-- The probability that the match ends after two games with player A's victory is 4/9. -/
theorem prob_match_ends_two_games_A_wins :
  prob_A_wins_game * prob_A_wins_game = 4 / 9 := by
  sorry

/-- The probability that the match ends exactly after four games is 20/81. -/
theorem prob_match_ends_four_games :
  2 * prob_A_wins_game * prob_B_wins_game * (prob_A_wins_game^2 + prob_B_wins_game^2) = 20 / 81 := by
  sorry

/-- The probability that player A wins the match overall is 74/81. -/
theorem prob_A_wins_overall :
  (prob_A_wins_game^2 + 2 * prob_A_wins_game * prob_B_wins_game * prob_A_wins_game^2
  + 2 * prob_A_wins_game * prob_B_wins_game * prob_A_wins_game * prob_B_wins_game) / (prob_A_wins_game + prob_B_wins_game) = 74 / 81 := by
  sorry

end prob_match_ends_two_games_A_wins_prob_match_ends_four_games_prob_A_wins_overall_l150_150073


namespace charlie_certain_instrument_l150_150463

theorem charlie_certain_instrument :
  ∃ (x : ℕ), (1 + 2 + x) + (2 + 1 + 0) = 7 → x = 1 :=
by
  sorry

end charlie_certain_instrument_l150_150463


namespace original_square_side_length_l150_150588

-- Defining the variables and conditions
variables (x : ℝ) (h₁ : 1.2 * x * (x - 2) = x * x)

-- Theorem statement to prove the side length of the original square is 12 cm
theorem original_square_side_length : x = 12 :=
by
  sorry

end original_square_side_length_l150_150588


namespace new_number_formed_l150_150793

theorem new_number_formed (t u : ℕ) (ht : t < 10) (hu : u < 10) : 3 * 100 + (10 * t + u) = 300 + 10 * t + u := 
by {
  sorry
}

end new_number_formed_l150_150793


namespace abs_neg_six_l150_150200

theorem abs_neg_six : abs (-6) = 6 := by
  sorry

end abs_neg_six_l150_150200


namespace minimum_additional_coins_l150_150331

-- The conditions
def total_friends : ℕ := 15
def current_coins : ℕ := 100

-- The fact that the total coins required to give each friend a unique number of coins from 1 to 15 is 120
def total_required_coins : ℕ := (total_friends * (total_friends + 1)) / 2

-- The theorem stating the required number of additional coins
theorem minimum_additional_coins (total_friends : ℕ) (current_coins : ℕ) (total_required_coins : ℕ) : ℕ :=
  sorry

end minimum_additional_coins_l150_150331


namespace compute_nested_operation_l150_150149

def my_op (a b : ℚ) : ℚ := (a^2 - b^2) / (1 - a * b)

theorem compute_nested_operation : my_op 1 (my_op 2 (my_op 3 4)) = -18 := by
  sorry

end compute_nested_operation_l150_150149


namespace branches_on_main_stem_l150_150218

theorem branches_on_main_stem (x : ℕ) (h : 1 + x + x^2 = 57) : x = 7 :=
  sorry

end branches_on_main_stem_l150_150218


namespace megatech_budget_allocation_l150_150753

theorem megatech_budget_allocation :
  let microphotonics := 14
  let food_additives := 10
  let gmo := 24
  let industrial_lubricants := 8
  let basic_astrophysics := 25
  microphotonics + food_additives + gmo + industrial_lubricants + basic_astrophysics = 81 →
  100 - 81 = 19 :=
by
  intros
  -- We are given the sums already, so directly calculate the remaining percentage.
  sorry

end megatech_budget_allocation_l150_150753


namespace compare_abc_l150_150767

noncomputable def a : ℝ := Real.log 10 / Real.log 5
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l150_150767


namespace total_points_combined_l150_150087

-- Definitions of the conditions
def Jack_points : ℕ := 8972
def Alex_Bella_points : ℕ := 21955

-- The problem statement to be proven
theorem total_points_combined : Jack_points + Alex_Bella_points = 30927 :=
by sorry

end total_points_combined_l150_150087


namespace initial_tomato_count_l150_150154

variable (T : ℝ)
variable (H1 : T - (1 / 4 * T + 20 + 40) = 15)

theorem initial_tomato_count : T = 100 :=
by
  sorry

end initial_tomato_count_l150_150154


namespace base9_digit_divisible_by_13_l150_150794

theorem base9_digit_divisible_by_13 :
    ∃ (d : ℕ), (0 ≤ d ∧ d ≤ 8) ∧ (13 ∣ (2 * 9^4 + d * 9^3 + 6 * 9^2 + d * 9 + 4)) :=
by
  sorry

end base9_digit_divisible_by_13_l150_150794


namespace jungkook_seokjin_books_l150_150781

/-- Given the number of books Jungkook and Seokjin originally had and the number of books they 
   bought, prove that Jungkook has 7 more books than Seokjin. -/
theorem jungkook_seokjin_books
  (jungkook_initial : ℕ)
  (seokjin_initial : ℕ)
  (jungkook_bought : ℕ)
  (seokjin_bought : ℕ)
  (h1 : jungkook_initial = 28)
  (h2 : seokjin_initial = 28)
  (h3 : jungkook_bought = 18)
  (h4 : seokjin_bought = 11) :
  (jungkook_initial + jungkook_bought) - (seokjin_initial + seokjin_bought) = 7 :=
by
  sorry

end jungkook_seokjin_books_l150_150781


namespace series_sum_half_l150_150609

theorem series_sum_half :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1/2 := 
sorry

end series_sum_half_l150_150609


namespace other_cube_side_length_l150_150787

theorem other_cube_side_length (s_1 s_2 : ℝ) (h1 : s_1 = 1) (h2 : 6 * s_2^2 / 6 = 36) : s_2 = 6 :=
by
  sorry

end other_cube_side_length_l150_150787


namespace workers_in_first_group_l150_150491

theorem workers_in_first_group
  (W D : ℕ)
  (h1 : 6 * W * D = 9450)
  (h2 : 95 * D = 9975) :
  W = 15 := 
sorry

end workers_in_first_group_l150_150491


namespace product_M1_M2_l150_150712

theorem product_M1_M2 :
  (∃ M1 M2 : ℝ, (∀ x : ℝ, x ≠ 1 ∧ x ≠ 3 →
    (45 * x - 36) / (x^2 - 4 * x + 3) = M1 / (x - 1) + M2 / (x - 3)) ∧
    M1 * M2 = -222.75) :=
sorry

end product_M1_M2_l150_150712


namespace walker_rate_l150_150837

theorem walker_rate (W : ℝ) :
  (∀ t : ℝ, t = 5 / 60 ∧ t = 20 / 60 → 20 * t = (5 * 20 / 3) ∧ W * (1 / 3) = 5 / 3) →
  W = 5 :=
by
  sorry

end walker_rate_l150_150837


namespace min_dist_l150_150795

open Complex

theorem min_dist (z w : ℂ) (hz : abs (z - (2 - 5 * I)) = 2) (hw : abs (w - (-3 + 4 * I)) = 4) :
  ∃ d, d = abs (z - w) ∧ d ≥ (Real.sqrt 106 - 6) := sorry

end min_dist_l150_150795


namespace triangle_cos_Z_l150_150290

theorem triangle_cos_Z (X Y Z : ℝ) (hXZ : X + Y + Z = π) 
  (sinX : Real.sin X = 4 / 5) (cosY : Real.cos Y = 3 / 5) : 
  Real.cos Z = 7 / 25 := 
sorry

end triangle_cos_Z_l150_150290


namespace factorization_of_polynomial_l150_150977

theorem factorization_of_polynomial :
  (x : ℤ) → x^10 + x^5 + 1 = (x^2 + x + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1) :=
by
  sorry

end factorization_of_polynomial_l150_150977


namespace perpendicular_lines_k_value_l150_150081

theorem perpendicular_lines_k_value (k : ℝ) : 
  (∃ (m₁ m₂ : ℝ), (m₁ = k/3) ∧ (m₂ = 3) ∧ (m₁ * m₂ = -1)) → k = -1 :=
by
  sorry

end perpendicular_lines_k_value_l150_150081


namespace line_is_x_axis_l150_150844

theorem line_is_x_axis (A B C : ℝ) (h : ∀ x : ℝ, A * x + B * 0 + C = 0) : A = 0 ∧ B ≠ 0 ∧ C = 0 :=
by sorry

end line_is_x_axis_l150_150844


namespace total_days_of_work_l150_150780

theorem total_days_of_work (r1 r2 r3 r4 : ℝ) (h1 : r1 = 1 / 12) (h2 : r2 = 1 / 8) (h3 : r3 = 1 / 24) (h4 : r4 = 1 / 16) : 
  (1 / (r1 + r2 + r3 + r4) = 3.2) :=
by 
  sorry

end total_days_of_work_l150_150780


namespace sum_of_roots_l150_150340

   theorem sum_of_roots : 
     let a := 2
     let b := 7
     let c := 3
     let roots := (-b / a : ℝ)
     roots = -3.5 :=
   by
     sorry
   
end sum_of_roots_l150_150340


namespace min_value_x_plus_inv_x_l150_150155

theorem min_value_x_plus_inv_x (x : ℝ) (hx : x > 0) : ∃ y, (y = x + 1/x) ∧ (∀ z, z = x + 1/x → z ≥ 2) :=
by
  sorry

end min_value_x_plus_inv_x_l150_150155


namespace days_in_month_find_days_in_month_l150_150755

noncomputable def computers_per_thirty_minutes : ℕ := 225 / 100 -- representing 2.25
def monthly_computers : ℕ := 3024
def hours_per_day : ℕ := 24

theorem days_in_month (computers_per_hour : ℕ) (daily_production : ℕ) : ℕ :=
  let computers_per_hour := (2 * computers_per_thirty_minutes)
  let daily_production := (computers_per_hour * hours_per_day)
  (monthly_computers / daily_production)

theorem find_days_in_month :
  days_in_month (2 * computers_per_thirty_minutes) ((2 * computers_per_thirty_minutes) * hours_per_day) = 28 :=
by
  sorry

end days_in_month_find_days_in_month_l150_150755


namespace mikko_should_attempt_least_questions_l150_150239

theorem mikko_should_attempt_least_questions (p : ℝ) (h_p : 0 < p ∧ p < 1) : 
  ∃ (x : ℕ), x ≥ ⌈1 / (2 * p - 1)⌉ :=
by
  sorry

end mikko_should_attempt_least_questions_l150_150239


namespace smallest_value_3a_plus_1_l150_150038

theorem smallest_value_3a_plus_1 (a : ℚ) (h : 8 * a^2 + 6 * a + 5 = 2) : 3 * a + 1 = -5 / 4 :=
sorry

end smallest_value_3a_plus_1_l150_150038


namespace algebra_problem_l150_150952

theorem algebra_problem (a b c d x : ℝ) (h1 : a = -b) (h2 : c * d = 1) (h3 : |x| = 3) : 
  (a + b) / 2023 + c * d - x^2 = -8 := by
  sorry

end algebra_problem_l150_150952


namespace part_a_part_b_l150_150118

theorem part_a (N : ℕ) : ∃ (a : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i → i ≤ N → a i > 0) ∧ (∀ i : ℕ, 2 ≤ i → i ≤ N → a i > a (i - 1)) ∧ 
(∀ i j : ℕ, 1 ≤ i → i < j → j ≤ N → (1 : ℚ) / a i - (1 : ℚ) / a j = (1 : ℚ) / a 1 - (1 : ℚ) / a 2) := sorry

theorem part_b : ¬ ∃ (a : ℕ → ℕ), (∀ i : ℕ, a i > 0) ∧ (∀ i : ℕ, a i < a (i + 1)) ∧ 
(∀ i j : ℕ, i < j → (1 : ℚ) / a i - (1 : ℚ) / a j = (1 : ℚ) / a 0 - (1 : ℚ) / a 1) := sorry

end part_a_part_b_l150_150118


namespace steinburg_marching_band_l150_150108

theorem steinburg_marching_band :
  ∃ n : ℤ, n > 0 ∧ 30 * n < 1200 ∧ 30 * n % 34 = 6 ∧ 30 * n = 720 := by
  sorry

end steinburg_marching_band_l150_150108


namespace simplify_expression_l150_150552

variable (x y z : ℝ)

theorem simplify_expression (hxz : x > z) (hzy : z > y) (hy0 : y > 0) :
  (x^z * z^y * y^x) / (z^z * y^y * x^x) = x^(z-x) * z^(y-z) * y^(x-y) :=
sorry

end simplify_expression_l150_150552


namespace white_balls_in_bag_l150_150140

open BigOperators

theorem white_balls_in_bag (N : ℕ) (N_green : ℕ) (N_yellow : ℕ) (N_red : ℕ) (N_purple : ℕ)
  (prob_not_red_nor_purple : ℝ) (W : ℕ)
  (hN : N = 100)
  (hN_green : N_green = 30)
  (hN_yellow : N_yellow = 10)
  (hN_red : N_red = 47)
  (hN_purple : N_purple = 3)
  (h_prob_not_red_nor_purple : prob_not_red_nor_purple = 0.5) :
  W = 10 :=
sorry

end white_balls_in_bag_l150_150140


namespace pipe_tank_overflow_l150_150257

theorem pipe_tank_overflow (t : ℕ) :
  let rateA := 1 / 30
  let rateB := 1 / 60
  let combined_rate := rateA + rateB
  let workA := rateA * (t - 15)
  let workB := rateB * t
  (workA + workB = 1) ↔ (t = 25) := by
  sorry

end pipe_tank_overflow_l150_150257


namespace days_in_april_l150_150739

-- Hannah harvests 5 strawberries daily for the whole month of April.
def harvest_per_day : ℕ := 5
-- She gives away 20 strawberries.
def strawberries_given_away : ℕ := 20
-- 30 strawberries are stolen.
def strawberries_stolen : ℕ := 30
-- She has 100 strawberries by the end of April.
def strawberries_final : ℕ := 100

theorem days_in_april : 
  ∃ (days : ℕ), (days * harvest_per_day = strawberries_final + strawberries_given_away + strawberries_stolen) :=
by
  sorry

end days_in_april_l150_150739


namespace square_points_sum_of_squares_l150_150519

theorem square_points_sum_of_squares 
  (a b c d : ℝ) 
  (h₀_a : 0 ≤ a ∧ a ≤ 1)
  (h₀_b : 0 ≤ b ∧ b ≤ 1)
  (h₀_c : 0 ≤ c ∧ c ≤ 1)
  (h₀_d : 0 ≤ d ∧ d ≤ 1) 
  :
  2 ≤ a^2 + (1 - d)^2 + b^2 + (1 - a)^2 + c^2 + (1 - b)^2 + d^2 + (1 - c)^2 ∧
  a^2 + (1 - d)^2 + b^2 + (1 - a)^2 + c^2 + (1 - b)^2 + d^2 + (1 - c)^2 ≤ 4 := 
by
  sorry

end square_points_sum_of_squares_l150_150519


namespace min_value_proof_l150_150409

noncomputable def min_value_expression (α β : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2

theorem min_value_proof :
  ∃ α β : ℝ, min_value_expression α β = 48 := by
  sorry

end min_value_proof_l150_150409


namespace number_of_solutions_decrease_l150_150990

-- Define the conditions and the main theorem
theorem number_of_solutions_decrease (a : ℝ) :
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) → 
  (∀ x y : ℝ, x^2 - x^2 = 0 ∧ (x - a)^2 + x^2 = 1) →
  a = 1 ∨ a = -1 := 
sorry

end number_of_solutions_decrease_l150_150990


namespace jason_total_games_l150_150336

theorem jason_total_games :
  let jan_games := 11
  let feb_games := 17
  let mar_games := 16
  let apr_games := 20
  let may_games := 14
  let jun_games := 14
  let jul_games := 14
  jan_games + feb_games + mar_games + apr_games + may_games + jun_games + jul_games = 106 :=
by
  sorry

end jason_total_games_l150_150336


namespace Samuel_fraction_spent_l150_150575

variable (totalAmount receivedRatio remainingAmount : ℕ)
variable (h1 : totalAmount = 240)
variable (h2 : receivedRatio = 3 / 4)
variable (h3 : remainingAmount = 132)

theorem Samuel_fraction_spent (spend : ℚ) : 
  (spend = (1 / 5)) :=
by
  sorry

end Samuel_fraction_spent_l150_150575


namespace skating_speeds_ratio_l150_150263

theorem skating_speeds_ratio (v_s v_f : ℝ) (h1 : v_f > v_s) (h2 : |v_f + v_s| / |v_f - v_s| = 5) :
  v_f / v_s = 3 / 2 :=
by
  sorry

end skating_speeds_ratio_l150_150263


namespace smallest_value_of_x_l150_150799

theorem smallest_value_of_x (x : ℝ) (hx : |3 * x + 7| = 26) : x = -11 :=
sorry

end smallest_value_of_x_l150_150799


namespace angle_measure_is_60_l150_150134

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l150_150134


namespace Taimour_paint_time_l150_150838

theorem Taimour_paint_time (T : ℝ) (H1 : ∀ t : ℝ, t = 2 / T → t ≠ 0) (H2 : (1 / T + 2 / T) = 1 / 3) : T = 9 :=
by
  sorry

end Taimour_paint_time_l150_150838


namespace kabob_cubes_calculation_l150_150713

-- Define the properties of a slab of beef
def cubes_per_slab := 80
def cost_per_slab := 25

-- Define Simon's usage and expenditure
def simons_budget := 50
def number_of_kabob_sticks := 40

-- Auxiliary calculations for proofs (making noncomputable if necessary)
noncomputable def cost_per_cube := cost_per_slab / cubes_per_slab
noncomputable def cubes_per_kabob_stick := (2 * cubes_per_slab) / number_of_kabob_sticks

-- The theorem we want to prove
theorem kabob_cubes_calculation :
  cubes_per_kabob_stick = 4 := by
  sorry

end kabob_cubes_calculation_l150_150713


namespace common_points_count_l150_150405

variable (x y : ℝ)

def curve1 : Prop := x^2 + 4 * y^2 = 4
def curve2 : Prop := 4 * x^2 + y^2 = 4
def curve3 : Prop := x^2 + y^2 = 1

theorem common_points_count : ∀ (x y : ℝ), curve1 x y ∧ curve2 x y ∧ curve3 x y → false := by
  intros
  sorry

end common_points_count_l150_150405


namespace value_of_T_l150_150070

-- Define the main variables and conditions
variables {M T : ℝ}

-- State the conditions given in the problem
def condition1 (M T : ℝ) := 2 * M + T = 7000
def condition2 (M T : ℝ) := M + 2 * T = 9800

-- State the theorem to be proved
theorem value_of_T : 
  ∀ (M T : ℝ), condition1 M T ∧ condition2 M T → T = 4200 :=
by 
  -- Proof would go here; for now, we use "sorry" to skip it
  sorry

end value_of_T_l150_150070


namespace total_candies_l150_150538

theorem total_candies (n p r : ℕ) (H1 : n = 157) (H2 : p = 235) (H3 : r = 98) :
  n * p + r = 36993 := by
  sorry

end total_candies_l150_150538


namespace number_of_non_empty_proper_subsets_of_A_l150_150190

noncomputable def A : Set ℤ := { x : ℤ | -1 < x ∧ x ≤ 2 }

theorem number_of_non_empty_proper_subsets_of_A : 
  (∃ (A : Set ℤ), A = { x : ℤ | -1 < x ∧ x ≤ 2 }) → 
  ∃ (n : ℕ), n = 6 := by
  sorry

end number_of_non_empty_proper_subsets_of_A_l150_150190


namespace yuna_solved_problems_l150_150158

def yuna_problems_per_day : ℕ := 8
def days_per_week : ℕ := 7
def yuna_weekly_problems : ℕ := 56

theorem yuna_solved_problems :
  yuna_problems_per_day * days_per_week = yuna_weekly_problems := by
  sorry

end yuna_solved_problems_l150_150158


namespace general_term_arithmetic_sequence_l150_150440

-- Consider an arithmetic sequence {a_n}
variable (a : ℕ → ℤ)

-- Conditions
def a1 : Prop := a 1 = 1
def a3 : Prop := a 3 = -3
def is_arithmetic_sequence : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d

-- Theorem statement
theorem general_term_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 1) (h3 : a 3 = -3) (h_arith : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d) :
  ∀ n : ℕ, a n = 3 - 2 * n :=
by
  sorry  -- proof is not required

end general_term_arithmetic_sequence_l150_150440


namespace roots_reciprocal_l150_150741

theorem roots_reciprocal {a b c x y : ℝ} (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (a * x^2 + b * x + c = 0) ↔ (c * y^2 + b * y + a = 0) := by
sorry

end roots_reciprocal_l150_150741


namespace paula_shirts_count_l150_150088

variable {P : Type}

-- Given conditions as variable definitions
def initial_money : ℕ := 109
def shirt_cost : ℕ := 11
def pants_cost : ℕ := 13
def money_left : ℕ := 74
def money_spent : ℕ := initial_money - money_left
def shirts_count : ℕ → ℕ := λ S => shirt_cost * S

-- Main proposition to prove
theorem paula_shirts_count (S : ℕ) (h : money_spent = shirts_count S + pants_cost) : 
  S = 2 := by
  /- 
    Following the steps of the proof:
    1. Calculate money spent is $35.
    2. Set up the equation $11S + 13 = 35.
    3. Solve for S.
  -/
  sorry

end paula_shirts_count_l150_150088


namespace unique_set_property_l150_150421

theorem unique_set_property (a b c : ℕ) (h1: 1 < a) (h2: 1 < b) (h3: 1 < c) 
    (gcd_ab_c: (Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1))
    (property_abc: (a * b) % c = (a * c) % b ∧ (a * c) % b = (b * c) % a) : 
    (a = 2 ∧ b = 3 ∧ c = 5) ∨ 
    (a = 2 ∧ b = 5 ∧ c = 3) ∨ 
    (a = 3 ∧ b = 2 ∧ c = 5) ∨ 
    (a = 3 ∧ b = 5 ∧ c = 2) ∨ 
    (a = 5 ∧ b = 2 ∧ c = 3) ∨ 
    (a = 5 ∧ b = 3 ∧ c = 2) := sorry

end unique_set_property_l150_150421


namespace length_of_AB_is_1_l150_150404

variables {A B C : ℝ} -- Points defining the triangle vertices
variables {a b c : ℝ} -- Lengths of triangle sides opposite to angles A, B, C respectively
variables {α β γ : ℝ} -- Angles at points A B C
variables {s₁ s₂ s₃ : ℝ} -- Sin values of the angles

noncomputable def length_of_AB (a b c : ℝ) : ℝ :=
  if a + b + c = 4 ∧ a + b = 3 * c then 1 else 0

theorem length_of_AB_is_1 : length_of_AB a b c = 1 :=
by
  have h_perimeter : a + b + c = 4 := sorry
  have h_sin_condition : a + b = 3 * c := sorry
  simp [length_of_AB, h_perimeter, h_sin_condition]
  sorry

end length_of_AB_is_1_l150_150404


namespace age_ratio_l150_150903

variable (A B : ℕ)
variable (k : ℕ)

-- Define the conditions
def sum_of_ages : Prop := A + B = 60
def multiple_of_age : Prop := A = k * B

-- Theorem to prove the ratio of ages
theorem age_ratio (h_sum : sum_of_ages A B) (h_multiple : multiple_of_age A B k) : A = 12 * B :=
by
  sorry

end age_ratio_l150_150903


namespace find_equations_of_lines_l150_150529

-- Define the given constants and conditions
def point_P := (2, 2)
def line_l1 (x y : ℝ) := 3 * x - 2 * y + 1 = 0
def line_l2 (x y : ℝ) := x + 3 * y + 4 = 0
def intersection_point := (-1, -1)
def slope_perpendicular_line := 3

-- The theorem that we need to prove
theorem find_equations_of_lines :
  (∀ k, k = 0 → line_l1 2 2 → (x = y ∨ x + y = 4)) ∧
  (line_l1 (-1) (-1) ∧ line_l2 (-1) (-1) →
   (3 * x - y + 2 = 0))
:=
sorry

end find_equations_of_lines_l150_150529


namespace pool_surface_area_l150_150674

/-
  Given conditions:
  1. The width of the pool is 3 meters.
  2. The length of the pool is 10 meters.

  To prove:
  The surface area of the pool is 30 square meters.
-/
def width : ℕ := 3
def length : ℕ := 10
def surface_area (length width : ℕ) : ℕ := length * width

theorem pool_surface_area : surface_area length width = 30 := by
  unfold surface_area
  rfl

end pool_surface_area_l150_150674


namespace smallest_k_l150_150803

theorem smallest_k (n k : ℕ) (h1: 2000 < n) (h2: n < 3000)
  (h3: ∀ i, 2 ≤ i → i ≤ k → n % i = i - 1) :
  k = 9 :=
by
  sorry

end smallest_k_l150_150803


namespace interest_after_5_years_l150_150494

noncomputable def initial_amount : ℝ := 2000
noncomputable def interest_rate : ℝ := 0.08
noncomputable def duration : ℕ := 5
noncomputable def final_amount : ℝ := initial_amount * (1 + interest_rate) ^ duration
noncomputable def interest_earned : ℝ := final_amount - initial_amount

theorem interest_after_5_years : interest_earned = 938.66 := by
  sorry

end interest_after_5_years_l150_150494


namespace find_f_neg_one_l150_150657

open Real

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * sin x + b * tan x + 3

theorem find_f_neg_one (a b : ℝ) (h : f a b 1 = 1) : f a b (-1) = 5 :=
by
  sorry

end find_f_neg_one_l150_150657


namespace maximum_value_when_t_is_2_solve_for_t_when_maximum_value_is_2_l150_150175

def f (x : ℝ) (t : ℝ) : ℝ := abs (2 * x - 1) - abs (t * x + 3)

theorem maximum_value_when_t_is_2 :
  ∃ x : ℝ, (f x 2) ≤ 4 ∧ ∀ y : ℝ, (f y 2) ≤ (f x 2) := sorry

theorem solve_for_t_when_maximum_value_is_2 :
  ∃ t : ℝ, t > 0 ∧ (∀ x : ℝ, (f x t) ≤ 2 ∧ (∃ y : ℝ, (f y t) = 2)) → t = 6 := sorry

end maximum_value_when_t_is_2_solve_for_t_when_maximum_value_is_2_l150_150175


namespace right_rectangular_prism_volume_l150_150591

theorem right_rectangular_prism_volume
    (a b c : ℝ)
    (H1 : a * b = 56)
    (H2 : b * c = 63)
    (H3 : a * c = 72)
    (H4 : c = 3 * a) :
    a * b * c = 2016 * Real.sqrt 6 :=
by
  sorry

end right_rectangular_prism_volume_l150_150591


namespace solve_for_x_l150_150686

theorem solve_for_x (x : ℤ) (h : -3 * x - 8 = 8 * x + 3) : x = -1 :=
by
  sorry

end solve_for_x_l150_150686


namespace total_digits_in_book_l150_150728

open Nat

theorem total_digits_in_book (n : Nat) (h : n = 10000) : 
    let pages_1_9 := 9
    let pages_10_99 := 90 * 2
    let pages_100_999 := 900 * 3
    let pages_1000_9999 := 9000 * 4
    let page_10000 := 5
    pages_1_9 + pages_10_99 + pages_100_999 + pages_1000_9999 + page_10000 = 38894 :=
by
    sorry

end total_digits_in_book_l150_150728


namespace mark_increase_reading_time_l150_150283

def initial_pages_per_day : ℕ := 100
def final_pages_per_week : ℕ := 1750
def days_in_week : ℕ := 7

def calculate_percentage_increase (initial_pages_per_day : ℕ) (final_pages_per_week : ℕ) (days_in_week : ℕ) : ℚ :=
  ((final_pages_per_week : ℚ) / ((initial_pages_per_day : ℚ) * (days_in_week : ℚ)) - 1) * 100

theorem mark_increase_reading_time :
  calculate_percentage_increase initial_pages_per_day final_pages_per_week days_in_week = 150 :=
by sorry

end mark_increase_reading_time_l150_150283


namespace new_car_travel_distance_l150_150973

theorem new_car_travel_distance
  (old_distance : ℝ)
  (new_distance : ℝ)
  (h1 : old_distance = 150)
  (h2 : new_distance = 1.30 * old_distance) : 
  new_distance = 195 := 
by 
  /- include required assumptions and skip the proof. -/
  sorry

end new_car_travel_distance_l150_150973


namespace perpendicular_condition_l150_150417

theorem perpendicular_condition (m : ℝ) : 
  (2 * (m + 1) * (m - 3) + 2 * (m - 3) = 0) ↔ (m = 3 ∨ m = -3) :=
by
  sorry

end perpendicular_condition_l150_150417


namespace min_area_rectangle_l150_150377

theorem min_area_rectangle (l w : ℝ) 
  (hl : 3.5 ≤ l ∧ l ≤ 4.5) 
  (hw : 5.5 ≤ w ∧ w ≤ 6.5) 
  (constraint : l ≥ 2 * w) : 
  l * w = 60.5 := 
sorry

end min_area_rectangle_l150_150377


namespace coconut_grove_problem_l150_150068

theorem coconut_grove_problem
  (x : ℤ)
  (T40 : ℤ := x + 2)
  (T120 : ℤ := x)
  (T180 : ℤ := x - 2)
  (N_total : ℤ := 40 * (x + 2) + 120 * x + 180 * (x - 2))
  (T_total : ℤ := (x + 2) + x + (x - 2))
  (average_yield : ℤ := 100) :
  (N_total / T_total) = average_yield → x = 7 :=
by
  sorry

end coconut_grove_problem_l150_150068


namespace least_value_l150_150123

-- Define the quadratic function and its conditions
def quadratic_function (p q r : ℝ) (x : ℝ) : ℝ :=
  p * x^2 + q * x + r

-- Define the conditions for p, q, and r
def conditions (p q r : ℝ) : Prop :=
  p > 0 ∧ (q^2 - 4 * p * r < 0)

-- State the theorem that given the conditions the least value is (4pr - q^2) / 4p
theorem least_value (p q r : ℝ) (h : conditions p q r) :
  ∃ x : ℝ, (∀ y : ℝ, quadratic_function p q r y ≥ quadratic_function p q r x) ∧
  quadratic_function p q r x = (4 * p * r - q^2) / (4 * p) :=
sorry

end least_value_l150_150123


namespace largest_B_is_9_l150_150305

def is_divisible_by_three (n : ℕ) : Prop :=
  n % 3 = 0

def is_divisible_by_four (n : ℕ) : Prop :=
  n % 4 = 0

def largest_B_divisible_by_3_and_4 (B : ℕ) : Prop :=
  is_divisible_by_three (21 + B) ∧ is_divisible_by_four 32

theorem largest_B_is_9 : largest_B_divisible_by_3_and_4 9 :=
by
  have h1 : is_divisible_by_three (21 + 9) := by sorry
  have h2 : is_divisible_by_four 32 := by sorry
  exact ⟨h1, h2⟩

end largest_B_is_9_l150_150305


namespace at_least_one_expression_is_leq_neg_two_l150_150480

variable (a b c : ℝ)

theorem at_least_one_expression_is_leq_neg_two 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 1 / b ≤ -2) ∨ (b + 1 / c ≤ -2) ∨ (c + 1 / a ≤ -2) :=
sorry

end at_least_one_expression_is_leq_neg_two_l150_150480


namespace perimeter_of_region_l150_150746

-- Define the condition
def area_of_region := 512 -- square centimeters
def number_of_squares := 8

-- Define the presumed perimeter
def presumed_perimeter := 144 -- the correct answer

-- Mathematical statement that needs proof
theorem perimeter_of_region (area_of_region: ℕ) (number_of_squares: ℕ) (presumed_perimeter: ℕ) : 
   area_of_region = 512 ∧ number_of_squares = 8 → presumed_perimeter = 144 :=
by 
  sorry

end perimeter_of_region_l150_150746


namespace valid_patents_growth_l150_150133

variable (a b : ℝ)

def annual_growth_rate : ℝ := 0.23

theorem valid_patents_growth (h1 : b = (1 + annual_growth_rate)^2 * a) : b = (1 + 0.23)^2 * a :=
by
  sorry

end valid_patents_growth_l150_150133


namespace find_A_l150_150697

theorem find_A (A B : ℕ) (h : 632 - (100 * A + 10 * B) = 41) : A = 5 :=
by 
  sorry

end find_A_l150_150697


namespace binomial_sum_zero_l150_150221

open BigOperators

theorem binomial_sum_zero {n m : ℕ} (h1 : 1 ≤ m) (h2 : m < n) :
  ∑ k in Finset.range (n + 1), (-1 : ℤ) ^ k * k ^ m * Nat.choose n k = 0 :=
by
  sorry

end binomial_sum_zero_l150_150221


namespace min_value_M_l150_150562

theorem min_value_M (a b c : ℝ) (h1 : a < b) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0): 
  ∃ M : ℝ, M = 8 ∧ M = (a + 2 * b + 4 * c) / (b - a) :=
sorry

end min_value_M_l150_150562


namespace fill_time_is_13_seconds_l150_150723

-- Define the given conditions as constants
def flow_rate_in (t : ℝ) : ℝ := 24 * t -- 24 gallons/second
def leak_rate (t : ℝ) : ℝ := 4 * t -- 4 gallons/second
def basin_capacity : ℝ := 260 -- 260 gallons

-- Main theorem to be proven
theorem fill_time_is_13_seconds : 
  ∀ t : ℝ, (flow_rate_in t - leak_rate t) * (13) = basin_capacity := 
sorry

end fill_time_is_13_seconds_l150_150723


namespace jessica_earned_from_washing_l150_150911

-- Conditions defined as per Problem a)
def weekly_allowance : ℕ := 10
def spent_on_movies : ℕ := weekly_allowance / 2
def remaining_after_movies : ℕ := weekly_allowance - spent_on_movies
def final_amount : ℕ := 11
def earned_from_washing : ℕ := final_amount - remaining_after_movies

-- Lean statement to prove Jessica earned $6 from washing the family car
theorem jessica_earned_from_washing :
  earned_from_washing = 6 := 
by
  -- Proof to be filled in later (skipped here with sorry)
  sorry

end jessica_earned_from_washing_l150_150911


namespace sum_alternating_series_l150_150148

theorem sum_alternating_series :
  (Finset.sum (Finset.range 2023) (λ k => (-1)^(k + 1))) = -1 := 
by
  sorry

end sum_alternating_series_l150_150148


namespace find_other_root_l150_150550

theorem find_other_root (a b c x : ℝ) (h₁ : a ≠ 0) 
  (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h₄ : a * (b + 2 * c) * x^2 + b * (2 * c - a) * x + c * (2 * a - b) = 0)
  (h₅ : a * (b + 2 * c) - b * (2 * c - a) + c * (2 * a - b) = 0) :
  ∃ y : ℝ, y = - (c * (2 * a - b)) / (a * (b + 2 * c)) :=
sorry

end find_other_root_l150_150550


namespace ellipse_equation_and_m_value_l150_150930

variable {a b : ℝ}
variable (e : ℝ) (F : ℝ × ℝ) (h1 : e = Real.sqrt 2 / 2) (h2 : F = (1, 0))

theorem ellipse_equation_and_m_value (h3 : a > b) (h4 : b > 0) 
  (h5 : (x y : ℝ) → (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1 → (x - 1) ^ 2 + y ^ 2 = 1) :
  (a = Real.sqrt 2 ∧ b = 1) ∧
  (∀ m : ℝ, (y = x + m) → 
  ((∃ A B : ℝ × ℝ, A = (x₁, x₁ + m) ∧ B = (x₂, x₂ + m) ∧
  (x₁ ^ 2) / 2 + (x₁ + m) ^ 2 = 1 ∧ (x₂ ^ 2) / 2 + (x₂ + m) ^ 2 = 1 ∧
  x₁ * x₂ + (x₁ + m) * (x₂ + m) = -1) ↔ m = Real.sqrt 3 / 3 ∨ m = - Real.sqrt 3 / 3))
  :=
sorry

end ellipse_equation_and_m_value_l150_150930


namespace rent_change_percent_l150_150193

open Real

noncomputable def elaine_earnings_last_year (E : ℝ) : ℝ :=
E

noncomputable def elaine_rent_last_year (E : ℝ) : ℝ :=
0.2 * E

noncomputable def elaine_earnings_this_year (E : ℝ) : ℝ :=
1.15 * E

noncomputable def elaine_rent_this_year (E : ℝ) : ℝ :=
0.25 * (1.15 * E)

noncomputable def rent_percentage_change (E : ℝ) : ℝ :=
(elaine_rent_this_year E) / (elaine_rent_last_year E) * 100

theorem rent_change_percent (E : ℝ) :
  rent_percentage_change E = 143.75 :=
by
  sorry

end rent_change_percent_l150_150193


namespace Vinnie_exceeded_word_limit_l150_150111

theorem Vinnie_exceeded_word_limit :
  let words_limit := 1000
  let words_saturday := 450
  let words_sunday := 650
  let total_words := words_saturday + words_sunday
  total_words - words_limit = 100 :=
by
  sorry

end Vinnie_exceeded_word_limit_l150_150111


namespace total_seniors_is_161_l150_150164

def total_students : ℕ := 240

def percentage_statistics : ℚ := 0.45
def percentage_geometry : ℚ := 0.35
def percentage_calculus : ℚ := 0.20

def percentage_stats_and_calc : ℚ := 0.10
def percentage_geom_and_calc : ℚ := 0.05

def percentage_seniors_statistics : ℚ := 0.90
def percentage_seniors_geometry : ℚ := 0.60
def percentage_seniors_calculus : ℚ := 0.80

def students_in_statistics : ℚ := percentage_statistics * total_students
def students_in_geometry : ℚ := percentage_geometry * total_students
def students_in_calculus : ℚ := percentage_calculus * total_students

def students_in_stats_and_calc : ℚ := percentage_stats_and_calc * students_in_statistics
def students_in_geom_and_calc : ℚ := percentage_geom_and_calc * students_in_geometry

def unique_students_in_statistics : ℚ := students_in_statistics - students_in_stats_and_calc
def unique_students_in_geometry : ℚ := students_in_geometry - students_in_geom_and_calc
def unique_students_in_calculus : ℚ := students_in_calculus - students_in_stats_and_calc - students_in_geom_and_calc

def seniors_in_statistics : ℚ := percentage_seniors_statistics * unique_students_in_statistics
def seniors_in_geometry : ℚ := percentage_seniors_geometry * unique_students_in_geometry
def seniors_in_calculus : ℚ := percentage_seniors_calculus * unique_students_in_calculus

def total_seniors : ℚ := seniors_in_statistics + seniors_in_geometry + seniors_in_calculus

theorem total_seniors_is_161 : total_seniors = 161 :=
by
  sorry

end total_seniors_is_161_l150_150164


namespace rain_first_hour_l150_150735

theorem rain_first_hour (x : ℝ) 
  (h1 : 22 = x + (2 * x + 7)) : x = 5 :=
by
  sorry

end rain_first_hour_l150_150735


namespace cos_240_eq_neg_half_l150_150656

/-- Prove that cos 240 degrees equals -1/2 --/
theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1/2 :=
  sorry

end cos_240_eq_neg_half_l150_150656


namespace water_left_ratio_l150_150308

theorem water_left_ratio (h1: 2 * (30 / 10) = 6)
                        (h2: 2 * (30 / 10) = 6)
                        (h3: 4 * (60 / 10) = 24)
                        (water_left: ℕ)
                        (total_water_collected: ℕ) 
                        (h4: water_left = 18)
                        (h5: total_water_collected = 36) : 
  water_left * 2 = total_water_collected :=
by
  sorry

end water_left_ratio_l150_150308


namespace factorize_expression_equilateral_triangle_of_sides_two_p_eq_m_plus_n_l150_150055

-- Problem 1: Factorize x^2 - y^2 + 2x - 2y
theorem factorize_expression (x y : ℝ) : x^2 - y^2 + 2 * x - 2 * y = (x - y) * (x + y + 2) := 
by sorry

-- Problem 2: Determine the shape of a triangle given a^2 + c^2 - 2b(a - b + c) = 0
theorem equilateral_triangle_of_sides (a b c : ℝ) (h : a^2 + c^2 - 2 * b * (a - b + c) = 0) : a = b ∧ b = c :=
by sorry

-- Problem 3: Prove that 2p = m + n given (1/4)(m - n)^2 = (p - n)(m - p)
theorem two_p_eq_m_plus_n (m n p : ℝ) (h : (1/4) * (m - n)^2 = (p - n) * (m - p)) : 2 * p = m + n := 
by sorry

end factorize_expression_equilateral_triangle_of_sides_two_p_eq_m_plus_n_l150_150055


namespace last_box_weight_l150_150500

theorem last_box_weight (a b c : ℕ) (h1 : a = 2) (h2 : b = 11) (h3 : a + b + c = 18) : c = 5 :=
by
  sorry

end last_box_weight_l150_150500


namespace noodles_initial_l150_150445

-- Definitions of our conditions
def given_away : ℝ := 12.0
def noodles_left : ℝ := 42.0
def initial_noodles : ℝ := 54.0

-- Theorem statement
theorem noodles_initial (a b : ℝ) (x : ℝ) (h₁ : a = 12.0) (h₂ : b = 42.0) (h₃ : x = a + b) : x = initial_noodles :=
by
  -- Placeholder for the proof
  sorry

end noodles_initial_l150_150445


namespace digit_after_decimal_l150_150983

theorem digit_after_decimal (n : ℕ) : 
  (Nat.floor (10 * (Real.sqrt (n^2 + n) - Nat.floor (Real.sqrt (n^2 + n))))) = 4 :=
by
  sorry

end digit_after_decimal_l150_150983


namespace sum_of_ais_l150_150098

theorem sum_of_ais :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 : ℕ), 
    (a1 > 0) ∧ (a2 > 0) ∧ (a3 > 0) ∧ (a4 > 0) ∧ (a5 > 0) ∧ (a6 > 0) ∧ (a7 > 0) ∧ (a8 > 0) ∧
    a1^2 + (2*a2)^2 + (3*a3)^2 + (4*a4)^2 + (5*a5)^2 + (6*a6)^2 + (7*a7)^2 + (8*a8)^2 = 204 ∧
    a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 8 :=
by
  sorry

end sum_of_ais_l150_150098


namespace incorrect_independence_test_conclusion_l150_150425

-- Definitions for each condition
def independence_test_principle_of_small_probability (A : Prop) : Prop :=
A  -- Statement A: The independence test is based on the principle of small probability.

def independence_test_conclusion_variability (C : Prop) : Prop :=
C  -- Statement C: Different samples may lead to different conclusions in the independence test.

def independence_test_not_the_only_method (D : Prop) : Prop :=
D  -- Statement D: The independence test is not the only method to determine whether two categorical variables are related.

-- Incorrect statement B
def independence_test_conclusion_always_correct (B : Prop) : Prop :=
B  -- Statement B: The conclusion drawn from the independence test is always correct.

-- Prove that statement B is incorrect given conditions A, C, and D
theorem incorrect_independence_test_conclusion (A B C D : Prop) 
  (hA : independence_test_principle_of_small_probability A)
  (hC : independence_test_conclusion_variability C)
  (hD : independence_test_not_the_only_method D) :
  ¬ independence_test_conclusion_always_correct B :=
sorry

end incorrect_independence_test_conclusion_l150_150425


namespace min_a_b_div_1176_l150_150685

theorem min_a_b_div_1176 (a b : ℕ) (h : b^3 = 1176 * a) : a = 63 :=
by sorry

end min_a_b_div_1176_l150_150685


namespace exists_h_l150_150268

noncomputable def F (x : ℝ) : ℝ := x^2 + 12 / x^2
noncomputable def G (x : ℝ) : ℝ := Real.sin (Real.pi * x^2)
noncomputable def H (x : ℝ) : ℝ := 1

theorem exists_h (h : ℝ → ℝ) (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 10) :
  |h x - x| < 1 / 3 :=
sorry

end exists_h_l150_150268


namespace sin_double_angle_l150_150477

theorem sin_double_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 :=
by sorry

end sin_double_angle_l150_150477


namespace athletes_in_camp_hours_l150_150604

theorem athletes_in_camp_hours (initial_athletes : ℕ) (left_rate : ℕ) (left_hours : ℕ) (arrived_rate : ℕ) 
  (difference : ℕ) (hours : ℕ) 
  (h_initial: initial_athletes = 300) 
  (h_left_rate: left_rate = 28) 
  (h_left_hours: left_hours = 4) 
  (h_arrived_rate: arrived_rate = 15) 
  (h_difference: difference = 7) 
  (h_left: left_rate * left_hours = 112) 
  (h_equation: initial_athletes - (left_rate * left_hours) + (arrived_rate * hours) = initial_athletes - difference) : 
  hours = 7 :=
by
  sorry

end athletes_in_camp_hours_l150_150604


namespace nth_permutation_2013_eq_3546127_l150_150775

-- Given the digits 1 through 7, there are 7! = 5040 permutations.
-- We want to prove that the 2013th permutation in ascending order is 3546127.

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def nth_permutation (n : ℕ) (digits : List ℕ) : List ℕ :=
  sorry

theorem nth_permutation_2013_eq_3546127 :
  nth_permutation 2013 digits = [3, 5, 4, 6, 1, 2, 7] :=
sorry

end nth_permutation_2013_eq_3546127_l150_150775


namespace instantaneous_velocity_at_2_l150_150891

def s (t : ℝ) : ℝ := 3 * t^3 - 2 * t^2 + t + 1

theorem instantaneous_velocity_at_2 : 
  (deriv s 2) = 29 :=
by
  -- The proof is skipped by using sorry
  sorry

end instantaneous_velocity_at_2_l150_150891


namespace train_length_l150_150961

namespace TrainProblem

def speed_kmh : ℤ := 60
def time_sec : ℤ := 18
def speed_ms : ℚ := (speed_kmh : ℚ) * (1000 / 1) * (1 / 3600)
def length_meter := speed_ms * (time_sec : ℚ)

theorem train_length :
  length_meter = 300.06 := by
  sorry

end TrainProblem

end train_length_l150_150961


namespace Lagrange_interpolation_poly_l150_150894

noncomputable def Lagrange_interpolation (P : ℝ → ℝ) : Prop :=
  P (-1) = -11 ∧ P (1) = -3 ∧ P (2) = 1 ∧ P (3) = 13

theorem Lagrange_interpolation_poly :
  ∃ P : ℝ → ℝ, Lagrange_interpolation P ∧ ∀ x, P x = x^3 - 2*x^2 + 3*x - 5 :=
by
  sorry

end Lagrange_interpolation_poly_l150_150894


namespace positive_two_digit_integers_remainder_4_div_9_l150_150790

theorem positive_two_digit_integers_remainder_4_div_9 : ∃ (n : ℕ), 
  (10 ≤ 9 * n + 4) ∧ (9 * n + 4 < 100) ∧ (∃ (k : ℕ), 1 ≤ k ∧ k ≤ 10 ∧ ∀ m, 1 ≤ m ∧ m ≤ 10 → n = k) :=
by
  sorry

end positive_two_digit_integers_remainder_4_div_9_l150_150790


namespace parametric_curve_intersects_itself_l150_150936

-- Given parametric equations
def param_x (t : ℝ) : ℝ := t^2 + 3
def param_y (t : ℝ) : ℝ := t^3 - 6 * t + 4

-- Existential statement for self-intersection
theorem parametric_curve_intersects_itself :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ param_x t1 = param_x t2 ∧ param_y t1 = param_y t2 ∧ param_x t1 = 9 ∧ param_y t1 = 4 :=
sorry

end parametric_curve_intersects_itself_l150_150936


namespace xyz_problem_l150_150875

variables {x y z : ℝ}

theorem xyz_problem
  (h1 : y + z = 10 - 4 * x)
  (h2 : x + z = -16 - 4 * y)
  (h3 : x + y = 9 - 4 * z) :
  3 * x + 3 * y + 3 * z = 1.5 :=
by 
  sorry

end xyz_problem_l150_150875


namespace soccer_team_selection_l150_150559

-- Definitions of the problem
def total_members := 16
def utility_exclusion_cond := total_members - 1

-- Lean statement for the proof problem, using the conditions and answer:
theorem soccer_team_selection :
  (utility_exclusion_cond) * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4) = 409500 :=
by
  sorry

end soccer_team_selection_l150_150559


namespace unique_bisecting_line_exists_l150_150182

noncomputable def triangle_area := 1 / 2 * 6 * 8
noncomputable def triangle_perimeter := 6 + 8 + 10

theorem unique_bisecting_line_exists :
  ∃ (line : ℝ → ℝ), 
    (∃ x y : ℝ, x + y = 12 ∧ x * y = 30 ∧ 
      1 / 2 * x * y * (24 / triangle_perimeter) = 12) ∧
    (∃ x' y' : ℝ, x' + y' = 12 ∧ x' * y' = 24 ∧ 
      1 / 2 * x' * y' * (24 / triangle_perimeter) = 12) ∧
    ((x = x' ∧ y = y') ∨ (x = y' ∧ y = x')) :=
sorry

end unique_bisecting_line_exists_l150_150182


namespace isosceles_triangle_base_angle_l150_150915

theorem isosceles_triangle_base_angle (α β γ : ℝ) 
  (h_triangle: α + β + γ = 180) 
  (h_isosceles: α = β ∨ α = γ ∨ β = γ) 
  (h_one_angle: α = 80 ∨ β = 80 ∨ γ = 80) : 
  (α = 50 ∨ β = 50 ∨ γ = 50) ∨ (α = 80 ∨ β = 80 ∨ γ = 80) :=
by 
  sorry

end isosceles_triangle_base_angle_l150_150915


namespace Tricia_is_five_years_old_l150_150502

noncomputable def Vincent_age : ℕ := 22
noncomputable def Rupert_age : ℕ := Vincent_age - 2
noncomputable def Khloe_age : ℕ := Rupert_age - 10
noncomputable def Eugene_age : ℕ := 3 * Khloe_age
noncomputable def Yorick_age : ℕ := 2 * Eugene_age
noncomputable def Amilia_age : ℕ := Yorick_age / 4
noncomputable def Tricia_age : ℕ := Amilia_age / 3

theorem Tricia_is_five_years_old : Tricia_age = 5 := by
  unfold Tricia_age Amilia_age Yorick_age Eugene_age Khloe_age Rupert_age Vincent_age
  sorry

end Tricia_is_five_years_old_l150_150502


namespace kelly_games_left_l150_150811

theorem kelly_games_left (initial_games : Nat) (given_away : Nat) (remaining_games : Nat) 
  (h1 : initial_games = 106) (h2 : given_away = 64) : remaining_games = 42 := by
  sorry

end kelly_games_left_l150_150811


namespace sunflower_mix_is_50_percent_l150_150625

-- Define the proportions and percentages given in the problem
def prop_A : ℝ := 0.60 -- 60% of the mix is Brand A
def prop_B : ℝ := 0.40 -- 40% of the mix is Brand B
def sf_A : ℝ := 0.60 -- Brand A is 60% sunflower
def sf_B : ℝ := 0.35 -- Brand B is 35% sunflower

-- Define the final percentage of sunflower in the mix
noncomputable def sunflower_mix_percentage : ℝ :=
  (sf_A * prop_A) + (sf_B * prop_B)

-- Statement to prove that the percentage of sunflower in the mix is 50%
theorem sunflower_mix_is_50_percent : sunflower_mix_percentage = 0.50 :=
by
  sorry

end sunflower_mix_is_50_percent_l150_150625


namespace x_cube_plus_y_cube_l150_150806

theorem x_cube_plus_y_cube (x y : ℝ) (h₁ : x + y = 1) (h₂ : x^2 + y^2 = 3) : x^3 + y^3 = 4 :=
sorry

end x_cube_plus_y_cube_l150_150806


namespace original_rice_amount_l150_150232

theorem original_rice_amount (x : ℝ) 
  (h1 : (x / 2) - 3 = 18) : 
  x = 42 :=
sorry

end original_rice_amount_l150_150232


namespace number_of_avocados_l150_150507

-- Constants for the given problem
def banana_cost : ℕ := 1
def apple_cost : ℕ := 2
def strawberry_cost_per_12 : ℕ := 4
def avocado_cost : ℕ := 3
def grape_cost_half_bunch : ℕ := 2
def total_cost : ℤ := 28

-- Quantities of the given fruits
def banana_qty : ℕ := 4
def apple_qty : ℕ := 3
def strawberry_qty : ℕ := 24
def grape_qty_full_bunch_cost : ℕ := 4 -- since half bunch cost $2, full bunch cost $4

-- Definition to calculate the cost of the known fruits
def known_fruit_cost : ℤ :=
  (banana_qty * banana_cost) +
  (apple_qty * apple_cost) +
  (strawberry_qty / 12 * strawberry_cost_per_12) +
  grape_qty_full_bunch_cost

-- The cost of avocados needed to fill the total cost
def avocado_cost_needed : ℤ := total_cost - known_fruit_cost

-- Finally, we need to prove that the number of avocados is 2
theorem number_of_avocados (n : ℕ) : n * avocado_cost = avocado_cost_needed → n = 2 :=
by
  -- Problem data
  have h_banana : ℕ := banana_qty * banana_cost
  have h_apple : ℕ := apple_qty * apple_cost
  have h_strawberry : ℕ := (strawberry_qty / 12) * strawberry_cost_per_12
  have h_grape : ℕ := grape_qty_full_bunch_cost
  have h_known : ℕ := h_banana + h_apple + h_strawberry + h_grape
  
  -- Calculation for number of avocados
  have h_avocado : ℤ := total_cost - h_known
  
  -- Proving number of avocados
  sorry

end number_of_avocados_l150_150507


namespace g_inv_f_7_l150_150249

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_g (x : ℝ) : f_inv (g x) = x^3 - 1
axiom g_exists_inv : ∀ y : ℝ, ∃ x : ℝ, g x = y

theorem g_inv_f_7 : g_inv (f 7) = 2 :=
by
  sorry

end g_inv_f_7_l150_150249


namespace price_of_first_tea_x_l150_150246

theorem price_of_first_tea_x (x : ℝ) :
  let price_second := 135
  let price_third := 173.5
  let avg_price := 152
  let ratio := [1, 1, 2]
  1 * x + 1 * price_second + 2 * price_third = 4 * avg_price -> x = 126 :=
by
  intros price_second price_third avg_price ratio h
  sorry

end price_of_first_tea_x_l150_150246


namespace find_missing_number_l150_150341

theorem find_missing_number (x : ℝ) :
  (20 + 40 + 60) / 3 = (10 + x + 35) / 3 + 5 → x = 60 :=
by
  sorry

end find_missing_number_l150_150341


namespace notecard_calculation_l150_150779

theorem notecard_calculation (N E : ℕ) (h₁ : N - E = 80) (h₂ : N = 3 * E) : N = 120 :=
sorry

end notecard_calculation_l150_150779


namespace n_leq_1972_l150_150717

theorem n_leq_1972 (n : ℕ) (h1 : 4 ^ 27 + 4 ^ 1000 + 4 ^ n = k ^ 2) : n ≤ 1972 :=
by
  sorry

end n_leq_1972_l150_150717


namespace divisor_is_20_l150_150165

theorem divisor_is_20 (D : ℕ) 
  (h1 : 242 % D = 11) 
  (h2 : 698 % D = 18) 
  (h3 : 940 % D = 9) :
  D = 20 :=
sorry

end divisor_is_20_l150_150165


namespace triangle_perimeter_l150_150314

theorem triangle_perimeter (x : ℕ) 
  (h1 : x % 2 = 1) 
  (h2 : 7 - 2 < x)
  (h3 : x < 2 + 7) :
  2 + 7 + x = 16 := 
sorry

end triangle_perimeter_l150_150314


namespace derivative_at_2_l150_150345

theorem derivative_at_2 (f : ℝ → ℝ) (h : ∀ x, f x = x^2 * deriv f 2 + 5 * x) :
    deriv f 2 = -5/3 :=
by
  sorry

end derivative_at_2_l150_150345


namespace complement_in_N_l150_150075

variable (M : Set ℕ) (N : Set ℕ)
def complement_N (M N : Set ℕ) : Set ℕ := { x ∈ N | x ∉ M }

theorem complement_in_N (M : Set ℕ) (N : Set ℕ) : 
  M = {2, 3, 4} → N = {0, 2, 3, 4, 5} → complement_N M N = {0, 5} :=
by
  intro hM hN
  subst hM
  subst hN 
  -- sorry is used to skip the proof
  sorry

end complement_in_N_l150_150075


namespace simplify_evaluate_expression_l150_150887

theorem simplify_evaluate_expression (a b : ℤ) (h1 : a = -2) (h2 : b = 4) : 
  (-(3 * a)^2 + 6 * a * b - (a^2 + 3 * (a - 2 * a * b))) = 14 :=
by
  rw [h1, h2]
  sorry

end simplify_evaluate_expression_l150_150887


namespace approximate_pi_value_l150_150371

theorem approximate_pi_value (r h : ℝ) (L : ℝ) (V : ℝ) (π : ℝ) 
  (hL : L = 2 * π * r)
  (hV : V = 1 / 3 * π * r^2 * h) 
  (approxV : V = 2 / 75 * L^2 * h) :
  π = 25 / 8 := 
by
  -- Proof goes here
  sorry

end approximate_pi_value_l150_150371


namespace binom_sum_l150_150315

theorem binom_sum : Nat.choose 18 4 + Nat.choose 5 2 = 3070 := 
by
  sorry

end binom_sum_l150_150315


namespace g_iterated_six_times_is_2_l150_150813

def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem g_iterated_six_times_is_2 : g (g (g (g (g (g 2))))) = 2 := 
by 
  sorry

end g_iterated_six_times_is_2_l150_150813


namespace fixed_constant_t_l150_150276

-- Representation of point on the Cartesian plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of the parabola y = 4x^2
def parabola (p : Point) : Prop := p.y = 4 * p.x^2

-- Definition of distance squared between two points
def distance_squared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Main theorem statement
theorem fixed_constant_t :
  ∃ (c : ℝ) (C : Point), c = 1/8 ∧ C = ⟨1, c⟩ ∧ 
  (∀ (A B : Point), parabola A ∧ parabola B ∧ 
  (∃ m k : ℝ, A.y = m * A.x + k ∧ B.y = m * B.x + k ∧ k = c - m) → 
  (1 / distance_squared A C + 1 / distance_squared B C = 16)) :=
by {
  -- Proof omitted
  sorry
}

end fixed_constant_t_l150_150276


namespace trajectory_circle_equation_l150_150035

theorem trajectory_circle_equation :
  (∀ (x y : ℝ), dist (x, y) (0, 0) = 4 ↔ x^2 + y^2 = 16) :=  
sorry

end trajectory_circle_equation_l150_150035


namespace increasing_function_condition_l150_150970

variable {x : ℝ} {a : ℝ}

theorem increasing_function_condition (h : 0 < a) :
  (∀ x ≥ 1, deriv (λ x => x^3 - a * x) x ≥ 0) ↔ (0 < a ∧ a ≤ 3) :=
by
  sorry

end increasing_function_condition_l150_150970


namespace difference_q_r_share_l150_150079

theorem difference_q_r_share (p q r : ℕ) (x : ℕ) (h_ratio : p = 3 * x) (h_ratio_q : q = 7 * x) (h_ratio_r : r = 12 * x) (h_diff_pq : q - p = 4400) : q - r = 5500 :=
by
  sorry

end difference_q_r_share_l150_150079


namespace division_problem_l150_150251

theorem division_problem 
  (a b c d e f g h i : ℕ) 
  (h1 : a = 7) 
  (h2 : b = 9) 
  (h3 : c = 8) 
  (h4 : d = 1) 
  (h5 : e = 2) 
  (h6 : f = 3) 
  (h7 : g = 4) 
  (h8 : h = 6) 
  (h9 : i = 0) 
  : 7981 / 23 = 347 := 
by 
  sorry

end division_problem_l150_150251


namespace cosine_neg_alpha_l150_150384

theorem cosine_neg_alpha (alpha : ℝ) (h : Real.sin (π/2 + alpha) = -3/5) : Real.cos (-alpha) = -3/5 :=
sorry

end cosine_neg_alpha_l150_150384


namespace additional_discount_percentage_l150_150294

-- Define constants representing the conditions
def price_shoes : ℝ := 200
def discount_shoes : ℝ := 0.30
def price_shirt : ℝ := 80
def number_shirts : ℕ := 2
def final_spent : ℝ := 285

-- Define the theorem to prove the additional discount percentage
theorem additional_discount_percentage :
  let discounted_shoes := price_shoes * (1 - discount_shoes)
  let total_before_additional_discount := discounted_shoes + number_shirts * price_shirt
  let additional_discount := total_before_additional_discount - final_spent
  (additional_discount / total_before_additional_discount) * 100 = 5 :=
by
  -- Lean proof goes here, but we'll skip it for now with sorry
  sorry

end additional_discount_percentage_l150_150294


namespace not_product_of_consecutive_integers_l150_150659

theorem not_product_of_consecutive_integers (n k : ℕ) (hn : n > 0) (hk : k > 0) :
  ∀ (m : ℕ), 2 * (n ^ k) ^ 3 + 4 * (n ^ k) + 10 ≠ m * (m + 1) := by
sorry

end not_product_of_consecutive_integers_l150_150659


namespace area_to_be_painted_l150_150671

def wall_height : ℕ := 8
def wall_length : ℕ := 15
def glass_painting_height : ℕ := 3
def glass_painting_length : ℕ := 5

theorem area_to_be_painted :
  (wall_height * wall_length) - (glass_painting_height * glass_painting_length) = 105 := by
  sorry

end area_to_be_painted_l150_150671


namespace bc_over_ad_l150_150740

noncomputable def a : ℝ := 32 / 3
noncomputable def b : ℝ := 16 * Real.pi
noncomputable def c : ℝ := 24 * Real.pi
noncomputable def d : ℝ := 16 * Real.pi

theorem bc_over_ad : (b * c) / (a * d) = 9 / 4 := 
by 
  sorry

end bc_over_ad_l150_150740


namespace problem_statement_l150_150152

noncomputable def even_increasing (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x) ∧ ∀ x y, x < y → f x < f y

theorem problem_statement {f : ℝ → ℝ} (hf_even_incr : even_increasing f)
  (x1 x2 : ℝ) (hx1_gt_0 : x1 > 0) (hx2_lt_0 : x2 < 0) (hf_lt : f x1 < f x2) : x1 + x2 > 0 :=
sorry

end problem_statement_l150_150152


namespace sum_of_a_and_b_l150_150332

theorem sum_of_a_and_b (a b : ℝ) (h : a^2 + b^2 + 2 * a - 4 * b + 5 = 0) :
  a + b = 1 :=
sorry

end sum_of_a_and_b_l150_150332


namespace value_of_4m_plus_2n_l150_150207

-- Given that the equation 2kx + 2m = 6 - 2x + nk 
-- has a solution independent of k
theorem value_of_4m_plus_2n (m n : ℝ) 
  (h : ∃ x : ℝ, ∀ k : ℝ, 2 * k * x + 2 * m = 6 - 2 * x + n * k) : 
  4 * m + 2 * n = 12 :=
by
  sorry

end value_of_4m_plus_2n_l150_150207


namespace relation_among_a_b_c_l150_150452

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log (1/3) / Real.log 2
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relation_among_a_b_c : c > a ∧ a > b :=
by
  -- Prove that c > a and a > b
  sorry

end relation_among_a_b_c_l150_150452


namespace ceil_e_add_pi_l150_150389

theorem ceil_e_add_pi : ⌈Real.exp 1 + Real.pi⌉ = 6 := by
  sorry

end ceil_e_add_pi_l150_150389


namespace omitted_angle_measure_l150_150295

theorem omitted_angle_measure (initial_sum correct_sum : ℝ) (H_initial : initial_sum = 2083) (H_correct : correct_sum = 2160) :
  correct_sum - initial_sum = 77 :=
by sorry

end omitted_angle_measure_l150_150295


namespace cade_marbles_now_l150_150485

def original_marbles : ℝ := 87.0
def added_marbles : ℝ := 8.0
def total_marbles : ℝ := original_marbles + added_marbles

theorem cade_marbles_now : total_marbles = 95.0 :=
by
  sorry

end cade_marbles_now_l150_150485


namespace tangency_condition_l150_150667

def functions_parallel (a b c : ℝ) (f g: ℝ → ℝ)
       (parallel: ∀ x, f x = a * x + b ∧ g x = a * x + c) := 
  ∀ x, f x = a * x + b ∧ g x = a * x + c

theorem tangency_condition (a b c A : ℝ)
    (h_parallel : a ≠ 0)
    (h_tangency : (∀ x, (a * x + b)^2 = 7 * (a * x + c))) :
  A = 0 ∨ A = -7 :=
sorry

end tangency_condition_l150_150667


namespace outfits_count_l150_150264

def num_outfits (redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats : ℕ) : ℕ :=
  (redShirts * pairsPants * (greenHats + blueHats)) +
  (greenShirts * pairsPants * (redHats + blueHats)) +
  (blueShirts * pairsPants * (redHats + greenHats))

theorem outfits_count :
  ∀ (redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats : ℕ),
  redShirts = 4 → greenShirts = 4 → blueShirts = 4 →
  pairsPants = 7 →
  greenHats = 6 → redHats = 6 → blueHats = 6 →
  num_outfits redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats = 1008 :=
by
  intros redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats
  intros hredShirts hgreenShirts hblueShirts hpairsPants hgreenHats hredHats hblueHats
  rw [hredShirts, hgreenShirts, hblueShirts, hpairsPants, hgreenHats, hredHats, hblueHats]
  sorry

end outfits_count_l150_150264


namespace cups_of_baking_mix_planned_l150_150237

-- Definitions
def butter_per_cup := 2 -- 2 ounces of butter per 1 cup of baking mix
def coconut_oil_per_butter := 2 -- 2 ounces of coconut oil can substitute 2 ounces of butter
def butter_remaining := 4 -- Chef had 4 ounces of butter
def coconut_oil_used := 8 -- Chef used 8 ounces of coconut oil

-- Statement to be proven
theorem cups_of_baking_mix_planned : 
  (butter_remaining / butter_per_cup) + (coconut_oil_used / coconut_oil_per_butter) = 6 := 
by 
  sorry

end cups_of_baking_mix_planned_l150_150237


namespace find_angle_x_l150_150090

theorem find_angle_x (x : ℝ) (h1 : 3 * x + 2 * x = 90) : x = 18 :=
  by
    sorry

end find_angle_x_l150_150090


namespace power_comparison_l150_150733

theorem power_comparison :
  2 ^ 16 = 256 * 16 ^ 2 := 
by
  sorry

end power_comparison_l150_150733


namespace intersection_points_of_parametric_curve_l150_150128

def parametric_curve_intersection_points (t : ℝ) : Prop :=
  let x := t - 1
  let y := t + 2
  (x = -3 ∧ y = 0) ∨ (x = 0 ∧ y = 3)

theorem intersection_points_of_parametric_curve :
  ∃ t1 t2 : ℝ, parametric_curve_intersection_points t1 ∧ parametric_curve_intersection_points t2 := 
by
  sorry

end intersection_points_of_parametric_curve_l150_150128


namespace interest_rate_annual_l150_150112

theorem interest_rate_annual :
  ∃ R : ℝ, 
    (5000 * 2 * R / 100) + (3000 * 4 * R / 100) = 2640 ∧ 
    R = 12 :=
sorry

end interest_rate_annual_l150_150112


namespace smallest_lcm_of_4digit_gcd_5_l150_150941

theorem smallest_lcm_of_4digit_gcd_5 :
  ∃ (m n : ℕ), (1000 ≤ m ∧ m < 10000) ∧ (1000 ≤ n ∧ n < 10000) ∧ 
               m.gcd n = 5 ∧ m.lcm n = 203010 :=
by sorry

end smallest_lcm_of_4digit_gcd_5_l150_150941


namespace maximize_GDP_growth_l150_150284

def projectA_investment : ℕ := 20  -- million yuan
def projectB_investment : ℕ := 10  -- million yuan

def total_investment (a b : ℕ) : ℕ := a + b
def total_electricity (a b : ℕ) : ℕ := 20000 * a + 40000 * b
def total_jobs (a b : ℕ) : ℕ := 24 * a + 36 * b
def total_GDP_increase (a b : ℕ) : ℕ := 26 * a + 20 * b  -- scaled by 10 to avoid decimals

theorem maximize_GDP_growth : 
  total_investment projectA_investment projectB_investment ≤ 30 ∧
  total_electricity projectA_investment projectB_investment ≤ 1000000 ∧
  total_jobs projectA_investment projectB_investment ≥ 840 → 
  total_GDP_increase projectA_investment projectB_investment = 860 := 
by
  -- Proof would be provided here
  sorry

end maximize_GDP_growth_l150_150284


namespace value_of_b_l150_150638

theorem value_of_b (a b c y1 y2 y3 : ℝ)
( h1 : y1 = a + b + c )
( h2 : y2 = a - b + c )
( h3 : y3 = 4 * a + 2 * b + c )
( h4 : y1 - y2 = 8 )
( h5 : y3 = y1 + 2 )
: b = 4 :=
sorry

end value_of_b_l150_150638


namespace find_base_l150_150943

theorem find_base (b x y : ℝ) (h₁ : b^x * 4^y = 59049) (h₂ : x = 10) (h₃ : x - y = 10) : b = 3 :=
by
  sorry

end find_base_l150_150943


namespace topsoil_cost_proof_l150_150924

-- Definitions
def cost_per_cubic_foot : ℕ := 8
def cubic_feet_per_cubic_yard : ℕ := 27
def amount_in_cubic_yards : ℕ := 7

-- Theorem
theorem topsoil_cost_proof : cost_per_cubic_foot * cubic_feet_per_cubic_yard * amount_in_cubic_yards = 1512 := by
  -- proof logic goes here
  sorry

end topsoil_cost_proof_l150_150924


namespace neg_mul_reverses_inequality_l150_150624

theorem neg_mul_reverses_inequality (a b : ℝ) (h : a < b) : -3 * a > -3 * b :=
  sorry

end neg_mul_reverses_inequality_l150_150624


namespace max_tension_of_pendulum_l150_150962

theorem max_tension_of_pendulum 
  (m g L θ₀ : ℝ) 
  (h₀ : θ₀ < π / 2) 
  (T₀ : ℝ) 
  (no_air_resistance : true) 
  (no_friction : true) : 
  ∃ T_max, T_max = m * g * (3 - 2 * Real.cos θ₀) := 
by 
  sorry

end max_tension_of_pendulum_l150_150962


namespace find_f_of_one_third_l150_150707

-- Define g function according to given condition
def g (x : ℝ) : ℝ := 1 - x^2

-- Define f function according to given condition, valid for x ≠ 0
noncomputable def f (x : ℝ) : ℝ := (1 - x) / x

-- State the theorem we need to prove
theorem find_f_of_one_third : f (1 / 3) = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end find_f_of_one_third_l150_150707


namespace fixed_point_of_invariant_line_l150_150316

theorem fixed_point_of_invariant_line :
  ∀ (m : ℝ) (x y : ℝ), (3 * m + 4) * x + (5 - 2 * m) * y + 7 * m - 6 = 0 →
  (x = -1 ∧ y = 2) :=
by
  intro m x y h
  sorry

end fixed_point_of_invariant_line_l150_150316


namespace fewer_twos_for_100_l150_150419

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l150_150419


namespace calculation_correct_l150_150606

noncomputable def problem_calculation : ℝ :=
  4 * Real.sin (Real.pi / 3) - abs (-1) + (Real.sqrt 3 - 1)^0 + Real.sqrt 48

theorem calculation_correct : problem_calculation = 6 * Real.sqrt 3 :=
by
  sorry

end calculation_correct_l150_150606


namespace determinant_computation_l150_150288

variable (x y z w : ℝ)
variable (det : ℝ)
variable (H : x * w - y * z = 7)

theorem determinant_computation : 
  (x + z) * w - (y + 2 * w) * z = 7 - w * z := by
  sorry

end determinant_computation_l150_150288


namespace find_center_of_circle_l150_150720

-- Condition 1: The circle is tangent to the lines 3x - 4y = 12 and 3x - 4y = -48
def tangent_line1 (x y : ℝ) : Prop := 3 * x - 4 * y = 12
def tangent_line2 (x y : ℝ) : Prop := 3 * x - 4 * y = -48

-- Condition 2: The center of the circle lies on the line x - 2y = 0
def center_line (x y : ℝ) : Prop := x - 2 * y = 0

-- The center of the circle
def circle_center (x y : ℝ) : Prop := 
  tangent_line1 x y ∧ tangent_line2 x y ∧ center_line x y

-- Statement to prove
theorem find_center_of_circle : 
  circle_center (-18) (-9) := 
sorry

end find_center_of_circle_l150_150720


namespace g_neither_even_nor_odd_l150_150601

noncomputable def g (x : ℝ) : ℝ := ⌈x⌉ - 1 / 2

theorem g_neither_even_nor_odd :
  (¬ ∀ x, g x = g (-x)) ∧ (¬ ∀ x, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l150_150601


namespace carmen_reaches_alex_in_17_5_minutes_l150_150335

-- Define the conditions
variable (initial_distance : ℝ := 30) -- Initial distance in kilometers
variable (rate_of_closure : ℝ := 2) -- Rate at which the distance decreases in km per minute
variable (minutes_before_stop : ℝ := 10) -- Minutes before Alex stops

-- Define the speeds
variable (v_A : ℝ) -- Alex's speed in km per hour
variable (v_C : ℝ := 2 * v_A) -- Carmen's speed is twice Alex's speed
variable (total_closure_rate : ℝ := 120) -- Closure rate in km per hour (2 km per minute)

-- Main theorem to prove:
theorem carmen_reaches_alex_in_17_5_minutes : 
  ∃ (v_A v_C : ℝ), v_C = 2 * v_A ∧ v_C + v_A = total_closure_rate ∧ 
    (initial_distance - rate_of_closure * minutes_before_stop 
    - v_C * ((initial_distance - rate_of_closure * minutes_before_stop) / v_C) / 60 = 0) ∧ 
    (minutes_before_stop + ((initial_distance - rate_of_closure * minutes_before_stop) / v_C) * 60 = 17.5) :=
by
  sorry

end carmen_reaches_alex_in_17_5_minutes_l150_150335


namespace fraction_identity_l150_150965

variables {a b c x : ℝ}

theorem fraction_identity (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : c ≠ a) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a + 2 * b + 3 * c) / (a - b - 3 * c) = (b * (x + 2) + 3 * c) / (b * (x - 1) - 3 * c) :=
by {
  sorry
}

end fraction_identity_l150_150965


namespace record_expenditure_l150_150757

theorem record_expenditure (income_recording : ℤ) (expenditure_amount : ℤ) (h : income_recording = 20) : -expenditure_amount = -50 :=
by sorry

end record_expenditure_l150_150757


namespace circle_area_isosceles_triangle_l150_150508

theorem circle_area_isosceles_triangle (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 2) :
  ∃ R : ℝ, R = (81 / 32) * Real.pi :=
by sorry

end circle_area_isosceles_triangle_l150_150508


namespace find_a_value_l150_150788

theorem find_a_value
    (a : ℝ)
    (line : ∀ (x y : ℝ), 3 * x + y + a = 0)
    (circle : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y = 0) :
    a = 1 := sorry

end find_a_value_l150_150788


namespace passed_in_both_subjects_l150_150909

theorem passed_in_both_subjects (A B C : ℝ)
  (hA : A = 0.25)
  (hB : B = 0.48)
  (hC : C = 0.27) :
  1 - (A + B - C) = 0.54 := by
  sorry

end passed_in_both_subjects_l150_150909


namespace shaded_area_correct_l150_150043

noncomputable def shaded_area (s r_small : ℝ) : ℝ :=
  let hex_area := (3 * Real.sqrt 3 / 2) * s^2
  let semi_area := 6 * (1/2 * Real.pi * (s/2)^2)
  let small_circle_area := 6 * (Real.pi * (r_small)^2)
  hex_area - (semi_area + small_circle_area)

theorem shaded_area_correct : shaded_area 4 0.5 = 24 * Real.sqrt 3 - (27 * Real.pi / 2) := by
  sorry

end shaded_area_correct_l150_150043


namespace total_volume_of_cubes_l150_150019

theorem total_volume_of_cubes (Jim_cubes : Nat) (Jim_side_length : Nat) 
    (Laura_cubes : Nat) (Laura_side_length : Nat)
    (h1 : Jim_cubes = 7) (h2 : Jim_side_length = 3) 
    (h3 : Laura_cubes = 4) (h4 : Laura_side_length = 4) : 
    (Jim_cubes * Jim_side_length^3 + Laura_cubes * Laura_side_length^3 = 445) :=
by
  sorry

end total_volume_of_cubes_l150_150019


namespace max_marks_l150_150281

theorem max_marks (marks_obtained failed_by : ℝ) (passing_percentage : ℝ) (M : ℝ) : 
  marks_obtained = 180 ∧ failed_by = 40 ∧ passing_percentage = 0.45 ∧ (marks_obtained + failed_by = passing_percentage * M) → M = 489 :=
by 
  sorry

end max_marks_l150_150281


namespace problem_statement_l150_150969

theorem problem_statement (a b c d : ℤ) (h1 : a - b = -3) (h2 : c + d = 2) : (b + c) - (a - d) = 5 :=
by
  -- Proof steps skipped.
  sorry

end problem_statement_l150_150969


namespace fraction_spent_on_sandwich_l150_150587
    
theorem fraction_spent_on_sandwich 
  (x : ℚ)
  (h1 : 90 * x + 90 * (1/6) + 90 * (1/2) + 12 = 90) : 
  x = 1/5 :=
by
  sorry

end fraction_spent_on_sandwich_l150_150587


namespace units_digit_sum_l150_150420

theorem units_digit_sum (n : ℕ) (h : n > 0) : (35^n % 10) + (93^45 % 10) = 8 :=
by
  -- Since the units digit of 35^n is always 5 
  have h1 : 35^n % 10 = 5 := sorry
  -- Since the units digit of 93^45 is 3 (since 45 mod 4 = 1 and the pattern repeats every 4),
  have h2 : 93^45 % 10 = 3 := sorry
  -- Therefore, combining the units digits
  calc
    (35^n % 10) + (93^45 % 10)
    = 5 + 3 := by rw [h1, h2]
    _ = 8 := by norm_num

end units_digit_sum_l150_150420


namespace max_product_of_sum_2020_l150_150024

/--
  Prove that the maximum product of two integers whose sum is 2020 is 1020100.
-/
theorem max_product_of_sum_2020 : 
  ∃ x : ℤ, (x + (2020 - x) = 2020) ∧ (x * (2020 - x) = 1020100) :=
by
  sorry

end max_product_of_sum_2020_l150_150024


namespace second_month_sale_l150_150016

theorem second_month_sale 
  (sale_1st: ℕ) (sale_3rd: ℕ) (sale_4th: ℕ) (sale_5th: ℕ) (sale_6th: ℕ) (avg_sale: ℕ)
  (h1: sale_1st = 5266) (h3: sale_3rd = 5864)
  (h4: sale_4th = 6122) (h5: sale_5th = 6588)
  (h6: sale_6th = 4916) (h_avg: avg_sale = 5750) :
  ∃ sale_2nd, (sale_1st + sale_2nd + sale_3rd + sale_4th + sale_5th + sale_6th) / 6 = avg_sale :=
by
  sorry

end second_month_sale_l150_150016


namespace four_digit_number_divisible_by_11_l150_150505

theorem four_digit_number_divisible_by_11 :
  ∃ (a b c d : ℕ), 
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
    a + b + c + d = 10 ∧ 
    (a + c) % 11 = (b + d) % 11 ∧
    (10 - a != 0 ∨ 10 - c != 0 ∨ 10 - b != 0 ∨ 10 - d != 0) := sorry

end four_digit_number_divisible_by_11_l150_150505


namespace ratio_of_stock_values_l150_150139

/-- Definitions and conditions -/
def value_expensive := 78
def shares_expensive := 14
def shares_other := 26
def total_assets := 2106

/-- The proof problem -/
theorem ratio_of_stock_values : 
  ∃ (V_other : ℝ), 26 * V_other = total_assets - (shares_expensive * value_expensive) ∧ 
  (value_expensive / V_other) = 2 :=
by
  sorry

end ratio_of_stock_values_l150_150139


namespace directrix_of_parabola_l150_150890

theorem directrix_of_parabola :
  ∀ (x : ℝ), y = x^2 / 4 → y = -1 :=
sorry

end directrix_of_parabola_l150_150890


namespace benny_seashells_l150_150381

-- Defining the conditions
def initial_seashells : ℕ := 66
def given_away_seashells : ℕ := 52

-- Statement of the proof problem
theorem benny_seashells : (initial_seashells - given_away_seashells) = 14 :=
by
  sorry

end benny_seashells_l150_150381


namespace range_of_a_l150_150393

def p (a : ℝ) : Prop := ∀ k : ℝ, ∃ x y : ℝ, (y = k * x + 1) ∧ (x^2 + (y^2) / a = 1)
def q (a : ℝ) : Prop := ∃ x0 : ℝ, 4^x0 - 2^x0 - a ≤ 0

theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) → -1/4 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l150_150393


namespace solve_congruence_y37_x3_11_l150_150197

theorem solve_congruence_y37_x3_11 (p : ℕ) (hp_pr : Nat.Prime p) (hp_le100 : p ≤ 100) : 
  ∃ (x y : ℕ), y^37 ≡ x^3 + 11 [MOD p] := 
sorry

end solve_congruence_y37_x3_11_l150_150197


namespace angle_measure_l150_150611

theorem angle_measure : 
  ∃ (x : ℝ), (x + (3 * x + 3) = 90) ∧ x = 21.75 := by
  sorry

end angle_measure_l150_150611


namespace smallest_n_for_sqrt_20n_int_l150_150270

theorem smallest_n_for_sqrt_20n_int (n : ℕ) (h : ∃ k : ℕ, 20 * n = k^2) : n = 5 :=
by sorry

end smallest_n_for_sqrt_20n_int_l150_150270


namespace find_number_of_cows_l150_150573

-- Definitions for the problem
def number_of_legs (cows chickens : ℕ) := 4 * cows + 2 * chickens
def twice_the_heads_plus_12 (cows chickens : ℕ) := 2 * (cows + chickens) + 12

-- Main statement to prove
theorem find_number_of_cows (h : ℕ) : ∃ c : ℕ, number_of_legs c h = twice_the_heads_plus_12 c h ∧ c = 6 := 
by
  -- Sorry is used as a placeholder for the proof
  sorry

end find_number_of_cows_l150_150573


namespace number_properties_l150_150744

def number : ℕ := 52300600

def position_of_2 : ℕ := 10^6

def value_of_2 : ℕ := 20000000

def position_of_5 : ℕ := 10^7

def value_of_5 : ℕ := 50000000

def read_number : String := "five hundred twenty-three million six hundred"

theorem number_properties : 
  position_of_2 = (10^6) ∧ value_of_2 = 20000000 ∧ 
  position_of_5 = (10^7) ∧ value_of_5 = 50000000 ∧ 
  read_number = "five hundred twenty-three million six hundred" :=
by sorry

end number_properties_l150_150744


namespace difference_value_l150_150857

theorem difference_value (N : ℝ) (h : 0.25 * N = 100) : N - (3/4) * N = 100 :=
by sorry

end difference_value_l150_150857


namespace factor_roots_l150_150820

noncomputable def checkRoots (a b c t : ℚ) : Prop :=
  a * t^2 + b * t + c = 0

theorem factor_roots (t : ℚ) :
  checkRoots 8 17 (-10) t ↔ t = 5/8 ∨ t = -2 := by
sorry

end factor_roots_l150_150820


namespace batsman_average_l150_150919

variable (x : ℝ)

theorem batsman_average (h1 : ∀ x, 11 * x + 55 = 12 * (x + 1)) : 
  x = 43 → (x + 1 = 44) :=
by
  sorry

end batsman_average_l150_150919


namespace theater_ticket_cost_l150_150829

theorem theater_ticket_cost
  (num_persons : ℕ) 
  (num_children : ℕ) 
  (num_adults : ℕ)
  (children_ticket_cost : ℕ)
  (total_receipts_cents : ℕ)
  (A : ℕ) :
  num_persons = 280 →
  num_children = 80 →
  children_ticket_cost = 25 →
  total_receipts_cents = 14000 →
  num_adults = num_persons - num_children →
  200 * A + (num_children * children_ticket_cost) = total_receipts_cents →
  A = 60 :=
by
  intros h_num_persons h_num_children h_children_ticket_cost h_total_receipts_cents h_num_adults h_eqn
  sorry

end theater_ticket_cost_l150_150829


namespace total_money_divided_l150_150227

theorem total_money_divided (x y : ℕ) (hx : x = 1000) (ratioxy : 2 * y = 8 * x) : x + y = 5000 := 
by
  sorry

end total_money_divided_l150_150227


namespace highest_value_of_a_l150_150476

theorem highest_value_of_a (a : ℕ) (h : 0 ≤ a ∧ a ≤ 9) : (365 * 10 ^ 3 + a * 10 ^ 2 + 16) % 8 = 0 → a = 8 := by
  sorry

end highest_value_of_a_l150_150476


namespace betty_cookies_brownies_l150_150473

theorem betty_cookies_brownies (cookies_per_day brownies_per_day initial_cookies initial_brownies days : ℕ) :
  cookies_per_day = 3 → brownies_per_day = 1 → initial_cookies = 60 → initial_brownies = 10 → days = 7 →
  initial_cookies - days * cookies_per_day - (initial_brownies - days * brownies_per_day) = 36 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end betty_cookies_brownies_l150_150473


namespace focus_of_given_parabola_is_correct_l150_150238

-- Define the problem conditions
def parabolic_equation (x y : ℝ) : Prop := y = 4 * x^2

-- Define what it means for a point to be the focus of the given parabola
def is_focus_of_parabola (x0 y0 : ℝ) : Prop := 
    x0 = 0 ∧ y0 = 1 / 16

-- Define the theorem to be proven
theorem focus_of_given_parabola_is_correct : 
  ∃ x0 y0, parabolic_equation x0 y0 ∧ is_focus_of_parabola x0 y0 :=
sorry

end focus_of_given_parabola_is_correct_l150_150238


namespace distance_equal_x_value_l150_150352

theorem distance_equal_x_value :
  (∀ P Q R : ℝ × ℝ × ℝ, P = (x, 2, 1) ∧ Q = (1, 1, 2) ∧ R = (2, 1, 1) →
  dist P Q = dist P R →
  x = 1) :=
by
  -- Define the points P, Q, R
  let P := (x, 2, 1)
  let Q := (1, 1, 2)
  let R := (2, 1, 1)

  -- Given the condition
  intro h
  sorry

end distance_equal_x_value_l150_150352


namespace g_675_eq_42_l150_150185

theorem g_675_eq_42 
  (g : ℕ → ℕ) 
  (h_mul : ∀ x y : ℕ, x > 0 → y > 0 → g (x * y) = g x + g y) 
  (h_g15 : g 15 = 18) 
  (h_g45 : g 45 = 24) : g 675 = 42 :=
by
  sorry

end g_675_eq_42_l150_150185


namespace incenter_correct_l150_150885

variable (P Q R : Type) [AddCommGroup P] [Module ℝ P]
variable (p q r : ℝ)
variable (P_vec Q_vec R_vec : P)

noncomputable def incenter_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ :=
  (p / (p + q + r), q / (p + q + r), r / (p + q + r))

theorem incenter_correct : 
  incenter_coordinates 8 10 6 = (1/3, 5/12, 1/4) := by
  sorry

end incenter_correct_l150_150885


namespace lcm_12_18_l150_150235

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l150_150235


namespace B_joined_with_54000_l150_150179

theorem B_joined_with_54000 :
  ∀ (x : ℕ),
    (36000 * 12) / (x * 4) = 2 → x = 54000 :=
by 
  intro x h
  sorry

end B_joined_with_54000_l150_150179


namespace jen_age_proof_l150_150918

variable (JenAge : ℕ) (SonAge : ℕ)

theorem jen_age_proof (h1 : SonAge = 16) (h2 : JenAge = 3 * SonAge - 7) : JenAge = 41 :=
by
  -- conditions
  rw [h1] at h2
  -- substitution and simplification
  have h3 : JenAge = 3 * 16 - 7 := h2
  norm_num at h3
  exact h3

end jen_age_proof_l150_150918


namespace smallest_number_l150_150704

theorem smallest_number (a b c : ℕ) (h1 : b = 29) (h2 : c = b + 7) (h3 : (a + b + c) / 3 = 30) : a = 25 :=
by
  sorry

end smallest_number_l150_150704


namespace cost_price_of_apple_l150_150255

theorem cost_price_of_apple (C : ℚ) (h1 : 19 = 5/6 * C) : C = 22.8 := by
  sorry

end cost_price_of_apple_l150_150255


namespace housewife_oil_cost_l150_150888

theorem housewife_oil_cost (P R M : ℝ) (hR : R = 45) (hReduction : (P - R) = (15 / 100) * P)
  (hMoreOil : M / P = M / R + 4) : M = 150.61 := 
by
  sorry

end housewife_oil_cost_l150_150888


namespace ryan_spanish_hours_l150_150466

theorem ryan_spanish_hours (S : ℕ) (h : 7 = S + 3) : S = 4 :=
sorry

end ryan_spanish_hours_l150_150466


namespace find_a_plus_b_l150_150896

def cubic_function (a b : ℝ) (x : ℝ) := x^3 - x^2 - a * x + b

def tangent_line (x : ℝ) := 2 * x + 1

theorem find_a_plus_b (a b : ℝ) 
  (h1 : tangent_line 0 = 1)
  (h2 : cubic_function a b 0 = 1)
  (h3 : deriv (cubic_function a b) 0 = 2) :
  a + b = -1 :=
by
  sorry

end find_a_plus_b_l150_150896


namespace jimin_initial_candies_l150_150198

theorem jimin_initial_candies : 
  let candies_given_to_yuna := 25
  let candies_given_to_sister := 13
  candies_given_to_yuna + candies_given_to_sister = 38 := 
  by 
    sorry

end jimin_initial_candies_l150_150198


namespace point_on_x_axis_right_of_origin_is_3_units_away_l150_150103

theorem point_on_x_axis_right_of_origin_is_3_units_away :
  ∃ (P : ℝ × ℝ), P.2 = 0 ∧ P.1 > 0 ∧ dist (P.1, P.2) (0, 0) = 3 ∧ P = (3, 0) := 
by
  sorry

end point_on_x_axis_right_of_origin_is_3_units_away_l150_150103


namespace tracy_sold_paintings_l150_150030

-- Definitions of conditions
def total_customers := 20
def first_group_customers := 4
def paintings_per_first_group_customer := 2
def second_group_customers := 12
def paintings_per_second_group_customer := 1
def third_group_customers := 4
def paintings_per_third_group_customer := 4

-- Statement of the problem
theorem tracy_sold_paintings :
  (first_group_customers * paintings_per_first_group_customer) +
  (second_group_customers * paintings_per_second_group_customer) +
  (third_group_customers * paintings_per_third_group_customer) = 36 :=
by
  sorry

end tracy_sold_paintings_l150_150030


namespace find_annual_interest_rate_l150_150668

/-- 
  Given:
  - Principal P = 10000
  - Interest I = 450
  - Time period T = 0.75 years

  Prove that the annual interest rate is 0.08.
-/
theorem find_annual_interest_rate (P I : ℝ) (T : ℝ) (hP : P = 10000) (hI : I = 450) (hT : T = 0.75) : 
  (I / (P * T) / T) = 0.08 :=
by
  sorry

end find_annual_interest_rate_l150_150668


namespace range_of_a_l150_150599

variable {α : Type}

def A (x : ℝ) : Prop := 1 ≤ x ∧ x < 5
def B (x a : ℝ) : Prop := -a < x ∧ x ≤ a + 3

theorem range_of_a (a : ℝ) :
  (∀ x, B x a → A x) → a ≤ -1 := by
  sorry

end range_of_a_l150_150599


namespace ratio_of_boys_to_girls_l150_150756

-- Define the given conditions and provable statement
theorem ratio_of_boys_to_girls (S G : ℕ) (h : (2/3 : ℚ) * G = (1/5 : ℚ) * S) : (S - G) * 3 = 7 * G :=
by
  -- This is a placeholder for solving the proof
  sorry

end ratio_of_boys_to_girls_l150_150756


namespace simplify_expression_l150_150498

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem simplify_expression :
  cube_root (8 + 27) * cube_root (8 + cube_root 27) = cube_root 385 :=
by
  sorry

end simplify_expression_l150_150498


namespace problem_1_and_2_problem_1_infinite_solutions_l150_150441

open Nat

theorem problem_1_and_2 (k : ℕ) (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^2 + b^2 + c^2 = k * a * b * c) →
  (k = 1 ∨ k = 3) :=
sorry

theorem problem_1_infinite_solutions (k : ℕ) (h_k : k = 1 ∨ k = 3) :
  ∃ (a_n b_n c_n : ℕ) (n : ℕ), 
  a_n > 0 ∧ b_n > 0 ∧ c_n > 0 ∧
  (a_n^2 + b_n^2 + c_n^2 = k * a_n * b_n * c_n) ∧
  ∀ x y : ℕ, (x = a_n ∧ y = b_n) ∨ (x = a_n ∧ y = c_n) ∨ (x = b_n ∧ y = c_n) →
    ∃ p q : ℕ, x * y = p^2 + q^2 :=
sorry

end problem_1_and_2_problem_1_infinite_solutions_l150_150441


namespace fraction_equality_l150_150212

def at_op (a b : ℕ) : ℕ := a * b - b^2 + b^3
def hash_op (a b : ℕ) : ℕ := a + b - a * b^2 + a * b^3

theorem fraction_equality : 
  ∀ (a b : ℕ), a = 7 → b = 3 → (at_op a b : ℚ) / (hash_op a b : ℚ) = 39 / 136 :=
by
  intros a b h_a h_b
  rw [h_a, h_b]
  sorry

end fraction_equality_l150_150212


namespace original_sandbox_capacity_l150_150823

theorem original_sandbox_capacity :
  ∃ (L W H : ℝ), 8 * (L * W * H) = 80 → L * W * H = 10 :=
by
  sorry

end original_sandbox_capacity_l150_150823


namespace arithmetic_sequence_properties_l150_150731

def a_n (n : ℕ) : ℤ := 2 * n + 1

def S_n (n : ℕ) : ℤ := n * (n + 2)

theorem arithmetic_sequence_properties : 
  (a_n 3 = 7) ∧ (a_n 5 + a_n 7 = 26) :=
by {
  -- Proof to be filled
  sorry
}

end arithmetic_sequence_properties_l150_150731


namespace sum_of_logs_in_acute_triangle_l150_150565

theorem sum_of_logs_in_acute_triangle (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) (hC : 0 < C ∧ C < π / 2) 
  (h_triangle : A + B + C = π) :
  (Real.log (Real.sin B) / Real.log (Real.sin A)) +
  (Real.log (Real.sin C) / Real.log (Real.sin B)) +
  (Real.log (Real.sin A) / Real.log (Real.sin C)) ≥ 3 := by
  sorry

end sum_of_logs_in_acute_triangle_l150_150565


namespace problem1_l150_150966

theorem problem1 (α β : ℝ) 
  (tan_sum : Real.tan (α + β) = 2 / 5) 
  (tan_diff : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := 
sorry

end problem1_l150_150966


namespace original_height_of_tree_l150_150853

theorem original_height_of_tree
  (current_height_in_inches : ℕ)
  (percent_taller : ℕ)
  (current_height_is_V := 180)
  (percent_taller_is_50 := 50) :
  (current_height_in_inches * 100) / (percent_taller + 100) / 12 = 10 := sorry

end original_height_of_tree_l150_150853


namespace residue_of_927_mod_37_l150_150109

-- Define the condition of the problem, which is the modulus and the number
def modulus : ℤ := 37
def number : ℤ := -927

-- Define the statement we need to prove: that the residue of -927 mod 37 is 35
theorem residue_of_927_mod_37 : (number % modulus + modulus) % modulus = 35 := by
  sorry

end residue_of_927_mod_37_l150_150109


namespace find_digit_A_l150_150802

theorem find_digit_A :
  ∃ A : ℕ, 
    2 * 10^6 + A * 10^5 + 9 * 10^4 + 9 * 10^3 + 5 * 10^2 + 6 * 10^1 + 1 = (3 * (523 + A)) ^ 2 
    ∧ A = 4 :=
by
  sorry

end find_digit_A_l150_150802


namespace least_three_digit_divisible_by_2_3_5_7_l150_150917

theorem least_three_digit_divisible_by_2_3_5_7 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ k, 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → n ≤ k) ∧
  (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) ∧ n = 210 :=
by sorry

end least_three_digit_divisible_by_2_3_5_7_l150_150917


namespace max_writers_and_editors_l150_150539

theorem max_writers_and_editors (total_people writers editors x : ℕ) (h_total_people : total_people = 100)
(h_writers : writers = 40) (h_editors : editors > 38) (h_both : 2 * x + (writers + editors - x) = total_people) :
x ≤ 21 := sorry

end max_writers_and_editors_l150_150539


namespace addition_problem_l150_150791

theorem addition_problem (m n p q : ℕ) (Hm : m = 2) (Hn : 2 + n + 7 + 5 = 20) (Hp : 1 + 6 + p + 8 = 24) (Hq : 3 + 2 + q = 12) (Hpositives : 0 < m ∧ 0 < n ∧ 0 < p ∧ 0 < q) :
  m + n + p + q = 24 :=
sorry

end addition_problem_l150_150791


namespace division_minutes_per_day_l150_150372

-- Define the conditions
def total_hours : ℕ := 5
def minutes_multiplication_per_day : ℕ := 10
def days_total : ℕ := 10

-- Convert hours to minutes
def total_minutes : ℕ := total_hours * 60

-- Total minutes spent on multiplication
def total_minutes_multiplication : ℕ := minutes_multiplication_per_day * days_total

-- Total minutes spent on division
def total_minutes_division : ℕ := total_minutes - total_minutes_multiplication

-- Minutes spent on division per day
def minutes_division_per_day : ℕ := total_minutes_division / days_total

-- The theorem to prove
theorem division_minutes_per_day : minutes_division_per_day = 20 := by
  sorry

end division_minutes_per_day_l150_150372


namespace ζ_sum_8_l150_150042

open Complex

def ζ1 : ℂ := sorry
def ζ2 : ℂ := sorry
def ζ3 : ℂ := sorry

def e1 := ζ1 + ζ2 + ζ3
def e2 := ζ1 * ζ2 + ζ2 * ζ3 + ζ3 * ζ1
def e3 := ζ1 * ζ2 * ζ3

axiom h1 : e1 = 2
axiom h2 : e1^2 - 2 * e2 = 8
axiom h3 : (e1^2 - 2 * e2)^2 - 2 * (e2^2 - 2 * e1 * e3) = 26

theorem ζ_sum_8 : ζ1^8 + ζ2^8 + ζ3^8 = 219 :=
by {
  -- The proof goes here, omitting solution steps as instructed.
  sorry
}

end ζ_sum_8_l150_150042


namespace intersection_sets_l150_150141

def setA : Set ℝ := { x | -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3 }
def setB : Set ℝ := { x | (x - 3) / (2 * x) ≤ 0 }

theorem intersection_sets (x : ℝ) : x ∈ setA ∧ x ∈ setB ↔ 0 < x ∧ x ≤ 1 := by
  sorry

end intersection_sets_l150_150141


namespace angle_equiv_terminal_side_l150_150048

theorem angle_equiv_terminal_side (θ : ℤ) : 
  let θ_deg := (750 : ℕ)
  let reduced_angle := θ_deg % 360
  0 ≤ reduced_angle ∧ reduced_angle < 360 ∧ reduced_angle = 30:=
by
  sorry

end angle_equiv_terminal_side_l150_150048


namespace largest_digit_M_divisible_by_6_l150_150699

theorem largest_digit_M_divisible_by_6 (M : ℕ) (h1 : 5172 * 10 + M % 2 = 0) (h2 : (5 + 1 + 7 + 2 + M) % 3 = 0) : M = 6 := by
  sorry

end largest_digit_M_divisible_by_6_l150_150699


namespace gen_term_seq_l150_150694

open Nat

def seq (a : ℕ → ℕ) : Prop := 
a 1 = 1 ∧ (∀ n : ℕ, n ≠ 0 → a (n + 1) = 2 * a n - 3)

theorem gen_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n : ℕ, a n = 3 - 2^n :=
by
  sorry

end gen_term_seq_l150_150694


namespace min_value_l150_150978

noncomputable def min_value_of_expression (a b: ℝ) :=
    a > 0 ∧ b > 0 ∧ a + b = 1 → (∃ (m : ℝ), (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) ∧ m = 3 + 2 * Real.sqrt 2)

theorem min_value (a b: ℝ) (h₀: a > 0) (h₁: b > 0) (h₂: a + b = 1) :
    ∃ m, (∀ x y, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) ∧ m = 3 + 2 * Real.sqrt 2 := 
by
    sorry

end min_value_l150_150978


namespace arithmetic_sequence_solution_l150_150062

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- The sequence is arithmetic
def is_arithmetic_sequence : Prop :=
  ∀ n, a (n+1) = a n + d

-- The given condition a_3 + a_5 = 12 - a_7
def condition : Prop :=
  a 3 + a 5 = 12 - a 7

-- The proof statement
theorem arithmetic_sequence_solution 
  (h_arith : is_arithmetic_sequence a d) 
  (h_cond : condition a): a 1 + a 9 = 8 :=
sorry

end arithmetic_sequence_solution_l150_150062


namespace right_triangle_power_inequality_l150_150504

theorem right_triangle_power_inequality {a b c x : ℝ} (hpos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a^2 = b^2 + c^2) (h_longest : a > b ∧ a > c) :
  (x > 2) → (a^x > b^x + c^x) :=
by sorry

end right_triangle_power_inequality_l150_150504


namespace jam_cost_is_162_l150_150122

theorem jam_cost_is_162 (N B J : ℕ) (h1 : N > 1) (h2 : 4 * B + 6 * J = 39) (h3 : N = 9) : 
  6 * N * J = 162 := 
by sorry

end jam_cost_is_162_l150_150122


namespace no_such_reals_exist_l150_150213

-- Define the existence of distinct real numbers such that the given condition holds
theorem no_such_reals_exist :
  ¬ ∃ x y z : ℝ, (x ≠ y) ∧ (y ≠ z) ∧ (z ≠ x) ∧ 
  (1 / (x^2 - y^2) + 1 / (y^2 - z^2) + 1 / (z^2 - x^2) = 0) :=
by
  -- Placeholder for proof
  sorry

end no_such_reals_exist_l150_150213


namespace find_value_of_expression_l150_150307

noncomputable def root_finder (a b c : ℝ) : Prop :=
  a^3 - 30*a^2 + 65*a - 42 = 0 ∧
  b^3 - 30*b^2 + 65*b - 42 = 0 ∧
  c^3 - 30*c^2 + 65*c - 42 = 0

theorem find_value_of_expression {a b c : ℝ} (h : root_finder a b c) :
  a + b + c = 30 ∧ ab + bc + ca = 65 ∧ abc = 42 → 
  (a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)) = 770/43 :=
by
  sorry

end find_value_of_expression_l150_150307


namespace parabola_hyperbola_tangent_l150_150721

theorem parabola_hyperbola_tangent : ∃ m : ℝ, 
  (∀ x y : ℝ, y = x^2 - 2 * x + 2 → y^2 - m * x^2 = 1) ↔ m = 1 :=
by
  sorry

end parabola_hyperbola_tangent_l150_150721


namespace no_intersection_points_l150_150241

def intersection_points_eq_zero : Prop :=
∀ x y : ℝ, (y = abs (3 * x + 6)) ∧ (y = -abs (4 * x - 3)) → false

theorem no_intersection_points :
  intersection_points_eq_zero :=
by
  intro x y h
  cases h
  sorry

end no_intersection_points_l150_150241


namespace fiftieth_statement_l150_150512

-- Define the types
inductive Inhabitant : Type
| knight : Inhabitant
| liar : Inhabitant

-- Define the function telling the statement
def statement (inhabitant : Inhabitant) : String :=
  match inhabitant with
  | Inhabitant.knight => "Knight"
  | Inhabitant.liar => "Liar"

-- Define the condition: knights tell the truth and liars lie
def tells_truth (inhabitant : Inhabitant) (statement_about_neighbor : String) : Prop :=
  match inhabitant with
  | Inhabitant.knight => statement_about_neighbor = "Knight"
  | Inhabitant.liar => statement_about_neighbor ≠ "Knight"

-- Define a function that determines what each inhabitant says about their right-hand neighbor
def what_they_say (idx : ℕ) : String :=
  if idx % 2 = 0 then "Liar" else "Knight"

-- Define the inhabitant pattern
def inhabitant_at (idx : ℕ) : Inhabitant :=
  if idx % 2 = 0 then Inhabitant.liar else Inhabitant.knight

-- The main theorem statement
theorem fiftieth_statement : tells_truth (inhabitant_at 49) (what_they_say 50) :=
by 
  -- This proof outlines the theorem statement only
  sorry

end fiftieth_statement_l150_150512


namespace sqrt_31_minus_2_in_range_l150_150951

-- Defining the conditions based on the problem statements
def five_squared : ℤ := 5 * 5
def six_squared : ℤ := 6 * 6
def thirty_one : ℤ := 31

theorem sqrt_31_minus_2_in_range : 
  (5 * 5 < thirty_one) ∧ (thirty_one < 6 * 6) →
  3 < (Real.sqrt thirty_one) - 2 ∧ (Real.sqrt thirty_one) - 2 < 4 :=
by
  sorry

end sqrt_31_minus_2_in_range_l150_150951


namespace number_divided_by_21_l150_150094

theorem number_divided_by_21 (x : ℝ) (h : 6000 - (x / 21.0) = 5995) : x = 105 :=
by
  sorry

end number_divided_by_21_l150_150094


namespace m_le_three_l150_150771

-- Definitions
def setA (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def setB (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

-- Theorem statement
theorem m_le_three (m : ℝ) : (∀ x : ℝ, setB m x → setA x) → m ≤ 3 := by
  sorry

end m_le_three_l150_150771


namespace angle_A_range_l150_150577

-- Definitions from the conditions
variable (A B C : ℝ)
variable (a b c : ℝ)
axiom triangle_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a
axiom longest_side_a : a > b ∧ a > c
axiom inequality_a : a^2 < b^2 + c^2

-- Target proof statement
theorem angle_A_range (triangle_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (longest_side_a : a > b ∧ a > c)
  (inequality_a : a^2 < b^2 + c^2) : 60 < A ∧ A < 90 := 
sorry

end angle_A_range_l150_150577


namespace chocolates_received_per_boy_l150_150350

theorem chocolates_received_per_boy (total_chocolates : ℕ) (total_people : ℕ)
(boys : ℕ) (girls : ℕ) (chocolates_per_girl : ℕ)
(h_total_chocolates : total_chocolates = 3000)
(h_total_people : total_people = 120)
(h_boys : boys = 60)
(h_girls : girls = 60)
(h_chocolates_per_girl : chocolates_per_girl = 3) :
  (total_chocolates - (girls * chocolates_per_girl)) / boys = 47 :=
by
  sorry

end chocolates_received_per_boy_l150_150350


namespace remainder_of_sum_div_17_l150_150097

-- Definitions based on the conditions from the problem
def numbers : List ℕ := [82, 83, 84, 85, 86, 87, 88, 89]
def divisor : ℕ := 17

-- The theorem statement proving the result
theorem remainder_of_sum_div_17 : List.sum numbers % divisor = 0 := by
  sorry

end remainder_of_sum_div_17_l150_150097


namespace accommodation_arrangements_l150_150848

-- Given conditions
def triple_room_capacity : Nat := 3
def double_room_capacity : Nat := 2
def single_room_capacity : Nat := 1
def num_adult_men : Nat := 4
def num_little_boys : Nat := 2

-- Ensuring little boys are always accompanied by an adult and all rooms are occupied
def is_valid_arrangement (triple double single : Nat × Nat) : Prop :=
  let (triple_adults, triple_boys) := triple
  let (double_adults, double_boys) := double
  let (single_adults, single_boys) := single
  triple_adults + double_adults + single_adults = num_adult_men ∧
  triple_boys + double_boys + single_boys = num_little_boys ∧
  triple = (triple_room_capacity, num_little_boys) ∨
  (triple = (triple_room_capacity, 1) ∧ double = (double_room_capacity, 1)) ∧
  triple_adults + triple_boys = triple_room_capacity ∧
  double_adults + double_boys = double_room_capacity ∧
  single_adults + single_boys = single_room_capacity

-- Main theorem statement
theorem accommodation_arrangements : ∃ (triple double single : Nat × Nat),
  is_valid_arrangement triple double single ∧
  -- The number 36 comes from the correct answer in the solution steps part b)
  (triple.1 + double.1 + single.1 = 4 ∧ triple.2 + double.2 + single.2 = 2) :=
sorry

end accommodation_arrangements_l150_150848


namespace solve_nat_numbers_equation_l150_150708

theorem solve_nat_numbers_equation (n k l m : ℕ) (h_l : l > 1) 
  (h_eq : (1 + n^k)^l = 1 + n^m) : (n = 2) ∧ (k = 1) ∧ (l = 2) ∧ (m = 3) := 
by
  sorry

end solve_nat_numbers_equation_l150_150708


namespace initial_notebooks_is_10_l150_150883

-- Define the conditions
def ordered_notebooks := 6
def lost_notebooks := 2
def current_notebooks := 14

-- Define the initial number of notebooks
def initial_notebooks (N : ℕ) :=
  N + ordered_notebooks - lost_notebooks = current_notebooks

-- The proof statement
theorem initial_notebooks_is_10 : initial_notebooks 10 :=
by
  sorry

end initial_notebooks_is_10_l150_150883


namespace cell_division_50_closest_to_10_15_l150_150580

theorem cell_division_50_closest_to_10_15 :
  10^14 < 2^50 ∧ 2^50 < 10^16 :=
sorry

end cell_division_50_closest_to_10_15_l150_150580


namespace rice_mixing_ratio_l150_150385

-- Definitions based on conditions
def rice_1_price : ℝ := 6
def rice_2_price : ℝ := 8.75
def mixture_price : ℝ := 7.50

-- Proof of the required ratio
theorem rice_mixing_ratio (x y : ℝ) (h : (rice_1_price * x + rice_2_price * y) / (x + y) = mixture_price) :
  y / x = 6 / 5 :=
by 
  sorry

end rice_mixing_ratio_l150_150385


namespace math_problem_l150_150161

theorem math_problem : 1012^2 - 992^2 - 1009^2 + 995^2 = 12024 := sorry

end math_problem_l150_150161


namespace sum_of_consecutive_integers_eq_pow_of_two_l150_150804

theorem sum_of_consecutive_integers_eq_pow_of_two (n : ℕ) : 
  (∀ a b : ℕ, a < b → 2 * n ≠ (a + b) * (b - a + 1)) ↔ ∃ k : ℕ, n = 2 ^ k := 
sorry

end sum_of_consecutive_integers_eq_pow_of_two_l150_150804


namespace astronaut_total_days_l150_150194

-- Definitions of the regular and leap seasons.
def regular_season_days := 49
def leap_season_days := 51

-- Definition of the number of days in different types of years.
def days_in_regular_year := 2 * regular_season_days + 3 * leap_season_days
def days_in_first_3_years := 2 * regular_season_days + 3 * (leap_season_days + 1)
def days_in_years_7_to_9 := 2 * regular_season_days + 3 * (leap_season_days + 2)

-- Calculation for visits.
def first_visit := regular_season_days
def second_visit := 2 * regular_season_days + 3 * (leap_season_days + 1)
def third_visit := 3 * (2 * regular_season_days + 3 * (leap_season_days + 1))
def fourth_visit := 4 * days_in_regular_year + 3 * days_in_first_3_years + 3 * days_in_years_7_to_9

-- Total days spent.
def total_days := first_visit + second_visit + third_visit + fourth_visit

-- The proof statement.
theorem astronaut_total_days : total_days = 3578 :=
by
  -- We place a sorry here to skip the proof.
  sorry

end astronaut_total_days_l150_150194


namespace exists_prime_with_composite_sequence_l150_150691

theorem exists_prime_with_composite_sequence (n : ℕ) (hn : n ≠ 0) : 
  ∃ p : ℕ, Nat.Prime p ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ¬ Nat.Prime (p + k) :=
sorry

end exists_prime_with_composite_sequence_l150_150691


namespace correct_statement_l150_150762

theorem correct_statement :
  (Real.sqrt (9 / 16) = 3 / 4) :=
by
  sorry

end correct_statement_l150_150762


namespace electric_car_charging_cost_l150_150162

/-- The fractional equation for the given problem,
    along with the correct solution for the average charging cost per kilometer. -/
theorem electric_car_charging_cost (
    x : ℝ
) : 
    (200 / x = 4 * (200 / (x + 0.6))) → x = 0.2 :=
by
  intros h_eq
  sorry

end electric_car_charging_cost_l150_150162


namespace determine_range_of_b_l150_150687

noncomputable def f (b x : ℝ) : ℝ := (Real.log x + (x - b) ^ 2) / x
noncomputable def f'' (b x : ℝ) : ℝ := (2 * Real.log x - 2) / x ^ 3

theorem determine_range_of_b (b : ℝ) (h : ∃ x ∈ Set.Icc (1 / 2) 2, f b x > -x * f'' b x) :
  b < 9 / 4 :=
by
  sorry

end determine_range_of_b_l150_150687


namespace cubic_polynomial_inequality_l150_150259

theorem cubic_polynomial_inequality
  (A B C : ℝ)
  (h : ∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    A = -(a + b + c) ∧ B = ab + bc + ca ∧ C = -abc) :
  A^2 + B^2 + 18 * C > 0 :=
by
  sorry

end cubic_polynomial_inequality_l150_150259


namespace charlie_and_elle_crayons_l150_150110

theorem charlie_and_elle_crayons :
  (∃ (Lizzie Bobbie Billie Charlie Dave Elle : ℕ),
  Billie = 18 ∧
  Bobbie = 3 * Billie ∧
  Lizzie = Bobbie / 2 ∧
  Charlie = 2 * Lizzie ∧
  Dave = 4 * Billie ∧
  Elle = (Bobbie + Dave) / 2 ∧
  Charlie + Elle = 117) :=
sorry

end charlie_and_elle_crayons_l150_150110


namespace area_of_parallelogram_l150_150556

theorem area_of_parallelogram
  (angle_deg : ℝ := 150)
  (side1 : ℝ := 10)
  (side2 : ℝ := 20)
  (adj_angle_deg : ℝ := 180 - angle_deg)
  (angle_rad : ℝ := (adj_angle_deg * Real.pi) / 180) :
  let height := side1 * (Real.sqrt 3 / 2)
  let area := side2 * height
  area = 100 * Real.sqrt 3 :=
by
  /- Proof skipped -/
  sorry

end area_of_parallelogram_l150_150556


namespace binomial_coefficients_sum_l150_150210

noncomputable def f (m n : ℕ) : ℕ :=
  (Nat.choose 6 m) * (Nat.choose 4 n)

theorem binomial_coefficients_sum : 
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := 
by
  sorry

end binomial_coefficients_sum_l150_150210


namespace Emily_sixth_score_l150_150557

theorem Emily_sixth_score :
  let scores := [91, 94, 88, 90, 101]
  let current_sum := scores.sum
  let desired_average := 95
  let num_quizzes := 6
  let total_score_needed := num_quizzes * desired_average
  let sixth_score := total_score_needed - current_sum
  sixth_score = 106 :=
by
  sorry

end Emily_sixth_score_l150_150557


namespace count_bottom_right_arrows_l150_150023

/-!
# Problem Statement
Each blank cell on the edge is to be filled with an arrow. The number in each square indicates the number of arrows pointing to that number. The arrows can point in the following directions: up, down, left, right, top-left, top-right, bottom-left, and bottom-right. Each arrow must point to a number. Figure 3 is provided and based on this, determine the number of arrows pointing to the bottom-right direction.
-/

def bottom_right_arrows_count : Nat :=
  2

theorem count_bottom_right_arrows :
  bottom_right_arrows_count = 2 :=
by
  sorry

end count_bottom_right_arrows_l150_150023


namespace Xiaogang_raised_arm_exceeds_head_l150_150574

theorem Xiaogang_raised_arm_exceeds_head :
  ∀ (height shadow_no_arm shadow_with_arm : ℝ),
    height = 1.7 → shadow_no_arm = 0.85 → shadow_with_arm = 1.1 →
    (height / shadow_no_arm) = ((shadow_with_arm - shadow_no_arm) * (height / shadow_no_arm)) →
    shadow_with_arm - shadow_no_arm = 0.25 →
    ((height / shadow_no_arm) * 0.25) = 0.5 :=
by
  intros height shadow_no_arm shadow_with_arm h_eq1 h_eq2 h_eq3 h_eq4 h_eq5
  sorry

end Xiaogang_raised_arm_exceeds_head_l150_150574


namespace math_problem_l150_150464

theorem math_problem (m : ℝ) (h : m^2 - m = 2) : (m - 1)^2 + (m + 2) * (m - 2) = 1 := 
by sorry

end math_problem_l150_150464


namespace max_zeros_in_product_of_three_natural_numbers_sum_1003_l150_150716

theorem max_zeros_in_product_of_three_natural_numbers_sum_1003 :
  ∀ (a b c : ℕ), a + b + c = 1003 →
    ∃ N, (a * b * c) % (10^N) = 0 ∧ N = 7 := by
  sorry

end max_zeros_in_product_of_three_natural_numbers_sum_1003_l150_150716


namespace find_larger_number_l150_150262

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1375) (h2 : L = 6 * S + 15) : L = 1647 :=
by
  -- proof to be filled
  sorry

end find_larger_number_l150_150262


namespace icosahedron_path_count_l150_150543

noncomputable def icosahedron_paths : ℕ := 
  sorry

theorem icosahedron_path_count : icosahedron_paths = 45 :=
  sorry

end icosahedron_path_count_l150_150543


namespace sqrt7_problem_l150_150976

theorem sqrt7_problem (x y : ℝ) (h1 : 2 < Real.sqrt 7) (h2 : Real.sqrt 7 < 3) (hx : x = 2) (hy : y = Real.sqrt 7 - 2) :
  (x + Real.sqrt 7) * y = 3 :=
by
  sorry

end sqrt7_problem_l150_150976


namespace equation_1_solution_equation_2_solution_l150_150095

theorem equation_1_solution (x : ℝ) (h : (2 * x - 3)^2 = 9 * x^2) : x = 3 / 5 ∨ x = -3 :=
sorry

theorem equation_2_solution (x : ℝ) (h : 2 * x * (x - 2) + x = 2) : x = 2 ∨ x = -1 / 2 :=
sorry

end equation_1_solution_equation_2_solution_l150_150095


namespace fraction_of_girls_l150_150220

variable (T G B : ℕ) -- The total number of students, number of girls, and number of boys
variable (x : ℚ) -- The fraction of the number of girls

-- Definitions based on the given conditions
def fraction_condition : Prop := x * G = (1/6) * T
def ratio_condition : Prop := (B : ℚ) / (G : ℚ) = 2
def total_students : Prop := T = B + G

-- The statement we need to prove
theorem fraction_of_girls (h1 : fraction_condition T G x)
                          (h2 : ratio_condition B G)
                          (h3 : total_students T G B):
  x = 1/2 :=
by
  sorry

end fraction_of_girls_l150_150220


namespace carla_initial_marbles_l150_150277

theorem carla_initial_marbles (total_marbles : ℕ) (bought_marbles : ℕ) (initial_marbles : ℕ) 
  (h1 : total_marbles = 187) (h2 : bought_marbles = 134) (h3 : total_marbles = initial_marbles + bought_marbles) : 
  initial_marbles = 53 := 
sorry

end carla_initial_marbles_l150_150277


namespace intersecting_points_are_same_l150_150592

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (3, -2)
def radius1 : ℝ := 5

def center2 : ℝ × ℝ := (3, 6)
def radius2 : ℝ := 3

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := (x - center1.1)^2 + (y + center1.2)^2 = radius1^2
def circle2 (x y : ℝ) : Prop := (x - center2.1)^2 + (y - center2.2)^2 = radius2^2

-- Prove that points C and D coincide
theorem intersecting_points_are_same : ∃ x y, circle1 x y ∧ circle2 x y → (0 = 0) :=
by
  sorry

end intersecting_points_are_same_l150_150592


namespace sum_of_squares_inequality_l150_150765

theorem sum_of_squares_inequality (a b c : ℝ) : 
  a^2 + b^2 + c^2 ≥ ab + bc + ca :=
by
  sorry

end sum_of_squares_inequality_l150_150765


namespace uncle_dave_ice_cream_sandwiches_l150_150078

theorem uncle_dave_ice_cream_sandwiches (n : ℕ) (s : ℕ) (total : ℕ) 
  (h1 : n = 11) (h2 : s = 13) (h3 : total = n * s) : total = 143 := by
  sorry

end uncle_dave_ice_cream_sandwiches_l150_150078


namespace largest_of_numbers_l150_150590

theorem largest_of_numbers (a b c d : ℝ) 
  (ha : a = 0) (hb : b = -1) (hc : c = 3.5) (hd : d = Real.sqrt 13) : 
  ∃ x, x = Real.sqrt 13 ∧ (x > a) ∧ (x > b) ∧ (x > c) ∧ (x > d) :=
by
  sorry

end largest_of_numbers_l150_150590


namespace find_a_l150_150296

theorem find_a (x y a : ℝ) (h1 : x + 3 * y = 4 - a) 
  (h2 : x - y = -3 * a) (h3 : x + y = 0) : a = 1 :=
sorry

end find_a_l150_150296


namespace geometric_sequence_b_l150_150146

theorem geometric_sequence_b (b : ℝ) (r : ℝ) (hb : b > 0)
  (h1 : 10 * r = b)
  (h2 : b * r = 10 / 9)
  (h3 : (10 / 9) * r = 10 / 81) :
  b = 10 :=
sorry

end geometric_sequence_b_l150_150146


namespace inequality_proof_l150_150642

theorem inequality_proof {a b c d e f : ℝ} (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
sorry

end inequality_proof_l150_150642


namespace carlos_jogged_distance_l150_150985

def carlos_speed := 4 -- Carlos's speed in miles per hour
def jogging_time := 2 -- Time in hours

theorem carlos_jogged_distance : carlos_speed * jogging_time = 8 :=
by
  sorry

end carlos_jogged_distance_l150_150985


namespace tangent_alpha_l150_150430

open Real

noncomputable def a (α : ℝ) : ℝ × ℝ := (sin α, 2)
noncomputable def b (α : ℝ) : ℝ × ℝ := (-cos α, 1)

theorem tangent_alpha (α : ℝ) (h : ∀ k : ℝ, a α = (k • b α)) : tan α = -2 := by
  have h1 : sin α / -cos α = 2 := by sorry
  have h2 : tan α = -2 := by sorry
  exact h2

end tangent_alpha_l150_150430


namespace candles_must_be_odd_l150_150484

theorem candles_must_be_odd (n k : ℕ) (h : n * k = (n * (n + 1)) / 2) : n % 2 = 1 :=
by
  -- Given that the total burn time for all n candles = k * n
  -- And the sum of the first n natural numbers = (n * (n + 1)) / 2
  -- We have the hypothesis h: n * k = (n * (n + 1)) / 2
  -- We need to prove that n must be odd
  sorry

end candles_must_be_odd_l150_150484


namespace clay_capacity_second_box_l150_150633

-- Define the dimensions and clay capacity of the first box
def height1 : ℕ := 4
def width1 : ℕ := 2
def length1 : ℕ := 3
def clay1 : ℕ := 24

-- Define the dimensions of the second box
def height2 : ℕ := 3 * height1
def width2 : ℕ := 2 * width1
def length2 : ℕ := length1

-- The volume relation
def volume_relation (height width length clay: ℕ) : ℕ :=
  height * width * length * clay

theorem clay_capacity_second_box (height1 width1 length1 clay1 : ℕ) (height2 width2 length2 : ℕ) :
  height1 = 4 →
  width1 = 2 →
  length1 = 3 →
  clay1 = 24 →
  height2 = 3 * height1 →
  width2 = 2 * width1 →
  length2 = length1 →
  volume_relation height2 width2 length2 1 = 6 * volume_relation height1 width1 length1 1 →
  volume_relation height2 width2 length2 clay1 / volume_relation height1 width1 length1 1 = 144 :=
by
  intros h1 w1 l1 c1 h2 w2 l2 vol_rel
  sorry

end clay_capacity_second_box_l150_150633


namespace am_gm_inequality_l150_150876

theorem am_gm_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a^3 + b^3 + a + b ≥ 4 * a * b :=
by
  sorry

end am_gm_inequality_l150_150876


namespace difference_length_breadth_l150_150153

theorem difference_length_breadth (B L A : ℕ) (h1 : B = 11) (h2 : A = 21 * B) (h3 : A = L * B) :
  L - B = 10 :=
by
  sorry

end difference_length_breadth_l150_150153


namespace bmw_cars_sold_l150_150005

def percentage_non_bmw (ford_pct nissan_pct chevrolet_pct : ℕ) : ℕ :=
  ford_pct + nissan_pct + chevrolet_pct

def percentage_bmw (total_pct non_bmw_pct : ℕ) : ℕ :=
  total_pct - non_bmw_pct

def number_of_bmws (total_cars bmw_pct : ℕ) : ℕ :=
  (total_cars * bmw_pct) / 100

theorem bmw_cars_sold (total_cars ford_pct nissan_pct chevrolet_pct : ℕ)
  (h_total_cars : total_cars = 300)
  (h_ford_pct : ford_pct = 20)
  (h_nissan_pct : nissan_pct = 25)
  (h_chevrolet_pct : chevrolet_pct = 10) :
  number_of_bmws total_cars (percentage_bmw 100 (percentage_non_bmw ford_pct nissan_pct chevrolet_pct)) = 135 := by
  sorry

end bmw_cars_sold_l150_150005


namespace time_with_family_l150_150357

theorem time_with_family : 
    let hours_in_day := 24
    let sleep_fraction := 1 / 3
    let school_fraction := 1 / 6
    let assignment_fraction := 1 / 12
    let sleep_hours := sleep_fraction * hours_in_day
    let school_hours := school_fraction * hours_in_day
    let assignment_hours := assignment_fraction * hours_in_day
    let total_hours_occupied := sleep_hours + school_hours + assignment_hours
    hours_in_day - total_hours_occupied = 10 :=
by
  sorry

end time_with_family_l150_150357


namespace percentage_of_part_of_whole_l150_150991

theorem percentage_of_part_of_whole :
  let part := 375.2
  let whole := 12546.8
  (part / whole) * 100 = 2.99 :=
by
  sorry

end percentage_of_part_of_whole_l150_150991


namespace find_unknown_number_l150_150413

theorem find_unknown_number (x : ℤ) (h : (20 + 40 + 60) / 3 = 9 + (10 + 70 + x) / 3) : x = 13 :=
by
  sorry

end find_unknown_number_l150_150413


namespace number_of_n_l150_150523

theorem number_of_n (n : ℕ) (hn : n ≤ 500) (hk : ∃ k : ℕ, 21 * n = k^2) : 
  ∃ m : ℕ, m = 4 := by
  sorry

end number_of_n_l150_150523


namespace percent_area_shaded_l150_150530

-- Conditions: Square $ABCD$ has a side length of 10, and square $PQRS$ has a side length of 15.
-- The overlap of these squares forms a rectangle $AQRD$ with dimensions $20 \times 25$.

theorem percent_area_shaded 
  (side_ABCD : ℕ := 10) 
  (side_PQRS : ℕ := 15) 
  (dim_AQRD_length : ℕ := 25) 
  (dim_AQRD_width : ℕ := 20) 
  (area_AQRD : ℕ := dim_AQRD_length * dim_AQRD_width)
  (overlap_side : ℕ := 10) 
  (area_shaded : ℕ := overlap_side * overlap_side)
  : (area_shaded * 100) / area_AQRD = 20 := 
by 
  sorry

end percent_area_shaded_l150_150530


namespace sum_R1_R2_eq_19_l150_150927

-- Definitions for F_1 and F_2 in base R_1 and R_2
def F1_R1 : ℚ := 37 / 99
def F2_R1 : ℚ := 73 / 99
def F1_R2 : ℚ := 25 / 99
def F2_R2 : ℚ := 52 / 99

-- Prove that the sum of R1 and R2 is 19
theorem sum_R1_R2_eq_19 (R1 R2 : ℕ) (hF1R1 : F1_R1 = (3 * R1 + 7) / (R1^2 - 1))
  (hF2R1 : F2_R1 = (7 * R1 + 3) / (R1^2 - 1))
  (hF1R2 : F1_R2 = (2 * R2 + 5) / (R2^2 - 1))
  (hF2R2 : F2_R2 = (5 * R2 + 2) / (R2^2 - 1)) :
  R1 + R2 = 19 :=
  sorry

end sum_R1_R2_eq_19_l150_150927


namespace rectangle_perimeter_l150_150696

theorem rectangle_perimeter (w : ℝ) (P : ℝ) (l : ℝ) (A : ℝ) 
  (h1 : l = 18)
  (h2 : A = l * w)
  (h3 : P = 2 * l + 2 * w) 
  (h4 : A + P = 2016) : 
  P = 234 :=
by
  sorry

end rectangle_perimeter_l150_150696


namespace harmonious_division_condition_l150_150297

theorem harmonious_division_condition (a b c d e k : ℕ) (h : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e) (hk : 3 * k = a + b + c + d + e) (hk_pos : k > 0) :
  (∀ i j l : ℕ, i ≠ j ∧ j ≠ l ∧ i ≠ l → a ≤ k) ↔ (a ≤ k) :=
sorry

end harmonious_division_condition_l150_150297


namespace round_to_nearest_hundredth_l150_150963

noncomputable def recurring_decimal (n : ℕ) : ℝ :=
  if n = 87 then 87 + 36 / 99 else 0 -- Defines 87.3636... for n = 87

theorem round_to_nearest_hundredth : recurring_decimal 87 = 87.36 :=
by sorry

end round_to_nearest_hundredth_l150_150963


namespace chord_length_condition_l150_150821

theorem chord_length_condition (c : ℝ) (h : c > 0) :
  (∃ (x1 x2 : ℝ), 
    x1 ≠ x2 ∧ 
    dist (x1, x1^2) (x2, x2^2) = 2 ∧ 
    ∃ k : ℝ, x1 * k + c = x1^2 ∧ x2 * k + c = x2^2 ) 
    ↔ c > 0 :=
sorry

end chord_length_condition_l150_150821


namespace tree_last_tree_height_difference_l150_150475

noncomputable def treeHeightDifference : ℝ :=
  let t1 := 1000
  let t2 := 500
  let t3 := 500
  let avgHeight := 800
  let lastTreeHeight := 4 * avgHeight - (t1 + t2 + t3)
  lastTreeHeight - t1

theorem tree_last_tree_height_difference :
  treeHeightDifference = 200 := sorry

end tree_last_tree_height_difference_l150_150475


namespace value_of_a_plus_d_l150_150411

variable {R : Type} [LinearOrderedField R]
variables {a b c d : R}

theorem value_of_a_plus_d (h1 : a + b = 16) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 13 := by
  sorry

end value_of_a_plus_d_l150_150411


namespace least_n_froods_l150_150882

theorem least_n_froods (n : ℕ) : (∃ n, n ≥ 30 ∧ (n * (n + 1)) / 2 > 15 * n) ∧ (∀ m < 30, (m * (m + 1)) / 2 ≤ 15 * m) :=
sorry

end least_n_froods_l150_150882


namespace greatest_number_of_pieces_leftover_l150_150433

theorem greatest_number_of_pieces_leftover (y : ℕ) (q r : ℕ) 
  (h : y = 6 * q + r) (hrange : r < 6) : r = 5 := sorry

end greatest_number_of_pieces_leftover_l150_150433


namespace largest_positive_integer_l150_150461

def binary_operation (n : Int) : Int := n - (n * 5)

theorem largest_positive_integer (n : Int) : (∀ m : Int, m > 0 → n - (n * 5) < -19 → m ≤ n) 
  ↔ n = 5 := 
by
  sorry

end largest_positive_integer_l150_150461


namespace campers_difference_l150_150847

theorem campers_difference (a_morning : ℕ) (b_morning_afternoon : ℕ) (a_afternoon : ℕ) (a_afternoon_evening : ℕ) (c_evening_only : ℕ) :
  a_morning = 33 ∧ b_morning_afternoon = 11 ∧ a_afternoon = 34 ∧ a_afternoon_evening = 20 ∧ c_evening_only = 10 →
  a_afternoon - (a_afternoon_evening + c_evening_only) = 4 := 
by
  -- The actual proof would go here
  sorry

end campers_difference_l150_150847


namespace largest_on_edge_l150_150730

/-- On a grid, each cell contains a number which is the arithmetic mean of the four numbers around it 
    and all numbers are different. Prove that the largest number is located on the edge of the grid. -/
theorem largest_on_edge 
    (grid : ℕ → ℕ → ℝ) 
    (h_condition : ∀ (i j : ℕ), grid i j = (grid (i+1) j + grid (i-1) j + grid i (j+1) + grid i (j-1)) / 4)
    (h_unique : ∀ (i1 j1 i2 j2 : ℕ), (i1 ≠ i2 ∨ j1 ≠ j2) → grid i1 j1 ≠ grid i2 j2)
    : ∃ (i j : ℕ), (i = 0 ∨ j = 0 ∨ i = max_i ∨ j = max_j) ∧ ∀ (x y : ℕ), grid x y ≤ grid i j :=
sorry

end largest_on_edge_l150_150730


namespace beta_interval_solution_l150_150375

/-- 
Prove that the values of β in the set {β | β = π/6 + 2*k*π, k ∈ ℤ} 
that satisfy the interval (-2*π, 2*π) are β = π/6 or β = -11*π/6.
-/
theorem beta_interval_solution :
  ∀ β : ℝ, (∃ k : ℤ, β = (π / 6) + 2 * k * π) → (-2 * π < β ∧ β < 2 * π) →
  (β = π / 6 ∨ β = -11 * π / 6) :=
by
  intros β h_exists h_interval
  sorry

end beta_interval_solution_l150_150375


namespace rectangular_solid_surface_area_l150_150810

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hneq1 : a ≠ b) (hneq2 : b ≠ c) (hneq3 : a ≠ c) (hvol : a * b * c = 770) : 2 * (a * b + b * c + c * a) = 1098 :=
by
  sorry

end rectangular_solid_surface_area_l150_150810


namespace range_of_y_div_x_l150_150627

theorem range_of_y_div_x (x y : ℝ) (h : x^2 + (y-3)^2 = 1) : 
  (∃ k : ℝ, k = y / x ∧ (k ≤ -2 * Real.sqrt 2 ∨ k ≥ 2 * Real.sqrt 2)) :=
sorry

end range_of_y_div_x_l150_150627


namespace quadratic_function_vertex_form_l150_150252

theorem quadratic_function_vertex_form :
  ∃ f : ℝ → ℝ, (∀ x, f x = (x - 2)^2 - 2) ∧ (f 0 = 2) ∧ (∀ x, f x = a * (x - 2)^2 - 2 → a = 1) := by
  sorry

end quadratic_function_vertex_form_l150_150252


namespace trigonometric_identity_l150_150979

open Real

theorem trigonometric_identity (α : ℝ) (hα : sin (2 * π - α) = 4 / 5) (hα_range : 3 * π / 2 < α ∧ α < 2 * π) : 
  (sin α + cos α) / (sin α - cos α) = 1 / 7 := 
by
  sorry

end trigonometric_identity_l150_150979


namespace find_value_of_y_l150_150819

theorem find_value_of_y (y : ℚ) (h : 3 * y / 7 = 14) : y = 98 / 3 := 
by
  /- Proof to be completed -/
  sorry

end find_value_of_y_l150_150819


namespace dot_product_is_4_l150_150127

-- Define the vectors a and b
def a (k : ℝ) : ℝ × ℝ := (1, k)
def b : ℝ × ℝ := (2, 2)

-- Define collinearity condition
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Define k based on the collinearity condition
def k_value : ℝ := 1 -- derived from solving the collinearity condition in the problem

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the dot product of a and b is 4 when k = 1
theorem dot_product_is_4 {k : ℝ} (h : k = k_value) : dot_product (a k) b = 4 :=
by
  rw [h]
  sorry

end dot_product_is_4_l150_150127


namespace smallest_three_digit_divisible_by_4_and_5_l150_150312

-- Define the problem conditions and goal as a Lean theorem statement
theorem smallest_three_digit_divisible_by_4_and_5 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 4 = 0 ∧ m % 5 = 0 → n ≤ m) :=
sorry

end smallest_three_digit_divisible_by_4_and_5_l150_150312


namespace find_f_l150_150001

theorem find_f (f : ℝ → ℝ) (h₀ : f 0 = 1) (h₁ : ∀ x y, f (x * y) = f ((x^2 + y^2) / 2) + (x - y)^2) : 
  ∀ x, f x = 1 - 2 * x :=
by
  sorry  -- Proof not required

end find_f_l150_150001


namespace total_bricks_in_wall_l150_150796

theorem total_bricks_in_wall :
  let bottom_row_bricks := 18
  let rows := [bottom_row_bricks, bottom_row_bricks - 1, bottom_row_bricks - 2, bottom_row_bricks - 3, bottom_row_bricks - 4]
  (rows.sum = 80) := 
by
  let bottom_row_bricks := 18
  let rows := [bottom_row_bricks, bottom_row_bricks - 1, bottom_row_bricks - 2, bottom_row_bricks - 3, bottom_row_bricks - 4]
  sorry

end total_bricks_in_wall_l150_150796


namespace graph_three_lines_no_common_point_l150_150665

theorem graph_three_lines_no_common_point :
  ∀ x y : ℝ, x^2 * (x + 2*y - 3) = y^2 * (x + 2*y - 3) →
    x + 2*y - 3 = 0 ∨ x = y ∨ x = -y :=
by sorry

end graph_three_lines_no_common_point_l150_150665


namespace age_of_B_l150_150522

variable (A B C : ℕ)

theorem age_of_B (h1 : A + B + C = 84) (h2 : A + C = 58) : B = 26 := by
  sorry

end age_of_B_l150_150522


namespace determine_p_q_l150_150703

theorem determine_p_q (r1 r2 p q : ℝ) (h1 : r1 + r2 = 5) (h2 : r1 * r2 = 6) (h3 : r1^2 + r2^2 = -p) (h4 : r1^2 * r2^2 = q) : p = -13 ∧ q = 36 :=
by
  sorry

end determine_p_q_l150_150703


namespace range_a_l150_150166

theorem range_a (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 - x - 1 = 0 ∧ 
  ∀ y : ℝ, (0 < y ∧ y < 1 ∧ a * y^2 - y - 1 = 0 → y = x)) ↔ a > 2 :=
by
  sorry

end range_a_l150_150166


namespace units_digit_7_pow_2050_l150_150769

theorem units_digit_7_pow_2050 : (7 ^ 2050) % 10 = 9 := 
by 
  sorry

end units_digit_7_pow_2050_l150_150769


namespace interval_of_increase_logb_l150_150902

noncomputable def f (x : ℝ) := Real.logb 5 (2 * x + 1)

-- Define the domain
def domain : Set ℝ := {x | 2 * x + 1 > 0}

-- Define the interval of monotonic increase for the function
def interval_of_increase (f : ℝ → ℝ) : Set ℝ := {x | ∀ y, x < y → f x < f y}

-- Statement of the problem
theorem interval_of_increase_logb :
  interval_of_increase f = {x | x > - (1 / 2)} :=
by
  have h_increase : ∀ x y, x < y → f x < f y := sorry
  exact sorry

end interval_of_increase_logb_l150_150902


namespace simplify_and_evaluate_l150_150323

theorem simplify_and_evaluate (x y : ℝ) (h1 : x = 1/25) (h2 : y = -25) :
  x * (x + 2 * y) - (x + 1) ^ 2 + 2 * x = -3 :=
by
  sorry

end simplify_and_evaluate_l150_150323


namespace speed_of_stream_l150_150053

def upstream_speed (v : ℝ) := 72 - v
def downstream_speed (v : ℝ) := 72 + v

theorem speed_of_stream (v : ℝ) (h : 1 / upstream_speed v = 2 * (1 / downstream_speed v)) : v = 24 :=
by 
  sorry

end speed_of_stream_l150_150053


namespace remainder_property_l150_150738

theorem remainder_property (a : ℤ) (h : ∃ k : ℤ, a = 45 * k + 36) :
  ∃ n : ℤ, a = 45 * n + 36 :=
by {
  sorry
}

end remainder_property_l150_150738


namespace day50_previous_year_is_Wednesday_l150_150997

-- Given conditions
variable (N : ℕ) (dayOfWeek : ℕ → ℕ → ℕ)

-- Provided conditions stating specific days are Fridays
def day250_is_Friday : Prop := dayOfWeek 250 N = 5
def day150_is_Friday_next_year : Prop := dayOfWeek 150 (N+1) = 5

-- Proving the day of week for the 50th day of year N-1
def day50_previous_year : Prop := dayOfWeek 50 (N-1) = 3

-- Main theorem tying it together
theorem day50_previous_year_is_Wednesday (N : ℕ) (dayOfWeek : ℕ → ℕ → ℕ)
  (h1 : day250_is_Friday N dayOfWeek)
  (h2 : day150_is_Friday_next_year N dayOfWeek) :
  day50_previous_year N dayOfWeek :=
sorry -- Placeholder for actual proof

end day50_previous_year_is_Wednesday_l150_150997


namespace base_5_representation_l150_150598

theorem base_5_representation (n : ℕ) (h : n = 84) : 
  ∃ (a b c : ℕ), 
  a < 5 ∧ b < 5 ∧ c < 5 ∧ 
  n = a * 5^2 + b * 5^1 + c * 5^0 ∧ 
  a = 3 ∧ b = 1 ∧ c = 4 :=
by 
  -- Placeholder for the proof
  sorry

end base_5_representation_l150_150598


namespace element_of_set_l150_150130

theorem element_of_set : -1 ∈ { x : ℝ | x^2 - 1 = 0 } :=
sorry

end element_of_set_l150_150130


namespace fraction_addition_correct_l150_150576

theorem fraction_addition_correct : (3 / 5 : ℚ) + (2 / 5) = 1 := 
by
  sorry

end fraction_addition_correct_l150_150576


namespace sum_of_remainders_11111k_43210_eq_141_l150_150184

theorem sum_of_remainders_11111k_43210_eq_141 :
  (List.sum (List.map (fun k => (11111 * k + 43210) % 31) [0, 1, 2, 3, 4, 5])) = 141 :=
by
  -- Proof is omitted: sorry
  sorry

end sum_of_remainders_11111k_43210_eq_141_l150_150184


namespace solve_opposite_numbers_product_l150_150373

theorem solve_opposite_numbers_product :
  ∃ (x : ℤ), 3 * x - 2 * (-x) = 30 ∧ x * (-x) = -36 :=
by
  sorry

end solve_opposite_numbers_product_l150_150373


namespace subtraction_makes_divisible_l150_150082

theorem subtraction_makes_divisible :
  ∃ n : Nat, 9671 - n % 2 = 0 ∧ n = 1 :=
by
  sorry

end subtraction_makes_divisible_l150_150082


namespace car_miles_per_gallon_l150_150459

-- Define the conditions
def distance_home : ℕ := 220
def additional_distance : ℕ := 100
def total_distance : ℕ := distance_home + additional_distance
def tank_capacity : ℕ := 16 -- in gallons
def miles_per_gallon : ℕ := total_distance / tank_capacity

-- State the goal
theorem car_miles_per_gallon : miles_per_gallon = 20 := by
  sorry

end car_miles_per_gallon_l150_150459


namespace repeating_decimal_computation_l150_150116

noncomputable def x := 864 / 999
noncomputable def y := 579 / 999
noncomputable def z := 135 / 999

theorem repeating_decimal_computation :
  x - y - z = 50 / 333 :=
by
  sorry

end repeating_decimal_computation_l150_150116


namespace total_handshakes_l150_150492

def gremlins := 30
def pixies := 12
def unfriendly_gremlins := 15
def friendly_gremlins := 15

def handshake_count : Nat :=
  let handshakes_friendly_gremlins := friendly_gremlins * (friendly_gremlins - 1) / 2
  let handshakes_friendly_unfriendly := friendly_gremlins * unfriendly_gremlins
  let handshakes_gremlins_pixies := gremlins * pixies
  handshakes_friendly_gremlins + handshakes_friendly_unfriendly + handshakes_gremlins_pixies

theorem total_handshakes : handshake_count = 690 := by
  sorry

end total_handshakes_l150_150492


namespace hyperbola_equation_l150_150291

variable (a b c : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (asymptote_cond : -b / a = -1 / 2)
variable (foci_cond : c = 5)
variable (hyperbola_rel : a^2 + b^2 = c^2)

theorem hyperbola_equation : 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ -b / a = -1 / 2 ∧ c = 5 ∧ a^2 + b^2 = c^2 
  ∧ ∀ x y : ℝ, (x^2 / 20 - y^2 / 5 = 1)) := 
sorry

end hyperbola_equation_l150_150291


namespace exists_increasing_triplet_l150_150912

theorem exists_increasing_triplet (f : ℕ → ℕ) (bij : Function.Bijective f) :
  ∃ (a d : ℕ), 0 < a ∧ 0 < d ∧ f a < f (a + d) ∧ f (a + d) < f (a + 2 * d) :=
by
  sorry

end exists_increasing_triplet_l150_150912


namespace number_of_women_per_table_l150_150482

theorem number_of_women_per_table
  (tables : ℕ) (men_per_table : ℕ) 
  (total_customers : ℕ) (total_tables : tables = 9) 
  (men_at_each_table : men_per_table = 3) 
  (customers : total_customers = 90) 
  (total_men : 3 * 9 = 27) 
  (total_women : 90 - 27 = 63) :
  (63 / 9 = 7) :=
by
  sorry

end number_of_women_per_table_l150_150482


namespace lucille_house_difference_l150_150584

-- Define the heights of the houses as given in the conditions.
def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

-- Define the total height of the houses.
def total_height : ℕ := height_neighbor1 + height_lucille + height_neighbor2

-- Define the average height of the houses.
def average_height : ℕ := total_height / 3

-- Define the height difference between Lucille's house and the average height.
def height_difference : ℕ := average_height - height_lucille

-- The theorem to prove.
theorem lucille_house_difference :
  height_difference = 3 := by
  sorry

end lucille_house_difference_l150_150584


namespace monotonically_increasing_interval_l150_150690

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (k * Real.pi - 5 * Real.pi / 12 <= x ∧ x <= k * Real.pi + Real.pi / 12) →
    (∃ r : ℝ, f (x + r) > f x ∨ f (x + r) < f x) := by
  sorry

end monotonically_increasing_interval_l150_150690


namespace least_four_digit_integer_has_3_7_11_as_factors_l150_150570

theorem least_four_digit_integer_has_3_7_11_as_factors :
  ∃ x : ℕ, (1000 ≤ x ∧ x < 10000) ∧ (3 ∣ x) ∧ (7 ∣ x) ∧ (11 ∣ x) ∧ x = 1155 := by
  sorry

end least_four_digit_integer_has_3_7_11_as_factors_l150_150570


namespace compute_f_g_f_3_l150_150651

def f (x : ℤ) : ℤ := 5 * x + 5
def g (x : ℤ) : ℤ := 6 * x + 4

theorem compute_f_g_f_3 : f (g (f 3)) = 625 := sorry

end compute_f_g_f_3_l150_150651


namespace not_collinear_C_vector_decomposition_l150_150215

namespace VectorProof

open Function

structure Vector2 where
  x : ℝ
  y : ℝ

def add (v1 v2 : Vector2) : Vector2 := ⟨v1.x + v2.x, v1.y + v2.y⟩
def scale (c : ℝ) (v : Vector2) : Vector2 := ⟨c * v.x, c * v.y⟩

def collinear (v1 v2 : Vector2) : Prop :=
  ∃ k : ℝ, v2 = scale k v1

def vector_a : Vector2 := ⟨3, 4⟩
def e₁_C : Vector2 := ⟨-1, 2⟩
def e₂_C : Vector2 := ⟨3, -1⟩

theorem not_collinear_C :
  ¬ collinear e₁_C e₂_C :=
sorry

theorem vector_decomposition :
  ∃ (x y : ℝ), vector_a = add (scale x e₁_C) (scale y e₂_C) :=
sorry

end VectorProof

end not_collinear_C_vector_decomposition_l150_150215


namespace hypotenuse_length_l150_150975

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 := 
by
  sorry

end hypotenuse_length_l150_150975


namespace correct_propositions_l150_150256

-- Definitions of the conditions in the Math problem

variable (triangle_outside_plane : Prop)
variable (triangle_side_intersections_collinear : Prop)
variable (parallel_lines_coplanar : Prop)
variable (noncoplanar_points_planes : Prop)

-- Math proof problem statement
theorem correct_propositions :
  (triangle_outside_plane ∧ 
   parallel_lines_coplanar ∧ 
   ¬noncoplanar_points_planes) →
  2 = 2 :=
by
  sorry

end correct_propositions_l150_150256


namespace length_of_each_train_l150_150842

theorem length_of_each_train
  (L : ℝ) -- length of each train
  (speed_fast : ℝ) (speed_slow : ℝ) -- speeds of the fast and slow trains in km/hr
  (time_pass : ℝ) -- time for the slower train to pass the driver of the faster one in seconds
  (h_speed_fast : speed_fast = 45) -- speed of the faster train
  (h_speed_slow : speed_slow = 15) -- speed of the slower train
  (h_time_pass : time_pass = 60) -- time to pass
  (h_same_length : ∀ (x y : ℝ), x = y → x = L) :  
  L = 1000 :=
  by
  -- Skipping the proof as instructed
  sorry

end length_of_each_train_l150_150842


namespace problem_statement_l150_150369

-- Define C and D as specified in the problem conditions.
def C : ℕ := 4500
def D : ℕ := 3000

-- The final statement of the problem to prove C + D = 7500.
theorem problem_statement : C + D = 7500 := by
  -- This proof can be completed by checking arithmetic.
  sorry

end problem_statement_l150_150369


namespace initial_roses_l150_150501

theorem initial_roses (x : ℕ) (h1 : x - 3 + 34 = 36) : x = 5 :=
by 
  sorry

end initial_roses_l150_150501


namespace find_a_l150_150957

theorem find_a (a : ℝ) (h : 1 / Real.log 5 / Real.log a + 1 / Real.log 6 / Real.log a + 1 / Real.log 10 / Real.log a = 1) : a = 300 :=
sorry

end find_a_l150_150957


namespace count_four_digit_numbers_l150_150954

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l150_150954


namespace percentage_of_managers_l150_150178

theorem percentage_of_managers (P : ℝ) :
  (200 : ℝ) * (P / 100) - 99.99999999999991 = 0.98 * (200 - 99.99999999999991) →
  P = 99 := 
sorry

end percentage_of_managers_l150_150178


namespace largest_multiple_l150_150107

theorem largest_multiple (n : ℤ) (h8 : 8 ∣ n) (h : -n > -80) : n = 72 :=
by 
  sorry

end largest_multiple_l150_150107


namespace ab_value_l150_150603

theorem ab_value (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : a^2 + 3 * b = 33) : a * b = 24 := 
by 
  sorry

end ab_value_l150_150603


namespace find_primes_l150_150114

theorem find_primes (A B C : ℕ) (hA : A < 20) (hB : B < 20) (hC : C < 20)
  (hA_prime : Prime A) (hB_prime : Prime B) (hC_prime : Prime C)
  (h_sum : A + B + C = 30) : 
  (A = 2 ∧ B = 11 ∧ C = 17) ∨ (A = 2 ∧ B = 17 ∧ C = 11) ∨ 
  (A = 11 ∧ B = 2 ∧ C = 17) ∨ (A = 11 ∧ B = 17 ∧ C = 2) ∨ 
  (A = 17 ∧ B = 2 ∧ C = 11) ∨ (A = 17 ∧ B = 11 ∧ C = 2) :=
sorry

end find_primes_l150_150114


namespace fraction_div_addition_l150_150770

theorem fraction_div_addition : ( (3 / 7 : ℚ) / 4) + (1 / 28) = (1 / 7) :=
  sorry

end fraction_div_addition_l150_150770


namespace am_gm_hm_inequality_l150_150211

variable {x y : ℝ}

-- Conditions: x and y are positive real numbers and x < y
def conditions (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ x < y

-- Proof statement: A.M. > G.M. > H.M. under given conditions
theorem am_gm_hm_inequality (x y : ℝ) (h : conditions x y) :
  (x + y) / 2 > Real.sqrt (x * y) ∧ Real.sqrt (x * y) > (2 * x * y) / (x + y) :=
sorry

end am_gm_hm_inequality_l150_150211


namespace pure_imaginary_number_a_l150_150439

theorem pure_imaginary_number_a (a : ℝ) 
  (h1 : a^2 + 2 * a - 3 = 0)
  (h2 : a^2 - 4 * a + 3 ≠ 0) : a = -3 :=
sorry

end pure_imaginary_number_a_l150_150439


namespace missing_digits_pairs_l150_150709

theorem missing_digits_pairs (x y : ℕ) : (2 + 4 + 6 + x + y + 8) % 9 = 0 ↔ x + y = 7 := by
  sorry

end missing_digits_pairs_l150_150709


namespace most_likely_outcome_is_D_l150_150867

-- Define the basic probability of rolling any specific number with a fair die
def probability_of_specific_roll : ℚ := 1/6

-- Define the probability of each option
def P_A : ℚ := probability_of_specific_roll
def P_B : ℚ := 2 * probability_of_specific_roll
def P_C : ℚ := 3 * probability_of_specific_roll
def P_D : ℚ := 4 * probability_of_specific_roll

-- Define the proof problem statement
theorem most_likely_outcome_is_D : P_D = max P_A (max P_B (max P_C P_D)) :=
sorry

end most_likely_outcome_is_D_l150_150867


namespace Mitzi_score_l150_150360

-- Definitions based on the conditions
def Gretchen_score : ℕ := 120
def Beth_score : ℕ := 85
def average_score (total_score : ℕ) (num_bowlers : ℕ) : ℕ := total_score / num_bowlers

-- Theorem stating that Mitzi's bowling score is 113
theorem Mitzi_score (m : ℕ) (h : average_score (Gretchen_score + m + Beth_score) 3 = 106) :
  m = 113 :=
by sorry

end Mitzi_score_l150_150360


namespace birdhouse_price_l150_150311

theorem birdhouse_price (S : ℤ) : 
  (2 * 22) + (2 * 16) + (3 * S) = 97 → 
  S = 7 :=
by
  sorry

end birdhouse_price_l150_150311


namespace alex_piles_of_jelly_beans_l150_150689

theorem alex_piles_of_jelly_beans : 
  ∀ (initial_weight eaten weight_per_pile remaining_weight piles : ℕ),
    initial_weight = 36 →
    eaten = 6 →
    weight_per_pile = 10 →
    remaining_weight = initial_weight - eaten →
    piles = remaining_weight / weight_per_pile →
    piles = 3 :=
by
  intros initial_weight eaten weight_per_pile remaining_weight piles h_init h_eat h_wpile h_remaining h_piles
  sorry

end alex_piles_of_jelly_beans_l150_150689


namespace dimensions_of_triangle_from_square_l150_150089

theorem dimensions_of_triangle_from_square :
  ∀ (a : ℝ) (triangle : ℝ × ℝ × ℝ), 
    a = 10 →
    triangle = (a, a, a * Real.sqrt 2) →
    triangle = (10, 10, 10 * Real.sqrt 2) :=
by
  intros a triangle a_eq triangle_eq
  -- Proof
  sorry

end dimensions_of_triangle_from_square_l150_150089


namespace values_of_x_l150_150004

theorem values_of_x (x : ℝ) (h1 : x^2 - 3 * x - 10 < 0) (h2 : 1 < x) : 1 < x ∧ x < 5 := 
sorry

end values_of_x_l150_150004


namespace final_length_of_movie_l150_150934

theorem final_length_of_movie :
  let original_length := 3600 -- original movie length in seconds
  let cut_1 := 3 * 60 -- first scene cut in seconds
  let cut_2 := (5 * 60) + 30 -- second scene cut in seconds
  let cut_3 := (2 * 60) + 15 -- third scene cut in seconds
  let total_cut := cut_1 + cut_2 + cut_3 -- total cut time in seconds
  let final_length_seconds := original_length - total_cut -- final length in seconds
  final_length_seconds = 2955 ∧ final_length_seconds / 60 = 49 ∧ final_length_seconds % 60 = 15
:= by
  sorry

end final_length_of_movie_l150_150934


namespace volleyballs_remaining_l150_150658

def initial_volleyballs := 9
def lent_volleyballs := 5

theorem volleyballs_remaining : initial_volleyballs - lent_volleyballs = 4 := 
by
  sorry

end volleyballs_remaining_l150_150658


namespace compute_expression_l150_150995

theorem compute_expression : (7^2 - 2 * 5 + 2^3) = 47 :=
by
  sorry

end compute_expression_l150_150995


namespace deductive_reasoning_example_l150_150398

-- Definitions for the conditions
def Metal (x : Type) : Prop := sorry
def ConductsElectricity (x : Type) : Prop := sorry
def Iron : Type := sorry

-- The problem statement
theorem deductive_reasoning_example (H1 : ∀ x, Metal x → ConductsElectricity x) (H2 : Metal Iron) : ConductsElectricity Iron :=
by sorry

end deductive_reasoning_example_l150_150398


namespace four_edge_trips_count_l150_150509

-- Defining points and edges of the cube
inductive Point
| A | B | C | D | E | F | G | H

open Point

-- Edges of the cube are connections between points
def Edge (p1 p2 : Point) : Prop :=
  ∃ (edges : List (Point × Point)), 
    edges = [(A, B), (A, D), (A, E), (B, C), (B, E), (B, F), (C, D), (C, F), (C, G), (D, E), (D, F), (D, H), (E, F), (E, H), (F, G), (F, H), (G, H)] ∧ 
    ((p1, p2) ∈ edges ∨ (p2, p1) ∈ edges)

-- Define the proof statement
theorem four_edge_trips_count : 
  ∃ (num_paths : ℕ), num_paths = 12 :=
sorry

end four_edge_trips_count_l150_150509


namespace average_salary_all_employees_l150_150636

-- Define the given conditions
def average_salary_officers : ℝ := 440
def average_salary_non_officers : ℝ := 110
def number_of_officers : ℕ := 15
def number_of_non_officers : ℕ := 480

-- Define the proposition we need to prove
theorem average_salary_all_employees :
  let total_salary_officers := average_salary_officers * number_of_officers
  let total_salary_non_officers := average_salary_non_officers * number_of_non_officers
  let total_salary_all_employees := total_salary_officers + total_salary_non_officers
  let total_number_of_employees := number_of_officers + number_of_non_officers
  let average_salary_all_employees := total_salary_all_employees / total_number_of_employees
  average_salary_all_employees = 120 :=
by {
  -- Skipping the proof steps
  sorry
}

end average_salary_all_employees_l150_150636


namespace x_gt_one_iff_x_cube_gt_one_l150_150049

theorem x_gt_one_iff_x_cube_gt_one (x : ℝ) : x > 1 ↔ x^3 > 1 :=
by sorry

end x_gt_one_iff_x_cube_gt_one_l150_150049


namespace find_b_l150_150563

theorem find_b (b : ℝ) (tangent_condition : ∀ x y : ℝ, y = -2 * x + b → y^2 = 8 * x) : b = -1 :=
sorry

end find_b_l150_150563


namespace hyperbola_eccentricity_b_value_l150_150938

theorem hyperbola_eccentricity_b_value (b : ℝ) (a : ℝ) (e : ℝ) 
  (h1 : a^2 = 1) (h2 : e = 2) 
  (h3 : b > 0) (h4 : b^2 = 4 - 1) : 
  b = Real.sqrt 3 := 
by 
  sorry

end hyperbola_eccentricity_b_value_l150_150938


namespace count_valid_triples_l150_150202

def S (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def satisfies_conditions (a b c : ℕ) : Prop :=
  is_three_digit a ∧ is_three_digit b ∧ is_three_digit c ∧ 
  (a + b + c = 2005) ∧ (S a + S b + S c = 61)

def number_of_valid_triples : ℕ := sorry

theorem count_valid_triples : number_of_valid_triples = 17160 :=
sorry

end count_valid_triples_l150_150202


namespace triangle_cosine_condition_l150_150768

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to angles A, B, and C

-- Definitions according to the problem conditions
def law_of_sines (a b : ℝ) (A B : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B

theorem triangle_cosine_condition (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : law_of_sines a b A B)
  (h1 : a > b) : Real.cos (2 * A) < Real.cos (2 * B) ↔ a > b :=
by
  sorry

end triangle_cosine_condition_l150_150768


namespace tan_sum_eq_l150_150777

theorem tan_sum_eq (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1/2 :=
by sorry

end tan_sum_eq_l150_150777


namespace base10_representation_of_n_l150_150338

theorem base10_representation_of_n (a b c n : ℕ) (ha : a > 0)
  (h14 : n = 14^2 * a + 14 * b + c)
  (h15 : n = 15^2 * a + 15 * c + b)
  (h6 : n = 6^3 * a + 6^2 * c + 6 * a + c) : n = 925 :=
by sorry

end base10_representation_of_n_l150_150338


namespace old_clock_slower_l150_150119

-- Given conditions
def old_clock_coincidence_minutes : ℕ := 66

-- Standard clock coincidences in 24 hours
def standard_clock_coincidences_in_24_hours : ℕ := 22

-- Standard 24 hours in minutes
def standard_24_hours_in_minutes : ℕ := 24 * 60

-- Total time for old clock in minutes over what should be 24 hours
def total_time_for_old_clock : ℕ := standard_clock_coincidences_in_24_hours * old_clock_coincidence_minutes

-- Problem statement: prove that the old clock's 24 hours is 12 minutes slower 
theorem old_clock_slower : total_time_for_old_clock = standard_24_hours_in_minutes + 12 := by
  sorry

end old_clock_slower_l150_150119


namespace solution_set_of_inequality_l150_150714

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) / (x^2 - x - 6) ≥ 0} = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (3 < x)} := 
sorry

end solution_set_of_inequality_l150_150714


namespace unique_abc_solution_l150_150310

theorem unique_abc_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
    (h4 : a^4 + b^2 * c^2 = 16 * a) (h5 : b^4 + c^2 * a^2 = 16 * b) (h6 : c^4 + a^2 * b^2 = 16 * c) : 
    (a, b, c) = (2, 2, 2) :=
  by
    sorry

end unique_abc_solution_l150_150310


namespace smallest_positive_period_max_min_value_interval_l150_150460

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin (x + Real.pi / 3))^2 - (Real.cos x)^2 + (Real.sin x)^2

theorem smallest_positive_period : (∀ x : ℝ, f (x + Real.pi) = f x) :=
by sorry

theorem max_min_value_interval :
  (∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 6), 
    f x ≤ 3 / 2 ∧ f x ≥ 0 ∧ 
    (f (-Real.pi / 6) = 0) ∧ 
    (f (Real.pi / 6) = 3 / 2)) :=
by sorry

end smallest_positive_period_max_min_value_interval_l150_150460


namespace arrange_animals_adjacent_l150_150400

theorem arrange_animals_adjacent:
  let chickens := 5
  let dogs := 3
  let cats := 6
  let rabbits := 4
  let total_animals := 18
  let group_orderings := 24 -- 4!
  let chicken_orderings := 120 -- 5!
  let dog_orderings := 6 -- 3!
  let cat_orderings := 720 -- 6!
  let rabbit_orderings := 24 -- 4!
  total_animals = chickens + dogs + cats + rabbits →
  chickens > 0 ∧ dogs > 0 ∧ cats > 0 ∧ rabbits > 0 →
  group_orderings * chicken_orderings * dog_orderings * cat_orderings * rabbit_orderings = 17863680 :=
  by intros; sorry

end arrange_animals_adjacent_l150_150400


namespace power_function_properties_l150_150907

theorem power_function_properties (α : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ α) 
    (h_point : f (1/2) = 2 ) :
    (∀ x : ℝ, f x = 1 / x) ∧ (∀ x : ℝ, 0 < x → (f x) < (f (x / 2))) ∧ (∀ x : ℝ, f (-x) = - (f x)) :=
by
  sorry

end power_function_properties_l150_150907


namespace compare_negatives_l150_150517

theorem compare_negatives : (-1.5 : ℝ) < (-1 + -1/5 : ℝ) :=
by 
  sorry

end compare_negatives_l150_150517


namespace min_value_a_plus_b_l150_150520

theorem min_value_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : Real.sqrt (3^a * 3^b) = 3^((a + b) / 2)) : a + b = 4 := by
  sorry

end min_value_a_plus_b_l150_150520


namespace solution_l150_150329

/-- Definition of the number with 2023 ones. -/
def x_2023 : ℕ := (10^2023 - 1) / 9

/-- Definition of the polynomial equation. -/
def polynomial_eq (x : ℕ) : ℤ :=
  567 * x^3 + 171 * x^2 + 15 * x - (7 * x + 5 * 10^2023 + 3 * 10^(2*2023))

/-- The solution x_2023 satisfies the polynomial equation. -/
theorem solution : polynomial_eq x_2023 = 0 := sorry

end solution_l150_150329


namespace ratio_of_a_and_b_l150_150374

theorem ratio_of_a_and_b (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : (a * Real.sin (Real.pi / 7) + b * Real.cos (Real.pi / 7)) / 
        (a * Real.cos (Real.pi / 7) - b * Real.sin (Real.pi / 7)) = 
        Real.tan (10 * Real.pi / 21)) :
  b / a = Real.sqrt 3 :=
sorry

end ratio_of_a_and_b_l150_150374


namespace neg_and_eq_or_not_l150_150548

theorem neg_and_eq_or_not (p q : Prop) : ¬(p ∧ q) ↔ ¬p ∨ ¬q :=
by sorry

end neg_and_eq_or_not_l150_150548


namespace male_students_tree_planting_l150_150834

theorem male_students_tree_planting (average_trees : ℕ) (female_trees : ℕ) 
    (male_trees : ℕ) : 
    (average_trees = 6) →
    (female_trees = 15) → 
    (1 / male_trees + 1 / female_trees = 1 / average_trees) → 
    male_trees = 10 :=
by
  intros h_avg h_fem h_eq
  sorry

end male_students_tree_planting_l150_150834


namespace jackson_has_1900_more_than_brandon_l150_150057

-- Conditions
def initial_investment : ℝ := 500
def jackson_multiplier : ℝ := 4
def brandon_multiplier : ℝ := 0.20

-- Final values
def jackson_final_value := jackson_multiplier * initial_investment
def brandon_final_value := brandon_multiplier * initial_investment

-- Statement to prove the difference
theorem jackson_has_1900_more_than_brandon : jackson_final_value - brandon_final_value = 1900 := 
    by sorry

end jackson_has_1900_more_than_brandon_l150_150057


namespace age_proof_l150_150754

noncomputable def father_age_current := 33
noncomputable def xiaolin_age_current := 3

def father_age (X : ℕ) := 11 * X
def future_father_age (F : ℕ) := F + 7
def future_xiaolin_age (X : ℕ) := X + 7

theorem age_proof (F X : ℕ) (h1 : F = father_age X) 
  (h2 : future_father_age F = 4 * future_xiaolin_age X) : 
  F = father_age_current ∧ X = xiaolin_age_current :=
by 
  sorry

end age_proof_l150_150754


namespace average_weight_correct_l150_150321

-- Define the number of men and women
def number_of_men : ℕ := 8
def number_of_women : ℕ := 6

-- Define the average weights of men and women
def average_weight_men : ℕ := 190
def average_weight_women : ℕ := 120

-- Define the total weight of men and women
def total_weight_men : ℕ := number_of_men * average_weight_men
def total_weight_women : ℕ := number_of_women * average_weight_women

-- Define the total number of individuals
def total_individuals : ℕ := number_of_men + number_of_women

-- Define the combined total weight
def total_weight : ℕ := total_weight_men + total_weight_women

-- Define the average weight of all individuals
def average_weight_all : ℕ := total_weight / total_individuals

theorem average_weight_correct :
  average_weight_all = 160 :=
  by sorry

end average_weight_correct_l150_150321


namespace sin_squared_minus_cos_squared_l150_150861

theorem sin_squared_minus_cos_squared {α : ℝ} (h : Real.sin α = Real.sqrt 5 / 5) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 = -3 / 5 :=
by
  sorry -- Proof is omitted

end sin_squared_minus_cos_squared_l150_150861


namespace trendy_haircut_cost_l150_150615

theorem trendy_haircut_cost (T : ℝ) (H1 : 5 * 5 * 7 + 3 * 6 * 7 + 2 * T * 7 = 413) : T = 8 :=
by linarith

end trendy_haircut_cost_l150_150615


namespace trisha_interest_l150_150639

noncomputable def total_amount (P : ℝ) (r : ℝ) (D : ℝ) (t : ℕ) : ℝ :=
  let rec compute (n : ℕ) (A : ℝ) :=
    if n = 0 then A
    else let A_next := A * (1 + r) + D
         compute (n - 1) A_next
  compute t P

noncomputable def total_deposits (D : ℝ) (t : ℕ) : ℝ :=
  D * t

noncomputable def total_interest (P : ℝ) (r : ℝ) (D : ℝ) (t : ℕ) : ℝ :=
  total_amount P r D t - P - total_deposits D t

theorem trisha_interest :
  total_interest 2000 0.05 300 5 = 710.25 :=
by
  sorry

end trisha_interest_l150_150639


namespace star_polygon_points_l150_150864

theorem star_polygon_points (p : ℕ) (ϕ : ℝ) :
  (∀ i : Fin p, ∃ Ci Di : ℝ, Ci = Di + 15) →
  (p * ϕ + p * (ϕ + 15) = 360) →
  p = 24 :=
by
  sorry

end star_polygon_points_l150_150864


namespace xy_identity_l150_150245

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = 6) : x^2 + y^2 = 4 := 
by 
  sorry

end xy_identity_l150_150245


namespace fraction_of_water_in_mixture_l150_150195

theorem fraction_of_water_in_mixture (r : ℚ) (h : r = 2 / 3) : (3 / (2 + 3) : ℚ) = 3 / 5 :=
by
  sorry

end fraction_of_water_in_mixture_l150_150195


namespace shortest_side_length_triangle_l150_150831

noncomputable def triangle_min_angle_side_length (A B : ℝ) (c : ℝ) (tanA tanB : ℝ) (ha : tanA = 1 / 4) (hb : tanB = 3 / 5) (hc : c = Real.sqrt 17) : ℝ :=
   Real.sqrt 2

theorem shortest_side_length_triangle {A B c : ℝ} {tanA tanB : ℝ} 
  (ha : tanA = 1 / 4) (hb : tanB = 3 / 5) (hc : c = Real.sqrt 17) :
  triangle_min_angle_side_length A B c tanA tanB ha hb hc = Real.sqrt 2 :=
sorry

end shortest_side_length_triangle_l150_150831


namespace min_a_b_l150_150865

theorem min_a_b : 
  (∀ x : ℝ, 3 * a * (Real.sin x + Real.cos x) + 2 * b * Real.sin (2 * x) ≤ 3) →
  a + b = -2 →
  a = -4 / 5 :=
by
  sorry

end min_a_b_l150_150865


namespace compute_inverse_10_mod_1729_l150_150701

def inverse_of_10_mod_1729 : ℕ :=
  1537

theorem compute_inverse_10_mod_1729 :
  (10 * inverse_of_10_mod_1729) % 1729 = 1 :=
by
  sorry

end compute_inverse_10_mod_1729_l150_150701


namespace sum_of_solutions_l150_150317

theorem sum_of_solutions (x : ℝ) (h : x^2 - 3 * x = 12) : x = 3 := by
  sorry

end sum_of_solutions_l150_150317


namespace find_y_l150_150926

noncomputable def x : ℝ := (4 / 25)^(1 / 3)

theorem find_y (y : ℝ) (h1 : 0 < y) (h2 : y < x) (h3 : x^x = y^y) : y = (32 / 3125)^(1 / 3) :=
sorry

end find_y_l150_150926


namespace least_subtracted_12702_is_26_l150_150684

theorem least_subtracted_12702_is_26 : 12702 % 99 = 26 :=
by
  sorry

end least_subtracted_12702_is_26_l150_150684


namespace sum_of_y_coordinates_of_other_vertices_l150_150069

theorem sum_of_y_coordinates_of_other_vertices
  (A B : ℝ × ℝ)
  (C D : ℝ × ℝ)
  (hA : A = (2, 15))
  (hB : B = (8, -2))
  (h_mid : midpoint ℝ A B = midpoint ℝ C D) :
  C.snd + D.snd = 13 := 
sorry

end sum_of_y_coordinates_of_other_vertices_l150_150069


namespace jacob_twice_as_old_l150_150187

theorem jacob_twice_as_old (x : ℕ) : 18 + x = 2 * (9 + x) → x = 0 := by
  intro h
  linarith

end jacob_twice_as_old_l150_150187


namespace turquoise_more_green_l150_150718

-- Definitions based on given conditions
def total_people : ℕ := 150
def more_blue : ℕ := 90
def both_blue_green : ℕ := 40
def neither_blue_green : ℕ := 20

-- Theorem statement to prove the number of people who believe turquoise is more green
theorem turquoise_more_green : (total_people - neither_blue_green - (more_blue - both_blue_green) - both_blue_green) + both_blue_green = 80 := by
  sorry

end turquoise_more_green_l150_150718


namespace circle_reflection_l150_150066

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3)
    (new_x new_y : ℝ) (hne_x : new_x = 3) (hne_y : new_y = -8) :
    (new_x, new_y) = (-y, -x) := by
  sorry

end circle_reflection_l150_150066


namespace jen_age_proof_l150_150553

-- Definitions
def son_age := 16
def son_present_age := son_age
def jen_present_age := 41

-- Conditions
axiom jen_older_25 (x : ℕ) : ∀ y : ℕ, x = y + 25 → y = son_present_age
axiom jen_age_formula (j s : ℕ) : j = 3 * s - 7 → j = son_present_age + 25

-- Proof problem statement
theorem jen_age_proof : jen_present_age = 41 :=
by
  -- Declare variables
  let j := jen_present_age
  let s := son_present_age
  -- Apply conditions (in Lean, sorry will skip the proof)
  sorry

end jen_age_proof_l150_150553


namespace diana_took_six_candies_l150_150989

-- Define the initial number of candies in the box
def initial_candies : ℕ := 88

-- Define the number of candies left in the box after Diana took some
def remaining_candies : ℕ := 82

-- Define the number of candies taken by Diana
def candies_taken : ℕ := initial_candies - remaining_candies

-- The theorem we need to prove
theorem diana_took_six_candies : candies_taken = 6 := by
  sorry

end diana_took_six_candies_l150_150989


namespace maximum_root_l150_150044

noncomputable def max_root (α β γ : ℝ) : ℝ := 
  if α ≥ β ∧ α ≥ γ then α 
  else if β ≥ α ∧ β ≥ γ then β 
  else γ

theorem maximum_root :
  ∃ α β γ : ℝ, α + β + γ = 14 ∧ α^2 + β^2 + γ^2 = 84 ∧ α^3 + β^3 + γ^3 = 584 ∧ max_root α β γ = 8 :=
by
  sorry

end maximum_root_l150_150044


namespace distance_to_origin_l150_150558

theorem distance_to_origin (x y : ℤ) (hx : x = -5) (hy : y = 12) :
  Real.sqrt (x^2 + y^2) = 13 := by
  rw [hx, hy]
  norm_num
  sorry

end distance_to_origin_l150_150558


namespace initial_employees_l150_150849

theorem initial_employees (E : ℕ)
  (salary_per_employee : ℕ)
  (laid_off_fraction : ℚ)
  (total_paid_remaining : ℕ)
  (remaining_employees : ℕ) :
  salary_per_employee = 2000 →
  laid_off_fraction = 1 / 3 →
  total_paid_remaining = 600000 →
  remaining_employees = total_paid_remaining / salary_per_employee →
  (2 / 3 : ℚ) * E = remaining_employees →
  E = 450 := by
  sorry

end initial_employees_l150_150849


namespace add_neg_eq_neg_add_neg_ten_plus_neg_twelve_l150_150923

theorem add_neg_eq_neg_add (a b : Int) : a + -b = a - b := by
  sorry

theorem neg_ten_plus_neg_twelve : -10 + (-12) = -22 := by
  have h1 : -10 + (-12) = -10 - 12 := add_neg_eq_neg_add _ _
  have h2 : -10 - 12 = -(10 + 12) := by
    sorry -- This step corresponds to recognizing the arithmetic rule for subtraction.
  have h3 : -(10 + 12) = -22 := by
    sorry -- This step is the concrete calculation.
  exact Eq.trans h1 (Eq.trans h2 h3)

end add_neg_eq_neg_add_neg_ten_plus_neg_twelve_l150_150923


namespace decaf_percentage_correct_l150_150805

def initial_stock : ℝ := 400
def initial_decaf_percent : ℝ := 0.20
def additional_stock : ℝ := 100
def additional_decaf_percent : ℝ := 0.70

theorem decaf_percentage_correct :
  ((initial_decaf_percent * initial_stock + additional_decaf_percent * additional_stock) / (initial_stock + additional_stock)) * 100 = 30 :=
by
  sorry

end decaf_percentage_correct_l150_150805


namespace smallest_coins_l150_150634

theorem smallest_coins (n : ℕ) (n_min : ℕ) (h1 : ∃ n, n % 8 = 5 ∧ n % 7 = 4 ∧ n = 53) (h2 : n_min = n):
  (n_min ≡ 5 [MOD 8]) ∧ (n_min ≡ 4 [MOD 7]) ∧ (n_min = 53) ∧ (53 % 9 = 8) :=
by
  sorry

end smallest_coins_l150_150634


namespace fibby_numbers_l150_150619

def is_fibby (k : ℕ) : Prop :=
  k ≥ 3 ∧ ∃ (n : ℕ) (d : ℕ → ℕ),
  (∀ j, 1 ≤ j ∧ j ≤ k - 2 → d (j + 2) = d (j + 1) + d j) ∧
  (∀ (j : ℕ), 1 ≤ j ∧ j ≤ k → d j ∣ n) ∧
  (∀ (m : ℕ), m ∣ n → m < d 1 ∨ m > d k)

theorem fibby_numbers : ∀ (k : ℕ), is_fibby k → k = 3 ∨ k = 4 :=
sorry

end fibby_numbers_l150_150619


namespace P_started_following_J_l150_150534

theorem P_started_following_J :
  ∀ (t : ℝ),
    (6 * 7.3 + 3 = 8 * (7.3 - t)) → t = 1.45 → t + 12 = 13.45 :=
by
  sorry

end P_started_following_J_l150_150534


namespace product_sum_125_l150_150045

theorem product_sum_125 :
  ∀ (m n : ℕ), m ≥ n ∧
              (∀ (k : ℕ), 0 < k → |Real.log m - Real.log k| < Real.log n → k ≠ 0)
              → (m * n = 125) :=
by sorry

end product_sum_125_l150_150045


namespace speed_faster_train_correct_l150_150382

noncomputable def speed_faster_train_proof
  (time_seconds : ℝ) 
  (speed_slower_train : ℝ)
  (train_length_meters : ℝ) :
  Prop :=
  let time_hours := time_seconds / 3600
  let train_length_km := train_length_meters / 1000
  let total_distance_km := train_length_km + train_length_km
  let relative_speed_km_hr := total_distance_km / time_hours
  let speed_faster_train := relative_speed_km_hr + speed_slower_train
  speed_faster_train = 46

theorem speed_faster_train_correct :
  speed_faster_train_proof 36.00001 36 50.000013888888894 :=
by 
  -- proof steps would go here
  sorry

end speed_faster_train_correct_l150_150382


namespace value_of_x4_plus_1_div_x4_l150_150859

theorem value_of_x4_plus_1_div_x4 (x : ℝ) (hx : x^2 + 1 / x^2 = 2) : x^4 + 1 / x^4 = 2 := 
sorry

end value_of_x4_plus_1_div_x4_l150_150859


namespace rectangular_prism_volume_l150_150191

theorem rectangular_prism_volume
  (l w h : ℝ)
  (h1 : l * w = 15)
  (h2 : w * h = 10)
  (h3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end rectangular_prism_volume_l150_150191


namespace mechanic_worked_days_l150_150285

-- Definitions of conditions as variables
def hourly_rate : ℝ := 60
def hours_per_day : ℝ := 8
def cost_of_parts : ℝ := 2500
def total_amount_paid : ℝ := 9220

-- Definition to calculate the total labor cost
def total_labor_cost : ℝ := total_amount_paid - cost_of_parts

-- Definition to calculate the daily labor cost
def daily_labor_cost : ℝ := hourly_rate * hours_per_day

-- Proof (statement only) that the number of days the mechanic worked on the car is 14
theorem mechanic_worked_days : total_labor_cost / daily_labor_cost = 14 := by
  sorry

end mechanic_worked_days_l150_150285


namespace space_convex_polyhedron_euler_characteristic_l150_150897

-- Definition of space convex polyhedron
structure Polyhedron where
  F : ℕ    -- number of faces
  V : ℕ    -- number of vertices
  E : ℕ    -- number of edges

-- Problem statement: Prove that for any space convex polyhedron, F + V - E = 2
theorem space_convex_polyhedron_euler_characteristic (P : Polyhedron) : P.F + P.V - P.E = 2 := by
  sorry

end space_convex_polyhedron_euler_characteristic_l150_150897


namespace part1_l150_150273

theorem part1 (m : ℝ) (a b : ℝ) (h : m > 0) : 
  ( (a + m * b) / (1 + m) )^2 ≤ (a^2 + m * b^2) / (1 + m) :=
sorry

end part1_l150_150273


namespace parabola_focus_coordinates_l150_150032

noncomputable def parabola_focus (a b : ℝ) := (0, (1 / (4 * a)) + 2)

theorem parabola_focus_coordinates (a b : ℝ) (h₀ : a ≠ 0) (h₁ : ∀ x : ℝ, abs (a * x^2 + b * x + 2) ≥ 2) :
  parabola_focus a b = (0, 2 + (1 / (4 * a))) := sorry

end parabola_focus_coordinates_l150_150032


namespace simplify_and_evaluate_l150_150399

-- Define the variables
variables (x y : ℝ)

-- Define the expression
def expression := 2 * x * y + (3 * x * y - 2 * y^2) - 2 * (x * y - y^2)

-- Introduce the conditions
theorem simplify_and_evaluate : 
  (x = -1) → (y = 2) → expression x y = -6 := 
by 
  intro hx hy 
  sorry

end simplify_and_evaluate_l150_150399


namespace max_value_of_expr_l150_150833

theorem max_value_of_expr (x : ℝ) (h : x ≠ 0) : 
  (∀ y : ℝ, y = (x^2) / (x^6 - 2*x^5 - 2*x^4 + 4*x^3 + 4*x^2 + 16) → y ≤ 1/8) :=
sorry

end max_value_of_expr_l150_150833


namespace find_six_digit_numbers_l150_150067

variable (m n : ℕ)

-- Definition that the original number becomes six-digit when multiplied by 4
def is_six_digit (x : ℕ) : Prop := x ≥ 100000 ∧ x < 1000000

-- Conditions
def original_number := 100 * m + n
def new_number := 10000 * n + m
def satisfies_conditions (m n : ℕ) : Prop :=
  is_six_digit (100 * m + n) ∧
  is_six_digit (10000 * n + m) ∧
  4 * (100 * m + n) = 10000 * n + m

-- Theorem statement
theorem find_six_digit_numbers (h₁ : satisfies_conditions 1428 57)
                               (h₂ : satisfies_conditions 1904 76)
                               (h₃ : satisfies_conditions 2380 95) :
  ∃ m n, satisfies_conditions m n :=
  sorry -- Proof omitted

end find_six_digit_numbers_l150_150067


namespace certain_number_is_11_l150_150677

theorem certain_number_is_11 (x : ℝ) (h : 15 * x = 165) : x = 11 :=
by {
  sorry
}

end certain_number_is_11_l150_150677


namespace costOfBrantsRoyalBananaSplitSundae_l150_150868

-- Define constants for the prices of the known sundaes
def yvette_sundae_cost : ℝ := 9.00
def alicia_sundae_cost : ℝ := 7.50
def josh_sundae_cost : ℝ := 8.50

-- Define the tip percentage
def tip_percentage : ℝ := 0.20

-- Define the final bill amount
def final_bill : ℝ := 42.00

-- Calculate the total known sundaes cost
def total_known_sundaes_cost : ℝ := yvette_sundae_cost + alicia_sundae_cost + josh_sundae_cost

-- Define a proof to show that the cost of Brant's sundae is $10.00
theorem costOfBrantsRoyalBananaSplitSundae : 
  total_known_sundaes_cost + b = final_bill / (1 + tip_percentage) → b = 10 :=
sorry

end costOfBrantsRoyalBananaSplitSundae_l150_150868


namespace bucket_full_weight_l150_150448

variable {a b x y : ℝ}

theorem bucket_full_weight (h1 : x + 2/3 * y = a) (h2 : x + 1/2 * y = b) : 
  (x + y) = 3 * a - 2 * b := 
sorry

end bucket_full_weight_l150_150448


namespace eliminate_duplicates_3n_2m1_l150_150495

theorem eliminate_duplicates_3n_2m1 :
  ∀ k: ℤ, ∃ n m: ℤ, 3 * n ≠ 2 * m + 1 ↔ 2 * m + 1 = 12 * k + 1 ∨ 2 * m + 1 = 12 * k + 5 :=
by
  sorry

end eliminate_duplicates_3n_2m1_l150_150495


namespace gym_monthly_revenue_l150_150925

theorem gym_monthly_revenue (members_per_month_fee : ℕ) (num_members : ℕ) 
  (h1 : members_per_month_fee = 18 * 2) 
  (h2 : num_members = 300) : 
  num_members * members_per_month_fee = 10800 := 
by 
  -- calculation rationale goes here
  sorry

end gym_monthly_revenue_l150_150925


namespace simplify_and_evaluate_l150_150950

-- Define the expression as a function of a and b
def expr (a b : ℚ) : ℚ := 5 * a * b - 2 * (3 * a * b - (4 * a * b^2 + (1/2) * a * b)) - 5 * a * b^2

-- State the condition and the target result
theorem simplify_and_evaluate : 
  let a : ℚ := -1
  let b : ℚ := 1 / 2
  expr a b = -3 / 4 :=
by
  -- Proof goes here
  sorry

end simplify_and_evaluate_l150_150950


namespace circle_area_l150_150929

-- Definition of the given circle equation
def circle_eq (x y : ℝ) : Prop := 3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0

-- Prove the area of the circle defined by circle_eq (x y) is 25/4 * π
theorem circle_area (x y : ℝ) (h : circle_eq x y) : ∃ r : ℝ, r = 5 / 2 ∧ π * r^2 = 25 / 4 * π :=
by
  sorry

end circle_area_l150_150929


namespace alan_tickets_l150_150230

theorem alan_tickets (a m : ℕ) (h1 : a + m = 150) (h2 : m = 5 * a - 6) : a = 26 :=
by
  sorry

end alan_tickets_l150_150230


namespace area_excluding_garden_proof_l150_150547

noncomputable def area_land_excluding_garden (length width r : ℝ) : ℝ :=
  let area_rec := length * width
  let area_circle := Real.pi * (r ^ 2)
  area_rec - area_circle

theorem area_excluding_garden_proof :
  area_land_excluding_garden 8 12 3 = 96 - 9 * Real.pi :=
by
  unfold area_land_excluding_garden
  sorry

end area_excluding_garden_proof_l150_150547


namespace sin_cos_half_angle_sum_l150_150499

theorem sin_cos_half_angle_sum 
  (θ : ℝ)
  (hcos : Real.cos θ = -7/25) 
  (hθ : θ ∈ Set.Ioo (-Real.pi) 0) : 
  Real.sin (θ/2) + Real.cos (θ/2) = -1/5 := 
sorry

end sin_cos_half_angle_sum_l150_150499


namespace probability_all_same_color_l150_150028

theorem probability_all_same_color :
  let total_marbles := 20
  let red_marbles := 5
  let white_marbles := 7
  let blue_marbles := 8
  let total_ways_to_draw_3 := (total_marbles * (total_marbles - 1) * (total_marbles - 2)) / 6
  let ways_to_draw_3_red := (red_marbles * (red_marbles - 1) * (red_marbles - 2)) / 6
  let ways_to_draw_3_white := (white_marbles * (white_marbles - 1) * (white_marbles - 2)) / 6
  let ways_to_draw_3_blue := (blue_marbles * (blue_marbles - 1) * (blue_marbles - 2)) / 6
  let probability := (ways_to_draw_3_red + ways_to_draw_3_white + ways_to_draw_3_blue) / total_ways_to_draw_3
  probability = 101/1140 :=
by
  sorry

end probability_all_same_color_l150_150028


namespace max_GREECE_val_l150_150408

variables (V E R I A G C : ℕ)
noncomputable def verify : Prop :=
  (V * 100 + E * 10 + R - (I * 10 + A)) = G^(R^E) * (G * 100 + R * 10 + E + E * 100 + C * 10 + E) ∧
  G ≠ 0 ∧ E ≠ 0 ∧ V ≠ 0 ∧ I ≠ 0 ∧
  V ≠ E ∧ V ≠ R ∧ V ≠ I ∧ V ≠ A ∧ V ≠ G ∧ V ≠ C ∧
  E ≠ R ∧ E ≠ I ∧ E ≠ A ∧ E ≠ G ∧ E ≠ C ∧
  R ≠ I ∧ R ≠ A ∧ R ≠ G ∧ R ≠ C ∧
  I ≠ A ∧ I ≠ G ∧ I ≠ C ∧
  A ≠ G ∧ A ≠ C ∧
  G ≠ C

theorem max_GREECE_val : ∃ V E R I A G C : ℕ, verify V E R I A G C ∧ (G * 100000 + R * 10000 + E * 1000 + E * 100 + C * 10 + E = 196646) :=
sorry

end max_GREECE_val_l150_150408


namespace tangent_line_eq_at_0_max_min_values_l150_150428

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_eq_at_0 : ∀ x : ℝ, x = 0 → f x = 1 :=
by
  sorry

theorem max_min_values : (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f 0 ≥ f x) ∧ (f (Real.pi / 2) = -Real.pi / 2) :=
by
  sorry

end tangent_line_eq_at_0_max_min_values_l150_150428


namespace roots_of_polynomial_l150_150233

theorem roots_of_polynomial :
  (x^2 - 5 * x + 6) * (x - 1) * (x + 3) = 0 ↔ (x = -3 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
by {
  sorry
}

end roots_of_polynomial_l150_150233


namespace parallel_planes_transitivity_l150_150626

-- Define different planes α, β, γ
variables (α β γ : Plane)

-- Define the parallel relation between planes
axiom parallel : Plane → Plane → Prop

-- Conditions
axiom diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ
axiom β_parallel_α : parallel β α
axiom γ_parallel_α : parallel γ α

-- Statement to prove
theorem parallel_planes_transitivity (α β γ : Plane) 
  (h1 : parallel β α) 
  (h2 : parallel γ α) 
  (h3 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) : parallel β γ :=
sorry

end parallel_planes_transitivity_l150_150626


namespace complement_intersection_l150_150920

def P : Set ℝ := {y | ∃ x, y = (1 / 2) ^ x ∧ 0 < x}
def Q : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_intersection :
  (Set.univ \ P) ∩ Q = {x | 1 ≤ x ∧ x < 2} :=
sorry

end complement_intersection_l150_150920


namespace shared_friends_count_l150_150542

theorem shared_friends_count (james_friends : ℕ) (total_combined : ℕ) (john_factor : ℕ) 
  (h1 : james_friends = 75) 
  (h2 : john_factor = 3) 
  (h3 : total_combined = 275) : 
  james_friends + (john_factor * james_friends) - total_combined = 25 := 
by
  sorry

end shared_friends_count_l150_150542


namespace find_m_of_parallelepiped_volume_l150_150554

theorem find_m_of_parallelepiped_volume 
  {m : ℝ} 
  (h_pos : m > 0) 
  (h_vol : abs (3 * (m^2 - 9) - 2 * (4 * m - 15) + 2 * (12 - 5 * m)) = 20) : 
  m = (9 + Real.sqrt 249) / 6 :=
sorry

end find_m_of_parallelepiped_volume_l150_150554


namespace men_with_tv_at_least_11_l150_150906

-- Definitions for the given conditions
def total_men : ℕ := 100
def married_men : ℕ := 81
def men_with_radio : ℕ := 85
def men_with_ac : ℕ := 70
def men_with_tv_radio_ac_and_married : ℕ := 11

-- The proposition to prove the minimum number of men with TV
theorem men_with_tv_at_least_11 :
  ∃ (T : ℕ), T ≥ men_with_tv_radio_ac_and_married := 
by
  sorry

end men_with_tv_at_least_11_l150_150906


namespace quadratic_equation_from_absolute_value_l150_150881

theorem quadratic_equation_from_absolute_value :
  ∃ b c : ℝ, (∀ x : ℝ, |x - 8| = 3 ↔ x^2 + b * x + c = 0) ∧ (b, c) = (-16, 55) :=
sorry

end quadratic_equation_from_absolute_value_l150_150881


namespace problem_statement_l150_150525

theorem problem_statement
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := 
by
  sorry

end problem_statement_l150_150525


namespace naomi_saw_wheels_l150_150424

theorem naomi_saw_wheels :
  let regular_bikes := 7
  let children's_bikes := 11
  let wheels_per_regular_bike := 2
  let wheels_per_children_bike := 4
  let total_wheels := regular_bikes * wheels_per_regular_bike + children's_bikes * wheels_per_children_bike
  total_wheels = 58 := by
  sorry

end naomi_saw_wheels_l150_150424


namespace percentage_increase_in_savings_l150_150579

theorem percentage_increase_in_savings (I : ℝ) (hI : 0 < I) :
  let E := 0.75 * I
  let S := I - E
  let I_new := 1.20 * I
  let E_new := 0.825 * I
  let S_new := I_new - E_new
  ((S_new - S) / S) * 100 = 50 :=
by
  let E := 0.75 * I
  let S := I - E
  let I_new := 1.20 * I
  let E_new := 0.825 * I
  let S_new := I_new - E_new
  sorry

end percentage_increase_in_savings_l150_150579


namespace area_of_triangle_ABC_l150_150826

theorem area_of_triangle_ABC : 
  let A := (1, 1)
  let B := (4, 1)
  let C := (1, 5)
  let area := 6
  (1:ℝ) * abs (1 * (1 - 5) + 4 * (5 - 1) + 1 * (1 - 1)) / 2 = area := 
by
  sorry

end area_of_triangle_ABC_l150_150826


namespace construction_costs_l150_150339

theorem construction_costs 
  (land_cost_per_sq_meter : ℕ := 50)
  (bricks_cost_per_1000 : ℕ := 100)
  (roof_tile_cost_per_tile : ℕ := 10)
  (land_area : ℕ := 2000)
  (number_of_bricks : ℕ := 10000)
  (number_of_roof_tiles : ℕ := 500) :
  50 * 2000 + (100 / 1000) * 10000 + 10 * 500 = 106000 :=
by sorry

end construction_costs_l150_150339


namespace fraction_sum_l150_150913

theorem fraction_sum : ((10 : ℚ) / 9 + (9 : ℚ) / 10 = 2.0 + (0.1 + 0.1 / 9)) :=
by sorry

end fraction_sum_l150_150913


namespace ratio_xz_y2_l150_150228

-- Define the system of equations
def system (k x y z : ℝ) : Prop := 
  x + k * y + 4 * z = 0 ∧ 
  4 * x + k * y - 3 * z = 0 ∧ 
  3 * x + 5 * y - 4 * z = 0

-- Our main theorem to prove the value of xz / y^2 given the system with k = 7.923
theorem ratio_xz_y2 (x y z : ℝ) (h : system 7.923 x y z) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) : 
  ∃ r : ℝ, r = (x * z) / (y ^ 2) :=
sorry

end ratio_xz_y2_l150_150228


namespace smaller_triangle_area_14_365_l150_150443

noncomputable def smaller_triangle_area (A : ℝ) (H_reduction : ℝ) : ℝ :=
  A * (H_reduction)^2

theorem smaller_triangle_area_14_365 :
  smaller_triangle_area 34 0.65 = 14.365 :=
by
  -- Proof will be provided here
  sorry

end smaller_triangle_area_14_365_l150_150443


namespace inequality_not_always_true_l150_150113

-- Declare the variables and conditions
variables {a b c : ℝ}

-- Given conditions
axiom h1 : a < b 
axiom h2 : b < c 
axiom h3 : a * c < 0

-- Statement of the problem
theorem inequality_not_always_true : ¬ (∀ a b c, (a < b ∧ b < c ∧ a * c < 0) → (c^2 / a < b^2 / a)) :=
by { sorry }

end inequality_not_always_true_l150_150113


namespace fifi_pink_hangers_l150_150415

theorem fifi_pink_hangers :
  ∀ (g b y p : ℕ), 
  g = 4 →
  b = g - 1 →
  y = b - 1 →
  16 = g + b + y + p →
  p = 7 :=
by
  intros
  sorry

end fifi_pink_hangers_l150_150415


namespace arithmetic_sequence_ratio_l150_150904

variable {a_n b_n : ℕ → ℕ}
variable {S_n T_n : ℕ → ℕ}

-- Given two arithmetic sequences a_n and b_n, their sums of the first n terms are S_n and T_n respectively.
-- Given that S_n / T_n = (2n + 2) / (n + 3).
-- Prove that a_10 / b_10 = 20 / 11.

theorem arithmetic_sequence_ratio (h : ∀ n, S_n n / T_n n = (2 * n + 2) / (n + 3)) : (a_n 10) / (b_n 10) = 20 / 11 := 
by
  sorry

end arithmetic_sequence_ratio_l150_150904


namespace part1_part2_part3_l150_150872

open Set

variable (x : ℝ)

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | 2 < x ∧ x < 10}

theorem part1 : A ∩ B = {x | 3 ≤ x ∧ x < 7} :=
sorry

theorem part2 : (Aᶜ : Set ℝ) = {x | x < 3 ∨ x ≥ 7} :=
sorry

theorem part3 : (A ∪ B)ᶜ = {x | x ≤ 2 ∨ x ≥ 10} :=
sorry

end part1_part2_part3_l150_150872


namespace triangle_angle_problem_l150_150451

open Real

-- Define degrees to radians conversion (if necessary)
noncomputable def degrees (d : ℝ) : ℝ := d * π / 180

-- Define the problem conditions and goal
theorem triangle_angle_problem
  (x y : ℝ)
  (h1 : degrees 3 * x + degrees y = degrees 90) :
  x = 18 ∧ y = 36 := by
  sorry

end triangle_angle_problem_l150_150451


namespace ratio_of_x_and_y_l150_150959

theorem ratio_of_x_and_y (x y : ℝ) (h : 0.80 * x = 0.20 * y) : x / y = 0.25 :=
by
  sorry

end ratio_of_x_and_y_l150_150959


namespace part_I_solution_set_part_II_prove_inequality_l150_150729

-- Definition for part (I)
def f (x: ℝ) := |x - 2|
def g (x: ℝ) := 4 - |x - 1|

-- Theorem for part (I)
theorem part_I_solution_set :
  {x : ℝ | f x ≥ g x} = {x : ℝ | x ≤ -1/2} ∪ {x : ℝ | x ≥ 7/2} :=
by sorry

-- Definition for part (II)
def satisfiable_range (a: ℝ) := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def density_equation (m n a: ℝ) := (1 / m) + (1 / (2 * n)) = a

-- Theorem for part (II)
theorem part_II_prove_inequality (m n: ℝ) (hm: 0 < m) (hn: 0 < n) 
  (a: ℝ) (h_a: satisfiable_range a = {x : ℝ | abs (x - a) ≤ 1}) (h_density: density_equation m n a) :
  m + 2 * n ≥ 4 :=
by sorry

end part_I_solution_set_part_II_prove_inequality_l150_150729


namespace subway_boarding_probability_l150_150511

theorem subway_boarding_probability :
  ∀ (total_interval boarding_interval : ℕ),
  total_interval = 10 →
  boarding_interval = 1 →
  (boarding_interval : ℚ) / total_interval = 1 / 10 := by
  intros total_interval boarding_interval ht hb
  rw [hb, ht]
  norm_num

end subway_boarding_probability_l150_150511


namespace quadratic_trinomial_bound_l150_150150

theorem quadratic_trinomial_bound (a b : ℤ) (f : ℝ → ℝ)
  (h_def : ∀ x : ℝ, f x = x^2 + a * x + b)
  (h_bound : ∀ x : ℝ, f x ≥ -9 / 10) :
  ∀ x : ℝ, f x ≥ -1 / 4 :=
sorry

end quadratic_trinomial_bound_l150_150150


namespace solve_for_x_y_l150_150135

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def triangle_ABC (A B C E F : V) (x y : ℝ) : Prop :=
  (E - A) = (1 / 2) • (B - A) ∧
  (C - F) = (2 : ℝ) • (A - F) ∧
  (E - F) = x • (B - A) + y • (C - A)

theorem solve_for_x_y (A B C E F : V) (x y : ℝ) :
  triangle_ABC A B C E F x y →
  x + y = - (1 / 6 : ℝ) :=
by
  sorry

end solve_for_x_y_l150_150135


namespace mean_rest_scores_l150_150342

theorem mean_rest_scores (n : ℕ) (h : 15 < n) 
  (overall_mean : ℝ := 10)
  (mean_of_fifteen : ℝ := 12)
  (total_score : ℝ := n * overall_mean): 
  (180 + p * (n - 15) = total_score) →
  p = (10 * n - 180) / (n - 15) :=
sorry

end mean_rest_scores_l150_150342


namespace concert_people_count_l150_150958

variable {W M : ℕ}

theorem concert_people_count (h1 : W * 2 = M) (h2 : (W - 12) * 3 = M - 29) : W + M = 21 := 
sorry

end concert_people_count_l150_150958


namespace total_height_correct_l150_150816

-- Stack and dimensions setup
def height_of_disc_stack (top_diameter bottom_diameter disc_thickness : ℕ) : ℕ :=
  let num_discs := (top_diameter - bottom_diameter) / 2 + 1
  num_discs * disc_thickness

def total_height (top_diameter bottom_diameter disc_thickness cylinder_height : ℕ) : ℕ :=
  height_of_disc_stack top_diameter bottom_diameter disc_thickness + cylinder_height

-- Given conditions
def top_diameter := 15
def bottom_diameter := 1
def disc_thickness := 2
def cylinder_height := 10
def correct_answer := 26

-- Proof problem
theorem total_height_correct :
  total_height top_diameter bottom_diameter disc_thickness cylinder_height = correct_answer :=
by
  sorry

end total_height_correct_l150_150816


namespace pm_star_eq_6_l150_150363

open Set

-- Definitions based on the conditions
def universal_set : Set ℕ := univ
def M : Set ℕ := {1, 2, 3, 4, 5}
def P : Set ℕ := {2, 3, 6}
def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- The theorem to prove
theorem pm_star_eq_6 : star P M = {6} :=
sorry

end pm_star_eq_6_l150_150363


namespace a_1000_value_l150_150607

theorem a_1000_value :
  ∃ (a : ℕ → ℤ), 
    (a 1 = 2010) ∧
    (a 2 = 2011) ∧
    (∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n + 3) ∧
    (a 1000 = 2676) :=
by {
  -- sorry is used to skip the proof
  sorry 
}

end a_1000_value_l150_150607


namespace ellipse_standard_equation_midpoint_trajectory_equation_l150_150946

theorem ellipse_standard_equation :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ (∀ x y, (x, y) = (2, 0) → x^2 / a^2 + y^2 / b^2 = 1) → (a = 2 ∧ b = 1) :=
sorry

theorem midpoint_trajectory_equation :
  ∀ x y : ℝ,
  (∃ x0 y0 : ℝ, x0 = 2 * x - 1 ∧ y0 = 2 * y - 1 / 2 ∧ (x0^2 / 4 + y0^2 = 1)) →
  (x - 1 / 2)^2 + 4 * (y - 1 / 4)^2 = 1 :=
sorry

end ellipse_standard_equation_midpoint_trajectory_equation_l150_150946


namespace unique_four_digit_number_l150_150160

theorem unique_four_digit_number (N : ℕ) (a : ℕ) (x : ℕ) :
  (N = 1000 * a + x) ∧ (N = 7 * x) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (1 ≤ a ∧ a ≤ 9) →
  N = 3500 :=
by sorry

end unique_four_digit_number_l150_150160


namespace baseball_card_count_l150_150589

-- Define initial conditions
def initial_cards := 15

-- Maria takes half of one more than the number of initial cards
def maria_takes := (initial_cards + 1) / 2

-- Remaining cards after Maria takes her share
def remaining_after_maria := initial_cards - maria_takes

-- You give Peter 1 card
def remaining_after_peter := remaining_after_maria - 1

-- Paul triples the remaining cards
def final_cards := remaining_after_peter * 3

-- Theorem statement to prove
theorem baseball_card_count :
  final_cards = 18 := by
sorry

end baseball_card_count_l150_150589


namespace shortest_distance_between_circles_l150_150858

theorem shortest_distance_between_circles :
  let circle1 := (x^2 - 12*x + y^2 - 6*y + 9 = 0)
  let circle2 := (x^2 + 10*x + y^2 + 8*y + 34 = 0)
  -- Centers and radii from conditions above:
  let center1 := (6, 3)
  let radius1 := 3
  let center2 := (-5, -4)
  let radius2 := Real.sqrt 7
  let distance_centers := Real.sqrt ((6 - (-5))^2 + (3 - (-4))^2)
  -- Calculate shortest distance
  distance_centers - (radius1 + radius2) = Real.sqrt 170 - 3 - Real.sqrt 7 := sorry

end shortest_distance_between_circles_l150_150858


namespace max_wickets_in_innings_l150_150131

-- Define the max wickets a bowler can take per over
def max_wickets_per_over : ℕ := 3

-- Define the number of overs bowled by the bowler
def overs_bowled : ℕ := 6

-- Assume the total players in a cricket team
def total_players : ℕ := 11

-- Lean statement that proves the maximum number of wickets the bowler can take in an innings
theorem max_wickets_in_innings :
  3 * 6 ≥ total_players - 1 →
  max_wickets_per_over * overs_bowled ≥ total_players - 1 :=
by
  sorry

end max_wickets_in_innings_l150_150131


namespace range_of_m_l150_150526

def proposition_p (m : ℝ) : Prop := (m^2 - 4 ≥ 0)
def proposition_q (m : ℝ) : Prop := (4 - 4 * m < 0)
def p_or_q (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def not_p (m : ℝ) : Prop := ¬ proposition_p m

theorem range_of_m (m : ℝ) (h1 : p_or_q m) (h2 : not_p m) : 1 < m ∧ m < 2 :=
sorry

end range_of_m_l150_150526


namespace new_students_l150_150390

theorem new_students (S_i : ℕ) (L : ℕ) (S_f : ℕ) (N : ℕ) 
  (h₁ : S_i = 11) 
  (h₂ : L = 6) 
  (h₃ : S_f = 47) 
  (h₄ : S_f = S_i - L + N) : 
  N = 42 :=
by 
  rw [h₁, h₂, h₃] at h₄
  sorry

end new_students_l150_150390


namespace find_length_of_AL_l150_150046

noncomputable def length_of_AL 
  (A B C L : ℝ) 
  (AB AC AL : ℝ)
  (BC : ℝ)
  (AB_ratio_AC : AB / AC = 5 / 2)
  (BAC_bisector : ∃k, L = k * BC)
  (vector_magnitude : (2 * AB + 5 * AC) = 2016) : Prop :=
  AL = 288

theorem find_length_of_AL 
  (A B C L : ℝ)
  (AB AC AL : ℝ)
  (BC : ℝ)
  (h1 : AB / AC = 5 / 2)
  (h2 : ∃k, L = k * BC)
  (h3 : (2 * AB + 5 * AC) = 2016) : length_of_AL A B C L AB AC AL BC h1 h2 h3 := sorry

end find_length_of_AL_l150_150046


namespace obtain_26_kg_of_sand_l150_150013

theorem obtain_26_kg_of_sand :
  ∃ (x y : ℕ), (37 - x = x + 3) ∧ (20 - y = y + 2) ∧ (x + y = 26) := by
  sorry

end obtain_26_kg_of_sand_l150_150013


namespace average_number_of_carnations_l150_150711

-- Define the number of carnations in each bouquet
def n1 : ℤ := 9
def n2 : ℤ := 23
def n3 : ℤ := 13
def n4 : ℤ := 36
def n5 : ℤ := 28
def n6 : ℤ := 45

-- Define the number of bouquets
def number_of_bouquets : ℤ := 6

-- Prove that the average number of carnations in the bouquets is 25.67
theorem average_number_of_carnations :
  ((n1 + n2 + n3 + n4 + n5 + n6) : ℚ) / (number_of_bouquets : ℚ) = 25.67 := 
by
  sorry

end average_number_of_carnations_l150_150711


namespace possible_values_of_m_l150_150827

theorem possible_values_of_m (m : ℝ) :
  let A := {x | x^2 - 4 * x + 3 = 0}
  let B := {x | ∃ m : ℝ, m * x + 1 = 0}
  (∀ x, x ∈ B → x ∈ A) ↔ m = 0 ∨ m = -1 ∨ m = -1 / 3 :=
by
  let A := {x | x^2 - 4 * x + 3 = 0}
  let B := {x | ∃ m : ℝ, m * x + 1 = 0}
  sorry -- Proof needed

end possible_values_of_m_l150_150827


namespace marbles_total_l150_150672

theorem marbles_total (fabian kyle miles : ℕ) (h1 : fabian = 3 * kyle) (h2 : fabian = 5 * miles) (h3 : fabian = 15) : kyle + miles = 8 := by
  sorry

end marbles_total_l150_150672


namespace value_of_x_l150_150026

theorem value_of_x (x y : ℕ) (h1 : x / y = 3) (h2 : y = 25) : x = 75 := by
  sorry

end value_of_x_l150_150026


namespace f_2010_plus_f_2011_l150_150462

-- Definition of f being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Conditions in Lean 4
variables (f : ℝ → ℝ)

axiom f_odd : odd_function f
axiom f_symmetry : ∀ x, f (1 + x) = f (1 - x)
axiom f_1 : f 1 = 2

-- The theorem to be proved
theorem f_2010_plus_f_2011 : f (2010) + f (2011) = -2 :=
by
  sorry

end f_2010_plus_f_2011_l150_150462


namespace rooms_with_two_beds_l150_150683

variable (x y : ℕ)

theorem rooms_with_two_beds:
  x + y = 13 →
  2 * x + 3 * y = 31 →
  x = 8 :=
by
  intros h1 h2
  sorry

end rooms_with_two_beds_l150_150683


namespace set_operation_example_l150_150356

def set_operation (A B : Set ℝ) := {x | (x ∈ A ∪ B) ∧ (x ∉ A ∩ B)}

def M := {x : ℝ | -2 < x ∧ x < 2}
def N := {x : ℝ | 1 < x ∧ x < 3}

theorem set_operation_example : set_operation M N = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (2 ≤ x ∧ x < 3)} :=
by {
  sorry
}

end set_operation_example_l150_150356


namespace counting_4digit_integers_l150_150309

theorem counting_4digit_integers (x y : ℕ) (a b c d : ℕ) :
  (x = 1000 * a + 100 * b + 10 * c + d) →
  (y = 1000 * d + 100 * c + 10 * b + a) →
  (y - x = 3177) →
  (1 ≤ a) → (a ≤ 6) →
  (0 ≤ b) → (b ≤ 7) →
  (c = b + 2) →
  (d = a + 3) →
  ∃ n : ℕ, n = 48 := 
sorry

end counting_4digit_integers_l150_150309


namespace interest_rate_increase_60_percent_l150_150287

noncomputable def percentage_increase (A P A' t : ℝ) : ℝ :=
  let r₁ := (A - P) / (P * t)
  let r₂ := (A' - P) / (P * t)
  ((r₂ - r₁) / r₁) * 100

theorem interest_rate_increase_60_percent :
  percentage_increase 920 800 992 3 = 60 := by
  sorry

end interest_rate_increase_60_percent_l150_150287


namespace balls_in_boxes_l150_150745

theorem balls_in_boxes : 
  (number_of_ways : ℕ) = 52 :=
by
  let number_of_balls := 5
  let number_of_boxes := 4
  let balls_indistinguishable := true
  let boxes_distinguishable := true
  let max_balls_per_box := 3
  
  -- Proof omitted
  sorry

end balls_in_boxes_l150_150745


namespace sin_cos_sum_eq_one_or_neg_one_l150_150869

theorem sin_cos_sum_eq_one_or_neg_one (α : ℝ) (h : (Real.sin α)^4 + (Real.cos α)^4 = 1) : (Real.sin α + Real.cos α) = 1 ∨ (Real.sin α + Real.cos α) = -1 :=
sorry

end sin_cos_sum_eq_one_or_neg_one_l150_150869


namespace tangent_line_at_2_is_12x_minus_y_minus_17_eq_0_range_of_m_for_three_distinct_real_roots_l150_150874

-- Define the function f
noncomputable def f (x : ℝ) := 2 * x^3 - 3 * x^2 + 3

-- First proof problem: Equation of the tangent line at (2, 7)
theorem tangent_line_at_2_is_12x_minus_y_minus_17_eq_0 :
  ∀ x y : ℝ, y = f x → (x = 2) → y = 7 → (∃ (m b : ℝ), (m = 12) ∧ (b = -17) ∧ (∀ x, 12 * x - y - 17 = 0)) :=
by
  sorry

-- Second proof problem: Range of m for three distinct real roots
theorem range_of_m_for_three_distinct_real_roots :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) → -3 < m ∧ m < -2 :=
by 
  sorry

end tangent_line_at_2_is_12x_minus_y_minus_17_eq_0_range_of_m_for_three_distinct_real_roots_l150_150874


namespace max_value_expression_le_380_l150_150702

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_expression_le_380 (a b c d : ℝ)
  (ha : -9.5 ≤ a ∧ a ≤ 9.5)
  (hb : -9.5 ≤ b ∧ b ≤ 9.5)
  (hc : -9.5 ≤ c ∧ c ≤ 9.5)
  (hd : -9.5 ≤ d ∧ d ≤ 9.5) :
  max_value_expression a b c d ≤ 380 :=
sorry

end max_value_expression_le_380_l150_150702


namespace intersection_P_Q_l150_150967

def P : Set ℝ := { x | x^2 - x = 0 }
def Q : Set ℝ := { x | x^2 + x = 0 }

theorem intersection_P_Q : (P ∩ Q) = {0} := 
by
  sorry

end intersection_P_Q_l150_150967


namespace race_dead_heat_l150_150998

variable (v_B v_A L x : ℝ)

theorem race_dead_heat (h : v_A = 17 / 14 * v_B) : x = 3 / 17 * L :=
by
  sorry

end race_dead_heat_l150_150998


namespace distinct_seatings_l150_150050

theorem distinct_seatings : 
  ∃ n : ℕ, (n = 288000) ∧ 
  (∀ (men wives : Fin 6 → ℕ),
  ∃ (f : (Fin 12) → ℕ), 
  (∀ i, f (i + 1) % 12 ≠ f i) ∧
  (∀ i, f i % 2 = 0) ∧
  (∀ j, f (2 * j) = men j ∧ f (2 * j + 1) = wives j)) :=
by
  sorry

end distinct_seatings_l150_150050


namespace function_range_y_eq_1_div_x_minus_2_l150_150908

theorem function_range_y_eq_1_div_x_minus_2 (x : ℝ) : (∀ y : ℝ, y = 1 / (x - 2) ↔ x ∈ {x : ℝ | x ≠ 2}) :=
sorry

end function_range_y_eq_1_div_x_minus_2_l150_150908


namespace five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one_l150_150351

theorem five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one (n : ℕ) (hn : n > 0) : ¬ (4 ^ n - 1 ∣ 5 ^ n - 1) :=
sorry

end five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one_l150_150351


namespace smallest_n_satisfies_condition_l150_150680

theorem smallest_n_satisfies_condition : 
  ∃ (n : ℕ), n = 1806 ∧ ∀ (p : ℕ), Nat.Prime p → n % (p - 1) = 0 → n % p = 0 := 
sorry

end smallest_n_satisfies_condition_l150_150680


namespace two_std_dev_less_than_mean_l150_150618

def mean : ℝ := 14.0
def std_dev : ℝ := 1.5

theorem two_std_dev_less_than_mean : (mean - 2 * std_dev) = 11.0 := 
by sorry

end two_std_dev_less_than_mean_l150_150618


namespace hotel_charge_problem_l150_150346

theorem hotel_charge_problem (R G P : ℝ) 
  (h1 : P = 0.5 * R) 
  (h2 : P = 0.9 * G) : 
  (R - G) / G * 100 = 80 :=
by
  sorry

end hotel_charge_problem_l150_150346


namespace john_february_bill_l150_150231

-- Define the conditions as constants
def base_cost : ℝ := 25
def cost_per_text : ℝ := 0.1 -- 10 cents
def cost_per_over_minute : ℝ := 0.1 -- 10 cents
def texts_sent : ℝ := 200
def hours_talked : ℝ := 51
def included_hours : ℝ := 50
def minutes_per_hour : ℝ := 60

-- Total cost computation
def total_cost : ℝ :=
  base_cost +
  (texts_sent * cost_per_text) +
  ((hours_talked - included_hours) * minutes_per_hour * cost_per_over_minute)

-- Proof statement
theorem john_february_bill : total_cost = 51 := by
  -- Proof omitted
  sorry

end john_february_bill_l150_150231


namespace sum_of_coefficients_l150_150020

noncomputable def P (x : ℤ) : ℤ := (x ^ 2 - 3 * x + 1) ^ 100

theorem sum_of_coefficients : P 1 = 1 := by
  sorry

end sum_of_coefficients_l150_150020


namespace sum_of_integers_l150_150649

theorem sum_of_integers (a b : ℕ) (h1 : a * b + a + b = 255) (h2 : a < 30) (h3 : b < 30) (h4 : a % 2 = 1) :
  a + b = 30 := 
sorry

end sum_of_integers_l150_150649


namespace short_answer_question_time_l150_150736

-- Definitions from the conditions
def minutes_per_paragraph := 15
def minutes_per_essay := 60
def num_essays := 2
def num_paragraphs := 5
def num_short_answer_questions := 15
def total_minutes := 4 * 60

-- Auxiliary calculations
def total_minutes_essays := num_essays * minutes_per_essay
def total_minutes_paragraphs := num_paragraphs * minutes_per_paragraph
def total_minutes_used := total_minutes_essays + total_minutes_paragraphs

-- The time per short-answer question is 3 minutes
theorem short_answer_question_time (x : ℕ) : (total_minutes - total_minutes_used) / num_short_answer_questions = 3 :=
by
  -- x is defined as the time per short-answer question
  let x := (total_minutes - total_minutes_used) / num_short_answer_questions
  have time_for_short_answer_questions : total_minutes - total_minutes_used = 45 := by sorry
  have time_per_short_answer_question : 45 / num_short_answer_questions = 3 := by sorry
  have x_equals_3 : x = 3 := by sorry
  exact x_equals_3

end short_answer_question_time_l150_150736


namespace volume_box_l150_150426

theorem volume_box (x y : ℝ) :
  (16 - 2 * x) * (12 - 2 * y) * y = 4 * x * y ^ 2 - 24 * x * y + 192 * y - 32 * y ^ 2 :=
by sorry

end volume_box_l150_150426


namespace constant_term_binomial_expansion_l150_150532

theorem constant_term_binomial_expansion (a : ℝ) (h : 15 * a^2 = 120) : a = 2 * Real.sqrt 2 :=
sorry

end constant_term_binomial_expansion_l150_150532


namespace points_on_circle_l150_150422

theorem points_on_circle (t : ℝ) : (∃ (x y : ℝ), x = Real.cos (2 * t) ∧ y = Real.sin (2 * t) ∧ (x^2 + y^2 = 1)) := by
  sorry

end points_on_circle_l150_150422


namespace julia_money_remaining_l150_150060

theorem julia_money_remaining 
  (initial_amount : ℝ)
  (tablet_percentage : ℝ)
  (phone_percentage : ℝ)
  (game_percentage : ℝ)
  (case_percentage : ℝ) 
  (final_money : ℝ) :
  initial_amount = 120 → 
  tablet_percentage = 0.45 → 
  phone_percentage = 1/3 → 
  game_percentage = 0.25 → 
  case_percentage = 0.10 → 
  final_money = initial_amount * (1 - tablet_percentage) * (1 - phone_percentage) * (1 - game_percentage) * (1 - case_percentage) →
  final_money = 29.70 :=
by
  intros
  sorry

end julia_money_remaining_l150_150060


namespace problem_1_problem_2_l150_150840

-- Definitions for sets A and B
def A : Set ℝ := {x | x^2 - 2 * x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x | abs (x - 1) < a}

-- Define the first problem statement: If A ⊂ B, then a > 2.
theorem problem_1 (a : ℝ) : (A ⊂ B a) → (2 < a) := by
  sorry

-- Define the second problem statement: If B ⊂ A, then a ≤ 0 or (0 < a < 2).
theorem problem_2 (a : ℝ) : (B a ⊂ A) → (a ≤ 0 ∨ (0 < a ∧ a < 2)) := by
  sorry

end problem_1_problem_2_l150_150840


namespace ratio_of_bottles_given_to_first_house_l150_150852

theorem ratio_of_bottles_given_to_first_house 
  (total_bottles : ℕ) 
  (bottles_only_cider : ℕ) 
  (bottles_only_beer : ℕ) 
  (bottles_mixed : ℕ) 
  (first_house_bottles : ℕ) 
  (h1 : total_bottles = 180) 
  (h2 : bottles_only_cider = 40) 
  (h3 : bottles_only_beer = 80) 
  (h4 : bottles_mixed = total_bottles - bottles_only_cider - bottles_only_beer) 
  (h5 : first_house_bottles = 90) : 
  first_house_bottles / total_bottles = 1 / 2 :=
by 
  -- Proof goes here
  sorry

end ratio_of_bottles_given_to_first_house_l150_150852


namespace remaining_stock_is_120_l150_150052

-- Definitions derived from conditions
def green_beans_weight : ℕ := 60
def rice_weight : ℕ := green_beans_weight - 30
def sugar_weight : ℕ := green_beans_weight - 10
def rice_lost_weight : ℕ := rice_weight / 3
def sugar_lost_weight : ℕ := sugar_weight / 5
def remaining_rice : ℕ := rice_weight - rice_lost_weight
def remaining_sugar : ℕ := sugar_weight - sugar_lost_weight
def remaining_stock_weight : ℕ := remaining_rice + remaining_sugar + green_beans_weight

-- Theorem
theorem remaining_stock_is_120 : remaining_stock_weight = 120 := by
  sorry

end remaining_stock_is_120_l150_150052


namespace ships_meeting_count_l150_150326

theorem ships_meeting_count :
  ∀ (n : ℕ) (east_sailing west_sailing : ℕ),
    n = 10 →
    east_sailing = 5 →
    west_sailing = 5 →
    east_sailing + west_sailing = n →
    (∀ (v : ℕ), v > 0) →
    25 = east_sailing * west_sailing :=
by
  intros n east_sailing west_sailing h1 h2 h3 h4 h5
  sorry

end ships_meeting_count_l150_150326


namespace conversion_problems_l150_150137

-- Define the conversion factors
def square_meters_to_hectares (sqm : ℕ) : ℕ := sqm / 10000
def hectares_to_square_kilometers (ha : ℕ) : ℕ := ha / 100
def square_kilometers_to_hectares (sqkm : ℕ) : ℕ := sqkm * 100

-- Define the specific values from the problem
def value1_m2 : ℕ := 5000000
def value2_km2 : ℕ := 70000

-- The theorem to prove
theorem conversion_problems :
  (square_meters_to_hectares value1_m2 = 500) ∧
  (hectares_to_square_kilometers 500 = 5) ∧
  (square_kilometers_to_hectares value2_km2 = 7000000) :=
by
  sorry

end conversion_problems_l150_150137


namespace point_on_circle_l150_150776

theorem point_on_circle 
    (P : ℝ × ℝ) 
    (h_l1 : 2 * P.1 - 3 * P.2 + 4 = 0)
    (h_l2 : 3 * P.1 - 2 * P.2 + 1 = 0) 
    (h_circle : (P.1 - 2) ^ 2 + (P.2 - 4) ^ 2 = 5) : 
    (P.1 - 2) ^ 2 + (P.2 - 4) ^ 2 = 5 :=
by
  sorry

end point_on_circle_l150_150776


namespace min_value_ineq_l150_150011

open Real

theorem min_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 2) * (y^2 + 5 * y + 2) * (z^2 + 5 * z + 2) / (x * y * z) ≥ 512 :=
by sorry

noncomputable def optimal_min_value : ℝ := 512

end min_value_ineq_l150_150011


namespace min_value_l150_150274

theorem min_value (a : ℝ) (h : a > 1) : a + 1 / (a - 1) ≥ 3 :=
sorry

end min_value_l150_150274


namespace pow_zero_eq_one_l150_150830

theorem pow_zero_eq_one : (-2023)^0 = 1 :=
by
  -- The proof of this theorem will go here.
  sorry

end pow_zero_eq_one_l150_150830


namespace height_difference_is_correct_l150_150138

-- Define the heights of the trees as rational numbers.
def maple_tree_height : ℚ := 10 + 1 / 4
def spruce_tree_height : ℚ := 14 + 1 / 2

-- Prove that the spruce tree is 19 3/4 feet taller than the maple tree.
theorem height_difference_is_correct :
  spruce_tree_height - maple_tree_height = 19 + 3 / 4 := 
sorry

end height_difference_is_correct_l150_150138


namespace alpha_value_l150_150695

theorem alpha_value (α : ℝ) (h : 0 ≤ α ∧ α ≤ 2 * Real.pi 
    ∧ ∃β : ℝ, β = 2 * Real.pi / 3 ∧ (Real.sin β, Real.cos β) = (Real.sin α, Real.cos α)) : 
    α = 5 * Real.pi / 3 := 
  by
    sorry

end alpha_value_l150_150695


namespace expression_even_nat_l150_150698

theorem expression_even_nat (m n : ℕ) : 
  2 ∣ (5 * m + n + 1) * (3 * m - n + 4) := 
sorry

end expression_even_nat_l150_150698


namespace calculate_total_cost_l150_150988

def num_chicken_nuggets := 100
def num_per_box := 20
def cost_per_box := 4

theorem calculate_total_cost :
  (num_chicken_nuggets / num_per_box) * cost_per_box = 20 := by
  sorry

end calculate_total_cost_l150_150988


namespace percentage_chromium_first_alloy_l150_150234

theorem percentage_chromium_first_alloy 
  (x : ℝ) (w1 w2 : ℝ) (p2 p_new : ℝ) 
  (h1 : w1 = 10) 
  (h2 : w2 = 30) 
  (h3 : p2 = 0.08)
  (h4 : p_new = 0.09):
  ((x / 100) * w1 + p2 * w2) = p_new * (w1 + w2) → x = 12 :=
by
  sorry

end percentage_chromium_first_alloy_l150_150234


namespace fraction_ordering_l150_150380

theorem fraction_ordering :
  (8 : ℚ) / 24 < (6 : ℚ) / 17 ∧ (6 : ℚ) / 17 < (10 : ℚ) / 27 :=
by
  sorry

end fraction_ordering_l150_150380


namespace calculate_fraction_l150_150125

variables (n_bl: ℕ) (deg_warm: ℕ) (total_deg: ℕ) (total_bl: ℕ)

def blanket_fraction_added := total_deg / deg_warm

theorem calculate_fraction (h1: deg_warm = 3) (h2: total_deg = 21) (h3: total_bl = 14) :
  (blanket_fraction_added total_deg deg_warm) / total_bl = 1 / 2 :=
by {
  sorry
}

end calculate_fraction_l150_150125


namespace evaluate_expression_l150_150364

def ceil (x : ℚ) : ℤ := sorry -- Implement the ceiling function for rational numbers as needed

theorem evaluate_expression :
  (ceil ((23 : ℚ) / 9 - ceil ((35 : ℚ) / 23))) 
  / (ceil ((35 : ℚ) / 9 + ceil ((9 * 23 : ℚ) / 35))) = (1 / 10 : ℚ) :=
by
  intros
  -- Proof goes here
  sorry

end evaluate_expression_l150_150364


namespace cost_of_each_steak_meal_l150_150348

variable (x : ℝ)

theorem cost_of_each_steak_meal :
  (2 * x + 2 * 3.5 + 3 * 2 = 99 - 38) → x = 24 := 
by
  intro h
  sorry

end cost_of_each_steak_meal_l150_150348


namespace allan_balloons_l150_150472

theorem allan_balloons (a j t : ℕ) (h1 : t = 6) (h2 : j = 4) (h3 : t = a + j) : a = 2 := by
  sorry

end allan_balloons_l150_150472


namespace value_of_expression_l150_150188

variable (x y : ℝ)

theorem value_of_expression (h1 : x + y = 6) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 228498 := by
  sorry

end value_of_expression_l150_150188


namespace factorize_ax_squared_minus_9a_l150_150855

theorem factorize_ax_squared_minus_9a (a x : ℝ) : 
  a * x^2 - 9 * a = a * (x - 3) * (x + 3) :=
sorry

end factorize_ax_squared_minus_9a_l150_150855


namespace least_x_divisible_by_3_l150_150595

theorem least_x_divisible_by_3 : ∃ x : ℕ, (∀ y : ℕ, (2 + 3 + 5 + 7 + y) % 3 = 0 → y = 1) :=
by
  sorry

end least_x_divisible_by_3_l150_150595


namespace least_three_digit_with_factors_l150_150597

theorem least_three_digit_with_factors (n : ℕ) :
  (n ≥ 100 ∧ n < 1000 ∧ 2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 3 ∣ n) → n = 210 := by
  sorry

end least_three_digit_with_factors_l150_150597


namespace quadratic_inequality_solution_l150_150817

theorem quadratic_inequality_solution (a b: ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 + a * x + b < 0) ∧
  (2 + 3 = -a) ∧
  (2 * 3 = b) →
  ∀ x : ℝ, (b * x^2 + a * x + 1 > 0) ↔ (x < 1/3 ∨ x > 1/2) :=
by
  sorry

end quadratic_inequality_solution_l150_150817


namespace total_score_is_938_l150_150366

-- Define the average score condition
def average_score (S : ℤ) : Prop := 85.25 ≤ (S : ℚ) / 11 ∧ (S : ℚ) / 11 < 85.35

-- Define the condition that each student's score is an integer
def total_score (S : ℤ) : Prop := average_score S ∧ ∃ n : ℕ, S = n

-- Lean 4 statement for the proof problem
theorem total_score_is_938 : ∃ S : ℤ, total_score S ∧ S = 938 :=
by
  sorry

end total_score_is_938_l150_150366


namespace product_approximation_l150_150455

-- Define the approximation condition
def approxProduct (x y : ℕ) (approxX approxY : ℕ) : ℕ :=
  approxX * approxY

-- State the theorem
theorem product_approximation :
  let x := 29
  let y := 32
  let approxX := 30
  let approxY := 30
  approxProduct x y approxX approxY = 900 := by
  sorry

end product_approximation_l150_150455


namespace piecewise_function_continuity_l150_150486

theorem piecewise_function_continuity :
  (∃ a c : ℝ, (2 * a * 2 + 4 = 2^2 - 2) ∧ (4 - 2 = 3 * (-2) - c) ∧ a + c = -17 / 2) :=
by
  sorry

end piecewise_function_continuity_l150_150486


namespace bounces_less_than_50_l150_150203

noncomputable def minBouncesNeeded (initialHeight : ℝ) (bounceFactor : ℝ) (thresholdHeight : ℝ) : ℕ :=
  ⌈(Real.log (thresholdHeight / initialHeight) / Real.log (bounceFactor))⌉₊

theorem bounces_less_than_50 :
  minBouncesNeeded 360 (3/4 : ℝ) 50 = 8 :=
by
  sorry

end bounces_less_than_50_l150_150203


namespace worked_days_proof_l150_150815

theorem worked_days_proof (W N : ℕ) (hN : N = 24) (h0 : 100 * W = 25 * N) : W + N = 30 :=
by
  sorry

end worked_days_proof_l150_150815


namespace part1_part2_l150_150871

noncomputable def f (a x : ℝ) : ℝ := x^2 + (a+1)*x + a

theorem part1 (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → f a x < 0) → a ≤ -2 := sorry

theorem part2 (a x : ℝ) :
  f a x > 0 ↔
  (a > 1 ∧ (x < -a ∨ x > -1)) ∨
  (a = 1 ∧ x ≠ -1) ∨
  (a < 1 ∧ (x < -1 ∨ x > -a)) := sorry

end part1_part2_l150_150871


namespace arithmetic_series_sum_l150_150265

variable (a₁ aₙ d S : ℝ)
variable (n : ℕ)

-- Defining the conditions (a₁, aₙ, d, and the formula for arithmetic series sum)
def first_term : a₁ = 10 := sorry
def last_term : aₙ = 70 := sorry
def common_diff : d = 1 / 7 := sorry

-- Equation to find number of terms (n)
def find_n : 70 = 10 + (n - 1) * (1 / 7) := sorry

-- Formula for the sum of an arithmetic series
def series_sum : S = (n * (10 + 70)) / 2 := sorry

-- The proof problem statement
theorem arithmetic_series_sum : 
  a₁ = 10 → 
  aₙ = 70 → 
  d = 1 / 7 → 
  (70 = 10 + (n - 1) * (1 / 7)) → 
  S = (n * (10 + 70)) / 2 → 
  S = 16840 := by 
  intros h1 h2 h3 h4 h5 
  -- proof steps would go here
  sorry

end arithmetic_series_sum_l150_150265


namespace train_speed_is_42_point_3_km_per_h_l150_150940

-- Definitions for the conditions.
def train_length : ℝ := 150
def bridge_length : ℝ := 320
def crossing_time : ℝ := 40
def meter_per_sec_to_km_per_hour : ℝ := 3.6
def total_distance : ℝ := train_length + bridge_length

-- The theorem we want to prove
theorem train_speed_is_42_point_3_km_per_h : 
    (total_distance / crossing_time) * meter_per_sec_to_km_per_hour = 42.3 :=
by 
    -- Proof omitted
    sorry

end train_speed_is_42_point_3_km_per_h_l150_150940


namespace unit_price_ratio_l150_150467

theorem unit_price_ratio (v p : ℝ) (hv : 0 < v) (hp : 0 < p) :
  (1.1 * p / (1.4 * v)) / (0.85 * p / (1.3 * v)) = 13 / 11 :=
by
  sorry

end unit_price_ratio_l150_150467


namespace find_f_neg2_l150_150002

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - 3 else -(2^(-x) - 3)

theorem find_f_neg2 : f (-2) = -1 :=
sorry

end find_f_neg2_l150_150002


namespace cos2_minus_sin2_pi_over_12_l150_150727

theorem cos2_minus_sin2_pi_over_12 : 
  (Real.cos (Real.pi / 12))^2 - (Real.sin (Real.pi / 12))^2 = Real.cos (Real.pi / 6) := 
by
  sorry

end cos2_minus_sin2_pi_over_12_l150_150727


namespace expression_values_l150_150432

noncomputable def sign (x : ℝ) : ℝ := 
if x > 0 then 1 else -1

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ v ∈ ({-4, 0, 4} : Set ℝ), 
    sign a + sign b + sign c + sign (a * b * c) = v := by
  sorry

end expression_values_l150_150432


namespace three_wheels_possible_two_wheels_not_possible_l150_150773

-- Define the conditions as hypotheses
def wheels_spokes (total_spokes_visible : ℕ) (max_spokes_per_wheel : ℕ) (wheels : ℕ) : Prop :=
  total_spokes_visible >= wheels * max_spokes_per_wheel ∧ wheels ≥ 1

-- Prove if a) three wheels is a possible solution
theorem three_wheels_possible : ∃ wheels, wheels = 3 ∧ wheels_spokes 7 3 wheels := by
  sorry

-- Prove if b) two wheels is not a possible solution
theorem two_wheels_not_possible : ¬ ∃ wheels, wheels = 2 ∧ wheels_spokes 7 3 wheels := by
  sorry

end three_wheels_possible_two_wheels_not_possible_l150_150773


namespace repeating_decimals_sum_l150_150225

theorem repeating_decimals_sum : 
  (0.3333333333333333 : ℝ) + (0.0404040404040404 : ℝ) + (0.005005005005005 : ℝ) = (14 / 37 : ℝ) :=
by {
  sorry
}

end repeating_decimals_sum_l150_150225


namespace range_of_f_l150_150086

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x + 1 else 2 ^ x

theorem range_of_f :
  {x : ℝ | f x + f (x - 0.5) > 1} = {x : ℝ | x > -0.25} :=
by
  sorry

end range_of_f_l150_150086


namespace percentage_of_a_l150_150388

theorem percentage_of_a (x a : ℝ) (paise_in_rupee : ℝ := 100) (a_value : a = 160 * paise_in_rupee) (h : (x / 100) * a = 80) : x = 0.5 :=
by sorry

end percentage_of_a_l150_150388


namespace calculate_expression_l150_150487

theorem calculate_expression (a b c : ℤ) (ha : a = 3) (hb : b = 7) (hc : c = 2) :
  ((a * b - c) - (a + b * c)) - ((a * c - b) - (a - b * c)) = -8 :=
by
  rw [ha, hb, hc]  -- Substitute a, b, c with 3, 7, 2 respectively
  sorry  -- Placeholder for the proof

end calculate_expression_l150_150487


namespace quarts_of_water_required_l150_150286

-- Define the ratio of water to juice
def ratio_water_to_juice : Nat := 5 / 3

-- Define the total punch to prepare in gallons
def total_punch_in_gallons : Nat := 2

-- Define the conversion factor from gallons to quarts
def quarts_per_gallon : Nat := 4

-- Define the total number of parts
def total_parts : Nat := 5 + 3

-- Define the total punch in quarts
def total_punch_in_quarts : Nat := total_punch_in_gallons * quarts_per_gallon

-- Define the amount of water per part
def quarts_per_part : Nat := total_punch_in_quarts / total_parts

-- Prove the required amount of water in quarts
theorem quarts_of_water_required : quarts_per_part * 5 = 5 := 
by
  -- Proof is omitted, represented by sorry
  sorry

end quarts_of_water_required_l150_150286


namespace final_score_l150_150582

def dart1 : ℕ := 50
def dart2 : ℕ := 0
def dart3 : ℕ := dart1 / 2

theorem final_score : dart1 + dart2 + dart3 = 75 := by
  sorry

end final_score_l150_150582


namespace ratio_of_students_to_professors_l150_150710

theorem ratio_of_students_to_professors (total : ℕ) (students : ℕ) (professors : ℕ)
  (h1 : total = 40000) (h2 : students = 37500) (h3 : total = students + professors) :
  students / professors = 15 :=
by
  sorry

end ratio_of_students_to_professors_l150_150710


namespace ellipse_properties_l150_150450

theorem ellipse_properties 
  (foci1 foci2 : ℝ × ℝ) 
  (point_on_ellipse : ℝ × ℝ) 
  (h k a b : ℝ) 
  (a_pos : a > 0) 
  (b_pos : b > 0) 
  (ellipse_condition : foci1 = (-4, 1) ∧ foci2 = (-4, 5) ∧ point_on_ellipse = (1, 3))
  (ellipse_eqn : (x y : ℝ) → ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1) :
  a + k = 8 :=
by
  sorry

end ellipse_properties_l150_150450


namespace smallest_value_for_x_9_l150_150328

theorem smallest_value_for_x_9 :
  let x := 9
  ∃ i, i = (8 / (x + 2)) ∧ 
  (i < (8 / x) ∧ 
   i < (8 / (x - 2)) ∧ 
   i < (x / 8) ∧ 
   i < ((x + 2) / 8)) :=
by
  let x := 9
  use (8 / (x + 2))
  sorry

end smallest_value_for_x_9_l150_150328


namespace find_k_for_linear_dependence_l150_150104

structure vector2 :=
  (x : ℝ)
  (y : ℝ)

def linear_dependent (v1 v2 : vector2) :=
  ∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧
  c1 * v1.x + c2 * v2.x = 0 ∧
  c1 * v1.y + c2 * v2.y = 0

theorem find_k_for_linear_dependence :
  ∀ (k : ℝ), linear_dependent (vector2.mk 2 3) (vector2.mk 4 k) ↔ k = 6 :=
by sorry

end find_k_for_linear_dependence_l150_150104


namespace find_g_neg_three_l150_150981

namespace ProofProblem

def g (d e f x : ℝ) : ℝ := d * x^5 + e * x^3 + f * x + 6

theorem find_g_neg_three (d e f : ℝ) (h : g d e f 3 = -9) : g d e f (-3) = 21 := by
  sorry

end ProofProblem

end find_g_neg_three_l150_150981


namespace find_value_of_b_l150_150846

theorem find_value_of_b (x b : ℕ) 
    (h1 : 5 * (x + 8) = 5 * x + b + 33) : b = 7 :=
sorry

end find_value_of_b_l150_150846


namespace simplify_expression_l150_150726

theorem simplify_expression (x y : ℝ) (m : ℤ) : 
  ((x + y)^(2 * m + 1) / (x + y)^(m - 1) = (x + y)^(m + 2)) :=
by sorry

end simplify_expression_l150_150726


namespace avg_fish_in_bodies_of_water_l150_150884

def BoastPoolFish : ℕ := 75
def OnumLakeFish : ℕ := BoastPoolFish + 25
def RiddlePondFish : ℕ := OnumLakeFish / 2
def RippleCreekFish : ℕ := 2 * (OnumLakeFish - BoastPoolFish)
def WhisperingSpringsFish : ℕ := (3 * RiddlePondFish) / 2

def totalFish : ℕ := BoastPoolFish + OnumLakeFish + RiddlePondFish + RippleCreekFish + WhisperingSpringsFish
def averageFish : ℕ := totalFish / 5

theorem avg_fish_in_bodies_of_water : averageFish = 68 :=
by
  sorry

end avg_fish_in_bodies_of_water_l150_150884


namespace solve_equation_l150_150014

theorem solve_equation (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a^2 = b * (b + 7) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
by
  sorry

end solve_equation_l150_150014


namespace judy_expense_correct_l150_150814

noncomputable def judy_expense : ℝ :=
  let carrots := 5 * 1
  let milk := 3 * 3
  let pineapples := 2 * 4
  let original_flour_price := 5
  let discount := original_flour_price * 0.25
  let discounted_flour_price := original_flour_price - discount
  let flour := 2 * discounted_flour_price
  let ice_cream := 7
  let total_no_coupon := carrots + milk + pineapples + flour + ice_cream
  if total_no_coupon >= 30 then total_no_coupon - 10 else total_no_coupon

theorem judy_expense_correct : judy_expense = 26.5 := by
  sorry

end judy_expense_correct_l150_150814


namespace sum_of_digits_smallest_N_l150_150910

theorem sum_of_digits_smallest_N :
  ∃ (N : ℕ), N ≤ 999 ∧ 72 * N < 1000 ∧ (N = 13) ∧ (1 + 3 = 4) := by
  sorry

end sum_of_digits_smallest_N_l150_150910


namespace xiao_wang_original_plan_l150_150572

theorem xiao_wang_original_plan (p d1 extra_pages : ℕ) (original_days : ℝ) (x : ℝ) 
  (h1 : p = 200)
  (h2 : d1 = 5)
  (h3 : extra_pages = 5)
  (h4 : original_days = p / x)
  (h5 : original_days - 1 = d1 + (p - (d1 * x)) / (x + extra_pages)) :
  x = 20 := 
  sorry

end xiao_wang_original_plan_l150_150572


namespace height_at_end_of_2_years_l150_150818

-- Step d): Define the conditions and state the theorem

-- Define a function modeling the height of the tree each year
def tree_height (initial_height : ℕ) (years : ℕ) : ℕ :=
  initial_height * 3^years

-- Given conditions as definitions
def year_4_height := 81 -- height at the end of 4 years

-- Theorem that we need to prove
theorem height_at_end_of_2_years (initial_height : ℕ) (h : tree_height initial_height 4 = year_4_height) :
  tree_height initial_height 2 = 9 :=
sorry

end height_at_end_of_2_years_l150_150818


namespace length_of_shorter_piece_l150_150299

theorem length_of_shorter_piece (x : ℕ) (h1 : x + (x + 12) = 68) : x = 28 :=
by
  sorry

end length_of_shorter_piece_l150_150299


namespace largest_5_digit_integer_congruent_to_19_mod_26_l150_150025

theorem largest_5_digit_integer_congruent_to_19_mod_26 :
  ∃ n : ℕ, 10000 ≤ 26 * n + 19 ∧ 26 * n + 19 < 100000 ∧ (26 * n + 19 ≡ 19 [MOD 26]) ∧ 26 * n + 19 = 99989 :=
by
  sorry

end largest_5_digit_integer_congruent_to_19_mod_26_l150_150025


namespace intersection_A_B_l150_150528

def A : Set ℝ := { x | -2 < x ∧ x < 2 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

end intersection_A_B_l150_150528


namespace value_of_m_solve_system_relationship_x_y_l150_150600

-- Part 1: Prove the value of m is 1
theorem value_of_m (x : ℝ) (m : ℝ) (h1 : 2 - x = x + 4) (h2 : m * (1 - x) = x + 3) : m = 1 := sorry

-- Part 2: Solve the system of equations given m = 1
theorem solve_system (x y : ℝ) (h1 : 3 * x + 2 * 1 = - y) (h2 : 2 * x + 2 * y = 1 - 1) : x = -1 ∧ y = 1 := sorry

-- Part 3: Relationship between x and y regardless of m
theorem relationship_x_y (x y m : ℝ) (h1 : 3 * x + y = -2 * m) (h2 : 2 * x + 2 * y = m - 1) : 7 * x + 5 * y = -2 := sorry

end value_of_m_solve_system_relationship_x_y_l150_150600


namespace height_cylinder_l150_150490

variables (r_c h_c r_cy h_cy : ℝ)
variables (V_cone V_cylinder : ℝ)
variables (r_c_val : r_c = 15)
variables (h_c_val : h_c = 20)
variables (r_cy_val : r_cy = 30)
variables (V_cone_eq : V_cone = (1/3) * π * r_c^2 * h_c)
variables (V_cylinder_eq : V_cylinder = π * r_cy^2 * h_cy)

theorem height_cylinder : h_cy = 1.67 :=
by
  rw [r_c_val, h_c_val, r_cy_val] at *
  have V_cone := V_cone_eq
  have V_cylinder := V_cylinder_eq
  sorry

end height_cylinder_l150_150490


namespace relationship_abc_l150_150901

theorem relationship_abc (a b c : ℝ) 
  (h₁ : a = Real.log 0.5 / Real.log 2) 
  (h₂ : b = Real.sqrt 2) 
  (h₃ : c = 0.5 ^ 2) : 
  a < c ∧ c < b := by
  sorry

end relationship_abc_l150_150901


namespace max_blocks_in_box_l150_150743

def volume (l w h : ℕ) : ℕ := l * w * h

-- Define the dimensions of the box and the block
def box_length := 4
def box_width := 3
def box_height := 2
def block_length := 3
def block_width := 1
def block_height := 1

-- Define the volumes of the box and the block using the dimensions
def V_box : ℕ := volume box_length box_width box_height
def V_block : ℕ := volume block_length block_width block_height

theorem max_blocks_in_box : V_box / V_block = 8 :=
  sorry

end max_blocks_in_box_l150_150743


namespace quadratic_inequality_solution_l150_150083

theorem quadratic_inequality_solution
  (a b : ℝ)
  (h1 : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ (x < -2 ∨ -1/2 < x)) :
  ∀ x : ℝ, b * x^2 + a * x + 1 < 0 ↔ -2 < x ∧ x < -1/2 :=
by
  sorry

end quadratic_inequality_solution_l150_150083


namespace subway_speed_increase_l150_150737

theorem subway_speed_increase (s : ℝ) (h₀ : 0 ≤ s) (h₁ : s ≤ 7) : 
  (s^2 + 2 * s = 63) ↔ (s = 7) :=
by
  sorry 

end subway_speed_increase_l150_150737


namespace largest_avg_5_l150_150418

def arithmetic_avg (a l : ℕ) : ℚ :=
  (a + l) / 2

def multiples_avg_2 (n : ℕ) : ℚ :=
  arithmetic_avg 2 (n - (n % 2))

def multiples_avg_3 (n : ℕ) : ℚ :=
  arithmetic_avg 3 (n - (n % 3))

def multiples_avg_4 (n : ℕ) : ℚ :=
  arithmetic_avg 4 (n - (n % 4))

def multiples_avg_5 (n : ℕ) : ℚ :=
  arithmetic_avg 5 (n - (n % 5))

def multiples_avg_6 (n : ℕ) : ℚ :=
  arithmetic_avg 6 (n - (n % 6))

theorem largest_avg_5 (n : ℕ) (h : n = 101) : 
  multiples_avg_5 n > multiples_avg_2 n ∧ 
  multiples_avg_5 n > multiples_avg_3 n ∧ 
  multiples_avg_5 n > multiples_avg_4 n ∧ 
  multiples_avg_5 n > multiples_avg_6 n :=
by
  sorry

end largest_avg_5_l150_150418


namespace ellipse_with_given_foci_and_point_l150_150376

noncomputable def areFociEqual (a b c₁ c₂ : ℝ) : Prop :=
  c₁ = Real.sqrt (a^2 - b^2) ∧ c₂ = Real.sqrt (a^2 - b^2)

noncomputable def isPointOnEllipse (x₀ y₀ a₂ b₂ : ℝ) : Prop :=
  (x₀^2 / a₂) + (y₀^2 / b₂) = 1

theorem ellipse_with_given_foci_and_point :
  ∃a b : ℝ, 
    areFociEqual 8 3 a b ∧
    a = Real.sqrt 5 ∧ b = Real.sqrt 5 ∧
    isPointOnEllipse 3 (-2) 15 10  :=
sorry

end ellipse_with_given_foci_and_point_l150_150376


namespace line_through_circle_center_slope_one_eq_l150_150942

theorem line_through_circle_center_slope_one_eq (x y : ℝ) :
  (∃ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 ∧ y = 2) →
  (∃ m : ℝ, m = 1 ∧ (x + 1) = m * (y - 2)) →
  (x - y + 3 = 0) :=
sorry

end line_through_circle_center_slope_one_eq_l150_150942


namespace milk_production_days_l150_150029

variable (x : ℕ)
def cows := 2 * x
def cans := 2 * x + 2
def days := 2 * x + 1
def total_cows := 2 * x + 4
def required_cans := 2 * x + 10

theorem milk_production_days :
  (total_cows * required_cans) = ((2 * x) * (2 * x + 1) * required_cans) / ((2 * x + 2) * (2 * x + 4)) :=
sorry

end milk_production_days_l150_150029


namespace second_number_removed_l150_150402

theorem second_number_removed (S : ℝ) (X : ℝ) (h1 : S / 50 = 38) (h2 : (S - 45 - X) / 48 = 37.5) : X = 55 :=
by
  sorry

end second_number_removed_l150_150402


namespace solve_inequality_l150_150628

theorem solve_inequality (a : ℝ) : 
    (∀ x : ℝ, x^2 + (a + 2)*x + 2*a < 0 ↔ 
        (if a < 2 then -2 < x ∧ x < -a
         else if a = 2 then false
         else -a < x ∧ x < -2)) :=
by
  sorry

end solve_inequality_l150_150628


namespace arithmetic_sequence_twentieth_term_l150_150953

theorem arithmetic_sequence_twentieth_term
  (a1 : ℤ) (a13 : ℤ) (a20 : ℤ) (d : ℤ)
  (h1 : a1 = 3)
  (h2 : a13 = 27)
  (h3 : a13 = a1 + 12 * d)
  (h4 : a20 = a1 + 19 * d) : 
  a20 = 41 :=
by
  --  We assume a20 and prove it equals 41 instead of solving it in steps
  sorry

end arithmetic_sequence_twentieth_term_l150_150953


namespace minimum_study_tools_l150_150747

theorem minimum_study_tools (n : Nat) : n^3 ≥ 366 → n ≥ 8 := by
  intros h
  sorry

end minimum_study_tools_l150_150747


namespace find_angle_QPR_l150_150974

-- Define the angles and line segment
variables (R S Q T P : Type) 
variables (line_RT : R ≠ S)
variables (x : ℝ) 
variables (angle_PTQ : ℝ := 62)
variables (angle_RPS : ℝ := 34)

-- Hypothesis that PQ = PT, making triangle PQT isosceles
axiom eq_PQ_PT : ℝ

-- Conditions
axiom lie_on_RT : ∀ {R S Q T : Type}, R ≠ S 
axiom angle_PTQ_eq : angle_PTQ = 62
axiom angle_RPS_eq : angle_RPS = 34

-- Hypothesis that defines the problem structure
theorem find_angle_QPR : x = 11 := by
sorry

end find_angle_QPR_l150_150974


namespace parallel_line_through_point_l150_150630

theorem parallel_line_through_point :
  ∃ c : ℝ, ∀ x y : ℝ, (x = -1) → (y = 3) → (x - 2*y + 3 = 0) → (x - 2*y + c = 0) :=
sorry

end parallel_line_through_point_l150_150630


namespace angle_sum_impossible_l150_150208

theorem angle_sum_impossible (A1 A2 A3 : ℝ) (h : A1 + A2 + A3 = 180) :
  ¬ ((A1 > 90 ∧ A2 > 90 ∧ A3 < 90) ∨ (A1 > 90 ∧ A3 > 90 ∧ A2 < 90) ∨ (A2 > 90 ∧ A3 > 90 ∧ A1 < 90)) :=
sorry

end angle_sum_impossible_l150_150208


namespace H2O_required_for_NaH_reaction_l150_150304

theorem H2O_required_for_NaH_reaction
  (n_NaH : ℕ) (n_H2O : ℕ) (n_NaOH : ℕ) (n_H2 : ℕ)
  (h_eq : n_NaH = 2) (balanced_eq : n_NaH = n_H2O ∧ n_H2O = n_NaOH ∧ n_NaOH = n_H2) :
  n_H2O = 2 :=
by
  -- The proof is omitted as we only need to declare the statement.
  sorry

end H2O_required_for_NaH_reaction_l150_150304


namespace area_region_sum_l150_150136

theorem area_region_sum (r1 r2 : ℝ) (angle : ℝ) (a b c : ℕ) : 
  r1 = 6 → r2 = 3 → angle = 30 → (54 * Real.sqrt 3 + (9 : ℝ) * Real.pi - (9 : ℝ) * Real.pi = a * Real.sqrt b + c * Real.pi) → a + b + c = 10 :=
by
  intros
  -- We fill this with the actual proof steps later
  sorry

end area_region_sum_l150_150136


namespace particle_motion_inverse_relationship_l150_150272

theorem particle_motion_inverse_relationship 
  {k : ℝ} 
  (inverse_relationship : ∀ {n : ℕ}, ∃ t_n d_n, d_n = k / t_n)
  (second_mile : ∃ t_2 d_2, t_2 = 2 ∧ d_2 = 1) : 
  ∃ t_4 d_4, t_4 = 4 ∧ d_4 = 0.5 :=
by
  sorry

end particle_motion_inverse_relationship_l150_150272


namespace max_distance_from_earth_to_sun_l150_150244

-- Assume the semi-major axis 'a' and semi-minor axis 'b' specified in the problem.
def semi_major_axis : ℝ := 1.5 * 10^8
def semi_minor_axis : ℝ := 3 * 10^6

-- Define the theorem stating the maximum distance from the Earth to the Sun.
theorem max_distance_from_earth_to_sun :
  let a := semi_major_axis
  let b := semi_minor_axis
  a + b = 1.53 * 10^8 :=
by
  -- Proof will be completed
  sorry

end max_distance_from_earth_to_sun_l150_150244


namespace kenny_trumpet_hours_l150_150663

variables (x y : ℝ)
def basketball_hours := 10
def running_hours := 2 * basketball_hours
def trumpet_hours := 2 * running_hours

theorem kenny_trumpet_hours (x y : ℝ) (H : basketball_hours + running_hours + trumpet_hours = x + y) :
  trumpet_hours = 40 :=
by
  sorry

end kenny_trumpet_hours_l150_150663


namespace model_to_statue_ratio_l150_150330

theorem model_to_statue_ratio 
  (statue_height : ℝ) 
  (model_height_feet : ℝ)
  (model_height_inches : ℝ)
  (conversion_factor : ℝ) :
  statue_height = 45 → model_height_feet = 3 → conversion_factor = 12 → model_height_inches = model_height_feet * conversion_factor →
  (45 / model_height_inches) = 1.25 :=
by
  sorry

end model_to_statue_ratio_l150_150330


namespace domain_sqrt_l150_150003

noncomputable def domain_of_function := {x : ℝ | x ≥ 0 ∧ x - 1 ≥ 0}

theorem domain_sqrt : domain_of_function = {x : ℝ | 1 ≤ x} := by {
  sorry
}

end domain_sqrt_l150_150003


namespace find_ab_l150_150646

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 39) : a * b = 15 :=
by
  sorry

end find_ab_l150_150646


namespace simplify_expression_l150_150521

def expr_initial (y : ℝ) := 3*y + 4*y^2 + 2 - (7 - 3*y - 4*y^2)
def expr_simplified (y : ℝ) := 8*y^2 + 6*y - 5

theorem simplify_expression (y : ℝ) : expr_initial y = expr_simplified y :=
by
  sorry

end simplify_expression_l150_150521


namespace clusters_per_spoonful_l150_150585

theorem clusters_per_spoonful (spoonfuls_per_bowl : ℕ) (clusters_per_box : ℕ) (bowls_per_box : ℕ) 
  (h_spoonfuls : spoonfuls_per_bowl = 25) 
  (h_clusters : clusters_per_box = 500)
  (h_bowls : bowls_per_box = 5) : 
  clusters_per_box / bowls_per_box / spoonfuls_per_bowl = 4 := 
by 
  have clusters_per_bowl := clusters_per_box / bowls_per_box
  have clusters_per_spoonful := clusters_per_bowl / spoonfuls_per_bowl
  sorry

end clusters_per_spoonful_l150_150585


namespace fractions_equivalence_l150_150143

theorem fractions_equivalence (k : ℝ) (h : k ≠ -5) : (k + 3) / (k + 5) = 3 / 5 ↔ k = 0 := 
by 
  sorry

end fractions_equivalence_l150_150143


namespace side_length_square_eq_4_l150_150583

theorem side_length_square_eq_4 (s : ℝ) (h : s^2 - 3 * s = 4) : s = 4 :=
sorry

end side_length_square_eq_4_l150_150583


namespace perimeter_trapezoid_l150_150397

theorem perimeter_trapezoid 
(E F G H : Point)
(EF GH : ℝ)
(HJ EI FG EH : ℝ)
(h_eq1 : EF = GH)
(h_FG : FG = 10)
(h_EH : EH = 20)
(h_EI : EI = 5)
(h_HJ : HJ = 5)
(h_EF_HG : EF = Real.sqrt (EI^2 + ((EH - FG) / 2)^2)) :
  2 * EF + FG + EH = 30 + 10 * Real.sqrt 2 :=
by
  sorry

end perimeter_trapezoid_l150_150397


namespace mutter_paid_correct_amount_l150_150742

def total_lagaan_collected : ℝ := 344000
def mutter_land_percentage : ℝ := 0.0023255813953488372
def mutter_lagaan_paid : ℝ := 800

theorem mutter_paid_correct_amount : 
  mutter_lagaan_paid = total_lagaan_collected * mutter_land_percentage := by
  sorry

end mutter_paid_correct_amount_l150_150742


namespace ninja_star_ratio_l150_150935

-- Define variables for the conditions
variables (Eric_stars Chad_stars Jeff_stars Total_stars : ℕ) (Jeff_bought : ℕ)

/-- Given the following conditions:
1. Eric has 4 ninja throwing stars.
2. Jeff now has 6 throwing stars.
3. Jeff bought 2 ninja stars from Chad.
4. Altogether, they have 16 ninja throwing stars.

We want to prove that the ratio of the number of ninja throwing stars Chad has to the number Eric has is 2:1. --/
theorem ninja_star_ratio
  (h1 : Eric_stars = 4)
  (h2 : Jeff_stars = 6)
  (h3 : Jeff_bought = 2)
  (h4 : Total_stars = 16)
  (h5 : Eric_stars + Jeff_stars - Jeff_bought + Chad_stars = Total_stars) :
  Chad_stars / Eric_stars = 2 :=
by
  sorry

end ninja_star_ratio_l150_150935


namespace hyperbola_eccentricity_range_l150_150673

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  (∃ P₁ P₂ : { p : ℝ × ℝ // p ≠ (0, b) ∧ p ≠ (c, 0) ∧ ((0, b) - p).1 * ((c, 0) - p).1 + ((0, b) - p).2 * ((c, 0) - p).2 = 0},
   true) -- This encodes the existence of the required points P₁ and P₂ on line segment BF excluding endpoints
  → 1 < (Real.sqrt ((a^2 + b^2) / a^2)) ∧ (Real.sqrt ((a^2 + b^2) / a^2)) < (Real.sqrt 5 + 1)/2 :=
sorry

end hyperbola_eccentricity_range_l150_150673


namespace choose_bar_chart_for_comparisons_l150_150174

/-- 
To easily compare the quantities of various items, one should choose a bar chart 
based on the characteristics of statistical charts.
-/
theorem choose_bar_chart_for_comparisons 
  (chart_type: Type) 
  (is_bar_chart: chart_type → Prop)
  (is_ideal_chart_for_comparison: chart_type → Prop)
  (bar_chart_ideal: ∀ c, is_bar_chart c → is_ideal_chart_for_comparison c) 
  (comparison_chart : chart_type) 
  (h: is_bar_chart comparison_chart): 
  is_ideal_chart_for_comparison comparison_chart := 
by
  exact bar_chart_ideal comparison_chart h

end choose_bar_chart_for_comparisons_l150_150174


namespace maximum_k_l150_150365

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

theorem maximum_k (x : ℝ) (h₀ : x > 0) (k : ℤ) (a := 1) (h₁ : (x - k) * f_prime x a + x + 1 > 0) : k = 2 :=
sorry

end maximum_k_l150_150365


namespace houses_before_boom_l150_150446

theorem houses_before_boom (T B H : ℕ) (hT : T = 2000) (hB : B = 574) : H = 1426 := by
  sorry

end houses_before_boom_l150_150446


namespace simplify_expression_l150_150531

theorem simplify_expression : ((1 + 2 + 3 + 4 + 5 + 6) / 3 + (3 * 5 + 12) / 4) = 13.75 :=
by
-- Proof steps would go here, but we replace them with 'sorry' for now.
sorry

end simplify_expression_l150_150531


namespace toy_truck_cost_is_correct_l150_150594

-- Define the initial amount, amount spent on the pencil case, and the final amount
def initial_amount : ℝ := 10
def pencil_case_cost : ℝ := 2
def final_amount : ℝ := 5

-- Define the amount spent on the toy truck
def toy_truck_cost : ℝ := initial_amount - pencil_case_cost - final_amount

-- Prove that the amount spent on the toy truck is 3 dollars
theorem toy_truck_cost_is_correct : toy_truck_cost = 3 := by
  sorry

end toy_truck_cost_is_correct_l150_150594


namespace fishing_problem_l150_150219

theorem fishing_problem (a b c d : ℕ)
  (h1 : a + b + c + d = 11)
  (h2 : 1 ≤ a) 
  (h3 : 1 ≤ b) 
  (h4 : 1 ≤ c) 
  (h5 : 1 ≤ d) : 
  a < 3 ∨ b < 3 ∨ c < 3 ∨ d < 3 :=
by
  -- This is a placeholder for the proof
  sorry

end fishing_problem_l150_150219


namespace bowling_ball_weight_l150_150809

-- Definitions for the conditions
def kayak_weight : ℕ := 36
def total_weight_of_two_kayaks := 2 * kayak_weight
def total_weight_of_nine_bowling_balls (ball_weight : ℕ) := 9 * ball_weight  

theorem bowling_ball_weight (w : ℕ) (h1 : total_weight_of_two_kayaks = total_weight_of_nine_bowling_balls w) : w = 8 :=
by
  -- Proof goes here
  sorry

end bowling_ball_weight_l150_150809


namespace sylvia_carla_together_time_l150_150173

-- Define the conditions
def sylviaRate := 1 / 45
def carlaRate := 1 / 30

-- Define the combined work rate and the time taken to complete the job together
def combinedRate := sylviaRate + carlaRate
def timeTogether := 1 / combinedRate

-- Theorem stating the desired result
theorem sylvia_carla_together_time : timeTogether = 18 := by
  sorry

end sylvia_carla_together_time_l150_150173


namespace initial_pills_count_l150_150077

theorem initial_pills_count 
  (pills_taken_first_2_days : ℕ)
  (pills_taken_next_3_days : ℕ)
  (pills_taken_sixth_day : ℕ)
  (pills_left : ℕ)
  (h1 : pills_taken_first_2_days = 2 * 3 * 2)
  (h2 : pills_taken_next_3_days = 1 * 3 * 3)
  (h3 : pills_taken_sixth_day = 2)
  (h4 : pills_left = 27) :
  ∃ initial_pills : ℕ, initial_pills = pills_taken_first_2_days + pills_taken_next_3_days + pills_taken_sixth_day + pills_left :=
by
  sorry

end initial_pills_count_l150_150077


namespace mica_should_have_28_26_euros_l150_150797

namespace GroceryShopping

def pasta_cost : ℝ := 3 * 1.70
def ground_beef_cost : ℝ := 0.5 * 8.20
def pasta_sauce_base_cost : ℝ := 3 * 2.30
def pasta_sauce_discount : ℝ := pasta_sauce_base_cost * 0.10
def pasta_sauce_discounted_cost : ℝ := pasta_sauce_base_cost - pasta_sauce_discount
def quesadillas_cost : ℝ := 11.50

def total_cost_before_vat : ℝ :=
  pasta_cost + ground_beef_cost + pasta_sauce_discounted_cost + quesadillas_cost

def vat : ℝ := total_cost_before_vat * 0.05

def total_cost_including_vat : ℝ := total_cost_before_vat + vat

theorem mica_should_have_28_26_euros :
  total_cost_including_vat = 28.26 := by
  -- This is the statement without the proof. 
  sorry

end GroceryShopping

end mica_should_have_28_26_euros_l150_150797


namespace students_not_visiting_any_l150_150650

-- Define the given conditions as Lean definitions
def total_students := 52
def visited_botanical := 12
def visited_animal := 26
def visited_technology := 23
def visited_botanical_animal := 5
def visited_botanical_technology := 2
def visited_animal_technology := 4
def visited_all_three := 1

-- Translate the problem statement and proof goal
theorem students_not_visiting_any :
  total_students - (visited_botanical + visited_animal + visited_technology 
  - visited_botanical_animal - visited_botanical_technology 
  - visited_animal_technology + visited_all_three) = 1 :=
by
  -- The proof is omitted
  sorry

end students_not_visiting_any_l150_150650


namespace total_games_in_season_l150_150009

theorem total_games_in_season :
  let num_teams := 14
  let teams_per_division := 7
  let games_within_division_per_team := 6 * 3
  let games_against_other_division_per_team := 7
  let games_per_team := games_within_division_per_team + games_against_other_division_per_team
  let total_initial_games := games_per_team * num_teams
  let total_games := total_initial_games / 2
  total_games = 175 :=
by
  sorry

end total_games_in_season_l150_150009


namespace solution_range_for_m_l150_150719

theorem solution_range_for_m (x m : ℝ) (h₁ : 2 * x - 1 > 3 * (x - 2)) (h₂ : x < m) : m ≥ 5 :=
by {
  sorry
}

end solution_range_for_m_l150_150719


namespace other_solution_of_quadratic_l150_150670

theorem other_solution_of_quadratic (x : ℚ) (h₁ : 81 * 2/9 * 2/9 + 220 = 196 * 2/9 - 15) (h₂ : 81*x^2 - 196*x + 235 = 0) : x = 2/9 ∨ x = 5/9 :=
by
  sorry

end other_solution_of_quadratic_l150_150670


namespace eval_fraction_l150_150163

theorem eval_fraction : (3 : ℚ) / (2 - 5 / 4) = 4 := 
by 
  sorry

end eval_fraction_l150_150163


namespace fraction_equality_l150_150386

variables {R : Type*} [Field R] {m n p q : R}

theorem fraction_equality 
  (h1 : m / n = 15)
  (h2 : p / n = 3)
  (h3 : p / q = 1 / 10) :
  m / q = 1 / 2 :=
sorry

end fraction_equality_l150_150386


namespace find_common_ratio_l150_150772

def first_term : ℚ := 4 / 7
def second_term : ℚ := 12 / 7

theorem find_common_ratio (r : ℚ) : second_term = first_term * r → r = 3 :=
by
  sorry

end find_common_ratio_l150_150772


namespace intersection_on_circle_l150_150156

def parabola1 (X : ℝ) : ℝ := X^2 + X - 41
def parabola2 (Y : ℝ) : ℝ := Y^2 + Y - 40

theorem intersection_on_circle (X Y : ℝ) :
  parabola1 X = Y ∧ parabola2 Y = X → X^2 + Y^2 = 81 :=
by {
  sorry
}

end intersection_on_circle_l150_150156


namespace pencils_left_proof_l150_150170

noncomputable def total_pencils_left (a d : ℕ) : ℕ :=
  let total_initial_pencils : ℕ := 30
  let total_pencils_given_away : ℕ := 15 * a + 105 * d
  total_initial_pencils - total_pencils_given_away

theorem pencils_left_proof (a d : ℕ) :
  total_pencils_left a d = 30 - (15 * a + 105 * d) :=
by
  sorry

end pencils_left_proof_l150_150170


namespace avg_salary_rest_of_workers_l150_150515

theorem avg_salary_rest_of_workers (avg_all : ℝ) (avg_technicians : ℝ) (total_workers : ℕ) (n_technicians : ℕ) (avg_rest : ℝ) :
  avg_all = 8000 ∧ avg_technicians = 20000 ∧ total_workers = 49 ∧ n_technicians = 7 →
  avg_rest = 6000 :=
by
  sorry

end avg_salary_rest_of_workers_l150_150515


namespace three_pairs_exist_l150_150945

theorem three_pairs_exist :
  ∃! S P : ℕ, 5 * S + 7 * P = 90 :=
by
  sorry

end three_pairs_exist_l150_150945


namespace annual_decrease_rate_l150_150481

theorem annual_decrease_rate
  (P0 : ℕ := 8000)
  (P2 : ℕ := 6480) :
  ∃ r : ℝ, 8000 * (1 - r / 100)^2 = 6480 ∧ r = 10 :=
by
  use 10
  sorry

end annual_decrease_rate_l150_150481


namespace least_number_to_subtract_l150_150692

theorem least_number_to_subtract (n : ℕ) (k : ℕ) (r : ℕ) (h : n = 3674958423) (div : k = 47) (rem : r = 30) :
  (n % k = r) → 3674958423 % 47 = 30 :=
by
  sorry

end least_number_to_subtract_l150_150692


namespace games_within_division_l150_150379

variables (N M : ℕ)
  (h1 : N > 2 * M)
  (h2 : M > 4)
  (h3 : 3 * N + 4 * M = 76)

theorem games_within_division :
  3 * N = 48 :=
sorry

end games_within_division_l150_150379


namespace converse_inverse_contrapositive_count_l150_150621

theorem converse_inverse_contrapositive_count
  (a b : ℝ) : (a = 0 → ab = 0) →
  (if (ab = 0 → a = 0) then 1 else 0) +
  (if (a ≠ 0 → ab ≠ 0) then 1 else 0) +
  (if (ab ≠ 0 → a ≠ 0) then 1 else 0) = 1 :=
sorry

end converse_inverse_contrapositive_count_l150_150621


namespace fraction_identity_l150_150236

variable (x y z : ℝ)

theorem fraction_identity (h : (x / (y + z)) + (y / (z + x)) + (z / (x + y)) = 1) :
  (x^2 / (y + z)) + (y^2 / (z + x)) + (z^2 / (x + y)) = 0 :=
  sorry

end fraction_identity_l150_150236


namespace total_number_of_components_l150_150785

-- Definitions based on the conditions in the problem
def number_of_B_components := 300
def number_of_C_components := 200
def sample_size := 45
def number_of_A_components_drawn := 20
def number_of_C_components_drawn := 10

-- The statement to be proved
theorem total_number_of_components :
  (number_of_A_components_drawn * (number_of_B_components + number_of_C_components) / sample_size) 
  + number_of_B_components 
  + number_of_C_components 
  = 900 := 
by 
  sorry

end total_number_of_components_l150_150785


namespace quadratic_equation_roots_l150_150774

theorem quadratic_equation_roots (m n : ℝ) 
  (h_sum : m + n = -3) 
  (h_prod : m * n = 1) 
  (h_equation : m^2 + 3 * m + 1 = 0) :
  (3 * m + 1) / (m^3 * n) = -1 := 
by sorry

end quadratic_equation_roots_l150_150774


namespace defective_and_shipped_percent_l150_150524

def defective_percent : ℝ := 0.05
def shipped_percent : ℝ := 0.04

theorem defective_and_shipped_percent : (defective_percent * shipped_percent) * 100 = 0.2 :=
by
  sorry

end defective_and_shipped_percent_l150_150524


namespace andrena_has_more_dolls_than_debelyn_l150_150878

-- Definitions based on the given conditions
def initial_dolls_debelyn := 20
def initial_gift_debelyn_to_andrena := 2

def initial_dolls_christel := 24
def gift_christel_to_andrena := 5
def gift_christel_to_belissa := 3

def initial_dolls_belissa := 15
def gift_belissa_to_andrena := 4

-- Final number of dolls after exchanges
def final_dolls_debelyn := initial_dolls_debelyn - initial_gift_debelyn_to_andrena
def final_dolls_christel := initial_dolls_christel - gift_christel_to_andrena - gift_christel_to_belissa
def final_dolls_belissa := initial_dolls_belissa - gift_belissa_to_andrena + gift_christel_to_belissa
def final_dolls_andrena := initial_gift_debelyn_to_andrena + gift_christel_to_andrena + gift_belissa_to_andrena

-- Additional conditions
def andrena_more_than_christel := final_dolls_andrena = final_dolls_christel + 2
def belissa_equals_debelyn := final_dolls_belissa = final_dolls_debelyn

-- Proof Statement
theorem andrena_has_more_dolls_than_debelyn :
  andrena_more_than_christel →
  belissa_equals_debelyn →
  final_dolls_andrena - final_dolls_debelyn = 4 :=
by
  sorry

end andrena_has_more_dolls_than_debelyn_l150_150878


namespace max_abs_x2_is_2_l150_150551

noncomputable def max_abs_x2 {x₁ x₂ x₃ : ℝ} (h : x₁^2 + x₂^2 + x₃^2 + x₁ * x₂ + x₂ * x₃ = 2) : ℝ :=
2

theorem max_abs_x2_is_2 {x₁ x₂ x₃ : ℝ} (h : x₁^2 + x₂^2 + x₃^2 + x₁ * x₂ + x₂ * x₃ = 2) :
  max_abs_x2 h = 2 := 
sorry

end max_abs_x2_is_2_l150_150551


namespace find_added_amount_l150_150706

theorem find_added_amount (x y : ℕ) (h1 : x = 18) (h2 : 3 * (2 * x + y) = 123) : y = 5 :=
by
  sorry

end find_added_amount_l150_150706


namespace sqrt_9_eq_pm3_l150_150334

theorem sqrt_9_eq_pm3 : ∃ x : ℤ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by sorry

end sqrt_9_eq_pm3_l150_150334


namespace quadratic_inequality_solution_set_l150_150325

theorem quadratic_inequality_solution_set (m t : ℝ)
  (h : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - m*x + t < 0) : 
  m - t = -1 := sorry

end quadratic_inequality_solution_set_l150_150325


namespace total_pens_left_l150_150189

def initial_blue_pens := 9
def removed_blue_pens := 4
def initial_black_pens := 21
def removed_black_pens := 7
def initial_red_pens := 6

def remaining_blue_pens := initial_blue_pens - removed_blue_pens
def remaining_black_pens := initial_black_pens - removed_black_pens
def remaining_red_pens := initial_red_pens

def total_remaining_pens := remaining_blue_pens + remaining_black_pens + remaining_red_pens

theorem total_pens_left : total_remaining_pens = 25 :=
by
  -- Proof will be provided here
  sorry

end total_pens_left_l150_150189


namespace factor_t_squared_minus_81_l150_150313

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) :=
by
  sorry

end factor_t_squared_minus_81_l150_150313


namespace seats_capacity_l150_150516

theorem seats_capacity (x : ℕ) (h1 : 15 * x + 12 * x + 8 = 89) : x = 3 :=
by
  -- proof to be filled in
  sorry

end seats_capacity_l150_150516


namespace lamp_post_ratio_l150_150854

theorem lamp_post_ratio (x k m : ℕ) (h1 : 9 * x = k) (h2 : 99 * x = m) : m = 11 * k :=
by sorry

end lamp_post_ratio_l150_150854


namespace cos_alpha_value_l150_150679

-- Define our conditions
variables (α : ℝ)
axiom sin_alpha : Real.sin α = -5 / 13
axiom tan_alpha_pos : Real.tan α > 0

-- State our goal
theorem cos_alpha_value : Real.cos α = -12 / 13 :=
by
  sorry

end cos_alpha_value_l150_150679


namespace tan_45_eq_one_l150_150436

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l150_150436


namespace f_0_plus_f_1_l150_150760

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom f_neg1 : f (-1) = 2

theorem f_0_plus_f_1 : f 0 + f 1 = -2 :=
by
  sorry

end f_0_plus_f_1_l150_150760


namespace speed_of_stream_l150_150271

def boatSpeedDownstream (V_b V_s : ℝ) : ℝ :=
  V_b + V_s

def boatSpeedUpstream (V_b V_s : ℝ) : ℝ :=
  V_b - V_s

theorem speed_of_stream (V_b V_s : ℝ) (h1 : V_b + V_s = 25) (h2 : V_b - V_s = 5) : V_s = 10 :=
by {
  sorry
}

end speed_of_stream_l150_150271


namespace find_a_l150_150851

theorem find_a (a : ℝ) :
  (∀ x : ℝ, deriv (fun x => a * x^3 - 2) x * x = 1) → a = 1 / 3 :=
by
  intro h
  have slope_at_minus_1 := h (-1)
  sorry -- here we stop as proof isn't needed

end find_a_l150_150851


namespace min_equal_area_triangles_l150_150454

theorem min_equal_area_triangles (chessboard_area missing_area : ℕ) (total_area : ℕ := chessboard_area - missing_area) 
(H1 : chessboard_area = 64) (H2 : missing_area = 1) : 
∃ n : ℕ, n = 18 ∧ (total_area = 63) → total_area / ((7:ℕ)/2) = n := 
sorry

end min_equal_area_triangles_l150_150454


namespace problem_l150_150391

-- Definitions for conditions
def countMultiplesOf (n upperLimit : ℕ) : ℕ :=
  (upperLimit - 1) / n

def a : ℕ := countMultiplesOf 4 40
def b : ℕ := countMultiplesOf 4 40

-- Statement to prove
theorem problem : (a + b)^2 = 324 := by
  sorry

end problem_l150_150391


namespace evaluate_cubic_diff_l150_150354

theorem evaluate_cubic_diff (x y : ℝ) (h1 : x + y = 12) (h2 : 2 * x + y = 16) : x^3 - y^3 = -448 := 
by
    sorry

end evaluate_cubic_diff_l150_150354


namespace reduced_fraction_numerator_l150_150643

theorem reduced_fraction_numerator :
  let numerator := 4128 
  let denominator := 4386 
  let gcd := Nat.gcd numerator denominator
  let reduced_numerator := numerator / gcd 
  let reduced_denominator := denominator / gcd 
  (reduced_numerator : ℚ) / (reduced_denominator : ℚ) = 16 / 17 → reduced_numerator = 16 :=
by
  intros
  sorry

end reduced_fraction_numerator_l150_150643


namespace conic_section_eccentricity_l150_150403

noncomputable def eccentricity (m : ℝ) : ℝ :=
if m = 2 then 1 / Real.sqrt 2 else
if m = -2 then Real.sqrt 3 else
0

theorem conic_section_eccentricity (m : ℝ) (h : 4 * 1 = m * m) :
  eccentricity m = 1 / Real.sqrt 2 ∨ eccentricity m = Real.sqrt 3 :=
by
  sorry

end conic_section_eccentricity_l150_150403


namespace sin_A_eq_one_half_l150_150355

theorem sin_A_eq_one_half (a b : ℝ) (sin_B : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sin_B = 2/3) : 
  ∃ (sin_A : ℝ), sin_A = 1/2 := 
by
  let sin_A := a * sin_B / b
  existsi sin_A
  sorry

end sin_A_eq_one_half_l150_150355


namespace Beth_crayons_proof_l150_150688

def Beth_packs_of_crayons (packs_crayons : ℕ) (total_crayons extra_crayons : ℕ) : ℕ :=
  total_crayons - extra_crayons

theorem Beth_crayons_proof
  (packs_crayons : ℕ)
  (each_pack_contains total_crayons extra_crayons : ℕ)
  (h_each_pack : each_pack_contains = 10) 
  (h_extra : extra_crayons = 6)
  (h_total : total_crayons = 40) 
  (valid_packs : packs_crayons = (Beth_packs_of_crayons total_crayons extra_crayons / each_pack_contains)) :
  packs_crayons = 3 :=
by
  rw [h_each_pack, h_extra, h_total] at valid_packs
  sorry

end Beth_crayons_proof_l150_150688


namespace no_real_solution_equation_l150_150675

theorem no_real_solution_equation (x : ℝ) (h : x ≠ -9) : 
  ¬ ∃ x, (8*x^2 + 90*x + 2) / (3*x + 27) = 4*x + 2 :=
by
  sorry

end no_real_solution_equation_l150_150675


namespace last_three_digits_of_2_pow_9000_l150_150644

-- The proof statement
theorem last_three_digits_of_2_pow_9000 (h : 2 ^ 300 ≡ 1 [MOD 1000]) : 2 ^ 9000 ≡ 1 [MOD 1000] :=
by
  sorry

end last_three_digits_of_2_pow_9000_l150_150644


namespace possible_values_of_5x_plus_2_l150_150250

theorem possible_values_of_5x_plus_2 (x : ℝ) :
  (x - 4) * (5 * x + 2) = 0 →
  (5 * x + 2 = 0 ∨ 5 * x + 2 = 22) :=
by
  intro h
  sorry

end possible_values_of_5x_plus_2_l150_150250


namespace find_salary_l150_150877

def salary_remaining (S : ℝ) (food : ℝ) (house_rent : ℝ) (clothes : ℝ) (remaining : ℝ) : Prop :=
  S - food * S - house_rent * S - clothes * S = remaining

theorem find_salary :
  ∀ S : ℝ, 
  salary_remaining S (1/5) (1/10) (3/5) 15000 → 
  S = 150000 :=
by
  intros S h
  sorry

end find_salary_l150_150877


namespace find_a_l150_150750

variables (a b c : ℝ) (A B C : ℝ) (sin : ℝ → ℝ)
variables (sqrt_three_two sqrt_two_two : ℝ)

-- Assume that A = 60 degrees, B = 45 degrees, and b = sqrt(6)
def angle_A : A = π / 3 := by
  sorry

def angle_B : B = π / 4 := by
  sorry

def side_b : b = Real.sqrt 6 := by
  sorry

def sin_60 : sin (π / 3) = sqrt_three_two := by
  sorry

def sin_45 : sin (π / 4) = sqrt_two_two := by
  sorry

-- Prove that a = 3 based on the given conditions
theorem find_a (sin_rule : a / sin A = b / sin B)
  (sin_60_def : sqrt_three_two = Real.sqrt 3 / 2)
  (sin_45_def : sqrt_two_two = Real.sqrt 2 / 2) : a = 3 := by
  sorry

end find_a_l150_150750


namespace smallest_positive_y_l150_150653

theorem smallest_positive_y (y : ℕ) (h : 42 * y + 8 ≡ 4 [MOD 24]) : y = 2 :=
sorry

end smallest_positive_y_l150_150653


namespace total_revenue_correct_l150_150612

def original_price_sneakers : ℝ := 80
def discount_sneakers : ℝ := 0.25
def pairs_sneakers_sold : ℕ := 2

def original_price_sandals : ℝ := 60
def discount_sandals : ℝ := 0.35
def pairs_sandals_sold : ℕ := 4

def original_price_boots : ℝ := 120
def discount_boots : ℝ := 0.40
def pairs_boots_sold : ℕ := 11

def calculate_total_revenue : ℝ := 
  let revenue_sneakers := pairs_sneakers_sold * (original_price_sneakers * (1 - discount_sneakers))
  let revenue_sandals := pairs_sandals_sold * (original_price_sandals * (1 - discount_sandals))
  let revenue_boots := pairs_boots_sold * (original_price_boots * (1 - discount_boots))
  revenue_sneakers + revenue_sandals + revenue_boots

theorem total_revenue_correct : calculate_total_revenue = 1068 := by
  sorry

end total_revenue_correct_l150_150612


namespace max_gcd_b_eq_1_l150_150261

-- Define bn as bn = 2^n - 1 for natural number n
def b (n : ℕ) : ℕ := 2^n - 1

-- Define en as the greatest common divisor of bn and bn+1
def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

-- The theorem to prove:
theorem max_gcd_b_eq_1 (n : ℕ) : e n = 1 :=
  sorry

end max_gcd_b_eq_1_l150_150261


namespace range_of_a_l150_150337

open Set

variable {x a : ℝ}

def p (x a : ℝ) := x^2 + 2 * a * x - 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) := x^2 + 2 * x - 8 < 0

theorem range_of_a (h : ∀ x, p x a → q x): 0 < a ∧ a ≤ 4 / 3 := 
  sorry

end range_of_a_l150_150337


namespace find_natural_numbers_l150_150761

theorem find_natural_numbers (n k : ℕ) (h : 2^n - 5^k = 7) : n = 5 ∧ k = 2 :=
by
  sorry

end find_natural_numbers_l150_150761


namespace initial_value_l150_150993

theorem initial_value (x k : ℤ) (h : x + 335 = k * 456) : x = 121 := sorry

end initial_value_l150_150993


namespace symmetric_line_equation_l150_150931

-- Definitions of the given conditions.
def original_line_equation (x y : ℝ) : Prop := 2 * x + 3 * y + 6 = 0
def line_of_symmetry (x y : ℝ) : Prop := y = x

-- The theorem statement to prove:
theorem symmetric_line_equation (x y : ℝ) : original_line_equation y x ↔ (3 * x + 2 * y + 6 = 0) :=
sorry

end symmetric_line_equation_l150_150931


namespace spadesuit_problem_l150_150442

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := abs (a - b)

-- Theorem statement
theorem spadesuit_problem : spadesuit (spadesuit 2 3) (spadesuit 6 (spadesuit 9 4)) = 0 := 
sorry

end spadesuit_problem_l150_150442


namespace range_of_p_l150_150414

theorem range_of_p (a b : ℝ) :
  (∀ x y p q : ℝ, p + q = 1 → (p * (x^2 + a * x + b) + q * (y^2 + a * y + b) ≥ ((p * x + q * y)^2 + a * (p * x + q * y) + b))) →
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 1) :=
sorry

end range_of_p_l150_150414


namespace contrapositive_eq_inverse_l150_150629

variable (p q : Prop)

theorem contrapositive_eq_inverse (h1 : p → q) :
  (¬ p → ¬ q) ↔ (q → p) := by
  sorry

end contrapositive_eq_inverse_l150_150629


namespace sqrt3_op_sqrt3_l150_150999

def custom_op (x y : ℝ) : ℝ :=
  (x + y)^2 - (x - y)^2

theorem sqrt3_op_sqrt3 : custom_op (Real.sqrt 3) (Real.sqrt 3) = 12 :=
  sorry

end sqrt3_op_sqrt3_l150_150999


namespace probability_of_matching_colors_l150_150267

theorem probability_of_matching_colors :
  let abe_jelly_beans := ["green", "red", "blue"]
  let bob_jelly_beans := ["green", "green", "yellow", "yellow", "red", "red", "red"]
  let abe_probs := (1 / 3, 1 / 3, 1 / 3)
  let bob_probs := (2 / 7, 3 / 7, 0)
  let matching_prob := (1 / 3 * 2 / 7) + (1 / 3 * 3 / 7)
  matching_prob = 5 / 21 := by sorry

end probability_of_matching_colors_l150_150267


namespace Jordan_Lee_debt_equal_l150_150278

theorem Jordan_Lee_debt_equal (initial_debt_jordan : ℝ) (additional_debt_jordan : ℝ)
  (rate_jordan : ℝ) (initial_debt_lee : ℝ) (rate_lee : ℝ) :
  initial_debt_jordan + additional_debt_jordan + (initial_debt_jordan + additional_debt_jordan) * rate_jordan * 33.333333333333336 
  = initial_debt_lee + initial_debt_lee * rate_lee * 33.333333333333336 :=
by
  let t := 33.333333333333336
  have rate_jordan := 0.12
  have rate_lee := 0.08
  have initial_debt_jordan := 200
  have additional_debt_jordan := 20
  have initial_debt_lee := 300
  sorry

end Jordan_Lee_debt_equal_l150_150278


namespace amanda_speed_l150_150006

-- Defining the conditions
def distance : ℝ := 6 -- 6 miles
def time : ℝ := 3 -- 3 hours

-- Stating the question with the conditions and the correct answer
theorem amanda_speed : (distance / time) = 2 :=
by 
  -- the proof is skipped as instructed
  sorry

end amanda_speed_l150_150006


namespace contractor_initial_people_l150_150581

theorem contractor_initial_people (P : ℕ) (days_total days_done : ℕ) 
  (percent_done : ℚ) (additional_people : ℕ) (T : ℕ) :
  days_total = 50 →
  days_done = 25 →
  percent_done = 0.4 →
  additional_people = 90 →
  T = P + additional_people →
  (P : ℚ) * 62.5 = (T : ℚ) * 50 →
  P = 360 :=
by
  intros h_days_total h_days_done h_percent_done h_additional_people h_T h_eq
  sorry

end contractor_initial_people_l150_150581


namespace opposite_of_2023_is_neg_2023_l150_150880

theorem opposite_of_2023_is_neg_2023 : (2023 + (-2023) = 0) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l150_150880


namespace find_f_l150_150059

theorem find_f (d e f : ℝ) (h_g : 16 = g) 
  (h_mean_of_zeros : -d / 12 = 3 + d + e + f + 16) 
  (h_product_of_zeros_two_at_a_time : -d / 12 = e / 3) : 
  f = -39 :=
by
  sorry

end find_f_l150_150059


namespace sin_cos_eq_values_l150_150269

theorem sin_cos_eq_values (θ : ℝ) (hθ : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  (∃ t : ℝ, 
    0 < t ∧ 
    t ≤ 2 * Real.pi ∧ 
    (2 + 4 * Real.sin t - 3 * Real.cos (2 * t) = 0)) ↔ (∃ n : ℕ, n = 4) :=
by 
  sorry

end sin_cos_eq_values_l150_150269


namespace tim_pencils_l150_150031

-- Problem statement: If x = 2 and z = 5, then y = z - x where y is the number of pencils Tim placed.
def pencils_problem (x y z : Nat) : Prop :=
  x = 2 ∧ z = 5 → y = z - x

theorem tim_pencils : pencils_problem 2 3 5 :=
by
  sorry

end tim_pencils_l150_150031


namespace percentage_saved_is_10_l150_150124

-- Given conditions
def rent_expenses : ℕ := 5000
def milk_expenses : ℕ := 1500
def groceries_expenses : ℕ := 4500
def education_expenses : ℕ := 2500
def petrol_expenses : ℕ := 2000
def misc_expenses : ℕ := 3940
def savings : ℕ := 2160

-- Define the total expenses
def total_expenses : ℕ := rent_expenses + milk_expenses + groceries_expenses + education_expenses + petrol_expenses + misc_expenses

-- Define the total monthly salary
def total_monthly_salary : ℕ := total_expenses + savings

-- Define the percentage of savings
def percentage_saved : ℕ := (savings * 100) / total_monthly_salary

-- Prove that the percentage saved is 10%
theorem percentage_saved_is_10 :
  percentage_saved = 10 :=
sorry

end percentage_saved_is_10_l150_150124


namespace symmetric_line_equation_l150_150347

def line_1 (x y : ℝ) : Prop := 2 * x - y + 3 = 0
def line_2 (x y : ℝ) : Prop := x - y + 2 = 0
def symmetric_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

theorem symmetric_line_equation :
  ∀ x y : ℝ, line_1 x y → line_2 x y → symmetric_line x y := 
sorry

end symmetric_line_equation_l150_150347


namespace sum_possible_n_k_l150_150216

theorem sum_possible_n_k (n k : ℕ) (h1 : 3 * k + 3 = n) (h2 : 5 * (k + 2) = 3 * (n - k - 1)) : 
  n + k = 19 := 
by {
  sorry -- proof steps would go here
}

end sum_possible_n_k_l150_150216


namespace number_of_integers_with_three_divisors_l150_150564

def has_exactly_three_positive_divisors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p * p

theorem number_of_integers_with_three_divisors (n : ℕ) :
  n = 2012 → Nat.card { x : ℕ | x ≤ n ∧ has_exactly_three_positive_divisors x } = 14 :=
by
  sorry

end number_of_integers_with_three_divisors_l150_150564


namespace problem_statement_l150_150247

theorem problem_statement (m : ℝ) (h : m^2 + m - 1 = 0) : m^4 + 2*m^3 - m + 2007 = 2007 := 
by 
  sorry

end problem_statement_l150_150247


namespace determinant_calculation_l150_150722

variable {R : Type*} [CommRing R]

def matrix_example (a b c : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![1, a, b], ![1, a + b, b + c], ![1, a, a + c]]

theorem determinant_calculation (a b c : R) :
  (matrix_example a b c).det = ab + b^2 + bc :=
by sorry

end determinant_calculation_l150_150722


namespace complex_modulus_l150_150319

noncomputable def z : ℂ := (1 + 3 * Complex.I) / (1 + Complex.I)

theorem complex_modulus 
  (h : (1 + Complex.I) * z = 1 + 3 * Complex.I) : 
  Complex.abs (z^2) = 5 := 
by
  sorry

end complex_modulus_l150_150319


namespace a_3_value_l150_150434

def arithmetic_seq (a: ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n - 3

theorem a_3_value :
  ∃ a : ℕ → ℤ, a 1 = 19 ∧ arithmetic_seq a ∧ a 3 = 13 :=
by
  sorry

end a_3_value_l150_150434


namespace bikers_meet_again_in_36_minutes_l150_150056

theorem bikers_meet_again_in_36_minutes :
    Nat.lcm 12 18 = 36 :=
sorry

end bikers_meet_again_in_36_minutes_l150_150056


namespace ways_A_to_C_via_B_l150_150889

def ways_A_to_B : Nat := 2
def ways_B_to_C : Nat := 3

theorem ways_A_to_C_via_B : ways_A_to_B * ways_B_to_C = 6 := by
  sorry

end ways_A_to_C_via_B_l150_150889


namespace range_of_a_l150_150474

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (0 : ℝ) a, f x ≤ 3) ∧ (∃ x ∈ Set.Icc (0 : ℝ) a, f x = 3) ∧ (∀ x ∈ Set.Icc (0 : ℝ) a, f x ≥ 2) ∧ (∃ x ∈ Set.Icc (0 : ℝ) a, f x = 2) ↔ 1 ≤ a ∧ a ≤ 2 := 
by 
  sorry

end range_of_a_l150_150474


namespace scientific_notation_of_number_l150_150240

theorem scientific_notation_of_number :
  ∃ (a : ℝ) (n : ℤ), 0.00000002 = a * 10^n ∧ a = 2 ∧ n = -8 :=
by
  sorry

end scientific_notation_of_number_l150_150240


namespace angle_measure_triple_complement_l150_150987

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l150_150987


namespace geometric_sequence_value_a3_l150_150944

-- Define the geometric sequence
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

-- Conditions given in the problem
variable (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 2)
variable (h₂ : (geometric_sequence a₁ q 4) * (geometric_sequence a₁ q 6) = 4 * (geometric_sequence a₁ q 7) ^ 2)

-- The goal is to prove that a₃ = 1
theorem geometric_sequence_value_a3 : geometric_sequence a₁ q 3 = 1 :=
by
  sorry

end geometric_sequence_value_a3_l150_150944


namespace average_last_four_numbers_l150_150392

theorem average_last_four_numbers (numbers : List ℝ) 
  (h1 : numbers.length = 7)
  (h2 : (numbers.sum / 7) = 62)
  (h3 : (numbers.take 3).sum / 3 = 58) : 
  ((numbers.drop 3).sum / 4) = 65 :=
by
  sorry

end average_last_four_numbers_l150_150392


namespace coeff_x3_in_product_l150_150678

theorem coeff_x3_in_product :
  let p1 := 3 * (Polynomial.X ^ 3) + 4 * (Polynomial.X ^ 2) + 5 * Polynomial.X + 6
  let p2 := 7 * (Polynomial.X ^ 2) + 8 * Polynomial.X + 9
  (Polynomial.coeff (p1 * p2) 3) = 94 :=
by
  sorry

end coeff_x3_in_product_l150_150678


namespace area_of_rectangle_is_32_proof_l150_150518

noncomputable def triangle_sides : ℝ := 7.3 + 5.4 + 11.3
def equality_of_perimeters (rectangle_length rectangle_width : ℝ) : Prop := 
  2 * (rectangle_length + rectangle_width) = triangle_sides

def rectangle_length (rectangle_width : ℝ) : ℝ := 2 * rectangle_width

def area_of_rectangle_is_32 (rectangle_width : ℝ) : Prop :=
  rectangle_length rectangle_width * rectangle_width = 32

theorem area_of_rectangle_is_32_proof : 
  ∃ (rectangle_width : ℝ), 
  equality_of_perimeters (rectangle_length rectangle_width) rectangle_width ∧ area_of_rectangle_is_32 rectangle_width :=
by
  sorry

end area_of_rectangle_is_32_proof_l150_150518


namespace eval_expression_correct_l150_150835

noncomputable def evaluate_expression : ℝ :=
    3 + Real.sqrt 3 + (3 - Real.sqrt 3) / 6 + (1 / (Real.cos (Real.pi / 4) - 3))

theorem eval_expression_correct : 
  evaluate_expression = (3 * Real.sqrt 3 - 5 * Real.sqrt 2) / 34 :=
by
  -- Proof can be filled in later
  sorry

end eval_expression_correct_l150_150835


namespace find_b_for_real_root_l150_150571

noncomputable def polynomial_has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^4 + b * x^3 - 2 * x^2 + b * x + 2 = 0

theorem find_b_for_real_root :
  ∀ b : ℝ, polynomial_has_real_root b → b ≤ 0 := by
  sorry

end find_b_for_real_root_l150_150571


namespace fraction_value_condition_l150_150015

theorem fraction_value_condition (m n : ℚ) (h : m / n = 2 / 3) : m / (m + n) = 2 / 5 :=
sorry

end fraction_value_condition_l150_150015


namespace bookmark_position_second_book_l150_150778

-- Definitions for the conditions
def pages_per_book := 250
def cover_thickness_ratio := 10
def total_books := 2
def distance_bookmarks_factor := 1 / 3

-- Derived constants
def cover_thickness := cover_thickness_ratio * pages_per_book
def total_pages := (pages_per_book * total_books) + (cover_thickness * total_books * 2)
def distance_between_bookmarks := total_pages * distance_bookmarks_factor
def midpoint_pages_within_book := (pages_per_book / 2) + cover_thickness

-- Definitions for bookmarks positions
def first_bookmark_position := midpoint_pages_within_book
def remaining_pages_after_first_bookmark := distance_between_bookmarks - midpoint_pages_within_book
def second_bookmark_position := remaining_pages_after_first_bookmark - cover_thickness

-- Theorem stating the goal
theorem bookmark_position_second_book :
  35 ≤ second_bookmark_position ∧ second_bookmark_position < 36 :=
sorry

end bookmark_position_second_book_l150_150778


namespace find_n_l150_150453

theorem find_n (n : ℕ) : (10^n = (10^5)^3) → n = 15 :=
by sorry

end find_n_l150_150453


namespace lunch_cost_before_tax_and_tip_l150_150301

theorem lunch_cost_before_tax_and_tip (C : ℝ) (h1 : 1.10 * C = 110) : C = 100 := by
  sorry

end lunch_cost_before_tax_and_tip_l150_150301


namespace arithmetic_sequence_a6_l150_150438

-- Definitions representing the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a_n (n + m) = a_n n + a_n m + n

def sum_of_first_n_terms (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = (n / 2) * (2 * a_n 1 + (n - 1) * (a_n 2 - a_n 1))

theorem arithmetic_sequence_a6 (S : ℕ → ℝ) (a_n : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_of_first_n_terms S a_n)
  (h_cond : S 9 - S 2 = 35) : 
  a_n 6 = 5 :=
by
  sorry

end arithmetic_sequence_a6_l150_150438


namespace book_pages_l150_150489

-- Define the number of pages Sally reads on weekdays and weekends
def pages_on_weekdays : ℕ := 10
def pages_on_weekends : ℕ := 20

-- Define the number of weekdays and weekends in 2 weeks
def weekdays_in_two_weeks : ℕ := 5 * 2
def weekends_in_two_weeks : ℕ := 2 * 2

-- Total number of pages read in 2 weeks
def total_pages_read (pages_on_weekdays : ℕ) (pages_on_weekends : ℕ) (weekdays_in_two_weeks : ℕ) (weekends_in_two_weeks : ℕ) : ℕ :=
  (pages_on_weekdays * weekdays_in_two_weeks) + (pages_on_weekends * weekends_in_two_weeks)

-- Prove the number of pages in the book
theorem book_pages : total_pages_read 10 20 10 4 = 180 := by
  sorry

end book_pages_l150_150489


namespace prob_first_given_defective_correct_l150_150192

-- Definitions from problem conditions
def first_box : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def second_box : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
def defective_first_box : Set ℕ := {1, 2, 3}
def defective_second_box : Set ℕ := {1, 2}

-- Probability values as defined
def prob_first_box : ℚ := 1 / 2
def prob_second_box : ℚ := 1 / 2
def prob_defective_given_first : ℚ := 3 / 10
def prob_defective_given_second : ℚ := 1 / 10

-- Calculation of total probability of defective component
def prob_defective : ℚ := (prob_first_box * prob_defective_given_first) + (prob_second_box * prob_defective_given_second)

-- Bayes' Theorem application to find the required probability
def prob_first_given_defective : ℚ := (prob_first_box * prob_defective_given_first) / prob_defective

-- Lean statement to verify the computed probability is as expected
theorem prob_first_given_defective_correct : prob_first_given_defective = 3 / 4 :=
by
  unfold prob_first_given_defective prob_defective
  sorry

end prob_first_given_defective_correct_l150_150192


namespace proof_problem_l150_150949

variables {m n : ℝ}

theorem proof_problem (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 2 * m * n) :
  (mn : ℝ) ≥ 1 ∧ (m^2 + n^2 ≥ 2) :=
  sorry

end proof_problem_l150_150949


namespace monotonic_decreasing_interval_of_f_l150_150061

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_decreasing_interval_of_f :
  { x : ℝ | x > Real.exp 1 } = {y : ℝ | ∀ ε > 0, (x : ℝ) → (0 < x → (f (x + ε) < f x) ∧ (f x < f (x + ε)))}
:=
sorry

end monotonic_decreasing_interval_of_f_l150_150061


namespace xy_value_l150_150072

theorem xy_value :
  ∃ a b c x y : ℝ,
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧
    3 * a + 2 * b + c = 5 ∧
    2 * a + b - 3 * c = 1 ∧
    (∀ m, m = 3 * a + b - 7 * c → (m = x ∨ m = y)) ∧
    x = -5 / 7 ∧
    y = -1 / 11 ∧
    x * y = 5 / 77 :=
sorry

end xy_value_l150_150072


namespace max_c_for_log_inequality_l150_150898

theorem max_c_for_log_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) : 
  ∃ c : ℝ, c = 1 / 3 ∧ (1 / (3 + Real.log b / Real.log a) + 1 / (3 + Real.log a / Real.log b) ≥ c) :=
by
  use 1 / 3
  sorry

end max_c_for_log_inequality_l150_150898


namespace problem1_problem2_l150_150201

theorem problem1 (x : ℕ) : 
  2 / 8^x * 16^x = 2^5 → x = 4 := 
by
  sorry

theorem problem2 (x : ℕ) : 
  2^(x+2) + 2^(x+1) = 24 → x = 2 := 
by
  sorry

end problem1_problem2_l150_150201


namespace frac_diff_zero_l150_150093

theorem frac_diff_zero (a b : ℝ) (h : a + b = a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a) - (1 / b) = 0 := 
sorry

end frac_diff_zero_l150_150093


namespace sandbox_side_length_l150_150758

theorem sandbox_side_length (side_length : ℝ) (sand_sq_inches_per_pound : ℝ := 80 / 30) (total_sand_pounds : ℝ := 600) :
  (side_length ^ 2 = total_sand_pounds * sand_sq_inches_per_pound) → side_length = 40 := 
by
  sorry

end sandbox_side_length_l150_150758


namespace train_crossing_time_is_correct_l150_150479

noncomputable def train_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

theorem train_crossing_time_is_correct :
  train_crossing_time 250 180 120 = 12.9 :=
by
  sorry

end train_crossing_time_is_correct_l150_150479


namespace equation_solution_l150_150892

noncomputable def solve_equation : Set ℝ := {x : ℝ | (3 * x + 2) / (x ^ 2 + 5 * x + 6) = 3 * x / (x - 1)
                                             ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 1}

theorem equation_solution (r : ℝ) (h : r ∈ solve_equation) : 3 * r ^ 3 + 12 * r ^ 2 + 19 * r + 2 = 0 :=
sorry

end equation_solution_l150_150892


namespace problem1_problem2_l150_150856

-- Problem (1)
variables {p q : ℝ}

theorem problem1 (hpq : p^3 + q^3 = 2) : p + q ≤ 2 := sorry

-- Problem (2)
variables {a b : ℝ}

theorem problem2 (hab : |a| + |b| < 1) : ∀ x : ℝ, (x^2 + a * x + b = 0) → |x| < 1 := sorry

end problem1_problem2_l150_150856


namespace rectangular_prism_volume_l150_150693

theorem rectangular_prism_volume
  (a b c : ℕ) 
  (h1 : 4 * ((a - 2) + (b - 2) + (c - 2)) = 40)
  (h2 : 2 * ((a - 2) * (b - 2) + (a - 2) * (c - 2) + (b - 2) * (c - 2)) = 66) :
  a * b * c = 150 :=
by sorry

end rectangular_prism_volume_l150_150693


namespace quadratic_equation_m_l150_150652

theorem quadratic_equation_m (m b : ℝ) (h : (m - 2) * x ^ |m| - b * x - 1 = 0) : m = -2 :=
by
  sorry

end quadratic_equation_m_l150_150652


namespace cos_A_minus_cos_C_l150_150836

-- Definitions representing the conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (h₁ : 4 * b * Real.sin A = Real.sqrt 7 * a)
variables (h₂ : 2 * b = a + c) (h₃ : A < B) (h₄ : B < C)

-- Statement of the proof problem
theorem cos_A_minus_cos_C (A B C a b c : ℝ)
  (h₁ : 4 * b * Real.sin A = Real.sqrt 7 * a)
  (h₂ : 2 * b = a + c)
  (h₃ : A < B)
  (h₄ : B < C) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 :=
by
  sorry

end cos_A_minus_cos_C_l150_150836


namespace vector_dot_product_calculation_l150_150217

theorem vector_dot_product_calculation : 
  let a := (2, 3, -1)
  let b := (2, 0, 3)
  let c := (0, 2, 2)
  (2 * (2 + 0) + 3 * (0 + 2) + -1 * (3 + 2)) = 5 := 
by
  sorry

end vector_dot_product_calculation_l150_150217


namespace sequence_sum_consecutive_l150_150222

theorem sequence_sum_consecutive 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 20) 
  (h8 : a 8 = 16) 
  (h_sum : ∀ i, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 100) :
  a 2 = 16 ∧ a 3 = 64 ∧ a 4 = 20 ∧ a 5 = 16 ∧ a 6 = 64 ∧ a 7 = 20 :=
  sorry

end sequence_sum_consecutive_l150_150222


namespace work_rate_c_l150_150483

theorem work_rate_c (A B C : ℝ) (h1 : A + B = 1 / 4) (h2 : B + C = 1 / 6) (h3 : C + A = 1 / 3) :
    1 / C = 8 :=
by
  sorry

end work_rate_c_l150_150483


namespace parabola_has_one_x_intercept_l150_150968

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- State the theorem that proves the number of x-intercepts
theorem parabola_has_one_x_intercept : ∃! x, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
by
  -- Proof goes here, but it's omitted
  sorry

end parabola_has_one_x_intercept_l150_150968


namespace scale_model_height_is_correct_l150_150172

noncomputable def height_of_scale_model (h_real : ℝ) (V_real : ℝ) (V_scale : ℝ) : ℝ :=
  h_real / (V_real / V_scale)^(1/3:ℝ)

theorem scale_model_height_is_correct :
  height_of_scale_model 90 500000 0.2 = 0.66 :=
by
  sorry

end scale_model_height_is_correct_l150_150172


namespace ratio_avg_speed_round_trip_l150_150764

def speed_boat := 20
def speed_current := 4
def distance := 2

theorem ratio_avg_speed_round_trip :
  let downstream_speed := speed_boat + speed_current
  let upstream_speed := speed_boat - speed_current
  let time_down := distance / downstream_speed
  let time_up := distance / upstream_speed
  let total_time := time_down + time_up
  let total_distance := distance + distance
  let avg_speed := total_distance / total_time
  avg_speed / speed_boat = 24 / 25 :=
by sorry

end ratio_avg_speed_round_trip_l150_150764


namespace alice_paid_24_percent_l150_150199

theorem alice_paid_24_percent (P : ℝ) (h1 : P > 0) :
  let MP := 0.60 * P
  let price_paid := 0.40 * MP
  (price_paid / P) * 100 = 24 :=
by
  sorry

end alice_paid_24_percent_l150_150199


namespace solution_set_of_x_squared_geq_four_l150_150569

theorem solution_set_of_x_squared_geq_four :
  {x : ℝ | x^2 ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
sorry

end solution_set_of_x_squared_geq_four_l150_150569


namespace intersection_when_a_eq_1_range_for_A_inter_B_eq_A_l150_150041

noncomputable def f (x : ℝ) : ℝ := Real.log (3 - abs (x - 1))

def setA : Set ℝ := { x | 3 - abs (x - 1) > 0 }

def setB (a : ℝ) : Set ℝ := { x | x^2 - (a + 5) * x + 5 * a < 0 }

theorem intersection_when_a_eq_1 : (setA ∩ setB 1) = { x | 1 < x ∧ x < 4 } :=
by
  sorry

theorem range_for_A_inter_B_eq_A : { a | (setA ∩ setB a) = setA } = { a | a ≤ -2 } :=
by
  sorry

end intersection_when_a_eq_1_range_for_A_inter_B_eq_A_l150_150041


namespace complex_star_angle_sum_correct_l150_150822

-- Definitions corresponding to the conditions
def complex_star_interior_angle_sum (n : ℕ) (h : n ≥ 7) : ℕ :=
  180 * (n - 4)

-- The theorem stating the problem
theorem complex_star_angle_sum_correct (n : ℕ) (h : n ≥ 7) :
  complex_star_interior_angle_sum n h = 180 * (n - 4) :=
sorry

end complex_star_angle_sum_correct_l150_150822


namespace cube_volume_l150_150860

-- Define the given condition: The surface area of the cube
def surface_area (A : ℕ) := A = 294

-- The key proposition we need to prove using the given condition
theorem cube_volume (s : ℕ) (A : ℕ) (V : ℕ) 
  (area_condition : surface_area A)
  (side_length_condition : s ^ 2 = A / 6) 
  (volume_condition : V = s ^ 3) : 
  V = 343 := by
  sorry

end cube_volume_l150_150860


namespace middle_segment_proportion_l150_150147

theorem middle_segment_proportion (a b c : ℝ) (h_a : a = 1) (h_b : b = 3) :
  (a / c = c / b) → c = Real.sqrt 3 :=
by
  sorry

end middle_segment_proportion_l150_150147


namespace max_side_length_triangle_l150_150637

theorem max_side_length_triangle (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_perimeter : a + b + c = 20) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : max a (max b c) = 9 := 
sorry

end max_side_length_triangle_l150_150637


namespace total_respondents_l150_150223

theorem total_respondents (X Y : ℕ) (h1 : X = 60) (h2 : 3 * Y = X) : X + Y = 80 :=
by
  sorry

end total_respondents_l150_150223


namespace hyperbola_eqn_l150_150206

theorem hyperbola_eqn
  (P : ℝ × ℝ) (Q : ℝ × ℝ)
  (C1 : P = (-3, 2 * Real.sqrt 7))
  (C2 : Q = (-6 * Real.sqrt 2, -7))
  (asymptote_hyperbola : ∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1)
  (special_point : ℝ × ℝ)
  (C3 : special_point = (2, 2 * Real.sqrt 3)) :
  ∃ (a b : ℝ), ¬(a = 0) ∧ ¬(b = 0) ∧ 
  (∀ x y : ℝ, (y^2 / b - x^2 / a = 1 → 
    ((y^2 / 25 - x^2 / 75 = 1) ∨ 
    (y^2 / 9 - x^2 / 12 = 1)))) :=
by
  sorry

end hyperbola_eqn_l150_150206


namespace pints_in_two_liters_l150_150037

theorem pints_in_two_liters (p : ℝ) (h : p = 1.575 / 0.75) : 2 * p = 4.2 := 
sorry

end pints_in_two_liters_l150_150037


namespace geometric_sequence_a1_l150_150144

theorem geometric_sequence_a1 (a1 a2 a3 S3 : ℝ) (q : ℝ)
  (h1 : S3 = a1 + (1 / 2) * a2)
  (h2 : a3 = (1 / 4))
  (h3 : S3 = a1 * (1 + q + q^2))
  (h4 : a2 = a1 * q)
  (h5 : a3 = a1 * q^2) :
  a1 = 1 :=
sorry

end geometric_sequence_a1_l150_150144


namespace polygon_is_hexagon_l150_150616

-- Definitions
def side_length : ℝ := 8
def perimeter : ℝ := 48

-- The main theorem to prove
theorem polygon_is_hexagon : (perimeter / side_length = 6) ∧ (48 / 8 = 6) := 
by
  sorry

end polygon_is_hexagon_l150_150616


namespace geometric_common_ratio_arithmetic_sequence_l150_150407

theorem geometric_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : S 3 = a 1 * (1 - q^3) / (1 - q)) (h2 : S 3 = 3 * a 1) :
  q = 2 ∨ q^3 = - (1 / 2) := by
  sorry

theorem arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h : S 3 = a 1 * (1 - q^3) / (1 - q))
  (h3 : 2 * S 9 = S 3 + S 6) (h4 : q ≠ 1) :
  a 2 + a 5 = 2 * a 8 := by
  sorry

end geometric_common_ratio_arithmetic_sequence_l150_150407


namespace concert_duration_l150_150561

def duration_in_minutes (hours : Int) (extra_minutes : Int) : Int :=
  hours * 60 + extra_minutes

theorem concert_duration : duration_in_minutes 7 45 = 465 :=
by
  sorry

end concert_duration_l150_150561


namespace largest_fraction_proof_l150_150435

theorem largest_fraction_proof 
  (w x y z : ℕ)
  (hw : 0 < w)
  (hx : w < x)
  (hy : x < y)
  (hz : y < z)
  (w_eq : w = 1)
  (x_eq : x = y - 1)
  (z_eq : z = y + 1)
  (y_eq : y = x!) : 
  (max (max (w + z) (w + x)) (max (x + z) (max (x + y) (y + z))) = 5 / 3) := 
sorry

end largest_fraction_proof_l150_150435


namespace sandy_initial_payment_l150_150091

theorem sandy_initial_payment (P : ℝ) (H1 : P + 300 < P + 1320)
  (H2 : 1320 = 1.10 * (P + 300)) : P = 900 :=
sorry

end sandy_initial_payment_l150_150091


namespace average_typed_words_per_minute_l150_150344

def rudy_wpm := 64
def joyce_wpm := 76
def gladys_wpm := 91
def lisa_wpm := 80
def mike_wpm := 89
def num_team_members := 5

theorem average_typed_words_per_minute : 
  (rudy_wpm + joyce_wpm + gladys_wpm + lisa_wpm + mike_wpm) / num_team_members = 80 := 
by
  sorry

end average_typed_words_per_minute_l150_150344


namespace sin_pi_div_three_l150_150470

theorem sin_pi_div_three : Real.sin (π / 3) = Real.sqrt 3 / 2 := 
sorry

end sin_pi_div_three_l150_150470


namespace exam_correct_answers_l150_150916

theorem exam_correct_answers (C W : ℕ) 
  (h1 : C + W = 60)
  (h2 : 4 * C - W = 160) : 
  C = 44 :=
sorry

end exam_correct_answers_l150_150916


namespace find_value_of_expression_l150_150395

variable (α : ℝ)

theorem find_value_of_expression 
  (h : Real.tan α = 3) : 
  (Real.sin (2 * α) / (Real.cos α)^2) = 6 := 
by 
  sorry

end find_value_of_expression_l150_150395


namespace non_neg_int_solutions_l150_150367

theorem non_neg_int_solutions :
  (∀ m n k : ℕ, 2 * m + 3 * n = k ^ 2 →
    (m = 0 ∧ n = 1 ∧ k = 2) ∨
    (m = 3 ∧ n = 0 ∧ k = 3) ∨
    (m = 4 ∧ n = 2 ∧ k = 5)) :=
by
  intro m n k h
  -- outline proof steps here
  sorry

end non_neg_int_solutions_l150_150367


namespace second_lock_less_than_three_times_first_l150_150640

variable (first_lock_time : ℕ := 5)
variable (second_lock_time : ℕ)
variable (combined_lock_time : ℕ := 60)

-- Assuming the second lock time is a fraction of the combined lock time
axiom h1 : 5 * second_lock_time = combined_lock_time

theorem second_lock_less_than_three_times_first : (3 * first_lock_time - second_lock_time) = 3 :=
by
  -- prove that the theorem is true based on given conditions.
  sorry

end second_lock_less_than_three_times_first_l150_150640


namespace find_a_l150_150186

theorem find_a (a : ℤ) :
  (∃! x : ℤ, |a * x + a + 2| < 2) ↔ a = 3 ∨ a = -3 := 
sorry

end find_a_l150_150186


namespace cash_realized_without_brokerage_l150_150427

theorem cash_realized_without_brokerage
  (C : ℝ)
  (h1 : (1 / 4) * (1 / 100) = 1 / 400)
  (h2 : C + (C / 400) = 108) :
  C = 43200 / 401 :=
by
  sorry

end cash_realized_without_brokerage_l150_150427


namespace determine_digit_l150_150431

theorem determine_digit (Θ : ℚ) (h : 312 / Θ = 40 + 2 * Θ) : Θ = 6 :=
sorry

end determine_digit_l150_150431


namespace exists_quadratic_polynomial_distinct_remainders_l150_150654

theorem exists_quadratic_polynomial_distinct_remainders :
  ∃ (a b c : ℤ), 
    (¬ (2014 ∣ a)) ∧ 
    (∀ x y : ℤ, (1 ≤ x ∧ x ≤ 2014) ∧ (1 ≤ y ∧ y ≤ 2014) → x ≠ y → 
      (1007 * x^2 + 1008 * x + c) % 2014 ≠ (1007 * y^2 + 1008 * y + c) % 2014) :=
  sorry

end exists_quadratic_polynomial_distinct_remainders_l150_150654


namespace find_vanessa_age_l150_150242

/-- Define the initial conditions and goal -/
theorem find_vanessa_age (V : ℕ) (Kevin_age current_time future_time : ℕ) :
  Kevin_age = 16 ∧ future_time = current_time + 5 ∧
  (Kevin_age + future_time - current_time) = 3 * (V + future_time - current_time) →
  V = 2 := 
by
  sorry

end find_vanessa_age_l150_150242


namespace find_a_l150_150085

theorem find_a 
  (x y a : ℝ) 
  (hx : x = 1) 
  (hy : y = -3) 
  (h : a * x - y = 1) : 
  a = -2 := 
  sorry

end find_a_l150_150085


namespace geometric_series_smallest_b_l150_150996

theorem geometric_series_smallest_b (a b c : ℝ) (h_geometric : a * c = b^2) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_product : a * b * c = 216) : b = 6 :=
sorry

end geometric_series_smallest_b_l150_150996


namespace farmer_steven_total_days_l150_150759

theorem farmer_steven_total_days 
(plow_acres_per_day : ℕ)
(mow_acres_per_day : ℕ)
(farmland_acres : ℕ)
(grassland_acres : ℕ)
(h_plow : plow_acres_per_day = 10)
(h_mow : mow_acres_per_day = 12)
(h_farmland : farmland_acres = 55)
(h_grassland : grassland_acres = 30) :
((farmland_acres / plow_acres_per_day) + (grassland_acres / mow_acres_per_day) = 8) := by
  sorry

end farmer_steven_total_days_l150_150759


namespace quadratic_inequality_l150_150457

theorem quadratic_inequality 
  (a b c : ℝ) 
  (h₁ : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1)
  (x : ℝ) 
  (hx : |x| ≤ 1) : 
  |c * x^2 + b * x + a| ≤ 2 := 
sorry

end quadratic_inequality_l150_150457


namespace speed_in_kmph_l150_150933

noncomputable def speed_conversion (speed_mps: ℝ) : ℝ :=
  speed_mps * 3.6

theorem speed_in_kmph : speed_conversion 18.334799999999998 = 66.00528 :=
by
  -- proof steps would go here
  sorry

end speed_in_kmph_l150_150933


namespace cos_A_value_compare_angles_l150_150972

variable (A B C : ℝ) (a b c : ℝ)

-- Given conditions
variable (h1 : a = 3) (h2 : b = 2 * Real.sqrt 6) (h3 : B = 2 * A)

-- Problem (I) statement
theorem cos_A_value (hcosA : Real.cos A = Real.sqrt 6 / 3) : 
  Real.cos A = Real.sqrt 6 / 3 :=
by 
  sorry

-- Problem (II) statement
theorem compare_angles (hcosA : Real.cos A = Real.sqrt 6 / 3) (hcosC : Real.cos C = Real.sqrt 6 / 9) :
  B < C :=
by
  sorry

end cos_A_value_compare_angles_l150_150972


namespace math_proof_problem_l150_150493
noncomputable def expr : ℤ := 3000 * (3000 ^ 3000) + 3000 ^ 2

theorem math_proof_problem : expr = 3000 ^ 3001 + 9000000 :=
by
  -- Proof
  sorry

end math_proof_problem_l150_150493


namespace find_real_a_l150_150209

theorem find_real_a (a : ℝ) : 
  (a ^ 2 + 2 * a - 15 = 0) ∧ (a ^ 2 + 4 * a - 5 ≠ 0) → a = 3 :=
by 
  sorry

end find_real_a_l150_150209


namespace correct_factorization_l150_150862

-- Definitions from conditions
def A: Prop := ∀ x y: ℝ, x^2 - 4*y^2 = (x + y) * (x - 4*y)
def B: Prop := ∀ x: ℝ, (x + 4) * (x - 4) = x^2 - 16
def C: Prop := ∀ x: ℝ, x^2 - 2*x + 1 = (x - 1)^2
def D: Prop := ∀ x: ℝ, x^2 - 8*x + 9 = (x - 4)^2 - 7

-- Goal is to prove that C is a correct factorization
theorem correct_factorization: C := by
  sorry

end correct_factorization_l150_150862


namespace ordered_pairs_satisfy_equation_l150_150497

theorem ordered_pairs_satisfy_equation :
  (∃ (a : ℝ) (b : ℤ), a > 0 ∧ 3 ≤ b ∧ b ≤ 203 ∧ (Real.log a / Real.log b) ^ 2021 = Real.log (a ^ 2021) / Real.log b) :=
sorry

end ordered_pairs_satisfy_equation_l150_150497


namespace binomial_fermat_l150_150899

theorem binomial_fermat (p : ℕ) (a b : ℤ) (hp : p.Prime) : 
  ((a + b)^p - a^p - b^p) % p = 0 := by
  sorry

end binomial_fermat_l150_150899


namespace river_current_speed_l150_150074

theorem river_current_speed :
  ∀ (D v A_speed B_speed time_interval : ℝ),
    D = 200 →
    A_speed = 36 →
    B_speed = 64 →
    time_interval = 4 →
    3 * D = (A_speed + v) * 2 * (1 + time_interval / ((A_speed + v) + (B_speed - v))) * 200 :=
sorry

end river_current_speed_l150_150074


namespace amusement_park_ticket_price_l150_150064

theorem amusement_park_ticket_price
  (num_people_weekday : ℕ)
  (num_people_saturday : ℕ)
  (num_people_sunday : ℕ)
  (total_people_week : ℕ)
  (total_revenue_week : ℕ)
  (people_per_day_weekday : num_people_weekday = 100)
  (people_saturday : num_people_saturday = 200)
  (people_sunday : num_people_sunday = 300)
  (total_people : total_people_week = 1000)
  (total_revenue : total_revenue_week = 3000)
  (total_people_calc : 5 * num_people_weekday + num_people_saturday + num_people_sunday = total_people_week)
  (revenue_eq : total_people_week * 3 = total_revenue_week) :
  3 = 3 :=
by
  sorry

end amusement_park_ticket_price_l150_150064


namespace focus_of_parabola_l150_150047

/-- Given a quadratic function f(x) = ax^2 + bx + 2 where a ≠ 0, and for any real number x, it holds that |f(x)| ≥ 2,
    prove that the coordinates of the focus of the parabolic curve are (0, 1 / (4 * a) + 2). -/
theorem focus_of_parabola (a b : ℝ) (h_a : a ≠ 0)
  (h_f : ∀ x : ℝ, |a * x^2 + b * x + 2| ≥ 2) :
  (0, (1 / (4 * a) + 2)) = (0, (1 / (4 * a) + 2)) :=
by
  sorry

end focus_of_parabola_l150_150047


namespace cost_of_carton_l150_150614

-- Definition of given conditions
def totalCost : ℝ := 4.88
def numberOfCartons : ℕ := 4
def costPerCarton : ℝ := 1.22

-- The proof statement
theorem cost_of_carton
  (h : totalCost = 4.88) 
  (n : numberOfCartons = 4) :
  totalCost / numberOfCartons = costPerCarton := 
sorry

end cost_of_carton_l150_150614


namespace jones_elementary_school_students_l150_150300

theorem jones_elementary_school_students
  (X : ℕ)
  (boys_percent_total : ℚ)
  (num_students_represented : ℕ)
  (percent_of_boys : ℚ)
  (h1 : boys_percent_total = 0.60)
  (h2 : num_students_represented = 90)
  (h3 : percent_of_boys * (boys_percent_total * X) = 90)
  : X = 150 :=
by
  sorry

end jones_elementary_school_students_l150_150300


namespace arithmetic_sequence_problem_l150_150784

theorem arithmetic_sequence_problem :
  let sum_first_sequence := (100 / 2) * (2501 + 2600)
  let sum_second_sequence := (100 / 2) * (401 + 500)
  let sum_third_sequence := (50 / 2) * (401 + 450)
  sum_first_sequence - sum_second_sequence - sum_third_sequence = 188725 :=
by
  let sum_first_sequence := (100 / 2) * (2501 + 2600)
  let sum_second_sequence := (100 / 2) * (401 + 500)
  let sum_third_sequence := (50 / 2) * (401 + 450)
  sorry

end arithmetic_sequence_problem_l150_150784


namespace depth_of_water_l150_150608

variable (RonHeight DepthOfWater : ℝ)

-- Definitions based on conditions
def RonStandingHeight := 12 -- Ron's height is 12 feet
def DepthOfWaterCalculation := 5 * RonStandingHeight -- Depth is 5 times Ron's height

-- Theorem statement to prove
theorem depth_of_water (hRon : RonHeight = RonStandingHeight) (hDepth : DepthOfWater = DepthOfWaterCalculation) :
  DepthOfWater = 60 := by
  sorry

end depth_of_water_l150_150608


namespace students_selected_milk_l150_150115

theorem students_selected_milk
    (total_students : ℕ)
    (students_soda students_milk students_juice : ℕ)
    (soda_percentage : ℚ)
    (milk_percentage : ℚ)
    (juice_percentage : ℚ)
    (h1 : soda_percentage = 0.7)
    (h2 : milk_percentage = 0.2)
    (h3 : juice_percentage = 0.1)
    (h4 : students_soda = 84)
    (h5 : total_students = students_soda / soda_percentage)
    : students_milk = total_students * milk_percentage :=
by
    sorry

end students_selected_milk_l150_150115


namespace tangent_condition_l150_150992

def curve1 (x y : ℝ) : Prop := y = x ^ 3 + 2
def curve2 (x y m : ℝ) : Prop := y^2 - m * x = 1

theorem tangent_condition (m : ℝ) (h : ∃ x y : ℝ, curve1 x y ∧ curve2 x y m) :
  m = 4 + 2 * Real.sqrt 3 :=
sorry

end tangent_condition_l150_150992


namespace angle_SVU_l150_150449

theorem angle_SVU (TU SV SU : ℝ) (angle_STU_T : ℝ) (angle_STU_S : ℝ) :
  TU = SV → angle_STU_T = 75 → angle_STU_S = 30 →
  TU = SU → SU = SV → S_V_U = 65 :=
by
  intros H1 H2 H3 H4 H5
  -- skip proof
  sorry

end angle_SVU_l150_150449


namespace total_population_calculation_l150_150488

theorem total_population_calculation :
  ∀ (total_lions total_leopards adult_lions adult_leopards : ℕ)
  (female_lions male_lions female_leopards male_leopards : ℕ)
  (adult_elephants baby_elephants total_elephants total_zebras : ℕ),
  total_lions = 200 →
  total_lions = 2 * total_leopards →
  adult_lions = 3 * total_lions / 4 →
  adult_leopards = 3 * total_leopards / 5 →
  female_lions = 3 * total_lions / 5 →
  male_lions = 2 * total_lions / 5 →
  female_leopards = 2 * total_leopards / 3 →
  male_leopards = total_leopards / 3 →
  adult_elephants = (adult_lions + adult_leopards) / 2 →
  baby_elephants = 100 →
  total_elephants = adult_elephants + baby_elephants →
  total_zebras = adult_elephants + total_leopards →
  total_lions + total_leopards + total_elephants + total_zebras = 710 :=
by sorry

end total_population_calculation_l150_150488


namespace zero_in_interval_l150_150798

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^(1/3)

theorem zero_in_interval : ∃ x ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ), f x = 0 :=
by
  -- The correct statement only
  sorry

end zero_in_interval_l150_150798


namespace square_side_length_equals_nine_l150_150021

-- Definitions based on the conditions
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 8
def rectangle_perimeter (length width : ℕ) : ℕ := 2 * length + 2 * width
def side_length_of_square (perimeter : ℕ) : ℕ := perimeter / 4

-- The theorem we want to prove
theorem square_side_length_equals_nine : 
  side_length_of_square (rectangle_perimeter rectangle_length rectangle_width) = 9 :=
by
  -- proof goes here
  sorry

end square_side_length_equals_nine_l150_150021


namespace sugar_ratio_l150_150645

theorem sugar_ratio (r : ℝ) (H1 : 24 * r^3 = 3) : (24 * r / 24 = 1 / 2) :=
by
  sorry

end sugar_ratio_l150_150645


namespace senior_ticket_cost_is_13_l150_150863

theorem senior_ticket_cost_is_13
    (adult_ticket_cost : ℕ)
    (child_ticket_cost : ℕ)
    (senior_ticket_cost : ℕ)
    (total_cost : ℕ)
    (num_adults : ℕ)
    (num_children : ℕ)
    (num_senior_citizens : ℕ)
    (age_child1 : ℕ)
    (age_child2 : ℕ)
    (age_child3 : ℕ) :
    adult_ticket_cost = 11 → 
    child_ticket_cost = 8 →
    total_cost = 64 →
    num_adults = 2 →
    num_children = 2 → -- children with discount tickets
    num_senior_citizens = 2 →
    age_child1 = 7 → 
    age_child2 = 10 → 
    age_child3 = 14 → -- this child does not get discount
    senior_ticket_cost * num_senior_citizens = total_cost - (num_adults * adult_ticket_cost + num_children * child_ticket_cost) →
    senior_ticket_cost = 13 :=
by
  intros
  sorry

end senior_ticket_cost_is_13_l150_150863


namespace and_false_iff_not_both_true_l150_150610

variable (p q : Prop)

theorem and_false_iff_not_both_true (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
by
    sorry

end and_false_iff_not_both_true_l150_150610


namespace solve_b_values_l150_150248

open Int

theorem solve_b_values :
  {b : ℤ | ∃ x1 x2 x3 : ℤ, x1^2 + b * x1 - 2 ≤ 0 ∧ x2^2 + b * x2 - 2 ≤ 0 ∧ x3^2 + b * x3 - 2 ≤ 0 ∧
  ∀ x : ℤ, x ≠ x1 ∧ x ≠ x2 ∧ x ≠ x3 → x^2 + b * x - 2 > 0} = { -4, -3 } :=
by sorry

end solve_b_values_l150_150248


namespace grasshopper_visit_all_points_min_jumps_l150_150099

noncomputable def grasshopper_min_jumps : ℕ := 18

theorem grasshopper_visit_all_points_min_jumps (n m : ℕ) (h₁ : n = 2014) (h₂ : m = 18) :
  ∃ k : ℕ, k ≤ m ∧ (∀ i : ℤ, 0 ≤ i → i < n → ∃ j : ℕ, j < k ∧ (j * 57 + i * 10) % n = i) :=
sorry

end grasshopper_visit_all_points_min_jumps_l150_150099


namespace find_rectangle_length_l150_150406

-- Define the problem conditions
def length_is_three_times_breadth (l b : ℕ) : Prop := l = 3 * b
def area_of_rectangle (l b : ℕ) : Prop := l * b = 6075

-- Define the theorem to prove the length of the rectangle given the conditions
theorem find_rectangle_length (l b : ℕ) (h1 : length_is_three_times_breadth l b) (h2 : area_of_rectangle l b) : l = 135 := 
sorry

end find_rectangle_length_l150_150406


namespace total_amount_after_refunds_l150_150540

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem total_amount_after_refunds : 
  individual_bookings + group_bookings - refunds = 26400 :=
by 
  -- The proof goes here.
  sorry

end total_amount_after_refunds_l150_150540


namespace more_action_figures_than_books_l150_150033

-- Definitions of initial conditions
def books : ℕ := 3
def initial_action_figures : ℕ := 4
def added_action_figures : ℕ := 2

-- Definition of final number of action figures
def final_action_figures : ℕ := initial_action_figures + added_action_figures

-- Proposition to be proved
theorem more_action_figures_than_books : final_action_figures - books = 3 := by
  -- We leave the proof empty
  sorry

end more_action_figures_than_books_l150_150033


namespace geometric_sequence_division_condition_l150_150322

variable {a : ℕ → ℝ}
variable {q : ℝ}

/-- a is a geometric sequence with common ratio q -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = a 1 * q ^ (n - 1)

/-- 3a₁, 1/2a₅, and 2a₃ forming an arithmetic sequence -/
def arithmetic_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  3 * a 1 + 2 * (a 1 * q ^ 2) = 2 * (1 / 2 * (a 1 * q ^ 4))

theorem geometric_sequence_division_condition
  (h1 : is_geometric_sequence a q)
  (h2 : arithmetic_sequence_condition a q) :
  (a 9 + a 10) / (a 7 + a 8) = 3 :=
sorry

end geometric_sequence_division_condition_l150_150322


namespace factorize_x_squared_minus_4_factorize_2mx_squared_minus_4mx_plus_2m_factorize_y_quad_l150_150510

-- Problem 1
theorem factorize_x_squared_minus_4 (x : ℝ) :
  x^2 - 4 = (x + 2) * (x - 2) :=
by { 
  sorry
}

-- Problem 2
theorem factorize_2mx_squared_minus_4mx_plus_2m (x m : ℝ) :
  2 * m * x^2 - 4 * m * x + 2 * m = 2 * m * (x - 1)^2 :=
by { 
  sorry
}

-- Problem 3
theorem factorize_y_quad (y : ℝ) :
  (y^2 - 1)^2 - 6 * (y^2 - 1) + 9 = (y + 2)^2 * (y - 2)^2 :=
by { 
  sorry
}

end factorize_x_squared_minus_4_factorize_2mx_squared_minus_4mx_plus_2m_factorize_y_quad_l150_150510


namespace emails_received_l150_150318

variable (x y : ℕ)

theorem emails_received (h1 : 3 + 6 = 9) (h2 : x + y + 9 = 10) : x + y = 1 := by
  sorry

end emails_received_l150_150318


namespace solution_set_inequality_l150_150705

theorem solution_set_inequality (x : ℝ) : 
  (-x^2 + 3 * x - 2 ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) :=
sorry

end solution_set_inequality_l150_150705


namespace even_function_a_eq_neg_one_l150_150167

-- Definitions for the function f and the condition for it being an even function
def f (x a : ℝ) := (x - 1) * (x - a)

-- The theorem stating that if f is an even function, then a = -1
theorem even_function_a_eq_neg_one (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 :=
by
  sorry

end even_function_a_eq_neg_one_l150_150167


namespace y_intercept_3x_minus_4y_eq_12_l150_150496

theorem y_intercept_3x_minus_4y_eq_12 :
  (- 4 * -3) = 12 :=
by
  sorry

end y_intercept_3x_minus_4y_eq_12_l150_150496


namespace arithmetic_sequence_geometric_sequence_l150_150145

-- Arithmetic sequence proof problem
theorem arithmetic_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n - a (n - 1) = 2) :
  ∀ n, a n = 2 * n - 1 :=
by 
  sorry

-- Geometric sequence proof problem
theorem geometric_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n / a (n - 1) = 2) :
  ∀ n, a n = 2 ^ (n - 1) :=
by 
  sorry

end arithmetic_sequence_geometric_sequence_l150_150145


namespace minimize_y_l150_150324

noncomputable def y (x a b k : ℝ) : ℝ :=
  (x - a) ^ 2 + (x - b) ^ 2 + k * x

theorem minimize_y (a b k : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x a b k ≤ y x' a b k) ∧ x = (a + b - k / 2) / 2 :=
by
  have x := (a + b - k / 2) / 2
  use x
  sorry

end minimize_y_l150_150324


namespace money_distribution_l150_150986

theorem money_distribution (A B C : ℝ) 
  (h1 : A + B + C = 500) 
  (h2 : A + C = 200) 
  (h3 : B + C = 340) : 
  C = 40 := 
sorry

end money_distribution_l150_150986


namespace simplify_tan_expression_l150_150666

noncomputable def tan_15 := Real.tan (Real.pi / 12)
noncomputable def tan_30 := Real.tan (Real.pi / 6)

theorem simplify_tan_expression : (1 + tan_15) * (1 + tan_30) = 2 := by
  sorry

end simplify_tan_expression_l150_150666


namespace average_of_consecutive_integers_l150_150387

variable (c : ℕ)
variable (d : ℕ)

-- Given condition: d == (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7
def condition1 : Prop := d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7

-- The theorem to prove
theorem average_of_consecutive_integers : condition1 c d → 
  (d + 1 + d + 2 + d + 3 + d + 4 + d + 5 + d + 6 + d + 7 + d + 8 + d + 9) / 10 = c + 9 :=
sorry

end average_of_consecutive_integers_l150_150387


namespace kim_boxes_on_thursday_l150_150578

theorem kim_boxes_on_thursday (Tues Wed Thurs : ℕ) 
(h1 : Tues = 4800)
(h2 : Tues = 2 * Wed)
(h3 : Wed = 2 * Thurs) : Thurs = 1200 :=
by
  sorry

end kim_boxes_on_thursday_l150_150578


namespace largest_base8_3digit_to_base10_l150_150289

theorem largest_base8_3digit_to_base10 : (7 * 8^2 + 7 * 8^1 + 7 * 8^0) = 511 := by
  sorry

end largest_base8_3digit_to_base10_l150_150289


namespace number_of_ways_to_enter_and_exit_l150_150631

theorem number_of_ways_to_enter_and_exit (n : ℕ) (h : n = 4) : (n * n) = 16 := by
  sorry

end number_of_ways_to_enter_and_exit_l150_150631


namespace lisa_quiz_goal_l150_150102

theorem lisa_quiz_goal (total_quizzes earned_A_on_first earned_A_goal remaining_quizzes additional_A_needed max_quizzes_below_A : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : earned_A_on_first = 30)
  (h3 : earned_A_goal = total_quizzes * 85 / 100)
  (h4 : remaining_quizzes = total_quizzes - 40)
  (h5 : additional_A_needed = earned_A_goal - earned_A_on_first)
  (h6 : max_quizzes_below_A = remaining_quizzes - additional_A_needed):
  max_quizzes_below_A = 0 :=
by sorry

end lisa_quiz_goal_l150_150102


namespace value_of_k_l150_150886

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := 2 * x^2 - k * x + 7

theorem value_of_k (k : ℝ) : f 5 - g 5 k = 40 → k = 1.4 := by
  sorry

end value_of_k_l150_150886


namespace min_fraction_sum_l150_150596

theorem min_fraction_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (∃ (z : ℝ), z = (1 / (x + 1)) + (4 / (y + 2)) ∧ z = 9 / 4) :=
by 
  sorry

end min_fraction_sum_l150_150596


namespace sequence_a_n_l150_150825

theorem sequence_a_n (a : ℕ → ℕ) (h₁ : a 1 = 1)
(h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = a (n / 2) + a ((n + 1) / 2)) :
∀ n : ℕ, a n = n :=
by
  -- skip the proof with sorry
  sorry

end sequence_a_n_l150_150825


namespace value_of_b_l150_150647

-- Definitions
def A := 45  -- in degrees
def B := 60  -- in degrees
def a := 10  -- length of side a

-- Assertion
theorem value_of_b : (b : ℝ) = 5 * Real.sqrt 6 :=
by
  -- Definitions used in previous problem conditions
  let sin_A := Real.sin (Real.pi * A / 180)
  let sin_B := Real.sin (Real.pi * B / 180)
  -- Applying the Law of Sines
  have law_of_sines := (a / sin_A) = (b / sin_B)
  -- Simplified calculation of b (not provided here; proof required later)
  sorry

end value_of_b_l150_150647


namespace sequence_properties_l150_150292

theorem sequence_properties (S : ℕ → ℝ) (a : ℕ → ℝ) :
  S 2 = 4 →
  (∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1) →
  a 1 = 1 ∧ S 5 = 121 :=
by
  intros hS2 ha
  sorry

end sequence_properties_l150_150292


namespace range_of_b_l150_150320

theorem range_of_b {b : ℝ} (h_b_ne_zero : b ≠ 0) :
  (∃ x : ℝ, (0 ≤ x ∧ x ≤ 3) ∧ (2 * x + b = 3)) ↔ -3 ≤ b ∧ b ≤ 3 ∧ b ≠ 0 :=
by
  sorry

end range_of_b_l150_150320


namespace distance_between_points_l150_150027

theorem distance_between_points :
  let x1 := 1
  let y1 := 16
  let x2 := 9
  let y2 := 3
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance = Real.sqrt 233 :=
by
  sorry

end distance_between_points_l150_150027


namespace fixed_point_of_function_l150_150605

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a^(1 - x) - 2

theorem fixed_point_of_function (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 1 = -1 := by
  sorry

end fixed_point_of_function_l150_150605


namespace find_number_l150_150602

theorem find_number {x : ℝ} 
  (h : 973 * x - 739 * x = 110305) : 
  x = 471.4 := 
by 
  sorry

end find_number_l150_150602


namespace relationship_of_points_on_inverse_proportion_l150_150669

theorem relationship_of_points_on_inverse_proportion :
  let y_1 := - 3 / - 3
  let y_2 := - 3 / - 1
  let y_3 := - 3 / (1 / 3)
  y_3 < y_1 ∧ y_1 < y_2 :=
by
  let y_1 := - 3 / - 3
  let y_2 := - 3 / - 1
  let y_3 := - 3 / (1 / 3)
  sorry

end relationship_of_points_on_inverse_proportion_l150_150669


namespace directrix_of_parabola_l150_150394

-- Define the condition given in the problem
def parabola_eq (x y : ℝ) : Prop := x^2 = 2 * y

-- Define the directrix equation property we want to prove
theorem directrix_of_parabola (x : ℝ) :
  (∃ y : ℝ, parabola_eq x y) → (∃ y : ℝ, y = -1 / 2) :=
by sorry

end directrix_of_parabola_l150_150394


namespace number_of_correct_statements_l150_150469

-- Define the statements
def statement_1 : Prop := ∀ (a : ℚ), |a| < |0| → a = 0
def statement_2 : Prop := ∃ (b : ℚ), ∀ (c : ℚ), b < 0 ∧ b ≥ c → c = b
def statement_3 : Prop := -4^6 = (-4) * (-4) * (-4) * (-4) * (-4) * (-4)
def statement_4 : Prop := ∀ (a b : ℚ), a + b = 0 → a ≠ 0 → b ≠ 0 → (a / b = -1)
def statement_5 : Prop := ∀ (c : ℚ), (0 / c = 0 ↔ c ≠ 0)

-- Define the overall proof problem
theorem number_of_correct_statements : (statement_1 ∧ statement_4) ∧ ¬(statement_2 ∨ statement_3 ∨ statement_5) :=
by
  sorry

end number_of_correct_statements_l150_150469


namespace range_of_g_l150_150786

noncomputable def g (x : ℝ) : ℤ :=
if x > -3 then
  ⌈1 / ((x + 3)^2)⌉
else
  ⌊1 / ((x + 3)^2)⌋

theorem range_of_g :
  ∀ y : ℤ, (∃ x : ℝ, g x = y) ↔ (∃ n : ℕ, y = n + 1) :=
by sorry

end range_of_g_l150_150786


namespace find_m_value_l150_150873

theorem find_m_value (m : ℚ) :
  (m * 2 / 3 + m * 4 / 9 + m * 8 / 27 = 1) → m = 27 / 38 :=
by 
  intro h
  sorry

end find_m_value_l150_150873


namespace michael_remaining_yards_l150_150306

theorem michael_remaining_yards (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) (m y : ℕ)
  (h1 : miles_per_marathon = 50)
  (h2 : yards_per_marathon = 800)
  (h3 : yards_per_mile = 1760)
  (h4 : num_marathons = 5)
  (h5 : y = (yards_per_marathon * num_marathons) % yards_per_mile)
  (h6 : m = miles_per_marathon * num_marathons + (yards_per_marathon * num_marathons) / yards_per_mile) :
  y = 480 :=
sorry

end michael_remaining_yards_l150_150306


namespace extended_morse_code_symbols_l150_150939

def symbol_count (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 2
  else if n = 3 then 1
  else if n = 4 then 1 + 4 + 1
  else if n = 5 then 1 + 8
  else 0

theorem extended_morse_code_symbols : 
  (symbol_count 1 + symbol_count 2 + symbol_count 3 + symbol_count 4 + symbol_count 5) = 20 :=
by sorry

end extended_morse_code_symbols_l150_150939


namespace john_total_cost_l150_150157

-- Define the costs and usage details
def base_cost : ℕ := 25
def cost_per_text_cent : ℕ := 10
def cost_per_extra_minute_cent : ℕ := 15
def included_hours : ℕ := 20
def texts_sent : ℕ := 150
def hours_talked : ℕ := 22

-- Prove that the total cost John had to pay is $58
def total_cost_john : ℕ :=
  let base_cost_dollars := base_cost
  let text_cost_dollars := (texts_sent * cost_per_text_cent) / 100
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_minutes_cost_dollars := (extra_minutes * cost_per_extra_minute_cent) / 100
  base_cost_dollars + text_cost_dollars + extra_minutes_cost_dollars

theorem john_total_cost (h1 : base_cost = 25)
                        (h2 : cost_per_text_cent = 10)
                        (h3 : cost_per_extra_minute_cent = 15)
                        (h4 : included_hours = 20)
                        (h5 : texts_sent = 150)
                        (h6 : hours_talked = 22) : 
  total_cost_john = 58 := by
  sorry

end john_total_cost_l150_150157


namespace raisin_weight_l150_150353

theorem raisin_weight (Wg : ℝ) (dry_grapes_fraction : ℝ) (dry_raisins_fraction : ℝ) :
  Wg = 101.99999999999999 → dry_grapes_fraction = 0.10 → dry_raisins_fraction = 0.85 → 
  Wg * dry_grapes_fraction / dry_raisins_fraction = 12 := 
by
  intros h1 h2 h3
  sorry

end raisin_weight_l150_150353


namespace buoy_radius_l150_150545

-- Define the conditions based on the given problem
def is_buoy_hole (width : ℝ) (depth : ℝ) : Prop :=
  width = 30 ∧ depth = 10

-- Define the statement to prove the radius of the buoy
theorem buoy_radius : ∀ r x : ℝ, is_buoy_hole 30 10 → (x^2 + 225 = (x + 10)^2) → r = x + 10 → r = 16.25 := by
  intros r x h_cond h_eq h_add
  sorry

end buoy_radius_l150_150545


namespace inequality_solution_fractional_equation_solution_l150_150623

-- Proof Problem 1
theorem inequality_solution (x : ℝ) : (1 - x) / 3 - x < 3 - (x + 2) / 4 → x > -2 :=
by
  sorry

-- Proof Problem 2
theorem fractional_equation_solution (x : ℝ) : (x - 2) / (2 * x - 1) + 1 = 3 / (2 * (1 - 2 * x)) → false :=
by
  sorry

end inequality_solution_fractional_equation_solution_l150_150623


namespace number_of_books_in_box_l150_150841

theorem number_of_books_in_box :
  ∀ (total_weight : ℕ) (empty_box_weight : ℕ) (book_weight : ℕ),
  total_weight = 42 →
  empty_box_weight = 6 →
  book_weight = 3 →
  (total_weight - empty_box_weight) / book_weight = 12 :=
by
  intros total_weight empty_box_weight book_weight htwe hebe hbw
  sorry

end number_of_books_in_box_l150_150841


namespace right_triangle_inequality_l150_150183

theorem right_triangle_inequality (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : b > a) (h3 : b / a < 2) :
  a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) > 4 / 9 :=
by
  sorry

end right_triangle_inequality_l150_150183


namespace greatest_sundays_in_56_days_l150_150905

theorem greatest_sundays_in_56_days (days_in_first: ℕ) (days_in_week: ℕ) (sundays_in_week: ℕ) : ℕ :=
by 
  -- Given conditions
  have days_in_first := 56
  have days_in_week := 7
  have sundays_in_week := 1

  -- Conclusion
  let num_weeks := days_in_first / days_in_week

  -- Answer
  exact num_weeks * sundays_in_week

-- This theorem establishes that the greatest number of Sundays in 56 days is indeed 8.
-- Proof: The number of Sundays in 56 days is given by the number of weeks (which is 8) times the number of Sundays per week (which is 1).

example : greatest_sundays_in_56_days 56 7 1 = 8 := 
by 
  unfold greatest_sundays_in_56_days
  exact rfl

end greatest_sundays_in_56_days_l150_150905


namespace odd_f_even_g_fg_eq_g_increasing_min_g_sum_l150_150275

noncomputable def f (x : ℝ) : ℝ := (0.5) * ((2:ℝ)^x - (2:ℝ)^(-x))
noncomputable def g (x : ℝ) : ℝ := (0.5) * ((2:ℝ)^x + (2:ℝ)^(-x))

theorem odd_f (x : ℝ) : f (-x) = -f (x) := sorry
theorem even_g (x : ℝ) : g (-x) = g (x) := sorry
theorem fg_eq (x : ℝ) : f (x) + g (x) = (2:ℝ)^x := sorry
theorem g_increasing (x : ℝ) : x ≥ 0 → ∀ y, 0 ≤ y ∧ y < x → g y < g x := sorry
theorem min_g_sum (x : ℝ) : ∃ t, t ≥ 2 ∧ (g x + g (2 * x) = 2) := sorry

end odd_f_even_g_fg_eq_g_increasing_min_g_sum_l150_150275


namespace even_three_digit_numbers_l150_150007

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l150_150007


namespace max_g_on_interval_l150_150370

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_on_interval : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3 :=
by
  sorry

end max_g_on_interval_l150_150370


namespace distance_from_center_to_line_l150_150955

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l150_150955


namespace distinct_numbers_in_T_l150_150850

-- Definitions of sequences as functions
def seq1 (k: ℕ) : ℕ := 5 * k - 3
def seq2 (l: ℕ) : ℕ := 8 * l - 5

-- Definition of sets A and B
def A : Finset ℕ := Finset.image seq1 (Finset.range 3000)
def B : Finset ℕ := Finset.image seq2 (Finset.range 3000)

-- Definition of set T as the union of A and B
def T := A ∪ B

-- Proof statement
theorem distinct_numbers_in_T : T.card = 5400 := by
  sorry

end distinct_numbers_in_T_l150_150850


namespace simplify_and_evaluate_expression_l150_150682

theorem simplify_and_evaluate_expression :
  let x := -1
  let y := Real.sqrt 2
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 5 :=
by
  let x := -1
  let y := Real.sqrt 2
  sorry

end simplify_and_evaluate_expression_l150_150682


namespace number_of_B_students_l150_150513

/-- Let x be the number of students who earn a B. 
    Given the conditions:
    - The number of students who earn an A is 0.5x.
    - The number of students who earn a C is 2x.
    - The number of students who earn a D is 0.3x.
    - The total number of students in the class is 40.
    Prove the number of students who earn a B is 40 / 3.8 = 200 / 19, approximately 11. -/
theorem number_of_B_students (x : ℝ) (h_bA: x * 0.5 + x + x * 2 + x * 0.3 = 40) : 
  x = 40 / 3.8 :=
by 
  sorry

end number_of_B_students_l150_150513


namespace trapezoid_length_relation_l150_150260

variables {A B C D M N : Type}
variables [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
variables (a b c d m n : A)
variables (h_parallel_ab_cd : A) (h_parallel_mn_ab : A) 

-- The required proof statement
theorem trapezoid_length_relation (H1 : a = h_parallel_ab_cd) 
(H2 : b = m * n + h_parallel_mn_ab - m * d)
(H3 : c = d * (h_parallel_mn_ab - a))
(H4 : n = d / (n - a))
(H5 : n = c - h_parallel_ab_cd) :
c * m * a + b * c * d = n * d * a :=
sorry

end trapezoid_length_relation_l150_150260


namespace sandy_worked_days_l150_150383

-- Definitions based on the conditions
def total_hours_worked : ℕ := 45
def hours_per_day : ℕ := 9

-- The theorem that we need to prove
theorem sandy_worked_days : total_hours_worked / hours_per_day = 5 :=
by sorry

end sandy_worked_days_l150_150383


namespace total_carrots_l150_150204

-- Define constants for the number of carrots grown by each person
def Joan_carrots : ℕ := 29
def Jessica_carrots : ℕ := 11
def Michael_carrots : ℕ := 37
def Taylor_carrots : ℕ := 24

-- The proof problem: Prove that the total number of carrots grown is 101
theorem total_carrots : Joan_carrots + Jessica_carrots + Michael_carrots + Taylor_carrots = 101 :=
by
  sorry

end total_carrots_l150_150204


namespace expr_eval_l150_150514

def expr : ℕ := 3 * 3^4 - 9^27 / 9^25

theorem expr_eval : expr = 162 := by
  -- Proof will be written here if needed
  sorry

end expr_eval_l150_150514


namespace line_through_points_decreasing_direct_proportion_function_m_l150_150282

theorem line_through_points_decreasing (x₁ x₂ y₁ y₂ k b : ℝ) (h1 : x₁ < x₂) (h2 : y₁ = k * x₁ + b) (h3 : y₂ = k * x₂ + b) (h4 : k < 0) : y₁ > y₂ :=
sorry

theorem direct_proportion_function_m (x₁ x₂ y₁ y₂ m : ℝ) (h1 : x₁ < x₂) (h2 : y₁ = (1 - 2 * m) * x₁) (h3 : y₂ = (1 - 2 * m) * x₂) (h4 : y₁ > y₂) : m > 1/2 :=
sorry

end line_through_points_decreasing_direct_proportion_function_m_l150_150282


namespace janet_more_cards_than_brenda_l150_150180

theorem janet_more_cards_than_brenda : ∀ (J B M : ℕ), M = 2 * J → J + B + M = 211 → M = 150 - 40 → J - B = 9 :=
by
  intros J B M h1 h2 h3
  sorry

end janet_more_cards_than_brenda_l150_150180


namespace employee_count_l150_150100

theorem employee_count (avg_salary : ℕ) (manager_salary : ℕ) (new_avg_increase : ℕ) (E : ℕ) :
  (avg_salary = 1500) ∧ (manager_salary = 4650) ∧ (new_avg_increase = 150) →
  1500 * E + 4650 = 1650 * (E + 1) → E = 20 :=
by
  sorry

end employee_count_l150_150100


namespace ab_plus_cd_l150_150327

variables (a b c d : ℝ)

axiom h1 : a + b + c = 1
axiom h2 : a + b + d = 6
axiom h3 : a + c + d = 15
axiom h4 : b + c + d = 10

theorem ab_plus_cd : a * b + c * d = 45.33333333333333 := 
by 
  sorry

end ab_plus_cd_l150_150327


namespace boston_trip_distance_l150_150789

theorem boston_trip_distance :
  ∃ d : ℕ, 40 * d = 440 :=
by
  sorry

end boston_trip_distance_l150_150789


namespace gcd_lcm_sum_l150_150824

theorem gcd_lcm_sum :
  ∀ (a b c d : ℕ), gcd a b + lcm c d = 74 :=
by
  let a := 42
  let b := 70
  let c := 20
  let d := 15
  sorry

end gcd_lcm_sum_l150_150824


namespace problem_statement_l150_150017

variables {totalBuyers : ℕ}
variables {C M K CM CK MK CMK : ℕ}

-- Given conditions
def conditions (totalBuyers : ℕ) (C : ℕ) (M : ℕ) (K : ℕ)
  (CM : ℕ) (CK : ℕ) (MK : ℕ) (CMK : ℕ) : Prop :=
  totalBuyers = 150 ∧
  C = 70 ∧
  M = 60 ∧
  K = 50 ∧
  CM = 25 ∧
  CK = 15 ∧
  MK = 10 ∧
  CMK = 5

-- Number of buyers who purchase at least one mixture
def buyersAtLeastOne (C : ℕ) (M : ℕ) (K : ℕ)
  (CM : ℕ) (CK : ℕ) (MK : ℕ) (CMK : ℕ) : ℕ :=
  C + M + K - CM - CK - MK + CMK

-- Number of buyers who purchase none
def buyersNone (totalBuyers : ℕ) (buyersAtLeastOne : ℕ) : ℕ :=
  totalBuyers - buyersAtLeastOne

-- Probability computation
def probabilityNone (totalBuyers : ℕ) (buyersNone : ℕ) : ℚ :=
  buyersNone / totalBuyers

-- Theorem statement
theorem problem_statement : conditions totalBuyers C M K CM CK MK CMK →
  probabilityNone totalBuyers (buyersNone totalBuyers (buyersAtLeastOne C M K CM CK MK CMK)) = 0.1 :=
by
  intros h
  -- Assumptions from the problem
  have h_total : totalBuyers = 150 := h.left
  have hC : C = 70 := h.right.left
  have hM : M = 60 := h.right.right.left
  have hK : K = 50 := h.right.right.right.left
  have hCM : CM = 25 := h.right.right.right.right.left
  have hCK : CK = 15 := h.right.right.right.right.right.left
  have hMK : MK = 10 := h.right.right.right.right.right.right.left
  have hCMK : CMK = 5 := h.right.right.right.right.right.right.right
  sorry

end problem_statement_l150_150017


namespace time_to_fill_bucket_completely_l150_150749

-- Define the conditions given in the problem
def time_to_fill_two_thirds (time_filled: ℕ) : ℕ := 90

-- Define what we need to prove
theorem time_to_fill_bucket_completely (time_filled: ℕ) : 
  time_to_fill_two_thirds time_filled = 90 → time_filled = 135 :=
by
  sorry

end time_to_fill_bucket_completely_l150_150749


namespace women_left_room_is_3_l150_150732

-- Definitions and conditions
variables (M W x : ℕ)
variables (ratio : M * 5 = W * 4) 
variables (men_entered : M + 2 = 14) 
variables (women_left : 2 * (W - x) = 24)

-- Theorem statement
theorem women_left_room_is_3 
  (ratio : M * 5 = W * 4) 
  (men_entered : M + 2 = 14) 
  (women_left : 2 * (W - x) = 24) : 
  x = 3 :=
sorry

end women_left_room_is_3_l150_150732


namespace triangle_30_60_90_PQ_l150_150660

theorem triangle_30_60_90_PQ (PR : ℝ) (hPR : PR = 18 * Real.sqrt 3) : 
  ∃ PQ : ℝ, PQ = 54 :=
by
  sorry

end triangle_30_60_90_PQ_l150_150660


namespace log_expression_l150_150960

variable (a : ℝ) (log3 : ℝ → ℝ)
axiom h_a : a = log3 2
axiom log3_8_eq : log3 8 = 3 * log3 2
axiom log3_6_eq : log3 6 = log3 2 + 1

theorem log_expression (log_def : log3 8 - 2 * log3 6 = a - 2) :
  log3 8 - 2 * log3 6 = a - 2 := by
  sorry

end log_expression_l150_150960


namespace product_of_axes_l150_150506

-- Definitions based on conditions
def ellipse (a b : ℝ) : Prop :=
  a^2 - b^2 = 64

def triangle_incircle_diameter (a b : ℝ) : Prop :=
  b + 8 - a = 4

-- Proving that (AB)(CD) = 240
theorem product_of_axes (a b : ℝ) (h₁ : ellipse a b) (h₂ : triangle_incircle_diameter a b) : 
  (2 * a) * (2 * b) = 240 :=
by
  sorry

end product_of_axes_l150_150506


namespace math_problem_l150_150676

-- Define the first part of the problem
def line_area_to_axes (line_eq : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  line_eq x y ∧ x = 4 ∧ y = -4

-- Define the second part of the problem
def line_through_fixed_point (m : ℝ) : Prop :=
  ∃ (x y : ℝ), (m * x) + y + m = 0 ∧ x = -1 ∧ y = 0

-- Theorem combining both parts
theorem math_problem (line_eq : ℝ → ℝ → Prop) (m : ℝ) :
  (∃ x y, line_area_to_axes line_eq x y → 8 = (1 / 2) * 4 * 4) ∧ line_through_fixed_point m :=
sorry

end math_problem_l150_150676


namespace smallest_n_ineq_l150_150362

theorem smallest_n_ineq : ∃ n : ℕ, 3 * Real.sqrt n - 2 * Real.sqrt (n - 1) < 0.03 ∧ 
  (∀ m : ℕ, (3 * Real.sqrt m - 2 * Real.sqrt (m - 1) < 0.03) → n ≤ m) ∧ n = 433715589 :=
by
  sorry

end smallest_n_ineq_l150_150362


namespace whipped_cream_needed_l150_150828

def total_days : ℕ := 15
def odd_days_count : ℕ := 8
def even_days_count : ℕ := 7

def pumpkin_pies_on_odd_days : ℕ := 3 * odd_days_count
def apple_pies_on_odd_days : ℕ := 2 * odd_days_count

def pumpkin_pies_on_even_days : ℕ := 2 * even_days_count
def apple_pies_on_even_days : ℕ := 4 * even_days_count

def total_pumpkin_pies_baked : ℕ := pumpkin_pies_on_odd_days + pumpkin_pies_on_even_days
def total_apple_pies_baked : ℕ := apple_pies_on_odd_days + apple_pies_on_even_days

def tiffany_pumpkin_pies_consumed : ℕ := 2
def tiffany_apple_pies_consumed : ℕ := 5

def remaining_pumpkin_pies : ℕ := total_pumpkin_pies_baked - tiffany_pumpkin_pies_consumed
def remaining_apple_pies : ℕ := total_apple_pies_baked - tiffany_apple_pies_consumed

def whipped_cream_for_pumpkin_pies : ℕ := 2 * remaining_pumpkin_pies
def whipped_cream_for_apple_pies : ℕ := remaining_apple_pies

def total_whipped_cream_needed : ℕ := whipped_cream_for_pumpkin_pies + whipped_cream_for_apple_pies

theorem whipped_cream_needed : total_whipped_cream_needed = 111 := by
  -- Proof omitted
  sorry

end whipped_cream_needed_l150_150828


namespace decreasing_interval_of_f_l150_150224

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x)

theorem decreasing_interval_of_f :
  ∀ x y : ℝ, (1 < x ∧ x < y) → f y < f x :=
by
  sorry

end decreasing_interval_of_f_l150_150224


namespace pies_sold_by_mcgee_l150_150254

/--
If Smith's Bakery sold 70 pies, and they sold 6 more than four times the number of pies that Mcgee's Bakery sold,
prove that Mcgee's Bakery sold 16 pies.
-/
theorem pies_sold_by_mcgee (x : ℕ) (h1 : 4 * x + 6 = 70) : x = 16 :=
by
  sorry

end pies_sold_by_mcgee_l150_150254


namespace cost_of_each_orange_l150_150807

theorem cost_of_each_orange (calories_per_orange : ℝ) (total_money : ℝ) (calories_needed : ℝ) (money_left : ℝ) :
  calories_per_orange = 80 → 
  total_money = 10 → 
  calories_needed = 400 → 
  money_left = 4 → 
  (total_money - money_left) / (calories_needed / calories_per_orange) = 1.2 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end cost_of_each_orange_l150_150807


namespace functional_equation_solution_l150_150106

theorem functional_equation_solution (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (x + y) * (f x - f y) = a * (x - y) * f (x + y)) :
  (a = 1 → ∃ α β : ℝ, ∀ x : ℝ, f x = α * x^2 + β * x) ∧
  (a ≠ 1 ∧ a ≠ 0 → ∀ x : ℝ, f x = 0) ∧
  (a = 0 → ∃ c : ℝ, ∀ x : ℝ, f x = c) :=
by sorry

end functional_equation_solution_l150_150106


namespace number_of_goats_l150_150533

theorem number_of_goats (C G : ℕ) 
  (h1 : C = 2) 
  (h2 : ∀ G : ℕ, 460 * C + 60 * G = 1400) 
  (h3 : 460 = 460) 
  (h4 : 60 = 60) : 
  G = 8 :=
by
  sorry

end number_of_goats_l150_150533


namespace sum_of_squares_l150_150129

theorem sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 4^2 - 2^2 = 272 :=
by
  sorry

end sum_of_squares_l150_150129


namespace cost_of_50_snacks_l150_150258

-- Definitions based on conditions
def travel_time_to_work : ℕ := 2 -- hours
def cost_of_snack : ℕ := 10 * (2 * travel_time_to_work) -- Ten times the round trip time

-- The theorem to prove
theorem cost_of_50_snacks : (50 * cost_of_snack) = 2000 := by
  sorry

end cost_of_50_snacks_l150_150258


namespace tangent_parallel_to_line_at_point_l150_150039

theorem tangent_parallel_to_line_at_point (P0 : ℝ × ℝ) 
  (curve : ℝ → ℝ) (line_slope : ℝ) : 
  curve = (fun x => x^3 + x - 2) ∧ line_slope = 4 ∧
  (∃ x0, P0 = (x0, curve x0) ∧ 3*x0^2 + 1 = line_slope) → 
  P0 = (1, 0) :=
by 
  sorry

end tangent_parallel_to_line_at_point_l150_150039


namespace red_tulips_for_smile_l150_150177

/-
Problem Statement:
Anna wants to plant red and yellow tulips in the shape of a smiley face. Given the following conditions:
1. Anna needs 8 red tulips for each eye.
2. She needs 9 times the number of red tulips in the smile to make the yellow background of the face.
3. The total number of tulips needed is 196.

Prove:
The number of red tulips needed for the smile is 18.
-/

-- Defining the conditions
def red_tulips_per_eye : Nat := 8
def total_tulips : Nat := 196
def yellow_multiplier : Nat := 9

-- Proving the number of red tulips for the smile
theorem red_tulips_for_smile (R : Nat) :
  2 * red_tulips_per_eye + R + yellow_multiplier * R = total_tulips → R = 18 :=
by
  sorry

end red_tulips_for_smile_l150_150177


namespace train_speed_correct_l150_150151

noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (time_seconds : ℝ) : ℝ :=
  (train_length + bridge_length) / time_seconds

theorem train_speed_correct :
  train_speed (400 : ℝ) (300 : ℝ) (45 : ℝ) = 700 / 45 :=
by
  sorry

end train_speed_correct_l150_150151


namespace grade_assignment_ways_l150_150243

-- Definitions
def num_students : ℕ := 10
def num_choices_per_student : ℕ := 3

-- Theorem statement
theorem grade_assignment_ways : num_choices_per_student ^ num_students = 59049 := by
  sorry

end grade_assignment_ways_l150_150243


namespace problems_per_worksheet_l150_150410

theorem problems_per_worksheet (total_worksheets : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ) (h1 : total_worksheets = 15) (h2 : graded_worksheets = 7) (h3 : remaining_problems = 24) : (remaining_problems / (total_worksheets - graded_worksheets)) = 3 :=
by {
  sorry
}

end problems_per_worksheet_l150_150410


namespace find_angle_A_find_value_of_c_l150_150447

variable {a b c A B C : ℝ}

-- Define the specific conditions as Lean 'variables' and 'axioms'
-- Condition: In triangle ABC, the sides opposite to angles A, B and C are a, b, and c respectively.
axiom triangle_ABC_sides : b = 2 * (a * Real.cos B - c)

-- Part (1): Prove the value of angle A
theorem find_angle_A (h : b = 2 * (a * Real.cos B - c)) : A = (2 * Real.pi) / 3 :=
by
  sorry

-- Condition: a * cos C = sqrt 3 and b = 1
axiom cos_C_value : a * Real.cos C = Real.sqrt 3
axiom b_value : b = 1

-- Part (2): Prove the value of c
theorem find_value_of_c (h1 : a * Real.cos C = Real.sqrt 3) (h2 : b = 1) : c = 2 * Real.sqrt 3 - 2 :=
by
  sorry

end find_angle_A_find_value_of_c_l150_150447


namespace minimum_reflection_number_l150_150832

theorem minimum_reflection_number (a b : ℕ) :
  ((a + 2) * (b + 2) = 4042) ∧ (Nat.gcd (a + 1) (b + 1) = 1) → 
  (a + b = 129) :=
sorry

end minimum_reflection_number_l150_150832


namespace initial_overs_l150_150921

theorem initial_overs {x : ℝ} (h1 : 4.2 * x + (83 / 15) * 30 = 250) : x = 20 :=
by
  sorry

end initial_overs_l150_150921


namespace more_girls_than_boys_l150_150994

theorem more_girls_than_boys (total_kids girls boys : ℕ) (h1 : total_kids = 34) (h2 : girls = 28) (h3 : total_kids = girls + boys) : girls - boys = 22 :=
by
  -- Proof placeholder
  sorry

end more_girls_than_boys_l150_150994


namespace sqrt_sqrt_81_is_9_l150_150681

theorem sqrt_sqrt_81_is_9 : Real.sqrt (Real.sqrt 81) = 3 := sorry

end sqrt_sqrt_81_is_9_l150_150681


namespace trivia_team_students_l150_150018

theorem trivia_team_students (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) (h_not_picked : not_picked = 9) 
(h_groups : groups = 3) (h_students_per_group : students_per_group = 9) :
    not_picked + (groups * students_per_group) = 36 := by
  sorry

end trivia_team_students_l150_150018


namespace find_k_eq_neg_four_thirds_l150_150808

-- Definitions based on conditions
def hash_p (k : ℚ) (p : ℚ) : ℚ := k * p + 20

-- Using the initial condition
def triple_hash_18 (k : ℚ) : ℚ :=
  let hp := hash_p k 18
  let hhp := hash_p k hp
  hash_p k hhp

-- The Lean statement for the desired proof
theorem find_k_eq_neg_four_thirds (k : ℚ) (h : triple_hash_18 k = -4) : k = -4 / 3 :=
sorry

end find_k_eq_neg_four_thirds_l150_150808


namespace product_of_reciprocals_is_9_over_4_l150_150586

noncomputable def product_of_reciprocals (a b : ℝ) : ℝ :=
  (1 / a) * (1 / b)

theorem product_of_reciprocals_is_9_over_4 (a b : ℝ) (h : a + b = 3 * a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  product_of_reciprocals a b = 9 / 4 :=
sorry

end product_of_reciprocals_is_9_over_4_l150_150586


namespace total_third_graders_l150_150662

theorem total_third_graders (num_girls : ℕ) (num_boys : ℕ) (h1 : num_girls = 57) (h2 : num_boys = 66) : num_girls + num_boys = 123 :=
by
  sorry

end total_third_graders_l150_150662


namespace gcd_sequence_terms_l150_150527

theorem gcd_sequence_terms (d m : ℕ) (hd : d > 1) (hm : m > 0) :
    ∃ k l : ℕ, k ≠ l ∧ gcd (2 ^ (2 ^ k) + d) (2 ^ (2 ^ l) + d) > m := 
sorry

end gcd_sequence_terms_l150_150527


namespace passing_marks_l150_150063

theorem passing_marks (T P : ℝ) (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) : P = 120 := 
by
  sorry

end passing_marks_l150_150063


namespace find_third_integer_l150_150036

noncomputable def third_odd_integer (x : ℤ) :=
  x + 4

theorem find_third_integer (x : ℤ) (h : 3 * x = 2 * (x + 4) + 3) : third_odd_integer x = 15 :=
by
  sorry

end find_third_integer_l150_150036


namespace evaluate_expression_l150_150076

theorem evaluate_expression : (1 - 1 / (1 - 1 / (1 + 2))) = (-1 / 2) :=
by sorry

end evaluate_expression_l150_150076


namespace mary_total_spent_l150_150766

def store1_shirt : ℝ := 13.04
def store1_jacket : ℝ := 12.27
def store2_shoes : ℝ := 44.15
def store2_dress : ℝ := 25.50
def hat_price : ℝ := 9.99
def discount : ℝ := 0.10
def store4_handbag : ℝ := 30.93
def store4_scarf : ℝ := 7.42
def sunglasses_price : ℝ := 20.75
def sales_tax : ℝ := 0.05

def store1_total : ℝ := store1_shirt + store1_jacket
def store2_total : ℝ := store2_shoes + store2_dress
def store3_total : ℝ := 
  let hat_cost := hat_price * 2
  let discount_amt := hat_cost * discount
  hat_cost - discount_amt
def store4_total : ℝ := store4_handbag + store4_scarf
def store5_total : ℝ := 
  let tax := sunglasses_price * sales_tax
  sunglasses_price + tax

def total_spent : ℝ := store1_total + store2_total + store3_total + store4_total + store5_total

theorem mary_total_spent : total_spent = 173.08 := sorry

end mary_total_spent_l150_150766


namespace greater_number_l150_150928

theorem greater_number (x y : ℕ) (h1 : x + y = 30) (h2 : x - y = 8) (h3 : x > y) : x = 19 := 
by 
  sorry

end greater_number_l150_150928


namespace inequality_proof_l150_150566

theorem inequality_proof 
  {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ} (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) (h₆ : x₆ > 0) :
  (x₂ / x₁)^5 + (x₄ / x₂)^5 + (x₆ / x₃)^5 + (x₁ / x₄)^5 + (x₃ / x₅)^5 + (x₅ / x₆)^5 ≥ 
  (x₁ / x₂) + (x₂ / x₄) + (x₃ / x₆) + (x₄ / x₁) + (x₅ / x₃) + (x₆ / x₅) := 
  sorry

end inequality_proof_l150_150566


namespace least_subtraction_to_divisible_by_prime_l150_150549

theorem least_subtraction_to_divisible_by_prime :
  ∃ k : ℕ, (k = 46) ∧ (856324 - k) % 101 = 0 :=
by
  sorry

end least_subtraction_to_divisible_by_prime_l150_150549


namespace sqrt_sum_equality_l150_150783

open Real

theorem sqrt_sum_equality :
  (sqrt (18 - 8 * sqrt 2) + sqrt (18 + 8 * sqrt 2) = 8) :=
sorry

end sqrt_sum_equality_l150_150783


namespace largest_tan_B_l150_150620

-- The context of the problem involves a triangle with given side lengths
variables (ABC : Triangle) -- A triangle ABC

-- Define the lengths of sides AB and BC
variables (AB BC : ℝ) 
-- Define the value of tan B
variable (tanB : ℝ)

-- The given conditions
def condition_1 := AB = 25
def condition_2 := BC = 20

-- Define the actual statement we need to prove
theorem largest_tan_B (ABC : Triangle) (AB BC tanB : ℝ) : 
  AB = 25 → BC = 20 → tanB = 3 / 4 := sorry

end largest_tan_B_l150_150620


namespace train_speed_in_kmph_l150_150298

theorem train_speed_in_kmph (length_in_m : ℝ) (time_in_s : ℝ) (length_in_m_eq : length_in_m = 800.064) (time_in_s_eq : time_in_s = 18) : 
  (length_in_m / 1000) / (time_in_s / 3600) = 160.0128 :=
by
  rw [length_in_m_eq, time_in_s_eq]
  /-
  To convert length in meters to kilometers, divide by 1000.
  To convert time in seconds to hours, divide by 3600.
  The speed is then computed by dividing the converted length by the converted time.
  -/
  sorry

end train_speed_in_kmph_l150_150298


namespace base9_perfect_square_l150_150302

theorem base9_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : ∃ k : ℕ, (729 * a + 81 * b + 36 + d) = k * k) :
    d = 0 ∨ d = 1 ∨ d = 4 ∨ d = 7 :=
sorry

end base9_perfect_square_l150_150302


namespace chess_tournament_participants_l150_150982

theorem chess_tournament_participants (n : ℕ) 
  (h : (n * (n - 1)) / 2 = 15) : n = 6 :=
sorry

end chess_tournament_participants_l150_150982


namespace problem_statement_l150_150416

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 5}
def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem problem_statement : star A B = {1, 7} := by
  sorry

end problem_statement_l150_150416


namespace students_neither_football_nor_cricket_l150_150034

def total_students : ℕ := 450
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def both_players : ℕ := 100

theorem students_neither_football_nor_cricket : 
  total_students - (football_players + cricket_players - both_players) = 50 := by
  sorry

end students_neither_football_nor_cricket_l150_150034


namespace jean_grandchildren_total_giveaway_l150_150845

theorem jean_grandchildren_total_giveaway :
  let num_grandchildren := 3
  let cards_per_grandchild_per_year := 2
  let amount_per_card := 80
  let total_amount_per_grandchild_per_year := cards_per_grandchild_per_year * amount_per_card
  let total_amount_per_year := num_grandchildren * total_amount_per_grandchild_per_year
  total_amount_per_year = 480 :=
by
  sorry

end jean_grandchildren_total_giveaway_l150_150845


namespace g_at_5_eq_9_l150_150632

-- Define the polynomial function g as given in the conditions
def g (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 3

-- Define the hypothesis that g(-5) = -3
axiom g_neg5 (a b c : ℝ) : g a b c (-5) = -3

-- State the theorem to prove that g(5) = 9 given the conditions
theorem g_at_5_eq_9 (a b c : ℝ) : g a b c 5 = 9 := 
by sorry

end g_at_5_eq_9_l150_150632


namespace find_f_of_2_l150_150700

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

theorem find_f_of_2 : f 2 = 5 := by
  sorry

end find_f_of_2_l150_150700


namespace integral_f_x_l150_150752

theorem integral_f_x (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2 * ∫ t in (0 : ℝ)..1, f t) : 
  ∫ t in (0 : ℝ)..1, f t = -1 / 3 := by
  sorry

end integral_f_x_l150_150752


namespace find_base_of_numeral_system_l150_150980

def base_of_numeral_system (x : ℕ) : Prop :=
  (3 * x + 4)^2 = x^3 + 5 * x^2 + 5 * x + 2

theorem find_base_of_numeral_system :
  ∃ x : ℕ, base_of_numeral_system x ∧ x = 7 := sorry

end find_base_of_numeral_system_l150_150980


namespace max_value_of_expression_eq_two_l150_150984

noncomputable def max_value_of_expression (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_a : a = 3) : ℝ :=
  (a^2 + b^2 + c^2) / c^2

theorem max_value_of_expression_eq_two (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_a : a = 3) :
  max_value_of_expression a b c h_right_triangle h_a = 2 := by
  sorry

end max_value_of_expression_eq_two_l150_150984


namespace cars_fell_in_lot_l150_150715

theorem cars_fell_in_lot (initial_cars went_out_cars came_in_cars final_cars: ℕ) (h1 : initial_cars = 25) 
    (h2 : went_out_cars = 18) (h3 : came_in_cars = 12) (h4 : final_cars = initial_cars - went_out_cars + came_in_cars) :
    initial_cars - final_cars = 6 :=
    sorry

end cars_fell_in_lot_l150_150715


namespace problem1_problem2_l150_150956

open Real

theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  sqrt a + sqrt b ≤ 2 :=
sorry

theorem problem2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (a + b^3) * (a^3 + b) ≥ 4 :=
sorry

end problem1_problem2_l150_150956


namespace Tim_carrots_count_l150_150444

theorem Tim_carrots_count (initial_potatoes new_potatoes initial_carrots final_potatoes final_carrots : ℕ) 
  (h_ratio : 3 * final_potatoes = 4 * final_carrots)
  (h_initial_potatoes : initial_potatoes = 32)
  (h_new_potatoes : new_potatoes = 28)
  (h_final_potatoes : final_potatoes = initial_potatoes + new_potatoes)
  (h_initial_ratio : 3 * 32 = 4 * initial_carrots) : 
  final_carrots = 45 :=
by {
  sorry
}

end Tim_carrots_count_l150_150444


namespace problem_statement_l150_150010

theorem problem_statement (a b : ℝ) (h : a > b) : a - 1 > b - 1 :=
sorry

end problem_statement_l150_150010


namespace equation_roots_l150_150465

theorem equation_roots (x : ℝ) : x * x = 2 * x ↔ (x = 0 ∨ x = 2) := by
  sorry

end equation_roots_l150_150465


namespace no_integer_pairs_satisfy_equation_l150_150401

def equation_satisfaction (m n : ℤ) : Prop :=
  m^3 + 3 * m^2 + 2 * m = 8 * n^3 + 12 * n^2 + 6 * n + 1

theorem no_integer_pairs_satisfy_equation :
  ¬ ∃ (m n : ℤ), equation_satisfaction m n :=
by
  sorry

end no_integer_pairs_satisfy_equation_l150_150401


namespace calculate_expression_l150_150800

theorem calculate_expression :
  (-3)^4 + (-3)^3 + 3^3 + 3^4 = 162 :=
by
  -- Since all necessary conditions are listed in the problem statement, we honor this structure
  -- The following steps are required logically but are not presently necessary for detailed proof means.
  sorry

end calculate_expression_l150_150800


namespace minimize_sum_of_digits_l150_150226

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the expression in the problem
def expression (p : ℕ) : ℕ :=
  p^4 - 5 * p^2 + 13

-- Proposition stating the conditions and the expected result
theorem minimize_sum_of_digits (p : ℕ) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  (∀ q : ℕ, Nat.Prime q → q % 2 = 1 → sum_of_digits (expression q) ≥ sum_of_digits (expression 5)) →
  p = 5 :=
by
  sorry

end minimize_sum_of_digits_l150_150226


namespace percent_of_x_is_y_l150_150303

-- Given the condition
def condition (x y : ℝ) : Prop :=
  0.70 * (x - y) = 0.30 * (x + y)

-- Prove y / x = 0.40
theorem percent_of_x_is_y (x y : ℝ) (h : condition x y) : y / x = 0.40 :=
by
  sorry

end percent_of_x_is_y_l150_150303


namespace count_even_fibonacci_first_2007_l150_150471

def fibonacci (n : Nat) : Nat :=
  if h : n = 0 then 0
  else if h : n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

def fibonacci_parity : List Bool := List.map (fun x => fibonacci x % 2 = 0) (List.range 2008)

def count_even (l : List Bool) : Nat :=
  l.foldl (fun acc x => if x then acc + 1 else acc) 0

theorem count_even_fibonacci_first_2007 : count_even (fibonacci_parity.take 2007) = 669 :=
sorry

end count_even_fibonacci_first_2007_l150_150471


namespace least_clock_equiv_square_l150_150932

def clock_equiv (h k : ℕ) : Prop := (h - k) % 24 = 0

theorem least_clock_equiv_square : ∃ (h : ℕ), h > 6 ∧ (h^2) % 24 = h % 24 ∧ (∀ (k : ℕ), k > 6 ∧ clock_equiv k (k^2) → h ≤ k) :=
sorry

end least_clock_equiv_square_l150_150932


namespace custom_op_difference_l150_150724

def custom_op (x y : ℕ) : ℕ := x * y - (x + y)

theorem custom_op_difference : custom_op 7 4 - custom_op 4 7 = 0 :=
by
  sorry

end custom_op_difference_l150_150724


namespace volume_frustum_as_fraction_of_original_l150_150080

theorem volume_frustum_as_fraction_of_original :
  let original_base_edge := 40
  let original_altitude := 20
  let smaller_altitude := original_altitude / 3
  let smaller_base_edge := original_base_edge / 3
  let volume_original := (1 / 3) * (original_base_edge * original_base_edge) * original_altitude
  let volume_smaller := (1 / 3) * (smaller_base_edge * smaller_base_edge) * smaller_altitude
  let volume_frustum := volume_original - volume_smaller
  (volume_frustum / volume_original) = (87 / 96) :=
by
  let original_base_edge := 40
  let original_altitude := 20
  let smaller_altitude := original_altitude / 3
  let smaller_base_edge := original_base_edge / 3
  let volume_original := (1 / 3) * (original_base_edge * original_base_edge) * original_altitude
  let volume_smaller := (1 / 3) * (smaller_base_edge * smaller_base_edge) * smaller_altitude
  let volume_frustum := volume_original - volume_smaller
  have h : volume_frustum / volume_original = 87 / 96 := sorry
  exact h

end volume_frustum_as_fraction_of_original_l150_150080


namespace cylinder_volume_eq_sphere_volume_l150_150900

theorem cylinder_volume_eq_sphere_volume (a h R x : ℝ) (h_pos : h > 0) (a_pos : a > 0) (R_pos : R > 0)
  (h_volume_eq : (a - h) * x^2 - a * h * x + 2 * h * R^2 = 0) :
  ∃ x : ℝ, a > h ∧ x > 0 ∧ x < h ∧ x = 2 * R^2 / a ∨ 
           h < a ∧ 0 < x ∧ x = (a * h / (a - h)) - h ∧ R^2 < h^2 / 2 :=
sorry

end cylinder_volume_eq_sphere_volume_l150_150900


namespace arrangement_count_l150_150105

def number_of_arrangements (n : ℕ) : ℕ :=
  if n = 6 then 5 * (Nat.factorial 5) else 0

theorem arrangement_count : number_of_arrangements 6 = 600 :=
by
  sorry

end arrangement_count_l150_150105


namespace daughter_work_alone_12_days_l150_150159

/-- Given a man, his wife, and their daughter working together on a piece of work. The man can complete the work in 4 days, the wife in 6 days, and together with their daughter, they can complete it in 2 days. Prove that the daughter alone would take 12 days to complete the work. -/
theorem daughter_work_alone_12_days (h1 : (1/4 : ℝ) + (1/6) + D = 1/2) : D = 1/12 :=
by
  sorry

end daughter_work_alone_12_days_l150_150159


namespace average_weight_of_all_boys_l150_150622

theorem average_weight_of_all_boys (total_boys_16 : ℕ) (avg_weight_boys_16 : ℝ)
  (total_boys_8 : ℕ) (avg_weight_boys_8 : ℝ) 
  (h1 : total_boys_16 = 16) (h2 : avg_weight_boys_16 = 50.25)
  (h3 : total_boys_8 = 8) (h4 : avg_weight_boys_8 = 45.15) : 
  (total_boys_16 * avg_weight_boys_16 + total_boys_8 * avg_weight_boys_8) / (total_boys_16 + total_boys_8) = 48.55 :=
by
  sorry

end average_weight_of_all_boys_l150_150622


namespace part_one_equation_of_line_part_two_equation_of_line_l150_150641

-- Definition of line passing through a given point
def line_through_point (a b : ℝ) (P : ℝ × ℝ) : Prop := P.1 / a + P.2 / b = 1

-- Condition: the sum of intercepts is 12
def sum_of_intercepts (a b : ℝ) : Prop := a + b = 12

-- Condition: area of triangle is 12
def area_of_triangle (a b : ℝ) : Prop := (1/2) * (abs (a * b)) = 12

-- First part: equation of the line when the sum of intercepts is 12
theorem part_one_equation_of_line (a b : ℝ) : 
  (line_through_point a b (3, 2)) ∧ (sum_of_intercepts a b) →
  (∃ x, (x = 2 ∧ (2*x)+x - 8 = 0) ∨ (x = 3 ∧ x + 3*x - 9 = 0)) :=
by
  sorry

-- Second part: equation of the line when the area of the triangle is 12
theorem part_two_equation_of_line (a b : ℝ) : 
  (line_through_point a b (3, 2)) ∧ (area_of_triangle a b) →
  ∃ x, x = 2 ∧ (2*x + 3*x - 12 = 0) :=
by
  sorry

end part_one_equation_of_line_part_two_equation_of_line_l150_150641


namespace opposite_of_neg_three_l150_150429

def opposite (x : Int) : Int := -x

theorem opposite_of_neg_three : opposite (-3) = 3 := by
  -- To be proven using Lean
  sorry

end opposite_of_neg_three_l150_150429


namespace pet_store_dogs_l150_150567

theorem pet_store_dogs (cats dogs : ℕ) (h1 : 18 = cats) (h2 : 3 * dogs = 4 * cats) : dogs = 24 :=
by
  sorry

end pet_store_dogs_l150_150567


namespace rotation_volumes_l150_150423

theorem rotation_volumes (a b c V1 V2 V3 : ℝ) (h : a^2 + b^2 = c^2)
    (hV1 : V1 = (1 / 3) * Real.pi * a^2 * b^2 / c)
    (hV2 : V2 = (1 / 3) * Real.pi * b^2 * a)
    (hV3 : V3 = (1 / 3) * Real.pi * a^2 * b) : 
    (1 / V1^2) = (1 / V2^2) + (1 / V3^2) :=
sorry

end rotation_volumes_l150_150423


namespace kerosene_cost_l150_150361

/-- Given that:
    - A dozen eggs cost as much as a pound of rice.
    - A half-liter of kerosene costs as much as 8 eggs.
    - The cost of each pound of rice is $0.33.
    - One dollar has 100 cents.
Prove that a liter of kerosene costs 44 cents.
-/
theorem kerosene_cost :
  let egg_cost := 0.33 / 12  -- Cost per egg in dollars
  let kerosene_half_liter_cost := egg_cost * 8  -- Half-liter of kerosene cost in dollars
  let kerosene_liter_cost := kerosene_half_liter_cost * 2  -- Liter of kerosene cost in dollars
  let kerosene_liter_cost_cents := kerosene_liter_cost * 100  -- Liter of kerosene cost in cents
  kerosene_liter_cost_cents = 44 :=
by
  sorry

end kerosene_cost_l150_150361


namespace parallel_lines_cond_l150_150101

theorem parallel_lines_cond (a c : ℝ) :
    (∀ (x y : ℝ), (a * x - 2 * y - 1 = 0) ↔ (6 * x - 4 * y + c = 0)) → 
        (a = 3 ∧ ∃ (c : ℝ), c ≠ -2) ∨ (a = 3 ∧ c = -2) := 
sorry

end parallel_lines_cond_l150_150101


namespace gcd_fx_x_l150_150117

noncomputable def f (x : ℕ) : ℕ := (5 * x + 3) * (8 * x + 2) * (12 * x + 7) * (3 * x + 11)

theorem gcd_fx_x (x : ℕ) (h : ∃ k : ℕ, x = 18720 * k) : Nat.gcd (f x) x = 462 :=
sorry

end gcd_fx_x_l150_150117


namespace extrema_f_unique_solution_F_l150_150546

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (1 / 2) * x^2 - m * Real.log x
noncomputable def F (x : ℝ) (m : ℝ) : ℝ := - (1 / 2) * x^2 + (m + 1) * x - m * Real.log x

theorem extrema_f (m : ℝ) :
  (m ≤ 0 → ∀ x > 0, ∀ y > 0, x ≠ y → f x m ≠ f y m) ∧
  (m > 0 → ∃ x₀ > 0, ∀ x > 0, f x₀ m ≤ f x m) :=
sorry

theorem unique_solution_F (m : ℝ) (h : m ≥ 1) :
  ∃ x₀ > 0, ∀ x > 0, F x₀ m = 0 ∧ (F x m = 0 → x = x₀) :=
sorry

end extrema_f_unique_solution_F_l150_150546
