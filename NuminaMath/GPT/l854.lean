import Mathlib

namespace NUMINAMATH_GPT_find_point_P_l854_85464

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨2, 2⟩
def N : Point := ⟨5, -2⟩

def is_on_x_axis (P : Point) : Prop :=
  P.y = 0

def is_right_angle (M N P : Point) : Prop :=
  (M.x - P.x)*(N.x - P.x) + (M.y - P.y)*(N.y - P.y) = 0

noncomputable def P1 : Point := ⟨1, 0⟩
noncomputable def P2 : Point := ⟨6, 0⟩

theorem find_point_P :
  ∃ P : Point, is_on_x_axis P ∧ is_right_angle M N P ∧ (P = P1 ∨ P = P2) :=
by
  sorry

end NUMINAMATH_GPT_find_point_P_l854_85464


namespace NUMINAMATH_GPT_value_of_box_l854_85434

theorem value_of_box (a b c : ℕ) (h1 : a + b = c) (h2 : a + b + c = 100) : c = 50 :=
sorry

end NUMINAMATH_GPT_value_of_box_l854_85434


namespace NUMINAMATH_GPT_carla_initial_marbles_l854_85439

theorem carla_initial_marbles
  (marbles_bought : ℕ)
  (total_marbles_now : ℕ)
  (h1 : marbles_bought = 134)
  (h2 : total_marbles_now = 187) :
  total_marbles_now - marbles_bought = 53 :=
by
  sorry

end NUMINAMATH_GPT_carla_initial_marbles_l854_85439


namespace NUMINAMATH_GPT_average_net_sales_per_month_l854_85426

def sales_jan : ℕ := 120
def sales_feb : ℕ := 80
def sales_mar : ℕ := 50
def sales_apr : ℕ := 130
def sales_may : ℕ := 90
def sales_jun : ℕ := 160

def monthly_expense : ℕ := 30
def num_months : ℕ := 6

def total_sales := sales_jan + sales_feb + sales_mar + sales_apr + sales_may + sales_jun
def total_expenses := monthly_expense * num_months
def net_total_sales := total_sales - total_expenses

theorem average_net_sales_per_month : net_total_sales / num_months = 75 :=
by {
  -- Lean code for proof here
  sorry
}

end NUMINAMATH_GPT_average_net_sales_per_month_l854_85426


namespace NUMINAMATH_GPT_bowling_tournament_l854_85450

def num_possible_orders : ℕ := 32

theorem bowling_tournament : num_possible_orders = 2 * 2 * 2 * 2 * 2 := by
  -- The structure of the playoff with 2 choices per match until all matches are played,
  -- leading to a total of 5 rounds and 2 choices per round, hence 2^5 = 32.
  sorry

end NUMINAMATH_GPT_bowling_tournament_l854_85450


namespace NUMINAMATH_GPT_sequence_properties_l854_85449

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_seq (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given conditions
variables (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ)
  (h_arith : arithmetic_seq a 2)
  (h_sum_prop : sum_seq a S)
  (h_ratio : ∀ n, S (2 * n) / S n = 4)
  (b : ℕ → ℤ) (T : ℕ → ℤ)
  (h_b : ∀ n, b n = a n * 2 ^ (n - 1))

-- Prove the sequences
theorem sequence_properties :
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, S n = n^2) ∧
  (∀ n, T n = (2 * n - 3) * 2^n + 3) :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l854_85449


namespace NUMINAMATH_GPT_system_solution_l854_85490

theorem system_solution (u v w : ℚ) 
  (h1 : 3 * u - 4 * v + w = 26)
  (h2 : 6 * u + 5 * v - 2 * w = -17) :
  u + v + w = 101 / 3 :=
sorry

end NUMINAMATH_GPT_system_solution_l854_85490


namespace NUMINAMATH_GPT_transform_f_to_g_l854_85451

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem transform_f_to_g :
  ∀ x : ℝ, g x = f (x + (π / 8)) :=
by
  sorry

end NUMINAMATH_GPT_transform_f_to_g_l854_85451


namespace NUMINAMATH_GPT_intercepts_of_line_l854_85410

theorem intercepts_of_line (x y : ℝ) (h_eq : 4 * x + 7 * y = 28) :
  (∃ y, (x = 0 ∧ y = 4) ∧ ∃ x, (y = 0 ∧ x = 7)) :=
by
  sorry

end NUMINAMATH_GPT_intercepts_of_line_l854_85410


namespace NUMINAMATH_GPT_combined_distance_l854_85430

noncomputable def radius_wheel1 : ℝ := 22.4
noncomputable def revolutions_wheel1 : ℕ := 750

noncomputable def radius_wheel2 : ℝ := 15.8
noncomputable def revolutions_wheel2 : ℕ := 950

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def distance_covered (r : ℝ) (rev : ℕ) : ℝ := circumference r * rev

theorem combined_distance :
  distance_covered radius_wheel1 revolutions_wheel1 + distance_covered radius_wheel2 revolutions_wheel2 = 199896.96 := by
  sorry

end NUMINAMATH_GPT_combined_distance_l854_85430


namespace NUMINAMATH_GPT_product_of_prs_l854_85478

theorem product_of_prs 
  (p r s : Nat) 
  (h1 : 3^p + 3^5 = 270) 
  (h2 : 2^r + 58 = 122) 
  (h3 : 7^2 + 5^s = 2504) : 
  p * r * s = 54 := 
sorry

end NUMINAMATH_GPT_product_of_prs_l854_85478


namespace NUMINAMATH_GPT_theta_value_l854_85407

theorem theta_value (theta : ℝ) (h1 : 0 ≤ theta ∧ theta ≤ 90)
    (h2 : Real.cos 60 = Real.cos 45 * Real.cos theta) : theta = 45 :=
  sorry

end NUMINAMATH_GPT_theta_value_l854_85407


namespace NUMINAMATH_GPT_MH_greater_than_MK_l854_85415

-- Defining the conditions: BH perpendicular to HK and BH = 2
def BH := 2

-- Defining the conditions: CK perpendicular to HK and CK = 5
def CK := 5

-- M is the midpoint of BC, which implicitly means MB = MC in length
def M_midpoint_BC (MB MC : ℝ) :=
  MB = MC

theorem MH_greater_than_MK (MB MC MH MK : ℝ) 
  (hM_midpoint : M_midpoint_BC MB MC)
  (hMH : MH^2 + BH^2 = MB^2)
  (hMK : MK^2 + CK^2 = MC^2) :
  MH > MK :=
by
  sorry

end NUMINAMATH_GPT_MH_greater_than_MK_l854_85415


namespace NUMINAMATH_GPT_wechat_group_member_count_l854_85482

theorem wechat_group_member_count :
  (∃ x : ℕ, x * (x - 1) / 2 = 72) → ∃ x : ℕ, x = 9 :=
by
  sorry

end NUMINAMATH_GPT_wechat_group_member_count_l854_85482


namespace NUMINAMATH_GPT_similar_triangle_perimeter_l854_85466

theorem similar_triangle_perimeter 
  (a b c : ℝ) (ha : a = 12) (hb : b = 12) (hc : c = 24) 
  (k : ℝ) (hk : k = 1.5) : 
  (1.5 * a) + (1.5 * b) + (1.5 * c) = 72 :=
by
  sorry

end NUMINAMATH_GPT_similar_triangle_perimeter_l854_85466


namespace NUMINAMATH_GPT_line_intersects_circle_l854_85414

theorem line_intersects_circle : 
  ∀ (x y : ℝ), 
  (2 * x + y = 0) ∧ (x^2 + y^2 + 2 * x - 4 * y - 4 = 0) ↔
    ∃ (x0 y0 : ℝ), (2 * x0 + y0 = 0) ∧ ((x0 + 1)^2 + (y0 - 2)^2 = 9) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_circle_l854_85414


namespace NUMINAMATH_GPT_large_beaker_multiple_small_beaker_l854_85465

variables (S L : ℝ) (k : ℝ)

theorem large_beaker_multiple_small_beaker 
  (h1 : Small_beaker = S)
  (h2 : Large_beaker = k * S)
  (h3 : Salt_water_in_small = S/2)
  (h4 : Fresh_water_in_large = (Large_beaker) / 5)
  (h5 : (Salt_water_in_small + Fresh_water_in_large = 0.3 * (Large_beaker))) :
  k = 5 :=
sorry

end NUMINAMATH_GPT_large_beaker_multiple_small_beaker_l854_85465


namespace NUMINAMATH_GPT_solve_inequality_l854_85479

theorem solve_inequality :
  {x : ℝ | x ∈ { y | (y^2 - 5*y + 6) / (y - 3)^2 > 0 }} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l854_85479


namespace NUMINAMATH_GPT_no_students_unable_to_partner_l854_85416

def students_males_females :=
  let males_6th_class1 : Nat := 17
  let females_6th_class1 : Nat := 13
  let males_6th_class2 : Nat := 14
  let females_6th_class2 : Nat := 18
  let males_6th_class3 : Nat := 15
  let females_6th_class3 : Nat := 17
  let males_7th_class : Nat := 22
  let females_7th_class : Nat := 20

  let total_males := males_6th_class1 + males_6th_class2 + males_6th_class3 + males_7th_class
  let total_females := females_6th_class1 + females_6th_class2 + females_6th_class3 + females_7th_class

  total_males == total_females

theorem no_students_unable_to_partner : students_males_females = true := by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_no_students_unable_to_partner_l854_85416


namespace NUMINAMATH_GPT_no_integer_solutions_l854_85424

theorem no_integer_solutions (a : ℕ) (h : a % 4 = 3) : ¬∃ (x y : ℤ), x^2 + y^2 = a := by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l854_85424


namespace NUMINAMATH_GPT_consecutive_natural_numbers_sum_l854_85452

theorem consecutive_natural_numbers_sum :
  (∃ (n : ℕ), 0 < n → n ≤ 4 ∧ (n-1) + n + (n+1) ≤ 12) → 
  (∃ n_sets : ℕ, n_sets = 4) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_natural_numbers_sum_l854_85452


namespace NUMINAMATH_GPT_shorter_piece_length_l854_85471

theorem shorter_piece_length (P : ℝ) (Q : ℝ) (h1 : P + Q = 68) (h2 : Q = P + 12) : P = 28 := 
by
  sorry

end NUMINAMATH_GPT_shorter_piece_length_l854_85471


namespace NUMINAMATH_GPT_polynomial_term_equality_l854_85481

theorem polynomial_term_equality (p q : ℝ) (hpq_pos : 0 < p) (hq_pos : 0 < q) 
  (h_sum : p + q = 1) (h_eq : 28 * p^6 * q^2 = 56 * p^5 * q^3) : p = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_term_equality_l854_85481


namespace NUMINAMATH_GPT_geometric_sequence_value_of_b_l854_85436

theorem geometric_sequence_value_of_b :
  ∀ (a b c : ℝ), 
  (∃ q : ℝ, q ≠ 0 ∧ a = 1 * q ∧ b = 1 * q^2 ∧ c = 1 * q^3 ∧ 4 = 1 * q^4) → 
  b = 2 :=
by
  intro a b c
  intro h
  obtain ⟨q, hq0, ha, hb, hc, hd⟩ := h
  sorry

end NUMINAMATH_GPT_geometric_sequence_value_of_b_l854_85436


namespace NUMINAMATH_GPT_simplify_and_evaluate_l854_85420

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : 
  ( ( (x^2 - 4 * x + 4) / (x^2 - 4) ) / ( (x-2) / (x^2 + 2*x) ) ) + 3 = 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l854_85420


namespace NUMINAMATH_GPT_unique_zero_a_neg_l854_85475

noncomputable def f (a x : ℝ) : ℝ := 3 * Real.exp (abs (x - 1)) - a * (2^(x - 1) + 2^(1 - x)) - a^2

theorem unique_zero_a_neg (a : ℝ) (h_unique : ∃! x : ℝ, f a x = 0) (h_neg : a < 0) : a = -3 := 
sorry

end NUMINAMATH_GPT_unique_zero_a_neg_l854_85475


namespace NUMINAMATH_GPT_false_statement_d_l854_85443

-- Define lines and planes
variables (l m : Type*) (α β : Type*)

-- Define parallel relation
def parallel (l m : Type*) : Prop := sorry

-- Define subset relation
def in_plane (l : Type*) (α : Type*) : Prop := sorry

-- Define the given conditions
axiom l_parallel_alpha : parallel l α
axiom m_in_alpha : in_plane m α

-- Main theorem statement: prove \( l \parallel m \) is false given the conditions.
theorem false_statement_d : ¬ parallel l m :=
sorry

end NUMINAMATH_GPT_false_statement_d_l854_85443


namespace NUMINAMATH_GPT_work_completion_rate_l854_85496

theorem work_completion_rate (A B D : ℝ) (W : ℝ) (hB : B = W / 9) (hA : A = W / 10) (hD : D = 90 / 19) : 
  (A + B) * D = W := 
by 
  sorry

end NUMINAMATH_GPT_work_completion_rate_l854_85496


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l854_85463

theorem solve_eq1 (x : ℝ) : 3 * (x - 2) ^ 2 = 27 ↔ (x = 5 ∨ x = -1) :=
by
  sorry

theorem solve_eq2 (x : ℝ) : (x + 5) ^ 3 + 27 = 0 ↔ x = -8 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l854_85463


namespace NUMINAMATH_GPT_sum_d_e_f_l854_85497

-- Define the variables
variables (d e f : ℤ)

-- Given conditions
def condition1 : Prop := ∀ x : ℤ, x^2 + 18 * x + 77 = (x + d) * (x + e)
def condition2 : Prop := ∀ x : ℤ, x^2 - 19 * x + 88 = (x - e) * (x - f)

-- Prove the statement
theorem sum_d_e_f : condition1 d e → condition2 e f → d + e + f = 26 :=
by
  intros h1 h2
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_d_e_f_l854_85497


namespace NUMINAMATH_GPT_number_of_tetrises_l854_85456

theorem number_of_tetrises 
  (points_per_single : ℕ := 1000)
  (points_per_tetris : ℕ := 8 * points_per_single)
  (singles_scored : ℕ := 6)
  (total_score : ℕ := 38000) :
  (total_score - (singles_scored * points_per_single)) / points_per_tetris = 4 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_tetrises_l854_85456


namespace NUMINAMATH_GPT_current_speed_l854_85444

-- Define the constants based on conditions
def rowing_speed_kmph : Float := 24
def distance_meters : Float := 40
def time_seconds : Float := 4.499640028797696

-- Intermediate calculation: Convert rowing speed from km/h to m/s
def rowing_speed_mps : Float := rowing_speed_kmph * 1000 / 3600

-- Calculate downstream speed
def downstream_speed_mps : Float := distance_meters / time_seconds

-- Define the expected speed of the current
def expected_current_speed : Float := 2.22311111

-- The theorem to prove
theorem current_speed : 
  (downstream_speed_mps - rowing_speed_mps) = expected_current_speed :=
by 
  -- skipping the proof steps, as instructed
  sorry

end NUMINAMATH_GPT_current_speed_l854_85444


namespace NUMINAMATH_GPT_does_not_uniquely_determine_equilateral_l854_85421

def equilateral_triangle (a b c : ℕ) : Prop :=
a = b ∧ b = c

def right_triangle (a b c : ℕ) : Prop :=
a^2 + b^2 = c^2

def isosceles_triangle (a b c : ℕ) : Prop :=
a = b ∨ b = c ∨ a = c

def scalene_triangle (a b c : ℕ) : Prop :=
a ≠ b ∧ b ≠ c ∧ a ≠ c

def circumscribed_circle_radius (a b c r : ℕ) : Prop :=
r = a * b * c / (4 * (a * b * c))

def angle_condition (α β γ : ℕ) (t : ℕ → ℕ → ℕ → Prop) : Prop :=
∃ (a b c : ℕ), t a b c ∧ α + β + γ = 180

theorem does_not_uniquely_determine_equilateral :
  ¬ ∃ (α β : ℕ), equilateral_triangle α β β ∧ α + β = 120 :=
sorry

end NUMINAMATH_GPT_does_not_uniquely_determine_equilateral_l854_85421


namespace NUMINAMATH_GPT_sequence_recurrence_l854_85446

theorem sequence_recurrence (v : ℕ → ℝ) (h_rec : ∀ n, v (n + 2) = 3 * v (n + 1) + 2 * v n) 
    (h_v3 : v 3 = 8) (h_v6 : v 6 = 245) : v 5 = 70 :=
sorry

end NUMINAMATH_GPT_sequence_recurrence_l854_85446


namespace NUMINAMATH_GPT_smallest_positive_integer_n_l854_85432

theorem smallest_positive_integer_n :
  ∃ (n : ℕ), 5 * n ≡ 1978 [MOD 26] ∧ n = 16 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_l854_85432


namespace NUMINAMATH_GPT_integer_values_of_x_for_equation_l854_85440

theorem integer_values_of_x_for_equation 
  (a b c : ℤ) (h1 : a ≠ 0) (h2 : a = b + c ∨ b = c + a ∨ c = b + a) : 
  ∃ x : ℤ, a * x + b = c :=
sorry

end NUMINAMATH_GPT_integer_values_of_x_for_equation_l854_85440


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l854_85455

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l854_85455


namespace NUMINAMATH_GPT_value_of_expression_l854_85427

-- Define the hypothesis and the goal
theorem value_of_expression (x y : ℝ) (h : 3 * y - x^2 = -5) : 6 * y - 2 * x^2 - 6 = -16 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l854_85427


namespace NUMINAMATH_GPT_range_of_a_l854_85413

theorem range_of_a {a : ℝ} (h : ∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) :
  a ≤ -1 ∧ a ≠ -2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l854_85413


namespace NUMINAMATH_GPT_alternate_seating_boys_l854_85437

theorem alternate_seating_boys (B : ℕ) (girl : ℕ) (ways : ℕ)
  (h1 : girl = 1)
  (h2 : ways = 24)
  (h3 : ways = B - 1) :
  B = 25 :=
sorry

end NUMINAMATH_GPT_alternate_seating_boys_l854_85437


namespace NUMINAMATH_GPT_willowbrook_team_combinations_l854_85467

theorem willowbrook_team_combinations :
  let girls := 5
  let boys := 5
  let choose_three (n : ℕ) := n.choose 3
  let team_count := choose_three girls * choose_three boys
  team_count = 100 :=
by
  let girls := 5
  let boys := 5
  let choose_three (n : ℕ) := n.choose 3
  let team_count := choose_three girls * choose_three boys
  have h1 : choose_three girls = 10 := by sorry
  have h2 : choose_three boys = 10 := by sorry
  have h3 : team_count = 10 * 10 := by sorry
  exact h3

end NUMINAMATH_GPT_willowbrook_team_combinations_l854_85467


namespace NUMINAMATH_GPT_convex_polygon_sides_ne_14_l854_85470

noncomputable def side_length : ℝ := 1

def is_triangle (s : ℝ) : Prop :=
  s = side_length

def is_dodecagon (s : ℝ) : Prop :=
  s = side_length

def side_coincide (t : ℝ) (d : ℝ) : Prop :=
  is_triangle t ∧ is_dodecagon d ∧ t = d

def valid_resulting_sides (s : ℤ) : Prop :=
  s = 11 ∨ s = 12 ∨ s = 13

theorem convex_polygon_sides_ne_14 : ∀ t d, side_coincide t d → ¬ valid_resulting_sides 14 := 
by
  intro t d h
  sorry

end NUMINAMATH_GPT_convex_polygon_sides_ne_14_l854_85470


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l854_85428

theorem geometric_sequence_third_term 
  (a r : ℝ)
  (h1 : a = 3)
  (h2 : a * r^4 = 243) : 
  a * r^2 = 27 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l854_85428


namespace NUMINAMATH_GPT_initial_riding_time_l854_85425

theorem initial_riding_time (t : ℝ) (h1 : t * 60 + 90 + 30 + 120 = 270) : t * 60 = 30 :=
by sorry

end NUMINAMATH_GPT_initial_riding_time_l854_85425


namespace NUMINAMATH_GPT_alyssa_games_last_year_l854_85489

theorem alyssa_games_last_year (games_this_year games_next_year games_total games_last_year : ℕ) (h1 : games_this_year = 11) (h2 : games_next_year = 15) (h3 : games_total = 39) (h4 : games_last_year + games_this_year + games_next_year = games_total) : games_last_year = 13 :=
by
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_alyssa_games_last_year_l854_85489


namespace NUMINAMATH_GPT_geometric_sequence_S4_l854_85492

-- Definitions from the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n)

def sum_of_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = a 1 * ((1 - (a 2 / a 1)^(n+1)) / (1 - (a 2 / a 1)))

def given_condition (S : ℕ → ℝ) : Prop :=
S 7 - 4 * S 6 + 3 * S 5 = 0

-- Problem statement to prove
theorem geometric_sequence_S4 (a : ℕ → ℝ) (S : ℕ → ℝ) (h_geom : is_geometric_sequence a)
  (h_a1 : a 1 = 1) (h_sum : sum_of_geometric_sequence a S) (h_cond : given_condition S) :
  S 4 = 40 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_S4_l854_85492


namespace NUMINAMATH_GPT_find_speed_of_first_train_l854_85447

variable (L1 L2 : ℝ) (V1 V2 : ℝ) (t : ℝ)

theorem find_speed_of_first_train (hL1 : L1 = 100) (hL2 : L2 = 200) (hV2 : V2 = 30) (ht: t = 14.998800095992321) :
  V1 = 42.005334224 := by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_find_speed_of_first_train_l854_85447


namespace NUMINAMATH_GPT_john_monthly_income_l854_85461

theorem john_monthly_income (I : ℝ) (h : I - 0.05 * I = 1900) : I = 2000 :=
by
  sorry

end NUMINAMATH_GPT_john_monthly_income_l854_85461


namespace NUMINAMATH_GPT_ethanol_percentage_in_fuel_A_l854_85459

noncomputable def percent_ethanol_in_fuel_A : ℝ := 0.12

theorem ethanol_percentage_in_fuel_A
  (fuel_tank_capacity : ℝ)
  (fuel_A_volume : ℝ)
  (fuel_B_volume : ℝ)
  (fuel_B_ethanol_percent : ℝ)
  (total_ethanol : ℝ) :
  fuel_tank_capacity = 218 → 
  fuel_A_volume = 122 → 
  fuel_B_volume = 96 → 
  fuel_B_ethanol_percent = 0.16 → 
  total_ethanol = 30 → 
  (fuel_A_volume * percent_ethanol_in_fuel_A) + (fuel_B_volume * fuel_B_ethanol_percent) = total_ethanol :=
by
  sorry

end NUMINAMATH_GPT_ethanol_percentage_in_fuel_A_l854_85459


namespace NUMINAMATH_GPT_distributor_profit_percentage_l854_85499

theorem distributor_profit_percentage 
    (commission_rate : ℝ) (cost_price : ℝ) (final_price : ℝ) (P : ℝ) (profit : ℝ) 
    (profit_percentage: ℝ) :
  commission_rate = 0.20 →
  cost_price = 15 →
  final_price = 19.8 →
  0.80 * P = final_price →
  P = cost_price + profit →
  profit_percentage = (profit / cost_price) * 100 →
  profit_percentage = 65 :=
by
  intros h_commission_rate h_cost_price h_final_price h_equation h_profit_eq h_percent_eq
  sorry

end NUMINAMATH_GPT_distributor_profit_percentage_l854_85499


namespace NUMINAMATH_GPT_school_dinner_theater_tickets_l854_85486

theorem school_dinner_theater_tickets (x y : ℕ)
  (h1 : x + y = 225)
  (h2 : 6 * x + 9 * y = 1875) :
  x = 50 :=
by
  sorry

end NUMINAMATH_GPT_school_dinner_theater_tickets_l854_85486


namespace NUMINAMATH_GPT_fred_gave_balloons_to_sandy_l854_85419

-- Define the number of balloons Fred originally had
def original_balloons : ℕ := 709

-- Define the number of balloons Fred has now
def current_balloons : ℕ := 488

-- Define the number of balloons Fred gave to Sandy
def balloons_given := original_balloons - current_balloons

-- Theorem: The number of balloons given to Sandy is 221
theorem fred_gave_balloons_to_sandy : balloons_given = 221 :=
by
  sorry

end NUMINAMATH_GPT_fred_gave_balloons_to_sandy_l854_85419


namespace NUMINAMATH_GPT_algebraic_expression_value_l854_85441

theorem algebraic_expression_value (m : ℝ) (h : m^2 + 2*m - 1 = 0) : 2*m^2 + 4*m + 2021 = 2023 := 
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l854_85441


namespace NUMINAMATH_GPT_triangle_is_right_angled_l854_85403

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

def is_right_angle_triangle (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  dot_product AB BC = 0

theorem triangle_is_right_angled :
  let A := { x := 2, y := 5 }
  let B := { x := 5, y := 2 }
  let C := { x := 10, y := 7 }
  is_right_angle_triangle A B C :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_right_angled_l854_85403


namespace NUMINAMATH_GPT_total_stamps_l854_85431

def num_foreign_stamps : ℕ := 90
def num_old_stamps : ℕ := 70
def num_both_foreign_old_stamps : ℕ := 20
def num_neither_stamps : ℕ := 60

theorem total_stamps :
  (num_foreign_stamps + num_old_stamps - num_both_foreign_old_stamps + num_neither_stamps) = 220 :=
  by
    sorry

end NUMINAMATH_GPT_total_stamps_l854_85431


namespace NUMINAMATH_GPT_coordinates_of_B_l854_85477

theorem coordinates_of_B (A B : ℝ × ℝ) (h1 : A = (-2, 3)) (h2 : (A.1 = B.1 ∨ A.1 + 1 = B.1 ∨ A.1 - 1 = B.1)) (h3 : A.2 = B.2) : 
  B = (-1, 3) ∨ B = (-3, 3) := 
sorry

end NUMINAMATH_GPT_coordinates_of_B_l854_85477


namespace NUMINAMATH_GPT_arun_working_days_l854_85484

theorem arun_working_days (A T : ℝ) 
  (h1 : A + T = 1/10) 
  (h2 : A = 1/18) : 
  (1 / A) = 18 :=
by
  -- Proof will be skipped
  sorry

end NUMINAMATH_GPT_arun_working_days_l854_85484


namespace NUMINAMATH_GPT_increased_time_between_maintenance_checks_l854_85494

theorem increased_time_between_maintenance_checks (original_time : ℕ) (percentage_increase : ℕ) : 
  original_time = 20 → percentage_increase = 25 →
  original_time + (original_time * percentage_increase / 100) = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_increased_time_between_maintenance_checks_l854_85494


namespace NUMINAMATH_GPT_cost_of_supplies_l854_85458

theorem cost_of_supplies (x y z : ℝ) 
  (h1 : 3 * x + 7 * y + z = 3.15) 
  (h2 : 4 * x + 10 * y + z = 4.2) :
  (x + y + z = 1.05) :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_supplies_l854_85458


namespace NUMINAMATH_GPT_div_inside_parentheses_l854_85408

theorem div_inside_parentheses :
  100 / (6 / 2) = 100 / 3 :=
by
  sorry

end NUMINAMATH_GPT_div_inside_parentheses_l854_85408


namespace NUMINAMATH_GPT_inequality_holds_l854_85493

theorem inequality_holds (k n : ℕ) (x : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ 1) :
  (1 - (1 - x)^n)^k ≥ 1 - (1 - x^k)^n :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l854_85493


namespace NUMINAMATH_GPT_smallest_k_l854_85411

theorem smallest_k (m n k : ℤ) (h : 221 * m + 247 * n + 323 * k = 2001) (hk : k > 100) : 
∃ k', k' = 111 ∧ k' > 100 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_l854_85411


namespace NUMINAMATH_GPT_series_value_is_correct_l854_85417

noncomputable def check_series_value : ℚ :=
  let p : ℚ := 1859 / 84
  let q : ℚ := -1024 / 63
  let r : ℚ := 512 / 63
  let m : ℕ := 3907
  let n : ℕ := 84
  100 * m + n

theorem series_value_is_correct : check_series_value = 390784 := 
by 
  sorry

end NUMINAMATH_GPT_series_value_is_correct_l854_85417


namespace NUMINAMATH_GPT_total_pencils_correct_l854_85406

def Mitchell_pencils := 30
def Antonio_pencils := Mitchell_pencils - 6
def total_pencils := Antonio_pencils + Mitchell_pencils

theorem total_pencils_correct : total_pencils = 54 := by
  sorry

end NUMINAMATH_GPT_total_pencils_correct_l854_85406


namespace NUMINAMATH_GPT_allen_mother_age_l854_85405

variable (A M : ℕ)

theorem allen_mother_age (h1 : A = M - 25) (h2 : (A + 3) + (M + 3) = 41) : M = 30 :=
by
  sorry

end NUMINAMATH_GPT_allen_mother_age_l854_85405


namespace NUMINAMATH_GPT_exp_calculation_l854_85498

theorem exp_calculation : 0.125^8 * (-8)^7 = -0.125 :=
by
  -- conditions used directly in proof
  have h1 : 0.125 = 1 / 8 := sorry
  have h2 : (-1)^7 = -1 := sorry
  -- the problem statement
  sorry

end NUMINAMATH_GPT_exp_calculation_l854_85498


namespace NUMINAMATH_GPT_roots_of_polynomial_l854_85487

-- Define the polynomial P(x) = x^3 - 3x^2 - x + 3
def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

-- Define the statement to prove the roots of the polynomial
theorem roots_of_polynomial :
  ∀ x : ℝ, (P x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l854_85487


namespace NUMINAMATH_GPT_restaurant_total_spent_l854_85453

theorem restaurant_total_spent (appetizer_cost : ℕ) (entree_cost : ℕ) (num_entrees : ℕ) (tip_rate : ℚ) 
  (H1 : appetizer_cost = 10) (H2 : entree_cost = 20) (H3 : num_entrees = 4) (H4 : tip_rate = 0.20) :
  appetizer_cost + num_entrees * entree_cost + (appetizer_cost + num_entrees * entree_cost) * tip_rate = 108 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_total_spent_l854_85453


namespace NUMINAMATH_GPT_count_not_squares_or_cubes_l854_85488

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end NUMINAMATH_GPT_count_not_squares_or_cubes_l854_85488


namespace NUMINAMATH_GPT_no_common_points_eq_l854_85462

theorem no_common_points_eq (a : ℝ) : 
  ((∀ x y : ℝ, y = (a^2 - a) * x + 1 - a → y ≠ 2 * x - 1) ↔ (a = -1)) :=
by
  sorry

end NUMINAMATH_GPT_no_common_points_eq_l854_85462


namespace NUMINAMATH_GPT_number_increase_when_reversed_l854_85401

theorem number_increase_when_reversed :
  let n := 253
  let reversed_n := 352
  reversed_n - n = 99 :=
by
  let n := 253
  let reversed_n := 352
  sorry

end NUMINAMATH_GPT_number_increase_when_reversed_l854_85401


namespace NUMINAMATH_GPT_students_in_two_courses_l854_85422

def total_students := 400
def num_math_modelling := 169
def num_chinese_literacy := 158
def num_international_perspective := 145
def num_all_three := 30
def num_none := 20

theorem students_in_two_courses : 
  ∃ x y z, 
    (num_math_modelling + num_chinese_literacy + num_international_perspective - (x + y + z) + num_all_three + num_none = total_students) ∧
    (x + y + z = 32) := 
  by
  sorry

end NUMINAMATH_GPT_students_in_two_courses_l854_85422


namespace NUMINAMATH_GPT_stock_price_end_of_second_year_l854_85457

noncomputable def initial_price : ℝ := 120
noncomputable def price_after_first_year (initial_price : ℝ) : ℝ := initial_price * 2
noncomputable def price_after_second_year (price_after_first_year : ℝ) : ℝ := price_after_first_year * 0.7

theorem stock_price_end_of_second_year : 
  price_after_second_year (price_after_first_year initial_price) = 168 := 
by 
  sorry

end NUMINAMATH_GPT_stock_price_end_of_second_year_l854_85457


namespace NUMINAMATH_GPT_min_value_quadratic_l854_85468

noncomputable def quadratic_expr (x : ℝ) : ℝ :=
  x^2 - 4 * x - 2019

theorem min_value_quadratic :
  ∀ x : ℝ, quadratic_expr x ≥ -2023 :=
by
  sorry

end NUMINAMATH_GPT_min_value_quadratic_l854_85468


namespace NUMINAMATH_GPT_problem_a_l854_85448

theorem problem_a (f : ℕ → ℕ) (h1 : f 1 = 2) (h2 : ∀ n, f (f n) = f n + 3 * n) : f 26 = 59 := 
sorry

end NUMINAMATH_GPT_problem_a_l854_85448


namespace NUMINAMATH_GPT_find_circle_center_l854_85404

noncomputable def circle_center_lemma (a b : ℝ) : Prop :=
  -- Condition: Circle passes through (1, 0)
  (a - 1)^2 + b^2 = (a - 1)^2 + (b - 0)^2 ∧
  -- Condition: Circle is tangent to the parabola y = x^2 at (1, 1)
  (a - 1)^2 + (b - 1)^2 = 0

theorem find_circle_center : ∃ a b : ℝ, circle_center_lemma a b ∧ a = 1 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_circle_center_l854_85404


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l854_85485

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem monotonic_increasing_interval :
  ∃ a b : ℝ, a < b ∧
    ∀ x y : ℝ, (a < x ∧ x < b) → (a < y ∧ y < b) → x < y → f x < f y ∧ a = -Real.pi / 6 ∧ b = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l854_85485


namespace NUMINAMATH_GPT_tan_ratio_l854_85472

theorem tan_ratio (α β : ℝ) 
  (h1 : Real.sin (α + β) = (Real.sqrt 3) / 2) 
  (h2 : Real.sin (α - β) = (Real.sqrt 2) / 2) : 
  (Real.tan α) / (Real.tan β) = (5 + 2 * Real.sqrt 6) / (5 - 2 * Real.sqrt 6) :=
by
  sorry

end NUMINAMATH_GPT_tan_ratio_l854_85472


namespace NUMINAMATH_GPT_promotional_event_probabilities_l854_85412

def P_A := 1 / 1000
def P_B := 1 / 100
def P_C := 1 / 20
def P_A_B_C := P_A + P_B + P_C
def P_A_B := P_A + P_B
def P_complement_A_B := 1 - P_A_B

theorem promotional_event_probabilities :
  P_A = 1 / 1000 ∧
  P_B = 1 / 100 ∧
  P_C = 1 / 20 ∧
  P_A_B_C = 61 / 1000 ∧
  P_complement_A_B = 989 / 1000 :=
by
  sorry

end NUMINAMATH_GPT_promotional_event_probabilities_l854_85412


namespace NUMINAMATH_GPT_expected_value_is_10_l854_85429

noncomputable def expected_value_adjacent_pairs (boys girls : ℕ) (total_people : ℕ) : ℕ :=
  if total_people = 20 ∧ boys = 8 ∧ girls = 12 then 10 else sorry

theorem expected_value_is_10 : expected_value_adjacent_pairs 8 12 20 = 10 :=
by
  -- Intuition and all necessary calculations (proof steps) have already been explained.
  -- Here we are directly stating the conclusion based on given problem conditions.
  trivial

end NUMINAMATH_GPT_expected_value_is_10_l854_85429


namespace NUMINAMATH_GPT_value_of_p_l854_85433

theorem value_of_p (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 = 3 * x2 ∧ x^2 - (3 * p - 2) * x + p^2 - 1 = 0) →
  (p = 2 ∨ p = 14 / 11) :=
by
  sorry

end NUMINAMATH_GPT_value_of_p_l854_85433


namespace NUMINAMATH_GPT_set_intersection_complement_l854_85418

open Set

noncomputable def A : Set ℝ := { x | abs (x - 1) > 2 }
noncomputable def B : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }
noncomputable def notA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
noncomputable def targetSet : Set ℝ := { x | 2 < x ∧ x ≤ 3 }

theorem set_intersection_complement :
  (notA ∩ B) = targetSet :=
  by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l854_85418


namespace NUMINAMATH_GPT_term_omit_perfect_squares_300_l854_85445

theorem term_omit_perfect_squares_300 (n : ℕ) (hn : n = 300) : 
  ∃ k : ℕ, k = 317 ∧ (∀ m : ℕ, (m < k → m * m ≠ k)) := 
sorry

end NUMINAMATH_GPT_term_omit_perfect_squares_300_l854_85445


namespace NUMINAMATH_GPT_xiaoming_bus_time_l854_85473

-- Definitions derived from the conditions:
def total_time : ℕ := 40
def transfer_time : ℕ := 6
def subway_time : ℕ := 30
def bus_time : ℕ := 50

-- Theorem statement to prove the bus travel time equals 10 minutes
theorem xiaoming_bus_time : (total_time - transfer_time = 34) ∧ (subway_time = 30 ∧ bus_time = 50) → 
  ∃ (T_bus : ℕ), T_bus = 10 := by
  sorry

end NUMINAMATH_GPT_xiaoming_bus_time_l854_85473


namespace NUMINAMATH_GPT_find_number_l854_85402

theorem find_number (x : ℝ) (h : x / 4 + 15 = 4 * x - 15) : x = 8 :=
sorry

end NUMINAMATH_GPT_find_number_l854_85402


namespace NUMINAMATH_GPT_f_val_at_100_l854_85442

theorem f_val_at_100 (f : ℝ → ℝ) (h₀ : ∀ x, f x * f (x + 3) = 12) (h₁ : f 1 = 4) : f 100 = 3 :=
sorry

end NUMINAMATH_GPT_f_val_at_100_l854_85442


namespace NUMINAMATH_GPT_sum_of_solutions_sum_of_possible_values_l854_85409

theorem sum_of_solutions (y : ℝ) (h : y^2 = 81) : y = 9 ∨ y = -9 :=
sorry

theorem sum_of_possible_values (y : ℝ) (h : y^2 = 81) : (∀ x, x = 9 ∨ x = -9 → x = 9 ∨ x = -9 → x = 9 + (-9)) :=
by
  have y_sol : y = 9 ∨ y = -9 := sum_of_solutions y h
  sorry

end NUMINAMATH_GPT_sum_of_solutions_sum_of_possible_values_l854_85409


namespace NUMINAMATH_GPT_inequality_for_positive_reals_l854_85438

theorem inequality_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (b * (a + b)) + 1 / (c * (b + c)) + 1 / (a * (c + a)) ≥ 27 / (2 * (a + b + c)^2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_positive_reals_l854_85438


namespace NUMINAMATH_GPT_wood_needed_l854_85491

variable (total_needed : ℕ) (friend_pieces : ℕ) (brother_pieces : ℕ)

/-- Alvin's total needed wood is 376 pieces, he got 123 from his friend and 136 from his brother.
    Prove that Alvin needs 117 more pieces. -/
theorem wood_needed (h1 : total_needed = 376) (h2 : friend_pieces = 123) (h3 : brother_pieces = 136) :
  total_needed - (friend_pieces + brother_pieces) = 117 := by
  sorry

end NUMINAMATH_GPT_wood_needed_l854_85491


namespace NUMINAMATH_GPT_bottles_count_l854_85495

-- Defining the conditions from the problem statement
def condition1 (x y : ℕ) : Prop := 3 * x + 4 * y = 108
def condition2 (x y : ℕ) : Prop := 2 * x + 3 * y = 76

-- The proof statement combining conditions and the solution
theorem bottles_count (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 20 ∧ y = 12 :=
sorry

end NUMINAMATH_GPT_bottles_count_l854_85495


namespace NUMINAMATH_GPT_find_x_l854_85435

theorem find_x
  (x : ℕ)
  (h1 : x % 7 = 0)
  (h2 : x > 0)
  (h3 : x^2 > 144)
  (h4 : x < 25) : x = 14 := 
  sorry

end NUMINAMATH_GPT_find_x_l854_85435


namespace NUMINAMATH_GPT_simplify_expression_l854_85400

theorem simplify_expression (x : ℝ) (h : x^2 + 2 * x - 6 = 0) : 
  ((x - 1) / (x - 3) - (x + 1) / x) / ((x^2 + 3 * x) / (x^2 - 6 * x + 9)) = -1/2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l854_85400


namespace NUMINAMATH_GPT_car_passing_time_l854_85476

open Real

theorem car_passing_time
  (vX : ℝ) (lX : ℝ)
  (vY : ℝ) (lY : ℝ)
  (t : ℝ)
  (h_vX : vX = 90)
  (h_lX : lX = 5)
  (h_vY : vY = 91)
  (h_lY : lY = 6)
  :
  (t * (vY - vX) / 3600) = 0.011 → t = 39.6 := 
by
  sorry

end NUMINAMATH_GPT_car_passing_time_l854_85476


namespace NUMINAMATH_GPT_infinite_geometric_series_common_ratio_l854_85423

theorem infinite_geometric_series_common_ratio
  (a S : ℝ)
  (h₁ : a = 500)
  (h₂ : S = 4000)
  (h₃ : S = a / (1 - (r : ℝ))) :
  r = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_common_ratio_l854_85423


namespace NUMINAMATH_GPT_ratio_of_octagon_areas_l854_85483

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_octagon_areas_l854_85483


namespace NUMINAMATH_GPT_weight_of_daughter_l854_85460

theorem weight_of_daughter 
  (M D C : ℝ)
  (h1 : M + D + C = 120)
  (h2 : D + C = 60)
  (h3 : C = (1 / 5) * M)
  : D = 48 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_daughter_l854_85460


namespace NUMINAMATH_GPT_max_value_log_function_l854_85454

theorem max_value_log_function (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1/2) :
  ∃ u : ℝ, (u = Real.logb (1/2) (8*x*y + 4*y^2 + 1)) ∧ (u ≤ 0) :=
sorry

end NUMINAMATH_GPT_max_value_log_function_l854_85454


namespace NUMINAMATH_GPT_Pyarelal_loss_share_l854_85480

-- Define the conditions
variables (P : ℝ) (A : ℝ) (total_loss : ℝ)

-- Ashok's capital is 1/9 of Pyarelal's capital
axiom Ashok_capital : A = (1 / 9) * P

-- Total loss is Rs 900
axiom total_loss_val : total_loss = 900

-- Prove Pyarelal's share of the loss is Rs 810
theorem Pyarelal_loss_share : (P / (A + P)) * total_loss = 810 :=
by
  sorry

end NUMINAMATH_GPT_Pyarelal_loss_share_l854_85480


namespace NUMINAMATH_GPT_centroid_of_triangle_l854_85474

theorem centroid_of_triangle :
  let A := (2, 8)
  let B := (6, 2)
  let C := (0, 4)
  let centroid := ( (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3 )
  centroid = (8 / 3, 14 / 3) := 
by
  sorry

end NUMINAMATH_GPT_centroid_of_triangle_l854_85474


namespace NUMINAMATH_GPT_number_of_boys_is_810_l854_85469

theorem number_of_boys_is_810 (B G : ℕ) (h1 : B + G = 900) (h2 : G = B / 900 * 100) : B = 810 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_is_810_l854_85469
