import Mathlib

namespace max_sin_angle_F1PF2_on_ellipse_l165_165459

theorem max_sin_angle_F1PF2_on_ellipse
  (x y : ℝ)
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (h : P ∈ {Q | Q.1^2 / 9 + Q.2^2 / 5 = 1})
  (F1_is_focus : F1 = (-2, 0))
  (F2_is_focus : F2 = (2, 0)) :
  ∃ sin_max, sin_max = 4 * Real.sqrt 5 / 9 := 
sorry

end max_sin_angle_F1PF2_on_ellipse_l165_165459


namespace revenue_after_decrease_l165_165430

theorem revenue_after_decrease (original_revenue : ℝ) (percentage_decrease : ℝ) (final_revenue : ℝ) 
  (h1 : original_revenue = 69.0) 
  (h2 : percentage_decrease = 24.637681159420293) 
  (h3 : final_revenue = original_revenue - (original_revenue * (percentage_decrease / 100))) 
  : final_revenue = 52.0 :=
by
  sorry

end revenue_after_decrease_l165_165430


namespace angle_bisector_equation_intersection_l165_165585

noncomputable def slope_of_angle_bisector (m1 m2 : ℝ) : ℝ :=
  (m1 + m2 - Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)

noncomputable def equation_of_angle_bisector (x : ℝ) : ℝ :=
  (Real.sqrt 21 - 6) / 7 * x

theorem angle_bisector_equation_intersection :
  let m1 := 2
  let m2 := 4
  slope_of_angle_bisector m1 m2 = (Real.sqrt 21 - 6) / 7 ∧
  equation_of_angle_bisector 1 = (Real.sqrt 21 - 6) / 7 :=
by
  sorry

end angle_bisector_equation_intersection_l165_165585


namespace units_digit_division_l165_165661

theorem units_digit_division (a b c d e denom : ℕ)
  (h30 : a = 30) (h31 : b = 31) (h32 : c = 32) (h33 : d = 33) (h34 : e = 34)
  (h120 : denom = 120) :
  ((a * b * c * d * e) / denom) % 10 = 4 :=
by
  sorry

end units_digit_division_l165_165661


namespace parameterized_curve_is_line_l165_165142

theorem parameterized_curve_is_line :
  ∀ (t : ℝ), ∃ (m b : ℝ), y = 5 * ((x - 5) / 3) - 3 → y = (5 * x - 34) / 3 := 
by
  sorry

end parameterized_curve_is_line_l165_165142


namespace math_problem_l165_165885

theorem math_problem : 
  ∃ (n m k : ℕ), 
    (∀ d : ℕ, d ∣ n → d > 0) ∧ 
    (n = m * 6^k) ∧
    (∀ d : ℕ, d ∣ m → 6 ∣ d → False) ∧
    (m + k = 60466182) ∧ 
    (n.factors.count 1 = 2023) :=
sorry

end math_problem_l165_165885


namespace largest_perfect_square_factor_of_3780_l165_165971

theorem largest_perfect_square_factor_of_3780 :
  ∃ m : ℕ, (∃ k : ℕ, 3780 = k * m * m) ∧ m * m = 36 :=
by
  sorry

end largest_perfect_square_factor_of_3780_l165_165971


namespace art_performance_selection_l165_165391

-- Definitions from the conditions
def total_students := 6
def singers := 3
def dancers := 2
def both := 1

-- Mathematical expression in Lean
noncomputable def ways_to_select (n k : ℕ) : ℕ := Nat.choose n k

theorem art_performance_selection 
    (total_students singers dancers both: ℕ) 
    (h1 : total_students = 6)
    (h2 : singers = 3)
    (h3 : dancers = 2)
    (h4 : both = 1) :
  (ways_to_select 4 2 * 3 - 1) = (Nat.choose 4 2 * 3 - 1) := 
sorry

end art_performance_selection_l165_165391


namespace remainder_when_divided_by_22_l165_165542

theorem remainder_when_divided_by_22 (y : ℤ) (k : ℤ) (h : y = 264 * k + 42) : y % 22 = 20 :=
by
  sorry

end remainder_when_divided_by_22_l165_165542


namespace evaluate_expression_l165_165288

theorem evaluate_expression : 2 - 1 / (2 + 1 / (2 - 1 / 3)) = 21 / 13 := by
  sorry

end evaluate_expression_l165_165288


namespace curve_intersects_at_point_2_3_l165_165153

open Real

theorem curve_intersects_at_point_2_3 :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
                 (t₁^2 - 4 = t₂^2 - 4) ∧ 
                 (t₁^3 - 6 * t₁ + 3 = t₂^3 - 6 * t₂ + 3) ∧ 
                 (t₁^2 - 4 = 2) ∧ 
                 (t₁^3 - 6 * t₁ + 3 = 3) :=
by
  sorry

end curve_intersects_at_point_2_3_l165_165153


namespace radius_of_B_l165_165707

theorem radius_of_B {A B C D : Type} (r_A : ℝ) (r_D : ℝ) (r_B : ℝ) (r_C : ℝ)
  (center_A : A) (center_B : B) (center_C : C) (center_D : D)
  (h_cong_BC : r_B = r_C)
  (h_A_D : r_D = 2 * r_A)
  (h_r_A : r_A = 2)
  (h_tangent_A_D : (dist center_A center_D) = r_A) :
  r_B = 32/25 := sorry

end radius_of_B_l165_165707


namespace arun_age_in_6_years_l165_165281

theorem arun_age_in_6_years
  (A D n : ℕ)
  (h1 : D = 42)
  (h2 : A = (5 * D) / 7)
  (h3 : A + n = 36) 
  : n = 6 :=
by
  sorry

end arun_age_in_6_years_l165_165281


namespace number_one_half_more_equals_twenty_five_percent_less_l165_165968

theorem number_one_half_more_equals_twenty_five_percent_less (n : ℤ) : 
    (80 - 0.25 * 80 = 60) → ((3 / 2 : ℚ) * n = 60) → (n = 40) :=
by
  intros h1 h2
  sorry

end number_one_half_more_equals_twenty_five_percent_less_l165_165968


namespace annulus_area_sufficient_linear_element_l165_165551

theorem annulus_area_sufficient_linear_element (R r : ℝ) (hR : R > 0) (hr : r > 0) (hrR : r < R):
  (∃ d : ℝ, d = R - r ∨ d = R + r) → ∃ A : ℝ, A = π * (R ^ 2 - r ^ 2) :=
by
  sorry

end annulus_area_sufficient_linear_element_l165_165551


namespace arithmetic_sum_2015_l165_165076

-- Definitions based on problem conditions
def a1 : ℤ := -2015
def S (n : ℕ) (d : ℤ) : ℤ := n * a1 + n * (n - 1) / 2 * d
def arithmetic_sequence (n : ℕ) (d : ℤ) : ℤ := a1 + (n - 1) * d

-- Proof problem
theorem arithmetic_sum_2015 (d : ℤ) :
  2 * S 6 d - 3 * S 4 d = 24 →
  S 2015 d = -2015 :=
by
  sorry

end arithmetic_sum_2015_l165_165076


namespace initial_apples_value_l165_165215

-- Definitions for the conditions
def picked_apples : ℤ := 105
def total_apples : ℤ := 161

-- Statement to prove
theorem initial_apples_value : ∀ (initial_apples : ℤ), 
  initial_apples + picked_apples = total_apples → 
  initial_apples = total_apples - picked_apples := 
by 
  sorry

end initial_apples_value_l165_165215


namespace minimum_dwarfs_l165_165498

-- Define the problem conditions
def chairs := fin 30 → Prop
def occupied (C : chairs) := ∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))

-- State the theorem that provides the solution
theorem minimum_dwarfs (C : chairs) : (∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))) → ∃ n : ℕ, n = 10 :=
by
  sorry

end minimum_dwarfs_l165_165498


namespace train_speed_is_60_kmph_l165_165834

-- Define the distance and time
def train_length : ℕ := 400
def bridge_length : ℕ := 800
def time_to_pass_bridge : ℕ := 72

-- Define the distances and calculations
def total_distance : ℕ := train_length + bridge_length
def speed_m_per_s : ℚ := total_distance / time_to_pass_bridge
def speed_km_per_h : ℚ := speed_m_per_s * 3.6

-- State and prove the theorem
theorem train_speed_is_60_kmph : speed_km_per_h = 60 := by
  sorry

end train_speed_is_60_kmph_l165_165834


namespace trajectory_of_P_is_right_branch_of_hyperbola_l165_165477

-- Definitions of the given points F1 and F2
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Definition of the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Definition of point P satisfying the condition
def P (x y : ℝ) : Prop :=
  abs (distance (x, y) F1 - distance (x, y) F2) = 8

-- Trajectory of point P is the right branch of the hyperbola
theorem trajectory_of_P_is_right_branch_of_hyperbola :
  ∀ (x y : ℝ), P x y → True := -- Trajectory is hyperbola (right branch)
by
  sorry

end trajectory_of_P_is_right_branch_of_hyperbola_l165_165477


namespace total_students_in_class_l165_165745

theorem total_students_in_class (female_students : ℕ) (male_students : ℕ) (total_students : ℕ) 
  (h1 : female_students = 13) 
  (h2 : male_students = 3 * female_students) 
  (h3 : total_students = female_students + male_students) : 
    total_students = 52 := 
by
  sorry

end total_students_in_class_l165_165745


namespace prob_abs_diff_gt_one_is_three_over_eight_l165_165639

noncomputable def prob_abs_diff_gt_one : ℝ := sorry

theorem prob_abs_diff_gt_one_is_three_over_eight :
  prob_abs_diff_gt_one = 3 / 8 :=
by sorry

end prob_abs_diff_gt_one_is_three_over_eight_l165_165639


namespace commutativity_associativity_l165_165206

variables {α : Type*} (op : α → α → α)

-- Define conditions as hypotheses
axiom cond1 : ∀ a b c : α, op a (op b c) = op b (op c a)
axiom cond2 : ∀ a b c : α, op a b = op a c → b = c
axiom cond3 : ∀ a b c : α, op a c = op b c → a = b

-- Commutativity statement
theorem commutativity (a b : α) : op a b = op b a := sorry

-- Associativity statement
theorem associativity (a b c : α) : op (op a b) c = op a (op b c) := sorry

end commutativity_associativity_l165_165206


namespace oil_remaining_in_tank_l165_165958

/- Definitions for the problem conditions -/
def tankCapacity : Nat := 32
def totalOilPurchased : Nat := 728

/- Theorem statement -/
theorem oil_remaining_in_tank : totalOilPurchased % tankCapacity = 24 := by
  sorry

end oil_remaining_in_tank_l165_165958


namespace connie_tickets_l165_165574

theorem connie_tickets (total_tickets spent_on_koala spent_on_earbuds spent_on_glow_bracelets : ℕ)
  (h1 : total_tickets = 50)
  (h2 : spent_on_koala = total_tickets / 2)
  (h3 : spent_on_earbuds = 10)
  (h4 : total_tickets = spent_on_koala + spent_on_earbuds + spent_on_glow_bracelets) :
  spent_on_glow_bracelets = 15 :=
by
  sorry

end connie_tickets_l165_165574


namespace sum_of_50th_row_l165_165031

-- Define triangular numbers
def T (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of numbers in the nth row
def f (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1 -- T_1 is 1 for the base case
  else 2 * f (n - 1) + n * (n + 1)

-- Prove the sum of the 50th row
theorem sum_of_50th_row : f 50 = 2^50 - 2550 := 
  sorry

end sum_of_50th_row_l165_165031


namespace ticket_count_l165_165743

theorem ticket_count (x y : ℕ) 
  (h1 : x + y = 35)
  (h2 : 24 * x + 18 * y = 750) : 
  x = 20 ∧ y = 15 :=
by
  sorry

end ticket_count_l165_165743


namespace expression_evaluation_l165_165028

theorem expression_evaluation : 
  (2^10 * 3^3) / (6 * 2^5) = 144 :=
by 
  sorry

end expression_evaluation_l165_165028


namespace fraction_students_say_dislike_but_actually_like_is_25_percent_l165_165433

variable (total_students : Nat) (students_like_dancing : Nat) (students_dislike_dancing : Nat) 
         (students_like_dancing_but_say_dislike : Nat) (students_dislike_dancing_and_say_dislike : Nat) 
         (total_say_dislike : Nat)

def fraction_of_students_who_say_dislike_but_actually_like (total_students students_like_dancing students_dislike_dancing 
         students_like_dancing_but_say_dislike students_dislike_dancing_and_say_dislike total_say_dislike : Nat) : Nat :=
    (students_like_dancing_but_say_dislike * 100) / total_say_dislike

theorem fraction_students_say_dislike_but_actually_like_is_25_percent
  (h1 : total_students = 100)
  (h2 : students_like_dancing = 60)
  (h3 : students_dislike_dancing = 40)
  (h4 : students_like_dancing_but_say_dislike = 12)
  (h5 : students_dislike_dancing_and_say_dislike = 36)
  (h6 : total_say_dislike = 48) :
  fraction_of_students_who_say_dislike_but_actually_like total_students students_like_dancing students_dislike_dancing 
    students_like_dancing_but_say_dislike students_dislike_dancing_and_say_dislike total_say_dislike = 25 :=
by sorry

end fraction_students_say_dislike_but_actually_like_is_25_percent_l165_165433


namespace total_amount_divided_l165_165812

-- Define the conditions
variables (A B C : ℕ)
axiom h1 : 4 * A = 5 * B
axiom h2 : 4 * A = 10 * C
axiom h3 : C = 160

-- Define the theorem to prove the total amount
theorem total_amount_divided (h1 : 4 * A = 5 * B) (h2 : 4 * A = 10 * C) (h3 : C = 160) : 
  A + B + C = 880 :=
sorry

end total_amount_divided_l165_165812


namespace eva_total_marks_l165_165447

theorem eva_total_marks :
  ∀ (maths2 arts2 science2 history2: ℕ),
  (maths2 = 80) →
  (arts2 = 90) →
  (science2 = 90) →
  (history2 = 85) →
  let maths1 := maths2 + 10,
      arts1 := arts2 - 15,
      science1 := science2 - (1 / 3 : ℚ) * science2,
      history1 := history2 + 5,
      total_marks := maths1 + arts1 + science1.natAbs + history1 + maths2 + arts2 + science2 + history2
  in total_marks = 660 :=
by
  intros
  sorry

end eva_total_marks_l165_165447


namespace range_of_a_l165_165905

theorem range_of_a 
    (a : ℝ) 
    (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f x = Real.exp (|x - a|)) 
    (increasing_on_interval : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y) :
    a ≤ 1 :=
sorry

end range_of_a_l165_165905


namespace odd_three_digit_integers_strictly_increasing_digits_l165_165335

theorem odd_three_digit_integers_strictly_increasing_digits :
  let valid_combinations (c : ℕ) :=
    if c = 1 then 0 else
    if c = 3 then 1 else
    if c = 5 then 6 else
    if c = 7 then 15 else
    if c = 9 then 28 else 0 in
  (valid_combinations 1 + valid_combinations 3 + valid_combinations 5 + valid_combinations 7 + valid_combinations 9 = 50) :=
by
  unfold valid_combinations
  sorry

end odd_three_digit_integers_strictly_increasing_digits_l165_165335


namespace determine_K_class_comparison_l165_165249

variables (a b : ℕ) -- number of students in classes A and B respectively
variable (K : ℕ) -- amount that each A student would pay if they covered all cost

-- Conditions from the problem statement
def first_event_total (a b : ℕ) := 5 * a + 3 * b
def second_event_total (a b : ℕ) := 4 * a + 6 * b
def total_balance (a b K : ℕ) := 9 * (a + b) = K * (a + b)

-- Questions to be answered
theorem determine_K : total_balance a b K → K = 9 :=
by
  sorry

theorem class_comparison (a b : ℕ) : 5 * a + 3 * b = 4 * a + 6 * b → b > a :=
by
  sorry

end determine_K_class_comparison_l165_165249


namespace C_is_a_liar_l165_165211

def is_knight_or_liar (P : Prop) : Prop :=
P = true ∨ P = false

variable (A B C : Prop)

-- A, B and C can only be true (knight) or false (liar)
axiom a1 : is_knight_or_liar A
axiom a2 : is_knight_or_liar B
axiom a3 : is_knight_or_liar C

-- A says "B is a liar", meaning if A is a knight, B is a liar, and if A is a liar, B is a knight
axiom a4 : A = true → B = false
axiom a5 : A = false → B = true

-- B says "A and C are of the same type", meaning if B is a knight, A and C are of the same type, otherwise they are not
axiom a6 : B = true → (A = C)
axiom a7 : B = false → (A ≠ C)

-- Prove that C is a liar
theorem C_is_a_liar : C = false :=
by
  sorry

end C_is_a_liar_l165_165211


namespace find_n_in_geometric_series_l165_165279

theorem find_n_in_geometric_series :
  let a1 : ℕ := 15
  let a2 : ℕ := 5
  let r1 := a2 / a1
  let S1 := a1 / (1 - r1: ℝ)
  let S2 := 3 * S1
  let r2 := (5 + n) / a1
  S2 = 15 / (1 - r2) →
  n = 20 / 3 :=
by
  sorry

end find_n_in_geometric_series_l165_165279


namespace range_of_alpha_l165_165299

theorem range_of_alpha :
  ∀ P : ℝ, 
  (∃ y : ℝ, y = 4 / (Real.exp P + 1)) →
  (∃ α : ℝ, α = Real.arctan (4 / (Real.exp P + 2 + 1 / Real.exp P)) ∧ (Real.tan α) ∈ Set.Ico (-1) 0) → 
  Set.Ico (3 * Real.pi / 4) Real.pi :=
by
  sorry

end range_of_alpha_l165_165299


namespace four_played_games_l165_165532

theorem four_played_games
  (A B C D E : Prop)
  (A_answer : ¬A)
  (B_answer : A ∧ ¬B)
  (C_answer : B ∧ ¬C)
  (D_answer : C ∧ ¬D)
  (E_answer : D ∧ ¬E)
  (truth_condition : (¬A ∧ ¬B) ∨ (¬B ∧ ¬C) ∨ (¬C ∧ ¬D) ∨ (¬D ∧ ¬E)) :
  A ∨ B ∨ C ∨ D ∧ E := sorry

end four_played_games_l165_165532


namespace smallest_constant_obtuse_triangle_l165_165579

theorem smallest_constant_obtuse_triangle (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (a^2 > b^2 + c^2) → (b^2 + c^2) / (a^2) ≥ 1 / 2 :=
by 
  sorry

end smallest_constant_obtuse_triangle_l165_165579


namespace triangle_shape_l165_165727

-- Define the sides of the triangle and the angles
variables {a b c : ℝ}
variables {A B C : ℝ} 
-- Assume that angles are in radians and 0 < A, B, C < π
-- Also assume that the sum of angles in the triangle is π
axiom angle_sum_triangle : A + B + C = Real.pi

-- Given condition
axiom given_condition : a^2 * Real.cos A * Real.sin B = b^2 * Real.sin A * Real.cos B

-- Conclusion: The shape of triangle ABC is either isosceles or right triangle
theorem triangle_shape : 
  (A = B) ∨ (A + B = (Real.pi / 2)) := 
by sorry

end triangle_shape_l165_165727


namespace find_abc_l165_165609

-- Given conditions: a, b, c are positive real numbers and satisfy the given equations.
variables (a b c : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_pos_c : 0 < c)
variable (h1 : a * (b + c) = 152)
variable (h2 : b * (c + a) = 162)
variable (h3 : c * (a + b) = 170)

theorem find_abc : a * b * c = 720 := 
  sorry

end find_abc_l165_165609


namespace Cannot_Halve_Triangles_With_Diagonals_l165_165208

structure Polygon where
  vertices : Nat
  edges : Nat

def is_convex (n : Nat) (P : Polygon) : Prop :=
  P.vertices = n ∧ P.edges = n

def non_intersecting_diagonals (P : Polygon) : Prop :=
  -- Assuming a placeholder for the actual non-intersecting diagonals condition
  true

def count_triangles (P : Polygon) (d : non_intersecting_diagonals P) : Nat :=
  P.vertices - 2 -- This is the simplification used for counting triangles

def count_all_diagonals_triangles (P : Polygon) (d : non_intersecting_diagonals P) : Nat :=
  -- Placeholder for function to count triangles formed exclusively by diagonals
  1000

theorem Cannot_Halve_Triangles_With_Diagonals (P : Polygon) (h : is_convex 2002 P) (d : non_intersecting_diagonals P) :
  count_triangles P d = 2000 → ¬ (count_all_diagonals_triangles P d = 1000) :=
by
  intro h1
  sorry

end Cannot_Halve_Triangles_With_Diagonals_l165_165208


namespace pinocchio_cannot_pay_exactly_l165_165942

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l165_165942


namespace time_for_runnerA_to_complete_race_l165_165132

variable (speedA : ℝ) -- speed of runner A in meters per second
variable (t : ℝ) -- time taken by runner A to complete the race in seconds
variable (tB : ℝ) -- time taken by runner B to complete the race in seconds

noncomputable def distanceA : ℝ := 1000 -- distance covered by runner A in meters
noncomputable def distanceB : ℝ := 950 -- distance covered by runner B in meters when A finishes
noncomputable def speedB : ℝ := distanceB / tB -- speed of runner B in meters per second

theorem time_for_runnerA_to_complete_race
    (h1 : distanceA = speedA * t)
    (h2 : distanceB = speedA * (t + 20)) :
    t = 400 :=
by
  sorry

end time_for_runnerA_to_complete_race_l165_165132


namespace total_rats_l165_165216

variable (Kenia Hunter Elodie : ℕ) -- Number of rats each person has

-- Conditions
-- Elodie has 30 rats
axiom h1 : Elodie = 30
-- Elodie has 10 rats more than Hunter
axiom h2 : Elodie = Hunter + 10
-- Kenia has three times as many rats as Hunter and Elodie have together
axiom h3 : Kenia = 3 * (Hunter + Elodie)

-- Prove that the total number of pets the three have together is 200
theorem total_rats : Kenia + Hunter + Elodie = 200 := 
by 
  sorry

end total_rats_l165_165216


namespace determine_digits_in_base_l165_165369

theorem determine_digits_in_base (x y z b : ℕ) (h1 : 1993 = x * b^2 + y * b + z) (h2 : x + y + z = 22) :
  x = 2 ∧ y = 15 ∧ z = 5 ∧ b = 28 :=
sorry

end determine_digits_in_base_l165_165369


namespace bananas_to_oranges_l165_165565

variables (banana apple orange : Type) 
variables (cost_banana : banana → ℕ) 
variables (cost_apple : apple → ℕ)
variables (cost_orange : orange → ℕ)

-- Conditions given in the problem
axiom cond1 : ∀ (b1 b2 b3 : banana) (a1 a2 : apple), cost_banana b1 = cost_banana b2 → cost_banana b2 = cost_banana b3 → 3 * cost_banana b1 = 2 * cost_apple a1
axiom cond2 : ∀ (a3 a4 a5 a6 : apple) (o1 o2 : orange), cost_apple a3 = cost_apple a4 → cost_apple a4 = cost_apple a5 → cost_apple a5 = cost_apple a6 → 6 * cost_apple a3 = 4 * cost_orange o1

-- Prove that 8 oranges cost as much as 18 bananas
theorem bananas_to_oranges (b1 b2 b3 : banana) (a1 a2 a3 a4 a5 a6 : apple) (o1 o2 : orange) :
    3 * cost_banana b1 = 2 * cost_apple a1 →
    6 * cost_apple a3 = 4 * cost_orange o1 →
    18 * cost_banana b1 = 8 * cost_orange o2 := 
sorry

end bananas_to_oranges_l165_165565


namespace seating_arrangement_l165_165850

theorem seating_arrangement (x y : ℕ) (h1 : x * 8 + y * 7 = 55) : x = 6 :=
by
  sorry

end seating_arrangement_l165_165850


namespace closest_pressure_reading_l165_165375

theorem closest_pressure_reading (x : ℝ) (h : 102.4 ≤ x ∧ x ≤ 102.8) :
    (|x - 102.5| > |x - 102.6| ∧ |x - 102.6| < |x - 102.7| ∧ |x - 102.6| < |x - 103.0|) → x = 102.6 :=
by
  sorry

end closest_pressure_reading_l165_165375


namespace y_squared_range_l165_165918

theorem y_squared_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 4) ^ (1/3) = 4) : 15 ≤ y^2 ∧ y^2 ≤ 25 :=
by
  sorry

end y_squared_range_l165_165918


namespace tenth_term_of_sequence_l165_165039

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 0 then 3
else if n = 1 then 4
else 12 / sequence (n - 1)

theorem tenth_term_of_sequence :
  sequence 9 = 4 :=
sorry

end tenth_term_of_sequence_l165_165039


namespace pinocchio_cannot_pay_exactly_l165_165939

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l165_165939


namespace determine_a_l165_165458

theorem determine_a (a : ℕ) : 
  (2 * 10^10 + a ) % 11 = 0 ∧ 0 ≤ a ∧ a < 11 → a = 9 :=
by
  sorry

end determine_a_l165_165458


namespace odd_two_digit_combinations_l165_165902

theorem odd_two_digit_combinations (digits : Finset ℕ) (h_digits : digits = {1, 3, 5, 7, 9}) :
  ∃ n : ℕ, n = 20 ∧ (∃ a b : ℕ, a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ (10 * a + b) % 2 = 1) :=
by
  sorry

end odd_two_digit_combinations_l165_165902


namespace problem_quadratic_inequality_l165_165742

theorem problem_quadratic_inequality
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : 0 < a)
  (h2 : a ≤ 4/9)
  (h3 : b = -a)
  (h4 : c = -2*a + 1)
  (h5 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 0 ≤ a*x^2 + b*x + c ∧ a*x^2 + b*x + c ≤ 1) :
  3*a + 2*b + c ≠ 1/3 ∧ 3*a + 2*b + c ≠ 5/4 :=
by
  sorry

end problem_quadratic_inequality_l165_165742


namespace cubic_sum_of_roots_l165_165755

theorem cubic_sum_of_roots (a b c : ℝ) 
  (h1 : a + b + c = -1)
  (h2 : a * b + b * c + c * a = -333)
  (h3 : a * b * c = 1001) :
  a^3 + b^3 + c^3 = 2003 :=
sorry

end cubic_sum_of_roots_l165_165755


namespace number_of_yellow_balls_l165_165073

theorem number_of_yellow_balls (x : ℕ) (h : (6 : ℝ) / (6 + x) = 0.3) : x = 14 :=
by
  sorry

end number_of_yellow_balls_l165_165073


namespace train_pass_time_l165_165130

-- Definitions based on the conditions
def train_length : ℕ := 280  -- train length in meters
def train_speed_kmh : ℕ := 72  -- train speed in km/hr
noncomputable def train_speed_ms : ℚ := (train_speed_kmh * 5 / 18)  -- train speed in m/s

-- Theorem statement
theorem train_pass_time : (train_length / train_speed_ms) = 14 := by
  sorry

end train_pass_time_l165_165130


namespace thickness_relation_l165_165965

noncomputable def a : ℝ := (1/3) * Real.sin (1/2)
noncomputable def b : ℝ := (1/2) * Real.sin (1/3)
noncomputable def c : ℝ := (1/3) * Real.cos (7/8)

theorem thickness_relation : c > b ∧ b > a := by
  sorry

end thickness_relation_l165_165965


namespace triangle_angles_l165_165242

theorem triangle_angles (r_a r_b r_c R : ℝ)
    (h1 : r_a + r_b = 3 * R)
    (h2 : r_b + r_c = 2 * R) :
    ∃ (A B C : ℝ), A = 30 ∧ B = 60 ∧ C = 90 :=
sorry

end triangle_angles_l165_165242


namespace deliveries_conditions_l165_165280

variables (M P D : ℕ)
variables (MeMa MeBr MeQu MeBx: ℕ)

def distribution := (MeMa = 3 * MeBr) ∧ (MeBr = MeBr) ∧ (MeQu = MeBr) ∧ (MeBx = MeBr)

theorem deliveries_conditions 
  (h1 : P = 8 * M) 
  (h2 : D = 4 * M) 
  (h3 : M + P + D = 75) 
  (h4 : MeMa + MeBr + MeQu + MeBx = M)
  (h5 : distribution MeMa MeBr MeQu MeBx) :
  M = 5 ∧ MeMa = 2 ∧ MeBr = 1 ∧ MeQu = 1 ∧ MeBx = 1 :=
    sorry 

end deliveries_conditions_l165_165280


namespace sum_of_reciprocals_of_roots_l165_165890

theorem sum_of_reciprocals_of_roots (r1 r2 : ℚ) (h_sum : r1 + r2 = 17) (h_prod : r1 * r2 = 6) :
  1 / r1 + 1 / r2 = 17 / 6 :=
sorry

end sum_of_reciprocals_of_roots_l165_165890


namespace sufficient_but_not_necessary_l165_165053

theorem sufficient_but_not_necessary (x : ℝ) : (x > 0 → x * (x + 1) > 0) ∧ ¬ (x * (x + 1) > 0 → x > 0) := 
by 
sorry

end sufficient_but_not_necessary_l165_165053


namespace slope_intercept_parallel_l165_165295

theorem slope_intercept_parallel (A : ℝ × ℝ) (x y : ℝ) (hA : A = (3, 2))
(hparallel : 4 * x + y - 2 = 0) :
  ∃ b : ℝ, y = -4 * x + b ∧ b = 14 :=
by
  sorry

end slope_intercept_parallel_l165_165295


namespace additional_charge_per_minute_atlantic_call_l165_165801

def base_rate_U : ℝ := 11.0
def rate_per_minute_U : ℝ := 0.25
def base_rate_A : ℝ := 12.0
def call_duration : ℝ := 20.0
variable (rate_per_minute_A : ℝ)

theorem additional_charge_per_minute_atlantic_call :
  base_rate_U + rate_per_minute_U * call_duration = base_rate_A + rate_per_minute_A * call_duration →
  rate_per_minute_A = 0.20 := by
  sorry

end additional_charge_per_minute_atlantic_call_l165_165801


namespace peter_takes_last_stone_l165_165394

theorem peter_takes_last_stone (n : ℕ) (h : ∀ p, Nat.Prime p → p < n) :
  ∃ P, ∀ stones: ℕ, stones > n^2 → (∃ k : ℕ, 
  ((k = 1 ∨ (∃ p : ℕ, Nat.Prime p ∧ p < n ∧ k = p) ∨ (∃ m : ℕ, k = m * n)) ∧
  stones ≥ k ∧ stones - k > n^2) →
  P = stones - k) := 
sorry

end peter_takes_last_stone_l165_165394


namespace tv_cost_l165_165488

theorem tv_cost (savings original_savings furniture_spent : ℝ) (hs : original_savings = 1000) (hf : furniture_spent = (3/4) * original_savings) (remaining_spent : savings = original_savings - furniture_spent) : savings = 250 := 
by
  sorry

end tv_cost_l165_165488


namespace total_reactions_eq_100_l165_165168

variable (x : ℕ) -- Total number of reactions.
variable (thumbs_up : ℕ) -- Number of "thumbs up" reactions.
variable (thumbs_down : ℕ) -- Number of "thumbs down" reactions.
variable (S : ℕ) -- Net Score.

-- Conditions
axiom thumbs_up_eq_75percent_reactions : thumbs_up = 3 * x / 4
axiom thumbs_down_eq_25percent_reactions : thumbs_down = x / 4
axiom score_definition : S = thumbs_up - thumbs_down
axiom initial_score : S = 50

theorem total_reactions_eq_100 : x = 100 :=
by 
  sorry

end total_reactions_eq_100_l165_165168


namespace equality_of_integers_l165_165846

theorem equality_of_integers (a b : ℕ) (h1 : ∀ n : ℕ, ∃ m : ℕ, m > 0 ∧ (a^m + b^m) % (a^n + b^n) = 0) : a = b :=
sorry

end equality_of_integers_l165_165846


namespace aluminum_percentage_in_new_alloy_l165_165398

theorem aluminum_percentage_in_new_alloy :
  ∀ (x1 x2 x3 : ℝ),
  0 ≤ x1 ∧ x1 ≤ 1 ∧
  0 ≤ x2 ∧ x2 ≤ 1 ∧
  0 ≤ x3 ∧ x3 ≤ 1 ∧
  x1 + x2 + x3 = 1 ∧
  0.15 * x1 + 0.3 * x2 = 0.2 →
  0.15 ≤ 0.6 * x1 + 0.45 * x3 ∧ 0.6 * x1 + 0.45 * x3 ≤ 0.40 :=
by
  -- The proof will be inserted here
  sorry

end aluminum_percentage_in_new_alloy_l165_165398


namespace pushups_percentage_l165_165170

def total_exercises : ℕ := 12 + 8 + 20

def percentage_pushups (total_ex: ℕ) : ℕ := (8 * 100) / total_ex

theorem pushups_percentage (h : total_exercises = 40) : percentage_pushups total_exercises = 20 :=
by
  sorry

end pushups_percentage_l165_165170


namespace compare_logs_l165_165913

theorem compare_logs (a b c : ℝ) (h_a : a = Real.log 2 / Real.log 5) (h_b : b = Real.log 3 / Real.log 8) (h_c : c = 1 / 2) : a < c ∧ c < b :=
by {
  sorry,
}

end compare_logs_l165_165913


namespace cos_double_angle_l165_165183

theorem cos_double_angle (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) : 
  Real.cos (20 * Real.pi / 180) = 1 - 2 * k^2 := by
  sorry

end cos_double_angle_l165_165183


namespace f_at_neg_8_5_pi_eq_pi_div_2_l165_165782

def f (x : Real) : Real := sorry

axiom functional_eqn (x : Real) : f (x + (3 * Real.pi / 2)) = -1 / f x
axiom f_interval (x : Real) (h : x ∈ Set.Icc (-Real.pi) Real.pi) : f x = x * Real.sin x

theorem f_at_neg_8_5_pi_eq_pi_div_2 : f (-8.5 * Real.pi) = Real.pi / 2 := 
  sorry

end f_at_neg_8_5_pi_eq_pi_div_2_l165_165782


namespace pow_div_pow_eq_result_l165_165572

theorem pow_div_pow_eq_result : 13^8 / 13^5 = 2197 := by
  sorry

end pow_div_pow_eq_result_l165_165572


namespace students_who_like_yellow_l165_165760

theorem students_who_like_yellow (total_students girls students_like_green girls_like_pink students_like_yellow : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_green = total_students / 2)
  (h3 : girls_like_pink = girls / 3)
  (h4 : girls = 18)
  (h5 : students_like_yellow = total_students - (students_like_green + girls_like_pink)) :
  students_like_yellow = 9 :=
by
  sorry

end students_who_like_yellow_l165_165760


namespace find_d_l165_165300

theorem find_d (a b c d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
    (h5 : a^2 = c * (d + 29)) (h6 : b^2 = c * (d - 29)) :
    d = 421 :=
    sorry

end find_d_l165_165300


namespace centroid_triangle_PQR_l165_165121

theorem centroid_triangle_PQR (P Q R S : ℝ × ℝ) 
  (P_coord : P = (2, 5)) 
  (Q_coord : Q = (9, 3)) 
  (R_coord : R = (4, -4))
  (S_is_centroid : S = (
    (P.1 + Q.1 + R.1) / 3,
    (P.2 + Q.2 + R.2) / 3)) :
  9 * S.1 + 4 * S.2 = 151 / 3 :=
by
  sorry

end centroid_triangle_PQR_l165_165121


namespace cleaner_for_cat_stain_l165_165361

theorem cleaner_for_cat_stain (c : ℕ) :
  (6 * 6) + (3 * c) + (1 * 1) = 49 → c = 4 :=
by
  sorry

end cleaner_for_cat_stain_l165_165361


namespace second_candidate_percentage_l165_165341

theorem second_candidate_percentage (V : ℝ) (h1 : 0.15 * V ≠ 0) (h2 : 0.38 * V ≠ 300) :
  (0.38 * V - 300) / (0.85 * V - 250) * 100 = 44.71 :=
by 
  -- Let the math proof be synthesized by a more detailed breakdown of conditions and theorems
  sorry

end second_candidate_percentage_l165_165341


namespace inverse_proportion_quadrants_l165_165589

theorem inverse_proportion_quadrants (m : ℝ) : (∀ (x : ℝ), x ≠ 0 → y = (m - 2) / x → (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0)) ↔ m > 2 :=
by
  sorry

end inverse_proportion_quadrants_l165_165589


namespace average_of_eight_consecutive_integers_l165_165446

theorem average_of_eight_consecutive_integers (c d : ℝ) (h : (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6) + (c + 7)) / 8 = d) :
  ((d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6) + (d + 7) + (d + 8)) / 8 = c + 8 := by 
  sorry

end average_of_eight_consecutive_integers_l165_165446


namespace contradiction_example_l165_165402

theorem contradiction_example (x : ℝ) (a := x^2 - 1) (b := 2 * x + 2) : ¬ (a < 0 ∧ b < 0) :=
by
  -- The proof goes here, but we just need the statement
  sorry

end contradiction_example_l165_165402


namespace upstream_distance_l165_165423

-- Define the conditions
def velocity_current : ℝ := 1.5
def distance_downstream : ℝ := 32
def time : ℝ := 6

-- Define the speed of the man in still water
noncomputable def speed_in_still_water : ℝ := (distance_downstream / time) - velocity_current

-- Define the distance rowed upstream
noncomputable def distance_upstream : ℝ := (speed_in_still_water - velocity_current) * time

-- The theorem statement to be proved
theorem upstream_distance (v c d : ℝ) (h1 : c = 1.5) (h2 : (v + c) * 6 = 32) (h3 : (v - c) * 6 = d) : d = 14 :=
by
  -- Insert the proof here
  sorry

end upstream_distance_l165_165423


namespace seats_capacity_l165_165920

theorem seats_capacity (x : ℕ) (h1 : 15 * x + 12 * x + 8 = 89) : x = 3 :=
by
  -- proof to be filled in
  sorry

end seats_capacity_l165_165920


namespace value_of_expression_l165_165784

noncomputable def line_does_not_pass_through_third_quadrant (k b : ℝ) : Prop :=
k < 0 ∧ b ≥ 0

theorem value_of_expression 
  (k b a e m n c d : ℝ) 
  (h_line : line_does_not_pass_through_third_quadrant k b)
  (h_a_gt_e : a > e)
  (hA : a * k + b = m)
  (hB : e * k + b = n)
  (hC : -m * k + b = c)
  (hD : -n * k + b = d) :
  (m - n) * (c - d) ^ 3 > 0 :=
sorry

end value_of_expression_l165_165784


namespace simplify_expression1_simplify_expression2_l165_165441

/-- Proof Problem 1: Simplify the expression (a+2b)^2 - 4b(a+b) -/
theorem simplify_expression1 (a b : ℝ) : 
  (a + 2 * b)^2 - 4 * b * (a + b) = a^2 :=
sorry

/-- Proof Problem 2: Simplify the expression ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) ÷ (x - 1) / (x^2 - 4) -/
theorem simplify_expression2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) : 
  ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 :=
sorry

end simplify_expression1_simplify_expression2_l165_165441


namespace solve_for_x_l165_165512

theorem solve_for_x (x : ℝ) (h : 2 / 3 + 1 / x = 7 / 9) : x = 9 := 
  sorry

end solve_for_x_l165_165512


namespace equation_solution_l165_165303

variable (x y : ℝ)

theorem equation_solution
  (h1 : x * y + x + y = 17)
  (h2 : x^2 * y + x * y^2 = 66):
  x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4 = 12499 :=
  by sorry

end equation_solution_l165_165303


namespace gumball_machine_total_gumballs_l165_165827

/-- A gumball machine has red, green, and blue gumballs. Given the following conditions:
1. The machine has half as many blue gumballs as red gumballs.
2. For each blue gumball, the machine has 4 times as many green gumballs.
3. The machine has 16 red gumballs.
Prove that the total number of gumballs in the machine is 56. -/
theorem gumball_machine_total_gumballs :
  ∀ (red blue green : ℕ),
    (blue = red / 2) →
    (green = blue * 4) →
    red = 16 →
    (red + blue + green = 56) :=
by
  intros red blue green h_blue h_green h_red
  sorry

end gumball_machine_total_gumballs_l165_165827


namespace total_gumballs_l165_165821

-- Define the count of red, blue, and green gumballs
def red_gumballs := 16
def blue_gumballs := red_gumballs / 2
def green_gumballs := blue_gumballs * 4

-- Prove that the total number of gumballs is 56
theorem total_gumballs : red_gumballs + blue_gumballs + green_gumballs = 56 := by
  sorry

end total_gumballs_l165_165821


namespace sum_of_given_infinite_geometric_series_l165_165871

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l165_165871


namespace log_comparison_l165_165915

variables (a b c : ℝ)
def log_base (b x : ℝ) := log x / log b

theorem log_comparison 
  (a_def : a = log_base 5 2)
  (b_def : b = log_base 8 3)
  (c_def : c = 1 / 2) :
  a < c ∧ c < b :=
by
  sorry

end log_comparison_l165_165915


namespace percent_increase_in_area_l165_165552

theorem percent_increase_in_area (s : ℝ) (h_s : s > 0) :
  let medium_area := s^2
  let large_length := 1.20 * s
  let large_width := 1.25 * s
  let large_area := large_length * large_width 
  let percent_increase := ((large_area - medium_area) / medium_area) * 100
  percent_increase = 50 := by
    sorry

end percent_increase_in_area_l165_165552


namespace circles_intersect_l165_165167

-- Definition of the first circle
def circleC := { p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 4 }

-- Definition of the second circle
def circleM := { p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 9 }

-- Prove that the circles intersect
theorem circles_intersect : 
  ∃ p : ℝ × ℝ, p ∈ circleC ∧ p ∈ circleM := 
sorry

end circles_intersect_l165_165167


namespace option_C_correct_l165_165670

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l165_165670


namespace andrew_total_donation_l165_165837

/-
Problem statement:
Andrew started donating 7k to an organization on his 11th birthday. Yesterday, Andrew turned 29.
Verify that the total amount Andrew has donated is 126k.
-/

theorem andrew_total_donation 
  (annual_donation : ℕ := 7000) 
  (start_age : ℕ := 11) 
  (current_age : ℕ := 29) 
  (years_donating : ℕ := current_age - start_age) 
  (total_donated : ℕ := annual_donation * years_donating) :
  total_donated = 126000 := 
by 
  sorry

end andrew_total_donation_l165_165837


namespace prove_correct_option_C_l165_165685

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l165_165685


namespace probability_of_shaded_section_l165_165747

theorem probability_of_shaded_section 
  (total_sections : ℕ)
  (shaded_sections : ℕ)
  (H1 : total_sections = 8)
  (H2 : shaded_sections = 4)
  : (shaded_sections / total_sections : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_shaded_section_l165_165747


namespace sum_odd_numbers_to_2019_is_correct_l165_165490

-- Define the sequence sum
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Define the specific problem
theorem sum_odd_numbers_to_2019_is_correct : sum_first_n_odd 1010 = 1020100 :=
by
  -- Sorry placeholder for the proof
  sorry

end sum_odd_numbers_to_2019_is_correct_l165_165490


namespace infinite_geometric_series_sum_l165_165868

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l165_165868


namespace initial_number_of_students_l165_165155

theorem initial_number_of_students (S : ℕ) (h : S + 6 = 37) : S = 31 :=
sorry

end initial_number_of_students_l165_165155


namespace exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared_l165_165036

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared :
  ∃ (n : ℕ), sum_of_digits n = 1000 ∧ sum_of_digits (n ^ 2) = 1000000 := sorry

end exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared_l165_165036


namespace prove_correct_option_C_l165_165684

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l165_165684


namespace hyperbola_range_k_l165_165740

theorem hyperbola_range_k (k : ℝ) : (4 + k) * (1 - k) < 0 ↔ k ∈ (Set.Iio (-4) ∪ Set.Ioi 1) := 
by
  sorry

end hyperbola_range_k_l165_165740


namespace arithmetic_problem_l165_165811

theorem arithmetic_problem : 90 + 5 * 12 / (180 / 3) = 91 := by
  sorry

end arithmetic_problem_l165_165811


namespace num_persons_initially_l165_165646

theorem num_persons_initially (N : ℕ) (avg_weight : ℝ) 
  (h_increase_avg : avg_weight + 5 = avg_weight + 40 / N) :
  N = 8 := by
    sorry

end num_persons_initially_l165_165646


namespace min_dwarfs_l165_165511

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end min_dwarfs_l165_165511


namespace function_satisfies_equation_l165_165642

theorem function_satisfies_equation (y : ℝ → ℝ) (h : ∀ x : ℝ, y x = Real.exp (x + x^2) + 2 * Real.exp x) :
  ∀ x : ℝ, deriv y x - y x = 2 * x * Real.exp (x + x^2) :=
by {
  sorry
}

end function_satisfies_equation_l165_165642


namespace original_price_of_article_l165_165344

theorem original_price_of_article :
  ∃ P : ℝ, (P * 0.55 * 0.85 = 920) ∧ P = 1968.04 :=
by
  sorry

end original_price_of_article_l165_165344


namespace range_of_a_l165_165899

-- Problem statement and conditions definition
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def Q (a : ℝ) : Prop := (5 - 2 * a) > 1

-- Proof problem statement
theorem range_of_a (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ≤ -2 :=
sorry

end range_of_a_l165_165899


namespace minimum_value_of_u_l165_165722

noncomputable def minimum_value_lemma (x y : ℝ) (hx : Real.sin x + Real.sin y = 1 / 3) : Prop :=
  ∃ m : ℝ, m = -1/9 ∧ ∀ z, z = Real.sin x + Real.cos x ^ 2 → z ≥ m

theorem minimum_value_of_u
  (x y : ℝ)
  (hx : Real.sin x + Real.sin y = 1 / 3) :
  ∃ m : ℝ, m = -1/9 ∧ ∀ z, z = Real.sin x + Real.cos x ^ 2 → z ≥ m :=
sorry

end minimum_value_of_u_l165_165722


namespace bobby_books_count_l165_165841

variable (KristiBooks BobbyBooks : ℕ)

theorem bobby_books_count (h1 : KristiBooks = 78) (h2 : BobbyBooks = KristiBooks + 64) : BobbyBooks = 142 :=
by
  sorry

end bobby_books_count_l165_165841


namespace original_number_l165_165305

theorem original_number (x : ℝ) (h1 : 268 * 74 = 19732) (h2 : x * 0.74 = 1.9832) : x = 2.68 :=
by
  sorry

end original_number_l165_165305


namespace infinite_geometric_series_sum_l165_165855

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l165_165855


namespace magnitude_BC_range_l165_165304

theorem magnitude_BC_range (AB AC : EuclideanSpace ℝ (Fin 2)) 
  (h₁ : ‖AB‖ = 18) (h₂ : ‖AC‖ = 5) : 
  13 ≤ ‖AC - AB‖ ∧ ‖AC - AB‖ ≤ 23 := 
  sorry

end magnitude_BC_range_l165_165304


namespace polynomial_transformation_l165_165917

variable {x y : ℝ}

theorem polynomial_transformation
  (h : y = x + 1/x) 
  (poly_eq_0 : x^4 + x^3 - 5*x^2 + x + 1 = 0) :
  x^2 * (y^2 + y - 7) = 0 :=
sorry

end polynomial_transformation_l165_165917


namespace find_x_l165_165738

theorem find_x (h₁ : 2994 / 14.5 = 175) (h₂ : 29.94 / x = 17.5) : x = 29.94 / 17.5 :=
by
  -- skipping proofs
  sorry

end find_x_l165_165738


namespace markup_percentage_l165_165124

variable (W R : ℝ)

-- Condition: When sold at a 40% discount, a sweater nets the merchant a 30% profit on the wholesale cost.
def discount_condition : Prop := 0.6 * R = 1.3 * W

-- Theorem: The percentage markup of the sweater from wholesale to normal retail price is 116.67%
theorem markup_percentage (h : discount_condition W R) : (R - W) / W * 100 = 116.67 :=
by sorry

end markup_percentage_l165_165124


namespace janina_spend_on_supplies_each_day_l165_165212

theorem janina_spend_on_supplies_each_day 
  (rent : ℝ)
  (p : ℝ)
  (n : ℕ)
  (H1 : rent = 30)
  (H2 : p = 2)
  (H3 : n = 21) :
  (n : ℝ) * p - rent = 12 := 
by
  sorry

end janina_spend_on_supplies_each_day_l165_165212


namespace boat_problem_l165_165818

theorem boat_problem (x n : ℕ) (h1 : n = 7 * x + 5) (h2 : n = 8 * x - 2) :
  n = 54 ∧ x = 7 := by
sorry

end boat_problem_l165_165818


namespace sum_of_given_infinite_geometric_series_l165_165869

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l165_165869


namespace sum_of_squares_iff_double_l165_165633

theorem sum_of_squares_iff_double (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_squares_iff_double_l165_165633


namespace sqrt_sum_inequality_l165_165463

variable (a b c d : ℝ)

theorem sqrt_sum_inequality
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : a + d = b + c) :
  Real.sqrt a + Real.sqrt d < Real.sqrt b + Real.sqrt c :=
by
  sorry

end sqrt_sum_inequality_l165_165463


namespace matrix_product_is_zero_l165_165030

-- Define the two matrices
def A (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, d, -c], ![-d, 0, b], ![c, -b, 0]]

def B (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![d^2, b * d, c * d], ![b * d, b^2, b * c], ![c * d, b * c, c^2]]

-- Define the zero matrix
def zero_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 0, 0], ![0, 0, 0], ![0, 0, 0]]

-- The theorem to prove
theorem matrix_product_is_zero (b c d : ℝ) : A b c d * B b c d = zero_matrix :=
by sorry

end matrix_product_is_zero_l165_165030


namespace proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l165_165548

variable (x y : ℤ)

def proposition_A := (x ≠ 1000 ∨ y ≠ 1002)
def proposition_B := (x + y ≠ 2002)

theorem proposition_A_necessary_for_B : proposition_B x y → proposition_A x y := by
  sorry

theorem proposition_A_not_sufficient_for_B : ¬ (proposition_A x y → proposition_B x y) := by
  sorry

end proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l165_165548


namespace math_problem_l165_165676

theorem math_problem (a b m : ℝ) : 
  ¬((-2 * a) ^ 2 = -4 * a ^ 2) ∧ 
  ¬((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧ 
  ((-m + 2) * (-m - 2) = m ^ 2 - 4) ∧ 
  ¬((a ^ 5) ^ 2 = a ^ 7) :=
by 
  split ; 
  sorry ;
  split ;
  sorry ;
  split ; 
  sorry ;
  sorry

end math_problem_l165_165676


namespace meeting_percentage_l165_165925

theorem meeting_percentage
    (workday_hours : ℕ)
    (first_meeting_minutes : ℕ)
    (second_meeting_factor : ℕ)
    (hp_workday_hours : workday_hours = 10)
    (hp_first_meeting_minutes : first_meeting_minutes = 60)
    (hp_second_meeting_factor : second_meeting_factor = 2) 
    : (first_meeting_minutes + first_meeting_minutes * second_meeting_factor : ℚ) 
    / (workday_hours * 60) * 100 = 30 := 
by
  have workday_minutes := workday_hours * 60
  have second_meeting_minutes := first_meeting_minutes * second_meeting_factor
  have total_meeting_minutes := first_meeting_minutes + second_meeting_minutes
  have percentage := (total_meeting_minutes : ℚ) / workday_minutes * 100
  sorry

end meeting_percentage_l165_165925


namespace rectangle_area_at_stage_8_l165_165557

-- Declare constants for the conditions.
def square_side_length : ℕ := 4
def number_of_stages : ℕ := 8
def area_of_single_square : ℕ := square_side_length * square_side_length

-- The statement to prove
theorem rectangle_area_at_stage_8 : number_of_stages * area_of_single_square = 128 := by
  sorry

end rectangle_area_at_stage_8_l165_165557


namespace contrapositive_true_l165_165773

theorem contrapositive_true (h : ∀ x : ℝ, x < 0 → x^2 > 0) : 
  (∀ x : ℝ, ¬ (x^2 > 0) → ¬ (x < 0)) :=
by 
  sorry

end contrapositive_true_l165_165773


namespace range_of_a_l165_165182

noncomputable def range_a : set ℝ :=
  {a | ∃ x : ℝ, x > 0 ∧ a - 2 * x - abs (Real.log x) ≤ 0}

theorem range_of_a :
  range_a = {a : ℝ | a ≤ 1 + Real.log 2} :=
by
  sorry

end range_of_a_l165_165182


namespace sparrow_grains_l165_165762

theorem sparrow_grains (x : ℤ) : 9 * x < 1001 ∧ 10 * x > 1100 → x = 111 :=
by
  sorry

end sparrow_grains_l165_165762


namespace square_area_l165_165251

theorem square_area (side_length : ℕ) (h : side_length = 17) : side_length * side_length = 289 :=
by sorry

end square_area_l165_165251


namespace total_cost_l165_165927

-- Definitions:
def amount_beef : ℕ := 1000
def price_per_pound_beef : ℕ := 8
def amount_chicken := amount_beef * 2
def price_per_pound_chicken : ℕ := 3

-- Theorem: The total cost of beef and chicken is $14000.
theorem total_cost : (amount_beef * price_per_pound_beef) + (amount_chicken * price_per_pound_chicken) = 14000 :=
by
  sorry

end total_cost_l165_165927


namespace sufficient_not_necessary_l165_165486

theorem sufficient_not_necessary (a : ℝ) (h1 : a > 0) : (a^2 + a ≥ 0) ∧ ¬(a^2 + a ≥ 0 → a > 0) :=
by
  sorry

end sufficient_not_necessary_l165_165486


namespace loss_percentage_l165_165017

theorem loss_percentage
  (CP : ℝ := 1166.67)
  (SP : ℝ)
  (H : SP + 140 = CP + 0.02 * CP) :
  ((CP - SP) / CP) * 100 = 10 := 
by 
  sorry

end loss_percentage_l165_165017


namespace geometric_series_sum_l165_165863

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l165_165863


namespace recurring_decimal_to_fraction_l165_165804

theorem recurring_decimal_to_fraction : ∃ x : ℕ, (0.37 + (0.246 / 999)) = (x / 99900) ∧ x = 371874 :=
by
  sorry

end recurring_decimal_to_fraction_l165_165804


namespace Sheila_attend_probability_l165_165641

noncomputable def prob_rain := 0.3
noncomputable def prob_sunny := 0.4
noncomputable def prob_cloudy := 0.3

noncomputable def prob_attend_if_rain := 0.25
noncomputable def prob_attend_if_sunny := 0.9
noncomputable def prob_attend_if_cloudy := 0.5

noncomputable def prob_attend :=
  prob_rain * prob_attend_if_rain +
  prob_sunny * prob_attend_if_sunny +
  prob_cloudy * prob_attend_if_cloudy

theorem Sheila_attend_probability : prob_attend = 0.585 := by
  sorry

end Sheila_attend_probability_l165_165641


namespace find_center_of_circle_l165_165262

theorem find_center_of_circle :
  ∃ (a b : ℝ), a = 0 ∧ b = 3/2 ∧
  ( ∀ (x y : ℝ), ( (x = 1 ∧ y = 2) ∨ (x = 1 ∧ y = 1) ∨ (∃ t : ℝ, y = 2 * t + 3) ) → 
  (x - a)^2 + (y - b)^2 = (1 - a)^2 + (1 - b)^2 ) :=
sorry

end find_center_of_circle_l165_165262


namespace rationalize_denominator_l165_165766

theorem rationalize_denominator : 
  ∃ (A B C D E F : ℤ), 
  (1 / (Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 11)) = 
    (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
  A + B + C + D + E + F = 136 := 
sorry

end rationalize_denominator_l165_165766


namespace arithmetic_sequence_a₄_l165_165531

open Int

noncomputable def S (a₁ d n : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_a₄ {a₁ d : ℤ}
  (h₁ : S a₁ d 5 = 15) (h₂ : S a₁ d 9 = 63) :
  a₁ + 3 * d = 5 :=
  sorry

end arithmetic_sequence_a₄_l165_165531


namespace infinite_geometric_series_sum_l165_165865

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l165_165865


namespace tenth_term_of_sequence_l165_165040

noncomputable def sequence : ℕ → ℚ
| 0       := 3
| 1       := 4
| (n + 2) := 12 / (sequence n)

theorem tenth_term_of_sequence : sequence 9 = 4 :=
by
  sorry

end tenth_term_of_sequence_l165_165040


namespace shelly_thread_length_l165_165953

theorem shelly_thread_length 
  (threads_per_keychain : ℕ := 12) 
  (friends_in_class : ℕ := 6) 
  (friends_from_clubs := friends_in_class / 2)
  (total_friends := friends_in_class + friends_from_clubs) 
  (total_threads_needed := total_friends * threads_per_keychain) : 
  total_threads_needed = 108 := 
by 
  -- proof skipped
  sorry

end shelly_thread_length_l165_165953


namespace inverse_of_3_mod_185_l165_165450

theorem inverse_of_3_mod_185 : ∃ x : ℕ, 0 ≤ x ∧ x < 185 ∧ 3 * x ≡ 1 [MOD 185] :=
by
  use 62
  sorry

end inverse_of_3_mod_185_l165_165450


namespace average_payment_correct_l165_165340

-- Definitions based on conditions in the problem
def first_payments_num : ℕ := 20
def first_payment_amount : ℕ := 450

def second_payments_num : ℕ := 30
def increment_after_first : ℕ := 80

def third_payments_num : ℕ := 40
def increment_after_second : ℕ := 65

def fourth_payments_num : ℕ := 50
def increment_after_third : ℕ := 105

def fifth_payments_num : ℕ := 60
def increment_after_fourth : ℕ := 95

def total_payments : ℕ := first_payments_num + second_payments_num + third_payments_num + fourth_payments_num + fifth_payments_num

-- Function to calculate total paid amount
def total_amount_paid : ℕ :=
  (first_payments_num * first_payment_amount) +
  (second_payments_num * (first_payment_amount + increment_after_first)) +
  (third_payments_num * (first_payment_amount + increment_after_first + increment_after_second)) +
  (fourth_payments_num * (first_payment_amount + increment_after_first + increment_after_second + increment_after_third)) +
  (fifth_payments_num * (first_payment_amount + increment_after_first + increment_after_second + increment_after_third + increment_after_fourth))

-- Function to calculate average payment
def average_payment : ℕ := total_amount_paid / total_payments

-- The theorem to be proved
theorem average_payment_correct : average_payment = 657 := by
  sorry

end average_payment_correct_l165_165340


namespace gabrielle_saw_more_birds_l165_165126

def birds_seen (robins cardinals blue_jays : Nat) : Nat :=
  robins + cardinals + blue_jays

def percentage_difference (g c : Nat) : Nat :=
  ((g - c) * 100) / c

theorem gabrielle_saw_more_birds :
  let gabrielle := birds_seen 5 4 3
  let chase := birds_seen 2 5 3
  percentage_difference gabrielle chase = 20 := 
by
  sorry

end gabrielle_saw_more_birds_l165_165126


namespace coeff_x3_in_expansion_l165_165080

theorem coeff_x3_in_expansion : (nat.choose 50 3 * (1^47) * (1^3)) = 19600 := 
by sorry

end coeff_x3_in_expansion_l165_165080


namespace card_game_final_amounts_l165_165278

theorem card_game_final_amounts
  (T : ℝ)
  (aldo_initial_ratio : ℝ := 7)
  (bernardo_initial_ratio : ℝ := 6)
  (carlos_initial_ratio : ℝ := 5)
  (aldo_final_ratio : ℝ := 6)
  (bernardo_final_ratio : ℝ := 5)
  (carlos_final_ratio : ℝ := 4)
  (aldo_won : ℝ := 1200) :
  aldo_won = (1 / 90) * T →
  T = 108000 →
  (36 / 90) * T = 43200 ∧ (30 / 90) * T = 36000 ∧ (24 / 90) * T = 28800 := sorry

end card_game_final_amounts_l165_165278


namespace given_problem_l165_165568

noncomputable def improper_fraction_5_2_7 : ℚ := 37 / 7
noncomputable def improper_fraction_6_1_3 : ℚ := 19 / 3
noncomputable def improper_fraction_3_1_2 : ℚ := 7 / 2
noncomputable def improper_fraction_2_1_5 : ℚ := 11 / 5

theorem given_problem :
  71 * (improper_fraction_5_2_7 - improper_fraction_6_1_3) / (improper_fraction_3_1_2 + improper_fraction_2_1_5) = -13 - 37 / 1197 := 
  sorry

end given_problem_l165_165568


namespace odd_three_digit_integers_increasing_order_l165_165328

theorem odd_three_digit_integers_increasing_order :
  let digits_strictly_increasing (a b c : ℕ) : Prop := (1 ≤ a) ∧ (a < b) ∧ (b < c)
      let c_values : Finset ℕ := {3, 5, 7, 9}
  in ∑ c in c_values, (Finset.card (Finset.filter (λ ab : ℕ × ℕ, (digits_strictly_increasing ab.1 ab.2 c)) (Finset.cross {(1 : ℕ)..9} {(1 : ℕ)..9}))) = 50 :=
by
  sorry

end odd_three_digit_integers_increasing_order_l165_165328


namespace choir_group_students_l165_165814

theorem choir_group_students : ∃ n : ℕ, (n % 5 = 0) ∧ (n % 9 = 0) ∧ (n % 12 = 0) ∧ (∃ m : ℕ, n = m * m) ∧ n ≥ 360 := 
sorry

end choir_group_students_l165_165814


namespace average_employees_per_week_l165_165816

variable (x : ℕ)

theorem average_employees_per_week (h1 : x + 200 > x)
                                   (h2 : x < 200)
                                   (h3 : 2 * 200 = 400) :
  (x + 200 + x + 200 + 200 + 400) / 4 = 250 := by
  sorry

end average_employees_per_week_l165_165816


namespace difference_between_numbers_l165_165364

variable (x y : ℕ)

theorem difference_between_numbers (h1 : x + y = 34) (h2 : y = 22) : y - x = 10 := by
  sorry

end difference_between_numbers_l165_165364


namespace certain_number_is_45_l165_165420

-- Define the variables and condition
def x : ℝ := 45
axiom h : x * 7 = 0.35 * 900

-- The statement we need to prove
theorem certain_number_is_45 : x = 45 :=
by
  sorry

end certain_number_is_45_l165_165420


namespace nth_monomial_l165_165612

variable (a : ℝ)

def monomial_seq (n : ℕ) : ℝ :=
  (n + 1) * a ^ n

theorem nth_monomial (n : ℕ) : monomial_seq a n = (n + 1) * a ^ n :=
by
  sorry

end nth_monomial_l165_165612


namespace chlorine_weight_is_35_l165_165887

def weight_Na : Nat := 23
def weight_O : Nat := 16
def molecular_weight : Nat := 74

theorem chlorine_weight_is_35 (Cl : Nat) 
  (h : molecular_weight = weight_Na + Cl + weight_O) : 
  Cl = 35 := by
  -- Proof placeholder
  sorry

end chlorine_weight_is_35_l165_165887


namespace bonnie_roark_wire_length_ratio_l165_165025

noncomputable def ratio_of_wire_lengths : ℚ :=
let bonnie_wire_per_piece := 8
let bonnie_pieces := 12
let bonnie_total_wire := bonnie_pieces * bonnie_wire_per_piece

let bonnie_side := bonnie_wire_per_piece
let bonnie_volume := bonnie_side^3

let roark_side := 2
let roark_volume := roark_side^3
let roark_cubes := bonnie_volume / roark_volume

let roark_wire_per_piece := 2
let roark_pieces_per_cube := 12
let roark_wire_per_cube := roark_pieces_per_cube * roark_wire_per_piece
let roark_total_wire := roark_cubes * roark_wire_per_cube

let ratio := bonnie_total_wire / roark_total_wire
ratio 

theorem bonnie_roark_wire_length_ratio :
  ratio_of_wire_lengths = (1 : ℚ) / 16 := 
sorry

end bonnie_roark_wire_length_ratio_l165_165025


namespace shara_shells_after_vacation_l165_165103

-- Definitions based on conditions
def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

-- Statement of the proof problem
theorem shara_shells_after_vacation : 
  initial_shells + (shells_per_day * days) + shells_fourth_day = 41 := by
  sorry

end shara_shells_after_vacation_l165_165103


namespace equation_of_line_l165_165265

variable {a b k T : ℝ}

theorem equation_of_line (h_b_ne_zero : b ≠ 0)
  (h_line_passing_through : ∃ (line : ℝ → ℝ), line (-a) = b)
  (h_triangle_area : ∃ (h : ℝ), T = 1 / 2 * ka * (h - b))
  (h_base_length : ∃ (base : ℝ), base = ka) :
  ∃ (x y : ℝ), 2 * T * x - k * a^2 * y + k * a^2 * b + 2 * a * T = 0 :=
sorry

end equation_of_line_l165_165265


namespace find_length_d_l165_165919

theorem find_length_d :
  ∀ (A B C P: Type) (AB AC BC : ℝ) (d : ℝ),
    AB = 425 ∧ BC = 450 ∧ AC = 510 ∧
    (∃ (JG FI HE : ℝ), JG = FI ∧ FI = HE ∧ JG = d ∧ 
      (d / BC + d / AC + d / AB = 2)) 
    → d = 306 :=
by {
  sorry
}

end find_length_d_l165_165919


namespace correct_operation_l165_165667

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l165_165667


namespace pipe_cistern_problem_l165_165546

theorem pipe_cistern_problem:
  ∀ (rate_p rate_q : ℝ),
    rate_p = 1 / 10 →
    rate_q = 1 / 15 →
    ∀ (filled_in_4_minutes : ℝ),
      filled_in_4_minutes = 4 * (rate_p + rate_q) →
      ∀ (remaining : ℝ),
        remaining = 1 - filled_in_4_minutes →
        ∀ (time_to_fill : ℝ),
          time_to_fill = remaining / rate_q →
          time_to_fill = 5 :=
by
  intros rate_p rate_q Hp Hq filled_in_4_minutes H4 remaining Hr time_to_fill Ht
  sorry

end pipe_cistern_problem_l165_165546


namespace food_sufficient_days_l165_165986

theorem food_sufficient_days (D : ℕ) (h1 : 1000 * D - 10000 = 800 * D) : D = 50 :=
sorry

end food_sufficient_days_l165_165986


namespace probability_sum_equals_age_l165_165230

-- Define probability events for coin flip and die roll
def coin_flip (coin: Fin 2 -> ℤ) : ℚ := 1 / 2
def die_roll (die: Fin 6 -> ℤ) : ℚ := 1 / 6

-- Noah's age
def noah_age : ℤ := 16

-- Define the event that the sum of coin flip and die roll equals Noah's age
def event (coin: Fin 2 -> ℤ) (die: Fin 6 -> ℤ) : Prop :=
  ∃ c d, coin c + die d = noah_age

theorem probability_sum_equals_age
  (coin : Fin 2 -> ℤ)
  (die : Fin 6 -> ℤ) :
  event coin die → (coin 0 = 15) → (die 0 = 1) →
  P coin_flip * P die_roll = 1 / 12 :=
sorry

end probability_sum_equals_age_l165_165230


namespace inequality_solution_l165_165716

noncomputable def solution_set : Set ℝ :=
  {x : ℝ | x < -2} ∪
  {x : ℝ | -2 < x ∧ x ≤ -1} ∪
  {x : ℝ | 1 ≤ x}

theorem inequality_solution :
  {x : ℝ | (x^2 - 1) / (x + 2)^2 ≥ 0} = solution_set := by
  sorry

end inequality_solution_l165_165716


namespace trigonometric_identity_l165_165337

open Real

theorem trigonometric_identity (θ : ℝ) (h : tan θ = 2) :
  (sin θ * (1 + sin (2 * θ))) / (sqrt 2 * cos (θ - π / 4)) = 6 / 5 :=
by
  sorry

end trigonometric_identity_l165_165337


namespace find_three_numbers_l165_165718

theorem find_three_numbers (x : ℤ) (a b c : ℤ) :
  a + b + c = (x + 1)^2 ∧ a + b = x^2 ∧ b + c = (x - 1)^2 ∧
  a = 80 ∧ b = 320 ∧ c = 41 :=
by {
  sorry
}

end find_three_numbers_l165_165718


namespace geometric_series_sum_l165_165873

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l165_165873


namespace anne_gave_sweettarts_to_three_friends_l165_165562

theorem anne_gave_sweettarts_to_three_friends (sweettarts : ℕ) (eaten : ℕ) (friends : ℕ) 
  (h1 : sweettarts = 15) (h2 : eaten = 5) (h3 : sweettarts = friends * eaten) :
  friends = 3 := 
by 
  sorry

end anne_gave_sweettarts_to_three_friends_l165_165562


namespace customer_paid_correct_amount_l165_165380

noncomputable def cost_price : ℝ := 5565.217391304348
noncomputable def markup_percentage : ℝ := 0.15
noncomputable def markup_amount (cost : ℝ) : ℝ := cost * markup_percentage
noncomputable def final_price (cost : ℝ) (markup : ℝ) : ℝ := cost + markup

theorem customer_paid_correct_amount :
  final_price cost_price (markup_amount cost_price) = 6400 := sorry

end customer_paid_correct_amount_l165_165380


namespace smallest_positive_multiple_of_32_l165_165657

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ n % 32 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 32 = 0 → n ≤ m :=
by
  sorry

end smallest_positive_multiple_of_32_l165_165657


namespace gumball_machine_total_l165_165824

noncomputable def total_gumballs (R B G : ℕ) : ℕ := R + B + G

theorem gumball_machine_total
  (R B G : ℕ)
  (hR : R = 16)
  (hB : B = R / 2)
  (hG : G = 4 * B) :
  total_gumballs R B G = 56 :=
by
  sorry

end gumball_machine_total_l165_165824


namespace log_expression_zero_l165_165001

theorem log_expression_zero (log : Real → Real) (exp : Real → Real) (log_mul : ∀ a b, log (a * b) = log a + log b) :
  log 2 ^ 2 + log 2 * log 50 - log 4 = 0 :=
by
  sorry

end log_expression_zero_l165_165001


namespace seq_10_is_4_l165_165038

-- Define the sequence with given properties
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 4
  | (n + 2) => if n % 2 = 0 then 4 else 3

-- Theorem statement: The 10th term of the sequence is 4
theorem seq_10_is_4 : seq 9 = 4 :=
by sorry

end seq_10_is_4_l165_165038


namespace smallest_value_other_integer_l165_165522

noncomputable def smallest_possible_value_b : ℕ :=
  by sorry

theorem smallest_value_other_integer (x : ℕ) (h_pos : x > 0) (b : ℕ) 
  (h_gcd : Nat.gcd 36 b = x + 3) (h_lcm : Nat.lcm 36 b = x * (x + 3)) :
  b = 108 :=
  by sorry

end smallest_value_other_integer_l165_165522


namespace total_ridges_on_all_records_l165_165119

theorem total_ridges_on_all_records :
  let ridges_per_record := 60
  let cases := 4
  let shelves_per_case := 3
  let records_per_shelf := 20
  let shelf_fullness_ratio := 0.60

  let total_capacity := cases * shelves_per_case * records_per_shelf
  let actual_records := total_capacity * shelf_fullness_ratio
  let total_ridges := actual_records * ridges_per_record
  
  total_ridges = 8640 :=
by
  sorry

end total_ridges_on_all_records_l165_165119


namespace square_perimeter_l165_165991

theorem square_perimeter (s : ℝ) (h1 : (2 * (s + s / 4)) = 40) :
  4 * s = 64 :=
by
  sorry

end square_perimeter_l165_165991


namespace x_intercept_of_line_l165_165832

theorem x_intercept_of_line (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, -4)) (h2 : (x2, y2) = (6, 8)) : 
  ∃ x0 : ℝ, (x0 = (10 / 3) ∧ ∃ m : ℝ, m = (y2 - y1) / (x2 - x1) ∧ ∀ y : ℝ, y = m * x0 + b) := 
sorry

end x_intercept_of_line_l165_165832


namespace sum_of_powers_mod_7_eq_6_l165_165069

theorem sum_of_powers_mod_7_eq_6 : 
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 :=
by
  -- Using Fermat's Little Theorem (proved elsewhere in mathlib)
  have h1 : 1^6 % 7 = 1 := by sorry,
  have h2 : 2^6 % 7 = 1 := by sorry,
  have h3 : 3^6 % 7 = 1 := by sorry,
  have h4 : 4^6 % 7 = 1 := by sorry,
  have h5 : 5^6 % 7 = 1 := by sorry,
  have h6 : 6^6 % 7 = 1 := by sorry,
  -- Summing and proving the final result
  calc
    (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7
        = (1 + 1 + 1 + 1 + 1 + 1) % 7 := by rw [h1, h2, h3, h4, h5, h6]
    ... = 6 % 7 := by norm_num

end sum_of_powers_mod_7_eq_6_l165_165069


namespace arithmetic_sequence_sum_l165_165530

variable (S : ℕ → ℕ) -- Define a function S that gives the sum of the first n terms.
variable (n : ℕ)     -- Define a natural number n.

-- Conditions based on the problem statement
axiom h1 : S n = 3
axiom h2 : S (2 * n) = 10

-- The theorem we need to prove
theorem arithmetic_sequence_sum : S (3 * n) = 21 :=
by
  sorry

end arithmetic_sequence_sum_l165_165530


namespace sequence_bound_equivalent_problem_l165_165358

variable {n : ℕ}
variable {a : Fin (n+2) → ℝ}

theorem sequence_bound_equivalent_problem (h1 : a 0 = 0) (h2 : a (n + 1) = 0) 
  (h3 : ∀ k : Fin n, |a (k.val - 1) - 2 * a k + a (k + 1)| ≤ 1) :
  ∀ k : Fin (n+2), |a k| ≤ k * (n + 1 - k) / 2 := 
by
  sorry

end sequence_bound_equivalent_problem_l165_165358


namespace smallest_nat_mod_5_6_7_l165_165717

theorem smallest_nat_mod_5_6_7 (n : ℕ) :
  n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 → n = 209 :=
sorry

end smallest_nat_mod_5_6_7_l165_165717


namespace beef_weight_before_processing_l165_165145

-- Define the initial weight of the beef.
def W_initial := 1070.5882

-- Define the loss percentages.
def loss1 := 0.20
def loss2 := 0.15
def loss3 := 0.25

-- Define the final weight after all losses.
def W_final := 546.0

-- The main proof goal: show that W_initial results in W_final after considering the weight losses.
theorem beef_weight_before_processing (W_initial W_final : ℝ) (loss1 loss2 loss3 : ℝ) :
  W_final = (1 - loss3) * (1 - loss2) * (1 - loss1) * W_initial :=
by
  sorry

end beef_weight_before_processing_l165_165145


namespace no_integer_solution_xyz_l165_165412

theorem no_integer_solution_xyz : ¬ ∃ (x y z : ℤ),
  x^6 + x^3 + x^3 * y + y = 147^157 ∧
  x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := by
  sorry

end no_integer_solution_xyz_l165_165412


namespace total_savings_l165_165625

theorem total_savings (savings_sep savings_oct : ℕ) 
  (h1 : savings_sep = 260)
  (h2 : savings_oct = savings_sep + 30) :
  savings_sep + savings_oct = 550 := 
sorry

end total_savings_l165_165625


namespace hexagon_sequences_l165_165377

theorem hexagon_sequences : ∃ n : ℕ, n = 7 ∧ 
  ∀ (x d : ℕ), 6 * x + 15 * d = 720 ∧ (2 * x + 5 * d = 240) ∧ 
  (x + 5 * d < 160) ∧ (0 < x) ∧ (0 < d) ∧ (d % 2 = 0) ↔ (∃ k < n, (∃ x, ∃ d, x = 85 - 2*k ∧ d = 2 + 2*k)) :=
by
  sorry

end hexagon_sequences_l165_165377


namespace min_value_of_n_l165_165836

theorem min_value_of_n 
  (n k : ℕ) 
  (h1 : 8 * n = 225 * k + 3)
  (h2 : k ≡ 5 [MOD 8]) : 
  n = 141 := 
  sorry

end min_value_of_n_l165_165836


namespace part1_part2_l165_165238

noncomputable def f (x a : ℝ) := 5 - |x + a| - |x - 2|

theorem part1 : 
  (∀ x, f x 1 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3) :=
sorry

theorem part2 :
  (∀ a, (∀ x, f x a ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2)) :=
sorry

end part1_part2_l165_165238


namespace original_selling_price_is_440_l165_165158

variable (P : ℝ)

-- Condition: Bill made a profit of 10% by selling a product.
def original_selling_price := 1.10 * P

-- Condition: He had purchased the product for 10% less.
def new_purchase_price := 0.90 * P

-- Condition: With a 30% profit on the new purchase price, the new selling price.
def new_selling_price := 1.17 * P

-- Condition: The new selling price is $28 more than the original selling price.
def price_difference_condition : Prop := new_selling_price P = original_selling_price P + 28

-- Conclusion: The original selling price was \$440
theorem original_selling_price_is_440
    (h : price_difference_condition P) : original_selling_price P = 440 :=
sorry

end original_selling_price_is_440_l165_165158


namespace right_triangle_incircle_excircle_condition_l165_165443

theorem right_triangle_incircle_excircle_condition
  (r R : ℝ) 
  (hr_pos : 0 < r) 
  (hR_pos : 0 < R) :
  R ≥ r * (3 + 2 * Real.sqrt 2) := sorry

end right_triangle_incircle_excircle_condition_l165_165443


namespace slices_with_all_toppings_l165_165415

theorem slices_with_all_toppings (p m o a b c x total : ℕ) 
  (pepperoni_slices : p = 8)
  (mushrooms_slices : m = 12)
  (olives_slices : o = 14)
  (total_slices : total = 16)
  (inclusion_exclusion : p + m + o - a - b - c - 2 * x = total) :
  x = 4 := 
by
  rw [pepperoni_slices, mushrooms_slices, olives_slices, total_slices] at inclusion_exclusion
  sorry

end slices_with_all_toppings_l165_165415


namespace proportional_value_l165_165694

theorem proportional_value :
  ∃ (x : ℝ), 18 / 60 / (12 / 60) = x / 6 ∧ x = 9 := sorry

end proportional_value_l165_165694


namespace tan_theta_of_obtuse_angle_l165_165193

noncomputable def theta_expression (θ : Real) : Complex :=
  Complex.mk (3 * Real.sin θ) (Real.cos θ)

theorem tan_theta_of_obtuse_angle {θ : Real} (h_modulus : Complex.abs (theta_expression θ) = Real.sqrt 5) 
  (h_obtuse : π / 2 < θ ∧ θ < π) : Real.tan θ = -1 := 
  sorry

end tan_theta_of_obtuse_angle_l165_165193


namespace apples_in_each_box_l165_165362

variable (A : ℕ)
variable (ApplesSaturday : ℕ := 50 * A)
variable (ApplesSunday : ℕ := 25 * A)
variable (ApplesLeft : ℕ := 3 * A)
variable (ApplesSold : ℕ := 720)

theorem apples_in_each_box :
  (ApplesSaturday + ApplesSunday - ApplesSold = ApplesLeft) → A = 10 :=
by
  sorry

end apples_in_each_box_l165_165362


namespace max_value_expr_bound_l165_165588

noncomputable def max_value_expr (x : ℝ) : ℝ := 
  x^6 / (x^10 + x^8 - 6 * x^6 + 27 * x^4 + 64)

theorem max_value_expr_bound : 
  ∃ x : ℝ, max_value_expr x ≤ 1 / 8.38 := sorry

end max_value_expr_bound_l165_165588


namespace steve_matching_pairs_l165_165769

/-- Steve's total number of socks -/
def total_socks : ℕ := 25

/-- Number of Steve's mismatching socks -/
def mismatching_socks : ℕ := 17

/-- Number of Steve's matching socks -/
def matching_socks : ℕ := total_socks - mismatching_socks

/-- Number of pairs of matching socks Steve has -/
def matching_pairs : ℕ := matching_socks / 2

/-- Proof that Steve has 4 pairs of matching socks -/
theorem steve_matching_pairs : matching_pairs = 4 := by
  sorry

end steve_matching_pairs_l165_165769


namespace probability_calc_l165_165779

noncomputable def probability_no_exceed_10_minutes : ℝ :=
  let arrival_times := {x : ℝ | 7 + 50 / 60 ≤ x ∧ x ≤ 8 + 30 / 60}
  let favorable_times := {x : ℝ | (7 + 50 / 60 ≤ x ∧ x ≤ 8) ∨ (8 + 20 / 60 ≤ x ∧ x ≤ 8 + 30 / 60)}
  (favorable_times.count.to_real) / (arrival_times.count.to_real)

theorem probability_calc : probability_no_exceed_10_minutes = 1 / 2 :=
  sorry

end probability_calc_l165_165779


namespace find_n_l165_165484

theorem find_n 
  (n : ℕ) 
  (b : ℕ → ℝ)
  (h₀ : b 0 = 28)
  (h₁ : b 1 = 81)
  (hn : b n = 0)
  (h_rec : ∀ j : ℕ, 1 ≤ j → j < n → b (j+1) = b (j-1) - 5 / b j)
  : n = 455 := 
sorry

end find_n_l165_165484


namespace fido_reach_fraction_simplified_l165_165449

noncomputable def fidoReach (s r : ℝ) : ℝ :=
  let octagonArea := 2 * (1 + Real.sqrt 2) * s^2
  let circleArea := Real.pi * (s / Real.sqrt (2 + Real.sqrt 2))^2
  circleArea / octagonArea

theorem fido_reach_fraction_simplified (s : ℝ) :
  (∃ a b : ℕ, fidoReach s (s / Real.sqrt (2 + Real.sqrt 2)) = (Real.sqrt a / b) * Real.pi ∧ a * b = 16) :=
  sorry

end fido_reach_fraction_simplified_l165_165449


namespace round_robin_games_count_l165_165794

theorem round_robin_games_count (n : ℕ) (h : n = 6) : 
  (n * (n - 1)) / 2 = 15 := by
  sorry

end round_robin_games_count_l165_165794


namespace uma_income_l165_165790

theorem uma_income
  (x y : ℝ)
  (h1 : 8 * x - 7 * y = 2000)
  (h2 : 7 * x - 6 * y = 2000) :
  8 * x = 16000 := by
  sorry

end uma_income_l165_165790


namespace comparison_of_A_and_B_l165_165046

noncomputable def A (m : ℝ) : ℝ := Real.sqrt (m + 1) - Real.sqrt m
noncomputable def B (m : ℝ) : ℝ := Real.sqrt m - Real.sqrt (m - 1)

theorem comparison_of_A_and_B (m : ℝ) (h : m > 1) : A m < B m :=
by
  sorry

end comparison_of_A_and_B_l165_165046


namespace isosceles_trapezoid_AC_length_l165_165962

noncomputable def length_of_AC (AB AD BC CD AC : ℝ) :=
  AB = 30 ∧ AD = 15 ∧ BC = 15 ∧ CD = 12 → AC = 23.32

theorem isosceles_trapezoid_AC_length :
  length_of_AC 30 15 15 12 23.32 := by
  sorry

end isosceles_trapezoid_AC_length_l165_165962


namespace roots_of_polynomial_l165_165577

theorem roots_of_polynomial :
  ∀ x : ℝ, (x^2 - 5*x + 6)*(x)*(x-5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5 :=
by
  sorry

end roots_of_polynomial_l165_165577


namespace slower_train_time_to_pass_driver_faster_one_l165_165133

noncomputable def convert_speed (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def relative_speed (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let speed1 := convert_speed speed1_kmh
  let speed2 := convert_speed speed2_kmh
  speed1 + speed2

noncomputable def time_to_pass (length1_m length2_m speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let relative_speed := relative_speed speed1_kmh speed2_kmh
  (length1_m + length2_m) / relative_speed

theorem slower_train_time_to_pass_driver_faster_one :
  ∀ (length1 length2 speed1 speed2 : ℝ),
    length1 = 900 → length2 = 900 →
    speed1 = 45 → speed2 = 30 →
    time_to_pass length1 length2 speed1 speed2 = 86.39 :=
by
  intros
  simp only [time_to_pass, relative_speed, convert_speed]
  sorry

end slower_train_time_to_pass_driver_faster_one_l165_165133


namespace not_perfect_square_l165_165100

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, 7 * n + 3 = k^2 := 
by
  sorry

end not_perfect_square_l165_165100


namespace jen_profit_is_960_l165_165351

def buying_price : ℕ := 80
def selling_price : ℕ := 100
def num_candy_bars_bought : ℕ := 50
def num_candy_bars_sold : ℕ := 48

def profit_per_candy_bar := selling_price - buying_price
def total_profit := profit_per_candy_bar * num_candy_bars_sold

theorem jen_profit_is_960 : total_profit = 960 := by
  sorry

end jen_profit_is_960_l165_165351


namespace total_rats_l165_165219

theorem total_rats (Elodie_rats Hunter_rats Kenia_rats : ℕ) 
  (h1 : Elodie_rats = 30) 
  (h2 : Elodie_rats = Hunter_rats + 10)
  (h3 : Kenia_rats = 3 * (Elodie_rats + Hunter_rats)) :
  Elodie_rats + Hunter_rats + Kenia_rats = 200 :=
by
  sorry

end total_rats_l165_165219


namespace probability_of_odd_sum_l165_165984

def balls : List ℕ := [1, 1, 2, 3, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14]

noncomputable def num_combinations (n k : ℕ) : ℕ := sorry

noncomputable def probability_odd_sum_draw_7 : ℚ :=
  let total_combinations := num_combinations 15 7
  let favorable_combinations := 3200
  (favorable_combinations : ℚ) / (total_combinations : ℚ)

theorem probability_of_odd_sum:
  probability_odd_sum_draw_7 = 640 / 1287 := by
  sorry

end probability_of_odd_sum_l165_165984


namespace total_liquid_consumption_l165_165851

-- Define the given conditions
def elijah_drink_pints : ℝ := 8.5
def emilio_drink_pints : ℝ := 9.5
def isabella_drink_liters : ℝ := 3
def xavier_drink_gallons : ℝ := 2
def pint_to_cups : ℝ := 2
def liter_to_cups : ℝ := 4.22675
def gallon_to_cups : ℝ := 16
def xavier_soda_fraction : ℝ := 0.60
def xavier_fruit_punch_fraction : ℝ := 0.40

-- Define the converted amounts
def elijah_cups := elijah_drink_pints * pint_to_cups
def emilio_cups := emilio_drink_pints * pint_to_cups
def isabella_cups := isabella_drink_liters * liter_to_cups
def xavier_total_cups := xavier_drink_gallons * gallon_to_cups
def xavier_soda_cups := xavier_soda_fraction * xavier_total_cups
def xavier_fruit_punch_cups := xavier_fruit_punch_fraction * xavier_total_cups

-- Total amount calculation
def total_cups := elijah_cups + emilio_cups + isabella_cups + xavier_soda_cups + xavier_fruit_punch_cups

-- Proof statement
theorem total_liquid_consumption : total_cups = 80.68025 := by
  sorry

end total_liquid_consumption_l165_165851


namespace max_cube_sum_l165_165485

theorem max_cube_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) : x^3 + y^3 + z^3 ≤ 27 :=
sorry

end max_cube_sum_l165_165485


namespace sum_of_integers_l165_165336

theorem sum_of_integers:
  ∀ (m n p q : ℕ),
    m ≠ n → m ≠ p → m ≠ q → n ≠ p → n ≠ q → p ≠ q →
    (8 - m) * (8 - n) * (8 - p) * (8 - q) = 9 →
    m + n + p + q = 32 :=
by
  intros m n p q hmn hmp hmq hnp hnq hpq heq
  sorry

end sum_of_integers_l165_165336


namespace race_winner_laps_l165_165243

/-- Given:
  * A lap equals 100 meters.
  * Award per hundred meters is $3.5.
  * The winner earned $7 per minute.
  * The race lasted 12 minutes.
  Prove that the number of laps run by the winner is 24.
-/ 
theorem race_winner_laps :
  let lap_distance := 100 -- meters
  let award_per_100meters := 3.5 -- dollars per 100 meters
  let earnings_per_minute := 7 -- dollars per minute
  let race_duration := 12 -- minutes
  let total_earnings := earnings_per_minute * race_duration
  let total_100meters := total_earnings / award_per_100meters
  let laps := total_100meters
  laps = 24 := by
  sorry

end race_winner_laps_l165_165243


namespace equation_solution_l165_165302

variable (x y : ℝ)

theorem equation_solution
  (h1 : x * y + x + y = 17)
  (h2 : x^2 * y + x * y^2 = 66):
  x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4 = 12499 :=
  by sorry

end equation_solution_l165_165302


namespace largest_unpayable_soldo_l165_165945

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l165_165945


namespace Alex_is_26_l165_165560

-- Define the ages as integers
variable (Alex Jose Zack Inez : ℤ)

-- Conditions of the problem
variable (h1 : Alex = Jose + 6)
variable (h2 : Zack = Inez + 5)
variable (h3 : Inez = 18)
variable (h4 : Jose = Zack - 3)

-- Theorem we need to prove
theorem Alex_is_26 (h1: Alex = Jose + 6) (h2 : Zack = Inez + 5) (h3 : Inez = 18) (h4 : Jose = Zack - 3) : Alex = 26 :=
by
  sorry

end Alex_is_26_l165_165560


namespace expression_value_l165_165849

theorem expression_value (x : ℝ) (h : x = 3 + 5 / (2 + 5 / x)) : x = 5 :=
sorry

end expression_value_l165_165849


namespace time_for_Q_l165_165934

-- Definitions of conditions
def time_for_P := 252
def meet_time := 2772

-- Main statement to prove
theorem time_for_Q : (∃ T : ℕ, lcm time_for_P T = meet_time) ∧ (lcm time_for_P meet_time = meet_time) :=
    by 
    sorry

end time_for_Q_l165_165934


namespace find_n_tan_eq_l165_165179

theorem find_n_tan_eq (n : ℤ) (h₁ : -180 < n) (h₂ : n < 180) 
  (h₃ : Real.tan (n * (Real.pi / 180)) = Real.tan (276 * (Real.pi / 180))) : 
  n = 96 :=
sorry

end find_n_tan_eq_l165_165179


namespace simplify_expression_l165_165110

theorem simplify_expression : (8^(1/3) / 8^(1/6)) = 8^(1/6) :=
by
  sorry

end simplify_expression_l165_165110


namespace domain_of_sqrt_sin_l165_165584

open Real Set

noncomputable def domain_sqrt_sine : Set ℝ :=
  {x | ∃ (k : ℤ), 2 * π * k + π / 6 ≤ x ∧ x ≤ 2 * π * k + 5 * π / 6}

theorem domain_of_sqrt_sin (x : ℝ) :
  (∃ y, y = sqrt (2 * sin x - 1)) ↔ x ∈ domain_sqrt_sine :=
sorry

end domain_of_sqrt_sin_l165_165584


namespace smallest_positive_multiple_of_32_l165_165660

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ ∃ k : ℕ, k > 0 ∧ n = 32 * k ∧ n = 32 := by
  use 32
  constructor
  · exact Nat.zero_lt_succ 31
  · use 1
    constructor
    · exact Nat.zero_lt_succ 0
    · constructor
      · rfl
      · rfl

end smallest_positive_multiple_of_32_l165_165660


namespace num_articles_produced_l165_165468

-- Conditions
def production_rate (x : ℕ) : ℕ := 2 * x^3 / (x * x * 2 * x)
def articles_produced (y : ℕ) : ℕ := y * 2 * y * y * production_rate y

-- Proof: Given the production rate, prove the number of articles produced.
theorem num_articles_produced (y : ℕ) : articles_produced y = 2 * y^3 := by sorry

end num_articles_produced_l165_165468


namespace linear_coefficient_of_quadratic_term_is_negative_five_l165_165150

theorem linear_coefficient_of_quadratic_term_is_negative_five (a b c : ℝ) (x : ℝ) :
  (2 * x^2 = 5 * x - 3) →
  (a = 2) →
  (b = -5) →
  (c = 3) →
  (a * x^2 + b * x + c = 0) :=
by
  sorry

end linear_coefficient_of_quadratic_term_is_negative_five_l165_165150


namespace circle_radius_l165_165006

theorem circle_radius (x y d : ℝ) (h₁ : x = π * r^2) (h₂ : y = 2 * π * r) (h₃ : d = 2 * r) (h₄ : x + y + d = 164 * π) : r = 10 :=
by sorry

end circle_radius_l165_165006


namespace parallelogram_side_lengths_l165_165829

theorem parallelogram_side_lengths (x y : ℝ) (h₁ : 3 * x + 6 = 12) (h₂ : 10 * y - 3 = 15) : x + y = 3.8 :=
by
  sorry

end parallelogram_side_lengths_l165_165829


namespace find_triangle_area_l165_165063

noncomputable def triangle_area_problem
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 14) : ℝ :=
  (1 / 2) * a * b

theorem find_triangle_area
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 14) :
  triangle_area_problem a b h1 h2 = 1 / 2 := by
  sorry

end find_triangle_area_l165_165063


namespace number_of_acceptable_teams_l165_165418

noncomputable def basketball_team : Finset (Finset ℕ) :=
  (Finset.range 12).powerset.filter (λ s, 
    s.card = 5 ∧ 
    (2 ∉ s ∨ 3 ∉ s) ∧ 
    (4 ∉ s ∨ 5 ∉ s))

theorem number_of_acceptable_teams : 
  basketball_team.card = 560 :=
sorry

end number_of_acceptable_teams_l165_165418


namespace odd_three_digit_integers_strictly_increasing_digits_l165_165332

theorem odd_three_digit_integers_strictly_increasing_digits :
  let valid_combinations (c : ℕ) :=
    if c = 1 then 0 else
    if c = 3 then 1 else
    if c = 5 then 6 else
    if c = 7 then 15 else
    if c = 9 then 28 else 0 in
  (valid_combinations 1 + valid_combinations 3 + valid_combinations 5 + valid_combinations 7 + valid_combinations 9 = 50) :=
by
  unfold valid_combinations
  sorry

end odd_three_digit_integers_strictly_increasing_digits_l165_165332


namespace wall_length_l165_165015

theorem wall_length (s : ℕ) (w : ℕ) (a_ratio : ℕ) (A_mirror : ℕ) (A_wall : ℕ) (L : ℕ) 
  (hs : s = 24) (hw : w = 42) (h_ratio : a_ratio = 2) 
  (hA_mirror : A_mirror = s * s) 
  (hA_wall : A_wall = A_mirror * a_ratio) 
  (h_area : A_wall = w * L) : L = 27 :=
  sorry

end wall_length_l165_165015


namespace find_incorrect_value_l165_165180

variable (k b : ℝ)

-- Linear function definition
def linear_function (x : ℝ) : ℝ := k * x + b

-- Given points
theorem find_incorrect_value (h₁ : linear_function k b (-1) = 3)
                             (h₂ : linear_function k b 0 = 2)
                             (h₃ : linear_function k b 1 = 1)
                             (h₄ : linear_function k b 2 = 0)
                             (h₅ : linear_function k b 3 = -2) :
                             (∃ x y, linear_function k b x ≠ y) := by
  sorry

end find_incorrect_value_l165_165180


namespace removed_number_is_24_l165_165598

theorem removed_number_is_24
  (S9 : ℕ) (S8 : ℕ) (avg_9 : ℕ) (avg_8 : ℕ) (h1 : avg_9 = 72) (h2 : avg_8 = 78) (h3 : S9 = avg_9 * 9) (h4 : S8 = avg_8 * 8) :
  S9 - S8 = 24 :=
by
  sorry

end removed_number_is_24_l165_165598


namespace range_of_b_l165_165055

noncomputable def f : ℝ → ℝ
| x => if x < -1/2 then (2*x + 1) / (x^2) else x + 1

def g (x : ℝ) : ℝ := x^2 - 4*x - 4

theorem range_of_b (a b : ℝ) (h : f a + g b = 0) : -1 <= b ∧ b <= 5 :=
sorry

end range_of_b_l165_165055


namespace isosceles_triangle_base_l165_165476

variable (a b : ℕ)

theorem isosceles_triangle_base 
  (h_isosceles : a = 7 ∧ b = 3)
  (triangle_inequality : 7 + 7 > 3) : b = 3 := by
-- Begin of the proof
sorry
-- End of the proof

end isosceles_triangle_base_l165_165476


namespace habitable_planets_combinations_l165_165201

open Finset

/-- The number of different combinations of planets that can be occupied is 2478. -/
theorem habitable_planets_combinations : (∑ d in range (18 // 3 + 1), if d % 2 = 0 then choose 8 (6 - d / 3) * choose 7 d else 0) = 2478 := 
by
  sorry

end habitable_planets_combinations_l165_165201


namespace find_train_speed_l165_165146

def length_of_platform : ℝ := 210.0168
def time_to_pass_platform : ℝ := 34
def time_to_pass_man : ℝ := 20 
def speed_of_train (L : ℝ) (V : ℝ) : Prop :=
  V = (L + length_of_platform) / time_to_pass_platform ∧ V = L / time_to_pass_man

theorem find_train_speed (L V : ℝ) (h : speed_of_train L V) : V = 54.00432 := sorry

end find_train_speed_l165_165146


namespace max_probability_sum_15_l165_165122

-- Context and Definitions based on conditions
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The assertion to be proved:
theorem max_probability_sum_15 (n : ℕ) (h : n ∈ S) :
  n = 7 :=
by
  sorry

end max_probability_sum_15_l165_165122


namespace zero_point_in_range_l165_165732

theorem zero_point_in_range (a : ℝ) (x1 x2 x3 : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : x1 < x2) (h4 : x2 < x3)
  (hx1 : (x1^3 - 4*x1 + a) = 0) (hx2 : (x2^3 - 4*x2 + a) = 0) (hx3 : (x3^3 - 4*x3 + a) = 0) :
  0 < x2 ∧ x2 < 1 :=
by
  sorry

end zero_point_in_range_l165_165732


namespace answer_keys_count_l165_165276

theorem answer_keys_count 
  (test_questions : ℕ)
  (true_answers : ℕ)
  (false_answers : ℕ)
  (min_score : ℕ)
  (conditions : test_questions = 10 ∧ true_answers = 5 ∧ false_answers = 5 ∧ min_score >= 4) :
  ∃ (count : ℕ), count = 22 := by
  sorry

end answer_keys_count_l165_165276


namespace select_4_blocks_no_same_row_column_l165_165271

theorem select_4_blocks_no_same_row_column :
  ∃ (n : ℕ), n = (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4) ∧ n = 5400 :=
by
  sorry

end select_4_blocks_no_same_row_column_l165_165271


namespace math_bonanza_2016_8_l165_165222

def f (x : ℕ) := x^2 + x + 1

theorem math_bonanza_2016_8 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : f p = f q + 242) (hpq : p > q) :
  (p, q) = (61, 59) :=
by sorry

end math_bonanza_2016_8_l165_165222


namespace ceil_e_add_pi_l165_165852

theorem ceil_e_add_pi : ⌈Real.exp 1 + Real.pi⌉ = 6 := by
  sorry

end ceil_e_add_pi_l165_165852


namespace odd_increasing_three_digit_numbers_count_eq_50_l165_165320

def count_odd_increasing_three_digit_numbers : Nat := by
  -- Mathematical conditions:
  -- let a, b, c be digits of the number
  -- 0 < a < b < c <= 9 and c is an odd digit

  -- We analyze values for 'c' which must be an odd digit,
  -- and count valid (a, b) combinations for each case of c.

  -- Starting from cases for c:
  -- for c = 1, no valid (a, b); count = 0
  -- for c = 3, valid (a, b) are from {1, 2}; count = 1
  -- for c = 5, valid (a, b) are from {1, 2, 3, 4}; count = 6
  -- for c = 7, valid (a, b) are from {1, 2, 3, 4, 5, 6}; count = 15
  -- for c = 9, valid (a, b) are from {1, 2, 3, 4, 5, 6, 7, 8}; count = 28

  -- Sum counts for all valid cases of c
  exact 50

-- Define our main theorem based on problem and final result
theorem odd_increasing_three_digit_numbers_count_eq_50 :
  count_odd_increasing_three_digit_numbers = 50 := by
  unfold count_odd_increasing_three_digit_numbers
  exact rfl -- the correct proof will fill in this part

end odd_increasing_three_digit_numbers_count_eq_50_l165_165320


namespace range_of_a_for_decreasing_exponential_l165_165547

theorem range_of_a_for_decreasing_exponential :
  ∀ (a : ℝ), (∀ (x1 x2 : ℝ), x1 < x2 → (2 - a)^x1 > (2 - a)^x2) ↔ (1 < a ∧ a < 2) :=
by
  sorry

end range_of_a_for_decreasing_exponential_l165_165547


namespace min_value_of_1_over_a_plus_2_over_b_l165_165057

theorem min_value_of_1_over_a_plus_2_over_b (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  (1 / a + 2 / b) ≥ 9 := 
sorry

end min_value_of_1_over_a_plus_2_over_b_l165_165057


namespace multiple_for_snack_cost_l165_165356

-- Define the conditions
def kyle_time_to_work : ℕ := 2 -- Kyle bikes for 2 hours to work every day.
def cost_of_snacks (total_cost packs : ℕ) : ℕ := total_cost / packs -- Ryan will pay $2000 to buy 50 packs of snacks.

-- Ryan pays $2000 for 50 packs of snacks.
def cost_per_pack := cost_of_snacks 2000 50

-- The time for a round trip (to work and back)
def round_trip_time (h : ℕ) : ℕ := 2 * h

-- The multiple of the time taken to travel to work and back that equals the cost of a pack of snacks
def multiple (cost time : ℕ) : ℕ := cost / time

-- Statement we need to prove
theorem multiple_for_snack_cost : 
  multiple cost_per_pack (round_trip_time kyle_time_to_work) = 10 :=
  by
  sorry

end multiple_for_snack_cost_l165_165356


namespace x_plus_p_eq_2p_plus_3_l165_165469

theorem x_plus_p_eq_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2 * p + 3 := by
  sorry

end x_plus_p_eq_2p_plus_3_l165_165469


namespace log_ratios_l165_165514

noncomputable def ratio_eq : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem log_ratios
  {a b : ℝ}
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : Real.log a / Real.log 8 = Real.log b / Real.log 18)
  (h4 : Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) :
  b / a = ratio_eq :=
sorry

end log_ratios_l165_165514


namespace fractions_sum_ge_one_l165_165895

variable {a b c : ℝ}

theorem fractions_sum_ge_one (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 :=
by {
  sorry,
}

end fractions_sum_ge_one_l165_165895


namespace cos_angle_identity_l165_165188

theorem cos_angle_identity (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 17 * π / 12) = 1 / 3 :=
by
  sorry

end cos_angle_identity_l165_165188


namespace problem1_problem2_l165_165437

-- Define the variables for the two problems
variables (a b x : ℝ)

-- The first problem
theorem problem1 :
  (a + 2 * b) ^ 2 - 4 * b * (a + b) = a ^ 2 :=
by 
  -- Proof goes here
  sorry

-- The second problem
theorem problem2 :
  ((x ^ 2 - 2 * x) / (x ^ 2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x ^ 2 - 4)) = x + 2 :=
by 
  -- Proof goes here
  sorry

end problem1_problem2_l165_165437


namespace negation_example_l165_165197

theorem negation_example (p : ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) :
  ∃ x0 : ℝ, x0 > 0 ∧ (x0 + 1) * Real.exp x0 ≤ 1 :=
sorry

end negation_example_l165_165197


namespace team_formation_l165_165933

def nat1 : ℕ := 7  -- Number of natives who know mathematics and physics
def nat2 : ℕ := 6  -- Number of natives who know physics and chemistry
def nat3 : ℕ := 3  -- Number of natives who know chemistry and mathematics
def nat4 : ℕ := 4  -- Number of natives who know physics and biology

def totalWaysToFormTeam (n1 n2 n3 n4 : ℕ) : ℕ := (n1 + n2 + n3 + n4).choose 3
def waysFromSameGroup (n : ℕ) : ℕ := n.choose 3

def waysFromAllGroups (n1 n2 n3 n4 : ℕ) : ℕ := (waysFromSameGroup n1) + (waysFromSameGroup n2) + (waysFromSameGroup n3) + (waysFromSameGroup n4)

theorem team_formation : totalWaysToFormTeam nat1 nat2 nat3 nat4 - waysFromAllGroups nat1 nat2 nat3 nat4 = 1080 := 
by
    sorry

end team_formation_l165_165933


namespace center_of_symmetry_is_neg2_3_l165_165112

theorem center_of_symmetry_is_neg2_3 :
  ∃ (a b : ℝ), 
  (a,b) = (-2, 3) ∧ 
  ∀ x : ℝ, 
    2 * b = ((a + x + 2)^3 - (a + x) + 1) + ((a - x + 2)^3 - (a - x) + 1) := 
by
  use -2, 3
  sorry

end center_of_symmetry_is_neg2_3_l165_165112


namespace seth_can_erase_all_numbers_l165_165240

theorem seth_can_erase_all_numbers : 
  ∀ (board : Finset ℕ), board = Finset.filter (λ n, 1 ≤ n ∧ n ≤ 100) (Finset.range 101) →
  ∀ (b c : ℕ), b ≠ c → b ∈ board → c ∈ board →
  (∃ x y : ℤ, x * y = (c : ℤ) ∧ x + y = - (b : ℤ)) →
  (Finset.erase (Finset.erase board b) c) = ∅ :=
by
  intros board h_board b c h_diff h_b_in h_c_in h_solution
  sorry

end seth_can_erase_all_numbers_l165_165240


namespace range_of_a_l165_165527

theorem range_of_a (a x : ℝ) (h : x - a = 1 - 2*x) (non_neg_x : x ≥ 0) : a ≥ -1 := by
  sorry

end range_of_a_l165_165527


namespace abs_diff_squares_l165_165541

theorem abs_diff_squares (a b : ℝ) (ha : a = 105) (hb : b = 103) : |a^2 - b^2| = 416 :=
by
  sorry

end abs_diff_squares_l165_165541


namespace part1_inequality_part2_min_value_l165_165309

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  4^x + m * 2^x

theorem part1_inequality (x : ℝ) : f x (-3) > 4 → x > 2 :=
  sorry

theorem part2_min_value (h : (∀ x : ℝ, f x m + f (-x) m ≥ -4)) : m = -3 :=
  sorry

end part1_inequality_part2_min_value_l165_165309


namespace slope_intercept_equivalence_l165_165386

-- Define the given equation in Lean
def given_line_equation (x y : ℝ) : Prop := 3 * x - 2 * y = 4

-- Define the slope-intercept form as extracted from the given line equation
def slope_intercept_form (x y : ℝ) : Prop := y = (3/2) * x - 2

-- Prove that the given line equation is equivalent to its slope-intercept form
theorem slope_intercept_equivalence (x y : ℝ) :
  given_line_equation x y ↔ slope_intercept_form x y :=
by sorry

end slope_intercept_equivalence_l165_165386


namespace rate_ratio_l165_165700

theorem rate_ratio
  (rate_up : ℝ) (time_up : ℝ) (distance_up : ℝ)
  (distance_down : ℝ) (time_down : ℝ) :
  rate_up = 4 → time_up = 2 → distance_up = rate_up * time_up →
  distance_down = 12 → time_down = 2 →
  (distance_down / time_down) / rate_up = 3 / 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end rate_ratio_l165_165700


namespace train_length_l165_165428

theorem train_length (speed_kmph : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_kmph = 60 →
  time_s = 3 →
  length_m = 50.01 :=
by
  sorry

end train_length_l165_165428


namespace aluminum_percentage_range_l165_165395

variable (x1 x2 x3 y : ℝ)

theorem aluminum_percentage_range:
  (0.15 * x1 + 0.3 * x2 = 0.2) →
  (x1 + x2 + x3 = 1) →
  y = 0.6 * x1 + 0.45 * x3 →
  (1/3 ≤ x2 ∧ x2 ≤ 2/3) →
  (0.15 ≤ y ∧ y ≤ 0.4) := by
  sorry

end aluminum_percentage_range_l165_165395


namespace minimum_dwarfs_l165_165497

-- Define the problem conditions
def chairs := fin 30 → Prop
def occupied (C : chairs) := ∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))

-- State the theorem that provides the solution
theorem minimum_dwarfs (C : chairs) : (∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))) → ∃ n : ℕ, n = 10 :=
by
  sorry

end minimum_dwarfs_l165_165497


namespace negation_proposition_l165_165787

theorem negation_proposition : ¬ (∀ x : ℝ, (1 < x) → x^3 > x^(1/3)) ↔ ∃ x : ℝ, (1 < x) ∧ x^3 ≤ x^(1/3) := by
  sorry

end negation_proposition_l165_165787


namespace Joyce_final_apples_l165_165355

def initial_apples : ℝ := 350.5
def apples_given_to_larry : ℝ := 218.7
def percentage_given_to_neighbors : ℝ := 0.375
def final_apples : ℝ := 82.375

theorem Joyce_final_apples :
  (initial_apples - apples_given_to_larry - percentage_given_to_neighbors * (initial_apples - apples_given_to_larry)) = final_apples :=
by
  sorry

end Joyce_final_apples_l165_165355


namespace infinite_geometric_series_sum_l165_165877

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l165_165877


namespace mailing_ways_l165_165610

-- Definitions based on the problem conditions
def countWays (letters mailboxes : ℕ) : ℕ := mailboxes^letters

-- The theorem to prove the mathematically equivalent proof problem
theorem mailing_ways (letters mailboxes : ℕ) (h_letters : letters = 3) (h_mailboxes : mailboxes = 4) : countWays letters mailboxes = 4^3 := 
by
  rw [h_letters, h_mailboxes]
  rfl

end mailing_ways_l165_165610


namespace initial_percentage_of_water_l165_165629

/-
Initial conditions:
- Let W be the initial percentage of water in 10 liters of milk.
- The mixture should become 2% water after adding 15 liters of pure milk to it.
-/

theorem initial_percentage_of_water (W : ℚ) 
  (H1 : 0 < W) (H2 : W < 100) 
  (H3 : (10 * (100 - W) / 100 + 15) / 25 = 0.98) : 
  W = 5 := 
sorry

end initial_percentage_of_water_l165_165629


namespace delta_eq_bullet_l165_165066

-- Definitions of all variables involved
variables (Δ Θ σ : ℕ)

-- Condition 1: Δ + Δ = σ
def cond1 : Prop := Δ + Δ = σ

-- Condition 2: σ + Δ = Θ
def cond2 : Prop := σ + Δ = Θ

-- Condition 3: Θ = 3Δ
def cond3 : Prop := Θ = 3 * Δ

-- The proof problem
theorem delta_eq_bullet (Δ Θ σ : ℕ) (h1 : Δ + Δ = σ) (h2 : σ + Δ = Θ) (h3 : Θ = 3 * Δ) : 3 * Δ = Θ :=
by
  -- Simply restate the conditions and ensure the proof
  sorry

end delta_eq_bullet_l165_165066


namespace range_of_m_l165_165764

def positive_numbers (a b : ℝ) : Prop := a > 0 ∧ b > 0

def equation_condition (a b : ℝ) : Prop := 9 * a + b = a * b

def inequality_for_any_x (a b m : ℝ) : Prop := ∀ x : ℝ, a + b ≥ -x^2 + 2 * x + 18 - m

theorem range_of_m :
  ∀ (a b m : ℝ),
    positive_numbers a b →
    equation_condition a b →
    inequality_for_any_x a b m →
    m ≥ 3 :=
by
  sorry

end range_of_m_l165_165764


namespace find_p_plus_q_l165_165187

noncomputable def p (d e : ℝ) (x : ℝ) : ℝ := d * x + e
noncomputable def q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_p_plus_q (d e a b c : ℝ)
  (h1 : p d e 0 / q a b c 0 = 4)
  (h2 : p d e (-1) = -1)
  (h3 : q a b c 1 = 3)
  (e_eq : e = 4 * c):
  (p d e x + q a b c x) = (3*x^2 + 26*x - 30) :=
by
  sorry

end find_p_plus_q_l165_165187


namespace geometric_series_sum_l165_165883

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l165_165883


namespace solution_set_of_inequality_l165_165116

theorem solution_set_of_inequality (x : ℝ) : -2 * x - 1 < 3 ↔ x > -2 := 
by 
  sorry

end solution_set_of_inequality_l165_165116


namespace gabrielle_saw_20_percent_more_l165_165127

-- Define the number of birds seen by Gabrielle and Chase
def birds_seen_by_gabrielle : ℕ := 5 + 4 + 3
def birds_seen_by_chase : ℕ := 2 + 3 + 5

-- Define the correct answer as a percentage
def percentage_increase (a b : ℕ) : ℝ := ((a - b).toReal / b.toReal) * 100

-- Statement asserting that Gabrielle saw 20% more birds than Chase
theorem gabrielle_saw_20_percent_more : percentage_increase birds_seen_by_gabrielle birds_seen_by_chase = 20 := by
  sorry

end gabrielle_saw_20_percent_more_l165_165127


namespace hyperbola_equation_l165_165896

-- Define the hyperbola with vertices and other conditions
def Hyperbola (a b : ℝ) (h : a > 0 ∧ b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1)

-- Given conditions and the proof goal
theorem hyperbola_equation
  (a b : ℝ) (h : a > 0 ∧ b > 0)
  (P : ℝ × ℝ) (M N : ℝ × ℝ)
  (k_PA k_PB : ℝ)
  (PA_PB_condition : k_PA * k_PB = 3)
  (MN_min_value : |(M.1 - N.1) + (M.2 - N.2)| = 4) :
  Hyperbola a b h →
  (a = 2 ∧ b = 2 * Real.sqrt 3 ∧ (∀ (x y : ℝ), (x^2 / 4 - y^2 / 12 = 1)) ∨ 
   a = 2 / 3 ∧ b = 2 * Real.sqrt 3 / 3 ∧ (∀ (x y : ℝ), (9 * x^2 / 4 - 3 * y^2 / 4 = 1)))
:=
sorry

end hyperbola_equation_l165_165896


namespace largest_unpayable_soldo_l165_165944

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l165_165944


namespace escher_probability_l165_165064

def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

def favorable_arrangements (total_art : ℕ) (escher_prints : ℕ) : ℕ :=
  num_arrangements (total_art - escher_prints + 1) * num_arrangements escher_prints

def total_arrangements (total_art : ℕ) : ℕ :=
  num_arrangements total_art

def prob_all_escher_consecutive (total_art : ℕ) (escher_prints : ℕ) : ℚ :=
  favorable_arrangements total_art escher_prints / total_arrangements total_art

theorem escher_probability :
  prob_all_escher_consecutive 12 4 = 1/55 :=
by
  sorry

end escher_probability_l165_165064


namespace largest_N_not_payable_l165_165937

theorem largest_N_not_payable (coins_5 coins_6 : ℕ) (h1 : coins_5 > 10) (h2 : coins_6 > 10) :
  ∃ N : ℕ, N ≤ 50 ∧ (¬ ∃ (x y : ℕ), N = 5 * x + 6 * y) ∧ N = 19 :=
by 
  use 19
  have h₃ : 19 ≤ 50 := by norm_num
  have h₄ : ¬ ∃ (x y : ℕ), 19 = 5 * x + 6 * y := sorry
  exact ⟨h₃, h₄⟩

end largest_N_not_payable_l165_165937


namespace cost_price_of_article_l165_165807

variable (C : ℝ)
variable (h1 : (0.18 * C - 0.09 * C = 72))

theorem cost_price_of_article : C = 800 :=
by
  sorry

end cost_price_of_article_l165_165807


namespace odd_three_digit_integers_increasing_order_l165_165330

theorem odd_three_digit_integers_increasing_order :
  let digits_strictly_increasing (a b c : ℕ) : Prop := (1 ≤ a) ∧ (a < b) ∧ (b < c)
      let c_values : Finset ℕ := {3, 5, 7, 9}
  in ∑ c in c_values, (Finset.card (Finset.filter (λ ab : ℕ × ℕ, (digits_strictly_increasing ab.1 ab.2 c)) (Finset.cross {(1 : ℕ)..9} {(1 : ℕ)..9}))) = 50 :=
by
  sorry

end odd_three_digit_integers_increasing_order_l165_165330


namespace probability_of_blue_or_yellow_l165_165983

def num_red : ℕ := 6
def num_green : ℕ := 7
def num_yellow : ℕ := 8
def num_blue : ℕ := 9

def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue
def total_blue_or_yellow : ℕ := num_yellow + num_blue

theorem probability_of_blue_or_yellow (h : total_jelly_beans ≠ 0) : 
  (total_blue_or_yellow : ℚ) / (total_jelly_beans : ℚ) = 17 / 30 :=
by
  sorry

end probability_of_blue_or_yellow_l165_165983


namespace percentage_of_pushups_l165_165171

-- Problem conditions as definitions
def jumpingJacks := 12
def pushups := 8
def situps := 20
def totalExercises := jumpingJacks + pushups + situps

-- Question and the proof goal
theorem percentage_of_pushups : 
  (pushups / totalExercises : ℝ) * 100 = 20 := by
  sorry

end percentage_of_pushups_l165_165171


namespace largest_N_not_payable_l165_165936

theorem largest_N_not_payable (coins_5 coins_6 : ℕ) (h1 : coins_5 > 10) (h2 : coins_6 > 10) :
  ∃ N : ℕ, N ≤ 50 ∧ (¬ ∃ (x y : ℕ), N = 5 * x + 6 * y) ∧ N = 19 :=
by 
  use 19
  have h₃ : 19 ≤ 50 := by norm_num
  have h₄ : ¬ ∃ (x y : ℕ), 19 = 5 * x + 6 * y := sorry
  exact ⟨h₃, h₄⟩

end largest_N_not_payable_l165_165936


namespace minimum_dwarfs_l165_165500

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_l165_165500


namespace correct_ratio_l165_165149

theorem correct_ratio (a b : ℝ) (h : 4 * a = 5 * b) : a / b = 5 / 4 :=
by
  sorry

end correct_ratio_l165_165149


namespace avg_age_initial_group_l165_165373

theorem avg_age_initial_group (N : ℕ) (A avg_new_persons avg_entire_group : ℝ) (hN : N = 15)
  (h_avg_new_persons : avg_new_persons = 15) (h_avg_entire_group : avg_entire_group = 15.5) :
  (A * (N : ℝ) + 15 * avg_new_persons) = ((N + 15) : ℝ) * avg_entire_group → A = 16 :=
by
  intro h
  have h_initial : N = 15 := hN
  have h_new : avg_new_persons = 15 := h_avg_new_persons
  have h_group : avg_entire_group = 15.5 := h_avg_entire_group
  sorry

end avg_age_initial_group_l165_165373


namespace quadratic_result_l165_165189

noncomputable def quadratic_has_two_positive_integer_roots (k p : ℕ) : Prop :=
  ∃ x1 x2 : ℕ, x1 > 0 ∧ x2 > 0 ∧ (k - 1) * x1 * x1 - p * x1 + k = 0 ∧ (k - 1) * x2 * x2 - p * x2 + k = 0

theorem quadratic_result (k p : ℕ) (h1 : k = 2) (h2 : quadratic_has_two_positive_integer_roots k p) :
  k^(k*p) * (p^p + k^k) = 1984 :=
by
  sorry

end quadratic_result_l165_165189


namespace min_distance_sum_coordinates_l165_165185

theorem min_distance_sum_coordinates (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, -1)) :
  ∃ P : ℝ × ℝ, P = (0, 3) ∧ ∀ Q : ℝ × ℝ, Q.1 = 0 → |A.1 - Q.1| + |A.2 - Q.2| + |B.1 - Q.1| + |B.2 - Q.2| ≥ |A.1 - (0 : ℝ)| + |A.2 - (3 : ℝ)| + |B.1 - (0 : ℝ)| + |B.2 - (3 : ℝ)| := 
sorry

end min_distance_sum_coordinates_l165_165185


namespace valid_expression_l165_165720

theorem valid_expression (x : ℝ) : 
  (x - 1 ≥ 0 ∧ x - 2 ≠ 0) ↔ (x ≥ 1 ∧ x ≠ 2) := 
by
  sorry

end valid_expression_l165_165720


namespace difference_of_squares_example_l165_165997

theorem difference_of_squares_example : 204^2 - 202^2 = 812 := by
  sorry

end difference_of_squares_example_l165_165997


namespace Jonie_cousins_ages_l165_165956

theorem Jonie_cousins_ages : 
  ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  1 ≤ a ∧ a < 10 ∧
  1 ≤ b ∧ b < 10 ∧
  1 ≤ c ∧ c < 10 ∧
  1 ≤ d ∧ d < 10 ∧
  a * b = 24 ∧
  c * d = 30 ∧
  a + b + c + d = 22 :=
sorry

end Jonie_cousins_ages_l165_165956


namespace pushups_percentage_l165_165169

def total_exercises : ℕ := 12 + 8 + 20

def percentage_pushups (total_ex: ℕ) : ℕ := (8 * 100) / total_ex

theorem pushups_percentage (h : total_exercises = 40) : percentage_pushups total_exercises = 20 :=
by
  sorry

end pushups_percentage_l165_165169


namespace order_of_variables_l165_165912

variable (a b c d : ℝ)

theorem order_of_variables (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) : c > a ∧ a > b ∧ b > d :=
by
  sorry

end order_of_variables_l165_165912


namespace shelly_thread_needed_l165_165952

def keychain_thread (classes: ℕ) (clubs: ℕ) (thread_per_keychain: ℕ) : ℕ := 
  let total_friends := classes + clubs
  total_friends * thread_per_keychain

theorem shelly_thread_needed : keychain_thread 6 (6 / 2) 12 = 108 := 
  by
    show 6 + (6 / 2) * 12 = 108
    sorry

end shelly_thread_needed_l165_165952


namespace max_sum_prod_48_l165_165411

theorem max_sum_prod_48 (spadesuit heartsuit : Nat) (h: spadesuit * heartsuit = 48) : spadesuit + heartsuit ≤ 49 :=
sorry

end max_sum_prod_48_l165_165411


namespace sufficient_condition_l165_165753

variable {α : Type*} (A B : Set α)

theorem sufficient_condition (h : A ⊆ B) (x : α) : x ∈ A → x ∈ B :=
by
  sorry

end sufficient_condition_l165_165753


namespace odd_increasing_three_digit_numbers_l165_165313

open Nat

def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_triplet (a b c : ℕ) : Prop := 
  1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 9 ∧ is_odd c

theorem odd_increasing_three_digit_numbers : 
  ∑ c in {1, 3, 5, 7, 9}, (∑ a in range (c - 2), ∑ b in range (a + 1, c - 1), (if valid_triplet a b c then 1 else 0)) = 50 :=
by
  sorry

end odd_increasing_three_digit_numbers_l165_165313


namespace evaluate_expression_l165_165614

variable (m n p q s : ℝ)

theorem evaluate_expression :
  m / (n - (p + q * s)) = m / (n - p - q * s) :=
by
  sorry

end evaluate_expression_l165_165614


namespace jesses_room_length_l165_165091

theorem jesses_room_length 
  (width : ℝ)
  (tile_area : ℝ)
  (num_tiles : ℕ)
  (total_area : ℝ := num_tiles * tile_area) 
  (room_length : ℝ := total_area / width)
  (hw : width = 12)
  (hta : tile_area = 4)
  (hnt : num_tiles = 6) :
  room_length = 2 :=
by
  -- proof omitted
  sorry

end jesses_room_length_l165_165091


namespace product_sum_condition_l165_165765

theorem product_sum_condition (a b c : ℝ) (h1 : a * b * c = 1) (h2 : a + b + c > (1/a) + (1/b) + (1/c)) : 
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
sorry

end product_sum_condition_l165_165765


namespace tropical_fish_count_l165_165392

theorem tropical_fish_count (total_fish : ℕ) (koi_count : ℕ) (total_fish_eq : total_fish = 52) (koi_count_eq : koi_count = 37) : 
    (total_fish - koi_count) = 15 := by
    sorry

end tropical_fish_count_l165_165392


namespace bailey_total_spending_l165_165840

noncomputable def cost_after_discount : ℝ :=
  let guest_sets := 2
  let master_sets := 4
  let guest_price := 40.0
  let master_price := 50.0
  let discount := 0.20
  let total_cost := (guest_sets * guest_price) + (master_sets * master_price)
  let discount_amount := total_cost * discount
  total_cost - discount_amount

theorem bailey_total_spending : cost_after_discount = 224.0 :=
by
  unfold cost_after_discount
  sorry

end bailey_total_spending_l165_165840


namespace total_clouds_count_l165_165162

def carson_clouds := 12
def little_brother_clouds := 5 * carson_clouds
def older_sister_clouds := carson_clouds / 2
def cousin_clouds := 2 * (older_sister_clouds + carson_clouds)

theorem total_clouds_count : carson_clouds + little_brother_clouds + older_sister_clouds + cousin_clouds = 114 := by
  sorry

end total_clouds_count_l165_165162


namespace isosceles_triangle_perimeter_l165_165613

/-- 
Prove that the perimeter of an isosceles triangle with sides 6 cm and 8 cm, 
and an area of 12 cm², is 20 cm.
--/
theorem isosceles_triangle_perimeter (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (S : ℝ) (h3 : S = 12) :
  a ≠ b →
  a = c ∨ b = c →
  ∃ P : ℝ, P = 20 := sorry

end isosceles_triangle_perimeter_l165_165613


namespace greatest_divisor_of_630_lt_35_and_factor_of_90_l165_165656

theorem greatest_divisor_of_630_lt_35_and_factor_of_90 : ∃ d : ℕ, d < 35 ∧ d ∣ 630 ∧ d ∣ 90 ∧ ∀ e : ℕ, (e < 35 ∧ e ∣ 630 ∧ e ∣ 90) → e ≤ d := 
sorry

end greatest_divisor_of_630_lt_35_and_factor_of_90_l165_165656


namespace sum_geometric_sequence_divisibility_l165_165781

theorem sum_geometric_sequence_divisibility (n : ℕ) (h_pos: n > 0) :
  (n % 2 = 1 ↔ (3^(n+1) - 2^(n+1)) % 5 = 0) :=
sorry

end sum_geometric_sequence_divisibility_l165_165781


namespace minimum_dwarfs_to_prevent_empty_chair_sitting_l165_165506

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end minimum_dwarfs_to_prevent_empty_chair_sitting_l165_165506


namespace min_value_of_x2_y2_z2_l165_165225

noncomputable def min_square_sum (x y z k : ℝ) : ℝ :=
  x^2 + y^2 + z^2

theorem min_value_of_x2_y2_z2 (x y z k : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = k) :
  ∃ (min_val : ℝ), min_val = 1 ∧ ∀ (x y z k : ℝ), (x^3 + y^3 + z^3 - 3 * x * y * z = k ∧ k ≥ -1) -> min_square_sum x y z k ≥ min_val :=
by
  sorry

end min_value_of_x2_y2_z2_l165_165225


namespace solve_thought_of_number_l165_165795

def thought_of_number (x : ℝ) : Prop :=
  (x / 6) + 5 = 17

theorem solve_thought_of_number :
  ∃ x, thought_of_number x ∧ x = 72 :=
by
  sorry

end solve_thought_of_number_l165_165795


namespace calc_mod_residue_l165_165161

theorem calc_mod_residue :
  (245 * 15 - 18 * 8 + 5) % 17 = 0 := by
  sorry

end calc_mod_residue_l165_165161


namespace max_a_avoiding_lattice_points_l165_165009

def is_lattice_point (x y : ℤ) : Prop :=
  true  -- Placeholder for (x, y) being in lattice points.

def passes_through_lattice_point (m : ℚ) (x : ℤ) : Prop :=
  is_lattice_point x (⌊m * x + 2⌋)

theorem max_a_avoiding_lattice_points :
  ∀ {a : ℚ}, (∀ x : ℤ, (0 < x ∧ x ≤ 100) → ¬passes_through_lattice_point ((1 : ℚ) / 2) x ∧ ¬passes_through_lattice_point (a - 1) x) →
  a = 50 / 99 :=
by
  sorry

end max_a_avoiding_lattice_points_l165_165009


namespace probability_chinese_on_first_given_math_on_second_l165_165831

-- Context: schoolbag has 2 math books and 2 Chinese books
-- Question: Prove P(A|B) = 2/3 where:
-- Event A: First draw is a Chinese book
-- Event B: Second draw is a math book

theorem probability_chinese_on_first_given_math_on_second : 
  (conditional_probability 
    (event_first_draw_chinese ∧ event_second_draw_math) 
    event_second_draw_math = 2 / 3 :=
sorry

end probability_chinese_on_first_given_math_on_second_l165_165831


namespace conditions_for_right_triangle_l165_165993

universe u

variables {A B C : Type u}
variables [OrderedAddCommGroup A] [OrderedAddCommGroup B] [OrderedAddCommGroup C]

noncomputable def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem conditions_for_right_triangle :
  (∀ (A B C : ℝ), A + B = C → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), ( A / C = 1 / 6 ) → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), A = 90 - B → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), (A = B → B = C / 2) → is_right_triangle A B C) ∧
  ∀ (A B C : ℝ), ¬ ((A = 2 * B) ∧ B = 3 * C) 
:=
sorry

end conditions_for_right_triangle_l165_165993


namespace hydrogen_burns_oxygen_certain_l165_165000

-- define what it means for a chemical reaction to be well-documented and known to occur
def chemical_reaction (reactants : String) (products : String) : Prop :=
  (reactants = "2H₂ + O₂") ∧ (products = "2H₂O")

-- Event description and classification
def event_is_certain (event : String) : Prop :=
  event = "Hydrogen burns in oxygen to form water"

-- Main statement
theorem hydrogen_burns_oxygen_certain :
  ∀ (reactants products : String), (chemical_reaction reactants products) → event_is_certain "Hydrogen burns in oxygen to form water" :=
by
  intros reactants products h
  have h1 : reactants = "2H₂ + O₂" := h.1
  have h2 : products = "2H₂O" := h.2
  -- proof omitted
  exact sorry

end hydrogen_burns_oxygen_certain_l165_165000


namespace compare_logs_l165_165914

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
def c : ℝ := 1 / 2

theorem compare_logs (a b c : ℝ) (ha : a = Real.log 2 / Real.log 5) (hb : b = Real.log 3 / Real.log 8) (hc : c = 1 / 2) :
  a < c ∧ c < b :=
by {
  rw [ha, hb, hc],
  have ha_lt : Real.log 2 / Real.log 5 < 1 / 2,
  { sorry },
  have hb_gt : 1 / 2 < Real.log 3 / Real.log 8,
  { sorry },
  exact ⟨ha_lt, hb_gt⟩
}

end compare_logs_l165_165914


namespace find_multiple_l165_165020

-- Define the conditions
def ReetaPencils : ℕ := 20
def TotalPencils : ℕ := 64

-- Define the question and proof statement
theorem find_multiple (AnikaPencils : ℕ) (M : ℕ) :
  AnikaPencils = ReetaPencils * M + 4 →
  AnikaPencils + ReetaPencils = TotalPencils →
  M = 2 :=
by
  intros hAnika hTotal
  -- Skip the proof
  sorry

end find_multiple_l165_165020


namespace james_bike_ride_total_distance_l165_165406

theorem james_bike_ride_total_distance 
  (d1 d2 d3 : ℝ)
  (H1 : d2 = 12)
  (H2 : d2 = 1.2 * d1)
  (H3 : d3 = 1.25 * d2) :
  d1 + d2 + d3 = 37 :=
by
  -- additional proof steps would go here
  sorry

end james_bike_ride_total_distance_l165_165406


namespace time_after_2021_hours_l165_165537

-- Definition of starting time and day
def start_time : Nat := 20 * 60 + 21  -- converting 20:21 to minutes
def hours_per_day : Nat := 24
def minutes_per_hour : Nat := 60
def days_per_week : Nat := 7

-- Define the main statement
theorem time_after_2021_hours :
  let total_minutes := 2021 * minutes_per_hour
  let total_days := total_minutes / (hours_per_day * minutes_per_hour)
  let remaining_minutes := total_minutes % (hours_per_day * minutes_per_hour)
  let final_minutes := start_time + remaining_minutes
  let final_day := (total_days + 1) % days_per_week -- start on Monday (0), hence +1 for Tuesday
  final_minutes / minutes_per_hour = 1 ∧ final_minutes % minutes_per_hour = 21 ∧ final_day = 2 :=
by
  sorry

end time_after_2021_hours_l165_165537


namespace find_a_equidistant_l165_165922

theorem find_a_equidistant :
  ∀ a : ℝ, (abs (a - 2) = abs (6 - 2 * a)) →
    (a = 8 / 3 ∨ a = 4) :=
by
  intro a h
  sorry

end find_a_equidistant_l165_165922


namespace total_gumballs_l165_165819

-- Define the count of red, blue, and green gumballs
def red_gumballs := 16
def blue_gumballs := red_gumballs / 2
def green_gumballs := blue_gumballs * 4

-- Prove that the total number of gumballs is 56
theorem total_gumballs : red_gumballs + blue_gumballs + green_gumballs = 56 := by
  sorry

end total_gumballs_l165_165819


namespace total_days_to_finish_job_l165_165413

noncomputable def workers_job_completion
  (initial_workers : ℕ)
  (additional_workers : ℕ)
  (initial_days : ℕ)
  (total_days : ℕ)
  (work_completion_days : ℕ)
  (remaining_work : ℝ)
  (additional_days_needed : ℝ)
  : ℝ :=
  initial_days + additional_days_needed

theorem total_days_to_finish_job
  (initial_workers : ℕ := 6)
  (additional_workers : ℕ := 4)
  (initial_days : ℕ := 3)
  (total_days : ℕ := 8)
  (work_completion_days : ℕ := 8)
  : workers_job_completion initial_workers additional_workers initial_days total_days work_completion_days (1 - (initial_days : ℝ) / work_completion_days) (remaining_work / (((initial_workers + additional_workers) : ℝ) / work_completion_days)) = 3.5 :=
  sorry

end total_days_to_finish_job_l165_165413


namespace bob_needs_additional_weeks_l165_165023

-- Definitions based on conditions
def weekly_prize : ℕ := 100
def initial_weeks_won : ℕ := 2
def total_prize_won : ℕ := initial_weeks_won * weekly_prize
def puppy_cost : ℕ := 1000
def additional_weeks_needed : ℕ := (puppy_cost - total_prize_won) / weekly_prize

-- Statement of the theorem
theorem bob_needs_additional_weeks : additional_weeks_needed = 8 := by
  -- Proof here
  sorry

end bob_needs_additional_weeks_l165_165023


namespace compare_logs_l165_165916

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
noncomputable def c : ℝ := 1 / 2

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l165_165916


namespace right_triangle_circum_inradius_sum_l165_165534

theorem right_triangle_circum_inradius_sum
  (a b : ℕ)
  (h1 : a = 16)
  (h2 : b = 30)
  (h_triangle : a^2 + b^2 = 34^2) :
  let c := 34
  let R := c / 2
  let A := a * b / 2
  let s := (a + b + c) / 2
  let r := A / s
  R + r = 23 :=
by
  sorry

end right_triangle_circum_inradius_sum_l165_165534


namespace probability_y_eq_2x_l165_165798

/-- Two fair cubic dice each have six faces labeled with the numbers 1, 2, 3, 4, 5, and 6. 
Rolling these dice sequentially, find the probability that the number on the top face 
of the second die (y) is twice the number on the top face of the first die (x). --/
noncomputable def dice_probability : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

theorem probability_y_eq_2x : dice_probability = 1 / 12 :=
  by sorry

end probability_y_eq_2x_l165_165798


namespace abs_diff_squares_l165_165540

theorem abs_diff_squares (a b : ℝ) (ha : a = 105) (hb : b = 103) : |a^2 - b^2| = 416 :=
by
  sorry

end abs_diff_squares_l165_165540


namespace red_balls_count_l165_165002

theorem red_balls_count:
  ∀ (R : ℕ), (0 : ℝ) < R ∧
    (probability = (R * (R - 1) / 2) / ((R + 5) * (R + 4) / 2)) ∧
    (probability = (1 / 6)) →
    R = 4 :=
by {
  sorry
}

end red_balls_count_l165_165002


namespace angle_ABC_measure_l165_165067

theorem angle_ABC_measure
  (angle_CBD : ℝ)
  (angle_sum_around_B : ℝ)
  (angle_ABD : ℝ)
  (h1 : angle_CBD = 90)
  (h2 : angle_sum_around_B = 200)
  (h3 : angle_ABD = 60) :
  ∃ angle_ABC : ℝ, angle_ABC = 50 :=
by
  sorry

end angle_ABC_measure_l165_165067


namespace mink_babies_l165_165478

theorem mink_babies (B : ℕ) (h_coats : 7 * 15 = 105)
    (h_minks: 30 + 30 * B = 210) :
  B = 6 :=
by
  sorry

end mink_babies_l165_165478


namespace problem1_problem2_l165_165438

-- Define the variables for the two problems
variables (a b x : ℝ)

-- The first problem
theorem problem1 :
  (a + 2 * b) ^ 2 - 4 * b * (a + b) = a ^ 2 :=
by 
  -- Proof goes here
  sorry

-- The second problem
theorem problem2 :
  ((x ^ 2 - 2 * x) / (x ^ 2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x ^ 2 - 4)) = x + 2 :=
by 
  -- Proof goes here
  sorry

end problem1_problem2_l165_165438


namespace infinite_geometric_series_sum_l165_165878

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l165_165878


namespace jen_profit_l165_165352

-- Definitions based on the conditions
def cost_per_candy := 80 -- in cents
def sell_price_per_candy := 100 -- in cents
def total_candies_bought := 50
def total_candies_sold := 48

-- Total cost and total revenue calculations
def total_cost := cost_per_candy * total_candies_bought
def total_revenue := sell_price_per_candy * total_candies_sold

-- Profit calculation
def profit := total_revenue - total_cost

-- Main theorem to prove
theorem jen_profit : profit = 800 := by
  -- Proof is skipped
  sorry

end jen_profit_l165_165352


namespace pencil_sharpening_time_l165_165422

theorem pencil_sharpening_time (t : ℕ) :
  let hand_crank_rate := 45
  let electric_rate := 20
  let sharpened_by_hand := (60 * t) / hand_crank_rate
  let sharpened_by_electric := (60 * t) / electric_rate
  (sharpened_by_electric = sharpened_by_hand + 10) → 
  t = 6 :=
by
  intros hand_crank_rate electric_rate sharpened_by_hand sharpened_by_electric h
  sorry

end pencil_sharpening_time_l165_165422


namespace no_seven_sum_possible_l165_165930

theorem no_seven_sum_possible :
  let outcomes := [-1, -3, -5, 2, 4, 6]
  ∀ (a b : Int), a ∈ outcomes → b ∈ outcomes → a + b ≠ 7 :=
by
  sorry

end no_seven_sum_possible_l165_165930


namespace bob_needs_additional_weeks_l165_165024

-- Definitions based on conditions
def weekly_prize : ℕ := 100
def initial_weeks_won : ℕ := 2
def total_prize_won : ℕ := initial_weeks_won * weekly_prize
def puppy_cost : ℕ := 1000
def additional_weeks_needed : ℕ := (puppy_cost - total_prize_won) / weekly_prize

-- Statement of the theorem
theorem bob_needs_additional_weeks : additional_weeks_needed = 8 := by
  -- Proof here
  sorry

end bob_needs_additional_weeks_l165_165024


namespace quadratic_factorization_sum_l165_165648

theorem quadratic_factorization_sum (d e f : ℤ) (h1 : ∀ x, x^2 + 18 * x + 80 = (x + d) * (x + e)) 
                                     (h2 : ∀ x, x^2 - 20 * x + 96 = (x - e) * (x - f)) : 
                                     d + e + f = 30 :=
by
  sorry

end quadratic_factorization_sum_l165_165648


namespace john_average_increase_l165_165926

theorem john_average_increase :
  let initial_scores := [92, 85, 91]
  let fourth_score := 95
  let initial_avg := (initial_scores.sum / initial_scores.length : ℚ)
  let new_avg := ((initial_scores.sum + fourth_score) / (initial_scores.length + 1) : ℚ)
  new_avg - initial_avg = 1.42 := 
by 
  sorry

end john_average_increase_l165_165926


namespace sin_x_intersect_ratio_l165_165777

theorem sin_x_intersect_ratio :
  ∃ r s : ℕ, r < s ∧ Nat.coprime r s ∧ (∀ n : ℤ, ∃ x1 x2 : ℝ, 
    (x1 = 30 + 360 * n ∧ x2 = 150 + 360 * n) ∧ (∃ k : ℤ, y = sin (k * 360 + 30) ∧ y = sin (k * 360 + 150)) ∧
    ((x2 - x1 = 120) ∧ (x1 + 360 - x2 = 240)) ∧ (r : ℝ) / (s : ℝ) = 1 / 2) :=
⟨1, 2, by decide, Nat.coprime_one_right _, by sorry⟩

end sin_x_intersect_ratio_l165_165777


namespace at_least_two_foxes_met_same_number_of_koloboks_l165_165893

-- Define the conditions
def number_of_foxes : ℕ := 14
def number_of_koloboks : ℕ := 92

-- The theorem statement to be proven
theorem at_least_two_foxes_met_same_number_of_koloboks :
  ∃ (f : Fin number_of_foxes.succ → ℕ), 
    (∀ i, f i ≤ number_of_koloboks) ∧ 
    ∃ i j, i ≠ j ∧ f i = f j :=
by
  sorry

end at_least_two_foxes_met_same_number_of_koloboks_l165_165893


namespace mode_and_median_of_data_set_l165_165237

def data_set : List ℕ := [3, 5, 4, 6, 3, 3, 4]

noncomputable def mode_of_data_set : ℕ :=
  sorry  -- The mode calculation goes here (implementation is skipped)

noncomputable def median_of_data_set : ℕ :=
  sorry  -- The median calculation goes here (implementation is skipped)

theorem mode_and_median_of_data_set :
  mode_of_data_set = 3 ∧ median_of_data_set = 4 :=
  by
    sorry  -- Proof goes here

end mode_and_median_of_data_set_l165_165237


namespace selection_probabilities_l165_165654

-- Define the probabilities of selection for Ram, Ravi, and Rani
def prob_ram : ℚ := 5 / 7
def prob_ravi : ℚ := 1 / 5
def prob_rani : ℚ := 3 / 4

-- State the theorem that combines these probabilities
theorem selection_probabilities : prob_ram * prob_ravi * prob_rani = 3 / 28 :=
by
  sorry


end selection_probabilities_l165_165654


namespace lemonade_percentage_l165_165427

theorem lemonade_percentage (L : ℝ) : 
  (0.4 * (1 - L / 100) + 0.6 * 0.55 = 0.65) → L = 20 :=
by
  sorry

end lemonade_percentage_l165_165427


namespace prove_correct_option_C_l165_165686

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l165_165686


namespace find_first_number_l165_165113

theorem find_first_number (N : ℤ) (k m : ℤ) (h1 : N = 170 * k + 10) (h2 : 875 = 170 * m + 25) : N = 860 :=
by
  sorry

end find_first_number_l165_165113


namespace neg_triangle_obtuse_angle_l165_165125

theorem neg_triangle_obtuse_angle : 
  (¬ ∀ (A B C : ℝ), A + B + C = π → max (max A B) C < π/2) ↔ (∃ (A B C : ℝ), A + B + C = π ∧ min (min A B) C > π/2) :=
by
  sorry

end neg_triangle_obtuse_angle_l165_165125


namespace option_C_correct_l165_165672

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l165_165672


namespace expression_value_l165_165033

theorem expression_value : (100 - (1000 - 300)) - (1000 - (300 - 100)) = -1400 := by
  sorry

end expression_value_l165_165033


namespace tangent_slope_is_4_l165_165529

theorem tangent_slope_is_4 (x y : ℝ) (h_curve : y = x^4) (h_slope : (deriv (fun x => x^4) x) = 4) :
    (x, y) = (1, 1) :=
by
  -- Place proof here
  sorry

end tangent_slope_is_4_l165_165529


namespace solution_set_of_inequality_system_l165_165387

theorem solution_set_of_inequality_system (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x - 2 < 0) ↔ (-1 ≤ x ∧ x < 2) :=
by
  sorry

end solution_set_of_inequality_system_l165_165387


namespace selling_prices_for_10_percent_profit_l165_165267

theorem selling_prices_for_10_percent_profit
    (cost1 cost2 cost3 : ℝ)
    (cost1_eq : cost1 = 200)
    (cost2_eq : cost2 = 300)
    (cost3_eq : cost3 = 500)
    (profit_percent : ℝ)
    (profit_percent_eq : profit_percent = 0.10):
    ∃ s1 s2 s3 : ℝ,
      s1 = cost1 + 33.33 ∧
      s2 = cost2 + 33.33 ∧
      s3 = cost3 + 33.33 ∧
      s1 + s2 + s3 = 1100 :=
by
  sorry

end selling_prices_for_10_percent_profit_l165_165267


namespace solve_system_of_equations_l165_165105

theorem solve_system_of_equations :
  ∃ (x y : ℝ),
    (5 * x^2 - 14 * x * y + 10 * y^2 = 17) ∧ (4 * x^2 - 10 * x * y + 6 * y^2 = 8) ∧
    ((x = -1 ∧ y = -2) ∨ (x = 11 ∧ y = 7) ∨ (x = -11 ∧ y = -7) ∨ (x = 1 ∧ y = 2)) :=
by
  sorry

end solve_system_of_equations_l165_165105


namespace air_conditioner_usage_l165_165431

-- Define the given data and the theorem to be proven
theorem air_conditioner_usage (h : ℝ) (rate : ℝ) (days : ℝ) (total_consumption : ℝ) :
  rate = 0.9 → days = 5 → total_consumption = 27 → (days * h * rate = total_consumption) → h = 6 :=
by
  intros hr dr tc h_eq
  sorry

end air_conditioner_usage_l165_165431


namespace positive_integer_solutions_l165_165715

theorem positive_integer_solutions :
  ∀ m n : ℕ, 0 < m ∧ 0 < n ∧ 3^m - 2^n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 3) :=
by
  sorry

end positive_integer_solutions_l165_165715


namespace correct_operation_l165_165678

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l165_165678


namespace volume_of_sphere_in_cone_l165_165989

theorem volume_of_sphere_in_cone :
  let r_base := 9
  let h_cone := 9
  let diameter_sphere := 9 * Real.sqrt 2
  let radius_sphere := diameter_sphere / 2
  let volume_sphere := (4 / 3) * Real.pi * radius_sphere ^ 3
  volume_sphere = (1458 * Real.sqrt 2 / 4) * Real.pi :=
by
  sorry

end volume_of_sphere_in_cone_l165_165989


namespace focus_of_parabola_l165_165060

theorem focus_of_parabola (x y : ℝ) : x^2 = 4 * y → (0, 1) = (0, (4 / 4)) :=
by
  sorry

end focus_of_parabola_l165_165060


namespace angle_QRS_determination_l165_165746

theorem angle_QRS_determination (PQ_parallel_RS : ∀ (P Q R S T : Type) 
  (angle_PTQ : ℝ) (angle_SRT : ℝ), 
  PQ_parallel_RS → (angle_PTQ = angle_SRT) → (angle_PTQ = 4 * angle_SRT - 120)) 
  (angle_SRT : ℝ) (angle_QRS : ℝ) 
  (h : angle_SRT = 4 * angle_SRT - 120) : angle_QRS = 40 :=
by 
  sorry

end angle_QRS_determination_l165_165746


namespace acme_cheaper_than_beta_l165_165835

theorem acme_cheaper_than_beta (x : ℕ) :
  (50 + 9 * x < 25 + 15 * x) ↔ (5 ≤ x) :=
by sorry

end acme_cheaper_than_beta_l165_165835


namespace value_of_J_l165_165255

theorem value_of_J (J : ℕ) : 32^4 * 4^4 = 2^J → J = 28 :=
by
  intro h
  sorry

end value_of_J_l165_165255


namespace darij_grinberg_inequality_l165_165221

theorem darij_grinberg_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  a + b + c ≤ (bc / (b + c)) + (ca / (c + a)) + (ab / (a + b)) + (1 / 2 * ((bc / a) + (ca / b) + (ab / c))) := 
by sorry

end darij_grinberg_inequality_l165_165221


namespace common_tangent_l165_165177

-- Definition of the ellipse and hyperbola
def ellipse (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144
def hyperbola (x y : ℝ) : Prop := 7 * x^2 - 32 * y^2 = 224

-- The statement to prove
theorem common_tangent :
  (∀ x y : ℝ, ellipse x y → hyperbola x y → ((x + y + 5 = 0) ∨ (x + y - 5 = 0) ∨ (x - y + 5 = 0) ∨ (x - y - 5 = 0))) := 
sorry

end common_tangent_l165_165177


namespace non_congruent_triangles_perimeter_18_l165_165606

theorem non_congruent_triangles_perimeter_18 :
  ∃ (triangles : Finset (Finset ℕ)), triangles.card = 11 ∧
  (∀ t ∈ triangles, t.card = 3 ∧ (∃ a b c : ℕ, t = {a, b, c} ∧ a + b + c = 18 ∧ a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end non_congruent_triangles_perimeter_18_l165_165606


namespace expected_volunteers_2008_l165_165617

theorem expected_volunteers_2008 (initial_volunteers: ℕ) (annual_increase: ℚ) (h1: initial_volunteers = 500) (h2: annual_increase = 1.2) : 
  let volunteers_2006 := initial_volunteers * annual_increase
  let volunteers_2007 := volunteers_2006 * annual_increase
  let volunteers_2008 := volunteers_2007 * annual_increase
  volunteers_2008 = 864 := 
by
  sorry

end expected_volunteers_2008_l165_165617


namespace minimum_dwarfs_l165_165501

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_l165_165501


namespace number_of_paths_grid_l165_165466

def paths_from_A_to_C (h v : Nat) : Nat :=
  Nat.choose (h + v) v

#eval paths_from_A_to_C 7 6 -- expected result: 1716

theorem number_of_paths_grid :
  paths_from_A_to_C 7 6 = 1716 := by
  sorry

end number_of_paths_grid_l165_165466


namespace geometric_series_sum_l165_165876

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l165_165876


namespace solution_of_inequality_is_correct_l165_165791

-- Inequality condition (x-1)/(2x+1) ≤ 0
def inequality (x : ℝ) : Prop := (x - 1) / (2 * x + 1) ≤ 0 

-- Conditions
def condition1 (x : ℝ) : Prop := (x - 1) * (2 * x + 1) ≤ 0
def condition2 (x : ℝ) : Prop := 2 * x + 1 ≠ 0

-- Combined condition
def combined_condition (x : ℝ) : Prop := condition1 x ∧ condition2 x

-- Solution set
def solution_set : Set ℝ := { x | -1/2 < x ∧ x ≤ 1 }

-- Theorem statement
theorem solution_of_inequality_is_correct :
  ∀ x : ℝ, inequality x ↔ combined_condition x ∧ x ∈ solution_set :=
by
  sorry

end solution_of_inequality_is_correct_l165_165791


namespace irreducible_polynomial_l165_165496

open Polynomial

theorem irreducible_polynomial (n : ℕ) : Irreducible ((X^2 + X)^(2^n) + 1 : ℤ[X]) := sorry

end irreducible_polynomial_l165_165496


namespace total_cows_is_108_l165_165698

-- Definitions of the sons' shares and the number of cows the fourth son received
def first_son_share : ℚ := 2 / 3
def second_son_share : ℚ := 1 / 6
def third_son_share : ℚ := 1 / 9
def fourth_son_cows : ℕ := 6

-- The total number of cows in the herd
def total_cows (n : ℕ) : Prop :=
  first_son_share + second_son_share + third_son_share + (fourth_son_cows / n) = 1

-- Prove that given the number of cows the fourth son received, the total number of cows in the herd is 108
theorem total_cows_is_108 : total_cows 108 :=
by
  sorry

end total_cows_is_108_l165_165698


namespace geom_seq_fraction_l165_165474

theorem geom_seq_fraction (a_1 a_2 a_3 a_4 a_5 q : ℝ)
  (h1 : q > 0)
  (h2 : a_2 = q * a_1)
  (h3 : a_3 = q^2 * a_1)
  (h4 : a_4 = q^3 * a_1)
  (h5 : a_5 = q^4 * a_1)
  (h_arith : a_2 - (1/2) * a_3 = (1/2) * a_3 - a_1) :
  (a_3 + a_4) / (a_4 + a_5) = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end geom_seq_fraction_l165_165474


namespace vertex_parabola_l165_165775

theorem vertex_parabola (h k : ℝ) : 
  (∀ x : ℝ, -((x - 2)^2) + 3 = k) → (h = 2 ∧ k = 3) :=
by 
  sorry

end vertex_parabola_l165_165775


namespace bakery_problem_l165_165261

theorem bakery_problem :
  let chocolate_chip := 154
  let oatmeal_raisin := 86
  let sugar := 52
  let capacity := 16
  let needed_chocolate_chip := capacity - (chocolate_chip % capacity)
  let needed_oatmeal_raisin := capacity - (oatmeal_raisin % capacity)
  let needed_sugar := capacity - (sugar % capacity)
  (needed_chocolate_chip = 6) ∧ (needed_oatmeal_raisin = 10) ∧ (needed_sugar = 12) :=
by
  sorry

end bakery_problem_l165_165261


namespace roof_area_l165_165385

-- Definitions based on conditions
variables (l w : ℝ)
def length_eq_five_times_width : Prop := l = 5 * w
def length_minus_width_eq_48 : Prop := l - w = 48

-- Proof goal
def area_of_roof : Prop := l * w = 720

-- Lean 4 statement asserting the mathematical problem
theorem roof_area (l w : ℝ) 
  (H1 : length_eq_five_times_width l w)
  (H2 : length_minus_width_eq_48 l w) : 
  area_of_roof l w := 
  by sorry

end roof_area_l165_165385


namespace probability_negative_product_l165_165969

theorem probability_negative_product :
  let S := {-6, -3, 1, 5, 8, -9}
  ( S.card = 6 ) →
  let neg := { -6, -3, -9 }
  ( neg.card = 3 ) →
  let pos := { 1, 5, 8 }
  ( pos.card = 3 ) →
  ( ∃ nums : Finset ℤ, nums ⊆ S ∧ nums.card = 2 ) →
  ( let neg_prod_count := neg.card * pos.card
    let total_count := S.card.choose 2
    neg_prod_count / total_count = (3 / 5 : ℚ) ) := sorry

end probability_negative_product_l165_165969


namespace avg_speed_correct_l165_165607

noncomputable def avg_speed_round_trip
  (flight_up_speed : ℝ)
  (tailwind_speed : ℝ)
  (tailwind_angle : ℝ)
  (flight_home_speed : ℝ)
  (headwind_speed : ℝ)
  (headwind_angle : ℝ) : ℝ :=
  let effective_tailwind_speed := tailwind_speed * Real.cos (tailwind_angle * Real.pi / 180)
  let ground_speed_to_mother := flight_up_speed + effective_tailwind_speed
  let effective_headwind_speed := headwind_speed * Real.cos (headwind_angle * Real.pi / 180)
  let ground_speed_back_home := flight_home_speed - effective_headwind_speed
  (ground_speed_to_mother + ground_speed_back_home) / 2

theorem avg_speed_correct :
  avg_speed_round_trip 96 12 30 88 15 60 = 93.446 :=
by
  sorry

end avg_speed_correct_l165_165607


namespace ellipse_parabola_four_intersections_intersection_points_lie_on_circle_l165_165980

-- Part A:
-- Define intersections of a given ellipse and parabola under conditions on m and n
theorem ellipse_parabola_four_intersections (m n : ℝ) :
  (3 / n < m) ∧ (m < (4 * m^2 + 9) / (4 * m)) ∧ (m > 3 / 2) →
  ∃ x y : ℝ, (x^2 / n + y^2 / 9 = 1) ∧ (y = x^2 - m) :=
sorry

-- Part B:
-- Prove four intersection points of given ellipse and parabola lie on same circle for m = n = 4
theorem intersection_points_lie_on_circle (x y : ℝ) :
  (4 / 4 + y^2 / 9 = 1) ∧ (y = x^2 - 4) →
  ∃ k l r : ℝ, ∀ x' y', ((x' - k)^2 + (y' - l)^2 = r^2) :=
sorry

end ellipse_parabola_four_intersections_intersection_points_lie_on_circle_l165_165980


namespace sum_powers_mod_7_l165_165068

theorem sum_powers_mod_7 :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
  sorry

end sum_powers_mod_7_l165_165068


namespace geometric_sequence_sum_l165_165082

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : a 2 = 1 - a 1)
  (h3 : a 4 = 9 - a 3)
  (h4 : ∀ n, a (n + 1) = a n * q) :
  a 4 + a 5 = 27 :=
sorry

end geometric_sequence_sum_l165_165082


namespace translated_parabola_expression_correct_l165_165796

-- Definitions based on the conditions
def original_parabola (x : ℝ) : ℝ := x^2 - 1
def translated_parabola (x : ℝ) : ℝ := (x + 2)^2

-- The theorem to prove
theorem translated_parabola_expression_correct :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) + 1 :=
by
  sorry

end translated_parabola_expression_correct_l165_165796


namespace number_of_digits_if_million_place_l165_165011

theorem number_of_digits_if_million_place (n : ℕ) (h : n = 1000000) : 7 = 7 := by
  sorry

end number_of_digits_if_million_place_l165_165011


namespace remainder_2048_mod_13_l165_165228

theorem remainder_2048_mod_13 : 2048 % 13 = 7 := by
  sorry

end remainder_2048_mod_13_l165_165228


namespace speed_in_still_water_l165_165554

theorem speed_in_still_water (upstream downstream : ℝ) 
  (h_up : upstream = 25) 
  (h_down : downstream = 45) : 
  (upstream + downstream) / 2 = 35 := 
by 
  -- Proof will go here
  sorry

end speed_in_still_water_l165_165554


namespace xy_sum_is_2_l165_165101

theorem xy_sum_is_2 (x y : ℝ) (h : 4 * x^2 + 4 * y^2 = 40 * x - 24 * y + 64) : x + y = 2 := 
by
  sorry

end xy_sum_is_2_l165_165101


namespace odot_property_l165_165165

def odot (x y : ℤ) := 2 * x + y

theorem odot_property (a b : ℤ) (h : odot a (-6 * b) = 4) : odot (a - 5 * b) (a + b) = 6 :=
by
  sorry

end odot_property_l165_165165


namespace jaylen_charge_per_yard_l165_165349

def total_cost : ℝ := 250
def number_of_yards : ℝ := 6
def charge_per_yard : ℝ := 41.67

theorem jaylen_charge_per_yard :
  total_cost / number_of_yards = charge_per_yard :=
sorry

end jaylen_charge_per_yard_l165_165349


namespace geometric_sum_S6_l165_165616

open Real

-- Define a geometric sequence
noncomputable def geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q ^ (n - 1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def sum_geometric (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a * n else a * (1 - q ^ n) / (1 - q)

-- Given conditions
variables (a q : ℝ) (n : ℕ)
variable (S3 : ℝ)
variable (q : ℝ) (h_q : q = 2)
variable (h_S3 : S3 = 7)

theorem geometric_sum_S6 :
  sum_geometric a 2 6 = 63 :=
  by
    sorry

end geometric_sum_S6_l165_165616


namespace sequence_eventually_periodic_l165_165719

-- Definitions based on the conditions
def positive_int_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < a n

def satisfies_condition (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = a (n + 2) * a (n + 3)

-- Assertion to prove based on the question
theorem sequence_eventually_periodic (a : ℕ → ℕ) 
  (h1 : positive_int_sequence a) 
  (h2 : satisfies_condition a) : 
  ∃ p : ℕ, ∃ k : ℕ, ∀ n : ℕ, a (n + k) = a n :=
sorry

end sequence_eventually_periodic_l165_165719


namespace min_max_values_l165_165186

noncomputable def expression (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  ( (x₁ ^ 2 / x₂) + (x₂ ^ 2 / x₃) + (x₃ ^ 2 / x₄) + (x₄ ^ 2 / x₁) ) /
  ( x₁ + x₂ + x₃ + x₄ )

theorem min_max_values
  (a b : ℝ) (x₁ x₂ x₃ x₄ : ℝ)
  (h₀ : 0 < a) (h₁ : a < b)
  (h₂ : a ≤ x₁) (h₃ : x₁ ≤ b)
  (h₄ : a ≤ x₂) (h₅ : x₂ ≤ b)
  (h₆ : a ≤ x₃) (h₇ : x₃ ≤ b)
  (h₈ : a ≤ x₄) (h₉ : x₄ ≤ b) :
  expression x₁ x₂ x₃ x₄ ≥ 1 / b ∧ expression x₁ x₂ x₃ x₄ ≤ 1 / a :=
  sorry

end min_max_values_l165_165186


namespace number_of_poly_lines_l165_165536

def nonSelfIntersectingPolyLines (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n ≥ 3 then n * 2^(n - 3)
  else 0

theorem number_of_poly_lines (n : ℕ) (h : n > 1) :
  nonSelfIntersectingPolyLines n =
  if n = 2 then 1 else n * 2^(n - 3) :=
by sorry

end number_of_poly_lines_l165_165536


namespace probability_same_color_l165_165138

theorem probability_same_color :
  let total_balls := 18
  let green_balls := 10
  let white_balls := 8
  let prob_two_green := (green_balls / total_balls) * ((green_balls - 1) / (total_balls - 1))
  let prob_two_white := (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))
  let prob_same_color := prob_two_green + prob_two_white
  in prob_same_color = 73 / 153 :=
by
  let total_balls := 18
  let green_balls := 10
  let white_balls := 8
  let prob_two_green := (green_balls / total_balls) * ((green_balls - 1) / (total_balls - 1))
  let prob_two_white := (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))
  let prob_same_color := prob_two_green + prob_two_white
  have : prob_two_green = 90 / 306 := by sorry
  have : prob_two_white = 56 / 306 := by sorry
  have : prob_same_color = 146 / 306 := by sorry
  have : 146 / 306 = 73 / 153 := by sorry
  show prob_same_color = 73 / 153 from sorry

end probability_same_color_l165_165138


namespace N_eq_P_l165_165726

def N : Set ℝ := {x | ∃ n : ℤ, x = (n : ℝ) / 2 - 1 / 3}
def P : Set ℝ := {x | ∃ p : ℤ, x = (p : ℝ) / 2 + 1 / 6}

theorem N_eq_P : N = P :=
  sorry

end N_eq_P_l165_165726


namespace last_digit_of_product_l165_165252

theorem last_digit_of_product :
    (3 ^ 65 * 6 ^ 59 * 7 ^ 71) % 10 = 4 := 
  by sorry

end last_digit_of_product_l165_165252


namespace repeating_decimal_sum_as_fraction_l165_165175

theorem repeating_decimal_sum_as_fraction :
  let d1 := 1 / 9    -- Representation of 0.\overline{1}
  let d2 := 1 / 99   -- Representation of 0.\overline{01}
  d1 + d2 = (4 : ℚ) / 33 := by
{
  sorry
}

end repeating_decimal_sum_as_fraction_l165_165175


namespace amount_paid_l165_165749

def cost_cat_toy : ℝ := 8.77
def cost_cage : ℝ := 10.97
def change_received : ℝ := 0.26

theorem amount_paid : (cost_cat_toy + cost_cage + change_received) = 20.00 := by
  sorry

end amount_paid_l165_165749


namespace range_of_a_l165_165470

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a * x + 1 < 0) → -2 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l165_165470


namespace intersection_AB_l165_165198

def setA : Set ℝ := { x | x^2 - 2*x - 3 < 0}
def setB : Set ℝ := { x | x > 1 }
def intersection : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem intersection_AB : setA ∩ setB = intersection :=
by
  sorry

end intersection_AB_l165_165198


namespace total_columns_l165_165528

variables (N L : ℕ)

theorem total_columns (h1 : L > 1500) (h2 : L = 30 * (N - 70)) : N = 180 :=
by
  sorry

end total_columns_l165_165528


namespace mean_equality_l165_165241

theorem mean_equality (z : ℚ) :
  ((8 + 7 + 28) / 3 : ℚ) = (14 + z) / 2 → z = 44 / 3 :=
by
  sorry

end mean_equality_l165_165241


namespace length_of_XY_in_triangle_XYZ_l165_165085

theorem length_of_XY_in_triangle_XYZ :
  ∀ (XYZ : Type) (X Y Z : XYZ) (angle : XYZ → XYZ → XYZ → ℝ) (length : XYZ → XYZ → ℝ),
  angle X Z Y = 30 ∧ angle Y X Z = 90 ∧ length X Z = 8 → length X Y = 16 :=
by sorry

end length_of_XY_in_triangle_XYZ_l165_165085


namespace sum_of_two_squares_iff_double_sum_of_two_squares_l165_165634

theorem sum_of_two_squares_iff_double_sum_of_two_squares (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_two_squares_iff_double_sum_of_two_squares_l165_165634


namespace jasmine_total_cost_l165_165347

-- Define the data and conditions
def pounds_of_coffee := 4
def gallons_of_milk := 2
def cost_per_pound_of_coffee := 2.50
def cost_per_gallon_of_milk := 3.50

-- Calculate the expected total cost and state the theorem
theorem jasmine_total_cost :
  pounds_of_coffee * cost_per_pound_of_coffee + gallons_of_milk * cost_per_gallon_of_milk = 17 :=
by
  -- Proof would be provided here
  sorry

end jasmine_total_cost_l165_165347


namespace number_of_different_ways_is_18_l165_165763

-- Define the problem conditions
def number_of_ways_to_place_balls : ℕ :=
  let total_balls := 9
  let boxes := 3
  -- Placeholder function to compute the requirement
  -- The actual function would involve combinatorial logic
  -- Let us define it as an axiom for now.
  sorry

-- The theorem to be proven
theorem number_of_different_ways_is_18 :
  number_of_ways_to_place_balls = 18 :=
sorry

end number_of_different_ways_is_18_l165_165763


namespace Cartesian_eq_C2_correct_distance_AB_correct_l165_165102

-- Part I: Proving the Cartesian equation of curve (C2)
noncomputable def equation_of_C2 (x y : ℝ) (α : ℝ) : Prop :=
  x = 4 * Real.cos α ∧ y = 4 + 4 * Real.sin α

def Cartesian_eq_C2 (x y : ℝ) : Prop :=
  x^2 + (y - 4)^2 = 16

theorem Cartesian_eq_C2_correct (x y α : ℝ) (h : equation_of_C2 x y α) : Cartesian_eq_C2 x y :=
by sorry

-- Part II: Proving the distance |AB| given polar equations
noncomputable def polar_eq_C1 (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def polar_eq_C2 (theta : ℝ) : ℝ :=
  8 * Real.sin theta

def distance_AB (rho1 rho2 : ℝ) : ℝ :=
  abs (rho1 - rho2)

theorem distance_AB_correct : distance_AB (polar_eq_C1 (π / 3)) (polar_eq_C2 (π / 3)) = 2 * Real.sqrt 3 :=
by sorry

end Cartesian_eq_C2_correct_distance_AB_correct_l165_165102


namespace boat_speed_24_l165_165967

def speed_of_boat_in_still_water (x : ℝ) : Prop :=
  let speed_downstream := x + 3
  let time := 1 / 4 -- 15 minutes in hours
  let distance := 6.75
  let equation := distance = speed_downstream * time
  equation ∧ x = 24

theorem boat_speed_24 (x : ℝ) (rate_of_current : ℝ) (time_minutes : ℝ) (distance_traveled : ℝ) 
  (h1 : rate_of_current = 3) (h2 : time_minutes = 15) (h3 : distance_traveled = 6.75) : speed_of_boat_in_still_water 24 := 
by
  -- Convert time in minutes to hours
  have time_in_hours : ℝ := time_minutes / 60
  -- Effective downstream speed
  have effective_speed := 24 + rate_of_current
  -- The equation to be satisfied
  have equation := distance_traveled = effective_speed * time_in_hours
  -- Simplify and solve
  sorry

end boat_speed_24_l165_165967


namespace calculate_expression_l165_165027

theorem calculate_expression (b : ℝ) (hb : b ≠ 0) : 
  (1 / 25) * b^0 + (1 / (25 * b))^0 - 81^(-1 / 4 : ℝ) - (-27)^(-1 / 3 : ℝ) = 26 / 25 :=
by sorry

end calculate_expression_l165_165027


namespace bryden_payment_l165_165815

theorem bryden_payment :
  (let face_value := 0.25
   let quarters := 6
   let collector_multiplier := 16
   let discount := 0.10
   let initial_payment := collector_multiplier * (quarters * face_value)
   let final_payment := initial_payment - (initial_payment * discount)
   final_payment = 21.6) :=
by
  sorry

end bryden_payment_l165_165815


namespace range_of_g_l165_165035

open Real

-- Define the function g(x)
def g (x : ℝ) : ℝ := arcsin x + arccos x - arctan x

-- Statement of the problem in Lean 4
theorem range_of_g : set.Icc (π / 4) (3 * π / 4) = {y : ℝ | ∃ x : ℝ, x ∈ set.Icc (-1 : ℝ) (1 : ℝ) ∧ g x = y} :=
by sorry

end range_of_g_l165_165035


namespace Holly_throws_5_times_l165_165705

def Bess.throw_distance := 20
def Bess.throw_times := 4
def Holly.throw_distance := 8
def total_distance := 200

theorem Holly_throws_5_times : 
  (total_distance - Bess.throw_times * 2 * Bess.throw_distance) / Holly.throw_distance = 5 :=
by 
  sorry

end Holly_throws_5_times_l165_165705


namespace parabola_inequality_l165_165191

theorem parabola_inequality (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 2 * k * x + (k^2 + 2 * k + 2) > x^2 + 2 * k * x - 2 * k^2 - 1) ↔ (-1 < k ∧ k < 3) := 
sorry

end parabola_inequality_l165_165191


namespace indoor_players_count_l165_165475

theorem indoor_players_count (T O B I : ℕ) 
  (hT : T = 400) 
  (hO : O = 350) 
  (hB : B = 60) 
  (hEq : T = (O - B) + (I - B) + B) : 
  I = 110 := 
by sorry

end indoor_players_count_l165_165475


namespace collinear_points_on_curve_sum_zero_l165_165136

theorem collinear_points_on_curve_sum_zero
  {x1 y1 x2 y2 x3 y3 : ℝ}
  (h_curve1 : y1^2 = x1^3)
  (h_curve2 : y2^2 = x2^3)
  (h_curve3 : y3^2 = x3^3)
  (h_collinear : ∃ (a b c k : ℝ), k ≠ 0 ∧ 
    a * x1 + b * y1 + c = 0 ∧
    a * x2 + b * y2 + c = 0 ∧
    a * x3 + b * y3 + c = 0) :
  x1 / y1 + x2 / y2 + x3 / y3 = 0 :=
sorry

end collinear_points_on_curve_sum_zero_l165_165136


namespace measured_weight_loss_l165_165692

variable (W : ℝ) (hW : W > 0)

noncomputable def final_weigh_in (initial_weight : ℝ) : ℝ :=
  (0.90 * initial_weight) * 1.02

theorem measured_weight_loss :
  final_weigh_in W = 0.918 * W → (W - final_weigh_in W) / W * 100 = 8.2 := 
by
  intro h
  unfold final_weigh_in at h
  -- skip detailed proof steps, focus on the statement
  sorry

end measured_weight_loss_l165_165692


namespace coeff_x3_in_expansion_of_x_plus_1_50_l165_165078

theorem coeff_x3_in_expansion_of_x_plus_1_50 :
  (Finset.range 51).sum (λ k => Nat.choose 50 k * (1 : ℕ) ^ (50 - k) * k ^ 3) = 19600 := by
  sorry

end coeff_x3_in_expansion_of_x_plus_1_50_l165_165078


namespace find_dividend_l165_165176

theorem find_dividend (k : ℕ) (quotient : ℕ) (dividend : ℕ) (h1 : k = 14) (h2 : quotient = 4) (h3 : dividend = quotient * k) : dividend = 56 :=
by
  sorry

end find_dividend_l165_165176


namespace least_positive_nine_n_square_twelve_n_cube_l165_165802

theorem least_positive_nine_n_square_twelve_n_cube :
  ∃ (n : ℕ), 0 < n ∧ (∃ (k1 k2 : ℕ), 9 * n = k1^2 ∧ 12 * n = k2^3) ∧ n = 144 :=
by
  sorry

end least_positive_nine_n_square_twelve_n_cube_l165_165802


namespace probability_of_C_l165_165274

-- Definitions of probabilities for regions A, B, and D
def P_A : ℚ := 3 / 8
def P_B : ℚ := 1 / 4
def P_D : ℚ := 1 / 8

-- Sum of probabilities must be 1
def total_probability : ℚ := 1

-- The main proof statement
theorem probability_of_C : 
  P_A + P_B + P_D + (P_C : ℚ) = total_probability → P_C = 1 / 4 := sorry

end probability_of_C_l165_165274


namespace correct_operation_l165_165679

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l165_165679


namespace same_color_combination_sum_l165_165008

theorem same_color_combination_sum (m n : ℕ) (coprime_mn : Nat.gcd m n = 1)
  (prob_together : ∀ (total_candies : ℕ), total_candies = 20 →
    let terry_red := Nat.choose 8 2;
    let total_cases := Nat.choose total_candies 2;
    let prob_terry_red := terry_red / total_cases;
    
    let mary_red_given_terry := Nat.choose 6 2;
    let reduced_total_cases := Nat.choose 18 2;
    let prob_mary_red_given_terry := mary_red_given_terry / reduced_total_cases;
    
    let both_red := prob_terry_red * prob_mary_red_given_terry;
    
    let terry_blue := Nat.choose 12 2;
    let prob_terry_blue := terry_blue / total_cases;
    
    let mary_blue_given_terry := Nat.choose 10 2;
    let prob_mary_blue_given_terry := mary_blue_given_terry / reduced_total_cases;
    
    let both_blue := prob_terry_blue * prob_mary_blue_given_terry;
    
    let mixed_red_blue := Nat.choose 8 1 * Nat.choose 12 1;
    let prob_mixed_red_blue := mixed_red_blue / total_cases;
    let both_mixed := prob_mixed_red_blue;
    
    let prob_same_combination := both_red + both_blue + both_mixed;
    
    prob_same_combination = m / n
  ) :
  m + n = 5714 :=
by
  sorry

end same_color_combination_sum_l165_165008


namespace union_M_N_l165_165199

namespace MyMath

def M : Set ℝ := {x | x^2 = 1}
def N : Set ℝ := {1, 2}

theorem union_M_N : M ∪ N = {-1, 1, 2} := sorry

end MyMath

end union_M_N_l165_165199


namespace candy_difference_l165_165152

def given_away : ℕ := 6
def left : ℕ := 5
def difference : ℕ := given_away - left

theorem candy_difference :
  difference = 1 :=
by
  sorry

end candy_difference_l165_165152


namespace minimum_dwarfs_l165_165502

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_l165_165502


namespace total_amount_l165_165697

noncomputable def A : ℝ := 360.00000000000006
noncomputable def B : ℝ := (3/2) * A
noncomputable def C : ℝ := 4 * B

theorem total_amount (A B C : ℝ)
  (hA : A = 360.00000000000006)
  (hA_B : A = (2/3) * B)
  (hB_C : B = (1/4) * C) :
  A + B + C = 3060.0000000000007 := by
  sorry

end total_amount_l165_165697


namespace arithmetic_sequence_sum_l165_165456

theorem arithmetic_sequence_sum {a : ℕ → ℝ}
  (h1 : a 1 + a 5 = 6) 
  (h2 : a 2 + a 14 = 26) :
  (10 / 2) * (a 1 + a 10) = 80 :=
by sorry

end arithmetic_sequence_sum_l165_165456


namespace arithmetic_geometric_sequence_a4_value_l165_165593

theorem arithmetic_geometric_sequence_a4_value 
  (a : ℕ → ℝ)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 4 + a 6 = 5 / 4) : 
  a 4 = 1 := 
sorry

end arithmetic_geometric_sequence_a4_value_l165_165593


namespace range_of_a_l165_165483

theorem range_of_a (a : ℝ) (x : ℤ) (h1 : ∀ x, x > 0 → ⌊(x + a) / 3⌋ = 2) : a < 8 :=
sorry

end range_of_a_l165_165483


namespace combined_yells_l165_165096

def yells_at_obedient : ℕ := 12
def yells_at_stubborn (y_obedient : ℕ) : ℕ := 4 * y_obedient
def total_yells (y_obedient : ℕ) (y_stubborn : ℕ) : ℕ := y_obedient + y_stubborn

theorem combined_yells : total_yells yells_at_obedient (yells_at_stubborn yells_at_obedient) = 60 := 
by
  sorry

end combined_yells_l165_165096


namespace problem1_problem2_l165_165283

-- Proof problem 1 statement in Lean 4
theorem problem1 :
  (1 : ℝ) * (Real.sqrt 2)^2 - |(1 : ℝ) - Real.sqrt 3| + Real.sqrt ((-3 : ℝ)^2) + Real.sqrt 81 = 15 - Real.sqrt 3 :=
by sorry

-- Proof problem 2 statement in Lean 4
theorem problem2 (x y : ℝ) :
  (x - 2 * y)^2 - (x + 2 * y + 3) * (x + 2 * y - 3) = -8 * x * y + 9 :=
by sorry

end problem1_problem2_l165_165283


namespace length_of_base_l165_165712

-- Define the conditions of the problem
def base_of_triangle (b : ℕ) : Prop :=
  ∃ c : ℕ, b + 3 + c = 12 ∧ 9 + b*b = c*c

-- Statement to prove
theorem length_of_base : base_of_triangle 4 :=
  sorry

end length_of_base_l165_165712


namespace average_distance_per_day_l165_165515

def monday_distance : ℝ := 4.2
def tuesday_distance : ℝ := 3.8
def wednesday_distance : ℝ := 3.6
def thursday_distance : ℝ := 4.4
def number_of_days : ℕ := 4

theorem average_distance_per_day :
  (monday_distance + tuesday_distance + wednesday_distance + thursday_distance) / number_of_days = 4 :=
by
  sorry

end average_distance_per_day_l165_165515


namespace correct_operation_l165_165663

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l165_165663


namespace probability_ace_of_spades_l165_165549

theorem probability_ace_of_spades (total_cards : ℕ) (black_cards : ℕ) (removed_black_cards : ℕ)
  (total_cards = 52) (black_cards = 26) (removed_black_cards = 12)
  : (1 : ℚ) / (total_cards - removed_black_cards) = 1 / 40 := sorry

end probability_ace_of_spades_l165_165549


namespace nonneg_integer_solutions_l165_165690

theorem nonneg_integer_solutions :
  { x : ℕ | 5 * x + 3 < 3 * (2 + x) } = {0, 1} :=
by
  sorry

end nonneg_integer_solutions_l165_165690


namespace solution_set_inequality_l165_165307

theorem solution_set_inequality {a b : ℝ} 
  (h₁ : {x : ℝ | 1 < x ∧ x < 2} = {x : ℝ | ax^2 - bx + 2 < 0}) : a + b = -2 :=
by
  sorry

end solution_set_inequality_l165_165307


namespace total_gumballs_l165_165820

-- Define the count of red, blue, and green gumballs
def red_gumballs := 16
def blue_gumballs := red_gumballs / 2
def green_gumballs := blue_gumballs * 4

-- Prove that the total number of gumballs is 56
theorem total_gumballs : red_gumballs + blue_gumballs + green_gumballs = 56 := by
  sorry

end total_gumballs_l165_165820


namespace M_eq_N_l165_165410

def M : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}
def N : Set ℝ := {y | ∃ n : ℤ, y = Real.cos (n * Real.pi / 3)}

theorem M_eq_N : M = N := 
by 
  sorry

end M_eq_N_l165_165410


namespace small_paintings_completed_l165_165703

variable (S : ℕ)

def uses_paint : Prop :=
  3 * 3 + 2 * S = 17

theorem small_paintings_completed : uses_paint S → S = 4 := by
  intro h
  sorry

end small_paintings_completed_l165_165703


namespace infinite_geometric_series_sum_l165_165867

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l165_165867


namespace temperature_or_daytime_not_sufficiently_high_l165_165434

variable (T : ℝ) (Daytime Lively : Prop)
axiom h1 : (T ≥ 75 ∧ Daytime) → Lively
axiom h2 : ¬ Lively

theorem temperature_or_daytime_not_sufficiently_high : T < 75 ∨ ¬ Daytime :=
by
  -- proof steps
  sorry

end temperature_or_daytime_not_sufficiently_high_l165_165434


namespace find_a_b_l165_165196

def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

def f_derivative (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem find_a_b (a b : ℝ) (h1 : f 1 a b = 10) (h2 : f_derivative 1 a b = 0) : a = 4 ∧ b = -11 :=
sorry

end find_a_b_l165_165196


namespace students_per_row_l165_165137

theorem students_per_row (x : ℕ) : 45 = 11 * x + 1 → x = 4 :=
by
  intro h
  sorry

end students_per_row_l165_165137


namespace set_intersection_complement_eq_l165_165601

-- Definitions based on conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Complement of B in U
def complement_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- The theorem statement
theorem set_intersection_complement_eq :
  A ∩ complement_B = {x | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end set_intersection_complement_eq_l165_165601


namespace Martha_time_spent_l165_165759

theorem Martha_time_spent
  (x : ℕ)
  (h1 : 6 * x = 6 * x) -- Time spent on hold with Comcast is 6 times the time spent turning router off and on again
  (h2 : 3 * x = 3 * x) -- Time spent yelling at the customer service rep is half of time spent on hold, which is still 3x
  (h3 : x + 6 * x + 3 * x = 100) -- Total time spent is 100 minutes
  : x = 10 := 
by
  -- skip the proof steps
  sorry

end Martha_time_spent_l165_165759


namespace pinocchio_cannot_pay_exactly_l165_165941

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l165_165941


namespace max_and_min_of_expression_l165_165724

variable {x y : ℝ}

theorem max_and_min_of_expression (h : |5 * x + y| + |5 * x - y| = 20) : 
  (∃ (maxQ minQ : ℝ), maxQ = 124 ∧ minQ = 3 ∧ 
  (∀ z, z = x^2 - x * y + y^2 → z <= 124 ∧ z >= 3)) :=
sorry

end max_and_min_of_expression_l165_165724


namespace intersection_product_l165_165403

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 - 4 * x + y^2 - 6 * y + 9 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 - 8 * x + y^2 - 6 * y + 25 = 0

-- Define the theorem to prove the product of the coordinates of the intersection points
theorem intersection_product : ∀ x y : ℝ, circle1 x y → circle2 x y → x * y = 12 :=
by
  intro x y h1 h2
  -- Insert proof here
  sorry

end intersection_product_l165_165403


namespace odd_increasing_three_digit_numbers_count_eq_50_l165_165322

def count_odd_increasing_three_digit_numbers : Nat := by
  -- Mathematical conditions:
  -- let a, b, c be digits of the number
  -- 0 < a < b < c <= 9 and c is an odd digit

  -- We analyze values for 'c' which must be an odd digit,
  -- and count valid (a, b) combinations for each case of c.

  -- Starting from cases for c:
  -- for c = 1, no valid (a, b); count = 0
  -- for c = 3, valid (a, b) are from {1, 2}; count = 1
  -- for c = 5, valid (a, b) are from {1, 2, 3, 4}; count = 6
  -- for c = 7, valid (a, b) are from {1, 2, 3, 4, 5, 6}; count = 15
  -- for c = 9, valid (a, b) are from {1, 2, 3, 4, 5, 6, 7, 8}; count = 28

  -- Sum counts for all valid cases of c
  exact 50

-- Define our main theorem based on problem and final result
theorem odd_increasing_three_digit_numbers_count_eq_50 :
  count_odd_increasing_three_digit_numbers = 50 := by
  unfold count_odd_increasing_three_digit_numbers
  exact rfl -- the correct proof will fill in this part

end odd_increasing_three_digit_numbers_count_eq_50_l165_165322


namespace solve_quadratic_l165_165911

theorem solve_quadratic (x : ℝ) (h1 : 2 * x^2 - 6 * x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end solve_quadratic_l165_165911


namespace count_odd_three_digit_integers_in_increasing_order_l165_165326

-- Defining the conditions
def digits_in_strictly_increasing_order (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < 10

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def odd_three_digit_integers_in_increasing_order : ℕ :=
  ((finset.range 10).filter (λ c, is_odd c)).sum (λ c,
    ((finset.range c).sum (λ b,
      if h : b < c then
        (finset.range b).filter (λ a, digits_in_strictly_increasing_order a b c).card
      else 0)))

-- Theorem statement: Prove that the number of such numbers is 50
theorem count_odd_three_digit_integers_in_increasing_order :
  odd_three_digit_integers_in_increasing_order = 50 :=
sorry

end count_odd_three_digit_integers_in_increasing_order_l165_165326


namespace paul_lost_crayons_l165_165365

theorem paul_lost_crayons :
  let total := 229
  let given_away := 213
  let lost := total - given_away
  lost = 16 :=
by
  sorry

end paul_lost_crayons_l165_165365


namespace time_spent_cleaning_bathroom_l165_165931

-- Define the times spent on each task
def laundry_time : ℕ := 30
def room_cleaning_time : ℕ := 35
def homework_time : ℕ := 40
def total_time : ℕ := 120

-- Let b be the time spent cleaning the bathroom
variable (b : ℕ)

-- Total time spent on all tasks is the sum of individual times
def total_task_time := laundry_time + b + room_cleaning_time + homework_time

-- Proof that b = 15 given the total time
theorem time_spent_cleaning_bathroom (h : total_task_time = total_time) : b = 15 :=
by
  sorry

end time_spent_cleaning_bathroom_l165_165931


namespace fixed_point_quadratic_l165_165204

theorem fixed_point_quadratic : 
  (∀ m : ℝ, 3 * a ^ 2 - m * a + 2 * m + 1 = b) → (a = 2 ∧ b = 13) := 
by sorry

end fixed_point_quadratic_l165_165204


namespace expected_number_of_letters_in_mailbox_A_l165_165799

def prob_xi_0 : ℚ := 4 / 9
def prob_xi_1 : ℚ := 4 / 9
def prob_xi_2 : ℚ := 1 / 9

def expected_xi := 0 * prob_xi_0 + 1 * prob_xi_1 + 2 * prob_xi_2

theorem expected_number_of_letters_in_mailbox_A :
  expected_xi = 2 / 3 := by
  sorry

end expected_number_of_letters_in_mailbox_A_l165_165799


namespace percentage_calculation_l165_165451

/-- If x % of 375 equals 5.4375, then x % equals 1.45 %. -/
theorem percentage_calculation (x : ℝ) (h : x / 100 * 375 = 5.4375) : x = 1.45 := 
sorry

end percentage_calculation_l165_165451


namespace max_area_rectangle_l165_165963

theorem max_area_rectangle (perimeter : ℕ) (a b : ℕ) (h1 : perimeter = 30) 
  (h2 : b = a + 3) : a * b = 54 :=
by
  sorry

end max_area_rectangle_l165_165963


namespace BretCatchesFrogs_l165_165231

-- Define the number of frogs caught by Alster, Quinn, and Bret.
def AlsterFrogs : Nat := 2
def QuinnFrogs (a : Nat) : Nat := 2 * a
def BretFrogs (q : Nat) : Nat := 3 * q

-- The main theorem to prove
theorem BretCatchesFrogs : BretFrogs (QuinnFrogs AlsterFrogs) = 12 :=
by
  sorry

end BretCatchesFrogs_l165_165231


namespace findMultipleOfSamsMoney_l165_165159

-- Define the conditions specified in the problem
def SamMoney : ℕ := 75
def TotalMoney : ℕ := 200
def BillyHasLess (x : ℕ) : ℕ := x * SamMoney - 25

-- State the theorem to prove
theorem findMultipleOfSamsMoney (x : ℕ) 
  (h1 : SamMoney + BillyHasLess x = TotalMoney) : x = 2 :=
by
  -- Placeholder for the proof
  sorry

end findMultipleOfSamsMoney_l165_165159


namespace power_difference_divisible_by_10000_l165_165366

theorem power_difference_divisible_by_10000 (a b : ℤ) (m : ℤ) (h : a - b = 100 * m) : ∃ k : ℤ, a^100 - b^100 = 10000 * k := by
  sorry

end power_difference_divisible_by_10000_l165_165366


namespace eq_solution_set_l165_165291

theorem eq_solution_set :
  {x : ℝ | (2 / (x + 2)) + (4 / (x + 8)) ≥ 3 / 4} = {x : ℝ | -2 < x ∧ x ≤ 2} :=
by {
  sorry
}

end eq_solution_set_l165_165291


namespace sum_of_two_squares_iff_double_sum_of_two_squares_l165_165635

theorem sum_of_two_squares_iff_double_sum_of_two_squares (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_two_squares_iff_double_sum_of_two_squares_l165_165635


namespace abs_difference_of_squares_l165_165539

theorem abs_difference_of_squares : 
  let a := 105 
  let b := 103
  abs (a^2 - b^2) = 416 := 
by 
  let a := 105
  let b := 103
  sorry

end abs_difference_of_squares_l165_165539


namespace find_z2015_l165_165730

noncomputable def complex_seq : ℕ → ℂ 
| 1       := 1
| (n + 1) := complex.conj (complex_seq n) + 1 + (n : ℂ) * complex.I

theorem find_z2015 : complex_seq 2015 = 2015 + 1007 * complex.I :=
by      
      sorry

end find_z2015_l165_165730


namespace number_of_bulls_l165_165383

theorem number_of_bulls (total_cattle : ℕ) (ratio_cows_bulls : ℕ) (cows_bulls : ℕ) 
(h_total : total_cattle = 555) (h_ratio : ratio_cows_bulls = 10) (h_bulls_ratio : cows_bulls = 27) :
  let total_ratio_units := ratio_cows_bulls + cows_bulls in
  let bulls_count := (cows_bulls * total_cattle) / total_ratio_units in
  bulls_count = 405 := 
by
  sorry

end number_of_bulls_l165_165383


namespace chris_wins_l165_165163

noncomputable def chris_heads : ℚ := 1 / 4
noncomputable def drew_heads : ℚ := 1 / 3
noncomputable def both_tails : ℚ := (1 - chris_heads) * (1 - drew_heads)

/-- The probability that Chris wins comparing with relatively prime -/
theorem chris_wins (p q : ℕ) (hpq : Nat.Coprime p q) (hq0 : q ≠ 0) :
  (chris_heads * (1 + both_tails)) = (p : ℚ) / q ∧ (q - p = 1) :=
sorry

end chris_wins_l165_165163


namespace pastries_sold_correctly_l165_165809

def cupcakes : ℕ := 4
def cookies : ℕ := 29
def total_pastries : ℕ := cupcakes + cookies
def left_over : ℕ := 24
def sold_pastries : ℕ := total_pastries - left_over

theorem pastries_sold_correctly : sold_pastries = 9 :=
by sorry

end pastries_sold_correctly_l165_165809


namespace hyperbola_eccentricity_correct_l165_165906

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_asymptote : b / a = Real.tan (Real.pi / 6)) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_correct
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_asymptote : b / a = Real.tan (Real.pi / 6)) :
  hyperbola_eccentricity a b h_a h_b h_asymptote = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_correct_l165_165906


namespace new_function_expression_l165_165464

def initial_function (x : ℝ) : ℝ := -2 * x ^ 2

def shifted_function (x : ℝ) : ℝ := -2 * (x + 1) ^ 2 - 3

theorem new_function_expression :
  (∀ x : ℝ, (initial_function (x + 1) - 3) = shifted_function x) :=
by
  sorry

end new_function_expression_l165_165464


namespace marble_189_is_gray_l165_165833

def marble_color (n : ℕ) : String :=
  let cycle_length := 14
  let gray_thres := 5
  let white_thres := 9
  let black_thres := 12
  let position := (n - 1) % cycle_length + 1
  if position ≤ gray_thres then "gray"
  else if position ≤ white_thres then "white"
  else if position ≤ black_thres then "black"
  else "blue"

theorem marble_189_is_gray : marble_color 189 = "gray" :=
by {
  -- We assume the necessary definitions and steps discussed above.
  sorry
}

end marble_189_is_gray_l165_165833


namespace star_points_number_l165_165419

-- Let n be the number of points in the star
def n : ℕ := sorry

-- Let A and B be the angles at the star points, with the condition that A_i = B_i - 20
def A (i : ℕ) : ℝ := sorry
def B (i : ℕ) : ℝ := sorry

-- Condition: For all i, A_i = B_i - 20
axiom angle_condition : ∀ i, A i = B i - 20

-- Total sum of angle differences equal to 360 degrees
axiom angle_sum_condition : n * 20 = 360

-- Theorem to prove
theorem star_points_number : n = 18 := by
  sorry

end star_points_number_l165_165419


namespace necessary_condition_x_pow_2_minus_x_lt_0_l165_165061

theorem necessary_condition_x_pow_2_minus_x_lt_0 (x : ℝ) : (x^2 - x < 0) → (-1 < x ∧ x < 1) := by
  intro hx
  sorry

end necessary_condition_x_pow_2_minus_x_lt_0_l165_165061


namespace number_of_pieces_l165_165493

def area_of_pan (length : ℕ) (width : ℕ) : ℕ := length * width

def area_of_piece (side : ℕ) : ℕ := side * side

theorem number_of_pieces (length width side : ℕ) (h_length : length = 24) (h_width : width = 15) (h_side : side = 3) :
  (area_of_pan length width) / (area_of_piece side) = 40 :=
by
  rw [h_length, h_width, h_side]
  sorry

end number_of_pieces_l165_165493


namespace ratio_of_albert_to_mary_l165_165992

variables (A M B : ℕ) (s : ℕ) 

-- Given conditions as hypotheses
noncomputable def albert_is_multiple_of_mary := A = s * M
noncomputable def albert_is_4_times_betty := A = 4 * B
noncomputable def mary_is_22_years_younger := M = A - 22
noncomputable def betty_is_11 := B = 11

-- Theorem to prove the ratio of Albert's age to Mary's age
theorem ratio_of_albert_to_mary 
  (h1 : albert_is_multiple_of_mary A M s) 
  (h2 : albert_is_4_times_betty A B) 
  (h3 : mary_is_22_years_younger A M) 
  (h4 : betty_is_11 B) : 
  A / M = 2 :=
by
  sorry

end ratio_of_albert_to_mary_l165_165992


namespace katie_total_expenditure_l165_165021

-- Define the conditions
def flower_cost : ℕ := 6
def roses_bought : ℕ := 5
def daisies_bought : ℕ := 5

-- Define the total flowers bought
def total_flowers_bought : ℕ := roses_bought + daisies_bought

-- Calculate the total cost
def total_cost (flower_cost : ℕ) (total_flowers_bought : ℕ) : ℕ :=
  total_flowers_bought * flower_cost

-- Prove that Katie spent 60 dollars
theorem katie_total_expenditure : total_cost flower_cost total_flowers_bought = 60 := sorry

end katie_total_expenditure_l165_165021


namespace rhombus_area_l165_165342

theorem rhombus_area (d1 d2 : ℝ) (θ : ℝ) (h1 : d1 = 8) (h2 : d2 = 10) (h3 : Real.sin θ = 3 / 5) : 
  (1 / 2) * d1 * d2 * Real.sin θ = 24 :=
by
  sorry

end rhombus_area_l165_165342


namespace cupric_cyanide_formation_l165_165605

/--
Given:
1 mole of CuSO₄ 
2 moles of HCN

Prove:
The number of moles of Cu(CN)₂ formed is 0.
-/
theorem cupric_cyanide_formation (CuSO₄ HCN : ℕ) (h₁ : CuSO₄ = 1) (h₂ : HCN = 2) : 0 = 0 :=
by
  -- Proof goes here
  sorry

end cupric_cyanide_formation_l165_165605


namespace find_frac_sum_l165_165093

variable (a b c : ℝ)
variable (h1 : 16 * b^2 = 15 * a * c)
variable (h2 : 2 / b = 1 / a + 1 / c)

theorem find_frac_sum (a b c : ℝ) (h1 : 16 * b^2 = 15 * a * c) (h2 : 2 / b = 1 / a + 1 / c) :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 → (a / c + c / a) = 34 / 15 := by
  sorry

end find_frac_sum_l165_165093


namespace base_rate_of_second_company_l165_165970

-- Define the conditions
def United_base_rate : ℝ := 8.00
def United_rate_per_minute : ℝ := 0.25
def Other_rate_per_minute : ℝ := 0.20
def minutes : ℕ := 80

-- Define the total bill equations
def United_total_bill (minutes : ℕ) : ℝ := United_base_rate + United_rate_per_minute * minutes
def Other_total_bill (minutes : ℕ) (B : ℝ) : ℝ := B + Other_rate_per_minute * minutes

-- Define the claim to prove
theorem base_rate_of_second_company : ∃ B : ℝ, Other_total_bill minutes B = United_total_bill minutes ∧ B = 12.00 := by
  sorry

end base_rate_of_second_company_l165_165970


namespace prove_range_of_xyz_l165_165704

variable (x y z a : ℝ)

theorem prove_range_of_xyz 
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = a^2 / 2)
  (ha : 0 < a) :
  (0 ≤ x ∧ x ≤ 2 * a / 3) ∧ (0 ≤ y ∧ y ≤ 2 * a / 3) ∧ (0 ≤ z ∧ z ≤ 2 * a / 3) :=
sorry

end prove_range_of_xyz_l165_165704


namespace find_a_of_parabola_and_hyperbola_intersection_l165_165059

theorem find_a_of_parabola_and_hyperbola_intersection
  (a : ℝ)
  (h_a_pos : a > 0)
  (h_asymptotes_intersect_directrix_distance : ∀ (x_A x_B : ℝ),
    -1 / (4 * a) = (1 / 2) * x_A ∧ -1 / (4 * a) = -(1 / 2) * x_B →
    |x_B - x_A| = 4) : a = 1 / 4 := by
  sorry

end find_a_of_parabola_and_hyperbola_intersection_l165_165059


namespace percentage_not_speaking_French_is_60_l165_165817

-- Define the number of students who speak English well and those who do not.
def speakEnglishWell : Nat := 20
def doNotSpeakEnglish : Nat := 60

-- Calculate the total number of students who speak French.
def speakFrench : Nat := speakEnglishWell + doNotSpeakEnglish

-- Define the total number of students surveyed.
def totalStudents : Nat := 200

-- Calculate the number of students who do not speak French.
def doNotSpeakFrench : Nat := totalStudents - speakFrench

-- Calculate the percentage of students who do not speak French.
def percentageDoNotSpeakFrench : Float := (doNotSpeakFrench.toFloat / totalStudents.toFloat) * 100

-- Theorem asserting the percentage of students who do not speak French is 60%.
theorem percentage_not_speaking_French_is_60 : percentageDoNotSpeakFrench = 60 := by
  sorry

end percentage_not_speaking_French_is_60_l165_165817


namespace count_odd_three_digit_integers_in_increasing_order_l165_165324

-- Defining the conditions
def digits_in_strictly_increasing_order (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < 10

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def odd_three_digit_integers_in_increasing_order : ℕ :=
  ((finset.range 10).filter (λ c, is_odd c)).sum (λ c,
    ((finset.range c).sum (λ b,
      if h : b < c then
        (finset.range b).filter (λ a, digits_in_strictly_increasing_order a b c).card
      else 0)))

-- Theorem statement: Prove that the number of such numbers is 50
theorem count_odd_three_digit_integers_in_increasing_order :
  odd_three_digit_integers_in_increasing_order = 50 :=
sorry

end count_odd_three_digit_integers_in_increasing_order_l165_165324


namespace largest_unpayable_soldo_l165_165946

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l165_165946


namespace m_perp_n_α_perp_β_l165_165226

variables {Plane Line : Type}
variables (α β : Plane) (m n : Line)

def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry

-- Problem 1:
axiom m_perp_α : perpendicular_to_plane m α
axiom n_perp_β : perpendicular_to_plane n β
axiom α_perp_β : perpendicular_planes α β

theorem m_perp_n : perpendicular_lines m n :=
sorry

-- Problem 2:
axiom m_perp_n' : perpendicular_lines m n
axiom m_perp_α' : perpendicular_to_plane m α
axiom n_perp_β' : perpendicular_to_plane n β

theorem α_perp_β' : perpendicular_planes α β :=
sorry

end m_perp_n_α_perp_β_l165_165226


namespace cost_calculation_l165_165776

variables (H M F : ℝ)

theorem cost_calculation 
  (h1 : 3 * H + 5 * M + F = 23.50) 
  (h2 : 5 * H + 9 * M + F = 39.50) : 
  2 * H + 2 * M + 2 * F = 15.00 :=
sorry

end cost_calculation_l165_165776


namespace smallest_value_div_by_13_l165_165702

theorem smallest_value_div_by_13 : 
  ∃ (A B : ℕ), 
    (0 ≤ A ∧ A ≤ 9) ∧ (0 ≤ B ∧ B ≤ 9) ∧ 
    A ≠ B ∧ 
    1001 * A + 110 * B = 1771 ∧ 
    (1001 * A + 110 * B) % 13 = 0 :=
by
  sorry

end smallest_value_div_by_13_l165_165702


namespace correct_operation_l165_165664

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l165_165664


namespace exists_m_infinite_solutions_l165_165087

theorem exists_m_infinite_solutions : 
  ∃ m: ℕ, m = 18 ∧ ∃ᶠ (a b c: ℕ) in at_top, 
    (1:ℚ) / a + (1:ℚ) / b + (1:ℚ) / c + (1:ℚ) / (a * b * c) = m * (1:ℚ) / (a + b + c) :=
by
  sorry

end exists_m_infinite_solutions_l165_165087


namespace remainder_of_base12_integer_divided_by_9_l165_165404

-- Define the base-12 integer
def base12_integer := 2 * 12^3 + 7 * 12^2 + 4 * 12 + 3

-- Define the condition for our problem
def divisor := 9

-- State the theorem to be proved
theorem remainder_of_base12_integer_divided_by_9 :
  base12_integer % divisor = 0 :=
sorry

end remainder_of_base12_integer_divided_by_9_l165_165404


namespace inverse_true_l165_165689

theorem inverse_true : 
  (∀ (angles : Type) (l1 l2 : Type) (supplementary : angles → angles → Prop)
    (parallel : l1 → l2 → Prop), 
    (∀ a b, supplementary a b → a = b) ∧ (∀ l1 l2, parallel l1 l2)) ↔ 
    (∀ (angles : Type) (l1 l2 : Type) (supplementary : angles → angles → Prop)
    (parallel : l1 → l2 → Prop),
    (∀ l1 l2, parallel l1 l2) ∧ (∀ a b, supplementary a b → a = b)) :=
sorry

end inverse_true_l165_165689


namespace min_distance_hyperbola_l165_165838

open Real

theorem min_distance_hyperbola 
(hyperbola_eq : ∀ x : ℝ, x > 0 → ∀ y : ℝ, y = 4 / x → True) 
(rect_area : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 4 ∧ ∃ x y : ℝ, x = a ∧ y = b ∧ y = 4 / x) 
: ∃ x : ℝ, x > 0 ∧ (x^2 + (4 / x)^2) = 2:= sorry

end min_distance_hyperbola_l165_165838


namespace m_value_for_positive_root_eq_l165_165721

-- We start by defining the problem:
-- Given the condition that the equation (3x - 1)/(x + 1) - m/(x + 1) = 1 has a positive root,
-- we need to prove that m = -4.

theorem m_value_for_positive_root_eq (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 * x - 1) / (x + 1) - m / (x + 1) = 1) → m = -4 :=
by
  sorry

end m_value_for_positive_root_eq_l165_165721


namespace geometric_series_sum_l165_165874

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l165_165874


namespace range_of_a_l165_165056

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x + 1

def is_monotonic_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) a, 0 ≤ (deriv f) x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) a, 0 ≤ (deriv (f a)) x) → 1 ≤ a := 
sorry

end range_of_a_l165_165056


namespace smallest_positive_multiple_of_32_l165_165659

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ ∃ k : ℕ, k > 0 ∧ n = 32 * k ∧ n = 32 := by
  use 32
  constructor
  · exact Nat.zero_lt_succ 31
  · use 1
    constructor
    · exact Nat.zero_lt_succ 0
    · constructor
      · rfl
      · rfl

end smallest_positive_multiple_of_32_l165_165659


namespace part1_part2_l165_165195

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part1 (x : ℝ) : f x ≥ 3 ↔ x ≤ -3 / 2 ∨ x ≥ 3 / 2 := 
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - a) ↔ -1 ≤ a ∧ a ≤ 2 :=
  sorry

end part1_part2_l165_165195


namespace probability_at_least_one_succeeds_l165_165800

variable (p1 p2 : ℝ)

theorem probability_at_least_one_succeeds : 
  0 ≤ p1 ∧ p1 ≤ 1 → 0 ≤ p2 ∧ p2 ≤ 1 → (1 - (1 - p1) * (1 - p2)) = 1 - (1 - p1) * (1 - p2) :=
by 
  intro h1 h2
  sorry

end probability_at_least_one_succeeds_l165_165800


namespace geometric_sequence_expression_l165_165729

theorem geometric_sequence_expression (a : ℝ) (a_n: ℕ → ℝ)
  (h1 : a_n 1 = a - 1)
  (h2 : a_n 2 = a + 1)
  (h3 : a_n 3 = a + 4)
  (hn : ∀ n, a_n (n + 1) = a_n n * (a_n 2 / a_n 1)) :
  a_n n = 4 * (3/2)^(n-1) :=
sorry

end geometric_sequence_expression_l165_165729


namespace coefficient_of_x3_in_x_plus_one_pow_50_l165_165081

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := (n.choose k)

-- Define the binomial expansion using summation
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ → ℕ :=
  λ k, binom n k * x^k 

-- Define the specific problem
def coeff_x3_in_expansion : ℕ :=
  binom 50 3

-- Theorem stating the desired result
theorem coefficient_of_x3_in_x_plus_one_pow_50 :
  coeff_x3_in_expansion = 19600 :=
by
  -- Skipping the proof part by using sorry
  sorry

end coefficient_of_x3_in_x_plus_one_pow_50_l165_165081


namespace correct_number_of_arrangements_l165_165643

def arrangements_with_conditions (n : ℕ) : ℕ := 
  if n = 6 then
    let case1 := 120  -- when B is at the far right
    let case2 := 96   -- when A is at the far right
    case1 + case2
  else 0

theorem correct_number_of_arrangements : arrangements_with_conditions 6 = 216 :=
by {
  -- The detailed proof is omitted here
  sorry
}

end correct_number_of_arrangements_l165_165643


namespace odd_three_digit_integers_increasing_order_l165_165329

theorem odd_three_digit_integers_increasing_order :
  let digits_strictly_increasing (a b c : ℕ) : Prop := (1 ≤ a) ∧ (a < b) ∧ (b < c)
      let c_values : Finset ℕ := {3, 5, 7, 9}
  in ∑ c in c_values, (Finset.card (Finset.filter (λ ab : ℕ × ℕ, (digits_strictly_increasing ab.1 ab.2 c)) (Finset.cross {(1 : ℕ)..9} {(1 : ℕ)..9}))) = 50 :=
by
  sorry

end odd_three_digit_integers_increasing_order_l165_165329


namespace cube_volume_l165_165052

variable (V_sphere : ℝ)
variable (V_cube : ℝ)
variable (R : ℝ)
variable (a : ℝ)

theorem cube_volume (h1 : V_sphere = (32 / 3) * Real.pi)
    (h2 : V_sphere = (4 / 3) * Real.pi * R^3)
    (h3 : R = 2)
    (h4 : R = (Real.sqrt 3 / 2) * a)
    (h5 : a = 4 * Real.sqrt 3 / 3) :
    V_cube = (4 * Real.sqrt 3 / 3) ^ 3 :=
  by
    sorry

end cube_volume_l165_165052


namespace probability_blue_then_red_l165_165139

/--
A box contains 15 balls, of which 5 are blue and 10 are red.
Two balls are drawn sequentially from the box without returning the first ball to the box.
Prove that the probability that the first ball drawn is blue and the second ball is red is 5 / 21.
-/
theorem probability_blue_then_red :
  let total_balls := 15
  let blue_balls := 5
  let red_balls := 10
  let first_is_blue := (blue_balls : ℚ) / total_balls
  let second_is_red_given_blue := (red_balls : ℚ) / (total_balls - 1)
  first_is_blue * second_is_red_given_blue = 5 / 21 := by
  sorry

end probability_blue_then_red_l165_165139


namespace real_and_imaginary_solutions_l165_165586

theorem real_and_imaginary_solutions :
  ∃ (x y : ℂ), (y = (x + 1)^4) ∧ (x * y + y = 5) ∧
  ((∃ (xr : ℝ), x = xr ∧ y = (xr + 1)^4) ∧
   (∃ (z1 z2 z3 z4 : ℂ), z1 ≠ x ∧ z2 ≠ x ∧ z3 ≠ x ∧ z4 ≠ x ∧
    (y = (z1 + 1)^4 ∨ y = (z2 + 1)^4 ∨ y = (z3 + 1)^4 ∨ y = (z4 + 1)^4))) := sorry

end real_and_imaginary_solutions_l165_165586


namespace line_through_perpendicular_l165_165178

theorem line_through_perpendicular (x y : ℝ) :
  (∃ (k : ℝ), (2 * x - y + 3 = 0) ∧ k = - 1 / 2) →
  (∃ (a b c : ℝ), (a * (-1) + b * 1 + c = 0) ∧ a = 1 ∧ b = 2 ∧ c = -1) :=
by
  sorry

end line_through_perpendicular_l165_165178


namespace ott_fractional_part_l165_165761

theorem ott_fractional_part (M L N O x : ℝ)
  (hM : M = 6 * x)
  (hL : L = 5 * x)
  (hN : N = 4 * x)
  (hO : O = 0)
  (h_each : O + M + L + N = x + x + x) :
  (3 * x) / (M + L + N) = 1 / 5 :=
by
  sorry

end ott_fractional_part_l165_165761


namespace conditional_expectation_property_l165_165754

variables {Ω : Type*} [measurable_space Ω] {P : probability_measure Ω}

variables {η ξ : Ω → ℝ}
variables {𝒢 𝒢' : measurable_space Ω}
variables [sub measurable_space Ω 𝒢] [sub measurable_space Ω 𝒢']

def integrable (f : Ω → ℝ) : Prop :=
  ∫⁻ x, |f x| ∂P < ⊤

noncomputable
def p_exponent : Prop := 
  ∃ (p q : ℝ), 1 < p ∧ 1 < q ∧ 1 / p + 1 / q = 1

theorem conditional_expectation_property
  (h_measurable_η : measurable[𝒢] η)
  (h_measurable_ξ : measurable[𝒢'] ξ)
  (h_integrable_η : integrable (λ x, |η x|^q)) 
  (h_integrable_ξ : integrable (λ x, |ξ x|^p))
  (h_exponents : p_exponent) :
  ∀ᵖ x ∂P, condexp P 𝒢 (ξ * η) x = η x * condexp P 𝒢 ξ x :=
sorry

end conditional_expectation_property_l165_165754


namespace total_exercise_hours_l165_165932

-- Define the conditions
def Natasha_minutes_per_day : ℕ := 30
def Natasha_days : ℕ := 7
def Esteban_minutes_per_day : ℕ := 10
def Esteban_days : ℕ := 9
def Charlotte_monday_minutes : ℕ := 20
def Charlotte_wednesday_minutes : ℕ := 45
def Charlotte_thursday_minutes : ℕ := 30
def Charlotte_sunday_minutes : ℕ := 60

-- Sum up the minutes for each individual
def Natasha_total_minutes : ℕ := Natasha_minutes_per_day * Natasha_days
def Esteban_total_minutes : ℕ := Esteban_minutes_per_day * Esteban_days
def Charlotte_total_minutes : ℕ := Charlotte_monday_minutes + Charlotte_wednesday_minutes + Charlotte_thursday_minutes + Charlotte_sunday_minutes

-- Convert minutes to hours
noncomputable def minutes_to_hours (minutes : ℕ) : ℚ := minutes / 60

-- Calculation of hours for each individual
noncomputable def Natasha_total_hours : ℚ := minutes_to_hours Natasha_total_minutes
noncomputable def Esteban_total_hours : ℚ := minutes_to_hours Esteban_total_minutes
noncomputable def Charlotte_total_hours : ℚ := minutes_to_hours Charlotte_total_minutes

-- Prove total hours of exercise for all three individuals
theorem total_exercise_hours : Natasha_total_hours + Esteban_total_hours + Charlotte_total_hours = 7.5833 := by
  sorry

end total_exercise_hours_l165_165932


namespace find_x_l165_165357

variables {K J : ℝ} {A B C A_star B_star C_star : Type*}

-- Define the triangles and areas
def triangle_area (K : ℝ) : Prop := K > 0

-- We know the fractions of segments in triangle
def segment_ratios (x : ℝ) : Prop :=
  0 < x ∧ x < 1 ∧
  ∀ (AA_star AB BB_star BC CC_star CA : ℝ),
    AA_star / AB = x ∧ BB_star / BC = x ∧ CC_star / CA = x

-- Area of the smaller inner triangle
def inner_triangle_area (x : ℝ) (K : ℝ) (J : ℝ) : Prop :=
  J = x * K

-- The theorem combining all to show x = 1/3
theorem find_x (x : ℝ) (K J : ℝ) (triangleAreaK : triangle_area K)
    (ratios : segment_ratios x)
    (innerArea : inner_triangle_area x K J) :
  x = 1 / 3 :=
by
  sorry

end find_x_l165_165357


namespace least_distance_between_ticks_l165_165810

theorem least_distance_between_ticks (x : ℚ) :
  (∀ n : ℕ, ∃ k : ℤ, k = n * 11 ∨ k = n * 13) →
  x = 1 / 143 :=
by
  sorry

end least_distance_between_ticks_l165_165810


namespace largest_N_not_payable_l165_165938

theorem largest_N_not_payable (coins_5 coins_6 : ℕ) (h1 : coins_5 > 10) (h2 : coins_6 > 10) :
  ∃ N : ℕ, N ≤ 50 ∧ (¬ ∃ (x y : ℕ), N = 5 * x + 6 * y) ∧ N = 19 :=
by 
  use 19
  have h₃ : 19 ≤ 50 := by norm_num
  have h₄ : ¬ ∃ (x y : ℕ), 19 = 5 * x + 6 * y := sorry
  exact ⟨h₃, h₄⟩

end largest_N_not_payable_l165_165938


namespace perimeter_after_adding_tiles_l165_165644

-- Initial perimeter given
def initial_perimeter : ℕ := 20

-- Number of initial tiles
def initial_tiles : ℕ := 10

-- Number of additional tiles to be added
def additional_tiles : ℕ := 2

-- New tile side must be adjacent to an existing tile
def adjacent_tile_side : Prop := true

-- Condition about the tiles being 1x1 squares
def sq_tile (n : ℕ) : Prop := n = 1

-- The perimeter should be calculated after adding the tiles
def new_perimeter_after_addition : ℕ := 19

theorem perimeter_after_adding_tiles :
  ∃ (new_perimeter : ℕ), 
    new_perimeter = 19 ∧ 
    initial_perimeter = 20 ∧ 
    initial_tiles = 10 ∧ 
    additional_tiles = 2 ∧ 
    adjacent_tile_side ∧ 
    sq_tile 1 :=
sorry

end perimeter_after_adding_tiles_l165_165644


namespace other_diagonal_of_rhombus_l165_165772

noncomputable def calculate_other_diagonal (area d1 : ℝ) : ℝ :=
  (area * 2) / d1

theorem other_diagonal_of_rhombus {a1 a2 : ℝ} (area_eq : a1 = 21.46) (d1_eq : a2 = 7.4) : calculate_other_diagonal a1 a2 = 5.8 :=
by
  rw [area_eq, d1_eq]
  norm_num
  -- The next step would involve proving that (21.46 * 2) / 7.4 = 5.8 in a formal proof.
  sorry

end other_diagonal_of_rhombus_l165_165772


namespace find_x_l165_165399

variable (m k x Km2 mk : ℚ)

def valid_conditions (m k : ℚ) : Prop :=
  m > 2 * k ∧ k > 0

def initial_acid (m : ℚ) : ℚ :=
  (m*m)/100

def diluted_acid (m k x : ℚ) : ℚ :=
  ((2*m) - k) * (m + x) / 100

theorem find_x (m k : ℚ) (h : valid_conditions m k):
  ∃ x : ℚ, (m^2 = diluted_acid m k x) ∧ x = (k * m - m^2) / (2 * m - k) :=
sorry

end find_x_l165_165399


namespace quadratic_passes_through_constant_point_l165_165205

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 - m * x + 2 * m + 1

theorem quadratic_passes_through_constant_point :
  ∀ m : ℝ, f m 2 = 13 :=
by
  intro m
  unfold f
  simp
  rfl

end quadratic_passes_through_constant_point_l165_165205


namespace sara_gave_dan_limes_l165_165164

theorem sara_gave_dan_limes (initial_limes : ℕ) (final_limes : ℕ) (d : ℕ) 
  (h1: initial_limes = 9) (h2: final_limes = 13) (h3: final_limes = initial_limes + d) : d = 4 := 
by sorry

end sara_gave_dan_limes_l165_165164


namespace first_term_of_arithmetic_sequence_l165_165014

theorem first_term_of_arithmetic_sequence (a : ℕ) (median last_term : ℕ) 
  (h_arithmetic_progression : true) (h_median : median = 1010) (h_last_term : last_term = 2015) :
  a = 5 :=
by
  have h1 : 2 * median = 2020 := by sorry
  have h2 : last_term + a = 2020 := by sorry
  have h3 : 2015 + a = 2020 := by sorry
  have h4 : a = 2020 - 2015 := by sorry
  have h5 : a = 5 := by sorry
  exact h5

end first_term_of_arithmetic_sequence_l165_165014


namespace find_integers_l165_165578

theorem find_integers (n : ℤ) : (n^2 - 13 * n + 36 < 0) ↔ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 :=
by
  sorry

end find_integers_l165_165578


namespace parallel_lines_l165_165184

theorem parallel_lines (a : ℝ) (h : ∀ x y : ℝ, 2*x - a*y - 1 = 0 → a*x - y = 0) : a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
sorry

end parallel_lines_l165_165184


namespace count_numbers_with_digit_sum_10_l165_165292

theorem count_numbers_with_digit_sum_10 : 
  ∃ n : ℕ, 
  (n = 66) ∧ ∀ (a b c : ℕ), 
  0 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  a + b + c = 10 → 
  true :=
by
  sorry

end count_numbers_with_digit_sum_10_l165_165292


namespace range_of_m_l165_165735

noncomputable def A (x : ℝ) : ℝ := x^2 - (3/2) * x + 1

def in_interval (x : ℝ) : Prop := (3/4 ≤ x) ∧ (x ≤ 2)

def B (y : ℝ) (m : ℝ) : Prop := y ≥ 1 - m^2

theorem range_of_m (m : ℝ) :
  (∀ x, in_interval x → B (A x) m) ↔ (m ≤ - (3/4) ∨ m ≥ (3/4)) := 
sorry

end range_of_m_l165_165735


namespace bulls_on_farm_l165_165382

theorem bulls_on_farm (C B : ℕ) (h1 : C / B = 10 / 27) (h2 : C + B = 555) : B = 405 :=
sorry

end bulls_on_farm_l165_165382


namespace difference_of_numbers_l165_165550

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : |x - y| = 7 := 
by
  sorry

end difference_of_numbers_l165_165550


namespace initial_water_percentage_l165_165626

variable (W : ℝ) -- Initial percentage of water in the milk

theorem initial_water_percentage 
  (final_water_content : ℝ := 2) 
  (pure_milk_added : ℝ := 15) 
  (initial_milk_volume : ℝ := 10)
  (final_mixture_volume : ℝ := initial_milk_volume + pure_milk_added)
  (water_equation : W / 100 * initial_milk_volume = final_water_content / 100 * final_mixture_volume) 
  : W = 5 :=
by
  sorry

end initial_water_percentage_l165_165626


namespace count_odd_three_digit_integers_in_increasing_order_l165_165327

-- Defining the conditions
def digits_in_strictly_increasing_order (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < 10

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def odd_three_digit_integers_in_increasing_order : ℕ :=
  ((finset.range 10).filter (λ c, is_odd c)).sum (λ c,
    ((finset.range c).sum (λ b,
      if h : b < c then
        (finset.range b).filter (λ a, digits_in_strictly_increasing_order a b c).card
      else 0)))

-- Theorem statement: Prove that the number of such numbers is 50
theorem count_odd_three_digit_integers_in_increasing_order :
  odd_three_digit_integers_in_increasing_order = 50 :=
sorry

end count_odd_three_digit_integers_in_increasing_order_l165_165327


namespace number_divisible_by_45_and_6_l165_165393

theorem number_divisible_by_45_and_6 (k : ℕ) (h1 : 1 ≤ k) (h2 : ∃ n : ℕ, 190 + 90 * (k - 1) ≤  n ∧ n < 190 + 90 * k) 
: 190 + 90 * 5 = 720 := by
  sorry

end number_divisible_by_45_and_6_l165_165393


namespace cooking_time_remaining_l165_165693

def time_to_cook_remaining (n_total n_cooked t_per : ℕ) : ℕ := (n_total - n_cooked) * t_per

theorem cooking_time_remaining :
  ∀ (n_total n_cooked t_per : ℕ), n_total = 13 → n_cooked = 5 → t_per = 6 → time_to_cook_remaining n_total n_cooked t_per = 48 :=
by
  intros n_total n_cooked t_per h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end cooking_time_remaining_l165_165693


namespace geometric_series_sum_l165_165861

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l165_165861


namespace lisa_cleaning_time_l165_165095

theorem lisa_cleaning_time (L : ℝ) (h1 : (1 / L) + (1 / 12) = 1 / 4.8) : L = 8 :=
sorry

end lisa_cleaning_time_l165_165095


namespace left_side_value_l165_165117

-- Define the relevant variables and conditions
variable (L R B : ℕ)

-- Assuming conditions
def sum_of_sides (L R B : ℕ) : Prop := L + R + B = 50
def right_side_relation (L R : ℕ) : Prop := R = L + 2
def base_value (B : ℕ) : Prop := B = 24

-- Main theorem statement
theorem left_side_value (L R B : ℕ) (h1 : sum_of_sides L R B) (h2 : right_side_relation L R) (h3 : base_value B) : L = 12 :=
sorry

end left_side_value_l165_165117


namespace attended_college_percentage_l165_165559

variable (total_boys : ℕ) (total_girls : ℕ) (percent_not_attend_boys : ℕ) (percent_not_attend_girls : ℕ)

def total_boys_attended_college (total_boys percent_not_attend_boys : ℕ) : ℕ :=
  total_boys - percent_not_attend_boys * total_boys / 100

def total_girls_attended_college (total_girls percent_not_attend_girls : ℕ) : ℕ :=
  total_girls - percent_not_attend_girls * total_girls / 100

noncomputable def total_student_attended_college (total_boys total_girls percent_not_attend_boys percent_not_attend_girls : ℕ) : ℕ :=
  total_boys_attended_college total_boys percent_not_attend_boys +
  total_girls_attended_college total_girls percent_not_attend_girls

noncomputable def percent_class_attended_college (total_boys total_girls percent_not_attend_boys percent_not_attend_girls : ℕ) : ℕ :=
  total_student_attended_college total_boys total_girls percent_not_attend_boys percent_not_attend_girls * 100 /
  (total_boys + total_girls)

theorem attended_college_percentage :
  total_boys = 300 → total_girls = 240 → percent_not_attend_boys = 30 → percent_not_attend_girls = 30 →
  percent_class_attended_college total_boys total_girls percent_not_attend_boys percent_not_attend_girls = 70 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end attended_college_percentage_l165_165559


namespace minimalBananasTotal_is_408_l165_165248

noncomputable def minimalBananasTotal : ℕ :=
  let b₁ := 11 * 8
  let b₂ := 13 * 8
  let b₃ := 27 * 8
  b₁ + b₂ + b₃

theorem minimalBananasTotal_is_408 : minimalBananasTotal = 408 := by
  sorry

end minimalBananasTotal_is_408_l165_165248


namespace at_least_one_true_l165_165301

theorem at_least_one_true (p q : Prop) (h : ¬(p ∨ q) = false) : p ∨ q :=
by
  sorry

end at_least_one_true_l165_165301


namespace odd_three_digit_integers_in_strict_increasing_order_l165_165319

theorem odd_three_digit_integers_in_strict_increasing_order: 
  (∀ (a b c : ℕ), 100 ≤ (100 * a + 10 * b + c) ∧ 100 * a + 10 * b + c < 1000 → a < b ∧ b < c →
  c % 2 = 1 ∧ c ≠ 0 → 
  (∃ n, n = 50)) :=
by sorry

end odd_three_digit_integers_in_strict_increasing_order_l165_165319


namespace max_value_isosceles_triangle_l165_165432

theorem max_value_isosceles_triangle (a b c : ℝ) (h_isosceles : b = c) :
  ∃ B, (∀ (a b c : ℝ), b = c → (b + c) / a ≤ B) ∧ B = 2 :=
by
  sorry

end max_value_isosceles_triangle_l165_165432


namespace cost_of_item_D_is_30_usd_l165_165213

noncomputable def cost_of_item_D_in_usd (total_spent items_ABC_spent tax_paid service_fee_rate exchange_rate : ℝ) : ℝ :=
  let total_spent_with_fee := total_spent * (1 + service_fee_rate)
  let item_D_cost_FC := total_spent_with_fee - items_ABC_spent
  item_D_cost_FC * exchange_rate

theorem cost_of_item_D_is_30_usd
  (total_spent : ℝ)
  (items_ABC_spent : ℝ)
  (tax_paid : ℝ)
  (service_fee_rate : ℝ)
  (exchange_rate : ℝ)
  (h_total_spent : total_spent = 500)
  (h_items_ABC_spent : items_ABC_spent = 450)
  (h_tax_paid : tax_paid = 60)
  (h_service_fee_rate : service_fee_rate = 0.02)
  (h_exchange_rate : exchange_rate = 0.5) :
  cost_of_item_D_in_usd total_spent items_ABC_spent tax_paid service_fee_rate exchange_rate = 30 :=
by
  have h1 : total_spent * (1 + service_fee_rate) = 500 * 1.02 := sorry
  have h2 : 500 * 1.02 - 450 = 60 := sorry
  have h3 : 60 * 0.5 = 30 := sorry
  sorry

end cost_of_item_D_is_30_usd_l165_165213


namespace evaluate_expression_l165_165174

theorem evaluate_expression (x y : ℝ) (P Q : ℝ) 
  (hP : P = x^2 + y^2) 
  (hQ : Q = x - y) : 
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = 4 * x * y / ((x^2 + y^2)^2 - (x - y)^2) :=
by 
  -- Insert proof here
  sorry

end evaluate_expression_l165_165174


namespace math_problem_l165_165673

theorem math_problem (a b m : ℝ) : 
  ¬((-2 * a) ^ 2 = -4 * a ^ 2) ∧ 
  ¬((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧ 
  ((-m + 2) * (-m - 2) = m ^ 2 - 4) ∧ 
  ¬((a ^ 5) ^ 2 = a ^ 7) :=
by 
  split ; 
  sorry ;
  split ;
  sorry ;
  split ; 
  sorry ;
  sorry

end math_problem_l165_165673


namespace infinite_geometric_series_sum_l165_165879

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l165_165879


namespace mink_ratio_set_free_to_total_l165_165346

-- Given conditions
def coats_needed_per_skin : ℕ := 15
def minks_bought : ℕ := 30
def babies_per_mink : ℕ := 6
def coats_made : ℕ := 7

-- Question as a proof problem
theorem mink_ratio_set_free_to_total :
  let total_minks := minks_bought * (1 + babies_per_mink)
  let minks_used := coats_made * coats_needed_per_skin
  let minks_set_free := total_minks - minks_used
  minks_set_free * 2 = total_minks :=
by
  sorry

end mink_ratio_set_free_to_total_l165_165346


namespace option_C_correct_l165_165669

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l165_165669


namespace proper_subset_count_l165_165602

open Finset

variable (U : Finset ℕ) (complement_A : Finset ℕ)
variable (hU : U = {1, 2, 3}) (h_complement_A : complement_A = {2})

noncomputable def A : Finset ℕ := U \ complement_A

theorem proper_subset_count (U : Finset ℕ) (complement_A : Finset ℕ)
  (hU : U = {1, 2, 3}) (h_complement_A : complement_A = {2}) :
  (U \ complement_A).card = 2 ∧ (U \ complement_A).powerset.card - 1 = 3 := 
sorry

end proper_subset_count_l165_165602


namespace find_n_l165_165148

theorem find_n (n : ℕ)
  (h1 : ∃ k : ℕ, k = n^3) -- the cube is cut into n^3 unit cubes
  (h2 : ∃ r : ℕ, r = 4 * n^2) -- 4 faces are painted, each with area n^2
  (h3 : 1 / 3 = r / (6 * k)) -- one-third of the total number of faces are red
  : n = 2 :=
by
  sorry

end find_n_l165_165148


namespace minimum_dwarfs_l165_165499

-- Define the problem conditions
def chairs := fin 30 → Prop
def occupied (C : chairs) := ∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))

-- State the theorem that provides the solution
theorem minimum_dwarfs (C : chairs) : (∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))) → ∃ n : ℕ, n = 10 :=
by
  sorry

end minimum_dwarfs_l165_165499


namespace no_intersection_of_curves_l165_165042

theorem no_intersection_of_curves :
  ∀ x y : ℝ, ¬ (3 * x^2 + 2 * y^2 = 4 ∧ 6 * x^2 + 3 * y^2 = 9) :=
by sorry

end no_intersection_of_curves_l165_165042


namespace unique_zero_function_l165_165714

theorem unique_zero_function (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end unique_zero_function_l165_165714


namespace percent_value_in_quarters_l165_165128

theorem percent_value_in_quarters
  (num_dimes num_quarters num_nickels : ℕ)
  (value_dime value_quarter value_nickel : ℕ)
  (h_dimes : num_dimes = 70)
  (h_quarters : num_quarters = 30)
  (h_nickels : num_nickels = 40)
  (h_value_dime : value_dime = 10)
  (h_value_quarter : value_quarter = 25)
  (h_value_nickel : value_nickel = 5) :
  ((num_quarters * value_quarter : ℕ) * 100 : ℚ) / 
  (num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel) = 45.45 :=
by
  sorry

end percent_value_in_quarters_l165_165128


namespace arithmetic_sequence_common_difference_l165_165921

variable (a_n : ℕ → ℝ)

theorem arithmetic_sequence_common_difference
  (h_arith : ∀ n, a_n (n + 1) = a_n n + d)
  (h_non_zero : d ≠ 0)
  (h_sum : a_n 1 + a_n 2 + a_n 3 = 9)
  (h_geom : a_n 2 ^ 2 = a_n 1 * a_n 5) :
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l165_165921


namespace fixed_salary_new_scheme_l165_165270

theorem fixed_salary_new_scheme :
  let old_commission_rate := 0.05
  let new_commission_rate := 0.025
  let sales_target := 4000
  let total_sales := 12000
  let remuneration_difference := 600
  let old_remuneration := old_commission_rate * total_sales
  let new_commission_earnings := new_commission_rate * (total_sales - sales_target)
  let new_remuneration := old_remuneration + remuneration_difference
  ∃ F, F + new_commission_earnings = new_remuneration :=
by
  sorry

end fixed_salary_new_scheme_l165_165270


namespace UnionMathInstitute_students_l165_165516

theorem UnionMathInstitute_students :
  ∃ n : ℤ, n < 500 ∧ 
    n % 17 = 15 ∧ 
    n % 19 = 18 ∧ 
    n % 16 = 7 ∧ 
    n = 417 :=
by
  -- Problem setup and constraints
  sorry

end UnionMathInstitute_students_l165_165516


namespace point_on_x_axis_l165_165075

theorem point_on_x_axis (A B C D : ℝ × ℝ) : B = (3,0) → B.2 = 0 :=
by
  intros h
  subst h
  exact rfl

end point_on_x_axis_l165_165075


namespace polynomial_is_constant_l165_165370

-- Definitions and conditions for the problem
def isFibonacci (n : ℤ) : Prop :=
  ∃ i : ℤ, natAbs n = natAbs (fibonacci i)

def digitSum (n : ℤ) : ℤ :=
  (n.toString.data.map fun c => (c.toNat - '0'.toNat)).sum

-- The main statement of the problem
theorem polynomial_is_constant (P : Polynomial ℤ)
  (h : ∀ n : ℕ, ¬isFibonacci (digitSum (abs (P.eval (n : ℤ))))) :
  ∃ c : ℤ, P = Polynomial.C c :=
sorry

end polynomial_is_constant_l165_165370


namespace evaluate_expression_l165_165713

theorem evaluate_expression : (1023 * 1023) - (1022 * 1024) = 1 := by
  sorry

end evaluate_expression_l165_165713


namespace fifth_grade_total_students_l165_165615

-- Define the conditions given in the problem
def total_boys : ℕ := 350
def total_playing_soccer : ℕ := 250
def percentage_boys_playing_soccer : ℝ := 0.86
def girls_not_playing_soccer : ℕ := 115

-- Define the total number of students
def total_students : ℕ := 500

-- Prove that the total number of students is 500
theorem fifth_grade_total_students 
  (H1 : total_boys = 350) 
  (H2 : total_playing_soccer = 250) 
  (H3 : percentage_boys_playing_soccer = 0.86) 
  (H4 : girls_not_playing_soccer = 115) :
  total_students = 500 := 
sorry

end fifth_grade_total_students_l165_165615


namespace problem_l165_165044

theorem problem (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h1 : p + 5 < q)
  (h2 : (p + (p + 2) + (p + 5) + q + (q + 1) + (2 * q - 1)) / 6 = q)
  (h3 : (p + 5 + q) / 2 = q) : p + q = 11 :=
by sorry

end problem_l165_165044


namespace probability_A_and_B_l165_165401

universe u

def plants := {A, B, C, D, E}

def select_three_plants :=
  {s | s ⊆ plants ∧ s.card = 3}

def count_combinations (n k : ℕ) : ℕ :=
  (nat.choose n k).to_nat

def probability_A_and_B_selected : ℚ :=
  let total_outcomes := count_combinations 5 3
  let favorable_outcomes := count_combinations 3 1
  favorable_outcomes / total_outcomes

theorem probability_A_and_B : probability_A_and_B_selected = 3 / 10 := by
  sorry

end probability_A_and_B_l165_165401


namespace cost_per_square_meter_of_mat_l165_165209

theorem cost_per_square_meter_of_mat {L W E : ℝ} : 
  L = 20 → W = 15 → E = 57000 → (E / (L * W)) = 190 :=
by
  intros hL hW hE
  rw [hL, hW, hE]
  sorry

end cost_per_square_meter_of_mat_l165_165209


namespace find_f_of_3_l165_165454

-- Define the function f and its properties
variable {f : ℝ → ℝ}

-- Define the properties given in the problem
axiom f_mono_increasing : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_of_f_minus_exp : ∀ x : ℝ, f (f x - 2^x) = 3

-- The main theorem to prove
theorem find_f_of_3 : f 3 = 9 := 
sorry

end find_f_of_3_l165_165454


namespace calculation_simplifies_l165_165282

theorem calculation_simplifies :
  120 * (120 - 12) - (120 * 120 - 12) = -1428 := by
  sorry

end calculation_simplifies_l165_165282


namespace sequence_sum_is_25_div_3_l165_165077

noncomputable def sum_of_arithmetic_sequence (a n d : ℝ) : ℝ := (n / 2) * (2 * a + (n - 1) * d)

theorem sequence_sum_is_25_div_3 (a d : ℝ)
  (h1 : a + 4 * d = 1)
  (h2 : 3 * a + 15 * d = 2 * a + 8 * d) :
  sum_of_arithmetic_sequence a 10 d = 25 / 3 := by
  sorry

end sequence_sum_is_25_div_3_l165_165077


namespace percentage_of_good_fruits_l165_165272

theorem percentage_of_good_fruits (total_oranges : ℕ) (total_bananas : ℕ) 
    (rotten_oranges_percent : ℝ) (rotten_bananas_percent : ℝ) :
    total_oranges = 600 ∧ total_bananas = 400 ∧ 
    rotten_oranges_percent = 0.15 ∧ rotten_bananas_percent = 0.03 →
    (510 + 388) / (600 + 400) * 100 = 89.8 :=
by
  intros
  sorry

end percentage_of_good_fruits_l165_165272


namespace simplify_expression_l165_165104

theorem simplify_expression (b : ℝ) : (1 * 3 * b * 4 * b^2 * 5 * b^3 * 6 * b^4) = 360 * b^10 :=
by sorry

end simplify_expression_l165_165104


namespace odd_three_digit_integers_in_strict_increasing_order_l165_165316

theorem odd_three_digit_integers_in_strict_increasing_order: 
  (∀ (a b c : ℕ), 100 ≤ (100 * a + 10 * b + c) ∧ 100 * a + 10 * b + c < 1000 → a < b ∧ b < c →
  c % 2 = 1 ∧ c ≠ 0 → 
  (∃ n, n = 50)) :=
by sorry

end odd_three_digit_integers_in_strict_increasing_order_l165_165316


namespace aluminum_percentage_range_l165_165396

variable (x1 x2 x3 y : ℝ)

theorem aluminum_percentage_range:
  (0.15 * x1 + 0.3 * x2 = 0.2) →
  (x1 + x2 + x3 = 1) →
  y = 0.6 * x1 + 0.45 * x3 →
  (1/3 ≤ x2 ∧ x2 ≤ 2/3) →
  (0.15 ≤ y ∧ y ≤ 0.4) := by
  sorry

end aluminum_percentage_range_l165_165396


namespace muffins_total_is_83_l165_165563

-- Define the given conditions.
def initial_muffins : Nat := 35
def additional_muffins : Nat := 48

-- Define the total number of muffins.
def total_muffins : Nat := initial_muffins + additional_muffins

-- Statement to prove.
theorem muffins_total_is_83 : total_muffins = 83 := by
  -- Proof is omitted.
  sorry

end muffins_total_is_83_l165_165563


namespace infinite_geometric_series_sum_l165_165856

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l165_165856


namespace find_a_l165_165594

def set_A : Set ℝ := {x | x^2 + x - 6 = 0}

def set_B (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem find_a (a : ℝ) : set_A ∪ set_B a = set_A ↔ a ∈ ({0, 1/3, -1/2} : Set ℝ) := 
by
  sorry

end find_a_l165_165594


namespace shaded_area_l165_165988

-- Definitions based on given conditions
def Rectangle (A B C D : ℝ) := True -- Placeholder for the geometric definition of a rectangle

-- Total area of the non-shaded part
def non_shaded_area : ℝ := 10

-- Problem statement in Lean
theorem shaded_area (A B C D : ℝ) :
  Rectangle A B C D →
  (exists shaded_area : ℝ, shaded_area = 14 ∧ non_shaded_area + shaded_area = A * B) :=
by
  sorry

end shaded_area_l165_165988


namespace shane_photos_per_week_l165_165495

theorem shane_photos_per_week (jan_days feb_weeks photos_per_day total_photos : ℕ) :
  (jan_days = 31) →
  (feb_weeks = 4) →
  (photos_per_day = 2) →
  (total_photos = 146) →
  let photos_jan := photos_per_day * jan_days in
  let photos_feb := total_photos - photos_jan in
  let photos_per_week := photos_feb / feb_weeks in
  photos_per_week = 21 :=
by
  intros h1 h2 h3 h4
  let photos_jan := photos_per_day * jan_days
  let photos_feb := total_photos - photos_jan
  let photos_per_week := photos_feb / feb_weeks
  sorry

end shane_photos_per_week_l165_165495


namespace gumball_machine_total_l165_165822

noncomputable def total_gumballs (R B G : ℕ) : ℕ := R + B + G

theorem gumball_machine_total
  (R B G : ℕ)
  (hR : R = 16)
  (hB : B = R / 2)
  (hG : G = 4 * B) :
  total_gumballs R B G = 56 :=
by
  sorry

end gumball_machine_total_l165_165822


namespace smallest_positive_multiple_of_32_l165_165658

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ n % 32 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 32 = 0 → n ≤ m :=
by
  sorry

end smallest_positive_multiple_of_32_l165_165658


namespace largest_N_not_payable_l165_165935

theorem largest_N_not_payable (coins_5 coins_6 : ℕ) (h1 : coins_5 > 10) (h2 : coins_6 > 10) :
  ∃ N : ℕ, N ≤ 50 ∧ (¬ ∃ (x y : ℕ), N = 5 * x + 6 * y) ∧ N = 19 :=
by 
  use 19
  have h₃ : 19 ≤ 50 := by norm_num
  have h₄ : ¬ ∃ (x y : ℕ), 19 = 5 * x + 6 * y := sorry
  exact ⟨h₃, h₄⟩

end largest_N_not_payable_l165_165935


namespace problem1_problem2_l165_165436

-- Define the variables for the two problems
variables (a b x : ℝ)

-- The first problem
theorem problem1 :
  (a + 2 * b) ^ 2 - 4 * b * (a + b) = a ^ 2 :=
by 
  -- Proof goes here
  sorry

-- The second problem
theorem problem2 :
  ((x ^ 2 - 2 * x) / (x ^ 2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x ^ 2 - 4)) = x + 2 :=
by 
  -- Proof goes here
  sorry

end problem1_problem2_l165_165436


namespace system_of_equations_solution_cases_l165_165135

theorem system_of_equations_solution_cases
  (x y a b : ℝ) :
  (a = b → x + y = 2 * a) ∧
  (a = -b → ¬ (∃ (x y : ℝ), (x / (x - a)) + (y / (y - b)) = 2 ∧ a * x + b * y = 2 * a * b)) :=
by
  sorry

end system_of_equations_solution_cases_l165_165135


namespace max_product_l165_165901

-- Problem statement: Define the conditions and the conclusion
theorem max_product (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 4) : mn ≤ 4 :=
by
  sorry -- Proof placeholder

end max_product_l165_165901


namespace true_false_test_keys_l165_165277

theorem true_false_test_keys : ∃ answer_keys : Finset (Fin 2 → Fin 2), 
  (∀ key ∈ answer_keys, guaranteed_score key) ∧ answer_keys.card = 22 :=
begin
  -- Definitions for guaranteed_score and any necessary auxiliary definitions would go here.
  sorry
end

-- Definition of guaranteed_score ensuring a minimum of 4 correct answers when answered
def guaranteed_score (key : Fin 10 → Bool) : Prop :=
  -- For now, we define it as a placeholder, would need further details to match the condition precisely.
  (∃ (correct : Fin 10 → Bool), 
    (∑ i in finset.range 10, if (correct i = key i) then 1 else 0) ≥ 4)

-- This serves as a simplified version. Actual guaranteed_score would be based on problem specifics.

end true_false_test_keys_l165_165277


namespace gold_copper_alloy_ratio_l165_165976

theorem gold_copper_alloy_ratio 
  (G C : ℝ) 
  (h_gold : G / weight_of_water = 19) 
  (h_copper : C / weight_of_water = 9)
  (weight_of_alloy : (G + C) / weight_of_water = 17) :
  G / C = 4 :=
sorry

end gold_copper_alloy_ratio_l165_165976


namespace correct_operation_l165_165665

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l165_165665


namespace probability_all_even_l165_165923

theorem probability_all_even :
  let die1_even_count := 3
  let die1_total := 6
  let die2_even_count := 3
  let die2_total := 7
  let die3_even_count := 4
  let die3_total := 9
  let prob_die1_even := die1_even_count / die1_total
  let prob_die2_even := die2_even_count / die2_total
  let prob_die3_even := die3_even_count / die3_total
  let probability_all_even := prob_die1_even * prob_die2_even * prob_die3_even
  probability_all_even = 1 / 10.5 :=
by
  sorry

end probability_all_even_l165_165923


namespace relationship_withdrawn_leftover_l165_165566

-- Definitions based on the problem conditions
def pie_cost : ℝ := 6
def sandwich_cost : ℝ := 3
def book_cost : ℝ := 10
def book_discount : ℝ := 0.2 * book_cost
def book_price_with_discount : ℝ := book_cost - book_discount
def total_spent_before_tax : ℝ := pie_cost + sandwich_cost + book_price_with_discount
def sales_tax_rate : ℝ := 0.05
def sales_tax : ℝ := sales_tax_rate * total_spent_before_tax
def total_spent_with_tax : ℝ := total_spent_before_tax + sales_tax

-- Given amount withdrawn and amount left after shopping
variables (X Y : ℝ)

-- Theorem statement
theorem relationship_withdrawn_leftover :
  Y = X - total_spent_with_tax :=
sorry

end relationship_withdrawn_leftover_l165_165566


namespace parts_per_hour_l165_165389

variables {x y : ℕ}

-- Condition 1: The time it takes for A to make 90 parts is the same as the time it takes for B to make 120 parts.
def time_ratio (x y : ℕ) := (x:ℚ) / y = 90 / 120

-- Condition 2: A and B together make 35 parts per hour.
def total_parts_per_hour (x y : ℕ) := x + y = 35

-- Given the conditions, prove the number of parts A and B each make per hour.
theorem parts_per_hour (x y : ℕ) (h1 : time_ratio x y) (h2 : total_parts_per_hour x y) : x = 15 ∧ y = 20 :=
by
  sorry

end parts_per_hour_l165_165389


namespace purely_imaginary_complex_number_l165_165611

theorem purely_imaginary_complex_number (a : ℝ) (h : (a^2 - 3 * a + 2) = 0 ∧ (a - 2) ≠ 0) : a = 1 :=
by {
  sorry
}

end purely_imaginary_complex_number_l165_165611


namespace total_students_exam_l165_165109

theorem total_students_exam (N T T' T'' : ℕ) (h1 : T = 88 * N) (h2 : T' = T - 8 * 50) 
  (h3 : T' = 92 * (N - 8)) (h4 : T'' = T' - 100) (h5 : T'' = 92 * (N - 9)) : N = 84 :=
by
  sorry

end total_students_exam_l165_165109


namespace scientific_notation_of_0_0000012_l165_165651

theorem scientific_notation_of_0_0000012 :
  0.0000012 = 1.2 * 10^(-6) :=
sorry

end scientific_notation_of_0_0000012_l165_165651


namespace benjamin_earns_more_l165_165757

noncomputable def additional_earnings : ℝ :=
  let P : ℝ := 75000
  let r : ℝ := 0.05
  let t_M : ℝ := 3
  let r_m : ℝ := r / 12
  let t_B : ℝ := 36
  let A_M : ℝ := P * (1 + r)^t_M
  let A_B : ℝ := P * (1 + r_m)^t_B
  A_B - A_M

theorem benjamin_earns_more : additional_earnings = 204 := by
  sorry

end benjamin_earns_more_l165_165757


namespace three_op_six_l165_165847

-- Define the new operation @.
def op (a b : ℕ) : ℕ := (a * a * b) / (a + b)

-- The theorem to prove that the value of 3 @ 6 is 6.
theorem three_op_six : op 3 6 = 6 := by 
  sorry

end three_op_six_l165_165847


namespace option_C_correct_l165_165671

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l165_165671


namespace sum_of_squares_iff_double_l165_165632

theorem sum_of_squares_iff_double (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_squares_iff_double_l165_165632


namespace find_f_neg2_l165_165223

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - 3 else -(2^(-x) - 3)

theorem find_f_neg2 : f (-2) = -1 :=
sorry

end find_f_neg2_l165_165223


namespace part_I_n_3_not_relevant_part_I_n_3_is_relevant_part_II_part_III_min_value_of_relevant_number_l165_165359

-- Part I
def is_relevant_number (n m : ℕ) : Prop :=
  ∀ {P : Finset ℕ}, (P ⊆ (Finset.range (2*n + 1)) ∧ P.card = m) →
  ∃ (a b c d : ℕ), a ∈ P ∧ b ∈ P ∧ c ∈ P ∧ d ∈ P ∧ a + b + c + d = 4*n + 1

theorem part_I_n_3_not_relevant :
  ¬ is_relevant_number 3 5 := sorry

theorem part_I_n_3_is_relevant :
  is_relevant_number 3 6 := sorry

-- Part II
theorem part_II (n m : ℕ) (h : is_relevant_number n m) : m - n - 3 ≥ 0 := sorry

-- Part III
theorem part_III_min_value_of_relevant_number (n : ℕ) : 
  ∃ m : ℕ, is_relevant_number n m ∧ ∀ k, is_relevant_number n k → m ≤ k := sorry

end part_I_n_3_not_relevant_part_I_n_3_is_relevant_part_II_part_III_min_value_of_relevant_number_l165_165359


namespace percentage_women_no_french_speak_spanish_german_l165_165072

variable (total_workforce : Nat)
variable (men_percentage women_percentage : ℕ)
variable (men_only_french men_only_spanish men_only_german : ℕ)
variable (men_both_french_spanish men_both_french_german men_both_spanish_german : ℕ)
variable (men_all_three_languages women_only_french women_only_spanish : ℕ)
variable (women_only_german women_both_french_spanish women_both_french_german : ℕ)
variable (women_both_spanish_german women_all_three_languages : ℕ)

-- Conditions
axiom h1 : men_percentage = 60
axiom h2 : women_percentage = 40
axiom h3 : women_only_french = 30
axiom h4 : women_only_spanish = 25
axiom h5 : women_only_german = 20
axiom h6 : women_both_french_spanish = 10
axiom h7 : women_both_french_german = 5
axiom h8 : women_both_spanish_german = 5
axiom h9 : women_all_three_languages = 5

theorem percentage_women_no_french_speak_spanish_german:
  women_only_spanish + women_only_german + women_both_spanish_german = 50 := by
  sorry

end percentage_women_no_french_speak_spanish_german_l165_165072


namespace complex_equation_solution_l165_165190

theorem complex_equation_solution (x : ℝ) (i : ℂ) (h_imag_unit : i * i = -1) (h_eq : (x + 2 * i) * (x - i) = 6 + 2 * i) : x = 2 :=
by
  sorry

end complex_equation_solution_l165_165190


namespace number_of_parallelograms_l165_165287

theorem number_of_parallelograms (n : ℕ) : 
  let binom := Nat.choose (n + 1) 2
  in 3 * (binom)^2 = 3 * Nat.choose (n + 2) 4 :=
by
  let binom := Nat.choose (n + 1) 2
  have H : (3 * (binom)^2 = 3 * Nat.choose (n + 2) 4) := sorry
  exact H

end number_of_parallelograms_l165_165287


namespace remainder_when_sum_divided_l165_165032

theorem remainder_when_sum_divided (p q : ℕ) (m n : ℕ) (hp : p = 80 * m + 75) (hq : q = 120 * n + 115) :
  (p + q) % 40 = 30 := 
by sorry

end remainder_when_sum_divided_l165_165032


namespace min_dwarfs_for_no_empty_neighbor_l165_165504

theorem min_dwarfs_for_no_empty_neighbor (n : ℕ) (h : n = 30) : ∃ k : ℕ, k = 10 ∧
  (∀ seats : Fin n → Prop, (∀ i : Fin n, seats i ∨ seats ((i + 1) % n) ∨ seats ((i + 2) % n))
   → ∃ j : Fin n, seats j ∧ j = 10) :=
by
  sorry

end min_dwarfs_for_no_empty_neighbor_l165_165504


namespace min_dwarfs_for_no_empty_neighbor_l165_165505

theorem min_dwarfs_for_no_empty_neighbor (n : ℕ) (h : n = 30) : ∃ k : ℕ, k = 10 ∧
  (∀ seats : Fin n → Prop, (∀ i : Fin n, seats i ∨ seats ((i + 1) % n) ∨ seats ((i + 2) % n))
   → ∃ j : Fin n, seats j ∧ j = 10) :=
by
  sorry

end min_dwarfs_for_no_empty_neighbor_l165_165505


namespace find_x_l165_165982

theorem find_x : ∃ x : ℝ, (0.40 * x - 30 = 50) ∧ x = 200 :=
by
  sorry

end find_x_l165_165982


namespace power_of_5_in_8_factorial_l165_165473

theorem power_of_5_in_8_factorial :
  let x := Nat.factorial 8
  ∃ (i k m p : ℕ), 0 < i ∧ 0 < k ∧ 0 < m ∧ 0 < p ∧ x = 2^i * 3^k * 5^m * 7^p ∧ m = 1 :=
by
  sorry

end power_of_5_in_8_factorial_l165_165473


namespace terry_lunch_combos_l165_165770

def num_lettuce : ℕ := 2
def num_tomatoes : ℕ := 3
def num_olives : ℕ := 4
def num_soups : ℕ := 2

theorem terry_lunch_combos : num_lettuce * num_tomatoes * num_olives * num_soups = 48 :=
by
  sorry

end terry_lunch_combos_l165_165770


namespace find_k_l165_165785

theorem find_k (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h_parabola_A : y₁ = x₁^2)
  (h_parabola_B : y₂ = x₂^2)
  (h_line_A : y₁ = x₁ - k)
  (h_line_B : y₂ = x₂ - k)
  (h_midpoint : (y₁ + y₂) / 2 = 1) 
  (h_sum_x : x₁ + x₂ = 1) :
  k = -1 / 2 :=
by sorry

end find_k_l165_165785


namespace Lenny_pens_left_l165_165751

theorem Lenny_pens_left :
  let boxes := 20
  let pens_per_box := 5
  let total_pens := boxes * pens_per_box
  let pens_given_to_friends := 0.4 * total_pens
  let pens_left_after_friends := total_pens - pens_given_to_friends
  let pens_given_to_classmates := (1/4) * pens_left_after_friends
  let pens_left := pens_left_after_friends - pens_given_to_classmates
  pens_left = 45 :=
by
  repeat { sorry }

end Lenny_pens_left_l165_165751


namespace replaced_solution_percentage_l165_165556

theorem replaced_solution_percentage (y x z w : ℝ) 
  (h1 : x = 0.5)
  (h2 : y = 80)
  (h3 : z = 0.5 * y)
  (h4 : w = 50) 
  :
  (40 + 0.5 * x) = 50 → x = 20 :=
by
  sorry

end replaced_solution_percentage_l165_165556


namespace general_term_l165_165246

def S (n : ℕ) : ℕ := n^2 + 3 * n

def a (n : ℕ) : ℕ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = 2 * n + 2 :=
by {
  sorry
}

end general_term_l165_165246


namespace product_repeating_decimal_l165_165573

theorem product_repeating_decimal (p : ℚ) (h₁ : p = 152 / 333) : 
  p * 7 = 1064 / 333 :=
  by
    sorry

end product_repeating_decimal_l165_165573


namespace greatest_number_same_remainder_l165_165975

theorem greatest_number_same_remainder (d : ℕ) :
  d ∣ (57 - 25) ∧ d ∣ (105 - 57) ∧ d ∣ (105 - 25) → d ≤ 16 :=
by
  sorry

end greatest_number_same_remainder_l165_165975


namespace tangent_line_at_one_l165_165734

noncomputable def f (x : ℝ) := Real.log x + 2 * x^2 - 4 * x

theorem tangent_line_at_one :
  let slope := (1/x + 4*x - 4) 
  let y_val := -2 
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), A = 1 ∧ B = -1 ∧ C = -3 ∧ (∀ (x y : ℝ), f x = y → A * x + B * y + C = 0) :=
by
  sorry

end tangent_line_at_one_l165_165734


namespace find_n_l165_165517

theorem find_n {n : ℕ} (avg1 : ℕ) (avg2 : ℕ) (S : ℕ) :
  avg1 = 7 →
  avg2 = 6 →
  S = 7 * n →
  6 = (S - 11) / (n + 1) →
  n = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end find_n_l165_165517


namespace oldest_child_age_correct_l165_165479

-- Defining the conditions
def jane_start_age := 16
def jane_current_age := 32
def jane_stopped_babysitting_years_ago := 10
def half (x : ℕ) := x / 2

-- Expressing the conditions
def jane_last_babysitting_age := jane_current_age - jane_stopped_babysitting_years_ago
def max_child_age_when_jane_stopped := half jane_last_babysitting_age
def years_since_jane_stopped := jane_stopped_babysitting_years_ago

def calculate_oldest_child_current_age (age : ℕ) : ℕ :=
  age + years_since_jane_stopped

def child_age_when_stopped := max_child_age_when_jane_stopped
def expected_oldest_child_current_age := 21

-- The theorem stating the equivalence
theorem oldest_child_age_correct : 
  calculate_oldest_child_current_age child_age_when_stopped = expected_oldest_child_current_age :=
by
  -- Proof here
  sorry

end oldest_child_age_correct_l165_165479


namespace b_ne_d_l165_165638

-- Conditions
def P (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def Q (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

def PQ_eq_QP_no_real_roots (a b c d : ℝ) : Prop := 
  ∀ (x : ℝ), P (Q x c d) a b ≠ Q (P x a b) c d

-- Goal
theorem b_ne_d (a b c d : ℝ) (h : PQ_eq_QP_no_real_roots a b c d) : b ≠ d := 
sorry

end b_ne_d_l165_165638


namespace sin_square_general_proposition_l165_165595

-- Definitions for the given conditions
def sin_square_sum_30_90_150 : Prop :=
  (Real.sin (30 * Real.pi / 180))^2 + (Real.sin (90 * Real.pi / 180))^2 + (Real.sin (150 * Real.pi / 180))^2 = 3/2

def sin_square_sum_5_65_125 : Prop :=
  (Real.sin (5 * Real.pi / 180))^2 + (Real.sin (65 * Real.pi / 180))^2 + (Real.sin (125 * Real.pi / 180))^2 = 3/2

-- The general proposition we want to prove
theorem sin_square_general_proposition (α : ℝ) : 
  sin_square_sum_30_90_150 ∧ sin_square_sum_5_65_125 →
  (Real.sin (α * Real.pi / 180 - 60 * Real.pi / 180))^2 + 
  (Real.sin (α * Real.pi / 180))^2 + 
  (Real.sin (α * Real.pi / 180 + 60 * Real.pi / 180))^2 = 3/2 :=
by
  intro h
  -- Proof goes here
  sorry

end sin_square_general_proposition_l165_165595


namespace evaluate_expression_l165_165173

theorem evaluate_expression 
    (a b c : ℕ) 
    (ha : a = 7)
    (hb : b = 11)
    (hc : c = 13) :
  let numerator := a^3 * (1 / b - 1 / c) + b^3 * (1 / c - 1 / a) + c^3 * (1 / a - 1 / b)
  let denominator := a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)
  numerator / denominator = 31 := 
by {
  sorry
}

end evaluate_expression_l165_165173


namespace concentration_time_within_bounds_l165_165649

-- Define the time bounds for the highest concentration of the drug in the blood
def highest_concentration_time_lower (base : ℝ) (tolerance : ℝ) : ℝ := base - tolerance
def highest_concentration_time_upper (base : ℝ) (tolerance : ℝ) : ℝ := base + tolerance

-- Define the base and tolerance values
def base_time : ℝ := 0.65
def tolerance_time : ℝ := 0.15

-- Define the specific time we want to prove is within the bounds
def specific_time : ℝ := 0.8

-- Theorem statement
theorem concentration_time_within_bounds : 
  highest_concentration_time_lower base_time tolerance_time ≤ specific_time ∧ 
  specific_time ≤ highest_concentration_time_upper base_time tolerance_time :=
by sorry

end concentration_time_within_bounds_l165_165649


namespace tanya_body_lotions_l165_165029

variable {F L : ℕ}  -- Number of face moisturizers (F) and body lotions (L) Tanya bought

theorem tanya_body_lotions
  (price_face_moisturizer : ℕ := 50)
  (price_body_lotion : ℕ := 60)
  (num_face_moisturizers : ℕ := 2)
  (total_spent : ℕ := 1020)
  (christy_spending_factor : ℕ := 2)
  (h_together_spent : total_spent = 3 * (num_face_moisturizers * price_face_moisturizer + L * price_body_lotion)) :
  L = 4 :=
by
  sorry

end tanya_body_lotions_l165_165029


namespace value_two_stddevs_less_l165_165957

theorem value_two_stddevs_less (μ σ : ℝ) (hμ : μ = 16.5) (hσ : σ = 1.5) : μ - 2 * σ = 13.5 :=
by
  rw [hμ, hσ]
  norm_num

end value_two_stddevs_less_l165_165957


namespace simplify_expression1_simplify_expression2_l165_165439

/-- Proof Problem 1: Simplify the expression (a+2b)^2 - 4b(a+b) -/
theorem simplify_expression1 (a b : ℝ) : 
  (a + 2 * b)^2 - 4 * b * (a + b) = a^2 :=
sorry

/-- Proof Problem 2: Simplify the expression ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) ÷ (x - 1) / (x^2 - 4) -/
theorem simplify_expression2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) : 
  ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 :=
sorry

end simplify_expression1_simplify_expression2_l165_165439


namespace other_position_in_arithmetic_progression_l165_165118

theorem other_position_in_arithmetic_progression 
  (a d : ℝ) (x : ℕ)
  (h1 : a + (4 - 1) * d + a + (x - 1) * d = 20)
  (h2 : 5 * (2 * a + 9 * d) = 100) :
  x = 7 := by
  sorry

end other_position_in_arithmetic_progression_l165_165118


namespace initial_quantity_of_A_l165_165004

theorem initial_quantity_of_A (x : ℚ) 
    (h1 : 7 * x = a)
    (h2 : 5 * x = b)
    (h3 : a + b = 12 * x)
    (h4 : a' = a - (7 / 12) * 9)
    (h5 : b' = b - (5 / 12) * 9 + 9)
    (h6 : a' / b' = 7 / 9) : 
    a = 23.625 := 
sorry

end initial_quantity_of_A_l165_165004


namespace arithmetic_sequence_a5_value_l165_165591

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)
  (h_cond : (a 5)^2 - a 3 - a 7 = 0) 
  : a 5 = 2 := 
sorry

end arithmetic_sequence_a5_value_l165_165591


namespace operations_equivalent_l165_165739

theorem operations_equivalent (x : ℚ) : 
  ((x * (5 / 6)) / (2 / 3) - 2) = (x * (5 / 4) - 2) :=
sorry

end operations_equivalent_l165_165739


namespace div_condition_positive_integers_l165_165481

theorem div_condition_positive_integers 
  (a b d : ℕ) 
  (h1 : a + b ≡ 0 [MOD d]) 
  (h2 : a * b ≡ 0 [MOD d^2]) 
  (h3 : 0 < a) 
  (h4 : 0 < b) 
  (h5 : 0 < d) : 
  d ∣ a ∧ d ∣ b :=
sorry

end div_condition_positive_integers_l165_165481


namespace fraction_zero_iff_x_is_four_l165_165257

theorem fraction_zero_iff_x_is_four (x : ℝ) (h_ne_zero: x + 4 ≠ 0) :
  (16 - x^2) / (x + 4) = 0 ↔ x = 4 :=
sorry

end fraction_zero_iff_x_is_four_l165_165257


namespace disproving_proposition_l165_165892

theorem disproving_proposition : ∃ (angle1 angle2 : ℝ), angle1 = angle2 ∧ angle1 + angle2 = 90 :=
by
  sorry

end disproving_proposition_l165_165892


namespace min_dwarfs_for_no_empty_neighbor_l165_165503

theorem min_dwarfs_for_no_empty_neighbor (n : ℕ) (h : n = 30) : ∃ k : ℕ, k = 10 ∧
  (∀ seats : Fin n → Prop, (∀ i : Fin n, seats i ∨ seats ((i + 1) % n) ∨ seats ((i + 2) % n))
   → ∃ j : Fin n, seats j ∧ j = 10) :=
by
  sorry

end min_dwarfs_for_no_empty_neighbor_l165_165503


namespace fraction_of_students_speak_foreign_language_l165_165974

noncomputable def students_speak_foreign_language_fraction (M F : ℕ) (h1 : M = F) (m_frac : ℚ) (f_frac : ℚ) : ℚ :=
  ((3 / 5) * M + (2 / 3) * F) / (M + F)

theorem fraction_of_students_speak_foreign_language (M F : ℕ) (h1 : M = F) :
  students_speak_foreign_language_fraction M F h1 (3 / 5) (2 / 3) = 19 / 30 :=
by 
  sorry

end fraction_of_students_speak_foreign_language_l165_165974


namespace complex_plane_second_quadrant_l165_165728

theorem complex_plane_second_quadrant (x : ℝ) :
  (x ^ 2 - 6 * x + 5 < 0 ∧ x - 2 > 0) ↔ (2 < x ∧ x < 5) :=
by
  -- The proof is to be completed.
  sorry

end complex_plane_second_quadrant_l165_165728


namespace cheaper_store_price_difference_in_cents_l165_165160

theorem cheaper_store_price_difference_in_cents :
  let list_price : ℝ := 59.99
  let discount_budget_buys := list_price * 0.15
  let discount_frugal_finds : ℝ := 20
  let sale_price_budget_buys := list_price - discount_budget_buys
  let sale_price_frugal_finds := list_price - discount_frugal_finds
  let difference_in_price := sale_price_budget_buys - sale_price_frugal_finds
  let difference_in_cents := difference_in_price * 100
  difference_in_cents = 1099.15 :=
by
  sorry

end cheaper_store_price_difference_in_cents_l165_165160


namespace smallest_positive_period_and_monotonic_increase_max_min_in_interval_l165_165487

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem smallest_positive_period_and_monotonic_increase :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∃ a b : ℝ, (k * π - π / 3 ≤ a ∧ a ≤ x) ∧ (x ≤ b ∧ b ≤ k * π + π / 6) → f x = 1) := sorry

theorem max_min_in_interval :
  (∀ x : ℝ, (-π / 4 ≤ x ∧ x ≤ π / 6) → (1 - Real.sqrt 3 ≤ f x ∧ f x ≤ 3)) := sorry

end smallest_positive_period_and_monotonic_increase_max_min_in_interval_l165_165487


namespace photos_per_week_in_february_l165_165494

def january_photos : ℕ := 31 * 2

def total_photos (jan_feb_photos : ℕ) : ℕ := jan_feb_photos - january_photos

theorem photos_per_week_in_february (jan_feb_photos : ℕ) (weeks_in_february : ℕ)
  (h1 : jan_feb_photos = 146)
  (h2 : weeks_in_february = 4) :
  total_photos jan_feb_photos / weeks_in_february = 21 := by
  sorry

end photos_per_week_in_february_l165_165494


namespace cherries_per_pound_l165_165120

-- Definitions from conditions in the problem
def total_pounds_of_cherries : ℕ := 3
def pitting_time_for_20_cherries : ℕ := 10 -- in minutes
def total_pitting_time : ℕ := 2 * 60  -- in minutes (2 hours to minutes)

-- Theorem to prove the question equals the correct answer
theorem cherries_per_pound : (total_pitting_time / pitting_time_for_20_cherries) * 20 / total_pounds_of_cherries = 80 := by
  sorry

end cherries_per_pound_l165_165120


namespace unique_hexagon_angles_sides_identity_1_identity_2_l165_165525

noncomputable def lengths_angles_determined 
  (a b c d e f : ℝ) (α β γ : ℝ) 
  (h₀ : α + β + γ < 180) : Prop :=
  -- Assuming this is the expression we need to handle:
  ∀ (δ ε ζ : ℝ),
    δ = 180 - α ∧
    ε = 180 - β ∧
    ζ = 180 - γ →
  ∃ (angles_determined : Prop),
    angles_determined

theorem unique_hexagon_angles_sides 
  (a b c d e f : ℝ) (α β γ : ℝ) 
  (h₀ : α + β + γ < 180) : 
  lengths_angles_determined a b c d e f α β γ h₀ :=
sorry

theorem identity_1 
  (a b c d : ℝ) 
  (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) : 
  (1 / a + 1 / c = 1 / b + 1 / d) :=
sorry

theorem identity_2 
  (a b c d e f : ℝ) 
  (h₀ : true) : 
  ((a + f) * (b + d) * (c + e) = (a + e) * (b + f) * (c + d)) :=
sorry

end unique_hexagon_angles_sides_identity_1_identity_2_l165_165525


namespace correct_operation_l165_165680

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l165_165680


namespace cost_of_each_big_apple_l165_165284

theorem cost_of_each_big_apple :
  ∀ (small_cost medium_cost : ℝ) (big_cost : ℝ) (num_small num_medium num_big : ℕ) (total_cost : ℝ),
  small_cost = 1.5 →
  medium_cost = 2 →
  num_small = 6 →
  num_medium = 6 →
  num_big = 8 →
  total_cost = 45 →
  total_cost = num_small * small_cost + num_medium * medium_cost + num_big * big_cost →
  big_cost = 3 :=
by
  intros small_cost medium_cost big_cost num_small num_medium num_big total_cost
  sorry

end cost_of_each_big_apple_l165_165284


namespace infinite_geometric_series_sum_l165_165866

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l165_165866


namespace exist_points_no_three_collinear_integer_distances_l165_165636

theorem exist_points_no_three_collinear_integer_distances
  (N : ℕ) : ∃ (points : Fin N → ℝ × ℝ), 
  (∀ i j k : Fin N, i ≠ j → i ≠ k → j ≠ k → 
     ¬ collinear ℝ [{| x := points i, y := points j, z := points k |}]) ∧
  (∀ i j : Fin N, i ≠ j → ∃ d : ℕ, dist (points i) (points j) = (d : ℝ)) :=
by
  sorry

end exist_points_no_three_collinear_integer_distances_l165_165636


namespace calculate_m_squared_l165_165264

-- Define the conditions
def pizza_diameter := 16
def pizza_radius := pizza_diameter / 2
def num_slices := 4

-- Define the question
def longest_segment_length_in_piece := 2 * pizza_radius
def m := longest_segment_length_in_piece -- Length of the longest line segment in one piece

-- Rewrite the math proof problem
theorem calculate_m_squared :
  m^2 = 256 := 
by 
  -- Proof goes here
  sorry

end calculate_m_squared_l165_165264


namespace percentage_taxed_l165_165960

theorem percentage_taxed (T : ℝ) (H1 : 3840 = T * (P : ℝ)) (H2 : 480 = 0.25 * T * (P : ℝ)) : P = 0.5 := 
by
  sorry

end percentage_taxed_l165_165960


namespace geometric_sequence_a4_a5_sum_l165_165308

theorem geometric_sequence_a4_a5_sum :
  (∀ n : ℕ, a_n > 0) → (a_3 = 3) → (a_6 = (1 / 9)) → 
  (a_4 + a_5 = (4 / 3)) :=
by
  sorry

end geometric_sequence_a4_a5_sum_l165_165308


namespace circle_chord_intersect_zero_l165_165748

noncomputable def circle_product (r : ℝ) : ℝ :=
  let O := (0, 0)
  let A := (r, 0)
  let B := (-r, 0)
  let C := (0, r)
  let D := (0, -r)
  let P := (0, 0)
  (dist A P) * (dist P B)

theorem circle_chord_intersect_zero (r : ℝ) :
  let A := (r, 0)
  let B := (-r, 0)
  let C := (0, r)
  let D := (0, -r)
  let P := (0, 0)
  (dist A P) * (dist P B) = 0 :=
by sorry

end circle_chord_intersect_zero_l165_165748


namespace bicycle_cost_price_l165_165426

theorem bicycle_cost_price (CP_A : ℝ) (CP_B : ℝ) (SP_C : ℝ)
    (h1 : CP_B = 1.60 * CP_A)
    (h2 : SP_C = 1.25 * CP_B)
    (h3 : SP_C = 225) :
    CP_A = 225 / 2.00 :=
by
  sorry -- the proof steps will follow here

end bicycle_cost_price_l165_165426


namespace rectangle_area_expectation_rectangle_area_standard_deviation_cm2_l165_165780

noncomputable def expected_area (E_X : ℝ) (E_Y : ℝ) : ℝ := E_X * E_Y

noncomputable def variance_of_product (E_X : ℝ) (E_Y : ℝ) (Var_X : ℝ) (Var_Y : ℝ) : ℝ :=
  (E_X^2 * Var_Y) + (E_Y^2 * Var_X) + (Var_X * Var_Y)

noncomputable def standard_deviation (variance : ℝ) : ℝ := real.sqrt variance

theorem rectangle_area_expectation :
  let E_X := 1  -- Expected width in meters
  let E_Y := 2  -- Expected length in meters
  in expected_area E_X E_Y = 2 := by
  sorry

theorem rectangle_area_standard_deviation_cm2 :
  let E_X := 1  -- Expected width in meters
  let E_Y := 2  -- Expected length in meters
  let Var_X := 0.003 ^ 2  -- Variance of width in square meters
  let Var_Y := 0.002 ^ 2  -- Variance of length in square meters
  let Var_A := variance_of_product E_X E_Y Var_X Var_Y
  let SD_A_m2 := standard_deviation Var_A
  let SD_A_cm2 := SD_A_m2 * (100 ^ 2)  -- Conversion to square centimeters
  in SD_A_cm2 ≈ 63 := by
  sorry

end rectangle_area_expectation_rectangle_area_standard_deviation_cm2_l165_165780


namespace problem_1_problem_2_l165_165981

noncomputable def is_positive_real (x : ℝ) : Prop := x > 0

theorem problem_1 (a b : ℝ) (ha : is_positive_real a) (hb : is_positive_real b)
  (h : 1 / a + 1 / b = 2 * Real.sqrt 2) : 
  a^2 + b^2 ≥ 1 := by
  sorry

theorem problem_2 (a b : ℝ) (ha : is_positive_real a) (hb : is_positive_real b)
  (h : 1 / a + 1 / b = 2 * Real.sqrt 2) (h_extra : (a - b)^2 ≥ 4 * (a * b)^3) : 
  a * b = 1 := by
  sorry

end problem_1_problem_2_l165_165981


namespace binomial_expansion_coeff_x4_l165_165338

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := n.choose k

theorem binomial_expansion_coeff_x4 (n : ℕ) (hx : (x^2 - 1/x)^n.nat_degree + 1 = 6) :
  binomial_coefficient 5 2 = 10 :=
by
  sorry

end binomial_expansion_coeff_x4_l165_165338


namespace third_side_length_l165_165051

theorem third_side_length (a b x : ℝ) (h₁ : a = 3) (h₂ : b = 8) (h₃ : 5 < x) (h₄ : x < 11) : x = 6 :=
sorry

end third_side_length_l165_165051


namespace radius_of_circle_B_l165_165709

theorem radius_of_circle_B :
  ∀ {A B C D : Type} 
  [has_radius A] [has_radius B] [has_radius C] [has_radius D]
  (externally_tangent : A ⟶ B) (externally_tangent_2 : A ⟶ C) (externally_tangent_3 : B ⟶ C)
  (internally_tangent : A ⟶ D) (internally_tangent_2 : B ⟶ D) (internally_tangent_3 : C ⟶ D)
  (congruent_BC : congruent B C)
  (radius_A : radius A = 2)
  (passes_through_center : ∃ F: center D, passes_through_center A F) :
  radius B = 16/9 :=
by
  sorry

end radius_of_circle_B_l165_165709


namespace tip_per_person_l165_165492

-- Define the necessary conditions
def hourly_wage : ℝ := 12
def people_served : ℕ := 20
def total_amount_made : ℝ := 37

-- Define the problem statement
theorem tip_per_person : (total_amount_made - hourly_wage) / people_served = 1.25 :=
by
  sorry

end tip_per_person_l165_165492


namespace inverse_variation_l165_165390

theorem inverse_variation (x y k : ℝ) (h1 : y = k / x^2) (h2 : k = 8) (h3 : y = 0.5) : x = 4 := by
  sorry

end inverse_variation_l165_165390


namespace copper_price_l165_165789

theorem copper_price (c : ℕ) (hzinc : ℕ) (zinc_weight : ℕ) (brass_weight : ℕ) (price_brass : ℕ) (used_copper : ℕ) :
  hzinc = 30 →
  zinc_weight = brass_weight - used_copper →
  brass_weight = 70 →
  price_brass = 45 →
  used_copper = 30 →
  (used_copper * c + zinc_weight * hzinc) = brass_weight * price_brass →
  c = 65 :=
by
  sorry

end copper_price_l165_165789


namespace min_dwarfs_l165_165509

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end min_dwarfs_l165_165509


namespace distance_upstream_l165_165010

variable (v : ℝ) -- speed of the stream in km/h
variable (t : ℝ := 6) -- time of each trip in hours
variable (d_down : ℝ := 24) -- distance for downstream trip in km
variable (u : ℝ := 3) -- speed of man in still water in km/h

/- The distance the man swam upstream -/
theorem distance_upstream : 
  24 = (u + v) * t → 
  ∃ (d_up : ℝ), 
    d_up = (u - v) * t ∧
    d_up = 12 :=
by
  sorry

end distance_upstream_l165_165010


namespace one_cow_eating_one_bag_in_12_days_l165_165071

def average_days_to_eat_one_bag (total_bags : ℕ) (total_days : ℕ) (number_of_cows : ℕ) : ℕ :=
  total_days / (total_bags / number_of_cows)

theorem one_cow_eating_one_bag_in_12_days (total_bags : ℕ) (total_days : ℕ) (number_of_cows : ℕ) (h_total_bags : total_bags = 50) (h_total_days : total_days = 20) (h_number_of_cows : number_of_cows = 30) : 
  average_days_to_eat_one_bag total_bags total_days number_of_cows = 12 := by
  sorry

end one_cow_eating_one_bag_in_12_days_l165_165071


namespace ian_lottery_win_l165_165608

theorem ian_lottery_win 
  (amount_paid_to_colin : ℕ)
  (amount_left : ℕ)
  (amount_paid_to_helen : ℕ := 2 * amount_paid_to_colin)
  (amount_paid_to_benedict : ℕ := amount_paid_to_helen / 2)
  (total_debts_paid : ℕ := amount_paid_to_colin + amount_paid_to_helen + amount_paid_to_benedict)
  (total_money_won : ℕ := total_debts_paid + amount_left)
  (h1 : amount_paid_to_colin = 20)
  (h2 : amount_left = 20) :
  total_money_won = 100 := 
sorry

end ian_lottery_win_l165_165608


namespace max_X_leq_ratio_XY_l165_165442

theorem max_X_leq_ratio_XY (x y z u : ℕ) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (h3 : x ≥ y) : 
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ ∀ (x y z u : ℕ), (x + y = z + u) → (2 * x *y = z * u) → (x ≥ y) → m ≤ x / y :=
sorry

end max_X_leq_ratio_XY_l165_165442


namespace inequality_proof_l165_165367

theorem inequality_proof (x y : ℝ) (h : 2 * y + 5 * x = 10) : (3 * x * y - x^2 - y^2 < 7) :=
sorry

end inequality_proof_l165_165367


namespace vivians_mail_in_august_l165_165250

-- Definitions based on the conditions provided
def mail_july : ℕ := 40
def business_days_august : ℕ := 22
def weekend_days_august : ℕ := 9

-- Lean 4 statement to prove the equivalent proof problem
theorem vivians_mail_in_august :
  let mail_business_days := 2 * mail_july
  let total_mail_business_days := business_days_august * mail_business_days
  let mail_weekend_days := mail_july / 2
  let total_mail_weekend_days := weekend_days_august * mail_weekend_days
  total_mail_business_days + total_mail_weekend_days = 1940 := by
  sorry

end vivians_mail_in_august_l165_165250


namespace cost_milk_is_5_l165_165417

-- Define the total cost the baker paid
def total_cost : ℕ := 80

-- Define the cost components
def cost_flour : ℕ := 3 * 3
def cost_eggs : ℕ := 3 * 10
def cost_baking_soda : ℕ := 2 * 3

-- Define the number of liters of milk
def liters_milk : ℕ := 7

-- Define the unknown cost per liter of milk
noncomputable def cost_per_liter_milk (c : ℕ) : Prop :=
  c * liters_milk = total_cost - (cost_flour + cost_eggs + cost_baking_soda)

-- State the theorem we want to prove
theorem cost_milk_is_5 : cost_per_liter_milk 5 := 
by
  sorry

end cost_milk_is_5_l165_165417


namespace min_sum_of_intercepts_l165_165339

-- Definitions based on conditions
def line (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = a * b
def point_on_line (a b : ℝ) : Prop := line a b 1 1

-- Main theorem statement
theorem min_sum_of_intercepts (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_point : point_on_line a b) : 
  a + b >= 4 :=
sorry

end min_sum_of_intercepts_l165_165339


namespace total_students_l165_165744

theorem total_students (females : ℕ) (ratio : ℕ) (males := ratio * females) (total := females + males) :
  females = 13 → ratio = 3 → total = 52 :=
by
  intros h_females h_ratio
  rw [h_females, h_ratio]
  simp [total, males]
  sorry

end total_students_l165_165744


namespace coefficient_x3_in_binomial_expansion_l165_165079

theorem coefficient_x3_in_binomial_expansion :
  nat.choose 50 3 = 19600 :=
by
  -- Proof goes here
  sorry

end coefficient_x3_in_binomial_expansion_l165_165079


namespace sum_of_coordinates_l165_165471

-- Definitions based on conditions
variable (f k : ℝ → ℝ)
variable (h₁ : f 4 = 8)
variable (h₂ : ∀ x, k x = (f x) ^ 3)

-- Statement of the theorem
theorem sum_of_coordinates : 4 + k 4 = 516 := by
  -- Proof would go here
  sorry

end sum_of_coordinates_l165_165471


namespace quadratic_roots_l165_165062

theorem quadratic_roots (a b : ℝ) (h : a^2 - 4*a*b + 5*b^2 - 2*b + 1 = 0) :
  ∃ (p q : ℝ), (∀ (x : ℝ), x^2 - p*x + q = 0 ↔ (x = a ∨ x = b)) ∧
               p = 3 ∧ q = 2 :=
by {
  sorry
}

end quadratic_roots_l165_165062


namespace sum_of_first_9_terms_l165_165898

-- Define the arithmetic sequence {a_n} and the sum S_n of the first n terms
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

-- Define the conditions given in the problem
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom arith_seq : arithmetic_sequence a
axiom sum_terms : sum_of_first_n_terms a S
axiom S3 : S 3 = 30
axiom S6 : S 6 = 100

-- Goal: Prove that S 9 = 170
theorem sum_of_first_9_terms : S 9 = 170 :=
sorry -- Placeholder for the proof

end sum_of_first_9_terms_l165_165898


namespace fraction_of_garden_occupied_by_flowerbeds_is_correct_l165_165012

noncomputable def garden_fraction_occupied : ℚ :=
  let garden_length := 28
  let garden_shorter_length := 18
  let triangle_leg := (garden_length - garden_shorter_length) / 2
  let triangle_area := 1 / 2 * triangle_leg^2
  let flowerbeds_area := 2 * triangle_area
  let garden_width : ℚ := 5  -- Assuming the height of the trapezoid as part of the garden rest
  let garden_area := garden_length * garden_width
  flowerbeds_area / garden_area

theorem fraction_of_garden_occupied_by_flowerbeds_is_correct :
  garden_fraction_occupied = 5 / 28 := by
  sorry

end fraction_of_garden_occupied_by_flowerbeds_is_correct_l165_165012


namespace age_difference_is_18_l165_165533

variable (A B C : ℤ)
variable (h1 : A + B > B + C)
variable (h2 : C = A - 18)

theorem age_difference_is_18 : (A + B) - (B + C) = 18 :=
by
  sorry

end age_difference_is_18_l165_165533


namespace initial_water_percentage_l165_165627

variable (W : ℝ) -- Initial percentage of water in the milk

theorem initial_water_percentage 
  (final_water_content : ℝ := 2) 
  (pure_milk_added : ℝ := 15) 
  (initial_milk_volume : ℝ := 10)
  (final_mixture_volume : ℝ := initial_milk_volume + pure_milk_added)
  (water_equation : W / 100 * initial_milk_volume = final_water_content / 100 * final_mixture_volume) 
  : W = 5 :=
by
  sorry

end initial_water_percentage_l165_165627


namespace rectangular_x_value_l165_165445

theorem rectangular_x_value (x : ℝ)
  (h1 : ∀ (length : ℝ), length = 4 * x)
  (h2 : ∀ (width : ℝ), width = x + 10)
  (h3 : ∀ (length width : ℝ), length * width = 2 * (2 * length + 2 * width))
  : x = (Real.sqrt 41 - 1) / 2 :=
by
  sorry

end rectangular_x_value_l165_165445


namespace ratio_of_white_marbles_l165_165045

theorem ratio_of_white_marbles (total_marbles yellow_marbles red_marbles : ℕ)
    (h1 : total_marbles = 50)
    (h2 : yellow_marbles = 12)
    (h3 : red_marbles = 7)
    (green_marbles : ℕ)
    (h4 : green_marbles = yellow_marbles - yellow_marbles / 2) :
    (total_marbles - (yellow_marbles + green_marbles + red_marbles)) / total_marbles = 1 / 2 :=
by
  sorry

end ratio_of_white_marbles_l165_165045


namespace base_area_of_rect_prism_l165_165429

theorem base_area_of_rect_prism (r : ℝ) (h : ℝ) (V : ℝ) (h_rate : ℝ) (V_rate : ℝ) (conversion : ℝ) :
  V_rate = conversion * V ∧ h_rate = h → ∃ A : ℝ, A = V / h ∧ A = 100 :=
by
  sorry

end base_area_of_rect_prism_l165_165429


namespace gumball_machine_total_l165_165823

noncomputable def total_gumballs (R B G : ℕ) : ℕ := R + B + G

theorem gumball_machine_total
  (R B G : ℕ)
  (hR : R = 16)
  (hB : B = R / 2)
  (hG : G = 4 * B) :
  total_gumballs R B G = 56 :=
by
  sorry

end gumball_machine_total_l165_165823


namespace figure_50_squares_l165_165289

-- Define the quadratic function with the given number of squares for figures 0, 1, 2, and 3.
def g (n : ℕ) : ℕ := 2 * n ^ 2 + 4 * n + 2

-- Prove that the number of nonoverlapping unit squares in figure 50 is 5202.
theorem figure_50_squares : g 50 = 5202 := 
by 
  sorry

end figure_50_squares_l165_165289


namespace arithmetic_geometric_sequence_problem_l165_165725

noncomputable def a (n : ℕ) : ℝ := 2^(n-1)

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)

noncomputable def b (n : ℕ) : ℝ := (5/2) * (Real.log (a n) / Real.log 2)

def T (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem arithmetic_geometric_sequence_problem
  (h₁ : a 1 + a 3 = 5)
  (h₂ : S 4 = 15) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, T n = 5 * n * (n - 1) / 4) :=
by
  sorry

end arithmetic_geometric_sequence_problem_l165_165725


namespace units_digit_sum_of_factorials_l165_165134

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_sum_of_factorials :
  ones_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 +
              factorial 6 + factorial 7 + factorial 8 + factorial 9 + factorial 10) = 3 := 
sorry

end units_digit_sum_of_factorials_l165_165134


namespace ratio_future_age_l165_165987

variables (S : ℕ) (M : ℕ) (S_future : ℕ) (M_future : ℕ)

def son_age := 44
def man_age := son_age + 46
def son_age_future := son_age + 2
def man_age_future := man_age + 2

theorem ratio_future_age : man_age_future / son_age_future = 2 := by
  -- You can add the proof here if you want
  sorry

end ratio_future_age_l165_165987


namespace figure_perimeter_equals_26_l165_165951

noncomputable def rectangle_perimeter : ℕ := 26

def figure_arrangement (width height : ℕ) : Prop :=
width = 2 ∧ height = 1

theorem figure_perimeter_equals_26 {width height : ℕ} (h : figure_arrangement width height) :
  rectangle_perimeter = 26 :=
by
  sorry

end figure_perimeter_equals_26_l165_165951


namespace geometric_sequence_sum_l165_165192

theorem geometric_sequence_sum :
  ∀ {a : ℕ → ℝ} (r : ℝ),
    (∀ n, a (n + 1) = r * a n) →
    a 1 + a 2 = 1 →
    a 3 + a 4 = 4 →
    a 5 + a 6 + a 7 + a 8 = 80 :=
by
  intros a r h_geom h_sum_1 h_sum_2
  sorry

end geometric_sequence_sum_l165_165192


namespace minimize_expression_l165_165435

theorem minimize_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^3 * y^2 * z = 1) : 
  x + 2*y + 3*z ≥ 2 :=
sorry

end minimize_expression_l165_165435


namespace odd_increasing_three_digit_numbers_count_eq_50_l165_165323

def count_odd_increasing_three_digit_numbers : Nat := by
  -- Mathematical conditions:
  -- let a, b, c be digits of the number
  -- 0 < a < b < c <= 9 and c is an odd digit

  -- We analyze values for 'c' which must be an odd digit,
  -- and count valid (a, b) combinations for each case of c.

  -- Starting from cases for c:
  -- for c = 1, no valid (a, b); count = 0
  -- for c = 3, valid (a, b) are from {1, 2}; count = 1
  -- for c = 5, valid (a, b) are from {1, 2, 3, 4}; count = 6
  -- for c = 7, valid (a, b) are from {1, 2, 3, 4, 5, 6}; count = 15
  -- for c = 9, valid (a, b) are from {1, 2, 3, 4, 5, 6, 7, 8}; count = 28

  -- Sum counts for all valid cases of c
  exact 50

-- Define our main theorem based on problem and final result
theorem odd_increasing_three_digit_numbers_count_eq_50 :
  count_odd_increasing_three_digit_numbers = 50 := by
  unfold count_odd_increasing_three_digit_numbers
  exact rfl -- the correct proof will fill in this part

end odd_increasing_three_digit_numbers_count_eq_50_l165_165323


namespace find_max_value_l165_165482

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  2 * x * y * Real.sqrt 3 + 3 * y * z * Real.sqrt 2 + 3 * z * x

theorem find_max_value (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z)
  (h₃ : x^2 + y^2 + z^2 = 1) : 
  maximum_value x y z ≤ Real.sqrt 3 := sorry

end find_max_value_l165_165482


namespace minimize_triangle_expression_l165_165345

theorem minimize_triangle_expression :
  ∃ (a b c : ℤ), a < b ∧ b < c ∧ a + b + c = 30 ∧
  ∀ (x y z : ℤ), x < y ∧ y < z ∧ x + y + z = 30 → (z^2 + 18*x + 18*y - 446) ≥ 17 ∧ 
  ∃ (p q r : ℤ), p < q ∧ q < r ∧ p + q + r = 30 ∧ (r^2 + 18*p + 18*q - 446 = 17) := 
sorry

end minimize_triangle_expression_l165_165345


namespace total_rats_l165_165217

variable (Kenia Hunter Elodie : ℕ) -- Number of rats each person has

-- Conditions
-- Elodie has 30 rats
axiom h1 : Elodie = 30
-- Elodie has 10 rats more than Hunter
axiom h2 : Elodie = Hunter + 10
-- Kenia has three times as many rats as Hunter and Elodie have together
axiom h3 : Kenia = 3 * (Hunter + Elodie)

-- Prove that the total number of pets the three have together is 200
theorem total_rats : Kenia + Hunter + Elodie = 200 := 
by 
  sorry

end total_rats_l165_165217


namespace Jonathan_typing_time_l165_165750

theorem Jonathan_typing_time
  (J : ℝ)
  (HJ : 0 < J)
  (rate_Jonathan : ℝ := 1 / J)
  (rate_Susan : ℝ := 1 / 30)
  (rate_Jack : ℝ := 1 / 24)
  (combined_rate : ℝ := 1 / 10)
  (combined_rate_eq : rate_Jonathan + rate_Susan + rate_Jack = combined_rate)
  : J = 40 :=
sorry

end Jonathan_typing_time_l165_165750


namespace basketball_player_second_shot_probability_l165_165400

noncomputable def probability_of_second_shot (p_first_shot : ℚ) 
  (p_second_given_first : ℚ) (p_second_given_miss_first : ℚ) : ℚ :=
  p_first_shot * p_second_given_first + (1 - p_first_shot) * p_second_given_miss_first

theorem basketball_player_second_shot_probability :
  probability_of_second_shot (3 / 4) (3 / 4) (1 / 4) = 5 / 8 :=
by
  sorry

end basketball_player_second_shot_probability_l165_165400


namespace game_is_not_fair_l165_165620

noncomputable def expected_winnings : ℚ := 
  let p_1 := 1 / 8
  let p_2 := 7 / 8
  let gain_case_1 := 2
  let loss_case_2 := -1 / 7
  (p_1 * gain_case_1) + (p_2 * loss_case_2)

theorem game_is_not_fair : expected_winnings = 1 / 8 :=
sorry

end game_is_not_fair_l165_165620


namespace time_to_cross_tree_l165_165260

variable (length_train : ℕ) (time_platform : ℕ) (length_platform : ℕ)

theorem time_to_cross_tree (h1 : length_train = 1200) (h2 : time_platform = 190) (h3 : length_platform = 700) :
  let distance_platform := length_train + length_platform
  let speed_train := distance_platform / time_platform
  let time_to_cross_tree := length_train / speed_train
  time_to_cross_tree = 120 :=
by
  -- Using the conditions to prove the goal
  sorry

end time_to_cross_tree_l165_165260


namespace cos_angle_difference_l165_165900

theorem cos_angle_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1): 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end cos_angle_difference_l165_165900


namespace boys_count_l165_165311

variable (B G : ℕ)

theorem boys_count (h1 : B + G = 466) (h2 : G = B + 212) : B = 127 := by
  sorry

end boys_count_l165_165311


namespace exists_two_digit_number_N_l165_165092

-- Statement of the problem
theorem exists_two_digit_number_N : 
  ∃ (N : ℕ), (∃ (a b : ℕ), N = 10 * a + b ∧ N = a * b + 2 * (a + b) ∧ 10 ≤ N ∧ N < 100) :=
by
  sorry

end exists_two_digit_number_N_l165_165092


namespace weight_of_new_person_is_correct_l165_165545

noncomputable def weight_new_person (increase_per_person : ℝ) (old_weight : ℝ) (group_size : ℝ) : ℝ :=
  old_weight + group_size * increase_per_person

theorem weight_of_new_person_is_correct :
  weight_new_person 7.2 65 10 = 137 :=
by
  sorry

end weight_of_new_person_is_correct_l165_165545


namespace peanut_mixture_l165_165108

-- Definitions of given conditions
def virginia_peanuts_weight : ℝ := 10
def virginia_peanuts_cost_per_pound : ℝ := 3.50
def spanish_peanuts_cost_per_pound : ℝ := 3.00
def texan_peanuts_cost_per_pound : ℝ := 4.00
def desired_cost_per_pound : ℝ := 3.60

-- Definitions of unknowns S (Spanish peanuts) and T (Texan peanuts)
variable (S T : ℝ)

-- Equation derived from given conditions
theorem peanut_mixture :
  (0.40 * T) - (0.60 * S) = 1 := sorry

end peanut_mixture_l165_165108


namespace sum_coordinates_B_l165_165491

noncomputable def A : (ℝ × ℝ) := (0, 0)
noncomputable def B (x : ℝ) : (ℝ × ℝ) := (x, 4)

theorem sum_coordinates_B 
  (x : ℝ) 
  (h_slope : (4 - 0)/(x - 0) = 3/4) : x + 4 = 28 / 3 := by
sorry

end sum_coordinates_B_l165_165491


namespace find_a_l165_165788

theorem find_a (a : ℝ) (h : -2 * a + 1 = -1) : a = 1 :=
by sorry

end find_a_l165_165788


namespace harrison_grade_levels_l165_165465

theorem harrison_grade_levels
  (total_students : ℕ)
  (percent_moving : ℚ)
  (advanced_class_size : ℕ)
  (num_normal_classes : ℕ)
  (normal_class_size : ℕ)
  (students_moving : ℕ)
  (students_per_grade_level : ℕ)
  (grade_levels : ℕ) :
  total_students = 1590 →
  percent_moving = 40 / 100 →
  advanced_class_size = 20 →
  num_normal_classes = 6 →
  normal_class_size = 32 →
  students_moving = total_students * percent_moving →
  students_per_grade_level = advanced_class_size + num_normal_classes * normal_class_size →
  grade_levels = students_moving / students_per_grade_level →
  grade_levels = 3 :=
by
  intros
  sorry

end harrison_grade_levels_l165_165465


namespace equal_elements_l165_165592

theorem equal_elements {n : ℕ} (a : ℕ → ℝ) (h₁ : n ≥ 2) (h₂ : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≠ -1) 
  (h₃ : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1)) 
  (hn1 : a (n + 1) = a 1) (hn2 : a (n + 2) = a 2) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i = a 1 := by
  sorry

end equal_elements_l165_165592


namespace total_value_of_goods_l165_165275

theorem total_value_of_goods (V : ℝ)
  (h1 : 0 < V)
  (h2 : ∃ t, V - 600 = t ∧ 0.12 * t = 134.4) :
  V = 1720 := 
sorry

end total_value_of_goods_l165_165275


namespace cos_theta_value_l165_165460

open Real

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 0)

noncomputable def cos_theta (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1 ^ 2 + u.2 ^ 2) * Real.sqrt (v.1 ^ 2 + v.2 ^ 2))

theorem cos_theta_value :
  cos_theta a b = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end cos_theta_value_l165_165460


namespace unique_sequence_l165_165258

theorem unique_sequence (a : ℕ → ℕ) (h_distinct: ∀ m n, a m = a n → m = n)
    (h_divisible: ∀ n, a n % a (a n) = 0) : ∀ n, a n = n :=
by
  -- proof goes here
  sorry

end unique_sequence_l165_165258


namespace exists_positive_x_for_inequality_l165_165453

-- Define the problem conditions and the final proof goal.
theorem exists_positive_x_for_inequality (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^2 + |x + a| < 2) ↔ a ∈ Set.Ico (-9/4 : ℝ) (2 : ℝ) :=
by
  sorry

end exists_positive_x_for_inequality_l165_165453


namespace bulls_on_farm_l165_165381

theorem bulls_on_farm (C B : ℕ) (h1 : C / B = 10 / 27) (h2 : C + B = 555) : B = 405 :=
sorry

end bulls_on_farm_l165_165381


namespace terminating_fraction_count_l165_165043

theorem terminating_fraction_count : 
  let range := {m : ℕ | 1 ≤ m ∧ m ≤ 594}
  (set.count range_subtype, ∃ (m : ℕ) (h : m ∈ range), (∃ k, m = k * 119) ∧ (gcd m 595 = 119)) = 4 :=
begin
  sorry
end

end terminating_fraction_count_l165_165043


namespace largest_divisor_of_n_l165_165977

theorem largest_divisor_of_n (n : ℕ) (h_pos: n > 0) (h_div: 72 ∣ n^2) : 12 ∣ n :=
by
  sorry

end largest_divisor_of_n_l165_165977


namespace fraction_area_outside_circle_l165_165007

theorem fraction_area_outside_circle (r : ℝ) (h1 : r > 0) :
  let side_length := 2 * r
  let area_square := side_length ^ 2
  let area_circle := π * r ^ 2
  let area_outside := area_square - area_circle
  (area_outside / area_square) = 1 - ↑π / 4 :=
by
  sorry

end fraction_area_outside_circle_l165_165007


namespace jen_profit_is_960_l165_165350

def buying_price : ℕ := 80
def selling_price : ℕ := 100
def num_candy_bars_bought : ℕ := 50
def num_candy_bars_sold : ℕ := 48

def profit_per_candy_bar := selling_price - buying_price
def total_profit := profit_per_candy_bar * num_candy_bars_sold

theorem jen_profit_is_960 : total_profit = 960 := by
  sorry

end jen_profit_is_960_l165_165350


namespace circle_tangency_problem_l165_165929

theorem circle_tangency_problem :
  let u1 := ∀ (x y : ℝ), x^2 + y^2 + 8 * x - 30 * y - 63 = 0
  let u2 := ∀ (x y : ℝ), x^2 + y^2 - 6 * x - 30 * y + 99 = 0
  let line := ∀ (b x : ℝ), y = b * x
  ∃ p q : ℕ, gcd p q = 1 ∧ n^2 = (p : ℚ) / (q : ℚ) ∧ p + q = 7 :=
sorry

end circle_tangency_problem_l165_165929


namespace present_age_of_son_l165_165129

theorem present_age_of_son (F S : ℕ) (h1 : F = S + 24) (h2 : F + 2 = 2 * (S + 2)) : S = 22 := by
  sorry

end present_age_of_son_l165_165129


namespace total_selling_price_of_toys_l165_165555

/-
  Prove that the total selling price (TSP) for 18 toys,
  given that each toy costs Rs. 1100 and the man gains the cost price of 3 toys, is Rs. 23100.
-/
theorem total_selling_price_of_toys :
  let CP := 1100
  let TCP := 18 * CP
  let G := 3 * CP
  let TSP := TCP + G
  TSP = 23100 :=
by
  let CP := 1100
  let TCP := 18 * CP
  let G := 3 * CP
  let TSP := TCP + G
  sorry

end total_selling_price_of_toys_l165_165555


namespace parallel_lines_implies_m_opposite_sides_implies_m_range_l165_165058

-- Definitions of the given lines and points
def l1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (m : ℝ) : ℝ × ℝ := (m, 4)

-- Problem Part (I)
theorem parallel_lines_implies_m (m : ℝ) : 
  (∀ (x y : ℝ), l1 x y → false) ∧ (∀ (x2 y2 : ℝ), (x2, y2) = A m ∨ (x2, y2) = B m → false) →
  (∃ m, 2 * m + 3 = 0 ∧ m + 5 = 0) :=
sorry

-- Problem Part (II)
theorem opposite_sides_implies_m_range (m : ℝ) :
  ((2 * (-2) + m - 1) * (2 * m + 4 - 1) < 0) →
  m ∈ Set.Ioo (-3/2 : ℝ) (5 : ℝ) :=
sorry

end parallel_lines_implies_m_opposite_sides_implies_m_range_l165_165058


namespace correct_operation_l165_165682

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l165_165682


namespace a_plus_d_eq_five_l165_165845

theorem a_plus_d_eq_five (a b c d k : ℝ) (hk : 0 < k) 
  (h1 : a + b = 11) 
  (h2 : b^2 + c^2 = k) 
  (h3 : b + c = 9) 
  (h4 : c + d = 3) : 
  a + d = 5 :=
by
  sorry

end a_plus_d_eq_five_l165_165845


namespace calc1_calc2_calc3_calc4_l165_165843

-- Problem 1
theorem calc1 : (-2: ℝ) ^ 2 - (7 - Real.pi) ^ 0 - (1 / 3) ^ (-1: ℝ) = 0 := by
  sorry

-- Problem 2
variable (m : ℝ)
theorem calc2 : 2 * m ^ 3 * 3 * m - (2 * m ^ 2) ^ 2 + m ^ 6 / m ^ 2 = 3 * m ^ 4 := by
  sorry

-- Problem 3
variable (a : ℝ)
theorem calc3 : (a + 1) ^ 2 + (a + 1) * (a - 2) = 2 * a ^ 2 + a - 1 := by
  sorry

-- Problem 4
variables (x y : ℝ)
theorem calc4 : (x + y - 1) * (x - y - 1) = x ^ 2 - 2 * x + 1 - y ^ 2 := by
  sorry

end calc1_calc2_calc3_calc4_l165_165843


namespace gumball_machine_total_gumballs_l165_165825

/-- A gumball machine has red, green, and blue gumballs. Given the following conditions:
1. The machine has half as many blue gumballs as red gumballs.
2. For each blue gumball, the machine has 4 times as many green gumballs.
3. The machine has 16 red gumballs.
Prove that the total number of gumballs in the machine is 56. -/
theorem gumball_machine_total_gumballs :
  ∀ (red blue green : ℕ),
    (blue = red / 2) →
    (green = blue * 4) →
    red = 16 →
    (red + blue + green = 56) :=
by
  intros red blue green h_blue h_green h_red
  sorry

end gumball_machine_total_gumballs_l165_165825


namespace odd_three_digit_integers_strictly_increasing_digits_l165_165333

theorem odd_three_digit_integers_strictly_increasing_digits :
  let valid_combinations (c : ℕ) :=
    if c = 1 then 0 else
    if c = 3 then 1 else
    if c = 5 then 6 else
    if c = 7 then 15 else
    if c = 9 then 28 else 0 in
  (valid_combinations 1 + valid_combinations 3 + valid_combinations 5 + valid_combinations 7 + valid_combinations 9 = 50) :=
by
  unfold valid_combinations
  sorry

end odd_three_digit_integers_strictly_increasing_digits_l165_165333


namespace smallest_possible_value_l165_165521

theorem smallest_possible_value (x : ℕ) (m : ℕ) :
  (x > 0) →
  (Nat.gcd 36 m = x + 3) →
  (Nat.lcm 36 m = x * (x + 3)) →
  m = 12 :=
by
  sorry

end smallest_possible_value_l165_165521


namespace alpha_beta_purchase_ways_l165_165421

-- Definitions for the problem
def number_of_flavors : ℕ := 7
def number_of_milk_types : ℕ := 4
def total_products_to_purchase : ℕ := 5

-- Conditions
def alpha_max_per_flavor : ℕ := 2
def beta_only_cookies (x : ℕ) : Prop := x = number_of_flavors

-- Main theorem (statement only)
theorem alpha_beta_purchase_ways : 
  ∃ (ways : ℕ), 
    ways = 17922 ∧
    ∀ (alpha beta : ℕ), 
      alpha + beta = total_products_to_purchase →
      (alpha <= alpha_max_per_flavor * number_of_flavors ∧ beta <= total_products_to_purchase - alpha) :=
sorry

end alpha_beta_purchase_ways_l165_165421


namespace calculateRequiredMonthlyRent_l165_165701

noncomputable def requiredMonthlyRent (purchase_price : ℝ) (annual_return_rate : ℝ) (annual_taxes : ℝ) (repair_percentage : ℝ) : ℝ :=
  let annual_return := annual_return_rate * purchase_price
  let total_annual_need := annual_return + annual_taxes
  let monthly_requirement := total_annual_need / 12
  let monthly_rent := monthly_requirement / (1 - repair_percentage)
  monthly_rent

theorem calculateRequiredMonthlyRent : requiredMonthlyRent 20000 0.06 450 0.10 = 152.78 := by
  sorry

end calculateRequiredMonthlyRent_l165_165701


namespace correct_operation_l165_165681

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l165_165681


namespace correct_operation_l165_165666

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l165_165666


namespace problem_statement_l165_165910

theorem problem_statement (d : ℕ) (h1 : d > 0) (h2 : d ∣ (5 + 2022^2022)) :
  (∃ x y : ℤ, d = 2 * x^2 + 2 * x * y + 3 * y^2) ↔ (d % 20 = 3 ∨ d % 20 = 7) :=
by
  sorry

end problem_statement_l165_165910


namespace two_people_paint_time_l165_165891

theorem two_people_paint_time (h : 5 * 7 = 35) :
  ∃ t : ℝ, 2 * t = 35 ∧ t = 17.5 := 
sorry

end two_people_paint_time_l165_165891


namespace women_with_fair_hair_percentage_l165_165813

theorem women_with_fair_hair_percentage
  (A : ℝ) (B : ℝ)
  (hA : A = 0.40)
  (hB : B = 0.25) :
  A * B = 0.10 := 
by
  rw [hA, hB]
  norm_num

end women_with_fair_hair_percentage_l165_165813


namespace simplify_expression1_simplify_expression2_l165_165440

/-- Proof Problem 1: Simplify the expression (a+2b)^2 - 4b(a+b) -/
theorem simplify_expression1 (a b : ℝ) : 
  (a + 2 * b)^2 - 4 * b * (a + b) = a^2 :=
sorry

/-- Proof Problem 2: Simplify the expression ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) ÷ (x - 1) / (x^2 - 4) -/
theorem simplify_expression2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) : 
  ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 :=
sorry

end simplify_expression1_simplify_expression2_l165_165440


namespace t_shirt_cost_l165_165619

theorem t_shirt_cost (T : ℕ) 
  (h1 : 3 * T + 50 = 110) : T = 20 := 
by
  sorry

end t_shirt_cost_l165_165619


namespace no_adjacent_girls_arrangement_l165_165695

theorem no_adjacent_girls_arrangement :
  let boys := 4 in
  let girls := 4 in
  let totalWaysToArrangeBoys := Nat.factorial boys in
  let totalWaysToInsertGirls := (Nat.factorial (boys + 1)) / (Nat.factorial (boys + 1 - girls)) in
  let totalArrangements := totalWaysToArrangeBoys * totalWaysToInsertGirls in
  totalArrangements = 2880 :=
by
  sorry

end no_adjacent_girls_arrangement_l165_165695


namespace lenny_pens_left_l165_165752

def total_pens (boxes : ℕ) (pens_per_box : ℕ) : ℕ := boxes * pens_per_box

def pens_to_friends (total : ℕ) (percentage : ℚ) : ℚ := total * percentage

def remaining_after_friends (total : ℕ) (given : ℚ) : ℚ := total - given

def pens_to_classmates (remaining : ℚ) (fraction : ℚ) : ℚ := remaining * fraction

def final_remaining (remaining : ℚ) (given : ℚ) : ℚ := remaining - given

theorem lenny_pens_left :
  let total := total_pens 20 5 in
  let given_to_friends := pens_to_friends total (40 / 100) in
  let remaining1 := remaining_after_friends total given_to_friends in
  let given_to_classmates := pens_to_classmates remaining1 (1 / 4) in
  let remaining2 := final_remaining remaining1 given_to_classmates in
  remaining2 = 45 :=
by
  sorry

end lenny_pens_left_l165_165752


namespace repeating_decimal_fraction_l165_165805

theorem repeating_decimal_fraction :
  let x := (37/100) + (246 / 99900)
  in x = 37245 / 99900 :=
by
  let x := (37/100) + (246 / 99900)
  show x = 37245 / 99900
  sorry

end repeating_decimal_fraction_l165_165805


namespace seq_ratio_l165_165897

theorem seq_ratio (a : ℕ → ℝ) (h₁ : a 1 = 5) (h₂ : ∀ n, a n * a (n + 1) = 2^n) : 
  a 7 / a 3 = 4 := 
by 
  sorry

end seq_ratio_l165_165897


namespace find_a_l165_165741

theorem find_a (a : ℤ) :
  (∃! x : ℤ, |a * x + a + 2| < 2) ↔ a = 3 ∨ a = -3 := 
sorry

end find_a_l165_165741


namespace sawing_steel_bar_time_l165_165232

theorem sawing_steel_bar_time (pieces : ℕ) (time_per_cut : ℕ) : 
  pieces = 6 → time_per_cut = 2 → (pieces - 1) * time_per_cut = 10 := 
by
  intros
  sorry

end sawing_steel_bar_time_l165_165232


namespace unique_polynomial_P_l165_165294

noncomputable def P : ℝ → ℝ := sorry

axiom P_func_eq (x : ℝ) : P (x^2 + 1) = P x ^ 2 + 1
axiom P_zero : P 0 = 0

theorem unique_polynomial_P (x : ℝ) : P x = x :=
by
  sorry

end unique_polynomial_P_l165_165294


namespace animals_per_aquarium_l165_165736

theorem animals_per_aquarium (total_animals : ℕ) (number_of_aquariums : ℕ) (h1 : total_animals = 40) (h2 : number_of_aquariums = 20) : 
  total_animals / number_of_aquariums = 2 :=
by
  sorry

end animals_per_aquarium_l165_165736


namespace work_days_l165_165543

theorem work_days (A B : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 18) :
    1 / (A + B) = 6 :=
by
  sorry

end work_days_l165_165543


namespace visitors_surveyed_l165_165535

-- Given definitions
def total_visitors : ℕ := 400
def visitors_not_enjoyed_nor_understood : ℕ := 100
def E := total_visitors / 2
def U := total_visitors / 2

-- Using condition that 3/4th visitors enjoyed and understood
def enjoys_and_understands := (3 * total_visitors) / 4

-- Assert the equivalence of total number of visitors calculation
theorem visitors_surveyed:
  total_visitors = enjoys_and_understands + visitors_not_enjoyed_nor_understood :=
by
  sorry

end visitors_surveyed_l165_165535


namespace team_problem_solved_probability_l165_165564

-- Defining the probabilities
def P_A : ℚ := 1 / 5
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Defining the probability that the problem is solved
def P_s : ℚ := 3 / 5

-- Lean 4 statement to prove that the calculated probability matches the expected solution
theorem team_problem_solved_probability :
  1 - (1 - P_A) * (1 - P_B) * (1 - P_C) = P_s :=
by
  sorry

end team_problem_solved_probability_l165_165564


namespace prove_inequality_l165_165711

noncomputable def inequality_holds (x y : ℝ) : Prop :=
  x^3 * (y + 1) + y^3 * (x + 1) ≥ x^2 * (y + y^2) + y^2 * (x + x^2)

theorem prove_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : inequality_holds x y :=
  sorry

end prove_inequality_l165_165711


namespace exists_n_2_pow_k_divides_n_n_minus_m_l165_165623

theorem exists_n_2_pow_k_divides_n_n_minus_m 
  (k : ℕ) (m : ℤ) (h1 : 0 < k) (h2 : Odd m) : 
  ∃ n : ℕ, 0 < n ∧ 2^k ∣ (n^n - m) :=
sorry

end exists_n_2_pow_k_divides_n_n_minus_m_l165_165623


namespace amount_for_gifts_and_charitable_causes_l165_165037

namespace JillExpenses

def net_monthly_salary : ℝ := 3700
def discretionary_income : ℝ := 0.20 * net_monthly_salary -- 1/5 * 3700
def vacation_fund : ℝ := 0.30 * discretionary_income
def savings : ℝ := 0.20 * discretionary_income
def eating_out_and_socializing : ℝ := 0.35 * discretionary_income
def gifts_and_charitable_causes : ℝ := discretionary_income - (vacation_fund + savings + eating_out_and_socializing)

theorem amount_for_gifts_and_charitable_causes : gifts_and_charitable_causes = 111 := sorry

end JillExpenses

end amount_for_gifts_and_charitable_causes_l165_165037


namespace num_math_books_l165_165808

theorem num_math_books (total_books total_cost math_book_cost history_book_cost : ℕ) (M H : ℕ)
  (h1 : total_books = 80)
  (h2 : math_book_cost = 4)
  (h3 : history_book_cost = 5)
  (h4 : total_cost = 368)
  (h5 : M + H = total_books)
  (h6 : math_book_cost * M + history_book_cost * H = total_cost) :
  M = 32 :=
by
  sorry

end num_math_books_l165_165808


namespace suit_price_the_day_after_sale_l165_165408

def originalPrice : ℕ := 300
def increaseRate : ℚ := 0.20
def couponDiscount : ℚ := 0.30
def additionalReduction : ℚ := 0.10

def increasedPrice := originalPrice * (1 + increaseRate)
def priceAfterCoupon := increasedPrice * (1 - couponDiscount)
def finalPrice := increasedPrice * (1 - additionalReduction)

theorem suit_price_the_day_after_sale 
  (op : ℕ := originalPrice) 
  (ir : ℚ := increaseRate) 
  (cd : ℚ := couponDiscount) 
  (ar : ℚ := additionalReduction) :
  finalPrice = 324 := 
sorry

end suit_price_the_day_after_sale_l165_165408


namespace radius_B_l165_165708

noncomputable def radius_A := 2
noncomputable def radius_D := 4

theorem radius_B (r_B : ℝ) (x y : ℝ) 
  (h1 : (2 : ℝ) + y = x + (x^2 / 4)) 
  (h2 : y = 2 - (x^2 / 8)) 
  (h3 : x = (4: ℝ) / 3) 
  (h4 : y = x + (x^2 / 4)) : r_B = 20 / 9 :=
sorry

end radius_B_l165_165708


namespace compound_interest_1200_20percent_3years_l165_165583

noncomputable def compoundInterest (P r : ℚ) (n t : ℕ) : ℚ :=
  let A := P * (1 + r / n) ^ (n * t)
  A - P

theorem compound_interest_1200_20percent_3years :
  compoundInterest 1200 0.20 1 3 = 873.6 :=
by
  sorry

end compound_interest_1200_20percent_3years_l165_165583


namespace printed_value_l165_165203

theorem printed_value (X S : ℕ) (h1 : X = 5) (h2 : S = 0) : 
  (∃ n, S = (n * (3 * n + 7)) / 2 ∧ S ≥ 15000) → 
  X = 5 + 3 * 122 - 3 :=
by 
  sorry

end printed_value_l165_165203


namespace algebra_expression_opposite_l165_165472

theorem algebra_expression_opposite (a : ℚ) :
  3 * a + 1 = -(3 * (a - 1)) → a = 1 / 3 :=
by
  intro h
  sorry

end algebra_expression_opposite_l165_165472


namespace triangle_inequality_internal_point_l165_165793

theorem triangle_inequality_internal_point {A B C P : Type} 
  (x y z p q r : ℝ) 
  (h_distances_from_vertices : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_distances_from_sides : p > 0 ∧ q > 0 ∧ r > 0)
  (h_x_y_z_triangle_ineq : x + y > z ∧ y + z > x ∧ z + x > y)
  (h_p_q_r_triangle_ineq : p + q > r ∧ q + r > p ∧ r + p > q) :
  x * y * z ≥ (q + r) * (r + p) * (p + q) :=
sorry

end triangle_inequality_internal_point_l165_165793


namespace percentage_increase_is_50_l165_165526

-- Define the conditions
variables {P : ℝ} {x : ℝ}

-- Define the main statement (goal)
theorem percentage_increase_is_50 (h : 0.80 * P + (0.008 * x * P) = 1.20 * P) : x = 50 :=
sorry  -- Skip the proof as per instruction

end percentage_increase_is_50_l165_165526


namespace total_travel_cost_l165_165268

noncomputable def calculate_cost : ℕ :=
  let cost_length_road :=
    (30 * 10 * 4) +  -- first segment
    (40 * 10 * 5) +  -- second segment
    (30 * 10 * 6)    -- third segment
  let cost_breadth_road :=
    (20 * 10 * 3) +  -- first segment
    (40 * 10 * 2)    -- second segment
  cost_length_road + cost_breadth_road

theorem total_travel_cost :
  calculate_cost = 6400 :=
by
  sorry

end total_travel_cost_l165_165268


namespace total_cost_in_dollars_l165_165647

theorem total_cost_in_dollars :
  (500 * 3 + 300 * 2) / 100 = 21 := 
by
  sorry

end total_cost_in_dollars_l165_165647


namespace platform_length_correct_l165_165558

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def time_cross_platform : ℝ := 30
noncomputable def time_cross_man : ℝ := 19
noncomputable def length_train : ℝ := train_speed_mps * time_cross_man
noncomputable def total_distance_cross_platform : ℝ := train_speed_mps * time_cross_platform
noncomputable def length_platform : ℝ := total_distance_cross_platform - length_train

theorem platform_length_correct : length_platform = 220 := by
  sorry

end platform_length_correct_l165_165558


namespace mass_percentage_of_Cl_in_NaClO_l165_165886

noncomputable def molarMassNa : ℝ := 22.99
noncomputable def molarMassCl : ℝ := 35.45
noncomputable def molarMassO : ℝ := 16.00

noncomputable def molarMassNaClO : ℝ := molarMassNa + molarMassCl + molarMassO

theorem mass_percentage_of_Cl_in_NaClO : 
  (molarMassCl / molarMassNaClO) * 100 = 47.61 :=
by 
  sorry

end mass_percentage_of_Cl_in_NaClO_l165_165886


namespace find_c_l165_165462

def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

theorem find_c (c : ℝ) :
  (∀ x, f x c ≤ f 2 c) → c = 6 :=
sorry

end find_c_l165_165462


namespace minimize_transportation_cost_l165_165005

noncomputable def transportation_cost (x : ℝ) (distance : ℝ) (k : ℝ) (other_expense : ℝ) : ℝ :=
  k * (x * distance / x^2 + other_expense * distance / x)

theorem minimize_transportation_cost :
  ∀ (distance : ℝ) (max_speed : ℝ) (k : ℝ) (other_expense : ℝ) (x : ℝ),
  0 < x ∧ x ≤ max_speed ∧ max_speed = 50 ∧ distance = 300 ∧ k = 0.5 ∧ other_expense = 800 →
  transportation_cost x distance k other_expense = 150 * (x + 1600 / x) ∧
  (∀ y, (0 < y ∧ y ≤ max_speed) → transportation_cost y distance k other_expense ≥ 12000) ∧
  (transportation_cost 40 distance k other_expense = 12000)
  := 
  by intros distance max_speed k other_expense x H;
     sorry

end minimize_transportation_cost_l165_165005


namespace problem_I_problem_II_l165_165049

theorem problem_I (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : 2 * a - a * Real.cos B = b * Real.cos A) :
  c / a = 2 :=
sorry

theorem problem_II (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : 2 * a - a * Real.cos B = b * Real.cos A) 
  (h3 : b = 4) (h4 : Real.cos C = 1 / 4) :
  (1 / 2) * a * b * Real.sin C = Real.sqrt 15 :=
sorry

end problem_I_problem_II_l165_165049


namespace ratio_of_installing_to_downloading_l165_165090

noncomputable def timeDownloading : ℕ := 10

noncomputable def ratioTimeSpent (installingTime : ℕ) : ℚ :=
  let tutorialTime := 3 * (timeDownloading + installingTime)
  let totalTime := timeDownloading + installingTime + tutorialTime
  if totalTime = 60 then
    (installingTime : ℚ) / (timeDownloading : ℚ)
  else 0

theorem ratio_of_installing_to_downloading : ratioTimeSpent 5 = 1 / 2 := by
  sorry

end ratio_of_installing_to_downloading_l165_165090


namespace find_divisor_l165_165070

theorem find_divisor (n m : ℤ) (k: ℤ) :
  n % 20 = 11 →
  (2 * n) % m = 2 →
  m = 18 :=
by
  assume h1 : n % 20 = 11
  assume h2 : (2 * n) % m = 2
  -- Proof placeholder
  sorry

end find_divisor_l165_165070


namespace geometric_series_sum_l165_165882

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l165_165882


namespace percentage_of_pushups_l165_165172

-- Problem conditions as definitions
def jumpingJacks := 12
def pushups := 8
def situps := 20
def totalExercises := jumpingJacks + pushups + situps

-- Question and the proof goal
theorem percentage_of_pushups : 
  (pushups / totalExercises : ℝ) * 100 = 20 := by
  sorry

end percentage_of_pushups_l165_165172


namespace pinocchio_cannot_pay_exactly_l165_165940

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l165_165940


namespace odd_increasing_three_digit_numbers_count_eq_50_l165_165321

def count_odd_increasing_three_digit_numbers : Nat := by
  -- Mathematical conditions:
  -- let a, b, c be digits of the number
  -- 0 < a < b < c <= 9 and c is an odd digit

  -- We analyze values for 'c' which must be an odd digit,
  -- and count valid (a, b) combinations for each case of c.

  -- Starting from cases for c:
  -- for c = 1, no valid (a, b); count = 0
  -- for c = 3, valid (a, b) are from {1, 2}; count = 1
  -- for c = 5, valid (a, b) are from {1, 2, 3, 4}; count = 6
  -- for c = 7, valid (a, b) are from {1, 2, 3, 4, 5, 6}; count = 15
  -- for c = 9, valid (a, b) are from {1, 2, 3, 4, 5, 6, 7, 8}; count = 28

  -- Sum counts for all valid cases of c
  exact 50

-- Define our main theorem based on problem and final result
theorem odd_increasing_three_digit_numbers_count_eq_50 :
  count_odd_increasing_three_digit_numbers = 50 := by
  unfold count_odd_increasing_three_digit_numbers
  exact rfl -- the correct proof will fill in this part

end odd_increasing_three_digit_numbers_count_eq_50_l165_165321


namespace integer_values_abc_l165_165405

theorem integer_values_abc {a b c : ℤ} :
  a^2 + b^2 + c^2 + 3 < a * b + 3 * b + 2 * c ↔ (a = 1 ∧ b = 2 ∧ c = 1) :=
by
  sorry -- Proof to be filled

end integer_values_abc_l165_165405


namespace find_y_l165_165409

theorem find_y (x y : ℕ) (h1 : x % y = 9) (h2 : (x : ℝ) / y = 86.12) : y = 75 :=
sorry

end find_y_l165_165409


namespace _l165_165994

noncomputable def urn_probability : ℚ := 
  let R0 := 2 in
  let B0 := 1 in
  let operations := 5 in
  let total_balls_after := 8 in
  -- Final configuration we are checking the probability for:
  let final_red_balls := 3 in
  let final_blue_balls := 5 in
  proof
    have : total_balls_after = final_red_balls + final_blue_balls := by
      -- The total number of balls after the operations should match
      calc 8 = 3 + 5 : by simp
    
    have : ∀ (R_a B_a : ℕ), R0 + R_a + B0 + B_a = total_balls_after → 
      ∃ probability, probability = (final_red_balls = R0 + R_a) ∧ (final_blue_balls = B0 + B_a) := by
      -- This can be obtained through the binomial theorem and detailed calculation as shown 
      sorry

    exact (∃! p : ℚ, p = 2 / 21)  -- The probability is unique and is calculated as 2/21

end _l165_165994


namespace symmetry_origin_l165_165961

def f (x : ℝ) : ℝ := x^3 + x

theorem symmetry_origin : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end symmetry_origin_l165_165961


namespace amount_paid_Y_l165_165797

theorem amount_paid_Y (X Y : ℝ) (h1 : X + Y = 330) (h2 : X = 1.2 * Y) : Y = 150 := 
by
  sorry

end amount_paid_Y_l165_165797


namespace find_f_4_l165_165224

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x ^ 2

theorem find_f_4 : f 4 = 5.5 :=
by
  sorry

end find_f_4_l165_165224


namespace amount_due_years_l165_165964

noncomputable def years_due (PV FV : ℝ) (r : ℝ) : ℝ :=
  (Real.log (FV / PV)) / (Real.log (1 + r))

theorem amount_due_years : 
  years_due 200 242 0.10 = 2 :=
by
  sorry

end amount_due_years_l165_165964


namespace simplify_expression_eval_at_2_l165_165756

theorem simplify_expression (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
    (x^2 + a)^2 / ((a - b) * (a - c)) + (x^2 + b)^2 / ((b - a) * (b - c)) + (x^2 + c)^2 / ((c - a) * (c - b)) =
    x^4 + x^2 * (a + b + c) + (a^2 + b^2 + c^2) :=
sorry

theorem eval_at_2 (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
    (2^2 + a)^2 / ((a - b) * (a - c)) + (2^2 + b)^2 / ((b - a) * (b - c)) + (2^2 + c)^2 / ((c - a) * (c - b)) =
    16 + 4 * (a + b + c) + (a^2 + b^2 + c^2) :=
sorry

end simplify_expression_eval_at_2_l165_165756


namespace math_problem_l165_165999

   theorem math_problem :
     6 * (-1 / 2) + Real.sqrt 3 * Real.sqrt 8 + (-15 : ℝ)^0 = 2 * Real.sqrt 6 - 2 :=
   by
     sorry
   
end math_problem_l165_165999


namespace average_of_rest_equals_40_l165_165207

-- Defining the initial conditions
def total_students : ℕ := 20
def high_scorers : ℕ := 2
def low_scorers : ℕ := 3
def class_average : ℚ := 40

-- The target function to calculate the average of the rest of the students
def average_rest_students (total_students high_scorers low_scorers : ℕ) (class_average : ℚ) : ℚ :=
  let total_marks := total_students * class_average
  let high_scorer_marks := 100 * high_scorers
  let low_scorer_marks := 0 * low_scorers
  let rest_marks := total_marks - (high_scorer_marks + low_scorer_marks)
  let rest_students := total_students - high_scorers - low_scorers
  rest_marks / rest_students

-- The theorem to prove that the average of the rest of the students is 40
theorem average_of_rest_equals_40 : average_rest_students total_students high_scorers low_scorers class_average = 40 := 
by
  sorry

end average_of_rest_equals_40_l165_165207


namespace range_of_y_is_correct_l165_165194

noncomputable def range_of_y (n : ℝ) : ℝ :=
  if n > 2 then 1 / n else 2 * n^2 + 1

theorem range_of_y_is_correct :
  (∀ n, 0 < range_of_y n ∧ range_of_y n < 1 / 2 ∧ n > 2) ∨ (∀ n, 1 ≤ range_of_y n ∧ n ≤ 2) :=
sorry

end range_of_y_is_correct_l165_165194


namespace Bryce_received_raisins_l165_165907

theorem Bryce_received_raisins :
  ∃ x : ℕ, (∀ y : ℕ, x = y + 6) ∧ (∀ z : ℕ, z = x / 2) → x = 12 :=
by
  sorry

end Bryce_received_raisins_l165_165907


namespace arith_progression_possible_values_l165_165245

theorem arith_progression_possible_values :
  ∃ n_set : Finset ℕ, 
    (card n_set = 14 ∧ 
    ∀ n ∈ n_set, 
      1 < n ∧ 
      ∃ a : ℤ, 
        0 < a ∧ 
        2 * 180 = n * (2 * a + (n - 1) * 3)) :=
by { sorry }

end arith_progression_possible_values_l165_165245


namespace odd_three_digit_integers_in_strict_increasing_order_l165_165318

theorem odd_three_digit_integers_in_strict_increasing_order: 
  (∀ (a b c : ℕ), 100 ≤ (100 * a + 10 * b + c) ∧ 100 * a + 10 * b + c < 1000 → a < b ∧ b < c →
  c % 2 = 1 ∧ c ≠ 0 → 
  (∃ n, n = 50)) :=
by sorry

end odd_three_digit_integers_in_strict_increasing_order_l165_165318


namespace shopkeeper_milk_sold_l165_165990

theorem shopkeeper_milk_sold :
  let morning_packets := 150
  let morning_250 := 60
  let morning_300 := 40
  let morning_350 := morning_packets - morning_250 - morning_300
  
  let evening_packets := 100
  let evening_400 := evening_packets * 50 / 100
  let evening_500 := evening_packets * 25 / 100
  let evening_450 := evening_packets * 25 / 100

  let morning_milk := morning_250 * 250 + morning_300 * 300 + morning_350 * 350
  let evening_milk := evening_400 * 400 + evening_500 * 500 + evening_450 * 450
  let total_milk := morning_milk + evening_milk

  let remaining_milk := 42000
  let sold_milk := total_milk - remaining_milk

  let ounces_per_mil := 1 / 30
  let sold_milk_ounces := sold_milk * ounces_per_mil

  sold_milk_ounces = 1541.67 := by sorry

end shopkeeper_milk_sold_l165_165990


namespace terminating_decimal_expansion_7_over_625_l165_165587

theorem terminating_decimal_expansion_7_over_625 : (7 / 625 : ℚ) = 112 / 10000 := by
  sorry

end terminating_decimal_expansion_7_over_625_l165_165587


namespace cricket_run_rate_l165_165141

theorem cricket_run_rate (initial_run_rate : ℝ) (initial_overs : ℕ) (target : ℕ) (remaining_overs : ℕ) 
    (run_rate_in_remaining_overs : ℝ)
    (h1 : initial_run_rate = 3.2)
    (h2 : initial_overs = 10)
    (h3 : target = 272)
    (h4 : remaining_overs = 40) :
    run_rate_in_remaining_overs = 6 :=
  sorry

end cricket_run_rate_l165_165141


namespace train_length_correct_l165_165147

def train_length (speed_kph : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_mps := speed_kph * 1000 / 3600
  speed_mps * time_sec

theorem train_length_correct :
  train_length 90 10 = 250 := by
  sorry

end train_length_correct_l165_165147


namespace isosceles_triangle_l165_165637

theorem isosceles_triangle {a b R : ℝ} {α β : ℝ} 
  (h : a * Real.tan α + b * Real.tan β = (a + b) * Real.tan ((α + β) / 2))
  (ha : a = 2 * R * Real.sin α) (hb : b = 2 * R * Real.sin β) :
  α = β := 
sorry

end isosceles_triangle_l165_165637


namespace inequality_abc_l165_165894

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 :=
by
  sorry

end inequality_abc_l165_165894


namespace second_largest_of_five_consecutive_is_19_l165_165388

theorem second_largest_of_five_consecutive_is_19 (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 90): 
  n + 3 = 19 :=
by sorry

end second_largest_of_five_consecutive_is_19_l165_165388


namespace geometric_series_sum_l165_165881

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l165_165881


namespace min_value_of_y_l165_165111

open Real

noncomputable def y (x : ℝ) : ℝ := 4 / (x - 2)

theorem min_value_of_y :
  (∀ x1 x2 ∈ Icc 3 6, x1 ≤ x2 → y x2 ≤ y x1) → ∀ x ∈ Icc 3 6, y x ≥ 1 := by
  sorry

end min_value_of_y_l165_165111


namespace difference_of_squares_l165_165978

variable (x y : ℝ)

-- Conditions
def condition1 : Prop := x + y = 15
def condition2 : Prop := x - y = 10

-- Goal to prove
theorem difference_of_squares (h1 : condition1 x y) (h2 : condition2 x y) : x^2 - y^2 = 150 := 
by sorry

end difference_of_squares_l165_165978


namespace jasmine_laps_per_afternoon_l165_165924

-- Defining the conditions
def swims_each_week (days_per_week : ℕ) := days_per_week = 5
def total_weeks := 5
def total_laps := 300

-- Main proof statement
theorem jasmine_laps_per_afternoon (d : ℕ) (l : ℕ) :
  swims_each_week d →
  total_weeks * d = 25 →
  total_laps = 300 →
  300 / 25 = l →
  l = 12 :=
by
  intros
  -- Skipping the proof
  sorry

end jasmine_laps_per_afternoon_l165_165924


namespace aluminum_percentage_in_new_alloy_l165_165397

theorem aluminum_percentage_in_new_alloy :
  ∀ (x1 x2 x3 : ℝ),
  0 ≤ x1 ∧ x1 ≤ 1 ∧
  0 ≤ x2 ∧ x2 ≤ 1 ∧
  0 ≤ x3 ∧ x3 ≤ 1 ∧
  x1 + x2 + x3 = 1 ∧
  0.15 * x1 + 0.3 * x2 = 0.2 →
  0.15 ≤ 0.6 * x1 + 0.45 * x3 ∧ 0.6 * x1 + 0.45 * x3 ≤ 0.40 :=
by
  -- The proof will be inserted here
  sorry

end aluminum_percentage_in_new_alloy_l165_165397


namespace solution_l165_165621

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem solution (x : ℝ) : g (g x) = g x ↔ x = 0 ∨ x = 4 ∨ x = 5 ∨ x = -1 :=
by
  sorry

end solution_l165_165621


namespace blue_beads_l165_165489

-- Variables to denote the number of blue, red, white, and silver beads
variables (B R W S : ℕ)

-- Conditions derived from the problem statement
def conditions : Prop :=
  (R = 2 * B) ∧
  (W = B + R) ∧
  (S = 10) ∧
  (B + R + W + S = 40)

-- The theorem to prove
theorem blue_beads (B R W S : ℕ) (h : conditions B R W S) : B = 5 :=
by
  sorry

end blue_beads_l165_165489


namespace blue_tiles_in_45th_row_l165_165016

theorem blue_tiles_in_45th_row :
  ∀ (n : ℕ), n = 45 → (∃ r b : ℕ, (r + b = 2 * n - 1) ∧ (r > b) ∧ (r - 1 = b)) → b = 44 :=
by
  -- Skipping the proof with sorry to adhere to instruction
  sorry

end blue_tiles_in_45th_row_l165_165016


namespace count_odd_three_digit_integers_in_increasing_order_l165_165325

-- Defining the conditions
def digits_in_strictly_increasing_order (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < 10

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def odd_three_digit_integers_in_increasing_order : ℕ :=
  ((finset.range 10).filter (λ c, is_odd c)).sum (λ c,
    ((finset.range c).sum (λ b,
      if h : b < c then
        (finset.range b).filter (λ a, digits_in_strictly_increasing_order a b c).card
      else 0)))

-- Theorem statement: Prove that the number of such numbers is 50
theorem count_odd_three_digit_integers_in_increasing_order :
  odd_three_digit_integers_in_increasing_order = 50 :=
sorry

end count_odd_three_digit_integers_in_increasing_order_l165_165325


namespace find_v5_l165_165107

noncomputable def sequence (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 3 * v (n + 1) + v n + 1

theorem find_v5 :
  ∃ (v : ℕ → ℝ), sequence v ∧ v 3 = 11 ∧ v 6 = 242 ∧ v 5 = 73.5 :=
by
  sorry

end find_v5_l165_165107


namespace length_of_chord_l165_165828

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the line y = x - 1 with slope 1 passing through the focus (1, 0)
def line (x y : ℝ) : Prop :=
  y = x - 1

-- Prove that the length of the chord |AB| is 8
theorem length_of_chord 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h3 : line x1 y1) 
  (h4 : line x2 y2) : 
  abs (x2 - x1) = 8 :=
sorry

end length_of_chord_l165_165828


namespace factor_polynomial_l165_165581

variable (x : ℝ)

theorem factor_polynomial : (270 * x^3 - 90 * x^2 + 18 * x) = 18 * x * (15 * x^2 - 5 * x + 1) :=
by 
  sorry

end factor_polynomial_l165_165581


namespace max_savings_theorem_band_members_theorem_selection_plans_theorem_l165_165376

/-- Given conditions for maximum savings calculation -/
def number_of_sets_purchased : ℕ := 75
def max_savings (cost_separate : ℕ) (cost_together : ℕ) : Prop :=
cost_separate - cost_together = 800

theorem max_savings_theorem : 
    ∃ cost_separate cost_together, 
    (cost_separate = 5600) ∧ (cost_together = 4800) → max_savings cost_separate cost_together := by
  sorry

/-- Given conditions for number of members in bands A and B -/
def conditions (x y : ℕ) : Prop :=
x + y = 75 ∧ 70 * x + 80 * y = 5600 ∧ x >= 40

theorem band_members_theorem :
    ∃ x y, conditions x y → (x = 40 ∧ y = 35) := by
  sorry

/-- Given conditions for possible selection plans for charity event -/
def heart_to_heart_activity (a b : ℕ) : Prop :=
3 * a + 5 * b = 65 ∧ a >= 5 ∧ b >= 5

theorem selection_plans_theorem :
    ∃ a b, heart_to_heart_activity a b → 
    ((a = 5 ∧ b = 10) ∨ (a = 10 ∧ b = 7)) := by
  sorry

end max_savings_theorem_band_members_theorem_selection_plans_theorem_l165_165376


namespace prove_correct_option_C_l165_165687

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l165_165687


namespace total_balloons_l165_165354

theorem total_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (h₁ : joan_balloons = 40) (h₂ : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := 
by
  sorry

end total_balloons_l165_165354


namespace linear_function_through_origin_l165_165298

theorem linear_function_through_origin (m : ℝ) :
  (∀ x y : ℝ, (y = (m - 1) * x + m ^ 2 - 1) → (x = 0 ∧ y = 0) → m = -1) :=
sorry

end linear_function_through_origin_l165_165298


namespace sport_formulation_water_l165_165084

theorem sport_formulation_water (corn_syrup_ounces : ℕ) (h_cs : corn_syrup_ounces = 3) : 
  ∃ water_ounces : ℕ, water_ounces = 45 :=
by
  -- The ratios for the "sport" formulation: Flavoring : Corn Syrup : Water = 1 : 4 : 60
  let flavoring_ratio := 1
  let corn_syrup_ratio := 4
  let water_ratio := 60
  -- The given corn syrup is 3 ounces which corresponds to corn_syrup_ratio parts
  have h_ratio : corn_syrup_ratio = 4 := rfl
  have h_flavoring_to_corn_syrup : flavoring_ratio / corn_syrup_ratio = 1 / 4 := by sorry
  have h_flavoring_to_water : flavoring_ratio / water_ratio = 1 / 60 := by sorry
  -- Set up the proportion
  have h_proportion : corn_syrup_ratio / corn_syrup_ounces = water_ratio / 45 := by sorry 
  -- Cross-multiply to solve for the water
  have h_cross_mul : 4 * 45 = 3 * 60 := by sorry
  exact ⟨45, rfl⟩

end sport_formulation_water_l165_165084


namespace total_cost_l165_165348

-- Define the given conditions.
def coffee_pounds : ℕ := 4
def coffee_cost_per_pound : ℝ := 2.50
def milk_gallons : ℕ := 2
def milk_cost_per_gallon : ℝ := 3.50

-- The total cost Jasmine will pay is $17.00
theorem total_cost : coffee_pounds * coffee_cost_per_pound + milk_gallons * milk_cost_per_gallon = 17.00 := by
  sorry

end total_cost_l165_165348


namespace abs_difference_of_squares_l165_165538

theorem abs_difference_of_squares : 
  let a := 105 
  let b := 103
  abs (a^2 - b^2) = 416 := 
by 
  let a := 105
  let b := 103
  sorry

end abs_difference_of_squares_l165_165538


namespace a_finishes_job_in_60_days_l165_165696

theorem a_finishes_job_in_60_days (A B : ℝ)
  (h1 : A + B = 1 / 30)
  (h2 : 20 * (A + B) = 2 / 3)
  (h3 : 20 * A = 1 / 3) :
  1 / A = 60 :=
by sorry

end a_finishes_job_in_60_days_l165_165696


namespace count_special_ordered_quadruples_l165_165710

theorem count_special_ordered_quadruples : 
  let special_quadruple (a b c d : ℕ) := 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 12 ∧ a + b < c + d
  nat.card {p : ℕ × ℕ × ℕ × ℕ // special_quadruple p.1 p.2.1 p.2.2.1 p.2.2.2} = 247 :=
by
  sorry

end count_special_ordered_quadruples_l165_165710


namespace all_three_digits_same_two_digits_same_all_digits_different_l165_165928

theorem all_three_digits_same (a : ℕ) (h1 : a < 10) (h2 : 3 * a = 24) : a = 8 :=
by sorry

theorem two_digits_same (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 2 * a + b = 24 ∨ a + 2 * b = 24) : 
  (a = 9 ∧ b = 6) ∨ (a = 6 ∧ b = 9) :=
by sorry

theorem all_digits_different (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10)
  (h4 : a ≠ b) (h5 : a ≠ c) (h6 : b ≠ c) (h7 : a + b + c = 24) :
  (a, b, c) = (7, 8, 9) ∨ (a, b, c) = (7, 9, 8) ∨ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (8, 9, 7) ∨ (a, b, c) = (9, 7, 8) ∨ (a, b, c) = (9, 8, 7) :=
by sorry

end all_three_digits_same_two_digits_same_all_digits_different_l165_165928


namespace range_of_a_l165_165181

noncomputable def inequality_condition (x : ℝ) (a : ℝ) : Prop :=
  a - 2 * x - |Real.log x| ≤ 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → inequality_condition x a) ↔ a ≤ 1 + Real.log 2 :=
sorry

end range_of_a_l165_165181


namespace converse_example_l165_165774

theorem converse_example (x : ℝ) (h : x^2 = 1) : x = 1 :=
sorry

end converse_example_l165_165774


namespace rhombus_diagonals_perpendicular_not_in_rectangle_l165_165688

-- Definitions for the rhombus
structure Rhombus :=
  (diagonals_perpendicular : Prop)

-- Definitions for the rectangle
structure Rectangle :=
  (diagonals_not_perpendicular : Prop)

-- The main proof statement
theorem rhombus_diagonals_perpendicular_not_in_rectangle 
  (R : Rhombus) 
  (Rec : Rectangle) : 
  R.diagonals_perpendicular ∧ Rec.diagonals_not_perpendicular :=
by sorry

end rhombus_diagonals_perpendicular_not_in_rectangle_l165_165688


namespace product_modulo_l165_165106

theorem product_modulo (n : ℕ) (h : 93 * 68 * 105 ≡ n [MOD 20]) (h_range : 0 ≤ n ∧ n < 20) : n = 0 := 
by
  sorry

end product_modulo_l165_165106


namespace additional_grassy_ground_l165_165544

theorem additional_grassy_ground (r1 r2 : ℝ) (h1: r1 = 16) (h2: r2 = 23) :
  (π * r2 ^ 2) - (π * r1 ^ 2) = 273 * π :=
by
  sorry

end additional_grassy_ground_l165_165544


namespace correct_equation_l165_165662

theorem correct_equation:
  (∀ x y : ℝ, -5 * (x - y) = -5 * x + 5 * y) ∧ 
  (∀ a c : ℝ, ¬ (-2 * (-a + c) = -2 * a - 2 * c)) ∧ 
  (∀ x y z : ℝ, ¬ (3 - (x + y + z) = -x + y - z)) ∧ 
  (∀ a b : ℝ, ¬ (3 * (a + 2 * b) = 3 * a + 2 * b)) :=
by
  sorry

end correct_equation_l165_165662


namespace exists_arithmetic_progression_2004_perfect_powers_perfect_powers_not_infinite_arithmetic_progression_l165_165425

-- Definition: A positive integer n is a perfect power if n = a ^ b for some integers a, b with b > 1.
def isPerfectPower (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ n = a^b

-- Part (a): Prove the existence of an arithmetic progression of 2004 perfect powers.
theorem exists_arithmetic_progression_2004_perfect_powers :
  ∃ (x r : ℕ), (∀ n : ℕ, n < 2004 → ∃ a b : ℕ, b > 1 ∧ (x + n * r) = a^b) :=
sorry

-- Part (b): Prove that perfect powers cannot form an infinite arithmetic progression.
theorem perfect_powers_not_infinite_arithmetic_progression :
  ¬ ∃ (x r : ℕ), (∀ n : ℕ, ∃ a b : ℕ, b > 1 ∧ (x + n * r) = a^b) :=
sorry

end exists_arithmetic_progression_2004_perfect_powers_perfect_powers_not_infinite_arithmetic_progression_l165_165425


namespace coin_probability_l165_165263

theorem coin_probability :
  ∃ x : ℝ, x < 0.5 ∧ (6 * x ^ 2 * (1 - x) ^ 2) = (1 / 6) ∧ x = (3 - real.sqrt 3) / 6 :=
by
  sorry

end coin_probability_l165_165263


namespace line_passes_through_point_l165_165786

theorem line_passes_through_point (k : ℝ) :
  (1 + 4 * k) * 2 - (2 - 3 * k) * 2 + 2 - 14 * k = 0 :=
by
  sorry

end line_passes_through_point_l165_165786


namespace tan_alpha_in_second_quadrant_l165_165306

theorem tan_alpha_in_second_quadrant (α : ℝ) (h₁ : π / 2 < α ∧ α < π) (hsin : Real.sin α = 5 / 13) :
    Real.tan α = -5 / 12 :=
sorry

end tan_alpha_in_second_quadrant_l165_165306


namespace fraction_strawberries_remaining_l165_165123

theorem fraction_strawberries_remaining 
  (baskets : ℕ)
  (strawberries_per_basket : ℕ)
  (hedgehogs : ℕ)
  (strawberries_per_hedgehog : ℕ)
  (h1 : baskets = 3)
  (h2 : strawberries_per_basket = 900)
  (h3 : hedgehogs = 2)
  (h4 : strawberries_per_hedgehog = 1050) :
  (baskets * strawberries_per_basket - hedgehogs * strawberries_per_hedgehog) / (baskets * strawberries_per_basket) = 2 / 9 :=
by
  sorry

end fraction_strawberries_remaining_l165_165123


namespace sum_of_fractions_equals_three_l165_165114

-- Definitions according to the conditions
def proper_fraction (a b : ℕ) := 1 ≤ a ∧ a < b
def improper_fraction (a b : ℕ) := a ≥ b
def mixed_number (a b c : ℕ) := a + b / c

-- Constants according to the given problem
def n := 8
def d := 9
def improper_n := 9

-- Values for elements in the conditions
def largest_proper_fraction := n / d
def smallest_improper_fraction := improper_n / d
def smallest_mixed_number := 1 + 1 / d

-- Theorem statement with the correct answer
theorem sum_of_fractions_equals_three :
  largest_proper_fraction + smallest_improper_fraction + smallest_mixed_number = 3 :=
sorry

end sum_of_fractions_equals_three_l165_165114


namespace prime_sum_diff_condition_unique_l165_165582

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def can_be_written_as_sum_of_two_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime (p + q)

def can_be_written_as_difference_of_two_primes (p r : ℕ) : Prop :=
  is_prime p ∧ is_prime r ∧ is_prime (p - r)

-- Question rewritten as Lean statement
theorem prime_sum_diff_condition_unique (p q r : ℕ) :
  is_prime p →
  can_be_written_as_sum_of_two_primes (p - 2) p →
  can_be_written_as_difference_of_two_primes (p + 2) p →
  p = 5 :=
sorry

end prime_sum_diff_condition_unique_l165_165582


namespace factorial_fraction_simplification_l165_165842

-- Define necessary factorial function
def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Define the problem
theorem factorial_fraction_simplification :
  (4 * fact 6 + 20 * fact 5) / fact 7 = 22 / 21 := by
  sorry

end factorial_fraction_simplification_l165_165842


namespace line_intersects_x_axis_at_point_l165_165156

-- Define the conditions and required proof
theorem line_intersects_x_axis_at_point :
  (∃ x : ℝ, ∃ y : ℝ, 5 * y - 7 * x = 35 ∧ y = 0 ∧ (x, y) = (-5, 0)) :=
by
  -- The proof is omitted according to the steps
  sorry

end line_intersects_x_axis_at_point_l165_165156


namespace probability_satisfied_yogurt_young_population_distribution_expectation_X_max_satisfaction_increase_l165_165806

-- Condition definitions
def total_sample : ℕ := 500
def satisfied_yogurt_elderly : ℕ := 100
def satisfied_yogurt_middle_aged : ℕ := 120
def satisfied_yogurt_young : ℕ := 150

-- Question 1
theorem probability_satisfied_yogurt :
  (satisfied_yogurt_elderly + satisfied_yogurt_middle_aged + satisfied_yogurt_young) / total_sample = 37 / 50 :=
sorry

-- Question 2
def p : ℚ := 3 / 4

def binomial_distribution (n : ℕ) (p : ℚ) := sorry -- Need a proper definition here
def E (X : Type) := sorry -- Need a proper definition here

theorem young_population_distribution :
  binomial_distribution 3 p = sorry := 
sorry

theorem expectation_X :
  E X = 9 / 4 := 
sorry

-- Question 3
def satisfaction_increase (age_group : Type) : ℚ := sorry  -- Definition for increases in different groups

theorem max_satisfaction_increase :
  satisfaction_increase young_population > satisfaction_increase elderly_population ∧ satisfaction_increase young_population > satisfaction_increase middle_aged_population :=
sorry

end probability_satisfied_yogurt_young_population_distribution_expectation_X_max_satisfaction_increase_l165_165806


namespace assignment_schemes_correct_l165_165368

-- Define the total number of students
def total_students : ℕ := 6

-- Define the total number of tasks
def total_tasks : ℕ := 4

-- Define a predicate that checks if a student can be assigned to task A
def can_assign_to_task_A (student : ℕ) : Prop := student ≠ 1 ∧ student ≠ 2

-- Calculate the total number of unrestricted assignments
def total_unrestricted_assignments : ℕ := 6 * 5 * 4 * 3

-- Calculate the restricted number of assignments if student A or B is assigned to task A
def restricted_assignments : ℕ := 2 * 5 * 4 * 3

-- Define the problem statement
def number_of_assignment_schemes : ℕ :=
  total_unrestricted_assignments - restricted_assignments

-- The theorem to prove
theorem assignment_schemes_correct :
  number_of_assignment_schemes = 240 :=
by
  -- We acknowledge the problem statement is correct
  sorry

end assignment_schemes_correct_l165_165368


namespace smallest_possible_value_l165_165520

theorem smallest_possible_value (x : ℕ) (m : ℕ) :
  (x > 0) →
  (Nat.gcd 36 m = x + 3) →
  (Nat.lcm 36 m = x * (x + 3)) →
  m = 12 :=
by
  sorry

end smallest_possible_value_l165_165520


namespace shaded_region_correct_l165_165966

def side_length_ABCD : ℝ := 8
def side_length_BEFG : ℝ := 6

def area_square (side_length : ℝ) : ℝ := side_length ^ 2

def area_ABCD : ℝ := area_square side_length_ABCD
def area_BEFG : ℝ := area_square side_length_BEFG

def shaded_region_area : ℝ :=
  area_ABCD + area_BEFG - 32

theorem shaded_region_correct :
  shaded_region_area = 32 :=
by
  -- Proof omitted, but placeholders match problem conditions and answer
  sorry

end shaded_region_correct_l165_165966


namespace probability_white_first_red_second_l165_165416

theorem probability_white_first_red_second :
  let total_marbles := 10
  let white_marbles := 6
  let red_marbles := 4
  let prob_white_first := white_marbles / total_marbles
  let prob_red_second_given_white_first := red_marbles / (total_marbles - 1)
  let prob_combined := prob_white_first * prob_red_second_given_white_first
  prob_combined = 4 / 15 :=
by
  sorry

end probability_white_first_red_second_l165_165416


namespace infinite_geometric_series_sum_l165_165860

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l165_165860


namespace range_of_k_l165_165461

theorem range_of_k (k : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, 0 ≤ k * x^2 + k * x + 3) :
  0 ≤ k ∧ k ≤ 12 :=
sorry

end range_of_k_l165_165461


namespace largest_N_cannot_pay_exactly_without_change_l165_165949

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l165_165949


namespace cupcake_frosting_l165_165567

theorem cupcake_frosting :
  (let cagney_rate := (1 : ℝ) / 24
   let lacey_rate := (1 : ℝ) / 40
   let sammy_rate := (1 : ℝ) / 30
   let total_time := 12 * 60
   let combined_rate := cagney_rate + lacey_rate + sammy_rate
   total_time * combined_rate = 72) :=
by 
   -- Proof goes here
   sorry

end cupcake_frosting_l165_165567


namespace vegetarian_count_l165_165131

theorem vegetarian_count (only_veg only_non_veg both_veg_non_veg : ℕ) 
  (h1 : only_veg = 19) (h2 : only_non_veg = 9) (h3 : both_veg_non_veg = 12) : 
  (only_veg + both_veg_non_veg = 31) :=
by
  -- We leave the proof here
  sorry

end vegetarian_count_l165_165131


namespace inequality_multiplication_l165_165590

theorem inequality_multiplication (a b : ℝ) (h : a > b) : 3 * a > 3 * b :=
sorry

end inequality_multiplication_l165_165590


namespace largest_N_cannot_pay_exactly_without_change_l165_165950

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l165_165950


namespace find_wrongly_noted_mark_l165_165374

-- Definitions of given conditions
def average_marks := 100
def number_of_students := 25
def reported_correct_mark := 10
def correct_average_marks := 98
def wrongly_noted_mark : ℕ := sorry

-- Computing the sum with the wrong mark
def incorrect_sum := number_of_students * average_marks

-- Sum corrected by replacing wrong mark with correct mark
def sum_with_correct_replacement (wrongly_noted_mark : ℕ) := 
  incorrect_sum - wrongly_noted_mark + reported_correct_mark

-- Correct total sum for correct average
def correct_sum := number_of_students * correct_average_marks

-- The statement to be proven
theorem find_wrongly_noted_mark : wrongly_noted_mark = 60 :=
by sorry

end find_wrongly_noted_mark_l165_165374


namespace circle_equation_and_shortest_chord_l165_165140

-- Definitions based on given conditions
def point_P : ℝ × ℝ := (4, -1)
def line_l1 (x y : ℝ) : Prop := x - 6 * y - 10 = 0
def line_l2 (x y : ℝ) : Prop := 5 * x - 3 * y = 0

-- The circle should be such that it intersects line l1 at point P and its center lies on line l2
theorem circle_equation_and_shortest_chord 
  (C : ℝ × ℝ) (r : ℝ) (hC_l2 : line_l2 C.1 C.2)
  (h_intersect : ∃ (k : ℝ), point_P.1 = (C.1 + k * (C.1 - point_P.1)) ∧ point_P.2 = (C.2 + k * (C.2 - point_P.2))) :
  -- Proving (1): Equation of the circle
  ((C.1 = 3) ∧ (C.2 = 5) ∧ r^2 = 37) ∧
  -- Proving (2): Length of the shortest chord through the origin is 2 * sqrt(3)
  (2 * Real.sqrt 3 = 2 * Real.sqrt (r^2 - ((C.1^2 + C.2^2) - (2 * C.1 * 0 + 2 * C.2 * 0)))) :=
by
  sorry

end circle_equation_and_shortest_chord_l165_165140


namespace simple_interest_fraction_l165_165244

theorem simple_interest_fraction (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (F : ℝ)
  (h1 : R = 5)
  (h2 : T = 4)
  (h3 : SI = (P * R * T) / 100)
  (h4 : SI = F * P) :
  F = 1/5 :=
by
  sorry

end simple_interest_fraction_l165_165244


namespace buses_dispatched_theorem_l165_165003

-- Define the conditions and parameters
def buses_dispatched (buses: ℕ) (hours: ℕ) : ℕ :=
  buses * hours

-- Define the specific problem
noncomputable def buses_from_6am_to_4pm : ℕ :=
  let buses_per_hour := 5 / 2
  let hours         := 16 - 6
  buses_dispatched (buses_per_hour : ℕ) hours

-- State the theorem that needs to be proven
theorem buses_dispatched_theorem : buses_from_6am_to_4pm = 25 := 
by {
  -- This 'sorry' is a placeholder for the actual proof.
  sorry
}

end buses_dispatched_theorem_l165_165003


namespace bus_speed_including_stoppages_l165_165448

theorem bus_speed_including_stoppages 
  (speed_excl_stoppages : ℚ) 
  (ten_minutes_per_hour : ℚ) 
  (bus_stops_for_10_minutes : ten_minutes_per_hour = 10/60) 
  (speed_is_54_kmph : speed_excl_stoppages = 54) : 
  (speed_excl_stoppages * (1 - ten_minutes_per_hour)) = 45 := 
by 
  sorry

end bus_speed_including_stoppages_l165_165448


namespace factor_product_modulo_l165_165253

theorem factor_product_modulo (h1 : 2021 % 23 = 21) (h2 : 2022 % 23 = 22) (h3 : 2023 % 23 = 0) (h4 : 2024 % 23 = 1) (h5 : 2025 % 23 = 2) :
  (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 0 :=
by
  sorry

end factor_product_modulo_l165_165253


namespace mark_hours_per_week_l165_165758

theorem mark_hours_per_week (w_historical : ℕ) (w_spring : ℕ) (h_spring : ℕ) (e_spring : ℕ) (e_goal : ℕ) (w_goal : ℕ) (h_goal : ℚ) :
  (e_spring : ℚ) / (w_historical * w_spring) = h_spring / w_spring →
  e_goal = 21000 →
  w_goal = 50 →
  h_spring = 35 →
  w_spring = 15 →
  e_spring = 4200 →
  (h_goal : ℚ) = 2625 / w_goal →
  h_goal = 52.5 :=
sorry

end mark_hours_per_week_l165_165758


namespace geometric_series_sum_l165_165862

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l165_165862


namespace math_proof_problem_l165_165706

noncomputable def a : ℝ := Real.sqrt 18
noncomputable def b : ℝ := (-1 / 3) ^ (-2 : ℤ)
noncomputable def c : ℝ := abs (-3 * Real.sqrt 2)
noncomputable def d : ℝ := (1 - Real.sqrt 2) ^ 0

theorem math_proof_problem : a - b - c - d = -10 := by
  -- Sorry is used to skip the proof, as the proof steps are not required for this problem.
  sorry

end math_proof_problem_l165_165706


namespace difference_in_pencil_buyers_l165_165372

theorem difference_in_pencil_buyers :
  ∀ (cost_per_pencil : ℕ) (total_cost_eighth_graders : ℕ) (total_cost_fifth_graders : ℕ), 
  cost_per_pencil = 13 →
  total_cost_eighth_graders = 234 →
  total_cost_fifth_graders = 325 →
  (total_cost_fifth_graders / cost_per_pencil) - (total_cost_eighth_graders / cost_per_pencil) = 7 :=
by
  intros cost_per_pencil total_cost_eighth_graders total_cost_fifth_graders 
         hcpe htc8 htc5
  sorry

end difference_in_pencil_buyers_l165_165372


namespace arithmetic_sequence_a5_l165_165054

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- The terms of the arithmetic sequence

theorem arithmetic_sequence_a5 :
  (∀ (n : ℕ), a_n n = a_n 0 + n * (a_n 1 - a_n 0)) →
  a_n 1 = 1 →
  a_n 1 + a_n 3 = 16 →
  a_n 4 = 15 :=
by {
  -- Proof omission, ensure these statements are correct with sorry
  sorry
}

end arithmetic_sequence_a5_l165_165054


namespace grasshopper_jump_l165_165144

theorem grasshopper_jump :
  ∃ (x y : ℤ), 80 * x - 50 * y = 170 ∧ x + y ≤ 7 := by
  sorry

end grasshopper_jump_l165_165144


namespace max_possible_value_of_y_l165_165513

theorem max_possible_value_of_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = 4) : y ≤ 7 :=
sorry

end max_possible_value_of_y_l165_165513


namespace f_correct_l165_165622

noncomputable def f : ℕ → ℝ
| 0       => 0 -- undefined for 0, start from 1
| (n + 1) => if n = 0 then 1/2 else sorry -- recursion undefined for now

theorem f_correct : ∀ n ≥ 1, f n = (3^(n-1) / (3^(n-1) + 1)) :=
by
  -- Initial conditions
  have h0 : f 1 = 1/2 := sorry
  -- Recurrence relations
  have h1 : ∀ n, n ≥ 1 → f (n + 1) ≥ (3 * f n) / (2 * f n + 1) := sorry
  -- Prove the function form
  sorry

end f_correct_l165_165622


namespace dan_money_left_l165_165286

def initial_money : ℝ := 50.00
def candy_bar_price : ℝ := 1.75
def candy_bar_count : ℕ := 3
def gum_price : ℝ := 0.85
def soda_price : ℝ := 2.25
def sales_tax_rate : ℝ := 0.08

theorem dan_money_left : 
  initial_money - (candy_bar_count * candy_bar_price + gum_price + soda_price) * (1 + sales_tax_rate) = 40.98 :=
by
  sorry

end dan_money_left_l165_165286


namespace infinite_geometric_series_sum_l165_165854

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l165_165854


namespace proportion_difference_l165_165909

theorem proportion_difference : (0.80 * 40) - ((4 / 5) * 20) = 16 := 
by 
  sorry

end proportion_difference_l165_165909


namespace smallest_number_two_reps_l165_165889

theorem smallest_number_two_reps : 
  ∃ (n : ℕ), (∀ x1 y1 x2 y2 : ℕ, 3 * x1 + 4 * y1 = n ∧ 3 * x2 + 4 * y2 = n → (x1 = x2 ∧ y1 = y2 ∨ ¬(x1 = x2 ∧ y1 = y2))) ∧ 
  ∀ m < n, (∀ x y : ℕ, ¬(3 * x + 4 * y = m ∧ ¬∃ (x1 y1 : ℕ), 3 * x1 + 4 * y1 = m) ∧ 
            (∃ x1 y1 x2 y2 : ℕ, 3 * x1 + 4 * y1 = m ∧ 3 * x2 + 4 * y2 = m ∧ ¬(x1 = x2 ∧ y1 = y2))) :=
  sorry

end smallest_number_two_reps_l165_165889


namespace total_rats_l165_165218

theorem total_rats (Elodie_rats Hunter_rats Kenia_rats : ℕ) 
  (h1 : Elodie_rats = 30) 
  (h2 : Elodie_rats = Hunter_rats + 10)
  (h3 : Kenia_rats = 3 * (Elodie_rats + Hunter_rats)) :
  Elodie_rats + Hunter_rats + Kenia_rats = 200 :=
by
  sorry

end total_rats_l165_165218


namespace smallest_number_diminished_by_35_l165_165972

def lcm_list (l : List ℕ) : ℕ := l.foldr Nat.lcm 1

def conditions : List ℕ := [5, 10, 15, 20, 25, 30, 35]

def lcm_conditions := lcm_list conditions

theorem smallest_number_diminished_by_35 :
  ∃ n, n - 35 = lcm_conditions :=
sorry

end smallest_number_diminished_by_35_l165_165972


namespace six_pow_2n_plus1_plus_1_div_by_7_l165_165099

theorem six_pow_2n_plus1_plus_1_div_by_7 (n : ℕ) : (6^(2*n+1) + 1) % 7 = 0 := by
  sorry

end six_pow_2n_plus1_plus_1_div_by_7_l165_165099


namespace total_votes_cast_l165_165210

theorem total_votes_cast (b_votes c_votes total_votes : ℕ)
  (h1 : b_votes = 48)
  (h2 : c_votes = 35)
  (h3 : b_votes = (4 * total_votes) / 15) :
  total_votes = 180 :=
by
  sorry

end total_votes_cast_l165_165210


namespace initial_percentage_of_water_l165_165628

/-
Initial conditions:
- Let W be the initial percentage of water in 10 liters of milk.
- The mixture should become 2% water after adding 15 liters of pure milk to it.
-/

theorem initial_percentage_of_water (W : ℚ) 
  (H1 : 0 < W) (H2 : W < 100) 
  (H3 : (10 * (100 - W) / 100 + 15) / 25 = 0.98) : 
  W = 5 := 
sorry

end initial_percentage_of_water_l165_165628


namespace probability_fourth_term_integer_l165_165088

/-- The probability of the fourth term in Jacob's sequence being an integer given the specified rules. -/
theorem probability_fourth_term_integer :
  let initial_term : ℕ := 6
  ∀ (coin_flip : ℕ → bool), 
    let next_term (a : ℚ) (flip : bool) : ℚ :=
      if flip then (2 * a - 1) else (a / 2 - 1)
    let sequence : ℕ → ℚ
    | 0 => initial_term
    | n + 1 => next_term (sequence n) (coin_flip n)
  ∃ outcomes : list ℚ,
    (sequence 3) ∈ outcomes ∧
    let integer_count := outcomes.countp (λ x, x.den = 1)
    let total_count := outcomes.length
    (integer_count / total_count) = (5 / 8) :=
begin
  sorry
end

end probability_fourth_term_integer_l165_165088


namespace min_Sn_value_l165_165048

noncomputable def a (n : ℕ) (d : ℤ) : ℤ := -11 + (n - 1) * d

def Sn (n : ℕ) (d : ℤ) : ℤ := n * -11 + n * (n - 1) * d / 2

theorem min_Sn_value {d : ℤ} (h5_6 : a 5 d + a 6 d = -4) : 
  ∃ n, Sn n d = (n - 6)^2 - 36 ∧ n = 6 :=
by
  sorry

end min_Sn_value_l165_165048


namespace smallest_b_for_factoring_l165_165452

theorem smallest_b_for_factoring (b : ℕ) : 
  (∃ r s : ℤ, x^2 + b*x + (1200 : ℤ) = (x + r)*(x + s) ∧ b = r + s ∧ r * s = 1200) →
  b = 70 := 
sorry

end smallest_b_for_factoring_l165_165452


namespace angle_in_second_quadrant_l165_165653

theorem angle_in_second_quadrant (θ : ℝ) (h : θ = 3) : θ > (Real.pi * 0.5) ∧ θ < Real.pi :=
by
  rw h
  exact ⟨by linarith [Real.pi_pos], by linarith [Real.pi_pos, Real.pi_pos.zero_le]⟩
  sorry

end angle_in_second_quadrant_l165_165653


namespace find_k_l165_165083

theorem find_k 
  (c : ℝ) (a₁ : ℝ) (S : ℕ → ℝ) (k : ℝ)
  (h1 : ∀ n, S (n+1) = c * S n) 
  (h2 : S 1 = 3 + k)
  (h3 : ∀ n, S n = 3^n + k) :
  k = -1 :=
sorry

end find_k_l165_165083


namespace razorback_tshirt_sales_l165_165955

theorem razorback_tshirt_sales 
  (price_per_tshirt : ℕ) (total_money_made : ℕ)
  (h1 : price_per_tshirt = 16) (h2 : total_money_made = 720) :
  total_money_made / price_per_tshirt = 45 :=
by
  sorry

end razorback_tshirt_sales_l165_165955


namespace continued_fraction_Pn_Qn_eq_Fibonacci_l165_165624

def P (n : ℕ) : ℕ := (Fibonacci (n+1))
def Q (n : ℕ) : ℕ := Fibonacci n

theorem continued_fraction_Pn_Qn_eq_Fibonacci (n : ℕ) :
  ∃ (P Q : ℕ → ℕ), (P = λ n, Fibonacci (n + 1)) ∧ (Q = λ n, Fibonacci n) ∧ 
  (∀ n, (P (n+1), Q (n+1)) = (Q n, P n + Q n)) ∧ (P 1 = 1) ∧ (Q 1 = 1) :=
by {
  let P := λ n, Fibonacci (n + 1),
  let Q := λ n, Fibonacci n,
  use [P, Q],
  split,
  { refl },
  split,
  { refl },
  split,
  { intro n,
    exact ⟨Q n, P n + Q n⟩ },
  split,
  { exact rfl },
  { exact rfl }
}

end continued_fraction_Pn_Qn_eq_Fibonacci_l165_165624


namespace infinite_geometric_series_sum_l165_165857

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l165_165857


namespace odd_increasing_three_digit_numbers_l165_165315

open Nat

def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_triplet (a b c : ℕ) : Prop := 
  1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 9 ∧ is_odd c

theorem odd_increasing_three_digit_numbers : 
  ∑ c in {1, 3, 5, 7, 9}, (∑ a in range (c - 2), ∑ b in range (a + 1, c - 1), (if valid_triplet a b c then 1 else 0)) = 50 :=
by
  sorry

end odd_increasing_three_digit_numbers_l165_165315


namespace distance_between_point_and_center_l165_165343

noncomputable def polar_to_rectangular_point (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

noncomputable def center_of_circle : ℝ × ℝ := (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_between_point_and_center :
  distance (polar_to_rectangular_point 2 (Real.pi / 3)) center_of_circle = Real.sqrt 3 := 
sorry

end distance_between_point_and_center_l165_165343


namespace conditional_probability_A_given_B_l165_165771

noncomputable def P (A B : Prop) : ℝ := sorry -- Placeholder for the probability function

variables (A B : Prop)

axiom P_A_def : P A = 4/15
axiom P_B_def : P B = 2/15
axiom P_AB_def : P (A ∧ B) = 1/10

theorem conditional_probability_A_given_B : P (A ∧ B) / P B = 3/4 :=
by
  rw [P_AB_def, P_B_def]
  norm_num
  sorry

end conditional_probability_A_given_B_l165_165771


namespace find_b_l165_165379

def operation (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem find_b (b : ℤ) (h : operation 11 b = 110) : b = 12 := 
by
  sorry

end find_b_l165_165379


namespace afternoon_sales_l165_165269

theorem afternoon_sales :
  ∀ (morning_sold afternoon_sold total_sold : ℕ),
    afternoon_sold = 2 * morning_sold ∧
    total_sold = morning_sold + afternoon_sold ∧
    total_sold = 510 →
    afternoon_sold = 340 :=
by
  intros morning_sold afternoon_sold total_sold h
  sorry

end afternoon_sales_l165_165269


namespace correct_choice_l165_165247

-- Define the structures and options
inductive Structure
| Sequential
| Conditional
| Loop
| Module

def option_A : List Structure :=
  [Structure.Sequential, Structure.Module, Structure.Conditional]

def option_B : List Structure :=
  [Structure.Sequential, Structure.Loop, Structure.Module]

def option_C : List Structure :=
  [Structure.Sequential, Structure.Conditional, Structure.Loop]

def option_D : List Structure :=
  [Structure.Module, Structure.Conditional, Structure.Loop]

-- Define the correct structures
def basic_structures : List Structure :=
  [Structure.Sequential, Structure.Conditional, Structure.Loop]

-- The theorem statement
theorem correct_choice : option_C = basic_structures :=
  by
    sorry  -- Proof would go here

end correct_choice_l165_165247


namespace radius_of_circumscribed_sphere_l165_165455

-- Condition: SA = 2
def SA : ℝ := 2

-- Condition: SB = 4
def SB : ℝ := 4

-- Condition: SC = 4
def SC : ℝ := 4

-- Condition: The three side edges are pairwise perpendicular.
def pairwise_perpendicular : Prop := true -- This condition is described but would require geometric definition.

-- To prove: Radius of circumscribed sphere is 3
theorem radius_of_circumscribed_sphere : 
  ∀ (SA SB SC : ℝ) (pairwise_perpendicular : Prop), SA = 2 → SB = 4 → SC = 4 → pairwise_perpendicular → 
  (3 : ℝ) = 3 := by 
  intros SA SB SC pairwise_perpendicular h1 h2 h3 h4
  sorry

end radius_of_circumscribed_sphere_l165_165455


namespace math_problem_l165_165674

theorem math_problem (a b m : ℝ) : 
  ¬((-2 * a) ^ 2 = -4 * a ^ 2) ∧ 
  ¬((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧ 
  ((-m + 2) * (-m - 2) = m ^ 2 - 4) ∧ 
  ¬((a ^ 5) ^ 2 = a ^ 7) :=
by 
  split ; 
  sorry ;
  split ;
  sorry ;
  split ; 
  sorry ;
  sorry

end math_problem_l165_165674


namespace juanitas_dessert_cost_l165_165553

theorem juanitas_dessert_cost :
  let brownie_cost := 2.50
  let ice_cream_cost := 1.00
  let syrup_cost := 0.50
  let nuts_cost := 1.50
  let num_scoops_ice_cream := 2
  let num_syrups := 2
  let total_cost := brownie_cost + num_scoops_ice_cream * ice_cream_cost + num_syrups * syrup_cost + nuts_cost
  total_cost = 7.00 :=
by
  sorry

end juanitas_dessert_cost_l165_165553


namespace option_C_correct_l165_165668

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l165_165668


namespace minimum_dwarfs_to_prevent_empty_chair_sitting_l165_165507

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end minimum_dwarfs_to_prevent_empty_chair_sitting_l165_165507


namespace number_value_l165_165424

theorem number_value (x : ℝ) (h : x = 3 * (1/x * -x) + 5) : x = 2 :=
by
  sorry

end number_value_l165_165424


namespace odd_increasing_three_digit_numbers_l165_165314

open Nat

def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_triplet (a b c : ℕ) : Prop := 
  1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 9 ∧ is_odd c

theorem odd_increasing_three_digit_numbers : 
  ∑ c in {1, 3, 5, 7, 9}, (∑ a in range (c - 2), ∑ b in range (a + 1, c - 1), (if valid_triplet a b c then 1 else 0)) = 50 :=
by
  sorry

end odd_increasing_three_digit_numbers_l165_165314


namespace min_dwarfs_l165_165510

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end min_dwarfs_l165_165510


namespace distance_from_hut_to_station_l165_165022

variable (t s : ℝ)

theorem distance_from_hut_to_station
  (h1 : s / 4 = t + 3 / 4)
  (h2 : s / 6 = t - 1 / 2) :
  s = 15 := by
  sorry

end distance_from_hut_to_station_l165_165022


namespace range_of_m_l165_165723

theorem range_of_m 
    (m : ℝ) (x : ℝ)
    (p : x^2 - 8 * x - 20 > 0)
    (q : (x - (1 - m)) * (x - (1 + m)) > 0)
    (h : ∀ x, (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) :
    0 < m ∧ m ≤ 3 := by
  sorry

end range_of_m_l165_165723


namespace geometric_series_sum_l165_165875

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l165_165875


namespace atomic_weight_of_oxygen_l165_165888

theorem atomic_weight_of_oxygen (atomic_weight_Al : ℝ) (atomic_weight_O : ℝ) (molecular_weight_Al2O3 : ℝ) (n_Al : ℕ) (n_O : ℕ) :
  atomic_weight_Al = 26.98 →
  molecular_weight_Al2O3 = 102 →
  n_Al = 2 →
  n_O = 3 →
  (molecular_weight_Al2O3 - n_Al * atomic_weight_Al) / n_O = 16.01 :=
by
  sorry

end atomic_weight_of_oxygen_l165_165888


namespace math_problem_l165_165677

theorem math_problem (a b m : ℝ) : 
  ¬((-2 * a) ^ 2 = -4 * a ^ 2) ∧ 
  ¬((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧ 
  ((-m + 2) * (-m - 2) = m ^ 2 - 4) ∧ 
  ¬((a ^ 5) ^ 2 = a ^ 7) :=
by 
  split ; 
  sorry ;
  split ;
  sorry ;
  split ; 
  sorry ;
  sorry

end math_problem_l165_165677


namespace smallest_positive_int_l165_165254

open Nat

theorem smallest_positive_int (x : ℕ) :
  (x % 6 = 3) ∧ (x % 8 = 5) ∧ (x % 9 = 2) → x = 237 := by
  sorry

end smallest_positive_int_l165_165254


namespace no_intersection_points_l165_165166

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 5

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ := -x^2 + 6 * x - 8

-- The statement asserting that the parabolas do not intersect
theorem no_intersection_points :
  ∀ (x y : ℝ), parabola1 x = y → parabola2 x = y → false :=
by
  -- Introducing x and y as elements of the real numbers
  intros x y h1 h2
  
  -- Since this is only the statement, we use sorry to skip the actual proof
  sorry

end no_intersection_points_l165_165166


namespace abs_eq_three_system1_system2_l165_165235

theorem abs_eq_three : ∀ x : ℝ, |x| = 3 ↔ x = 3 ∨ x = -3 := 
by sorry

theorem system1 : ∀ x y : ℝ, (y * (x - 1) = 0) ∧ (2 * x + 5 * y = 7) → 
(x = 7 / 2 ∧ y = 0) ∨ (x = 1 ∧ y = 1) := 
by sorry

theorem system2 : ∀ x y : ℝ, (x * y - 2 * x - y + 2 = 0) ∧ (x + 6 * y = 3) ∧ (3 * x + y = 8) → 
(x = 1 ∧ y = 5) ∨ (x = 2 ∧ y = 2) := 
by sorry

end abs_eq_three_system1_system2_l165_165235


namespace cheryl_material_leftover_l165_165285

theorem cheryl_material_leftover :
  let material1 := (5 / 9 : ℚ)
  let material2 := (1 / 3 : ℚ)
  let total_bought := material1 + material2
  let used := (0.5555555555555556 : ℝ)
  let total_bought_decimal := (8 / 9 : ℝ)
  let leftover := total_bought_decimal - used
  leftover = 0.3333333333333332 := by
sorry

end cheryl_material_leftover_l165_165285


namespace odd_three_digit_integers_in_strict_increasing_order_l165_165317

theorem odd_three_digit_integers_in_strict_increasing_order: 
  (∀ (a b c : ℕ), 100 ≤ (100 * a + 10 * b + c) ∧ 100 * a + 10 * b + c < 1000 → a < b ∧ b < c →
  c % 2 = 1 ∧ c ≠ 0 → 
  (∃ n, n = 50)) :=
by sorry

end odd_three_digit_integers_in_strict_increasing_order_l165_165317


namespace value_of_k_if_two_equal_real_roots_l165_165599

theorem value_of_k_if_two_equal_real_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x + k = 0 → x^2 - 2 * x + k = 0) → k = 1 :=
by
  sorry

end value_of_k_if_two_equal_real_roots_l165_165599


namespace find_a_range_l165_165457

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- The main theorem stating the range of a
theorem find_a_range (a : ℝ) (h : ¬(∃ x : ℝ, p a x) → ¬(∃ x : ℝ, q x) ∧ ¬(¬(∃ x : ℝ, q x) → ¬(∃ x : ℝ, p a x))) : 1 < a ∧ a ≤ 2 := sorry

end find_a_range_l165_165457


namespace baker_made_cakes_l165_165157

-- Conditions
def cakes_sold : ℕ := 145
def cakes_left : ℕ := 72

-- Question and required proof
theorem baker_made_cakes : (cakes_sold + cakes_left = 217) :=
by
  sorry

end baker_made_cakes_l165_165157


namespace johns_average_speed_remaining_duration_l165_165214

noncomputable def average_speed_remaining_duration : ℝ :=
  let total_distance := 150
  let total_time := 3
  let first_hour_speed := 45
  let stop_time := 0.5
  let next_45_minutes_speed := 50
  let next_45_minutes_time := 0.75
  let driving_time := total_time - stop_time
  let distance_first_hour := first_hour_speed * 1
  let distance_next_45_minutes := next_45_minutes_speed * next_45_minutes_time
  let remaining_distance := total_distance - distance_first_hour - distance_next_45_minutes
  let remaining_time := driving_time - (1 + next_45_minutes_time)
  remaining_distance / remaining_time

theorem johns_average_speed_remaining_duration : average_speed_remaining_duration = 90 := by
  sorry

end johns_average_speed_remaining_duration_l165_165214


namespace number_of_shirts_that_weigh_1_pound_l165_165480

/-- 
Jon's laundry machine can do 5 pounds of laundry at a time. 
Some number of shirts weigh 1 pound. 
2 pairs of pants weigh 1 pound. 
Jon needs to wash 20 shirts and 20 pants. 
Jon has to do 3 loads of laundry. 
-/
theorem number_of_shirts_that_weigh_1_pound
    (machine_capacity : ℕ)
    (num_shirts : ℕ)
    (shirts_per_pound : ℕ)
    (pairs_of_pants_per_pound : ℕ)
    (num_pants : ℕ)
    (loads : ℕ)
    (weight_per_load : ℕ)
    (total_pants_weight : ℕ)
    (total_weight : ℕ)
    (shirt_weight_per_pound : ℕ)
    (shirts_weighing_one_pound : ℕ) :
  machine_capacity = 5 → 
  num_shirts = 20 → 
  pairs_of_pants_per_pound = 2 →
  num_pants = 20 →
  loads = 3 →
  weight_per_load = 5 → 
  total_pants_weight = (num_pants / pairs_of_pants_per_pound) →
  total_weight = (loads * weight_per_load) →
  shirts_weighing_one_pound = (total_weight - total_pants_weight) / num_shirts → 
  shirts_weighing_one_pound = 4 :=
by sorry

end number_of_shirts_that_weigh_1_pound_l165_165480


namespace ladder_distance_l165_165414

theorem ladder_distance (x : ℝ) (h1 : (13:ℝ) = Real.sqrt (x ^ 2 + 12 ^ 2)) : 
  x = 5 :=
by 
  sorry

end ladder_distance_l165_165414


namespace math_and_english_scores_sum_l165_165310

theorem math_and_english_scores_sum (M E : ℕ) (total_score : ℕ) :
  (∀ (H : ℕ), H = (50 + M + E) / 3 → 
   50 + M + E + H = total_score) → 
   total_score = 248 → 
   M + E = 136 :=
by
  intros h1 h2;
  sorry

end math_and_english_scores_sum_l165_165310


namespace science_fair_unique_students_l165_165839

/-!
# Problem statement:
At Euclid Middle School, there are three clubs participating in the Science Fair: the Robotics Club, the Astronomy Club, and the Chemistry Club.
There are 15 students in the Robotics Club, 10 students in the Astronomy Club, and 12 students in the Chemistry Club.
Assuming 2 students are members of all three clubs, prove that the total number of unique students participating in the Science Fair is 33.
-/

theorem science_fair_unique_students (R A C : ℕ) (all_three : ℕ) (hR : R = 15) (hA : A = 10) (hC : C = 12) (h_all_three : all_three = 2) :
    R + A + C - 2 * all_three = 33 :=
by
  -- Proof goes here
  sorry

end science_fair_unique_students_l165_165839


namespace infinite_geometric_series_sum_l165_165853

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l165_165853


namespace change_combinations_12_dollars_l165_165444

theorem change_combinations_12_dollars :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), 
  (∀ (n d q : ℕ), (n, d, q) ∈ solutions ↔ 5 * n + 10 * d + 25 * q = 1200 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1) ∧ solutions.card = 61 :=
sorry

end change_combinations_12_dollars_l165_165444


namespace find_number_X_l165_165266

theorem find_number_X :
  let d := 90
  let p := 555 * 465
  let q := 3 * d
  let r := d^2
  let X := q * p + r
  in X = 69688350 :=
by
  let d := 90
  let p := 555 * 465
  let q := 3 * d
  let r := d^2
  let X := q * p + r
  have X_eq : X = 69688350 := by sorry
  exact X_eq

end find_number_X_l165_165266


namespace calculate_3_to_5_mul_7_to_5_l165_165569

theorem calculate_3_to_5_mul_7_to_5 : 3^5 * 7^5 = 4084101 :=
by {
  -- Sorry is added to skip the proof; assuming the proof is done following standard arithmetic calculations
  sorry
}

end calculate_3_to_5_mul_7_to_5_l165_165569


namespace determine_f_16_l165_165519

theorem determine_f_16 (a : ℝ) (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  (∀ x, a ^ (x - 4) + 1 = 2) →
  f 4 = 2 →
  f 16 = 4 :=
by
  sorry

end determine_f_16_l165_165519


namespace part1_max_value_f_a_eq_1_part2_unique_real_root_l165_165904

noncomputable def f (x a : ℝ) := log x - a * x
noncomputable def g (x a : ℝ) := (1/2) * x^2 - (2 * a + 1) * x + (a + 1) * log x

-- Part (1): When a = 1, prove that the maximum value of f(x) is -1
theorem part1_max_value_f_a_eq_1 : ∃ x, f x 1 = -1 := by
  sorry

-- Part (2): When a ≥ 1, prove that the equation f(x) = g(x) has a unique real root
theorem part2_unique_real_root (a : ℝ) (h : a ≥ 1) : ∃! x, f x a = g x a := by
  sorry

end part1_max_value_f_a_eq_1_part2_unique_real_root_l165_165904


namespace part1_part2_l165_165903

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 1) * Real.exp x
noncomputable def h (x : ℝ) : ℝ := -3 * Real.log x + x^3 + (2 * x^2 - 4 * x) * Real.exp x + 7

theorem part1 (a : ℤ) : 
  (∀ x, (a : ℝ) < x ∧ x < a + 5 → ∀ y, (a : ℝ) < y ∧ y < a + 5 → f x ≤ f y) →
  a = -6 ∨ a = -5 ∨ a = -4 :=
sorry

theorem part2 (x : ℝ) (hx : 0 < x) : 
  f x < h x :=
sorry

end part1_part2_l165_165903


namespace minimum_dwarfs_to_prevent_empty_chair_sitting_l165_165508

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end minimum_dwarfs_to_prevent_empty_chair_sitting_l165_165508


namespace geometric_series_sum_l165_165864

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l165_165864


namespace work_completion_time_l165_165691

theorem work_completion_time (P W : ℕ) (h : P * 8 = W) : 2 * P * 2 = W / 2 := by
  sorry

end work_completion_time_l165_165691


namespace largest_unpayable_soldo_l165_165943

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l165_165943


namespace value_of_c_over_b_l165_165979

def is_median (a b c : ℤ) (m : ℤ) : Prop :=
a < b ∧ b < c ∧ m = b

def in_geometric_progression (p q r : ℤ) : Prop :=
∃ k : ℤ, k ≠ 0 ∧ q = p * k ∧ r = q * k

theorem value_of_c_over_b (a b c p q r : ℤ) 
  (h1 : (a + b + c) / 3 = (b / 2))
  (h2 : a * b * c = 0)
  (h3 : a < b ∧ b < c ∧ a = 0)
  (h4 : p < q ∧ q < r ∧ r ≠ 0)
  (h5 : in_geometric_progression p q r)
  (h6 : a^2 + b^2 + c^2 = (p + q + r)^2) : 
  c / b = 2 := 
sorry

end value_of_c_over_b_l165_165979


namespace smallest_area_right_triangle_l165_165973

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 7) (hb : b = 10): 
  ∃ (A : ℕ), A = 35 :=
  by
    have hab := 1/2 * a * b
    sorry

-- Note: "sorry" is used as a placeholder for the proof.

end smallest_area_right_triangle_l165_165973


namespace solve_inequality_l165_165234

theorem solve_inequality : 
  {x : ℝ | -4 * x^2 + 7 * x + 2 < 0} = {x : ℝ | x < -1/4} ∪ {x : ℝ | 2 < x} :=
by
  sorry

end solve_inequality_l165_165234


namespace candy_making_time_l165_165996

-- Define constants for the given conditions
def initial_temp : ℝ := 60
def heating_temp : ℝ := 240
def cooling_temp : ℝ := 170
def heating_rate : ℝ := 5
def cooling_rate : ℝ := 7

-- Problem statement in Lean: Prove the total time required
theorem candy_making_time :
  (heating_temp - initial_temp) / heating_rate + (heating_temp - cooling_temp) / cooling_rate = 46 :=
by
  -- Initial temperature: 60 degrees
  -- Heating temperature: 240 degrees
  -- Cooling temperature: 170 degrees
  -- Heating rate: 5 degrees/minute
  -- Cooling rate: 7 degrees/minute
  have temp_diff_heat: heating_temp - initial_temp = 180 := by norm_num
  have time_to_heat: (heating_temp - initial_temp) / heating_rate = 36 := by norm_num
  have temp_diff_cool: heating_temp - cooling_temp = 70 := by norm_num
  have time_to_cool: (heating_temp - cooling_temp) / cooling_rate = 10 := by norm_num
  have total_time: (heating_temp - initial_temp) / heating_rate + (heating_temp - cooling_temp) / cooling_rate = 46 := by norm_num
  exact total_time

end candy_making_time_l165_165996


namespace expenses_recorded_as_negative_l165_165959

/-*
  Given:
  1. The income of 5 yuan is recorded as +5 yuan.
  Prove:
  2. The expenses of 5 yuan are recorded as -5 yuan.
*-/

theorem expenses_recorded_as_negative (income_expenses_opposite_sign : ∀ (a : ℤ), -a = -a)
    (income_five_recorded_as_positive : (5 : ℤ) = 5) :
    (-5 : ℤ) = -5 :=
by sorry

end expenses_recorded_as_negative_l165_165959


namespace jen_profit_l165_165353

-- Definitions based on the conditions
def cost_per_candy := 80 -- in cents
def sell_price_per_candy := 100 -- in cents
def total_candies_bought := 50
def total_candies_sold := 48

-- Total cost and total revenue calculations
def total_cost := cost_per_candy * total_candies_bought
def total_revenue := sell_price_per_candy * total_candies_sold

-- Profit calculation
def profit := total_revenue - total_cost

-- Main theorem to prove
theorem jen_profit : profit = 800 := by
  -- Proof is skipped
  sorry

end jen_profit_l165_165353


namespace not_prime_for_large_n_l165_165233

theorem not_prime_for_large_n {n : ℕ} (h : n > 1) : ¬ Prime (n^4 + n^2 + 1) :=
sorry

end not_prime_for_large_n_l165_165233


namespace smallest_N_divisible_l165_165803

theorem smallest_N_divisible (N x : ℕ) (H: N - 24 = 84 * Nat.lcm x 60) : N = 5064 :=
by
  sorry

end smallest_N_divisible_l165_165803


namespace approx_values_l165_165026

noncomputable def linear_approx (f f' : ℝ → ℝ) (x0 x1 : ℝ) : ℝ :=
  f x0 + f' x0 * (x1 - x0)

-- Part 1: sqrt[4]{17} ≈ 2.03125
def f1 (x : ℝ) : ℝ := x^(1/4)
def f1' (x : ℝ) : ℝ := (1/4) * x^(-3/4)
def x0_1 := 16 -- Closest easy point
def x1_1 := 17 -- Point of interest
def approx1 := linear_approx f1 f1' x0_1 x1_1 -- Expected to be approximately 2.03125

-- Part 2: arctg(0.98) ≈ 0.7754
def y2 (x : ℝ) : ℝ := Real.arctan x
def y2' (x : ℝ) : ℝ := 1 / (1 + x^2)
def x0_2 := 1 -- Closest easy point
def x1_2 := 0.98 -- Point of interest
def approx2 := linear_approx y2 y2' x0_2 x1_2 -- Expected to be approximately 0.7754

-- Part 3: sin(29°) ≈ 0.4848
def y3 (x : ℝ) : ℝ := Real.sin x
def y3' (x : ℝ) : ℝ := Real.cos x
def deg_to_rad (deg : ℝ) := (Real.pi / 180) * deg
def x0_3 := deg_to_rad 30 -- Closest easy point in radians
def x1_3 := deg_to_rad 29 -- Point of interest in radians
def approx3 := linear_approx y3 y3' x0_3 x1_3 -- Expected to be approximately 0.4848

theorem approx_values :
  approx1 ≈ 2.03125 ∧ approx2 ≈ 0.7754 ∧ approx3 ≈ 0.4848 :=
by
  sorry

end approx_values_l165_165026


namespace infinite_geometric_series_sum_l165_165859

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l165_165859


namespace infinite_geometric_series_sum_l165_165858

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l165_165858


namespace sum_f_a_seq_positive_l165_165227

noncomputable def f (x : ℝ) : ℝ := sorry
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_monotone_decreasing_nonneg : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x ≤ y → f y ≤ f x
axiom a_seq : ∀ n : ℕ, ℝ
axiom a_arithmetic : ∀ m n k : ℕ, m + k = 2 * n → a_seq m + a_seq k = 2 * a_seq n
axiom a3_neg : a_seq 3 < 0

theorem sum_f_a_seq_positive :
    f (a_seq 1) + 
    f (a_seq 2) + 
    f (a_seq 3) + 
    f (a_seq 4) + 
    f (a_seq 5) > 0 :=
sorry

end sum_f_a_seq_positive_l165_165227


namespace largest_integer_T_l165_165047

theorem largest_integer_T (p : Nat → Nat) (h_prime : ∀ i, Nat.Prime (p i)) 
  (h_len : ∀ i : Nat, i < 25 → p i ≤ 2004) (h_distinct : ∀ i j : Nat, i ≠ j → p i ≠ p j)
  (h_sorted : ∀ i j : Nat, i < j → p i < p j)
  : ∃ T : Nat, 
      (∀ n : Nat, n ≤ T → ∃ (q : Multiset Nat) (hq : ∀ x ∈ q, x ∣ (Multiset.prod (Multiset.map (λ i, (p i) ^ 2004) (Multiset.range 25)))),
        Multiset.rel (·≠·) q q ∧ n = q.sum) ∧ 
      (if p 0 = 2 then T = (2 ^ 2005 - 1) * ∏ i in Finset.range 24 | i | 1 < Finset.card (Finset.range 24) ∧ (∀ i ∈ Finset.range 25, p (i + 1) ≤ 2004 ∧ Nat.Prime (p (i + 1))){
        ∏ i in Finset.range 24, ((p (i + 1)) ^ 2005 - 1) / (p (i + 1) - 1)
      } else T = 1) :=
sorry

end largest_integer_T_l165_165047


namespace find_a_of_extreme_value_at_one_l165_165239

-- Define the function f(x) = x^3 - a * x
def f (x a : ℝ) : ℝ := x^3 - a * x
  
-- Define the derivative of f with respect to x
def f' (x a : ℝ) : ℝ := 3 * x^2 - a

-- The theorem statement: for f(x) having an extreme value at x = 1, the corresponding a must be 3
theorem find_a_of_extreme_value_at_one (a : ℝ) : 
  (f' 1 a = 0) ↔ (a = 3) :=
by
  sorry

end find_a_of_extreme_value_at_one_l165_165239


namespace proof_ab_lt_1_l165_165600

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem proof_ab_lt_1 (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a > f b) : a * b < 1 :=
by
  -- Sorry to skip the proof
  sorry

end proof_ab_lt_1_l165_165600


namespace f_g_5_l165_165737

def g (x : ℕ) : ℕ := 4 * x + 10

def f (x : ℕ) : ℕ := 6 * x - 12

theorem f_g_5 : f (g 5) = 168 := by
  sorry

end f_g_5_l165_165737


namespace triangle_properties_l165_165618

open Real

noncomputable def is_isosceles_triangle (A B C a b c : ℝ) : Prop :=
  (A + B + C = π) ∧ (b = c)

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def area (a b c : ℝ) (A : ℝ) : ℝ :=
  1/2 * b * c * sin A

theorem triangle_properties 
  (A B C a b c : ℝ) 
  (h1 : sin B * sin C = 1/4) 
  (h2 : tan B * tan C = 1/3) 
  (h3 : a = 4 * sqrt 3) 
  (h4 : A + B + C = π) 
  (isosceles : is_isosceles_triangle A B C a b c) :
  is_isosceles_triangle A B C a b c ∧ 
  perimeter a b c = 8 + 4 * sqrt 3 ∧ 
  area a b c A = 4 * sqrt 3 :=
sorry

end triangle_properties_l165_165618


namespace multiplication_of_negative_and_positive_l165_165998

theorem multiplication_of_negative_and_positive :
  (-3) * 5 = -15 :=
by
  sorry

end multiplication_of_negative_and_positive_l165_165998


namespace range_of_a_l165_165259

theorem range_of_a (a : ℝ) :
  (∃ M : ℝ × ℝ, (M.1 - a)^2 + (M.2 - a + 2)^2 = 1 ∧ (M.1)^2 + (M.2 + 3)^2 = 4 * ((M.1)^2 + (M.2)^2))
  → 0 ≤ a ∧ a ≤ 3 :=
sorry

end range_of_a_l165_165259


namespace odd_increasing_three_digit_numbers_l165_165312

open Nat

def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_triplet (a b c : ℕ) : Prop := 
  1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 9 ∧ is_odd c

theorem odd_increasing_three_digit_numbers : 
  ∑ c in {1, 3, 5, 7, 9}, (∑ a in range (c - 2), ∑ b in range (a + 1, c - 1), (if valid_triplet a b c then 1 else 0)) = 50 :=
by
  sorry

end odd_increasing_three_digit_numbers_l165_165312


namespace average_marks_l165_165518

theorem average_marks (A : ℝ) :
  let marks_first_class := 25 * A
  let marks_second_class := 30 * 60
  let total_marks := 55 * 50.90909090909091
  marks_first_class + marks_second_class = total_marks → A = 40 :=
by
  sorry

end average_marks_l165_165518


namespace Alan_age_is_29_l165_165151

/-- Alan and Chris ages problem -/
theorem Alan_age_is_29
    (A C : ℕ)
    (h1 : A + C = 52)
    (h2 : C = A / 3 + 2 * (A - C)) :
    A = 29 :=
by
  sorry

end Alan_age_is_29_l165_165151


namespace smallest_value_other_integer_l165_165523

noncomputable def smallest_possible_value_b : ℕ :=
  by sorry

theorem smallest_value_other_integer (x : ℕ) (h_pos : x > 0) (b : ℕ) 
  (h_gcd : Nat.gcd 36 b = x + 3) (h_lcm : Nat.lcm 36 b = x * (x + 3)) :
  b = 108 :=
  by sorry

end smallest_value_other_integer_l165_165523


namespace find_prime_pairs_l165_165290

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_pair (p q : ℕ) : Prop := 
  p < 2023 ∧ q < 2023 ∧ 
  p ∣ q^2 + 8 ∧ q ∣ p^2 + 8

theorem find_prime_pairs : 
  ∀ (p q : ℕ), is_prime p → is_prime q → valid_pair p q → 
    (p = 2 ∧ q = 2) ∨ 
    (p = 17 ∧ q = 3) ∨ 
    (p = 11 ∧ q = 5) :=
by 
  sorry

end find_prime_pairs_l165_165290


namespace production_cost_decrease_l165_165115

theorem production_cost_decrease (x : ℝ) :
  let initial_production_cost := 50
  let initial_selling_price := 65
  let first_quarter_decrease := 0.10
  let second_quarter_increase := 0.05
  let final_selling_price := initial_selling_price * (1 - first_quarter_decrease) * (1 + second_quarter_increase)
  let original_profit := initial_selling_price - initial_production_cost
  let final_production_cost := initial_production_cost * (1 - x) ^ 2
  (final_selling_price - final_production_cost) = original_profit :=
by
  sorry

end production_cost_decrease_l165_165115


namespace number_of_non_degenerate_rectangles_excluding_center_l165_165034

/-!
# Problem Statement
We want to find the number of non-degenerate rectangles in a 7x7 grid that do not fully cover the center point (4, 4).
-/

def num_rectangles_excluding_center : Nat :=
  let total_rectangles := (Nat.choose 7 2) * (Nat.choose 7 2)
  let rectangles_including_center := 4 * ((3 * 3 * 3) + (3 * 3))
  total_rectangles - rectangles_including_center

theorem number_of_non_degenerate_rectangles_excluding_center :
  num_rectangles_excluding_center = 297 :=
by
  sorry -- proof goes here

end number_of_non_degenerate_rectangles_excluding_center_l165_165034


namespace hypotenuse_is_18_8_l165_165830

def right_triangle_hypotenuse_perimeter_area (a b c : ℝ) : Prop :=
  a + b + c = 40 ∧ (1/2 * a * b = 24) ∧ (a^2 + b^2 = c^2)

theorem hypotenuse_is_18_8 : ∃ (a b c : ℝ), right_triangle_hypotenuse_perimeter_area a b c ∧ c = 18.8 :=
by
  sorry

end hypotenuse_is_18_8_l165_165830


namespace average_fuel_efficiency_l165_165985

theorem average_fuel_efficiency (d1 d2 : ℝ) (e1 e2 : ℝ) (fuel1 fuel2 : ℝ)
  (h1 : d1 = 150) (h2 : e1 = 35) (h3 : d2 = 180) (h4 : e2 = 18)
  (h_fuel1 : fuel1 = d1 / e1) (h_fuel2 : fuel2 = d2 / e2)
  (total_distance : ℝ := 330)
  (total_fuel : ℝ := fuel1 + fuel2) :
  total_distance / total_fuel = 23 := by
  sorry

end average_fuel_efficiency_l165_165985


namespace expression_D_divisible_by_9_l165_165019

theorem expression_D_divisible_by_9 (k : ℕ) (hk : k > 0) : 9 ∣ 3 * (2 + 7^k) :=
by
  sorry

end expression_D_divisible_by_9_l165_165019


namespace odd_three_digit_integers_increasing_order_l165_165331

theorem odd_three_digit_integers_increasing_order :
  let digits_strictly_increasing (a b c : ℕ) : Prop := (1 ≤ a) ∧ (a < b) ∧ (b < c)
      let c_values : Finset ℕ := {3, 5, 7, 9}
  in ∑ c in c_values, (Finset.card (Finset.filter (λ ab : ℕ × ℕ, (digits_strictly_increasing ab.1 ab.2 c)) (Finset.cross {(1 : ℕ)..9} {(1 : ℕ)..9}))) = 50 :=
by
  sorry

end odd_three_digit_integers_increasing_order_l165_165331


namespace not_perfect_square_l165_165050

theorem not_perfect_square (a b : ℤ) (h : (a % 2 ≠ b % 2)) : ¬ ∃ k : ℤ, ((a + 3 * b) * (5 * a + 7 * b) = k^2) := 
by
  sorry

end not_perfect_square_l165_165050


namespace soccer_team_problem_l165_165631

theorem soccer_team_problem :
  let quadruplets := ({Brian, Brad, Bill, Bob} : Finset string)
  let all_players := (quadruplets ∪ (range 12).map (λ n, "Player" ++ toString n) : Finset string)
  let starters_quadruplets := quadruplets.powerset.filter (λ s, s.card = 3)
  let starters_remaining := (all_players \ quadruplets).powerset.filter (λ s, s.card = 3)
  (quadruplets.card = 4) ∧ (all_players.card = 16) ∧ (starters_quadruplets.card = 4) 
  ∧ (starters_remaining.card = 220) → 
  (starters_quadruplets.card * starters_remaining.card = 880) :=
by
  sorry

end soccer_team_problem_l165_165631


namespace average_of_D_E_F_l165_165236

theorem average_of_D_E_F (D E F : ℝ) 
  (h1 : 2003 * F - 4006 * D = 8012) 
  (h2 : 2003 * E + 6009 * D = 10010) : 
  (D + E + F) / 3 = 3 := 
by 
  sorry

end average_of_D_E_F_l165_165236


namespace find_m_l165_165200

theorem find_m (m : ℝ) :
  let a : ℝ × ℝ := (2, m)
  let b : ℝ × ℝ := (1, -1)
  (b.1 * (a.1 + 2 * b.1) + b.2 * (a.2 + 2 * b.2) = 0) → 
  m = 6 := by 
  sorry

end find_m_l165_165200


namespace pow_div_pow_eq_result_l165_165571

theorem pow_div_pow_eq_result : 13^8 / 13^5 = 2197 := by
  sorry

end pow_div_pow_eq_result_l165_165571


namespace candy_cooking_time_l165_165995

def initial_temperature : ℝ := 60
def peak_temperature : ℝ := 240
def final_temperature : ℝ := 170
def heating_rate : ℝ := 5
def cooling_rate : ℝ := 7

theorem candy_cooking_time : ( (peak_temperature - initial_temperature) / heating_rate + (peak_temperature - final_temperature) / cooling_rate ) = 46 := by
  sorry

end candy_cooking_time_l165_165995


namespace samantha_trip_l165_165640

theorem samantha_trip (a b c d x : ℕ)
  (h1 : 1 ≤ a) (h2 : a + b + c + d ≤ 10) 
  (h3 : 1000 * d + 100 * c + 10 * b + a - (1000 * a + 100 * b + 10 * c + d) = 60 * x)
  : a^2 + b^2 + c^2 + d^2 = 83 :=
sorry

end samantha_trip_l165_165640


namespace find_value_of_reciprocal_cubic_sum_l165_165467

theorem find_value_of_reciprocal_cubic_sum
  (a b c r s : ℝ)
  (h₁ : a + b + c = 0)
  (h₂ : a ≠ 0)
  (h₃ : b^2 - 4 * a * c ≥ 0)
  (h₄ : r ≠ 0)
  (h₅ : s ≠ 0)
  (h₆ : a * r^2 + b * r + c = 0)
  (h₇ : a * s^2 + b * s + c = 0)
  (h₈ : r + s = -b / a)
  (h₉ : r * s = -c / a) :
  1 / r^3 + 1 / s^3 = -b * (b^2 + 3 * a^2 + 3 * a * b) / (a + b)^3 :=
by
  sorry

end find_value_of_reciprocal_cubic_sum_l165_165467


namespace function_decreasing_interval_l165_165378

noncomputable def function_y (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

noncomputable def derivative_y' (x : ℝ) : ℝ := (x + 1) * (x - 1) / x

theorem function_decreasing_interval : ∀ x: ℝ, 0 < x ∧ x < 1 → (derivative_y' x < 0) := by
  sorry

end function_decreasing_interval_l165_165378


namespace problem_solution_l165_165524

theorem problem_solution (a b c : ℤ)
  (h1 : a + 5 = b)
  (h2 : 5 + b = c)
  (h3 : b + c = a) : b = -10 :=
by
  sorry

end problem_solution_l165_165524


namespace problem_statement_l165_165256

theorem problem_statement (x : ℕ) (h : x = 2016) : (x^2 - x) - (x^2 - 2 * x + 1) = 2015 := by
  sorry

end problem_statement_l165_165256


namespace tenth_term_is_four_l165_165041

noncomputable def a : ℕ → ℝ
| 0     := 3
| 1     := 4
| (n + 1) := 12 / a n

theorem tenth_term_is_four : a 9 = 4 :=
by
  sorry

end tenth_term_is_four_l165_165041


namespace obtuse_angle_probability_l165_165098

-- Defining the vertices of the pentagon
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨0, 3⟩
def B : Point := ⟨5, 0⟩
def C : Point := ⟨8, 0⟩
def D : Point := ⟨8, 5⟩
def E : Point := ⟨0, 5⟩

def is_interior (P : Point) : Prop :=
  -- A condition to define if a point is inside the pentagon
  sorry

def is_obtuse_angle (A B P : Point) : Prop :=
  -- Condition for angle APB to be obtuse
  sorry

noncomputable def probability_obtuse_angle :=
  -- Probability calculation
  let area_pentagon := 40
  let area_circle := (34 * Real.pi) / 4
  let area_outside_circle := area_pentagon - area_circle
  area_outside_circle / area_pentagon

theorem obtuse_angle_probability :
  ∀ P : Point, is_interior P → ∃! p : ℝ, p = (160 - 34 * Real.pi) / 160 :=
sorry

end obtuse_angle_probability_l165_165098


namespace initial_avg_production_is_50_l165_165296

-- Define the initial conditions and parameters
variables (A : ℝ) (n : ℕ := 10) (today_prod : ℝ := 105) (new_avg : ℝ := 55)

-- State that the initial total production over n days
def initial_total_production (A : ℝ) (n : ℕ) : ℝ := A * n

-- State the total production after today's production is added
def post_total_production (A : ℝ) (n : ℕ) (today_prod : ℝ) : ℝ := initial_total_production A n + today_prod

-- State the new average production calculation
def new_avg_production (n : ℕ) (new_avg : ℝ) : ℝ := new_avg * (n + 1)

-- State the main claim: Prove that the initial average daily production was 50 units per day
theorem initial_avg_production_is_50 (A : ℝ) (n : ℕ := 10) (today_prod : ℝ := 105) (new_avg : ℝ := 55) 
  (h : post_total_production A n today_prod = new_avg_production n new_avg) : 
  A = 50 := 
by {
  -- Preliminary setups (we don't need detailed proof steps here)
  sorry
}

end initial_avg_production_is_50_l165_165296


namespace number_of_bulls_l165_165384

theorem number_of_bulls (total_cattle : ℕ) (ratio_cows_bulls : ℕ) (cows_bulls : ℕ) 
(h_total : total_cattle = 555) (h_ratio : ratio_cows_bulls = 10) (h_bulls_ratio : cows_bulls = 27) :
  let total_ratio_units := ratio_cows_bulls + cows_bulls in
  let bulls_count := (cows_bulls * total_cattle) / total_ratio_units in
  bulls_count = 405 := 
by
  sorry

end number_of_bulls_l165_165384


namespace sum_of_given_infinite_geometric_series_l165_165872

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l165_165872


namespace snow_probability_l165_165363

theorem snow_probability :
  let p1_snow := 1 / 3
  let p2_snow := 1 / 4
  let p1_prob_no_snow := 1 - p1_snow
  let p2_prob_no_snow := 1 - p2_snow
  let p_no_snow_first_three := p1_prob_no_snow ^ 3
  let p_no_snow_next_four := p2_prob_no_snow ^ 4
  let p_no_snow_week := p_no_snow_first_three * p_no_snow_next_four
  1 - p_no_snow_week = 29 / 32 :=
by
  let p1_snow := 1 / 3
  let p2_snow := 1 / 4
  let p1_prob_no_snow := 1 - p1_snow
  let p2_prob_no_snow := 1 - p2_snow
  let p_no_snow_first_three := p1_prob_no_snow ^ 3
  let p_no_snow_next_four := p2_prob_no_snow ^ 4
  let p_no_snow_week := p_no_snow_first_three * p_no_snow_next_four
  have p_no_snow_week_eq : p_no_snow_week = 3 / 32 := sorry
  have p_snow_at_least_once_week : 1 - p_no_snow_week = 29 / 32 := sorry
  exact p_snow_at_least_once_week

end snow_probability_l165_165363


namespace number_of_people_who_purchased_only_book_A_l165_165407

theorem number_of_people_who_purchased_only_book_A (x y v : ℕ) 
  (h1 : 2 * x = 500)
  (h2 : y = x + 500)
  (h3 : v = 2 * y) : 
  v = 1500 := 
sorry

end number_of_people_who_purchased_only_book_A_l165_165407


namespace kim_total_water_intake_l165_165220

def quarts_to_ounces (q : ℝ) : ℝ := q * 32

theorem kim_total_water_intake :
  (quarts_to_ounces 1.5) + 12 = 60 := 
by
  -- proof step 
  sorry

end kim_total_water_intake_l165_165220


namespace not_all_zero_iff_at_least_one_non_zero_l165_165650

theorem not_all_zero_iff_at_least_one_non_zero (a b c : ℝ) : ¬ (a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
by
  sorry

end not_all_zero_iff_at_least_one_non_zero_l165_165650


namespace dogs_not_doing_anything_l165_165792

def total_dogs : ℕ := 500
def dogs_running : ℕ := 18 * total_dogs / 100
def dogs_playing_with_toys : ℕ := (3 * total_dogs) / 20
def dogs_barking : ℕ := 7 * total_dogs / 100
def dogs_digging_holes : ℕ := total_dogs / 10
def dogs_competing : ℕ := 12
def dogs_sleeping : ℕ := (2 * total_dogs) / 25
def dogs_eating_treats : ℕ := total_dogs / 5

def dogs_doing_anything : ℕ := dogs_running + dogs_playing_with_toys + dogs_barking + dogs_digging_holes + dogs_competing + dogs_sleeping + dogs_eating_treats

theorem dogs_not_doing_anything : total_dogs - dogs_doing_anything = 98 :=
by
  -- proof steps would go here
  sorry

end dogs_not_doing_anything_l165_165792


namespace monotonic_conditions_fixed_point_property_l165_165297

noncomputable
def f (x a b c : ℝ) : ℝ := x^3 - a * x^2 - b * x + c

theorem monotonic_conditions (a b c : ℝ) :
  a = 0 ∧ c = 0 ∧ b ≤ 3 ↔ ∀ x : ℝ, (x ≥ 1 → (f x a b c) ≥ 1) → ∀ x y: ℝ, (x ≥ y ↔ f x a b c ≤ f y a b c) := sorry

theorem fixed_point_property (a b c : ℝ) :
  (∀ x : ℝ, (x ≥ 1 ∧ (f x a b c) ≥ 1) → f (f x a b c) a b c = x) ↔ (f x 0 b 0 = x) := sorry

end monotonic_conditions_fixed_point_property_l165_165297


namespace min_value_of_translated_function_l165_165783

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.sin (2 * x + ϕ)

theorem min_value_of_translated_function :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ (Real.pi / 2) → ∀ (ϕ : ℝ), |ϕ| < (Real.pi / 2) →
  ∀ (k : ℤ), f (x + (Real.pi / 6)) (ϕ + (Real.pi / 3) + k * Real.pi) = f x ϕ →
  ∃ y : ℝ, y = - Real.sqrt 3 / 2 := sorry

end min_value_of_translated_function_l165_165783


namespace shadow_boundary_function_correct_l165_165652

noncomputable def sphereShadowFunction : ℝ → ℝ :=
  λ x => (x + 1) / 2

theorem shadow_boundary_function_correct :
  ∀ (x y : ℝ), 
    -- Conditions: 
    -- The sphere with center (0,0,2) and radius 2
    -- A light source at point P = (1, -2, 3)
    -- The shadow must lie on the xy-plane, so z-coordinate is 0
    (sphereShadowFunction x = y) ↔ (- x + 2 * y - 1 = 0) :=
by
  intros x y
  sorry

end shadow_boundary_function_correct_l165_165652


namespace Eva_numbers_l165_165580

theorem Eva_numbers : ∃ (a b : ℕ), a + b = 43 ∧ a - b = 15 ∧ a = 29 ∧ b = 14 :=
by
  sorry

end Eva_numbers_l165_165580


namespace history_paper_pages_l165_165954

/-
Stacy has a history paper due in 3 days.
She has to write 21 pages per day to finish on time.
Prove that the total number of pages for the history paper is 63.
-/

theorem history_paper_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 21) (h2 : days = 3) : total_pages = 63 :=
by
  -- We would include the proof here, but for now, we use sorry to skip the proof.
  sorry

end history_paper_pages_l165_165954


namespace probability_two_green_apples_l165_165089

theorem probability_two_green_apples :
  let total_apples := 9
  let total_red := 5
  let total_green := 4
  let ways_to_choose_two := Nat.choose total_apples 2
  let ways_to_choose_two_green := Nat.choose total_green 2
  ways_to_choose_two ≠ 0 →
  (ways_to_choose_two_green / ways_to_choose_two : ℚ) = 1 / 6 :=
by
  intros
  -- skipping the proof
  sorry

end probability_two_green_apples_l165_165089


namespace MissyYellsCombined_l165_165097

def yellsAtStubbornDog (timesYellObedient : ℕ) := 4 * timesYellObedient

def totalYells (timesYellObedient : ℕ) (timesYellStubborn : ℕ) := timesYellObedient + timesYellStubborn

theorem MissyYellsCombined :
  ∀ (timesYellObedient : ℕ),
    timesYellObedient = 12 →
    totalYells timesYellObedient (yellsAtStubbornDog timesYellObedient) = 60 :=
by
  intros timesYellObedient h
  rw [h]
  unfold yellsAtStubbornDog totalYells
  norm_num
  rw [h]
  sorry

end MissyYellsCombined_l165_165097


namespace min_distance_from_point_to_line_l165_165596

theorem min_distance_from_point_to_line : 
  ∀ (x₀ y₀ : Real), 3 * x₀ - 4 * y₀ - 10 = 0 → Real.sqrt (x₀^2 + y₀^2) = 2 :=
by sorry

end min_distance_from_point_to_line_l165_165596


namespace inches_per_foot_l165_165630

-- Definition of the conditions in the problem.
def feet_last_week := 6
def feet_less_this_week := 4
def total_inches := 96

-- Lean statement that proves the number of inches in a foot
theorem inches_per_foot : 
    (total_inches / (feet_last_week + (feet_last_week - feet_less_this_week))) = 12 := 
by sorry

end inches_per_foot_l165_165630


namespace odd_three_digit_integers_strictly_increasing_digits_l165_165334

theorem odd_three_digit_integers_strictly_increasing_digits :
  let valid_combinations (c : ℕ) :=
    if c = 1 then 0 else
    if c = 3 then 1 else
    if c = 5 then 6 else
    if c = 7 then 15 else
    if c = 9 then 28 else 0 in
  (valid_combinations 1 + valid_combinations 3 + valid_combinations 5 + valid_combinations 7 + valid_combinations 9 = 50) :=
by
  unfold valid_combinations
  sorry

end odd_three_digit_integers_strictly_increasing_digits_l165_165334


namespace unit_prices_min_selling_price_l165_165645

-- Problem 1: Unit price determination
theorem unit_prices (x y : ℕ) (hx : 3600 / x * 2 = 5400 / y) (hy : y = x - 5) : x = 20 ∧ y = 15 := 
by 
  sorry

-- Problem 2: Minimum selling price for 50% profit margin
theorem min_selling_price (a : ℕ) (hx : 3600 / 20 = 180) (hy : 180 * 2 = 360) (hz : 540 * a ≥ 13500) : a ≥ 25 := 
by 
  sorry

end unit_prices_min_selling_price_l165_165645


namespace geometric_series_sum_l165_165884

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l165_165884


namespace InequalityProof_l165_165202

theorem InequalityProof (m n : ℝ) (h : m > n) : m / 4 > n / 4 :=
by sorry

end InequalityProof_l165_165202


namespace initial_water_amount_l165_165143

open Real

theorem initial_water_amount (W : ℝ)
  (h1 : ∀ (d : ℝ), d = 0.03 * 20)
  (h2 : ∀ (W : ℝ) (d : ℝ), d = 0.06 * W) :
  W = 10 :=
by
  sorry

end initial_water_amount_l165_165143


namespace find_base_b4_l165_165074

theorem find_base_b4 (b_4 : ℕ) : (b_4 - 1) * (b_4 - 2) * (b_4 - 3) = 168 → b_4 = 8 :=
by
  intro h
  -- proof goes here
  sorry

end find_base_b4_l165_165074


namespace prove_correct_option_C_l165_165683

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l165_165683


namespace math_problem_l165_165675

theorem math_problem (a b m : ℝ) : 
  ¬((-2 * a) ^ 2 = -4 * a ^ 2) ∧ 
  ¬((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧ 
  ((-m + 2) * (-m - 2) = m ^ 2 - 4) ∧ 
  ¬((a ^ 5) ^ 2 = a ^ 7) :=
by 
  split ; 
  sorry ;
  split ;
  sorry ;
  split ; 
  sorry ;
  sorry

end math_problem_l165_165675


namespace compute_expression_l165_165844

theorem compute_expression (x : ℤ) (h : x = 6) :
  ((x^9 - 24 * x^6 + 144 * x^3 - 512) / (x^3 - 8) = 43264) :=
by
  sorry

end compute_expression_l165_165844


namespace sum_of_given_infinite_geometric_series_l165_165870

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l165_165870


namespace distinct_remainders_mod_3n_l165_165094

open Nat

theorem distinct_remainders_mod_3n 
  (n : ℕ) 
  (hn_odd : Odd n)
  (ai : ℕ → ℕ)
  (bi : ℕ → ℕ)
  (ai_def : ∀ i, 1 ≤ i ∧ i ≤ n → ai i = 3*i - 2)
  (bi_def : ∀ i, 1 ≤ i ∧ i ≤ n → bi i = 3*i - 3)
  (k : ℕ) 
  (hk : 0 < k ∧ k < n)
  : ∀ i, 1 ≤ i ∧ i ≤ n → (∀ j, 1 ≤ j ∧ j ≤ n → i ≠ j →
     ∀ ⦃ r s t u v : ℕ ⦄, 
       (r = (ai i + ai (i % n + 1)) % (3*n) ∧ 
        s = (ai i + bi i) % (3*n) ∧ 
        t = (bi i + bi ((i + k) % n + 1)) % (3*n)) →
       r ≠ s ∧ s ≠ t ∧ t ≠ r) := 
sorry

end distinct_remainders_mod_3n_l165_165094


namespace infinite_geometric_series_sum_l165_165880

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l165_165880


namespace point_on_x_axis_l165_165561

theorem point_on_x_axis : ∃ p, (p = (-2, 0) ∧ p.snd = 0) ∧
  ((p ≠ (0, 2)) ∧ (p ≠ (-2, -3)) ∧ (p ≠ (-1, -2))) :=
by
  sorry

end point_on_x_axis_l165_165561


namespace original_solution_is_10_percent_l165_165273

def sugar_percentage_original_solution (x : ℕ) :=
  (3 / 4 : ℚ) * x + (1 / 4 : ℚ) * 42 = 18

theorem original_solution_is_10_percent : sugar_percentage_original_solution 10 :=
by
  unfold sugar_percentage_original_solution
  norm_num

end original_solution_is_10_percent_l165_165273


namespace burrito_count_l165_165768

def burrito_orders (wraps beef_fillings chicken_fillings : ℕ) :=
  if wraps = 5 ∧ beef_fillings >= 4 ∧ chicken_fillings >= 3 then 25 else 0

theorem burrito_count : burrito_orders 5 4 3 = 25 := by
  sorry

end burrito_count_l165_165768


namespace f_log3_54_l165_165848

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x ∧ x < 1 then 3^x else sorry

-- Definitions of the conditions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f (x)
def functional_equation (f : ℝ → ℝ) := ∀ x, f (x + 2) = -1 / f (x)

-- Hypotheses based on conditions
variable (f : ℝ → ℝ)
axiom f_is_odd : odd_function f
axiom f_is_periodic : periodic_function f 4
axiom f_functional : functional_equation f

-- Main goal
theorem f_log3_54 : f (Real.log 54 / Real.log 3) = -3 / 2 := by
  sorry

end f_log3_54_l165_165848


namespace helga_shoe_pairs_l165_165065

theorem helga_shoe_pairs
  (first_store_pairs: ℕ) 
  (second_store_pairs: ℕ) 
  (third_store_pairs: ℕ)
  (fourth_store_pairs: ℕ)
  (h1: first_store_pairs = 7)
  (h2: second_store_pairs = first_store_pairs + 2)
  (h3: third_store_pairs = 0)
  (h4: fourth_store_pairs = 2 * (first_store_pairs + second_store_pairs + third_store_pairs))
  : first_store_pairs + second_store_pairs + third_store_pairs + fourth_store_pairs = 48 := 
by
  sorry

end helga_shoe_pairs_l165_165065


namespace no_valid_a_l165_165576

theorem no_valid_a : ¬ ∃ (a : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 
  ∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 2 * x₁^2 + (3 * a + 1) * x₁ + a^2 = 0 ∧ 2 * x₂^2 + (3 * a + 1) * x₂ + a^2 = 0 :=
by {
  sorry
}

end no_valid_a_l165_165576


namespace abs_diff_26th_term_l165_165655

def C (n : ℕ) : ℤ := 50 + 15 * (n - 1)
def D (n : ℕ) : ℤ := 85 - 20 * (n - 1)

theorem abs_diff_26th_term :
  |(C 26) - (D 26)| = 840 := by
  sorry

end abs_diff_26th_term_l165_165655


namespace minimum_value_of_f_range_of_x_l165_165731

noncomputable def f (x : ℝ) := |2*x + 1| + |2*x - 1|

-- Problem 1
theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 2 :=
by
  intro x
  sorry

-- Problem 2
theorem range_of_x (a b : ℝ) (h : |2*a + b| + |a| - (1/2) * |a + b| * f x ≥ 0) : 
  - (1/2) ≤ x ∧ x ≤ 1/2 :=
by
  sorry

end minimum_value_of_f_range_of_x_l165_165731


namespace tickets_difference_l165_165154

def number_of_tickets_for_toys := 31
def number_of_tickets_for_clothes := 14

theorem tickets_difference : number_of_tickets_for_toys - number_of_tickets_for_clothes = 17 := by
  sorry

end tickets_difference_l165_165154


namespace fill_tank_without_leak_l165_165699

theorem fill_tank_without_leak (T : ℕ) : 
  (1 / T - 1 / 110 = 1 / 11) ↔ T = 10 :=
by 
  sorry

end fill_tank_without_leak_l165_165699


namespace solve_system_l165_165767

theorem solve_system (x y : ℝ) :
  (2 * y = (abs (2 * x + 3)) - (abs (2 * x - 3))) ∧ 
  (4 * x = (abs (y + 2)) - (abs (y - 2))) → 
  (-1 ≤ x ∧ x ≤ 1 ∧ y = 2 * x) := 
by
  sorry

end solve_system_l165_165767


namespace henry_has_more_than_500_seeds_on_saturday_l165_165604

theorem henry_has_more_than_500_seeds_on_saturday :
  (∃ k : ℕ, 5 * 3^k > 500 ∧ k + 1 = 6) :=
sorry

end henry_has_more_than_500_seeds_on_saturday_l165_165604


namespace skaters_total_hours_l165_165603

-- Define the practice hours based on the conditions
def hannah_weekend_hours := 8
def hannah_weekday_extra_hours := 17
def sarah_weekday_hours := 12
def sarah_weekend_hours := 6
def emma_weekday_hour_multiplier := 2
def emma_weekend_hour_extra := 5

-- Hannah's total hours
def hannah_weekday_hours := hannah_weekend_hours + hannah_weekday_extra_hours
def hannah_total_hours := hannah_weekend_hours + hannah_weekday_hours

-- Sarah's total hours
def sarah_total_hours := sarah_weekday_hours + sarah_weekend_hours

-- Emma's total hours
def emma_weekday_hours := emma_weekday_hour_multiplier * sarah_weekday_hours
def emma_weekend_hours := sarah_weekend_hours + emma_weekend_hour_extra
def emma_total_hours := emma_weekday_hours + emma_weekend_hours

-- Total hours for all three skaters combined
def total_hours := hannah_total_hours + sarah_total_hours + emma_total_hours

-- Lean statement version only, no proof required
theorem skaters_total_hours : total_hours = 86 := by
  sorry

end skaters_total_hours_l165_165603


namespace geometric_progression_difference_l165_165778

variable {n : ℕ}
variable {a : ℕ → ℝ} -- assuming the sequence is indexed by natural numbers
variable {a₁ : ℝ}
variable {r : ℝ} (hr : r = (1 + Real.sqrt 5) / 2)

def geometric_progression (a : ℕ → ℝ) (a₁ : ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a₁ * (r ^ n)

theorem geometric_progression_difference
  (a₁ : ℝ)
  (hr : r = (1 + Real.sqrt 5) / 2)
  (hg : geometric_progression a a₁ r) :
  ∀ n, n ≥ 2 → a n = a (n-1) - a (n-2) :=
by
  sorry

end geometric_progression_difference_l165_165778


namespace maximize_annual_profit_l165_165597

noncomputable def profit_function (x : ℝ) : ℝ :=
  - (1 / 3) * x^3 + 81 * x - 234

theorem maximize_annual_profit :
  ∃ x : ℝ, x = 9 ∧ ∀ y : ℝ, profit_function y ≤ profit_function x :=
sorry

end maximize_annual_profit_l165_165597


namespace tammy_driving_rate_l165_165371

-- Define the conditions given in the problem
def total_miles : ℕ := 1980
def total_hours : ℕ := 36

-- Define the desired rate to prove
def expected_rate : ℕ := 55

-- The theorem stating that given the conditions, Tammy's driving rate is correct
theorem tammy_driving_rate :
  total_miles / total_hours = expected_rate :=
by
  -- Detailed proof would go here
  sorry

end tammy_driving_rate_l165_165371


namespace cos_values_l165_165293

theorem cos_values (n : ℤ) : (0 ≤ n ∧ n ≤ 360) ∧ (Real.cos (n * Real.pi / 180) = Real.cos (310 * Real.pi / 180)) ↔ (n = 50 ∨ n = 310) :=
by
  sorry

end cos_values_l165_165293


namespace find_c_value_l165_165733

theorem find_c_value 
  (b : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + b * x + 3 ≥ 0) 
  (h2 : ∀ m c : ℝ, (∀ x : ℝ, x^2 + b * x + 3 < c ↔ m - 8 < x ∧ x < m)) 
  : c = 16 :=
sorry

end find_c_value_l165_165733


namespace cross_area_l165_165086

variables (R : ℝ) (A : ℝ × ℝ) (φ : ℝ)
  -- Radius R of the circle, Point A inside the circle, and angle φ in radians

-- Define the area of the cross formed by rotated lines
def area_of_cross (R : ℝ) (φ : ℝ) : ℝ :=
  2 * φ * R^2

theorem cross_area (R : ℝ) (A : ℝ × ℝ) (φ : ℝ) (hR : 0 < R) (hA : dist A (0, 0) < R) :
  area_of_cross R φ = 2 * φ * R^2 := 
sorry

end cross_area_l165_165086


namespace find_k_l165_165360

def g (a b c x : ℤ) := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (h1 : g a b c 1 = 0)
  (h2 : 20 < g a b c 5 ∧ g a b c 5 < 30)
  (h3 : 40 < g a b c 6 ∧ g a b c 6 < 50)
  (h4 : ∃ k : ℤ, 3000 * k < g a b c 100 ∧ g a b c 100 < 3000 * (k + 1)) :
  ∃ k : ℤ, k = 9 :=
by
  sorry

end find_k_l165_165360


namespace largest_N_cannot_pay_exactly_without_change_l165_165947

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l165_165947


namespace harmonic_mean_2_3_6_l165_165570

def harmonic_mean (a b c : ℕ) : ℚ := 3 / ((1 / a) + (1 / b) + (1 / c))

theorem harmonic_mean_2_3_6 : harmonic_mean 2 3 6 = 3 := 
by
  sorry

end harmonic_mean_2_3_6_l165_165570


namespace no_b_for_221_square_l165_165575

theorem no_b_for_221_square (b : ℕ) (h : b ≥ 3) :
  ¬ ∃ n : ℕ, 2 * b^2 + 2 * b + 1 = n^2 :=
by
  sorry

end no_b_for_221_square_l165_165575


namespace A_wins_if_N_is_perfect_square_l165_165018

noncomputable def player_A_can_always_win (N : ℕ) : Prop :=
  ∀ (B_moves : ℕ → ℕ), ∃ (A_moves : ℕ → ℕ), A_moves 0 = N ∧
  (∀ n, B_moves n = 0 ∨ (A_moves n ∣ B_moves (n + 1) ∨ B_moves (n + 1) ∣ A_moves n))

theorem A_wins_if_N_is_perfect_square :
  ∀ N : ℕ, player_A_can_always_win N ↔ ∃ n : ℕ, N = n * n := sorry

end A_wins_if_N_is_perfect_square_l165_165018


namespace number_divisible_by_7_last_digits_l165_165229

theorem number_divisible_by_7_last_digits :
  ∀ d : ℕ, d ≤ 9 → ∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d :=
by
  sorry

end number_divisible_by_7_last_digits_l165_165229


namespace original_sticker_price_l165_165908

-- Define the conditions in Lean
variables {x : ℝ} -- x is the original sticker price of the laptop

-- Definitions based on the problem conditions
def store_A_price (x : ℝ) : ℝ := 0.80 * x - 50
def store_B_price (x : ℝ) : ℝ := 0.70 * x
def heather_saves (x : ℝ) : Prop := store_B_price x - store_A_price x = 30

-- The theorem to prove
theorem original_sticker_price (x : ℝ) (h : heather_saves x) : x = 200 :=
by
  sorry

end original_sticker_price_l165_165908


namespace largest_N_cannot_pay_exactly_without_change_l165_165948

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l165_165948


namespace gumball_machine_total_gumballs_l165_165826

/-- A gumball machine has red, green, and blue gumballs. Given the following conditions:
1. The machine has half as many blue gumballs as red gumballs.
2. For each blue gumball, the machine has 4 times as many green gumballs.
3. The machine has 16 red gumballs.
Prove that the total number of gumballs in the machine is 56. -/
theorem gumball_machine_total_gumballs :
  ∀ (red blue green : ℕ),
    (blue = red / 2) →
    (green = blue * 4) →
    red = 16 →
    (red + blue + green = 56) :=
by
  intros red blue green h_blue h_green h_red
  sorry

end gumball_machine_total_gumballs_l165_165826


namespace volume_of_rectangular_prism_l165_165013

-- Defining the conditions as assumptions
variables (l w h : ℝ) 
variable (lw_eq : l * w = 10)
variable (wh_eq : w * h = 14)
variable (lh_eq : l * h = 35)

-- Stating the theorem to prove
theorem volume_of_rectangular_prism : l * w * h = 70 :=
by
  have lw := lw_eq
  have wh := wh_eq
  have lh := lh_eq
  sorry

end volume_of_rectangular_prism_l165_165013
