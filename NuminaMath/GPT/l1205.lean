import Mathlib

namespace NUMINAMATH_GPT_exists_integer_div_15_sqrt_range_l1205_120534

theorem exists_integer_div_15_sqrt_range :
  ∃ n : ℕ, (25^2 ≤ n ∧ n ≤ 26^2) ∧ (n % 15 = 0) :=
by
  sorry

end NUMINAMATH_GPT_exists_integer_div_15_sqrt_range_l1205_120534


namespace NUMINAMATH_GPT_factor_count_l1205_120516

theorem factor_count (x : ℤ) : 
  (x^12 - x^3) = x^3 * (x - 1) * (x^2 + x + 1) * (x^6 + x^3 + 1) -> 4 = 4 :=
by
  sorry

end NUMINAMATH_GPT_factor_count_l1205_120516


namespace NUMINAMATH_GPT_max_books_borrowed_l1205_120547

theorem max_books_borrowed 
  (num_students : ℕ)
  (num_no_books : ℕ)
  (num_one_book : ℕ)
  (num_two_books : ℕ)
  (average_books : ℕ)
  (h_num_students : num_students = 32)
  (h_num_no_books : num_no_books = 2)
  (h_num_one_book : num_one_book = 12)
  (h_num_two_books : num_two_books = 10)
  (h_average_books : average_books = 2)
  : ∃ max_books : ℕ, max_books = 11 := 
by
  sorry

end NUMINAMATH_GPT_max_books_borrowed_l1205_120547


namespace NUMINAMATH_GPT_sum_of_rationals_l1205_120502

theorem sum_of_rationals (r1 r2 : ℚ) : ∃ r : ℚ, r = r1 + r2 :=
sorry

end NUMINAMATH_GPT_sum_of_rationals_l1205_120502


namespace NUMINAMATH_GPT_sequence_is_geometric_l1205_120584

theorem sequence_is_geometric {a : ℝ} (h : a ≠ 0) (S : ℕ → ℝ) (H : ∀ n, S n = a^n - 1) 
: ∃ r, ∀ n, (n ≥ 1) → S n - S (n-1) = r * (S (n-1) - S (n-2)) :=
sorry

end NUMINAMATH_GPT_sequence_is_geometric_l1205_120584


namespace NUMINAMATH_GPT_distance_traveled_l1205_120532

theorem distance_traveled 
    (P_b : ℕ) (P_f : ℕ) (R_b : ℕ) (R_f : ℕ)
    (h1 : P_b = 9)
    (h2 : P_f = 7)
    (h3 : R_f = R_b + 10) 
    (h4 : R_b * P_b = R_f * P_f) :
    R_b * P_b = 315 :=
by
  sorry

end NUMINAMATH_GPT_distance_traveled_l1205_120532


namespace NUMINAMATH_GPT_correct_average_of_corrected_number_l1205_120535

theorem correct_average_of_corrected_number (num_list : List ℤ) (wrong_num correct_num : ℤ) (n : ℕ)
  (hn : n = 10)
  (haverage : (num_list.sum / n) = 5)
  (hwrong : wrong_num = 26)
  (hcorrect : correct_num = 36)
  (hnum_list_sum : num_list.sum + correct_num - wrong_num = num_list.sum + 10) :
  (num_list.sum + 10) / n = 6 :=
by
  sorry

end NUMINAMATH_GPT_correct_average_of_corrected_number_l1205_120535


namespace NUMINAMATH_GPT_minimize_fraction_l1205_120527

theorem minimize_fraction (n : ℕ) (h : 0 < n) : 
  (n = 9) → (∀ m : ℕ, 0 < m → (n = m) → (3 * m + 27 / m ≥ 6)) := sorry

end NUMINAMATH_GPT_minimize_fraction_l1205_120527


namespace NUMINAMATH_GPT_perimeter_original_square_l1205_120565

theorem perimeter_original_square (s : ℝ) (h1 : (3 / 4) * s^2 = 48) : 4 * s = 32 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_original_square_l1205_120565


namespace NUMINAMATH_GPT_middle_odd_number_is_26_l1205_120551

theorem middle_odd_number_is_26 (x : ℤ) 
  (h : (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 130) : x = 26 := 
by 
  sorry

end NUMINAMATH_GPT_middle_odd_number_is_26_l1205_120551


namespace NUMINAMATH_GPT_current_speed_is_one_l1205_120556

noncomputable def motorboat_rate_of_current (b h t : ℝ) : ℝ :=
  let eq1 := (b + 1 - h) * 4
  let eq2 := (b - 1 + t) * 6
  if eq1 = 24 ∧ eq2 = 24 then 1 else sorry

theorem current_speed_is_one (b h t : ℝ) : motorboat_rate_of_current b h t = 1 :=
by
  sorry

end NUMINAMATH_GPT_current_speed_is_one_l1205_120556


namespace NUMINAMATH_GPT_volume_of_polyhedron_l1205_120548

theorem volume_of_polyhedron (V : ℝ) (hV : 0 ≤ V) :
  ∃ P : ℝ, P = V / 6 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_polyhedron_l1205_120548


namespace NUMINAMATH_GPT_total_sand_volume_l1205_120558

noncomputable def cone_diameter : ℝ := 10
noncomputable def cone_radius : ℝ := cone_diameter / 2
noncomputable def cone_height : ℝ := 0.75 * cone_diameter
noncomputable def cylinder_height : ℝ := 0.5 * cone_diameter
noncomputable def total_volume : ℝ := (1 / 3 * Real.pi * cone_radius^2 * cone_height) + (Real.pi * cone_radius^2 * cylinder_height)

theorem total_sand_volume : total_volume = 187.5 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_total_sand_volume_l1205_120558


namespace NUMINAMATH_GPT_income_to_expenditure_ratio_l1205_120562

theorem income_to_expenditure_ratio (I E S : ℕ) (hI : I = 15000) (hS : S = 7000) (hSavings : S = I - E) :
  I / E = 15 / 8 := by
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_income_to_expenditure_ratio_l1205_120562


namespace NUMINAMATH_GPT_ones_digit_of_p_is_3_l1205_120533

theorem ones_digit_of_p_is_3 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h_seq : q = p + 8 ∧ r = p + 16 ∧ s = p + 24) (p_gt_5 : p > 5) : p % 10 = 3 :=
sorry

end NUMINAMATH_GPT_ones_digit_of_p_is_3_l1205_120533


namespace NUMINAMATH_GPT_reciprocal_of_8_l1205_120523

theorem reciprocal_of_8:
  (1 : ℝ) / 8 = (1 / 8 : ℝ) := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_8_l1205_120523


namespace NUMINAMATH_GPT_proof_method_characterization_l1205_120563

-- Definitions of each method
def synthetic_method := "proceeds from cause to effect, in a forward manner"
def analytic_method := "seeks the cause from the effect, working backwards"
def proof_by_contradiction := "assumes the negation of the proposition to be true, and derives a contradiction"
def mathematical_induction := "base case and inductive step: which shows that P holds for all natural numbers"

-- Main theorem to prove
theorem proof_method_characterization :
  (analytic_method == "seeks the cause from the effect, working backwards") :=
by
  sorry

end NUMINAMATH_GPT_proof_method_characterization_l1205_120563


namespace NUMINAMATH_GPT_rho_square_max_value_l1205_120510

variable {a b x y c : ℝ}
variable (ha_pos : a > 0) (hb_pos : b > 0)
variable (ha_ge_b : a ≥ b)
variable (hx_range : 0 ≤ x ∧ x < a)
variable (hy_range : 0 ≤ y ∧ y < b)
variable (h_eq1 : a^2 + y^2 = b^2 + x^2)
variable (h_eq2 : b^2 + x^2 = (a - x)^2 + (b - y)^2 + c^2)

theorem rho_square_max_value : (a / b) ^ 2 ≤ 4 / 3 :=
sorry

end NUMINAMATH_GPT_rho_square_max_value_l1205_120510


namespace NUMINAMATH_GPT_A_B_symmetric_x_axis_l1205_120504

-- Definitions of points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (-2, -3)

-- Theorem stating the symmetry relationship between points A and B with respect to the x-axis
theorem A_B_symmetric_x_axis (xA yA xB yB : ℝ) (hA : A = (xA, yA)) (hB : B = (xB, yB)) :
  xA = xB ∧ yA = -yB := by
  sorry

end NUMINAMATH_GPT_A_B_symmetric_x_axis_l1205_120504


namespace NUMINAMATH_GPT_correct_statement_l1205_120577

variables {Line Plane : Type}
variable (a b c : Line)
variable (M N : Plane)

/- Definitions for the conditions -/
def lies_on_plane (l : Line) (p : Plane) : Prop := sorry
def intersection (p1 p2 : Plane) : Line := sorry
def parallel (l1 l2 : Line) : Prop := sorry

/- Conditions -/
axiom h1 : lies_on_plane a M
axiom h2 : lies_on_plane b N
axiom h3 : intersection M N = c

/- The correct statement to be proved -/
theorem correct_statement : parallel a b → parallel a c :=
by sorry

end NUMINAMATH_GPT_correct_statement_l1205_120577


namespace NUMINAMATH_GPT_problem_1_problem_2_l1205_120552

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2
noncomputable def g (x : ℝ) : ℝ := 1 / 2 + Real.sqrt 3 * Real.sin x * Real.cos x
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Proof Problem 1
theorem problem_1 (a : ℝ) (h_symmetry : ∃ k : ℤ, a = k * Real.pi / 2) : g (2 * a) = 1 / 2 := by
  sorry

-- Proof Problem 2
theorem problem_2 (x : ℝ) (h_range : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  ∃ y : ℝ, y = h x ∧ 1/2 ≤ y ∧ y ≤ 2 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1205_120552


namespace NUMINAMATH_GPT_Michael_selection_l1205_120550

theorem Michael_selection :
  (Nat.choose 8 3) * (Nat.choose 5 2) = 560 :=
by
  sorry

end NUMINAMATH_GPT_Michael_selection_l1205_120550


namespace NUMINAMATH_GPT_sum_first_10_terms_l1205_120519

def arithmetic_sequence (a d : Int) (n : Int) : Int :=
  a + (n - 1) * d

def arithmetic_sum (a d : Int) (n : Int) : Int :=
  (n : Int) * a + (n * (n - 1) / 2) * d

theorem sum_first_10_terms  
  (a d : Int)
  (h1 : (a + 3 * d)^2 = (a + 2 * d) * (a + 6 * d))
  (h2 : arithmetic_sum a d 8 = 32)
  : arithmetic_sum a d 10 = 60 :=
sorry

end NUMINAMATH_GPT_sum_first_10_terms_l1205_120519


namespace NUMINAMATH_GPT_simplify_fraction_l1205_120501

theorem simplify_fraction (d : ℤ) : (5 + 4 * d) / 9 - 3 = (4 * d - 22) / 9 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1205_120501


namespace NUMINAMATH_GPT_rick_gives_miguel_cards_l1205_120525

/-- Rick starts with 130 cards, keeps 15 cards for himself, gives 
12 cards each to 8 friends, and gives 3 cards each to his 2 sisters. 
We need to prove that Rick gives 13 cards to Miguel. --/
theorem rick_gives_miguel_cards :
  let initial_cards := 130
  let kept_cards := 15
  let friends := 8
  let cards_per_friend := 12
  let sisters := 2
  let cards_per_sister := 3
  initial_cards - kept_cards - (friends * cards_per_friend) - (sisters * cards_per_sister) = 13 :=
by
  sorry

end NUMINAMATH_GPT_rick_gives_miguel_cards_l1205_120525


namespace NUMINAMATH_GPT_find_z_l1205_120590

/- Definitions of angles and their relationships -/
def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

/- Given conditions -/
def ABC : ℝ := 75
def BAC : ℝ := 55
def BCA : ℝ := 180 - ABC - BAC  -- This follows from the angle sum property of triangle ABC
def DCE : ℝ := BCA
def CDE : ℝ := 90

/- Prove z given the above conditions -/
theorem find_z : ∃ (z : ℝ), z = 90 - DCE := by
  use 40
  sorry

end NUMINAMATH_GPT_find_z_l1205_120590


namespace NUMINAMATH_GPT_sandwiches_provided_l1205_120520

theorem sandwiches_provided (original_count sold_out : ℕ) (h1 : original_count = 9) (h2 : sold_out = 5) : (original_count - sold_out = 4) :=
by
  sorry

end NUMINAMATH_GPT_sandwiches_provided_l1205_120520


namespace NUMINAMATH_GPT_arithmetic_expression_eval_l1205_120561

theorem arithmetic_expression_eval : (10 - 9 + 8) * 7 + 6 - 5 * (4 - 3 + 2) - 1 = 53 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_eval_l1205_120561


namespace NUMINAMATH_GPT_bus_trip_cost_l1205_120500

-- Problem Statement Definitions
def distance_AB : ℕ := 4500
def cost_per_kilometer_bus : ℚ := 0.20

-- Theorem Statement
theorem bus_trip_cost : distance_AB * cost_per_kilometer_bus = 900 := by
  sorry

end NUMINAMATH_GPT_bus_trip_cost_l1205_120500


namespace NUMINAMATH_GPT_measure_of_each_interior_angle_l1205_120530

theorem measure_of_each_interior_angle (n : ℕ) (hn : 3 ≤ n) : 
  ∃ angle : ℝ, angle = (n - 2) * 180 / n :=
by
  sorry

end NUMINAMATH_GPT_measure_of_each_interior_angle_l1205_120530


namespace NUMINAMATH_GPT_find_tangent_line_equation_l1205_120573

-- Define the curve as a function
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Define the point of tangency
def P : ℝ × ℝ := (-1, 3)

-- Define the slope of the tangent line at point P
def slope_at_P : ℝ := curve_derivative P.1

-- Define the expected equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 4 * x + y + 1 = 0

-- The theorem to prove that the tangent line at point P has the expected equation
theorem find_tangent_line_equation : 
  tangent_line P.1 (curve P.1) :=
  sorry

end NUMINAMATH_GPT_find_tangent_line_equation_l1205_120573


namespace NUMINAMATH_GPT_expected_value_is_correct_l1205_120531

-- Define the probabilities of heads and tails
def P_H := 2 / 5
def P_T := 3 / 5

-- Define the winnings for heads and the loss for tails
def W_H := 5
def L_T := -4

-- Calculate the expected value
def expected_value := P_H * W_H + P_T * L_T

-- Prove that the expected value is -2/5
theorem expected_value_is_correct : expected_value = -2 / 5 := by
  sorry

end NUMINAMATH_GPT_expected_value_is_correct_l1205_120531


namespace NUMINAMATH_GPT_find_expression_value_l1205_120507

def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

theorem find_expression_value (p q r s : ℝ) (h1 : g p q r s (-1) = 2) (h2 : g p q r s (-2) = -1) (h3 : g p q r s (1) = -2) :
  9 * p - 3 * q + 3 * r - s = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_expression_value_l1205_120507


namespace NUMINAMATH_GPT_original_salary_l1205_120557

theorem original_salary (S : ℝ) (h1 : S + 0.10 * S = 1.10 * S) (h2: 1.10 * S - 0.05 * (1.10 * S) = 1.10 * S * 0.95) (h3: 1.10 * S * 0.95 = 2090) : S = 2000 :=
sorry

end NUMINAMATH_GPT_original_salary_l1205_120557


namespace NUMINAMATH_GPT_range_of_a_l1205_120515

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (1 < x ∧ x < 4) ∧ (2 * x^2 - 9 * x + a < 0)) ↔ (a < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1205_120515


namespace NUMINAMATH_GPT_choir_members_number_l1205_120570

theorem choir_members_number
  (n : ℕ)
  (h1 : n % 12 = 10)
  (h2 : n % 14 = 12)
  (h3 : 300 ≤ n ∧ n ≤ 400) :
  n = 346 :=
sorry

end NUMINAMATH_GPT_choir_members_number_l1205_120570


namespace NUMINAMATH_GPT_find_a_find_cos_2C_l1205_120521

noncomputable def triangle_side_a (A B : Real) (b : Real) (cosA : Real) : Real := 
  3

theorem find_a (A : Real) (B : Real) (b : Real) (cosA : Real) 
  (h₁ : b = 3 * Real.sqrt 2) 
  (h₂ : cosA = Real.sqrt 6 / 3) 
  (h₃ : B = A + Real.pi / 2) : 
  triangle_side_a A B b cosA = 3 := by
  sorry

noncomputable def cos_2C (A B C a b : Real) (cosA sinC : Real) : Real :=
  7 / 9

theorem find_cos_2C (A : Real) (B : Real) (C : Real) (a : Real) (b : Real) (cosA : Real) (sinC: Real)
  (h₁ : b = 3 * Real.sqrt 2) 
  (h₂ : cosA = Real.sqrt 6 / 3)
  (h₃ : B = A + Real.pi /2)
  (h₄ : a = 3)
  (h₅ : sinC = 1 / 3) :
  cos_2C A B C a b cosA sinC = 7 / 9 := by
  sorry

end NUMINAMATH_GPT_find_a_find_cos_2C_l1205_120521


namespace NUMINAMATH_GPT_prove_proposition_false_l1205_120566

def proposition (a : ℝ) := ∃ x : ℝ, x^2 - 4*a*x + 3 < 0

theorem prove_proposition_false : proposition 0 = False :=
by
sorry

end NUMINAMATH_GPT_prove_proposition_false_l1205_120566


namespace NUMINAMATH_GPT_new_shape_perimeter_l1205_120588

-- Definitions based on conditions
def square_side : ℕ := 64 / 4
def is_tri_isosceles (a b c : ℕ) : Prop := a = b

-- Definition of given problem setup and perimeter calculation
theorem new_shape_perimeter
  (side : ℕ)
  (tri_side1 tri_side2 base : ℕ)
  (h_square_side : side = 64 / 4)
  (h_tri1 : tri_side1 = side)
  (h_tri2 : tri_side2 = side)
  (h_base : base = side) :
  (side * 5) = 80 :=
by
  sorry

end NUMINAMATH_GPT_new_shape_perimeter_l1205_120588


namespace NUMINAMATH_GPT_cats_count_l1205_120518

-- Definitions based on conditions
def heads_eqn (H C : ℕ) : Prop := H + C = 15
def legs_eqn (H C : ℕ) : Prop := 2 * H + 4 * C = 44

-- The main proof problem
theorem cats_count (H C : ℕ) (h1 : heads_eqn H C) (h2 : legs_eqn H C) : C = 7 :=
by
  sorry

end NUMINAMATH_GPT_cats_count_l1205_120518


namespace NUMINAMATH_GPT_vertex_of_parabola_l1205_120512

theorem vertex_of_parabola :
  ∀ x : ℝ, (x - 2) ^ 2 + 4 = (x - 2) ^ 2 + 4 → (2, 4) = (2, 4) :=
by
  intro x
  intro h
  -- We know that the vertex of y = (x - 2)^2 + 4 is at (2, 4)
  admit

end NUMINAMATH_GPT_vertex_of_parabola_l1205_120512


namespace NUMINAMATH_GPT_solve_sqrt_eq_l1205_120593

theorem solve_sqrt_eq (z : ℚ) (h : Real.sqrt (5 - 4 * z) = 10) : z = -95 / 4 := by
  sorry

end NUMINAMATH_GPT_solve_sqrt_eq_l1205_120593


namespace NUMINAMATH_GPT_line_slope_and_point_l1205_120567

noncomputable def line_equation (x : ℝ) (m b : ℝ) : ℝ := m * x + b

theorem line_slope_and_point (m b : ℝ) (x₀ y₀ : ℝ) (h₁ : m = -3) (h₂ : x₀ = 5) (h₃ : y₀ = 2) (h₄ : y₀ = line_equation x₀ m b) :
  m + b = 14 :=
by
  sorry

end NUMINAMATH_GPT_line_slope_and_point_l1205_120567


namespace NUMINAMATH_GPT_average_speed_of_train_l1205_120546

def ChicagoTime (t : String) : Prop := t = "5:00 PM"
def NewYorkTime (t : String) : Prop := t = "10:00 AM"
def TimeDifference (d : Nat) : Prop := d = 1
def Distance (d : Nat) : Prop := d = 480

theorem average_speed_of_train :
  ∀ (d t1 t2 diff : Nat), 
  Distance d → (NewYorkTime "10:00 AM") → (ChicagoTime "5:00 PM") → TimeDifference diff →
  (t2 = 5 ∧ t1 = (10 - diff)) →
  (d / (t2 - t1) = 60) :=
by
  intros d t1 t2 diff hD ht1 ht2 hDiff hTimes
  sorry

end NUMINAMATH_GPT_average_speed_of_train_l1205_120546


namespace NUMINAMATH_GPT_find_bloom_day_l1205_120587

def days := {d : Fin 7 // 1 ≤ d.val ∧ d.val ≤ 7}

def sunflowers_bloom (d : days) : Prop :=
¬ (d.val = 2 ∨ d.val = 4 ∨ d.val = 7)

def lilies_bloom (d : days) : Prop :=
¬ (d.val = 4 ∨ d.val = 6)

def magnolias_bloom (d : days) : Prop :=
¬ (d.val = 7)

def all_bloom_together (d : days) : Prop :=
sunflowers_bloom d ∧ lilies_bloom d ∧ magnolias_bloom d

def blooms_simultaneously (d : days) : Prop :=
∀ d1 d2 d3 : days, (d1 = d ∧ d2 = d ∧ d3 = d) →
(all_bloom_together d1 ∧ all_bloom_together d2 ∧ all_bloom_together d3)

theorem find_bloom_day :
  ∃ d : days, blooms_simultaneously d :=
sorry

end NUMINAMATH_GPT_find_bloom_day_l1205_120587


namespace NUMINAMATH_GPT_minimum_bag_count_l1205_120575

theorem minimum_bag_count (n a b : ℕ) (h1 : 7 * a + 11 * b = 77) (h2 : a + b = n) : n = 17 :=
by
  sorry

end NUMINAMATH_GPT_minimum_bag_count_l1205_120575


namespace NUMINAMATH_GPT_probability_of_at_least_one_boy_and_one_girl_is_correct_l1205_120539

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  (1 - ((1/2)^4 + (1/2)^4))

theorem probability_of_at_least_one_boy_and_one_girl_is_correct : 
  probability_at_least_one_boy_and_one_girl = 7/8 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_at_least_one_boy_and_one_girl_is_correct_l1205_120539


namespace NUMINAMATH_GPT_convert_to_rectangular_form_l1205_120503

theorem convert_to_rectangular_form :
  2 * Real.sqrt 3 * Complex.exp (13 * Real.pi * Complex.I / 6) = 3 + Complex.I * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_convert_to_rectangular_form_l1205_120503


namespace NUMINAMATH_GPT_bird_difference_l1205_120578

-- Variables representing given conditions
def num_migrating_families : Nat := 86
def num_remaining_families : Nat := 45
def avg_birds_per_migrating_family : Nat := 12
def avg_birds_per_remaining_family : Nat := 8

-- Definition to calculate total number of birds for migrating families
def total_birds_migrating : Nat := num_migrating_families * avg_birds_per_migrating_family

-- Definition to calculate total number of birds for remaining families
def total_birds_remaining : Nat := num_remaining_families * avg_birds_per_remaining_family

-- The statement that we need to prove
theorem bird_difference (h : total_birds_migrating - total_birds_remaining = 672) : 
  total_birds_migrating - total_birds_remaining = 672 := 
sorry

end NUMINAMATH_GPT_bird_difference_l1205_120578


namespace NUMINAMATH_GPT_prove_smallest_number_l1205_120594

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

lemma smallest_number_to_add (n : ℕ) (k : ℕ) (h: sum_of_digits n % k = r) : n % k = r →
  n % k = r → (k - r) = 7 :=
by
  sorry

theorem prove_smallest_number (n : ℕ) (k : ℕ) (r : ℕ) :
  (27452 % 9 = r) ∧ (9 - r = 7) :=
by
  sorry

end NUMINAMATH_GPT_prove_smallest_number_l1205_120594


namespace NUMINAMATH_GPT_decagon_diagonals_l1205_120559

-- Define the number of sides of a decagon
def n : ℕ := 10

-- Define the formula for the number of diagonals in an n-sided polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem decagon_diagonals : num_diagonals n = 35 := by
  sorry

end NUMINAMATH_GPT_decagon_diagonals_l1205_120559


namespace NUMINAMATH_GPT_total_spider_legs_l1205_120544

-- Definition of the number of spiders
def number_of_spiders : ℕ := 5

-- Definition of the number of legs per spider
def legs_per_spider : ℕ := 8

-- Theorem statement to prove the total number of spider legs
theorem total_spider_legs : number_of_spiders * legs_per_spider = 40 :=
by 
  -- We've planned to use 'sorry' to skip the proof
  sorry

end NUMINAMATH_GPT_total_spider_legs_l1205_120544


namespace NUMINAMATH_GPT_at_least_one_ge_two_l1205_120554

theorem at_least_one_ge_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  a + b + c ≥ 6 → (a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2) :=
by
  intros
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  sorry

end NUMINAMATH_GPT_at_least_one_ge_two_l1205_120554


namespace NUMINAMATH_GPT_age_of_student_who_left_l1205_120560

/-- 
The average student age of a class with 30 students is 10 years.
After one student leaves and the teacher (who is 41 years old) is included,
the new average age is 11 years. Prove that the student who left is 11 years old.
-/
theorem age_of_student_who_left (x : ℕ) (h1 : (30 * 10) = 300)
    (h2 : (300 - x + 41) / 30 = 11) : x = 11 :=
by 
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_age_of_student_who_left_l1205_120560


namespace NUMINAMATH_GPT_min_cubes_l1205_120579

-- Define the conditions
structure Cube := (x : ℕ) (y : ℕ) (z : ℕ)
def shares_face (c1 c2 : Cube) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ (c1.z = c2.z + 1 ∨ c1.z = c2.z - 1)) ∨
  (c1.x = c2.x ∧ c1.z = c2.z ∧ (c1.y = c2.y + 1 ∨ c1.y = c2.y - 1)) ∨
  (c1.y = c2.y ∧ c1.z = c2.z ∧ (c1.x = c2.x + 1 ∨ c1.x = c2.x - 1))

def front_view (cubes : List Cube) : Prop :=
  -- Representation of L-shape in xy-plane
  ∃ (c1 c2 c3 c4 c5 : Cube),
  cubes = [c1, c2, c3, c4, c5] ∧
  (c1.x = 0 ∧ c1.y = 0 ∧ c1.z = 0) ∧
  (c2.x = 1 ∧ c2.y = 0 ∧ c2.z = 0) ∧
  (c3.x = 2 ∧ c3.y = 0 ∧ c3.z = 0) ∧
  (c4.x = 2 ∧ c4.y = 1 ∧ c4.z = 0) ∧
  (c5.x = 1 ∧ c5.y = 2 ∧ c5.z = 0)

def side_view (cubes : List Cube) : Prop :=
  -- Representation of Z-shape in yz-plane
  ∃ (c1 c2 c3 c4 c5 : Cube),
  cubes = [c1, c2, c3, c4, c5] ∧
  (c1.x = 0 ∧ c1.y = 0 ∧ c1.z = 0) ∧
  (c2.x = 0 ∧ c2.y = 1 ∧ c2.z = 0) ∧
  (c3.x = 0 ∧ c3.y = 1 ∧ c3.z = 1) ∧
  (c4.x = 0 ∧ c4.y = 2 ∧ c4.z = 1) ∧
  (c5.x = 0 ∧ c5.y = 2 ∧ c5.z = 2)

-- Proof statement
theorem min_cubes (cubes : List Cube) (h1 : front_view cubes) (h2 : side_view cubes) : cubes.length = 5 :=
by sorry

end NUMINAMATH_GPT_min_cubes_l1205_120579


namespace NUMINAMATH_GPT_alcohol_mixture_l1205_120592

theorem alcohol_mixture (y : ℕ) :
  let x_vol := 200 -- milliliters
  let y_conc := 30 / 100 -- 30% alcohol
  let x_conc := 10 / 100 -- 10% alcohol
  let final_conc := 20 / 100 -- 20% target alcohol concentration
  let x_alcohol := x_vol * x_conc -- alcohol in x
  (x_alcohol + y * y_conc) / (x_vol + y) = final_conc ↔ y = 200 :=
by 
  sorry

end NUMINAMATH_GPT_alcohol_mixture_l1205_120592


namespace NUMINAMATH_GPT_relationship_M_N_l1205_120597

-- Define the sets M and N based on the conditions
def M : Set ℕ := {x | ∃ n : ℕ, x = 3^n}
def N : Set ℕ := {x | ∃ n : ℕ, x = 3 * n}

-- The statement to be proved
theorem relationship_M_N : ¬ (M ⊆ N) ∧ ¬ (N ⊆ M) :=
by
  sorry

end NUMINAMATH_GPT_relationship_M_N_l1205_120597


namespace NUMINAMATH_GPT_daisies_bought_l1205_120595

-- Definitions from the given conditions
def cost_per_flower : ℕ := 6
def num_roses : ℕ := 7
def total_spent : ℕ := 60

-- Proving the number of daisies Maria bought
theorem daisies_bought : ∃ (D : ℕ), D = 3 ∧ total_spent = num_roses * cost_per_flower + D * cost_per_flower :=
by
  sorry

end NUMINAMATH_GPT_daisies_bought_l1205_120595


namespace NUMINAMATH_GPT_beth_lost_red_marbles_l1205_120505

-- Definitions from conditions
def total_marbles : ℕ := 72
def marbles_per_color : ℕ := total_marbles / 3
variable (R : ℕ)  -- Number of red marbles Beth lost
def blue_marbles_lost : ℕ := 2 * R
def yellow_marbles_lost : ℕ := 3 * R
def marbles_left : ℕ := 42

-- Theorem we want to prove
theorem beth_lost_red_marbles (h : total_marbles - (R + blue_marbles_lost R + yellow_marbles_lost R) = marbles_left) :
  R = 5 :=
by
  sorry

end NUMINAMATH_GPT_beth_lost_red_marbles_l1205_120505


namespace NUMINAMATH_GPT_f_f_f_f_f_of_1_l1205_120564

def f (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem f_f_f_f_f_of_1 : f (f (f (f (f 1)))) = 4687 :=
by
  sorry

end NUMINAMATH_GPT_f_f_f_f_f_of_1_l1205_120564


namespace NUMINAMATH_GPT_volume_of_pyramid_l1205_120599

noncomputable def volume_of_regular_triangular_pyramid (h R : ℝ) : ℝ :=
  (h ^ 2 * (2 * R - h) * Real.sqrt 3) / 4

theorem volume_of_pyramid (h R : ℝ) : volume_of_regular_triangular_pyramid h R = (h ^ 2 * (2 * R - h) * Real.sqrt 3) / 4 :=
  by sorry

end NUMINAMATH_GPT_volume_of_pyramid_l1205_120599


namespace NUMINAMATH_GPT_z_is_233_percent_greater_than_w_l1205_120511

theorem z_is_233_percent_greater_than_w
  (w e x y z : ℝ)
  (h1 : w = 0.5 * e)
  (h2 : e = 0.4 * x)
  (h3 : x = 0.3 * y)
  (h4 : z = 0.2 * y) :
  z = 2.3333 * w :=
by
  sorry

end NUMINAMATH_GPT_z_is_233_percent_greater_than_w_l1205_120511


namespace NUMINAMATH_GPT_total_number_of_questions_l1205_120542

theorem total_number_of_questions (N : ℕ)
  (hp : 0.8 * N = (4 / 5 : ℝ) * N)
  (hv : 35 = 35)
  (hb : (N / 2 : ℕ) = 1 * (N.div 2))
  (ha : N - 7 = N - 7) : N = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_questions_l1205_120542


namespace NUMINAMATH_GPT_find_first_number_l1205_120538

variable (a : ℕ → ℤ)

axiom recurrence_rel : ∀ (n : ℕ), n ≥ 4 → a n = a (n - 1) + a (n - 2) + a (n - 3)
axiom a8_val : a 8 = 29
axiom a9_val : a 9 = 56
axiom a10_val : a 10 = 108

theorem find_first_number : a 1 = 32 :=
sorry

end NUMINAMATH_GPT_find_first_number_l1205_120538


namespace NUMINAMATH_GPT_quadratic_roots_sum_product_l1205_120540

theorem quadratic_roots_sum_product (m n : ℝ) (h1 : m / 2 = 10) (h2 : n / 2 = 24) : m + n = 68 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_sum_product_l1205_120540


namespace NUMINAMATH_GPT_number_of_buses_used_l1205_120582

-- Definitions based on the conditions
def total_students : ℕ := 360
def students_per_bus : ℕ := 45

-- The theorem we need to prove
theorem number_of_buses_used : total_students / students_per_bus = 8 := 
by sorry

end NUMINAMATH_GPT_number_of_buses_used_l1205_120582


namespace NUMINAMATH_GPT_monotonicity_f_geq_f_neg_l1205_120598

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem monotonicity (a : ℝ) :
  (a ≤ 0 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2) ∧
  (a > 0 →
    (∀ x1 x2 : ℝ, x1 > Real.log a → x2 > Real.log a → x1 ≤ x2 → f a x1 ≤ f a x2) ∧
    (∀ x1 x2 : ℝ, x1 < Real.log a → x2 < Real.log a → x1 ≤ x2 → f a x1 ≤ f a x2)) :=
by sorry

theorem f_geq_f_neg (x : ℝ) (hx : x ≥ 0) : f 1 x ≥ f 1 (-x) :=
by sorry

end NUMINAMATH_GPT_monotonicity_f_geq_f_neg_l1205_120598


namespace NUMINAMATH_GPT_proposition_range_l1205_120555

theorem proposition_range (m : ℝ) : 
  (m < 1/2 ∧ m ≠ 1/3) ∨ (m = 3) ↔ m ∈ Set.Iio (1/3:ℝ) ∪ Set.Ioo (1/3:ℝ) (1/2:ℝ) ∪ {3} :=
sorry

end NUMINAMATH_GPT_proposition_range_l1205_120555


namespace NUMINAMATH_GPT_bert_spent_fraction_at_hardware_store_l1205_120571

variable (f : ℝ)

def initial_money : ℝ := 41.99
def after_hardware (f : ℝ) := (1 - f) * initial_money
def after_dry_cleaners (f : ℝ) := after_hardware f - 7
def after_grocery (f : ℝ) := 0.5 * after_dry_cleaners f

theorem bert_spent_fraction_at_hardware_store 
(h1 : after_grocery f = 10.50) : 
  f = 0.3332 :=
by
  sorry

end NUMINAMATH_GPT_bert_spent_fraction_at_hardware_store_l1205_120571


namespace NUMINAMATH_GPT_solve_for_x_l1205_120585

theorem solve_for_x (x : ℝ) (h : 2 - 1 / (1 - x) = 1 / (1 - x)) : x = 0 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1205_120585


namespace NUMINAMATH_GPT_isosceles_triangle_area_l1205_120581

theorem isosceles_triangle_area
  (a b : ℝ) -- sides of the triangle
  (inradius : ℝ) (perimeter : ℝ)
  (angle : ℝ) -- angle in degrees
  (h_perimeter : 2 * a + b = perimeter)
  (h_inradius : inradius = 2.5)
  (h_angle : angle = 40)
  (h_perimeter_value : perimeter = 20)
  (h_semiperimeter : (perimeter / 2) = 10) :
  (inradius * (perimeter / 2) = 25) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l1205_120581


namespace NUMINAMATH_GPT_geese_percentage_among_non_swan_birds_l1205_120536

theorem geese_percentage_among_non_swan_birds :
  let total_birds := 100
  let geese := 0.40 * total_birds
  let swans := 0.20 * total_birds
  let non_swans := total_birds - swans
  let geese_percentage_among_non_swans := (geese / non_swans) * 100
  geese_percentage_among_non_swans = 50 := 
by sorry

end NUMINAMATH_GPT_geese_percentage_among_non_swan_birds_l1205_120536


namespace NUMINAMATH_GPT_find_y_l1205_120537

theorem find_y (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : y = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1205_120537


namespace NUMINAMATH_GPT_days_b_worked_l1205_120576

theorem days_b_worked (A_days B_days A_remaining_days : ℝ) (A_work_rate B_work_rate total_work : ℝ)
  (hA_rate : A_work_rate = 1 / A_days)
  (hB_rate : B_work_rate = 1 / B_days)
  (hA_days : A_days = 9)
  (hB_days : B_days = 15)
  (hA_remaining : A_remaining_days = 3)
  (h_total_work : ∀ x : ℝ, (x * B_work_rate + A_remaining_days * A_work_rate = total_work)) :
  ∃ x : ℝ, x = 10 :=
by
  sorry

end NUMINAMATH_GPT_days_b_worked_l1205_120576


namespace NUMINAMATH_GPT_oliver_january_money_l1205_120545

variable (x y z : ℕ)

-- Given conditions
def condition1 := y = x - 4
def condition2 := z = y + 32
def condition3 := z = 61

-- Statement to prove
theorem oliver_january_money (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z) : x = 33 :=
by
  sorry

end NUMINAMATH_GPT_oliver_january_money_l1205_120545


namespace NUMINAMATH_GPT_floor_sum_min_value_l1205_120574

theorem floor_sum_min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (⌊(x + y + z) / x⌋ + ⌊(x + y + z) / y⌋ + ⌊(x + y + z) / z⌋) = 7 :=
sorry

end NUMINAMATH_GPT_floor_sum_min_value_l1205_120574


namespace NUMINAMATH_GPT_max_value_2ab_plus_2bc_sqrt2_l1205_120514

theorem max_value_2ab_plus_2bc_sqrt2 (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_value_2ab_plus_2bc_sqrt2_l1205_120514


namespace NUMINAMATH_GPT_largest_lcm_value_l1205_120543

-- Define the conditions as local constants 
def lcm_18_3 : ℕ := Nat.lcm 18 3
def lcm_18_6 : ℕ := Nat.lcm 18 6
def lcm_18_9 : ℕ := Nat.lcm 18 9
def lcm_18_15 : ℕ := Nat.lcm 18 15
def lcm_18_21 : ℕ := Nat.lcm 18 21
def lcm_18_27 : ℕ := Nat.lcm 18 27

-- Statement to prove
theorem largest_lcm_value : max lcm_18_3 (max lcm_18_6 (max lcm_18_9 (max lcm_18_15 (max lcm_18_21 lcm_18_27)))) = 126 :=
by
  -- We assume the necessary calculations have been made
  have h1 : lcm_18_3 = 18 := by sorry
  have h2 : lcm_18_6 = 18 := by sorry
  have h3 : lcm_18_9 = 18 := by sorry
  have h4 : lcm_18_15 = 90 := by sorry
  have h5 : lcm_18_21 = 126 := by sorry
  have h6 : lcm_18_27 = 54 := by sorry

  -- Using above results to determine the maximum
  exact (by rw [h1, h2, h3, h4, h5, h6]; exact rfl)

end NUMINAMATH_GPT_largest_lcm_value_l1205_120543


namespace NUMINAMATH_GPT_stick_horisontal_fall_position_l1205_120596

-- Definitions based on the conditions
def stick_length : ℝ := 120 -- length of the stick in cm
def projection_distance : ℝ := 70 -- distance between projections of the ends of the stick on the floor

-- The main theorem to prove
theorem stick_horisontal_fall_position :
  ∀ (L d : ℝ), L = stick_length ∧ d = projection_distance → 
  ∃ x : ℝ, x = 25 :=
by
  intros L d h
  have h1 : L = stick_length := h.1
  have h2 : d = projection_distance := h.2
  -- The detailed proof steps will be here
  sorry

end NUMINAMATH_GPT_stick_horisontal_fall_position_l1205_120596


namespace NUMINAMATH_GPT_sum_of_first_five_terms_l1205_120553

noncomputable -- assuming non-computable for general proof involving sums
def arithmetic_sequence_sum (a_n : ℕ → ℤ) := ∃ d m : ℤ, ∀ n : ℕ, a_n = m + n * d

theorem sum_of_first_five_terms 
(a_n : ℕ → ℤ) 
(h_arith : arithmetic_sequence_sum a_n)
(h_cond : a_n 5 + a_n 8 - a_n 10 = 2)
: ((a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) = 10) := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_first_five_terms_l1205_120553


namespace NUMINAMATH_GPT_triangle_angle_y_l1205_120583

theorem triangle_angle_y (y : ℝ) (h1 : 2 * y + (y + 10) + 4 * y = 180) : 
  y = 170 / 7 := 
by
  sorry

end NUMINAMATH_GPT_triangle_angle_y_l1205_120583


namespace NUMINAMATH_GPT_asphalt_road_proof_l1205_120508

-- We define the initial conditions given in the problem
def man_hours (men days hours_per_day : Nat) : Nat :=
  men * days * hours_per_day

-- Given the conditions for asphalting 1 km road
def conditions_1 (men1 days1 hours_per_day1 : Nat) : Prop :=
  man_hours men1 days1 hours_per_day1 = 2880

-- Given that the second road is 2 km long
def conditions_2 (man_hours1 : Nat) : Prop :=
  2 * man_hours1 = 5760

-- Given the working conditions for the second road
def conditions_3 (men2 days2 hours_per_day2 : Nat) : Prop :=
  men2 * days2 * hours_per_day2 = 5760

-- The theorem to prove
theorem asphalt_road_proof 
  (men1 days1 hours_per_day1 days2 hours_per_day2 men2 : Nat)
  (H1 : conditions_1 men1 days1 hours_per_day1)
  (H2 : conditions_2 (man_hours men1 days1 hours_per_day1))
  (H3 : men2 * days2 * hours_per_day2 = 5760)
  : men2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_asphalt_road_proof_l1205_120508


namespace NUMINAMATH_GPT_jane_age_problem_l1205_120541

variables (J M a b c : ℕ)
variables (h1 : J = 2 * (a + b))
variables (h2 : J / 2 = a + b)
variables (h3 : c = 2 * J)
variables (h4 : M > 0)

theorem jane_age_problem (h5 : J - M = 3 * ((J / 2) - 2 * M))
                         (h6 : J - M = c - M)
                         (h7 : c = 2 * J) :
  J / M = 10 :=
sorry

end NUMINAMATH_GPT_jane_age_problem_l1205_120541


namespace NUMINAMATH_GPT_expand_expression_l1205_120549

theorem expand_expression (x y : ℝ) : 5 * (4 * x^3 - 3 * x * y + 7) = 20 * x^3 - 15 * x * y + 35 := 
sorry

end NUMINAMATH_GPT_expand_expression_l1205_120549


namespace NUMINAMATH_GPT_intersection_M_N_l1205_120589

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-1, 0, 1, 5}
def N : Set ℤ := {-2, 1, 2, 5}

-- The theorem stating that the intersection of M and N is {1, 5}
theorem intersection_M_N :
  M ∩ N = {1, 5} :=
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1205_120589


namespace NUMINAMATH_GPT_cos_squared_alpha_plus_pi_over_4_correct_l1205_120509

variable (α : ℝ)
axiom sin_two_alpha : Real.sin (2 * α) = 2 / 3

theorem cos_squared_alpha_plus_pi_over_4_correct :
  Real.cos (α + Real.pi / 4) ^ 2 = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_cos_squared_alpha_plus_pi_over_4_correct_l1205_120509


namespace NUMINAMATH_GPT_selection_count_l1205_120513

noncomputable def choose (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_count :
  let boys := 4
  let girls := 3
  let total := boys + girls
  let choose_boys_girls : ℕ := (choose 4 2) * (choose 3 1) + (choose 4 1) * (choose 3 2)
  choose_boys_girls = 30 := 
by
  sorry

end NUMINAMATH_GPT_selection_count_l1205_120513


namespace NUMINAMATH_GPT_max_area_enclosed_by_fencing_l1205_120591

theorem max_area_enclosed_by_fencing (l w : ℕ) (h : 2 * (l + w) = 142) : l * w ≤ 1260 :=
sorry

end NUMINAMATH_GPT_max_area_enclosed_by_fencing_l1205_120591


namespace NUMINAMATH_GPT_crushing_load_value_l1205_120522

-- Given definitions
def W : ℕ := 3
def T : ℕ := 2
def H : ℕ := 6
def L : ℕ := (30 * W^3 * T^5) / H^3

-- Theorem statement
theorem crushing_load_value :
  L = 120 :=
by {
  -- We provided definitions using the given conditions.
  -- Placeholder for proof is provided
  sorry
}

end NUMINAMATH_GPT_crushing_load_value_l1205_120522


namespace NUMINAMATH_GPT_expression_that_gives_value_8_l1205_120580

theorem expression_that_gives_value_8 (a b : ℝ) 
  (h_eq1 : a = 2) 
  (h_eq2 : b = 2) 
  (h_roots : ∀ x, (x - a) * (x - b) = x^2 - 4 * x + 4) : 
  2 * (a + b) = 8 :=
by
  sorry

end NUMINAMATH_GPT_expression_that_gives_value_8_l1205_120580


namespace NUMINAMATH_GPT_infinite_series_eq_1_div_400_l1205_120526

theorem infinite_series_eq_1_div_400 :
  (∑' n:ℕ, (4 * n + 2) / ((4 * n + 1)^2 * (4 * n + 5)^2)) = 1 / 400 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_eq_1_div_400_l1205_120526


namespace NUMINAMATH_GPT_sum_of_ages_l1205_120568

theorem sum_of_ages (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l1205_120568


namespace NUMINAMATH_GPT_father_cannot_see_boy_more_than_half_time_l1205_120524

def speed_boy := 10 -- speed in km/h
def speed_father := 5 -- speed in km/h

def cannot_see_boy_more_than_half_time (school_perimeter : ℝ) : Prop :=
  ¬(∃ T : ℝ, T > school_perimeter / (2 * speed_boy) ∧ T < school_perimeter / speed_boy)

theorem father_cannot_see_boy_more_than_half_time (school_perimeter : ℝ) (h_school_perimeter : school_perimeter > 0) :
  cannot_see_boy_more_than_half_time school_perimeter :=
by
  sorry

end NUMINAMATH_GPT_father_cannot_see_boy_more_than_half_time_l1205_120524


namespace NUMINAMATH_GPT_triangle_base_length_l1205_120569

theorem triangle_base_length (h : 3 = (b * 3) / 2) : b = 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_base_length_l1205_120569


namespace NUMINAMATH_GPT_min_length_l1205_120529

def length (a b : ℝ) : ℝ := b - a

noncomputable def M (m : ℝ) := {x | m ≤ x ∧ x ≤ m + 3 / 4}
noncomputable def N (n : ℝ) := {x | n - 1 / 3 ≤ x ∧ x ≤ n}
noncomputable def intersection (m n : ℝ) := {x | max m (n - 1 / 3) ≤ x ∧ x ≤ min (m + 3 / 4) n}

theorem min_length (m n : ℝ) (hM : ∀ x, x ∈ M m → 0 ≤ x ∧ x ≤ 1) (hN : ∀ x, x ∈ N n → 0 ≤ x ∧ x ≤ 1) :
  length (max m (n - 1 / 3)) (min (m + 3 / 4) n) = 1 / 12 :=
sorry

end NUMINAMATH_GPT_min_length_l1205_120529


namespace NUMINAMATH_GPT_exists_sequences_x_y_l1205_120517

def seq_a (a : ℕ → ℕ) : Prop :=
  a 0 = 4 ∧ a 1 = 22 ∧ ∀ n : ℕ, n ≥ 2 → a (n) = 6 * a (n - 1) - a (n - 2)

def seq_b (b : ℕ → ℕ) : Prop :=
  b 0 = 2 ∧ b 1 = 1 ∧ ∀ n : ℕ, n ≥ 2 → b (n) = 2 * b (n - 1) + b (n - 2)

theorem exists_sequences_x_y (a b : ℕ → ℕ) (x y : ℕ → ℕ) :
  seq_a a → seq_b b →
  (∀ n : ℕ, a n = (y n * y n + 7) / (x n - y n)) ↔ 
  (∀ n : ℕ, y n = b (2 * n + 1) ∧ x n = b (2 * n) + y n) :=
sorry

end NUMINAMATH_GPT_exists_sequences_x_y_l1205_120517


namespace NUMINAMATH_GPT_find_full_price_l1205_120586

-- Defining the conditions
variables (P : ℝ) 
-- The condition that 20% of the laptop's total cost is $240.
def condition : Prop := 0.2 * P = 240

-- The proof goal is to show that the full price P is $1200 given the condition
theorem find_full_price (h : condition P) : P = 1200 :=
sorry

end NUMINAMATH_GPT_find_full_price_l1205_120586


namespace NUMINAMATH_GPT_find_number_of_sides_l1205_120528

-- Defining the problem conditions
def sum_of_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

-- Statement of the problem
theorem find_number_of_sides (h : sum_of_interior_angles n = 1260) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_sides_l1205_120528


namespace NUMINAMATH_GPT_jack_total_books_is_541_l1205_120572

-- Define the number of books in each section
def american_books : ℕ := 6 * 34
def british_books : ℕ := 8 * 29
def world_books : ℕ := 5 * 21

-- Define the total number of books based on the given sections
def total_books : ℕ := american_books + british_books + world_books

-- Prove that the total number of books is 541
theorem jack_total_books_is_541 : total_books = 541 :=
by
  sorry

end NUMINAMATH_GPT_jack_total_books_is_541_l1205_120572


namespace NUMINAMATH_GPT_negation_equiv_l1205_120506

variable (p : Prop) [Nonempty ℝ]

def proposition := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

def negation_of_proposition : Prop := ∀ x : ℝ, Real.exp x - x - 1 > 0

theorem negation_equiv
  (h : proposition = p) : (¬ proposition) = negation_of_proposition := by
  sorry

end NUMINAMATH_GPT_negation_equiv_l1205_120506
