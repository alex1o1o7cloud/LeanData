import Mathlib

namespace NUMINAMATH_GPT_find_pq_l1555_155561

noncomputable def p_and_q (p q : ℝ) := 
  (Complex.I * 2 - 3) ∈ {z : Complex | z^2 * 2 + z * (p : Complex) + (q : Complex) = 0} ∧ 
  - (Complex.I * 2 + 3) ∈ {z : Complex | z^2 * 2 + z * (p : Complex) + (q : Complex) = 0}

theorem find_pq : ∃ (p q : ℝ), p_and_q p q ∧ p + q = 38 :=
by
  sorry

end NUMINAMATH_GPT_find_pq_l1555_155561


namespace NUMINAMATH_GPT_negation_of_proposition_l1555_155526

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^2 + 1 > 0) ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1555_155526


namespace NUMINAMATH_GPT_interest_years_l1555_155578

theorem interest_years (P : ℝ) (R : ℝ) (N : ℝ) (H1 : P = 2400) (H2 : (P * (R + 1) * N) / 100 - (P * R * N) / 100 = 72) : N = 3 :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_interest_years_l1555_155578


namespace NUMINAMATH_GPT_cos_double_angle_l1555_155500

theorem cos_double_angle (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 1 / 3) :
  Real.cos (2 * θ) = -7 / 9 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l1555_155500


namespace NUMINAMATH_GPT_least_whole_number_subtracted_from_ratio_l1555_155558

theorem least_whole_number_subtracted_from_ratio (x : ℕ) : 
  (6 - x) / (7 - x) < 16 / 21 := by
  sorry

end NUMINAMATH_GPT_least_whole_number_subtracted_from_ratio_l1555_155558


namespace NUMINAMATH_GPT_last_digit_fib_mod_12_l1555_155511

noncomputable def F : ℕ → ℕ
| 0       => 1
| 1       => 1
| (n + 2) => (F n + F (n + 1)) % 12

theorem last_digit_fib_mod_12 : ∃ N, ∀ n < N, (∃ k, F k % 12 = n) ∧ ∀ m > N, F m % 12 ≠ 11 :=
sorry

end NUMINAMATH_GPT_last_digit_fib_mod_12_l1555_155511


namespace NUMINAMATH_GPT_find_missing_exponent_l1555_155532

theorem find_missing_exponent (b e₁ e₂ e₃ e₄ : ℝ) (h1 : e₁ = 5.6) (h2 : e₂ = 10.3) (h3 : e₃ = 13.33744) (h4 : e₄ = 2.56256) :
  (b ^ e₁ * b ^ e₂) / b ^ e₄ = b ^ e₃ :=
by
  have h5 : e₁ + e₂ = 15.9 := sorry
  have h6 : 15.9 - e₄ = 13.33744 := sorry
  exact sorry

end NUMINAMATH_GPT_find_missing_exponent_l1555_155532


namespace NUMINAMATH_GPT_area_of_hexagon_l1555_155567

def isRegularHexagon (A B C D E F : Type) : Prop := sorry
def isInsideQuadrilateral (P : Type) (A B C D : Type) : Prop := sorry
def areaTriangle (P X Y : Type) : Real := sorry

theorem area_of_hexagon (A B C D E F P : Type)
    (h1 : isRegularHexagon A B C D E F)
    (h2 : isInsideQuadrilateral P A B C D)
    (h3 : areaTriangle P B C = 20)
    (h4 : areaTriangle P A D = 23) :
    ∃ area : Real, area = 189 :=
sorry

end NUMINAMATH_GPT_area_of_hexagon_l1555_155567


namespace NUMINAMATH_GPT_length_of_platform_proof_l1555_155522

def convert_speed_to_mps (kmph : Float) : Float := kmph * (5/18)

def distance_covered (speed : Float) (time : Float) : Float := speed * time

def length_of_platform (total_distance : Float) (train_length : Float) : Float := total_distance - train_length

theorem length_of_platform_proof :
  let speed_kmph := 72.0
  let speed_mps := convert_speed_to_mps speed_kmph
  let time_seconds := 36.0
  let train_length := 470.06
  let total_distance := distance_covered speed_mps time_seconds
  length_of_platform total_distance train_length = 249.94 :=
by
  sorry

end NUMINAMATH_GPT_length_of_platform_proof_l1555_155522


namespace NUMINAMATH_GPT_curve_meets_line_once_l1555_155552

theorem curve_meets_line_once (a : ℝ) (h : a > 0) :
  (∃! P : ℝ × ℝ, (∃ θ : ℝ, P.1 = a + 4 * Real.cos θ ∧ P.2 = 1 + 4 * Real.sin θ)
  ∧ (3 * P.1 + 4 * P.2 = 5)) → a = 7 :=
sorry

end NUMINAMATH_GPT_curve_meets_line_once_l1555_155552


namespace NUMINAMATH_GPT_expression_equality_l1555_155550

theorem expression_equality : 1 + 2 / (3 + 4 / 5) = 29 / 19 := by
  sorry

end NUMINAMATH_GPT_expression_equality_l1555_155550


namespace NUMINAMATH_GPT_geometric_progression_first_term_l1555_155541

theorem geometric_progression_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 8) 
  (h2 : a + a * r = 5) : 
  a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6) := 
by sorry

end NUMINAMATH_GPT_geometric_progression_first_term_l1555_155541


namespace NUMINAMATH_GPT_correct_statements_for_function_l1555_155533

-- Definitions and the problem statement
def f (x b c : ℝ) := x * |x| + b * x + c

theorem correct_statements_for_function (b c : ℝ) :
  (c = 0 → ∀ x, f x b c = -f (-x) b c) ∧
  (b = 0 ∧ c > 0 → ∀ x, f x b c = 0 → x = 0) ∧
  (∀ x, f x b c = f (-x) b (-c)) :=
sorry

end NUMINAMATH_GPT_correct_statements_for_function_l1555_155533


namespace NUMINAMATH_GPT_f_neg4_plus_f_0_range_of_a_l1555_155560

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else if x < 0 then -log (-x) / log 2 else 0

/- Prove that f(-4) + f(0) = -2 given the function properties -/
theorem f_neg4_plus_f_0 : f (-4) + f 0 = -2 :=
sorry

/- Prove the range of a such that f(a) > f(-a) is a > 1 or -1 < a < 0 given the function properties -/
theorem range_of_a (a : ℝ) : f a > f (-a) ↔ a > 1 ∨ (-1 < a ∧ a < 0) :=
sorry

end NUMINAMATH_GPT_f_neg4_plus_f_0_range_of_a_l1555_155560


namespace NUMINAMATH_GPT_add_to_fraction_eq_l1555_155564

theorem add_to_fraction_eq (n : ℤ) : (4 + n : ℤ) / (7 + n) = (2 : ℤ) / 3 → n = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_add_to_fraction_eq_l1555_155564


namespace NUMINAMATH_GPT_knights_and_liars_l1555_155598

/--
Suppose we have a set of natives, each of whom is either a liar or a knight.
Each native declares to all others: "You are all liars."
This setup implies that there must be exactly one knight among them.
-/
theorem knights_and_liars (natives : Type) (is_knight : natives → Prop) (is_liar : natives → Prop)
  (h1 : ∀ x, is_knight x ∨ is_liar x) 
  (h2 : ∀ x y, x ≠ y → (is_knight x → is_liar y) ∧ (is_liar x → is_knight y))
  : ∃! x, is_knight x :=
by
  sorry

end NUMINAMATH_GPT_knights_and_liars_l1555_155598


namespace NUMINAMATH_GPT_percentage_increase_second_year_l1555_155529

theorem percentage_increase_second_year :
  let initial_deposit : ℤ := 1000
  let balance_first_year : ℤ := 1100
  let total_balance_two_years : ℤ := 1320
  let percent_increase_first_year : ℚ := ((balance_first_year - initial_deposit) / initial_deposit) * 100
  let percent_increase_total : ℚ := ((total_balance_two_years - initial_deposit) / initial_deposit) * 100
  let increase_second_year : ℤ := total_balance_two_years - balance_first_year
  let percent_increase_second_year : ℚ := (increase_second_year / balance_first_year) * 100
  percent_increase_first_year = 10 ∧
  percent_increase_total = 32 ∧
  increase_second_year = 220 → 
  percent_increase_second_year = 20 := by
  intros initial_deposit balance_first_year total_balance_two_years percent_increase_first_year
         percent_increase_total increase_second_year percent_increase_second_year
  sorry

end NUMINAMATH_GPT_percentage_increase_second_year_l1555_155529


namespace NUMINAMATH_GPT_nathan_dice_roll_probability_l1555_155554

noncomputable def probability_nathan_rolls : ℚ :=
  let prob_less4_first_die : ℚ := 3 / 8
  let prob_greater5_second_die : ℚ := 3 / 8
  prob_less4_first_die * prob_greater5_second_die

theorem nathan_dice_roll_probability : probability_nathan_rolls = 9 / 64 := by
  sorry

end NUMINAMATH_GPT_nathan_dice_roll_probability_l1555_155554


namespace NUMINAMATH_GPT_money_left_after_purchase_l1555_155577

noncomputable def total_cost : ℝ := 250 + 25 + 35 + 45 + 90

def savings_erika : ℝ := 155

noncomputable def savings_rick : ℝ := total_cost / 2

def savings_sam : ℝ := 175

def combined_cost_cake_flowers_skincare : ℝ := 25 + 35 + 45

noncomputable def savings_amy : ℝ := 2 * combined_cost_cake_flowers_skincare

noncomputable def total_savings : ℝ := savings_erika + savings_rick + savings_sam + savings_amy

noncomputable def money_left : ℝ := total_savings - total_cost

theorem money_left_after_purchase : money_left = 317.5 := by
  sorry

end NUMINAMATH_GPT_money_left_after_purchase_l1555_155577


namespace NUMINAMATH_GPT_proposition_A_l1555_155580

variables {m n : Line} {α β : Plane}

def parallel (x y : Line) : Prop := sorry -- definition for parallel lines
def perpendicular (x : Line) (P : Plane) : Prop := sorry -- definition for perpendicular line to plane
def parallel_planes (P Q : Plane) : Prop := sorry -- definition for parallel planes

theorem proposition_A (hmn : parallel m n) (hperp_mα : perpendicular m α) (hperp_nβ : perpendicular n β) : parallel_planes α β :=
sorry

end NUMINAMATH_GPT_proposition_A_l1555_155580


namespace NUMINAMATH_GPT_value_of_3W5_l1555_155546

def W (a b : ℕ) : ℕ := b + 7 * a - a ^ 2

theorem value_of_3W5 : W 3 5 = 17 := by 
  sorry

end NUMINAMATH_GPT_value_of_3W5_l1555_155546


namespace NUMINAMATH_GPT_largest_among_a_b_c_d_l1555_155527

noncomputable def a : ℝ := Real.log 2022 / Real.log 2021
noncomputable def b : ℝ := Real.log 2023 / Real.log 2022
noncomputable def c : ℝ := 2022 / 2021
noncomputable def d : ℝ := 2023 / 2022

theorem largest_among_a_b_c_d : max a (max b (max c d)) = c := 
sorry

end NUMINAMATH_GPT_largest_among_a_b_c_d_l1555_155527


namespace NUMINAMATH_GPT_max_regions_quadratic_trinomials_l1555_155595

theorem max_regions_quadratic_trinomials (a b c : Fin 100 → ℝ) :
  ∃ R, (∀ (n : ℕ), n ≤ 100 → R = n^2 + 1) → R = 10001 := 
  sorry

end NUMINAMATH_GPT_max_regions_quadratic_trinomials_l1555_155595


namespace NUMINAMATH_GPT_value_of_b_l1555_155501

theorem value_of_b (b : ℚ) (h : b + b / 4 = 3) : b = 12 / 5 := by
  sorry

end NUMINAMATH_GPT_value_of_b_l1555_155501


namespace NUMINAMATH_GPT_find_k_l1555_155559

theorem find_k (x k : ℝ) (h₁ : (x^2 - k) * (x - k) = x^3 - k * (x^2 + x + 3))
               (h₂ : k ≠ 0) : k = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1555_155559


namespace NUMINAMATH_GPT_original_price_of_dish_l1555_155592

theorem original_price_of_dish : 
  ∀ (P : ℝ), 
  1.05 * P - 1.035 * P = 0.54 → 
  P = 36 :=
by
  intros P h
  sorry

end NUMINAMATH_GPT_original_price_of_dish_l1555_155592


namespace NUMINAMATH_GPT_divisor_of_a_l1555_155505

theorem divisor_of_a (a b : ℕ) (hx : a % x = 3) (hb : b % 6 = 5) (hab : (a * b) % 48 = 15) : x = 48 :=
by sorry

end NUMINAMATH_GPT_divisor_of_a_l1555_155505


namespace NUMINAMATH_GPT_quadratic_roots_range_l1555_155542

theorem quadratic_roots_range (m : ℝ) :
  (∃ p n : ℝ, p > 0 ∧ n < 0 ∧ 2 * p^2 + (m + 1) * p + m = 0 ∧ 2 * n^2 + (m + 1) * n + m = 0) →
  m < 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_range_l1555_155542


namespace NUMINAMATH_GPT_video_streaming_budget_l1555_155581

theorem video_streaming_budget 
  (weekly_food_budget : ℕ) 
  (weeks : ℕ) 
  (total_food_budget : ℕ) 
  (rent : ℕ) 
  (phone : ℕ) 
  (savings_rate : ℝ)
  (total_savings : ℕ) 
  (total_expenses : ℕ) 
  (known_expenses: ℕ) 
  (total_spending : ℕ):
  weekly_food_budget = 100 →
  weeks = 4 →
  total_food_budget = weekly_food_budget * weeks →
  rent = 1500 →
  phone = 50 →
  savings_rate = 0.10 →
  total_savings = 198 →
  total_expenses = total_food_budget + rent + phone →
  total_spending = (total_savings : ℝ) / savings_rate →
  known_expenses = total_expenses →
  total_spending - known_expenses = 30 :=
by sorry

end NUMINAMATH_GPT_video_streaming_budget_l1555_155581


namespace NUMINAMATH_GPT_beth_cans_of_corn_l1555_155536

theorem beth_cans_of_corn (C P : ℕ) (h1 : P = 2 * C + 15) (h2 : P = 35) : C = 10 :=
by
  sorry

end NUMINAMATH_GPT_beth_cans_of_corn_l1555_155536


namespace NUMINAMATH_GPT_problem_B_height_l1555_155503

noncomputable def point_B_height (cos : ℝ → ℝ) : ℝ :=
  let θ := 30 * (Real.pi / 180)
  let cos30 := cos θ
  let original_vertical_height := 1 / 2
  let additional_height := cos30 * (1 / 2)
  original_vertical_height + additional_height

theorem problem_B_height : 
  point_B_height Real.cos = (2 + Real.sqrt 3) / 4 := 
by 
  sorry

end NUMINAMATH_GPT_problem_B_height_l1555_155503


namespace NUMINAMATH_GPT_markers_last_group_correct_l1555_155594

-- Definition of conditions in Lean 4
def total_students : ℕ := 30
def boxes_of_markers : ℕ := 22
def markers_per_box : ℕ := 5
def students_in_first_group : ℕ := 10
def markers_per_student_first_group : ℕ := 2
def students_in_second_group : ℕ := 15
def markers_per_student_second_group : ℕ := 4

-- Calculate total markers allocated to the first and second groups
def markers_used_by_first_group : ℕ := students_in_first_group * markers_per_student_first_group
def markers_used_by_second_group : ℕ := students_in_second_group * markers_per_student_second_group

-- Total number of markers available
def total_markers : ℕ := boxes_of_markers * markers_per_box

-- Markers left for last group
def markers_remaining : ℕ := total_markers - (markers_used_by_first_group + markers_used_by_second_group)

-- Number of students in the last group
def students_in_last_group : ℕ := total_students - (students_in_first_group + students_in_second_group)

-- Number of markers per student in the last group
def markers_per_student_last_group : ℕ := markers_remaining / students_in_last_group

-- The proof problem in Lean 4
theorem markers_last_group_correct : markers_per_student_last_group = 6 :=
  by
  -- Proof is to be filled here
  sorry

end NUMINAMATH_GPT_markers_last_group_correct_l1555_155594


namespace NUMINAMATH_GPT_problem_xyz_inequality_l1555_155585

theorem problem_xyz_inequality (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h_eq : x^2 + y^2 + z^2 + x * y * z = 4) :
  x * y * z ≤ x * y + y * z + z * x ∧ x * y + y * z + z * x ≤ x * y * z + 2 :=
by 
  sorry

end NUMINAMATH_GPT_problem_xyz_inequality_l1555_155585


namespace NUMINAMATH_GPT_max_H2O_produced_l1555_155515

theorem max_H2O_produced :
  ∀ (NaOH H2SO4 H2O : ℝ)
  (n_NaOH : NaOH = 1.5)
  (n_H2SO4 : H2SO4 = 1)
  (balanced_reaction : 2 * NaOH + H2SO4 = 2 * H2O + 1 * (NaOH + H2SO4)),
  H2O = 1.5 :=
by
  intros NaOH H2SO4 H2O n_NaOH n_H2SO4 balanced_reaction
  sorry

end NUMINAMATH_GPT_max_H2O_produced_l1555_155515


namespace NUMINAMATH_GPT_number_of_people_in_group_l1555_155510

/-- The number of people in the group N is such that when one of the people weighing 65 kg is replaced
by a new person weighing 100 kg, the average weight of the group increases by 3.5 kg. -/
theorem number_of_people_in_group (N : ℕ) (W : ℝ) 
  (h1 : (W + 35) / N = W / N + 3.5) 
  (h2 : W + 35 = W - 65 + 100) : 
  N = 10 :=
sorry

end NUMINAMATH_GPT_number_of_people_in_group_l1555_155510


namespace NUMINAMATH_GPT_number_of_fences_painted_l1555_155508

-- Definitions based on the problem conditions
def meter_fee : ℝ := 0.2
def fence_length : ℝ := 500
def total_earnings : ℝ := 5000

-- Target statement
theorem number_of_fences_painted : (total_earnings / (fence_length * meter_fee)) = 50 := by
sorry

end NUMINAMATH_GPT_number_of_fences_painted_l1555_155508


namespace NUMINAMATH_GPT_max_area_l1555_155555

theorem max_area (l w : ℝ) (h : l + 3 * w = 500) : l * w ≤ 62500 :=
by
  sorry

end NUMINAMATH_GPT_max_area_l1555_155555


namespace NUMINAMATH_GPT_shortest_distance_correct_l1555_155544

noncomputable def shortest_distance_a_to_c1 (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2 + 2 * b * c)

theorem shortest_distance_correct (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) :
  shortest_distance_a_to_c1 a b c h₁ h₂ = Real.sqrt (a^2 + b^2 + c^2 + 2 * b * c) :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_shortest_distance_correct_l1555_155544


namespace NUMINAMATH_GPT_number_of_ways_to_select_officers_l1555_155534

-- Definitions based on conditions
def boys : ℕ := 6
def girls : ℕ := 4
def total_people : ℕ := boys + girls
def officers_to_select : ℕ := 3

-- Number of ways to choose 3 individuals out of 10
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def total_choices : ℕ := choose total_people officers_to_select

-- Number of ways to choose 3 boys out of 6 (0 girls)
def all_boys_choices : ℕ := choose boys officers_to_select

-- Number of ways to choose at least 1 girl
def at_least_one_girl_choices : ℕ := total_choices - all_boys_choices

-- Theorem to prove the number of ways to select the officers
theorem number_of_ways_to_select_officers :
  at_least_one_girl_choices = 100 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_select_officers_l1555_155534


namespace NUMINAMATH_GPT_integer_coordinates_point_exists_l1555_155530

theorem integer_coordinates_point_exists (p q : ℤ) (h : p^2 - 4 * q = 0) :
  ∃ a b : ℤ, b = a^2 + p * a + q ∧ (a = -p ∧ b = q) ∧ (a ≠ -p → (a = p ∧ b = q) → (p^2 - 4 * b = 0)) :=
by
  sorry

end NUMINAMATH_GPT_integer_coordinates_point_exists_l1555_155530


namespace NUMINAMATH_GPT_value_of_m_l1555_155574

-- Definitions of the conditions
def base6_num (m : ℕ) : ℕ := 2 + m * 6^2
def dec_num (d : ℕ) := d = 146

-- Theorem to prove
theorem value_of_m (m : ℕ) (h1 : base6_num m = 146) : m = 4 := 
sorry

end NUMINAMATH_GPT_value_of_m_l1555_155574


namespace NUMINAMATH_GPT_exactly_two_roots_iff_l1555_155584

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end NUMINAMATH_GPT_exactly_two_roots_iff_l1555_155584


namespace NUMINAMATH_GPT_division_of_expression_l1555_155572

theorem division_of_expression (x y : ℝ) (hy : y ≠ 0) (hx : x ≠ 0) : (12 * x^2 * y) / (-6 * x * y) = -2 * x := by
  sorry

end NUMINAMATH_GPT_division_of_expression_l1555_155572


namespace NUMINAMATH_GPT_problem_statement_l1555_155593

theorem problem_statement {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  1 < 1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z) ∧ 1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z) < 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1555_155593


namespace NUMINAMATH_GPT_total_dots_l1555_155518

def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

theorem total_dots :
  (ladybugs_monday + ladybugs_tuesday) * dots_per_ladybug = 78 :=
by
  sorry

end NUMINAMATH_GPT_total_dots_l1555_155518


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1555_155565

-- System (1)
theorem system1_solution (x y : ℚ) (h1 : 4 * x + 8 * y = 12) (h2 : 3 * x - 2 * y = 5) :
  x = 2 ∧ y = 1 / 2 := by
  sorry

-- System (2)
theorem system2_solution (x y : ℚ) (h1 : (1/2) * x - (y + 1) / 3 = 1) (h2 : 6 * x + 2 * y = 10) :
  x = 2 ∧ y = -1 := by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1555_155565


namespace NUMINAMATH_GPT_max_hours_at_regular_rate_l1555_155549

-- Define the maximum hours at regular rate H
def max_regular_hours (H : ℕ) : Prop := 
  let regular_rate := 16
  let overtime_rate := 16 + (0.75 * 16)
  let total_hours := 60
  let total_compensation := 1200
  16 * H + 28 * (total_hours - H) = total_compensation

theorem max_hours_at_regular_rate : ∃ H, max_regular_hours H ∧ H = 40 :=
sorry

end NUMINAMATH_GPT_max_hours_at_regular_rate_l1555_155549


namespace NUMINAMATH_GPT_time_for_second_train_to_cross_l1555_155556

def length_first_train : ℕ := 100
def speed_first_train : ℕ := 10
def length_second_train : ℕ := 150
def speed_second_train : ℕ := 15
def distance_between_trains : ℕ := 50

def total_distance : ℕ := length_first_train + length_second_train + distance_between_trains
def relative_speed : ℕ := speed_second_train - speed_first_train

theorem time_for_second_train_to_cross :
  total_distance / relative_speed = 60 :=
by
  -- Definitions and intermediate steps would be handled in the proof here
  sorry

end NUMINAMATH_GPT_time_for_second_train_to_cross_l1555_155556


namespace NUMINAMATH_GPT_employees_count_l1555_155568

-- Let E be the number of employees excluding the manager
def E (employees : ℕ) : ℕ := employees

-- Let T be the total salary of employees excluding the manager
def T (employees : ℕ) : ℕ := employees * 1500

-- Conditions given in the problem
def average_salary (employees : ℕ) : ℕ := T employees / E employees
def new_average_salary (employees : ℕ) : ℕ := (T employees + 22500) / (E employees + 1)

theorem employees_count : (average_salary employees = 1500) ∧ (new_average_salary employees = 2500) ∧ (manager_salary = 22500) → (E employees = 20) :=
  by sorry

end NUMINAMATH_GPT_employees_count_l1555_155568


namespace NUMINAMATH_GPT_speed_of_first_half_of_journey_l1555_155596

theorem speed_of_first_half_of_journey
  (total_time : ℝ)
  (speed_second_half : ℝ)
  (total_distance : ℝ)
  (first_half_distance : ℝ)
  (second_half_distance : ℝ)
  (time_second_half : ℝ)
  (time_first_half : ℝ)
  (speed_first_half : ℝ) :
  total_time = 15 →
  speed_second_half = 24 →
  total_distance = 336 →
  first_half_distance = total_distance / 2 →
  second_half_distance = total_distance / 2 →
  time_second_half = second_half_distance / speed_second_half →
  time_first_half = total_time - time_second_half →
  speed_first_half = first_half_distance / time_first_half →
  speed_first_half = 21 :=
by intros; sorry

end NUMINAMATH_GPT_speed_of_first_half_of_journey_l1555_155596


namespace NUMINAMATH_GPT_combined_value_of_silver_and_gold_l1555_155528

noncomputable def silver_cube_side : ℝ := 3
def silver_weight_per_cubic_inch : ℝ := 6
def silver_price_per_ounce : ℝ := 25
def gold_layer_fraction : ℝ := 0.5
def gold_weight_per_square_inch : ℝ := 0.1
def gold_price_per_ounce : ℝ := 1800
def markup_percentage : ℝ := 1.10

def calculate_combined_value (side weight_per_cubic_inch silver_price layer_fraction weight_per_square_inch gold_price markup : ℝ) : ℝ :=
  let volume := side^3
  let weight_silver := volume * weight_per_cubic_inch
  let value_silver := weight_silver * silver_price
  let surface_area := 6 * side^2
  let area_gold := surface_area * layer_fraction
  let weight_gold := area_gold * weight_per_square_inch
  let value_gold := weight_gold * gold_price
  let total_value_before_markup := value_silver + value_gold
  let selling_price := total_value_before_markup * (1 + markup)
  selling_price

theorem combined_value_of_silver_and_gold :
  calculate_combined_value silver_cube_side silver_weight_per_cubic_inch silver_price_per_ounce gold_layer_fraction gold_weight_per_square_inch gold_price_per_ounce markup_percentage = 18711 :=
by
  sorry

end NUMINAMATH_GPT_combined_value_of_silver_and_gold_l1555_155528


namespace NUMINAMATH_GPT_sufficient_condition_for_ellipse_with_foci_y_axis_l1555_155509

theorem sufficient_condition_for_ellipse_with_foci_y_axis (m n : ℝ) (h : m > n ∧ n > 0) :
  (∃ a b : ℝ, (a^2 = m / n) ∧ (b^2 = 1 / n) ∧ (a > b)) ∧ ¬(∀ u v : ℝ, (u^2 = m / v) → (v^2 = 1 / v) → (u > v) → (v = n ∧ u = m)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_ellipse_with_foci_y_axis_l1555_155509


namespace NUMINAMATH_GPT_acres_left_untouched_l1555_155566

def total_acres := 65057
def covered_acres := 64535

theorem acres_left_untouched : total_acres - covered_acres = 522 :=
by
  sorry

end NUMINAMATH_GPT_acres_left_untouched_l1555_155566


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1555_155576

variable {a : ℕ → ℕ}

-- Defining the arithmetic sequence condition
axiom arithmetic_sequence_condition : a 3 + a 7 = 37

-- The goal is to prove that the total of a_2 + a_4 + a_6 + a_8 is 74
theorem arithmetic_sequence_sum : a 2 + a 4 + a 6 + a 8 = 74 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1555_155576


namespace NUMINAMATH_GPT_max_andy_l1555_155548

def max_cookies_eaten_by_andy (total : ℕ) (k1 k2 a b c : ℤ) : Prop :=
  a + b + c = total ∧ b = 2 * a + 2 ∧ c = a - 3

theorem max_andy (total : ℕ) (a : ℤ) :
  (∀ b c, max_cookies_eaten_by_andy total 2 (-3) a b c) → a ≤ 7 :=
by
  intros H
  sorry

end NUMINAMATH_GPT_max_andy_l1555_155548


namespace NUMINAMATH_GPT_alpha_eq_pi_over_3_l1555_155519

theorem alpha_eq_pi_over_3 (α β γ : ℝ) (h1 : 0 < α ∧ α < π) (h2 : α + β + γ = π) 
    (h3 : 2 * Real.sin α + Real.tan β + Real.tan γ = 2 * Real.sin α * Real.tan β * Real.tan γ) :
    α = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_alpha_eq_pi_over_3_l1555_155519


namespace NUMINAMATH_GPT_complement_A_eq_interval_l1555_155521

open Set

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := {x : ℝ | True}

-- Define the set A according to the given conditions
def A : Set ℝ := {x : ℝ | x ≥ 1} ∪ {x : ℝ | x ≤ 0}

-- State the theorem that the complement of A with respect to U is (0, 1)
theorem complement_A_eq_interval : ∀ x : ℝ, x ∈ U \ A ↔ x ∈ Ioo 0 1 := by
  intros x
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_complement_A_eq_interval_l1555_155521


namespace NUMINAMATH_GPT_shaded_region_area_l1555_155506

def isosceles_triangle (AB AC BC : ℝ) (BAC : ℝ) : Prop :=
  AB = AC ∧ BAC = 120 ∧ BC = 32

def circle_with_diameter (diameter : ℝ) (radius : ℝ) : Prop :=
  radius = diameter / 2

theorem shaded_region_area :
  ∀ (AB AC BC : ℝ) (BAC : ℝ) (O : Type) (a b c : ℕ),
    isosceles_triangle AB AC BC BAC →
    circle_with_diameter BC 8 →
    (a = 43) ∧ (b = 128) ∧ (c = 3) →
    a + b + c = 174 :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1555_155506


namespace NUMINAMATH_GPT_kendra_words_learned_l1555_155517

theorem kendra_words_learned (Goal : ℕ) (WordsNeeded : ℕ) (WordsAlreadyLearned : ℕ) 
  (h1 : Goal = 60) (h2 : WordsNeeded = 24) :
  WordsAlreadyLearned = Goal - WordsNeeded :=
sorry

end NUMINAMATH_GPT_kendra_words_learned_l1555_155517


namespace NUMINAMATH_GPT_combined_weight_l1555_155573

variables (G D C : ℝ)

def grandmother_weight (G D C : ℝ) := G + D + C = 150
def daughter_weight (D : ℝ) := D = 42
def child_weight (G C : ℝ) := C = 1/5 * G

theorem combined_weight (G D C : ℝ) (h1 : grandmother_weight G D C) (h2 : daughter_weight D) (h3 : child_weight G C) : D + C = 60 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_l1555_155573


namespace NUMINAMATH_GPT_correct_propositions_l1555_155537

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_propositions :
  ¬ ∀ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 → ∃ k : ℤ, x1 - x2 = k * Real.pi ∧
  (∀ (x : ℝ), f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (f (- (Real.pi / 6)) = 0) ∧
  ¬ ∀ (x : ℝ), f x = f (-x - Real.pi / 6) :=
sorry

end NUMINAMATH_GPT_correct_propositions_l1555_155537


namespace NUMINAMATH_GPT_train_length_l1555_155523

theorem train_length (time_crossing : ℝ) (speed_train : ℝ) (speed_man : ℝ) (rel_speed : ℝ) (length_train : ℝ) 
    (h1 : time_crossing = 39.99680025597952)
    (h2 : speed_train = 56)
    (h3 : speed_man = 2)
    (h4 : rel_speed = (speed_train - speed_man) * (1000 / 3600))
    (h5 : length_train = rel_speed * time_crossing):
 length_train = 599.9520038396928 :=
by 
  sorry

end NUMINAMATH_GPT_train_length_l1555_155523


namespace NUMINAMATH_GPT_probability_of_rolling_3_or_5_is_1_over_4_l1555_155575

def fair_8_sided_die := {outcome : Fin 8 // true}

theorem probability_of_rolling_3_or_5_is_1_over_4 :
  (1 / 4 : ℚ) = 2 / 8 :=
by sorry

end NUMINAMATH_GPT_probability_of_rolling_3_or_5_is_1_over_4_l1555_155575


namespace NUMINAMATH_GPT_find_digit_D_l1555_155587

def is_digit (n : ℕ) : Prop := n < 10

theorem find_digit_D (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C)
  (h5 : B ≠ D) (h6 : C ≠ D) (h7 : is_digit A) (h8 : is_digit B) (h9 : is_digit C) (h10 : is_digit D) :
  (1000 * A + 100 * B + 10 * C + D) * 2 = 5472 → D = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_digit_D_l1555_155587


namespace NUMINAMATH_GPT_reciprocal_and_fraction_l1555_155599

theorem reciprocal_and_fraction (a b : ℝ) (h1 : a * b = 1) (h2 : (2/5) * a = 20) : 
  b = (1/a) ∧ (1/3) * a = (50/3) := 
by 
  sorry

end NUMINAMATH_GPT_reciprocal_and_fraction_l1555_155599


namespace NUMINAMATH_GPT_area_of_walkways_is_214_l1555_155524

-- Definitions for conditions
def width_of_flower_beds : ℕ := 2 * 7  -- two beds each 7 feet wide
def walkways_between_beds_width : ℕ := 3 * 2  -- three walkways each 2 feet wide (one on each side and one in between)
def total_width : ℕ := width_of_flower_beds + walkways_between_beds_width  -- Total width

def height_of_flower_beds : ℕ := 3 * 3  -- three rows of beds each 3 feet high
def walkways_between_beds_height : ℕ := 4 * 2  -- four walkways each 2 feet wide (one on each end and one between each row)
def total_height : ℕ := height_of_flower_beds + walkways_between_beds_height  -- Total height

def total_area_of_garden : ℕ := total_width * total_height  -- Total area of the garden including walkways

def area_of_one_flower_bed : ℕ := 7 * 3  -- Area of one flower bed
def total_area_of_flower_beds : ℕ := 6 * area_of_one_flower_bed  -- Total area of six flower beds

def total_area_walkways : ℕ := total_area_of_garden - total_area_of_flower_beds  -- Total area of the walkways

-- Theorem to prove the area of the walkways
theorem area_of_walkways_is_214 : total_area_walkways = 214 := sorry

end NUMINAMATH_GPT_area_of_walkways_is_214_l1555_155524


namespace NUMINAMATH_GPT_power_sum_l1555_155540

theorem power_sum (n : ℕ) : (-2 : ℤ)^n + (-2 : ℤ)^(n+1) = 2^n := by
  sorry

end NUMINAMATH_GPT_power_sum_l1555_155540


namespace NUMINAMATH_GPT_sum_of_extreme_values_of_x_l1555_155512

open Real

theorem sum_of_extreme_values_of_x 
  (x y z : ℝ)
  (h1 : x + y + z = 6)
  (h2 : x^2 + y^2 + z^2 = 14) : 
  (min x + max x) = (10 / 3) :=
sorry

end NUMINAMATH_GPT_sum_of_extreme_values_of_x_l1555_155512


namespace NUMINAMATH_GPT_right_triangle_area_l1555_155504

theorem right_triangle_area (a b c : ℝ)
    (h1 : a = 16)
    (h2 : ∃ r, r = 6)
    (h3 : c = Real.sqrt (a^2 + b^2)) 
    (h4 : a^2 + b^2 = c^2) :
    1/2 * a * b = 240 := 
by
  -- given:
  -- a = 16
  -- ∃ r, r = 6
  -- c = Real.sqrt (a^2 + b^2)
  -- a^2 + b^2 = c^2
  -- Prove: 1/2 * a * b = 240
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1555_155504


namespace NUMINAMATH_GPT_problem_l1555_155514

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end NUMINAMATH_GPT_problem_l1555_155514


namespace NUMINAMATH_GPT_range_of_f_l1555_155589

noncomputable def f (x : ℕ) : ℤ := x^2 - 2*x

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ ({0, 1, 2, 3} : Finset ℕ), f x = y} = {-1, 0, 3} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1555_155589


namespace NUMINAMATH_GPT_total_painting_cost_l1555_155520

variable (house_area : ℕ) (price_per_sqft : ℕ)

theorem total_painting_cost (h1 : house_area = 484) (h2 : price_per_sqft = 20) :
  house_area * price_per_sqft = 9680 :=
by
  sorry

end NUMINAMATH_GPT_total_painting_cost_l1555_155520


namespace NUMINAMATH_GPT_peter_age_l1555_155531

theorem peter_age (P Q : ℕ) (h1 : Q - P = P / 2) (h2 : P + Q = 35) : Q = 21 :=
  sorry

end NUMINAMATH_GPT_peter_age_l1555_155531


namespace NUMINAMATH_GPT_smallest_positive_multiple_45_l1555_155563

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_45_l1555_155563


namespace NUMINAMATH_GPT_no_such_a_b_exists_l1555_155562

open Set

def A (a b : ℝ) : Set (ℤ × ℝ) :=
  { p | ∃ x : ℤ, p = (x, a * x + b) }

def B : Set (ℤ × ℝ) :=
  { p | ∃ x : ℤ, p = (x, 3 * (x : ℝ) ^ 2 + 15) }

def C : Set (ℝ × ℝ) :=
  { p | p.1 ^ 2 + p.2 ^ 2 ≤ 144 }

theorem no_such_a_b_exists :
  ¬ ∃ (a b : ℝ), 
    ((A a b ∩ B).Nonempty) ∧ ((a, b) ∈ C) :=
sorry

end NUMINAMATH_GPT_no_such_a_b_exists_l1555_155562


namespace NUMINAMATH_GPT_part_1_part_2_l1555_155590

def f (x a : ℝ) : ℝ := |x - a| + 5 * x

theorem part_1 (x : ℝ) : (|x + 1| + 5 * x ≤ 5 * x + 3) ↔ (x ∈ Set.Icc (-4 : ℝ) 2) :=
by
  sorry

theorem part_2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) :=
by
  sorry

end NUMINAMATH_GPT_part_1_part_2_l1555_155590


namespace NUMINAMATH_GPT_largest_lcm_18_l1555_155586

theorem largest_lcm_18 :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by sorry

end NUMINAMATH_GPT_largest_lcm_18_l1555_155586


namespace NUMINAMATH_GPT_linear_function_difference_l1555_155545

variable (g : ℝ → ℝ)
variable (h_linear : ∀ x y, g (x + y) = g x + g y)
variable (h_value : g 8 - g 4 = 16)

theorem linear_function_difference : g 16 - g 4 = 48 := by
  sorry

end NUMINAMATH_GPT_linear_function_difference_l1555_155545


namespace NUMINAMATH_GPT_fx_root_and_decreasing_l1555_155547

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x - Real.log x / Real.log 2

theorem fx_root_and_decreasing (a x0 : ℝ) (h0 : 0 < a) (hx0 : 0 < x0) (h_cond : a < x0) (hf_root : f x0 = 0) 
  (hf_decreasing : ∀ x y : ℝ, x < y → f y < f x) : f a > 0 := 
sorry

end NUMINAMATH_GPT_fx_root_and_decreasing_l1555_155547


namespace NUMINAMATH_GPT_fill_two_thirds_of_bucket_time_l1555_155551

theorem fill_two_thirds_of_bucket_time (fill_entire_bucket_time : ℝ) (h : fill_entire_bucket_time = 3) : (2 / 3) * fill_entire_bucket_time = 2 :=
by 
  sorry

end NUMINAMATH_GPT_fill_two_thirds_of_bucket_time_l1555_155551


namespace NUMINAMATH_GPT_probabilities_equal_l1555_155582

noncomputable def probability (m1 m2 : ℕ) : ℚ := m1 / (m1 + m2 : ℚ)

theorem probabilities_equal 
  (u j p b : ℕ) 
  (huj : u > j) 
  (hbp : b > p) : 
  (probability u p) * (probability b u) * (probability j b) * (probability p j) = 
  (probability u b) * (probability p u) * (probability j p) * (probability b j) :=
by
  sorry

end NUMINAMATH_GPT_probabilities_equal_l1555_155582


namespace NUMINAMATH_GPT_geometric_sequence_a5_l1555_155502

variable {a : Nat → ℝ} {q : ℝ}

-- Conditions
def is_geometric_sequence (a : Nat → ℝ) (q : ℝ) :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = q * a n

def condition_eq (a : Nat → ℝ) :=
  a 5 + a 4 = 3 * (a 3 + a 2)

-- Proof statement
theorem geometric_sequence_a5 (hq : q ≠ -1)
  (hg : is_geometric_sequence a q)
  (hc : condition_eq a) : a 5 = 9 :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l1555_155502


namespace NUMINAMATH_GPT_sum_of_base4_numbers_is_correct_l1555_155557

-- Define the four base numbers
def n1 : ℕ := 2 * 4^2 + 1 * 4^1 + 2 * 4^0
def n2 : ℕ := 1 * 4^2 + 0 * 4^1 + 3 * 4^0
def n3 : ℕ := 3 * 4^2 + 2 * 4^1 + 1 * 4^0

-- Define the expected sum in base 4 interpreted as a natural number
def expected_sum : ℕ := 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0

-- State the theorem
theorem sum_of_base4_numbers_is_correct : n1 + n2 + n3 = expected_sum := by
  sorry

end NUMINAMATH_GPT_sum_of_base4_numbers_is_correct_l1555_155557


namespace NUMINAMATH_GPT_find_x_l1555_155535

-- Define the condition from the problem statement
def condition1 (x : ℝ) : Prop := 70 = 0.60 * x + 22

-- Translate the question to the Lean statement form
theorem find_x (x : ℝ) (h : condition1 x) : x = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_l1555_155535


namespace NUMINAMATH_GPT_cosine_identity_l1555_155525

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end NUMINAMATH_GPT_cosine_identity_l1555_155525


namespace NUMINAMATH_GPT_max_distinct_counts_proof_l1555_155583

-- Define the number of boys (B) and girls (G)
def B : ℕ := 29
def G : ℕ := 15

-- Define the maximum distinct dance counts achievable
def max_distinct_counts : ℕ := 29

-- The theorem to prove
theorem max_distinct_counts_proof:
  ∃ (distinct_counts : ℕ), distinct_counts = max_distinct_counts ∧ distinct_counts <= B + G := 
by
  sorry

end NUMINAMATH_GPT_max_distinct_counts_proof_l1555_155583


namespace NUMINAMATH_GPT_find_x_l1555_155539

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 135) (h2 : x > 0) : x = 11.25 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1555_155539


namespace NUMINAMATH_GPT_trader_cloth_sold_l1555_155579

variable (x : ℕ)
variable (profit_per_meter total_profit : ℕ)

theorem trader_cloth_sold (h_profit_per_meter : profit_per_meter = 55)
  (h_total_profit : total_profit = 2200) :
  55 * x = 2200 → x = 40 :=
by 
  sorry

end NUMINAMATH_GPT_trader_cloth_sold_l1555_155579


namespace NUMINAMATH_GPT_possible_values_sin_plus_cos_l1555_155588

variable (x : ℝ)

theorem possible_values_sin_plus_cos (h : 2 * Real.cos x - 3 * Real.sin x = 2) :
    ∃ (values : Set ℝ), values = {3, -31 / 13} ∧ (Real.sin x + 3 * Real.cos x) ∈ values := by
  sorry

end NUMINAMATH_GPT_possible_values_sin_plus_cos_l1555_155588


namespace NUMINAMATH_GPT_find_box_depth_l1555_155553

-- Definitions and conditions
noncomputable def length : ℝ := 1.6
noncomputable def width : ℝ := 1.0
noncomputable def edge : ℝ := 0.2
noncomputable def number_of_blocks : ℝ := 120

-- The goal is to find the depth of the box
theorem find_box_depth (d : ℝ) :
  length * width * d = number_of_blocks * (edge ^ 3) →
  d = 0.6 := 
sorry

end NUMINAMATH_GPT_find_box_depth_l1555_155553


namespace NUMINAMATH_GPT_problem1_problem2_l1555_155516

-- Problem 1
theorem problem1 : 2023^2 - 2024 * 2022 = 1 :=
sorry

-- Problem 2
variables (a b c : ℝ)
theorem problem2 : 5 * a^2 * b^3 * (-1/10 * a * b^3 * c) / (1/2 * a * b^2)^3 = -4 * c :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1555_155516


namespace NUMINAMATH_GPT_largest_of_three_numbers_l1555_155591

noncomputable def hcf := 23
noncomputable def factors := [11, 12, 13]

/-- The largest of the three numbers, given the H.C.F is 23 and the other factors of their L.C.M are 11, 12, and 13, is 39468. -/
theorem largest_of_three_numbers : hcf * factors.prod = 39468 := by
  sorry

end NUMINAMATH_GPT_largest_of_three_numbers_l1555_155591


namespace NUMINAMATH_GPT_proof_f_2017_l1555_155538

-- Define the conditions provided in the problem
variable (f : ℝ → ℝ)
variable (hf : ∀ x, f (-x) = -f x) -- f is an odd function
variable (h1 : ∀ x, f (-x + 1) = f (x + 1))
variable (h2 : f (-1) = 1)

-- Define the Lean statement that proves the correct answer
theorem proof_f_2017 : f 2017 = -1 :=
sorry

end NUMINAMATH_GPT_proof_f_2017_l1555_155538


namespace NUMINAMATH_GPT_money_raised_is_correct_l1555_155543

noncomputable def total_money_raised : ℝ :=
  let ticket_sales := 120 * 2.50 + 80 * 4.50 + 40 * 8.00 + 15 * 14.00
  let donations := 3 * 20.00 + 2 * 55.00 + 75.00 + 95.00 + 150.00
  ticket_sales + donations

theorem money_raised_is_correct :
  total_money_raised = 1680 := by
  sorry

end NUMINAMATH_GPT_money_raised_is_correct_l1555_155543


namespace NUMINAMATH_GPT_solution_set_inequality_l1555_155571

theorem solution_set_inequality (x : ℝ) : 2 * x^2 - x - 3 > 0 ↔ x > 3 / 2 ∨ x < -1 := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1555_155571


namespace NUMINAMATH_GPT_tenth_term_of_arithmetic_sequence_l1555_155507

theorem tenth_term_of_arithmetic_sequence :
  ∃ a : ℕ → ℤ, (∀ n : ℕ, a n + 1 - a n = 2) ∧ a 1 = 1 ∧ a 10 = 19 :=
sorry

end NUMINAMATH_GPT_tenth_term_of_arithmetic_sequence_l1555_155507


namespace NUMINAMATH_GPT_sqrt_9_eq_3_or_neg3_l1555_155597

theorem sqrt_9_eq_3_or_neg3 :
  { x : ℝ | x^2 = 9 } = {3, -3} :=
sorry

end NUMINAMATH_GPT_sqrt_9_eq_3_or_neg3_l1555_155597


namespace NUMINAMATH_GPT_find_a10_l1555_155513

noncomputable def ladder_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), (a (n + 3))^2 = a n * a (n + 6)

theorem find_a10 {a : ℕ → ℝ} (h1 : ladder_geometric_sequence a) 
(h2 : a 1 = 1) 
(h3 : a 4 = 2) : a 10 = 8 :=
sorry

end NUMINAMATH_GPT_find_a10_l1555_155513


namespace NUMINAMATH_GPT_div_neg_cancel_neg_div_example_l1555_155570

theorem div_neg_cancel (x y : Int) (h : y ≠ 0) : (-x) / (-y) = x / y := by
  sorry

theorem neg_div_example : (-64 : Int) / (-32) = 2 := by
  apply div_neg_cancel
  norm_num

end NUMINAMATH_GPT_div_neg_cancel_neg_div_example_l1555_155570


namespace NUMINAMATH_GPT_ratio_of_dividends_l1555_155569

-- Definitions based on conditions
def expected_earnings : ℝ := 0.80
def actual_earnings : ℝ := 1.10
def additional_per_increment : ℝ := 0.04
def increment_size : ℝ := 0.10

-- Definition for the base dividend D which remains undetermined
variable (D : ℝ)

-- Stating the theorem
theorem ratio_of_dividends 
  (h1 : actual_earnings = 1.10)
  (h2 : expected_earnings = 0.80)
  (h3 : additional_per_increment = 0.04)
  (h4 : increment_size = 0.10) :
  let additional_earnings := actual_earnings - expected_earnings
  let increments := additional_earnings / increment_size
  let additional_dividend := increments * additional_per_increment
  let total_dividend := D + additional_dividend
  let ratio := total_dividend / actual_earnings
  ratio = (D + 0.12) / 1.10 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_dividends_l1555_155569
