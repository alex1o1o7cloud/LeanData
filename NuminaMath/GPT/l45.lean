import Mathlib

namespace NUMINAMATH_GPT_find_first_offset_l45_4591

theorem find_first_offset (d b : ℝ) (Area : ℝ) :
  d = 22 → b = 6 → Area = 165 → (first_offset : ℝ) → 22 * (first_offset + 6) / 2 = 165 → first_offset = 9 :=
by
  intros hd hb hArea first_offset heq
  sorry

end NUMINAMATH_GPT_find_first_offset_l45_4591


namespace NUMINAMATH_GPT_find_vector_at_t4_l45_4551

def vector_at (t : ℝ) (a d : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := a
  let (dx, dy, dz) := d
  (x + t * dx, y + t * dy, z + t * dz)

theorem find_vector_at_t4 :
  ∀ (a d : ℝ × ℝ × ℝ),
    vector_at (-2) a d = (2, 6, 16) →
    vector_at 1 a d = (-1, -5, -10) →
    vector_at 4 a d = (-16, -60, -140) :=
by
  intros a d h1 h2
  sorry

end NUMINAMATH_GPT_find_vector_at_t4_l45_4551


namespace NUMINAMATH_GPT_perimeter_of_one_rectangle_l45_4555

theorem perimeter_of_one_rectangle (s : ℝ) (rectangle_perimeter rectangle_length rectangle_width : ℝ) (h1 : 4 * s = 240) (h2 : rectangle_width = (1/2) * s) (h3 : rectangle_length = s) (h4 : rectangle_perimeter = 2 * (rectangle_length + rectangle_width)) :
  rectangle_perimeter = 180 := 
sorry

end NUMINAMATH_GPT_perimeter_of_one_rectangle_l45_4555


namespace NUMINAMATH_GPT_sum_of_four_digit_multiples_of_5_l45_4521

theorem sum_of_four_digit_multiples_of_5 :
  let a := 1000
  let l := 9995
  let d := 5
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  S = 9895500 :=
by
  let a := 1000
  let l := 9995
  let d := 5
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  sorry

end NUMINAMATH_GPT_sum_of_four_digit_multiples_of_5_l45_4521


namespace NUMINAMATH_GPT_correct_statement_is_D_l45_4596

/-
Given the following statements and their conditions:
A: Conducting a comprehensive survey is not an accurate approach to understand the sleep situation of middle school students in Changsha.
B: The mode of the dataset \(-1\), \(2\), \(5\), \(5\), \(7\), \(7\), \(4\) is not \(7\) only, because both \(5\) and \(7\) are modes.
C: A probability of precipitation of \(90\%\) does not guarantee it will rain tomorrow.
D: If two datasets, A and B, have the same mean, and the variances \(s_{A}^{2} = 0.3\) and \(s_{B}^{2} = 0.02\), then set B with a lower variance \(s_{B}^{2}\) is more stable.

Prove that the correct statement based on these conditions is D.
-/
theorem correct_statement_is_D
  (dataset_A dataset_B : Type)
  (mean_A mean_B : ℝ)
  (sA2 sB2 : ℝ)
  (h_same_mean: mean_A = mean_B)
  (h_variances: sA2 = 0.3 ∧ sB2 = 0.02)
  (h_stability: sA2 > sB2) :
  (if sA2 = 0.3 ∧ sB2 = 0.02 ∧ sA2 > sB2 then "D" else "not D") = "D" := by
  sorry

end NUMINAMATH_GPT_correct_statement_is_D_l45_4596


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l45_4587

theorem no_positive_integer_solutions (x y : ℕ) (hx : x > 0) (hy : y > 0) : x^5 ≠ y^2 + 4 := 
by sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l45_4587


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l45_4578

-- Definitions of the lines
def l1 (a : ℝ) : ℝ × ℝ × ℝ :=
  (2 * a + 1, a + 2, 3)

def l2 (a : ℝ) : ℝ × ℝ × ℝ :=
  (a - 1, -2, 2)

-- Parallel lines condition
def parallel_lines (a : ℝ) : Prop :=
  let (A1, B1, C1) := l1 a
  let (A2, B2, C2) := l2 a
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ (A1 / B1) = (A2 / B2)

-- Perpendicular lines condition
def perpendicular_lines (a : ℝ) : Prop :=
  let (A1, B1, C1) := l1 a
  let (A2, B2, C2) := l2 a
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ (A1 * A2 + B1 * B2 = 0)

-- Statement for part 1
theorem part1_solution (a : ℝ) : parallel_lines a ↔ a = 0 :=
  sorry

-- Statement for part 2
theorem part2_solution (a : ℝ) : perpendicular_lines a ↔ (a = -1 ∨ a = 5 / 2) :=
  sorry


end NUMINAMATH_GPT_part1_solution_part2_solution_l45_4578


namespace NUMINAMATH_GPT_eraser_cost_l45_4519

theorem eraser_cost (initial_money : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) (erasers_count : ℕ) (remaining_money : ℕ) :
    initial_money = 100 →
    scissors_count = 8 →
    scissors_price = 5 →
    erasers_count = 10 →
    remaining_money = 20 →
    (initial_money - scissors_count * scissors_price - remaining_money) / erasers_count = 4 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_eraser_cost_l45_4519


namespace NUMINAMATH_GPT_condition_for_positive_expression_l45_4582

theorem condition_for_positive_expression (a b c : ℝ) :
  (∀ x y : ℝ, x^2 + x * y + y^2 + a * x + b * y + c > 0) ↔ a^2 - a * b + b^2 < 3 * c :=
by
  -- Proof should be provided here
  sorry

end NUMINAMATH_GPT_condition_for_positive_expression_l45_4582


namespace NUMINAMATH_GPT_geom_seq_sum_first_10_terms_l45_4500

variable (a : ℕ → ℝ) (a₁ : ℝ) (q : ℝ)
variable (h₀ : a₁ = 1/4)
variable (h₁ : ∀ n, a (n + 1) = a₁ * q ^ n)
variable (S : ℕ → ℝ)
variable (h₂ : S n = a₁ * (1 - q ^ n) / (1 - q))

theorem geom_seq_sum_first_10_terms :
  a 1 = 1 / 4 →
  (a 3) * (a 5) = 4 * ((a 4) - 1) →
  S 10 = 1023 / 4 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_sum_first_10_terms_l45_4500


namespace NUMINAMATH_GPT_no_cracked_seashells_l45_4517

theorem no_cracked_seashells (tom_seashells : ℕ) (fred_seashells : ℕ) (total_seashells : ℕ)
  (h1 : tom_seashells = 15) (h2 : fred_seashells = 43) (h3 : total_seashells = 58)
  (h4 : tom_seashells + fred_seashells = total_seashells) : 
  (total_seashells - (tom_seashells + fred_seashells) = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_cracked_seashells_l45_4517


namespace NUMINAMATH_GPT_select_integers_divisible_l45_4546

theorem select_integers_divisible (k : ℕ) (s : Finset ℤ) (h₁ : s.card = 2 * 2^k - 1) :
  ∃ t : Finset ℤ, t ⊆ s ∧ t.card = 2^k ∧ (t.sum id) % 2^k = 0 :=
sorry

end NUMINAMATH_GPT_select_integers_divisible_l45_4546


namespace NUMINAMATH_GPT_towels_per_load_l45_4508

-- Defining the given conditions
def total_towels : ℕ := 42
def number_of_loads : ℕ := 6

-- Defining the problem statement: Prove the number of towels per load
theorem towels_per_load : total_towels / number_of_loads = 7 := by 
  sorry

end NUMINAMATH_GPT_towels_per_load_l45_4508


namespace NUMINAMATH_GPT_square_area_in_ellipse_l45_4535

theorem square_area_in_ellipse :
  (∃ t : ℝ, 
    (∀ x y : ℝ, ((x = t ∨ x = -t) ∧ (y = t ∨ y = -t)) → (x^2 / 4 + y^2 / 8 = 1)) 
    ∧ t > 0 
    ∧ ((2 * t)^2 = 32 / 3)) :=
sorry

end NUMINAMATH_GPT_square_area_in_ellipse_l45_4535


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l45_4536

theorem problem1 : 24 - (-16) + (-25) - 32 = -17 := by
  sorry

theorem problem2 : (-1 / 2) * 2 / 2 * (-1 / 2) = 1 / 4 := by
  sorry

theorem problem3 : -2^2 * 5 - (-2)^3 * (1 / 8) + 1 = -18 := by
  sorry

theorem problem4 : ((-1 / 4) - (5 / 6) + (8 / 9)) / (-1 / 6)^2 + (-2)^2 * (-6)= -31 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l45_4536


namespace NUMINAMATH_GPT_cos_thirteen_pi_over_three_l45_4560

theorem cos_thirteen_pi_over_three : Real.cos (13 * Real.pi / 3) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_cos_thirteen_pi_over_three_l45_4560


namespace NUMINAMATH_GPT_find_angle_A_l45_4563

noncomputable def exists_angle_A (A B C : ℝ) (a b : ℝ) : Prop :=
  C = (A + B) / 2 ∧ 
  A + B + C = 180 ∧ 
  (a + b) / 2 = Real.sqrt 3 + 1 ∧ 
  C = 2 * Real.sqrt 2

theorem find_angle_A : ∃ A B C a b, 
  exists_angle_A A B C a b ∧ (A = 75 ∨ A = 45) :=
by
  -- This is where the detailed proof would go
  sorry

end NUMINAMATH_GPT_find_angle_A_l45_4563


namespace NUMINAMATH_GPT_unique_integral_root_l45_4575

theorem unique_integral_root {x : ℤ} :
  x - 12 / (x - 3) = 5 - 12 / (x - 3) ↔ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_unique_integral_root_l45_4575


namespace NUMINAMATH_GPT_abs_value_sum_l45_4503

noncomputable def sin_theta_in_bounds (θ : ℝ) : Prop :=
  -1 ≤ Real.sin θ ∧ Real.sin θ ≤ 1

noncomputable def x_satisfies_log_eq (θ x : ℝ) : Prop :=
  Real.log x / Real.log 3 = 1 + Real.sin θ

theorem abs_value_sum (θ x : ℝ) (h1 : x_satisfies_log_eq θ x) (h2 : sin_theta_in_bounds θ) :
  |x - 1| + |x - 9| = 8 :=
sorry

end NUMINAMATH_GPT_abs_value_sum_l45_4503


namespace NUMINAMATH_GPT_balloon_rearrangements_l45_4594

-- Define the letters involved: vowels and consonants
def vowels := ['A', 'O', 'O', 'O']
def consonants := ['B', 'L', 'L', 'N']

-- State the problem in Lean 4:
theorem balloon_rearrangements : 
  ∃ n : ℕ, 
  (∀ (vowels := ['A', 'O', 'O', 'O']) 
     (consonants := ['B', 'L', 'L', 'N']), 
     n = 32) := sorry  -- we state that the number of rearrangements is 32 but do not provide the proof itself.

end NUMINAMATH_GPT_balloon_rearrangements_l45_4594


namespace NUMINAMATH_GPT_natalie_bushes_to_zucchinis_l45_4597

/-- Each of Natalie's blueberry bushes yields ten containers of blueberries,
    and she trades six containers of blueberries for three zucchinis.
    Given this setup, prove that the number of bushes Natalie needs to pick
    in order to get sixty zucchinis is twelve. --/
theorem natalie_bushes_to_zucchinis :
  (∀ (bush_yield containers_needed : ℕ), bush_yield = 10 ∧ containers_needed = 60 * (6 / 3)) →
  (∀ (containers_total bushes_needed : ℕ), containers_total = 60 * (6 / 3) ∧ bushes_needed = containers_total * (1 / bush_yield)) →
  bushes_needed = 12 :=
by
  sorry

end NUMINAMATH_GPT_natalie_bushes_to_zucchinis_l45_4597


namespace NUMINAMATH_GPT_max_curved_sides_l45_4531

theorem max_curved_sides (n : ℕ) (h : 2 ≤ n) : 
  ∃ m, m = 2 * n - 2 :=
sorry

end NUMINAMATH_GPT_max_curved_sides_l45_4531


namespace NUMINAMATH_GPT_parametric_eq_of_curve_C_max_x_plus_y_on_curve_C_l45_4525

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := 
(2 + Real.sqrt 2 * Real.cos θ, 
 2 + Real.sqrt 2 * Real.sin θ)

theorem parametric_eq_of_curve_C (θ : ℝ) : 
    ∃ x y, 
    (x, y) = curve_C θ ∧ 
    (x - 2)^2 + (y - 2)^2 = 2 := by sorry

theorem max_x_plus_y_on_curve_C :
    ∃ x y θ, 
    (x, y) = curve_C θ ∧ 
    (∀ p : ℝ × ℝ, (p.1, p.2) = curve_C θ → 
    p.1 + p.2 ≤ 6) ∧
    x + y = 6 ∧
    x = 3 ∧ 
    y = 3 := by sorry

end NUMINAMATH_GPT_parametric_eq_of_curve_C_max_x_plus_y_on_curve_C_l45_4525


namespace NUMINAMATH_GPT_sculpture_height_is_34_inches_l45_4540

-- Define the height of the base in inches
def height_of_base_in_inches : ℕ := 2

-- Define the total height in feet
def total_height_in_feet : ℕ := 3

-- Convert feet to inches (1 foot = 12 inches)
def total_height_in_inches (feet : ℕ) : ℕ := feet * 12

-- The height of the sculpture, given the base and total height
def height_of_sculpture (total_height base_height : ℕ) : ℕ := total_height - base_height

-- State the theorem that the height of the sculpture is 34 inches
theorem sculpture_height_is_34_inches :
  height_of_sculpture (total_height_in_inches total_height_in_feet) height_of_base_in_inches = 34 := by
  sorry

end NUMINAMATH_GPT_sculpture_height_is_34_inches_l45_4540


namespace NUMINAMATH_GPT_david_pushups_more_than_zachary_l45_4576

theorem david_pushups_more_than_zachary :
  ∀ (Z D J : ℕ), Z = 51 → J = 69 → J = D - 4 → D = Z + 22 :=
by
  intros Z D J hZ hJ hJD
  sorry

end NUMINAMATH_GPT_david_pushups_more_than_zachary_l45_4576


namespace NUMINAMATH_GPT_range_of_a_l45_4553

theorem range_of_a :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → (0 < a ∧ a < 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_range_of_a_l45_4553


namespace NUMINAMATH_GPT_train_late_average_speed_l45_4527

theorem train_late_average_speed 
  (distance : ℝ) (on_time_speed : ℝ) (late_time_additional : ℝ) 
  (on_time : distance / on_time_speed = 1.75) 
  (late : distance / (on_time_speed * 2/2.5) = 2) :
  distance / 2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_train_late_average_speed_l45_4527


namespace NUMINAMATH_GPT_find_a_for_opposite_roots_l45_4501

-- Define the equation and condition using the given problem details
theorem find_a_for_opposite_roots (a : ℝ) 
  (h : ∀ (x : ℝ), x^2 - (a^2 - 2 * a - 15) * x + a - 1 = 0 
    → (∃! (x1 x2 : ℝ), x1 + x2 = 0)) :
  a = -3 := 
sorry

end NUMINAMATH_GPT_find_a_for_opposite_roots_l45_4501


namespace NUMINAMATH_GPT_john_initial_clean_jerk_weight_l45_4568

def initial_snatch_weight : ℝ := 50
def increase_rate : ℝ := 1.8
def total_new_lifting_capacity : ℝ := 250

theorem john_initial_clean_jerk_weight :
  ∃ (C : ℝ), 2 * C + (increase_rate * initial_snatch_weight) = total_new_lifting_capacity ∧ C = 80 := by
  sorry

end NUMINAMATH_GPT_john_initial_clean_jerk_weight_l45_4568


namespace NUMINAMATH_GPT_units_digit_17_pow_53_l45_4584

theorem units_digit_17_pow_53 : (17^53) % 10 = 7 := 
by sorry

end NUMINAMATH_GPT_units_digit_17_pow_53_l45_4584


namespace NUMINAMATH_GPT_exist_positive_integers_summing_to_one_l45_4547

theorem exist_positive_integers_summing_to_one :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (1 / (x:ℚ) + 1 / (y:ℚ) + 1 / (z:ℚ) = 1)
    ∧ ((x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 3 ∧ y = 3 ∧ z = 3)) :=
by
  sorry

end NUMINAMATH_GPT_exist_positive_integers_summing_to_one_l45_4547


namespace NUMINAMATH_GPT_problem_domains_equal_l45_4581

/-- Proof problem:
    Prove that the domain of the function y = (x - 1)^(-1/2) is equal to the domain of the function y = ln(x - 1).
--/
theorem problem_domains_equal :
  {x : ℝ | x > 1} = {x : ℝ | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_problem_domains_equal_l45_4581


namespace NUMINAMATH_GPT_drawing_red_ball_is_certain_l45_4589

def certain_event (balls : List String) : Prop :=
  ∀ ball ∈ balls, ball = "red"

theorem drawing_red_ball_is_certain:
  certain_event ["red", "red", "red", "red", "red"] :=
by
  sorry

end NUMINAMATH_GPT_drawing_red_ball_is_certain_l45_4589


namespace NUMINAMATH_GPT_initial_bottles_proof_l45_4590

-- Define the conditions as variables and statements
def initial_bottles (X : ℕ) : Prop :=
X - 8 + 45 = 51

-- Theorem stating the proof problem
theorem initial_bottles_proof : initial_bottles 14 :=
by
  -- We need to prove the following:
  -- 14 - 8 + 45 = 51
  sorry

end NUMINAMATH_GPT_initial_bottles_proof_l45_4590


namespace NUMINAMATH_GPT_divisible_by_1989_l45_4572

theorem divisible_by_1989 (n : ℕ) : 
  1989 ∣ (13 * (-50)^n + 17 * 40^n - 30) :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_1989_l45_4572


namespace NUMINAMATH_GPT_direct_proportion_b_zero_l45_4518

theorem direct_proportion_b_zero (b : ℝ) (x y : ℝ) 
  (h : ∀ x, y = x + b → ∃ k, y = k * x) : b = 0 :=
sorry

end NUMINAMATH_GPT_direct_proportion_b_zero_l45_4518


namespace NUMINAMATH_GPT_game_points_product_l45_4544

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2, 6]
def betty_rolls : List ℕ := [6, 3, 3, 2, 1]

def calculate_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_points_product :
  calculate_points allie_rolls * calculate_points betty_rolls = 702 :=
by
  sorry

end NUMINAMATH_GPT_game_points_product_l45_4544


namespace NUMINAMATH_GPT_circle_reflection_l45_4593

-- Definition of the original center of the circle
def original_center : ℝ × ℝ := (8, -3)

-- Definition of the reflection transformation over the line y = x
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Theorem stating that reflecting the original center over the line y = x results in a specific point
theorem circle_reflection : reflect original_center = (-3, 8) :=
  by
  -- skipping the proof part
  sorry

end NUMINAMATH_GPT_circle_reflection_l45_4593


namespace NUMINAMATH_GPT_find_volume_of_pure_alcohol_l45_4585

variable (V1 Vf V2 : ℝ)
variable (P1 Pf : ℝ)

theorem find_volume_of_pure_alcohol
  (h : V2 = Vf * Pf / 100 - V1 * P1 / 100) : 
  V2 = Vf * (Pf / 100) - V1 * (P1 / 100) :=
by
  -- This is the theorem statement. The proof is omitted.
  sorry

end NUMINAMATH_GPT_find_volume_of_pure_alcohol_l45_4585


namespace NUMINAMATH_GPT_algebraic_expr_value_at_neg_one_l45_4524

-- Define the expression "3 times the square of x minus 5"
def algebraic_expr (x : ℝ) : ℝ := 3 * x^2 + 5

-- Theorem to state the value when x = -1 is 8
theorem algebraic_expr_value_at_neg_one : algebraic_expr (-1) = 8 := 
by
  -- The steps to prove are skipped with 'sorry'
  sorry

end NUMINAMATH_GPT_algebraic_expr_value_at_neg_one_l45_4524


namespace NUMINAMATH_GPT_convert_seven_cubic_yards_l45_4541

-- Define the conversion factor from yards to feet
def yardToFeet : ℝ := 3
-- Define the conversion factor from cubic yards to cubic feet
def cubicYardToCubicFeet : ℝ := yardToFeet ^ 3
-- Define the conversion function from cubic yards to cubic feet
noncomputable def convertVolume (volumeInCubicYards : ℝ) : ℝ :=
  volumeInCubicYards * cubicYardToCubicFeet

-- Statement to prove: 7 cubic yards is equivalent to 189 cubic feet
theorem convert_seven_cubic_yards : convertVolume 7 = 189 := by
  sorry

end NUMINAMATH_GPT_convert_seven_cubic_yards_l45_4541


namespace NUMINAMATH_GPT_leakage_empty_time_l45_4515

variables (a : ℝ) (h1 : a > 0) -- Assuming a is positive for the purposes of the problem

theorem leakage_empty_time (h : 7 * a > 0) : (7 * a) / 6 = 7 * a / 6 :=
by
  sorry

end NUMINAMATH_GPT_leakage_empty_time_l45_4515


namespace NUMINAMATH_GPT_inches_per_foot_l45_4554

-- Definition of the conditions in the problem.
def feet_last_week := 6
def feet_less_this_week := 4
def total_inches := 96

-- Lean statement that proves the number of inches in a foot
theorem inches_per_foot : 
    (total_inches / (feet_last_week + (feet_last_week - feet_less_this_week))) = 12 := 
by sorry

end NUMINAMATH_GPT_inches_per_foot_l45_4554


namespace NUMINAMATH_GPT_sin_alpha_cos_alpha_l45_4537

theorem sin_alpha_cos_alpha {α : ℝ} (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) :
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_cos_alpha_l45_4537


namespace NUMINAMATH_GPT_remainder_of_3024_l45_4559

theorem remainder_of_3024 (M : ℤ) (hM1 : M = 3024) (h_condition : ∃ k : ℤ, M = 24 * k + 13) :
  M % 1821 = 1203 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_3024_l45_4559


namespace NUMINAMATH_GPT_number_of_rocks_tossed_l45_4516

-- Conditions
def pebbles : ℕ := 6
def rocks : ℕ := 3
def boulders : ℕ := 2
def pebble_splash : ℚ := 1 / 4
def rock_splash : ℚ := 1 / 2
def boulder_splash : ℚ := 2

-- Total width of the splashes
def total_splash (R : ℕ) : ℚ := 
  pebbles * pebble_splash + R * rock_splash + boulders * boulder_splash

-- Given condition
def total_splash_condition : ℚ := 7

theorem number_of_rocks_tossed : 
  total_splash rocks = total_splash_condition → rocks = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_rocks_tossed_l45_4516


namespace NUMINAMATH_GPT_minimum_ab_value_is_two_l45_4505

noncomputable def minimum_value_ab (a b : ℝ) (h1 : a^2 ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b = a^2 + 1) : ℝ :=
|a * b|

theorem minimum_ab_value_is_two (a b : ℝ) (h1 : a^2 ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b = a^2 + 1) : minimum_value_ab a b h1 h2 h3 = 2 := by
  sorry

end NUMINAMATH_GPT_minimum_ab_value_is_two_l45_4505


namespace NUMINAMATH_GPT_total_seashells_found_l45_4534

-- Defining the conditions
def joan_daily_seashells : ℕ := 6
def jessica_daily_seashells : ℕ := 8
def length_of_vacation : ℕ := 7

-- Stating the theorem
theorem total_seashells_found : 
  (joan_daily_seashells + jessica_daily_seashells) * length_of_vacation = 98 :=
by
  sorry

end NUMINAMATH_GPT_total_seashells_found_l45_4534


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l45_4528

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (hx : x = 2)
    (ha : a = (x, 1)) (hb : b = (4, x)) : 
    (∃ k : ℝ, a = (k * b.1, k * b.2)) ∧ (¬ (∀ k : ℝ, a = (k * b.1, k * b.2))) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l45_4528


namespace NUMINAMATH_GPT_probability_alpha_in_interval_l45_4577

def vector_of_die_rolls_angle_probability : ℚ := 
  let total_outcomes := 36
  let favorable_pairs := 15
  favorable_pairs / total_outcomes

theorem probability_alpha_in_interval (m n : ℕ)
  (hm : 1 ≤ m ∧ m ≤ 6) (hn : 1 ≤ n ∧ n ≤ 6) :
  (vector_of_die_rolls_angle_probability = 5 / 12) := by
  sorry

end NUMINAMATH_GPT_probability_alpha_in_interval_l45_4577


namespace NUMINAMATH_GPT_calc_expression_solve_equation_l45_4583

-- Problem 1: Calculation

theorem calc_expression : 
  |Real.sqrt 3 - 2| + Real.sqrt 12 - 6 * Real.sin (Real.pi / 6) + (-1/2 : Real)⁻¹ = Real.sqrt 3 - 3 := 
by {
  sorry
}

-- Problem 2: Solve the Equation

theorem solve_equation (x : Real) : 
  x * (x + 6) = -5 ↔ (x = -5 ∨ x = -1) := 
by {
  sorry
}

end NUMINAMATH_GPT_calc_expression_solve_equation_l45_4583


namespace NUMINAMATH_GPT_find_y_values_l45_4512

theorem find_y_values
  (y₁ y₂ y₃ y₄ y₅ : ℝ)
  (h₁ : y₁ + 3 * y₂ + 6 * y₃ + 10 * y₄ + 15 * y₅ = 3)
  (h₂ : 3 * y₁ + 6 * y₂ + 10 * y₃ + 15 * y₄ + 21 * y₅ = 20)
  (h₃ : 6 * y₁ + 10 * y₂ + 15 * y₃ + 21 * y₄ + 28 * y₅ = 86)
  (h₄ : 10 * y₁ + 15 * y₂ + 21 * y₃ + 28 * y₄ + 36 * y₅ = 225) :
  15 * y₁ + 21 * y₂ + 28 * y₃ + 36 * y₄ + 45 * y₅ = 395 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_values_l45_4512


namespace NUMINAMATH_GPT_detergent_per_pound_l45_4502

-- Define the conditions
def total_ounces_detergent := 18
def total_pounds_clothes := 9

-- Define the question to prove the amount of detergent per pound of clothes
theorem detergent_per_pound : total_ounces_detergent / total_pounds_clothes = 2 := by
  sorry

end NUMINAMATH_GPT_detergent_per_pound_l45_4502


namespace NUMINAMATH_GPT_original_denominator_value_l45_4545

theorem original_denominator_value (d : ℕ) (h1 : 3 + 3 = 6) (h2 : ((6 : ℕ) / (d + 3 : ℕ) = (1 / 3 : ℚ))) : d = 15 :=
sorry

end NUMINAMATH_GPT_original_denominator_value_l45_4545


namespace NUMINAMATH_GPT_impossible_arrangement_l45_4595

theorem impossible_arrangement : 
  ∀ (a : Fin 111 → ℕ), (∀ i, a i ≤ 500) → (∀ i j, i ≠ j → a i ≠ a j) → 
  ¬ ∀ i : Fin 111, (a i % 10 = ((Finset.univ.sum (λ j => if j = i then 0 else a j)) % 10)) :=
by 
  sorry

end NUMINAMATH_GPT_impossible_arrangement_l45_4595


namespace NUMINAMATH_GPT_green_balls_in_bag_l45_4567

theorem green_balls_in_bag (b : ℕ) (P_blue : ℚ) (g : ℕ) (h1 : b = 8) (h2 : P_blue = 1 / 3) (h3 : P_blue = (b : ℚ) / (b + g)) :
  g = 16 :=
by
  sorry

end NUMINAMATH_GPT_green_balls_in_bag_l45_4567


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l45_4507

variables (α β : Plane) (m : Line)

-- Define what it means for planes and lines to be perpendicular
def plane_perpendicular (p1 p2 : Plane) : Prop := sorry
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry

-- The main theorem to be established
theorem necessary_but_not_sufficient :
  (plane_perpendicular α β) → (line_perpendicular_plane m β) ∧ ¬ ((plane_perpendicular α β) ↔ (line_perpendicular_plane m β)) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l45_4507


namespace NUMINAMATH_GPT_inequality_abc_l45_4514

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b - c) * (b + c - a) * (c + a - b) ≤ a * b * c := 
sorry

end NUMINAMATH_GPT_inequality_abc_l45_4514


namespace NUMINAMATH_GPT_centroid_calculation_correct_l45_4504

-- Define the vertices of the triangle
def P : ℝ × ℝ := (2, 3)
def Q : ℝ × ℝ := (-1, 4)
def R : ℝ × ℝ := (4, -2)

-- Define the coordinates of the centroid
noncomputable def S : ℝ × ℝ := ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

-- Prove that 7x + 2y = 15 for the centroid
theorem centroid_calculation_correct : 7 * S.1 + 2 * S.2 = 15 :=
by 
  -- Placeholder for the proof steps
  sorry

end NUMINAMATH_GPT_centroid_calculation_correct_l45_4504


namespace NUMINAMATH_GPT_yolanda_walking_rate_l45_4579

-- Definitions for the conditions given in the problem
variables (X Y : ℝ) -- Points X and Y
def distance_X_to_Y := 52 -- Distance between X and Y in miles
def Bob_rate := 4 -- Bob's walking rate in miles per hour
def Bob_distance_walked := 28 -- The distance Bob walked in miles
def start_time_diff := 1 -- The time difference (in hours) between Yolanda and Bob starting

-- The statement to prove
theorem yolanda_walking_rate : 
  ∃ (y : ℝ), (distance_X_to_Y = y * (Bob_distance_walked / Bob_rate + start_time_diff) + Bob_distance_walked) ∧ y = 3 := by 
  sorry

end NUMINAMATH_GPT_yolanda_walking_rate_l45_4579


namespace NUMINAMATH_GPT_maximise_expression_l45_4523

theorem maximise_expression {x : ℝ} (hx : 0 < x ∧ x < 1) : 
  ∃ (x_max : ℝ), x_max = 1/2 ∧ 
  (∀ y : ℝ, (0 < y ∧ y < 1) → 3 * y * (1 - y) ≤ 3 * x_max * (1 - x_max)) :=
sorry

end NUMINAMATH_GPT_maximise_expression_l45_4523


namespace NUMINAMATH_GPT_calculate_f_at_2_l45_4558

noncomputable def f (x : ℝ) : ℝ := sorry

theorem calculate_f_at_2 :
  (∀ x : ℝ, 25 * f (x / 1580) + (3 - Real.sqrt 34) * f (1580 / x) = 2017 * x) →
  f 2 = 265572 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_calculate_f_at_2_l45_4558


namespace NUMINAMATH_GPT_function_eq_l45_4569

noncomputable def f (x : ℝ) : ℝ := x^4 - 2

theorem function_eq (f : ℝ → ℝ) (h1 : ∀ x : ℝ, deriv f x = 4 * x^3) (h2 : f 1 = -1) : 
  ∀ x : ℝ, f x = x^4 - 2 :=
by
  intro x
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_function_eq_l45_4569


namespace NUMINAMATH_GPT_Riley_fewer_pairs_l45_4539

-- Define the conditions
def Ellie_pairs : ℕ := 8
def Total_pairs : ℕ := 13

-- Prove the statement
theorem Riley_fewer_pairs : (Total_pairs - Ellie_pairs) - Ellie_pairs = 3 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_Riley_fewer_pairs_l45_4539


namespace NUMINAMATH_GPT_problem_statement_l45_4548

def has_arithmetic_square_root (x : ℝ) : Prop :=
  ∃ y : ℝ, y * y = x

theorem problem_statement :
  (¬ has_arithmetic_square_root (-abs 9)) ∧
  (has_arithmetic_square_root ((-1/4)^2)) ∧
  (has_arithmetic_square_root 0) ∧
  (has_arithmetic_square_root (10^2)) := 
sorry

end NUMINAMATH_GPT_problem_statement_l45_4548


namespace NUMINAMATH_GPT_total_price_is_correct_l45_4522

def total_price_of_hats (total_hats : ℕ) (blue_hat_cost green_hat_cost : ℕ) (num_green_hats : ℕ) : ℕ :=
  let num_blue_hats := total_hats - num_green_hats
  let cost_green_hats := num_green_hats * green_hat_cost
  let cost_blue_hats := num_blue_hats * blue_hat_cost
  cost_green_hats + cost_blue_hats

theorem total_price_is_correct : total_price_of_hats 85 6 7 40 = 550 := 
  sorry

end NUMINAMATH_GPT_total_price_is_correct_l45_4522


namespace NUMINAMATH_GPT_inequality_x2_y2_l45_4561

theorem inequality_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  |x^2 + y^2| / (x + y) < |x^2 - y^2| / (x - y) :=
sorry

end NUMINAMATH_GPT_inequality_x2_y2_l45_4561


namespace NUMINAMATH_GPT_total_distance_flash_runs_l45_4509

-- Define the problem with given conditions
theorem total_distance_flash_runs (v k d a : ℝ) (hk : k > 1) : 
  let t := d / (v * (k - 1))
  let distance_to_catch_ace := k * v * t
  let total_distance := distance_to_catch_ace + a
  total_distance = (k * d) / (k - 1) + a := 
by
  sorry

end NUMINAMATH_GPT_total_distance_flash_runs_l45_4509


namespace NUMINAMATH_GPT_trivia_team_original_members_l45_4592

theorem trivia_team_original_members (x : ℕ) (h1 : 6 * (x - 2) = 18) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_trivia_team_original_members_l45_4592


namespace NUMINAMATH_GPT_fraction_to_decimal_l45_4526

theorem fraction_to_decimal :
  (11:ℚ) / 16 = 0.6875 :=
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l45_4526


namespace NUMINAMATH_GPT_jimmy_more_sheets_than_tommy_l45_4549

theorem jimmy_more_sheets_than_tommy 
  (jimmy_initial_sheets : ℕ)
  (tommy_initial_sheets : ℕ)
  (additional_sheets : ℕ)
  (h1 : tommy_initial_sheets = jimmy_initial_sheets + 25)
  (h2 : jimmy_initial_sheets = 58)
  (h3 : additional_sheets = 85) :
  (jimmy_initial_sheets + additional_sheets) - tommy_initial_sheets = 60 := 
by
  sorry

end NUMINAMATH_GPT_jimmy_more_sheets_than_tommy_l45_4549


namespace NUMINAMATH_GPT_probability_two_even_balls_l45_4510

theorem probability_two_even_balls
  (total_balls : ℕ)
  (even_balls : ℕ)
  (h_total : total_balls = 16)
  (h_even : even_balls = 8)
  (first_draw : ℕ → ℚ)
  (second_draw : ℕ → ℚ)
  (h_first : first_draw even_balls = even_balls / total_balls)
  (h_second : second_draw (even_balls - 1) = (even_balls - 1) / (total_balls - 1)) :
  (first_draw even_balls) * (second_draw (even_balls - 1)) = 7 / 30 := 
sorry

end NUMINAMATH_GPT_probability_two_even_balls_l45_4510


namespace NUMINAMATH_GPT_percentage_vets_recommend_puppy_kibble_l45_4571

theorem percentage_vets_recommend_puppy_kibble :
  ∀ (P : ℝ), (30 / 100 * 1000 = 300) → (1000 * P / 100 + 100 = 300) → P = 20 :=
by
  intros P h1 h2
  sorry

end NUMINAMATH_GPT_percentage_vets_recommend_puppy_kibble_l45_4571


namespace NUMINAMATH_GPT_min_value_l45_4562

open Real

noncomputable def func (x y z : ℝ) : ℝ := 1 / x + 1 / y + 1 / z

theorem min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) :
  func x y z ≥ 4.5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_l45_4562


namespace NUMINAMATH_GPT_radius_of_ball_l45_4599

theorem radius_of_ball (diameter depth : ℝ) (h₁ : diameter = 30) (h₂ : depth = 10) : 
  ∃ r : ℝ, r = 25 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_ball_l45_4599


namespace NUMINAMATH_GPT_x_squared_minus_y_squared_l45_4574

theorem x_squared_minus_y_squared
    (x y : ℚ) 
    (h1 : x + y = 3 / 8) 
    (h2 : x - y = 1 / 4) : x^2 - y^2 = 3 / 32 := 
by 
    sorry

end NUMINAMATH_GPT_x_squared_minus_y_squared_l45_4574


namespace NUMINAMATH_GPT_same_face_probability_correct_l45_4511

-- Define the number of sides on the dice
def sides_20 := 20
def sides_16 := 16

-- Define the number of colored sides for each dice category
def maroon_20 := 5
def teal_20 := 8
def cyan_20 := 6
def sparkly_20 := 1

def maroon_16 := 4
def teal_16 := 6
def cyan_16 := 5
def sparkly_16 := 1

-- Define the probabilities of each color matching
def prob_maroon : ℚ := (maroon_20 / sides_20) * (maroon_16 / sides_16)
def prob_teal : ℚ := (teal_20 / sides_20) * (teal_16 / sides_16)
def prob_cyan : ℚ := (cyan_20 / sides_20) * (cyan_16 / sides_16)
def prob_sparkly : ℚ := (sparkly_20 / sides_20) * (sparkly_16 / sides_16)

-- Define the total probability of same face
def prob_same_face := prob_maroon + prob_teal + prob_cyan + prob_sparkly

-- The theorem we need to prove
theorem same_face_probability_correct : 
  prob_same_face = 99 / 320 :=
by
  sorry

end NUMINAMATH_GPT_same_face_probability_correct_l45_4511


namespace NUMINAMATH_GPT_bagel_pieces_after_10_cuts_l45_4552

def bagel_pieces_after_cuts (initial_pieces : ℕ) (cuts : ℕ) : ℕ :=
  initial_pieces + cuts

theorem bagel_pieces_after_10_cuts : bagel_pieces_after_cuts 1 10 = 11 := by
  sorry

end NUMINAMATH_GPT_bagel_pieces_after_10_cuts_l45_4552


namespace NUMINAMATH_GPT_runway_show_time_correct_l45_4598

def runwayShowTime (bathing_suit_sets evening_wear_sets formal_wear_sets models trip_time_in_minutes : ℕ) : ℕ :=
  let trips_per_model := bathing_suit_sets + evening_wear_sets + formal_wear_sets
  let total_trips := models * trips_per_model
  total_trips * trip_time_in_minutes

theorem runway_show_time_correct :
  runwayShowTime 3 4 2 10 3 = 270 :=
by
  sorry

end NUMINAMATH_GPT_runway_show_time_correct_l45_4598


namespace NUMINAMATH_GPT_polygon_sides_l45_4529

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 + 360 = 1800) : n = 10 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_l45_4529


namespace NUMINAMATH_GPT_tens_digit_2023_pow_2024_minus_2025_l45_4538

theorem tens_digit_2023_pow_2024_minus_2025 :
  (2023 ^ 2024 - 2025) % 100 / 10 % 10 = 5 :=
sorry

end NUMINAMATH_GPT_tens_digit_2023_pow_2024_minus_2025_l45_4538


namespace NUMINAMATH_GPT_max_profit_max_profit_price_l45_4586

-- Definitions based on the conditions
def purchase_price : ℝ := 80
def initial_selling_price : ℝ := 120
def initial_sales : ℕ := 20
def extra_sales_per_unit_decrease : ℕ := 2
def cost_price_constraint (x : ℝ) : Prop := 0 < x ∧ x ≤ 40

-- Expression for the profit function
def profit_function (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

-- Prove the maximum profit given the conditions
theorem max_profit : ∃ x : ℝ, cost_price_constraint x ∧ profit_function x = 1250 :=
by
  sorry

-- Proving that the selling price for max profit is 105 yuan
theorem max_profit_price : ∃ x : ℝ, cost_price_constraint x ∧ profit_function x = 1250 ∧ (initial_selling_price - x) = 105 :=
by
  sorry

end NUMINAMATH_GPT_max_profit_max_profit_price_l45_4586


namespace NUMINAMATH_GPT_t_n_closed_form_t_2022_last_digit_l45_4565

noncomputable def t_n (n : ℕ) : ℕ :=
  (4^n - 3 * 3^n + 3 * 2^n - 1) / 6

theorem t_n_closed_form (n : ℕ) (hn : 0 < n) :
  t_n n = (4^n - 3 * 3^n + 3 * 2^n - 1) / 6 :=
by
  sorry

theorem t_2022_last_digit :
  (t_n 2022) % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_t_n_closed_form_t_2022_last_digit_l45_4565


namespace NUMINAMATH_GPT_largest_prime_divisor_of_1202102_5_l45_4532

def base_5_to_decimal (n : String) : ℕ := 
  let digits := n.toList.map (λ c => c.toNat - '0'.toNat)
  digits.foldr (λ (digit acc : ℕ) => acc * 5 + digit) 0

def largest_prime_factor (n : ℕ) : ℕ := sorry -- Placeholder for the actual factorization logic.

theorem largest_prime_divisor_of_1202102_5 : 
  largest_prime_factor (base_5_to_decimal "1202102") = 307 := 
sorry

end NUMINAMATH_GPT_largest_prime_divisor_of_1202102_5_l45_4532


namespace NUMINAMATH_GPT_pigs_and_dogs_more_than_sheep_l45_4543

-- Define the number of pigs and sheep
def numberOfPigs : ℕ := 42
def numberOfSheep : ℕ := 48

-- Define the number of dogs such that it is the same as the number of pigs
def numberOfDogs : ℕ := numberOfPigs

-- Define the total number of pigs and dogs
def totalPigsAndDogs : ℕ := numberOfPigs + numberOfDogs

-- State the theorem about the difference between pigs and dogs and the number of sheep
theorem pigs_and_dogs_more_than_sheep :
  totalPigsAndDogs - numberOfSheep = 36 := 
sorry

end NUMINAMATH_GPT_pigs_and_dogs_more_than_sheep_l45_4543


namespace NUMINAMATH_GPT_complement_of_P_union_Q_in_Z_is_M_l45_4588

-- Definitions of the sets M, P, Q
def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}
def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

-- Theorem statement
theorem complement_of_P_union_Q_in_Z_is_M : (Set.univ \ (P ∪ Q)) = M :=
by 
  sorry

end NUMINAMATH_GPT_complement_of_P_union_Q_in_Z_is_M_l45_4588


namespace NUMINAMATH_GPT_most_suitable_method_l45_4520

theorem most_suitable_method {x : ℝ} (h : (x - 1) ^ 2 = 4) :
  "Direct method of taking square root" = "Direct method of taking square root" :=
by
  -- We observe that the equation is already in a form 
  -- that is conducive to applying the direct method of taking the square root,
  -- because the equation is already a perfect square on one side and a constant on the other side.
  sorry

end NUMINAMATH_GPT_most_suitable_method_l45_4520


namespace NUMINAMATH_GPT_number_of_children_is_4_l45_4557

-- Define the conditions from the problem
def youngest_child_age : ℝ := 1.5
def sum_of_ages : ℝ := 12
def common_difference : ℝ := 1

-- Define the number of children
def n : ℕ := 4

-- Prove that the number of children is 4 given the conditions
theorem number_of_children_is_4 :
  (∃ n : ℕ, (n / 2) * (2 * youngest_child_age + (n - 1) * common_difference) = sum_of_ages) ↔ n = 4 :=
by sorry

end NUMINAMATH_GPT_number_of_children_is_4_l45_4557


namespace NUMINAMATH_GPT_graveling_cost_l45_4566

def lawn_length : ℝ := 110
def lawn_breadth: ℝ := 60
def road_width : ℝ := 10
def cost_per_sq_meter : ℝ := 3

def road_1_area : ℝ := lawn_length * road_width
def intersecting_length : ℝ := lawn_breadth - road_width
def road_2_area : ℝ := intersecting_length * road_width
def total_area : ℝ := road_1_area + road_2_area
def total_cost : ℝ := total_area * cost_per_sq_meter

theorem graveling_cost :
  total_cost = 4800 := 
  by
    sorry

end NUMINAMATH_GPT_graveling_cost_l45_4566


namespace NUMINAMATH_GPT_sequence_formula_l45_4530

theorem sequence_formula (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, 0 < n → a (n + 1) = a n / (1 + a n)) : 
  ∀ n : ℕ, 0 < n → a n = 1 / n := 
by 
  sorry

end NUMINAMATH_GPT_sequence_formula_l45_4530


namespace NUMINAMATH_GPT_homework_problems_left_l45_4542

def math_problems : ℕ := 43
def science_problems : ℕ := 12
def finished_problems : ℕ := 44

theorem homework_problems_left :
  (math_problems + science_problems - finished_problems) = 11 :=
by
  sorry

end NUMINAMATH_GPT_homework_problems_left_l45_4542


namespace NUMINAMATH_GPT_smallest_integer_relative_prime_to_2310_l45_4550

theorem smallest_integer_relative_prime_to_2310 (n : ℕ) : (2 < n → n ≤ 13 → ¬ (n ∣ 2310)) → n = 13 := by
  sorry

end NUMINAMATH_GPT_smallest_integer_relative_prime_to_2310_l45_4550


namespace NUMINAMATH_GPT_greatest_value_l45_4556

theorem greatest_value (x : ℝ) : -x^2 + 9 * x - 18 ≥ 0 → x ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_l45_4556


namespace NUMINAMATH_GPT_jamshid_taimour_painting_problem_l45_4513

/-- Jamshid and Taimour Painting Problem -/
theorem jamshid_taimour_painting_problem (T : ℝ) (h1 : T > 0)
  (h2 : 1 / T + 2 / T = 1 / 5) : T = 15 :=
by
  -- solving the theorem
  sorry

end NUMINAMATH_GPT_jamshid_taimour_painting_problem_l45_4513


namespace NUMINAMATH_GPT_find_m_and_max_profit_l45_4533

theorem find_m_and_max_profit (m : ℝ) (y : ℝ) (x : ℝ) (ln : ℝ → ℝ) 
    (h1 : y = m * ln x - 1 / 100 * x ^ 2 + 101 / 50 * x + ln 10)
    (h2 : 10 < x) 
    (h3 : y = 35.7) 
    (h4 : x = 20)
    (ln_2 : ln 2 = 0.7) 
    (ln_5 : ln 5 = 1.6) :
    m = -1 ∧ ∃ x, (x = 50 ∧ (-ln x - 1 / 100 * x ^ 2 + 51 / 50 * x + ln 10 - x) = 24.4) := by
  sorry

end NUMINAMATH_GPT_find_m_and_max_profit_l45_4533


namespace NUMINAMATH_GPT_inequality1_inequality2_l45_4573

theorem inequality1 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) + a * b * c ≥ 2 * Real.sqrt 3 :=
by
  sorry

theorem inequality2 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  (1 / A) + (1 / B) + (1 / C) ≥ 9 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_inequality1_inequality2_l45_4573


namespace NUMINAMATH_GPT_max_value_trig_expression_l45_4580

theorem max_value_trig_expression : ∀ x : ℝ, (3 * Real.cos x + 4 * Real.sin x) ≤ 5 := 
sorry

end NUMINAMATH_GPT_max_value_trig_expression_l45_4580


namespace NUMINAMATH_GPT_truth_probability_of_A_l45_4570

theorem truth_probability_of_A (P_B : ℝ) (P_AB : ℝ) (h : P_AB = 0.45 ∧ P_B = 0.60 ∧ ∀ (P_A : ℝ), P_AB = P_A * P_B) : 
  ∃ (P_A : ℝ), P_A = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_truth_probability_of_A_l45_4570


namespace NUMINAMATH_GPT_average_number_of_carnations_l45_4564

-- Define the conditions in Lean
def number_of_bouquet_1 : ℕ := 9
def number_of_bouquet_2 : ℕ := 14
def number_of_bouquet_3 : ℕ := 13
def total_bouquets : ℕ := 3

-- The main statement to be proved
theorem average_number_of_carnations : 
  (number_of_bouquet_1 + number_of_bouquet_2 + number_of_bouquet_3) / total_bouquets = 12 := 
by
  sorry

end NUMINAMATH_GPT_average_number_of_carnations_l45_4564


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l45_4506

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -3) : (1 + 1/(x+1)) / ((x^2 + 4*x + 4) / (x+1)) = -1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l45_4506
