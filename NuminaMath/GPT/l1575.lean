import Mathlib

namespace NUMINAMATH_GPT_selection_count_l1575_157589

theorem selection_count (word : String) (vowels : Finset Char) (consonants : Finset Char)
  (hword : word = "УЧЕБНИК")
  (hvowels : vowels = {'У', 'Е', 'И'})
  (hconsonants : consonants = {'Ч', 'Б', 'Н', 'К'})
  :
  vowels.card * consonants.card = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_selection_count_l1575_157589


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1575_157540

open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → (x+1) * exp x > 1) ↔ ∃ x : ℝ, x > 0 ∧ (x+1) * exp x ≤ 1 :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1575_157540


namespace NUMINAMATH_GPT_smallest_natural_number_B_l1575_157584

theorem smallest_natural_number_B (A : ℕ) (h : A % 2 = 0 ∧ A % 3 = 0) :
    ∃ B : ℕ, (360 / (A^3 / B) = 5) ∧ B = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_natural_number_B_l1575_157584


namespace NUMINAMATH_GPT_geometry_problem_l1575_157564

-- Definitions for geometrical entities
variable {Point : Type} -- type representing points

variable (Line : Type) -- type representing lines
variable (Plane : Type) -- type representing planes

-- Parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop) 
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Given entities
variables (m : Line) (n : Line) (α : Plane) (β : Plane)

-- Given conditions
axiom condition1 : perpendicular α β
axiom condition2 : perpendicular_line_plane m β
axiom condition3 : ¬ contained_in m α

-- Statement of the problem in Lean 4
theorem geometry_problem : parallel m α :=
by
  -- proof will involve using the axioms and definitions
  sorry

end NUMINAMATH_GPT_geometry_problem_l1575_157564


namespace NUMINAMATH_GPT_inv_prop_x_y_l1575_157599

theorem inv_prop_x_y (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x = 4) (h3 : y = 2) (h4 : y = 10) : x = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_inv_prop_x_y_l1575_157599


namespace NUMINAMATH_GPT_range_x1_x2_l1575_157536

theorem range_x1_x2
  (x1 x2 x3 : ℝ)
  (hx3_le_x2 : x3 ≤ x2)
  (hx2_le_x1 : x2 ≤ x1)
  (hx_sum : x1 + x2 + x3 = 1)
  (hfx_sum : (x1^2) + (x2^2) + (x3^2) = 1) :
  (2 / 3 : ℝ) ≤ x1 + x2 ∧ x1 + x2 ≤ (4 / 3 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_x1_x2_l1575_157536


namespace NUMINAMATH_GPT_solve_for_pairs_l1575_157526
-- Import necessary libraries

-- Define the operation
def diamond (a b c d : ℤ) : ℤ × ℤ :=
  (a * c - b * d, a * d + b * c)

theorem solve_for_pairs : ∃! (x y : ℤ), diamond x 3 x y = (6, 0) ∧ (x, y) = (0, -2) := by
  sorry

end NUMINAMATH_GPT_solve_for_pairs_l1575_157526


namespace NUMINAMATH_GPT_nth_term_formula_l1575_157504

theorem nth_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * n^2 + n)
  (h2 : a 1 = S 1)
  (h3 : ∀ n ≥ 2, a n = S n - S (n - 1))
  : ∀ n, a n = 4 * n - 1 := by
  sorry

end NUMINAMATH_GPT_nth_term_formula_l1575_157504


namespace NUMINAMATH_GPT_max_value_l1575_157581

def a_n (n : ℕ) : ℤ := -2 * (n : ℤ)^2 + 29 * (n : ℤ) + 3

theorem max_value : ∃ n : ℕ, a_n n = 108 ∧ ∀ m : ℕ, a_n m ≤ 108 := by
  sorry

end NUMINAMATH_GPT_max_value_l1575_157581


namespace NUMINAMATH_GPT_police_officers_on_duty_l1575_157594

theorem police_officers_on_duty
  (female_officers : ℕ)
  (percent_female_on_duty : ℚ)
  (total_female_on_duty : ℕ)
  (total_officers_on_duty : ℕ)
  (H1 : female_officers = 1000)
  (H2 : percent_female_on_duty = 15 / 100)
  (H3 : total_female_on_duty = percent_female_on_duty * female_officers)
  (H4 : 2 * total_female_on_duty = total_officers_on_duty) :
  total_officers_on_duty = 300 :=
by
  sorry

end NUMINAMATH_GPT_police_officers_on_duty_l1575_157594


namespace NUMINAMATH_GPT_people_counted_on_second_day_l1575_157508

theorem people_counted_on_second_day (x : ℕ) (H1 : 2 * x + x = 1500) : x = 500 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_people_counted_on_second_day_l1575_157508


namespace NUMINAMATH_GPT_eight_sided_dice_theorem_l1575_157501
open Nat

noncomputable def eight_sided_dice_probability : ℚ :=
  let total_outcomes := 8^8
  let favorable_outcomes := 8!
  let probability_all_different := favorable_outcomes / total_outcomes
  let probability_at_least_two_same := 1 - probability_all_different
  probability_at_least_two_same

theorem eight_sided_dice_theorem :
  eight_sided_dice_probability = 16736996 / 16777216 := by
    sorry

end NUMINAMATH_GPT_eight_sided_dice_theorem_l1575_157501


namespace NUMINAMATH_GPT_find_C_and_D_l1575_157522

theorem find_C_and_D :
  (∀ x, x^2 - 3 * x - 10 ≠ 0 → (4 * x - 3) / (x^2 - 3 * x - 10) = (17 / 7) / (x - 5) + (11 / 7) / (x + 2)) :=
by
  sorry

end NUMINAMATH_GPT_find_C_and_D_l1575_157522


namespace NUMINAMATH_GPT_complement_of_M_l1575_157582

noncomputable def U : Set ℝ := Set.univ
noncomputable def M : Set ℝ := {a | a ^ 2 - 2 * a > 0}
noncomputable def C_U_M : Set ℝ := U \ M

theorem complement_of_M :
  C_U_M = {a | 0 ≤ a ∧ a ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_l1575_157582


namespace NUMINAMATH_GPT_combined_mean_score_l1575_157541

-- Definitions based on the conditions
def mean_score_class1 : ℕ := 90
def mean_score_class2 : ℕ := 80
def ratio_students (n1 n2 : ℕ) : Prop := n1 / n2 = 2 / 3

-- Proof statement
theorem combined_mean_score (n1 n2 : ℕ) 
  (h1 : ratio_students n1 n2) 
  (h2 : mean_score_class1 = 90) 
  (h3 : mean_score_class2 = 80) : 
  ((mean_score_class1 * n1) + (mean_score_class2 * n2)) / (n1 + n2) = 84 := 
by
  sorry

end NUMINAMATH_GPT_combined_mean_score_l1575_157541


namespace NUMINAMATH_GPT_shortest_third_stick_length_l1575_157557

-- Definitions of the stick lengths
def length1 := 6
def length2 := 9

-- Statement: The shortest length of the third stick that forms a triangle with lengths 6 and 9 should be 4
theorem shortest_third_stick_length : ∃ length3, length3 = 4 ∧
  (length1 + length2 > length3) ∧ (length1 + length3 > length2) ∧ (length2 + length3 > length1) :=
sorry

end NUMINAMATH_GPT_shortest_third_stick_length_l1575_157557


namespace NUMINAMATH_GPT_greatest_possible_difference_l1575_157579

theorem greatest_possible_difference (x y : ℝ) (hx1 : 6 < x) (hx2 : x < 10) (hy1 : 10 < y) (hy2 : y < 17) :
  ∃ n : ℤ, n = 9 ∧ ∀ x' y' : ℤ, (6 < x' ∧ x' < 10 ∧ 10 < y' ∧ y' < 17) → (y' - x' ≤ n) :=
by {
  -- here goes the actual proof
  sorry
}

end NUMINAMATH_GPT_greatest_possible_difference_l1575_157579


namespace NUMINAMATH_GPT_work_problem_l1575_157552

theorem work_problem (x : ℕ) (b_work : ℕ) (a_b_together_work : ℕ) (h1: b_work = 24) (h2: a_b_together_work = 8) :
  (1 / x) + (1 / b_work) = (1 / a_b_together_work) → x = 12 :=
by 
  intros h_eq
  have h_b : b_work = 24 := h1
  have h_ab : a_b_together_work = 8 := h2
  -- Full proof is omitted
  sorry

end NUMINAMATH_GPT_work_problem_l1575_157552


namespace NUMINAMATH_GPT_recurring_decimal_difference_fraction_l1575_157575

noncomputable def recurring_decimal_seventy_three := 73 / 99
noncomputable def decimal_seventy_three := 73 / 100

theorem recurring_decimal_difference_fraction :
  recurring_decimal_seventy_three - decimal_seventy_three = 73 / 9900 := sorry

end NUMINAMATH_GPT_recurring_decimal_difference_fraction_l1575_157575


namespace NUMINAMATH_GPT_solve_fraction_equation_l1575_157566

theorem solve_fraction_equation (t : ℝ) (h₀ : t ≠ 6) (h₁ : t ≠ -4) :
  (t = -2 ∨ t = -5) ↔ (t^2 - 3 * t - 18) / (t - 6) = 2 / (t + 4) := 
by
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l1575_157566


namespace NUMINAMATH_GPT_line_intersects_x_axis_at_point_l1575_157510

-- Define the conditions and required proof
theorem line_intersects_x_axis_at_point :
  (∃ x : ℝ, ∃ y : ℝ, 5 * y - 7 * x = 35 ∧ y = 0 ∧ (x, y) = (-5, 0)) :=
by
  -- The proof is omitted according to the steps
  sorry

end NUMINAMATH_GPT_line_intersects_x_axis_at_point_l1575_157510


namespace NUMINAMATH_GPT_value_of_a_l1575_157538

theorem value_of_a (a : ℝ) (A : ℝ × ℝ) (h : A = (1, 0)) : (a * A.1 + 3 * A.2 - 2 = 0) → a = 2 :=
by
  intro h1
  rw [h] at h1
  sorry

end NUMINAMATH_GPT_value_of_a_l1575_157538


namespace NUMINAMATH_GPT_flagpole_height_l1575_157511

theorem flagpole_height
  (AB : ℝ) (AD : ℝ) (BC : ℝ)
  (h1 : AB = 10)
  (h2 : BC = 3)
  (h3 : 2 * AD^2 = AB^2 + BC^2) :
  AD = Real.sqrt 54.5 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_flagpole_height_l1575_157511


namespace NUMINAMATH_GPT_p_and_not_q_l1575_157567

def p : Prop :=
  ∀ x : ℕ, x > 0 → (1 / 2) ^ x ≥ (1 / 3) ^ x

def q : Prop :=
  ∃ x : ℕ, x > 0 ∧ 2^x + 2^(1-x) = 2 * Real.sqrt 2

theorem p_and_not_q : p ∧ ¬q :=
by
  have h_p : p := sorry
  have h_not_q : ¬q := sorry
  exact ⟨h_p, h_not_q⟩

end NUMINAMATH_GPT_p_and_not_q_l1575_157567


namespace NUMINAMATH_GPT_repeat_decimal_to_fraction_l1575_157514

theorem repeat_decimal_to_fraction : 0.36666 = 11 / 30 :=
by {
    sorry
}

end NUMINAMATH_GPT_repeat_decimal_to_fraction_l1575_157514


namespace NUMINAMATH_GPT_sam_distance_l1575_157597

theorem sam_distance (miles_marguerite : ℕ) (hours_marguerite : ℕ) (hours_sam : ℕ) 
  (speed_increase : ℚ) (avg_speed_marguerite : ℚ) (speed_sam : ℚ) (distance_sam : ℚ) :
  miles_marguerite = 120 ∧ hours_marguerite = 3 ∧ hours_sam = 4 ∧ speed_increase = 1.20 ∧
  avg_speed_marguerite = miles_marguerite / hours_marguerite ∧ 
  speed_sam = avg_speed_marguerite * speed_increase ∧
  distance_sam = speed_sam * hours_sam →
  distance_sam = 192 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sam_distance_l1575_157597


namespace NUMINAMATH_GPT_find_r_over_s_at_2_l1575_157542

noncomputable def r (x : ℝ) := 6 * x
noncomputable def s (x : ℝ) := (x + 4) * (x - 1)

theorem find_r_over_s_at_2 :
  r 2 / s 2 = 2 :=
by
  -- The corresponding steps to show this theorem.
  sorry

end NUMINAMATH_GPT_find_r_over_s_at_2_l1575_157542


namespace NUMINAMATH_GPT_bacteria_growth_rate_l1575_157544

theorem bacteria_growth_rate (r : ℝ) 
  (h1 : ∀ n : ℕ, n = 22 → ∃ c : ℝ, c * r^n = c) 
  (h2 : ∀ n : ℕ, n = 21 → ∃ c : ℝ, 2 * c * r^n = c) : 
  r = 2 := 
by
  sorry

end NUMINAMATH_GPT_bacteria_growth_rate_l1575_157544


namespace NUMINAMATH_GPT_find_xyz_l1575_157576

theorem find_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 14 / 3 := 
sorry

end NUMINAMATH_GPT_find_xyz_l1575_157576


namespace NUMINAMATH_GPT_steven_owes_jeremy_l1575_157577

-- Define the payment per room
def payment_per_room : ℚ := 13 / 3

-- Define the number of rooms cleaned
def rooms_cleaned : ℚ := 5 / 2

-- Calculate the total amount owed
def total_amount_owed : ℚ := payment_per_room * rooms_cleaned

-- The theorem statement to prove
theorem steven_owes_jeremy :
  total_amount_owed = 65 / 6 :=
by
  sorry

end NUMINAMATH_GPT_steven_owes_jeremy_l1575_157577


namespace NUMINAMATH_GPT_quadratic_roots_l1575_157502

theorem quadratic_roots (a b : ℝ) (h : a^2 - 4*a*b + 5*b^2 - 2*b + 1 = 0) :
  ∃ (p q : ℝ), (∀ (x : ℝ), x^2 - p*x + q = 0 ↔ (x = a ∨ x = b)) ∧
               p = 3 ∧ q = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_roots_l1575_157502


namespace NUMINAMATH_GPT_icing_cubes_count_l1575_157516

theorem icing_cubes_count :
  let n := 5
  let total_cubes := n * n * n
  let side_faces := 4
  let cubes_per_edge_per_face := (n - 2) * (n - 1)
  let shared_edges := 4
  let icing_cubes := (side_faces * cubes_per_edge_per_face) / 2
  icing_cubes = 32 := sorry

end NUMINAMATH_GPT_icing_cubes_count_l1575_157516


namespace NUMINAMATH_GPT_sum_of_cubes_l1575_157507

variable {R : Type} [OrderedRing R] [Field R] [DecidableEq R]

theorem sum_of_cubes (a b c : R) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
    (h₄ : (a^3 + 12) / a = (b^3 + 12) / b) (h₅ : (b^3 + 12) / b = (c^3 + 12) / c) :
    a^3 + b^3 + c^3 = -36 := by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l1575_157507


namespace NUMINAMATH_GPT_spider_total_distance_l1575_157529

-- Define points where spider starts and moves
def start_position : ℤ := 3
def first_move : ℤ := -4
def second_move : ℤ := 8
def final_move : ℤ := 2

-- Define the total distance the spider crawls
def total_distance : ℤ :=
  |first_move - start_position| +
  |second_move - first_move| +
  |final_move - second_move|

-- Theorem statement
theorem spider_total_distance : total_distance = 25 :=
sorry

end NUMINAMATH_GPT_spider_total_distance_l1575_157529


namespace NUMINAMATH_GPT_interest_calculation_years_l1575_157571

theorem interest_calculation_years
  (principal : ℤ) (rate : ℝ) (difference : ℤ) (n : ℤ)
  (h_principal : principal = 2400)
  (h_rate : rate = 0.04)
  (h_difference : difference = 1920)
  (h_equation : (principal : ℝ) * rate * n = principal - difference) :
  n = 5 := 
sorry

end NUMINAMATH_GPT_interest_calculation_years_l1575_157571


namespace NUMINAMATH_GPT_interval_intersection_l1575_157561

open Set

-- Define the conditions
def condition_3x (x : ℝ) : Prop := 2 < 3 * x ∧ 3 * x < 3
def condition_4x (x : ℝ) : Prop := 2 < 4 * x ∧ 4 * x < 3

-- Define the problem statement
theorem interval_intersection :
  {x : ℝ | condition_3x x} ∩ {x | condition_4x x} = Ioo (2 / 3) (3 / 4) :=
by sorry

end NUMINAMATH_GPT_interval_intersection_l1575_157561


namespace NUMINAMATH_GPT_Bobby_candy_l1575_157535

theorem Bobby_candy (initial_candy remaining_candy1 remaining_candy2 : ℕ)
  (H1 : initial_candy = 21)
  (H2 : remaining_candy1 = initial_candy - 5)
  (H3 : remaining_candy2 = remaining_candy1 - 9):
  remaining_candy2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_Bobby_candy_l1575_157535


namespace NUMINAMATH_GPT_numberOfCubesWithNoMoreThanFourNeighbors_l1575_157573

def unitCubesWithAtMostFourNeighbors (a b c : ℕ) (h1 : a > 4) (h2 : b > 4) (h3 : c > 4) 
(h4 : (a - 2) * (b - 2) * (c - 2) = 836) : ℕ := 
  4 * (a - 2 + b - 2 + c - 2) + 8

theorem numberOfCubesWithNoMoreThanFourNeighbors (a b c : ℕ) 
(h1 : a > 4) (h2 : b > 4) (h3 : c > 4)
(h4 : (a - 2) * (b - 2) * (c - 2) = 836) :
  unitCubesWithAtMostFourNeighbors a b c h1 h2 h3 h4 = 144 :=
sorry

end NUMINAMATH_GPT_numberOfCubesWithNoMoreThanFourNeighbors_l1575_157573


namespace NUMINAMATH_GPT_system_inconsistent_l1575_157531

theorem system_inconsistent :
  ¬(∃ (x1 x2 x3 x4 : ℝ), 
    (5 * x1 + 12 * x2 + 19 * x3 + 25 * x4 = 25) ∧
    (10 * x1 + 22 * x2 + 16 * x3 + 39 * x4 = 25) ∧
    (5 * x1 + 12 * x2 + 9 * x3 + 25 * x4 = 30) ∧
    (20 * x1 + 46 * x2 + 34 * x3 + 89 * x4 = 70)) := 
by
  sorry

end NUMINAMATH_GPT_system_inconsistent_l1575_157531


namespace NUMINAMATH_GPT_lcm_100_40_is_200_l1575_157537

theorem lcm_100_40_is_200 : Nat.lcm 100 40 = 200 := by
  sorry

end NUMINAMATH_GPT_lcm_100_40_is_200_l1575_157537


namespace NUMINAMATH_GPT_find_h_l1575_157554

theorem find_h (h : ℝ) (j k : ℝ) 
  (y_eq1 : ∀ x : ℝ, (4 * (x - h)^2 + j) = 2030)
  (y_eq2 : ∀ x : ℝ, (5 * (x - h)^2 + k) = 2040)
  (int_xint1 : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 ≠ x2 → (4 * x1 * x2 = 2032) )
  (int_xint2 : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 ≠ x2 → (5 * x1 * x2 = 2040) ) :
  h = 20.5 :=
by
  sorry

end NUMINAMATH_GPT_find_h_l1575_157554


namespace NUMINAMATH_GPT_gcd_m_n_l1575_157569

namespace GCDProof

def m : ℕ := 33333333
def n : ℕ := 666666666

theorem gcd_m_n : gcd m n = 2 := 
  sorry

end GCDProof

end NUMINAMATH_GPT_gcd_m_n_l1575_157569


namespace NUMINAMATH_GPT_and_or_distrib_left_or_and_distrib_right_l1575_157546

theorem and_or_distrib_left (A B C : Prop) : A ∧ (B ∨ C) ↔ (A ∧ B) ∨ (A ∧ C) :=
sorry

theorem or_and_distrib_right (A B C : Prop) : A ∨ (B ∧ C) ↔ (A ∨ B) ∧ (A ∨ C) :=
sorry

end NUMINAMATH_GPT_and_or_distrib_left_or_and_distrib_right_l1575_157546


namespace NUMINAMATH_GPT_evaluate_f_2x_l1575_157505

def f (x : ℝ) : ℝ := x^2 - 1

theorem evaluate_f_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_2x_l1575_157505


namespace NUMINAMATH_GPT_Lucas_identity_l1575_157500

def Lucas (L : ℕ → ℤ) (F : ℕ → ℤ) : Prop :=
  ∀ n, L n = F (n + 1) + F (n - 1)

def Fib_identity1 (F : ℕ → ℤ) : Prop :=
  ∀ n, F (2 * n + 1) = F (n + 1) ^ 2 + F n ^ 2

def Fib_identity2 (F : ℕ → ℤ) : Prop :=
  ∀ n, F n ^ 2 = F (n + 1) * F (n - 1) - (-1) ^ n

theorem Lucas_identity {L F : ℕ → ℤ} (hL : Lucas L F) (hF1 : Fib_identity1 F) (hF2 : Fib_identity2 F) :
  ∀ n, L (2 * n) = L n ^ 2 - 2 * (-1) ^ n := 
sorry

end NUMINAMATH_GPT_Lucas_identity_l1575_157500


namespace NUMINAMATH_GPT_weight_7_moles_AlI3_l1575_157518

-- Definitions from the conditions
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_I : ℝ := 126.90
def molecular_weight_AlI3 : ℝ := atomic_weight_Al + 3 * atomic_weight_I
def weight_of_compound (moles : ℝ) (molecular_weight : ℝ) : ℝ := moles * molecular_weight

-- Theorem stating the weight of 7 moles of AlI3
theorem weight_7_moles_AlI3 : 
  weight_of_compound 7 molecular_weight_AlI3 = 2853.76 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_weight_7_moles_AlI3_l1575_157518


namespace NUMINAMATH_GPT_num_positive_k_for_solution_to_kx_minus_18_eq_3k_l1575_157512

theorem num_positive_k_for_solution_to_kx_minus_18_eq_3k : 
  ∃ (k_vals : Finset ℕ), 
  (∀ k ∈ k_vals, ∃ x : ℤ, k * x - 18 = 3 * k) ∧ 
  k_vals.card = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_positive_k_for_solution_to_kx_minus_18_eq_3k_l1575_157512


namespace NUMINAMATH_GPT_number_of_students_is_four_l1575_157559

-- Definitions from the conditions
def average_weight_decrease := 8
def replaced_student_weight := 96
def new_student_weight := 64
def weight_decrease := replaced_student_weight - new_student_weight

-- Goal: Prove that the number of students is 4
theorem number_of_students_is_four
  (average_weight_decrease: ℕ)
  (replaced_student_weight new_student_weight: ℕ)
  (weight_decrease: ℕ) :
  weight_decrease / average_weight_decrease = 4 := 
by
  sorry

end NUMINAMATH_GPT_number_of_students_is_four_l1575_157559


namespace NUMINAMATH_GPT_find_g_5_l1575_157560

noncomputable def g : ℝ → ℝ := sorry

axiom g_property (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0

theorem find_g_5 : g 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_g_5_l1575_157560


namespace NUMINAMATH_GPT_reflect_P_y_axis_l1575_157578

def P : ℝ × ℝ := (2, 1)

def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

theorem reflect_P_y_axis :
  reflect_y_axis P = (-2, 1) :=
by
  sorry

end NUMINAMATH_GPT_reflect_P_y_axis_l1575_157578


namespace NUMINAMATH_GPT_convert_536_oct_to_base7_l1575_157524

def octal_to_decimal (n : ℕ) : ℕ :=
  n % 10 + (n / 10 % 10) * 8 + (n / 100 % 10) * 64

def decimal_to_base7 (n : ℕ) : ℕ :=
  n % 7 + (n / 7 % 7) * 10 + (n / 49 % 7) * 100 + (n / 343 % 7) * 1000

theorem convert_536_oct_to_base7 : 
  decimal_to_base7 (octal_to_decimal 536) = 1010 :=
by
  sorry

end NUMINAMATH_GPT_convert_536_oct_to_base7_l1575_157524


namespace NUMINAMATH_GPT_intersection_right_complement_l1575_157530

open Set

def A := {x : ℝ | x - 1 ≥ 0}
def B := {x : ℝ | 3 / x ≤ 1}

theorem intersection_right_complement :
  A ∩ (compl B) = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_right_complement_l1575_157530


namespace NUMINAMATH_GPT_parallel_line_equation_perpendicular_line_equation_l1575_157534

theorem parallel_line_equation {x y : ℝ} (P : ∃ x y, 2 * x + y - 5 = 0 ∧ x - 2 * y = 0) :
  (∃ (l : ℝ), ∀ x y, 4 * x - y - 7 = 0) :=
sorry

theorem perpendicular_line_equation {x y : ℝ} (P : ∃ x y, 2 * x + y - 5 = 0 ∧ x - 2 * y = 0) :
  (∃ (l : ℝ), ∀ x y, x + 4 * y - 6 = 0) :=
sorry

end NUMINAMATH_GPT_parallel_line_equation_perpendicular_line_equation_l1575_157534


namespace NUMINAMATH_GPT_find_y_l1575_157551

theorem find_y (y : ℝ) (h : (y - 8) / (5 - (-3)) = -5 / 4) : y = -2 :=
by sorry

end NUMINAMATH_GPT_find_y_l1575_157551


namespace NUMINAMATH_GPT_ascorbic_acid_molecular_weight_l1575_157568

theorem ascorbic_acid_molecular_weight (C H O : ℕ → ℝ)
  (C_weight : C 6 = 6 * 12.01)
  (H_weight : H 8 = 8 * 1.008)
  (O_weight : O 6 = 6 * 16.00)
  (total_mass_given : 528 = 6 * 12.01 + 8 * 1.008 + 6 * 16.00) :
  6 * 12.01 + 8 * 1.008 + 6 * 16.00 = 176.124 := 
by 
  sorry

end NUMINAMATH_GPT_ascorbic_acid_molecular_weight_l1575_157568


namespace NUMINAMATH_GPT_box_volume_l1575_157565

-- Given conditions
variables (a b c : ℝ)
axiom ab_eq : a * b = 30
axiom bc_eq : b * c = 18
axiom ca_eq : c * a = 45

-- Prove that the volume of the box (a * b * c) equals 90 * sqrt(3)
theorem box_volume : a * b * c = 90 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_box_volume_l1575_157565


namespace NUMINAMATH_GPT_sylvia_buttons_l1575_157547

theorem sylvia_buttons (n : ℕ) (h₁: n % 10 = 0) (h₂: n ≥ 80):
  (∃ w : ℕ, w = (n - (n / 2) - (n / 5) - 8)) ∧ (n - (n / 2) - (n / 5) - 8 = 1) :=
by
  sorry

end NUMINAMATH_GPT_sylvia_buttons_l1575_157547


namespace NUMINAMATH_GPT_train_speed_in_km_per_hr_l1575_157562

-- Definitions from the problem conditions
def length_of_train : ℝ := 50
def time_to_cross_pole : ℝ := 3

-- Conversion factor from the problem 
def meter_per_sec_to_km_per_hr : ℝ := 3.6

-- Lean theorem statement based on problem conditions and solution
theorem train_speed_in_km_per_hr : 
  (length_of_train / time_to_cross_pole) * meter_per_sec_to_km_per_hr = 60 := by
  sorry

end NUMINAMATH_GPT_train_speed_in_km_per_hr_l1575_157562


namespace NUMINAMATH_GPT_final_selling_price_l1575_157550

-- Define the conditions as constants
def CP := 750
def loss_percentage := 20 / 100
def sales_tax_percentage := 10 / 100

-- Define the final selling price after loss and adding sales tax
theorem final_selling_price 
  (CP : ℝ) 
  (loss_percentage : ℝ)
  (sales_tax_percentage : ℝ) 
  : 750 = CP ∧ 20 / 100 = loss_percentage ∧ 10 / 100 = sales_tax_percentage → 
    (CP - (loss_percentage * CP) + (sales_tax_percentage * CP) = 675) := 
by
  intros
  sorry

end NUMINAMATH_GPT_final_selling_price_l1575_157550


namespace NUMINAMATH_GPT_probability_at_least_one_white_l1575_157591

def total_number_of_pairs : ℕ := 10
def number_of_pairs_with_at_least_one_white_ball : ℕ := 7

theorem probability_at_least_one_white :
  (number_of_pairs_with_at_least_one_white_ball : ℚ) / (total_number_of_pairs : ℚ) = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_white_l1575_157591


namespace NUMINAMATH_GPT_base_six_equals_base_b_l1575_157548

theorem base_six_equals_base_b (b : ℕ) (h1 : 3 * 6 ^ 1 + 4 * 6 ^ 0 = 22)
  (h2 : b ^ 2 + 2 * b + 1 = 22) : b = 3 :=
sorry

end NUMINAMATH_GPT_base_six_equals_base_b_l1575_157548


namespace NUMINAMATH_GPT_intersection_of_intervals_l1575_157580

theorem intersection_of_intervals (m n x : ℝ) (h1 : -1 < m) (h2 : m < 0) (h3 : 0 < n) :
  (m < x ∧ x < n) ∧ (-1 < x ∧ x < 0) ↔ -1 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_GPT_intersection_of_intervals_l1575_157580


namespace NUMINAMATH_GPT_square_field_side_length_l1575_157545

theorem square_field_side_length (time_sec : ℕ) (speed_kmh : ℕ) (perimeter : ℕ) (side_length : ℕ)
  (h1 : time_sec = 96)
  (h2 : speed_kmh = 9)
  (h3 : perimeter = (9 * 1000 / 3600 : ℕ) * 96)
  (h4 : perimeter = 4 * side_length) :
  side_length = 60 :=
by
  sorry

end NUMINAMATH_GPT_square_field_side_length_l1575_157545


namespace NUMINAMATH_GPT_kevin_marbles_l1575_157558

theorem kevin_marbles (M : ℕ) (h1 : 40 * 3 = 120) (h2 : 4 * M = 320 - 120) :
  M = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_kevin_marbles_l1575_157558


namespace NUMINAMATH_GPT_rectangle_width_l1575_157556

theorem rectangle_width
  (l w : ℕ)
  (h1 : l * w = 1638)
  (h2 : 10 * l = 390) :
  w = 42 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l1575_157556


namespace NUMINAMATH_GPT_solve_for_A_in_terms_of_B_l1575_157515

noncomputable def f (A B x : ℝ) := A * x - 2 * B^2
noncomputable def g (B x : ℝ) := B * x

theorem solve_for_A_in_terms_of_B (A B : ℝ) (hB : B ≠ 0) (h : f A B (g B 1) = 0) : A = 2 * B := by
  sorry

end NUMINAMATH_GPT_solve_for_A_in_terms_of_B_l1575_157515


namespace NUMINAMATH_GPT_meadowbrook_total_not_74_l1575_157506

theorem meadowbrook_total_not_74 (h c : ℕ) : 
  21 * h + 6 * c ≠ 74 := sorry

end NUMINAMATH_GPT_meadowbrook_total_not_74_l1575_157506


namespace NUMINAMATH_GPT_selection_methods_including_both_boys_and_girls_l1575_157520

def boys : ℕ := 4
def girls : ℕ := 3
def total_people : ℕ := boys + girls
def select : ℕ := 4

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_methods_including_both_boys_and_girls :
  combination 7 4 - combination boys 4 = 34 :=
by
  sorry

end NUMINAMATH_GPT_selection_methods_including_both_boys_and_girls_l1575_157520


namespace NUMINAMATH_GPT_linda_spent_total_l1575_157543

noncomputable def total_spent (notebooks_price_euro : ℝ) (notebooks_count : ℕ) 
    (pencils_price_pound : ℝ) (pencils_gift_card_pound : ℝ)
    (pens_price_yen : ℝ) (pens_points : ℝ) 
    (markers_price_dollar : ℝ) (calculator_price_dollar : ℝ)
    (marker_discount : ℝ) (coupon_discount : ℝ) (sales_tax : ℝ)
    (euro_to_dollar : ℝ) (pound_to_dollar : ℝ) (yen_to_dollar : ℝ) : ℝ :=
  let notebooks_cost := (notebooks_price_euro * notebooks_count) * euro_to_dollar
  let pencils_cost := 0
  let pens_cost := 0
  let marked_price := markers_price_dollar * (1 - marker_discount)
  let us_total_before_tax := (marked_price + calculator_price_dollar) * (1 - coupon_discount)
  let us_total_after_tax := us_total_before_tax * (1 + sales_tax)
  notebooks_cost + pencils_cost + pens_cost + us_total_after_tax

theorem linda_spent_total : 
  total_spent 1.2 3 1.5 5 170 200 2.8 12.5 0.15 0.10 0.05 1.1 1.25 0.009 = 18.0216 := 
  by
  sorry

end NUMINAMATH_GPT_linda_spent_total_l1575_157543


namespace NUMINAMATH_GPT_sin_mul_cos_eq_quarter_l1575_157521

open Real

theorem sin_mul_cos_eq_quarter (α : ℝ) (h : sin α - cos α = sqrt 2 / 2) : sin α * cos α = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_mul_cos_eq_quarter_l1575_157521


namespace NUMINAMATH_GPT_product_modulo_l1575_157539

theorem product_modulo (n : ℕ) (h : 93 * 68 * 105 ≡ n [MOD 20]) (h_range : 0 ≤ n ∧ n < 20) : n = 0 := 
by
  sorry

end NUMINAMATH_GPT_product_modulo_l1575_157539


namespace NUMINAMATH_GPT_value_of_M_l1575_157583

noncomputable def a : ℝ := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2)
noncomputable def b : ℝ := Real.sqrt (5 - 2 * Real.sqrt 6)
noncomputable def M : ℝ := a - b

theorem value_of_M : M = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_M_l1575_157583


namespace NUMINAMATH_GPT_find_a_range_l1575_157570

open Real

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (4, 1)
def B : (ℝ × ℝ) := (-1, -6)
def C : (ℝ × ℝ) := (-3, 2)

-- Define the system of inequalities representing the region D
def region_D (x y : ℝ) : Prop :=
  7 * x - 5 * y - 23 ≤ 0 ∧
  x + 7 * y - 11 ≤ 0 ∧
  4 * x + y + 10 ≥ 0

-- Define the inequality condition for points B and C on opposite sides of the line 4x - 3y - a = 0
def opposite_sides (a : ℝ) : Prop :=
  (14 - a) * (-18 - a) < 0

-- Lean statement to prove the given problem
theorem find_a_range : 
  ∃ a : ℝ, region_D 0 0 ∧ opposite_sides a → -18 < a ∧ a < 14 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_range_l1575_157570


namespace NUMINAMATH_GPT_number_conversion_l1575_157595

theorem number_conversion (a b c d : ℕ) : 
  4090000 = 409 * 10000 ∧ (a = 800000) ∧ (b = 5000) ∧ (c = 20) ∧ (d = 4) → 
  (a + b + c + d = 805024) :=
by
  sorry

end NUMINAMATH_GPT_number_conversion_l1575_157595


namespace NUMINAMATH_GPT_polynomial_has_real_root_l1575_157533

open Real

theorem polynomial_has_real_root (a : ℝ) : 
  ∃ x : ℝ, x^5 + a * x^4 - x^3 + a * x^2 - x + a = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_has_real_root_l1575_157533


namespace NUMINAMATH_GPT_positive_difference_is_30_l1575_157517

-- Define the absolute value equation condition
def abs_condition (x : ℝ) : Prop := abs (x - 3) = 15

-- Define the solutions to the absolute value equation
def solution1 : ℝ := 18
def solution2 : ℝ := -12

-- Define the positive difference of the solutions
def positive_difference : ℝ := abs (solution1 - solution2)

-- Theorem statement: the positive difference is 30
theorem positive_difference_is_30 : positive_difference = 30 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_is_30_l1575_157517


namespace NUMINAMATH_GPT_largest_positive_integer_n_l1575_157527

theorem largest_positive_integer_n :
  ∃ n : ℕ, (∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≥ 2 / n) ∧ (∀ m : ℕ, m > n → ∃ x : ℝ, (Real.sin x) ^ m + (Real.cos x) ^ m < 2 / m) :=
sorry

end NUMINAMATH_GPT_largest_positive_integer_n_l1575_157527


namespace NUMINAMATH_GPT_length_major_axis_eq_six_l1575_157598

-- Define the given equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 9) = 1

-- The theorem stating the length of the major axis
theorem length_major_axis_eq_six (x y : ℝ) (h : ellipse_equation x y) : 
  2 * (Real.sqrt 9) = 6 :=
by
  sorry

end NUMINAMATH_GPT_length_major_axis_eq_six_l1575_157598


namespace NUMINAMATH_GPT_vasya_can_guess_number_in_10_questions_l1575_157588

noncomputable def log2 (n : ℕ) : ℝ := 
  Real.log n / Real.log 2

theorem vasya_can_guess_number_in_10_questions (n q : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 1000) (h3 : q = 10) :
  q ≥ log2 n := 
by
  sorry

end NUMINAMATH_GPT_vasya_can_guess_number_in_10_questions_l1575_157588


namespace NUMINAMATH_GPT_driving_speed_ratio_l1575_157528

theorem driving_speed_ratio
  (x : ℝ) (y : ℝ)
  (h1 : y = 2 * x) :
  y / x = 2 := by
  sorry

end NUMINAMATH_GPT_driving_speed_ratio_l1575_157528


namespace NUMINAMATH_GPT_union_complements_eq_l1575_157586

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_complements_eq :
  U = {0, 1, 3, 5, 6, 8} →
  A = {1, 5, 8} →
  B = {2} →
  (U \ A) ∪ B = {0, 2, 3, 6} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  -- Prove that (U \ A) ∪ B = {0, 2, 3, 6}
  sorry

end NUMINAMATH_GPT_union_complements_eq_l1575_157586


namespace NUMINAMATH_GPT_linemen_count_l1575_157574

-- Define the initial conditions
def linemen_drink := 8
def skill_position_players_drink := 6
def total_skill_position_players := 10
def cooler_capacity := 126
def skill_position_players_drink_first := 5

-- Define the number of ounces drunk by skill position players during the first break
def skill_position_players_first_break := skill_position_players_drink_first * skill_position_players_drink

-- Define the theorem stating that the number of linemen (L) is 12 given the conditions
theorem linemen_count :
  ∃ L : ℕ, linemen_drink * L + skill_position_players_first_break = cooler_capacity ∧ L = 12 :=
by {
  sorry -- Proof to be provided.
}

end NUMINAMATH_GPT_linemen_count_l1575_157574


namespace NUMINAMATH_GPT_focus_of_parabola_l1575_157532

theorem focus_of_parabola :
  (∃ f : ℝ, ∀ y : ℝ, (x = -1 / 4 * y^2) = (x = (y^2 / 4 + f)) -> f = -1) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l1575_157532


namespace NUMINAMATH_GPT_distance_from_origin_is_correct_l1575_157519

noncomputable def is_distance_8_from_x_axis (x y : ℝ) := y = 8
noncomputable def is_distance_12_from_point (x y : ℝ) := (x - 1)^2 + (y - 6)^2 = 144
noncomputable def x_greater_than_1 (x : ℝ) := x > 1
noncomputable def distance_from_origin (x y : ℝ) := Real.sqrt (x^2 + y^2)

theorem distance_from_origin_is_correct (x y : ℝ)
  (h1 : is_distance_8_from_x_axis x y)
  (h2 : is_distance_12_from_point x y)
  (h3 : x_greater_than_1 x) :
  distance_from_origin x y = Real.sqrt (205 + 2 * Real.sqrt 140) :=
by
  sorry

end NUMINAMATH_GPT_distance_from_origin_is_correct_l1575_157519


namespace NUMINAMATH_GPT_floor_of_sqrt_sum_eq_floor_of_sqrt_expr_l1575_157592

theorem floor_of_sqrt_sum_eq_floor_of_sqrt_expr (n : ℤ): 
  Int.floor (Real.sqrt n + Real.sqrt (n + 1)) = Int.floor (Real.sqrt (4 * n + 2)) := 
sorry

end NUMINAMATH_GPT_floor_of_sqrt_sum_eq_floor_of_sqrt_expr_l1575_157592


namespace NUMINAMATH_GPT_true_proposition_l1575_157513

-- Definitions based on the conditions
def p (x : ℝ) := x * (x - 1) ≠ 0 → x ≠ 0 ∧ x ≠ 1
def q (a b c : ℝ) := a > b → c > 0 → a * c > b * c

-- The theorem based on the question and the conditions
theorem true_proposition (x a b c : ℝ) (hp : p x) (hq_false : ¬ q a b c) : p x ∨ q a b c :=
by
  sorry

end NUMINAMATH_GPT_true_proposition_l1575_157513


namespace NUMINAMATH_GPT_consecutive_integers_divisible_product_l1575_157553

theorem consecutive_integers_divisible_product (m n : ℕ) (h : m < n) :
  ∀ k : ℕ, ∃ i j : ℕ, i ≠ j ∧ k + i < k + n ∧ k + j < k + n ∧ (k + i) * (k + j) % (m * n) = 0 :=
by sorry

end NUMINAMATH_GPT_consecutive_integers_divisible_product_l1575_157553


namespace NUMINAMATH_GPT_completing_square_l1575_157525

theorem completing_square (x : ℝ) (h : x^2 - 6 * x - 7 = 0) : (x - 3)^2 = 16 := 
sorry

end NUMINAMATH_GPT_completing_square_l1575_157525


namespace NUMINAMATH_GPT_fraction_of_married_men_l1575_157590

theorem fraction_of_married_men (prob_single_woman : ℚ) (H : prob_single_woman = 3 / 7) :
  ∃ (fraction_married_men : ℚ), fraction_married_men = 4 / 11 :=
by
  -- Further proof steps would go here if required
  sorry

end NUMINAMATH_GPT_fraction_of_married_men_l1575_157590


namespace NUMINAMATH_GPT_adam_and_simon_50_miles_apart_l1575_157509

noncomputable def time_when_50_miles_apart (x : ℝ) : Prop :=
  let adam_distance := 10 * x
  let simon_distance := 8 * x
  (adam_distance^2 + simon_distance^2 = 50^2) 

theorem adam_and_simon_50_miles_apart : 
  ∃ x : ℝ, time_when_50_miles_apart x ∧ x = 50 / 12.8 := 
sorry

end NUMINAMATH_GPT_adam_and_simon_50_miles_apart_l1575_157509


namespace NUMINAMATH_GPT_postage_cost_correct_l1575_157523

-- Conditions
def base_rate : ℕ := 35
def additional_rate_per_ounce : ℕ := 25
def weight_in_ounces : ℚ := 5.25
def first_ounce : ℚ := 1
def fraction_weight : ℚ := weight_in_ounces - first_ounce
def num_additional_charges : ℕ := Nat.ceil (fraction_weight)

-- Question and correct answer
def total_postage_cost : ℕ := base_rate + (num_additional_charges * additional_rate_per_ounce)
def answer_in_cents : ℕ := 160

theorem postage_cost_correct : total_postage_cost = answer_in_cents := by sorry

end NUMINAMATH_GPT_postage_cost_correct_l1575_157523


namespace NUMINAMATH_GPT_average_salary_for_company_l1575_157587

theorem average_salary_for_company
    (number_of_managers : Nat)
    (number_of_associates : Nat)
    (average_salary_managers : Nat)
    (average_salary_associates : Nat)
    (hnum_managers : number_of_managers = 15)
    (hnum_associates : number_of_associates = 75)
    (has_managers : average_salary_managers = 90000)
    (has_associates : average_salary_associates = 30000) : 
    (number_of_managers * average_salary_managers + number_of_associates * average_salary_associates) / 
    (number_of_managers + number_of_associates) = 40000 := 
    by
    sorry

end NUMINAMATH_GPT_average_salary_for_company_l1575_157587


namespace NUMINAMATH_GPT_total_chairs_l1575_157572

def numIndoorTables := 9
def numOutdoorTables := 11
def chairsPerIndoorTable := 10
def chairsPerOutdoorTable := 3

theorem total_chairs :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 :=
by
  sorry

end NUMINAMATH_GPT_total_chairs_l1575_157572


namespace NUMINAMATH_GPT_roots_k_m_l1575_157596

theorem roots_k_m (k m : ℝ) 
  (h1 : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 11 ∧ a * b + b * c + c * a = k ∧ a * b * c = m)
  : k + m = 52 :=
sorry

end NUMINAMATH_GPT_roots_k_m_l1575_157596


namespace NUMINAMATH_GPT_no_real_solution_l1575_157563

theorem no_real_solution (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : 1 / a + 1 / b = 1 / (a + b)) : False :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_l1575_157563


namespace NUMINAMATH_GPT_find_a_value_l1575_157585

noncomputable def solve_for_a (y : ℝ) (a : ℝ) : Prop :=
  0 < y ∧ (a * y) / 20 + (3 * y) / 10 = 0.6499999999999999 * y 

theorem find_a_value (y : ℝ) (a : ℝ) (h : solve_for_a y a) : a = 7 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_value_l1575_157585


namespace NUMINAMATH_GPT_area_of_billboard_l1575_157593

variable (L W : ℕ) (P : ℕ)
variable (hW : W = 8) (hP : P = 46)

theorem area_of_billboard (h1 : P = 2 * L + 2 * W) : L * W = 120 :=
by
  sorry

end NUMINAMATH_GPT_area_of_billboard_l1575_157593


namespace NUMINAMATH_GPT_parametric_eq_to_ordinary_l1575_157555

theorem parametric_eq_to_ordinary (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
    let x := abs (Real.sin (θ / 2) + Real.cos (θ / 2))
    let y := 1 + Real.sin θ
    x ^ 2 = y := by sorry

end NUMINAMATH_GPT_parametric_eq_to_ordinary_l1575_157555


namespace NUMINAMATH_GPT_apple_count_l1575_157549

-- Definitions of initial conditions and calculations.
def B_0 : Int := 5  -- initial number of blue apples
def R_0 : Int := 3  -- initial number of red apples
def Y : Int := 2 * B_0  -- number of yellow apples given by neighbor
def R : Int := R_0 - 2  -- number of red apples after giving away to a friend
def B : Int := B_0 - 3  -- number of blue apples after 3 rot
def G : Int := (B + Y) / 3  -- number of green apples received
def Y' : Int := Y - 2  -- number of yellow apples after eating 2
def R' : Int := R - 1  -- number of red apples after eating 1

-- Lean theorem statement
theorem apple_count (B_0 R_0 Y R B G Y' R' : ℤ)
  (h1 : B_0 = 5)
  (h2 : R_0 = 3)
  (h3 : Y = 2 * B_0)
  (h4 : R = R_0 - 2)
  (h5 : B = B_0 - 3)
  (h6 : G = (B + Y) / 3)
  (h7 : Y' = Y - 2)
  (h8 : R' = R - 1)
  : B + Y' + G + R' = 14 := 
by
  sorry

end NUMINAMATH_GPT_apple_count_l1575_157549


namespace NUMINAMATH_GPT_ark5_ensures_metabolic_energy_l1575_157503

-- Define conditions
def inhibits_ark5_activity (inhibits: Bool) (balance: Bool): Prop :=
  if inhibits then ¬balance else balance

def cancer_cells_proliferate_without_energy (proliferate: Bool) (die_due_to_insufficient_energy: Bool) : Prop :=
  proliferate → die_due_to_insufficient_energy

-- Define the hypothesis based on conditions
def hypothesis (inhibits: Bool) (balance: Bool) (proliferate: Bool) (die_due_to_insufficient_energy: Bool): Prop :=
  inhibits_ark5_activity inhibits balance ∧ cancer_cells_proliferate_without_energy proliferate die_due_to_insufficient_energy

-- Define the theorem to be proved
theorem ark5_ensures_metabolic_energy
  (inhibits : Bool)
  (balance : Bool)
  (proliferate : Bool)
  (die_due_to_insufficient_energy : Bool)
  (h : hypothesis inhibits balance proliferate die_due_to_insufficient_energy) :
  ensures_metabolic_energy :=
  sorry

end NUMINAMATH_GPT_ark5_ensures_metabolic_energy_l1575_157503
