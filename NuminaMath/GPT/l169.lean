import Mathlib

namespace bike_license_combinations_l169_169017

theorem bike_license_combinations : 
  let letters := 3
  let digits := 10
  let total_combinations := letters * digits^4
  total_combinations = 30000 := by
  let letters := 3
  let digits := 10
  let total_combinations := letters * digits^4
  sorry

end bike_license_combinations_l169_169017


namespace smallest_even_in_sequence_sum_400_l169_169011

theorem smallest_even_in_sequence_sum_400 :
  ∃ (n : ℤ), (n - 6) + (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 400 ∧ (n - 6) % 2 = 0 ∧ n - 6 = 52 :=
sorry

end smallest_even_in_sequence_sum_400_l169_169011


namespace max_min_S_l169_169429

theorem max_min_S (x y : ℝ) (h : (x - 1)^2 + (y + 2)^2 = 4) : 
  (∃ S_max S_min : ℝ, S_max = 4 + 2 * Real.sqrt 5 ∧ S_min = 4 - 2 * Real.sqrt 5 ∧ 
  (∀ S : ℝ, (∃ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = 4 ∧ S = 2 * x + y) → S ≤ S_max ∧ S ≥ S_min)) :=
sorry

end max_min_S_l169_169429


namespace unique_triple_solution_l169_169954

theorem unique_triple_solution :
  ∃! (x y z : ℕ), (y > 1) ∧ Prime y ∧
                  (¬(3 ∣ z ∧ y ∣ z)) ∧
                  (x^3 - y^3 = z^2) ∧
                  (x = 8 ∧ y = 7 ∧ z = 13) :=
by
  sorry

end unique_triple_solution_l169_169954


namespace square_area_with_tangent_circles_l169_169801

theorem square_area_with_tangent_circles :
  let r := 3 -- radius of each circle in inches
  let d := 2 * r -- diameter of each circle in inches
  let side_length := 2 * d -- side length of the square in inches
  let area := side_length * side_length -- area of the square in square inches
  side_length = 12 ∧ area = 144 :=
by
  let r := 3
  let d := 2 * r
  let side_length := 2 * d
  let area := side_length * side_length
  sorry

end square_area_with_tangent_circles_l169_169801


namespace number_of_chocolates_l169_169414

-- Define the dimensions of the box
def W_box := 30
def L_box := 20
def H_box := 5

-- Define the dimensions of one chocolate
def W_chocolate := 6
def L_chocolate := 4
def H_chocolate := 1

-- Calculate the volume of the box
def V_box := W_box * L_box * H_box

-- Calculate the volume of one chocolate
def V_chocolate := W_chocolate * L_chocolate * H_chocolate

-- Lean theorem statement for the proof problem
theorem number_of_chocolates : V_box / V_chocolate = 125 := 
by
  sorry

end number_of_chocolates_l169_169414


namespace janeth_balloons_l169_169424

/-- Janeth's total remaining balloons after accounting for burst ones. -/
def total_remaining_balloons (round_bags : Nat) (round_per_bag : Nat) (burst_round : Nat)
    (long_bags : Nat) (long_per_bag : Nat) (burst_long : Nat)
    (heart_bags : Nat) (heart_per_bag : Nat) (burst_heart : Nat) : Nat :=
  let total_round := round_bags * round_per_bag - burst_round
  let total_long := long_bags * long_per_bag - burst_long
  let total_heart := heart_bags * heart_per_bag - burst_heart
  total_round + total_long + total_heart

theorem janeth_balloons :
  total_remaining_balloons 5 25 5 4 35 7 3 40 3 = 370 :=
by
  let round_bags := 5
  let round_per_bag := 25
  let burst_round := 5
  let long_bags := 4
  let long_per_bag := 35
  let burst_long := 7
  let heart_bags := 3
  let heart_per_bag := 40
  let burst_heart := 3
  show total_remaining_balloons round_bags round_per_bag burst_round long_bags long_per_bag burst_long heart_bags heart_per_bag burst_heart = 370
  sorry

end janeth_balloons_l169_169424


namespace moms_took_chocolates_l169_169015

theorem moms_took_chocolates (N : ℕ) (A : ℕ) (M : ℕ) : 
  N = 10 → 
  A = 3 * N →
  A - M = N + 15 →
  M = 5 :=
by
  intros h1 h2 h3
  sorry

end moms_took_chocolates_l169_169015


namespace difference_is_693_l169_169980

noncomputable def one_tenth_of_seven_thousand : ℕ := 1 / 10 * 7000
noncomputable def one_tenth_percent_of_seven_thousand : ℕ := (1 / 10 / 100) * 7000
noncomputable def difference : ℕ := one_tenth_of_seven_thousand - one_tenth_percent_of_seven_thousand

theorem difference_is_693 :
  difference = 693 :=
by
  sorry

end difference_is_693_l169_169980


namespace cube_volume_l169_169794

theorem cube_volume (d_AF : Real) (h : d_AF = 6 * Real.sqrt 2) : ∃ (V : Real), V = 216 :=
by {
  sorry
}

end cube_volume_l169_169794


namespace savings_calculation_l169_169315

theorem savings_calculation (income expenditure : ℝ) (h_ratio : income = 5 / 4 * expenditure) (h_income : income = 19000) :
  income - expenditure = 3800 := 
by
  -- The solution will be filled in here,
  -- showing the calculus automatically.
  sorry

end savings_calculation_l169_169315


namespace find_m_l169_169571

def l1 (m x y: ℝ) : Prop := 2 * x + m * y - 2 = 0
def l2 (m x y: ℝ) : Prop := m * x + 2 * y - 1 = 0
def perpendicular (m : ℝ) : Prop :=
  let slope_l1 := -2 / m
  let slope_l2 := -m / 2
  slope_l1 * slope_l2 = -1

theorem find_m (m : ℝ) (h : perpendicular m) : m = 2 :=
sorry

end find_m_l169_169571


namespace t_shaped_region_slope_divides_area_in_half_l169_169147

theorem t_shaped_region_slope_divides_area_in_half :
  ∃ (m : ℚ), (m = 4 / 11) ∧ (
    let area1 := 2 * (m * 2 * 4)
    let area2 := ((4 - m * 2) * 4) + 6
    area1 = area2
  ) :=
by
  sorry

end t_shaped_region_slope_divides_area_in_half_l169_169147


namespace number_of_special_three_digit_numbers_l169_169774

noncomputable def count_special_three_digit_numbers : ℕ :=
  Nat.choose 9 3

theorem number_of_special_three_digit_numbers : count_special_three_digit_numbers = 84 := by
  sorry

end number_of_special_three_digit_numbers_l169_169774


namespace fraction_oj_is_5_over_13_l169_169911

def capacity_first_pitcher : ℕ := 800
def capacity_second_pitcher : ℕ := 500
def fraction_oj_first_pitcher : ℚ := 1 / 4
def fraction_oj_second_pitcher : ℚ := 3 / 5

def amount_oj_first_pitcher : ℚ := capacity_first_pitcher * fraction_oj_first_pitcher
def amount_oj_second_pitcher : ℚ := capacity_second_pitcher * fraction_oj_second_pitcher

def total_amount_oj : ℚ := amount_oj_first_pitcher + amount_oj_second_pitcher
def total_capacity : ℚ := capacity_first_pitcher + capacity_second_pitcher

def fraction_oj_large_container : ℚ := total_amount_oj / total_capacity

theorem fraction_oj_is_5_over_13 : fraction_oj_large_container = (5 / 13) := by
  -- Proof would go here
  sorry

end fraction_oj_is_5_over_13_l169_169911


namespace exceeding_fraction_l169_169810

def repeatingDecimal : ℚ := 8 / 33
def decimalFraction : ℚ := 6 / 25
def difference : ℚ := repeatingDecimal - decimalFraction

theorem exceeding_fraction :
  difference = 2 / 825 := by
  sorry

end exceeding_fraction_l169_169810


namespace FI_squared_l169_169825

-- Definitions for the given conditions
-- Note: Further geometric setup and formalization might be necessary to carry 
-- out the complete proof in Lean, but the setup will follow these basic definitions.

-- Let ABCD be a square
def ABCD_square (A B C D : ℝ × ℝ) : Prop :=
  -- conditions for ABCD being a square (to be properly defined based on coordinates and properties)
  sorry

-- Triangle AEH is an equilateral triangle with side length sqrt(3)
def equilateral_AEH (A E H : ℝ × ℝ) (s : ℝ) : Prop :=
  dist A E = s ∧ dist E H = s ∧ dist H A = s 

-- Points E and H lie on AB and DA respectively
-- Points F and G lie on BC and CD respectively
-- Points I and J lie on EH with FI ⊥ EH and GJ ⊥ EH
-- Areas of triangles and quadrilaterals
def geometric_conditions (A B C D E F G H I J : ℝ × ℝ) : Prop :=
  sorry

-- Final statement to prove
theorem FI_squared (A B C D E F G H I J : ℝ × ℝ) (s : ℝ) 
  (h_square: ABCD_square A B C D) 
  (h_equilateral: equilateral_AEH A E H (Real.sqrt 3))
  (h_geo: geometric_conditions A B C D E F G H I J) :
  dist F I ^ 2 = 4 / 3 :=
sorry

end FI_squared_l169_169825


namespace cos_alpha_beta_l169_169443

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x * (sin x) ^ 2 - (1 / 2)

theorem cos_alpha_beta :
  ∀ (α β : ℝ), 
    (0 < α ∧ α < π / 2) →
    (0 < β ∧ β < π / 2) →
    f (α / 2) = sqrt 5 / 5 →
    f (β / 2) = 3 * sqrt 10 / 10 →
    cos (α - β) = sqrt 2 / 2 :=
by
  intros α β hα hβ h1 h2
  sorry

end cos_alpha_beta_l169_169443


namespace find_x_l169_169918

theorem find_x (x : ℝ) (h : x - 1/10 = x / 10) : x = 1 / 9 := 
  sorry

end find_x_l169_169918


namespace strictly_increasing_difference_l169_169627

variable {a b : ℝ}
variable {f g : ℝ → ℝ}

theorem strictly_increasing_difference
  (h_diff : ∀ x ∈ Set.Icc a b, DifferentiableAt ℝ f x ∧ DifferentiableAt ℝ g x)
  (h_eq : f a = g a)
  (h_diff_ineq : ∀ x ∈ Set.Ioo a b, (deriv f x : ℝ) > (deriv g x : ℝ)) :
  ∀ x ∈ Set.Ioo a b, f x > g x := by
  sorry

end strictly_increasing_difference_l169_169627


namespace pencil_length_difference_l169_169903

theorem pencil_length_difference (a b : ℝ) (h1 : a = 1) (h2 : b = 4/9) :
  a - b - b = 1/9 :=
by
  rw [h1, h2]
  sorry

end pencil_length_difference_l169_169903


namespace smallest_number_l169_169324

theorem smallest_number (a b c d : ℤ) (h_a : a = 0) (h_b : b = -1) (h_c : c = -4) (h_d : d = 5) : 
  c < b ∧ c < a ∧ c < d :=
by {
  sorry
}

end smallest_number_l169_169324


namespace find_a_l169_169874

theorem find_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : x - a * y = 3) : a = -1 :=
by
  -- Solution steps go here
  sorry

end find_a_l169_169874


namespace correct_statements_l169_169004

-- Define the propositions p and q
variables (p q : Prop)

-- Define the given statements as logical conditions
def statement1 := (p ∧ q) → (p ∨ q)
def statement2 := ¬(p ∧ q) → (p ∨ q)
def statement3 := (p ∨ q) ↔ ¬¬p
def statement4 := (¬p) → ¬(p ∧ q)

-- Define the proof problem
theorem correct_statements :
  ((statement1 p q) ∧ (¬statement2 p q) ∧ (statement3 p q) ∧ (¬statement4 p q)) :=
by {
  -- Here you would prove that
  -- statement1 is correct,
  -- statement2 is incorrect,
  -- statement3 is correct,
  -- statement4 is incorrect
  sorry
}

end correct_statements_l169_169004


namespace total_students_in_lunchroom_l169_169108

theorem total_students_in_lunchroom :
  (34 * 6) + 15 = 219 :=
by
  sorry

end total_students_in_lunchroom_l169_169108


namespace rectangular_plot_area_l169_169237

-- Define the conditions
def breadth := 11  -- breadth in meters
def length := 3 * breadth  -- length is thrice the breadth

-- Define the function to calculate area
def area (length breadth : ℕ) := length * breadth

-- The theorem to prove
theorem rectangular_plot_area : area length breadth = 363 := by
  sorry

end rectangular_plot_area_l169_169237


namespace range_of_a_l169_169256

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 1| > Real.logb 2 a) →
  0 < a ∧ a < 8 :=
by
  sorry

end range_of_a_l169_169256


namespace Tammy_average_speed_second_day_l169_169397

theorem Tammy_average_speed_second_day : 
  ∀ (t v : ℝ), 
    (t + (t - 2) + (t + 1) = 20) → 
    (7 * v + 5 * (v + 0.5) + 8 * (v + 1.5) = 85) → 
    (v + 0.5 = 4.025) := 
by 
  intros t v ht hv 
  sorry

end Tammy_average_speed_second_day_l169_169397


namespace behemoth_and_rita_finish_ice_cream_l169_169037

theorem behemoth_and_rita_finish_ice_cream (x y : ℝ) (h : 3 * x + 2 * y = 1) : 3 * (x + y) ≥ 1 :=
by
  sorry

end behemoth_and_rita_finish_ice_cream_l169_169037


namespace part1_part2_l169_169351

def custom_op (a b : ℤ) : ℤ := a^2 - b + a * b

theorem part1  : custom_op (-3) (-2) = 17 := by
  sorry

theorem part2 : custom_op (-2) (custom_op (-3) (-2)) = -47 := by
  sorry

end part1_part2_l169_169351


namespace find_sum_l169_169437

variable {a : ℕ → ℝ} {r : ℝ}

-- Conditions: a_n > 0 for all n
axiom pos : ∀ n : ℕ, a n > 0

-- Given equation: a_1 * a_5 + 2 * a_3 * a_5 + a_3 * a_7 = 25
axiom given_eq : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25

theorem find_sum : a 3 + a 5 = 5 :=
by
  sorry

end find_sum_l169_169437


namespace transformed_quadratic_l169_169546

theorem transformed_quadratic (a b c n x : ℝ) (h : a * x^2 + b * x + c = 0) :
  a * x^2 + n * b * x + n^2 * c = 0 :=
sorry

end transformed_quadratic_l169_169546


namespace wooden_block_length_is_correct_l169_169518

noncomputable def length_of_block : ℝ :=
  let initial_length := 31
  let reduction := 30 / 100
  initial_length - reduction

theorem wooden_block_length_is_correct :
  length_of_block = 30.7 :=
by
  sorry

end wooden_block_length_is_correct_l169_169518


namespace proof_l169_169016

-- Definition of the logical statements
def all_essays_correct (maria : Type) : Prop := sorry
def passed_course (maria : Type) : Prop := sorry

-- Condition provided in the problem
axiom condition : ∀ (maria : Type), all_essays_correct maria → passed_course maria

-- We need to prove this
theorem proof (maria : Type) : ¬ (passed_course maria) → ¬ (all_essays_correct maria) :=
by sorry

end proof_l169_169016


namespace merchants_and_cost_l169_169223

theorem merchants_and_cost (n C : ℕ) (h1 : 8 * n = C + 3) (h2 : 7 * n = C - 4) : n = 7 ∧ C = 53 := 
by 
  sorry

end merchants_and_cost_l169_169223


namespace average_of_remaining_two_numbers_l169_169332

theorem average_of_remaining_two_numbers 
  (avg_6 : ℝ) (avg1_2 : ℝ) (avg2_2 : ℝ)
  (n1 n2 n3 : ℕ)
  (h_avg6 : n1 = 6 ∧ avg_6 = 4.60)
  (h_avg1_2 : n2 = 2 ∧ avg1_2 = 3.4)
  (h_avg2_2 : n3 = 2 ∧ avg2_2 = 3.8) :
  ∃ avg_rem2 : ℝ, avg_rem2 = 6.6 :=
by {
  sorry
}

end average_of_remaining_two_numbers_l169_169332


namespace totalMarbles_l169_169811

def originalMarbles : ℕ := 22
def marblesGiven : ℕ := 20

theorem totalMarbles : originalMarbles + marblesGiven = 42 := by
  sorry

end totalMarbles_l169_169811


namespace biscuits_per_dog_l169_169870

-- Define constants for conditions
def total_biscuits : ℕ := 6
def number_of_dogs : ℕ := 2

-- Define the statement to prove
theorem biscuits_per_dog : total_biscuits / number_of_dogs = 3 := by
  -- Calculation here
  sorry

end biscuits_per_dog_l169_169870


namespace unique_two_digit_number_l169_169735

-- Definition of the problem in Lean
def is_valid_number (n : ℕ) : Prop :=
  n % 4 = 1 ∧ n % 17 = 1 ∧ 10 ≤ n ∧ n ≤ 99

theorem unique_two_digit_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 69 :=
by
  sorry

end unique_two_digit_number_l169_169735


namespace plane_equation_correct_l169_169111

-- Define points A, B, and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := { x := 1, y := -1, z := 8 }
def B : Point3D := { x := -4, y := -3, z := 10 }
def C : Point3D := { x := -1, y := -1, z := 7 }

-- Define the vector BC
def vecBC (B C : Point3D) : Point3D :=
  { x := C.x - B.x, y := C.y - B.y, z := C.z - B.z }

-- Define the equation of the plane
def planeEquation (P : Point3D) (normal : Point3D) : ℝ × ℝ × ℝ × ℝ :=
  (normal.x, normal.y, normal.z, -(normal.x * P.x + normal.y * P.y + normal.z * P.z))

-- Calculate the equation of the plane passing through A and perpendicular to vector BC
def planeThroughAperpToBC : ℝ × ℝ × ℝ × ℝ :=
  let normal := vecBC B C
  planeEquation A normal

-- The expected result
def expectedPlaneEquation : ℝ × ℝ × ℝ × ℝ := (3, 2, -3, 23)

-- The theorem to be proved
theorem plane_equation_correct : planeThroughAperpToBC = expectedPlaneEquation := by
  sorry

end plane_equation_correct_l169_169111


namespace solution_when_a_is_1_solution_for_arbitrary_a_l169_169488

-- Let's define the inequality and the solution sets
def inequality (a x : ℝ) : Prop :=
  ((a + 1) * x - 3) / (x - 1) < 1

def solutionSet_a_eq_1 (x : ℝ) : Prop :=
  1 < x ∧ x < 2

def solutionSet_a_eq_0 (x : ℝ) : Prop :=
  1 < x
  
def solutionSet_a_lt_0 (a x : ℝ) : Prop :=
  x < (2 / a) ∨ 1 < x

def solutionSet_0_lt_a_lt_2 (a x : ℝ) : Prop :=
  1 < x ∧ x < (2 / a)

def solutionSet_a_eq_2 : Prop :=
  false

def solutionSet_a_gt_2 (a x : ℝ) : Prop :=
  (2 / a) < x ∧ x < 1

-- Prove the solution for a = 1
theorem solution_when_a_is_1 : ∀ (x : ℝ), inequality 1 x ↔ solutionSet_a_eq_1 x :=
by sorry

-- Prove the solution for arbitrary real number a
theorem solution_for_arbitrary_a : ∀ (a x : ℝ),
  (a < 0 → inequality a x ↔ solutionSet_a_lt_0 a x) ∧
  (a = 0 → inequality a x ↔ solutionSet_a_eq_0 x) ∧
  (0 < a ∧ a < 2 → inequality a x ↔ solutionSet_0_lt_a_lt_2 a x) ∧
  (a = 2 → inequality a x → solutionSet_a_eq_2) ∧
  (a > 2 → inequality a x ↔ solutionSet_a_gt_2 a x) :=
by sorry

end solution_when_a_is_1_solution_for_arbitrary_a_l169_169488


namespace foci_ellipsoid_hyperboloid_l169_169353

theorem foci_ellipsoid_hyperboloid (a b : ℝ) 
(h1 : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → dist (0,y) (0, 5) = 5)
(h2 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → dist (x,0) (7, 0) = 7) :
  |a * b| = Real.sqrt 444 := sorry

end foci_ellipsoid_hyperboloid_l169_169353


namespace evaluate_g_expressions_l169_169213

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_g_expressions : 3 * g 5 + 4 * g (-2) = 287 := by
  sorry

end evaluate_g_expressions_l169_169213


namespace range_of_x_l169_169493

-- Define the function h(a).
def h (a : ℝ) : ℝ := a^2 + 2 * a + 3

-- Define the main theorem
theorem range_of_x (a : ℝ) (x : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) : 
  x^2 + 4 * x - 2 ≤ h a → -5 ≤ x ∧ x ≤ 1 :=
sorry

end range_of_x_l169_169493


namespace number_of_white_tiles_l169_169699

theorem number_of_white_tiles (n : ℕ) : 
  ∃ a_n : ℕ, a_n = 4 * n + 2 :=
sorry

end number_of_white_tiles_l169_169699


namespace ratio_expression_value_l169_169101

theorem ratio_expression_value (x y : ℝ) (h : x ≠ 0) (h' : y ≠ 0) (h_eq : x^2 - y^2 = x + y) : 
  x / y + y / x = 2 + 1 / (y^2 + y) :=
by
  sorry

end ratio_expression_value_l169_169101


namespace probability_is_correct_l169_169947

noncomputable def probability_total_more_than_7 : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 15
  favorable_outcomes / total_outcomes

theorem probability_is_correct :
  probability_total_more_than_7 = 5 / 12 :=
by
  sorry

end probability_is_correct_l169_169947


namespace inversely_proportional_rs_l169_169683

theorem inversely_proportional_rs (r s : ℝ) (k : ℝ) 
(h_invprop : r * s = k) 
(h1 : r = 40) (h2 : s = 5) 
(h3 : s = 8) : r = 25 := by
  sorry

end inversely_proportional_rs_l169_169683


namespace range_a_ge_one_l169_169399

theorem range_a_ge_one (a : ℝ) (x : ℝ) 
  (p : Prop := |x + 1| > 2) 
  (q : Prop := x > a) 
  (suff_not_necess_cond : ¬p → ¬q) : a ≥ 1 :=
sorry

end range_a_ge_one_l169_169399


namespace sequence_values_l169_169141

theorem sequence_values (x y z : ℕ) 
    (h1 : x = 14 * 3) 
    (h2 : y = x - 1) 
    (h3 : z = y * 3) : 
    x = 42 ∧ y = 41 ∧ z = 123 := by 
    sorry

end sequence_values_l169_169141


namespace find_original_number_l169_169358

def is_valid_digit (d : ℕ) : Prop := d < 10

def original_number (a b c : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
  222 * (a + b + c) - 5 * (100 * a + 10 * b + c) = 3194

theorem find_original_number (a b c : ℕ) (h_valid: is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c)
  (h_sum : 222 * (a + b + c) - 5 * (100 * a + 10 * b + c) = 3194) : 
  100 * a + 10 * b + c = 358 := 
sorry

end find_original_number_l169_169358


namespace proof_problem_l169_169407

theorem proof_problem (a b : ℝ) (H1 : ∀ x : ℝ, (ax^2 - 3*x + 6 > 4) ↔ (x < 1 ∨ x > b)) :
  a = 1 ∧ b = 2 ∧
  (∀ c : ℝ, (ax^2 - (a*c + b)*x + b*c < 0) ↔ 
   (if c > 2 then 2 < x ∧ x < c
    else if c < 2 then c < x ∧ x < 2
    else false)) :=
by
  sorry

end proof_problem_l169_169407


namespace fractional_part_exceeds_bound_l169_169680

noncomputable def x (a b : ℕ) : ℝ := Real.sqrt a + Real.sqrt b

theorem fractional_part_exceeds_bound
  (a b : ℕ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hx_not_int : ¬ (∃ n : ℤ, x a b = n))
  (hx_lt : x a b < 1976) :
    x a b % 1 > 3.24e-11 :=
sorry

end fractional_part_exceeds_bound_l169_169680


namespace double_series_evaluation_l169_169195

theorem double_series_evaluation :
    (∑' m : ℕ, ∑' n : ℕ, (1 : ℝ) / (m * n * (m + n + 2))) = (3 / 2 : ℝ) :=
sorry

end double_series_evaluation_l169_169195


namespace tetrahedron_volume_formula_l169_169402

variables (r₀ S₀ S₁ S₂ S₃ V : ℝ)

theorem tetrahedron_volume_formula
  (h : V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀) :
  V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀ :=
by { sorry }

end tetrahedron_volume_formula_l169_169402


namespace we_the_people_cows_l169_169104

theorem we_the_people_cows (W : ℕ) (h1 : ∃ H : ℕ, H = 3 * W + 2) (h2 : W + 3 * W + 2 = 70) : W = 17 :=
sorry

end we_the_people_cows_l169_169104


namespace find_line_equation_through_ellipse_midpoint_l169_169897

theorem find_line_equation_through_ellipse_midpoint {A B : ℝ × ℝ} 
  (hA : (A.fst^2 / 2) + A.snd^2 = 1) 
  (hB : (B.fst^2 / 2) + B.snd^2 = 1) 
  (h_midpoint : (A.fst + B.fst) / 2 = 1 ∧ (A.snd + B.snd) / 2 = 1 / 2) : 
  ∃ k : ℝ, (k = -1) ∧ (∀ x y : ℝ, (y - 1/2 = k * (x - 1)) → 2*x + 2*y - 3 = 0) :=
sorry

end find_line_equation_through_ellipse_midpoint_l169_169897


namespace Hoelder_l169_169225

variable (A B p q : ℝ)

theorem Hoelder (hA : 0 < A) (hB : 0 < B) (hp : 0 < p) (hq : 0 < q) (h : 1 / p + 1 / q = 1) : 
  A^(1/p) * B^(1/q) ≤ A / p + B / q := 
sorry

end Hoelder_l169_169225


namespace compute_modulo_l169_169059

theorem compute_modulo :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end compute_modulo_l169_169059


namespace average_income_l169_169523

theorem average_income :
  let income_day1 := 300
  let income_day2 := 150
  let income_day3 := 750
  let income_day4 := 200
  let income_day5 := 600
  (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / 5 = 400 := by
  sorry

end average_income_l169_169523


namespace part1_part2_part3_l169_169795

-- Part 1
theorem part1 (a b : ℝ) : 
    3 * (a - b) ^ 2 - 6 * (a - b) ^ 2 + 2 * (a - b) ^ 2 = - (a - b) ^ 2 := 
    sorry

-- Part 2
theorem part2 (x y : ℝ) (h : x ^ 2 - 2 * y = 4) : 
    3 * x ^ 2 - 6 * y - 21 = -9 := 
    sorry

-- Part 3
theorem part3 (a b c d : ℝ) (h1 : a - 5 * b = 3) (h2 : 5 * b - 3 * c = -5) (h3 : 3 * c - d = 10) : 
    (a - 3 * c) + (5 * b - d) - (5 * b - 3 * c) = 8 := 
    sorry

end part1_part2_part3_l169_169795


namespace dinner_potatoes_l169_169169

def lunch_potatoes : ℕ := 5
def total_potatoes : ℕ := 7

theorem dinner_potatoes : total_potatoes - lunch_potatoes = 2 :=
by
  sorry

end dinner_potatoes_l169_169169


namespace coordinates_of_P_tangent_line_equation_l169_169502

-- Define point P and center of the circle
def point_P : ℝ × ℝ := (-2, 1)
def center_C : ℝ × ℝ := (-1, 0)

-- Define the circle equation (x + 1)^2 + y^2 = 2
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the tangent line at point P
def tangent_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Prove the coordinates of point P are (-2, 1) given the conditions
theorem coordinates_of_P (n : ℝ) (h1 : n > 0) (h2 : circle_equation (-2) n) :
  point_P = (-2, 1) :=
by
  -- Proof steps would go here
  sorry

-- Prove the equation of the tangent line to the circle C passing through point P is x - y + 3 = 0
theorem tangent_line_equation :
  tangent_line (-2) 1 :=
by
  -- Proof steps would go here
  sorry

end coordinates_of_P_tangent_line_equation_l169_169502


namespace find_m_if_extraneous_root_l169_169283

theorem find_m_if_extraneous_root :
  (∃ x : ℝ, x = 2 ∧ (∀ z : ℝ, z ≠ 2 → (m / (z-2) - 2*z / (2-z) = 1)) ∧ m = -4) :=
sorry

end find_m_if_extraneous_root_l169_169283


namespace total_seashells_l169_169729

-- Conditions
def sam_seashells : Nat := 18
def mary_seashells : Nat := 47

-- Theorem stating the question and the final answer
theorem total_seashells : sam_seashells + mary_seashells = 65 :=
by
  sorry

end total_seashells_l169_169729


namespace prod_of_extrema_l169_169060

noncomputable def f (x k : ℝ) : ℝ := (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

theorem prod_of_extrema (k : ℝ) (h : ∀ x : ℝ, f x k ≥ 0 ∧ f x k ≤ 1 + (k - 1) / 3) :
  (∀ x : ℝ, f x k ≤ (k + 2) / 3) ∧ (∀ x : ℝ, f x k ≥ 1) → 
  (∃ φ ψ : ℝ, φ = 1 ∧ ψ = (k + 2) / 3 ∧ ∀ x y : ℝ, f x k = φ → f y k = ψ) → 
  (∃ φ ψ : ℝ, φ * ψ = (k + 2) / 3) :=
sorry

end prod_of_extrema_l169_169060


namespace shaded_area_l169_169177

theorem shaded_area (side_len : ℕ) (triangle_base : ℕ) (triangle_height : ℕ)
  (h1 : side_len = 40) (h2 : triangle_base = side_len / 2)
  (h3 : triangle_height = side_len / 2) : 
  side_len^2 - 2 * (1/2 * triangle_base * triangle_height) = 1200 := 
  sorry

end shaded_area_l169_169177


namespace value_of_expression_l169_169832

theorem value_of_expression (x y : ℝ) (h1 : x = Real.sqrt 5 + Real.sqrt 3) (h2 : y = Real.sqrt 5 - Real.sqrt 3) : x^2 + x * y + y^2 = 18 :=
by sorry

end value_of_expression_l169_169832


namespace a2b_etc_ge_9a2b2c2_l169_169408

theorem a2b_etc_ge_9a2b2c2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 :=
by
  sorry

end a2b_etc_ge_9a2b2c2_l169_169408


namespace counterpositive_prop_l169_169215

theorem counterpositive_prop (a b c : ℝ) (h : a^2 + b^2 + c^2 < 3) : a + b + c ≠ 3 := 
sorry

end counterpositive_prop_l169_169215


namespace revenue_increase_l169_169227

theorem revenue_increase (P Q : ℝ) :
    let R := P * Q
    let P_new := 1.7 * P
    let Q_new := 0.8 * Q
    let R_new := P_new * Q_new
    R_new = 1.36 * R :=
sorry

end revenue_increase_l169_169227


namespace exists_five_numbers_l169_169338

theorem exists_five_numbers :
  ∃ a1 a2 a3 a4 a5 : ℤ,
  a1 + a2 < 0 ∧
  a2 + a3 < 0 ∧
  a3 + a4 < 0 ∧
  a4 + a5 < 0 ∧
  a5 + a1 < 0 ∧
  a1 + a2 + a3 + a4 + a5 > 0 :=
by
  sorry

end exists_five_numbers_l169_169338


namespace cowboy_shortest_distance_l169_169189

noncomputable def distance : ℝ :=
  let C := (0, 5)
  let B := (-10, 11)
  let C' := (0, -5)
  5 + Real.sqrt ((C'.1 - B.1)^2 + (C'.2 - B.2)^2)

theorem cowboy_shortest_distance :
  distance = 5 + Real.sqrt 356 :=
by
  sorry

end cowboy_shortest_distance_l169_169189


namespace multiple_of_students_in_restroom_l169_169598

theorem multiple_of_students_in_restroom 
    (num_desks_per_row : ℕ)
    (num_rows : ℕ)
    (desk_fill_fraction : ℚ)
    (total_students : ℕ)
    (students_restroom : ℕ)
    (absent_students : ℕ)
    (m : ℕ) :
    num_desks_per_row = 6 →
    num_rows = 4 →
    desk_fill_fraction = 2 / 3 →
    total_students = 23 →
    students_restroom = 2 →
    (num_rows * num_desks_per_row : ℕ) * desk_fill_fraction = 16 →
    (16 - students_restroom) = 14 →
    total_students - 14 - 2 = absent_students →
    absent_students = 7 →
    2 * m - 1 = 7 →
    m = 4
:= by
    intros;
    sorry

end multiple_of_students_in_restroom_l169_169598


namespace james_parking_tickets_l169_169572

-- Define the conditions
def ticket_cost_1 := 150
def ticket_cost_2 := 150
def ticket_cost_3 := 1 / 3 * ticket_cost_1
def total_cost := ticket_cost_1 + ticket_cost_2 + ticket_cost_3
def roommate_pays := total_cost / 2
def james_remaining_money := 325
def james_original_money := james_remaining_money + roommate_pays

-- Define the theorem we want to prove
theorem james_parking_tickets (h1: ticket_cost_1 = 150)
                              (h2: ticket_cost_1 = ticket_cost_2)
                              (h3: ticket_cost_3 = 1 / 3 * ticket_cost_1)
                              (h4: total_cost = ticket_cost_1 + ticket_cost_2 + ticket_cost_3)
                              (h5: roommate_pays = total_cost / 2)
                              (h6: james_remaining_money = 325)
                              (h7: james_original_money = james_remaining_money + roommate_pays):
                              total_cost = 350 :=
by
  sorry

end james_parking_tickets_l169_169572


namespace dave_apps_files_difference_l169_169966

theorem dave_apps_files_difference :
  let initial_apps := 15
  let initial_files := 24
  let final_apps := 21
  let final_files := 4
  final_apps - final_files = 17 :=
by
  intros
  sorry

end dave_apps_files_difference_l169_169966


namespace log_eq_condition_pq_l169_169899

theorem log_eq_condition_pq :
  ∀ (p q : ℝ), p > 0 → q > 0 → (Real.log p + Real.log q = Real.log (2 * p + q)) → p = 3 ∧ q = 3 :=
by
  intros p q hp hq hlog
  sorry

end log_eq_condition_pq_l169_169899


namespace expression_expansion_l169_169325

noncomputable def expand_expression : Polynomial ℤ :=
 -2 * (5 * Polynomial.X^3 - 7 * Polynomial.X^2 + Polynomial.X - 4)

theorem expression_expansion :
  expand_expression = -10 * Polynomial.X^3 + 14 * Polynomial.X^2 - 2 * Polynomial.X + 8 :=
by
  sorry

end expression_expansion_l169_169325


namespace benzoic_acid_molecular_weight_l169_169348

-- Definitions for atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Molecular formula for Benzoic acid: C7H6O2
def benzoic_acid_formula : ℕ × ℕ × ℕ := (7, 6, 2)

-- Definition for the molecular weight calculation
def molecular_weight := λ (c h o : ℝ) (nC nH nO : ℕ) => 
  (nC * c) + (nH * h) + (nO * o)

-- Proof statement
theorem benzoic_acid_molecular_weight :
  molecular_weight atomic_weight_C atomic_weight_H atomic_weight_O 7 6 2 = 122.118 := by
  sorry

end benzoic_acid_molecular_weight_l169_169348


namespace inequality_sqrt_three_l169_169791

theorem inequality_sqrt_three (a b : ℤ) (h1 : a > b) (h2 : b > 1)
  (h3 : (a + b) ∣ (a * b + 1))
  (h4 : (a - b) ∣ (a * b - 1)) : a < Real.sqrt 3 * b := by
  sorry

end inequality_sqrt_three_l169_169791


namespace necessary_and_sufficient_condition_l169_169652

variable (x a : ℝ)

-- Condition 1: For all x in [1, 2], x^2 - a ≥ 0
def condition1 (x a : ℝ) : Prop := 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

-- Condition 2: There exists an x in ℝ such that x^2 + 2ax + 2 - a = 0
def condition2 (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Proof problem: The necessary and sufficient condition for p ∧ q is a ≤ -2 ∨ a = 1
theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) ↔ (a ≤ -2 ∨ a = 1) :=
sorry

end necessary_and_sufficient_condition_l169_169652


namespace opposite_exprs_have_value_l169_169464

theorem opposite_exprs_have_value (x : ℝ) : (4 * x - 8 = -(3 * x - 6)) → x = 2 :=
by
  intro h
  sorry

end opposite_exprs_have_value_l169_169464


namespace rational_add_positive_square_l169_169610

theorem rational_add_positive_square (a : ℚ) : a^2 + 1 > 0 := by
  sorry

end rational_add_positive_square_l169_169610


namespace total_pure_acid_in_mixture_l169_169873

-- Definitions of the conditions
def solution1_volume : ℝ := 8
def solution1_concentration : ℝ := 0.20
def solution2_volume : ℝ := 5
def solution2_concentration : ℝ := 0.35

-- Proof statement
theorem total_pure_acid_in_mixture :
  solution1_volume * solution1_concentration + solution2_volume * solution2_concentration = 3.35 := by
  sorry

end total_pure_acid_in_mixture_l169_169873


namespace smallest_d_l169_169803

-- Constants and conditions
variables (c d : ℝ)
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions involving c and d
def conditions (c d : ℝ) : Prop :=
  2 < c ∧ c < d ∧ ¬triangle_inequality 2 c d ∧ ¬triangle_inequality (1/d) (1/c) 2

-- Goal statement: the smallest possible value of d
theorem smallest_d (c d : ℝ) (h : conditions c d) : d = 2 + Real.sqrt 2 :=
sorry

end smallest_d_l169_169803


namespace min_value_expression_l169_169655

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ (m : ℝ), m = 3 / 2 ∧ ∀ t > 0, (2 * x / (x + 2 * y) + y / x) ≥ m :=
by
  use 3 / 2
  sorry

end min_value_expression_l169_169655


namespace turban_as_part_of_salary_l169_169956

-- Definitions of the given conditions
def annual_salary (T : ℕ) : ℕ := 90 + 70 * T
def nine_month_salary (T : ℕ) : ℕ := 3 * (90 + 70 * T) / 4
def leaving_amount : ℕ := 50 + 70

-- Proof problem statement in Lean 4
theorem turban_as_part_of_salary (T : ℕ) (h : nine_month_salary T = leaving_amount) : T = 1 := 
sorry

end turban_as_part_of_salary_l169_169956


namespace opposite_of_two_thirds_l169_169678

theorem opposite_of_two_thirds : - (2/3) = -2/3 :=
by
  sorry

end opposite_of_two_thirds_l169_169678


namespace train_speed_initial_l169_169452

variable (x : ℝ)
variable (v : ℝ)
variable (average_speed : ℝ := 40 / 3)
variable (initial_distance : ℝ := x)
variable (initial_speed : ℝ := v)
variable (next_distance : ℝ := 4 * x)
variable (next_speed : ℝ := 20)

theorem train_speed_initial : 
  (5 * x) / ((x / v) + (x / 5)) = 40 / 3 → v = 40 / 7 :=
by
  -- Definition of average speed in the context of the problem
  let t1 := x / v
  let t2 := (4 * x) / 20
  let total_distance := 5 * x
  let total_time := t1 + t2
  have avg_speed_eq : total_distance / total_time = 40 / 3 := by sorry
  sorry

end train_speed_initial_l169_169452


namespace find_larger_number_l169_169070

theorem find_larger_number 
  (x y : ℤ)
  (h1 : x + y = 37)
  (h2 : x - y = 5) : max x y = 21 := 
sorry

end find_larger_number_l169_169070


namespace mixed_operation_with_rationals_l169_169318

theorem mixed_operation_with_rationals :
  (- (2 / 21)) / (1 / 6 - 3 / 14 + 2 / 3 - 9 / 7) = 1 / 7 := 
by 
  sorry

end mixed_operation_with_rationals_l169_169318


namespace number_of_sandwiches_l169_169264

-- Define the constants and assumptions

def soda_cost : ℤ := 1
def number_of_sodas : ℤ := 3
def cost_of_sodas : ℤ := number_of_sodas * soda_cost

def number_of_soups : ℤ := 2
def soup_cost : ℤ := cost_of_sodas
def cost_of_soups : ℤ := number_of_soups * soup_cost

def sandwich_cost : ℤ := 3 * soup_cost
def total_cost : ℤ := 18

-- The mathematical statement we want to prove
theorem number_of_sandwiches :
  ∃ n : ℤ, (n * sandwich_cost + cost_of_sodas + cost_of_soups = total_cost) ∧ n = 1 :=
by
  sorry

end number_of_sandwiches_l169_169264


namespace solve_for_x_l169_169865

theorem solve_for_x :
  ∃ x : ℝ, (1 / (x + 5) + 1 / (x + 2) + 1 / (2 * x) = 1 / x) ∧ (1 / (x + 5) + 1 / (x + 2) = 1 / (x + 3)) ∧ x = 2 :=
by
  sorry

end solve_for_x_l169_169865


namespace part1_part2_l169_169098

section 
variable {a b : ℚ}

-- Define the new operation as given in the condition
def odot (a b : ℚ) : ℚ := a * (a + b) - 1

-- Prove the given results
theorem part1 : odot 3 (-2) = 2 :=
by
  -- Proof omitted
  sorry

theorem part2 : odot (-2) (odot 3 5) = -43 :=
by
  -- Proof omitted
  sorry

end

end part1_part2_l169_169098


namespace roots_real_roots_equal_l169_169633

noncomputable def discriminant (a : ℝ) : ℝ :=
  let b := 4 * a
  let c := 2 * a^2 - 1 + 3 * a
  b^2 - 4 * 1 * c

theorem roots_real (a : ℝ) : discriminant a ≥ 0 ↔ a ≤ 1/2 ∨ a ≥ 1 := sorry

theorem roots_equal (a : ℝ) : discriminant a = 0 ↔ a = 1 ∨ a = 1/2 := sorry

end roots_real_roots_equal_l169_169633


namespace solve_for_constants_l169_169557

theorem solve_for_constants : 
  ∃ (t s : ℚ), (∀ x : ℚ, (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + 12) = 15 * x^4 + s * x^3 + 33 * x^2 + 12 * x + 108) ∧ 
  t = 37 / 5 ∧ 
  s = 11 / 5 :=
by
  sorry

end solve_for_constants_l169_169557


namespace new_stamps_ratio_l169_169392

theorem new_stamps_ratio (x : ℕ) (h1 : 7 * x = P) (h2 : 4 * x = Q)
  (h3 : P - 8 = 8 + (Q + 8)) : (P - 8) / gcd (P - 8) (Q + 8) = 6 ∧ (Q + 8) / gcd (P - 8) (Q + 8) = 5 :=
by
  sorry

end new_stamps_ratio_l169_169392


namespace expressing_population_in_scientific_notation_l169_169345

def population_in_scientific_notation (population : ℝ) : Prop :=
  population = 1.412 * 10^9

theorem expressing_population_in_scientific_notation : 
  population_in_scientific_notation (1.412 * 10^9) :=
by
  sorry

end expressing_population_in_scientific_notation_l169_169345


namespace stephen_hawking_philosophical_implications_l169_169538

/-- Stephen Hawking's statements -/
def stephen_hawking_statement_1 := "The universe was not created by God"
def stephen_hawking_statement_2 := "Modern science can explain the origin of the universe"

/-- Definitions implied by Hawking's statements -/
def unity_of_world_lies_in_materiality := "The unity of the world lies in its materiality"
def thought_and_existence_identical := "Thought and existence are identical"

/-- Combined implication of Stephen Hawking's statements -/
def correct_philosophical_implications := [unity_of_world_lies_in_materiality, thought_and_existence_identical]

/-- Theorem: The correct philosophical implications of Stephen Hawking's statements are ① and ②. -/
theorem stephen_hawking_philosophical_implications :
  (stephen_hawking_statement_1 = "The universe was not created by God") →
  (stephen_hawking_statement_2 = "Modern science can explain the origin of the universe") →
  correct_philosophical_implications = ["The unity of the world lies in its materiality", "Thought and existence are identical"] :=
by
  sorry

end stephen_hawking_philosophical_implications_l169_169538


namespace evaluate_f_g3_l169_169090

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 1
def g (x : ℝ) : ℝ := x + 3

theorem evaluate_f_g3 : f (g 3) = 97 := by
  sorry

end evaluate_f_g3_l169_169090


namespace Carla_total_marbles_l169_169585

def initial_marbles : ℝ := 187.0
def bought_marbles : ℝ := 134.0

theorem Carla_total_marbles : initial_marbles + bought_marbles = 321.0 := 
by 
  sorry

end Carla_total_marbles_l169_169585


namespace bakery_water_requirement_l169_169914

theorem bakery_water_requirement (flour water : ℕ) (total_flour : ℕ) (h : flour = 300) (w : water = 75) (t : total_flour = 900) : 
  225 = (total_flour / flour) * water :=
by
  sorry

end bakery_water_requirement_l169_169914


namespace other_train_speed_l169_169503

noncomputable def speed_of_other_train (l1 l2 v1 : ℕ) (t : ℝ) : ℝ := 
  let relative_speed := (l1 + l2) / 1000 / (t / 3600)
  relative_speed - v1

theorem other_train_speed :
  speed_of_other_train 210 260 40 16.918646508279338 = 60 := 
by
  sorry

end other_train_speed_l169_169503


namespace pet_store_profit_is_205_l169_169665

def brandon_selling_price : ℤ := 100
def pet_store_selling_price : ℤ := 5 + 3 * brandon_selling_price
def pet_store_profit : ℤ := pet_store_selling_price - brandon_selling_price

theorem pet_store_profit_is_205 :
  pet_store_profit = 205 := by
  sorry

end pet_store_profit_is_205_l169_169665


namespace independent_variable_range_l169_169099

theorem independent_variable_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) ↔ x ≥ 0 ∧ x ≠ 1 := 
by
  sorry

end independent_variable_range_l169_169099


namespace complex_number_powers_l169_169293

theorem complex_number_powers (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^97 + z^98 + z^99 + z^100 + z^101 = -1 :=
sorry

end complex_number_powers_l169_169293


namespace sufficient_but_not_necessary_condition_l169_169603

theorem sufficient_but_not_necessary_condition (a : ℝ) (h₁ : a > 2) : a ≥ 1 ∧ ¬(∀ (a : ℝ), a ≥ 1 → a > 2) := 
by
  sorry

end sufficient_but_not_necessary_condition_l169_169603


namespace multiply_and_simplify_l169_169007

variable (a b : ℝ)

theorem multiply_and_simplify :
  (3 * a + 2 * b) * (a - 2 * b) = 3 * a^2 - 4 * a * b - 4 * b^2 :=
by
  sorry

end multiply_and_simplify_l169_169007


namespace polynomial_sum_of_coefficients_l169_169466

theorem polynomial_sum_of_coefficients {v : ℕ → ℝ} (h1 : v 1 = 7)
  (h2 : ∀ n : ℕ, v (n + 1) - v n = 5 * n - 2) :
  ∃ (a b c : ℝ), (∀ n : ℕ, v n = a * n^2 + b * n + c) ∧ (a + b + c = 7) :=
by
  sorry

end polynomial_sum_of_coefficients_l169_169466


namespace decreasing_sufficient_condition_l169_169412

theorem decreasing_sufficient_condition {a : ℝ} (h_pos : 0 < a) (h_neq_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x > a^y) →
  (∀ x y : ℝ, x < y → (a-2)*x^3 > (a-2)*y^3) :=
by
  sorry

end decreasing_sufficient_condition_l169_169412


namespace polynomial_remainder_l169_169535

theorem polynomial_remainder (x : ℂ) :
  (x ^ 2030 + 1) % (x ^ 6 - x ^ 4 + x ^ 2 - 1) = x ^ 2 - 1 :=
by
  sorry

end polynomial_remainder_l169_169535


namespace good_carrots_total_l169_169457

-- Define the number of carrots picked by Carol and her mother
def carolCarrots := 29
def motherCarrots := 16

-- Define the number of bad carrots
def badCarrots := 7

-- Define the total number of carrots picked by Carol and her mother
def totalCarrots := carolCarrots + motherCarrots

-- Define the total number of good carrots
def goodCarrots := totalCarrots - badCarrots

-- The theorem to prove that the total number of good carrots is 38
theorem good_carrots_total : goodCarrots = 38 := by
  sorry

end good_carrots_total_l169_169457


namespace discount_rate_l169_169079

variable (P P_b P_s D : ℝ)

-- Conditions
variable (h1 : P_s = 1.24 * P)
variable (h2 : P_s = 1.55 * P_b)
variable (h3 : P_b = P * (1 - D))

theorem discount_rate :
  D = 0.2 :=
by
  sorry

end discount_rate_l169_169079


namespace distance_between_sets_is_zero_l169_169745

noncomputable def A (x : ℝ) : ℝ := 2 * x - 1
noncomputable def B (x : ℝ) : ℝ := x^2 + 1

theorem distance_between_sets_is_zero : 
  ∃ (a b : ℝ), (∃ x₀ : ℝ, a = A x₀) ∧ (∃ y₀ : ℝ, b = B y₀) ∧ abs (a - b) = 0 := 
sorry

end distance_between_sets_is_zero_l169_169745


namespace jars_contain_k_balls_eventually_l169_169881

theorem jars_contain_k_balls_eventually
  (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hkp : k < 2 * p + 1) :
  ∃ n : ℕ, ∃ x y : ℕ, x + y = 2 * p + 1 ∧ (x = k ∨ y = k) :=
by
  sorry

end jars_contain_k_balls_eventually_l169_169881


namespace solve_quadratic_l169_169971

theorem solve_quadratic (x : ℝ) : x^2 - 4 * x + 3 = 0 ↔ (x = 1 ∨ x = 3) :=
by
  sorry

end solve_quadratic_l169_169971


namespace age_ratio_problem_l169_169575

def age_condition (s a : ℕ) : Prop :=
  s - 2 = 2 * (a - 2) ∧ s - 4 = 3 * (a - 4)

def future_ratio (s a x : ℕ) : Prop :=
  (s + x) * 2 = (a + x) * 3

theorem age_ratio_problem :
  ∃ s a x : ℕ, age_condition s a ∧ future_ratio s a x ∧ x = 2 :=
by
  sorry

end age_ratio_problem_l169_169575


namespace no_partition_exists_l169_169122

noncomputable section

open Set

def partition_N (A B C : Set ℕ) : Prop := 
  A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧  -- Non-empty sets
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧  -- Disjoint sets
  A ∪ B ∪ C = univ ∧  -- Covers the whole ℕ
  (∀ a ∈ A, ∀ b ∈ B, a + b + 2008 ∈ C) ∧
  (∀ b ∈ B, ∀ c ∈ C, b + c + 2008 ∈ A) ∧
  (∀ c ∈ C, ∀ a ∈ A, c + a + 2008 ∈ B)

theorem no_partition_exists : ¬ ∃ (A B C : Set ℕ), partition_N A B C :=
by
  sorry

end no_partition_exists_l169_169122


namespace remainder_of_P_div_D_is_25158_l169_169094

noncomputable def P (x : ℝ) := 4 * x^8 - 2 * x^6 + 5 * x^4 - x^3 + 3 * x - 15
def D (x : ℝ) := 2 * x - 6

theorem remainder_of_P_div_D_is_25158 : P 3 = 25158 := by
  sorry

end remainder_of_P_div_D_is_25158_l169_169094


namespace anne_equals_bob_l169_169328

-- Define the conditions as constants and functions
def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.06
def discount_rate : ℝ := 0.25

-- Calculation models for Anne and Bob
def anne_total (price : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  (price * (1 + tax)) * (1 - discount)

def bob_total (price : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  (price * (1 - discount)) * (1 + tax)

-- The theorem that states what we need to prove
theorem anne_equals_bob : anne_total original_price tax_rate discount_rate = bob_total original_price tax_rate discount_rate :=
by
  sorry

end anne_equals_bob_l169_169328


namespace commute_times_variance_l169_169739

theorem commute_times_variance (x y : ℝ) :
  (x + y + 10 + 11 + 9) / 5 = 10 ∧
  ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2 →
  |x - y| = 4 :=
by
  sorry

end commute_times_variance_l169_169739


namespace family_age_problem_l169_169929

theorem family_age_problem (T y : ℕ)
  (h1 : T = 5 * 17)
  (h2 : (T + 5 * y + 2) = 6 * 17)
  : y = 3 := by
  sorry

end family_age_problem_l169_169929


namespace lcm_one_to_twelve_l169_169629

theorem lcm_one_to_twelve : 
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 
  (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 10 (Nat.lcm 11 12)))))))))) = 27720 := 
by sorry

end lcm_one_to_twelve_l169_169629


namespace red_pieces_count_l169_169764

-- Define the conditions
def total_pieces : ℕ := 3409
def blue_pieces : ℕ := 3264

-- Prove the number of red pieces
theorem red_pieces_count : total_pieces - blue_pieces = 145 :=
by sorry

end red_pieces_count_l169_169764


namespace solve_system_of_equations_solve_algebraic_equation_l169_169878

-- Problem 1: System of Equations
theorem solve_system_of_equations (x y : ℝ) (h1 : x + 2 * y = 3) (h2 : 2 * x - y = 1) : x = 1 ∧ y = 1 :=
sorry

-- Problem 2: Algebraic Equation
theorem solve_algebraic_equation (x : ℝ) (h : 1 / (x - 1) + 2 = 5 / (1 - x)) : x = -2 :=
sorry

end solve_system_of_equations_solve_algebraic_equation_l169_169878


namespace selection_methods_eq_total_students_l169_169930

def num_boys := 36
def num_girls := 28
def total_students : ℕ := num_boys + num_girls

theorem selection_methods_eq_total_students :
    total_students = 64 :=
by
  -- Placeholder for the proof
  sorry

end selection_methods_eq_total_students_l169_169930


namespace find_f_at_8_l169_169818

theorem find_f_at_8 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (3 * x - 1) = x^2 + 2 * x + 4) :
  f 8 = 19 :=
sorry

end find_f_at_8_l169_169818


namespace annual_interest_earned_l169_169373
noncomputable section

-- Define the total money
def total_money : ℝ := 3200

-- Define the first part of the investment
def P1 : ℝ := 800

-- Define the second part of the investment as total money minus the first part
def P2 : ℝ := total_money - P1

-- Define the interest rates for both parts
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

-- Define the time period (in years)
def time_period : ℝ := 1

-- Define the interest earned from each part
def interest1 : ℝ := P1 * rate1 * time_period
def interest2 : ℝ := P2 * rate2 * time_period

-- The total interest earned from both investments
def total_interest : ℝ := interest1 + interest2

-- The proof statement
theorem annual_interest_earned : total_interest = 144 := by
  sorry

end annual_interest_earned_l169_169373


namespace number_of_elements_in_set_P_l169_169159

theorem number_of_elements_in_set_P
  (p q : ℕ) -- we are dealing with non-negative integers here
  (h1 : p = 3 * q)
  (h2 : p + q = 4500)
  : p = 3375 :=
by
  sorry -- Proof goes here

end number_of_elements_in_set_P_l169_169159


namespace divide_oranges_into_pieces_l169_169448

-- Definitions for conditions
def oranges : Nat := 80
def friends : Nat := 200
def pieces_per_friend : Nat := 4

-- Theorem stating the problem and the answer
theorem divide_oranges_into_pieces :
    (oranges > 0) → (friends > 0) → (pieces_per_friend > 0) →
    ((friends * pieces_per_friend) / oranges = 10) :=
by
  intros
  sorry

end divide_oranges_into_pieces_l169_169448


namespace chris_wins_l169_169977

noncomputable def chris_heads : ℚ := 1 / 4
noncomputable def drew_heads : ℚ := 1 / 3
noncomputable def both_tails : ℚ := (1 - chris_heads) * (1 - drew_heads)

/-- The probability that Chris wins comparing with relatively prime -/
theorem chris_wins (p q : ℕ) (hpq : Nat.Coprime p q) (hq0 : q ≠ 0) :
  (chris_heads * (1 + both_tails)) = (p : ℚ) / q ∧ (q - p = 1) :=
sorry

end chris_wins_l169_169977


namespace problem_solution_l169_169074

theorem problem_solution :
  ∀ (a b c d : ℝ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
    (a^2 = 7 ∨ a^2 = 8) →
    (b^2 = 7 ∨ b^2 = 8) →
    (c^2 = 7 ∨ c^2 = 8) →
    (d^2 = 7 ∨ d^2 = 8) →
    a^2 + b^2 + c^2 + d^2 = 30 :=
by sorry

end problem_solution_l169_169074


namespace circles_condition_l169_169064

noncomputable def circles_intersect_at (p1 p2 : ℝ × ℝ) (m c : ℝ) : Prop :=
  p1 = (1, 3) ∧ p2 = (m, 1) ∧ (∃ (x y : ℝ), (x - y + c / 2 = 0) ∧ 
    (p1.1 - x)^2 + (p1.2 - y)^2 = (p2.1 - x)^2 + (p2.2 - y)^2)

theorem circles_condition (m c : ℝ) (h : circles_intersect_at (1, 3) (m, 1) m c) : m + c = 3 :=
sorry

end circles_condition_l169_169064


namespace polygon_sides_l169_169065

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_l169_169065


namespace hamburgers_served_l169_169135

def hamburgers_made : ℕ := 9
def hamburgers_leftover : ℕ := 6

theorem hamburgers_served : ∀ (total : ℕ) (left : ℕ), total = hamburgers_made → left = hamburgers_leftover → total - left = 3 := 
by
  intros total left h_total h_left
  rw [h_total, h_left]
  rfl

end hamburgers_served_l169_169135


namespace value_of_algebraic_expression_l169_169802

noncomputable def quadratic_expression (m : ℝ) : ℝ :=
  3 * m * (2 * m - 3) - 1

theorem value_of_algebraic_expression (m : ℝ) (h : 2 * m^2 - 3 * m - 1 = 0) : quadratic_expression m = 2 :=
by {
  sorry
}

end value_of_algebraic_expression_l169_169802


namespace not_entire_field_weedy_l169_169925

-- Define the conditions
def field_divided_into_100_plots : Prop :=
  ∃ (a b : ℕ), a * b = 100

def initial_weedy_plots : Prop :=
  ∃ (weedy_plots : Finset (ℕ × ℕ)), weedy_plots.card = 9

def plot_becomes_weedy (weedy_plots : Finset (ℕ × ℕ)) (p : ℕ × ℕ) : Prop :=
  (p.fst ≠ 0 ∧ (p.fst - 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 0 ∧ (p.fst, p.snd - 1) ∈ weedy_plots) ∨
  (p.fst ≠ 0 ∧ (p.fst - 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 100 ∧ (p.fst, p.snd + 1) ∈ weedy_plots) ∨
  (p.fst ≠ 100 ∧ (p.fst + 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 0 ∧ (p.fst, p.snd - 1) ∈ weedy_plots) ∨
  (p.fst ≠ 100 ∧ (p.fst + 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 100 ∧ (p.fst, p.snd + 1) ∈ weedy_plots)

-- Theorem statement
theorem not_entire_field_weedy :
  field_divided_into_100_plots →
  initial_weedy_plots →
  (∀ weedy_plots : Finset (ℕ × ℕ), (∀ p : ℕ × ℕ, plot_becomes_weedy weedy_plots p → weedy_plots ∪ {p} = weedy_plots) → weedy_plots.card < 100) :=
  sorry

end not_entire_field_weedy_l169_169925


namespace fraction_subtraction_l169_169520

theorem fraction_subtraction (h : ((8 : ℚ) / 21 - (10 / 63) = (2 / 9))) : 
  8 / 21 - 10 / 63 = 2 / 9 :=
by
  sorry

end fraction_subtraction_l169_169520


namespace pyramid_base_side_length_l169_169206

theorem pyramid_base_side_length (area : ℕ) (slant_height : ℕ) (s : ℕ) 
  (h1 : area = 100) 
  (h2 : slant_height = 20) 
  (h3 : area = (1 / 2) * s * slant_height) :
  s = 10 := 
by 
  sorry

end pyramid_base_side_length_l169_169206


namespace arithmetic_mean_of_primes_l169_169558

variable (list : List ℕ) 
variable (primes : List ℕ)
variable (h1 : list = [24, 25, 29, 31, 33])
variable (h2 : primes = [29, 31])

theorem arithmetic_mean_of_primes : (primes.sum / primes.length : ℝ) = 30 := by
  sorry

end arithmetic_mean_of_primes_l169_169558


namespace inequality_solutions_l169_169321

theorem inequality_solutions (a : ℝ) (h_pos : 0 < a) 
  (h_ineq_1 : ∃! x : ℕ, 10 < a ^ x ∧ a ^ x < 100) : ∃! x : ℕ, 100 < a ^ x ∧ a ^ x < 1000 :=
by
  sorry

end inequality_solutions_l169_169321


namespace geometric_seq_ratio_l169_169290

theorem geometric_seq_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a (n+1) = q * a n)
  (h2 : 0 < q)                    -- ensuring positivity
  (h3 : 3 * a 0 + 2 * q * a 0 = q^2 * a 0)  -- condition from problem
  : ∀ n, (a (n+3) + a (n+2)) / (a (n+1) + a n) = 9 :=
by
  sorry

end geometric_seq_ratio_l169_169290


namespace find_b_l169_169091

-- Define what it means for b to be a solution
def is_solution (b : ℤ) : Prop :=
  b > 4 ∧ ∃ k : ℤ, 4 * b + 5 = k * k

-- State the problem
theorem find_b : ∃ b : ℤ, is_solution b ∧ ∀ b' : ℤ, is_solution b' → b' ≥ 5 := by
  sorry

end find_b_l169_169091


namespace intersection_A_B_l169_169560

-- Define sets A and B
def A : Set ℤ := {1, 3, 5}
def B : Set ℤ := {-1, 0, 1}

-- Prove that the intersection of A and B is {1}
theorem intersection_A_B : A ∩ B = {1} := by 
  sorry

end intersection_A_B_l169_169560


namespace robin_gum_packages_l169_169517

theorem robin_gum_packages (P : ℕ) (h1 : 7 * P + 6 = 41) : P = 5 :=
by
  sorry

end robin_gum_packages_l169_169517


namespace arithmetic_seq_proof_l169_169198

theorem arithmetic_seq_proof
  (x : ℕ → ℝ)
  (h : ∀ n ≥ 3, x (n-1) = (x n + x (n-1) + x (n-2)) / 3):
  (x 300 - x 33) / (x 333 - x 3) = 89 / 110 := by
  sorry

end arithmetic_seq_proof_l169_169198


namespace variance_stability_l169_169463

theorem variance_stability (S2_A S2_B : ℝ) (hA : S2_A = 1.1) (hB : S2_B = 2.5) : ¬(S2_B < S2_A) :=
by {
  sorry
}

end variance_stability_l169_169463


namespace min_value_of_y_l169_169251

theorem min_value_of_y (x : ℝ) : ∃ x0 : ℝ, (∀ x : ℝ, 4 * x^2 + 8 * x + 16 ≥ 12) ∧ (4 * x0^2 + 8 * x0 + 16 = 12) :=
sorry

end min_value_of_y_l169_169251


namespace find_g3_l169_169861

-- Define a function g from ℝ to ℝ
variable (g : ℝ → ℝ)

-- Condition: ∀ x, g(3^x) + 2 * x * g(3^(-x)) = 3
axiom condition : ∀ x : ℝ, g (3^x) + 2 * x * g (3^(-x)) = 3

-- The theorem we need to prove
theorem find_g3 : g 3 = -3 := 
by 
  sorry

end find_g3_l169_169861


namespace fraction_subtraction_simplify_l169_169144

theorem fraction_subtraction_simplify :
  (9 / 19 - 3 / 57 - 1 / 3) = 5 / 57 :=
by
  sorry

end fraction_subtraction_simplify_l169_169144


namespace focus_of_curve_is_4_0_l169_169714

noncomputable def is_focus (p : ℝ × ℝ) (curve : ℝ × ℝ → Prop) : Prop :=
  ∃ c : ℝ, ∀ x y : ℝ, curve (x, y) ↔ (y^2 = -16 * c * (x - 4))

def curve (p : ℝ × ℝ) : Prop := p.2^2 = -16 * p.1 + 64

theorem focus_of_curve_is_4_0 : is_focus (4, 0) curve :=
by
sorry

end focus_of_curve_is_4_0_l169_169714


namespace circle_radius_l169_169342

theorem circle_radius (k r : ℝ) (h : k > 8) 
  (h1 : r = |k - 8|)
  (h2 : r = k / Real.sqrt 5) : 
  r = 8 * Real.sqrt 5 + 8 := 
sorry

end circle_radius_l169_169342


namespace right_triangle_hypotenuse_l169_169043

theorem right_triangle_hypotenuse
  (a b c : ℝ)
  (h₀ : a = 24)
  (h₁ : a^2 + b^2 + c^2 = 2500)
  (h₂ : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end right_triangle_hypotenuse_l169_169043


namespace inequality_proof_l169_169856

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) +
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) +
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by
  sorry

end inequality_proof_l169_169856


namespace rectangle_cut_into_square_l169_169199

theorem rectangle_cut_into_square (a b : ℝ) (h : a ≤ 4 * b) : 4 * b ≥ a := 
by 
  exact h

end rectangle_cut_into_square_l169_169199


namespace initial_outlay_l169_169337

-- Definition of given conditions
def manufacturing_cost (I : ℝ) (sets : ℕ) (cost_per_set : ℝ) : ℝ := I + sets * cost_per_set
def revenue (sets : ℕ) (price_per_set : ℝ) : ℝ := sets * price_per_set
def profit (revenue manufacturing_cost : ℝ) : ℝ := revenue - manufacturing_cost

-- Given data
def sets : ℕ := 500
def cost_per_set : ℝ := 20
def price_per_set : ℝ := 50
def given_profit : ℝ := 5000

-- The statement to prove
theorem initial_outlay (I : ℝ) : 
  profit (revenue sets price_per_set) (manufacturing_cost I sets cost_per_set) = given_profit → 
  I = 10000 := by
  sorry

end initial_outlay_l169_169337


namespace profit_difference_l169_169289

variable (P : ℕ) -- P is the total profit
variable (r1 r2 : ℚ) -- r1 and r2 are the parts of the ratio for X and Y, respectively

noncomputable def X_share (P : ℕ) (r1 r2 : ℚ) : ℚ :=
  (r1 / (r1 + r2)) * P

noncomputable def Y_share (P : ℕ) (r1 r2 : ℚ) : ℚ :=
  (r2 / (r1 + r2)) * P

theorem profit_difference (P : ℕ) (r1 r2 : ℚ) (hP : P = 800) (hr1 : r1 = 1/2) (hr2 : r2 = 1/3) :
  X_share P r1 r2 - Y_share P r1 r2 = 160 := by
  sorry

end profit_difference_l169_169289


namespace maximum_value_of_reciprocals_l169_169431

theorem maximum_value_of_reciprocals (c b : ℝ) (h0 : 0 < b ∧ b < c)
  (e1 : ℝ) (e2 : ℝ)
  (h1 : e1 = c / (Real.sqrt (c^2 + (2 * b)^2)))
  (h2 : e2 = c / (Real.sqrt (c^2 - b^2)))
  (h3 : 1 / e1^2 + 4 / e2^2 = 5) :
  ∃ max_val, max_val = 5 / 2 :=
by
  sorry

end maximum_value_of_reciprocals_l169_169431


namespace minimum_cubes_required_l169_169146

def box_length := 12
def box_width := 16
def box_height := 6
def cube_volume := 3

def volume_box := box_length * box_width * box_height

theorem minimum_cubes_required : volume_box / cube_volume = 384 := by
  sorry

end minimum_cubes_required_l169_169146


namespace sample_size_correct_l169_169461

def population_size : Nat := 8000
def sampled_students : List Nat := List.replicate 400 1 -- We use 1 as a placeholder for the heights

theorem sample_size_correct : sampled_students.length = 400 := by
  sorry

end sample_size_correct_l169_169461


namespace max_a_plus_2b_l169_169055

theorem max_a_plus_2b (a b : ℝ) (h : a^2 + 2 * b^2 = 1) : a + 2 * b ≤ Real.sqrt 3 := 
sorry

end max_a_plus_2b_l169_169055


namespace smallest_bottom_right_value_l169_169915

theorem smallest_bottom_right_value :
  ∃ (grid : ℕ × ℕ × ℕ → ℕ), -- grid as a function from row/column pairs to natural numbers
    (∀ i j, 1 ≤ i ∧ i ≤ 3 → 1 ≤ j ∧ j ≤ 3 → grid (i, j) ≠ 0) ∧ -- all grid values are non-zero
    (grid (1, 1) ≠ grid (1, 2) ∧ grid (1, 1) ≠ grid (1, 3) ∧ grid (1, 2) ≠ grid (1, 3) ∧
     grid (2, 1) ≠ grid (2, 2) ∧ grid (2, 1) ≠ grid (2, 3) ∧ grid (2, 2) ≠ grid (2, 3) ∧
     grid (3, 1) ≠ grid (3, 2) ∧ grid (3, 1) ≠ grid (3, 3) ∧ grid (3, 2) ≠ grid (3, 3)) ∧ -- all grid values are distinct
    (grid (1, 1) + grid (1, 2) = grid (1, 3)) ∧ 
    (grid (2, 1) + grid (2, 2) = grid (2, 3)) ∧ 
    (grid (3, 1) + grid (3, 2) = grid (3, 3)) ∧ -- row sum conditions
    (grid (1, 1) + grid (2, 1) = grid (3, 1)) ∧ 
    (grid (1, 2) + grid (2, 2) = grid (3, 2)) ∧ 
    (grid (1, 3) + grid (2, 3) = grid (3, 3)) ∧ -- column sum conditions
    (grid (3, 3) = 12) :=
by
  sorry

end smallest_bottom_right_value_l169_169915


namespace minimum_even_N_for_A_2015_turns_l169_169380

noncomputable def a (n : ℕ) : ℕ :=
  6 * 2^n - 4

def A_minimum_even_moves_needed (k : ℕ) : ℕ :=
  2015 - 1

theorem minimum_even_N_for_A_2015_turns :
  ∃ N : ℕ, 2 ∣ N ∧ A_minimum_even_moves_needed 2015 ≤ N ∧ a 1007 = 6 * 2^1007 - 4 := by
  sorry

end minimum_even_N_for_A_2015_turns_l169_169380


namespace circle_inscribed_isosceles_trapezoid_l169_169306

theorem circle_inscribed_isosceles_trapezoid (r a c : ℝ) : 
  (∃ base1 base2 : ℝ,  2 * a = base1 ∧ 2 * c = base2) →
  (∃ O : ℝ, O = r) →
  r^2 = a * c :=
by
  sorry

end circle_inscribed_isosceles_trapezoid_l169_169306


namespace name_tag_area_l169_169720

-- Define the side length of the square
def side_length : ℕ := 11

-- Define the area calculation for a square
def square_area (side : ℕ) : ℕ := side * side

-- State the theorem: the area of a square with side length of 11 cm is 121 cm²
theorem name_tag_area : square_area side_length = 121 :=
by
  sorry

end name_tag_area_l169_169720


namespace pieces_of_paper_picked_up_l169_169955

theorem pieces_of_paper_picked_up (Olivia : ℕ) (Edward : ℕ) (h₁ : Olivia = 16) (h₂ : Edward = 3) : Olivia + Edward = 19 :=
by
  sorry

end pieces_of_paper_picked_up_l169_169955


namespace sum_s_h_e_base_three_l169_169295

def distinct_non_zero_digits (S H E : ℕ) : Prop :=
  S ≠ 0 ∧ H ≠ 0 ∧ E ≠ 0 ∧ S < 3 ∧ H < 3 ∧ E < 3 ∧ S ≠ H ∧ H ≠ E ∧ S ≠ E

def base_three_addition (S H E : ℕ) :=
  (S + H * 3 + E * 9) + (H + E * 3) == (H * 3 + S * 9 + S*27)

theorem sum_s_h_e_base_three (S H E : ℕ) (h1 : distinct_non_zero_digits S H E) (h2 : base_three_addition S H E) :
  (S + H + E = 5) := by sorry

end sum_s_h_e_base_three_l169_169295


namespace same_quadratic_function_b_l169_169212

theorem same_quadratic_function_b (a c b : ℝ) :
    (∀ x : ℝ, a * (x - 2)^2 + c = (2 * x - 5) * (x - b)) → b = 3 / 2 :=
by
  sorry

end same_quadratic_function_b_l169_169212


namespace cone_height_l169_169583

theorem cone_height (r : ℝ) (θ : ℝ) (h : ℝ)
  (hr : r = 1)
  (hθ : θ = (2 / 3) * Real.pi)
  (h_eq : h = 2 * Real.sqrt 2) :
  ∃ l : ℝ, l = 3 ∧ h = Real.sqrt (l^2 - r^2) :=
by
  sorry

end cone_height_l169_169583


namespace range_of_m_l169_169105

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (x^2 - 4*|x| + 5 - m = 0) → (∃ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)) → (1 < m ∧ m < 5) :=
by
  sorry

end range_of_m_l169_169105


namespace required_force_18_inch_wrench_l169_169193

def inverse_force (l : ℕ) (k : ℕ) : ℕ := k / l

def extra_force : ℕ := 50

def initial_force : ℕ := 300

noncomputable
def handle_length_1 : ℕ := 12

noncomputable
def handle_length_2 : ℕ := 18

noncomputable
def adjusted_force : ℕ := inverse_force handle_length_2 (initial_force * handle_length_1)

theorem required_force_18_inch_wrench : 
  adjusted_force + extra_force = 250 := 
by
  sorry

end required_force_18_inch_wrench_l169_169193


namespace scenario1_winner_scenario2_winner_l169_169656

def optimal_play_winner1 (n : ℕ) (start_by_anna : Bool := true) : String :=
  if n % 6 = 0 then "Balázs"
  else "Anna"

def optimal_play_winner2 (n : ℕ) (start_by_anna : Bool := true) : String :=
  if n % 4 = 0 then "Balázs"
  else "Anna"

theorem scenario1_winner:
  optimal_play_winner1 39 true = "Balázs" :=
by 
  sorry

theorem scenario2_winner:
  optimal_play_winner2 39 true = "Anna" :=
by
  sorry

end scenario1_winner_scenario2_winner_l169_169656


namespace proof_problem_l169_169301

noncomputable def question (a b c d m : ℚ) : ℚ :=
  2 * a + 2 * b + (a + b - 3 * (c * d)) - m

def condition1 (m : ℚ) : Prop :=
  abs (m + 1) = 4

def condition2 (a b : ℚ) : Prop :=
  a = -b

def condition3 (c d : ℚ) : Prop :=
  c * d = 1

theorem proof_problem (a b c d m : ℚ) :
  condition1 m → condition2 a b → condition3 c d →
  (question a b c d m = 2 ∨ question a b c d m = -6) :=
by
  sorry

end proof_problem_l169_169301


namespace train_speed_in_kmh_l169_169545

def length_of_train : ℝ := 156
def length_of_bridge : ℝ := 219.03
def time_to_cross_bridge : ℝ := 30
def speed_of_train_kmh : ℝ := 45.0036

theorem train_speed_in_kmh :
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = speed_of_train_kmh :=
by
  sorry

end train_speed_in_kmh_l169_169545


namespace BigDigMiningCopperOutput_l169_169255

theorem BigDigMiningCopperOutput :
  (∀ (total_output : ℝ) (nickel_percentage : ℝ) (iron_percentage : ℝ) (amount_of_nickel : ℝ),
      nickel_percentage = 0.10 → 
      iron_percentage = 0.60 → 
      amount_of_nickel = 720 →
      total_output = amount_of_nickel / nickel_percentage →
      (1 - nickel_percentage - iron_percentage) * total_output = 2160) :=
sorry

end BigDigMiningCopperOutput_l169_169255


namespace vehicles_traveled_l169_169027

theorem vehicles_traveled (V : ℕ)
  (h1 : 40 * V = 800 * 100000000) : 
  V = 2000000000 := 
sorry

end vehicles_traveled_l169_169027


namespace problem_1_problem_2_l169_169668

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 / 6 + 1 / x - a * Real.log x

theorem problem_1 (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 3 → f x a ≤ f 3 a) → a ≥ 8 / 3 :=
sorry

theorem problem_2 (a : ℝ) (h1 : 0 < a) (x0 : ℝ) :
  (∃! t : ℝ, 0 < t ∧ f t a = 0) → Real.log x0 = (x0^3 + 6) / (2 * (x0^3 - 3)) :=
sorry

end problem_1_problem_2_l169_169668


namespace unique_solution_k_l169_169211

theorem unique_solution_k (k : ℝ) :
  (∀ x : ℝ, (x + 3) / (k * x + 2) = x) ↔ (k = -1 / 12) :=
  sorry

end unique_solution_k_l169_169211


namespace minimum_value_l169_169661

theorem minimum_value (x y z : ℝ) (h : x + y + z = 1) : 2 * x^2 + y^2 + 3 * z^2 ≥ 3 / 7 := by
  sorry

end minimum_value_l169_169661


namespace rectangles_with_one_gray_cell_l169_169618

/- Definitions from conditions -/
def total_gray_cells : ℕ := 40
def blue_cells : ℕ := 36
def red_cells : ℕ := 4

/- The number of rectangles containing exactly one gray cell is the proof goal -/
theorem rectangles_with_one_gray_cell :
  (blue_cells * 4 + red_cells * 8) = 176 :=
sorry

end rectangles_with_one_gray_cell_l169_169618


namespace sum_of_numbers_l169_169139

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 12) (h2 : (1 / x) = 3 * (1 / y)) :
  x + y = 8 :=
sorry

end sum_of_numbers_l169_169139


namespace correct_statement_is_B_l169_169922

def coefficient_of_x : Int := 1
def is_monomial (t : String) : Bool := t = "1x^0"
def coefficient_of_neg_3x : Int := -3
def degree_of_5x2y : Int := 3

theorem correct_statement_is_B :
  (coefficient_of_x = 0) = false ∧ 
  (is_monomial "1x^0" = true) ∧ 
  (coefficient_of_neg_3x = 3) = false ∧ 
  (degree_of_5x2y = 2) = false ∧ 
  (B = "1 is a monomial") :=
by {
  sorry
}

end correct_statement_is_B_l169_169922


namespace sphere_volume_l169_169570

theorem sphere_volume {r : ℝ} (h: 4 * Real.pi * r^2 = 256 * Real.pi) : (4 / 3) * Real.pi * r^3 = (2048 / 3) * Real.pi :=
by
  sorry

end sphere_volume_l169_169570


namespace hyperbola_asymptote_b_value_l169_169841

theorem hyperbola_asymptote_b_value (b : ℝ) (hb : 0 < b) : 
  (∀ x y, x^2 - y^2 / b^2 = 1 → y = 3 * x ∨ y = -3 * x) → b = 3 := 
by
  sorry

end hyperbola_asymptote_b_value_l169_169841


namespace find_k_l169_169274

theorem find_k (k : ℝ) (h1 : k > 1) 
(h2 : ∑' n : ℕ, (7 * (n + 1) - 3) / k^(n + 1) = 2) : 
  k = 2 + 3 * Real.sqrt 2 / 2 := 
sorry

end find_k_l169_169274


namespace shadow_length_when_eight_meters_away_l169_169265

noncomputable def lamp_post_height : ℝ := 8
noncomputable def sam_initial_distance : ℝ := 12
noncomputable def shadow_initial_length : ℝ := 4
noncomputable def sam_initial_height : ℝ := 2 -- derived from the problem's steps

theorem shadow_length_when_eight_meters_away :
  ∀ (L : ℝ), (L * lamp_post_height) / (lamp_post_height + sam_initial_distance - shadow_initial_length) = 2 → L = 8 / 3 :=
by
  intro L
  sorry

end shadow_length_when_eight_meters_away_l169_169265


namespace sakshi_work_days_l169_169382

theorem sakshi_work_days (x : ℝ) (efficiency_tanya : ℝ) (days_tanya : ℝ) 
  (h_efficiency : efficiency_tanya = 1.25) 
  (h_days : days_tanya = 4)
  (h_relationship : x / efficiency_tanya = days_tanya) : 
  x = 5 :=
by 
  -- Lean proof would go here
  sorry

end sakshi_work_days_l169_169382


namespace men_entered_l169_169528

theorem men_entered (M W x : ℕ) 
  (h1 : 5 * M = 4 * W)
  (h2 : M + x = 14)
  (h3 : 2 * (W - 3) = 24) : 
  x = 2 :=
by
  sorry

end men_entered_l169_169528


namespace solve_equation_l169_169187

theorem solve_equation 
  (x : ℚ)
  (h : (x^2 + 3*x + 4)/(x + 5) = x + 6) :
  x = -13/4 := 
by
  sorry

end solve_equation_l169_169187


namespace line_through_intersection_and_origin_l169_169495

-- Definitions of the lines
def l1 (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def l2 (x y : ℝ) : Prop := y = 1 - x

-- Prove that the line passing through the intersection of l1 and l2 and the origin has the equation 3x + 2y = 0
theorem line_through_intersection_and_origin (x y : ℝ) 
  (h1 : 2 * x - y + 7 = 0) (h2 : y = 1 - x) : 3 * x + 2 * y = 0 := 
sorry

end line_through_intersection_and_origin_l169_169495


namespace geese_population_1996_l169_169717

theorem geese_population_1996 (k x : ℝ) 
  (h1 : x - 39 = k * 60) 
  (h2 : 123 - 60 = k * x) : 
  x = 84 := 
by
  sorry

end geese_population_1996_l169_169717


namespace find_c_l169_169356

noncomputable def f (x c : ℝ) := x * (x - c) ^ 2
noncomputable def f' (x c : ℝ) := 3 * x ^ 2 - 4 * c * x + c ^ 2
noncomputable def f'' (x c : ℝ) := 6 * x - 4 * c

theorem find_c (c : ℝ) : f' 2 c = 0 ∧ f'' 2 c < 0 → c = 6 :=
by {
  sorry
}

end find_c_l169_169356


namespace min_cans_for_gallon_l169_169578

-- Define conditions
def can_capacity : ℕ := 12
def gallon_to_ounces : ℕ := 128

-- Define the minimum number of cans function.
def min_cans (capacity : ℕ) (required : ℕ) : ℕ :=
  (required + capacity - 1) / capacity -- This is the ceiling of required / capacity

-- Statement asserting the required minimum number of cans.
theorem min_cans_for_gallon (h : min_cans can_capacity gallon_to_ounces = 11) : 
  can_capacity > 0 ∧ gallon_to_ounces > 0 := by
  sorry

end min_cans_for_gallon_l169_169578


namespace value_of_expression_l169_169396

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

end value_of_expression_l169_169396


namespace last_three_digits_of_8_pow_104_l169_169577

def last_three_digits_of_pow (x n : ℕ) : ℕ :=
  (x ^ n) % 1000

theorem last_three_digits_of_8_pow_104 : last_three_digits_of_pow 8 104 = 984 := 
by
  sorry

end last_three_digits_of_8_pow_104_l169_169577


namespace intersection_eq_union_eq_l169_169740

def A := { x : ℝ | x ≥ 2 }
def B := { x : ℝ | 1 < x ∧ x ≤ 4 }

theorem intersection_eq : A ∩ B = { x : ℝ | 2 ≤ x ∧ x ≤ 4 } :=
by sorry

theorem union_eq : A ∪ B = { x : ℝ | 1 < x } :=
by sorry

end intersection_eq_union_eq_l169_169740


namespace largest_lcm_value_is_60_l169_169191

-- Define the conditions
def lcm_values : List ℕ := [Nat.lcm 15 3, Nat.lcm 15 5, Nat.lcm 15 9, Nat.lcm 15 12, Nat.lcm 15 10, Nat.lcm 15 15]

-- State the proof problem
theorem largest_lcm_value_is_60 : lcm_values.maximum = some 60 :=
by
  repeat { sorry }

end largest_lcm_value_is_60_l169_169191


namespace initial_roses_l169_169827

theorem initial_roses (R : ℕ) (h : R + 16 = 23) : R = 7 :=
sorry

end initial_roses_l169_169827


namespace route_speeds_l169_169428

theorem route_speeds (x : ℝ) (hx : x > 0) :
  (25 / x) - (21 / (1.4 * x)) = (20 / 60) := by
  sorry

end route_speeds_l169_169428


namespace a_7_is_127_l169_169266

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0       => 0  -- Define a_0 which is not used but useful for indexing
| 1       => 1
| (n + 2) => 2 * (a (n + 1)) + 1

-- Prove that a_7 = 127
theorem a_7_is_127 : a 7 = 127 := 
sorry

end a_7_is_127_l169_169266


namespace geometric_progression_identity_l169_169547

theorem geometric_progression_identity 
  (a b c d : ℝ) 
  (h1 : c^2 = b * d) 
  (h2 : b^2 = a * c) 
  (h3 : a * d = b * c) : 
  (a - c)^2 + (b - c)^2 + (b - d)^2 = (a - d)^2 :=
by 
  sorry

end geometric_progression_identity_l169_169547


namespace largest_vs_smallest_circles_l169_169839

variable (M : Type) [MetricSpace M] [MeasurableSpace M]

def non_overlapping_circles (M : Type) [MetricSpace M] [MeasurableSpace M] : ℕ := sorry

def covering_circles (M : Type) [MetricSpace M] [MeasurableSpace M] : ℕ := sorry

theorem largest_vs_smallest_circles (M : Type) [MetricSpace M] [MeasurableSpace M] :
  non_overlapping_circles M ≥ covering_circles M :=
sorry

end largest_vs_smallest_circles_l169_169839


namespace problem_statement_l169_169615

noncomputable def f (n : ℕ) : ℝ := Real.log (n^2) / Real.log 3003

theorem problem_statement : f 33 + f 13 + f 7 = 2 := 
by
  sorry

end problem_statement_l169_169615


namespace profit_calculation_l169_169632

-- Define conditions based on investments
def JohnInvestment := 700
def MikeInvestment := 300

-- Define the equality condition where John received $800 more than Mike
theorem profit_calculation (P : ℝ) 
  (h1 : (P / 6 + (7 / 10) * (2 * P / 3)) - (P / 6 + (3 / 10) * (2 * P / 3)) = 800) : 
  P = 3000 := 
sorry

end profit_calculation_l169_169632


namespace Kendall_dimes_l169_169675

theorem Kendall_dimes (total_value : ℝ) (quarters : ℝ) (dimes : ℝ) (nickels : ℝ) 
  (num_quarters : ℕ) (num_nickels : ℕ) 
  (total_amount : total_value = 4)
  (quarter_amount : quarters = num_quarters * 0.25)
  (num_quarters_val : num_quarters = 10)
  (nickel_amount : nickels = num_nickels * 0.05) 
  (num_nickels_val : num_nickels = 6) :
  dimes = 12 := by
  sorry

end Kendall_dimes_l169_169675


namespace arithmetic_mean_of_distribution_l169_169435

-- Defining conditions
def stddev : ℝ := 2.3
def value : ℝ := 11.6

-- Proving the mean (μ) is 16.2
theorem arithmetic_mean_of_distribution : ∃ μ : ℝ, μ = 16.2 ∧ value = μ - 2 * stddev :=
by
  use 16.2
  sorry

end arithmetic_mean_of_distribution_l169_169435


namespace gcf_of_lcm_9_15_and_10_21_is_5_l169_169182

theorem gcf_of_lcm_9_15_and_10_21_is_5
  (h9 : 9 = 3 ^ 2)
  (h15 : 15 = 3 * 5)
  (h10 : 10 = 2 * 5)
  (h21 : 21 = 3 * 7) :
  Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 5 := by
  sorry

end gcf_of_lcm_9_15_and_10_21_is_5_l169_169182


namespace value_of_c_l169_169003

theorem value_of_c :
  ∃ (a b c : ℕ), 
  30 = 2 * (10 + a) ∧ 
  b = 2 * (a + 30) ∧ 
  c = 2 * (b + 30) ∧ 
  c = 200 := 
sorry

end value_of_c_l169_169003


namespace find_second_sum_l169_169786

def sum : ℕ := 2717
def interest_rate_first : ℚ := 3 / 100
def interest_rate_second : ℚ := 5 / 100
def time_first : ℕ := 8
def time_second : ℕ := 3

theorem find_second_sum (x : ℚ) (h : x * interest_rate_first * time_first = (sum - x) * interest_rate_second * time_second) : 
  sum - x = 2449 :=
by
  sorry

end find_second_sum_l169_169786


namespace algebraic_expression_value_l169_169292

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3 * x - 3 = 0) : x^3 + 2 * x^2 - 6 * x + 3 = 0 := 
sorry

end algebraic_expression_value_l169_169292


namespace maria_initial_cookies_l169_169005

theorem maria_initial_cookies (X : ℕ) 
  (h1: X - 5 = 2 * (5 + 2)) 
  (h2: X ≥ 5)
  : X = 19 := 
by
  sorry

end maria_initial_cookies_l169_169005


namespace sum_first_9_terms_l169_169828

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable (a_1 a_2 a_3 a_4 a_5 a_6 : ℝ)

-- Conditions
axiom h1 : a 1 + a 5 = 10
axiom h2 : a 2 + a 6 = 14

-- Calculations
axiom h3 : a 3 = 5
axiom h4 : a 4 = 7
axiom h5 : d = 2
axiom h6 : a 5 = 9

-- The sum of the first 9 terms
axiom h7 : S 9 = 9 * a 5

theorem sum_first_9_terms : S 9 = 81 :=
by {
  sorry
}

end sum_first_9_terms_l169_169828


namespace tire_circumference_is_one_meter_l169_169234

-- Definitions for the given conditions
def car_speed : ℕ := 24 -- in km/h
def tire_rotations_per_minute : ℕ := 400

-- Conversion factors
def km_to_m : ℕ := 1000
def hour_to_min : ℕ := 60

-- The equivalent proof problem
theorem tire_circumference_is_one_meter 
  (hs : car_speed * km_to_m / hour_to_min = 400 * tire_rotations_per_minute)
  : 400 = 400 * 1 := 
by
  sorry

end tire_circumference_is_one_meter_l169_169234


namespace math_problem_l169_169987

def cond1 (R r a b c p : ℝ) : Prop := R * r = (a * b * c) / (4 * p)
def cond2 (a b c p : ℝ) : Prop := a * b * c ≤ 8 * p^3
def cond3 (a b c p : ℝ) : Prop := p^2 ≤ (3 * (a^2 + b^2 + c^2)) / 4
def cond4 (m_a m_b m_c R : ℝ) : Prop := m_a^2 + m_b^2 + m_c^2 ≤ (27 * R^2) / 4

theorem math_problem (R r a b c p m_a m_b m_c : ℝ) 
  (h1 : cond1 R r a b c p)
  (h2 : cond2 a b c p)
  (h3 : cond3 a b c p)
  (h4 : cond4 m_a m_b m_c R) : 
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ (27 * R^2) / 2 :=
by 
  sorry

end math_problem_l169_169987


namespace clothes_donation_l169_169742

variable (initial_clothes : ℕ)
variable (clothes_thrown_away : ℕ)
variable (final_clothes : ℕ)
variable (x : ℕ)

theorem clothes_donation (h1 : initial_clothes = 100) 
                        (h2 : clothes_thrown_away = 15) 
                        (h3 : final_clothes = 65) 
                        (h4 : 4 * x = initial_clothes - final_clothes - clothes_thrown_away) :
  x = 5 := by
  sorry

end clothes_donation_l169_169742


namespace decrease_in_demand_l169_169471

theorem decrease_in_demand (init_price new_price demand : ℝ) (init_demand : ℕ) (price_increase : ℝ) (original_revenue new_demand : ℝ) :
  init_price = 20 ∧ init_demand = 500 ∧ price_increase = 5 ∧ demand = init_price + price_increase ∧ 
  original_revenue = init_price * init_demand ∧ new_demand ≤ init_demand ∧ 
  new_demand * demand ≥ original_revenue → 
  init_demand - new_demand = 100 :=
by 
  sorry

end decrease_in_demand_l169_169471


namespace arithmetic_sequence_a6_eq_1_l169_169161

theorem arithmetic_sequence_a6_eq_1
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : S 11 = 11)
  (h2 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h3 : ∃ d, ∀ n, a n = a 1 + (n - 1) * d) :
  a 6 = 1 :=
by
  sorry

end arithmetic_sequence_a6_eq_1_l169_169161


namespace complement_intersection_l169_169744

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection (hU : U = {2, 3, 6, 8}) (hA : A = {2, 3}) (hB : B = {2, 6, 8}) :
  ((U \ A) ∩ B) = {6, 8} := 
by
  sorry

end complement_intersection_l169_169744


namespace inequality_proof_l169_169771

theorem inequality_proof
  (x y z : ℝ)
  (h_x : x ≥ 0)
  (h_y : y ≥ 0)
  (h_z : z > 0)
  (h_xy : x ≥ y)
  (h_yz : y ≥ z) :
  (x + y + z) * (x + y - z) * (x - y + z) / (x * y * z) ≥ 3 := by
  sorry

end inequality_proof_l169_169771


namespace min_value_of_z_l169_169485

-- Define the conditions as separate hypotheses.
variable (x y : ℝ)

def condition1 : Prop := x - y + 1 ≥ 0
def condition2 : Prop := x + y - 1 ≥ 0
def condition3 : Prop := x ≤ 3

-- Define the objective function.
def z : ℝ := 2 * x - 3 * y

-- State the theorem to prove the minimum value of z given the conditions.
theorem min_value_of_z (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x) :
  ∃ x y, condition1 x y ∧ condition2 x y ∧ condition3 x ∧ z x y = -6 :=
sorry

end min_value_of_z_l169_169485


namespace intersection_point_correct_l169_169851

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Line :=
(p1 : Point3D) (p2 : Point3D)

structure Plane :=
(trace : Line) (point : Point3D)

noncomputable def intersection_point (l : Line) (β : Plane) : Point3D := sorry

theorem intersection_point_correct (l : Line) (β : Plane) (P : Point3D) :
  let res := intersection_point l β
  res = P :=
sorry

end intersection_point_correct_l169_169851


namespace find_a_minus_b_l169_169605

theorem find_a_minus_b (a b : ℝ) :
  (∀ (x : ℝ), x^4 - 8 * x^3 + a * x^2 + b * x + 16 = 0 → x > 0) →
  a - b = 56 :=
by
  sorry

end find_a_minus_b_l169_169605


namespace intersection_of_sets_l169_169276

theorem intersection_of_sets :
  let M := { x : ℝ | -3 < x ∧ x ≤ 5 }
  let N := { x : ℝ | -5 < x ∧ x < 5 }
  M ∩ N = { x : ℝ | -3 < x ∧ x < 5 } := 
by
  sorry

end intersection_of_sets_l169_169276


namespace equilateral_triangle_properties_l169_169819

noncomputable def equilateral_triangle_perimeter (a : ℝ) : ℝ :=
3 * a

noncomputable def equilateral_triangle_bisector_length (a : ℝ) : ℝ :=
(a * Real.sqrt 3) / 2

theorem equilateral_triangle_properties (a : ℝ) (h : a = 10) :
  equilateral_triangle_perimeter a = 30 ∧
  equilateral_triangle_bisector_length a = 5 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_properties_l169_169819


namespace marcy_total_spears_l169_169427

-- Define the conditions
def can_make_spears_from_sapling (spears_per_sapling : ℕ) (saplings : ℕ) : ℕ :=
  spears_per_sapling * saplings

def can_make_spears_from_log (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  spears_per_log * logs

-- Number of spears Marcy can make from 6 saplings and 1 log
def total_spears (spears_per_sapling : ℕ) (saplings : ℕ) (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  can_make_spears_from_sapling spears_per_sapling saplings + can_make_spears_from_log spears_per_log logs

-- Given conditions
theorem marcy_total_spears (saplings : ℕ) (logs : ℕ) : 
  total_spears 3 6 9 1 = 27 :=
by
  sorry

end marcy_total_spears_l169_169427


namespace amanda_average_speed_l169_169401

def amanda_distance1 : ℝ := 450
def amanda_time1 : ℝ := 7.5
def amanda_distance2 : ℝ := 420
def amanda_time2 : ℝ := 7

def total_distance : ℝ := amanda_distance1 + amanda_distance2
def total_time : ℝ := amanda_time1 + amanda_time2
def expected_average_speed : ℝ := 60

theorem amanda_average_speed :
  (total_distance / total_time) = expected_average_speed := by
  sorry

end amanda_average_speed_l169_169401


namespace complex_multiplication_l169_169406

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : (2 + i) * (1 - 3 * i) = 5 - 5 * i := 
by
  sorry

end complex_multiplication_l169_169406


namespace speed_ratio_A_to_B_l169_169362

variables {u v : ℝ}

axiom perp_lines_intersect_at_o : true
axiom points_move_along_lines_at_constant_speed : true
axiom point_A_at_O_B_500_yards_away_at_t_0 : true
axiom after_2_minutes_A_and_B_equidistant : 2 * u = 500 - 2 * v
axiom after_10_minutes_A_and_B_equidistant : 10 * u = 10 * v - 500

theorem speed_ratio_A_to_B : u / v = 2 / 3 :=
by 
  sorry

end speed_ratio_A_to_B_l169_169362


namespace coffee_shop_lattes_l169_169912

theorem coffee_shop_lattes (T : ℕ) (L : ℕ) (hT : T = 6) (hL : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_lattes_l169_169912


namespace opposite_of_2023_l169_169969

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l169_169969


namespace triangle_side_lengths_l169_169562

-- Define the variables a, b, and c
variables {a b c : ℝ}

-- Assume that a, b, and c are the lengths of the sides of a triangle
-- and the given equation holds
theorem triangle_side_lengths (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) 
    (h_eq : a^2 + 4*a*c + 3*c^2 - 3*a*b - 7*b*c + 2*b^2 = 0) : 
    a + c - 2*b = 0 :=
by
  sorry

end triangle_side_lengths_l169_169562


namespace Nancy_picked_l169_169707

def Alyssa_picked : ℕ := 42
def Total_picked : ℕ := 59

theorem Nancy_picked : Total_picked - Alyssa_picked = 17 := by
  sorry

end Nancy_picked_l169_169707


namespace value_multiplied_by_l169_169154

theorem value_multiplied_by (x : ℝ) (h : (7.5 / 6) * x = 15) : x = 12 :=
by
  sorry

end value_multiplied_by_l169_169154


namespace problem_I4_1_l169_169232

theorem problem_I4_1 
  (x y : ℝ)
  (h : (10 * x - 3 * y) / (x + 2 * y) = 2) :
  (y + x) / (y - x) = 15 :=
sorry

end problem_I4_1_l169_169232


namespace remainder_when_divided_by_198_l169_169530

-- Define the conditions as Hypotheses
variables (x : ℤ)

-- Hypotheses stating the given conditions
def cond1 : Prop := 2 + x ≡ 9 [ZMOD 8]
def cond2 : Prop := 3 + x ≡ 4 [ZMOD 27]
def cond3 : Prop := 11 + x ≡ 49 [ZMOD 1331]

-- Final statement to prove
theorem remainder_when_divided_by_198 (h1 : cond1 x) (h2 : cond2 x) (h3 : cond3 x) : x ≡ 1 [ZMOD 198] := by
  sorry

end remainder_when_divided_by_198_l169_169530


namespace roden_total_fish_l169_169230

def total_goldfish : Nat :=
  15 + 10 + 3 + 4

def total_blue_fish : Nat :=
  7 + 12 + 7 + 8

def total_green_fish : Nat :=
  5 + 9 + 6

def total_purple_fish : Nat :=
  2

def total_red_fish : Nat :=
  1

def total_fish : Nat :=
  total_goldfish + total_blue_fish + total_green_fish + total_purple_fish + total_red_fish

theorem roden_total_fish : total_fish = 89 :=
by
  unfold total_fish total_goldfish total_blue_fish total_green_fish total_purple_fish total_red_fish
  sorry

end roden_total_fish_l169_169230


namespace hike_up_days_l169_169173

theorem hike_up_days (R_up R_down D_down D_up : ℝ) 
  (H1 : R_up = 8) 
  (H2 : R_down = 1.5 * R_up)
  (H3 : D_down = 24)
  (H4 : D_up / R_up = D_down / R_down) : 
  D_up / R_up = 2 :=
by
  sorry

end hike_up_days_l169_169173


namespace total_cost_of_water_l169_169133

-- Define conditions in Lean 4
def cost_per_liter : ℕ := 1
def liters_per_bottle : ℕ := 2
def number_of_bottles : ℕ := 6

-- Define the theorem to prove the total cost
theorem total_cost_of_water : (number_of_bottles * (liters_per_bottle * cost_per_liter)) = 12 :=
by
  sorry

end total_cost_of_water_l169_169133


namespace number_of_houses_on_block_l169_169304

theorem number_of_houses_on_block 
  (total_mail : ℕ) 
  (white_mailboxes : ℕ) 
  (red_mailboxes : ℕ) 
  (mail_per_house : ℕ) 
  (total_white_mail : ℕ) 
  (total_red_mail : ℕ) 
  (remaining_mail : ℕ)
  (additional_houses : ℕ)
  (total_houses : ℕ) :
  total_mail = 48 ∧ 
  white_mailboxes = 2 ∧ 
  red_mailboxes = 3 ∧ 
  mail_per_house = 6 ∧ 
  total_white_mail = white_mailboxes * mail_per_house ∧
  total_red_mail = red_mailboxes * mail_per_house ∧
  remaining_mail = total_mail - (total_white_mail + total_red_mail) ∧
  additional_houses = remaining_mail / mail_per_house ∧
  total_houses = white_mailboxes + red_mailboxes + additional_houses →
  total_houses = 8 :=
by 
  sorry

end number_of_houses_on_block_l169_169304


namespace turnip_bag_weighs_l169_169885

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l169_169885


namespace parabola_slope_l169_169601

theorem parabola_slope (p k : ℝ) (h1 : p > 0)
  (h_focus_distance : (p / 2) * (3^(1/2)) / (3 + 1^(1/2))^(1/2) = 3^(1/2))
  (h_AF_FB : exists A B : ℝ × ℝ, (A.1 = 2 - p / 2 ∧ 2 * (B.1 - 2) = 2)
    ∧ (A.2 = p - p / 2 ∧ A.2 = -2 * B.2)) :
  abs k = 2 * (2^(1/2)) :=
sorry

end parabola_slope_l169_169601


namespace Lyka_saves_for_8_weeks_l169_169511

theorem Lyka_saves_for_8_weeks : 
  ∀ (C I W : ℕ), C = 160 → I = 40 → W = 15 → (C - I) / W = 8 := 
by 
  intros C I W hC hI hW
  sorry

end Lyka_saves_for_8_weeks_l169_169511


namespace region_area_l169_169905

noncomputable def area_of_region := 4 * Real.pi

theorem region_area :
  (∃ x y, x^2 + y^2 - 4 * x + 2 * y + 1 = 0) →
  Real.pi * 4 = area_of_region :=
by
  sorry

end region_area_l169_169905


namespace discount_percentage_l169_169602

theorem discount_percentage (SP CP SP' discount_gain_percentage: ℝ) 
  (h1 : SP = 30) 
  (h2 : SP = CP + 0.25 * CP) 
  (h3 : SP' = CP + 0.125 * CP) 
  (h4 : discount_gain_percentage = ((SP - SP') / SP) * 100) :
  discount_gain_percentage = 10 :=
by
  -- Skipping the proof
  sorry

end discount_percentage_l169_169602


namespace proof_problem_l169_169487

-- Define the system of equations
def system_of_equations (x y a : ℝ) : Prop :=
  (3 * x + y = 2 + 3 * a) ∧ (x + 3 * y = 2 + a)

-- Define the condition x + y < 0
def condition (x y : ℝ) : Prop := x + y < 0

-- Prove that if the system of equations has a solution with x + y < 0, then a < -1 and |1 - a| + |a + 1 / 2| = 1 / 2 - 2 * a
theorem proof_problem (x y a : ℝ) (h1 : system_of_equations x y a) (h2 : condition x y) :
  a < -1 ∧ |1 - a| + |a + 1 / 2| = (1 / 2) - 2 * a := 
sorry

end proof_problem_l169_169487


namespace sum_of_solutions_l169_169175

theorem sum_of_solutions (x1 x2 : ℝ) (h : ∀ (x : ℝ), x^2 - 10 * x + 14 = 0 → x = x1 ∨ x = x2) :
  x1 + x2 = 10 :=
sorry

end sum_of_solutions_l169_169175


namespace a_perpendicular_to_a_minus_b_l169_169970

def vector := ℝ × ℝ

def dot_product (u v : vector) : ℝ := u.1 * v.1 + u.2 * v.2

def a : vector := (-2, 1)
def b : vector := (-1, 3)

def a_minus_b : vector := (a.1 - b.1, a.2 - b.2) 

theorem a_perpendicular_to_a_minus_b : dot_product a a_minus_b = 0 := by
  sorry

end a_perpendicular_to_a_minus_b_l169_169970


namespace trajectory_of_center_of_moving_circle_l169_169385

theorem trajectory_of_center_of_moving_circle
  (x y : ℝ)
  (C1 : (x + 4)^2 + y^2 = 2)
  (C2 : (x - 4)^2 + y^2 = 2) :
  ((x = 0) ∨ (x^2 / 2 - y^2 / 14 = 1)) :=
sorry

end trajectory_of_center_of_moving_circle_l169_169385


namespace simplify_expr_l169_169637

-- Define the condition on b
def condition (b : ℚ) : Prop :=
  b ≠ -1 / 2

-- Define the expression to be evaluated
def expression (b : ℚ) : ℚ :=
  1 - 1 / (1 + b / (1 + b))

-- Define the simplified form
def simplified_expr (b : ℚ) : ℚ :=
  b / (1 + 2 * b)

-- The theorem statement showing the equivalence
theorem simplify_expr (b : ℚ) (h : condition b) : expression b = simplified_expr b :=
by
  sorry

end simplify_expr_l169_169637


namespace min_hypotenuse_l169_169617

theorem min_hypotenuse {a b : ℝ} (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 10) :
  ∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ c ≥ 5 * Real.sqrt 2 :=
by
  sorry

end min_hypotenuse_l169_169617


namespace solution_set_inequality_l169_169067

noncomputable def solution_set (x : ℝ) : Prop :=
  (2 * x - 1) / (x + 2) > 1

theorem solution_set_inequality :
  { x : ℝ | solution_set x } = { x : ℝ | x < -2 ∨ x > 3 } := by
  sorry

end solution_set_inequality_l169_169067


namespace find_f_of_2_l169_169286

variable (f : ℝ → ℝ)

-- Given condition: f is the inverse function of the exponential function 2^x
def inv_function : Prop := ∀ x, f (2^x) = x ∧ 2^(f x) = x

theorem find_f_of_2 (h : inv_function f) : f 2 = 1 :=
by sorry

end find_f_of_2_l169_169286


namespace correct_proposition_l169_169434

theorem correct_proposition (a b : ℝ) (h : |a| < b) : a^2 < b^2 :=
sorry

end correct_proposition_l169_169434


namespace number_of_lines_through_point_intersect_hyperbola_once_l169_169316

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 = 1

noncomputable def point_P : ℝ × ℝ :=
  (-4, 1)

noncomputable def line_through (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  l P

noncomputable def one_point_intersection (l : ℝ × ℝ → Prop) (H : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, l p ∧ H p.1 p.2

theorem number_of_lines_through_point_intersect_hyperbola_once :
  (∃ (l₁ l₂ : ℝ × ℝ → Prop),
    line_through point_P l₁ ∧
    line_through point_P l₂ ∧
    one_point_intersection l₁ hyperbola ∧
    one_point_intersection l₂ hyperbola ∧
    l₁ ≠ l₂) ∧ ¬ (∃ (l₃ : ℝ × ℝ → Prop),
    line_through point_P l₃ ∧
    one_point_intersection l₃ hyperbola ∧
    ∃! (other_line : ℝ × ℝ → Prop),
    line_through point_P other_line ∧
    one_point_intersection other_line hyperbola ∧
    l₃ ≠ other_line) :=
sorry

end number_of_lines_through_point_intersect_hyperbola_once_l169_169316


namespace n_is_square_l169_169224

theorem n_is_square (n m : ℕ) (h1 : 3 ≤ n) (h2 : m = (n * (n - 1)) / 2) (h3 : ∃ (cards : Finset ℕ), 
  (cards.card = n) ∧ (∀ i ∈ cards, i ∈ Finset.range (m + 1)) ∧ 
  (∀ (i j : ℕ) (hi : i ∈ cards) (hj : j ∈ cards), i ≠ j → 
    ((i + j) % m) ≠ ((i + j) % m))) : 
  ∃ k : ℕ, n = k * k := 
sorry

end n_is_square_l169_169224


namespace distinct_book_arrangements_l169_169426

def num_books := 7
def num_identical_books := 3
def num_unique_books := num_books - num_identical_books

theorem distinct_book_arrangements :
  (Nat.factorial num_books) / (Nat.factorial num_identical_books) = 840 := 
  by 
  sorry

end distinct_book_arrangements_l169_169426


namespace recurring_decimal_product_l169_169258

theorem recurring_decimal_product : (0.3333333333 : ℝ) * (0.4545454545 : ℝ) = (5 / 33 : ℝ) :=
sorry

end recurring_decimal_product_l169_169258


namespace cylinder_volume_expansion_l169_169113

theorem cylinder_volume_expansion (r h : ℝ) :
  (π * (2 * r)^2 * h) = 4 * (π * r^2 * h) :=
by
  sorry

end cylinder_volume_expansion_l169_169113


namespace cuboid_surface_area_l169_169674

/--
Given a cuboid with length 10 cm, breadth 8 cm, and height 6 cm, the surface area is 376 cm².
-/
theorem cuboid_surface_area 
  (length : ℝ) 
  (breadth : ℝ) 
  (height : ℝ) 
  (h_length : length = 10) 
  (h_breadth : breadth = 8) 
  (h_height : height = 6) : 
  2 * (length * height + length * breadth + breadth * height) = 376 := 
by 
  -- Replace these placeholders with the actual proof steps.
  sorry

end cuboid_surface_area_l169_169674


namespace problem_sum_congruent_mod_11_l169_169084

theorem problem_sum_congruent_mod_11 : 
  (2 + 333 + 5555 + 77777 + 999999 + 11111111 + 222222222) % 11 = 3 := 
by
  -- Proof needed here
  sorry

end problem_sum_congruent_mod_11_l169_169084


namespace rolls_combinations_l169_169366

theorem rolls_combinations {n k : ℕ} (h_n : n = 4) (h_k : k = 5) :
  (Nat.choose (n + k - 1) k) = 56 :=
by
  rw [h_n, h_k]
  norm_num
  sorry

end rolls_combinations_l169_169366


namespace roots_product_of_polynomials_l169_169526

theorem roots_product_of_polynomials :
  ∃ (b c : ℤ), (∀ r : ℂ, r ^ 2 - 2 * r - 1 = 0 → r ^ 5 - b * r - c = 0) ∧ b * c = 348 :=
by 
  sorry

end roots_product_of_polynomials_l169_169526


namespace supplement_of_angle_l169_169830

theorem supplement_of_angle (complement_of_angle : ℝ) (h1 : complement_of_angle = 30) :
  ∃ (angle supplement_angle : ℝ), angle + complement_of_angle = 90 ∧ angle + supplement_angle = 180 ∧ supplement_angle = 120 :=
by
  sorry

end supplement_of_angle_l169_169830


namespace total_angles_sum_l169_169708

variables (A B C D E : Type)
variables (angle1 angle2 angle3 angle4 angle5 angle7 : ℝ)

-- Conditions about the geometry
axiom angle_triangle_ABC : angle1 + angle2 + angle3 = 180
axiom angle_triangle_BDE : angle7 + angle4 + angle5 = 180
axiom shared_angle_B : angle2 + angle7 = 180 -- since they form a straight line at vertex B

-- Proof statement
theorem total_angles_sum (A B C D E : Type) (angle1 angle2 angle3 angle4 angle5 angle7 : ℝ) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle7 - 180 = 180 :=
by
  sorry

end total_angles_sum_l169_169708


namespace find_f_729_l169_169024

variable (f : ℕ+ → ℕ+) -- Define the function f on the positive integers.

-- Conditions of the problem.
axiom h1 : ∀ n : ℕ+, f (f n) = 3 * n
axiom h2 : ∀ n : ℕ+, f (3 * n + 1) = 3 * n + 2 

-- Proof statement.
theorem find_f_729 : f 729 = 729 :=
by
  sorry -- Placeholder for the proof.

end find_f_729_l169_169024


namespace inequality_proof_l169_169981

variable (a b : Real)
variable (θ : Real)

-- Line equation and point condition
def line_eq := ∀ x y, x / a + y / b = 1 → (x, y) = (Real.cos θ, Real.sin θ)
-- Main theorem to prove
theorem inequality_proof : (line_eq a b θ) → 1 / (a^2) + 1 / (b^2) ≥ 1 := sorry

end inequality_proof_l169_169981


namespace custom_mul_expansion_l169_169927

variable {a b x y : ℝ}

def custom_mul (a b : ℝ) : ℝ := (a - b)^2

theorem custom_mul_expansion (x y : ℝ) : custom_mul (x^2) (y^2) = (x + y)^2 * (x - y)^2 := by
  sorry

end custom_mul_expansion_l169_169927


namespace students_with_one_problem_l169_169393

theorem students_with_one_problem :
  ∃ (n_1 n_2 n_3 n_4 n_5 n_6 n_7 : ℕ) (k_1 k_2 k_3 k_4 k_5 k_6 k_7 : ℕ),
    (n_1 + n_2 + n_3 + n_4 + n_5 + n_6 + n_7 = 39) ∧
    (n_1 * k_1 + n_2 * k_2 + n_3 * k_3 + n_4 * k_4 + n_5 * k_5 + n_6 * k_6 + n_7 * k_7 = 60) ∧
    (k_1 ≠ 0) ∧ (k_2 ≠ 0) ∧ (k_3 ≠ 0) ∧ (k_4 ≠ 0) ∧ (k_5 ≠ 0) ∧ (k_6 ≠ 0) ∧ (k_7 ≠ 0) ∧
    (k_1 ≠ k_2) ∧ (k_1 ≠ k_3) ∧ (k_1 ≠ k_4) ∧ (k_1 ≠ k_5) ∧ (k_1 ≠ k_6) ∧ (k_1 ≠ k_7) ∧
    (k_2 ≠ k_3) ∧ (k_2 ≠ k_4) ∧ (k_2 ≠ k_5) ∧ (k_2 ≠ k_6) ∧ (k_2 ≠ k_7) ∧
    (k_3 ≠ k_4) ∧ (k_3 ≠ k_5) ∧ (k_3 ≠ k_6) ∧ (k_3 ≠ k_7) ∧
    (k_4 ≠ k_5) ∧ (k_4 ≠ k_6) ∧ (k_4 ≠ k_7) ∧
    (k_5 ≠ k_6) ∧ (k_5 ≠ k_7) ∧
    (k_6 ≠ k_7) ∧
    (n_1 = 33) :=
sorry

end students_with_one_problem_l169_169393


namespace range_of_m_for_false_p_and_q_l169_169974

theorem range_of_m_for_false_p_and_q (m : ℝ) :
  (¬ (∀ x y : ℝ, (x^2 / (1 - m) + y^2 / (m + 2) = 1) ∧ ∀ x y : ℝ, (x^2 / (2 * m) + y^2 / (2 - m) = 1))) →
  (m ≤ 1 ∨ m ≥ 2) :=
sorry

end range_of_m_for_false_p_and_q_l169_169974


namespace problem_statement_l169_169847

def number_of_combinations (n k : ℕ) : ℕ := Nat.choose n k

def successful_outcomes : ℕ :=
  (number_of_combinations 3 1) * (number_of_combinations 5 1) * (number_of_combinations 4 5) +
  (number_of_combinations 3 2) * (number_of_combinations 4 5)

def total_outcomes : ℕ := number_of_combinations 12 7

def probability_at_least_75_cents : ℚ :=
  successful_outcomes / total_outcomes

theorem problem_statement : probability_at_least_75_cents = 3 / 22 := by
  sorry

end problem_statement_l169_169847


namespace number_of_complete_decks_l169_169449

theorem number_of_complete_decks (total_cards : ℕ) (additional_cards : ℕ) (cards_per_deck : ℕ) 
(h1 : total_cards = 319) (h2 : additional_cards = 7) (h3 : cards_per_deck = 52) : 
total_cards - additional_cards = (cards_per_deck * 6) :=
by
  sorry

end number_of_complete_decks_l169_169449


namespace Margo_paired_with_Irma_probability_l169_169805

noncomputable def probability_Margo_paired_with_Irma : ℚ :=
  1 / 29

theorem Margo_paired_with_Irma_probability :
  let total_students := 30
  let number_of_pairings := total_students - 1
  probability_Margo_paired_with_Irma = 1 / number_of_pairings := 
by
  sorry

end Margo_paired_with_Irma_probability_l169_169805


namespace area_of_circle_l169_169077

def circle_area (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y = 1

theorem area_of_circle : ∃ (area : ℝ), area = 6 * Real.pi :=
by sorry

end area_of_circle_l169_169077


namespace white_marbles_in_C_equals_15_l169_169999

variables (A_red A_yellow B_green B_yellow C_yellow : ℕ) (w : ℕ)

-- Conditions from the problem
def conditions : Prop :=
  A_red = 4 ∧ A_yellow = 2 ∧
  B_green = 6 ∧ B_yellow = 1 ∧
  C_yellow = 9 ∧
  (A_red - A_yellow = 2) ∧
  (B_green - B_yellow = 5) ∧
  (w - C_yellow = 6)

-- Proving w = 15 given the conditions
theorem white_marbles_in_C_equals_15 (h : conditions A_red A_yellow B_green B_yellow C_yellow w) : w = 15 :=
  sorry

end white_marbles_in_C_equals_15_l169_169999


namespace lunch_break_duration_l169_169701

/-- Define the total recess time as a sum of two 15-minute breaks and one 20-minute break. -/
def total_recess_time : ℕ := 15 + 15 + 20

/-- Define the total time spent outside of class. -/
def total_outside_class_time : ℕ := 80

/-- Prove that the lunch break is 30 minutes long. -/
theorem lunch_break_duration : total_outside_class_time - total_recess_time = 30 :=
by
  sorry

end lunch_break_duration_l169_169701


namespace find_A_l169_169474

theorem find_A (A B : ℝ) (h1 : B = 10 * A) (h2 : 211.5 = B - A) : A = 23.5 :=
by {
  sorry
}

end find_A_l169_169474


namespace compute_expression_l169_169197

theorem compute_expression (x : ℤ) (h : x = 3) : (x^8 + 24 * x^4 + 144) / (x^4 + 12) = 93 :=
by
  rw [h]
  sorry

end compute_expression_l169_169197


namespace education_expenses_l169_169888

noncomputable def totalSalary (savings : ℝ) (savingsPercentage : ℝ) : ℝ :=
  savings / savingsPercentage

def totalExpenses (rent milk groceries petrol misc : ℝ) : ℝ :=
  rent + milk + groceries + petrol + misc

def amountSpentOnEducation (totalSalary totalExpenses savings : ℝ) : ℝ :=
  totalSalary - (totalExpenses + savings)

theorem education_expenses :
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let petrol := 2000
  let misc := 700
  let savings := 1800
  let savingsPercentage := 0.10
  amountSpentOnEducation (totalSalary savings savingsPercentage) 
                          (totalExpenses rent milk groceries petrol misc) 
                          savings = 2500 :=
by
  sorry

end education_expenses_l169_169888


namespace village_population_rate_l169_169760

theorem village_population_rate (R : ℕ) :
  (76000 - 17 * R = 42000 + 17 * 800) → R = 1200 :=
by
  intro h
  -- The actual proof is omitted.
  sorry

end village_population_rate_l169_169760


namespace solve_chestnut_problem_l169_169829

def chestnut_problem : Prop :=
  ∃ (P M L : ℕ), (M = 2 * P) ∧ (L = P + 2) ∧ (P + M + L = 26) ∧ (M = 12)

theorem solve_chestnut_problem : chestnut_problem :=
by 
  sorry

end solve_chestnut_problem_l169_169829


namespace max_value_of_sequence_l169_169069

theorem max_value_of_sequence :
  ∃ a : ℕ → ℕ, (∀ i, 1 ≤ i ∧ i ≤ 101 → 0 < a i) →
              (∀ i, 1 ≤ i ∧ i < 101 → (a i + 1) % a (i + 1) = 0) →
              (a 102 = a 1) →
              (∀ n, (1 ≤ n ∧ n ≤ 101) → a n ≤ 201) :=
by
  sorry

end max_value_of_sequence_l169_169069


namespace find_pairs_solution_l169_169267

theorem find_pairs_solution (x y : ℝ) :
  (x^3 + x^2 * y + x * y^2 + y^3 = 8 * (x^2 + x * y + y^2 + 1)) ↔ 
  (x, y) = (8, -2) ∨ (x, y) = (-2, 8) ∨ 
  (x, y) = (4 + Real.sqrt 15, 4 - Real.sqrt 15) ∨ 
  (x, y) = (4 - Real.sqrt 15, 4 + Real.sqrt 15) :=
by 
  sorry

end find_pairs_solution_l169_169267


namespace division_proof_l169_169160

theorem division_proof :
  ((2 * 4 * 6) / (1 + 3 + 5 + 7) - (1 * 3 * 5) / (2 + 4 + 6)) / (1 / 2) = 3.5 :=
by
  -- definitions based on conditions
  let numerator1 := 2 * 4 * 6
  let denominator1 := 1 + 3 + 5 + 7
  let numerator2 := 1 * 3 * 5
  let denominator2 := 2 + 4 + 6
  -- the statement of the theorem
  sorry

end division_proof_l169_169160


namespace total_milk_consumed_l169_169438

theorem total_milk_consumed (regular_milk : ℝ) (soy_milk : ℝ) (H1 : regular_milk = 0.5) (H2: soy_milk = 0.1) :
    regular_milk + soy_milk = 0.6 :=
  by
  sorry

end total_milk_consumed_l169_169438


namespace cos_2theta_plus_sin_2theta_l169_169440

theorem cos_2theta_plus_sin_2theta (θ : ℝ) (h : 3 * Real.sin θ = Real.cos θ) : 
  Real.cos (2 * θ) + Real.sin (2 * θ) = 7 / 5 :=
by
  sorry

end cos_2theta_plus_sin_2theta_l169_169440


namespace negation_of_all_honest_l169_169238

-- Define the needed predicates
variable {Man : Type} -- Type for men
variable (man : Man → Prop)
variable (age : Man → ℕ)
variable (honest : Man → Prop)

-- Define the conditions and the statement we want to prove
theorem negation_of_all_honest :
  (∀ x, man x → age x > 30 → honest x) →
  (∃ x, man x ∧ age x > 30 ∧ ¬ honest x) :=
sorry

end negation_of_all_honest_l169_169238


namespace final_jacket_price_is_correct_l169_169884

-- Define the initial price, the discounts, and the tax rate
def initial_price : ℝ := 120
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25
def sales_tax : ℝ := 0.05

-- Calculate the final price using the given conditions
noncomputable def price_after_first_discount := initial_price * (1 - first_discount)
noncomputable def price_after_second_discount := price_after_first_discount * (1 - second_discount)
noncomputable def final_price := price_after_second_discount * (1 + sales_tax)

-- The theorem to prove
theorem final_jacket_price_is_correct : final_price = 75.60 := by
  -- The proof is omitted
  sorry

end final_jacket_price_is_correct_l169_169884


namespace seventh_oblong_number_l169_169855

/-- An oblong number is the number of dots in a rectangular grid where the number of rows is one more than the number of columns. -/
def is_oblong_number (n : ℕ) (x : ℕ) : Prop :=
  x = n * (n + 1)

/-- The 7th oblong number is 56. -/
theorem seventh_oblong_number : ∃ x, is_oblong_number 7 x ∧ x = 56 :=
by 
  use 56
  unfold is_oblong_number
  constructor
  rfl -- This confirms the computation 7 * 8 = 56
  sorry -- Wrapping up the proof, no further steps needed

end seventh_oblong_number_l169_169855


namespace cube_surface_area_increase_l169_169663

theorem cube_surface_area_increase (s : ℝ) : 
    let initial_area := 6 * s^2
    let new_edge := 1.3 * s
    let new_area := 6 * (new_edge)^2
    let incr_area := new_area - initial_area
    let percentage_increase := (incr_area / initial_area) * 100
    percentage_increase = 69 :=
by
  let initial_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_area := 6 * (new_edge)^2
  let incr_area := new_area - initial_area
  let percentage_increase := (incr_area / initial_area) * 100
  sorry

end cube_surface_area_increase_l169_169663


namespace right_triangle_sides_l169_169320

theorem right_triangle_sides (a d : ℝ) (k : ℕ) (h_pos_a : 0 < a) (h_pos_d : 0 < d) (h_pos_k : 0 < k) :
  (a = 3) ∧ (d = 1) ∧ (k = 2) ↔ (a^2 + (a + d)^2 = (a + k * d)^2) :=
by 
  sorry

end right_triangle_sides_l169_169320


namespace horner_eval_f_at_5_eval_f_at_5_l169_169524

def f (x: ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem horner_eval_f_at_5 :
  f 5 = ((((5 - 2) * 5 + 1) * 5 + 1) * 5 - 1) * 5 - 5 := by
  sorry

theorem eval_f_at_5 : f 5 = 2015 := by 
  have h : f 5 = ((((5 - 2) * 5 + 1) * 5 + 1) * 5 - 1) * 5 - 5 := by
    apply horner_eval_f_at_5
  rw [h]
  norm_num

end horner_eval_f_at_5_eval_f_at_5_l169_169524


namespace cubic_polynomial_roots_l169_169640

variables (a b c : ℚ)

theorem cubic_polynomial_roots (a b c : ℚ) :
  (c = 0 → ∃ x y z : ℚ, (x = 0 ∧ y = 1 ∧ z = -2) ∧ (-a = x + y + z) ∧ (b = x*y + y*z + z*x) ∧ (-c = x*y*z)) ∧
  (c ≠ 0 → ∃ x y z : ℚ, (x = 1 ∧ y = -1 ∧ z = -1) ∧ (-a = x + y + z) ∧ (b = x*y + y*z + z*x) ∧ (-c = x*y*z)) :=
by
  sorry

end cubic_polynomial_roots_l169_169640


namespace number_of_common_tangents_between_circleC_and_circleD_l169_169012

noncomputable def circleC := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }

noncomputable def circleD := { p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.1 + 2 * p.2 - 4 = 0 }

theorem number_of_common_tangents_between_circleC_and_circleD : 
    ∃ (num_tangents : ℕ), num_tangents = 2 :=
by
    -- Proving the number of common tangents is 2
    sorry

end number_of_common_tangents_between_circleC_and_circleD_l169_169012


namespace find_change_l169_169439

def initial_amount : ℝ := 1.80
def cost_of_candy_bar : ℝ := 0.45
def change : ℝ := 1.35

theorem find_change : initial_amount - cost_of_candy_bar = change :=
by sorry

end find_change_l169_169439


namespace division_problem_l169_169676

theorem division_problem
  (R : ℕ) (D : ℕ) (Q : ℕ) (Div : ℕ)
  (hR : R = 5)
  (hD1 : D = 3 * Q)
  (hD2 : D = 3 * R + 3) :
  Div = D * Q + R :=
by
  have hR : R = 5 := hR
  have hD2 := hD2
  have hDQ := hD1
  -- Proof continues with steps leading to the final desired conclusion
  sorry

end division_problem_l169_169676


namespace cornbread_pieces_count_l169_169317

def cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ) : ℕ := 
  (pan_length * pan_width) / (piece_length * piece_width)

theorem cornbread_pieces_count :
  cornbread_pieces 24 20 3 3 = 53 :=
by
  -- The definitions and the equivalence transformation tell us that this is true
  sorry

end cornbread_pieces_count_l169_169317


namespace find_m_l169_169445

theorem find_m (S : ℕ → ℕ) (a : ℕ → ℕ) (m : ℕ) (h1 : ∀ n, S n = n^2)
  (h2 : S m = (a m + a (m + 1)) / 2)
  (h3 : ∀ n > 1, a n = S n - S (n - 1))
  (h4 : a 1 = 1) :
  m = 2 :=
sorry

end find_m_l169_169445


namespace length_of_train_B_l169_169110

-- Given conditions
def lengthTrainA := 125  -- in meters
def speedTrainA := 54    -- in km/hr
def speedTrainB := 36    -- in km/hr
def timeToCross := 11    -- in seconds

-- Conversion factor from km/hr to m/s
def kmhr_to_mps (v : ℕ) : ℕ := v * 5 / 18

-- Relative speed of the trains in m/s
def relativeSpeed := kmhr_to_mps (speedTrainA + speedTrainB)

-- Distance covered in the given time
def distanceCovered := relativeSpeed * timeToCross

-- Proof statement
theorem length_of_train_B : distanceCovered - lengthTrainA = 150 := 
by
  -- Proof will go here
  sorry

end length_of_train_B_l169_169110


namespace subtraction_result_l169_169441

-- Define the condition as given: x - 46 = 15
def condition (x : ℤ) := x - 46 = 15

-- Define the theorem that gives us the equivalent mathematical statement we want to prove
theorem subtraction_result (x : ℤ) (h : condition x) : x - 29 = 32 :=
by
  -- Here we would include the proof steps, but as per instructions we will use 'sorry' to skip the proof
  sorry

end subtraction_result_l169_169441


namespace arithmetic_sequence_k_value_l169_169721

theorem arithmetic_sequence_k_value (a_1 d : ℕ) (h1 : a_1 = 1) (h2 : d = 2) (k : ℕ) (S : ℕ → ℕ) (h_sum : ∀ n, S n = n * (2 * a_1 + (n - 1) * d) / 2) (h_condition : S (k + 2) - S k = 24) : k = 5 :=
by {
  sorry
}

end arithmetic_sequence_k_value_l169_169721


namespace best_fitting_model_l169_169388

theorem best_fitting_model :
  ∀ (R1 R2 R3 R4 : ℝ), R1 = 0.976 → R2 = 0.776 → R3 = 0.076 → R4 = 0.351 →
  R1 = max R1 (max R2 (max R3 R4)) :=
by
  intros R1 R2 R3 R4 hR1 hR2 hR3 hR4
  rw [hR1, hR2, hR3, hR4]
  sorry

end best_fitting_model_l169_169388


namespace initially_caught_and_tagged_fish_l169_169887

theorem initially_caught_and_tagged_fish (N T : ℕ) (hN : N = 800) (h_ratio : 2 / 40 = T / N) : T = 40 :=
by
  have hN : N = 800 := hN
  have h_ratio : 2 / 40 = T / 800 := by rw [hN] at h_ratio; exact h_ratio
  sorry

end initially_caught_and_tagged_fish_l169_169887


namespace inequality_reciprocal_of_negative_l169_169788

variable {a b : ℝ}

theorem inequality_reciprocal_of_negative (h : a < b) (h_neg_a : a < 0) (h_neg_b : b < 0) : 
  (1 / a) > (1 / b) := by
  sorry

end inequality_reciprocal_of_negative_l169_169788


namespace factor_x4_minus_81_l169_169499

theorem factor_x4_minus_81 : ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intro x
  sorry

end factor_x4_minus_81_l169_169499


namespace find_symbols_l169_169849

theorem find_symbols (x y otimes oplus : ℝ) 
  (h1 : x + otimes * y = 3) 
  (h2 : 3 * x - otimes * y = 1) 
  (h3 : x = oplus) 
  (h4 : y = 1) : 
  otimes = 2 ∧ oplus = 1 := 
by
  sorry

end find_symbols_l169_169849


namespace simplify_cube_root_18_24_30_l169_169695

noncomputable def cube_root_simplification (a b c : ℕ) : ℕ :=
  let sum_cubes := a^3 + b^3 + c^3
  36

theorem simplify_cube_root_18_24_30 : 
  cube_root_simplification 18 24 30 = 36 :=
by {
  -- Proof steps would go here
  sorry
}

end simplify_cube_root_18_24_30_l169_169695


namespace infinite_series_k3_over_3k_l169_169866

theorem infinite_series_k3_over_3k :
  ∑' k : ℕ, (k^3 : ℝ) / 3^k = 165 / 16 := 
sorry

end infinite_series_k3_over_3k_l169_169866


namespace distance_missouri_to_new_york_by_car_l169_169906

variable (d_flight d_car : ℚ)

theorem distance_missouri_to_new_york_by_car :
  d_car = 1.4 * d_flight → 
  d_car = 1400 → 
  (d_car / 2 = 700) :=
by
  intros h1 h2
  sorry

end distance_missouri_to_new_york_by_car_l169_169906


namespace sum_of_integers_l169_169749

theorem sum_of_integers (a b c d : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) (h4 : d > 1)
    (h_prod : a * b * c * d = 1000000)
    (h_gcd1 : Nat.gcd a b = 1) (h_gcd2 : Nat.gcd a c = 1) (h_gcd3 : Nat.gcd a d = 1)
    (h_gcd4 : Nat.gcd b c = 1) (h_gcd5 : Nat.gcd b d = 1) (h_gcd6 : Nat.gcd c d = 1) : 
    a + b + c + d = 15698 :=
sorry

end sum_of_integers_l169_169749


namespace seashells_increase_l169_169245

def initial_seashells : ℕ := 50
def final_seashells : ℕ := 130
def week_increment (x : ℕ) : ℕ := 4 * x + initial_seashells

theorem seashells_increase (x : ℕ) (h: final_seashells = week_increment x) : x = 8 :=
by {
  sorry
}

end seashells_increase_l169_169245


namespace correct_statements_l169_169931

-- Statement B
def statementB : Prop := 
∀ x : ℝ, x < 1/2 → (∃ y : ℝ, y = 2 * x + 1 / (2 * x - 1) ∧ y = -1)

-- Statement D
def statementD : Prop :=
∃ y : ℝ, (∀ x : ℝ, y = 1 / (Real.sin x) ^ 2 + 4 / (Real.cos x) ^ 2) ∧ y = 9

-- Combined proof problem
theorem correct_statements : statementB ∧ statementD :=
sorry

end correct_statements_l169_169931


namespace initial_bottles_of_water_l169_169367

theorem initial_bottles_of_water {B : ℕ} (h1 : 100 - (6 * B + 5) = 71) : B = 4 :=
by
  sorry

end initial_bottles_of_water_l169_169367


namespace renaming_not_unnoticeable_l169_169096

-- Define the conditions as necessary structures for cities and connections
structure City := (name : String)
structure Connection := (city1 city2 : City)

-- Definition of the king's list of connections
def kingList : List Connection := sorry  -- The complete list of connections

-- The renaming function represented generically
def rename (c1 c2 : City) : City := sorry  -- The renaming function which is unspecified here

-- The main theorem statement
noncomputable def renaming_condition (c1 c2 : City) : Prop :=
  -- This condition represents that renaming preserves the king's perception of connections
  ∀ c : City, sorry  -- The specific condition needs full details of renaming logic

-- The theorem to prove, which states that the renaming is not always unnoticeable
theorem renaming_not_unnoticeable : ∃ c1 c2 : City, ¬ renaming_condition c1 c2 := sorry

end renaming_not_unnoticeable_l169_169096


namespace positive_multiples_of_6_l169_169799

theorem positive_multiples_of_6 (k a b : ℕ) (h₁ : a = (3 + 3 * k))
  (h₂ : b = 24) (h₃ : a^2 - b^2 = 0) : k = 7 :=
sorry

end positive_multiples_of_6_l169_169799


namespace average_age_across_rooms_l169_169904

theorem average_age_across_rooms :
  let room_a_people := 8
  let room_a_average_age := 35
  let room_b_people := 5
  let room_b_average_age := 30
  let room_c_people := 7
  let room_c_average_age := 25
  let total_people := room_a_people + room_b_people + room_c_people
  let total_age := (room_a_people * room_a_average_age) + (room_b_people * room_b_average_age) + (room_c_people * room_c_average_age)
  let average_age := total_age / total_people
  average_age = 30.25 := by
{
  sorry
}

end average_age_across_rooms_l169_169904


namespace metal_contest_winner_l169_169892

theorem metal_contest_winner (x y : ℕ) (hx : 95 * x + 74 * y = 2831) : x = 15 ∧ y = 19 ∧ 95 * 15 > 74 * 19 := by
  sorry

end metal_contest_winner_l169_169892


namespace tiger_time_to_pass_specific_point_l169_169697

theorem tiger_time_to_pass_specific_point :
  ∀ (distance_tree : ℝ) (time_tree : ℝ) (length_tiger : ℝ),
  distance_tree = 20 →
  time_tree = 5 →
  length_tiger = 5 →
  (length_tiger / (distance_tree / time_tree)) = 1.25 :=
by
  intros distance_tree time_tree length_tiger h1 h2 h3
  rw [h1, h2, h3]
  sorry

end tiger_time_to_pass_specific_point_l169_169697


namespace average_branches_per_foot_l169_169394

theorem average_branches_per_foot :
  let b1 := 200
  let h1 := 50
  let b2 := 180
  let h2 := 40
  let b3 := 180
  let h3 := 60
  let b4 := 153
  let h4 := 34
  (b1 / h1 + b2 / h2 + b3 / h3 + b4 / h4) / 4 = 4 := by
  sorry

end average_branches_per_foot_l169_169394


namespace print_shop_x_charges_l169_169181

theorem print_shop_x_charges (x : ℝ) (h1 : ∀ y : ℝ, y = 1.70) (h2 : 40 * x + 20 = 40 * 1.70) : x = 1.20 :=
by
  sorry

end print_shop_x_charges_l169_169181


namespace angle_C_exceeds_120_degrees_l169_169417

theorem angle_C_exceeds_120_degrees 
  (a b : ℝ) (h_a : a = Real.sqrt 3) (h_b : b = Real.sqrt 3) (c : ℝ) (h_c : c > 3) :
  ∀ (C : ℝ), C = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) 
             → C > 120 :=
by
  sorry

end angle_C_exceeds_120_degrees_l169_169417


namespace maria_workers_problem_l169_169909

-- Define the initial conditions
def initial_days : ℕ := 40
def days_passed : ℕ := 10
def fraction_completed : ℚ := 2/5
def initial_workers : ℕ := 10

-- Define the required minimum number of workers to complete the job on time
def minimum_workers_required : ℕ := 5

-- The theorem statement
theorem maria_workers_problem 
  (initial_days : ℕ)
  (days_passed : ℕ)
  (fraction_completed : ℚ)
  (initial_workers : ℕ) :
  ( ∀ (total_days remaining_days : ℕ), 
    initial_days = 40 ∧ days_passed = 10 ∧ fraction_completed = 2/5 ∧ initial_workers = 10 → 
    remaining_days = initial_days - days_passed ∧ 
    total_days = initial_days ∧ 
    fraction_completed + (remaining_days / total_days) = 1) →
  minimum_workers_required = 5 := 
sorry

end maria_workers_problem_l169_169909


namespace population_net_increase_one_day_l169_169641

-- Define the given rates and constants
def birth_rate := 10 -- people per 2 seconds
def death_rate := 2 -- people per 2 seconds
def seconds_per_day := 24 * 60 * 60 -- seconds

-- Define the expected net population increase per second
def population_increase_per_sec := (birth_rate / 2) - (death_rate / 2)

-- Define the expected net population increase per day
def expected_population_increase_per_day := population_increase_per_sec * seconds_per_day

theorem population_net_increase_one_day :
  expected_population_increase_per_day = 345600 := by
  -- This will skip the proof implementation.
  sorry

end population_net_increase_one_day_l169_169641


namespace exists_polynomial_for_divisors_l169_169031

open Polynomial

theorem exists_polynomial_for_divisors (n : ℕ) :
  (∃ P : ℤ[X], ∀ d : ℕ, d ∣ n → P.eval (d : ℤ) = (n / d : ℤ)^2) ↔
  (Nat.Prime n ∨ n = 1 ∨ n = 6) := by
  sorry

end exists_polynomial_for_divisors_l169_169031


namespace find_fz_l169_169097

noncomputable def v (x y : ℝ) : ℝ :=
  3^x * Real.sin (y * Real.log 3)

theorem find_fz (x y : ℝ) (C : ℂ) (z : ℂ) (hz : z = x + y * Complex.I) :
  ∃ f : ℂ → ℂ, f z = 3^z + C :=
by
  sorry

end find_fz_l169_169097


namespace total_cost_of_puzzles_l169_169470

-- Definitions for the costs of large and small puzzles
def large_puzzle_cost : ℕ := 15
def small_puzzle_cost : ℕ := 23 - large_puzzle_cost

-- Theorem statement
theorem total_cost_of_puzzles :
  (large_puzzle_cost + 3 * small_puzzle_cost) = 39 :=
by
  -- Placeholder for the proof
  sorry

end total_cost_of_puzzles_l169_169470


namespace polynomial_divisible_by_5040_l169_169793

theorem polynomial_divisible_by_5040 (n : ℤ) (hn : n > 3) :
  5040 ∣ (n^7 - 14 * n^5 + 49 * n^3 - 36 * n) :=
sorry

end polynomial_divisible_by_5040_l169_169793


namespace WidgetsPerHour_l169_169497

theorem WidgetsPerHour 
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (widgets_per_week : ℕ) 
  (H1 : hours_per_day = 8)
  (H2 : days_per_week = 5)
  (H3 : widgets_per_week = 800) : 
  widgets_per_week / (hours_per_day * days_per_week) = 20 := 
sorry

end WidgetsPerHour_l169_169497


namespace tan_identity_l169_169512

theorem tan_identity
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 3 / 7)
  (h2 : Real.tan (β - Real.pi / 4) = -1 / 3)
  : Real.tan (α + Real.pi / 4) = 8 / 9 := by
  sorry

end tan_identity_l169_169512


namespace fraction_of_age_l169_169477

theorem fraction_of_age (jane_age_current : ℕ) (years_since_babysit : ℕ) (age_oldest_babysat_current : ℕ) :
  jane_age_current = 32 →
  years_since_babysit = 12 →
  age_oldest_babysat_current = 23 →
  ∃ (f : ℚ), f = 11 / 20 :=
by
  intros
  sorry

end fraction_of_age_l169_169477


namespace sqrt_product_simplification_l169_169163

theorem sqrt_product_simplification (q : ℝ) : 
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 14 * q * Real.sqrt (21 * q) :=
by sorry

end sqrt_product_simplification_l169_169163


namespace anns_age_l169_169162

theorem anns_age (a b : ℕ)
  (h1 : a + b = 72)
  (h2 : ∃ y, y = a - b)
  (h3 : b = a / 3 + 2 * (a - b)) : a = 36 :=
by
  sorry

end anns_age_l169_169162


namespace worker_b_time_l169_169792

theorem worker_b_time (T_B : ℝ) : 
  (1 / 10) + (1 / T_B) = 1 / 6 → T_B = 15 := by
  intro h
  sorry

end worker_b_time_l169_169792


namespace calvin_total_insects_l169_169220

-- Definitions based on the conditions
def roaches := 12
def scorpions := 3
def crickets := roaches / 2
def caterpillars := scorpions * 2

-- Statement of the problem
theorem calvin_total_insects : 
  roaches + scorpions + crickets + caterpillars = 27 :=
  by
    sorry

end calvin_total_insects_l169_169220


namespace min_valid_n_l169_169831

theorem min_valid_n (n : ℕ) (h_pos : 0 < n) (h_int : ∃ m : ℕ, m * m = 51 + n) : n = 13 :=
  sorry

end min_valid_n_l169_169831


namespace number_of_workers_who_read_all_three_books_l169_169307

theorem number_of_workers_who_read_all_three_books
  (W S K A SK SA KA SKA N : ℝ)
  (hW : W = 75)
  (hS : S = 1 / 2 * W)
  (hK : K = 1 / 4 * W)
  (hA : A = 1 / 5 * W)
  (hSK : SK = 2 * SKA)
  (hN : N = S - (SK + SA + SKA) - 1)
  (hTotal : S + K + A - (SK + SA + KA - SKA) + N = W) :
  SKA = 6 :=
by
  -- The proof steps are omitted
  sorry

end number_of_workers_who_read_all_three_books_l169_169307


namespace adult_ticket_cost_is_19_l169_169648

variable (A : ℕ) -- the cost for an adult ticket
def child_ticket_cost : ℕ := 15
def total_receipts : ℕ := 7200
def total_attendance : ℕ := 400
def adults_attendance : ℕ := 280
def children_attendance : ℕ := 120

-- The equation representing the total receipts
theorem adult_ticket_cost_is_19 (h : total_receipts = 280 * A + 120 * child_ticket_cost) : A = 19 :=
  by sorry

end adult_ticket_cost_is_19_l169_169648


namespace sin_cos_identity_l169_169958

theorem sin_cos_identity :
  (Real.sin (75 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) -
   Real.sin (15 * Real.pi / 180) * Real.sin (150 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  sorry

end sin_cos_identity_l169_169958


namespace Kolya_can_form_triangles_l169_169475

theorem Kolya_can_form_triangles :
  ∃ (K1a K1b K1c K3a K3b K3c V1 V2 V3 : ℝ), 
  (K1a + K1b + K1c = 1) ∧
  (K3a + K3b + K3c = 1) ∧
  (V1 + V2 + V3 = 1) ∧
  (K1a = 0.5) ∧ (K1b = 0.25) ∧ (K1c = 0.25) ∧
  (K3a = 0.5) ∧ (K3b = 0.25) ∧ (K3c = 0.25) ∧
  (∀ (V1 V2 V3 : ℝ), V1 + V2 + V3 = 1 → 
  (
    (K1a + V1 > K3b ∧ K1a + K3b > V1 ∧ V1 + K3b > K1a) ∧ 
    (K1b + V2 > K3a ∧ K1b + K3a > V2 ∧ V2 + K3a > K1b) ∧ 
    (K1c + V3 > K3c ∧ K1c + K3c > V3 ∧ V3 + K3c > K1c)
  )) :=
sorry

end Kolya_can_form_triangles_l169_169475


namespace find_expression_value_find_m_value_find_roots_and_theta_l169_169248

-- Define the conditions
variable (θ : ℝ) (m : ℝ)
variable (h1 : θ > 0) (h2 : θ < 2 * Real.pi)
variable (h3 : ∀ x, (2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0) → (x = Real.sin θ ∨ x = Real.cos θ))

-- Theorem 1: Find the value of a given expression
theorem find_expression_value :
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + Real.cos θ / (1 - Real.tan θ) = (Real.sqrt 3 + 1) / 2 :=
  sorry

-- Theorem 2: Find the value of m
theorem find_m_value :
  m = Real.sqrt 3 / 2 :=
  sorry

-- Theorem 3: Find the roots of the equation and the value of θ
theorem find_roots_and_theta :
  (∀ x, (2 * x^2 - (Real.sqrt 3 + 1) * x + Real.sqrt 3 / 2 = 0) → (x = Real.sqrt 3 / 2 ∨ x = 1 / 2)) ∧
  (θ = Real.pi / 6 ∨ θ = Real.pi / 3) :=
  sorry

end find_expression_value_find_m_value_find_roots_and_theta_l169_169248


namespace find_a_from_derivative_l169_169372

-- Define the function f(x) = ax^3 + 3x^2 - 6
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

-- State the theorem to prove that a = 10/3 given f'(-1) = 4
theorem find_a_from_derivative (a : ℝ) (h : f' a (-1) = 4) : a = 10 / 3 := 
  sorry

end find_a_from_derivative_l169_169372


namespace geometric_sequence_common_ratio_l169_169008

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {S : ℕ → ℝ} (q : ℝ) 
  (h_pos : ∀ n, 0 < a n)
  (h_geo : ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q))  
  (h_condition : ∀ n : ℕ+, S (2 * n) / S n < 5) :
  0 < q ∧ q ≤ 1 :=
sorry

end geometric_sequence_common_ratio_l169_169008


namespace find_point_B_coordinates_l169_169755

theorem find_point_B_coordinates : 
  ∃ B : ℝ × ℝ, 
    (∀ A C B : ℝ × ℝ, A = (2, 3) ∧ C = (0, 1) ∧ 
    (B.1 - A.1, B.2 - A.2) = (-2) • (C.1 - B.1, C.2 - B.2)) → B = (-2, -1) :=
by 
  sorry

end find_point_B_coordinates_l169_169755


namespace activity_participants_l169_169235

variable (A B C D : Prop)

theorem activity_participants (h1 : A → B) (h2 : ¬C → ¬B) (h3 : C → ¬D) : B ∧ C ∧ ¬A ∧ ¬D :=
by
  sorry

end activity_participants_l169_169235


namespace parallel_lines_condition_l169_169536

-- We define the conditions as Lean definitions
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + (a^2 - 1) = 0
def parallel_condition (a : ℝ) : Prop := (a ≠ 0) ∧ (a ≠ 1) ∧ (a ≠ -1) ∧ (a * (a^2 - 1) ≠ 6)

-- Mathematically equivalent Lean 4 statement
theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, line1 a x y → line2 a x y → (line1 a x y ↔ line2 a x y)) ↔ (a = -1) :=
by 
  -- The full proof would be written here
  sorry

end parallel_lines_condition_l169_169536


namespace no_intersection_abs_functions_l169_169138

open Real

theorem no_intersection_abs_functions : 
  ∀ f g : ℝ → ℝ, 
  (∀ x, f x = |2 * x + 5|) → 
  (∀ x, g x = -|3 * x - 2|) → 
  (∀ y, ∀ x1 x2, f x1 = y ∧ g x2 = y → y = 0 ∧ x1 = -5/2 ∧ x2 = 2/3 → (x1 ≠ x2)) → 
  (∃ x, f x = g x) → 
  false := 
  by
    intro f g hf hg h
    sorry

end no_intersection_abs_functions_l169_169138


namespace percent_increase_output_l169_169935

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end percent_increase_output_l169_169935


namespace angle_B_range_l169_169625

theorem angle_B_range (A B C : ℝ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : A + B + C = 180) (h4 : 2 * B = 5 * A) :
  0 < B ∧ B ≤ 75 :=
by
  sorry

end angle_B_range_l169_169625


namespace compute_expression_l169_169400

theorem compute_expression : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end compute_expression_l169_169400


namespace measure_of_angle_A_proof_range_of_values_of_b_plus_c_over_a_proof_l169_169270

noncomputable def measure_of_angle_a (a b c : ℝ) (S : ℝ) (h_c : c = 2) (h_S : b * Real.cos (A / 2) = S) : Prop :=
  A = Real.pi / 3

theorem measure_of_angle_A_proof (a b c : ℝ) (S : ℝ) (h_c : c = 2) (h_S : b * Real.cos (A / 2) = S) : measure_of_angle_a a b c S h_c h_S :=
sorry

noncomputable def range_of_values_of_b_plus_c_over_a (a b c : ℝ) (A : ℝ) (h_A : A = Real.pi / 3) (h_c : c = 2) : Set ℝ :=
  {x : ℝ | 1 < x ∧ x ≤ 2}

theorem range_of_values_of_b_plus_c_over_a_proof (a b c : ℝ) (A : ℝ) (h_A : A = Real.pi / 3) (h_c : c = 2) : 
  ∃ x, x ∈ range_of_values_of_b_plus_c_over_a a b c A h_A h_c :=
sorry

end measure_of_angle_A_proof_range_of_values_of_b_plus_c_over_a_proof_l169_169270


namespace mike_net_spending_l169_169646

-- Definitions for given conditions
def trumpet_cost : ℝ := 145.16
def song_book_revenue : ℝ := 5.84

-- Theorem stating the result
theorem mike_net_spending : trumpet_cost - song_book_revenue = 139.32 :=
by 
  sorry

end mike_net_spending_l169_169646


namespace rhombus_area_2sqrt2_l169_169992

structure Rhombus (α : Type _) :=
  (side_length : ℝ)
  (angle : ℝ)

theorem rhombus_area_2sqrt2 (R : Rhombus ℝ) (h_side : R.side_length = 2) (h_angle : R.angle = 45) :
  ∃ A : ℝ, A = 2 * Real.sqrt 2 :=
by
  let A := 2 * Real.sqrt 2
  existsi A
  sorry

end rhombus_area_2sqrt2_l169_169992


namespace total_number_of_items_l169_169607

theorem total_number_of_items (total_items : ℕ) (selected_items : ℕ) (h1 : total_items = 50) (h2 : selected_items = 10) : total_items = 50 :=
by
  exact h1

end total_number_of_items_l169_169607


namespace current_average_is_35_l169_169563

noncomputable def cricket_avg (A : ℝ) : Prop :=
  let innings := 10
  let next_runs := 79
  let increase := 4
  (innings * A + next_runs = (A + increase) * (innings + 1))

theorem current_average_is_35 : cricket_avg 35 :=
by
  unfold cricket_avg
  simp only
  sorry

end current_average_is_35_l169_169563


namespace ivanov_voted_against_kuznetsov_l169_169331

theorem ivanov_voted_against_kuznetsov
    (members : List String)
    (vote : String → String)
    (majority_dismissed : (String × Nat))
    (petrov_statement : String)
    (ivanov_concluded : Bool) :
  members = ["Ivanov", "Petrov", "Sidorov", "Kuznetsov"] →
  (∀ x ∈ members, vote x ∈ members ∧ vote x ≠ x) →
  majority_dismissed = ("Ivanov", 3) →
  petrov_statement = "Petrov voted against Kuznetsov" →
  ivanov_concluded = True →
  vote "Ivanov" = "Kuznetsov" :=
by
  intros members_cond vote_cond majority_cond petrov_cond ivanov_cond
  sorry

end ivanov_voted_against_kuznetsov_l169_169331


namespace trapezoid_base_lengths_l169_169341

noncomputable def trapezoid_bases (d h : Real) : Real × Real :=
  let b := h - 2 * d
  let B := h + 2 * d
  (b, B)

theorem trapezoid_base_lengths :
  ∀ (d : Real), d = Real.sqrt 3 →
  ∀ (h : Real), h = Real.sqrt 48 →
  ∃ (b B : Real), trapezoid_bases d h = (b, B) ∧ b = Real.sqrt 48 - 2 * Real.sqrt 3 ∧ B = Real.sqrt 48 + 2 * Real.sqrt 3 := by 
  sorry

end trapezoid_base_lengths_l169_169341


namespace exists_excircle_radius_at_least_three_times_incircle_radius_l169_169609

variable (a b c s T r ra rb rc : ℝ)
variable (ha : ra = T / (s - a))
variable (hb : rb = T / (s - b))
variable (hc : rc = T / (s - c))
variable (hincircle : r = T / s)

theorem exists_excircle_radius_at_least_three_times_incircle_radius
  (ha : ra = T / (s - a)) (hb : rb = T / (s - b)) (hc : rc = T / (s - c)) (hincircle : r = T / s) :
  ∃ rc, rc ≥ 3 * r :=
by {
  use rc,
  sorry
}

end exists_excircle_radius_at_least_three_times_incircle_radius_l169_169609


namespace evaluate_expression_l169_169949

theorem evaluate_expression (a : ℤ) : ((a + 10) - a + 3) * ((a + 10) - a - 2) = 104 := by
  sorry

end evaluate_expression_l169_169949


namespace inequality_solution_l169_169647

theorem inequality_solution (x : ℝ) (hx : x > 0) : 
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) := 
by
  sorry

end inequality_solution_l169_169647


namespace theorem_1_valid_theorem_6_valid_l169_169504

theorem theorem_1_valid (a b : ℤ) (h1 : a % 7 = 0) (h2 : b % 7 = 0) : (a + b) % 7 = 0 :=
by sorry

theorem theorem_6_valid (a b : ℤ) (h : (a + b) % 7 ≠ 0) : a % 7 ≠ 0 ∨ b % 7 ≠ 0 :=
by sorry

end theorem_1_valid_theorem_6_valid_l169_169504


namespace ruth_gave_janet_53_stickers_l169_169687

-- Definitions: Janet initially has 3 stickers, after receiving more from Ruth, she has 56 stickers in total.
def janet_initial : ℕ := 3
def janet_total : ℕ := 56

-- The statement to prove: Ruth gave Janet 53 stickers.
def stickers_from_ruth (initial: ℕ) (total: ℕ) : ℕ :=
  total - initial

theorem ruth_gave_janet_53_stickers : stickers_from_ruth janet_initial janet_total = 53 :=
by sorry

end ruth_gave_janet_53_stickers_l169_169687


namespace students_in_miss_evans_class_l169_169671

theorem students_in_miss_evans_class
  (total_contribution : ℕ)
  (class_funds : ℕ)
  (contribution_per_student : ℕ)
  (remaining_contribution : ℕ)
  (num_students : ℕ)
  (h1 : total_contribution = 90)
  (h2 : class_funds = 14)
  (h3 : contribution_per_student = 4)
  (h4 : remaining_contribution = total_contribution - class_funds)
  (h5 : num_students = remaining_contribution / contribution_per_student)
  : num_students = 19 :=
sorry

end students_in_miss_evans_class_l169_169671


namespace average_speed_interval_l169_169371

theorem average_speed_interval {s t : ℝ → ℝ} (h_eq : ∀ t, s t = t^2 + 1) : 
  (s 2 - s 1) / (2 - 1) = 3 :=
by
  sorry

end average_speed_interval_l169_169371


namespace amount_exceeds_l169_169352

theorem amount_exceeds (N : ℕ) (A : ℕ) (h1 : N = 1925) (h2 : N / 7 - N / 11 = A) :
  A = 100 :=
sorry

end amount_exceeds_l169_169352


namespace minimum_y_l169_169957

theorem minimum_y (x : ℝ) (h : x > 1) : (∃ y : ℝ, y = x + 1 / (x - 1) ∧ y = 3) :=
by
  sorry

end minimum_y_l169_169957


namespace rose_bushes_after_work_l169_169736

def initial_rose_bushes := 2
def planned_rose_bushes := 4
def planting_rate := 3
def removed_rose_bushes := 5

theorem rose_bushes_after_work :
  initial_rose_bushes + (planned_rose_bushes * planting_rate) - removed_rose_bushes = 9 :=
by
  sorry

end rose_bushes_after_work_l169_169736


namespace product_of_two_numbers_l169_169205

theorem product_of_two_numbers
  (x y : ℝ)
  (h1 : x - y = 12)
  (h2 : x^2 + y^2 = 106) :
  x * y = 32 := by 
  sorry

end product_of_two_numbers_l169_169205


namespace smallest_total_hot_dogs_l169_169361

def packs_hot_dogs := 12
def packs_buns := 9
def packs_mustard := 18
def packs_ketchup := 24

theorem smallest_total_hot_dogs : Nat.lcm (Nat.lcm (Nat.lcm packs_hot_dogs packs_buns) packs_mustard) packs_ketchup = 72 := by
  sorry

end smallest_total_hot_dogs_l169_169361


namespace marco_strawberries_weight_l169_169312

theorem marco_strawberries_weight 
  (m : ℕ) 
  (total_weight : ℕ := 40) 
  (dad_weight : ℕ := 32) 
  (h : total_weight = m + dad_weight) : 
  m = 8 := 
sorry

end marco_strawberries_weight_l169_169312


namespace gcd_of_powers_l169_169137

theorem gcd_of_powers (m n : ℕ) (h1 : m = 2^2016 - 1) (h2 : n = 2^2008 - 1) : 
  Nat.gcd m n = 255 :=
by
  -- (Definitions and steps are omitted as only the statement is required)
  sorry

end gcd_of_powers_l169_169137


namespace classroom_desks_l169_169455

theorem classroom_desks (N y : ℕ) (h : 16 * y = 21 * N)
  (hN_le: N <= 30 * 16 / 21) (hMultiple: 3 * N % 4 = 0)
  (hy_le: y ≤ 30)
  : y = 21 := by
  sorry

end classroom_desks_l169_169455


namespace count_valid_triangles_l169_169303

def triangle_area (a b c : ℕ) : ℕ :=
  let s := (a + b + c) / 2
  s * (s - a) * (s - b) * (s - c)

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b + c < 20 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2

theorem count_valid_triangles : { n : ℕ // n = 24 } :=
  sorry

end count_valid_triangles_l169_169303


namespace choir_average_age_l169_169886

-- Conditions
def women_count : ℕ := 12
def men_count : ℕ := 10
def avg_age_women : ℝ := 25.0
def avg_age_men : ℝ := 40.0

-- Expected Answer
def expected_avg_age : ℝ := 31.82

-- Proof Statement
theorem choir_average_age :
  ((women_count * avg_age_women) + (men_count * avg_age_men)) / (women_count + men_count) = expected_avg_age :=
by
  sorry

end choir_average_age_l169_169886


namespace cone_lateral_surface_area_ratio_l169_169115

theorem cone_lateral_surface_area_ratio (r l S_lateral S_base : ℝ) (h1 : l = 3 * r)
  (h2 : S_lateral = π * r * l) (h3 : S_base = π * r^2) :
  S_lateral / S_base = 3 :=
by
  sorry

end cone_lateral_surface_area_ratio_l169_169115


namespace tangent_perpendicular_intersection_x_4_l169_169808

noncomputable def f (x : ℝ) := (x^2 / 4) - (4 * Real.log x)
noncomputable def f' (x : ℝ) := (1/2 : ℝ) * x - 4 / x

theorem tangent_perpendicular_intersection_x_4 :
  ∀ x : ℝ, (0 < x) → (f' x = 1) → (x = 4) :=
by {
  sorry
}

end tangent_perpendicular_intersection_x_4_l169_169808


namespace area_of_region_l169_169938

theorem area_of_region (x y : ℝ) : |4 * x - 24| + |3 * y + 10| ≤ 6 → ∃ A : ℝ, A = 12 :=
by
  sorry

end area_of_region_l169_169938


namespace complement_intersection_l169_169176

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 3}

theorem complement_intersection : (U \ N) ∩ M = {4, 5} :=
by 
  sorry

end complement_intersection_l169_169176


namespace problem1_problem2_problem2_equality_l169_169591

variable {a b c d : ℝ}

-- Problem 1
theorem problem1 (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : (a - b) * (b - c) * (c - d) * (d - a) = -3) (h5 : a + b + c + d = 6) : d < 0.36 :=
sorry

-- Problem 2
theorem problem2 (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : (a - b) * (b - c) * (c - d) * (d - a) = -3) (h5 : a^2 + b^2 + c^2 + d^2 = 14) : (a + c) * (b + d) ≤ 8 :=
sorry

theorem problem2_equality (h1 : a = 3) (h2 : b = 2) (h3 : c = 1) (h4 : d = 0) : (a + c) * (b + d) = 8 :=
sorry

end problem1_problem2_problem2_equality_l169_169591


namespace min_sine_range_l169_169263

noncomputable def min_sine_ratio (α β γ : ℝ) := min (Real.sin β / Real.sin α) (Real.sin γ / Real.sin β)

theorem min_sine_range (α β γ : ℝ) (h1 : 0 < α) (h2 : α ≤ β) (h3 : β ≤ γ) (h4 : α + β + γ = Real.pi) :
  1 ≤ min_sine_ratio α β γ ∧ min_sine_ratio α β γ < (1 + Real.sqrt 5) / 2 :=
by
  sorry

end min_sine_range_l169_169263


namespace distance_P_to_outer_circle_l169_169228

theorem distance_P_to_outer_circle
  (r_large r_small : ℝ) 
  (h_tangent_inner : true) 
  (h_tangent_diameter : true) 
  (P : ℝ) 
  (O1P : ℝ)
  (O2P : ℝ := r_small)
  (O1O2 : ℝ := r_large - r_small)
  (h_O1O2_eq_680 : O1O2 = 680)
  (h_O2P_eq_320 : O2P = 320) :
  r_large - O1P = 400 :=
by
  sorry

end distance_P_to_outer_circle_l169_169228


namespace smallest_odd_number_with_five_different_prime_factors_l169_169333

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l169_169333


namespace remainder_18_l169_169986

theorem remainder_18 (x : ℤ) (k : ℤ) (h : x = 62 * k + 7) :
  (x + 11) % 31 = 18 :=
by
  sorry

end remainder_18_l169_169986


namespace find_value_of_2_times_x_minus_y_squared_minus_3_l169_169551

-- Define the conditions as noncomputable variables
variables (x y : ℝ)

-- State the main theorem
theorem find_value_of_2_times_x_minus_y_squared_minus_3 :
  (x^2 - x*y = 12) →
  (y^2 - y*x = 15) →
  2 * (x - y)^2 - 3 = 51 :=
by
  intros h1 h2
  sorry

end find_value_of_2_times_x_minus_y_squared_minus_3_l169_169551


namespace cost_of_2000_pieces_of_gum_l169_169370

theorem cost_of_2000_pieces_of_gum
  (cost_per_piece_in_cents : Nat)
  (pieces_of_gum : Nat)
  (conversion_rate_cents_to_dollars : Nat)
  (h1 : cost_per_piece_in_cents = 5)
  (h2 : pieces_of_gum = 2000)
  (h3 : conversion_rate_cents_to_dollars = 100) :
  (cost_per_piece_in_cents * pieces_of_gum) / conversion_rate_cents_to_dollars = 100 := 
by
  sorry

end cost_of_2000_pieces_of_gum_l169_169370


namespace fill_in_blank_with_warning_l169_169454

-- Definitions corresponding to conditions
def is_noun (word : String) : Prop :=
  -- definition of being a noun
  sorry

def corresponds_to_chinese_hint (word : String) (hint : String) : Prop :=
  -- definition of corresponding to a Chinese hint
  sorry

-- The theorem we want to prove
theorem fill_in_blank_with_warning : ∀ word : String, 
  (is_noun word ∧ corresponds_to_chinese_hint word "警告") → word = "warning" :=
by {
  sorry
}

end fill_in_blank_with_warning_l169_169454


namespace perpendicular_line_through_point_l169_169587

theorem perpendicular_line_through_point 
 {x y : ℝ}
 (p : (ℝ × ℝ)) 
 (point : p = (-2, 1)) 
 (perpendicular : ∀ x y, 2 * x - y + 4 = 0) : 
 (∀ x y, x + 2 * y = 0) ∧ (p.fst = -2 ∧ p.snd = 1) :=
by
  sorry

end perpendicular_line_through_point_l169_169587


namespace rabbit_carrots_l169_169716

theorem rabbit_carrots (h_r h_f x : ℕ) (H1 : 5 * h_r = x) (H2 : 6 * h_f = x) (H3 : h_r = h_f + 2) : x = 60 :=
by
  sorry

end rabbit_carrots_l169_169716


namespace Victor_Total_Money_l169_169275

-- Definitions for the conditions
def originalAmount : Nat := 10
def allowance : Nat := 8

-- The proof problem statement
theorem Victor_Total_Money : originalAmount + allowance = 18 := by
  sorry

end Victor_Total_Money_l169_169275


namespace simplify_expr1_simplify_expr2_l169_169384

variable (a b x y : ℝ)

theorem simplify_expr1 : 6 * a + 7 * b^2 - 9 + 4 * a - b^2 + 6 = 10 * a + 6 * b^2 - 3 :=
by
  sorry

theorem simplify_expr2 : 5 * x - 2 * (4 * x + 5 * y) + 3 * (3 * x - 4 * y) = 6 * x - 22 * y :=
by
  sorry

end simplify_expr1_simplify_expr2_l169_169384


namespace min_value_l169_169872

theorem min_value (x : ℝ) (h : x > 1) : x + 4 / (x - 1) ≥ 5 :=
sorry

end min_value_l169_169872


namespace Robert_diff_C_l169_169882

/- Define the conditions as hypotheses -/
variables (C : ℕ) -- Assuming the number of photos Claire has taken as a natural number.

-- Lisa has taken 3 times as many photos as Claire.
def Lisa_photos := 3 * C

-- Robert has taken the same number of photos as Lisa.
def Robert_photos := Lisa_photos C -- which will be 3 * C

-- Proof of the difference.
theorem Robert_diff_C : (Robert_photos C) - C = 2 * C :=
by
  sorry

end Robert_diff_C_l169_169882


namespace exterior_angle_BAC_l169_169690

theorem exterior_angle_BAC (angle_octagon angle_rectangle : ℝ) (h_oct_135 : angle_octagon = 135) (h_rec_90 : angle_rectangle = 90) :
  360 - (angle_octagon + angle_rectangle) = 135 := 
by
  simp [h_oct_135, h_rec_90]
  sorry

end exterior_angle_BAC_l169_169690


namespace evaluate_expression_1_evaluate_expression_2_l169_169844

-- Problem 1
def expression_1 (a b : Int) : Int :=
  2 * a + 3 * b - 2 * a * b - a - 4 * b - a * b

theorem evaluate_expression_1 : expression_1 6 (-1) = 25 :=
by
  sorry

-- Problem 2
def expression_2 (m n : Int) : Int :=
  m^2 + 2 * m * n + n^2

theorem evaluate_expression_2 (m n : Int) (hm : |m| = 3) (hn : |n| = 2) (hmn : m < n) : expression_2 m n = 1 :=
by
  sorry

end evaluate_expression_1_evaluate_expression_2_l169_169844


namespace tailwind_speed_rate_of_change_of_ground_speed_l169_169040

-- Define constants and variables
variables (Vp Vw : ℝ) (altitude Vg1 Vg2 : ℝ)

-- Define conditions
def conditions := Vg1 = Vp + Vw ∧ altitude = 10000 ∧ Vg1 = 460 ∧
                  Vg2 = Vp - Vw ∧ altitude = 5000 ∧ Vg2 = 310

-- Define theorems to prove
theorem tailwind_speed (Vp Vw : ℝ) (altitude Vg1 Vg2 : ℝ) :
  conditions Vp Vw altitude Vg1 Vg2 → Vw = 75 :=
by
  sorry

theorem rate_of_change_of_ground_speed (altitude1 altitude2 Vg1 Vg2 : ℝ) :
  altitude1 = 10000 → altitude2 = 5000 → Vg1 = 460 → Vg2 = 310 →
  (Vg2 - Vg1) / (altitude2 - altitude1) = 0.03 :=
by
  sorry

end tailwind_speed_rate_of_change_of_ground_speed_l169_169040


namespace midpoint_one_sixth_one_ninth_l169_169883

theorem midpoint_one_sixth_one_ninth : (1 / 6 + 1 / 9) / 2 = 5 / 36 := by
  sorry

end midpoint_one_sixth_one_ninth_l169_169883


namespace inequality_range_m_l169_169109

theorem inequality_range_m:
  (∀ x ∈ Set.Icc (Real.sqrt 2) 4, (5 / 2) * x^2 ≥ m * (x - 1)) → m ≤ 10 :=
by 
  intros h 
  sorry

end inequality_range_m_l169_169109


namespace additional_interest_rate_l169_169242

variable (P A1 A2 T SI1 SI2 R AR : ℝ)
variable (h_P : P = 9000)
variable (h_A1 : A1 = 10200)
variable (h_A2 : A2 = 10740)
variable (h_T : T = 3)
variable (h_SI1 : SI1 = A1 - P)
variable (h_SI2 : SI2 = A2 - A1)
variable (h_R : SI1 = P * R * T / 100)
variable (h_AR : SI2 = P * AR * T / 100)

theorem additional_interest_rate :
  AR = 2 := by
  sorry

end additional_interest_rate_l169_169242


namespace tetrahedron_volume_lower_bound_l169_169196

noncomputable def volume_tetrahedron (d1 d2 d3 : ℝ) : ℝ := sorry

theorem tetrahedron_volume_lower_bound {d1 d2 d3 : ℝ} (h1 : d1 > 0) (h2 : d2 > 0) (h3 : d3 > 0) :
  volume_tetrahedron d1 d2 d3 ≥ (1 / 3) * d1 * d2 * d3 :=
sorry

end tetrahedron_volume_lower_bound_l169_169196


namespace other_root_l169_169322

theorem other_root (m n : ℝ) (h : (3 : ℂ) + (1 : ℂ) * Complex.I ∈ {x : ℂ | x^2 + ↑m * x + ↑n = 0}) : 
    (3 : ℂ) - (1 : ℂ) * Complex.I ∈ {x : ℂ | x^2 + ↑m * x + ↑n = 0} :=
sorry

end other_root_l169_169322


namespace smallest_N_exists_l169_169478

theorem smallest_N_exists : ∃ N : ℕ, 
  (N % 9 = 8) ∧
  (N % 8 = 7) ∧
  (N % 7 = 6) ∧
  (N % 6 = 5) ∧
  (N % 5 = 4) ∧
  (N % 4 = 3) ∧
  (N % 3 = 2) ∧
  (N % 2 = 1) ∧
  N = 503 :=
by {
  sorry
}

end smallest_N_exists_l169_169478


namespace difference_of_numbers_l169_169741

theorem difference_of_numbers (x y : ℕ) (h₁ : x + y = 50) (h₂ : Nat.gcd x y = 5) :
  (x - y = 20 ∨ y - x = 20 ∨ x - y = 40 ∨ y - x = 40) :=
sorry

end difference_of_numbers_l169_169741


namespace real_solutions_eq_31_l169_169858

noncomputable def number_of_real_solutions : ℕ :=
  let zero := 0
  let fifty := 50
  let neg_fifty := -50
  let num_intervals := 8
  let num_solutions_per_interval := 2
  let total_solutions := num_intervals * num_solutions_per_interval * 2 - 1
  total_solutions

theorem real_solutions_eq_31 : number_of_real_solutions = 31 := by
  sorry

end real_solutions_eq_31_l169_169858


namespace precision_of_rounded_value_l169_169948

-- Definition of the original problem in Lean 4
def original_value := 27390000000

-- Proof statement to check the precision of the rounded value to the million place
theorem precision_of_rounded_value :
  (original_value % 1000000 = 0) :=
sorry

end precision_of_rounded_value_l169_169948


namespace shooting_accuracy_l169_169498

theorem shooting_accuracy (S : ℕ → ℕ) (H1 : ∀ n, S n < 10 * n / 9) (H2 : ∀ n, S n > 10 * n / 9) :
  ∃ n, 10 * (S n) = 9 * n :=
by
  sorry

end shooting_accuracy_l169_169498


namespace range_of_m_l169_169269

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.exp x) * (2 * x - 1) - m * x + m

def exists_unique_int_n (m : ℝ) : Prop :=
∃! n : ℤ, f n m < 0

theorem range_of_m {m : ℝ} (h : m < 1) (h2 : exists_unique_int_n m) : 
  (Real.exp 1) * (1 / 2) ≤ m ∧ m < 1 :=
sorry

end range_of_m_l169_169269


namespace rooms_needed_l169_169854

/-
  We are given that there are 30 students and each hotel room accommodates 5 students.
  Prove that the number of rooms required to accommodate all students is 6.
-/
theorem rooms_needed (total_students : ℕ) (students_per_room : ℕ) (h1 : total_students = 30) (h2 : students_per_room = 5) : total_students / students_per_room = 6 := by
  -- proof
  sorry

end rooms_needed_l169_169854


namespace amaya_movie_watching_time_l169_169355

theorem amaya_movie_watching_time :
  let uninterrupted_time_1 := 35
  let uninterrupted_time_2 := 45
  let uninterrupted_time_3 := 20
  let rewind_time_1 := 5
  let rewind_time_2 := 15
  let total_uninterrupted := uninterrupted_time_1 + uninterrupted_time_2 + uninterrupted_time_3
  let total_rewind := rewind_time_1 + rewind_time_2
  let total_time := total_uninterrupted + total_rewind
  total_time = 120 := by
  sorry

end amaya_movie_watching_time_l169_169355


namespace find_g_2_l169_169204

variable (g : ℝ → ℝ)

-- Function satisfying the given conditions
axiom g_functional : ∀ (x y : ℝ), g (x - y) = g x * g y
axiom g_nonzero : ∀ (x : ℝ), g x ≠ 0

-- The proof statement
theorem find_g_2 : g 2 = 1 := by
  sorry

end find_g_2_l169_169204


namespace product_of_x_and_y_l169_169383

theorem product_of_x_and_y :
  ∀ (x y : ℝ), (∀ p : ℝ × ℝ, (p = (x, 6) ∨ p = (10, y)) → p.2 = (1 / 2) * p.1) → x * y = 60 :=
by
  intros x y h
  have hx : 6 = (1 / 2) * x := by exact h (x, 6) (Or.inl rfl)
  have hy : y = (1 / 2) * 10 := by exact h (10, y) (Or.inr rfl)
  sorry

end product_of_x_and_y_l169_169383


namespace perpendicular_line_through_circle_center_l169_169700

theorem perpendicular_line_through_circle_center :
  ∃ (a b c : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 2*x - 8 = 0 → x + 2*y = 0 → a * x + b * y + c = 0) ∧
  a = 2 ∧ b = -1 ∧ c = -2 :=
by
  sorry

end perpendicular_line_through_circle_center_l169_169700


namespace savings_duration_before_investment_l169_169522

---- Definitions based on conditions ----
def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def stock_price_per_share : ℕ := 50
def shares_bought : ℕ := 25

---- Derived conditions based on definitions ----
def total_spent_on_stocks := shares_bought * stock_price_per_share
def total_savings_before_investment := 2 * total_spent_on_stocks
def monthly_savings_wife := weekly_savings_wife * 4
def total_monthly_savings := monthly_savings_wife + monthly_savings_husband

---- The theorem statement ----
theorem savings_duration_before_investment :
  total_savings_before_investment / total_monthly_savings = 4 :=
sorry

end savings_duration_before_investment_l169_169522


namespace problem_I_problem_II_l169_169712

variable (t a : ℝ)

-- Problem (I)
theorem problem_I (h1 : a = 1) (h2 : t^2 - 5 * a * t + 4 * a^2 < 0) (h3 : (t - 2) * (t - 6) < 0) : 2 < t ∧ t < 4 := 
by 
  sorry   -- Proof omitted as per instructions

-- Problem (II)
theorem problem_II (h1 : (t - 2) * (t - 6) < 0 → t^2 - 5 * a * t + 4 * a^2 < 0) : 3 / 2 ≤ a ∧ a ≤ 2 :=
by 
  sorry   -- Proof omitted as per instructions

end problem_I_problem_II_l169_169712


namespace loss_per_metre_is_5_l169_169157

-- Definitions
def selling_price (total_meters : ℕ) : ℕ := 18000
def cost_price_per_metre : ℕ := 65
def total_meters : ℕ := 300

-- Loss per meter calculation
def loss_per_metre (selling_price : ℕ) (cost_price_per_metre : ℕ) (total_meters : ℕ) : ℕ :=
  ((cost_price_per_metre * total_meters) - selling_price) / total_meters

-- Theorem statement
theorem loss_per_metre_is_5 : loss_per_metre (selling_price total_meters) cost_price_per_metre total_meters = 5 :=
by
  sorry

end loss_per_metre_is_5_l169_169157


namespace john_purchased_large_bottles_l169_169080

noncomputable def large_bottle_cost : ℝ := 1.75
noncomputable def small_bottle_cost : ℝ := 1.35
noncomputable def num_small_bottles : ℝ := 690
noncomputable def avg_price_paid : ℝ := 1.6163438256658595
noncomputable def total_small_cost : ℝ := num_small_bottles * small_bottle_cost
noncomputable def total_cost (L : ℝ) : ℝ := large_bottle_cost * L + total_small_cost
noncomputable def total_bottles (L : ℝ) : ℝ := L + num_small_bottles

theorem john_purchased_large_bottles : ∃ L : ℝ, 
  (total_cost L / total_bottles L = avg_price_paid) ∧ 
  (L = 1380) := 
sorry

end john_purchased_large_bottles_l169_169080


namespace chairs_bought_l169_169645

theorem chairs_bought (C : ℕ) (tables chairs total_time time_per_furniture : ℕ)
  (h1 : tables = 4)
  (h2 : time_per_furniture = 6)
  (h3 : total_time = 48)
  (h4 : total_time = time_per_furniture * (tables + chairs)) :
  C = 4 :=
by
  -- proof steps are omitted
  sorry

end chairs_bought_l169_169645


namespace multiples_of_7_between_50_and_200_l169_169910

theorem multiples_of_7_between_50_and_200 : 
  ∃ n, n = 21 ∧ ∀ k, (k ≥ 50 ∧ k ≤ 200) ↔ ∃ m, k = 7 * m := sorry

end multiples_of_7_between_50_and_200_l169_169910


namespace min_trucks_needed_l169_169259

theorem min_trucks_needed (n : ℕ) (w : ℕ) (t : ℕ) (total_weight : ℕ) (max_box_weight : ℕ) : 
    (total_weight = 10) → 
    (max_box_weight = 1) → 
    (t = 3) →
    (n * max_box_weight = total_weight) →
    (n ≥ 10) →
    ∀ min_trucks : ℕ, (min_trucks * t ≥ total_weight) → 
    min_trucks = 5 :=
by
  intro total_weight_eq max_box_weight_eq truck_capacity box_total_weight_eq n_lower_bound min_trucks min_trucks_condition
  sorry

end min_trucks_needed_l169_169259


namespace value_of_squared_difference_l169_169180

theorem value_of_squared_difference (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) :
  (x - y)^2 = 9 :=
by
  sorry

end value_of_squared_difference_l169_169180


namespace find_a_l169_169723

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem find_a
  (a : ℝ)
  (h₁ : ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f a x ≤ 4)
  (h₂ : ∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f a x = 4) :
  a = -3 ∨ a = 3 / 8 :=
by
  sorry

end find_a_l169_169723


namespace equation1_solution_equation2_solution_l169_169002

theorem equation1_solution (x : ℝ) (h : 2 * (x - 1) = 2 - 5 * (x + 2)) : x = -6 / 7 :=
sorry

theorem equation2_solution (x : ℝ) (h : (5 * x + 1) / 2 - (6 * x + 2) / 4 = 1) : x = 1 :=
sorry

end equation1_solution_equation2_solution_l169_169002


namespace cake_volume_icing_area_sum_l169_169129

-- Define the conditions based on the problem description
def cube_edge_length : ℕ := 4
def volume_of_piece := 16
def icing_area := 12

-- Define the statements to be proven
theorem cake_volume_icing_area_sum : 
  volume_of_piece + icing_area = 28 := 
sorry

end cake_volume_icing_area_sum_l169_169129


namespace binomial_7_2_eq_21_l169_169246

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l169_169246


namespace fixed_monthly_charge_l169_169453

variables (F C_J : ℝ)

-- Conditions
def january_bill := F + C_J = 46
def february_bill := F + 2 * C_J = 76

-- The proof goal
theorem fixed_monthly_charge
  (h_jan : january_bill F C_J)
  (h_feb : february_bill F C_J)
  (h_calls : C_J = 30) : F = 16 :=
by sorry

end fixed_monthly_charge_l169_169453


namespace diana_apollo_probability_l169_169095

theorem diana_apollo_probability :
  let outcomes := (6 * 6)
  let successful := (5 + 4 + 3 + 2 + 1)
  (successful / outcomes) = 5 / 12 := sorry

end diana_apollo_probability_l169_169095


namespace det_scaled_matrix_l169_169616

theorem det_scaled_matrix (a b c d : ℝ) (h : a * d - b * c = 5) : 
  (3 * a) * (3 * d) - (3 * b) * (3 * c) = 45 :=
by 
  sorry

end det_scaled_matrix_l169_169616


namespace calculate_F_l169_169239

def f(a : ℝ) : ℝ := a^2 - 5 * a + 6
def F(a b c : ℝ) : ℝ := b^2 + a * c + 1

theorem calculate_F : F 3 (f 3) (f 5) = 19 :=
by
  sorry

end calculate_F_l169_169239


namespace sara_quarters_l169_169281

theorem sara_quarters (initial_quarters : ℕ) (additional_quarters : ℕ) (total_quarters : ℕ) 
    (h1 : initial_quarters = 21) 
    (h2 : additional_quarters = 49) 
    (h3 : total_quarters = initial_quarters + additional_quarters) : 
    total_quarters = 70 :=
sorry

end sara_quarters_l169_169281


namespace initial_deposit_l169_169029

variable (P R : ℝ)

theorem initial_deposit (h1 : P + (P * R * 3) / 100 = 11200)
                       (h2 : P + (P * (R + 2) * 3) / 100 = 11680) :
  P = 8000 :=
by
  sorry

end initial_deposit_l169_169029


namespace sequence_divisibility_l169_169919

theorem sequence_divisibility (a b c : ℤ) (u v : ℕ → ℤ) (N : ℕ)
  (hu0 : u 0 = 1) (hu1 : u 1 = 1)
  (hu : ∀ n ≥ 2, u n = 2 * u (n - 1) - 3 * u (n - 2))
  (hv0 : v 0 = a) (hv1 : v 1 = b) (hv2 : v 2 = c)
  (hv : ∀ n ≥ 3, v n = v (n - 1) - 3 * v (n - 2) + 27 * v (n - 3))
  (hdiv : ∀ n ≥ N, u n ∣ v n) : 3 * a = 2 * b + c :=
by
  sorry

end sequence_divisibility_l169_169919


namespace morleys_theorem_l169_169893

def is_trisector (A B C : Point) (p : Point) : Prop :=
sorry -- Definition that this point p is on one of the trisectors of ∠BAC

def triangle (A B C : Point) : Prop :=
sorry -- Definition that points A, B, C form a triangle

def equilateral (A B C : Point) : Prop :=
sorry -- Definition that triangle ABC is equilateral

theorem morleys_theorem (A B C D E F : Point)
  (hABC : triangle A B C)
  (hD : is_trisector A B C D)
  (hE : is_trisector B C A E)
  (hF : is_trisector C A B F) :
  equilateral D E F :=
sorry

end morleys_theorem_l169_169893


namespace smallest_sum_97_l169_169494

theorem smallest_sum_97 (X Y Z W : ℕ) 
  (h1 : X + Y + Z = 3)
  (h2 : 4 * Z = 7 * Y)
  (h3 : 16 ∣ Y) : 
  X + Y + Z + W = 97 :=
by
  sorry

end smallest_sum_97_l169_169494


namespace smallest_n_inequality_l169_169747

theorem smallest_n_inequality:
  ∃ n : ℤ, (∀ x y z : ℝ, (x^2 + 2 * y^2 + z^2)^2 ≤ n * (x^4 + 3 * y^4 + z^4)) ∧ n = 4 :=
by
  sorry

end smallest_n_inequality_l169_169747


namespace inscribed_circle_radius_of_DEF_l169_169812

theorem inscribed_circle_radius_of_DEF (DE DF EF : ℝ) (r : ℝ)
  (hDE : DE = 8) (hDF : DF = 8) (hEF : EF = 10) :
  r = 5 * Real.sqrt 39 / 13 :=
by
  sorry

end inscribed_circle_radius_of_DEF_l169_169812


namespace line_tangent_to_parabola_l169_169389

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4 * x + 7 * y + k = 0 ↔ y^2 = 16 * x) → k = 49 :=
by
  sorry

end line_tangent_to_parabola_l169_169389


namespace sarah_flour_total_l169_169253

def rye_flour : ℕ := 5
def whole_wheat_bread_flour : ℕ := 10
def chickpea_flour : ℕ := 3
def whole_wheat_pastry_flour : ℕ := 2

def total_flour : ℕ := rye_flour + whole_wheat_bread_flour + chickpea_flour + whole_wheat_pastry_flour

theorem sarah_flour_total : total_flour = 20 := by
  sorry

end sarah_flour_total_l169_169253


namespace arithmetic_sequence_condition_l169_169823

theorem arithmetic_sequence_condition (a b c d : ℝ) :
  (∃ k : ℝ, b = a + k ∧ c = a + 2*k ∧ d = a + 3*k) ↔ (a + d = b + c) :=
sorry

end arithmetic_sequence_condition_l169_169823


namespace average_monthly_income_l169_169314

theorem average_monthly_income (P Q R : ℝ) (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250) (h3 : P = 4000) : (P + R) / 2 = 5200 := by
  sorry

end average_monthly_income_l169_169314


namespace line_equation_isosceles_triangle_l169_169359

theorem line_equation_isosceles_triangle 
  (x y : ℝ)
  (l : ℝ → ℝ → Prop)
  (h1 : l 3 2)
  (h2 : ∀ x y, l x y → (x = y ∨ x + y = 2 * intercept))
  (intercept : ℝ) :
  l x y ↔ (x - y = 1 ∨ x + y = 5) :=
by
  sorry

end line_equation_isosceles_triangle_l169_169359


namespace water_hydrogen_oxygen_ratio_l169_169643

/-- In a mixture of water with a total mass of 171 grams, 
    where 19 grams are hydrogen, the ratio of hydrogen to oxygen by mass is 1:8. -/
theorem water_hydrogen_oxygen_ratio 
  (h_total_mass : ℝ) 
  (h_mass : ℝ) 
  (o_mass : ℝ) 
  (h_condition : h_total_mass = 171) 
  (h_hydrogen_mass : h_mass = 19) 
  (h_oxygen_mass : o_mass = h_total_mass - h_mass) :
  h_mass / o_mass = 1 / 8 := 
by
  sorry

end water_hydrogen_oxygen_ratio_l169_169643


namespace distance_between_parallel_lines_l169_169552

theorem distance_between_parallel_lines (a d : ℝ) :
  (∀ x y : ℝ, 2 * x - y + 3 = 0 ∧ a * x - y + 4 = 0 → (2 = a ∧ d = |(3 - 4)| / Real.sqrt (2 ^ 2 + (-1) ^ 2))) → 
  (a = 2 ∧ d = Real.sqrt 5 / 5) :=
by 
  sorry

end distance_between_parallel_lines_l169_169552


namespace tory_video_games_l169_169483

theorem tory_video_games (T J: ℕ) :
    (3 * J + 5 = 11) → (J = T / 3) → T = 6 :=
by
  sorry

end tory_video_games_l169_169483


namespace no_real_sqrt_neg_six_pow_three_l169_169357

theorem no_real_sqrt_neg_six_pow_three : 
  ∀ x : ℝ, 
    (¬ ∃ y : ℝ, y * y = -6 ^ 3) :=
by
  sorry

end no_real_sqrt_neg_six_pow_three_l169_169357


namespace find_remainder_l169_169035

noncomputable def remainder_expr_division (β : ℂ) (hβ : β^4 + β^3 + β^2 + β + 1 = 0) : ℂ :=
  1 - β

theorem find_remainder (β : ℂ) (hβ : β^4 + β^3 + β^2 + β + 1 = 0) :
  ∃ r, (x^45 + x^34 + x^23 + x^12 + 1) % (x^4 + x^3 + x^2 + x + 1) = r ∧ r = remainder_expr_division β hβ :=
sorry

end find_remainder_l169_169035


namespace floating_time_l169_169048

theorem floating_time (boat_with_current: ℝ) (boat_against_current: ℝ) (distance: ℝ) (time: ℝ) : 
boat_with_current = 28 ∧ boat_against_current = 24 ∧ distance = 20 ∧ 
time = distance / ((boat_with_current - boat_against_current) / 2) → 
time = 10 := by
  sorry

end floating_time_l169_169048


namespace solution_of_inequality_l169_169715

-- Let us define the inequality and the solution set
def inequality (x : ℝ) := (x - 1)^2023 - 2^2023 * x^2023 ≤ x + 1
def solution_set (x : ℝ) := x ≥ -1

-- The theorem statement to prove that the solution set matches the inequality
theorem solution_of_inequality :
  {x : ℝ | inequality x} = {x : ℝ | solution_set x} := sorry

end solution_of_inequality_l169_169715


namespace fixed_point_l169_169703

theorem fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : (1, 4) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1) + 3)} :=
by
  sorry

end fixed_point_l169_169703


namespace cricket_team_members_l169_169964

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℕ := 27)
  (wk_age : ℕ := captain_age + 1)
  (total_avg_age : ℕ := 23)
  (remaining_avg_age : ℕ := total_avg_age - 1)
  (total_age : ℕ := n * total_avg_age)
  (captain_and_wk_age : ℕ := captain_age + wk_age)
  (remaining_age : ℕ := (n - 2) * remaining_avg_age) : n = 11 := 
by
  sorry

end cricket_team_members_l169_169964


namespace diff_of_squares_525_475_l169_169473

theorem diff_of_squares_525_475 : 525^2 - 475^2 = 50000 := by
  sorry

end diff_of_squares_525_475_l169_169473


namespace line_parallel_unique_a_l169_169038

theorem line_parallel_unique_a (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y + a + 3 = 0 → x + (a + 1)*y + 4 = 0) → a = -2 :=
  by
  sorry

end line_parallel_unique_a_l169_169038


namespace max_books_borrowed_l169_169895

theorem max_books_borrowed (total_students books_per_student : ℕ) (students_with_no_books: ℕ) (students_with_one_book students_with_two_books: ℕ) (rest_at_least_three_books students : ℕ) :
  total_students = 20 →
  books_per_student = 2 →
  students_with_no_books = 2 →
  students_with_one_book = 8 →
  students_with_two_books = 3 →
  rest_at_least_three_books = total_students - (students_with_no_books + students_with_one_book + students_with_two_books) →
  (students_with_no_books * 0 + students_with_one_book * 1 + students_with_two_books * 2 + students * books_per_student = total_students * books_per_student) →
  (students * 3 + some_student_max = 26) →
  some_student_max ≥ 8 :=
by
  introv h1 h2 h3 h4 h5 h6 h7
  sorry

end max_books_borrowed_l169_169895


namespace sum_abcd_l169_169118

variable (a b c d x : ℝ)

axiom eq1 : a + 2 = x
axiom eq2 : b + 3 = x
axiom eq3 : c + 4 = x
axiom eq4 : d + 5 = x
axiom eq5 : a + b + c + d + 10 = x

theorem sum_abcd : a + b + c + d = -26 / 3 :=
by
  -- We state the condition given in the problem
  sorry

end sum_abcd_l169_169118


namespace neither_necessary_nor_sufficient_l169_169561

theorem neither_necessary_nor_sufficient (x : ℝ) :
  ¬ ((-1 < x ∧ x < 2) → (|x - 2| < 1)) ∧ ¬ ((|x - 2| < 1) → (-1 < x ∧ x < 2)) :=
by
  sorry

end neither_necessary_nor_sufficient_l169_169561


namespace pure_milk_in_final_solution_l169_169151

noncomputable def final_quantity_of_milk (initial_milk : ℕ) (milk_removed_each_step : ℕ) (steps : ℕ) : ℝ :=
  let remaining_milk_step1 := initial_milk - milk_removed_each_step
  let proportion := (milk_removed_each_step : ℝ) / (initial_milk : ℝ)
  let milk_removed_step2 := proportion * remaining_milk_step1
  remaining_milk_step1 - milk_removed_step2

theorem pure_milk_in_final_solution :
  final_quantity_of_milk 30 9 2 = 14.7 :=
by
  sorry

end pure_milk_in_final_solution_l169_169151


namespace selling_price_percentage_l169_169280

  variable (L : ℝ)  -- List price
  variable (C : ℝ)  -- Cost price after discount
  variable (M : ℝ)  -- Marked price
  variable (S : ℝ)  -- Selling price after discount

  -- Conditions
  def cost_price_condition (L : ℝ) : ℝ := 0.7 * L
  def profit_condition (C S : ℝ) : Prop := 0.75 * S = C
  def marked_price_condition (S M : ℝ) : Prop := 0.85 * M = S

  theorem selling_price_percentage (L : ℝ) (h1 : C = cost_price_condition L)
    (h2 : profit_condition C S) (h3 : marked_price_condition S M) :
    S = 0.9333 * L :=
  by
    -- This is where the proof would go
    sorry
  
end selling_price_percentage_l169_169280


namespace cylinder_height_in_sphere_l169_169852

noncomputable def height_of_cylinder (r R : ℝ) : ℝ :=
  2 * Real.sqrt (R ^ 2 - r ^ 2)

theorem cylinder_height_in_sphere :
  height_of_cylinder 3 6 = 6 * Real.sqrt 3 :=
by
  sorry

end cylinder_height_in_sphere_l169_169852


namespace correct_calculation_l169_169815

variable {a : ℝ} (ha : a ≠ 0)

theorem correct_calculation (a : ℝ) (ha : a ≠ 0) : (a^2 * a^3 = a^5) :=
by sorry

end correct_calculation_l169_169815


namespace circle_tangent_radius_l169_169254

theorem circle_tangent_radius (k : ℝ) (r : ℝ) (hk : k > 4) 
  (h_tangent1 : dist (0, k) (x, x) = r)
  (h_tangent2 : dist (0, k) (x, -x) = r) 
  (h_tangent3 : dist (0, k) (x, 4) = r) : 
  r = 4 * Real.sqrt 2 := 
sorry

end circle_tangent_radius_l169_169254


namespace bicycle_distance_l169_169374

theorem bicycle_distance (P_b P_f : ℝ) (h1 : P_b = 9) (h2 : P_f = 7) (h3 : ∀ D : ℝ, D / P_f = D / P_b + 10) :
  315 = 315 :=
by
  sorry

end bicycle_distance_l169_169374


namespace sum_of_digits_l169_169375

theorem sum_of_digits :
  ∃ (a b : ℕ), (4 * 100 + a * 10 + 5) + 457 = (9 * 100 + b * 10 + 2) ∧
                (((9 + 2) - b) % 11 = 0) ∧
                (a + b = 4) :=
sorry

end sum_of_digits_l169_169375


namespace average_score_of_class_l169_169119

theorem average_score_of_class (total_students : ℕ)
  (perc_assigned_day perc_makeup_day : ℝ)
  (average_assigned_day average_makeup_day : ℝ)
  (h_total : total_students = 100)
  (h_perc_assigned_day : perc_assigned_day = 0.70)
  (h_perc_makeup_day : perc_makeup_day = 0.30)
  (h_average_assigned_day : average_assigned_day = 55)
  (h_average_makeup_day : average_makeup_day = 95) :
  ((perc_assigned_day * total_students * average_assigned_day + perc_makeup_day * total_students * average_makeup_day) / total_students) = 67 := by
  sorry

end average_score_of_class_l169_169119


namespace racket_price_l169_169653

theorem racket_price (cost_sneakers : ℕ) (cost_outfit : ℕ) (total_spent : ℕ) 
  (h_sneakers : cost_sneakers = 200) 
  (h_outfit : cost_outfit = 250) 
  (h_total : total_spent = 750) : 
  (total_spent - cost_sneakers - cost_outfit) = 300 :=
sorry

end racket_price_l169_169653


namespace Jia_age_is_24_l169_169636

variable (Jia Yi Bing Ding : ℕ)

theorem Jia_age_is_24
  (h1 : (Jia + Yi + Bing) / 3 = (Jia + Yi + Bing + Ding) / 4 + 1)
  (h2 : (Jia + Yi) / 2 = (Jia + Yi + Bing) / 3 + 1)
  (h3 : Jia = Yi + 4)
  (h4 : Ding = 17) :
  Jia = 24 :=
by
  sorry

end Jia_age_is_24_l169_169636


namespace problem_I_problem_II_l169_169693

-- Definitions
noncomputable def function_f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + (a - 2) * x - Real.log x

-- Problem (I)
theorem problem_I (a : ℝ) (h_min : ∀ x : ℝ, function_f a 1 ≤ function_f a x) :
  a = 1 ∧ (∀ x : ℝ, 0 < x ∧ x < 1 → (function_f a x < function_f a 1)) ∧ (∀ x : ℝ, x > 1 → (function_f a x > function_f a 1)) :=
sorry

-- Problem (II)
theorem problem_II (a x0 : ℝ) (h_a_gt_1 : a > 1) (h_x0_pos : 0 < x0) (h_x0_lt_1 : x0 < 1)
    (h_min : ∀ x : ℝ, function_f a (1/a) ≤ function_f a x) :
  ∀ x : ℝ, function_f a 0 > 0
:= sorry

end problem_I_problem_II_l169_169693


namespace rs_value_l169_169107

theorem rs_value (r s : ℝ) (hr : 0 < r) (hs: 0 < s) (h1 : r^2 + s^2 = 1) (h2 : r^4 + s^4 = 3 / 4) :
  r * s = Real.sqrt 2 / 4 :=
sorry

end rs_value_l169_169107


namespace solve_equation1_solve_equation2_l169_169123

theorem solve_equation1 (x : ℝ) : 3 * (x - 1)^3 = 24 ↔ x = 3 := by
  sorry

theorem solve_equation2 (x : ℝ) : (x - 3)^2 = 64 ↔ x = 11 ∨ x = -5 := by
  sorry

end solve_equation1_solve_equation2_l169_169123


namespace gcd_of_polynomials_l169_169087

theorem gcd_of_polynomials (b : ℤ) (k : ℤ) (hk : k % 2 = 0) (hb : b = 1187 * k) : 
  Int.gcd (2 * b^2 + 31 * b + 67) (b + 15) = 1 :=
by 
  sorry

end gcd_of_polynomials_l169_169087


namespace greatest_number_of_matching_pairs_l169_169758

theorem greatest_number_of_matching_pairs 
  (original_pairs : ℕ := 27)
  (lost_shoes : ℕ := 9) 
  (remaining_pairs : ℕ := original_pairs - (lost_shoes / 1))
  : remaining_pairs = 18 := by
  sorry

end greatest_number_of_matching_pairs_l169_169758


namespace integer_inequality_l169_169743

theorem integer_inequality (x y : ℤ) : x * (x + 1) ≠ 2 * (5 * y + 2) := 
  sorry

end integer_inequality_l169_169743


namespace smallest_number_l169_169210

theorem smallest_number (a b c d : ℝ) (h1 : a = 1) (h2 : b = -2) (h3 : c = 0) (h4 : d = -1/2) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by
  sorry

end smallest_number_l169_169210


namespace neither_sufficient_nor_necessary_l169_169879

theorem neither_sufficient_nor_necessary (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬ ((a > 0 ∧ b > 0) ↔ (ab < ((a + b) / 2)^2)) :=
sorry

end neither_sufficient_nor_necessary_l169_169879


namespace log_prime_factor_inequality_l169_169976

open Real

noncomputable def num_prime_factors (n : ℕ) : ℕ := sorry 

theorem log_prime_factor_inequality (n : ℕ) (h : 0 < n) : 
  log n ≥ num_prime_factors n * log 2 := 
sorry

end log_prime_factor_inequality_l169_169976


namespace compute_value_l169_169783

theorem compute_value
  (x y z : ℝ)
  (h1 : (xz / (x + y)) + (yx / (y + z)) + (zy / (z + x)) = -9)
  (h2 : (yz / (x + y)) + (zx / (y + z)) + (xy / (z + x)) = 15) :
  (y / (x + y)) + (z / (y + z)) + (x / (z + x)) = 13.5 :=
by
  sorry

end compute_value_l169_169783


namespace diff_of_squares_odd_divisible_by_8_l169_169188

theorem diff_of_squares_odd_divisible_by_8 (m n : ℤ) :
  ((2 * m + 1) ^ 2 - (2 * n + 1) ^ 2) % 8 = 0 :=
by 
  sorry

end diff_of_squares_odd_divisible_by_8_l169_169188


namespace loan_repayment_l169_169730

open Real

theorem loan_repayment
  (a r : ℝ) (h_r : 0 ≤ r) :
  ∃ x : ℝ, 
    x = (a * r * (1 + r)^5) / ((1 + r)^5 - 1) :=
sorry

end loan_repayment_l169_169730


namespace restore_to_original_without_limited_discount_restore_to_original_with_limited_discount_l169_169732

-- Let P be the original price of the jacket
variable (P : ℝ)

-- The price of the jacket after successive reductions
def price_after_discount (P : ℝ) : ℝ := 0.60 * P

-- The price of the jacket after all discounts including the limited-time offer
def price_after_full_discount (P : ℝ) : ℝ := 0.54 * P

-- Prove that to restore 0.60P back to P a 66.67% increase is needed
theorem restore_to_original_without_limited_discount :
  ∀ (P : ℝ), (P > 0) → (0.60 * P) * (1 + 66.67 / 100) = P :=
by sorry

-- Prove that to restore 0.54P back to P an 85.19% increase is needed
theorem restore_to_original_with_limited_discount :
  ∀ (P : ℝ), (P > 0) → (0.54 * P) * (1 + 85.19 / 100) = P :=
by sorry

end restore_to_original_without_limited_discount_restore_to_original_with_limited_discount_l169_169732


namespace pauline_spent_in_all_l169_169481

theorem pauline_spent_in_all
  (cost_taco_shells : ℝ := 5)
  (cost_bell_pepper : ℝ := 1.5)
  (num_bell_peppers : ℕ := 4)
  (cost_meat_per_pound : ℝ := 3)
  (num_pounds_meat : ℝ := 2) :
  (cost_taco_shells + num_bell_peppers * cost_bell_pepper + num_pounds_meat * cost_meat_per_pound = 17) :=
by
  sorry

end pauline_spent_in_all_l169_169481


namespace expected_socks_pairs_l169_169458

noncomputable def expected_socks (n : ℕ) : ℝ :=
2 * n

theorem expected_socks_pairs (n : ℕ) :
  @expected_socks n = 2 * n :=
by
  sorry

end expected_socks_pairs_l169_169458


namespace max_side_length_l169_169689

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l169_169689


namespace Luke_spent_money_l169_169460

theorem Luke_spent_money : ∀ (initial_money additional_money current_money x : ℕ),
  initial_money = 48 →
  additional_money = 21 →
  current_money = 58 →
  (initial_money + additional_money - current_money) = x →
  x = 11 :=
by
  intros initial_money additional_money current_money x h1 h2 h3 h4
  sorry

end Luke_spent_money_l169_169460


namespace hannah_total_spent_l169_169621

-- Definitions based on conditions
def sweatshirts_bought : ℕ := 3
def t_shirts_bought : ℕ := 2
def cost_per_sweatshirt : ℕ := 15
def cost_per_t_shirt : ℕ := 10

-- Definition of the theorem that needs to be proved
theorem hannah_total_spent : 
  (sweatshirts_bought * cost_per_sweatshirt + t_shirts_bought * cost_per_t_shirt) = 65 :=
by
  sorry

end hannah_total_spent_l169_169621


namespace no_real_roots_ffx_l169_169126

theorem no_real_roots_ffx 
  (b c : ℝ) 
  (h : ∀ x : ℝ, (x^2 + (b - 1) * x + (c - 1) ≠ 0 ∨ ∀x: ℝ, (b - 1)^2 - 4 * (c - 1) < 0)) 
  : ∀ x : ℝ, (x^2 + bx + c)^2 + b * (x^2 + bx + c) + c ≠ x :=
by
  sorry

end no_real_roots_ffx_l169_169126


namespace cos_150_deg_eq_neg_half_l169_169604

noncomputable def cos_of_angle (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem cos_150_deg_eq_neg_half :
  cos_of_angle 150 = -1/2 :=
by
  /-
    The conditions used directly in the problem include:
    - θ = 150 (Given angle)
  -/
  sorry

end cos_150_deg_eq_neg_half_l169_169604


namespace vacation_costs_l169_169001

theorem vacation_costs :
  let a := 15
  let b := 22.5
  let c := 22.5
  a + b + c = 45 → b - a = 7.5 := by
sorry

end vacation_costs_l169_169001


namespace total_amount_l169_169514

theorem total_amount (x y z : ℝ) 
  (hx : y = 0.45 * x) 
  (hz : z = 0.50 * x) 
  (hy_share : y = 63) : 
  x + y + z = 273 :=
by 
  sorry

end total_amount_l169_169514


namespace at_least_three_points_in_circle_l169_169770

noncomputable def point_in_circle (p : ℝ × ℝ) (c : ℝ × ℝ) (r : ℝ) : Prop :=
(dist p c) ≤ r

theorem at_least_three_points_in_circle (points : Fin 51 → (ℝ × ℝ)) (side_length : ℝ) (circle_radius : ℝ)
  (h_side_length : side_length = 1) (h_circle_radius : circle_radius = 1 / 7) : 
  ∃ (c : ℝ × ℝ), ∃ (p1 p2 p3 : Fin 51), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    point_in_circle (points p1) c circle_radius ∧ 
    point_in_circle (points p2) c circle_radius ∧ 
    point_in_circle (points p3) c circle_radius :=
sorry

end at_least_three_points_in_circle_l169_169770


namespace speed_of_boat_is_15_l169_169833

noncomputable def speed_of_boat_in_still_water (x : ℝ) : Prop :=
  ∃ (t : ℝ), t = 1 / 5 ∧ (x + 3) * t = 3.6 ∧ x = 15

theorem speed_of_boat_is_15 (x : ℝ) (t : ℝ) (rate_of_current : ℝ) (distance_downstream : ℝ) :
  rate_of_current = 3 →
  distance_downstream = 3.6 →
  t = 1 / 5 →
  (x + rate_of_current) * t = distance_downstream →
  x = 15 :=
by
  intros h1 h2 h3 h4
  -- proof goes here
  sorry

end speed_of_boat_is_15_l169_169833


namespace number_of_balls_sold_l169_169142

-- Let n be the number of balls sold
variable (n : ℕ)

-- The given conditions
def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 60
def loss := 5 * cost_price_per_ball

-- Prove that if the selling price of 'n' balls is Rs. 720 and 
-- the loss is equal to the cost price of 5 balls, then the 
-- number of balls sold (n) is 17.
theorem number_of_balls_sold (h1 : selling_price = 720) 
                             (h2 : cost_price_per_ball = 60) 
                             (h3 : loss = 5 * cost_price_per_ball) 
                             (hsale : n * cost_price_per_ball - selling_price = loss) : 
  n = 17 := 
by
  sorry

end number_of_balls_sold_l169_169142


namespace range_of_m_l169_169853

theorem range_of_m (m : ℝ) (h : 9 > m^2 ∧ m ≠ 0) : m ∈ Set.Ioo (-3) 0 ∨ m ∈ Set.Ioo 0 3 := 
sorry

end range_of_m_l169_169853


namespace hexagon_largest_angle_l169_169773

theorem hexagon_largest_angle (x : ℝ) 
    (h_sum : (x + 2) + (2*x + 4) + (3*x - 6) + (4*x + 8) + (5*x - 10) + (6*x + 12) = 720) :
    (6*x + 12) = 215 :=
by
  sorry

end hexagon_largest_angle_l169_169773


namespace marble_ratio_l169_169540

-- Definitions based on conditions
def dan_marbles : ℕ := 5
def mary_marbles : ℕ := 10

-- Statement of the theorem to prove the ratio is 2:1
theorem marble_ratio : mary_marbles / dan_marbles = 2 := by
  sorry

end marble_ratio_l169_169540


namespace coordinates_of_P_l169_169631

open Real

theorem coordinates_of_P (P : ℝ × ℝ) (h1 : P.1 = 2 * cos (2 * π / 3)) (h2 : P.2 = 2 * sin (2 * π / 3)) :
  P = (-1, sqrt 3) :=
by
  sorry

end coordinates_of_P_l169_169631


namespace expand_product_l169_169102

noncomputable def a (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
noncomputable def b (x : ℝ) : ℝ := x^2 + x + 3

theorem expand_product (x : ℝ) : (a x) * (b x) = 2 * x^4 - x^3 + 4 * x^2 - 8 * x + 3 :=
by
  sorry

end expand_product_l169_169102


namespace AgathaAdditionalAccessories_l169_169044

def AgathaBudget : ℕ := 250
def Frame : ℕ := 85
def FrontWheel : ℕ := 35
def RearWheel : ℕ := 40
def Seat : ℕ := 25
def HandlebarTape : ℕ := 15
def WaterBottleCage : ℕ := 10
def BikeLock : ℕ := 20
def FutureExpenses : ℕ := 10

theorem AgathaAdditionalAccessories :
  AgathaBudget - (Frame + FrontWheel + RearWheel + Seat + HandlebarTape + WaterBottleCage + BikeLock + FutureExpenses) = 10 := by
  sorry

end AgathaAdditionalAccessories_l169_169044


namespace probability_at_least_one_bean_distribution_of_X_expectation_of_X_l169_169967

noncomputable def total_ways := Nat.choose 6 3
noncomputable def ways_select_2_egg_1_bean := (Nat.choose 4 2) * (Nat.choose 2 1)
noncomputable def ways_select_1_egg_2_bean := (Nat.choose 4 1) * (Nat.choose 2 2)
noncomputable def at_least_one_bean_probability := (ways_select_2_egg_1_bean + ways_select_1_egg_2_bean) / total_ways

theorem probability_at_least_one_bean : at_least_one_bean_probability = 4 / 5 :=
by sorry

noncomputable def p_X_eq_0 := (Nat.choose 4 3) / total_ways
noncomputable def p_X_eq_1 := ways_select_2_egg_1_bean / total_ways
noncomputable def p_X_eq_2 := ways_select_1_egg_2_bean / total_ways

theorem distribution_of_X : p_X_eq_0 = 1 / 5 ∧ p_X_eq_1 = 3 / 5 ∧ p_X_eq_2 = 1 / 5 :=
by sorry

noncomputable def E_X := (0 * p_X_eq_0) + (1 * p_X_eq_1) + (2 * p_X_eq_2)

theorem expectation_of_X : E_X = 1 :=
by sorry

end probability_at_least_one_bean_distribution_of_X_expectation_of_X_l169_169967


namespace contestant_score_l169_169050

theorem contestant_score (highest_score lowest_score : ℕ) (average_score : ℕ)
  (h_hs : highest_score = 86)
  (h_ls : lowest_score = 45)
  (h_avg : average_score = 76) :
  (76 * 9 - 86 - 45) / 7 = 79 := 
by 
  sorry

end contestant_score_l169_169050


namespace min_radius_circle_line_intersection_l169_169010

theorem min_radius_circle_line_intersection (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) (r : ℝ) (hr : r > 0)
    (intersect : ∃ (x y : ℝ), (x - Real.cos θ)^2 + (y - Real.sin θ)^2 = r^2 ∧ 2 * x - y - 10 = 0) :
    r ≥ 2 * Real.sqrt 5 - 1 :=
  sorry

end min_radius_circle_line_intersection_l169_169010


namespace seventeen_number_selection_l169_169555

theorem seventeen_number_selection : ∃ (n : ℕ), (∀ s : Finset ℕ, (s ⊆ Finset.range 17) → (Finset.card s = n) → ∃ x y : ℕ, (x ∈ s) ∧ (y ∈ s) ∧ (x ≠ y) ∧ (x = 3 * y ∨ y = 3 * x)) ∧ (n = 13) :=
by
  sorry

end seventeen_number_selection_l169_169555


namespace trajectory_of_midpoint_l169_169128

-- Definitions based on the conditions identified in the problem
variables {x y x1 y1 : ℝ}

-- Condition that point P is on the curve y = 2x^2 + 1
def point_on_curve (x1 y1 : ℝ) : Prop :=
  y1 = 2 * x1^2 + 1

-- Definition of the midpoint M conditions
def midpoint_def (x y x1 y1 : ℝ) : Prop :=
  x = (x1 + 0) / 2 ∧ y = (y1 - 1) / 2

-- Final theorem statement to be proved
theorem trajectory_of_midpoint (x y x1 y1 : ℝ) :
  point_on_curve x1 y1 → midpoint_def x y x1 y1 → y = 4 * x^2 :=
sorry

end trajectory_of_midpoint_l169_169128


namespace toy_robot_shipment_l169_169973

-- Define the conditions provided in the problem
def thirty_percent_displayed (total: ℕ) : ℕ := (3 * total) / 10
def seventy_percent_stored (total: ℕ) : ℕ := (7 * total) / 10

-- The main statement to prove: if 70% of the toy robots equal 140, then the total number of toy robots is 200
theorem toy_robot_shipment (total : ℕ) (h : seventy_percent_stored total = 140) : total = 200 :=
by
  -- We will fill in the proof here
  sorry

end toy_robot_shipment_l169_169973


namespace color_swap_rectangle_l169_169296

theorem color_swap_rectangle 
  (n : ℕ) 
  (square_size : ℕ := 2*n - 1) 
  (colors : Finset ℕ := Finset.range n) 
  (vertex_colors : Fin (square_size + 1) × Fin (square_size + 1) → ℕ) 
  (h_vertex_colors : ∀ v, vertex_colors v ∈ colors) :
  ∃ row, ∃ (v₁ v₂ : Fin (square_size + 1) × Fin (square_size + 1)),
    (v₁.1 = row ∧ v₂.1 = row ∧ v₁ ≠ v₂ ∧
    (∃ r₀ r₁ r₂, r₀ ≠ r₁ ∧ r₁ ≠ r₂ ∧ r₂ ≠ r₀ ∧
    vertex_colors v₁ = vertex_colors (r₀, v₁.2) ∧
    vertex_colors v₂ = vertex_colors (r₀, v₂.2) ∧
    vertex_colors (r₁, v₁.2) = vertex_colors (r₂, v₂.2))) := 
sorry

end color_swap_rectangle_l169_169296


namespace clocks_resynchronize_after_days_l169_169092

/-- Arthur's clock gains 15 minutes per day. -/
def arthurs_clock_gain_per_day : ℕ := 15

/-- Oleg's clock gains 12 minutes per day. -/
def olegs_clock_gain_per_day : ℕ := 12

/-- The clocks display time in a 12-hour format, which is equivalent to 720 minutes. -/
def twelve_hour_format_in_minutes : ℕ := 720

/-- 
  After how many days will this situation first repeat given the 
  conditions of gain in Arthur's and Oleg's clocks and the 12-hour format.
-/
theorem clocks_resynchronize_after_days :
  ∃ (N : ℕ), N * arthurs_clock_gain_per_day % twelve_hour_format_in_minutes = 0 ∧
             N * olegs_clock_gain_per_day % twelve_hour_format_in_minutes = 0 ∧
             N = 240 :=
by
  sorry

end clocks_resynchronize_after_days_l169_169092


namespace price_of_battery_l169_169624

def cost_of_tire : ℕ := 42
def cost_of_tires (num_tires : ℕ) : ℕ := num_tires * cost_of_tire
def total_cost : ℕ := 224
def num_tires : ℕ := 4
def cost_of_battery : ℕ := total_cost - cost_of_tires num_tires

theorem price_of_battery : cost_of_battery = 56 := by
  sorry

end price_of_battery_l169_169624


namespace combined_population_of_New_England_and_New_York_l169_169901

noncomputable def population_of_New_England : ℕ := 2100000

noncomputable def population_of_New_York := (2/3 : ℚ) * population_of_New_England

theorem combined_population_of_New_England_and_New_York :
  population_of_New_England + population_of_New_York = 3500000 :=
by sorry

end combined_population_of_New_England_and_New_York_l169_169901


namespace bees_count_on_fifth_day_l169_169158

theorem bees_count_on_fifth_day
  (initial_count : ℕ) (h_initial : initial_count = 1)
  (growth_factor : ℕ) (h_growth : growth_factor = 3) :
  let bees_at_day (n : ℕ) : ℕ := initial_count * (growth_factor + 1) ^ n
  bees_at_day 5 = 1024 := 
by {
  sorry
}

end bees_count_on_fifth_day_l169_169158


namespace fish_per_bowl_l169_169709

theorem fish_per_bowl : 6003 / 261 = 23 := by
  sorry

end fish_per_bowl_l169_169709


namespace feasible_stations_l169_169644

theorem feasible_stations (n : ℕ) (h: n > 0) 
  (pairings : ∀ (i j : ℕ), i ≠ j → i < n → j < n → ∃ k, (i+k) % n = j ∨ (j+k) % n = i) : n = 4 :=
sorry

end feasible_stations_l169_169644


namespace observer_height_proof_l169_169179

noncomputable def height_observer (d m α β : ℝ) : ℝ :=
  let cot_alpha := 1 / Real.tan α
  let cot_beta := 1 / Real.tan β
  let u := (d * (m * cot_beta - d)) / (2 * d - m * (cot_beta - cot_alpha))
  20 + Real.sqrt (400 + u * m * cot_alpha - u^2)

theorem observer_height_proof :
  height_observer 290 40 (11.4 * Real.pi / 180) (4.7 * Real.pi / 180) = 52 := sorry

end observer_height_proof_l169_169179


namespace rope_length_loss_l169_169694

theorem rope_length_loss
  (stories_needed : ℕ)
  (feet_per_story : ℕ)
  (pieces_of_rope : ℕ)
  (feet_per_rope : ℕ)
  (total_feet_needed : ℕ)
  (total_feet_bought : ℕ)
  (percentage_lost : ℕ) :
  
  stories_needed = 6 →
  feet_per_story = 10 →
  pieces_of_rope = 4 →
  feet_per_rope = 20 →
  total_feet_needed = stories_needed * feet_per_story →
  total_feet_bought = pieces_of_rope * feet_per_rope →
  total_feet_needed <= total_feet_bought →
  percentage_lost = ((total_feet_bought - total_feet_needed) * 100) / total_feet_bought →
  percentage_lost = 25 :=
by
  intros h_stories h_feet_story h_pieces h_feet_rope h_total_needed h_total_bought h_needed_bought h_percentage
  sorry

end rope_length_loss_l169_169694


namespace systematic_sampling_selects_616_l169_169997

theorem systematic_sampling_selects_616 (n : ℕ) (h₁ : n = 1000) (h₂ : (∀ i : ℕ, ∃ j : ℕ, i = 46 + j * 10) → True) :
  (∃ m : ℕ, m = 616) :=
  by
  sorry

end systematic_sampling_selects_616_l169_169997


namespace music_player_winner_l169_169042

theorem music_player_winner (n : ℕ) (h1 : ∀ k, k % n = 0 → k = 35) (h2 : 35 % 7 = 0) (h3 : 35 % n = 0) (h4 : n ≠ 1) (h5 : n ≠ 7) (h6 : n ≠ 35) : n = 5 := 
sorry

end music_player_winner_l169_169042


namespace necessary_condition_for_x_gt_5_l169_169330

theorem necessary_condition_for_x_gt_5 (x : ℝ) : x > 5 → x > 3 :=
by
  intros h
  exact lt_trans (show 3 < 5 from by linarith) h

end necessary_condition_for_x_gt_5_l169_169330


namespace cans_for_credit_l169_169482

theorem cans_for_credit (P C R : ℕ) : 
  (3 * P = 2 * C) → (C ≠ 0) → (R ≠ 0) → P * R / C = (P * R / C : ℕ) :=
by
  intros h1 h2 h3
  -- proof required here
  sorry

end cans_for_credit_l169_169482


namespace value_of_stamp_collection_l169_169796

theorem value_of_stamp_collection 
  (n m : ℕ) (v_m : ℝ)
  (hn : n = 18) 
  (hm : m = 6)
  (hv_m : v_m = 15)
  (uniform_value : ∀ (k : ℕ), k ≤ m → v_m / m = v_m / k):
  ∃ v_total : ℝ, v_total = 45 :=
by 
  sorry

end value_of_stamp_collection_l169_169796


namespace intersecting_lines_l169_169000

theorem intersecting_lines (p q r s t : ℝ) : (∃ u v : ℝ, p * u^2 + q * v^2 + r * u + s * v + t = 0) →
  ( ∃ p q : ℝ, p * q < 0 ∧ 4 * t = r^2 / p + s^2 / q ) :=
sorry

end intersecting_lines_l169_169000


namespace sampling_is_simple_random_l169_169114

-- Definitions based on conditions
def total_students := 200
def students_sampled := 20
def sampling_method := "Simple Random Sampling"

-- The problem: given the random sampling of 20 students from 200, prove that the method is simple random sampling.
theorem sampling_is_simple_random :
  (total_students = 200 ∧ students_sampled = 20) → sampling_method = "Simple Random Sampling" := 
by
  sorry

end sampling_is_simple_random_l169_169114


namespace expression_negativity_l169_169201

-- Given conditions: a, b, and c are lengths of the sides of a triangle
variables (a b c : ℝ)
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : b + c > a
axiom triangle_inequality3 : c + a > b

-- To prove: (a - b)^2 - c^2 < 0
theorem expression_negativity (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  (a - b)^2 - c^2 < 0 :=
sorry

end expression_negativity_l169_169201


namespace derivative_at_pi_div_2_l169_169308

noncomputable def f (x : ℝ) : ℝ := x * Real.sin (2 * x)

theorem derivative_at_pi_div_2 : deriv f (Real.pi / 2) = -Real.pi := by
  sorry

end derivative_at_pi_div_2_l169_169308


namespace cats_left_l169_169867

theorem cats_left (siamese house sold : ℕ) (h1 : siamese = 12) (h2 : house = 20) (h3 : sold = 20) :  
  (siamese + house) - sold = 12 := 
by
  sorry

end cats_left_l169_169867


namespace shaded_area_percentage_l169_169593

def area_square (side : ℕ) : ℕ := side * side

def shaded_percentage (total_area shaded_area : ℕ) : ℚ :=
  ((shaded_area : ℚ) / total_area) * 100 

theorem shaded_area_percentage (side : ℕ) (total_area : ℕ) (shaded_area : ℕ) 
  (h_side : side = 7) (h_total_area : total_area = area_square side) 
  (h_shaded_area : shaded_area = 4 + 16 + 13) : 
  shaded_percentage total_area shaded_area = 3300 / 49 :=
by
  -- The proof will go here
  sorry

end shaded_area_percentage_l169_169593


namespace annual_increase_in_living_space_l169_169006

-- Definitions based on conditions
def population_2000 : ℕ := 200000
def living_space_2000_per_person : ℝ := 8
def target_living_space_2004_per_person : ℝ := 10
def annual_growth_rate : ℝ := 0.01
def years : ℕ := 4

-- Goal stated as a theorem
theorem annual_increase_in_living_space :
  let final_population := population_2000 * (1 + annual_growth_rate)^years
  let total_living_space_2004 := target_living_space_2004_per_person * final_population
  let initial_living_space := living_space_2000_per_person * population_2000
  let total_additional_space := total_living_space_2004 - initial_living_space
  let average_annual_increase := total_additional_space / years
  average_annual_increase = 120500.0 :=
sorry

end annual_increase_in_living_space_l169_169006


namespace find_integer_l169_169190

theorem find_integer
  (x y : ℤ)
  (h1 : 4 * x + y = 34)
  (h2 : 2 * x - y = 20)
  (h3 : y^2 = 4) :
  y = -2 :=
by
  sorry

end find_integer_l169_169190


namespace find_f_3_l169_169140

def quadratic_function (b c : ℝ) (x : ℝ) : ℝ :=
- x^2 + b * x + c

theorem find_f_3 (b c : ℝ) (h1 : quadratic_function b c 2 + quadratic_function b c 4 = 12138)
                       (h2 : 3*b + c = 6079) :
  quadratic_function b c 3 = 6070 := 
by
  sorry

end find_f_3_l169_169140


namespace joan_has_6_balloons_l169_169136

theorem joan_has_6_balloons (initial_balloons : ℕ) (lost_balloons : ℕ) (h1 : initial_balloons = 8) (h2 : lost_balloons = 2) : initial_balloons - lost_balloons = 6 :=
sorry

end joan_has_6_balloons_l169_169136


namespace log_sum_eq_five_l169_169704

variable {a : ℕ → ℝ}

-- Conditions from the problem
def geometric_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 3 * a n 

def sum_condition (a : ℕ → ℝ) : Prop :=
a 2 + a 4 + a 9 = 9

-- The mathematical statement to prove
theorem log_sum_eq_five (h1 : geometric_seq a) (h2 : sum_condition a) :
  Real.logb 3 (a 5 + a 7 + a 9) = 5 := 
sorry

end log_sum_eq_five_l169_169704


namespace find_number_of_small_gardens_l169_169231

-- Define the conditions
def seeds_total : Nat := 52
def seeds_big_garden : Nat := 28
def seeds_per_small_garden : Nat := 4

-- Define the target value
def num_small_gardens : Nat := 6

-- The statement of the proof problem
theorem find_number_of_small_gardens 
  (H1 : seeds_total = 52) 
  (H2 : seeds_big_garden = 28) 
  (H3 : seeds_per_small_garden = 4) 
  : seeds_total - seeds_big_garden = 24 ∧ (seeds_total - seeds_big_garden) / seeds_per_small_garden = num_small_gardens := 
sorry

end find_number_of_small_gardens_l169_169231


namespace rectangle_height_l169_169209

theorem rectangle_height (y : ℝ) (h_pos : 0 < y) 
  (h_area : let length := 5 - (-3)
            let height := y - (-2)
            length * height = 112) : y = 12 := 
by 
  -- The proof is omitted
  sorry

end rectangle_height_l169_169209


namespace linear_coefficient_l169_169962

theorem linear_coefficient (m x : ℝ) (h1 : (m - 3) * x ^ (m^2 - 2 * m - 1) - m * x + 6 = 0) (h2 : (m^2 - 2 * m - 1 = 2)) (h3 : m ≠ 3) : 
  ∃ a b c : ℝ, a * x ^ 2 + b * x + c = 0 ∧ b = 1 :=
by
  sorry

end linear_coefficient_l169_169962


namespace unit_digit_div_l169_169784

theorem unit_digit_div (n : ℕ) : (33 * 10) % (2 ^ 1984) = n % 10 :=
by
  have h := 2 ^ 1984
  have u_digit_2_1984 := 6 -- Since 1984 % 4 = 0, last digit in the cycle of 2^n for n ≡ 0 [4] is 6
  sorry
  
example : (33 * 10) / (2 ^ 1984) % 10 = 6 :=
by sorry

end unit_digit_div_l169_169784


namespace sally_investment_l169_169542

theorem sally_investment (m : ℝ) (hmf : 0 ≤ m) 
  (total_investment : m + 7 * m = 200000) : 
  7 * m = 175000 :=
by
  -- Proof goes here
  sorry

end sally_investment_l169_169542


namespace complete_square_l169_169186

-- Definitions based on conditions
def row_sum_piece2 := 2 + 1 + 3 + 1
def total_sum_square := 4 * row_sum_piece2
def sum_piece1 := 7
def sum_piece2 := 8
def sum_piece3 := 8
def total_given_pieces := sum_piece1 + sum_piece2 + sum_piece3
def sum_missing_piece := total_sum_square - total_given_pieces

-- Statement to prove that the missing piece has the correct sum
theorem complete_square : (sum_missing_piece = 5) :=
by 
  -- It is a placeholder for the proof steps, the actual proof steps are not needed
  sorry

end complete_square_l169_169186


namespace range_of_omega_l169_169837

theorem range_of_omega (ω : ℝ) (h_pos : ω > 0) (h_three_high_points : (9 * π / 2) ≤ ω + π / 4 ∧ ω + π / 4 < 6 * π + π / 2) : 
           (17 * π / 4) ≤ ω ∧ ω < (25 * π / 4) :=
  sorry

end range_of_omega_l169_169837


namespace number_of_valid_subsets_l169_169467

def setA : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def oddSet : Finset ℕ := {1, 3, 5, 7}
def evenSet : Finset ℕ := {2, 4, 6}

theorem number_of_valid_subsets : 
  (oddSet.powerset.card * (evenSet.powerset.card - 1) - oddSet.powerset.card) = 96 :=
by sorry

end number_of_valid_subsets_l169_169467


namespace evaluate_expression_l169_169103

theorem evaluate_expression : (527 * 527 - 526 * 528) = 1 := by
  sorry

end evaluate_expression_l169_169103


namespace rahul_savings_is_correct_l169_169034

def Rahul_Savings_Problem : Prop :=
  ∃ (NSC PPF : ℝ), 
    (1/3) * NSC = (1/2) * PPF ∧ 
    NSC + PPF = 180000 ∧ 
    PPF = 72000

theorem rahul_savings_is_correct : Rahul_Savings_Problem :=
  sorry

end rahul_savings_is_correct_l169_169034


namespace number_of_1989_periodic_points_l169_169554

noncomputable def f (z : ℂ) (m : ℕ) : ℂ := z ^ m

noncomputable def is_periodic_point (z : ℂ) (f : ℂ → ℂ) (n : ℕ) : Prop :=
f^[n] z = z ∧ ∀ k : ℕ, k < n → (f^[k] z) ≠ z

noncomputable def count_periodic_points (m n : ℕ) : ℕ :=
m^n - m^(n / 3) - m^(n / 13) - m^(n / 17) + m^(n / 39) + m^(n / 51) + m^(n / 117) - m^(n / 153)

theorem number_of_1989_periodic_points (m : ℕ) (hm : 1 < m) :
  count_periodic_points m 1989 = m^1989 - m^663 - m^153 - m^117 + m^51 + m^39 + m^9 - m^3 :=
sorry

end number_of_1989_periodic_points_l169_169554


namespace complex_inv_condition_l169_169667

theorem complex_inv_condition (i : ℂ) (h : i^2 = -1) : (i - 2 * i⁻¹)⁻¹ = -i / 3 :=
by
  sorry

end complex_inv_condition_l169_169667


namespace no_int_solutions_for_quadratics_l169_169995

theorem no_int_solutions_for_quadratics :
  ¬ ∃ a b c : ℤ, (∃ x1 x2 : ℤ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) ∧
                (∃ y1 y2 : ℤ, (a + 1) * y1^2 + (b + 1) * y1 + (c + 1) = 0 ∧ 
                              (a + 1) * y2^2 + (b + 1) * y2 + (c + 1) = 0) :=
by
  sorry

end no_int_solutions_for_quadratics_l169_169995


namespace sqrt_12_minus_sqrt_27_l169_169216

theorem sqrt_12_minus_sqrt_27 :
  (Real.sqrt 12 - Real.sqrt 27 = -Real.sqrt 3) := by
  sorry

end sqrt_12_minus_sqrt_27_l169_169216


namespace ellipse_equation_x_axis_ellipse_equation_y_axis_parabola_equation_x_axis_parabola_equation_y_axis_l169_169368

-- Define the conditions for the ellipse problem
def major_axis_length : ℝ := 10
def focal_length : ℝ := 4

-- Define the conditions for the parabola problem
def point_P : ℝ × ℝ := (-2, -4)

-- The equations to be proven
theorem ellipse_equation_x_axis :
  2 * (5 : ℝ) = major_axis_length ∧ 2 * (2 : ℝ) = focal_length →
  (5 : ℝ)^2 - (2 : ℝ)^2 = 21 →
  (∀ x y : ℝ, x^2 / 25 + y^2 / 21 = 1) := sorry

theorem ellipse_equation_y_axis :
  2 * (5 : ℝ) = major_axis_length ∧ 2 * (2 : ℝ) = focal_length →
  (5 : ℝ)^2 - (2 : ℝ)^2 = 21 →
  (∀ x y : ℝ, y^2 / 25 + x^2 / 21 = 1) := sorry

theorem parabola_equation_x_axis :
  point_P = (-2, -4) →
  (∀ x y : ℝ, y^2 = -8 * x) := sorry

theorem parabola_equation_y_axis :
  point_P = (-2, -4) →
  (∀ x y : ℝ, x^2 = -y) := sorry

end ellipse_equation_x_axis_ellipse_equation_y_axis_parabola_equation_x_axis_parabola_equation_y_axis_l169_169368


namespace probability_one_head_two_tails_l169_169430

-- Define an enumeration for Coin with two possible outcomes: heads and tails.
inductive Coin
| heads
| tails

-- Function to count the number of heads in a list of Coin.
def countHeads : List Coin → Nat
| [] => 0
| Coin.heads :: xs => 1 + countHeads xs
| Coin.tails :: xs => countHeads xs

-- Function to calculate the probability of a specific event given the total outcomes.
def probability (specific_events total_outcomes : Nat) : Rat :=
  (specific_events : Rat) / (total_outcomes : Rat)

-- The main theorem
theorem probability_one_head_two_tails : probability 3 8 = (3 / 8 : Rat) :=
sorry

end probability_one_head_two_tails_l169_169430


namespace solve_problem_l169_169447

noncomputable def problem_statement : Prop :=
  ∀ (T0 Ta T t1 T1 h t2 T2 : ℝ),
    T0 = 88 ∧ Ta = 24 ∧ T1 = 40 ∧ t1 = 20 ∧
    T1 - Ta = (T0 - Ta) * ((1/2)^(t1/h)) ∧
    T2 = 32 ∧ T2 - Ta = (T1 - Ta) * ((1/2)^(t2/h)) →
    t2 = 10

theorem solve_problem : problem_statement := sorry

end solve_problem_l169_169447


namespace carnival_rent_l169_169360

-- Define the daily popcorn earnings
def daily_popcorn : ℝ := 50
-- Define the multiplier for cotton candy earnings
def multiplier : ℝ := 3
-- Define the number of operational days
def days : ℕ := 5
-- Define the cost of ingredients
def ingredients_cost : ℝ := 75
-- Define the net earnings after expenses
def net_earnings : ℝ := 895
-- Define the total earnings from selling popcorn for all days
def total_popcorn_earnings : ℝ := daily_popcorn * days
-- Define the total earnings from selling cotton candy for all days
def total_cottoncandy_earnings : ℝ := (daily_popcorn * multiplier) * days
-- Define the total earnings before expenses
def total_earnings : ℝ := total_popcorn_earnings + total_cottoncandy_earnings
-- Define the amount remaining after paying the rent (which includes net earnings and ingredient cost)
def remaining_after_rent : ℝ := net_earnings + ingredients_cost
-- Define the rent
def rent : ℝ := total_earnings - remaining_after_rent

theorem carnival_rent : rent = 30 := by
  sorry

end carnival_rent_l169_169360


namespace available_spaces_l169_169086

noncomputable def numberOfBenches : ℕ := 50
noncomputable def capacityPerBench : ℕ := 4
noncomputable def peopleSeated : ℕ := 80

theorem available_spaces :
  let totalCapacity := numberOfBenches * capacityPerBench;
  let availableSpaces := totalCapacity - peopleSeated;
  availableSpaces = 120 := by
    sorry

end available_spaces_l169_169086


namespace parabola_passing_through_4_neg2_l169_169983

theorem parabola_passing_through_4_neg2 :
  (∃ p : ℝ, y^2 = 2 * p * x ∧ y = -2 ∧ x = 4 ∧ (y^2 = x)) ∨
  (∃ p : ℝ, x^2 = -2 * p * y ∧ y = -2 ∧ x = 4 ∧ (x^2 = -8 * y)) :=
by
  sorry

end parabola_passing_through_4_neg2_l169_169983


namespace gabby_l169_169516

-- Define variables and conditions
variables (watermelons peaches plums total_fruit : ℕ)
variables (h_watermelons : watermelons = 1)
variables (h_peaches : peaches = watermelons + 12)
variables (h_plums : plums = 3 * peaches)
variables (h_total_fruit : total_fruit = watermelons + peaches + plums)

-- The theorem we aim to prove
theorem gabby's_fruit_count (h_watermelons : watermelons = 1)
                           (h_peaches : peaches = watermelons + 12)
                           (h_plums : plums = 3 * peaches)
                           (h_total_fruit : total_fruit = watermelons + peaches + plums) :
  total_fruit = 53 := by
sorry

end gabby_l169_169516


namespace num_nat_numbers_divisible_by_7_between_100_and_250_l169_169896

noncomputable def countNatNumbersDivisibleBy7InRange : ℕ :=
  let smallest := Nat.ceil (100 / 7) * 7
  let largest := Nat.floor (250 / 7) * 7
  (largest - smallest) / 7 + 1

theorem num_nat_numbers_divisible_by_7_between_100_and_250 :
  countNatNumbersDivisibleBy7InRange = 21 :=
by
  -- Placeholder for the proof steps
  sorry

end num_nat_numbers_divisible_by_7_between_100_and_250_l169_169896


namespace greatest_possible_value_of_x_l169_169574

theorem greatest_possible_value_of_x : 
  (∀ x : ℚ, ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20) → 
  x ≤ 9/5 := sorry

end greatest_possible_value_of_x_l169_169574


namespace simplify_fraction_expression_l169_169942

theorem simplify_fraction_expression :
  5 * (12 / 7) * (49 / (-60)) = -7 := 
sorry

end simplify_fraction_expression_l169_169942


namespace kanul_total_amount_l169_169419

theorem kanul_total_amount (T : ℝ) (h1 : 35000 + 40000 + 0.2 * T = T) : T = 93750 := 
by
  sorry

end kanul_total_amount_l169_169419


namespace other_root_of_quadratic_l169_169278

theorem other_root_of_quadratic (a b : ℝ) (h : (1:ℝ) = 1) (h_root : (1:ℝ) ^ 2 + a * (1:ℝ) + 2 = 0): b = 2 :=
by
  sorry

end other_root_of_quadratic_l169_169278


namespace find_l_in_triangle_l169_169442

/-- In triangle XYZ, if XY = 5, YZ = 12, XZ = 13, and YM is the angle bisector from vertex Y with YM = l * sqrt 2, then l equals 60/17. -/
theorem find_l_in_triangle (XY YZ XZ : ℝ) (YM l : ℝ) (hXY : XY = 5) (hYZ : YZ = 12) (hXZ : XZ = 13) (hYM : YM = l * Real.sqrt 2) : 
    l = 60 / 17 :=
sorry

end find_l_in_triangle_l169_169442


namespace James_bought_3_CDs_l169_169226

theorem James_bought_3_CDs :
  ∃ (cd1 cd2 cd3 : ℝ), cd1 = 1.5 ∧ cd2 = 1.5 ∧ cd3 = 2 * cd1 ∧ cd1 + cd2 + cd3 = 6 ∧ 3 = 3 :=
by
  sorry

end James_bought_3_CDs_l169_169226


namespace expand_and_solve_solve_quadratic_l169_169153

theorem expand_and_solve (x : ℝ) :
  6 * (x - 3) * (x + 5) = 6 * x^2 + 12 * x - 90 :=
by sorry

theorem solve_quadratic (x : ℝ) :
  6 * x^2 + 12 * x - 90 = 0 ↔ x = -5 ∨ x = 3 :=
by sorry

end expand_and_solve_solve_quadratic_l169_169153


namespace fixed_point_quadratic_l169_169506

theorem fixed_point_quadratic : 
  (∀ m : ℝ, 3 * a ^ 2 - m * a + 2 * m + 1 = b) → (a = 2 ∧ b = 13) := 
by sorry

end fixed_point_quadratic_l169_169506


namespace marco_score_percentage_less_l169_169156

theorem marco_score_percentage_less
  (average_score : ℕ)
  (margaret_score : ℕ)
  (margaret_more_than_marco : ℕ)
  (h1 : average_score = 90)
  (h2 : margaret_score = 86)
  (h3 : margaret_more_than_marco = 5) :
  (average_score - (margaret_score - margaret_more_than_marco)) * 100 / average_score = 10 :=
by
  sorry

end marco_score_percentage_less_l169_169156


namespace sum_of_three_integers_l169_169501

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 5^3) : a + b + c = 31 := by
  sorry

end sum_of_three_integers_l169_169501


namespace incorrect_option_B_l169_169066

-- Definitions of the given conditions
def optionA (a : ℝ) : Prop := (8 * a = 8 * a)
def optionB (a : ℝ) : Prop := (a - (0.08 * a) = 8 * a)
def optionC (a : ℝ) : Prop := (8 * a = 8 * a)
def optionD (a : ℝ) : Prop := (a * 8 = 8 * a)

-- The statement to be proved
theorem incorrect_option_B (a : ℝ) : 
  optionA a ∧ ¬optionB a ∧ optionC a ∧ optionD a := 
by
  sorry

end incorrect_option_B_l169_169066


namespace linda_winning_probability_l169_169767

noncomputable def probability_linda_wins : ℝ :=
  (1 / 16 : ℝ) / (1 - (1 / 32 : ℝ))

theorem linda_winning_probability :
  probability_linda_wins = 2 / 31 :=
sorry

end linda_winning_probability_l169_169767


namespace eighth_term_l169_169776

noncomputable def S (n : ℕ) (a : ℕ → ℤ) : ℤ := (n * (a 1 + a n)) / 2

variables {a : ℕ → ℤ} {d : ℤ}

-- Conditions
axiom sum_of_first_n_terms : ∀ n : ℕ, S n a = (n * (a 1 + a n)) / 2
axiom second_term : a 2 = 3
axiom sum_of_first_five_terms : S 5 a = 25

-- Question
theorem eighth_term : a 8 = 15 :=
sorry

end eighth_term_l169_169776


namespace evaluate_expression_l169_169787

theorem evaluate_expression : 
  (1 - (2 / 5)) / (1 - (1 / 4)) = (4 / 5) := 
by 
  sorry

end evaluate_expression_l169_169787


namespace trapezoid_longer_side_length_l169_169928

theorem trapezoid_longer_side_length (x : ℝ) (h₁ : 4 = 2*2) (h₂ : ∃ AP DQ O : ℝ, ∀ (S : ℝ), 
  S = (1/2) * (x + 2) * 1 → S = 2) : 
  x = 2 :=
by sorry

end trapezoid_longer_side_length_l169_169928


namespace clock_ticks_six_times_l169_169377

-- Define the conditions
def time_between_ticks (ticks : Nat) : Nat :=
  ticks - 1

def interval_duration (total_time : Nat) (ticks : Nat) : Nat :=
  total_time / time_between_ticks ticks

def number_of_ticks (total_time : Nat) (interval_time : Nat) : Nat :=
  total_time / interval_time + 1

-- Given conditions
def specific_time_intervals : Nat := 30
def eight_oclock_intervals : Nat := 42

-- Proven result
theorem clock_ticks_six_times : number_of_ticks specific_time_intervals (interval_duration eight_oclock_intervals 8) = 6 := 
sorry

end clock_ticks_six_times_l169_169377


namespace reciprocal_of_lcm_24_221_l169_169765

theorem reciprocal_of_lcm_24_221 : (1 / Nat.lcm 24 221) = (1 / 5304) :=
by 
  sorry

end reciprocal_of_lcm_24_221_l169_169765


namespace average_ratio_one_l169_169951

theorem average_ratio_one (scores : List ℝ) (h_len : scores.length = 50) :
  let A := (scores.sum / 50)
  let scores_with_averages := scores ++ [A, A]
  let A' := (scores_with_averages.sum / 52)
  A' = A :=
by
  sorry

end average_ratio_one_l169_169951


namespace lower_limit_of_range_with_multiples_l169_169939

theorem lower_limit_of_range_with_multiples (n : ℕ) (h : 2000 - n ≥ 198 * 10 ∧ n % 10 = 0 ∧ n + 1980 ≤ 2000) :
  n = 30 :=
by
  sorry

end lower_limit_of_range_with_multiples_l169_169939


namespace ratio_odd_even_divisors_l169_169432

def sum_of_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of divisors

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of odd divisors

def sum_of_even_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of even divisors

theorem ratio_odd_even_divisors (M : ℕ) (h : M = 36 * 36 * 98 * 210) :
  sum_of_odd_divisors M / sum_of_even_divisors M = 1 / 60 :=
by {
  sorry
}

end ratio_odd_even_divisors_l169_169432


namespace minimum_value_g_l169_169480

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - x - 2

def g (x : ℝ) : ℝ := (x + a)^2 - (x + a) - 2 + x

theorem minimum_value_g (a : ℝ) :
  (if 1 ≤ a then g a (-1) = a^2 - 3 * a - 1 else
   if -3 < a ∧ a < 1 then g a (-a) = -a - 2 else
   if a ≤ -3 then g a 3 = a^2 + 5 * a + 7 else false) :=
by
  sorry

end minimum_value_g_l169_169480


namespace find_a_values_l169_169657

theorem find_a_values (a x₁ x₂ : ℝ) (h1 : x^2 + a * x - 2 = 0)
                      (h2 : x₁ ≠ x₂)
                      (h3 : x₁^3 + 22 / x₂ = x₂^3 + 22 / x₁) :
                      a = 3 ∨ a = -3 :=
by
  sorry

end find_a_values_l169_169657


namespace A_leaves_after_one_day_l169_169850

-- Define and state all the conditions
def A_work_rate := 1 / 21
def B_work_rate := 1 / 28
def C_work_rate := 1 / 35
def total_work := 1
def B_time_after_A_leave := 21
def C_intermittent_working_cycle := 3 / 1 -- C works 1 out of every 3 days

-- The statement that needs to be proved
theorem A_leaves_after_one_day :
  ∃ x : ℕ, x = 1 ∧
  (A_work_rate * x + B_work_rate * x + (C_work_rate * (x / C_intermittent_working_cycle)) + (B_work_rate * B_time_after_A_leave) + (C_work_rate * (B_time_after_A_leave / C_intermittent_working_cycle)) = total_work) :=
sorry

end A_leaves_after_one_day_l169_169850


namespace Levi_has_5_lemons_l169_169934

theorem Levi_has_5_lemons
  (Levi Jayden Eli Ian : ℕ)
  (h1 : Jayden = Levi + 6)
  (h2 : Eli = 3 * Jayden)
  (h3 : Ian = 2 * Eli)
  (h4 : Levi + Jayden + Eli + Ian = 115) :
  Levi = 5 := 
sorry

end Levi_has_5_lemons_l169_169934


namespace min_value_of_expression_l169_169859

theorem min_value_of_expression (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_abc : a * b * c = 4) :
  (3 * a + b) * (2 * b + 3 * c) * (a * c + 4) ≥ 384 := 
by sorry

end min_value_of_expression_l169_169859


namespace part1_part2_l169_169020

theorem part1 (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a * b > 0) : a + b = 8 ∨ a + b = -8 :=
sorry

theorem part2 (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h4 : |a + b| = a + b) : a - b = 4 ∨ a - b = 8 :=
sorry

end part1_part2_l169_169020


namespace min_value_fract_ineq_l169_169411

theorem min_value_fract_ineq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (1 / a + 9 / b) ≥ 16 := 
sorry

end min_value_fract_ineq_l169_169411


namespace hardcover_books_count_l169_169509

theorem hardcover_books_count (h p : ℕ) (h_condition : h + p = 12) (cost_condition : 30 * h + 15 * p = 270) : h = 6 :=
by
  sorry

end hardcover_books_count_l169_169509


namespace scientific_notation_6500_l169_169589

theorem scientific_notation_6500 : (6500 : ℝ) = 6.5 * 10^3 := 
by 
  sorry

end scientific_notation_6500_l169_169589


namespace positive_number_percentage_of_itself_is_9_l169_169268

theorem positive_number_percentage_of_itself_is_9 (x : ℝ) (hx_pos : 0 < x) (h_condition : 0.01 * x^2 = 9) : x = 30 :=
by
  sorry

end positive_number_percentage_of_itself_is_9_l169_169268


namespace locker_count_proof_l169_169233

theorem locker_count_proof (cost_per_digit : ℕ := 3)
  (total_cost : ℚ := 224.91) :
  (N : ℕ) = 2151 :=
by
  sorry

end locker_count_proof_l169_169233


namespace product_of_divisors_of_30_l169_169363

open Nat

def divisors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

theorem product_of_divisors_of_30 :
  (divisors_of_30.foldr (· * ·) 1) = 810000 := by
  sorry

end product_of_divisors_of_30_l169_169363


namespace kathryn_more_pints_than_annie_l169_169597

-- Definitions for conditions
def annie_pints : ℕ := 8
def ben_pints (kathryn_pints : ℕ) : ℕ := kathryn_pints - 3
def total_pints (annie_pints kathryn_pints ben_pints : ℕ) : ℕ := annie_pints + kathryn_pints + ben_pints

-- The problem statement
theorem kathryn_more_pints_than_annie (k : ℕ) (h1 : total_pints annie_pints k (ben_pints k) = 25) : k - annie_pints = 2 :=
sorry

end kathryn_more_pints_than_annie_l169_169597


namespace find_m_range_l169_169410

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | -1 ≤ x ∧ x ≤ m + 1 }

theorem find_m_range (m : ℝ) : (B m ⊆ A) ↔ (-2 ≤ m ∧ m ≤ 3) := by
  sorry

end find_m_range_l169_169410


namespace max_value_expr_l169_169219

theorem max_value_expr (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 := 
sorry

end max_value_expr_l169_169219


namespace total_black_balls_l169_169628

-- Conditions
def number_of_white_balls (B : ℕ) : ℕ := 6 * B

def total_balls (B : ℕ) : ℕ := B + number_of_white_balls B

-- Theorem to prove
theorem total_black_balls (h : total_balls B = 56) : B = 8 :=
by
  sorry

end total_black_balls_l169_169628


namespace sin_160_eq_sin_20_l169_169952

theorem sin_160_eq_sin_20 : Real.sin (160 * Real.pi / 180) = Real.sin (20 * Real.pi / 180) :=
by
  sorry

end sin_160_eq_sin_20_l169_169952


namespace polynomial_roots_correct_l169_169940

theorem polynomial_roots_correct :
  (∃ (s : Finset ℝ), s = {1, 2, 4} ∧ (∀ x, x ∈ s ↔ (Polynomial.eval x (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 7 * Polynomial.X^2 + Polynomial.C 14 * Polynomial.X - Polynomial.C 8) = 0))) :=
by
  sorry

end polynomial_roots_correct_l169_169940


namespace tan_A_mul_tan_B_lt_one_l169_169500

theorem tan_A_mul_tan_B_lt_one (A B C : ℝ) (hC: C > 90) (hABC : A + B + C = 180) :
    Real.tan A * Real.tan B < 1 :=
sorry

end tan_A_mul_tan_B_lt_one_l169_169500


namespace minimum_value_of_function_l169_169531

theorem minimum_value_of_function :
  ∃ (y : ℝ), y > 0 ∧
  (∀ z : ℝ, z > 0 → y^2 + 10 * y + 100 / y^3 ≤ z^2 + 10 * z + 100 / z^3) ∧ 
  y^2 + 10 * y + 100 / y^3 = 50^(2/3) + 10 * 50^(1/3) + 2 := 
sorry

end minimum_value_of_function_l169_169531


namespace inequality_holds_l169_169820

theorem inequality_holds (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 := 
sorry

end inequality_holds_l169_169820


namespace sprinkles_remaining_l169_169908

theorem sprinkles_remaining (initial_cans : ℕ) (remaining_cans : ℕ) 
  (h1 : initial_cans = 12) 
  (h2 : remaining_cans = (initial_cans / 2) - 3) : 
  remaining_cans = 3 := 
by
  sorry

end sprinkles_remaining_l169_169908


namespace parabola_equation_l169_169639

theorem parabola_equation (p : ℝ) (h : 2 * p = 8) :
  ∃ (a : ℝ), a = 8 ∧ (y^2 = a * x ∨ y^2 = -a * x) :=
by
  sorry

end parabola_equation_l169_169639


namespace most_stable_city_l169_169513

def variance_STD : ℝ := 12.5
def variance_A : ℝ := 18.3
def variance_B : ℝ := 17.4
def variance_C : ℝ := 20.1

theorem most_stable_city : variance_STD < variance_A ∧ variance_STD < variance_B ∧ variance_STD < variance_C :=
by {
  -- Proof skipped
  sorry
}

end most_stable_city_l169_169513


namespace geometric_sequence_sum_l169_169634

-- Define the problem conditions and the result
theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ), a 1 + a 2 = 16 ∧ a 3 + a 4 = 24 → a 7 + a 8 = 54 :=
by
  -- Preliminary steps and definitions to prove the theorem
  sorry

end geometric_sequence_sum_l169_169634


namespace monotonic_intervals_slope_tangent_line_inequality_condition_l169_169398

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * (a + 2) * x^2 + 2 * a * x
noncomputable def g (a x : ℝ) : ℝ := (1/2) * (a - 5) * x^2

theorem monotonic_intervals (a : ℝ) (h : a ≥ 4) :
  (∀ x, deriv (f a) x = x^2 - (a + 2) * x + 2 * a) ∧
  ((∀ x, x < 2 → deriv (f a) x > 0) ∧ (∀ x, x > a → deriv (f a) x > 0)) ∧
  (∀ x, 2 < x ∧ x < a → deriv (f a) x < 0) :=
sorry

theorem slope_tangent_line (a : ℝ) (h : a ≥ 4) :
  (∀ x, deriv (f a) x = x^2 - (a + 2) * x + 2 * a) ∧
  (∀ x_0 y_0 k, y_0 = f a x_0 ∧ k = deriv (f a) x_0 ∧ k ≥ -(25/4) →
    4 ≤ a ∧ a ≤ 7) :=
sorry

theorem inequality_condition (a : ℝ) (h : a ≥ 4) :
  (∀ x_1 x_2, 3 ≤ x_1 ∧ x_1 < x_2 ∧ x_2 ≤ 4 →
    abs (f a x_1 - f a x_2) > abs (g a x_1 - g a x_2)) →
  (14/3 ≤ a ∧ a ≤ 6) :=
sorry

end monotonic_intervals_slope_tangent_line_inequality_condition_l169_169398


namespace smallest_number_of_slices_l169_169780

def cheddar_slices : ℕ := 12
def swiss_slices : ℕ := 28
def gouda_slices : ℕ := 18

theorem smallest_number_of_slices : Nat.lcm (Nat.lcm cheddar_slices swiss_slices) gouda_slices = 252 :=
by 
  sorry

end smallest_number_of_slices_l169_169780


namespace find_divisor_l169_169194

theorem find_divisor (x : ℝ) (h : x / n = 0.01 * (x * n)) : n = 10 :=
sorry

end find_divisor_l169_169194


namespace kwik_e_tax_revenue_l169_169943

def price_federal : ℕ := 50
def price_state : ℕ := 30
def price_quarterly : ℕ := 80

def num_federal : ℕ := 60
def num_state : ℕ := 20
def num_quarterly : ℕ := 10

def revenue_federal := num_federal * price_federal
def revenue_state := num_state * price_state
def revenue_quarterly := num_quarterly * price_quarterly

def total_revenue := revenue_federal + revenue_state + revenue_quarterly

theorem kwik_e_tax_revenue : total_revenue = 4400 := by
  sorry

end kwik_e_tax_revenue_l169_169943


namespace tangent_line_through_point_l169_169761

theorem tangent_line_through_point (t : ℝ) :
    (∃ l : ℝ → ℝ, (∃ m : ℝ, (∀ x, l x = 2 * m * x - m^2) ∧ (t = m - 2 * m + 2 * m * m) ∧ m = 1/2) ∧ l t = 0)
    → t = 1/4 :=
by
  sorry

end tangent_line_through_point_l169_169761


namespace find_ratio_l169_169541

theorem find_ratio (a b : ℝ) (h : a / b = 3 / 2) : (a + b) / a = 5 / 3 :=
sorry

end find_ratio_l169_169541


namespace joe_commute_time_l169_169124

theorem joe_commute_time
  (d : ℝ) -- total one-way distance from home to school
  (rw : ℝ) -- Joe's walking rate
  (rr : ℝ := 4 * rw) -- Joe's running rate (4 times walking rate)
  (walking_time_for_one_third : ℝ := 9) -- Joe takes 9 minutes to walk one-third distance
  (walking_time_two_thirds : ℝ := 2 * walking_time_for_one_third) -- time to walk two-thirds distance
  (running_time_two_thirds : ℝ := walking_time_two_thirds / 4) -- time to run two-thirds 
  : (2 * walking_time_two_thirds + running_time_two_thirds) = 40.5 := -- total travel time
by
  sorry

end joe_commute_time_l169_169124


namespace range_of_a_l169_169405

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (1 / 2) * Real.log x

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := (2 * a * x^2 + 1) / (2 * x)

def p (a : ℝ) : Prop := ∀ x, 1 ≤ x → f_prime (a) (x) ≤ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1

theorem range_of_a (a : ℝ) : (p a ∧ q a) → -1 < a ∧ a ≤ -1 / 2 :=
by
  sorry

end range_of_a_l169_169405


namespace number_that_multiplies_b_l169_169489

theorem number_that_multiplies_b (a b x : ℝ) (h0 : 4 * a = x * b) (h1 : a * b ≠ 0) (h2 : (a / 5) / (b / 4) = 1) : x = 5 :=
by
  sorry

end number_that_multiplies_b_l169_169489


namespace sequence_period_2016_l169_169236

theorem sequence_period_2016 : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = 1 / (1 - a n)) → 
  a 1 = 1 / 2 → 
  a 2016 = -1 :=
by
  sorry

end sequence_period_2016_l169_169236


namespace team_lineup_count_l169_169073

theorem team_lineup_count (total_members specialized_kickers remaining_players : ℕ) 
  (captain_assignments : specialized_kickers = 2) 
  (available_members : total_members = 20) 
  (choose_players : remaining_players = 8) : 
  (2 * (Nat.choose 19 remaining_players)) = 151164 := 
by
  sorry

end team_lineup_count_l169_169073


namespace m_eq_half_l169_169054

theorem m_eq_half (m : ℝ) (h1 : m > 0) (h2 : ∀ x, (0 < x ∧ x < m) → (x * (x - 1) < 0))
  (h3 : ∃ x, (0 < x ∧ x < 1) ∧ ¬(0 < x ∧ x < m)) : m = 1 / 2 :=
sorry

end m_eq_half_l169_169054


namespace find_students_that_got_As_l169_169291

variables (Emily Frank Grace Harry : Prop)

theorem find_students_that_got_As
  (cond1 : Emily → Frank)
  (cond2 : Frank → Grace)
  (cond3 : Grace → Harry)
  (cond4 : Harry → ¬ Emily)
  (three_A_students : ¬ (Emily ∧ Frank ∧ Grace ∧ Harry) ∧
                      (Emily ∧ Frank ∧ Grace ∧ ¬ Harry ∨
                       Emily ∧ Frank ∧ ¬ Grace ∧ Harry ∨
                       Emily ∧ ¬ Frank ∧ Grace ∧ Harry ∨
                       ¬ Emily ∧ Frank ∧ Grace ∧ Harry)) :
  (¬ Emily ∧ Frank ∧ Grace ∧ Harry) :=
by {
  sorry
}

end find_students_that_got_As_l169_169291


namespace completing_square_solution_l169_169313

theorem completing_square_solution (x : ℝ) :
  x^2 - 4*x - 3 = 0 ↔ (x - 2)^2 = 7 :=
sorry

end completing_square_solution_l169_169313


namespace fraction_without_cable_or_vcr_l169_169826

theorem fraction_without_cable_or_vcr (T : ℕ) (h1 : ℚ) (h2 : ℚ) (h3 : ℚ) 
  (h1 : h1 = 1 / 5 * T) 
  (h2 : h2 = 1 / 10 * T) 
  (h3 : h3 = 1 / 3 * (1 / 5 * T)) 
: (T - (1 / 5 * T + 1 / 10 * T - 1 / 3 * (1 / 5 * T))) / T = 23 / 30 := 
by 
  sorry

end fraction_without_cable_or_vcr_l169_169826


namespace mean_equality_l169_169273

-- Define average calculation function
def average (a b c : ℕ) : ℕ :=
  (a + b + c) / 3

def average_two (a b : ℕ) : ℕ :=
  (a + b) / 2

theorem mean_equality (x : ℕ) 
  (h : average 8 16 24 = average_two 10 x) : 
  x = 22 :=
by {
  -- The actual proof is here
  sorry
}

end mean_equality_l169_169273


namespace quadratic_real_roots_m_l169_169797

theorem quadratic_real_roots_m (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x1 + 4 * x1 + m = 0 ∧ x2 * x2 + 4 * x2 + m = 0) →
  m ≤ 4 :=
by
  sorry

end quadratic_real_roots_m_l169_169797


namespace good_goods_not_cheap_l169_169821

-- Define the propositions "good goods" and "not cheap"
variables (p q : Prop)

-- State that "good goods are not cheap" is expressed by the implication p → q
theorem good_goods_not_cheap : p → q → (p → q) ↔ (p ∧ q → p ∧ q) := by
  sorry

end good_goods_not_cheap_l169_169821


namespace lunch_break_duration_l169_169804

/-- Paula and her two helpers start at 7:00 AM and paint 60% of a house together,
    finishing at 5:00 PM. The next day, only the helpers paint and manage to
    paint 30% of another house, finishing at 3:00 PM. On the third day, Paula
    paints alone and paints the remaining 40% of the house, finishing at 4:00 PM.
    Prove that the length of their lunch break each day is 1 hour (60 minutes). -/
theorem lunch_break_duration :
  ∃ (L : ℝ), 
    (0 < L) ∧ 
    (L < 10) ∧
    (∃ (p h : ℝ), 
       (10 - L) * (p + h) = 0.6 ∧
       (8 - L) * h = 0.3 ∧
       (9 - L) * p = 0.4) ∧  
    L = 1 :=
by
  sorry

end lunch_break_duration_l169_169804


namespace number_of_cutlery_pieces_added_l169_169923

-- Define the initial conditions
def forks_initial := 6
def knives_initial := forks_initial + 9
def spoons_initial := 2 * knives_initial
def teaspoons_initial := forks_initial / 2
def total_initial_cutlery := forks_initial + knives_initial + spoons_initial + teaspoons_initial
def total_final_cutlery := 62

-- Define the total number of cutlery pieces added
def cutlery_added := total_final_cutlery - total_initial_cutlery

-- Define the theorem to prove
theorem number_of_cutlery_pieces_added : cutlery_added = 8 := by
  sorry

end number_of_cutlery_pieces_added_l169_169923


namespace relationship_y1_y2_l169_169083

theorem relationship_y1_y2 (y1 y2 : ℝ) (m : ℝ) (h_m : m ≠ 0) 
  (hA : y1 = m * (-2) + 4) (hB : 3 = m * 1 + 4) (hC : y2 = m * 3 + 4) : y1 > y2 :=
by
  sorry

end relationship_y1_y2_l169_169083


namespace red_marbles_count_l169_169623

theorem red_marbles_count (R : ℕ) (h1 : 48 - R > 0) (h2 : ((48 - R) / 48 : ℚ) * ((48 - R) / 48) = 9 / 16) : R = 12 :=
sorry

end red_marbles_count_l169_169623


namespace max_n_leq_V_l169_169425

theorem max_n_leq_V (n : ℤ) (V : ℤ) (h1 : 102 * n^2 <= V) (h2 : ∀ k : ℤ, (102 * k^2 <= V) → k <= 8) : V >= 6528 :=
sorry

end max_n_leq_V_l169_169425


namespace geom_S4_eq_2S2_iff_abs_q_eq_1_l169_169902

variable {α : Type*} [LinearOrderedField α]

-- defining the sum of first n terms of a geometric sequence
def geom_series_sum (a q : α) (n : ℕ) :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

noncomputable def S (a q : α) (n : ℕ) := geom_series_sum a q n

theorem geom_S4_eq_2S2_iff_abs_q_eq_1 
  (a q : α) : 
  S a q 4 = 2 * S a q 2 ↔ |q| = 1 :=
sorry

end geom_S4_eq_2S2_iff_abs_q_eq_1_l169_169902


namespace custom_operation_correct_l169_169299

noncomputable def custom_operation (a b c : ℕ) : ℝ :=
  (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

theorem custom_operation_correct : custom_operation 6 15 5 = 2 := by
  sorry

end custom_operation_correct_l169_169299


namespace simplify_and_evaluate_l169_169168

noncomputable def simplified_expr (x y : ℝ) : ℝ :=
  ((-2 * x + y)^2 - (2 * x - y) * (y + 2 * x) - 6 * y) / (2 * y)

theorem simplify_and_evaluate :
  let x := -1
  let y := 2
  simplified_expr x y = 1 :=
by
  -- Proof will go here
  sorry

end simplify_and_evaluate_l169_169168


namespace carlson_total_land_l169_169772

open Real

theorem carlson_total_land 
  (initial_land : ℝ)
  (cost_additional_land1 : ℝ)
  (cost_additional_land2 : ℝ)
  (cost_per_square_meter : ℝ) :
  initial_land = 300 →
  cost_additional_land1 = 8000 →
  cost_additional_land2 = 4000 →
  cost_per_square_meter = 20 →
  (initial_land + (cost_additional_land1 + cost_additional_land2) / cost_per_square_meter) = 900 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  done

end carlson_total_land_l169_169772


namespace total_books_l169_169898

noncomputable def num_books_on_shelf : ℕ := 8

theorem total_books (p h s : ℕ) (assump1 : p = 2) (assump2 : h = 6) (assump3 : s = 36) :
  p + h = num_books_on_shelf :=
by {
  -- leaving the proof construction out as per instructions
  sorry
}

end total_books_l169_169898


namespace arccos_cos_11_l169_169862

theorem arccos_cos_11 : Real.arccos (Real.cos 11) = 1.425 :=
by
  sorry

end arccos_cos_11_l169_169862


namespace max_ab_l169_169022

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 8) : 
  ab ≤ 8 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2 * b = 8 ∧ ab = 8 :=
by
  sorry

end max_ab_l169_169022


namespace original_price_of_house_l169_169608

theorem original_price_of_house (P : ℝ) 
  (h1 : P * 0.56 = 56000) : P = 100000 :=
sorry

end original_price_of_house_l169_169608


namespace usual_travel_time_l169_169046

theorem usual_travel_time
  (S : ℝ) (T : ℝ) 
  (h0 : S > 0)
  (h1 : (S / T) = (4 / 5 * S / (T + 6))) : 
  T = 30 :=
by sorry

end usual_travel_time_l169_169046


namespace average_income_correct_l169_169544

-- Define the incomes for each day
def income_day_1 : ℕ := 300
def income_day_2 : ℕ := 150
def income_day_3 : ℕ := 750
def income_day_4 : ℕ := 400
def income_day_5 : ℕ := 500

-- Define the number of days
def number_of_days : ℕ := 5

-- Define the total income
def total_income : ℕ := income_day_1 + income_day_2 + income_day_3 + income_day_4 + income_day_5

-- Define the average income
def average_income : ℕ := total_income / number_of_days

-- State that the average income is 420
theorem average_income_correct :
  average_income = 420 := by
  sorry

end average_income_correct_l169_169544


namespace geometric_sequence_b_general_term_a_l169_169660

-- Definitions of sequences and given conditions
def a (n : ℕ) : ℕ := sorry -- The sequence a_n
def S (n : ℕ) : ℕ := sorry -- The sum of the first n terms S_n

axiom a1_condition : a 1 = 2
axiom recursion_formula (n : ℕ): S (n+1) = 4 * a n + 2

def b (n : ℕ) : ℕ := a (n+1) - 2 * a n -- Definition of b_n

-- Theorem 1: Prove that b_n is a geometric sequence
theorem geometric_sequence_b (n : ℕ) : ∃ q, ∀ m, b (m+1) = q * b m :=
  sorry

-- Theorem 2: Find the general term formula for a_n
theorem general_term_a (n : ℕ) : a n = n * 2^n :=
  sorry

end geometric_sequence_b_general_term_a_l169_169660


namespace TylerWeightDifference_l169_169728

-- Define the problem conditions
def PeterWeight : ℕ := 65
def SamWeight : ℕ := 105
def TylerWeight := 2 * PeterWeight

-- State the theorem
theorem TylerWeightDifference : (TylerWeight - SamWeight = 25) :=
by
  -- proof goes here
  sorry

end TylerWeightDifference_l169_169728


namespace algebra_eq_iff_sum_eq_one_l169_169846

-- Definitions from conditions
def expr1 (a b c : ℝ) : ℝ := a + b * c
def expr2 (a b c : ℝ) : ℝ := (a + b) * (a + c)

-- Lean statement for the proof problem
theorem algebra_eq_iff_sum_eq_one (a b c : ℝ) : expr1 a b c = expr2 a b c ↔ a + b + c = 1 :=
by
  sorry

end algebra_eq_iff_sum_eq_one_l169_169846


namespace hikers_rate_l169_169550

noncomputable def rate_up (rate_down := 15) : ℝ := 5

theorem hikers_rate :
  let R := rate_up
  let distance_down := rate_down
  let time := 2
  let rate_down := 1.5 * R
  distance_down = rate_down * time → R = 5 :=
by
  intro h
  sorry

end hikers_rate_l169_169550


namespace rectangle_minimal_area_l169_169889

theorem rectangle_minimal_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (l + w) = 120) : l * w = 675 :=
by
  -- Proof will go here
  sorry

end rectangle_minimal_area_l169_169889


namespace point_on_y_axis_coordinates_l169_169349

theorem point_on_y_axis_coordinates (m : ℤ) (P : ℤ × ℤ) (hP : P = (m - 1, m + 3)) (hY : P.1 = 0) : P = (0, 4) :=
sorry

end point_on_y_axis_coordinates_l169_169349


namespace jane_oldest_babysat_age_l169_169836

-- Given conditions
def jane_babysitting_has_constraints (jane_current_age jane_stop_babysitting_age jane_start_babysitting_age : ℕ) : Prop :=
  jane_current_age - jane_stop_babysitting_age = 10 ∧
  jane_stop_babysitting_age - jane_start_babysitting_age = 2

-- Helper definition for prime number constraint
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (n % m = 0)

-- Main goal: the current age of the oldest person Jane could have babysat is 19
theorem jane_oldest_babysat_age
  (jane_current_age jane_stop_babysitting_age jane_start_babysitting_age : ℕ)
  (H_constraints : jane_babysitting_has_constraints jane_current_age jane_stop_babysitting_age jane_start_babysitting_age) :
  ∃ (child_age : ℕ), child_age = 19 ∧ is_prime child_age ∧
  (child_age = (jane_stop_babysitting_age / 2 + 10) ∨ child_age = (jane_stop_babysitting_age / 2 + 9)) :=
sorry  -- Proof to be filled in.

end jane_oldest_babysat_age_l169_169836


namespace range_for_a_l169_169479

def f (a : ℝ) (x : ℝ) := 2 * x^3 - a * x^2 + 1

def two_zeros_in_interval (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (1/2 ≤ x1 ∧ x1 ≤ 2) ∧ (1/2 ≤ x2 ∧ x2 ≤ 2) ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0

theorem range_for_a {a : ℝ} : (3/2 : ℝ) < a ∧ a ≤ (17/4 : ℝ) ↔ two_zeros_in_interval a :=
by sorry

end range_for_a_l169_169479


namespace multiples_of_6_or_8_but_not_both_l169_169568

theorem multiples_of_6_or_8_but_not_both (n : ℕ) : 
  n = 25 ∧ (n = 18) ∧ (n = 6) → (25 - 6) + (18 - 6) = 31 :=
by
  sorry

end multiples_of_6_or_8_but_not_both_l169_169568


namespace part_a_part_b_l169_169946

noncomputable def sequence_a (n : ℕ) : ℝ :=
  Real.cos (10^n * Real.pi / 180)

theorem part_a (h : 100 > 2) : sequence_a 100 > 0 := by
  sorry

theorem part_b : |sequence_a 100| < 0.18 := by
  sorry

end part_a_part_b_l169_169946


namespace negation_of_universal_prop_l169_169913

variable (P : ∀ x : ℝ, Real.cos x ≤ 1)

theorem negation_of_universal_prop : ∃ x₀ : ℝ, Real.cos x₀ > 1 :=
sorry

end negation_of_universal_prop_l169_169913


namespace Ivan_defeats_Koschei_l169_169822

-- Definitions of the springs and conditions based on the problem
section

variable (S: ℕ → Prop)  -- S(n) means the water from spring n
variable (deadly: ℕ → Prop)  -- deadly(n) if water from nth spring is deadly

-- Conditions
axiom accessibility (n: ℕ): (1 ≤ n ∧ n ≤ 9 → ∀ i: ℕ, S i)
axiom koschei_access: S 10
axiom lethality (n: ℕ): (S n → deadly n)
axiom neutralize (i j: ℕ): (1 ≤ i ∧ i < j ∧ j ≤ 9 → ∃ k: ℕ, S k ∧ k > j → ¬deadly i)

-- Statement to prove
theorem Ivan_defeats_Koschei:
  ∃ i: ℕ, (1 ≤ i ∧ i ≤ 9) → (S 10 → ¬deadly i) ∧ (S 0 ∧ (S 10 → deadly 0)) :=
sorry

end

end Ivan_defeats_Koschei_l169_169822


namespace knights_and_liars_solution_l169_169082

-- Definitions of each person's statement as predicates
def person1_statement (liar : ℕ → Prop) : Prop := liar 2 ∧ liar 3 ∧ liar 4 ∧ liar 5 ∧ liar 6
def person2_statement (liar : ℕ → Prop) : Prop := liar 1 ∧ ∀ i, i ≠ 1 → ¬ liar i
def person3_statement (liar : ℕ → Prop) : Prop := liar 4 ∧ liar 5 ∧ liar 6 ∧ ¬ liar 3 ∧ ¬ liar 2 ∧ ¬ liar 1
def person4_statement (liar : ℕ → Prop) : Prop := liar 1 ∧ liar 2 ∧ liar 3 ∧ ∀ i, i > 3 → ¬ liar i
def person5_statement (liar : ℕ → Prop) : Prop := liar 6 ∧ ∀ i, i ≠ 6 → ¬ liar i
def person6_statement (liar : ℕ → Prop) : Prop := liar 5 ∧ ∀ i, i ≠ 5 → ¬ liar i

-- Definition of a knight and a liar
def is_knight (statement : Prop) : Prop := statement
def is_liar (statement : Prop) : Prop := ¬ statement

-- Defining the theorem
theorem knights_and_liars_solution (knight liar : ℕ → Prop) : 
  is_liar (person1_statement liar) ∧ 
  is_knight (person2_statement liar) ∧ 
  is_liar (person3_statement liar) ∧ 
  is_liar (person4_statement liar) ∧ 
  is_knight (person5_statement liar) ∧ 
  is_liar (person6_statement liar) :=
by
  sorry

end knights_and_liars_solution_l169_169082


namespace river_length_l169_169539

theorem river_length (x : ℝ) (h1 : 3 * x + x = 80) : x = 20 :=
sorry

end river_length_l169_169539


namespace rectangular_garden_shorter_side_length_l169_169779

theorem rectangular_garden_shorter_side_length
  (a b : ℕ)
  (h1 : 2 * a + 2 * b = 46)
  (h2 : a * b = 108) :
  b = 9 :=
by 
  sorry

end rectangular_garden_shorter_side_length_l169_169779


namespace bn_is_arithmetic_an_general_formula_l169_169594

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l169_169594


namespace base7_to_base10_l169_169063

theorem base7_to_base10 (a b c d e : ℕ) (h : 45321 = a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0)
  (ha : a = 4) (hb : b = 5) (hc : c = 3) (hd : d = 2) (he : e = 1) : 
  a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0 = 11481 := 
by 
  sorry

end base7_to_base10_l169_169063


namespace sum_of_seven_consecutive_integers_l169_169365

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_of_seven_consecutive_integers_l169_169365


namespace product_remainder_l169_169993

theorem product_remainder (a b : ℕ) (m n : ℤ) (ha : a = 3 * m + 2) (hb : b = 3 * n + 2) : 
  (a * b) % 3 = 1 := 
by 
  sorry

end product_remainder_l169_169993


namespace polynomial_A_l169_169415

variables {a b : ℝ} (A : ℝ)
variables (h1 : 2 ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0)

theorem polynomial_A (h : A / (2 * a * b) = 1 - 4 * a ^ 2) : 
  A = 2 * a * b - 8 * a ^ 3 * b :=
by
  sorry

end polynomial_A_l169_169415


namespace geom_sum_eq_six_l169_169149

variable (a : ℕ → ℝ)
variable (r : ℝ) -- common ratio for geometric sequence

-- Conditions
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom pos_seq (n : ℕ) : a (n + 1) > 0
axiom given_eq : a 1 * a 3 + 2 * a 2 * a 5 + a 4 * a 6 = 36

-- Proof statement
theorem geom_sum_eq_six : a 2 + a 5 = 6 :=
sorry

end geom_sum_eq_six_l169_169149


namespace smallest_mul_seven_perfect_square_l169_169247

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Define the problem statement
theorem smallest_mul_seven_perfect_square :
  ∀ x : ℕ, x > 0 → (is_perfect_square (7 * x) ↔ x = 7) := 
by {
  sorry
}

end smallest_mul_seven_perfect_square_l169_169247


namespace certain_number_sixth_powers_l169_169682

theorem certain_number_sixth_powers :
  ∃ N, (∀ n : ℕ, n < N → ∃ a : ℕ, n = a^6) ∧
       (∃ m ≤ N, (∀ n < m, ∃ k : ℕ, n = k^6) ∧ ¬ ∃ k : ℕ, m = k^6) :=
sorry

end certain_number_sixth_powers_l169_169682


namespace problem_sol_l169_169953

-- Defining the operations as given
def operation_hash (a b c : ℤ) : ℤ := 4 * a ^ 3 + 4 * b ^ 3 + 8 * a ^ 2 * b + c
def operation_star (a b d : ℤ) : ℤ := 2 * a ^ 2 - 3 * b ^ 2 + d ^ 3

-- Main theorem statement
theorem problem_sol (a b x c d : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (hc : c > 0) (hd : d > 0) 
  (h3 : operation_hash a x c = 250)
  (h4 : operation_star a b d + x = 50) :
  False := sorry

end problem_sol_l169_169953


namespace line_containing_chord_l169_169612

variable {x y x₁ y₁ x₂ y₂ : ℝ}

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 4 = 1)

def midpoint_condition (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) : Prop := 
  (x₁ + x₂ = 2) ∧ (y₁ + y₂ = 2)

theorem line_containing_chord (h₁ : ellipse_eq x₁ y₁) 
                               (h₂ : ellipse_eq x₂ y₂) 
                               (hmp : midpoint_condition x₁ x₂ y₁ y₂)
    : 4 * 1 + 9 * 1 - 13 = 0 := 
sorry

end line_containing_chord_l169_169612


namespace find_smallest_even_number_l169_169062

theorem find_smallest_even_number (x : ℕ) (h1 : 
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14)) = 424) : 
  x = 46 := 
by
  sorry

end find_smallest_even_number_l169_169062


namespace range_of_t_l169_169057

theorem range_of_t (a b c : ℝ) (t : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_inequality : ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → (1 / a^2) + (4 / b^2) + (t / c^2) ≥ 0) :
  t ≥ -9 :=
sorry

end range_of_t_l169_169057


namespace intersection_A_B_l169_169543

/-- Definition of set A -/
def A : Set ℕ := {1, 2, 3, 4}

/-- Definition of set B -/
def B : Set ℕ := {x | x > 2}

/-- The theorem to prove the intersection of sets A and B -/
theorem intersection_A_B : A ∩ B = {3, 4} :=
by
  sorry

end intersection_A_B_l169_169543


namespace mean_of_combined_sets_l169_169052

theorem mean_of_combined_sets (mean_set1 : ℝ) (mean_set2 : ℝ) (n1 : ℕ) (n2 : ℕ)
  (h1 : mean_set1 = 15) (h2 : mean_set2 = 27) (h3 : n1 = 7) (h4 : n2 = 8) :
  (mean_set1 * n1 + mean_set2 * n2) / (n1 + n2) = 21.4 := 
sorry

end mean_of_combined_sets_l169_169052


namespace parabola_tangent_line_l169_169100

theorem parabola_tangent_line (a : ℝ) : 
  (∀ x : ℝ, (y = ax^2 + 6 ↔ y = x)) → a = 1 / 24 :=
by
  sorry

end parabola_tangent_line_l169_169100


namespace distance_to_focus_l169_169921

open Real

theorem distance_to_focus {P : ℝ × ℝ} 
  (h₁ : P.2 ^ 2 = 4 * P.1)
  (h₂ : abs (P.1 + 3) = 5) :
  dist P ⟨1, 0⟩ = 3 := 
sorry

end distance_to_focus_l169_169921


namespace jill_first_show_length_l169_169078

theorem jill_first_show_length : 
  ∃ (x : ℕ), (x + 4 * x = 150) ∧ (x = 30) :=
sorry

end jill_first_show_length_l169_169078


namespace permissible_range_n_l169_169134

theorem permissible_range_n (n x y m : ℝ) (hn : n ≤ x) (hxy : x < y) (hy : y ≤ n+1)
  (hm_in: x < m ∧ m < y) (habs_eq : |y| = |m| + |x|): 
  -1 < n ∧ n < 1 := sorry

end permissible_range_n_l169_169134


namespace find_numbers_l169_169875

theorem find_numbers (x : ℚ) (a : ℚ) (b : ℚ) (h₁ : a = 8 * x) (h₂ : b = x^2 - 1) :
  (a * b + a = (2 * x)^3) ∧ (a * b + b = (2 * x - 1)^3) → 
  x = 14 / 13 ∧ a = 112 / 13 ∧ b = 27 / 169 :=
by
  intros h
  sorry

end find_numbers_l169_169875


namespace largest_negative_integer_l169_169026

theorem largest_negative_integer :
  ∃ (n : ℤ), (∀ m : ℤ, m < 0 → m ≤ n) ∧ n = -1 := by
  sorry

end largest_negative_integer_l169_169026


namespace quadratic_one_pos_one_neg_l169_169595

theorem quadratic_one_pos_one_neg (a : ℝ) : 
  (a < -1) → (∃ x1 x2 : ℝ, x1 * x2 < 0 ∧ x1 + x2 > 0 ∧ (x1^2 + x1 + a = 0 ∧ x2^2 + x2 + a = 0)) :=
sorry

end quadratic_one_pos_one_neg_l169_169595


namespace games_went_this_year_l169_169917

theorem games_went_this_year (t l : ℕ) (h1 : t = 13) (h2 : l = 9) : (t - l = 4) :=
by
  sorry

end games_went_this_year_l169_169917


namespace melanies_plums_l169_169842

variable (pickedPlums : ℕ)
variable (gavePlums : ℕ)

theorem melanies_plums (h1 : pickedPlums = 7) (h2 : gavePlums = 3) : (pickedPlums - gavePlums) = 4 :=
by
  sorry

end melanies_plums_l169_169842


namespace jacob_three_heads_probability_l169_169672

noncomputable section

def probability_three_heads_after_two_tails : ℚ := 1 / 96

theorem jacob_three_heads_probability :
  let p := (1 / 2) ^ 4 * (1 / 6)
  p = probability_three_heads_after_two_tails := by
sorry

end jacob_three_heads_probability_l169_169672


namespace max_sin_a_given_sin_a_plus_b_l169_169726

theorem max_sin_a_given_sin_a_plus_b (a b : ℝ) (sin_add : Real.sin (a + b) = Real.sin a + Real.sin b) : 
  Real.sin a ≤ 1 := 
sorry

end max_sin_a_given_sin_a_plus_b_l169_169726


namespace sum_of_roots_l169_169959

theorem sum_of_roots : 
  let a := 1
  let b := 2001
  let c := -2002
  ∀ x y: ℝ, (x^2 + b*x + c = 0) ∧ (y^2 + b*y + c = 0) -> (x + y = -b) :=
by
  sorry

end sum_of_roots_l169_169959


namespace flowers_bees_butterflies_comparison_l169_169982

def num_flowers : ℕ := 12
def num_bees : ℕ := 7
def num_butterflies : ℕ := 4
def difference_flowers_bees : ℕ := num_flowers - num_bees

theorem flowers_bees_butterflies_comparison :
  difference_flowers_bees - num_butterflies = 1 :=
by
  -- The proof will go here
  sorry

end flowers_bees_butterflies_comparison_l169_169982


namespace price_per_butterfly_l169_169018

theorem price_per_butterfly (jars : ℕ) (caterpillars_per_jar : ℕ) (fail_percentage : ℝ) (total_money : ℝ) (price : ℝ) :
  jars = 4 →
  caterpillars_per_jar = 10 →
  fail_percentage = 0.40 →
  total_money = 72 →
  price = 3 :=
by
  intros h_jars h_caterpillars h_fail_percentage h_total_money
  -- Full proof here
  sorry

end price_per_butterfly_l169_169018


namespace product_of_remaining_numbers_l169_169222

theorem product_of_remaining_numbers {a b c d : ℕ} (h1 : a = 11) (h2 : b = 22) (h3 : c = 33) (h4 : d = 44) :
  ∃ (x y z : ℕ), 
  (∃ n: ℕ, (a + b + c + d) - n * 3 = 3 ∧ -- We removed n groups of 3 different numbers
             x + y + z = 2 * n + (a + b + c + d)) ∧ -- We added 2 * n numbers back
  x * y * z = 12 := 
sorry

end product_of_remaining_numbers_l169_169222


namespace luke_clothing_distribution_l169_169945

theorem luke_clothing_distribution (total_clothing: ℕ) (first_load: ℕ) (num_loads: ℕ) 
  (remaining_clothing : total_clothing - first_load = 30)
  (equal_load_per_small_load: (total_clothing - first_load) / num_loads = 6) : 
  total_clothing = 47 ∧ first_load = 17 ∧ num_loads = 5 :=
by
  have h1 : total_clothing - first_load = 30 := remaining_clothing
  have h2 : (total_clothing - first_load) / num_loads = 6 := equal_load_per_small_load
  sorry

end luke_clothing_distribution_l169_169945


namespace four_students_same_acquaintances_l169_169019

theorem four_students_same_acquaintances
  (students : Finset ℕ)
  (acquainted : ∀ s ∈ students, (students \ {s}).card ≥ 68)
  (count : students.card = 102) :
  ∃ n, ∃ cnt, cnt ≥ 4 ∧ (∃ S, S ⊆ students ∧ S.card = cnt ∧ ∀ x ∈ S, (students \ {x}).card = n) :=
sorry

end four_students_same_acquaintances_l169_169019


namespace value_of_a_plus_b_l169_169130

variables (a b : ℝ)

theorem value_of_a_plus_b (ha : abs a = 1) (hb : abs b = 4) (hab : a * b < 0) : a + b = 3 ∨ a + b = -3 := by
  sorry

end value_of_a_plus_b_l169_169130


namespace initial_pipes_l169_169843

variables (x : ℕ)

-- Defining the conditions
def one_pipe_time := x -- time for 1 pipe to fill the tank in hours
def eight_pipes_time := 1 / 4 -- 15 minutes = 1/4 hour

-- Proving the number of pipes
theorem initial_pipes (h1 : eight_pipes_time * 8 = one_pipe_time) : x = 2 :=
by
  sorry

end initial_pipes_l169_169843


namespace interest_rate_10_percent_l169_169877

-- Definitions for the problem
variables (P : ℝ) (R : ℝ) (T : ℝ)

-- Condition that the money doubles in 10 years on simple interest
def money_doubles_in_10_years (P R : ℝ) : Prop :=
  P = (P * R * 10) / 100

-- Statement that R is 10% if the money doubles in 10 years
theorem interest_rate_10_percent {P : ℝ} (h : money_doubles_in_10_years P R) : R = 10 :=
by
  sorry

end interest_rate_10_percent_l169_169877


namespace helen_chocolate_chip_cookies_l169_169679

theorem helen_chocolate_chip_cookies :
  let cookies_yesterday := 527
  let cookies_morning := 554
  cookies_yesterday + cookies_morning = 1081 :=
by
  let cookies_yesterday := 527
  let cookies_morning := 554
  show cookies_yesterday + cookies_morning = 1081
  -- The proof is omitted according to the provided instructions 
  sorry

end helen_chocolate_chip_cookies_l169_169679


namespace smallest_product_not_factor_60_l169_169806

theorem smallest_product_not_factor_60 : ∃ (a b : ℕ), a ≠ b ∧ a ∣ 60 ∧ b ∣ 60 ∧ ¬ (a * b) ∣ 60 ∧ a * b = 8 := sorry

end smallest_product_not_factor_60_l169_169806


namespace rowing_time_l169_169032

def man_speed_still := 10.0
def river_speed := 1.2
def total_distance := 9.856

def upstream_speed := man_speed_still - river_speed
def downstream_speed := man_speed_still + river_speed

def one_way_distance := total_distance / 2
def time_upstream := one_way_distance / upstream_speed
def time_downstream := one_way_distance / downstream_speed

theorem rowing_time :
  time_upstream + time_downstream = 1 :=
by
  sorry

end rowing_time_l169_169032


namespace area_of_garden_l169_169775

theorem area_of_garden (L P : ℝ) (H1 : 1500 = 30 * L) (H2 : 1500 = 12 * P) (H3 : P = 2 * L + 2 * (P / 2 - L)) : 
  (L * (P/2 - L)) = 625 :=
by
  sorry

end area_of_garden_l169_169775


namespace sonny_received_45_boxes_l169_169076

def cookies_received (cookies_given_brother : ℕ) (cookies_given_sister : ℕ) (cookies_given_cousin : ℕ) (cookies_left : ℕ) : ℕ :=
  cookies_given_brother + cookies_given_sister + cookies_given_cousin + cookies_left

theorem sonny_received_45_boxes :
  cookies_received 12 9 7 17 = 45 :=
by
  sorry

end sonny_received_45_boxes_l169_169076


namespace exists_h_not_divisible_l169_169718

noncomputable def h : ℝ := 1969^2 / 1968

theorem exists_h_not_divisible (h := 1969^2 / 1968) :
  ∃ (h : ℝ), ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
by
  use h
  intro n
  sorry

end exists_h_not_divisible_l169_169718


namespace jacob_age_l169_169990

/- Conditions:
1. Rehana's current age is 25.
2. In five years, Rehana's age is three times Phoebe's age.
3. Jacob's current age is 3/5 of Phoebe's current age.

Prove that Jacob's current age is 3.
-/

theorem jacob_age (R P J : ℕ) (h1 : R = 25) (h2 : R + 5 = 3 * (P + 5)) (h3 : J = 3 / 5 * P) : J = 3 :=
by
  sorry

end jacob_age_l169_169990


namespace induction_step_n_eq_1_l169_169994

theorem induction_step_n_eq_1 : (1 + 2 + 3 = (1+1)*(2*1+1)) :=
by
  -- Proof would go here
  sorry

end induction_step_n_eq_1_l169_169994


namespace function_domain_real_l169_169611

theorem function_domain_real (k : ℝ) : 0 ≤ k ∧ k < 4 ↔ (∀ x : ℝ, k * x^2 + k * x + 1 ≠ 0) :=
by
  sorry

end function_domain_real_l169_169611


namespace owen_work_hours_l169_169737

def total_hours := 24
def chores_hours := 7
def sleep_hours := 11

theorem owen_work_hours : total_hours - chores_hours - sleep_hours = 6 := by
  sorry

end owen_work_hours_l169_169737


namespace three_pow_12_mul_three_pow_8_equals_243_pow_4_l169_169335

theorem three_pow_12_mul_three_pow_8_equals_243_pow_4 : 3^12 * 3^8 = 243^4 := 
by sorry

end three_pow_12_mul_three_pow_8_equals_243_pow_4_l169_169335


namespace bad_games_count_l169_169635

/-- 
  Oliver bought a total of 11 video games, and 6 of them worked.
  Prove that the number of bad games he bought is 5.
-/
theorem bad_games_count (total_games : ℕ) (working_games : ℕ) (h1 : total_games = 11) (h2 : working_games = 6) : total_games - working_games = 5 :=
by
  sorry

end bad_games_count_l169_169635


namespace exist_non_negative_product_l169_169777

theorem exist_non_negative_product (a1 a2 a3 a4 a5 a6 a7 a8 : ℝ) :
  0 ≤ a1 * a3 + a2 * a4 ∨
  0 ≤ a1 * a5 + a2 * a6 ∨
  0 ≤ a1 * a7 + a2 * a8 ∨
  0 ≤ a3 * a5 + a4 * a6 ∨
  0 ≤ a3 * a7 + a4 * a8 ∨
  0 ≤ a5 * a7 + a6 * a8 :=
sorry

end exist_non_negative_product_l169_169777


namespace complex_expression_evaluation_l169_169569

noncomputable def imaginary_i := Complex.I 

theorem complex_expression_evaluation : 
  ((2 + imaginary_i) / (1 - imaginary_i)) - (1 - imaginary_i) = -1/2 + (5/2) * imaginary_i :=
by 
  sorry

end complex_expression_evaluation_l169_169569


namespace Juan_run_time_l169_169284

theorem Juan_run_time
  (d : ℕ) (s : ℕ) (t : ℕ)
  (H1: d = 80)
  (H2: s = 10)
  (H3: t = d / s) :
  t = 8 := 
sorry

end Juan_run_time_l169_169284


namespace binom_eight_three_l169_169403

theorem binom_eight_three : Nat.choose 8 3 = 56 := by
  sorry

end binom_eight_three_l169_169403


namespace thrushes_left_l169_169507

theorem thrushes_left {init_thrushes : ℕ} (additional_thrushes : ℕ) (killed_ratio : ℚ) (killed : ℕ) (remaining : ℕ) :
  init_thrushes = 20 →
  additional_thrushes = 4 * 2 →
  killed_ratio = 1 / 7 →
  killed = killed_ratio * (init_thrushes + additional_thrushes) →
  remaining = init_thrushes + additional_thrushes - killed →
  remaining = 24 :=
by sorry

end thrushes_left_l169_169507


namespace equal_mass_piles_l169_169047

theorem equal_mass_piles (n : ℕ) (hn : n > 3) (hn_mod : n % 3 = 0 ∨ n % 3 = 2) : 
  ∃ A B C : Finset ℕ, A ∪ B ∪ C = {i | i ∈ Finset.range (n + 1)} ∧
  Disjoint A B ∧ Disjoint A C ∧ Disjoint B C ∧
  A.sum id = B.sum id ∧ B.sum id = C.sum id :=
sorry

end equal_mass_piles_l169_169047


namespace equal_chord_segments_l169_169965

theorem equal_chord_segments 
  (a x y : ℝ) 
  (AM CM : ℝ → ℝ → Prop) 
  (AB CD : ℝ → Prop)
  (intersect_chords_theorem : AM x (a - x) = CM y (a - y)) :
  x = y ∨ x = a - y :=
by
  sorry

end equal_chord_segments_l169_169965


namespace complement_intersection_l169_169588

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2 * x > 0}

def B : Set ℝ := {x | -3 < x ∧ x < 1}

def compA : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem complement_intersection :
  (compA ∩ B) = {x | 0 ≤ x ∧ x < 1} := by
  -- The proof goes here
  sorry

end complement_intersection_l169_169588


namespace team_members_count_l169_169214

theorem team_members_count (x : ℕ) (h1 : 3 * x + 2 * x = 33 ∨ 4 * x + 2 * x = 33) : x = 6 := by
  sorry

end team_members_count_l169_169214


namespace find_a3_a4_a5_l169_169164

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 2 * a n

noncomputable def sum_first_three (a : ℕ → ℝ) : Prop :=
a 0 + a 1 + a 2 = 21

theorem find_a3_a4_a5 (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : sum_first_three a) :
  a 2 + a 3 + a 4 = 84 :=
by
  sorry

end find_a3_a4_a5_l169_169164


namespace triangle_right_if_angle_difference_l169_169669

noncomputable def is_right_triangle (A B C : ℝ) : Prop := 
  A = 90

theorem triangle_right_if_angle_difference (A B C : ℝ) (h : A - B = C) (sum_angles : A + B + C = 180) :
  is_right_triangle A B C :=
  sorry

end triangle_right_if_angle_difference_l169_169669


namespace find_constants_l169_169710

theorem find_constants (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 → (x^2 - 7) / ((x - 2) * (x - 3) * (x - 5)) = A / (x - 2) + B / (x - 3) + C / (x - 5))
  ↔ (A = -1 ∧ B = -1 ∧ C = 3) :=
by
  sorry

end find_constants_l169_169710


namespace sum_of_consecutive_integers_l169_169622

theorem sum_of_consecutive_integers (x y z : ℤ) (h1 : y = x + 1) (h2 : z = y + 1) (h3 : z = 12) :
  x + y + z = 33 :=
sorry

end sum_of_consecutive_integers_l169_169622


namespace find_k_perpendicular_l169_169390

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (2, -3)

-- Define a function for the vector k * a - 2 * b
def vec_expression (k : ℝ) : ℝ × ℝ :=
  (k * vec_a.1 - 2 * vec_b.1, k * vec_a.2 - 2 * vec_b.2)

-- Prove that if the dot product of vec_expression k and vec_a is zero, then k = -1
theorem find_k_perpendicular (k : ℝ) :
  ((vec_expression k).1 * vec_a.1 + (vec_expression k).2 * vec_a.2 = 0) → k = -1 :=
by
  sorry

end find_k_perpendicular_l169_169390


namespace roots_of_equation_l169_169319

theorem roots_of_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end roots_of_equation_l169_169319


namespace coefficient_of_x_is_nine_l169_169469

theorem coefficient_of_x_is_nine (x : ℝ) (c : ℝ) (h : x = 0.5) (eq : 2 * x^2 + c * x - 5 = 0) : c = 9 :=
by
  sorry

end coefficient_of_x_is_nine_l169_169469


namespace intersection_S_T_eq_T_l169_169626

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l169_169626


namespace cheryl_walking_speed_l169_169698

theorem cheryl_walking_speed (H : 12 = 6 * v) : v = 2 := 
by
  -- proof here
  sorry

end cheryl_walking_speed_l169_169698


namespace staff_discount_l169_169249

theorem staff_discount (d : ℝ) (S : ℝ) (h1 : d > 0)
    (h2 : 0.455 * d = (1 - S / 100) * (0.65 * d)) : S = 30 := by
    sorry

end staff_discount_l169_169249


namespace white_tshirts_per_package_l169_169089

theorem white_tshirts_per_package (p t : ℕ) (h1 : p = 28) (h2 : t = 56) :
  t / p = 2 :=
by 
  sorry

end white_tshirts_per_package_l169_169089


namespace parallel_lines_have_equal_slopes_l169_169863

theorem parallel_lines_have_equal_slopes (m : ℝ) :
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0) → m = -1 / 2 :=
by
  sorry

end parallel_lines_have_equal_slopes_l169_169863


namespace sequence_an_solution_l169_169261

noncomputable def a_n (n : ℕ) : ℝ := (
  (1 / 2) * (2 + Real.sqrt 3)^n + 
  (1 / 2) * (2 - Real.sqrt 3)^n
)^2

theorem sequence_an_solution (n : ℕ) : 
  ∀ (a b : ℕ → ℝ),
  a 0 = 1 → 
  b 0 = 0 → 
  (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) → 
  (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) → 
  a n = a_n n := sorry

end sequence_an_solution_l169_169261


namespace neg_p_exists_x_l169_169989

-- Let p be the proposition: For all x in ℝ, x^2 - 3x + 3 > 0
def p : Prop := ∀ x : ℝ, x^2 - 3 * x + 3 > 0

-- Prove that the negation of p implies that there exists some x in ℝ such that x^2 - 3x + 3 ≤ 0
theorem neg_p_exists_x : ¬p ↔ ∃ x : ℝ, x^2 - 3 * x + 3 ≤ 0 :=
by {
  sorry
}

end neg_p_exists_x_l169_169989


namespace no_positive_integer_exists_l169_169751

theorem no_positive_integer_exists
  (P1 P2 : ℤ → ℤ)
  (a : ℤ)
  (h_a_neg : a < 0)
  (h_common_root : P1 a = 0 ∧ P2 a = 0) :
  ¬ ∃ b : ℤ, b > 0 ∧ P1 b = 2007 ∧ P2 b = 2008 :=
sorry

end no_positive_integer_exists_l169_169751


namespace number_of_bracelets_l169_169093

-- Define the conditions as constants
def metal_beads_nancy := 40
def pearl_beads_nancy := 60
def crystal_beads_rose := 20
def stone_beads_rose := 40
def beads_per_bracelet := 2

-- Define the number of sets each person can make
def sets_of_metal_beads := metal_beads_nancy / beads_per_bracelet
def sets_of_pearl_beads := pearl_beads_nancy / beads_per_bracelet
def sets_of_crystal_beads := crystal_beads_rose / beads_per_bracelet
def sets_of_stone_beads := stone_beads_rose / beads_per_bracelet

-- Define the theorem to prove
theorem number_of_bracelets : min sets_of_metal_beads (min sets_of_pearl_beads (min sets_of_crystal_beads sets_of_stone_beads)) = 10 := by
  -- Placeholder for the proof
  sorry

end number_of_bracelets_l169_169093


namespace least_positive_int_to_next_multiple_l169_169533

theorem least_positive_int_to_next_multiple (x : ℕ) (n : ℕ) (h : x = 365 ∧ n > 0) 
  (hm : (x + n) % 5 = 0) : n = 5 :=
by
  sorry

end least_positive_int_to_next_multiple_l169_169533


namespace r_n_m_smallest_m_for_r_2006_l169_169023

def euler_totient (n : ℕ) : ℕ := 
  n * (1 - (1 / 2)) * (1 - (1 / 17)) * (1 - (1 / 59))

def r (n m : ℕ) : ℕ :=
  m * euler_totient n

theorem r_n_m (n m : ℕ) : r n m = m * euler_totient n := 
  by sorry

theorem smallest_m_for_r_2006 (n m : ℕ) (h : n = 2006) (h2 : r n m = 841 * 928) : 
  ∃ m, r n m = 841^2 := 
  by sorry

end r_n_m_smallest_m_for_r_2006_l169_169023


namespace number_multiplies_p_plus_1_l169_169068

theorem number_multiplies_p_plus_1 (p q x : ℕ) 
  (hp : 1 < p) (hq : 1 < q)
  (hEq : x * (p + 1) = 25 * (q + 1))
  (hSum : p + q = 40) :
  x = 325 :=
sorry

end number_multiplies_p_plus_1_l169_169068


namespace area_triangle_QXY_l169_169769

-- Definition of the problem
def length_rectangle (PQ PS : ℝ) : Prop :=
  PQ = 8 ∧ PS = 6

def diagonal_division (PR : ℝ) (X Y : ℝ) : Prop :=
  PR = 10 ∧ X = 2.5 ∧ Y = 2.5

-- The statement we need to prove
theorem area_triangle_QXY
  (PQ PS PR X Y : ℝ)
  (h1 : length_rectangle PQ PS)
  (h2 : diagonal_division PR X Y)
  : ∃ (A : ℝ), A = 6 := by
  sorry

end area_triangle_QXY_l169_169769


namespace geometric_sequence_sum_l169_169525

theorem geometric_sequence_sum (a : ℕ → ℝ) (S₄ : ℝ) (S₈ : ℝ) (r : ℝ) 
    (h1 : r = 2) 
    (h2 : S₄ = a 0 + a 0 * r + a 0 * r^2 + a 0 * r^3)
    (h3 : S₄ = 1) 
    (h4 : S₈ = a 0 + a 0 * r + a 0 * r^2 + a 0 * r^3 + a 0 * r^4 + a 0 * r^5 + a 0 * r^6 + a 0 * r^7) :
    S₈ = 17 := by
  sorry

end geometric_sequence_sum_l169_169525


namespace maximum_profit_is_achieved_at_14_yuan_l169_169757

-- Define the initial conditions
def cost_per_unit : ℕ := 8
def initial_selling_price : ℕ := 10
def initial_selling_quantity : ℕ := 100

-- Define the sales volume decrease per price increase
def decrease_per_yuan_increase : ℕ := 10

-- Define the profit function
def profit (price_increase : ℕ) : ℕ :=
  let new_selling_price := initial_selling_price + price_increase
  let new_selling_quantity := initial_selling_quantity - (decrease_per_yuan_increase * price_increase)
  (new_selling_price - cost_per_unit) * new_selling_quantity

-- Define the statement to be proved
theorem maximum_profit_is_achieved_at_14_yuan :
  ∃ price_increase : ℕ, price_increase = 4 ∧ profit price_increase = profit 4 := by
  sorry

end maximum_profit_is_achieved_at_14_yuan_l169_169757


namespace initial_yards_lost_l169_169036

theorem initial_yards_lost (x : ℤ) (h : -x + 7 = 2) : x = 5 := by
  sorry

end initial_yards_lost_l169_169036


namespace combined_percentage_basketball_l169_169387

theorem combined_percentage_basketball (N_students : ℕ) (S_students : ℕ) 
  (N_percent_basketball : ℚ) (S_percent_basketball : ℚ) :
  N_students = 1800 → S_students = 3000 →
  N_percent_basketball = 0.25 → S_percent_basketball = 0.35 →
  ((N_students * N_percent_basketball) + (S_students * S_percent_basketball)) / (N_students + S_students) * 100 = 31 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  norm_num
  sorry

end combined_percentage_basketball_l169_169387


namespace initial_percentage_filled_l169_169582

theorem initial_percentage_filled (capacity : ℝ) (added : ℝ) (final_fraction : ℝ) (initial_water : ℝ) :
  capacity = 80 → added = 20 → final_fraction = 3/4 → 
  initial_water = (final_fraction * capacity - added) → 
  100 * (initial_water / capacity) = 50 :=
by
  intros
  sorry

end initial_percentage_filled_l169_169582


namespace perp_DM_PN_l169_169725

-- Definitions of the triangle and its elements
variables {A B C M N P D : Point}
variables (triangle_incircle_touch : ∀ (A B C : Point) (triangle : Triangle ABC),
  touches_incircle_at triangle B C M ∧ 
  touches_incircle_at triangle C A N ∧ 
  touches_incircle_at triangle A B P)
variables (point_D : lies_on_segment D N P)
variables {BD CD DP DN : ℝ}
variables (ratio_condition : DP / DN = BD / CD)

-- The theorem statement
theorem perp_DM_PN 
  (h1 : triangle_incircle_touch A B C) 
  (h2 : point_D)
  (h3 : ratio_condition) : 
  is_perpendicular D M P N := 
sorry

end perp_DM_PN_l169_169725


namespace intersection_of_sets_l169_169565

def A (x : ℝ) : Prop := x > -2
def B (x : ℝ) : Prop := 1 - x > 0

theorem intersection_of_sets :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | x > -2 ∧ x < 1} := by
  sorry

end intersection_of_sets_l169_169565


namespace eq_abs_distinct_solution_count_l169_169581

theorem eq_abs_distinct_solution_count :
  ∃! x : ℝ, |x - 10| = |x + 5| + 2 := 
sorry

end eq_abs_distinct_solution_count_l169_169581


namespace tan_A_area_triangle_ABC_l169_169933
open Real

-- Define the given conditions
def conditions (A : ℝ) (AC AB : ℝ) : Prop :=
  (sin A + cos A = sqrt 2 / 2) ∧ (AC = 2) ∧ (AB = 3)

-- State the first proof problem for tan A
theorem tan_A (A : ℝ) (hcond : conditions A 2 3) : tan A = -(2 + sqrt 3) := 
by 
  -- sorry for the proof placeholder
  sorry

-- State the second proof problem for the area of triangle ABC
theorem area_triangle_ABC (A B C : ℝ) (C_eq : C = 90) 
  (hcond : conditions A 2 3)
  (hBC : BC = sqrt ((AC^2) + (AB^2) - 2 * AC * AB * cos B)) : 
  (1/2) * AC * AB * sin A = (3 / 4) * (sqrt 6 + sqrt 2) := 
by 
  -- sorry for the proof placeholder
  sorry

end tan_A_area_triangle_ABC_l169_169933


namespace number_for_B_expression_l169_169738

-- Define the number for A as a variable
variable (a : ℤ)

-- Define the number for B in terms of a
def number_for_B (a : ℤ) : ℤ := 2 * a - 1

-- Statement to prove
theorem number_for_B_expression (a : ℤ) : number_for_B a = 2 * a - 1 := by
  sorry

end number_for_B_expression_l169_169738


namespace quadratic_root_sum_product_l169_169519

theorem quadratic_root_sum_product (m n : ℝ)
  (h1 : m + n = 4)
  (h2 : m * n = -1) :
  m + n - m * n = 5 :=
sorry

end quadratic_root_sum_product_l169_169519


namespace files_more_than_apps_l169_169724

def initial_apps : ℕ := 11
def initial_files : ℕ := 3
def remaining_apps : ℕ := 2
def remaining_files : ℕ := 24

theorem files_more_than_apps : remaining_files - remaining_apps = 22 :=
by
  sorry

end files_more_than_apps_l169_169724


namespace sunny_lead_l169_169731

-- Define the context of the race
variables {s m : ℝ}  -- s: Sunny's speed, m: Misty's speed
variables (distance_first : ℝ) (distance_ahead_first : ℝ)
variables (additional_distance_sunny_second : ℝ) (correct_answer : ℝ)

-- Given conditions
def conditions : Prop :=
  distance_first = 400 ∧
  distance_ahead_first = 20 ∧
  additional_distance_sunny_second = 40 ∧
  correct_answer = 20 

-- The math proof problem in Lean 4
theorem sunny_lead (h : conditions distance_first distance_ahead_first additional_distance_sunny_second correct_answer) :
  ∀ s m : ℝ, s / m = (400 / 380 : ℝ) → 
  (s / m) * 400 + additional_distance_sunny_second = (m / s) * 440 + correct_answer :=
sorry

end sunny_lead_l169_169731


namespace number_of_men_l169_169300

theorem number_of_men (M W C : ℕ) 
  (h1 : M + W + C = 10000)
  (h2 : C = 2500)
  (h3 : C = 5 * W) : 
  M = 7000 := 
by
  sorry

end number_of_men_l169_169300


namespace intersection_count_l169_169835

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_count : ∃! (x1 x2 : ℝ), 
  x1 > 0 ∧ x2 > 0 ∧ f x1 = g x1 ∧ f x2 = g x2 ∧ x1 ≠ x2 :=
sorry

end intersection_count_l169_169835


namespace expression_for_f_minimum_positive_period_of_f_range_of_f_l169_169170

noncomputable def f (x : ℝ) : ℝ :=
  let A := (2, 0) 
  let B := (0, 2)
  let C := (Real.cos (2 * x), Real.sin (2 * x))
  let AB := (B.1 - A.1, B.2 - A.2) 
  let AC := (C.1 - A.1, C.2 - A.2)
  AB.fst * AC.fst + AB.snd * AC.snd 

theorem expression_for_f (x : ℝ) :
  f x = 2 * Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4) + 4 :=
by sorry

theorem minimum_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
by sorry

theorem range_of_f (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) :
  2 < f x ∧ f x ≤ 4 + 2 * Real.sqrt 2 :=
by sorry

end expression_for_f_minimum_positive_period_of_f_range_of_f_l169_169170


namespace decrease_hours_worked_l169_169754

theorem decrease_hours_worked (initial_hourly_wage : ℝ) (initial_hours_worked : ℝ) :
  let new_hourly_wage := initial_hourly_wage * 1.25
  let new_hours_worked := (initial_hourly_wage * initial_hours_worked) / new_hourly_wage
  initial_hours_worked > 0 → 
  initial_hourly_wage > 0 → 
  new_hours_worked < initial_hours_worked :=
by
  intros initial_hours_worked_pos initial_hourly_wage_pos
  let new_hourly_wage := initial_hourly_wage * 1.25
  let new_hours_worked := (initial_hourly_wage * initial_hours_worked) / new_hourly_wage
  sorry

end decrease_hours_worked_l169_169754


namespace tiger_initial_leaps_behind_l169_169051

theorem tiger_initial_leaps_behind (tiger_leap_distance deer_leap_distance tiger_leaps_per_minute deer_leaps_per_minute total_distance_to_catch initial_leaps_behind : ℕ) 
  (h1 : tiger_leap_distance = 8) 
  (h2 : deer_leap_distance = 5) 
  (h3 : tiger_leaps_per_minute = 5) 
  (h4 : deer_leaps_per_minute = 4) 
  (h5 : total_distance_to_catch = 800) :
  initial_leaps_behind = 40 := 
by
  -- Leaving proof body incomplete as it is not required
  sorry

end tiger_initial_leaps_behind_l169_169051


namespace carol_lollipops_l169_169564

theorem carol_lollipops (total_lollipops : ℝ) (first_day_lollipops : ℝ) (delta_lollipops : ℝ) :
  total_lollipops = 150 → delta_lollipops = 5 →
  (first_day_lollipops + (first_day_lollipops + 5) + (first_day_lollipops + 10) +
  (first_day_lollipops + 15) + (first_day_lollipops + 20) + (first_day_lollipops + 25) = total_lollipops) →
  (first_day_lollipops = 12.5) →
  (first_day_lollipops + 15 = 27.5) :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end carol_lollipops_l169_169564


namespace unique_solution_inequality_l169_169061

theorem unique_solution_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a * x + a ∧ x^2 - a * x + a ≤ 1) → a = 2 :=
by
  sorry

end unique_solution_inequality_l169_169061


namespace grain_milling_necessary_pounds_l169_169106

theorem grain_milling_necessary_pounds (x : ℝ) (h : 0.90 * x = 100) : x = 111 + 1 / 9 := 
by
  sorry

end grain_milling_necessary_pounds_l169_169106


namespace compute_105_squared_l169_169155

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l169_169155


namespace rectangle_perimeter_l169_169241

theorem rectangle_perimeter (L W : ℝ) 
  (h1 : L - 4 = W + 3) 
  (h2 : (L - 4) * (W + 3) = L * W) : 
  2 * L + 2 * W = 50 :=
by
  -- Proving the theorem here
  sorry

end rectangle_perimeter_l169_169241


namespace ramola_rank_last_is_14_l169_169409

-- Define the total number of students
def total_students : ℕ := 26

-- Define Ramola's rank from the start
def ramola_rank_start : ℕ := 14

-- Define a function to calculate the rank from the last given the above conditions
def ramola_rank_from_last (total_students ramola_rank_start : ℕ) : ℕ :=
  total_students - ramola_rank_start + 1

-- Theorem stating that Ramola's rank from the last is 14th
theorem ramola_rank_last_is_14 :
  ramola_rank_from_last total_students ramola_rank_start = 14 :=
by
  -- Proof goes here
  sorry

end ramola_rank_last_is_14_l169_169409


namespace find_x_floor_l169_169719

theorem find_x_floor : ∃ (x : ℚ), (⌊x⌋ : ℚ) + x = 29 / 4 ∧ x = 29 / 4 := 
by
  sorry

end find_x_floor_l169_169719


namespace tan_45_eq_one_l169_169350

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l169_169350


namespace sum_first_75_odd_numbers_l169_169734

theorem sum_first_75_odd_numbers : (75^2) = 5625 :=
by
  sorry

end sum_first_75_odd_numbers_l169_169734


namespace candy_eating_l169_169491

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l169_169491


namespace identically_zero_on_interval_l169_169567

variable (f : ℝ → ℝ) (a b : ℝ)
variable (h_cont : ContinuousOn f (Set.Icc a b))
variable (h_int : ∀ n : ℕ, ∫ x in a..b, (x : ℝ)^n * f x = 0)

theorem identically_zero_on_interval : ∀ x ∈ Set.Icc a b, f x = 0 := 
by 
  sorry

end identically_zero_on_interval_l169_169567


namespace Carol_saves_9_per_week_l169_169132

variable (C : ℤ)

def Carol_savings (weeks : ℤ) : ℤ :=
  60 + weeks * C

def Mike_savings (weeks : ℤ) : ℤ :=
  90 + weeks * 3

theorem Carol_saves_9_per_week (h : Carol_savings C 5 = Mike_savings 5) : C = 9 :=
by
  dsimp [Carol_savings, Mike_savings] at h
  sorry

end Carol_saves_9_per_week_l169_169132


namespace roots_in_ap_difference_one_l169_169486

theorem roots_in_ap_difference_one :
  ∀ (r1 r2 r3 : ℝ), 
    64 * r1^3 - 144 * r1^2 + 92 * r1 - 15 = 0 ∧
    64 * r2^3 - 144 * r2^2 + 92 * r2 - 15 = 0 ∧
    64 * r3^3 - 144 * r3^2 + 92 * r3 - 15 = 0 ∧
    (r2 - r1 = r3 - r2) →
    max (max r1 r2) r3 - min (min r1 r2) r3 = 1 := 
by
  intros r1 r2 r3 h
  sorry

end roots_in_ap_difference_one_l169_169486


namespace gcf_factorial_5_6_l169_169244

theorem gcf_factorial_5_6 : Nat.gcd (Nat.factorial 5) (Nat.factorial 6) = Nat.factorial 5 := by
  sorry

end gcf_factorial_5_6_l169_169244


namespace find_x_l169_169369

noncomputable def angle_sum_triangle (A B C: ℝ) : Prop :=
  A + B + C = 180

noncomputable def vertical_angles_equal (A B: ℝ) : Prop :=
  A = B

noncomputable def right_angle_sum (D E: ℝ) : Prop :=
  D + E = 90

theorem find_x 
  (angle_ABC angle_BAC angle_DCE : ℝ) 
  (h1 : angle_ABC = 70)
  (h2 : angle_BAC = 50)
  (h3 : angle_sum_triangle angle_ABC angle_BAC angle_DCE)
  (h4 : vertical_angles_equal angle_DCE angle_DCE)
  (h5 : right_angle_sum angle_DCE 30) :
  angle_DCE = 60 :=
by
  sorry

end find_x_l169_169369


namespace sum_of_star_tip_angles_l169_169785

noncomputable def sum_star_tip_angles : ℝ :=
  let segment_angle := 360 / 8
  let subtended_arc := 3 * segment_angle
  let theta := subtended_arc / 2
  8 * theta

theorem sum_of_star_tip_angles:
  sum_star_tip_angles = 540 := by
  sorry

end sum_of_star_tip_angles_l169_169785


namespace C_plus_D_l169_169590

theorem C_plus_D (D C : ℚ) (h1 : ∀ x : ℚ, (Dx - 17) / ((x - 2) * (x - 4)) = C / (x - 2) + 4 / (x - 4))
  (h2 : ∀ x : ℚ, (x - 2) * (x - 4) = x^2 - 6 * x + 8) :
  C + D = 8.5 := sorry

end C_plus_D_l169_169590


namespace least_distinct_values_l169_169468

theorem least_distinct_values (lst : List ℕ) (h_len : lst.length = 2023) (h_mode : ∃ m, (∀ n ≠ m, lst.count n < lst.count m) ∧ lst.count m = 13) : ∃ x, x = 169 :=
by
  sorry

end least_distinct_values_l169_169468


namespace candle_lighting_time_l169_169404

theorem candle_lighting_time 
  (l : ℕ) -- initial length of the candles
  (t_diff : ℤ := 206) -- the time difference in minutes, correlating to 1:34 PM.
  : t_diff = 206 :=
by sorry

end candle_lighting_time_l169_169404


namespace total_days_2000_to_2003_correct_l169_169391

-- Define the days in each type of year
def days_in_leap_year : ℕ := 366
def days_in_common_year : ℕ := 365

-- Define each year and its corresponding number of days
def year_2000 := days_in_leap_year
def year_2001 := days_in_common_year
def year_2002 := days_in_common_year
def year_2003 := days_in_common_year

-- Calculate the total number of days from 2000 to 2003
def total_days_2000_to_2003 : ℕ := year_2000 + year_2001 + year_2002 + year_2003

theorem total_days_2000_to_2003_correct : total_days_2000_to_2003 = 1461 := 
by
  unfold total_days_2000_to_2003 year_2000 year_2001 year_2002 year_2003 
        days_in_leap_year days_in_common_year 
  exact rfl

end total_days_2000_to_2003_correct_l169_169391


namespace find_reciprocal_sum_of_roots_l169_169684

theorem find_reciprocal_sum_of_roots
  {x₁ x₂ : ℝ}
  (h1 : 5 * x₁ ^ 2 - 3 * x₁ - 2 = 0)
  (h2 : 5 * x₂ ^ 2 - 3 * x₂ - 2 = 0)
  (h_diff : x₁ ≠ x₂) :
  (1 / x₁ + 1 / x₂) = -3 / 2 :=
by {
  sorry
}

end find_reciprocal_sum_of_roots_l169_169684


namespace largest_square_perimeter_l169_169376

-- Define the conditions
def rectangle_length : ℕ := 80
def rectangle_width : ℕ := 60

-- Define the theorem to prove
theorem largest_square_perimeter : 4 * rectangle_width = 240 := by
  -- The proof steps are omitted
  sorry

end largest_square_perimeter_l169_169376


namespace odd_indexed_terms_geometric_sequence_l169_169165

open Nat

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (2 * n + 3) = r * a (2 * n + 1)

theorem odd_indexed_terms_geometric_sequence (b : ℕ → ℝ) (h : ∀ n, b n * b (n + 1) = 3 ^ n) :
  is_geometric_sequence b 3 :=
by
  sorry

end odd_indexed_terms_geometric_sequence_l169_169165


namespace complex_number_quadrant_l169_169592

theorem complex_number_quadrant :
  let z := (2 - (1 * Complex.I)) / (1 + (1 * Complex.I))
  (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end complex_number_quadrant_l169_169592


namespace find_pq_l169_169706

theorem find_pq (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (hline : ∀ x y : ℝ, px + qy = 24) 
  (harea : (1 / 2) * (24 / p) * (24 / q) = 48) : p * q = 12 :=
by
  sorry

end find_pq_l169_169706


namespace juniper_initial_bones_l169_169420

theorem juniper_initial_bones (B : ℕ) (h : 2 * B - 2 = 6) : B = 4 := 
by
  sorry

end juniper_initial_bones_l169_169420


namespace fraction_equivalent_to_decimal_l169_169944

theorem fraction_equivalent_to_decimal : 
  (0.4 -- using appropriate representation for repeating decimal 0.4\overline{13}
      + 13 / 990) = 409 / 990 ∧ Nat.gcd 409 990 = 1 := 
sorry

end fraction_equivalent_to_decimal_l169_169944


namespace simplify_expression_l169_169759

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b ^ 2 + 2 * b) - 4 * b ^ 2 = 9 * b ^ 3 + 2 * b ^ 2 :=
by
  sorry

end simplify_expression_l169_169759


namespace number_of_students_l169_169685

theorem number_of_students 
  (N : ℕ)
  (avg_age : ℕ → ℕ)
  (h1 : avg_age N = 15)
  (h2 : avg_age 5 = 12)
  (h3 : avg_age 9 = 16)
  (h4 : N = 15 ∧ avg_age 1 = 21) : 
  N = 15 :=
by
  sorry

end number_of_students_l169_169685


namespace div_identity_l169_169009

theorem div_identity (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := by
  sorry

end div_identity_l169_169009


namespace cost_of_each_book_is_six_l169_169282

-- Define variables for the number of books bought
def books_about_animals := 8
def books_about_outer_space := 6
def books_about_trains := 3

-- Define the total number of books
def total_books := books_about_animals + books_about_outer_space + books_about_trains

-- Define the total amount spent
def total_amount_spent := 102

-- Define the cost per book
def cost_per_book := total_amount_spent / total_books

-- Prove that the cost per book is $6
theorem cost_of_each_book_is_six : cost_per_book = 6 := by
  sorry

end cost_of_each_book_is_six_l169_169282


namespace circle_ways_l169_169920

noncomputable def count3ConsecutiveCircles : ℕ :=
  let longSideWays := 1 + 2 + 3 + 4 + 5 + 6
  let perpendicularWays := (4 + 4 + 4 + 3 + 2 + 1) * 2
  longSideWays + perpendicularWays

theorem circle_ways : count3ConsecutiveCircles = 57 := by
  sorry

end circle_ways_l169_169920


namespace correct_propositions_l169_169277

-- Definitions of parallel and perpendicular
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Main theorem
theorem correct_propositions (m n α β γ : Type) :
  ( (parallel m α ∧ parallel n β ∧ parallel α β → parallel m n) ∧
    (parallel α γ ∧ parallel β γ → parallel α β) ∧
    (perpendicular m α ∧ perpendicular n β ∧ parallel α β → parallel m n) ∧
    (perpendicular α γ ∧ perpendicular β γ → parallel α β) ) →
  ( (parallel α γ ∧ parallel β γ → parallel α β) ∧
    (perpendicular m α ∧ perpendicular n β ∧ parallel α β → parallel m n) ) :=
  sorry

end correct_propositions_l169_169277


namespace units_digit_3542_pow_876_l169_169630

theorem units_digit_3542_pow_876 : (3542 ^ 876) % 10 = 6 := by 
  sorry

end units_digit_3542_pow_876_l169_169630


namespace tan_20_plus_4_sin_20_eq_sqrt_3_l169_169532

theorem tan_20_plus_4_sin_20_eq_sqrt_3 : Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end tan_20_plus_4_sin_20_eq_sqrt_3_l169_169532


namespace task_D_cannot_be_sampled_l169_169824

def task_A := "Measuring the range of a batch of shells"
def task_B := "Determining the content of a certain microorganism in ocean waters"
def task_C := "Calculating the difficulty of each question on the math test after the college entrance examination"
def task_D := "Checking the height and weight of all sophomore students in a school"

def sampling_method (description: String) : Prop :=
  description = task_A ∨ description = task_B ∨ description = task_C

theorem task_D_cannot_be_sampled : ¬ sampling_method task_D :=
sorry

end task_D_cannot_be_sampled_l169_169824


namespace john_horizontal_distance_l169_169984

-- Define the conditions and the question
def elevation_initial : ℕ := 100
def elevation_final : ℕ := 1450
def vertical_to_horizontal_ratio (v h : ℕ) : Prop := v * 2 = h

-- Define the proof problem: the horizontal distance John moves
theorem john_horizontal_distance : ∃ h, vertical_to_horizontal_ratio (elevation_final - elevation_initial) h ∧ h = 2700 := 
by 
  sorry

end john_horizontal_distance_l169_169984


namespace perimeter_of_regular_nonagon_l169_169381

def regular_nonagon_side_length := 3
def number_of_sides := 9

theorem perimeter_of_regular_nonagon (h1 : number_of_sides = 9) (h2 : regular_nonagon_side_length = 3) :
  9 * 3 = 27 :=
by
  sorry

end perimeter_of_regular_nonagon_l169_169381


namespace stock_AB_increase_factor_l169_169686

-- Define the conditions as mathematical terms
def stock_A_initial := 300
def stock_B_initial := 300
def stock_C_initial := 300
def stock_C_final := stock_C_initial / 2
def total_final := 1350
def AB_combined_initial := stock_A_initial + stock_B_initial
def AB_combined_final := total_final - stock_C_final

-- The statement to prove that the factor by which stocks A and B increased in value is 2.
theorem stock_AB_increase_factor :
  AB_combined_final / AB_combined_initial = 2 :=
  by
    sorry

end stock_AB_increase_factor_l169_169686


namespace infinite_series_problem_l169_169171

noncomputable def infinite_series_sum : ℝ := ∑' n : ℕ, (2 * (n + 1)^2 - 3 * (n + 1) + 2) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2))

theorem infinite_series_problem :
  infinite_series_sum = -4 :=
by sorry

end infinite_series_problem_l169_169171


namespace find_x_l169_169310

def digit_sum (n : ℕ) : ℕ := 
  n.digits 10 |> List.sum

def k := (10^45 - 999999999999999999999999999999999999999999994 : ℕ)

theorem find_x :
  digit_sum k = 397 := 
sorry

end find_x_l169_169310


namespace solve_for_y_l169_169329

theorem solve_for_y (y : ℕ) : 9^y = 3^12 → y = 6 :=
by
  sorry

end solve_for_y_l169_169329


namespace gumballs_result_l169_169378

def gumballs_after_sharing_equally (initial_joanna : ℕ) (initial_jacques : ℕ) (multiplier : ℕ) : ℕ :=
  let joanna_total := initial_joanna + initial_joanna * multiplier
  let jacques_total := initial_jacques + initial_jacques * multiplier
  (joanna_total + jacques_total) / 2

theorem gumballs_result :
  gumballs_after_sharing_equally 40 60 4 = 250 :=
by
  sorry

end gumballs_result_l169_169378


namespace problem_7_sqrt_13_l169_169664

theorem problem_7_sqrt_13 : 
  let m := Int.floor (Real.sqrt 13)
  let n := 10 - Real.sqrt 13 - Int.floor (10 - Real.sqrt 13)
  m + n = 7 - Real.sqrt 13 :=
by
  sorry

end problem_7_sqrt_13_l169_169664


namespace member_sum_of_two_others_l169_169941

def numMembers : Nat := 1978
def numCountries : Nat := 6

theorem member_sum_of_two_others :
  ∃ m : ℕ, m ∈ Finset.range numMembers.succ ∧
  ∃ a b : ℕ, a ∈ Finset.range numMembers.succ ∧ b ∈ Finset.range numMembers.succ ∧ 
  ∃ country : Fin (numCountries + 1), (a = m + b ∧ country = country) :=
by
  sorry

end member_sum_of_two_others_l169_169941


namespace find_value_of_x_l169_169838

theorem find_value_of_x :
  (0.47 * 1442 - 0.36 * 1412) + 65 = 234.42 := 
by
  sorry

end find_value_of_x_l169_169838


namespace travel_distance_l169_169753

noncomputable def distance_traveled (AB BC : ℝ) : ℝ :=
  let BD := Real.sqrt (AB^2 + BC^2)
  let arc1 := (2 * Real.pi * BD) / 4
  let arc2 := (2 * Real.pi * AB) / 4
  arc1 + arc2

theorem travel_distance (hAB : AB = 3) (hBC : BC = 4) : 
  distance_traveled AB BC = 4 * Real.pi := by
    sorry

end travel_distance_l169_169753


namespace max_x_y3_z4_l169_169451

noncomputable def max_value_expression (x y z : ℝ) : ℝ :=
  x + y^3 + z^4

theorem max_x_y3_z4 (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  max_value_expression x y z ≤ 1 :=
sorry

end max_x_y3_z4_l169_169451


namespace smallest_x_for_multiple_l169_169809

theorem smallest_x_for_multiple (x : ℕ) (h720 : 720 = 2^4 * 3^2 * 5) (h1250 : 1250 = 2 * 5^4) : 
  (∃ x, (x > 0) ∧ (1250 ∣ (720 * x))) → x = 125 :=
by
  sorry

end smallest_x_for_multiple_l169_169809


namespace common_diff_necessary_sufficient_l169_169421

section ArithmeticSequence

variable {α : Type*} [OrderedAddCommGroup α] {a : ℕ → α} {d : α}

-- Define an arithmetic sequence with common difference d
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Prove that d > 0 is the necessary and sufficient condition for a_2 > a_1
theorem common_diff_necessary_sufficient (a : ℕ → α) (d : α) :
    (is_arithmetic_sequence a d) → (d > 0 ↔ a 2 > a 1) :=
by
  sorry

end ArithmeticSequence

end common_diff_necessary_sufficient_l169_169421


namespace jean_total_jail_time_l169_169789

def arson_counts := 3
def burglary_counts := 2
def petty_larceny_multiplier := 6
def arson_sentence_per_count := 36
def burglary_sentence_per_count := 18
def petty_larceny_fraction := 1/3

def total_jail_time :=
  arson_counts * arson_sentence_per_count +
  burglary_counts * burglary_sentence_per_count +
  (petty_larceny_multiplier * burglary_counts) * (petty_larceny_fraction * burglary_sentence_per_count)

theorem jean_total_jail_time : total_jail_time = 216 :=
by
  sorry

end jean_total_jail_time_l169_169789


namespace range_of_m_l169_169152

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) (hf : ∀ x, -1 ≤ x ∧ x ≤ 1 → ∃ y, f y = x) :
  (∀ x, ∃ y, y = f (x + m) - f (x - m)) →
  -1 ≤ m ∧ m ≤ 1 :=
by
  intro hF
  sorry

end range_of_m_l169_169152


namespace greatest_possible_integer_l169_169666

theorem greatest_possible_integer (m : ℕ) (h1 : m < 150) (h2 : ∃ a : ℕ, m = 10 * a - 2) (h3 : ∃ b : ℕ, m = 9 * b - 4) : m = 68 := 
  by sorry

end greatest_possible_integer_l169_169666


namespace avg_zits_per_kid_mr_jones_class_l169_169620

-- Define the conditions
def avg_zits_ms_swanson_class := 5
def num_kids_ms_swanson_class := 25
def num_kids_mr_jones_class := 32
def extra_zits_mr_jones_class := 67

-- Define the total number of zits in Ms. Swanson's class
def total_zits_ms_swanson_class := avg_zits_ms_swanson_class * num_kids_ms_swanson_class

-- Define the total number of zits in Mr. Jones' class
def total_zits_mr_jones_class := total_zits_ms_swanson_class + extra_zits_mr_jones_class

-- Define the problem statement to prove: the average number of zits per kid in Mr. Jones' class
theorem avg_zits_per_kid_mr_jones_class : 
  total_zits_mr_jones_class / num_kids_mr_jones_class = 6 := by
  sorry

end avg_zits_per_kid_mr_jones_class_l169_169620


namespace quadratic_solution_l169_169869

theorem quadratic_solution (x : ℝ) : -x^2 + 4 * x + 5 < 0 ↔ x > 5 ∨ x < -1 :=
sorry

end quadratic_solution_l169_169869


namespace least_number_divisible_by_digits_and_5_l169_169950

/-- Define a predicate to check if a number is divisible by all of its digits -/
def divisible_by_digits (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10 % 10, n / 10 % 10, n % 10]
  ∀ d ∈ digits, d ≠ 0 → n % d = 0

/-- Define the main theorem stating the least four-digit number divisible by 5 and each of its digits is 1425 -/
theorem least_number_divisible_by_digits_and_5 
  (n : ℕ) (hn : 1000 ≤ n ∧ n < 10000)
  (hd : (∀ i j : ℕ, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)))
  (hdiv5 : n % 5 = 0)
  (hdiv_digits : divisible_by_digits n) 
  : n = 1425 :=
sorry

end least_number_divisible_by_digits_and_5_l169_169950


namespace candidate_knows_Excel_and_willing_nights_l169_169033

variable (PExcel PXNight : ℝ)
variable (H1 : PExcel = 0.20) (H2 : PXNight = 0.30)

theorem candidate_knows_Excel_and_willing_nights : (PExcel * PXNight) = 0.06 :=
by
  rw [H1, H2]
  norm_num

end candidate_knows_Excel_and_willing_nights_l169_169033


namespace percentage_increase_correct_l169_169864

variable {R1 E1 P1 R2 E2 P2 R3 E3 P3 : ℝ}

-- Conditions
axiom H1 : P1 = R1 - E1
axiom H2 : R2 = 1.20 * R1
axiom H3 : E2 = 1.10 * E1
axiom H4 : P2 = R2 - E2
axiom H5 : P2 = 1.15 * P1
axiom H6 : R3 = 1.25 * R2
axiom H7 : E3 = 1.20 * E2
axiom H8 : P3 = R3 - E3
axiom H9 : P3 = 1.35 * P2

theorem percentage_increase_correct :
  ((P3 - P1) / P1) * 100 = 55.25 :=
by sorry

end percentage_increase_correct_l169_169864


namespace intersection_S_T_l169_169606

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l169_169606


namespace non_poli_sci_gpa_below_or_eq_3_is_10_l169_169659

-- Definitions based on conditions
def total_applicants : ℕ := 40
def poli_sci_majors : ℕ := 15
def gpa_above_3 : ℕ := 20
def poli_sci_gpa_above_3 : ℕ := 5

-- Derived conditions from the problem
def poli_sci_gpa_below_or_eq_3 : ℕ := poli_sci_majors - poli_sci_gpa_above_3
def total_gpa_below_or_eq_3 : ℕ := total_applicants - gpa_above_3
def non_poli_sci_gpa_below_or_eq_3 : ℕ := total_gpa_below_or_eq_3 - poli_sci_gpa_below_or_eq_3

-- Statement to be proven
theorem non_poli_sci_gpa_below_or_eq_3_is_10 : non_poli_sci_gpa_below_or_eq_3 = 10 := by
  sorry

end non_poli_sci_gpa_below_or_eq_3_is_10_l169_169659


namespace gcd_45_75_l169_169800

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l169_169800


namespace inequality_condition_l169_169021

theorem inequality_condition {a b x y : ℝ} (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) : 
  (a^2 / x) + (b^2 / y) ≥ ((a + b)^2 / (x + y)) ∧ (a^2 / x) + (b^2 / y) = ((a + b)^2 / (x + y)) ↔ (x / y) = (a / b) :=
sorry

end inequality_condition_l169_169021


namespace largest_N_cannot_pay_exactly_without_change_l169_169766

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l169_169766


namespace not_all_crows_gather_on_one_tree_l169_169711

theorem not_all_crows_gather_on_one_tree :
  ∀ (crows : Fin 6 → ℕ), 
  (∀ i, crows i = 1) →
  (∀ t1 t2, abs (t1 - t2) = 1 → crows t1 = crows t1 - 1 ∧ crows t2 = crows t2 + 1) →
  ¬(∃ i, crows i = 6 ∧ (∀ j ≠ i, crows j = 0)) :=
by
  sorry

end not_all_crows_gather_on_one_tree_l169_169711


namespace eliza_irons_dress_in_20_minutes_l169_169978

def eliza_iron_time : Prop :=
∃ d : ℕ, 
  (d ≠ 0 ∧  -- To avoid division by zero
  8 + 180 / d = 17 ∧
  d = 20)

theorem eliza_irons_dress_in_20_minutes : eliza_iron_time :=
sorry

end eliza_irons_dress_in_20_minutes_l169_169978


namespace prize_selection_count_l169_169446

theorem prize_selection_count :
  (Nat.choose 20 1) * (Nat.choose 19 2) * (Nat.choose 17 4) = 8145600 := 
by 
  sorry

end prize_selection_count_l169_169446


namespace increase_is_50_percent_l169_169658

theorem increase_is_50_percent (original new : ℕ) (h1 : original = 60) (h2 : new = 90) :
  ((new - original) * 100 / original) = 50 :=
by
  -- Proof can be filled here.
  sorry

end increase_is_50_percent_l169_169658


namespace man_speed_with_current_l169_169579

-- Define the conditions
def current_speed : ℕ := 3
def man_speed_against_current : ℕ := 14

-- Define the man's speed in still water (v) based on the given speed against the current
def man_speed_in_still_water : ℕ := man_speed_against_current + current_speed

-- Prove that the man's speed with the current is 20 kmph
theorem man_speed_with_current : man_speed_in_still_water + current_speed = 20 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end man_speed_with_current_l169_169579


namespace range_of_a_l169_169991

noncomputable def f (a x : ℝ) : ℝ := min (Real.exp x - 2) (Real.exp (2 * x) - a * Real.exp x + a + 24)

def has_three_zeros (f : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0

theorem range_of_a (a : ℝ) :
  has_three_zeros (f a) ↔ 12 < a ∧ a < 28 :=
sorry

end range_of_a_l169_169991


namespace largest_integer_solution_l169_169250

theorem largest_integer_solution (x : ℤ) (h : 3 - 2 * x > 0) : x ≤ 1 :=
by sorry

end largest_integer_solution_l169_169250


namespace cafeteria_sales_comparison_l169_169814

theorem cafeteria_sales_comparison
  (S : ℝ) -- initial sales
  (a : ℝ) -- monthly increment for Cafeteria A
  (p : ℝ) -- monthly percentage increment for Cafeteria B
  (h1 : S > 0) -- initial sales are positive
  (h2 : a > 0) -- constant increment for Cafeteria A is positive
  (h3 : p > 0) -- constant percentage increment for Cafeteria B is positive
  (h4 : S + 8 * a = S * (1 + p) ^ 8) -- sales are equal in September 2013
  (h5 : S = S) -- sales are equal in January 2013 (trivially true)
  : S + 4 * a > S * (1 + p) ^ 4 := 
sorry

end cafeteria_sales_comparison_l169_169814


namespace area_of_square_B_l169_169208

theorem area_of_square_B (c : ℝ) (hA : ∃ sA, sA * sA = 2 * c^2) (hB : ∃ sA, exists sB, sB * sB = 3 * (sA * sA)) : 
∃ sB, sB * sB = 6 * c^2 :=
by
  sorry

end area_of_square_B_l169_169208


namespace minimum_value_of_f_on_neg_interval_l169_169932

theorem minimum_value_of_f_on_neg_interval (f : ℝ → ℝ) 
    (h_even : ∀ x, f (-x) = f x) 
    (h_increasing : ∀ x y, 1 ≤ x → x ≤ y → y ≤ 2 → f x ≤ f y) 
  : ∀ x, -2 ≤ x → x ≤ -1 → f (-1) ≤ f x := 
by
  sorry

end minimum_value_of_f_on_neg_interval_l169_169932


namespace triangle_inequality_satisfied_for_n_six_l169_169364

theorem triangle_inequality_satisfied_for_n_six :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ a + c > b ∧ b + c > a) := sorry

end triangle_inequality_satisfied_for_n_six_l169_169364


namespace students_in_diligence_before_transfer_l169_169053

theorem students_in_diligence_before_transfer (D I P : ℕ)
  (h_total : D + I + P = 75)
  (h_equal : D + 2 = I - 2 + 3 ∧ D + 2 = P - 3) :
  D = 23 :=
by
  sorry

end students_in_diligence_before_transfer_l169_169053


namespace students_taking_history_but_not_statistics_l169_169041

-- Definitions based on conditions
def T : Nat := 150
def H : Nat := 58
def S : Nat := 42
def H_union_S : Nat := 95

-- Statement to prove
theorem students_taking_history_but_not_statistics : H - (H + S - H_union_S) = 53 :=
by
  sorry

end students_taking_history_but_not_statistics_l169_169041


namespace perimeter_of_octagon_l169_169580

theorem perimeter_of_octagon :
  let base := 10
  let left_side := 9
  let right_side := 11
  let top_left_diagonal := 6
  let top_right_diagonal := 7
  let small_side1 := 2
  let small_side2 := 3
  let small_side3 := 4
  base + left_side + right_side + top_left_diagonal + top_right_diagonal + small_side1 + small_side2 + small_side3 = 52 :=
by
  -- This automatically assumes all the definitions and shows the equation
  sorry

end perimeter_of_octagon_l169_169580


namespace tap_b_fill_time_l169_169343

theorem tap_b_fill_time (t : ℝ) (h1 : t > 0) : 
  (∀ (A_fill B_fill together_fill : ℝ), 
    A_fill = 1/45 ∧ 
    B_fill = 1/t ∧ 
    together_fill = A_fill + B_fill ∧ 
    (9 * A_fill) + (23 * B_fill) = 1) → 
    t = 115 / 4 :=
by
  sorry

end tap_b_fill_time_l169_169343


namespace part1_part2_part3_l169_169030

-- Part 1: Proving a₁ for given a₃, p, and q
theorem part1 (a : ℕ → ℝ) (p q : ℝ) (h1 : p = (1/2)) (h2 : q = 2) 
  (h3 : a 3 = 41 / 20) (h4 : ∀ n, a (n + 1) = p * a n + q / a n) :
  a 1 = 1 ∨ a 1 = 4 := 
sorry

-- Part 2: Finding the sum Sₙ of the first n terms given a₁ and p * q = 0
theorem part2 (a : ℕ → ℝ) (p q : ℝ) (h1 : a 1 = 5) (h2 : p * q = 0) 
  (h3 : ∀ n, a (n + 1) = p * a n + q / a n) 
  (S : ℕ → ℝ) (n : ℕ) :
    S n = (25 * n + q * n + q - 25) / 10 ∨ 
    S n = (25 * n + q * n) / 10 ∨ 
    S n = (5 * (p^n - 1)) / (p - 1) ∨ 
    S n = 5 * n :=
sorry

-- Part 3: Proving the range of p given a₁, q and that the sequence is monotonically decreasing
theorem part3 (a : ℕ → ℝ) (p q : ℝ) (h1 : a 1 = 2) (h2 : q = 1) 
  (h3 : ∀ n, a (n + 1) = p * a n + q / a n) 
  (h4 : ∀ n, a (n + 1) < a n) :
  1/2 < p ∧ p < 3/4 :=
sorry

end part1_part2_part3_l169_169030


namespace gillian_more_than_three_times_sandi_l169_169798

-- Definitions of the conditions
def sandi_initial : ℕ := 600
def sandi_spent : ℕ := sandi_initial / 2
def gillian_spent : ℕ := 1050
def three_times_sandi_spent : ℕ := 3 * sandi_spent

-- Theorem statement with the proof to be added
theorem gillian_more_than_three_times_sandi :
  gillian_spent - three_times_sandi_spent = 150 := 
sorry

end gillian_more_than_three_times_sandi_l169_169798


namespace correctness_of_option_C_l169_169298

noncomputable def vec_a : ℝ × ℝ := (-1/2, Real.sqrt 3 / 2)
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3 / 2, -1/2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def is_orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

theorem correctness_of_option_C :
  is_orthogonal (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2) :=
by
  sorry

end correctness_of_option_C_l169_169298


namespace minimum_n_minus_m_abs_l169_169677

theorem minimum_n_minus_m_abs (f g : ℝ → ℝ)
  (hf : ∀ x, f x = Real.exp x + 2 * x)
  (hg : ∀ x, g x = 4 * x)
  (m n : ℝ)
  (h_cond : f m = g n) :
  |n - m| = (1 / 2) - (1 / 2) * Real.log 2 := 
sorry

end minimum_n_minus_m_abs_l169_169677


namespace students_on_bleachers_l169_169816

theorem students_on_bleachers (F B : ℕ) (h1 : F + B = 26) (h2 : F / (F + B) = 11 / 13) : B = 4 :=
by sorry

end students_on_bleachers_l169_169816


namespace cube_red_faces_one_third_l169_169508

theorem cube_red_faces_one_third (n : ℕ) (h : 6 * n^3 ≠ 0) : 
  (2 * n^2) / (6 * n^3) = 1 / 3 → n = 1 :=
by sorry

end cube_red_faces_one_third_l169_169508


namespace solve_first_equation_solve_second_equation_l169_169619

theorem solve_first_equation (x : ℝ) : (8 * x = -2 * (x + 5)) → (x = -1) :=
by
  intro h
  sorry

theorem solve_second_equation (x : ℝ) : ((x - 1) / 4 = (5 * x - 7) / 6 + 1) → (x = -1 / 7) :=
by
  intro h
  sorry

end solve_first_equation_solve_second_equation_l169_169619


namespace sin_x1_x2_value_l169_169860

open Real

theorem sin_x1_x2_value (m x1 x2 : ℝ) :
  (2 * sin (2 * x1) + cos (2 * x1) = m) →
  (2 * sin (2 * x2) + cos (2 * x2) = m) →
  (0 ≤ x1 ∧ x1 ≤ π / 2) →
  (0 ≤ x2 ∧ x2 ≤ π / 2) →
  sin (x1 + x2) = 2 * sqrt 5 / 5 := 
by
  sorry

end sin_x1_x2_value_l169_169860


namespace max_distance_sum_l169_169807

theorem max_distance_sum {P : ℝ × ℝ} 
  (C : Set (ℝ × ℝ)) 
  (hC : ∀ (P : ℝ × ℝ), P ∈ C ↔ (P.1 - 3)^2 + (P.2 - 4)^2 = 1)
  (A : ℝ × ℝ := (0, -1))
  (B : ℝ × ℝ := (0, 1)) :
  ∃ P : ℝ × ℝ, 
    P ∈ C ∧ (P = (18 / 5, 24 / 5)) :=
by
  sorry

end max_distance_sum_l169_169807


namespace compute_g_neg_101_l169_169876

noncomputable def g (x : ℝ) : ℝ := sorry

theorem compute_g_neg_101 (g_condition : ∀ x y : ℝ, g (x * y) + x = x * g y + g x)
                         (g1 : g 1 = 7) :
    g (-101) = -95 := 
by 
  sorry

end compute_g_neg_101_l169_169876


namespace num_parallel_edge_pairs_correct_l169_169673

-- Define a rectangular prism with given dimensions
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

-- Function to count the number of pairs of parallel edges
def num_parallel_edge_pairs (p : RectangularPrism) : ℕ :=
  4 * ((p.length + p.width + p.height) - 3)

-- Given conditions
def given_prism : RectangularPrism := { length := 4, width := 3, height := 2 }

-- Main theorem statement
theorem num_parallel_edge_pairs_correct :
  num_parallel_edge_pairs given_prism = 12 :=
by
  -- Skipping proof steps
  sorry

end num_parallel_edge_pairs_correct_l169_169673


namespace contrapositive_lemma_l169_169880

theorem contrapositive_lemma (a : ℝ) (h : a^2 ≤ 9) : a < 4 := 
sorry

end contrapositive_lemma_l169_169880


namespace exists_circular_chain_of_four_l169_169422

-- Let A and B be the two teams, each with a set of players.
variable {A B : Type}
-- Assume there exists a relation "beats" that determines match outcomes.
variable (beats : A → B → Prop)

-- Each player in both teams has at least one win and one loss against the opposite team.
axiom each_has_win_and_loss (a : A) : ∃ b1 b2 : B, beats a b1 ∧ ¬beats a b2 ∧ b1 ≠ b2
axiom each_has_win_and_loss' (b : B) : ∃ a1 a2 : A, beats a1 b ∧ ¬beats a2 b ∧ a1 ≠ a2

-- Main theorem: Exist four players forming a circular chain of victories.
theorem exists_circular_chain_of_four :
  ∃ (a1 a2 : A) (b1 b2 : B), beats a1 b1 ∧ ¬beats a1 b2 ∧ beats a2 b2 ∧ ¬beats a2 b1 ∧ b1 ≠ b2 ∧ a1 ≠ a2 :=
sorry

end exists_circular_chain_of_four_l169_169422


namespace volunteer_hours_per_year_l169_169670

def volunteer_sessions_per_month := 2
def hours_per_session := 3
def months_per_year := 12

theorem volunteer_hours_per_year : 
  (volunteer_sessions_per_month * months_per_year * hours_per_session) = 72 := 
by
  sorry

end volunteer_hours_per_year_l169_169670


namespace solve_for_y_l169_169049

-- Define the variables and conditions
variable (y : ℝ)
variable (h_pos : y > 0)
variable (h_seq : (4 + y^2 = 2 * y^2 ∧ y^2 + 25 = 2 * y^2))

-- State the theorem
theorem solve_for_y : y = Real.sqrt 14.5 :=
by sorry

end solve_for_y_l169_169049


namespace relationship_C1_C2_A_l169_169614

variables (A B C C1 C2 : ℝ)

-- Given conditions
def TriangleABC : Prop := B = 2 * A
def AngleSumProperty : Prop := A + B + C = 180
def AltitudeDivides := C1 = 90 - A ∧ C2 = 90 - 2 * A

-- Theorem to prove the relationship between C1, C2, and A
theorem relationship_C1_C2_A (h1: TriangleABC A B) (h2: AngleSumProperty A B C) (h3: AltitudeDivides C1 C2 A) : 
  C1 - C2 = A :=
by sorry

end relationship_C1_C2_A_l169_169614


namespace quadratic_root_property_l169_169733

theorem quadratic_root_property (m p : ℝ) 
  (h1 : (p^2 - 2 * p + m - 1 = 0)) 
  (h2 : (p^2 - 2 * p + 3) * (m + 4) = 7)
  (h3 : ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 - 2 * r1 + m - 1 = 0 ∧ r2^2 - 2 * r2 + m - 1 = 0) : 
  m = -3 :=
by 
  sorry

end quadratic_root_property_l169_169733


namespace sum_of_positive_factors_36_l169_169386

def sum_of_divisors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (fun m => n % m = 0) |>.sum

theorem sum_of_positive_factors_36 : sum_of_divisors 36 = 91 := by
  sorry

end sum_of_positive_factors_36_l169_169386


namespace P_eq_Q_l169_169924

open Set Real

def P : Set ℝ := {m | -1 < m ∧ m ≤ 0}
def Q : Set ℝ := {m | ∀ (x : ℝ), m * x^2 + 4 * m * x - 4 < 0}

theorem P_eq_Q : P = Q :=
by
  sorry

end P_eq_Q_l169_169924


namespace D_is_painting_l169_169272

def A_activity (act : String) : Prop := 
  act ≠ "walking" ∧ act ≠ "playing basketball"

def B_activity (act : String) : Prop :=
  act ≠ "dancing" ∧ act ≠ "running"

def C_activity_implies_A_activity (C_act A_act : String) : Prop :=
  C_act = "walking" → A_act = "dancing"

def D_activity (act : String) : Prop :=
  act ≠ "playing basketball" ∧ act ≠ "running"

def C_activity (act : String) : Prop :=
  act ≠ "dancing" ∧ act ≠ "playing basketball"

theorem D_is_painting :
  (∃ a b c d : String,
    A_activity a ∧
    B_activity b ∧
    C_activity_implies_A_activity c a ∧
    D_activity d ∧
    C_activity c) →
  ∃ d : String, d = "painting" :=
by
  intros h
  sorry

end D_is_painting_l169_169272


namespace min_value_l169_169418

noncomputable def min_value_expr (a b c d : ℝ) : ℝ :=
  (a + b) / c + (b + c) / a + (c + d) / b

theorem min_value 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min_value_expr a b c d ≥ 6 
  := sorry

end min_value_l169_169418


namespace cups_of_rice_in_afternoon_l169_169423

-- Definitions for conditions
def morning_cups : ℕ := 3
def evening_cups : ℕ := 5
def fat_per_cup : ℕ := 10
def weekly_total_fat : ℕ := 700

-- Theorem statement
theorem cups_of_rice_in_afternoon (morning_cups evening_cups fat_per_cup weekly_total_fat : ℕ) :
  (weekly_total_fat - (morning_cups + evening_cups) * fat_per_cup * 7) / fat_per_cup = 14 :=
by
  sorry

end cups_of_rice_in_afternoon_l169_169423


namespace pete_miles_walked_l169_169505

noncomputable def steps_from_first_pedometer (flips1 : ℕ) (final_reading1 : ℕ) : ℕ :=
  flips1 * 100000 + final_reading1 

noncomputable def steps_from_second_pedometer (flips2 : ℕ) (final_reading2 : ℕ) : ℕ :=
  flips2 * 400000 + final_reading2 * 4

noncomputable def total_steps (flips1 flips2 final_reading1 final_reading2 : ℕ) : ℕ :=
  steps_from_first_pedometer flips1 final_reading1 + steps_from_second_pedometer flips2 final_reading2

noncomputable def miles_walked (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

theorem pete_miles_walked
  (flips1 flips2 final_reading1 final_reading2 steps_per_mile : ℕ)
  (h_flips1 : flips1 = 50)
  (h_final_reading1 : final_reading1 = 25000)
  (h_flips2 : flips2 = 15)
  (h_final_reading2 : final_reading2 = 30000)
  (h_steps_per_mile : steps_per_mile = 1500) :
  miles_walked (total_steps flips1 flips2 final_reading1 final_reading2) steps_per_mile = 7430 :=
by sorry

end pete_miles_walked_l169_169505


namespace charles_ate_no_bananas_l169_169979

theorem charles_ate_no_bananas (W C B : ℝ) (h1 : W = 48) (h2 : C = 35) (h3 : W + C = 83) : B = 0 :=
by
  -- Proof goes here
  sorry

end charles_ate_no_bananas_l169_169979


namespace part_one_solution_set_part_two_lower_bound_l169_169085

def f (x a b : ℝ) : ℝ := abs (x - a) + abs (x + b)

-- Part (I)
theorem part_one_solution_set (a b x : ℝ) (h1 : a = 1) (h2 : b = 2) :
  (f x a b ≤ 5) ↔ -3 ≤ x ∧ x ≤ 2 := by
  rw [h1, h2]
  sorry

-- Part (II)
theorem part_two_lower_bound (a b x : ℝ) (h : a > 0) (h' : b > 0) (h'' : a + 4 * b = 2 * a * b) :
  f x a b ≥ 9 / 2 := by
  sorry

end part_one_solution_set_part_two_lower_bound_l169_169085


namespace all_visitors_can_buy_ticket_l169_169654

-- Define the coin types
inductive Coin
  | Three
  | Five

-- Define a function to calculate the total money from a list of coins
def totalMoney (coins : List Coin) : Int :=
  coins.foldr (fun c acc => acc + (match c with | Coin.Three => 3 | Coin.Five => 5)) 0

-- Define the initial state: each person has 22 tugriks in some combination of 3 and 5 tugrik coins
def initial_money := 22
def ticket_cost := 4

-- Each visitor and the cashier has 22 tugriks initially
axiom visitor_money_all_22 (n : Nat) : n ≤ 200 → totalMoney (List.replicate 2 Coin.Five ++ List.replicate 4 Coin.Three) = initial_money

-- We want to prove that all visitors can buy a ticket
theorem all_visitors_can_buy_ticket :
  ∀ n, n ≤ 200 → ∃ coins: List Coin, totalMoney coins = initial_money ∧ totalMoney coins ≥ ticket_cost := by
    sorry -- Proof goes here

end all_visitors_can_buy_ticket_l169_169654


namespace simplify_exponent_product_l169_169537

theorem simplify_exponent_product :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_product_l169_169537


namespace rectangle_perimeter_l169_169200

theorem rectangle_perimeter 
(area : ℝ) (width : ℝ) (h1 : area = 200) (h2 : width = 10) : 
    ∃ (perimeter : ℝ), perimeter = 60 :=
by
  sorry

end rectangle_perimeter_l169_169200


namespace minimum_value_expr_l169_169028

theorem minimum_value_expr (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  (1 + (1 / m)) * (1 + (1 / n)) = 9 :=
sorry

end minimum_value_expr_l169_169028


namespace fraction_simplification_l169_169963

theorem fraction_simplification 
  (a b c : ℝ)
  (h₀ : a ≠ 0) 
  (h₁ : b ≠ 0) 
  (h₂ : c ≠ 0) 
  (h₃ : a^2 + b^2 + c^2 ≠ 0) :
  (a^2 * b^2 + 2 * a^2 * b * c + a^2 * c^2 - b^4) / (a^4 - b^2 * c^2 + 2 * a * b * c^2 + c^4) =
  ((a * b + a * c + b^2) * (a * b + a * c - b^2)) / ((a^2 + b^2 - c^2) * (a^2 - b^2 + c^2)) :=
sorry

end fraction_simplification_l169_169963


namespace product_fraction_simplification_l169_169346

theorem product_fraction_simplification :
  (1 - (1 / 3)) * (1 - (1 / 4)) * (1 - (1 / 5)) = 2 / 5 :=
by
  sorry

end product_fraction_simplification_l169_169346


namespace lotus_leaves_not_odd_l169_169145

theorem lotus_leaves_not_odd (n : ℕ) (h1 : n > 1) (h2 : ∀ t : ℕ, ∃ r : ℕ, 0 ≤ r ∧ r < n ∧ (t * (t + 1) / 2 - 1) % n = r) : ¬ Odd n :=
sorry

end lotus_leaves_not_odd_l169_169145


namespace triangle_fraction_squared_l169_169257

theorem triangle_fraction_squared (a b c : ℝ) (h1 : b > a) 
  (h2 : a / b = (1 / 2) * (b / c)) (h3 : a + b + c = 12) 
  (h4 : c = Real.sqrt (a^2 + b^2)) : 
  (a / b)^2 = 1 / 2 := 
by 
  sorry

end triangle_fraction_squared_l169_169257


namespace initial_amount_proof_l169_169937

noncomputable def initial_amount (A B : ℝ) : ℝ :=
  A + B

theorem initial_amount_proof :
  ∃ (A B : ℝ), B = 4000.0000000000005 ∧ 
               (A * 0.15 * 2 = B * 0.18 * 2 + 360) ∧ 
               initial_amount A B = 10000.000000000002 :=
by
  sorry

end initial_amount_proof_l169_169937


namespace elder_age_is_30_l169_169576

-- Define the ages of the younger and elder persons
variables (y e : ℕ)

-- We have the following conditions:
-- Condition 1: The elder's age is 16 years more than the younger's age
def age_difference := e = y + 16

-- Condition 2: Six years ago, the elder's age was three times the younger's age
def six_years_ago := e - 6 = 3 * (y - 6)

-- We need to prove that the present age of the elder person is 30
theorem elder_age_is_30 (y e : ℕ) (h1 : age_difference y e) (h2 : six_years_ago y e) : e = 30 :=
sorry

end elder_age_is_30_l169_169576


namespace pipe_A_fill_time_l169_169174

theorem pipe_A_fill_time (x : ℝ) (h1 : ∀ t : ℝ, t = 45) (h2 : ∀ t : ℝ, t = 18) :
  (1/x + 1/45 = 1/18) → x = 30 :=
by {
  -- Proof is omitted
  sorry
}

end pipe_A_fill_time_l169_169174


namespace problem1_problem2_l169_169279

-- Define the function f(x) = |x + 2| + |x - 1|
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- 1. Prove the solution set of f(x) > 5 is {x | x < -3 or x > 2}
theorem problem1 : {x : ℝ | f x > 5} = {x : ℝ | x < -3 ∨ x > 2} :=
by
  sorry

-- 2. Prove that if f(x) ≥ a^2 - 2a always holds, then -1 ≤ a ≤ 3
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, f x ≥ a^2 - 2 * a) : -1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end problem1_problem2_l169_169279


namespace necessary_but_not_sufficient_for_x_gt_4_l169_169120

theorem necessary_but_not_sufficient_for_x_gt_4 (x : ℝ) : (x^2 > 16) → ¬ (x > 4) :=
by
  sorry

end necessary_but_not_sufficient_for_x_gt_4_l169_169120


namespace max_writers_at_conference_l169_169436

variables (T E W x : ℕ)

-- Defining the conditions
def conference_conditions (T E W x : ℕ) : Prop :=
  T = 90 ∧ E > 38 ∧ x ≤ 6 ∧ 2 * x + (W + E - x) = T ∧ W = T - E - x

-- Statement to prove the number of writers
theorem max_writers_at_conference : ∃ W, conference_conditions 90 39 W 1 :=
by
  sorry

end max_writers_at_conference_l169_169436


namespace product_of_slopes_constant_l169_169527

noncomputable def ellipse (x y : ℝ) := x^2 / 8 + y^2 / 4 = 1

theorem product_of_slopes_constant (a b : ℝ) (h_a_gt_b : a > b) (h_a_b_pos : 0 < a ∧ 0 < b)
  (e : ℝ) (h_eccentricity : e = (Real.sqrt 2) / 2) (P : ℝ × ℝ) (h_point_on_ellipse : (P.1, P.2) = (2, Real.sqrt 2)) :
  (∃ C : ℝ → ℝ → Prop, C = ellipse) ∧ (∃ k : ℝ, -k * 1/2 = -1 / 2) := sorry

end product_of_slopes_constant_l169_169527


namespace lorelai_jellybeans_l169_169379

variable (Gigi Rory Luke Lane Lorelai : ℕ)
variable (h1 : Gigi = 15)
variable (h2 : Rory = Gigi + 30)
variable (h3 : Luke = 2 * Rory)
variable (h4 : Lane = Gigi + 10)
variable (h5 : Lorelai = 3 * (Gigi + Luke + Lane))

theorem lorelai_jellybeans : Lorelai = 390 := by
  sorry

end lorelai_jellybeans_l169_169379


namespace score_order_l169_169125

variable (A B C D : ℕ)

theorem score_order
  (h1 : A + C = B + D)
  (h2 : B > D)
  (h3 : C > A + B) :
  C > B ∧ B > A ∧ A > D :=
by 
  sorry

end score_order_l169_169125


namespace most_appropriate_sampling_l169_169013

def total_students := 126 + 280 + 95
def adjusted_total_students := 126 - 1 + 280 + 95
def required_sample_size := 100

def elementary_proportion (total : Nat) (sample : Nat) : Nat := (sample * 126) / total
def middle_proportion (total : Nat) (sample : Nat) : Nat := (sample * 280) / total
def high_proportion (total : Nat) (sample : Nat) : Nat := (sample * 95) / total

theorem most_appropriate_sampling :
  required_sample_size = elementary_proportion adjusted_total_students required_sample_size + 
                         middle_proportion adjusted_total_students required_sample_size + 
                         high_proportion adjusted_total_students required_sample_size :=
by
  sorry

end most_appropriate_sampling_l169_169013


namespace remainder_of_large_number_l169_169649

theorem remainder_of_large_number :
  (102938475610 % 12) = 10 :=
by
  have h1 : (102938475610 % 4) = 2 := sorry
  have h2 : (102938475610 % 3) = 1 := sorry
  sorry

end remainder_of_large_number_l169_169649


namespace problem_statement_l169_169559

theorem problem_statement
  (a b A B : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ θ : ℝ, f θ ≥ 0)
  (def_f : ∀ θ : ℝ, f θ = 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ)) :
  a ^ 2 + b ^ 2 ≤ 2 ∧ A ^ 2 + B ^ 2 ≤ 1 := 
by
  sorry

end problem_statement_l169_169559


namespace angle_in_triangle_PQR_l169_169327

theorem angle_in_triangle_PQR
  (Q P R : ℝ)
  (h1 : P = 2 * Q)
  (h2 : R = 5 * Q)
  (h3 : Q + P + R = 180) : 
  P = 45 := 
by sorry

end angle_in_triangle_PQR_l169_169327


namespace scallops_per_person_l169_169202

theorem scallops_per_person 
    (scallops_per_pound : ℕ)
    (cost_per_pound : ℝ)
    (total_cost : ℝ)
    (people : ℕ)
    (total_pounds : ℝ)
    (total_scallops : ℕ)
    (scallops_per_person : ℕ)
    (h1 : scallops_per_pound = 8)
    (h2 : cost_per_pound = 24)
    (h3 : total_cost = 48)
    (h4 : people = 8)
    (h5 : total_pounds = total_cost / cost_per_pound)
    (h6 : total_scallops = scallops_per_pound * total_pounds)
    (h7 : scallops_per_person = total_scallops / people) : 
    scallops_per_person = 2 := 
by {
    sorry
}

end scallops_per_person_l169_169202


namespace find_second_divisor_l169_169311

theorem find_second_divisor:
  ∃ x: ℝ, (8900 / 6) / x = 370.8333333333333 ∧ x = 4 :=
sorry

end find_second_divisor_l169_169311


namespace car_kilometers_per_gallon_l169_169252

theorem car_kilometers_per_gallon :
  ∀ (distance gallon_used : ℝ), distance = 120 → gallon_used = 6 →
  distance / gallon_used = 20 :=
by
  intros distance gallon_used h_distance h_gallon_used
  sorry

end car_kilometers_per_gallon_l169_169252


namespace relationship_among_log_sin_exp_l169_169746

theorem relationship_among_log_sin_exp (x : ℝ) (h₁ : 0 < x) (h₂ : x < 1) (a b c : ℝ) 
(h₃ : a = Real.log 3 / Real.log x) (h₄ : b = Real.sin x)
(h₅ : c = 2 ^ x) : a < b ∧ b < c := 
sorry

end relationship_among_log_sin_exp_l169_169746


namespace problem_statement_l169_169081

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - 2 / x^2 + a / x

theorem problem_statement (a : ℝ) (k : ℝ) : 
  0 < a ∧ a ≤ 4 →
  (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 →
  |f x1 a - f x2 a| > k * |x1 - x2|) ↔
  k ≤ 2 - a^3 / 108 :=
by
  sorry

end problem_statement_l169_169081


namespace original_price_dish_l169_169339

-- Conditions
variables (P : ℝ) -- Original price of the dish
-- Discount and tips
def john_discounted_and_tip := 0.9 * P + 0.15 * P
def jane_discounted_and_tip := 0.9 * P + 0.135 * P

-- Condition of payment difference
def payment_difference := john_discounted_and_tip P = jane_discounted_and_tip P + 0.36

-- The theorem to prove
theorem original_price_dish : payment_difference P → P = 24 :=
by
  intro h
  sorry

end original_price_dish_l169_169339


namespace problem_l169_169243

theorem problem (x y : ℚ) (h1 : x + y = 10 / 21) (h2 : x - y = 1 / 63) : 
  x^2 - y^2 = 10 / 1323 := 
by 
  sorry

end problem_l169_169243


namespace three_digit_2C4_not_multiple_of_5_l169_169960

theorem three_digit_2C4_not_multiple_of_5 : ∀ C : ℕ, C < 10 → ¬(∃ n : ℕ, 2 * 100 + C * 10 + 4 = 5 * n) :=
by
  sorry

end three_digit_2C4_not_multiple_of_5_l169_169960


namespace sum_of_ages_l169_169548

variable (S T : ℕ)

theorem sum_of_ages (h1 : S = T + 7) (h2 : S + 10 = 3 * (T - 3)) : S + T = 33 := by
  sorry

end sum_of_ages_l169_169548


namespace solution_set_inequality_l169_169988

theorem solution_set_inequality (x : ℝ) : (x ≠ 1) → 
  ((x - 3) * (x + 2) / (x - 1) > 0 ↔ (-2 < x ∧ x < 1) ∨ x > 3) :=
by
  intros h
  sorry

end solution_set_inequality_l169_169988


namespace problem1_problem2_l169_169857

noncomputable def f (x : ℝ) (a : ℝ) := (1 / 2) * x^2 + 2 * a * x
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) := 3 * a^2 * Real.log x + b

theorem problem1 (a b x₀ : ℝ) (h : x₀ = a):
  a > 0 →
  (1 / 2) * x₀^2 + 2 * a * x₀ = 3 * a^2 * Real.log x₀ + b →
  x₀ + 2 * a = 3 * a^2 / x₀ →
  b = (5 * a^2 / 2) - 3 * a^2 * Real.log a := sorry

theorem problem2 (a b : ℝ):
  -2 ≤ b ∧ b ≤ 2 →
  ∀ x > 0, x < 4 →
  ∀ x, x - b + 3 * a^2 / x ≥ 0 →
  a ≥ Real.sqrt 3 / 3 ∨ a ≤ -Real.sqrt 3 / 3 := sorry

end problem1_problem2_l169_169857


namespace natural_numbers_between_sqrt_100_and_101_l169_169916

theorem natural_numbers_between_sqrt_100_and_101 :
  ∃ (n : ℕ), n = 200 ∧ (∀ k : ℕ, 100 < Real.sqrt k ∧ Real.sqrt k < 101 -> 10000 < k ∧ k < 10201) := 
by
  sorry

end natural_numbers_between_sqrt_100_and_101_l169_169916


namespace melanies_mother_gave_l169_169600

-- Define initial dimes, dad's contribution, and total dimes now
def initial_dimes : ℕ := 7
def dad_dimes : ℕ := 8
def total_dimes : ℕ := 19

-- Define the number of dimes the mother gave
def mother_dimes := total_dimes - (initial_dimes + dad_dimes)

-- Proof statement
theorem melanies_mother_gave : mother_dimes = 4 := by
  sorry

end melanies_mother_gave_l169_169600


namespace compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l169_169817

-- Part 1
theorem compare_ab_1_to_a_b {a b : ℝ} (h1 : a^b * b^a + Real.log b / Real.log a = 0) (ha : a > 0) (hb : b > 0) : ab + 1 < a + b := sorry

-- Part 2
theorem two_pow_b_eq_one_div_b {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : 2^b = 1 / b := sorry

-- Part 3
theorem sign_of_expression {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : (2 * b + 1 - Real.sqrt 5) * (3 * b - 2) < 0 := sorry

end compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l169_169817


namespace drink_costs_l169_169996

theorem drink_costs (cost_of_steak_per_person : ℝ) (total_tip_paid : ℝ) (tip_percentage : ℝ) (billy_tip_coverage_percentage : ℝ) (total_tip_percentage : ℝ) :
  cost_of_steak_per_person = 20 → 
  total_tip_paid = 8 → 
  tip_percentage = 0.20 → 
  billy_tip_coverage_percentage = 0.80 → 
  total_tip_percentage = 0.20 → 
  ∃ (cost_of_drink : ℝ), cost_of_drink = 1.60 :=
by
  intros
  sorry

end drink_costs_l169_169996


namespace repeating_decimal_fraction_l169_169778

noncomputable def repeating_decimal := 4 + 36 / 99

theorem repeating_decimal_fraction : 
  repeating_decimal = 144 / 33 := 
sorry

end repeating_decimal_fraction_l169_169778


namespace ballsInBoxes_theorem_l169_169834

noncomputable def ballsInBoxes : ℕ :=
  let distributions := [(5,0,0), (4,1,0), (3,2,0), (3,1,1), (2,2,1)]
  distributions.foldl (λ acc (x : ℕ × ℕ × ℕ) =>
    acc + match x with
      | (5,0,0) => 3
      | (4,1,0) => 6
      | (3,2,0) => 6
      | (3,1,1) => 6
      | (2,2,1) => 3
      | _ => 0
  ) 0

theorem ballsInBoxes_theorem : ballsInBoxes = 24 :=
by
  unfold ballsInBoxes
  rfl

end ballsInBoxes_theorem_l169_169834


namespace inequality_chain_l169_169305

open Real

theorem inequality_chain (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end inequality_chain_l169_169305


namespace annual_interest_rate_is_correct_l169_169178

-- Definitions of the conditions
def true_discount : ℚ := 210
def bill_amount : ℚ := 1960
def time_period_years : ℚ := 3 / 4

-- The present value of the bill
def present_value : ℚ := bill_amount - true_discount

-- The formula for simple interest given principal, rate, and time
def simple_interest (P R T : ℚ) : ℚ :=
  P * R * T / 100

-- Proof statement
theorem annual_interest_rate_is_correct : 
  ∃ (R : ℚ), simple_interest present_value R time_period_years = true_discount ∧ R = 16 :=
by
  use 16
  sorry

end annual_interest_rate_is_correct_l169_169178


namespace isosceles_triangle_leg_l169_169926

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ a = c ∨ b = c)

theorem isosceles_triangle_leg
  (a b c : ℝ)
  (h1 : is_isosceles_triangle a b c)
  (h2 : a + b + c = 18)
  (h3 : a = 8 ∨ b = 8 ∨ c = 8) :
  (a = 5 ∨ b = 5 ∨ c = 5 ∨ a = 8 ∨ b = 8 ∨ c = 8) :=
sorry

end isosceles_triangle_leg_l169_169926


namespace intersection_points_count_l169_169459

noncomputable def f1 (x : ℝ) : ℝ := abs (3 * x - 2)
noncomputable def f2 (x : ℝ) : ℝ := -abs (2 * x + 5)

theorem intersection_points_count : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f1 x1 = f2 x1 ∧ f1 x2 = f2 x2 ∧ 
    (∀ x : ℝ, f1 x = f2 x → x = x1 ∨ x = x2)) :=
sorry

end intersection_points_count_l169_169459


namespace race_distance_l169_169727

theorem race_distance (Va Vb Vc : ℝ) (D : ℝ) :
    (Va / Vb = 10 / 9) →
    (Va / Vc = 80 / 63) →
    (Vb / Vc = 8 / 7) →
    (D - 100) / D = 7 / 8 → 
    D = 700 :=
by
  intros h1 h2 h3 h4 
  sorry

end race_distance_l169_169727


namespace complement_union_l169_169166

open Set

def U : Set ℕ := {x | x < 6}

def A : Set ℕ := {1, 3}

def B : Set ℕ := {3, 5}

theorem complement_union :
  (U \ (A ∪ B)) = {0, 2, 4} :=
by
  sorry

end complement_union_l169_169166


namespace sqrt_factorial_squared_l169_169998

theorem sqrt_factorial_squared :
  (Real.sqrt ((Nat.factorial 5) * (Nat.factorial 4))) ^ 2 = 2880 :=
by sorry

end sqrt_factorial_squared_l169_169998


namespace abs_neg_2023_l169_169294

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l169_169294


namespace calories_350_grams_mint_lemonade_l169_169285

-- Definitions for the weights of ingredients in grams
def lemon_juice_weight := 150
def sugar_weight := 200
def water_weight := 300
def mint_weight := 50
def total_weight := lemon_juice_weight + sugar_weight + water_weight + mint_weight

-- Definitions for the caloric content per specified weight
def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 400
def mint_calories_per_10g := 7
def water_calories := 0

-- Calculate total calories from each ingredient
def lemon_juice_calories := (lemon_juice_calories_per_100g * lemon_juice_weight) / 100
def sugar_calories := (sugar_calories_per_100g * sugar_weight) / 100
def mint_calories := (mint_calories_per_10g * mint_weight) / 10

-- Calculate total calories in the lemonade
def total_calories := lemon_juice_calories + sugar_calories + mint_calories + water_calories

noncomputable def calories_in_350_grams : ℕ := (total_calories * 350) / total_weight

-- Theorem stating the number of calories in 350 grams of Marco’s lemonade
theorem calories_350_grams_mint_lemonade : calories_in_350_grams = 440 := 
by
  sorry

end calories_350_grams_mint_lemonade_l169_169285


namespace find_length_of_segment_l169_169553

noncomputable def radius : ℝ := 4
noncomputable def volume_cylinder (L : ℝ) : ℝ := 16 * Real.pi * L
noncomputable def volume_hemispheres : ℝ := 2 * (128 / 3) * Real.pi
noncomputable def total_volume (L : ℝ) : ℝ := volume_cylinder L + volume_hemispheres

theorem find_length_of_segment (L : ℝ) (h : total_volume L = 544 * Real.pi) : 
  L = 86 / 3 :=
by sorry

end find_length_of_segment_l169_169553


namespace xy_sum_l169_169395

theorem xy_sum (x y : ℝ) (h1 : x^3 - 6 * x^2 + 15 * x = 12) (h2 : y^3 - 6 * y^2 + 15 * y = 16) : x + y = 4 := 
sorry

end xy_sum_l169_169395


namespace total_population_l169_169121

theorem total_population (b g t : ℕ) (h1 : b = 2 * g) (h2 : g = 4 * t) : b + g + t = 13 * t :=
by
  sorry

end total_population_l169_169121


namespace anne_trip_shorter_l169_169763

noncomputable def john_walk_distance : ℝ := 2 + 1

noncomputable def anne_walk_distance : ℝ := Real.sqrt (2^2 + 1^2)

noncomputable def distance_difference : ℝ := john_walk_distance - anne_walk_distance

noncomputable def percentage_reduction : ℝ := (distance_difference / john_walk_distance) * 100

theorem anne_trip_shorter :
  20 ≤ percentage_reduction ∧ percentage_reduction < 30 :=
by
  sorry

end anne_trip_shorter_l169_169763


namespace profit_percentage_no_initial_discount_l169_169112

theorem profit_percentage_no_initial_discount
  (CP : ℝ := 100)
  (bulk_discount : ℝ := 0.02)
  (sales_tax : ℝ := 0.065)
  (no_discount_price : ℝ := CP - CP * bulk_discount)
  (selling_price : ℝ := no_discount_price + no_discount_price * sales_tax)
  (profit : ℝ := selling_price - CP) :
  (profit / CP) * 100 = 4.37 :=
by
  -- proof here
  sorry

end profit_percentage_no_initial_discount_l169_169112


namespace gcf_of_48_180_120_l169_169521

theorem gcf_of_48_180_120 : Nat.gcd (Nat.gcd 48 180) 120 = 12 := by
  sorry

end gcf_of_48_180_120_l169_169521


namespace abs_difference_of_opposite_signs_l169_169490

theorem abs_difference_of_opposite_signs (a b : ℝ) (ha : |a| = 4) (hb : |b| = 2) (hdiff : a * b < 0) : |a - b| = 6 := 
sorry

end abs_difference_of_opposite_signs_l169_169490


namespace p_pow_four_minus_one_divisible_by_ten_l169_169088

theorem p_pow_four_minus_one_divisible_by_ten
  (p : Nat) (prime_p : Nat.Prime p) (h₁ : p ≠ 2) (h₂ : p ≠ 5) : 
  10 ∣ (p^4 - 1) := 
by
  sorry

end p_pow_four_minus_one_divisible_by_ten_l169_169088


namespace intersection_P_Q_l169_169961

def P : Set ℝ := { x | x > 1 }
def Q : Set ℝ := { x | x < 2 }

theorem intersection_P_Q : P ∩ Q = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_P_Q_l169_169961


namespace smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60_l169_169651

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → ¬(m ∣ n)
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def has_prime_factor_less_than (n k : ℕ) : Prop := ∃ p : ℕ, p < k ∧ is_prime p ∧ p ∣ n

theorem smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60 :
  ∃ m : ℕ, 
    m = 4091 ∧ 
    ¬is_prime m ∧ 
    ¬is_square m ∧ 
    ¬has_prime_factor_less_than m 60 ∧ 
    (∀ n : ℕ, ¬is_prime n ∧ ¬is_square n ∧ ¬has_prime_factor_less_than n 60 → 4091 ≤ n) :=
by
  sorry

end smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60_l169_169651


namespace extreme_value_at_one_symmetric_points_range_l169_169782

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then
  x^2 + 3 * a * x
else
  2 * Real.exp x - x^2 + 2 * a * x

theorem extreme_value_at_one (a : ℝ) :
  (∀ x > 0, f x a = 2 * Real.exp x - x^2 + 2 * a * x) →
  (∀ x < 0, f x a = x^2 + 3 * a * x) →
  (∀ x > 0, deriv (fun x => f x a) x = 2 * Real.exp x - 2 * x + 2 * a) →
  deriv (fun x => f x a) 1 = 0 →
  a = 1 - Real.exp 1 :=
  sorry

theorem symmetric_points_range (a : ℝ) :
  (∃ x0 > 0, (∃ y0 : ℝ, 
  (f x0 a = y0 ∧ f (-x0) a = -y0))) →
  a ≥ 2 * Real.exp 1 :=
  sorry

end extreme_value_at_one_symmetric_points_range_l169_169782


namespace area_of_region_l169_169681

def plane_region (x y : ℝ) : Prop := |x| ≤ 1 ∧ |y| ≤ 1

def inequality_holds (a b : ℝ) : Prop := ∀ x y : ℝ, plane_region x y → a * x - 2 * b * y ≤ 2

theorem area_of_region (a b : ℝ) (h : inequality_holds a b) : 
  (-2 ≤ a ∧ a ≤ 2) ∧ (-1 ≤ b ∧ b ≤ 1) ∧ (4 * 2 = 8) :=
sorry

end area_of_region_l169_169681


namespace area_of_triangle_from_line_l169_169045

-- Define the conditions provided in the problem
def line_eq (B : ℝ) (x y : ℝ) := B * x + 9 * y = 18
def B_val := (36 : ℝ)

theorem area_of_triangle_from_line (B : ℝ) (hB : B = B_val) : 
  (∃ C : ℝ, C = 1 / 2) := by
  sorry

end area_of_triangle_from_line_l169_169045


namespace inequality_ineqs_l169_169127

theorem inequality_ineqs (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_cond : x * y + y * z + z * x = 1) :
  (27 / 4) * (x + y) * (y + z) * (z + x) 
  ≥ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2
  ∧ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2 
  ≥ 
  6 * Real.sqrt 3 := by
  sorry

end inequality_ineqs_l169_169127


namespace maria_bottles_count_l169_169868

-- Definitions from the given conditions
def b_initial : ℕ := 23
def d : ℕ := 12
def g : ℕ := 5
def b : ℕ := 65

-- Definition of the question based on conditions
def b_final : ℕ := b_initial - d - g + b

-- The statement to prove the correctness of the answer
theorem maria_bottles_count : b_final = 71 := by
  -- We skip the proof for this statement
  sorry

end maria_bottles_count_l169_169868


namespace Janet_saves_154_minutes_per_week_l169_169692

-- Definitions for the time spent on each activity daily
def timeLookingForKeys := 8 -- minutes
def timeComplaining := 3 -- minutes
def timeSearchingForPhone := 5 -- minutes
def timeLookingForWallet := 4 -- minutes
def timeSearchingForSunglasses := 2 -- minutes

-- Total time spent daily on these activities
def totalDailyTime := timeLookingForKeys + timeComplaining + timeSearchingForPhone + timeLookingForWallet + timeSearchingForSunglasses
-- Time savings calculation for a week
def weeklySaving := totalDailyTime * 7

-- The proof statement that Janet will save 154 minutes every week
theorem Janet_saves_154_minutes_per_week : weeklySaving = 154 := by
  sorry

end Janet_saves_154_minutes_per_week_l169_169692


namespace binom_12_9_eq_220_l169_169705

open Nat

theorem binom_12_9_eq_220 : Nat.choose 12 9 = 220 := by
  sorry

end binom_12_9_eq_220_l169_169705


namespace contradiction_even_odd_l169_169750

theorem contradiction_even_odd (a b c : ℕ) (h1 : (a % 2 = 1 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ c % 2 = 1) ∨ (b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1)) :
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) :=
by
  -- proof by contradiction
  sorry

end contradiction_even_odd_l169_169750


namespace dishonest_dealer_uses_correct_weight_l169_169768

noncomputable def dishonest_dealer_weight (profit_percent : ℝ) (true_weight : ℝ) : ℝ :=
  true_weight - (profit_percent / 100 * true_weight)

theorem dishonest_dealer_uses_correct_weight :
  dishonest_dealer_weight 11.607142857142861 1 = 0.8839285714285714 :=
by
  -- We skip the proof here
  sorry

end dishonest_dealer_uses_correct_weight_l169_169768


namespace consumption_increased_by_27_91_percent_l169_169071
noncomputable def percentage_increase_in_consumption (T C : ℝ) : ℝ :=
  let new_tax_rate := 0.86 * T
  let new_revenue_effect := 1.1000000000000085
  let cons_percentage_increase (P : ℝ) := (new_tax_rate * (C * (1 + P))) = new_revenue_effect * (T * C)
  let P_solution := 0.2790697674418605
  if cons_percentage_increase P_solution then P_solution * 100 else 0

-- The statement we are proving
theorem consumption_increased_by_27_91_percent (T C : ℝ) (hT : 0 < T) (hC : 0 < C) :
  percentage_increase_in_consumption T C = 27.91 :=
by
  sorry

end consumption_increased_by_27_91_percent_l169_169071


namespace ratio_time_A_to_B_l169_169900

-- Definition of total examination time in minutes
def total_time : ℕ := 180

-- Definition of time spent on type A problems
def time_A : ℕ := 40

-- Definition of time spent on type B problems as total_time - time_A
def time_B : ℕ := total_time - time_A

-- Statement that we need to prove
theorem ratio_time_A_to_B : time_A * 7 = time_B * 2 :=
by
  -- Implementation of the proof will go here
  sorry

end ratio_time_A_to_B_l169_169900


namespace total_number_of_chips_l169_169691

theorem total_number_of_chips 
  (viviana_chocolate : ℕ) (susana_chocolate : ℕ) (viviana_vanilla : ℕ) (susana_vanilla : ℕ)
  (manuel_vanilla : ℕ) (manuel_chocolate : ℕ)
  (h1 : viviana_chocolate = susana_chocolate + 5)
  (h2 : susana_vanilla = (3 * viviana_vanilla) / 4)
  (h3 : viviana_vanilla = 20)
  (h4 : susana_chocolate = 25)
  (h5 : manuel_vanilla = 2 * susana_vanilla)
  (h6 : manuel_chocolate = viviana_chocolate / 2) :
  viviana_chocolate + susana_chocolate + manuel_chocolate + viviana_vanilla + susana_vanilla + manuel_vanilla = 135 :=
sorry

end total_number_of_chips_l169_169691


namespace articles_correct_l169_169287

-- Define the problem conditions
def refersToSpecific (word : String) : Prop :=
  word = "keyboard"

def refersToGeneral (word : String) : Prop :=
  word = "computer"

-- Define the articles
def the_article : String := "the"
def a_article : String := "a"

-- State the theorem for the corresponding solution
theorem articles_correct :
  refersToSpecific "keyboard" → refersToGeneral "computer" →  
  (the_article, a_article) = ("the", "a") :=
by
  intro h1 h2
  sorry

end articles_correct_l169_169287


namespace John_break_time_l169_169529

-- Define the constants
def John_dancing_hours : ℕ := 8

-- Define the condition for James's dancing time 
def James_dancing_time (B : ℕ) : ℕ := 
  let total_time := John_dancing_hours + B
  total_time + total_time / 3

-- State the problem as a theorem
theorem John_break_time (B : ℕ) : John_dancing_hours + James_dancing_time B = 20 → B = 1 := 
  by sorry

end John_break_time_l169_169529


namespace no_such_point_exists_l169_169762

theorem no_such_point_exists 
  (side_length : ℝ)
  (original_area : ℝ)
  (total_area_after_first_rotation : ℝ)
  (total_area_after_second_rotation : ℝ)
  (no_overlapping_exists : Prop) :
  side_length = 12 → 
  original_area = 144 → 
  total_area_after_first_rotation = 211 → 
  total_area_after_second_rotation = 287 →
  no_overlapping_exists := sorry

end no_such_point_exists_l169_169762


namespace equivalent_expression_l169_169218

theorem equivalent_expression :
  (5+3) * (5^2 + 3^2) * (5^4 + 3^4) * (5^8 + 3^8) * (5^16 + 3^16) * 
  (5^32 + 3^32) * (5^64 + 3^64) = 5^128 - 3^128 := 
  sorry

end equivalent_expression_l169_169218


namespace number_of_rows_with_exactly_7_students_l169_169240

theorem number_of_rows_with_exactly_7_students 
  (total_students : ℕ) (rows_with_6_students rows_with_7_students : ℕ) 
  (total_students_eq : total_students = 53)
  (seats_condition : total_students = 6 * rows_with_6_students + 7 * rows_with_7_students) 
  (no_seat_unoccupied : rows_with_6_students + rows_with_7_students = rows_with_6_students + rows_with_7_students) :
  rows_with_7_students = 5 := by
  sorry

end number_of_rows_with_exactly_7_students_l169_169240


namespace log_equation_solution_l169_169344

theorem log_equation_solution (x : ℝ) (hpos : x > 0) (hneq : x ≠ 1) : (Real.log 8 / Real.log x) * (2 * Real.log x / Real.log 2) = 6 * Real.log 2 :=
by
  sorry

end log_equation_solution_l169_169344


namespace optimal_strategies_and_value_l169_169262

-- Define the payoff matrix for the two-player zero-sum game
def payoff_matrix : Matrix (Fin 2) (Fin 2) ℕ := ![![12, 22], ![32, 2]]

-- Define the optimal mixed strategies for both players
def optimal_strategy_row_player : Fin 2 → ℚ
| 0 => 3 / 4
| 1 => 1 / 4

def optimal_strategy_column_player : Fin 2 → ℚ
| 0 => 1 / 2
| 1 => 1 / 2

-- Define the value of the game
def value_of_game := (17 : ℚ)

theorem optimal_strategies_and_value :
  (∀ i j, (optimal_strategy_row_player 0 * payoff_matrix 0 j + optimal_strategy_row_player 1 * payoff_matrix 1 j = value_of_game) ∧
           (optimal_strategy_column_player 0 * payoff_matrix i 0 + optimal_strategy_column_player 1 * payoff_matrix i 1 = value_of_game)) :=
by 
  -- sorry is used as a placeholder for the proof
  sorry

end optimal_strategies_and_value_l169_169262


namespace quadratic_polynomial_value_bound_l169_169549

theorem quadratic_polynomial_value_bound (a b : ℝ) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, |(x^2 + a * x + b)| ≥ 1/2 :=
by
  sorry

end quadratic_polynomial_value_bound_l169_169549


namespace area_of_rectangular_field_l169_169014

-- Define the conditions
variables (l w : ℝ)

def perimeter_condition : Prop := 2 * l + 2 * w = 100
def length_width_relation : Prop := l = 3 * w

-- Define the area
def area : ℝ := l * w

-- Prove the area given the conditions
theorem area_of_rectangular_field (h1 : perimeter_condition l w) (h2 : length_width_relation l w) : area l w = 468.75 :=
by sorry

end area_of_rectangular_field_l169_169014


namespace probability_divisible_by_five_l169_169416

def is_three_digit_number (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000

def ends_with_five (n: ℕ) : Prop := n % 10 = 5

def divisible_by_five (n: ℕ) : Prop := n % 5 = 0

theorem probability_divisible_by_five {N : ℕ} (h1: is_three_digit_number N) (h2: ends_with_five N) : 
  ∃ p : ℚ, p = 1 ∧ ∀ n, (is_three_digit_number n ∧ ends_with_five n) → (divisible_by_five n) :=
by
  sorry

end probability_divisible_by_five_l169_169416


namespace person_age_is_30_l169_169288

-- Definitions based on the conditions
def age (x : ℕ) := x
def age_5_years_hence (x : ℕ) := x + 5
def age_5_years_ago (x : ℕ) := x - 5

-- The main theorem to prove
theorem person_age_is_30 (x : ℕ) (h : 3 * age_5_years_hence x - 3 * age_5_years_ago x = age x) : x = 30 :=
by
  sorry

end person_age_is_30_l169_169288


namespace line_through_fixed_point_fixed_points_with_constant_slope_l169_169143

-- Point structure definition
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define curves C1 and C2
def curve_C1 (p : Point) : Prop :=
  p.x^2 + (p.y - 1/4)^2 = 1 ∧ p.y ≥ 1/4

def curve_C2 (p : Point) : Prop :=
  p.x^2 = 8 * p.y - 1 ∧ abs p.x ≥ 1

-- Line passing through fixed point for given perpendicularity condition
theorem line_through_fixed_point (A B M : Point) (l : ℝ → ℝ → Prop) :
  curve_C2 A → curve_C2 B →
  (∃ k b, ∀ x y, l x y ↔ y = k * x + b) →
  (M = ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩) →
  ((M.x = A.x ∧ M.y = (A.y + B.y) / 2) → A.x * B.x = -16) →
  ∀ x y, l x y → y = (17 / 8) := sorry

-- Existence of two fixed points on y-axis with constant slope product
theorem fixed_points_with_constant_slope (P T1 T2 M : Point) (l : ℝ → ℝ → Prop) :
  curve_C1 P →
  (T1 = ⟨0, -1⟩) →
  (T2 = ⟨0, 1⟩) →
  l P.x P.y →
  (∃ k b, ∀ x y, l x y ↔ y = k * x + b) →
  (M.y^2 - (M.x^2 / 16) = 1) →
  (M.x ≠ 0) →
  ((M.y + 1) / M.x) * ((M.y - 1) / M.x) = (1 / 16) := sorry

end line_through_fixed_point_fixed_points_with_constant_slope_l169_169143


namespace cyclist_C_speed_l169_169573

variable (c d : ℕ)

def distance_to_meeting (c d : ℕ) : Prop :=
  d = c + 6 ∧
  90 + 30 = 120 ∧
  ((90 - 30) / c) = (120 / d) ∧
  (60 / c) = (120 / (c + 6))

theorem cyclist_C_speed : distance_to_meeting c d → c = 6 :=
by
  intro h
  -- To be filled in with the proof using the conditions
  sorry

end cyclist_C_speed_l169_169573


namespace no_overlapping_sale_days_l169_169116

def bookstore_sale_days (d : ℕ) : Prop :=
  d % 4 = 0 ∧ 1 ≤ d ∧ d ≤ 31

def shoe_store_sale_days (d : ℕ) : Prop :=
  ∃ k : ℕ, d = 2 + 8 * k ∧ 1 ≤ d ∧ d ≤ 31

theorem no_overlapping_sale_days : 
  ∀ d : ℕ, bookstore_sale_days d → ¬ shoe_store_sale_days d :=
by
  intros d h1 h2
  sorry

end no_overlapping_sale_days_l169_169116


namespace range_of_values_for_m_l169_169340

theorem range_of_values_for_m (m : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < m) → m > 1 :=
by
  sorry

end range_of_values_for_m_l169_169340


namespace abs_eq_5_iff_l169_169203

theorem abs_eq_5_iff (a : ℝ) : |a| = 5 ↔ a = 5 ∨ a = -5 :=
by
  sorry

end abs_eq_5_iff_l169_169203


namespace problem_l169_169229

theorem problem 
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2010)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2006)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2010)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2006)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2010)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2006) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 996 / 1005 :=
sorry

end problem_l169_169229


namespace sum_largest_and_smallest_l169_169217

-- Define the three-digit number properties
def hundreds_digit := 4
def tens_digit := 8
def A : ℕ := sorry  -- Placeholder for the digit A

-- Define the number based on the digits
def number (A : ℕ) : ℕ := 100 * hundreds_digit + 10 * tens_digit + A

-- Hypotheses
axiom A_range : 0 ≤ A ∧ A ≤ 9

-- Largest and smallest possible numbers
def largest_number := number 9
def smallest_number := number 0

-- Prove the sum
theorem sum_largest_and_smallest : largest_number + smallest_number = 969 :=
by
  sorry

end sum_largest_and_smallest_l169_169217


namespace equiangular_hexagon_sides_l169_169790

variable {a b c d e f : ℝ}

-- Definition of the equiangular hexagon condition
def equiangular_hexagon (a b c d e f : ℝ) := true

theorem equiangular_hexagon_sides (h : equiangular_hexagon a b c d e f) :
  a - d = e - b ∧ e - b = c - f :=
by
  sorry

end equiangular_hexagon_sides_l169_169790


namespace ratio_of_border_to_tile_l169_169586

variable {s d : ℝ}

theorem ratio_of_border_to_tile (h1 : 900 = 30 * 30)
  (h2 : 0.81 = (900 * s^2) / (30 * s + 60 * d)^2) :
  d / s = 1 / 18 := by {
  sorry }

end ratio_of_border_to_tile_l169_169586


namespace exists_increasing_sequence_l169_169192

theorem exists_increasing_sequence (n : ℕ) : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ∃ x : ℕ → ℕ, (∀ i : ℕ, 1 ≤ i → i ≤ n → x i < x (i + 1)) :=
by
  sorry

end exists_increasing_sequence_l169_169192


namespace initial_books_l169_169613

-- Definitions for the conditions.

def boxes (b : ℕ) : ℕ := 3 * b -- Box count
def booksInRoom : ℕ := 21 -- Books in the room
def booksOnTable : ℕ := 4 -- Books on the coffee table
def cookbooks : ℕ := 18 -- Cookbooks in the kitchen
def booksGrabbed : ℕ := 12 -- Books grabbed from the donation center
def booksNow : ℕ := 23 -- Books Henry has now

-- Define total number of books donated
def totalBooksDonated (inBoxes : ℕ) (additionalBooks : ℕ) : ℕ :=
  inBoxes + additionalBooks - booksGrabbed

-- Define number of books Henry initially had
def initialBooks (netDonated : ℕ) (booksCurrently : ℕ) : ℕ :=
  netDonated + booksCurrently

-- Proof goal
theorem initial_books (b : ℕ) (inBox : ℕ) (additionalBooks : ℕ) : 
  let totalBooks := booksInRoom + booksOnTable + cookbooks
  let inBoxes := boxes b
  let totalDonated := totalBooksDonated inBoxes totalBooks
  initialBooks totalDonated booksNow = 99 :=
by 
  simp [initialBooks, totalBooksDonated, boxes, booksInRoom, booksOnTable, cookbooks, booksGrabbed, booksNow]
  sorry

end initial_books_l169_169613


namespace derivative_of_x_log_x_l169_169702

noncomputable def y (x : ℝ) := x * Real.log x

theorem derivative_of_x_log_x (x : ℝ) (hx : 0 < x) :
  (deriv y x) = Real.log x + 1 :=
sorry

end derivative_of_x_log_x_l169_169702


namespace krishan_money_l169_169566

/-- Given that the ratio of money between Ram and Gopal is 7:17, the ratio of money between Gopal and Krishan is 7:17, and Ram has Rs. 588, prove that Krishan has Rs. 12,065. -/
theorem krishan_money (R G K : ℝ) (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : R = 588) : K = 12065 :=
by
  sorry

end krishan_money_l169_169566


namespace unique_positive_integer_triples_l169_169354

theorem unique_positive_integer_triples (a b c : ℕ) (h1 : ab + 3 * b * c = 63) (h2 : ac + 3 * b * c = 39) : 
∃! (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ ab + 3 * b * c = 63 ∧ ac + 3 * b * c = 39 :=
by sorry

end unique_positive_integer_triples_l169_169354


namespace correct_number_of_outfits_l169_169465

-- Define the number of each type of clothing
def num_red_shirts := 4
def num_green_shirts := 4
def num_blue_shirts := 4
def num_pants := 10
def num_red_hats := 6
def num_green_hats := 6
def num_blue_hats := 4

-- Define the total number of outfits that meet the conditions
def total_outfits : ℕ :=
  (num_red_shirts * num_pants * (num_green_hats + num_blue_hats)) +
  (num_green_shirts * num_pants * (num_red_hats + num_blue_hats)) +
  (num_blue_shirts * num_pants * (num_red_hats + num_green_hats))

-- The proof statement asserting that the total number of valid outfits is 1280
theorem correct_number_of_outfits : total_outfits = 1280 := by
  sorry

end correct_number_of_outfits_l169_169465


namespace hyperbola_asymptote_l169_169309

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, (x^2 - y^2 / a^2) = 1 → (y = 2*x ∨ y = -2*x)) → a = 2 :=
by
  intro h_asymptote
  sorry

end hyperbola_asymptote_l169_169309


namespace triangle_PQR_area_l169_169207

/-

Define the points P, Q, and R.
Define a function to calculate the area of a triangle given three points.
Then write a theorem to state that the area of triangle PQR is 12.

-/

structure Point where
  x : ℕ
  y : ℕ

def P : Point := ⟨2, 6⟩
def Q : Point := ⟨2, 2⟩
def R : Point := ⟨8, 5⟩

def area (A B C : Point) : ℚ :=
  abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2)

theorem triangle_PQR_area : area P Q R = 12 := by
  /- 
    The proof should involve calculating the area using the given points.
   -/
  sorry

end triangle_PQR_area_l169_169207


namespace factorize_x_cubed_minus_9x_l169_169985

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l169_169985


namespace inequality_f_c_f_a_f_b_l169_169813

-- Define the function f and the conditions
def f : ℝ → ℝ := sorry

noncomputable def a : ℝ := Real.log (1 / Real.pi)
noncomputable def b : ℝ := (Real.log Real.pi) ^ 2
noncomputable def c : ℝ := Real.log (Real.sqrt Real.pi)

-- Theorem statement
theorem inequality_f_c_f_a_f_b :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2) →
  f c > f a ∧ f a > f b :=
by
  -- Proof omitted
  sorry

end inequality_f_c_f_a_f_b_l169_169813


namespace intersection_A_B_l169_169185

noncomputable def A : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def B : Set ℝ := {y | ∃ x, y = Real.log (x^2 + 1) ∧ y ≥ 0}

theorem intersection_A_B : A ∩ {x | ∃ y, y = Real.log (x^2 + 1) ∧ y ≥ 0} = {x | 0 < x ∧ x < 2} :=
  sorry

end intersection_A_B_l169_169185


namespace days_for_30_men_to_build_wall_l169_169752

theorem days_for_30_men_to_build_wall 
  (men1 days1 men2 k : ℕ)
  (h1 : men1 = 18)
  (h2 : days1 = 5)
  (h3 : men2 = 30)
  (h_k : men1 * days1 = k)
  : (men2 * 3 = k) := by 
sorry

end days_for_30_men_to_build_wall_l169_169752


namespace product_of_five_consecutive_is_divisible_by_sixty_l169_169484

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l169_169484


namespace minimum_number_of_guests_l169_169336

def total_food : ℤ := 327
def max_food_per_guest : ℤ := 2

theorem minimum_number_of_guests :
  ∀ (n : ℤ), total_food ≤ n * max_food_per_guest → n = 164 :=
by
  sorry

end minimum_number_of_guests_l169_169336


namespace value_of_a_l169_169492

theorem value_of_a (a : ℝ) (h : a = -a) : a = 0 :=
by
  sorry

end value_of_a_l169_169492


namespace fraction_equality_implies_equality_l169_169072

theorem fraction_equality_implies_equality (a b c : ℝ) (hc : c ≠ 0) :
  (a / c = b / c) → (a = b) :=
by {
  sorry
}

end fraction_equality_implies_equality_l169_169072


namespace total_painted_surface_area_l169_169638

-- Defining the conditions
def num_cubes := 19
def top_layer := 1
def middle_layer := 5
def bottom_layer := 13
def exposed_faces_top_layer := 5
def exposed_faces_middle_corner := 3
def exposed_faces_middle_center := 1
def exposed_faces_bottom_layer := 1

-- Question: How many square meters are painted?
theorem total_painted_surface_area : 
  let top_layer_area := top_layer * exposed_faces_top_layer
  let middle_layer_area := (4 * exposed_faces_middle_corner) + exposed_faces_middle_center
  let bottom_layer_area := bottom_layer * exposed_faces_bottom_layer
  top_layer_area + middle_layer_area + bottom_layer_area = 31 :=
by
  sorry

end total_painted_surface_area_l169_169638


namespace range_of_m_l169_169534

-- Given definitions and conditions
def sequence_a (n : ℕ) : ℕ := if n = 1 then 2 else n * 2^n

def vec_a : ℕ × ℤ := (2, -1)

def vec_b (n : ℕ) : ℕ × ℤ := (sequence_a n + 2^n, sequence_a (n + 1))

def orthogonal (v1 v2 : ℕ × ℤ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Translate the proof problem
theorem range_of_m (n : ℕ) (m : ℝ) (h1 : orthogonal vec_a (vec_b n))
  (h2 : ∀ n : ℕ, n > 0 → (sequence_a n) / (n * (n + 1)^2) > (m^2 - 3 * m) / 9) :
  -1 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l169_169534


namespace contrapositive_proposition_l169_169025

theorem contrapositive_proposition :
  (∀ x : ℝ, (x^2 < 4 → -2 < x ∧ x < 2)) ↔ (∀ x : ℝ, (x ≤ -2 ∨ x ≥ 2 → x^2 ≥ 4)) :=
by
  sorry

end contrapositive_proposition_l169_169025


namespace must_be_odd_l169_169472

theorem must_be_odd (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) :=
sorry

end must_be_odd_l169_169472


namespace range_of_a_l169_169975

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.exp x - 2 * a * x - a ^ 2 + 3

theorem range_of_a (h : ∀ x, x ≥ 0 → f x a - x ^ 2 ≥ 0) :
  -Real.sqrt 5 ≤ a ∧ a ≤ 3 - Real.log 3 := sorry

end range_of_a_l169_169975


namespace ratio_of_S_to_R_l169_169696

noncomputable def find_ratio (total_amount : ℕ) (diff_SP : ℕ) (n : ℕ) (k : ℕ) (P : ℕ) (Q : ℕ) (R : ℕ) (S : ℕ) (ratio_SR : ℕ) :=
  Q = n ∧ R = n ∧ P = k * n ∧ S = ratio_SR * n ∧ P + Q + R + S = total_amount ∧ S - P = diff_SP

theorem ratio_of_S_to_R :
  ∃ n k ratio_SR, k = 2 ∧ ratio_SR = 4 ∧ 
  find_ratio 1000 250 n k 250 125 125 500 ratio_SR :=
by
  sorry

end ratio_of_S_to_R_l169_169696


namespace part1_part2_part3_l169_169039

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  abs (x^2 - 1) + x^2 + k * x

theorem part1 (h : 2 = 2) :
  (f (- (1 + Real.sqrt 3) /2) 2 = 0) ∧ (f (-1/2) 2 = 0) := by
  sorry

theorem part2 (h_alpha : 0 < α) (h_beta : α < β) (h_beta2 : β < 2) (h_f_alpha : f α k = 0) (h_f_beta : f β k = 0) :
  -7/2 < k ∧ k < -1 := by
  sorry

theorem part3 (h_alpha : 0 < α) (h_alpha1 : α ≤ 1) (h_beta1 : 1 < β) (h_beta2 : β < 2) (h1 : k = - 1 / α) (h2 : 2 * β^2 + k * β - 1 = 0) :
  1/α + 1/β < 4 := by
  sorry

end part1_part2_part3_l169_169039


namespace triangle_angle_C_and_area_l169_169433

theorem triangle_angle_C_and_area (A B C : ℝ) (a b c : ℝ) 
  (h1 : 2 * c * Real.cos B = 2 * a - b)
  (h2 : c = Real.sqrt 3)
  (h3 : b - a = 1) :
  (C = Real.pi / 3) ∧
  (1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2) :=
by
  sorry

end triangle_angle_C_and_area_l169_169433


namespace granger_bought_12_cans_of_spam_l169_169148

theorem granger_bought_12_cans_of_spam : 
  ∀ (S : ℕ), 
    (3 * 5 + 4 * 2 + 3 * S = 59) → 
    (S = 12) := 
by
  intro S h
  sorry

end granger_bought_12_cans_of_spam_l169_169148


namespace geometric_sequence_solve_a1_l169_169722

noncomputable def geometric_sequence_a1 (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < q)
    (h2 : a 2 = 1) (h3 : a 3 * a 9 = 2 * (a 5 ^ 2)) :=
  a 1 = (Real.sqrt 2) / 2

-- Define the main statement
theorem geometric_sequence_solve_a1 (a : ℕ → ℝ) (q : ℝ)
    (hq : 0 < q) (ha2 : a 2 = 1) (ha3_ha9 : a 3 * a 9 = 2 * (a 5 ^ 2)) :
    a 1 = (Real.sqrt 2) / 2 :=
sorry  -- The proof will be written here

end geometric_sequence_solve_a1_l169_169722


namespace intersection_of_sets_l169_169297

noncomputable def U : Set ℝ := Set.univ

noncomputable def M : Set ℝ := {x | x < -1 ∨ x > 1}

noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}

noncomputable def complement_U_M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

noncomputable def intersection_N_complement_U_M : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets :
  N ∩ complement_U_M = intersection_N_complement_U_M := 
sorry

end intersection_of_sets_l169_169297


namespace lila_stickers_correct_l169_169450

-- Defining the constants for number of stickers each has
def Kristoff_stickers : ℕ := 85
def Riku_stickers : ℕ := 25 * Kristoff_stickers
def Lila_stickers : ℕ := 2 * (Kristoff_stickers + Riku_stickers)

-- The theorem to prove
theorem lila_stickers_correct : Lila_stickers = 4420 := 
by {
  sorry
}

end lila_stickers_correct_l169_169450


namespace symmetric_point_coordinates_l169_169642

-- Define the type for 3D points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetric point function with respect to the x-axis
def symmetricPointWithRespectToXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Define the specific point
def givenPoint : Point3D := { x := 2, y := 3, z := 4 }

-- State the theorem to be proven
theorem symmetric_point_coordinates : 
  symmetricPointWithRespectToXAxis givenPoint = { x := 2, y := -3, z := -4 } :=
by
  sorry

end symmetric_point_coordinates_l169_169642


namespace talia_father_age_l169_169871

theorem talia_father_age 
  (t tf tm ta : ℕ) 
  (h1 : t + 7 = 20)
  (h2 : tm = 3 * t)
  (h3 : tf + 3 = tm)
  (h4 : ta = (tm - t) / 2)
  (h5 : ta + 2 = tf + 5) : 
  tf = 36 :=
by
  sorry

end talia_father_age_l169_169871


namespace find_b_l169_169496

theorem find_b (b : ℝ) : (∃ x : ℝ, (x^3 - 3*x^2 = -3*x + b ∧ (3*x^2 - 6*x = -3))) → b = 1 :=
by
  intros h
  sorry

end find_b_l169_169496


namespace total_cost_of_panels_l169_169894

theorem total_cost_of_panels
    (sidewall_width : ℝ)
    (sidewall_height : ℝ)
    (triangle_base : ℝ)
    (triangle_height : ℝ)
    (panel_width : ℝ)
    (panel_height : ℝ)
    (panel_cost : ℝ)
    (total_cost : ℝ)
    (h_sidewall : sidewall_width = 9)
    (h_sidewall_height : sidewall_height = 7)
    (h_triangle_base : triangle_base = 9)
    (h_triangle_height : triangle_height = 6)
    (h_panel_width : panel_width = 10)
    (h_panel_height : panel_height = 15)
    (h_panel_cost : panel_cost = 32)
    (h_total_cost : total_cost = 32) :
    total_cost = panel_cost :=
by
  sorry

end total_cost_of_panels_l169_169894


namespace average_death_rate_l169_169584

variable (birth_rate : ℕ) (net_increase_day : ℕ)

noncomputable def death_rate_per_two_seconds (birth_rate net_increase_day : ℕ) : ℕ :=
  let seconds_per_day := 86400
  let net_increase_per_second := net_increase_day / seconds_per_day
  let birth_rate_per_second := birth_rate / 2
  let death_rate_per_second := birth_rate_per_second - net_increase_per_second
  2 * death_rate_per_second

theorem average_death_rate
  (birth_rate : ℕ := 4) 
  (net_increase_day : ℕ := 86400) :
  death_rate_per_two_seconds birth_rate net_increase_day = 2 :=
sorry

end average_death_rate_l169_169584


namespace number_of_white_balls_l169_169756

theorem number_of_white_balls (total : ℕ) (freq_red freq_black : ℚ) (h1 : total = 120) 
                              (h2 : freq_red = 0.15) (h3 : freq_black = 0.45) : 
                              (total - total * freq_red - total * freq_black = 48) :=
by sorry

end number_of_white_balls_l169_169756


namespace smallest_angle_opposite_smallest_side_l169_169890

theorem smallest_angle_opposite_smallest_side 
  (a b c : ℝ) 
  (h_triangle : triangle_inequality_proof)
  (h_condition : 3 * a = b + c) :
  smallest_angle_proof :=
sorry

end smallest_angle_opposite_smallest_side_l169_169890


namespace circle_intersection_line_l169_169444

theorem circle_intersection_line (d : ℝ) :
  (∃ (x y : ℝ), (x - 5)^2 + (y + 2)^2 = 49 ∧ (x + 1)^2 + (y - 5)^2 = 25 ∧ x + y = d) ↔ d = 6.5 :=
by
  sorry

end circle_intersection_line_l169_169444


namespace ratio_fifteenth_term_l169_169260

-- Definitions of S_n and T_n based on the given conditions
def S_n (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
def T_n (b e n : ℕ) : ℕ := n * (2 * b + (n - 1) * e) / 2

-- Statement of the problem
theorem ratio_fifteenth_term 
  (a b d e : ℕ) 
  (h : ∀ n, (S_n a d n : ℚ) / (T_n b e n : ℚ) = (9 * n + 5) / (6 * n + 31)) : 
  (a + 14 * d : ℚ) / (b + 14 * e : ℚ) = (92 : ℚ) / 71 :=
by sorry

end ratio_fifteenth_term_l169_169260


namespace geo_sequence_necessity_l169_169662

theorem geo_sequence_necessity (a1 a2 a3 a4 : ℝ) (h_non_zero: a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0 ∧ a4 ≠ 0) :
  (a1 * a4 = a2 * a3) → (∀ r : ℝ, (a2 = a1 * r) ∧ (a3 = a2 * r) ∧ (a4 = a3 * r)) → False :=
sorry

end geo_sequence_necessity_l169_169662


namespace decreasing_exponential_range_l169_169845

theorem decreasing_exponential_range {a : ℝ} (h : ∀ x y : ℝ, x < y → (a + 1)^x > (a + 1)^y) : -1 < a ∧ a < 0 :=
sorry

end decreasing_exponential_range_l169_169845


namespace new_student_bmi_l169_169456

theorem new_student_bmi 
(average_weight_29 : ℚ)
(average_height_29 : ℚ)
(average_weight_30 : ℚ)
(average_height_30 : ℚ)
(new_student_height : ℚ)
(bmi : ℚ)
(h1 : average_weight_29 = 28)
(h2 : average_height_29 = 1.5)
(h3 : average_weight_30 = 27.5)
(h4 : average_height_30 = 1.5)
(h5 : new_student_height = 1.4)
: bmi = 6.63 := 
sorry

end new_student_bmi_l169_169456


namespace perfect_square_trinomial_l169_169117

theorem perfect_square_trinomial (m : ℤ) : 
  (∃ x y : ℝ, 16 * x^2 + m * x * y + 25 * y^2 = (4 * x + 5 * y)^2 ∨ 16 * x^2 + m * x * y + 25 * y^2 = (4 * x - 5 * y)^2) ↔ (m = 40 ∨ m = -40) :=
by
  sorry

end perfect_square_trinomial_l169_169117


namespace find_f_2017_l169_169936

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_2017 (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_func_eq : ∀ x : ℝ, f (x + 3) * f x = -1)
  (h_val : f (-1) = 2) :
  f 2017 = -2 := sorry

end find_f_2017_l169_169936


namespace circle_diameter_l169_169150

-- The problem statement in Lean 4

theorem circle_diameter
  (d α β : ℝ) :
  ∃ r: ℝ,
  r * 2 = d * (Real.sin α) * (Real.sin β) / (Real.cos ((α + β) / 2) * (Real.sin ((α - β) / 2))) :=
sorry

end circle_diameter_l169_169150


namespace ratio_of_geometric_sequence_sum_l169_169167

theorem ratio_of_geometric_sequence_sum (a : ℕ → ℕ) 
    (q : ℕ) (h_q_pos : 0 < q) (h_q_ne_one : q ≠ 1)
    (h_geo_seq : ∀ n : ℕ, a (n + 1) = a n * q)
    (h_arith_seq : 2 * a (3 + 2) = a 3 - a (3 + 1)) :
  (a 4 * (1 - q ^ 4) / (1 - q)) / (a 4 * (1 - q ^ 2) / (1 - q)) = 5 / 4 := 
  sorry

end ratio_of_geometric_sequence_sum_l169_169167


namespace greatest_integer_2e_minus_5_l169_169058

noncomputable def e : ℝ := 2.718

theorem greatest_integer_2e_minus_5 : ⌊2 * e - 5⌋ = 0 :=
by
  -- This is a placeholder for the actual proof. 
  sorry

end greatest_integer_2e_minus_5_l169_169058


namespace stream_speed_l169_169840

variable (D : ℝ) -- Distance rowed

theorem stream_speed (v : ℝ) (h : D / (60 - v) = 2 * (D / (60 + v))) : v = 20 :=
by
  sorry

end stream_speed_l169_169840


namespace max_A_value_l169_169891

-- Variables
variables {x1 x2 x3 y1 y2 y3 z1 z2 z3 : ℝ}

-- Assumptions
axiom pos_x1 : 0 < x1
axiom pos_x2 : 0 < x2
axiom pos_x3 : 0 < x3
axiom pos_y1 : 0 < y1
axiom pos_y2 : 0 < y2
axiom pos_y3 : 0 < y3
axiom pos_z1 : 0 < z1
axiom pos_z2 : 0 < z2
axiom pos_z3 : 0 < z3

-- Statement
theorem max_A_value :
  ∃ A : ℝ, 
    (∀ x1 x2 x3 y1 y2 y3 z1 z2 z3, 
    (0 < x1) → (0 < x2) → (0 < x3) →
    (0 < y1) → (0 < y2) → (0 < y3) →
    (0 < z1) → (0 < z2) → (0 < z3) →
    (x1^3 + x2^3 + x3^3 + 1) * (y1^3 + y2^3 + y3^3 + 1) * (z1^3 + z2^3 + z3^3 + 1) ≥
    A * (x1 + y1 + z1) * (x2 + y2 + z2) * (x3 + y3 + z3)) ∧ 
    A = 9/2 := 
by 
  exists 9/2 
  sorry

end max_A_value_l169_169891


namespace songs_in_first_two_albums_l169_169334

/-
Beyonce releases 5 different singles on iTunes.
She releases 2 albums that each has some songs.
She releases 1 album that has 20 songs.
Beyonce has released 55 songs in total.
Prove that the total number of songs in the first two albums is 30.
-/

theorem songs_in_first_two_albums {A B : ℕ} 
  (h1 : 5 + A + B + 20 = 55) : 
  A + B = 30 :=
by
  sorry

end songs_in_first_two_albums_l169_169334


namespace prob_same_color_is_correct_l169_169172

noncomputable def prob_same_color : ℚ :=
  let green_prob := (8 : ℚ) / 10
  let red_prob := (2 : ℚ) / 10
  (green_prob)^2 + (red_prob)^2

theorem prob_same_color_is_correct :
  prob_same_color = 17 / 25 := by
  sorry

end prob_same_color_is_correct_l169_169172


namespace min_value_of_squares_l169_169972

theorem min_value_of_squares (x y z : ℝ) (h : x + y + z = 1) : x^2 + y^2 + z^2 ≥ 1 / 3 := sorry

end min_value_of_squares_l169_169972


namespace three_friends_expenses_l169_169462

theorem three_friends_expenses :
  let ticket_cost := 7
  let number_of_tickets := 3
  let popcorn_cost := 1.5
  let number_of_popcorn := 2
  let milk_tea_cost := 3
  let number_of_milk_tea := 3
  let total_expenses := (ticket_cost * number_of_tickets) + (popcorn_cost * number_of_popcorn) + (milk_tea_cost * number_of_milk_tea)
  let amount_per_friend := total_expenses / 3
  amount_per_friend = 11 := 
by
  sorry

end three_friends_expenses_l169_169462


namespace ratio_female_male_l169_169688

theorem ratio_female_male (f m : ℕ) 
  (h1 : (50 * f) / f = 50) 
  (h2 : (30 * m) / m = 30) 
  (h3 : (50 * f + 30 * m) / (f + m) = 35) : 
  f / m = 1 / 3 := 
by
  sorry

end ratio_female_male_l169_169688


namespace mark_cans_l169_169347

theorem mark_cans (R J M : ℕ) (h1 : J = 2 * R + 5) (h2 : M = 4 * J) (h3 : R + J + M = 135) : M = 100 :=
by
  sorry

end mark_cans_l169_169347


namespace complex_sum_is_2_l169_169075

theorem complex_sum_is_2 
  (a b c d e f : ℂ) 
  (hb : b = 4) 
  (he : e = 2 * (-a - c)) 
  (hr : a + c + e = 0) 
  (hi : b + d + f = 6) 
  : d + f = 2 := 
  by
  sorry

end complex_sum_is_2_l169_169075


namespace smallest_digit_divisible_by_9_l169_169413

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (5 + 2 + 8 + 4 + 6 + d) % 9 = 0 ∧ ∀ e : ℕ, (5 + 2 + 8 + 4 + 6 + e) % 9 = 0 → d ≤ e := 
by {
  sorry
}

end smallest_digit_divisible_by_9_l169_169413


namespace smallest_angle_of_trapezoid_l169_169781

theorem smallest_angle_of_trapezoid (a d : ℝ) (h1 : a + 3 * d = 120) (h2 : 4 * a + 6 * d = 360) :
  a = 60 := by
  sorry

end smallest_angle_of_trapezoid_l169_169781


namespace fraction_of_students_paired_l169_169056

theorem fraction_of_students_paired {t s : ℕ} 
  (h1 : t / 4 = s / 3) : 
  (t / 4 + s / 3) / (t + s) = 2 / 7 := by sorry

end fraction_of_students_paired_l169_169056


namespace Tim_total_payment_l169_169184

-- Define the context for the problem
def manicure_cost : ℝ := 30
def tip_percentage : ℝ := 0.3

-- Define the total amount paid as the sum of the manicure cost and the tip
def total_amount_paid (cost : ℝ) (tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- The theorem to be proven
theorem Tim_total_payment : total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end Tim_total_payment_l169_169184


namespace imaginary_condition_l169_169556

noncomputable def is_imaginary (z : ℂ) : Prop := z.im ≠ 0

theorem imaginary_condition (z1 z2 : ℂ) :
  ( ∃ (z1 : ℂ), is_imaginary z1 ∨ is_imaginary z2 ∨ (is_imaginary (z1 - z2))) ↔
  ∃ (z1 z2 : ℂ), is_imaginary z1 ∨ is_imaginary z2 ∧ ¬ (is_imaginary (z1 - z2)) :=
sorry

end imaginary_condition_l169_169556


namespace day_50_of_year_N_minus_1_l169_169510

-- Definitions for the problem conditions
def day_of_week (n : ℕ) : ℕ := n % 7

-- Given that the 250th day of year N is a Friday
axiom day_250_of_year_N_is_friday : day_of_week 250 = 5

-- Given that the 150th day of year N+1 is a Friday
axiom day_150_of_year_N_plus_1_is_friday : day_of_week 150 = 5

-- Calculate the day of the week for the 50th day of year N-1
theorem day_50_of_year_N_minus_1 :
  day_of_week 50 = 4 :=
  sorry

end day_50_of_year_N_minus_1_l169_169510


namespace theorem_perimeter_shaded_region_theorem_area_shaded_region_l169_169323

noncomputable section

-- Definitions based on the conditions
def r : ℝ := Real.sqrt (1 / Real.pi)  -- radius of the unit circle

-- Define the perimeter and area functions for the shaded region
def perimeter_shaded_region (r : ℝ) : ℝ :=
  2 * Real.sqrt Real.pi

def area_shaded_region (r : ℝ) : ℝ :=
  1 / 5

-- Main theorem statements to prove
theorem theorem_perimeter_shaded_region
  (h : Real.pi * r^2 = 1) : perimeter_shaded_region r = 2 * Real.sqrt Real.pi :=
by
  sorry

theorem theorem_area_shaded_region
  (h : Real.pi * r^2 = 1) : area_shaded_region r = 1 / 5 :=
by
  sorry

end theorem_perimeter_shaded_region_theorem_area_shaded_region_l169_169323


namespace count_original_scissors_l169_169183

def originalScissors (addedScissors totalScissors : ℕ) : ℕ := totalScissors - addedScissors

theorem count_original_scissors :
  ∃ (originalScissorsCount : ℕ), originalScissorsCount = originalScissors 13 52 := 
  sorry

end count_original_scissors_l169_169183


namespace committee_probability_l169_169599

/--
Suppose there are 24 members in a club: 12 boys and 12 girls.
A 5-person committee is chosen at random.
Prove that the probability of having at least 2 boys and at least 2 girls in the committee is 121/177.
-/
theorem committee_probability :
  let boys := 12
  let girls := 12
  let total_members := 24
  let committee_size := 5
  let all_ways := Nat.choose total_members committee_size
  let invalid_ways := 2 * Nat.choose boys committee_size + 2 * (Nat.choose boys 1 * Nat.choose girls 4)
  let valid_ways := all_ways - invalid_ways
  let probability := valid_ways / all_ways
  probability = 121 / 177 :=
by
  sorry

end committee_probability_l169_169599


namespace quadrilateral_area_l169_169221

-- Define the angles in the quadrilateral ABCD
def ABD : ℝ := 20
def DBC : ℝ := 60
def ADB : ℝ := 30
def BDC : ℝ := 70

-- Define the side lengths
variables (AB CD AD BC AC BD : ℝ)

-- Prove that the area of the quadrilateral ABCD is half the product of its sides
theorem quadrilateral_area (h1 : ABD = 20) (h2 : DBC = 60) (h3 : ADB = 30) (h4 : BDC = 70)
  : (1 / 2) * (AB * CD + AD * BC) = (1 / 2) * (AB * CD + AD * BC) :=
by
  sorry

end quadrilateral_area_l169_169221


namespace problem_statement_l169_169131

noncomputable def P1 (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def P2 (β : ℝ) : ℝ × ℝ := (Real.cos β, -Real.sin β)
noncomputable def P3 (α β : ℝ) : ℝ × ℝ := (Real.cos (α + β), Real.sin (α + β))
noncomputable def A : ℝ × ℝ := (1, 0)

theorem problem_statement (α β : ℝ) :
  (Prod.fst (P1 α))^2 + (Prod.snd (P1 α))^2 = 1 ∧
  (Prod.fst (P2 β))^2 + (Prod.snd (P2 β))^2 = 1 ∧
  (Prod.fst (P1 α) * Prod.fst (P2 β) + Prod.snd (P1 α) * Prod.snd (P2 β)) = Real.cos (α + β) :=
by
  sorry

end problem_statement_l169_169131


namespace infinite_perfect_squares_of_form_l169_169748

theorem infinite_perfect_squares_of_form (k : ℕ) (hk : 0 < k) : 
  ∃ (n : ℕ), ∀ m : ℕ, ∃ a : ℕ, (n + m) * 2^k - 7 = a^2 :=
sorry

end infinite_perfect_squares_of_form_l169_169748


namespace relationship_l169_169968

noncomputable def a : ℝ := (2 / 5) ^ (2 / 5)
noncomputable def b : ℝ := (3 / 5) ^ (2 / 5)
noncomputable def c : ℝ := Real.logb (3 / 5) (2 / 5)

theorem relationship : a < b ∧ b < c :=
by
  -- proof will go here
  sorry


end relationship_l169_169968


namespace inequality_1_inequality_3_l169_169713

variable (a b : ℝ)
variable (hab : a > b ∧ b ≥ 2)

theorem inequality_1 (hab : a > b ∧ b ≥ 2) : b ^ 2 > 3 * b - a :=
by sorry

theorem inequality_3 (hab : a > b ∧ b ≥ 2) : a * b > a + b :=
by sorry

end inequality_1_inequality_3_l169_169713


namespace determine_pairs_l169_169515

theorem determine_pairs (p q : ℕ) (h : (p + 1)^(p - 1) + (p - 1)^(p + 1) = q^q) : (p = 1 ∧ q = 1) ∨ (p = 2 ∧ q = 2) :=
by
  sorry

end determine_pairs_l169_169515


namespace perpendicular_vectors_dot_product_zero_l169_169650

theorem perpendicular_vectors_dot_product_zero (m : ℝ) :
  let a := (1, 2)
  let b := (m + 1, -m)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 1 :=
by
  intros a b h_eq
  sorry

end perpendicular_vectors_dot_product_zero_l169_169650


namespace coin_difference_l169_169848

-- Definitions based on problem conditions
def denominations : List ℕ := [5, 10, 25, 50]
def amount_owed : ℕ := 55

-- Proof statement
theorem coin_difference :
  let min_coins := 1 + 1 -- one 50-cent coin and one 5-cent coin
  let max_coins := 11 -- eleven 5-cent coins
  max_coins - min_coins = 9 :=
by
  -- Proof details skipped
  sorry

end coin_difference_l169_169848


namespace determine_120_percent_of_y_l169_169302

def x := 0.80 * 350
def y := 0.60 * x
def result := 1.20 * y

theorem determine_120_percent_of_y : result = 201.6 := by
  sorry

end determine_120_percent_of_y_l169_169302


namespace danny_watermelon_slices_l169_169271

theorem danny_watermelon_slices : 
  ∀ (x : ℕ), 3 * x + 15 = 45 -> x = 10 := by
  intros x h
  sorry

end danny_watermelon_slices_l169_169271


namespace find_b_l169_169476

noncomputable def p (x : ℝ) : ℝ := 3 * x - 8
noncomputable def q (x : ℝ) (b : ℝ) : ℝ := 4 * x - b

theorem find_b (b : ℝ) : p (q 3 b) = 10 → b = 6 :=
by
  unfold p q
  intro h
  sorry

end find_b_l169_169476


namespace right_angled_triangle_solution_l169_169596

theorem right_angled_triangle_solution:
  ∃ (a b c : ℕ),
    (a^2 + b^2 = c^2) ∧
    (a + b + c = (a * b) / 2) ∧
    ((a, b, c) = (6, 8, 10) ∨ (a, b, c) = (5, 12, 13)) :=
by
  sorry

end right_angled_triangle_solution_l169_169596


namespace solution_proof_l169_169907

def count_multiples (n : ℕ) (m : ℕ) (limit : ℕ) : ℕ :=
  (limit - 1) / m + 1

def problem_statement : Prop :=
  let multiples_of_10 := count_multiples 1 10 300
  let multiples_of_10_and_6 := count_multiples 1 30 300
  let multiples_of_10_and_11 := count_multiples 1 110 300
  let unwanted_multiples := multiples_of_10_and_6 + multiples_of_10_and_11
  multiples_of_10 - unwanted_multiples = 20

theorem solution_proof : problem_statement :=
  by {
    sorry
  }

end solution_proof_l169_169907


namespace pet_store_has_70_birds_l169_169326

-- Define the given conditions
def num_cages : ℕ := 7
def parrots_per_cage : ℕ := 4
def parakeets_per_cage : ℕ := 3
def cockatiels_per_cage : ℕ := 2
def canaries_per_cage : ℕ := 1

-- Total number of birds in one cage
def birds_per_cage : ℕ := parrots_per_cage + parakeets_per_cage + cockatiels_per_cage + canaries_per_cage

-- Total number of birds in all cages
def total_birds := birds_per_cage * num_cages

-- Prove that the total number of birds is 70
theorem pet_store_has_70_birds : total_birds = 70 :=
sorry

end pet_store_has_70_birds_l169_169326
