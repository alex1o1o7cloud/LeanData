import Mathlib

namespace g_at_neg_1001_l2247_224744

-- Defining the function g and the conditions
def g (x : ℝ) : ℝ := 2.5 * x - 0.5

-- Defining the main theorem to be proved
theorem g_at_neg_1001 : g (-1001) = -2503 := by
  sorry

end g_at_neg_1001_l2247_224744


namespace find_D_l2247_224716

-- This representation assumes 'ABCD' represents digits A, B, C, and D forming a four-digit number.
def four_digit_number (A B C D : ℕ) : ℕ :=
  1000 * A + 100 * B + 10 * C + D

theorem find_D (A B C D : ℕ) (h1 : 1000 * A + 100 * B + 10 * C + D 
                            = 2736) (h2: A ≠ B) (h3: A ≠ C) 
  (h4: A ≠ D) (h5: B ≠ C) (h6: B ≠ D) (h7: C ≠ D) : D = 6 := 
sorry

end find_D_l2247_224716


namespace interior_angles_of_n_plus_4_sided_polygon_l2247_224709

theorem interior_angles_of_n_plus_4_sided_polygon (n : ℕ) (hn : 180 * (n - 2) = 1800) : 
  180 * (n + 4 - 2) = 2520 :=
by sorry

end interior_angles_of_n_plus_4_sided_polygon_l2247_224709


namespace greatest_integer_not_exceeding_1000x_l2247_224721

-- Given the conditions of the problem
variables (x : ℝ)
-- Cond 1: Edge length of the cube
def edge_length := 2
-- Cond 2: Point light source is x centimeters above a vertex
-- Cond 3: Shadow area excluding the area beneath the cube is 98 square centimeters
def shadow_area_excluding_cube := 98
-- This is the condition total area of the shadow
def total_shadow_area := shadow_area_excluding_cube + edge_length ^ 2

-- Statement: Prove that the greatest integer not exceeding 1000x is 8100:
theorem greatest_integer_not_exceeding_1000x (h1 : total_shadow_area = 102) : x ≤ 8.1 :=
by
  sorry

end greatest_integer_not_exceeding_1000x_l2247_224721


namespace phoenix_equal_roots_implies_a_eq_c_l2247_224795

-- Define the "phoenix" equation property
def is_phoenix (a b c : ℝ) : Prop := a + b + c = 0

-- Define the property that a quadratic equation has equal real roots
def has_equal_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

theorem phoenix_equal_roots_implies_a_eq_c (a b c : ℝ) (h₀ : a ≠ 0) 
  (h₁ : is_phoenix a b c) (h₂ : has_equal_real_roots a b c) : a = c :=
sorry

end phoenix_equal_roots_implies_a_eq_c_l2247_224795


namespace temperature_difference_l2247_224740

theorem temperature_difference 
  (lowest: ℤ) (highest: ℤ) 
  (h_lowest : lowest = -4)
  (h_highest : highest = 5) :
  highest - lowest = 9 := 
by
  --relies on the correctness of problem and given simplyifying
  sorry

end temperature_difference_l2247_224740


namespace f_2021_l2247_224745

noncomputable def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)
axiom period_f : ∀ x : ℝ, f (x) = f (2 - x)
axiom f_neg1 : f (-1) = 1

theorem f_2021 : f (2021) = -1 :=
by
  sorry

end f_2021_l2247_224745


namespace find_parallelogram_base_length_l2247_224752

variable (A h b : ℕ)
variable (parallelogram_area : A = 240)
variable (parallelogram_height : h = 10)
variable (area_formula : A = b * h)

theorem find_parallelogram_base_length : b = 24 :=
by
  have h₁ : A = 240 := parallelogram_area
  have h₂ : h = 10 := parallelogram_height
  have h₃ : A = b * h := area_formula
  sorry

end find_parallelogram_base_length_l2247_224752


namespace smallest_positive_period_symmetry_axis_range_of_f_l2247_224786
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 6))

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem symmetry_axis (k : ℤ) :
  ∃ k : ℤ, ∃ x : ℝ, f x = f (x + k * (Real.pi / 2)) ∧ x = (Real.pi / 6) + k * (Real.pi / 2) := sorry

theorem range_of_f : 
  ∀ x, -Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 → -1/2 ≤ f x ∧ f x ≤ 1 := sorry

end smallest_positive_period_symmetry_axis_range_of_f_l2247_224786


namespace dot_product_PA_PB_l2247_224713

theorem dot_product_PA_PB (x_0 : ℝ) (h : x_0 > 0):
  let P := (x_0, x_0 + 2/x_0)
  let A := ((x_0 + 2/x_0) / 2, (x_0 + 2/x_0) / 2)
  let B := (0, x_0 + 2/x_0)
  let vector_PA := ((x_0 + 2/x_0) / 2 - x_0, (x_0 + 2/x_0) / 2 - (x_0 + 2/x_0))
  let vector_PB := (0 - x_0, (x_0 + 2/x_0) - (x_0 + 2/x_0))
  vector_PA.1 * vector_PB.1 + vector_PA.2 * vector_PB.2 = -1 := by
  sorry

end dot_product_PA_PB_l2247_224713


namespace figure_50_squares_l2247_224751

-- Define the quadratic function with the given number of squares for figures 0, 1, 2, and 3.
def g (n : ℕ) : ℕ := 2 * n ^ 2 + 4 * n + 2

-- Prove that the number of nonoverlapping unit squares in figure 50 is 5202.
theorem figure_50_squares : g 50 = 5202 := 
by 
  sorry

end figure_50_squares_l2247_224751


namespace geometric_sequence_a6_l2247_224719

theorem geometric_sequence_a6 (a : ℕ → ℕ) (r : ℕ)
  (h₁ : a 1 = 1)
  (h₄ : a 4 = 8)
  (h_geometric : ∀ n, a n = a 1 * r^(n-1)) : 
  a 6 = 32 :=
by
  sorry

end geometric_sequence_a6_l2247_224719


namespace inscribed_circle_radius_l2247_224706

theorem inscribed_circle_radius (r : ℝ) (R : ℝ) (angle : ℝ):
  R = 6 → angle = 2 * Real.pi / 3 → r = (6 * Real.sqrt 3) / 5 :=
by
  sorry

end inscribed_circle_radius_l2247_224706


namespace caitlins_team_number_l2247_224782

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the two-digit prime numbers
def two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ is_prime n

-- Lean statement
theorem caitlins_team_number (h_date birthday_before today birthday_after : ℕ)
  (p₁ p₂ p₃ : ℕ)
  (h1 : two_digit_prime p₁)
  (h2 : two_digit_prime p₂)
  (h3 : two_digit_prime p₃)
  (h4 : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h5 : p₁ + p₂ = today ∨ p₁ + p₃ = today ∨ p₂ + p₃ = today)
  (h6 : (p₁ + p₂ = birthday_before ∨ p₁ + p₃ = birthday_before ∨ p₂ + p₃ = birthday_before)
       ∧ birthday_before < today)
  (h7 : (p₁ + p₂ = birthday_after ∨ p₁ + p₃ = birthday_after ∨ p₂ + p₃ = birthday_after)
       ∧ birthday_after > today) :
  p₃ = 11 := by
  sorry

end caitlins_team_number_l2247_224782


namespace tom_age_ratio_l2247_224710

variable (T N : ℕ)

theorem tom_age_ratio (h_sum : T = T) (h_relation : T - N = 3 * (T - 3 * N)) : T / N = 4 :=
sorry

end tom_age_ratio_l2247_224710


namespace songs_before_camp_l2247_224753

theorem songs_before_camp (total_songs : ℕ) (learned_at_camp : ℕ) (songs_before_camp : ℕ) (h1 : total_songs = 74) (h2 : learned_at_camp = 18) : songs_before_camp = 56 :=
by
  sorry

end songs_before_camp_l2247_224753


namespace sixth_element_row_20_l2247_224787

theorem sixth_element_row_20 : (Nat.choose 20 5) = 15504 := by
  sorry

end sixth_element_row_20_l2247_224787


namespace no_two_digit_number_divisible_l2247_224711

theorem no_two_digit_number_divisible (a b : ℕ) (distinct : a ≠ b)
  (h₁ : 1 ≤ a ∧ a ≤ 9) (h₂ : 1 ≤ b ∧ b ≤ 9)
  : ¬ ∃ k : ℕ, (1 < k ∧ k ≤ 9) ∧ (10 * a + b = k * (10 * b + a)) :=
by
  sorry

end no_two_digit_number_divisible_l2247_224711


namespace sum_remainders_l2247_224722

theorem sum_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4 + n % 5 = 4) :=
by
  sorry

end sum_remainders_l2247_224722


namespace find_solutions_l2247_224783

def is_solution (a b c d : ℕ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = 1 ∧
  a ∣ (b + c) ∧
  b ∣ (c + d) ∧
  c ∣ (d + a) ∧
  d ∣ (a + b)

theorem find_solutions : ∀ (a b c d : ℕ),
  is_solution a b c d →
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 1) ∨
  (a = 5 ∧ b = 3 ∧ c = 2 ∧ d = 1) ∨
  (a = 5 ∧ b = 4 ∧ c = 1 ∧ d = 3) ∨
  (a = 7 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
  (a = 3 ∧ b = 1 ∧ c = 2 ∧ d = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = 4 ∧ d = 3) ∨
  (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 1) ∨
  (a = 7 ∧ b = 2 ∧ c = 5 ∧ d = 3) ∨
  (a = 7 ∧ b = 3 ∧ c = 4 ∧ d = 5) :=
by
  intros a b c d h
  sorry

end find_solutions_l2247_224783


namespace slope_of_line_l2247_224788

-- Definition of the line equation in slope-intercept form
def line_eq (x : ℝ) : ℝ := -5 * x + 9

-- Statement: The slope of the line y = -5x + 9 is -5
theorem slope_of_line : (∀ x : ℝ, ∃ m b : ℝ, line_eq x = m * x + b ∧ m = -5) :=
by
  -- proof goes here
  sorry

end slope_of_line_l2247_224788


namespace tan_angle_add_l2247_224798

theorem tan_angle_add (x : ℝ) (h : Real.tan x = -3) : Real.tan (x + Real.pi / 6) = 2 * Real.sqrt 3 + 1 := 
by
  sorry

end tan_angle_add_l2247_224798


namespace midpoint_of_intersection_l2247_224739

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 2 * t)

noncomputable def polar_curve (θ : ℝ) : ℝ :=
  2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2)

theorem midpoint_of_intersection :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, 
      A = parametric_line t₁ ∧ 
      B = parametric_line t₂ ∧ 
      (A.1 ^ 2 / 4 + A.2 ^ 2 = 1) ∧ 
      (B.1 ^ 2 / 4 + B.2 ^ 2 = 1)) ∧
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (4 / 5, -1 / 5) :=
sorry

end midpoint_of_intersection_l2247_224739


namespace slope_of_arithmetic_sequence_l2247_224774

variable {α : Type*} [LinearOrderedField α]

noncomputable def S (a_1 d n : α) : α := n * a_1 + n * (n-1) / 2 * d

theorem slope_of_arithmetic_sequence (a_1 d n : α) 
  (hS2 : S a_1 d 2 = 10)
  (hS5 : S a_1 d 5 = 55)
  : (a_1 + 2 * d - a_1) / 2 = 4 :=
by
  sorry

end slope_of_arithmetic_sequence_l2247_224774


namespace similar_triangle_side_length_l2247_224761

theorem similar_triangle_side_length
  (A_1 A_2 : ℕ)
  (area_diff : A_1 - A_2 = 32)
  (area_ratio : A_1 = 9 * A_2)
  (side_small_triangle : ℕ)
  (side_small_triangle_eq : side_small_triangle = 5)
  (side_ratio : ∃ r : ℕ, r = 3) :
  ∃ side_large_triangle : ℕ, side_large_triangle = side_small_triangle * 3 := by
sorry

end similar_triangle_side_length_l2247_224761


namespace B_share_correct_l2247_224729

noncomputable def total_share : ℕ := 120
noncomputable def B_share : ℕ := 20
noncomputable def A_share (x : ℕ) : ℕ := x + 20
noncomputable def C_share (x : ℕ) : ℕ := x + 40

theorem B_share_correct : ∃ x : ℕ, total_share = (A_share x) + x + (C_share x) ∧ x = B_share := by
  sorry

end B_share_correct_l2247_224729


namespace factorize_polynomial_l2247_224742

theorem factorize_polynomial (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3) ^ 2 :=
by sorry

end factorize_polynomial_l2247_224742


namespace part1_l2247_224746

theorem part1 (a n : ℕ) (hne : a % 2 = 1) : (4 ∣ a^n - 1) → (n % 2 = 0) :=
by
  sorry

end part1_l2247_224746


namespace milburg_children_count_l2247_224765

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 → grown_ups = 5256 → 
  (total_population - grown_ups) = 2987 :=
by
  intros total_population grown_ups h1 h2
  sorry

end milburg_children_count_l2247_224765


namespace freshmen_more_than_sophomores_l2247_224790

theorem freshmen_more_than_sophomores :
  ∀ (total_students juniors not_sophomores not_freshmen seniors adv_grade freshmen sophomores : ℕ),
    total_students = 1200 →
    juniors = 264 →
    not_sophomores = 660 →
    not_freshmen = 300 →
    seniors = 240 →
    adv_grade = 20 →
    freshmen = total_students - not_freshmen - seniors - adv_grade →
    sophomores = total_students - not_sophomores - seniors - adv_grade →
    freshmen - sophomores = 360 :=
by
  intros total_students juniors not_sophomores not_freshmen seniors adv_grade freshmen sophomores
  intros h_total h_juniors h_not_sophomores h_not_freshmen h_seniors h_adv_grade h_freshmen h_sophomores
  sorry

end freshmen_more_than_sophomores_l2247_224790


namespace probability_of_defective_product_l2247_224794

theorem probability_of_defective_product :
  let total_products := 10
  let defective_products := 2
  (defective_products: ℚ) / total_products = 1 / 5 :=
by
  let total_products := 10
  let defective_products := 2
  have h : (defective_products: ℚ) / total_products = 1 / 5
  {
    exact sorry
  }
  exact h

end probability_of_defective_product_l2247_224794


namespace min_voters_for_Tall_victory_l2247_224785

def total_voters := 105
def districts := 5
def sections_per_district := 7
def voters_per_section := 3
def sections_to_win_district := 4
def districts_to_win := 3
def sections_to_win := sections_to_win_district * districts_to_win
def min_voters_to_win_section := 2

theorem min_voters_for_Tall_victory : 
  (total_voters = 105) ∧ 
  (districts = 5) ∧ 
  (sections_per_district = 7) ∧ 
  (voters_per_section = 3) ∧ 
  (sections_to_win_district = 4) ∧ 
  (districts_to_win = 3) 
  → 
  min_voters_to_win_section * sections_to_win = 24 :=
by
  sorry
  
end min_voters_for_Tall_victory_l2247_224785


namespace value_of_expression_at_three_l2247_224777

theorem value_of_expression_at_three (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 := 
by
  sorry

end value_of_expression_at_three_l2247_224777


namespace value_x_plus_2y_plus_3z_l2247_224793

variable (x y z : ℝ)

theorem value_x_plus_2y_plus_3z :
  x + y = 5 →
  z^2 = x * y + y - 9 →
  x + 2 * y + 3 * z = 8 :=
by
  intro h1 h2
  sorry

end value_x_plus_2y_plus_3z_l2247_224793


namespace abs_eq_neg_iff_nonpositive_l2247_224759

theorem abs_eq_neg_iff_nonpositive (x : ℝ) : |x| = -x ↔ x ≤ 0 := by
  sorry

end abs_eq_neg_iff_nonpositive_l2247_224759


namespace digit_makes_57A2_divisible_by_9_l2247_224736

theorem digit_makes_57A2_divisible_by_9 (A : ℕ) (h : 0 ≤ A ∧ A ≤ 9) : 
  (5 + 7 + A + 2) % 9 = 0 ↔ A = 4 :=
by
  sorry

end digit_makes_57A2_divisible_by_9_l2247_224736


namespace toms_remaining_speed_l2247_224771

-- Defining the constants and conditions
def total_distance : ℝ := 100
def first_leg_distance : ℝ := 50
def first_leg_speed : ℝ := 20
def avg_speed : ℝ := 28.571428571428573

-- Proving Tom's speed during the remaining part of the trip
theorem toms_remaining_speed :
  ∃ (remaining_leg_speed : ℝ),
    (remaining_leg_speed = 50) ∧
    (total_distance = first_leg_distance + 50) ∧
    ((first_leg_distance / first_leg_speed + 50 / remaining_leg_speed) = total_distance / avg_speed) :=
by
  sorry

end toms_remaining_speed_l2247_224771


namespace more_than_half_millet_on_day_5_l2247_224748

noncomputable def millet_amount (n : ℕ) : ℚ :=
  1 - (3 / 4)^n

theorem more_than_half_millet_on_day_5 : millet_amount 5 > 1 / 2 :=
by
  sorry

end more_than_half_millet_on_day_5_l2247_224748


namespace ratio_water_to_orange_juice_l2247_224762

variable (O W : ℝ)

-- Conditions:
-- 1. Amount of orange juice is O for both days.
-- 2. Amount of water is W on the first day and 2W on the second day.
-- 3. Price per glass is $0.60 on the first day and $0.40 on the second day.

theorem ratio_water_to_orange_juice 
  (h : (O + W) * 0.60 = (O + 2 * W) * 0.40) : 
  W / O = 1 := 
by 
  -- The proof is skipped
  sorry

end ratio_water_to_orange_juice_l2247_224762


namespace employee_gross_pay_l2247_224775

theorem employee_gross_pay
  (pay_rate_regular : ℝ) (pay_rate_overtime : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ)
  (h1 : pay_rate_regular = 11.25)
  (h2 : pay_rate_overtime = 16)
  (h3 : regular_hours = 40)
  (h4 : overtime_hours = 10.75) :
  (pay_rate_regular * regular_hours + pay_rate_overtime * overtime_hours = 622) :=
by
  sorry

end employee_gross_pay_l2247_224775


namespace algorithm_output_l2247_224712

theorem algorithm_output (x y: Int) (h_x: x = -5) (h_y: y = 15) : 
  let x := if x < 0 then y + 3 else x;
  x - y = 3 ∧ x + y = 33 :=
by
  sorry

end algorithm_output_l2247_224712


namespace geometric_sequence_S6_l2247_224724

-- We first need to ensure our definitions match the given conditions.
noncomputable def a1 : ℝ := 1 -- root of x^2 - 5x + 4 = 0
noncomputable def a3 : ℝ := 4 -- root of x^2 - 5x + 4 = 0

-- Definition of the geometric sequence
noncomputable def q : ℝ := 2 -- common ratio derived from geometric sequence where a3 = a1 * q^2

-- Definition of the n-th term of the geometric sequence
noncomputable def a (n : ℕ) : ℝ := a1 * q^((n : ℝ) - 1)

-- Definition of the sum of the first n terms of the geometric sequence
noncomputable def S (n : ℕ) : ℝ := (a1 * (1 - q^n)) / (1 - q)

-- The theorem we want to prove
theorem geometric_sequence_S6 : S 6 = 63 :=
  by sorry

end geometric_sequence_S6_l2247_224724


namespace geometric_sequence_sum_q_value_l2247_224714

theorem geometric_sequence_sum_q_value (q : ℝ) (a S : ℕ → ℝ) :
  a 1 = 4 →
  (∀ n, a (n+1) = a n * q ) →
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →
  (∀ n, (S n + 2) = (S 1 + 2) * (q ^ (n - 1))) →
  q = 3
:= 
by
  sorry

end geometric_sequence_sum_q_value_l2247_224714


namespace division_remainder_l2247_224781

theorem division_remainder :
  (1225 * 1227 * 1229) % 12 = 3 :=
by sorry

end division_remainder_l2247_224781


namespace order_of_abc_l2247_224732

noncomputable def a : ℝ := (1 / 3) * Real.logb 2 (1 / 4)
noncomputable def b : ℝ := 1 - Real.logb 2 3
noncomputable def c : ℝ := Real.cos (5 * Real.pi / 6)

theorem order_of_abc : c < a ∧ a < b := by
  sorry

end order_of_abc_l2247_224732


namespace average_weight_of_class_is_61_67_l2247_224743

noncomputable def totalWeightA (avgWeightA : ℝ) (numStudentsA : ℕ) : ℝ := avgWeightA * numStudentsA
noncomputable def totalWeightB (avgWeightB : ℝ) (numStudentsB : ℕ) : ℝ := avgWeightB * numStudentsB
noncomputable def totalWeightClass (totalWeightA : ℝ) (totalWeightB : ℝ) : ℝ := totalWeightA + totalWeightB
noncomputable def totalStudentsClass (numStudentsA : ℕ) (numStudentsB : ℕ) : ℕ := numStudentsA + numStudentsB
noncomputable def averageWeightClass (totalWeightClass : ℝ) (totalStudentsClass : ℕ) : ℝ := totalWeightClass / totalStudentsClass

theorem average_weight_of_class_is_61_67 :
  averageWeightClass (totalWeightClass (totalWeightA 50 50) (totalWeightB 70 70))
    (totalStudentsClass 50 70) = 61.67 := by
  sorry

end average_weight_of_class_is_61_67_l2247_224743


namespace Q_share_of_profit_l2247_224734

theorem Q_share_of_profit (P Q T : ℕ) (hP : P = 54000) (hQ : Q = 36000) (hT : T = 18000) : Q's_share = 7200 :=
by
  -- Definitions and conditions
  let P := 54000
  let Q := 36000
  let T := 18000
  have P_ratio := 3
  have Q_ratio := 2
  have ratio_sum := P_ratio + Q_ratio
  have Q's_share := (T * Q_ratio) / ratio_sum
  
  -- Q's share of the profit
  sorry

end Q_share_of_profit_l2247_224734


namespace seonmi_initial_money_l2247_224796

theorem seonmi_initial_money (M : ℝ) (h1 : M/6 = 250) : M = 1500 :=
by
  sorry

end seonmi_initial_money_l2247_224796


namespace complex_num_sum_l2247_224727

def is_complex_num (a b : ℝ) (z : ℂ) : Prop :=
  z = a + b * Complex.I

theorem complex_num_sum (a b : ℝ) (z : ℂ) (h : is_complex_num a b z) :
  z = (1 - Complex.I) ^ 2 / (1 + Complex.I) → a + b = -2 :=
by
  sorry

end complex_num_sum_l2247_224727


namespace total_mice_eaten_in_decade_l2247_224700

-- Define the number of weeks in a year
def weeks_in_year (is_leap : Bool) : ℕ := if is_leap then 52 else 52

-- Define the number of mice eaten in the first year
def mice_first_year :
  ℕ := weeks_in_year false / 4

-- Define the number of mice eaten in the second year
def mice_second_year :
  ℕ := weeks_in_year false / 3

-- Define the number of mice eaten per year for years 3 to 10
def mice_per_year :
  ℕ := weeks_in_year false / 2

-- Define the total mice eaten in eight years (years 3 to 10)
def mice_eight_years :
  ℕ := 8 * mice_per_year

-- Define the total mice eaten over a decade
def total_mice_eaten :
  ℕ := mice_first_year + mice_second_year + mice_eight_years

-- Theorem to check if the total number of mice equals 238
theorem total_mice_eaten_in_decade :
  total_mice_eaten = 238 :=
by
  -- Calculation for the total number of mice
  sorry

end total_mice_eaten_in_decade_l2247_224700


namespace mod_arith_proof_l2247_224799

theorem mod_arith_proof (m : ℕ) (hm1 : 0 ≤ m) (hm2 : m < 50) : 198 * 935 % 50 = 30 := 
by
  sorry

end mod_arith_proof_l2247_224799


namespace solution_set_of_inequality_l2247_224789

theorem solution_set_of_inequality : {x : ℝ | x^2 < 2 * x} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l2247_224789


namespace num_integers_satisfying_inequality_l2247_224766

theorem num_integers_satisfying_inequality :
  ∃ (x : ℕ), ∀ (y: ℤ), (-3 ≤ 3 * y + 2 → 3 * y + 2 ≤ 8) ↔ 4 = x :=
by
  sorry

end num_integers_satisfying_inequality_l2247_224766


namespace angle_relationship_l2247_224704

variables {AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1}
variables {angleA angleA1 angleB angleB1 angleC angleC1 angleD angleD1 : ℝ}

-- Define the conditions
def conditions (AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 : ℝ)
  (angleA angleA1 : ℝ) : Prop :=
  AB = A_1B_1 ∧ BC = B_1C_1 ∧ CD = C_1D_1 ∧ DA = D_1A_1 ∧ angleA > angleA1

theorem angle_relationship (AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 : ℝ)
  (angleA angleA1 angleB angleB1 angleC angleC1 angleD angleD1 : ℝ)
  (h : conditions AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 angleA angleA1) :
  angleB < angleB1 ∧ angleC > angleC1 ∧ angleD < angleD1 :=
by {
  sorry
}

end angle_relationship_l2247_224704


namespace product_of_two_special_numbers_is_perfect_square_l2247_224778

-- Define the structure of the required natural numbers
structure SpecialNumber where
  m : ℕ
  n : ℕ
  value : ℕ := 2^m * 3^n

-- The main theorem to be proved
theorem product_of_two_special_numbers_is_perfect_square :
  ∀ (a b c d e : SpecialNumber),
  ∃ x y : SpecialNumber, ∃ k : ℕ, (x.value * y.value) = k * k :=
by
  sorry

end product_of_two_special_numbers_is_perfect_square_l2247_224778


namespace collinear_condition_l2247_224756

variable {R : Type*} [LinearOrderedField R]
variable {x1 y1 x2 y2 x3 y3 : R}

theorem collinear_condition : 
  x1 * y2 + x2 * y3 + x3 * y1 = y1 * x2 + y2 * x3 + y3 * x1 →
  ∃ k l m : R, k * (x2 - x1) = l * (y2 - y1) ∧ k * (x3 - x1) = m * (y3 - y1) :=
by
  sorry

end collinear_condition_l2247_224756


namespace curve_intersections_l2247_224730

theorem curve_intersections (m : ℝ) :
  (∃ x y : ℝ, ((x-1)^2 + y^2 = 1) ∧ (y = mx + m) ∧ (y ≠ 0) ∧ (y^2 = 0)) =
  ((m > -Real.sqrt 3 / 3) ∧ (m < 0)) ∨ ((m > 0) ∧ (m < Real.sqrt 3 / 3)) := 
sorry

end curve_intersections_l2247_224730


namespace g_18_equals_5832_l2247_224708

noncomputable def g (n : ℕ) : ℕ := sorry

axiom cond1 : ∀ (n : ℕ), (0 < n) → g (n + 1) > g n
axiom cond2 : ∀ (m n : ℕ), (0 < m ∧ 0 < n) → g (m * n) = g m * g n
axiom cond3 : ∀ (m n : ℕ), (0 < m ∧ 0 < n ∧ m ≠ n ∧ m^2 = n^3) → (g m = n ∨ g n = m)

theorem g_18_equals_5832 : g 18 = 5832 :=
by sorry

end g_18_equals_5832_l2247_224708


namespace suraj_innings_count_l2247_224728

theorem suraj_innings_count
  (A : ℕ := 24)  -- average before the last innings
  (new_average : ℕ := 28)  -- Suraj’s average after the last innings
  (last_score : ℕ := 92)  -- Suraj’s score in the last innings
  (avg_increase : ℕ := 4)  -- the increase in average after the last innings
  (n : ℕ)  -- number of innings before the last one
  (h_avg : A + avg_increase = new_average)  -- A + 4 = 28
  (h_eqn : n * A + last_score = (n + 1) * new_average) :  -- n * 24 + 92 = (n + 1) * 28
  n = 16 :=
by {
  sorry
}

end suraj_innings_count_l2247_224728


namespace solve_eq1_solve_eq2_l2247_224702

theorem solve_eq1 (x : ℝ) :
  3 * x^2 - 11 * x + 9 = 0 ↔ x = (11 + Real.sqrt 13) / 6 ∨ x = (11 - Real.sqrt 13) / 6 :=
by
  sorry

theorem solve_eq2 (x : ℝ) :
  5 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 / 2 :=
by
  sorry

end solve_eq1_solve_eq2_l2247_224702


namespace pirate_loot_l2247_224780

theorem pirate_loot (a b c d e : ℕ) (h1 : a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1 ∨ e = 1)
  (h2 : a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 ∨ e = 2)
  (h3 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h4 : a + b = 2 * (c + d) ∨ b + c = 2 * (a + e)) :
  (a, b, c, d, e) = (1, 1, 1, 1, 2) ∨ 
  (a, b, c, d, e) = (1, 1, 2, 2, 2) ∨
  (a, b, c, d, e) = (1, 2, 3, 3, 3) ∨
  (a, b, c, d, e) = (1, 2, 2, 2, 3) :=
sorry

end pirate_loot_l2247_224780


namespace subtraction_example_l2247_224754

theorem subtraction_example : -1 - 3 = -4 := 
  sorry

end subtraction_example_l2247_224754


namespace upper_limit_of_x_l2247_224770

theorem upper_limit_of_x 
  {x : ℤ} 
  (h1 : 0 < x) 
  (h2 : x < 15) 
  (h3 : -1 < x) 
  (h4 : x < 5) 
  (h5 : 0 < x) 
  (h6 : x < 3) 
  (h7 : x + 2 < 4) 
  (h8 : x = 1) : 
  0 < x ∧ x < 2 := 
by 
  sorry

end upper_limit_of_x_l2247_224770


namespace ratio_of_7th_terms_l2247_224776

theorem ratio_of_7th_terms (a b : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : ∀ n, T n = n * (b 1 + b n) / 2)
  (h3 : ∀ n, S n / T n = (5 * n + 10) / (2 * n - 1)) :
  a 7 / b 7 = 3 :=
by
  sorry

end ratio_of_7th_terms_l2247_224776


namespace polynomial_value_l2247_224757

theorem polynomial_value (x y : ℝ) (h : x - 2 * y + 3 = 8) : x - 2 * y = 5 :=
by
  sorry

end polynomial_value_l2247_224757


namespace sheets_taken_l2247_224703

noncomputable def remaining_sheets_mean (b c : ℕ) : ℚ :=
  (b * (2 * b + 1) + (100 - 2 * (b + c)) * (2 * (b + c) + 101)) / 2 / (100 - 2 * c)

theorem sheets_taken (b c : ℕ) (h1 : 100 = 2 * 50) 
(h2 : ∀ n, n > 0 → 2 * n = n + n) 
(hmean : remaining_sheets_mean b c = 31) : 
  c = 17 := 
sorry

end sheets_taken_l2247_224703


namespace max_geometric_sequence_sum_l2247_224737

theorem max_geometric_sequence_sum (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a * b * c = 216) (h4 : ∃ r : ℕ, b = a * r ∧ c = b * r) : 
  a + b + c ≤ 43 :=
sorry

end max_geometric_sequence_sum_l2247_224737


namespace sufficient_condition_implies_range_of_p_l2247_224792

open Set Real

theorem sufficient_condition_implies_range_of_p (p : ℝ) :
  (∀ x : ℝ, 4 * x + p < 0 → x^2 - x - 2 > 0) →
  (∃ x : ℝ, x^2 - x - 2 > 0 ∧ ¬ (4 * x + p < 0)) →
  p ∈ Set.Ici 4 :=
by
  sorry

end sufficient_condition_implies_range_of_p_l2247_224792


namespace keith_and_jason_books_l2247_224725

theorem keith_and_jason_books :
  let K := 20
  let J := 21
  K + J = 41 :=
by
  sorry

end keith_and_jason_books_l2247_224725


namespace correct_inequality_l2247_224772

-- Define the conditions
variables (a b : ℝ)
variable (h : a > 1 ∧ 1 > b ∧ b > 0)

-- State the theorem to prove
theorem correct_inequality (h : a > 1 ∧ 1 > b ∧ b > 0) : 
  (1 / Real.log a) > (1 / Real.log b) :=
sorry

end correct_inequality_l2247_224772


namespace four_digit_number_2010_l2247_224735

theorem four_digit_number_2010 (a b c d : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 1000 ≤ 1000 * a + 100 * b + 10 * c + d ∧
        1000 * a + 100 * b + 10 * c + d < 10000)
  (h_eq : a * (a + b + c + d) * (a^2 + b^2 + c^2 + d^2) * (a^6 + 2 * b^6 + 3 * c^6 + 4 * d^6)
          = 1000 * a + 100 * b + 10 * c + d)
  : 1000 * a + 100 * b + 10 * c + d = 2010 :=
sorry

end four_digit_number_2010_l2247_224735


namespace smallest_positive_period_pi_increasing_intervals_in_0_to_pi_range_of_m_l2247_224791

noncomputable def f (x m : ℝ) := 2 * (Real.cos x) ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + m

theorem smallest_positive_period_pi (m : ℝ) :
  ∀ x : ℝ, f (x + π) m = f x m := sorry

theorem increasing_intervals_in_0_to_pi (m : ℝ) :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 6) ∨ (2 * π / 3 ≤ x ∧ x ≤ π) →
  ∀ y : ℝ, ((0 ≤ y ∧ y ≤ π / 6 ∨ (2 * π / 3 ≤ y ∧ y ≤ π)) ∧ x < y) → f x m < f y m := sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 6) → -4 < f x m ∧ f x m < 4) ↔ (-6 < m ∧ m < 1) := sorry

end smallest_positive_period_pi_increasing_intervals_in_0_to_pi_range_of_m_l2247_224791


namespace science_and_technology_group_total_count_l2247_224758

theorem science_and_technology_group_total_count 
  (number_of_girls : ℕ)
  (number_of_boys : ℕ)
  (h1 : number_of_girls = 18)
  (h2 : number_of_girls = 2 * number_of_boys - 2)
  : number_of_girls + number_of_boys = 28 := 
by
  sorry

end science_and_technology_group_total_count_l2247_224758


namespace sally_book_pages_l2247_224720

def pages_read_weekdays (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def pages_read_weekends (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def total_pages (weekdays: ℕ) (weekends: ℕ) (pages_weekdays: ℕ) (pages_weekends: ℕ): ℕ :=
  pages_read_weekdays weekdays pages_weekdays + pages_read_weekends weekends pages_weekends

theorem sally_book_pages :
  total_pages 10 4 10 20 = 180 :=
sorry

end sally_book_pages_l2247_224720


namespace tiling_possible_with_one_type_l2247_224784

theorem tiling_possible_with_one_type
  {a b m n : ℕ} (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n)
  (H : (∃ (k : ℕ), a = k * n) ∨ (∃ (l : ℕ), b = l * m)) :
  (∃ (i : ℕ), a = i * n) ∨ (∃ (j : ℕ), b = j * m) :=
  sorry

end tiling_possible_with_one_type_l2247_224784


namespace total_hockey_games_l2247_224717

theorem total_hockey_games (games_per_month : ℕ) (months_in_season : ℕ) 
(h1 : games_per_month = 13) (h2 : months_in_season = 14) : 
games_per_month * months_in_season = 182 := 
by
  -- we can simplify using the given conditions
  sorry

end total_hockey_games_l2247_224717


namespace max_tan_B_l2247_224705

theorem max_tan_B (A B : ℝ) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2) (h : Real.tan (A + B) = 2 * Real.tan A) :
  ∃ B_max, B_max = Real.tan B ∧ B_max ≤ Real.sqrt 2 / 4 :=
by
  sorry

end max_tan_B_l2247_224705


namespace minimum_doors_to_safety_l2247_224773

-- Definitions in Lean 4 based on the conditions provided
def spaceship (corridors : ℕ) : Prop := corridors = 23

def command_closes (N : ℕ) (corridors : ℕ) : Prop := N ≤ corridors

-- Theorem based on the question and conditions
theorem minimum_doors_to_safety (N : ℕ) (corridors : ℕ)
  (h_corridors : spaceship corridors)
  (h_command : command_closes N corridors) :
  N = 22 :=
sorry

end minimum_doors_to_safety_l2247_224773


namespace lily_patch_cover_entire_lake_l2247_224764

noncomputable def days_to_cover_half (initial_days : ℕ) := 33

theorem lily_patch_cover_entire_lake (initial_days : ℕ) (h : days_to_cover_half initial_days = 33) :
  initial_days + 1 = 34 :=
by
  sorry

end lily_patch_cover_entire_lake_l2247_224764


namespace inequality_nonnegative_reals_l2247_224760

theorem inequality_nonnegative_reals (a b c : ℝ) (h_a : 0 ≤ a) (h_b : 0 ≤ b) (h_c : 0 ≤ c) :
  |(c * a - a * b)| + |(a * b - b * c)| + |(b * c - c * a)| ≤ |(b^2 - c^2)| + |(c^2 - a^2)| + |(a^2 - b^2)| :=
by
  sorry

end inequality_nonnegative_reals_l2247_224760


namespace sandy_shopping_l2247_224741

theorem sandy_shopping (T : ℝ) (h : 0.70 * T = 217) : T = 310 := sorry

end sandy_shopping_l2247_224741


namespace alicia_candies_problem_l2247_224755

theorem alicia_candies_problem :
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ (n % 9 = 7) ∧ (n % 7 = 5) ∧ n = 124 :=
by
  sorry

end alicia_candies_problem_l2247_224755


namespace wendy_chocolates_l2247_224733

theorem wendy_chocolates (h : ℕ) : 
  let chocolates_per_4_hours := 1152
  let chocolates_per_hour := chocolates_per_4_hours / 4
  (chocolates_per_hour * h) = 288 * h :=
by
  sorry

end wendy_chocolates_l2247_224733


namespace angle_measure_l2247_224726

theorem angle_measure (x : ℝ) (h1 : 180 - x = 6 * (90 - x)) : x = 72 := by
  sorry

end angle_measure_l2247_224726


namespace combined_age_in_years_l2247_224769

theorem combined_age_in_years (years : ℕ) (adam_age : ℕ) (tom_age : ℕ) (target_age : ℕ) :
  adam_age = 8 → tom_age = 12 → target_age = 44 → (adam_age + tom_age) + 2 * years = target_age → years = 12 :=
by
  intros h_adam h_tom h_target h_combined
  rw [h_adam, h_tom, h_target] at h_combined
  linarith

end combined_age_in_years_l2247_224769


namespace technical_class_average_age_l2247_224707

noncomputable def average_age_in_technical_class : ℝ :=
  let average_age_arts := 21
  let num_arts_classes := 8
  let num_technical_classes := 5
  let overall_average_age := 19.846153846153847
  let total_classes := num_arts_classes + num_technical_classes
  let total_age_university := overall_average_age * total_classes
  ((total_age_university - (average_age_arts * num_arts_classes)) / num_technical_classes)

theorem technical_class_average_age :
  average_age_in_technical_class = 990.4 :=
by
  sorry  -- Proof to be provided

end technical_class_average_age_l2247_224707


namespace part_I_part_II_l2247_224718

noncomputable def f (x : ℝ) : ℝ := abs x

theorem part_I (x : ℝ) : f (x-1) > 2 ↔ x < -1 ∨ x > 3 := 
by sorry

theorem part_II (x y z : ℝ) (h : f x ^ 2 + y ^ 2 + z ^ 2 = 9) : ∃ (min_val : ℝ), min_val = -9 ∧ ∀ (a b c : ℝ), f a ^ 2 + b ^ 2 + c ^ 2 = 9 → (a + 2 * b + 2 * c) ≥ min_val := 
by sorry

end part_I_part_II_l2247_224718


namespace bhanu_house_rent_expenditure_l2247_224767

variable (Income house_rent_expenditure petrol_expenditure remaining_income : ℝ)
variable (h1 : petrol_expenditure = (30 / 100) * Income)
variable (h2 : remaining_income = Income - petrol_expenditure)
variable (h3 : house_rent_expenditure = (20 / 100) * remaining_income)
variable (h4 : petrol_expenditure = 300)

theorem bhanu_house_rent_expenditure :
  house_rent_expenditure = 140 :=
by sorry

end bhanu_house_rent_expenditure_l2247_224767


namespace find_x1_value_l2247_224750

theorem find_x1_value (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3) (h2 : x3 ≤ x2) (h3 : x2 ≤ x1) (h4 : x1 ≤ 1) 
  (h_eq : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1 / 3) : 
  x1 = 2 / 3 := 
sorry

end find_x1_value_l2247_224750


namespace cos_of_sin_given_l2247_224738

theorem cos_of_sin_given (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by
  sorry

end cos_of_sin_given_l2247_224738


namespace gcd_eq_gcd_of_division_l2247_224731

theorem gcd_eq_gcd_of_division (a b q r : ℕ) (h1 : a = b * q + r) (h2 : 0 < r) (h3 : r < b) (h4 : a > b) : 
  Nat.gcd a b = Nat.gcd b r :=
by
  sorry

end gcd_eq_gcd_of_division_l2247_224731


namespace shaded_area_percentage_is_100_l2247_224701

-- Definitions and conditions
def square_side := 6
def square_area := square_side * square_side

def rect1_area := 2 * 2
def rect2_area := (5 * 5) - (3 * 3)
def rect3_area := 6 * 6

-- Percentage shaded calculation
def shaded_area := square_area
def percentage_shaded := (shaded_area / square_area) * 100

-- Lean 4 statement for the problem
theorem shaded_area_percentage_is_100 :
  percentage_shaded = 100 :=
by
  sorry

end shaded_area_percentage_is_100_l2247_224701


namespace nonpositive_sum_of_products_l2247_224715

theorem nonpositive_sum_of_products {a b c d : ℝ} (h : a + b + c + d = 0) :
  ab + ac + ad + bc + bd + cd ≤ 0 :=
sorry

end nonpositive_sum_of_products_l2247_224715


namespace equation_represents_point_l2247_224797

theorem equation_represents_point 
  (a b x y : ℝ) 
  (h : (x - a) ^ 2 + (y + b) ^ 2 = 0) : 
  x = a ∧ y = -b := 
by
  sorry

end equation_represents_point_l2247_224797


namespace cost_price_l2247_224779

theorem cost_price (MP : ℝ) (SP : ℝ) (C : ℝ) 
  (h1 : MP = 87.5) 
  (h2 : SP = 0.95 * MP) 
  (h3 : SP = 1.25 * C) : 
  C = 66.5 := 
by
  sorry

end cost_price_l2247_224779


namespace am_gm_inequality_l2247_224763

theorem am_gm_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (x + y + z)^2 / 3 ≥ x * Real.sqrt (y * z) + y * Real.sqrt (z * x) + z * Real.sqrt (x * y) := 
by sorry

end am_gm_inequality_l2247_224763


namespace numbers_must_be_equal_l2247_224749

theorem numbers_must_be_equal
  (n : ℕ) (nums : Fin n → ℕ)
  (hn_pos : n = 99)
  (hbound : ∀ i, nums i < 100)
  (hdiv : ∀ (s : Finset (Fin n)) (hs : 2 ≤ s.card), ¬ 100 ∣ s.sum nums) :
  ∀ i j, nums i = nums j := 
sorry

end numbers_must_be_equal_l2247_224749


namespace quadrilateral_parallelogram_iff_l2247_224768

variable (a b c d e f MN : ℝ)

-- Define a quadrilateral as a structure with sides and diagonals 
structure Quadrilateral :=
  (a b c d e f : ℝ)

-- Define the condition: sum of squares of diagonals equals sum of squares of sides
def sum_of_squares_condition (q : Quadrilateral) : Prop :=
  q.e ^ 2 + q.f ^ 2 = q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2

-- Define what it means for a quadrilateral to be a parallelogram:
-- Midpoints of the diagonals coincide (MN = 0)
def is_parallelogram (q : Quadrilateral) (MN : ℝ) : Prop :=
  MN = 0

-- Main theorem to prove
theorem quadrilateral_parallelogram_iff (q : Quadrilateral) (MN : ℝ) :
  is_parallelogram q MN ↔ sum_of_squares_condition q :=
sorry

end quadrilateral_parallelogram_iff_l2247_224768


namespace complex_roots_equilateral_l2247_224747

noncomputable def omega : ℂ := -1/2 + Complex.I * Real.sqrt 3 / 2

theorem complex_roots_equilateral (z1 z2 p q : ℂ) (h₁ : z2 = omega * z1) (h₂ : -p = (1 + omega) * z1) (h₃ : q = omega * z1 ^ 2) :
  p^2 / q = 1 + Complex.I * Real.sqrt 3 :=
by sorry

end complex_roots_equilateral_l2247_224747


namespace problem_solution_l2247_224723

theorem problem_solution : (121^2 - 110^2) / 11 = 231 := 
by
  sorry

end problem_solution_l2247_224723
