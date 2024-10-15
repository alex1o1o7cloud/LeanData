import Mathlib

namespace NUMINAMATH_GPT_faye_complete_bouquets_l556_55602

theorem faye_complete_bouquets :
  let roses_initial := 48
  let lilies_initial := 40
  let tulips_initial := 76
  let sunflowers_initial := 34
  let roses_wilted := 24
  let lilies_wilted := 10
  let tulips_wilted := 14
  let sunflowers_wilted := 7
  let roses_remaining := roses_initial - roses_wilted
  let lilies_remaining := lilies_initial - lilies_wilted
  let tulips_remaining := tulips_initial - tulips_wilted
  let sunflowers_remaining := sunflowers_initial - sunflowers_wilted
  let bouquets_roses := roses_remaining / 2
  let bouquets_lilies := lilies_remaining
  let bouquets_tulips := tulips_remaining / 3
  let bouquets_sunflowers := sunflowers_remaining
  let bouquets := min (min bouquets_roses bouquets_lilies) (min bouquets_tulips bouquets_sunflowers)
  bouquets = 12 :=
by
  sorry

end NUMINAMATH_GPT_faye_complete_bouquets_l556_55602


namespace NUMINAMATH_GPT_mod_product_l556_55671

theorem mod_product (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 50) : 
  173 * 927 % 50 = n := 
  by
    sorry

end NUMINAMATH_GPT_mod_product_l556_55671


namespace NUMINAMATH_GPT_sufficient_condition_implies_true_l556_55629

variable {p q : Prop}

theorem sufficient_condition_implies_true (h : p → q) : (p → q) = true :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_implies_true_l556_55629


namespace NUMINAMATH_GPT_Wendy_earned_45_points_l556_55662

-- Definitions for the conditions
def points_per_bag : Nat := 5
def total_bags : Nat := 11
def unrecycled_bags : Nat := 2

-- The variable for recycled bags and total points earned
def recycled_bags := total_bags - unrecycled_bags
def total_points := recycled_bags * points_per_bag

theorem Wendy_earned_45_points : total_points = 45 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Wendy_earned_45_points_l556_55662


namespace NUMINAMATH_GPT_proof_problem1_proof_problem2_proof_problem3_proof_problem4_l556_55606

noncomputable def problem1 : Prop := 
  2500 * (1/10000) = 0.25

noncomputable def problem2 : Prop := 
  20 * (1/100) = 0.2

noncomputable def problem3 : Prop := 
  45 * (1/60) = 3/4

noncomputable def problem4 : Prop := 
  1250 * (1/10000) = 0.125

theorem proof_problem1 : problem1 := by
  sorry

theorem proof_problem2 : problem2 := by
  sorry

theorem proof_problem3 : problem3 := by
  sorry

theorem proof_problem4 : problem4 := by
  sorry

end NUMINAMATH_GPT_proof_problem1_proof_problem2_proof_problem3_proof_problem4_l556_55606


namespace NUMINAMATH_GPT_work_problem_l556_55666

theorem work_problem (W : ℝ) (A B C : ℝ)
  (h1 : B + C = W / 24)
  (h2 : C + A = W / 12)
  (h3 : C = W / 32) : A + B = W / 16 := 
by
  sorry

end NUMINAMATH_GPT_work_problem_l556_55666


namespace NUMINAMATH_GPT_difference_in_zits_l556_55653

variable (avgZitsSwanson : ℕ := 5)
variable (avgZitsJones : ℕ := 6)
variable (numKidsSwanson : ℕ := 25)
variable (numKidsJones : ℕ := 32)
variable (totalZitsSwanson : ℕ := avgZitsSwanson * numKidsSwanson)
variable (totalZitsJones : ℕ := avgZitsJones * numKidsJones)

theorem difference_in_zits :
  totalZitsJones - totalZitsSwanson = 67 := by
  sorry

end NUMINAMATH_GPT_difference_in_zits_l556_55653


namespace NUMINAMATH_GPT_last_digit_of_7_power_7_power_7_l556_55638

theorem last_digit_of_7_power_7_power_7 : (7 ^ (7 ^ 7)) % 10 = 3 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_of_7_power_7_power_7_l556_55638


namespace NUMINAMATH_GPT_overlapping_area_of_rectangular_strips_l556_55654

theorem overlapping_area_of_rectangular_strips (theta : ℝ) (h_theta : theta ≠ 0) :
  let width := 2
  let diag_1 := width
  let diag_2 := width / Real.sin theta
  let area := (diag_1 * diag_2) / 2
  area = 2 / Real.sin theta :=
by
  let width := 2
  let diag_1 := width
  let diag_2 := width / Real.sin theta
  let area := (diag_1 * diag_2) / 2
  sorry

end NUMINAMATH_GPT_overlapping_area_of_rectangular_strips_l556_55654


namespace NUMINAMATH_GPT_largest_inscribed_equilateral_triangle_area_l556_55640

theorem largest_inscribed_equilateral_triangle_area 
  (r : ℝ) (h_r : r = 10) : 
  ∃ A : ℝ, 
    A = 100 * Real.sqrt 3 ∧ 
    (∃ s : ℝ, s = 2 * r ∧ A = (Real.sqrt 3 / 4) * s^2) := 
  sorry

end NUMINAMATH_GPT_largest_inscribed_equilateral_triangle_area_l556_55640


namespace NUMINAMATH_GPT_max_possible_number_under_operations_l556_55661

theorem max_possible_number_under_operations :
  ∀ x : ℕ, x < 17 →
    ∀ n : ℕ, (∃ k : ℕ, k < n ∧ (x + 17 * k) % 19 = 0) →
    ∃ m : ℕ, m = (304 : ℕ) :=
sorry

end NUMINAMATH_GPT_max_possible_number_under_operations_l556_55661


namespace NUMINAMATH_GPT_no_a_b_exist_no_a_b_c_exist_l556_55675

-- Part (a):
theorem no_a_b_exist (a b : ℕ) (h0 : 0 < a) (h1 : 0 < b) :
  ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, a * 2^n + b * 5^n = k^2) :=
sorry

-- Part (b):
theorem no_a_b_c_exist (a b c : ℕ) (h0 : 0 < a) (h1 : 0 < b) (h2 : 0 < c) :
  ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, a * 2^n + b * 5^n + c = k^2) :=
sorry

end NUMINAMATH_GPT_no_a_b_exist_no_a_b_c_exist_l556_55675


namespace NUMINAMATH_GPT_triangle_angle_sum_acute_l556_55695

theorem triangle_angle_sum_acute (x : ℝ) (h1 : 60 + 70 + x = 180) (h2 : x ≠ 60 ∧ x ≠ 70) :
  x = 50 ∧ (60 < 90 ∧ 70 < 90 ∧ x < 90) := by
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_acute_l556_55695


namespace NUMINAMATH_GPT_min_Sn_l556_55679

variable {a : ℕ → ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (a₄ : ℤ) (d : ℤ) : Prop :=
  a 4 = a₄ ∧ ∀ n : ℕ, n > 0 → a n = a 1 + (n - 1) * d

def Sn (a : ℕ → ℤ) (n : ℕ) :=
  n / 2 * (2 * a 1 + (n - 1) * 3)

theorem min_Sn (a : ℕ → ℤ) (h1 : arithmetic_sequence a (-15) 3) :
  ∃ n : ℕ, (Sn a n = -108) :=
sorry

end NUMINAMATH_GPT_min_Sn_l556_55679


namespace NUMINAMATH_GPT_solve_system1_solve_system2_l556_55677

-- Definition for System (1)
theorem solve_system1 (x y : ℤ) (h1 : x - 2 * y = 0) (h2 : 3 * x - y = 5) : x = 2 ∧ y = 1 := 
by
  sorry

-- Definition for System (2)
theorem solve_system2 (x y : ℤ) 
  (h1 : 3 * (x - 1) - 4 * (y + 1) = -1) 
  (h2 : (x / 2) + (y / 3) = -2) : x = -2 ∧ y = -3 := 
by
  sorry

end NUMINAMATH_GPT_solve_system1_solve_system2_l556_55677


namespace NUMINAMATH_GPT_obtuse_is_second_quadrant_l556_55643

-- Define the boundaries for an obtuse angle.
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Define the second quadrant condition.
def is_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- The proof problem: Prove that an obtuse angle is a second quadrant angle.
theorem obtuse_is_second_quadrant (θ : ℝ) : is_obtuse θ → is_second_quadrant θ :=
by
  intro h
  sorry

end NUMINAMATH_GPT_obtuse_is_second_quadrant_l556_55643


namespace NUMINAMATH_GPT_number_of_valid_three_digit_numbers_l556_55625

theorem number_of_valid_three_digit_numbers : 
  (∃ A B C : ℕ, 
      (100 * A + 10 * B + C + 297 = 100 * C + 10 * B + A) ∧ 
      (0 ≤ A ∧ A ≤ 9) ∧ 
      (0 ≤ B ∧ B ≤ 9) ∧ 
      (0 ≤ C ∧ C ≤ 9)) 
    ∧ (number_of_such_valid_numbers = 70) :=
by
  sorry

def number_of_such_valid_numbers : ℕ := 
  sorry

end NUMINAMATH_GPT_number_of_valid_three_digit_numbers_l556_55625


namespace NUMINAMATH_GPT_solve_equation_l556_55696

theorem solve_equation :
  ∀ (x : ℝ), x * (3 * x + 6) = 7 * (3 * x + 6) → (x = 7 ∨ x = -2) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_equation_l556_55696


namespace NUMINAMATH_GPT_sequence_term_value_l556_55673

theorem sequence_term_value :
  ∃ (a : ℕ → ℚ), a 1 = 2 ∧ (∀ n, a (n + 1) = a n + 1 / 2) ∧ a 101 = 52 :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_value_l556_55673


namespace NUMINAMATH_GPT_max_value_set_x_graph_transformation_l556_55658

noncomputable def function_y (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6)) + 2

theorem max_value_set_x :
  ∃ k : ℤ, ∀ x : ℝ, x = k * Real.pi + Real.pi / 6 → function_y x = 4 :=
by
  sorry

theorem graph_transformation :
  ∀ x : ℝ, ∃ y : ℝ, (y = Real.sin x → y = 2 * Real.sin (2 * x + (Real.pi / 6)) + 2) :=
by
  sorry

end NUMINAMATH_GPT_max_value_set_x_graph_transformation_l556_55658


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l556_55628

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l556_55628


namespace NUMINAMATH_GPT_adrianna_gum_pieces_l556_55627

-- Definitions based on conditions
def initial_gum_pieces : ℕ := 10
def additional_gum_pieces : ℕ := 3
def friends_count : ℕ := 11

-- Expression to calculate the final pieces of gum
def total_gum_pieces : ℕ := initial_gum_pieces + additional_gum_pieces
def gum_left : ℕ := total_gum_pieces - friends_count

-- Lean statement we want to prove
theorem adrianna_gum_pieces: gum_left = 2 := 
by 
  sorry

end NUMINAMATH_GPT_adrianna_gum_pieces_l556_55627


namespace NUMINAMATH_GPT_circulation_ratio_l556_55644

variable (A : ℕ) -- Assuming A to be a natural number for simplicity

theorem circulation_ratio (h : ∀ t : ℕ, t = 1971 → t = 4 * A) : 4 / 13 = 4 / 13 := 
by
  sorry

end NUMINAMATH_GPT_circulation_ratio_l556_55644


namespace NUMINAMATH_GPT_evaluate_expression_l556_55634

theorem evaluate_expression 
  (a c : ℝ)
  (h : a + c = 9) :
  (a * (-1)^2 + (-1) + c) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l556_55634


namespace NUMINAMATH_GPT_parabola_focus_l556_55663

theorem parabola_focus (x y : ℝ) (h : y = 4 * x^2) : (0, 1 / 16) = (0, 1 / 16) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l556_55663


namespace NUMINAMATH_GPT_angle_bisector_segment_rel_l556_55690

variable (a b c : ℝ) -- The sides of the triangle
variable (u v : ℝ)   -- The segments into which fa divides side a
variable (fa : ℝ)    -- The length of the angle bisector

-- Statement setting up the given conditions and the proof we need
theorem angle_bisector_segment_rel : 
  (u : ℝ) = a * c / (b + c) → 
  (v : ℝ) = a * b / (b + c) → 
  (fa : ℝ) = 2 * (Real.sqrt (b * s * (s - a) * c)) / (b + c) → 
  fa^2 = b * c - u * v :=
sorry

end NUMINAMATH_GPT_angle_bisector_segment_rel_l556_55690


namespace NUMINAMATH_GPT_remainder_of_1998_to_10_mod_10k_l556_55655

theorem remainder_of_1998_to_10_mod_10k : 
  let x := 1998
  let y := 10^4
  x^10 % y = 1024 := 
by
  let x := 1998
  let y := 10^4
  sorry

end NUMINAMATH_GPT_remainder_of_1998_to_10_mod_10k_l556_55655


namespace NUMINAMATH_GPT_find_m_value_l556_55647

-- Define the points P and Q and the condition of perpendicularity
def points_PQ (m : ℝ) : Prop := 
  let P := (-2, m)
  let Q := (m, 4)
  let slope_PQ := (m - 4) / (-2 - m)
  slope_PQ * (-1) = -1

-- Problem statement: Find the value of m such that the above condition holds
theorem find_m_value : ∃ (m : ℝ), points_PQ m ∧ m = 1 :=
by sorry

end NUMINAMATH_GPT_find_m_value_l556_55647


namespace NUMINAMATH_GPT_base6_addition_problem_l556_55610

theorem base6_addition_problem (X Y : ℕ) (h1 : 3 * 6^2 + X * 6 + Y + 24 = 6 * 6^2 + 1 * 6 + X) :
  X = 5 ∧ Y = 1 ∧ X + Y = 6 := by
  sorry

end NUMINAMATH_GPT_base6_addition_problem_l556_55610


namespace NUMINAMATH_GPT_mean_of_observations_decreased_l556_55631

noncomputable def original_mean : ℕ := 200

theorem mean_of_observations_decreased (S' : ℕ) (M' : ℕ) (n : ℕ) (d : ℕ)
  (h1 : n = 50)
  (h2 : d = 15)
  (h3 : M' = 185)
  (h4 : S' = M' * n)
  : original_mean = (S' + d * n) / n :=
by
  rw [original_mean]
  sorry

end NUMINAMATH_GPT_mean_of_observations_decreased_l556_55631


namespace NUMINAMATH_GPT_inequality_not_hold_l556_55632

theorem inequality_not_hold (x y : ℝ) (h : x > y) : ¬ (1 - x > 1 - y) :=
by
  -- condition and given statements
  sorry

end NUMINAMATH_GPT_inequality_not_hold_l556_55632


namespace NUMINAMATH_GPT_find_value_l556_55698

theorem find_value
  (y1 y2 y3 y4 y5 : ℝ)
  (h1 : y1 + 4 * y2 + 9 * y3 + 16 * y4 + 25 * y5 = 3)
  (h2 : 4 * y1 + 9 * y2 + 16 * y3 + 25 * y4 + 36 * y5 = 20)
  (h3 : 9 * y1 + 16 * y2 + 25 * y3 + 36 * y4 + 49 * y5 = 150) :
  16 * y1 + 25 * y2 + 36 * y3 + 49 * y4 + 64 * y5 = 336 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l556_55698


namespace NUMINAMATH_GPT_height_of_the_carton_l556_55694

noncomputable def carton_height : ℕ :=
  let carton_length := 25
  let carton_width := 42
  let soap_box_length := 7
  let soap_box_width := 6
  let soap_box_height := 10
  let max_soap_boxes := 150
  let boxes_per_row := carton_length / soap_box_length
  let boxes_per_column := carton_width / soap_box_width
  let boxes_per_layer := boxes_per_row * boxes_per_column
  let layers := max_soap_boxes / boxes_per_layer
  layers * soap_box_height

theorem height_of_the_carton :
  carton_height = 70 :=
by
  -- The computation and necessary assumptions for proving the height are encapsulated above.
  sorry

end NUMINAMATH_GPT_height_of_the_carton_l556_55694


namespace NUMINAMATH_GPT_three_digit_number_l556_55646

theorem three_digit_number (m : ℕ) : (300 * m + 10 * m + (m - 1)) = (311 * m - 1) :=
by 
  sorry

end NUMINAMATH_GPT_three_digit_number_l556_55646


namespace NUMINAMATH_GPT_problem_statement_l556_55608

noncomputable def f (x : ℝ) : ℝ :=
  1 - x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2

theorem problem_statement : f (1 / 2) + f (-1 / 2) = 2 := sorry

end NUMINAMATH_GPT_problem_statement_l556_55608


namespace NUMINAMATH_GPT_tan_neq_sqrt3_sufficient_but_not_necessary_l556_55642

-- Definition of the condition: tan(α) ≠ √3
def condition_tan_neq_sqrt3 (α : ℝ) : Prop := Real.tan α ≠ Real.sqrt 3

-- Definition of the statement: α ≠ π/3
def statement_alpha_neq_pi_div_3 (α : ℝ) : Prop := α ≠ Real.pi / 3

-- The theorem to be proven
theorem tan_neq_sqrt3_sufficient_but_not_necessary {α : ℝ} :
  condition_tan_neq_sqrt3 α → statement_alpha_neq_pi_div_3 α :=
sorry

end NUMINAMATH_GPT_tan_neq_sqrt3_sufficient_but_not_necessary_l556_55642


namespace NUMINAMATH_GPT_large_bottle_water_amount_l556_55601

noncomputable def sport_drink_water_amount (C V : ℝ) (prop_e : ℝ) : ℝ :=
  let F := C / 4
  let W := (C * 15)
  W

theorem large_bottle_water_amount (C V : ℝ) (prop_e : ℝ) (hc : C = 7) (hprop_e : prop_e = 0.05) : sport_drink_water_amount C V prop_e = 105 := by
  sorry

end NUMINAMATH_GPT_large_bottle_water_amount_l556_55601


namespace NUMINAMATH_GPT_largest_even_number_l556_55685

theorem largest_even_number (n : ℤ) 
    (h1 : (n-6) % 2 = 0) 
    (h2 : (n+6) = 3 * (n-6)) :
    (n + 6) = 18 :=
by
  sorry

end NUMINAMATH_GPT_largest_even_number_l556_55685


namespace NUMINAMATH_GPT_integer_root_of_polynomial_l556_55689

/-- Prove that -6 is a root of the polynomial equation x^3 + bx + c = 0,
    where b and c are rational numbers and 3 - sqrt(5) is a root
 -/
theorem integer_root_of_polynomial (b c : ℚ)
  (h : ∀ x : ℝ, (x^3 + (b : ℝ)*x + (c : ℝ) = 0) → x = (3 - Real.sqrt 5) ∨ x = (3 + Real.sqrt 5) ∨ x = -6) :
  ∃ x : ℤ, x^3 + b*x + c = 0 ∧ x = -6 :=
by
  sorry

end NUMINAMATH_GPT_integer_root_of_polynomial_l556_55689


namespace NUMINAMATH_GPT_consistent_system_l556_55613

variable (x y : ℕ)

def condition1 := x + y = 40
def condition2 := 2 * 15 * x = 20 * y

theorem consistent_system :
  condition1 x y ∧ condition2 x y ↔ 
  (x + y = 40 ∧ 2 * 15 * x = 20 * y) :=
by
  sorry

end NUMINAMATH_GPT_consistent_system_l556_55613


namespace NUMINAMATH_GPT_largest_unrepresentable_l556_55619

theorem largest_unrepresentable (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : Nat.gcd b c = 1) (h3 : Nat.gcd c a = 1)
  : ¬ ∃ (x y z : ℕ), x * b * c + y * c * a + z * a * b = 2 * a * b * c - a * b - b * c - c * a :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_largest_unrepresentable_l556_55619


namespace NUMINAMATH_GPT_find_b_for_integer_a_l556_55665

theorem find_b_for_integer_a (a : ℤ) (b : ℝ) (h1 : 0 ≤ b) (h2 : b < 1) (h3 : (a:ℝ)^2 = 2 * b * (a + b)) :
  b = 0 ∨ b = (-1 + Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_find_b_for_integer_a_l556_55665


namespace NUMINAMATH_GPT_sum_of_squares_l556_55691

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 21) (h2 : x * y = 43) : x^2 + y^2 = 355 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l556_55691


namespace NUMINAMATH_GPT_first_player_wins_l556_55651

theorem first_player_wins :
  ∀ (sticks : ℕ), (sticks = 1) →
  (∀ (break_rule : ℕ → ℕ → Prop),
  (∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z → break_rule x y → break_rule x z)
  → (∃ n : ℕ, n % 3 = 0 ∧ break_rule n (n + 1) → ∃ t₁ t₂ t₃ : ℕ, t₁ = t₂ ∧ t₂ = t₃ ∧ t₁ + t₂ + t₃ = n))
  → (∃ w : ℕ, w = 1) := sorry

end NUMINAMATH_GPT_first_player_wins_l556_55651


namespace NUMINAMATH_GPT_exponent_sum_l556_55678

theorem exponent_sum : (-2:ℝ) ^ 4 + (-2:ℝ) ^ (3 / 2) + (-2:ℝ) ^ 1 + 2 ^ 1 + 2 ^ (3 / 2) + 2 ^ 4 = 32 := by
  sorry

end NUMINAMATH_GPT_exponent_sum_l556_55678


namespace NUMINAMATH_GPT_beads_per_bracelet_is_10_l556_55688

-- Definitions of given conditions
def num_necklaces_Monday : ℕ := 10
def num_necklaces_Tuesday : ℕ := 2
def num_necklaces : ℕ := num_necklaces_Monday + num_necklaces_Tuesday

def beads_per_necklace : ℕ := 20
def beads_necklaces : ℕ := num_necklaces * beads_per_necklace

def num_earrings : ℕ := 7
def beads_per_earring : ℕ := 5
def beads_earrings : ℕ := num_earrings * beads_per_earring

def total_beads_used : ℕ := 325
def beads_used_for_necklaces_and_earrings : ℕ := beads_necklaces + beads_earrings
def beads_remaining_for_bracelets : ℕ := total_beads_used - beads_used_for_necklaces_and_earrings

def num_bracelets : ℕ := 5
def beads_per_bracelet : ℕ := beads_remaining_for_bracelets / num_bracelets

-- Theorem statement to prove
theorem beads_per_bracelet_is_10 : beads_per_bracelet = 10 := by
  sorry

end NUMINAMATH_GPT_beads_per_bracelet_is_10_l556_55688


namespace NUMINAMATH_GPT_new_average_daily_production_l556_55620

theorem new_average_daily_production (n : ℕ) (avg_past_n_days : ℕ) (today_production : ℕ) (h1 : avg_past_n_days = 50) (h2 : today_production = 90) (h3 : n = 9) : 
  (avg_past_n_days * n + today_production) / (n + 1) = 54 := 
by
  sorry

end NUMINAMATH_GPT_new_average_daily_production_l556_55620


namespace NUMINAMATH_GPT_skateboarder_speed_l556_55649

-- Defining the conditions
def distance_feet : ℝ := 476.67
def time_seconds : ℝ := 25
def feet_per_mile : ℝ := 5280
def seconds_per_hour : ℝ := 3600

-- Defining the expected speed in miles per hour
def expected_speed_mph : ℝ := 13.01

-- The problem statement: Prove that the skateboarder's speed is 13.01 mph given the conditions
theorem skateboarder_speed : (distance_feet / feet_per_mile) / (time_seconds / seconds_per_hour) = expected_speed_mph := by
  sorry

end NUMINAMATH_GPT_skateboarder_speed_l556_55649


namespace NUMINAMATH_GPT_quadratic_polynomials_perfect_square_l556_55609

variables {x y p q a b c : ℝ}

theorem quadratic_polynomials_perfect_square (h1 : ∃ a, x^2 + p * x + q = (x + a) * (x + a))
  (h2 : ∃ a b, a^2 * x^2 + 2 * b^2 * x * y + c^2 * y^2 = (a * x + b * y) * (a * x + b * y)) :
  q = (p^2 / 4) ∧ b^2 = a * c :=
by
  sorry

end NUMINAMATH_GPT_quadratic_polynomials_perfect_square_l556_55609


namespace NUMINAMATH_GPT_min_value_a_l556_55664

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - 6 > 0 → x > a) ∧
  ¬ (∀ x : ℝ, x > a → x^2 - x - 6 > 0) ↔ a = 3 :=
sorry

end NUMINAMATH_GPT_min_value_a_l556_55664


namespace NUMINAMATH_GPT_question_1_question_2_l556_55660

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem question_1 :
  f 1 * f 2 * f 3 = 36 * 108 * 360 := by
  sorry

theorem question_2 :
  ∃ m ≥ 2, ∀ n : ℕ, n > 0 → f n % m = 0 ∧ m = 36 := by
  sorry

end NUMINAMATH_GPT_question_1_question_2_l556_55660


namespace NUMINAMATH_GPT_find_angle_B_l556_55641

theorem find_angle_B (A B C : ℝ) (a b c : ℝ) 
  (h1 : A = 45) 
  (h2 : a = 6) 
  (h3 : b = 3 * Real.sqrt 2)
  (h4 : ∀ A' B' C' : ℝ, 
        ∃ a' b' c' : ℝ, 
        (a' = a) ∧ (b' = b) ∧ (A' = A) ∧ 
        (b' < a') → (B' < A') ∧ (A' = 45)) :
  B = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_B_l556_55641


namespace NUMINAMATH_GPT_mask_production_rates_l556_55626

theorem mask_production_rates (x : ℝ) (y : ℝ) :
  (280 / x) - (280 / (1.4 * x)) = 2 →
  x = 40 ∧ y = 1.4 * x →
  y = 56 :=
by {
  sorry
}

end NUMINAMATH_GPT_mask_production_rates_l556_55626


namespace NUMINAMATH_GPT_coloring_ways_l556_55621

def num_colorings (total_circles blue_circles green_circles red_circles : ℕ) : ℕ :=
  if total_circles = blue_circles + green_circles + red_circles then
    (Nat.choose total_circles (green_circles + red_circles)) * (Nat.factorial (green_circles + red_circles) / (Nat.factorial green_circles * Nat.factorial red_circles))
  else
    0

theorem coloring_ways :
  num_colorings 6 4 1 1 = 30 :=
by sorry

end NUMINAMATH_GPT_coloring_ways_l556_55621


namespace NUMINAMATH_GPT_solution_set_of_log_inequality_l556_55604

noncomputable def log_a (a x : ℝ) : ℝ := sorry -- The precise definition of the log base 'a' is skipped for brevity.

theorem solution_set_of_log_inequality (a x : ℝ)
  (ha_pos : a > 0)
  (ha_ne_one : a ≠ 1)
  (h_max : ∃ y, log_a a (y^2 - 2*y + 3) = y):
  log_a a (x - 1) > 0 ↔ (1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_GPT_solution_set_of_log_inequality_l556_55604


namespace NUMINAMATH_GPT_alex_class_size_l556_55623

theorem alex_class_size 
  (n : ℕ) 
  (h_top : 30 ≤ n)
  (h_bottom : 30 ≤ n) 
  (h_better : n - 30 > 0)
  (h_worse : n - 30 > 0)
  : n = 59 := 
sorry

end NUMINAMATH_GPT_alex_class_size_l556_55623


namespace NUMINAMATH_GPT_probability_of_purple_is_one_fifth_l556_55650

-- Definitions related to the problem
def total_faces : ℕ := 10
def purple_faces : ℕ := 2
def probability_purple := (purple_faces : ℚ) / (total_faces : ℚ)

theorem probability_of_purple_is_one_fifth : probability_purple = 1 / 5 := 
by
  -- Converting the numbers to rationals explicitly ensures division is defined.
  change (2 : ℚ) / (10 : ℚ) = 1 / 5
  norm_num
  -- sorry (if finishing the proof manually isn't desired)

end NUMINAMATH_GPT_probability_of_purple_is_one_fifth_l556_55650


namespace NUMINAMATH_GPT_simplify_expression_l556_55648

variable (x : ℝ)

theorem simplify_expression :
  (3 * x - 2) * (5 * x ^ 12 - 3 * x ^ 11 + 2 * x ^ 9 - x ^ 6) =
  15 * x ^ 13 - 19 * x ^ 12 - 6 * x ^ 11 + 6 * x ^ 10 - 4 * x ^ 9 - 3 * x ^ 7 + 2 * x ^ 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l556_55648


namespace NUMINAMATH_GPT_increase_in_cases_second_day_l556_55618

-- Define the initial number of cases.
def initial_cases : ℕ := 2000

-- Define the number of recoveries on the second day.
def recoveries_day2 : ℕ := 50

-- Define the number of new cases on the third day and the recoveries on the third day.
def new_cases_day3 : ℕ := 1500
def recoveries_day3 : ℕ := 200

-- Define the total number of positive cases after the third day.
def total_cases_day3 : ℕ := 3750

-- Lean statement to prove the increase in cases on the second day is 750.
theorem increase_in_cases_second_day : 
  ∃ x : ℕ, initial_cases + x - recoveries_day2 + new_cases_day3 - recoveries_day3 = total_cases_day3 ∧ x = 750 :=
by
  sorry

end NUMINAMATH_GPT_increase_in_cases_second_day_l556_55618


namespace NUMINAMATH_GPT_systematic_sampling_first_group_l556_55657

theorem systematic_sampling_first_group (x : ℕ) (n : ℕ) (k : ℕ) (total_students : ℕ) (sampled_students : ℕ) 
  (interval : ℕ) (group_num : ℕ) (group_val : ℕ) 
  (h1 : total_students = 1000) (h2 : sampled_students = 40) (h3 : interval = total_students / sampled_students)
  (h4 : interval = 25) (h5 : group_num = 18) 
  (h6 : group_val = 443) (h7 : group_val = x + (group_num - 1) * interval) : 
  x = 18 := 
by 
  sorry

end NUMINAMATH_GPT_systematic_sampling_first_group_l556_55657


namespace NUMINAMATH_GPT_probability_different_suits_correct_l556_55684

-- Definitions based on conditions
def cards_in_deck : ℕ := 52
def cards_picked : ℕ := 3
def first_card_suit_not_matter : Prop := True
def second_card_different_suit : Prop := True
def third_card_different_suit : Prop := True

-- Definition of the probability function
def probability_different_suits (cards_total : ℕ) (cards_picked : ℕ) : Rat :=
  let first_card_prob := 1
  let second_card_prob := 39 / 51
  let third_card_prob := 26 / 50
  first_card_prob * second_card_prob * third_card_prob

-- The theorem statement to prove the probability each card is of a different suit
theorem probability_different_suits_correct :
  probability_different_suits cards_in_deck cards_picked = 169 / 425 :=
by
  -- Proof should be written here
  sorry

end NUMINAMATH_GPT_probability_different_suits_correct_l556_55684


namespace NUMINAMATH_GPT_part_a_total_time_part_b_average_time_part_c_probability_l556_55699

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end NUMINAMATH_GPT_part_a_total_time_part_b_average_time_part_c_probability_l556_55699


namespace NUMINAMATH_GPT_notebook_cost_l556_55605

theorem notebook_cost
  (n c : ℝ)
  (h1 : n + c = 2.20)
  (h2 : n = c + 2) :
  n = 2.10 :=
by
  sorry

end NUMINAMATH_GPT_notebook_cost_l556_55605


namespace NUMINAMATH_GPT_stayed_days_calculation_l556_55600

theorem stayed_days_calculation (total_cost : ℕ) (charge_1st_week : ℕ) (charge_additional_week : ℕ) (first_week_days : ℕ) :
  total_cost = 302 ∧ charge_1st_week = 18 ∧ charge_additional_week = 11 ∧ first_week_days = 7 →
  ∃ D : ℕ, D = 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_stayed_days_calculation_l556_55600


namespace NUMINAMATH_GPT_right_triangle_AB_is_approximately_8point3_l556_55611

noncomputable def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

theorem right_triangle_AB_is_approximately_8point3 :
  ∀ (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (BC AB : ℝ),
  angle_A = 40 ∧ angle_B = 90 ∧ BC = 7 →
  AB = 7 / tan_deg 40 →
  abs (AB - 8.3) < 0.1 :=
by
  intros A B C angle_A angle_B BC AB h_cond h_AB
  sorry

end NUMINAMATH_GPT_right_triangle_AB_is_approximately_8point3_l556_55611


namespace NUMINAMATH_GPT_math_proof_problem_l556_55681

open Nat

noncomputable def number_of_pairs := 
  let N := 20^19
  let num_divisors := (38 + 1) * (19 + 1)
  let total_pairs := num_divisors * num_divisors
  let ab_dividing_pairs := 780 * 210
  total_pairs - ab_dividing_pairs

theorem math_proof_problem : number_of_pairs = 444600 := 
  by exact sorry

end NUMINAMATH_GPT_math_proof_problem_l556_55681


namespace NUMINAMATH_GPT_sum_odd_divisors_90_eq_78_l556_55674

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end NUMINAMATH_GPT_sum_odd_divisors_90_eq_78_l556_55674


namespace NUMINAMATH_GPT_regular_14_gon_inequality_l556_55616

noncomputable def side_length_of_regular_14_gon : ℝ := 2 * Real.sin (Real.pi / 14)

theorem regular_14_gon_inequality (a : ℝ) (h : a = side_length_of_regular_14_gon) :
  (2 - a) / (2 * a) > Real.sqrt (3 * Real.cos (Real.pi / 7)) :=
by
  sorry

end NUMINAMATH_GPT_regular_14_gon_inequality_l556_55616


namespace NUMINAMATH_GPT_part1_l556_55612

theorem part1 (a x0 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a ^ x0 = 2) : a ^ (3 * x0) = 8 := by
  sorry

end NUMINAMATH_GPT_part1_l556_55612


namespace NUMINAMATH_GPT_number_of_repeating_decimals_l556_55672

open Nat

theorem number_of_repeating_decimals :
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 15) → (¬ ∃ k : ℕ, k * 18 = n) :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_number_of_repeating_decimals_l556_55672


namespace NUMINAMATH_GPT_find_x_for_prime_power_l556_55676

theorem find_x_for_prime_power (x : ℤ) :
  (∃ p k : ℕ, Nat.Prime p ∧ k > 0 ∧ (2 * x * x + x - 6 = p ^ k)) → (x = -3 ∨ x = 2 ∨ x = 5) := by
  sorry

end NUMINAMATH_GPT_find_x_for_prime_power_l556_55676


namespace NUMINAMATH_GPT_floor_sqrt_17_squared_eq_16_l556_55615

theorem floor_sqrt_17_squared_eq_16 :
  (⌊Real.sqrt 17⌋ : Real)^2 = 16 := by
  sorry

end NUMINAMATH_GPT_floor_sqrt_17_squared_eq_16_l556_55615


namespace NUMINAMATH_GPT_sum_of_legs_equal_l556_55683

theorem sum_of_legs_equal
  (a b c d e f g h : ℝ)
  (x y : ℝ)
  (h_similar_shaded1 : a = a * x ∧ b = a * y)
  (h_similar_shaded2 : c = c * x ∧ d = c * y)
  (h_similar_shaded3 : e = e * x ∧ f = e * y)
  (h_similar_shaded4 : g = g * x ∧ h = g * y)
  (h_similar_unshaded1 : h = h * x ∧ a = h * y)
  (h_similar_unshaded2 : b = b * x ∧ c = b * y)
  (h_similar_unshaded3 : d = d * x ∧ e = d * y)
  (h_similar_unshaded4 : f = f * x ∧ g = f * y)
  (x_non_zero : x ≠ 0) (y_non_zero : y ≠ 0) : 
  (a * y + b + c * x) + (c * y + d + e * x) + (e * y + f + g * x) + (g * y + h + a * x) 
  = (h * x + a + b * y) + (b * x + c + d * y) + (d * x + e + f * y) + (f * x + g + h * y) :=
sorry

end NUMINAMATH_GPT_sum_of_legs_equal_l556_55683


namespace NUMINAMATH_GPT_probability_not_losing_l556_55693

theorem probability_not_losing (P_winning P_drawing : ℚ)
  (h_winning : P_winning = 1/3)
  (h_drawing : P_drawing = 1/4) :
  P_winning + P_drawing = 7/12 := 
by
  sorry

end NUMINAMATH_GPT_probability_not_losing_l556_55693


namespace NUMINAMATH_GPT_original_board_is_120_l556_55645

-- Define the two given conditions
def S : ℕ := 35
def L : ℕ := 2 * S + 15

-- Define the length of the original board
def original_board_length : ℕ := S + L

-- The theorem we want to prove
theorem original_board_is_120 : original_board_length = 120 :=
by
  -- Skipping the actual proof
  sorry

end NUMINAMATH_GPT_original_board_is_120_l556_55645


namespace NUMINAMATH_GPT_total_viewing_time_l556_55607

theorem total_viewing_time (video_length : ℕ) (num_videos : ℕ) (lila_speed_factor : ℕ) :
  video_length = 100 ∧ num_videos = 6 ∧ lila_speed_factor = 2 →
  (num_videos * (video_length / lila_speed_factor) + num_videos * video_length) = 900 :=
by
  sorry

end NUMINAMATH_GPT_total_viewing_time_l556_55607


namespace NUMINAMATH_GPT_correctness_of_propositions_l556_55667

-- Definitions of the conditions
def residual_is_random_error (e : ℝ) : Prop := ∃ (y : ℝ) (y_hat : ℝ), e = y - y_hat
def data_constraints (a b c d : ℕ) : Prop := a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ d ≥ 5
def histogram_judgement : Prop := ∀ (H : Type) (rel : H → H → Prop), ¬(H ≠ H) ∨ (∀ x y : H, rel x y ↔ true)

-- The mathematical equivalence proof problem
theorem correctness_of_propositions (e : ℝ) (a b c d : ℕ) : 
  (residual_is_random_error e → false) ∧
  (data_constraints a b c d → true) ∧
  (histogram_judgement → true) :=
by
  sorry

end NUMINAMATH_GPT_correctness_of_propositions_l556_55667


namespace NUMINAMATH_GPT_neither_chemistry_nor_biology_l556_55680

variable (club_size chemistry_students biology_students both_students neither_students : ℕ)

def students_in_club : Prop :=
  club_size = 75

def students_taking_chemistry : Prop :=
  chemistry_students = 40

def students_taking_biology : Prop :=
  biology_students = 35

def students_taking_both : Prop :=
  both_students = 25

theorem neither_chemistry_nor_biology :
  students_in_club club_size ∧ 
  students_taking_chemistry chemistry_students ∧
  students_taking_biology biology_students ∧
  students_taking_both both_students →
  neither_students = 75 - ((chemistry_students - both_students) + (biology_students - both_students) + both_students) :=
by
  intros
  sorry

end NUMINAMATH_GPT_neither_chemistry_nor_biology_l556_55680


namespace NUMINAMATH_GPT_cubed_identity_l556_55692

variable (x : ℝ)

theorem cubed_identity (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
by
  sorry

end NUMINAMATH_GPT_cubed_identity_l556_55692


namespace NUMINAMATH_GPT_prove_statements_l556_55630

theorem prove_statements (x y z : ℝ) (h : x + y + z = x * y * z) :
  ( (∀ (x y : ℝ), x + y = 0 → (∃ (z : ℝ), (x + y + z = x * y * z) → z = 0))
  ∧ (∀ (x y : ℝ), x = 0 → (∃ (z : ℝ), (x + y + z = x * y * z) → y = -z))
  ∧ z = (x + y) / (x * y - 1) ) :=
by
  sorry

end NUMINAMATH_GPT_prove_statements_l556_55630


namespace NUMINAMATH_GPT_minimum_value_inequality_l556_55617

theorem minimum_value_inequality (x y z : ℝ) (hx : 2 ≤ x) (hxy : x ≤ y) (hyz : y ≤ z) (hz : z ≤ 5) :
    (x - 2)^2 + (y / x - 2)^2 + (z / y - 2)^2 + (5 / z - 2)^2 ≥ 4 * (Real.sqrt (Real.sqrt 5) - 2)^2 := 
    sorry

end NUMINAMATH_GPT_minimum_value_inequality_l556_55617


namespace NUMINAMATH_GPT_cost_per_mile_l556_55652

theorem cost_per_mile (x : ℝ) (daily_fee : ℝ) (daily_budget : ℝ) (max_miles : ℝ)
  (h1 : daily_fee = 50)
  (h2 : daily_budget = 88)
  (h3 : max_miles = 190)
  (h4 : daily_budget = daily_fee + x * max_miles) :
  x = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_mile_l556_55652


namespace NUMINAMATH_GPT_total_players_count_l556_55687

theorem total_players_count (M W : ℕ) (h1 : W = M + 4) (h2 : (M : ℚ) / W = 5 / 9) : M + W = 14 :=
sorry

end NUMINAMATH_GPT_total_players_count_l556_55687


namespace NUMINAMATH_GPT_number_of_distinct_rationals_l556_55639

theorem number_of_distinct_rationals (L : ℕ) :
  L = 26 ↔
  (∃ (k : ℚ), |k| < 100 ∧ (∃ (x : ℤ), 7 * x^2 + k * x + 20 = 0)) :=
sorry

end NUMINAMATH_GPT_number_of_distinct_rationals_l556_55639


namespace NUMINAMATH_GPT_trip_duration_l556_55633

/--
Given:
1. The car averages 30 miles per hour for the first 5 hours of the trip.
2. The car averages 42 miles per hour for the rest of the trip.
3. The average speed for the entire trip is 34 miles per hour.

Prove: 
The total duration of the trip is 7.5 hours.
-/
theorem trip_duration (t T : ℝ) (h1 : 150 + 42 * t = 34 * T) (h2 : T = 5 + t) : T = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_trip_duration_l556_55633


namespace NUMINAMATH_GPT_remainder_mod_41_l556_55682

theorem remainder_mod_41 (M : ℤ) (hM1 : M = 1234567891011123940) : M % 41 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_mod_41_l556_55682


namespace NUMINAMATH_GPT_remainder_sum_15_div_11_l556_55637

theorem remainder_sum_15_div_11 :
  let n := 15 
  let a := 1 
  let l := 15 
  let S := (n * (a + l)) / 2
  S % 11 = 10 :=
by
  let n := 15
  let a := 1
  let l := 15
  let S := (n * (a + l)) / 2
  show S % 11 = 10
  sorry

end NUMINAMATH_GPT_remainder_sum_15_div_11_l556_55637


namespace NUMINAMATH_GPT_wellington_population_l556_55668

theorem wellington_population 
  (W P L : ℕ)
  (h1 : P = 7 * W)
  (h2 : P = L + 800)
  (h3 : P + L = 11800) : 
  W = 900 :=
by
  sorry

end NUMINAMATH_GPT_wellington_population_l556_55668


namespace NUMINAMATH_GPT_percentage_needed_to_pass_l556_55614

-- Define conditions
def student_score : ℕ := 80
def marks_shortfall : ℕ := 40
def total_marks : ℕ := 400

-- Theorem statement: The percentage of marks required to pass the test.
theorem percentage_needed_to_pass : (student_score + marks_shortfall) * 100 / total_marks = 30 := by
  sorry

end NUMINAMATH_GPT_percentage_needed_to_pass_l556_55614


namespace NUMINAMATH_GPT_xiao_ming_should_choose_store_A_l556_55656

def storeB_cost (x : ℕ) : ℝ := 0.85 * x

def storeA_cost (x : ℕ) : ℝ :=
  if x ≤ 10 then x
  else 0.7 * x + 3

theorem xiao_ming_should_choose_store_A (x : ℕ) (h : x = 22) :
  storeA_cost x < storeB_cost x := by
  sorry

end NUMINAMATH_GPT_xiao_ming_should_choose_store_A_l556_55656


namespace NUMINAMATH_GPT_find_amplitude_l556_55622

theorem find_amplitude (A D : ℝ) (h1 : D + A = 5) (h2 : D - A = -3) : A = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_amplitude_l556_55622


namespace NUMINAMATH_GPT_edge_length_of_cube_l556_55697

/--
Given:
1. A cuboid with base width of 70 cm, base length of 40 cm, and height of 150 cm.
2. A cube-shaped cabinet whose volume is 204,000 cm³ smaller than that of the cuboid.

Prove that one edge of the cube-shaped cabinet is 60 cm.
-/
theorem edge_length_of_cube (W L H V_diff : ℝ) (cuboid_vol : ℝ) (cube_vol : ℝ) (edge : ℝ) :
  W = 70 ∧ L = 40 ∧ H = 150 ∧ V_diff = 204000 ∧ 
  cuboid_vol = W * L * H ∧ cube_vol = cuboid_vol - V_diff ∧ edge ^ 3 = cube_vol -> 
  edge = 60 :=
by
  sorry

end NUMINAMATH_GPT_edge_length_of_cube_l556_55697


namespace NUMINAMATH_GPT_min_value_frac_l556_55635

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 2) : 
  (2 / x) + (1 / y) ≥ 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_frac_l556_55635


namespace NUMINAMATH_GPT_amount_distributed_l556_55670

theorem amount_distributed (A : ℝ) (h : A / 20 = A / 25 + 120) : A = 12000 :=
by
  sorry

end NUMINAMATH_GPT_amount_distributed_l556_55670


namespace NUMINAMATH_GPT_total_hours_worked_l556_55659

-- Define the number of hours worked on Saturday
def hours_saturday : ℕ := 6

-- Define the number of hours worked on Sunday
def hours_sunday : ℕ := 4

-- Define the total number of hours worked on both days
def total_hours : ℕ := hours_saturday + hours_sunday

-- The theorem to prove the total number of hours worked on Saturday and Sunday
theorem total_hours_worked : total_hours = 10 := by
  sorry

end NUMINAMATH_GPT_total_hours_worked_l556_55659


namespace NUMINAMATH_GPT_carmen_sprigs_left_l556_55624

-- Definitions based on conditions
def initial_sprigs : ℕ := 25
def whole_sprigs_used : ℕ := 8
def half_sprigs_plates : ℕ := 12
def half_sprigs_total_used : ℕ := half_sprigs_plates / 2

-- Total sprigs used
def total_sprigs_used : ℕ := whole_sprigs_used + half_sprigs_total_used

-- Leftover sprigs computation
def sprigs_left : ℕ := initial_sprigs - total_sprigs_used

-- Statement to prove
theorem carmen_sprigs_left : sprigs_left = 11 :=
by
  sorry

end NUMINAMATH_GPT_carmen_sprigs_left_l556_55624


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l556_55636
noncomputable def equilateral_triangle_side (r R : ℝ) (h : R > r) : ℝ :=
  r * R * Real.sqrt 3 / (Real.sqrt (r ^ 2 - r * R + R ^ 2))

theorem equilateral_triangle_side_length
  (r R : ℝ) (hRgr : R > r) :
  ∃ a, a = equilateral_triangle_side r R hRgr :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_side_length_l556_55636


namespace NUMINAMATH_GPT_detectives_sons_ages_l556_55686

theorem detectives_sons_ages (x y : ℕ) (h1 : x < 5) (h2 : y < 5) (h3 : x * y = 4) (h4 : (∃ x₁ y₁ : ℕ, (x₁ * y₁ = 4 ∧ x₁ < 5 ∧ y₁ < 5) ∧ x₁ ≠ x ∨ y₁ ≠ y)) :
  (x = 1 ∨ x = 4) ∧ (y = 1 ∨ y = 4) :=
by
  sorry

end NUMINAMATH_GPT_detectives_sons_ages_l556_55686


namespace NUMINAMATH_GPT_cost_of_expensive_feed_l556_55669

open Lean Real

theorem cost_of_expensive_feed (total_feed : Real)
                              (total_cost_per_pound : Real) 
                              (cheap_feed_weight : Real)
                              (cheap_cost_per_pound : Real)
                              (expensive_feed_weight : Real)
                              (expensive_cost_per_pound : Real):
  total_feed = 35 ∧ 
  total_cost_per_pound = 0.36 ∧ 
  cheap_feed_weight = 17 ∧ 
  cheap_cost_per_pound = 0.18 ∧ 
  expensive_feed_weight = total_feed - cheap_feed_weight →
  total_feed * total_cost_per_pound - cheap_feed_weight * cheap_cost_per_pound = expensive_feed_weight * expensive_cost_per_pound →
  expensive_cost_per_pound = 0.53 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_expensive_feed_l556_55669


namespace NUMINAMATH_GPT_inequality_1_l556_55603

theorem inequality_1 (x : ℝ) : (x - 2) * (1 - 3 * x) > 2 → 1 < x ∧ x < 4 / 3 :=
by sorry

end NUMINAMATH_GPT_inequality_1_l556_55603
