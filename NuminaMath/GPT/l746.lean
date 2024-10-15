import Mathlib

namespace NUMINAMATH_GPT_sum_of_abcd_is_1_l746_74678

theorem sum_of_abcd_is_1
  (a b c d : ℤ)
  (h1 : (x^2 + a*x + b)*(x^2 + c*x + d) = x^4 + 2*x^3 + x^2 + 8*x - 12) :
  a + b + c + d = 1 := by
  sorry

end NUMINAMATH_GPT_sum_of_abcd_is_1_l746_74678


namespace NUMINAMATH_GPT_solve_equation_l746_74656

theorem solve_equation (x : ℝ) : (3 * x - 2 * (10 - x) = 5) → x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_equation_l746_74656


namespace NUMINAMATH_GPT_tan_double_angle_l746_74633

theorem tan_double_angle (α : ℝ) (x y : ℝ) (hxy : y / x = -2) : 
  2 * y / (1 - (y / x)^2) = (4 : ℝ) / 3 :=
by sorry

end NUMINAMATH_GPT_tan_double_angle_l746_74633


namespace NUMINAMATH_GPT_area_transformation_l746_74692

variable {g : ℝ → ℝ}

theorem area_transformation (h : ∫ x, g x = 20) : ∫ x, -4 * g (x + 3) = 80 := by
  sorry

end NUMINAMATH_GPT_area_transformation_l746_74692


namespace NUMINAMATH_GPT_sum_of_squares_l746_74632

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 72) : x^2 + y^2 = 180 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l746_74632


namespace NUMINAMATH_GPT_problem1_l746_74612

variable (x y : ℝ)
variable (h1 : x = Real.sqrt 3 + Real.sqrt 5)
variable (h2 : y = Real.sqrt 3 - Real.sqrt 5)

theorem problem1 : 2 * x^2 - 4 * x * y + 2 * y^2 = 40 :=
by sorry

end NUMINAMATH_GPT_problem1_l746_74612


namespace NUMINAMATH_GPT_max_expression_value_l746_74653

theorem max_expression_value :
  ∀ (a b : ℝ), (100 ≤ a ∧ a ≤ 500) → (500 ≤ b ∧ b ≤ 1500) → 
  (∃ x, x = (b - 100) / (a + 50) ∧ ∀ y, y = (b - 100) / (a + 50) → y ≤ (28 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_max_expression_value_l746_74653


namespace NUMINAMATH_GPT_expected_balls_in_original_pos_after_two_transpositions_l746_74602

theorem expected_balls_in_original_pos_after_two_transpositions :
  ∃ (n : ℚ), n = 3.2 := 
sorry

end NUMINAMATH_GPT_expected_balls_in_original_pos_after_two_transpositions_l746_74602


namespace NUMINAMATH_GPT_floor_double_l746_74681

theorem floor_double (a : ℝ) (h : 0 < a) : 
  ⌊2 * a⌋ = ⌊a⌋ + ⌊a + 1/2⌋ :=
sorry

end NUMINAMATH_GPT_floor_double_l746_74681


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l746_74647

def quadratic_function_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k
  let b := -4
  let c := -2
  b * b - 4 * a * c > 0 ∧ a ≠ 0

theorem quadratic_two_distinct_real_roots (k : ℝ) :
  quadratic_function_has_two_distinct_real_roots k ↔ (k > -2 ∧ k ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l746_74647


namespace NUMINAMATH_GPT_gcd_number_between_75_and_90_is_5_l746_74616

theorem gcd_number_between_75_and_90_is_5 :
  ∃ n : ℕ, 75 ≤ n ∧ n ≤ 90 ∧ Nat.gcd 15 n = 5 :=
sorry

end NUMINAMATH_GPT_gcd_number_between_75_and_90_is_5_l746_74616


namespace NUMINAMATH_GPT_solve_basketball_court_dimensions_l746_74674

theorem solve_basketball_court_dimensions 
  (A B C D E F : ℕ) 
  (h1 : A - B = C) 
  (h2 : D = 2 * (A + B)) 
  (h3 : E = A * B) 
  (h4 : F = 3) : 
  A = 28 ∧ B = 15 ∧ C = 13 ∧ D = 86 ∧ E = 420 ∧ F = 3 := 
by 
  sorry

end NUMINAMATH_GPT_solve_basketball_court_dimensions_l746_74674


namespace NUMINAMATH_GPT_smallest_integer_for_perfect_square_l746_74676

-- Given condition: y = 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11
def y : ℕ := 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11

-- The statement to prove
theorem smallest_integer_for_perfect_square (y : ℕ) : ∃ n : ℕ, n = 110 ∧ ∃ m : ℕ, (y * n) = m^2 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_integer_for_perfect_square_l746_74676


namespace NUMINAMATH_GPT_smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5_l746_74683

def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := n.digits b
  digits = digits.reverse

theorem smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5 :
  ∃ n : ℕ, n = 0b11011 ∧ is_palindrome n 2 ∧ is_palindrome n 5 :=
by
  existsi 0b11011
  sorry

end NUMINAMATH_GPT_smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5_l746_74683


namespace NUMINAMATH_GPT_sin_sum_leq_3_sqrt3_over_2_l746_74636

theorem sin_sum_leq_3_sqrt3_over_2 
  (A B C : ℝ) 
  (h₁ : A + B + C = Real.pi) 
  (h₂ : 0 < A ∧ A < Real.pi)
  (h₃ : 0 < B ∧ B < Real.pi)
  (h₄ : 0 < C ∧ C < Real.pi) :
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_sin_sum_leq_3_sqrt3_over_2_l746_74636


namespace NUMINAMATH_GPT_ordered_triple_solution_l746_74650

theorem ordered_triple_solution (a b c : ℝ) (h1 : a > 5) (h2 : b > 5) (h3 : c > 5)
  (h4 : (a + 3) * (a + 3) / (b + c - 5) + (b + 5) * (b + 5) / (c + a - 7) + (c + 7) * (c + 7) / (a + b - 9) = 49) :
  (a, b, c) = (13, 9, 6) :=
sorry

end NUMINAMATH_GPT_ordered_triple_solution_l746_74650


namespace NUMINAMATH_GPT_initial_amounts_l746_74675

theorem initial_amounts (x y z : ℕ) (h1 : x + y + z = 24)
  (h2 : z = 24 - x - y)
  (h3 : x - (y + z) = 8)
  (h4 : y - (x + z) = 12) :
  x = 13 ∧ y = 7 ∧ z = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_amounts_l746_74675


namespace NUMINAMATH_GPT_center_of_tangent_circle_l746_74691

theorem center_of_tangent_circle (x y : ℝ) 
    (h1 : 3 * x - 4 * y = 20) 
    (h2 : 3 * x - 4 * y = -40) 
    (h3 : x - 3 * y = 0) : 
    (x, y) = (-6, -2) := 
by
    sorry

end NUMINAMATH_GPT_center_of_tangent_circle_l746_74691


namespace NUMINAMATH_GPT_tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n_l746_74607

/-- Given the trigonometric identity and the ratio, we want to prove the relationship between the tangents of the angles. -/
theorem tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n
  (α β m n : ℝ)
  (h : (Real.sin (α + β)) / (Real.sin (α - β)) = m / n) :
  (Real.tan β) / (Real.tan α) = (m - n) / (m + n) :=
  sorry

end NUMINAMATH_GPT_tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n_l746_74607


namespace NUMINAMATH_GPT_find_k_l746_74688

theorem find_k (k : ℝ) (h : 64 / k = 8) : k = 8 := 
sorry

end NUMINAMATH_GPT_find_k_l746_74688


namespace NUMINAMATH_GPT_comic_books_stacking_order_l746_74629

-- Definitions of the conditions
def num_spiderman_books : ℕ := 6
def num_archie_books : ℕ := 5
def num_garfield_books : ℕ := 4

-- Calculations of factorials
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

-- Grouping and order calculation
def ways_to_arrange_group_books : ℕ :=
  factorial num_spiderman_books *
  factorial num_archie_books *
  factorial num_garfield_books

def num_groups : ℕ := 3

def ways_to_arrange_groups : ℕ :=
  factorial num_groups

def total_ways_to_stack_books : ℕ :=
  ways_to_arrange_group_books * ways_to_arrange_groups

-- Theorem stating the total number of different orders
theorem comic_books_stacking_order :
  total_ways_to_stack_books = 12441600 :=
by
  sorry

end NUMINAMATH_GPT_comic_books_stacking_order_l746_74629


namespace NUMINAMATH_GPT_arithmetic_geometric_fraction_l746_74673

theorem arithmetic_geometric_fraction (a x₁ x₂ b y₁ y₂ : ℝ) 
  (h₁ : x₁ + x₂ = a + b) 
  (h₂ : y₁ * y₂ = ab) : 
  (x₁ + x₂) / (y₁ * y₂) = (a + b) / (ab) := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_fraction_l746_74673


namespace NUMINAMATH_GPT_duration_of_period_l746_74679

noncomputable def birth_rate : ℕ := 7
noncomputable def death_rate : ℕ := 3
noncomputable def net_increase : ℕ := 172800

theorem duration_of_period : (net_increase / ((birth_rate - death_rate) / 2)) / 3600 = 12 := by
  sorry

end NUMINAMATH_GPT_duration_of_period_l746_74679


namespace NUMINAMATH_GPT_arrangements_7_people_no_A_at_head_no_B_in_middle_l746_74644

theorem arrangements_7_people_no_A_at_head_no_B_in_middle :
  let n := 7
  let total_arrangements := Nat.factorial n
  let A_at_head := Nat.factorial (n - 1)
  let B_in_middle := A_at_head
  let overlap := Nat.factorial (n - 2)
  total_arrangements - 2 * A_at_head + overlap = 3720 :=
by
  let n := 7
  let total_arrangements := Nat.factorial n
  let A_at_head := Nat.factorial (n - 1)
  let B_in_middle := A_at_head
  let overlap := Nat.factorial (n - 2)
  show total_arrangements - 2 * A_at_head + overlap = 3720
  sorry

end NUMINAMATH_GPT_arrangements_7_people_no_A_at_head_no_B_in_middle_l746_74644


namespace NUMINAMATH_GPT_intersection_complement_eq_singleton_l746_74699

def U : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) }
def M : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ (y - 3) / (x - 2) = 1 }
def N : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ y = x + 1 }
def complement_U (M : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := { p | p ∈ U ∧ p ∉ M }

theorem intersection_complement_eq_singleton :
  N ∩ complement_U M = {(2,3)} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_singleton_l746_74699


namespace NUMINAMATH_GPT_textopolis_word_count_l746_74625

theorem textopolis_word_count :
  let alphabet_size := 26
  let total_one_letter := 2 -- only "A" and "B"
  let total_two_letter := alphabet_size^2
  let excl_two_letter := (alphabet_size - 2)^2
  let total_three_letter := alphabet_size^3
  let excl_three_letter := (alphabet_size - 2)^3
  let total_four_letter := alphabet_size^4
  let excl_four_letter := (alphabet_size - 2)^4
  let valid_two_letter := total_two_letter - excl_two_letter
  let valid_three_letter := total_three_letter - excl_three_letter
  let valid_four_letter := total_four_letter - excl_four_letter
  2 + valid_two_letter + valid_three_letter + valid_four_letter = 129054 := by
  -- To be proved
  sorry

end NUMINAMATH_GPT_textopolis_word_count_l746_74625


namespace NUMINAMATH_GPT_second_die_sides_l746_74663

theorem second_die_sides (p : ℚ) (n : ℕ) (h1 : p = 0.023809523809523808) (h2 : n ≠ 0) :
  let first_die_sides := 6
  let probability := (1 : ℚ) / first_die_sides * (1 : ℚ) / n
  probability = p → n = 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_second_die_sides_l746_74663


namespace NUMINAMATH_GPT_percent_not_participating_music_sports_l746_74672

theorem percent_not_participating_music_sports
  (total_students : ℕ) 
  (both : ℕ) 
  (music_only : ℕ) 
  (sports_only : ℕ) 
  (not_participating : ℕ)
  (percentage_not_participating : ℝ) :
  total_students = 50 →
  both = 5 →
  music_only = 15 →
  sports_only = 20 →
  not_participating = total_students - (both + music_only + sports_only) →
  percentage_not_participating = (not_participating : ℝ) / (total_students : ℝ) * 100 →
  percentage_not_participating = 20 :=
by
  sorry

end NUMINAMATH_GPT_percent_not_participating_music_sports_l746_74672


namespace NUMINAMATH_GPT_maximum_value_of_x_minus_y_is_sqrt8_3_l746_74628

variable {x y z : ℝ}

noncomputable def maximum_value_of_x_minus_y (x y z : ℝ) : ℝ :=
  x - y

theorem maximum_value_of_x_minus_y_is_sqrt8_3 (h1 : x + y + z = 2) (h2 : x * y + y * z + z * x = 1) : 
  maximum_value_of_x_minus_y x y z = Real.sqrt (8 / 3) :=
sorry

end NUMINAMATH_GPT_maximum_value_of_x_minus_y_is_sqrt8_3_l746_74628


namespace NUMINAMATH_GPT_painters_workdays_l746_74665

theorem painters_workdays (d₁ d₂ : ℚ) (p₁ p₂ : ℕ)
  (h1 : p₁ = 5) (h2 : p₂ = 4) (rate: 5 * d₁ = 7.5) :
  (p₂:ℚ) * d₂ = 7.5 → d₂ = 1 + 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_painters_workdays_l746_74665


namespace NUMINAMATH_GPT_scientific_notation_of_170000_l746_74630

-- Define the concept of scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  (1 ≤ a) ∧ (a < 10) ∧ (x = a * 10^n)

-- The main statement to prove
theorem scientific_notation_of_170000 : is_scientific_notation 1.7 5 170000 :=
by sorry

end NUMINAMATH_GPT_scientific_notation_of_170000_l746_74630


namespace NUMINAMATH_GPT_number_of_buildings_l746_74626

theorem number_of_buildings (studio_apartments : ℕ) (two_person_apartments : ℕ) (four_person_apartments : ℕ)
    (occupancy_percentage : ℝ) (current_occupancy : ℕ)
    (max_occupancy_building : ℕ) (max_occupancy_complex : ℕ) (num_buildings : ℕ)
    (h_studio : studio_apartments = 10)
    (h_two_person : two_person_apartments = 20)
    (h_four_person : four_person_apartments = 5)
    (h_occupancy_percentage : occupancy_percentage = 0.75)
    (h_current_occupancy : current_occupancy = 210)
    (h_max_occupancy_building : max_occupancy_building = 10 * 1 + 20 * 2 + 5 * 4)
    (h_max_occupancy_complex : max_occupancy_complex = current_occupancy / occupancy_percentage)
    (h_num_buildings : num_buildings = max_occupancy_complex / max_occupancy_building) :
    num_buildings = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_buildings_l746_74626


namespace NUMINAMATH_GPT_acute_triangle_angle_A_is_60_degrees_l746_74693

open Real

variables {A B C : ℝ} -- Assume A, B, C are reals representing the angles of the triangle

theorem acute_triangle_angle_A_is_60_degrees
  (h_acute : A < 90 ∧ B < 90 ∧ C < 90)
  (h_eq_dist : dist A O = dist A H) : A = 60 :=
  sorry

end NUMINAMATH_GPT_acute_triangle_angle_A_is_60_degrees_l746_74693


namespace NUMINAMATH_GPT_line_shift_up_l746_74634

theorem line_shift_up (x y : ℝ) (k : ℝ) (h : y = -2 * x - 4) : 
    y + k = -2 * x - 1 := by
  sorry

end NUMINAMATH_GPT_line_shift_up_l746_74634


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l746_74649

theorem equation1_solution (x : ℝ) : x^2 - 10*x + 16 = 0 ↔ x = 2 ∨ x = 8 :=
by sorry

theorem equation2_solution (x : ℝ) : 2*x*(x-1) = x-1 ↔ x = 1 ∨ x = 1/2 :=
by sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l746_74649


namespace NUMINAMATH_GPT_july14_2030_is_sunday_l746_74610

-- Define the given condition that July 3, 2030 is a Wednesday. 
def july3_2030_is_wednesday : Prop := true -- Assume the existence and correctness of this statement.

-- Define the proof problem that July 14, 2030 is a Sunday given the above condition.
theorem july14_2030_is_sunday : july3_2030_is_wednesday → (14 % 7 = 0) := 
sorry

end NUMINAMATH_GPT_july14_2030_is_sunday_l746_74610


namespace NUMINAMATH_GPT_area_change_correct_l746_74654

theorem area_change_correct (L B : ℝ) (A : ℝ) (x : ℝ) (hx1 : A = L * B)
  (hx2 : ((L + (x / 100) * L) * (B - (x / 100) * B)) = A - (1 / 100) * A) :
  x = 10 := by
  sorry

end NUMINAMATH_GPT_area_change_correct_l746_74654


namespace NUMINAMATH_GPT_evaluate_expression_l746_74642

theorem evaluate_expression
  (p q r s : ℚ)
  (h1 : p / q = 4 / 5)
  (h2 : r / s = 3 / 7) :
  (18 / 7) + ((2 * q - p) / (2 * q + p)) - ((3 * s + r) / (3 * s - r)) = 5 / 3 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l746_74642


namespace NUMINAMATH_GPT_martin_boxes_l746_74655

theorem martin_boxes (total_crayons : ℕ) (crayons_per_box : ℕ) (number_of_boxes : ℕ) 
  (h1 : total_crayons = 56) (h2 : crayons_per_box = 7) 
  (h3 : total_crayons = crayons_per_box * number_of_boxes) : 
  number_of_boxes = 8 :=
by 
  sorry

end NUMINAMATH_GPT_martin_boxes_l746_74655


namespace NUMINAMATH_GPT_not_every_constant_is_geometric_l746_74658

def is_constant_sequence (s : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, s n = s m

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem not_every_constant_is_geometric :
  (¬ ∀ s : ℕ → ℝ, is_constant_sequence s → is_geometric_sequence s) ↔
  ∃ s : ℕ → ℝ, is_constant_sequence s ∧ ¬ is_geometric_sequence s := 
by
  sorry

end NUMINAMATH_GPT_not_every_constant_is_geometric_l746_74658


namespace NUMINAMATH_GPT_tangent_line_at_1_l746_74664

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x
def tangent_line_eq : ℝ × ℝ → ℝ := fun ⟨x, y⟩ => x - y - 1

theorem tangent_line_at_1 : tangent_line_eq (1, f 1) = 0 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_tangent_line_at_1_l746_74664


namespace NUMINAMATH_GPT_Lagrange_interpol_equiv_x_squared_l746_74627

theorem Lagrange_interpol_equiv_x_squared (a b c x : ℝ)
    (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
    c^2 * ((x - a) * (x - b)) / ((c - a) * (c - b)) +
    b^2 * ((x - a) * (x - c)) / ((b - a) * (b - c)) +
    a^2 * ((x - b) * (x - c)) / ((a - b) * (a - c)) = x^2 := 
    sorry

end NUMINAMATH_GPT_Lagrange_interpol_equiv_x_squared_l746_74627


namespace NUMINAMATH_GPT_tan_alpha_eq_one_then_expr_value_l746_74631

theorem tan_alpha_eq_one_then_expr_value (α : ℝ) (h : Real.tan α = 1) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_one_then_expr_value_l746_74631


namespace NUMINAMATH_GPT_Olivia_paint_area_l746_74684

theorem Olivia_paint_area
  (length width height : ℕ) (door_window_area : ℕ) (bedrooms : ℕ)
  (h_length : length = 14) 
  (h_width : width = 11) 
  (h_height : height = 9) 
  (h_door_window_area : door_window_area = 70) 
  (h_bedrooms : bedrooms = 4) :
  (2 * (length * height) + 2 * (width * height) - door_window_area) * bedrooms = 1520 :=
by
  sorry

end NUMINAMATH_GPT_Olivia_paint_area_l746_74684


namespace NUMINAMATH_GPT_percentage_difference_l746_74690

variables (G P R : ℝ)

-- Conditions
def condition1 : Prop := P = 0.9 * G
def condition2 : Prop := R = 3.0000000000000006 * G

-- Theorem to prove
theorem percentage_difference (h1 : condition1 P G) (h2 : condition2 R G) : 
  (R - P) / R * 100 = 70 :=
sorry

end NUMINAMATH_GPT_percentage_difference_l746_74690


namespace NUMINAMATH_GPT_triangle_inequality_inequality_l746_74677

variable {a b c : ℝ}
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (triangle_ineq : a + b > c)

theorem triangle_inequality_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) (triangle_ineq : a + b > c) :
  a^3 + b^3 + 3 * a * b * c > c^3 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_inequality_l746_74677


namespace NUMINAMATH_GPT_find_c_find_cos_2B_minus_pi_over_4_l746_74614

variable (A B C : Real) (a b c : Real)

-- Given conditions
def conditions (a b c : Real) (A : Real) : Prop :=
  a = 4 * Real.sqrt 3 ∧
  b = 6 ∧
  Real.cos A = -1 / 3

-- Proof of question 1
theorem find_c (h : conditions a b c A) : c = 2 :=
sorry

-- Proof of question 2
theorem find_cos_2B_minus_pi_over_4 (h : conditions a b c A) (B : Real) :
  (angle_opp_b : b = Real.sin B) → -- This is to ensure B is the angle opposite to side b
  Real.cos (2 * B - Real.pi / 4) = (4 - Real.sqrt 2) / 6 :=
sorry

end NUMINAMATH_GPT_find_c_find_cos_2B_minus_pi_over_4_l746_74614


namespace NUMINAMATH_GPT_smallest_positive_m_l746_74652

theorem smallest_positive_m (m : ℕ) (h : ∃ n : ℤ, m^3 - 90 = n * (m + 9)) : m = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_m_l746_74652


namespace NUMINAMATH_GPT_line_and_circle_condition_l746_74624

theorem line_and_circle_condition (P Q : ℝ × ℝ) (radius : ℝ) 
  (x y m : ℝ) (n : ℝ) (l : ℝ × ℝ → Prop)
  (hPQ : P = (4, -2)) 
  (hPQ' : Q = (-1, 3)) 
  (hC : ∀ (x y : ℝ), (x - 1)^2 + y^2 = radius) 
  (hr : radius < 5) 
  (h_y_segment : ∃ (k : ℝ), |k - 0| = 4 * Real.sqrt 3) 
  : (∀ (x y : ℝ), x + y = 2) ∧ 
    ((∀ (x y : ℝ), l (x, y) ↔ x + y + m = 0 ∨ x + y = 0) 
    ∧ (m = 3 ∨ m = -4) 
    ∧ (∀ A B : ℝ × ℝ, l A → l B → (A.1 - B.1)^2 + (A.2 - B.2)^2 = radius)) := 
  by
  sorry

end NUMINAMATH_GPT_line_and_circle_condition_l746_74624


namespace NUMINAMATH_GPT_gcd_9011_2147_l746_74689

theorem gcd_9011_2147 : Int.gcd 9011 2147 = 1 := sorry

end NUMINAMATH_GPT_gcd_9011_2147_l746_74689


namespace NUMINAMATH_GPT_bread_products_wasted_l746_74615

theorem bread_products_wasted :
  (50 * 8 - (20 * 5 + 15 * 4 + 10 * 10 * 1.5)) / 1.5 = 60 := by
  -- The proof steps are omitted here
  sorry

end NUMINAMATH_GPT_bread_products_wasted_l746_74615


namespace NUMINAMATH_GPT_positive_difference_of_fraction_results_l746_74648

theorem positive_difference_of_fraction_results :
  let a := 8
  let expr1 := (a ^ 2 - a ^ 2) / a
  let expr2 := (a ^ 2 * a ^ 2) / a
  expr1 = 0 ∧ expr2 = 512 ∧ (expr2 - expr1) = 512 := 
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_fraction_results_l746_74648


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_m_eq_1_sufficient_m_eq_1_not_necessary_l746_74640

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (4, -2)

def perp_vectors (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem sufficient_not_necessary_condition :
  perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) ↔ (m = 1 ∨ m = -3) :=
by
  sorry

theorem m_eq_1_sufficient :
  (m = 1) → perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) :=
by
  sorry

theorem m_eq_1_not_necessary :
  perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) → (m = 1 ∨ m = -3) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_m_eq_1_sufficient_m_eq_1_not_necessary_l746_74640


namespace NUMINAMATH_GPT_find_sixth_term_l746_74621

noncomputable def first_term : ℝ := Real.sqrt 3
noncomputable def fifth_term : ℝ := Real.sqrt 243
noncomputable def common_ratio (q : ℝ) : Prop := fifth_term = first_term * q^4
noncomputable def sixth_term (b6 : ℝ) (q : ℝ) : Prop := b6 = fifth_term * q

theorem find_sixth_term (q : ℝ) (b6 : ℝ) : 
  first_term = Real.sqrt 3 ∧
  fifth_term = Real.sqrt 243 ∧
  common_ratio q ∧ 
  sixth_term b6 q → 
  b6 = 27 ∨ b6 = -27 := 
by
  intros
  sorry

end NUMINAMATH_GPT_find_sixth_term_l746_74621


namespace NUMINAMATH_GPT_ratio_of_lateral_edges_l746_74651

theorem ratio_of_lateral_edges (A B : ℝ) (hA : A > 0) (hB : B > 0) (h : A / B = 4 / 9) : 
  let upper_length_ratio := 2
  let lower_length_ratio := 3
  upper_length_ratio / lower_length_ratio = 2 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_lateral_edges_l746_74651


namespace NUMINAMATH_GPT_expr_value_l746_74669

-- Define the constants
def w : ℤ := 3
def x : ℤ := -2
def y : ℤ := 1
def z : ℤ := 4

-- Define the expression
def expr : ℤ := (w^2 * x^2 * y * z) - (w * x^2 * y * z^2) + (w * y^3 * z^2) - (w * y^2 * x * z^4)

-- Statement to be proved
theorem expr_value : expr = 1536 :=
by
  -- Proof is omitted, so we use sorry.
  sorry

end NUMINAMATH_GPT_expr_value_l746_74669


namespace NUMINAMATH_GPT_integer_modulo_problem_l746_74698

theorem integer_modulo_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ (-250 % 23 = n) := 
  sorry

end NUMINAMATH_GPT_integer_modulo_problem_l746_74698


namespace NUMINAMATH_GPT_sum_of_common_ratios_l746_74662

variable {k a_2 a_3 b_2 b_3 p r : ℝ}
variable (hp : a_2 = k * p) (ha3 : a_3 = k * p^2)
variable (hr : b_2 = k * r) (hb3 : b_3 = k * r^2)
variable (hcond : a_3 - b_3 = 5 * (a_2 - b_2))

theorem sum_of_common_ratios (h_nonconst : k ≠ 0) (p_ne_r : p ≠ r) : p + r = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_common_ratios_l746_74662


namespace NUMINAMATH_GPT_john_speed_first_part_l746_74635

theorem john_speed_first_part (S : ℝ) (h1 : 2 * S + 3 * 55 = 255) : S = 45 :=
by
  sorry

end NUMINAMATH_GPT_john_speed_first_part_l746_74635


namespace NUMINAMATH_GPT_melanie_turnips_l746_74639

theorem melanie_turnips (b : ℕ) (d : ℕ) (h_b : b = 113) (h_d : d = 26) : b + d = 139 :=
by
  sorry

end NUMINAMATH_GPT_melanie_turnips_l746_74639


namespace NUMINAMATH_GPT_minimum_value_of_a_squared_plus_b_squared_l746_74611

def quadratic (a b x : ℝ) : ℝ := a * x^2 + (2 * b + 1) * x - a - 2

theorem minimum_value_of_a_squared_plus_b_squared (a b : ℝ) (hab : a ≠ 0)
  (hroot : ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 4 ∧ quadratic a b x = 0) :
  a^2 + b^2 = 1 / 100 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_a_squared_plus_b_squared_l746_74611


namespace NUMINAMATH_GPT_alcohol_percentage_l746_74604

theorem alcohol_percentage (x : ℝ)
  (h1 : 8 * x / 100 + 2 * 12 / 100 = 22.4 * 10 / 100) : x = 25 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_alcohol_percentage_l746_74604


namespace NUMINAMATH_GPT_solve_sqrt_eq_l746_74600

theorem solve_sqrt_eq (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2)^x) + Real.sqrt ((3 - 2 * Real.sqrt 2)^x) = 5) ↔ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_GPT_solve_sqrt_eq_l746_74600


namespace NUMINAMATH_GPT_four_at_three_equals_thirty_l746_74638

def custom_operation (a b : ℕ) : ℕ :=
  3 * a^2 - 2 * b^2

theorem four_at_three_equals_thirty : custom_operation 4 3 = 30 :=
by
  sorry

end NUMINAMATH_GPT_four_at_three_equals_thirty_l746_74638


namespace NUMINAMATH_GPT_smallest_right_triangle_area_l746_74643

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end NUMINAMATH_GPT_smallest_right_triangle_area_l746_74643


namespace NUMINAMATH_GPT_tangent_triangle_perimeter_acute_tangent_triangle_perimeter_obtuse_l746_74686

theorem tangent_triangle_perimeter_acute (a b c: ℝ) (h1: a^2 + b^2 > c^2) (h2: b^2 + c^2 > a^2) (h3: c^2 + a^2 > b^2) :
  2 * a * b * c * (1 / (b^2 + c^2 - a^2) + 1 / (c^2 + a^2 - b^2) + 1 / (a^2 + b^2 - c^2)) = 
  2 * a * b * c * (1 / (b^2 + c^2 - a^2) + 1 / (c^2 + a^2 - b^2) + 1 / (a^2 + b^2 - c^2)) := 
by sorry -- proof goes here

theorem tangent_triangle_perimeter_obtuse (a b c: ℝ) (h1: a^2 > b^2 + c^2) :
  2 * a * b * c / (a^2 - b^2 - c^2) = 2 * a * b * c / (a^2 - b^2 - c^2) := 
by sorry -- proof goes here

end NUMINAMATH_GPT_tangent_triangle_perimeter_acute_tangent_triangle_perimeter_obtuse_l746_74686


namespace NUMINAMATH_GPT_total_new_cans_l746_74668

-- Define the condition
def initial_cans : ℕ := 256
def first_term : ℕ := 64
def ratio : ℚ := 1 / 4
def terms : ℕ := 4

-- Define the sum of the geometric series
noncomputable def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * ((1 - r ^ n) / (1 - r))

-- Problem statement in Lean 4
theorem total_new_cans : geometric_series_sum first_term ratio terms = 85 := by
  sorry

end NUMINAMATH_GPT_total_new_cans_l746_74668


namespace NUMINAMATH_GPT_express_2011_with_digit_1_l746_74680

theorem express_2011_with_digit_1 :
  ∃ (a b c d e: ℕ), 2011 = a * b - c * d + e - f + g ∧
  (a = 1111 ∧ b = 1111) ∧ (c = 111 ∧ d = 11111) ∧ (e = 1111) ∧ (f = 111) ∧ (g = 11) ∧
  (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ g) :=
sorry

end NUMINAMATH_GPT_express_2011_with_digit_1_l746_74680


namespace NUMINAMATH_GPT_remainder_7n_mod_4_l746_74619

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by sorry

end NUMINAMATH_GPT_remainder_7n_mod_4_l746_74619


namespace NUMINAMATH_GPT_hyperbola_eccentricity_range_l746_74603

theorem hyperbola_eccentricity_range {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) :
  ∃ e : ℝ, e = (Real.sqrt (a^2 + b^2)) / a ∧ 1 < e ∧ e < Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_range_l746_74603


namespace NUMINAMATH_GPT_pump_A_time_l746_74613

theorem pump_A_time (B C A : ℝ) (hB : B = 1/3) (hC : C = 1/6)
(h : (A + B - C) * 0.75 = 0.5) : 1 / A = 2 :=
by
sorry

end NUMINAMATH_GPT_pump_A_time_l746_74613


namespace NUMINAMATH_GPT_pupils_in_class_l746_74609

theorem pupils_in_class (n : ℕ) (wrong_entry_increase : n * (1/2) = 13) : n = 26 :=
sorry

end NUMINAMATH_GPT_pupils_in_class_l746_74609


namespace NUMINAMATH_GPT_inequality_proof_l746_74646

theorem inequality_proof (x y z : ℝ) (hx : x ≥ y) (hy : y ≥ z) (hz : z > 0) :
  (x^2 * y / z + y^2 * z / x + z^2 * x / y) ≥ (x^2 + y^2 + z^2) := 
  sorry

end NUMINAMATH_GPT_inequality_proof_l746_74646


namespace NUMINAMATH_GPT_servant_service_duration_l746_74682

variables (x : ℕ) (total_compensation full_months received_compensation : ℕ)
variables (price_uniform compensation_cash : ℕ)

theorem servant_service_duration :
  total_compensation = 1000 →
  full_months = 12 →
  received_compensation = (compensation_cash + price_uniform) →
  received_compensation = 750 →
  total_compensation = (compensation_cash + price_uniform) →
  x / full_months = 750 / total_compensation →
  x = 9 :=
by sorry

end NUMINAMATH_GPT_servant_service_duration_l746_74682


namespace NUMINAMATH_GPT_find_n_mod_60_l746_74657

theorem find_n_mod_60 {x y : ℤ} (hx : x ≡ 45 [ZMOD 60]) (hy : y ≡ 98 [ZMOD 60]) :
  ∃ n, 150 ≤ n ∧ n ≤ 210 ∧ (x - y ≡ n [ZMOD 60]) ∧ n = 187 := by
  sorry

end NUMINAMATH_GPT_find_n_mod_60_l746_74657


namespace NUMINAMATH_GPT_problem_series_sum_l746_74685

noncomputable def series_sum : ℝ := ∑' n : ℕ, (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem problem_series_sum :
  series_sum = 1 / 200 :=
sorry

end NUMINAMATH_GPT_problem_series_sum_l746_74685


namespace NUMINAMATH_GPT_count_multiples_of_5_not_10_or_15_l746_74645

theorem count_multiples_of_5_not_10_or_15 : 
  ∃ n : ℕ, n = 33 ∧ (∀ x : ℕ, x < 500 ∧ (x % 5 = 0) ∧ (x % 10 ≠ 0) ∧ (x % 15 ≠ 0) → x < 500 ∧ (x % 5 = 0) ∧ (x % 10 ≠ 0) ∧ (x % 15 ≠ 0)) :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_of_5_not_10_or_15_l746_74645


namespace NUMINAMATH_GPT_count_integers_with_block_178_l746_74670

theorem count_integers_with_block_178 (a b : ℕ) : 10000 ≤ a ∧ a < 100000 → 10000 ≤ b ∧ b < 100000 → a = b → b - a = 99999 → ∃ n, n = 280 ∧ (n = a + b) := sorry

end NUMINAMATH_GPT_count_integers_with_block_178_l746_74670


namespace NUMINAMATH_GPT_increasing_decreasing_intervals_l746_74618

noncomputable def f (x : ℝ) : ℝ := Real.sin (-2 * x + 3 * Real.pi / 4)

theorem increasing_decreasing_intervals : (∀ k : ℤ, 
    ∀ x, 
      ((k : ℝ) * Real.pi + 5 * Real.pi / 8 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 9 * Real.pi / 8) 
      → 0 < f x ∧ f x < 1) 
  ∧ 
    (∀ k : ℤ, 
    ∀ x, 
      ((k : ℝ) * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 5 * Real.pi / 8) 
      → -1 < f x ∧ f x < 0) :=
by
  sorry

end NUMINAMATH_GPT_increasing_decreasing_intervals_l746_74618


namespace NUMINAMATH_GPT_constant_sums_l746_74641

theorem constant_sums (n : ℕ) 
  (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : x * y * z = 1) 
  : (x^n + y^n + z^n = 0 ∨ x^n + y^n + z^n = 3) ↔ (n = 1 ∨ n = 3) :=
by sorry

end NUMINAMATH_GPT_constant_sums_l746_74641


namespace NUMINAMATH_GPT_angle_terminal_side_l746_74620

def angle_on_line (β : ℝ) : Prop :=
  ∃ n : ℤ, β = 135 + n * 180

def angle_in_range (β : ℝ) : Prop :=
  -360 < β ∧ β < 360

theorem angle_terminal_side :
  ∀ β, angle_on_line β → angle_in_range β → β = -225 ∨ β = -45 ∨ β = 135 ∨ β = 315 :=
by
  intros β h_line h_range
  sorry

end NUMINAMATH_GPT_angle_terminal_side_l746_74620


namespace NUMINAMATH_GPT_max_value_neg7s_squared_plus_56s_plus_20_l746_74606

theorem max_value_neg7s_squared_plus_56s_plus_20 :
  ∃ s : ℝ, s = 4 ∧ ∀ t : ℝ, -7 * t^2 + 56 * t + 20 ≤ 132 := 
by
  sorry

end NUMINAMATH_GPT_max_value_neg7s_squared_plus_56s_plus_20_l746_74606


namespace NUMINAMATH_GPT_positive_difference_balances_l746_74671

noncomputable def cedric_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

noncomputable def daniel_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem positive_difference_balances :
  let P : ℝ := 15000
  let r_cedric : ℝ := 0.06
  let r_daniel : ℝ := 0.08
  let t : ℕ := 15
  let A_cedric := cedric_balance P r_cedric t
  let A_daniel := daniel_balance P r_daniel t
  (A_daniel - A_cedric) = 11632.65 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_balances_l746_74671


namespace NUMINAMATH_GPT_finish_remaining_work_l746_74697

theorem finish_remaining_work (x y : ℕ) (hx : x = 30) (hy : y = 15) (hy_work_days : y_work_days = 10) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_finish_remaining_work_l746_74697


namespace NUMINAMATH_GPT_percentage_of_burpees_is_10_l746_74667

-- Definitions for each exercise count
def jumping_jacks : ℕ := 25
def pushups : ℕ := 15
def situps : ℕ := 30
def burpees : ℕ := 10
def lunges : ℕ := 20

-- Total number of exercises
def total_exercises : ℕ := jumping_jacks + pushups + situps + burpees + lunges

-- The proof statement
theorem percentage_of_burpees_is_10 :
  (burpees * 100) / total_exercises = 10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_burpees_is_10_l746_74667


namespace NUMINAMATH_GPT_cos_of_angle_sum_l746_74608

variable (θ : ℝ)

-- Given condition
axiom sin_theta : Real.sin θ = 1 / 4

-- To prove
theorem cos_of_angle_sum : Real.cos (3 * Real.pi / 2 + θ) = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cos_of_angle_sum_l746_74608


namespace NUMINAMATH_GPT_incorrect_options_l746_74622

variable (a b : ℚ) (h : a / b = 5 / 6)

theorem incorrect_options :
  (2 * a - b ≠ b * 6 / 4) ∧
  (a + 3 * b ≠ 2 * a * 19 / 10) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_options_l746_74622


namespace NUMINAMATH_GPT_martha_meeting_distance_l746_74660

theorem martha_meeting_distance (t : ℝ) (d : ℝ)
  (h1 : 0 < t)
  (h2 : d = 45 * (t + 0.75))
  (h3 : d - 45 = 55 * (t - 1)) :
  d = 230.625 := 
  sorry

end NUMINAMATH_GPT_martha_meeting_distance_l746_74660


namespace NUMINAMATH_GPT_builder_total_amount_paid_l746_74659

theorem builder_total_amount_paid :
  let cost_drill_bits := 5 * 6
  let tax_drill_bits := 0.10 * cost_drill_bits
  let total_cost_drill_bits := cost_drill_bits + tax_drill_bits

  let cost_hammers := 3 * 8
  let discount_hammers := 0.05 * cost_hammers
  let total_cost_hammers := cost_hammers - discount_hammers

  let cost_toolbox := 25
  let tax_toolbox := 0.15 * cost_toolbox
  let total_cost_toolbox := cost_toolbox + tax_toolbox

  let total_amount_paid := total_cost_drill_bits + total_cost_hammers + total_cost_toolbox

  total_amount_paid = 84.55 :=
by
  sorry

end NUMINAMATH_GPT_builder_total_amount_paid_l746_74659


namespace NUMINAMATH_GPT_solve_for_nabla_l746_74605

theorem solve_for_nabla (nabla : ℤ) (h : 3 * (-2) = nabla + 2) : nabla = -8 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_nabla_l746_74605


namespace NUMINAMATH_GPT_arithmetic_sequence_l746_74695

theorem arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) (h1 : a 2 = 3) (h2 : a (n - 1) = 17) (h3 : n ≥ 2) (h4 : (n * (3 + 17)) / 2 = 100) : n = 10 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_l746_74695


namespace NUMINAMATH_GPT_complex_square_l746_74617

theorem complex_square (z : ℂ) (i : ℂ) (h1 : z = 2 - 3 * i) (h2 : i^2 = -1) : z^2 = -5 - 12 * i :=
sorry

end NUMINAMATH_GPT_complex_square_l746_74617


namespace NUMINAMATH_GPT_proof_inequality_l746_74696

noncomputable def problem_statement (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : Prop :=
  a + b + c ≤ (a ^ 4 + b ^ 4 + c ^ 4) / (a * b * c)

theorem proof_inequality (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  problem_statement a b c h_a h_b h_c :=
by
  sorry

end NUMINAMATH_GPT_proof_inequality_l746_74696


namespace NUMINAMATH_GPT_Uncle_Bradley_bills_l746_74666

theorem Uncle_Bradley_bills :
  let total_money := 1000
  let fifty_bills_portion := 3 / 10
  let fifty_bill_value := 50
  let hundred_bill_value := 100
  -- Calculate the number of $50 bills
  let fifty_bills_count := (total_money * fifty_bills_portion) / fifty_bill_value
  -- Calculate the number of $100 bills
  let hundred_bills_count := (total_money * (1 - fifty_bills_portion)) / hundred_bill_value
  -- Calculate the total number of bills
  fifty_bills_count + hundred_bills_count = 13 :=
by 
  -- Note: Proof omitted, as it is not required 
  sorry

end NUMINAMATH_GPT_Uncle_Bradley_bills_l746_74666


namespace NUMINAMATH_GPT_contrapositive_proposition_l746_74637

theorem contrapositive_proposition (x y : ℝ) :
  (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
sorry

end NUMINAMATH_GPT_contrapositive_proposition_l746_74637


namespace NUMINAMATH_GPT_smallest_third_term_arith_seq_l746_74687

theorem smallest_third_term_arith_seq {a d : ℕ} 
  (h1 : a > 0) 
  (h2 : d > 0) 
  (sum_eq : 5 * a + 10 * d = 80) : 
  a + 2 * d = 16 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_third_term_arith_seq_l746_74687


namespace NUMINAMATH_GPT_net_calorie_deficit_l746_74661

-- Define the conditions as constants.
def total_distance : ℕ := 3
def calories_burned_per_mile : ℕ := 150
def calories_in_candy_bar : ℕ := 200

-- Prove the net calorie deficit.
theorem net_calorie_deficit : total_distance * calories_burned_per_mile - calories_in_candy_bar = 250 := by
  sorry

end NUMINAMATH_GPT_net_calorie_deficit_l746_74661


namespace NUMINAMATH_GPT_number_of_ways_to_write_2024_l746_74623

theorem number_of_ways_to_write_2024 :
  (∃ a b c : ℕ, 2 * a + 3 * b + 4 * c = 2024) -> 
  (∃ n m p : ℕ, a = 3 * n + 2 * m + p ∧ n + m + p = 337) ->
  (∃ n m p : ℕ, n + m + p = 337 ∧ 2 * n * 3 + m * 2 + p * 6 = 2 * (57231 + 498)) :=
sorry

end NUMINAMATH_GPT_number_of_ways_to_write_2024_l746_74623


namespace NUMINAMATH_GPT_inequality_proof_l746_74601

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
  (1 / a + 4 / (1 - a) ≥ 9) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l746_74601


namespace NUMINAMATH_GPT_therapy_hours_l746_74694

theorem therapy_hours (x n : ℕ) : 
  (x + 30) + 2 * x = 252 → 
  104 + (n - 1) * x = 400 → 
  x = 74 → 
  n = 5 := 
by
  sorry

end NUMINAMATH_GPT_therapy_hours_l746_74694
