import Mathlib

namespace NUMINAMATH_GPT_line_tangent_to_circle_l1627_162747

theorem line_tangent_to_circle (l : ℝ → ℝ) (P : ℝ × ℝ) 
  (hP1 : P = (0, 1)) (hP2 : ∀ x y : ℝ, x^2 + y^2 = 1 -> l x = y)
  (hTangent : ∀ x y : ℝ, l x = y ↔ x^2 + y^2 = 1 ∧ y = 1):
  l x = 1 := by
  sorry

end NUMINAMATH_GPT_line_tangent_to_circle_l1627_162747


namespace NUMINAMATH_GPT_M_subsetneq_P_l1627_162756

def U : Set ℝ := Set.univ
def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x^2 > 1}

theorem M_subsetneq_P : M ⊂ P :=
by sorry

end NUMINAMATH_GPT_M_subsetneq_P_l1627_162756


namespace NUMINAMATH_GPT_nail_insertion_l1627_162751

theorem nail_insertion (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  (4/7) + (4/7) * k + (4/7) * k^2 = 1 :=
by sorry

end NUMINAMATH_GPT_nail_insertion_l1627_162751


namespace NUMINAMATH_GPT_time_to_cross_bridge_l1627_162778

noncomputable def train_crossing_time
  (length_train : ℕ)
  (length_bridge : ℕ)
  (speed_kmph : ℕ)
  (conversion_factor : ℚ) : ℚ :=
  (length_train + length_bridge) / (speed_kmph * conversion_factor)

theorem time_to_cross_bridge :
  train_crossing_time 135 240 45 (5 / 18) = 30 := by
  sorry

end NUMINAMATH_GPT_time_to_cross_bridge_l1627_162778


namespace NUMINAMATH_GPT_min_value_of_2x_plus_y_l1627_162799

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + 2*x*y - 3 = 0) :
  2*x + y ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_2x_plus_y_l1627_162799


namespace NUMINAMATH_GPT_paul_collected_total_cans_l1627_162783

theorem paul_collected_total_cans :
  let saturday_bags := 10
  let sunday_bags := 5
  let saturday_cans_per_bag := 12
  let sunday_cans_per_bag := 15
  let saturday_total_cans := saturday_bags * saturday_cans_per_bag
  let sunday_total_cans := sunday_bags * sunday_cans_per_bag
  let total_cans := saturday_total_cans + sunday_total_cans
  total_cans = 195 := 
by
  sorry

end NUMINAMATH_GPT_paul_collected_total_cans_l1627_162783


namespace NUMINAMATH_GPT_find_a_l1627_162720

theorem find_a (x a : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) :
  A = (7, 1) ∧ B = (1, 4) ∧ C = (x, a * x) ∧ 
  (x - 7, a * x - 1) = (2 * (1 - x), 2 * (4 - a * x)) → 
  a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l1627_162720


namespace NUMINAMATH_GPT_equilateral_triangle_area_with_inscribed_circle_l1627_162773

theorem equilateral_triangle_area_with_inscribed_circle
  (r : ℝ) (area_circle : ℝ) (area_triangle : ℝ) 
  (h_inscribed_circle_area : area_circle = 9 * Real.pi)
  (h_radius : r = 3) :
  area_triangle = 27 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_area_with_inscribed_circle_l1627_162773


namespace NUMINAMATH_GPT_find_k_l1627_162724

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 3)

def vec_add_2b (k : ℝ) : ℝ × ℝ := (2 + 2 * k, 7)
def vec_sub_b (k : ℝ) : ℝ × ℝ := (4 - k, -1)

def vectors_not_parallel (k : ℝ) : Prop :=
  (vec_add_2b k).fst * (vec_sub_b k).snd ≠ (vec_add_2b k).snd * (vec_sub_b k).fst

theorem find_k (k : ℝ) (h : vectors_not_parallel k) : k ≠ 6 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1627_162724


namespace NUMINAMATH_GPT_none_of_these_valid_l1627_162798

variables {x y z w u v : ℝ}

def statement_1 (x y z w : ℝ) := x > y → z < w
def statement_2 (z w u v : ℝ) := z > w → u < v

theorem none_of_these_valid (h₁ : statement_1 x y z w) (h₂ : statement_2 z w u v) :
  ¬ ( (x < y → u < v) ∨ (u < v → x < y) ∨ (u > v → x > y) ∨ (x > y → u > v) ) :=
by {
  sorry
}

end NUMINAMATH_GPT_none_of_these_valid_l1627_162798


namespace NUMINAMATH_GPT_pentagon_area_l1627_162775

open Real

/-- The area of a pentagon with sides 18, 25, 30, 28, and 25 units is 950 square units -/
theorem pentagon_area (a b c d e : ℝ) (h₁ : a = 18) (h₂ : b = 25) (h₃ : c = 30) (h₄ : d = 28) (h₅ : e = 25) : 
  ∃ (area : ℝ), area = 950 :=
by {
  sorry
}

end NUMINAMATH_GPT_pentagon_area_l1627_162775


namespace NUMINAMATH_GPT_find_number_l1627_162746

theorem find_number (x : ℝ) (h : 3034 - x / 200.4 = 3029) : x = 1002 :=
sorry

end NUMINAMATH_GPT_find_number_l1627_162746


namespace NUMINAMATH_GPT_largest_sum_product_l1627_162703

theorem largest_sum_product (p q : ℕ) (h1 : p * q = 100) (h2 : 0 < p) (h3 : 0 < q) : p + q ≤ 101 :=
sorry

end NUMINAMATH_GPT_largest_sum_product_l1627_162703


namespace NUMINAMATH_GPT_solve_for_x_l1627_162759

theorem solve_for_x :
  ∀ x : ℝ, 4 * x + 9 * x = 360 - 9 * (x - 4) → x = 18 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1627_162759


namespace NUMINAMATH_GPT_right_triangle_median_l1627_162702

variable (A B C M N : Type) [LinearOrder B] [LinearOrder C] [LinearOrder A] [LinearOrder M] [LinearOrder N]
variable (AC BC AM BN AB : ℝ)
variable (right_triangle : AC * AC + BC * BC = AB * AB)
variable (median_A : AC * AC + (1 / 4) * BC * BC = 81)
variable (median_B : BC * BC + (1 / 4) * AC * AC = 99)

theorem right_triangle_median :
  ∀ (AC BC AB : ℝ),
  (AC * AC + BC * BC = 144) → (AC * AC + BC * BC = AB * AB) → AB = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_right_triangle_median_l1627_162702


namespace NUMINAMATH_GPT_largest_multiple_of_15_less_than_neg_150_l1627_162719

theorem largest_multiple_of_15_less_than_neg_150 : ∃ m : ℤ, m % 15 = 0 ∧ m < -150 ∧ (∀ n : ℤ, n % 15 = 0 ∧ n < -150 → n ≤ m) ∧ m = -165 := sorry

end NUMINAMATH_GPT_largest_multiple_of_15_less_than_neg_150_l1627_162719


namespace NUMINAMATH_GPT_find_n_for_positive_root_l1627_162776

theorem find_n_for_positive_root :
  ∃ x : ℝ, x > 0 ∧ (∃ n : ℝ, (n / (x - 1) + 2 / (1 - x) = 1)) ↔ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_for_positive_root_l1627_162776


namespace NUMINAMATH_GPT_closest_perfect_square_multiple_of_4_l1627_162734

theorem closest_perfect_square_multiple_of_4 (n : ℕ) (h1 : ∃ k : ℕ, k^2 = n) (h2 : n % 4 = 0) : n = 324 := by
  -- Define 350 as the target
  let target := 350

  -- Conditions
  have cond1 : ∃ k : ℕ, k^2 = n := h1
  
  have cond2 : n % 4 = 0 := h2

  -- Check possible values meeting conditions
  by_cases h : n = 324
  { exact h }
  
  -- Exclude non-multiples of 4 and perfect squares further away from 350
  sorry

end NUMINAMATH_GPT_closest_perfect_square_multiple_of_4_l1627_162734


namespace NUMINAMATH_GPT_Kato_finishes_first_l1627_162731

-- Define constants and variables from the problem conditions
def Kato_total_pages : ℕ := 10
def Kato_lines_per_page : ℕ := 20
def Gizi_lines_per_page : ℕ := 30
def conversion_ratio : ℚ := 3 / 4
def initial_pages_written_by_Kato : ℕ := 4
def initial_additional_lines_by_Kato : ℚ := 2.5
def Kato_to_Gizi_writing_ratio : ℚ := 3 / 4

-- Calculate total lines in Kato's manuscript
def Kato_total_lines : ℕ := Kato_total_pages * Kato_lines_per_page

-- Convert Kato's lines to Gizi's format
def Kato_lines_in_Gizi_format : ℚ := Kato_total_lines * conversion_ratio

-- Calculate total pages Gizi needs to type
def Gizi_total_pages : ℚ := Kato_lines_in_Gizi_format / Gizi_lines_per_page

-- Calculate initial lines by Kato before Gizi starts typing
def initial_lines_by_Kato : ℚ := initial_pages_written_by_Kato * Kato_lines_per_page + initial_additional_lines_by_Kato

-- Lines Kato writes for every page Gizi types including setup time consideration
def additional_lines_by_Kato_per_Gizi_page : ℚ := Gizi_lines_per_page * Kato_to_Gizi_writing_ratio + initial_additional_lines_by_Kato / Gizi_total_pages

-- Calculate total lines Kato writes while Gizi finishes 5 pages
def final_lines_by_Kato : ℚ := additional_lines_by_Kato_per_Gizi_page * Gizi_total_pages

-- Remaining lines after initial setup for Kato
def remaining_lines_by_Kato_after_initial : ℚ := Kato_total_lines - initial_lines_by_Kato

-- Final proof statement
theorem Kato_finishes_first : final_lines_by_Kato ≥ remaining_lines_by_Kato_after_initial :=
by sorry

end NUMINAMATH_GPT_Kato_finishes_first_l1627_162731


namespace NUMINAMATH_GPT_fraction_white_surface_area_l1627_162710

-- Definitions of the given conditions
def cube_side_length : ℕ := 4
def small_cubes : ℕ := 64
def black_cubes : ℕ := 34
def white_cubes : ℕ := 30
def total_surface_area : ℕ := 6 * cube_side_length^2
def black_faces_exposed : ℕ := 32 
def white_faces_exposed : ℕ := total_surface_area - black_faces_exposed

-- The proof statement
theorem fraction_white_surface_area (cube_side_length_eq : cube_side_length = 4)
                                    (small_cubes_eq : small_cubes = 64)
                                    (black_cubes_eq : black_cubes = 34)
                                    (white_cubes_eq : white_cubes = 30)
                                    (black_faces_eq : black_faces_exposed = 32)
                                    (total_surface_area_eq : total_surface_area = 96)
                                    (white_faces_eq : white_faces_exposed = 64) : 
                                    (white_faces_exposed : ℚ) / (total_surface_area : ℚ) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_white_surface_area_l1627_162710


namespace NUMINAMATH_GPT_work_problem_l1627_162740

/--
Given:
1. A and B together can finish the work in 16 days.
2. B alone can finish the work in 48 days.
To Prove:
A alone can finish the work in 24 days.
-/
theorem work_problem (a b : ℕ)
  (h1 : a + b = 16)
  (h2 : b = 48) :
  a = 24 := 
sorry

end NUMINAMATH_GPT_work_problem_l1627_162740


namespace NUMINAMATH_GPT_unique_flavors_l1627_162757

noncomputable def distinctFlavors : Nat :=
  let redCandies := 5
  let greenCandies := 4
  let blueCandies := 2
  (90 - 15 - 18 - 30 + 3 + 5 + 6) / 3  -- Adjustments and consideration for equivalent ratios.
  
theorem unique_flavors :
  distinctFlavors = 11 :=
  by
    sorry

end NUMINAMATH_GPT_unique_flavors_l1627_162757


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1627_162749

theorem arithmetic_sequence_fifth_term:
  ∀ (a₁ aₙ : ℕ) (n : ℕ),
    n = 20 → a₁ = 2 → aₙ = 59 →
    ∃ d a₅, d = (59 - 2) / (20 - 1) ∧ a₅ = 2 + (5 - 1) * d ∧ a₅ = 14 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1627_162749


namespace NUMINAMATH_GPT_Jerry_age_l1627_162716

theorem Jerry_age (M J : ℕ) (h1 : M = 2 * J - 6) (h2 : M = 22) : J = 14 :=
by
  sorry

end NUMINAMATH_GPT_Jerry_age_l1627_162716


namespace NUMINAMATH_GPT_gcd_poly_l1627_162729

theorem gcd_poly {b : ℕ} (h : 1116 ∣ b) : Nat.gcd (b^2 + 11 * b + 36) (b + 6) = 6 :=
by
  sorry

end NUMINAMATH_GPT_gcd_poly_l1627_162729


namespace NUMINAMATH_GPT_min_S_min_S_values_range_of_c_l1627_162755

-- Part 1
theorem min_S (a b c : ℝ) (h : a + b + c = 1) : 
  2 * a^2 + 3 * b^2 + c^2 ≥ (6 / 11) :=
sorry

-- Part 1, finding exact values of a, b, c where minimum is reached
theorem min_S_values (a b c : ℝ) (h : a + b + c = 1) :
  2 * a^2 + 3 * b^2 + c^2 = (6 / 11) ↔ a = (3 / 11) ∧ b = (2 / 11) ∧ c = (6 / 11) :=
sorry
  
-- Part 2
theorem range_of_c (a b c : ℝ) (h1 : 2 * a^2 + 3 * b^2 + c^2 = 1) : 
  (1 / 11) ≤ c ∧ c ≤ 1 :=
sorry

end NUMINAMATH_GPT_min_S_min_S_values_range_of_c_l1627_162755


namespace NUMINAMATH_GPT_second_less_than_first_third_less_than_first_l1627_162726

variable (X : ℝ)

def first_number : ℝ := 0.70 * X
def second_number : ℝ := 0.63 * X
def third_number : ℝ := 0.59 * X

theorem second_less_than_first : 
  ((first_number X - second_number X) / first_number X * 100) = 10 :=
by
  sorry

theorem third_less_than_first : 
  ((third_number X - first_number X) / first_number X * 100) = -15.71 :=
by
  sorry

end NUMINAMATH_GPT_second_less_than_first_third_less_than_first_l1627_162726


namespace NUMINAMATH_GPT_lemonade_percentage_correct_l1627_162737
noncomputable def lemonade_percentage (first_lemonade first_carbon second_carbon mixture_carbon first_portion : ℝ) : ℝ :=
  100 - second_carbon

theorem lemonade_percentage_correct :
  let first_lemonade := 20
  let first_carbon := 80
  let second_carbon := 55
  let mixture_carbon := 60
  let first_portion := 19.99999999999997
  lemonade_percentage first_lemonade first_carbon second_carbon mixture_carbon first_portion = 45 :=
by
  -- Proof to be completed.
  sorry

end NUMINAMATH_GPT_lemonade_percentage_correct_l1627_162737


namespace NUMINAMATH_GPT_volume_pyramid_correct_l1627_162762

noncomputable def volume_of_regular_triangular_pyramid 
  (R : ℝ) (β : ℝ) (a : ℝ) : ℝ :=
  (a^3 * (Real.tan β)) / 24

theorem volume_pyramid_correct 
  (R : ℝ) (β : ℝ) (a : ℝ) : 
  volume_of_regular_triangular_pyramid R β a = (a^3 * (Real.tan β)) / 24 :=
sorry

end NUMINAMATH_GPT_volume_pyramid_correct_l1627_162762


namespace NUMINAMATH_GPT_find_interest_rate_l1627_162722

-- Define the given conditions
variables (P A t n CI : ℝ) (r : ℝ)

-- Suppose given conditions
variables (hP : P = 1200)
variables (hCI : CI = 240)
variables (hA : A = P + CI)
variables (ht : t = 1)
variables (hn : n = 1)

-- Define the statement to prove 
theorem find_interest_rate : (A = P * (1 + r / n)^(n * t)) → (r = 0.2) :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l1627_162722


namespace NUMINAMATH_GPT_solution_set_of_inequalities_l1627_162735

-- Define the conditions of the inequality system
def inequality1 (x : ℝ) : Prop := x - 2 ≥ -5
def inequality2 (x : ℝ) : Prop := 3 * x < x + 2

-- The statement to prove the solution set of the inequalities
theorem solution_set_of_inequalities :
  { x : ℝ | inequality1 x ∧ inequality2 x } = { x : ℝ | -3 ≤ x ∧ x < 1 } :=
  sorry

end NUMINAMATH_GPT_solution_set_of_inequalities_l1627_162735


namespace NUMINAMATH_GPT_train_crossing_time_l1627_162785

-- Definitions based on conditions from the problem
def length_of_train_and_platform := 900 -- in meters
def speed_km_per_hr := 108 -- in km/hr
def distance := 2 * length_of_train_and_platform -- distance to be covered
def speed_m_per_s := (speed_km_per_hr * 1000) / 3600 -- converted speed

-- Theorem stating the time to cross the platform is 60 seconds
theorem train_crossing_time : distance / speed_m_per_s = 60 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1627_162785


namespace NUMINAMATH_GPT_find_number_of_elements_l1627_162789

theorem find_number_of_elements (n S : ℕ) (h1 : S + 26 = 19 * n) (h2 : S + 76 = 24 * n) : n = 10 := 
sorry

end NUMINAMATH_GPT_find_number_of_elements_l1627_162789


namespace NUMINAMATH_GPT_number_of_dots_in_120_circles_l1627_162754

theorem number_of_dots_in_120_circles :
  ∃ n : ℕ, (n = 14) ∧ (∀ m : ℕ, m * (m + 1) / 2 + m ≤ 120 → m ≤ n) :=
by
  sorry

end NUMINAMATH_GPT_number_of_dots_in_120_circles_l1627_162754


namespace NUMINAMATH_GPT_train_length_l1627_162771

theorem train_length (bridge_length time_seconds speed_kmh : ℝ) (S : speed_kmh = 64) (T : time_seconds = 45) (B : bridge_length = 300) : 
  ∃ (train_length : ℝ), train_length = 500 := 
by
  -- Add your proof here 
  sorry

end NUMINAMATH_GPT_train_length_l1627_162771


namespace NUMINAMATH_GPT_total_distance_covered_l1627_162745

theorem total_distance_covered (up_speed down_speed up_time down_time : ℕ) (H1 : up_speed = 30) (H2 : down_speed = 50) (H3 : up_time = 5) (H4 : down_time = 5) :
  (up_speed * up_time + down_speed * down_time) = 400 := 
by
  sorry

end NUMINAMATH_GPT_total_distance_covered_l1627_162745


namespace NUMINAMATH_GPT_base_500_in_base_has_six_digits_l1627_162765

theorem base_500_in_base_has_six_digits (b : ℕ) : b^5 ≤ 500 ∧ 500 < b^6 ↔ b = 3 := 
by
  sorry

end NUMINAMATH_GPT_base_500_in_base_has_six_digits_l1627_162765


namespace NUMINAMATH_GPT_scientific_notation_400000000_l1627_162796

theorem scientific_notation_400000000 : 400000000 = 4 * 10^8 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_400000000_l1627_162796


namespace NUMINAMATH_GPT_triangle_area_problem_l1627_162750

theorem triangle_area_problem (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h_area : (∃ t : ℝ, t > 0 ∧ (2 * c * t + 3 * d * (12 / (2 * c)) = 12) ∧ (∃ s : ℝ, s > 0 ∧ 2 * c * (12 / (3 * d)) + 3 * d * s = 12)) ∧ 
    ((1 / 2) * (12 / (2 * c)) * (12 / (3 * d)) = 12)) : c * d = 1 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_problem_l1627_162750


namespace NUMINAMATH_GPT_sextuple_angle_terminal_side_on_xaxis_l1627_162701

-- Define angle and conditions
variable (α : ℝ)
variable (isPositiveAngle : 0 < α ∧ α < 360)
variable (sextupleAngleOnXAxis : ∃ k : ℕ, 6 * α = k * 360)

-- Prove the possible values of the angle
theorem sextuple_angle_terminal_side_on_xaxis :
  α = 60 ∨ α = 120 ∨ α = 180 ∨ α = 240 ∨ α = 300 :=
  sorry

end NUMINAMATH_GPT_sextuple_angle_terminal_side_on_xaxis_l1627_162701


namespace NUMINAMATH_GPT_sum_of_squares_99_in_distinct_ways_l1627_162760

theorem sum_of_squares_99_in_distinct_ways : 
  ∃ a b c d e f g h i j k l : ℕ, 
    (a^2 + b^2 + c^2 + d^2 = 99) ∧ (e^2 + f^2 + g^2 + h^2 = 99) ∧ (i^2 + j^2 + k^2 + l^2 = 99) ∧ 
    (a ≠ e ∨ b ≠ f ∨ c ≠ g ∨ d ≠ h) ∧ 
    (a ≠ i ∨ b ≠ j ∨ c ≠ k ∨ d ≠ l) ∧ 
    (i ≠ e ∨ j ≠ f ∨ k ≠ g ∨ l ≠ h) 
    :=
sorry

end NUMINAMATH_GPT_sum_of_squares_99_in_distinct_ways_l1627_162760


namespace NUMINAMATH_GPT_final_student_count_is_correct_l1627_162797

-- Define the initial conditions
def initial_students : ℕ := 11
def students_left_first_semester : ℕ := 6
def students_joined_first_semester : ℕ := 25
def additional_students_second_semester : ℕ := 15
def students_transferred_second_semester : ℕ := 3
def students_switched_class_second_semester : ℕ := 2

-- Define the final number of students to be proven
def final_number_of_students : ℕ := 
  initial_students - students_left_first_semester + students_joined_first_semester + 
  additional_students_second_semester - students_transferred_second_semester - students_switched_class_second_semester

-- The theorem we need to prove
theorem final_student_count_is_correct : final_number_of_students = 40 := by
  sorry

end NUMINAMATH_GPT_final_student_count_is_correct_l1627_162797


namespace NUMINAMATH_GPT_C_is_necessary_but_not_sufficient_for_A_l1627_162744

-- Define C, B, A to be logical propositions
variables (A B C : Prop)

-- The conditions given
axiom h1 : A → B
axiom h2 : ¬ (B → A)
axiom h3 : B ↔ C

-- The conclusion: Prove that C is a necessary but not sufficient condition for A
theorem C_is_necessary_but_not_sufficient_for_A : (A → C) ∧ ¬ (C → A) :=
by
  sorry

end NUMINAMATH_GPT_C_is_necessary_but_not_sufficient_for_A_l1627_162744


namespace NUMINAMATH_GPT_minimum_passed_l1627_162712

def total_participants : Nat := 100
def num_questions : Nat := 10
def correct_answers : List Nat := [93, 90, 86, 91, 80, 83, 72, 75, 78, 59]
def passing_criteria : Nat := 6

theorem minimum_passed (total_participants : ℕ) (num_questions : ℕ) (correct_answers : List ℕ) (passing_criteria : ℕ) :
  100 = total_participants → 10 = num_questions → correct_answers = [93, 90, 86, 91, 80, 83, 72, 75, 78, 59] →
  passing_criteria = 6 → 
  ∃ p : ℕ, p = 62 := 
by
  sorry

end NUMINAMATH_GPT_minimum_passed_l1627_162712


namespace NUMINAMATH_GPT_matrix_equation_l1627_162723

open Matrix

-- Define matrix B
def B : Matrix (Fin 2) (Fin 2) (ℤ) :=
  ![![1, -2], 
    ![-3, 5]]

-- The proof problem statement in Lean 4
theorem matrix_equation (r s : ℤ) (I : Matrix (Fin 2) (Fin 2) (ℤ))  [DecidableEq (ℤ)] [Fintype (Fin 2)] : 
  I = 1 ∧ B ^ 6 = r • B + s • I ↔ r = 2999 ∧ s = 2520 := by {
    sorry
}

end NUMINAMATH_GPT_matrix_equation_l1627_162723


namespace NUMINAMATH_GPT_mod_equiv_inverse_sum_l1627_162730

theorem mod_equiv_inverse_sum :
  (3^15 + 3^14 + 3^13 + 3^12) % 17 = 5 :=
by sorry

end NUMINAMATH_GPT_mod_equiv_inverse_sum_l1627_162730


namespace NUMINAMATH_GPT_base9_number_perfect_square_l1627_162779

theorem base9_number_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : 0 ≤ d ∧ d ≤ 8) (n : ℕ) 
  (h3 : n = 729 * a + 81 * b + 45 + d) (h4 : ∃ k : ℕ, k * k = n) : d = 0 := 
sorry

end NUMINAMATH_GPT_base9_number_perfect_square_l1627_162779


namespace NUMINAMATH_GPT_total_rainfall_january_l1627_162714

theorem total_rainfall_january (R1 R2 T : ℝ) (h1 : R2 = 1.5 * R1) (h2 : R2 = 21) : T = 35 :=
by 
  let R1 := 14
  let R2 := 21
  let T := R1 + R2
  sorry

end NUMINAMATH_GPT_total_rainfall_january_l1627_162714


namespace NUMINAMATH_GPT_bn_is_arithmetic_seq_an_general_term_l1627_162742

def seq_an (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ ∀ n, (a (n + 1) - 1) * (a n - 1) = 3 * (a n - a (n + 1))

def seq_bn (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
∀ n, b n = 1 / (a n - 1)

theorem bn_is_arithmetic_seq (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : seq_an a) (h2 : seq_bn a b) : 
∀ n, b (n + 1) - b n = 1 / 3 :=
sorry

theorem an_general_term (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : seq_an a) (h2 : seq_bn a b) : 
∀ n, a n = (n + 5) / (n + 2) :=
sorry

end NUMINAMATH_GPT_bn_is_arithmetic_seq_an_general_term_l1627_162742


namespace NUMINAMATH_GPT_smallest_geometric_third_term_l1627_162791

theorem smallest_geometric_third_term (d : ℝ) (a₁ a₂ a₃ g₁ g₂ g₃ : ℝ) 
  (h_AP : a₁ = 5 ∧ a₂ = 5 + d ∧ a₃ = 5 + 2 * d)
  (h_GP : g₁ = a₁ ∧ g₂ = a₂ + 3 ∧ g₃ = a₃ + 15)
  (h_geom : (g₂)^2 = g₁ * g₃) : g₃ = -4 := 
by
  -- We would provide the proof here.
  sorry

end NUMINAMATH_GPT_smallest_geometric_third_term_l1627_162791


namespace NUMINAMATH_GPT_people_with_fewer_than_seven_cards_l1627_162774

theorem people_with_fewer_than_seven_cards (total_cards : ℕ) (num_people : ℕ) (cards_per_person : ℕ) (extra_cards : ℕ)
  (h1 : total_cards = 52) (h2 : num_people = 8) (h3 : total_cards = num_people * cards_per_person + extra_cards) (h4 : extra_cards < num_people) :
  ∃ fewer_than_seven : ℕ, num_people - extra_cards = fewer_than_seven :=
by
  have remainder := (52 % 8)
  have cards_per_person := (52 / 8)
  have number_fewer_than_seven := num_people - remainder
  existsi number_fewer_than_seven
  sorry

end NUMINAMATH_GPT_people_with_fewer_than_seven_cards_l1627_162774


namespace NUMINAMATH_GPT_speeds_and_time_l1627_162704

theorem speeds_and_time (x s : ℕ) (t : ℝ)
  (h1 : ∀ {t : ℝ}, t = 2 → x * t > s * t + 24)
  (h2 : ∀ {t : ℝ}, t = 0.5 → x * t = 8) :
  x = 16 ∧ s = 4 ∧ t = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_speeds_and_time_l1627_162704


namespace NUMINAMATH_GPT_sand_needed_l1627_162732

def area_rectangular_patch : ℕ := 6 * 7
def area_square_patch : ℕ := 5 * 5
def sand_per_square_inch : ℕ := 3

theorem sand_needed : area_rectangular_patch + area_square_patch * sand_per_square_inch = 201 := sorry

end NUMINAMATH_GPT_sand_needed_l1627_162732


namespace NUMINAMATH_GPT_same_gender_probability_l1627_162717

-- Define the total number of teachers in School A and their gender distribution.
def schoolA_teachers : Nat := 3
def schoolA_males : Nat := 2
def schoolA_females : Nat := 1

-- Define the total number of teachers in School B and their gender distribution.
def schoolB_teachers : Nat := 3
def schoolB_males : Nat := 1
def schoolB_females : Nat := 2

-- Calculate the probability of selecting two teachers of the same gender.
theorem same_gender_probability :
  (schoolA_males * schoolB_males + schoolA_females * schoolB_females) / (schoolA_teachers * schoolB_teachers) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_same_gender_probability_l1627_162717


namespace NUMINAMATH_GPT_number_of_salads_bought_l1627_162753

variable (hot_dogs_cost : ℝ := 5 * 1.50)
variable (initial_money : ℝ := 2 * 10)
variable (change_given_back : ℝ := 5)
variable (total_spent : ℝ := initial_money - change_given_back)
variable (salad_cost : ℝ := 2.50)

theorem number_of_salads_bought : (total_spent - hot_dogs_cost) / salad_cost = 3 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_salads_bought_l1627_162753


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1627_162706

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℝ), x^2 ≥ 0) ↔ ∃ (x : ℝ), x^2 < 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1627_162706


namespace NUMINAMATH_GPT_no_real_solution_l1627_162782

theorem no_real_solution :
    ∀ x : ℝ, (5 * x^2 - 3 * x + 2) / (x + 2) ≠ 2 * x - 3 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_real_solution_l1627_162782


namespace NUMINAMATH_GPT_find_other_integer_l1627_162764

theorem find_other_integer (x y : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 1 ≤ y ∧ y ≤ 9) (h3 : 7 * x + y = 68) : y = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_other_integer_l1627_162764


namespace NUMINAMATH_GPT_find_sum_l1627_162767

noncomputable def principal_sum (P R : ℝ) := 
  let I := (P * R * 10) / 100
  let new_I := (P * (R + 5) * 10) / 100
  I + 600 = new_I

theorem find_sum (P R : ℝ) (h : principal_sum P R) : P = 1200 := 
  sorry

end NUMINAMATH_GPT_find_sum_l1627_162767


namespace NUMINAMATH_GPT_triangle_neg3_4_l1627_162741

def triangle (a b : ℚ) : ℚ := -a + b

theorem triangle_neg3_4 : triangle (-3) 4 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_neg3_4_l1627_162741


namespace NUMINAMATH_GPT_Hezekiah_age_l1627_162707

variable (H : ℕ)
variable (R : ℕ) -- Ryanne's age

-- Defining the conditions
def condition1 : Prop := R = H + 7
def condition2 : Prop := H + R = 15

-- The main theorem we want to prove
theorem Hezekiah_age : condition1 H R → condition2 H R → H = 4 :=
by  -- proof will be here
  sorry

end NUMINAMATH_GPT_Hezekiah_age_l1627_162707


namespace NUMINAMATH_GPT_seats_not_occupied_l1627_162700

theorem seats_not_occupied (seats_per_row : ℕ) (rows : ℕ) (fraction_allowed : ℚ) (total_seats : ℕ) (allowed_seats_per_row : ℕ) (allowed_total : ℕ) (unoccupied_seats : ℕ) :
  seats_per_row = 8 →
  rows = 12 →
  fraction_allowed = 3 / 4 →
  total_seats = seats_per_row * rows →
  allowed_seats_per_row = seats_per_row * fraction_allowed →
  allowed_total = allowed_seats_per_row * rows →
  unoccupied_seats = total_seats - allowed_total →
  unoccupied_seats = 24 :=
by sorry

end NUMINAMATH_GPT_seats_not_occupied_l1627_162700


namespace NUMINAMATH_GPT_Iesha_num_books_about_school_l1627_162772

theorem Iesha_num_books_about_school (total_books sports_books : ℕ) (h1 : total_books = 58) (h2 : sports_books = 39) : total_books - sports_books = 19 :=
by
  sorry

end NUMINAMATH_GPT_Iesha_num_books_about_school_l1627_162772


namespace NUMINAMATH_GPT_milk_water_ratio_l1627_162748

theorem milk_water_ratio (total_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ) (added_water : ℕ)
  (h₁ : total_volume = 45) (h₂ : initial_milk_ratio = 4) (h₃ : initial_water_ratio = 1) (h₄ : added_water = 9) :
  (36 : ℕ) / (18 : ℕ) = 2 :=
by sorry

end NUMINAMATH_GPT_milk_water_ratio_l1627_162748


namespace NUMINAMATH_GPT_find_cookies_on_second_plate_l1627_162752

theorem find_cookies_on_second_plate (a : ℕ → ℕ) :
  (a 1 = 5) ∧ (a 3 = 10) ∧ (a 4 = 14) ∧ (a 5 = 19) ∧ (a 6 = 25) ∧
  (∀ n, a (n + 2) - a (n + 1) = if (n + 1) % 2 = 0 then 5 else 4) →
  a 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_cookies_on_second_plate_l1627_162752


namespace NUMINAMATH_GPT_average_weight_l1627_162743

theorem average_weight :
  ∀ (A B C : ℝ),
    (A + B = 84) → 
    (B + C = 86) → 
    (B = 35) → 
    (A + B + C) / 3 = 45 :=
by
  intros A B C hab hbc hb
  -- proof omitted
  sorry

end NUMINAMATH_GPT_average_weight_l1627_162743


namespace NUMINAMATH_GPT_markup_rate_l1627_162736

theorem markup_rate (S : ℝ) (C : ℝ) (hS : S = 8) (h1 : 0.20 * S = 0.10 * S + (S - C)) :
  ((S - C) / C) * 100 = 42.857 :=
by
  -- Assume given conditions and reasoning to conclude the proof
  sorry

end NUMINAMATH_GPT_markup_rate_l1627_162736


namespace NUMINAMATH_GPT_smallest_b_for_quadratic_factorization_l1627_162768

theorem smallest_b_for_quadratic_factorization : ∃ (b : ℕ), 
  (∀ r s : ℤ, (r * s = 4032) ∧ (r + s = b) → b ≥ 127) ∧ 
  (∃ r s : ℤ, (r * s = 4032) ∧ (r + s = b) ∧ (b = 127))
:= sorry

end NUMINAMATH_GPT_smallest_b_for_quadratic_factorization_l1627_162768


namespace NUMINAMATH_GPT_ccamathbonanza_2016_2_1_l1627_162786

-- Definitions of the speeds of the runners
def bhairav_speed := 28 -- in miles per hour
def daniel_speed := 15 -- in miles per hour
def tristan_speed := 10 -- in miles per hour

-- Distance of the race
def race_distance := 15 -- in miles

-- Time conversion from hours to minutes
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

-- Time taken by each runner to complete the race (in hours)
def time_bhairav := race_distance / bhairav_speed
def time_daniel := race_distance / daniel_speed
def time_tristan := race_distance / tristan_speed

-- Time taken by each runner to complete the race (in minutes)
def time_bhairav_minutes := hours_to_minutes time_bhairav
def time_daniel_minutes := hours_to_minutes time_daniel
def time_tristan_minutes := hours_to_minutes time_tristan

-- Time differences between consecutive runners' finishes (in minutes)
def time_diff_bhairav_daniel := time_daniel_minutes - time_bhairav_minutes
def time_diff_daniel_tristan := time_tristan_minutes - time_daniel_minutes

-- Greatest length of time between consecutive runners' finishes
def greatest_time_diff := max time_diff_bhairav_daniel time_diff_daniel_tristan

-- The theorem we need to prove
theorem ccamathbonanza_2016_2_1 : greatest_time_diff = 30 := by
  sorry

end NUMINAMATH_GPT_ccamathbonanza_2016_2_1_l1627_162786


namespace NUMINAMATH_GPT_fifth_inequality_nth_inequality_solve_given_inequality_l1627_162769

theorem fifth_inequality :
  ∀ x, 1 < x ∧ x < 2 → (x + 2 / x < 3) →
  ∀ x, 3 < x ∧ x < 4 → (x + 12 / x < 7) →
  ∀ x, 5 < x ∧ x < 6 → (x + 30 / x < 11) →
  (x + 90 / x < 19) := by
  sorry

theorem nth_inequality (n : ℕ) :
  ∀ x, (2 * n - 1 < x ∧ x < 2 * n) →
  (x + 2 * n * (2 * n - 1) / x < 4 * n - 1) := by
  sorry

theorem solve_given_inequality (a : ℕ) (x : ℝ) (h_a_pos: 0 < a) :
  x + 12 * a / (x + 1) < 4 * a + 2 →
  (2 < x ∧ x < 4 * a - 1) := by
  sorry

end NUMINAMATH_GPT_fifth_inequality_nth_inequality_solve_given_inequality_l1627_162769


namespace NUMINAMATH_GPT_triangular_square_is_triangular_l1627_162790

-- Definition of a triangular number
def is_triang_number (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * (x + 1) / 2

-- The main theorem statement
theorem triangular_square_is_triangular :
  ∃ x : ℕ, 
    is_triang_number x ∧ 
    is_triang_number (x * x) :=
sorry

end NUMINAMATH_GPT_triangular_square_is_triangular_l1627_162790


namespace NUMINAMATH_GPT_average_of_five_digits_l1627_162739

theorem average_of_five_digits 
  (S : ℝ)
  (S3 : ℝ)
  (h_avg8 : S / 8 = 20)
  (h_avg3 : S3 / 3 = 33.333333333333336) :
  (S - S3) / 5 = 12 := 
by
  sorry

end NUMINAMATH_GPT_average_of_five_digits_l1627_162739


namespace NUMINAMATH_GPT_number_of_employees_l1627_162705

def fixed_time_coffee : ℕ := 5
def time_per_status_update : ℕ := 2
def time_per_payroll_update : ℕ := 3
def total_morning_routine : ℕ := 50

def time_per_employee : ℕ := time_per_status_update + time_per_payroll_update
def time_spent_on_employees : ℕ := total_morning_routine - fixed_time_coffee

theorem number_of_employees : (time_spent_on_employees / time_per_employee) = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_employees_l1627_162705


namespace NUMINAMATH_GPT_geometric_sequence_Sn_geometric_sequence_Sn_l1627_162770

noncomputable def Sn (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1/3 then (27/2) - (1/2) * 3^(n - 3)
  else if q = 3 then (3^n - 1) / 2
  else 0

theorem geometric_sequence_Sn (a1 : ℝ) (n : ℕ) (h1 : a1 * (1/3) = 3)
  (h2 : a1 + a1 * (1/3)^2 = 10) : 
  Sn a1 (1/3) n = (27/2) - (1/2) * 3^(n - 3) :=
by
  sorry

theorem geometric_sequence_Sn' (a1 : ℝ) (n : ℕ) (h1 : a1 * 3 = 3) 
  (h2 : a1 + a1 * 3^2 = 10) : 
  Sn a1 3 n = (3^n - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_Sn_geometric_sequence_Sn_l1627_162770


namespace NUMINAMATH_GPT_area_PST_correct_l1627_162784

noncomputable def area_of_triangle_PST : ℚ :=
  let P : ℚ × ℚ := (0, 0)
  let Q : ℚ × ℚ := (4, 0)
  let R : ℚ × ℚ := (0, 4)
  let S : ℚ × ℚ := (0, 2)
  let T : ℚ × ℚ := (8 / 3, 4 / 3)
  1 / 2 * (|P.1 * (S.2 - T.2) + S.1 * (T.2 - P.2) + T.1 * (P.2 - S.2)|)

theorem area_PST_correct : area_of_triangle_PST = 8 / 3 := sorry

end NUMINAMATH_GPT_area_PST_correct_l1627_162784


namespace NUMINAMATH_GPT_distinct_sums_l1627_162758

theorem distinct_sums (n : ℕ) (a : Fin n → ℕ) (h_distinct : Function.Injective a) :
  ∃ S : Finset ℕ, S.card ≥ n * (n + 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_distinct_sums_l1627_162758


namespace NUMINAMATH_GPT_regular_price_one_bag_l1627_162715

theorem regular_price_one_bag (p : ℕ) (h : 3 * p + 5 = 305) : p = 100 :=
by
  sorry

end NUMINAMATH_GPT_regular_price_one_bag_l1627_162715


namespace NUMINAMATH_GPT_balloons_left_l1627_162711

def total_balloons (r w g c: Nat) : Nat := r + w + g + c

def num_friends : Nat := 10

theorem balloons_left (r w g c : Nat) (total := total_balloons r w g c) (h_r : r = 24) (h_w : w = 38) (h_g : g = 68) (h_c : c = 75) :
  total % num_friends = 5 := by
  sorry

end NUMINAMATH_GPT_balloons_left_l1627_162711


namespace NUMINAMATH_GPT_no_solutions_to_equation_l1627_162792

theorem no_solutions_to_equation :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x ^ 2 - 2 * y ^ 2 = 5 := by
  sorry

end NUMINAMATH_GPT_no_solutions_to_equation_l1627_162792


namespace NUMINAMATH_GPT_proposition_C_is_true_l1627_162713

theorem proposition_C_is_true :
  (∀ θ : ℝ, 90 < θ ∧ θ < 180 → θ > 90) :=
by
  sorry

end NUMINAMATH_GPT_proposition_C_is_true_l1627_162713


namespace NUMINAMATH_GPT_negation_of_proposition_l1627_162763

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x)) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1627_162763


namespace NUMINAMATH_GPT_trig_inequality_l1627_162727

open Real

theorem trig_inequality (x : ℝ) (n m : ℕ) (hx : 0 < x ∧ x < π / 2) (hnm : n > m) : 
  2 * abs (sin x ^ n - cos x ^ n) ≤ 3 * abs (sin x ^ m - cos x ^ m) := 
sorry

end NUMINAMATH_GPT_trig_inequality_l1627_162727


namespace NUMINAMATH_GPT_trapezium_shorter_side_l1627_162708

theorem trapezium_shorter_side (a b h : ℝ) (H1 : a = 10) (H2 : b = 18) (H3 : h = 10.00001) : a = 10 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_shorter_side_l1627_162708


namespace NUMINAMATH_GPT_cos_double_beta_eq_24_over_25_l1627_162728

theorem cos_double_beta_eq_24_over_25
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 3 / 5)
  (h2 : Real.cos (α + β) = -3 / 5)
  (h3 : α - β ∈ Set.Ioo (π / 2) π)
  (h4 : α + β ∈ Set.Ioo (π / 2) π) :
  Real.cos (2 * β) = 24 / 25 := sorry

end NUMINAMATH_GPT_cos_double_beta_eq_24_over_25_l1627_162728


namespace NUMINAMATH_GPT_train_speed_conversion_l1627_162766

/-- Define a function to convert kmph to m/s --/
def kmph_to_ms (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

/-- Theorem stating that 72 kmph is equivalent to 20 m/s --/
theorem train_speed_conversion : kmph_to_ms 72 = 20 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_conversion_l1627_162766


namespace NUMINAMATH_GPT_total_bronze_needed_l1627_162738

theorem total_bronze_needed (w1 w2 w3 : ℕ) (h1 : w1 = 50) (h2 : w2 = 2 * w1) (h3 : w3 = 4 * w2) : w1 + w2 + w3 = 550 :=
by
  -- We'll complete the proof later
  sorry

end NUMINAMATH_GPT_total_bronze_needed_l1627_162738


namespace NUMINAMATH_GPT_cos_double_angle_identity_l1627_162718

theorem cos_double_angle_identity (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_identity_l1627_162718


namespace NUMINAMATH_GPT_new_mixture_concentration_l1627_162793

def vessel1_capacity : ℝ := 2
def vessel1_concentration : ℝ := 0.30
def vessel2_capacity : ℝ := 6
def vessel2_concentration : ℝ := 0.40
def total_volume : ℝ := 8
def expected_concentration : ℝ := 37.5

theorem new_mixture_concentration :
  ((vessel1_capacity * vessel1_concentration + vessel2_capacity * vessel2_concentration) / total_volume) * 100 = expected_concentration :=
by
  sorry

end NUMINAMATH_GPT_new_mixture_concentration_l1627_162793


namespace NUMINAMATH_GPT_average_percentage_decrease_l1627_162725

theorem average_percentage_decrease : 
  ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ ((2000 * (1 - x)^2 = 1280) ↔ (x = 0.18)) :=
by
  sorry

end NUMINAMATH_GPT_average_percentage_decrease_l1627_162725


namespace NUMINAMATH_GPT_triangle_inequality_l1627_162781

variables {a b c h : ℝ}
variable {n : ℕ}

theorem triangle_inequality
  (h_triangle : a^2 + b^2 = c^2)
  (h_height : a * b = c * h)
  (h_cond : a + b < c + h)
  (h_pos_n : n > 0) :
  a^n + b^n < c^n + h^n :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1627_162781


namespace NUMINAMATH_GPT_matrix_eq_sum_35_l1627_162788

theorem matrix_eq_sum_35 (a b c d : ℤ) (h1 : 2 * a = 14 * a - 15 * b)
  (h2 : 2 * b = 9 * a - 10 * b)
  (h3 : 3 * c = 14 * c - 15 * d)
  (h4 : 3 * d = 9 * c - 10 * d) :
  a + b + c + d = 35 :=
sorry

end NUMINAMATH_GPT_matrix_eq_sum_35_l1627_162788


namespace NUMINAMATH_GPT_koala_fiber_consumption_l1627_162761

theorem koala_fiber_consumption
  (absorbed_fiber : ℝ) (total_fiber : ℝ) 
  (h1 : absorbed_fiber = 0.40 * total_fiber)
  (h2 : absorbed_fiber = 12) :
  total_fiber = 30 := 
by
  sorry

end NUMINAMATH_GPT_koala_fiber_consumption_l1627_162761


namespace NUMINAMATH_GPT_find_natural_numbers_with_integer_roots_l1627_162794

theorem find_natural_numbers_with_integer_roots :
  ∃ (p q : ℕ), 
    (∀ x : ℤ, x * x - (p * q) * x + (p + q) = 0 → ∃ (x1 x2 : ℤ), x = x1 ∧ x = x2 ∧ x1 + x2 = p * q ∧ x1 * x2 = p + q) ↔
    ((p = 1 ∧ q = 5) ∨ (p = 5 ∧ q = 1) ∨ (p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
-- proof skipped
sorry

end NUMINAMATH_GPT_find_natural_numbers_with_integer_roots_l1627_162794


namespace NUMINAMATH_GPT_y_not_multiple_of_7_y_not_multiple_of_14_y_not_multiple_of_21_y_not_multiple_of_28_l1627_162733

def y : ℕ := 42 + 98 + 210 + 333 + 175 + 28

theorem y_not_multiple_of_7 : ¬ (7 ∣ y) := sorry
theorem y_not_multiple_of_14 : ¬ (14 ∣ y) := sorry
theorem y_not_multiple_of_21 : ¬ (21 ∣ y) := sorry
theorem y_not_multiple_of_28 : ¬ (28 ∣ y) := sorry

end NUMINAMATH_GPT_y_not_multiple_of_7_y_not_multiple_of_14_y_not_multiple_of_21_y_not_multiple_of_28_l1627_162733


namespace NUMINAMATH_GPT_find_g50_l1627_162721

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g50 (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x * y) = y * g x)
  (h1 : g 1 = 10) : g 50 = 50 * 10 :=
by
  -- The proof sketch here; the detailed proof is omitted
  sorry

end NUMINAMATH_GPT_find_g50_l1627_162721


namespace NUMINAMATH_GPT_range_of_a_l1627_162709

theorem range_of_a (a : ℝ) : (-1/Real.exp 1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1/Real.exp 1) :=
  sorry

end NUMINAMATH_GPT_range_of_a_l1627_162709


namespace NUMINAMATH_GPT_area_of_cos_integral_l1627_162780

theorem area_of_cos_integral : 
  (∫ x in (0:ℝ)..(3 * Real.pi / 2), |Real.cos x|) = 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_cos_integral_l1627_162780


namespace NUMINAMATH_GPT_entry_exit_options_l1627_162787

theorem entry_exit_options :
  let south_gates := 4
  let north_gates := 3
  let total_gates := south_gates + north_gates
  (total_gates * total_gates = 49) :=
by {
  let south_gates := 4
  let north_gates := 3
  let total_gates := south_gates + north_gates
  show total_gates * total_gates = 49
  sorry
}

end NUMINAMATH_GPT_entry_exit_options_l1627_162787


namespace NUMINAMATH_GPT_edward_made_in_summer_l1627_162777

def edward_made_in_spring := 2
def cost_of_supplies := 5
def money_left_over := 24

theorem edward_made_in_summer : edward_made_in_spring + x - cost_of_supplies = money_left_over → x = 27 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_edward_made_in_summer_l1627_162777


namespace NUMINAMATH_GPT_students_at_end_l1627_162795

def initial_students := 11
def students_left := 6
def new_students := 42

theorem students_at_end (init : ℕ := initial_students) (left : ℕ := students_left) (new : ℕ := new_students) :
    (init - left + new) = 47 := 
by
  sorry

end NUMINAMATH_GPT_students_at_end_l1627_162795
