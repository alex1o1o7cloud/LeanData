import Mathlib

namespace NUMINAMATH_GPT_roots_sum_product_l708_70848

theorem roots_sum_product (p q : ℝ) (h_sum : p / 3 = 8) (h_prod : q / 3 = 12) : p + q = 60 := 
by 
  sorry

end NUMINAMATH_GPT_roots_sum_product_l708_70848


namespace NUMINAMATH_GPT_height_of_box_l708_70801

-- Define box dimensions
def box_length := 6
def box_width := 6

-- Define spherical radii
def radius_large := 3
def radius_small := 2

-- Define coordinates
def box_volume (h : ℝ) : Prop :=
  ∃ (z : ℝ), z = 2 + Real.sqrt 23 ∧ 
  z + radius_large = h

theorem height_of_box (h : ℝ) : box_volume h ↔ h = 5 + Real.sqrt 23 := by
  sorry

end NUMINAMATH_GPT_height_of_box_l708_70801


namespace NUMINAMATH_GPT_cost_prices_l708_70856

theorem cost_prices (C_t C_c C_b : ℝ)
  (h1 : 2 * C_t = 1000)
  (h2 : 1.75 * C_c = 1750)
  (h3 : 0.75 * C_b = 1500) :
  C_t = 500 ∧ C_c = 1000 ∧ C_b = 2000 :=
by
  sorry

end NUMINAMATH_GPT_cost_prices_l708_70856


namespace NUMINAMATH_GPT_students_helped_on_fourth_day_l708_70873

theorem students_helped_on_fourth_day (total_books : ℕ) (books_per_student : ℕ)
  (day1_students : ℕ) (day2_students : ℕ) (day3_students : ℕ)
  (H1 : total_books = 120) (H2 : books_per_student = 5)
  (H3 : day1_students = 4) (H4 : day2_students = 5) (H5 : day3_students = 6) :
  (total_books - (day1_students * books_per_student + day2_students * books_per_student + day3_students * books_per_student)) / books_per_student = 9 :=
by
  sorry

end NUMINAMATH_GPT_students_helped_on_fourth_day_l708_70873


namespace NUMINAMATH_GPT_probability_face_then_number_l708_70861

theorem probability_face_then_number :
  let total_cards := 52
  let total_ways_to_draw_two := total_cards * (total_cards - 1)
  let face_cards := 3 * 4
  let number_cards := 9 * 4
  let probability := (face_cards * number_cards) / total_ways_to_draw_two
  probability = 8 / 49 :=
by
  sorry

end NUMINAMATH_GPT_probability_face_then_number_l708_70861


namespace NUMINAMATH_GPT_greatest_two_digit_with_product_9_l708_70867

theorem greatest_two_digit_with_product_9 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ a b : ℕ, n = 10 * a + b ∧ a * b = 9) ∧ (∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ (∃ c d : ℕ, m = 10 * c + d ∧ c * d = 9) → m ≤ 91) :=
by
  sorry

end NUMINAMATH_GPT_greatest_two_digit_with_product_9_l708_70867


namespace NUMINAMATH_GPT_slope_of_line_l708_70864

theorem slope_of_line : 
  let A := Real.sin (Real.pi / 6)
  let B := Real.cos (5 * Real.pi / 6)
  (- A / B) = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l708_70864


namespace NUMINAMATH_GPT_area_of_quadrilateral_APQC_l708_70883

-- Define the geometric entities and conditions
structure RightTriangle (a b c : ℝ) :=
  (hypotenuse_eq: c = Real.sqrt (a ^ 2 + b ^ 2))

-- Triangles PAQ and PQC are right triangles with given sides
def PAQ := RightTriangle 9 12 (Real.sqrt (9^2 + 12^2))
def PQC := RightTriangle 12 9 (Real.sqrt (15^2 - 12^2))

-- Prove that the area of quadrilateral APQC is 108 square units
theorem area_of_quadrilateral_APQC :
  let area_PAQ := 1/2 * 9 * 12
  let area_PQC := 1/2 * 12 * 9
  area_PAQ + area_PQC = 108 :=
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_APQC_l708_70883


namespace NUMINAMATH_GPT_integer_values_of_a_l708_70845

-- Define the polynomial P(x)
def P (a x : ℤ) : ℤ := x^3 + a * x^2 + 3 * x + 7

-- Define the main theorem
theorem integer_values_of_a (a x : ℤ) (hx : P a x = 0) (hx_is_int : x = 1 ∨ x = -1 ∨ x = 7 ∨ x = -7) :
  a = -11 ∨ a = -3 :=
by
  sorry

end NUMINAMATH_GPT_integer_values_of_a_l708_70845


namespace NUMINAMATH_GPT_area_equivalence_l708_70840

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def angle_bisector (A B C : Point) : Point := sorry
noncomputable def arc_midpoint (A B C : Point) : Point := sorry
noncomputable def is_concyclic (P Q R S : Point) : Prop := sorry
noncomputable def area_of_quad (A B C D : Point) : ℝ := sorry
noncomputable def area_of_pent (A B C D E : Point) : ℝ := sorry

theorem area_equivalence (A B C I X Y M : Point)
  (h1 : I = incenter A B C)
  (h2 : X = angle_bisector B A C)
  (h3 : Y = angle_bisector C A B)
  (h4 : M = arc_midpoint A B C)
  (h5 : is_concyclic M X I Y) :
  area_of_quad M B I C = area_of_pent B X I Y C := 
sorry

end NUMINAMATH_GPT_area_equivalence_l708_70840


namespace NUMINAMATH_GPT_additional_ice_cubes_made_l708_70868

def original_ice_cubes : ℕ := 2
def total_ice_cubes : ℕ := 9

theorem additional_ice_cubes_made :
  (total_ice_cubes - original_ice_cubes) = 7 :=
by
  sorry

end NUMINAMATH_GPT_additional_ice_cubes_made_l708_70868


namespace NUMINAMATH_GPT_regular_price_of_ticket_l708_70837

theorem regular_price_of_ticket (P : Real) (discount_paid : Real) (discount_rate : Real) (paid : Real)
  (h_discount_rate : discount_rate = 0.40)
  (h_paid : paid = 9)
  (h_discount_paid : discount_paid = P * (1 - discount_rate))
  (h_paid_eq_discount_paid : paid = discount_paid) :
  P = 15 := 
by
  sorry

end NUMINAMATH_GPT_regular_price_of_ticket_l708_70837


namespace NUMINAMATH_GPT_trig_identity_l708_70872

open Real

theorem trig_identity :
  3.4173 * sin (2 * pi / 17) + sin (4 * pi / 17) - sin (6 * pi / 17) - (1/2) * sin (8 * pi / 17) =
  8 * (sin (2 * pi / 17))^3 * (cos (pi / 17))^2 :=
by sorry

end NUMINAMATH_GPT_trig_identity_l708_70872


namespace NUMINAMATH_GPT_tracy_initial_candies_l708_70855

theorem tracy_initial_candies (x y : ℕ) (h₁ : x = 108) (h₂ : 2 ≤ y ∧ y ≤ 6) : 
  let remaining_after_eating := (3 / 4) * x 
  let remaining_after_giving := (2 / 3) * remaining_after_eating
  let remaining_after_mom := remaining_after_giving - 40
  remaining_after_mom - y = 10 :=
by 
  sorry

end NUMINAMATH_GPT_tracy_initial_candies_l708_70855


namespace NUMINAMATH_GPT_string_length_l708_70889

def cylindrical_post_circumference : ℝ := 6
def cylindrical_post_height : ℝ := 15
def loops : ℝ := 3

theorem string_length :
  (cylindrical_post_height / loops)^2 + cylindrical_post_circumference^2 = 61 → 
  loops * Real.sqrt 61 = 3 * Real.sqrt 61 :=
by
  sorry

end NUMINAMATH_GPT_string_length_l708_70889


namespace NUMINAMATH_GPT_least_value_expression_l708_70806

theorem least_value_expression (x : ℝ) (h : x < -2) :
  2 * x < x ∧ 2 * x < x + 2 ∧ 2 * x < (1 / 2) * x ∧ 2 * x < x - 2 :=
by
  sorry

end NUMINAMATH_GPT_least_value_expression_l708_70806


namespace NUMINAMATH_GPT_solve_for_a4b4_l708_70846

theorem solve_for_a4b4 (
    a1 a2 a3 a4 b1 b2 b3 b4 : ℝ
) (h1 : a1 * b1 + a2 * b3 = 1) 
  (h2 : a1 * b2 + a2 * b4 = 0) 
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) : 
  a4 * b4 = -6 :=
sorry

end NUMINAMATH_GPT_solve_for_a4b4_l708_70846


namespace NUMINAMATH_GPT_MN_length_correct_l708_70853

open Real

noncomputable def MN_segment_length (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ :=
  sqrt (a * b)

theorem MN_length_correct (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ (MN : ℝ), MN = MN_segment_length a b h1 h2 :=
by
  use sqrt (a * b)
  exact rfl

end NUMINAMATH_GPT_MN_length_correct_l708_70853


namespace NUMINAMATH_GPT_find_principal_l708_70810

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * ((1 + r / n) ^ (n * t))

theorem find_principal
  (A : ℝ) (r : ℝ) (n t : ℕ)
  (hA : A = 4410)
  (hr : r = 0.05)
  (hn : n = 1)
  (ht : t = 2) :
  ∃ (P : ℝ), compound_interest P r n t = A ∧ P = 4000 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l708_70810


namespace NUMINAMATH_GPT_can_cut_one_more_square_l708_70882

theorem can_cut_one_more_square (G : Finset (Fin 29 × Fin 29)) (hG : G.card = 99) :
  (∃ S : Finset (Fin 29 × Fin 29), S.card = 4 ∧ (S ⊆ G) ∧ (∀ s1 s2 : Fin 29 × Fin 29, s1 ∈ S → s2 ∈ S → s1 ≠ s2 → (|s1.1 - s2.1| > 2 ∨ |s1.2 - s2.2| > 2))) :=
sorry

end NUMINAMATH_GPT_can_cut_one_more_square_l708_70882


namespace NUMINAMATH_GPT_find_x_l708_70814

theorem find_x (P0 P1 P2 P3 P4 P5 : ℝ) (y : ℝ) (h1 : P1 = P0 * 1.10)
                                      (h2 : P2 = P1 * 0.85)
                                      (h3 : P3 = P2 * 1.20)
                                      (h4 : P4 = P3 * (1 - x/100))
                                      (h5 : y = 0.15)
                                      (h6 : P5 = P4 * 1.15)
                                      (h7 : P5 = P0) : x = 23 :=
sorry

end NUMINAMATH_GPT_find_x_l708_70814


namespace NUMINAMATH_GPT_range_of_a_div_b_l708_70858

theorem range_of_a_div_b (a b : ℝ) (h1 : 1 < a ∧ a < 4) (h2 : 2 < b ∧ b < 8) : 
  1 / 8 < a / b ∧ a / b < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_div_b_l708_70858


namespace NUMINAMATH_GPT_value_of_c_l708_70871

theorem value_of_c (c : ℝ) : (∃ x : ℝ, x^2 + c * x - 36 = 0 ∧ x = -9) → c = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_c_l708_70871


namespace NUMINAMATH_GPT_fraction_of_sy_not_declared_major_l708_70897

-- Conditions
variables (T : ℝ) -- Total number of students
variables (first_year : ℝ) -- Fraction of first-year students
variables (second_year : ℝ) -- Fraction of second-year students
variables (decl_fy_major : ℝ) -- Fraction of first-year students who have declared a major
variables (decl_sy_major : ℝ) -- Fraction of second-year students who have declared a major

-- Definitions from conditions
def fraction_first_year_students := 1 / 2
def fraction_second_year_students := 1 / 2
def fraction_fy_declared_major := 1 / 5
def fraction_sy_declared_major := 4 * fraction_fy_declared_major

-- Hollow statement
theorem fraction_of_sy_not_declared_major :
  first_year = fraction_first_year_students →
  second_year = fraction_second_year_students →
  decl_fy_major = fraction_fy_declared_major →
  decl_sy_major = fraction_sy_declared_major →
  (1 - decl_sy_major) * second_year = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_sy_not_declared_major_l708_70897


namespace NUMINAMATH_GPT_find_x_l708_70852

variables (t x : ℕ)

theorem find_x (h1 : 0 < t) (h2 : t = 4) (h3 : ((9 / 10 : ℚ) * (t * x : ℚ)) - 6 = 48) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l708_70852


namespace NUMINAMATH_GPT_inequality_solution_set_inequality_proof_l708_70829

def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

theorem inequality_solution_set :
  ∀ x : ℝ, -2 < f x ∧ f x < 0 ↔ -1/2 < x ∧ x < 1/2 :=
by
  sorry

theorem inequality_proof (m n : ℝ) (h_m : -1/2 < m ∧ m < 1/2) (h_n : -1/2 < n ∧ n < 1/2) :
  |1 - 4 * m * n| > 2 * |m - n| :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_inequality_proof_l708_70829


namespace NUMINAMATH_GPT_point_B_value_l708_70865

theorem point_B_value (A : ℝ) (B : ℝ) (hA : A = -5) (hB : B = -1 ∨ B = -9) :
  ∃ B : ℝ, (B = A + 4 ∨ B = A - 4) :=
by sorry

end NUMINAMATH_GPT_point_B_value_l708_70865


namespace NUMINAMATH_GPT_trigonometric_identity_l708_70824

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  1 + Real.sin α * Real.cos α = 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l708_70824


namespace NUMINAMATH_GPT_paper_clips_in_morning_l708_70820

variable (p : ℕ) (used left : ℕ)

theorem paper_clips_in_morning (h1 : left = 26) (h2 : used = 59) (h3 : left = p - used) : p = 85 :=
by
  sorry

end NUMINAMATH_GPT_paper_clips_in_morning_l708_70820


namespace NUMINAMATH_GPT_megan_pictures_l708_70804

theorem megan_pictures (pictures_zoo pictures_museum pictures_deleted : ℕ)
  (hzoo : pictures_zoo = 15)
  (hmuseum : pictures_museum = 18)
  (hdeleted : pictures_deleted = 31) :
  (pictures_zoo + pictures_museum) - pictures_deleted = 2 :=
by
  sorry

end NUMINAMATH_GPT_megan_pictures_l708_70804


namespace NUMINAMATH_GPT_y_pow_x_eq_x_pow_y_l708_70890

theorem y_pow_x_eq_x_pow_y (n : ℕ) (hn : 0 < n) :
    let x := (1 + 1 / (n : ℝ)) ^ n
    let y := (1 + 1 / (n : ℝ)) ^ (n + 1)
    y ^ x = x ^ y := 
    sorry

end NUMINAMATH_GPT_y_pow_x_eq_x_pow_y_l708_70890


namespace NUMINAMATH_GPT_find_fraction_l708_70866

theorem find_fraction
  (a₁ a₂ b₁ b₂ c₁ c₂ x y : ℚ)
  (h₁ : a₁ = 3) (h₂ : a₂ = 7) (h₃ : b₁ = 6) (h₄ : b₂ = 5)
  (h₅ : c₁ = 1) (h₆ : c₂ = 7)
  (h : (a₁ / a₂) / (b₁ / b₂) = (c₁ / c₂) / (x / y)) :
  (x / y) = 2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_find_fraction_l708_70866


namespace NUMINAMATH_GPT_largest_n_l708_70841

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def valid_n (x y : ℕ) : Prop :=
  x < 10 ∧ y < 10 ∧ x ≠ y ∧ is_prime x ∧ is_prime y ∧ is_prime (10 * y + x) ∧
  100 ≤ x * y * (10 * y + x) ∧ x * y * (10 * y + x) < 1000

theorem largest_n : ∃ x y : ℕ, valid_n x y ∧ x * y * (10 * y + x) = 777 := by
  sorry

end NUMINAMATH_GPT_largest_n_l708_70841


namespace NUMINAMATH_GPT_number_of_diagonals_excluding_dividing_diagonals_l708_70832

theorem number_of_diagonals_excluding_dividing_diagonals (n : ℕ) (h1 : n = 150) :
  let totalDiagonals := n * (n - 3) / 2
  let dividingDiagonals := n / 2
  totalDiagonals - dividingDiagonals = 10950 :=
by
  sorry

end NUMINAMATH_GPT_number_of_diagonals_excluding_dividing_diagonals_l708_70832


namespace NUMINAMATH_GPT_child_running_speed_l708_70842

theorem child_running_speed
  (c s t : ℝ)
  (h1 : (74 - s) * 3 = 165)
  (h2 : (74 + s) * t = 372) :
  c = 74 :=
by sorry

end NUMINAMATH_GPT_child_running_speed_l708_70842


namespace NUMINAMATH_GPT_marco_might_need_at_least_n_tables_n_tables_are_sufficient_l708_70893
open Function

variables (n : ℕ) (friends_sticker_sets : Fin n → Finset (Fin n))

-- Each friend is missing exactly one unique sticker
def each_friend_missing_one_unique_sticker :=
  ∀ i : Fin n, ∃ j : Fin n, friends_sticker_sets i = (Finset.univ \ {j})

-- A pair of friends is wholesome if their combined collection has all stickers
def is_wholesome_pair (i j : Fin n) :=
  ∀ s : Fin n, s ∈ friends_sticker_sets i ∨ s ∈ friends_sticker_sets j

-- Main problem statements
-- Problem 1: Marco might need to reserve at least n different tables
theorem marco_might_need_at_least_n_tables 
  (h : each_friend_missing_one_unique_sticker n friends_sticker_sets) : 
  ∃ i j : Fin n, i ≠ j ∧ is_wholesome_pair n friends_sticker_sets i j :=
sorry

-- Problem 2: n tables will always be enough for Marco to achieve his goal
theorem n_tables_are_sufficient
  (h : each_friend_missing_one_unique_sticker n friends_sticker_sets) :
  ∃ arrangement : Fin n → Fin n, ∀ i j, i ≠ j → arrangement i ≠ arrangement j :=
sorry

end NUMINAMATH_GPT_marco_might_need_at_least_n_tables_n_tables_are_sufficient_l708_70893


namespace NUMINAMATH_GPT_find_angle_CBO_l708_70836

theorem find_angle_CBO :
  ∀ (BAO CAO CBO ABO ACO BCO AOC : ℝ), 
  BAO = CAO → 
  CBO = ABO → 
  ACO = BCO → 
  AOC = 110 →
  CBO = 20 :=
by
  intros BAO CAO CBO ABO ACO BCO AOC hBAO_CAOC hCBO_ABO hACO_BCO hAOC
  sorry

end NUMINAMATH_GPT_find_angle_CBO_l708_70836


namespace NUMINAMATH_GPT_impossible_d_values_count_l708_70844

def triangle_rectangle_difference (d : ℕ) : Prop :=
  ∃ (l w : ℕ),
  l = 2 * w ∧
  6 * w > 0 ∧
  (6 * w + 2 * d) - 6 * w = 1236 ∧
  d > 0

theorem impossible_d_values_count : ∀ d : ℕ, d ≠ 618 → ¬triangle_rectangle_difference d :=
by
  sorry

end NUMINAMATH_GPT_impossible_d_values_count_l708_70844


namespace NUMINAMATH_GPT_prove_u_div_p_l708_70847

theorem prove_u_div_p (p r s u : ℚ) 
  (h1 : p / r = 8)
  (h2 : s / r = 5)
  (h3 : s / u = 1 / 3) : 
  u / p = 15 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_prove_u_div_p_l708_70847


namespace NUMINAMATH_GPT_radius_of_smaller_molds_l708_70835

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r ^ 3

theorem radius_of_smaller_molds :
  (64 * hemisphere_volume (1/2)) = hemisphere_volume 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_smaller_molds_l708_70835


namespace NUMINAMATH_GPT_leak_empty_tank_time_l708_70809

theorem leak_empty_tank_time (A L : ℝ) (hA : A = 1 / 10) (hAL : A - L = 1 / 15) : (1 / L = 30) :=
sorry

end NUMINAMATH_GPT_leak_empty_tank_time_l708_70809


namespace NUMINAMATH_GPT_person_speed_in_kmph_l708_70891

-- Define the distance in meters
def distance_meters : ℕ := 300

-- Define the time in minutes
def time_minutes : ℕ := 4

-- Function to convert distance from meters to kilometers
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000

-- Function to convert time from minutes to hours
def minutes_to_hours (min : ℕ) : ℚ := min / 60

-- Define the expected speed in km/h
def expected_speed : ℚ := 4.5

-- Proof statement
theorem person_speed_in_kmph : 
  meters_to_kilometers distance_meters / minutes_to_hours time_minutes = expected_speed :=
by 
  -- This is where the steps to verify the theorem would be located, currently omitted for the sake of the statement.
  sorry

end NUMINAMATH_GPT_person_speed_in_kmph_l708_70891


namespace NUMINAMATH_GPT_sum_of_fractions_l708_70808

-- Definition of the fractions given as conditions
def frac1 := 2 / 10
def frac2 := 4 / 40
def frac3 := 6 / 60
def frac4 := 8 / 30

-- Statement of the theorem to prove
theorem sum_of_fractions : frac1 + frac2 + frac3 + frac4 = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l708_70808


namespace NUMINAMATH_GPT_real_roots_of_f_l708_70887

noncomputable def f (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem real_roots_of_f :
  {x | f x = 0} = {-1, 1, 2, 3} :=
sorry

end NUMINAMATH_GPT_real_roots_of_f_l708_70887


namespace NUMINAMATH_GPT_total_number_of_students_l708_70884

theorem total_number_of_students (b h p s : ℕ) 
  (h1 : b = 30)
  (h2 : b = 2 * h)
  (h3 : p = h + 5)
  (h4 : s = 3 * p) :
  b + h + p + s = 125 :=
by sorry

end NUMINAMATH_GPT_total_number_of_students_l708_70884


namespace NUMINAMATH_GPT_eval_exp_l708_70817

theorem eval_exp : (3^3)^2 = 729 := sorry

end NUMINAMATH_GPT_eval_exp_l708_70817


namespace NUMINAMATH_GPT_largest_three_digit_multiple_of_12_and_sum_of_digits_24_l708_70849

def sum_of_digits (n : ℕ) : ℕ :=
  ((n / 100) + ((n / 10) % 10) + (n % 10))

def is_multiple_of_12 (n : ℕ) : Prop :=
  n % 12 = 0

def largest_three_digit_multiple_of_12_with_digits_sum_24 : ℕ :=
  996

theorem largest_three_digit_multiple_of_12_and_sum_of_digits_24 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ sum_of_digits n = 24 ∧ is_multiple_of_12 n ∧ n = largest_three_digit_multiple_of_12_with_digits_sum_24 :=
by 
  sorry

end NUMINAMATH_GPT_largest_three_digit_multiple_of_12_and_sum_of_digits_24_l708_70849


namespace NUMINAMATH_GPT_tile_covering_problem_l708_70863

theorem tile_covering_problem :
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12  -- converting feet to inches
  let region_width := 3 * 12   -- converting feet to inches
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  region_area / tile_area = 144 := 
by 
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12
  let region_width := 3 * 12
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  sorry

end NUMINAMATH_GPT_tile_covering_problem_l708_70863


namespace NUMINAMATH_GPT_probability_of_green_ball_l708_70896

-- Definitions according to the conditions.
def containerA : ℕ × ℕ := (4, 6) -- 4 red balls, 6 green balls
def containerB : ℕ × ℕ := (6, 4) -- 6 red balls, 4 green balls
def containerC : ℕ × ℕ := (6, 4) -- 6 red balls, 4 green balls

-- Proving the probability of selecting a green ball.
theorem probability_of_green_ball :
  let pA := 1 / 3
  let pB := 1 / 3
  let pC := 1 / 3
  let pGreenA := (containerA.2 : ℚ) / (containerA.1 + containerA.2)
  let pGreenB := (containerB.2 : ℚ) / (containerB.1 + containerB.2)
  let pGreenC := (containerC.2 : ℚ) / (containerC.1 + containerC.2)
  pA * pGreenA + pB * pGreenB + pC * pGreenC = 7 / 15
  :=
by
  -- Formal proof will be filled in here.
  sorry

end NUMINAMATH_GPT_probability_of_green_ball_l708_70896


namespace NUMINAMATH_GPT_total_students_like_sports_l708_70850

def Total_students := 30

def B : ℕ := 12
def C : ℕ := 10
def S : ℕ := 8
def BC : ℕ := 4
def BS : ℕ := 3
def CS : ℕ := 2
def BCS : ℕ := 1

theorem total_students_like_sports : 
  (B + C + S - (BC + BS + CS) + BCS = 22) := by
  sorry

end NUMINAMATH_GPT_total_students_like_sports_l708_70850


namespace NUMINAMATH_GPT_cost_price_percentage_l708_70878

-- Define the condition that the profit percent is 11.11111111111111%
def profit_percent (CP SP: ℝ) : Prop :=
  ((SP - CP) / CP) * 100 = 11.11111111111111

-- Prove that under this condition, the cost price (CP) is 90% of the selling price (SP).
theorem cost_price_percentage (CP SP : ℝ) (h: profit_percent CP SP) : (CP / SP) * 100 = 90 :=
sorry

end NUMINAMATH_GPT_cost_price_percentage_l708_70878


namespace NUMINAMATH_GPT_inequality_b_c_a_l708_70885

-- Define the values of a, b, and c
def a := 8^53
def b := 16^41
def c := 64^27

-- State the theorem to prove the inequality b > c > a
theorem inequality_b_c_a : b > c ∧ c > a := by
  sorry

end NUMINAMATH_GPT_inequality_b_c_a_l708_70885


namespace NUMINAMATH_GPT_sum_a4_a6_l708_70828

variable (a : ℕ → ℝ) (d : ℝ)
variable (h_arith : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
variable (h_sum : a 2 + a 3 + a 7 + a 8 = 8)

theorem sum_a4_a6 : a 4 + a 6 = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_a4_a6_l708_70828


namespace NUMINAMATH_GPT_legs_total_l708_70888

def number_of_legs_bee := 6
def number_of_legs_spider := 8
def number_of_bees := 5
def number_of_spiders := 2
def total_legs := number_of_bees * number_of_legs_bee + number_of_spiders * number_of_legs_spider

theorem legs_total : total_legs = 46 := by
  sorry

end NUMINAMATH_GPT_legs_total_l708_70888


namespace NUMINAMATH_GPT_church_members_l708_70876

theorem church_members (M A C : ℕ) (h1 : A = 4/10 * M)
  (h2 : C = 6/10 * M) (h3 : C = A + 24) : M = 120 := 
  sorry

end NUMINAMATH_GPT_church_members_l708_70876


namespace NUMINAMATH_GPT_trigonometric_identity_l708_70879

theorem trigonometric_identity 
  (α β : ℝ) 
  (h : α + β = π / 3)  -- Note: 60 degrees is π/3 radians
  (tan_add : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) 
  (tan_60 : Real.tan (π / 3) = Real.sqrt 3) :
  Real.tan α + Real.tan β + Real.sqrt 3 * Real.tan α * Real.tan β = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l708_70879


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l708_70816

-- Defining the given condition
def ratio_condition (x y : ℝ) : Prop :=
  (3 * x - 2 * y) / (2 * x + y) = 3 / 5

-- The theorem to be proven
theorem ratio_of_x_to_y (x y : ℝ) (h : ratio_condition x y) : x / y = 13 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l708_70816


namespace NUMINAMATH_GPT_average_and_fourth_number_l708_70830

theorem average_and_fourth_number {x : ℝ} (h_avg : ((1 + 2 + 4 + 6 + 9 + 9 + 10 + 12 + x) / 9) = 7) :
  x = 10 ∧ 6 = 6 :=
by
  sorry

end NUMINAMATH_GPT_average_and_fourth_number_l708_70830


namespace NUMINAMATH_GPT_wyatt_headmaster_duration_l708_70805

def duration_of_wyatt_job (start_month end_month total_months : ℕ) : Prop :=
  start_month <= end_month ∧ total_months = end_month - start_month + 1

theorem wyatt_headmaster_duration : duration_of_wyatt_job 3 12 9 :=
by
  sorry

end NUMINAMATH_GPT_wyatt_headmaster_duration_l708_70805


namespace NUMINAMATH_GPT_min_value_expression_l708_70827

theorem min_value_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 3) :
  ∃ (M : ℝ), M = (2 : ℝ) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 3 → ((y / x) + (3 / (y + 1)) ≥ M)) :=
by
  use 2
  sorry

end NUMINAMATH_GPT_min_value_expression_l708_70827


namespace NUMINAMATH_GPT_perpendicular_lines_l708_70851

theorem perpendicular_lines (a : ℝ) :
  (a + 2) * (a - 1) + (1 - a) * (2 * a + 3) = 0 ↔ (a = 1 ∨ a = -1) := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_l708_70851


namespace NUMINAMATH_GPT_average_nums_correct_l708_70821

def nums : List ℕ := [55, 48, 507, 2, 684, 42]

theorem average_nums_correct :
  (List.sum nums) / (nums.length) = 223 := by
  sorry

end NUMINAMATH_GPT_average_nums_correct_l708_70821


namespace NUMINAMATH_GPT_four_digit_numbers_neither_5_nor_7_l708_70823

-- Define the range of four-digit numbers
def four_digit_numbers : Set ℕ := {x | 1000 ≤ x ∧ x ≤ 9999}

-- Define the predicates for multiples of 5, 7, and 35
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0
def is_multiple_of_7 (n : ℕ) : Prop := n % 7 = 0
def is_multiple_of_35 (n : ℕ) : Prop := n % 35 = 0

-- Using set notation to define the sets of multiples
def multiples_of_5 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_5 n}
def multiples_of_7 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_7 n}
def multiples_of_35 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_35 n}

-- Total count of 4-digit numbers
def total_four_digit_numbers : ℕ := 9000

-- Count of multiples of 5, 7, and 35 within 4-digit numbers
def count_multiples_of_5 : ℕ := 1800
def count_multiples_of_7 : ℕ := 1286
def count_multiples_of_35 : ℕ := 257

-- Count of multiples of 5 or 7 using the principle of inclusion-exclusion
def count_multiples_of_5_or_7 : ℕ := count_multiples_of_5 + count_multiples_of_7 - count_multiples_of_35

-- Prove that the number of 4-digit numbers which are multiples of neither 5 nor 7 is 6171
theorem four_digit_numbers_neither_5_nor_7 : 
  (total_four_digit_numbers - count_multiples_of_5_or_7) = 6171 := 
by 
  sorry

end NUMINAMATH_GPT_four_digit_numbers_neither_5_nor_7_l708_70823


namespace NUMINAMATH_GPT_slices_remaining_is_correct_l708_70854

def slices_per_pizza : ℕ := 8
def pizzas_ordered : ℕ := 2
def slices_eaten : ℕ := 7
def total_slices : ℕ := slices_per_pizza * pizzas_ordered
def slices_remaining : ℕ := total_slices - slices_eaten

theorem slices_remaining_is_correct : slices_remaining = 9 := by
  sorry

end NUMINAMATH_GPT_slices_remaining_is_correct_l708_70854


namespace NUMINAMATH_GPT_problem_proof_l708_70886

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 * x + 2 - x

-- Condition given in the problem
axiom h : ∃ a : ℝ, f a = 3

-- Theorem statement
theorem problem_proof : ∃ a : ℝ, f a = 3 → f (2 * a) = 7 :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l708_70886


namespace NUMINAMATH_GPT_total_crosswalk_lines_l708_70880

theorem total_crosswalk_lines (n m l : ℕ) (h1 : n = 5) (h2 : m = 4) (h3 : l = 20) :
  n * (m * l) = 400 := by
  sorry

end NUMINAMATH_GPT_total_crosswalk_lines_l708_70880


namespace NUMINAMATH_GPT_triangle_side_relation_l708_70839

theorem triangle_side_relation (a b c : ℝ) (α β γ : ℝ)
  (h1 : 3 * α + 2 * β = 180)
  (h2 : α + β + γ = 180) :
  a^2 + b * c - c^2 = 0 :=
sorry

end NUMINAMATH_GPT_triangle_side_relation_l708_70839


namespace NUMINAMATH_GPT_probability_both_groups_stop_same_round_l708_70881

noncomputable def probability_same_round : ℚ :=
  let probability_fair_coin_stop (n : ℕ) : ℚ := (1/2)^n
  let probability_biased_coin_stop (n : ℕ) : ℚ := (2/3)^(n-1) * (1/3)
  let probability_fair_coin_group_stop (n : ℕ) : ℚ := (probability_fair_coin_stop n)^3
  let probability_biased_coin_group_stop (n : ℕ) : ℚ := (probability_biased_coin_stop n)^3
  let combined_round_probability (n : ℕ) : ℚ := 
    probability_fair_coin_group_stop n * probability_biased_coin_group_stop n
  let total_probability : ℚ := ∑' n, combined_round_probability n
  total_probability

theorem probability_both_groups_stop_same_round :
  probability_same_round = 1 / 702 := by sorry

end NUMINAMATH_GPT_probability_both_groups_stop_same_round_l708_70881


namespace NUMINAMATH_GPT_three_digit_divisible_by_8_l708_70862

theorem three_digit_divisible_by_8 : ∃ n : ℕ, n / 100 = 5 ∧ n % 10 = 3 ∧ n % 8 = 0 :=
by
  use 533
  sorry

end NUMINAMATH_GPT_three_digit_divisible_by_8_l708_70862


namespace NUMINAMATH_GPT_area_of_smallest_square_containing_circle_l708_70800

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 7) : ∃ s, s = 14 ∧ s * s = 196 :=
by
  sorry

end NUMINAMATH_GPT_area_of_smallest_square_containing_circle_l708_70800


namespace NUMINAMATH_GPT_age_difference_l708_70819

theorem age_difference (S M : ℕ) 
  (h1 : S = 35)
  (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 37 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l708_70819


namespace NUMINAMATH_GPT_find_k_l708_70811

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^2 + 2 * x + 1) / (k * x - 1)

theorem find_k (k : ℝ) : (∀ x : ℝ, f k (f k x) = x) ↔ k = -2 :=
  sorry

end NUMINAMATH_GPT_find_k_l708_70811


namespace NUMINAMATH_GPT_factor_expression_l708_70834

theorem factor_expression :
  (12 * x ^ 6 + 40 * x ^ 4 - 6) - (2 * x ^ 6 - 6 * x ^ 4 - 6) = 2 * x ^ 4 * (5 * x ^ 2 + 23) :=
by sorry

end NUMINAMATH_GPT_factor_expression_l708_70834


namespace NUMINAMATH_GPT_tangent_function_range_l708_70812

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + 1
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x

theorem tangent_function_range {a : ℝ} :
  (∃ (m : ℝ), 4 * m^3 - 3 * a * m^2 + 6 = 0) ↔ a > 2 * Real.sqrt 33 :=
sorry -- proof omitted

end NUMINAMATH_GPT_tangent_function_range_l708_70812


namespace NUMINAMATH_GPT_parabola_through_P_l708_70898

-- Define the point P
def P : ℝ × ℝ := (4, -2)

-- Define a condition function for equations y^2 = a*x
def satisfies_y_eq_ax (a : ℝ) : Prop := 
  ∃ x y, (x, y) = P ∧ y^2 = a * x

-- Define a condition function for equations x^2 = b*y
def satisfies_x_eq_by (b : ℝ) : Prop := 
  ∃ x y, (x, y) = P ∧ x^2 = b * y

-- Lean's theorem statement
theorem parabola_through_P : satisfies_y_eq_ax 1 ∨ satisfies_x_eq_by (-8) :=
sorry

end NUMINAMATH_GPT_parabola_through_P_l708_70898


namespace NUMINAMATH_GPT_smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l708_70869

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l708_70869


namespace NUMINAMATH_GPT_initial_number_of_observations_l708_70818

theorem initial_number_of_observations (n : ℕ) 
  (initial_mean : ℝ := 100) 
  (wrong_obs : ℝ := 75) 
  (corrected_obs : ℝ := 50) 
  (corrected_mean : ℝ := 99.075) 
  (h1 : (n:ℝ) * initial_mean = n * corrected_mean + wrong_obs - corrected_obs) 
  (h2 : n = (25 : ℝ) / 0.925) 
  : n = 27 := 
sorry

end NUMINAMATH_GPT_initial_number_of_observations_l708_70818


namespace NUMINAMATH_GPT_max_y_difference_intersection_l708_70860

noncomputable def f (x : ℝ) : ℝ := 4 - x^2 + x^3
noncomputable def g (x : ℝ) : ℝ := 2 + x^2 + x^3

theorem max_y_difference_intersection :
  let x1 := 1
  let y1 := g x1
  let x2 := -1
  let y2 := g x2
  y1 - y2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_max_y_difference_intersection_l708_70860


namespace NUMINAMATH_GPT_factor_in_form_of_2x_l708_70874

theorem factor_in_form_of_2x (w : ℕ) (hw : w = 144) : ∃ x : ℕ, 936 * w = 2^x * P → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_factor_in_form_of_2x_l708_70874


namespace NUMINAMATH_GPT_find_k_l708_70826

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def construct_number (k : ℕ) : ℕ :=
  let n := 1000
  let a := (10^(2000 - k) - 1) / 9
  let b := (10^(1001) - 1) / 9
  a * 10^(1001) + k * 10^(1001 - k) - b

theorem find_k : ∀ k : ℕ, (construct_number k > 0) ∧ (isPerfectSquare (construct_number k) ↔ k = 2) := 
by 
  intro k
  sorry

end NUMINAMATH_GPT_find_k_l708_70826


namespace NUMINAMATH_GPT_two_p_plus_q_l708_70833

variable {p q : ℚ}

theorem two_p_plus_q (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 :=
by
  sorry

end NUMINAMATH_GPT_two_p_plus_q_l708_70833


namespace NUMINAMATH_GPT_volume_relation_l708_70815

variable {x y z V : ℝ}

theorem volume_relation
  (top_area : x * y = A)
  (side_area : y * z = B)
  (volume : x * y * z = V) :
  (y * z) * (x * y * z)^2 = z^3 * V := by
  sorry

end NUMINAMATH_GPT_volume_relation_l708_70815


namespace NUMINAMATH_GPT_selling_price_correct_l708_70892

def meters_of_cloth : ℕ := 45
def profit_per_meter : ℝ := 12
def cost_price_per_meter : ℝ := 88
def total_selling_price : ℝ := 4500

theorem selling_price_correct :
  (cost_price_per_meter * meters_of_cloth) + (profit_per_meter * meters_of_cloth) = total_selling_price :=
by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l708_70892


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l708_70894

theorem quadratic_inequality_solution (m : ℝ) :
  {x : ℝ | (x - m) * (x - (m + 1)) > 0} = {x | x < m ∨ x > m + 1} := sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l708_70894


namespace NUMINAMATH_GPT_morning_snowfall_l708_70875

theorem morning_snowfall (total_snowfall afternoon_snowfall morning_snowfall : ℝ) 
  (h1 : total_snowfall = 0.625) 
  (h2 : afternoon_snowfall = 0.5) 
  (h3 : total_snowfall = morning_snowfall + afternoon_snowfall) : 
  morning_snowfall = 0.125 :=
by
  sorry

end NUMINAMATH_GPT_morning_snowfall_l708_70875


namespace NUMINAMATH_GPT_point_P_below_line_l708_70825

def line_equation (x y : ℝ) : ℝ := 2 * x - y + 3

def point_below_line (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  2 * x - y + 3 > 0

theorem point_P_below_line :
  point_below_line (1, -1) :=
by
  sorry

end NUMINAMATH_GPT_point_P_below_line_l708_70825


namespace NUMINAMATH_GPT_train_length_is_360_l708_70899

-- Conditions from the problem
variable (speed_kmph : ℕ) (time_sec : ℕ) (platform_length_m : ℕ)

-- Definitions to be used for the conditions
def speed_ms (speed_kmph : ℕ) : ℤ := (speed_kmph * 1000) / 3600 -- Speed in m/s
def total_distance (speed_ms : ℤ) (time_sec : ℕ) : ℤ := speed_ms * (time_sec : ℤ) -- Total distance covered
def train_length (total_distance : ℤ) (platform_length : ℤ) : ℤ := total_distance - platform_length -- Length of the train

-- Assertion statement
theorem train_length_is_360 : train_length (total_distance (speed_ms speed_kmph) time_sec) platform_length_m = 360 := 
  by sorry

end NUMINAMATH_GPT_train_length_is_360_l708_70899


namespace NUMINAMATH_GPT_exists_rectangle_in_inscribed_right_triangle_l708_70843

theorem exists_rectangle_in_inscribed_right_triangle :
  ∃ (L W : ℝ), 
    (45^2 / (1 + (5/2)^2) = L * L) ∧
    (2 * L = 45) ∧
    (2 * W = 45) ∧
    ((L = 25 ∧ W = 10) ∨ (L = 18.75 ∧ W = 7.5)) :=
by sorry

end NUMINAMATH_GPT_exists_rectangle_in_inscribed_right_triangle_l708_70843


namespace NUMINAMATH_GPT_circle_ellipse_intersect_four_points_l708_70831

theorem circle_ellipse_intersect_four_points (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = a^2 → y = x^2 / 2 - a) →
  a > 1 :=
by
  sorry

end NUMINAMATH_GPT_circle_ellipse_intersect_four_points_l708_70831


namespace NUMINAMATH_GPT_find_unknown_numbers_l708_70807

def satisfies_condition1 (A B : ℚ) : Prop := 
  0.05 * A = 0.20 * 650 + 0.10 * B

def satisfies_condition2 (A B : ℚ) : Prop := 
  A + B = 4000

def satisfies_condition3 (B C : ℚ) : Prop := 
  C = 2 * B

def satisfies_condition4 (A B C D : ℚ) : Prop := 
  A + B + C = 0.40 * D

theorem find_unknown_numbers (A B C D : ℚ) :
  satisfies_condition1 A B → satisfies_condition2 A B →
  satisfies_condition3 B C → satisfies_condition4 A B C D →
  A = 3533 + 1/3 ∧ B = 466 + 2/3 ∧ C = 933 + 1/3 ∧ D = 12333 + 1/3 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_numbers_l708_70807


namespace NUMINAMATH_GPT_trip_time_l708_70895

theorem trip_time (distance half_dist speed1 speed2 : ℝ) 
  (h_distance : distance = 360) 
  (h_half_distance : half_dist = distance / 2) 
  (h_speed1 : speed1 = 50) 
  (h_speed2 : speed2 = 45) : 
  (half_dist / speed1 + half_dist / speed2) = 7.6 := 
by
  -- Simplify the expressions based on provided conditions
  sorry

end NUMINAMATH_GPT_trip_time_l708_70895


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l708_70813

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), is_isosceles a b c ∧ ((a = 3 ∧ b = 3 ∧ c = 4 ∧ a + b + c = 10) ∨ (a = 3 ∧ b = 4 ∧ c = 4 ∧ a + b + c = 11)) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l708_70813


namespace NUMINAMATH_GPT_johnny_red_pencils_l708_70859

noncomputable def number_of_red_pencils (packs_total : ℕ) (extra_packs : ℕ) (extra_per_pack : ℕ) : ℕ :=
  packs_total + extra_packs * extra_per_pack

theorem johnny_red_pencils : number_of_red_pencils 15 3 2 = 21 := by
  sorry

end NUMINAMATH_GPT_johnny_red_pencils_l708_70859


namespace NUMINAMATH_GPT_solve_system_eq_l708_70822

theorem solve_system_eq (x1 x2 x3 x4 x5 : ℝ) :
  (x3 + x4 + x5)^5 = 3 * x1 ∧
  (x4 + x5 + x1)^5 = 3 * x2 ∧
  (x5 + x1 + x2)^5 = 3 * x3 ∧
  (x1 + x2 + x3)^5 = 3 * x4 ∧
  (x2 + x3 + x4)^5 = 3 * x5 →
  (x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨
  (x1 = 1/3 ∧ x2 = 1/3 ∧ x3 = 1/3 ∧ x4 = 1/3 ∧ x5 = 1/3) ∨
  (x1 = -1/3 ∧ x2 = -1/3 ∧ x3 = -1/3 ∧ x4 = -1/3 ∧ x5 = -1/3) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_eq_l708_70822


namespace NUMINAMATH_GPT_correct_calculation_l708_70838

variable (a b : ℝ)

theorem correct_calculation : (ab)^2 = a^2 * b^2 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l708_70838


namespace NUMINAMATH_GPT_isabella_most_efficient_jumper_l708_70870

noncomputable def weight_ricciana : ℝ := 120
noncomputable def jump_ricciana : ℝ := 4

noncomputable def weight_margarita : ℝ := 110
noncomputable def jump_margarita : ℝ := 2 * jump_ricciana - 1

noncomputable def weight_isabella : ℝ := 100
noncomputable def jump_isabella : ℝ := jump_ricciana + 3

noncomputable def ratio_ricciana : ℝ := weight_ricciana / jump_ricciana
noncomputable def ratio_margarita : ℝ := weight_margarita / jump_margarita
noncomputable def ratio_isabella : ℝ := weight_isabella / jump_isabella

theorem isabella_most_efficient_jumper :
  ratio_isabella < ratio_margarita ∧ ratio_isabella < ratio_ricciana :=
by
  sorry

end NUMINAMATH_GPT_isabella_most_efficient_jumper_l708_70870


namespace NUMINAMATH_GPT_chess_tournament_games_l708_70877

def stage1_games (players : ℕ) : ℕ := (players * (players - 1) * 2) / 2
def stage2_games (players : ℕ) : ℕ := (players * (players - 1) * 2) / 2
def stage3_games : ℕ := 4

def total_games (stage1 stage2 stage3 : ℕ) : ℕ := stage1 + stage2 + stage3

theorem chess_tournament_games : total_games (stage1_games 20) (stage2_games 10) stage3_games = 474 :=
by
  unfold stage1_games
  unfold stage2_games
  unfold total_games
  simp
  sorry

end NUMINAMATH_GPT_chess_tournament_games_l708_70877


namespace NUMINAMATH_GPT_max_ballpoint_pens_l708_70803

def ballpoint_pen_cost : ℕ := 10
def gel_pen_cost : ℕ := 30
def fountain_pen_cost : ℕ := 60
def total_pens : ℕ := 20
def total_cost : ℕ := 500

theorem max_ballpoint_pens : ∃ (x y z : ℕ), 
  x + y + z = total_pens ∧ 
  ballpoint_pen_cost * x + gel_pen_cost * y + fountain_pen_cost * z = total_cost ∧ 
  1 ≤ x ∧ 
  1 ≤ y ∧
  1 ≤ z ∧
  ∀ x', ((∃ y' z', x' + y' + z' = total_pens ∧ 
                    ballpoint_pen_cost * x' + gel_pen_cost * y' + fountain_pen_cost * z' = total_cost ∧ 
                    1 ≤ x' ∧ 
                    1 ≤ y' ∧
                    1 ≤ z') → x' ≤ x) :=
  sorry

end NUMINAMATH_GPT_max_ballpoint_pens_l708_70803


namespace NUMINAMATH_GPT_lines_parallel_l708_70857

def l1 (x : ℝ) : ℝ := 2 * x + 1
def l2 (x : ℝ) : ℝ := 2 * x + 5

theorem lines_parallel : ∀ x1 x2 : ℝ, l1 x1 = l2 x2 → false := 
by
  intros x1 x2 h
  rw [l1, l2] at h
  sorry

end NUMINAMATH_GPT_lines_parallel_l708_70857


namespace NUMINAMATH_GPT_weighted_average_inequality_l708_70802

variable (x y z : ℝ)
variable (h1 : x < y) (h2 : y < z)

theorem weighted_average_inequality :
  (4 * z + x + y) / 6 > (x + y + 2 * z) / 4 :=
by
  sorry

end NUMINAMATH_GPT_weighted_average_inequality_l708_70802
