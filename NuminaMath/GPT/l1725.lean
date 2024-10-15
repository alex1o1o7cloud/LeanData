import Mathlib

namespace NUMINAMATH_GPT_product_of_integer_with_100_l1725_172575

theorem product_of_integer_with_100 (x : ℝ) (h : 10 * x = x + 37.89) : 100 * x = 421 :=
by
  -- insert the necessary steps to solve the problem
  sorry

end NUMINAMATH_GPT_product_of_integer_with_100_l1725_172575


namespace NUMINAMATH_GPT_Arman_worked_last_week_l1725_172547

variable (H : ℕ) -- hours worked last week
variable (wage_last_week wage_this_week : ℝ)
variable (hours_this_week worked_this_week two_weeks_earning : ℝ)
variable (worked_last_week : Prop)

-- Define assumptions based on the problem conditions
def condition1 : wage_last_week = 10 := by sorry
def condition2 : wage_this_week = 10.5 := by sorry
def condition3 : hours_this_week = 40 := by sorry
def condition4 : worked_this_week = wage_this_week * hours_this_week := by sorry
def condition5 : worked_this_week = 420 := by sorry -- 10.5 * 40
def condition6 : two_weeks_earning = wage_last_week * (H : ℝ) + worked_this_week := by sorry
def condition7 : two_weeks_earning = 770 := by sorry

-- Proof statement
theorem Arman_worked_last_week : worked_last_week := by
  have h1 : wage_last_week * (H : ℝ) + worked_this_week = two_weeks_earning := sorry
  have h2 : wage_last_week * (H : ℝ) + 420 = 770 := sorry
  have h3 : wage_last_week * (H : ℝ) = 350 := sorry
  have h4 : (10 : ℝ) * (H : ℝ) = 350 := sorry
  have h5 : H = 35 := sorry
  sorry

end NUMINAMATH_GPT_Arman_worked_last_week_l1725_172547


namespace NUMINAMATH_GPT_number_of_possible_ordered_pairs_l1725_172513

theorem number_of_possible_ordered_pairs (n : ℕ) (f m : ℕ) 
  (cond1 : n = 6) 
  (cond2 : f ≥ 0) 
  (cond3 : m ≥ 0) 
  (cond4 : f + m ≤ 12) 
  : ∃ s : Finset (ℕ × ℕ), s.card = 6 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_possible_ordered_pairs_l1725_172513


namespace NUMINAMATH_GPT_part_a_solutions_l1725_172525

theorem part_a_solutions (x : ℝ) : (⌊x⌋^2 - x = -0.99) ↔ (x = 0.99 ∨ x = 1.99) :=
sorry

end NUMINAMATH_GPT_part_a_solutions_l1725_172525


namespace NUMINAMATH_GPT_tan_sub_pi_div_four_eq_neg_seven_f_range_l1725_172509

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 4)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

-- Proof for the first part
theorem tan_sub_pi_div_four_eq_neg_seven (x : ℝ) (h : 3 / 4 * Real.cos x + Real.sin x = 0) :
  Real.tan (x - Real.pi / 4) = -7 := sorry

noncomputable def f (x : ℝ) : ℝ := 
  2 * ((a x).fst + (b x).fst) * (b x).fst + 2 * ((a x).snd + (b x).snd) * (b x).snd

-- Proof for the second part
theorem f_range (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  1 / 2 < f x ∧ f x < 3 / 2 + Real.sqrt 2 := sorry

end NUMINAMATH_GPT_tan_sub_pi_div_four_eq_neg_seven_f_range_l1725_172509


namespace NUMINAMATH_GPT_same_side_interior_not_complementary_l1725_172599

-- Defining the concept of same-side interior angles and complementary angles
def same_side_interior (α β : ℝ) : Prop := 
  α + β = 180 

def complementary (α β : ℝ) : Prop :=
  α + β = 90

-- To state the proposition that should be proven false
theorem same_side_interior_not_complementary (α β : ℝ) (h : same_side_interior α β) : ¬ complementary α β :=
by
  -- We state the observable contradiction here, and since the proof is not required we use sorry
  sorry

end NUMINAMATH_GPT_same_side_interior_not_complementary_l1725_172599


namespace NUMINAMATH_GPT_sum_greater_than_two_l1725_172516

variables {x y : ℝ}

theorem sum_greater_than_two (hx : x^7 > y^6) (hy : y^7 > x^6) : x + y > 2 :=
sorry

end NUMINAMATH_GPT_sum_greater_than_two_l1725_172516


namespace NUMINAMATH_GPT_first_term_geometric_sequence_l1725_172528

theorem first_term_geometric_sequence (a r : ℚ) 
    (h1 : a * r^2 = 8) 
    (h2 : a * r^4 = 27 / 4) : 
    a = 256 / 27 :=
by sorry

end NUMINAMATH_GPT_first_term_geometric_sequence_l1725_172528


namespace NUMINAMATH_GPT_copies_made_in_half_hour_l1725_172524

theorem copies_made_in_half_hour
  (rate1 rate2 : ℕ)  -- rates of the two copy machines
  (time : ℕ)         -- time considered
  (h_rate1 : rate1 = 40)  -- the first machine's rate
  (h_rate2 : rate2 = 55)  -- the second machine's rate
  (h_time : time = 30)    -- time in minutes
  : (rate1 * time + rate2 * time = 2850) := 
sorry

end NUMINAMATH_GPT_copies_made_in_half_hour_l1725_172524


namespace NUMINAMATH_GPT_remainder_p_q_add_42_l1725_172515

def p (k : ℤ) : ℤ := 98 * k + 84
def q (m : ℤ) : ℤ := 126 * m + 117

theorem remainder_p_q_add_42 (k m : ℤ) : 
  (p k + q m) % 42 = 33 := by
  sorry

end NUMINAMATH_GPT_remainder_p_q_add_42_l1725_172515


namespace NUMINAMATH_GPT_distinct_ints_divisibility_l1725_172550

theorem distinct_ints_divisibility
  (x y z : ℤ) 
  (h1 : x ≠ y) 
  (h2 : y ≠ z) 
  (h3 : z ≠ x) : 
  ∃ k : ℤ, (x - y) ^ 5 + (y - z) ^ 5 + (z - x) ^ 5 = 5 * (y - z) * (z - x) * (x - y) * k := 
by 
  sorry

end NUMINAMATH_GPT_distinct_ints_divisibility_l1725_172550


namespace NUMINAMATH_GPT_translation_coordinates_l1725_172564

theorem translation_coordinates :
  ∀ (x y : ℤ) (a : ℤ), 
  (x, y) = (3, -4) → a = 5 → (x - a, y) = (-2, -4) :=
by
  sorry

end NUMINAMATH_GPT_translation_coordinates_l1725_172564


namespace NUMINAMATH_GPT_intersection_of_sets_l1725_172535

theorem intersection_of_sets :
  let A := {-2, -1, 0, 1, 2}
  let B := {x | -2 < x ∧ x ≤ 2}
  A ∩ B = {-1, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1725_172535


namespace NUMINAMATH_GPT_modulus_of_z_l1725_172592

-- Define the complex number z
def z : ℂ := -5 + 12 * Complex.I

-- Define a theorem stating the modulus of z is 13
theorem modulus_of_z : Complex.abs z = 13 :=
by
  -- This will be the place to provide proof steps
  sorry

end NUMINAMATH_GPT_modulus_of_z_l1725_172592


namespace NUMINAMATH_GPT_cards_ratio_l1725_172549

variable (x : ℕ)

def partially_full_decks_cards := 3 * x
def full_decks_cards := 3 * 52
def total_cards_before := 200 + 34

theorem cards_ratio (h : 3 * x + full_decks_cards = total_cards_before) : x / 52 = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_cards_ratio_l1725_172549


namespace NUMINAMATH_GPT_probability_to_form_computers_l1725_172593

def letters_in_campus : Finset Char := {'C', 'A', 'M', 'P', 'U', 'S'}
def letters_in_threads : Finset Char := {'T', 'H', 'R', 'E', 'A', 'D', 'S'}
def letters_in_glow : Finset Char := {'G', 'L', 'O', 'W'}
def letters_in_computers : Finset Char := {'C', 'O', 'M', 'P', 'U', 'T', 'E', 'R', 'S'}

noncomputable def probability_campus : ℚ := 1 / Nat.choose 6 3
noncomputable def probability_threads : ℚ := 1 / Nat.choose 7 5
noncomputable def probability_glow : ℚ := 1 / (Nat.choose 4 2 / Nat.choose 3 1)

noncomputable def overall_probability : ℚ :=
  probability_campus * probability_threads * probability_glow

theorem probability_to_form_computers :
  overall_probability = 1 / 840 := by
  sorry

end NUMINAMATH_GPT_probability_to_form_computers_l1725_172593


namespace NUMINAMATH_GPT_rectangle_similarity_l1725_172520

structure Rectangle :=
(length : ℝ)
(width : ℝ)

def is_congruent (A B : Rectangle) : Prop :=
  A.length = B.length ∧ A.width = B.width

def is_similar (A B : Rectangle) : Prop :=
  A.length / A.width = B.length / B.width

theorem rectangle_similarity (A B : Rectangle)
  (h1 : ∀ P, is_congruent P A → ∃ Q, is_similar Q B)
  : ∀ P, is_congruent P B → ∃ Q, is_similar Q A :=
by sorry

end NUMINAMATH_GPT_rectangle_similarity_l1725_172520


namespace NUMINAMATH_GPT_total_packs_l1725_172529

theorem total_packs (cards_per_person : ℕ) (cards_per_pack : ℕ) (people_count : ℕ) (cards_per_person_eq : cards_per_person = 540) (cards_per_pack_eq : cards_per_pack = 20) (people_count_eq : people_count = 4) :
  (cards_per_person / cards_per_pack) * people_count = 108 :=
by
  sorry

end NUMINAMATH_GPT_total_packs_l1725_172529


namespace NUMINAMATH_GPT_age_problem_l1725_172570

theorem age_problem (F : ℝ) (M : ℝ) (Y : ℝ)
  (hF : F = 40.00000000000001)
  (hM : M = (2/5) * F)
  (hY : M + Y = (1/2) * (F + Y)) :
  Y = 8.000000000000002 :=
sorry

end NUMINAMATH_GPT_age_problem_l1725_172570


namespace NUMINAMATH_GPT_sum_xyz_le_two_l1725_172536

theorem sum_xyz_le_two (x y z : ℝ) (h : 2 * x + y^2 + z^2 ≤ 2) : x + y + z ≤ 2 :=
sorry

end NUMINAMATH_GPT_sum_xyz_le_two_l1725_172536


namespace NUMINAMATH_GPT_find_m_and_e_l1725_172507

theorem find_m_and_e (m e : ℕ) (hm : 0 < m) (he : e < 10) 
(h1 : 4 * m^2 + m + e = 346) 
(h2 : 4 * m^2 + m + 6 = 442 + 7 * e) : 
  m + e = 22 := by
  sorry

end NUMINAMATH_GPT_find_m_and_e_l1725_172507


namespace NUMINAMATH_GPT_sum_reciprocal_geo_seq_l1725_172537

theorem sum_reciprocal_geo_seq {a_5 a_6 a_7 a_8 : ℝ}
  (h_sum : a_5 + a_6 + a_7 + a_8 = 15 / 8)
  (h_prod : a_6 * a_7 = -9 / 8) :
  (1 / a_5) + (1 / a_6) + (1 / a_7) + (1 / a_8) = -5 / 3 := by
  sorry

end NUMINAMATH_GPT_sum_reciprocal_geo_seq_l1725_172537


namespace NUMINAMATH_GPT_bucket_water_l1725_172565

theorem bucket_water (oz1 oz2 oz3 oz4 oz5 total1 total2: ℕ) 
  (h1 : oz1 = 11)
  (h2 : oz2 = 13)
  (h3 : oz3 = 12)
  (h4 : oz4 = 16)
  (h5 : oz5 = 10)
  (h_total : total1 = oz1 + oz2 + oz3 + oz4 + oz5)
  (h_second_bucket : total2 = 39)
  : total1 - total2 = 23 :=
sorry

end NUMINAMATH_GPT_bucket_water_l1725_172565


namespace NUMINAMATH_GPT_largest_constant_C_l1725_172574

theorem largest_constant_C :
  ∃ C, C = 2 / Real.sqrt 3 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z) := sorry

end NUMINAMATH_GPT_largest_constant_C_l1725_172574


namespace NUMINAMATH_GPT_segment_length_is_13_l1725_172578

def point := (ℝ × ℝ)

def p1 : point := (2, 3)
def p2 : point := (7, 15)

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem segment_length_is_13 : distance p1 p2 = 13 := by
  sorry

end NUMINAMATH_GPT_segment_length_is_13_l1725_172578


namespace NUMINAMATH_GPT_solve_inequality_system_l1725_172581

theorem solve_inequality_system (x : ℝ) :
  (x + 2 < 3 * x) ∧ ((5 - x) / 2 + 1 < 0) → (x > 7) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l1725_172581


namespace NUMINAMATH_GPT_problem_goal_l1725_172548

-- Define the problem stating that there is a graph of points (x, y) satisfying the condition
def area_of_graph_satisfying_condition : Real :=
  let A := 2013
  -- Define the pairs (a, b) which are multiples of 2013
  let pairs := [(1, 2013), (3, 671), (11, 183), (33, 61)]
  -- Calculate the area of each region formed by pairs
  let area := pairs.length * 4
  area

-- Problem goal statement proving the area is equal to 16
theorem problem_goal : area_of_graph_satisfying_condition = 16 := by
  sorry

end NUMINAMATH_GPT_problem_goal_l1725_172548


namespace NUMINAMATH_GPT_smallest_n_l1725_172597

theorem smallest_n (n : ℕ) : 
  (n > 0 ∧ ((n^2 + n + 1)^2 > 1999) ∧ ∀ m : ℕ, (m > 0 ∧ (m^2 + m + 1)^2 > 1999) → m ≥ n) → n = 7 :=
sorry

end NUMINAMATH_GPT_smallest_n_l1725_172597


namespace NUMINAMATH_GPT_speed_of_water_l1725_172527

theorem speed_of_water (v : ℝ) (swim_speed_still_water : ℝ)
  (distance : ℝ) (time : ℝ)
  (h1 : swim_speed_still_water = 4) 
  (h2 : distance = 14) 
  (h3 : time = 7) 
  (h4 : 4 - v = distance / time) : 
  v = 2 := 
sorry

end NUMINAMATH_GPT_speed_of_water_l1725_172527


namespace NUMINAMATH_GPT_simple_interest_amount_is_58_l1725_172543

noncomputable def principal (CI : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  CI / ((1 + r / 100)^t - 1)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t / 100

theorem simple_interest_amount_is_58 (CI : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) :
  CI = 59.45 -> r = 5 -> t = 2 -> P = principal CI r t ->
  simple_interest P r t = 58 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_amount_is_58_l1725_172543


namespace NUMINAMATH_GPT_total_pieces_l1725_172567

def pieces_from_friend : ℕ := 123
def pieces_from_brother : ℕ := 136
def pieces_needed : ℕ := 117

theorem total_pieces :
  pieces_from_friend + pieces_from_brother + pieces_needed = 376 :=
by
  unfold pieces_from_friend pieces_from_brother pieces_needed
  sorry

end NUMINAMATH_GPT_total_pieces_l1725_172567


namespace NUMINAMATH_GPT_four_digit_perfect_square_l1725_172500

theorem four_digit_perfect_square : 
  ∃ (N : ℕ), (1000 ≤ N ∧ N ≤ 9999) ∧ (∃ (a b : ℕ), a = N / 1000 ∧ b = (N % 100) / 10 ∧ a = N / 100 - (N / 100 % 10) ∧ b = (N % 100 / 10) - N % 10) ∧ (∃ (n : ℕ), N = n * n) →
  N = 7744 := 
sorry

end NUMINAMATH_GPT_four_digit_perfect_square_l1725_172500


namespace NUMINAMATH_GPT_street_lights_per_side_l1725_172598

theorem street_lights_per_side
  (neighborhoods : ℕ)
  (roads_per_neighborhood : ℕ)
  (total_street_lights : ℕ)
  (total_neighborhoods : neighborhoods = 10)
  (roads_in_each_neighborhood : roads_per_neighborhood = 4)
  (street_lights_in_town : total_street_lights = 20000) :
  (total_street_lights / (neighborhoods * roads_per_neighborhood * 2) = 250) :=
by
  sorry

end NUMINAMATH_GPT_street_lights_per_side_l1725_172598


namespace NUMINAMATH_GPT_cost_price_is_3000_l1725_172544

variable (CP SP : ℝ)

-- Condition: selling price (SP) is 20% more than the cost price (CP)
def sellingPrice : ℝ := CP + 0.20 * CP

-- Condition: selling price (SP) is Rs. 3600
axiom selling_price_eq : SP = 3600

-- Given the above conditions, prove that the cost price (CP) is Rs. 3000
theorem cost_price_is_3000 (h : sellingPrice CP = SP) : CP = 3000 := by
  sorry

end NUMINAMATH_GPT_cost_price_is_3000_l1725_172544


namespace NUMINAMATH_GPT_total_rope_in_inches_l1725_172559

-- Definitions for conditions
def feet_last_week : ℕ := 6
def feet_less : ℕ := 4
def inches_per_foot : ℕ := 12

-- Condition: rope bought this week
def feet_this_week := feet_last_week - feet_less

-- Condition: total rope bought in feet
def total_feet := feet_last_week + feet_this_week

-- Condition: total rope bought in inches
def total_inches := total_feet * inches_per_foot

-- Theorem statement
theorem total_rope_in_inches : total_inches = 96 := by
  sorry

end NUMINAMATH_GPT_total_rope_in_inches_l1725_172559


namespace NUMINAMATH_GPT_equation_of_lamps_l1725_172557

theorem equation_of_lamps (n k : ℕ) (N M : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : k ≥ n) (h4 : (k - n) % 2 = 0) : 
  N = 2^(k - n) * M := 
sorry

end NUMINAMATH_GPT_equation_of_lamps_l1725_172557


namespace NUMINAMATH_GPT_parabola1_right_of_parabola2_l1725_172584

def parabola1 (x : ℝ) : ℝ := x^2 - x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + x + 5

theorem parabola1_right_of_parabola2 :
  ∃ x1 x2 : ℝ, x1 > x2 ∧ parabola1 x1 < parabola2 x2 :=
by
  sorry

end NUMINAMATH_GPT_parabola1_right_of_parabola2_l1725_172584


namespace NUMINAMATH_GPT_range_of_a_l1725_172545

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 2 < a ∧ a ≤ 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l1725_172545


namespace NUMINAMATH_GPT_students_per_minibus_calculation_l1725_172517

-- Define the conditions
variables (vans minibusses total_students students_per_van : ℕ)
variables (students_per_minibus : ℕ)

-- Define the given conditions based on the problem
axiom six_vans : vans = 6
axiom four_minibusses : minibusses = 4
axiom ten_students_per_van : students_per_van = 10
axiom total_students_are_156 : total_students = 156

-- Define the problem statement in Lean
theorem students_per_minibus_calculation
  (h1 : vans = 6)
  (h2 : minibusses = 4)
  (h3 : students_per_van = 10)
  (h4 : total_students = 156) :
  students_per_minibus = 24 :=
sorry

end NUMINAMATH_GPT_students_per_minibus_calculation_l1725_172517


namespace NUMINAMATH_GPT_max_sum_pyramid_on_hexagonal_face_l1725_172558

structure hexagonal_prism :=
(faces_initial : ℕ)
(vertices_initial : ℕ)
(edges_initial : ℕ)

structure pyramid_added :=
(faces_total : ℕ)
(vertices_total : ℕ)
(edges_total : ℕ)
(total_sum : ℕ)

theorem max_sum_pyramid_on_hexagonal_face (h : hexagonal_prism) :
  (h = ⟨8, 12, 18⟩) →
  ∃ p : pyramid_added, 
    p = ⟨13, 13, 24, 50⟩ :=
by
  sorry

end NUMINAMATH_GPT_max_sum_pyramid_on_hexagonal_face_l1725_172558


namespace NUMINAMATH_GPT_cortney_downloads_all_files_in_2_hours_l1725_172552

theorem cortney_downloads_all_files_in_2_hours :
  let speed := 2 -- internet speed in megabits per minute
  let file1 := 80 -- file size in megabits
  let file2 := 90 -- file size in megabits
  let file3 := 70 -- file size in megabits
  let time1 := file1 / speed -- time to download first file in minutes
  let time2 := file2 / speed -- time to download second file in minutes
  let time3 := file3 / speed -- time to download third file in minutes
  let total_time_minutes := time1 + time2 + time3
  let total_time_hours := total_time_minutes / 60
  total_time_hours = 2 :=
by
  sorry

end NUMINAMATH_GPT_cortney_downloads_all_files_in_2_hours_l1725_172552


namespace NUMINAMATH_GPT_tv_horizontal_length_l1725_172585

noncomputable def rectangleTvLengthRatio (l h : ℝ) : Prop :=
  l / h = 16 / 9

noncomputable def rectangleTvDiagonal (l h d : ℝ) : Prop :=
  l^2 + h^2 = d^2

theorem tv_horizontal_length
  (h : ℝ)
  (h_positive : h > 0)
  (d : ℝ)
  (h_ratio : rectangleTvLengthRatio l h)
  (h_diagonal : rectangleTvDiagonal l h d)
  (h_diagonal_value : d = 36) :
  l = 56.27 :=
by
  sorry

end NUMINAMATH_GPT_tv_horizontal_length_l1725_172585


namespace NUMINAMATH_GPT_not_p_and_not_q_true_l1725_172568

variable (p q: Prop)

theorem not_p_and_not_q_true (h1: ¬ (p ∧ q)) (h2: ¬ (p ∨ q)) : ¬ p ∧ ¬ q :=
by
  sorry

end NUMINAMATH_GPT_not_p_and_not_q_true_l1725_172568


namespace NUMINAMATH_GPT_largest_possible_a_l1725_172589

theorem largest_possible_a :
  ∀ (a b c d : ℕ), a < 3 * b ∧ b < 4 * c ∧ c < 5 * d ∧ d < 80 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d → a ≤ 4724 := by
  sorry

end NUMINAMATH_GPT_largest_possible_a_l1725_172589


namespace NUMINAMATH_GPT_power_of_thousand_l1725_172506

-- Define the notion of googol
def googol := 10^100

-- Prove that 1000^100 is equal to googol^3
theorem power_of_thousand : (1000 ^ 100) = googol^3 := by
  -- proof step to be filled here
  sorry

end NUMINAMATH_GPT_power_of_thousand_l1725_172506


namespace NUMINAMATH_GPT_find_a_l1725_172571

variables {a b c : ℂ}

-- Given conditions
variables (h1 : a + b + c = 5) 
variables (h2 : a * b + b * c + c * a = 5) 
variables (h3 : a * b * c = 5)
variables (h4 : a.im = 0) -- a is real

theorem find_a : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1725_172571


namespace NUMINAMATH_GPT_range_of_m_l1725_172556

open Set Real

-- Define over the real numbers ℝ
noncomputable def A : Set ℝ := { x : ℝ | x^2 - 2*x - 3 ≤ 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x : ℝ | x^2 - 2*m*x + m^2 - 4 ≤ 0 }
noncomputable def CRB (m : ℝ) : Set ℝ := { x : ℝ | x < m - 2 ∨ x > m + 2 }

-- Main theorem statement
theorem range_of_m (m : ℝ) (h : A ⊆ CRB m) : m < -3 ∨ m > 5 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1725_172556


namespace NUMINAMATH_GPT_even_three_digit_numbers_count_l1725_172594

theorem even_three_digit_numbers_count :
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  count = 18 :=
by
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  show count = 18
  sorry

end NUMINAMATH_GPT_even_three_digit_numbers_count_l1725_172594


namespace NUMINAMATH_GPT_gcd_2048_2101_eq_1_l1725_172561

theorem gcd_2048_2101_eq_1 : Int.gcd 2048 2101 = 1 := sorry

end NUMINAMATH_GPT_gcd_2048_2101_eq_1_l1725_172561


namespace NUMINAMATH_GPT_perpendicular_vectors_eq_l1725_172577

theorem perpendicular_vectors_eq {x : ℝ} (h : (x - 5) * 2 + 3 * x = 0) : x = 2 :=
sorry

end NUMINAMATH_GPT_perpendicular_vectors_eq_l1725_172577


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1725_172540

theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 / 4 - y^2 = 1) → (y = x / 2 ∨ y = -x / 2) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1725_172540


namespace NUMINAMATH_GPT_problem_1_problem_2_l1725_172504

-- Definitions according to the conditions
def f (x a : ℝ) := |2 * x + a| + |x - 2|

-- The first part of the problem: Proof when a = -4, solve f(x) >= 6
theorem problem_1 (x : ℝ) : 
  f x (-4) ≥ 6 ↔ x ≤ 0 ∨ x ≥ 4 := by
  sorry

-- The second part of the problem: Prove the range of a for inequality f(x) >= 3a^2 - |2 - x|
theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3 * a^2 - |2 - x|) ↔ (-1 ≤ a ∧ a ≤ 4 / 3) := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1725_172504


namespace NUMINAMATH_GPT_six_digit_number_l1725_172576

noncomputable def number_of_digits (N : ℕ) : ℕ := sorry

theorem six_digit_number :
  ∀ (N : ℕ),
    (N % 2020 = 0) ∧
    (∀ a b : ℕ, (a ≠ b ∧ N / 10^a % 10 ≠ N / 10^b % 10)) ∧
    (∀ a b : ℕ, (a ≠ b) → ((N / 10^a % 10 = N / 10^b % 10) -> (N % 2020 ≠ 0))) →
    number_of_digits N = 6 :=
sorry

end NUMINAMATH_GPT_six_digit_number_l1725_172576


namespace NUMINAMATH_GPT_range_of_k_l1725_172566

noncomputable def h (x : ℝ) (k : ℝ) : ℝ := 2 * x - k / x + k / 3

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → 2 + k / x^2 > 0) ↔ k ≥ -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1725_172566


namespace NUMINAMATH_GPT_compute_value_l1725_172562

def Δ (p q : ℕ) : ℕ := p^3 - q

theorem compute_value : Δ (5^Δ 2 7) (4^Δ 4 8) = 125 - 4^56 := by
  sorry

end NUMINAMATH_GPT_compute_value_l1725_172562


namespace NUMINAMATH_GPT_hour_division_convenience_dozen_division_convenience_l1725_172508

theorem hour_division_convenience :
  ∃ (a b c d e f g h i j : ℕ), 
  60 = 2 * a ∧
  60 = 3 * b ∧
  60 = 4 * c ∧
  60 = 5 * d ∧
  60 = 6 * e ∧
  60 = 10 * f ∧
  60 = 12 * g ∧
  60 = 15 * h ∧
  60 = 20 * i ∧
  60 = 30 * j := by
  -- to be filled with a proof later
  sorry

theorem dozen_division_convenience :
  ∃ (a b c d : ℕ),
  12 = 2 * a ∧
  12 = 3 * b ∧
  12 = 4 * c ∧
  12 = 6 * d := by
  -- to be filled with a proof later
  sorry

end NUMINAMATH_GPT_hour_division_convenience_dozen_division_convenience_l1725_172508


namespace NUMINAMATH_GPT_Isabella_exchange_l1725_172546

/-
Conditions:
1. Isabella exchanged d U.S. dollars to receive (8/5)d Canadian dollars.
2. After spending 80 Canadian dollars, she had d + 20 Canadian dollars left.
3. Sum of the digits of d is 14.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (.+.) 0

theorem Isabella_exchange (d : ℕ) (h : (8 * d / 5) - 80 = d + 20) : sum_of_digits d = 14 :=
by sorry

end NUMINAMATH_GPT_Isabella_exchange_l1725_172546


namespace NUMINAMATH_GPT_num_comics_bought_l1725_172539

def initial_comic_books : ℕ := 14
def current_comic_books : ℕ := 13
def comic_books_sold (initial : ℕ) : ℕ := initial / 2
def comics_bought (initial current : ℕ) : ℕ :=
  current - (initial - comic_books_sold initial)

theorem num_comics_bought :
  comics_bought initial_comic_books current_comic_books = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_comics_bought_l1725_172539


namespace NUMINAMATH_GPT_calculate_total_customers_l1725_172573

theorem calculate_total_customers 
    (num_no_tip : ℕ) 
    (total_tip_amount : ℕ) 
    (tip_per_customer : ℕ) 
    (number_tipped_customers : ℕ) 
    (number_total_customers : ℕ)
    (h1 : num_no_tip = 5) 
    (h2 : total_tip_amount = 15) 
    (h3 : tip_per_customer = 3) 
    (h4 : number_tipped_customers = total_tip_amount / tip_per_customer) :
    number_total_customers = number_tipped_customers + num_no_tip := 
by {
    sorry
}

end NUMINAMATH_GPT_calculate_total_customers_l1725_172573


namespace NUMINAMATH_GPT_train_speed_is_64_98_kmph_l1725_172582

noncomputable def train_length : ℝ := 200
noncomputable def bridge_length : ℝ := 180
noncomputable def passing_time : ℝ := 21.04615384615385
noncomputable def speed_in_kmph : ℝ := 3.6 * (train_length + bridge_length) / passing_time

theorem train_speed_is_64_98_kmph : abs (speed_in_kmph - 64.98) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_64_98_kmph_l1725_172582


namespace NUMINAMATH_GPT_miniature_tower_height_l1725_172519

-- Definitions of conditions
def actual_tower_height := 60
def actual_dome_volume := 200000 -- in liters
def miniature_dome_volume := 0.4 -- in liters

-- Goal: Prove the height of the miniature tower
theorem miniature_tower_height
  (actual_tower_height: ℝ)
  (actual_dome_volume: ℝ)
  (miniature_dome_volume: ℝ) : 
  actual_tower_height = 60 ∧ actual_dome_volume = 200000 ∧ miniature_dome_volume = 0.4 →
  (actual_tower_height / ( (actual_dome_volume / miniature_dome_volume)^(1/3) )) = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_miniature_tower_height_l1725_172519


namespace NUMINAMATH_GPT_toy_discount_price_l1725_172551

theorem toy_discount_price (original_price : ℝ) (discount_rate : ℝ) (price_after_first_discount : ℝ) (price_after_second_discount : ℝ) : 
  original_price = 200 → 
  discount_rate = 0.1 →
  price_after_first_discount = original_price * (1 - discount_rate) →
  price_after_second_discount = price_after_first_discount * (1 - discount_rate) →
  price_after_second_discount = 162 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_toy_discount_price_l1725_172551


namespace NUMINAMATH_GPT_inequality_solution_set_l1725_172533

theorem inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x, x ∈ Set.Icc (-3) 1 → ax^2 + (a + b)*x + 2 > 0) : 
  a + b = -4/3 := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1725_172533


namespace NUMINAMATH_GPT_geometric_sum_l1725_172503

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sum (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 6) (h3 : a 3 = -18) :
  a 1 + a 2 + a 3 + a 4 = 40 :=
sorry

end NUMINAMATH_GPT_geometric_sum_l1725_172503


namespace NUMINAMATH_GPT_output_for_input_8_is_8_over_65_l1725_172596

def function_f (n : ℕ) : ℚ := n / (n^2 + 1)

theorem output_for_input_8_is_8_over_65 : function_f 8 = 8 / 65 := by
  sorry

end NUMINAMATH_GPT_output_for_input_8_is_8_over_65_l1725_172596


namespace NUMINAMATH_GPT_total_games_played_l1725_172534

def number_of_games (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem total_games_played :
  number_of_games 9 2 = 36 :=
by
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_total_games_played_l1725_172534


namespace NUMINAMATH_GPT_heather_shared_blocks_l1725_172587

-- Define the initial number of blocks Heather starts with
def initial_blocks : ℕ := 86

-- Define the final number of blocks Heather ends up with
def final_blocks : ℕ := 45

-- Define the number of blocks Heather shared
def blocks_shared (initial final : ℕ) : ℕ := initial - final

-- Prove that the number of blocks Heather shared is 41
theorem heather_shared_blocks : blocks_shared initial_blocks final_blocks = 41 := by
  -- Proof steps will be added here
  sorry

end NUMINAMATH_GPT_heather_shared_blocks_l1725_172587


namespace NUMINAMATH_GPT_largest_integer_less_than_100_with_remainder_4_l1725_172538

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end NUMINAMATH_GPT_largest_integer_less_than_100_with_remainder_4_l1725_172538


namespace NUMINAMATH_GPT_geometric_progression_arcsin_sin_l1725_172583

noncomputable def least_positive_t : ℝ :=
  9 + 4 * Real.sqrt 5

theorem geometric_progression_arcsin_sin 
  (α : ℝ) 
  (hα1: 0 < α) 
  (hα2: α < Real.pi / 2) 
  (t : ℝ) 
  (h : ∀ (a b c d : ℝ), 
    a = Real.arcsin (Real.sin α) ∧ 
    b = Real.arcsin (Real.sin (3 * α)) ∧ 
    c = Real.arcsin (Real.sin (5 * α)) ∧ 
    d = Real.arcsin (Real.sin (t * α)) → 
    b / a = c / b ∧ c / b = d / c) : 
  t = least_positive_t :=
sorry

end NUMINAMATH_GPT_geometric_progression_arcsin_sin_l1725_172583


namespace NUMINAMATH_GPT_amount_spent_on_drink_l1725_172502

-- Definitions based on conditions provided
def initialAmount : ℝ := 9
def remainingAmount : ℝ := 6
def additionalSpending : ℝ := 1.25

-- Theorem to prove the amount spent on the drink
theorem amount_spent_on_drink : 
  initialAmount - remainingAmount - additionalSpending = 1.75 := 
by 
  sorry

end NUMINAMATH_GPT_amount_spent_on_drink_l1725_172502


namespace NUMINAMATH_GPT_least_amount_of_money_l1725_172526

variable (Money : Type) [LinearOrder Money]
variable (Anne Bo Coe Dan El : Money)

-- Conditions from the problem
axiom anne_less_than_bo : Anne < Bo
axiom dan_less_than_bo : Dan < Bo
axiom coe_less_than_anne : Coe < Anne
axiom coe_less_than_el : Coe < El
axiom coe_less_than_dan : Coe < Dan
axiom dan_less_than_anne : Dan < Anne

theorem least_amount_of_money : (∀ x, x = Anne ∨ x = Bo ∨ x = Coe ∨ x = Dan ∨ x = El → Coe < x) :=
by
  sorry

end NUMINAMATH_GPT_least_amount_of_money_l1725_172526


namespace NUMINAMATH_GPT_max_area_of_triangle_l1725_172521

theorem max_area_of_triangle :
  ∀ (O O' : EuclideanSpace ℝ (Fin 2)) (M : EuclideanSpace ℝ (Fin 2)),
  dist O O' = 2014 →
  dist O M = 1 ∨ dist O' M = 1 →
  ∃ (A : ℝ), A = 1007 :=
by
  intros O O' M h₁ h₂
  sorry

end NUMINAMATH_GPT_max_area_of_triangle_l1725_172521


namespace NUMINAMATH_GPT_original_bananas_total_l1725_172530

theorem original_bananas_total (willie_bananas : ℝ) (charles_bananas : ℝ) : willie_bananas = 48.0 → charles_bananas = 35.0 → willie_bananas + charles_bananas = 83.0 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_original_bananas_total_l1725_172530


namespace NUMINAMATH_GPT_mittens_per_box_l1725_172532

theorem mittens_per_box (boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) (h_boxes : boxes = 7) (h_scarves : scarves_per_box = 3) (h_total : total_clothing = 49) : 
  let total_scarves := boxes * scarves_per_box
  let total_mittens := total_clothing - total_scarves
  total_mittens / boxes = 4 :=
by
  sorry

end NUMINAMATH_GPT_mittens_per_box_l1725_172532


namespace NUMINAMATH_GPT_problem_I_problem_II_l1725_172523

noncomputable def f (a x : ℝ) : ℝ := x^2 - (2 * a + 1) * x + a * Real.log x
noncomputable def g (a x : ℝ) : ℝ := (1 - a) * x
noncomputable def h (x : ℝ) : ℝ := (x^2 - 2 * x) / (x - Real.log x)

theorem problem_I (a : ℝ) (ha : a > 1 / 2) :
  (∀ x : ℝ, 0 < x ∧ x < 1 / 2 → deriv (f a) x > 0) ∧
  (∀ x : ℝ, 1 / 2 < x ∧ x < a → deriv (f a) x < 0) ∧
  (∀ x : ℝ, a < x → deriv (f a) x > 0) :=
sorry

theorem problem_II (a : ℝ) :
  (∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ Real.exp 1 ∧ f a x₀ ≥ g a x₀) ↔ a ≤ (Real.exp 1 * (Real.exp 1 - 2)) / (Real.exp 1 - 1) :=
sorry

end NUMINAMATH_GPT_problem_I_problem_II_l1725_172523


namespace NUMINAMATH_GPT_ladder_base_distance_l1725_172542

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_ladder_base_distance_l1725_172542


namespace NUMINAMATH_GPT_color_of_217th_marble_l1725_172514

-- Definitions of conditions
def total_marbles := 240
def pattern_length := 15
def red_marbles := 6
def blue_marbles := 5
def green_marbles := 4
def position := 217

-- Lean 4 statement
theorem color_of_217th_marble :
  (position % pattern_length ≤ red_marbles) :=
by sorry

end NUMINAMATH_GPT_color_of_217th_marble_l1725_172514


namespace NUMINAMATH_GPT_find_fx_l1725_172595

theorem find_fx (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = x^2 - 2 * x) : ∀ x, f x = x^2 - 1 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_find_fx_l1725_172595


namespace NUMINAMATH_GPT_constant_COG_of_mercury_column_l1725_172591

theorem constant_COG_of_mercury_column (L : ℝ) (A : ℝ) (beta_g : ℝ) (beta_m : ℝ) (alpha_g : ℝ) (x : ℝ) :
  L = 1 ∧ A = 1e-4 ∧ beta_g = 1 / 38700 ∧ beta_m = 1 / 5550 ∧ alpha_g = beta_g / 3 ∧
  x = (2 / (3 * 38700)) / ((1 / 5550) - (2 / 116100)) →
  x = 0.106 :=
by
  sorry

end NUMINAMATH_GPT_constant_COG_of_mercury_column_l1725_172591


namespace NUMINAMATH_GPT_find_number_l1725_172505

-- Assume the necessary definitions and conditions
variable (x : ℝ)

-- Sixty-five percent of the number is 21 less than four-fifths of the number
def condition := 0.65 * x = 0.8 * x - 21

-- Final proof goal: We need to prove that the number x is 140
theorem find_number (h : condition x) : x = 140 := by
  sorry

end NUMINAMATH_GPT_find_number_l1725_172505


namespace NUMINAMATH_GPT_find_two_digit_number_l1725_172555

open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_two_digit_number :
  ∃ N : ℕ, 
    (10 ≤ N ∧ N < 100) ∧ 
    (N % 2 = 1) ∧ 
    (N % 9 = 0) ∧ 
    is_perfect_square ((N / 10) * (N % 10)) ∧ 
    N = 99 :=
by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l1725_172555


namespace NUMINAMATH_GPT_smallest_brownie_pan_size_l1725_172588

theorem smallest_brownie_pan_size :
  ∃ s : ℕ, (s - 2) ^ 2 = 4 * s - 4 ∧ ∀ t : ℕ, (t - 2) ^ 2 = 4 * t - 4 → s <= t :=
by
  sorry

end NUMINAMATH_GPT_smallest_brownie_pan_size_l1725_172588


namespace NUMINAMATH_GPT_ratio_of_ages_l1725_172560

theorem ratio_of_ages (a b : ℕ) (h1 : a - 10 = (b - 10) / 2) (h2 : a + b = 35) : a / gcd a b = 3 ∧ b / gcd a b = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l1725_172560


namespace NUMINAMATH_GPT_problem_I_solution_set_problem_II_range_a_l1725_172510

-- Problem (I)
-- Given f(x) = |x-1|, g(x) = 2|x+1|, and a=1, prove that the inequality f(x) - g(x) > 1 has the solution set (-1, -1/3)
theorem problem_I_solution_set (x: ℝ) : abs (x - 1) - 2 * abs (x + 1) > 1 ↔ -1 < x ∧ x < -1 / 3 := 
by sorry

-- Problem (II)
-- Given f(x) = |x-1|, g(x) = 2|x+a|, prove that if 2f(x) + g(x) ≤ (a + 1)^2 has a solution for x,
-- then a ∈ (-∞, -3] ∪ [1, ∞)
theorem problem_II_range_a (a x: ℝ) (h : ∃ x, 2 * abs (x - 1) + 2 * abs (x + a) ≤ (a + 1) ^ 2) : 
  a ≤ -3 ∨ a ≥ 1 := 
by sorry

end NUMINAMATH_GPT_problem_I_solution_set_problem_II_range_a_l1725_172510


namespace NUMINAMATH_GPT_correct_order_option_C_l1725_172590

def length_unit_ordered (order : List String) : Prop :=
  order = ["kilometer", "meter", "centimeter", "millimeter"]

def option_A := ["kilometer", "meter", "millimeter", "centimeter"]
def option_B := ["meter", "kilometer", "centimeter", "millimeter"]
def option_C := ["kilometer", "meter", "centimeter", "millimeter"]

theorem correct_order_option_C : length_unit_ordered option_C := by
  sorry

end NUMINAMATH_GPT_correct_order_option_C_l1725_172590


namespace NUMINAMATH_GPT_trajectory_of_M_l1725_172511

theorem trajectory_of_M
  (x y : ℝ)
  (h : Real.sqrt ((x + 5)^2 + y^2) - Real.sqrt ((x - 5)^2 + y^2) = 8) :
  (x^2 / 16) - (y^2 / 9) = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_of_M_l1725_172511


namespace NUMINAMATH_GPT_paper_cups_calculation_l1725_172586

def total_pallets : Nat := 20
def paper_towels : Nat := total_pallets / 2
def tissues : Nat := total_pallets / 4
def paper_plates : Nat := total_pallets / 5
def other_paper_products : Nat := paper_towels + tissues + paper_plates
def paper_cups : Nat := total_pallets - other_paper_products

theorem paper_cups_calculation : paper_cups = 1 := by
  sorry

end NUMINAMATH_GPT_paper_cups_calculation_l1725_172586


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1725_172501

theorem sum_of_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (Polynomial.eval x1 (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-3) * Polynomial.X + Polynomial.C (-4)) = 0) ∧ 
                 (Polynomial.eval x2 (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-3) * Polynomial.X + Polynomial.C (-4)) = 0) -> 
                 x1 + x2 = 3 := 
by
  intro x1 x2
  intro H
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1725_172501


namespace NUMINAMATH_GPT_solve_system_l1725_172572

theorem solve_system :
  ∃ x y z : ℝ, (8 * (x^3 + y^3 + z^3) = 73) ∧
              (2 * (x^2 + y^2 + z^2) = 3 * (x * y + y * z + z * x)) ∧
              (x * y * z = 1) ∧
              (x, y, z) = (1, 2, 0.5) ∨ (x, y, z) = (1, 0.5, 2) ∨
              (x, y, z) = (2, 1, 0.5) ∨ (x, y, z) = (2, 0.5, 1) ∨
              (x, y, z) = (0.5, 1, 2) ∨ (x, y, z) = (0.5, 2, 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1725_172572


namespace NUMINAMATH_GPT_Sandy_fingernails_reach_world_record_in_20_years_l1725_172518

-- Definitions for the conditions of the problem
def world_record_len : ℝ := 26
def current_len : ℝ := 2
def growth_rate : ℝ := 0.1

-- Proof goal
theorem Sandy_fingernails_reach_world_record_in_20_years :
  (world_record_len - current_len) / growth_rate / 12 = 20 :=
by
  sorry

end NUMINAMATH_GPT_Sandy_fingernails_reach_world_record_in_20_years_l1725_172518


namespace NUMINAMATH_GPT_island_length_l1725_172580

theorem island_length (area width : ℝ) (h_area : area = 50) (h_width : width = 5) : 
  area / width = 10 := 
by
  sorry

end NUMINAMATH_GPT_island_length_l1725_172580


namespace NUMINAMATH_GPT_calc_mod_residue_l1725_172569

theorem calc_mod_residue :
  (245 * 15 - 18 * 8 + 5) % 17 = 0 := by
  sorry

end NUMINAMATH_GPT_calc_mod_residue_l1725_172569


namespace NUMINAMATH_GPT_find_real_number_a_l1725_172512

theorem find_real_number_a (a : ℝ) (h : (a^2 - 3*a + 2 = 0)) (h' : (a - 2) ≠ 0) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_real_number_a_l1725_172512


namespace NUMINAMATH_GPT_seating_arrangement_l1725_172579

theorem seating_arrangement (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 3) : 
  (∃ n : ℕ, n = (boys.factorial * girls.factorial) + (girls.factorial * boys.factorial) ∧ n = 288) :=
by 
  sorry

end NUMINAMATH_GPT_seating_arrangement_l1725_172579


namespace NUMINAMATH_GPT_simplify_expression_l1725_172554

theorem simplify_expression (p : ℝ) : ((7 * p - 3) - 3 * p * 2) * 2 + (5 - 2 / 2) * (8 * p - 12) = 34 * p - 54 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1725_172554


namespace NUMINAMATH_GPT_union_A_B_l1725_172522

def A : Set ℝ := {x | ∃ y : ℝ, y = Real.log x}
def B : Set ℝ := {x | x < 1}

theorem union_A_B : (A ∪ B) = Set.univ :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_l1725_172522


namespace NUMINAMATH_GPT_abs_lt_one_sufficient_not_necessary_l1725_172553

theorem abs_lt_one_sufficient_not_necessary (x : ℝ) : (|x| < 1) -> (x < 1) ∧ ¬(x < 1 -> |x| < 1) :=
by
  sorry

end NUMINAMATH_GPT_abs_lt_one_sufficient_not_necessary_l1725_172553


namespace NUMINAMATH_GPT_trigonometric_identity_l1725_172563

theorem trigonometric_identity (α : ℝ) :
    1 - 1/4 * (Real.sin (2 * α)) ^ 2 + Real.cos (2 * α) = (Real.cos α) ^ 2 + (Real.cos α) ^ 4 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1725_172563


namespace NUMINAMATH_GPT_problem1_inner_problem2_inner_l1725_172541

-- Problem 1
theorem problem1_inner {m n : ℤ} (hm : |m| = 5) (hn : |n| = 4) (opposite_signs : m * n < 0) :
  m^2 - m * n + n = 41 ∨ m^2 - m * n + n = 49 :=
sorry

-- Problem 2
theorem problem2_inner {a b c d x : ℝ} (opposite_ab : a + b = 0) (reciprocals_cd : c * d = 1) (hx : |x| = 5) (hx_pos : x > 0) :
  3 * (a + b) - 2 * (c * d) + x = 3 :=
sorry

end NUMINAMATH_GPT_problem1_inner_problem2_inner_l1725_172541


namespace NUMINAMATH_GPT_min_value_expression_l1725_172531

theorem min_value_expression (a b: ℝ) (h : 2 * a + b = 1) : (a - 1) ^ 2 + (b - 1) ^ 2 = 4 / 5 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1725_172531
