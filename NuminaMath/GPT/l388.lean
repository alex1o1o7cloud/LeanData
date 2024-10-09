import Mathlib

namespace time_to_cross_man_l388_38804

-- Define the conversion from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℕ := (speed_kmh * 1000) / 3600

-- Given conditions
def length_of_train : ℕ := 150
def speed_of_train_kmh : ℕ := 180

-- Calculate speed in m/s
def speed_of_train_ms : ℕ := kmh_to_ms speed_of_train_kmh

-- Proof problem statement
theorem time_to_cross_man : (length_of_train : ℕ) / (speed_of_train_ms : ℕ) = 3 := by
  sorry

end time_to_cross_man_l388_38804


namespace sum_of_altitudes_is_less_than_perimeter_l388_38897

theorem sum_of_altitudes_is_less_than_perimeter 
  (a b c h_a h_b h_c : ℝ) 
  (h_a_le_b : h_a ≤ b) 
  (h_b_le_c : h_b ≤ c) 
  (h_c_le_a : h_c ≤ a) 
  (strict_inequality : h_a < b ∨ h_b < c ∨ h_c < a) : h_a + h_b + h_c < a + b + c := 
by 
  sorry

end sum_of_altitudes_is_less_than_perimeter_l388_38897


namespace simplify_and_evaluate_l388_38879

theorem simplify_and_evaluate (a b : ℝ) (h : a - 2 * b = -1) :
  -3 * a * (a - 2 * b)^5 + 6 * b * (a - 2 * b)^5 - 5 * (-a + 2 * b)^3 = -8 :=
by
  sorry

end simplify_and_evaluate_l388_38879


namespace twelfth_term_of_geometric_sequence_l388_38823

theorem twelfth_term_of_geometric_sequence 
  (a : ℕ → ℕ)
  (h₁ : a 4 = 4)
  (h₂ : a 7 = 32)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * r) : 
  a 12 = 1024 :=
sorry

end twelfth_term_of_geometric_sequence_l388_38823


namespace arithmetic_geometric_sequence_l388_38899

theorem arithmetic_geometric_sequence :
  ∀ (a : ℕ → ℕ) (b : ℕ → ℕ),
    (a 1 + a 2 = 10) →
    (a 4 - a 3 = 2) →
    (b 2 = a 3) →
    (b 3 = a 7) →
    a 15 = b 4 :=
by
  intros a b h1 h2 h3 h4
  sorry

end arithmetic_geometric_sequence_l388_38899


namespace solve_system_of_equations_l388_38858

theorem solve_system_of_equations :
  (∃ x y : ℚ, 2 * x + 4 * y = 9 ∧ 3 * x - 5 * y = 8) ↔ 
  (∃ x y : ℚ, x = 7 / 2 ∧ y = 1 / 2) := by
  sorry

end solve_system_of_equations_l388_38858


namespace find_m_l388_38866

theorem find_m (m : ℕ) (hm : 0 < m)
  (a : ℕ := Nat.choose (2 * m) m)
  (b : ℕ := Nat.choose (2 * m + 1) m)
  (h : 13 * a = 7 * b) : m = 6 := by
  sorry

end find_m_l388_38866


namespace inequality_solution_set_l388_38863

theorem inequality_solution_set (a : ℝ) : (∀ x : ℝ, x > 5 ∧ x > a ↔ x > 5) → a ≤ 5 :=
by
  sorry

end inequality_solution_set_l388_38863


namespace zorbs_of_60_deg_l388_38892

-- Define the measurement on Zorblat
def zorbs_in_full_circle := 600
-- Define the Earth angle in degrees
def earth_degrees_full_circle := 360
def angle_in_degrees := 60
-- Calculate the equivalent angle in zorbs
def zorbs_in_angle := zorbs_in_full_circle * angle_in_degrees / earth_degrees_full_circle

theorem zorbs_of_60_deg (h1 : zorbs_in_full_circle = 600)
                        (h2 : earth_degrees_full_circle = 360)
                        (h3 : angle_in_degrees = 60) :
  zorbs_in_angle = 100 :=
by sorry

end zorbs_of_60_deg_l388_38892


namespace dice_product_probability_l388_38859

def is_valid_die_value (n : ℕ) : Prop := n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)

theorem dice_product_probability :
  ∃ (a b c : ℕ), is_valid_die_value a ∧ is_valid_die_value b ∧ is_valid_die_value c ∧ 
  a * b * c = 8 ∧ 
  (1 / 6 : ℝ) * (1 / 6) * (1 / 6) * (6 + 1) = (7 / 216 : ℝ) :=
sorry

end dice_product_probability_l388_38859


namespace find_d_l388_38830

theorem find_d (a b c d : ℝ) (h : a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c - d + 3)) : 
  d = 13 / 4 :=
sorry

end find_d_l388_38830


namespace travel_time_equation_l388_38835

theorem travel_time_equation (x : ℝ) (h1 : ∀ d : ℝ, d > 0) :
  (x / 160) - (x / 200) = 2.5 :=
sorry

end travel_time_equation_l388_38835


namespace fourth_student_guess_l388_38810

theorem fourth_student_guess :
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let total := first_guess + second_guess + third_guess
  let average := total / 3
  let fourth_guess := average + 25
  fourth_guess = 525 :=
by
  sorry

end fourth_student_guess_l388_38810


namespace symmetric_points_sum_l388_38868

theorem symmetric_points_sum (a b : ℝ) (hA1 : A = (a, 1)) (hB1 : B = (5, b))
    (h_symmetric : (a, 1) = -(5, b)) : a + b = -6 :=
by
  sorry

end symmetric_points_sum_l388_38868


namespace evaluate_expression_l388_38832

noncomputable def repeating_to_fraction_06 : ℚ := 2 / 3
noncomputable def repeating_to_fraction_02 : ℚ := 2 / 9
noncomputable def repeating_to_fraction_04 : ℚ := 4 / 9

theorem evaluate_expression : 
  ((repeating_to_fraction_06 * repeating_to_fraction_02) - repeating_to_fraction_04) = -8 / 27 := 
by 
  sorry

end evaluate_expression_l388_38832


namespace real_part_of_z_l388_38813

variable (z : ℂ) (a : ℝ)

noncomputable def condition1 : Prop := z / (2 + (a : ℂ) * Complex.I) = 2 / (1 + Complex.I)
noncomputable def condition2 : Prop := z.im = -3

theorem real_part_of_z (h1 : condition1 z a) (h2 : condition2 z) : z.re = 1 := sorry

end real_part_of_z_l388_38813


namespace identical_digits_divisible_l388_38898

  theorem identical_digits_divisible (n : ℕ) (hn : n > 0) : 
    ∀ a : ℕ, (10^(3^n - 1) * a / 9) % 3^n = 0 := 
  by
    intros
    sorry
  
end identical_digits_divisible_l388_38898


namespace maximum_marks_l388_38836

theorem maximum_marks (M : ℝ) (mark_obtained failed_by : ℝ) (pass_percentage : ℝ) 
  (h1 : pass_percentage = 0.6) (h2 : mark_obtained = 250) (h3 : failed_by = 50) :
  (pass_percentage * M = mark_obtained + failed_by) → M = 500 :=
by 
  sorry

end maximum_marks_l388_38836


namespace no_sport_members_count_l388_38853

theorem no_sport_members_count (n B T B_and_T : ℕ) (h1 : n = 27) (h2 : B = 17) (h3 : T = 19) (h4 : B_and_T = 11) : 
  n - (B + T - B_and_T) = 2 :=
by
  sorry

end no_sport_members_count_l388_38853


namespace highway_extension_completion_l388_38841

def current_length := 200
def final_length := 650
def built_first_day := 50
def built_second_day := 3 * built_first_day

theorem highway_extension_completion :
  (final_length - current_length - built_first_day - built_second_day) = 250 := by
  sorry

end highway_extension_completion_l388_38841


namespace index_card_area_l388_38807

theorem index_card_area (a b : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : (a - 2) * b = 21) : (a * (b - 1)) = 30 := by
  sorry

end index_card_area_l388_38807


namespace decagon_ratio_bisect_l388_38824

theorem decagon_ratio_bisect (area_decagon unit_square area_trapezoid : ℕ) 
  (h_area_decagon : area_decagon = 12) 
  (h_bisect : ∃ RS : ℕ, ∃ XR : ℕ, RS * 2 = area_decagon) 
  (below_RS : ∃ base1 base2 height : ℕ, base1 = 3 ∧ base2 = 3 ∧ base1 * height + 1 = 6) 
  : ∃ XR RS : ℕ, RS ≠ 0 ∧ XR / RS = 1 := 
sorry

end decagon_ratio_bisect_l388_38824


namespace reduced_price_per_dozen_l388_38854

theorem reduced_price_per_dozen 
  (P : ℝ) -- original price per apple
  (R : ℝ) -- reduced price per apple
  (A : ℝ) -- number of apples originally bought for Rs. 30
  (H1 : R = 0.7 * P) 
  (H2 : A * P = (A + 54) * R) :
  30 / (A + 54) * 12 = 2 :=
by
  sorry

end reduced_price_per_dozen_l388_38854


namespace common_ratio_neg_two_l388_38811

theorem common_ratio_neg_two (a : ℕ → ℝ) (q : ℝ) 
  (h : ∀ n, a (n + 1) = a n * q)
  (H : 8 * a 2 + a 5 = 0) : 
  q = -2 :=
sorry

end common_ratio_neg_two_l388_38811


namespace cookies_eq_23_l388_38802

def total_packs : Nat := 27
def cakes : Nat := 4
def cookies : Nat := total_packs - cakes

theorem cookies_eq_23 : cookies = 23 :=
by
  -- Proof goes here
  sorry

end cookies_eq_23_l388_38802


namespace rail_elevation_correct_angle_l388_38895

noncomputable def rail_elevation_angle (v : ℝ) (R : ℝ) (g : ℝ) : ℝ :=
  Real.arctan (v^2 / (R * g))

theorem rail_elevation_correct_angle :
  rail_elevation_angle (60 * (1000 / 3600)) 200 9.8 = 8.09 := by
  sorry

end rail_elevation_correct_angle_l388_38895


namespace correct_option_l388_38873

-- Definitions based on the conditions in step a
def option_a : Prop := (-3 - 1 = -2)
def option_b : Prop := (-2 * (-1 / 2) = 1)
def option_c : Prop := (16 / (-4 / 3) = 12)
def option_d : Prop := (- (3^2) / 4 = (9 / 4))

-- The proof problem statement asserting that only option B is correct.
theorem correct_option : option_b ∧ ¬ option_a ∧ ¬ option_c ∧ ¬ option_d :=
by sorry

end correct_option_l388_38873


namespace students_in_both_clubs_l388_38849

theorem students_in_both_clubs (total_students drama_club science_club : ℕ) 
  (students_either_or_both both_clubs : ℕ) 
  (h_total_students : total_students = 250)
  (h_drama_club : drama_club = 80)
  (h_science_club : science_club = 120)
  (h_students_either_or_both : students_either_or_both = 180)
  (h_inclusion_exclusion : students_either_or_both = drama_club + science_club - both_clubs) :
  both_clubs = 20 :=
  by sorry

end students_in_both_clubs_l388_38849


namespace income_expenditure_ratio_l388_38833

noncomputable def I : ℝ := 19000
noncomputable def S : ℝ := 3800
noncomputable def E : ℝ := I - S

theorem income_expenditure_ratio : (I / E) = 5 / 4 := by
  sorry

end income_expenditure_ratio_l388_38833


namespace real_coeffs_with_even_expression_are_integers_l388_38884

theorem real_coeffs_with_even_expression_are_integers
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h : ∀ x y : ℤ, (∃ k1 : ℤ, a1 * x + b1 * y + c1 = 2 * k1) ∨ (∃ k2 : ℤ, a2 * x + b2 * y + c2 = 2 * k2)) :
  (∃ (i1 j1 k1 : ℤ), a1 = i1 ∧ b1 = j1 ∧ c1 = k1) ∨
  (∃ (i2 j2 k2 : ℤ), a2 = i2 ∧ b2 = j2 ∧ c2 = k2) := by
  sorry

end real_coeffs_with_even_expression_are_integers_l388_38884


namespace smallest_value_wawbwcwd_l388_38875

noncomputable def g (x : ℝ) : ℝ := x^4 + 10 * x^3 + 35 * x^2 + 50 * x + 24

theorem smallest_value_wawbwcwd (w1 w2 w3 w4 : ℝ) : 
  (∀ x : ℝ, g x = 0 ↔ x = w1 ∨ x = w2 ∨ x = w3 ∨ x = w4) →
  |w1 * w2 + w3 * w4| = 12 ∨ |w1 * w3 + w2 * w4| = 12 ∨ |w1 * w4 + w2 * w3| = 12 :=
by 
  sorry

end smallest_value_wawbwcwd_l388_38875


namespace child_ticket_price_l388_38821

theorem child_ticket_price
    (num_people : ℕ)
    (num_adults : ℕ)
    (num_seniors : ℕ)
    (num_children : ℕ)
    (adult_ticket_cost : ℝ)
    (senior_discount : ℝ)
    (total_bill : ℝ) :
    num_people = 50 →
    num_adults = 25 →
    num_seniors = 15 →
    num_children = 10 →
    adult_ticket_cost = 15 →
    senior_discount = 0.25 →
    total_bill = 600 →
    ∃ x : ℝ, x = 5.63 :=
by {
  sorry
}

end child_ticket_price_l388_38821


namespace ratio_sheep_horses_eq_six_seven_l388_38885

noncomputable def total_food_per_day : ℕ := 12880
noncomputable def food_per_horse_per_day : ℕ := 230
noncomputable def num_sheep : ℕ := 48
noncomputable def num_horses : ℕ := total_food_per_day / food_per_horse_per_day
noncomputable def ratio_sheep_to_horses := num_sheep / num_horses

theorem ratio_sheep_horses_eq_six_seven :
  ratio_sheep_to_horses = 6 / 7 :=
by
  sorry

end ratio_sheep_horses_eq_six_seven_l388_38885


namespace train_speed_in_kph_l388_38855

-- Define the given conditions
def length_of_train : ℝ := 200 -- meters
def time_crossing_pole : ℝ := 16 -- seconds

-- Define conversion factor
def mps_to_kph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

-- Statement of the theorem
theorem train_speed_in_kph : mps_to_kph (length_of_train / time_crossing_pole) = 45 := 
sorry

end train_speed_in_kph_l388_38855


namespace lunch_cost_before_tip_l388_38887

theorem lunch_cost_before_tip (C : ℝ) (h : C + 0.2 * C = 60.6) : C = 50.5 :=
sorry

end lunch_cost_before_tip_l388_38887


namespace find_m_l388_38867

theorem find_m
  (x y : ℝ)
  (h1 : 100 = 300 * x + 200 * y)
  (h2 : 120 = 240 * x + 300 * y)
  (h3 : ∃ m : ℝ, 50 * 3 = 150 * x + m * y):
  ∃ m : ℝ, m = 450 :=
by
  sorry

end find_m_l388_38867


namespace arithmetic_geometric_progression_inequality_l388_38844

theorem arithmetic_geometric_progression_inequality
  {a b c d e f D g : ℝ}
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
  (e_pos : 0 < e) (f_pos : 0 < f)
  (h1 : b = a + D)
  (h2 : c = a + 2 * D)
  (h3 : e = a * g)
  (h4 : f = a * g^2)
  (h5 : d = a + 3 * D)
  (h6 : d = a * g^3) : 
  b * c ≥ e * f :=
by sorry

end arithmetic_geometric_progression_inequality_l388_38844


namespace more_seventh_graders_than_sixth_graders_l388_38839

theorem more_seventh_graders_than_sixth_graders 
  (n m : ℕ)
  (H1 : ∀ x : ℕ, x = n → 7 * n = 6 * m) : 
  m > n := 
by
  -- Proof is not required and will be skipped with sorry.
  sorry

end more_seventh_graders_than_sixth_graders_l388_38839


namespace find_k_l388_38845

theorem find_k (k : ℝ) :
  (∀ x, x ≠ 1 → (1 / (x^2 - x) + (k - 5) / (x^2 + x) = (k - 1) / (x^2 - 1))) →
  (1 / (1^2 - 1) + (k - 5) / (1^2 + 1) ≠ (k - 1) / (1^2 - 1)) →
  k = 3 :=
by
  sorry

end find_k_l388_38845


namespace prime_fraction_sum_l388_38891

theorem prime_fraction_sum (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
    (h : a + b + c + a * b * c = 99) :
    |(1 / a : ℚ) - (1 / b : ℚ)| + |(1 / b : ℚ) - (1 / c : ℚ)| + |(1 / c : ℚ) - (1 / a : ℚ)| = 9 / 11 := 
sorry

end prime_fraction_sum_l388_38891


namespace intersection_A_B_l388_38815

def A : Set ℤ := {x | x > 0 }
def B : Set ℤ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {1, 2, 3} :=
by
  sorry

end intersection_A_B_l388_38815


namespace no_triangle_with_perfect_square_sides_l388_38822

theorem no_triangle_with_perfect_square_sides :
  ∃ (a b : ℕ), a > 1000 ∧ b > 1000 ∧
    ∀ (c : ℕ), (∃ d : ℕ, c = d^2) → 
    ¬ (a + b > c ∧ b + c > a ∧ a + c > b) :=
sorry

end no_triangle_with_perfect_square_sides_l388_38822


namespace max_value_of_quadratic_l388_38889

theorem max_value_of_quadratic (x : ℝ) (h : 0 < x ∧ x < 6) : (6 - x) * x ≤ 9 := 
by
  sorry

end max_value_of_quadratic_l388_38889


namespace carpet_coverage_percentage_l388_38883

variable (l w : ℝ) (floor_area carpet_area : ℝ)

theorem carpet_coverage_percentage 
  (h_carpet_area: carpet_area = l * w) 
  (h_floor_area: floor_area = 180) 
  (hl : l = 4) 
  (hw : w = 9) : 
  carpet_area / floor_area * 100 = 20 := by
  sorry

end carpet_coverage_percentage_l388_38883


namespace quadratic_rewrite_de_value_l388_38870

theorem quadratic_rewrite_de_value : 
  ∃ (d e f : ℤ), (d^2 * x^2 + 2 * d * e * x + e^2 + f = 4 * x^2 - 16 * x + 2) → (d * e = -8) :=
by
  sorry

end quadratic_rewrite_de_value_l388_38870


namespace linear_in_one_variable_linear_in_two_variables_l388_38809

namespace MathProof

-- Definition of the equation
def equation (k x y : ℝ) : ℝ := (k^2 - 1) * x^2 + (k + 1) * x + (k - 7) * y - (k + 2)

-- Theorem for linear equation in one variable
theorem linear_in_one_variable (k : ℝ) (x y : ℝ) :
  k = -1 → equation k x y = 0 → ∃ y' : ℝ, equation k 0 y' = 0 :=
by
  sorry

-- Theorem for linear equation in two variables
theorem linear_in_two_variables (k : ℝ) (x y : ℝ) :
  k = 1 → equation k x y = 0 → ∃ x' y' : ℝ, equation k x' y' = 0 :=
by
  sorry

end MathProof

end linear_in_one_variable_linear_in_two_variables_l388_38809


namespace sum_of_x_values_l388_38827

noncomputable def arithmetic_angles_triangle (x : ℝ) : Prop :=
  let α := 30 * Real.pi / 180
  let β := (30 + 40) * Real.pi / 180
  let γ := (30 + 80) * Real.pi / 180
  (x = 6) ∨ (x = 8) ∨ (x = (7 + Real.sqrt 36 + Real.sqrt 83))

theorem sum_of_x_values : ∀ x : ℝ, 
  arithmetic_angles_triangle x → 
  (∃ p q r : ℝ, x = p + Real.sqrt q + Real.sqrt r ∧ p = 7 ∧ q = 36 ∧ r = 83) := 
by
  sorry

end sum_of_x_values_l388_38827


namespace liking_songs_proof_l388_38837

def num_ways_liking_songs : ℕ :=
  let total_songs := 6
  let pair1 := 1
  let pair2 := 2
  let ways_to_choose_pair1 := Nat.choose total_songs pair1
  let remaining_songs := total_songs - pair1
  let ways_to_choose_pair2 := Nat.choose remaining_songs pair2 * Nat.choose (remaining_songs - pair2) pair2
  let final_song_choices := 4
  ways_to_choose_pair1 * ways_to_choose_pair2 * final_song_choices * 3 -- multiplied by 3 for the three possible pairs

theorem liking_songs_proof :
  num_ways_liking_songs = 2160 :=
  by sorry

end liking_songs_proof_l388_38837


namespace depth_of_ship_l388_38886

-- Condition definitions
def rate : ℝ := 80  -- feet per minute
def time : ℝ := 50  -- minutes

-- Problem Statement
theorem depth_of_ship : rate * time = 4000 :=
by
  sorry

end depth_of_ship_l388_38886


namespace sum_powers_l388_38816

theorem sum_powers {a b : ℝ}
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 :=
by
  sorry

end sum_powers_l388_38816


namespace train_speed_l388_38860

noncomputable def train_length : ℝ := 120
noncomputable def crossing_time : ℝ := 2.699784017278618

theorem train_speed : (train_length / crossing_time) = 44.448 := by
  sorry

end train_speed_l388_38860


namespace multiplication_division_l388_38838

theorem multiplication_division:
  (213 * 16 = 3408) → (1.6 * 2.13 = 3.408) :=
by
  sorry

end multiplication_division_l388_38838


namespace no_integer_n_for_fractions_l388_38876

theorem no_integer_n_for_fractions (n : ℤ) : ¬ (∃ n : ℤ, (n - 6) % 15 = 0 ∧ (n - 5) % 24 = 0) :=
by sorry

end no_integer_n_for_fractions_l388_38876


namespace horizontal_asymptote_at_3_l388_38819

noncomputable def rational_function (x : ℝ) : ℝ :=
  (15 * x^4 + 2 * x^3 + 11 * x^2 + 6 * x + 4) / (5 * x^4 + x^3 + 10 * x^2 + 4 * x + 2)

theorem horizontal_asymptote_at_3 : 
  (∀ ε > 0, ∃ N > 0, ∀ x > N, |rational_function x - 3| < ε) := 
by
  sorry

end horizontal_asymptote_at_3_l388_38819


namespace graveling_cost_is_3900_l388_38812

noncomputable def cost_of_graveling_roads 
  (length : ℕ) (breadth : ℕ) (width_road : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  let area_road_length := length * width_road
  let area_road_breadth := (breadth - width_road) * width_road
  let total_area := area_road_length + area_road_breadth
  total_area * cost_per_sq_m

theorem graveling_cost_is_3900 :
  cost_of_graveling_roads 80 60 10 3 = 3900 := 
by 
  unfold cost_of_graveling_roads
  sorry

end graveling_cost_is_3900_l388_38812


namespace nth_monomial_l388_38856

variable (a : ℝ)

def monomial_seq (n : ℕ) : ℝ :=
  (n + 1) * a ^ n

theorem nth_monomial (n : ℕ) : monomial_seq a n = (n + 1) * a ^ n :=
by
  sorry

end nth_monomial_l388_38856


namespace Linda_total_sales_l388_38800

theorem Linda_total_sales (necklaces_sold : ℕ) (rings_sold : ℕ) 
    (necklace_price : ℕ) (ring_price : ℕ) 
    (total_sales : ℕ) : 
    necklaces_sold = 4 → 
    rings_sold = 8 → 
    necklace_price = 12 → 
    ring_price = 4 → 
    total_sales = necklaces_sold * necklace_price + rings_sold * ring_price → 
    total_sales = 80 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end Linda_total_sales_l388_38800


namespace negation_universal_proposition_l388_38801

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, 0 < x → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, 0 < x ∧ x^2 + x + 1 ≤ 0) :=
by
  sorry

end negation_universal_proposition_l388_38801


namespace shoes_per_person_l388_38828

theorem shoes_per_person (friends : ℕ) (pairs_of_shoes : ℕ) 
  (h1 : friends = 35) (h2 : pairs_of_shoes = 36) : 
  (pairs_of_shoes * 2) / (friends + 1) = 2 := by
  sorry

end shoes_per_person_l388_38828


namespace sufficient_condition_for_having_skin_l388_38825

theorem sufficient_condition_for_having_skin (H_no_skin_no_hair : ¬skin → ¬hair) :
  (hair → skin) :=
sorry

end sufficient_condition_for_having_skin_l388_38825


namespace problem_l388_38851

variables {a b : ℝ}

theorem problem (h₁ : -1 < a) (h₂ : a < b) (h₃ : b < 0) : 
  (1/a > 1/b) ∧ (a^2 + b^2 > 2 * a * b) ∧ (a + (1/a) > b + (1/b)) :=
by
  sorry

end problem_l388_38851


namespace number_of_terminating_decimals_l388_38857

theorem number_of_terminating_decimals :
  ∃ (count : ℕ), count = 64 ∧ ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 449 → (∃ k : ℕ, n = 7 * k) → (∃ k : ℕ, (∃ m : ℕ, 560 = 2^m * 5^k * n)) :=
sorry

end number_of_terminating_decimals_l388_38857


namespace total_population_of_cities_l388_38896

theorem total_population_of_cities 
    (number_of_cities : ℕ) 
    (average_population : ℕ) 
    (h1 : number_of_cities = 25) 
    (h2 : average_population = (5200 + 5700) / 2) : 
    number_of_cities * average_population = 136250 := by 
    sorry

end total_population_of_cities_l388_38896


namespace faster_train_speed_is_45_l388_38861

noncomputable def speedOfFasterTrain (V_s : ℝ) (length_train : ℝ) (time : ℝ) : ℝ :=
  let V_r : ℝ := (length_train * 2) / (time / 3600)
  V_r - V_s

theorem faster_train_speed_is_45 
  (length_train : ℝ := 0.5)
  (V_s : ℝ := 30)
  (time : ℝ := 47.99616030717543) :
  speedOfFasterTrain V_s length_train time = 45 :=
sorry

end faster_train_speed_is_45_l388_38861


namespace letters_into_mailboxes_l388_38850

theorem letters_into_mailboxes (letters : ℕ) (mailboxes : ℕ) (h_letters: letters = 3) (h_mailboxes: mailboxes = 4) :
  (mailboxes ^ letters) = 64 := by
  sorry

end letters_into_mailboxes_l388_38850


namespace range_of_a_l388_38831

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + a * x + a = 0}

theorem range_of_a (a : ℝ) (h : set_A ∪ set_B a = set_A) : 0 ≤ a ∧ a < 4 := 
sorry

end range_of_a_l388_38831


namespace flowers_sold_l388_38888

theorem flowers_sold (lilacs roses gardenias : ℕ) 
  (h1 : lilacs = 10)
  (h2 : roses = 3 * lilacs)
  (h3 : gardenias = lilacs / 2) : 
  lilacs + roses + gardenias = 45 :=
by
  sorry

end flowers_sold_l388_38888


namespace avg_books_per_student_l388_38864

theorem avg_books_per_student 
  (total_students : ℕ)
  (students_zero_books : ℕ)
  (students_one_book : ℕ)
  (students_two_books : ℕ)
  (max_books_per_student : ℕ) 
  (remaining_students_min_books : ℕ)
  (total_books : ℕ)
  (avg_books : ℚ)
  (h1 : total_students = 32)
  (h2 : students_zero_books = 2)
  (h3 : students_one_book = 12)
  (h4 : students_two_books = 10)
  (h5 : max_books_per_student = 11)
  (h6 : remaining_students_min_books = 8)
  (h7 : total_books = 0 * students_zero_books + 1 * students_one_book + 2 * students_two_books + 3 * remaining_students_min_books)
  (h8 : avg_books = total_books / total_students) :
  avg_books = 1.75 :=
by {
  -- Additional constraints and intermediate steps can be added here if necessary
  sorry
}

end avg_books_per_student_l388_38864


namespace sam_puppies_count_l388_38862

variable (initial_puppies : ℝ) (given_away_puppies : ℝ)

theorem sam_puppies_count (h1 : initial_puppies = 6.0) 
                          (h2 : given_away_puppies = 2.0) : 
                          initial_puppies - given_away_puppies = 4.0 :=
by simp [h1, h2]; sorry

end sam_puppies_count_l388_38862


namespace positive_difference_of_squares_l388_38826

theorem positive_difference_of_squares (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 12) : x^2 - y^2 = 600 := by
  sorry

end positive_difference_of_squares_l388_38826


namespace remaining_slices_correct_l388_38869

def pies : Nat := 2
def slices_per_pie : Nat := 8
def slices_total : Nat := pies * slices_per_pie
def slices_rebecca_initial : Nat := 1 * pies
def slices_remaining_after_rebecca : Nat := slices_total - slices_rebecca_initial
def slices_family_friends : Nat := 7
def slices_remaining_after_family_friends : Nat := slices_remaining_after_rebecca - slices_family_friends
def slices_rebecca_husband_last : Nat := 2
def slices_remaining : Nat := slices_remaining_after_family_friends - slices_rebecca_husband_last

theorem remaining_slices_correct : slices_remaining = 5 := 
by sorry

end remaining_slices_correct_l388_38869


namespace square_side_length_eq_area_and_perimeter_l388_38805

theorem square_side_length_eq_area_and_perimeter (a : ℝ) (h : a^2 = 4 * a) : a = 4 :=
by sorry

end square_side_length_eq_area_and_perimeter_l388_38805


namespace vet_fees_cat_result_l388_38806

-- Given conditions
def vet_fees_dog : ℕ := 15
def families_dogs : ℕ := 8
def families_cats : ℕ := 3
def vet_donation : ℕ := 53

-- Mathematical equivalency in Lean
noncomputable def vet_fees_cat (C : ℕ) : Prop :=
  (1 / 3 : ℚ) * (families_dogs * vet_fees_dog + families_cats * C) = vet_donation

-- Prove the vet fees for cats are 13 using above conditions
theorem vet_fees_cat_result : ∃ (C : ℕ), vet_fees_cat C ∧ C = 13 :=
by {
  use 13,
  sorry
}

end vet_fees_cat_result_l388_38806


namespace intersection_correct_l388_38882

def setA := {x : ℝ | (x - 2) * (2 * x + 1) ≤ 0}
def setB := {x : ℝ | x < 1}
def expectedIntersection := {x : ℝ | -1 / 2 ≤ x ∧ x < 1}

theorem intersection_correct : (setA ∩ setB) = expectedIntersection := by
  sorry

end intersection_correct_l388_38882


namespace Albert_eats_48_slices_l388_38818

theorem Albert_eats_48_slices (large_pizzas : ℕ) (small_pizzas : ℕ) (slices_large : ℕ) (slices_small : ℕ) 
  (h1 : large_pizzas = 2) (h2 : small_pizzas = 2) (h3 : slices_large = 16) (h4 : slices_small = 8) :
  (large_pizzas * slices_large + small_pizzas * slices_small) = 48 := 
by 
  -- sorry is used to skip the proof.
  sorry

end Albert_eats_48_slices_l388_38818


namespace matching_pair_probability_l388_38894

def total_pairs : ℕ := 17

def black_pairs : ℕ := 8
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 3
def red_pairs : ℕ := 2

def total_shoes : ℕ := 2 * (black_pairs + brown_pairs + gray_pairs + red_pairs)

def prob_match (n_pairs : ℕ) (total_shoes : ℕ) :=
  (2 * n_pairs / total_shoes) * (n_pairs / (total_shoes - 1))

noncomputable def probability_of_matching_pair :=
  (prob_match black_pairs total_shoes) +
  (prob_match brown_pairs total_shoes) +
  (prob_match gray_pairs total_shoes) +
  (prob_match red_pairs total_shoes)

theorem matching_pair_probability :
  probability_of_matching_pair = 93 / 551 :=
sorry

end matching_pair_probability_l388_38894


namespace inequality_cannot_hold_l388_38893

noncomputable def f (a b c x : ℝ) := a * x ^ 2 + b * x + c

theorem inequality_cannot_hold
  (a b c : ℝ)
  (h_symm : ∀ x, f a b c x = f a b c (2 - x)) :
  ¬ (f a b c (1 - a) < f a b c (1 - 2 * a) ∧ f a b c (1 - 2 * a) < f a b c 1) :=
by {
  sorry
}

end inequality_cannot_hold_l388_38893


namespace point_in_quadrant_I_l388_38808

theorem point_in_quadrant_I (x y : ℝ) (h1 : 4 * x + 6 * y = 24) (h2 : y = x + 3) : x > 0 ∧ y > 0 :=
by sorry

end point_in_quadrant_I_l388_38808


namespace value_of_b_l388_38881

theorem value_of_b (b : ℝ) : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^3 - b*x1^2 + 1/2 = 0) ∧ (x2^3 - b*x2^2 + 1/2 = 0)) → b = 3/2 :=
by
  sorry

end value_of_b_l388_38881


namespace triangle_is_isosceles_l388_38852

theorem triangle_is_isosceles (a b c : ℝ) (h : 3 * a^3 + 6 * a^2 * b - 3 * a^2 * c - 6 * a * b * c = 0) 
  (habc : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) : 
  (a = c) := 
by
  sorry

end triangle_is_isosceles_l388_38852


namespace sufficient_but_not_necessary_condition_l388_38878

open Real

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 1) → (a + b > 2 ∧ a * b > 1) ∧ ¬((a + b > 2 ∧ a * b > 1) → (a > 1 ∧ b > 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l388_38878


namespace asian_games_volunteer_selection_l388_38871

-- Define the conditions.

def total_volunteers : ℕ := 5
def volunteer_A_cannot_serve_language_services : Prop := true

-- Define the main problem.
-- We are supposed to find the number of ways to assign three roles given the conditions.
def num_ways_to_assign_roles : ℕ :=
  let num_ways_language_services := 4 -- A cannot serve this role, so 4 choices
  let num_ways_other_roles := 4 * 3 -- We need to choose and arrange 2 volunteers out of remaining
  num_ways_language_services * num_ways_other_roles

-- The target theorem.
theorem asian_games_volunteer_selection : num_ways_to_assign_roles = 48 :=
by
  sorry

end asian_games_volunteer_selection_l388_38871


namespace total_cans_collected_l388_38834

-- Definitions based on conditions
def bags_on_saturday : ℕ := 6
def bags_on_sunday : ℕ := 3
def cans_per_bag : ℕ := 8

-- The theorem statement
theorem total_cans_collected : bags_on_saturday + bags_on_sunday * cans_per_bag = 72 :=
by
  sorry

end total_cans_collected_l388_38834


namespace sufficient_condition_for_beta_l388_38846

theorem sufficient_condition_for_beta (m : ℝ) : 
  (∀ x, (1 ≤ x ∧ x ≤ 3) → (x ≤ m)) → (3 ≤ m) :=
by
  sorry

end sufficient_condition_for_beta_l388_38846


namespace range_of_a_l388_38842

theorem range_of_a (f : ℝ → ℝ) (h_increasing : ∀ x y, x < y → f x < f y) (a : ℝ) :
  f (a^2 - a) > f (2 * a^2 - 4 * a) → 0 < a ∧ a < 3 :=
by
  -- We translate the condition f(a^2 - a) > f(2a^2 - 4a) to the inequality
  intro h
  -- Apply the fact that f is increasing to deduce the inequality on a
  sorry

end range_of_a_l388_38842


namespace stones_equally_distributed_l388_38814

theorem stones_equally_distributed (n k : ℕ) 
    (h : ∃ piles : Fin n → ℕ, (∀ i j, 2 * piles i + piles j = k * n)) :
  ∃ m : ℕ, k = 2^m :=
by
  sorry

end stones_equally_distributed_l388_38814


namespace no_real_roots_range_l388_38803

theorem no_real_roots_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 ≠ 0) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end no_real_roots_range_l388_38803


namespace relationship_between_M_and_P_l388_38890

def M := {y : ℝ | ∃ x : ℝ, y = x^2 - 4}
def P := {x : ℝ | 2 ≤ x ∧ x ≤ 4}

theorem relationship_between_M_and_P : ∀ y ∈ {y : ℝ | ∃ x ∈ P, y = x^2 - 4}, y ∈ M :=
by
  sorry

end relationship_between_M_and_P_l388_38890


namespace characterize_solution_pairs_l388_38865

/-- Define a set S --/
def S : Set ℝ := { x : ℝ | x > 0 ∧ x ≠ 1 }

/-- log inequality --/
def log_inequality (a b : ℝ) : Prop :=
  Real.log b / Real.log a < Real.log (b + 1) / Real.log (a + 1)

/-- Define the solution sets --/
def sol1 : Set (ℝ × ℝ) := {p | p.2 = 1 ∧ p.1 > 0 ∧ p.1 ≠ 1}
def sol2 : Set (ℝ × ℝ) := {p | p.1 > p.2 ∧ p.2 > 1}
def sol3 : Set (ℝ × ℝ) := {p | p.2 > 1 ∧ p.2 > p.1}
def sol4 : Set (ℝ × ℝ) := {p | p.1 < p.2 ∧ p.2 < 1}
def sol5 : Set (ℝ × ℝ) := {p | p.2 < 1 ∧ p.2 < p.1}

/-- Prove the log inequality and characterize the solution pairs --/
theorem characterize_solution_pairs (a b : ℝ) (h1 : a ∈ S) (h2 : b > 0) :
  log_inequality a b ↔
  (a, b) ∈ sol1 ∨ (a, b) ∈ sol2 ∨ (a, b) ∈ sol3 ∨ (a, b) ∈ sol4 ∨ (a, b) ∈ sol5 :=
sorry

end characterize_solution_pairs_l388_38865


namespace frog_jump_probability_is_one_fifth_l388_38829

noncomputable def frog_jump_probability : ℝ := sorry

theorem frog_jump_probability_is_one_fifth : frog_jump_probability = 1 / 5 := sorry

end frog_jump_probability_is_one_fifth_l388_38829


namespace rightmost_three_digits_of_7_pow_1994_l388_38817

theorem rightmost_three_digits_of_7_pow_1994 :
  (7 ^ 1994) % 800 = 49 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1994_l388_38817


namespace log_sum_equals_18084_l388_38847

theorem log_sum_equals_18084 : 
  (Finset.sum (Finset.range 2013) (λ x => (Int.floor (Real.log x / Real.log 2)))) = 18084 :=
by
  sorry

end log_sum_equals_18084_l388_38847


namespace solve_ineq_l388_38874

theorem solve_ineq (x : ℝ) : (x > 0 ∧ x < 3 ∨ x > 8) → x^3 - 9 * x^2 + 24 * x > 0 :=
by
  sorry

end solve_ineq_l388_38874


namespace tshirt_costs_more_than_jersey_l388_38880

-- Definitions based on the conditions
def cost_of_tshirt : ℕ := 192
def cost_of_jersey : ℕ := 34

-- Theorem statement
theorem tshirt_costs_more_than_jersey : cost_of_tshirt - cost_of_jersey = 158 := by
  sorry

end tshirt_costs_more_than_jersey_l388_38880


namespace min_value_x_squared_y_cubed_z_l388_38848

theorem min_value_x_squared_y_cubed_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
(h : 1 / x + 1 / y + 1 / z = 9) : x^2 * y^3 * z ≥ 729 / 6912 :=
sorry

end min_value_x_squared_y_cubed_z_l388_38848


namespace find_radius_l388_38877

theorem find_radius
  (sector_area : ℝ)
  (arc_length : ℝ)
  (sector_area_eq : sector_area = 11.25)
  (arc_length_eq : arc_length = 4.5) :
  ∃ r : ℝ, 11.25 = (1/2 : ℝ) * r * arc_length ∧ r = 5 := 
by
  sorry

end find_radius_l388_38877


namespace average_tree_height_l388_38840

theorem average_tree_height :
  let tree1 := 8
  let tree2 := if tree3 = 16 then 4 else 16
  let tree3 := 16
  let tree4 := if tree5 = 32 then 8 else 32
  let tree5 := 32
  let tree6 := if tree5 = 32 then 64 else 16
  let total_sum := tree1 + tree2 + tree3 + tree4 + tree5 + tree6
  let average_height := total_sum / 6
  average_height = 14 :=
by
  sorry

end average_tree_height_l388_38840


namespace number_of_cows_l388_38843

theorem number_of_cows (C H : ℕ) (L : ℕ) (h1 : L = 4 * C + 2 * H) (h2 : L = 2 * (C + H) + 20) : C = 10 :=
by
  sorry

end number_of_cows_l388_38843


namespace shaded_ratio_l388_38820

theorem shaded_ratio (full_rectangles half_rectangles : ℕ) (n m : ℕ) (rectangle_area shaded_area total_area : ℝ)
  (h1 : n = 4) (h2 : m = 5) (h3 : rectangle_area = n * m) 
  (h4 : full_rectangles = 3) (h5 : half_rectangles = 4)
  (h6 : shaded_area = full_rectangles * 1 + 0.5 * half_rectangles * 1)
  (h7 : total_area = rectangle_area) :
  shaded_area / total_area = 1 / 4 := by
  sorry

end shaded_ratio_l388_38820


namespace water_added_l388_38872

def container_capacity : ℕ := 80
def initial_fill_percentage : ℝ := 0.5
def final_fill_percentage : ℝ := 0.75
def initial_fill_amount (capacity : ℕ) (percentage : ℝ) : ℝ := percentage * capacity
def final_fill_amount (capacity : ℕ) (percentage : ℝ) : ℝ := percentage * capacity

theorem water_added (capacity : ℕ) (initial_percentage final_percentage : ℝ) :
  final_fill_amount capacity final_percentage - initial_fill_amount capacity initial_percentage = 20 :=
by {
  sorry
}

end water_added_l388_38872
