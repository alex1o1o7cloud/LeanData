import Mathlib

namespace NUMINAMATH_GPT_polygon_sides_l786_78610

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) : 
  n = 6 :=
by sorry

end NUMINAMATH_GPT_polygon_sides_l786_78610


namespace NUMINAMATH_GPT_Faye_age_correct_l786_78662

def ages (C D E F G : ℕ) : Prop :=
  D = E - 2 ∧
  C = E + 3 ∧
  F = C - 1 ∧
  D = 16 ∧
  G = D - 5

theorem Faye_age_correct (C D E F G : ℕ) (h : ages C D E F G) : F = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_Faye_age_correct_l786_78662


namespace NUMINAMATH_GPT_x_plus_2y_equals_2_l786_78623

theorem x_plus_2y_equals_2 (x y : ℝ) (h : |x + 3| + (2 * y - 5)^2 = 0) : x + 2 * y = 2 := 
sorry

end NUMINAMATH_GPT_x_plus_2y_equals_2_l786_78623


namespace NUMINAMATH_GPT_number_of_times_difference_fits_is_20_l786_78617

-- Definitions for Ralph's pictures
def ralph_wild_animals := 75
def ralph_landscapes := 36
def ralph_family_events := 45
def ralph_cars := 20
def ralph_total_pictures := ralph_wild_animals + ralph_landscapes + ralph_family_events + ralph_cars

-- Definitions for Derrick's pictures
def derrick_wild_animals := 95
def derrick_landscapes := 42
def derrick_family_events := 55
def derrick_cars := 25
def derrick_airplanes := 10
def derrick_total_pictures := derrick_wild_animals + derrick_landscapes + derrick_family_events + derrick_cars + derrick_airplanes

-- Combined total number of pictures
def combined_total_pictures := ralph_total_pictures + derrick_total_pictures

-- Difference in wild animals pictures
def difference_wild_animals := derrick_wild_animals - ralph_wild_animals

-- Number of times the difference fits into the combined total (rounded down)
def times_difference_fits := combined_total_pictures / difference_wild_animals

-- Statement of the problem
theorem number_of_times_difference_fits_is_20 : times_difference_fits = 20 := by
  -- The proof will be written here
  sorry

end NUMINAMATH_GPT_number_of_times_difference_fits_is_20_l786_78617


namespace NUMINAMATH_GPT_problem_1_problem_2_l786_78696

-- Proof Problem 1
theorem problem_1 (a : ℝ) (h₀ : a = 1) (h₁ : ∀ x : ℝ, x^2 - 5 * a * x + 4 * a^2 < 0)
                                    (h₂ : ∀ x : ℝ, (x - 2) * (x - 5) < 0) :
  ∃ x : ℝ, 2 < x ∧ x < 4 :=
by sorry

-- Proof Problem 2
theorem problem_2 (p q : ℝ → Prop) (h₀ : ∀ x : ℝ, p x → q x) 
                                (p_def : ∀ (a : ℝ) (x : ℝ), 0 < a → p x ↔ a < x ∧ x < 4 * a) 
                                (q_def : ∀ x : ℝ, q x ↔ 2 < x ∧ x < 5) :
  ∃ a : ℝ, (5 / 4) ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l786_78696


namespace NUMINAMATH_GPT_cost_of_paving_is_correct_l786_78678

def length_of_room : ℝ := 5.5
def width_of_room : ℝ := 4
def rate_per_square_meter : ℝ := 950
def area_of_room : ℝ := length_of_room * width_of_room
def cost_of_paving : ℝ := area_of_room * rate_per_square_meter

theorem cost_of_paving_is_correct : cost_of_paving = 20900 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_paving_is_correct_l786_78678


namespace NUMINAMATH_GPT_imaginary_part_of_complex_l786_78645

open Complex -- Opens the complex numbers namespace

theorem imaginary_part_of_complex:
  ∀ (a b c d : ℂ), (a = (2 + I) / (1 - I) - (2 - I) / (1 + I)) → (a.im = 3) :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_complex_l786_78645


namespace NUMINAMATH_GPT_eq_square_sum_five_l786_78638

theorem eq_square_sum_five (a b : ℝ) (i : ℂ) (h : i * i = -1) (h_eq : (a - 2 * i) * i^2013 = b - i) : a^2 + b^2 = 5 :=
by
  -- Proof will be filled in later
  sorry

end NUMINAMATH_GPT_eq_square_sum_five_l786_78638


namespace NUMINAMATH_GPT_distinct_real_roots_eq_one_l786_78616

theorem distinct_real_roots_eq_one : 
  (∃ x : ℝ, |x| - 4/x = (3 * |x|) / x) ∧ 
  ¬∃ x1 x2 : ℝ, 
    x1 ≠ x2 ∧ 
    (|x1| - 4/x1 = (3 * |x1|) / x1) ∧ 
    (|x2| - 4/x2 = (3 * |x2|) / x2) :=
sorry

end NUMINAMATH_GPT_distinct_real_roots_eq_one_l786_78616


namespace NUMINAMATH_GPT_meaningful_fraction_l786_78652

theorem meaningful_fraction (x : ℝ) : (x - 1 ≠ 0) ↔ (x ≠ 1) :=
by sorry

end NUMINAMATH_GPT_meaningful_fraction_l786_78652


namespace NUMINAMATH_GPT_find_c_value_l786_78621

theorem find_c_value (a b : ℝ) (h1 : 12 = (6 / 100) * a) (h2 : 6 = (12 / 100) * b) : b / a = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_find_c_value_l786_78621


namespace NUMINAMATH_GPT_nesbitt_inequality_l786_78613

theorem nesbitt_inequality (a b c : ℝ) (h_pos1 : 0 < a) (h_pos2 : 0 < b) (h_pos3 : 0 < c) (h_abc: a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 :=
sorry

end NUMINAMATH_GPT_nesbitt_inequality_l786_78613


namespace NUMINAMATH_GPT_four_p_minus_three_is_square_l786_78622

theorem four_p_minus_three_is_square
  (n : ℕ) (p : ℕ)
  (hn_pos : n > 1)
  (hp_prime : Prime p)
  (h1 : n ∣ (p - 1))
  (h2 : p ∣ (n^3 - 1)) : ∃ k : ℕ, 4 * p - 3 = k^2 := sorry

end NUMINAMATH_GPT_four_p_minus_three_is_square_l786_78622


namespace NUMINAMATH_GPT_max_value_a7_a14_l786_78680

noncomputable def arithmetic_sequence_max_product (a_1 d : ℝ) : ℝ :=
  let a_7 := a_1 + 6 * d
  let a_14 := a_1 + 13 * d
  a_7 * a_14

theorem max_value_a7_a14 {a_1 d : ℝ} 
  (h : 10 = 2 * a_1 + 19 * d)
  (sum_first_20 : 100 = (10) * (a_1 + a_1 + 19 * d)) :
  arithmetic_sequence_max_product a_1 d = 25 :=
by
  sorry

end NUMINAMATH_GPT_max_value_a7_a14_l786_78680


namespace NUMINAMATH_GPT_solve_for_x_l786_78659

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 5 / (x / 3)) : x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l786_78659


namespace NUMINAMATH_GPT_ratio_depends_on_S_and_r_l786_78687

theorem ratio_depends_on_S_and_r
    (S : ℝ) (r : ℝ) (P1 : ℝ) (C2 : ℝ)
    (h1 : P1 = 4 * S)
    (h2 : C2 = 2 * Real.pi * r) :
    (P1 / C2 = 4 * S / (2 * Real.pi * r)) := by
  sorry

end NUMINAMATH_GPT_ratio_depends_on_S_and_r_l786_78687


namespace NUMINAMATH_GPT_unrelated_statement_l786_78698

-- Definitions
def timely_snow_promises_harvest : Prop := true -- assumes it has a related factor
def upper_beam_not_straight_lower_beam_crooked : Prop := true -- assumes it has a related factor
def smoking_harmful_to_health : Prop := true -- assumes it has a related factor
def magpies_signify_joy_crows_signify_mourning : Prop := false -- does not have an inevitable relationship

-- Theorem
theorem unrelated_statement :
  ¬magpies_signify_joy_crows_signify_mourning :=
by 
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_unrelated_statement_l786_78698


namespace NUMINAMATH_GPT_inconsistent_linear_system_l786_78688

theorem inconsistent_linear_system :
  ¬ ∃ (x1 x2 x3 : ℝ), 
    (2 * x1 + 5 * x2 - 4 * x3 = 8) ∧
    (3 * x1 + 15 * x2 - 9 * x3 = 5) ∧
    (5 * x1 + 5 * x2 - 7 * x3 = 1) :=
by
  -- Proof of inconsistency
  sorry

end NUMINAMATH_GPT_inconsistent_linear_system_l786_78688


namespace NUMINAMATH_GPT_three_lines_intersect_at_three_points_l786_78694

-- Define the lines as propositions expressing the equations
def line1 (x y : ℝ) := 2 * y - 3 * x = 4
def line2 (x y : ℝ) := x + 3 * y = 3
def line3 (x y : ℝ) := 3 * x - 4.5 * y = 7.5

-- Define a proposition stating that there are exactly 3 unique points of intersection among the three lines
def number_of_intersections : ℕ := 3

-- Prove that the number of unique intersection points is exactly 3 given the lines
theorem three_lines_intersect_at_three_points : 
  ∃! p1 p2 p3 : ℝ × ℝ, 
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧ 
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧ 
    (line3 p3.1 p3.2 ∧ line1 p3.1 p3.2) :=
sorry

end NUMINAMATH_GPT_three_lines_intersect_at_three_points_l786_78694


namespace NUMINAMATH_GPT_find_m_l786_78693

theorem find_m (m : ℕ) :
  (2022 ^ 2 - 4) * (2021 ^ 2 - 4) = 2024 * 2020 * 2019 * m → 
  m = 2023 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l786_78693


namespace NUMINAMATH_GPT_compute_xy_l786_78646

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 198) : xy = 5 :=
by
  sorry

end NUMINAMATH_GPT_compute_xy_l786_78646


namespace NUMINAMATH_GPT_inequality_wxyz_l786_78691

theorem inequality_wxyz 
  (w x y z : ℝ) 
  (h₁ : w^2 + y^2 ≤ 1) : 
  (w * x + y * z - 1)^2 ≥ (w^2 + y^2 - 1) * (x^2 + z^2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_wxyz_l786_78691


namespace NUMINAMATH_GPT_perimeter_difference_l786_78674

-- Define the dimensions of the two figures
def width1 : ℕ := 6
def height1 : ℕ := 3
def width2 : ℕ := 6
def height2 : ℕ := 2

-- Define the perimeters of the two figures
def perimeter1 : ℕ := 2 * (width1 + height1)
def perimeter2 : ℕ := 2 * (width2 + height2)

-- Prove the positive difference in perimeters is 2 units
theorem perimeter_difference : (perimeter1 - perimeter2) = 2 := by
  sorry

end NUMINAMATH_GPT_perimeter_difference_l786_78674


namespace NUMINAMATH_GPT_hillary_stops_short_of_summit_l786_78651

noncomputable def distance_to_summit_from_base_camp : ℝ := 4700
noncomputable def hillary_climb_rate : ℝ := 800
noncomputable def eddy_climb_rate : ℝ := 500
noncomputable def hillary_descent_rate : ℝ := 1000
noncomputable def time_of_departure : ℝ := 6
noncomputable def time_of_passing : ℝ := 12

theorem hillary_stops_short_of_summit :
  ∃ x : ℝ, 
    (time_of_passing - time_of_departure) * hillary_climb_rate = distance_to_summit_from_base_camp - x →
    (time_of_passing - time_of_departure) * eddy_climb_rate = x →
    x = 2900 :=
by
  sorry

end NUMINAMATH_GPT_hillary_stops_short_of_summit_l786_78651


namespace NUMINAMATH_GPT_complex_expression_evaluation_l786_78640

theorem complex_expression_evaluation (i : ℂ) (h1 : i^(4 : ℤ) = 1) (h2 : i^(1 : ℤ) = i)
   (h3 : i^(2 : ℤ) = -1) (h4 : i^(3 : ℤ) = -i) (h5 : i^(0 : ℤ) = 1) : 
   i^(245 : ℤ) + i^(246 : ℤ) + i^(247 : ℤ) + i^(248 : ℤ) + i^(249 : ℤ) = i :=
by
  sorry

end NUMINAMATH_GPT_complex_expression_evaluation_l786_78640


namespace NUMINAMATH_GPT_square_number_n_value_l786_78654

theorem square_number_n_value
  (n : ℕ)
  (h : ∃ k : ℕ, 2^6 + 2^9 + 2^n = k^2) :
  n = 10 :=
sorry

end NUMINAMATH_GPT_square_number_n_value_l786_78654


namespace NUMINAMATH_GPT_find_b10_l786_78635

def sequence_b (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = b (n + 1) + b n

theorem find_b10 (b : ℕ → ℕ) (h0 : ∀ n, b n > 0) (h1 : b 9 = 544) (h2 : sequence_b b) : b 10 = 883 :=
by
  -- We could provide steps of the proof here, but we use 'sorry' to omit the proof content
  sorry

end NUMINAMATH_GPT_find_b10_l786_78635


namespace NUMINAMATH_GPT_logarithmic_relationship_l786_78658

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_relationship
  (h1 : 0 < Real.cos 1)
  (h2 : Real.cos 1 < Real.sin 1)
  (h3 : Real.sin 1 < 1)
  (h4 : 1 < Real.tan 1) :
  log_base (Real.sin 1) (Real.tan 1) < log_base (Real.cos 1) (Real.tan 1) ∧
  log_base (Real.cos 1) (Real.tan 1) < log_base (Real.cos 1) (Real.sin 1) ∧
  log_base (Real.cos 1) (Real.sin 1) < log_base (Real.sin 1) (Real.cos 1) :=
sorry

end NUMINAMATH_GPT_logarithmic_relationship_l786_78658


namespace NUMINAMATH_GPT_total_books_in_classroom_l786_78620

-- Define the given conditions using Lean definitions
def num_children : ℕ := 15
def books_per_child : ℕ := 12
def additional_books : ℕ := 22

-- Define the hypothesis and the corresponding proof statement
theorem total_books_in_classroom : num_children * books_per_child + additional_books = 202 := 
by sorry

end NUMINAMATH_GPT_total_books_in_classroom_l786_78620


namespace NUMINAMATH_GPT_solve_frac_difference_of_squares_l786_78661

theorem solve_frac_difference_of_squares :
  (108^2 - 99^2) / 9 = 207 := by
  sorry

end NUMINAMATH_GPT_solve_frac_difference_of_squares_l786_78661


namespace NUMINAMATH_GPT_set_B_can_form_right_angled_triangle_l786_78605

-- Definition and condition from the problem
def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- The actual proof problem statement
theorem set_B_can_form_right_angled_triangle : isRightAngledTriangle 1 (Real.sqrt 3) 2 :=
sorry

end NUMINAMATH_GPT_set_B_can_form_right_angled_triangle_l786_78605


namespace NUMINAMATH_GPT_maximum_distance_l786_78685

-- Definitions from the conditions
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def distance_driven : ℝ := 244
def gallons_used : ℝ := 20

-- Problem statement
theorem maximum_distance (h: (distance_driven / gallons_used = highway_mpg)): 
  (distance_driven = 244) :=
sorry

end NUMINAMATH_GPT_maximum_distance_l786_78685


namespace NUMINAMATH_GPT_max_knights_seated_next_to_two_knights_l786_78633

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end NUMINAMATH_GPT_max_knights_seated_next_to_two_knights_l786_78633


namespace NUMINAMATH_GPT_inverse_of_original_l786_78673

-- Definitions based on conditions
def original_proposition : Prop := ∀ (x y : ℝ), x = y → |x| = |y|

def inverse_proposition : Prop := ∀ (x y : ℝ), |x| = |y| → x = y

-- Lean 4 statement
theorem inverse_of_original : original_proposition → inverse_proposition :=
sorry

end NUMINAMATH_GPT_inverse_of_original_l786_78673


namespace NUMINAMATH_GPT_bears_in_shipment_l786_78643

theorem bears_in_shipment
  (initial_bears : ℕ) (shelves : ℕ) (bears_per_shelf : ℕ)
  (total_bears_after_shipment : ℕ) 
  (initial_bears_eq : initial_bears = 5)
  (shelves_eq : shelves = 2)
  (bears_per_shelf_eq : bears_per_shelf = 6)
  (total_bears_calculation : total_bears_after_shipment = shelves * bears_per_shelf)
  : total_bears_after_shipment - initial_bears = 7 :=
by
  sorry

end NUMINAMATH_GPT_bears_in_shipment_l786_78643


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l786_78663

theorem quadratic_two_distinct_real_roots (k : ℝ) (h1 : k ≠ 0) : 
  (∀ Δ > 0, Δ = (-2)^2 - 4 * k * (-1)) ↔ (k > -1) :=
by
  -- Since Δ = 4 + 4k, we need to show that (4 + 4k > 0) ↔ (k > -1)
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l786_78663


namespace NUMINAMATH_GPT_count_4_letter_words_with_A_l786_78631

-- Define the alphabet set and the properties
def Alphabet : Finset (Char) := {'A', 'B', 'C', 'D', 'E'}
def total_words := (Alphabet.card ^ 4 : ℕ)
def total_words_without_A := (Alphabet.erase 'A').card ^ 4
def total_words_with_A := total_words - total_words_without_A

-- The key theorem to prove
theorem count_4_letter_words_with_A : total_words_with_A = 369 := sorry

end NUMINAMATH_GPT_count_4_letter_words_with_A_l786_78631


namespace NUMINAMATH_GPT_cubic_polynomial_solution_l786_78618

noncomputable def q (x : ℝ) : ℝ := - (4 / 3) * x^3 + 6 * x^2 - (50 / 3) * x - (14 / 3)

theorem cubic_polynomial_solution :
  q 1 = -8 ∧ q 2 = -12 ∧ q 3 = -20 ∧ q 4 = -40 := by
  have h₁ : q 1 = -8 := by sorry
  have h₂ : q 2 = -12 := by sorry
  have h₃ : q 3 = -20 := by sorry
  have h₄ : q 4 = -40 := by sorry
  exact ⟨h₁, h₂, h₃, h₄⟩

end NUMINAMATH_GPT_cubic_polynomial_solution_l786_78618


namespace NUMINAMATH_GPT_largest_perfect_square_factor_of_3780_l786_78607

theorem largest_perfect_square_factor_of_3780 :
  ∃ m : ℕ, (∃ k : ℕ, 3780 = k * m * m) ∧ m * m = 36 :=
by
  sorry

end NUMINAMATH_GPT_largest_perfect_square_factor_of_3780_l786_78607


namespace NUMINAMATH_GPT_smallest_value_of_x_l786_78660

theorem smallest_value_of_x : ∃ x, (2 * x^2 + 30 * x - 84 = x * (x + 15)) ∧ (∀ y, (2 * y^2 + 30 * y - 84 = y * (y + 15)) → x ≤ y) ∧ x = -28 := by
  sorry

end NUMINAMATH_GPT_smallest_value_of_x_l786_78660


namespace NUMINAMATH_GPT_divide_area_into_squares_l786_78666

theorem divide_area_into_squares :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x / y = 4 / 3 ∧ (x^2 + y^2 = 100) ∧ x = 8 ∧ y = 6) := 
by {
  sorry
}

end NUMINAMATH_GPT_divide_area_into_squares_l786_78666


namespace NUMINAMATH_GPT_proof_speed_of_man_in_still_water_l786_78609

def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
  50 / 4 = v_m + v_s ∧ 30 / 6 = v_m - v_s

theorem proof_speed_of_man_in_still_water (v_m v_s : ℝ) :
  speed_of_man_in_still_water v_m v_s → v_m = 8.75 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_proof_speed_of_man_in_still_water_l786_78609


namespace NUMINAMATH_GPT_quadrilateral_area_offset_l786_78647

theorem quadrilateral_area_offset
  (d : ℝ) (x : ℝ) (y : ℝ) (A : ℝ)
  (h_d : d = 26)
  (h_y : y = 6)
  (h_A : A = 195) :
  A = 1/2 * (x + y) * d → x = 9 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_offset_l786_78647


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l786_78644

-- Definitions based on conditions
def set_A : Set ℝ := {x | x ≥ 3 ∨ x ≤ -1}
def set_B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Statement of the proof problem
theorem intersection_of_A_and_B : set_A ∩ set_B = {x | -2 ≤ x ∧ x ≤ -1} :=
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l786_78644


namespace NUMINAMATH_GPT_number_of_distributions_room_receives_three_people_number_of_distributions_room_receives_at_least_one_person_l786_78624

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of rooms
def total_rooms : ℕ := 2

-- For the first question, define: each room must receive exact three people
def room_receives_three_people (n m : ℕ) : Prop :=
  n = 3 ∧ m = 3

-- For the second question, define: each room must receive at least one person
def room_receives_at_least_one_person (n m : ℕ) : Prop :=
  n ≥ 1 ∧ m ≥ 1

theorem number_of_distributions_room_receives_three_people :
  ∃ (ways : ℕ), ways = 20 :=
by
  sorry

theorem number_of_distributions_room_receives_at_least_one_person :
  ∃ (ways : ℕ), ways = 62 :=
by
  sorry

end NUMINAMATH_GPT_number_of_distributions_room_receives_three_people_number_of_distributions_room_receives_at_least_one_person_l786_78624


namespace NUMINAMATH_GPT_partnership_total_profit_l786_78648

theorem partnership_total_profit
  (total_capital : ℝ)
  (A_share : ℝ := 1/3)
  (B_share : ℝ := 1/4)
  (C_share : ℝ := 1/5)
  (D_share : ℝ := 1 - (A_share + B_share + C_share))
  (A_profit : ℝ := 805)
  (A_capital : ℝ := total_capital * A_share)
  (total_capital_positive : 0 < total_capital)
  (shares_add_up : A_share + B_share + C_share + D_share = 1) :
  (A_profit / (total_capital * A_share)) * total_capital = 2415 :=
by
  -- Proof will go here.
  sorry

end NUMINAMATH_GPT_partnership_total_profit_l786_78648


namespace NUMINAMATH_GPT_sum_of_reciprocals_l786_78614

theorem sum_of_reciprocals (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 56) : (1/x) + (1/y) = 15/56 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l786_78614


namespace NUMINAMATH_GPT_fly_total_distance_l786_78625

noncomputable def total_distance_traveled (r : ℝ) (d3 : ℝ) : ℝ :=
  let d1 := 2 * r
  let d2 := Real.sqrt (d1^2 - d3^2)
  d1 + d2 + d3

theorem fly_total_distance (r : ℝ) (h_r : r = 60) (d3 : ℝ) (h_d3 : d3 = 90) :
  total_distance_traveled r d3 = 289.37 :=
by
  rw [h_r, h_d3]
  simp [total_distance_traveled]
  sorry

end NUMINAMATH_GPT_fly_total_distance_l786_78625


namespace NUMINAMATH_GPT_sum_and_product_of_roots_l786_78608

theorem sum_and_product_of_roots :
  let a := 1
  let b := -7
  let c := 12
  (∀ x: ℝ, x^2 - 7*x + 12 = 0 → (x = 3 ∨ x = 4)) →
  (-b/a = 7) ∧ (c/a = 12) := 
by
  sorry

end NUMINAMATH_GPT_sum_and_product_of_roots_l786_78608


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l786_78683

theorem no_positive_integer_solutions (x y z : ℕ) (h_cond : x^2 + y^2 = 7 * z^2) : 
  x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l786_78683


namespace NUMINAMATH_GPT_cost_of_fencing_is_289_l786_78639

def side_lengths : List ℕ := [10, 20, 15, 18, 12, 22]

def cost_per_meter : List ℚ := [3, 2, 4, 3.5, 2.5, 3]

def cost_of_side (length : ℕ) (rate : ℚ) : ℚ :=
  (length : ℚ) * rate

def total_cost : ℚ :=
  List.zipWith cost_of_side side_lengths cost_per_meter |>.sum

theorem cost_of_fencing_is_289 : total_cost = 289 := by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_is_289_l786_78639


namespace NUMINAMATH_GPT_intersecting_graphs_l786_78636

theorem intersecting_graphs (a b c d : ℝ) 
  (h1 : -2 * |1 - a| + b = 4) 
  (h2 : 2 * |1 - c| + d = 4)
  (h3 : -2 * |7 - a| + b = 0) 
  (h4 : 2 * |7 - c| + d = 0) : a + c = 10 := 
sorry

end NUMINAMATH_GPT_intersecting_graphs_l786_78636


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l786_78681

def condition_neq_1_or_neq_2 (a b : ℤ) : Prop :=
  a ≠ 1 ∨ b ≠ 2

def statement_sum_neq_3 (a b : ℤ) : Prop :=
  a + b ≠ 3

theorem necessary_but_not_sufficient_condition :
  ∀ (a b : ℤ), condition_neq_1_or_neq_2 a b → ¬ (statement_sum_neq_3 a b) → false :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l786_78681


namespace NUMINAMATH_GPT_Bella_catch_correct_l786_78686

def Martha_catch : ℕ := 3 + 7
def Cara_catch : ℕ := 5 * Martha_catch - 3
def T : ℕ := Martha_catch + Cara_catch
def Andrew_catch : ℕ := T^2 + 2
def F : ℕ := Martha_catch + Cara_catch + Andrew_catch
def Bella_catch : ℕ := 2 ^ (F / 3)

theorem Bella_catch_correct : Bella_catch = 2 ^ 1102 := by
  sorry

end NUMINAMATH_GPT_Bella_catch_correct_l786_78686


namespace NUMINAMATH_GPT_faye_total_books_l786_78641

def initial_books : ℕ := 34
def books_given_away : ℕ := 3
def books_bought : ℕ := 48

theorem faye_total_books : initial_books - books_given_away + books_bought = 79 :=
by
  sorry

end NUMINAMATH_GPT_faye_total_books_l786_78641


namespace NUMINAMATH_GPT_optimal_cylinder_dimensions_l786_78606

variable (R : ℝ)

noncomputable def optimal_cylinder_height : ℝ := (2 * R) / Real.sqrt 3
noncomputable def optimal_cylinder_radius : ℝ := R * Real.sqrt (2 / 3)

theorem optimal_cylinder_dimensions :
  ∃ (h r : ℝ), 
    (h = optimal_cylinder_height R ∧ r = optimal_cylinder_radius R) ∧
    ∀ (h' r' : ℝ), (4 * R^2 = 4 * r'^2 + h'^2) → 
      (h' = optimal_cylinder_height R ∧ r' = optimal_cylinder_radius R) → 
      (π * r' ^ 2 * h' ≤ π * r ^ 2 * h) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_optimal_cylinder_dimensions_l786_78606


namespace NUMINAMATH_GPT_sum_of_distinct_products_l786_78634

theorem sum_of_distinct_products (G H : ℕ) (hG : G < 10) (hH : H < 10) :
  (3 * H + 8) % 8 = 0 ∧ ((6 + 2 + 8 + G + 4 + 0 + 9 + 3 + H + 8) % 9 = 0) →
  (G * H = 6 ∨ G * H = 48) →
  6 + 48 = 54 :=
by
  intros _ _
  sorry

end NUMINAMATH_GPT_sum_of_distinct_products_l786_78634


namespace NUMINAMATH_GPT_find_slope_of_intersecting_line_l786_78690

-- Define the conditions
def line_p (x : ℝ) : ℝ := 2 * x + 3
def line_q (x : ℝ) (m : ℝ) : ℝ := m * x + 1

-- Define the point of intersection
def intersection_point : ℝ × ℝ := (4, 11)

-- Prove that the slope m of line q such that both lines intersect at (4, 11) is 2.5
theorem find_slope_of_intersecting_line (m : ℝ) :
  line_q 4 m = 11 → m = 2.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_slope_of_intersecting_line_l786_78690


namespace NUMINAMATH_GPT_kerosene_price_increase_l786_78682

theorem kerosene_price_increase (P C : ℝ) (x : ℝ)
  (h1 : 1 = (1 + x / 100) * 0.8) :
  x = 25 := by
  sorry

end NUMINAMATH_GPT_kerosene_price_increase_l786_78682


namespace NUMINAMATH_GPT_find_a_plus_b_eq_102_l786_78601

theorem find_a_plus_b_eq_102 :
  ∃ (a b : ℕ), (1600^(1 / 2) - 24 = (a^(1 / 2) - b)^2) ∧ (a + b = 102) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_plus_b_eq_102_l786_78601


namespace NUMINAMATH_GPT_subtraction_contradiction_l786_78677

theorem subtraction_contradiction (k t : ℕ) (hk_non_zero : k ≠ 0) (ht_non_zero : t ≠ 0) : 
  ¬ ((8 * 100 + k * 10 + 8) - (k * 100 + 8 * 10 + 8) = 1 * 100 + 6 * 10 + t * 1) :=
by
  sorry

end NUMINAMATH_GPT_subtraction_contradiction_l786_78677


namespace NUMINAMATH_GPT_a_range_l786_78653

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.log x - (1 / 2) * x^2 + 3 * x

def is_monotonic_on_interval (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a (a + 1), 4 / x - x + 3 > 0

theorem a_range (a : ℝ) :
  is_monotonic_on_interval a → (0 < a ∧ a ≤ 3) :=
by 
  sorry

end NUMINAMATH_GPT_a_range_l786_78653


namespace NUMINAMATH_GPT_probability_at_least_one_pen_l786_78602

noncomputable def PAs  := 3/5
noncomputable def PBs  := 2/3
noncomputable def PABs := PAs * PBs

theorem probability_at_least_one_pen : PAs + PBs - PABs = 13 / 15 := by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_pen_l786_78602


namespace NUMINAMATH_GPT_divide_talers_l786_78611

theorem divide_talers (loaves1 loaves2 : ℕ) (coins : ℕ) (loavesShared : ℕ) :
  loaves1 = 3 → loaves2 = 5 → coins = 8 → loavesShared = (loaves1 + loaves2) →
  (3 - loavesShared / 3) * coins / loavesShared = 1 ∧ (5 - loavesShared / 3) * coins / loavesShared = 7 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_divide_talers_l786_78611


namespace NUMINAMATH_GPT_earnings_ratio_l786_78626

-- Definitions for conditions
def jerusha_earnings : ℕ := 68
def total_earnings : ℕ := 85
def lottie_earnings : ℕ := total_earnings - jerusha_earnings

-- Prove that the ratio of Jerusha's earnings to Lottie's earnings is 4:1
theorem earnings_ratio : 
  ∃ (k : ℕ), jerusha_earnings = k * lottie_earnings ∧ (jerusha_earnings + lottie_earnings = total_earnings) ∧ (jerusha_earnings = 68) ∧ (total_earnings = 85) →
  68 / (total_earnings - 68) = 4 := 
by
  sorry

end NUMINAMATH_GPT_earnings_ratio_l786_78626


namespace NUMINAMATH_GPT_area_of_curve_l786_78603

noncomputable def polar_curve (φ : Real) : Real :=
  (1 / 2) + Real.sin φ

noncomputable def area_enclosed_by_polar_curve : Real :=
  2 * ((1 / 2) * ∫ (φ : Real) in (-Real.pi / 2)..(Real.pi / 2), (polar_curve φ) ^ 2)

theorem area_of_curve : area_enclosed_by_polar_curve = (3 * Real.pi) / 4 :=
by
  sorry

end NUMINAMATH_GPT_area_of_curve_l786_78603


namespace NUMINAMATH_GPT_gingerbreads_per_tray_l786_78664

-- Given conditions
def total_baked_gb (x : ℕ) : Prop := 4 * 25 + 3 * x = 160

-- The problem statement
theorem gingerbreads_per_tray (x : ℕ) (h : total_baked_gb x) : x = 20 := 
by sorry

end NUMINAMATH_GPT_gingerbreads_per_tray_l786_78664


namespace NUMINAMATH_GPT_unique_real_y_l786_78612

def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

theorem unique_real_y (y : ℝ) : (∃! y : ℝ, star 4 y = 10) :=
  by {
    sorry
  }

end NUMINAMATH_GPT_unique_real_y_l786_78612


namespace NUMINAMATH_GPT_sin_cos_sum_2018_l786_78699

theorem sin_cos_sum_2018 {x : ℝ} (h : Real.sin x + Real.cos x = 1) :
  (Real.sin x)^2018 + (Real.cos x)^2018 = 1 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_sum_2018_l786_78699


namespace NUMINAMATH_GPT_cos_theta_value_sin_theta_plus_pi_over_3_value_l786_78656

variable (θ : ℝ)
variable (H1 : 0 < θ ∧ θ < π / 2)
variable (H2 : Real.sin θ = 4 / 5)

theorem cos_theta_value : Real.cos θ = 3 / 5 := sorry

theorem sin_theta_plus_pi_over_3_value : 
    Real.sin (θ + π / 3) = (4 + 3 * Real.sqrt 3) / 10 := sorry

end NUMINAMATH_GPT_cos_theta_value_sin_theta_plus_pi_over_3_value_l786_78656


namespace NUMINAMATH_GPT_longest_side_of_triangle_l786_78628

theorem longest_side_of_triangle (x : ℝ) (h1 : 8 + (2 * x + 5) + (3 * x + 2) = 40) : 
  max (max 8 (2 * x + 5)) (3 * x + 2) = 17 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_longest_side_of_triangle_l786_78628


namespace NUMINAMATH_GPT_solve_equation_l786_78655

theorem solve_equation (x : ℝ) (h : 3 + 1 / (2 - x) = 2 * (1 / (2 - x))) : x = 5 / 3 := 
  sorry

end NUMINAMATH_GPT_solve_equation_l786_78655


namespace NUMINAMATH_GPT_line_tangent_to_circle_l786_78689

theorem line_tangent_to_circle (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 2 * k + 3 = 0 → x^2 + (y + 1)^2 = 4) → k = 3 / 4 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_line_tangent_to_circle_l786_78689


namespace NUMINAMATH_GPT_relationship_between_y_values_l786_78692

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variables (m : ℝ) (y1 y2 y3 : ℝ)
variables (h : m > 0)
variables (h1 : y1 = quadratic_function m (-1))
variables (h2 : y2 = quadratic_function m (5 / 2))
variables (h3 : y3 = quadratic_function m 6)

theorem relationship_between_y_values : y3 > y1 ∧ y1 > y2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_y_values_l786_78692


namespace NUMINAMATH_GPT_arcsin_of_neg_one_l786_78672

theorem arcsin_of_neg_one : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_of_neg_one_l786_78672


namespace NUMINAMATH_GPT_four_nat_nums_prime_condition_l786_78697

theorem four_nat_nums_prime_condition (a b c d : ℕ) (h₁ : a = 1) (h₂ : b = 2) (h₃ : c = 3) (h₄ : d = 5) :
  Nat.Prime (a * b + c * d) ∧ Nat.Prime (a * c + b * d) ∧ Nat.Prime (a * d + b * c) :=
by
  sorry

end NUMINAMATH_GPT_four_nat_nums_prime_condition_l786_78697


namespace NUMINAMATH_GPT_find_schnauzers_l786_78695

theorem find_schnauzers (D S : ℕ) (h : 3 * D - 5 + (D - S) = 90) (hD : D = 20) : S = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_schnauzers_l786_78695


namespace NUMINAMATH_GPT_no_real_solution_l786_78657

theorem no_real_solution :
  ∀ x : ℝ, ((x - 4 * x + 15)^2 + 3)^2 + 1 ≠ -|x|^2 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_real_solution_l786_78657


namespace NUMINAMATH_GPT_sqrt7_minus_3_lt_sqrt5_minus_2_l786_78650

theorem sqrt7_minus_3_lt_sqrt5_minus_2:
  (2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3) ∧ (2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3) -> 
  Real.sqrt 7 - 3 < Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_GPT_sqrt7_minus_3_lt_sqrt5_minus_2_l786_78650


namespace NUMINAMATH_GPT_total_bottles_capped_in_10_minutes_l786_78669

-- Define the capacities per minute for the three machines
def machine_a_capacity : ℕ := 12
def machine_b_capacity : ℕ := machine_a_capacity - 2
def machine_c_capacity : ℕ := machine_b_capacity + 5

-- Define the total capping capacity for 10 minutes
def total_capacity_in_10_minutes (a b c : ℕ) : ℕ := a * 10 + b * 10 + c * 10

-- The theorem we aim to prove
theorem total_bottles_capped_in_10_minutes :
  total_capacity_in_10_minutes machine_a_capacity machine_b_capacity machine_c_capacity = 370 :=
by
  -- Directly use the capacities defined above
  sorry

end NUMINAMATH_GPT_total_bottles_capped_in_10_minutes_l786_78669


namespace NUMINAMATH_GPT_parallel_lines_slope_l786_78676

theorem parallel_lines_slope {a : ℝ} 
    (h1 : ∀ x y : ℝ, 4 * y + 3 * x - 5 = 0 → y = -3 / 4 * x + 5 / 4)
    (h2 : ∀ x y : ℝ, 6 * y + a * x + 4 = 0 → y = -a / 6 * x - 2 / 3)
    (h_parallel : ∀ x₁ y₁ x₂ y₂ : ℝ, (4 * y₁ + 3 * x₁ - 5 = 0 ∧ 6 * y₂ + a * x₂ + 4 = 0) → -3 / 4 = -a / 6) : 
  a = 4.5 := sorry

end NUMINAMATH_GPT_parallel_lines_slope_l786_78676


namespace NUMINAMATH_GPT_sum_of_zeros_gt_two_l786_78637

noncomputable def f (a x : ℝ) := 2 * a * Real.log x + x ^ 2 - 2 * (a + 1) * x

theorem sum_of_zeros_gt_two (a x1 x2 : ℝ) (h_a : -0.5 < a ∧ a < 0)
  (h_fx_zeros : f a x1 = 0 ∧ f a x2 = 0) (h_x_order : x1 < x2) : x1 + x2 > 2 := 
sorry

end NUMINAMATH_GPT_sum_of_zeros_gt_two_l786_78637


namespace NUMINAMATH_GPT_vertex_parabola_l786_78627

theorem vertex_parabola (h k : ℝ) : 
  (∀ x : ℝ, -((x - 2)^2) + 3 = k) → (h = 2 ∧ k = 3) :=
by 
  sorry

end NUMINAMATH_GPT_vertex_parabola_l786_78627


namespace NUMINAMATH_GPT_machine_B_fewer_bottles_l786_78675

-- Definitions and the main theorem statement
def MachineA_caps_per_minute : ℕ := 12
def MachineC_additional_capacity : ℕ := 5
def total_bottles_in_10_minutes : ℕ := 370

theorem machine_B_fewer_bottles (B : ℕ) 
  (h1 : MachineA_caps_per_minute * 10 + 10 * B + 10 * (B + MachineC_additional_capacity) = total_bottles_in_10_minutes) :
  MachineA_caps_per_minute - B = 2 :=
by
  sorry

end NUMINAMATH_GPT_machine_B_fewer_bottles_l786_78675


namespace NUMINAMATH_GPT_inequality_C_l786_78667

theorem inequality_C (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_C_l786_78667


namespace NUMINAMATH_GPT_number_of_apps_needed_l786_78600

-- Definitions based on conditions
variable (cost_per_app : ℕ) (total_money : ℕ) (remaining_money : ℕ)

-- Assume the conditions given
axiom cost_app_eq : cost_per_app = 4
axiom total_money_eq : total_money = 66
axiom remaining_money_eq : remaining_money = 6

-- The goal is to determine the number of apps Lidia needs to buy
theorem number_of_apps_needed (n : ℕ) (h : total_money - remaining_money = cost_per_app * n) :
  n = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_apps_needed_l786_78600


namespace NUMINAMATH_GPT_profit_percent_l786_78619

variable (P C : ℝ)
variable (h₁ : (2/3) * P = 0.84 * C)

theorem profit_percent (P C : ℝ) (h₁ : (2/3) * P = 0.84 * C) : 
  ((P - C) / C) * 100 = 26 :=
by
  sorry

end NUMINAMATH_GPT_profit_percent_l786_78619


namespace NUMINAMATH_GPT_negation_of_forall_prop_l786_78630

theorem negation_of_forall_prop :
  ¬ (∀ x : ℝ, x^2 + x > 0) ↔ ∃ x : ℝ, x^2 + x ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_forall_prop_l786_78630


namespace NUMINAMATH_GPT_cost_of_bananas_l786_78629

theorem cost_of_bananas (A B : ℝ) (h1 : A + B = 5) (h2 : 2 * A + B = 7) : B = 3 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_bananas_l786_78629


namespace NUMINAMATH_GPT_measles_cases_1995_l786_78615

-- Definitions based on the conditions
def initial_cases_1970 : ℕ := 300000
def final_cases_2000 : ℕ := 200
def cases_1990 : ℕ := 1000
def decrease_rate : ℕ := 14950 -- Annual linear decrease from 1970-1990
def a : ℤ := -8 -- Coefficient for the quadratic phase

-- Function modeling the number of cases in the quadratic phase (1990-2000)
def measles_cases (x : ℕ) : ℤ := a * (x - 1990)^2 + cases_1990

-- The statement we want to prove
theorem measles_cases_1995 : measles_cases 1995 = 800 := by
  sorry

end NUMINAMATH_GPT_measles_cases_1995_l786_78615


namespace NUMINAMATH_GPT_round_to_nearest_tenth_l786_78671

theorem round_to_nearest_tenth (x : Float) (h : x = 42.63518) : Float.round (x * 10) / 10 = 42.6 := by
  sorry

end NUMINAMATH_GPT_round_to_nearest_tenth_l786_78671


namespace NUMINAMATH_GPT_calculate_regular_rate_l786_78642

def regular_hours_per_week : ℕ := 6 * 10
def total_weeks : ℕ := 4
def total_regular_hours : ℕ := regular_hours_per_week * total_weeks
def total_worked_hours : ℕ := 245
def overtime_hours : ℕ := total_worked_hours - total_regular_hours
def overtime_rate : ℚ := 4.20
def total_earning : ℚ := 525
def total_overtime_pay : ℚ := overtime_hours * overtime_rate
def total_regular_pay : ℚ := total_earning - total_overtime_pay
def regular_rate : ℚ := total_regular_pay / total_regular_hours

theorem calculate_regular_rate : regular_rate = 2.10 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_calculate_regular_rate_l786_78642


namespace NUMINAMATH_GPT_increase_in_average_age_l786_78684

variable (A : ℝ)
variable (A_increase : ℝ)
variable (orig_age_sum : ℝ)
variable (new_age_sum : ℝ)

def original_total_age (A : ℝ) := 8 * A
def new_total_age (A : ℝ) := original_total_age A - 20 - 22 + 29 + 29

theorem increase_in_average_age (A : ℝ) (orig_age_sum := original_total_age A) (new_age_sum := new_total_age A) : 
  (new_age_sum / 8) = (A + 2) := 
by
  unfold new_total_age
  unfold original_total_age
  sorry

end NUMINAMATH_GPT_increase_in_average_age_l786_78684


namespace NUMINAMATH_GPT_jars_proof_l786_78670

def total_plums : ℕ := 240
def exchange_ratio : ℕ := 7
def mangoes_per_jar : ℕ := 5

def ripe_plums (total_plums : ℕ) := total_plums / 4
def unripe_plums (total_plums : ℕ) := 3 * total_plums / 4
def unripe_plums_kept : ℕ := 46

def plums_for_trade (total_plums unripe_plums_kept : ℕ) : ℕ :=
  ripe_plums total_plums + (unripe_plums total_plums - unripe_plums_kept)

def mangoes_received (plums_for_trade exchange_ratio : ℕ) : ℕ :=
  plums_for_trade / exchange_ratio

def jars_of_mangoes (mangoes_received mangoes_per_jar : ℕ) : ℕ :=
  mangoes_received / mangoes_per_jar

theorem jars_proof : jars_of_mangoes (mangoes_received (plums_for_trade total_plums unripe_plums_kept) exchange_ratio) mangoes_per_jar = 5 :=
by
  sorry

end NUMINAMATH_GPT_jars_proof_l786_78670


namespace NUMINAMATH_GPT_surveyed_parents_women_l786_78604

theorem surveyed_parents_women (W : ℝ) :
  (5/6 : ℝ) * W + (3/4 : ℝ) * (1 - W) = 0.8 → W = 0.6 :=
by
  intro h
  have hw : W * (1/6) + (1 - W) * (1/4) = 0.2 := sorry
  have : W = 0.6 := sorry
  exact this

end NUMINAMATH_GPT_surveyed_parents_women_l786_78604


namespace NUMINAMATH_GPT_point_relationship_l786_78679

variables {m x1 x2 y1 y2 : ℝ}

def quadratic_function (x : ℝ) (m : ℝ) : ℝ :=
  (x + m - 3)*(x - m) + 3

theorem point_relationship 
  (hx1_lt_x2 : x1 < x2)
  (hA : y1 = quadratic_function x1 m)
  (hB : y2 = quadratic_function x2 m)
  (h_sum_lt : x1 + x2 < 3) :
  y1 > y2 :=
sorry

end NUMINAMATH_GPT_point_relationship_l786_78679


namespace NUMINAMATH_GPT_man_rate_in_still_water_l786_78665

-- The conditions
def speed_with_stream : ℝ := 20
def speed_against_stream : ℝ := 4

-- The problem rephrased as a Lean statement
theorem man_rate_in_still_water : 
  (speed_with_stream + speed_against_stream) / 2 = 12 := 
by
  sorry

end NUMINAMATH_GPT_man_rate_in_still_water_l786_78665


namespace NUMINAMATH_GPT_school_students_l786_78649

theorem school_students (x y : ℕ) (h1 : x + y = 432) (h2 : x - 16 = (y + 16) + 24) : x = 244 ∧ y = 188 := by
  sorry

end NUMINAMATH_GPT_school_students_l786_78649


namespace NUMINAMATH_GPT_total_cups_used_l786_78632

theorem total_cups_used (butter flour sugar : ℕ) (h1 : 2 * sugar = 3 * butter) (h2 : 5 * sugar = 3 * flour) (h3 : sugar = 12) : butter + flour + sugar = 40 :=
by
  sorry

end NUMINAMATH_GPT_total_cups_used_l786_78632


namespace NUMINAMATH_GPT_original_money_l786_78668

theorem original_money (M : ℕ) (h1 : 3 * M / 8 ≤ M)
  (h2 : 1 * (M - 3 * M / 8) / 5 ≤ M - 3 * M / 8)
  (h3 : M - 3 * M / 8 - (1 * (M - 3 * M / 8) / 5) = 36) : M = 72 :=
sorry

end NUMINAMATH_GPT_original_money_l786_78668
