import Mathlib

namespace remainder_of_polynomial_l570_57006

theorem remainder_of_polynomial 
  (P : ℝ → ℝ) 
  (h₁ : P 15 = 16)
  (h₂ : P 10 = 4) :
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - 10) * (x - 15) * Q x + (12 / 5 * x - 20) :=
by
  sorry

end remainder_of_polynomial_l570_57006


namespace library_science_books_count_l570_57092

-- Definitions based on the problem conditions
def initial_science_books := 120
def borrowed_books := 40
def returned_books := 15
def books_on_hold := 10
def borrowed_from_other_library := 20
def lost_books := 2
def damaged_books := 1

-- Statement for the proof.
theorem library_science_books_count :
  initial_science_books - borrowed_books + returned_books - books_on_hold + borrowed_from_other_library - lost_books - damaged_books = 102 :=
by
  sorry

end library_science_books_count_l570_57092


namespace ball_travel_distance_l570_57048

noncomputable def total_distance : ℝ :=
  200 + (2 * (200 * (1 / 3))) + (2 * (200 * ((1 / 3) ^ 2))) +
  (2 * (200 * ((1 / 3) ^ 3))) + (2 * (200 * ((1 / 3) ^ 4)))

theorem ball_travel_distance :
  total_distance = 397.2 :=
by
  sorry

end ball_travel_distance_l570_57048


namespace sheep_drowned_proof_l570_57058

def animal_problem_statement (S : ℕ) : Prop :=
  let initial_sheep := 20
  let initial_cows := 10
  let initial_dogs := 14
  let total_animals_made_shore := 35
  let sheep_drowned := S
  let cows_drowned := 2 * S
  let dogs_survived := initial_dogs
  let animals_made_shore := initial_sheep + initial_cows + initial_dogs - (sheep_drowned + cows_drowned)
  30 - 3 * S = 35 - 14

theorem sheep_drowned_proof : ∃ S : ℕ, animal_problem_statement S ∧ S = 3 :=
by
  sorry

end sheep_drowned_proof_l570_57058


namespace surface_area_is_correct_l570_57055

structure CubicSolid where
  base_layer : ℕ
  second_layer : ℕ
  third_layer : ℕ
  top_layer : ℕ

def conditions : CubicSolid := ⟨4, 4, 3, 1⟩

theorem surface_area_is_correct : 
  (conditions.base_layer + conditions.second_layer + conditions.third_layer + conditions.top_layer + 7 + 7 + 3 + 3) = 28 := 
  by
  sorry

end surface_area_is_correct_l570_57055


namespace evaluate_f_g_3_l570_57075

def g (x : ℝ) := x^3
def f (x : ℝ) := 3 * x - 2

theorem evaluate_f_g_3 : f (g 3) = 79 := by
  sorry

end evaluate_f_g_3_l570_57075


namespace find_f_6_5_l570_57061

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : is_even_function f
axiom periodic_f : ∀ x, f (x + 4) = f x
axiom f_in_interval : ∀ x, 1 ≤ x ∧ x ≤ 2 → f x = x - 2

theorem find_f_6_5 : f 6.5 = -0.5 := by
  sorry

end find_f_6_5_l570_57061


namespace ninth_term_arithmetic_sequence_l570_57076

def first_term : ℚ := 3 / 4
def seventeenth_term : ℚ := 6 / 7

theorem ninth_term_arithmetic_sequence :
  let a1 := first_term
  let a17 := seventeenth_term
  (a1 + a17) / 2 = 45 / 56 := 
sorry

end ninth_term_arithmetic_sequence_l570_57076


namespace mass_percentage_of_H_in_ascorbic_acid_l570_57045

-- Definitions based on the problem conditions
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.01
def molar_mass_O : ℝ := 16.00

def ascorbic_acid_molecular_formula_C : ℝ := 6
def ascorbic_acid_molecular_formula_H : ℝ := 8
def ascorbic_acid_molecular_formula_O : ℝ := 6

noncomputable def ascorbic_acid_molar_mass : ℝ :=
  ascorbic_acid_molecular_formula_C * molar_mass_C + 
  ascorbic_acid_molecular_formula_H * molar_mass_H + 
  ascorbic_acid_molecular_formula_O * molar_mass_O

noncomputable def hydrogen_mass_in_ascorbic_acid : ℝ :=
  ascorbic_acid_molecular_formula_H * molar_mass_H

noncomputable def hydrogen_mass_percentage_in_ascorbic_acid : ℝ :=
  (hydrogen_mass_in_ascorbic_acid / ascorbic_acid_molar_mass) * 100

theorem mass_percentage_of_H_in_ascorbic_acid :
  hydrogen_mass_percentage_in_ascorbic_acid = 4.588 :=
by
  sorry

end mass_percentage_of_H_in_ascorbic_acid_l570_57045


namespace factor_tree_X_value_l570_57088

theorem factor_tree_X_value :
  let F := 2 * 5
  let G := 7 * 3
  let Y := 7 * F
  let Z := 11 * G
  let X := Y * Z
  X = 16170 := by
sorry

end factor_tree_X_value_l570_57088


namespace exam_students_l570_57040

noncomputable def totalStudents (N : ℕ) (T : ℕ) := T = 70 * N
noncomputable def marksOfExcludedStudents := 5 * 50
noncomputable def remainingStudents (N : ℕ) := N - 5
noncomputable def remainingMarksCondition (N T : ℕ) := (T - marksOfExcludedStudents) / remainingStudents N = 90

theorem exam_students (N : ℕ) (T : ℕ) 
  (h1 : totalStudents N T) 
  (h2 : remainingMarksCondition N T) : 
  N = 10 :=
by 
  sorry

end exam_students_l570_57040


namespace correct_reasoning_l570_57096

-- Define that every multiple of 9 is a multiple of 3
def multiple_of_9_is_multiple_of_3 : Prop :=
  ∀ n : ℤ, n % 9 = 0 → n % 3 = 0

-- Define that a certain odd number is a multiple of 9
def odd_multiple_of_9 (n : ℤ) : Prop :=
  n % 2 = 1 ∧ n % 9 = 0

-- The goal: Prove that the reasoning process is completely correct
theorem correct_reasoning (H1 : multiple_of_9_is_multiple_of_3)
                          (n : ℤ)
                          (H2 : odd_multiple_of_9 n) : 
                          (n % 3 = 0) :=
by
  -- Explanation of the proof here
  sorry

end correct_reasoning_l570_57096


namespace area_of_L_equals_22_l570_57065

-- Define the dimensions of the rectangles
def big_rectangle_length := 8
def big_rectangle_width := 5
def small_rectangle_length := big_rectangle_length - 2
def small_rectangle_width := big_rectangle_width - 2

-- Define the areas
def area_big_rectangle := big_rectangle_length * big_rectangle_width
def area_small_rectangle := small_rectangle_length * small_rectangle_width

-- Define the area of the "L" shape
def area_L := area_big_rectangle - area_small_rectangle

-- State the theorem
theorem area_of_L_equals_22 : area_L = 22 := by
  -- The proof would go here
  sorry

end area_of_L_equals_22_l570_57065


namespace correct_product_of_0_035_and_3_84_l570_57069

theorem correct_product_of_0_035_and_3_84 : 
  (0.035 * 3.84 = 0.1344) := sorry

end correct_product_of_0_035_and_3_84_l570_57069


namespace danny_initial_caps_l570_57023

-- Define the conditions
variables (lostCaps : ℕ) (currentCaps : ℕ)
-- Assume given conditions
axiom lost_caps_condition : lostCaps = 66
axiom current_caps_condition : currentCaps = 25

-- Define the total number of bottle caps Danny had at first
def originalCaps (lostCaps currentCaps : ℕ) : ℕ := lostCaps + currentCaps

-- State the theorem to prove the number of bottle caps Danny originally had is 91
theorem danny_initial_caps : originalCaps lostCaps currentCaps = 91 :=
by
  -- Insert the proof here when available
  sorry

end danny_initial_caps_l570_57023


namespace pencils_per_pack_l570_57009

def packs := 28
def rows := 42
def pencils_per_row := 16

theorem pencils_per_pack (total_pencils : ℕ) : total_pencils = rows * pencils_per_row → total_pencils / packs = 24 :=
by
  sorry

end pencils_per_pack_l570_57009


namespace probability_both_selected_l570_57087

theorem probability_both_selected (P_C : ℚ) (P_B : ℚ) (hC : P_C = 4/5) (hB : P_B = 3/5) : 
  ((4/5) * (3/5)) = (12/25) := by
  sorry

end probability_both_selected_l570_57087


namespace cheenu_time_difference_l570_57029

def cheenu_bike_time_per_mile (distance_bike : ℕ) (time_bike : ℕ) : ℕ := time_bike / distance_bike
def cheenu_walk_time_per_mile (distance_walk : ℕ) (time_walk : ℕ) : ℕ := time_walk / distance_walk
def time_difference (time1 : ℕ) (time2 : ℕ) : ℕ := time2 - time1

theorem cheenu_time_difference 
  (distance_bike : ℕ) (time_bike : ℕ) 
  (distance_walk : ℕ) (time_walk : ℕ) 
  (H_bike : distance_bike = 20) (H_time_bike : time_bike = 80) 
  (H_walk : distance_walk = 8) (H_time_walk : time_walk = 160) :
  time_difference (cheenu_bike_time_per_mile distance_bike time_bike) (cheenu_walk_time_per_mile distance_walk time_walk) = 16 := 
by
  sorry

end cheenu_time_difference_l570_57029


namespace acute_angle_of_parallelogram_l570_57098

theorem acute_angle_of_parallelogram
  (a b : ℝ) (h : a < b)
  (parallelogram_division : ∀ x y : ℝ, x + y = a → b = x + 2 * Real.sqrt (x * y) + y) :
  ∃ α : ℝ, α = Real.arcsin ((b / a) - 1) :=
sorry

end acute_angle_of_parallelogram_l570_57098


namespace radian_measure_of_sector_l570_57059

theorem radian_measure_of_sector
  (perimeter : ℝ) (area : ℝ) (radian_measure : ℝ)
  (h1 : perimeter = 8)
  (h2 : area = 4) :
  radian_measure = 2 :=
sorry

end radian_measure_of_sector_l570_57059


namespace equal_number_of_digits_l570_57097

noncomputable def probability_equal_digits : ℚ := (20 * (9/16)^3 * (7/16)^3)

theorem equal_number_of_digits :
  probability_equal_digits = 3115125 / 10485760 := by
  sorry

end equal_number_of_digits_l570_57097


namespace log_function_domain_l570_57077

theorem log_function_domain (x : ℝ) : 
  (3 - x > 0) ∧ (x - 1 > 0) ∧ (x - 1 ≠ 1) -> (1 < x ∧ x < 3 ∧ x ≠ 2) :=
by
  intro h
  sorry

end log_function_domain_l570_57077


namespace avg_age_family_now_l570_57015

namespace average_age_family

-- Define initial conditions
def avg_age_husband_wife_marriage := 23
def years_since_marriage := 5
def age_child := 1
def number_of_family_members := 3

-- Prove that the average age of the family now is 19 years
theorem avg_age_family_now :
  (2 * avg_age_husband_wife_marriage + 2 * years_since_marriage + age_child) / number_of_family_members = 19 := by
  sorry

end average_age_family

end avg_age_family_now_l570_57015


namespace lollipop_distribution_l570_57070

theorem lollipop_distribution 
  (P1 P2 P_total L x : ℕ) 
  (h1 : P1 = 45) 
  (h2 : P2 = 15) 
  (h3 : L = 12) 
  (h4 : P_total = P1 + P2) 
  (h5 : P_total = 60) : 
  x = 5 := 
by 
  sorry

end lollipop_distribution_l570_57070


namespace find_modulus_l570_57017

open Complex -- Open the Complex namespace for convenience

noncomputable def modulus_of_z (a : ℝ) (h : (1 + 2 * Complex.I) * (a + Complex.I : ℂ) = Complex.re ((1 + 2 * Complex.I) * (a + Complex.I)) + Complex.im ((1 + 2 * Complex.I) * (a + Complex.I)) * Complex.I) : ℝ :=
  Complex.abs ((1 + 2 * Complex.I) * (a + Complex.I))

theorem find_modulus : modulus_of_z (-3) (by {
  -- Provide the condition that real part equals imaginary part
  admit -- This 'admit' serves as a placeholder for the proof of the condition 
}) = 5 * Real.sqrt 2 := sorry

end find_modulus_l570_57017


namespace smallest_possible_recording_l570_57000

theorem smallest_possible_recording :
  ∃ (A B C : ℤ), 
      (0 ≤ A ∧ A ≤ 10) ∧ 
      (0 ≤ B ∧ B ≤ 10) ∧ 
      (0 ≤ C ∧ C ≤ 10) ∧ 
      (A + B + C = 12) ∧ 
      (A + B + C) % 5 = 0 ∧ 
      A = 0 :=
by
  sorry

end smallest_possible_recording_l570_57000


namespace fraction_addition_l570_57073

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l570_57073


namespace workshop_cost_l570_57056

theorem workshop_cost
  (x : ℝ)
  (h1 : 0 < x) -- Given the cost must be positive
  (h2 : (x / 4) - 15 = x / 7) :
  x = 140 :=
by
  sorry

end workshop_cost_l570_57056


namespace inequality_proof_l570_57042

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l570_57042


namespace complementary_angles_of_same_angle_are_equal_l570_57037

def complementary_angles (α β : ℝ) := α + β = 90 

theorem complementary_angles_of_same_angle_are_equal 
        (θ : ℝ) (α β : ℝ) 
        (h1 : complementary_angles θ α) 
        (h2 : complementary_angles θ β) : 
        α = β := 
by 
  sorry

end complementary_angles_of_same_angle_are_equal_l570_57037


namespace depth_notation_l570_57078

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l570_57078


namespace geom_seq_value_l570_57050

noncomputable def geom_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ (n : ℕ), a (n + 1) = a n * q

theorem geom_seq_value
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom_seq : geom_sequence a q)
  (h_a5 : a 5 = 2)
  (h_a6_a8 : a 6 * a 8 = 8) :
  (a 2018 - a 2016) / (a 2014 - a 2012) = 2 :=
sorry

end geom_seq_value_l570_57050


namespace sum_smallest_largest_even_integers_l570_57012

theorem sum_smallest_largest_even_integers (m b z : ℕ) (hm_even : m % 2 = 0)
  (h_mean : z = (b + (b + 2 * (m - 1))) / 2) :
  (b + (b + 2 * (m - 1))) = 2 * z :=
by
  sorry

end sum_smallest_largest_even_integers_l570_57012


namespace number_of_moles_of_OC_NH2_2_formed_l570_57081

-- Definition: Chemical reaction condition
def reaction_eqn (x y : ℕ) : Prop := 
  x ≥ 1 ∧ y ≥ 2 ∧ x * 2 = y

-- Theorem: Prove that combining 3 moles of CO2 and 6 moles of NH3 results in 3 moles of OC(NH2)2
theorem number_of_moles_of_OC_NH2_2_formed (x y : ℕ) 
(h₁ : reaction_eqn x y)
(h₂ : x = 3)
(h₃ : y = 6) : 
x =  y / 2 :=
by {
    -- Proof is not provided
    sorry 
}

end number_of_moles_of_OC_NH2_2_formed_l570_57081


namespace total_distance_between_first_and_fifth_poles_l570_57039

noncomputable def distance_between_poles (n : ℕ) (d : ℕ) : ℕ :=
  d / n

theorem total_distance_between_first_and_fifth_poles :
  ∀ (n : ℕ) (d : ℕ), (n = 3 ∧ d = 90) → (4 * distance_between_poles n d = 120) :=
by
  sorry

end total_distance_between_first_and_fifth_poles_l570_57039


namespace contrapositive_inverse_converse_negation_false_l570_57011

theorem contrapositive (a b : ℤ) : (a ≤ b) → (a - 2 ≤ b - 2) :=
sorry

theorem inverse (a b : ℤ) : (a - 2 ≤ b - 2) → (a ≤ b) :=
sorry

theorem converse (a b : ℤ) : (a - 2 > b - 2) → (a > b) :=
sorry

theorem negation_false (a b : ℤ) : ¬ ((a > b) → (a - 2 ≤ b - 2)) :=
sorry

end contrapositive_inverse_converse_negation_false_l570_57011


namespace exceeded_by_600_l570_57062

noncomputable def ken_collected : ℕ := 600
noncomputable def mary_collected (ken : ℕ) : ℕ := 5 * ken
noncomputable def scott_collected (mary : ℕ) : ℕ := mary / 3
noncomputable def total_collected (ken mary scott : ℕ) : ℕ := ken + mary + scott
noncomputable def goal : ℕ := 4000
noncomputable def exceeded_goal (total goal : ℕ) : ℕ := total - goal

theorem exceeded_by_600 : exceeded_goal (total_collected ken_collected (mary_collected ken_collected) (scott_collected (mary_collected ken_collected))) goal = 600 := by
  sorry

end exceeded_by_600_l570_57062


namespace projection_area_rectangular_board_l570_57093

noncomputable def projection_area (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) : ℝ :=
  let width := AB
  let height := BC
  let shadow_width := 5
  (1 / 2) * (width + shadow_width) * height

theorem projection_area_rectangular_board (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) :
  AB = 3 → BC = 2 → NE = 3 → MN = 5 → projection_area AB BC NE MN ABCD_perp_ground E_mid_AB light_at_M = 8 :=
by
  intros
  sorry

end projection_area_rectangular_board_l570_57093


namespace max_square_side_length_l570_57026

theorem max_square_side_length (AC BC : ℝ) (hAC : AC = 3) (hBC : BC = 7) : 
  ∃ s : ℝ, s = 2.1 := by
  sorry

end max_square_side_length_l570_57026


namespace greatest_difference_four_digit_numbers_l570_57090

theorem greatest_difference_four_digit_numbers : 
  ∃ (d1 d2 d3 d4 : ℕ), (d1 = 0 ∨ d1 = 3 ∨ d1 = 4 ∨ d1 = 8) ∧ 
                      (d2 = 0 ∨ d2 = 3 ∨ d2 = 4 ∨ d2 = 8) ∧ 
                      (d3 = 0 ∨ d3 = 3 ∨ d3 = 4 ∨ d3 = 8) ∧ 
                      (d4 = 0 ∨ d4 = 3 ∨ d4 = 4 ∨ d4 = 8) ∧ 
                      d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ 
                      d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧ 
                      (∃ n1 n2, n1 = 1000 * 8 + 100 * 4 + 10 * 3 + 0 ∧ 
                                n2 = 1000 * 3 + 100 * 0 + 10 * 4 + 8 ∧ 
                                n1 - n2 = 5382) :=
by {
  sorry
}

end greatest_difference_four_digit_numbers_l570_57090


namespace option_c_correct_l570_57063

theorem option_c_correct (x y : ℝ) (h : x < y) : -x > -y := 
sorry

end option_c_correct_l570_57063


namespace point_B_coordinates_l570_57025

def move_up (x y : Int) (units : Int) : Int := y + units
def move_left (x y : Int) (units : Int) : Int := x - units

theorem point_B_coordinates :
  let A : Int × Int := (1, -1)
  let B : Int × Int := (move_left A.1 A.2 3, move_up A.1 A.2 2)
  B = (-2, 1) := 
by
  -- This is where the proof would go, but we omit it with "sorry"
  sorry

end point_B_coordinates_l570_57025


namespace find_functional_form_l570_57064

theorem find_functional_form (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by
  sorry

end find_functional_form_l570_57064


namespace evaluate_polynomial_at_3_l570_57074

noncomputable def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

theorem evaluate_polynomial_at_3 : f 3 = 1 := by
  sorry

end evaluate_polynomial_at_3_l570_57074


namespace stickers_on_first_day_l570_57084

theorem stickers_on_first_day (s e total : ℕ) (h1 : e = 22) (h2 : total = 61) (h3 : total = s + e) : s = 39 :=
by
  sorry

end stickers_on_first_day_l570_57084


namespace similar_polygon_area_sum_l570_57068

theorem similar_polygon_area_sum 
  (t1 t2 a1 a2 b : ℝ)
  (h_ratio: t1 / t2 = a1^2 / a2^2)
  (t3 : ℝ := t1 + t2)
  (h_area_eq : t3 = b^2 * a1^2 / a2^2): 
  b = Real.sqrt (a1^2 + a2^2) :=
by
  sorry

end similar_polygon_area_sum_l570_57068


namespace expression_divisible_by_17_l570_57080

theorem expression_divisible_by_17 (n : ℕ) : 
  (6^(2*n) + 2^(n+2) + 12 * 2^n) % 17 = 0 :=
by
  sorry

end expression_divisible_by_17_l570_57080


namespace toothpick_count_l570_57066

theorem toothpick_count (height width : ℕ) (h_height : height = 20) (h_width : width = 10) : 
  (21 * width + 11 * height) = 430 :=
by
  sorry

end toothpick_count_l570_57066


namespace solve_system_of_equations_l570_57041

theorem solve_system_of_equations 
  (a b c s : ℝ) (x y z : ℝ)
  (h1 : y^2 - z * x = a * (x + y + z)^2)
  (h2 : x^2 - y * z = b * (x + y + z)^2)
  (h3 : z^2 - x * y = c * (x + y + z)^2)
  (h4 : a^2 + b^2 + c^2 - (a * b + b * c + c * a) = a + b + c) :
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ x + y + z = 0) ∨
  ((x + y + z ≠ 0) ∧
   (x = (2 * c - a - b + 1) * s) ∧
   (y = (2 * a - b - c + 1) * s) ∧
   (z = (2 * b - c - a + 1) * s)) :=
by
  sorry

end solve_system_of_equations_l570_57041


namespace largest_good_number_smallest_bad_number_l570_57003

def is_good_number (M : ℕ) : Prop :=
  ∃ a b c d : ℤ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

theorem largest_good_number :
  ∀ M : ℕ, is_good_number M ↔ M ≤ 576 :=
by sorry

theorem smallest_bad_number :
  ∀ M : ℕ, ¬ is_good_number M ↔ M ≥ 443 :=
by sorry

end largest_good_number_smallest_bad_number_l570_57003


namespace incorrect_population_growth_statement_l570_57046

def population_growth_behavior (p: ℝ → ℝ) : Prop :=
(p 0 < p 1) ∧ (∃ t₁ t₂, t₁ < t₂ ∧ (∀ t < t₁, p t < p (t + 1)) ∧
 (∀ t > t₁, (p t < p (t - 1)) ∨ (p t = p (t - 1))))

def stabilizes_at_K (p: ℝ → ℝ) (K: ℝ) : Prop :=
∃ t₀, ∀ t > t₀, p t = K

def K_value_definition (K: ℝ) (environmental_conditions: ℝ → ℝ) : Prop :=
∀ t, environmental_conditions t = K

theorem incorrect_population_growth_statement (p: ℝ → ℝ) (K: ℝ) (environmental_conditions: ℝ → ℝ)
(h1: population_growth_behavior p)
(h2: stabilizes_at_K p K)
(h3: K_value_definition K environmental_conditions) :
(p 0 > p 1) ∨ (¬ (∃ t₁ t₂, t₁ < t₂ ∧ (∀ t < t₁, p t < p (t + 1)) ∧
 (∀ t > t₁, (p t < p (t - 1)) ∨ (p t = p (t - 1))))) :=
sorry

end incorrect_population_growth_statement_l570_57046


namespace cyclist_A_speed_l570_57060

theorem cyclist_A_speed (a b : ℝ) (h1 : b = a + 5)
    (h2 : 80 / a = 120 / b) : a = 10 :=
by
  sorry

end cyclist_A_speed_l570_57060


namespace union_complement_eq_l570_57031

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l570_57031


namespace number_of_four_digit_integers_with_digit_sum_nine_l570_57086

theorem number_of_four_digit_integers_with_digit_sum_nine :
  ∃ (n : ℕ), (n = 165) ∧ (
    ∃ (a b c d : ℕ), 
      1 ≤ a ∧ 
      a + b + c + d = 9 ∧ 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (0 ≤ b ∧ b ≤ 9) ∧ 
      (0 ≤ c ∧ c ≤ 9) ∧ 
      (0 ≤ d ∧ d ≤ 9)) := 
sorry

end number_of_four_digit_integers_with_digit_sum_nine_l570_57086


namespace james_total_payment_is_correct_l570_57014

-- Define the constants based on the conditions
def numDirtBikes : Nat := 3
def costPerDirtBike : Nat := 150
def numOffRoadVehicles : Nat := 4
def costPerOffRoadVehicle : Nat := 300
def numTotalVehicles : Nat := numDirtBikes + numOffRoadVehicles
def registrationCostPerVehicle : Nat := 25

-- Define the total calculation using the given conditions
def totalPaidByJames : Nat :=
  (numDirtBikes * costPerDirtBike) +
  (numOffRoadVehicles * costPerOffRoadVehicle) +
  (numTotalVehicles * registrationCostPerVehicle)

-- State the proof problem
theorem james_total_payment_is_correct : totalPaidByJames = 1825 := by
  sorry

end james_total_payment_is_correct_l570_57014


namespace union_of_A_and_B_l570_57079

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {1, 2}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 2} := sorry

end union_of_A_and_B_l570_57079


namespace interest_rate_proof_l570_57083

noncomputable def compound_interest_rate (P A : ℝ) (n : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r)^n

noncomputable def interest_rate (initial  final: ℝ) (years : ℕ) : ℝ := 
  (4: ℝ)^(1/(years: ℝ)) - 1

theorem interest_rate_proof :
  compound_interest_rate 8000 32000 36 (interest_rate 8000 32000 36) ∧
  abs (interest_rate 8000 32000 36 * 100 - 3.63) < 0.01 :=
by
  -- Conditions from the problem for compound interest
  -- Using the formula for interest rate and the condition checks
  sorry

end interest_rate_proof_l570_57083


namespace fraction_addition_l570_57051

theorem fraction_addition : (2 / 5 + 3 / 8) = 31 / 40 :=
by
  sorry

end fraction_addition_l570_57051


namespace apple_counts_l570_57089

theorem apple_counts (x y : ℤ) (h1 : y - x = 2) (h2 : y = 3 * x - 4) : x = 3 ∧ y = 5 := 
by
  sorry

end apple_counts_l570_57089


namespace functional_eq_solution_l570_57044

theorem functional_eq_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) →
  (∀ x : ℝ, f x = x) :=
by
  intro h
  sorry

end functional_eq_solution_l570_57044


namespace mike_max_marks_l570_57053

theorem mike_max_marks
  (M : ℝ)
  (h1 : 0.30 * M = 234)
  (h2 : 234 = 212 + 22) : M = 780 := 
sorry

end mike_max_marks_l570_57053


namespace islander_parity_l570_57047

-- Define the concept of knights and liars
def is_knight (x : ℕ) : Prop := x % 2 = 0 -- Knight count is even
def is_liar (x : ℕ) : Prop := ¬(x % 2 = 1) -- Liar count being odd is false, so even

-- Define the total inhabitants on the island and conditions
theorem islander_parity (K L : ℕ) (h₁ : is_knight K) (h₂ : is_liar L) (h₃ : K + L = 2021) : false := sorry

end islander_parity_l570_57047


namespace solution_set_of_inequality_l570_57008

theorem solution_set_of_inequality :
  {x : ℝ // (2 < x ∨ x < 2) ∧ x ≠ 3} =
  {x : ℝ // x < 2 ∨ 3 < x } :=
sorry

end solution_set_of_inequality_l570_57008


namespace abs_neg_2023_l570_57085

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l570_57085


namespace greatest_sum_of_int_pairs_squared_eq_64_l570_57002

theorem greatest_sum_of_int_pairs_squared_eq_64 :
  ∃ (x y : ℤ), x^2 + y^2 = 64 ∧ (∀ (a b : ℤ), a^2 + b^2 = 64 → a + b ≤ 8) ∧ x + y = 8 :=
by 
  sorry

end greatest_sum_of_int_pairs_squared_eq_64_l570_57002


namespace reciprocals_and_opposites_l570_57091

theorem reciprocals_and_opposites (a b c d : ℝ) (h_ab : a * b = 1) (h_cd : c + d = 0) : 
  (c + d)^2 - a * b = -1 := by
  sorry

end reciprocals_and_opposites_l570_57091


namespace problem_solution_sets_l570_57036

theorem problem_solution_sets (x y : ℝ) :
  (x^2 * y + y^3 = 2 * x^2 + 2 * y^2 ∧ x * y + 1 = x + y) →
  ( (x = 0 ∧ y = 0) ∨ y = 2 ∨ x = 1 ∨ y = 1 ) :=
by
  sorry

end problem_solution_sets_l570_57036


namespace train_distance_l570_57027

theorem train_distance (t : ℕ) (d : ℕ) (rate : d / t = 1 / 2) (total_time : ℕ) (h : total_time = 90) : ∃ distance : ℕ, distance = 45 := by
  sorry

end train_distance_l570_57027


namespace condition_sufficient_not_necessary_l570_57049

theorem condition_sufficient_not_necessary (x : ℝ) :
  (0 < x ∧ x < 2) → (x < 2) ∧ ¬((x < 2) → (0 < x ∧ x < 2)) :=
by
  sorry

end condition_sufficient_not_necessary_l570_57049


namespace infinite_geometric_series_sum_l570_57038

theorem infinite_geometric_series_sum (a : ℕ → ℝ) (a1 : a 1 = 1) (r : ℝ) (h : r = 1 / 3) (S : ℝ) (H : S = a 1 / (1 - r)) : S = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l570_57038


namespace lisa_total_spoons_l570_57033

def num_children := 4
def spoons_per_child := 3
def decorative_spoons := 2
def large_spoons := 10
def teaspoons := 15

def total_spoons := num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

theorem lisa_total_spoons : total_spoons = 39 := by
  sorry

end lisa_total_spoons_l570_57033


namespace diff_of_squares_expression_l570_57034

theorem diff_of_squares_expression (m n : ℝ) :
  (3 * m + n) * (3 * m - n) = (3 * m)^2 - n^2 :=
by
  sorry

end diff_of_squares_expression_l570_57034


namespace sum_of_ages_l570_57067

theorem sum_of_ages (age1 age2 age3 : ℕ) (h : age1 * age2 * age3 = 128) : age1 + age2 + age3 = 18 :=
sorry

end sum_of_ages_l570_57067


namespace cuboid_surface_area_4_8_6_l570_57032

noncomputable def cuboid_surface_area (length width height : ℕ) : ℕ :=
  2 * (length * width + length * height + width * height)

theorem cuboid_surface_area_4_8_6 : cuboid_surface_area 4 8 6 = 208 := by
  sorry

end cuboid_surface_area_4_8_6_l570_57032


namespace marites_saves_120_per_year_l570_57057

def current_internet_speed := 10 -- Mbps
def current_monthly_bill := 20 -- dollars

def monthly_cost_20mbps := current_monthly_bill + 10 -- dollars
def monthly_cost_30mbps := current_monthly_bill * 2 -- dollars

def bundled_cost_20mbps := 80 -- dollars per month
def bundled_cost_30mbps := 90 -- dollars per month

def annual_cost_20mbps := bundled_cost_20mbps * 12 -- dollars per year
def annual_cost_30mbps := bundled_cost_30mbps * 12 -- dollars per year

theorem marites_saves_120_per_year :
  annual_cost_30mbps - annual_cost_20mbps = 120 := 
by
  sorry

end marites_saves_120_per_year_l570_57057


namespace problem_statement_l570_57004

theorem problem_statement (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end problem_statement_l570_57004


namespace base_conversion_and_addition_l570_57024

theorem base_conversion_and_addition :
  let n1 := 2 * (8:ℕ)^2 + 4 * 8^1 + 3 * 8^0
  let d1 := 1 * 4^1 + 3 * 4^0
  let n2 := 2 * 7^2 + 0 * 7^1 + 4 * 7^0
  let d2 := 2 * 5^1 + 3 * 5^0
  n1 / d1 + n2 / d2 = 31 + 51 / 91 := by
  sorry

end base_conversion_and_addition_l570_57024


namespace divisible_by_other_l570_57021

theorem divisible_by_other (y : ℕ) 
  (h1 : y = 20)
  (h2 : y % 4 = 0)
  (h3 : y % 8 ≠ 0) : (∃ n, n ≠ 4 ∧ y % n = 0 ∧ n = 5) :=
by 
  sorry

end divisible_by_other_l570_57021


namespace infinite_divisibility_of_2n_plus_n2_by_100_l570_57043

theorem infinite_divisibility_of_2n_plus_n2_by_100 :
  ∃ᶠ n in at_top, 100 ∣ (2^n + n^2) :=
sorry

end infinite_divisibility_of_2n_plus_n2_by_100_l570_57043


namespace find_J_salary_l570_57099

variable (J F M A : ℝ)

theorem find_J_salary (h1 : (J + F + M + A) / 4 = 8000) (h2 : (F + M + A + 6500) / 4 = 8900) :
  J = 2900 := by
  sorry

end find_J_salary_l570_57099


namespace stan_run_duration_l570_57016

def run_duration : ℕ := 100

def num_3_min_songs : ℕ := 10
def num_2_min_songs : ℕ := 15
def time_per_3_min_song : ℕ := 3
def time_per_2_min_song : ℕ := 2
def additional_time_needed : ℕ := 40

theorem stan_run_duration :
  (num_3_min_songs * time_per_3_min_song) + (num_2_min_songs * time_per_2_min_song) + additional_time_needed = run_duration := by
  sorry

end stan_run_duration_l570_57016


namespace f_leq_zero_l570_57030

noncomputable def f (x a : ℝ) := x * Real.log x - a * x^2 + (2 * a - 1) * x

theorem f_leq_zero (a x : ℝ) (h1 : 1/2 < a) (h2 : a ≤ 1) (hx : 0 < x) :
  f x a ≤ 0 :=
sorry

end f_leq_zero_l570_57030


namespace workdays_ride_l570_57005

-- Define the conditions
def work_distance : ℕ := 20
def weekend_ride : ℕ := 200
def speed : ℕ := 25
def hours_per_week : ℕ := 16

-- Define the question
def total_distance : ℕ := speed * hours_per_week
def distance_during_workdays : ℕ := total_distance - weekend_ride
def round_trip_distance : ℕ := 2 * work_distance

theorem workdays_ride : 
  (distance_during_workdays / round_trip_distance) = 5 :=
by
  sorry

end workdays_ride_l570_57005


namespace karen_piggy_bank_total_l570_57022

theorem karen_piggy_bank_total (a r n : ℕ) (h1 : a = 2) (h2 : r = 3) (h3 : n = 7) :
  (a * ((1 - r^n) / (1 - r))) = 2186 := by
  sorry

end karen_piggy_bank_total_l570_57022


namespace minerals_now_l570_57007

def minerals_yesterday (M : ℕ) : Prop := (M / 2 = 21)

theorem minerals_now (M : ℕ) (H : minerals_yesterday M) : (M + 6 = 48) :=
by 
  unfold minerals_yesterday at H
  sorry

end minerals_now_l570_57007


namespace problem_l570_57001

variable {x y : ℝ}

theorem problem (hx : 0 < x) (hy : 0 < y) (h : x^2 - y^2 = 3 * x * y) :
  (x^2 / y^2) + (y^2 / x^2) - 2 = 9 :=
sorry

end problem_l570_57001


namespace sum_of_squares_l570_57013

theorem sum_of_squares (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 512 * x ^ 3 + 125 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 6410 := 
sorry

end sum_of_squares_l570_57013


namespace cube_root_neg_eighth_l570_57054

theorem cube_root_neg_eighth : ∃ x : ℚ, x^3 = -1 / 8 ∧ x = -1 / 2 :=
by
  sorry

end cube_root_neg_eighth_l570_57054


namespace bob_after_alice_l570_57095

def race_distance : ℕ := 15
def alice_speed : ℕ := 7
def bob_speed : ℕ := 9

def alice_time : ℕ := alice_speed * race_distance
def bob_time : ℕ := bob_speed * race_distance

theorem bob_after_alice : bob_time - alice_time = 30 := by
  sorry

end bob_after_alice_l570_57095


namespace floor_difference_l570_57094

theorem floor_difference (x : ℝ) (h : x = 15.3) : 
  (⌊ x^2 ⌋ - ⌊ x ⌋ * ⌊ x ⌋ + 5) = 14 := 
by
  -- Skipping proof
  sorry

end floor_difference_l570_57094


namespace quadratic_polynomial_solution_l570_57052

theorem quadratic_polynomial_solution :
  ∃ a b c : ℚ, 
    (∀ x : ℚ, ax*x + bx + c = 8 ↔ x = -2) ∧ 
    (∀ x : ℚ, ax*x + bx + c = 2 ↔ x = 1) ∧ 
    (∀ x : ℚ, ax*x + bx + c = 10 ↔ x = 3) ∧ 
    a = 6 / 5 ∧ 
    b = -4 / 5 ∧ 
    c = 8 / 5 :=
by {
  sorry
}

end quadratic_polynomial_solution_l570_57052


namespace number_of_solutions_l570_57028

theorem number_of_solutions : ∃ (s : Finset ℕ), (∀ x ∈ s, 100 ≤ x^2 ∧ x^2 ≤ 200) ∧ s.card = 5 :=
by
  sorry

end number_of_solutions_l570_57028


namespace special_case_m_l570_57082

theorem special_case_m (m : ℝ) :
  (∀ x : ℝ, mx^2 - 4 * x + 3 = 0 → y = mx^2 - 4 * x + 3 → (x = 0 ∧ m = 0) ∨ (x ≠ 0 ∧ m = 4/3)) :=
sorry

end special_case_m_l570_57082


namespace sum_of_valid_two_digit_numbers_l570_57010

theorem sum_of_valid_two_digit_numbers
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (a - b) ∣ (10 * a + b))
  (h4 : (a * b) ∣ (10 * a + b)) :
  (10 * a + b = 21) → (21 = 21) :=
sorry

end sum_of_valid_two_digit_numbers_l570_57010


namespace translated_point_is_correct_l570_57019

-- Cartesian Point definition
structure Point where
  x : Int
  y : Int

-- Define the translation function
def translate (p : Point) (dx dy : Int) : Point :=
  Point.mk (p.x + dx) (p.y - dy)

-- Define the initial point A and the translation amounts
def A : Point := ⟨-3, 2⟩
def dx : Int := 3
def dy : Int := 2

-- The proof goal
theorem translated_point_is_correct :
  translate A dx dy = ⟨0, 0⟩ :=
by
  -- This is where the proof would normally go
  sorry

end translated_point_is_correct_l570_57019


namespace gcd_of_168_56_224_l570_57035

theorem gcd_of_168_56_224 : (Nat.gcd 168 56 = 56) ∧ (Nat.gcd 56 224 = 56) ∧ (Nat.gcd 168 224 = 56) :=
by
  sorry

end gcd_of_168_56_224_l570_57035


namespace min_m_n_l570_57018

theorem min_m_n (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 108 * m = n^3) : m + n = 8 :=
sorry

end min_m_n_l570_57018


namespace claire_gerbils_l570_57071

theorem claire_gerbils (G H : ℕ) (h1 : G + H = 92) (h2 : (1/4 : ℚ) * G + (1/3 : ℚ) * H = 25) : G = 68 :=
sorry

end claire_gerbils_l570_57071


namespace nails_remaining_l570_57072

theorem nails_remaining (nails_initial : ℕ) (kitchen_fraction : ℚ) (fence_fraction : ℚ) (nails_used_kitchen : ℕ) (nails_remaining_after_kitchen : ℕ) (nails_used_fence : ℕ) (nails_remaining_final : ℕ) 
  (h1 : nails_initial = 400) 
  (h2 : kitchen_fraction = 0.30) 
  (h3 : nails_used_kitchen = kitchen_fraction * nails_initial) 
  (h4 : nails_remaining_after_kitchen = nails_initial - nails_used_kitchen) 
  (h5 : fence_fraction = 0.70) 
  (h6 : nails_used_fence = fence_fraction * nails_remaining_after_kitchen) 
  (h7 : nails_remaining_final = nails_remaining_after_kitchen - nails_used_fence) :
  nails_remaining_final = 84 := by
sorry

end nails_remaining_l570_57072


namespace cylinder_height_l570_57020

theorem cylinder_height (h : ℝ)
  (circumference : ℝ)
  (rectangle_diagonal : ℝ)
  (C_eq : circumference = 12)
  (d_eq : rectangle_diagonal = 20) :
  h = 16 :=
by
  -- We derive the result based on the given conditions and calculations
  sorry -- Skipping the proof part

end cylinder_height_l570_57020
