import Mathlib

namespace NUMINAMATH_GPT_range_y_over_x_l1539_153989

theorem range_y_over_x {x y : ℝ} (h : (x-4)^2 + (y-2)^2 ≤ 4) : 
  ∃ k : ℝ, k = y / x ∧ 0 ≤ k ∧ k ≤ 4/3 :=
sorry

end NUMINAMATH_GPT_range_y_over_x_l1539_153989


namespace NUMINAMATH_GPT_arithmetic_sequence_eighth_term_l1539_153969

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_eighth_term_l1539_153969


namespace NUMINAMATH_GPT_min_value_f_l1539_153915

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem min_value_f (a : ℝ) (h : -2 < a) :
  ∃ m, (∀ x ∈ Set.Icc (-2 : ℝ) a, f x ≥ m) ∧ 
  ((a ≤ 1 → m = a^2 - 2 * a) ∧ (1 < a → m = -1)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_l1539_153915


namespace NUMINAMATH_GPT_students_taking_neither_l1539_153998

theorem students_taking_neither (total_students music art science music_and_art music_and_science art_and_science three_subjects : ℕ)
  (h1 : total_students = 800)
  (h2 : music = 80)
  (h3 : art = 60)
  (h4 : science = 50)
  (h5 : music_and_art = 30)
  (h6 : music_and_science = 25)
  (h7 : art_and_science = 20)
  (h8 : three_subjects = 15) :
  total_students - (music + art + science - music_and_art - music_and_science - art_and_science + three_subjects) = 670 :=
by sorry

end NUMINAMATH_GPT_students_taking_neither_l1539_153998


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l1539_153986

noncomputable def log_base_1_div_3 (t : ℝ) := Real.log t / Real.log (1/3)

def quadratic (x : ℝ) := 4 + 3 * x - x^2

theorem monotonic_increasing_interval :
  ∃ (a b : ℝ), (∀ x, a < x ∧ x < b → (log_base_1_div_3 (quadratic x)) < (log_base_1_div_3 (quadratic (x + ε))) ∧
               ((-1 : ℝ) < x ∧ x < 4) ∧ (quadratic x > 0)) ↔ (a, b) = (3 / 2, 4) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l1539_153986


namespace NUMINAMATH_GPT_price_of_remote_controlled_airplane_l1539_153916

theorem price_of_remote_controlled_airplane (x : ℝ) (h : 300 = 0.8 * x) : x = 375 :=
by
  sorry

end NUMINAMATH_GPT_price_of_remote_controlled_airplane_l1539_153916


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_l1539_153920

noncomputable def f (x : ℝ) : ℝ := 1 - 3 * Real.sin (x + Real.pi / 4) ^ 2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + Real.pi) = f x :=
by
  intros
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_l1539_153920


namespace NUMINAMATH_GPT_forester_total_trees_planted_l1539_153913

theorem forester_total_trees_planted (initial_trees monday_trees tuesday_trees wednesday_trees total_trees : ℕ)
    (h1 : initial_trees = 30)
    (h2 : total_trees = 300)
    (h3 : monday_trees = 2 * initial_trees)
    (h4 : tuesday_trees = monday_trees / 3)
    (h5 : wednesday_trees = 2 * tuesday_trees) : 
    (monday_trees + tuesday_trees + wednesday_trees = 120) := by
  sorry

end NUMINAMATH_GPT_forester_total_trees_planted_l1539_153913


namespace NUMINAMATH_GPT_magnitude_of_complex_l1539_153973

open Complex

theorem magnitude_of_complex :
  abs (Complex.mk (2/3) (-4/5)) = Real.sqrt 244 / 15 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_magnitude_of_complex_l1539_153973


namespace NUMINAMATH_GPT_FO_gt_DI_l1539_153941

-- Definitions and conditions
variables (F I D O : Type) [MetricSpace F] [MetricSpace I] [MetricSpace D] [MetricSpace O]
variables (FI DO DI FO : ℝ) (angle_FIO angle_DIO : ℝ)
variable (convex_FIDO : ConvexQuadrilateral F I D O)

-- Conditions
axiom FI_DO_equal : FI = DO
axiom FI_DO_gt_DI : FI > DI
axiom angles_equal : angle_FIO = angle_DIO

-- Goal
theorem FO_gt_DI : FO > DI :=
sorry

end NUMINAMATH_GPT_FO_gt_DI_l1539_153941


namespace NUMINAMATH_GPT_karen_cookies_grandparents_l1539_153933

theorem karen_cookies_grandparents :
  ∀ (total_cookies cookies_kept class_size cookies_per_person : ℕ)
  (cookies_given_class cookies_left cookies_to_grandparents : ℕ),
  total_cookies = 50 →
  cookies_kept = 10 →
  class_size = 16 →
  cookies_per_person = 2 →
  cookies_given_class = class_size * cookies_per_person →
  cookies_left = total_cookies - cookies_kept - cookies_given_class →
  cookies_to_grandparents = cookies_left →
  cookies_to_grandparents = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_karen_cookies_grandparents_l1539_153933


namespace NUMINAMATH_GPT_skittles_distribution_l1539_153923

-- Given problem conditions
variable (Brandon_initial : ℕ := 96) (Bonnie_initial : ℕ := 4) 
variable (Brandon_loss : ℕ := 9)
variable (combined_skittles : ℕ := (Brandon_initial - Brandon_loss) + Bonnie_initial)
variable (individual_share : ℕ := combined_skittles / 4)
variable (remainder : ℕ := combined_skittles % 4)
variable (Chloe_share : ℕ := individual_share)
variable (Dylan_share_initial : ℕ := individual_share)
variable (Chloe_to_Dylan : ℕ := Chloe_share / 2)
variable (Dylan_new_share : ℕ := Dylan_share_initial + Chloe_to_Dylan)
variable (Dylan_to_Bonnie : ℕ := Dylan_new_share / 3)
variable (final_Bonnie : ℕ := individual_share + Dylan_to_Bonnie)
variable (final_Chloe : ℕ := Chloe_share - Chloe_to_Dylan)
variable (final_Dylan : ℕ := Dylan_new_share - Dylan_to_Bonnie)

-- The theorem to be proved
theorem skittles_distribution : 
  individual_share = 22 ∧ final_Bonnie = 33 ∧ final_Chloe = 11 ∧ final_Dylan = 22 :=
by
  -- The proof would go here, but it’s not required for this task.
  sorry

end NUMINAMATH_GPT_skittles_distribution_l1539_153923


namespace NUMINAMATH_GPT_circle_represents_valid_a_l1539_153937

theorem circle_represents_valid_a (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 2 * a * x - 4 * y + 5 * a = 0) → (a > 4 ∨ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_circle_represents_valid_a_l1539_153937


namespace NUMINAMATH_GPT_stock_price_end_second_year_l1539_153992

theorem stock_price_end_second_year
  (P₀ : ℝ) (r₁ r₂ : ℝ) 
  (h₀ : P₀ = 150)
  (h₁ : r₁ = 0.80)
  (h₂ : r₂ = 0.30) :
  let P₁ := P₀ + r₁ * P₀
  let P₂ := P₁ - r₂ * P₁
  P₂ = 189 :=
by
  sorry

end NUMINAMATH_GPT_stock_price_end_second_year_l1539_153992


namespace NUMINAMATH_GPT_solve_for_y_l1539_153967

theorem solve_for_y (y : ℝ) (h : (y * (y^5)^(1/4))^(1/3) = 4) : y = 2^(8/3) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_y_l1539_153967


namespace NUMINAMATH_GPT_min_value_l1539_153902

theorem min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (1 / x) + (4 / y) ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_l1539_153902


namespace NUMINAMATH_GPT_train_stop_time_l1539_153919

theorem train_stop_time
  (D : ℝ)
  (h1 : D > 0)
  (T_no_stop : ℝ := D / 300)
  (T_with_stop : ℝ := D / 200)
  (T_stop : ℝ := T_with_stop - T_no_stop):
  T_stop = 6 / 60 := by
    sorry

end NUMINAMATH_GPT_train_stop_time_l1539_153919


namespace NUMINAMATH_GPT_probability_of_blue_face_l1539_153911

theorem probability_of_blue_face (total_faces blue_faces : ℕ) (h_total : total_faces = 8) (h_blue : blue_faces = 5) : 
  blue_faces / total_faces = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_blue_face_l1539_153911


namespace NUMINAMATH_GPT_max_total_cut_length_l1539_153960

theorem max_total_cut_length :
  let side_length := 30
  let num_pieces := 225
  let area_per_piece := (side_length ^ 2) / num_pieces
  let outer_perimeter := 4 * side_length
  let max_perimeter_per_piece := 10
  (num_pieces * max_perimeter_per_piece - outer_perimeter) / 2 = 1065 :=
by
  sorry

end NUMINAMATH_GPT_max_total_cut_length_l1539_153960


namespace NUMINAMATH_GPT_find_b_l1539_153971

theorem find_b (a b c : ℤ) (h1 : a + b + c = 120) (h2 : a + 4 = b - 12) (h3 : a + 4 = 3 * c) : b = 60 :=
sorry

end NUMINAMATH_GPT_find_b_l1539_153971


namespace NUMINAMATH_GPT_tower_height_proof_l1539_153932

-- Definitions corresponding to given conditions
def elev_angle_A : ℝ := 45
def distance_AD : ℝ := 129
def elev_angle_D : ℝ := 60
def tower_height : ℝ := 305

-- Proving the height of Liaoning Broadcasting and Television Tower
theorem tower_height_proof (h : ℝ) (AC CD : ℝ) (h_eq_AC : h = AC) (h_eq_CD_sqrt3 : h = CD * (Real.sqrt 3)) (AC_CD_sum : AC + CD = 129) :
  h = 305 :=
by
  sorry

end NUMINAMATH_GPT_tower_height_proof_l1539_153932


namespace NUMINAMATH_GPT_find_number_l1539_153910

theorem find_number (some_number : ℤ) : 45 - (28 - (some_number - (15 - 19))) = 58 ↔ some_number = 37 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l1539_153910


namespace NUMINAMATH_GPT_moles_of_naoh_needed_l1539_153945

-- Define the chemical reaction
def balanced_eqn (nh4no3 naoh nano3 nh4oh : ℕ) : Prop :=
  nh4no3 = naoh ∧ nh4no3 = nano3

-- Theorem stating the moles of NaOH required to form 2 moles of NaNO3 from 2 moles of NH4NO3
theorem moles_of_naoh_needed (nh4no3 naoh nano3 nh4oh : ℕ) (h_balanced_eqn : balanced_eqn nh4no3 naoh nano3 nh4oh) 
  (h_nano3: nano3 = 2) (h_nh4no3: nh4no3 = 2) : naoh = 2 :=
by
  unfold balanced_eqn at h_balanced_eqn
  sorry

end NUMINAMATH_GPT_moles_of_naoh_needed_l1539_153945


namespace NUMINAMATH_GPT_tenth_term_is_513_l1539_153903

def nth_term (n : ℕ) : ℕ :=
  2^(n-1) + 1

theorem tenth_term_is_513 : nth_term 10 = 513 := 
by 
  sorry

end NUMINAMATH_GPT_tenth_term_is_513_l1539_153903


namespace NUMINAMATH_GPT_Tom_green_marbles_l1539_153968

-- Define the given variables
def Sara_green_marbles : Nat := 3
def Total_green_marbles : Nat := 7

-- The statement to be proven
theorem Tom_green_marbles : (Total_green_marbles - Sara_green_marbles) = 4 := by
  sorry

end NUMINAMATH_GPT_Tom_green_marbles_l1539_153968


namespace NUMINAMATH_GPT_determinant_of_A_l1539_153980

def A : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, -2], ![8, 5, -4], ![3, 3, 7]]  -- Defining matrix A

def A' : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, -2], ![5, 4, -2], ![0, 2, 9]]  -- Defining matrix A' after row operations

theorem determinant_of_A' : Matrix.det A' = 55 := by -- Proving that the determinant of A' is 55
  sorry

end NUMINAMATH_GPT_determinant_of_A_l1539_153980


namespace NUMINAMATH_GPT_compute_fraction_l1539_153959

theorem compute_fraction (a b c : ℝ) (h1 : a + b = 20) (h2 : b + c = 22) (h3 : c + a = 2022) :
  (a - b) / (c - a) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_l1539_153959


namespace NUMINAMATH_GPT_fraction_computation_l1539_153953

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_computation_l1539_153953


namespace NUMINAMATH_GPT_probability_of_non_touching_square_is_correct_l1539_153949

def square_not_touching_perimeter_or_center_probability : ℚ :=
  let total_squares := 100
  let perimeter_squares := 24
  let center_line_squares := 16
  let touching_squares := perimeter_squares + center_line_squares
  let non_touching_squares := total_squares - touching_squares
  non_touching_squares / total_squares

theorem probability_of_non_touching_square_is_correct :
  square_not_touching_perimeter_or_center_probability = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_non_touching_square_is_correct_l1539_153949


namespace NUMINAMATH_GPT_mailman_should_give_junk_mail_l1539_153991

-- Definitions from the conditions
def houses_in_block := 20
def junk_mail_per_house := 32

-- The mathematical equivalent proof problem statement in Lean 4
theorem mailman_should_give_junk_mail : 
  junk_mail_per_house * houses_in_block = 640 :=
  by sorry

end NUMINAMATH_GPT_mailman_should_give_junk_mail_l1539_153991


namespace NUMINAMATH_GPT_remaining_area_l1539_153962

-- Definitions based on conditions
def large_rectangle_length (x : ℝ) : ℝ := 2 * x + 8
def large_rectangle_width (x : ℝ) : ℝ := x + 6
def hole_length (x : ℝ) : ℝ := 3 * x - 4
def hole_width (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem remaining_area (x : ℝ) : (large_rectangle_length x) * (large_rectangle_width x) - (hole_length x) * (hole_width x) = -x^2 + 21 * x + 52 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_remaining_area_l1539_153962


namespace NUMINAMATH_GPT_bisecting_line_of_circle_l1539_153958

theorem bisecting_line_of_circle : 
  (∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y + 1 = 0 → x - y + 1 = 0) := 
sorry

end NUMINAMATH_GPT_bisecting_line_of_circle_l1539_153958


namespace NUMINAMATH_GPT_find_number_l1539_153940

theorem find_number (x : ℝ) : 
  (3 * x / 5 - 220) * 4 + 40 = 360 → x = 500 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l1539_153940


namespace NUMINAMATH_GPT_room_length_calculation_l1539_153982

-- Definitions of the problem conditions
def room_volume : ℝ := 10000
def room_width : ℝ := 10
def room_height : ℝ := 10

-- Statement to prove
theorem room_length_calculation : ∃ L : ℝ, L = room_volume / (room_width * room_height) ∧ L = 100 :=
by
  sorry

end NUMINAMATH_GPT_room_length_calculation_l1539_153982


namespace NUMINAMATH_GPT_rational_number_addition_l1539_153974

theorem rational_number_addition :
  (-206 : ℚ) + (401 + 3 / 4) + (-(204 + 2 / 3)) + (-(1 + 1 / 2)) = -10 - 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_rational_number_addition_l1539_153974


namespace NUMINAMATH_GPT_inequalities_hold_l1539_153935

theorem inequalities_hold (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  a^2 + b^2 ≥ 2 ∧ (1 / a + 1 / b) ≥ 2 := by
  sorry

end NUMINAMATH_GPT_inequalities_hold_l1539_153935


namespace NUMINAMATH_GPT_expected_value_of_fair_dodecahedral_die_l1539_153963

theorem expected_value_of_fair_dodecahedral_die : 
  (1/12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end NUMINAMATH_GPT_expected_value_of_fair_dodecahedral_die_l1539_153963


namespace NUMINAMATH_GPT_f_f_of_2020_l1539_153944

def f (x : ℕ) : ℕ :=
  if x ≤ 1 then 1
  else if 1 < x ∧ x ≤ 1837 then 2
  else if 1837 < x ∧ x < 2019 then 3
  else 2018

theorem f_f_of_2020 : f (f 2020) = 3 := by
  sorry

end NUMINAMATH_GPT_f_f_of_2020_l1539_153944


namespace NUMINAMATH_GPT_total_painting_area_correct_l1539_153975

def barn_width : ℝ := 12
def barn_length : ℝ := 15
def barn_height : ℝ := 6

def area_to_be_painted (width length height : ℝ) : ℝ := 
  2 * (width * height + length * height) + width * length

theorem total_painting_area_correct : area_to_be_painted barn_width barn_length barn_height = 828 := 
  by sorry

end NUMINAMATH_GPT_total_painting_area_correct_l1539_153975


namespace NUMINAMATH_GPT_part1_part2_l1539_153947

def setA : Set ℝ := {x | (x - 2) / (x + 1) < 0}
def setB (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 - k}

theorem part1 : (setB (-1)).union setA = {x : ℝ | -1 < x ∧ x < 3 } := by
  sorry

theorem part2 (k : ℝ) : (setA ∩ setB k = setB k ↔ 0 ≤ k) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1539_153947


namespace NUMINAMATH_GPT_hexagon_perimeter_l1539_153976

-- Define the length of a side of the hexagon
def side_length : ℕ := 7

-- Define the number of sides of the hexagon
def num_sides : ℕ := 6

-- Define the perimeter of the hexagon
def perimeter (num_sides side_length : ℕ) : ℕ :=
  num_sides * side_length

-- Theorem stating the perimeter of the hexagon with given side length is 42 inches
theorem hexagon_perimeter : perimeter num_sides side_length = 42 := by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_l1539_153976


namespace NUMINAMATH_GPT_middle_integer_is_five_l1539_153925

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def are_consecutive_odd_integers (a b c : ℤ) : Prop :=
  a < b ∧ b < c ∧ (∃ n : ℤ, a = b - 2 ∧ c = b + 2 ∧ is_odd a ∧ is_odd b ∧ is_odd c)

def sum_is_one_eighth_product (a b c : ℤ) : Prop :=
  a + b + c = (a * b * c) / 8

theorem middle_integer_is_five :
  ∃ (a b c : ℤ), are_consecutive_odd_integers a b c ∧ sum_is_one_eighth_product a b c ∧ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_middle_integer_is_five_l1539_153925


namespace NUMINAMATH_GPT_smallest_sector_angle_24_l1539_153921

theorem smallest_sector_angle_24
  (a : ℕ) (d : ℕ)
  (h1 : ∀ i, i < 8 → ((a + i * d) : ℤ) > 0)
  (h2 : (2 * a + 7 * d = 90)) : a = 24 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sector_angle_24_l1539_153921


namespace NUMINAMATH_GPT_solve_discriminant_l1539_153912

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem solve_discriminant : 
  discriminant 2 (2 + 1/2) (1/2) = 2.25 :=
by
  -- The proof can be filled in here
  -- Assuming a = 2, b = 2.5, c = 1/2
  -- discriminant 2 2.5 0.5 will be computed
  sorry

end NUMINAMATH_GPT_solve_discriminant_l1539_153912


namespace NUMINAMATH_GPT_total_cakes_served_l1539_153965

theorem total_cakes_served (l : ℝ) (p : ℝ) (s : ℝ) (total_cakes_served_today : ℝ) :
  l = 48.5 → p = 0.6225 → s = 95 → total_cakes_served_today = 108 :=
by
  intros hl hp hs
  sorry

end NUMINAMATH_GPT_total_cakes_served_l1539_153965


namespace NUMINAMATH_GPT_second_number_is_46_l1539_153985

theorem second_number_is_46 (sum_is_330 : ∃ (a b c d : ℕ), a + b + c + d = 330)
    (first_is_twice_second : ∀ (b : ℕ), ∃ (a : ℕ), a = 2 * b)
    (third_is_one_third_of_first : ∀ (a : ℕ), ∃ (c : ℕ), c = a / 3)
    (fourth_is_half_difference : ∀ (a b : ℕ), ∃ (d : ℕ), d = (a - b) / 2) :
  ∃ (b : ℕ), b = 46 :=
by
  -- Proof goes here, inserted for illustrating purposes only
  sorry

end NUMINAMATH_GPT_second_number_is_46_l1539_153985


namespace NUMINAMATH_GPT_min_height_of_box_with_surface_area_condition_l1539_153983

theorem min_height_of_box_with_surface_area_condition {x : ℕ}  
(h : 2*x^2 + 4*x*(x + 6) ≥ 150) (hx: x ≥ 5) : (x + 6) = 11 := by
  sorry

end NUMINAMATH_GPT_min_height_of_box_with_surface_area_condition_l1539_153983


namespace NUMINAMATH_GPT_common_root_equation_l1539_153981

theorem common_root_equation (a b r : ℝ) (h₁ : a ≠ b)
  (h₂ : r^2 + 2019 * a * r + b = 0)
  (h₃ : r^2 + 2019 * b * r + a = 0) :
  r = 1 / 2019 :=
by
  sorry

end NUMINAMATH_GPT_common_root_equation_l1539_153981


namespace NUMINAMATH_GPT_scientific_notation_of_18M_l1539_153943

theorem scientific_notation_of_18M : 18000000 = 1.8 * 10^7 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_18M_l1539_153943


namespace NUMINAMATH_GPT_train_journey_duration_l1539_153926

variable (z x : ℝ)
variable (h1 : 1.7 = 1 + 42 / 60)
variable (h2 : (0.9 * z / (1.2 * x) + 0.1 * z / (1.25 * x)) = z / x - 1.7)

theorem train_journey_duration (z x : ℝ)
    (h1 : 1.7 = 1 + 42 / 60)
    (h2 : (0.9 * z / (1.2 * x) + 0.1 * z / (1.25 * x)) = z / x - 1.7):
    z / x = 10 := 
by
  sorry

end NUMINAMATH_GPT_train_journey_duration_l1539_153926


namespace NUMINAMATH_GPT_simplify_expression_l1539_153956

theorem simplify_expression :
  (2 * 10^9) - (6 * 10^7) / (2 * 10^2) = 1999700000 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1539_153956


namespace NUMINAMATH_GPT_find_S2012_l1539_153922

section Problem

variable {a : ℕ → ℝ} -- Defining the sequence

-- Conditions
def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ q, ∀ n, a (n + 1) = a n * q

def sum_S (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (Finset.range n).sum a

axiom a1 : a 1 = 2011
axiom recurrence_relation (n : ℕ) : a n + 2*a (n + 1) + a (n + 2) = 0

-- Proof statement
theorem find_S2012 (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ):
  geometric_sequence a →
  (∀ n, S n = sum_S a n) →
  S 2012 = 0 :=
by
  sorry

end Problem

end NUMINAMATH_GPT_find_S2012_l1539_153922


namespace NUMINAMATH_GPT_smallest_a_l1539_153904

def f (x : ℕ) : ℕ :=
  if x % 21 = 0 then x / 21
  else if x % 7 = 0 then 3 * x
  else if x % 3 = 0 then 7 * x
  else x + 3

def f_iterate (a : ℕ) (x : ℕ) : ℕ :=
  Nat.iterate f a x

theorem smallest_a (a : ℕ) : a > 1 ∧ f_iterate a 2 = f 2 ↔ a = 7 := 
sorry

end NUMINAMATH_GPT_smallest_a_l1539_153904


namespace NUMINAMATH_GPT_sum_g_squared_l1539_153970

noncomputable def g (n : ℕ) : ℝ :=
  ∑' m, if m ≥ 3 then 1 / (m : ℝ)^n else 0

theorem sum_g_squared :
  (∑' n, if n ≥ 3 then (g n)^2 else 0) = 1 / 288 :=
by
  sorry

end NUMINAMATH_GPT_sum_g_squared_l1539_153970


namespace NUMINAMATH_GPT_perp_line_through_point_l1539_153936

variable (x y c : ℝ)

def line_perpendicular (x y : ℝ) : Prop :=
  x - 2*y + 1 = 0

def perpendicular_line (x y c : ℝ) : Prop :=
  2*x + y + c = 0

theorem perp_line_through_point :
  (line_perpendicular x y) ∧ (perpendicular_line (-2) 3 1) :=
by
  -- The first part asserts that the given line equation holds
  have h1 : line_perpendicular x y := sorry
  -- The second part asserts that our calculated line passes through the point (-2, 3) and is perpendicular
  have h2 : perpendicular_line (-2) 3 1 := sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_perp_line_through_point_l1539_153936


namespace NUMINAMATH_GPT_polynomial_identity_l1539_153907

theorem polynomial_identity 
  (P : Polynomial ℤ)
  (a b : ℤ) 
  (h_distinct : a ≠ b)
  (h_eq : P.eval a * P.eval b = -(a - b) ^ 2) : 
  P.eval a + P.eval b = 0 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_identity_l1539_153907


namespace NUMINAMATH_GPT_solve_system_l1539_153939

theorem solve_system (x y : ℚ) 
  (h₁ : 7 * x - 14 * y = 3) 
  (h₂ : 3 * y - x = 5) : 
  x = 79 / 7 ∧ y = 38 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_solve_system_l1539_153939


namespace NUMINAMATH_GPT_olaf_total_toy_cars_l1539_153987

def olaf_initial_collection : ℕ := 150
def uncle_toy_cars : ℕ := 5
def auntie_toy_cars : ℕ := uncle_toy_cars + 1 -- 6 toy cars
def grandpa_toy_cars : ℕ := 2 * uncle_toy_cars -- 10 toy cars
def dad_toy_cars : ℕ := 10
def mum_toy_cars : ℕ := dad_toy_cars + 5 -- 15 toy cars
def toy_cars_received : ℕ := grandpa_toy_cars + uncle_toy_cars + dad_toy_cars + mum_toy_cars + auntie_toy_cars -- total toy cars received
def olaf_total_collection : ℕ := olaf_initial_collection + toy_cars_received

theorem olaf_total_toy_cars : olaf_total_collection = 196 := by
  sorry

end NUMINAMATH_GPT_olaf_total_toy_cars_l1539_153987


namespace NUMINAMATH_GPT_total_boys_eq_350_l1539_153984

variable (Total : ℕ)
variable (SchoolA : ℕ)
variable (NotScience : ℕ)

axiom h1 : SchoolA = 20 * Total / 100
axiom h2 : NotScience = 70 * SchoolA / 100
axiom h3 : NotScience = 49

theorem total_boys_eq_350 : Total = 350 :=
by
  sorry

end NUMINAMATH_GPT_total_boys_eq_350_l1539_153984


namespace NUMINAMATH_GPT_pyramid_base_side_length_l1539_153988

theorem pyramid_base_side_length
  (lateral_face_area : Real)
  (slant_height : Real)
  (s : Real)
  (h_lateral_face_area : lateral_face_area = 200)
  (h_slant_height : slant_height = 40)
  (h_area_formula : lateral_face_area = 0.5 * s * slant_height) :
  s = 10 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_base_side_length_l1539_153988


namespace NUMINAMATH_GPT_triangle_right_angle_l1539_153942

theorem triangle_right_angle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : A + B = 90)
  (h2 : (a + b) * (a - b) = c ^ 2)
  (h3 : A / B = 1 / 2) :
  C = 90 :=
sorry

end NUMINAMATH_GPT_triangle_right_angle_l1539_153942


namespace NUMINAMATH_GPT_mryak_bryak_problem_l1539_153961

variable (m b : ℚ)

theorem mryak_bryak_problem
  (h1 : 3 * m = 5 * b + 10)
  (h2 : 6 * m = 8 * b + 31) :
  7 * m - 9 * b = 38 := sorry

end NUMINAMATH_GPT_mryak_bryak_problem_l1539_153961


namespace NUMINAMATH_GPT_jerusha_earnings_l1539_153928

variable (L : ℝ) 

theorem jerusha_earnings (h1 : L + 4 * L = 85) : 4 * L = 68 := 
by
  sorry

end NUMINAMATH_GPT_jerusha_earnings_l1539_153928


namespace NUMINAMATH_GPT_parallel_vectors_tan_l1539_153954

/-- Given vector a and vector b, and given the condition that a is parallel to b,
prove that the value of tan α is 1/4. -/
theorem parallel_vectors_tan (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (Real.sin α, Real.cos α - 2 * Real.sin α))
  (hb : b = (1, 2))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) : 
  Real.tan α = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_parallel_vectors_tan_l1539_153954


namespace NUMINAMATH_GPT_triangle_BC_length_l1539_153924

theorem triangle_BC_length
  (y_eq_2x2 : ∀ (x : ℝ), ∃ (y : ℝ), y = 2 * x ^ 2)
  (area_ABC : ∃ (A B C : ℝ × ℝ), 
    A = (0, 0) ∧ (∃ (a : ℝ), B = (a, 2 * a ^ 2) ∧ C = (-a, 2 * a ^ 2) ∧ 2 * a ^ 3 = 128))
  : ∃ (a : ℝ), 2 * a = 8 := 
sorry

end NUMINAMATH_GPT_triangle_BC_length_l1539_153924


namespace NUMINAMATH_GPT_second_divisor_13_l1539_153946

theorem second_divisor_13 (N D : ℤ) (k m : ℤ) 
  (h1 : N = 39 * k + 17) 
  (h2 : N = D * m + 4) : 
  D = 13 := 
sorry

end NUMINAMATH_GPT_second_divisor_13_l1539_153946


namespace NUMINAMATH_GPT_expression_square_minus_three_times_l1539_153938

-- Defining the statement
theorem expression_square_minus_three_times (a b : ℝ) : a^2 - 3 * b = a^2 - 3 * b := 
by
  sorry

end NUMINAMATH_GPT_expression_square_minus_three_times_l1539_153938


namespace NUMINAMATH_GPT_books_from_second_shop_l1539_153952

-- Define the conditions
def num_books_first_shop : ℕ := 65
def cost_first_shop : ℕ := 1280
def cost_second_shop : ℕ := 880
def total_cost : ℤ := cost_first_shop + cost_second_shop
def average_price_per_book : ℤ := 18

-- Define the statement to be proved
theorem books_from_second_shop (x : ℕ) :
  (num_books_first_shop + x) * average_price_per_book = total_cost →
  x = 55 :=
by
  sorry

end NUMINAMATH_GPT_books_from_second_shop_l1539_153952


namespace NUMINAMATH_GPT_intersecting_chords_l1539_153995

theorem intersecting_chords (n : ℕ) (h1 : 0 < n) :
  ∃ intersecting_points : ℕ, intersecting_points ≥ n :=
  sorry

end NUMINAMATH_GPT_intersecting_chords_l1539_153995


namespace NUMINAMATH_GPT_equation_of_line_AC_l1539_153966

-- Define the given points A and B
def A : (ℝ × ℝ) := (1, 1)
def B : (ℝ × ℝ) := (-3, -5)

-- Define the line m as a predicate
def line_m (p : ℝ × ℝ) : Prop := 2 * p.1 + p.2 + 6 = 0

-- Define the condition that line m is the angle bisector of ∠ACB
def is_angle_bisector (A B C : ℝ × ℝ) (m : (ℝ × ℝ) → Prop) : Prop := sorry

-- The symmetric point of B with respect to line m
def symmetric_point (B : ℝ × ℝ) (m : (ℝ × ℝ) → Prop) : (ℝ × ℝ) := sorry

-- Proof statement
theorem equation_of_line_AC :
  ∀ (A B : ℝ × ℝ) (m : (ℝ × ℝ) → Prop),
  A = (1, 1) →
  B = (-3, -5) →
  m = line_m →
  is_angle_bisector A B (symmetric_point B m) m →
  AC = {p : ℝ × ℝ | p.1 = 1} := sorry

end NUMINAMATH_GPT_equation_of_line_AC_l1539_153966


namespace NUMINAMATH_GPT_area_of_set_K_l1539_153972

open Metric

def set_K :=
  {p : ℝ × ℝ | (abs p.1 + abs (3 * p.2) - 6) * (abs (3 * p.1) + abs p.2 - 6) ≤ 0}

def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry -- Define the area function for a general set s

theorem area_of_set_K : area set_K = 24 :=
  sorry

end NUMINAMATH_GPT_area_of_set_K_l1539_153972


namespace NUMINAMATH_GPT_value_of_g_of_h_at_2_l1539_153930

def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := -5 * x^3 + 4

theorem value_of_g_of_h_at_2 : g (h 2) = 3890 := by
  sorry

end NUMINAMATH_GPT_value_of_g_of_h_at_2_l1539_153930


namespace NUMINAMATH_GPT_harry_morning_ratio_l1539_153990

-- Define the total morning routine time
def total_morning_routine_time : ℕ := 45

-- Define the time taken to buy coffee and a bagel
def time_buying_coffee_and_bagel : ℕ := 15

-- Calculate the time spent reading the paper and eating
def time_reading_and_eating : ℕ :=
  total_morning_routine_time - time_buying_coffee_and_bagel

-- Define the ratio of the time spent reading and eating to buying coffee and a bagel
def ratio_reading_eating_to_buying_coffee_bagel : ℚ :=
  (time_reading_and_eating : ℚ) / (time_buying_coffee_and_bagel : ℚ)

-- State the theorem
theorem harry_morning_ratio : ratio_reading_eating_to_buying_coffee_bagel = 2 := 
by
  sorry

end NUMINAMATH_GPT_harry_morning_ratio_l1539_153990


namespace NUMINAMATH_GPT_fill_in_the_blanks_correctly_l1539_153908

def remote_areas_need : String := "what the remote areas need"
def children : String := "children"
def education : String := "education"
def good_textbooks : String := "good textbooks"

-- Defining the grammatical agreement condition
def subject_verb_agreement (s : String) (v : String) : Prop :=
  (s = remote_areas_need ∧ v = "is") ∨ (s = children ∧ v = "are")

-- The main theorem statement
theorem fill_in_the_blanks_correctly : 
  subject_verb_agreement remote_areas_need "is" ∧ subject_verb_agreement children "are" :=
sorry

end NUMINAMATH_GPT_fill_in_the_blanks_correctly_l1539_153908


namespace NUMINAMATH_GPT_meal_cost_l1539_153993

theorem meal_cost (total_people total_bill : ℕ) (h1 : total_people = 2 + 5) (h2 : total_bill = 21) :
  total_bill / total_people = 3 := by
  sorry

end NUMINAMATH_GPT_meal_cost_l1539_153993


namespace NUMINAMATH_GPT_total_stamps_l1539_153999

-- Definitions based on the conditions
def AJ := 370
def KJ := AJ / 2
def CJ := 2 * KJ + 5

-- Proof Statement
theorem total_stamps : AJ + KJ + CJ = 930 := by
  sorry

end NUMINAMATH_GPT_total_stamps_l1539_153999


namespace NUMINAMATH_GPT_sarees_with_6_shirts_l1539_153929

-- Define the prices of sarees, shirts and the equation parameters
variables (S T : ℕ) (X : ℕ)

-- Define the conditions as hypotheses
def condition1 := 2 * S + 4 * T = 1600
def condition2 := 12 * T = 2400
def condition3 := X * S + 6 * T = 1600

-- Define the theorem to prove X = 1 under these conditions
theorem sarees_with_6_shirts
  (h1 : condition1 S T)
  (h2 : condition2 T)
  (h3 : condition3 S T X) : 
  X = 1 :=
sorry

end NUMINAMATH_GPT_sarees_with_6_shirts_l1539_153929


namespace NUMINAMATH_GPT_custom_op_value_l1539_153901

variable {a b : ℤ}
def custom_op (a b : ℤ) := 1/a + 1/b

axiom h1 : a + b = 15
axiom h2 : a * b = 56

theorem custom_op_value : custom_op a b = 15/56 :=
by
  sorry

end NUMINAMATH_GPT_custom_op_value_l1539_153901


namespace NUMINAMATH_GPT_problem_conditions_l1539_153951

open Real

variable {m n : ℝ}

theorem problem_conditions (h1 : 0 < m) (h2 : 0 < n) (h3 : m + n = 2 * m * n) :
  (min (m + n) = 2) ∧ (min (sqrt (m * n)) = 1) ∧
  (min ((n^2) / m + (m^2) / n) = 2) ∧ 
  (max ((sqrt m + sqrt n) / sqrt (m * n)) = 2) :=
by sorry

end NUMINAMATH_GPT_problem_conditions_l1539_153951


namespace NUMINAMATH_GPT_smallest_n_candy_price_l1539_153917

theorem smallest_n_candy_price :
  ∃ n : ℕ, 25 * n = Nat.lcm (Nat.lcm 20 18) 24 ∧ ∀ k : ℕ, k > 0 ∧ 25 * k = Nat.lcm (Nat.lcm 20 18) 24 → n ≤ k :=
sorry

end NUMINAMATH_GPT_smallest_n_candy_price_l1539_153917


namespace NUMINAMATH_GPT_sum_first_3n_terms_is_36_l1539_153955

-- Definitions and conditions
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2
def sum_first_2n_terms (a d : ℤ) (n : ℕ) : ℤ := 2 * n * (2 * a + (2 * n - 1) * d) / 2
def sum_first_3n_terms (a d : ℤ) (n : ℕ) : ℤ := 3 * n * (2 * a + (3 * n - 1) * d) / 2

axiom h1 : ∀ (a d : ℤ) (n : ℕ), sum_first_n_terms a d n = 48
axiom h2 : ∀ (a d : ℤ) (n : ℕ), sum_first_2n_terms a d n = 60

theorem sum_first_3n_terms_is_36 (a d : ℤ) (n : ℕ) : sum_first_3n_terms a d n = 36 := by
  sorry

end NUMINAMATH_GPT_sum_first_3n_terms_is_36_l1539_153955


namespace NUMINAMATH_GPT_question_l1539_153979
-- Importing necessary libraries

-- Stating the problem
theorem question (x : ℤ) (h : (x + 12) / 8 = 9) : 35 - (x / 2) = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_question_l1539_153979


namespace NUMINAMATH_GPT_gcd_problem_l1539_153918

-- Define the two numbers
def a : ℕ := 1000000000
def b : ℕ := 1000000005

-- Define the problem to prove the GCD
theorem gcd_problem : Nat.gcd a b = 5 :=
by 
  sorry

end NUMINAMATH_GPT_gcd_problem_l1539_153918


namespace NUMINAMATH_GPT_rectangle_area_l1539_153909

theorem rectangle_area (x : ℝ) (l : ℝ) (h1 : 3 * l = x^2 / 10) : 
  3 * l^2 = 3 * x^2 / 10 :=
by sorry

end NUMINAMATH_GPT_rectangle_area_l1539_153909


namespace NUMINAMATH_GPT_total_amount_paid_is_correct_l1539_153996

-- Definitions for the conditions
def original_price : ℝ := 150
def sale_discount : ℝ := 0.30
def coupon_discount : ℝ := 10
def sales_tax : ℝ := 0.10

-- Calculation
def final_amount : ℝ :=
  let discounted_price := original_price * (1 - sale_discount)
  let price_after_coupon := discounted_price - coupon_discount
  let final_price_after_tax := price_after_coupon * (1 + sales_tax)
  final_price_after_tax

-- Statement to prove
theorem total_amount_paid_is_correct : final_amount = 104.50 := by
  sorry

end NUMINAMATH_GPT_total_amount_paid_is_correct_l1539_153996


namespace NUMINAMATH_GPT_find_a6_l1539_153927

variable (S : ℕ → ℝ) (a : ℕ → ℝ)
variable (h1 : ∀ n ≥ 2, S n = 2 * a n)
variable (h2 : S 5 = 8)

theorem find_a6 : a 6 = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_a6_l1539_153927


namespace NUMINAMATH_GPT_sum_divisible_by_12_l1539_153905

theorem sum_divisible_by_12 :
  ((2150 + 2151 + 2152 + 2153 + 2154 + 2155) % 12) = 3 := by
  sorry

end NUMINAMATH_GPT_sum_divisible_by_12_l1539_153905


namespace NUMINAMATH_GPT_units_digit_2_1501_5_1602_11_1703_l1539_153957

theorem units_digit_2_1501_5_1602_11_1703 : 
  (2 ^ 1501 * 5 ^ 1602 * 11 ^ 1703) % 10 = 0 :=
  sorry

end NUMINAMATH_GPT_units_digit_2_1501_5_1602_11_1703_l1539_153957


namespace NUMINAMATH_GPT_students_more_than_rabbits_by_64_l1539_153948

-- Define the conditions as constants
def number_of_classrooms : ℕ := 4
def students_per_classroom : ℕ := 18
def rabbits_per_classroom : ℕ := 2

-- Define the quantities that need calculations
def total_students : ℕ := number_of_classrooms * students_per_classroom
def total_rabbits : ℕ := number_of_classrooms * rabbits_per_classroom
def difference_students_rabbits : ℕ := total_students - total_rabbits

-- State the theorem to be proven
theorem students_more_than_rabbits_by_64 :
  difference_students_rabbits = 64 := by
  sorry

end NUMINAMATH_GPT_students_more_than_rabbits_by_64_l1539_153948


namespace NUMINAMATH_GPT_find_a_value_l1539_153994

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 := 
by 
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_find_a_value_l1539_153994


namespace NUMINAMATH_GPT_polar_equation_is_circle_l1539_153934

-- Define the polar coordinates equation condition
def polar_equation (r θ : ℝ) : Prop := r = 5

-- Define what it means for a set of points to form a circle centered at the origin with a radius of 5
def is_circle_radius_5 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- State the theorem we want to prove
theorem polar_equation_is_circle (r θ : ℝ) (x y : ℝ) (h1 : polar_equation r θ)
  (h2 : x = r * Real.cos θ) (h3 : y = r * Real.sin θ) : is_circle_radius_5 x y := 
sorry

end NUMINAMATH_GPT_polar_equation_is_circle_l1539_153934


namespace NUMINAMATH_GPT_find_smaller_number_l1539_153931

theorem find_smaller_number (x y : ℕ) (h₁ : y - x = 1365) (h₂ : y = 6 * x + 15) : x = 270 :=
sorry

end NUMINAMATH_GPT_find_smaller_number_l1539_153931


namespace NUMINAMATH_GPT_value_of_a7_minus_a8_l1539_153900

variable {a : ℕ → ℤ} (d a₁ : ℤ)

-- Definition that this is an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℤ) (a₁ d : ℤ) : Prop :=
  ∀ n, a n = a₁ + (n - 1) * d

-- Given condition
def condition (a : ℕ → ℤ) : Prop :=
  a 2 + a 6 + a 8 + a 10 = 80

-- The goal to prove
theorem value_of_a7_minus_a8 (a : ℕ → ℤ) (h_arith : is_arithmetic_seq a a₁ d)
  (h_cond : condition a) : a 7 - a 8 = 8 :=
sorry

end NUMINAMATH_GPT_value_of_a7_minus_a8_l1539_153900


namespace NUMINAMATH_GPT_number_is_divisible_by_divisor_l1539_153964

-- Defining the number after replacing y with 3
def number : ℕ := 7386038

-- Defining the divisor which we need to prove 
def divisor : ℕ := 7

-- Stating the property that 7386038 is divisible by 7
theorem number_is_divisible_by_divisor : number % divisor = 0 := by
  sorry

end NUMINAMATH_GPT_number_is_divisible_by_divisor_l1539_153964


namespace NUMINAMATH_GPT_last_score_is_87_l1539_153950

-- Definitions based on conditions:
def scores : List ℕ := [73, 78, 82, 84, 87, 95]
def total_sum := 499
def final_median := 83

-- Prove that the last score is 87 under given conditions.
theorem last_score_is_87 (h1 : total_sum = 499)
                        (h2 : ∀ n ∈ scores, (499 - n) % 6 ≠ 0)
                        (h3 : final_median = 83) :
  87 ∈ scores := sorry

end NUMINAMATH_GPT_last_score_is_87_l1539_153950


namespace NUMINAMATH_GPT_packs_with_extra_red_pencils_eq_3_l1539_153978

def total_packs : Nat := 15
def regular_red_per_pack : Nat := 1
def total_red_pencils : Nat := 21
def extra_red_per_pack : Nat := 2

theorem packs_with_extra_red_pencils_eq_3 :
  ∃ (packs_with_extra : Nat), packs_with_extra * extra_red_per_pack + (total_packs - packs_with_extra) * regular_red_per_pack = total_red_pencils ∧ packs_with_extra = 3 :=
by
  sorry

end NUMINAMATH_GPT_packs_with_extra_red_pencils_eq_3_l1539_153978


namespace NUMINAMATH_GPT_intersection_complement_l1539_153977

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 > 4}

-- Define set N
def N : Set ℝ := {x : ℝ | (x - 3) / (x + 1) < 0}

-- Complement of N in U
def complement_N : Set ℝ := {x : ℝ | x <= -1} ∪ {x : ℝ | x >= 3}

-- Final proof to show intersection
theorem intersection_complement :
  M ∩ complement_N = {x : ℝ | x < -2} ∪ {x : ℝ | x >= 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1539_153977


namespace NUMINAMATH_GPT_cube_root_inequality_l1539_153914

theorem cube_root_inequality (a b : ℝ) (h : a > b) : (a ^ (1/3)) > (b ^ (1/3)) :=
sorry

end NUMINAMATH_GPT_cube_root_inequality_l1539_153914


namespace NUMINAMATH_GPT_number_of_insects_l1539_153997

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 30) (h2 : legs_per_insect = 6) :
  total_legs / legs_per_insect = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_insects_l1539_153997


namespace NUMINAMATH_GPT_find_f_2011_l1539_153906

-- Definitions of given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic_of_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- Main theorem to be proven
theorem find_f_2011 (f : ℝ → ℝ) 
  (hf_even: is_even_function f) 
  (hf_periodic: is_periodic_of_period f 4) 
  (hf_at_1: f 1 = 1) : 
  f 2011 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_f_2011_l1539_153906
