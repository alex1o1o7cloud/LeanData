import Mathlib

namespace total_exterior_angles_l1940_194008

-- Define that the sum of the exterior angles of any convex polygon is 360 degrees
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Given four polygons: a triangle, a quadrilateral, a pentagon, and a hexagon
def triangle_exterior_sum := sum_exterior_angles 3
def quadrilateral_exterior_sum := sum_exterior_angles 4
def pentagon_exterior_sum := sum_exterior_angles 5
def hexagon_exterior_sum := sum_exterior_angles 6

-- The total sum of the exterior angles of these four polygons combined
def total_exterior_angle_sum := 
  triangle_exterior_sum + 
  quadrilateral_exterior_sum + 
  pentagon_exterior_sum + 
  hexagon_exterior_sum

-- The final proof statement
theorem total_exterior_angles : total_exterior_angle_sum = 1440 := by
  sorry

end total_exterior_angles_l1940_194008


namespace find_point_P_l1940_194059

theorem find_point_P :
  ∃ (P : ℝ × ℝ), P.1 = 1 ∧ P.2 = 0 ∧ 
  (P.2 = P.1^4 - P.1) ∧
  (∃ m, m = 4 * P.1^3 - 1 ∧ m = 3) :=
by
  sorry

end find_point_P_l1940_194059


namespace value_of_x_l1940_194030

-- Define the conditions
variable (C S x : ℝ)
variable (h1 : 20 * C = x * S)
variable (h2 : (S - C) / C * 100 = 25)

-- Define the statement to be proved
theorem value_of_x : x = 16 :=
by
  sorry

end value_of_x_l1940_194030


namespace proof_inequality_l1940_194012

variable {a b c d : ℝ}

theorem proof_inequality (h1 : a + b + c + d = 6) (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 :=
sorry

end proof_inequality_l1940_194012


namespace cost_of_grapes_and_watermelon_l1940_194011

theorem cost_of_grapes_and_watermelon (p g w f : ℝ)
  (h1 : p + g + w + f = 30)
  (h2 : f = 2 * p)
  (h3 : p - g = w) :
  g + w = 7.5 :=
by
  sorry

end cost_of_grapes_and_watermelon_l1940_194011


namespace total_apples_count_l1940_194090

-- Definitions based on conditions
def red_apples := 16
def green_apples := red_apples + 12
def total_apples := green_apples + red_apples

-- Statement to prove
theorem total_apples_count : total_apples = 44 := 
by
  sorry

end total_apples_count_l1940_194090


namespace g_f_of_3_l1940_194057

def f (x : ℝ) : ℝ := x^3 - 4
def g (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 2

theorem g_f_of_3 : g (f 3) = 1704 := by
  sorry

end g_f_of_3_l1940_194057


namespace mark_ate_in_first_four_days_l1940_194014

-- Definitions based on conditions
def total_fruit : ℕ := 10
def fruit_kept : ℕ := 2
def fruit_brought_on_friday : ℕ := 3

-- Statement to be proved
theorem mark_ate_in_first_four_days : total_fruit - fruit_kept - fruit_brought_on_friday = 5 := 
by sorry

end mark_ate_in_first_four_days_l1940_194014


namespace number_of_sides_of_regular_polygon_l1940_194060

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end number_of_sides_of_regular_polygon_l1940_194060


namespace smallest_angle_half_largest_l1940_194087

open Real

-- Statement of the problem
theorem smallest_angle_half_largest (a b c : ℝ) (α β γ : ℝ)
  (h_sides : a = 4 ∧ b = 5 ∧ c = 6)
  (h_angles : α < β ∧ β < γ)
  (h_cos_alpha : cos α = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_cos_gamma : cos γ = (a^2 + b^2 - c^2) / (2 * a * b)) :
  2 * α = γ := 
sorry

end smallest_angle_half_largest_l1940_194087


namespace complex_purely_imaginary_condition_l1940_194043

theorem complex_purely_imaginary_condition (a : ℝ) :
  (a = 1 → (a - 1) * (a + 2) + (a + 3) * Complex.I.im = (0 : ℝ)) ∧
  ¬(a = 1 ∧ ¬a = -2 → (a - 1) * (a + 2) + (a + 3) * Complex.I.im = (0 : ℝ)) :=
  sorry

end complex_purely_imaginary_condition_l1940_194043


namespace number_of_hens_l1940_194051

theorem number_of_hens
    (H C : ℕ) -- Hens and Cows
    (h1 : H + C = 44) -- Condition 1: The number of heads
    (h2 : 2 * H + 4 * C = 128) -- Condition 2: The number of feet
    : H = 24 :=
by
  sorry

end number_of_hens_l1940_194051


namespace real_roots_of_cubic_equation_l1940_194085

theorem real_roots_of_cubic_equation : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, (x^3 - 2 * x + 1)^2 = 9) ∧ S.card = 2 := 
by
  sorry

end real_roots_of_cubic_equation_l1940_194085


namespace jose_completion_time_l1940_194041

noncomputable def rate_jose : ℚ := 1 / 30
noncomputable def rate_jane : ℚ := 1 / 6

theorem jose_completion_time :
  ∀ (J A : ℚ), 
    (J + A = 1 / 5) ∧ (J = rate_jose) ∧ (A = rate_jane) → 
    (1 / J = 30) :=
by
  intros J A h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end jose_completion_time_l1940_194041


namespace translate_line_upwards_l1940_194039

theorem translate_line_upwards {x y : ℝ} (h : y = -2 * x + 1) :
  y = -2 * x + 3 := by
  sorry

end translate_line_upwards_l1940_194039


namespace lcm_of_28_and_24_is_168_l1940_194022

/-- Racing car A completes the track in 28 seconds.
    Racing car B completes the track in 24 seconds.
    Both cars start at the same time.
    We want to prove that the time after which both cars will be side by side again
    (least common multiple of their lap times) is 168 seconds. -/
theorem lcm_of_28_and_24_is_168 :
  Nat.lcm 28 24 = 168 :=
sorry

end lcm_of_28_and_24_is_168_l1940_194022


namespace solve_for_x_l1940_194081

theorem solve_for_x : ∀ (x : ℝ), (x ≠ 3) → ((x - 3) / (x + 2) + (3 * x - 6) / (x - 3) = 2) → x = 1 / 2 := 
by
  intros x hx h
  sorry

end solve_for_x_l1940_194081


namespace blue_socks_count_l1940_194082

-- Defining the total number of socks
def total_socks : ℕ := 180

-- Defining the number of white socks as two thirds of the total socks
def white_socks : ℕ := (2 * total_socks) / 3

-- Defining the number of blue socks as the difference between total socks and white socks
def blue_socks : ℕ := total_socks - white_socks

-- The theorem to prove
theorem blue_socks_count : blue_socks = 60 := by
  sorry

end blue_socks_count_l1940_194082


namespace range_of_a_l1940_194049

noncomputable def A : Set ℝ := Set.Ico 1 5 -- A = [1, 5)
noncomputable def B (a : ℝ) : Set ℝ := Set.Iio a -- B = (-∞, a)

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 5 ≤ a :=
sorry

end range_of_a_l1940_194049


namespace four_people_seven_chairs_l1940_194028

def num_arrangements (total_chairs : ℕ) (num_reserved : ℕ) (num_people : ℕ) : ℕ :=
  (total_chairs - num_reserved).choose num_people * (num_people.factorial)

theorem four_people_seven_chairs (total_chairs : ℕ) (chairs_occupied : ℕ) (num_people : ℕ): 
    total_chairs = 7 → chairs_occupied = 2 → num_people = 4 →
    num_arrangements total_chairs chairs_occupied num_people = 120 :=
by
  intros
  unfold num_arrangements
  sorry

end four_people_seven_chairs_l1940_194028


namespace corrected_mean_l1940_194077

theorem corrected_mean (mean : ℝ) (num_observations : ℕ) 
  (incorrect_observation correct_observation : ℝ)
  (h_mean : mean = 36) (h_num_observations : num_observations = 50)
  (h_incorrect_observation : incorrect_observation = 23) 
  (h_correct_observation : correct_observation = 44)
  : (mean * num_observations + (correct_observation - incorrect_observation)) / num_observations = 36.42 := 
by
  sorry

end corrected_mean_l1940_194077


namespace businessmen_drink_neither_l1940_194045

theorem businessmen_drink_neither (n c t b : ℕ) 
  (h_n : n = 30) 
  (h_c : c = 15) 
  (h_t : t = 13) 
  (h_b : b = 7) : 
  n - (c + t - b) = 9 := 
  by
  sorry

end businessmen_drink_neither_l1940_194045


namespace test_two_categorical_features_l1940_194032

-- Definitions based on the problem conditions
def is_testing_method (method : String) : Prop :=
  method = "Three-dimensional bar chart" ∨
  method = "Two-dimensional bar chart" ∨
  method = "Contour bar chart" ∨
  method = "Independence test"

noncomputable def correct_method : String :=
  "Independence test"

-- Theorem statement based on the problem and solution
theorem test_two_categorical_features :
  ∀ m : String, is_testing_method m → m = correct_method :=
by
  sorry

end test_two_categorical_features_l1940_194032


namespace range_of_a_l1940_194010

open Set

theorem range_of_a (a : ℝ) : (-3 < a ∧ a < -1) ↔ (∀ x, x < -1 ∨ 5 < x ∨ (a < x ∧ x < a+8)) :=
sorry

end range_of_a_l1940_194010


namespace part_a_solution_part_b_solution_l1940_194044

-- Part (a)
theorem part_a_solution (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y + 13 = 0 ↔ (x = 2 ∧ y = -3) :=
sorry

-- Part (b)
theorem part_b_solution (x y : ℝ) :
  xy - 1 = x - y ↔ ((x = 1 ∨ y = 1) ∨ (x ≠ 1 ∧ y ≠ 1)) :=
sorry

end part_a_solution_part_b_solution_l1940_194044


namespace vertex_of_parabola_l1940_194064

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := (x + 2)^2 - 1

-- Define the vertex point
def vertex : ℝ × ℝ := (-2, -1)

-- The theorem we need to prove
theorem vertex_of_parabola : ∀ x : ℝ, parabola x = (x + 2)^2 - 1 → vertex = (-2, -1) := 
by
  sorry

end vertex_of_parabola_l1940_194064


namespace least_possible_value_l1940_194098

theorem least_possible_value (x y : ℝ) : (x + y - 1)^2 + (x * y)^2 ≥ 0 :=
by 
  sorry

end least_possible_value_l1940_194098


namespace triangle_perimeter_l1940_194061

theorem triangle_perimeter (x : ℕ) (h_odd : x % 2 = 1) (h_range : 1 < x ∧ x < 5) : 2 + 3 + x = 8 :=
by
  sorry

end triangle_perimeter_l1940_194061


namespace sum_of_factors_of_30_is_72_l1940_194027

-- Define the set of whole-number factors of 30
def factors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15, 30}

-- Sum of the factors of 30
def sum_factors_30 : ℕ := factors_30.sum id

-- Theorem stating the sum of the whole-number factors of 30 is 72 
theorem sum_of_factors_of_30_is_72 : sum_factors_30 = 72 := by
  sorry

end sum_of_factors_of_30_is_72_l1940_194027


namespace rectangle_dimensions_l1940_194079

theorem rectangle_dimensions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_area : x * y = 36) (h_perimeter : 2 * x + 2 * y = 30) : 
  (x = 12 ∧ y = 3) ∨ (x = 3 ∧ y = 12) :=
by
  sorry

end rectangle_dimensions_l1940_194079


namespace probability_two_slate_rocks_l1940_194063

theorem probability_two_slate_rocks 
    (n_slate : ℕ) (n_pumice : ℕ) (n_granite : ℕ)
    (h_slate : n_slate = 12)
    (h_pumice : n_pumice = 16)
    (h_granite : n_granite = 8) :
    (n_slate / (n_slate + n_pumice + n_granite)) * ((n_slate - 1) / (n_slate + n_pumice + n_granite - 1)) = 11 / 105 :=
by
    sorry

end probability_two_slate_rocks_l1940_194063


namespace largest_fraction_l1940_194088

theorem largest_fraction (a b c d e : ℝ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (b + d + e) / (a + c) > max ((a + b + e) / (c + d))
                        (max ((a + d) / (b + e))
                            (max ((b + c) / (a + e)) ((c + e) / (a + b + d)))) := 
sorry

end largest_fraction_l1940_194088


namespace overall_avg_is_60_l1940_194031

-- Define the number of students and average marks for each class
def classA_students : ℕ := 30
def classA_avg_marks : ℕ := 40

def classB_students : ℕ := 50
def classB_avg_marks : ℕ := 70

def classC_students : ℕ := 25
def classC_avg_marks : ℕ := 55

def classD_students : ℕ := 45
def classD_avg_marks : ℕ := 65

-- Calculate the total number of students
def total_students : ℕ := 
  classA_students + classB_students + classC_students + classD_students

-- Calculate the total marks for each class
def total_marks_A : ℕ := classA_students * classA_avg_marks
def total_marks_B : ℕ := classB_students * classB_avg_marks
def total_marks_C : ℕ := classC_students * classC_avg_marks
def total_marks_D : ℕ := classD_students * classD_avg_marks

-- Calculate the combined total marks of all classes
def combined_total_marks : ℕ := 
  total_marks_A + total_marks_B + total_marks_C + total_marks_D

-- Calculate the overall average marks
def overall_avg_marks : ℕ := combined_total_marks / total_students

-- Prove that the overall average marks is 60
theorem overall_avg_is_60 : overall_avg_marks = 60 := by
  sorry -- Proof will be written here

end overall_avg_is_60_l1940_194031


namespace trays_from_first_table_l1940_194091

-- Definitions based on conditions
def trays_per_trip : ℕ := 4
def trips : ℕ := 3
def trays_from_second_table : ℕ := 2

-- Theorem statement to prove the number of trays picked up from the first table
theorem trays_from_first_table : trays_per_trip * trips - trays_from_second_table = 10 := by
  sorry

end trays_from_first_table_l1940_194091


namespace arithmetic_sequence_a4_l1940_194001

theorem arithmetic_sequence_a4 (S n : ℕ) (a : ℕ → ℕ) (h1 : S = 28) (h2 : S = 7 * a 4) : a 4 = 4 :=
by sorry

end arithmetic_sequence_a4_l1940_194001


namespace find_triplets_l1940_194053

theorem find_triplets (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  (a^m + 1 ∣ (a + 1)^n) ↔ ((a = 1) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
by
  sorry

end find_triplets_l1940_194053


namespace stones_in_10th_pattern_l1940_194054

def stones_in_nth_pattern (n : ℕ) : ℕ :=
n * (3 * n - 1) / 2 + 1

theorem stones_in_10th_pattern : stones_in_nth_pattern 10 = 145 :=
by
  sorry

end stones_in_10th_pattern_l1940_194054


namespace suff_not_necessary_no_real_solutions_l1940_194073

theorem suff_not_necessary_no_real_solutions :
  ∀ m : ℝ, |m| < 1 → (m : ℝ)^2 < 4 ∧ ∃ x, x^2 - m * x + 1 = 0 →
  ∀ a b : ℝ, (a = 1) ∧ (b = -m) ∧ (c = 1) → (b^2 - 4 * a * c) < 0 ∧ (m > -2) ∧ (m < 2) :=
by
  sorry

end suff_not_necessary_no_real_solutions_l1940_194073


namespace find_m_l1940_194096

open Nat

theorem find_m (m : ℕ) (h_pos : 0 < m) 
  (a : ℕ := Nat.choose (2 * m) m) 
  (b : ℕ := Nat.choose (2 * m + 1) m)
  (h_eq : 13 * a = 7 * b) : 
  m = 6 :=
by
  sorry

end find_m_l1940_194096


namespace Sam_age_proof_l1940_194034

-- Define the conditions (Phoebe's current age, Raven's age relation, Sam's age definition)
def Phoebe_current_age : ℕ := 10
def Raven_in_5_years (R : ℕ) : Prop := R + 5 = 4 * (Phoebe_current_age + 5)
def Sam_age (R : ℕ) : ℕ := 2 * ((R + 3) - (Phoebe_current_age + 3))

-- The proof statement for Sam's current age
theorem Sam_age_proof (R : ℕ) (h : Raven_in_5_years R) : Sam_age R = 90 := by
  sorry

end Sam_age_proof_l1940_194034


namespace correct_option_l1940_194002

theorem correct_option (a b c : ℝ) : 
  (5 * a - (b + 2 * c) = 5 * a + b - 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a - b + 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a + b + 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a - b - 2 * c) ↔ 
  (5 * a - (b + 2 * c) = 5 * a - b - 2 * c) :=
by
  sorry

end correct_option_l1940_194002


namespace find_number_l1940_194042

theorem find_number (x number : ℝ) (h₁ : 5 - (5 / x) = number + (4 / x)) (h₂ : x = 9) : number = 4 :=
by
  subst h₂
  -- proof steps
  sorry

end find_number_l1940_194042


namespace triangle_area_l1940_194084

theorem triangle_area (a c : ℝ) (B : ℝ) (h_a : a = 7) (h_c : c = 5) (h_B : B = 120 * Real.pi / 180) : 
  (1 / 2 * a * c * Real.sin B) = 35 * Real.sqrt 3 / 4 := by
  sorry

end triangle_area_l1940_194084


namespace inequality_proof_l1940_194072

variable (a b : ℝ)

theorem inequality_proof (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 4) : 
  (1 / (a^2 + b^2) ≤ 1 / 8) :=
by
  sorry

end inequality_proof_l1940_194072


namespace remainder_of_product_div_10_l1940_194016

theorem remainder_of_product_div_10 : 
  (3251 * 7462 * 93419) % 10 = 8 := 
sorry

end remainder_of_product_div_10_l1940_194016


namespace ratio_green_to_yellow_l1940_194070

theorem ratio_green_to_yellow (yellow fish blue fish green fish total fish : ℕ) 
  (h_yellow : yellow = 12)
  (h_blue : blue = yellow / 2)
  (h_total : total = yellow + blue + green)
  (h_aquarium_total : total = 42) : 
  green / yellow = 2 := 
sorry

end ratio_green_to_yellow_l1940_194070


namespace speed_of_stream_l1940_194089

theorem speed_of_stream (v_d v_u : ℝ) (h_d : v_d = 13) (h_u : v_u = 8) :
  (v_d - v_u) / 2 = 2.5 :=
by
  -- Insert proof steps here
  sorry

end speed_of_stream_l1940_194089


namespace female_students_count_l1940_194078

variable (F : ℕ)

theorem female_students_count
    (avg_all_students : ℕ)
    (avg_male_students : ℕ)
    (avg_female_students : ℕ)
    (num_male_students : ℕ)
    (condition1 : avg_all_students = 90)
    (condition2 : avg_male_students = 82)
    (condition3 : avg_female_students = 92)
    (condition4 : num_male_students = 8)
    (condition5 : 8 * 82 + F * 92 = (8 + F) * 90) : 
    F = 32 := 
by 
  sorry

end female_students_count_l1940_194078


namespace isosceles_triangle_perimeter_l1940_194038

variable (a b c : ℝ) (h_iso : a = b ∨ a = c ∨ b = c) (h_a : a = 6) (h_b : b = 6) (h_c : c = 3)
variable (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)

theorem isosceles_triangle_perimeter : a + b + c = 15 :=
by 
  -- Given definitions and triangle inequality
  have h_valid : a = 6 ∧ b = 6 ∧ c = 3 := ⟨h_a, h_b, h_c⟩
  sorry

end isosceles_triangle_perimeter_l1940_194038


namespace students_taking_french_l1940_194025

theorem students_taking_french 
  (Total : ℕ) (G : ℕ) (B : ℕ) (Neither : ℕ) (H_total : Total = 87)
  (H_G : G = 22) (H_B : B = 9) (H_neither : Neither = 33) : 
  ∃ F : ℕ, F = 41 := 
by
  sorry

end students_taking_french_l1940_194025


namespace find_first_number_l1940_194055

theorem find_first_number (x : ℝ) :
  (20 + 40 + 60) / 3 = (x + 70 + 13) / 3 + 9 → x = 10 :=
by
  sorry

end find_first_number_l1940_194055


namespace find_a_plus_c_l1940_194026

def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def g (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

theorem find_a_plus_c (a b c d : ℝ) 
  (h_vertex_f : -a / 2 = v) (h_vertex_g : -c / 2 = w)
  (h_root_v_g : g v c d = 0) (h_root_w_f : f w a b = 0)
  (h_intersect : f 50 a b = -200 ∧ g 50 c d = -200)
  (h_min_value_f : ∀ x, f (-a / 2) a b ≤ f x a b)
  (h_min_value_g : ∀ x, g (-c / 2) c d ≤ g x c d)
  (h_min_difference : f (-a / 2) a b = g (-c / 2) c d - 50) :
  a + c = sorry :=
sorry

end find_a_plus_c_l1940_194026


namespace complex_imaginary_axis_l1940_194006

theorem complex_imaginary_axis (a : ℝ) : (a^2 - 2 * a = 0) ↔ (a = 0 ∨ a = 2) := 
by
  sorry

end complex_imaginary_axis_l1940_194006


namespace parabola_focus_distance_l1940_194097

noncomputable def distance_to_focus (p : ℝ) (M : ℝ × ℝ) : ℝ :=
  let focus := (p, 0)
  let (x1, y1) := M
  let (x2, y2) := focus
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) + p

theorem parabola_focus_distance
  (M : ℝ × ℝ) (p : ℝ)
  (hM : M = (2, 2))
  (hp : p = 1) :
  distance_to_focus p M = Real.sqrt 5 + 1 :=
by
  sorry

end parabola_focus_distance_l1940_194097


namespace contrapositive_l1940_194047

variable (Line Circle : Type) (distance : Line → Circle → ℝ) (radius : Circle → ℝ)
variable (is_tangent : Line → Circle → Prop)

-- Original proposition in Lean notation:
def original_proposition (l : Line) (c : Circle) : Prop :=
  distance l c ≠ radius c → ¬ is_tangent l c

-- Contrapositive of the original proposition:
theorem contrapositive (l : Line) (c : Circle) : Prop :=
  is_tangent l c → distance l c = radius c

end contrapositive_l1940_194047


namespace carpet_needed_l1940_194066

/-- A rectangular room with dimensions 15 feet by 9 feet has a non-carpeted area occupied by 
a table with dimensions 3 feet by 2 feet. We want to prove that the number of square yards 
of carpet needed to cover the rest of the floor is 15. -/
theorem carpet_needed
  (room_length : ℝ) (room_width : ℝ) (table_length : ℝ) (table_width : ℝ)
  (h_room : room_length = 15) (h_room_width : room_width = 9)
  (h_table : table_length = 3) (h_table_width : table_width = 2) : 
  (⌈(((room_length * room_width) - (table_length * table_width)) / 9 : ℝ)⌉ = 15) := 
by
  sorry

end carpet_needed_l1940_194066


namespace proof_sum_of_drawn_kinds_l1940_194048

def kindsGrains : Nat := 40
def kindsVegetableOils : Nat := 10
def kindsAnimalFoods : Nat := 30
def kindsFruitsAndVegetables : Nat := 20
def totalKindsFood : Nat := kindsGrains + kindsVegetableOils + kindsAnimalFoods + kindsFruitsAndVegetables
def sampleSize : Nat := 20
def samplingRatio : Nat := sampleSize / totalKindsFood

def numKindsVegetableOilsDrawn : Nat := kindsVegetableOils / 5
def numKindsFruitsAndVegetablesDrawn : Nat := kindsFruitsAndVegetables / 5
def sumVegetableOilsAndFruitsAndVegetablesDrawn : Nat := numKindsVegetableOilsDrawn + numKindsFruitsAndVegetablesDrawn

theorem proof_sum_of_drawn_kinds : sumVegetableOilsAndFruitsAndVegetablesDrawn = 6 := by
  have h1 : totalKindsFood = 100 := by rfl
  have h2 : samplingRatio = 1 / 5 := by
    calc
      sampleSize / totalKindsFood
      _ = 20 / 100 := rfl
      _ = 1 / 5 := by norm_num
  have h3 : numKindsVegetableOilsDrawn = 2 := by
    calc
      kindsVegetableOils / 5
      _ = 10 / 5 := rfl
      _ = 2 := by norm_num
  have h4 : numKindsFruitsAndVegetablesDrawn = 4 := by
    calc
      kindsFruitsAndVegetables / 5
      _ = 20 / 5 := rfl
      _ = 4 := by norm_num
  calc
    sumVegetableOilsAndFruitsAndVegetablesDrawn
    _ = numKindsVegetableOilsDrawn + numKindsFruitsAndVegetablesDrawn := rfl
    _ = 2 + 4 := by rw [h3, h4]
    _ = 6 := by norm_num

end proof_sum_of_drawn_kinds_l1940_194048


namespace prove_min_period_and_max_value_l1940_194075

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2

theorem prove_min_period_and_max_value :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ y : ℝ, y ≤ f y) :=
by
  -- Proof goes here
  sorry

end prove_min_period_and_max_value_l1940_194075


namespace no_solution_intervals_l1940_194080

theorem no_solution_intervals (a : ℝ) :
  (a < -17 ∨ a > 0) → ¬∃ x : ℝ, 7 * |x - 4 * a| + |x - a^2| + 6 * x - 3 * a = 0 :=
by
  sorry

end no_solution_intervals_l1940_194080


namespace minimum_value_l1940_194083

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) :
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) ≥ 9 / 4 :=
by sorry

end minimum_value_l1940_194083


namespace expression_value_zero_l1940_194099

variable (x : ℝ)

theorem expression_value_zero (h : x^2 + 3 * x - 3 = 0) : x^3 + 2 * x^2 - 6 * x + 3 = 0 := 
by
  sorry

end expression_value_zero_l1940_194099


namespace basic_computer_price_l1940_194015

variable (C P : ℕ)

theorem basic_computer_price 
  (h1 : C + P = 2500)
  (h2 : P = (C + 500 + P) / 3) : 
  C = 1500 := 
sorry

end basic_computer_price_l1940_194015


namespace locus_of_M_is_ellipse_l1940_194056

theorem locus_of_M_is_ellipse :
  ∀ (a b : ℝ) (M : ℝ × ℝ),
  a > b → b > 0 → (∃ x y : ℝ, 
  (M = (x, y)) ∧ 
  ∃ (P : ℝ × ℝ),
  (∃ x0 y0 : ℝ, P = (x0, y0) ∧ (x0^2 / a^2 + y0^2 / b^2 = 1)) ∧ 
  P ≠ (a, 0) ∧ P ≠ (-a, 0) ∧
  (∃ t : ℝ, t = (x^2 + y^2 - a^2) / (2 * y)) ∧ 
  (∃ x0 y0 : ℝ, 
    x0 = -x ∧ 
    y0 = 2 * t - y ∧
    x0^2 / a^2 + y0^2 / b^2 = 1)) →
  ∃ (x y : ℝ),
  M = (x, y) ∧ 
  (x^2 / a^2 + y^2 / (a^4 / b^2) = 1) := 
sorry

end locus_of_M_is_ellipse_l1940_194056


namespace soup_adult_feeding_l1940_194069

theorem soup_adult_feeding (cans_of_soup : ℕ) (cans_for_children : ℕ) (feeding_ratio : ℕ) 
  (children : ℕ) (adults : ℕ) :
  feeding_ratio = 4 → cans_of_soup = 10 → children = 20 →
  cans_for_children = (children / feeding_ratio) → 
  adults = feeding_ratio * (cans_of_soup - cans_for_children) →
  adults = 20 :=
by
  intros h1 h2 h3 h4 h5
  -- proof goes here
  sorry

end soup_adult_feeding_l1940_194069


namespace largest_three_digit_int_l1940_194067

theorem largest_three_digit_int (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : 75 * n ≡ 225 [MOD 300]) : n = 999 :=
sorry

end largest_three_digit_int_l1940_194067


namespace derivative_of_cos_over_x_l1940_194035

open Real

noncomputable def f (x : ℝ) : ℝ := (cos x) / x

theorem derivative_of_cos_over_x (x : ℝ) (h : x ≠ 0) : 
  deriv f x = - (x * sin x + cos x) / (x^2) :=
sorry

end derivative_of_cos_over_x_l1940_194035


namespace probability_f4_positive_l1940_194093

theorem probability_f4_positive {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x)
  (h_fn : ∀ x < 0, f x = a + x + Real.logb 2 (-x)) (h_a : a > -4 ∧ a < 5) :
  (1/3 : ℝ) < (2/3 : ℝ) :=
sorry

end probability_f4_positive_l1940_194093


namespace group_allocation_minimizes_time_total_duration_after_transfer_l1940_194076

theorem group_allocation_minimizes_time :
  ∃ x y : ℕ,
  x + y = 52 ∧
  (x = 20 ∧ y = 32) ∧
  (min (60 / x) (100 / y) = 25 / 8) := sorry

theorem total_duration_after_transfer (x y x' y' : ℕ) (H : x = 20) (H1 : y = 32) (H2 : x' = x - 6) (H3 : y' = y + 6) :
  min ((100 * (2/5)) / x') ((152 * (2/3)) / y') = 27 / 7 := sorry

end group_allocation_minimizes_time_total_duration_after_transfer_l1940_194076


namespace probability_cd_l1940_194036

theorem probability_cd (P_A P_B : ℚ) (h1 : P_A = 1/4) (h2 : P_B = 1/3) :
  (1 - P_A - P_B = 5/12) :=
by
  -- Placeholder for the proof
  sorry

end probability_cd_l1940_194036


namespace sum_of_solutions_l1940_194000

-- Given the quadratic equation: x^2 + 3x - 20 = 7x + 8
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x - 20 = 7*x + 8

-- Prove that the sum of the solutions to this quadratic equation is 4
theorem sum_of_solutions : 
  ∀ x1 x2 : ℝ, (quadratic_equation x1) ∧ (quadratic_equation x2) → x1 + x2 = 4 :=
by
  sorry

end sum_of_solutions_l1940_194000


namespace geometric_series_first_term_l1940_194095

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_ratio : r = 1/4) (h_sum : S = 80) (h_series : S = a / (1 - r)) :
  a = 60 :=
by
  sorry

end geometric_series_first_term_l1940_194095


namespace extremum_values_of_function_l1940_194065

noncomputable def maxValue := Real.sqrt 2 + 1 / Real.sqrt 2
noncomputable def minValue := -Real.sqrt 2 + 1 / Real.sqrt 2

theorem extremum_values_of_function :
  ∀ x : ℝ, - (Real.sqrt 2) + (1 / Real.sqrt 2) ≤ (Real.sin x + Real.cos x + 1 / Real.sqrt (1 + |Real.sin (2 * x)|)) ∧ 
            (Real.sin x + Real.cos x + 1 / Real.sqrt (1 + |Real.sin (2 * x)|)) ≤ (Real.sqrt 2 + 1 / Real.sqrt 2) := 
by
  sorry

end extremum_values_of_function_l1940_194065


namespace area_computation_l1940_194050

noncomputable def areaOfBoundedFigure : ℝ :=
  let x (t : ℝ) := 2 * Real.sqrt 2 * Real.cos t
  let y (t : ℝ) := 5 * Real.sqrt 2 * Real.sin t
  let rectArea := 20
  let integral := ∫ t in (3 * Real.pi / 4)..(Real.pi / 4), 
    (5 * Real.sqrt 2 * Real.sin t) * (-2 * Real.sqrt 2 * Real.sin t)
  (integral / 2) - rectArea

theorem area_computation :
  let x (t : ℝ) := 2 * Real.sqrt 2 * Real.cos t
  let y (t : ℝ) := 5 * Real.sqrt 2 * Real.sin t
  let rectArea := 20
  let integral := ∫ t in (3 * Real.pi / 4)..(Real.pi / 4),
    (5 * Real.sqrt 2 * Real.sin t) * (-2 * Real.sqrt 2 * Real.sin t)
  ((integral / 2) - rectArea) = (5 * Real.pi - 10) :=
by
  sorry

end area_computation_l1940_194050


namespace find_r_l1940_194018

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = Real.log 9 / Real.log 3 := by
  sorry

end find_r_l1940_194018


namespace abs_nested_expression_l1940_194052

theorem abs_nested_expression : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry

end abs_nested_expression_l1940_194052


namespace more_sparrows_than_pigeons_l1940_194021

-- Defining initial conditions
def initial_sparrows := 3
def initial_starlings := 5
def initial_pigeons := 2
def additional_sparrows := 4
def additional_starlings := 2
def additional_pigeons := 3

-- Final counts after additional birds join
def final_sparrows := initial_sparrows + additional_sparrows
def final_pigeons := initial_pigeons + additional_pigeons

-- The statement to be proved
theorem more_sparrows_than_pigeons:
  final_sparrows - final_pigeons = 2 :=
by
  -- proof skipped
  sorry

end more_sparrows_than_pigeons_l1940_194021


namespace coefficient_of_x2_in_expansion_l1940_194020

theorem coefficient_of_x2_in_expansion :
  (x - (2 : ℤ)/x) ^ 4 = 8 * x^2 := sorry

end coefficient_of_x2_in_expansion_l1940_194020


namespace min_value_of_sum_of_squares_l1940_194033

theorem min_value_of_sum_of_squares (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h : a^2 + b^2 - c = 2022) : 
  a^2 + b^2 + c^2 = 2034 ∧ a = 27 ∧ b = 36 ∧ c = 3 := 
sorry

end min_value_of_sum_of_squares_l1940_194033


namespace determine_m_type_l1940_194071

theorem determine_m_type (m : ℝ) :
  ((m^2 + 2*m - 8 = 0) ↔ (m = -4)) ∧
  ((m^2 - 2*m = 0) ↔ (m = 0 ∨ m = 2)) ∧
  ((m^2 - 2*m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 2)) :=
by sorry

end determine_m_type_l1940_194071


namespace inequality_xyz_l1940_194040

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 1/x + 1/y + 1/z = 3) : 
  (x - 1) * (y - 1) * (z - 1) ≤ (1/4) * (x * y * z - 1) := 
by 
  sorry

end inequality_xyz_l1940_194040


namespace tank_capacity_ratio_l1940_194074

-- Definitions from the problem conditions
def tank1_filled : ℝ := 300
def tank2_filled : ℝ := 450
def tank2_percentage_filled : ℝ := 0.45
def additional_needed : ℝ := 1250

-- Theorem statement
theorem tank_capacity_ratio (C1 C2 : ℝ) 
  (h1 : tank1_filled + tank2_filled + additional_needed = C1 + C2)
  (h2 : tank2_filled = tank2_percentage_filled * C2) : 
  C1 / C2 = 2 :=
by
  sorry

end tank_capacity_ratio_l1940_194074


namespace pentagon_area_calc_l1940_194058

noncomputable def pentagon_area : ℝ :=
  let triangle1 := (1 / 2) * 18 * 22
  let triangle2 := (1 / 2) * 30 * 26
  let trapezoid := (1 / 2) * (22 + 30) * 10
  triangle1 + triangle2 + trapezoid

theorem pentagon_area_calc :
  pentagon_area = 848 := by
  sorry

end pentagon_area_calc_l1940_194058


namespace number_of_groups_is_correct_l1940_194092

-- Define the number of students
def number_of_students : ℕ := 16

-- Define the group size
def group_size : ℕ := 4

-- Define the expected number of groups
def expected_number_of_groups : ℕ := 4

-- Prove the expected number of groups when grouping students into groups of four
theorem number_of_groups_is_correct :
  number_of_students / group_size = expected_number_of_groups := by
  sorry

end number_of_groups_is_correct_l1940_194092


namespace fourth_number_in_first_set_88_l1940_194003

theorem fourth_number_in_first_set_88 (x y : ℝ)
  (h1 : (28 + x + 70 + y + 104) / 5 = 67)
  (h2 : (50 + 62 + 97 + 124 + x) / 5 = 75.6) :
  y = 88 :=
by
  sorry

end fourth_number_in_first_set_88_l1940_194003


namespace smallest_n_for_divisibility_l1940_194086

theorem smallest_n_for_divisibility (n : ℕ) (h1 : 24 ∣ n^2) (h2 : 1080 ∣ n^3) : n = 120 :=
sorry

end smallest_n_for_divisibility_l1940_194086


namespace calculation_correct_l1940_194037

theorem calculation_correct : 469111 * 9999 = 4690428889 := 
by sorry

end calculation_correct_l1940_194037


namespace find_other_two_sides_of_isosceles_right_triangle_l1940_194024

noncomputable def is_isosceles_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  ((AB.1 ^ 2 + AB.2 ^ 2 = AC.1 ^ 2 + AC.2 ^ 2 ∧ BC.1 ^ 2 + BC.2 ^ 2 = 2 * (AB.1 ^ 2 + AB.2 ^ 2)) ∨
   (AB.1 ^ 2 + AB.2 ^ 2 = BC.1 ^ 2 + BC.2 ^ 2 ∧ AC.1 ^ 2 + AC.2 ^ 2 = 2 * (AB.1 ^ 2 + AB.2 ^ 2)) ∨
   (AC.1 ^ 2 + AC.2 ^ 2 = BC.1 ^ 2 + BC.2 ^ 2 ∧ AB.1 ^ 2 + AB.2 ^ 2 = 2 * (AC.1 ^ 2 + AC.2 ^ 2)))

theorem find_other_two_sides_of_isosceles_right_triangle (A B C : ℝ × ℝ)
  (h : is_isosceles_right_triangle A B C)
  (line_AB : 2 * A.1 - A.2 = 0)
  (midpoint_hypotenuse : (B.1 + C.1) / 2 = 4 ∧ (B.2 + C.2) / 2 = 2) :
  (A.1 + 2 * A.2 = 2 ∨ A.1 + 2 * A.2 = 14) ∧ 
  ((A.2 = 2 * A.1) ∨ (A.1 = 4)) :=
sorry

end find_other_two_sides_of_isosceles_right_triangle_l1940_194024


namespace crayons_per_friend_l1940_194004

theorem crayons_per_friend (total_crayons : ℕ) (num_friends : ℕ) (h1 : total_crayons = 210) (h2 : num_friends = 30) : total_crayons / num_friends = 7 :=
by
  sorry

end crayons_per_friend_l1940_194004


namespace parabola_directrix_eq_neg_2_l1940_194023

noncomputable def parabola_directrix (a b c : ℝ) : ℝ :=
  (b^2 - 4 * a * c) / (4 * a)

theorem parabola_directrix_eq_neg_2 (x : ℝ) :
  parabola_directrix 1 (-4) 4 = -2 :=
by
  -- proof steps go here
  sorry

end parabola_directrix_eq_neg_2_l1940_194023


namespace f_1987_eq_5_l1940_194013

noncomputable def f : ℕ → ℝ := sorry

axiom f_def : ∀ x : ℕ, x ≥ 0 → ∃ y : ℝ, f x = y
axiom f_one : f 1 = 2
axiom functional_eq : ∀ a b : ℕ, a ≥ 0 → b ≥ 0 → f (a + b) = f a + f b - 3 * f (a * b) + 1

theorem f_1987_eq_5 : f 1987 = 5 := sorry

end f_1987_eq_5_l1940_194013


namespace students_enrolled_in_only_english_l1940_194007

theorem students_enrolled_in_only_english (total_students both_english_german total_german : ℕ) (h1 : total_students = 40) (h2 : both_english_german = 12) (h3 : total_german = 22) (h4 : ∀ s, s < 40) :
  (total_students - (total_german - both_english_german) - both_english_german) = 18 := 
by {
  sorry
}

end students_enrolled_in_only_english_l1940_194007


namespace train_length_l1940_194062

noncomputable def length_of_train (speed_kmph : ℝ) (time_sec : ℝ) (length_platform_m : ℝ) : ℝ :=
  let speed_ms := (speed_kmph * 1000) / 3600
  let distance_covered := speed_ms * time_sec
  distance_covered - length_platform_m

theorem train_length :
  length_of_train 72 25 340.04 = 159.96 := by
  sorry

end train_length_l1940_194062


namespace proof_problem_l1940_194068

variable (a b c d x : ℤ)

-- Conditions
def are_opposite (a b : ℤ) : Prop := a + b = 0
def are_reciprocals (c d : ℤ) : Prop := c * d = 1
def largest_negative_integer (x : ℤ) : Prop := x = -1

theorem proof_problem 
  (h1 : are_opposite a b) 
  (h2 : are_reciprocals c d) 
  (h3 : largest_negative_integer x) :
  x^2 - (a + b - c * d)^(2012 : ℕ) + (-c * d)^(2011 : ℕ) = -1 :=
by
  sorry

end proof_problem_l1940_194068


namespace fifth_iteration_perimeter_l1940_194094

theorem fifth_iteration_perimeter :
  let A1_side_length := 1
  let P1 := 3 * A1_side_length
  let P2 := 3 * (A1_side_length * 4 / 3)
  ∀ n : ℕ, P_n = 3 * (4 / 3) ^ (n - 1) →
  P_5 = 3 * (4 / 3) ^ 4 :=
  by sorry

end fifth_iteration_perimeter_l1940_194094


namespace hyperbola_satisfies_conditions_l1940_194046

-- Define the equations of the hyperbolas as predicates
def hyperbola_A (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1
def hyperbola_B (x y : ℝ) : Prop := (x^2 / 4) - y^2 = 1
def hyperbola_C (x y : ℝ) : Prop := (y^2 / 4) - x^2 = 1
def hyperbola_D (x y : ℝ) : Prop := y^2 - (x^2 / 4) = 1

-- Define the conditions on foci and asymptotes
def foci_on_y_axis (h : (ℝ → ℝ → Prop)) : Prop := 
  h = hyperbola_C ∨ h = hyperbola_D

def has_asymptotes (h : (ℝ → ℝ → Prop)) : Prop :=
  ∀ x y, h x y → (y = (1/2) * x ∨ y = -(1/2) * x)

-- The proof statement
theorem hyperbola_satisfies_conditions :
  foci_on_y_axis hyperbola_D ∧ has_asymptotes hyperbola_D ∧ 
    (¬ (foci_on_y_axis hyperbola_A ∧ has_asymptotes hyperbola_A)) ∧ 
    (¬ (foci_on_y_axis hyperbola_B ∧ has_asymptotes hyperbola_B)) ∧ 
    (¬ (foci_on_y_axis hyperbola_C ∧ has_asymptotes hyperbola_C)) := 
by
  sorry

end hyperbola_satisfies_conditions_l1940_194046


namespace mark_total_votes_l1940_194005

-- Definitions based on conditions

def voters_area1 : ℕ := 100000
def percentage_won_area1 : ℝ := 0.7
def votes_area1 := (voters_area1 : ℝ) * percentage_won_area1
def votes_area2 := 2 * votes_area1

-- Theorem statement
theorem mark_total_votes :
  (votes_area1 + votes_area2) = 210000 := 
sorry

end mark_total_votes_l1940_194005


namespace billy_has_2_cherries_left_l1940_194017

-- Define the initial number of cherries
def initialCherries : Nat := 74

-- Define the number of cherries eaten
def eatenCherries : Nat := 72

-- Define the number of remaining cherries
def remainingCherries : Nat := initialCherries - eatenCherries

-- Theorem statement: Prove that remainingCherries is equal to 2
theorem billy_has_2_cherries_left : remainingCherries = 2 := by
  sorry

end billy_has_2_cherries_left_l1940_194017


namespace Joe_spent_800_on_hotel_l1940_194009

noncomputable def Joe'sExpenses : Prop :=
  let S := 6000 -- Joe's total savings
  let F := 1200 -- Expense on the flight
  let FD := 3000 -- Expense on food
  let R := 1000 -- Remaining amount after all expenses
  let H := S - R - (F + FD) -- Calculating hotel expense
  H = 800 -- We need to prove the hotel expense equals $800

theorem Joe_spent_800_on_hotel : Joe'sExpenses :=
by {
  -- Proof goes here; currently skipped
  sorry
}

end Joe_spent_800_on_hotel_l1940_194009


namespace circles_are_disjoint_l1940_194019

noncomputable def positional_relationship_of_circles (R₁ R₂ d : ℝ) (h₁ : R₁ ≠ R₂)
  (h₂ : (2 * R₁)^2 - 4 * (R₂^2 - d * (R₂ - R₁)) = 0) : Prop :=
R₁ + R₂ = d

theorem circles_are_disjoint {R₁ R₂ d : ℝ} (h₁ : R₁ ≠ R₂)
  (h₂ : (2 * R₁)^2 - 4 * (R₂^2 - d * (R₂ - R₁)) = 0) : positional_relationship_of_circles R₁ R₂ d h₁ h₂ :=
by sorry

end circles_are_disjoint_l1940_194019


namespace Dan_age_is_28_l1940_194029

theorem Dan_age_is_28 (B D : ℕ) (h1 : B = D - 3) (h2 : B + D = 53) : D = 28 :=
by
  sorry

end Dan_age_is_28_l1940_194029
