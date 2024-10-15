import Mathlib

namespace NUMINAMATH_GPT_intersection_of_function_and_inverse_l939_93948

theorem intersection_of_function_and_inverse (b a : Int) 
  (h₁ : a = 2 * (-4) + b) 
  (h₂ : a = (-4 - b) / 2) 
  : a = -4 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_function_and_inverse_l939_93948


namespace NUMINAMATH_GPT_find_f_five_thirds_l939_93990

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = - f x
def functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (-x)
def specific_value (f : ℝ → ℝ) := f (-1/3) = 1/3

theorem find_f_five_thirds
  (hf_odd : odd_function f)
  (hf_fe : functional_equation f)
  (hf_value : specific_value f) :
  f (5 / 3) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_find_f_five_thirds_l939_93990


namespace NUMINAMATH_GPT_total_money_shared_l939_93988

-- Define the variables and conditions
def joshua_share : ℕ := 30
def justin_share : ℕ := joshua_share / 3
def total_shared_money : ℕ := joshua_share + justin_share

-- State the theorem to prove
theorem total_money_shared : total_shared_money = 40 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_total_money_shared_l939_93988


namespace NUMINAMATH_GPT_problem_solution_set_l939_93918

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^3 - 8 else (-x)^3 - 8

theorem problem_solution_set : 
  { x : ℝ | f (x-2) > 0 } = {x : ℝ | x < 0} ∪ {x : ℝ | x > 4} :=
by sorry

end NUMINAMATH_GPT_problem_solution_set_l939_93918


namespace NUMINAMATH_GPT_geometric_series_six_terms_l939_93914

theorem geometric_series_six_terms :
  (1/4 - 1/16 + 1/64 - 1/256 + 1/1024 - 1/4096 : ℚ) = 4095 / 20480 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_six_terms_l939_93914


namespace NUMINAMATH_GPT_smallest_eraser_packs_needed_l939_93952

def yazmin_packs_condition (pencils_packs erasers_packs pencils_per_pack erasers_per_pack : ℕ) : Prop :=
  pencils_packs * pencils_per_pack = erasers_packs * erasers_per_pack

theorem smallest_eraser_packs_needed (pencils_per_pack erasers_per_pack : ℕ) (h_pencils_5 : pencils_per_pack = 5) (h_erasers_7 : erasers_per_pack = 7) : ∃ erasers_packs, yazmin_packs_condition 7 erasers_packs pencils_per_pack erasers_per_pack ∧ erasers_packs = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_eraser_packs_needed_l939_93952


namespace NUMINAMATH_GPT_mrs_hilt_water_fountain_trips_l939_93938

theorem mrs_hilt_water_fountain_trips (d : ℕ) (t : ℕ) (n : ℕ) 
  (h1 : d = 30) 
  (h2 : t = 120) 
  (h3 : 2 * d * n = t) : 
  n = 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_mrs_hilt_water_fountain_trips_l939_93938


namespace NUMINAMATH_GPT_cos_A_condition_is_isosceles_triangle_tan_sum_l939_93951

variable {A B C a b c : ℝ}

theorem cos_A_condition (h : (3 * b - c) * Real.cos A - a * Real.cos C = 0) :
  Real.cos A = 1 / 3 := sorry

theorem is_isosceles_triangle (ha : a = 2 * Real.sqrt 3)
  (hs : 1 / 2 * b * c * Real.sin A = 3 * Real.sqrt 2) :
  c = 3 ∧ b = 3 := sorry

theorem tan_sum (h_sin : Real.sin B * Real.sin C = 2 / 3)
  (h_cos : Real.cos A = 1 / 3) :
  Real.tan A + Real.tan B + Real.tan C = 4 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_cos_A_condition_is_isosceles_triangle_tan_sum_l939_93951


namespace NUMINAMATH_GPT_cyclic_quadrilateral_area_l939_93954

variable (a b c d R : ℝ)
noncomputable def p : ℝ := (a + b + c + d) / 2
noncomputable def Brahmagupta_area : ℝ := Real.sqrt ((p a b c d - a) * (p a b c d - b) * (p a b c d - c) * (p a b c d - d))

theorem cyclic_quadrilateral_area :
  Brahmagupta_area a b c d = Real.sqrt ((a * b + c * d) * (a * d + b * c) * (a * c + b * d)) / (4 * R) := sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_area_l939_93954


namespace NUMINAMATH_GPT_value_of_y_l939_93986

theorem value_of_y (x y : ℝ) (h1 : x^(2 * y) = 9) (h2 : x = 3) : y = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_y_l939_93986


namespace NUMINAMATH_GPT_winnie_balloons_remainder_l939_93927

theorem winnie_balloons_remainder :
  let red_balloons := 20
  let white_balloons := 40
  let green_balloons := 70
  let chartreuse_balloons := 90
  let violet_balloons := 15
  let friends := 10
  let total_balloons := red_balloons + white_balloons + green_balloons + chartreuse_balloons + violet_balloons
  total_balloons % friends = 5 :=
by
  sorry

end NUMINAMATH_GPT_winnie_balloons_remainder_l939_93927


namespace NUMINAMATH_GPT_calculate_fraction_l939_93999

theorem calculate_fraction (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -1) : 
  (1 / (x - 1)) - (2 / (x^2 - 1)) = 1 / (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l939_93999


namespace NUMINAMATH_GPT_f_at_7_l939_93971

noncomputable def f (x : ℝ) (a b c d : ℝ) := a * x^7 + b * x^5 + c * x^3 + d * x + 5

theorem f_at_7 (a b c d : ℝ) (h : f (-7) a b c d = -7) : f 7 a b c d = 17 := 
by
  sorry

end NUMINAMATH_GPT_f_at_7_l939_93971


namespace NUMINAMATH_GPT_conor_chop_eggplants_l939_93934

theorem conor_chop_eggplants (E : ℕ) 
  (condition1 : E + 9 + 8 = (E + 17))
  (condition2 : 4 * (E + 9 + 8) = 116) :
  E = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_conor_chop_eggplants_l939_93934


namespace NUMINAMATH_GPT_number_of_papers_l939_93978

-- Define the conditions
def folded_pieces (folds : ℕ) : ℕ := 2 ^ folds
def notes_per_day : ℕ := 10
def days_per_notepad : ℕ := 4
def notes_per_notepad : ℕ := notes_per_day * days_per_notepad
def notes_per_paper (folds : ℕ) : ℕ := folded_pieces folds

-- Lean statement for the proof problem
theorem number_of_papers (folds : ℕ) (h_folds : folds = 3) :
  (notes_per_notepad / notes_per_paper folds) = 5 :=
by
  rw [h_folds]
  simp [notes_per_notepad, notes_per_paper, folded_pieces]
  sorry

end NUMINAMATH_GPT_number_of_papers_l939_93978


namespace NUMINAMATH_GPT_reduced_price_of_oil_is_40_l939_93995

variables 
  (P R : ℝ) 
  (hP : 0 < P)
  (hR : R = 0.75 * P)
  (hw : 800 / (0.75 * P) = 800 / P + 5)

theorem reduced_price_of_oil_is_40 : R = 40 :=
sorry

end NUMINAMATH_GPT_reduced_price_of_oil_is_40_l939_93995


namespace NUMINAMATH_GPT_take_home_pay_is_correct_l939_93906

-- Definitions and Conditions
def pay : ℤ := 650
def tax_rate : ℤ := 10

-- Calculations
def tax_amount := pay * tax_rate / 100
def take_home_pay := pay - tax_amount

-- The Proof Statement
theorem take_home_pay_is_correct : take_home_pay = 585 := by
  sorry

end NUMINAMATH_GPT_take_home_pay_is_correct_l939_93906


namespace NUMINAMATH_GPT_complex_series_sum_l939_93925

theorem complex_series_sum (ω : ℂ) (h₁ : ω^7 = 1) (h₂ : ω ≠ 1) :
  (ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 + ω^32 + 
   ω^34 + ω^36 + ω^38 + ω^40 + ω^42 + ω^44 + ω^46 + ω^48 + ω^50 + 
   ω^52 + ω^54) = -1 :=
by
  sorry

end NUMINAMATH_GPT_complex_series_sum_l939_93925


namespace NUMINAMATH_GPT_roses_cut_l939_93953

variable (initial final : ℕ) -- Declare variables for initial and final numbers of roses

-- Define the theorem stating the solution
theorem roses_cut (h1 : initial = 6) (h2 : final = 16) : final - initial = 10 :=
sorry -- Use sorry to skip the proof

end NUMINAMATH_GPT_roses_cut_l939_93953


namespace NUMINAMATH_GPT_cubic_roots_quadratic_l939_93917

theorem cubic_roots_quadratic (A B C p : ℚ)
  (hA : A ≠ 0)
  (h1 : (∀ x : ℚ, A * x^2 + B * x + C = 0 ↔ x = (root1) ∨ x = (root2)))
  (h2 : root1 + root2 = - B / A)
  (h3 : root1 * root2 = C / A)
  (new_eq : ∀ x : ℚ, x^2 + p*x + q = 0 ↔ x = root1^3 ∨ x = root2^3) :
  p = (B^3 - 3 * A * B * C) / A^3 :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_quadratic_l939_93917


namespace NUMINAMATH_GPT_shorter_side_of_quilt_l939_93928

theorem shorter_side_of_quilt :
  ∀ (x : ℕ), (∃ y : ℕ, 24 * y = 144) -> x = 6 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_shorter_side_of_quilt_l939_93928


namespace NUMINAMATH_GPT_smallest_term_at_n_is_4_or_5_l939_93963

def a_n (n : ℕ) : ℝ :=
  n^2 - 9 * n - 100

theorem smallest_term_at_n_is_4_or_5 :
  ∃ n, n = 4 ∨ n = 5 ∧ a_n n = min (a_n 4) (a_n 5) :=
by
  sorry

end NUMINAMATH_GPT_smallest_term_at_n_is_4_or_5_l939_93963


namespace NUMINAMATH_GPT_unique_sum_of_two_cubes_lt_1000_l939_93947

theorem unique_sum_of_two_cubes_lt_1000 
  : ∃ (sums : Finset ℕ), 
    (∀ x ∈ sums, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ x = a^3 + b^3) 
    ∧ sums.card = 40 
    ∧ ∀ x ∈ sums, x < 1000 := 
by sorry

end NUMINAMATH_GPT_unique_sum_of_two_cubes_lt_1000_l939_93947


namespace NUMINAMATH_GPT_determine_xyz_l939_93941

variables {x y z : ℝ}

theorem determine_xyz (h : (x - y - 3)^2 + (y - z)^2 + (x - z)^2 = 3) : 
  x = z + 1 ∧ y = z - 1 := 
sorry

end NUMINAMATH_GPT_determine_xyz_l939_93941


namespace NUMINAMATH_GPT_percentage_of_towns_correct_l939_93996

def percentage_of_towns_with_fewer_than_50000_residents (p1 p2 p3 : ℝ) : ℝ :=
  p1 + p2

theorem percentage_of_towns_correct (p1 p2 p3 : ℝ) (h1 : p1 = 0.15) (h2 : p2 = 0.30) (h3 : p3 = 0.55) :
  percentage_of_towns_with_fewer_than_50000_residents p1 p2 p3 = 0.45 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_of_towns_correct_l939_93996


namespace NUMINAMATH_GPT_simplify_expression_l939_93937

theorem simplify_expression (y : ℝ) : (5 * y) ^ 3 + (4 * y) * (y ^ 2) = 129 * (y ^ 3) := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l939_93937


namespace NUMINAMATH_GPT_range_of_a_l939_93931

/-- Given that the point (1, 1) is located inside the circle (x - a)^2 + (y + a)^2 = 4, 
    proving that the range of values for a is -1 < a < 1. -/
theorem range_of_a (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → 
  (-1 < a ∧ a < 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l939_93931


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l939_93957

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : a 1 + a 9 = 10)
  (h2 : a 2 = -1)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l939_93957


namespace NUMINAMATH_GPT_chord_bisected_vertically_by_line_l939_93968

theorem chord_bisected_vertically_by_line (p : ℝ) (h : p > 0) (l : ℝ → ℝ) (focus : ℝ × ℝ) 
  (h_focus: focus = (p / 2, 0)) (h_line: ∀ x, l x ≠ 0) :
  ¬ ∃ (A B : ℝ × ℝ), 
     A.1 ≠ B.1 ∧
     A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1 ∧ 
     (A.1 + B.1) / 2 = focus.1 ∧ 
     l ((A.1 + B.1) / 2) = focus.2 :=
sorry

end NUMINAMATH_GPT_chord_bisected_vertically_by_line_l939_93968


namespace NUMINAMATH_GPT_equilateral_division_l939_93900

theorem equilateral_division (k : ℕ) :
  (k = 1 ∨ k = 3 ∨ k = 4 ∨ k = 9 ∨ k = 12 ∨ k = 36) ↔
  (k ∣ 36 ∧ ¬ (k = 2 ∨ k = 6 ∨ k = 18)) := by
  sorry

end NUMINAMATH_GPT_equilateral_division_l939_93900


namespace NUMINAMATH_GPT_D_96_equals_112_l939_93962

def multiplicative_decompositions (n : ℕ) : ℕ :=
  sorry -- Define how to find the number of multiplicative decompositions

theorem D_96_equals_112 : multiplicative_decompositions 96 = 112 :=
  sorry

end NUMINAMATH_GPT_D_96_equals_112_l939_93962


namespace NUMINAMATH_GPT_brother_catch_up_in_3_minutes_l939_93933

variables (v_s v_b : ℝ) (t t_new : ℝ)

-- Conditions
def brother_speed_later_leaves_catch (v_b : ℝ) (v_s : ℝ) : Prop :=
18 * v_s = 12 * v_b

def new_speed_of_brother (v_b v_s : ℝ) : ℝ :=
2 * v_b

def time_to_catch_up (v_s : ℝ) (t_new : ℝ) : Prop :=
6 + t_new = 3 * t_new

-- Goal: prove that t_new = 3
theorem brother_catch_up_in_3_minutes (v_s v_b : ℝ) (t_new : ℝ) :
  (brother_speed_later_leaves_catch v_b v_s) → 
  (new_speed_of_brother v_b v_s) = 3 * v_s → 
  time_to_catch_up v_s t_new → 
  t_new = 3 :=
by sorry

end NUMINAMATH_GPT_brother_catch_up_in_3_minutes_l939_93933


namespace NUMINAMATH_GPT_students_not_enrolled_in_course_l939_93907

def total_students : ℕ := 150
def french_students : ℕ := 61
def german_students : ℕ := 32
def spanish_students : ℕ := 45
def french_and_german : ℕ := 15
def french_and_spanish : ℕ := 12
def german_and_spanish : ℕ := 10
def all_three_courses : ℕ := 5

theorem students_not_enrolled_in_course : total_students - 
    (french_students + german_students + spanish_students - 
     french_and_german - french_and_spanish - german_and_spanish + 
     all_three_courses) = 44 := by
  sorry

end NUMINAMATH_GPT_students_not_enrolled_in_course_l939_93907


namespace NUMINAMATH_GPT_isosceles_triangle_l939_93911

variable (a b c : ℝ)
variable (α β γ : ℝ)
variable (h1 : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β))
variable (triangle_angles : γ = π - (α + β))

theorem isosceles_triangle : α = β :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_l939_93911


namespace NUMINAMATH_GPT_angle_sum_proof_l939_93972

theorem angle_sum_proof (A B C x y : ℝ) 
  (hA : A = 35) 
  (hB : B = 65) 
  (hC : C = 40) 
  (hx : x = 130 - C)
  (hy : y = 90 - A) :
  x + y = 140 := by
  sorry

end NUMINAMATH_GPT_angle_sum_proof_l939_93972


namespace NUMINAMATH_GPT_find_remainder_l939_93998

theorem find_remainder (dividend divisor quotient : ℕ) (h1 : dividend = 686) (h2 : divisor = 36) (h3 : quotient = 19) :
  ∃ remainder, dividend = (divisor * quotient) + remainder ∧ remainder = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_remainder_l939_93998


namespace NUMINAMATH_GPT_weight_difference_l939_93987

noncomputable def W_A : ℝ := 78

variable (W_B W_C W_D W_E : ℝ)

axiom cond1 : (W_A + W_B + W_C) / 3 = 84
axiom cond2 : (W_A + W_B + W_C + W_D) / 4 = 80
axiom cond3 : (W_B + W_C + W_D + W_E) / 4 = 79

theorem weight_difference : W_E - W_D = 6 :=
by
  have h1 : W_A = 78 := rfl
  sorry

end NUMINAMATH_GPT_weight_difference_l939_93987


namespace NUMINAMATH_GPT_calc_h_one_l939_93966

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 6
noncomputable def g (x : ℝ) : ℝ := Real.exp (f x) - 3
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- the final theorem that we are proving
theorem calc_h_one : h 1 = 3 * Real.exp 26 - 14 * Real.exp 13 + 21 := by
  sorry

end NUMINAMATH_GPT_calc_h_one_l939_93966


namespace NUMINAMATH_GPT_problem_statement_l939_93969

noncomputable def A := 5 * Real.pi / 12
noncomputable def B := Real.pi / 3
noncomputable def C := Real.pi / 4
noncomputable def b := Real.sqrt 3
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3

theorem problem_statement :
  (Set.Icc (-2 : ℝ) 2 = Set.image f Set.univ) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∃ (area : ℝ), area = (3 + Real.sqrt 3) / 4)
:= sorry

end NUMINAMATH_GPT_problem_statement_l939_93969


namespace NUMINAMATH_GPT_find_center_and_radius_sum_l939_93991

-- Define the given equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 16 * x + y^2 + 10 * y = -75

-- Define the center of the circle
def center (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → (x = a) ∧ (y = b)

-- Define the radius of the circle
def radius (r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → (x^2 - 16 * x + y^2 + 10 * y = r^2)

-- Main theorem to prove a + b + r = 3 + sqrt 14
theorem find_center_and_radius_sum (a b r : ℝ) (h_cen : center a b) (h_rad : radius r) : 
  a + b + r = 3 + Real.sqrt 14 :=
  sorry

end NUMINAMATH_GPT_find_center_and_radius_sum_l939_93991


namespace NUMINAMATH_GPT_symmetry_x_axis_l939_93910

theorem symmetry_x_axis (a b : ℝ) (h1 : a - 3 = 2) (h2 : 1 = -(b + 1)) : a + b = 3 :=
by
  sorry

end NUMINAMATH_GPT_symmetry_x_axis_l939_93910


namespace NUMINAMATH_GPT_speed_of_policeman_l939_93912

theorem speed_of_policeman 
  (d_initial : ℝ) 
  (v_thief : ℝ) 
  (d_thief : ℝ)
  (d_policeman : ℝ)
  (h_initial : d_initial = 100) 
  (h_v_thief : v_thief = 8) 
  (h_d_thief : d_thief = 400) 
  (h_d_policeman : d_policeman = 500) 
  : ∃ (v_p : ℝ), v_p = 10 :=
by
  -- Use the provided conditions
  sorry

end NUMINAMATH_GPT_speed_of_policeman_l939_93912


namespace NUMINAMATH_GPT_people_sharing_cookies_l939_93977

theorem people_sharing_cookies (total_cookies : ℕ) (cookies_per_person : ℕ) (people : ℕ) 
  (h1 : total_cookies = 24) (h2 : cookies_per_person = 4) (h3 : total_cookies = cookies_per_person * people) : 
  people = 6 :=
by
  sorry

end NUMINAMATH_GPT_people_sharing_cookies_l939_93977


namespace NUMINAMATH_GPT_factor_of_land_increase_l939_93967

-- Definitions of the conditions in the problem:
def initial_money_given_by_blake : ℝ := 20000
def money_received_by_blake_after_sale : ℝ := 30000

-- The main theorem to prove
theorem factor_of_land_increase (F : ℝ) : 
  (1/2) * (initial_money_given_by_blake * F) = money_received_by_blake_after_sale → 
  F = 3 :=
by sorry

end NUMINAMATH_GPT_factor_of_land_increase_l939_93967


namespace NUMINAMATH_GPT_solve_inequalities_solve_linear_system_l939_93982

-- System of Inequalities
theorem solve_inequalities (x : ℝ) (h1 : x + 2 > 1) (h2 : 2 * x < x + 3) : -1 < x ∧ x < 3 :=
by
  sorry

-- System of Linear Equations
theorem solve_linear_system (x y : ℝ) (h1 : 3 * x + 2 * y = 12) (h2 : 2 * x - y = 1) : x = 2 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_solve_linear_system_l939_93982


namespace NUMINAMATH_GPT_no_such_function_exists_l939_93909

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∃ M > 0, ∀ x : ℝ, -M ≤ f x ∧ f x ≤ M) ∧
                    (f 1 = 1) ∧
                    (∀ x : ℝ, x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l939_93909


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l939_93922

theorem hyperbola_eccentricity (a b c e : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1 → (x, y) ≠ (3, -4))
  (h2 : b / a = 4 / 3)
  (h3 : b^2 = c^2 - a^2)
  (h4 : c / a = e):
  e = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l939_93922


namespace NUMINAMATH_GPT_type_B_ratio_l939_93964

theorem type_B_ratio
    (num_A : ℕ)
    (total_bricks : ℕ)
    (other_bricks : ℕ)
    (h1 : num_A = 40)
    (h2 : total_bricks = 150)
    (h3 : other_bricks = 90) :
    (total_bricks - num_A - other_bricks) / num_A = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_type_B_ratio_l939_93964


namespace NUMINAMATH_GPT_correct_answers_unanswered_minimum_correct_answers_l939_93919

-- Definition of the conditions in the problem
def total_questions := 25
def unanswered_questions := 1
def correct_points := 4
def wrong_points := -1
def total_score_1 := 86
def total_score_2 := 90

-- Part 1: Define the conditions and prove that x = 22
theorem correct_answers_unanswered (x : ℕ) (h1 : total_questions - unanswered_questions = 24)
  (h2 : 4 * x + wrong_points * (total_questions - unanswered_questions - x) = total_score_1) : x = 22 :=
sorry

-- Part 2: Define the conditions and prove that at least 23 correct answers are needed
theorem minimum_correct_answers (a : ℕ)
  (h3 : correct_points * a + wrong_points * (total_questions - a) ≥ total_score_2) : a ≥ 23 :=
sorry

end NUMINAMATH_GPT_correct_answers_unanswered_minimum_correct_answers_l939_93919


namespace NUMINAMATH_GPT_yoongi_initial_books_l939_93913

theorem yoongi_initial_books 
  (Y E U : ℕ)
  (h1 : Y - 5 + 15 = 45)
  (h2 : E + 5 - 10 = 45)
  (h3 : U - 15 + 10 = 45) : 
  Y = 35 := 
by 
  -- To be completed with proof
  sorry

end NUMINAMATH_GPT_yoongi_initial_books_l939_93913


namespace NUMINAMATH_GPT_no_solution_for_k_eq_2_l939_93915

theorem no_solution_for_k_eq_2 :
  ∀ m n : ℕ, m ≠ n → ¬ (lcm m n - gcd m n = 2 * (m - n)) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_k_eq_2_l939_93915


namespace NUMINAMATH_GPT_train_stop_times_l939_93983

theorem train_stop_times :
  ∀ (speed_without_stops_A speed_with_stops_A speed_without_stops_B speed_with_stops_B : ℕ),
  speed_without_stops_A = 45 →
  speed_with_stops_A = 30 →
  speed_without_stops_B = 60 →
  speed_with_stops_B = 40 →
  (60 * (speed_without_stops_A - speed_with_stops_A) / speed_without_stops_A = 20) ∧
  (60 * (speed_without_stops_B - speed_with_stops_B) / speed_without_stops_B = 20) :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_stop_times_l939_93983


namespace NUMINAMATH_GPT_evaluate_expr_l939_93994

theorem evaluate_expr :
  (3 * Real.sqrt 7) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 11) = 
  -1 / 6 * (Real.sqrt 21 + Real.sqrt 35 - Real.sqrt 77) - 7 / 3 := by
  sorry

end NUMINAMATH_GPT_evaluate_expr_l939_93994


namespace NUMINAMATH_GPT_simplify_fraction_l939_93926

theorem simplify_fraction (a b m n : ℕ) (h : a ≠ 0 ∧ b ≠ 0 ∧ m ≠ 0 ∧ n ≠ 0) : 
  (a^2 * b) / (m * n^2) / ((a * b) / (3 * m * n)) = 3 * a / n :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l939_93926


namespace NUMINAMATH_GPT_center_of_circle_from_diameter_l939_93974

theorem center_of_circle_from_diameter (x1 y1 x2 y2 : ℝ) 
  (h1 : x1 = 3) (h2 : y1 = -3) (h3 : x2 = 13) (h4 : y2 = 17) :
  (x1 + x2) / 2 = 8 ∧ (y1 + y2) / 2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_from_diameter_l939_93974


namespace NUMINAMATH_GPT_wrong_conclusion_l939_93944

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem wrong_conclusion {a b c : ℝ} (h₀ : a ≠ 0) (h₁ : 2 * a + b = 0) (h₂ : a + b + c = 3) (h₃ : 4 * a + 2 * b + c = 8) :
  quadratic a b c (-1) ≠ 0 :=
sorry

end NUMINAMATH_GPT_wrong_conclusion_l939_93944


namespace NUMINAMATH_GPT_complex_addition_l939_93935

def imag_unit_squared (i : ℂ) : Prop := i * i = -1

theorem complex_addition (a b : ℝ) (i : ℂ)
  (h1 : a + b * i = i * i)
  (h2 : imag_unit_squared i) : a + b = -1 := 
sorry

end NUMINAMATH_GPT_complex_addition_l939_93935


namespace NUMINAMATH_GPT_negation_of_existential_l939_93984

theorem negation_of_existential :
  (∀ x : ℝ, x^2 + x - 1 ≤ 0) ↔ ¬ (∃ x : ℝ, x^2 + x - 1 > 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_existential_l939_93984


namespace NUMINAMATH_GPT_workers_contribution_l939_93939

theorem workers_contribution (W C : ℕ) 
    (h1 : W * C = 300000) 
    (h2 : W * (C + 50) = 325000) : 
    W = 500 :=
by
    sorry

end NUMINAMATH_GPT_workers_contribution_l939_93939


namespace NUMINAMATH_GPT_sqrt_of_4_eq_2_l939_93930

theorem sqrt_of_4_eq_2 : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_of_4_eq_2_l939_93930


namespace NUMINAMATH_GPT_down_payment_calculation_l939_93908

theorem down_payment_calculation 
  (purchase_price : ℝ)
  (monthly_payment : ℝ)
  (n : ℕ)
  (interest_rate : ℝ)
  (down_payment : ℝ) :
  purchase_price = 127 ∧ 
  monthly_payment = 10 ∧ 
  n = 12 ∧ 
  interest_rate = 0.2126 ∧
  down_payment + (n * monthly_payment) = purchase_price * (1 + interest_rate) 
  → down_payment = 34 := 
sorry

end NUMINAMATH_GPT_down_payment_calculation_l939_93908


namespace NUMINAMATH_GPT_sum_is_two_l939_93973

noncomputable def compute_sum (x : ℂ) (hx : x^7 = 1) (hx_ne : x ≠ 1) : ℂ :=
  (x^2 / (x - 1)) + (x^4 / (x^2 - 1)) + (x^6 / (x^3 - 1)) + (x^8 / (x^4 - 1)) + (x^10 / (x^5 - 1)) + (x^12 / (x^6 - 1))

theorem sum_is_two (x : ℂ) (hx : x^7 = 1) (hx_ne : x ≠ 1) : compute_sum x hx hx_ne = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_is_two_l939_93973


namespace NUMINAMATH_GPT_sarah_initial_money_l939_93920

def initial_money 
  (cost_toy_car : ℕ)
  (cost_scarf : ℕ)
  (cost_beanie : ℕ)
  (remaining_money : ℕ)
  (number_of_toy_cars : ℕ) : ℕ :=
  remaining_money + cost_beanie + cost_scarf + number_of_toy_cars * cost_toy_car

theorem sarah_initial_money : 
  (initial_money 11 10 14 7 2) = 53 :=
by
  rfl 

end NUMINAMATH_GPT_sarah_initial_money_l939_93920


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l939_93943

def line1 (a : ℝ) (x y : ℝ) := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) := (a - 1) * x - y + a = 0
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, line1 a x y = line2 a x y

theorem necessary_and_sufficient_condition (a : ℝ) : 
  (a = 2 ↔ parallel a) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l939_93943


namespace NUMINAMATH_GPT_container_ratio_l939_93975

theorem container_ratio (V1 V2 V3 : ℝ)
  (h1 : (3 / 4) * V1 = (5 / 8) * V2)
  (h2 : (5 / 8) * V2 = (1 / 2) * V3) :
  V1 / V3 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_container_ratio_l939_93975


namespace NUMINAMATH_GPT_cost_of_orange_juice_l939_93923

theorem cost_of_orange_juice (O : ℝ) (H1 : ∀ (apple_juice_cost : ℝ), apple_juice_cost = 0.60 ):
  let total_bottles := 70
  let total_cost := 46.20
  let orange_juice_bottles := 42
  let apple_juice_bottles := total_bottles - orange_juice_bottles
  let equation := (orange_juice_bottles * O + apple_juice_bottles * 0.60 = total_cost)
  equation -> O = 0.70 := by
  sorry

end NUMINAMATH_GPT_cost_of_orange_juice_l939_93923


namespace NUMINAMATH_GPT_expression_evaluation_l939_93956

theorem expression_evaluation (a b c : ℤ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 3) :
  (2 * a - (b - 2 * c)) - ((2 * a - b) - 2 * c) + 3 * (a - c) = 27 :=
by
  have ha : a = 8 := h₁
  have hb : b = 10 := h₂
  have hc : c = 3 := h₃
  rw [ha, hb, hc]
  sorry

end NUMINAMATH_GPT_expression_evaluation_l939_93956


namespace NUMINAMATH_GPT_spheres_volume_ratio_l939_93945

theorem spheres_volume_ratio (S1 S2 V1 V2 : ℝ)
  (h1 : S1 / S2 = 1 / 9) 
  (h2a : S1 = 4 * π * r1^2) 
  (h2b : S2 = 4 * π * r2^2)
  (h3a : V1 = 4 / 3 * π * r1^3)
  (h3b : V2 = 4 / 3 * π * r2^3)
  : V1 / V2 = 1 / 27 :=
by
  sorry

end NUMINAMATH_GPT_spheres_volume_ratio_l939_93945


namespace NUMINAMATH_GPT_simon_legos_l939_93955

theorem simon_legos (k b s : ℕ) 
  (h_kent : k = 40)
  (h_bruce : b = k + 20)
  (h_simon : s = b + b / 5) : 
  s = 72 := by
  -- sorry, proof not required.
  sorry

end NUMINAMATH_GPT_simon_legos_l939_93955


namespace NUMINAMATH_GPT_base_of_log_is_176_l939_93950

theorem base_of_log_is_176 
    (x : ℕ)
    (h : ∃ q r : ℕ, x = 19 * q + r ∧ q = 9 ∧ r = 5) :
    x = 176 :=
by
  sorry

end NUMINAMATH_GPT_base_of_log_is_176_l939_93950


namespace NUMINAMATH_GPT_rabbits_to_hamsters_l939_93904

theorem rabbits_to_hamsters (rabbits hamsters : ℕ) (h_ratio : 3 * hamsters = 4 * rabbits) (h_rabbits : rabbits = 18) : hamsters = 24 :=
by
  sorry

end NUMINAMATH_GPT_rabbits_to_hamsters_l939_93904


namespace NUMINAMATH_GPT_abs_neg_five_is_five_l939_93959

theorem abs_neg_five_is_five : abs (-5) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_abs_neg_five_is_five_l939_93959


namespace NUMINAMATH_GPT_vec_expression_l939_93936

def vec_a : ℝ × ℝ := (1, -2)
def vec_b : ℝ × ℝ := (3, 5)

theorem vec_expression : 2 • vec_a + vec_b = (5, 1) := by
  sorry

end NUMINAMATH_GPT_vec_expression_l939_93936


namespace NUMINAMATH_GPT_determine_real_coins_l939_93985

def has_fake_coin (coins : List ℝ) : Prop :=
  ∃ fake_coin ∈ coins, (∀ coin ∈ coins, coin ≠ fake_coin)

theorem determine_real_coins (coins : List ℝ) (h : has_fake_coin coins) (h_length : coins.length = 101) :
  ∃ real_coins : List ℝ, ∀ r ∈ real_coins, r ∈ coins ∧ real_coins.length ≥ 50 :=
by
  sorry

end NUMINAMATH_GPT_determine_real_coins_l939_93985


namespace NUMINAMATH_GPT_fred_balloons_l939_93942

theorem fred_balloons (T S D F : ℕ) (hT : T = 72) (hS : S = 46) (hD : D = 16) (hTotal : T = F + S + D) : F = 10 := 
by
  sorry

end NUMINAMATH_GPT_fred_balloons_l939_93942


namespace NUMINAMATH_GPT_total_age_10_years_from_now_is_75_l939_93961

-- Define the conditions
def eldest_age_now : ℕ := 20
def age_difference : ℕ := 5

-- Define the ages of the siblings 10 years from now
def eldest_age_10_years_from_now : ℕ := eldest_age_now + 10
def second_age_10_years_from_now : ℕ := (eldest_age_now - age_difference) + 10
def third_age_10_years_from_now : ℕ := (eldest_age_now - 2 * age_difference) + 10

-- Define the total age of the siblings 10 years from now
def total_age_10_years_from_now : ℕ := 
  eldest_age_10_years_from_now + 
  second_age_10_years_from_now + 
  third_age_10_years_from_now

-- The theorem statement
theorem total_age_10_years_from_now_is_75 : total_age_10_years_from_now = 75 := 
  by sorry

end NUMINAMATH_GPT_total_age_10_years_from_now_is_75_l939_93961


namespace NUMINAMATH_GPT_negation_of_proposition_p_l939_93980

theorem negation_of_proposition_p :
  (¬(∃ x : ℝ, 0 < x ∧ Real.log x > x - 1)) ↔ (∀ x : ℝ, 0 < x → Real.log x ≤ x - 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_p_l939_93980


namespace NUMINAMATH_GPT_sunzi_problem_solution_l939_93921

theorem sunzi_problem_solution (x y : ℝ) :
  (y = x + 4.5) ∧ (0.5 * y = x - 1) ↔ (y = x + 4.5 ∧ 0.5 * y = x - 1) :=
by 
  sorry

end NUMINAMATH_GPT_sunzi_problem_solution_l939_93921


namespace NUMINAMATH_GPT_chocolates_vs_gums_l939_93949

theorem chocolates_vs_gums 
    (c g : ℝ) 
    (Kolya_claim : 2 * c > 5 * g) 
    (Sasha_claim : ¬ ( 3 * c > 8 * g )) : 
    7 * c ≤ 19 * g := 
sorry

end NUMINAMATH_GPT_chocolates_vs_gums_l939_93949


namespace NUMINAMATH_GPT_correct_conclusions_count_l939_93924

theorem correct_conclusions_count :
  (¬ (¬ p → (q ∨ r)) ↔ (¬ p → ¬ q ∧ ¬ r)) = false ∧
  ((¬ p → q) ↔ (p → ¬ q)) = false ∧
  (¬ ∃ n : ℕ, n > 0 ∧ (n ^ 2 + 3 * n) % 10 = 0 ∧ (∀ n : ℕ, n > 0 → (n ^ 2 + 3 * n) % 10 ≠ 0)) = true ∧
  (¬ ∀ x, x ^ 2 - 2 * x + 3 > 0 ∧ (∃ x, x ^ 2 - 2 * x + 3 < 0)) = false :=
by
  sorry

end NUMINAMATH_GPT_correct_conclusions_count_l939_93924


namespace NUMINAMATH_GPT_smallest_number_is_33_l939_93981

theorem smallest_number_is_33 (x : ℝ) 
  (h1 : 2 * x = third)
  (h2 : 4 * x = second)
  (h3 : (x + 2 * x + 4 * x) / 3 = 77) : 
  x = 33 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_number_is_33_l939_93981


namespace NUMINAMATH_GPT_proof_of_problem_l939_93903

noncomputable def problem : Prop :=
  (1 + Real.cos (20 * Real.pi / 180)) / (2 * Real.sin (20 * Real.pi / 180)) -
  (Real.sin (10 * Real.pi / 180) * 
  (1 / Real.tan (5 * Real.pi / 180) - Real.tan (5 * Real.pi / 180))) =
  (Real.sqrt 3) / 2

theorem proof_of_problem : problem :=
by
  sorry

end NUMINAMATH_GPT_proof_of_problem_l939_93903


namespace NUMINAMATH_GPT_rectangle_to_total_height_ratio_l939_93976

theorem rectangle_to_total_height_ratio 
  (total_area : ℕ)
  (width : ℕ)
  (area_per_side : ℕ)
  (height : ℕ)
  (triangle_base : ℕ)
  (triangle_area : ℕ)
  (rect_area : ℕ)
  (total_height : ℕ)
  (ratio : ℚ)
  (h_eqn : 3 * height = area_per_side)
  (h_value : height = total_area / (2 * 3))
  (total_height_eqn : total_height = 2 * height)
  (ratio_eqn : ratio = height / total_height) :
  total_area = 12 → width = 3 → area_per_side = 6 → triangle_base = 3 →
  triangle_area = triangle_base * height / 2 → rect_area = width * height →
  rect_area = area_per_side → ratio = 1 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rectangle_to_total_height_ratio_l939_93976


namespace NUMINAMATH_GPT_sets_equal_l939_93965

def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }
def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem sets_equal : E = F :=
  sorry

end NUMINAMATH_GPT_sets_equal_l939_93965


namespace NUMINAMATH_GPT_larger_tent_fabric_amount_l939_93940

-- Define the fabric used for the small tent
def small_tent_fabric : ℝ := 4

-- Define the fabric computation for the larger tent
def larger_tent_fabric (small_tent_fabric : ℝ) : ℝ :=
  2 * small_tent_fabric

-- Theorem stating the amount of fabric needed for the larger tent
theorem larger_tent_fabric_amount : larger_tent_fabric small_tent_fabric = 8 :=
by
  -- Skip the actual proof
  sorry

end NUMINAMATH_GPT_larger_tent_fabric_amount_l939_93940


namespace NUMINAMATH_GPT_anne_clean_house_in_12_hours_l939_93932

theorem anne_clean_house_in_12_hours (B A : ℝ) (h1 : 4 * (B + A) = 1) (h2 : 3 * (B + 2 * A) = 1) : A = 1 / 12 ∧ (1 / A) = 12 :=
by
  -- We will leave the proof as a placeholder
  sorry

end NUMINAMATH_GPT_anne_clean_house_in_12_hours_l939_93932


namespace NUMINAMATH_GPT_average_inside_time_l939_93989

theorem average_inside_time (j_awake_frac : ℚ) (j_inside_awake_frac : ℚ) (r_awake_frac : ℚ) (r_inside_day_frac : ℚ) :
  j_awake_frac = 2 / 3 →
  j_inside_awake_frac = 1 / 2 →
  r_awake_frac = 3 / 4 →
  r_inside_day_frac = 2 / 3 →
  (24 * j_awake_frac * j_inside_awake_frac + 24 * r_awake_frac * r_inside_day_frac) / 2 = 10 := 
by
    sorry

end NUMINAMATH_GPT_average_inside_time_l939_93989


namespace NUMINAMATH_GPT_bank_teller_bills_l939_93970

theorem bank_teller_bills (x y : ℕ) (h1 : x + y = 54) (h2 : 5 * x + 20 * y = 780) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_bank_teller_bills_l939_93970


namespace NUMINAMATH_GPT_probability_of_3_black_2_white_l939_93902

def total_balls := 15
def black_balls := 10
def white_balls := 5
def drawn_balls := 5
def drawn_black_balls := 3
def drawn_white_balls := 2

noncomputable def probability_black_white_draw : ℝ :=
  (Nat.choose black_balls drawn_black_balls * Nat.choose white_balls drawn_white_balls : ℝ) /
  (Nat.choose total_balls drawn_balls : ℝ)

theorem probability_of_3_black_2_white :
  probability_black_white_draw = 400 / 1001 := by
  sorry

end NUMINAMATH_GPT_probability_of_3_black_2_white_l939_93902


namespace NUMINAMATH_GPT_union_P_Q_l939_93946

noncomputable def P : Set ℝ := {x : ℝ | abs x ≥ 3}
noncomputable def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x - 1}

theorem union_P_Q :
  (P ∪ Q) = Set.Iic (-3) ∪ Set.Ici (-1) :=
by {
  sorry
}

end NUMINAMATH_GPT_union_P_Q_l939_93946


namespace NUMINAMATH_GPT_probability_at_most_one_correct_in_two_rounds_l939_93960

theorem probability_at_most_one_correct_in_two_rounds :
  let pA := 3 / 5
  let pB := 2 / 3
  let pA_incorrect := 1 - pA
  let pB_incorrect := 1 - pB
  let p_0_correct := pA_incorrect * pA_incorrect * pB_incorrect * pB_incorrect
  let p_1_correct_A1 := pA * pA_incorrect * pB_incorrect * pB_incorrect
  let p_1_correct_A2 := pA_incorrect * pA * pB_incorrect * pB_incorrect
  let p_1_correct_B1 := pA_incorrect * pA_incorrect * pB * pB_incorrect
  let p_1_correct_B2 := pA_incorrect * pA_incorrect * pB_incorrect * pB
  let p_at_most_one := p_0_correct + p_1_correct_A1 + p_1_correct_A2 + 
      p_1_correct_B1 + p_1_correct_B2
  p_at_most_one = 32 / 225 := 
  sorry

end NUMINAMATH_GPT_probability_at_most_one_correct_in_two_rounds_l939_93960


namespace NUMINAMATH_GPT_reflected_coordinates_l939_93905

-- Define the coordinates of point P
def point_P : ℝ × ℝ := (-2, -3)

-- Define the function for reflection across the origin
def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- State the theorem to prove
theorem reflected_coordinates :
  reflect_origin point_P = (2, 3) := by
  sorry

end NUMINAMATH_GPT_reflected_coordinates_l939_93905


namespace NUMINAMATH_GPT_repeating_decimal_fraction_l939_93979

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_l939_93979


namespace NUMINAMATH_GPT_fractions_order_l939_93992

theorem fractions_order :
  (21 / 17) < (18 / 13) ∧ (18 / 13) < (16 / 11) := by
  sorry

end NUMINAMATH_GPT_fractions_order_l939_93992


namespace NUMINAMATH_GPT_four_digit_sum_l939_93993

theorem four_digit_sum (A B : ℕ) (hA : 1000 ≤ A ∧ A < 10000) (hB : 1000 ≤ B ∧ B < 10000) (h : A * B = 16^5 + 2^10) : A + B = 2049 := 
by sorry

end NUMINAMATH_GPT_four_digit_sum_l939_93993


namespace NUMINAMATH_GPT_find_quotient_l939_93901

def dividend : ℕ := 55053
def divisor : ℕ := 456
def remainder : ℕ := 333

theorem find_quotient (Q : ℕ) (h : dividend = (divisor * Q) + remainder) : Q = 120 := by
  sorry

end NUMINAMATH_GPT_find_quotient_l939_93901


namespace NUMINAMATH_GPT_solve_equation_l939_93958

theorem solve_equation (x y z : ℕ) : (3 ^ x + 5 ^ y + 14 = z!) ↔ ((x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l939_93958


namespace NUMINAMATH_GPT_sum_series_eq_one_l939_93997

noncomputable def series : ℝ := ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1)

theorem sum_series_eq_one : series = 1 := 
by sorry

end NUMINAMATH_GPT_sum_series_eq_one_l939_93997


namespace NUMINAMATH_GPT_fraction_to_decimal_l939_93916

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l939_93916


namespace NUMINAMATH_GPT_length_of_AE_l939_93929

/-- Given the conditions on the pentagon ABCDE:
1. AB = 2, BC = 2, CD = 5, DE = 7
2. AC is the largest side in triangle ABC
3. CE is the smallest side in triangle ECD
4. In triangle ACE all sides are integers and have distinct lengths,
prove that the length of side AE is 5. -/
theorem length_of_AE
  (AB BC CD DE : ℕ)
  (hAB : AB = 2)
  (hBC : BC = 2)
  (hCD : CD = 5)
  (hDE : DE = 7)
  (AC : ℕ) 
  (hAC_large : AB < AC ∧ BC < AC)
  (CE : ℕ)
  (hCE_small : CE < CD ∧ CE < DE)
  (AE : ℕ)
  (distinct_sides : ∀ x y z : ℕ, x ≠ y → x ≠ z → y ≠ z → (AC = x ∨ CE = x ∨ AE = x) → (AC = y ∨ CE = y ∨ AE = y) → (AC = z ∨ CE = z ∨ AE = z)) :
  AE = 5 :=
sorry

end NUMINAMATH_GPT_length_of_AE_l939_93929
