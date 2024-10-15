import Mathlib

namespace NUMINAMATH_GPT_center_of_circle_l1859_185944

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 2 = 0

-- Define the condition for the center of the circle
def is_center_of_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = 4

-- The main theorem to be proved
theorem center_of_circle : is_center_of_circle 1 (-1) :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_l1859_185944


namespace NUMINAMATH_GPT_John_has_15_snakes_l1859_185935

theorem John_has_15_snakes (S : ℕ)
  (H1 : ∀ M, M = 2 * S)
  (H2 : ∀ M L, L = M - 5)
  (H3 : ∀ L P, P = L + 8)
  (H4 : ∀ P D, D = P / 3)
  (H5 : S + (2 * S) + ((2 * S) - 5) + (((2 * S) - 5) + 8) + (((((2 * S) - 5) + 8) / 3)) = 114) :
  S = 15 :=
by sorry

end NUMINAMATH_GPT_John_has_15_snakes_l1859_185935


namespace NUMINAMATH_GPT_expression_equal_a_five_l1859_185910

noncomputable def a : ℕ := sorry

theorem expression_equal_a_five (a : ℕ) : (a^4 * a) = a^5 := by
  sorry

end NUMINAMATH_GPT_expression_equal_a_five_l1859_185910


namespace NUMINAMATH_GPT_combined_height_is_320_cm_l1859_185931

-- Define Maria's height in inches
def Maria_height_in_inches : ℝ := 54

-- Define Ben's height in inches
def Ben_height_in_inches : ℝ := 72

-- Define the conversion factor from inches to centimeters
def inch_to_cm : ℝ := 2.54

-- Define the combined height of Maria and Ben in centimeters
def combined_height_in_cm : ℝ := (Maria_height_in_inches + Ben_height_in_inches) * inch_to_cm

-- State and prove that the combined height is 320.0 cm
theorem combined_height_is_320_cm : combined_height_in_cm = 320.0 := by
  sorry

end NUMINAMATH_GPT_combined_height_is_320_cm_l1859_185931


namespace NUMINAMATH_GPT_two_digit_number_l1859_185933

theorem two_digit_number (x y : ℕ) (h1 : x + y = 7) (h2 : (x + 2) + 10 * (y + 2) = 2 * (x + 10 * y) - 3) : (10 * y + x) = 25 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_number_l1859_185933


namespace NUMINAMATH_GPT_average_six_conseq_ints_l1859_185930

theorem average_six_conseq_ints (c d : ℝ) (h₁ : d = c + 2.5) :
  (d - 2 + d - 1 + d + d + 1 + d + 2 + d + 3) / 6 = c + 3 :=
by
  sorry

end NUMINAMATH_GPT_average_six_conseq_ints_l1859_185930


namespace NUMINAMATH_GPT_inequality_am_gm_l1859_185978

theorem inequality_am_gm (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/2) :
  (1 - a) * (1 - b) ≤ 9 / 16 := 
by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l1859_185978


namespace NUMINAMATH_GPT_root_sum_product_eq_l1859_185946

theorem root_sum_product_eq (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) :
  p + q = 69 :=
by 
  sorry

end NUMINAMATH_GPT_root_sum_product_eq_l1859_185946


namespace NUMINAMATH_GPT_percent_decrease_is_80_l1859_185947

-- Definitions based on the conditions
def original_price := 100
def sale_price := 20

-- Theorem statement to prove the percent decrease
theorem percent_decrease_is_80 :
  ((original_price - sale_price) / original_price * 100) = 80 := 
by
  sorry

end NUMINAMATH_GPT_percent_decrease_is_80_l1859_185947


namespace NUMINAMATH_GPT_min_fencing_l1859_185956

variable (w l : ℝ)

noncomputable def area := w * l

noncomputable def length := 2 * w

theorem min_fencing (h1 : area w l ≥ 500) (h2 : l = length w) : 
  w = 5 * Real.sqrt 10 ∧ l = 10 * Real.sqrt 10 :=
  sorry

end NUMINAMATH_GPT_min_fencing_l1859_185956


namespace NUMINAMATH_GPT_vasya_numbers_l1859_185970

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end NUMINAMATH_GPT_vasya_numbers_l1859_185970


namespace NUMINAMATH_GPT_maximum_pizzas_baked_on_Friday_l1859_185938

def george_bakes := 
  let total_pizzas : ℕ := 1000
  let monday_pizzas := total_pizzas * 7 / 10
  let tuesday_pizzas := if monday_pizzas * 4 / 5 < monday_pizzas * 9 / 10 
                        then monday_pizzas * 4 / 5 
                        else monday_pizzas * 9 / 10
  let wednesday_pizzas := if tuesday_pizzas * 4 / 5 < tuesday_pizzas * 9 / 10 
                          then tuesday_pizzas * 4 / 5 
                          else tuesday_pizzas * 9 / 10
  let thursday_pizzas := if wednesday_pizzas * 4 / 5 < wednesday_pizzas * 9 / 10 
                         then wednesday_pizzas * 4 / 5 
                         else wednesday_pizzas * 9 / 10
  let friday_pizzas := if thursday_pizzas * 4 / 5 < thursday_pizzas * 9 / 10 
                       then thursday_pizzas * 4 / 5 
                       else thursday_pizzas * 9 / 10
  friday_pizzas

theorem maximum_pizzas_baked_on_Friday : george_bakes = 2 := by
  sorry

end NUMINAMATH_GPT_maximum_pizzas_baked_on_Friday_l1859_185938


namespace NUMINAMATH_GPT_perpendicular_d_to_BC_l1859_185932

def vector := (ℝ × ℝ)

noncomputable def AB : vector := (1, 1)
noncomputable def AC : vector := (2, 3)

noncomputable def BC : vector := (AC.1 - AB.1, AC.2 - AB.2)

def is_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

noncomputable def d : vector := (-6, 3)

theorem perpendicular_d_to_BC : is_perpendicular d BC :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_d_to_BC_l1859_185932


namespace NUMINAMATH_GPT_marble_solid_color_percentage_l1859_185993

theorem marble_solid_color_percentage (a b : ℕ) (h1 : a = 5) (h2 : b = 85) : a + b = 90 := 
by
  sorry

end NUMINAMATH_GPT_marble_solid_color_percentage_l1859_185993


namespace NUMINAMATH_GPT_dina_dolls_count_l1859_185904

-- Define the conditions
variable (Ivy_dolls : ℕ)
variable (Collectors_Ivy_dolls : ℕ := 20)
variable (Dina_dolls : ℕ)

-- Condition: Ivy has 2/3 of her dolls as collectors editions
def collectors_edition_condition : Prop := (2 / 3 : ℝ) * Ivy_dolls = Collectors_Ivy_dolls

-- Condition: Dina has twice as many dolls as Ivy
def dina_ivy_dolls_relationship : Prop := Dina_dolls = 2 * Ivy_dolls

-- Theorem: Prove that Dina has 60 dolls
theorem dina_dolls_count : collectors_edition_condition Ivy_dolls ∧ dina_ivy_dolls_relationship Ivy_dolls Dina_dolls → Dina_dolls = 60 := by
  sorry

end NUMINAMATH_GPT_dina_dolls_count_l1859_185904


namespace NUMINAMATH_GPT_min_value_arithmetic_sequence_l1859_185943

theorem min_value_arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h_arith_seq : a n = 1 + (n - 1) * 1)
  (h_sum : S n = n * (1 + n) / 2) :
  ∃ n, (S n + 8) / a n = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_arithmetic_sequence_l1859_185943


namespace NUMINAMATH_GPT_find_k_l1859_185912

def a : ℕ := 786
def b : ℕ := 74
def c : ℝ := 1938.8

theorem find_k (k : ℝ) : (a * b) / k = c → k = 30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_k_l1859_185912


namespace NUMINAMATH_GPT_solve_m_n_l1859_185983

theorem solve_m_n (m n : ℤ) :
  (m * 1 + n * 1 = 6) ∧ (m * 2 + n * -1 = 6) → (m = 4) ∧ (n = 2) := by
  sorry

end NUMINAMATH_GPT_solve_m_n_l1859_185983


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_is_39_l1859_185966

theorem number_of_terms_in_arithmetic_sequence_is_39 :
  ∀ (a d l : ℤ), 
  d ≠ 0 → 
  a = 128 → 
  d = -3 → 
  l = 14 → 
  ∃ n : ℕ, (a + (↑n - 1) * d = l) ∧ (n = 39) :=
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_is_39_l1859_185966


namespace NUMINAMATH_GPT_percentage_of_ducks_among_non_heron_l1859_185989

def birds_percentage (geese swans herons ducks total_birds : ℕ) : ℕ :=
  let non_heron_birds := total_birds - herons
  let duck_percentage := (ducks * 100) / non_heron_birds
  duck_percentage

theorem percentage_of_ducks_among_non_heron : 
  birds_percentage 28 20 15 32 100 = 37 :=   /- 37 approximates 37.6 -/
sorry

end NUMINAMATH_GPT_percentage_of_ducks_among_non_heron_l1859_185989


namespace NUMINAMATH_GPT_triangle_perimeter_l1859_185959

/-- The lengths of two sides of a triangle are 3 and 5 respectively. The third side is a root of the equation x^2 - 7x + 12 = 0. Find the perimeter of the triangle. -/
theorem triangle_perimeter :
  let side1 := 3
  let side2 := 5
  let third_side1 := 3
  let third_side2 := 4
  (third_side1 * third_side1 - 7 * third_side1 + 12 = 0) ∧
  (third_side2 * third_side2 - 7 * third_side2 + 12 = 0) →
  (side1 + side2 + third_side1 = 11 ∨ side1 + side2 + third_side2 = 12) :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1859_185959


namespace NUMINAMATH_GPT_real_part_sum_l1859_185973

-- Definitions of a and b as real numbers and i as the imaginary unit
variables (a b : ℝ)
def i := Complex.I

-- Condition given in the problem
def given_condition : Prop := (a + b * i) / (2 - i) = 3 + i

-- Statement to prove
theorem real_part_sum : given_condition a b → a + b = 20 := by
  sorry

end NUMINAMATH_GPT_real_part_sum_l1859_185973


namespace NUMINAMATH_GPT_matching_times_l1859_185918

noncomputable def chargeAtTime (t : Nat) : ℚ :=
  100 - t / 6

def isMatchingTime (hh mm : Nat) : Prop :=
  hh * 60 + mm = 100 - (hh * 60 + mm) / 6

theorem matching_times:
  isMatchingTime 4 52 ∨
  isMatchingTime 5 43 ∨
  isMatchingTime 6 35 ∨
  isMatchingTime 7 26 ∨
  isMatchingTime 9 9 :=
by
  repeat { sorry }

end NUMINAMATH_GPT_matching_times_l1859_185918


namespace NUMINAMATH_GPT_range_of_x_for_f_lt_0_l1859_185949

noncomputable def f (x : ℝ) : ℝ := x^2 - x^(1/2)

theorem range_of_x_for_f_lt_0 :
  {x : ℝ | f x < 0} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_for_f_lt_0_l1859_185949


namespace NUMINAMATH_GPT_tammy_speed_on_second_day_l1859_185905

-- Definitions of the conditions
variables (t v : ℝ)
def total_hours := 14
def total_distance := 52

-- Distance equation
def distance_eq := v * t + (v + 0.5) * (t - 2) = total_distance

-- Time equation
def time_eq := t + (t - 2) = total_hours

theorem tammy_speed_on_second_day :
  (time_eq t ∧ distance_eq v t) → v + 0.5 = 4 :=
by sorry

end NUMINAMATH_GPT_tammy_speed_on_second_day_l1859_185905


namespace NUMINAMATH_GPT_sum_of_roots_l1859_185982

variable (x1 x2 k m : ℝ)
variable (h1 : x1 ≠ x2)
variable (h2 : 4 * x1^2 - k * x1 = m)
variable (h3 : 4 * x2^2 - k * x2 = m)

theorem sum_of_roots (x1 x2 k m : ℝ) (h1 : x1 ≠ x2)
  (h2 : 4 * x1 ^ 2 - k * x1 = m) (h3 : 4 * x2 ^ 2 - k * x2 = m) :
  x1 + x2 = k / 4 := sorry

end NUMINAMATH_GPT_sum_of_roots_l1859_185982


namespace NUMINAMATH_GPT_angle_bisector_slope_l1859_185991

/-
Given conditions:
1. line1: y = 2x
2. line2: y = 4x
Prove:
k = (sqrt(21) - 6) / 7
-/

theorem angle_bisector_slope :
  let m1 := 2
  let m2 := 4
  let k := (Real.sqrt 21 - 6) / 7
  (1 - m1 * m2) ≠ 0 →
  k = (m1 + m2 - Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)
:=
sorry

end NUMINAMATH_GPT_angle_bisector_slope_l1859_185991


namespace NUMINAMATH_GPT_value_range_a_l1859_185919

theorem value_range_a (a : ℝ) :
  (∀ (x : ℝ), |x + 2| * |x - 3| ≥ 4 / (a - 1)) ↔ (a < 1 ∨ a = 3) :=
by
  sorry

end NUMINAMATH_GPT_value_range_a_l1859_185919


namespace NUMINAMATH_GPT_complement_of_B_in_A_l1859_185992

def complement (A B : Set Int) := { x ∈ A | x ∉ B }

theorem complement_of_B_in_A (A B : Set Int) (a : Int) (h1 : A = {2, 3, 4}) (h2 : B = {a + 2, a}) (h3 : A ∩ B = B)
: complement A B = {3} :=
  sorry

end NUMINAMATH_GPT_complement_of_B_in_A_l1859_185992


namespace NUMINAMATH_GPT_trig_inequality_l1859_185924

theorem trig_inequality (theta : ℝ) (h1 : Real.pi / 4 < theta) (h2 : theta < Real.pi / 2) : 
  Real.cos theta < Real.sin theta ∧ Real.sin theta < Real.tan theta :=
sorry

end NUMINAMATH_GPT_trig_inequality_l1859_185924


namespace NUMINAMATH_GPT_sum_of_cubes_equals_square_l1859_185995

theorem sum_of_cubes_equals_square :
  1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_cubes_equals_square_l1859_185995


namespace NUMINAMATH_GPT_range_of_x_l1859_185900

theorem range_of_x 
  (x : ℝ)
  (h1 : 1 / x < 4) 
  (h2 : 1 / x > -6) 
  (h3 : x < 0) : 
  -1 / 6 < x ∧ x < 0 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_x_l1859_185900


namespace NUMINAMATH_GPT_greatest_x_lcm_l1859_185957

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end NUMINAMATH_GPT_greatest_x_lcm_l1859_185957


namespace NUMINAMATH_GPT_buying_ways_l1859_185986

theorem buying_ways (students : ℕ) (choices : ℕ) (at_least_one_pencil : ℕ) : 
  students = 4 ∧ choices = 2 ∧ at_least_one_pencil = 1 → 
  (choices^students - 1) = 15 :=
by
  sorry

end NUMINAMATH_GPT_buying_ways_l1859_185986


namespace NUMINAMATH_GPT_sum_p_q_r_l1859_185977

theorem sum_p_q_r :
  ∃ (p q r : ℤ), 
    (∀ x : ℤ, x ^ 2 + 20 * x + 96 = (x + p) * (x + q)) ∧ 
    (∀ x : ℤ, x ^ 2 - 22 * x + 120 = (x - q) * (x - r)) ∧ 
    p + q + r = 30 :=
by 
  sorry

end NUMINAMATH_GPT_sum_p_q_r_l1859_185977


namespace NUMINAMATH_GPT_calculate_weight_difference_l1859_185921

noncomputable def joe_weight := 43 -- Joe's weight in kg
noncomputable def original_avg_weight := 30 -- Original average weight in kg
noncomputable def new_avg_weight := 31 -- New average weight in kg after Joe joins
noncomputable def final_avg_weight := 30 -- Final average weight after two students leave

theorem calculate_weight_difference :
  ∃ (n : ℕ) (x : ℝ), 
  (original_avg_weight * n + joe_weight) / (n + 1) = new_avg_weight ∧
  (new_avg_weight * (n + 1) - 2 * x) / (n - 1) = final_avg_weight →
  x - joe_weight = -6.5 :=
by
  sorry

end NUMINAMATH_GPT_calculate_weight_difference_l1859_185921


namespace NUMINAMATH_GPT_quadratic_has_real_solution_l1859_185969

theorem quadratic_has_real_solution (a b c : ℝ) : 
  ∃ x : ℝ, x^2 + (a - b) * x + (b - c) = 0 ∨ 
           x^2 + (b - c) * x + (c - a) = 0 ∨ 
           x^2 + (c - a) * x + (a - b) = 0 :=
  sorry

end NUMINAMATH_GPT_quadratic_has_real_solution_l1859_185969


namespace NUMINAMATH_GPT_union_A_B_equiv_l1859_185963

def A : Set ℝ := {x : ℝ | x > 2}
def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

theorem union_A_B_equiv : A ∪ B = {x : ℝ | x ≥ 1} :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_equiv_l1859_185963


namespace NUMINAMATH_GPT_probability_two_white_balls_l1859_185980

def bagA := [1, 1]
def bagB := [2, 1]

def total_outcomes := 6
def favorable_outcomes := 2

theorem probability_two_white_balls : (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_white_balls_l1859_185980


namespace NUMINAMATH_GPT_jason_seashells_l1859_185988

theorem jason_seashells (initial_seashells : ℕ) (given_seashells : ℕ) (remaining_seashells : ℕ) 
(h1 : initial_seashells = 49) (h2 : given_seashells = 13) :
remaining_seashells = initial_seashells - given_seashells := by
  sorry

end NUMINAMATH_GPT_jason_seashells_l1859_185988


namespace NUMINAMATH_GPT_smallest_sum_a_b_l1859_185984

theorem smallest_sum_a_b (a b: ℕ) (h₀: 0 < a) (h₁: 0 < b) (h₂: a ≠ b) (h₃: 1 / (a: ℝ) + 1 / (b: ℝ) = 1 / 15) : a + b = 64 :=
sorry

end NUMINAMATH_GPT_smallest_sum_a_b_l1859_185984


namespace NUMINAMATH_GPT_parallel_line_eq_l1859_185936

theorem parallel_line_eq (a b c : ℝ) (p1 p2 : ℝ) :
  (∃ m b1 b2, 3 * a + 6 * b * p1 = 12 ∧ p2 = - (1 / 2) * p1 + b1 ∧
    - (1 / 2) * p1 - m * p1 = b2) → 
    (∃ b', p2 = - (1 / 2) * p1 + b' ∧ b' = 0) := 
sorry

end NUMINAMATH_GPT_parallel_line_eq_l1859_185936


namespace NUMINAMATH_GPT_students_chemistry_or_physics_not_both_l1859_185994

variables (total_chemistry total_both total_physics_only : ℕ)

theorem students_chemistry_or_physics_not_both
  (h1 : total_chemistry = 30)
  (h2 : total_both = 15)
  (h3 : total_physics_only = 18) :
  total_chemistry - total_both + total_physics_only = 33 :=
by
  sorry

end NUMINAMATH_GPT_students_chemistry_or_physics_not_both_l1859_185994


namespace NUMINAMATH_GPT_juice_difference_proof_l1859_185942

def barrel_initial_A := 10
def barrel_initial_B := 8
def transfer_amount := 3

def barrel_final_A := barrel_initial_A + transfer_amount
def barrel_final_B := barrel_initial_B - transfer_amount

def juice_difference := barrel_final_A - barrel_final_B

theorem juice_difference_proof : juice_difference = 8 := by
  sorry

end NUMINAMATH_GPT_juice_difference_proof_l1859_185942


namespace NUMINAMATH_GPT_depth_of_melted_ice_cream_l1859_185907

theorem depth_of_melted_ice_cream (r_sphere r_cylinder : ℝ) (Vs : ℝ) (Vc : ℝ) :
  r_sphere = 3 →
  r_cylinder = 12 →
  Vs = (4 / 3) * Real.pi * r_sphere^3 →
  Vc = Real.pi * r_cylinder^2 * (1 / 4) →
  Vs = Vc →
  (1 / 4) = 1 / 4 := 
by
  intros hr_sphere hr_cylinder hVs hVc hVs_eq_Vc
  sorry

end NUMINAMATH_GPT_depth_of_melted_ice_cream_l1859_185907


namespace NUMINAMATH_GPT_geometric_sequence_a3_q_l1859_185926

theorem geometric_sequence_a3_q (a_5 a_4 a_3 a_2 a_1 : ℝ) (q : ℝ) :
  a_5 - a_1 = 15 →
  a_4 - a_2 = 6 →
  (q = 2 ∧ a_3 = 4) ∨ (q = 1/2 ∧ a_3 = -4) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_q_l1859_185926


namespace NUMINAMATH_GPT_ordinary_eq_of_curve_l1859_185927

theorem ordinary_eq_of_curve 
  (t : ℝ) (x : ℝ) (y : ℝ)
  (ht : t > 0) 
  (hx : x = Real.sqrt t - 1 / Real.sqrt t)
  (hy : y = 3 * (t + 1 / t)) :
  3 * x^2 - y + 6 = 0 ∧ y ≥ 6 :=
sorry

end NUMINAMATH_GPT_ordinary_eq_of_curve_l1859_185927


namespace NUMINAMATH_GPT_program_output_l1859_185990

theorem program_output (x : ℤ) : 
  (if x < 0 then -1 else if x = 0 then 0 else 1) = 1 ↔ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_program_output_l1859_185990


namespace NUMINAMATH_GPT_discount_amount_l1859_185929

def tshirt_cost : ℕ := 30
def backpack_cost : ℕ := 10
def blue_cap_cost : ℕ := 5
def total_spent : ℕ := 43

theorem discount_amount : (tshirt_cost + backpack_cost + blue_cap_cost) - total_spent = 2 := by
  sorry

end NUMINAMATH_GPT_discount_amount_l1859_185929


namespace NUMINAMATH_GPT_triangle_minimum_perimeter_l1859_185902

/--
In a triangle ABC where sides have integer lengths such that no two sides are equal, let ω be a circle with its center at the incenter of ΔABC. Suppose one excircle is tangent to AB and internally tangent to ω, while excircles tangent to AC and BC are externally tangent to ω.
Prove that the minimum possible perimeter of ΔABC is 12.
-/
theorem triangle_minimum_perimeter {a b c : ℕ} (h1 : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
    (h2 : ∀ (r rA rB rC s : ℝ),
      rA = r * s / (s - a) → rB = r * s / (s - b) → rC = r * s / (s - c) →
      r + rA = rB ∧ r + rA = rC) :
  a + b + c = 12 :=
sorry

end NUMINAMATH_GPT_triangle_minimum_perimeter_l1859_185902


namespace NUMINAMATH_GPT_math_problem_l1859_185937

-- Condition 1: The solution set of the inequality \(\frac{x-2}{ax+b} > 0\) is \((-1,2)\)
def solution_set_condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, (x > -1 ∧ x < 2) ↔ ((x - 2) * (a * x + b) > 0)

-- Condition 2: \(m\) is the geometric mean of \(a\) and \(b\)
def geometric_mean_condition (a b m : ℝ) : Prop :=
  a * b = m^2

-- The mathematical statement to prove: \(\frac{3m^{2}a}{a^{3}+2b^{3}} = 1\)
theorem math_problem (a b m : ℝ) (h1 : solution_set_condition a b) (h2 : geometric_mean_condition a b m) :
  3 * m^2 * a / (a^3 + 2 * b^3) = 1 :=
sorry

end NUMINAMATH_GPT_math_problem_l1859_185937


namespace NUMINAMATH_GPT_intersection_A_B_l1859_185976

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2 * x > 0}

-- Prove the intersection of A and B
theorem intersection_A_B :
  (A ∩ B) = {x | x < (3 / 2)} := sorry

end NUMINAMATH_GPT_intersection_A_B_l1859_185976


namespace NUMINAMATH_GPT_average_licks_l1859_185908

theorem average_licks 
  (Dan_licks : ℕ := 58)
  (Michael_licks : ℕ := 63)
  (Sam_licks : ℕ := 70)
  (David_licks : ℕ := 70)
  (Lance_licks : ℕ := 39) :
  (Dan_licks + Michael_licks + Sam_licks + David_licks + Lance_licks) / 5 = 60 := 
sorry

end NUMINAMATH_GPT_average_licks_l1859_185908


namespace NUMINAMATH_GPT_Jacob_has_48_graham_crackers_l1859_185903

def marshmallows_initial := 6
def marshmallows_needed := 18
def marshmallows_total := marshmallows_initial + marshmallows_needed
def graham_crackers_per_smore := 2

def smores_total := marshmallows_total
def graham_crackers_total := smores_total * graham_crackers_per_smore

theorem Jacob_has_48_graham_crackers (h1 : marshmallows_initial = 6)
                                     (h2 : marshmallows_needed = 18)
                                     (h3 : graham_crackers_per_smore = 2)
                                     (h4 : marshmallows_total = marshmallows_initial + marshmallows_needed)
                                     (h5 : smores_total = marshmallows_total)
                                     (h6 : graham_crackers_total = smores_total * graham_crackers_per_smore) :
                                     graham_crackers_total = 48 :=
by
  sorry

end NUMINAMATH_GPT_Jacob_has_48_graham_crackers_l1859_185903


namespace NUMINAMATH_GPT_woman_work_time_l1859_185971

theorem woman_work_time :
  ∀ (M W B : ℝ), (M = 1/6) → (B = 1/12) → (M + W + B = 1/3) → (W = 1/12) → (1 / W = 12) :=
by
  intros M W B hM hB h_combined hW
  sorry

end NUMINAMATH_GPT_woman_work_time_l1859_185971


namespace NUMINAMATH_GPT_polynomial_relation_l1859_185975

def M (m : ℚ) : ℚ := 5 * m^2 - 8 * m + 1
def N (m : ℚ) : ℚ := 4 * m^2 - 8 * m - 1

theorem polynomial_relation (m : ℚ) : M m > N m := by
  sorry

end NUMINAMATH_GPT_polynomial_relation_l1859_185975


namespace NUMINAMATH_GPT_joan_gave_sam_seashells_l1859_185967

theorem joan_gave_sam_seashells (original_seashells : ℕ) (left_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 70) (h2 : left_seashells = 27) : given_seashells = 43 :=
by
  have h3 : given_seashells = original_seashells - left_seashells := sorry
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_joan_gave_sam_seashells_l1859_185967


namespace NUMINAMATH_GPT_same_terminal_side_l1859_185928

theorem same_terminal_side (α : ℝ) (k : ℤ) : 
  ∃ k : ℤ, α = k * 360 + 60 → α = -300 := 
by
  sorry

end NUMINAMATH_GPT_same_terminal_side_l1859_185928


namespace NUMINAMATH_GPT_tangent_line_eq_l1859_185920

def perp_eq (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 - 1

theorem tangent_line_eq (x y : ℝ) (h1 : perp_eq x y) (h2 : y = curve x) : 
  ∃ (m : ℝ), y = -3 * x + m ∧ y = -3 * x - 2 := 
sorry

end NUMINAMATH_GPT_tangent_line_eq_l1859_185920


namespace NUMINAMATH_GPT_geom_sequence_50th_term_l1859_185901

theorem geom_sequence_50th_term (a a_2 : ℤ) (n : ℕ) (r : ℤ) (h1 : a = 8) (h2 : a_2 = -16) (h3 : r = a_2 / a) (h4 : n = 50) :
  a * r^(n-1) = -8 * 2^49 :=
by
  sorry

end NUMINAMATH_GPT_geom_sequence_50th_term_l1859_185901


namespace NUMINAMATH_GPT_sum_of_a_b_l1859_185934

theorem sum_of_a_b (a b : ℝ) (h1 : a * b = 1) (h2 : (3 * a + 2 * b) * (3 * b + 2 * a) = 295) : a + b = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_a_b_l1859_185934


namespace NUMINAMATH_GPT_digit_possibilities_for_mod4_count_possibilities_is_3_l1859_185915

theorem digit_possibilities_for_mod4 (N : ℕ) (h : N < 10): 
  (80 + N) % 4 = 0 → N = 0 ∨ N = 4 ∨ N = 8 → true := 
by
  -- proof is not needed
  sorry

def count_possibilities : ℕ := 
  (if (80 + 0) % 4 = 0 then 1 else 0) +
  (if (80 + 1) % 4 = 0 then 1 else 0) +
  (if (80 + 2) % 4 = 0 then 1 else 0) +
  (if (80 + 3) % 4 = 0 then 1 else 0) +
  (if (80 + 4) % 4 = 0 then 1 else 0) +
  (if (80 + 5) % 4 = 0 then 1 else 0) +
  (if (80 + 6) % 4 = 0 then 1 else 0) +
  (if (80 + 7) % 4 = 0 then 1 else 0) +
  (if (80 + 8) % 4 = 0 then 1 else 0) +
  (if (80 + 9) % 4 = 0 then 1 else 0)

theorem count_possibilities_is_3: count_possibilities = 3 := 
by
  -- proof is not needed
  sorry

end NUMINAMATH_GPT_digit_possibilities_for_mod4_count_possibilities_is_3_l1859_185915


namespace NUMINAMATH_GPT_percent_of_x_is_y_l1859_185979

variable (x y : ℝ)

theorem percent_of_x_is_y
  (h : 0.5 * (x - y) = 0.2 * (x + y)) :
  y = 0.4286 * x := by
  sorry

end NUMINAMATH_GPT_percent_of_x_is_y_l1859_185979


namespace NUMINAMATH_GPT_Shinyoung_ate_most_of_cake_l1859_185952

noncomputable def Shinyoung_portion := (1 : ℚ) / 3
noncomputable def Seokgi_portion := (1 : ℚ) / 4
noncomputable def Woong_portion := (1 : ℚ) / 5

theorem Shinyoung_ate_most_of_cake :
  Shinyoung_portion > Seokgi_portion ∧ Shinyoung_portion > Woong_portion := by
  sorry

end NUMINAMATH_GPT_Shinyoung_ate_most_of_cake_l1859_185952


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1859_185922

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1859_185922


namespace NUMINAMATH_GPT_time_comparison_l1859_185974

variable (s : ℝ) (h_pos : s > 0)

noncomputable def t1 : ℝ := 120 / s
noncomputable def t2 : ℝ := 480 / (4 * s)

theorem time_comparison : t1 s = t2 s := by
  rw [t1, t2]
  field_simp [h_pos]
  norm_num
  sorry

end NUMINAMATH_GPT_time_comparison_l1859_185974


namespace NUMINAMATH_GPT_last_digit_of_N_l1859_185941

def sum_of_first_n_natural_numbers (N : ℕ) : ℕ :=
  N * (N + 1) / 2

theorem last_digit_of_N (N : ℕ) (h : sum_of_first_n_natural_numbers N = 3080) :
  N % 10 = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_last_digit_of_N_l1859_185941


namespace NUMINAMATH_GPT_value_of_sum_cubes_l1859_185909

theorem value_of_sum_cubes (x : ℝ) (hx : x ≠ 0) (h : 47 = x^6 + (1 / x^6)) : (x^3 + (1 / x^3)) = 7 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_sum_cubes_l1859_185909


namespace NUMINAMATH_GPT_trajectory_equation_find_m_value_l1859_185939

def point (α : Type) := (α × α)
def fixed_points (α : Type) := point α

noncomputable def slopes (x y : ℝ) : ℝ := y / x

theorem trajectory_equation (x y : ℝ) (P : point ℝ) (A B : fixed_points ℝ)
  (k1 k2 : ℝ) (hk : k1 * k2 = -1/4) :
  A = (-2, 0) → B = (2, 0) →
  P = (x, y) → 
  slopes (x + 2) y * slopes (x - 2) y = -1/4 →
  (x^2 / 4) + y^2 = 1 :=
sorry

theorem find_m_value (m x₁ x₂ y₁ y₂ : ℝ) (k : ℝ) (hx : (4 * k^2) + 1 - m^2 > 0)
  (hroots_sum : x₁ + x₂ = -((8 * k * m) / ((4 * k^2) + 1)))
  (hroots_prod : x₁ * x₂ = (4 * m^2 - 4) / ((4 * k^2) + 1))
  (hperp : x₁ * x₂ + y₁ * y₂ = 0) :
  y₁ = k * x₁ + m → y₂ = k * x₂ + m →
  m^2 = 4/5 * (k^2 + 1) →
  m = 2 ∨ m = -2 :=
sorry

end NUMINAMATH_GPT_trajectory_equation_find_m_value_l1859_185939


namespace NUMINAMATH_GPT_odd_function_k_eq_neg_one_f_x_greater_2_neg_x_k_gt_zero_l1859_185955

noncomputable def f (x : ℝ) (k : ℝ) := 2^x + k * 2^(-x)

-- Prove that if f(x) is an odd function, then k = -1.
theorem odd_function_k_eq_neg_one {k : ℝ} (h : ∀ x, f x k = -f (-x) k) : k = -1 :=
by sorry

-- Prove that if for all x in [0, +∞), f(x) > 2^(-x), then k > 0.
theorem f_x_greater_2_neg_x_k_gt_zero {k : ℝ} (h : ∀ x, 0 ≤ x → f x k > 2^(-x)) : k > 0 :=
by sorry

end NUMINAMATH_GPT_odd_function_k_eq_neg_one_f_x_greater_2_neg_x_k_gt_zero_l1859_185955


namespace NUMINAMATH_GPT_find_a_given_solution_l1859_185954

theorem find_a_given_solution (a : ℝ) (x : ℝ) (h : x = 1) (eqn : a * (x + 1) = 2 * (2 * x - a)) : a = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_a_given_solution_l1859_185954


namespace NUMINAMATH_GPT_count_three_digit_numbers_with_identical_digits_l1859_185951

/-!
# Problem Statement:
Prove that the number of three-digit numbers with at least two identical digits is 252,
given that three-digit numbers range from 100 to 999.

## Definitions:
- Three-digit numbers are those in the range 100 to 999.

## Theorem:
The number of three-digit numbers with at least two identical digits is 252.
-/
theorem count_three_digit_numbers_with_identical_digits : 
    (∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ 
    ∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ (d1 = d2 ∨ d1 = d3 ∨ d2 = d3)) :=
sorry

end NUMINAMATH_GPT_count_three_digit_numbers_with_identical_digits_l1859_185951


namespace NUMINAMATH_GPT_find_number_l1859_185916

theorem find_number (n : ℤ) 
  (h : (69842 * 69842 - n * n) / (69842 - n) = 100000) : 
  n = 30158 :=
sorry

end NUMINAMATH_GPT_find_number_l1859_185916


namespace NUMINAMATH_GPT_min_value_x_plus_9_div_x_l1859_185985

theorem min_value_x_plus_9_div_x (x : ℝ) (hx : x > 0) : x + 9 / x ≥ 6 := by
  -- sorry indicates that the proof is omitted.
  sorry

end NUMINAMATH_GPT_min_value_x_plus_9_div_x_l1859_185985


namespace NUMINAMATH_GPT_part1_part2_l1859_185913

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x - m|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * f x m - f (x + m) m

theorem part1 (h : ∀ x, g x m ≥ -1) : m = 1 :=
  sorry

theorem part2 {a b m : ℝ} (ha : |a| < m) (hb : |b| < m) (a_ne_zero : a ≠ 0) (hm: m = 1) : 
  f (a * b) m > |a| * f (b / a) m :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1859_185913


namespace NUMINAMATH_GPT_parallelogram_properties_l1859_185961

noncomputable def length_adjacent_side_and_area (base height : ℝ) (angle : ℕ) : ℝ × ℝ :=
  let hypotenuse := height / Real.sin (angle * Real.pi / 180)
  let area := base * height
  (hypotenuse, area)

theorem parallelogram_properties :
  ∀ (base height : ℝ) (angle : ℕ),
  base = 12 → height = 6 → angle = 30 →
  length_adjacent_side_and_area base height angle = (12, 72) :=
by
  intros
  sorry

end NUMINAMATH_GPT_parallelogram_properties_l1859_185961


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l1859_185906

noncomputable def atomic_weight_carbon : ℝ := 12.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.008
noncomputable def atomic_weight_oxygen : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def num_oxygen_atoms : ℕ := 1

noncomputable def molecular_weight (num_C num_H num_O : ℕ) : ℝ :=
  (num_C * atomic_weight_carbon) + (num_H * atomic_weight_hydrogen) + (num_O * atomic_weight_oxygen)

theorem molecular_weight_of_compound :
  molecular_weight num_carbon_atoms num_hydrogen_atoms num_oxygen_atoms = 65.048 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l1859_185906


namespace NUMINAMATH_GPT_mixture_cost_in_july_l1859_185958

theorem mixture_cost_in_july :
  (∀ C : ℝ, C > 0 → 
    (cost_green_tea_july : ℝ) = 0.1 → 
    (cost_green_tea_july = 0.1 * C) →
    (equal_quantities_mixture:  ℝ) = 1.5 →
    (cost_coffee_july: ℝ) = 2 * C →
    (total_mixture_cost: ℝ) = equal_quantities_mixture * cost_green_tea_july + equal_quantities_mixture * cost_coffee_july →
    total_mixture_cost = 3.15) :=
by
  sorry

end NUMINAMATH_GPT_mixture_cost_in_july_l1859_185958


namespace NUMINAMATH_GPT_permutation_problem_l1859_185997

noncomputable def permutation (n r : ℕ) : ℕ := (n.factorial) / ( (n - r).factorial)

theorem permutation_problem : 5 * permutation 5 3 + 4 * permutation 4 2 = 348 := by
  sorry

end NUMINAMATH_GPT_permutation_problem_l1859_185997


namespace NUMINAMATH_GPT_part1_part2_l1859_185999

-- Problem Part 1
theorem part1 : (-((-8)^(1/3)) - |(3^(1/2) - 2)| + ((-3)^2)^(1/2) + -3^(1/2) = 3) :=
by {
  sorry
}

-- Problem Part 2
theorem part2 (x : ℤ) : (2 * x + 5 ≤ 3 * (x + 2) ∧ 2 * x - (1 + 3 * x) / 2 < 1) ↔ (x = 0 ∨ x = 1 ∨ x = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_part2_l1859_185999


namespace NUMINAMATH_GPT_round_to_nearest_tenth_l1859_185945

theorem round_to_nearest_tenth : 
  let x := 36.89753 
  let tenth_place := 8
  let hundredth_place := 9
  (hundredth_place > 5) → (Float.round (10 * x) / 10 = 36.9) := 
by
  intros x tenth_place hundredth_place h
  sorry

end NUMINAMATH_GPT_round_to_nearest_tenth_l1859_185945


namespace NUMINAMATH_GPT_greatest_divisor_of_arithmetic_sequence_l1859_185965

theorem greatest_divisor_of_arithmetic_sequence (x c : ℕ) : ∃ d, d = 15 ∧ ∀ S, S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_arithmetic_sequence_l1859_185965


namespace NUMINAMATH_GPT_average_marks_l1859_185925

variable (M P C : ℤ)

-- Conditions
axiom h1 : M + P = 50
axiom h2 : C = P + 20

-- Theorem statement
theorem average_marks : (M + C) / 2 = 35 := by
  sorry

end NUMINAMATH_GPT_average_marks_l1859_185925


namespace NUMINAMATH_GPT_melon_weights_l1859_185940

-- We start by defining the weights of the individual melons.
variables {D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 : ℝ}

-- Define the weights of the given sets of three melons.
def W1 := D1 + D2 + D3
def W2 := D2 + D3 + D4
def W3 := D1 + D3 + D4
def W4 := D1 + D2 + D4
def W5 := D5 + D6 + D7
def W6 := D8 + D9 + D10

-- State the theorem to be proven.
theorem melon_weights (W1 W2 W3 W4 W5 W6 : ℝ) :
  (W1 + W2 + W3 + W4) / 3 + W5 + W6 = D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 :=
sorry 

end NUMINAMATH_GPT_melon_weights_l1859_185940


namespace NUMINAMATH_GPT_find_PS_l1859_185911

theorem find_PS 
    (P Q R S : Type)
    (PQ PR : ℝ)
    (h : ℝ) 
    (ratio_QS_SR : ℝ)
    (hyp1 : PQ = 13)
    (hyp2 : PR = 20)
    (hyp3 : ratio_QS_SR = 3/7) :
    h = Real.sqrt (117.025) :=
by
  -- Proof steps would go here, but we are just stating the theorem
  sorry

end NUMINAMATH_GPT_find_PS_l1859_185911


namespace NUMINAMATH_GPT_ratio_of_thermometers_to_hotwater_bottles_l1859_185917

theorem ratio_of_thermometers_to_hotwater_bottles (T H : ℕ) (thermometer_price hotwater_bottle_price total_sales : ℕ) 
  (h1 : thermometer_price = 2) (h2 : hotwater_bottle_price = 6) (h3 : total_sales = 1200) (h4 : H = 60) 
  (h5 : total_sales = thermometer_price * T + hotwater_bottle_price * H) : 
  T / H = 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_thermometers_to_hotwater_bottles_l1859_185917


namespace NUMINAMATH_GPT_subtract_decimal_numbers_l1859_185914

theorem subtract_decimal_numbers : 3.75 - 1.46 = 2.29 := by
  sorry

end NUMINAMATH_GPT_subtract_decimal_numbers_l1859_185914


namespace NUMINAMATH_GPT_existence_of_b_l1859_185962

theorem existence_of_b's (n m : ℕ) (h1 : 1 < n) (h2 : 1 < m) 
  (a : Fin m → ℕ) (h3 : ∀ i, 0 < a i ∧ a i ≤ n^m) :
  ∃ b : Fin m → ℕ, (∀ i, 0 < b i ∧ b i ≤ n) ∧ (∀ i, a i + b i < n) :=
by
  sorry

end NUMINAMATH_GPT_existence_of_b_l1859_185962


namespace NUMINAMATH_GPT_last_number_remaining_l1859_185972

theorem last_number_remaining :
  (∃ f : ℕ → ℕ, ∃ n : ℕ, (∀ k < n, f (2 * k) = 2 * k + 2 ∧
                         ∀ k < n, f (2 * k + 1) = 2 * k + 1 + 2^(k+1)) ∧ 
                         n = 200 ∧ f (2 * n) = 128) :=
sorry

end NUMINAMATH_GPT_last_number_remaining_l1859_185972


namespace NUMINAMATH_GPT_hot_dogs_leftover_l1859_185998

theorem hot_dogs_leftover :
  36159782 % 6 = 2 :=
by
  sorry

end NUMINAMATH_GPT_hot_dogs_leftover_l1859_185998


namespace NUMINAMATH_GPT_central_angle_of_sector_l1859_185968

theorem central_angle_of_sector (r : ℝ) (θ : ℝ) (h_perimeter: 2 * r + θ * r = π * r / 2) : θ = π - 2 :=
sorry

end NUMINAMATH_GPT_central_angle_of_sector_l1859_185968


namespace NUMINAMATH_GPT_find_xyz_l1859_185953

theorem find_xyz
  (x y z : ℝ)
  (h1 : x + y + z = 38)
  (h2 : x * y * z = 2002)
  (h3 : 0 < x ∧ x ≤ 11)
  (h4 : z ≥ 14) :
  x = 11 ∧ y = 13 ∧ z = 14 :=
sorry

end NUMINAMATH_GPT_find_xyz_l1859_185953


namespace NUMINAMATH_GPT_buckets_required_l1859_185964

variable (C : ℝ) (N : ℝ)

theorem buckets_required (h : N * C = 105 * (2 / 5) * C) : N = 42 := 
  sorry

end NUMINAMATH_GPT_buckets_required_l1859_185964


namespace NUMINAMATH_GPT_jason_needs_201_grams_l1859_185923

-- Define the conditions
def rectangular_patch_length : ℕ := 6
def rectangular_patch_width : ℕ := 7
def square_path_side_length : ℕ := 5
def sand_per_square_inch : ℕ := 3

-- Define the areas
def rectangular_patch_area : ℕ := rectangular_patch_length * rectangular_patch_width
def square_path_area : ℕ := square_path_side_length * square_path_side_length

-- Define the total area
def total_area : ℕ := rectangular_patch_area + square_path_area

-- Define the total sand needed
def total_sand_needed : ℕ := total_area * sand_per_square_inch

-- State the proof problem
theorem jason_needs_201_grams : total_sand_needed = 201 := by
    sorry

end NUMINAMATH_GPT_jason_needs_201_grams_l1859_185923


namespace NUMINAMATH_GPT_dad_borrowed_nickels_l1859_185987

-- Definitions for the initial and remaining nickels
def initial_nickels : ℕ := 31
def remaining_nickels : ℕ := 11

-- Statement of the problem in Lean
theorem dad_borrowed_nickels : initial_nickels - remaining_nickels = 20 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_dad_borrowed_nickels_l1859_185987


namespace NUMINAMATH_GPT_find_150th_letter_l1859_185981

theorem find_150th_letter (n : ℕ) (pattern : ℕ → Char) (h : ∀ m, pattern (m % 3) = if m % 3 = 0 then 'A' else if m % 3 = 1 then 'B' else 'C') :
  pattern 149 = 'C' :=
by
  sorry

end NUMINAMATH_GPT_find_150th_letter_l1859_185981


namespace NUMINAMATH_GPT_probability_not_late_probability_late_and_misses_bus_l1859_185950

variable (P_Sam_late : ℚ)
variable (P_miss_bus_given_late : ℚ)

theorem probability_not_late (h1 : P_Sam_late = 5/9) :
  1 - P_Sam_late = 4/9 := by
  rw [h1]
  norm_num

theorem probability_late_and_misses_bus (h1 : P_Sam_late = 5/9) (h2 : P_miss_bus_given_late = 1/3) :
  P_Sam_late * P_miss_bus_given_late = 5/27 := by
  rw [h1, h2]
  norm_num

#check probability_not_late
#check probability_late_and_misses_bus

end NUMINAMATH_GPT_probability_not_late_probability_late_and_misses_bus_l1859_185950


namespace NUMINAMATH_GPT_det_M_pow_three_eq_twenty_seven_l1859_185960

-- Define a matrix M
variables (M : Matrix (Fin n) (Fin n) ℝ)

-- Given condition: det M = 3
axiom det_M_eq_3 : Matrix.det M = 3

-- State the theorem we aim to prove
theorem det_M_pow_three_eq_twenty_seven : Matrix.det (M^3) = 27 :=
by
  sorry

end NUMINAMATH_GPT_det_M_pow_three_eq_twenty_seven_l1859_185960


namespace NUMINAMATH_GPT_max_min_y_l1859_185996

noncomputable def y (x : ℝ) : ℝ := (Real.sin x)^(2:ℝ) + 2 * (Real.sin x) * (Real.cos x) + 3 * (Real.cos x)^(2:ℝ)

theorem max_min_y : 
  ∀ x : ℝ, 
  2 - Real.sqrt 2 ≤ y x ∧ y x ≤ 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_max_min_y_l1859_185996


namespace NUMINAMATH_GPT_distinct_integers_integer_expression_l1859_185948

theorem distinct_integers_integer_expression 
  (x y z : ℤ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (n : ℕ) : 
  ∃ k : ℤ, k = (x^n / ((x - y) * (x - z)) + y^n / ((y - x) * (y - z)) + z^n / ((z - x) * (z - y))) := 
sorry

end NUMINAMATH_GPT_distinct_integers_integer_expression_l1859_185948
