import Mathlib

namespace average_age_of_John_Mary_Tonya_is_35_l40_4023

-- Define the ages of the individuals
variable (John Mary Tonya : ℕ)

-- Conditions given in the problem
def John_is_twice_as_old_as_Mary : Prop := John = 2 * Mary
def John_is_half_as_old_as_Tonya : Prop := John = Tonya / 2
def Tonya_is_60 : Prop := Tonya = 60

-- The average age calculation
def average_age (a b c : ℕ) : ℕ := (a + b + c) / 3

-- The statement we need to prove
theorem average_age_of_John_Mary_Tonya_is_35 :
  John_is_twice_as_old_as_Mary John Mary →
  John_is_half_as_old_as_Tonya John Tonya →
  Tonya_is_60 Tonya →
  average_age John Mary Tonya = 35 :=
by
  sorry

end average_age_of_John_Mary_Tonya_is_35_l40_4023


namespace graph_of_conic_section_is_straight_lines_l40_4093

variable {x y : ℝ}

theorem graph_of_conic_section_is_straight_lines:
  (x^2 - 9 * y^2 = 0) ↔ (x = 3 * y ∨ x = -3 * y) := by
  sorry

end graph_of_conic_section_is_straight_lines_l40_4093


namespace difference_in_ages_is_54_l40_4002

theorem difference_in_ages_is_54 (c d : ℕ) (h1 : 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100) 
    (h2 : 10 * c + d - (10 * d + c) = 9 * (c - d)) 
    (h3 : 10 * c + d + 10 = 3 * (10 * d + c + 10)) : 
    10 * c + d - (10 * d + c) = 54 :=
by
sorry

end difference_in_ages_is_54_l40_4002


namespace unique_function_solution_l40_4099

theorem unique_function_solution :
  ∀ f : ℕ+ → ℕ+, (∀ x y : ℕ+, f (x + y * f x) = x * f (y + 1)) → (∀ x : ℕ+, f x = x) :=
by
  sorry

end unique_function_solution_l40_4099


namespace regular_decagon_triangle_probability_l40_4006

theorem regular_decagon_triangle_probability :
  let total_triangles := Nat.choose 10 3
  let favorable_triangles := 10
  let probability := favorable_triangles / total_triangles
  probability = (1 : ℚ) / 12 :=
by
  sorry

end regular_decagon_triangle_probability_l40_4006


namespace sin_A_is_eight_ninths_l40_4032

variable (AB AC : ℝ) (A : ℝ)

-- Given conditions
def area_triangle := 1 / 2 * AB * AC * Real.sin A = 100
def geometric_mean := Real.sqrt (AB * AC) = 15

-- Proof statement
theorem sin_A_is_eight_ninths (h1 : area_triangle AB AC A) (h2 : geometric_mean AB AC) :
  Real.sin A = 8 / 9 := sorry

end sin_A_is_eight_ninths_l40_4032


namespace math_problem_l40_4073

theorem math_problem (a b c k : ℝ) (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h2 : a + b + c = 0) (h3 : a^2 = k * b^2) (hk : k ≠ 0) :
  (a^2 * b^2) / ((a^2 - b * c) * (b^2 - a * c)) + (a^2 * c^2) / ((a^2 - b * c) * (c^2 - a * b)) + (b^2 * c^2) / ((b^2 - a * c) * (c^2 - a * b)) = 1 :=
by
  sorry

end math_problem_l40_4073


namespace flammable_ice_storage_capacity_l40_4035

theorem flammable_ice_storage_capacity (billion : ℕ) (h : billion = 10^9) : (800 * billion = 8 * 10^11) :=
by
  sorry

end flammable_ice_storage_capacity_l40_4035


namespace shara_monthly_payment_l40_4037

theorem shara_monthly_payment : 
  ∀ (T M : ℕ), 
  (T / 2 = 6 * M) → 
  (T / 2 - 4 * M = 20) → 
  M = 10 :=
by
  intros T M h1 h2
  sorry

end shara_monthly_payment_l40_4037


namespace device_failure_probability_l40_4088

noncomputable def probability_fail_device (p1 p2 p3 : ℝ) (p_one p_two p_three : ℝ) : ℝ :=
  0.006 * p3 + 0.092 * p_two + 0.398 * p_one

theorem device_failure_probability
  (p1 p2 p3 : ℝ) (p_one p_two p_three : ℝ)
  (h1 : p1 = 0.1)
  (h2 : p2 = 0.2)
  (h3 : p3 = 0.3)
  (h4 : p_one = 0.25)
  (h5 : p_two = 0.6)
  (h6 : p_three = 0.9) :
  probability_fail_device p1 p2 p3 p_one p_two p_three = 0.1601 :=
by
  sorry

end device_failure_probability_l40_4088


namespace pyramid_height_is_6_l40_4003

-- Define the conditions for the problem
def square_side_length : ℝ := 18
def pyramid_base_side_length (s : ℝ) : Prop := s * s = (square_side_length / 2) * (square_side_length / 2)
def pyramid_slant_height (s l : ℝ) : Prop := 2 * s * l = square_side_length * square_side_length

-- State the main theorem
theorem pyramid_height_is_6 (s l h : ℝ) (hs : pyramid_base_side_length s) (hl : pyramid_slant_height s l) : h = 6 := 
sorry

end pyramid_height_is_6_l40_4003


namespace harmonic_sum_base_case_l40_4005

theorem harmonic_sum_base_case : 1 + 1/2 + 1/3 < 2 := 
sorry

end harmonic_sum_base_case_l40_4005


namespace max_area_of_garden_l40_4061

theorem max_area_of_garden (p : ℝ) (h : p = 36) : 
  ∃ A : ℝ, (∀ l w : ℝ, l + l + w + w = p → l * w ≤ A) ∧ A = 81 :=
by
  sorry

end max_area_of_garden_l40_4061


namespace last_digit_2_to_2010_l40_4098

theorem last_digit_2_to_2010 : (2 ^ 2010) % 10 = 4 := 
by
  -- proofs and lemmas go here
  sorry

end last_digit_2_to_2010_l40_4098


namespace length_increase_100_l40_4075

theorem length_increase_100 (n : ℕ) (h : (n + 2) / 2 = 100) : n = 198 :=
sorry

end length_increase_100_l40_4075


namespace solve_y_l40_4096

theorem solve_y (y : ℚ) (h : (3 * y) / 7 = 14) : y = 98 / 3 := 
by sorry

end solve_y_l40_4096


namespace minimum_ab_bc_ca_l40_4056

theorem minimum_ab_bc_ca {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = a^3) (h5 : a * b * c = a^3) : 
  ab + bc + ca ≥ 9 :=
sorry

end minimum_ab_bc_ca_l40_4056


namespace line_AC_eqn_l40_4082

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

-- Define point A
def A : Point := { x := 3, y := 1 }

-- Define point B
def B : Point := { x := -1, y := 2 }

-- Define the line equation y = x + 1
def line_eq (p : Point) : Prop := p.y = p.x + 1

-- Define the bisector being on line y=x+1 as a condition
axiom bisector_on_line (C : Point) : 
  line_eq C → (∃ k : ℝ, (C.y - B.y) = k * (C.x - B.x))

-- Define the final goal to prove the equation of line AC
theorem line_AC_eqn (C : Point) :
  line_eq C → ((A.x - C.x) * (B.y - C.y) = (B.x - C.x) * (A.y - C.y)) → C.x = -3 ∧ C.y = -2 → 
  (A.x - 2 * A.y = 1) := sorry

end line_AC_eqn_l40_4082


namespace trapezoid_proof_l40_4010

variables {Point : Type} [MetricSpace Point]

-- Definitions of the points and segments as given conditions.
variables (A B C D E : Point)

-- Definitions representing the trapezoid and point E's property.
def is_trapezoid (ABCD : (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)) : Prop :=
  (A ≠ B) ∧ (C ≠ D)

def on_segment (E : Point) (A D : Point) : Prop :=
  -- This definition will encompass the fact that E is on segment AD.
  -- Representing the notion that E lies between A and D.
  dist A E + dist E D = dist A D

def equal_perimeters (E : Point) (A B C D : Point) : Prop :=
  let p1 := (dist A B + dist B E + dist E A)
  let p2 := (dist B C + dist C E + dist E B)
  let p3 := (dist C D + dist D E + dist E C)
  p1 = p2 ∧ p2 = p3

-- The theorem we need to prove.
theorem trapezoid_proof (ABCD : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) (onSeg : on_segment E A D) (eqPerim : equal_perimeters E A B C D) : 
  dist B C = dist A D / 2 :=
sorry

end trapezoid_proof_l40_4010


namespace cone_base_area_l40_4045

theorem cone_base_area (r l : ℝ) (h1 : (1/2) * π * l^2 = 2 * π) (h2 : 2 * π * r = 2 * π) :
  π * r^2 = π :=
by 
  sorry

end cone_base_area_l40_4045


namespace number_of_friends_gave_money_l40_4007

-- Definition of given data in conditions
def amount_per_friend : ℕ := 6
def total_amount : ℕ := 30

-- Theorem to be proved
theorem number_of_friends_gave_money : total_amount / amount_per_friend = 5 :=
by
  sorry

end number_of_friends_gave_money_l40_4007


namespace calculate_expression_l40_4080

theorem calculate_expression (x : ℕ) (h : x = 3) : x + x * x^(x - 1) = 30 := by
  rw [h]
  -- Proof steps would go here but we are including only the statement
  sorry

end calculate_expression_l40_4080


namespace base_conversion_and_addition_l40_4072

theorem base_conversion_and_addition :
  let a₈ : ℕ := 3 * 8^2 + 5 * 8^1 + 6 * 8^0
  let c₁₄ : ℕ := 4 * 14^2 + 12 * 14^1 + 3 * 14^0
  a₈ + c₁₄ = 1193 :=
by
  sorry

end base_conversion_and_addition_l40_4072


namespace floor_x_mul_x_eq_54_l40_4051

def positive_real (x : ℝ) : Prop := x > 0

theorem floor_x_mul_x_eq_54 (x : ℝ) (h_pos : positive_real x) : ⌊x⌋ * x = 54 ↔ x = 54 / 7 :=
by
  sorry

end floor_x_mul_x_eq_54_l40_4051


namespace simplify_expression_frac_l40_4053

theorem simplify_expression_frac (a b k : ℤ) (h : (6*k + 12) / 6 = a * k + b) : a = 1 ∧ b = 2 → a / b = 1 / 2 := by
  sorry

end simplify_expression_frac_l40_4053


namespace num_passed_candidates_l40_4034

theorem num_passed_candidates
  (total_candidates : ℕ)
  (avg_passed_marks : ℕ)
  (avg_failed_marks : ℕ)
  (overall_avg_marks : ℕ)
  (h1 : total_candidates = 120)
  (h2 : avg_passed_marks = 39)
  (h3 : avg_failed_marks = 15)
  (h4 : overall_avg_marks = 35) :
  ∃ (P : ℕ), P = 100 :=
by
  sorry

end num_passed_candidates_l40_4034


namespace triangle_third_side_one_third_perimeter_l40_4042

theorem triangle_third_side_one_third_perimeter
  (a b x y p c : ℝ)
  (h1 : x^2 - y^2 = a^2 - b^2)
  (h2 : p = (a + b + c) / 2)
  (h3 : x - y = 2 * (a - b)) :
  c = (a + b + c) / 3 := by
  sorry

end triangle_third_side_one_third_perimeter_l40_4042


namespace convert_decimal_to_vulgar_fraction_l40_4011

theorem convert_decimal_to_vulgar_fraction : (32 : ℝ) / 100 = (8 : ℝ) / 25 :=
by
  sorry

end convert_decimal_to_vulgar_fraction_l40_4011


namespace total_snakes_owned_l40_4041

theorem total_snakes_owned 
  (total_people : ℕ)
  (only_dogs only_cats only_birds only_snakes : ℕ)
  (cats_and_dogs birds_and_dogs birds_and_cats snakes_and_dogs snakes_and_cats snakes_and_birds : ℕ)
  (cats_dogs_snakes cats_dogs_birds cats_birds_snakes dogs_birds_snakes all_four_pets : ℕ)
  (h1 : total_people = 150)
  (h2 : only_dogs = 30)
  (h3 : only_cats = 25)
  (h4 : only_birds = 10)
  (h5 : only_snakes = 7)
  (h6 : cats_and_dogs = 15)
  (h7 : birds_and_dogs = 12)
  (h8 : birds_and_cats = 8)
  (h9 : snakes_and_dogs = 3)
  (h10 : snakes_and_cats = 4)
  (h11 : snakes_and_birds = 2)
  (h12 : cats_dogs_snakes = 5)
  (h13 : cats_dogs_birds = 4)
  (h14 : cats_birds_snakes = 6)
  (h15 : dogs_birds_snakes = 9)
  (h16 : all_four_pets = 10) : 
  7 + 3 + 4 + 2 + 5 + 6 + 9 + 10 = 46 := 
sorry

end total_snakes_owned_l40_4041


namespace find_f_at_3_l40_4047

noncomputable def f (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * (x^32 + 1) - 1) / (x^(2^6 - 1) - 1)

theorem find_f_at_3 : f 3 = 3 :=
by
  sorry

end find_f_at_3_l40_4047


namespace find_ages_l40_4026

theorem find_ages (J sister cousin : ℝ)
  (h1 : J + 9 = 3 * (J - 11))
  (h2 : sister = 2 * J)
  (h3 : cousin = (J + sister) / 2) :
  J = 21 ∧ sister = 42 ∧ cousin = 31.5 :=
by
  sorry

end find_ages_l40_4026


namespace intersection_M_N_l40_4063

-- Define sets M and N
def M := { x : ℝ | ∃ t : ℝ, x = 2^(-t) }
def N := { y : ℝ | ∃ x : ℝ, y = Real.sin x }

-- Prove the intersection of M and N
theorem intersection_M_N : M ∩ N = { y : ℝ | 0 < y ∧ y ≤ 1 } :=
by sorry

end intersection_M_N_l40_4063


namespace value_range_of_log_function_l40_4015

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 2*x + 4

noncomputable def log_base_3 (x : ℝ) : ℝ :=
  Real.log x / Real.log 3

theorem value_range_of_log_function :
  ∀ x : ℝ, log_base_3 (quadratic_function x) ≥ 1 := by
  sorry

end value_range_of_log_function_l40_4015


namespace image_of_3_5_pre_image_of_3_5_l40_4087

def f (x y : ℤ) : ℤ × ℤ := (x - y, x + y)

theorem image_of_3_5 : f 3 5 = (-2, 8) :=
by
  sorry

theorem pre_image_of_3_5 : ∃ (x y : ℤ), f x y = (3, 5) ∧ x = 4 ∧ y = 1 :=
by
  sorry

end image_of_3_5_pre_image_of_3_5_l40_4087


namespace valid_lineups_count_l40_4046

-- Definitions of the problem conditions
def num_players : ℕ := 18
def quadruplets : Finset ℕ := {0, 1, 2, 3} -- Indices of Benjamin, Brenda, Brittany, Bryan
def total_starters : ℕ := 8

-- Function to count lineups based on given constraints
noncomputable def count_valid_lineups : ℕ :=
  let others := num_players - quadruplets.card
  Nat.choose others total_starters + quadruplets.card * Nat.choose others (total_starters - 1)

-- The theorem to prove the count of valid lineups
theorem valid_lineups_count : count_valid_lineups = 16731 := by
  -- Placeholder for the actual proof
  sorry

end valid_lineups_count_l40_4046


namespace decreasing_interval_f_l40_4065

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 4 * x - 3

-- Statement to prove that the interval where f is monotonically decreasing is [2, +∞)
theorem decreasing_interval_f : (∀ x₁ x₂ : ℝ, 2 ≤ x₁ ∧ x₁ ≤ x₂ → f x₁ ≥ f x₂) :=
by
  sorry

end decreasing_interval_f_l40_4065


namespace xiao_ming_total_score_l40_4029

-- Definitions for the given conditions
def score_regular : ℝ := 70
def score_midterm : ℝ := 80
def score_final : ℝ := 85

def weight_regular : ℝ := 0.3
def weight_midterm : ℝ := 0.3
def weight_final : ℝ := 0.4

-- The statement that we need to prove
theorem xiao_ming_total_score : 
  (score_regular * weight_regular) + (score_midterm * weight_midterm) + (score_final * weight_final) = 79 := 
by
  sorry

end xiao_ming_total_score_l40_4029


namespace sqrt_product_l40_4077

theorem sqrt_product : (Real.sqrt 121) * (Real.sqrt 49) * (Real.sqrt 11) = 77 * (Real.sqrt 11) := by
  -- This is just the theorem statement as requested.
  sorry

end sqrt_product_l40_4077


namespace sufficient_but_not_necessary_condition_l40_4020

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (1 ≤ x ∧ x ≤ 4) ↔ (1 ≤ x^2 ∧ x^2 ≤ 16) :=
by
  sorry

end sufficient_but_not_necessary_condition_l40_4020


namespace exist_circle_tangent_to_three_circles_l40_4069

variable (h1 k1 r1 h2 k2 r2 h3 k3 r3 h k r : ℝ)

def condition1 : Prop := (h - h1)^2 + (k - k1)^2 = (r + r1)^2
def condition2 : Prop := (h - h2)^2 + (k - k2)^2 = (r + r2)^2
def condition3 : Prop := (h - h3)^2 + (k - k3)^2 = (r + r3)^2

theorem exist_circle_tangent_to_three_circles : 
  ∃ (h k r : ℝ), condition1 h1 k1 r1 h k r ∧ condition2 h2 k2 r2 h k r ∧ condition3 h3 k3 r3 h k r :=
by
  sorry

end exist_circle_tangent_to_three_circles_l40_4069


namespace heptagon_diagonals_l40_4040

-- Define the number of sides of the polygon
def heptagon_sides : ℕ := 7

-- Define the formula for the number of diagonals of an n-gon
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- State the theorem we want to prove, i.e., the number of diagonals in a convex heptagon is 14
theorem heptagon_diagonals : diagonals heptagon_sides = 14 := by
  sorry

end heptagon_diagonals_l40_4040


namespace min_value_of_z_l40_4090

theorem min_value_of_z : ∀ (x : ℝ), ∃ z : ℝ, z = 5 * x^2 - 20 * x + 45 ∧ z ≥ 25 :=
by sorry

end min_value_of_z_l40_4090


namespace hammer_nail_cost_l40_4070

variable (h n : ℝ)

theorem hammer_nail_cost (h n : ℝ)
    (h1 : 4 * h + 5 * n = 10.45)
    (h2 : 3 * h + 9 * n = 12.87) :
  20 * h + 25 * n = 52.25 :=
sorry

end hammer_nail_cost_l40_4070


namespace incorrect_statement_C_l40_4076

theorem incorrect_statement_C (x : ℝ) (h : x > -2) : (6 / x) > -3 :=
sorry

end incorrect_statement_C_l40_4076


namespace largest_n_unique_k_l40_4049

theorem largest_n_unique_k : 
  ∃ n : ℕ, (∀ k : ℤ, (5 / 12 : ℚ) < n / (n + k) ∧ n / (n + k) < (4 / 9 : ℚ) → k = 9) ∧ n = 7 :=
by
  sorry

end largest_n_unique_k_l40_4049


namespace max_bk_at_k_l40_4081
open Nat Real

theorem max_bk_at_k :
  let B_k (k : ℕ) := (choose 2000 k) * (0.1 : ℝ) ^ k
  ∃ k : ℕ, (k = 181) ∧ (∀ m : ℕ, B_k m ≤ B_k k) :=
sorry

end max_bk_at_k_l40_4081


namespace satisfactory_fraction_l40_4017

theorem satisfactory_fraction :
  let num_students_A := 8
  let num_students_B := 7
  let num_students_C := 6
  let num_students_D := 5
  let num_students_F := 4
  let satisfactory_grades := num_students_A + num_students_B + num_students_C
  let total_students := num_students_A + num_students_B + num_students_C + num_students_D + num_students_F
  satisfactory_grades / total_students = 7 / 10 :=
by
  let num_students_A := 8
  let num_students_B := 7
  let num_students_C := 6
  let num_students_D := 5
  let num_students_F := 4
  let satisfactory_grades := num_students_A + num_students_B + num_students_C
  let total_students := num_students_A + num_students_B + num_students_C + num_students_D + num_students_F
  have h1: satisfactory_grades = 21 := by sorry
  have h2: total_students = 30 := by sorry
  have fraction := (satisfactory_grades: ℚ) / total_students
  have simplified_fraction := fraction = 7 / 10
  exact sorry

end satisfactory_fraction_l40_4017


namespace find_triangle_l40_4036

theorem find_triangle (q : ℝ) (triangle : ℝ) (h1 : 3 * triangle * q = 63) (h2 : 7 * (triangle + q) = 161) : triangle = 1 :=
sorry

end find_triangle_l40_4036


namespace number_of_zookeepers_12_l40_4071

theorem number_of_zookeepers_12 :
  let P := 30 -- number of penguins
  let Zr := 22 -- number of zebras
  let T := 8 -- number of tigers
  let A_heads := P + Zr + T -- total number of animal heads
  let A_feet := (2 * P) + (4 * Zr) + (4 * T) -- total number of animal feet
  ∃ Z : ℕ, -- number of zookeepers
  (A_heads + Z) + 132 = A_feet + (2 * Z) → Z = 12 :=
by
  sorry

end number_of_zookeepers_12_l40_4071


namespace rods_in_one_mile_l40_4030

theorem rods_in_one_mile :
  (1 * 80 * 4 = 320) :=
sorry

end rods_in_one_mile_l40_4030


namespace johns_total_money_l40_4094

-- Defining the given conditions
def initial_amount : ℕ := 5
def amount_spent : ℕ := 2
def allowance : ℕ := 26

-- Constructing the proof statement
theorem johns_total_money : initial_amount - amount_spent + allowance = 29 :=
by
  sorry

end johns_total_money_l40_4094


namespace total_marbles_correct_l40_4078

-- Define the number of marbles Mary has
def MaryYellowMarbles := 9
def MaryBlueMarbles := 7
def MaryGreenMarbles := 6

-- Define the number of marbles Joan has
def JoanYellowMarbles := 3
def JoanBlueMarbles := 5
def JoanGreenMarbles := 4

-- Define the total number of marbles for Mary and Joan combined
def TotalMarbles := MaryYellowMarbles + MaryBlueMarbles + MaryGreenMarbles + JoanYellowMarbles + JoanBlueMarbles + JoanGreenMarbles

-- We want to prove that the total number of marbles is 34
theorem total_marbles_correct : TotalMarbles = 34 := by
  -- The proof is skipped with sorry
  sorry

end total_marbles_correct_l40_4078


namespace nth_derivative_ln_correct_l40_4009

noncomputable def nth_derivative_ln (n : ℕ) : ℝ → ℝ
| x => (-1)^(n-1) * (Nat.factorial (n-1)) / (1 + x) ^ n

theorem nth_derivative_ln_correct (n : ℕ) (x : ℝ) :
  deriv^[n] (λ x => Real.log (1 + x)) x = nth_derivative_ln n x := 
by
  sorry

end nth_derivative_ln_correct_l40_4009


namespace initial_tickets_l40_4039

theorem initial_tickets (X : ℕ) (h : (X - 22) + 15 = 18) : X = 25 :=
by
  sorry

end initial_tickets_l40_4039


namespace city_population_distribution_l40_4038

theorem city_population_distribution :
  (20 + 35) = 55 :=
by
  sorry

end city_population_distribution_l40_4038


namespace sum_derivatives_positive_l40_4018

noncomputable def f (x : ℝ) : ℝ := -x^2 - x^4 - x^6
noncomputable def f' (x : ℝ) : ℝ := -2*x - 4*x^3 - 6*x^5

theorem sum_derivatives_positive (x1 x2 x3 : ℝ) (h1 : x1 + x2 < 0) (h2 : x2 + x3 < 0) (h3 : x3 + x1 < 0) :
  f' x1 + f' x2 + f' x3 > 0 := 
sorry

end sum_derivatives_positive_l40_4018


namespace lucas_seq_mod_50_l40_4008

def lucas_seq : ℕ → ℕ
| 0       => 2
| 1       => 5
| (n + 2) => lucas_seq n + lucas_seq (n + 1)

theorem lucas_seq_mod_50 : lucas_seq 49 % 5 = 0 := 
by
  sorry

end lucas_seq_mod_50_l40_4008


namespace altitude_of_isosceles_triangle_l40_4052

noncomputable def radius_X (C : ℝ) := C / (2 * Real.pi)
noncomputable def radius_Y (radius_X : ℝ) := radius_X
noncomputable def a (radius_Y : ℝ) := radius_Y / 2

-- Define the theorem to be proven
theorem altitude_of_isosceles_triangle (C : ℝ) (h_C : C = 14 * Real.pi) (radius_X := radius_X C) (radius_Y := radius_Y radius_X) (a := a radius_Y) :
  ∃ h : ℝ, h = a * Real.sqrt 3 :=
sorry

end altitude_of_isosceles_triangle_l40_4052


namespace intersection_of_sets_l40_4019

-- Defining the sets as given in the conditions
def setM : Set ℝ := { x | (x + 1) * (x - 3) ≤ 0 }
def setN : Set ℝ := { x | 1 < x ∧ x < 4 }

-- Statement to prove
theorem intersection_of_sets :
  { x | (x + 1) * (x - 3) ≤ 0 } ∩ { x | 1 < x ∧ x < 4 } = { x | 1 < x ∧ x ≤ 3 } := by
sorry

end intersection_of_sets_l40_4019


namespace find_y_value_l40_4068

variable (x y z k : ℝ)

-- Conditions
def inverse_relation_y (x y : ℝ) (k : ℝ) : Prop := 5 * y = k / (x^2)
def direct_relation_z (x z : ℝ) : Prop := 3 * z = x

-- Constant from conditions
def k_constant := 500

-- Problem statement
theorem find_y_value (h1 : inverse_relation_y 2 25 k_constant) (h2 : direct_relation_z 4 6) :
  y = 6.25 :=
by
  sorry

-- Auxiliary instance to fulfill the proof requirement
noncomputable def y_value : ℝ := 6.25

end find_y_value_l40_4068


namespace fourth_power_nested_sqrt_l40_4079

noncomputable def nested_sqrt := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))

theorem fourth_power_nested_sqrt :
  (nested_sqrt ^ 4) = 6 + 4 * Real.sqrt (2 + Real.sqrt 2) :=
sorry

end fourth_power_nested_sqrt_l40_4079


namespace find_cement_used_lexi_l40_4095

def cement_used_total : ℝ := 15.1
def cement_used_tess : ℝ := 5.1
def cement_used_lexi : ℝ := cement_used_total - cement_used_tess

theorem find_cement_used_lexi : cement_used_lexi = 10 := by
  sorry

end find_cement_used_lexi_l40_4095


namespace find_optimal_price_and_units_l40_4004

noncomputable def price_and_units (x : ℝ) : Prop := 
  let cost_price := 40
  let initial_units := 500
  let profit_goal := 8000
  50 ≤ x ∧ x ≤ 70 ∧ (x - cost_price) * (initial_units - 10 * (x - 50)) = profit_goal

theorem find_optimal_price_and_units : 
  ∃ x units, price_and_units x ∧ units = 500 - 10 * (x - 50) ∧ x = 60 ∧ units = 400 := 
sorry

end find_optimal_price_and_units_l40_4004


namespace Robert_older_than_Elizabeth_l40_4062

-- Define the conditions
def Patrick_half_Robert (Patrick Robert : ℕ) : Prop := Patrick = Robert / 2
def Robert_turn_30_in_2_years (Robert : ℕ) : Prop := Robert + 2 = 30
def Elizabeth_4_years_younger_than_Patrick (Elizabeth Patrick : ℕ) : Prop := Elizabeth = Patrick - 4

-- The theorem we need to prove
theorem Robert_older_than_Elizabeth
  (Patrick Robert Elizabeth : ℕ)
  (h1 : Patrick_half_Robert Patrick Robert)
  (h2 : Robert_turn_30_in_2_years Robert)
  (h3 : Elizabeth_4_years_younger_than_Patrick Elizabeth Patrick) :
  Robert - Elizabeth = 18 :=
sorry

end Robert_older_than_Elizabeth_l40_4062


namespace bob_homework_time_l40_4055

variable (T_Alice T_Bob : ℕ)

theorem bob_homework_time (h_Alice : T_Alice = 40) (h_Bob : T_Bob = (3 * T_Alice) / 8) : T_Bob = 15 :=
by
  rw [h_Alice] at h_Bob
  norm_num at h_Bob
  exact h_Bob

-- Assuming T_Alice represents the time taken by Alice to complete her homework
-- and T_Bob represents the time taken by Bob to complete his homework,
-- we prove that T_Bob is 15 minutes given the conditions.

end bob_homework_time_l40_4055


namespace largest_fraction_l40_4092

theorem largest_fraction
  (a b c d : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d) :
  (c + d) / (a + b) ≥ (a + b) / (c + d)
  ∧ (c + d) / (a + b) ≥ (a + d) / (b + c)
  ∧ (c + d) / (a + b) ≥ (b + c) / (a + d)
  ∧ (c + d) / (a + b) ≥ (b + d) / (a + c) :=
by
  sorry

end largest_fraction_l40_4092


namespace campers_morning_count_l40_4012

theorem campers_morning_count (afternoon_count : ℕ) (additional_morning : ℕ) (h1 : afternoon_count = 39) (h2 : additional_morning = 5) :
  afternoon_count + additional_morning = 44 :=
by
  sorry

end campers_morning_count_l40_4012


namespace kanul_raw_material_expense_l40_4027

theorem kanul_raw_material_expense
  (total_amount : ℝ)
  (machinery_cost : ℝ)
  (raw_materials_cost : ℝ)
  (cash_fraction : ℝ)
  (h_total_amount : total_amount = 137500)
  (h_machinery_cost : machinery_cost = 30000)
  (h_cash_fraction: cash_fraction = 0.20)
  (h_eq : total_amount = raw_materials_cost + machinery_cost + cash_fraction * total_amount) :
  raw_materials_cost = 80000 :=
by
  rw [h_total_amount, h_machinery_cost, h_cash_fraction] at h_eq
  sorry

end kanul_raw_material_expense_l40_4027


namespace simplify_fraction_l40_4024

-- Define the fraction and the GCD condition
def fraction_numerator : ℕ := 66
def fraction_denominator : ℕ := 4356
def gcd_condition : ℕ := Nat.gcd fraction_numerator fraction_denominator

-- State the theorem that the fraction simplifies to 1/66 given the GCD condition
theorem simplify_fraction (h : gcd_condition = 66) : (fraction_numerator / fraction_denominator = 1 / 66) :=
  sorry

end simplify_fraction_l40_4024


namespace jeans_original_price_l40_4014

theorem jeans_original_price 
  (discount : ℝ -> ℝ)
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (final_price : ℝ) 
  (customer_payment : ℝ) : 
  discount_percentage = 0.10 -> 
  discount x = x * (1 - discount_percentage) -> 
  final_price = discount (2 * original_price) + original_price -> 
  customer_payment = 112 -> 
  final_price = 112 -> 
  original_price = 40 := 
by
  intros
  sorry

end jeans_original_price_l40_4014


namespace triangle_angle_condition_l40_4016

theorem triangle_angle_condition (a b h_3 : ℝ) (A C : ℝ) 
  (h : 1/(h_3^2) = 1/(a^2) + 1/(b^2)) :
  C = 90 ∨ |A - C| = 90 := 
sorry

end triangle_angle_condition_l40_4016


namespace polynomial_identity_l40_4074

theorem polynomial_identity (x y : ℝ) (h : x - y = 1) : 
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 := 
  sorry

end polynomial_identity_l40_4074


namespace arithmetic_sequence_n_value_l40_4054

def arithmetic_seq_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_n_value :
  ∀ (a1 d n an : ℕ), a1 = 3 → d = 2 → an = 25 → arithmetic_seq_nth_term a1 d n = an → n = 12 :=
by
  intros a1 d n an ha1 hd han h
  sorry

end arithmetic_sequence_n_value_l40_4054


namespace find_third_number_in_second_set_l40_4067

theorem find_third_number_in_second_set (x y: ℕ) 
    (h1 : (28 + x + 42 + 78 + 104) / 5 = 90) 
    (h2 : (128 + 255 + y + 1023 + x) / 5 = 423) 
: y = 511 := 
sorry

end find_third_number_in_second_set_l40_4067


namespace find_a_value_l40_4083

/-- Given the distribution of the random variable ξ as p(ξ = k) = a (1/3)^k for k = 1, 2, 3, 
    prove that the value of a that satisfies the probabilities summing to 1 is 27/13. -/
theorem find_a_value (a : ℝ) :
  (a * (1 / 3) + a * (1 / 3)^2 + a * (1 / 3)^3 = 1) → a = 27 / 13 :=
by 
  intro h
  sorry

end find_a_value_l40_4083


namespace find_d_h_l40_4084

theorem find_d_h (a b c d g h : ℂ) (h1 : b = 4) (h2 : g = -a - c) (h3 : a + c + g = 0) (h4 : b + d + h = 3) : 
  d + h = -1 := 
by
  sorry

end find_d_h_l40_4084


namespace investment_in_real_estate_l40_4058

def total_investment : ℝ := 200000
def ratio_real_estate_to_mutual_funds : ℝ := 7

theorem investment_in_real_estate (mutual_funds_investment real_estate_investment: ℝ) 
  (h1 : mutual_funds_investment + real_estate_investment = total_investment)
  (h2 : real_estate_investment = ratio_real_estate_to_mutual_funds * mutual_funds_investment) :
  real_estate_investment = 175000 := sorry

end investment_in_real_estate_l40_4058


namespace prob_red_or_blue_l40_4044

-- Total marbles and given probabilities
def total_marbles : ℕ := 120
def prob_white : ℚ := 1 / 4
def prob_green : ℚ := 1 / 3

-- Problem statement
theorem prob_red_or_blue : (1 - (prob_white + prob_green)) = 5 / 12 :=
by
  sorry

end prob_red_or_blue_l40_4044


namespace base8_addition_l40_4089

theorem base8_addition (X Y : ℕ) 
  (h1 : 5 * 8 + X + Y + 3 * 8 + 2 = 6 * 64 + 4 * 8 + X) :
  X + Y = 16 := by
  sorry

end base8_addition_l40_4089


namespace average_score_l40_4097

variable (T : ℝ) -- Total number of students
variable (M : ℝ) -- Number of male students
variable (F : ℝ) -- Number of female students

variable (avgM : ℝ) -- Average score for male students
variable (avgF : ℝ) -- Average score for female students

-- Conditions
def M_condition : Prop := M = 0.4 * T
def F_condition : Prop := F = 0.6 * T
def avgM_condition : Prop := avgM = 75
def avgF_condition : Prop := avgF = 80

theorem average_score (h1 : M_condition T M) (h2 : F_condition T F) 
    (h3 : avgM_condition avgM) (h4 : avgF_condition avgF) :
    (75 * M + 80 * F) / T = 78 := by
  sorry

end average_score_l40_4097


namespace new_function_expression_l40_4031

def initial_function (x : ℝ) : ℝ := -2 * x ^ 2

def shifted_function (x : ℝ) : ℝ := -2 * (x + 1) ^ 2 - 3

theorem new_function_expression :
  (∀ x : ℝ, (initial_function (x + 1) - 3) = shifted_function x) :=
by
  sorry

end new_function_expression_l40_4031


namespace sin_subtract_of_obtuse_angle_l40_4057

open Real -- Open the Real namespace for convenience.

theorem sin_subtract_of_obtuse_angle (α : ℝ) 
  (h1 : (π / 2) < α) (h2 : α < π)
  (h3 : sin (π / 4 + α) = 3 / 4)
  : sin (π / 4 - α) = - (sqrt 7) / 4 := 
by 
  sorry -- Proof placeholder.

end sin_subtract_of_obtuse_angle_l40_4057


namespace dog_food_duration_l40_4001

-- Definitions for the given conditions
def number_of_dogs : ℕ := 4
def meals_per_day : ℕ := 2
def grams_per_meal : ℕ := 250
def sacks_of_food : ℕ := 2
def kilograms_per_sack : ℝ := 50
def grams_per_kilogram : ℝ := 1000

-- Lean statement to prove the correct answer
theorem dog_food_duration : 
  ((number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) * sacks_of_food * kilograms_per_sack) / 
  (number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) = 50 :=
by 
  simp only [number_of_dogs, meals_per_day, grams_per_meal, sacks_of_food, kilograms_per_sack, grams_per_kilogram]
  norm_num
  sorry

end dog_food_duration_l40_4001


namespace solve_equation_l40_4091

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x = -4/3) ↔ (x^2 + 2 * x + 2) / (x + 2) = x + 3 :=
by
  sorry

end solve_equation_l40_4091


namespace evaluate_square_of_sum_l40_4050

theorem evaluate_square_of_sum (x y : ℕ) (h1 : x + y = 20) (h2 : 2 * x + y = 27) : (x + y) ^ 2 = 400 :=
by
  sorry

end evaluate_square_of_sum_l40_4050


namespace xiao_ming_polygon_l40_4085

theorem xiao_ming_polygon (n : ℕ) (h : (n - 2) * 180 = 2185) : n = 14 :=
by sorry

end xiao_ming_polygon_l40_4085


namespace race_positions_l40_4060

theorem race_positions
  (positions : Fin 15 → String) 
  (h_quinn_lucas : ∃ n : Fin 15, positions n = "Quinn" ∧ positions (n + 4) = "Lucas")
  (h_oliver_quinn : ∃ n : Fin 15, positions (n - 1) = "Oliver" ∧ positions n = "Quinn")
  (h_naomi_oliver : ∃ n : Fin 15, positions n = "Naomi" ∧ positions (n + 3) = "Oliver")
  (h_emma_lucas : ∃ n : Fin 15, positions n = "Lucas" ∧ positions (n + 1) = "Emma")
  (h_sara_naomi : ∃ n : Fin 15, positions n = "Naomi" ∧ positions (n + 1) = "Sara")
  (h_naomi_4th : ∃ n : Fin 15, n = 3 ∧ positions n = "Naomi") :
  positions 6 = "Oliver" :=
by
  sorry

end race_positions_l40_4060


namespace percentage_above_wholesale_correct_l40_4021

variable (wholesale_cost retail_cost employee_payment : ℝ)
variable (employee_discount percentage_above_wholesale : ℝ)

theorem percentage_above_wholesale_correct :
  wholesale_cost = 200 → 
  employee_discount = 0.25 → 
  employee_payment = 180 → 
  retail_cost = wholesale_cost + (percentage_above_wholesale / 100) * wholesale_cost →
  employee_payment = (1 - employee_discount) * retail_cost →
  percentage_above_wholesale = 20 :=
by
  intros
  sorry

end percentage_above_wholesale_correct_l40_4021


namespace range_of_a_l40_4064

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (x^2 + (a + 2) * x + 1) * ((3 - 2 * a) * x^2 + 5 * x + (3 - 2 * a)) ≥ 0) : a ∈ Set.Icc (-4 : ℝ) 0 := sorry

end range_of_a_l40_4064


namespace min_value_objective_l40_4048

variable (x y : ℝ)

def constraints : Prop :=
  3 * x + y - 6 ≥ 0 ∧ x - y - 2 ≤ 0 ∧ y - 3 ≤ 0

def objective (x y : ℝ) : ℝ := y - 2 * x

theorem min_value_objective :
  constraints x y → ∃ x y, objective x y = -7 :=
by
  sorry

end min_value_objective_l40_4048


namespace max_popsicles_l40_4033

theorem max_popsicles (total_money : ℝ) (cost_per_popsicle : ℝ) (h_money : total_money = 19.23) (h_cost : cost_per_popsicle = 1.60) : 
  ∃ (x : ℕ), x = ⌊total_money / cost_per_popsicle⌋ ∧ x = 12 :=
by
    sorry

end max_popsicles_l40_4033


namespace arithmetic_sequence_a18_value_l40_4025

theorem arithmetic_sequence_a18_value 
  (a : ℕ → ℕ) (d : ℕ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_sum : a 2 + a 5 + a 8 = 33)
  (h_geom : (a 5 + 1) ^ 2 = (a 2 + 1) * (a 8 + 7)) :
  a 18 = 37 :=
sorry

end arithmetic_sequence_a18_value_l40_4025


namespace sum_a_b_l40_4066

theorem sum_a_b (a b : ℕ) (h1 : 2 + 2 / 3 = 2^2 * (2 / 3))
(h2: 3 + 3 / 8 = 3^2 * (3 / 8)) 
(h3: 4 + 4 / 15 = 4^2 * (4 / 15)) 
(h_n : ∀ n, n + n / (n^2 - 1) = n^2 * (n / (n^2 - 1)) → 
(a = 9^2 - 1) ∧ (b = 9)) : 
a + b = 89 := 
sorry

end sum_a_b_l40_4066


namespace points_opposite_sides_l40_4022

theorem points_opposite_sides (x y : ℝ) (h : (3 * x + 2 * y - 8) * (-1) < 0) : 3 * x + 2 * y > 8 := 
by
  sorry

end points_opposite_sides_l40_4022


namespace number_of_roses_l40_4086

theorem number_of_roses 
  (R L T : ℕ)
  (h1 : R + L + T = 100)
  (h2 : R = L + 22)
  (h3 : R = T - 20) : R = 34 := 
sorry

end number_of_roses_l40_4086


namespace sqrt5_times_sqrt6_minus_1_over_sqrt5_bound_l40_4000

theorem sqrt5_times_sqrt6_minus_1_over_sqrt5_bound :
  4 < (Real.sqrt 5) * ((Real.sqrt 6) - 1 / (Real.sqrt 5)) ∧ (Real.sqrt 5) * ((Real.sqrt 6) - 1 / (Real.sqrt 5)) < 5 :=
by
  sorry

end sqrt5_times_sqrt6_minus_1_over_sqrt5_bound_l40_4000


namespace average_last_three_l40_4013

noncomputable def average (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem average_last_three (l : List ℝ) (h₁ : l.length = 7) (h₂ : average l = 62) 
  (h₃ : average (l.take 4) = 58) :
  average (l.drop 4) = 202 / 3 := 
by 
  sorry

end average_last_three_l40_4013


namespace remainder_division_l40_4028

theorem remainder_division (β : ℂ) 
  (h1 : β^6 + β^5 + β^4 + β^3 + β^2 + β + 1 = 0) 
  (h2 : β^7 = 1) : (β^100 + β^75 + β^50 + β^25 + 1) % (β^6 + β^5 + β^4 + β^3 + β^2 + β + 1) = -1 :=
by
  sorry

end remainder_division_l40_4028


namespace value_of_b_minus_d_squared_l40_4059

variable {a b c d : ℤ}

theorem value_of_b_minus_d_squared (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 3) : (b - d) ^ 2 = 25 := 
by
  sorry

end value_of_b_minus_d_squared_l40_4059


namespace problem_solution_l40_4043

theorem problem_solution : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 10) / (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) = 360 :=
by
  sorry

end problem_solution_l40_4043
