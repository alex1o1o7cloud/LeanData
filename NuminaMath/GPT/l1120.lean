import Mathlib

namespace NUMINAMATH_GPT_dinner_plates_percentage_l1120_112043

/-- Define the cost of silverware and the total cost of both items -/
def silverware_cost : ℝ := 20
def total_cost : ℝ := 30

/-- Define the percentage of the silverware cost that the dinner plates cost -/
def percentage_of_silverware_cost := 50

theorem dinner_plates_percentage :
  ∃ (P : ℝ) (S : ℝ) (x : ℝ), S = silverware_cost ∧ (P + S = total_cost) ∧ (P = (x / 100) * S) ∧ x = percentage_of_silverware_cost :=
by {
  sorry
}

end NUMINAMATH_GPT_dinner_plates_percentage_l1120_112043


namespace NUMINAMATH_GPT_arithmetic_sequence_number_of_terms_l1120_112023

theorem arithmetic_sequence_number_of_terms 
  (a d : ℝ) (n : ℕ) 
  (h1 : a + (a + d) + (a + 2 * d) = 34) 
  (h2 : (a + (n-3) * d) + (a + (n-2) * d) + (a + (n-1) * d) = 146) 
  (h3 : (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 390) : 
  n = 13 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_number_of_terms_l1120_112023


namespace NUMINAMATH_GPT_area_of_triangle_hyperbola_focus_l1120_112039

theorem area_of_triangle_hyperbola_focus :
  let F₁ := (-Real.sqrt 2, 0)
  let F₂ := (Real.sqrt 2, 0)
  let hyperbola := {p : ℝ × ℝ | p.1 ^ 2 - p.2 ^ 2 = 1}
  let asymptote (p : ℝ × ℝ) := p.1 = p.2
  let circle := {p : ℝ × ℝ | (p.1 - F₁.1 / 2) ^ 2 + (p.2 - F₁.2 / 2) ^ 2 = (Real.sqrt 2) ^ 2}
  let P := (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
  let Q := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let area (p1 p2 p3 : ℝ × ℝ) := 0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))
  area F₁ P Q = Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_area_of_triangle_hyperbola_focus_l1120_112039


namespace NUMINAMATH_GPT_sum_of_GCF_and_LCM_l1120_112070

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end NUMINAMATH_GPT_sum_of_GCF_and_LCM_l1120_112070


namespace NUMINAMATH_GPT_integer_quotient_is_perfect_square_l1120_112056

theorem integer_quotient_is_perfect_square (a b : ℕ) (h : 0 < a ∧ 0 < b) (h_int : (a + b) ^ 2 % (4 * a * b + 1) = 0) :
  ∃ k : ℕ, (a + b) ^ 2 = k ^ 2 * (4 * a * b + 1) := sorry

end NUMINAMATH_GPT_integer_quotient_is_perfect_square_l1120_112056


namespace NUMINAMATH_GPT_chickens_do_not_lay_eggs_l1120_112096

theorem chickens_do_not_lay_eggs (total_chickens : ℕ) 
  (roosters : ℕ) (hens : ℕ) (hens_lay_eggs : ℕ) (hens_do_not_lay_eggs : ℕ) 
  (chickens_do_not_lay_eggs : ℕ) :
  total_chickens = 80 →
  roosters = total_chickens / 4 →
  hens = total_chickens - roosters →
  hens_lay_eggs = 3 * hens / 4 →
  hens_do_not_lay_eggs = hens - hens_lay_eggs →
  chickens_do_not_lay_eggs = hens_do_not_lay_eggs + roosters →
  chickens_do_not_lay_eggs = 35 :=
by
  intros h0 h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_chickens_do_not_lay_eggs_l1120_112096


namespace NUMINAMATH_GPT_multiplicative_inverse_137_391_l1120_112055

theorem multiplicative_inverse_137_391 :
  ∃ (b : ℕ), (b ≤ 390) ∧ (137 * b) % 391 = 1 :=
sorry

end NUMINAMATH_GPT_multiplicative_inverse_137_391_l1120_112055


namespace NUMINAMATH_GPT_Ivan_bought_10_cards_l1120_112074

-- Define variables and conditions
variables (x : ℕ) -- Number of Uno Giant Family Cards bought
def original_price : ℕ := 12
def discount_per_card : ℕ := 2
def discounted_price := original_price - discount_per_card
def total_paid : ℕ := 100

-- Lean 4 theorem statement
theorem Ivan_bought_10_cards (h : discounted_price * x = total_paid) : x = 10 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_Ivan_bought_10_cards_l1120_112074


namespace NUMINAMATH_GPT_cone_prism_ratio_is_pi_over_16_l1120_112077

noncomputable def cone_prism_volume_ratio 
  (prism_length : ℝ) (prism_width : ℝ) (prism_height : ℝ) 
  (cone_base_radius : ℝ) (cone_height : ℝ)
  (h_length : prism_length = 3) (h_width : prism_width = 4) (h_height : prism_height = 5)
  (h_radius_cone : cone_base_radius = 1.5) (h_cone_height : cone_height = 5) : ℝ :=
  (1/3) * Real.pi * cone_base_radius^2 * cone_height / (prism_length * prism_width * prism_height)

theorem cone_prism_ratio_is_pi_over_16
  (prism_length : ℝ) (prism_width : ℝ) (prism_height : ℝ)
  (cone_base_radius : ℝ) (cone_height : ℝ)
  (h_length : prism_length = 3) (h_width : prism_width = 4) (h_height : prism_height = 5)
  (h_radius_cone : cone_base_radius = 1.5) (h_cone_height : cone_height = 5) :
  cone_prism_volume_ratio prism_length prism_width prism_height cone_base_radius cone_height
    h_length h_width h_height h_radius_cone h_cone_height = Real.pi / 16 := 
by
  sorry

end NUMINAMATH_GPT_cone_prism_ratio_is_pi_over_16_l1120_112077


namespace NUMINAMATH_GPT_triangle_interior_angles_l1120_112086

theorem triangle_interior_angles (E1 E2 E3 : ℝ) (I1 I2 I3 : ℝ) (x : ℝ)
  (h1 : E1 = 12 * x) 
  (h2 : E2 = 13 * x) 
  (h3 : E3 = 15 * x)
  (h4 : E1 + E2 + E3 = 360) 
  (h5 : I1 = 180 - E1) 
  (h6 : I2 = 180 - E2) 
  (h7 : I3 = 180 - E3) :
  I1 = 72 ∧ I2 = 63 ∧ I3 = 45 :=
by
  sorry

end NUMINAMATH_GPT_triangle_interior_angles_l1120_112086


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1120_112071

theorem problem1 (a : ℝ) : |a + 2| = 4 → (a = 2 ∨ a = -6) :=
sorry

theorem problem2 (a : ℝ) (h₀ : -4 < a) (h₁ : a < 2) : |a + 4| + |a - 2| = 6 :=
sorry

theorem problem3 (a : ℝ) : ∃ x ∈ Set.Icc (-2 : ℝ) 1, |x-1| + |x+2| = 3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1120_112071


namespace NUMINAMATH_GPT_matrix_not_invertible_x_l1120_112004

theorem matrix_not_invertible_x (x : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![2 + x, 9], ![4 - x, 10]]
  A.det = 0 ↔ x = 16 / 19 := sorry

end NUMINAMATH_GPT_matrix_not_invertible_x_l1120_112004


namespace NUMINAMATH_GPT_positional_relationship_l1120_112090

-- Definitions of skew_lines and parallel_lines
def skew_lines (a b : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, ¬ (a x y ∨ b x y) 

def parallel_lines (a c : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ x y, c x y = a (k * x) (k * y)

-- Theorem statement
theorem positional_relationship (a b c : ℝ → ℝ → Prop) 
  (h1 : skew_lines a b) 
  (h2 : parallel_lines a c) : 
  skew_lines c b ∨ (∃ x y, c x y ∧ b x y) :=
sorry

end NUMINAMATH_GPT_positional_relationship_l1120_112090


namespace NUMINAMATH_GPT_zero_count_at_end_of_45_320_125_product_l1120_112027

theorem zero_count_at_end_of_45_320_125_product :
  let p := 45 * 320 * 125
  45 = 5 * 3^2 ∧ 320 = 2^6 * 5 ∧ 125 = 5^3 →
  p = 2^6 * 3^2 * 5^5 →
  p % 10^5 = 0 ∧ p % 10^6 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_count_at_end_of_45_320_125_product_l1120_112027


namespace NUMINAMATH_GPT_gcd_90_450_l1120_112032

theorem gcd_90_450 : Int.gcd 90 450 = 90 := by
  sorry

end NUMINAMATH_GPT_gcd_90_450_l1120_112032


namespace NUMINAMATH_GPT_student_correct_sums_l1120_112081

theorem student_correct_sums (x wrong total : ℕ) (h1 : wrong = 2 * x) (h2 : total = x + wrong) (h3 : total = 54) : x = 18 :=
by
  sorry

end NUMINAMATH_GPT_student_correct_sums_l1120_112081


namespace NUMINAMATH_GPT_perimeter_of_billboard_l1120_112007
noncomputable def perimeter_billboard : ℝ :=
  let width := 8
  let area := 104
  let length := area / width
  let perimeter := 2 * (length + width)
  perimeter

theorem perimeter_of_billboard (width area : ℝ) (P : width = 8 ∧ area = 104) :
    perimeter_billboard = 42 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_billboard_l1120_112007


namespace NUMINAMATH_GPT_sum_first_13_terms_l1120_112094

variable {a_n : ℕ → ℝ} (S : ℕ → ℝ)
variable (a_1 d : ℝ)

-- Arithmetic sequence properties
axiom arithmetic_sequence (n : ℕ) : a_n n = a_1 + (n - 1) * d

-- Sum of the first n terms
axiom sum_of_terms (n : ℕ) : S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom sum_specific_terms : a_n 2 + a_n 7 + a_n 12 = 30

-- Theorem to prove
theorem sum_first_13_terms : S 13 = 130 := sorry

end NUMINAMATH_GPT_sum_first_13_terms_l1120_112094


namespace NUMINAMATH_GPT_bc_together_l1120_112008

theorem bc_together (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 20) : B + C = 320 :=
by
  sorry

end NUMINAMATH_GPT_bc_together_l1120_112008


namespace NUMINAMATH_GPT_correct_operation_is_B_l1120_112059

-- Definitions of the operations as conditions
def operation_A (x : ℝ) : Prop := 3 * x - x = 3
def operation_B (x : ℝ) : Prop := x^2 * x^3 = x^5
def operation_C (x : ℝ) : Prop := x^6 / x^2 = x^3
def operation_D (x : ℝ) : Prop := (x^2)^3 = x^5

-- Prove that the correct operation is B
theorem correct_operation_is_B (x : ℝ) : operation_B x :=
by
  show x^2 * x^3 = x^5
  sorry

end NUMINAMATH_GPT_correct_operation_is_B_l1120_112059


namespace NUMINAMATH_GPT_factorial_div_l1120_112003

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end NUMINAMATH_GPT_factorial_div_l1120_112003


namespace NUMINAMATH_GPT_B_work_rate_l1120_112033

-- Definitions for the conditions
def A (t : ℝ) := 1 / 15 -- A's work rate per hour
noncomputable def B : ℝ := 1 / 10 - 1 / 15 -- Definition using the condition of the combined work rate

-- Lean 4 statement for the proof problem
theorem B_work_rate : B = 1 / 30 := by sorry

end NUMINAMATH_GPT_B_work_rate_l1120_112033


namespace NUMINAMATH_GPT_jackson_pays_2100_l1120_112041

def tile_cost (length : ℝ) (width : ℝ) (tiles_per_sqft : ℝ) (percent_green : ℝ) (cost_green : ℝ) (cost_red : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * percent_green
  let red_tiles := total_tiles - green_tiles
  let cost_green_total := green_tiles * cost_green
  let cost_red_total := red_tiles * cost_red
  cost_green_total + cost_red_total

theorem jackson_pays_2100 :
  tile_cost 10 25 4 0.4 3 1.5 = 2100 :=
by
  sorry

end NUMINAMATH_GPT_jackson_pays_2100_l1120_112041


namespace NUMINAMATH_GPT_ratio_of_speeds_l1120_112038

theorem ratio_of_speeds (k r t V1 V2 : ℝ) (hk : 0 < k) (hr : 0 < r) (ht : 0 < t)
    (h1 : r * (V1 - V2) = k) (h2 : t * (V1 + V2) = k) :
    |r + t| / |r - t| = V1 / V2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l1120_112038


namespace NUMINAMATH_GPT_positive_value_of_X_l1120_112036

def hash_relation (X Y : ℕ) : ℕ := X^2 + Y^2

theorem positive_value_of_X (X : ℕ) (h : hash_relation X 7 = 290) : X = 17 :=
by sorry

end NUMINAMATH_GPT_positive_value_of_X_l1120_112036


namespace NUMINAMATH_GPT_squared_difference_l1120_112050

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_squared_difference_l1120_112050


namespace NUMINAMATH_GPT_solution_set_of_linear_inequalities_l1120_112000

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_linear_inequalities_l1120_112000


namespace NUMINAMATH_GPT_relationship_of_y_values_l1120_112058

theorem relationship_of_y_values (m n y1 y2 y3 : ℝ) (h1 : m < 0) (h2 : n > 0) 
  (hA : y1 = m * (-2) + n) (hB : y2 = m * (-3) + n) (hC : y3 = m * 1 + n) :
  y3 < y1 ∧ y1 < y2 := 
by 
  sorry

end NUMINAMATH_GPT_relationship_of_y_values_l1120_112058


namespace NUMINAMATH_GPT_slope_of_line_l1120_112054

theorem slope_of_line (x y : ℝ) : (∃ (m b : ℝ), (3 * y + 2 * x = 12) ∧ (m = -2 / 3) ∧ (y = m * x + b)) :=
sorry

end NUMINAMATH_GPT_slope_of_line_l1120_112054


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l1120_112080

theorem distance_between_parallel_lines (a d : ℝ) (d_pos : 0 ≤ d) (a_pos : 0 ≤ a) :
  {d_ | d_ = d + a ∨ d_ = |d - a|} = {d + a, abs (d - a)} :=
by
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l1120_112080


namespace NUMINAMATH_GPT_sum_of_digits_of_a_l1120_112028

-- Define a as 10^10 - 47
def a : ℕ := (10 ^ 10) - 47

-- Function to compute the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- The theorem to prove that the sum of all the digits of a is 81
theorem sum_of_digits_of_a : sum_of_digits a = 81 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_a_l1120_112028


namespace NUMINAMATH_GPT_proof_find_C_proof_find_cos_A_l1120_112037

noncomputable def find_C {a b c : ℝ} {B : ℝ} (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) : Prop :=
  ∃ (C : ℝ), 0 < C ∧ C < Real.pi ∧ C = Real.pi / 6

noncomputable def find_cos_A {a b c : ℝ} {B : ℝ} (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) (h2 : Real.cos B = 2 / 3) : Prop :=
  ∃ (A : ℝ), Real.cos A = (Real.sqrt 5 - 2 * Real.sqrt 3) / 6

theorem proof_find_C (a b c B : ℝ) (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) : find_C h1 :=
  sorry

theorem proof_find_cos_A (a b c B : ℝ) (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) (h2 : Real.cos B = 2 / 3) : find_cos_A h1 h2 :=
  sorry

end NUMINAMATH_GPT_proof_find_C_proof_find_cos_A_l1120_112037


namespace NUMINAMATH_GPT_molecular_weight_of_ammonium_bromide_l1120_112016

-- Define the atomic weights for the elements.
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90

-- Define the molecular weight of ammonium bromide based on its formula NH4Br.
def molecular_weight_NH4Br : ℝ := (1 * atomic_weight_N) + (4 * atomic_weight_H) + (1 * atomic_weight_Br)

-- State the theorem that the molecular weight of ammonium bromide is approximately 97.95 g/mol.
theorem molecular_weight_of_ammonium_bromide :
  molecular_weight_NH4Br = 97.95 :=
by
  -- The following line is a placeholder to mark the theorem as correct without proving it.
  sorry

end NUMINAMATH_GPT_molecular_weight_of_ammonium_bromide_l1120_112016


namespace NUMINAMATH_GPT_min_value_expression_l1120_112042

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 2 / b) * (a + 2 / b - 1010) + (b + 2 / a) * (b + 2 / a - 1010) + 101010 = -404040 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1120_112042


namespace NUMINAMATH_GPT_mo_tea_cups_l1120_112073

theorem mo_tea_cups (n t : ℤ) 
  (h1 : 2 * n + 5 * t = 26) 
  (h2 : 5 * t = 2 * n + 14) :
  t = 4 :=
sorry

end NUMINAMATH_GPT_mo_tea_cups_l1120_112073


namespace NUMINAMATH_GPT_additional_people_required_l1120_112024

-- Define conditions
def people := 8
def time1 := 3
def total_work := people * time1 -- This gives us the constant k

-- Define the second condition where 12 people are needed to complete in 2 hours
def required_people (t : Nat) := total_work / t

-- The number of additional people required
def additional_people := required_people 2 - people

-- State the theorem
theorem additional_people_required : additional_people = 4 :=
by 
  show additional_people = 4
  sorry

end NUMINAMATH_GPT_additional_people_required_l1120_112024


namespace NUMINAMATH_GPT_triangle_XYZ_XY2_XZ2_difference_l1120_112044

-- Define the problem parameters and conditions
def YZ : ℝ := 10
def XM : ℝ := 6
def midpoint_YZ (M : ℝ) := 2 * M = YZ

-- The main theorem to be proved
theorem triangle_XYZ_XY2_XZ2_difference :
  ∀ (XY XZ : ℝ), 
  (∀ (M : ℝ), midpoint_YZ M) →
  ((∃ (x : ℝ), (0 ≤ x ∧ x ≤ 10) ∧ XY^2 + XZ^2 = 2 * x^2 - 20 * x + 2 * (11 * x - x^2 - 11) + 100)) →
  (120 - 100 = 20) :=
by
  sorry

end NUMINAMATH_GPT_triangle_XYZ_XY2_XZ2_difference_l1120_112044


namespace NUMINAMATH_GPT_polynomial_non_negative_l1120_112088

theorem polynomial_non_negative (x : ℝ) : x^8 + x^6 - 4*x^4 + x^2 + 1 ≥ 0 := 
sorry

end NUMINAMATH_GPT_polynomial_non_negative_l1120_112088


namespace NUMINAMATH_GPT_original_ratio_l1120_112018

theorem original_ratio (x y : ℕ) (h1 : x = y + 5) (h2 : (x - 5) / (y - 5) = 5 / 4) : x / y = 6 / 5 :=
by sorry

end NUMINAMATH_GPT_original_ratio_l1120_112018


namespace NUMINAMATH_GPT_complement_of_A_in_I_is_246_l1120_112006

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def complement_A_in_I : Set ℕ := {2, 4, 6}

theorem complement_of_A_in_I_is_246 :
  (universal_set \ set_A) = complement_A_in_I :=
  by sorry

end NUMINAMATH_GPT_complement_of_A_in_I_is_246_l1120_112006


namespace NUMINAMATH_GPT_max_balls_drawn_l1120_112062

structure DrawingConditions where
  yellow_items : ℕ
  round_items : ℕ
  edible_items : ℕ
  tomato_properties : Prop
  ball_properties : Prop
  banana_properties : Prop

axiom petya_conditions : DrawingConditions → Prop 

theorem max_balls_drawn (cond : DrawingConditions)
  (h1 : cond.yellow_items = 15)
  (h2 : cond.round_items = 18)
  (h3 : cond.edible_items = 13)
  (h4 : cond.tomato_properties)
  (h5 : cond.ball_properties)
  (h6 : cond.banana_properties) :
  ∀ balls : ℕ, balls ≤ 18 :=
by
  sorry

end NUMINAMATH_GPT_max_balls_drawn_l1120_112062


namespace NUMINAMATH_GPT_abs_a_lt_abs_b_add_abs_c_l1120_112048

theorem abs_a_lt_abs_b_add_abs_c (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end NUMINAMATH_GPT_abs_a_lt_abs_b_add_abs_c_l1120_112048


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1120_112022

def quadradois (n : ℕ) : Prop :=
  ∃ (S1 S2 : ℕ), S1 ≠ S2 ∧ (S1 * S1 + S2 * S2 ≤ S1 * S1 + S2 * S2 + (n - 2))

theorem part_a : quadradois 6 := 
sorry

theorem part_b : quadradois 2015 := 
sorry

theorem part_c : ∀ (n : ℕ), n > 5 → quadradois n := 
sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1120_112022


namespace NUMINAMATH_GPT_trains_crossing_time_l1120_112078

noncomputable def length_first_train : ℝ := 120
noncomputable def length_second_train : ℝ := 160
noncomputable def speed_first_train_kmph : ℝ := 60
noncomputable def speed_second_train_kmph : ℝ := 40
noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

noncomputable def speed_first_train : ℝ := kmph_to_mps speed_first_train_kmph
noncomputable def speed_second_train : ℝ := kmph_to_mps speed_second_train_kmph
noncomputable def relative_speed : ℝ := speed_first_train + speed_second_train
noncomputable def total_distance : ℝ := length_first_train + length_second_train
noncomputable def crossing_time : ℝ := total_distance / relative_speed

theorem trains_crossing_time :
  crossing_time = 10.08 := by
  sorry

end NUMINAMATH_GPT_trains_crossing_time_l1120_112078


namespace NUMINAMATH_GPT_line_intersects_circle_l1120_112025

theorem line_intersects_circle (a : ℝ) (h : a ≠ 0) :
  ∃ p : ℝ × ℝ, (p.1 ^ 2 + p.2 ^ 2 = 9) ∧ (a * p.1 - p.2 + 2 * a = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_circle_l1120_112025


namespace NUMINAMATH_GPT_marbles_total_l1120_112087

theorem marbles_total (r b g y : ℝ) 
  (h1 : r = 1.30 * b)
  (h2 : g = 1.50 * r)
  (h3 : y = 0.80 * g) :
  r + b + g + y = 4.4692 * r :=
by
  sorry

end NUMINAMATH_GPT_marbles_total_l1120_112087


namespace NUMINAMATH_GPT_minimum_sugar_correct_l1120_112019

noncomputable def minimum_sugar (f : ℕ) (s : ℕ) : ℕ := 
  if (f ≥ 8 + s / 2 ∧ f ≤ 3 * s) then s else sorry

theorem minimum_sugar_correct (f s : ℕ) : 
  (f ≥ 8 + s / 2 ∧ f ≤ 3 * s) → s ≥ 4 :=
by sorry

end NUMINAMATH_GPT_minimum_sugar_correct_l1120_112019


namespace NUMINAMATH_GPT_find_a_c_pair_l1120_112089

-- Given conditions in the problem
variable (a c : ℝ)

-- First condition: The quadratic equation has exactly one solution
def quadratic_eq_has_one_solution : Prop :=
  let discriminant := (30:ℝ)^2 - 4 * a * c
  discriminant = 0

-- Second condition: Sum of a and c
def sum_eq_41 : Prop := a + c = 41

-- Third condition: a is less than c
def a_lt_c : Prop := a < c

-- State the proof problem
theorem find_a_c_pair (a c : ℝ) (h1 : quadratic_eq_has_one_solution a c) (h2 : sum_eq_41 a c) (h3 : a_lt_c a c) : (a, c) = (6.525, 34.475) :=
sorry

end NUMINAMATH_GPT_find_a_c_pair_l1120_112089


namespace NUMINAMATH_GPT_first_night_percentage_is_20_l1120_112061

-- Conditions
variable (total_pages : ℕ) (pages_left : ℕ)
variable (pages_second_night : ℕ)
variable (pages_third_night : ℕ)
variable (first_night_percentage : ℕ)

-- Definitions
def total_read_pages (total_pages pages_left : ℕ) : ℕ := total_pages - pages_left

def pages_first_night (total_pages first_night_percentage : ℕ) : ℕ :=
  (first_night_percentage * total_pages) / 100

def total_read_on_three_nights (total_pages pages_left pages_second_night pages_third_night first_night_percentage : ℕ) : Prop :=
  total_read_pages total_pages pages_left = pages_first_night total_pages first_night_percentage + pages_second_night + pages_third_night

-- Theorem
theorem first_night_percentage_is_20 :
  ∀ total_pages pages_left pages_second_night pages_third_night,
  total_pages = 500 →
  pages_left = 150 →
  pages_second_night = 100 →
  pages_third_night = 150 →
  total_read_on_three_nights total_pages pages_left pages_second_night pages_third_night 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_first_night_percentage_is_20_l1120_112061


namespace NUMINAMATH_GPT_min_sum_reciprocal_l1120_112066

theorem min_sum_reciprocal (a b c : ℝ) (hp0 : 0 < a) (hp1 : 0 < b) (hp2 : 0 < c) (h : a + b + c = 1) : 
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_reciprocal_l1120_112066


namespace NUMINAMATH_GPT_samantha_total_cost_l1120_112013

-- Defining the conditions in Lean
def washer_cost : ℕ := 4
def dryer_cost_per_10_min : ℕ := 25
def loads : ℕ := 2
def num_dryers : ℕ := 3
def dryer_time : ℕ := 40

-- Proving the total cost Samantha spends is $11
theorem samantha_total_cost : (loads * washer_cost + num_dryers * (dryer_time / 10 * dryer_cost_per_10_min)) = 1100 :=
by
  sorry

end NUMINAMATH_GPT_samantha_total_cost_l1120_112013


namespace NUMINAMATH_GPT_Nedy_crackers_total_l1120_112068

theorem Nedy_crackers_total :
  let packs_from_Mon_to_Thu := 8
  let packs_on_Fri := 2 * packs_from_Mon_to_Thu
  (packs_from_Mon_to_Thu + packs_on_Fri) = 24 :=
by
  let packs_from_Mon_to_Thu := 8
  let packs_on_Fri := 2 * packs_from_Mon_to_Thu
  show packs_from_Mon_to_Thu + packs_on_Fri = 24
  sorry

end NUMINAMATH_GPT_Nedy_crackers_total_l1120_112068


namespace NUMINAMATH_GPT_total_shots_cost_l1120_112030

def numDogs : ℕ := 3
def puppiesPerDog : ℕ := 4
def shotsPerPuppy : ℕ := 2
def costPerShot : ℕ := 5

theorem total_shots_cost : (numDogs * puppiesPerDog * shotsPerPuppy * costPerShot) = 120 := by
  sorry

end NUMINAMATH_GPT_total_shots_cost_l1120_112030


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1120_112009

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1120_112009


namespace NUMINAMATH_GPT_ratio_removing_middle_digit_l1120_112060

theorem ratio_removing_middle_digit 
  (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9) 
  (hb : 0 ≤ b ∧ b ≤ 9) 
  (hc : 0 ≤ c ∧ c ≤ 9)
  (h1 : 10 * b + c = 8 * a) 
  (h2 : 10 * a + b = 8 * c) : 
  (10 * a + c) / b = 17 :=
by sorry

end NUMINAMATH_GPT_ratio_removing_middle_digit_l1120_112060


namespace NUMINAMATH_GPT_seq_general_formula_l1120_112076

open Nat

def seq (a : ℕ+ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, a (n + 1) = 2 * a n / (2 + a n)

theorem seq_general_formula (a : ℕ+ → ℝ) (h : seq a) :
  ∀ n : ℕ+, a n = 2 / (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_seq_general_formula_l1120_112076


namespace NUMINAMATH_GPT_train_passing_time_l1120_112079

noncomputable def first_train_length : ℝ := 270
noncomputable def first_train_speed_kmh : ℝ := 108
noncomputable def second_train_length : ℝ := 360
noncomputable def second_train_speed_kmh : ℝ := 72

noncomputable def convert_speed_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

noncomputable def first_train_speed_mps : ℝ := convert_speed_to_mps first_train_speed_kmh
noncomputable def second_train_speed_mps : ℝ := convert_speed_to_mps second_train_speed_kmh

noncomputable def relative_speed_mps : ℝ := first_train_speed_mps + second_train_speed_mps
noncomputable def total_distance : ℝ := first_train_length + second_train_length
noncomputable def time_to_pass : ℝ := total_distance / relative_speed_mps

theorem train_passing_time : time_to_pass = 12.6 :=
by 
  sorry

end NUMINAMATH_GPT_train_passing_time_l1120_112079


namespace NUMINAMATH_GPT_num_digits_divisible_l1120_112065

theorem num_digits_divisible (h : Nat) :
  (∃ n : Fin 10, (10 * 24 + n) % n = 0) -> h = 7 :=
by sorry

end NUMINAMATH_GPT_num_digits_divisible_l1120_112065


namespace NUMINAMATH_GPT_alexa_emily_profit_l1120_112063

def lemonade_stand_profit : ℕ :=
  let total_expenses := 10 + 5 + 3
  let price_per_cup := 4
  let cups_sold := 21
  let total_revenue := price_per_cup * cups_sold
  total_revenue - total_expenses

theorem alexa_emily_profit : lemonade_stand_profit = 66 :=
  by
  sorry

end NUMINAMATH_GPT_alexa_emily_profit_l1120_112063


namespace NUMINAMATH_GPT_calc_xy_square_l1120_112034

theorem calc_xy_square
  (x y z : ℝ)
  (h1 : 2 * x * (y + z) = 1 + y * z)
  (h2 : 1 / x - 2 / y = 3 / 2)
  (h3 : x + y + 1 / 2 = 0) :
  (x + y + z) ^ 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_calc_xy_square_l1120_112034


namespace NUMINAMATH_GPT_total_money_l1120_112049

theorem total_money (gold_value silver_value cash : ℕ) (num_gold num_silver : ℕ)
  (h_gold : gold_value = 50)
  (h_silver : silver_value = 25)
  (h_num_gold : num_gold = 3)
  (h_num_silver : num_silver = 5)
  (h_cash : cash = 30) :
  gold_value * num_gold + silver_value * num_silver + cash = 305 :=
by
  sorry

end NUMINAMATH_GPT_total_money_l1120_112049


namespace NUMINAMATH_GPT_maximum_m2_n2_l1120_112045

theorem maximum_m2_n2 
  (m n : ℤ)
  (hm : 1 ≤ m ∧ m ≤ 1981) 
  (hn : 1 ≤ n ∧ n ≤ 1981)
  (h : (n^2 - m*n - m^2)^2 = 1) :
  m^2 + n^2 ≤ 3524578 :=
sorry

end NUMINAMATH_GPT_maximum_m2_n2_l1120_112045


namespace NUMINAMATH_GPT_revision_cost_is_3_l1120_112012

def cost_first_time (pages : ℕ) : ℝ := 5 * pages

def cost_for_revisions (rev1 rev2 : ℕ) (rev_cost : ℝ) : ℝ := (rev1 * rev_cost) + (rev2 * 2 * rev_cost)

def total_cost (pages rev1 rev2 : ℕ) (rev_cost : ℝ) : ℝ := 
  cost_first_time pages + cost_for_revisions rev1 rev2 rev_cost

theorem revision_cost_is_3 :
  ∀ (pages rev1 rev2 : ℕ) (total : ℝ),
      pages = 100 →
      rev1 = 30 →
      rev2 = 20 →
      total = 710 →
      total_cost pages rev1 rev2 3 = total :=
by
  intros pages rev1 rev2 total pages_eq rev1_eq rev2_eq total_eq
  sorry

end NUMINAMATH_GPT_revision_cost_is_3_l1120_112012


namespace NUMINAMATH_GPT_president_vice_president_count_l1120_112085

/-- The club consists of 24 members, split evenly with 12 boys and 12 girls. 
    There are also two classes, each containing 6 boys and 6 girls. 
    Prove that the number of ways to choose a president and a vice-president 
    if they must be of the same gender and from different classes is 144. -/
theorem president_vice_president_count :
  ∃ n : ℕ, n = 144 ∧ 
  (∀ (club : Finset ℕ) (boys girls : Finset ℕ) 
     (class1_boys class1_girls class2_boys class2_girls : Finset ℕ),
     club.card = 24 →
     boys.card = 12 → girls.card = 12 →
     class1_boys.card = 6 → class1_girls.card = 6 →
     class2_boys.card = 6 → class2_girls.card = 6 →
     (∃ president vice_president : ℕ,
     president ∈ club ∧ vice_president ∈ club ∧
     ((president ∈ boys ∧ vice_president ∈ boys) ∨ 
      (president ∈ girls ∧ vice_president ∈ girls)) ∧
     ((president ∈ class1_boys ∧ vice_president ∈ class2_boys) ∨
      (president ∈ class2_boys ∧ vice_president ∈ class1_boys) ∨
      (president ∈ class1_girls ∧ vice_president ∈ class2_girls) ∨
      (president ∈ class2_girls ∧ vice_president ∈ class1_girls)) →
     n = 144)) :=
by
  sorry

end NUMINAMATH_GPT_president_vice_president_count_l1120_112085


namespace NUMINAMATH_GPT_remainder_equivalence_l1120_112064

theorem remainder_equivalence (x : ℕ) (r : ℕ) (hx_pos : 0 < x) 
  (h1 : ∃ q1, 100 = q1 * x + r) (h2 : ∃ q2, 197 = q2 * x + r) : 
  r = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_equivalence_l1120_112064


namespace NUMINAMATH_GPT_new_average_l1120_112075

theorem new_average (n : ℕ) (average : ℝ) (new_average : ℝ) 
  (h1 : n = 10)
  (h2 : average = 80)
  (h3 : new_average = (2 * average * n) / n) : 
  new_average = 160 := 
by 
  simp [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_new_average_l1120_112075


namespace NUMINAMATH_GPT_distinct_real_roots_l1120_112093

def otimes (a b : ℝ) : ℝ := b^2 - a * b

theorem distinct_real_roots (m x : ℝ) :
  otimes (m - 2) x = m -> ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (x^2 - (m - 2) * x - m = 0) := by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_l1120_112093


namespace NUMINAMATH_GPT_extreme_point_of_f_l1120_112026

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * x^2 - Real.log x

theorem extreme_point_of_f : 
  ∃ c : ℝ, c = Real.sqrt 3 / 3 ∧ (∀ x: ℝ, x > 0 → (f x > f c → x > c) ∧ (f x < f c → x < c)) := 
sorry

end NUMINAMATH_GPT_extreme_point_of_f_l1120_112026


namespace NUMINAMATH_GPT_red_balls_in_bag_l1120_112052

theorem red_balls_in_bag : ∃ x : ℕ, (3 : ℚ) / (4 + (x : ℕ)) = 1 / 2 ∧ x = 2 := sorry

end NUMINAMATH_GPT_red_balls_in_bag_l1120_112052


namespace NUMINAMATH_GPT_tangent_ellipse_hyperbola_l1120_112098

theorem tangent_ellipse_hyperbola {m : ℝ} :
    (∀ x y : ℝ, x^2 + 9*y^2 = 9 → x^2 - m*(y + 1)^2 = 1 → false) →
    m = 72 :=
sorry

end NUMINAMATH_GPT_tangent_ellipse_hyperbola_l1120_112098


namespace NUMINAMATH_GPT_fraction_filled_in_5_minutes_l1120_112029

-- Conditions
def fill_time : ℕ := 55 -- Total minutes to fill the cistern
def duration : ℕ := 5  -- Minutes we are examining

-- The theorem to prove that the fraction filled in 'duration' minutes is 1/11
theorem fraction_filled_in_5_minutes : (duration : ℚ) / (fill_time : ℚ) = 1 / 11 :=
by
  have fraction_per_minute : ℚ := 1 / fill_time
  have fraction_in_5_minutes : ℚ := duration * fraction_per_minute
  sorry -- Proof steps would go here, if needed.

end NUMINAMATH_GPT_fraction_filled_in_5_minutes_l1120_112029


namespace NUMINAMATH_GPT_simplified_expression_evaluation_l1120_112010

theorem simplified_expression_evaluation (x : ℝ) (hx : x = Real.sqrt 7) :
    (2 * x + 3) * (2 * x - 3) - (x + 2)^2 + 4 * (x + 3) = 20 :=
by
  sorry

end NUMINAMATH_GPT_simplified_expression_evaluation_l1120_112010


namespace NUMINAMATH_GPT_xyz_ratio_l1120_112082

theorem xyz_ratio (k x y z : ℝ) (h1 : x + k * y + 3 * z = 0)
                                (h2 : 3 * x + k * y - 2 * z = 0)
                                (h3 : 2 * x + 4 * y - 3 * z = 0)
                                (x_ne_zero : x ≠ 0)
                                (y_ne_zero : y ≠ 0)
                                (z_ne_zero : z ≠ 0) :
  (k = 11) → (x * z) / (y ^ 2) = 10 := by
  sorry

end NUMINAMATH_GPT_xyz_ratio_l1120_112082


namespace NUMINAMATH_GPT_sum_of_center_coordinates_l1120_112002

theorem sum_of_center_coordinates 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : (x1, y1) = (4, 3)) 
  (h2 : (x2, y2) = (-6, 5)) : 
  (x1 + x2) / 2 + (y1 + y2) / 2 = 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_center_coordinates_l1120_112002


namespace NUMINAMATH_GPT_denver_wood_used_per_birdhouse_l1120_112053

-- Definitions used in the problem
def cost_per_piee_of_wood : ℝ := 1.50
def profit_per_birdhouse : ℝ := 5.50
def price_for_two_birdhouses : ℝ := 32
def num_birdhouses_purchased : ℝ := 2

-- Property to prove
theorem denver_wood_used_per_birdhouse (W : ℝ) 
  (h : num_birdhouses_purchased * (cost_per_piee_of_wood * W + profit_per_birdhouse) = price_for_two_birdhouses) : 
  W = 7 :=
sorry

end NUMINAMATH_GPT_denver_wood_used_per_birdhouse_l1120_112053


namespace NUMINAMATH_GPT_cathy_wins_probability_l1120_112097

theorem cathy_wins_probability : 
  (∑' (n : ℕ), (1 / 6 : ℚ)^3 * (5 / 6)^(3 * n)) = 1 / 91 
:= by sorry

end NUMINAMATH_GPT_cathy_wins_probability_l1120_112097


namespace NUMINAMATH_GPT_smallest_sum_of_digits_l1120_112031

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_sum_of_digits (N : ℕ) (hN_pos : 0 < N) 
  (h : sum_of_digits N = 3 * sum_of_digits (N + 1)) :
  sum_of_digits N = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_sum_of_digits_l1120_112031


namespace NUMINAMATH_GPT_range_of_m_l1120_112095

theorem range_of_m {m : ℝ} : (-1 ≤ 1 - m ∧ 1 - m ≤ 1) ↔ (0 ≤ m ∧ m ≤ 2) := 
sorry

end NUMINAMATH_GPT_range_of_m_l1120_112095


namespace NUMINAMATH_GPT_base_subtraction_l1120_112015

def base8_to_base10 (n : Nat) : Nat :=
  -- base 8 number 54321 (in decimal representation)
  5 * 4096 + 4 * 512 + 3 * 64 + 2 * 8 + 1

def base5_to_base10 (n : Nat) : Nat :=
  -- base 5 number 4321 (in decimal representation)
  4 * 125 + 3 * 25 + 2 * 5 + 1

theorem base_subtraction :
  base8_to_base10 54321 - base5_to_base10 4321 = 22151 := by
  sorry

end NUMINAMATH_GPT_base_subtraction_l1120_112015


namespace NUMINAMATH_GPT_train_b_speed_l1120_112099

/-- Given:
    1. Length of train A: 150 m
    2. Length of train B: 150 m
    3. Speed of train A: 54 km/hr
    4. Time taken to cross train B: 12 seconds
    Prove: The speed of train B is 36 km/hr
-/
theorem train_b_speed (l_A l_B : ℕ) (V_A : ℕ) (t : ℕ) (h1 : l_A = 150) (h2 : l_B = 150) (h3 : V_A = 54) (h4 : t = 12) :
  ∃ V_B : ℕ, V_B = 36 := sorry

end NUMINAMATH_GPT_train_b_speed_l1120_112099


namespace NUMINAMATH_GPT_polynomial_divisibility_l1120_112011

theorem polynomial_divisibility (p q : ℝ) :
    (∀ x, x = -2 ∨ x = 3 → (x^6 - x^5 + x^4 - p*x^3 + q*x^2 - 7*x - 35) = 0) →
    (p, q) = (6.86, -36.21) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1120_112011


namespace NUMINAMATH_GPT_at_least_one_greater_than_16000_l1120_112057

open Nat

theorem at_least_one_greater_than_16000 (seq : Fin 20 → ℕ)
  (h_distinct : ∀ i j : Fin 20, i ≠ j → seq i ≠ seq j)
  (h_perfect_square : ∀ i : Fin 19, ∃ k : ℕ, (seq i) * (seq (i + 1)) = k^2)
  (h_first : seq 0 = 42) : ∃ i : Fin 20, seq i > 16000 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_greater_than_16000_l1120_112057


namespace NUMINAMATH_GPT_find_a_l1120_112040

def setA (a : ℝ) : Set ℝ := {2, 4, a^3 - 2 * a^2 - a + 7}
def setB (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem find_a (a : ℝ) : 
  (setA a ∩ setB a = {2, 5}) → (a = -1 ∨ a = 2) :=
sorry

end NUMINAMATH_GPT_find_a_l1120_112040


namespace NUMINAMATH_GPT_input_x_for_y_16_l1120_112021

noncomputable def output_y_from_input_x (x : Int) : Int :=
if x < 0 then (x + 1) * (x + 1)
else (x - 1) * (x - 1)

theorem input_x_for_y_16 (x : Int) (y : Int) (h : y = 16) :
  output_y_from_input_x x = y ↔ (x = 5 ∨ x = -5) :=
by
  sorry

end NUMINAMATH_GPT_input_x_for_y_16_l1120_112021


namespace NUMINAMATH_GPT_corn_harvest_l1120_112091

theorem corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) (total_bushels : ℕ) :
  rows = 5 →
  stalks_per_row = 80 →
  stalks_per_bushel = 8 →
  total_bushels = (rows * stalks_per_row) / stalks_per_bushel →
  total_bushels = 50 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, mul_comm 5 80] at h4
  norm_num at h4
  exact h4

end NUMINAMATH_GPT_corn_harvest_l1120_112091


namespace NUMINAMATH_GPT_curve_intersection_three_points_l1120_112051

theorem curve_intersection_three_points (a : ℝ) :
  (∀ x y : ℝ, ((x^2 - y^2 = a^2) ∧ ((x-1)^2 + y^2 = 1)) → (a = 0)) :=
by
  sorry

end NUMINAMATH_GPT_curve_intersection_three_points_l1120_112051


namespace NUMINAMATH_GPT_joe_total_cars_l1120_112035

def initial_cars := 50
def multiplier := 3

theorem joe_total_cars : initial_cars + (multiplier * initial_cars) = 200 := by
  sorry

end NUMINAMATH_GPT_joe_total_cars_l1120_112035


namespace NUMINAMATH_GPT_increasing_sequence_k_range_l1120_112092

theorem increasing_sequence_k_range (k : ℝ) (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = n^2 + k * n) :
  (∀ n : ℕ, a (n + 1) > a n) → (k ≥ -3) :=
  sorry

end NUMINAMATH_GPT_increasing_sequence_k_range_l1120_112092


namespace NUMINAMATH_GPT_rectangular_field_perimeter_l1120_112046

variable (length width : ℝ)

theorem rectangular_field_perimeter (h_area : length * width = 50) (h_width : width = 5) : 2 * (length + width) = 30 := by
  sorry

end NUMINAMATH_GPT_rectangular_field_perimeter_l1120_112046


namespace NUMINAMATH_GPT_polynomial_smallest_e_l1120_112072

theorem polynomial_smallest_e :
  ∃ (a b c d e : ℤ), (a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ∧ a ≠ 0 ∧ e > 0 ∧ (x + 3) * (x - 6) * (x - 10) * (2 * x + 1) = 0) 
  ∧ e = 180 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_smallest_e_l1120_112072


namespace NUMINAMATH_GPT_student_chose_number_l1120_112083

theorem student_chose_number (x : ℤ) (h : 2 * x - 148 = 110) : x = 129 := 
by
  sorry

end NUMINAMATH_GPT_student_chose_number_l1120_112083


namespace NUMINAMATH_GPT_fg_of_3_is_2810_l1120_112047

def f (x : ℕ) : ℕ := x^2 + 1
def g (x : ℕ) : ℕ := 2 * x^3 - 1

theorem fg_of_3_is_2810 : f (g 3) = 2810 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_is_2810_l1120_112047


namespace NUMINAMATH_GPT_units_digit_7_pow_451_l1120_112067

theorem units_digit_7_pow_451 : (7^451 % 10) = 3 := by
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_451_l1120_112067


namespace NUMINAMATH_GPT_product_of_two_consecutive_even_numbers_is_divisible_by_8_l1120_112084

theorem product_of_two_consecutive_even_numbers_is_divisible_by_8 (n : ℤ) : (4 * n * (n + 1)) % 8 = 0 :=
sorry

end NUMINAMATH_GPT_product_of_two_consecutive_even_numbers_is_divisible_by_8_l1120_112084


namespace NUMINAMATH_GPT_martina_success_rate_l1120_112069

theorem martina_success_rate
  (games_played : ℕ) (games_won : ℕ) (games_remaining : ℕ)
  (games_won_remaining : ℕ) :
  games_played = 15 → 
  games_won = 9 → 
  games_remaining = 5 → 
  games_won_remaining = 5 → 
  ((games_won + games_won_remaining) / (games_played + games_remaining) : ℚ) * 100 = 70 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_martina_success_rate_l1120_112069


namespace NUMINAMATH_GPT_trout_to_bass_ratio_l1120_112014

theorem trout_to_bass_ratio 
  (bass : ℕ) 
  (trout : ℕ) 
  (blue_gill : ℕ)
  (h1 : bass = 32) 
  (h2 : blue_gill = 2 * bass) 
  (h3 : bass + trout + blue_gill = 104) 
  : (trout / bass) = 1 / 4 :=
by 
  -- intermediate steps can be included here
  sorry

end NUMINAMATH_GPT_trout_to_bass_ratio_l1120_112014


namespace NUMINAMATH_GPT_paco_ate_sweet_cookies_l1120_112005

noncomputable def PacoCookies (sweet: Nat) (salty: Nat) (salty_eaten: Nat) (extra_sweet: Nat) : Nat :=
  let corrected_salty_eaten := if salty_eaten > salty then salty else salty_eaten
  corrected_salty_eaten + extra_sweet

theorem paco_ate_sweet_cookies : PacoCookies 39 6 23 9 = 15 := by
  sorry

end NUMINAMATH_GPT_paco_ate_sweet_cookies_l1120_112005


namespace NUMINAMATH_GPT_right_triangle_legs_from_medians_l1120_112020

theorem right_triangle_legs_from_medians
  (a b : ℝ) (x y : ℝ)
  (h1 : x^2 + 4 * y^2 = 4 * a^2)
  (h2 : 4 * x^2 + y^2 = 4 * b^2) :
  y^2 = (16 * a^2 - 4 * b^2) / 15 ∧ x^2 = (16 * b^2 - 4 * a^2) / 15 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_legs_from_medians_l1120_112020


namespace NUMINAMATH_GPT_meaningful_expression_l1120_112001

-- Definition stating the meaningfulness of the expression (condition)
def is_meaningful (a : ℝ) : Prop := (a - 1) ≠ 0

-- Theorem stating that for the expression to be meaningful, a ≠ 1
theorem meaningful_expression (a : ℝ) : is_meaningful a ↔ a ≠ 1 :=
by sorry

end NUMINAMATH_GPT_meaningful_expression_l1120_112001


namespace NUMINAMATH_GPT_original_daily_production_l1120_112017

theorem original_daily_production (x N : ℕ) (h1 : N = (x - 3) * 31 + 60) (h2 : N = (x + 3) * 25 - 60) : x = 8 :=
sorry

end NUMINAMATH_GPT_original_daily_production_l1120_112017
