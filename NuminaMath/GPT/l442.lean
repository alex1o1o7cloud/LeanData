import Mathlib

namespace contrapositive_example_l442_44235

theorem contrapositive_example (x : ℝ) : 
  (x = 1 ∨ x = 2) → (x^2 - 3 * x + 2 ≤ 0) :=
by
  sorry

end contrapositive_example_l442_44235


namespace average_of_modified_set_l442_44245

theorem average_of_modified_set (a1 a2 a3 a4 a5 : ℝ) (h : (a1 + a2 + a3 + a4 + a5) / 5 = 8) :
  ((a1 + 10) + (a2 - 10) + (a3 + 10) + (a4 - 10) + (a5 + 10)) / 5 = 10 :=
by 
  sorry

end average_of_modified_set_l442_44245


namespace solve_quadratic_eq_l442_44205

theorem solve_quadratic_eq (x : ℝ) : x^2 - 2 * x - 15 = 0 ↔ (x = -3 ∨ x = 5) :=
by
  sorry

end solve_quadratic_eq_l442_44205


namespace arithmetic_sequence_common_difference_l442_44273

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (h : ∀ n, 4 * a (n + 1) - 4 * a n - 9 = 0) :
  ∃ d, (∀ n, a (n + 1) - a n = d) ∧ d = 9 / 4 := 
  sorry

end arithmetic_sequence_common_difference_l442_44273


namespace girls_joined_l442_44274

theorem girls_joined (initial_girls : ℕ) (boys : ℕ) (girls_more_than_boys : ℕ) (G : ℕ) :
  initial_girls = 632 →
  boys = 410 →
  girls_more_than_boys = 687 →
  initial_girls + G = boys + girls_more_than_boys →
  G = 465 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end girls_joined_l442_44274


namespace largest_rectangle_in_circle_l442_44224

theorem largest_rectangle_in_circle {r : ℝ} (h : r = 6) : 
  ∃ A : ℝ, A = 72 := 
by 
  sorry

end largest_rectangle_in_circle_l442_44224


namespace linear_equation_m_not_eq_4_l442_44287

theorem linear_equation_m_not_eq_4 (m x y : ℝ) :
  (m * x + 3 * y = 4 * x - 1) → m ≠ 4 :=
by
  sorry

end linear_equation_m_not_eq_4_l442_44287


namespace correct_description_of_sperm_l442_44293

def sperm_carries_almost_no_cytoplasm (sperm : Type) : Prop := sorry

theorem correct_description_of_sperm : sperm_carries_almost_no_cytoplasm sperm := 
sorry

end correct_description_of_sperm_l442_44293


namespace intersection_is_line_l442_44200

-- Define the two planes as given in the conditions
def plane1 (x y z : ℝ) : Prop := x + 5 * y + 2 * z - 5 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x - 5 * y - z + 5 = 0

-- The intersection of the planes should satisfy both plane equations
def is_on_line (x y z : ℝ) : Prop := plane1 x y z ∧ plane2 x y z

-- Define the canonical equation of the line
def line_eq (x y z : ℝ) : Prop := (∃ k : ℝ, x = 5 * k ∧ y = 5 * k + 1 ∧ z = -15 * k)

-- The proof statement
theorem intersection_is_line :
  (∀ x y z : ℝ, is_on_line x y z → line_eq x y z) ∧ 
  (∀ x y z : ℝ, line_eq x y z → is_on_line x y z) :=
by
  sorry

end intersection_is_line_l442_44200


namespace sum_faces_edges_vertices_of_octagonal_pyramid_l442_44219

-- We define an octagonal pyramid with the given geometric properties.
structure OctagonalPyramid :=
  (base_vertices : ℕ) -- the number of vertices of the base
  (base_edges : ℕ)    -- the number of edges of the base
  (apex : ℕ)          -- the single apex of the pyramid
  (faces : ℕ)         -- the total number of faces: base face + triangular faces
  (edges : ℕ)         -- the total number of edges
  (vertices : ℕ)      -- the total number of vertices

-- Now we instantiate the structure based on the conditions.
def octagonalPyramid : OctagonalPyramid :=
  { base_vertices := 8,
    base_edges := 8,
    apex := 1,
    faces := 9,
    edges := 16,
    vertices := 9 }

-- We prove that the total number of faces, edges, and vertices sum to 34.
theorem sum_faces_edges_vertices_of_octagonal_pyramid : 
  (octagonalPyramid.faces + octagonalPyramid.edges + octagonalPyramid.vertices = 34) :=
by
  -- The proof steps are omitted as per instruction.
  sorry

end sum_faces_edges_vertices_of_octagonal_pyramid_l442_44219


namespace value_of_m_l442_44276

theorem value_of_m (m : ℝ) (h₁ : m^2 - 9 * m + 19 = 1) (h₂ : 2 * m^2 - 7 * m - 9 ≤ 0) : m = 3 :=
sorry

end value_of_m_l442_44276


namespace matrix_operation_value_l442_44278

theorem matrix_operation_value : 
  let p := 4 
  let q := 5
  let r := 2
  let s := 3 
  (p * s - q * r) = 2 :=
by
  sorry

end matrix_operation_value_l442_44278


namespace consecutive_even_number_difference_l442_44296

theorem consecutive_even_number_difference (x : ℤ) (h : x^2 - (x - 2)^2 = 2012) : x = 504 :=
sorry

end consecutive_even_number_difference_l442_44296


namespace find_number_l442_44294

theorem find_number (x : ℝ) (h : 0.45 * x = 162) : x = 360 :=
sorry

end find_number_l442_44294


namespace option1_cost_expression_option2_cost_expression_cost_comparison_x_20_more_cost_effective_strategy_cost_x_20_l442_44232

def teapot_price : ℕ := 20
def teacup_price : ℕ := 6
def discount_rate : ℝ := 0.9

def option1_cost (x : ℕ) : ℕ :=
  5 * teapot_price + (x - 5) * teacup_price

def option2_cost (x : ℕ) : ℝ :=
  discount_rate * (5 * teapot_price + x * teacup_price)

theorem option1_cost_expression (x : ℕ) (h : x > 5) : option1_cost x = 6 * x + 70 := by
  sorry

theorem option2_cost_expression (x : ℕ) (h : x > 5) : option2_cost x = 5.4 * x + 90 := by
  sorry

theorem cost_comparison_x_20 : option1_cost 20 < option2_cost 20 := by
  sorry

theorem more_cost_effective_strategy_cost_x_20 : (5 * teapot_price + 15 * teacup_price * discount_rate) = 181 := by
  sorry

end option1_cost_expression_option2_cost_expression_cost_comparison_x_20_more_cost_effective_strategy_cost_x_20_l442_44232


namespace proof_star_ast_l442_44252

noncomputable def star (a b : ℕ) : ℕ := sorry  -- representing binary operation for star
noncomputable def ast (a b : ℕ) : ℕ := sorry  -- representing binary operation for ast

theorem proof_star_ast :
  star 12 2 * ast 9 3 = 2 →
  (star 7 3 * ast 12 6) = 7 / 6 :=
by
  sorry

end proof_star_ast_l442_44252


namespace least_number_of_roots_l442_44266

variable (g : ℝ → ℝ) -- Declare the function g with domain ℝ and codomain ℝ

-- Define the conditions as assumptions.
variable (h1 : ∀ x : ℝ, g (3 + x) = g (3 - x))
variable (h2 : ∀ x : ℝ, g (8 + x) = g (8 - x))
variable (h3 : g 0 = 0)

-- State the theorem to prove the necessary number of roots.
theorem least_number_of_roots : ∀ a b : ℝ, a ≤ -2000 ∧ b ≥ 2000 → ∃ n ≥ 668, ∃ x : ℝ, g x = 0 ∧ a ≤ x ∧ x ≤ b :=
by
  -- To be filled in with the logic to prove the theorem.
  sorry

end least_number_of_roots_l442_44266


namespace max_ab_bc_cd_l442_44288

theorem max_ab_bc_cd {a b c d : ℝ} (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_sum : a + b + c + d = 200) (h_a : a = 2 * d) : 
  ab + bc + cd ≤ 14166.67 :=
sorry

end max_ab_bc_cd_l442_44288


namespace cone_volume_divided_by_pi_l442_44283

theorem cone_volume_divided_by_pi : 
  let r := 15
  let l := 20
  let h := 5 * Real.sqrt 7
  let V := (1/3:ℝ) * Real.pi * r^2 * h
  (V / Real.pi = 1125 * Real.sqrt 7) := sorry

end cone_volume_divided_by_pi_l442_44283


namespace find_sum_of_money_l442_44290

theorem find_sum_of_money (P : ℝ) (H1 : P * 0.18 * 2 - P * 0.12 * 2 = 840) : P = 7000 :=
by
  sorry

end find_sum_of_money_l442_44290


namespace crayons_total_l442_44269

theorem crayons_total (Wanda Dina Jacob: ℕ) (hW: Wanda = 62) (hD: Dina = 28) (hJ: Jacob = Dina - 2) :
  Wanda + Dina + Jacob = 116 :=
by
  sorry

end crayons_total_l442_44269


namespace no_rational_roots_l442_44258

theorem no_rational_roots : ¬ ∃ x : ℚ, 5 * x^3 - 4 * x^2 - 8 * x + 3 = 0 :=
by
  sorry

end no_rational_roots_l442_44258


namespace dot_product_eq_half_l442_44259

noncomputable def vector_dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2
  
theorem dot_product_eq_half :
  vector_dot_product (Real.cos (25 * Real.pi / 180), Real.sin (25 * Real.pi / 180))
                     (Real.cos (85 * Real.pi / 180), Real.cos (5 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end dot_product_eq_half_l442_44259


namespace population_increase_l442_44218

theorem population_increase (i j : ℝ) : 
  ∀ (m : ℝ), m * (1 + i / 100) * (1 + j / 100) = m * (1 + (i + j + i * j / 100) / 100) := 
by
  intro m
  sorry

end population_increase_l442_44218


namespace eval_expression_l442_44282

theorem eval_expression : 8 - (6 / (4 - 2)) = 5 := 
sorry

end eval_expression_l442_44282


namespace min_value_of_quadratic_l442_44229

theorem min_value_of_quadratic : ∀ x : ℝ, ∃ y : ℝ, y = (x - 1)^2 - 3 ∧ (∀ z : ℝ, (z - 1)^2 - 3 ≥ y) :=
by
  sorry

end min_value_of_quadratic_l442_44229


namespace x_y_n_sum_l442_44210

theorem x_y_n_sum (x y n : ℕ) (h1 : 10 ≤ x ∧ x ≤ 99) (h2 : 10 ≤ y ∧ y ≤ 99) (h3 : y = (x % 10) * 10 + (x / 10)) (h4 : x^2 + y^2 = n^2) : x + y + n = 132 :=
sorry

end x_y_n_sum_l442_44210


namespace circle_equation_tangent_line_l442_44220

theorem circle_equation_tangent_line :
  ∃ r : ℝ, ∀ x y : ℝ, (x - 3)^2 + (y + 5)^2 = r^2 ↔ x - 7 * y + 2 = 0 :=
sorry

end circle_equation_tangent_line_l442_44220


namespace percent_difference_l442_44234

variables (w q y z x : ℝ)

-- Given conditions
def cond1 : Prop := w = 0.60 * q
def cond2 : Prop := q = 0.60 * y
def cond3 : Prop := z = 0.54 * y
def cond4 : Prop := x = 1.30 * w

-- The proof problem
theorem percent_difference (h1 : cond1 w q)
                           (h2 : cond2 q y)
                           (h3 : cond3 z y)
                           (h4 : cond4 x w) :
  ((z - x) / w) * 100 = 20 :=
by
  sorry

end percent_difference_l442_44234


namespace product_defect_rate_correct_l442_44212

-- Definitions for the defect rates of the stages
def defect_rate_stage1 : ℝ := 0.10
def defect_rate_stage2 : ℝ := 0.03

-- Definitions for the probability of passing each stage without defects
def pass_rate_stage1 : ℝ := 1 - defect_rate_stage1
def pass_rate_stage2 : ℝ := 1 - defect_rate_stage2

-- Definition for the overall probability of a product not being defective
def pass_rate_overall : ℝ := pass_rate_stage1 * pass_rate_stage2

-- Definition for the overall defect rate based on the above probabilities
def defect_rate_product : ℝ := 1 - pass_rate_overall

-- The theorem statement to be proved
theorem product_defect_rate_correct : defect_rate_product = 0.127 :=
by
  -- Proof here
  sorry

end product_defect_rate_correct_l442_44212


namespace like_term_exists_l442_44257

variable (a b : ℝ) (x y : ℝ)

theorem like_term_exists : ∃ b : ℝ, b * x^5 * y^3 = 3 * x^5 * y^3 ∧ b ≠ a :=
by
  -- existence of b
  use 3
  -- proof is omitted
  sorry

end like_term_exists_l442_44257


namespace percentage_temporary_workers_l442_44295

-- Definitions based on the given conditions
def total_workers : ℕ := 100
def percentage_technicians : ℝ := 0.9
def percentage_non_technicians : ℝ := 0.1
def percentage_permanent_technicians : ℝ := 0.9
def percentage_permanent_non_technicians : ℝ := 0.1

-- Statement to prove that the percentage of temporary workers is 18%
theorem percentage_temporary_workers :
  100 * (1 - (percentage_permanent_technicians * percentage_technicians +
              percentage_permanent_non_technicians * percentage_non_technicians)) = 18 :=
by sorry

end percentage_temporary_workers_l442_44295


namespace quadratic_distinct_roots_iff_m_lt_four_l442_44238

theorem quadratic_distinct_roots_iff_m_lt_four (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 4 * x₁ + m = 0) ∧ (x₂^2 - 4 * x₂ + m = 0)) ↔ m < 4 :=
by sorry

end quadratic_distinct_roots_iff_m_lt_four_l442_44238


namespace A_finishes_race_in_36_seconds_l442_44247

-- Definitions of conditions
def distance_A := 130 -- A covers a distance of 130 meters
def distance_B := 130 -- B covers a distance of 130 meters
def time_B := 45 -- B covers the distance in 45 seconds
def distance_B_lag := 26 -- A beats B by 26 meters

-- Statement to prove
theorem A_finishes_race_in_36_seconds : 
  ∃ t : ℝ, distance_A / t + distance_B_lag = distance_B / time_B := sorry

end A_finishes_race_in_36_seconds_l442_44247


namespace sqrt_log_equality_l442_44255

noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem sqrt_log_equality {x y : ℝ} (hx : x > 0) (hy : y > 0) :
    Real.sqrt (log4 x + 2 * log2 y) = Real.sqrt (log2 (x * y^2)) / Real.sqrt 2 :=
sorry

end sqrt_log_equality_l442_44255


namespace virginia_eggs_l442_44237

-- Definitions and conditions
variable (eggs_start : Nat)
variable (eggs_taken : Nat := 3)
variable (eggs_end : Nat := 93)

-- Problem statement to prove
theorem virginia_eggs : eggs_start - eggs_taken = eggs_end → eggs_start = 96 :=
by
  intro h
  sorry

end virginia_eggs_l442_44237


namespace quadratic_rewrite_h_l442_44206

theorem quadratic_rewrite_h (a k h x : ℝ) :
  (3 * x^2 + 9 * x + 17) = a * (x - h)^2 + k ↔ h = -3/2 :=
by sorry

end quadratic_rewrite_h_l442_44206


namespace find_a5_l442_44277

-- Define the geometric sequence and the given conditions
def geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Define the conditions for our problem
def conditions (a : ℕ → ℝ) :=
  geom_sequence a 2 ∧ (∀ n, 0 < a n) ∧ a 3 * a 11 = 16

-- Our goal is to prove that a_5 = 1
theorem find_a5 (a : ℕ → ℝ) (h : conditions a) : a 5 = 1 := 
by 
  sorry

end find_a5_l442_44277


namespace tanks_difference_l442_44236

theorem tanks_difference (total_tanks german_tanks allied_tanks sanchalian_tanks : ℕ)
  (h_total : total_tanks = 115)
  (h_german_allied : german_tanks = 2 * allied_tanks + 2)
  (h_allied_sanchalian : allied_tanks = 3 * sanchalian_tanks + 1)
  (h_total_eq : german_tanks + allied_tanks + sanchalian_tanks = total_tanks) :
  german_tanks - sanchalian_tanks = 59 :=
sorry

end tanks_difference_l442_44236


namespace cos_C_of_triangle_l442_44201

theorem cos_C_of_triangle
  (sin_A : ℝ) (cos_B : ℝ) 
  (h1 : sin_A = 3/5)
  (h2 : cos_B = 5/13) :
  ∃ (cos_C : ℝ), cos_C = 16/65 :=
by
  -- Place for the proof
  sorry

end cos_C_of_triangle_l442_44201


namespace green_peaches_sum_l442_44270

theorem green_peaches_sum (G1 G2 G3 : ℕ) : 
  (4 + G1) + (4 + G2) + (3 + G3) = 20 → G1 + G2 + G3 = 9 :=
by
  intro h
  sorry

end green_peaches_sum_l442_44270


namespace find_seventh_value_l442_44284

theorem find_seventh_value (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
  (h₁ : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ = 0)
  (h₂ : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ = 10)
  (h₃ : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ = 100) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ = 210 :=
sorry

end find_seventh_value_l442_44284


namespace technicians_in_workshop_l442_44231

theorem technicians_in_workshop (T R : ℕ) 
    (h1 : 700 * 15 = 800 * T + 650 * R)
    (h2 : T + R = 15) : T = 5 := 
by
  sorry

end technicians_in_workshop_l442_44231


namespace cheryl_more_eggs_than_others_l442_44217

def kevin_eggs : ℕ := 5
def bonnie_eggs : ℕ := 13
def george_eggs : ℕ := 9
def cheryl_eggs : ℕ := 56

theorem cheryl_more_eggs_than_others : cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 :=
by
  sorry

end cheryl_more_eggs_than_others_l442_44217


namespace borrowed_sheets_l442_44291

theorem borrowed_sheets (sheets borrowed: ℕ) (average_page : ℝ) 
  (total_pages : ℕ := 80) (pages_per_sheet : ℕ := 2) (total_sheets : ℕ := 40) 
  (h1 : borrowed ≤ total_sheets)
  (h2 : sheets = total_sheets - borrowed)
  (h3 : average_page = 26) : borrowed = 17 :=
sorry 

end borrowed_sheets_l442_44291


namespace cassy_initial_jars_l442_44240

theorem cassy_initial_jars (boxes1 jars1 boxes2 jars2 leftover: ℕ) (h1: boxes1 = 10) (h2: jars1 = 12) (h3: boxes2 = 30) (h4: jars2 = 10) (h5: leftover = 80) : 
  boxes1 * jars1 + boxes2 * jars2 + leftover = 500 := 
by 
  sorry

end cassy_initial_jars_l442_44240


namespace clara_total_cookies_l442_44243

theorem clara_total_cookies :
  let cookies_per_box1 := 12
  let cookies_per_box2 := 20
  let cookies_per_box3 := 16
  let boxes_sold1 := 50
  let boxes_sold2 := 80
  let boxes_sold3 := 70
  (boxes_sold1 * cookies_per_box1 + boxes_sold2 * cookies_per_box2 + boxes_sold3 * cookies_per_box3) = 3320 :=
by
  sorry

end clara_total_cookies_l442_44243


namespace book_cost_in_cny_l442_44271

-- Conditions
def usd_to_nad : ℝ := 7      -- One US dollar to Namibian dollar
def usd_to_cny : ℝ := 6      -- One US dollar to Chinese yuan
def book_cost_nad : ℝ := 168 -- Cost of the book in Namibian dollars

-- Statement to prove
theorem book_cost_in_cny : book_cost_nad * (usd_to_cny / usd_to_nad) = 144 :=
sorry

end book_cost_in_cny_l442_44271


namespace equivalent_angle_terminal_side_l442_44215

theorem equivalent_angle_terminal_side (k : ℤ) (a : ℝ) (c : ℝ) (d : ℝ) : a = -3/10 * Real.pi → c = a * 180 / Real.pi → d = c + 360 * k →
   ∃ k : ℤ, d = 306 :=
sorry

end equivalent_angle_terminal_side_l442_44215


namespace koby_boxes_l442_44209

theorem koby_boxes (x : ℕ) (sparklers_per_box : ℕ := 3) (whistlers_per_box : ℕ := 5) 
    (cherie_sparklers : ℕ := 8) (cherie_whistlers : ℕ := 9) (total_fireworks : ℕ := 33) : 
    (sparklers_per_box * x + cherie_sparklers) + (whistlers_per_box * x + cherie_whistlers) = total_fireworks → x = 2 :=
by
  sorry

end koby_boxes_l442_44209


namespace largest_number_sum13_product36_l442_44254

-- helper definitions for sum and product of digits
def sum_digits (n : ℕ) : ℕ := Nat.digits 10 n |> List.sum
def mul_digits (n : ℕ) : ℕ := Nat.digits 10 n |> List.foldr (· * ·) 1

theorem largest_number_sum13_product36 : 
  ∃ n : ℕ, sum_digits n = 13 ∧ mul_digits n = 36 ∧ ∀ m : ℕ, sum_digits m = 13 ∧ mul_digits m = 36 → m ≤ n :=
sorry

end largest_number_sum13_product36_l442_44254


namespace primes_or_prime_squares_l442_44265

theorem primes_or_prime_squares (n : ℕ) (h1 : n > 1)
  (h2 : ∀ d, d ∣ n → d > 1 → (d - 1) ∣ (n - 1)) : 
  (∃ p, Nat.Prime p ∧ (n = p ∨ n = p * p)) :=
by
  sorry

end primes_or_prime_squares_l442_44265


namespace math_test_total_questions_l442_44241

theorem math_test_total_questions (Q : ℕ) (h : Q - 38 = 7) : Q = 45 :=
by
  sorry

end math_test_total_questions_l442_44241


namespace converse_and_inverse_false_l442_44221

-- Define the property of being a rhombus and a parallelogram
def is_rhombus (R : Type) : Prop := sorry
def is_parallelogram (P : Type) : Prop := sorry

-- Given: If a quadrilateral is a rhombus, then it is a parallelogram
def quad_imp (Q : Type) : Prop := is_rhombus Q → is_parallelogram Q

-- Prove that the converse and inverse are false
theorem converse_and_inverse_false (Q : Type) 
  (h1 : quad_imp Q) : 
  ¬(is_parallelogram Q → is_rhombus Q) ∧ ¬(¬(is_rhombus Q) → ¬(is_parallelogram Q)) :=
by
  sorry

end converse_and_inverse_false_l442_44221


namespace knocks_to_knicks_l442_44242

-- Define the conversion relationships as conditions
variable (knicks knacks knocks : ℝ)

-- 8 knicks = 3 knacks
axiom h1 : 8 * knicks = 3 * knacks

-- 5 knacks = 6 knocks
axiom h2 : 5 * knacks = 6 * knocks

-- Proving the equivalence of 30 knocks to 66.67 knicks
theorem knocks_to_knicks : 30 * knocks = 66.67 * knicks :=
by
  sorry

end knocks_to_knicks_l442_44242


namespace least_possible_integer_l442_44211

theorem least_possible_integer :
  ∃ N : ℕ,
    (∀ k, 1 ≤ k ∧ k ≤ 30 → k ≠ 24 → k ≠ 25 → N % k = 0) ∧
    (N % 24 ≠ 0) ∧
    (N % 25 ≠ 0) ∧
    N = 659375723440 :=
by
  sorry

end least_possible_integer_l442_44211


namespace largest_n_under_100000_l442_44225

theorem largest_n_under_100000 (n : ℕ) : 
  n < 100000 ∧ (9 * (n - 3)^6 - 3 * n^3 + 21 * n - 42) % 7 = 0 → n = 99996 :=
by
  sorry

end largest_n_under_100000_l442_44225


namespace abs_ineq_l442_44272

theorem abs_ineq (x : ℝ) (h : |x + 1| > 3) : x < -4 ∨ x > 2 :=
  sorry

end abs_ineq_l442_44272


namespace crayons_given_correct_l442_44263

def crayons_lost : ℕ := 161
def additional_crayons : ℕ := 410
def crayons_given (lost : ℕ) (additional : ℕ) : ℕ := lost + additional

theorem crayons_given_correct : crayons_given crayons_lost additional_crayons = 571 :=
by
  sorry

end crayons_given_correct_l442_44263


namespace dividend_calculation_l442_44289

theorem dividend_calculation 
  (D : ℝ) (Q : ℕ) (R : ℕ) 
  (hD : D = 164.98876404494382)
  (hQ : Q = 89)
  (hR : R = 14) :
  ⌈D * Q + R⌉ = 14698 :=
sorry

end dividend_calculation_l442_44289


namespace sum_of_ages_l442_44262

theorem sum_of_ages (juliet_age maggie_age ralph_age nicky_age : ℕ)
  (h1 : juliet_age = 10)
  (h2 : juliet_age = maggie_age + 3)
  (h3 : ralph_age = juliet_age + 2)
  (h4 : nicky_age = ralph_age / 2) :
  maggie_age + ralph_age + nicky_age = 25 :=
by
  sorry

end sum_of_ages_l442_44262


namespace proof_statements_l442_44246

namespace ProofProblem

-- Definitions for each condition
def is_factor (x y : ℕ) : Prop := ∃ n : ℕ, y = n * x
def is_divisor (x y : ℕ) : Prop := is_factor x y

-- Lean 4 statement for the problem
theorem proof_statements :
  is_factor 4 20 ∧
  (is_divisor 19 209 ∧ ¬ is_divisor 19 63) ∧
  (¬ is_divisor 12 75 ∧ ¬ is_divisor 12 29) ∧
  (is_divisor 11 33 ∧ ¬ is_divisor 11 64) ∧
  is_factor 9 180 :=
by
  sorry

end ProofProblem

end proof_statements_l442_44246


namespace product_of_numbers_l442_44230

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 460) : x * y = 40 := 
by 
  sorry

end product_of_numbers_l442_44230


namespace sum_S_17_33_50_l442_44299

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then - (n / 2)
  else (n / 2) + 1

theorem sum_S_17_33_50 : (S 17) + (S 33) + (S 50) = 1 := by
  sorry

end sum_S_17_33_50_l442_44299


namespace brad_more_pages_than_greg_l442_44261

def greg_pages_first_week : ℕ := 7 * 18
def greg_pages_next_two_weeks : ℕ := 14 * 22
def greg_total_pages : ℕ := greg_pages_first_week + greg_pages_next_two_weeks

def brad_pages_first_5_days : ℕ := 5 * 26
def brad_pages_remaining_12_days : ℕ := 12 * 20
def brad_total_pages : ℕ := brad_pages_first_5_days + brad_pages_remaining_12_days

def total_required_pages : ℕ := 800

theorem brad_more_pages_than_greg : brad_total_pages - greg_total_pages = 64 :=
by
  sorry

end brad_more_pages_than_greg_l442_44261


namespace simplify_expression_l442_44264

variables (a b : ℝ)

theorem simplify_expression (h₁ : a = 2) (h₂ : b = -1) :
  (2 * a^2 - a * b - b^2) - 2 * (a^2 - 2 * a * b + b^2) = -5 :=
by
  sorry

end simplify_expression_l442_44264


namespace farmer_harvest_correct_l442_44249

-- Define the conditions
def estimated_harvest : ℕ := 48097
def additional_harvest : ℕ := 684
def total_harvest : ℕ := 48781

-- The proof statement
theorem farmer_harvest_correct :
  estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end farmer_harvest_correct_l442_44249


namespace smallest_number_of_coins_l442_44244

theorem smallest_number_of_coins (p n d q : ℕ) (total : ℕ) :
  (total < 100) →
  (total = p * 1 + n * 5 + d * 10 + q * 25) →
  (∀ k < 100, ∃ (p n d q : ℕ), k = p * 1 + n * 5 + d * 10 + q * 25) →
  p + n + d + q = 10 :=
sorry

end smallest_number_of_coins_l442_44244


namespace second_statue_weight_l442_44207

theorem second_statue_weight (S : ℕ) :
  ∃ S : ℕ,
    (80 = 10 + S + 15 + 15 + 22) → S = 18 :=
by
  sorry

end second_statue_weight_l442_44207


namespace salary_based_on_tax_l442_44233

theorem salary_based_on_tax (salary tax paid_tax excess_800 excess_500 excess_500_2000 : ℤ) 
    (h1 : excess_800 = salary - 800)
    (h2 : excess_500 = min excess_800 500)
    (h3 : excess_500_2000 = excess_800 - excess_500)
    (h4 : paid_tax = (excess_500 * 5 / 100) + (excess_500_2000 * 10 / 100))
    (h5 : paid_tax = 80) :
  salary = 1850 := by
  sorry

end salary_based_on_tax_l442_44233


namespace k_eq_1_l442_44267

theorem k_eq_1 
  (n m k : ℕ) 
  (hn : n > 0) 
  (hm : m > 0) 
  (hk : k > 0) 
  (h : (n - 1) * n * (n + 1) = m^k) : 
  k = 1 := 
sorry

end k_eq_1_l442_44267


namespace tuples_and_triples_counts_are_equal_l442_44216

theorem tuples_and_triples_counts_are_equal (n : ℕ) (h : n > 0) :
  let countTuples := 8^n - 2 * 7^n + 6^n
  let countTriples := 8^n - 2 * 7^n + 6^n
  countTuples = countTriples :=
by
  sorry

end tuples_and_triples_counts_are_equal_l442_44216


namespace exists_strictly_positive_c_l442_44260

theorem exists_strictly_positive_c {a : ℕ → ℕ → ℝ} (h_diag_pos : ∀ i, a i i > 0)
  (h_off_diag_neg : ∀ i j, i ≠ j → a i j < 0) :
  ∃ (c : ℕ → ℝ), (∀ i, 
    0 < c i) ∧ 
    ((∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 > 0) ∨ 
     (∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 < 0) ∨ 
     (∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 = 0)) :=
by
  sorry

end exists_strictly_positive_c_l442_44260


namespace markese_earnings_l442_44228

-- Define the conditions
def earnings_relation (E M : ℕ) : Prop :=
  M = E - 5 ∧ M + E = 37

-- The theorem to prove
theorem markese_earnings (E M : ℕ) (h : earnings_relation E M) : M = 16 :=
by
  sorry

end markese_earnings_l442_44228


namespace tangent_line_at_point_l442_44251

def f (x : ℝ) : ℝ := x^3 + x - 16

def f' (x : ℝ) : ℝ := 3*x^2 + 1

def tangent_line (x : ℝ) (f'val : ℝ) (p_x p_y : ℝ) : ℝ := f'val * (x - p_x) + p_y

theorem tangent_line_at_point (x y : ℝ) (h : x = 2 ∧ y = -6 ∧ f 2 = -6) : 
  ∃ a b c : ℝ, a*x + b*y + c = 0 ∧ a = 13 ∧ b = -1 ∧ c = -32 :=
by
  use 13, -1, -32
  sorry

end tangent_line_at_point_l442_44251


namespace find_8b_l442_44214

variable (a b : ℚ)

theorem find_8b (h1 : 4 * a + 3 * b = 5) (h2 : a = b - 3) : 8 * b = 136 / 7 := by
  sorry

end find_8b_l442_44214


namespace triangle_is_right_angled_l442_44204

-- Define the internal angles of a triangle
variables (A B C : ℝ)
-- Condition: A, B, C are internal angles of a triangle
-- This directly implies 0 < A, B, C < pi and A + B + C = pi

-- Internal angles of a triangle sum to π
axiom angles_sum_pi : A + B + C = Real.pi

-- Condition given in the problem
axiom sin_condition : Real.sin A = Real.sin C * Real.cos B

-- We need to prove that triangle ABC is right-angled
theorem triangle_is_right_angled : C = Real.pi / 2 :=
by
  sorry

end triangle_is_right_angled_l442_44204


namespace modified_counting_game_53rd_term_l442_44256

theorem modified_counting_game_53rd_term :
  let a : ℕ := 1
  let d : ℕ := 2
  a + (53 - 1) * d = 105 :=
by 
  sorry

end modified_counting_game_53rd_term_l442_44256


namespace f_eq_2x_pow_5_l442_44292

def f (x : ℝ) : ℝ := (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1

theorem f_eq_2x_pow_5 (x : ℝ) : f x = (2*x)^5 :=
by
  sorry

end f_eq_2x_pow_5_l442_44292


namespace solution_set_f_over_x_lt_0_l442_44213

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_over_x_lt_0 :
  (∀ x, f (2 - x) = f (2 + x)) →
  (∀ x1 x2, x1 < 2 ∧ x2 < 2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0) →
  (f 4 = 0) →
  { x | f x / x < 0 } = { x | x < 0 } ∪ { x | 0 < x ∧ x < 4 } :=
by
  intros _ _ _
  sorry

end solution_set_f_over_x_lt_0_l442_44213


namespace no_negative_product_l442_44239

theorem no_negative_product (x y : ℝ) (n : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) 
(h1 : x ^ (2 * n) - y ^ (2 * n) > x) (h2 : y ^ (2 * n) - x ^ (2 * n) > y) : x * y ≥ 0 :=
sorry

end no_negative_product_l442_44239


namespace cos_plus_2sin_eq_one_l442_44223

theorem cos_plus_2sin_eq_one (α : ℝ) (h : (1 + Real.cos α) / Real.sin α = 1 / 2) : 
  Real.cos α + 2 * Real.sin α = 1 := 
by
  sorry

end cos_plus_2sin_eq_one_l442_44223


namespace imaginary_part_of_z_l442_44227

-- Define complex numbers and necessary conditions
variable (z : ℂ)

-- The main statement
theorem imaginary_part_of_z (h : z * (1 + 2 * I) = 3 - 4 * I) : 
  (z.im = -2) :=
sorry

end imaginary_part_of_z_l442_44227


namespace boat_speed_l442_44298

theorem boat_speed (v : ℝ) : 
  let rate_current := 7
  let distance := 35.93
  let time := 44 / 60
  (v + rate_current) * time = distance → v = 42 :=
by
  intro h
  sorry

end boat_speed_l442_44298


namespace not_in_range_l442_44208

noncomputable def g (x c: ℝ) : ℝ := x^2 + c * x + 5

theorem not_in_range (c : ℝ) (hc : -2 * Real.sqrt 2 < c ∧ c < 2 * Real.sqrt 2) :
  ∀ x : ℝ, g x c ≠ 3 :=
by
  intros
  sorry

end not_in_range_l442_44208


namespace final_number_independent_of_order_l442_44280

theorem final_number_independent_of_order 
  (p q r : ℕ) : 
  ∃ k : ℕ, 
    (p % 2 ≠ 0 ∨ q % 2 ≠ 0 ∨ r % 2 ≠ 0) ∧ 
    (∀ (p' q' r' : ℕ), 
       p' + q' + r' = p + q + r → 
       p' % 2 = p % 2 ∧ q' % 2 = q % 2 ∧ r' % 2 = r % 2 → 
       (p' = 1 ∧ q' = 0 ∧ r' = 0 ∨ 
        p' = 0 ∧ q' = 1 ∧ r' = 0 ∨ 
        p' = 0 ∧ q' = 0 ∧ r' = 1) → 
       k = p ∨ k = q ∨ k = r) := 
sorry

end final_number_independent_of_order_l442_44280


namespace geometric_sum_S6_l442_44268

open Real

-- Define a geometric sequence
noncomputable def geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q ^ (n - 1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def sum_geometric (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a * n else a * (1 - q ^ n) / (1 - q)

-- Given conditions
variables (a q : ℝ) (n : ℕ)
variable (S3 : ℝ)
variable (q : ℝ) (h_q : q = 2)
variable (h_S3 : S3 = 7)

theorem geometric_sum_S6 :
  sum_geometric a 2 6 = 63 :=
  by
    sorry

end geometric_sum_S6_l442_44268


namespace area_ratio_triangle_PQR_ABC_l442_44250

noncomputable def area (A B C : ℝ×ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * (x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)

theorem area_ratio_triangle_PQR_ABC {A B C P Q R : ℝ×ℝ} 
  (h1 : dist A B + dist B C + dist C A = 1)
  (h2 : dist A P + dist P Q + dist Q B + dist B C + dist C A = 1)
  (h3 : dist P Q + dist Q R + dist R P = 1)
  (h4 : P.1 <= A.1 ∧ A.1 <= Q.1 ∧ Q.1 <= B.1) :
  area P Q R / area A B C > 2 / 9 :=
by
  sorry

end area_ratio_triangle_PQR_ABC_l442_44250


namespace isosceles_triangle_l442_44285

noncomputable def triangle_is_isosceles (A B C a b c : ℝ) (h_triangle : a = 2 * b * Real.cos C) : Prop :=
  ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)

theorem isosceles_triangle
  (A B C a b c : ℝ)
  (h_sides : a = 2 * b * Real.cos C)
  (h_triangle : ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)) :
  B = C :=
sorry

end isosceles_triangle_l442_44285


namespace problem1_problem2_l442_44202

variables {a x y : ℝ}

theorem problem1 (h1 : a^x = 2) (h2 : a^y = 3) : a^(x + y) = 6 :=
sorry

theorem problem2 (h1 : a^x = 2) (h2 : a^y = 3) : a^(2 * x - 3 * y) = 4 / 27 :=
sorry

end problem1_problem2_l442_44202


namespace cos_17pi_over_4_eq_sqrt2_over_2_l442_44281

theorem cos_17pi_over_4_eq_sqrt2_over_2 : Real.cos (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_17pi_over_4_eq_sqrt2_over_2_l442_44281


namespace trigonometric_identity_l442_44297

theorem trigonometric_identity (α β : ℝ) : 
  ((Real.tan α + Real.tan β) / Real.tan (α + β)) 
  + ((Real.tan α - Real.tan β) / Real.tan (α - β)) 
  + 2 * (Real.tan α) ^ 2 
 = 2 / (Real.cos α) ^ 2 :=
  sorry

end trigonometric_identity_l442_44297


namespace rhombus_compression_problem_l442_44222

def rhombus_diagonal_lengths (side longer_diagonal : ℝ) (compression : ℝ) : ℝ × ℝ :=
  let new_longer_diagonal := longer_diagonal - compression
  let new_shorter_diagonal := 1.2 * compression + 24
  (new_longer_diagonal, new_shorter_diagonal)

theorem rhombus_compression_problem :
  let side := 20
  let longer_diagonal := 32
  let compression := 2.62
  rhombus_diagonal_lengths side longer_diagonal compression = (29.38, 27.14) :=
by sorry

end rhombus_compression_problem_l442_44222


namespace sum_all_products_eq_l442_44203

def group1 : List ℚ := [3/4, 3/20] -- Using 0.15 as 3/20 to work with rationals
def group2 : List ℚ := [4, 2/3]
def group3 : List ℚ := [3/5, 6/5] -- Using 1.2 as 6/5 to work with rationals

def allProducts (a b c : List ℚ) : List ℚ :=
  List.bind a (fun x =>
  List.bind b (fun y =>
  List.map (fun z => x * y * z) c))

theorem sum_all_products_eq :
  (allProducts group1 group2 group3).sum = 7.56 := by
  sorry

end sum_all_products_eq_l442_44203


namespace sum_binomial_coefficients_l442_44275

theorem sum_binomial_coefficients (a b : ℕ) (h1 : a = 2^3) (h2 : b = (2 + 1)^3) : a + b = 35 :=
by
  sorry

end sum_binomial_coefficients_l442_44275


namespace min_value_of_y_l442_44248

theorem min_value_of_y (x : ℝ) (h : x > 3) : y = x + 1/(x-3) → y ≥ 5 :=
sorry

end min_value_of_y_l442_44248


namespace max_triangle_area_l442_44286

theorem max_triangle_area (a b c : ℝ) (h1 : b + c = 8) (h2 : a + b > c)
  (h3 : a + c > b) (h4 : b + c > a) :
  (a - b + c) * (a + b - c) ≤ 64 / 17 :=
by sorry

end max_triangle_area_l442_44286


namespace initial_amount_l442_44253

theorem initial_amount (H P L : ℝ) (C : ℝ) (n : ℕ) (T M : ℝ) 
  (hH : H = 10) 
  (hP : P = 2) 
  (hC : C = 1.25) 
  (hn : n = 4) 
  (hL : L = 3) 
  (hT : T = H + P + n * C) 
  (hM : M = T + L) : 
  M = 20 := 
sorry

end initial_amount_l442_44253


namespace vacation_books_pair_count_l442_44279

/-- 
Given three distinct mystery novels, three distinct fantasy novels, and three distinct biographies,
we want to prove that the number of possible pairs of books of different genres is 27.
-/

theorem vacation_books_pair_count :
  let mystery_books := 3
  let fantasy_books := 3
  let biography_books := 3
  let total_books := mystery_books + fantasy_books + biography_books
  let pairs := (total_books * (total_books - 3)) / 2
  pairs = 27 := 
by
  sorry

end vacation_books_pair_count_l442_44279


namespace oblique_projection_correct_statements_l442_44226

-- Definitions of conditions
def oblique_projection_parallel_invariant : Prop :=
  ∀ (x_parallel y_parallel : Prop), x_parallel ∧ y_parallel

def oblique_projection_length_changes : Prop :=
  ∀ (x y : ℝ), x = y / 2 ∨ x = y

def triangle_is_triangle : Prop :=
  ∀ (t : Type), t = t

def square_is_rhombus : Prop :=
  ∀ (s : Type), s = s → false

def isosceles_trapezoid_is_parallelogram : Prop :=
  ∀ (it : Type), it = it → false

def rhombus_is_rhombus : Prop :=
  ∀ (r : Type), r = r → false

-- Math proof problem
theorem oblique_projection_correct_statements :
  (triangle_is_triangle ∧ oblique_projection_parallel_invariant ∧ oblique_projection_length_changes)
  → ¬square_is_rhombus ∧ ¬isosceles_trapezoid_is_parallelogram ∧ ¬rhombus_is_rhombus :=
by 
  sorry

end oblique_projection_correct_statements_l442_44226
