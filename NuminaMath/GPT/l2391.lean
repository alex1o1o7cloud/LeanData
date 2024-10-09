import Mathlib

namespace math_problem_l2391_239164

theorem math_problem 
  (a b : ℂ) (n : ℕ) (h1 : a + b = 0) (h2 : a ≠ 0) : 
  a^(2*n + 1) + b^(2*n + 1) = 0 := 
by 
  sorry

end math_problem_l2391_239164


namespace Jake_weight_loss_l2391_239175

variables (J K x : ℕ)

theorem Jake_weight_loss : 
  J = 198 ∧ J + K = 293 ∧ J - x = 2 * K → x = 8 := 
by {
  sorry
}

end Jake_weight_loss_l2391_239175


namespace potatoes_yield_l2391_239116

theorem potatoes_yield (steps_length : ℕ) (steps_width : ℕ) (step_size : ℕ) (yield_per_sqft : ℚ) 
  (h_steps_length : steps_length = 18) 
  (h_steps_width : steps_width = 25) 
  (h_step_size : step_size = 3) 
  (h_yield_per_sqft : yield_per_sqft = 1/3) 
  : (steps_length * step_size) * (steps_width * step_size) * yield_per_sqft = 1350 := 
by 
  sorry

end potatoes_yield_l2391_239116


namespace find_h_l2391_239128

theorem find_h (h : ℤ) (root_condition : (-3)^3 + h * (-3) - 18 = 0) : h = -15 :=
by
  sorry

end find_h_l2391_239128


namespace g_at_three_l2391_239179

noncomputable def g : ℝ → ℝ := sorry

axiom g_functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_nonzero_at_zero : g 0 ≠ 0
axiom g_at_one : g 1 = 2

theorem g_at_three : g 3 = 8 := sorry

end g_at_three_l2391_239179


namespace simplify_expression_eval_l2391_239139

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  ((x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2 * x + 1)))

theorem simplify_expression_eval : simplify_expression (Real.sqrt 5 - 1) = (Real.sqrt 5) / 5 :=
by
  sorry

end simplify_expression_eval_l2391_239139


namespace smallest_positive_period_of_f_minimum_value_of_f_on_interval_l2391_239137

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 2) * (Real.sin (x / 2)) * (Real.cos (x / 2)) - (Real.sqrt 2) * (Real.sin (x / 2)) ^ 2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + 2 * Real.pi) = f x :=
by sorry

theorem minimum_value_of_f_on_interval : 
  ∃ x ∈ Set.Icc (-Real.pi) 0, 
  f x = -1 - Real.sqrt 2 / 2 :=
by sorry

end smallest_positive_period_of_f_minimum_value_of_f_on_interval_l2391_239137


namespace carnival_earnings_l2391_239171

theorem carnival_earnings (days : ℕ) (total_earnings : ℕ) (h1 : days = 22) (h2 : total_earnings = 3168) : 
  (total_earnings / days) = 144 := 
by
  -- The proof would go here
  sorry

end carnival_earnings_l2391_239171


namespace factorization_1_factorization_2_l2391_239194

variables {x y m n : ℝ}

theorem factorization_1 : x^3 + 2 * x^2 * y + x * y^2 = x * (x + y)^2 :=
sorry

theorem factorization_2 : 4 * m^2 - n^2 - 4 * m + 1 = (2 * m - 1 + n) * (2 * m - 1 - n) :=
sorry

end factorization_1_factorization_2_l2391_239194


namespace min_value_frac_sum_l2391_239163

theorem min_value_frac_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 4) : 
  (4 / a^2 + 1 / b^2) ≥ 9 / 4 :=
by
  sorry

end min_value_frac_sum_l2391_239163


namespace range_of_m_l2391_239178

theorem range_of_m (x1 x2 y1 y2 m : ℝ) (h1 : x1 > x2) (h2 : y1 > y2) (h3 : y1 = (m-2)*x1) (h4 : y2 = (m-2)*x2) : m > 2 :=
by sorry

end range_of_m_l2391_239178


namespace happy_children_count_l2391_239188

theorem happy_children_count (total_children sad_children neither_children total_boys total_girls happy_boys sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : sad_children = 10)
  (h3 : neither_children = 20)
  (h4 : total_boys = 18)
  (h5 : total_girls = 42)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4) :
  ∃ happy_children, happy_children = 30 :=
  sorry

end happy_children_count_l2391_239188


namespace simplify_and_evaluate_l2391_239199

-- Define the expression
def expr (a : ℚ) : ℚ := (3 * a - 1) ^ 2 + 3 * a * (3 * a + 2)

-- Given the condition
def a_value : ℚ := -1 / 3

-- State the theorem
theorem simplify_and_evaluate : expr a_value = 3 :=
by
  -- Proof will be added here
  sorry

end simplify_and_evaluate_l2391_239199


namespace tangent_product_l2391_239118

noncomputable def tangent (x : ℝ) : ℝ := Real.tan x

theorem tangent_product : 
  tangent (20 * Real.pi / 180) * 
  tangent (40 * Real.pi / 180) * 
  tangent (60 * Real.pi / 180) * 
  tangent (80 * Real.pi / 180) = 3 :=
by
  -- Definitions and conditions
  have tg60 := Real.tan (60 * Real.pi / 180) = Real.sqrt 3
  
  -- Add tangent addition, subtraction, and triple angle formulas
  -- tangent addition formula
  have tg_add := ∀ x y : ℝ, tangent (x + y) = (tangent x + tangent y) / (1 - tangent x * tangent y)
  -- tangent subtraction formula
  have tg_sub := ∀ x y : ℝ, tangent (x - y) = (tangent x - tangent y) / (1 + tangent x * tangent y)
  -- tangent triple angle formula
  have tg_triple := ∀ α : ℝ, tangent (3 * α) = (3 * tangent α - tangent α^3) / (1 - 3 * tangent α^2)
  
  -- sorry to skip the proof
  sorry


end tangent_product_l2391_239118


namespace factor_expression_l2391_239110

theorem factor_expression (x : ℝ) : 72 * x^3 - 250 * x^7 = 2 * x^3 * (36 - 125 * x^4) :=
by
  sorry

end factor_expression_l2391_239110


namespace square_difference_identity_l2391_239138

theorem square_difference_identity (a b : ℕ) : (a - b)^2 = a^2 - 2 * a * b + b^2 :=
  by sorry

lemma evaluate_expression : (101 - 2)^2 = 9801 :=
  by
    have h := square_difference_identity 101 2
    exact h

end square_difference_identity_l2391_239138


namespace find_percentage_l2391_239107

theorem find_percentage (x p : ℝ) (h1 : 0.25 * x = p * 10 - 30) (h2 : x = 680) : p = 20 := 
sorry

end find_percentage_l2391_239107


namespace steve_take_home_pay_l2391_239140

-- Defining the conditions
def annual_salary : ℕ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℕ := 800

-- Defining the taxes function
def taxes (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the healthcare function
def healthcare (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the total deductions function
def total_deductions (salary : ℕ) (tax_rate : ℝ) (healthcare_rate : ℝ) (dues : ℕ) : ℝ :=
  (taxes salary tax_rate) + (healthcare salary healthcare_rate) + dues

-- Defining the take-home pay function
def take_home_pay (salary : ℕ) (deductions : ℝ) : ℝ :=
  salary - deductions

-- Using definitions to assert the take-home pay
theorem steve_take_home_pay : take_home_pay annual_salary (total_deductions annual_salary tax_rate healthcare_rate union_dues) = 27200 :=
by
  -- proof has been skipped
  sorry

end steve_take_home_pay_l2391_239140


namespace fit_max_blocks_l2391_239104

/-- Prove the maximum number of blocks of size 1-in x 3-in x 2-in that can fit into a box of size 4-in x 3-in x 5-in is 10. -/
theorem fit_max_blocks :
  ∀ (block_dim box_dim : ℕ → ℕ ),
  block_dim 1 = 1 ∧ block_dim 2 = 3 ∧ block_dim 3 = 2 →
  box_dim 1 = 4 ∧ box_dim 2 = 3 ∧ box_dim 3 = 5 →
  ∃ max_blocks : ℕ, max_blocks = 10 :=
by
  sorry

end fit_max_blocks_l2391_239104


namespace sum_of_squared_projections_l2391_239148

theorem sum_of_squared_projections (a l m n : ℝ) (l_proj m_proj n_proj : ℝ)
  (h : l_proj = a * Real.cos θ)
  (h1 : m_proj = a * Real.cos (Real.pi / 3 - θ))
  (h2 : n_proj = a * Real.cos (Real.pi / 3 + θ)) :
  l_proj ^ 2 + m_proj ^ 2 + n_proj ^ 2 = 3 / 2 * a ^ 2 :=
by sorry

end sum_of_squared_projections_l2391_239148


namespace symmetrical_point_l2391_239152

structure Point :=
  (x : ℝ)
  (y : ℝ)

def reflect_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem symmetrical_point (M : Point) (hM : M = {x := 3, y := -4}) : reflect_x_axis M = {x := 3, y := 4} :=
  by
  sorry

end symmetrical_point_l2391_239152


namespace total_bananas_in_collection_l2391_239145

theorem total_bananas_in_collection (groups_of_bananas : ℕ) (bananas_per_group : ℕ) 
    (h1 : groups_of_bananas = 7) (h2 : bananas_per_group = 29) :
    groups_of_bananas * bananas_per_group = 203 := by
  sorry

end total_bananas_in_collection_l2391_239145


namespace average_speeds_equation_l2391_239127

theorem average_speeds_equation (x : ℝ) (hx : 0 < x) : 
  10 / x - 7 / (1.4 * x) = 10 / 60 :=
by
  sorry

end average_speeds_equation_l2391_239127


namespace equivalent_discount_l2391_239189

variable (P d1 d2 d : ℝ)

-- Given conditions:
def original_price : ℝ := 50
def first_discount_rate : ℝ := 0.25
def second_discount_rate : ℝ := 0.10
def equivalent_single_discount_rate : ℝ := 0.325

-- Final conclusion:
theorem equivalent_discount :
  let final_price_after_first_discount := (original_price * (1 - first_discount_rate))
  let final_price_after_second_discount := (final_price_after_first_discount * (1 - second_discount_rate))
  final_price_after_second_discount = (original_price * (1 - equivalent_single_discount_rate)) :=
by
  sorry

end equivalent_discount_l2391_239189


namespace student_total_marks_l2391_239161

variables {M P C : ℕ}

theorem student_total_marks
  (h1 : C = P + 20)
  (h2 : (M + C) / 2 = 35) :
  M + P = 50 :=
sorry

end student_total_marks_l2391_239161


namespace smallest_multiple_l2391_239170

theorem smallest_multiple (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ m % 45 = 0 ∧ m % 60 = 0 ∧ m % 25 ≠ 0 ∧ m = n) → n = 180 :=
by
  sorry

end smallest_multiple_l2391_239170


namespace segment_parametrization_pqrs_l2391_239156

theorem segment_parametrization_pqrs :
  ∃ (p q r s : ℤ), 
    q = 1 ∧ 
    s = -3 ∧ 
    p + q = 6 ∧ 
    r + s = 4 ∧ 
    p^2 + q^2 + r^2 + s^2 = 84 :=
by
  use 5, 1, 7, -3
  sorry

end segment_parametrization_pqrs_l2391_239156


namespace ChipsEquivalence_l2391_239122

theorem ChipsEquivalence
  (x y : ℕ)
  (h1 : y = x - 2)
  (h2 : 3 * x - 3 = 4 * y - 4) :
  3 * x - 3 = 24 :=
by
  sorry

end ChipsEquivalence_l2391_239122


namespace percent_of_a_is_b_l2391_239185

variable (a b c : ℝ)
variable (h1 : c = 0.20 * a) (h2 : c = 0.10 * b)

theorem percent_of_a_is_b : b = 2 * a :=
by sorry

end percent_of_a_is_b_l2391_239185


namespace rows_colored_red_l2391_239151

theorem rows_colored_red (total_rows total_squares_per_row blue_rows green_squares red_squares_per_row red_rows : ℕ)
  (h_total_squares : total_rows * total_squares_per_row = 150)
  (h_blue_squares : blue_rows * total_squares_per_row = 60)
  (h_green_squares : green_squares = 66)
  (h_red_squares : 150 - 60 - 66 = 24)
  (h_red_rows : 24 / red_squares_per_row = 4) :
  red_rows = 4 := 
by sorry

end rows_colored_red_l2391_239151


namespace minimize_area_of_quadrilateral_l2391_239165

noncomputable def minimize_quad_area (AB BC CD DA A1 B1 C1 D1 : ℝ) (k : ℝ) : Prop :=
  -- Conditions
  AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0 ∧ k > 0 ∧
  A1 = k * AB ∧ B1 = k * BC ∧ C1 = k * CD ∧ D1 = k * DA →
  -- Conclusion
  k = 1 / 2

-- Statement without proof
theorem minimize_area_of_quadrilateral (AB BC CD DA : ℝ) : ∃ k : ℝ, minimize_quad_area AB BC CD DA (k * AB) (k * BC) (k * CD) (k * DA) k :=
sorry

end minimize_area_of_quadrilateral_l2391_239165


namespace goods_train_length_l2391_239186

-- Conditions
def train1_speed := 60 -- kmph
def train2_speed := 52 -- kmph
def passing_time := 9 -- seconds

-- Conversion factor from kmph to meters per second
def kmph_to_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps := kmph_to_mps (train1_speed + train2_speed)

-- Final theorem statement
theorem goods_train_length :
  relative_speed_mps * passing_time = 280 :=
sorry

end goods_train_length_l2391_239186


namespace graph_abs_symmetric_yaxis_l2391_239100

theorem graph_abs_symmetric_yaxis : 
  ∀ x : ℝ, |x| = |(-x)| :=
by
  intro x
  sorry

end graph_abs_symmetric_yaxis_l2391_239100


namespace slip_2_5_goes_to_B_l2391_239191

-- Defining the slips and their values
def slips : List ℝ := [1.5, 2, 2, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5, 6]

-- Defining the total sum of slips
def total_sum : ℝ := 52

-- Defining the cup sum values
def cup_sums : List ℝ := [11, 10, 9, 8, 7]

-- Conditions: slip with 4 goes into cup A, slip with 5 goes into cup D
def cup_A_contains : ℝ := 4
def cup_D_contains : ℝ := 5

-- Proof statement
theorem slip_2_5_goes_to_B : 
  ∃ (cup_A cup_B cup_C cup_D cup_E : List ℝ), 
    (cup_A.sum = 11 ∧ cup_B.sum = 10 ∧ cup_C.sum = 9 ∧ cup_D.sum = 8 ∧ cup_E.sum = 7) ∧
    (4 ∈ cup_A) ∧ (5 ∈ cup_D) ∧ (2.5 ∈ cup_B) :=
sorry

end slip_2_5_goes_to_B_l2391_239191


namespace rajesh_walked_distance_l2391_239103

theorem rajesh_walked_distance (H : ℝ) (D_R : ℝ) 
  (h1 : D_R = 4 * H - 10)
  (h2 : H + D_R = 25) :
  D_R = 18 :=
by
  sorry

end rajesh_walked_distance_l2391_239103


namespace sum_difference_l2391_239136

def sum_even (n : ℕ) : ℕ :=
  (n / 2) * (2 + n)

def sum_odd (n : ℕ) : ℕ :=
  (n / 2) * (1 + (n - 1))

theorem sum_difference : sum_even 100 - sum_odd 99 = 50 :=
by
  sorry

end sum_difference_l2391_239136


namespace tangent_line_at_point_l2391_239196

def tangent_line_equation (f : ℝ → ℝ) (slope : ℝ) (p : ℝ × ℝ) :=
  ∃ (a b c : ℝ), a * p.1 + b * p.2 + c = 0 ∧ a = slope ∧ p.2 = f p.1

noncomputable def curve (x : ℝ) : ℝ := x^3 + x + 1

theorem tangent_line_at_point : 
  tangent_line_equation curve 4 (1, 3) :=
sorry

end tangent_line_at_point_l2391_239196


namespace exists_perfect_square_in_sequence_of_f_l2391_239134

noncomputable def f (n : ℕ) : ℕ :=
  ⌊(n : ℝ) + Real.sqrt n⌋₊

theorem exists_perfect_square_in_sequence_of_f (m : ℕ) (h : m = 1111) :
  ∃ k, ∃ n, f^[n] m = k * k := 
sorry

end exists_perfect_square_in_sequence_of_f_l2391_239134


namespace common_divisor_is_19_l2391_239135

theorem common_divisor_is_19 (a d : ℤ) (h1 : d ∣ (35 * a + 57)) (h2 : d ∣ (45 * a + 76)) : d = 19 :=
sorry

end common_divisor_is_19_l2391_239135


namespace necessary_but_not_sufficient_condition_l2391_239143

theorem necessary_but_not_sufficient_condition (a b : ℤ) :
  (a ≠ 1 ∨ b ≠ 2) → (a + b ≠ 3) ∧ ¬((a + b ≠ 3) → (a ≠ 1 ∨ b ≠ 2)) :=
sorry

end necessary_but_not_sufficient_condition_l2391_239143


namespace min_possible_value_l2391_239192

theorem min_possible_value
  (a b c d e f g h : Int)
  (h_distinct : List.Nodup [a, b, c, d, e, f, g, h])
  (h_set_a : a ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_b : b ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_c : c ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_d : d ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_e : e ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_f : f ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_g : g ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_h : h ∈ [-9, -6, -3, 0, 1, 3, 6, 10]) :
  ∃ a b c d e f g h : Int,
  ((a + b + c + d)^2 + (e + f + g + h)^2) = 2
  :=
  sorry

end min_possible_value_l2391_239192


namespace find_intended_number_l2391_239195

theorem find_intended_number (n : ℕ) (h : 6 * n + 382 = 988) : n = 101 := 
by {
  sorry
}

end find_intended_number_l2391_239195


namespace find_g_l2391_239173

theorem find_g (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x+1) = 3 - 2 * x) (h2 : ∀ x : ℝ, f (g x) = 6 * x - 3) : 
  ∀ x : ℝ, g x = 4 - 3 * x := 
by
  sorry

end find_g_l2391_239173


namespace table_tennis_teams_equation_l2391_239141

-- Variables
variable (x : ℕ)

-- Conditions
def total_matches : ℕ := 28
def teams_playing_equation : Prop := x * (x - 1) = 28 * 2

-- Theorem Statement
theorem table_tennis_teams_equation : teams_playing_equation x :=
sorry

end table_tennis_teams_equation_l2391_239141


namespace mutually_exclusive_events_l2391_239153

-- Define the conditions
variable (redBalls greenBalls : ℕ)
variable (n : ℕ) -- Number of balls drawn
variable (event_one_red_ball event_two_green_balls : Prop)

-- Assumptions: more than two red balls and more than two green balls
axiom H1 : 2 < redBalls
axiom H2 : 2 < greenBalls

-- Assume that exactly one red ball and exactly two green balls are events
axiom H3 : event_one_red_ball = (n = 2 ∧ 1 ≤ redBalls ∧ 1 ≤ greenBalls)
axiom H4 : event_two_green_balls = (n = 2 ∧ greenBalls ≥ 2)

-- Definition of mutually exclusive events
def mutually_exclusive (A B : Prop) : Prop :=
  A ∧ B → false

-- Statement of the theorem
theorem mutually_exclusive_events :
  mutually_exclusive event_one_red_ball event_two_green_balls :=
by {
  sorry
}

end mutually_exclusive_events_l2391_239153


namespace super_cool_triangles_area_sum_l2391_239105

theorem super_cool_triangles_area_sum : 
  ∃ (areas : List ℕ), 
  (∀ (a b : ℕ), (a * b / 2 = 3 * (a + b)) → 
  (a * b / 2 ∈ areas)) ∧ 
  areas.sum = 471 := 
sorry

end super_cool_triangles_area_sum_l2391_239105


namespace equal_partitions_l2391_239167

def weights : List ℕ := List.range (81 + 1) |>.map (λ n => n * n)

theorem equal_partitions (h : weights.sum = 178605) :
  ∃ P1 P2 P3 : List ℕ, P1.sum = 59535 ∧ P2.sum = 59535 ∧ P3.sum = 59535 ∧ P1 ++ P2 ++ P3 = weights := sorry

end equal_partitions_l2391_239167


namespace complement_intersection_l2391_239150

open Set

-- Definitions based on conditions given
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 5, 7}

-- The mathematical proof problem
theorem complement_intersection :
  (U \ A) ∩ B = {1, 3, 7} :=
by
  sorry

end complement_intersection_l2391_239150


namespace no_permutation_exists_l2391_239182

open Function Set

theorem no_permutation_exists (f : ℕ → ℕ) (h : ∀ n m : ℕ, f n = f m ↔ n = m) :
  ¬ ∃ n : ℕ, (Finset.range n).image f = Finset.range n :=
by
  sorry

end no_permutation_exists_l2391_239182


namespace nut_weights_l2391_239113

noncomputable def part_weights (total_weight : ℝ) (total_parts : ℝ) : ℝ :=
  total_weight / total_parts

theorem nut_weights
  (total_weight : ℝ)
  (parts_almonds parts_walnuts parts_cashews ratio_pistachios_to_almonds : ℝ)
  (total_parts_without_pistachios total_parts_with_pistachios weight_per_part : ℝ)
  (weights_almonds weights_walnuts weights_cashews weights_pistachios : ℝ) :
  parts_almonds = 5 →
  parts_walnuts = 3 →
  parts_cashews = 2 →
  ratio_pistachios_to_almonds = 1 / 4 →
  total_parts_without_pistachios = parts_almonds + parts_walnuts + parts_cashews →
  total_parts_with_pistachios = total_parts_without_pistachios + (parts_almonds * ratio_pistachios_to_almonds) →
  weight_per_part = total_weight / total_parts_with_pistachios →
  weights_almonds = parts_almonds * weight_per_part →
  weights_walnuts = parts_walnuts * weight_per_part →
  weights_cashews = parts_cashews * weight_per_part →
  weights_pistachios = (parts_almonds * ratio_pistachios_to_almonds) * weight_per_part →
  total_weight = 300 →
  weights_almonds = 133.35 ∧
  weights_walnuts = 80.01 ∧
  weights_cashews = 53.34 ∧
  weights_pistachios = 33.34 :=
by
  intros
  sorry

end nut_weights_l2391_239113


namespace rate_of_second_batch_of_wheat_l2391_239154

theorem rate_of_second_batch_of_wheat (total_cost1 cost_per_kg1 weight1 weight2 total_weight total_cost selling_price_per_kg profit_rate cost_per_kg2 : ℝ)
  (H1 : total_cost1 = cost_per_kg1 * weight1)
  (H2 : total_weight = weight1 + weight2)
  (H3 : total_cost = total_cost1 + cost_per_kg2 * weight2)
  (H4 : selling_price_per_kg = (1 + profit_rate) * total_cost / total_weight)
  (H5 : profit_rate = 0.30)
  (H6 : cost_per_kg1 = 11.50)
  (H7 : weight1 = 30)
  (H8 : weight2 = 20)
  (H9 : selling_price_per_kg = 16.38) :
  cost_per_kg2 = 14.25 :=
by
  sorry

end rate_of_second_batch_of_wheat_l2391_239154


namespace marching_band_l2391_239133

theorem marching_band (total_members brass woodwind percussion : ℕ)
  (h1 : brass + woodwind + percussion = 110)
  (h2 : woodwind = 2 * brass)
  (h3 : percussion = 4 * woodwind) :
  brass = 10 := by
  sorry

end marching_band_l2391_239133


namespace find_m_to_make_z1_eq_z2_l2391_239120

def z1 (m : ℝ) : ℂ := (2 * m + 7 : ℝ) + (m^2 - 2 : ℂ) * Complex.I
def z2 (m : ℝ) : ℂ := (m^2 - 8 : ℝ) + (4 * m + 3 : ℂ) * Complex.I

theorem find_m_to_make_z1_eq_z2 : 
  ∃ m : ℝ, z1 m = z2 m ∧ m = 5 :=
by
  sorry

end find_m_to_make_z1_eq_z2_l2391_239120


namespace inequality_hold_l2391_239108

theorem inequality_hold (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 3) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ a^2 + b^2 + c^2 :=
by
  -- Proof goes here
  sorry

end inequality_hold_l2391_239108


namespace width_of_grass_field_l2391_239190

-- Define the conditions
def length_of_grass_field : ℝ := 75
def path_width : ℝ := 2.5
def cost_per_sq_m : ℝ := 2
def total_cost : ℝ := 1200

-- Define the width of the grass field as a variable
variable (w : ℝ)

-- Define the total length and width including the path
def total_length : ℝ := length_of_grass_field + 2 * path_width
def total_width (w : ℝ) : ℝ := w + 2 * path_width

-- Define the area of the path
def area_of_path (w : ℝ) : ℝ := (total_length * total_width w) - (length_of_grass_field * w)

-- Define the cost equation
def cost_eq (w : ℝ) : Prop := cost_per_sq_m * area_of_path w = total_cost

-- The theorem to prove
theorem width_of_grass_field : cost_eq 40 :=
by
  -- To be proved
  sorry

end width_of_grass_field_l2391_239190


namespace problem_angle_magnitude_and_sin_l2391_239111

theorem problem_angle_magnitude_and_sin (
  a b c : ℝ) (A B C : ℝ) 
  (h1 : a = Real.sqrt 7) (h2 : b = 3) 
  (h3 : Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3)
  (triangle_is_acute : A > 0 ∧ A < Real.pi / 2 ∧ B > 0 ∧ B < Real.pi / 2 ∧ C > 0 ∧ C < Real.pi / 2) : 
  A = Real.pi / 3 ∧ Real.sin (2 * B + Real.pi / 6) = -1 / 7 :=
by
  sorry

end problem_angle_magnitude_and_sin_l2391_239111


namespace cattle_train_speed_is_correct_l2391_239198

-- Given conditions as definitions
def cattle_train_speed (x : ℝ) : ℝ := x
def diesel_train_speed (x : ℝ) : ℝ := x - 33
def cattle_train_distance (x : ℝ) : ℝ := 6 * x
def diesel_train_distance (x : ℝ) : ℝ := 12 * (x - 33)

-- Statement to prove
theorem cattle_train_speed_is_correct (x : ℝ) :
  cattle_train_distance x + diesel_train_distance x = 1284 → 
  x = 93.33 :=
by
  intros h
  sorry

end cattle_train_speed_is_correct_l2391_239198


namespace probability_same_color_is_correct_l2391_239106

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l2391_239106


namespace polynomial_roots_l2391_239117

theorem polynomial_roots :
  ∃ (x : ℚ) (y : ℚ) (z : ℚ) (w : ℚ),
    (x = 1) ∧ (y = 1) ∧ (z = -2) ∧ (w = -1/2) ∧
    2*x^4 + x^3 - 6*x^2 + x + 2 = 0 ∧
    2*y^4 + y^3 - 6*y^2 + y + 2 = 0 ∧
    2*z^4 + z^3 - 6*z^2 + z + 2 = 0 ∧
    2*w^4 + w^3 - 6*w^2 + w + 2 = 0 :=
by
  sorry

end polynomial_roots_l2391_239117


namespace star_set_l2391_239131

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x}
def star (A B : Set ℝ) : Set ℝ := {x | (x ∈ A ∪ B) ∧ ¬(x ∈ A ∩ B)}

theorem star_set :
  star A B = {x | (0 ≤ x ∧ x < 1) ∨ (3 < x)} :=
by
  sorry

end star_set_l2391_239131


namespace first_day_bacteria_exceeds_200_l2391_239114

noncomputable def N : ℕ → ℕ := λ n => 5 * 3^n

theorem first_day_bacteria_exceeds_200 : ∃ n : ℕ, N n > 200 ∧ ∀ m : ℕ, m < n → N m ≤ 200 :=
by
  sorry

end first_day_bacteria_exceeds_200_l2391_239114


namespace range_of_x_l2391_239147

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)

theorem range_of_x (a b x : ℝ) (h : a ≠ 0) (h₁ : abs (a + b) + abs (a - b) ≥ abs a * f x) :
  0 ≤ x ∧ x ≤ 4 :=
sorry

end range_of_x_l2391_239147


namespace housewife_oil_expense_l2391_239159

theorem housewife_oil_expense:
  ∃ M P R: ℝ, (R = 30) ∧ (0.8 * P = R) ∧ ((M / R) - (M / P) = 10) ∧ (M = 1500) :=
by
  sorry

end housewife_oil_expense_l2391_239159


namespace average_of_next_seven_consecutive_integers_l2391_239125

theorem average_of_next_seven_consecutive_integers
  (a b : ℕ)
  (hb : b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7) :
  ((b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5) + (b + 6)) / 7) = a + 6 :=
by
  sorry

end average_of_next_seven_consecutive_integers_l2391_239125


namespace average_salary_feb_mar_apr_may_l2391_239187

theorem average_salary_feb_mar_apr_may 
  (average_jan_feb_mar_apr : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_months_1 : ℤ)
  (total_months_2 : ℤ)
  (total_sum_jan_apr : average_jan_feb_mar_apr * (total_months_1:ℝ) = 32000)
  (january_salary: salary_jan = 4700)
  (may_salary: salary_may = 6500)
  (total_months_1_eq: total_months_1 = 4)
  (total_months_2_eq: total_months_2 = 4):
  average_jan_feb_mar_apr * (total_months_1:ℝ) - salary_jan + salary_may/total_months_2 = 8450 :=
by
  sorry

end average_salary_feb_mar_apr_may_l2391_239187


namespace cassie_nails_l2391_239130

-- Define the number of pets
def num_dogs := 4
def num_parrots := 8
def num_cats := 2
def num_rabbits := 6

-- Define the number of nails/claws/toes per pet
def nails_per_dog := 4 * 4
def common_claws_per_parrot := 2 * 3
def extra_toed_parrot_claws := 2 * 4
def toes_per_cat := 2 * 5 + 2 * 4
def rear_nails_per_rabbit := 2 * 5
def front_nails_per_rabbit := 3 + 4

-- Calculations
def total_dog_nails := num_dogs * nails_per_dog
def total_parrot_claws := 7 * common_claws_per_parrot + extra_toed_parrot_claws
def total_cat_toes := num_cats * toes_per_cat
def total_rabbit_nails := num_rabbits * (rear_nails_per_rabbit + front_nails_per_rabbit)

-- Total nails/claws/toes
def total_nails := total_dog_nails + total_parrot_claws + total_cat_toes + total_rabbit_nails

-- Theorem stating the problem
theorem cassie_nails : total_nails = 252 :=
by
  -- Here we would normally have the proof, but we'll skip it with sorry
  sorry

end cassie_nails_l2391_239130


namespace negation_of_proposition_l2391_239146

theorem negation_of_proposition :
  ¬(∀ n : ℤ, (∃ k : ℤ, n = 2 * k) → (∃ m : ℤ, n = 2 * m)) ↔ ∃ n : ℤ, (∃ k : ℤ, n = 2 * k) ∧ ¬(∃ m : ℤ, n = 2 * m) := 
sorry

end negation_of_proposition_l2391_239146


namespace distinct_units_digits_of_perfect_cube_l2391_239149

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l2391_239149


namespace cos_alpha_half_l2391_239158

theorem cos_alpha_half (α : ℝ) (h : Real.cos (Real.pi + α) = -1/2) : Real.cos α = 1/2 := 
by 
  sorry

end cos_alpha_half_l2391_239158


namespace arlene_average_pace_l2391_239155

theorem arlene_average_pace :
  ∃ pace : ℝ, pace = 24 / (6 - 0.75) ∧ pace = 4.57 := 
by
  sorry

end arlene_average_pace_l2391_239155


namespace range_of_m_perimeter_of_isosceles_triangle_l2391_239180

-- Define the variables for the lengths of the sides and the range of m
variables (AB BC AC : ℝ) (m : ℝ)

-- Conditions given in the problem
def triangle_conditions (AB BC : ℝ) (AC : ℝ) (m : ℝ) : Prop :=
  AB = 17 ∧ BC = 8 ∧ AC = 2 * m - 1

-- Proof that the range for m is between 5 and 13
theorem range_of_m (AB BC : ℝ) (m : ℝ) (h : triangle_conditions AB BC (2 * m - 1) m) : 
  5 < m ∧ m < 13 :=
by
  sorry

-- Proof that the perimeter is 42 when triangle is isosceles with given conditions
theorem perimeter_of_isosceles_triangle (AB BC AC : ℝ) (h : triangle_conditions AB BC AC 0) : 
  (AB = AC ∨ BC = AC) → (2 * AB + BC = 42) :=
by
  sorry

end range_of_m_perimeter_of_isosceles_triangle_l2391_239180


namespace sum_a5_a8_eq_six_l2391_239168

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) := ∀ {m n : ℕ}, a (m + 1) / a m = a (n + 1) / a n

theorem sum_a5_a8_eq_six (h_seq : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 6 * a 4 + 2 * a 8 * a 5 + a 9 * a 7 = 36) :
  a 5 + a 8 = 6 := 
sorry

end sum_a5_a8_eq_six_l2391_239168


namespace problem_1_problem_2_l2391_239101

-- Define the propositions p and q
def proposition_p (x a : ℝ) := x^2 - (a + 1/a) * x + 1 < 0
def proposition_q (x : ℝ) := x^2 - 4 * x + 3 ≤ 0

-- Problem 1: Given a = 2 and both p and q are true, find the range of x
theorem problem_1 (a : ℝ) (x : ℝ) (ha : a = 2) (hp : proposition_p x a) (hq : proposition_q x) :
  1 ≤ x ∧ x < 2 :=
sorry

-- Problem 2: Prove that if p is a necessary but not sufficient condition for q, then 3 < a
theorem problem_2 (a : ℝ)
  (h_ns : ∀ x, proposition_q x → proposition_p x a)
  (h_not_s : ∃ x, ¬ (proposition_q x → proposition_p x a)) :
  3 < a :=
sorry

end problem_1_problem_2_l2391_239101


namespace katie_books_ratio_l2391_239197

theorem katie_books_ratio
  (d : ℕ)
  (k : ℚ)
  (g : ℕ)
  (total_books : ℕ)
  (hd : d = 6)
  (hk : ∃ k : ℚ, k = (k : ℚ))
  (hg : g = 5 * (d + k * d))
  (ht : total_books = d + k * d + g)
  (htotal : total_books = 54) :
  k = 1 / 2 :=
by
  sorry

end katie_books_ratio_l2391_239197


namespace number_of_years_borrowed_l2391_239144

theorem number_of_years_borrowed (n : ℕ)
  (H1 : ∃ (p : ℕ), 5000 = p ∧ 4 = 4 ∧ n * 200 = 150)
  (H2 : ∃ (q : ℕ), 5000 = q ∧ 7 = 7 ∧ n * 350 = 150)
  : n = 1 :=
by
  sorry

end number_of_years_borrowed_l2391_239144


namespace gross_profit_value_l2391_239193

theorem gross_profit_value
  (SP : ℝ) (C : ℝ) (GP : ℝ)
  (h1 : SP = 81)
  (h2 : GP = 1.7 * C)
  (h3 : SP = C + GP) :
  GP = 51 :=
by
  sorry

end gross_profit_value_l2391_239193


namespace count_valid_n_l2391_239166

theorem count_valid_n :
  ∃ (count : ℕ), count = 9 ∧ 
  (∀ (n : ℕ), 0 < n ∧ n ≤ 2000 ∧ ∃ (k : ℕ), 21 * n = k * k ↔ count = 9) :=
by
  sorry

end count_valid_n_l2391_239166


namespace range_of_x_range_of_a_l2391_239119

-- Definitions of the conditions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- Part (1)
theorem range_of_x (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
by sorry

-- Part (2)
theorem range_of_a (a : ℝ) (h : ∀ x, ¬ (p x a) → ¬ (q x)) : 1 < a ∧ a ≤ 2 :=
by sorry

end range_of_x_range_of_a_l2391_239119


namespace find_a_range_l2391_239183

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * a * x^2 + 2 * x + 1
def f' (x a : ℝ) : ℝ := x^2 - a * x + 2

theorem find_a_range (a : ℝ) :
  (0 < x1) ∧ (x1 < 1) ∧ (1 < x2) ∧ (x2 < 3) ∧
  (f' 0 a > 0) ∧ (f' 1 a < 0) ∧ (f' 3 a > 0) →
  3 < a ∧ a < 11 / 3 :=
by
  sorry

end find_a_range_l2391_239183


namespace tileable_by_hook_l2391_239121

theorem tileable_by_hook (m n : ℕ) : 
  (∃ a b : ℕ, m = 3 * a ∧ (n = 4 * b ∨ n = 12 * b) ∨ 
              n = 3 * a ∧ (m = 4 * b ∨ m = 12 * b)) ↔ 12 ∣ (m * n) :=
by
  sorry

end tileable_by_hook_l2391_239121


namespace small_cone_altitude_l2391_239142

theorem small_cone_altitude (h_f: ℝ) (a_lb: ℝ) (a_ub: ℝ) : 
  h_f = 24 → a_lb = 225 * Real.pi → a_ub = 25 * Real.pi → ∃ h_s, h_s = 12 := 
by
  intros h1 h2 h3
  sorry

end small_cone_altitude_l2391_239142


namespace quadratic_roots_condition_l2391_239123

theorem quadratic_roots_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ < 1 ∧ x₂ > 1 ∧ x₁^2 + (3 * a - 1) * x₁ + a + 8 = 0 ∧
  x₂^2 + (3 * a - 1) * x₂ + a + 8 = 0) → a < -2 :=
by
  sorry

end quadratic_roots_condition_l2391_239123


namespace surface_area_cube_l2391_239112

theorem surface_area_cube (a : ℕ) (b : ℕ) (h : a = 2) : b = 54 :=
  by
  sorry

end surface_area_cube_l2391_239112


namespace contrapositive_proof_l2391_239160

theorem contrapositive_proof (a : ℝ) (h : a ≤ 2 → a^2 ≤ 4) : a > 2 → a^2 > 4 :=
by
  intros ha
  sorry

end contrapositive_proof_l2391_239160


namespace tank_capacity_l2391_239174

variable (C : ℕ) (t : ℕ)
variable (hC_nonzero : C > 0)
variable (ht_nonzero : t > 0)
variable (h_rate_pipe_A : t = C / 5)
variable (h_rate_pipe_B : t = C / 8)
variable (h_rate_inlet : t = 4 * 60)
variable (h_combined_time : t = 5 + 3)

theorem tank_capacity (C : ℕ) (h1 : C / 5 + C / 8 - 4 * 60 = 8) : C = 1200 := 
by
  sorry

end tank_capacity_l2391_239174


namespace blue_balls_in_box_l2391_239157

theorem blue_balls_in_box (total_balls : ℕ) (p_two_blue : ℚ) (b : ℕ) 
  (h1 : total_balls = 12) (h2 : p_two_blue = 1/22) 
  (h3 : (↑b / 12) * (↑(b-1) / 11) = p_two_blue) : b = 3 :=
by {
  sorry
}

end blue_balls_in_box_l2391_239157


namespace percentage_of_men_l2391_239169

theorem percentage_of_men (M : ℝ) 
  (h1 : 0 < M ∧ M < 1) 
  (h2 : 0.2 * M + 0.4 * (1 - M) = 0.3) : M = 0.5 :=
by
  sorry

end percentage_of_men_l2391_239169


namespace secret_sharing_problem_l2391_239184

theorem secret_sharing_problem : 
  ∃ n : ℕ, (3280 = (3^(n + 1) - 1) / 2) ∧ (n = 7) :=
by
  use 7
  sorry

end secret_sharing_problem_l2391_239184


namespace value_of_expression_l2391_239124

theorem value_of_expression : (180^2 - 150^2) / 30 = 330 := by
  sorry

end value_of_expression_l2391_239124


namespace product_of_values_l2391_239172

theorem product_of_values (x : ℚ) (hx : abs ((18 / x) + 4) = 3) :
  x = -18 ∨ x = -18 / 7 ∧ -18 * (-18 / 7) = 324 / 7 :=
by sorry

end product_of_values_l2391_239172


namespace find_angle_A_l2391_239109

variables {A B C a b c : ℝ}
variables {triangle_ABC : (2 * b - c) * (Real.cos A) = a * (Real.cos C)}

theorem find_angle_A (h : (2 * b - c) * (Real.cos A) = a * (Real.cos C)) : A = Real.pi / 3 :=
by
  sorry

end find_angle_A_l2391_239109


namespace stratified_sampling_l2391_239176

theorem stratified_sampling 
  (total_teachers : ℕ)
  (senior_teachers : ℕ)
  (intermediate_teachers : ℕ)
  (junior_teachers : ℕ)
  (sample_size : ℕ)
  (x y z : ℕ) 
  (h1 : total_teachers = 150)
  (h2 : senior_teachers = 45)
  (h3 : intermediate_teachers = 90)
  (h4 : junior_teachers = 15)
  (h5 : sample_size = 30)
  (h6 : x + y + z = sample_size)
  (h7 : x * 10 = sample_size / 5)
  (h8 : y * 10 = (2 * sample_size) / 5)
  (h9 : z * 10 = sample_size / 15) :
  (x, y, z) = (9, 18, 3) := sorry

end stratified_sampling_l2391_239176


namespace simple_interest_fraction_l2391_239129

theorem simple_interest_fraction (P : ℝ) (R T : ℝ) (hR: R = 4) (hT: T = 5) :
  (P * R * T / 100) / P = 1 / 5 := 
by
  sorry

end simple_interest_fraction_l2391_239129


namespace arithmetic_sequence_property_l2391_239115

variable {a : ℕ → ℕ}

-- Given condition in the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ d c : ℕ, ∀ n : ℕ, a n = c + n * d

def condition (a : ℕ → ℕ) : Prop := a 4 + a 8 = 16

-- Problem statement
theorem arithmetic_sequence_property (a : ℕ → ℕ)
  (h_arith_seq : arithmetic_sequence a)
  (h_condition : condition a) :
  a 2 + a 6 + a 10 = 24 :=
sorry

end arithmetic_sequence_property_l2391_239115


namespace geometric_sequence_fifth_term_l2391_239126

theorem geometric_sequence_fifth_term (a₁ r : ℤ) (n : ℕ) (h_a₁ : a₁ = 5) (h_r : r = -2) (h_n : n = 5) :
  (a₁ * r^(n-1) = 80) :=
by
  rw [h_a₁, h_r, h_n]
  sorry

end geometric_sequence_fifth_term_l2391_239126


namespace ratio_of_speeds_l2391_239177

-- Conditions
def total_distance_Eddy : ℕ := 200 + 240 + 300
def total_distance_Freddy : ℕ := 180 + 420
def total_time_Eddy : ℕ := 5
def total_time_Freddy : ℕ := 6

-- Average speeds
def avg_speed_Eddy (d t : ℕ) : ℚ := d / t
def avg_speed_Freddy (d t : ℕ) : ℚ := d / t

-- Ratio of average speeds
def ratio_speeds (s1 s2 : ℚ) : ℚ := s1 / s2

theorem ratio_of_speeds : 
  ratio_speeds (avg_speed_Eddy total_distance_Eddy total_time_Eddy) 
               (avg_speed_Freddy total_distance_Freddy total_time_Freddy) 
  = 37 / 25 := by
  -- Proof omitted
  sorry

end ratio_of_speeds_l2391_239177


namespace arithmetic_sequence_general_formula_and_extremum_l2391_239102

noncomputable def a (n : ℕ) : ℤ := sorry
def S (n : ℕ) : ℤ := sorry

theorem arithmetic_sequence_general_formula_and_extremum :
  (a 1 + a 4 = 8) ∧ (a 2 * a 3 = 15) →
  (∃ c d : ℤ, (∀ n : ℕ, a n = c * n + d) ∨ (∀ n : ℕ, a n = -c * n + d)) ∧
  ((∃ n_min : ℕ, n_min > 0 ∧ S n_min = 1) ∧ (∃ n_max : ℕ, n_max > 0 ∧ S n_max = 16)) :=
by
  sorry

end arithmetic_sequence_general_formula_and_extremum_l2391_239102


namespace team_a_faster_than_team_t_l2391_239162

-- Definitions for the conditions
def course_length : ℕ := 300
def team_t_speed : ℕ := 20
def team_t_time : ℕ := course_length / team_t_speed
def team_a_time : ℕ := team_t_time - 3
def team_a_speed : ℕ := course_length / team_a_time

-- Theorem to prove
theorem team_a_faster_than_team_t :
  team_a_speed - team_t_speed = 5 :=
by
  -- Define the necessary elements based on conditions
  let course_length := 300
  let team_t_speed := 20
  let team_t_time := course_length / team_t_speed -- 15 hours
  let team_a_time := team_t_time - 3 -- 12 hours
  let team_a_speed := course_length / team_a_time -- 25 mph
  
  -- Prove the statement
  have h : team_a_speed - team_t_speed = 5 := by sorry
  exact h

end team_a_faster_than_team_t_l2391_239162


namespace distinct_triples_l2391_239132

theorem distinct_triples (a b c : ℕ) (h₁: 2 * a - 1 = k₁ * b) (h₂: 2 * b - 1 = k₂ * c) (h₃: 2 * c - 1 = k₃ * a) :
  (a, b, c) = (7, 13, 25) ∨ (a, b, c) = (13, 25, 7) ∨ (a, b, c) = (25, 7, 13) := sorry

end distinct_triples_l2391_239132


namespace difference_is_correct_l2391_239181

-- Define the given constants and conditions
def purchase_price : ℕ := 1500
def down_payment : ℕ := 200
def monthly_payment : ℕ := 65
def number_of_monthly_payments : ℕ := 24

-- Define the derived quantities based on the given conditions
def total_monthly_payments : ℕ := monthly_payment * number_of_monthly_payments
def total_amount_paid : ℕ := down_payment + total_monthly_payments
def difference : ℕ := total_amount_paid - purchase_price

-- The statement to be proven
theorem difference_is_correct : difference = 260 := by
  sorry

end difference_is_correct_l2391_239181
