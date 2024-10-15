import Mathlib

namespace NUMINAMATH_GPT_parameterized_curve_is_line_l740_74053

theorem parameterized_curve_is_line :
  ∀ (t : ℝ), ∃ (m b : ℝ), y = 5 * ((x - 5) / 3) - 3 → y = (5 * x - 34) / 3 := 
by
  sorry

end NUMINAMATH_GPT_parameterized_curve_is_line_l740_74053


namespace NUMINAMATH_GPT_simplify_1_simplify_2_l740_74028

theorem simplify_1 (a b : ℤ) : 2 * a - (a + b) = a - b :=
by
  sorry

theorem simplify_2 (x y : ℤ) : (x^2 - 2 * y^2) - 2 * (3 * y^2 - 2 * x^2) = 5 * x^2 - 8 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_1_simplify_2_l740_74028


namespace NUMINAMATH_GPT_max_smaller_rectangles_l740_74021

theorem max_smaller_rectangles (a : ℕ) (d : ℕ) (n : ℕ) 
    (ha : a = 100) (hd : d = 2) (hn : n = 50) : 
    n + 1 * (n + 1) = 2601 :=
by
  rw [hn]
  norm_num
  sorry

end NUMINAMATH_GPT_max_smaller_rectangles_l740_74021


namespace NUMINAMATH_GPT_scientific_notation_conversion_l740_74024

theorem scientific_notation_conversion :
  (6.1 * 10^9 = (6.1 : ℝ) * 10^8) :=
sorry

end NUMINAMATH_GPT_scientific_notation_conversion_l740_74024


namespace NUMINAMATH_GPT_point_probability_in_cone_l740_74037

noncomputable def volume_of_cone (S : ℝ) (h : ℝ) : ℝ :=
  (1/3) * S * h

theorem point_probability_in_cone (P M : ℝ) (S_ABC : ℝ) (h_P h_M : ℝ)
  (h_volume_condition : volume_of_cone S_ABC h_P ≤ volume_of_cone S_ABC h_M / 3) :
  (1 - (2 / 3) ^ 3) = 19 / 27 :=
by
  sorry

end NUMINAMATH_GPT_point_probability_in_cone_l740_74037


namespace NUMINAMATH_GPT_profit_condition_maximize_profit_l740_74098

noncomputable def profit (x : ℕ) : ℕ := 
  (x + 10) * (300 - 10 * x)

theorem profit_condition (x : ℕ) : profit x = 3360 ↔ x = 2 ∨ x = 18 := by
  sorry

theorem maximize_profit : ∃ x, x = 10 ∧ profit x = 4000 := by
  sorry

end NUMINAMATH_GPT_profit_condition_maximize_profit_l740_74098


namespace NUMINAMATH_GPT_expected_value_of_game_l740_74019

theorem expected_value_of_game :
  let heads_prob := 1 / 4
  let tails_prob := 1 / 2
  let edge_prob := 1 / 4
  let gain_heads := 4
  let loss_tails := -3
  let gain_edge := 0
  let expected_value := heads_prob * gain_heads + tails_prob * loss_tails + edge_prob * gain_edge
  expected_value = -0.5 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_game_l740_74019


namespace NUMINAMATH_GPT_sin_add_pi_over_three_l740_74077

theorem sin_add_pi_over_three (α : ℝ) (h : Real.sin (α - 2 * Real.pi / 3) = 1 / 4) : 
  Real.sin (α + Real.pi / 3) = -1 / 4 := by
  sorry

end NUMINAMATH_GPT_sin_add_pi_over_three_l740_74077


namespace NUMINAMATH_GPT_pizza_consumption_order_l740_74007

theorem pizza_consumption_order :
  let total_slices := 168
  let alex_slices := (1/6) * total_slices
  let beth_slices := (2/7) * total_slices
  let cyril_slices := (1/3) * total_slices
  let eve_slices_initial := (1/8) * total_slices
  let dan_slices_initial := total_slices - (alex_slices + beth_slices + cyril_slices + eve_slices_initial)
  let eve_slices := eve_slices_initial + 2
  let dan_slices := dan_slices_initial - 2
  (cyril_slices > beth_slices ∧ beth_slices > eve_slices ∧ eve_slices > alex_slices ∧ alex_slices > dan_slices) :=
  sorry

end NUMINAMATH_GPT_pizza_consumption_order_l740_74007


namespace NUMINAMATH_GPT_katerina_weight_correct_l740_74025

-- We define the conditions
def total_weight : ℕ := 95
def alexa_weight : ℕ := 46

-- Define the proposition to prove: Katerina's weight is the total weight minus Alexa's weight, which should be 49.
theorem katerina_weight_correct : (total_weight - alexa_weight = 49) :=
by
  -- We use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_katerina_weight_correct_l740_74025


namespace NUMINAMATH_GPT_coords_with_respect_to_origin_l740_74067

/-- Given that the coordinates of point A are (-1, 2), prove that the coordinates of point A
with respect to the origin in the plane rectangular coordinate system xOy are (-1, 2). -/
theorem coords_with_respect_to_origin (A : (ℝ × ℝ)) (h : A = (-1, 2)) : A = (-1, 2) :=
sorry

end NUMINAMATH_GPT_coords_with_respect_to_origin_l740_74067


namespace NUMINAMATH_GPT_pears_left_l740_74010

theorem pears_left (keith_initial : ℕ) (keith_given : ℕ) (mike_initial : ℕ) 
  (hk : keith_initial = 47) (hg : keith_given = 46) (hm : mike_initial = 12) :
  (keith_initial - keith_given) + mike_initial = 13 := by
  sorry

end NUMINAMATH_GPT_pears_left_l740_74010


namespace NUMINAMATH_GPT_flu_infection_equation_l740_74094

theorem flu_infection_equation (x : ℕ) (h : 1 + x + x^2 = 36) : 1 + x + x^2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_flu_infection_equation_l740_74094


namespace NUMINAMATH_GPT_students_know_mothers_birthday_l740_74086

-- Defining the given conditions
def total_students : ℕ := 40
def A : ℕ := 10
def B : ℕ := 12
def C : ℕ := 22
def D : ℕ := 26

-- Statement to prove
theorem students_know_mothers_birthday : (B + C) = 22 :=
by
  sorry

end NUMINAMATH_GPT_students_know_mothers_birthday_l740_74086


namespace NUMINAMATH_GPT_hundredth_power_remainders_l740_74076

theorem hundredth_power_remainders (a : ℤ) : 
  (a % 5 = 0 → a^100 % 125 = 0) ∧ (a % 5 ≠ 0 → a^100 % 125 = 1) :=
by
  sorry

end NUMINAMATH_GPT_hundredth_power_remainders_l740_74076


namespace NUMINAMATH_GPT_pallet_weight_l740_74068

theorem pallet_weight (box_weight : ℕ) (num_boxes : ℕ) (total_weight : ℕ) 
  (h1 : box_weight = 89) (h2 : num_boxes = 3) : total_weight = 267 := by
  sorry

end NUMINAMATH_GPT_pallet_weight_l740_74068


namespace NUMINAMATH_GPT_real_function_as_sum_of_symmetric_graphs_l740_74033

theorem real_function_as_sum_of_symmetric_graphs (f : ℝ → ℝ) :
  ∃ (g h : ℝ → ℝ), (∀ x, g x + h x = f x) ∧ (∀ x, g x = g (-x)) ∧ (∀ x, h (1 + x) = h (1 - x)) :=
sorry

end NUMINAMATH_GPT_real_function_as_sum_of_symmetric_graphs_l740_74033


namespace NUMINAMATH_GPT_cube_volume_l740_74071

theorem cube_volume (length width : ℝ) (h_length : length = 48) (h_width : width = 72) :
  let area := length * width
  let side_length_in_inches := Real.sqrt (area / 6)
  let side_length_in_feet := side_length_in_inches / 12
  let volume := side_length_in_feet ^ 3
  volume = 8 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_l740_74071


namespace NUMINAMATH_GPT_range_of_b_l740_74048

variable (a b c : ℝ)

theorem range_of_b (h1 : a * c = b^2) (h2 : a + b + c = 3) : -3 ≤ b ∧ b ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_b_l740_74048


namespace NUMINAMATH_GPT_remainder_equality_l740_74072

theorem remainder_equality (P P' : ℕ) (h1 : P = P' + 10) 
  (h2 : P % 10 = 0) (h3 : P' % 10 = 0) : 
  ((P^2 - P'^2) % 10 = 0) :=
by
  sorry

end NUMINAMATH_GPT_remainder_equality_l740_74072


namespace NUMINAMATH_GPT_existence_of_E_l740_74015

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 6) + (y^2 / 2) = 1

def point_on_x_axis (E : ℝ × ℝ) : Prop := E.snd = 0

def ea_dot_eb_constant (E A B : ℝ × ℝ) : ℝ :=
  let ea := (A.fst - E.fst, A.snd)
  let eb := (B.fst - E.fst, B.snd)
  ea.fst * eb.fst + ea.snd * eb.snd

noncomputable def E : ℝ × ℝ := (7/3, 0)

noncomputable def const_value : ℝ := (-5/9)

theorem existence_of_E :
  (∃ E, point_on_x_axis E ∧
        (∀ A B, ellipse_eq A.fst A.snd ∧ ellipse_eq B.fst B.snd →
                  ea_dot_eb_constant E A B = const_value)) :=
  sorry

end NUMINAMATH_GPT_existence_of_E_l740_74015


namespace NUMINAMATH_GPT_asha_win_probability_l740_74087

theorem asha_win_probability :
  let P_Lose := (3 : ℚ) / 8
  let P_Tie := (1 : ℚ) / 4
  P_Lose + P_Tie < 1 → 1 - P_Lose - P_Tie = (3 : ℚ) / 8 := 
by
  sorry

end NUMINAMATH_GPT_asha_win_probability_l740_74087


namespace NUMINAMATH_GPT_notebook_problem_l740_74064

theorem notebook_problem
    (total_notebooks : ℕ)
    (cost_price_A : ℕ)
    (cost_price_B : ℕ)
    (total_cost_price : ℕ)
    (selling_price_A : ℕ)
    (selling_price_B : ℕ)
    (discount_A : ℕ)
    (profit_condition : ℕ)
    (x y m : ℕ) 
    (h1 : total_notebooks = 350)
    (h2 : cost_price_A = 12)
    (h3 : cost_price_B = 15)
    (h4 : total_cost_price = 4800)
    (h5 : selling_price_A = 20)
    (h6 : selling_price_B = 25)
    (h7 : discount_A = 30)
    (h8 : 12 * x + 15 * y = 4800)
    (h9 : x + y = 350)
    (h10 : selling_price_A * m + selling_price_B * m + (x - m) * selling_price_A * 7 / 10 + (y - m) * cost_price_B - total_cost_price ≥ profit_condition):
    x = 150 ∧ m ≥ 128 :=
by
    sorry

end NUMINAMATH_GPT_notebook_problem_l740_74064


namespace NUMINAMATH_GPT_sally_initial_orange_balloons_l740_74044

def initial_orange_balloons (found_orange : ℝ) (total_orange : ℝ) : ℝ := 
  total_orange - found_orange

theorem sally_initial_orange_balloons : initial_orange_balloons 2.0 11 = 9 := 
by
  sorry

end NUMINAMATH_GPT_sally_initial_orange_balloons_l740_74044


namespace NUMINAMATH_GPT_spongebob_price_l740_74046

variable (x : ℝ)

theorem spongebob_price (h : 30 * x + 12 * 1.5 = 78) : x = 2 :=
by
  -- Given condition: 30 * x + 12 * 1.5 = 78
  sorry

end NUMINAMATH_GPT_spongebob_price_l740_74046


namespace NUMINAMATH_GPT_morning_rowers_count_l740_74069

def number_afternoon_rowers : ℕ := 7
def total_rowers : ℕ := 60

def number_morning_rowers : ℕ :=
  total_rowers - number_afternoon_rowers

theorem morning_rowers_count :
  number_morning_rowers = 53 := by
  sorry

end NUMINAMATH_GPT_morning_rowers_count_l740_74069


namespace NUMINAMATH_GPT_symmetric_point_reflection_y_axis_l740_74004

theorem symmetric_point_reflection_y_axis (x y : ℝ) (h : (x, y) = (-2, 3)) :
  (-x, y) = (2, 3) :=
sorry

end NUMINAMATH_GPT_symmetric_point_reflection_y_axis_l740_74004


namespace NUMINAMATH_GPT_case_D_has_two_solutions_l740_74006

-- Definitions for the conditions of each case
structure CaseA :=
(b : ℝ) (A : ℝ) (B : ℝ)

structure CaseB :=
(a : ℝ) (c : ℝ) (B : ℝ)

structure CaseC :=
(a : ℝ) (b : ℝ) (A : ℝ)

structure CaseD :=
(a : ℝ) (b : ℝ) (A : ℝ)

-- Setting the values based on the given conditions
def caseA := CaseA.mk 10 45 70
def caseB := CaseB.mk 60 48 100
def caseC := CaseC.mk 14 16 45
def caseD := CaseD.mk 7 5 80

-- Define a function that checks if a case has two solutions
def has_two_solutions (a b c : ℝ) (A B : ℝ) : Prop := sorry

-- The theorem to prove that out of the given cases, only Case D has two solutions
theorem case_D_has_two_solutions :
  has_two_solutions caseA.b caseB.B caseC.a caseC.b caseC.A = false →
  has_two_solutions caseB.a caseB.c caseB.B caseC.b caseC.A = false →
  has_two_solutions caseC.a caseC.b caseC.A caseD.a caseD.b = false →
  has_two_solutions caseD.a caseD.b caseD.A caseA.b caseA.A = true :=
sorry

end NUMINAMATH_GPT_case_D_has_two_solutions_l740_74006


namespace NUMINAMATH_GPT_cube_bug_probability_l740_74088

theorem cube_bug_probability :
  ∃ n : ℕ, (∃ p : ℚ, p = 547/2187) ∧ (p = n/6561) ∧ n = 1641 :=
by
  sorry

end NUMINAMATH_GPT_cube_bug_probability_l740_74088


namespace NUMINAMATH_GPT_exists_unique_t_exists_m_pos_l740_74057

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem exists_unique_t (m : ℝ) (h : m = 1) : 
  ∃! (t : ℝ), t ∈ Set.Ioc (1 / 2) 1 ∧ deriv (f 1) t = 0 := sorry

theorem exists_m_pos : ∃ (m : ℝ), 0 < m ∧ m < 1 ∧ ∀ (x : ℝ), 0 < x → f m x > 0 := sorry

end NUMINAMATH_GPT_exists_unique_t_exists_m_pos_l740_74057


namespace NUMINAMATH_GPT_right_triangles_not_1000_l740_74014

-- Definitions based on the conditions
def numPoints := 100
def numDiametricallyOppositePairs := numPoints / 2
def rightTrianglesPerPair := numPoints - 2
def totalRightTriangles := numDiametricallyOppositePairs * rightTrianglesPerPair

-- Theorem stating the final evaluation of the problem
theorem right_triangles_not_1000 :
  totalRightTriangles ≠ 1000 :=
by
  -- calculation shows it's impossible
  sorry

end NUMINAMATH_GPT_right_triangles_not_1000_l740_74014


namespace NUMINAMATH_GPT_even_function_x_lt_0_l740_74080

noncomputable def f (x : ℝ) : ℝ :=
if h : x >= 0 then 2^x + 1 else 2^(-x) + 1

theorem even_function_x_lt_0 (x : ℝ) (hx : x < 0) : f x = 2^(-x) + 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_even_function_x_lt_0_l740_74080


namespace NUMINAMATH_GPT_evaluate_operations_l740_74023

def spadesuit (x y : ℝ) := (x + y) * (x - y)
def heartsuit (x y : ℝ) := x ^ 2 - y ^ 2

theorem evaluate_operations : spadesuit 5 (heartsuit 3 2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_operations_l740_74023


namespace NUMINAMATH_GPT_parabola_expression_l740_74032

theorem parabola_expression (a c : ℝ) (h1 : a = 1/4 ∨ a = -1/4) (h2 : ∀ x : ℝ, x = 1 → (a * x^2 + c = 0)) :
  (a = 1/4 ∧ c = -1/4) ∨ (a = -1/4 ∧ c = 1/4) :=
by {
  sorry
}

end NUMINAMATH_GPT_parabola_expression_l740_74032


namespace NUMINAMATH_GPT_vinegar_evaporation_rate_l740_74012

def percentage_vinegar_evaporates_each_year (x : ℕ) : Prop :=
  let initial_vinegar : ℕ := 100
  let vinegar_left_after_first_year : ℕ := initial_vinegar - x
  let vinegar_left_after_two_years : ℕ := vinegar_left_after_first_year * (100 - x) / 100
  vinegar_left_after_two_years = 64

theorem vinegar_evaporation_rate :
  ∃ x : ℕ, percentage_vinegar_evaporates_each_year x ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_vinegar_evaporation_rate_l740_74012


namespace NUMINAMATH_GPT_domain_of_f_l740_74049

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 2*x - 3)

theorem domain_of_f : 
  {x : ℝ | (x^2 - 2*x - 3) ≠ 0} = {x : ℝ | x < -1} ∪ {x : ℝ | -1 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l740_74049


namespace NUMINAMATH_GPT_least_distinct_values_l740_74054

variable (L : List Nat) (h_len : L.length = 2023) (mode : Nat) 
variable (h_mode_unique : ∀ x ∈ L, L.count x ≤ 15 → x = mode)
variable (h_mode_count : L.count mode = 15)

theorem least_distinct_values : ∃ k, k = 145 ∧ (∀ d ∈ L, List.count d L ≤ 15) :=
by
  sorry

end NUMINAMATH_GPT_least_distinct_values_l740_74054


namespace NUMINAMATH_GPT_max_diagonals_in_grid_l740_74096

-- Define the dimensions of the grid
def grid_width := 8
def grid_height := 5

-- Define the number of 1x2 rectangles
def number_of_1x2_rectangles := grid_width / 2 * grid_height

-- State the theorem
theorem max_diagonals_in_grid : number_of_1x2_rectangles = 20 := 
by 
  -- Simplify the expression
  sorry

end NUMINAMATH_GPT_max_diagonals_in_grid_l740_74096


namespace NUMINAMATH_GPT_sum_of_primes_no_solution_congruence_l740_74058

theorem sum_of_primes_no_solution_congruence :
  2 + 5 = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_primes_no_solution_congruence_l740_74058


namespace NUMINAMATH_GPT_Julia_played_with_kids_l740_74079

theorem Julia_played_with_kids :
  (∃ k : ℕ, k = 4) ∧ (∃ n : ℕ, n = 4 + 12) → (n = 16) :=
by
  sorry

end NUMINAMATH_GPT_Julia_played_with_kids_l740_74079


namespace NUMINAMATH_GPT_no_valid_placement_for_digits_on_45gon_l740_74043

theorem no_valid_placement_for_digits_on_45gon (f : Fin 45 → Fin 10) :
  ¬ ∀ (a b : Fin 10), a ≠ b → ∃ (i j : Fin 45), i ≠ j ∧ f i = a ∧ f j = b :=
by {
  sorry
}

end NUMINAMATH_GPT_no_valid_placement_for_digits_on_45gon_l740_74043


namespace NUMINAMATH_GPT_soccer_team_arrangements_l740_74084

theorem soccer_team_arrangements : 
  ∃ (n : ℕ), n = 2 * (Nat.factorial 11)^2 := 
sorry

end NUMINAMATH_GPT_soccer_team_arrangements_l740_74084


namespace NUMINAMATH_GPT_triangle_perpendicular_bisector_properties_l740_74051

variables {A B C A1 A2 B1 B2 C1 C2 : Type} (triangle : triangle A B C)
  (A1_perpendicular : dropping_perpendicular_to_bisector A )
  (A2_perpendicular : dropping_perpendicular_to_bisector A )
  (B1_perpendicular : dropping_perpendicular_to_bisector B )
  (B2_perpendicular : dropping_perpendicular_to_bisector B )
  (C1_perpendicular : dropping_perpendicular_to_bisector C )
  (C2_perpendicular : dropping_perpendicular_to_bisector C )
  
-- Defining required structures
structure triangle (A B C : Type) :=
  (AB BC CA : ℝ)

structure dropping_perpendicular_to_bisector (v : Type) :=
  (perpendicular_to_bisector : ℝ)

namespace triangle_properties

theorem triangle_perpendicular_bisector_properties :
  2 * (A1_perpendicular.perpendicular_to_bisector + A2_perpendicular.perpendicular_to_bisector + 
       B1_perpendicular.perpendicular_to_bisector + B2_perpendicular.perpendicular_to_bisector + 
       C1_perpendicular.perpendicular_to_bisector + C2_perpendicular.perpendicular_to_bisector) = 
  (triangle.AB + triangle.BC + triangle.CA) :=
sorry

end triangle_properties

end NUMINAMATH_GPT_triangle_perpendicular_bisector_properties_l740_74051


namespace NUMINAMATH_GPT_small_barrel_5_tons_l740_74078

def total_oil : ℕ := 95
def large_barrel_capacity : ℕ := 6
def small_barrel_capacity : ℕ := 5

theorem small_barrel_5_tons :
  ∃ (num_large_barrels num_small_barrels : ℕ),
  num_small_barrels = 1 ∧
  total_oil = (num_large_barrels * large_barrel_capacity) + (num_small_barrels * small_barrel_capacity) :=
by
  sorry

end NUMINAMATH_GPT_small_barrel_5_tons_l740_74078


namespace NUMINAMATH_GPT_ratio_of_girls_to_boys_l740_74045

-- Define conditions
def num_boys : ℕ := 40
def children_per_counselor : ℕ := 8
def num_counselors : ℕ := 20

-- Total number of children
def total_children : ℕ := num_counselors * children_per_counselor

-- Number of girls
def num_girls : ℕ := total_children - num_boys

-- The ratio of girls to boys
def girls_to_boys_ratio : ℚ := num_girls / num_boys

-- The theorem we need to prove
theorem ratio_of_girls_to_boys : girls_to_boys_ratio = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_girls_to_boys_l740_74045


namespace NUMINAMATH_GPT_polynomials_with_conditions_l740_74056

theorem polynomials_with_conditions (n : ℕ) (h_pos : 0 < n) :
  (∃ P : Polynomial ℤ, Polynomial.degree P = n ∧ 
      (∃ (k : Fin n → ℤ), Function.Injective k ∧ (∀ i, P.eval (k i) = n) ∧ P.eval 0 = 0)) ↔ 
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
sorry

end NUMINAMATH_GPT_polynomials_with_conditions_l740_74056


namespace NUMINAMATH_GPT_star_six_three_l740_74047

-- Definition of the operation
def star (a b : ℕ) : ℕ := 4 * a + 5 * b - 2 * a * b

-- Statement to prove
theorem star_six_three : star 6 3 = 3 := by
  sorry

end NUMINAMATH_GPT_star_six_three_l740_74047


namespace NUMINAMATH_GPT_question_correctness_l740_74020

theorem question_correctness (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
by sorry

end NUMINAMATH_GPT_question_correctness_l740_74020


namespace NUMINAMATH_GPT_area_ratio_is_correct_l740_74073

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_is_correct_l740_74073


namespace NUMINAMATH_GPT_multiplication_in_S_l740_74039

-- Define the set S as given in the conditions
variable (S : Set ℝ)

-- Condition 1: 1 ∈ S
def condition1 : Prop := 1 ∈ S

-- Condition 2: ∀ a b ∈ S, a - b ∈ S
def condition2 : Prop := ∀ a b : ℝ, a ∈ S → b ∈ S → (a - b) ∈ S

-- Condition 3: ∀ a ∈ S, a ≠ 0 → 1 / a ∈ S
def condition3 : Prop := ∀ a : ℝ, a ∈ S → a ≠ 0 → (1 / a) ∈ S

-- Theorem to prove: ∀ a b ∈ S, ab ∈ S
theorem multiplication_in_S (h1 : condition1 S) (h2 : condition2 S) (h3 : condition3 S) :
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a * b) ∈ S := 
  sorry

end NUMINAMATH_GPT_multiplication_in_S_l740_74039


namespace NUMINAMATH_GPT_P_finishes_in_15_minutes_more_l740_74000

variable (P Q : ℝ)

def rate_p := 1 / 4
def rate_q := 1 / 15
def time_together := 3
def total_job := 1

theorem P_finishes_in_15_minutes_more :
  let combined_rate := rate_p + rate_q
  let completed_job_in_3_hours := combined_rate * time_together
  let remaining_job := total_job - completed_job_in_3_hours
  let time_for_P_to_finish := remaining_job / rate_p
  let minutes_needed := time_for_P_to_finish * 60
  minutes_needed = 15 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_P_finishes_in_15_minutes_more_l740_74000


namespace NUMINAMATH_GPT_bowling_ball_weight_l740_74093

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 5 * b = 4 * c) 
  (h2 : 2 * c = 80) : 
  b = 32 :=
by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l740_74093


namespace NUMINAMATH_GPT_half_angle_second_quadrant_l740_74066

theorem half_angle_second_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (k * π + π / 4 < α / 2 ∧ α / 2 < k * π + π / 2) :=
sorry

end NUMINAMATH_GPT_half_angle_second_quadrant_l740_74066


namespace NUMINAMATH_GPT_perfume_price_l740_74026

variable (P : ℝ)

theorem perfume_price (h_increase : 1.10 * P = P + 0.10 * P)
    (h_decrease : 0.935 * P = 1.10 * P - 0.15 * 1.10 * P)
    (h_final_price : P - 0.935 * P = 78) : P = 1200 := 
by
  sorry

end NUMINAMATH_GPT_perfume_price_l740_74026


namespace NUMINAMATH_GPT_tan_585_eq_one_l740_74034

theorem tan_585_eq_one : Real.tan (585 * Real.pi / 180) = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_585_eq_one_l740_74034


namespace NUMINAMATH_GPT_instantaneous_velocity_at_4_seconds_l740_74003

-- Define the equation of motion
def s (t : ℝ) : ℝ := t^2 - 2 * t + 5

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 2 * t - 2

theorem instantaneous_velocity_at_4_seconds : v 4 = 6 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_4_seconds_l740_74003


namespace NUMINAMATH_GPT_reciprocal_inequality_l740_74097

variable (a b : ℝ)

theorem reciprocal_inequality (ha : a < 0) (hb : b > 0) : (1 / a) < (1 / b) := sorry

end NUMINAMATH_GPT_reciprocal_inequality_l740_74097


namespace NUMINAMATH_GPT_greatest_divisor_of_product_of_any_four_consecutive_integers_l740_74070

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_product_of_any_four_consecutive_integers_l740_74070


namespace NUMINAMATH_GPT_number_of_skirts_l740_74008

theorem number_of_skirts (T Ca Cs S : ℕ) (hT : T = 50) (hCa : Ca = 20) (hCs : Cs = 15) (hS : T - Ca = S * Cs) : S = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_skirts_l740_74008


namespace NUMINAMATH_GPT_infinite_integer_and_noninteger_terms_l740_74030

theorem infinite_integer_and_noninteger_terms (m : Nat) (h_m : m > 0) :
  ∃ (infinite_int_terms : Nat → Prop) (infinite_nonint_terms : Nat → Prop),
  (∀ n, ∃ k, infinite_int_terms k ∧ ∀ k, infinite_int_terms k → ∃ N, k = n + N + 1) ∧
  (∀ n, ∃ k, infinite_nonint_terms k ∧ ∀ k, infinite_nonint_terms k → ∃ N, k = n + N + 1) :=
sorry

end NUMINAMATH_GPT_infinite_integer_and_noninteger_terms_l740_74030


namespace NUMINAMATH_GPT_cost_to_produce_program_l740_74063

theorem cost_to_produce_program
  (advertisement_revenue : ℝ)
  (number_of_copies : ℝ)
  (price_per_copy : ℝ)
  (desired_profit : ℝ)
  (total_revenue : ℝ)
  (revenue_from_sales : ℝ)
  (cost_to_produce : ℝ) :
  advertisement_revenue = 15000 →
  number_of_copies = 35000 →
  price_per_copy = 0.5 →
  desired_profit = 8000 →
  total_revenue = advertisement_revenue + desired_profit →
  revenue_from_sales = number_of_copies * price_per_copy →
  total_revenue = revenue_from_sales + cost_to_produce →
  cost_to_produce = 5500 :=
by
  sorry

end NUMINAMATH_GPT_cost_to_produce_program_l740_74063


namespace NUMINAMATH_GPT_find_x_l740_74052

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 3) (h2 : y = 27) : x = 108 := by
  sorry

end NUMINAMATH_GPT_find_x_l740_74052


namespace NUMINAMATH_GPT_coincide_green_square_pairs_l740_74091

structure Figure :=
  (green_squares : ℕ)
  (red_triangles : ℕ)
  (blue_triangles : ℕ)

theorem coincide_green_square_pairs (f : Figure) (hs : f.green_squares = 4)
  (rt : f.red_triangles = 3) (bt : f.blue_triangles = 6)
  (gs_coincide : ∀ n, n ≤ f.green_squares ⟶ n = f.green_squares) 
  (rt_coincide : ∃ n, n = 2) (bt_coincide : ∃ n, n = 2) 
  (red_blue_pairs : ∃ n, n = 3) : 
  ∃ pairs, pairs = 4 :=
by 
  sorry

end NUMINAMATH_GPT_coincide_green_square_pairs_l740_74091


namespace NUMINAMATH_GPT_youngest_child_age_l740_74005

theorem youngest_child_age (x : ℝ) (h : x + (x + 1) + (x + 2) + (x + 3) = 12) : x = 1.5 :=
by sorry

end NUMINAMATH_GPT_youngest_child_age_l740_74005


namespace NUMINAMATH_GPT_least_plates_to_ensure_matching_pair_l740_74060

theorem least_plates_to_ensure_matching_pair
  (white_plates : ℕ)
  (green_plates : ℕ)
  (red_plates : ℕ)
  (pink_plates : ℕ)
  (purple_plates : ℕ)
  (h_white : white_plates = 2)
  (h_green : green_plates = 6)
  (h_red : red_plates = 8)
  (h_pink : pink_plates = 4)
  (h_purple : purple_plates = 10) :
  ∃ n, n = 6 :=
by
  sorry

end NUMINAMATH_GPT_least_plates_to_ensure_matching_pair_l740_74060


namespace NUMINAMATH_GPT_range_of_a_l740_74009

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 9 ^ x - 2 * 3 ^ x + a - 3 > 0) → a > 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l740_74009


namespace NUMINAMATH_GPT_conic_section_type_l740_74085

theorem conic_section_type (x y : ℝ) : 
  9 * x^2 - 36 * y^2 = 36 → 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_conic_section_type_l740_74085


namespace NUMINAMATH_GPT_find_constant_l740_74040

noncomputable def expr (x C : ℝ) : ℝ :=
  (x - 1) * (x - 3) * (x - 4) * (x - 6) + C

theorem find_constant :
  (∀ x : ℝ, expr x (-0.5625) ≥ 1) → expr 3.5 (-0.5625) = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_constant_l740_74040


namespace NUMINAMATH_GPT_complement_of_A_in_U_l740_74083

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set A
def A : Set ℕ := {3, 4, 5}

-- Statement to prove the complement of A with respect to U
theorem complement_of_A_in_U : U \ A = {1, 2, 6} := 
  by sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l740_74083


namespace NUMINAMATH_GPT_frac_3125_over_1024_gt_e_l740_74038

theorem frac_3125_over_1024_gt_e : (3125 : ℝ) / 1024 > Real.exp 1 := sorry

end NUMINAMATH_GPT_frac_3125_over_1024_gt_e_l740_74038


namespace NUMINAMATH_GPT_value_of_expression_l740_74090

theorem value_of_expression (x y z : ℝ) (hz : z ≠ 0) 
    (h1 : 2 * x - 3 * y - z = 0) 
    (h2 : x + 3 * y - 14 * z = 0) : 
    (x^2 + 3 * x * y) / (y^2 + z^2) = 7 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l740_74090


namespace NUMINAMATH_GPT_blithe_initial_toys_l740_74092

-- Define the conditions as given in the problem
def lost_toys : ℤ := 6
def found_toys : ℤ := 9
def final_toys : ℤ := 43

-- Define the problem statement to prove the initial number of toys
theorem blithe_initial_toys (T : ℤ) (h : T - lost_toys + found_toys = final_toys) : T = 40 :=
sorry

end NUMINAMATH_GPT_blithe_initial_toys_l740_74092


namespace NUMINAMATH_GPT_sum_series_1_to_60_l740_74013

-- Define what it means to be the sum of the first n natural numbers
def sum_n (n : Nat) : Nat := n * (n + 1) / 2

theorem sum_series_1_to_60 : sum_n 60 = 1830 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_1_to_60_l740_74013


namespace NUMINAMATH_GPT_slope_angle_correct_l740_74089

def parametric_line (α : ℝ) : Prop :=
  α = 50 * (Real.pi / 180)

theorem slope_angle_correct : ∀ (t : ℝ),
  parametric_line 50 →
  ∀ α : ℝ, α = 140 * (Real.pi / 180) :=
by
  intro t
  intro h
  intro α
  sorry

end NUMINAMATH_GPT_slope_angle_correct_l740_74089


namespace NUMINAMATH_GPT_average_sales_l740_74018

/-- The sales for the first five months -/
def sales_first_five_months := [5435, 5927, 5855, 6230, 5562]

/-- The sale for the sixth month -/
def sale_sixth_month := 3991

/-- The correct average sale to be achieved -/
def correct_average_sale := 5500

theorem average_sales :
  (sales_first_five_months.sum + sale_sixth_month) / 6 = correct_average_sale :=
by
  sorry

end NUMINAMATH_GPT_average_sales_l740_74018


namespace NUMINAMATH_GPT_greatest_air_conditioning_but_no_racing_stripes_l740_74001

variable (total_cars : ℕ) (no_air_conditioning_cars : ℕ) (at_least_racing_stripes_cars : ℕ)
variable (total_cars_eq : total_cars = 100)
variable (no_air_conditioning_cars_eq : no_air_conditioning_cars = 37)
variable (at_least_racing_stripes_cars_ge : at_least_racing_stripes_cars ≥ 51)

theorem greatest_air_conditioning_but_no_racing_stripes
  (total_cars_eq : total_cars = 100)
  (no_air_conditioning_cars_eq : no_air_conditioning_cars = 37)
  (at_least_racing_stripes_cars_ge : at_least_racing_stripes_cars ≥ 51) :
  ∃ max_air_conditioning_no_racing_stripes : ℕ, max_air_conditioning_no_racing_stripes = 12 :=
by
  sorry

end NUMINAMATH_GPT_greatest_air_conditioning_but_no_racing_stripes_l740_74001


namespace NUMINAMATH_GPT_johns_donation_is_correct_l740_74022

/-
Conditions:
1. Alice, Bob, and Carol donated different amounts.
2. The ratio of Alice's, Bob's, and Carol's donations is 3:2:5.
3. The sum of Alice's and Bob's donations is $120.
4. The average contribution increases by 50% and reaches $75 per person after John donates.

The statement to prove:
John's donation is $240.
-/

def donations_ratio : ℕ × ℕ × ℕ := (3, 2, 5)
def sum_Alice_Bob : ℕ := 120
def new_avg_after_john : ℕ := 75
def num_people_before_john : ℕ := 3
def avg_increase_factor : ℚ := 1.5

theorem johns_donation_is_correct (A B C J : ℕ) 
  (h1 : A * 2 = B * 3) 
  (h2 : B * 5 = C * 2) 
  (h3 : A + B = sum_Alice_Bob) 
  (h4 : (A + B + C) / num_people_before_john = 80) 
  (h5 : ((A + B + C + J) / (num_people_before_john + 1)) = new_avg_after_john) :
  J = 240 := 
sorry

end NUMINAMATH_GPT_johns_donation_is_correct_l740_74022


namespace NUMINAMATH_GPT_question1_effective_purification_16days_question2_min_mass_optimal_purification_l740_74061

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then x^2 / 16 + 2
else if x > 4 then (x + 14) / (2 * x - 2)
else 0

-- Effective Purification Conditions
def effective_purification (m : ℝ) (x : ℝ) : Prop := m * f x ≥ 4

-- Optimal Purification Conditions
def optimal_purification (m : ℝ) (x : ℝ) : Prop := 4 ≤ m * f x ∧ m * f x ≤ 10

-- Proof for Question 1
theorem question1_effective_purification_16days (x : ℝ) (hx : 0 < x ∧ x ≤ 16) :
  effective_purification 4 x :=
by sorry

-- Finding Minimum m for Optimal Purification within 7 days
theorem question2_min_mass_optimal_purification :
  ∃ m : ℝ, (16 / 7 ≤ m ∧ m ≤ 10 / 3) ∧ ∀ (x : ℝ), (0 < x ∧ x ≤ 7) → optimal_purification m x :=
by sorry

end NUMINAMATH_GPT_question1_effective_purification_16days_question2_min_mass_optimal_purification_l740_74061


namespace NUMINAMATH_GPT_sarah_friends_apples_l740_74002

-- Definitions of initial conditions
def initial_apples : ℕ := 25
def left_apples : ℕ := 3
def apples_given_teachers : ℕ := 16
def apples_eaten : ℕ := 1

-- Theorem that states the number of friends who received apples
theorem sarah_friends_apples :
  (initial_apples - left_apples - apples_given_teachers - apples_eaten = 5) :=
by
  sorry

end NUMINAMATH_GPT_sarah_friends_apples_l740_74002


namespace NUMINAMATH_GPT_study_group_books_l740_74042

theorem study_group_books (x n : ℕ) (h1 : n = 5 * x - 2) (h2 : n = 4 * x + 3) : x = 5 ∧ n = 23 := by
  sorry

end NUMINAMATH_GPT_study_group_books_l740_74042


namespace NUMINAMATH_GPT_rectangle_area_l740_74059

/-- A figure is formed by a triangle and a rectangle, using 60 equal sticks.
Each side of the triangle uses 6 sticks, and each stick measures 5 cm in length.
Prove that the area of the rectangle is 2250 cm². -/
theorem rectangle_area (sticks_total : ℕ) (sticks_per_side_triangle : ℕ) (stick_length_cm : ℕ)
    (sticks_used_triangle : ℕ) (sticks_left_rectangle : ℕ) (sticks_per_width_rectangle : ℕ)
    (width_sticks_rectangle : ℕ) (length_sticks_rectangle : ℕ) (width_cm : ℕ) (length_cm : ℕ)
    (area_rectangle : ℕ) 
    (h_sticks_total : sticks_total = 60)
    (h_sticks_per_side_triangle : sticks_per_side_triangle = 6)
    (h_stick_length_cm : stick_length_cm = 5)
    (h_sticks_used_triangle  : sticks_used_triangle = sticks_per_side_triangle * 3)
    (h_sticks_left_rectangle : sticks_left_rectangle = sticks_total - sticks_used_triangle)
    (h_sticks_per_width_rectangle : sticks_per_width_rectangle = 6 * 2) 
    (h_width_sticks_rectangle : width_sticks_rectangle = 6)
    (h_length_sticks_rectangle : length_sticks_rectangle = (sticks_left_rectangle - sticks_per_width_rectangle) / 2)
    (h_width_cm : width_cm = width_sticks_rectangle * stick_length_cm)
    (h_length_cm : length_cm = length_sticks_rectangle * stick_length_cm)
    (h_area_rectangle : area_rectangle = width_cm * length_cm) :
    area_rectangle = 2250 := 
by sorry

end NUMINAMATH_GPT_rectangle_area_l740_74059


namespace NUMINAMATH_GPT_am_gm_hm_inequality_l740_74099

theorem am_gm_hm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : a ≠ c) : 
  (a + b + c) / 3 > (a * b * c) ^ (1 / 3) ∧ (a * b * c) ^ (1 / 3) > 3 * a * b * c / (a * b + b * c + c * a) :=
by
  sorry

end NUMINAMATH_GPT_am_gm_hm_inequality_l740_74099


namespace NUMINAMATH_GPT_max_value_of_x_plus_y_l740_74081

theorem max_value_of_x_plus_y (x y : ℝ) (h : x^2 / 16 + y^2 / 9 = 1) : x + y ≤ 5 :=
sorry

end NUMINAMATH_GPT_max_value_of_x_plus_y_l740_74081


namespace NUMINAMATH_GPT_S_11_eq_zero_l740_74027

noncomputable def S (n : ℕ) : ℝ := sorry
variable (a_n : ℕ → ℝ) (d : ℝ)
variable (h1 : ∀ n, a_n (n+1) = a_n n + d) -- common difference d ≠ 0
variable (h2 : S 5 = S 6)

theorem S_11_eq_zero (h_nonzero : d ≠ 0) : S 11 = 0 := by
  sorry

end NUMINAMATH_GPT_S_11_eq_zero_l740_74027


namespace NUMINAMATH_GPT_find_number_l740_74055

theorem find_number : ∃ n : ℕ, ∃ q : ℕ, ∃ r : ℕ, q = 6 ∧ r = 4 ∧ n = 9 * q + r ∧ n = 58 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l740_74055


namespace NUMINAMATH_GPT_determine_friends_l740_74035

inductive Grade
| first
| second
| third
| fourth

inductive Name
| Petya
| Kolya
| Alyosha
| Misha
| Dima
| Borya
| Vasya

inductive Surname
| Ivanov
| Krylov
| Petrov
| Orlov

structure Friend :=
  (name : Name)
  (surname : Surname)
  (grade : Grade)

def friends : List Friend :=
  [ {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first},
    {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second},
    {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third},
    {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ]

theorem determine_friends : ∃ l : List Friend, 
  {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first} ∈ l ∧
  {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second} ∈ l ∧
  {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third} ∈ l ∧
  {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ∈ l :=
by 
  use friends
  repeat { simp [friends] }


end NUMINAMATH_GPT_determine_friends_l740_74035


namespace NUMINAMATH_GPT_balls_in_boxes_l740_74062

-- Definition of the combinatorial function
def combinations (n k : ℕ) : ℕ :=
  n.choose k

-- Problem statement in Lean
theorem balls_in_boxes :
  combinations 7 2 = 21 :=
by
  -- Since the proof is not required here, we place sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l740_74062


namespace NUMINAMATH_GPT_price_difference_l740_74082

noncomputable def original_price (final_sale_price discount : ℝ) := final_sale_price / (1 - discount)

noncomputable def after_price_increase (price after_increase : ℝ) := price * (1 + after_increase)

theorem price_difference (final_sale_price : ℝ) (discount : ℝ) (price_increase : ℝ) 
    (h1 : final_sale_price = 85) (h2 : discount = 0.15) (h3 : price_increase = 0.25) : 
    after_price_increase final_sale_price price_increase - original_price final_sale_price discount = 6.25 := 
by 
    sorry

end NUMINAMATH_GPT_price_difference_l740_74082


namespace NUMINAMATH_GPT_positive_difference_l740_74095

theorem positive_difference (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 3 * y - 4 * x = 9) : 
  abs (y - x) = 129 / 7 - (30 - 129 / 7) := 
by {
  sorry
}

end NUMINAMATH_GPT_positive_difference_l740_74095


namespace NUMINAMATH_GPT_distance_between_points_l740_74011

theorem distance_between_points :
  let A : ℝ × ℝ × ℝ := (1, -2, 3)
  let B : ℝ × ℝ × ℝ := (1, 2, 3)
  let C : ℝ × ℝ × ℝ := (1, 2, -3)
  dist B C = 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l740_74011


namespace NUMINAMATH_GPT_max_p_plus_q_l740_74016

theorem max_p_plus_q (p q : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → 2 * p * x^2 + q * x - p + 1 ≥ 0) : p + q ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_p_plus_q_l740_74016


namespace NUMINAMATH_GPT_real_solution_count_l740_74036

theorem real_solution_count : 
  ∃ (n : ℕ), n = 1 ∧
    ∀ x : ℝ, 
      (3 * x / (x ^ 2 + 2 * x + 4) + 4 * x / (x ^ 2 - 4 * x + 4) = 1) ↔ (x = 2) :=
by
  sorry

end NUMINAMATH_GPT_real_solution_count_l740_74036


namespace NUMINAMATH_GPT_digit_sum_square_l740_74029

theorem digit_sum_square (n : ℕ) (hn : 0 < n) :
  let A := (4 * (10 ^ (2 * n) - 1)) / 9
  let B := (8 * (10 ^ n - 1)) / 9
  ∃ k : ℕ, A + 2 * B + 4 = k ^ 2 := 
by
  sorry

end NUMINAMATH_GPT_digit_sum_square_l740_74029


namespace NUMINAMATH_GPT_find_x_values_l740_74031

theorem find_x_values (x y : ℕ) (hx: x > y) (hy: y > 0) (h: x + y + x * y = 101) :
  x = 50 ∨ x = 16 :=
sorry

end NUMINAMATH_GPT_find_x_values_l740_74031


namespace NUMINAMATH_GPT_digit_1C3_multiple_of_3_l740_74041

theorem digit_1C3_multiple_of_3 :
  (∃ C : Fin 10, (1 + C.val + 3) % 3 = 0) ∧
  (∀ C : Fin 10, (1 + C.val + 3) % 3 = 0 → (C.val = 2 ∨ C.val = 5 ∨ C.val = 8)) :=
by
  sorry

end NUMINAMATH_GPT_digit_1C3_multiple_of_3_l740_74041


namespace NUMINAMATH_GPT_digit_to_make_multiple_of_5_l740_74065

theorem digit_to_make_multiple_of_5 (d : ℕ) (h : 0 ≤ d ∧ d ≤ 9) 
  (N := 71360 + d) : (N % 5 = 0) → (d = 0 ∨ d = 5) :=
by
  sorry

end NUMINAMATH_GPT_digit_to_make_multiple_of_5_l740_74065


namespace NUMINAMATH_GPT_unique_real_solution_l740_74074

noncomputable def cubic_eq (b x : ℝ) : ℝ :=
  x^3 - b * x^2 - 3 * b * x + b^2 - 2

theorem unique_real_solution (b : ℝ) :
  (∃! x : ℝ, cubic_eq b x = 0) ↔ b = 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_real_solution_l740_74074


namespace NUMINAMATH_GPT_computation_one_computation_two_l740_74017

-- Proof problem (1)
theorem computation_one :
  (-2)^3 + |(-3)| - Real.tan (Real.pi / 4) = -6 := by
  sorry

-- Proof problem (2)
theorem computation_two (a : ℝ) :
  (a + 2)^2 - a * (a - 4) = 8 * a + 4 := by
  sorry

end NUMINAMATH_GPT_computation_one_computation_two_l740_74017


namespace NUMINAMATH_GPT_daily_profit_1200_impossible_daily_profit_1600_l740_74050

-- Definitions of given conditions
def avg_shirts_sold_per_day : ℕ := 30
def profit_per_shirt : ℕ := 40

-- Function for the number of shirts sold given a price reduction
def shirts_sold (x : ℕ) : ℕ := avg_shirts_sold_per_day + 2 * x

-- Function for the profit per shirt given a price reduction
def new_profit_per_shirt (x : ℕ) : ℕ := profit_per_shirt - x

-- Function for the daily profit given a price reduction
def daily_profit (x : ℕ) : ℕ := (new_profit_per_shirt x) * (shirts_sold x)

-- Proving the desired conditions in Lean

-- Part 1: Prove that reducing the price by 25 yuan results in a daily profit of 1200 yuan
theorem daily_profit_1200 (x : ℕ) : daily_profit x = 1200 ↔ x = 25 :=
by
  { sorry }

-- Part 2: Prove that a daily profit of 1600 yuan is not achievable
theorem impossible_daily_profit_1600 (x : ℕ) : daily_profit x ≠ 1600 :=
by
  { sorry }

end NUMINAMATH_GPT_daily_profit_1200_impossible_daily_profit_1600_l740_74050


namespace NUMINAMATH_GPT_correct_statements_count_l740_74075

-- Definitions for each condition
def is_output_correct (stmt : String) : Prop :=
  stmt = "PRINT a, b, c"

def is_input_correct (stmt : String) : Prop :=
  stmt = "INPUT \"x=3\""

def is_assignment_correct_1 (stmt : String) : Prop :=
  stmt = "A=3"

def is_assignment_correct_2 (stmt : String) : Prop :=
  stmt = "A=B ∧ B=C"

-- The main theorem to be proven
theorem correct_statements_count (stmt1 stmt2 stmt3 stmt4 : String) :
  stmt1 = "INPUT a, b, c" → stmt2 = "INPUT x=3" → stmt3 = "3=A" → stmt4 = "A=B=C" →
  (¬ is_output_correct stmt1 ∧ ¬ is_input_correct stmt2 ∧ ¬ is_assignment_correct_1 stmt3 ∧ ¬ is_assignment_correct_2 stmt4) →
  0 = 0 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_correct_statements_count_l740_74075
