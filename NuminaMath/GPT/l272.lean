import Mathlib

namespace NUMINAMATH_GPT_a_alone_days_l272_27287

theorem a_alone_days 
  (B_days : ℕ)
  (B_days_eq : B_days = 8)
  (C_payment : ℝ)
  (C_payment_eq : C_payment = 450)
  (total_payment : ℝ)
  (total_payment_eq : total_payment = 3600)
  (combined_days : ℕ)
  (combined_days_eq : combined_days = 3)
  (combined_rate_eq : (1 / A + 1 / B_days + C = 1 / combined_days)) 
  (rate_proportion : (1 / A) / (1 / B_days) = 7 / 1) 
  : A = 56 :=
sorry

end NUMINAMATH_GPT_a_alone_days_l272_27287


namespace NUMINAMATH_GPT_correct_operation_l272_27299

theorem correct_operation :
  (∀ a : ℝ, a^4 * a^3 = a^7)
  ∧ (∀ a : ℝ, (a^2)^3 ≠ a^5)
  ∧ (∀ a : ℝ, 3 * a^2 - a^2 ≠ 2)
  ∧ (∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_operation_l272_27299


namespace NUMINAMATH_GPT_min_slope_at_a_half_l272_27291

theorem min_slope_at_a_half (a : ℝ) (h : 0 < a) :
  (∀ b : ℝ, 0 < b → 4 * b + 1 / b ≥ 4) → (4 * a + 1 / a = 4) → a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_slope_at_a_half_l272_27291


namespace NUMINAMATH_GPT_at_least_one_not_less_than_one_l272_27258

theorem at_least_one_not_less_than_one (x : ℝ) (a b c : ℝ) 
  (ha : a = x^2 + 1/2) 
  (hb : b = 2 - x) 
  (hc : c = x^2 - x + 1) : 
  (1 ≤ a) ∨ (1 ≤ b) ∨ (1 ≤ c) := 
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_one_l272_27258


namespace NUMINAMATH_GPT_sum_of_roots_l272_27254

theorem sum_of_roots {a b c d : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (h1 : c + d = -a) (h2 : c * d = b) (h3 : a + b = -c) (h4 : a * b = d) : 
    a + b + c + d = -2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l272_27254


namespace NUMINAMATH_GPT_solve_linear_equation_l272_27227

theorem solve_linear_equation (a b x : ℝ) (h : a - b = 0) (ha : a ≠ 0) : ax + b = 0 ↔ x = -1 :=
by sorry

end NUMINAMATH_GPT_solve_linear_equation_l272_27227


namespace NUMINAMATH_GPT_cities_below_50000_l272_27280

theorem cities_below_50000 (p1 p2 : ℝ) (h1 : p1 = 20) (h2: p2 = 65) :
  p1 + p2 = 85 := 
  by sorry

end NUMINAMATH_GPT_cities_below_50000_l272_27280


namespace NUMINAMATH_GPT_triangle_side_ratio_l272_27234

theorem triangle_side_ratio (a b c: ℝ) (A B C: ℝ) (h1: b * Real.cos C + c * Real.cos B = 2 * b) :
  a / b = 2 :=
sorry

end NUMINAMATH_GPT_triangle_side_ratio_l272_27234


namespace NUMINAMATH_GPT_document_completion_time_l272_27266

-- Define the typing rates for different typists
def fast_typist_rate := 1 / 4
def slow_typist_rate := 1 / 9
def additional_typist_rate := 1 / 4

-- Define the number of typists
def num_fast_typists := 2
def num_slow_typists := 3
def num_additional_typists := 2

-- Define the distraction time loss per typist every 30 minutes
def distraction_loss := 1 / 6

-- Define the combined rate without distractions
def combined_rate : ℚ :=
  (num_fast_typists * fast_typist_rate) +
  (num_slow_typists * slow_typist_rate) +
  (num_additional_typists * additional_typist_rate)

-- Define the distraction rate loss per hour (two distractions per hour)
def distraction_rate_loss_per_hour := 2 * distraction_loss

-- Define the effective combined rate considering distractions
def effective_combined_rate : ℚ := combined_rate - distraction_rate_loss_per_hour

-- Prove that the document is completed in 1 hour with the effective rate
theorem document_completion_time :
  effective_combined_rate = 1 :=
sorry

end NUMINAMATH_GPT_document_completion_time_l272_27266


namespace NUMINAMATH_GPT_minimum_value_of_expression_l272_27296

noncomputable def expression (x y : ℝ) : ℝ :=
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) /
  (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2)

theorem minimum_value_of_expression : 
  ∃ x y : ℝ, expression x y = 5/32 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l272_27296


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l272_27275

def p (x : ℝ) : Prop := x ^ 2 = 3 * x + 4
def q (x : ℝ) : Prop := x = Real.sqrt (3 * x + 4)

theorem necessary_but_not_sufficient (x : ℝ) : (p x → q x) ∧ ¬ (q x → p x) := by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l272_27275


namespace NUMINAMATH_GPT_number_of_zeros_of_f_is_3_l272_27282

def f (x : ℝ) : ℝ := x^3 - 64 * x

theorem number_of_zeros_of_f_is_3 : ∃ x1 x2 x3, (f x1 = 0) ∧ (f x2 = 0) ∧ (f x3 = 0) ∧ (x1 ≠ x2) ∧ (x2 ≠ x3) ∧ (x1 ≠ x3) :=
by
  sorry

end NUMINAMATH_GPT_number_of_zeros_of_f_is_3_l272_27282


namespace NUMINAMATH_GPT_incorrect_regression_statement_incorrect_statement_proof_l272_27210

-- Define the regression equation and the statement about y and x
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

-- Proof statement: given the regression equation, show that when x increases by one unit, y decreases by 5 units on average
theorem incorrect_regression_statement : 
  (regression_equation (x + 1) = regression_equation x + (-5)) :=
by sorry

-- Proof statement: prove that the statement "when the variable x increases by one unit, y increases by 5 units on average" is incorrect
theorem incorrect_statement_proof :
  ¬ (regression_equation (x + 1) = regression_equation x + 5) :=
by sorry  

end NUMINAMATH_GPT_incorrect_regression_statement_incorrect_statement_proof_l272_27210


namespace NUMINAMATH_GPT_height_of_circular_segment_l272_27276

theorem height_of_circular_segment (d a : ℝ) (h : ℝ) :
  (h = (d - Real.sqrt (d^2 - a^2)) / 2) ↔ 
  ((a / 2)^2 + (d / 2 - h)^2 = (d / 2)^2) :=
sorry

end NUMINAMATH_GPT_height_of_circular_segment_l272_27276


namespace NUMINAMATH_GPT_area_of_sector_l272_27225

theorem area_of_sector
  (θ : ℝ) (l : ℝ) (r : ℝ := l / θ)
  (h1 : θ = 2)
  (h2 : l = 4) :
  1 / 2 * r^2 * θ = 4 :=
by
  sorry

end NUMINAMATH_GPT_area_of_sector_l272_27225


namespace NUMINAMATH_GPT_complement_union_M_N_eq_set_l272_27239

open Set

-- Define the universe U
def U : Set (ℝ × ℝ) := { p | True }

-- Define the set M
def M : Set (ℝ × ℝ) := { p | (p.snd - 3) / (p.fst - 2) ≠ 1 }

-- Define the set N
def N : Set (ℝ × ℝ) := { p | p.snd ≠ p.fst + 1 }

-- Define the complement of M ∪ N in U
def complement_MN : Set (ℝ × ℝ) := compl (M ∪ N)

theorem complement_union_M_N_eq_set : complement_MN = { (2, 3) } :=
  sorry

end NUMINAMATH_GPT_complement_union_M_N_eq_set_l272_27239


namespace NUMINAMATH_GPT_correct_calculation_l272_27279

-- Definition of the expressions in the problem
def exprA (a : ℝ) : Prop := 2 * a^2 + a^3 = 3 * a^5
def exprB (x y : ℝ) : Prop := ((-3 * x^2 * y)^2 / (x * y) = 9 * x^5 * y^3)
def exprC (b : ℝ) : Prop := (2 * b^2)^3 = 8 * b^6
def exprD (x : ℝ) : Prop := (2 * x * 3 * x^5 = 6 * x^5)

-- The proof problem
theorem correct_calculation (a x y b : ℝ) : exprC b ∧ ¬ exprA a ∧ ¬ exprB x y ∧ ¬ exprD x :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_calculation_l272_27279


namespace NUMINAMATH_GPT_remainder_when_divided_by_9_l272_27204

variable (k : ℕ)

theorem remainder_when_divided_by_9 :
  (∃ k, k % 5 = 2 ∧ k % 6 = 3 ∧ k % 8 = 7 ∧ k < 100) →
  k % 9 = 6 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_9_l272_27204


namespace NUMINAMATH_GPT_perfect_square_trinomial_l272_27235

-- Define the conditions
theorem perfect_square_trinomial (k : ℤ) : 
  ∃ (a b : ℤ), (a^2 = 1 ∧ b^2 = 16 ∧ (x^2 + k * x * y + 16 * y^2 = (a * x + b * y)^2)) ↔ (k = 8 ∨ k = -8) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l272_27235


namespace NUMINAMATH_GPT_least_m_lcm_l272_27253

theorem least_m_lcm (m : ℕ) (h : m > 0) : Nat.lcm 15 m = Nat.lcm 42 m → m = 70 := by
  sorry

end NUMINAMATH_GPT_least_m_lcm_l272_27253


namespace NUMINAMATH_GPT_arithmetic_square_root_l272_27245

theorem arithmetic_square_root (n : ℝ) (h : (-5)^2 = n) : Real.sqrt n = 5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_l272_27245


namespace NUMINAMATH_GPT_parabola_unique_intersection_x_axis_l272_27222

theorem parabola_unique_intersection_x_axis (m : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + m = 0 ∧ ∀ y, y^2 - 6*y + m = 0 → y = x) → m = 9 :=
by
  sorry

end NUMINAMATH_GPT_parabola_unique_intersection_x_axis_l272_27222


namespace NUMINAMATH_GPT_angle_in_second_quadrant_l272_27271

theorem angle_in_second_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.sin α > 0) : 
    α ∈ Set.Ioo (π / 2) π := 
    sorry

end NUMINAMATH_GPT_angle_in_second_quadrant_l272_27271


namespace NUMINAMATH_GPT_volume_relation_l272_27244

theorem volume_relation 
  (r h : ℝ) 
  (heightC_eq_three_times_radiusD : h = 3 * r)
  (radiusC_eq_heightD : r = h)
  (volumeD_eq_three_times_volumeC : ∀ (π : ℝ), 3 * (π * h^2 * r) = π * r^2 * h) :
  3 = (3 : ℝ) := 
by
  sorry

end NUMINAMATH_GPT_volume_relation_l272_27244


namespace NUMINAMATH_GPT_isosceles_base_length_l272_27230

theorem isosceles_base_length :
  ∀ (equilateral_perimeter isosceles_perimeter side_length base_length : ℕ), 
  equilateral_perimeter = 60 →  -- Condition: Perimeter of the equilateral triangle is 60
  isosceles_perimeter = 45 →    -- Condition: Perimeter of the isosceles triangle is 45
  side_length = equilateral_perimeter / 3 →   -- Condition: Each side of the equilateral triangle
  isosceles_perimeter = side_length + side_length + base_length  -- Condition: Perimeter relation in isosceles triangle
  → base_length = 5  -- Result: The base length of the isosceles triangle is 5
:= 
sorry

end NUMINAMATH_GPT_isosceles_base_length_l272_27230


namespace NUMINAMATH_GPT_expression_value_l272_27202

theorem expression_value : 2013 * (2015 / 2014) + 2014 * (2016 / 2015) + (4029 / (2014 * 2015)) = 4029 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l272_27202


namespace NUMINAMATH_GPT_irrational_implies_irrational_l272_27220

-- Define irrational number proposition
def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

-- Define the main proposition to prove
theorem irrational_implies_irrational (a : ℝ) : is_irrational (a - 2) → is_irrational a :=
by
  sorry

end NUMINAMATH_GPT_irrational_implies_irrational_l272_27220


namespace NUMINAMATH_GPT_solve_for_x_l272_27221

def f (x : ℝ) : ℝ := x^2 + x - 1

theorem solve_for_x (x : ℝ) (h : f x = 5) : x = 2 ∨ x = -3 := 
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l272_27221


namespace NUMINAMATH_GPT_ploughing_solution_l272_27294

/-- Definition representing the problem of A and B ploughing the field together and alone --/
noncomputable def ploughing_problem : Prop :=
  ∃ (A : ℝ), (A > 0) ∧ (1 / A + 1 / 30 = 1 / 10) ∧ A = 15

theorem ploughing_solution : ploughing_problem :=
  by sorry

end NUMINAMATH_GPT_ploughing_solution_l272_27294


namespace NUMINAMATH_GPT_rate_of_current_l272_27250

theorem rate_of_current
  (D U R : ℝ)
  (hD : D = 45)
  (hU : U = 23)
  (hR : R = 34)
  : (D - R = 11) ∧ (R - U = 11) :=
by
  sorry

end NUMINAMATH_GPT_rate_of_current_l272_27250


namespace NUMINAMATH_GPT_total_pieces_of_gum_l272_27243

def packages : ℕ := 12
def pieces_per_package : ℕ := 20

theorem total_pieces_of_gum : packages * pieces_per_package = 240 :=
by
  -- proof is skipped
  sorry

end NUMINAMATH_GPT_total_pieces_of_gum_l272_27243


namespace NUMINAMATH_GPT_group9_40_41_right_angled_l272_27211

theorem group9_40_41_right_angled :
  ¬ (∃ a b c : ℝ, a = 3 ∧ b = 4 ∧ c = 7 ∧ a^2 + b^2 = c^2) ∧
  ¬ (∃ a b c : ℝ, a = 1/3 ∧ b = 1/4 ∧ c = 1/5 ∧ a^2 + b^2 = c^2) ∧
  ¬ (∃ a b c : ℝ, a = 4 ∧ b = 6 ∧ c = 8 ∧ a^2 + b^2 = c^2) ∧
  (∃ a b c : ℝ, a = 9 ∧ b = 40 ∧ c = 41 ∧ a^2 + b^2 = c^2) :=
by
  sorry

end NUMINAMATH_GPT_group9_40_41_right_angled_l272_27211


namespace NUMINAMATH_GPT_f_eq_f_inv_l272_27263

noncomputable def f (x : ℝ) : ℝ := 3 * x - 7

noncomputable def f_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem f_eq_f_inv (x : ℝ) : f x = f_inv x ↔ x = 3.5 := by
  sorry

end NUMINAMATH_GPT_f_eq_f_inv_l272_27263


namespace NUMINAMATH_GPT_fishing_ratio_l272_27295

variables (B C : ℝ)
variable (brian_per_trip : ℝ)
variable (chris_per_trip : ℝ)

-- Given conditions
def conditions : Prop :=
  C = 10 ∧
  brian_per_trip = 400 ∧
  chris_per_trip = 400 * (5 / 3) ∧
  B * brian_per_trip + 10 * chris_per_trip = 13600

-- The ratio of the number of times Brian goes fishing to the number of times Chris goes fishing
def ratio_correct : Prop :=
  B / C = 26 / 15

theorem fishing_ratio (h : conditions B C brian_per_trip chris_per_trip) : ratio_correct B C :=
by
  sorry

end NUMINAMATH_GPT_fishing_ratio_l272_27295


namespace NUMINAMATH_GPT_meet_time_opposite_directions_catch_up_time_same_direction_l272_27270

def length_of_track := 440
def speed_A := 5
def speed_B := 6

theorem meet_time_opposite_directions :
  (length_of_track / (speed_A + speed_B)) = 40 :=
by
  sorry

theorem catch_up_time_same_direction :
  (length_of_track / (speed_B - speed_A)) = 440 :=
by
  sorry

end NUMINAMATH_GPT_meet_time_opposite_directions_catch_up_time_same_direction_l272_27270


namespace NUMINAMATH_GPT_pyramid_prism_sum_l272_27290

-- Definitions based on conditions
structure Prism :=
  (vertices : ℕ)
  (edges : ℕ)
  (faces : ℕ)

-- The initial cylindrical-prism object
noncomputable def initial_prism : Prism :=
  { vertices := 8,
    edges := 10,
    faces := 5 }

-- Structure for Pyramid Addition
structure PyramidAddition :=
  (new_vertices : ℕ)
  (new_edges : ℕ)
  (new_faces : ℕ)

noncomputable def pyramid_addition : PyramidAddition := 
  { new_vertices := 1,
    new_edges := 4,
    new_faces := 4 }

-- Function to add pyramid to the prism
noncomputable def add_pyramid (prism : Prism) (pyramid : PyramidAddition) : Prism :=
  { vertices := prism.vertices + pyramid.new_vertices,
    edges := prism.edges + pyramid.new_edges,
    faces := prism.faces - 1 + pyramid.new_faces }

-- The resulting prism after adding the pyramid
noncomputable def resulting_prism := add_pyramid initial_prism pyramid_addition

-- Proof problem statement
theorem pyramid_prism_sum : 
  resulting_prism.vertices + resulting_prism.edges + resulting_prism.faces = 31 :=
by sorry

end NUMINAMATH_GPT_pyramid_prism_sum_l272_27290


namespace NUMINAMATH_GPT_stock_percentage_change_l272_27285

theorem stock_percentage_change :
  let initial_value := 100
  let value_after_first_day := initial_value * (1 - 0.25)
  let value_after_second_day := value_after_first_day * (1 + 0.35)
  let final_value := value_after_second_day * (1 - 0.15)
  let overall_percentage_change := ((final_value - initial_value) / initial_value) * 100
  overall_percentage_change = -13.9375 := 
by
  sorry

end NUMINAMATH_GPT_stock_percentage_change_l272_27285


namespace NUMINAMATH_GPT_certain_number_is_8000_l272_27289

theorem certain_number_is_8000 (x : ℕ) (h : x / 10 - x / 2000 = 796) : x = 8000 :=
sorry

end NUMINAMATH_GPT_certain_number_is_8000_l272_27289


namespace NUMINAMATH_GPT_one_statement_is_true_l272_27255

theorem one_statement_is_true :
  ∃ (S1 S2 S3 S4 S5 : Prop),
    ((S1 ↔ (¬S1 ∧ S2 ∧ S3 ∧ S4 ∧ S5)) ∧
     (S2 ↔ (¬S1 ∧ ¬S2 ∧ S3 ∧ S4 ∧ ¬S5)) ∧
     (S3 ↔ (¬S1 ∧ S2 ∧ ¬S3 ∧ S4 ∧ ¬S5)) ∧
     (S4 ↔ (¬S1 ∧ ¬S2 ∧ ¬S3 ∧ S4 ∧ ¬S5)) ∧
     (S5 ↔ (¬S1 ∧ ¬S2 ∧ ¬S3 ∧ ¬S4 ∧ ¬S5))) ∧
    (S2) ∧ (¬S1) ∧ (¬S3) ∧ (¬S4) ∧ (¬S5) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_one_statement_is_true_l272_27255


namespace NUMINAMATH_GPT_joshua_finishes_after_malcolm_l272_27252

-- Definitions based on conditions.
def malcolm_speed : ℕ := 6 -- Malcolm's speed in minutes per mile
def joshua_speed : ℕ := 8 -- Joshua's speed in minutes per mile
def race_distance : ℕ := 10 -- Race distance in miles

-- Theorem: How many minutes after Malcolm crosses the finish line will Joshua cross the finish line?
theorem joshua_finishes_after_malcolm :
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 20 :=
by
  -- sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_joshua_finishes_after_malcolm_l272_27252


namespace NUMINAMATH_GPT_top_angle_is_70_l272_27257

theorem top_angle_is_70
  (sum_angles : ℝ)
  (left_angle : ℝ)
  (right_angle : ℝ)
  (top_angle : ℝ)
  (h1 : sum_angles = 250)
  (h2 : left_angle = 2 * right_angle)
  (h3 : right_angle = 60)
  (h4 : sum_angles = left_angle + right_angle + top_angle) :
  top_angle = 70 :=
by
  sorry

end NUMINAMATH_GPT_top_angle_is_70_l272_27257


namespace NUMINAMATH_GPT_max_abc_value_l272_27246

theorem max_abc_value (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_equation : a * b + c = (a + c) * (b + c))
  (h_sum : a + b + c = 2) : abc ≤ 1/27 :=
by sorry

end NUMINAMATH_GPT_max_abc_value_l272_27246


namespace NUMINAMATH_GPT_largest_sum_fraction_l272_27218

theorem largest_sum_fraction :
  max 
    ((1/3) + (1/2))
    (max 
      ((1/3) + (1/5))
      (max 
        ((1/3) + (1/6))
        (max 
          ((1/3) + (1/9))
          ((1/3) + (1/10))
        )
      )
    ) = 5/6 :=
by sorry

end NUMINAMATH_GPT_largest_sum_fraction_l272_27218


namespace NUMINAMATH_GPT_total_short_trees_after_planting_l272_27200

def initial_short_trees : ℕ := 31
def planted_short_trees : ℕ := 64

theorem total_short_trees_after_planting : initial_short_trees + planted_short_trees = 95 := by
  sorry

end NUMINAMATH_GPT_total_short_trees_after_planting_l272_27200


namespace NUMINAMATH_GPT_problem_l272_27256

theorem problem (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (17 * x)) (x : ℝ) :
  f (Real.cos x) ^ 2 + f (Real.sin x) ^ 2 = 1 :=
sorry

end NUMINAMATH_GPT_problem_l272_27256


namespace NUMINAMATH_GPT_least_side_is_8_l272_27292

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end NUMINAMATH_GPT_least_side_is_8_l272_27292


namespace NUMINAMATH_GPT_find_prime_powers_l272_27251

open Nat

theorem find_prime_powers (p x y : ℕ) (hp : p.Prime) (hx : 0 < x) (hy : 0 < y) :
  p^x = y^3 + 1 ↔
  (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
sorry

end NUMINAMATH_GPT_find_prime_powers_l272_27251


namespace NUMINAMATH_GPT_evaluate_g_at_neg_one_l272_27273

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 - 3 * x + 9

theorem evaluate_g_at_neg_one : g (-1) = 7 :=
by 
  -- lean proof here
  sorry

end NUMINAMATH_GPT_evaluate_g_at_neg_one_l272_27273


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l272_27203

variable (p : ℕ) (k : ℕ)

theorem part_a (hp : Prime p) (h : p = 4 * k + 1) :
  ∃ x : ℤ, (x^2 + 1) % p = 0 :=
by
  sorry

theorem part_b (hp : Prime p) (h : p = 4 * k + 1)
  (x : ℤ) (r1 r2 s1 s2 : ℕ)
  (hr1 : 0 ≤ r1) (hr2 : 0 ≤ r2) (hr1_lt : r1 < Nat.sqrt p) (hr2_lt : r2 < Nat.sqrt p)
  (hs1 : 0 ≤ s1) (hs2 : 0 ≤ s2) (hs1_lt : s1 < Nat.sqrt p) (hs2_lt : s2 < Nat.sqrt p)
  (hneq : (r1, s1) ≠ (r2, s2)) :
  ∃ (r1 r2 s1 s2 : ℕ), (r1 * x + s1) % p = (r2 * x + s2) % p :=
by
  sorry

theorem part_c (hp : Prime p) (h : p = 4 * k + 1)
  (x : ℤ) (r1 r2 s1 s2 : ℕ)
  (hr1 : 0 ≤ r1) (hr2 : 0 ≤ r2) (hr1_lt : r1 < Nat.sqrt p) (hr2_lt : r2 < Nat.sqrt p)
  (hs1 : 0 ≤ s1) (hs2 : 0 ≤ s2) (hs1_lt : s1 < Nat.sqrt p) (hs2_lt : s2 < Nat.sqrt p)
  (hneq : (r1, s1) ≠ (r2, s2)):
  p = (Int.ofNat (r1 - r2))^2 + (Int.ofNat (s1 - s2))^2 :=
by
  sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l272_27203


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l272_27265

theorem arithmetic_seq_sum (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 3 + a 7 = 38) : 
  a 2 + a 4 + a 6 + a 8 = 76 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l272_27265


namespace NUMINAMATH_GPT_calculate_triple_hash_l272_27268

def hash (N : ℝ) : ℝ := 0.5 * N - 2

theorem calculate_triple_hash : hash (hash (hash 100)) = 9 := by
  sorry

end NUMINAMATH_GPT_calculate_triple_hash_l272_27268


namespace NUMINAMATH_GPT_circle_through_focus_l272_27233

open Real

-- Define the parabola as a set of points
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  (P.2 - 3) ^ 2 = 8 * (P.1 - 2)

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (4, 3)

-- Define the circle with center P and radius the distance from P to the y-axis
def is_tangent_circle (P : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  (P.1 ^ 2 + (P.2 - 3) ^ 2 = (C.1) ^ 2 + (C.2) ^ 2 ∧ C = (4, 3))

-- The main theorem
theorem circle_through_focus (P : ℝ × ℝ) 
  (hP_on_parabola : is_on_parabola P) 
  (hP_tangent_circle : is_tangent_circle P (4, 3)) :
  is_tangent_circle P (4, 3) :=
by sorry

end NUMINAMATH_GPT_circle_through_focus_l272_27233


namespace NUMINAMATH_GPT_part1_part2_part3_l272_27223

-- Definition of the given expression
def expr (a b : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + 2) - (5 * x^2 + 3 * x)

-- Condition 1: Given final result 2x^2 - 4x + 2
def target_expr1 (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 2

-- Condition 2: Given values for a and b by Student B
def student_b_expr (x : ℝ) : ℝ := (5 * x^2 - 3 * x + 2) - (5 * x^2 + 3 * x)

-- Condition 3: Result independent of x
def target_expr3 : ℝ := 2

-- Prove conditions and answers
theorem part1 (a b : ℝ) : (∀ x : ℝ, expr a b x = target_expr1 x) → a = 7 ∧ b = -1 :=
sorry

theorem part2 : (∀ x : ℝ, student_b_expr x = -6 * x + 2) :=
sorry

theorem part3 (a b : ℝ) : (∀ x : ℝ, expr a b x = 2) → a = 5 ∧ b = 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l272_27223


namespace NUMINAMATH_GPT_convert_spherical_to_rectangular_l272_27228

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem convert_spherical_to_rectangular : spherical_to_rectangular 5 (Real.pi / 2) (Real.pi / 3) = 
  (0, 5 * Real.sqrt 3 / 2, 5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_convert_spherical_to_rectangular_l272_27228


namespace NUMINAMATH_GPT_cylinder_radius_exists_l272_27224

theorem cylinder_radius_exists (r h : ℕ) (pr : r ≥ 1) :
  (π * ↑r ^ 2 * ↑h = 2 * π * ↑r * (↑h + ↑r)) ↔
  (r = 3 ∨ r = 4 ∨ r = 6) :=
by
  sorry

end NUMINAMATH_GPT_cylinder_radius_exists_l272_27224


namespace NUMINAMATH_GPT_shoveling_time_l272_27284

theorem shoveling_time :
  let kevin_time := 12
  let dave_time := 8
  let john_time := 6
  let allison_time := 4
  let kevin_rate := 1 / kevin_time
  let dave_rate := 1 / dave_time
  let john_rate := 1 / john_time
  let allison_rate := 1 / allison_time
  let combined_rate := kevin_rate + dave_rate + john_rate + allison_rate
  let total_minutes := 60
  let combined_rate_per_minute := combined_rate / total_minutes
  (1 / combined_rate_per_minute = 96) := 
  sorry

end NUMINAMATH_GPT_shoveling_time_l272_27284


namespace NUMINAMATH_GPT_Q_2_plus_Q_neg2_l272_27208

variable {k : ℝ}

noncomputable def Q (x : ℝ) : ℝ := 0 -- Placeholder definition, real polynomial will be defined in proof.

theorem Q_2_plus_Q_neg2 (hQ0 : Q 0 = 2 * k)
  (hQ1 : Q 1 = 3 * k)
  (hQ_minus1 : Q (-1) = 4 * k) :
  Q 2 + Q (-2) = 16 * k :=
sorry

end NUMINAMATH_GPT_Q_2_plus_Q_neg2_l272_27208


namespace NUMINAMATH_GPT_radical_product_l272_27259

theorem radical_product :
  (64 ^ (1 / 3) * 16 ^ (1 / 4) * 64 ^ (1 / 6) = 16) :=
by
  sorry

end NUMINAMATH_GPT_radical_product_l272_27259


namespace NUMINAMATH_GPT_negation_proposition_l272_27286

theorem negation_proposition (a b c : ℝ) : 
  (¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3)) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_negation_proposition_l272_27286


namespace NUMINAMATH_GPT_value_of_x_squared_minus_y_squared_l272_27297

theorem value_of_x_squared_minus_y_squared (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 := by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_minus_y_squared_l272_27297


namespace NUMINAMATH_GPT_train_complete_time_l272_27262

noncomputable def train_time_proof : Prop :=
  ∃ (t_x : ℕ) (v_x : ℝ) (v_y : ℝ),
    v_y = 140 / 3 ∧
    t_x = 140 / v_x ∧
    (∃ t : ℝ, 
      t * v_x = 60.00000000000001 ∧
      t * v_y = 140 - 60.00000000000001) ∧
    t_x = 4

theorem train_complete_time : train_time_proof := by
  sorry

end NUMINAMATH_GPT_train_complete_time_l272_27262


namespace NUMINAMATH_GPT_car_speed_second_hour_l272_27216

/-- The speed of the car in the first hour is 85 km/h, the average speed is 65 km/h over 2 hours,
proving that the speed of the car in the second hour is 45 km/h. -/
theorem car_speed_second_hour (v1 : ℕ) (v_avg : ℕ) (t : ℕ) (d1 : ℕ) (d2 : ℕ) 
  (h1 : v1 = 85) (h2 : v_avg = 65) (h3 : t = 2) (h4 : d1 = v1 * 1) (h5 : d2 = (v_avg * t) - d1) :
  d2 = 45 :=
sorry

end NUMINAMATH_GPT_car_speed_second_hour_l272_27216


namespace NUMINAMATH_GPT_least_number_of_tablets_l272_27236

theorem least_number_of_tablets (tablets_A : ℕ) (tablets_B : ℕ) (hA : tablets_A = 10) (hB : tablets_B = 13) :
  ∃ n, ((tablets_A ≤ 10 → n ≥ tablets_A + 2) ∧ (tablets_B ≤ 13 → n ≥ tablets_B + 2)) ∧ n = 12 :=
by
  sorry

end NUMINAMATH_GPT_least_number_of_tablets_l272_27236


namespace NUMINAMATH_GPT_rationalize_denominator_l272_27274

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem rationalize_denominator (a b : ℝ) (h : cube_root 27 = 3) : 
  1 / (cube_root 3 + cube_root 27) = (3 - cube_root 3) / (9 - 3 * cube_root 3) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l272_27274


namespace NUMINAMATH_GPT_cyclic_quadrilateral_eq_l272_27247

theorem cyclic_quadrilateral_eq (A B C D : ℝ) (AB AD BC DC : ℝ)
  (h1 : AB = AD) (h2 : based_on_laws_of_cosines) : AC ^ 2 = BC * DC + AB ^ 2 :=
sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_eq_l272_27247


namespace NUMINAMATH_GPT_find_X_l272_27248

theorem find_X 
  (X Y : ℕ)
  (h1 : 6 + X = 13)
  (h2 : Y = 7) :
  X = 7 := by
  sorry

end NUMINAMATH_GPT_find_X_l272_27248


namespace NUMINAMATH_GPT_fifth_power_last_digit_l272_27278

theorem fifth_power_last_digit (n : ℕ) : 
  (n % 10)^5 % 10 = n % 10 :=
by sorry

end NUMINAMATH_GPT_fifth_power_last_digit_l272_27278


namespace NUMINAMATH_GPT_proposition_false_at_4_l272_27213

open Nat

def prop (n : ℕ) : Prop := sorry -- the actual proposition is not specified, so we use sorry

theorem proposition_false_at_4 :
  (∀ k : ℕ, k > 0 → (prop k → prop (k + 1))) →
  ¬ prop 5 →
  ¬ prop 4 :=
by
  intros h_induction h_proposition_false_at_5
  sorry

end NUMINAMATH_GPT_proposition_false_at_4_l272_27213


namespace NUMINAMATH_GPT_eventually_composite_appending_threes_l272_27219

theorem eventually_composite_appending_threes (n : ℕ) :
  ∃ n' : ℕ, n' = 10 * n + 3 ∧ ∃ k : ℕ, k > 0 ∧ (3 * k + 3) % 7 ≠ 1 ∧ (3 * k + 3) % 7 ≠ 2 ∧ (3 * k + 3) % 7 ≠ 3 ∧
  (3 * k + 3) % 7 ≠ 5 ∧ (3 * k + 3) % 7 ≠ 6 :=
sorry

end NUMINAMATH_GPT_eventually_composite_appending_threes_l272_27219


namespace NUMINAMATH_GPT_fiona_shirt_number_l272_27212

def is_two_digit_prime (n : ℕ) : Prop := 
  (n ≥ 10 ∧ n < 100 ∧ Nat.Prime n)

theorem fiona_shirt_number (d e f : ℕ) 
  (h1 : is_two_digit_prime d)
  (h2 : is_two_digit_prime e)
  (h3 : is_two_digit_prime f)
  (h4 : e + f = 36)
  (h5 : d + e = 30)
  (h6 : d + f = 32) : 
  f = 19 := 
sorry

end NUMINAMATH_GPT_fiona_shirt_number_l272_27212


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l272_27272

theorem arithmetic_sequence_sum {a : ℕ → ℤ} (S : ℕ → ℤ) 
  (h₀ : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  (h₁ : S 9 = 27) :
  (a 4 + a 6) = 6 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l272_27272


namespace NUMINAMATH_GPT_intersection_A_B_l272_27215

namespace SetTheory

open Set

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end SetTheory

end NUMINAMATH_GPT_intersection_A_B_l272_27215


namespace NUMINAMATH_GPT_tan_105_eq_neg2_sub_sqrt3_l272_27238

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_105_eq_neg2_sub_sqrt3_l272_27238


namespace NUMINAMATH_GPT_student_community_arrangements_l272_27209

theorem student_community_arrangements :
  (3 ^ 4) = 81 :=
by
  sorry

end NUMINAMATH_GPT_student_community_arrangements_l272_27209


namespace NUMINAMATH_GPT_probability_one_white_ball_conditional_probability_P_B_given_A_l272_27237

-- Definitions for Problem 1
def red_balls : Nat := 4
def white_balls : Nat := 2
def total_balls : Nat := red_balls + white_balls

def C (n k : ℕ) : ℕ := n.choose k

theorem probability_one_white_ball :
  (C 2 1 * C 4 2 : ℚ) / C 6 3 = 3 / 5 :=
by sorry

-- Definitions for Problem 2
def total_after_first_draw : Nat := total_balls - 1
def remaining_red_balls : Nat := red_balls - 1

theorem conditional_probability_P_B_given_A :
  (remaining_red_balls : ℚ) / total_after_first_draw = 3 / 5 :=
by sorry

end NUMINAMATH_GPT_probability_one_white_ball_conditional_probability_P_B_given_A_l272_27237


namespace NUMINAMATH_GPT_distance_between_first_and_last_tree_l272_27260

theorem distance_between_first_and_last_tree
  (n : ℕ) (d_1_5 : ℝ) (h1 : n = 8) (h2 : d_1_5 = 100) :
  let interval_distance := d_1_5 / 4
  let total_intervals := n - 1
  let total_distance := interval_distance * total_intervals
  total_distance = 175 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_first_and_last_tree_l272_27260


namespace NUMINAMATH_GPT_equipment_total_cost_l272_27206

-- Definition of costs for each item of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80

-- Number of players
def num_players : ℕ := 16

-- Total cost for one player
def total_cost_one_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Total cost for all players
def total_cost_all_players : ℝ := total_cost_one_player * num_players

-- Theorem to prove
theorem equipment_total_cost : total_cost_all_players = 752 := by
  sorry

end NUMINAMATH_GPT_equipment_total_cost_l272_27206


namespace NUMINAMATH_GPT_difference_between_max_and_min_change_l272_27226

-- Define percentages as fractions for Lean
def initial_yes : ℚ := 60 / 100
def initial_no : ℚ := 40 / 100
def final_yes : ℚ := 80 / 100
def final_no : ℚ := 20 / 100
def new_students : ℚ := 10 / 100

-- Define the minimum and maximum possible values of changes (in percentage as a fraction)
def min_change : ℚ := 10 / 100
def max_change : ℚ := 50 / 100

-- The theorem we need to prove
theorem difference_between_max_and_min_change : (max_change - min_change) = 40 / 100 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_max_and_min_change_l272_27226


namespace NUMINAMATH_GPT_area_of_bounded_curve_is_64_pi_l272_27277

noncomputable def bounded_curve_area : Real :=
  let curve_eq (x y : ℝ) : Prop := (2 * x + 3 * y + 5) ^ 2 + (x + 2 * y - 3) ^ 2 = 64
  let S : Real := 64 * Real.pi
  S

theorem area_of_bounded_curve_is_64_pi : bounded_curve_area = 64 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_area_of_bounded_curve_is_64_pi_l272_27277


namespace NUMINAMATH_GPT_cubic_sum_identity_l272_27207

theorem cubic_sum_identity
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : ab + ac + bc = -3)
  (h3 : abc = 9) :
  a^3 + b^3 + c^3 = 22 :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_identity_l272_27207


namespace NUMINAMATH_GPT_tom_age_difference_l272_27217

/-- 
Tom Johnson's age is some years less than twice as old as his sister.
The sum of their ages is 14 years.
Tom's age is 9 years.
Prove that the number of years less Tom's age is than twice his sister's age is 1 year. 
-/ 
theorem tom_age_difference (T S : ℕ) 
  (h₁ : T = 9) 
  (h₂ : T + S = 14) : 
  2 * S - T = 1 := 
by 
  sorry

end NUMINAMATH_GPT_tom_age_difference_l272_27217


namespace NUMINAMATH_GPT_max_f_value_l272_27242

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_f_value : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 ∧ ∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), f x₀ = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_f_value_l272_27242


namespace NUMINAMATH_GPT_paisa_per_rupee_z_gets_l272_27241

theorem paisa_per_rupee_z_gets
  (y_share : ℝ)
  (y_per_x_paisa : ℝ)
  (total_amount : ℝ)
  (x_share : ℝ)
  (z_share : ℝ)
  (paisa_per_rupee : ℝ)
  (h1 : y_share = 36)
  (h2 : y_per_x_paisa = 0.45)
  (h3 : total_amount = 140)
  (h4 : x_share = y_share / y_per_x_paisa)
  (h5 : z_share = total_amount - (x_share + y_share))
  (h6 : paisa_per_rupee = (z_share / x_share) * 100) :
  paisa_per_rupee = 30 :=
by
  sorry

end NUMINAMATH_GPT_paisa_per_rupee_z_gets_l272_27241


namespace NUMINAMATH_GPT_parabola_focus_coincides_ellipse_focus_l272_27281

theorem parabola_focus_coincides_ellipse_focus (p : ℝ) :
  (∃ F : ℝ × ℝ, F = (2, 0) ∧ ∀ x y : ℝ, y^2 = 2 * p * x <-> x = p / 2)
  → p = 4 := 
by
  sorry 

end NUMINAMATH_GPT_parabola_focus_coincides_ellipse_focus_l272_27281


namespace NUMINAMATH_GPT_solve_for_x_l272_27267

theorem solve_for_x (x : ℚ) (h : (x - 75) / 4 = (5 - 3 * x) / 7) : x = 545 / 19 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l272_27267


namespace NUMINAMATH_GPT_part1_part2_l272_27288

def A (x : ℝ) : Prop := -2 < x ∧ x < 10
def B (x a : ℝ) : Prop := (x ≥ 1 + a ∨ x ≤ 1 - a) ∧ a > 0
def p (x : ℝ) : Prop := A x
def q (x a : ℝ) : Prop := B x a

theorem part1 (a : ℝ) (hA : ∀ x, A x → ¬ B x a) : a ≥ 9 :=
sorry

theorem part2 (a : ℝ) (hSuff : ∀ x, (x ≥ 10 ∨ x ≤ -2) → B x a) (hNotNec : ∃ x, ¬ (x ≥ 10 ∨ x ≤ -2) ∧ B x a) : 0 < a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l272_27288


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l272_27240

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  a 1 + a 2 = -1 →
  a 3 = 4 →
  (a 1 + 2 * d = 4) →
  ∀ n, a n = a 1 + (n - 1) * d →
  a 4 + a 5 = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l272_27240


namespace NUMINAMATH_GPT_triangle_ratio_l272_27214

variables (A B C : ℝ) (a b c : ℝ)

theorem triangle_ratio (h_cosB : Real.cos B = 4/5)
    (h_a : a = 5)
    (h_area : 1/2 * a * c * Real.sin B = 12) :
    (a + c) / (Real.sin A + Real.sin C) = 25 / 3 :=
sorry

end NUMINAMATH_GPT_triangle_ratio_l272_27214


namespace NUMINAMATH_GPT_female_officers_count_l272_27261

theorem female_officers_count
  (total_on_duty : ℕ)
  (on_duty_females : ℕ)
  (total_female_officers : ℕ)
  (h1 : total_on_duty = 240)
  (h2 : on_duty_females = total_on_duty / 2)
  (h3 : on_duty_females = (40 * total_female_officers) / 100) : 
  total_female_officers = 300 := 
by
  sorry

end NUMINAMATH_GPT_female_officers_count_l272_27261


namespace NUMINAMATH_GPT_largest_possible_e_l272_27232

noncomputable def diameter := (2 : ℝ)
noncomputable def PX := (4 / 5 : ℝ)
noncomputable def PY := (3 / 4 : ℝ)
noncomputable def e := (41 - 16 * Real.sqrt 25 : ℝ)
noncomputable def u := 41
noncomputable def v := 16
noncomputable def w := 25

theorem largest_possible_e (P Q X Y Z R S : Real) (d : diameter = 2)
  (PX_len : P - X = 4/5) (PY_len : P - Y = 3/4)
  (e_def : e = 41 - 16 * Real.sqrt 25)
  : u + v + w = 82 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_e_l272_27232


namespace NUMINAMATH_GPT_drum_wife_leopard_cost_l272_27264

-- Definitions
variables (x y z : ℤ)

def system1 := 2 * x + 3 * y + z = 111
def system2 := 3 * x + 4 * y - 2 * z = -8
def even_condition := z % 2 = 0

theorem drum_wife_leopard_cost:
  system1 x y z ∧ system2 x y z ∧ even_condition z →
  x = 20 ∧ y = 9 ∧ z = 44 :=
by
  intro h
  -- Full proof can be provided here
  sorry

end NUMINAMATH_GPT_drum_wife_leopard_cost_l272_27264


namespace NUMINAMATH_GPT_log_10_7_eqn_l272_27269

variables (p q : ℝ)
noncomputable def log_base (a b : ℝ) : ℝ := (Real.log b) / (Real.log a)

theorem log_10_7_eqn (h1 : log_base 4 5 = p) (h2 : log_base 5 7 = q) : 
  log_base 10 7 = (2 * p * q) / (2 * p + 1) :=
by 
  sorry

end NUMINAMATH_GPT_log_10_7_eqn_l272_27269


namespace NUMINAMATH_GPT_initial_mixture_volume_l272_27201

theorem initial_mixture_volume (x : ℝ) (hx1 : 0.10 * x + 10 = 0.28 * (x + 10)) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_initial_mixture_volume_l272_27201


namespace NUMINAMATH_GPT_identify_incorrect_proposition_l272_27205

-- Definitions based on problem conditions
def propositionA : Prop :=
  (∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0))

def propositionB : Prop :=
  (¬ (∃ x : ℝ, x^2 + x + 1 = 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0)

def propositionD (x : ℝ) : Prop :=
  (x > 2 → x^2 - 3*x + 2 > 0) ∧ (¬(x > 2) → ¬(x^2 - 3*x + 2 > 0))

-- Proposition C is given to be incorrect in the problem
def propositionC (p q : Prop) : Prop := ¬ (p ∧ q) → ¬p ∧ ¬q

theorem identify_incorrect_proposition (p q : Prop) : 
  (propositionA ∧ propositionB ∧ (∀ x : ℝ, propositionD x)) → 
  ¬ (propositionC p q) :=
by
  intros
  -- We know proposition C is false based on the problem's solution
  sorry

end NUMINAMATH_GPT_identify_incorrect_proposition_l272_27205


namespace NUMINAMATH_GPT_weight_of_10m_l272_27249

-- Defining the proportional weight conditions
variable (weight_of_rod : ℝ → ℝ)

-- Conditional facts about the weight function
axiom weight_proportional : ∀ (length1 length2 : ℝ), length1 ≠ 0 → length2 ≠ 0 → 
  weight_of_rod length1 / length1 = weight_of_rod length2 / length2
axiom weight_of_6m : weight_of_rod 6 = 14.04

-- Theorem stating the weight of a 10m rod
theorem weight_of_10m : weight_of_rod 10 = 23.4 := 
sorry

end NUMINAMATH_GPT_weight_of_10m_l272_27249


namespace NUMINAMATH_GPT_polynomial_is_perfect_cube_l272_27283

theorem polynomial_is_perfect_cube (p q n : ℚ) :
  (∃ a : ℚ, x^3 + p * x^2 + q * x + n = (x + a)^3) ↔ (q = p^2 / 3 ∧ n = p^3 / 27) :=
by sorry

end NUMINAMATH_GPT_polynomial_is_perfect_cube_l272_27283


namespace NUMINAMATH_GPT_statistical_measure_mode_l272_27231

theorem statistical_measure_mode (fav_dishes : List ℕ) :
  (∀ measure, (measure = "most frequently occurring value" → measure = "Mode")) :=
by
  intro measure
  intro h
  sorry

end NUMINAMATH_GPT_statistical_measure_mode_l272_27231


namespace NUMINAMATH_GPT_solve_first_l272_27298

theorem solve_first (x y : ℝ) (C : ℝ) :
  (1 + y^2) * (deriv id x) - (1 + x^2) * y * (deriv id y) = 0 →
  Real.arctan x = 1/2 * Real.log (1 + y^2) + Real.log C := 
sorry

end NUMINAMATH_GPT_solve_first_l272_27298


namespace NUMINAMATH_GPT_negation_proposition_l272_27293

theorem negation_proposition :
  ∃ (a : ℝ) (n : ℕ), n > 0 ∧ a ≠ n ∧ a * n = 2 * n :=
sorry

end NUMINAMATH_GPT_negation_proposition_l272_27293


namespace NUMINAMATH_GPT_part1_part2_part3_l272_27229

noncomputable def p1_cost (t : ℕ) : ℕ := 
  if t <= 150 then 58 else 58 + 25 * (t - 150) / 100

noncomputable def p2_cost (t : ℕ) (a : ℕ) : ℕ := 
  if t <= 350 then 88 else 88 + a * (t - 350)

-- Part 1: Prove the costs for 260 minutes
theorem part1 : p1_cost 260 = 855 / 10 ∧ p2_cost 260 30 = 88 :=
by 
  sorry

-- Part 2: Prove the existence of t for given a
theorem part2 (t : ℕ) : (a = 30) → (∃ t, p1_cost t = p2_cost t a) :=
by 
  sorry

-- Part 3: Prove a=45 and the range for which Plan 1 is cheaper
theorem part3 : 
  (a = 45) ↔ (p1_cost 450 = p2_cost 450 a) ∧ (∀ t, (0 ≤ t ∧ t < 270) ∨ (t > 450) → p1_cost t < p2_cost t 45 ) :=
by 
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l272_27229
