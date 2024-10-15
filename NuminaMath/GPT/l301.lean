import Mathlib

namespace NUMINAMATH_GPT_ratio_vegan_gluten_free_cupcakes_l301_30127

theorem ratio_vegan_gluten_free_cupcakes :
  let total_cupcakes := 80
  let gluten_free_cupcakes := total_cupcakes / 2
  let vegan_cupcakes := 24
  let non_vegan_gluten_cupcakes := 28
  let vegan_gluten_free_cupcakes := gluten_free_cupcakes - non_vegan_gluten_cupcakes
  (vegan_gluten_free_cupcakes / vegan_cupcakes) = 1 / 2 :=
by {
  let total_cupcakes := 80
  let gluten_free_cupcakes := total_cupcakes / 2
  let vegan_cupcakes := 24
  let non_vegan_gluten_cupcakes := 28
  let vegan_gluten_free_cupcakes := gluten_free_cupcakes - non_vegan_gluten_cupcakes
  have h : vegan_gluten_free_cupcakes = 12 := by norm_num
  have r : 12 / 24 = 1 / 2 := by norm_num
  exact r
}

end NUMINAMATH_GPT_ratio_vegan_gluten_free_cupcakes_l301_30127


namespace NUMINAMATH_GPT_hyperbola_parabola_focus_l301_30173

theorem hyperbola_parabola_focus (k : ℝ) (h : k > 0) :
  (∃ x y : ℝ, (1/k^2) * y^2 = 0 ∧ x^2 - (y^2 / k^2) = 1) ∧ (∃ x : ℝ, y^2 = 8 * x) →
  k = Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_hyperbola_parabola_focus_l301_30173


namespace NUMINAMATH_GPT_number_of_distinct_products_l301_30104

   -- We define the set S
   def S : Finset ℕ := {2, 3, 5, 11, 13}

   -- We define what it means to have a distinct product of two or more elements
   def distinctProducts (s : Finset ℕ) : Finset ℕ :=
     (s.powerset.filter (λ t => 2 ≤ t.card)).image (λ t => t.prod id)

   -- We state the theorem that there are exactly 26 distinct products
   theorem number_of_distinct_products : (distinctProducts S).card = 26 :=
   sorry
   
end NUMINAMATH_GPT_number_of_distinct_products_l301_30104


namespace NUMINAMATH_GPT_roast_cost_l301_30177

-- Given conditions as described in the problem.
def initial_money : ℝ := 100
def cost_vegetables : ℝ := 11
def money_left : ℝ := 72
def total_spent : ℝ := initial_money - money_left

-- The cost of the roast that we need to prove. We expect it to be €17.
def cost_roast : ℝ := total_spent - cost_vegetables

-- The theorem that states the cost of the roast given the conditions.
theorem roast_cost :
  cost_roast = 100 - 72 - 11 := by
  -- skipping the proof steps with sorry
  sorry

end NUMINAMATH_GPT_roast_cost_l301_30177


namespace NUMINAMATH_GPT_cars_at_2023_cars_less_than_15_l301_30116

def a_recurrence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 0.9 * a n + 8

def initial_condition (a : ℕ → ℝ) : Prop :=
a 1 = 300

theorem cars_at_2023 (a : ℕ → ℝ)
  (h_recurrence : a_recurrence a)
  (h_initial : initial_condition a) :
  a 4 = 240 :=
sorry

def shifted_geom_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - 80 = 0.9 * (a n - 80)

theorem cars_less_than_15 (a : ℕ → ℝ)
  (h_recurrence : a_recurrence a)
  (h_initial : initial_condition a)
  (h_geom_seq : shifted_geom_seq a) :
  ∃ n, n ≥ 12 ∧ a n < 15 :=
sorry

end NUMINAMATH_GPT_cars_at_2023_cars_less_than_15_l301_30116


namespace NUMINAMATH_GPT_expression_equals_negative_two_l301_30124

def f (x : ℝ) : ℝ := x^3 - x - 1
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem expression_equals_negative_two : 
  f 2023 + f' 2023 + f (-2023) - f' (-2023) = -2 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_negative_two_l301_30124


namespace NUMINAMATH_GPT_range_of_ab_l301_30113

-- Given two positive numbers a and b such that ab = a + b + 3, we need to prove ab ≥ 9.

theorem range_of_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = a + b + 3) : 9 ≤ a * b :=
by
  sorry

end NUMINAMATH_GPT_range_of_ab_l301_30113


namespace NUMINAMATH_GPT_number_of_sections_l301_30166

-- Definitions based on the conditions in a)
def num_reels : Nat := 3
def length_per_reel : Nat := 100
def section_length : Nat := 10

-- The math proof problem statement
theorem number_of_sections :
  (num_reels * length_per_reel) / section_length = 30 := by
  sorry

end NUMINAMATH_GPT_number_of_sections_l301_30166


namespace NUMINAMATH_GPT_devin_teaching_years_l301_30195

theorem devin_teaching_years :
  let calculus_years := 4
  let algebra_years := 2 * calculus_years
  let statistics_years := 5 * algebra_years
  calculus_years + algebra_years + statistics_years = 52 :=
by
  let calculus_years := 4
  let algebra_years := 2 * calculus_years
  let statistics_years := 5 * algebra_years
  show calculus_years + algebra_years + statistics_years = 52
  sorry

end NUMINAMATH_GPT_devin_teaching_years_l301_30195


namespace NUMINAMATH_GPT_min_abs_sum_of_diffs_l301_30109

theorem min_abs_sum_of_diffs (x : ℝ) (α β : ℝ)
  (h₁ : α * α - 6 * α + 5 = 0)
  (h₂ : β * β - 6 * β + 5 = 0)
  (h_ne : α ≠ β) :
  ∃ m, ∀ x, m = min (|x - α| + |x - β|) :=
by
  use (4)
  sorry

end NUMINAMATH_GPT_min_abs_sum_of_diffs_l301_30109


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l301_30163

theorem geometric_sequence_seventh_term (a r : ℕ) (h₁ : a = 6) (h₂ : a * r^4 = 486) : a * r^6 = 4374 :=
by
  -- The proof is not required, hence we use sorry.
  sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l301_30163


namespace NUMINAMATH_GPT_chameleons_color_change_l301_30180

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end NUMINAMATH_GPT_chameleons_color_change_l301_30180


namespace NUMINAMATH_GPT_shortest_player_height_l301_30126

-- let h_tall be the height of the tallest player
-- let h_short be the height of the shortest player
-- let diff be the height difference between the tallest and the shortest player

variable (h_tall h_short diff : ℝ)

-- conditions given in the problem
axiom tall_player_height : h_tall = 77.75
axiom height_difference : diff = 9.5
axiom height_relationship : h_tall = h_short + diff

-- the statement we need to prove
theorem shortest_player_height : h_short = 68.25 := by
  sorry

end NUMINAMATH_GPT_shortest_player_height_l301_30126


namespace NUMINAMATH_GPT_absolute_value_c_l301_30118

noncomputable def condition_polynomial (a b c : ℤ) : Prop :=
  a * (↑(Complex.ofReal 3) + Complex.I)^4 +
  b * (↑(Complex.ofReal 3) + Complex.I)^3 +
  c * (↑(Complex.ofReal 3) + Complex.I)^2 +
  b * (↑(Complex.ofReal 3) + Complex.I) +
  a = 0

noncomputable def coprime_integers (a b c : ℤ) : Prop :=
  Int.gcd (Int.gcd a b) c = 1

theorem absolute_value_c (a b c : ℤ) (h1 : condition_polynomial a b c) (h2 : coprime_integers a b c) :
  |c| = 97 :=
sorry

end NUMINAMATH_GPT_absolute_value_c_l301_30118


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l301_30182

def atomic_weight_Al : ℕ := 27
def atomic_weight_I : ℕ := 127
def atomic_weight_O : ℕ := 16

def num_Al : ℕ := 1
def num_I : ℕ := 3
def num_O : ℕ := 2

def molecular_weight (n_Al n_I n_O w_Al w_I w_O : ℕ) : ℕ :=
  (n_Al * w_Al) + (n_I * w_I) + (n_O * w_O)

theorem molecular_weight_of_compound :
  molecular_weight num_Al num_I num_O atomic_weight_Al atomic_weight_I atomic_weight_O = 440 := 
sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l301_30182


namespace NUMINAMATH_GPT_number_of_sequences_l301_30145

theorem number_of_sequences : 
  let n : ℕ := 7
  let ones : ℕ := 5
  let twos : ℕ := 2
  let comb := Nat.choose
  (ones + twos = n) ∧  
  comb (ones + 1) twos + comb (ones + 1) (twos - 1) = 21 := 
  by sorry

end NUMINAMATH_GPT_number_of_sequences_l301_30145


namespace NUMINAMATH_GPT_total_income_l301_30184

theorem total_income (I : ℝ) 
  (h1 : I * 0.225 = 40000) : 
  I = 177777.78 :=
by
  sorry

end NUMINAMATH_GPT_total_income_l301_30184


namespace NUMINAMATH_GPT_symmetric_line_equation_wrt_x_axis_l301_30191

theorem symmetric_line_equation_wrt_x_axis :
  (∀ x y : ℝ, 3 * x + 4 * y + 5 = 0 ↔ 3 * x - 4 * (-y) + 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_equation_wrt_x_axis_l301_30191


namespace NUMINAMATH_GPT_max_value_2ab_sqrt2_plus_2ac_l301_30138

theorem max_value_2ab_sqrt2_plus_2ac (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 1) : 
  2 * a * b * Real.sqrt 2 + 2 * a * c ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_2ab_sqrt2_plus_2ac_l301_30138


namespace NUMINAMATH_GPT_calculate_fraction_l301_30135

theorem calculate_fraction :
  ( (12^4 + 484) * (24^4 + 484) * (36^4 + 484) * (48^4 + 484) * (60^4 + 484) )
  /
  ( (6^4 + 484) * (18^4 + 484) * (30^4 + 484) * (42^4 + 484) * (54^4 + 484) )
  = 181 := by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l301_30135


namespace NUMINAMATH_GPT_arithmetic_sequence_m_value_l301_30159

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 0 + a (n - 1))) / 2

noncomputable def find_m (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : Prop :=
  (a (m + 1) + a (m - 1) - a m ^ 2 = 0) → (S (2 * m - 1) = 38) → m = 10

-- Problem Statement
theorem arithmetic_sequence_m_value :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ),
    arithmetic_sequence a → 
    sum_of_first_n_terms S a → 
    find_m a S m :=
by
  intros a S m ha hs h₁ h₂
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_m_value_l301_30159


namespace NUMINAMATH_GPT_correct_equation_for_tournament_l301_30146

theorem correct_equation_for_tournament (x : ℕ) (h : x * (x - 1) / 2 = 28) : True :=
sorry

end NUMINAMATH_GPT_correct_equation_for_tournament_l301_30146


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l301_30142

theorem arithmetic_geometric_sequence (a b : ℝ)
  (h1 : 2 * a = 1 + b)
  (h2 : b^2 = a)
  (h3 : a ≠ b) : a = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l301_30142


namespace NUMINAMATH_GPT_cyclist_traveled_18_miles_l301_30189

noncomputable def cyclist_distance (v t d : ℕ) : Prop :=
  (d = v * t) ∧ 
  (d = (v + 1) * (3 * t / 4)) ∧ 
  (d = (v - 1) * (t + 3))

theorem cyclist_traveled_18_miles : ∃ (d : ℕ), cyclist_distance 3 6 d ∧ d = 18 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_traveled_18_miles_l301_30189


namespace NUMINAMATH_GPT_solve_problem_l301_30164

variable (f : ℝ → ℝ)

axiom f_property : ∀ x : ℝ, f (x + 1) = x^2 - 2 * x

theorem solve_problem : f 2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l301_30164


namespace NUMINAMATH_GPT_solution_set_inequality_range_of_m_l301_30175

def f (x : ℝ) : ℝ := |2 * x + 1| + 2 * |x - 3|

theorem solution_set_inequality :
  ∀ x : ℝ, f x ≤ 7 * x ↔ x ≥ 1 :=
by sorry

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f x = |m|) ↔ (m ≥ 7 ∨ m ≤ -7) :=
by sorry

end NUMINAMATH_GPT_solution_set_inequality_range_of_m_l301_30175


namespace NUMINAMATH_GPT_exists_sum_of_two_squares_l301_30185

theorem exists_sum_of_two_squares (n : ℤ) (h : n > 10000) : ∃ m : ℤ, (∃ a b : ℤ, m = a^2 + b^2) ∧ 0 < m - n ∧ m - n < 3 * n^(1/4) :=
by
  sorry

end NUMINAMATH_GPT_exists_sum_of_two_squares_l301_30185


namespace NUMINAMATH_GPT_avg_marks_l301_30144

theorem avg_marks (P C M : ℕ) (h : P + C + M = P + 150) : (C + M) / 2 = 75 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_avg_marks_l301_30144


namespace NUMINAMATH_GPT_rectangle_area_l301_30160

theorem rectangle_area (w l : ℕ) (h1 : l = w + 8) (h2 : 2 * l + 2 * w = 176) :
  l * w = 1920 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l301_30160


namespace NUMINAMATH_GPT_value_of_g_at_3_l301_30186

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := 5 * x^3 - 6 * x^2 - 3 * x + 5

-- The theorem statement
theorem value_of_g_at_3 : g 3 = 77 := by
  -- This would require a proof, but we put sorry as instructed
  sorry

end NUMINAMATH_GPT_value_of_g_at_3_l301_30186


namespace NUMINAMATH_GPT_polynomial_proof_l301_30115

noncomputable def f (x a b c : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - a*b*c

theorem polynomial_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) (h3 : f a a b c = 0) (h4 : f b a b c = 0) (h5 : f c a b c = 0) : 
  f 0 a b c * f 1 a b c < 0 ∧ f 0 a b c * f 3 a b c > 0 :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_proof_l301_30115


namespace NUMINAMATH_GPT_prove_AF_eq_l301_30183

-- Definitions
variables {A B C E F : Type*}
variables [Field A] [Field B] [Field C] [Field E] [Field F]

-- Conditions
def triangle_ABC (AB AC : ℝ) (h : AB > AC) : Prop := true

def external_bisector (angleA : ℝ) (circumcircle_meets : ℝ) : Prop := true

def foot_perpendicular (E AB : ℝ) : Prop := true

-- Theorem statement
theorem prove_AF_eq (AB AC AF : ℝ) (h_triangle : triangle_ABC AB AC (by sorry))
  (h_external_bisector : external_bisector (by sorry) (by sorry))
  (h_foot_perpendicular : foot_perpendicular (by sorry) AB) :
  2 * AF = AB - AC := by
  sorry

end NUMINAMATH_GPT_prove_AF_eq_l301_30183


namespace NUMINAMATH_GPT_polynomial_coefficient_l301_30130

theorem polynomial_coefficient :
  ∀ d : ℝ, (2 * (2 : ℝ)^4 + 3 * (2 : ℝ)^3 + d * (2 : ℝ)^2 - 4 * (2 : ℝ) + 15 = 0) ↔ (d = -15.75) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coefficient_l301_30130


namespace NUMINAMATH_GPT_find_extrema_l301_30133

theorem find_extrema (x y : ℝ) (h1 : x < 0) (h2 : -1 < y) (h3 : y < 0) : 
  max (max x (x*y)) (x*y^2) = x*y ∧ min (min x (x*y)) (x*y^2) = x :=
by sorry

end NUMINAMATH_GPT_find_extrema_l301_30133


namespace NUMINAMATH_GPT_edge_length_of_cube_l301_30148

/-- Define the total paint volume, remaining paint and cube paint volume -/
def total_paint_volume : ℕ := 25 * 40
def remaining_paint : ℕ := 271
def cube_paint_volume : ℕ := total_paint_volume - remaining_paint

/-- Define the volume of the cube and the statement for edge length of the cube -/
theorem edge_length_of_cube (s : ℕ) : s^3 = cube_paint_volume → s = 9 :=
by
  have h1 : cube_paint_volume = 729 := by rfl
  sorry

end NUMINAMATH_GPT_edge_length_of_cube_l301_30148


namespace NUMINAMATH_GPT_union_complement_A_B_l301_30155

-- Definitions based on conditions
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | x < 6}
def C_R (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

-- The proof problem statement
theorem union_complement_A_B :
  (C_R B ∪ A = {x | 0 ≤ x}) :=
by 
  sorry

end NUMINAMATH_GPT_union_complement_A_B_l301_30155


namespace NUMINAMATH_GPT_august_8th_is_saturday_l301_30147

-- Defining the conditions
def august_has_31_days : Prop := true

def august_has_5_mondays : Prop := true

def august_has_4_tuesdays : Prop := true

-- Statement of the theorem
theorem august_8th_is_saturday (h1 : august_has_31_days) (h2 : august_has_5_mondays) (h3 : august_has_4_tuesdays) : ∃ d : ℕ, d = 6 :=
by
  -- Translate the correct answer "August 8th is a Saturday" into the equivalent proposition
  -- Saturday is represented by 6 if we assume 0 = Sunday, 1 = Monday, ..., 6 = Saturday.
  sorry

end NUMINAMATH_GPT_august_8th_is_saturday_l301_30147


namespace NUMINAMATH_GPT_area_of_shaded_region_l301_30106

noncomputable def shaded_region_area (β : ℝ) (cos_beta : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) : ℝ :=
  let sine_beta := Real.sqrt (1 - (3 / 5)^2)
  let tan_half_beta := sine_beta / (1 + 3 / 5)
  let bp := Real.tan (π / 4 - tan_half_beta)
  2 * (1 / 5) + 2 * (1 / 5)

theorem area_of_shaded_region (β : ℝ) (h : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) :
  shaded_region_area β h = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l301_30106


namespace NUMINAMATH_GPT_inequality_solution_set_l301_30117

noncomputable def f (x : ℝ) : ℝ := x + 1 / (2 * x) + 2

lemma f_increasing {x₁ x₂ : ℝ} (hx₁ : 1 ≤ x₁) (hx₂ : 1 ≤ x₂) (h : x₁ < x₂) : f x₁ < f x₂ := sorry

lemma solve_inequality (x : ℝ) (hx : 1 ≤ x) : (2 * x - 1 / 2 < x + 1007) → (f (2 * x - 1 / 2) < f (x + 1007)) := sorry

theorem inequality_solution_set {x : ℝ} : (1 ≤ x) → (2 * x - 1 / 2 < x + 1007) ↔ (3 / 4 ≤ x ∧ x < 2015 / 2) := sorry

end NUMINAMATH_GPT_inequality_solution_set_l301_30117


namespace NUMINAMATH_GPT_arrangement_count_equivalent_problem_l301_30129

noncomputable def number_of_unique_arrangements : Nat :=
  let n : Nat := 6 -- Number of balls and boxes
  let match_3_boxes_ways := Nat.choose n 3 -- Choosing 3 boxes out of 6
  let permute_remaining_boxes := 2 -- Permutations of the remaining 3 boxes such that no numbers match
  match_3_boxes_ways * permute_remaining_boxes

theorem arrangement_count_equivalent_problem :
  number_of_unique_arrangements = 40 := by
  sorry

end NUMINAMATH_GPT_arrangement_count_equivalent_problem_l301_30129


namespace NUMINAMATH_GPT_possible_values_of_D_plus_E_l301_30197

theorem possible_values_of_D_plus_E 
  (D E : ℕ) 
  (hD : 0 ≤ D ∧ D ≤ 9) 
  (hE : 0 ≤ E ∧ E ≤ 9) 
  (hdiv : (D + 8 + 6 + 4 + E + 7 + 2) % 9 = 0) : 
  D + E = 0 ∨ D + E = 9 ∨ D + E = 18 := 
sorry

end NUMINAMATH_GPT_possible_values_of_D_plus_E_l301_30197


namespace NUMINAMATH_GPT_comprehensive_score_correct_l301_30141

def comprehensive_score
  (study_score hygiene_score discipline_score participation_score : ℕ)
  (study_weight hygiene_weight discipline_weight participation_weight : ℚ) : ℚ :=
  study_score * study_weight +
  hygiene_score * hygiene_weight +
  discipline_score * discipline_weight +
  participation_score * participation_weight

theorem comprehensive_score_correct :
  let study_score := 80
  let hygiene_score := 90
  let discipline_score := 84
  let participation_score := 70
  let study_weight := 0.4
  let hygiene_weight := 0.25
  let discipline_weight := 0.25
  let participation_weight := 0.1
  comprehensive_score study_score hygiene_score discipline_score participation_score
                      study_weight hygiene_weight discipline_weight participation_weight
  = 82.5 :=
by 
  sorry

#eval comprehensive_score 80 90 84 70 0.4 0.25 0.25 0.1  -- output should be 82.5

end NUMINAMATH_GPT_comprehensive_score_correct_l301_30141


namespace NUMINAMATH_GPT_largest_x_floor_condition_l301_30128

theorem largest_x_floor_condition :
  ∃ x : ℝ, (⌊x⌋ : ℝ) / x = 8 / 9 ∧
      (∀ y : ℝ, (⌊y⌋ : ℝ) / y = 8 / 9 → y ≤ x) →
  x = 63 / 8 :=
by
  sorry

end NUMINAMATH_GPT_largest_x_floor_condition_l301_30128


namespace NUMINAMATH_GPT_new_angle_after_rotation_l301_30157

def initial_angle : ℝ := 25
def rotation_clockwise : ℝ := 350
def equivalent_rotation := rotation_clockwise - 360  -- equivalent to -10 degrees

theorem new_angle_after_rotation :
  initial_angle + equivalent_rotation = 15 := by
  sorry

end NUMINAMATH_GPT_new_angle_after_rotation_l301_30157


namespace NUMINAMATH_GPT_triangle_inequality_l301_30170

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (A / 2)) + Real.sqrt 3 * Real.tan (A / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (B / 2)) + Real.sqrt 3 * Real.tan (B / 2)) +
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (B / 2)) + Real.sqrt 3 * Real.tan (B / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (C / 2)) + Real.sqrt 3 * Real.tan (C / 2)) +
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (C / 2)) + Real.sqrt 3 * Real.tan (C / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (A / 2)) + Real.sqrt 3 * Real.tan (A / 2)) ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l301_30170


namespace NUMINAMATH_GPT_number_of_seasons_l301_30188

theorem number_of_seasons 
        (episodes_per_season : ℕ) 
        (fraction_watched : ℚ) 
        (remaining_episodes : ℕ) 
        (h_episodes_per_season : episodes_per_season = 20) 
        (h_fraction_watched : fraction_watched = 1 / 3) 
        (h_remaining_episodes : remaining_episodes = 160) : 
        ∃ (seasons : ℕ), seasons = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_seasons_l301_30188


namespace NUMINAMATH_GPT_plane_through_points_eq_l301_30149

-- Define the points M, N, P
def M := (1, 2, 0)
def N := (1, -1, 2)
def P := (0, 1, -1)

-- Define the target plane equation
def target_plane_eq (x y z : ℝ) := 5 * x - 2 * y + 3 * z - 1 = 0

-- Main theorem statement
theorem plane_through_points_eq :
  ∀ (x y z : ℝ),
    (∃ A B C : ℝ,
      A * (x - 1) + B * (y - 2) + C * z = 0 ∧
      A * (1 - 1) + B * (-1 - 2) + C * (2 - 0) = 0 ∧
      A * (0 - 1) + B * (1 - 2) + C * (-1 - 0) = 0) →
    target_plane_eq x y z :=
by
  sorry

end NUMINAMATH_GPT_plane_through_points_eq_l301_30149


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_for_circle_l301_30140

theorem sufficient_but_not_necessary_for_circle (m : ℝ) :
  (m = 0 → ∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + m = 0) ∧ ¬(∀m, ∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + m = 0 → m = 0) :=
 by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_for_circle_l301_30140


namespace NUMINAMATH_GPT_perimeter_inequality_l301_30161

-- Define the problem parameters
variables {R S : ℝ}  -- radius and area of the inscribed polygon
variables (P : ℝ)    -- perimeter of the convex polygon formed by chosen points

-- Define the various conditions
def circle_with_polygon (r : ℝ) := r > 0 -- Circle with positive radius
def polygon_with_area (s : ℝ) := s > 0 -- Polygon with positive area

-- Main theorem to be proven
theorem perimeter_inequality (hR : circle_with_polygon R) (hS : polygon_with_area S) :
  P ≥ (2 * S / R) :=
sorry

end NUMINAMATH_GPT_perimeter_inequality_l301_30161


namespace NUMINAMATH_GPT_simplify_expression_l301_30169

variable {a : ℝ}

theorem simplify_expression (h₁ : a ≠ 0) (h₂ : a ≠ -1) (h₃ : a ≠ 1) :
  ( ( (a^2 + 1) / a - 2 ) / ( (a^2 - 1) / (a^2 + a) ) ) = a - 1 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l301_30169


namespace NUMINAMATH_GPT_correct_sampling_method_order_l301_30187

-- Definitions for sampling methods
def simple_random_sampling (method : ℕ) : Bool :=
  method = 1

def systematic_sampling (method : ℕ) : Bool :=
  method = 2

def stratified_sampling (method : ℕ) : Bool :=
  method = 3

-- Main theorem stating the correct method order
theorem correct_sampling_method_order : simple_random_sampling 1 ∧ stratified_sampling 3 ∧ systematic_sampling 2 :=
by
  sorry

end NUMINAMATH_GPT_correct_sampling_method_order_l301_30187


namespace NUMINAMATH_GPT_distance_between_points_eq_l301_30192

noncomputable def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_between_points_eq :
  dist 1 5 7 2 = 3 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_eq_l301_30192


namespace NUMINAMATH_GPT_sin_square_alpha_minus_pi_div_4_l301_30121

theorem sin_square_alpha_minus_pi_div_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α - Real.pi / 4) ^ 2 = 1 / 6 := 
sorry

end NUMINAMATH_GPT_sin_square_alpha_minus_pi_div_4_l301_30121


namespace NUMINAMATH_GPT_sum_of_numbers_in_50th_row_l301_30158

-- Defining the array and the row sum
def row_sum (n : ℕ) : ℕ :=
  2^n

-- Proposition stating that the 50th row sum is equal to 2^50
theorem sum_of_numbers_in_50th_row : row_sum 50 = 2^50 :=
by sorry

end NUMINAMATH_GPT_sum_of_numbers_in_50th_row_l301_30158


namespace NUMINAMATH_GPT_cyclic_inequality_l301_30122

theorem cyclic_inequality
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (a^3 + b^3 + a * b * c) + 1 / (b^3 + c^3 + a * b * c) + 1 / (c^3 + a^3 + a * b * c)) ≤ 1 / (a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_cyclic_inequality_l301_30122


namespace NUMINAMATH_GPT_value_of_f_g_10_l301_30176

def g (x : ℤ) : ℤ := 4 * x + 6
def f (x : ℤ) : ℤ := 6 * x - 10

theorem value_of_f_g_10 : f (g 10) = 266 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_g_10_l301_30176


namespace NUMINAMATH_GPT_range_of_a_for_two_zeros_l301_30114

theorem range_of_a_for_two_zeros (a : ℝ) :
  (∀ x : ℝ, (x + 1) * Real.exp x - a = 0 → -- There's no need to delete this part, see below note 
                                              -- The question of "exactly" is virtually ensured by other parts of the Lean theories
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
                (x₁ + 1) * Real.exp x₁ - a = 0 ∧
                (x₂ + 1) * Real.exp x₂ - a = 0) → 
  (-1 / Real.exp 2 < a ∧ a < 0) :=
sorry

end NUMINAMATH_GPT_range_of_a_for_two_zeros_l301_30114


namespace NUMINAMATH_GPT_max_wickets_bowler_can_take_l301_30105

noncomputable def max_wickets_per_over : ℕ := 3
noncomputable def overs_bowled : ℕ := 6
noncomputable def max_possible_wickets := max_wickets_per_over * overs_bowled

theorem max_wickets_bowler_can_take : max_possible_wickets = 18 → max_possible_wickets == 10 :=
by
  sorry

end NUMINAMATH_GPT_max_wickets_bowler_can_take_l301_30105


namespace NUMINAMATH_GPT_evaluate_expression_l301_30120

variables (x : ℝ)

theorem evaluate_expression :
  x * (x * (x * (3 - x) - 5) + 13) + 1 = -x^4 + 3*x^3 - 5*x^2 + 13*x + 1 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l301_30120


namespace NUMINAMATH_GPT_sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l301_30132

theorem sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30 :
  (7^30 + 13^30) % 100 = 0 := 
sorry

end NUMINAMATH_GPT_sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l301_30132


namespace NUMINAMATH_GPT_f_2007_l301_30150

noncomputable def f : ℕ → ℝ :=
  sorry

axiom functional_eq (x y : ℕ) : f (x + y) = f x * f y

axiom f_one : f 1 = 2

theorem f_2007 : f 2007 = 2 ^ 2007 :=
by
  sorry

end NUMINAMATH_GPT_f_2007_l301_30150


namespace NUMINAMATH_GPT_least_positive_integer_divisible_by_three_smallest_primes_greater_than_five_l301_30131

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_primes_greater_than_five : List ℕ :=
  [7, 11, 13]

theorem least_positive_integer_divisible_by_three_smallest_primes_greater_than_five : 
  ∃ n : ℕ, n > 0 ∧ (∀ p ∈ smallest_primes_greater_than_five, p ∣ n) ∧ n = 1001 := by
  sorry

end NUMINAMATH_GPT_least_positive_integer_divisible_by_three_smallest_primes_greater_than_five_l301_30131


namespace NUMINAMATH_GPT_common_focus_hyperbola_ellipse_l301_30193

theorem common_focus_hyperbola_ellipse (p : ℝ) (c : ℝ) :
  (0 < p ∧ p < 8) →
  (c = Real.sqrt (3 + 1)) →
  (c = Real.sqrt (8 - p)) →
  p = 4 := by
sorry

end NUMINAMATH_GPT_common_focus_hyperbola_ellipse_l301_30193


namespace NUMINAMATH_GPT_solution_set_of_inequality_l301_30103

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) * (x - 2) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l301_30103


namespace NUMINAMATH_GPT_shaded_area_correct_l301_30136

noncomputable def side_length : ℝ := 24
noncomputable def radius : ℝ := side_length / 4
noncomputable def area_of_square : ℝ := side_length ^ 2
noncomputable def area_of_one_circle : ℝ := Real.pi * radius ^ 2
noncomputable def total_area_of_circles : ℝ := 5 * area_of_one_circle
noncomputable def shaded_area : ℝ := area_of_square - total_area_of_circles

theorem shaded_area_correct :
  shaded_area = 576 - 180 * Real.pi := by
  sorry

end NUMINAMATH_GPT_shaded_area_correct_l301_30136


namespace NUMINAMATH_GPT_A_share_in_profit_l301_30153

/-
Given:
1. a_contribution (A's amount contributed in Rs. 5000) and duration (in months 8)
2. b_contribution (B's amount contributed in Rs. 6000) and duration (in months 5)
3. total_profit (Total profit in Rs. 8400)

Prove that A's share in the total profit is Rs. 4800.
-/

theorem A_share_in_profit 
  (a_contribution : ℝ) (a_months : ℝ) 
  (b_contribution : ℝ) (b_months : ℝ) 
  (total_profit : ℝ) :
  a_contribution = 5000 → 
  a_months = 8 → 
  b_contribution = 6000 → 
  b_months = 5 → 
  total_profit = 8400 → 
  (a_contribution * a_months / (a_contribution * a_months + b_contribution * b_months) * total_profit) = 4800 := 
by {
  sorry
}

end NUMINAMATH_GPT_A_share_in_profit_l301_30153


namespace NUMINAMATH_GPT_octal_sum_l301_30167

open Nat

def octal_to_decimal (oct : ℕ) : ℕ :=
  match oct with
  | 0 => 0
  | n => let d3 := (n / 100) % 10
         let d2 := (n / 10) % 10
         let d1 := n % 10
         d3 * 8^2 + d2 * 8^1 + d1 * 8^0

def decimal_to_octal (dec : ℕ) : ℕ :=
  let rec aux (n : ℕ) (mul : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 8) (mul * 10) (acc + (n % 8) * mul)
  aux dec 1 0

theorem octal_sum :
  let a := 451
  let b := 167
  octal_to_decimal 451 + octal_to_decimal 167 = octal_to_decimal 640 := sorry

end NUMINAMATH_GPT_octal_sum_l301_30167


namespace NUMINAMATH_GPT_whole_numbers_count_between_cubic_roots_l301_30139

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end NUMINAMATH_GPT_whole_numbers_count_between_cubic_roots_l301_30139


namespace NUMINAMATH_GPT_percentage_2x_minus_y_of_x_l301_30111

noncomputable def x_perc_of_2x_minus_y (x y z : ℤ) (h1 : x / y = 4) (h2 : x + y = z) (h3 : z > 0) (h4 : y ≠ 0) : ℤ :=
  (2 * x - y) * 100 / x

theorem percentage_2x_minus_y_of_x (x y z : ℤ) (h1 : x / y = 4) (h2 : x + y = z) (h3 : z > 0) (h4 : y ≠ 0) :
  x_perc_of_2x_minus_y x y z h1 h2 h3 h4 = 175 :=
  sorry

end NUMINAMATH_GPT_percentage_2x_minus_y_of_x_l301_30111


namespace NUMINAMATH_GPT_jose_investment_l301_30190

theorem jose_investment 
  (T_investment : ℕ := 30000) -- Tom's investment in Rs.
  (J_months : ℕ := 10)        -- Jose's investment period in months
  (T_months : ℕ := 12)        -- Tom's investment period in months
  (total_profit : ℕ := 72000) -- Total profit in Rs.
  (jose_profit : ℕ := 40000)  -- Jose's share of profit in Rs.
  : ∃ X : ℕ, (jose_profit * (T_investment * T_months)) = ((total_profit - jose_profit) * (X * J_months)) ∧ X = 45000 :=
  sorry

end NUMINAMATH_GPT_jose_investment_l301_30190


namespace NUMINAMATH_GPT_find_common_difference_l301_30199

theorem find_common_difference
  (a_1 : ℕ := 1)
  (S : ℕ → ℕ)
  (h1 : S 5 = 20)
  (h2 : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d))
  : d = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_common_difference_l301_30199


namespace NUMINAMATH_GPT_swim_team_more_people_l301_30123

theorem swim_team_more_people :
  let car1_people := 5
  let car2_people := 4
  let van1_people := 3
  let van2_people := 3
  let van3_people := 5
  let minibus_people := 10

  let car_max_capacity := 6
  let van_max_capacity := 8
  let minibus_max_capacity := 15

  let actual_people := car1_people + car2_people + van1_people + van2_people + van3_people + minibus_people
  let max_capacity := 2 * car_max_capacity + 3 * van_max_capacity + minibus_max_capacity
  (max_capacity - actual_people : ℕ) = 21 := 
  by
    sorry

end NUMINAMATH_GPT_swim_team_more_people_l301_30123


namespace NUMINAMATH_GPT_percent_of_g_is_a_l301_30168

-- Definitions of the seven consecutive numbers
def consecutive_7_avg_9 (a b c d e f g : ℝ) : Prop :=
  a + b + c + d + e + f + g = 63

def is_median (d : ℝ) : Prop :=
  d = 9

def express_numbers (a b c d e f g : ℝ) : Prop :=
  a = d - 3 ∧ b = d - 2 ∧ c = d - 1 ∧ d = d ∧ e = d + 1 ∧ f = d + 2 ∧ g = d + 3

-- Main statement asserting the percentage relationship
theorem percent_of_g_is_a (a b c d e f g : ℝ) (h_avg : consecutive_7_avg_9 a b c d e f g)
  (h_median : is_median d) (h_express : express_numbers a b c d e f g) :
  (a / g) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percent_of_g_is_a_l301_30168


namespace NUMINAMATH_GPT_square_inscription_l301_30152

theorem square_inscription (a b : ℝ) (s1 s2 : ℝ)
  (h_eq_side_smaller : s1 = 4)
  (h_eq_side_larger : s2 = 3 * Real.sqrt 2)
  (h_sum_segments : a + b = s2)
  (h_eq_sum_squares : a^2 + b^2 = (4 * Real.sqrt 2)^2) :
  a * b = -7 := 
by sorry

end NUMINAMATH_GPT_square_inscription_l301_30152


namespace NUMINAMATH_GPT_problem_statement_l301_30107

noncomputable def roots (a b : ℝ) (coef1 coef2 : ℝ) :=
  ∃ x : ℝ, (x = a ∨ x = b) ∧ x^2 + coef1 * x + coef2 = 0

theorem problem_statement
  (a b c d : ℝ)
  (h1 : a + b = -57)
  (h2 : a * b = 1)
  (h3 : c + d = 57)
  (h4 : c * d = 1) :
  (a + c) * (b + c) * (a - d) * (b - d) = 0 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l301_30107


namespace NUMINAMATH_GPT_annual_interest_rate_l301_30154

theorem annual_interest_rate (principal total_paid: ℝ) (h_principal : principal = 150) (h_total_paid : total_paid = 162) : 
  ((total_paid - principal) / principal) * 100 = 8 :=
by
  sorry

end NUMINAMATH_GPT_annual_interest_rate_l301_30154


namespace NUMINAMATH_GPT_observe_three_cell_types_l301_30165

def biology_experiment
  (material : Type) (dissociation_fixative : material) (acetic_orcein_stain : material) (press_slide : Prop) : Prop :=
  ∃ (testes : material) (steps : material → Prop),
    steps testes ∧ press_slide ∧ (steps dissociation_fixative) ∧ (steps acetic_orcein_stain)

theorem observe_three_cell_types (material : Type)
  (dissociation_fixative acetic_orcein_stain : material)
  (press_slide : Prop)
  (steps : material → Prop) :
  biology_experiment material dissociation_fixative acetic_orcein_stain press_slide →
  ∃ (metaphase_of_mitosis metaphase_of_first_meiosis metaphase_of_second_meiosis : material), 
    steps metaphase_of_mitosis ∧ steps metaphase_of_first_meiosis ∧ steps metaphase_of_second_meiosis :=
sorry

end NUMINAMATH_GPT_observe_three_cell_types_l301_30165


namespace NUMINAMATH_GPT_f_is_decreasing_on_interval_l301_30181

noncomputable def f (x : ℝ) : ℝ := -3 * x ^ 2 - 2

theorem f_is_decreasing_on_interval :
  ∀ x y : ℝ, (1 ≤ x ∧ x < y ∧ y ≤ 2) → f y < f x :=
by
  sorry

end NUMINAMATH_GPT_f_is_decreasing_on_interval_l301_30181


namespace NUMINAMATH_GPT_units_digit_of_quotient_l301_30198

theorem units_digit_of_quotient : 
  (7 ^ 2023 + 4 ^ 2023) % 9 = 2 → 
  (7 ^ 2023 + 4 ^ 2023) / 9 % 10 = 0 :=
by
  -- condition: calculation of modulo result
  have h1 : (7 ^ 2023 + 4 ^ 2023) % 9 = 2 := sorry

  -- we have the target statement here
  exact sorry

end NUMINAMATH_GPT_units_digit_of_quotient_l301_30198


namespace NUMINAMATH_GPT_identify_false_statement_l301_30119

-- Definitions for the conditions
def isMultipleOf (n k : Nat) : Prop := ∃ m, n = k * m

def conditions : Prop :=
  isMultipleOf 12 2 ∧
  isMultipleOf 123 3 ∧
  isMultipleOf 1234 4 ∧
  isMultipleOf 12345 5 ∧
  isMultipleOf 123456 6

-- The statement which proves which condition is false
theorem identify_false_statement : conditions → ¬ (isMultipleOf 1234 4) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_identify_false_statement_l301_30119


namespace NUMINAMATH_GPT_brigade_harvest_time_l301_30156

theorem brigade_harvest_time (t : ℕ) :
  (t - 5 = (3 * t / 5) + ((t * (t - 8)) / (5 * (t - 4)))) → t = 20 := sorry

end NUMINAMATH_GPT_brigade_harvest_time_l301_30156


namespace NUMINAMATH_GPT_solve_fraction_eq_zero_l301_30151

theorem solve_fraction_eq_zero (x : ℝ) (h : x - 2 ≠ 0) : (x + 1) / (x - 2) = 0 ↔ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_eq_zero_l301_30151


namespace NUMINAMATH_GPT_max_d_n_l301_30134

def sequence_a (n : ℕ) : ℤ := 100 + n^2

def d_n (n : ℕ) : ℤ := Int.gcd (sequence_a n) (sequence_a (n + 1))

theorem max_d_n : ∃ n, d_n n = 401 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_max_d_n_l301_30134


namespace NUMINAMATH_GPT_calc_num_int_values_l301_30101

theorem calc_num_int_values (x : ℕ) (h : 121 ≤ x ∧ x < 144) : ∃ n : ℕ, n = 23 :=
by
  sorry

end NUMINAMATH_GPT_calc_num_int_values_l301_30101


namespace NUMINAMATH_GPT_largest_class_students_l301_30174

theorem largest_class_students (x : ℕ) (h1 : 8 * x - (4 + 8 + 12 + 16 + 20 + 24 + 28) = 380) : x = 61 :=
by
  sorry

end NUMINAMATH_GPT_largest_class_students_l301_30174


namespace NUMINAMATH_GPT_largest_reservoir_is_D_l301_30179

variables (a : ℝ) 
def final_amount_A : ℝ := a * (1 + 0.1) * (1 - 0.05)
def final_amount_B : ℝ := a * (1 + 0.09) * (1 - 0.04)
def final_amount_C : ℝ := a * (1 + 0.08) * (1 - 0.03)
def final_amount_D : ℝ := a * (1 + 0.07) * (1 - 0.02)

theorem largest_reservoir_is_D
  (hA : final_amount_A a = a * 1.045)
  (hB : final_amount_B a = a * 1.0464)
  (hC : final_amount_C a = a * 1.0476)
  (hD : final_amount_D a = a * 1.0486) :
  final_amount_D a > final_amount_A a ∧ 
  final_amount_D a > final_amount_B a ∧ 
  final_amount_D a > final_amount_C a :=
by sorry

end NUMINAMATH_GPT_largest_reservoir_is_D_l301_30179


namespace NUMINAMATH_GPT_jelly_bean_ratio_l301_30108

theorem jelly_bean_ratio 
  (Napoleon_jelly_beans : ℕ)
  (Sedrich_jelly_beans : ℕ)
  (Mikey_jelly_beans : ℕ)
  (h1 : Napoleon_jelly_beans = 17)
  (h2 : Sedrich_jelly_beans = Napoleon_jelly_beans + 4)
  (h3 : Mikey_jelly_beans = 19) :
  2 * (Napoleon_jelly_beans + Sedrich_jelly_beans) / Mikey_jelly_beans = 4 := 
sorry

end NUMINAMATH_GPT_jelly_bean_ratio_l301_30108


namespace NUMINAMATH_GPT_choir_meets_every_5_days_l301_30171

theorem choir_meets_every_5_days (n : ℕ) (h1 : n = 15) (h2 : ∃ k : ℕ, 15 = 3 * k) : ∃ x : ℕ, 15 = x * 3 ∧ x = 5 := 
by
  sorry

end NUMINAMATH_GPT_choir_meets_every_5_days_l301_30171


namespace NUMINAMATH_GPT_sarahs_score_is_140_l301_30125

theorem sarahs_score_is_140 (g s : ℕ) (h1 : s = g + 60) 
  (h2 : (s + g) / 2 = 110) (h3 : s + g < 450) : s = 140 :=
by
  sorry

end NUMINAMATH_GPT_sarahs_score_is_140_l301_30125


namespace NUMINAMATH_GPT_value_is_6_l301_30112

-- We know the conditions that the least number which needs an increment is 858
def least_number : ℕ := 858

-- Define the numbers 24, 32, 36, and 54
def num1 : ℕ := 24
def num2 : ℕ := 32
def num3 : ℕ := 36
def num4 : ℕ := 54

-- Define the LCM function to compute the least common multiple
def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

-- Define the LCM of the four numbers
def lcm_all : ℕ := lcm (lcm num1 num2) (lcm num3 num4)

-- Compute the value that needs to be added
def value_to_be_added : ℕ := lcm_all - least_number

-- Prove that this value equals to 6
theorem value_is_6 : value_to_be_added = 6 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_value_is_6_l301_30112


namespace NUMINAMATH_GPT_find_f_2011_l301_30172

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_2011 :
  (∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3 * x + 2) = 9 * x^2 - 15 * x) →
  f 2011 = 6029 :=
by
  intros hf
  sorry

end NUMINAMATH_GPT_find_f_2011_l301_30172


namespace NUMINAMATH_GPT_bus_children_count_l301_30162

theorem bus_children_count
  (initial_count : ℕ)
  (first_stop_add : ℕ)
  (second_stop_add : ℕ)
  (second_stop_remove : ℕ)
  (third_stop_remove : ℕ)
  (third_stop_add : ℕ)
  (final_count : ℕ)
  (h1 : initial_count = 18)
  (h2 : first_stop_add = 5)
  (h3 : second_stop_remove = 4)
  (h4 : third_stop_remove = 3)
  (h5 : third_stop_add = 5)
  (h6 : final_count = 25)
  (h7 : initial_count + first_stop_add = 23)
  (h8 : 23 + second_stop_add - second_stop_remove - third_stop_remove + third_stop_add = final_count) :
  second_stop_add = 4 :=
by
  sorry

end NUMINAMATH_GPT_bus_children_count_l301_30162


namespace NUMINAMATH_GPT_cars_count_l301_30194

-- Define the number of cars as x
variable (x : ℕ)

-- The conditions for the problem
def condition1 := 3 * (x - 2)
def condition2 := 2 * x + 9

-- The main theorem stating that under the given conditions, x = 15
theorem cars_count : condition1 x = condition2 x → x = 15 := by
  sorry

end NUMINAMATH_GPT_cars_count_l301_30194


namespace NUMINAMATH_GPT_probability_two_red_two_green_l301_30100

theorem probability_two_red_two_green (total_red total_blue total_green : ℕ)
  (total_marbles total_selected : ℕ) (probability : ℚ)
  (h_total_marbles: total_marbles = total_red + total_blue + total_green)
  (h_total_selected: total_selected = 4)
  (h_red_selected: 2 ≤ total_red)
  (h_green_selected: 2 ≤ total_green)
  (h_total_selected_le: total_selected ≤ total_marbles)
  (h_probability: probability = (Nat.choose total_red 2 * Nat.choose total_green 2) / (Nat.choose total_marbles total_selected))
  (h_total_red: total_red = 12)
  (h_total_blue: total_blue = 8)
  (h_total_green: total_green = 5):
  probability = 2 / 39 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_red_two_green_l301_30100


namespace NUMINAMATH_GPT_abcde_sum_l301_30143

theorem abcde_sum : 
  ∀ (a b c d e : ℝ), 
  a + 1 = b + 2 → 
  b + 2 = c + 3 → 
  c + 3 = d + 4 → 
  d + 4 = e + 5 → 
  e + 5 = a + b + c + d + e + 10 → 
  a + b + c + d + e = -35 / 4 :=
sorry

end NUMINAMATH_GPT_abcde_sum_l301_30143


namespace NUMINAMATH_GPT_bee_paths_to_hive_6_correct_l301_30196

noncomputable def num_paths_to_hive_6 : ℕ := 21

theorem bee_paths_to_hive_6_correct
  (start_pos : ℕ)
  (end_pos : ℕ)
  (bee_can_only_crawl : Prop)
  (bee_can_move_right : Prop)
  (bee_can_move_upper_right : Prop)
  (bee_can_move_lower_right : Prop)
  (total_hives : ℕ)
  (start_pos_is_initial : start_pos = 0)
  (end_pos_is_six : end_pos = 6) :
  num_paths_to_hive_6 = 21 :=
by
  sorry

end NUMINAMATH_GPT_bee_paths_to_hive_6_correct_l301_30196


namespace NUMINAMATH_GPT_find_y_l301_30137

-- Define the problem conditions
def avg_condition (y : ℝ) : Prop := (15 + 25 + y) / 3 = 23

-- Prove that the value of 'y' satisfying the condition is 29
theorem find_y (y : ℝ) (h : avg_condition y) : y = 29 :=
sorry

end NUMINAMATH_GPT_find_y_l301_30137


namespace NUMINAMATH_GPT_tan_45_add_reciprocal_half_add_abs_neg_two_eq_five_l301_30178

theorem tan_45_add_reciprocal_half_add_abs_neg_two_eq_five :
  (Real.tan (Real.pi / 4) + (1 / 2)⁻¹ + |(-2 : ℝ)|) = 5 :=
by
  -- Assuming the conditions provided in part a)
  have h1 : Real.tan (Real.pi / 4) = 1 := by sorry
  have h2 : (1 / 2 : ℝ)⁻¹ = 2 := by sorry
  have h3 : |(-2 : ℝ)| = 2 := by sorry

  -- Proof of the problem using the conditions
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_tan_45_add_reciprocal_half_add_abs_neg_two_eq_five_l301_30178


namespace NUMINAMATH_GPT_fraction_inspected_by_Jane_l301_30102

theorem fraction_inspected_by_Jane (P : ℝ) (x y : ℝ) 
    (h1: 0.007 * x * P + 0.008 * y * P = 0.0075 * P) 
    (h2: x + y = 1) : y = 0.5 :=
by sorry

end NUMINAMATH_GPT_fraction_inspected_by_Jane_l301_30102


namespace NUMINAMATH_GPT_alien_heads_l301_30110

theorem alien_heads (l o : ℕ) 
  (h1 : l + o = 60) 
  (h2 : 4 * l + o = 129) : 
  l + 2 * o = 97 := 
by 
  sorry

end NUMINAMATH_GPT_alien_heads_l301_30110
