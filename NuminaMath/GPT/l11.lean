import Mathlib

namespace distance_between_foci_of_hyperbola_l11_1100

theorem distance_between_foci_of_hyperbola :
  ∀ (x y : ℝ), x^2 - 6 * x - 4 * y^2 - 8 * y = 27 → (4 * Real.sqrt 10) = 4 * Real.sqrt 10 :=
by
  sorry

end distance_between_foci_of_hyperbola_l11_1100


namespace combined_area_difference_l11_1182

theorem combined_area_difference :
  let area_11x11 := 2 * (11 * 11)
  let area_5_5x11 := 2 * (5.5 * 11)
  area_11x11 - area_5_5x11 = 121 :=
by
  sorry

end combined_area_difference_l11_1182


namespace product_value_l11_1131

theorem product_value :
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
    -- Skipping the actual proof
    sorry

end product_value_l11_1131


namespace problem1_solution_problem2_solution_l11_1134

theorem problem1_solution (x : ℝ) : (x^2 - 4 * x = 5) → (x = 5 ∨ x = -1) :=
by sorry

theorem problem2_solution (x : ℝ) : (2 * x^2 - 3 * x + 1 = 0) → (x = 1 ∨ x = 1/2) :=
by sorry

end problem1_solution_problem2_solution_l11_1134


namespace real_roots_quadratic_l11_1174

theorem real_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 - 2 * k * x + k - 6 = 0) ↔ (k ≥ 1.5 ∧ k ≠ 2) :=
by {
  sorry
}

end real_roots_quadratic_l11_1174


namespace tileable_contains_domino_l11_1114

theorem tileable_contains_domino {m n a b : ℕ} (h_m : m ≥ a) (h_n : n ≥ b) :
  (∀ (x : ℕ) (y : ℕ), x + a ≤ m → y + b ≤ n → ∃ (p : ℕ) (q : ℕ), p = x ∧ q = y) :=
sorry

end tileable_contains_domino_l11_1114


namespace pow_mod_equality_l11_1171

theorem pow_mod_equality (h : 2^3 ≡ 1 [MOD 7]) : 2^30 ≡ 1 [MOD 7] :=
sorry

end pow_mod_equality_l11_1171


namespace line_ellipse_intersection_l11_1179

theorem line_ellipse_intersection (m : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + m ∧ (x^2 / 4 + y^2 / 2 = 1)) →
  (-3 * Real.sqrt 2 < m ∧ m < 3 * Real.sqrt 2) ∨
  (m = 3 * Real.sqrt 2 ∨ m = -3 * Real.sqrt 2) ∨ 
  (m < -3 * Real.sqrt 2 ∨ m > 3 * Real.sqrt 2) :=
sorry

end line_ellipse_intersection_l11_1179


namespace completely_factored_form_l11_1111

theorem completely_factored_form (x : ℤ) :
  (12 * x ^ 3 + 95 * x - 6) - (-3 * x ^ 3 + 5 * x - 6) = 15 * x * (x ^ 2 + 6) :=
by
  sorry

end completely_factored_form_l11_1111


namespace find_equation_for_second_machine_l11_1121

theorem find_equation_for_second_machine (x : ℝ) : 
  (1 / 6) + (1 / x) = 1 / 3 ↔ (x = 6) := 
by 
  sorry

end find_equation_for_second_machine_l11_1121


namespace correct_options_l11_1124

theorem correct_options (a b c : ℝ) (h1 : ∀ x : ℝ, (a*x^2 + b*x + c > 0) ↔ (-3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, (b*x + c > 0) ↔ x > 6) = False ∧ (∀ x, (c*x^2 + b*x + a < 0) ↔ (-1/3 < x ∧ x < 1/2)) :=
by 
  sorry

end correct_options_l11_1124


namespace smallest_natural_number_l11_1196

theorem smallest_natural_number (x : ℕ) : 
  (x % 5 = 2) ∧ (x % 6 = 2) ∧ (x % 7 = 3) → x = 122 := 
by
  sorry

end smallest_natural_number_l11_1196


namespace pencils_profit_goal_l11_1151

theorem pencils_profit_goal (n : ℕ) (price_purchase price_sale cost_goal : ℚ) (purchase_quantity : ℕ) 
  (h1 : price_purchase = 0.10) 
  (h2 : price_sale = 0.25) 
  (h3 : cost_goal = 100) 
  (h4 : purchase_quantity = 1500) 
  (h5 : n * price_sale ≥ purchase_quantity * price_purchase + cost_goal) :
  n ≥ 1000 :=
sorry

end pencils_profit_goal_l11_1151


namespace smallest_integer_expression_l11_1183

theorem smallest_integer_expression :
  ∃ m n : ℤ, 1237 * m + 78653 * n = 1 :=
sorry

end smallest_integer_expression_l11_1183


namespace tom_teaching_years_l11_1144

theorem tom_teaching_years :
  ∃ T D : ℕ, T + D = 70 ∧ D = (1 / 2) * T - 5 ∧ T = 50 :=
by
  sorry

end tom_teaching_years_l11_1144


namespace positive_integer_not_in_S_l11_1149

noncomputable def S : Set ℤ :=
  {n | ∃ (i : ℕ), n = 4^i * 3 ∨ n = -4^i * 2}

theorem positive_integer_not_in_S (n : ℤ) (hn : 0 < n) (hnS : n ∉ S) :
  ∃ (x y : ℤ), x ≠ y ∧ x ∈ S ∧ y ∈ S ∧ x + y = n :=
sorry

end positive_integer_not_in_S_l11_1149


namespace prove_n_prime_l11_1155

theorem prove_n_prime (n : ℕ) (p : ℕ) (k : ℕ) (hp : Prime p) (h1 : n > 0) (h2 : 3^n - 2^n = p^k) : Prime n :=
by {
  sorry
}

end prove_n_prime_l11_1155


namespace min_value_expr_l11_1102

theorem min_value_expr (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (hxyz : x * y * z = 1) :
  2 * x^2 + 8 * x * y + 6 * y^2 + 16 * y * z + 3 * z^2 ≥ 24 :=
by
  sorry

end min_value_expr_l11_1102


namespace cross_section_area_l11_1181

-- Definitions representing the conditions
variables (AK KD BP PC DM DC : ℝ)
variable (h : ℝ)
variable (Volume : ℝ)

-- Conditions
axiom hyp1 : AK = KD
axiom hyp2 : BP = PC
axiom hyp3 : DM = 0.4 * DC
axiom hyp4 : h = 1
axiom hyp5 : Volume = 5

-- Proof problem: Prove that the area S of the cross-section of the pyramid is 3
theorem cross_section_area (S : ℝ) : S = 3 :=
by sorry

end cross_section_area_l11_1181


namespace track_length_l11_1186

theorem track_length (V_A V_B V_C : ℝ) (x : ℝ) 
  (h1 : x / V_A = (x - 1) / V_B) 
  (h2 : x / V_A = (x - 2) / V_C) 
  (h3 : x / V_B = (x - 1.01) / V_C) : 
  110 - x = 9 :=
by 
  sorry

end track_length_l11_1186


namespace line_intersects_y_axis_l11_1116

-- Define the points
def P1 : ℝ × ℝ := (3, 18)
def P2 : ℝ × ℝ := (-9, -6)

-- State that the line passing through P1 and P2 intersects the y-axis at (0, 12)
theorem line_intersects_y_axis :
  ∃ y : ℝ, (∃ m b : ℝ, ∀ x : ℝ, y = m * x + b ∧ (m = (P2.2 - P1.2) / (P2.1 - P1.1)) ∧ (P1.2 = m * P1.1 + b) ∧ (x = 0) ∧ y = 12) :=
sorry

end line_intersects_y_axis_l11_1116


namespace sequence_is_decreasing_l11_1152

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

theorem sequence_is_decreasing (a : ℕ → ℝ) (h1 : a 1 < 0) (h2 : is_geometric_sequence a (1/3)) :
  ∀ n, a (n + 1) < a n :=
by
  -- Here should be the proof
  sorry

end sequence_is_decreasing_l11_1152


namespace johns_profit_l11_1192

-- Definitions based on Conditions
def original_price_per_bag : ℝ := 4
def discount_percentage : ℝ := 0.10
def discounted_price_per_bag := original_price_per_bag * (1 - discount_percentage)
def bags_bought : ℕ := 30
def cost_per_bag : ℝ := if bags_bought >= 20 then discounted_price_per_bag else original_price_per_bag
def total_cost := bags_bought * cost_per_bag
def bags_sold_to_adults : ℕ := 20
def bags_sold_to_children : ℕ := 10
def price_per_bag_for_adults : ℝ := 8
def price_per_bag_for_children : ℝ := 6
def revenue_from_adults := bags_sold_to_adults * price_per_bag_for_adults
def revenue_from_children := bags_sold_to_children * price_per_bag_for_children
def total_revenue := revenue_from_adults + revenue_from_children
def profit := total_revenue - total_cost

-- Lean Statement to be Proven
theorem johns_profit : profit = 112 :=
by
  sorry

end johns_profit_l11_1192


namespace river_length_l11_1188

theorem river_length (S C : ℝ) (h1 : S = C / 3) (h2 : S + C = 80) : S = 20 :=
by 
  sorry

end river_length_l11_1188


namespace cross_section_is_rectangle_l11_1137

def RegularTetrahedron : Type := sorry

def Plane : Type := sorry

variable (T : RegularTetrahedron) (P : Plane)

-- Conditions
axiom regular_tetrahedron (T : RegularTetrahedron) : Prop
axiom plane_intersects_tetrahedron (P : Plane) (T : RegularTetrahedron) : Prop
axiom plane_parallel_opposite_edges (P : Plane) (T : RegularTetrahedron) : Prop

-- The cross-section formed by intersecting a regular tetrahedron with a plane
-- that is parallel to two opposite edges is a rectangle.
theorem cross_section_is_rectangle (T : RegularTetrahedron) (P : Plane) 
  (hT : regular_tetrahedron T) 
  (hI : plane_intersects_tetrahedron P T) 
  (hP : plane_parallel_opposite_edges P T) :
  ∃ (shape : Type), shape = Rectangle := 
  sorry

end cross_section_is_rectangle_l11_1137


namespace original_population_l11_1165

theorem original_population (n : ℕ) (h1 : n + 1500 * 85 / 100 = n - 45) : n = 8800 := 
by
  sorry

end original_population_l11_1165


namespace rational_function_domain_l11_1180

noncomputable def h (x : ℝ) : ℝ := (x^3 - 3*x^2 - 4*x + 5) / (x^2 - 5*x + 4)

theorem rational_function_domain :
  {x : ℝ | ∃ y, h y = h x } = {x : ℝ | x ≠ 1 ∧ x ≠ 4} := 
sorry

end rational_function_domain_l11_1180


namespace father_age_l11_1161

theorem father_age (M F : ℕ) 
  (h1 : M = 2 * F / 5) 
  (h2 : M + 10 = (F + 10) / 2) : F = 50 :=
sorry

end father_age_l11_1161


namespace exams_in_fourth_year_l11_1162

noncomputable def student_exam_counts 
  (a_1 a_2 a_3 a_4 a_5 : ℕ) : Prop :=
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 ∧ 
  a_5 = 3 * a_1 ∧ 
  a_1 < a_2 ∧ 
  a_2 < a_3 ∧ 
  a_3 < a_4 ∧ 
  a_4 < a_5

theorem exams_in_fourth_year 
  (a_1 a_2 a_3 a_4 a_5 : ℕ) (h : student_exam_counts a_1 a_2 a_3 a_4 a_5) : 
  a_4 = 8 :=
sorry

end exams_in_fourth_year_l11_1162


namespace total_ranking_sequences_at_end_l11_1176

-- Define the teams
inductive Team
| E
| F
| G
| H

open Team

-- Conditions of the problem
def split_groups : (Team × Team) × (Team × Team) :=
  ((E, F), (G, H))

def saturday_matches : (Team × Team) × (Team × Team) :=
  ((E, F), (G, H))

-- Function to count total ranking sequences
noncomputable def total_ranking_sequences : ℕ := 4

-- Define the main theorem
theorem total_ranking_sequences_at_end : total_ranking_sequences = 4 :=
by
  sorry

end total_ranking_sequences_at_end_l11_1176


namespace james_total_catch_l11_1120

def pounds_of_trout : ℕ := 200
def pounds_of_salmon : ℕ := pounds_of_trout + (pounds_of_trout / 2)
def pounds_of_tuna : ℕ := 2 * pounds_of_salmon
def total_pounds_of_fish : ℕ := pounds_of_trout + pounds_of_salmon + pounds_of_tuna

theorem james_total_catch : total_pounds_of_fish = 1100 := by
  sorry

end james_total_catch_l11_1120


namespace max_dominoes_l11_1132

theorem max_dominoes (m n : ℕ) (h : n ≥ m) :
  ∃ k, k = m * n - (m / 2 : ℕ) :=
by sorry

end max_dominoes_l11_1132


namespace quadrilateral_sides_l11_1101

noncomputable def circle_radius : ℝ := 25
noncomputable def diagonal1_length : ℝ := 48
noncomputable def diagonal2_length : ℝ := 40

theorem quadrilateral_sides :
  ∃ (a b c d : ℝ),
    (a = 5 * Real.sqrt 10 ∧ 
    b = 9 * Real.sqrt 10 ∧ 
    c = 13 * Real.sqrt 10 ∧ 
    d = 15 * Real.sqrt 10) ∧ 
    (diagonal1_length = 48 ∧ 
    diagonal2_length = 40 ∧ 
    circle_radius = 25) :=
sorry

end quadrilateral_sides_l11_1101


namespace find_value_of_a_minus_b_l11_1123

variable (a b : ℝ)

theorem find_value_of_a_minus_b (h1 : |a| = 2) (h2 : b^2 = 9) (h3 : a < b) :
  a - b = -1 ∨ a - b = -5 := 
sorry

end find_value_of_a_minus_b_l11_1123


namespace average_price_of_racket_l11_1146

theorem average_price_of_racket
  (total_amount_made : ℝ)
  (number_of_pairs_sold : ℕ)
  (h1 : total_amount_made = 490) 
  (h2 : number_of_pairs_sold = 50) : 
  (total_amount_made / number_of_pairs_sold : ℝ) = 9.80 := 
  by
  sorry

end average_price_of_racket_l11_1146


namespace number_satisfies_equation_l11_1185

theorem number_satisfies_equation :
  ∃ x : ℝ, (x^2 + 100 = (x - 20)^2) ∧ x = 7.5 :=
by
  use 7.5
  sorry

end number_satisfies_equation_l11_1185


namespace intersection_correct_union_correct_intersection_complement_correct_l11_1125

def U := ℝ
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -3 ∨ x > 1}
def C_U_A : Set ℝ := {x | x ≤ 0 ∨ x > 2}
def C_U_B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

theorem intersection_correct : (A ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

theorem union_correct : (A ∪ B) = {x : ℝ | x < -3 ∨ x > 0} :=
sorry

theorem intersection_complement_correct : (C_U_A ∩ C_U_B) = {x : ℝ | -3 ≤ x ∧ x ≤ 0} :=
sorry

end intersection_correct_union_correct_intersection_complement_correct_l11_1125


namespace original_number_of_cats_l11_1158

theorem original_number_of_cats (C : ℕ) : 
  (C - 600) / 2 = 600 → C = 1800 :=
by
  sorry

end original_number_of_cats_l11_1158


namespace students_not_in_biology_l11_1103

theorem students_not_in_biology (total_students : ℕ) (percent_enrolled : ℝ) (students_enrolled : ℕ) (students_not_enrolled : ℕ) : 
  total_students = 880 ∧ percent_enrolled = 32.5 ∧ total_students - students_enrolled = students_not_enrolled ∧ students_enrolled = 286 ∧ students_not_enrolled = 594 :=
by
  sorry

end students_not_in_biology_l11_1103


namespace incorrect_statement_A_l11_1157

-- Define the statements based on conditions
def statementA : String := "INPUT \"MATH=\"; a+b+c"
def statementB : String := "PRINT \"MATH=\"; a+b+c"
def statementC : String := "a=b+c"
def statementD : String := "a=b-c"

-- Define a function to check if a statement is valid syntax
noncomputable def isValidSyntax : String → Prop :=
  λ stmt => 
    stmt = statementB ∨ stmt = statementC ∨ stmt = statementD

-- The proof problem
theorem incorrect_statement_A : ¬ isValidSyntax statementA :=
  sorry

end incorrect_statement_A_l11_1157


namespace train_speed_l11_1164

/-- 
A man sitting in a train which is traveling at a certain speed observes 
that a goods train, traveling in the opposite direction, takes 9 seconds 
to pass him. The goods train is 280 m long and its speed is 52 kmph. 
Prove that the speed of the train the man is sitting in is 60 kmph.
-/
theorem train_speed (t : ℝ) (h1 : 0 < t)
  (goods_speed_kmph : ℝ := 52)
  (goods_length_m : ℝ := 280)
  (time_seconds : ℝ := 9)
  (h2 : goods_length_m / time_seconds = (t + goods_speed_kmph) * (5 / 18)) :
  t = 60 :=
sorry

end train_speed_l11_1164


namespace point_A_in_Quadrant_IV_l11_1141

-- Define the coordinates of point A
def A : ℝ × ℝ := (5, -4)

-- Define the quadrants based on x and y signs
def in_Quadrant_I (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def in_Quadrant_II (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def in_Quadrant_III (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def in_Quadrant_IV (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Statement to prove that point A lies in Quadrant IV
theorem point_A_in_Quadrant_IV : in_Quadrant_IV A :=
by
  sorry

end point_A_in_Quadrant_IV_l11_1141


namespace differential_equation_solution_l11_1167

def C1 : ℝ := sorry
def C2 : ℝ := sorry

noncomputable def y (x : ℝ) : ℝ := C1 * Real.cos x + C2 * Real.sin x
noncomputable def z (x : ℝ) : ℝ := -C1 * Real.sin x + C2 * Real.cos x

theorem differential_equation_solution : 
  (∀ x : ℝ, deriv y x = z x) ∧ 
  (∀ x : ℝ, deriv z x = -y x) :=
by
  sorry

end differential_equation_solution_l11_1167


namespace quadratic_square_binomial_l11_1108

theorem quadratic_square_binomial (a : ℝ) :
  (∃ d : ℝ, 9 * x ^ 2 - 18 * x + a = (3 * x + d) ^ 2) → a = 9 :=
by
  intro h
  match h with
  | ⟨d, h_eq⟩ => sorry

end quadratic_square_binomial_l11_1108


namespace system_no_solution_iff_n_eq_neg_one_l11_1107

def no_solution_system (n : ℝ) : Prop :=
  ¬∃ x y z : ℝ, (n * x + y = 1) ∧ (n * y + z = 1) ∧ (x + n * z = 1)

theorem system_no_solution_iff_n_eq_neg_one (n : ℝ) : no_solution_system n ↔ n = -1 :=
sorry

end system_no_solution_iff_n_eq_neg_one_l11_1107


namespace rectangular_garden_side_length_l11_1145

theorem rectangular_garden_side_length (a b : ℝ) (h1 : 2 * a + 2 * b = 60) (h2 : a * b = 200) (h3 : b = 10) : a = 20 :=
by
  sorry

end rectangular_garden_side_length_l11_1145


namespace phone_numbers_even_phone_numbers_odd_phone_numbers_ratio_l11_1139

def even_digits : Set ℕ := { 0, 2, 4, 6, 8 }
def odd_digits : Set ℕ := { 1, 3, 5, 7, 9 }

theorem phone_numbers_even : (4 * 5^6) = 62500 := by
  sorry

theorem phone_numbers_odd : 5^7 = 78125 := by
  sorry

theorem phone_numbers_ratio
  (evens : (4 * 5^6) = 62500)
  (odds : 5^7 = 78125) :
  (78125 / 62500 : ℝ) = 1.25 := by
    sorry

end phone_numbers_even_phone_numbers_odd_phone_numbers_ratio_l11_1139


namespace brittany_average_correct_l11_1184

def brittany_first_score : ℤ :=
78

def brittany_second_score : ℤ :=
84

def brittany_average_after_second_test (score1 score2 : ℤ) : ℤ :=
(score1 + score2) / 2

theorem brittany_average_correct : 
  brittany_average_after_second_test brittany_first_score brittany_second_score = 81 := 
by
  sorry

end brittany_average_correct_l11_1184


namespace a100_gt_two_pow_99_l11_1140

theorem a100_gt_two_pow_99 
  (a : ℕ → ℤ) 
  (h1 : a 1 > a 0)
  (h2 : a 1 > 0)
  (h3 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2 ^ 99 :=
sorry

end a100_gt_two_pow_99_l11_1140


namespace diamond_comm_not_assoc_l11_1160

def diamond (a b : ℤ) : ℤ := (a * b + 5) / (a + b)

-- Lemma: Verify commutativity of the diamond operation
lemma diamond_comm (a b : ℤ) (ha : a > 1) (hb : b > 1) : 
  diamond a b = diamond b a := by
  sorry

-- Lemma: Verify non-associativity of the diamond operation
lemma diamond_not_assoc (a b c : ℤ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  diamond (diamond a b) c ≠ diamond a (diamond b c) := by
  sorry

-- Theorem: The diamond operation is commutative but not associative
theorem diamond_comm_not_assoc (a b c : ℤ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  diamond a b = diamond b a ∧ diamond (diamond a b) c ≠ diamond a (diamond b c) := by
  apply And.intro
  · apply diamond_comm
    apply ha
    apply hb
  · apply diamond_not_assoc
    apply ha
    apply hb
    apply hc

end diamond_comm_not_assoc_l11_1160


namespace avg_of_two_numbers_l11_1135

theorem avg_of_two_numbers (a b c d : ℕ) (h_different: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_positive: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_average: (a + b + c + d) / 4 = 4)
  (h_max_diff: ∀ x y : ℕ, (x ≠ y ∧ x > 0 ∧ y > 0 ∧ x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ x ≠ d ∧ y ≠ a ∧ y ≠ b ∧ y ≠ c ∧ y ≠ d) → (max x y - min x y <= max a d - min a d)) : 
  (a + b + c + d - min a (min b (min c d)) - max a (max b (max c d))) / 2 = 5 / 2 :=
by sorry

end avg_of_two_numbers_l11_1135


namespace positive_divisors_60_l11_1177

theorem positive_divisors_60 : ∃ n : ℕ, n = 12 ∧ (∀ d : ℕ, d ∣ 60 → d > 0 → ∃ (divisors_set : Finset ℕ), divisors_set.card = n ∧ ∀ x, x ∈ divisors_set ↔ x ∣ 60 ) :=
by
  sorry

end positive_divisors_60_l11_1177


namespace add_candies_to_equalize_l11_1197

-- Define the initial number of candies in basket A and basket B
def candiesInA : ℕ := 8
def candiesInB : ℕ := 17

-- Problem statement: Prove that adding 9 more candies to basket A
-- makes the number of candies in basket A equal to that in basket B.
theorem add_candies_to_equalize : ∃ n : ℕ, candiesInA + n = candiesInB :=
by
  use 9  -- The value we are adding to the candies in basket A
  sorry  -- Proof goes here

end add_candies_to_equalize_l11_1197


namespace cylinder_volume_increase_l11_1143

theorem cylinder_volume_increase (r h : ℝ) (V : ℝ) (hV : V = Real.pi * r^2 * h) :
    let new_height := 3 * h
    let new_radius := 2.5 * r
    let new_volume := Real.pi * (new_radius ^ 2) * new_height
    new_volume = 18.75 * V :=
by
  sorry

end cylinder_volume_increase_l11_1143


namespace total_distance_covered_l11_1153

noncomputable def radius : ℝ := 0.242
noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def number_of_revolutions : ℕ := 500
noncomputable def total_distance : ℝ := circumference * number_of_revolutions

theorem total_distance_covered :
  total_distance = 760 :=
by
  -- sorry Re-enable this line for the solver to automatically skip the proof 
  sorry

end total_distance_covered_l11_1153


namespace lemon_cookies_amount_l11_1130

def cookies_problem 
  (jenny_pb_cookies : ℕ) (jenny_cc_cookies : ℕ) (marcus_pb_cookies : ℕ) (marcus_lemon_cookies : ℕ)
  (total_pb_cookies : ℕ) (total_non_pb_cookies : ℕ) : Prop :=
  jenny_pb_cookies = 40 ∧
  jenny_cc_cookies = 50 ∧
  marcus_pb_cookies = 30 ∧
  total_pb_cookies = jenny_pb_cookies + marcus_pb_cookies ∧
  total_pb_cookies = 70 ∧
  total_non_pb_cookies = jenny_cc_cookies + marcus_lemon_cookies ∧
  total_pb_cookies = total_non_pb_cookies

theorem lemon_cookies_amount
  (jenny_pb_cookies : ℕ) (jenny_cc_cookies : ℕ) (marcus_pb_cookies : ℕ) (marcus_lemon_cookies : ℕ)
  (total_pb_cookies : ℕ) (total_non_pb_cookies : ℕ) :
  cookies_problem jenny_pb_cookies jenny_cc_cookies marcus_pb_cookies marcus_lemon_cookies total_pb_cookies total_non_pb_cookies →
  marcus_lemon_cookies = 20 :=
by
  sorry

end lemon_cookies_amount_l11_1130


namespace compute_expression_l11_1193

theorem compute_expression : 1004^2 - 996^2 - 1002^2 + 998^2 = 8000 := by
  sorry

end compute_expression_l11_1193


namespace highest_power_of_3_divides_N_l11_1112

-- Define the range of two-digit numbers and the concatenation function
def concatTwoDigitIntegers : ℕ := sorry  -- Placeholder for the concatenation implementation

-- Integer N formed by concatenating integers from 31 to 68
def N := concatTwoDigitIntegers

-- The statement proving the highest power of 3 dividing N is 3^1
theorem highest_power_of_3_divides_N :
  (∃ k : ℕ, 3^k ∣ N ∧ ¬ 3^(k+1) ∣ N) ∧ 3^1 ∣ N ∧ ¬ 3^2 ∣ N :=
by
  sorry  -- Placeholder for the proof

end highest_power_of_3_divides_N_l11_1112


namespace cost_to_cover_wall_with_tiles_l11_1198

/--
There is a wall in the shape of a rectangle with a width of 36 centimeters (cm) and a height of 72 centimeters (cm).
On this wall, you want to attach tiles that are 3 centimeters (cm) and 4 centimeters (cm) in length and width, respectively,
without any empty space. If it costs 2500 won per tile, prove that the total cost to cover the wall is 540,000 won.

Conditions:
- width_wall = 36
- height_wall = 72
- width_tile = 3
- height_tile = 4
- cost_per_tile = 2500

Target:
- Total_cost = 540,000 won
-/
theorem cost_to_cover_wall_with_tiles :
  let width_wall := 36
  let height_wall := 72
  let width_tile := 3
  let height_tile := 4
  let cost_per_tile := 2500
  let area_wall := width_wall * height_wall
  let area_tile := width_tile * height_tile
  let number_of_tiles := area_wall / area_tile
  let total_cost := number_of_tiles * cost_per_tile
  total_cost = 540000 := by
  sorry

end cost_to_cover_wall_with_tiles_l11_1198


namespace min_value_expression_l11_1126

theorem min_value_expression (a b c d : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2 + (5 / d - 1)^2 
  = 5^(5/4) - 10 * Real.sqrt (5^(1/4)) + 5 := 
sorry

end min_value_expression_l11_1126


namespace problem_statement_l11_1142

theorem problem_statement {a b c d : ℝ} (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (4 / (b - c)) + (9 / (c - d)) ≥ (36 / (a - d)) :=
by
  sorry -- proof is omitted according to the instructions

end problem_statement_l11_1142


namespace number_of_two_digit_factors_2_pow_18_minus_1_is_zero_l11_1128

theorem number_of_two_digit_factors_2_pow_18_minus_1_is_zero :
  (∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ n ∣ (2^18 - 1)) = false :=
by sorry

end number_of_two_digit_factors_2_pow_18_minus_1_is_zero_l11_1128


namespace worst_player_is_son_l11_1163

-- Define the types of players and relationships
inductive Sex
| male
| female

structure Player where
  name : String
  sex : Sex
  age : Nat

-- Define the four players
def woman := Player.mk "woman" Sex.female 30  -- Age is arbitrary
def brother := Player.mk "brother" Sex.male 30
def son := Player.mk "son" Sex.male 10
def daughter := Player.mk "daughter" Sex.female 10

-- Define the conditions
def opposite_sex (p1 p2 : Player) : Prop := p1.sex ≠ p2.sex
def same_age (p1 p2 : Player) : Prop := p1.age = p2.age

-- Define the worst player and the best player
variable (worst_player : Player) (best_player : Player)

-- Conditions as hypotheses
axiom twin_condition : ∃ twin : Player, (twin ≠ worst_player) ∧ (opposite_sex twin best_player)
axiom age_condition : same_age worst_player best_player
axiom not_same_player : worst_player ≠ best_player

-- Prove that the worst player is the son
theorem worst_player_is_son : worst_player = son :=
by
  sorry

end worst_player_is_son_l11_1163


namespace distance_point_to_line_zero_or_four_l11_1191

theorem distance_point_to_line_zero_or_four {b : ℝ} 
(h : abs (b - 2) / Real.sqrt 2 = Real.sqrt 2) : 
b = 0 ∨ b = 4 := 
sorry

end distance_point_to_line_zero_or_four_l11_1191


namespace weight_of_person_replaced_l11_1159

def initial_total_weight (W : ℝ) : ℝ := W
def new_person_weight : ℝ := 137
def average_increase : ℝ := 7.2
def group_size : ℕ := 10

theorem weight_of_person_replaced 
(W : ℝ) 
(weight_replaced : ℝ) 
(h1 : (W / group_size) + average_increase = (W - weight_replaced + new_person_weight) / group_size) : 
weight_replaced = 65 := 
sorry

end weight_of_person_replaced_l11_1159


namespace find_train_speed_l11_1138

def train_speed (v t_pole t_stationary d_stationary : ℕ) : ℕ := v

theorem find_train_speed (v : ℕ) (t_pole : ℕ) (t_stationary : ℕ) (d_stationary : ℕ) :
  t_pole = 5 →
  t_stationary = 25 →
  d_stationary = 360 →
  25 * v = 5 * v + d_stationary →
  v = 18 :=
by intros h1 h2 h3 h4; sorry

end find_train_speed_l11_1138


namespace sqrt_div_l11_1106

theorem sqrt_div (a b : ℝ) (h1 : a = 28) (h2 : b = 7) :
  Real.sqrt a / Real.sqrt b = 2 := 
by 
  sorry

end sqrt_div_l11_1106


namespace election_vote_percentage_l11_1117

theorem election_vote_percentage 
  (total_students : ℕ)
  (winner_percentage : ℝ)
  (loser_percentage : ℝ)
  (vote_difference : ℝ)
  (P : ℝ)
  (H1 : total_students = 2000)
  (H2 : winner_percentage = 0.55)
  (H3 : loser_percentage = 0.45)
  (H4 : vote_difference = 50)
  (H5 : 0.1 * P * (total_students / 100) = vote_difference) :
  P = 25 := 
sorry

end election_vote_percentage_l11_1117


namespace original_bet_is_40_l11_1166

-- Definition relating payout ratio and payout to original bet
def calculate_original_bet (payout_ratio payout : ℚ) : ℚ :=
  payout / payout_ratio

-- Given conditions
def payout_ratio : ℚ := 3 / 2
def received_payout : ℚ := 60

-- The proof goal
theorem original_bet_is_40 : calculate_original_bet payout_ratio received_payout = 40 :=
by
  sorry

end original_bet_is_40_l11_1166


namespace log_relationship_l11_1199

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem log_relationship :
  a > b ∧ b > c := by
  sorry

end log_relationship_l11_1199


namespace arithmetic_sum_problem_l11_1168

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sum_problem (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : arithmetic_sequence a)
  (h_S_def : ∀ n : ℕ, S n = sum_of_first_n_terms a n)
  (h_S13 : S 13 = 52) : a 4 + a 8 + a 9 = 12 :=
sorry

end arithmetic_sum_problem_l11_1168


namespace nut_game_winning_strategy_l11_1119

theorem nut_game_winning_strategy (N : ℕ) (h : N > 2) : ∃ second_player_wins : Prop, second_player_wins :=
sorry

end nut_game_winning_strategy_l11_1119


namespace promotional_rate_ratio_is_one_third_l11_1127

-- Define the conditions
def normal_monthly_charge : ℕ := 30
def extra_fee : ℕ := 15
def total_paid : ℕ := 175

-- Define the total data plan amount equation
def calculate_total (P : ℕ) : ℕ :=
  P + 2 * normal_monthly_charge + (normal_monthly_charge + extra_fee) + 2 * normal_monthly_charge

theorem promotional_rate_ratio_is_one_third (P : ℕ) (hP : calculate_total P = total_paid) :
  P * 3 = normal_monthly_charge :=
by sorry

end promotional_rate_ratio_is_one_third_l11_1127


namespace greg_initial_money_eq_36_l11_1147

theorem greg_initial_money_eq_36 
  (Earl_initial Fred_initial : ℕ)
  (Greg_initial : ℕ)
  (Earl_owes_Fred Fred_owes_Greg Greg_owes_Earl : ℕ)
  (Total_after_debt : ℕ)
  (hEarl_initial : Earl_initial = 90)
  (hFred_initial : Fred_initial = 48)
  (hEarl_owes_Fred : Earl_owes_Fred = 28)
  (hFred_owes_Greg : Fred_owes_Greg = 32)
  (hGreg_owes_Earl : Greg_owes_Earl = 40)
  (hTotal_after_debt : Total_after_debt = 130) :
  Greg_initial = 36 :=
sorry

end greg_initial_money_eq_36_l11_1147


namespace jericho_owes_annika_l11_1175

variable (J A M : ℝ)
variable (h1 : 2 * J = 60)
variable (h2 : M = A / 2)
variable (h3 : 30 - A - M = 9)

theorem jericho_owes_annika :
  A = 14 :=
by
  sorry

end jericho_owes_annika_l11_1175


namespace range_of_a_l11_1110

theorem range_of_a:
  (∃ x : ℝ, 1 ≤ x ∧ |x - a| + x - 4 ≤ 0) → (-2 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l11_1110


namespace correct_phone_call_sequence_l11_1169

-- Define the six steps as an enumerated type.
inductive Step
| Dial
| WaitDialTone
| PickUpHandset
| StartConversationOrHangUp
| WaitSignal
| EndCall

open Step

-- Define the problem as a theorem.
theorem correct_phone_call_sequence : 
  ∃ sequence : List Step, sequence = [PickUpHandset, WaitDialTone, Dial, WaitSignal, StartConversationOrHangUp, EndCall] :=
sorry

end correct_phone_call_sequence_l11_1169


namespace estimate_number_of_blue_cards_l11_1148

-- Define the given conditions:
def red_cards : ℕ := 8
def frequency_blue_card : ℚ := 0.6

-- Define the statement that needs to be proved:
theorem estimate_number_of_blue_cards (x : ℕ) 
  (h : (x : ℚ) / (x + red_cards) = frequency_blue_card) : 
  x = 12 :=
  sorry

end estimate_number_of_blue_cards_l11_1148


namespace school_fee_correct_l11_1172

-- Definitions
def mother_fifty_bills : ℕ := 1
def mother_twenty_bills : ℕ := 2
def mother_ten_bills : ℕ := 3

def father_fifty_bills : ℕ := 4
def father_twenty_bills : ℕ := 1
def father_ten_bills : ℕ := 1

def total_fifty_bills : ℕ := mother_fifty_bills + father_fifty_bills
def total_twenty_bills : ℕ := mother_twenty_bills + father_twenty_bills
def total_ten_bills : ℕ := mother_ten_bills + father_ten_bills

def value_fifty_bills : ℕ := 50 * total_fifty_bills
def value_twenty_bills : ℕ := 20 * total_twenty_bills
def value_ten_bills : ℕ := 10 * total_ten_bills

-- Theorem
theorem school_fee_correct :
  value_fifty_bills + value_twenty_bills + value_ten_bills = 350 :=
by
  sorry

end school_fee_correct_l11_1172


namespace volleyball_count_l11_1109

theorem volleyball_count (x y z : ℕ) (h1 : x + y + z = 20) (h2 : 6 * x + 3 * y + z = 33) : z = 15 :=
by
  sorry

end volleyball_count_l11_1109


namespace least_number_of_colors_needed_l11_1105

-- Define the tessellation of hexagons
structure HexagonalTessellation :=
(adjacent : (ℕ × ℕ) → (ℕ × ℕ) → Prop)
(symm : ∀ {a b : ℕ × ℕ}, adjacent a b → adjacent b a)
(irrefl : ∀ a : ℕ × ℕ, ¬ adjacent a a)
(hex_property : ∀ a : ℕ × ℕ, ∃ b1 b2 b3 b4 b5 b6,
  adjacent a b1 ∧ adjacent a b2 ∧ adjacent a b3 ∧ adjacent a b4 ∧ adjacent a b5 ∧ adjacent a b6)

-- Define a coloring function for a HexagonalTessellation
def coloring (T : HexagonalTessellation) (colors : ℕ) :=
(∀ (a b : ℕ × ℕ), T.adjacent a b → a ≠ b → colors ≥ 1 → colors ≤ 3)

-- Statement to prove the minimum number of colors required
theorem least_number_of_colors_needed (T : HexagonalTessellation) :
  ∃ colors, coloring T colors ∧ colors = 3 :=
sorry

end least_number_of_colors_needed_l11_1105


namespace Gianna_daily_savings_l11_1195

theorem Gianna_daily_savings 
  (total_saved : ℕ) (days_in_year : ℕ) 
  (H1 : total_saved = 14235) 
  (H2 : days_in_year = 365) : 
  total_saved / days_in_year = 39 := 
by 
  sorry

end Gianna_daily_savings_l11_1195


namespace find_r_in_geometric_series_l11_1194

theorem find_r_in_geometric_series
  (a r : ℝ)
  (h1 : a / (1 - r) = 15)
  (h2 : a / (1 - r^2) = 6) :
  r = 2 / 3 :=
sorry

end find_r_in_geometric_series_l11_1194


namespace students_selected_juice_l11_1104

def fraction_of_students_choosing_juice (students_selected_juice_ratio students_selected_soda_ratio : ℚ) : ℚ :=
  students_selected_juice_ratio / students_selected_soda_ratio

def num_students_selecting (students_selected_soda : ℕ) (fraction_juice : ℚ) : ℚ :=
  fraction_juice * students_selected_soda

theorem students_selected_juice (students_selected_soda : ℕ) : students_selected_soda = 120 ∧
    (fraction_of_students_choosing_juice 0.15 0.75) = 1/5 →
    num_students_selecting students_selected_soda (fraction_of_students_choosing_juice 0.15 0.75) = 24 :=
by
  intros h
  sorry

end students_selected_juice_l11_1104


namespace total_socks_l11_1190

-- Definitions based on conditions
def red_pairs : ℕ := 20
def red_socks : ℕ := red_pairs * 2
def black_socks : ℕ := red_socks / 2
def white_socks : ℕ := 2 * (red_socks + black_socks)

-- The main theorem we want to prove
theorem total_socks :
  (red_socks + black_socks + white_socks) = 180 := by
  sorry

end total_socks_l11_1190


namespace sum_of_first_11_terms_l11_1129

theorem sum_of_first_11_terms (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 4 + a 8 = 16) : (11 / 2) * (a 1 + a 11) = 88 :=
by
  sorry

end sum_of_first_11_terms_l11_1129


namespace totalPlayers_l11_1150

def kabadiParticipants : ℕ := 50
def khoKhoParticipants : ℕ := 80
def soccerParticipants : ℕ := 30
def kabadiAndKhoKhoParticipants : ℕ := 15
def kabadiAndSoccerParticipants : ℕ := 10
def khoKhoAndSoccerParticipants : ℕ := 25
def allThreeParticipants : ℕ := 8

theorem totalPlayers : kabadiParticipants + khoKhoParticipants + soccerParticipants 
                       - kabadiAndKhoKhoParticipants - kabadiAndSoccerParticipants 
                       - khoKhoAndSoccerParticipants + allThreeParticipants = 118 :=
by 
  sorry

end totalPlayers_l11_1150


namespace eval_nabla_l11_1178

def nabla (a b : ℕ) : ℕ := 3 + b^(a-1)

theorem eval_nabla : nabla (nabla 2 3) 4 = 1027 := by
  -- proof goes here
  sorry

end eval_nabla_l11_1178


namespace tangent_circle_line_l11_1136

theorem tangent_circle_line (r : ℝ) (h_pos : 0 < r) 
  (h_circle : ∀ x y : ℝ, x^2 + y^2 = r^2) 
  (h_line : ∀ x y : ℝ, x + y = r + 1) : 
  r = 1 + Real.sqrt 2 := 
by 
  sorry

end tangent_circle_line_l11_1136


namespace zero_point_in_range_l11_1122

theorem zero_point_in_range (a : ℝ) (x1 x2 x3 : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : x1 < x2) (h4 : x2 < x3)
  (hx1 : (x1^3 - 4*x1 + a) = 0) (hx2 : (x2^3 - 4*x2 + a) = 0) (hx3 : (x3^3 - 4*x3 + a) = 0) :
  0 < x2 ∧ x2 < 1 :=
by
  sorry

end zero_point_in_range_l11_1122


namespace product_not_power_of_two_l11_1118

theorem product_not_power_of_two (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℕ, (36 * a + b) * (a + 36 * b) ≠ 2^k :=
by
  sorry

end product_not_power_of_two_l11_1118


namespace min_value_N_l11_1154

theorem min_value_N (a b c d e f : ℤ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) (h₄ : 0 < e) (h₅ : 0 < f)
  (h_sum : a + b + c + d + e + f = 4020) :
  ∃ N : ℤ, N = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧ N = 805 :=
by
  sorry

end min_value_N_l11_1154


namespace total_turnips_l11_1133

theorem total_turnips (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := 
by sorry

end total_turnips_l11_1133


namespace number_of_extreme_value_points_l11_1115

noncomputable def f (x : ℝ) : ℝ := x^2 + x - Real.log x

theorem number_of_extreme_value_points : ∃! c : ℝ, c > 0 ∧ (deriv f c = 0) :=
by
  sorry

end number_of_extreme_value_points_l11_1115


namespace friend_redistribution_l11_1187

-- Definitions of friends' earnings
def earnings := [18, 22, 26, 32, 47]

-- Definition of total earnings
def totalEarnings := earnings.sum

-- Definition of equal share
def equalShare := totalEarnings / earnings.length

-- The amount that the friend who earned 47 needs to redistribute
def redistributionAmount := 47 - equalShare

-- The goal to prove
theorem friend_redistribution:
  redistributionAmount = 18 := by
  sorry

end friend_redistribution_l11_1187


namespace problem_solution_l11_1173

-- Definitions
def has_property_P (A : List ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ A.length →
    (A.get! (j - 1) + A.get! (i - 1) ∈ A ∨ A.get! (j - 1) - A.get! (i - 1) ∈ A)

def sequence_01234 := [0, 2, 4, 6]

-- Propositions
def proposition_1 : Prop := has_property_P sequence_01234

def proposition_2 (A : List ℕ) : Prop := 
  has_property_P A → (A.headI = 0)

def proposition_3 (A : List ℕ) : Prop :=
  has_property_P A → A.headI ≠ 0 →
  ∀ k, 1 ≤ k ∧ k < A.length → A.get! (A.length - 1) - A.get! (A.length - 1 - k) = A.get! k

def proposition_4 (A : List ℕ) : Prop :=
  has_property_P A → A.length = 3 →
  A.get! 2 = A.get! 0 + A.get! 1

-- Main statement
theorem problem_solution : 
  (proposition_1) ∧
  (∃ A, ¬ (proposition_2 A)) ∧
  (∃ A, proposition_3 A) ∧
  (∃ A, proposition_4 A) →
  3 = 3 := 
by sorry

end problem_solution_l11_1173


namespace probability_median_five_l11_1170

theorem probability_median_five {S : Finset ℕ} (hS : S = {1, 2, 3, 4, 5, 6, 7, 8}) :
  let n := 8
  let k := 5
  let total_ways := Nat.choose n k
  let ways_median_5 := Nat.choose 4 2 * Nat.choose 3 2
  (ways_median_5 : ℚ) / (total_ways : ℚ) = (9 : ℚ) / (28 : ℚ) :=
by
  sorry

end probability_median_five_l11_1170


namespace divisibility_of_2b_by_a_l11_1113

theorem divisibility_of_2b_by_a (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b)
  (h_cond : ∃ᶠ m in at_top, ∃ᶠ n in at_top, (∃ k₁ : ℕ, m^2 + a * n + b = k₁^2) ∧ (∃ k₂ : ℕ, n^2 + a * m + b = k₂^2)) :
  a ∣ 2 * b :=
sorry

end divisibility_of_2b_by_a_l11_1113


namespace suzie_store_revenue_l11_1156

theorem suzie_store_revenue 
  (S B : ℝ) 
  (h1 : B = S + 15) 
  (h2 : 22 * S + 16 * B = 460) : 
  8 * S + 32 * B = 711.60 :=
by
  sorry

end suzie_store_revenue_l11_1156


namespace work_together_days_l11_1189

-- Define the days it takes for A and B to complete the work individually.
def days_A : ℕ := 3
def days_B : ℕ := 6

-- Define the combined work rate.
def combined_work_rate : ℚ := (1 / days_A) + (1 / days_B)

-- State the theorem for the number of days A and B together can complete the work.
theorem work_together_days :
  1 / combined_work_rate = 2 := by
  sorry

end work_together_days_l11_1189
